//! Packed Q8_0 GEMV. The timed CLI and the correctness test both call [`gemv_q8_0`].
//!
//! No `unsafe` in this crate. LLVM emits NEON SDOT from the constant-32 i8
//! loops; the ≥2× vs 1-thread C is the 10-worker pool.

#![forbid(unsafe_code)]

use std::sync::LazyLock;

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

/// Ten workers: M4 Pro P-cores. E-cores stay unused.
const GEMV_THREADS: usize = 10;

static POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    ThreadPoolBuilder::new()
        .num_threads(GEMV_THREADS)
        .build()
        .expect("rayon GEMV pool")
});

pub const QK8_0: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ80 {
    pub d: f32,
    pub qs: [i8; QK8_0],
}

impl BlockQ80 {
    pub fn zero() -> Self {
        Self {
            d: 0.0,
            qs: [0; QK8_0],
        }
    }
}

/// y[m] = W[m, k] x[k] with W and x stored as Q8_0 blocks of 32.
///
/// `w.len() == y.len() * (k / 32)`, `x.len() == k / 32`.
pub fn gemv_q8_0(k: usize, w: &[BlockQ80], x: &[BlockQ80], y: &mut [f32]) {
    assert!(k.is_multiple_of(QK8_0), "k must be a multiple of {QK8_0}");
    let nb = k / QK8_0;
    let m = y.len();
    assert_eq!(x.len(), nb, "x blocks");
    assert_eq!(w.len(), m * nb, "W blocks");
    if m == 0 {
        return;
    }

    let nthreads = GEMV_THREADS.min(m).max(1);
    let chunk_rows = m.div_ceil(nthreads).max(1);
    POOL.install(|| {
        y.par_chunks_mut(chunk_rows)
            .zip(w.par_chunks(chunk_rows * nb))
            .for_each(|(y_part, w_part)| {
                gemv_serial(w_part, x, y_part);
            });
    });
}

fn gemv_serial(w: &[BlockQ80], x: &[BlockQ80], y: &mut [f32]) {
    let nb = x.len();
    let mut w_off = 0usize;
    let mut rows = y;
    while rows.len() >= 2 {
        let (pair, rest) = rows.split_at_mut(2);
        let w0 = &w[w_off..w_off + nb];
        let w1 = &w[w_off + nb..w_off + 2 * nb];
        let (a, b) = vec_dot_pair(w0, w1, x);
        pair[0] = a;
        pair[1] = b;
        w_off += 2 * nb;
        rows = rest;
    }
    if let Some(last) = rows.first_mut() {
        *last = vec_dot_row(&w[w_off..], x);
    }
}

#[inline(always)]
fn dot32(a: &[i8; QK8_0], b: &[i8; QK8_0]) -> i32 {
    // Constant trip count. With `-C target-cpu=native` LLVM emits SDOT.
    let mut acc = 0i32;
    for i in 0..QK8_0 {
        acc += a[i] as i32 * b[i] as i32;
    }
    acc
}

fn vec_dot_row(row: &[BlockQ80], x: &[BlockQ80]) -> f32 {
    let n = x.len();
    let mut i = 0usize;
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    while i + 1 < n {
        let a0 = &row[i];
        let a1 = &row[i + 1];
        let x0 = &x[i];
        let x1 = &x[i + 1];
        acc0 += dot32(&a0.qs, &x0.qs) as f32 * (a0.d * x0.d);
        acc1 += dot32(&a1.qs, &x1.qs) as f32 * (a1.d * x1.d);
        i += 2;
    }
    let mut sum = acc0 + acc1;
    while i < n {
        let wb = &row[i];
        let xb = &x[i];
        sum += dot32(&wb.qs, &xb.qs) as f32 * (wb.d * xb.d);
        i += 1;
    }
    sum
}

fn vec_dot_pair(w0: &[BlockQ80], w1: &[BlockQ80], x: &[BlockQ80]) -> (f32, f32) {
    let n = x.len();
    let mut i = 0usize;
    let mut s0a = 0.0f32;
    let mut s0b = 0.0f32;
    let mut s1a = 0.0f32;
    let mut s1b = 0.0f32;
    while i + 1 < n {
        let a0 = &w0[i];
        let a1 = &w0[i + 1];
        let b0 = &w1[i];
        let b1 = &w1[i + 1];
        let x0 = &x[i];
        let x1 = &x[i + 1];
        s0a += dot32(&a0.qs, &x0.qs) as f32 * (a0.d * x0.d);
        s0b += dot32(&a1.qs, &x1.qs) as f32 * (a1.d * x1.d);
        s1a += dot32(&b0.qs, &x0.qs) as f32 * (b0.d * x0.d);
        s1b += dot32(&b1.qs, &x1.qs) as f32 * (b1.d * x1.d);
        i += 2;
    }
    let mut s0 = s0a + s0b;
    let mut s1 = s1a + s1b;
    while i < n {
        let a = &w0[i];
        let b = &w1[i];
        let xv = &x[i];
        s0 += dot32(&a.qs, &xv.qs) as f32 * (a.d * xv.d);
        s1 += dot32(&b.qs, &xv.qs) as f32 * (b.d * xv.d);
        i += 1;
    }
    (s0, s1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_dot_row(row: &[BlockQ80], x: &[BlockQ80]) -> f32 {
        assert_eq!(row.len(), x.len());
        let mut sum = 0.0f32;
        for (wb, xb) in row.iter().zip(x.iter()) {
            let mut sumi = 0i32;
            for j in 0..QK8_0 {
                sumi += wb.qs[j] as i32 * xb.qs[j] as i32;
            }
            sum += sumi as f32 * (wb.d * xb.d);
        }
        sum
    }

    #[test]
    fn gemv_q8_0_matches_independent_scalar() {
        let k = 96usize;
        let m = 17usize;
        let nb = k / QK8_0;
        let mut w = vec![BlockQ80::zero(); m * nb];
        let mut x = vec![BlockQ80::zero(); nb];
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 33) as u32
        };
        for b in w.iter_mut() {
            b.d = 0.01 + (rnd() % 50) as f32 / 1000.0;
            for j in 0..QK8_0 {
                b.qs[j] = ((rnd() % 21) as i32 - 10) as i8;
            }
        }
        for b in x.iter_mut() {
            b.d = 0.02;
            for j in 0..QK8_0 {
                b.qs[j] = ((rnd() % 21) as i32 - 10) as i8;
            }
        }

        let mut y = vec![0.0f32; m];
        gemv_q8_0(k, &w, &x, &mut y);

        for r in 0..m {
            let row = &w[r * nb..(r + 1) * nb];
            let expected = scalar_dot_row(row, &x);
            let denom = 1.0 + expected.abs();
            let rel = (y[r] - expected).abs() / denom;
            assert!(
                rel < 1e-5,
                "row {r}: gemv={} scalar={} rel={rel}",
                y[r],
                expected
            );
        }
    }
}
