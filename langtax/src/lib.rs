//! Packed Q8_0 GEMV. The timed CLI and the correctness test both call [`gemv_q8_0`].

#![allow(clippy::needless_range_loop)]

use std::arch::aarch64::*;
use std::sync::LazyLock;

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

/// Eight workers: P-core count on this M4 Pro without dragging in E-cores.
const GEMV_THREADS: usize = 8;

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
                gemv_serial(k, w_part, x, y_part);
            });
    });
}

fn gemv_serial(k: usize, w: &[BlockQ80], x: &[BlockQ80], y: &mut [f32]) {
    let nb = k / QK8_0;
    let m = y.len();
    let mut r = 0usize;
    while r + 1 < m {
        unsafe {
            let (y0, y1) = vec_dot_q8_0_pair(
                k,
                w.as_ptr().add(r * nb),
                w.as_ptr().add((r + 1) * nb),
                x.as_ptr(),
            );
            y[r] = y0;
            y[r + 1] = y1;
        }
        r += 2;
    }
    if r < m {
        unsafe {
            y[r] = vec_dot_q8_0_neon(k, w.as_ptr().add(r * nb), x.as_ptr());
        }
    }
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn vec_dot_q8_0_neon(n: usize, x: *const BlockQ80, y: *const BlockQ80) -> f32 {
    let nb = n / QK8_0;
    let mut ib = 0usize;
    let mut sumv0 = vdupq_n_f32(0.0);
    let mut sumv1 = vdupq_n_f32(0.0);

    while ib + 1 < nb {
        let x0 = &*x.add(ib);
        let x1 = &*x.add(ib + 1);
        let y0 = &*y.add(ib);
        let y1 = &*y.add(ib + 1);

        let x0_0 = vld1q_s8(x0.qs.as_ptr());
        let x0_1 = vld1q_s8(x0.qs.as_ptr().add(16));
        let x1_0 = vld1q_s8(x1.qs.as_ptr());
        let x1_1 = vld1q_s8(x1.qs.as_ptr().add(16));
        let y0_0 = vld1q_s8(y0.qs.as_ptr());
        let y0_1 = vld1q_s8(y0.qs.as_ptr().add(16));
        let y1_0 = vld1q_s8(y1.qs.as_ptr());
        let y1_1 = vld1q_s8(y1.qs.as_ptr().add(16));

        let p0 = vaddq_s32(
            vdotq_s32(vdupq_n_s32(0), x0_0, y0_0),
            vdotq_s32(vdupq_n_s32(0), x0_1, y0_1),
        );
        let p1 = vaddq_s32(
            vdotq_s32(vdupq_n_s32(0), x1_0, y1_0),
            vdotq_s32(vdupq_n_s32(0), x1_1, y1_1),
        );
        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0), x0.d * y0.d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1), x1.d * y1.d);
        ib += 2;
    }
    let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
    while ib < nb {
        let xb = &*x.add(ib);
        let yb = &*y.add(ib);
        let mut sumi = 0i32;
        for j in 0..QK8_0 {
            sumi += xb.qs[j] as i32 * yb.qs[j] as i32;
        }
        sumf += sumi as f32 * (xb.d * yb.d);
        ib += 1;
    }
    sumf
}

/// Two W rows against one x. Shares the x loads.
#[target_feature(enable = "neon,dotprod")]
unsafe fn vec_dot_q8_0_pair(
    n: usize,
    w0: *const BlockQ80,
    w1: *const BlockQ80,
    x: *const BlockQ80,
) -> (f32, f32) {
    let nb = n / QK8_0;
    let mut ib = 0usize;
    let mut sum0a = vdupq_n_f32(0.0);
    let mut sum0b = vdupq_n_f32(0.0);
    let mut sum1a = vdupq_n_f32(0.0);
    let mut sum1b = vdupq_n_f32(0.0);

    while ib + 1 < nb {
        let a0 = &*w0.add(ib);
        let a1 = &*w0.add(ib + 1);
        let b0 = &*w1.add(ib);
        let b1 = &*w1.add(ib + 1);
        let x0 = &*x.add(ib);
        let x1 = &*x.add(ib + 1);

        let x0_0 = vld1q_s8(x0.qs.as_ptr());
        let x0_1 = vld1q_s8(x0.qs.as_ptr().add(16));
        let x1_0 = vld1q_s8(x1.qs.as_ptr());
        let x1_1 = vld1q_s8(x1.qs.as_ptr().add(16));

        let a00 = vld1q_s8(a0.qs.as_ptr());
        let a01 = vld1q_s8(a0.qs.as_ptr().add(16));
        let a10 = vld1q_s8(a1.qs.as_ptr());
        let a11 = vld1q_s8(a1.qs.as_ptr().add(16));
        let b00 = vld1q_s8(b0.qs.as_ptr());
        let b01 = vld1q_s8(b0.qs.as_ptr().add(16));
        let b10 = vld1q_s8(b1.qs.as_ptr());
        let b11 = vld1q_s8(b1.qs.as_ptr().add(16));

        let p_a0 = vaddq_s32(
            vdotq_s32(vdupq_n_s32(0), a00, x0_0),
            vdotq_s32(vdupq_n_s32(0), a01, x0_1),
        );
        let p_a1 = vaddq_s32(
            vdotq_s32(vdupq_n_s32(0), a10, x1_0),
            vdotq_s32(vdupq_n_s32(0), a11, x1_1),
        );
        let p_b0 = vaddq_s32(
            vdotq_s32(vdupq_n_s32(0), b00, x0_0),
            vdotq_s32(vdupq_n_s32(0), b01, x0_1),
        );
        let p_b1 = vaddq_s32(
            vdotq_s32(vdupq_n_s32(0), b10, x1_0),
            vdotq_s32(vdupq_n_s32(0), b11, x1_1),
        );

        sum0a = vmlaq_n_f32(sum0a, vcvtq_f32_s32(p_a0), a0.d * x0.d);
        sum0b = vmlaq_n_f32(sum0b, vcvtq_f32_s32(p_a1), a1.d * x1.d);
        sum1a = vmlaq_n_f32(sum1a, vcvtq_f32_s32(p_b0), b0.d * x0.d);
        sum1b = vmlaq_n_f32(sum1b, vcvtq_f32_s32(p_b1), b1.d * x1.d);
        ib += 2;
    }
    let mut s0 = vaddvq_f32(sum0a) + vaddvq_f32(sum0b);
    let mut s1 = vaddvq_f32(sum1a) + vaddvq_f32(sum1b);
    while ib < nb {
        let a = &*w0.add(ib);
        let b = &*w1.add(ib);
        let xv = &*x.add(ib);
        let mut sa = 0i32;
        let mut sb = 0i32;
        for j in 0..QK8_0 {
            let xq = xv.qs[j] as i32;
            sa += a.qs[j] as i32 * xq;
            sb += b.qs[j] as i32 * xq;
        }
        s0 += sa as f32 * (a.d * xv.d);
        s1 += sb as f32 * (b.d * xv.d);
        ib += 1;
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
