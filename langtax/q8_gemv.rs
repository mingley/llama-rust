// Twin of q8_gemv.c: same Q8_0 NEON nrc=1 kernel, Rust host.
#![allow(clippy::needless_range_loop)]

use std::arch::aarch64::*;
use std::time::Instant;

const QK8_0: usize = 32;

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockQ80 {
    d: f32,
    qs: [i8; QK8_0],
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn gemv_q8_0_neon(
    k: usize,
    m: usize,
    nb: usize,
    w: *const BlockQ80,
    x: *const BlockQ80,
    y: *mut f32,
) {
    for r in 0..m {
        *y.add(r) = vec_dot_q8_0_neon(k, w.add(r * nb), x);
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

fn vec_dot_q8_0_scalar(n: usize, x: &[BlockQ80], y: &[BlockQ80]) -> f32 {
    let nb = n / QK8_0;
    let mut sumf = 0.0f32;
    for ib in 0..nb {
        let mut sumi = 0i32;
        for j in 0..QK8_0 {
            sumi += x[ib].qs[j] as i32 * y[ib].qs[j] as i32;
        }
        sumf += sumi as f32 * (x[ib].d * y[ib].d);
    }
    sumf
}

fn main() {
    let k = 4096usize;
    let m = 4096usize;
    let nb = k / QK8_0;
    let niter = 8usize;

    let mut w = vec![
        BlockQ80 {
            d: 0.0,
            qs: [0; QK8_0]
        };
        m * nb
    ];
    let mut x = vec![
        BlockQ80 {
            d: 0.0,
            qs: [0; QK8_0]
        };
        nb
    ];
    let mut y = vec![0.0f32; m];

    let mut seed = 1u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 33) as u32
    };
    for b in w.iter_mut() {
        let amax = 0.01 + (rnd() % 1000) as f32 / 1000.0;
        b.d = amax / 127.0;
        for j in 0..QK8_0 {
            b.qs[j] = ((rnd() % 255) as i32 - 128) as i8;
        }
    }
    for b in x.iter_mut() {
        b.d = 1.0 / 127.0;
        for j in 0..QK8_0 {
            b.qs[j] = ((rnd() % 255) as i32 - 128) as i8;
        }
    }

    let rref = vec_dot_q8_0_scalar(k, &w[..nb], &x);
    let got = unsafe { vec_dot_q8_0_neon(k, w.as_ptr(), x.as_ptr()) };
    if (rref - got).abs() > 1e-2 * (1.0 + rref.abs()) {
        eprintln!("mismatch scalar={rref:.6} neon={got:.6}");
        std::process::exit(2);
    }

    unsafe {
        for r in 0..m {
            y[r] = vec_dot_q8_0_neon(k, w.as_ptr().add(r * nb), x.as_ptr());
        }
    }

    let t0 = Instant::now();
    let mut sink = 0.0f32;
    unsafe {
        for _ in 0..niter {
            gemv_q8_0_neon(k, m, nb, w.as_ptr(), x.as_ptr(), y.as_mut_ptr());
            sink += y[0];
        }
    }
    let sec = t0.elapsed().as_secs_f64();
    let wbytes = (niter * m * nb * std::mem::size_of::<BlockQ80>()) as f64;
    let bytes = wbytes + (niter * m * nb * std::mem::size_of::<BlockQ80>()) as f64;
    println!("lang=Rust kernel=q8_0_neon M={m} K={k} niter={niter}");
    println!("time_s={sec:.6} gemv/s={:.2}", niter as f64 / sec);
    println!(
        "weight_GiB/s={:.2}  (W+x GiB/s={:.2})",
        wbytes / sec / ((1u64 << 30) as f64),
        bytes / sec / ((1u64 << 30) as f64)
    );
    println!(
        "us_per_row={:.3} sink={sink:.4} check={:.4}",
        (sec * 1e6) / (niter * m) as f64,
        y[0]
    );
}
