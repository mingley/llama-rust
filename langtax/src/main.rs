use std::time::Instant;

use q8_gemv::{gemv_q8_0, BlockQ80, QK8_0};

fn main() {
    let k = 4096usize;
    let m = 4096usize;
    let nb = k / QK8_0;
    let niter = 8usize;

    let mut w = vec![BlockQ80::zero(); m * nb];
    let mut x = vec![BlockQ80::zero(); nb];
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

    for _ in 0..64 {
        gemv_q8_0(k, &w, &x, &mut y);
    }

    let t0 = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..niter {
        gemv_q8_0(k, &w, &x, &mut y);
        sink += y[0];
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
