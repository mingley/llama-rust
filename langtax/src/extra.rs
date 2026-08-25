//! Extra measured cells: Q8_0 GEMV size sweep + Q4_0 GEMV. Same pack functions as the gate CLI.

use std::time::Instant;

use q8_gemv::{gemv_q4_0, gemv_q8_0, BlockQ40, BlockQ80, QK4_0, QK8_0};

fn rnd(seed: &mut u64) -> u32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*seed >> 33) as u32
}

fn fill_q8(w: &mut [BlockQ80], x: &mut [BlockQ80]) {
    let mut seed = 1u64;
    for b in w.iter_mut() {
        let amax = 0.01 + (rnd(&mut seed) % 1000) as f32 / 1000.0;
        b.d = amax / 127.0;
        for j in 0..QK8_0 {
            b.qs[j] = ((rnd(&mut seed) % 255) as i32 - 128) as i8;
        }
    }
    for b in x.iter_mut() {
        b.d = 1.0 / 127.0;
        for j in 0..QK8_0 {
            b.qs[j] = ((rnd(&mut seed) % 255) as i32 - 128) as i8;
        }
    }
}

fn fill_q4(w: &mut [BlockQ40], x: &mut [BlockQ80]) {
    let mut seed = 3u64;
    for b in w.iter_mut() {
        b.d = 0.02 + (rnd(&mut seed) % 800) as f32 / 10000.0;
        for j in 0..(QK4_0 / 2) {
            b.qs[j] = (rnd(&mut seed) % 256) as u8;
        }
    }
    for b in x.iter_mut() {
        b.d = 1.0 / 127.0;
        for j in 0..QK8_0 {
            b.qs[j] = ((rnd(&mut seed) % 255) as i32 - 128) as i8;
        }
    }
}

fn bench_q8(m: usize, k: usize, niter: usize, warmup: usize) {
    let nb = k / QK8_0;
    let mut w = vec![BlockQ80::zero(); m * nb];
    let mut x = vec![BlockQ80::zero(); nb];
    let mut y = vec![0.0f32; m];
    fill_q8(&mut w, &mut x);
    for _ in 0..warmup {
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
    println!("lang=Rust kernel=q8_0_safe M={m} K={k} niter={niter}");
    println!("time_s={sec:.6} gemv/s={:.2}", niter as f64 / sec);
    println!(
        "weight_GiB/s={:.2} sink={sink:.4} check={:.4}",
        wbytes / sec / ((1u64 << 30) as f64),
        y[0]
    );
}

fn bench_q4(m: usize, k: usize, niter: usize, warmup: usize) {
    let nb = k / QK4_0;
    let mut w = vec![BlockQ40::zero(); m * nb];
    let mut x = vec![BlockQ80::zero(); nb];
    let mut y = vec![0.0f32; m];
    fill_q4(&mut w, &mut x);
    for _ in 0..warmup {
        gemv_q4_0(k, &w, &x, &mut y);
    }
    let t0 = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..niter {
        gemv_q4_0(k, &w, &x, &mut y);
        sink += y[0];
    }
    let sec = t0.elapsed().as_secs_f64();
    let wbytes = (niter * m * nb * std::mem::size_of::<BlockQ40>()) as f64;
    println!("lang=Rust kernel=q4_0_safe M={m} K={k} niter={niter}");
    println!("time_s={sec:.6} gemv/s={:.2}", niter as f64 / sec);
    println!(
        "weight_GiB/s={:.2} sink={sink:.4} check={:.4}",
        wbytes / sec / ((1u64 << 30) as f64),
        y[0]
    );
}

fn main() {
    let niter = 8usize;
    let warmup = 64usize;
    for &n in &[1024usize, 2048, 8192] {
        bench_q8(n, n, niter, warmup);
    }
    bench_q4(4096, 4096, niter, warmup);
}
