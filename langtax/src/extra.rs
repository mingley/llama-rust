//! Extra cells: GGUF-native Q8 size sweep + Q4_0 GEMV via write+load+matmul.

use std::time::Instant;

use llama_rust::{
    gemv_q4_0, gemv_q8_0, load_gguf, pack_q4_0_block, pack_q8_0_block, write_gguf, GgmlType,
    TensorWrite, QK4_0, QK8_0,
};

fn rnd(seed: &mut u64) -> u32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*seed >> 33) as u32
}

fn q8_pair(n_cols: usize, n_rows: usize, seed0: u64) -> (Vec<u8>, Vec<u8>) {
    let mut seed = seed0;
    let mut w = Vec::new();
    let mut x = Vec::new();
    for _ in 0..n_rows {
        for _ in 0..(n_cols / QK8_0) {
            let mut qs = [0i8; QK8_0];
            for q in &mut qs {
                *q = ((rnd(&mut seed) % 255) as i32 - 128) as i8;
            }
            w.extend_from_slice(&pack_q8_0_block(
                0.02 + (rnd(&mut seed) % 80) as f32 / 1000.0,
                &qs,
            ));
        }
    }
    for _ in 0..(n_cols / QK8_0) {
        let mut qs = [0i8; QK8_0];
        for q in &mut qs {
            *q = ((rnd(&mut seed) % 255) as i32 - 128) as i8;
        }
        x.extend_from_slice(&pack_q8_0_block(1.0 / 127.0, &qs));
    }
    (w, x)
}

fn bench_q8(n: usize) {
    let (w, x) = q8_pair(n, n, 1);
    let bytes = write_gguf(&[
        TensorWrite {
            name: "w_q8".into(),
            ty: GgmlType::Q8_0,
            shape: vec![n as u64, n as u64],
            data: w,
        },
        TensorWrite {
            name: "x_q8".into(),
            ty: GgmlType::Q8_0,
            shape: vec![n as u64],
            data: x,
        },
    ]);
    let g = load_gguf(&bytes).expect("load");
    let wt = g.tensor("w_q8").unwrap();
    let xt = g.tensor("x_q8").unwrap();
    let mut y = vec![0.0f32; n];
    let niter = 8usize;
    for _ in 0..16 {
        gemv_q8_0(n, &wt.data, &xt.data, &mut y);
    }
    let t0 = Instant::now();
    for _ in 0..niter {
        gemv_q8_0(n, &wt.data, &xt.data, &mut y);
    }
    let sec = t0.elapsed().as_secs_f64();
    println!("lang=Rust kernel=q8_0_gguf M={n} K={n} niter={niter}");
    println!(
        "time_s={sec:.6} gemv/s={:.2}",
        niter as f64 / sec.max(1e-12)
    );
    println!("y0={:.6}", y[0]);
}

fn bench_q4(n: usize) {
    let mut seed = 3u64;
    let mut w = Vec::new();
    let mut x = Vec::new();
    for _ in 0..n {
        for _ in 0..(n / QK4_0) {
            let mut qs = [0u8; QK4_0 / 2];
            for q in &mut qs {
                *q = (rnd(&mut seed) % 256) as u8;
            }
            w.extend_from_slice(&pack_q4_0_block(
                0.03 + (rnd(&mut seed) % 50) as f32 / 1000.0,
                &qs,
            ));
        }
    }
    for _ in 0..(n / QK8_0) {
        let mut qs = [0i8; QK8_0];
        for q in &mut qs {
            *q = ((rnd(&mut seed) % 255) as i32 - 128) as i8;
        }
        x.extend_from_slice(&pack_q8_0_block(1.0 / 127.0, &qs));
    }
    let bytes = write_gguf(&[
        TensorWrite {
            name: "w_q4".into(),
            ty: GgmlType::Q4_0,
            shape: vec![n as u64, n as u64],
            data: w,
        },
        TensorWrite {
            name: "x_q8".into(),
            ty: GgmlType::Q8_0,
            shape: vec![n as u64],
            data: x,
        },
    ]);
    let g = load_gguf(&bytes).expect("load");
    let wt = g.tensor("w_q4").unwrap();
    let xt = g.tensor("x_q8").unwrap();
    let mut y = vec![0.0f32; n];
    let niter = 8usize;
    for _ in 0..16 {
        gemv_q4_0(n, &wt.data, &xt.data, &mut y);
    }
    let t0 = Instant::now();
    for _ in 0..niter {
        gemv_q4_0(n, &wt.data, &xt.data, &mut y);
    }
    let sec = t0.elapsed().as_secs_f64();
    println!("lang=Rust kernel=q4_0_gguf M={n} K={n} niter={niter}");
    println!(
        "time_s={sec:.6} gemv/s={:.2}",
        niter as f64 / sec.max(1e-12)
    );
    println!("y0={:.6}", y[0]);
}

fn main() {
    for &n in &[128usize, 256, 512, 1024] {
        bench_q8(n);
    }
    bench_q4(256);
}
