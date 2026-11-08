//! Extra cells: GGUF-native Q8 size sweep + Q4_0 GEMV via write+load+matmul.

use std::time::Instant;

use llama_rust::gguf::{load_gguf, write_gguf, GgmlType, TensorWrite};
use llama_rust::kernels::{
    gemv_q4_0, gemv_q4_k, gemv_q8_0, pack_q4_0_block, pack_q4_k_block, pack_q8_0_block,
    pack_q8_k_block, QK4_0, QK8_0, QK_K,
};

fn rnd_u32(seed: &mut u64) -> u32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    u32::try_from(*seed >> 33).unwrap_or(0)
}

fn centered_i8(u: u32) -> i8 {
    let n = i32::try_from(u % 255).unwrap_or(0) - 128;
    i8::try_from(n).unwrap_or(0)
}

fn q8_pair(n_cols: usize, n_rows: usize, seed0: u64) -> (Vec<u8>, Vec<u8>) {
    let mut seed = seed0;
    let mut w = Vec::new();
    let mut x = Vec::new();
    for _ in 0..n_rows {
        for _ in 0..(n_cols / QK8_0) {
            let mut qs = [0i8; QK8_0];
            for q in &mut qs {
                *q = centered_i8(rnd_u32(&mut seed));
            }
            let extra = u16::try_from(rnd_u32(&mut seed) % 80).unwrap_or(0);
            w.extend_from_slice(&pack_q8_0_block(
                20.0 / 1000.0 + f32::from(extra) / 1000.0,
                &qs,
            ));
        }
    }
    for _ in 0..(n_cols / QK8_0) {
        let mut qs = [0i8; QK8_0];
        for q in &mut qs {
            *q = centered_i8(rnd_u32(&mut seed));
        }
        x.extend_from_slice(&pack_q8_0_block(1.0 / 127.0, &qs));
    }
    (w, x)
}

fn print_bench(kernel: &str, n: usize, niter: usize, sec: f64, y0: f32) {
    let niter_f = f64::from(u32::try_from(niter).unwrap_or(0));
    let gemv_s = if sec > 0.0 { niter_f / sec } else { 0.0 };
    println!("lang=Rust kernel={kernel} M={n} K={n} niter={niter}");
    println!("time_s={sec:.6} gemv/s={gemv_s:.2}");
    println!("y0={y0:.6}");
}

fn bench_q8(n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let (w, x) = q8_pair(n, n, 1);
    let n64 = u64::try_from(n).unwrap_or(0);
    let bytes = write_gguf(&[
        TensorWrite {
            name: "w_q8".into(),
            ty: GgmlType::Q8_0,
            shape: vec![n64, n64],
            data: w,
        },
        TensorWrite {
            name: "x_q8".into(),
            ty: GgmlType::Q8_0,
            shape: vec![n64],
            data: x,
        },
    ]);
    let g = load_gguf(&bytes)?;
    let wt = g
        .tensor("w_q8")
        .ok_or_else(|| "missing tensor w_q8".to_string())?;
    let xt = g
        .tensor("x_q8")
        .ok_or_else(|| "missing tensor x_q8".to_string())?;
    let mut y = vec![0.0f32; n];
    let niter = 8usize;
    for _ in 0..16 {
        gemv_q8_0(n, wt.data, xt.data, &mut y)?;
    }
    let t0 = Instant::now();
    for _ in 0..niter {
        gemv_q8_0(n, wt.data, xt.data, &mut y)?;
    }
    let sec = t0.elapsed().as_secs_f64();
    print_bench(
        "q8_0_gguf",
        n,
        niter,
        sec,
        y.first().copied().unwrap_or(0.0),
    );
    Ok(())
}

fn bench_q4(n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let mut seed = 3u64;
    let mut w = Vec::new();
    let mut x = Vec::new();
    for _ in 0..n {
        for _ in 0..(n / QK4_0) {
            let mut qs = [0u8; QK4_0 / 2];
            for q in &mut qs {
                *q = u8::try_from(rnd_u32(&mut seed) % 256).unwrap_or(0);
            }
            let extra = u16::try_from(rnd_u32(&mut seed) % 50).unwrap_or(0);
            w.extend_from_slice(&pack_q4_0_block(
                30.0 / 1000.0 + f32::from(extra) / 1000.0,
                &qs,
            ));
        }
    }
    for _ in 0..(n / QK8_0) {
        let mut qs = [0i8; QK8_0];
        for q in &mut qs {
            *q = centered_i8(rnd_u32(&mut seed));
        }
        x.extend_from_slice(&pack_q8_0_block(1.0 / 127.0, &qs));
    }
    let n64 = u64::try_from(n).unwrap_or(0);
    let bytes = write_gguf(&[
        TensorWrite {
            name: "w_q4".into(),
            ty: GgmlType::Q4_0,
            shape: vec![n64, n64],
            data: w,
        },
        TensorWrite {
            name: "x_q8".into(),
            ty: GgmlType::Q8_0,
            shape: vec![n64],
            data: x,
        },
    ]);
    let g = load_gguf(&bytes)?;
    let wt = g
        .tensor("w_q4")
        .ok_or_else(|| "missing tensor w_q4".to_string())?;
    let xt = g
        .tensor("x_q8")
        .ok_or_else(|| "missing tensor x_q8".to_string())?;
    let mut y = vec![0.0f32; n];
    let niter = 8usize;
    for _ in 0..16 {
        gemv_q4_0(n, wt.data, xt.data, &mut y)?;
    }
    let t0 = Instant::now();
    for _ in 0..niter {
        gemv_q4_0(n, wt.data, xt.data, &mut y)?;
    }
    let sec = t0.elapsed().as_secs_f64();
    print_bench(
        "q4_0_gguf",
        n,
        niter,
        sec,
        y.first().copied().unwrap_or(0.0),
    );
    Ok(())
}

fn bench_q4k(n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let mut seed = 5u64;
    let mut w = Vec::new();
    let mut x = Vec::new();
    for _ in 0..n {
        for _b in 0..(n / QK_K) {
            let mut qs = [0u8; QK_K];
            for q in &mut qs {
                *q = u8::try_from(rnd_u32(&mut seed) % 16).unwrap_or(0);
            }
            let mut sc = [0u8; 8];
            let mut mn = [0u8; 8];
            for s in &mut sc {
                *s = u8::try_from(1 + rnd_u32(&mut seed) % 8).unwrap_or(1);
            }
            for m in &mut mn {
                *m = u8::try_from(rnd_u32(&mut seed) % 4).unwrap_or(0);
            }
            let extra = u16::try_from(rnd_u32(&mut seed) % 50).unwrap_or(0);
            w.extend_from_slice(&pack_q4_k_block(
                20.0 / 1000.0 + f32::from(extra) / 1000.0,
                10.0 / 1000.0,
                &sc,
                &mn,
                &qs,
            ));
        }
    }
    for _b in 0..(n / QK_K) {
        let mut qs = [0i8; QK_K];
        for q in &mut qs {
            *q = centered_i8(rnd_u32(&mut seed));
        }
        x.extend_from_slice(&pack_q8_k_block(1.0 / 127.0, &qs));
    }
    let n64 = u64::try_from(n).unwrap_or(0);
    let bytes = write_gguf(&[
        TensorWrite {
            name: "w_q4k".into(),
            ty: GgmlType::Q4_K,
            shape: vec![n64, n64],
            data: w,
        },
        TensorWrite {
            name: "x_q8k".into(),
            ty: GgmlType::Q8_K,
            shape: vec![n64],
            data: x,
        },
    ]);
    let g = load_gguf(&bytes)?;
    let wt = g
        .tensor("w_q4k")
        .ok_or_else(|| "missing tensor w_q4k".to_string())?;
    let xt = g
        .tensor("x_q8k")
        .ok_or_else(|| "missing tensor x_q8k".to_string())?;
    let mut y = vec![0.0f32; n];
    let niter = 8usize;
    for _ in 0..16 {
        gemv_q4_k(n, wt.data, xt.data, &mut y)?;
    }
    let t0 = Instant::now();
    for _ in 0..niter {
        gemv_q4_k(n, wt.data, xt.data, &mut y)?;
    }
    let sec = t0.elapsed().as_secs_f64();
    print_bench(
        "q4_k_gguf",
        n,
        niter,
        sec,
        y.first().copied().unwrap_or(0.0),
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for n in [128usize, 256, 512, 1024] {
        bench_q8(n)?;
    }
    bench_q4(256)?;
    bench_q4k(256)?;
    Ok(())
}
