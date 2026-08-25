//! Load a GGUF (or write a tiny one) and GEMV Q8_0 W × Q8_0 x.

use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

use llama_rust::{gemv_q8_0, load_gguf, pack_q8_0_block, write_gguf, GgmlType, TensorWrite, QK8_0};

fn y_checksum(y: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in y {
        h ^= u64::from(v.to_bits());
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn demo_gguf(n_cols: usize, n_rows: usize) -> Vec<u8> {
    let mut w = Vec::new();
    let mut x = Vec::new();
    let mut seed = 1u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 33) as u32
    };
    for _r in 0..n_rows {
        for _b in 0..(n_cols / QK8_0) {
            let mut qs = [0i8; QK8_0];
            for q in &mut qs {
                *q = ((rnd() % 255) as i32 - 128) as i8;
            }
            let amax = 0.02 + (rnd() % 80) as f32 / 1000.0;
            w.extend_from_slice(&pack_q8_0_block(amax, &qs));
        }
    }
    for _b in 0..(n_cols / QK8_0) {
        let mut qs = [0i8; QK8_0];
        for q in &mut qs {
            *q = ((rnd() % 255) as i32 - 128) as i8;
        }
        x.extend_from_slice(&pack_q8_0_block(1.0 / 127.0, &qs));
    }
    write_gguf(&[
        TensorWrite {
            name: "w_q8".into(),
            ty: GgmlType::Q8_0,
            shape: vec![n_cols as u64, n_rows as u64],
            data: w,
        },
        TensorWrite {
            name: "x_q8".into(),
            ty: GgmlType::Q8_0,
            shape: vec![n_cols as u64],
            data: x,
        },
    ])
}

fn gemv_file(path: &Path) {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let g = load_gguf(&bytes).unwrap_or_else(|e| panic!("gguf: {e:?}"));
    let w = g.tensor("w_q8").expect("tensor w_q8");
    let x = g.tensor("x_q8").expect("tensor x_q8");
    let n_cols = w.n_cols();
    let n_rows = w.n_rows();
    let mut y = vec![0.0f32; n_rows];
    for _ in 0..8 {
        gemv_q8_0(n_cols, &w.data, &x.data, &mut y);
    }
    let niter = 8usize;
    let t0 = Instant::now();
    for _ in 0..niter {
        gemv_q8_0(n_cols, &w.data, &x.data, &mut y);
    }
    let sec = t0.elapsed().as_secs_f64();
    println!("lang=Rust kernel=q8_0_gguf M={n_rows} K={n_cols} niter={niter}");
    println!(
        "time_s={sec:.6} gemv/s={:.2}",
        niter as f64 / sec.max(1e-12)
    );
    println!("y_checksum={:016x} y0={:.6}", y_checksum(&y), y[0]);
}

fn main() {
    let mut args = env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| "gemv".into());
    match cmd.as_str() {
        "write" => {
            let path = args.next().expect("write <path>");
            let n_cols = 256usize;
            let n_rows = 128usize;
            let bytes = demo_gguf(n_cols, n_rows);
            fs::write(&path, &bytes).unwrap_or_else(|e| panic!("write {path}: {e}"));
            println!("wrote {path} bytes={}", bytes.len());
        }
        "gemv" => {
            let path = args.next().expect("gemv <path>");
            gemv_file(Path::new(&path));
        }
        other => panic!("usage: gguf_gemv write <path> | gguf_gemv gemv <path> (got {other})"),
    }
}
