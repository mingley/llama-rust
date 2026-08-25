//! Load a GGUF (or write a tiny one) and GEMV Q8_0 W × Q8_0 x.

use std::env;
use std::fs::File;
use std::io::{Read, Write};
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

fn rnd_u32(seed: &mut u64) -> u32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    u32::try_from(*seed >> 33).unwrap_or(0)
}

fn centered_i8(u: u32) -> i8 {
    let n = i32::try_from(u % 255).unwrap_or(0) - 128;
    i8::try_from(n).unwrap_or(0)
}

fn read_path(path: &Path) -> Result<Vec<u8>, std::io::Error> {
    let mut f = File::open(path)?;
    let mut buf = Vec::new();
    let _n = f.read_to_end(&mut buf)?;
    Ok(buf)
}

fn write_path(path: &Path, bytes: &[u8]) -> Result<(), std::io::Error> {
    let mut f = File::create(path)?;
    f.write_all(bytes)?;
    Ok(())
}

fn demo_gguf(n_cols: usize, n_rows: usize) -> Vec<u8> {
    let mut w = Vec::new();
    let mut x = Vec::new();
    let mut seed = 1u64;
    for _r in 0..n_rows {
        for _b in 0..(n_cols / QK8_0) {
            let mut qs = [0i8; QK8_0];
            for q in &mut qs {
                *q = centered_i8(rnd_u32(&mut seed));
            }
            let extra = u16::try_from(rnd_u32(&mut seed) % 80).unwrap_or(0);
            let amax = 20.0 / 1000.0 + f32::from(extra) / 1000.0;
            w.extend_from_slice(&pack_q8_0_block(amax, &qs));
        }
    }
    for _b in 0..(n_cols / QK8_0) {
        let mut qs = [0i8; QK8_0];
        for q in &mut qs {
            *q = centered_i8(rnd_u32(&mut seed));
        }
        x.extend_from_slice(&pack_q8_0_block(1.0 / 127.0, &qs));
    }
    write_gguf(&[
        TensorWrite {
            name: "w_q8".into(),
            ty: GgmlType::Q8_0,
            shape: vec![
                u64::try_from(n_cols).unwrap_or(0),
                u64::try_from(n_rows).unwrap_or(0),
            ],
            data: w,
        },
        TensorWrite {
            name: "x_q8".into(),
            ty: GgmlType::Q8_0,
            shape: vec![u64::try_from(n_cols).unwrap_or(0)],
            data: x,
        },
    ])
}

fn gemv_file(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = read_path(path)?;
    let g = load_gguf(&bytes)?;
    let w = g
        .tensor("w_q8")
        .ok_or_else(|| format!("missing tensor w_q8 in {}", path.display()))?;
    let x = g
        .tensor("x_q8")
        .ok_or_else(|| format!("missing tensor x_q8 in {}", path.display()))?;
    let n_cols = w.n_cols();
    let n_rows = w.n_rows();
    let mut y = vec![0.0f32; n_rows];
    for _ in 0..8 {
        gemv_q8_0(n_cols, &w.data, &x.data, &mut y)?;
    }
    let niter = 8usize;
    let t0 = Instant::now();
    for _ in 0..niter {
        gemv_q8_0(n_cols, &w.data, &x.data, &mut y)?;
    }
    let sec = t0.elapsed().as_secs_f64();
    let niter_f = f64::from(u32::try_from(niter).unwrap_or(0));
    let gemv_s = if sec > 0.0 { niter_f / sec } else { 0.0 };
    let y0 = y.first().copied().unwrap_or(0.0);
    println!("lang=Rust kernel=q8_0_gguf M={n_rows} K={n_cols} niter={niter}");
    println!("time_s={sec:.6} gemv/s={gemv_s:.2}");
    println!("y_checksum={:016x} y0={y0:.6}", y_checksum(&y));
    Ok(())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| "gemv".into());
    match cmd.as_str() {
        "write" => {
            let path = args.next().ok_or("write <path>")?;
            let n_cols = 256usize;
            let n_rows = 128usize;
            let bytes = demo_gguf(n_cols, n_rows);
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "gemv" => {
            let path = args.next().ok_or("gemv <path>")?;
            gemv_file(Path::new(&path))
        }
        other => Err(format!(
            "usage: gguf_gemv write <path> | gguf_gemv gemv <path> (got {other})"
        )
        .into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()
}
