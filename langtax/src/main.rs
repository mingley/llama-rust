//! Load a GGUF (or write a tiny one) and GEMV Q8_0 or Q4_K.

use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;

use llama_rust::{
    gemv_q4_k, gemv_q8_0, greedy_generate_with, load_gguf, pack_q4_k_block, pack_q8_0_block,
    pack_q8_k_block, parse_args, tiny_llama_gguf, tiny_qwen2_gguf, usage, write_gguf,
    write_gguf_with_kv, Cmd, GgmlType, InferArgs, Kv, Llama, TensorWrite, Tokenizer, QK8_0, QK_K,
};

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

fn demo_q4k_gguf(n_cols: usize, n_rows: usize) -> Vec<u8> {
    let mut w = Vec::new();
    let mut x = Vec::new();
    let mut seed = 1u64;
    for _r in 0..n_rows {
        for _b in 0..(n_cols / QK_K) {
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
            let d = 20.0 / 1000.0 + f32::from(extra) / 1000.0;
            w.extend_from_slice(&pack_q4_k_block(d, 10.0 / 1000.0, &sc, &mn, &qs));
        }
    }
    for _b in 0..(n_cols / QK_K) {
        let mut qs = [0i8; QK_K];
        for q in &mut qs {
            *q = centered_i8(rnd_u32(&mut seed));
        }
        x.extend_from_slice(&pack_q8_k_block(1.0 / 127.0, &qs));
    }
    write_gguf_with_kv(
        &[
            ("general.alignment".into(), Kv::U32(32)),
            ("general.name".into(), Kv::String("llama-rust".into())),
            ("llama.ok".into(), Kv::Bool(true)),
            ("llama.scale".into(), Kv::F32(1.5)),
            (
                "llama.ids".into(),
                Kv::Array {
                    elem: 4,
                    items: vec![Kv::U32(1), Kv::U32(2)],
                },
            ),
        ],
        &[
            TensorWrite {
                name: "w_q4k".into(),
                ty: GgmlType::Q4_K,
                shape: vec![
                    u64::try_from(n_cols).unwrap_or(0),
                    u64::try_from(n_rows).unwrap_or(0),
                ],
                data: w,
            },
            TensorWrite {
                name: "x_q8k".into(),
                ty: GgmlType::Q8_K,
                shape: vec![u64::try_from(n_cols).unwrap_or(0)],
                data: x,
            },
        ],
    )
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

fn gemv_q4k_file(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = read_path(path)?;
    let g = load_gguf(&bytes)?;
    let w = g
        .tensor("w_q4k")
        .ok_or_else(|| format!("missing tensor w_q4k in {}", path.display()))?;
    let x = g
        .tensor("x_q8k")
        .ok_or_else(|| format!("missing tensor x_q8k in {}", path.display()))?;
    let n_cols = w.n_cols();
    let n_rows = w.n_rows();
    let mut y = vec![0.0f32; n_rows];
    for _ in 0..8 {
        gemv_q4_k(n_cols, &w.data, &x.data, &mut y)?;
    }
    let niter = 8usize;
    let t0 = Instant::now();
    for _ in 0..niter {
        gemv_q4_k(n_cols, &w.data, &x.data, &mut y)?;
    }
    let sec = t0.elapsed().as_secs_f64();
    let niter_f = f64::from(u32::try_from(niter).unwrap_or(0));
    let gemv_s = if sec > 0.0 { niter_f / sec } else { 0.0 };
    let y0 = y.first().copied().unwrap_or(0.0);
    println!("lang=Rust kernel=q4_k_gguf M={n_rows} K={n_cols} niter={niter}");
    println!("time_s={sec:.6} gemv/s={gemv_s:.2}");
    println!("y_checksum={:016x} y0={y0:.6}", y_checksum(&y));
    Ok(())
}

fn infer_file(opts: &InferArgs) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = read_path(Path::new(&opts.path))?;
    let g = load_gguf(&bytes)?;
    let model = Llama::from_gguf(&g)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let text = greedy_generate_with(&model, &tok, &opts.prompt, opts.n_predict, opts.max_seq)?;
    print!("prompt={}", opts.prompt);
    print!(" n_predict={}", opts.n_predict);
    if let Some(m) = opts.max_seq {
        print!(" max_seq={m}");
    }
    println!();
    println!("generated={text}");
    Ok(())
}

fn write_bytes(path: &str, bytes: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    write_path(Path::new(path), bytes)?;
    println!("wrote {path} bytes={}", bytes.len());
    Ok(())
}

fn run_cmd(cmd: Cmd) -> Result<(), Box<dyn std::error::Error>> {
    match cmd {
        Cmd::Help => {
            println!("{}", usage());
            Ok(())
        }
        Cmd::Write { path } => write_bytes(&path, &demo_gguf(4096, 4096)),
        Cmd::Gemv { path } => gemv_file(Path::new(&path)),
        Cmd::WriteQ4k { path } => write_bytes(&path, &demo_q4k_gguf(256, 64)),
        Cmd::GemvQ4k { path } => gemv_q4k_file(Path::new(&path)),
        Cmd::WriteTiny { path } => write_bytes(&path, &tiny_llama_gguf()),
        Cmd::WriteTinyQwen2 { path } => write_bytes(&path, &tiny_qwen2_gguf()),
        Cmd::Infer(opts) => infer_file(&opts),
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    match parse_args(env::args().skip(1)) {
        Ok(cmd) => run_cmd(cmd),
        Err(e) => {
            eprintln!("{e}");
            eprintln!("{}", usage());
            std::process::exit(2);
        }
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
