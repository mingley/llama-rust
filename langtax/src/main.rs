//! Load a GGUF (or write a tiny one) and GEMV Q8_0 or Q4_K.

use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;

use llama_rust::{
    gemv_q4_k, gemv_q8_0, greedy_generate_ctx, greedy_generate_traced, load_gguf_owned,
    pack_q4_k_block, pack_q8_0_block, pack_q8_k_block, parse_chat_args, parse_engine_args,
    parse_infer_args, parse_serve_args, parse_trace_args, run_chat, run_engine, run_serve,
    tiny_bloom_gguf, tiny_gemma2_gguf, tiny_gemma3_gguf, tiny_gemma3n_gguf, tiny_gemma4_gguf,
    tiny_gemma4_moe_gguf, tiny_gemma_gguf, tiny_llama4_gguf, tiny_llama_gguf, tiny_llama_moe_gguf,
    tiny_phi2_gguf, tiny_qwen2_gguf, tiny_qwen2moe_gguf, tiny_qwen2vl_gguf, tiny_qwen35_gguf,
    tiny_qwen3_gguf, tiny_qwen3moe_2layer_gguf, tiny_qwen3moe_gguf, tiny_qwen3next_gguf,
    tiny_qwen3vl_gguf, write_gguf, write_gguf_with_kv, ChatCmd, EngineCmd, GgmlType, InferArgs,
    InferCmd, Kv, Llama, ServeCmd, TensorWrite, Tokenizer, TraceCmd, BIN_USAGE, CHAT_USAGE,
    ENGINE_USAGE, INFER_USAGE, QK8_0, QK_K, SERVE_USAGE, TRACE_USAGE,
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
    let g = load_gguf_owned(bytes)?;
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
        gemv_q8_0(n_cols, w.data, x.data, &mut y)?;
    }
    let niter = 8usize;
    let t0 = Instant::now();
    for _ in 0..niter {
        gemv_q8_0(n_cols, w.data, x.data, &mut y)?;
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
    let g = load_gguf_owned(bytes)?;
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
        gemv_q4_k(n_cols, w.data, x.data, &mut y)?;
    }
    let niter = 8usize;
    let t0 = Instant::now();
    for _ in 0..niter {
        gemv_q4_k(n_cols, w.data, x.data, &mut y)?;
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

fn infer_file(args: &InferArgs) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = read_path(Path::new(&args.path))?;
    let g = load_gguf_owned(bytes)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let model = Llama::from_gguf(g)?;
    let text = greedy_generate_ctx(&model, &tok, &args.prompt, args.n_predict, args.n_ctx)?;
    println!("prompt={} n_predict={}", args.prompt, args.n_predict);
    println!("generated={text}");
    Ok(())
}

fn trace_file(args: &llama_rust::TraceArgs) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = read_path(Path::new(&args.infer.path))?;
    let g = load_gguf_owned(bytes)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let model = Llama::from_gguf(g)?;
    let (text, trace) = greedy_generate_traced(
        &model,
        &tok,
        &args.infer.prompt,
        args.infer.n_predict,
        args.infer.n_ctx,
        0,
    )?;
    write_path(Path::new(&args.out), trace.to_jsonl().as_bytes())?;
    println!(
        "prompt={} n_predict={} events={} --out={}",
        args.infer.prompt,
        args.infer.n_predict,
        trace.events.len(),
        args.out
    );
    println!("generated={text}");
    println!("{}", expertvm::analyze(&trace).report());
    print!(
        "{}",
        expertvm::format_table(&expertvm::compare(&trace, args.capacity, 8))
    );
    Ok(())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| "gemv".into());
    match cmd.as_str() {
        "help" | "--help" | "-h" => {
            print!("{BIN_USAGE}");
            Ok(())
        }
        "write" => {
            let path = args.next().ok_or("write <path>")?;
            let n_cols = 4096usize;
            let n_rows = 4096usize;
            let bytes = demo_gguf(n_cols, n_rows);
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "gemv" => {
            let path = args.next().ok_or("gemv <path>")?;
            gemv_file(Path::new(&path))
        }
        "write-q4k" => {
            let path = args.next().ok_or("write-q4k <path>")?;
            let bytes = demo_q4k_gguf(256, 64);
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "gemv-q4k" => {
            let path = args.next().ok_or("gemv-q4k <path>")?;
            gemv_q4k_file(Path::new(&path))
        }
        "write-tiny" => {
            let path = args.next().ok_or("write-tiny <path>")?;
            let bytes = tiny_llama_gguf();
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "write-tiny-qwen2" => {
            let path = args.next().ok_or("write-tiny-qwen2 <path>")?;
            let bytes = tiny_qwen2_gguf();
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "write-tiny-qwen3" => {
            let path = args.next().ok_or("write-tiny-qwen3 <path>")?;
            let bytes = tiny_qwen3_gguf();
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "write-tiny-gemma" => {
            let path = args.next().ok_or("write-tiny-gemma <path>")?;
            let bytes = tiny_gemma_gguf();
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "write-tiny-gemma2" => {
            let path = args.next().ok_or("write-tiny-gemma2 <path>")?;
            let bytes = tiny_gemma2_gguf();
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "write-tiny-gemma3" => {
            let path = args.next().ok_or("write-tiny-gemma3 <path>")?;
            let bytes = tiny_gemma3_gguf();
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "write-tiny-gemma3n" => {
            let path = args.next().ok_or("write-tiny-gemma3n <path>")?;
            let bytes = tiny_gemma3n_gguf();
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "write-tiny-gemma4" => {
            let path = args.next().ok_or("write-tiny-gemma4 <path>")?;
            let bytes = tiny_gemma4_gguf();
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "write-tiny-gemma4-moe" => {
            let path = args.next().ok_or("write-tiny-gemma4-moe <path>")?;
            let bytes = tiny_gemma4_moe_gguf();
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "write-tiny-llama4" => {
            let path = args.next().ok_or("write-tiny-llama4 <path>")?;
            let bytes = tiny_llama4_gguf();
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "write-tiny-llama-moe" => {
            let path = args.next().ok_or("write-tiny-llama-moe <path>")?;
            let bytes = tiny_llama_moe_gguf();
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "write-tiny-qwen2moe" => {
            let path = args.next().ok_or("write-tiny-qwen2moe <path>")?;
            let bytes = tiny_qwen2moe_gguf();
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "write-tiny-qwen3moe" => {
            let path = args.next().ok_or("write-tiny-qwen3moe <path>")?;
            let bytes = tiny_qwen3moe_gguf();
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "write-tiny-qwen3moe-2layer" => {
            let path = args.next().ok_or("write-tiny-qwen3moe-2layer <path>")?;
            let bytes = tiny_qwen3moe_2layer_gguf();
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "write-tiny-qwen2vl" => {
            let path = args.next().ok_or("write-tiny-qwen2vl <path>")?;
            let bytes = tiny_qwen2vl_gguf();
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "write-tiny-qwen3vl" => {
            let path = args.next().ok_or("write-tiny-qwen3vl <path>")?;
            let bytes = tiny_qwen3vl_gguf();
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "write-tiny-qwen3next" => {
            let path = args.next().ok_or("write-tiny-qwen3next <path>")?;
            let bytes = tiny_qwen3next_gguf();
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "write-tiny-qwen35" => {
            let path = args.next().ok_or("write-tiny-qwen35 <path>")?;
            let bytes = tiny_qwen35_gguf();
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "write-tiny-phi2" => {
            let path = args.next().ok_or("write-tiny-phi2 <path>")?;
            let bytes = tiny_phi2_gguf();
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "write-tiny-bloom" => {
            let path = args.next().ok_or("write-tiny-bloom <path>")?;
            let bytes = tiny_bloom_gguf();
            write_path(Path::new(&path), &bytes)?;
            println!("wrote {path} bytes={}", bytes.len());
            Ok(())
        }
        "infer" => match parse_infer_args(args)? {
            InferCmd::Help => {
                print!("{INFER_USAGE}");
                Ok(())
            }
            InferCmd::Run(opts) => infer_file(&opts),
        },
        "trace" => match parse_trace_args(args)? {
            TraceCmd::Help => {
                print!("{TRACE_USAGE}");
                Ok(())
            }
            TraceCmd::Run(opts) => trace_file(&opts),
        },
        "chat" => match parse_chat_args(args)? {
            ChatCmd::Help => {
                print!("{CHAT_USAGE}");
                Ok(())
            }
            ChatCmd::Run(opts) => run_chat(&opts),
        },
        "serve" => match parse_serve_args(args)? {
            ServeCmd::Help => {
                print!("{SERVE_USAGE}");
                Ok(())
            }
            ServeCmd::Run(opts) => Ok(run_serve(&opts)?),
        },
        "engine" => match parse_engine_args(args)? {
            EngineCmd::Help => {
                print!("{ENGINE_USAGE}");
                Ok(())
            }
            EngineCmd::Run(opts) => run_engine(&opts),
        },
        other => Err(format!("{BIN_USAGE}got {other}").into()),
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
