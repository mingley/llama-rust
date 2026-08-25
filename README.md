# llama-rust

Pure-safe Rust GGUF v3 load + GEMV for Q4_0 / Q8_0 / Q4_K / Q8_K. No llama.cpp bind, no C GGML snapshot, no `unsafe`.

Blocks are **on-disk GGUF layout**. Q8_0 = 34 bytes (binary16 + qs), Q4_0 = 18, Q4_K = 144 (binary16 `d`/`dmin` + packed 6-bit scales/mins + qs), Q8_K = 292 (f32 `d` + qs + bsums). GEMV reads those bytes; it does not copy scales into a private f32 tensor.

The loader parses every GGUF v3 KV type (`UINT8`…`FLOAT64`, `BOOL`, `STRING`, `ARRAY` with nested array headers). Tensor payloads stay the file bytes.

Not [onehr/llama-rs](https://github.com/onehr/llama-rs) / [rustformers/llm](https://github.com/rustformers/llm). That was a full CPU inference CLI on frozen GGML. This is the load+matmul foundation.

```
cargo test --release --lib
cargo clippy --all-targets --all-features -- -D warnings
RUSTFLAGS='-C target-cpu=native' cargo build --release --bin gguf_gemv
./target/release/gguf_gemv write tiny.gguf
./target/release/gguf_gemv gemv tiny.gguf
./target/release/gguf_gemv write-q4k tiny-q4k.gguf
./target/release/gguf_gemv gemv-q4k tiny-q4k.gguf
```

Workspace `[lints]` + `clippy.toml` deny unwrap/expect/panic, indexing, wrap/truncation casts, `std::fs::{read,write}`, and `std::sync::{Mutex,RwLock}`. Tests may still unwrap/index/panic. File I/O is `File` + `Read`/`Write`.

`cargo test` writes a GGUF via the shipped writer (including `BOOL` / `FLOAT32` / `ARRAY` KV plus Q4_0/Q8_0/Q4_K tensors), loads it, runs GEMV on the tensor bytes, and compares Q4_K×Q8_K to an independent `dequantize_row_q4_K` / `dequantize_row_q8_K` then f32 dot of the **same file bytes**.

## GGUF CLI (this machine)

Apple M4 Pro.

Q8_0 demo: `w` [256, 128] × `x` [256]. `y_checksum=9fe974004a730987` `y0=9.397801`.

Q4_K demo: `w` [256, 64] × Q8_K `x` [256]. Two consecutive `gemv-q4k` launches:

| run | gemv/s | y_checksum | y0 |
|---|---:|---|---:|
| 1 | 28281 | `f1de24b06c8f494c` | -32.477642 |
| 2 | 43518 | `f1de24b06c8f494c` | -32.477642 |

Checksums match. `#![forbid(unsafe_code)]`. Lockfile deps: `llama-rust` + rayon/crossbeam/either only.

## Extra GGUF cells

`extra` writes/loads GGUF then GEMVs:

| kernel | M=K | gemv/s |
|---|---:|---:|
| q8_0_gguf | 128 | 32537 |
| q8_0_gguf | 256 | 24980 |
| q8_0_gguf | 512 | 44693 |
| q8_0_gguf | 1024 | 12521 |
| q4_0_gguf | 256 | 36377 |
| q4_k_gguf | 256 | 11996 |

## Historical 1-thread C vs synthetic f32-scale Rust

Older `langtax/q8_gemv.c` used f32 scales (not GGUF). Kept as a C kernel study only; not the inference path.

llama.cpp Metal on this Mac (Qwen2.5-3B Q4_K_M, not loaded by this crate): pp512 **1097**, tg128 **91**.
