# llama-rust

Pure-safe Rust GGUF v3 Q4_0 / Q8_0 load + GEMV. No llama.cpp bind, no C GGML snapshot, no `unsafe`.

Blocks are **on-disk GGUF layout**: IEEE binary16 scale + packed `qs` (Q8_0 = 34 bytes, Q4_0 = 18 bytes). GEMV reads those bytes; it does not copy into an f32-scale private struct.

Not [onehr/llama-rs](https://github.com/onehr/llama-rs) / [rustformers/llm](https://github.com/rustformers/llm). That was a full CPU inference CLI on frozen GGML. This is the load+matmul foundation: GGUF-native kernels first.

```
cargo test --release --manifest-path langtax/Cargo.toml --lib
RUSTFLAGS='-C target-cpu=native' cargo build --release --manifest-path langtax/Cargo.toml --bin gguf_gemv
./langtax/target/release/gguf_gemv write tiny.gguf
./langtax/target/release/gguf_gemv gemv tiny.gguf
```

`cargo test` writes a GGUF via the shipped writer, loads it, runs `gemv_q8_0` / `gemv_q4_0` on the tensor bytes, and compares to an independent fp16+qs unpack of the **same file bytes**.

## GGUF CLI (this machine)

Apple M4 Pro. Demo GGUF: Q8_0 `w` [256, 128] × Q8_0 `x` [256]. Two consecutive `gemv` launches:

| run | gemv/s | y_checksum | y0 |
|---|---:|---|---:|
| 1 | 23741.83 | `9fe974004a730987` | 9.397801 |
| 2 | 63576.33 | `9fe974004a730987` | 9.397801 |

Checksums match. `#![forbid(unsafe_code)]`. Lockfile deps: `llama-rust` + rayon/crossbeam/either only.

## Extra GGUF cells

`extra` writes/loads GGUF then GEMVs:

| kernel | M=K | gemv/s |
|---|---:|---:|
| q8_0_gguf | 128 | 78688 |
| q8_0_gguf | 256 | 34127 |
| q8_0_gguf | 512 | 25387 |
| q8_0_gguf | 1024 | 13155 |
| q4_0_gguf | 256 | 21648 |

## Historical 1-thread C vs synthetic f32-scale Rust

Older `langtax/q8_gemv.c` used f32 scales (not GGUF). Kept as a C kernel study only; not the inference path.

llama.cpp Metal on this Mac (Qwen2.5-3B Q4_K_M, not loaded by this crate): pp512 **1097**, tg128 **91**.
