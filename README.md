# llama-rust

Pure-safe Rust GGUF v3 Llama-family **prompt → text**. No llama.cpp bind, no C GGML snapshot, no `unsafe`.

Loads mixed Q4_K_M-shaped dtypes (**F32**, **Q4_K**, **Q6_K**, plus Q4_0/Q8_0/Q8_K). Quantized weights stay **on-disk bytes**. Decode is RMSNorm, RoPE, GQA + KV cache, SwiGLU, lm_head. Tokenizer is the vocab/merges embedded in the GGUF. Sampling is greedy.

Not [onehr/llama-rs](https://github.com/onehr/llama-rs) / [rustformers/llm]. That wrapped frozen GGML. This is GGUF-native.

```
cargo test --release --lib
cargo clippy --all-targets --all-features -- -D warnings
cargo build --release --bin gguf_gemv
./target/release/gguf_gemv write-tiny tiny-llama.gguf
./target/release/gguf_gemv infer tiny-llama.gguf
./target/release/gguf_gemv write tiny.gguf
./target/release/gguf_gemv gemv tiny.gguf
```

Workspace lints deny unwrap/panic/indexing/wrap-casts/`std::fs::{read,write}`. File I/O is `File` + `Read`/`Write`.

Tests write a GGUF (F32 + Q6_K + Q4_K, tokenizer KV), load it, compare logits to an independent scalar of the same ggml/Llama math on those bytes, then encode/greedy/decode.

## CLI (this machine)

Apple M4 Pro. Two consecutive `infer` launches on the writer-built tiny Llama GGUF:

```
prompt=ab n_predict=2
generated=ab
```

Both runs print that string. `#![forbid(unsafe_code)]`. Lockfile: `llama-rust` + rayon/crossbeam/either.

Q8_0 GEMV subcommand still `y_checksum=9fe974004a730987`.

## Extra GEMV cells

| kernel | M=K | gemv/s |
|---|---:|---:|
| q8_0_gguf | 128 | 32537 |
| q8_0_gguf | 256 | 24980 |
| q8_0_gguf | 512 | 44693 |
| q8_0_gguf | 1024 | 12521 |
| q4_0_gguf | 256 | 36377 |
| q4_k_gguf | 256 | 11996 |
