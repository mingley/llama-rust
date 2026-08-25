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

Both runs print that string. `#![forbid(unsafe_code)]`. No FFI. `Cargo.lock` names only `llama-rust`.

Q8_0 GEMV (`write`+`gemv`, M=K=4096, GGUF-byte blocks) two runs, same checksum both times and vs 1-thread C:

```
lang=Rust kernel=q8_0_gguf M=4096 K=4096 niter=8
time_s=0.001684 gemv/s=4751.42
y_checksum=80e188057fa1eef0 y0=78.165176

lang=Rust kernel=q8_0_gguf M=4096 K=4096 niter=8
time_s=0.001824 gemv/s=4386.36
y_checksum=80e188057fa1eef0 y0=78.165176

lang=C kernel=q8_0_gguf M=4096 K=4096 niter=8
time_s=0.008823 gemv/s=906.76
y_checksum=80e188057fa1eef0 y0=78.165176

lang=C kernel=q8_0_gguf M=4096 K=4096 niter=8
time_s=0.008421 gemv/s=950.02
y_checksum=80e188057fa1eef0 y0=78.165176
```

min(Rust gemv/s) / max(C gemv/s) = 4386.36 / 950.02 = **4.62**. C is `clang -O3 -mcpu=native` on `langtax/q8_gemv.c` (measurement binary, not linked into the crate). Rust is `--release` `-C target-cpu=native`.

Owned Metal counterpart (also not linked): `langtax/q8_gemv.metal` + `q8_gemv_mtl.m`. Same GGUF 34-byte blocks, runtime-compiled on `Apple M4 Pro`. Naive one-thread-per-row: **1058 gemv/s**, y0=78.165176. Occupancy/simdgroup work is still open; the kernel is ours.
