# llama-rust

Pure-safe Rust GGUF v3 Llama-family **prompt → text**. No llama.cpp bind, no C GGML snapshot, no `unsafe`.

See [STATUS.md](STATUS.md) for what shipped, what is not started, and the resume list.

Loads mixed Q4_K_M-shaped dtypes (**F32**, **Q4_K**, **Q6_K**, plus Q4_0/Q8_0/Q8_K) for `llama` / `qwen2` / `mistral` / `phi3`. Quantized weights **and token embeddings** stay **on-disk bytes**. Missing `{arch}.rope.dimension_count` is derived from embedding length / head count. Optional `attn_q`/`attn_k`/`attn_v` bias tensors are applied when present. `tokenizer.ggml.add_bos_token=false` is honored. Decode is RMSNorm, RoPE, GQA + KV cache, SwiGLU, lm_head. Sampling is greedy. Load/decode errors name the tensor, ggml type id, and/or KV key.

Not [onehr/llama-rs](https://github.com/onehr/llama-rs) / [rustformers/llm]. That wrapped frozen GGML. This is GGUF-native.

```
cargo test --release --lib
cargo clippy --all-targets --all-features -- -D warnings
cargo build --release --bin gguf_gemv
./target/release/gguf_gemv write-tiny tiny-llama.gguf
./target/release/gguf_gemv infer tiny-llama.gguf
./target/release/gguf_gemv infer tiny-llama.gguf --prompt a --n-predict 1
./target/release/gguf_gemv write-tiny-qwen2 tiny-qwen2.gguf
./target/release/gguf_gemv infer tiny-qwen2.gguf -p ab -n 2
./target/release/gguf_gemv write tiny.gguf
./target/release/gguf_gemv gemv tiny.gguf
```

Workspace lints deny unwrap/panic/indexing/wrap-casts/`std::fs::{read,write}`. File I/O is `File` + `Read`/`Write`.

Tests write GGUFs (F32 + Q6_K + Q4_K, quantized embeddings, QKV bias, missing rope dim, mistral/phi3 prefixes), load them, compare logits to an independent scalar of the same ggml/Llama math on those bytes, then encode/greedy/decode.

## CLI (this machine)

`infer` takes a GGUF path plus optional `--prompt` / `-p` and `--n-predict` / `-n` (defaults `ab` / `2`, same as the previous harness). Sampling is still seedless greedy.

Apple M4 Pro. Two consecutive `infer` launches on the writer-built tiny Llama GGUF (defaults):

```
prompt=ab n_predict=2
generated=ab
```

Both runs print that string.

Same binary on local `models/qwen2.5-3b-instruct-q4_k_m.gguf` (qwen2, no `rope.dimension_count`, Q4_K `token_embd`, Q6_K `output`, F32 QKV biases, `add_bos_token=false`), two runs:

```
prompt=ab n_predict=2
generated=abĊĊ
```

Both runs print that string (~1.4s). `#![forbid(unsafe_code)]`. No FFI. `Cargo.lock` names only `llama-rust`.

Q8_0 GEMV (`write`+`gemv`, M=K=4096, GGUF-byte blocks). Two-run `y0=` match across Rust CPU and owned Metal. Crate lockfile is still only `llama-rust`; Metal is a measurement binary (`q8_gemv.metal` + `q8_gemv_mtl.m`), not linked.

```
lang=Rust kernel=q8_0_gguf M=4096 K=4096 niter=8
time_s=0.002375 gemv/s=3368.42
y_checksum=80e188057fa1eef0 y0=78.165176

lang=Rust kernel=q8_0_gguf M=4096 K=4096 niter=8
time_s=0.005984 gemv/s=1336.81
y_checksum=80e188057fa1eef0 y0=78.165176

lang=Metal kernel=q8_0_gguf M=4096 K=4096 niter=32
device=Apple M4 Pro
time_s=0.004644 gemv/s=6890.06
y0=78.165176

lang=Metal kernel=q8_0_gguf M=4096 K=4096 niter=32
device=Apple M4 Pro
time_s=0.004751 gemv/s=6735.37
y0=78.165176
```

min(Metal gemv/s) / min(Rust gemv/s) = 6735.37 / 1336.81 = **5.04**. Metal: one 2-D dispatch (simdgroup K-split, 32 in-flight GEMVs), one commit/wait. C 1-thread on the same pack stays ~900 gemv/s (`langtax/q8_gemv.c`).

## Linux testing (free)

No paid VM required.

- **GitHub Actions `ubuntu-latest`** on this public repo (`.github/workflows/ci.yml`): Linux `cargo test` / clippy on every push. You do not operate a machine.
- **Persistent VM:** [Oracle Cloud Always Free](https://www.oracle.com/cloud/free/) Ampere A1 (aarch64) is the optional always-on Linux box if you want a shell. Same `cargo test --release --lib` there.

This crate is `forbid(unsafe_code)` and lockfile-only, so those Linux jobs do not need llama.cpp, Metal, or crates.io SIMD packages.
