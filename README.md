# llama-rust

Pure-safe Rust GGUF v3 Llama-family **prompt → text**. No llama.cpp bind, no C GGML snapshot, no `unsafe`.

See [STATUS.md](STATUS.md) for what shipped, what is not started, and the resume list.

Loads mixed Q4_K_M-shaped dtypes (**F32**, **F16**, **Q4_K**, **Q5_K**, **Q6_K**, **IQ2_XXS**, **IQ2_XS**, **IQ2_S**, **IQ3_XXS**, **IQ3_S**, **IQ4_NL**, **IQ4_XS**, plus Q4_0/Q8_0/Q8_K) for `llama` / `qwen2` / `mistral` / `phi3`. F16 2-D weights (`token_embd`, `output`, attn/ffn) stay on-disk IEEE binary16; 1-D norms stay F32. Q5_K 2-D weights stay on-disk `block_q5_K` bytes (`GGML_TYPE_Q5_K` = 13). IQ2_XXS 2-D weights stay on-disk `block_iq2_xxs` bytes (`GGML_TYPE_IQ2_XXS` = 16). IQ2_XS 2-D weights stay on-disk `block_iq2_xs` bytes (`GGML_TYPE_IQ2_XS` = 17). IQ2_S 2-D weights stay on-disk `block_iq2_s` bytes (`GGML_TYPE_IQ2_S` = 22). IQ3_XXS 2-D weights stay on-disk `block_iq3_xxs` bytes (`GGML_TYPE_IQ3_XXS` = 18). IQ3_S 2-D weights stay on-disk `block_iq3_s` bytes (`GGML_TYPE_IQ3_S` = 21). IQ4_NL 2-D weights stay on-disk `block_iq4_nl` bytes (`GGML_TYPE_IQ4_NL` = 20). IQ4_XS 2-D weights stay on-disk `block_iq4_xs` bytes (`GGML_TYPE_IQ4_XS` = 23). Remaining IQ* (IQ1_S, IQ1_M), and tied `output.weight` reuse are still rejected. Quantized weights **and token embeddings** stay **on-disk bytes** in **one file blob**; `Llama` takes that blob and addresses tensors by range (no per-matrix clone, no mmap). Missing `{arch}.rope.dimension_count` is derived from embedding length / head count. Optional `attn_q`/`attn_k`/`attn_v` bias tensors are applied when present. `tokenizer.ggml.add_bos_token=false` is honored. Tokenizer: GPT-2 / Qwen pieces use bytes-to-unicode (`Ċ` is newline, `Ġ` is space); SentencePiece uses `▁` and `<0xHH>` byte-fallback; `token_id` is a `HashMap`. `tokenizer.chat_template` is read, not rendered. Decode is RMSNorm, RoPE, GQA + KV cache, SwiGLU, lm_head. Prompt prefill is one causal GEMM pass over the prompt tokens; generation after that is one-token GEMV + KV. Sampling default is seedless greedy (argmax). `SampleParams` adds temperature, top-k, top-p, and unique-id repeat penalty; `temperature > 0` needs a seed (SplitMix64). `serve` is a std-only loopback HTTP/1.1 listener: one request at a time, `POST /generate` JSON, seedless greedy. Not a production inference server. Load/decode errors name the tensor, ggml type id, and/or KV key.

Not [onehr/llama-rs](https://github.com/onehr/llama-rs) / [rustformers/llm]. That wrapped frozen GGML. This is GGUF-native.

```
cargo test --release --lib
cargo clippy --all-targets --all-features -- -D warnings
cargo build --release --bin gguf_gemv
./target/release/gguf_gemv write-tiny tiny-llama.gguf
./target/release/gguf_gemv infer tiny-llama.gguf
./target/release/gguf_gemv infer tiny-llama.gguf --prompt a --n-predict 4
./target/release/gguf_gemv serve tiny-llama.gguf
./target/release/gguf_gemv write-tiny-qwen2 tiny-qwen2.gguf
./target/release/gguf_gemv infer tiny-qwen2.gguf -p ab -n 2
./target/release/gguf_gemv write tiny.gguf
./target/release/gguf_gemv gemv tiny.gguf
```

`infer` is seedless greedy. `--prompt` / `-p` (default `ab`), `--n-predict` / `-n` (default `2`), optional `--n-ctx` (KV capacity; default prompt + `n_predict` + 1). Temperature / top-k / top-p / repeat penalty live on `SampleParams` / `generate`; the CLI path is unchanged.

`serve` binds `127.0.0.1:8080` (override with `--bind HOST:PORT`; host must be `127.0.0.1` or `localhost`). One HTTP/1.1 request at a time: `POST /generate` with `{"prompt":"..."}` and optional `n_predict`, response `{"generated":"..."}`. Empty prompt and a missing GGUF file fail cleanly. No batching, no concurrent requests, no OpenAI-compat, no tok/s. Kernel Integrity has not signed it.

Workspace lints deny unwrap/panic/indexing/wrap-casts/`std::fs::{read,write}`. File I/O is `File` + `Read`/`Write`.

Tests write GGUFs (F32 + F16 + Q6_K + Q5_K + Q4_K + IQ2_XXS + IQ2_XS + IQ2_S + IQ3_XXS + IQ3_S + IQ4_NL + IQ4_XS, quantized embeddings, QKV bias, missing rope dim, mistral/phi3 prefixes), load them, compare logits to an independent scalar of the same ggml/Llama math on those bytes (including multi-token prefill, F16 `ggml_fp16_to_fp32`, Q5_K `dequantize_row_q5_K`, IQ2_XXS `dequantize_row_iq2_xxs`, IQ2_XS `dequantize_row_iq2_xs`, IQ2_S `dequantize_row_iq2_s`, IQ3_XXS `dequantize_row_iq3_xxs`, IQ3_S `dequantize_row_iq3_s`, IQ4_NL `dequantize_row_iq4_nl`, and IQ4_XS `dequantize_row_iq4_xs`), then encode/greedy/decode. Sampler tests compare temperature / top-k / top-p / repeat penalty to an independent candidate-list oracle of the same math, plus SplitMix64 published outputs. Tokenizer tests cover GPT-2 `Ċ`/`Ġ` piece decode, `<0xHH>` byte-fallback, SentencePiece `▁`, HashMap `token_id`, and the writer-built tiny merge. Serve tests cover CLI parse, loopback bind, HTTP/1.1 framing, JSON request/response shape, empty prompt, and tiny-model `POST /generate` vs `greedy_generate_ctx` (no 2GB GGUF).

## CLI (this machine)

Apple M4 Pro. Two consecutive `infer` launches on the writer-built tiny Llama GGUF:

```
prompt=ab n_predict=2
generated=ab
```

Both runs print that string.

Same binary on local `models/qwen2.5-3b-instruct-q4_k_m.gguf` (qwen2, no `rope.dimension_count`, Q4_K `token_embd`, Q6_K `output`, F32 QKV biases, `add_bos_token=false`), two runs recorded before GPT-2 piece decode:

```
prompt=ab n_predict=2
generated=abĊĊ
```

`Ċ` is the GPT-2 bytes-to-unicode mapping of newline. Decode now emits `\n` for that piece (the two-run string was two newlines printed raw). `#![forbid(unsafe_code)]`. No FFI. `Cargo.lock` names only `llama-rust`.

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
