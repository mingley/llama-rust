# Stopped 2026-08-27 — GGUF move-once (no mmap)

HEAD is this branch’s tip. Worktree should be clean before the next resume.
No in-flight code.

## Shipped (use this)

Repo: https://github.com/mingley/llama-rust
Local: `~/dev/llama-rust-perf`

- `forbid(unsafe_code)`, no llama.cpp/FFI, `Cargo.lock` crate-only (no SIMD crates, no rayon).
- GGUF v3: F32, Q4_0, Q8_0, Q4_K, Q6_K, Q8_K. Kernels read on-disk bytes (no private f32-scale copy).
- Decode: RMSNorm, RoPE, GQA+KV, SwiGLU, lm_head, greedy sample.
- Architectures: `llama`, `qwen2`, `mistral`, `phi3` `{arch}.*` KV.
- Q4_K_M shape that common OSS files actually have:
  - quantized `token_embd.weight` (Q4_K / Q6_K / F32)
  - missing `{arch}.rope.dimension_count` derived from `embedding_length / head_count`
  - optional F32 `attn_{q,k,v}.bias`
  - `tokenizer.ggml.add_bos_token=false` honored
- Load/decode errors name tensor, ggml type id, and/or KV key.
- CLI: `gguf_gemv infer <path> [--prompt TEXT] [--n-predict N] [--n-ctx N]`. Seedless greedy. Defaults remain `ab` / 2 so the shipped two-run command still works.
- **One file blob.** `load_gguf_owned(Vec<u8>)` keeps the file bytes. Tensor payloads are ranges of that blob. `Llama::from_gguf` takes the blob; `QuantMat` stores offsets, not a second copy. mmap is still forbidden (`unsafe` or a crate).
- Proven (unchanged): writer-built tiny two-run `generated=ab`. Real `models/qwen2.5-3b-instruct-q4_k_m.gguf` two-run `generated=abĊĊ` (~1.4s, M4 Pro).
- Metal Q8 GEMV is a **measurement binary** (`q8_gemv.metal` + `q8_gemv_mtl.m`), not linked into the crate. Occupied min 6735 gemv/s vs CPU min 1337, y0=78.165176 both.
- Constraints that stay: no `#[allow]`, clippy workspace lints, no `std::fs::{read,write}`, no Mutex/RwLock, `thread::scope` row dispatch, author `mingley`.

## In progress

Nothing. This slice is STATUS item 1 (stop cloning the GGUF without mmap).

## Still needed (production / researcher bar)

Ordered by how much they block “others can actually use this”:

1. **Prefill GEMM.** Prompt tokens are decoded one-by-one. Fine for short `--prompt`, not for a 2k-token prompt.
2. **Tokenizer.** GPT-2 BPE is character-then-merge, linear `id_of` over 151k vocab, no byte-fallback. Decode of Qwen pieces produced `ĊĊ`. Chat template is unread.
3. **Sampling.** Greedy only. No temperature / top-k / top-p / repeat penalty.
4. **Serving.** No HTTP, no OpenAI-compat, no batching, no multi-request. Not a production inference server.
5. **Metal-in-crate.** Owned MSL kernels exist as a sidecar. Decode still CPU.
6. **Dtypes / arches still rejected.** Q5_K, F16, IQ*, Gemma, MoE, vision, Qwen3, Llama4. Tied `output.weight` (reuse `token_embd`) untested.
7. **KV cache** sized to prompt+predict is the default; `{arch}.context_length` is still unused. `--n-ctx` is an override only.
8. **crates.io** unpublished. Linux proof is GHA tiny/oracle tests only (2GB GGUF is gitignored).

Non-goals that were explicitly parked: SIMD crates (`wide`/`pulp`/`std::simd`); matching llama.cpp tok/s; downloading HF checkpoints in CI; mmap (`unsafe` or a crate).

## Resume

```
cargo test --release --lib
cargo clippy --all-targets --all-features -- -D warnings
./target/release/gguf_gemv infer models/qwen2.5-3b-instruct-q4_k_m.gguf --prompt ab --n-predict 2
```

Next code change should be item 1 (prefill GEMM). Do not add crates.io runtime deps or `unsafe`.
