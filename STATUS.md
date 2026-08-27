# Status 2026-08-27

HEAD after this slice: infer CLI flags (prompt / n_predict / max-seq). Worktree should be clean.
No in-flight code beyond that slice.

## Shipped (use this)

Repo: https://github.com/mingley/llama-rust
Local: `~/dev/llama-rust-perf`
GHA: Linux `cargo fmt` / clippy / `cargo test --release --lib` on push.

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
- CLI: `gguf_gemv infer <path> [--prompt TEXT] [--n-predict N] [--max-seq N]`.
  Defaults stay `prompt=ab` `n_predict=2` so old one-arg `infer` still matches the
  measured tiny / Qwen2.5-3B strings below. `--max-seq` is optional; omit it and
  KV is `prompt tokens + n_predict + 1`. No clap. `help` / no-args print usage.
- Proven: writer-built tiny two-run `generated=ab`. Real `models/qwen2.5-3b-instruct-q4_k_m.gguf` two-run `generated=abĊĊ` (~1.4s, M4 Pro, defaults).
- Metal Q8 GEMV is a **measurement binary** (`q8_gemv.metal` + `q8_gemv_mtl.m`), not linked into the crate. Occupied min 6735 gemv/s vs CPU min 1337, y0=78.165176 both.
- Constraints that stay: no `#[allow]`, clippy workspace lints, no `std::fs::{read,write}`, no Mutex/RwLock, `thread::scope` row dispatch, author `mingley`.

## In progress

Nothing.

## Still needed (production / researcher bar)

Ordered by how much they block “others can actually use this”:

1. **Stop cloning the GGUF.** `Gguf` owns tensor bytes; `QuantMat` clones them again. 2.1GB file → ~4GB RSS. mmap or move the buffers once. mmap needs `unsafe` or a crate; both are currently parked.
2. **Prefill GEMM.** Prompt tokens are decoded one-by-one. Fine for `"ab"`, not for a 2k-token prompt.
3. **Tokenizer.** GPT-2 BPE is character-then-merge, linear `id_of` over 151k vocab, no byte-fallback. Decode of Qwen pieces produced `ĊĊ`. Chat template is unread.
4. **Sampling.** Greedy only. No temperature / top-k / top-p / repeat penalty.
5. **Serving.** No HTTP, no OpenAI-compat, no batching, no multi-request. Not a production inference server.
6. **Metal-in-crate.** Owned MSL kernels exist as a sidecar. Decode still CPU.
7. **Dtypes / arches still rejected.** Q5_K, F16, IQ*, Gemma, MoE, vision, Qwen3, Llama4. Tied `output.weight` (reuse `token_embd`) untested.
8. **KV cache** sized to prompt+predict, not `{arch}.context_length`. `--max-seq` can raise the cap; it does not read the GGUF context key.
9. **crates.io** unpublished. Linux proof is GHA tiny/oracle tests only (2GB GGUF is gitignored).

Non-goals that were explicitly parked: SIMD crates (`wide`/`pulp`/`std::simd`); matching llama.cpp tok/s; downloading HF checkpoints in CI.

## Resume

```
cargo test --release --lib
cargo clippy --all-targets --all-features -- -D warnings
./target/release/gguf_gemv infer models/qwen2.5-3b-instruct-q4_k_m.gguf
./target/release/gguf_gemv infer models/qwen2.5-3b-instruct-q4_k_m.gguf --prompt ab --n-predict 2
```

Next code change is item 1 (avoid the weight clone) unless the goal is tokenizer quality or prefill. Do not add crates.io runtime deps or `unsafe`.
