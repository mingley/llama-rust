# Stopped 2026-08-27 — tokenizer

HEAD is this branch’s tip. Worktree should be clean before the next resume.
No in-flight code.

## Shipped (use this)

Repo: https://github.com/mingley/llama-rust
Local: `~/dev/llama-rust-perf`

- `forbid(unsafe_code)`, no llama.cpp/FFI, `Cargo.lock` crate-only (no SIMD crates, no rayon).
- GGUF v3: F32, Q4_0, Q8_0, Q4_K, Q6_K, Q8_K. Kernels read on-disk bytes (no private f32-scale copy).
- Decode: RMSNorm, RoPE, GQA+KV, SwiGLU, lm_head, greedy sample.
- Prefill GEMM. Prompt tokens are one causal pass. A single token stays GEMV.
- Architectures: `llama`, `qwen2`, `mistral`, `phi3` `{arch}.*` KV.
- Q4_K_M shape that common OSS files actually have:
  - quantized `token_embd.weight` (Q4_K / Q6_K / F32)
  - missing `{arch}.rope.dimension_count` derived from `embedding_length / head_count`
  - optional F32 `attn_{q,k,v}.bias`
  - `tokenizer.ggml.add_bos_token=false` honored
- Load/decode errors name tensor, ggml type id, and/or KV key.
- CLI: `gguf_gemv infer <path> [--prompt TEXT] [--n-predict N] [--n-ctx N]`. Seedless greedy. Defaults remain `ab` / 2 so the shipped two-run command still works.
- One file blob. `load_gguf_owned(Vec<u8>)` keeps the file bytes. Tensor payloads are ranges of that blob. mmap is still forbidden (`unsafe` or a crate).
- **Tokenizer.** `token_id` / merge rank are `HashMap` lookups, not a linear scan of the vocab.
  - `tokenizer.ggml.model=gpt2` (and vocabs that contain `Ġ` / `Ċ`): UTF-8 bytes → GPT-2 bytes-to-unicode → BPE. Decode maps `Ċ` → `\n` and `Ġ` → space. The recorded Qwen `generated=abĊĊ` was two newline pieces printed raw.
  - `tokenizer.ggml.model=llama` (and vocabs with `▁` / `<0xHH>`): space → `▁`; unknown UTF-8 → `<0xHH>` when those tokens exist. Decode maps both.
  - Writer-built tiny vocabs stay character-then-merge (`encode("ab")=[3]`, `decode([1,2])="ab"`).
  - `tokenizer.chat_template` is stored on `Tokenizer`. It is not rendered (no Jinja crate).
- Proven (unchanged weights): writer-built tiny two-run `generated=ab`. Real `models/qwen2.5-3b-instruct-q4_k_m.gguf` two-run was `generated=abĊĊ` (~1.4s, M4 Pro) before piece decode; those pieces are `\n\n`.
- Metal Q8 GEMV is a **measurement binary** (`q8_gemv.metal` + `q8_gemv_mtl.m`), not linked into the crate. Occupied min 6735 gemv/s vs CPU min 1337, y0=78.165176 both.
- Constraints that stay: no `#[allow]`, clippy workspace lints, no `std::fs::{read,write}`, no Mutex/RwLock, `thread::scope` row dispatch, author `mingley`.

## In progress

Nothing. This slice is STATUS item 1 (tokenizer).

## Still needed (production / researcher bar)

Ordered by how much they block “others can actually use this”:

1. **Sampling.** Greedy only. No temperature / top-k / top-p / repeat penalty.
2. **Serving.** No HTTP, no OpenAI-compat, no batching, no multi-request. Not a production inference server.
3. **Metal-in-crate.** Owned MSL kernels exist as a sidecar. Decode still CPU.
4. **Dtypes / arches still rejected.** Q5_K, F16, IQ*, Gemma, MoE, vision, Qwen3, Llama4. Tied `output.weight` (reuse `token_embd`) untested.
5. **KV cache** sized to prompt+predict is the default; `{arch}.context_length` is still unused. `--n-ctx` is an override only.
6. **crates.io** unpublished. Linux proof is GHA tiny/oracle tests only (2GB GGUF is gitignored).
7. **Chat template apply.** The Jinja string is read. Rendering it (and special-token split of `<|im_start|>` in the prompt) is not started. BPE has no Unicode regex pre-tokenizer.

Non-goals that were explicitly parked: SIMD crates (`wide`/`pulp`/`std::simd`); matching llama.cpp tok/s; downloading HF checkpoints in CI; mmap (`unsafe` or a crate).

## Resume

```
cargo test --release --lib
cargo clippy --all-targets --all-features -- -D warnings
./target/release/gguf_gemv infer models/qwen2.5-3b-instruct-q4_k_m.gguf --prompt ab --n-predict 2
```

Next code change should be item 1 (sampling). Do not add crates.io runtime deps or `unsafe`.
