# Status 2026-08-27 — Kernel Integrity HOLD

Honesty pass on `main`. Parent of this file: `4beae44` (“Add infer CLI flags
for prompt, n_predict, and n_ctx”). After this commit, HEAD is this tree.

HOLD: **do not claim a win vs llama.cpp.** This repository has **no
llama.cpp binary** and **no tok/s table**. Do not invent llama.cpp numbers.

GitHub About still says `measured vs llama.cpp`. That is a repo-settings
field, not a file. It still needs a repo-settings write to:

`Safe Rust GGUF-native Llama decode. No llama.cpp bind. Not rustformers/llm.`

The honest in-repo About is that sentence. No measurement vs llama.cpp
exists in this tree.

CLI: #2 is on `main` as `4beae44` (Kernel Integrity passed it as a CLI
slice). #1 is closed. #3 stays draft. Do not merge #3.

Worktree should be clean. No in-flight code beyond this docs/status pass.

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
- CLI (on `4beae44`): `gguf_gemv infer <path> [--prompt TEXT] [--n-predict N] [--n-ctx N]`.
  Seedless greedy. Defaults remain `ab` / 2.
- Proven on Apple M4 Pro only (not this Linux host, not llama.cpp): writer-built
  tiny two-run `generated=ab`. Local `models/qwen2.5-3b-instruct-q4_k_m.gguf`
  two-run `generated=abĊĊ` (~1.4s).
- Metal Q8 GEMV is a **measurement binary** (`q8_gemv.metal` + `q8_gemv_mtl.m`),
  not linked into the crate. Occupied min 6735 gemv/s vs CPU min 1337,
  y0=78.165176 both — **same M4 Pro, Rust vs owned Metal**, not vs llama.cpp.
- Constraints that stay: no `#[allow]`, clippy workspace lints, no
  `std::fs::{read,write}`, no Mutex/RwLock, `thread::scope` row dispatch,
  author `mingley`.

## In progress

Nothing. Docs/status/About honesty only.

## Still needed (production / researcher bar)

Ordered by how much they block “others can actually use this”:

1. **Stop cloning the GGUF.** `Gguf` owns tensor bytes; `QuantMat` clones them again. 2.1GB file → ~4GB RSS. Move the buffers once (mmap needs `unsafe` or a crate — both forbidden).
2. **Prefill GEMM.** Prompt tokens are decoded one-by-one. Fine for short `--prompt`, not for a 2k-token prompt.
3. **Tokenizer.** GPT-2 BPE is character-then-merge, linear `id_of` over 151k vocab, no byte-fallback. Decode of Qwen pieces produced `ĊĊ`. Chat template is unread.
4. **Sampling.** Greedy only. No temperature / top-k / top-p / repeat penalty.
5. **Serving.** No HTTP, no OpenAI-compat, no batching, no multi-request. Not a production inference server.
6. **Metal-in-crate.** Owned MSL kernels exist as a sidecar. Decode still CPU.
7. **Dtypes / arches still rejected.** Q5_K, F16, IQ*, Gemma, MoE, vision, Qwen3, Llama4. Tied `output.weight` (reuse `token_embd`) untested.
8. **KV cache** sized to prompt+predict is the default; `{arch}.context_length` is still unused. `--n-ctx` is an override only.
9. **crates.io** unpublished. Linux proof is GHA tiny/oracle tests only (2GB GGUF is gitignored).

Parked: SIMD crates (`wide`/`pulp`/`std::simd`); a llama.cpp tok/s comparison
(no llama.cpp binary in this tree — do not invent one); downloading HF
checkpoints in CI. Kernel Integrity signs later.

## Resume

```
cargo test --release --lib
cargo clippy --all-targets --all-features -- -D warnings
./target/release/gguf_gemv infer models/qwen2.5-3b-instruct-q4_k_m.gguf --prompt ab --n-predict 2
```

Next code change should be item 1 (move weight bytes once; do not add
`unsafe` or a mmap crate) unless prefill GEMM is the goal. Do not add
crates.io runtime deps. #3 stays draft.
