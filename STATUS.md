# Stopped 2026-08-27 — IQ3_XXS 2-D weights

HEAD is this branch’s tip. Worktree should be clean before the next resume.
No in-flight code.

## Shipped (use this)

Repo: https://github.com/mingley/llama-rust
Local: `~/dev/llama-rust-perf`

- `forbid(unsafe_code)`, no llama.cpp/FFI, `Cargo.lock` crate-only (no SIMD crates, no rayon).
- GGUF v3: F32, F16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, Q8_K, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS. Kernels read on-disk bytes (no private f32-scale copy).
- F16 is IEEE binary16 (`GGML_TYPE_F16` = 1). Writer-built tiny uses F16 for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `ggml_fp16_to_fp32` math. No tok/s.
- Q5_K is `GGML_TYPE_Q5_K` = 13 (176-byte `block_q5_K`). Writer-built tiny uses Q5_K for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_q5_K` walk (`d*sc*q5 - dmin*m`, `qh` 5th bit). No tok/s.
- IQ4_XS is `GGML_TYPE_IQ4_XS` = 23 (136-byte `block_iq4_xs`). First IQ* type that common OSS `*-IQ4_XS.gguf` files actually have. Writer-built tiny uses IQ4_XS for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_iq4_xs` walk (`d*(ls-32)*kvalues_iq4nl[q]`). No tok/s.
- IQ4_NL is `GGML_TYPE_IQ4_NL` = 20 (18-byte `block_iq4_nl`). Next IQ* type that common OSS `*-IQ4_NL.gguf` files actually have (bartowski / mradermacher standalone, and mixed IQ*_M tensors). Writer-built tiny uses IQ4_NL for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_iq4_nl` walk (`d * kvalues_iq4nl[q]`). No tok/s.
- IQ3_S is `GGML_TYPE_IQ3_S` = 21 (110-byte `block_iq3_s`). Next remaining IQ* type that common OSS `*-IQ3_S.gguf` files actually have (bartowski / mradermacher standalone, and the primary 2-D dtype in mixed `*-IQ3_M.gguf`). Writer-built tiny uses IQ3_S for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_iq3_s` walk (`d*(1+2*ls)*iq3s_grid[q]*sign`). No tok/s.
- IQ3_XXS is `GGML_TYPE_IQ3_XXS` = 18 (98-byte `block_iq3_xxs`). Next remaining IQ* type that common OSS `*-IQ3_XXS.gguf` files actually have (bartowski / mradermacher standalone, and IQ3_XXS tensors in mixed `*-IQ3_XS.gguf`). Writer-built tiny uses IQ3_XXS for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_iq3_xxs` walk (`d*(0.5+ls)*0.5*iq3xxs_grid[q]*ksigns`). No tok/s.
- Decode: RMSNorm, RoPE, GQA+KV, SwiGLU, lm_head, greedy sample by default.
- **Sampling.** Seedless greedy (`temperature <= 0`, argmax, first index on ties) is still the `infer` / `greedy_generate` path. `SampleParams` + `generate` add temperature, top-k, top-p, and unique-id repeat penalty (`logit > 0` then `/=`, else `*=`). Stochastic draws use SplitMix64 and require a seed. No CLI sampling flags.
- Prefill GEMM. Prompt tokens are one causal pass. A single token stays GEMV.
- Architectures: `llama`, `qwen2`, `mistral`, `phi3` `{arch}.*` KV.
- Q4_K_M shape that common OSS files actually have:
  - quantized `token_embd.weight` (Q4_K / Q5_K / Q6_K / IQ3_XXS / IQ3_S / IQ4_NL / IQ4_XS / F32) or F16
  - missing `{arch}.rope.dimension_count` derived from `embedding_length / head_count`
  - optional F32 `attn_{q,k,v}.bias`
  - `tokenizer.ggml.add_bos_token=false` honored
- Load/decode errors name tensor, ggml type id, and/or KV key.
- CLI: `gguf_gemv infer <path> [--prompt TEXT] [--n-predict N] [--n-ctx N]`. Seedless greedy. Defaults remain `ab` / 2 so the shipped two-run command still works.
- **Serving.** Local `gguf_gemv serve <path> [--n-predict N] [--n-ctx N] [--bind HOST:PORT]`. Std `TcpListener` on `127.0.0.1` (default `:8080`; `localhost` allowed). One HTTP/1.1 request at a time: `POST /generate` JSON `{"prompt"}` optional `n_predict` → `{"generated"}`. Seedless greedy (`greedy_generate_ctx`). Missing file and empty prompt fail cleanly. No batching, no multi-request, no OpenAI-compat, no tok/s. Not a production inference server. Kernel Integrity has not signed it.
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

Nothing. This slice is STATUS item 2 (IQ3_XXS). Metal-in-crate was skipped: this Linux VM cannot compile or run Metal.

## Still needed (production / researcher bar)

Ordered by how much they block “others can actually use this”:

1. **Metal-in-crate.** Owned MSL kernels exist as a sidecar. Decode still CPU.
2. **Dtypes / arches still rejected.** Remaining IQ* (IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ1_M), Gemma, MoE, vision, Qwen3, Llama4. 1-D F16 norms/bias still rejected. Tied `output.weight` (reuse `token_embd`) untested.
3. **KV cache** sized to prompt+predict is the default; `{arch}.context_length` is still unused. `--n-ctx` is an override only.
4. **crates.io** unpublished. Linux proof is GHA tiny/oracle tests only (2GB GGUF is gitignored).
5. **Chat template apply.** The Jinja string is read. Rendering it (and special-token split of `<|im_start|>` in the prompt) is not started. BPE has no Unicode regex pre-tokenizer.
6. **Serving beyond local.** Loopback one-request `serve` exists. No batching, no concurrent requests, no OpenAI-compat. Not a production inference server.

Non-goals that were explicitly parked: SIMD crates (`wide`/`pulp`/`std::simd`); matching llama.cpp tok/s; downloading HF checkpoints in CI; mmap (`unsafe` or a crate).

## Resume

```
cargo test --release --lib
cargo clippy --all-targets --all-features -- -D warnings
./target/release/gguf_gemv infer models/qwen2.5-3b-instruct-q4_k_m.gguf --prompt ab --n-predict 2
./target/release/gguf_gemv serve tiny-llama.gguf
```

Next code change should be item 1 (Metal-in-crate) on a machine that can compile Metal, or remaining item-2 dtypes (remaining IQ* first). Do not add crates.io runtime deps or `unsafe`. Do not start Metal-in-crate on Linux.
