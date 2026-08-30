# llama-rust

Verifiable GGUF v3 Llama-family **prompt → text** in Rust. No llama.cpp bind, no C GGML snapshot. Default `simd` is the only `unsafe`; `--no-default-features` restores `forbid(unsafe_code)`. SIMD row kernels cover F32, F16, Q4_0, Q4_K, Q5_0, Q5_1, Q6_K, and Q8_0 (AVX2+FMA+F16C / NEON); every other ggml dtype stays on the scalar kernels.

This is the **correctness laboratory** for the research stack in [PLAN.md](PLAN.md). It answers “what does this model mean, and is the math right?” against an independent oracle and llama.cpp greedy on real checkpoints. The other workspace crates answer “where are the weights, and what would that cost?”

| Crate | Job |
| --- | --- |
| `llama-rust` | Model semantics. Oracle + llama.cpp greedy identity. |
| [`expertvm`](expertvm/) | Expert residency / virtual memory (leases, `ExpertPhase`, prefetch, replication). |
| [`gpu-sim`](gpu-sim/) | Deterministic GPU-systems VM. Exact CUDA-like invariants (`GpuOp` DAG, stream-ordered vs host-sync malloc/free/memcpy, `cudaDeviceSynchronize`), calibrated timing. |
| [`infer-bench`](infer-bench/) | Serving-shaped measurement over traces. No invented `$/M tokens`. |

See [STATUS.md](STATUS.md) for what shipped. Full ChatGPT share extract: [docs/chatgpt-share-6a920fe1/](docs/chatgpt-share-6a920fe1/).

## Model / Session

```rust
use llama_rust::{CachedStore, Engine, EngineCfg, LiveStore, Model};

let model = Model::from_bytes(bytes)?;
let ids = model.encode("ab")?;
let mut sess = model.session(4096)?;
let direct = model.llama().expert_direct_store()?;
sess.attach_expert_store(LiveStore::Cached(CachedStore::new(direct, 8)?));
let logits = sess.prefill(&ids)?;
// Later request with a shared prefix: Session::prompt reuses KV for the LCP.
let logits = sess.prompt(&ids)?;
// Opt-in paged KV (vLLM-style blocks + intern). Logits bit-match dense.
let mut paged = model.session_paged(4096, 16)?;
let logits = paged.prefill(&ids)?;
// Two sequences sharing interned prefixes:
let pool = model.paged_pool(16, 256)?;
let mut a = model.session_on_pool(4096, &pool)?;
let mut b = model.session_on_pool(4096, &pool)?;
// Continuous batching (chunked prefill + join mid-flight + waiting queue).
// Prefill chunks, replay tokens, and decode tokens GEMM together on the pool.
// Routed experts that share an expert id GEMM together (one acquire each).
// Router logits are one GEMM of the batch; Engine pin_hot keeps last-used
// experts resident (`slots - 1`) and plan_placement either D2Ds multi-GPU
// pins onto GPU0 or leaves them on the striped home (place_hot is fail-loud).
// A full pool preempts another sequence (recompute + replay).
let mut eng = Engine::new(model.llama(), EngineCfg::tiny())?;
eng.attach_expert_store(LiveStore::Direct(model.llama().expert_direct_store()?));
eng.enable_moe_trace();
let s0 = eng.add(&ids, 8)?;
eng.step()?;
let s1 = eng.add(&ids, 8)?;
eng.run()?;
// eng.stats().gemm_peak is the widest cross-sequence GEMM (not wall-clock tok/s).
// With SimulatedGpuStore, eng.expert_store_score() includes ttft_ns / itl_ns.
```

Default decode stays on the GGUF blob (`expert_store == None`). Attaching DirectStore, CachedStore, TieredStore, or SimulatedGpuStore GEMVs routed expert copies; identity tests require bit-equal logits vs the blob path. Shared experts stay on the blob. `TieredStore::memory` / `on_path` keep only `slots` experts in fast RAM (`WeightStorage::mmap` is parked).

`cargo run -p llama-rust --example session` runs the writer-built Qwen3MoE tiny through CachedStore. `cargo run -p llama-rust --example engine` runs two sequences on `Engine` (chunked prefill, intern hits, recompute preemption, DirectStore on the batched GEMM). `gguf_gemv engine` is the same scheduler from the binary (`--expert-slots 0` for DirectStore, `--expert-slots N` for CachedStore, `--expert-sim` for SimulatedGpuStore + gpu-sim score, `--expert-8gpu` / `--expert-bytes N` for 8×H100 placement, `--decode-first` to hold leftover prefill while any live sequence is decoding, `--slo-reject` / `--ttft-slo-ns N` to drop waiters whose gpu-sim queue wait already meets the TTFT budget (`--expert-sim`), `--itl-slo-ns N` to count later-token ITL misses, `--cuda-graphs` / `--graph-update` / `--graph-set-params` / `--graph-clone` / `--graph-build` / `--graph-piecewise` / `--graph-mem` / `--graph-auto-free` / `--timing-events` for SimulatedGpuStore CUDA-graph knobs, `--mapped` / `--managed` / `--vmm` / `--vmm-page N` for miss-page placement, `--host-func` / `--blocking-streams` / `--sync-alloc` / `--mempool` / `--shareable` / `--pageable` / `--accessed-by` / `--legacy-null` / `--stream-priority` / `--seq-streams` / `--kv-sim` / `--kv-bytes N` / `--decode-priority` / `--cooperative` / `--pdl` / `--l2-persist` / `--cluster N` / `--preferred-cluster N` / `--cluster-spread` / `--max-shared` / `--non-portable-cluster` / `--sync-policy` / `--shared-mem` / `--portable-cluster` / `--optin-shared` / `--dynamic-shared` / `--portable-shared` / `--nvlink-util` / `--device-launch` / `--device-updatable` / `--multicast` / `--compute-slots N` / `--decode-sms N` for the rest of `GpuStoreCfg`, `--prefetch none|copy-forward|markov|both` / `--plan-window N` / `--plan-threshold N` for Stay vs Fetch (default `both` / ungated; `--prefetch none` is demand paging), `--trace-out FILE` for batched MoE JSONL, `--bench` / `--capacity N` for `expertvm::report` on those traces).

Loads mixed Q4_K_M-shaped dtypes (**F32**, **F16**, **BF16**, **Q2_K**, **Q3_K**, **Q4_1**, **Q4_K**, **Q5_0**, **Q5_1**, **Q5_K**, **Q6_K**, **IQ1_M**, **IQ1_S**, **IQ2_XXS**, **IQ2_XS**, **IQ2_S**, **IQ3_XXS**, **IQ3_S**, **IQ4_NL**, **IQ4_XS**, **MXFP4**, **NVFP4**, **Q1_0**, **Q2_0**, **Q8_1**, **TQ1_0**, **TQ2_0**, plus Q4_0/Q8_0/Q8_K) for `llama` / `qwen2` / `mistral` / `phi3` / `gemma` / `qwen3` / `llama4` / `qwen2moe` / `qwen3moe` / `qwen2vl` / `qwen3vl` / `qwen3next` / `qwen35` / `phi2`. Official Mixtral GGUF is `architecture=llama` with `n_expert>0` (official convert writes MixtralForCausalLM as `llama`; there is no `mixtral` architecture). F16 2-D weights (`token_embd`, `output`, attn/ffn) stay on-disk IEEE binary16; 1-D attn/ffn/`output_norm` and optional attn_q/k/v bias may be F16 (same `ggml_fp16_to_fp32` walk) or F32. BF16 2-D weights stay on-disk `ggml_bf16_t` bytes (`GGML_TYPE_BF16` = 30). Q2_K 2-D weights stay on-disk `block_q2_K` bytes (`GGML_TYPE_Q2_K` = 10). Q3_K 2-D weights stay on-disk `block_q3_K` bytes (`GGML_TYPE_Q3_K` = 11). Q4_1 2-D weights stay on-disk `block_q4_1` bytes (`GGML_TYPE_Q4_1` = 3). Q5_0 2-D weights stay on-disk `block_q5_0` bytes (`GGML_TYPE_Q5_0` = 6). Q5_1 2-D weights stay on-disk `block_q5_1` bytes (`GGML_TYPE_Q5_1` = 7). Q5_K 2-D weights stay on-disk `block_q5_K` bytes (`GGML_TYPE_Q5_K` = 13). MXFP4 2-D weights stay on-disk `block_mxfp4` bytes (`GGML_TYPE_MXFP4` = 39). NVFP4 2-D weights stay on-disk `block_nvfp4` bytes (`GGML_TYPE_NVFP4` = 40). Q1_0 2-D weights stay on-disk `block_q1_0` bytes (`GGML_TYPE_Q1_0` = 41). Q2_0 2-D weights stay on-disk `block_q2_0` bytes (`GGML_TYPE_Q2_0` = 42). Q8_1 2-D weights stay on-disk `block_q8_1` bytes (`GGML_TYPE_Q8_1` = 9). TQ1_0 2-D weights stay on-disk `block_tq1_0` bytes (`GGML_TYPE_TQ1_0` = 34). TQ2_0 2-D weights stay on-disk `block_tq2_0` bytes (`GGML_TYPE_TQ2_0` = 35). IQ1_M 2-D weights stay on-disk `block_iq1_m` bytes (`GGML_TYPE_IQ1_M` = 29). IQ1_S 2-D weights stay on-disk `block_iq1_s` bytes (`GGML_TYPE_IQ1_S` = 19). IQ2_XXS 2-D weights stay on-disk `block_iq2_xxs` bytes (`GGML_TYPE_IQ2_XXS` = 16). IQ2_XS 2-D weights stay on-disk `block_iq2_xs` bytes (`GGML_TYPE_IQ2_XS` = 17). IQ2_S 2-D weights stay on-disk `block_iq2_s` bytes (`GGML_TYPE_IQ2_S` = 22). IQ3_XXS 2-D weights stay on-disk `block_iq3_xxs` bytes (`GGML_TYPE_IQ3_XXS` = 18). IQ3_S 2-D weights stay on-disk `block_iq3_s` bytes (`GGML_TYPE_IQ3_S` = 21). IQ4_NL 2-D weights stay on-disk `block_iq4_nl` bytes (`GGML_TYPE_IQ4_NL` = 20). IQ4_XS 2-D weights stay on-disk `block_iq4_xs` bytes (`GGML_TYPE_IQ4_XS` = 23). Tied `output.weight` reuses the already-loaded `token_embd.weight` range when the tensor is absent (same on-disk bytes, no matrix clone). Quantized weights **and token embeddings** stay **on-disk bytes** in **one file blob**; `Llama` takes that blob and addresses tensors by range (no per-matrix clone, no mmap). Missing `{arch}.rope.dimension_count` is derived from embedding length / head count. Optional `attn_q`/`attn_k`/`attn_v` bias tensors are applied when present. `tokenizer.ggml.add_bos_token=false` is honored. Tokenizer: GPT-2 / Qwen pieces use bytes-to-unicode (`Ċ` is newline, `Ġ` is space); SentencePiece uses `▁` and `<0xHH>` byte-fallback; `token_id` is a `HashMap`. `tokenizer.chat_template` is read, not rendered. Decode is RMSNorm, RoPE, GQA + KV cache, SwiGLU (or Gemma GeGLU), lm_head. `gemma` scales embeds by `sqrt(n_embd)`. Official `qwen3` applies per-head QK-Norm (`attn_q_norm` / `attn_k_norm` RMSNorm on Q and K after projection, before RoPE) and keeps SwiGLU. Official `llama4` text applies iRoPE/NoPE, unweighted QK-Norm after RoPE, and expert FFN (sigmoid top-k + shared expert) on MoE layers. Official llama MoE (`architecture=llama` with `n_expert>0`) applies softmax then top-k, SwiGLU, and weights after the expert with `norm_w` clamp `2^-14`. Official `qwen2moe` applies softmax then top-k without `norm_w`, SwiGLU experts, and a shared expert gated by `silu(x)/x` on `ffn_gate_inp_shexp`. Official `qwen3moe` applies Qwen3 QK-Norm plus softmax then top-k with `norm_w` clamp `2^-14` and no shared expert. Official `qwen2vl` applies the Qwen2 language walk plus m-RoPE (`ggml_rope_multi` / `rope.dimension_sections`, text `n_pos_per_embd=4`). Official `qwen3vl` applies Qwen3 QK-Norm plus interleaved m-RoPE (`ggml_rope_multi` / `LLAMA_ROPE_TYPE_IMROPE`, required `rope.dimension_sections`, text `n_pos_per_embd=4`). Official `qwen3next` applies gated full attention (joint Q+gate, QK-Norm, sigmoid after attn), `post_attention_norm`, and MoE softmax then top-k with `norm_w` plus a shared expert gated by sigmoid. Official `qwen35` applies gated full attention (joint Q+gate, QK-Norm, interleaved m-RoPE / `LLAMA_ROPE_TYPE_IMROPE`, sigmoid after attn), `post_attention_norm`, and dense SwiGLU; linear-attn / gated-delta layers are refused. Official `phi2` applies LayerNorm, NEOX RoPE, Q-scale then attn scale `1.0`, and parallel GELU-seq FFN. `gemma2` and `mixtral` are rejected. Prompt prefill is one causal GEMM pass over the prompt tokens; generation after that is one-token GEMV + KV. Sampling default is seedless greedy (argmax). `SampleParams` adds temperature, top-k, top-p, and unique-id repeat penalty; `temperature > 0` needs a seed (SplitMix64). `serve` is a std-only loopback HTTP/1.1 listener: one request at a time, `POST /generate` JSON, seedless greedy. Not a production inference server. Load/decode errors name the tensor, ggml type id, and/or KV key. ggml-removed type ids (36–38) fail as removed, not as a missing dequant.

Not [onehr/llama-rs](https://github.com/onehr/llama-rs) / [rustformers/llm]. That wrapped frozen GGML. This is GGUF-native.

```
cargo test --release --lib
cargo clippy --all-targets --all-features -- -D warnings
cargo build --release --bin gguf_gemv
./target/release/gguf_gemv write-tiny tiny-llama.gguf
./target/release/gguf_gemv infer tiny-llama.gguf
./target/release/gguf_gemv infer tiny-llama.gguf --prompt a --n-predict 4
./target/release/gguf_gemv serve tiny-llama.gguf
./target/release/gguf_gemv serve tiny-llama.gguf --engine --prefill-chunk 1
./target/release/gguf_gemv serve tiny-llama.gguf --engine --decode-first
./target/release/gguf_gemv engine tiny-llama.gguf -p a -p b --kv-page 2 --decode-first
./target/release/gguf_gemv engine tiny-qwen3moe.gguf -p a -p b --kv-page 2 --expert-slots 0
./target/release/gguf_gemv engine tiny-qwen3moe.gguf -p a -p b --kv-page 2 --expert-sim --cuda-graphs --graph-update
./target/release/gguf_gemv engine tiny-qwen3moe.gguf -p a -p b --kv-page 2 --expert-sim --graph-set-params
./target/release/gguf_gemv engine tiny-qwen3moe.gguf -p a -p b --kv-page 2 --expert-sim --host-func --stream-priority
./target/release/gguf_gemv engine tiny-qwen3moe-2layer.gguf -p a -p b --kv-page 2 --expert-sim --seq-streams --expert-bytes 33554432
./target/release/gguf_gemv engine tiny-qwen3moe-2layer.gguf -p a -p b --kv-page 2 --expert-sim --kv-sim --kv-bytes 1048576
./target/release/gguf_gemv engine tiny-qwen3moe-2layer.gguf -p a -p b --kv-page 2 --expert-slots 8 --prefetch copy-forward --plan-window 8
./target/release/gguf_gemv engine tiny-qwen3moe.gguf -p a -p b --kv-page 2 --trace-out /tmp/engine.jsonl
./target/release/gguf_gemv engine tiny-qwen3moe.gguf -p a -p b --kv-page 2 --bench --capacity 2
./target/release/gguf_gemv write-tiny-qwen2 tiny-qwen2.gguf
./target/release/gguf_gemv infer tiny-qwen2.gguf -p ab -n 2
./target/release/gguf_gemv write-tiny-gemma tiny-gemma.gguf
./target/release/gguf_gemv infer tiny-gemma.gguf -p ab -n 2
./target/release/gguf_gemv write-tiny-qwen3 tiny-qwen3.gguf
./target/release/gguf_gemv infer tiny-qwen3.gguf -p ab -n 2
./target/release/gguf_gemv write-tiny-llama4 tiny-llama4.gguf
./target/release/gguf_gemv infer tiny-llama4.gguf -p ab -n 2
./target/release/gguf_gemv write-tiny-llama-moe tiny-llama-moe.gguf
./target/release/gguf_gemv infer tiny-llama-moe.gguf -p ab -n 2
./target/release/gguf_gemv write-tiny-qwen2moe tiny-qwen2moe.gguf
./target/release/gguf_gemv infer tiny-qwen2moe.gguf -p ab -n 2
./target/release/gguf_gemv write-tiny-qwen3moe tiny-qwen3moe.gguf
./target/release/gguf_gemv infer tiny-qwen3moe.gguf -p ab -n 2
./target/release/gguf_gemv write-tiny-qwen3moe-2layer tiny-qwen3moe-2layer.gguf
./target/release/gguf_gemv infer tiny-qwen3moe-2layer.gguf -p ab -n 2
./target/release/gguf_gemv trace tiny-qwen3moe.gguf -p ab -n 8 --out tests/traces/tiny-qwen3moe.jsonl --capacity 2
./target/release/expertvm replay tests/traces/tiny-qwen3moe.jsonl --capacity 2
./target/release/expertvm replay tests/traces/cycling.jsonl --capacity 2
./target/release/expertvm bench adversarial --capacity 2 --profile cheap
./target/release/infer-bench adversarial --capacity 2 --profile cheap
./target/release/infer-bench workload batch-1 --tokens 32
./target/release/infer-bench workload batch-128 --tokens 8
./target/release/infer-bench trace tests/traces/tiny-qwen3moe-2layer.jsonl --capacity 4
./target/release/infer-bench schedule tests/traces/tiny-qwen3moe-2layer.jsonl --capacity 4
./target/release/infer-bench trace tests/traces/cycling.jsonl --capacity 2
./target/release/infer-bench remote tests/traces/cycling.jsonl --expert-bytes 1048576
./target/release/expertvm topology --bytes 1048576
./target/release/expertvm remote tests/traces/cycling.jsonl --expert-bytes 1048576
./target/release/expertvm schedule tests/traces/tiny-qwen3moe-2layer.jsonl --capacity 4 --prefetch copy-forward
./target/release/expertvm schedule tests/traces/cycling.jsonl --capacity 2 --max-batch 1 --interarrival-ns 1000000
./target/release/gpu-profile probe bad-numa
cargo run -p llama-rust --example session
cargo run -p llama-rust --example engine
./target/release/gguf_gemv write-tiny-qwen2vl tiny-qwen2vl.gguf
./target/release/gguf_gemv infer tiny-qwen2vl.gguf -p ab -n 2
./target/release/gguf_gemv write-tiny-qwen3vl tiny-qwen3vl.gguf
./target/release/gguf_gemv infer tiny-qwen3vl.gguf -p ab -n 2
./target/release/gguf_gemv write-tiny-qwen3next tiny-qwen3next.gguf
./target/release/gguf_gemv infer tiny-qwen3next.gguf -p ab -n 2
./target/release/gguf_gemv write-tiny-qwen35 tiny-qwen35.gguf
./target/release/gguf_gemv infer tiny-qwen35.gguf -p ab -n 2
./target/release/gguf_gemv write-tiny-phi2 tiny-phi2.gguf
./target/release/gguf_gemv infer tiny-phi2.gguf -p ab -n 2
./target/release/gguf_gemv write tiny.gguf
./target/release/gguf_gemv gemv tiny.gguf
```

`infer` is seedless greedy. `--prompt` / `-p` (default `ab`), `--n-predict` / `-n` (default `2`), optional `--n-ctx` (KV capacity; default prompt + `n_predict` + 1). Temperature / top-k / top-p / repeat penalty live on `SampleParams` / `generate`; the CLI path is unchanged.

`trace` is the same greedy path plus an opt-in MoE access log (`ExpertAccess` JSONL). `--out FILE` is required. After decode it prints the `expertvm replay` table (`--capacity`, default 8). Tracing does not change generated tokens. Workspace crates: [`expertvm/`](expertvm/), [`gpu-sim/`](gpu-sim/), [`infer-bench/`](infer-bench/). Plan: [`PLAN.md`](PLAN.md).

`serve` binds `127.0.0.1:8080` (override with `--bind HOST:PORT`; host must be `127.0.0.1` or `localhost`). Default: one HTTP/1.1 request at a time, `POST /generate` with `{"prompt":"..."}` and optional `n_predict`, response `{"generated":"...","prefix_hit":N,"page_hits":N}`. HTTP/1.1 keep-alive is on unless the client sends `Connection: close`. The listener keeps one KV cache and reuses a matching token prefix (`Llama::prompt` / vLLM Automatic Prefix Caching on this engine). `--n-ctx` sizes that cache. `--kv-page N` stores KV in interned blocks of `N` tokens (`Llama::new_paged_cache`); logits match the dense path. `--engine` admits concurrent `POST /generate` onto one `Engine` so prefills GEMM together (`--max-seqs`, default 4; `--n-ctx` default 64; `--kv-page` default 16). `--prefill-chunk N` interleaves long prefills with decode. `--decode-first` holds leftover prefill while any live sequence is already decoding. `--slo-reject` / `--ttft-slo-ns` drop a waiter whose gpu-sim queue wait meets the TTFT budget (`--expert-sim`). `--itl-slo-ns` counts later-token ITL misses (does not drop). `--cuda-graphs` / `--graph-update` / `--graph-clone` / `--graph-build` / `--graph-piecewise` / `--graph-mem` / `--graph-auto-free` / `--timing-events` are the SimulatedGpuStore CUDA-graph knobs. `--mapped` / `--managed` / `--vmm` / `--vmm-page N` choose miss-page placement (default pinned). `--host-func` / `--blocking-streams` / `--sync-alloc` / `--mempool` / `--shareable` / `--pageable` / `--accessed-by` / `--legacy-null` / `--stream-priority` / `--seq-streams` / `--kv-sim` / `--kv-bytes N` / `--decode-priority` / `--cooperative` / `--pdl` / `--l2-persist` / `--cluster N` / `--preferred-cluster N` / `--cluster-spread` / `--max-shared` / `--non-portable-cluster` / `--sync-policy` / `--shared-mem` / `--portable-cluster` / `--optin-shared` / `--dynamic-shared` / `--portable-shared` / `--nvlink-util` / `--device-launch` / `--device-updatable` / `--compute-slots N` / `--decode-sms N` are the remaining `GpuStoreCfg` knobs. `--prefetch` / `--plan-window` / `--plan-threshold` are the same Stay vs Fetch predictor as `gguf_gemv engine` (`--engine`; default `both` / ungated). `--trace-out FILE` appends batched MoE JSONL. `"stream": true` is chunked NDJSON token lines then a final `generated` object (`--engine` only; default serve ignores it). `--expert-slots` / `--expert-sim` park DirectStore, CachedStore, or SimulatedGpuStore on that Engine. Empty prompt and a missing GGUF file fail cleanly. No OpenAI-compat, no tok/s. Kernel Integrity has not signed it.

Workspace lints deny unwrap/panic/indexing/wrap-casts/`std::fs::{read,write}`. File I/O is `File` + `Read`/`Write`.

Tests write GGUFs (F32 + F16 + BF16 + Q6_K + Q5_K + Q5_1 + Q5_0 + Q4_K + Q4_1 + Q3_K + Q2_K + IQ1_M + IQ1_S + IQ2_XXS + IQ2_XS + IQ2_S + IQ3_XXS + IQ3_S + IQ4_NL + IQ4_XS + MXFP4 + NVFP4 + Q1_0 + Q2_0 + Q8_1 + TQ1_0 + TQ2_0, quantized embeddings, QKV bias, F16 1-D norms/bias, missing rope dim, mistral/phi3/gemma/qwen3/llama4/qwen2moe/qwen3moe/qwen2vl/qwen3vl/qwen3next/qwen35/phi2 prefixes, official llama MoE `n_expert>0` on `llama.*` KV, official Qwen2MoE `qwen2moe.*` KV, official Qwen3MoE `qwen3moe.*` KV, official Qwen2VL `qwen2vl.*` KV plus `rope.dimension_sections`, official Qwen3VL `qwen3vl.*` KV plus QK-Norm and `rope.dimension_sections`, official Qwen3Next `qwen3next.*` KV plus gated Q, `post_attention_norm`, MoE `norm_w` and shared expert, official Qwen35 `qwen35.*` KV plus gated Q, IMROPE, `post_attention_norm`, and dense SwiGLU, official Phi2 `phi2.*` KV plus LayerNorm, NEOX RoPE, Q-scale, and parallel GELU-seq FFN, tied omit of `output.weight`), load them, compare logits to an independent scalar of the same ggml/Llama-family math on those bytes (Gemma: embed `sqrt(n_embd)` + `ggml_gelu`; Qwen3: per-head QK-Norm; Llama4: iRoPE/NoPE + unweighted QK-Norm after RoPE + expert FFN; official llama MoE: softmax then top-k, weights after SwiGLU with `norm_w` clamp `2^-14`; official Qwen2MoE: softmax then top-k without `norm_w`, shared expert gated by `silu(x)/x`; official Qwen3MoE: Qwen3 QK-Norm plus softmax then top-k with `norm_w` clamp `2^-14`, no shared expert; official Qwen2VL: Qwen2 plus `ggml_rope_multi` m-RoPE; official Qwen3VL: Qwen3 QK-Norm plus `ggml_rope_multi` IMROPE; official Qwen3Next: gated full attention + `post_attention_norm` + MoE `norm_w` + sigmoid-gated shared expert; official Qwen35: gated full attention + IMROPE + `post_attention_norm` + dense SwiGLU; official Phi2: LayerNorm + NEOX RoPE + Q-scale + parallel GELU-seq) (including multi-token prefill, prefix KV reuse via `Llama::prompt`, paged KV intern via `Llama::new_paged_cache` / `Model::session_paged`, F16 `ggml_fp16_to_fp32` for 2-D weights and 1-D norms/bias, BF16 `dequantize_row_bf16`, Q2_K `dequantize_row_q2_K`, Q3_K `dequantize_row_q3_K`, Q4_1 `dequantize_row_q4_1`, Q5_0 `dequantize_row_q5_0`, Q5_1 `dequantize_row_q5_1`, Q5_K `dequantize_row_q5_K`, IQ1_M `dequantize_row_iq1_m`, IQ1_S `dequantize_row_iq1_s`, IQ2_XXS `dequantize_row_iq2_xxs`, IQ2_XS `dequantize_row_iq2_xs`, IQ2_S `dequantize_row_iq2_s`, IQ3_XXS `dequantize_row_iq3_xxs`, IQ3_S `dequantize_row_iq3_s`, IQ4_NL `dequantize_row_iq4_nl`, IQ4_XS `dequantize_row_iq4_xs`, MXFP4 `dequantize_row_mxfp4`, NVFP4 `dequantize_row_nvfp4`, Q1_0 `dequantize_row_q1_0`, Q2_0 `dequantize_row_q2_0`, Q8_1 `q*d`, TQ1_0 `dequantize_row_tq1_0`, and TQ2_0 `dequantize_row_tq2_0`), then encode/greedy/decode. Sampler tests compare temperature / top-k / top-p / repeat penalty to an independent candidate-list oracle of the same math, plus SplitMix64 published outputs. Tokenizer tests cover GPT-2 `Ċ`/`Ġ` piece decode, `<0xHH>` byte-fallback, SentencePiece `▁`, HashMap `token_id`, and the writer-built tiny merge. Serve tests cover CLI parse, loopback bind, HTTP/1.1 framing, JSON request/response shape, empty prompt, tiny-model `POST /generate` vs `greedy_generate_ctx`, persistent-cache `prefix_hit`, `--kv-page` identity, `--engine` concurrent posts sharing a prefill GEMM, DirectStore MoE acquires, intern `page_hits` across sequential Engine HTTP prompts, `--prefill-chunk`, `--trace-out` MoE JSONL, and `"stream": true` chunked NDJSON (no 2GB GGUF).

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

`Ċ` is the GPT-2 bytes-to-unicode mapping of newline. Decode now emits `\n` for that piece (the two-run string was two newlines printed raw). `#![forbid(unsafe_code)]` without `simd`. No FFI. Workspace members: `llama-rust`, `gpu-sim`, `expertvm`, `infer-bench` (path deps only).

Q8_0 GEMV (`write`+`gemv`, M=K=4096, GGUF-byte blocks). Two-run `y0=` match across Rust CPU and owned Metal. No crates.io deps; Metal is a measurement binary (`q8_gemv.metal` + `q8_gemv_mtl.m`), not linked.

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

This crate is `forbid(unsafe_code)` without `simd`, and the lockfile has no crates.io packages, so those Linux jobs do not need llama.cpp, Metal, or crates.io SIMD packages.
