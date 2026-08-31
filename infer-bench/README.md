# infer-bench

Serving-shaped measurement over MoE traces. Dual scores:

1. **Semantic** — `Ok` vs `gpu-sim` illegal GPU state (ordering, residency, OOM).
2. **Performance** — virtual `wall_ns`, HBM peak, bytes moved, `energy_uj`
   (profile TDP × wall), optional `ns_per_token`, `ttft_ns`, `itl_ns`,
   optional `usd_micros_per_m_tokens` when the profile sets
   `rent_usd_micros_per_hour` (example list price × wall / tokens). Example
   profiles leave rent at `0`.

```
infer-bench adversarial --capacity 2 --tokens 64 --experts 16 --profile cheap
infer-bench trace tests/traces/cycling.jsonl --capacity 2 --profile h100
infer-bench workload thrash --capacity 2
infer-bench workload batch --capacity 4
infer-bench workload batch-1 --tokens 32
infer-bench workload batch-128 --tokens 8
infer-bench workload prefill-heavy --tokens 16
infer-bench workload prefill-batch --tokens 8
infer-bench workload shared-prefix --tokens 8
infer-bench topology --bytes 1048576
infer-bench remote tests/traces/cycling.jsonl --expert-bytes 1048576
infer-bench remote tests/traces/cycling.jsonl --expert-bytes 1048576 --activation-bytes 128
infer-bench schedule tests/traces/tiny-qwen3moe-2layer.jsonl --capacity 4
infer-bench schedule tests/traces/cycling.jsonl --capacity 2 --prefill-chunk 1 --decode-first
infer-bench schedule tests/traces/cycling.jsonl --capacity 2 --max-batch 1 --ttft-slo-ns 1 --slo-reject
infer-bench schedule tests/traces/cycling.jsonl --capacity 2 --max-batch 1 --prefix-cache
infer-bench schedule tests/traces/cycling.jsonl --capacity 8 --place striped --profile 8xh100
infer-bench schedule tests/traces/cycling.jsonl --capacity 8 --place replicas --profile 8xh100
infer-bench schedule tests/traces/cycling.jsonl --capacity 8 --place remote --profile 2node-rdma --expert-bytes 1048576
infer-bench schedule tests/traces/tiny-qwen3moe-2layer.jsonl --capacity 2 --compute-slots 2 --decode-sms 250
infer-bench schedule tests/traces/tiny-qwen3moe-2layer.jsonl --capacity 2 --compute-slots 2 --pdl
infer-bench schedule tests/traces/tiny-qwen3moe-2layer.jsonl --capacity 2 --l2-persist
infer-bench schedule tests/traces/tiny-qwen3moe-2layer.jsonl --capacity 2 --seq-streams --compute-slots 2 --cluster 2
infer-bench schedule tests/traces/tiny-qwen3moe-2layer.jsonl --capacity 2 --seq-streams --compute-slots 4 --cluster 2 --preferred-cluster 4
infer-bench schedule tests/traces/tiny-qwen3moe-2layer.jsonl --capacity 2 --seq-streams --compute-slots 4 --cluster 2 --cluster-spread
infer-bench schedule tests/traces/tiny-qwen3moe-2layer.jsonl --capacity 2 --seq-streams --compute-slots 2 --max-shared
infer-bench schedule tests/traces/tiny-qwen3moe-2layer.jsonl --capacity 2 --seq-streams --compute-slots 2 --func-max-shared
infer-bench schedule tests/traces/tiny-qwen3moe-2layer.jsonl --capacity 2 --cluster 2 --non-portable-cluster
infer-bench schedule tests/traces/tiny-qwen3moe-2layer.jsonl --capacity 2 --decode-priority --sync-policy blocking
infer-bench schedule tests/traces/tiny-qwen3moe-2layer.jsonl --capacity 2 --shared-mem eight
infer-bench schedule tests/traces/tiny-qwen3moe-2layer.jsonl --capacity 2 --cluster 16 --portable-cluster non-portable
infer-bench schedule tests/traces/tiny-qwen3moe-2layer.jsonl --capacity 2 --dynamic-shared 65536 --portable-shared non-portable
infer-bench schedule tests/traces/tiny-qwen3moe-2layer.jsonl --capacity 2 --prefill-chunk 1 --decode-priority --compute-slots 2
```

Same numbers as `expertvm bench` / `expertvm workload`. Timing comes from a
[`HardwareProfile`](../gpu-sim), not from policy code.

`gguf_gemv engine --bench` prints the same `expertvm::report` on the
Engine's batched MoE traces (policy table; with `--expert-sim`, the same
sim A/B lines). That path lives in llama-rust so infer-bench does not
depend on the decoder.
