# infer-bench

Serving-shaped measurement over MoE traces. Dual scores:

1. **Semantic** — `Ok` vs `gpu-sim` illegal GPU state (ordering, residency, OOM).
2. **Performance** — virtual `wall_ns`, HBM peak, bytes moved, `energy_uj`
   (profile TDP × wall), optional `ns_per_token`, `ttft_ns`, `itl_ns`.

There is **no invented `$/M tokens`**. Dollars need a real electricity/rental
price; this crate refuses to hallucinate one.

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
infer-bench schedule tests/traces/tiny-qwen3moe-2layer.jsonl --capacity 2 --prefill-chunk 1 --decode-priority --compute-slots 2
```

Same numbers as `expertvm bench` / `expertvm workload`. Timing comes from a
[`HardwareProfile`](../gpu-sim), not from policy code.
