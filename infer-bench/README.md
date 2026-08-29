# infer-bench

Serving-shaped measurement over MoE traces. Dual scores:

1. **Semantic** — `Ok` vs `gpu-sim` illegal GPU state (ordering, residency, OOM).
2. **Performance** — virtual `wall_ns`, HBM peak, bytes moved, optional `ns_per_token`.

There is **no invented `$/M tokens`**. Dollars need a real electricity/rental
price; this crate refuses to hallucinate one.

```
infer-bench adversarial --capacity 2 --tokens 64 --experts 16 --profile cheap
infer-bench trace tests/traces/cycling.jsonl --capacity 2 --profile h100
infer-bench workload thrash --capacity 2
infer-bench workload batch --capacity 4
infer-bench workload prefill-heavy --tokens 16
```

Same numbers as `expertvm bench` / `expertvm workload`. Timing comes from a
[`HardwareProfile`](../gpu-sim), not from policy code.
