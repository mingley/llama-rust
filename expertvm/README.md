# expertvm

Engine-agnostic **virtual memory for sparse MoE expert weights**.

The decoder (`llama-rust`) answers “what does this model mean?”
This crate answers “where are the weights for that math right now?”

It is the wedge from the research plan: make a 300B–1T sparse MoE behave
as if its experts live in one address space, while a policy keeps the
right ones in a bounded fast tier (HBM, or a simulated HBM).

## What this is not

- Not DeepEP (token → resident expert all-to-all)
- Not FlashInfer (fused kernels)
- Not Mooncake (distributed KV)
- Not “LRU between RAM and GPU” as the research claim — that is the
  **baseline**. The interesting policies are layer-ahead prefetch, a
  one-layer predictor, and an oracle upper bound.

## Pipeline

```text
llama-rust routers  →  ExpertAccess JSONL
                         │
                         ▼
                   expertvm analyze     locality, Zipf, predictability
                         │
                         ▼
                   expertvm replay      LRU / LFU / predictor / oracle
                         │
                         ▼
                   expertvm sim         gpu-sim wall time under a profile
```

## Library

```rust
use expertvm::{analyze, compare, replay, Policy, Trace};

let trace = Trace::parse(jsonl).expect("trace");
let stats = analyze(&trace);
let lru = replay(&trace, 8, Policy::Lru, 8);
let table = compare(&trace, 8, 8);
assert!(table.iter().any(|r| r.policy == Policy::Oracle && r.hits >= lru.hits));
let _ = stats;
```

`DirectStore` is the identity backend (every acquire hits, bytes unchanged).
`CachedStore` is a bounded LRU with **leases** that pin in-use experts so
they cannot be evicted. `sim_replay` runs the same policy through
[`gpu-sim`](../gpu-sim): H2D on miss, grouped GEMM on acquire, stream-ordered
free on eviction. Timing comes from a `HardwareProfile`, not from the policy.

## CLI

```text
expertvm analyze trace.jsonl
expertvm replay trace.jsonl --capacity 8
expertvm sim    trace.jsonl --capacity 8 --expert-bytes 188743680 --profile h100
```

Traces are produced by `gguf_gemv trace`:

```
gguf_gemv write-tiny-qwen3moe tiny-qwen3moe.gguf
gguf_gemv trace tiny-qwen3moe.gguf -p ab -n 8 --out trace.jsonl
expertvm replay trace.jsonl --capacity 2
```

```json
{"sequence":0,"token":0,"layer":0,"experts":[3,7]}
```

is a valid input. Hit-rate tables are **measured** from that file. Do not
paste fictional percentages into docs.

## Policies

| Name | Victim |
| --- | --- |
| `random` | deterministic LCG among residents |
| `lru` | least-recently used |
| `lfu` | least-frequently used (this replay only) |
| `layer-ahead` | prefer evicting keys not in the next `lookahead` acquires |
| `predictor` | copy-forward: keep `(layer+1, same expert)` from the last acquire |
| `oracle` | Belady furthest-next-use (upper bound, not a deployable policy) |

If the best non-oracle policy on a **real** model trace is ~random, stop.
Do not build CUDA on top of an 18% hit rate.

## Safety

Public residency/lease APIs are safe Rust. A later CUDA/RDMA backend may
use `unsafe` under a tiny audited boundary. The simulator backend needs
none.
