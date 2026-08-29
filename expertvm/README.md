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
                         │
                         ▼
                   expertvm bench       replay table + sim score
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

One cache slot is one routed expert: [`ExpertParts`](crate::ExpertParts)
holds `gate` + `up` + `down` bytes.

| Store | Meaning |
| --- | --- |
| `DirectStore` | Identity catalog. Every acquire hits. Bytes unchanged. |
| `CachedStore` | Bounded LRU with **leases** so in-use experts cannot be evicted. `prefetch(keys)` skips unknown keys. `pin_hot` / `is_resident` / `take_victim`. |
| `TieredStore` | Fast RAM LRU in front of slow RAM, a paging **file** (seek+read, not mmap), or synthetic bytes. Only `slots` [`ExpertParts`](crate::ExpertParts) live in the fast map. `WeightStorage::mmap` is parked. |
| `SimulatedGpuStore` | CachedStore + [`gpu-sim`](../gpu-sim). H2D on a copy stream, GEMM waits that event. Prefetch is H2D without GEMM. `pin_hot` NVLink-replicates to GPU1 when `n_gpus >= 2`. `score()` is wall/HBM/bytes/`energy_uj`/`ns_per_token`. |
| `LiveStore` | Enum over Direct / Cached / Tiered / Simulated. Decode attaches this. |

`sim_replay` runs a policy through gpu-sim: H2D on miss, grouped GEMM on
acquire, stream-ordered free on eviction. Timing comes from a
`HardwareProfile`, not from the policy. The clock is sampled after each
token (`ttft_ns`, mean `itl_ns`). Planner helpers: `copy_forward`,
`hot_keys`, `plan_window`, `plan_placement` (move weights vs dispatch
activations). `topology_suite` / `probe_topology` compare
H2D and P2P costs across named meshes (PCIe P2P, NVLink, bad NUMA, RDMA,
asymmetric). `SimulatedGpuStore` can inject GPU unavailable, copy-stream
cancel, transfer delay, and next-H2D load failure.

Decode identity: `Llama::expert_direct_store` + `KvCache::attach_expert_store`
must bit-match the blob GEMV path. Shared experts stay on the blob.

## CLI

```text
expertvm analyze  trace.jsonl
expertvm replay   trace.jsonl --capacity 8
expertvm sim      trace.jsonl --capacity 8 --expert-bytes 188743680 --profile h100
expertvm bench    trace.jsonl --capacity 8 --profile h100
expertvm bench    adversarial --tokens 64 --experts 16 --capacity 2 --profile cheap
expertvm workload thrash --tokens 64 --experts 16 --capacity 2
expertvm workload batch --tokens 32 --experts 16 --capacity 4
expertvm topology --bytes 1048576
gpu-profile probe 2xh100-pcie --bytes 1048576
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
