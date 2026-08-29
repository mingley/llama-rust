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
                   expertvm analyze     locality, Zipf, lookback-2 persist
                         │
                         ▼
                   expertvm place       striped / colocated / hot replicas
                         │
                         ▼
                   expertvm replay      LRU / LFU / predictor / oracle
                         │
                         ▼
                   expertvm sim         gpu-sim wall; --prefetch markov
                         │
                         ▼
                   expertvm bench       replay table + sim score
                         │
                         ▼
                   expertvm ep          static EP vs GPU0 LRU (8-GPU profile)
                         │
                         ▼
                   expertvm remote      GPU0 compute vs remote-home RDMA fetch
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
| `CachedStore` | Bounded LRU with **leases** so in-use experts cannot be evicted. `prefetch(keys)` skips unknown keys. `pin_hot` / `is_resident` / `is_leased` / `phase` / `take_victim`. CPU `ExpertPhase` is Cold / Resident / Leased (fault-in is instant). |
| `TieredStore` | Fast RAM LRU in front of slow RAM, a paging **file** (seek+read, not mmap), or synthetic bytes. Only `slots` [`ExpertParts`](crate::ExpertParts) live in the fast map. `WeightStorage::mmap` is parked. |
| `SimulatedGpuStore` | CachedStore + [`gpu-sim`](../gpu-sim). **Pinned** H2D on a copy stream onto the striped home (`expert_id % n_gpus`); GEMM waits that event. A host-pinned staging alloc is created at construction and does not charge HBM. Prefetch is H2D without GEMM (`ExpertPhase::Transferring` until the copy event completes). After a drain or `synchronize_stream` on compute, an idle stream captures a per-page GEMM graph and later acquires `launch_graph`. Lease of Transferring/Cold/Evicting is refused. `pin_hot` waits the copy, then NVLink-replicates to GPU1 when `n_gpus >= 2`. `migrate(key, dst)` D2D-moves onto `dst` (copy stream; dest compute waits the event; src HBM dropped). `score()` is wall/HBM/bytes/`energy_uj`/`ns_per_token`. |
| `LiveStore` | Enum over Direct / Cached / Tiered / Simulated. Decode attaches this. |

`sim_replay` runs a policy through gpu-sim: pinned H2D on miss, grouped GEMM on
acquire, stream-ordered free on eviction. Timing comes from a
`HardwareProfile`, not from the policy. The clock is sampled after each
token (`ttft_ns`, mean `itl_ns`); a batch of sequences at the same token is
one sample. `--seq-streams` maps `sequence % n_streams` onto CUDA streams so
those copies can overlap. `--max-batch N` admits N sequences per engine
iteration at a token (`0` = the whole token) and still samples TTFT once.
`--cuda-graphs` captures grouped expert GEMMs after `synchronize_stream` on
that stream and replays them (`graph_launch_ns` once per launch). `--plan-window
N` runs [`plan_window`](crate::plan_window) Stay vs Fetch before prefetch (Stay
does not evict a resident working set). Replay reports `prefetch_hits` /
`prefetch_waste`. Planner helpers: `copy_forward`,
`hot_keys`, `plan_window`, `plan_placement` (move weights vs dispatch
activations), `Markov` / `Prefetch` (lookback-2 `P(to|from, from_prev)` with
order-1 backoff). `colocated` keeps co-fired experts
on one GPU; `with_hot_replicas` copies hot keys to a second GPU.
`compare_ep` / `sim_static_ep` place each expert on
`home_gpu` (`expert_id % n_gpus`) with no eviction. LRU-on-GPU0 can
survive restricted HBM by evicting; static EP OOMs if a home GPU cannot
hold its working set. On `8xh100`, a wide token's H2Ds run on eight PCIe
roots and beat serial GPU0 copies of the same payload.
`sim_remote_home` keeps compute on GPU0. A miss does pinned H2D onto the
home GPU, then `plan_placement` (online reuse, no future leak) either
D2Ds the expert weights onto GPU0 or ships a small activation payload to
home and GEMMs there. Decode acquires then **leases** each routed expert
for the GEMV and releases before the next (so `slots < top-k` still
works). `SimulatedGpuStore` holds a host-pinned staging alloc that does
not count toward HBM. `HardwareProfile::restrict_hbm` is the knob. `topology_suite` /
`probe_topology` compare H2D and P2P costs across named meshes (PCIe P2P,
NVLink, bad NUMA, RDMA, asymmetric). `SimulatedGpuStore` can inject GPU
unavailable, copy-stream cancel, transfer delay, and next-H2D load
failure.

Decode identity: `Llama::expert_direct_store` + `KvCache::attach_expert_store`
must bit-match the blob GEMV path. Shared experts stay on the blob.

## CLI

```text
expertvm analyze  trace.jsonl
expertvm replay   trace.jsonl --capacity 8
expertvm sim      trace.jsonl --capacity 8 --expert-bytes 188743680 --profile h100 --prefetch both
expertvm sim      trace.jsonl --capacity 8 --profile h100 --prefetch markov --seq-streams
expertvm sim      trace.jsonl --capacity 8 --prefetch copy-forward --plan-window 8 --cuda-graphs
expertvm sim      trace.jsonl --capacity 8 --seq-streams --max-batch 2
expertvm place    trace.jsonl --gpus 8 --hot-pt 200
expertvm bench    trace.jsonl --capacity 8 --profile h100
expertvm bench    adversarial --tokens 64 --experts 16 --capacity 2 --profile cheap
expertvm workload thrash --tokens 64 --experts 16 --capacity 2
expertvm workload batch --tokens 32 --experts 16 --capacity 4
expertvm topology --bytes 1048576
expertvm ep       trace.jsonl --capacity 8 --expert-bytes 1048576 --profile 8xh100
expertvm ep       trace.jsonl --hbm-bytes 4096 --profile 8xh100
expertvm remote   trace.jsonl --expert-bytes 1048576 --profile 2node-rdma
expertvm remote   trace.jsonl --expert-bytes 1048576 --activation-bytes 128
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
{"sequence":0,"token":0,"layer":0,"experts":[3,7],"w":[500,500]}
```

are valid inputs (`w` is optional router mass in permille). Hit-rate tables are **measured** from that file. Do not
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
