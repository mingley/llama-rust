# STATUS

Live research plan: [PLAN.md](PLAN.md).
Visible five-turn extract: [docs/chatgpt-share-6a920fe1.md](docs/chatgpt-share-6a920fe1.md).
Complete share-API extract: [docs/chatgpt-share-6a920fe1/](docs/chatgpt-share-6a920fe1/).
Work lands on `main`. No PRs.

## Shipped 2026-08-29 — sparse VMM maps charge only the mapped span

`Sim::va_map_range` / `va_unmap_range` / `vmm_mapped_bytes` map physical
pages into a reserved VA (vLLM KV-block analog). Overlap is `already
mapped`; a hole is not kernel-resident; H2D larger than the mapped span
is `NotResident`. `va_map` is still the whole VA. Dual score still has
no `$/M tokens`.

## Shipped 2026-08-29 — cudaLaunchHostFunc is host work, not a kernel

`Sim::host_func` is `cudaLaunchHostFunc`: stream-ordered, billed at
`GpuProfile::host_func_ns`, and it does not occupy compute or copy
engines so another stream can GEMM at the same virtual time. Graphs may
record it. `expertvm sim --host-func` / `expertvm bench` `sim-hostfn`
enqueue one callback after each event's GEMMs (CPU scheduler roundtrip).
Hits/misses unchanged. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — VMM VA pool remaps instead of re-reserving

`Sim::va_acquire` / `va_release` / `vmm_idle_len` keep unmapped VAs.
A later acquire of the same size remaps that pointer (map overhead only).
A different size does not share the pool. Map OOM parks the reserved VA.
`va_free` drops an idle entry. Capture still refuses reserve/map.
`expertvm sim --vmm` acquires on miss and releases on evict (no
`va_free` per miss). Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — host pin budget is mlock, not unlimited

`HardwareProfile::host_pin_bytes` / `restrict_pin` cap `cudaMallocHost`
and `cudaHostRegister` (`SimError::PinOom`). Pageable `alloc_host` does
not charge; register does. Example default is `u64::MAX`. Mapped expert
replay OOMs when slots × expert bytes exceed the pin cap.
`SimulatedGpuStore` still pins without a tight cap. Dual score still has
no `$/M tokens`.

## Shipped 2026-08-29 — CUDA VMM keeps a VA while HBM is mapped

`Sim::va_reserve` / `va_map` / `va_unmap` / `va_free` are
`cuMemAddressReserve` / `cuMemMap` / `cuMemUnmap` / `cuMemAddressFree`.
Reserve does not charge HBM; map does; unmap refunds and the pointer
stays so a later map can reuse it. Sparse sub-range maps are
`va_map_range` / `va_unmap_range`. Capture refuses reserve/map/unmap/free.
`expertvm sim --vmm` / `expertvm bench` `sim-vmm`. `SimulatedGpuStore`
stays on pinned H2D. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — cudaMallocManaged migrates, it does not replicate

`Sim::alloc_managed` / `prefetch` / `prefetch_host` are `cudaMallocManaged`
and `cudaMemPrefetchAsync` (GPU or `cudaCpuDeviceId`). Alloc does not
charge HBM; first-touch or prefetch migrates the page and bills PCIe /
NVLink. A second GPU prefetch **moves** the unique location (not a
replica). `cudaMalloc` of the remaining HBM can OOM a later prefetch
until that malloc is freed. Capture refuses `alloc_managed`; a graph
must record prefetch before the kernel. `expertvm sim --managed` /
`expertvm bench` `sim-managed`. `SimulatedGpuStore` stays on pinned H2D.
Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — mapped host is the no-H2D expert path

`Sim::alloc_host` / `host_register` / `host_register_mapped` /
`alloc_host_mapped` are pageable malloc, `cudaHostRegister`,
`cudaHostRegisterMapped`, and `cudaHostAllocMapped`. A mapped pointer is
kernel-readable over host PCIe with no device copy and no HBM charge.
Unregister is only for registered ids (`cudaMallocHost` still uses
`free_host_pinned`). Capture refuses host alloc/register.
`expertvm sim --mapped` / `expertvm bench` `sim-mapped` skip H2D.
`SimulatedGpuStore` stays on pinned H2D. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — CUDA memory pools hold unused HBM until trim

`Sim::create_pool` / `alloc_from_pool` / `set_pool_release_threshold` /
`pool_trim_to` are `cudaMemPoolCreate` / `cudaMallocFromPoolAsync` /
`cudaMemPoolAttrReleaseThreshold` / `cudaMemPoolTrimTo`. `alloc` uses the
device default pool (threshold `0`: free returns HBM). `u64::MAX` holds
unused bytes in `cudaMemGetInfo` used so `malloc` can OOM until trim.
Reuse of cached bytes pays `pool_reuse_ns`, not `alloc_overhead_ns`.
Capture refuses pool create/trim/set-attribute. `expertvm sim --mempool`
and `expertvm bench` `sim-pool` raise the default-pool threshold.
`SimulatedGpuStore` stays on threshold `0`. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — pageable `cudaMemcpyAsync` waits the stream

`memcpy_host_to_device` / `memcpy_device_to_host` (`Place::Host`) are
host-synchronous: the call does not return until that stream finishes
the copy, matching CUDA's pinned staging bounce. Pinned DMA
(`memcpy_pinned_to_device`) stays stream-ordered so two streams can
share PCIe. Capture refuses pageable copies. `SimulatedGpuStore` still
pins. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — `--sync-alloc` measures naive cudaMalloc

`SimCfg::sync_alloc` / `expertvm sim --sync-alloc` uses `malloc` /
`memcpy_sync` / `free_sync` on every miss so a two-stream batch cannot
overlap H2D. Default `sim`/`schedule` and `SimulatedGpuStore` stay on
`cudaMallocAsync`. `expertvm bench` prints `sim-async` vs `sim-malloc`.
Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — host-sync `cudaMalloc` / `cudaFree` / `cudaMemcpy`

`Sim::alloc` / `free` / `memcpy` stay stream-ordered (`cudaMallocAsync` /
`cudaFreeAsync` / `cudaMemcpyAsync`). `Sim::malloc` / `free_sync` /
`memcpy_sync` are the host-synchronous counterparts: `malloc` waits that
GPU (`synchronize_device` = `cudaDeviceSynchronize`) then the pointer is
usable and OOM is at the call; `free_sync` waits every GPU that holds the
id; `memcpy_sync` waits that stream. Capture refuses all four plus
`synchronize_device`. Default `sim_replay` / `SimulatedGpuStore` keep
using `alloc` so a miss does not device-sync (`--sync-alloc` opts into
the naive path). Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — remote prefetch fills `RemotePage` only

`schedule_remote --prefetch copy-forward|markov|both` H2Ds predicted
experts onto the home GPU (and D2Ds weights when `plan_placement` says
move) without running GEMM and without inserting a local `PageHandle`.
Demand then `remote_hit`s the filled page. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — prefix cache on `expertvm schedule`

Optional JSONL `"p"` is a content-addressed hash of the token ids in the
prefix (`prefix_hash`), not a prompt-class label. Decode emits it from
the actual ids. `--prefix-cache` skips GPU work for a token whose hash
already completed on another sequence; in-flight layers of the computing
sequence still run, and a hit consumes the whole remaining token (not
one prefill chunk). Workload `shared-prefix` is four sequences with the
same token-0 hash and diverging decode. `expertvm bench` prints
`schedule-prefix` when a multi-sequence trace has `"p"`. Dual score still
has no `$/M tokens`.

## Shipped 2026-08-29 — HBM caps beat loose `--capacity`

`restrict_hbm` / profile `hbm_bytes` is the real page budget. If `--capacity`
is larger than pages that fit, `schedule_placed` and `schedule_remote`
still evict so the next alloc cannot OOM. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — replica HBM is dest-capacity

Hot replicas occupy the destination GPU's `--capacity` slots. Evicting a
home page stream-orders a free of every replica; a dest miss frees only
that replica so the home copy can stay. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — per-home LRU inside `schedule`

`--capacity` is slots on the expert's home GPU, not one cluster-wide
walker. A miss on GPU1 cannot evict a resident expert on GPU0.
`schedule_placed` and `schedule_remote` share that rule. Dual score still
has no `$/M tokens`.

## Shipped 2026-08-29 — remote-home inside `schedule`

`schedule_remote` / `expertvm schedule --place remote` keeps compute on
GPU0. A miss H2Ds onto the striped home, then `plan_placement` either
D2Ds weights onto GPU0 or ships a small activation payload to home
(`--activation-bytes`). Hits GEMM where the first fetch left the weights.
`--prefetch` fills remote home pages (no GEMM until demand). `expertvm bench` on a multi-GPU profile prints `schedule-remote`
next to gpu0/striped. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — EP homes inside `schedule`

`schedule_placed` / `expertvm schedule --place striped` H2Ds a miss onto
the expert's `PlaceMap` home and GEMMs there, so a wide token uses every
GPU's copy engines instead of serial GPU0. `--place colocated` uses
coactivation homes; `--place replicas` NVLink-copies hot experts onto a
second GPU after the home H2D. `expertvm bench` on a multi-GPU profile
prints `schedule-gpu0` vs `schedule-striped`. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — decode-first, SLO reject, cudaStreamQuery

`--decode-first` holds leftover prefill while any running sequence is
already in decode, so ITL is not waiting on the rest of a long first
token (`SchedCfg::decode_first`). `--slo-reject` drops a waiting sequence
whose queue wait already meets `--ttft-slo-ns` instead of keeping hopeless
FCFS head-of-line work (`rejected=` on the schedule line). `query_stream`
is `cudaStreamQuery`; `mem_info` is `cudaMemGetInfo` `(free, total)`.
`expertvm bench` prints `schedule-decode-first` when a first token has
more than one layer and a later token exists. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — chunked prefill + cudaEventQuery

`--prefill-chunk N` advances a sequence's first token by at most N
layer-events per engine step so a short decode in the same batch is not
stuck behind a long prefill (`SchedCfg::chunked`). `query_event` is
`cudaEventQuery` (unknown id is semantic; incomplete is `Ok(false)`).
Workload `prefill-batch` is four sequences with 4-layer token-0 then
1-layer decode. `expertvm bench` prints `schedule-chunk1` when a first
token has more than one layer. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — event elapsed + schedule lines in benches

`Sim::event_elapsed_ns` is `cudaEventElapsedTime` in nanoseconds (both
records complete; end-before-start is invalid). `expertvm bench` /
`infer-bench` on a multi-sequence trace print `schedule-all` vs
`schedule-1` next to serial/overlap/graphs. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — open-loop continuous batching (`expertvm schedule`)

`Sim::idle_until` drains in-flight work, then jumps the virtual clock
(GPU idle waiting for the next arrival; never skips queued ops).
`schedule_replay` / `expertvm schedule` / `infer-bench schedule` admit
sequences FCFS up to `--max-batch` as they arrive (`--interarrival-ns`,
sequence `s` at `s * interarrival`). Each engine step runs one next token
layer-major across the running set, then `synchronize`. Finished
sequences leave so a later arrival can enter (true continuous batching,
not a token-0 barrier). TTFT is first-token end minus arrival; ITL is
the mean later-token gap; `queue_ns` is mean first-token wait before the
iteration starts. `--ttft-slo-ns` / `--itl-slo-ns` count misses.
The cache walker is demand paging: Oracle/layer-ahead cannot see
unscheduled JSONL future. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — GpuOp DAG, stream sync, expert phases, max-batch

Public `gpu_sim::GpuOp` / `gpu_sim::Operation` is the compiled
dependency DAG (`Sim::operations`, `Sim::operation`). `synchronize_stream`
is `cudaStreamSynchronize`: the virtual clock waits until that stream is
idle while other streams keep running; cancelled ops on *other* streams do
not fail a stream sync. `synchronize_event` is `cudaEventSynchronize` (later
ops on that stream keep running). `sim_replay --cuda-graphs` and `SimulatedGpuStore`
call it before capture so a miss-path H2D can still record a GEMM graph.
`ExpertPhase` is Cold → Transferring → Resident → Leased → Evicting → Cold
(`CachedStore` / `TieredStore` are instant; GPU copies are Transferring until
the copy event completes; `evict` stays Evicting until the free completes;
lease of Transferring is fatal). `Operation` records `submit_ns` / `start_ns` /
`done_ns`. `set_stream_priority` is CUDA stream priority. `SimCfg::max_batch`
admits N sequences per engine iteration at a token (`expertvm sim --max-batch
N`); TTFT/ITL still sample once per token. `expertvm bench` prints serial vs
graphs. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — planner-in-sim, CUDA-graph GEMMs, prefetch hits

`plan_window` Stay vs Fetch now gates prefetch inside `sim_replay` (no future
leak: the window starts at the next event). Stay skips prefetch so a sticky
working set is not evicted by copy-forward ghosts. Fetch still runs the
configured Markov/copy-forward fill and also faults in `window_keys`.
`SimCfg::cuda_graphs` captures grouped expert GEMMs on an idle stream and
replays them with `launch_graph`; graph launch pays `graph_launch_ns` once
instead of per-kernel `launch_overhead_ns`. `SimulatedGpuStore` does the same
per resident page after a drain (completed copy event, idle compute stream).
Replay lines report `prefetch_hits` / `prefetch_waste` / `graph_launches`.
gpu-sim also models `memset`, directed `enable_peer` / `disable_peer`, and the
legacy CUDA null stream (`set_legacy_null_stream`). Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — real-model sidecar JSON drives the llama.cpp check

`tests/reference/qwen2.5-0.5b-instruct-q4_k_m.json` is the source of tokens,
greedy ids, and the max-logit band (not hardcoded). A NORM Llama capture
drops in as `llama-3.2-1b-instruct-q4_k_m.json`; the test skips until that
file exists and still does not download Hugging Face in CI. Dual score still
has no `$/M tokens`.

## Shipped 2026-08-29 — order2 persist + batch stream overlap in benches

`analyze` reports causal `order2_persist‰` (`P(to|from, from_prev)` online, no
future leak). Multi-sequence traces (`workload batch`) print serial vs
`--seq-streams` gpu-sim lines in `expertvm bench` / `infer-bench`. A page's
GEMM stays on the stream that copied it so a later sequence cannot read
before H2D completes. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — alignment, lookback-2 Markov, seq-stream overlap

Memcpy bills `align_up(bytes, align_bytes) + ramp_bytes` so a 1-byte DMA
cannot beat a cache-line copy (`align_bytes` parse key; host PCIe default
128, NVLink 16, RDMA 64). Checked-in profiles now include `h200-sxm`,
`8xh100-nvlink`, and `cheap-48gb`. Online Markov is `P(to|from, from_prev)`
with order-1 backoff (still no prompt-class labels). Decode prefetch uses
the same table. `sim_replay` samples TTFT when `token` changes so a batch
of sequences at one token is one serving-shaped sample. `--seq-streams`
maps `sequence % copy_engines.max(2)` onto CUDA streams so those H2Ds can
overlap compute. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — HBM vs host-pinned residency

`Place::{Host, HostPinned, Device}`: pageable H2D (`memcpy_host_to_device`)
pays `LinkProfile::pageable_permille` (example default 500 = 2× pinned DMA);
pinned H2D is `memcpy_pinned_to_device`. `alloc_host_pinned` is immediate,
does not charge HBM, and a kernel on it is `NotResident` until a copy places
the object on a device. `probe_topology` / expert loads use pinned DMA.
`sim_remote_home` runs `plan_placement` on the home↔GPU0 hop (online reuse,
no future leak): large experts dispatch activations; equal volume moves
weights. CLI: `expertvm remote --activation-bytes N`. Decode leases each
routed expert for the GEMV and releases before the next (`slots=1` still
matches blob logits). `SimulatedGpuStore` holds a host-pinned staging
alloc that does not count toward HBM. Dual score still has
no `$/M tokens`.

## Shipped 2026-08-29 — router permille, decode Markov, RDMA remote-home

`ExpertAccess.weight_pt` is optional router mass in permille (`w` in JSONL;
legacy lines without `w` still parse). Decode records after router weights
and, when a store is attached, prefetches copy-forward ∪ online Markov.
`SimulatedGpuStore` H2D-places onto `expert_id % n_gpus`. `migrate` D2D-moves a page onto a peer (copy stream;
dest GEMM waits the event). `sim_remote_home` / `expertvm remote` compute
on GPU0 and fetch remote-home experts over the peer link (RDMA on
`2node-rdma`). `analyze` reports `mass‰` / `top20_mass‰` when traces carry
`w`. `Prefetch::Both` is copy-forward ∪ Markov (`expertvm sim --prefetch both`).
`with_hot_replicas` uses router mass when `w` is present.

## Shipped 2026-08-29 — Markov prefetch, co-activation placement

`analyze` reports seq persist, reuse-within-8, 90% working set, and
co-activation pair count. `Prefetch::{None,CopyForward,Markov}`: Markov
is online `P(to|from)` with no future leak. `colocated` keeps a co-fired
pair on one GPU; `with_hot_replicas` copies keys whose share ≥ `hot_pt` ‰
onto the next GPU. CLI: `expertvm sim --prefetch markov`, `expertvm place`.

## Shipped 2026-08-29 — kernel curve knobs

`gemm_util_permille` (achieved/peak) and `grouped_moe_permille` (grouped
vs dense duration) scale `kernel_ns`. Default 1000 is identity roofline.
Parse them; do not treat example 1000 as a capture.

## Shipped 2026-08-29 — static EP vs cached expertvm

`sim_static_ep` maps `expert_id % n_gpus` and never evicts. `compare_ep`
runs that next to LRU-on-GPU0. Restricted HBM (`restrict_hbm`) makes
static EP OOM while the cache still decodes. Wide tokens on `8xh100` pay
parallel per-GPU PCIe instead of one serial root. CLI: `expertvm ep`.

## Shipped 2026-08-29 — TTFT/ITL + move vs dispatch

`sim_replay` drains the virtual clock after each token and fills
`Score::{ttft_ns,itl_ns,ns_per_token}` plus `energy_uj`. `plan_placement`
chooses move-weights vs dispatch-activations from expert size, activation
size, fan-in, reuse, and link `bps` (volume crossover, still not `$/M`).

## Shipped 2026-08-29 — energy from profile TDP

`Score::energy_uj` is `node_tdp_mw * wall_ns / 1e6` microjoules. H100/H200
examples are 700 W; `cheap` is 300 W. Parse key `tdp_mw`. This is a
power-envelope estimate, not a cloud bill. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — gpu-sim faults + topology matrix

First-class semantic faults: GPU unavailable, stream cancel of queued
ops, injected memcpy/expert-load failure, extra transfer delay. Named
meshes: 1 GPU, 2× PCIe P2P, 8× NVLink, bad NUMA (far-root H2D), 2-node
RDMA, asymmetric NVLink chain. `probe_topology` measures H2D per GPU and
D2D per pair (`p2p=0->2:none` when the link is missing). CLI:
`gpu-profile names|example|parse|probe`, `expertvm topology`,
`infer-bench topology`. Dual score still has no `$/M tokens`. Ring
`allreduce` requires a real peer path on every hop. CUDA-graph capture
(`begin_capture` / `end_capture` / `launch_graph`) records kernels and
copies without running them; launch replays the graph stream-ordered.

## Shipped 2026-08-29 — Q4_0 SIMD + oracle-owned f16

Q4_0 row kernels (AVX2+FMA+F16C / NEON) on the f32 GEMV and GEMM path.
Dequantized weights stay bit-identical to the scalar kernel; only the
accumulation reassociates. Differential tests cover block sweeps, ragged
rows, every nibble in every lane, every finite binary16 scale, GEMV/GEMM
entry points, and all-zero blocks.

Decode, quant, and GGUF oracles convert binary16 through
`oracle_f16_to_f32` (IEEE arithmetic, repeated `*2`/`*0.5` for powers of
two — not libm `powi`, which is 1 ULP off under Miri) instead of
production bit-surgery `f16_to_f32`, so a repeat of the subnormal
off-by-one cannot hide.

## Shipped 2026-08-29 — TieredStore + adversarial shapes

`TieredStore` pages experts through fast RAM in front of slow RAM, a
seek+read paging file, or synthetic bytes. Only `slots` `ExpertParts`
live in the fast map. mmap stays parked (`WeightStorage::mmap` errors).
Qwen3MoE Tiny identity: TieredStore logits match the blob path.
Adversarial suite is eleven named workloads (coding/chat/long-context,
prefill-heavy, decode-heavy, batch-8, prefill-batch) plus the original four. gpu-sim
asserts concurrent H2D on two streams cannot finish in one-copy time.

`llama-rust` is the correctness laboratory (GGUF math, oracle + llama.cpp
greedy). `expertvm` is expert residency / virtual memory. `gpu-sim` is the
GPU-systems VM (exact invariants, profiled timing). `infer-bench` is
serving-shaped measurement over those traces. Traces are real JSONL
from the decoder; hit rates are measured by `expertvm replay`. Do not
invent `$/M tokens`.

## Shipped 2026-08-29 — ExpertStore decode seam + Session + infer-bench

- Decode GEMV for routed experts can go through `LiveStore`. Default
  `KvCache.expert_store = None` keeps the blob path (allocation-free
  dense decode unchanged). `Llama::expert_direct_store` catalogs every
  MoE layer’s gate/up/down part bytes. Identity: DirectStore, CachedStore
  (full slots), SimulatedGpuStore, and TieredStore bit-match blob logits on writer
  tinies (Qwen3MoE, llama MoE, Qwen2MoE, Llama4, Qwen3Next). Shared
  experts stay on the blob. After routing, prefetch is copy-forward
  `(layer+1, same experts)` union online Markov (`MoeTraceBuf`).
- Layered API: `Model::from_bytes` / `from_gguf` / `encode` / `session`.
  `Session::{prefill, decode, attach_expert_store, expert_metrics}`.
  Example: `cargo run -p llama-rust --example session`.
- Workspace crate [`infer-bench/`](infer-bench/): `adversarial | trace |
  workload`. Same numbers as `expertvm bench`. Score is
  `wall_ns` / `hbm_peak` / `bytes_moved` / optional `ns_per_token`.
- `expertvm` CLI also has `bench` and `workload`. `pin_hot` NVLink-replicates
  to GPU1 when the profile has `n_gpus >= 2`.

## Shipped 2026-08-28 — expertvm + gpu-sim + MoE traces

- Workspace crates: [`gpu-sim/`](gpu-sim/), [`expertvm/`](expertvm/).
- `gguf_gemv trace <gguf> --out FILE` emits `ExpertAccess` JSONL. Opt-in;
  greedy tokens match the untraced path. Dense models emit zero events.
- Checked-in traces: [`tests/traces/`](tests/traces/). Cycling synthetic is
  the policy discriminator (LRU 0‰ vs oracle 458‰ at capacity 2). Writer-built
  tinies have a 2-expert working set — not a 320B result.
- `expertvm analyze|replay|sim`. `DirectStore` / `CachedStore` (leases).
  `sim_replay` runs H2D+GEMM on `gpu-sim` profiles (`h100`, `h200`, `cheap`).
- Kill-switch still applies: do not build CUDA until a **real** MoE GGUF
  shows non-oracle policies beating random by a lot.

# Stopped 2026-08-28 — official phi2 (historical)

The phi2 slice shipped. Resume from PLAN.md, not from the old “Metal or bloom”
item below.

## Shipped (use this)

Repo: https://github.com/mingley/llama-rust
Local: `~/dev/llama-rust-perf`

- `forbid(unsafe_code)` without `simd`, no llama.cpp/FFI, `Cargo.lock` has no crates.io packages (workspace path crates `gpu-sim` / `expertvm` / `infer-bench` only).
- GGUF v3: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q1_0, Q2_0, TQ1_0, TQ2_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ1_M, IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, MXFP4, NVFP4. Kernels read on-disk bytes (no private f32-scale copy).
- F16 is IEEE binary16 (`GGML_TYPE_F16` = 1). Writer-built tiny uses F16 for 2-D weights (`token_embd`, `output`, attn/ffn). 1-D attn/ffn/`output_norm` and optional `attn_{q,k,v}.bias` may be F16 or F32; on-disk F16 stays IEEE binary16 and is applied via the same `ggml_fp16_to_fp32` scalar walk as 2-D F16. Writer-built tiny can emit F16 1-D norms (and F16 QKV bias). Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml/Llama math as the F32-norm twin. A 1-D F16 tensor that is not a norm/bias this crate applies fails with a named error. Not a new dtype. No tok/s.
- BF16 is ggml bfloat16 (`GGML_TYPE_BF16` = 30). Writer-built tiny uses BF16 for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_bf16` / `GGML_BF16_TO_FP32` math (IEEE binary16-adjacent: 8-bit exp, high-16 of f32). No tok/s.
- Q2_K is `GGML_TYPE_Q2_K` = 10 (84-byte `block_q2_K`). Writer-built tiny uses Q2_K for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_q2_K` walk (`d*(sc&0xF)*q2 - dmin*(sc>>4)`). No tok/s.
- Q3_K is `GGML_TYPE_Q3_K` = 11 (110-byte `block_q3_K`). Writer-built tiny uses Q3_K for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_q3_K` walk (`d*(sc-32)*q3`, `hmask` high bit). No tok/s.
- Q4_1 is `GGML_TYPE_Q4_1` = 3 (20-byte `block_q4_1`). Writer-built tiny uses Q4_1 for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_q4_1` walk (`q*d + m`, unsigned 4-bit). No tok/s.
- Q5_0 is `GGML_TYPE_Q5_0` = 6 (22-byte `block_q5_0`). Writer-built tiny uses Q5_0 for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_q5_0` walk (`(q-16)*d`, `qh` 5th bit). No tok/s.
- Q5_1 is `GGML_TYPE_Q5_1` = 7 (24-byte `block_q5_1`). Writer-built tiny uses Q5_1 for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_q5_1` walk (`q*d + m`, `qh` 5th bit). No tok/s.
- MXFP4 is `GGML_TYPE_MXFP4` = 39 (17-byte `block_mxfp4`). Writer-built tiny uses MXFP4 for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_mxfp4` walk (`kvalues_mxfp4[q] * GGML_E8M0_TO_FP32_HALF(e)`, lo nibble `y[j]`, hi nibble `y[j+16]`). No tok/s.
- NVFP4 is `GGML_TYPE_NVFP4` = 40 (36-byte `block_nvfp4`). Writer-built tiny uses NVFP4 for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_nvfp4` walk (`kvalues_fp4[q] * ggml_ue4m3_to_fp32(d[s])`, four 16-wide sub-blocks, lo nibble `yb[j]`, hi nibble `yb[j+8]`). No tok/s.
- Q1_0 is `GGML_TYPE_Q1_0` = 41 (18-byte `block_q1_0`). Writer-built tiny uses Q1_0 for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_q1_0` walk (`bit ? d : -d`, sequential LSB-first 1-bit pack, fp16 `d`). No tok/s.
- Q2_0 is `GGML_TYPE_Q2_0` = 42 (18-byte `block_q2_0`). Writer-built tiny uses Q2_0 for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_q2_0` walk (`(q-1)*d`, sequential LSB-first 2-bit pack, fp16 `d`, `QK2_0=64`). No tok/s.
- Q8_1 is `GGML_TYPE_Q8_1` = 9 (36-byte `block_q8_1`). Writer-built tiny uses Q8_1 for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml Q8_1 dequant walk (`q*d`, fp16 `d` + fp16 `s=d*sum(qs)` + `qs[32]` int8, `QK8_1=32`). Distinct from Q8_0=8 (34 B / 32, no `s`). No tok/s.
- TQ1_0 is `GGML_TYPE_TQ1_0` = 34 (54-byte `block_tq1_0`). Writer-built tiny uses TQ1_0 for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_tq1_0` walk (`y = (xi - 1) * d`, `qs[48]` 5 trits/byte then `qh[4]` 4 trits/byte, `q * pow3[l]` then `((q as u16 * 3) >> 8)`, `QK_K=256`). Distinct from TQ2_0=35 (66 B / 256), Q1_0=41 (18 B / 128), and Q8_1=9 (36 B / 32). No tok/s.
- TQ2_0 is `GGML_TYPE_TQ2_0` = 35 (66-byte `block_tq2_0`). Writer-built tiny uses TQ2_0 for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_tq2_0` walk (`y = (xi - 1) * d`, `qs[64]` 4 trits/byte at 2 bits, two 32-byte groups `j=0` then `j=32`, `l` then `m`, `(qs[j+m] >> (l*2)) & 3`, `QK_K=256`). Distinct from TQ1_0=34 (54 B / 256, 5 trits/byte + qh), Q2_0=42 (18 B / 64, sequential 2-bit), and Q1_0=41 (18 B / 128). No tok/s.
- ggml type ids 36/37/38 (`IQ4_NL_4_4` / `IQ4_NL_4_8` / `IQ4_NL_8_8`) are ggml-removed slots (`TYPE_IQ4_NL_4_4 REMOVED, use IQ4_NL with runtime repacking`, `blck_size = 0`, `type_size = 0`, `is_quantized = false`). Classified as ggml-removed, not a missing dequant, and not IQ4_NL (20). A 2-D tensor tagged type 36 fails with a named error (`ggml-removed type 36`). No `block_iq4_nl_4_4` dequant. No tok/s.
- Tied `output.weight`: when the tensor is absent, load reuses the already-loaded `token_embd.weight` (same on-disk bytes, same blob range, no matrix clone, no mmap). Writer-built tiny can omit `output.weight` and still load. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml/Llama math as an untied file whose `output.weight` is an identical copy of `token_embd`. Missing both tensors still fails with a named error. Not a dtype. No tok/s.
- Q5_K is `GGML_TYPE_Q5_K` = 13 (176-byte `block_q5_K`). Writer-built tiny uses Q5_K for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_q5_K` walk (`d*sc*q5 - dmin*m`, `qh` 5th bit). No tok/s.
- IQ4_XS is `GGML_TYPE_IQ4_XS` = 23 (136-byte `block_iq4_xs`). First IQ* type that common OSS `*-IQ4_XS.gguf` files actually have. Writer-built tiny uses IQ4_XS for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_iq4_xs` walk (`d*(ls-32)*kvalues_iq4nl[q]`). No tok/s.
- IQ4_NL is `GGML_TYPE_IQ4_NL` = 20 (18-byte `block_iq4_nl`). Next IQ* type that common OSS `*-IQ4_NL.gguf` files actually have (bartowski / mradermacher standalone, and mixed IQ*_M tensors). Writer-built tiny uses IQ4_NL for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_iq4_nl` walk (`d * kvalues_iq4nl[q]`). No tok/s.
- IQ3_S is `GGML_TYPE_IQ3_S` = 21 (110-byte `block_iq3_s`). Next remaining IQ* type that common OSS `*-IQ3_S.gguf` files actually have (bartowski / mradermacher standalone, and the primary 2-D dtype in mixed `*-IQ3_M.gguf`). Writer-built tiny uses IQ3_S for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_iq3_s` walk (`d*(1+2*ls)*iq3s_grid[q]*sign`). No tok/s.
- IQ3_XXS is `GGML_TYPE_IQ3_XXS` = 18 (98-byte `block_iq3_xxs`). Next remaining IQ* type that common OSS `*-IQ3_XXS.gguf` files actually have (bartowski / mradermacher standalone, and IQ3_XXS tensors in mixed `*-IQ3_XS.gguf`). Writer-built tiny uses IQ3_XXS for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_iq3_xxs` walk (`d*(0.5+ls)*0.5*iq3xxs_grid[q]*ksigns`). No tok/s.
- IQ2_S is `GGML_TYPE_IQ2_S` = 22 (82-byte `block_iq2_s`). Common OSS `*-IQ2_S.gguf` files actually have this type (bartowski / mradermacher standalone, and the primary 2-D dtype in mixed `*-IQ2_M.gguf`). Writer-built tiny uses IQ2_S for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_iq2_s` walk (`d*(0.5+ls)*0.25*iq2s_grid[q]*sign`). No tok/s.
- IQ2_XXS is `GGML_TYPE_IQ2_XXS` = 16 (66-byte `block_iq2_xxs`). Common OSS `*-IQ2_XXS.gguf` files actually have this type (bartowski / mradermacher standalone). Writer-built tiny uses IQ2_XXS for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_iq2_xxs` walk (`d*(0.5+ls)*0.25*iq2xxs_grid[q]*ksigns`). No tok/s.
- IQ2_XS is `GGML_TYPE_IQ2_XS` = 17 (74-byte `block_iq2_xs`). Next remaining IQ* type that common OSS `*-IQ2_XS.gguf` files actually have (bartowski / mradermacher standalone). Writer-built tiny uses IQ2_XS for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_iq2_xs` walk (`d*(0.5+ls)*0.25*iq2xs_grid[q]*ksigns`). No tok/s.
- IQ1_S is `GGML_TYPE_IQ1_S` = 19 (50-byte `block_iq1_s`). Common OSS `*-IQ1_S.gguf` files actually have this type (bartowski / mradermacher standalone). Writer-built tiny uses IQ1_S for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_iq1_s` walk (`d*(2*ls+1)*(iq1s_grid[q]±0.125)`). No tok/s.
- IQ1_M is `GGML_TYPE_IQ1_M` = 29 (56-byte `block_iq1_m`). Last remaining IQ* type that common OSS `*-IQ1_M.gguf` files actually have (bartowski / mradermacher standalone). Writer-built tiny uses IQ1_M for 2-D weights (`token_embd`, `output`, attn/ffn); 1-D norms stay F32. Load/GEMV/GEMM/embed logits match an independent scalar of the same ggml `dequantize_row_iq1_m` walk (`d*(2*ls+1)*(iq1s_grid[q]±0.125)`, fp16 `d` packed in scale high nibbles). No tok/s.
- Decode: RMSNorm, RoPE, GQA+KV, SwiGLU (llama/qwen2/mistral/phi3/qwen3/llama4/qwen2moe/qwen3moe/qwen2vl/qwen3vl/qwen3next/qwen35) or Gemma GeGLU or official Phi2 sequential GELU, lm_head, greedy sample by default. Gemma scales token embeds by `sqrt(n_embd)` (`ggml_scale`). Official Qwen3 applies per-head QK-Norm (`attn_q_norm` / `attn_k_norm` RMSNorm on Q and K after projection, before RoPE). Official Qwen3MoE applies the same QK-Norm plus `build_moe_ffn` softmax then top-k with `norm_w` clamp `2^-14` and no shared expert. Official Llama4 text applies iRoPE/NoPE, unweighted QK-Norm after RoPE, and expert FFN on MoE layers. Official llama MoE (`architecture=llama` with `n_expert>0`) applies `build_moe_ffn` softmax then top-k, SwiGLU, weights after the expert with `norm_w` clamp `2^-14`. Official Qwen2MoE (`architecture=qwen2moe`) applies softmax then top-k without `norm_w`, SwiGLU experts, and a shared expert gated by `silu(x)/x` on `ffn_gate_inp_shexp`. Official Qwen2VL applies the Qwen2 language walk plus m-RoPE (`ggml_rope_multi` / `LLAMA_ROPE_TYPE_MROPE`, `rope.dimension_sections`, text `n_pos_per_embd=4` with `[t,h,w,e]=[p,p,p,0]`). Official Qwen3VL applies Qwen3 QK-Norm plus interleaved m-RoPE (`ggml_rope_multi` / `LLAMA_ROPE_TYPE_IMROPE`, required `rope.dimension_sections`, text `n_pos_per_embd=4` with `[t,h,w,e]=[p,p,p,0]`). Official Qwen3Next applies gated full attention (joint Q+gate, QK-Norm, sigmoid after attn), official `post_attention_norm`, and `build_moe_ffn` softmax then top-k with `norm_w` clamp `2^-14` plus a shared expert gated by sigmoid. Official Qwen35 applies gated full attention (joint Q+gate, QK-Norm, IMROPE, sigmoid after attn), official `post_attention_norm`, and dense SwiGLU. Official Phi2 applies LayerNorm, NEOX RoPE, Q-scale then attn scale `1.0`, and parallel GELU-seq FFN. Linear-attn / gated-delta layers are refused. RMSNorm +1 is a convert-hf bake on GGUF `norm.weight` bytes; decode uses `LLM_NORM_RMS` as-is.
- **Gemma architecture.** `general.architecture=gemma` with `gemma.*` KV (same prefix pattern as mistral/phi3/qwen2). Same `blk.{i}.*` tensor names as llama. Writer-built `tiny-gemma` loads and greedy-generates. Load/GEMV/GEMM/embed logits match an independent scalar of the official llama.cpp Gemma walk (embed scale + `LLM_FFN_GELU`/`ggml_gelu` tanh approx). `gemma2` stays rejected (post-norms, SWA, softcap). Not a dtype. No tok/s.
- **Qwen3 architecture.** `general.architecture=qwen3` (gguf-py `MODEL_ARCH_NAMES[QWEN3] = "qwen3"`) with `qwen3.*` KV (same prefix pattern as mistral/phi3/qwen2/gemma). Official named difference vs qwen2/llama, measured from llama.cpp `src/models/qwen3.cpp`: QK-Norm (`blk.{i}.attn_q_norm` / `attn_k_norm`, `LLM_NORM_RMS` on Q and K after projection / before RoPE, weight shape `{n_embd_head_k}`). FFN stays SwiGLU (`LLM_FFN_SILU`). No embed-scale, GeGLU, extra norms, or softcap. Writer-built `tiny-qwen3` loads and greedy-generates. Load/GEMV/GEMM/embed logits match an independent scalar of the official Qwen3 walk on those GGUF bytes. Not a dtype. No tok/s.
- **Llama4 architecture.** `general.architecture=llama4` (gguf-py `MODEL_ARCH_NAMES[LLAMA4] = "llama4"`) with `llama4.*` KV. Official named differences vs llama/qwen3/gemma, measured from llama.cpp `src/models/llama4.cpp` text walk: iRoPE/NoPE (`use_rope = n_no_rope_layer_step > 0 && (il+1) % n_no_rope_layer_step != 0`, default step 4; NoPE scales Q by `log(floor((pos+1)/8192)+1)*0.1+1`); unweighted QK-Norm after RoPE (`ggml_rms_norm` / `Llama4TextL2Norm`, no `attn_q_norm` tensors; off when `expert_count == 128`); expert FFN on MoE layers (`(il+1) % interleave_moe_layer_step == 0`: `ffn_gate_inp` / `ffn_{gate,up,down}_exps` / `ffn_{gate,up,down}_shexp`, top-k on raw logits, sigmoid weights applied before SwiGLU, shared expert add). Dense layers stay SwiGLU. Official load rejects `expert_count == 0`. No embed-scale, GeGLU, extra norms, softcap, or vision. Writer-built `tiny-llama4` loads and greedy-generates. Load/GEMV/GEMM/embed logits match an independent scalar of the official Llama4 text walk on those GGUF bytes. `mixtral` stays rejected (not an official arch). Not a dtype. No tok/s.
- **Official llama MoE.** `general.architecture=llama` with `n_expert>0` / `llama.expert_count` and `llama.expert_used_count` on `llama.*` KV. Official convert writes MixtralForCausalLM as `general.architecture=llama` (`MixtralForCausalLM` → `"llama"`). Official llama.cpp has no Mixtral model class (no `mixtral.cpp`, no `LLM_ARCH_MIXTRAL`, no `"mixtral"` in `LLM_ARCH_NAMES`); load of `general.architecture=mixtral` stays `unknown architecture mixtral` (`LLM_ARCH_UNKNOWN`). Official walk measured from `src/models/llama.cpp` `build_moe_ffn`: softmax, then top-k; SwiGLU (`LLM_FFN_SILU`); weights after the expert with `norm_w` clamp `2^-14` (`6.103515625e-5`); no shared expert (Granite `n_ff_shexp` is not this slice). Not Llama4 sigmoid / raw-logit top-k / shared-expert / iRoPE / QK-Norm. Writer-built tiny is **llama**, not mixtral. Writer-built tiny loads and greedy-generates. Load/GEMV/GEMM/embed logits match an independent scalar of the official llama MoE walk on those GGUF bytes. Dense `llama` (`n_expert==0`) stays the existing dense SwiGLU path. Not a dtype. Not a new arch. No tok/s.
- **Qwen2MoE architecture.** `general.architecture=qwen2moe` (gguf-py `MODEL_ARCH_NAMES[QWEN2MOE] = "qwen2moe"`, llama.cpp `LLM_ARCH_QWEN2MOE = "qwen2moe"`, `src/models/qwen2moe.cpp`) with `qwen2moe.*` KV. Official convert writes `general.architecture=qwen2moe`. Official named differences vs llama MoE / Llama4, measured from `src/models/qwen2moe.cpp`: shared expert (`ffn_{gate,up,down}_shexp` / `ffn_gate_inp_shexp`, `n_ff_shexp` from `expert_shared_feed_forward_length` else `n_ff`); expert FF length (`n_ff_exp` from `expert_feed_forward_length` else `n_ff / n_expert_used`); `build_moe_ffn` softmax then top-k with `norm_w=false`; shared expert SwiGLU multiplied by `silu(x)/x` (sigmoid) of `ffn_gate_inp_shexp`. Official load rejects `n_expert==0` / `n_expert_used==0`. Not Llama4 sigmoid / raw-logit top-k / weight-before-FFN. Not llama-MoE `norm_w` clamp. No QK-Norm, embed-scale, GeGLU, extra norms, softcap, or vision. Writer-built `tiny-qwen2moe` loads and greedy-generates. Load/GEMV/GEMM/embed logits match an independent scalar of the official Qwen2MoE walk on those GGUF bytes. `mixtral` stays rejected as unknown architecture. Not a dtype. No tok/s.
- **Qwen3MoE architecture.** `general.architecture=qwen3moe` (gguf-py `MODEL_ARCH_NAMES[QWEN3MOE] = "qwen3moe"`, llama.cpp `LLM_ARCH_QWEN3MOE = "qwen3moe"`, `src/models/qwen3moe.cpp`) with `qwen3moe.*` KV. Official convert writes `general.architecture=qwen3moe`. Official named differences vs qwen3 / qwen2moe, measured from `src/models/qwen3moe.cpp`: Qwen3 QK-Norm (`blk.{i}.attn_q_norm` / `attn_k_norm` after projection / before RoPE); MoE experts (`ffn_gate_inp` / `ffn_{gate,up,down}_exps`, `n_ff_exp` from `expert_feed_forward_length` else `n_ff / n_expert_used`); `build_moe_ffn` softmax then top-k with `norm_w=true` clamp `2^-14`; no shared expert. Official load rejects `n_expert==0` / `n_expert_used==0`. Tied `output.weight` reuse is allowed. Not qwen2moe shexp / `norm_w=false`. Not Llama4 sigmoid / raw-logit top-k / weight-before-FFN. No embed-scale, GeGLU, extra norms, softcap, or vision. Writer-built `tiny-qwen3moe` loads and greedy-generates. Load/GEMV/GEMM/embed logits match an independent scalar of the official Qwen3MoE walk on those GGUF bytes. `mixtral` stays rejected as unknown architecture. Not a dtype. No tok/s.
- **Qwen2VL architecture.** `general.architecture=qwen2vl` (gguf-py `MODEL_ARCH_NAMES[QWEN2VL] = "qwen2vl"`, llama.cpp `LLM_ARCH_QWEN2VL = "qwen2vl"`, `src/models/qwen2vl.cpp`) with `qwen2vl.*` KV. Official convert writes `general.architecture=qwen2vl` (`Qwen2VLForConditionalGeneration` and `Qwen2_5_VLForConditionalGeneration` → `"qwen2vl"`). Official named difference vs qwen2, measured from `src/models/qwen2vl.cpp` plus `ggml_compute_forward_rope_flt`: m-RoPE (`ggml_rope_multi` / `LLAMA_ROPE_TYPE_MROPE`, required `qwen2vl.rope.dimension_sections` arr[i32,4], text `n_pos_per_embd=4` with `[t,h,w,e]=[p,p,p,0]` from `llama-graph.cpp`). Language-model tensors stay Qwen2 (SwiGLU, QKV bias, tied `output.weight` reuse). Vision / mmproj lives in official `tools/mtmd/models/qwen2vl.cpp` (clip); not a second language arch. `qwen3vlmoe` / a separate `qwen25vl` language arch stay rejected. Not Mixtral, not QK-Norm, not embed-scale, not GeGLU. Writer-built `tiny-qwen2vl` loads and greedy-generates. Load/GEMV/GEMM/embed logits match an independent scalar of the official Qwen2VL language-model walk on those GGUF bytes. `mixtral` stays rejected as unknown architecture. Not a dtype. No tok/s.
- **Qwen3VL architecture.** `general.architecture=qwen3vl` (gguf-py `MODEL_ARCH_NAMES[QWEN3VL] = "qwen3vl"`, llama.cpp `LLM_ARCH_QWEN3VL = "qwen3vl"`, `src/models/qwen3vl.cpp`) with `qwen3vl.*` KV. Official convert writes `general.architecture=qwen3vl` (`Qwen3VLForConditionalGeneration` → `"qwen3vl"`). Official named differences vs qwen3 / qwen2vl, measured from `src/models/qwen3vl.cpp` plus `llama_model_rope_type` / `ggml_mrope_cache_init`: Qwen3 QK-Norm (`blk.{i}.attn_q_norm` / `attn_k_norm` after projection / before RoPE) plus interleaved m-RoPE (`ggml_rope_multi` / `LLAMA_ROPE_TYPE_IMROPE`, required `qwen3vl.rope.dimension_sections` arr[i32,4], text `n_pos_per_embd=4` with `[t,h,w,e]=[p,p,p,0]` from `llama-graph.cpp` / `llama_hparams::n_pos_per_embd`). FFN stays dense SwiGLU (`LLM_FFN_SILU`). Tied `output.weight` reuse is allowed. Official `n_deepstack_layers` is optional (`false`) and vision-side; language-only load omits it (default 0). Vision / mmproj lives in official `tools/mtmd/models/qwen3vl.cpp` (clip); not a second language arch. `qwen3vlmoe` stays rejected in this slice. Not Mixtral, not qwen2vl MROPE, not qwen3moe experts, not Llama4, not embed-scale, not GeGLU, no extra norms. Writer-built `tiny-qwen3vl` loads and greedy-generates. Load/GEMV/GEMM/embed logits match an independent scalar of the official Qwen3VL language-model walk on those GGUF bytes. `mixtral` stays rejected as unknown architecture. Not a dtype. No tok/s.
- **Qwen3Next architecture.** `general.architecture=qwen3next` (gguf-py `MODEL_ARCH_NAMES[QWEN3NEXT] = "qwen3next"`, llama.cpp `LLM_ARCH_QWEN3NEXT = "qwen3next"`, `src/models/qwen3next.cpp`) with `qwen3next.*` KV. Official convert writes `general.architecture=qwen3next` (`Qwen3NextForCausalLM` → `"qwen3next"`). Official named differences vs qwen3 / qwen3moe / qwen2moe, measured from `src/models/qwen3next.cpp`: gated full attention (joint `attn_q` is query+gate `n_embd_head * n_head * 2`, QK-Norm after projection / before RoPE, sigmoid gate after attention); official `blk.{i}.post_attention_norm` (not `ffn_norm`); `build_moe_ffn` softmax then top-k with `norm_w=true` clamp `2^-14`; shared expert (`*_shexp` + `ffn_gate_inp_shexp`) gated by sigmoid; official convert writes `rope.dimension_count = head_dim * partial_rotary_factor` (default 0.25) and required `ssm.*` KV; `full_attention_interval` defaults to 4 (`is_recr = (il+1) % interval != 0`). Official load rejects `n_expert==0`. Writer-built `tiny-qwen3next` uses `full_attention_interval=1` so the single layer is the official full-attention path. Tied `output.weight` reuse is allowed. `qwen3vlmoe` stays rejected. Not Mixtral, not qwen3vl IMROPE, not a qwen3 / qwen3moe / qwen2moe redo, no invented extra norms. Writer-built `tiny-qwen3next` loads and greedy-generates. Load/GEMV/GEMM/embed logits match an independent scalar of the official Qwen3Next language-model walk on those GGUF bytes. `mixtral` stays rejected as unknown architecture. Not a dtype. No tok/s.
- **Qwen35 architecture.** `general.architecture=qwen35` (gguf-py `MODEL_ARCH_NAMES[QWEN35] = "qwen35"`, llama.cpp `LLM_ARCH_QWEN35 = "qwen35"`, `src/models/qwen35.cpp`) with `qwen35.*` KV. Official convert writes `general.architecture=qwen35` (`Qwen3_5ForConditionalGeneration` / `Qwen3_5ForCausalLM` → `"qwen35"`). Official named differences vs qwen3 / qwen3vl / qwen3next, measured from `src/models/qwen35.cpp`: gated full attention (joint `attn_q` is query+gate `n_embd_head * n_head * 2`, QK-Norm after projection / before RoPE, sigmoid gate after attention); interleaved m-RoPE (`ggml_rope_multi` / `LLAMA_ROPE_TYPE_IMROPE`, required `qwen35.rope.dimension_sections`, text `n_pos_per_embd=4` with `[t,h,w,e]=[p,p,p,0]`); official `blk.{i}.post_attention_norm` (not `ffn_norm`); dense SwiGLU (`LLM_FFN_SILU`; official assert: no `ffn_gate_inp`); official load requires `ssm.*` KV; `full_attention_interval` defaults to 4 (`is_recr = (il+1) % interval != 0`). Official convert default `mrope_section` is `[11, 11, 10, 0]`. Official GGUF files write `rope.dimension_count` as full `head_dim` (not qwen3next partial rotary). Writer-built `tiny-qwen35` uses `full_attention_interval=1` so the single layer is the official full-attention path. Linear-attn / gated-delta layers are refused. Tied `output.weight` reuse is allowed. `qwen3vlmoe` stays rejected. Not Mixtral, not a qwen3next MoE redo, not a qwen3vl redo, no invented extra norms. Writer-built `tiny-qwen35` loads and greedy-generates. Load/GEMV/GEMM/embed logits match an independent scalar of the official Qwen35 language-model walk on those GGUF bytes. `mixtral` stays rejected as unknown architecture. Not a dtype. No tok/s.
- **Phi2 architecture.** `general.architecture=phi2` (gguf-py `MODEL_ARCH_NAMES[PHI2] = "phi2"`, llama.cpp `LLM_ARCH_PHI2 = "phi2"`, `src/models/phi2.cpp`) with `phi2.*` KV. Official convert writes `general.architecture=phi2` (`PhiForCausalLM` → `"phi2"`). Official named differences vs llama / phi3 / gemma, measured from `src/models/phi2.cpp` plus `llama_model_rope_type` / `conversion/phi.py`: `LLM_NORM` (LayerNorm + bias on `attn_norm` / `output_norm`); `LLAMA_ROPE_TYPE_NEOX`; Q scaled by `1/sqrt(n_embd_head)` then `build_attn` scale `1.0`; parallel residual (`attn` and `LLM_FFN_GELU`/`LLM_FFN_SEQ` both from `attn_norm`; no `ffn_gate`, no `ffn_norm`); required `output.bias` / `attn_output.bias` / FFN biases. Official convert writes `attention.layer_norm_epsilon` (not rms), `feed_forward_length = 4 * n_embd`, `head_count_kv = n_head`, `rope.dimension_count = int(partial_rotary_factor * n_embd) // n_head`, and `add_bos_token=false`. Writer-built `tiny-phi2` uses that convert shape (`n_ff=1024`, `n_rot=32` of `64`). Tied `output.weight` reuse is allowed. `qwen3vlmoe` stays rejected. Not Mixtral, not a phi3 redo, not linear-attn, no invented extra norms. Writer-built `tiny-phi2` loads and greedy-generates. Load/GEMV/GEMM/embed logits match an independent scalar of the official Phi2 language-model walk on those GGUF bytes. `mixtral` stays rejected as unknown architecture. Not a dtype. No tok/s.
- **Sampling.** Seedless greedy (`temperature <= 0`, argmax, first index on ties) is still the `infer` / `greedy_generate` path. `SampleParams` + `generate` add temperature, top-k, top-p, and unique-id repeat penalty (`logit > 0` then `/=`, else `*=`). Stochastic draws use SplitMix64 and require a seed. No CLI sampling flags.
- Prefill GEMM. Prompt tokens are one causal pass. A single token stays GEMV.
- Architectures: `llama` (dense `n_expert==0`, or official llama MoE when `n_expert>0`), `qwen2`, `mistral`, `phi3`, `gemma`, `qwen3`, `llama4`, `qwen2moe`, `qwen3moe`, `qwen2vl`, `qwen3vl`, `qwen3next`, `qwen35`, `phi2` `{arch}.*` KV. `gemma2` and `mixtral` stay rejected. Official Mixtral GGUF is `architecture=llama` with `n_expert>0`, not a `mixtral` arch. `qwen3vlmoe` stays rejected. A separate `qwen25vl` language arch stays rejected (official Qwen2.5-VL text is `architecture=qwen2vl`). Later official families still in `LLM_ARCH_NAMES` stay rejected.
- Q4_K_M shape that common OSS files actually have:
  - quantized `token_embd.weight` (Q1_0 / Q2_0 / TQ1_0 / TQ2_0 / Q2_K / Q3_K / Q4_1 / Q4_K / Q5_0 / Q5_1 / Q5_K / Q6_K / Q8_1 / IQ1_M / IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ4_NL / IQ4_XS / MXFP4 / NVFP4 / F32) or F16 / BF16
  - missing `{arch}.rope.dimension_count` derived from `embedding_length / head_count`
  - optional F32 or F16 `attn_{q,k,v}.bias`
  - `tokenizer.ggml.add_bos_token=false` honored
- Load/decode errors name tensor, ggml type id, and/or KV key. ggml-removed type ids are named as removed.
- CLI: `gguf_gemv infer <path> [--prompt TEXT] [--n-predict N] [--n-ctx N]`. Seedless greedy. Defaults remain `ab` / 2 so the shipped two-run command still works.
- **MoE traces.** `gguf_gemv trace <path> --out FILE [--capacity N]`. Same greedy as `infer`, writes JSONL, prints the measured expertvm hit-rate table. Identity vs untraced greedy.
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

PLAN.md Phase 0 leftover: a second real-model fixture when a Llama
NORM-RoPE GGUF is on disk (NEOX Qwen capture already exists). No GGUF is
in this workspace, so that checkbox stays open. Physical Phase 4 stays
parked until a real MoE trace + sim hypothesis.

## Still needed (production / researcher bar)

Ordered by how much they block “others can actually use this”:

1. **Metal-in-crate.** Owned MSL kernels exist as a sidecar. Decode still CPU.
2. **Dtypes / arches still rejected.** Remaining official vision families, remaining MoE families. `gemma` `{arch}.*` KV loads (writer-built tiny). `qwen3` `{arch}.*` KV loads (writer-built tiny; official QK-Norm). `llama4` `{arch}.*` KV loads (writer-built tiny; official iRoPE/NoPE, unweighted QK-Norm after RoPE, expert FFN). Official llama MoE (`architecture=llama` with `n_expert>0`) loads (writer-built tiny; official `build_moe_ffn` softmax then top-k, `norm_w` clamp `2^-14`). `qwen2moe` `{arch}.*` KV loads (writer-built tiny; official softmax then top-k without `norm_w`, shared expert gated by `silu(x)/x`). `qwen3moe` `{arch}.*` KV loads (writer-built tiny; official Qwen3 QK-Norm plus softmax then top-k with `norm_w` clamp `2^-14`, no shared expert). `qwen2vl` `{arch}.*` KV loads (writer-built tiny; official Qwen2 plus m-RoPE / `ggml_rope_multi`). `qwen3vl` `{arch}.*` KV loads (writer-built tiny; official Qwen3 QK-Norm plus interleaved m-RoPE / `LLAMA_ROPE_TYPE_IMROPE`). `qwen3next` `{arch}.*` KV loads (writer-built tiny; official gated full attention, `post_attention_norm`, MoE `norm_w` plus sigmoid-gated shared expert). `qwen35` `{arch}.*` KV loads (writer-built tiny; official gated full attention, IMROPE, `post_attention_norm`, dense SwiGLU; linear-attn / gated-delta refused). `phi2` `{arch}.*` KV loads (writer-built tiny; official LayerNorm, NEOX RoPE, Q-scale, parallel GELU-seq FFN). `gemma2` is still rejected (different official decode family). `mixtral` is still rejected (`mixtral` is not an official arch; official Mixtral GGUF is `architecture=llama` with `n_expert>0`). `qwen3vlmoe` is still rejected. 1-D F16 norms/bias load (attn/ffn/`output_norm`, official Qwen3 `attn_{q,k}_norm`, official phi2 LayerNorm/FFN/output bias, and optional `attn_{q,k,v}.bias`). Tied `output.weight` reuses `token_embd.weight` when absent. Common OSS IQ* 2-D, BF16 2-D, Q2_K 2-D, Q3_K 2-D, Q4_1 2-D, Q5_0 2-D, Q5_1 2-D, MXFP4 2-D, NVFP4 2-D, Q1_0 2-D, Q2_0 2-D, Q8_1 2-D, TQ1_0 2-D, and TQ2_0 2-D are loaded (IQ1_M / IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ4_NL / IQ4_XS / BF16 / Q2_K / Q3_K / Q4_1 / Q5_0 / Q5_1 / MXFP4 / NVFP4 / Q1_0 / Q2_0 / Q8_1 / TQ1_0 / TQ2_0). IQ4_NL_4_4/4_8/8_8 (36..=38) are ggml-removed, not a remaining hole. Next remaining live rejected ggml weight type id: none. Official phi2 loaded; next remaining named item-2 hole is another official family still in `LLM_ARCH_NAMES` that is not qwen3vlmoe, not mixtral, and not linear-attn (`bloom`). Do not invent an arch. Do not invent a dtype. Do not list mixtral or qwen3vlmoe as accepted.
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
./target/release/expertvm bench adversarial --capacity 2
./target/release/infer-bench trace tests/traces/cycling.jsonl --capacity 2
./target/release/infer-bench remote tests/traces/cycling.jsonl --expert-bytes 1048576
./target/release/expertvm topology --bytes 1048576
./target/release/expertvm remote tests/traces/cycling.jsonl --expert-bytes 1048576
./target/release/expertvm schedule tests/traces/cycling.jsonl --capacity 2 --max-batch 1 --interarrival-ns 1000000 --prefill-chunk 1 --decode-first --slo-reject --ttft-slo-ns 1
./target/release/expertvm schedule tests/traces/cycling.jsonl --capacity 8 --max-batch 1 --prefix-cache
./target/release/expertvm workload shared-prefix
./target/release/expertvm schedule tests/traces/cycling.jsonl --capacity 8 --place striped --profile 8xh100 --expert-bytes 1048576
./target/release/expertvm schedule tests/traces/cycling.jsonl --capacity 8 --place replicas --profile 8xh100 --expert-bytes 1048576
./target/release/expertvm schedule tests/traces/cycling.jsonl --capacity 8 --place remote --profile 2node-rdma --expert-bytes 1048576 --prefetch copy-forward
./target/release/expertvm workload prefill-batch
./target/release/gpu-profile probe bad-numa --bytes 1048576
cargo run -p llama-rust --example session
```

Next code change is PLAN systems depth (`expertvm schedule --prefix-cache`
landed as the trace-level prefix reuse slice). Phase 0 leftover is a Llama
NORM real-model fixture when a GGUF is on disk. Physical Phase 4 stays
parked. Do not add crates.io
runtime deps. Do not start Metal-in-crate on Linux. Do not invent a
`block_iq4_nl_4_4` dequant. Do not invent an arch. Do not list mixtral
or qwen3vlmoe as an accepted arch. Do not invent `$/M tokens`.
