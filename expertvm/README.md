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
                   expertvm schedule    open-loop batching; --interarrival-ns
                         │
                         ▼
                   expertvm bench       replay table + sim + schedule-all/1
                         │
                         ▼
                   expertvm ep          static EP vs GPU0 LRU (8-GPU profile)
                         │
                         ▼
                   expertvm remote      GPU0 compute vs remote-home RDMA fetch
                         │
                         ▼
                   expertvm kv          paged VMM KV working set (map live pages)
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
| `CachedStore` | Bounded LRU. **Leases** pin in-use experts until `release`. **`pin_hot`** is a sticky pin that survives `release` (`unpin_all` clears it). `prefetch(keys)` skips unknown keys. `is_resident` / `is_leased` / `is_pinned` / `is_held` / `phase` / `evict` / `take_victim` / `slots` / `pin_budget`. CPU `ExpertPhase` is Cold / Resident / Leased (fault-in is instant; leased **or** pinned is Leased; `evict` is immediate Cold). |
| `TieredStore` | Fast RAM LRU in front of slow RAM, a paging **file** (seek+read, not mmap), or synthetic bytes. Only `slots` [`ExpertParts`](crate::ExpertParts) live in the fast map. `WeightStorage::mmap` is parked. |
| `SimulatedGpuStore` | CachedStore + [`gpu-sim`](../gpu-sim). **Pinned** H2D on a copy stream onto the striped home (`expert_id % n_gpus`); GEMM waits that event (`--wait-value` is `cuStreamWaitValue64` on an 8-byte mailbox instead). [`SimulatedGpuStore::with_managed`](crate::SimulatedGpuStore::with_managed) uses `cudaMallocManaged` + ReadMostly + PreferredLocation + prefetch. [`SimulatedGpuStore::with_mapped`](crate::SimulatedGpuStore::with_mapped) is `cudaHostAllocMapped` (PCIe kernel, no H2D, `hbm_peak` 0; cache slots also cap at `host_pin_bytes / expert_bytes`). [`SimulatedGpuStore::with_vmm`](crate::SimulatedGpuStore::with_vmm) is `va_acquire` + pinned H2D; evict `va_release`s. [`SimulatedGpuStore::with_cfg`](crate::SimulatedGpuStore::with_cfg) adds `host_func`, blocking compute, host-sync alloc, mempool hold, `mempool_trim` (`cudaMemPoolTrimTo(0)` after `score`), `vmm_page`, pageable H2D, `host_register` (`cudaHostRegister` then DMA), `sync_memops` (`cuPointerSetAttribute` SyncMemops), `device_sync_memops` (`cudaSetDeviceFlags` SyncMemops), `host_register_mapped` (`cudaHostRegisterMapped` expert pages; implies mapped), `memcpy_batch` (`cudaMemcpyBatchAsync` sibling H2D prefetch; demand acquire stays sequential), `SetAccessedBy` / VMM `va_set_access` / mempool `pool_set_access` (managed, VMM, or pinned-async migrate/pin keep home residency, dest HBM 0), legacy NULL, compute stream priority, `seq_streams` (per-sequence copy streams; grouped GEMM stays on `StreamId(n_copy)`), `kv_sim` (Engine interned KV on this Sim; default off), `decode_priority` (decode GEMMs on a higher-priority compute stream; default off), `mem_sync_domain` (`cudaLaunchAttributeMemSyncDomain` on the decode stream; Default identity; Remote isolates leftover prefill fence tax), `launch_completion` (`cudaLaunchAttributeLaunchCompletionEvent` on grouped GEMMs; replica D2D waits kernel start), `programmatic_event` (`cudaLaunchAttributeProgrammaticEvent` on grouped GEMMs; replica D2D waits the PDL trigger), `stream_attach` (`cudaStreamAttachMemAsync` Single on managed experts; prefetch on compute), `managed_host` (`cudaMallocManaged` Host attach then Global on the copy stream), `prefetch_host` (`cudaMemPrefetchAsync` to host on managed evict; next miss prefetches the same alloc back), and `graph_update` (`cudaGraphExecUpdate` of a parked leaf), and `graph_set_params` (`cudaGraphExecKernelNodeSetParams` of a parked leaf), and `graph_clone` (`cudaGraphClone` before instantiate), and `graph_enable` (`cudaGraphNodeSetEnabled` skip extra walker combo children; store GEMM stays per-leaf), and `timing_events` (`cudaEventElapsedTime` on copy start/end), and `event_blocking_sync` (`cudaEventBlockingSync` on those copy events; implies timing; `synchronize_event` pays `host_sync_blocking_ns`; distinct from `--sync-policy blocking`), and `shareable` (POSIX-FD mempool IPC; imported sibling shares cache). `new` stays pinned/async for decode identity. A host-pinned staging alloc is created at construction for pinned/managed/VMM (mapped uses pageable staging so it does not steal mlock) and does not charge HBM. Prefetch is H2D or managed migrate without GEMM (`ExpertPhase::Transferring` until the copy event completes). `evict` is `ExpertPhase::Evicting` until the stream-ordered free completes (managed uses host-sync `free_sync` unless `--prefetch-host`; mapped is `free_host_pinned` unless `--host-register-mapped` (`host_unregister` then `free_host`); VMM is `va_release`; pinned `sync_alloc` is `free_sync`). H2D uses `Sim::alloc` (`cudaMallocAsync`) unless `sync_alloc` (`malloc` / `memcpy_sync`). After a drain or `synchronize_stream` on compute, an idle stream captures a per-page GEMM graph and later acquires `launch_graph`. Lease of Transferring/Cold/Evicting is refused. `pin_hot` waits the copy and this page's GEMM lease (or launch-completion on pinned replica D2D), then NVLink-replicates onto `(home + 1) % n_gpus` when `n_gpus >= 2` (D2D, dest prefetch when managed unless AccessedBy, VMM `va_set_access` without dest HBM when AccessedBy else map+D2D, mempool `pool_set_access` without dest HBM when AccessedBy else D2D, no-op when mapped). `migrate(key, dst)` waits that page (and replica) then D2D-moves onto `dst` (copy stream; dest compute waits the event; src HBM dropped) or, when managed, prefetches `dst` and `drop_managed_copy`s the source unless AccessedBy (retarget GEMM, page stays home); mapped only retargets GEMM; VMM maps dest then D2D and unmaps src unless AccessedBy (retarget GEMM, keep home physicals); pinned AccessedBy retargets GEMM and keeps home physicals. `place_hot` is fail-loud. `score()` is wall/HBM/bytes/`energy_uj`/`ns_per_token`. |
| `LiveStore` | Enum over Direct / Cached / Tiered / Simulated. Decode attaches this. Engine `pin_hot` then `place_hot` (`plan_placement`: D2D onto GPU0 vs leave on the striped home) when `n_gpus >= 2`. `place_hot` is fail-loud (`Result`). Managed/VMM waits that page's GEMM lease before replica prefetch / `drop_managed_copy` / `va_unmap` (does not `synchronize()` the whole device). `migrate` stays unconditional. |

`sim_replay` runs a policy through gpu-sim: pinned H2D on miss, grouped GEMM on
acquire, stream-ordered free on eviction. Timing comes from a
`HardwareProfile`, not from the policy. The clock is sampled after each
token (`ttft_ns`, mean `itl_ns`); a batch of sequences at the same token is
one sample. `--seq-streams` maps `sequence % n_streams` onto CUDA streams so
those copies can overlap. `--compute-slots N` (`N>=2`) is Hyper-Q occupancy
so independent sequence GEMMs on those streams overlap at full issue rate
(default profile `1` is exclusive). `--pdl` is programmatic dependent
launch: consecutive same-stream expert GEMMs may overlap after the
previous kernel's trigger when `--compute-slots` is `>=2` (illegal with
`--cooperative`). `--l2-persist` is `cudaLaunchAttributeAccessPolicyWindow`
over expert pages (persisting L2 after the first fill). `--l2-reset` is
`cudaCtxResetPersistingL2Cache` after each GEMM (implies `--l2-persist`; live;
cannot capture; a reused expert does not keep persisting L2). `--l2-fetch N` is
`cudaLimitMaxL2FetchGranularity` (`32`/`64`/`128`; implies `--l2-persist`;
access-policy windows must align). `--cluster N` is
`cudaLaunchAttributeClusterDimension` on grouped expert GEMMs: occupies
`min(N, compute_slots)` Hyper-Q slots (Hopper portable max 8; legal with
`--pdl` and `--cooperative`). `--preferred-cluster N` is
`cudaLaunchAttributePreferredClusterDimension`: occupancy uses that size
when it fits in `compute_slots`, else the required `--cluster` (needs
`--cluster`; must be a multiple of it; legal with `--pdl` and `--cooperative`). `--cluster-spread` is
`cudaLaunchAttributeClusterSchedulingPolicyPreference` Spread: occupies
every Hyper-Q slot even when `N` is smaller than `compute_slots` (no-op
without `--cluster` of at least 2). `--func-cluster-spread` is
`cudaFuncSetAttribute` ClusterSchedulingPolicyPreference Spread: launch
Default inherits that occupancy (distinct from `--cluster-spread`). `--cluster-must-set` is
`cudaFuncSetAttribute` ClusterDimMustBeSet (needs `--cluster`; occupancy
matches `--cluster`; SetAttribute is +1 ns). `--required-cluster N` is
`cudaFuncSetAttribute` RequiredClusterWidth (needs `--cluster`; must match
`--cluster`; occupancy matches `--cluster`; SetAttribute is +1 ns). `--max-shared` is
`cudaLaunchAttributePreferredSharedMemoryCarveout` MaxShared: occupies
every Hyper-Q slot. `--func-max-shared` is `cudaFuncSetAttribute`
PreferredSharedMemoryCarveout MaxShared: launch Default inherits that
occupancy. `--non-portable-cluster` is
`cudaFuncAttributeNonPortableClusterSizeAllowed` so `--cluster N` may
exceed portable size up to the SKU max (Hopper portable 8). `--sync-policy
auto|spin|yield|blocking` is `cudaLaunchAttributeSynchronizationPolicy` on
created streams (decode-stream ITL pays `host_sync_*_ns` when `--decode-priority`;
Auto tax 0). `--mem-sync-domain default|remote` is
`cudaLaunchAttributeMemSyncDomain` on the decode compute stream (prefill
stays Default; Remote isolates leftover prefill `same_domain_fence_permille`;
walker does not imply `--decode-priority`). `--mem-sync-map identity|collapse` is
`cudaLaunchAttributeMemSyncDomainMap` on that decode stream (collapse maps
remote→0 and restores leftover prefill fence tax; needs `--mem-sync-domain
remote`). `--shared-mem default|four|eight` is
`cudaLaunchAttributeSharedMemoryMode` on grouped expert GEMMs (Default never
scales duration; FourByte / EightByte scale by `1000 / shared_mem_*_permille`).
`--func-shared-mem default|four|eight` is `cudaFuncSetSharedMemConfig`: launch
Default inherits that duration scale (distinct from `--shared-mem`).
`--device-shared-mem default|four|eight` is `cudaDeviceSetSharedMemConfig`:
launch Default inherits when function config is also Default.
`--portable-cluster default|portable|non-portable` is
`cudaLaunchAttributePortableClusterSizeMode` on grouped expert GEMMs (Default
uses the function attribute; `portable` always refuses oversize; `non-portable`
allows up to the SKU max even when `--non-portable-cluster` is off).
`--optin-shared` is `cudaFuncAttributeMaxDynamicSharedMemorySize` to the SKU
opt-in max. `--dynamic-shared N` is `cudaLaunchKernel` `sharedMemBytes`
(`N` must be > 0). `--portable-shared default|portable|non-portable` is CUDA 13
`cudaLaunchAttributeSharedMemoryMode` (`cudaSharedMemoryMode`; Default uses the
function attribute; `portable` always refuses oversize; `non-portable` allows
up to the SKU opt-in even when `--optin-shared` is off).
`--nvlink-util` is `cudaLaunchAttributeNvlinkUtilCentricScheduling`: occupies
every Hyper-Q slot when the profile has NVLink (`8xh100`); without NVLink
occupancy is unchanged.
`--device-launch` is `cudaGraphInstantiateFlagDeviceLaunch` plus
`device_launch_graph` (illegal with `--graph-mem` / `--graph-auto-free` /
`--graph-update`). `--device-updatable` is
`cudaLaunchAttributeDeviceUpdatableKernelNode` so `--graph-set-params` keeps
the exec uploaded (illegal with `--graph-update`). `--kernel-priority N` is
`cudaLaunchAttributePriority` on grouped expert GEMMs (`None` inherits stream
create priority; `0` is a valid override).
`--mem-sync-domain remote` puts decode GEMMs on
`cudaLaunchMemSyncDomainRemote` (prefill stays Default) so leftover prefill
fence tax does not flush decode ITL. Default is Default. `--mem-sync-map
collapse` maps remote→0 on that decode stream so leftover prefill fence tax
returns (needs `--mem-sync-domain remote`). gpu-sim allreduce
tags Remote so a non-zero `same_domain_fence_permille` does not flush
expert compute behind communication. `--cooperative` is
`cudaLaunchCooperativeKernel`: GEMMs occupy every Hyper-Q slot, so
independent sequences cannot overlap even with `--compute-slots 2`.
`--decode-sms N` (`1..=1000`) is a
green-context SM fraction on every replay stream (compute-bound kernels
scale; memory-bound keep full HBM; default unset is a full chip). `--multicast`
is Hopper NVLS replica fanout (implies `--vmm`; illegal with `--accessed-by`
/ `--vmm-page`; needs an NVLink clique). `--sync-alloc` uses host-sync `cudaMalloc` /
`cudaMemcpy` / `cudaFree` on every miss (`Sim::malloc`); the default is
stream-ordered `alloc` so a miss can overlap other streams. `--mempool`
sets the default pool release threshold to `u64::MAX` so unused
`cudaMallocAsync` bytes stay in `cudaMemGetInfo` used until trim (vLLM);
reuse of a same-size page pays `pool_reuse_ns` instead of
`alloc_overhead_ns`. `--mempool-trim` is `cudaMemPoolTrimTo(0)` after
`score()` / walker finish: hold during the run, return unused cache at
idle (vLLM `cudaMemPoolTrimTo` at idle). Implies `--mempool`. Illegal
with `--sync-alloc`. Hits/misses and `hbm_peak` stay the same; token ITL
does not trim. Decode identity stays off (CUDA default threshold 0 already
returns on free). `--mempool-no-reuse` is
`cudaMemPoolReuseAllowOpportunistic=0`: leftover cache stays reserved and
the next miss is an OS alloc (extra HBM, `alloc_overhead_ns`). Implies
`--mempool`. Illegal with `--sync-alloc`. `--shareable` is POSIX-FD mempool IPC
(`cudaMemPoolExportToShareableHandle` / `ImportFromShareableHandle` /
`ExportPointer` / `ImportPointer`): a shareable pool is `cudaDeviceSetMemPool`'d
so `cudaMallocAsync` draws from it, and an imported sibling shares
live/cached (no extra HBM). Implies `--mempool`. Illegal with
`--sync-alloc` / mapped / managed / vmm. `--mapped` uses `cudaHostAllocMapped`: experts stay in
pinned host, kernels run over PCIe, HBM is not charged (`hbm_peak=0`).
That is the “do not move the expert” alternative to H2D. `--managed` uses
`cudaMallocManaged`, `cudaMemAdviseSetReadMostly`,
`cudaMemAdviseSetPreferredLocation` on the home GPU, and
`cudaMemPrefetchAsync` on miss: alloc is free of HBM, prefetch migrates
(and replicates if another GPU later prefetches the same page). A remote
kernel read can keep the page at the preferred GPU.
`--place replicas` uses that prefetch onto dest GPUs; dest eviction
`drop_managed_copy`s one GPU without freeing the allocation. `--place replicas
--vmm` maps dest then D2D (`va_unmap_range` on dest eviction). `--multicast`
(implies `--vmm`) replaces that dest D2D with one Hopper NVLS kernel so N
dests pay one fabric hop on compute. Needs NVLink. `--accessed-by`
is `cudaMemAdviseSetAccessedBy` on every GPU at a managed fill, or
`cuMemSetAccess` PROT_READ at a VMM fill, or `cudaMemPoolSetAccess` ReadWrite
on every default pool for pinned `cudaMallocAsync`: dest GEMMs
read without migrating or charging dest HBM, and replica prefetch / VMM dest
map+D2D / pinned D2D is skipped (`cudaMalloc` / `--sync-alloc` still D2Ds). `--vmm` uses
`va_acquire` (remap an idle VA, else reserve+map) then pinned H2D into that
VA (evict `va_release`s the pointer so the next miss skips reserve).
`--vmm-page N` maps each expert in `N`-byte physicals (`va_acquire_paged`,
vLLM KV-block analog; implies `--vmm`). `expertvm kv` reserves per-sequence
KV VAs and `cuMemCreate`s interned pages (`kernel_bufs` plus H2D or
`--fill memset`; `--sequences N` maps the same physical into N VAs;
`--row-width W --pitch P` is a 2D miss fill (payload `W *
height`, not pitch padding): `memcpy_2d_async` (`cudaMemcpy2DAsync`) with `--fill h2d`,
`memset_2d_async` (`cudaMemset2DAsync`) with `--fill memset`. Peak HBM is unique pages, not the reservation. That is **simulated
VMM**, not the reference engine's paged KV (`Llama::new_paged_cache` /
`gguf_gemv serve --kv-page`, interned decode blocks).
`--host-func` enqueues `cudaLaunchHostFunc` after each event's GEMMs
(`host_func_ns`; other streams can still compute). `--blocking-streams`
marks created seq-streams as `cudaStreamCreate` (they serialize with the
default/null stream). Default is `cudaStreamNonBlocking`, so `--seq-streams`
can overlap a miss with another sequence's GEMM. Pair with `--seq-streams`
or it is a no-op (`n_streams = 1`). A
profile `host_pin_bytes` cap pages `min(capacity, pin / expert_bytes)`
mapped experts (`PinOom` only when even one expert cannot lock). `SimulatedGpuStore::with_mapped`
uses the same occupancy cap (pageable staging so construction does not steal mlock). `SimulatedGpuStore::new`
stays on the async H2D path with CUDA's default threshold (`0`), non-blocking
streams, and `cudaEventDisableTiming` copy events. `with_cfg` opts into the
`sim` knobs (`host_func`, blocking compute, `sync_alloc`, mempool, `mempool_trim`, `mempool_no_reuse`, `shareable`, `vmm_page`, pageable H2D, `host_register`, `host_register_mapped`, `sync_memops`, `device_sync_memops`, `device_sync_policy`, `memcpy_batch`, AccessedBy, legacy NULL, stream priority, `graph_update`, `graph_set_params`, `graph_clone`, `graph_build`, `graph_piecewise`, `graph_enable`, `graph_mem`, `graph_auto_free`, `timing_events`, `event_blocking_sync`, `cooperative`, `mem_sync_domain`, `mem_sync_collapse`, `launch_completion`, `programmatic_event`, `stream_attach`, `managed_host`, `prefetch_host`, `wait_value`). `--host-register` is `cudaHostRegister` on pageable staging so miss H2D is pinned DMA (implies `--pageable`; illegal with `--mapped` / `--managed`). `--host-register-mapped` is `cudaHostRegisterMapped` on expert pages (`alloc_host` then pin+map; implies `--mapped`; illegal with `--host-register`; identity mapped stays `cudaHostAllocMapped`). `--sync-memops` is `cuPointerSetAttribute` SyncMemops on miss device pages so H2D / managed prefetch is host-synchronous (illegal with `--mapped` / `--memcpy-batch`; identity stays async pinned H2D). `--device-sync-memops` is `cudaSetDeviceFlags(cudaDeviceSyncMemops)` so every memcpy/memset on that GPU is host-synchronous (illegal with `--mapped` / `--memcpy-batch`; identity stays async pinned H2D; distinct from per-page `--sync-memops`). `--device-sync-policy` is `cudaSetDeviceFlags` SCHEDULE_* so Auto streams inherit host-wait tax (explicit `--sync-policy` wins; ORs with `--device-sync-memops`). `--event-blocking-sync` is `cudaEventBlockingSync` on copy start/end events (implies `--timing-events`; `synchronize_event` pays `host_sync_blocking_ns`; distinct from `--sync-policy blocking`). `--func-max-shared` is `cudaFuncSetAttribute` PreferredSharedMemoryCarveout MaxShared (launch Default inherits; occupies every Hyper-Q slot; distinct from launch-attribute `--max-shared`). `--func-shared-mem` is `cudaFuncSetSharedMemConfig` (launch Default inherits bank-width duration; distinct from launch-attribute `--shared-mem`). `--l2-reset` is `cudaCtxResetPersistingL2Cache` after each GEMM (implies `--l2-persist`; live; cannot capture). `--memcpy-batch` is `cudaMemcpyBatchAsync` for a multi-expert pinned/VMM prefetch on one stream (sibling copies share a stream-order snapshot; demand acquire stays sequential). Illegal with `--pageable`, `--sync-alloc`, `--mapped`, or `--managed`. `--graph-enable` is `cudaGraphNodeSetEnabled` on a wide combo parent (implies `--cuda-graphs`; illegal with `--device-launch`). `--mem-sync-domain remote` is `cudaLaunchAttributeMemSyncDomain` on the decode stream (prefill stays Default). `--mem-sync-map collapse` is `cudaLaunchAttributeMemSyncDomainMap` `{default: 0, remote: 0}` on that stream (needs `--mem-sync-domain remote`; restores leftover prefill fence tax). `--launch-completion` is `cudaLaunchAttributeLaunchCompletionEvent` on grouped GEMMs (replica D2D waits kernel start; illegal with `--device-launch`). `--programmatic-event` is `cudaLaunchAttributeProgrammaticEvent` on those GEMMs (replica D2D waits the PDL trigger; illegal with `--device-launch`). `--stream-attach` is `cudaStreamAttachMemAsync` Single on managed experts (prefetch on compute; implies `--managed`; illegal with `--seq-streams`). `--managed-host` is `cudaMallocManaged(..., cudaMemAttachHost)` then Global attach on the copy stream (implies `--managed`; leftover GEMM still overlaps that prefetch unless `--stream-attach`). `--prefetch-host` is `cudaMemPrefetchAsync` to host on managed evict (implies `--managed`; next miss prefetches the same alloc back). `--wait-value` is `cuStreamWaitValue64` / `WriteValue64` after H2D (8-byte device mailbox; decode identity stays events). `--mempool-trim` is `cudaMemPoolTrimTo(0)` after score (implies `--mempool`; illegal with `--sync-alloc`; token ITL does not trim). `--max-batch N` admits N sequences per engine
iteration at a token (`0` = the whole token) and still samples TTFT once.
`--cuda-graphs` captures a leaf GEMM per resident expert alloc, instantiates
it, then a parent of `launch_graph` child nodes for a grouped launch
(combos reuse leaves when one expert is evicted). `--graph-update` parks
that leaf on evict and `cudaGraphExecUpdate`s the next miss on the same
`(device, stream)` (`graph_update_ns` instead of instantiate). `--graph-set-params`
parks that leaf and `cudaGraphExecKernelNodeSetParams` the unique kernel
(`graph_set_params_ns`; no second capture; legal with `--graph-mem`; also
`cudaGraphExecMemcpyNodeSetParams` / `cudaGraphExecMemsetNodeSetParams` a
unique memcpy or memset if present; combo parents use
`cudaGraphExecChildGraphNodeSetParams`). `--graph-clone`
clones a leaf capture before instantiate (`graph_clone_ns`; the src is
destroyed). `--graph-build` is `cudaGraphCreate` / `cudaGraphAdd*` instead of
stream capture (no idle-stream wait; implies `--cuda-graphs` on the walker;
combo parents are `graph_add_child` of instantiated leaves with no
`graph_add_dependencies` edge, so independent expert GEMMs may Hyper-Q
overlap). `--graph-piecewise` is `cudaStreamBeginCaptureToGraph` combo
parents (each leaf is an extra root on one parent; same Hyper-Q overlap;
illegal with `--graph-build`). Combo overlap stays separate capture
sessions: `cudaStreamUpdateCaptureDependencies` extra deps are additive
with stream-order, so they cannot split same-stream children. `--graph-enable`
is `cudaGraphNodeSetEnabled` on a wide combo parent so a later token that
GEMMs a subset skips extra child graphs instead of instantiating a new
parent (implies `--cuda-graphs`; illegal with `--device-launch`; store GEMM
stays per-leaf). `--graph-mem`
records a scratch `cudaMallocAsync` + free in each leaf GEMM graph (HBM peak
includes the workspace; `--graph-update` is skipped). `--graph-auto-free` is
the same scratch without a matching free, instantiated
`cudaGraphInstantiateFlagAutoFreeOnLaunch` (relaunch recharges HBM; not with
`--graph-mem`). Capture waits with
`synchronize_stream` so the compute stream is idle (CUDA). First launch of
a new graph pays `graph_instantiate_ns` then `graph_upload_ns`, then
`graph_launch_ns` once per launch. `--plan-window
N` runs [`plan_window`](crate::plan_window) Stay vs Fetch before prefetch (Stay
does not evict a resident working set). Replay reports `prefetch_hits` /
`prefetch_waste`. `schedule_replay` / `expertvm schedule` is open-loop
continuous batching: sequences arrive at `sequence * interarrival_ns`,
FCFS into a running set of `--max-batch` (`0` = unlimited), one next
chunk (layer-major) per engine step, finished sequences retire,
[`gpu_sim::Sim::idle_until`] waits for the next arrival. A chunk is the
whole next token unless `--prefill-chunk N` limits a sequence's first
token to N layer-events so a short decode is not stuck behind a long
prefill. `--decode-first` holds leftover prefill while any running
sequence is already in decode. `--decode-priority` runs decode GEMMs on a
higher-priority compute stream and samples ITL from that stream so leftover
prefill does not inflate it (implies `--stream-priority`; does not imply
from `--decode-sms`). `llama-rust` `EngineCfg::decode_first` /
`gguf_gemv engine --decode-first` is the same hold on real KV, not this
trace walker. `--slo-reject` drops a waiter whose queue
wait already meets `--ttft-slo-ns` (`rejected=` on the schedule line).
`llama-rust` `EngineCfg::slo_reject` / `gguf_gemv engine --slo-reject`
is the same drop on real KV waiters, using the SimulatedGpuStore clock.
`--itl-slo-ns` counts later-token gaps (`itl_slo_miss=`; does not drop);
`llama-rust` `EngineCfg::itl_slo_ns` / `gguf_gemv engine --itl-slo-ns`
is the same later-token count on real KV. `gguf_gemv engine --bench`
prints [`report`](crate::report) on the Engine's batched MoE traces (same
shape as `infer-bench trace` / `expertvm bench`; `--capacity` is the replay
cache). `gguf_gemv engine --prefetch`
/ `--plan-window` / `--plan-threshold` (and `serve --engine`) is the
same Stay vs Fetch predictor on real decode; upcoming keys are the
online predicted list, not this walker's JSONL future window.
`--expert-sim` captures
per-page GEMM graphs (`graph_launches=`; `--cuda-graphs` documents that).
`--graph-update` / `--graph-set-params` / `--graph-clone` / `--graph-build` / `--graph-piecewise` / `--graph-enable` / `--graph-mem` / `--graph-auto-free` / `--graph-mem-trim` / `--timing-events` / `--event-blocking-sync` are `GpuStoreCfg`
on the Engine store. `--mapped` / `--managed` / `--vmm` select `GpuFill`
(`gguf_gemv engine --expert-sim --managed`). `--host-func` /
`--blocking-streams` / `--sync-alloc` / `--mempool` / `--mempool-trim` / `--mempool-no-reuse` / `--shareable` / `--vmm-page` /
`--pageable` / `--host-register` / `--host-register-mapped` / `--sync-memops` / `--device-sync-memops` / `--memcpy-batch` / `--accessed-by` / `--legacy-null` / `--stream-priority` /
`--seq-streams` / `--kv-sim` / `--kv-bytes` / `--decode-priority` /
`--cooperative` / `--pdl` / `--l2-persist` / `--l2-reset` / `--l2-fetch` / `--cluster` / `--preferred-cluster` / `--cluster-spread` / `--func-cluster-spread` / `--cluster-must-set` / `--required-cluster` / `--max-shared` / `--func-max-shared` / `--non-portable-cluster` / `--sync-policy` / `--device-sync-policy` / `--event-blocking-sync` / `--mem-sync-domain` / `--mem-sync-map` / `--shared-mem` / `--func-shared-mem` / `--device-shared-mem` / `--portable-cluster` / `--optin-shared` / `--dynamic-shared` / `--portable-shared` / `--nvlink-util` / `--device-launch` / `--device-updatable` / `--kernel-priority` / `--launch-completion` / `--programmatic-event` / `--wait-value` / `--compute-slots` / `--decode-sms` / `--multicast` / `--shareable` are `GpuStoreCfg` knobs on `gguf_gemv engine`.
`expertvm sim` / `schedule` / `store` take `--compute-slots` / `--decode-sms`
/ `--decode-priority` / `--cooperative` / `--pdl` / `--l2-persist` / `--l2-reset` / `--l2-fetch` / `--cluster` / `--preferred-cluster` / `--cluster-spread` / `--func-cluster-spread` / `--cluster-must-set` / `--required-cluster` / `--max-shared` / `--func-max-shared` / `--non-portable-cluster` / `--sync-policy` / `--device-sync-policy` / `--mem-sync-domain` / `--mem-sync-map` / `--shared-mem` / `--func-shared-mem` / `--device-shared-mem` / `--portable-cluster` / `--optin-shared` / `--dynamic-shared` / `--portable-shared` / `--nvlink-util` / `--device-launch` / `--device-updatable` / `--kernel-priority` / `--launch-completion` / `--programmatic-event` / `--wait-value` / `--multicast` / `--shareable` (Hyper-Q occupancy, green-context SM fraction,
decode-stream ITL, exclusive cooperative GEMMs, same-stream PDL overlap, Hopper cluster occupancy / preferred dim / Spread scheduling / ClusterDimMustBeSet / RequiredClusterWidth / non-portable size, NVLS replica fanout, and POSIX-FD mempool IPC on the trace walker). Walker `--decode-sms` does **not**
imply `--decode-priority` (token 0 is prefill). Engine `--mem-sync-domain remote`
implies `--decode-priority`. `--decode-priority` implies
`--stream-priority` so leftover prefill does not inflate decode ITL.
`gguf_gemv engine --expert-sim --kv-sim` maps interned KV onto that Sim
(distinct from `expertvm kv`; `--kv-bytes` overrides intern geometry).
`gguf_gemv engine --expert-sim --decode-priority` runs decode GEMMs on a
second compute stream at higher CUDA priority than leftover prefill.
Token-boundary ITL samples that decode stream (leftover prefill stays in
flight). `--compute-slots N` (`N>=2`) is Hyper-Q occupancy so those two
streams' GEMMs overlap at full issue rate. `--pdl` lets consecutive
same-stream expert GEMMs overlap after the previous kernel's programmatic
trigger (needs `--compute-slots` >= 2; illegal with `--cooperative`).
`--l2-persist` keeps reused expert pages in persisting L2. `--l2-reset` is
`cudaCtxResetPersistingL2Cache` after each GEMM (implies `--l2-persist`; a
reused expert is cold). `--l2-fetch N` is
`cudaLimitMaxL2FetchGranularity` (`32`/`64`/`128`; implies `--l2-persist`;
windows must align). `--cluster N`
is a Hopper thread-block cluster so leftover kernels cannot overlap a
launch that fills Hyper-Q. `--preferred-cluster N` occupies the preferred
size when it fits in `--compute-slots` (needs `--cluster`; must be a
multiple of it). `--cluster-spread` occupies every Hyper-Q slot
even when `N` is smaller than `--compute-slots` (no-op without `--cluster`
of at least 2). `--func-cluster-spread` occupies every Hyper-Q slot via
function Spread policy (launch Default inherits). `--cluster-must-set` is
`cudaFuncAttributeClusterDimMustBeSet` (needs `--cluster`; occupancy matches
`--cluster`). `--required-cluster N` is
`cudaFuncAttributeRequiredClusterWidth` (needs `--cluster`; must match
`--cluster`; occupancy matches `--cluster`). `--max-shared` occupies every Hyper-Q slot via MaxShared
carveout. `--func-max-shared` occupies every Hyper-Q slot via function
MaxShared carveout (launch Default inherits). `--non-portable-cluster` allows `--cluster N` above portable size
up to the SKU max. `--sync-policy auto|spin|yield|blocking` is stream host-wait
(Auto tax 0). `--device-sync-policy auto|spin|yield|blocking` is device host-wait
(`cudaSetDeviceFlags` SCHEDULE_*; Auto streams inherit; explicit `--sync-policy`
wins). `--event-blocking-sync` is `cudaEventBlockingSync` on copy
start/end events (implies `--timing-events`; `synchronize_event` pays
`host_sync_blocking_ns`; distinct from `--sync-policy blocking`). `--mem-sync-domain default|remote` is decode-stream
`cudaLaunchAttributeMemSyncDomain` (Remote isolates leftover prefill fence
tax; engine implies `--decode-priority`). `--mem-sync-map identity|collapse` is
decode-stream `cudaLaunchAttributeMemSyncDomainMap` (collapse maps remote→0
and restores leftover prefill fence tax; needs `--mem-sync-domain remote`). `--shared-mem default|four|eight` is kernel-node bank width
(Default never scales). `--func-shared-mem default|four|eight` is function bank
width (launch Default inherits; distinct from `--shared-mem`). `--device-shared-mem default|four|eight` is
device bank width (launch Default inherits when function is Default). `--portable-cluster default|portable|non-portable` is
launch-time portable cluster mode (Default uses the function attribute).
`--optin-shared` is MaxDynamicSharedMemorySize to the SKU opt-in.
`--dynamic-shared N` is `sharedMemBytes` (`N` > 0). `--portable-shared
default|portable|non-portable` is CUDA 13 portable-shared mode (Default uses the
function attribute). `--nvlink-util` occupies every Hyper-Q slot when the
profile has NVLink (no-op occupancy without NVLink). `--device-launch` /
`--device-updatable` instantiate leaf GEMM graphs for `device_launch_graph`
and keep the exec uploaded after set-params. `--kernel-priority N` is
`cudaLaunchAttributePriority` on those GEMMs (`None` inherits stream create
priority). `--launch-completion` is `cudaLaunchAttributeLaunchCompletionEvent`
on grouped GEMMs (store replica D2D waits kernel start; illegal with
`--device-launch`). `--programmatic-event` is
`cudaLaunchAttributeProgrammaticEvent` on those GEMMs (store replica D2D
waits the PDL trigger; illegal with `--device-launch`). `--wait-value` is `cuStreamWaitValue64` / `WriteValue64`
after H2D (8-byte `cudaMallocAsync` mailbox, copy stream waited before
H2D so compute wait is resident during DMA; decode identity stays events).
`--mempool-trim` is `cudaMemPoolTrimTo(0)` after `score()` / walker finish
(implies `--mempool`; illegal with `--sync-alloc`; token ITL does not trim).
`--cooperative` is
`cudaLaunchCooperativeKernel`: those GEMMs occupy every Hyper-Q slot, so
leftover prefill cannot overlap even with `--compute-slots 2`. `--decode-sms N` (`1..=1000`)
is a green-context SM fraction on the decode stream (leftover prefill gets
the remainder; implies `--decode-priority`). Default `--expert-sim` keeps
one compute stream, exclusive compute (`compute_slots=1`), a full chip of
SMs, and a full-device clock sample.
`gguf_gemv engine --expert-sim --seq-streams` is the real-KV analog of
`expertvm sim --seq-streams`: per-sequence copy streams, grouped GEMM
on one compute stream. `--prefix-cache` skips GPU work for a token whose JSONL `"p"` hash
already completed on another sequence (`prefix_hits=` on the schedule
line). `"p"` is `prefix_hash` of the token ids, not a prompt class.
That is not the engine's KV prefix cache: `llama-rust` `Llama::prompt` /
`Session::prompt` reuses real K/V for a matching token prefix.
TTFT is from
arrival; `--ttft-slo-ns` / `--itl-slo-ns` count misses. `queue_ns` is mean
first-token wait (`iteration_start - arrival`) so a tight `max_batch`
shows queueing separately from GPU service. The cache walker
is demand paging (no JSONL future leak). [`schedule_placed`] H2Ds a miss onto the expert's [`PlaceMap`](crate::PlaceMap)
home (`--place striped|colocated|replicas`) so a wide token can use every GPU's copy
engines. `--capacity` is slots **on that home**, but `restrict_hbm` still
evicts when fewer pages fit. [`schedule_replay`] is GPU0. [`schedule_remote`] / `--place remote`
keeps compute on GPU0: a miss H2Ds onto the striped home, then
`plan_placement` either D2Ds weights onto GPU0 or ships `--activation-bytes`
to home (pair with `--profile 2node-rdma`). `--prefetch copy-forward|markov|both`
fills remote home pages without mixing local handles into the remote map
(no GEMM until demand). Planner helpers: `copy_forward`,
`hot_keys`, `plan_keys`, `plan_window`, `plan_placement` (move weights vs dispatch
activations), `Markov` / `Prefetch` (lookback-2 `P(to|from, from_prev)` with
order-1 backoff). `colocated` keeps co-fired experts
on one GPU; `with_hot_replicas` copies hot keys to a second GPU and those
replicas occupy dest `--capacity` (evict frees replica HBM).
`compare_ep` / `sim_static_ep` place each expert on
`home_gpu` (`expert_id % n_gpus`) with no eviction. LRU-on-GPU0 can
survive restricted HBM by evicting; static EP OOMs if a home GPU cannot
hold its working set. On `8xh100`, a wide token's H2Ds run on eight PCIe
roots and beat serial GPU0 copies of the same payload.
`sim_remote_home` keeps compute on GPU0. A miss does pinned H2D onto the
home GPU, then `plan_placement` (online reuse, no future leak) either
D2Ds the expert weights onto GPU0 or ships a small activation payload to
home and GEMMs there. `--managed --place remote` prefetches the expert
onto home (`SetPreferredLocation`) and GEMMs on GPU0 as a remote read
(weights are kernel reads, so the page stays on home). Decode acquires then **leases** each routed expert
for the GEMV and releases before the next (so `slots < top-k` still
works). `SimulatedGpuStore` holds a host-pinned staging alloc that does
not count toward HBM. `HardwareProfile::restrict_hbm` is the knob. `topology_suite` /
`probe_topology` compare H2D and P2P costs across named meshes (PCIe P2P,
NVLink, bad NUMA, RDMA, asymmetric). `expertvm bench` on a multi-sequence
trace prints `schedule-all` vs `schedule-1` (open-loop running set of
unlimited vs 1) and `schedule-decode-first` when a mixed prefill/decode
trace has a wide first token. Multi-sequence traces with `"p"` also print
`schedule-prefix`. Multi-GPU profiles also print
`schedule-gpu0` vs `schedule-striped` vs `schedule-remote`. `SimulatedGpuStore` can inject GPU
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
expertvm sim      trace.jsonl --capacity 8 --seq-streams --sync-alloc
expertvm sim      trace.jsonl --capacity 8 --mempool
expertvm sim      trace.jsonl --capacity 8 --mempool-trim
expertvm sim      trace.jsonl --capacity 8 --mempool-no-reuse
expertvm sim      trace.jsonl --capacity 8 --shareable
expertvm sim      trace.jsonl --capacity 8 --mapped
expertvm sim      trace.jsonl --capacity 8 --managed
expertvm sim      trace.jsonl --capacity 8 --vmm
expertvm sim      trace.jsonl --capacity 8 --vmm-page 2097152
expertvm sim      trace.jsonl --capacity 8 --host-func
expertvm sim      trace.jsonl --capacity 8 --seq-streams --blocking-streams
expertvm sim      trace.jsonl --capacity 8 --prefetch copy-forward --plan-window 8 --cuda-graphs
expertvm sim      trace.jsonl --capacity 1 --cuda-graphs --graph-update
expertvm sim      trace.jsonl --capacity 1 --graph-set-params
expertvm sim      trace.jsonl --capacity 1 --cuda-graphs --graph-clone
expertvm sim      trace.jsonl --capacity 1 --graph-build
expertvm sim      trace.jsonl --capacity 1 --graph-piecewise
expertvm sim      trace.jsonl --capacity 1 --graph-mem
expertvm sim      trace.jsonl --capacity 1 --graph-mem --graph-mem-trim
expertvm sim      trace.jsonl --capacity 1 --graph-auto-free
expertvm sim      trace.jsonl --capacity 8 --seq-streams --max-batch 2
expertvm schedule trace.jsonl --capacity 8 --max-batch 2 --interarrival-ns 1000000 --ttft-slo-ns 20000000
expertvm schedule trace.jsonl --capacity 8 --place striped --profile 8xh100 --expert-bytes 1048576
expertvm schedule trace.jsonl --capacity 8 --place replicas --profile 8xh100 --expert-bytes 1048576
expertvm schedule trace.jsonl --capacity 8 --place remote --profile 2node-rdma --expert-bytes 1048576 --prefetch copy-forward
expertvm schedule trace.jsonl --capacity 8 --prefill-chunk 1 --seq-streams --decode-first
expertvm schedule trace.jsonl --capacity 8 --max-batch 1 --ttft-slo-ns 1 --slo-reject
expertvm schedule trace.jsonl --capacity 8 --max-batch 1 --prefix-cache
expertvm place    trace.jsonl --gpus 8 --hot-pt 200
expertvm bench    trace.jsonl --capacity 8 --profile h100
expertvm bench    adversarial --tokens 64 --experts 16 --capacity 2 --profile cheap
expertvm workload thrash --tokens 64 --experts 16 --capacity 2
expertvm workload batch --tokens 32 --experts 16 --capacity 4
expertvm workload batch-1 --tokens 32 --experts 16 --capacity 4
expertvm workload batch-128 --tokens 8 --experts 16 --capacity 4
expertvm workload prefill-batch --tokens 8 --experts 16 --capacity 4
expertvm workload shared-prefix --tokens 8 --experts 16 --capacity 4
expertvm topology --bytes 1048576
expertvm ep       trace.jsonl --capacity 8 --expert-bytes 1048576 --profile 8xh100
expertvm ep       trace.jsonl --hbm-bytes 4096 --profile 8xh100
expertvm remote   trace.jsonl --expert-bytes 1048576 --profile 2node-rdma
expertvm remote   trace.jsonl --expert-bytes 1048576 --activation-bytes 128
expertvm kv       --pages 8 --page-bytes 4096 --capacity 2 --tokens 64
expertvm kv       --pages 8 --capacity 2 --fill memset
expertvm kv       --pages 8 --capacity 2 --row-width 256 --pitch 512
expertvm kv       --pages 8 --capacity 2 --sequences 2
expertvm store    trace.jsonl --capacity 2 --expert-bytes 4096 --profile h100
expertvm store    trace.jsonl --capacity 2 --managed --host-func
expertvm store    trace.jsonl --capacity 1 --prefetch markov
expertvm store    trace.jsonl --capacity 1 --graph-update
expertvm store    trace.jsonl --capacity 1 --graph-set-params
expertvm store    trace.jsonl --capacity 1 --graph-clone
expertvm store    trace.jsonl --capacity 1 --graph-build
expertvm store    trace.jsonl --capacity 1 --graph-piecewise
expertvm store    trace.jsonl --capacity 1 --graph-mem
expertvm store    trace.jsonl --capacity 1 --graph-auto-free
expertvm store    trace.jsonl --capacity 1 --timing-events
expertvm store    trace.jsonl --capacity 1 --event-blocking-sync
expertvm store    trace.jsonl --capacity 2 --managed --accessed-by --profile 2node-rdma
expertvm store    trace.jsonl --capacity 2 --vmm --accessed-by --profile 8xh100
expertvm store    trace.jsonl --capacity 2 --accessed-by --profile 8xh100
expertvm store    trace.jsonl --capacity 2 --legacy-null
expertvm sim      trace.jsonl --capacity 2 --seq-streams --stream-priority
expertvm sim      trace.jsonl --capacity 2 --seq-streams --compute-slots 2 --cooperative
expertvm sim      trace.jsonl --capacity 2 --compute-slots 2 --pdl
expertvm sim      trace.jsonl --capacity 2 --l2-persist
expertvm sim      trace.jsonl --capacity 2 --l2-reset
expertvm sim      trace.jsonl --capacity 2 --l2-fetch 32
expertvm sim      trace.jsonl --capacity 2 --func-shared-mem eight
expertvm sim      trace.jsonl --capacity 2 --device-shared-mem eight
expertvm sim      trace.jsonl --capacity 2 --seq-streams --compute-slots 2 --cluster 2
expertvm sim      trace.jsonl --capacity 2 --seq-streams --compute-slots 4 --cluster 2 --preferred-cluster 4
expertvm sim      trace.jsonl --capacity 2 --seq-streams --compute-slots 4 --cluster 2 --cluster-spread
expertvm sim      trace.jsonl --capacity 2 --seq-streams --compute-slots 4 --cluster 2 --func-cluster-spread
expertvm sim      trace.jsonl --capacity 2 --seq-streams --compute-slots 4 --cluster 2 --cluster-must-set
expertvm sim      trace.jsonl --capacity 2 --seq-streams --compute-slots 4 --cluster 2 --required-cluster 2
expertvm sim      trace.jsonl --capacity 2 --seq-streams --compute-slots 2 --decode-priority --mem-sync-domain remote --mem-sync-map collapse
expertvm sim      trace.jsonl --capacity 2 --seq-streams --compute-slots 2 --max-shared
expertvm sim      trace.jsonl --capacity 2 --seq-streams --compute-slots 2 --func-max-shared
expertvm schedule trace.jsonl --capacity 8 --place replicas --multicast --profile 8xh100
expertvm schedule trace.jsonl --capacity 8 --prefill-chunk 1 --decode-priority --compute-slots 2
expertvm sim      trace.jsonl --capacity 2 --managed --accessed-by --profile 2xh100-pcie
gpu-profile probe 2xh100-pcie --bytes 1048576
```

Traces are produced by `gguf_gemv trace`:

```
gguf_gemv write-tiny-qwen3moe tiny-qwen3moe.gguf
gguf_gemv trace tiny-qwen3moe.gguf -p ab -n 8 --out trace.jsonl
gguf_gemv write-tiny-qwen3moe-2layer tiny-qwen3moe-2layer.gguf
gguf_gemv trace tiny-qwen3moe-2layer.gguf -p ab -n 8 --out /tmp/tiny-qwen3moe-2layer.jsonl
expertvm analyze /tmp/tiny-qwen3moe-2layer.jsonl
expertvm schedule tests/traces/tiny-qwen3moe-2layer.jsonl --capacity 4 --prefetch copy-forward
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
