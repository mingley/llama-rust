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
| `SimulatedGpuStore` | CachedStore + [`gpu-sim`](../gpu-sim). **Pinned** H2D on a copy stream onto the striped home (`expert_id % n_gpus`); GEMM waits that event (`--wait-value` is `cuStreamWaitValue64` on an 8-byte mailbox instead). [`SimulatedGpuStore::with_managed`](crate::SimulatedGpuStore::with_managed) uses `cudaMallocManaged` + ReadMostly + PreferredLocation + prefetch. [`SimulatedGpuStore::with_mapped`](crate::SimulatedGpuStore::with_mapped) is `cudaHostAllocMapped` (PCIe kernel, no H2D, `hbm_peak` 0; cache slots also cap at `host_pin_bytes / expert_bytes`). [`SimulatedGpuStore::with_vmm`](crate::SimulatedGpuStore::with_vmm) is `va_acquire` + pinned H2D; evict `va_release`s. [`SimulatedGpuStore::with_cfg`](crate::SimulatedGpuStore::with_cfg) adds `host_func`, blocking compute, host-sync alloc, mempool hold, `mempool_trim` (`cudaMemPoolTrimTo(0)` after `score`), `vmm_page`, pageable H2D, `host_register` (`cudaHostRegister` then DMA), `host_unregister` (`cudaHostUnregister` after miss DMA), `sync_memops` (`cuPointerSetAttribute` SyncMemops), `device_sync_memops` (`cudaSetDeviceFlags` SyncMemops), `host_register_mapped` (`cudaHostRegisterMapped` expert pages; implies mapped), `memcpy_batch` (`cudaMemcpyBatchAsync` sibling H2D prefetch; demand acquire stays sequential), `memcpy_during` (`cudaMemcpySrcAccessOrderDuringApiCall` on that batch; the API waits those copies), `memcpy_any` (`cudaMemcpySrcAccessOrderAny`; empty deps; no API wait), `memcpy_attr` (`cudaMemcpyWithAttributesAsync` DuringApiCall on demand H2D; the API waits that copy), `d2h_evict` (`cudaMemcpyAsync` Device→HostPinned before pinned/VMM LRU free), `d2h_pageable` (`cudaMemcpyAsync` Device→Host pageable bounce-buffer before that free; host-sync; implies pageable), `SetAccessedBy` / VMM `va_set_access` / mempool `pool_set_access` (managed, VMM, or pinned-async migrate/pin keep home residency, dest HBM 0), legacy NULL, compute stream priority, `seq_streams` (per-sequence copy streams; grouped GEMM stays on `StreamId(n_copy)`), `kv_sim` (Engine interned KV on this Sim; default off), `decode_priority` (decode GEMMs on a higher-priority compute stream; default off), `mem_sync_domain` (`cudaLaunchAttributeMemSyncDomain` on the decode stream; Default identity; Remote isolates leftover prefill fence tax), `launch_completion` (`cudaLaunchAttributeLaunchCompletionEvent` on grouped GEMMs; replica D2D waits kernel start), `programmatic_event` (`cudaLaunchAttributeProgrammaticEvent` on grouped GEMMs; replica D2D waits the PDL trigger), `stream_attach` (`cudaStreamAttachMemAsync` Single on managed experts; prefetch on compute), `managed_host` (`cudaMallocManaged` Host attach then Global on the copy stream), `prefetch_host` (`cudaMemPrefetchAsync` to host on managed evict; next miss prefetches the same alloc back), and `graph_update` (`cudaGraphExecUpdate` of a parked leaf), and `graph_set_params` (`cudaGraphExecKernelNodeSetParams` of a parked leaf), and `graph_clone` (`cudaGraphClone` before instantiate), and `graph_enable` (`cudaGraphNodeSetEnabled` skip extra walker combo children; store GEMM stays per-leaf), and `timing_events` (`cudaEventElapsedTime` on copy start/end), and `event_blocking_sync` (`cudaEventBlockingSync` on those copy events; implies timing; `synchronize_event` pays `host_sync_blocking_ns`; distinct from `--sync-policy blocking`), and `shareable` (POSIX-FD mempool IPC; imported sibling shares cache), and `ipc` (`cudaIpcGetMemHandle` / `OpenMemHandle` of each miss `cudaMalloc`; alias shares source HBM; close before free; implies `sync_alloc`), and `share_ptr` (`cudaMemPoolExportPointer` / `ImportPointer` of each miss `cudaMallocAsync`; alias shares source HBM; `cudaFreeAsync` import before source; implies `shareable`). `new` stays pinned/async for decode identity. A host-pinned staging alloc is created at construction for pinned/managed/VMM (mapped uses pageable staging so it does not steal mlock) and does not charge HBM. Prefetch is H2D or managed migrate without GEMM (`ExpertPhase::Transferring` until the copy event completes). `evict` is `ExpertPhase::Evicting` until the stream-ordered free completes (managed uses host-sync `free_sync` unless `--prefetch-host`; mapped is `free_host_pinned` unless `--host-register-mapped` (`host_unregister` then `free_host`); VMM is `va_release`; pinned `sync_alloc` is `free_sync`). H2D uses `Sim::alloc` (`cudaMallocAsync`) unless `sync_alloc` (`malloc` / `memcpy_sync`). After a drain or `synchronize_stream` on compute, an idle stream captures a per-page GEMM graph and later acquires `launch_graph`. Lease of Transferring/Cold/Evicting is refused. `pin_hot` waits the copy and this page's GEMM lease (or launch-completion on pinned replica D2D), then NVLink-replicates onto `(home + 1) % n_gpus` when `n_gpus >= 2` (D2D, dest prefetch when managed unless AccessedBy, VMM `va_set_access` without dest HBM when AccessedBy else map+D2D, mempool `pool_set_access` without dest HBM when AccessedBy else D2D, no-op when mapped). `migrate(key, dst)` waits that page (and replica) then D2D-moves onto `dst` (copy stream; dest compute waits the event; src HBM dropped) or, when managed, prefetches `dst` and `drop_managed_copy`s the source unless AccessedBy (retarget GEMM, page stays home); mapped only retargets GEMM; VMM maps dest then D2D and unmaps src unless AccessedBy (retarget GEMM, keep home physicals); pinned AccessedBy retargets GEMM and keeps home physicals. `place_hot` is fail-loud. `score()` is wall/HBM/bytes/`energy_uj`/`ns_per_token`. |
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
access-policy windows must align). `--l2-ratio N` is CUDA `hitRatio` as ‰
(`1..=1000`; implies `--l2-persist`; unset is 1000; a partial ratio bills more
HBM than full persist). `--l2-streaming` is `cudaAccessPropertyStreaming` for
persist GEMM window hits (needs `--l2-persist`; a reused expert bills full HBM).
`--cluster N` is
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
Default inherits that occupancy (distinct from `--cluster-spread`). `--cluster-load-balance` is
`cudaLaunchAttributeClusterSchedulingPolicyPreference` LoadBalancing: needs
`--func-cluster-spread`, restores leftover Hyper-Q overlap,
exclusive with `--cluster-spread`. `--cluster-must-set` is
`cudaFuncSetAttribute` ClusterDimMustBeSet (needs `--cluster`; occupancy
matches `--cluster`; SetAttribute is +1 ns). `--required-cluster N` is
`cudaFuncSetAttribute` RequiredClusterWidth (needs `--cluster`; must match
`--cluster`; occupancy matches `--cluster`; SetAttribute is +1 ns). `--max-shared` is
`cudaLaunchAttributePreferredSharedMemoryCarveout` MaxShared: occupies
every Hyper-Q slot. `--func-max-shared` is `cudaFuncSetAttribute`
PreferredSharedMemoryCarveout MaxShared: launch Default inherits that
occupancy. `--max-l1` is `cudaLaunchAttributePreferredSharedMemoryCarveout`
MaxL1: needs `--func-max-shared`, restores leftover Hyper-Q overlap,
exclusive with `--max-shared`. `--non-portable-cluster` is
`cudaFuncAttributeNonPortableClusterSizeAllowed` so `--cluster N` may
exceed portable size up to the SKU max (Hopper portable 8). `--sync-policy
auto|spin|yield|blocking` is `cudaLaunchAttributeSynchronizationPolicy` on
created streams (decode-stream ITL pays `host_sync_*_ns` when `--decode-priority`;
Auto tax 0). Graph kernel nodes also accept that attribute via
`cudaGraphKernelNodeSetAttribute` (not `KernelAttrs`, not an Engine flag). `--mem-sync-domain default|remote` is
`cudaLaunchAttributeMemSyncDomain` on the decode compute stream (prefill
stays Default; Remote isolates leftover prefill `same_domain_fence_permille`;
walker does not imply `--decode-priority`). `--mem-sync-map identity|collapse` is
`cudaLaunchAttributeMemSyncDomainMap` on that decode stream (collapse maps
remote→0 and restores leftover prefill fence tax; needs `--mem-sync-domain
remote`). `--mem-sync-launch` is launch-attribute Remote on grouped expert
GEMMs (overrides prefill inherit-Default so leftover prefill shares the decode
Remote domain and fence tax returns; needs `--mem-sync-domain remote`). `--mem-sync-launch-map` is
launch-attribute collapse map on grouped expert GEMMs (keeps logical domains
different but maps both to physical 0 so leftover prefill fence tax returns;
needs `--mem-sync-domain remote`). `--shared-mem default|four|eight` is
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
`--graph-update`; gpu-sim named device-graph streams have no Engine flag). `--device-updatable` is
`cudaLaunchAttributeDeviceUpdatableKernelNode` so `--graph-set-params` keeps
the exec uploaded (illegal with `--graph-update`). `--kernel-priority N` is
`cudaLaunchAttributePriority` on grouped expert GEMMs (`None` inherits stream
create priority; `0` is a valid override).
`--mem-sync-domain remote` puts decode GEMMs on
`cudaLaunchMemSyncDomainRemote` (prefill stays Default) so leftover prefill
fence tax does not flush decode ITL. Default is Default. `--mem-sync-map
collapse` maps remote→0 on that decode stream so leftover prefill fence tax
returns (needs `--mem-sync-domain remote`). `--mem-sync-launch` puts Remote on
grouped GEMMs so leftover prefill shares that domain (needs `--mem-sync-domain
remote`). `--mem-sync-launch-map` puts collapse `{default: 0, remote: 0}` on
those GEMMs so leftover prefill maps Default→0 with decode Remote→0 (needs
`--mem-sync-domain remote`). gpu-sim allreduce
tags Remote so a non-zero `same_domain_fence_permille` does not flush
expert compute behind communication. `--cooperative` is
`cudaLaunchCooperativeKernel`: GEMMs occupy every Hyper-Q slot, so
independent sequences cannot overlap even with `--compute-slots 2`.
`--decode-sms N` (`1..=1000`) is a
duration-only SM fraction on every replay stream (compute-bound kernels
scale; memory-bound keep full HBM; default unset is a full chip; does not
partition Hyper-Q). `--green-ctx` binds complementary CUDA green contexts
on decode vs leftover prefill (implies `--decode-priority`; default 500/500;
`--decode-sms 1000` refused) so they may overlap even when `compute_slots`
is 1. `gpu-sim` `green_ctx_record_event` / `green_ctx_wait_event` are
`cuGreenCtxRecordEvent` / `cuGreenCtxWaitEvent` (join every bound stream;
later ctx work waits). `--multicast`
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
`--mempool`. Illegal with `--sync-alloc`. `--mempool-max N` is
`cudaMemPoolCreate` with `cudaMemPoolProps::maxSize` then
`cudaDeviceSetMemPool` so `cudaMallocAsync` draws from the capped pool
(`0` unset is unlimited). Implies `--mempool`. Illegal with `--sync-alloc`.
Hits/misses stay the same when `N` fits; leftover cache plus an OS alloc
that would exceed `N` is OOM. `--shareable` is POSIX-FD mempool IPC
(`cudaMemPoolExportToShareableHandle` / `ImportFromShareableHandle` /
`ExportPointer` / `ImportPointer`): a shareable pool is `cudaDeviceSetMemPool`'d
so `cudaMallocAsync` draws from it, and an imported sibling shares
live/cached (no extra HBM). Implies `--mempool`. Illegal with
`--sync-alloc` / mapped / managed / vmm. `--ipc` is `cudaIpcGetMemHandle` /
`cudaIpcOpenMemHandle` of each miss `cudaMalloc` (alias shares source HBM;
close before `cudaFree`). Implies `--sync-alloc`. Illegal with mapped /
managed / vmm and `--shareable`. Decode identity GEMMs on the `cudaMalloc`
pointer with no IPC handshake. `--share-ptr` is `cudaMemPoolExportPointer` /
`cudaMemPoolImportPointer` of each miss `cudaMallocAsync` (alias shares
source HBM; `cudaFreeAsync` the import before the source). Implies
`--shareable` (which implies `--mempool`). Illegal with `--ipc` / mapped /
managed / vmm / `--sync-alloc`. Decode identity is pool-level `--shareable`
without a per-page pointer handshake. `--mapped` uses `cudaHostAllocMapped`: experts stay in
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
map+D2D / pinned D2D is skipped (`cudaMalloc` / `--sync-alloc` still D2Ds). `--no-read-mostly`
skips `cudaMemAdviseSetReadMostly` (`UnsetReadMostly`; dest prefetch moves;
implies `--managed`). `--no-preferred` skips `cudaMemAdviseSetPreferredLocation`
(`UnsetPreferredLocation`; a remote GEMM first-touches instead of staying on
home; implies `--managed`). `--no-mem-prefetch` skips `cudaMemPrefetchAsync`
at managed fill (kernel first-touches on compute instead of copy-engine
prefetch; implies `--managed`). `--vmm` uses
`va_acquire` (remap an idle VA, else reserve+map) then pinned H2D into that
VA (evict `va_release`s the pointer so the next miss skips reserve).
`--vmm-page N` maps each expert in `N`-byte physicals (`va_acquire_paged`,
vLLM KV-block analog; implies `--vmm`). `--vmm-retain` is
`cuMemRetainAllocationHandle` after each VMM miss map (`alloc_overhead_ns`;
implies `--vmm`; paged VMM retains offset 0 only; `va_release_handle`
before `va_release` so hits stay the same; identity is `--vmm` without a
handle). `--vmm-handle` is `cuMemCreate` plus `cuMemMap` of that handle
instead of combined `va_map` (`alloc_overhead_ns` on create and map; implies
`--vmm`; paged VMM creates offset 0 only; exclusive with `--vmm-retain`;
identity is `--vmm` without create plus map). `expertvm kv` reserves per-sequence
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
`sim` knobs (`host_func`, blocking compute, `sync_alloc`, mempool, `mempool_trim`, `mempool_no_reuse`, `mempool_max`, `shareable`, `ipc`, `share_ptr`, `vmm_retain`, `vmm_handle`, `vmm_page`, pageable H2D, `host_register`, `host_unregister`, `host_register_mapped`, `sync_memops`, `device_sync_memops`, `device_sync_policy`, `memcpy_batch`, `memcpy_during`, `memcpy_any`, `memcpy_attr`, `memset_fill`, `copy_host`, AccessedBy, `no_read_mostly`, `no_preferred`, `no_mem_prefetch`, legacy NULL, stream priority, `graph_update`, `graph_set_params`, `graph_clone`, `graph_build`, `graph_piecewise`, `graph_enable`, `graph_if`, `graph_mem`, `graph_auto_free`, `timing_events`, `event_blocking_sync`, `cooperative`, `mem_sync_domain`, `mem_sync_collapse`, `mem_sync_launch`, `mem_sync_launch_map`, `launch_completion`, `programmatic_event`, `stream_attach`, `managed_host`, `prefetch_host`, `d2h_evict`, `d2h_pageable`, `wait_value`). `--host-register` is `cudaHostRegister` on pageable staging so miss H2D is pinned DMA (implies `--pageable`; illegal with `--mapped` / `--managed`). `--host-unregister` is `cudaHostUnregister` after each miss DMA (implies `--host-register`; pin refunded between misses; `synchronize`; identity keeps staging registered). `--ipc` is `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` of each miss `cudaMalloc` (implies `--sync-alloc`; alias shares source HBM; close before free; not with `--shareable` / mapped / managed / vmm; identity GEMMs on the `cudaMalloc` pointer). `--share-ptr` is `cudaMemPoolExportPointer` / `cudaMemPoolImportPointer` of each miss `cudaMallocAsync` (implies `--shareable`; alias shares source HBM; `cudaFreeAsync` import before source; not with `--ipc` / mapped / managed / vmm / `--sync-alloc`; identity is pool-level `--shareable`). `--host-register-mapped` is `cudaHostRegisterMapped` on expert pages (`alloc_host` then pin+map; implies `--mapped`; illegal with `--host-register`; identity mapped stays `cudaHostAllocMapped`). `--sync-memops` is `cuPointerSetAttribute` SyncMemops on miss device pages so H2D / managed prefetch is host-synchronous (illegal with `--mapped` / `--memcpy-batch`; identity stays async pinned H2D). `--device-sync-memops` is `cudaSetDeviceFlags(cudaDeviceSyncMemops)` so every memcpy/memset on that GPU is host-synchronous (illegal with `--mapped` / `--memcpy-batch`; identity stays async pinned H2D; distinct from per-page `--sync-memops`). `--device-sync-policy` is `cudaSetDeviceFlags` SCHEDULE_* so Auto streams inherit host-wait tax (explicit `--sync-policy` wins; ORs with `--device-sync-memops`). `--event-blocking-sync` is `cudaEventBlockingSync` on copy start/end events (implies `--timing-events`; `synchronize_event` pays `host_sync_blocking_ns`; distinct from `--sync-policy blocking`). `--func-max-shared` is `cudaFuncSetAttribute` PreferredSharedMemoryCarveout MaxShared (launch Default inherits; occupies every Hyper-Q slot; distinct from launch-attribute `--max-shared`). `--max-l1` is launch MaxL1 (needs `--func-max-shared`; restores leftover Hyper-Q overlap; exclusive with `--max-shared`). `--cluster-load-balance` is launch LoadBalancing (needs `--func-cluster-spread`; restores leftover Hyper-Q overlap; exclusive with `--cluster-spread`). `--func-shared-mem` is `cudaFuncSetSharedMemConfig` (launch Default inherits bank-width duration; distinct from launch-attribute `--shared-mem`). `--l2-reset` is `cudaCtxResetPersistingL2Cache` after each GEMM (implies `--l2-persist`; live; cannot capture). `--l2-streaming` is `cudaAccessPropertyStreaming` on persist GEMM windows (needs `--l2-persist`; reused expert bills full HBM). `--memcpy-batch` is `cudaMemcpyBatchAsync` for a multi-expert pinned/VMM prefetch on one stream (sibling copies share a stream-order snapshot; demand acquire stays sequential). Illegal with `--pageable`, `--sync-alloc`, `--mapped`, or `--managed`. `--memcpy-during` is `cudaMemcpySrcAccessOrderDuringApiCall` on that batch (needs `--memcpy-batch`; the batch API waits those copies before return; identity stays Stream order). `--memcpy-any` is `cudaMemcpySrcAccessOrderAny` on that batch (needs `--memcpy-batch`; empty deps; no API wait; exclusive with `--memcpy-during`). `--memcpy-attr` is `cudaMemcpyWithAttributesAsync` DuringApiCall on demand pinned/VMM miss H2D (the API waits that copy; does not imply `--memcpy-batch`; replica D2D stays memcpy; not with `--mapped` / `--managed` / `--pageable` / `--memset-fill` / `--sync-alloc` / `--sync-memops` / `--device-sync-memops`). `--d2h-evict` is `cudaMemcpyAsync` Device→HostPinned before pinned/VMM LRU free (extra PCIe; next miss still fills from staging; not with `--mapped` / `--managed`; distinct from `--prefetch-host`). `--d2h-pageable` is `cudaMemcpyAsync` Device→Host (pageable bounce-buffer) before that free (host-synchronous; implies `--pageable`; not with `--mapped` / `--managed` / `--host-register` / `--d2h-evict`; identity stays free with no D2H). `--memset-fill` is `cudaMemsetAsync` of pinned/VMM miss pages (HBM write, compute occupancy; not mapped/managed/pageable/memcpy-batch; distinct from `--graph-memset` scratch). `--copy-host` is `cudaLaunchHostFunc` after miss DMA / prefetch (`host_func_ns` on the DMA stream before copy-ready; does not imply `--host-func`; mapped misses are a no-op). `--graph-enable` is `cudaGraphNodeSetEnabled` on a wide combo parent (implies `--cuda-graphs`; illegal with `--device-launch`). `--mem-sync-domain remote` is `cudaLaunchAttributeMemSyncDomain` on the decode stream (prefill stays Default). `--mem-sync-map collapse` is `cudaLaunchAttributeMemSyncDomainMap` `{default: 0, remote: 0}` on that stream (needs `--mem-sync-domain remote`; restores leftover prefill fence tax). `--mem-sync-launch` is launch-attribute Remote on grouped GEMMs (needs `--mem-sync-domain remote`; restores leftover prefill fence tax). `--mem-sync-launch-map` is launch-attribute collapse map on grouped GEMMs (needs `--mem-sync-domain remote`; restores leftover prefill fence tax). `--launch-completion` is `cudaLaunchAttributeLaunchCompletionEvent` on grouped GEMMs (replica D2D waits kernel start; illegal with `--device-launch`). `--programmatic-event` is `cudaLaunchAttributeProgrammaticEvent` on those GEMMs (replica D2D waits the PDL trigger; illegal with `--device-launch`). `--stream-attach` is `cudaStreamAttachMemAsync` Single on managed experts (prefetch on compute; implies `--managed`; illegal with `--seq-streams`). `--managed-host` is `cudaMallocManaged(..., cudaMemAttachHost)` then Global attach on the copy stream (implies `--managed`; leftover GEMM still overlaps that prefetch unless `--stream-attach`). `--prefetch-host` is `cudaMemPrefetchAsync` to host on managed evict (implies `--managed`; next miss prefetches the same alloc back). `--wait-value` is `cuStreamWaitValue64` / `WriteValue64` after H2D (8-byte device mailbox; decode identity stays events). `--mempool-trim` is `cudaMemPoolTrimTo(0)` after score (implies `--mempool`; illegal with `--sync-alloc`; token ITL does not trim). `--mempool-max N` is `cudaMemPoolProps::maxSize` (implies `--mempool`; illegal with `--sync-alloc`). `--max-batch N` admits N sequences per engine
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
destroyed). `--graph-clone-parent` clones combo parents before instantiate
(recursive children; `graph_clone_ns` per id; does not imply `--graph-clone`;
store GEMM stays per-leaf). `--graph-build` is `cudaGraphCreate` / `cudaGraphAdd*` instead of
stream capture (no idle-stream wait; implies `--cuda-graphs` on the walker;
combo parents clone instantiated kernel leaves; graph-mem / auto-free
leaves MOVE an uninstantiated `cudaGraphClone` because CUDA clone-ownership
cannot name a child with mem nodes; independent children have no
`graph_add_dependencies` edge, so expert GEMMs may Hyper-Q overlap).
`--graph-build-deps` chains those children
(`graph_add_dependencies`; needs `--graph-build`; sibling GEMMs serialize;
does not imply graph-build; store GEMM stays per-leaf). `--graph-host` inserts
`cudaGraphAddHostNode` BETWEEN those children (`host_func_ns`; needs
`--graph-build`; does not imply `--host-func` or graph-build; store GEMM stays
per-leaf). `--graph-piecewise` is `cudaStreamBeginCaptureToGraph` combo
parents (each leaf is an extra root on one parent; same Hyper-Q overlap;
illegal with `--graph-build`). `--graph-capture-deps` chains those
fragments (`numDependencies > 0`; needs `--graph-piecewise`; sibling GEMMs
serialize; does not imply piecewise; store GEMM stays per-leaf). `--graph-capture-host`
is captured `cudaLaunchHostFunc` BETWEEN those fragments (`host_func_ns`; needs
`--graph-piecewise`; does not imply `--host-func` or piecewise; store GEMM stays
per-leaf). Combo overlap stays separate capture
sessions: `cudaStreamUpdateCaptureDependencies` extra deps are additive
with stream-order, so they cannot split same-stream children. `--graph-enable`
is `cudaGraphNodeSetEnabled` on a wide combo parent so a later token that
GEMMs a subset skips extra child graphs instead of instantiating a new
parent (implies `--cuda-graphs`; illegal with `--device-launch`; store GEMM
stays per-leaf). `--graph-if`
wraps `--graph-build` combo children in `cudaGraphAddIf` +
`cudaGraphSetConditional` so a later subset retargets extras with exec
SetParams (clears upload; needs `--graph-build`; not with `--device-launch`
or `--graph-enable`; store GEMM stays per-leaf). `--graph-mem`
records a scratch `cudaMallocAsync` + free in each leaf GEMM graph (HBM peak
includes the workspace; `--graph-update` is skipped). `--graph-memset` is
`cudaMemsetAsync` / `cudaGraphAddMemsetNode` of that scratch BETWEEN alloc
and GEMM (needs `--graph-mem`; extra HBM-write tax; store and walker leaves). `--graph-memcpy` is
`cudaMemcpyAsync` / `cudaGraphAddMemcpyNode` H2D of that scratch BETWEEN alloc
and GEMM (needs `--graph-mem`; copy-engine PCIe tax; store and walker leaves; legal
with `--graph-memset`). `--graph-leaf-host` is `cudaGraphAddHostNode` / captured
`cudaLaunchHostFunc` BEFORE the leaf GEMM (implies `--cuda-graphs`; each leaf
bills `host_func_ns`; not with `--device-launch`; store and walker leaves; does
not imply `--host-func`). `--graph-auto-free` is
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
`--graph-update` / `--graph-set-params` / `--graph-clone` / `--graph-clone-parent` / `--graph-build` / `--graph-build-deps` / `--graph-host` / `--graph-piecewise` / `--graph-capture-deps` / `--graph-capture-host` / `--graph-enable` / `--graph-if` / `--graph-mem` / `--graph-memset` / `--graph-memcpy` / `--graph-leaf-host` / `--graph-auto-free` / `--graph-mem-trim` / `--timing-events` / `--event-blocking-sync` are `GpuStoreCfg`
on the Engine store. `--mapped` / `--managed` / `--vmm` select `GpuFill`
(`gguf_gemv engine --expert-sim --managed`). `--host-func` /
`--blocking-streams` / `--sync-alloc` / `--ipc` / `--mempool` / `--mempool-trim` / `--mempool-no-reuse` / `--mempool-max` / `--shareable` / `--share-ptr` / `--vmm-retain` / `--vmm-handle` / `--vmm-page` /
`--pageable` / `--host-register` / `--host-unregister` / `--host-register-mapped` / `--sync-memops` / `--device-sync-memops` / `--memcpy-batch` / `--memcpy-during` / `--memcpy-any` / `--memcpy-attr` / `--d2h-evict` / `--d2h-pageable` / `--memset-fill` / `--copy-host` / `--accessed-by` / `--legacy-null` / `--stream-priority` /
`--seq-streams` / `--kv-sim` / `--kv-bytes` / `--decode-priority` /
`--cooperative` / `--pdl` / `--l2-persist` / `--l2-reset` / `--l2-fetch` / `--l2-ratio` / `--l2-streaming` / `--cluster` / `--preferred-cluster` / `--cluster-spread` / `--func-cluster-spread` / `--cluster-load-balance` / `--cluster-must-set` / `--required-cluster` / `--max-shared` / `--func-max-shared` / `--max-l1` / `--non-portable-cluster` / `--sync-policy` / `--device-sync-policy` / `--event-blocking-sync` / `--mem-sync-domain` / `--mem-sync-map` / `--mem-sync-launch` / `--mem-sync-launch-map` / `--shared-mem` / `--func-shared-mem` / `--device-shared-mem` / `--portable-cluster` / `--optin-shared` / `--dynamic-shared` / `--portable-shared` / `--nvlink-util` / `--device-launch` / `--device-updatable` / `--kernel-priority` / `--launch-completion` / `--programmatic-event` / `--wait-value` / `--compute-slots` / `--decode-sms` / `--green-ctx` / `--multicast` / `--shareable` / `--ipc` / `--share-ptr` / `--vmm-retain` / `--vmm-handle` are `GpuStoreCfg` knobs on `gguf_gemv engine`.
`expertvm sim` / `schedule` / `store` take `--compute-slots` / `--decode-sms` / `--green-ctx`
/ `--decode-priority` / `--cooperative` / `--pdl` / `--l2-persist` / `--l2-reset` / `--l2-fetch` / `--l2-ratio` / `--l2-streaming` / `--cluster` / `--preferred-cluster` / `--cluster-spread` / `--func-cluster-spread` / `--cluster-load-balance` / `--cluster-must-set` / `--required-cluster` / `--max-shared` / `--func-max-shared` / `--max-l1` / `--non-portable-cluster` / `--sync-policy` / `--device-sync-policy` / `--mem-sync-domain` / `--mem-sync-map` / `--mem-sync-launch` / `--mem-sync-launch-map` / `--shared-mem` / `--func-shared-mem` / `--device-shared-mem` / `--portable-cluster` / `--optin-shared` / `--dynamic-shared` / `--portable-shared` / `--nvlink-util` / `--device-launch` / `--device-updatable` / `--kernel-priority` / `--launch-completion` / `--programmatic-event` / `--wait-value` / `--multicast` / `--shareable` / `--ipc` / `--share-ptr` / `--vmm-retain` / `--vmm-handle` (Hyper-Q occupancy, duration-only SM fraction, CUDA green-context occupancy partition,
decode-stream ITL, exclusive cooperative GEMMs, same-stream PDL overlap, Hopper cluster occupancy / preferred dim / Spread scheduling / ClusterDimMustBeSet / RequiredClusterWidth / non-portable size, NVLS replica fanout, POSIX-FD mempool IPC, and cudaMalloc IPC on the trace walker). Walker `--decode-sms` does **not**
imply `--decode-priority` (token 0 is prefill). Walker `--green-ctx` **does**. Engine `--mem-sync-domain remote`
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
windows must align). `--l2-ratio N` is CUDA `hitRatio` as ‰ (`1..=1000`;
implies `--l2-persist`; unset is 1000; a partial ratio bills more HBM than
full persist). `--l2-streaming` is `cudaAccessPropertyStreaming` for persist
GEMM window hits (needs `--l2-persist`; a reused expert bills full HBM).
`--cluster N`
is a Hopper thread-block cluster so leftover kernels cannot overlap a
launch that fills Hyper-Q. `--preferred-cluster N` occupies the preferred
size when it fits in `--compute-slots` (needs `--cluster`; must be a
multiple of it). `--cluster-spread` occupies every Hyper-Q slot
even when `N` is smaller than `--compute-slots` (no-op without `--cluster`
of at least 2). `--func-cluster-spread` occupies every Hyper-Q slot via
function Spread policy (launch Default inherits). `--cluster-load-balance` is
launch LoadBalancing (needs `--func-cluster-spread`; restores Hyper-Q overlap;
exclusive with `--cluster-spread`). `--cluster-must-set` is
`cudaFuncAttributeClusterDimMustBeSet` (needs `--cluster`; occupancy matches
`--cluster`). `--required-cluster N` is
`cudaFuncAttributeRequiredClusterWidth` (needs `--cluster`; must match
`--cluster`; occupancy matches `--cluster`). `--max-shared` occupies every Hyper-Q slot via MaxShared
carveout. `--func-max-shared` occupies every Hyper-Q slot via function
MaxShared carveout (launch Default inherits). `--max-l1` is launch MaxL1
(needs `--func-max-shared`; restores Hyper-Q overlap; exclusive with
`--max-shared`). `--non-portable-cluster` allows `--cluster N` above portable size
up to the SKU max. `--sync-policy auto|spin|yield|blocking` is stream host-wait
(Auto tax 0). `--device-sync-policy auto|spin|yield|blocking` is device host-wait
(`cudaSetDeviceFlags` SCHEDULE_*; Auto streams inherit; explicit `--sync-policy`
wins). `--event-blocking-sync` is `cudaEventBlockingSync` on copy
start/end events (implies `--timing-events`; `synchronize_event` pays
`host_sync_blocking_ns`; distinct from `--sync-policy blocking`). `--mem-sync-domain default|remote` is decode-stream
`cudaLaunchAttributeMemSyncDomain` (Remote isolates leftover prefill fence
tax; engine implies `--decode-priority`). `--mem-sync-map identity|collapse` is
decode-stream `cudaLaunchAttributeMemSyncDomainMap` (collapse maps remote→0
and restores leftover prefill fence tax; needs `--mem-sync-domain remote`). `--mem-sync-launch` is
launch-attribute Remote on grouped GEMMs (overrides prefill inherit-Default;
needs `--mem-sync-domain remote`). `--mem-sync-launch-map` is launch-attribute
collapse map on grouped GEMMs (keeps logical domains different but maps both
to physical 0; needs `--mem-sync-domain remote`). `--shared-mem default|four|eight` is kernel-node bank width
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
and keep the exec uploaded after set-params (CUDA: once opted in, the node
cannot opt out, be destroyed, or take part in CopyAttributes). `--kernel-priority N` is
`cudaLaunchAttributePriority` on those GEMMs (`None` inherits stream create
priority). `--launch-completion` is `cudaLaunchAttributeLaunchCompletionEvent`
on grouped GEMMs (store replica D2D waits kernel start; illegal with
`--device-launch`). `--programmatic-event` is
`cudaLaunchAttributeProgrammaticEvent` on those GEMMs (store replica D2D
waits the PDL trigger; `triggerAtBlockStart` stays false; not External;
illegal with `--device-launch`). `--wait-value` is `cuStreamWaitValue64` / `WriteValue64`
after H2D (8-byte `cudaMallocAsync` mailbox, copy stream waited before
H2D so compute wait is resident during DMA; decode identity stays events).
`--mempool-trim` is `cudaMemPoolTrimTo(0)` after `score()` / walker finish
(implies `--mempool`; illegal with `--sync-alloc`; token ITL does not trim).
`--cooperative` is
`cudaLaunchCooperativeKernel`: those GEMMs occupy every Hyper-Q slot, so
leftover prefill cannot overlap even with `--compute-slots 2`. `--decode-sms N` (`1..=1000`)
is a duration-only SM fraction on the decode stream (leftover prefill gets
the remainder; implies `--decode-priority` on Engine, not on the walker).
`--green-ctx` binds complementary CUDA green contexts so leftover prefill
may overlap decode even when occupancy is exclusive (implies
`--decode-priority`; default 500/500; `--decode-sms 1000` refused). Store
GEMM graphs inherit the launch stream (`CUDA_KERNEL_NODE_PARAMS.ctx` None)
so an expert captured during prefill still uses the decode partition.
Distinct from `--decode-sms` alone (exclusive compute still serializes leftover
prefill). `gpu-sim` `green_ctx_record_event` / `green_ctx_wait_event` join or
hold every stream bound to a ctx (not a second `--green-ctx`).
`green_ctx_synchronize` is `cudaExecutionCtxSynchronize` (one ctx; other
ctxs keep running). `stream_get_dev_resource` is `cuStreamGetDevResource`.
`green_ctx_get_id` is `cuGreenCtxGetId` (not a second `--green-ctx`).
`green_ctx_get_device` is `cudaExecutionCtxGetDevice`.
`gpu-sim` `reset_device` is `cudaDeviceReset` (no Engine flag).
`gpu-sim` `device_primary_ctx_set_flags` is `cuDevicePrimaryCtxSetFlags`
(always Invalid; primary context already seeded; no Engine flag).
`SimError::error_name` / `error_string` are `cudaGetErrorName` /
`cudaGetErrorString` (no Engine flag; no thread-local last error).
`gpu-sim` compute capability is `cudaDevAttrComputeCapabilityMajor` and
`Minor` (example H100 Hopper 9.0; `cuDeviceComputeCapability` is the same
pair; no Engine flag).
`gpu-sim` `ctx_get_api_version` is `cuCtxGetApiVersion` (CUDA 13.0 for the
seeded primary context; no Engine flag).
`gpu-sim` `ctx_get_flags` is `cuCtxGetFlags` (same flags as
`get_device_flags`; no Engine flag).
`gpu-sim` `ctx_set_flags` is `cuCtxSetFlags` (identity with
`set_device_flags`; no Engine flag).
`gpu-sim` `ctx_get_cache_config` is `cuCtxGetCacheConfig` (same as
`get_cache_config`; no Engine flag).
`gpu-sim` `ctx_set_cache_config` is `cuCtxSetCacheConfig` (identity with
`set_cache_config`; no Engine flag).
`gpu-sim` `ctx_get_stream_priority_range` is `cuCtxGetStreamPriorityRange`
(example H100 `(0, -5)`; no Engine flag).
`gpu-sim` `ctx_get_limit` is `cuCtxGetLimit` (same as `get_limit`; no
Engine flag).
`gpu-sim` `ctx_set_limit` is `cuCtxSetLimit` (identity with `set_limit`;
no Engine flag).
`gpu-sim` `ctx_synchronize` is `cuCtxSynchronize` (same wait as
`synchronize_device`; no Engine flag).
`gpu-sim` `ctx_get_shared_mem_config` is `cuCtxGetSharedMemConfig` (same
as `get_shared_mem_config`; no Engine flag).
`gpu-sim` `ctx_set_shared_mem_config` is `cuCtxSetSharedMemConfig` (identity with
`set_shared_mem_config`; no Engine flag).
`gpu-sim` launch-geometry caps are `cudaDevAttrMaxThreadsPerBlock` 1024
and H100 block/grid dims (no Engine flag; not occupancy SM counts).
`gpu-sim` `MaxRegistersPerBlock` is `cudaDevAttrMaxRegistersPerBlock` 65536
(no Engine flag; not a register-file model).
`gpu-sim` `GlobalMemoryBusWidth` is `cudaDevAttrGlobalMemoryBusWidth`
(example H100 5120 bits, H200 6144; no Engine flag; not a memory clock).
`gpu-sim` `SingleToDoublePrecisionPerfRatio` is
`cudaDevAttrSingleToDoublePrecisionPerfRatio` 1 on example H100
(no Engine flag; not an FP64 duration model).
`gpu-sim` compiler-emitted `cudaFuncGetAttributes` fields
(`sharedSizeBytes`, `constSizeBytes`, `localSizeBytes`, `maxThreadsPerBlock`,
`ptxVersion`, `binaryVersion`, `cacheModeCA`) are always 0 (no Engine
flag; not a compiled kernel).
`gpu-sim` `func_get_name` is `cudaFuncGetName` (empty until a compiled
kernel exists; no Engine flag).
`gpu-sim` `func_get_param_info` is `cuFuncGetParamInfo` (Invalid until a
compiled kernel exists; no Engine flag).
`gpu-sim` `func_get_param_count` is `cuFuncGetParamCount` (Invalid until a
compiled kernel exists; no Engine flag).
`gpu-sim` `func_get_cache_config` is `cuFuncGetCacheConfig` (Invalid until a
compiled kernel exists; no Engine flag).
`gpu-sim` `memset_d8_async` is `cuMemsetD8Async`
(count is CUDA `N` of 8-bit values; no Engine flag).
`gpu-sim` `memset_d8` is `cuMemsetD8` (host-sync; capture refused; no Engine flag).
`gpu-sim` `event_query` is `cuEventQuery` (identity with `query_event`; no Engine flag).
`gpu-sim` `stream_query` is `cuStreamQuery` (identity with `query_stream`; no Engine flag).
`gpu-sim` `event_synchronize` is `cuEventSynchronize` (identity with `synchronize_event`; no Engine flag).
`gpu-sim` `stream_synchronize` is `cuStreamSynchronize` (identity with `synchronize_stream`; no Engine flag).
`gpu-sim` `event_destroy` is `cuEventDestroy` (identity with `destroy_event`; no Engine flag).
`gpu-sim` `event_create` is `cuEventCreate` (identity with `create_event`; no Engine flag).
`gpu-sim` `event_create_with_flags` is `cuEventCreateWithFlags` (identity with `create_event_with_flags`; no Engine flag).
`gpu-sim` `event_record` is `cuEventRecord` (identity with `record_event`; no Engine flag).
`gpu-sim` `event_record_with_flags` is `cuEventRecordWithFlags` (identity with `record_event_with_flags`; no Engine flag).
`gpu-sim` `stream_wait_event` is `cuStreamWaitEvent` (identity with `wait_event`; no Engine flag).
`gpu-sim` `stream_wait_event_with_flags` is `cuStreamWaitEvent` with flags (identity with `wait_event_with_flags`; no Engine flag).
`gpu-sim` `event_elapsed` is `cuEventElapsedTime` (identity with `event_elapsed_ns`; ns, not milliseconds; no Engine flag).
`gpu-sim` `mem_get_info` is `cuMemGetInfo` (identity with `mem_info`; no Engine flag).
`gpu-sim` `stream_create` is `cudaStreamCreate` / `cuStreamCreate` default flags (identity with `stream_create_with_flags` DEFAULT; blocking; no Engine flag).
`gpu-sim` `stream_create_priority` is `cuStreamCreateWithPriority` (identity with `stream_create_with_priority`; no Engine flag).
`gpu-sim` `stream_create_flags` is `cuStreamCreateWithFlags` (identity with `stream_create_with_flags`; no Engine flag).
`gpu-sim` `stream_flags` is `cuStreamGetFlags` (identity with `stream_get_flags`; no Engine flag).
`gpu-sim` `get_stream_priority` is `cuStreamGetPriority` (identity with `stream_get_priority`; no Engine flag).
`gpu-sim` `device_graph_mem_get` is `cuDeviceGetGraphMemAttribute` (identity with `graph_mem_get`; no Engine flag).
`gpu-sim` `device_graph_mem_set` is `cuDeviceSetGraphMemAttribute` (identity with `graph_mem_set`; no Engine flag).
`gpu-sim` `device_graph_mem_trim` is `cuDeviceGraphMemTrim` (identity with `graph_mem_trim`; no Engine flag).
`gpu-sim` `get_stream_id` is `cuStreamGetId` (identity with `stream_get_id`; no Engine flag).
`gpu-sim` `copy_stream_attributes` is `cuStreamCopyAttributes` (identity with `stream_copy_attributes`; no Engine flag).
`gpu-sim` `get_stream_attribute` is `cuStreamGetAttribute` (identity with `stream_get_attribute`; no Engine flag).
`gpu-sim` `set_stream_attribute` is `cuStreamSetAttribute` (identity with `stream_set_attribute`; no Engine flag).
`gpu-sim` `get_graph_kernel_node_attribute` is `cuGraphKernelNodeGetAttribute` (identity with `graph_kernel_node_get_attribute`; no Engine flag).
`gpu-sim` `set_graph_kernel_node_attribute` is `cuGraphKernelNodeSetAttribute` (identity with `graph_kernel_node_set_attribute`; no Engine flag).
`gpu-sim` `get_graph_exec_kernel_node_attribute` is `cuGraphExecKernelNodeGetAttribute` (identity with `graph_exec_kernel_node_get_attribute`; no Engine flag).
`gpu-sim` `set_graph_exec_kernel_node_attribute` is `cuGraphExecKernelNodeSetAttribute` (identity with `graph_exec_kernel_node_set_attribute`; no Engine flag).
`gpu-sim` `copy_graph_kernel_node_attributes` is `cuGraphKernelNodeCopyAttributes` (identity with `graph_kernel_node_copy_attributes`; no Engine flag).
`gpu-sim` `copy_graph_exec_kernel_node_attributes` is `cuGraphExecKernelNodeCopyAttributes` (identity with `graph_exec_kernel_node_copy_attributes`; no Engine flag).
`gpu-sim` `get_graph_kernel_node_params` is `cuGraphKernelNodeGetParams` (identity with `graph_kernel_get_params`; no Engine flag).
`gpu-sim` `get_graph_exec_kernel_node_params` is `cuGraphExecKernelNodeGetParams` (identity with `graph_exec_kernel_get_params`; no Engine flag).
`gpu-sim` `set_graph_kernel_node_params` is `cuGraphKernelNodeSetParams` (identity with `graph_kernel_set_params`; no Engine flag).
`gpu-sim` `set_graph_exec_kernel_node_params` is `cuGraphExecKernelNodeSetParams` (identity with `graph_exec_kernel_set_params`; no Engine flag).
`gpu-sim` `get_graph_memcpy_node_params` is `cuGraphMemcpyNodeGetParams` (identity with `graph_memcpy_get_params`; no Engine flag).
`gpu-sim` `get_graph_exec_memcpy_node_params` is `cuGraphExecMemcpyNodeGetParams` (identity with `graph_exec_memcpy_get_params`; no Engine flag).
`gpu-sim` `set_graph_memcpy_node_params` is `cuGraphMemcpyNodeSetParams` (identity with `graph_memcpy_set_params`; no Engine flag).
`gpu-sim` `set_graph_exec_memcpy_node_params` is `cuGraphExecMemcpyNodeSetParams` (identity with `graph_exec_memcpy_set_params`; no Engine flag).
`gpu-sim` `get_graph_memset_node_params` is `cuGraphMemsetNodeGetParams` (identity with `graph_memset_get_params`; no Engine flag).
`gpu-sim` `get_graph_exec_memset_node_params` is `cuGraphExecMemsetNodeGetParams` (identity with `graph_exec_memset_get_params`; no Engine flag).
`gpu-sim` `set_graph_memset_node_params` is `cuGraphMemsetNodeSetParams` (identity with `graph_memset_set_params`; no Engine flag).
`gpu-sim` `set_graph_exec_memset_node_params` is `cuGraphExecMemsetNodeSetParams` (identity with `graph_exec_memset_set_params`; no Engine flag).
`gpu-sim` `get_graph_host_node_params` is `cuGraphHostNodeGetParams` (identity with `graph_host_get_params`; no Engine flag).
`gpu-sim` `get_graph_exec_host_node_params` is `cuGraphExecHostNodeGetParams` (identity with `graph_exec_host_get_params`; no Engine flag).
`gpu-sim` `set_graph_host_node_params` is `cuGraphHostNodeSetParams` (identity with `graph_host_set_params`; no Engine flag).
`gpu-sim` `set_graph_exec_host_node_params` is `cuGraphExecHostNodeSetParams` (identity with `graph_exec_host_set_params`; no Engine flag).
`gpu-sim` `get_graph_batch_mem_op_node_params` is `cuGraphBatchMemOpNodeGetParams` (identity with `graph_batch_mem_ops_get_params`; no Engine flag).
`gpu-sim` `get_graph_exec_batch_mem_op_node_params` is `cuGraphExecBatchMemOpNodeGetParams` (identity with `graph_exec_batch_mem_ops_get_params`; no Engine flag).
`gpu-sim` `set_graph_batch_mem_op_node_params` is `cuGraphBatchMemOpNodeSetParams` (identity with `graph_batch_mem_op_set_params`; no Engine flag).
`gpu-sim` `set_graph_exec_batch_mem_op_node_params` is `cuGraphExecBatchMemOpNodeSetParams` (identity with `graph_exec_batch_mem_op_set_params`; no Engine flag).
`gpu-sim` `set_graph_event_record_node_event` is `cuGraphEventRecordNodeSetEvent` (identity with `graph_event_record_set_event`; no Engine flag).
`gpu-sim` `set_graph_exec_event_record_node_event` is `cuGraphExecEventRecordNodeSetEvent` (identity with `graph_exec_event_record_set_event`; no Engine flag).
`gpu-sim` `set_graph_event_wait_node_event` is `cuGraphEventWaitNodeSetEvent` (identity with `graph_event_wait_set_event`; no Engine flag).
`gpu-sim` `set_graph_exec_event_wait_node_event` is `cuGraphExecEventWaitNodeSetEvent` (identity with `graph_exec_event_wait_set_event`; no Engine flag).
`gpu-sim` `get_graph_event_record_node_event` is `cuGraphEventRecordNodeGetEvent` (identity with `graph_event_record_get_event`; no Engine flag).
`gpu-sim` `get_graph_exec_event_record_node_event` is `cuGraphExecEventRecordNodeGetEvent` (identity with `graph_exec_event_record_get_event`; no Engine flag).
`gpu-sim` `get_graph_event_wait_node_event` is `cuGraphEventWaitNodeGetEvent` (identity with `graph_event_wait_get_event`; no Engine flag).
`gpu-sim` `get_graph_exec_event_wait_node_event` is `cuGraphExecEventWaitNodeGetEvent` (identity with `graph_exec_event_wait_get_event`; no Engine flag).
`gpu-sim` `get_graph_child_graph_node_graph` is `cuGraphChildGraphNodeGetGraph` (identity with `graph_child_get_graph`; no Engine flag).
`gpu-sim` `get_graph_exec_child_graph_node_graph` is `cuGraphExecChildGraphNodeGetGraph` (identity with `graph_exec_child_get_graph`; no Engine flag).
`gpu-sim` `set_graph_child_graph_node_params` is `cuGraphChildGraphNodeSetParams` (identity with `graph_child_set_params`; no Engine flag).
`gpu-sim` `set_graph_exec_child_graph_node_params` is `cuGraphExecChildGraphNodeSetParams` (identity with `graph_exec_child_set_params`; no Engine flag).
`gpu-sim` `set_graph_node_params` is `cuGraphNodeSetParams` (identity with `graph_node_set_params`; no Engine flag).
`gpu-sim` `set_graph_exec_node_params` is `cuGraphExecNodeSetParams` (identity with `graph_exec_node_set_params`; no Engine flag).
`gpu-sim` `get_graph_node_params` is `cuGraphNodeGetParams` (identity with `graph_node_get_params`; no Engine flag).
`gpu-sim` `get_graph_exec_node_params` is `cuGraphExecNodeGetParams` (identity with `graph_exec_node_get_params`; no Engine flag).
`gpu-sim` `set_graph_node_enabled` is `cuGraphNodeSetEnabled` (identity with `graph_node_set_enabled`; no Engine flag).
`gpu-sim` `get_graph_node_enabled` is `cuGraphNodeGetEnabled` (identity with `graph_node_get_enabled`; no Engine flag).
`gpu-sim` `get_graph_exec_flags` is `cuGraphExecGetFlags` (identity with `graph_exec_get_flags`; no Engine flag).
`gpu-sim` `get_graph_id` is `cuGraphGetId` (identity with `graph_get_id`; no Engine flag).
`gpu-sim` `get_graph_exec_id` is `cuGraphExecGetId` (identity with `graph_get_id`; no Engine flag).
`gpu-sim` `get_graph_nodes` is `cuGraphGetNodes` (identity with `graph_nodes`; no Engine flag).
`gpu-sim` `get_graph_root_nodes` is `cuGraphGetRootNodes` (identity with `graph_root_nodes`; no Engine flag).
`gpu-sim` `get_graph_edges` is `cuGraphGetEdges` (identity with `graph_edges`; no Engine flag).
`gpu-sim` `get_graph_edges_with_data` is `cuGraphGetEdges` v2 (identity with `graph_edges_with_data`; no Engine flag).
`gpu-sim` `get_graph_node_dependencies` is `cuGraphNodeGetDependencies` (identity with `graph_node_deps`; no Engine flag).
`gpu-sim` `get_graph_node_dependencies_with_data` is `cuGraphNodeGetDependencies` v2 (identity with `graph_node_deps_with_data`; no Engine flag).
`gpu-sim` `get_graph_node_dependent_nodes` is `cuGraphNodeGetDependentNodes` (identity with `graph_node_dependents`; no Engine flag).
`gpu-sim` `get_graph_node_dependent_nodes_with_data` is `cuGraphNodeGetDependentNodes` v2 (identity with `graph_node_dependents_with_data`; no Engine flag).
`gpu-sim` `get_graph_node_type` is `cuGraphNodeGetType` (identity with `graph_node_kind`; no Engine flag).
`gpu-sim` `find_graph_node_in_clone` is `cuGraphNodeFindInClone` (identity with `graph_node_find_in_clone`; no Engine flag).
`gpu-sim` `graph_clone` is `cuGraphClone` (identity with `clone_graph`; no Engine flag).
`gpu-sim` `graph_debug_dot_print` is `cuGraphDebugDotPrint` (identity with `graph_debug_dot`; no Engine flag).
`gpu-sim` `graph_debug_dot_print_with_flags` is `cuGraphDebugDotPrint` with flags (identity with `graph_debug_dot_with_flags`; no Engine flag).
`gpu-sim` `graph_instantiate` is `cuGraphInstantiate` (identity with `instantiate_graph`; no Engine flag).
`gpu-sim` `graph_instantiate_with_flags` is `cuGraphInstantiateWithFlags` (identity with `instantiate_graph_with_flags`; no Engine flag).
`gpu-sim` `graph_instantiate_with_params` is `cuGraphInstantiateWithParams` (identity with `instantiate_graph_with_params`; no Engine flag).
`gpu-sim` `graph_launch` is `cuGraphLaunch` (identity with `launch_graph`; no Engine flag).
`gpu-sim` `graph_upload` is `cuGraphUpload` (identity with `upload_graph`; no Engine flag).
`gpu-sim` `graph_upload_async` is `cuGraphUpload` on a stream (identity with `upload_graph_async`; no Engine flag).
`gpu-sim` `graph_destroy` is `cuGraphDestroy` (identity with `destroy_graph`; no Engine flag).
`gpu-sim` `graph_exec_destroy` is `cuGraphExecDestroy` (identity with `destroy_graph`; no Engine flag).
`gpu-sim` `graph_exec_update` is `cuGraphExecUpdate` (identity with `update_graph`; no Engine flag).
`gpu-sim` `graph_exec_update_with_info` is `cuGraphExecUpdate` with info (identity with `update_graph_with_info`; no Engine flag).
`gpu-sim` `add_graph_dependencies` is `cuGraphAddDependencies` (identity with `graph_add_dependencies`; no Engine flag).
`gpu-sim` `add_graph_dependencies_n` is `cuGraphAddDependencies` of pairs (identity with `graph_add_dependencies_n`; no Engine flag).
`gpu-sim` `add_graph_dependencies_with_data` is `cuGraphAddDependencies` with data (identity with `graph_add_dependencies_with_data`; no Engine flag).
`gpu-sim` `add_graph_dependencies_n_with_data` is `cuGraphAddDependencies` v2 (identity with `graph_add_dependencies_n_with_data`; no Engine flag).
`gpu-sim` `remove_graph_dependencies` is `cuGraphRemoveDependencies` (identity with `graph_remove_dependencies`; no Engine flag).
`gpu-sim` `remove_graph_dependencies_n` is `cuGraphRemoveDependencies` of pairs (identity with `graph_remove_dependencies_n`; no Engine flag).
`gpu-sim` `remove_graph_dependencies_with_data` is `cuGraphRemoveDependencies` with data (identity with `graph_remove_dependencies_with_data`; no Engine flag).
`gpu-sim` `remove_graph_dependencies_n_with_data` is `cuGraphRemoveDependencies` v2 (identity with `graph_remove_dependencies_n_with_data`; no Engine flag).
`gpu-sim` `destroy_graph_node` is `cuGraphDestroyNode` (identity with `graph_destroy_node`; no Engine flag).
`gpu-sim` `launch_device_graph` is device-side `cuGraphLaunch` (identity with `device_launch_graph`; no Engine flag).
`gpu-sim` `get_current_graph_exec` is `cuGetCurrentGraphExec` (identity with `current_graph_exec`; no Engine flag).
`gpu-sim` `add_graph_empty` is `cuGraphAddEmptyNode` (identity with `graph_add_empty`; no Engine flag).
`gpu-sim` `add_graph_child` is `cuGraphAddChildGraphNode` (identity with `graph_add_child`; no Engine flag).
`gpu-sim` `add_graph_host` is `cuGraphAddHostNode` (identity with `graph_add_host_func_params`; no Engine flag).
`gpu-sim` `add_graph_event_record` is `cuGraphAddEventRecordNode` (identity with `graph_add_event_record`; no Engine flag).
`gpu-sim` `add_graph_event_wait` is `cuGraphAddEventWaitNode` (identity with `graph_add_event_wait`; no Engine flag).
`gpu-sim` `add_graph_kernel` is `cuGraphAddKernelNode` (identity with `graph_add_kernel`; no Engine flag).
`gpu-sim` `add_graph_memcpy` is `cuGraphAddMemcpyNode` (identity with `graph_add_memcpy`; no Engine flag).
`gpu-sim` `add_graph_memcpy_1d` is `cuGraphAddMemcpyNode1D` (identity with `graph_add_memcpy_1d`; no Engine flag).
`gpu-sim` `add_graph_memcpy_2d` is 2D `cuGraphAddMemcpyNode` (identity with `graph_add_memcpy_2d`; no Engine flag).
`gpu-sim` `add_graph_memcpy_3d` is 3D `cuGraphAddMemcpyNode` (identity with `graph_add_memcpy_3d`; no Engine flag).
`gpu-sim` `add_graph_memset` is packed 1D `cuGraphAddMemsetNode` (identity with `graph_add_memset`; no Engine flag).
`gpu-sim` `add_graph_memset_op` is `cuGraphAddMemsetNode` params (identity with `graph_add_memset_op`; no Engine flag).
`gpu-sim` `add_graph_memset_2d` is 2D `cuGraphAddMemsetNode` (identity with `graph_add_memset_2d`; no Engine flag).
`gpu-sim` `add_graph_memset_3d` is 3D `cuGraphAddMemsetNode` (identity with `graph_add_memset_3d`; no Engine flag).
`gpu-sim` `add_graph_batch_mem_op` is `cuGraphAddBatchMemOpNode` (identity with `graph_add_batch_mem_op`; no Engine flag).
`gpu-sim` `add_graph_batch_mem_op_with_flags` is `cuGraphAddBatchMemOpNode` flags (identity with `graph_add_batch_mem_op_with_flags`; no Engine flag).
`gpu-sim` `add_graph_alloc` is `cuGraphAddMemAllocNode` (identity with `graph_add_alloc`; no Engine flag).
`gpu-sim` `add_graph_alloc_with_access` is `cuGraphAddMemAllocNode` access (identity with `graph_add_alloc_with_access`; no Engine flag).
`gpu-sim` `add_graph_free` is `cuGraphAddMemFreeNode` (identity with `graph_add_free`; no Engine flag).
`gpu-sim` `add_graph_node` is `cuGraphAddNode` (identity with `graph_add_node`; no Engine flag).
`gpu-sim` `add_graph_node_with_data` is `cuGraphAddNode_v2` (identity with `graph_add_node_with_data`; no Engine flag).
`gpu-sim` `add_graph_if` is `cuGraphAddNode` IF (identity with `graph_add_if`; no Engine flag).
`gpu-sim` `add_graph_if_else` is `cuGraphAddNode` IF size 2 (identity with `graph_add_if_else`; no Engine flag).
`gpu-sim` `add_graph_while` is `cuGraphAddNode` WHILE (identity with `graph_add_while`; no Engine flag).
`gpu-sim` `add_graph_switch` is `cuGraphAddNode` SWITCH (identity with `graph_add_switch`; no Engine flag).
`gpu-sim` `add_graph_set_conditional` is graph-build `cuGraphSetConditional` (identity with `graph_add_set_conditional`; no Engine flag).
`gpu-sim` `add_graph_write_value64` is graph `cuStreamWriteValue64` (identity with `graph_add_write_value64`; no Engine flag).
`gpu-sim` `add_graph_write_value32` is graph `cuStreamWriteValue32` (identity with `graph_add_write_value32`; no Engine flag).
`gpu-sim` `add_graph_write_value64_with_flags` is graph `cuStreamWriteValue64` flags (identity with `graph_add_write_value64_with_flags`; no Engine flag).
`gpu-sim` `add_graph_write_value32_with_flags` is graph `cuStreamWriteValue32` flags (identity with `graph_add_write_value32_with_flags`; no Engine flag).
`gpu-sim` `add_graph_wait_value64` is graph `cuStreamWaitValue64` (identity with `graph_add_wait_value64`; no Engine flag).
`gpu-sim` `add_graph_wait_value32` is graph `cuStreamWaitValue32` (identity with `graph_add_wait_value32`; no Engine flag).
`gpu-sim` `add_graph_wait_value64_with_flags` is graph `cuStreamWaitValue64` flags (identity with `graph_add_wait_value64_with_flags`; no Engine flag).
`gpu-sim` `add_graph_wait_value32_with_flags` is graph `cuStreamWaitValue32` flags (identity with `graph_add_wait_value32_with_flags`; no Engine flag).
`gpu-sim` `add_graph_cooperative_kernel` is graph cooperative `cudaGraphAddKernelNode` (identity with `graph_add_cooperative_kernel`; no Engine flag).
`gpu-sim` `add_graph_host_func` is graph unnamed `cudaGraphAddHostNode` (identity with `graph_add_host_func`; no Engine flag).
`gpu-sim` `set_graph_memcpy_node_params_1d` is graph `cudaGraphMemcpyNodeSetParams1D` (identity with `graph_memcpy_set_params_1d`; no Engine flag).
`gpu-sim` `set_graph_exec_memcpy_node_params_1d` is graph `cudaGraphExecMemcpyNodeSetParams1D` (identity with `graph_exec_memcpy_set_params_1d`; no Engine flag).
`gpu-sim` `graph_create` is `cuGraphCreate` (identity with `create_graph`; no Engine flag).
`gpu-sim` `graph_create_with_flags` is `cuGraphCreate` flags (identity with `create_graph_with_flags`; no Engine flag).
`gpu-sim` `create_user_object` is `cuUserObjectCreate` (identity with `user_object_create`; no Engine flag).
`gpu-sim` `retain_user_object` is `cuUserObjectRetain` (identity with `user_object_retain`; no Engine flag).
`gpu-sim` `release_user_object` is `cuUserObjectRelease` (identity with `user_object_release`; no Engine flag).
`gpu-sim` `retain_graph_user_object` is `cuGraphRetainUserObject` (identity with `graph_retain_user_object`; no Engine flag).
`gpu-sim` `release_graph_user_object` is `cuGraphReleaseUserObject` (identity with `graph_release_user_object`; no Engine flag).
`gpu-sim` `get_graph_alloc_node_params` is `cuGraphMemAllocNodeGetParams` (identity with `graph_alloc_get_params`; no Engine flag).
`gpu-sim` `get_graph_exec_alloc_node_params` is `cuGraphExecMemAllocNodeGetParams` (identity with `graph_exec_alloc_get_params`; no Engine flag).
`gpu-sim` `get_graph_free_node_params` is `cuGraphMemFreeNodeGetParams` (identity with `graph_free_get_params`; no Engine flag).
`gpu-sim` `get_graph_exec_free_node_params` is `cuGraphExecMemFreeNodeGetParams` (identity with `graph_exec_free_get_params`; no Engine flag).
`gpu-sim` `set_graph_free_node_params` is `cuGraphMemFreeNodeSetParams` (identity with `graph_free_set_params`; no Engine flag).
`gpu-sim` `set_graph_exec_free_node_params` is `cuGraphExecMemFreeNodeSetParams` (identity with `graph_exec_free_set_params`; no Engine flag).
`gpu-sim` `set_graph_conditional_params` is `cuGraphNodeSetParams` for a set-conditional node (identity with `graph_set_conditional_params`; no Engine flag).
`gpu-sim` `set_graph_exec_conditional_params` is `cuGraphExecNodeSetParams` for a set-conditional node (identity with `graph_exec_set_conditional_params`; no Engine flag).
`gpu-sim` `create_graph_conditional_handle` is `cuGraphConditionalHandleCreate` (identity with `graph_conditional_create`; no Engine flag).
`gpu-sim` `create_graph_conditional_handle_with_flags` is `cuGraphConditionalHandleCreate` flags (identity with `graph_conditional_create_with_flags`; no Engine flag).
`gpu-sim` `create_graph_conditional_handle_with_ctx` is `cuGraphConditionalHandleCreate` with a ctx argument (identity with `graph_conditional_create_with_ctx`; no Engine flag).
`gpu-sim` `stream_begin_capture` is `cuStreamBeginCapture` (identity with `begin_capture`; no Engine flag).
`gpu-sim` `stream_begin_capture_with_mode` is `cuStreamBeginCapture` with mode (identity with `begin_capture_with_mode`; no Engine flag).
`gpu-sim` `stream_begin_capture_to_graph` is `cuStreamBeginCaptureToGraph` (identity with `begin_capture_to_graph`; no Engine flag).
`gpu-sim` `stream_begin_capture_to_graph_with_mode` is `cuStreamBeginCaptureToGraph` with mode (identity with `begin_capture_to_graph_with_mode`; no Engine flag).
`gpu-sim` `stream_begin_recapture_to_graph` is `cuStreamBeginRecaptureToGraph` (identity with `begin_recapture_to_graph`; no Engine flag).
`gpu-sim` `stream_begin_recapture_to_graph_with_mode` is `cuStreamBeginRecaptureToGraph` with mode (identity with `begin_recapture_to_graph_with_mode`; no Engine flag).
`gpu-sim` `stream_begin_recapture_to_graph_with_callback` is `cuStreamBeginRecaptureToGraph` with callback (identity with `begin_recapture_to_graph_with_callback`; no Engine flag).
`gpu-sim` `stream_end_capture` is `cuStreamEndCapture` (identity with `end_capture`; no Engine flag).
`gpu-sim` `update_stream_capture_dependencies` is `cuStreamUpdateCaptureDependencies` (identity with `stream_update_capture_dependencies`; no Engine flag).
`gpu-sim` `is_stream_capturing` is `cuStreamIsCapturing` (identity with `stream_is_capturing`; no Engine flag).
`gpu-sim` `get_stream_capture_info` is `cuStreamGetCaptureInfo` (identity with `stream_capture_info`; no Engine flag).
`gpu-sim` `exchange_thread_stream_capture_mode` is `cuThreadExchangeStreamCaptureMode` (identity with `thread_exchange_stream_capture_mode`; no Engine flag).
`gpu-sim` `get_stream_capture_mode` is the thread-default `cudaStreamCaptureMode` query (identity with `stream_capture_mode`; no Engine flag).
`gpu-sim` `event_flags` is `cuEventGetFlags` (identity with `event_get_flags`; no Engine flag).
`gpu-sim` `ctx_enable_peer_access` is `cuCtxEnablePeerAccess` (identity with `enable_peer`; no Engine flag).
`gpu-sim` `ctx_enable_peer_access_with_flags` is `cuCtxEnablePeerAccess` with flags (identity with `enable_peer_with_flags`; no Engine flag).
`gpu-sim` `ctx_disable_peer_access` is `cuCtxDisablePeerAccess` (identity with `disable_peer`; no Engine flag).
`gpu-sim` `can_device_access_peer` is `cuDeviceCanAccessPeer` (identity with `device_can_access_peer`; no Engine flag).
`gpu-sim` `device_p2p_attribute` is `cuDeviceGetP2PAttribute` (identity with `device_get_p2p_attribute`; no Engine flag).
`gpu-sim` `device_nvscisync_attributes` is `cuDeviceGetNvSciSyncAttributes` (identity with `device_get_nvscisync_attributes`; no Engine flag).
`gpu-sim` `device_flush_gpu_direct_rdma_writes` is `cuFlushGPUDirectRDMAWrites` (identity with `flush_gpu_direct_rdma_writes`; no Engine flag).
`gpu-sim` `mem_alloc_pitch` is `cudaMallocPitch` (identity with `malloc_pitch`; no Engine flag).
`gpu-sim` `mem_alloc_3d` is `cudaMalloc3D` (identity with `malloc_3d`; no Engine flag).
`gpu-sim` `launch_cooperative_kernel` is `cuLaunchCooperativeKernel` (identity with `cooperative_kernel`; no Engine flag).
`gpu-sim` `launch_cooperative_kernel_bufs` is `cuLaunchCooperativeKernel` spans (identity with `cooperative_kernel_bufs`; no Engine flag).
`gpu-sim` `launch_cooperative_kernel_multi_device` is `cuLaunchCooperativeKernelMultiDevice` (identity with `cooperative_kernel_multi_device`; no Engine flag).
`gpu-sim` `mem_set` is `cudaMemsetAsync` (identity with `memset`; no Engine flag).
`gpu-sim` `mem_set_buf` is `cudaMemsetAsync` spans (identity with `memset_buf`; no Engine flag).
`gpu-sim` `mem_set_op` is `cudaMemsetAsync` / `cudaMemset2DAsync` (identity with `memset_op`; no Engine flag).
`gpu-sim` `mem_set_sync` is `cudaMemset` (identity with `memset_sync`; no Engine flag).
`gpu-sim` `mem_set_op_sync` is `cudaMemset` / `cudaMemset2D` / `cudaMemset3D` (identity with `memset_op_sync`; no Engine flag).
`gpu-sim` `mem_set_2d_async` is `cudaMemset2DAsync` (identity with `memset_2d_async`; no Engine flag).
`gpu-sim` `mem_set_2d` is `cudaMemset2D` (identity with `memset_2d`; no Engine flag).
`gpu-sim` `mem_set_3d_async` is `cudaMemset3DAsync` (identity with `memset_3d_async`; no Engine flag).
`gpu-sim` `mem_set_3d` is `cudaMemset3D` (identity with `memset_3d`; no Engine flag).
`gpu-sim` `stream_write_value64` is `cuStreamWriteValue64` (identity with `write_value64`; no Engine flag).
`gpu-sim` `stream_write_value32` is `cuStreamWriteValue32` (identity with `write_value32`; no Engine flag).
`gpu-sim` `stream_write_value64_with_flags` is `cuStreamWriteValue64` flags (identity with `write_value64_with_flags`; no Engine flag).
`gpu-sim` `stream_write_value32_with_flags` is `cuStreamWriteValue32` flags (identity with `write_value32_with_flags`; no Engine flag).
`gpu-sim` `stream_wait_value64` is `cuStreamWaitValue64` (identity with `wait_value64`; no Engine flag).
`gpu-sim` `stream_wait_value32` is `cuStreamWaitValue32` (identity with `wait_value32`; no Engine flag).
`gpu-sim` `stream_wait_value64_with_flags` is `cuStreamWaitValue64` flags (identity with `wait_value64_with_flags`; no Engine flag).
`gpu-sim` `stream_wait_value32_with_flags` is `cuStreamWaitValue32` flags (identity with `wait_value32_with_flags`; no Engine flag).
`gpu-sim` `stream_batch_mem_op` is `cuStreamBatchMemOp` (identity with `batch_mem_op`; no Engine flag).
`gpu-sim` `stream_batch_mem_op_with_flags` is `cuStreamBatchMemOp` flags (identity with `batch_mem_op_with_flags`; no Engine flag).
`gpu-sim` `launch_kernel` is `cuLaunchKernel` (identity with `kernel`; no Engine flag).
`gpu-sim` `launch_kernel_bufs` is `cuLaunchKernel` spans (identity with `kernel_bufs`; no Engine flag).
`gpu-sim` `launch_kernel_ex` is `cuLaunchKernelEx` (identity with `kernel_with`; no Engine flag).
`gpu-sim` `launch_kernel_ex_bufs` is `cuLaunchKernelEx` spans (identity with `kernel_bufs_with`; no Engine flag).
`gpu-sim` `func_set_shared_mem_config` is `cuFuncSetSharedMemConfig` (identity with `set_func_shared_mem_config`; no Engine flag).

`gpu-sim` `func_get_shared_mem_config` is `cuFuncGetSharedMemConfig` (identity with `get_func_shared_mem_config`; no Engine flag).

`gpu-sim` `func_set_cache_config` is `cuFuncSetCacheConfig` (identity with `set_func_cache_config`; no Engine flag).

`gpu-sim` `func_set_carveout` is `cuFuncSetAttribute` carveout (identity with `set_func_carveout`; no Engine flag).

`gpu-sim` `func_get_carveout` is `cuFuncGetAttribute` carveout (identity with `get_func_carveout`; no Engine flag).

`gpu-sim` `func_set_cluster_policy` is `cuFuncSetAttribute` cluster policy (identity with `set_func_cluster_policy`; no Engine flag).

`gpu-sim` `func_get_cluster_policy` is `cuFuncGetAttribute` cluster policy (identity with `get_func_cluster_policy`; no Engine flag).

`gpu-sim` `func_set_cluster_dim_must_be_set` is `cuFuncSetAttribute` cluster dim must be set (identity with `set_cluster_dim_must_be_set`; no Engine flag).

`gpu-sim` `func_get_cluster_dim_must_be_set` is `cuFuncGetAttribute` cluster dim must be set (identity with `cluster_dim_must_be_set`; no Engine flag).

`gpu-sim` `func_set_required_cluster_width` is `cuFuncSetAttribute` required cluster width (identity with `set_required_cluster_width`; no Engine flag).

`gpu-sim` `func_get_required_cluster_width` is `cuFuncGetAttribute` required cluster width (identity with `required_cluster_width`; no Engine flag).

`gpu-sim` `func_set_required_cluster_height` is `cuFuncSetAttribute` required cluster height (identity with `set_required_cluster_height`; no Engine flag).

`gpu-sim` `func_get_required_cluster_height` is `cuFuncGetAttribute` required cluster height (identity with `required_cluster_height`; no Engine flag).

`gpu-sim` `func_set_required_cluster_depth` is `cuFuncSetAttribute` required cluster depth (identity with `set_required_cluster_depth`; no Engine flag).

`gpu-sim` `func_get_required_cluster_depth` is `cuFuncGetAttribute` required cluster depth (identity with `required_cluster_depth`; no Engine flag).

`gpu-sim` `func_set_non_portable_cluster_size_allowed` is `cuFuncSetAttribute` non-portable cluster size (identity with `set_non_portable_cluster_size_allowed`; no Engine flag).

`gpu-sim` `func_get_non_portable_cluster_size_allowed` is `cuFuncGetAttribute` non-portable cluster size (identity with `non_portable_cluster_size_allowed`; no Engine flag).

`gpu-sim` `func_set_max_dynamic_shared_memory` is `cuFuncSetAttribute` max dynamic shared memory (identity with `set_max_dynamic_shared_memory`; no Engine flag).

`gpu-sim` `func_get_max_dynamic_shared_memory` is `cuFuncGetAttribute` max dynamic shared memory (identity with `max_dynamic_shared_memory`; no Engine flag).

`gpu-sim` `event_create_disable_timing` is `cuEventCreateWithFlags` disable timing (identity with `create_event_disable_timing`; no Engine flag).

`gpu-sim` `event_create_interprocess` is `cuEventCreateWithFlags` interprocess (identity with `create_event_interprocess`; no Engine flag).

`gpu-sim` `event_create_blocking_sync` is `cuEventCreateWithFlags` blocking sync (identity with `create_event_blocking_sync`; no Engine flag).

`gpu-sim` `event_record_external` is `cuEventRecordWithFlags` external (identity with `record_event_external`; no Engine flag).

`gpu-sim` `stream_wait_event_external` is `cuStreamWaitEvent` external (identity with `wait_event_external`; no Engine flag).
`gpu-sim` `stream_set_mem_sync_domain` is `cuStreamSetAttribute` mem sync domain (identity with `set_stream_mem_sync_domain`; no Engine flag).
`gpu-sim` `stream_set_mem_sync_domain_map` is `cuStreamSetAttribute` mem sync domain map (identity with `set_stream_mem_sync_domain_map`; no Engine flag).
`gpu-sim` `stream_get_mem_sync_domain` is `cuStreamGetAttribute` mem sync domain (identity with `stream_mem_sync_domain`; no Engine flag).
`gpu-sim` `stream_get_mem_sync_domain_map` is `cuStreamGetAttribute` mem sync domain map (identity with `stream_mem_sync_domain_map`; no Engine flag).
`gpu-sim` `stream_set_sync_policy` is `cuStreamSetAttribute` sync policy (identity with `set_stream_sync_policy`; no Engine flag).
`gpu-sim` `stream_get_sync_policy` is `cuStreamGetAttribute` sync policy (identity with `stream_sync_policy`; no Engine flag).
`gpu-sim` `stream_set_nvlink_util_centric` is `cuStreamSetAttribute` nvlink util centric (identity with `set_stream_nvlink_util_centric`; no Engine flag).
`gpu-sim` `stream_get_nvlink_util_centric` is `cuStreamGetAttribute` nvlink util centric (identity with `stream_nvlink_util_centric`; no Engine flag).
`gpu-sim` `stream_set_access_policy` is `cuStreamSetAttribute` access policy (identity with `set_stream_access_policy`; no Engine flag).
`gpu-sim` `stream_get_access_policy` is `cuStreamGetAttribute` access policy (identity with `stream_access_policy`; no Engine flag).
`gpu-sim` `stream_set_priority` is `cuStreamSetAttribute` priority (identity with `set_stream_priority`; no Engine flag).
`gpu-sim` `stream_set_blocking` is `cuStreamCreate` blocking (identity with `set_stream_blocking`; no Engine flag).
`gpu-sim` `get_func_attributes` is `cuFuncGetAttributes` (identity with `func_get_attributes`; no Engine flag).
`gpu-sim` `get_device_name` is `cuDeviceGetName` (identity with `device_get_name`; no Engine flag).
`gpu-sim` `get_device_count` is `cuDeviceGetCount` (identity with `device_count`; no Engine flag).
`gpu-sim` `device_get_default_mempool` is `cuDeviceGetDefaultMemPool` (identity with `default_pool`; no Engine flag).
`gpu-sim` `device_get_mempool` is `cuDeviceGetMemPool` (identity with `device_mempool`; no Engine flag).
`gpu-sim` `device_set_mempool` is `cuDeviceSetMemPool` (identity with `set_device_mempool`; no Engine flag).
`gpu-sim` `mem_pool_create` is `cuMemPoolCreate` (identity with `create_pool`; no Engine flag).
`gpu-sim` `mem_pool_create_shareable` is `cuMemPoolCreate` POSIX (identity with `create_shareable_pool`; no Engine flag).
`gpu-sim` `mem_pool_create_with_props` is `cuMemPoolCreate` with props (identity with `create_pool_with_props`; no Engine flag).
`gpu-sim` `mem_pool_destroy` is `cuMemPoolDestroy` (identity with `destroy_pool`; no Engine flag).
`gpu-sim` `mem_alloc_from_pool` is `cuMemAllocFromPoolAsync` (identity with `alloc_from_pool`; no Engine flag).
`gpu-sim` `mem_pool_export` is `cuMemPoolExportToShareableHandle` (identity with `pool_export`; no Engine flag).
`gpu-sim` `mem_pool_import` is `cuMemPoolImportFromShareableHandle` (identity with `pool_import`; no Engine flag).
`gpu-sim` `mem_pool_export_with_type` is `cuMemPoolExportToShareableHandle` type (identity with `pool_export_with_type`; no Engine flag).
`gpu-sim` `mem_pool_import_with_type` is `cuMemPoolImportFromShareableHandle` type (identity with `pool_import_with_type`; no Engine flag).
`gpu-sim` `mem_pool_export_ptr` is `cuMemPoolExportPointer` (identity with `pool_export_ptr`; no Engine flag).
`gpu-sim` `mem_pool_import_ptr` is `cuMemPoolImportPointer` (identity with `pool_import_ptr`; no Engine flag).
`gpu-sim` `mem_pool_get_access` is `cuMemPoolGetAccess` (identity with `pool_get_access`; no Engine flag).
`gpu-sim` `mem_pool_set_access` is `cuMemPoolSetAccess` (identity with `pool_set_access`; no Engine flag).
`gpu-sim` `mem_pool_set_access_read` is `cuMemPoolSetAccess` ProtRead (identity with `pool_set_access_read`; no Engine flag).
`gpu-sim` `mem_pool_set_access_with_flags` is `cuMemPoolSetAccess` flags (identity with `pool_set_access_with_flags`; no Engine flag).
`gpu-sim` `mem_pool_set_access_n` is `cuMemPoolSetAccess` n (identity with `pool_set_access_n`; no Engine flag).
`gpu-sim` `mem_pool_unset_access` is `cuMemPoolSetAccess` ProtNone (identity with `pool_unset_access`; no Engine flag).
`gpu-sim` `mem_pool_get_attribute` is `cuMemPoolGetAttribute` (identity with `pool_get_attribute`; no Engine flag).
`gpu-sim` `mem_pool_set_attribute` is `cuMemPoolSetAttribute` (identity with `pool_set_attribute`; no Engine flag).
`gpu-sim` `mem_pool_trim_to` is `cuMemPoolTrimTo` (identity with `pool_trim_to`; no Engine flag).
`gpu-sim` `mem_pool_set_release_threshold` is `cuMemPoolSetAttribute` ReleaseThreshold (identity with `set_pool_release_threshold`; no Engine flag).
`gpu-sim` `mem_pool_set_max_size` is `cuMemPoolSetAttribute` MaxPoolSize (identity with `set_pool_max_size`; no Engine flag).
`gpu-sim` `mem_get_allocation_granularity` is `cuMemGetAllocationGranularity` (identity with `va_get_allocation_granularity`; no Engine flag).
`gpu-sim` `mem_create` is `cuMemCreate` (identity with `va_create`; no Engine flag).
`gpu-sim` `mem_create_with_prop` is `cuMemCreate` props (identity with `va_create_with_prop`; no Engine flag).
`gpu-sim` `mem_map_handle` is `cuMemMap` (identity with `va_map_handle`; no Engine flag).
`gpu-sim` `mem_map_handle_with_flags` is `cuMemMap` flags (identity with `va_map_handle_with_flags`; no Engine flag).
`gpu-sim` `mem_map_handle_with_size` is `cuMemMap` size (identity with `va_map_handle_with_size`; no Engine flag).
`gpu-sim` `mem_alloc` is `cuMemAlloc` (identity with `malloc`; no Engine flag).
`gpu-sim` `mem_free` is `cuMemFree` (identity with `free_sync`; no Engine flag).
`gpu-sim` `mem_free_host` is `cuMemFreeHost` (identity with `free_host_pinned`; no Engine flag).
`gpu-sim` `mem_host_alloc` is `cuMemHostAlloc` (identity with `alloc_host_with_flags`; no Engine flag).
`gpu-sim` `mem_host_get_flags` is `cuMemHostGetFlags` (identity with `host_get_flags`; no Engine flag).
`gpu-sim` `mem_host_get_device_pointer` is `cuMemHostGetDevicePointer` (identity with `host_get_device_pointer_with_flags`; no Engine flag).
`gpu-sim` `mem_host_register` is `cuMemHostRegister` (identity with `host_register_with_flags`; no Engine flag).
`gpu-sim` `mem_host_unregister` is `cuMemHostUnregister` (identity with `host_unregister`; no Engine flag).
`gpu-sim` `mem_host_register_with_size` is `cuMemHostRegister` size (identity with `host_register_with_size`; no Engine flag).
`gpu-sim` `ipc_get_mem_handle` is `cuIpcGetMemHandle` (identity with `ipc_get`; no Engine flag).
`gpu-sim` `ipc_open_mem_handle` is `cuIpcOpenMemHandle` (identity with `ipc_open_with_flags`; no Engine flag).
`gpu-sim` `ipc_close_mem_handle` is `cuIpcCloseMemHandle` (identity with `ipc_close`; no Engine flag).
`gpu-sim` `ipc_get_event_handle` is `cuIpcGetEventHandle` (identity with `ipc_get_event`; no Engine flag).
`gpu-sim` `ipc_open_event_handle` is `cuIpcOpenEventHandle` (identity with `ipc_open_event`; no Engine flag).
`gpu-sim` `mem_alloc_host` is `cuMemAllocHost` (identity with `alloc_host_pinned`; no Engine flag).
`gpu-sim` `mem_alloc_managed` is `cuMemAllocManaged` (identity with `alloc_managed_with_flags`; no Engine flag).
`gpu-sim` `mem_alloc_async` is `cuMemAllocAsync` (identity with `alloc`; no Engine flag).
`gpu-sim` `mem_free_async` is `cuMemFreeAsync` (identity with `free`; no Engine flag).
`gpu-sim` `mem_advise_n` is `cuMemAdvise` (identity with `mem_advise_with_size`; no Engine flag).
`gpu-sim` `mem_prefetch` is `cuMemPrefetchAsync` (identity with `prefetch`; no Engine flag).
`gpu-sim` `mem_prefetch_v2` is `cuMemPrefetchAsync_v2` (identity with `prefetch_with_flags`; no Engine flag).
`gpu-sim` `mem_prefetch_n` is `cuMemPrefetchAsync` count (identity with `prefetch_with_size`; no Engine flag).
`gpu-sim` `mem_prefetch_host` is host dest `cuMemPrefetchAsync` (identity with `prefetch_host`; no Engine flag).
`gpu-sim` `mem_prefetch_host_n` is host dest `cuMemPrefetchAsync` count (identity with `prefetch_host_with_size`; no Engine flag).
`gpu-sim` `mem_advise_v2` is `cuMemAdvise_v2` (identity with `mem_advise_with_location`; no Engine flag).
`gpu-sim` `mem_range_get` is `cuMemRangeGetAttribute` (identity with `mem_range_get_attribute`; no Engine flag).
`gpu-sim` `mem_range_get_n` is `cuMemRangeGetAttribute` count (identity with `mem_range_get_attribute_with_size`; no Engine flag).
`gpu-sim` `mem_range_gets` is `cuMemRangeGetAttributes` (identity with `mem_range_get_attributes`; no Engine flag).
`gpu-sim` `mem_range_gets_n` is `cuMemRangeGetAttributes` count (identity with `mem_range_get_attributes_with_size`; no Engine flag).
`gpu-sim` `mem_range_get_data` is `cuMemRangeGetAttribute` dataSize (identity with `mem_range_get_attribute_with_data_size`; no Engine flag).
`gpu-sim` `mem_range_gets_data` is `cuMemRangeGetAttributes` dataSizes (identity with `mem_range_get_attributes_with_data_sizes`; no Engine flag).
`gpu-sim` `stream_attach_mem` is `cuStreamAttachMemAsync` (identity with `stream_attach`; no Engine flag).
`gpu-sim` `stream_attach_n` is `cuStreamAttachMemAsync` length (identity with `stream_attach_with_size`; no Engine flag).
`gpu-sim` `stream_attach_flags` is `cuStreamAttachMemAsync` flags (identity with `stream_attach_with_flags`; no Engine flag).
`gpu-sim` `memcpy_async` is `cuMemcpyAsync` (identity with `memcpy`; no Engine flag).
`gpu-sim` `mem_cpy` is `cuMemcpy` (identity with `memcpy_sync`; no Engine flag).
`gpu-sim` `mem_address_range` is `cuMemGetAddressRange` (identity with `mem_get_address_range`; no Engine flag).
`gpu-sim` `mem_cpy_2d` is `cuMemcpy2D` (identity with `memcpy_2d`; no Engine flag).
`gpu-sim` `mem_cpy_2d_async` is `cuMemcpy2DAsync` (identity with `memcpy_2d_async`; no Engine flag).
`gpu-sim` `mem_cpy_3d` is `cuMemcpy3D` (identity with `memcpy_3d`; no Engine flag).
`gpu-sim` `mem_cpy_3d_async` is `cuMemcpy3DAsync` (identity with `memcpy_3d_async`; no Engine flag).
`gpu-sim` `mem_cpy_peer` is `cuMemcpyPeer` (identity with `memcpy_peer`; no Engine flag).
`gpu-sim` `mem_cpy_peer_async` is `cuMemcpyPeerAsync` (identity with `memcpy_peer_async`; no Engine flag).
`gpu-sim` `mem_cpy_peer_3d` is `cuMemcpy3DPeer` (identity with `memcpy_peer_3d`; no Engine flag).
`gpu-sim` `mem_cpy_peer_3d_async` is `cuMemcpy3DPeerAsync` (identity with `memcpy_peer_3d_async`; no Engine flag).
`gpu-sim` `mem_cpy_peer_2d` is `cuMemcpy2DPeer` (identity with `memcpy_peer_2d`; no Engine flag).
`gpu-sim` `mem_cpy_peer_2d_async` is `cuMemcpy2DPeerAsync` (identity with `memcpy_peer_2d_async`; no Engine flag).
`gpu-sim` `mem_cpy_batch_async` is `cuMemcpyBatchAsync` (identity with `memcpy_batch_async`; no Engine flag).
`gpu-sim` `mem_cpy_3d_batch_async` is `cuMemcpy3DBatchAsync` (identity with `memcpy_3d_batch_async`; no Engine flag).
`gpu-sim` `mem_cpy_3d_with_attributes` is `cuMemcpy3DWithAttributesAsync` (identity with `memcpy_3d_with_attributes`; no Engine flag).
`gpu-sim` `mem_cpy_with_attributes` is `cuMemcpyWithAttributesAsync` (identity with `memcpy_with_attributes`; no Engine flag).
`gpu-sim` `ctx_set_flags` is `cuCtxSetFlags` (identity with `set_device_flags`; no Engine flag).
`gpu-sim` `ctx_set_cache_config` is `cuCtxSetCacheConfig` (identity with `set_cache_config`; no Engine flag).
`gpu-sim` `ctx_set_limit` is `cuCtxSetLimit` (identity with `set_limit`; no Engine flag).
`gpu-sim` `ctx_set_shared_mem_config` is `cuCtxSetSharedMemConfig` (identity with `set_shared_mem_config`; no Engine flag).
`gpu-sim` `stream_create_priority` is `cuStreamCreateWithPriority` (identity with `stream_create_with_priority`; no Engine flag).
`gpu-sim` `stream_create_flags` is `cuStreamCreateWithFlags` (identity with `stream_create_with_flags`; no Engine flag).
`gpu-sim` `stream_flags` is `cuStreamGetFlags` (identity with `stream_get_flags`; no Engine flag).
`gpu-sim` `get_stream_priority` is `cuStreamGetPriority` (identity with `stream_get_priority`; no Engine flag).
`gpu-sim` `device_graph_mem_get` is `cuDeviceGetGraphMemAttribute` (identity with `graph_mem_get`; no Engine flag).
`gpu-sim` `device_graph_mem_set` is `cuDeviceSetGraphMemAttribute` (identity with `graph_mem_set`; no Engine flag).
`gpu-sim` `device_graph_mem_trim` is `cuDeviceGraphMemTrim` (identity with `graph_mem_trim`; no Engine flag).
`gpu-sim` `get_stream_id` is `cuStreamGetId` (identity with `stream_get_id`; no Engine flag).
`gpu-sim` `copy_stream_attributes` is `cuStreamCopyAttributes` (identity with `stream_copy_attributes`; no Engine flag).
`gpu-sim` `get_stream_attribute` is `cuStreamGetAttribute` (identity with `stream_get_attribute`; no Engine flag).
`gpu-sim` `set_stream_attribute` is `cuStreamSetAttribute` (identity with `stream_set_attribute`; no Engine flag).
`gpu-sim` `get_graph_kernel_node_attribute` is `cuGraphKernelNodeGetAttribute` (identity with `graph_kernel_node_get_attribute`; no Engine flag).
`gpu-sim` `set_graph_kernel_node_attribute` is `cuGraphKernelNodeSetAttribute` (identity with `graph_kernel_node_set_attribute`; no Engine flag).
`gpu-sim` `get_graph_exec_kernel_node_attribute` is `cuGraphExecKernelNodeGetAttribute` (identity with `graph_exec_kernel_node_get_attribute`; no Engine flag).
`gpu-sim` `set_graph_exec_kernel_node_attribute` is `cuGraphExecKernelNodeSetAttribute` (identity with `graph_exec_kernel_node_set_attribute`; no Engine flag).
`gpu-sim` `copy_graph_kernel_node_attributes` is `cuGraphKernelNodeCopyAttributes` (identity with `graph_kernel_node_copy_attributes`; no Engine flag).
`gpu-sim` `copy_graph_exec_kernel_node_attributes` is `cuGraphExecKernelNodeCopyAttributes` (identity with `graph_exec_kernel_node_copy_attributes`; no Engine flag).
`gpu-sim` `get_graph_kernel_node_params` is `cuGraphKernelNodeGetParams` (identity with `graph_kernel_get_params`; no Engine flag).
`gpu-sim` `get_graph_exec_kernel_node_params` is `cuGraphExecKernelNodeGetParams` (identity with `graph_exec_kernel_get_params`; no Engine flag).
`gpu-sim` `set_graph_kernel_node_params` is `cuGraphKernelNodeSetParams` (identity with `graph_kernel_set_params`; no Engine flag).
`gpu-sim` `set_graph_exec_kernel_node_params` is `cuGraphExecKernelNodeSetParams` (identity with `graph_exec_kernel_set_params`; no Engine flag).
`gpu-sim` `get_graph_memcpy_node_params` is `cuGraphMemcpyNodeGetParams` (identity with `graph_memcpy_get_params`; no Engine flag).
`gpu-sim` `get_graph_exec_memcpy_node_params` is `cuGraphExecMemcpyNodeGetParams` (identity with `graph_exec_memcpy_get_params`; no Engine flag).
`gpu-sim` `set_graph_memcpy_node_params` is `cuGraphMemcpyNodeSetParams` (identity with `graph_memcpy_set_params`; no Engine flag).
`gpu-sim` `set_graph_exec_memcpy_node_params` is `cuGraphExecMemcpyNodeSetParams` (identity with `graph_exec_memcpy_set_params`; no Engine flag).
`gpu-sim` `get_graph_memset_node_params` is `cuGraphMemsetNodeGetParams` (identity with `graph_memset_get_params`; no Engine flag).
`gpu-sim` `get_graph_exec_memset_node_params` is `cuGraphExecMemsetNodeGetParams` (identity with `graph_exec_memset_get_params`; no Engine flag).
`gpu-sim` `set_graph_memset_node_params` is `cuGraphMemsetNodeSetParams` (identity with `graph_memset_set_params`; no Engine flag).
`gpu-sim` `set_graph_exec_memset_node_params` is `cuGraphExecMemsetNodeSetParams` (identity with `graph_exec_memset_set_params`; no Engine flag).
`gpu-sim` `get_graph_host_node_params` is `cuGraphHostNodeGetParams` (identity with `graph_host_get_params`; no Engine flag).
`gpu-sim` `get_graph_exec_host_node_params` is `cuGraphExecHostNodeGetParams` (identity with `graph_exec_host_get_params`; no Engine flag).
`gpu-sim` `set_graph_host_node_params` is `cuGraphHostNodeSetParams` (identity with `graph_host_set_params`; no Engine flag).
`gpu-sim` `set_graph_exec_host_node_params` is `cuGraphExecHostNodeSetParams` (identity with `graph_exec_host_set_params`; no Engine flag).
`gpu-sim` `get_graph_batch_mem_op_node_params` is `cuGraphBatchMemOpNodeGetParams` (identity with `graph_batch_mem_ops_get_params`; no Engine flag).
`gpu-sim` `get_graph_exec_batch_mem_op_node_params` is `cuGraphExecBatchMemOpNodeGetParams` (identity with `graph_exec_batch_mem_ops_get_params`; no Engine flag).
`gpu-sim` `set_graph_batch_mem_op_node_params` is `cuGraphBatchMemOpNodeSetParams` (identity with `graph_batch_mem_op_set_params`; no Engine flag).
`gpu-sim` `set_graph_exec_batch_mem_op_node_params` is `cuGraphExecBatchMemOpNodeSetParams` (identity with `graph_exec_batch_mem_op_set_params`; no Engine flag).
`gpu-sim` `set_graph_event_record_node_event` is `cuGraphEventRecordNodeSetEvent` (identity with `graph_event_record_set_event`; no Engine flag).
`gpu-sim` `set_graph_exec_event_record_node_event` is `cuGraphExecEventRecordNodeSetEvent` (identity with `graph_exec_event_record_set_event`; no Engine flag).
`gpu-sim` `set_graph_event_wait_node_event` is `cuGraphEventWaitNodeSetEvent` (identity with `graph_event_wait_set_event`; no Engine flag).
`gpu-sim` `set_graph_exec_event_wait_node_event` is `cuGraphExecEventWaitNodeSetEvent` (identity with `graph_exec_event_wait_set_event`; no Engine flag).
`gpu-sim` `get_graph_event_record_node_event` is `cuGraphEventRecordNodeGetEvent` (identity with `graph_event_record_get_event`; no Engine flag).
`gpu-sim` `get_graph_exec_event_record_node_event` is `cuGraphExecEventRecordNodeGetEvent` (identity with `graph_exec_event_record_get_event`; no Engine flag).
`gpu-sim` `get_graph_event_wait_node_event` is `cuGraphEventWaitNodeGetEvent` (identity with `graph_event_wait_get_event`; no Engine flag).
`gpu-sim` `get_graph_exec_event_wait_node_event` is `cuGraphExecEventWaitNodeGetEvent` (identity with `graph_exec_event_wait_get_event`; no Engine flag).
`gpu-sim` `get_graph_child_graph_node_graph` is `cuGraphChildGraphNodeGetGraph` (identity with `graph_child_get_graph`; no Engine flag).
`gpu-sim` `get_graph_exec_child_graph_node_graph` is `cuGraphExecChildGraphNodeGetGraph` (identity with `graph_exec_child_get_graph`; no Engine flag).
`gpu-sim` `set_graph_child_graph_node_params` is `cuGraphChildGraphNodeSetParams` (identity with `graph_child_set_params`; no Engine flag).
`gpu-sim` `set_graph_exec_child_graph_node_params` is `cuGraphExecChildGraphNodeSetParams` (identity with `graph_exec_child_set_params`; no Engine flag).
`gpu-sim` `set_graph_node_params` is `cuGraphNodeSetParams` (identity with `graph_node_set_params`; no Engine flag).
`gpu-sim` `set_graph_exec_node_params` is `cuGraphExecNodeSetParams` (identity with `graph_exec_node_set_params`; no Engine flag).
`gpu-sim` `get_graph_node_params` is `cuGraphNodeGetParams` (identity with `graph_node_get_params`; no Engine flag).
`gpu-sim` `get_graph_exec_node_params` is `cuGraphExecNodeGetParams` (identity with `graph_exec_node_get_params`; no Engine flag).
`gpu-sim` `set_graph_node_enabled` is `cuGraphNodeSetEnabled` (identity with `graph_node_set_enabled`; no Engine flag).
`gpu-sim` `get_graph_node_enabled` is `cuGraphNodeGetEnabled` (identity with `graph_node_get_enabled`; no Engine flag).
`gpu-sim` `get_graph_exec_flags` is `cuGraphExecGetFlags` (identity with `graph_exec_get_flags`; no Engine flag).
`gpu-sim` `get_graph_id` is `cuGraphGetId` (identity with `graph_get_id`; no Engine flag).
`gpu-sim` `get_graph_exec_id` is `cuGraphExecGetId` (identity with `graph_get_id`; no Engine flag).
`gpu-sim` `get_graph_nodes` is `cuGraphGetNodes` (identity with `graph_nodes`; no Engine flag).
`gpu-sim` `get_graph_root_nodes` is `cuGraphGetRootNodes` (identity with `graph_root_nodes`; no Engine flag).
`gpu-sim` `get_graph_edges` is `cuGraphGetEdges` (identity with `graph_edges`; no Engine flag).
`gpu-sim` `get_graph_edges_with_data` is `cuGraphGetEdges` v2 (identity with `graph_edges_with_data`; no Engine flag).
`gpu-sim` `get_graph_node_dependencies` is `cuGraphNodeGetDependencies` (identity with `graph_node_deps`; no Engine flag).
`gpu-sim` `get_graph_node_dependencies_with_data` is `cuGraphNodeGetDependencies` v2 (identity with `graph_node_deps_with_data`; no Engine flag).
`gpu-sim` `get_graph_node_dependent_nodes` is `cuGraphNodeGetDependentNodes` (identity with `graph_node_dependents`; no Engine flag).
`gpu-sim` `get_graph_node_dependent_nodes_with_data` is `cuGraphNodeGetDependentNodes` v2 (identity with `graph_node_dependents_with_data`; no Engine flag).
`gpu-sim` `get_graph_node_type` is `cuGraphNodeGetType` (identity with `graph_node_kind`; no Engine flag).
`gpu-sim` `find_graph_node_in_clone` is `cuGraphNodeFindInClone` (identity with `graph_node_find_in_clone`; no Engine flag).
`gpu-sim` `graph_clone` is `cuGraphClone` (identity with `clone_graph`; no Engine flag).
`gpu-sim` `graph_debug_dot_print` is `cuGraphDebugDotPrint` (identity with `graph_debug_dot`; no Engine flag).
`gpu-sim` `graph_debug_dot_print_with_flags` is `cuGraphDebugDotPrint` with flags (identity with `graph_debug_dot_with_flags`; no Engine flag).
`gpu-sim` `graph_instantiate` is `cuGraphInstantiate` (identity with `instantiate_graph`; no Engine flag).
`gpu-sim` `graph_instantiate_with_flags` is `cuGraphInstantiateWithFlags` (identity with `instantiate_graph_with_flags`; no Engine flag).
`gpu-sim` `graph_instantiate_with_params` is `cuGraphInstantiateWithParams` (identity with `instantiate_graph_with_params`; no Engine flag).
`gpu-sim` `graph_launch` is `cuGraphLaunch` (identity with `launch_graph`; no Engine flag).
`gpu-sim` `graph_upload` is `cuGraphUpload` (identity with `upload_graph`; no Engine flag).
`gpu-sim` `graph_upload_async` is `cuGraphUpload` on a stream (identity with `upload_graph_async`; no Engine flag).
`gpu-sim` `graph_destroy` is `cuGraphDestroy` (identity with `destroy_graph`; no Engine flag).
`gpu-sim` `graph_exec_destroy` is `cuGraphExecDestroy` (identity with `destroy_graph`; no Engine flag).
`gpu-sim` `graph_exec_update` is `cuGraphExecUpdate` (identity with `update_graph`; no Engine flag).
`gpu-sim` `graph_exec_update_with_info` is `cuGraphExecUpdate` with info (identity with `update_graph_with_info`; no Engine flag).
`gpu-sim` `add_graph_dependencies` is `cuGraphAddDependencies` (identity with `graph_add_dependencies`; no Engine flag).
`gpu-sim` `add_graph_dependencies_n` is `cuGraphAddDependencies` of pairs (identity with `graph_add_dependencies_n`; no Engine flag).
`gpu-sim` `add_graph_dependencies_with_data` is `cuGraphAddDependencies` with data (identity with `graph_add_dependencies_with_data`; no Engine flag).
`gpu-sim` `add_graph_dependencies_n_with_data` is `cuGraphAddDependencies` v2 (identity with `graph_add_dependencies_n_with_data`; no Engine flag).
`gpu-sim` `remove_graph_dependencies` is `cuGraphRemoveDependencies` (identity with `graph_remove_dependencies`; no Engine flag).
`gpu-sim` `remove_graph_dependencies_n` is `cuGraphRemoveDependencies` of pairs (identity with `graph_remove_dependencies_n`; no Engine flag).
`gpu-sim` `remove_graph_dependencies_with_data` is `cuGraphRemoveDependencies` with data (identity with `graph_remove_dependencies_with_data`; no Engine flag).
`gpu-sim` `remove_graph_dependencies_n_with_data` is `cuGraphRemoveDependencies` v2 (identity with `graph_remove_dependencies_n_with_data`; no Engine flag).
`gpu-sim` `destroy_graph_node` is `cuGraphDestroyNode` (identity with `graph_destroy_node`; no Engine flag).
`gpu-sim` `launch_device_graph` is device-side `cuGraphLaunch` (identity with `device_launch_graph`; no Engine flag).
`gpu-sim` `get_current_graph_exec` is `cuGetCurrentGraphExec` (identity with `current_graph_exec`; no Engine flag).
`gpu-sim` `add_graph_empty` is `cuGraphAddEmptyNode` (identity with `graph_add_empty`; no Engine flag).
`gpu-sim` `add_graph_child` is `cuGraphAddChildGraphNode` (identity with `graph_add_child`; no Engine flag).
`gpu-sim` `add_graph_host` is `cuGraphAddHostNode` (identity with `graph_add_host_func_params`; no Engine flag).
`gpu-sim` `add_graph_event_record` is `cuGraphAddEventRecordNode` (identity with `graph_add_event_record`; no Engine flag).
`gpu-sim` `add_graph_event_wait` is `cuGraphAddEventWaitNode` (identity with `graph_add_event_wait`; no Engine flag).
`gpu-sim` `add_graph_kernel` is `cuGraphAddKernelNode` (identity with `graph_add_kernel`; no Engine flag).
`gpu-sim` `add_graph_memcpy` is `cuGraphAddMemcpyNode` (identity with `graph_add_memcpy`; no Engine flag).
`gpu-sim` `add_graph_memcpy_1d` is `cuGraphAddMemcpyNode1D` (identity with `graph_add_memcpy_1d`; no Engine flag).
`gpu-sim` `add_graph_memcpy_2d` is 2D `cuGraphAddMemcpyNode` (identity with `graph_add_memcpy_2d`; no Engine flag).
`gpu-sim` `add_graph_memcpy_3d` is 3D `cuGraphAddMemcpyNode` (identity with `graph_add_memcpy_3d`; no Engine flag).
`gpu-sim` `add_graph_memset` is packed 1D `cuGraphAddMemsetNode` (identity with `graph_add_memset`; no Engine flag).
`gpu-sim` `add_graph_memset_op` is `cuGraphAddMemsetNode` params (identity with `graph_add_memset_op`; no Engine flag).
`gpu-sim` `add_graph_memset_2d` is 2D `cuGraphAddMemsetNode` (identity with `graph_add_memset_2d`; no Engine flag).
`gpu-sim` `add_graph_memset_3d` is 3D `cuGraphAddMemsetNode` (identity with `graph_add_memset_3d`; no Engine flag).
`gpu-sim` `add_graph_batch_mem_op` is `cuGraphAddBatchMemOpNode` (identity with `graph_add_batch_mem_op`; no Engine flag).
`gpu-sim` `add_graph_batch_mem_op_with_flags` is `cuGraphAddBatchMemOpNode` flags (identity with `graph_add_batch_mem_op_with_flags`; no Engine flag).
`gpu-sim` `add_graph_alloc` is `cuGraphAddMemAllocNode` (identity with `graph_add_alloc`; no Engine flag).
`gpu-sim` `add_graph_alloc_with_access` is `cuGraphAddMemAllocNode` access (identity with `graph_add_alloc_with_access`; no Engine flag).
`gpu-sim` `add_graph_free` is `cuGraphAddMemFreeNode` (identity with `graph_add_free`; no Engine flag).
`gpu-sim` `add_graph_node` is `cuGraphAddNode` (identity with `graph_add_node`; no Engine flag).
`gpu-sim` `add_graph_node_with_data` is `cuGraphAddNode_v2` (identity with `graph_add_node_with_data`; no Engine flag).
`gpu-sim` `add_graph_if` is `cuGraphAddNode` IF (identity with `graph_add_if`; no Engine flag).
`gpu-sim` `add_graph_if_else` is `cuGraphAddNode` IF size 2 (identity with `graph_add_if_else`; no Engine flag).
`gpu-sim` `add_graph_while` is `cuGraphAddNode` WHILE (identity with `graph_add_while`; no Engine flag).
`gpu-sim` `add_graph_switch` is `cuGraphAddNode` SWITCH (identity with `graph_add_switch`; no Engine flag).
`gpu-sim` `add_graph_set_conditional` is graph-build `cuGraphSetConditional` (identity with `graph_add_set_conditional`; no Engine flag).
`gpu-sim` `add_graph_write_value64` is graph `cuStreamWriteValue64` (identity with `graph_add_write_value64`; no Engine flag).
`gpu-sim` `add_graph_write_value32` is graph `cuStreamWriteValue32` (identity with `graph_add_write_value32`; no Engine flag).
`gpu-sim` `add_graph_write_value64_with_flags` is graph `cuStreamWriteValue64` flags (identity with `graph_add_write_value64_with_flags`; no Engine flag).
`gpu-sim` `add_graph_write_value32_with_flags` is graph `cuStreamWriteValue32` flags (identity with `graph_add_write_value32_with_flags`; no Engine flag).
`gpu-sim` `add_graph_wait_value64` is graph `cuStreamWaitValue64` (identity with `graph_add_wait_value64`; no Engine flag).
`gpu-sim` `add_graph_wait_value32` is graph `cuStreamWaitValue32` (identity with `graph_add_wait_value32`; no Engine flag).
`gpu-sim` `add_graph_wait_value64_with_flags` is graph `cuStreamWaitValue64` flags (identity with `graph_add_wait_value64_with_flags`; no Engine flag).
`gpu-sim` `add_graph_wait_value32_with_flags` is graph `cuStreamWaitValue32` flags (identity with `graph_add_wait_value32_with_flags`; no Engine flag).
`gpu-sim` `add_graph_cooperative_kernel` is graph cooperative `cudaGraphAddKernelNode` (identity with `graph_add_cooperative_kernel`; no Engine flag).
`gpu-sim` `add_graph_host_func` is graph unnamed `cudaGraphAddHostNode` (identity with `graph_add_host_func`; no Engine flag).
`gpu-sim` `set_graph_memcpy_node_params_1d` is graph `cudaGraphMemcpyNodeSetParams1D` (identity with `graph_memcpy_set_params_1d`; no Engine flag).
`gpu-sim` `set_graph_exec_memcpy_node_params_1d` is graph `cudaGraphExecMemcpyNodeSetParams1D` (identity with `graph_exec_memcpy_set_params_1d`; no Engine flag).
`gpu-sim` `graph_create` is `cuGraphCreate` (identity with `create_graph`; no Engine flag).
`gpu-sim` `graph_create_with_flags` is `cuGraphCreate` flags (identity with `create_graph_with_flags`; no Engine flag).
`gpu-sim` `create_user_object` is `cuUserObjectCreate` (identity with `user_object_create`; no Engine flag).
`gpu-sim` `retain_user_object` is `cuUserObjectRetain` (identity with `user_object_retain`; no Engine flag).
`gpu-sim` `release_user_object` is `cuUserObjectRelease` (identity with `user_object_release`; no Engine flag).
`gpu-sim` `retain_graph_user_object` is `cuGraphRetainUserObject` (identity with `graph_retain_user_object`; no Engine flag).
`gpu-sim` `release_graph_user_object` is `cuGraphReleaseUserObject` (identity with `graph_release_user_object`; no Engine flag).
`gpu-sim` `get_graph_alloc_node_params` is `cuGraphMemAllocNodeGetParams` (identity with `graph_alloc_get_params`; no Engine flag).
`gpu-sim` `get_graph_exec_alloc_node_params` is `cuGraphExecMemAllocNodeGetParams` (identity with `graph_exec_alloc_get_params`; no Engine flag).
`gpu-sim` `get_graph_free_node_params` is `cuGraphMemFreeNodeGetParams` (identity with `graph_free_get_params`; no Engine flag).
`gpu-sim` `get_graph_exec_free_node_params` is `cuGraphExecMemFreeNodeGetParams` (identity with `graph_exec_free_get_params`; no Engine flag).
`gpu-sim` `set_graph_free_node_params` is `cuGraphMemFreeNodeSetParams` (identity with `graph_free_set_params`; no Engine flag).
`gpu-sim` `set_graph_exec_free_node_params` is `cuGraphExecMemFreeNodeSetParams` (identity with `graph_exec_free_set_params`; no Engine flag).
`gpu-sim` `set_graph_conditional_params` is `cuGraphNodeSetParams` for a set-conditional node (identity with `graph_set_conditional_params`; no Engine flag).
`gpu-sim` `set_graph_exec_conditional_params` is `cuGraphExecNodeSetParams` for a set-conditional node (identity with `graph_exec_set_conditional_params`; no Engine flag).
`gpu-sim` `create_graph_conditional_handle` is `cuGraphConditionalHandleCreate` (identity with `graph_conditional_create`; no Engine flag).
`gpu-sim` `create_graph_conditional_handle_with_flags` is `cuGraphConditionalHandleCreate` flags (identity with `graph_conditional_create_with_flags`; no Engine flag).
`gpu-sim` `create_graph_conditional_handle_with_ctx` is `cuGraphConditionalHandleCreate` with a ctx argument (identity with `graph_conditional_create_with_ctx`; no Engine flag).
`gpu-sim` `stream_begin_capture` is `cuStreamBeginCapture` (identity with `begin_capture`; no Engine flag).
`gpu-sim` `stream_begin_capture_with_mode` is `cuStreamBeginCapture` with mode (identity with `begin_capture_with_mode`; no Engine flag).
`gpu-sim` `stream_begin_capture_to_graph` is `cuStreamBeginCaptureToGraph` (identity with `begin_capture_to_graph`; no Engine flag).
`gpu-sim` `stream_begin_capture_to_graph_with_mode` is `cuStreamBeginCaptureToGraph` with mode (identity with `begin_capture_to_graph_with_mode`; no Engine flag).
`gpu-sim` `stream_begin_recapture_to_graph` is `cuStreamBeginRecaptureToGraph` (identity with `begin_recapture_to_graph`; no Engine flag).
`gpu-sim` `stream_begin_recapture_to_graph_with_mode` is `cuStreamBeginRecaptureToGraph` with mode (identity with `begin_recapture_to_graph_with_mode`; no Engine flag).
`gpu-sim` `stream_begin_recapture_to_graph_with_callback` is `cuStreamBeginRecaptureToGraph` with callback (identity with `begin_recapture_to_graph_with_callback`; no Engine flag).
`gpu-sim` `stream_end_capture` is `cuStreamEndCapture` (identity with `end_capture`; no Engine flag).
`gpu-sim` `update_stream_capture_dependencies` is `cuStreamUpdateCaptureDependencies` (identity with `stream_update_capture_dependencies`; no Engine flag).
`gpu-sim` `is_stream_capturing` is `cuStreamIsCapturing` (identity with `stream_is_capturing`; no Engine flag).
`gpu-sim` `get_stream_capture_info` is `cuStreamGetCaptureInfo` (identity with `stream_capture_info`; no Engine flag).
`gpu-sim` `exchange_thread_stream_capture_mode` is `cuThreadExchangeStreamCaptureMode` (identity with `thread_exchange_stream_capture_mode`; no Engine flag).
`gpu-sim` `get_stream_capture_mode` is the thread-default `cudaStreamCaptureMode` query (identity with `stream_capture_mode`; no Engine flag).
`gpu-sim` `event_flags` is `cuEventGetFlags` (identity with `event_get_flags`; no Engine flag).
`gpu-sim` `ctx_enable_peer_access` is `cuCtxEnablePeerAccess` (identity with `enable_peer`; no Engine flag).
`gpu-sim` `ctx_enable_peer_access_with_flags` is `cuCtxEnablePeerAccess` with flags (identity with `enable_peer_with_flags`; no Engine flag).
`gpu-sim` `ctx_disable_peer_access` is `cuCtxDisablePeerAccess` (identity with `disable_peer`; no Engine flag).
`gpu-sim` `can_device_access_peer` is `cuDeviceCanAccessPeer` (identity with `device_can_access_peer`; no Engine flag).
`gpu-sim` `device_p2p_attribute` is `cuDeviceGetP2PAttribute` (identity with `device_get_p2p_attribute`; no Engine flag).
`gpu-sim` `device_nvscisync_attributes` is `cuDeviceGetNvSciSyncAttributes` (identity with `device_get_nvscisync_attributes`; no Engine flag).
`gpu-sim` `device_flush_gpu_direct_rdma_writes` is `cuFlushGPUDirectRDMAWrites` (identity with `flush_gpu_direct_rdma_writes`; no Engine flag).
`gpu-sim` `mem_alloc_pitch` is `cudaMallocPitch` (identity with `malloc_pitch`; no Engine flag).
`gpu-sim` `mem_alloc_3d` is `cudaMalloc3D` (identity with `malloc_3d`; no Engine flag).
`gpu-sim` `launch_cooperative_kernel` is `cuLaunchCooperativeKernel` (identity with `cooperative_kernel`; no Engine flag).
`gpu-sim` `launch_cooperative_kernel_bufs` is `cuLaunchCooperativeKernel` spans (identity with `cooperative_kernel_bufs`; no Engine flag).
`gpu-sim` `launch_cooperative_kernel_multi_device` is `cuLaunchCooperativeKernelMultiDevice` (identity with `cooperative_kernel_multi_device`; no Engine flag).
`gpu-sim` `mem_set` is `cudaMemsetAsync` (identity with `memset`; no Engine flag).
`gpu-sim` `mem_set_buf` is `cudaMemsetAsync` spans (identity with `memset_buf`; no Engine flag).
`gpu-sim` `mem_set_op` is `cudaMemsetAsync` / `cudaMemset2DAsync` (identity with `memset_op`; no Engine flag).
`gpu-sim` `mem_set_sync` is `cudaMemset` (identity with `memset_sync`; no Engine flag).
`gpu-sim` `mem_set_op_sync` is `cudaMemset` / `cudaMemset2D` / `cudaMemset3D` (identity with `memset_op_sync`; no Engine flag).
`gpu-sim` `mem_set_2d_async` is `cudaMemset2DAsync` (identity with `memset_2d_async`; no Engine flag).
`gpu-sim` `mem_set_2d` is `cudaMemset2D` (identity with `memset_2d`; no Engine flag).
`gpu-sim` `mem_set_3d_async` is `cudaMemset3DAsync` (identity with `memset_3d_async`; no Engine flag).
`gpu-sim` `mem_set_3d` is `cudaMemset3D` (identity with `memset_3d`; no Engine flag).
`gpu-sim` `stream_write_value64` is `cuStreamWriteValue64` (identity with `write_value64`; no Engine flag).
`gpu-sim` `stream_write_value32` is `cuStreamWriteValue32` (identity with `write_value32`; no Engine flag).
`gpu-sim` `stream_write_value64_with_flags` is `cuStreamWriteValue64` flags (identity with `write_value64_with_flags`; no Engine flag).
`gpu-sim` `stream_write_value32_with_flags` is `cuStreamWriteValue32` flags (identity with `write_value32_with_flags`; no Engine flag).
`gpu-sim` `stream_wait_value64` is `cuStreamWaitValue64` (identity with `wait_value64`; no Engine flag).
`gpu-sim` `stream_wait_value32` is `cuStreamWaitValue32` (identity with `wait_value32`; no Engine flag).
`gpu-sim` `stream_wait_value64_with_flags` is `cuStreamWaitValue64` flags (identity with `wait_value64_with_flags`; no Engine flag).
`gpu-sim` `stream_wait_value32_with_flags` is `cuStreamWaitValue32` flags (identity with `wait_value32_with_flags`; no Engine flag).
`gpu-sim` `stream_batch_mem_op` is `cuStreamBatchMemOp` (identity with `batch_mem_op`; no Engine flag).
`gpu-sim` `stream_batch_mem_op_with_flags` is `cuStreamBatchMemOp` flags (identity with `batch_mem_op_with_flags`; no Engine flag).
`gpu-sim` `launch_kernel` is `cuLaunchKernel` (identity with `kernel`; no Engine flag).
`gpu-sim` `launch_kernel_bufs` is `cuLaunchKernel` spans (identity with `kernel_bufs`; no Engine flag).
`gpu-sim` `launch_kernel_ex` is `cuLaunchKernelEx` (identity with `kernel_with`; no Engine flag).
`gpu-sim` `launch_kernel_ex_bufs` is `cuLaunchKernelEx` spans (identity with `kernel_bufs_with`; no Engine flag).
`gpu-sim` `func_set_shared_mem_config` is `cuFuncSetSharedMemConfig` (identity with `set_func_shared_mem_config`; no Engine flag).

`gpu-sim` `func_get_shared_mem_config` is `cuFuncGetSharedMemConfig` (identity with `get_func_shared_mem_config`; no Engine flag).

`gpu-sim` `func_set_cache_config` is `cuFuncSetCacheConfig` (identity with `set_func_cache_config`; no Engine flag).

`gpu-sim` `func_set_carveout` is `cuFuncSetAttribute` carveout (identity with `set_func_carveout`; no Engine flag).

`gpu-sim` `func_get_carveout` is `cuFuncGetAttribute` carveout (identity with `get_func_carveout`; no Engine flag).

`gpu-sim` `func_set_cluster_policy` is `cuFuncSetAttribute` cluster policy (identity with `set_func_cluster_policy`; no Engine flag).

`gpu-sim` `func_get_cluster_policy` is `cuFuncGetAttribute` cluster policy (identity with `get_func_cluster_policy`; no Engine flag).

`gpu-sim` `func_set_cluster_dim_must_be_set` is `cuFuncSetAttribute` cluster dim must be set (identity with `set_cluster_dim_must_be_set`; no Engine flag).

`gpu-sim` `func_get_cluster_dim_must_be_set` is `cuFuncGetAttribute` cluster dim must be set (identity with `cluster_dim_must_be_set`; no Engine flag).

`gpu-sim` `func_set_required_cluster_width` is `cuFuncSetAttribute` required cluster width (identity with `set_required_cluster_width`; no Engine flag).

`gpu-sim` `func_get_required_cluster_width` is `cuFuncGetAttribute` required cluster width (identity with `required_cluster_width`; no Engine flag).

`gpu-sim` `func_set_required_cluster_height` is `cuFuncSetAttribute` required cluster height (identity with `set_required_cluster_height`; no Engine flag).

`gpu-sim` `func_get_required_cluster_height` is `cuFuncGetAttribute` required cluster height (identity with `required_cluster_height`; no Engine flag).

`gpu-sim` `func_set_required_cluster_depth` is `cuFuncSetAttribute` required cluster depth (identity with `set_required_cluster_depth`; no Engine flag).

`gpu-sim` `func_get_required_cluster_depth` is `cuFuncGetAttribute` required cluster depth (identity with `required_cluster_depth`; no Engine flag).

`gpu-sim` `func_set_non_portable_cluster_size_allowed` is `cuFuncSetAttribute` non-portable cluster size (identity with `set_non_portable_cluster_size_allowed`; no Engine flag).

`gpu-sim` `func_get_non_portable_cluster_size_allowed` is `cuFuncGetAttribute` non-portable cluster size (identity with `non_portable_cluster_size_allowed`; no Engine flag).

`gpu-sim` `func_set_max_dynamic_shared_memory` is `cuFuncSetAttribute` max dynamic shared memory (identity with `set_max_dynamic_shared_memory`; no Engine flag).

`gpu-sim` `func_get_max_dynamic_shared_memory` is `cuFuncGetAttribute` max dynamic shared memory (identity with `max_dynamic_shared_memory`; no Engine flag).

`gpu-sim` `event_create_disable_timing` is `cuEventCreateWithFlags` disable timing (identity with `create_event_disable_timing`; no Engine flag).

`gpu-sim` `event_create_interprocess` is `cuEventCreateWithFlags` interprocess (identity with `create_event_interprocess`; no Engine flag).

`gpu-sim` `event_create_blocking_sync` is `cuEventCreateWithFlags` blocking sync (identity with `create_event_blocking_sync`; no Engine flag).

`gpu-sim` `event_record_external` is `cuEventRecordWithFlags` external (identity with `record_event_external`; no Engine flag).

`gpu-sim` `stream_wait_event_external` is `cuStreamWaitEvent` external (identity with `wait_event_external`; no Engine flag).
`gpu-sim` `stream_set_mem_sync_domain` is `cuStreamSetAttribute` mem sync domain (identity with `set_stream_mem_sync_domain`; no Engine flag).
`gpu-sim` `stream_set_mem_sync_domain_map` is `cuStreamSetAttribute` mem sync domain map (identity with `set_stream_mem_sync_domain_map`; no Engine flag).
`gpu-sim` `stream_get_mem_sync_domain` is `cuStreamGetAttribute` mem sync domain (identity with `stream_mem_sync_domain`; no Engine flag).
`gpu-sim` `stream_get_mem_sync_domain_map` is `cuStreamGetAttribute` mem sync domain map (identity with `stream_mem_sync_domain_map`; no Engine flag).
`gpu-sim` `stream_set_sync_policy` is `cuStreamSetAttribute` sync policy (identity with `set_stream_sync_policy`; no Engine flag).
`gpu-sim` `stream_get_sync_policy` is `cuStreamGetAttribute` sync policy (identity with `stream_sync_policy`; no Engine flag).
`gpu-sim` `stream_set_nvlink_util_centric` is `cuStreamSetAttribute` nvlink util centric (identity with `set_stream_nvlink_util_centric`; no Engine flag).
`gpu-sim` `stream_get_nvlink_util_centric` is `cuStreamGetAttribute` nvlink util centric (identity with `stream_nvlink_util_centric`; no Engine flag).
`gpu-sim` `stream_set_access_policy` is `cuStreamSetAttribute` access policy (identity with `set_stream_access_policy`; no Engine flag).
`gpu-sim` `stream_get_access_policy` is `cuStreamGetAttribute` access policy (identity with `stream_access_policy`; no Engine flag).
`gpu-sim` `stream_set_priority` is `cuStreamSetAttribute` priority (identity with `set_stream_priority`; no Engine flag).
`gpu-sim` `stream_set_blocking` is `cuStreamCreate` blocking (identity with `set_stream_blocking`; no Engine flag).
`gpu-sim` `get_func_attributes` is `cuFuncGetAttributes` (identity with `func_get_attributes`; no Engine flag).
`gpu-sim` `get_device_name` is `cuDeviceGetName` (identity with `device_get_name`; no Engine flag).
`gpu-sim` `get_device_count` is `cuDeviceGetCount` (identity with `device_count`; no Engine flag).
`gpu-sim` `device_get_default_mempool` is `cuDeviceGetDefaultMemPool` (identity with `default_pool`; no Engine flag).
`gpu-sim` `device_get_mempool` is `cuDeviceGetMemPool` (identity with `device_mempool`; no Engine flag).
`gpu-sim` `device_set_mempool` is `cuDeviceSetMemPool` (identity with `set_device_mempool`; no Engine flag).
`gpu-sim` `mem_pool_create` is `cuMemPoolCreate` (identity with `create_pool`; no Engine flag).
`gpu-sim` `mem_pool_create_shareable` is `cuMemPoolCreate` POSIX (identity with `create_shareable_pool`; no Engine flag).
`gpu-sim` `mem_pool_create_with_props` is `cuMemPoolCreate` with props (identity with `create_pool_with_props`; no Engine flag).
`gpu-sim` `mem_pool_destroy` is `cuMemPoolDestroy` (identity with `destroy_pool`; no Engine flag).
`gpu-sim` `mem_alloc_from_pool` is `cuMemAllocFromPoolAsync` (identity with `alloc_from_pool`; no Engine flag).
`gpu-sim` `mem_pool_export` is `cuMemPoolExportToShareableHandle` (identity with `pool_export`; no Engine flag).
`gpu-sim` `mem_pool_import` is `cuMemPoolImportFromShareableHandle` (identity with `pool_import`; no Engine flag).
`gpu-sim` `mem_pool_export_with_type` is `cuMemPoolExportToShareableHandle` type (identity with `pool_export_with_type`; no Engine flag).
`gpu-sim` `mem_pool_import_with_type` is `cuMemPoolImportFromShareableHandle` type (identity with `pool_import_with_type`; no Engine flag).
`gpu-sim` `mem_pool_export_ptr` is `cuMemPoolExportPointer` (identity with `pool_export_ptr`; no Engine flag).
`gpu-sim` `mem_pool_import_ptr` is `cuMemPoolImportPointer` (identity with `pool_import_ptr`; no Engine flag).
`gpu-sim` `mem_pool_get_access` is `cuMemPoolGetAccess` (identity with `pool_get_access`; no Engine flag).
`gpu-sim` `mem_pool_set_access` is `cuMemPoolSetAccess` (identity with `pool_set_access`; no Engine flag).
`gpu-sim` `mem_pool_set_access_read` is `cuMemPoolSetAccess` ProtRead (identity with `pool_set_access_read`; no Engine flag).
`gpu-sim` `mem_pool_set_access_with_flags` is `cuMemPoolSetAccess` flags (identity with `pool_set_access_with_flags`; no Engine flag).
`gpu-sim` `mem_pool_set_access_n` is `cuMemPoolSetAccess` n (identity with `pool_set_access_n`; no Engine flag).
`gpu-sim` `mem_pool_unset_access` is `cuMemPoolSetAccess` ProtNone (identity with `pool_unset_access`; no Engine flag).
`gpu-sim` `mem_pool_get_attribute` is `cuMemPoolGetAttribute` (identity with `pool_get_attribute`; no Engine flag).
`gpu-sim` `mem_pool_set_attribute` is `cuMemPoolSetAttribute` (identity with `pool_set_attribute`; no Engine flag).
`gpu-sim` `mem_pool_trim_to` is `cuMemPoolTrimTo` (identity with `pool_trim_to`; no Engine flag).
`gpu-sim` `mem_pool_set_release_threshold` is `cuMemPoolSetAttribute` ReleaseThreshold (identity with `set_pool_release_threshold`; no Engine flag).
`gpu-sim` `mem_pool_set_max_size` is `cuMemPoolSetAttribute` MaxPoolSize (identity with `set_pool_max_size`; no Engine flag).
`gpu-sim` `mem_get_allocation_granularity` is `cuMemGetAllocationGranularity` (identity with `va_get_allocation_granularity`; no Engine flag).
`gpu-sim` `mem_create` is `cuMemCreate` (identity with `va_create`; no Engine flag).
`gpu-sim` `mem_create_with_prop` is `cuMemCreate` props (identity with `va_create_with_prop`; no Engine flag).
`gpu-sim` `mem_map_handle` is `cuMemMap` (identity with `va_map_handle`; no Engine flag).
`gpu-sim` `mem_map_handle_with_flags` is `cuMemMap` flags (identity with `va_map_handle_with_flags`; no Engine flag).
`gpu-sim` `mem_map_handle_with_size` is `cuMemMap` size (identity with `va_map_handle_with_size`; no Engine flag).
`gpu-sim` `func_is_loaded` is `cuFuncIsLoaded` (`false` until a compiled
kernel exists; no Engine flag).
`gpu-sim` `func_load` is `cuFuncLoad` (Invalid; no compiled kernel; no
Engine flag).
`gpu-sim` `func_get_module` is `cuFuncGetModule` (Invalid until a compiled
kernel exists; no Engine flag).
`gpu-sim` `driver_init` is `cuInit` (flags 0; already initialized; no
Engine flag).
`gpu-sim` `profiler_start` is `cuProfilerStart` (1 ns no-op; capture
refused; no Engine flag).
`gpu-sim` `profiler_stop` is `cuProfilerStop` (1 ns no-op; capture
refused; no Engine flag).
`gpu-sim` `profiler_initialize` is `cudaProfilerInitialize` (Invalid;
CUPTI config is not modeled; no Engine flag).
`gpu-sim` `module_get_loading_mode` is `cuModuleGetLoadingMode` (always
Eager; no Engine flag).
`gpu-sim` `module_load` is `cuModuleLoad` (Invalid; no cubin path; no
Engine flag).
`gpu-sim` `module_load_data` is `cuModuleLoadData` (Invalid; no cubin
image; no Engine flag).
`gpu-sim` `module_load_fat_binary` is `cuModuleLoadFatBinary` (Invalid;
no fatbin image; no Engine flag).
`gpu-sim` `module_load_data_ex` is `cuModuleLoadDataEx` (Invalid; no
JIT options; no Engine flag).
`gpu-sim` `module_get_function_count` is `cuModuleGetFunctionCount`
(Invalid; no `CUmodule` function list; no Engine flag).
`gpu-sim` `module_enumerate_functions` is `cuModuleEnumerateFunctions`
(Invalid; no `CUmodule` function list; no Engine flag).
`gpu-sim` `module_unload` is `cuModuleUnload` (Invalid; no `CUmodule`
handle; no Engine flag).
`gpu-sim` `module_get_function` is `cuModuleGetFunction` (Invalid; no
`CUmodule` function; no Engine flag).
`gpu-sim` `module_get_global` is `cuModuleGetGlobal` (Invalid; no
`CUmodule` device symbol; no Engine flag).
`gpu-sim` `module_get_tex_ref` is `cuModuleGetTexRef` (Invalid; no
`CUmodule` texref; no Engine flag).
`gpu-sim` `module_get_surf_ref` is `cuModuleGetSurfRef` (Invalid; no
`CUmodule` surfref; no Engine flag).
`gpu-sim` `library_load_data` is `cuLibraryLoadData` (Invalid; no cubin /
`CUlibrary`; no Engine flag).
`gpu-sim` `library_load_from_file` is `cuLibraryLoadFromFile` (Invalid;
no cubin path / `CUlibrary`; no Engine flag).
`gpu-sim` `library_unload` is `cuLibraryUnload` (Invalid; no `CUlibrary`
handle; no Engine flag).
`gpu-sim` `library_get_kernel` is `cuLibraryGetKernel` (Invalid; no
`CUlibrary` / `CUkernel`; no Engine flag).
`gpu-sim` `library_get_module` is `cuLibraryGetModule` (Invalid; no
`CUlibrary` / `CUmodule`; no Engine flag).
`gpu-sim` `library_get_global` is `cuLibraryGetGlobal` (Invalid; no
`CUlibrary` device symbol; no Engine flag).
`gpu-sim` `library_get_managed` is `cuLibraryGetManaged` (Invalid; no
`CUlibrary` managed symbol; no Engine flag).
`gpu-sim` `library_get_unified_function` is `cuLibraryGetUnifiedFunction`
(Invalid; no `CUlibrary` device function pointer; no Engine flag).
`gpu-sim` `library_get_kernel_count` is `cuLibraryGetKernelCount` (Invalid;
no `CUlibrary` kernel list; no Engine flag).
`gpu-sim` `library_enumerate_kernels` is `cuLibraryEnumerateKernels` (Invalid;
no `CUlibrary` kernel list; no Engine flag).
`gpu-sim` `kernel_get_library` is `cuKernelGetLibrary` (Invalid; no
`CUkernel` / `CUlibrary`; no Engine flag).
`gpu-sim` `kernel_get_function` is `cuKernelGetFunction` (Invalid; no
`CUkernel` / `CUfunction`; no Engine flag).
`gpu-sim` `kernel_get_param_info` is `cuKernelGetParamInfo` (Invalid; no
`CUkernel` parameter blob; no Engine flag).
`gpu-sim` `kernel_get_param_count` is `cuKernelGetParamCount` (Invalid; no
`CUkernel` parameter list; no Engine flag).
`gpu-sim` `kernel_get_attribute` is `cuKernelGetAttribute` (Invalid; no
`CUkernel` attribute; no Engine flag).
`gpu-sim` `kernel_set_attribute` is `cuKernelSetAttribute` (Invalid; no
`CUkernel` attribute; no Engine flag).
`gpu-sim` `kernel_set_cache_config` is `cuKernelSetCacheConfig` (Invalid;
no `CUkernel` cache config; no Engine flag).
`gpu-sim` `link_create` is `cuLinkCreate` (Invalid; no JIT linker; no
Engine flag).
`gpu-sim` `link_add_data` is `cuLinkAddData` (Invalid; no JIT linker; no
Engine flag).
`gpu-sim` `link_complete` is `cuLinkComplete` (Invalid; no JIT linker; no
Engine flag).
`gpu-sim` `link_destroy` is `cuLinkDestroy` (Invalid; no JIT linker; no
Engine flag).
`gpu-sim` `link_add_file` is `cuLinkAddFile` (Invalid; no JIT linker; no
Engine flag).
`gpu-sim` `get_proc_address` is `cuGetProcAddress` (Invalid; no C ABI
function pointers; no Engine flag).
`gpu-sim` `get_export_table` is `cuGetExportTable` (Invalid; no internal
driver tables; no Engine flag).
`gpu-sim` `coredump_get_attribute` is `cuCoredumpGetAttribute` (Invalid;
GPU coredumps are not modeled; no Engine flag).
`gpu-sim` `coredump_set_attribute` is `cuCoredumpSetAttribute` (Invalid;
GPU coredumps are not modeled; no Engine flag).
`gpu-sim` `coredump_get_attribute_global` is `cuCoredumpGetAttributeGlobal`
(Invalid; GPU coredumps are not modeled; no Engine flag).
`gpu-sim` `coredump_set_attribute_global` is `cuCoredumpSetAttributeGlobal`
(Invalid; GPU coredumps are not modeled; no Engine flag).
`gpu-sim` `checkpoint_process_lock` is `cuCheckpointProcessLock` (Invalid;
CUDA process checkpoint is not modeled; no Engine flag).
`gpu-sim` `checkpoint_process_checkpoint` is `cuCheckpointProcessCheckpoint`
(Invalid; CUDA process checkpoint is not modeled; no Engine flag).
`gpu-sim` `checkpoint_process_restore` is `cuCheckpointProcessRestore`
(Invalid; CUDA process checkpoint is not modeled; no Engine flag).
`gpu-sim` `checkpoint_process_unlock` is `cuCheckpointProcessUnlock`
(Invalid; CUDA process checkpoint is not modeled; no Engine flag).
`gpu-sim` `checkpoint_process_get_restore_thread_id` is
`cuCheckpointProcessGetRestoreThreadId` (Invalid; CUDA process checkpoint
is not modeled; no Engine flag).
`gpu-sim` `checkpoint_process_get_state` is `cuCheckpointProcessGetState`
(Invalid; CUDA process checkpoint is not modeled; no Engine flag).
`gpu-sim` `device_register_async_notification` is
`cuDeviceRegisterAsyncNotification` (Invalid; device async callbacks are
not modeled; no Engine flag).
`gpu-sim` `device_unregister_async_notification` is
`cuDeviceUnregisterAsyncNotification` (Invalid; device async callbacks are
not modeled; no Engine flag).
`gpu-sim` `ctx_get_device` is `cuCtxGetDevice` (explicit device of the
seeded primary context; no Engine flag).
`gpu-sim` `ctx_reset_persisting_l2_cache` is `cuCtxResetPersistingL2Cache`
(wraps `reset_persisting_l2_cache`; no Engine flag).
`gpu-sim` `ctx_get_exec_affinity` is `cuCtxGetExecAffinity` (SM_COUNT
unsupported; no Engine flag).
`gpu-sim` `mem_batch_decompress_async` is `cuMemBatchDecompressAsync`
(Invalid; hardware decompress is not modeled; no Engine flag).
`gpu-sim` `tensor_map_encode_tiled` is `cuTensorMapEncodeTiled` (Invalid;
TMA is not modeled; no Engine flag).
`gpu-sim` `tensor_map_encode_im2col` is `cuTensorMapEncodeIm2col` (Invalid;
TMA is not modeled; no Engine flag).
`gpu-sim` `tensor_map_encode_im2col_wide` is `cuTensorMapEncodeIm2colWide`
(Invalid; TMA is not modeled; no Engine flag).
`gpu-sim` `tensor_map_replace_aligned_addr` is `cuTensorMapReplaceAlignedAddr`
(Invalid; TMA is not modeled; no Engine flag).
`gpu-sim` `cooperative_kernel_multi_device` is
`cudaLaunchCooperativeKernelMultiDevice` (Invalid; no Engine flag).
`gpu-sim` `array_create` is `cuArrayCreate` (Invalid; CUDA arrays are not
modeled; no Engine flag).
`gpu-sim` `array_destroy` is `cuArrayDestroy` (Invalid; no array handles;
no Engine flag).
`gpu-sim` `array_get_descriptor` is `cuArrayGetDescriptor` (Invalid; no
array handles; no Engine flag).
`gpu-sim` `array_3d_get_descriptor` is `cuArray3DGetDescriptor` (Invalid;
no array handles; no Engine flag).
`gpu-sim` `array_get_sparse_properties` is `cuArrayGetSparseProperties`
(Invalid; sparse CUDA arrays are not modeled; no Engine flag).
`gpu-sim` `mem_map_array_async` is `cuMemMapArrayAsync` (Invalid; sparse
CUDA array mapping is not modeled; no Engine flag).
`gpu-sim` `array_get_plane` is `cuArrayGetPlane` (Invalid; no array
handles; no Engine flag).
`gpu-sim` `array_get_memory_requirements` is
`cuArrayGetMemoryRequirements` (Invalid; no array handles; no Engine flag).
`gpu-sim` `mipmapped_array_get_memory_requirements` is
`cuMipmappedArrayGetMemoryRequirements` (Invalid; no mipmapped-array
handles; no Engine flag).
`gpu-sim` `mipmapped_array_get_sparse_properties` is
`cuMipmappedArrayGetSparseProperties` (Invalid; sparse CUDA mipmapped
arrays are not modeled; no Engine flag).
`gpu-sim` `tex_ref_create` is `cuTexRefCreate` (Invalid; no `CUtexref`
handles; no Engine flag).
`gpu-sim` `tex_ref_destroy` is `cuTexRefDestroy` (Invalid; no `CUtexref`
handles; no Engine flag).
`gpu-sim` `tex_ref_set_array` is `cuTexRefSetArray` (Invalid; no `CUtexref`
or `CUarray` handles; no Engine flag).
`gpu-sim` `tex_ref_set_mipmapped_array` is `cuTexRefSetMipmappedArray`
(Invalid; no `CUtexref` or mipmapped-array handles; no Engine flag).
`gpu-sim` `tex_ref_set_address` is `cuTexRefSetAddress` (Invalid; no
`CUtexref` linear bindings; no Engine flag).
`gpu-sim` `tex_ref_set_address_2d` is `cuTexRefSetAddress2D` (Invalid; no
`CUtexref` pitched 2D bindings; no Engine flag).
`gpu-sim` `tex_ref_set_format` is `cuTexRefSetFormat` (Invalid; no
`CUtexref` channel format; no Engine flag).
`gpu-sim` `tex_ref_set_address_mode` is `cuTexRefSetAddressMode` (Invalid;
no `CUtexref` addressing; no Engine flag).
`gpu-sim` `tex_ref_set_filter_mode` is `cuTexRefSetFilterMode` (Invalid;
no `CUtexref` filtering; no Engine flag).
`gpu-sim` `tex_ref_set_mipmap_filter_mode` is `cuTexRefSetMipmapFilterMode`
(Invalid; no `CUtexref` mipmap filtering; no Engine flag).
`gpu-sim` `tex_ref_set_mipmap_level_bias` is `cuTexRefSetMipmapLevelBias`
(Invalid; no `CUtexref` mipmap LOD bias; no Engine flag).
`gpu-sim` `tex_ref_set_mipmap_level_clamp` is `cuTexRefSetMipmapLevelClamp`
(Invalid; no `CUtexref` mipmap LOD clamp; no Engine flag).
`gpu-sim` `tex_ref_set_max_anisotropy` is `cuTexRefSetMaxAnisotropy`
(Invalid; no `CUtexref` anisotropy; no Engine flag).
`gpu-sim` `tex_ref_set_border_color` is `cuTexRefSetBorderColor` (Invalid;
no `CUtexref` border color; no Engine flag).
`gpu-sim` `tex_ref_set_flags` is `cuTexRefSetFlags` (Invalid; no
`CUtexref` flags word; no Engine flag).
`gpu-sim` `tex_ref_get_array` is `cuTexRefGetArray` (Invalid; no
`CUtexref` or `CUarray` handles; no Engine flag).
`gpu-sim` `tex_ref_get_mipmapped_array` is `cuTexRefGetMipmappedArray`
(Invalid; no `CUtexref` or mipmapped-array handles; no Engine flag).
`gpu-sim` `tex_ref_get_address` is `cuTexRefGetAddress` (Invalid; no
`CUtexref` linear bindings; no Engine flag).
`gpu-sim` `tex_ref_get_address_mode` is `cuTexRefGetAddressMode` (Invalid;
no `CUtexref` addressing; no Engine flag).
`gpu-sim` `tex_ref_get_filter_mode` is `cuTexRefGetFilterMode` (Invalid;
no `CUtexref` filtering; no Engine flag).
`gpu-sim` `tex_ref_get_format` is `cuTexRefGetFormat` (Invalid; no
`CUtexref` channel format; no Engine flag).
`gpu-sim` `tex_ref_get_mipmap_filter_mode` is `cuTexRefGetMipmapFilterMode`
(Invalid; no `CUtexref` mipmap filtering; no Engine flag).
`gpu-sim` `tex_ref_get_mipmap_level_bias` is `cuTexRefGetMipmapLevelBias`
(Invalid; no `CUtexref` mipmap LOD bias; no Engine flag).
`gpu-sim` `tex_ref_get_mipmap_level_clamp` is `cuTexRefGetMipmapLevelClamp`
(Invalid; no `CUtexref` mipmap LOD clamp; no Engine flag).
`gpu-sim` `tex_ref_get_max_anisotropy` is `cuTexRefGetMaxAnisotropy`
(Invalid; no `CUtexref` anisotropy; no Engine flag).
`gpu-sim` `tex_ref_get_border_color` is `cuTexRefGetBorderColor`
(Invalid; no `CUtexref` border color; no Engine flag).
`gpu-sim` `tex_ref_get_flags` is `cuTexRefGetFlags`
(Invalid; no `CUtexref` flags word; no Engine flag).
`gpu-sim` `surf_ref_set_array` is `cuSurfRefSetArray`
(Invalid; no `CUsurfref` array binding; no Engine flag).
`gpu-sim` `surf_ref_get_array` is `cuSurfRefGetArray`
(Invalid; no `CUsurfref` array binding; no Engine flag).
`gpu-sim` `memcpy_dto_a` is `cuMemcpyDtoA`
(Invalid; no `CUarray` device-to-array copy; no Engine flag).
`gpu-sim` `memcpy_ato_d` is `cuMemcpyAtoD`
(Invalid; no `CUarray` array-to-device copy; no Engine flag).
`gpu-sim` `memcpy_hto_a` is `cuMemcpyHtoA`
(Invalid; no `CUarray` host-to-array copy; no Engine flag).
`gpu-sim` `memcpy_ato_h` is `cuMemcpyAtoH`
(Invalid; no `CUarray` array-to-host copy; no Engine flag).
`gpu-sim` `memcpy_ato_a` is `cuMemcpyAtoA`
(Invalid; no `CUarray` array-to-array copy; no Engine flag).
`gpu-sim` `memcpy_dto_a_async` is `cuMemcpyDtoAAsync`
(Invalid; no `CUarray` device-to-array copy; no Engine flag).
`gpu-sim` `memcpy_ato_d_async` is `cuMemcpyAtoDAsync`
(Invalid; no `CUarray` array-to-device copy; no Engine flag).
`gpu-sim` `memcpy_hto_a_async` is `cuMemcpyHtoAAsync`
(Invalid; no `CUarray` host-to-array copy; no Engine flag).
`gpu-sim` `memcpy_ato_h_async` is `cuMemcpyAtoHAsync`
(Invalid; no `CUarray` array-to-host copy; no Engine flag).
`gpu-sim` `memcpy_ato_a_async` is `cuMemcpyAtoAAsync`
(Invalid; no `CUarray` array-to-array copy; no Engine flag).
`gpu-sim` `memcpy_2d_to_array` is `cuMemcpy2DToArray`
(Invalid; no `CUarray` 2D copy; no Engine flag).
`gpu-sim` `memcpy_2d_from_array` is `cuMemcpy2DFromArray`
(Invalid; no `CUarray` 2D copy; no Engine flag).
`gpu-sim` `memcpy_2d_array_to_array` is `cuMemcpy2DArrayToArray`
(Invalid; no `CUarray` 2D copy; no Engine flag).
`gpu-sim` `memcpy_2d_to_array_async` is `cuMemcpy2DToArrayAsync`
(Invalid; no `CUarray` 2D copy; no Engine flag).
`gpu-sim` `memcpy_2d_from_array_async` is `cuMemcpy2DFromArrayAsync`
(Invalid; no `CUarray` 2D copy; no Engine flag).
`gpu-sim` `memcpy_2d_array_to_array_async` is `cuMemcpy2DArrayToArrayAsync`
(Invalid; no `CUarray` 2D copy; no Engine flag).
`gpu-sim` `mipmapped_array_create` is `cuMipmappedArrayCreate` (Invalid;
CUDA mipmapped arrays are not modeled; no Engine flag).
`gpu-sim` `mipmapped_array_get_level` is `cuMipmappedArrayGetLevel`
(Invalid; no mipmapped-array handles; no Engine flag).
`gpu-sim` `mipmapped_array_destroy` is `cuMipmappedArrayDestroy`
(Invalid; no mipmapped-array handles; no Engine flag).
`gpu-sim` `import_external_memory` is `cuImportExternalMemory` (Invalid;
no Engine flag).
`gpu-sim` `destroy_external_memory` is `cuDestroyExternalMemory` (Invalid;
no external-memory handles; no Engine flag).
`gpu-sim` `external_memory_get_mapped_buffer` is
`cuExternalMemoryGetMappedBuffer` (Invalid; no external-memory handles;
no Engine flag).
`gpu-sim` `external_memory_get_mapped_mipmapped_array` is
`cuExternalMemoryGetMappedMipmappedArray` (Invalid; no external-memory
handles; no Engine flag).
`gpu-sim` `import_external_semaphore` is `cuImportExternalSemaphore`
(Invalid; no external-semaphore handles; no Engine flag).
`gpu-sim` `destroy_external_semaphore` is `cuDestroyExternalSemaphore`
(Invalid; no external-semaphore handles; no Engine flag).
`gpu-sim` `signal_external_semaphores_async` is
`cuSignalExternalSemaphoresAsync` (Invalid; no external-semaphore handles;
no Engine flag).
`gpu-sim` `wait_external_semaphores_async` is
`cuWaitExternalSemaphoresAsync` (Invalid; no external-semaphore handles;
no Engine flag).
`gpu-sim` `surf_object_create` is `cuSurfObjectCreate` (Invalid; CUDA
surfaces are not modeled; no Engine flag).
`gpu-sim` `surf_object_destroy` is `cuSurfObjectDestroy` (Invalid; no
surface-object handles; no Engine flag).
`gpu-sim` `surf_object_get_resource_desc` is `cuSurfObjectGetResourceDesc`
(Invalid; no surface-object handles; no Engine flag).
`gpu-sim` `tex_object_create` is `cuTexObjectCreate` (Invalid; CUDA
textures are not modeled; no Engine flag).
`gpu-sim` `tex_object_destroy` is `cuTexObjectDestroy` (Invalid; no
texture-object handles; no Engine flag).
`gpu-sim` `tex_object_get_resource_desc` is `cuTexObjectGetResourceDesc`
(Invalid; no texture-object handles; no Engine flag).
`gpu-sim` `tex_object_get_texture_desc` is `cuTexObjectGetTextureDesc`
(Invalid; no texture-object handles; no Engine flag).
`gpu-sim` `tex_object_get_resource_view_desc` is
`cuTexObjectGetResourceViewDesc` (Invalid; no texture-object handles; no
Engine flag).
`gpu-sim` texture 2D/3D dim caps are always 0 (`cudaDevAttrMaxTexture2DWidth`
and Height, `MaxTexture3DWidth` / Height / Depth; no Engine flag).
`gpu-sim` alternate texture 3D dim caps are always 0
(`cudaDevAttrMaxTexture3DWidthAlt`, HeightAlt, and DepthAlt; no Engine flag).
`gpu-sim` `MpsEnabled` is always 0 (`cudaDevAttrMpsEnabled`; CUDA
Multi-Process Service is not modeled; no Engine flag).
`gpu-sim` `D3D12CigSupported` is always 0 (`cudaDevAttrD3D12CigSupported`;
D3D12 CUDA-in-graphics is not modeled; no Engine flag).
`gpu-sim` `graphics_map_resources` is `cuGraphicsMapResources` (Invalid;
graphics resources are not modeled; no Engine flag).
`gpu-sim` `graphics_unmap_resources` is `cuGraphicsUnmapResources`
(Invalid; no graphics-resource handles; no Engine flag).
`gpu-sim` `graphics_resource_get_mapped_pointer` is
`cuGraphicsResourceGetMappedPointer` (Invalid; no graphics-resource
handles; no Engine flag).
`gpu-sim` `graphics_subresource_get_mapped_array` is
`cuGraphicsSubResourceGetMappedArray` (Invalid; no graphics-resource
handles; no Engine flag).
`gpu-sim` `graphics_resource_get_mapped_mipmapped_array` is
`cuGraphicsResourceGetMappedMipmappedArray` (Invalid; no graphics-resource
handles; no Engine flag).
`gpu-sim` `graphics_unregister_resource` is `cuGraphicsUnregisterResource`
(Invalid; no graphics-resource handles; no Engine flag).
`gpu-sim` `graphics_resource_set_map_flags` is `cuGraphicsResourceSetMapFlags`
(Invalid; no graphics-resource handles; no Engine flag).
`gpu-sim` `graphics_gl_register_buffer` is `cuGraphicsGLRegisterBuffer`
(Invalid; OpenGL interop is not modeled; no Engine flag).
`gpu-sim` `graphics_gl_register_image` is `cuGraphicsGLRegisterImage`
(Invalid; OpenGL interop is not modeled; no Engine flag).
`gpu-sim` `graphics_egl_register_image` is `cuGraphicsEGLRegisterImage`
(Invalid; EGL interop is not modeled; no Engine flag).
`gpu-sim` `egl_stream_consumer_connect` is `cuEGLStreamConsumerConnect`
(Invalid; EGL streams are not modeled; no Engine flag).
`gpu-sim` `egl_stream_consumer_disconnect` is
`cuEGLStreamConsumerDisconnect` (Invalid; EGL streams are not modeled;
no Engine flag).
`gpu-sim` `egl_stream_consumer_acquire_frame` is
`cuEGLStreamConsumerAcquireFrame` (Invalid; EGL streams are not modeled;
no Engine flag).
`gpu-sim` `egl_stream_consumer_release_frame` is
`cuEGLStreamConsumerReleaseFrame` (Invalid; EGL streams are not modeled;
no Engine flag).
`gpu-sim` `egl_stream_producer_connect` is `cuEGLStreamProducerConnect`
(Invalid; EGL streams are not modeled; no Engine flag).
`gpu-sim` `egl_stream_producer_disconnect` is
`cuEGLStreamProducerDisconnect` (Invalid; EGL streams are not modeled;
no Engine flag).
`gpu-sim` `egl_stream_producer_present_frame` is
`cuEGLStreamProducerPresentFrame` (Invalid; EGL streams are not modeled;
no Engine flag).
`gpu-sim` `egl_stream_producer_return_frame` is
`cuEGLStreamProducerReturnFrame` (Invalid; EGL streams are not modeled;
no Engine flag).
`gpu-sim` `gl_get_devices` is `cuGLGetDevices` (Invalid; OpenGL interop
is not modeled; no Engine flag).
`gpu-sim` `gl_ctx_create` is `cuGLCtxCreate` (Invalid; OpenGL interop is
not modeled; no Engine flag).
`gpu-sim` `gl_register_buffer_object` is `cuGLRegisterBufferObject`
(Invalid; legacy OpenGL interop is not modeled; no Engine flag).
`gpu-sim` `gl_map_buffer_object` is `cuGLMapBufferObject` (Invalid;
legacy OpenGL interop is not modeled; no Engine flag).
`gpu-sim` `gl_unregister_buffer_object` is `cuGLUnregisterBufferObject`
(Invalid; legacy OpenGL interop is not modeled; no Engine flag).
`gpu-sim` `gl_unmap_buffer_object` is `cuGLUnmapBufferObject`
(Invalid; legacy OpenGL interop is not modeled; no Engine flag).
`gpu-sim` `gl_unmap_buffer_object_async` is `cuGLUnmapBufferObjectAsync`
(Invalid; legacy OpenGL interop is not modeled; no Engine flag).
`gpu-sim` `gl_map_buffer_object_async` is `cuGLMapBufferObjectAsync`
(Invalid; legacy OpenGL interop is not modeled; no Engine flag).
`gpu-sim` `gl_set_gl_device` is `cudaGLSetGLDevice`
(Invalid; OpenGL interop is not modeled; no Engine flag).
`gpu-sim` `d3d11_get_devices` is `cuD3D11GetDevices` (Invalid; Direct3D 11
interop is not modeled; no Engine flag).
`gpu-sim` `d3d11_get_device` is `cuD3D11GetDevice` (Invalid; Direct3D 11
interop is not modeled; no Engine flag).
`gpu-sim` `d3d11_ctx_create` is `cuD3D11CtxCreate` (Invalid; Direct3D 11
interop is not modeled; no Engine flag).
`gpu-sim` `d3d11_ctx_create_on_device` is `cuD3D11CtxCreateOnDevice`
(Invalid; Direct3D 11 interop is not modeled; no Engine flag).
`gpu-sim` `graphics_d3d11_register_resource` is
`cuGraphicsD3D11RegisterResource` (Invalid; Direct3D 11 interop is not
modeled; no Engine flag).
`gpu-sim` `d3d12_get_devices` is `cuD3D12GetDevices` (Invalid; Direct3D 12
interop is not modeled; distinct from `D3D12CigSupported`; no Engine flag).
`gpu-sim` `d3d12_get_device` is `cuD3D12GetDevice` (Invalid; Direct3D 12
interop is not modeled; distinct from `D3D12CigSupported`; no Engine flag).
`gpu-sim` `d3d12_ctx_create` is `cuD3D12CtxCreate` (Invalid; Direct3D 12
interop is not modeled; distinct from `D3D12CigSupported`; no Engine flag).
`gpu-sim` `d3d12_ctx_create_on_device` is `cuD3D12CtxCreateOnDevice`
(Invalid; Direct3D 12 interop is not modeled; distinct from `D3D12CigSupported`;
no Engine flag).
`gpu-sim` `graphics_d3d12_register_resource` is
`cuGraphicsD3D12RegisterResource` (Invalid; Direct3D 12 interop is not
modeled; no Engine flag).
`gpu-sim` `vdpau_get_device` is `cuVDPAUGetDevice` (Invalid; VDPAU interop
is not modeled; no Engine flag).
`gpu-sim` `vdpau_set_vdpau_device` is `cudaVDPAUSetVDPAUDevice` (Invalid;
VDPAU interop is not modeled; no Engine flag).
`gpu-sim` `vdpau_ctx_create` is `cuVDPAUCtxCreate` (Invalid; VDPAU interop
is not modeled; no Engine flag).
`gpu-sim` `graphics_vdpau_register_output_surface` is
`cuGraphicsVDPAURegisterOutputSurface` (Invalid; VDPAU interop is not
modeled; no Engine flag).
`gpu-sim` `graphics_vdpau_register_video_surface` is
`cuGraphicsVDPAURegisterVideoSurface` (Invalid; VDPAU interop is not
modeled; no Engine flag).
`gpu-sim` `d3d9_get_devices` is `cuD3D9GetDevices` (Invalid; Direct3D 9
interop is not modeled; no Engine flag).
`gpu-sim` `d3d9_get_device` is `cuD3D9GetDevice` (Invalid; Direct3D 9
interop is not modeled; no Engine flag).
`gpu-sim` `d3d9_ctx_create` is `cuD3D9CtxCreate` (Invalid; Direct3D 9
interop is not modeled; no Engine flag).
`gpu-sim` `d3d9_ctx_create_on_device` is `cuD3D9CtxCreateOnDevice`
(Invalid; Direct3D 9 interop is not modeled; no Engine flag).
`gpu-sim` `graphics_d3d9_register_resource` is
`cuGraphicsD3D9RegisterResource` (Invalid; Direct3D 9 interop is not
modeled; no Engine flag).
`gpu-sim` `d3d10_get_devices` is `cuD3D10GetDevices` (Invalid; Direct3D 10
interop is not modeled; no Engine flag).
`gpu-sim` `d3d10_get_device` is `cuD3D10GetDevice` (Invalid; Direct3D 10
interop is not modeled; no Engine flag).
`gpu-sim` `d3d10_ctx_create` is `cuD3D10CtxCreate` (Invalid; Direct3D 10
interop is not modeled; no Engine flag).
`gpu-sim` `d3d10_ctx_create_on_device` is `cuD3D10CtxCreateOnDevice`
(Invalid; Direct3D 10 interop is not modeled; no Engine flag).
`gpu-sim` `graphics_d3d10_register_resource` is
`cuGraphicsD3D10RegisterResource` (Invalid; Direct3D 10 interop is not
modeled; no Engine flag).
`gpu-sim` `MaxSharedMemoryPerMultiprocessor` matches
`MaxSharedMemoryPerBlockOptin` (`cudaDevAttrMaxSharedMemoryPerMultiprocessor`;
reserved shared memory is 0; no Engine flag; not occupancy SM counts).
`gpu-sim` linear texture 1D/2D dim caps are always 0
(`cudaDevAttrMaxTexture1DLinearWidth`, `MaxTexture2DLinearWidth`, Height,
and Pitch; `cuDeviceGetTexture1DLinearMaxWidth` is the same 0; no Engine flag).
`gpu-sim` texture 2D gather dim caps are always 0
(`cudaDevAttrMaxTexture2DGatherWidth` and Height; no Engine flag).
`gpu-sim` mipmapped texture 1D/2D dim caps are always 0
(`cudaDevAttrMaxTexture1DMipmappedWidth`, `MaxTexture2DMipmappedWidth`
and Height; no Engine flag).
`gpu-sim` cubemap texture width is always 0
(`cudaDevAttrMaxTextureCubemapWidth`; no Engine flag).
`gpu-sim` layered texture 1D/2D dim caps are always 0
(`cudaDevAttrMaxTexture1DLayeredWidth` and Layers, `MaxTexture2DLayeredWidth`,
Height, and Layers; no Engine flag).
`gpu-sim` cubemap layered texture dim caps are always 0
(`cudaDevAttrMaxTextureCubemapLayeredWidth` and Layers; no Engine flag).
`gpu-sim` surface 1D/2D/3D dim caps are always 0 (`cudaDevAttrMaxSurface1DWidth`,
`MaxSurface2DWidth` and Height, `MaxSurface3DWidth` / Height / Depth;
no Engine flag).
`gpu-sim` layered surface 1D/2D dim caps are always 0
(`cudaDevAttrMaxSurface1DLayeredWidth` and Layers, `MaxSurface2DLayeredWidth`,
Height, and Layers; no Engine flag).
`gpu-sim` cubemap surface dim caps are always 0
(`cudaDevAttrMaxSurfaceCubemapWidth`, `MaxSurfaceCubemapLayeredWidth`
and Layers; no Engine flag).
`gpu-sim` `pciSubSystemID` is always 0 (synthetic PCI; no Engine flag).
`gpu-sim` `GpuPciDeviceId` is always 0 (`cudaDevAttrGpuPciDeviceId`; no
NVIDIA PCI vendor/device id; no Engine flag).
`gpu-sim` `GpuPciSubsystemId` is always 0 (`cudaDevAttrGpuPciSubsystemId`;
same 0 as `pciSubSystemID`; no Engine flag).
`gpu-sim` `luid` and `luidDeviceNodeMask` are always 0 (`cuDeviceGetLuid`;
no Engine flag).
Default `--expert-sim` keeps
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
(weights are kernel reads, so the page stays on home). `--no-preferred`
unsets that advise so the dest GEMM first-touches. `--no-mem-prefetch` skips
fill prefetch so the kernel first-touches. Decode acquires then **leases** each routed expert
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
expertvm sim      trace.jsonl --capacity 8 --mempool-max 1048576
expertvm sim      trace.jsonl --capacity 8 --shareable
expertvm sim      trace.jsonl --capacity 8 --share-ptr
expertvm sim      trace.jsonl --capacity 8 --ipc
expertvm sim      trace.jsonl --capacity 8 --mapped
expertvm sim      trace.jsonl --capacity 8 --managed
expertvm sim      trace.jsonl --capacity 8 --vmm
expertvm sim      trace.jsonl --capacity 8 --vmm-retain
expertvm sim      trace.jsonl --capacity 8 --vmm-handle
expertvm sim      trace.jsonl --capacity 8 --vmm-page 2097152
expertvm sim      trace.jsonl --capacity 8 --host-func
expertvm sim      trace.jsonl --capacity 8 --seq-streams --blocking-streams
expertvm sim      trace.jsonl --capacity 8 --prefetch copy-forward --plan-window 8 --cuda-graphs
expertvm sim      trace.jsonl --capacity 1 --cuda-graphs --graph-update
expertvm sim      trace.jsonl --capacity 1 --graph-set-params
expertvm sim      trace.jsonl --capacity 1 --cuda-graphs --graph-clone
expertvm sim      trace.jsonl --capacity 1 --graph-build
expertvm sim      trace.jsonl --capacity 1 --graph-build --graph-build-deps
expertvm sim      trace.jsonl --capacity 1 --graph-build --graph-host
expertvm sim      trace.jsonl --capacity 1 --graph-piecewise
expertvm sim      trace.jsonl --capacity 1 --graph-piecewise --graph-capture-deps
expertvm sim      trace.jsonl --capacity 1 --graph-piecewise --graph-capture-host
expertvm sim      trace.jsonl --capacity 1 --graph-mem
expertvm sim      trace.jsonl --capacity 1 --graph-mem --graph-memset
expertvm sim      trace.jsonl --capacity 1 --graph-mem --graph-memcpy
expertvm sim      trace.jsonl --capacity 1 --graph-leaf-host
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
expertvm store    trace.jsonl --capacity 1 --graph-build --graph-build-deps
expertvm store    trace.jsonl --capacity 1 --graph-build --graph-host
expertvm store    trace.jsonl --capacity 1 --graph-piecewise
expertvm store    trace.jsonl --capacity 1 --graph-piecewise --graph-capture-deps
expertvm store    trace.jsonl --capacity 1 --graph-piecewise --graph-capture-host
expertvm store    trace.jsonl --capacity 1 --graph-mem
expertvm store    trace.jsonl --capacity 1 --graph-mem --graph-memset
expertvm store    trace.jsonl --capacity 1 --graph-mem --graph-memcpy
expertvm store    trace.jsonl --capacity 1 --graph-leaf-host
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
expertvm sim      trace.jsonl --capacity 2 --seq-streams --compute-slots 4 --cluster 2 --func-cluster-spread --cluster-load-balance
expertvm sim      trace.jsonl --capacity 2 --seq-streams --compute-slots 4 --cluster 2 --cluster-must-set
expertvm sim      trace.jsonl --capacity 2 --seq-streams --compute-slots 4 --cluster 2 --required-cluster 2
expertvm sim      trace.jsonl --capacity 2 --seq-streams --compute-slots 2 --decode-priority --mem-sync-domain remote --mem-sync-map collapse
expertvm sim      trace.jsonl --capacity 2 --seq-streams --compute-slots 2 --decode-priority --mem-sync-domain remote --mem-sync-launch
expertvm sim      trace.jsonl --capacity 2 --seq-streams --compute-slots 2 --decode-priority --mem-sync-domain remote --mem-sync-launch-map
expertvm sim      trace.jsonl --capacity 2 --seq-streams --compute-slots 2 --max-shared
expertvm sim      trace.jsonl --capacity 2 --seq-streams --compute-slots 2 --func-max-shared
expertvm sim      trace.jsonl --capacity 2 --seq-streams --compute-slots 2 --func-max-shared --max-l1
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
