# gpu-sim

Deterministic **GPU-systems** virtual machine for inference software.

Agents and researchers can optimize expert residency, prefetch, and
scheduling against this crate **without a physical GPU**. Semantics are
exact. Timing comes from a [`HardwareProfile`](crate::HardwareProfile),
not from folklore baked into policy code.

This is **not** Accel-Sim / GPGPU-Sim. It does not model warps, register
banks, or Tensor Core circuitry. Those effects enter only as empirical
kernel curves in a profile.

```text
inference system
        │
GPU systems semantics   ← exact (streams, events, residency, OOM, topology)
        │
resource contention     ← copy engines, links, Hyper-Q compute_slots, SM permille
        │
calibrated op costs     ← HardwareProfile
        │
       STOP
─────────────────────────
warp scheduler, L1, …   ← do not model
```

## Exact vs timed

| Exact (tests fail if violated) | Timed (profile data) |
| --- | --- |
| HBM capacity, alloc/free, OOM | kernel microseconds |
| `cudaMallocAsync` (`alloc`) is stream-ordered; pointer not usable until the stream catches up | `alloc_overhead_ns` (first touch) / `pool_reuse_ns` (cached) |
| `cudaMallocAsync` from a pool reuses cached bytes; `cudaMemGetInfo` still counts them used until `pool_trim_to` | `pool_reuse_ns` |
| `cudaMemPoolSetAccess` (`pool_set_access`) ReadWrite on a peer; dest HBM stays 0; writes allowed | interconnect, not local HBM |
| `cudaMalloc` (`malloc`) device-syncs that GPU, then the pointer is usable; it cannot consume another pool's cache | `alloc_overhead_ns` (charged at the call) |
| `cudaIpcGetMemHandle` / `ipc_open` / `ipc_close` share physicals | `alloc_overhead_ns` (export/import) |
| `cudaMemPoolExportToShareableHandle` / `pool_import` share live/cached | `alloc_overhead_ns` (export/import) |
| `cudaMemPoolExportPointer` / `pool_import_ptr` alias pool allocs | `alloc_overhead_ns` (export/import) |
| `cudaDeviceSetMemPool` rebinds `alloc` (`set_device_mempool`) | `alloc_overhead_ns` |
| `cudaHostRegister` pins pageable host for DMA (`host_register`) | `alloc_overhead_ns` (mlock, host-sync) |
| `cudaHostAllocMapped` / `host_register_mapped`: kernel may read host with no H2D | host PCIe vs HBM |
| `cudaMallocManaged` (`alloc_managed`) does not charge HBM until migrate | `alloc_overhead_ns` (VA reserve at the call) |
| `cudaStreamAttachMemAsync` (`stream_attach`) Host/Single visibility | 1 ns stream-ordered |
| `cudaMemAdviseSetReadMostly`: prefetch replicates | same DMA as a move |
| `drop_managed_copy`: dest eviction of one ReadMostly GPU | other copies stay |
| `cudaMemAdviseSetAccessedBy`: kernel may read without migrating | interconnect, not local HBM |
| `cudaMemAdviseSetPreferredLocation`: stay if already there | interconnect on remote read; writes migrate |
| `cudaMemPrefetchAsync` (`prefetch` / `prefetch_host`) **moves** unless ReadMostly | PCIe / NVLink (1 ns if already local) |
| `cuMemAddressReserve` (`va_reserve`) is a VA with no physical pages | `alloc_overhead_ns` (at the call) |
| `cuMemMap` (`va_map`) charges HBM; `va_unmap` refunds; the VA is reusable | `alloc_overhead_ns` (map) |
| `cuMemCreate` (`va_create`) charges HBM with no VA; `va_map_handle` maps it without a second charge; `va_retain_handle` increments refs; `va_release_handle` refunds when refs and maps are 0 | `alloc_overhead_ns` (create, map, retain) |
| `cuMemGetAllocationGranularity` (`va_granularity_bytes`): reserve/map sizes align (`0`/`1` = any) | not timed |
| `cuMemSetAccess` (`va_set_access`) PROT_READ on a peer; dest HBM stays 0; `va_set_access_write` is PROT_READWRITE | interconnect, not local HBM |
| `va_map_range` / `va_unmap_range` map a span; `kernel()` needs the whole VA; `kernel_bufs` / `memset_buf` / `MemcpyOp::offset` touch a mapped span | HBM = mapped bytes |
| `va_release` parks an unmapped VA; `va_acquire` remaps same size | map only on reuse (no second reserve) |
| `va_acquire_paged` maps the VA in `page` physicals | `alloc_overhead_ns` per block |
| `cudaLaunchHostFunc` (`host_func` / `host_func_params`) is stream-ordered host work; `fn_id` / `user_data` are `cudaHostNodeParams` | `host_func_ns` (no compute / copy occupancy) |
| `cuStreamWriteValue32/64` (`write_value32` / `write_value64`) writes a mailbox on complete | 1 ns Solo (no compute / copy occupancy) |
| `cuStreamWaitValue32/64` (`wait_value32` / `wait_value64`) stays pending until the mailbox compare matches; unwritten locations read as 0; kernel/memset/memcpy stores are not modeled | 1 ns Solo when ready; unsatisfied wait + `synchronize` is deadlock |
| `cuStreamBatchMemOp` (`batch_mem_op`) is one stream op for a wait/write vector; a wait sees earlier writes in that vector | 1 ns Solo when ready |
| `cudaStreamCreate` (`set_stream_blocking`) serializes with NULL | copy/compute overlap vs NULL |
| host pin / `mlock` budget (`host_pin_bytes`) | `SimError::PinOom` |
| `cudaFree` (`free_sync`) waits owning GPU(s), then every copy is gone | stream-ordered `free` refunds when that stream runs |
| `cudaMemcpyAsync` of pageable host memory is host-synchronous | `pageable_permille` (bounce + DMA) |
| `cudaMemcpyAsync` of pinned / device memory is stream-ordered | PCIe / NVLink bandwidth |
| `cudaMemcpy` (`memcpy_sync`) waits that stream | pinned `memcpy` does not |
| `synchronize_device` waits one GPU | other GPUs keep running |
| stream order, event dependencies | memcpy microseconds |
| residency: a kernel may only read **device**, **mapped-host**, VMM peer `va_set_access` (reads) / `va_set_access_write` (read/write), or mempool peer `pool_set_access` (read/write) allocations; managed first-touch at kernel start | PCIe / NVLink / HBM bandwidth |
| HBM vs host-pinned: `alloc_host_pinned` does not charge HBM | pageable vs pinned H2D (`pageable_permille`) |
| copy-engine occupancy | launch overhead |
| peer accessibility | size-dependent efficiency |
| graph capture does not execute; launch replays | GEMM util / grouped-MoE ‰ |
| forked capture: `wait_event` on a captured record joins that stream | copy/compute overlap inside one launch |
| `cudaEventRecordExternal` / `cudaEventWaitExternal` do not join capture | live waiters overlap graph launch |
| `launch_graph` during capture is a child-graph node | nested exec expanded at parent launch |
| independent streams stay live during Relaxed capture (default); ThreadLocal/Global refuse uncaptured-stream submits | query/sync of a capturing stream is Invalid |
| graph instantiate is host-sync and returns a new exec id; first launch of a definition creates a primary exec; `instantiate_graph_auto_free` is AutoFreeOnLaunch | `graph_instantiate_ns` |
| graph upload is host-sync after instantiate; first launch pays it once | `graph_upload_ns` |
| `cudaGraphKernelNodeSetParams` / `MemcpyNodeSetParams` / `MemsetNodeSetParams` / `HostNodeSetParams` patch the graph, not an already-instantiated exec | 1 ns host-sync |
| `cudaGraph*NodeGetParams` reads the definition; `graph_exec_*_get_params` reads the exec snapshot | query |
| graph update replaces the exec snapshot when topology matches (device, stream, kind, deps); mem nodes are Invalid; `update_graph_with_info` fills `cudaGraphExecUpdateResultInfo` | `graph_update_ns` |
| `cudaGraphExecKernelNodeSetParams` patches one instantiated kernel node's pointers / kind (mem nodes legal) | `graph_set_params_ns` |
| `cudaGraphExecMemcpyNodeSetParams` patches one instantiated memcpy node's `MemcpyOp` (mem nodes legal) | `graph_set_params_ns` |
| `cudaGraphExecMemsetNodeSetParams` patches one instantiated memset node's dest span (mem nodes legal) | `graph_set_params_ns` |
| `cudaGraphExecHostNodeSetParams` patches one instantiated host node's `fn_id` / `userData` (mem nodes legal) | `graph_set_params_ns` |
| `cudaGraphNodeSetEnabled` skips an instantiated node at launch (mem nodes illegal) | `graph_set_params_ns` |
| `cudaGraphExecChildGraphNodeSetParams` swaps one instantiated child-graph node's nested graph (nested topology must match; child ids are topology for ExecUpdate) | `graph_set_params_ns` |
| `cudaGraphExecEventRecordNodeSetEvent` / `WaitNodeSetEvent` retarget the event on an instantiated record/wait node (External flag is topology) | `graph_set_params_ns` |
| graph mem alloc/free nodes (`cudaMallocAsync` / `cudaFreeAsync` during capture, or `graph_add_alloc` / `graph_add_free`) | `pool_reuse_ns` on relaunch without free |
| graph clone is an independent uninstantiated copy; child graphs cloned recursively; mem alloc nodes get new ids | `graph_clone_ns` |
| `cudaGraphCreate` (`create_graph`) is an empty uninstantiated graph | 1 ns host-sync |
| `cudaStreamBeginCaptureToGraph` (`begin_capture_to_graph`) appends captured nodes onto an existing uninstantiated graph; empty deps are extra roots | not timed (capture) |
| `cudaGraphGetRootNodes` / `GetEdges` / `NodeGetDependentNodes` | query |
| `cudaGraphAddKernelNode` / memcpy / memset / host / empty / event / child / mem alloc/free / cooperative kernel / dependencies (`graph_add_*`) | not timed (host-side topology) |
| graph destroy drops the id (`cudaGraphDestroy`); remaining graph mem is refunded; user-object refs held by the graph are released | 1 ns host-sync |
| `cudaUserObjectCreate` / `Retain` / `Release`; last ref records the destroy `fn_id` | 1 ns host-sync |
| `cudaGraphRetainUserObject` / `ReleaseUserObject` on a definition (`MOVE` transfers one caller ref); clone does not copy retains | 1 ns host-sync |
| graph launch amortizes per-kernel launch overhead | `graph_launch_ns` |
| `synchronize_stream` waits one stream only | other streams keep running |
| `synchronize_event` waits the record only | later ops on that stream keep running |
| `idle_until` drains, then jumps the clock | GPU idle until the next arrival |
| `event_elapsed_ns` is record-to-record delta | `cudaEventElapsedTime` (ns) |
| `cudaEventDisableTiming` forbids elapsed | wait / query still work |
| `query_event` is non-blocking | `cudaEventQuery` |
| `query_stream` is non-blocking | `cudaStreamQuery` |
| `mem_info` is `(free, total)` HBM | `cudaMemGetInfo` |
| graph-mem attr is live graph allocs only; reserved equals used | `cudaDeviceGetGraphMemAttribute` |
| stream[i+1].start ≥ stream[i].finish (`Operation` timestamps) | queue wait vs run |
| higher `set_stream_priority` starts first under contention | launch overhead |
| `compute_slots>=2` overlaps independent kernels at full issue rate | kernel duration (not SM-partition) |
| `cudaLaunchCooperativeKernel` occupies every compute slot (`cooperative_kernel`) | leftover kernels cannot Hyper-Q overlap |
| programmatic dependent launch: wait kernel starts after the previous same-stream trigger (`pdl_trigger_permille`) | overlap needs `compute_slots >= 2` |
| `cudaLaunchAttributeAccessPolicyWindow` persisting hits (`kernel_access_policy`) | HBM discount after `set_persisting_l2_cache_size`; CUDA default size is 0 |
| `cudaLaunchAttributeMemSyncDomain` fence isolation (`kernel_with` / allreduce Remote) | `same_domain_fence_permille` of leftover same-domain traffic; tax default 0 |
| `cudaLaunchAttributeClusterDimension` (`kernel_with` cluster) | occupies `min(blocks, compute_slots)`; Hopper portable max 8 |
| `set_stream_sm_permille` is a green-context SM fraction (‰) | compute-bound kernels scale; memory-bound keep full HBM |
| `memset` / `memset_buf` needs the filled span resident (not mapped host) | HBM write + launch overhead |
| peer D2D needs topology + `enable_peer` | link bandwidth |
| legacy null stream serializes (opt-in) | copy/compute overlap |
| `cudaStreamCreate` blocking stream serializes with NULL | `cudaStreamNonBlocking` overlap |

## Anti-Goodhart timing

Memcpy cost is

```text
T = T_fixed + (align_up(bytes, align_bytes) + ramp) / peak_bandwidth
```

`align_bytes` (host PCIe default 128) so a 1-byte DMA cannot beat a
cache-line copy. Eight thousand tiny copies cannot harvest full PCIe
bandwidth. Concurrent copies on the same link share bandwidth. Kernels on
one GPU default to exclusive compute (`compute_slots=1`); `>=2` is Hyper-Q
occupancy at full issue rate (not an SM-partition model).
`set_stream_sm_permille` is the green-context SM fraction: compute-bound
kernels scale as `1000 / permille`; memory-bound keep full HBM. Default
unset is a full chip (`1000`). Copy engines still overlap compute. Profile knobs `gemm_util_permille` (achieved/peak) and `grouped_moe_permille`
(grouped vs dense duration) scale kernel time. Defaults are 1000
(identity roofline). They are parseable; they are not a capture. Host PCIe
links also carry `pageable_permille` (default `500`: pageable H2D takes
twice pinned DMA). That is an example knob, not a capture.

## Example profiles

`profiles/*.profile` are **examples for tests and development**. They are
order-of-magnitude public-spec numbers, not a capture from a real node.
Replace them with `HardwareProfile::parse` of a `gpu-profile capture`
once someone with a GPU runs one.

## Library

```rust
use gpu_sim::{DeviceId, GpuOp, HardwareProfile, KernelKind, Sim, StreamId};

let mut sim = Sim::new(HardwareProfile::example_h100_sxm());
let d0 = DeviceId(0);
let s0 = StreamId(0);
let a = sim.alloc(d0, 8 << 20, s0).expect("hbm");
sim.memcpy_pinned_to_device(d0, a, 8 << 20, s0).expect("h2d");
sim.kernel(d0, KernelKind::other(1 << 30, 8 << 20), &[a], &[a], s0)
    .expect("k");
sim.synchronize().expect("sync");
assert!(sim.clock_ns() > 0);
let dag: Vec<_> = sim.operations().collect();
assert!(dag.iter().any(|op| matches!(op.kind, GpuOp::Kernel { .. })));
```

## Scores

[`Score`](crate::Score) splits the two numbers agents must not mix:

- **semantic**: `Ok` or a [`SimError`](crate::SimError) (binary)
- **performance**: `wall_ns`, `hbm_peak`, `bytes_moved`, `energy_uj`, optional
  `ns_per_token`, `ttft_ns`, `itl_ns`, optional `usd_micros_per_m_tokens`

`Score::with_tokens(n)` fills `ns_per_token = wall_ns / n` and, when the
profile has `rent_usd_micros_per_hour > 0`,
`usd_micros_per_m_tokens = rent × wall / tokens` (microdollars per million
tokens). Example profiles leave rent at `0`.
`Score::with_latencies` attaches TTFT / mean ITL when a caller samples the
virtual clock at token boundaries (`expertvm::sim_replay` does this).
`energy_uj` is profile board TDP × virtual wall (`mW × ns / 1e6`). Device-to-device replica copies charge the
destination HBM at memcpy start and OOM if that GPU is full. `free` of a
replicated allocation only drops `live` when no device still holds it.

Adversarial memcpy: many tiny copies cannot beat one large copy of the
same payload (fixed overhead + size-dependent bandwidth). Two concurrent
**pinned** H2D copies on separate streams share PCIe and cannot finish in
one-copy time. Pageable `memcpy_host_to_device` waits the stream (CUDA
staging bounce); two pageable copies from the host cannot DMA together.

## Topologies

Named example profiles (order-of-magnitude, **not captures**):

| Name | Mesh |
| --- | --- |
| `h100` / `h200` / `cheap` | 1 GPU, host PCIe |
| `2xh100-pcie` | 2 GPU, PCIe P2P (no NVLink) |
| `8xh100` | 8 GPU NVLink clique + per-GPU PCIe |
| `bad-numa` | 2 GPU, GPU1 on a slow far PCIe root, no P2P |
| `2node-rdma` | 2 GPU, GPU-direct RDMA, no NVLink |
| `asymmetric` | 3 GPU NVLink chain 0–1–2; 0↔2 is `NoPeer` |

`HardwareProfile::by_name`, `probe_topology`, and `gpu-profile probe NAME`
measure **pinned** H2D per GPU and D2D per pair. Missing links print
`p2p=0->2:none`. `restrict_hbm(bytes)` caps every GPU for the static-EP vs
cache experiment. `Place::{Host, HostPinned, Device}`: pageable H2D is
`memcpy_host_to_device` and is **host-synchronous** (real
`cudaMemcpyAsync` of pageable memory); DMA is `memcpy_pinned_to_device`.
Pageable D2H is `memcpy_device_to_host`. Capture cannot include a pageable copy.
`alloc_host_pinned` is live immediately, does not count toward HBM, and a
kernel on it fails `NotResident` until a copy places it on a device.

## Faults

| Inject | Semantic error |
| --- | --- |
| `Sim::set_unavailable` | `SimError::Unavailable` |
| `Sim::cancel_stream` (queued ops) | `SimError::Cancelled` |
| `Sim::fail_next_memcpy` | `SimError::TransferFailed` (expert load) |
| `Sim::set_extra_transfer_ns` | longer memcpy / allreduce, still `Ok` |
| over-capacity alloc | `SimError::Oom` |

CUDA graphs: `begin_capture` / `begin_capture_to_graph` / `end_capture` /
`instantiate_graph` /
`update_graph` / `clone_graph` / `destroy_graph` / `launch_graph`. Capture does
not advance the virtual clock. Independent streams stay live. A stream that
`wait_event`s an event recorded in this capture joins (CUDA forked capture);
`record_event_external` / `wait_event_external` (`cudaEventRecordExternal` /
`cudaEventWaitExternal`) do not join, so a live waiter can overlap graph launch.
`launch_graph` remaps origin-stream nodes onto the launch stream so copy and
compute can overlap. Query or `synchronize_stream` of a capturing stream, and
node `synchronize`, are `Invalid`. `launch_graph` during capture records a
child-graph node if the child is already instantiated; parent launch expands
it. Independent streams still launch live. `cudaMallocAsync` / `cudaFreeAsync`
(`alloc` / `free`) during capture are graph mem alloc/free nodes.
`graph_add_alloc` / `graph_add_free` are `cudaGraphAddMemAllocNode` /
`cudaGraphAddMemFreeNode` (same reuse / AutoFreeOnLaunch rules).
`graph_add_empty` is `cudaGraphAddEmptyNode` (1 ns; no compute/copy occupancy).
`graph_add_write_value64` / `graph_add_wait_value64` /
`graph_add_batch_mem_op` are `cudaGraphAddBatchMemOpNode` (`cuStreamWaitValue` /
`WriteValue`; a multi-item batch is **one** node holding the vector).
`batch_mem_op` is live `cuStreamBatchMemOp`.
`graph_add_dependencies` is `cudaGraphAddDependencies` (independent nodes
may Hyper-Q overlap at launch; capture records same-stream edges).
`graph_remove_dependencies` is `cudaGraphRemoveDependencies` (illegal on an
exec and during capture). `begin_capture_to_graph` is
`cudaStreamBeginCaptureToGraph`: append captured nodes onto an existing
uninstantiated graph; capture roots additionally depend on the given node
indices (empty `deps` means extra roots, so they may Hyper-Q overlap).
`graph_root_nodes` / `graph_edges` / `graph_node_dependents` are
`cudaGraphGetRootNodes` / `GetEdges` / `NodeGetDependentNodes`. Host-sync
`malloc` / `free_sync` / `memcpy_sync` / `synchronize_device` / VMM / mempool
create cannot be captured. A graph that allocates without a matching free
reuses the pointer on later launches (no second HBM charge) unless
`instantiate_graph_auto_free` (`cudaGraphInstantiateFlagAutoFreeOnLaunch`)
stream-ordered-frees those allocs before relaunch. `clone_graph`
forks those ids. `destroy_graph` refunds remaining graph mem. `update_graph`
of mem nodes is `Invalid`.
Instantiate, update, and upload are host-synchronous and cannot run during capture.
`instantiate_graph_with_flags` is `cudaGraphInstantiateWithFlags`
(`GraphInstantiateFlags::UPLOAD` uploads during instantiate;
`USE_NODE_PRIORITY` schedules recorded kernels with the add/capture
priority; `DEVICE_LAUNCH` enables `device_launch_graph` after upload —
host `launch_graph` stays legal; mem alloc/free, events, child graphs,
conditionals, and host nodes are Invalid; `update_graph` of a
device-launch exec is Invalid).
`instantiate_graph_with_params` is `cudaGraphInstantiateWithParams`
(`GraphInstantiateParams` result and err node).
`graph_exec_get_flags` is `cudaGraphExecGetFlags`.
Instantiate returns a new exec id (`cudaGraphExec_t`); the source graph
stays a definition. `launch_graph` of a definition uses the primary exec.
`clone_graph` is `cudaGraphClone` (`graph_clone_ns`): an independent
uninstantiated copy; child-graph nodes are cloned recursively (a diamond
of shared children becomes one cloned child). Destroying the original
child still breaks a parent that names it; a recursive clone of that
parent keeps working. `destroy_graph` is `cudaGraphDestroy` (1 ns;
later launch is unknown; remaining graph mem is refunded; user-object
refs held by the graph are released).
`user_object_create` is `cudaUserObjectCreate`
(`UserObjectFlags::NO_DESTRUCTOR_SYNC`; last ref records `destroy_fn`).
`graph_retain_user_object` / `graph_release_user_object` are
`cudaGraphRetainUserObject` / `ReleaseUserObject` on a definition
(`GraphUserObjectFlags::MOVE` transfers one caller ref). Clone does not
copy retains. Capture cannot include them. First launch instantiates if needed (`graph_instantiate_ns` once)
then uploads if needed (`graph_upload_ns`). `upload_graph` is `cudaGraphUpload`.
`update_graph` copies source steps into the exec snapshot when the
device, stream, op kinds, and dependency edges match (`graph_update_ns`); a topology
mismatch is `Invalid`. Graphs with mem alloc/free nodes cannot be updated.
`update_graph_with_info` is `cudaGraphExecUpdate` with
`cudaGraphExecUpdateResultInfo` (filled even on `Err`: node type, deps,
mem nodes, device-launch). `update_graph` uses that path and keeps the
same `why` strings.
`graph_kernel_set_params` / `graph_memcpy_set_params` /
`graph_memset_set_params` / `graph_batch_mem_op_set_params` /
`graph_batch_mem_ops_set_params` are
`cudaGraphKernelNodeSetParams` /
`MemcpyNodeSetParams` / `MemsetNodeSetParams` / `BatchMemOpNodeSetParams` on the graph
definition (do not retarget an already-instantiated exec).
`graph_*_get_params` / `graph_exec_*_get_params` are
`cudaGraph*NodeGetParams` / `cudaGraphExec*NodeGetParams`
(query; no clock tick; capture is legal). Graph GetParams reads the
definition; Exec GetParams reads the snapshot (`as_exec`; uninstantiated
is Invalid). Unique-node helpers (`graph_unique_kernel`, …) still use
the launched/primary snapshot.
`graph_exec_kernel_set_params` / `graph_exec_memcpy_set_params` /
`graph_exec_memset_set_params` / `graph_exec_batch_mem_op_set_params` /
`graph_exec_batch_mem_ops_set_params` are
`cudaGraphExecKernelNodeSetParams` / `cudaGraphExecMemcpyNodeSetParams` /
`cudaGraphExecMemsetNodeSetParams` / `cudaGraphExecBatchMemOpNodeSetParams`
(`graph_set_params_ns`; mem nodes legal; pageable memcpy stays illegal;
a `GpuOp::BatchMem` item list is a parameter).
`graph_node_set_enabled` is `cudaGraphNodeSetEnabled` (skip a node at launch;
mem alloc/free cannot be disabled).
`graph_exec_child_set_params` is `cudaGraphExecChildGraphNodeSetParams`
(swap the nested graph; nested topology must match; child ids are topology
for `update_graph`; mem nodes legal).
`graph_exec_event_record_set_event` / `graph_exec_event_wait_set_event` are
`cudaGraphExecEventRecordNodeSetEvent` / `WaitNodeSetEvent` (event id is the
parameter; External is topology).
`stream_update_capture_dependencies` is `cudaStreamUpdateCaptureDependencies`
(extra deps for the next captured node, in addition to stream-order; `Set`
replaces, `Add` unions). `stream_is_capturing` / `stream_capture_info` are
`cudaStreamIsCapturing` / `GetCaptureInfo` (includes capture mode).
`begin_capture_with_mode` is `cudaStreamBeginCapture` with
`StreamCaptureMode` (default Relaxed: independent streams stay live; a wait
of a captured record still joins. ThreadLocal/Global refuse uncaptured-stream
submits). `thread_exchange_stream_capture_mode` is
`cudaThreadExchangeStreamCaptureMode`. `graph_node_kind` is
`cudaGraphNodeGetType`.
`graph_conditional_create` / `graph_add_if` are
`cudaGraphConditionalHandleCreate` and an IF node. Body ops skip at
start when the handle is `0`. `set_conditional` is device
`cudaGraphSetConditional` (each launch resets to the create-time default).
`graph_add_while` / `graph_while_nodes` / `graph_add_switch` /
`graph_switch_nodes` are WHILE / SWITCH (WHILE caps at
64 iterations; SWITCH runs body `i` when the handle equals `i`).
`graph_node_find_in_clone` is `cudaGraphNodeFindInClone` (same index on a
graph produced by `clone_graph` of that original).
`expertvm --graph-set-params` parks a leaf and retargets the unique kernel
(and a unique memcpy or memset if present). `expertvm --graph-update` parks a leaf GEMM on
evict and updates the next miss instead of instantiate. `--graph-clone`
copies the capture (`cudaGraphClone`) before instantiate. `--graph-build` is
`cudaGraphCreate` / `cudaGraphAdd*` (no idle stream; combo children may
Hyper-Q overlap unless `graph_add_dependencies` chains them). `--graph-piecewise`
is `cudaStreamBeginCaptureToGraph` combo parents (independent child roots).
`--graph-mem` is in-graph
scratch (`graph_add_alloc` / capture `alloc`). `--graph-auto-free` is
AutoFreeOnLaunch (relaunch recharges HBM; not with `--graph-mem`).
`cooperative_kernel` / `graph_add_cooperative_kernel` are
`cudaLaunchCooperativeKernel` (occupy every Hyper-Q slot; capture allowed).
Launch pays `graph_launch_ns` once; recorded
kernels skip per-kernel launch overhead.
`memset` is an HBM-write kernel on a resident alloc. `host_func` is
`cudaLaunchHostFunc`: stream-ordered host work that does not occupy compute
or copy engines (other streams may GEMM). Unnamed callback by default;
`host_func_params` records `HostNodeParams`. `write_value64` / `wait_value64`
are `cuStreamWriteValue64` / `WaitValue64` (mailbox; no occupancy).
`batch_mem_op` is `cuStreamBatchMemOp`. Peer D2D requires a
topology link **and** directed `enable_peer` (seeded on for every GPU↔GPU
link; `disable_peer` → `PeerDisabled`). [`StreamId::NULL`] is the CUDA null
stream; `set_legacy_null_stream(true)` serializes it with every other stream
on that device (CUDA legacy default stream). Off by default is the
per-thread default: NULL serializes only with `set_stream_blocking`
streams (`cudaStreamCreate`). Created streams default to
`cudaStreamNonBlocking`.
`synchronize_stream` is `cudaStreamSynchronize`. `synchronize_device` is
`cudaDeviceSynchronize` (one GPU; other GPUs keep running).
`synchronize_event` is `cudaEventSynchronize` (later ops on that stream keep running).
`idle_until` drains in-flight work, then jumps the virtual clock so an
open-loop arrival can wait without `sleep`.
`event_elapsed_ns` is `cudaEventElapsedTime` in nanoseconds (both records
must be complete and timing-enabled; `create_event_disable_timing` is
invalid; end-before-start is invalid).
`query_event` is `cudaEventQuery` (unknown id is semantic; incomplete is
`Ok(false)`).
`query_stream` is `cudaStreamQuery` (unknown device is semantic; a busy
stream is `Ok(false)`; the clock does not advance).
`mem_info` is `cudaMemGetInfo` `(free, total)` HBM bytes.
`graph_mem_get` / `graph_mem_set` / `graph_mem_trim` are
`cudaDeviceGetGraphMemAttribute` / `SetGraphMemAttribute` / `GraphMemTrim`
(graph allocs only; reserved equals used; trim does not change `mem_info`).
Default `cudaMallocAsync` uses the device mempool with release threshold
`0` (unused bytes return to the OS when the stream-ordered free
completes). `create_pool` / `alloc_from_pool` /
`set_pool_release_threshold` / `pool_trim_to` are `cudaMemPoolCreate` /
`cudaMallocFromPoolAsync` / `cudaMemPoolAttrReleaseThreshold` /
`cudaMemPoolTrimTo`. `u64::MAX` holds unused bytes so `malloc` can OOM
until trim. Capture cannot include pool create/trim/set-attribute.
`ipc_get` / `ipc_open` / `ipc_close` are `cudaIpcGetMemHandle` /
`cudaIpcOpenMemHandle` / `cudaIpcCloseMemHandle`: the import aliases the
source physicals (no extra HBM). Free of the source while imports are live
is Invalid. `ipc_get` of a mempool alloc is Invalid. Capture cannot include IPC.
`create_shareable_pool` is `cudaMemPoolCreate` with a POSIX-FD handle type.
`pool_export` / `pool_import` are `cudaMemPoolExportToShareableHandle` /
`ImportFromShareableHandle`: the import is a new pool id that shares
live/cached/threshold with the exporter. `pool_export_ptr` /
`pool_import_ptr` are `cudaMemPoolExportPointer` / `ImportPointer` (alias,
no extra HBM). `set_device_mempool` is `cudaDeviceSetMemPool`. Default and
`create_pool` pools cannot be exported. Capture cannot include shareable
export/import.
`alloc_host` is pageable; `host_register` / `host_register_mapped` are
`cudaHostRegister` (host-synchronous). `alloc_host_mapped` is
`cudaHostAllocMapped`: a kernel may read it with no H2D, billed at host
PCIe, and it does not charge HBM. Capture cannot include host
alloc/register. `alloc_managed` is `cudaMallocManaged` (no HBM until
`prefetch` / first-touch at kernel start). Default attach is Global.
`alloc_managed_host` is `cudaMemAttachHost`. `stream_attach` is
`cudaStreamAttachMemAsync` (stream-ordered; Host and other-stream Single
fail device kernels / memset / device prefetch; Single cannot use the NULL
stream; capture is refused). `mem_advise` is `cudaMemAdvise` (host-sync).
`SetReadMostly` makes prefetch replicate; `SetAccessedBy` lets a kernel
read without migrating. `SetPreferredLocation` keeps a page already at
that GPU there on a remote read (writes still migrate; host preferred
does not skip kernel first-touch). `prefetch` / `prefetch_host` are
`cudaMemPrefetchAsync` and **move** unless ReadMostly. Capture of
`alloc_managed` / `mem_advise` / `stream_attach` is refused; a graph must record prefetch
before the kernel unless AccessedBy or PreferredLocation covers that GPU.
`va_reserve` / `va_map` / `va_unmap` / `va_free` are CUDA virtual memory.
`va_granularity_bytes` is `cuMemGetAllocationGranularity` (`0`/`1` accepts
any size; a 2 MiB profile rejects unaligned reserve/map).
`va_map_range` / `va_unmap_range` map sparse physicals (HBM is the mapped
span). `va_create` is `cuMemCreate` (HBM, no VA). `va_map_handle` is `cuMemMap`
of that handle (no second HBM charge; two VAs may share it). `va_retain_handle`
is `cuMemRetainAllocationHandle` (combined `va_map` spans are promoted).
`va_release_handle` is `cuMemRelease` while mapped; HBM refunds when refs and
maps are 0. `va_map` still Create+Maps in one call.
`va_set_access` is `cuMemSetAccess` PROT_READ on a peer (no dest HBM;
writes still need a local map). `va_set_access_write` is PROT_READWRITE
(peer writes, no dest HBM). `pool_set_access` is `cudaMemPoolSetAccess`
ReadWrite on a peer (no dest HBM; kernels may write). `kernel()` needs the whole VA covered; `kernel_bufs`, `memset_buf`, and
`MemcpyOp::offset` touch a mapped page (paged KV). `va_acquire` remaps an idle VA of the same
size (or reserves); `va_acquire_paged` maps KV-block physicals covering the VA;
`va_release` unmaps into that pool. Capture cannot
include them.
`host_func` is `cudaLaunchHostFunc` (stream-ordered; other streams can compute;
unnamed callback). `host_func_params` / `graph_add_host_func_params` record
`HostNodeParams`. `graph_host_set_params` is `cudaGraphHostNodeSetParams`
(definition; does not retarget an exec). `graph_exec_host_set_params` is
`cudaGraphExecHostNodeSetParams`.
`write_value64` / `wait_value64` are `cuStreamWriteValue64` /
`cuStreamWaitValue64` (mailbox on complete; unwritten locations read as 0;
kernel/memset/memcpy stores are not modeled; no compute/copy occupancy).
`batch_mem_op` is `cuStreamBatchMemOp` (one stream op; a wait sees earlier
writes in that vector).
`set_stream_blocking` is `cudaStreamCreate` vs `cudaStreamNonBlocking`
(NULL serializes with blocking streams; created streams default to
non-blocking). `set_legacy_null_stream` is the CUDA legacy default
stream (NULL serializes with every stream).
`host_pin_bytes` caps page-locked host (`cudaMallocHost` / `cudaHostRegister`);
overflow is `PinOom`. Example default is unlimited.
`set_stream_priority` is `cudaStreamCreateWithPriority` (higher first when
compute contends). `stream_copy_attributes` is `cudaStreamCopyAttributes`
(priority, SM permille, and mem-sync domain/map). `graph_kernel_node_get_priority` /
`set_priority` / `copy_attributes` are `cudaGraphKernelNodeGetAttribute` /
`SetAttribute` / `CopyAttributes` for priority, programmatic dependent
launch (`ProgrammaticLaunch`), programmatic event (`ProgrammaticEvent`),
access-policy window (`AccessPolicyWindow`), and mem-sync domain/map.
`kernel_pdl` is `cudaLaunchKernelEx` PDL:
a wait kernel may start after the previous same-stream kernel's trigger
(`pdl_trigger_permille`) instead of its completion. Overlap needs
`compute_slots >= 2`. A later `cudaFreeAsync` on that stream still waits
for the overlapped primary (all preceding work). `kernel_pdl_event` is
`cudaLaunchAttributeProgrammaticEvent`: other streams may `wait_event`
at the trigger instead of kernel completion. `kernel_launch_completion` is
`cudaLaunchAttributeLaunchCompletionEvent`: the event records when the
kernel starts. `expertvm sim --pdl` / `gguf_gemv engine --expert-sim --pdl`
launch grouped expert GEMMs that way. `kernel_access_policy` is
`cudaLaunchAttributeAccessPolicyWindow`: persisting hits reduce billed HBM
after `set_persisting_l2_cache_size` (CUDA default is 0). `expertvm sim --l2-persist`
enables the persist limit and attaches a window to expert GEMMs.
`kernel_with` also accepts `cudaLaunchAttributeMemSyncDomain` /
`MemSyncDomainMap`: a completing kernel waits `same_domain_fence_permille` of
leftover same-physical-domain traffic (default tax 0). Remote (and allreduce)
isolates communication. `ClusterDim` is `cudaLaunchAttributeClusterDimension`:
the launch occupies `min(blocks, compute_slots)` Hyper-Q slots (Hopper portable
max 8). `expertvm sim --cluster N` / `gguf_gemv engine --expert-sim --cluster N`
launch grouped expert GEMMs that way. Decode identity stays `kernel`. `USE_NODE_PRIORITY` at
instantiate schedules those node priorities instead of the launch stream. `set_created_streams_priority` assigns created streams
their id. `set_stream_sm_permille` is a green-context SM fraction
(compute-bound kernels scale; memory-bound keep full HBM; default unset is
a full chip). `Operation` carries `submit_ns` / `start_ns` / `done_ns`
so stream[i+1].start ≥ stream[i].finish is inspectable. `GpuOp` /
`Operation` is the compiled submit DAG (`Sim::operations`).

In-flight ops are not cancelled. `gpu-profile capture` is refused in this
crate: someone with a GPU writes a `key=value` file; agents `parse` it.
