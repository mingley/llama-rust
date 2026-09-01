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
| `ReuseAllowOpportunistic=0` skips that cache reuse (OS alloc; cache stays reserved) | `alloc_overhead_ns` |
| `cudaMemPoolProps::maxSize` (`create_pool_with_props`); reserved cannot grow past it | OOM |
| `cudaMemPoolAttrMaxPoolSize` (`pool_set_attribute` / `set_pool_max_size`); Get reports the cap (`0` unlimited); shrinking does not free live allocs | OOM on later alloc |
| `cudaMemPoolSetAccess` (`pool_set_access`) ReadWrite on a peer; dest HBM stays 0; writes allowed | interconnect, not local HBM |
| `cudaMemPoolSetAccess` ProtRead (`pool_set_access_read`); dest HBM stays 0; writes `NotResident` | interconnect, not local HBM |
| `cudaMalloc` (`malloc`) device-syncs that GPU, then the pointer is usable; it cannot consume another pool's cache | `alloc_overhead_ns` (charged at the call) |
| `cudaIpcGetMemHandle` / `ipc_open` / `ipc_close` share physicals (`ipc_open_with_flags` lazy-peer is a no-op) | `alloc_overhead_ns` (export/import) |
| `cudaIpcGetEventHandle` / `ipc_open_event` share the source record | 1 ns (export/import) |
| `cudaMemPoolExportToShareableHandle` / `pool_import` share live/cached (`pool_export_with_type` / `pool_import_with_type` POSIX-FD flags 0) | `alloc_overhead_ns` (export/import) |
| `cudaMemPoolExportPointer` / `pool_import_ptr` alias pool allocs | `alloc_overhead_ns` (export/import) |
| `cudaDeviceSetMemPool` rebinds `alloc` (`set_device_mempool`); GetDefaultMemPool stays | `alloc_overhead_ns` |
| `cudaHostRegister` pins pageable host for DMA (`host_register`) | `alloc_overhead_ns` (mlock, host-sync) |
| `cudaHostAllocMapped` / `host_register_mapped`: kernel may read host with no H2D | host PCIe vs HBM |
| `cudaMallocManaged` (`alloc_managed` / `alloc_managed_with_flags` Global/Host) does not charge HBM until migrate | `alloc_overhead_ns` (VA reserve at the call) |
| `cudaStreamAttachMemAsync` (`stream_attach` / `stream_attach_with_flags` / `stream_attach_with_size`) Host/Single visibility | 1 ns stream-ordered |
| `cudaMemAdviseSetReadMostly`: prefetch replicates | same DMA as a move |
| `drop_managed_copy`: dest eviction of one ReadMostly GPU | other copies stay |
| `cudaMemAdviseSetAccessedBy`: kernel may read without migrating | interconnect, not local HBM |
| `cudaMemAdviseSetPreferredLocation`: stay if already there | interconnect on remote read; writes migrate |
| `cudaMemPrefetchAsync` (`prefetch` / `prefetch_host` / `prefetch_with_flags` / `prefetch_with_size`) **moves** unless ReadMostly | PCIe / NVLink (1 ns if already local) |
| `cudaMemPrefetchBatchAsync` / `DiscardBatchAsync` / `DiscardAndPrefetchBatchAsync` require CMA on every GPU | this VM reports `ConcurrentManagedAccess` 0 → Invalid |
| `cuMemAddressReserve` (`va_reserve`) is a VA with no physical pages | `alloc_overhead_ns` (at the call) |
| `cuMemMap` (`va_map`) charges HBM; `va_unmap` refunds; the VA is reusable | `alloc_overhead_ns` (map) |
| `cuMemCreate` (`va_create`) charges HBM with no VA; `va_map_handle` maps it without a second charge; `va_retain_handle` increments refs; `va_release_handle` refunds when refs and maps are 0 | `alloc_overhead_ns` (create, map, retain) |
| `cuMemGetAllocationGranularity` (`va_granularity_bytes`): reserve/map sizes align (`0`/`1` = any) | not timed |
| `cuMemSetAccess` (`va_set_access` / `va_set_access_with_flags`) PROT_READ on a peer; dest HBM stays 0; `va_set_access_write` is PROT_READWRITE | interconnect, not local HBM |
| `va_map_range` / `va_unmap_range` map a span; `kernel()` needs the whole VA; `kernel_bufs` / `memset_buf` / `MemcpyOp::offset` touch a mapped span | HBM = mapped bytes |
| `va_release` parks an unmapped VA; `va_acquire` remaps same size; `va_reserve_idle` reuses that VA with no physicals | map only on reuse (no second reserve) |
| `va_acquire_paged` maps the VA in `page` physicals | `alloc_overhead_ns` per block |
| `cudaLaunchHostFunc` (`host_func` / `host_func_params`) is stream-ordered host work; `fn_id` / `user_data` are `cudaHostNodeParams` | `host_func_ns` (no compute / copy occupancy) |
| `cudaStreamAddCallback` (`stream_add_callback` / `stream_add_callback_params`) is the same host enqueue as `host_func` but cannot be captured; `stream_add_callback_with_flags` is the CUDA flags word (`StreamCallbackFlags`; must be 0) | `host_func_ns` (no compute / copy occupancy) |
| `cuStreamWriteValue32/64` (`write_value32` / `write_value64`) writes a mailbox on complete; `write_value32_with_flags` / `write_value64_with_flags` is the CUDA flags word (`WriteValueFlags`; NO_MEMORY_BARRIER Invalid) | 1 ns Solo (no compute / copy occupancy) |
| `cuStreamWaitValue32/64` (`wait_value32` / `wait_value64`) stays pending until the mailbox compare matches; unwritten locations read as 0; kernel/memset/memcpy stores are not modeled; `wait_value32_with_flags` / `wait_value64_with_flags` is the CUDA flags word (`WaitValueFlags`; `FLUSH` is stream-ordered RDMA flush) | 1 ns Solo when ready; unsatisfied wait + `synchronize` is deadlock |
| `cuStreamBatchMemOp` (`batch_mem_op`) is one stream op for a wait/write/flush vector; a wait sees earlier writes in that vector; `FlushRemoteWrites` is stream-ordered 1 ns Solo on RDMA SKUs (not host-sync); `batch_mem_op_with_flags` is the CUDA flags word (`BatchMemOpFlags`; must be 0) | 1 ns Solo when ready |
| `cudaStreamCreate` (`set_stream_blocking`) serializes with NULL | copy/compute overlap vs NULL |
| host pin / `mlock` budget (`host_pin_bytes`) | `SimError::PinOom` |
| `cudaFree` (`free_sync`) waits owning GPU(s), then every copy is gone | stream-ordered `free` refunds when that stream runs |
| `cudaMemcpyAsync` of pageable host memory is host-synchronous | `pageable_permille` (bounce + DMA) |
| `cudaMemcpyAsync` of pinned / device memory is stream-ordered | PCIe / NVLink bandwidth |
| `cudaMemcpy` (`memcpy_sync`) waits that stream | pinned `memcpy` does not |
| `cudaMemcpyPeer` (`memcpy_peer`) waits that stream; `memcpy_peer_async` is stream-ordered | NVLink / PCIe P2P |
| `cudaMemcpy3DPeer` (`memcpy_peer_3d`) waits that stream; `memcpy_peer_3d_async` bills payload not padding | NVLink / PCIe P2P |
| `cudaMemcpy2DPeer` (`memcpy_peer_2d`) waits that stream; `memcpy_peer_2d_async` bills payload not padding | NVLink / PCIe P2P |
| `cudaMemcpy2D` (`memcpy_2d`) waits that stream; `memcpy_2d_async` bills payload not padding | PCIe / NVLink / HBM |
| `cuMemcpyHtoD` (`memcpy_htod`) waits that stream; `memcpy_pinned_to_device` is `cuMemcpyHtoDAsync` | PCIe |
| `cuMemcpyDtoH` (`memcpy_dtoh`) waits that stream; `memcpy_device_to_pinned` is `cuMemcpyDtoHAsync` | PCIe |
| `cudaMemcpy3D` (`memcpy_3d`) waits that stream; `memcpy_3d_async` bills payload not padding | PCIe / NVLink / HBM |
| `cuMemcpy3DUnaligned` (`memcpy_3d_unaligned`) is identity with `memcpy_3d` | no 3D alignment tax |
| `cudaMemcpyBatchAsync` (`memcpy_batch_async`) 1D only; intra-batch copies share one stream-order snapshot (or empty DuringApiCall/Any deps); same-stream `cudaMallocAsync` needs no host sync | copy-engine occupancy; DuringApiCall waits those copies |
| `cudaMemcpyWithAttributesAsync` (`memcpy_with_attributes`) Stream is `memcpy`; DuringApiCall/Any are a one-copy batch | `PreferOverlapWithCompute` ignored (discrete) |
| `cudaMemcpy3DBatchAsync` (`memcpy_3d_batch_async`) 3D pointer-to-pointer; `flags` 0; CUDA arrays not modeled | same sibling copy-engine occupancy as 1D batch |
| `cudaMemcpy3DWithAttributesAsync` (`memcpy_3d_with_attributes`) Stream is `memcpy_3d_async` | reserved `flags` must be 0 |
| `cudaMemset2D` (`memset_2d`) waits that stream; `memset_2d_async` bills payload not padding | HBM write |
| `cudaMemset3D` (`memset_3d`) waits that stream; `memset_3d_async` bills payload not padding | HBM write |
| `synchronize_device` waits one GPU | other GPUs keep running |
| stream order, event dependencies | memcpy microseconds |
| residency: a kernel may only read **device**, **mapped-host**, VMM peer `va_set_access` (reads) / `va_set_access_write` (read/write), or mempool peer `pool_set_access_read` (reads) / `pool_set_access` (read/write) allocations; managed first-touch at kernel start | PCIe / NVLink / HBM bandwidth |
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
| `cudaGraphKernelNodeSetParams` / `MemcpyNodeSetParams` / `MemcpyNodeSetParams1D` / `MemsetNodeSetParams` / `HostNodeSetParams` patch the graph, not an already-instantiated exec | 1 ns host-sync |
| `cudaGraph*NodeGetParams` reads the definition; `graph_exec_*_get_params` reads the exec snapshot | query |
| graph update replaces the exec snapshot when topology matches (device, stream, kind, deps, edge ports); UseNodePriority forbids kernel-node priority changes; 2D/3D memset may change address only; memcpy memory type cannot change; kernel function variant cannot change (`FunctionChanged`; work sizes may); IF / WHILE / SWITCH handle may change (bodies stay topology); mem nodes are Invalid; `update_graph_with_info` fills `cudaGraphExecUpdateResultInfo` | `graph_update_ns` |
| `cudaGraphExecKernelNodeSetParams` patches one instantiated kernel node's pointers / kind (mem nodes legal; device-updatable nodes keep the exec uploaded) | `graph_set_params_ns` |
| `cudaGraphExecMemcpyNodeSetParams` / `SetParams1D` patches one instantiated memcpy node's `MemcpyOp` (mem nodes legal; 1D may convert 2D/3D) | `graph_set_params_ns` |
| `cudaGraphExecMemsetNodeSetParams` patches one instantiated memset node's dest span (mem nodes legal) | `graph_set_params_ns` |
| `cudaGraphExecHostNodeSetParams` patches one instantiated host node's `fn_id` / `userData` (mem nodes legal) | `graph_set_params_ns` |
| `cudaGraphNodeSetEnabled` skips an instantiated node at launch (mem nodes illegal) | `graph_set_params_ns` |
| `cudaGraphExecChildGraphNodeSetParams` swaps one instantiated child-graph node's nested graph (nested topology must match; child ids are topology for ExecUpdate) | `graph_set_params_ns` |
| `cudaGraphExecEventRecordNodeSetEvent` / `WaitNodeSetEvent` retarget the event on an instantiated record/wait node (External flag is topology) | `graph_set_params_ns` |
| `cudaGraphConditionalNodeSetParams` / `ExecConditionalNodeSetParams` retarget the IF / IfElse / WHILE / SWITCH handle (type, size, bodies stay topology) | 1 ns / `graph_set_params_ns` |
| graph mem alloc/free nodes (`cudaMallocAsync` / `cudaFreeAsync` during capture, or `graph_add_alloc` / `graph_add_free`) | `pool_reuse_ns` on relaunch without free |
| graph clone is an independent uninstantiated copy; child graphs cloned recursively; mem alloc nodes get new ids | `graph_clone_ns` |
| `cudaGraphCreate` (`create_graph` / `create_graph_with_flags`) is an empty uninstantiated graph; flags 0 | 1 ns host-sync |
| `cudaGraphConditionalHandleCreate` (`graph_conditional_create` / `with_flags` / `with_ctx`) | 1 ns host-sync; `ASSIGN_DEFAULT` resets each launch; flags 0 persists; ctx must match the node |
| `cudaStreamBeginCaptureToGraph` (`begin_capture_to_graph`) appends captured nodes onto an existing uninstantiated graph; empty deps are extra roots | not timed (capture) |
| `cudaStreamBeginRecaptureToGraph` (`begin_recapture_to_graph`) updates an existing graph in place; topology/alloc-free fail immediately; other params update unless the callback fails; failure is `"undefined graph"` | not timed (capture) |
| `cudaStreamGetCaptureInfo_v3` (`StreamCaptureInfo::edge_data`) | Default `GraphEdgeData` per capture dep; query |
| `cudaGraphGetNodes` / `GetRootNodes` / `GetEdges` / `NodeGetDependentNodes` | query |
| `cudaGraphNodeGetDependencies` / `GetDependentNodes` v2 | Default `GraphEdgeData`; query |
| `cudaGraphGetId` (`graph_get_id`) | unique id matching debug-dot HANDLES; exec/clone differ |
| `cuGraphNodeGetLocalId` (`graph_node_get_local_id`) | live node id matching debug-dot `n0`; parked exec is unknown |
| `cuGraphNodeGetToolsId` (`graph_node_get_tools_id`) | unique tools id; parked exec is unknown |
| `cuGraphNodeGetContainingGraph` (`graph_node_get_containing_graph`) | owning graph; child-graph node stays parent; parked exec is unknown |
| `cudaGraphRemoveDependencies` v2 (`graph_remove_dependencies_n_with_data`) matching `(from, to, data)`; missing matching edge is Invalid; v1 missing is a no-op | not timed (host-side topology) |
| `cudaGraphDebugDotFlagsRuntimeTypes` | runtime `cudaGraphNodeType*` names; flags `0` stays Debug |
| `cudaGraphDebugDotFlagsExtraTopoInfo` | numbers existing edges; launch-completion dumps `from_port=2` |
| `cudaGraphChildGraphNodeGetGraph` / `EventRecordNodeGetEvent` / `WaitNodeGetEvent` / `MemAllocNodeGetParams` | query |
| `cudaGraphAddKernelNode` / memcpy / `AddMemcpyNode1D` / `graph_add_memcpy_2d` / `graph_add_memcpy_3d` / `graph_add_memset_2d` / `graph_add_memset_3d` / memset / host / empty / event / child / mem alloc/free / cooperative kernel / dependencies (`graph_add_*`) | not timed (host-side topology) |
| `cudaGraphDestroyNode` (`graph_destroy_node`) drops a definition node and incident edges; remaining indices stay valid | not timed (host-side topology) |
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
| `event_get_flags` is the create flags word | `cudaEventGetFlags` |
| `event_get_id` is unique per event handle (`EventId + 1`) | `cuEventGetId` |
| `pool_get_id` is unique per pool handle (`PoolId + 1`); graph-memory pools are legal | `cuMemPoolGetId` |
| `query_stream` is non-blocking | `cudaStreamQuery` |
| `mem_info` is `(free, total)` HBM | `cudaMemGetInfo` |
| `pointer_get_attributes` classifies Unregistered / Host / Device / Managed | `cudaPointerGetAttributes` |
| `pointer_get_attribute` wraps type / mapped / pool / range / ordinal / start / buffer id / IPC / RDMA / handle types / VMM map / hw decompress 0 / VMM block id; SyncMemops is settable | `cuPointerGetAttribute` / `SetAttribute` |
| `pointer_get_access_flags` is kernel residency on an explicit device (`MemAccessFlags`; enable_peer is D2D memcpy only) | `CU_POINTER_ATTRIBUTE_ACCESS_FLAGS` |
| `pointer_get_attribute_n` is batch `cuPointerGetAttribute` (distinct from the `cudaPointerGetAttributes` struct) | `cuPointerGetAttributes` |
| `host_get_device_pointer` of mapped host returns the same id (`flags` must be 0) | `cudaHostGetDevicePointer` |
| `host_get_flags` returns the stored `HostAllocFlags` word | `cudaHostGetFlags` |
| `alloc_host_with_flags` / `host_register_with_flags` (MAPPED / PORTABLE / WRITE_COMBINED stored; IoMemory / ReadOnly Invalid) | `cudaHostAlloc` / `cudaHostRegister` |
| `device_get_attribute` exposes modeled SKU caps (incl. ComputeCapabilityMajor/Minor Hopper 9.0 on example H100; MaxThreadsPerBlock 1024 and H100 block/grid dims; MaxRegistersPerBlock 65536; MaxSharedMemoryPerMultiprocessor matches optin; GlobalMemoryBusWidth 5120 bits on example H100 / 6144 on example H200; SingleToDoublePrecisionPerfRatio 1 on example H100; linear texture 1D/2D dims always 0; texture 2D gather dims always 0; mipmapped texture 1D/2D dims always 0; cubemap texture width always 0; layered texture 1D/2D dims always 0; cubemap layered texture dims always 0; layered surface 1D/2D dims always 0; cubemap surface dims always 0; texture 3D alt dims always 0; GPUDirect RDMA / CanFlushRemoteWrites / FlushWritesOptions Host / WritesOrdering None / WithCudaVMM from `LinkKind::Rdma`; MulticastSupported from `LinkKind::Nvlink`; ConcurrentManaged / DirectManagedHost / PageableHostPageTables / HostNativeAtomic / OnlyPartialHostNativeAtomic / CooperativeMultiDevice / Integrated / GenericCompression / Win32 / Win32Kmt / Fabric handle types always 0; D3D12CigSupported always 0; VulkanCigSupported always 0; ComputeMode always Default; MpsEnabled always 0; NumaConfig always None; GpuPciDeviceId always 0; GpuPciSubsystemId always 0; TccDriver / KernelExecTimeout / TensorMapAccessSupported / UnifiedFunctionPointers / TimelineSemaphoreInteropSupported / MemDecompressAlgorithmMask / MemDecompressMaximumLength / HostNumaVirtualMemoryManagementSupported / HostNumaMemoryPoolsSupported / HostNumaMultinodeIpcSupported always 0; CanUse64BitStreamMemOps / CanUseStreamMemOps / CanUseStreamWaitValueNor always 1) | `cudaDeviceGetAttribute` |
| `device_get_exec_affinity_support` `SM_COUNT` is 0 (permille green ctx, not occupancy SM counts) | `cuDeviceGetExecAffinitySupport` |
| `device_get_properties` wraps the same SKU caps (incl. synthetic `uuid` and PCI ids) | `cudaGetDeviceProperties` |
| `device_compute_capability` is Hopper 9.0 on example H100 | `cuDeviceComputeCapability` |
| `device_get_uuid` is a synthetic 16-octet id (also `DeviceProperties.uuid`) | `cuDeviceGetUuid` |
| `device_get_by_uuid` is the inverse of `device_get_uuid` | `cuDeviceGetByUuid` |
| `device_get_luid` is always-zero LUID plus node mask (also `DeviceProperties.luid`) | `cuDeviceGetLuid` |
| `device_get_texture_1d_linear_max_width` is always 0 (CUDA linear textures are not modeled) | `cuDeviceGetTexture1DLinearMaxWidth` |
| `device_get_pci_bus_id` is a synthetic `domain:bus:device.function` (also `DeviceProperties` PCI ids) | `cudaDeviceGetPciBusId` |
| `device_get_by_pci_bus_id` is the inverse of `device_get_pci_bus_id` | `cudaDeviceGetByPCIBusId` |
| `stream_get_flags` is 0 blocking / 1 NonBlocking | `cudaStreamGetFlags` |
| `stream_get_priority` is the create priority | `cudaStreamGetPriority` |
| `stream_get_id` is unique per device/stream | `cudaStreamGetId` |
| `stream_get_device` is the device of the stream (green-ctx streams return the ctx device) | `cudaStreamGetDevice` / `cuStreamGetDevice` |
| `stream_get_attribute` / `stream_set_attribute` wrap existing stream state | `cudaStreamGetAttribute` / `SetAttribute` |
| `device_count` is the profile GPU count | `cudaGetDeviceCount` |
| `driver_get_version` / `runtime_get_version` report CUDA 13.0 | `cudaDriverGetVersion` / `cudaRuntimeGetVersion` |
| `driver_init` is a 1 ns no-op; flags must be 0 | `cuInit` |
| `device_get` is the ordinal in `0 .. count` | `cuDeviceGet` |
| `flush_gpu_direct_rdma_writes` is a 1 ns host-sync barrier on RDMA SKUs (no write-visibility) | 1 ns |
| `BatchMemOp::FlushRemoteWrites` is stream-ordered `CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES` (capture legal; never a no-op) | 1 ns Solo |
| `set_limit` / `get_limit` wrap persisting L2 plus stack / printf / heap / CDP / L2 fetch; `DevRuntimePendingLaunchCount` caps in-flight `device_launch_graph` | `cudaDeviceSetLimit` / `GetLimit` |
| `set_shared_mem_config` / `get_shared_mem_config`; Default kernels inherit function then device | `cudaDeviceSetSharedMemConfig` / `GetSharedMemConfig` |
| `set_func_shared_mem_config` / `get_func_shared_mem_config`; per-device function config | `cudaFuncSetSharedMemConfig` / `GetSharedMemConfig` |
| `set_device_flags` / `get_device_flags` schedule + MapHost / Lmem / SyncMemops; Auto streams inherit the tax | `cudaSetDeviceFlags` / `GetDeviceFlags` |
| `init_device` / `init_device_with_flags` seed is already done; `FLAGS_ARE_VALID` applies `deviceFlags` | `cudaInitDevice` |
| `reset_device` waits that GPU then frees `cudaMalloc` (not `cudaMallocAsync`); user streams except NULL are unknown until create | `cudaDeviceReset` |
| `SimError::error_name` / `error_string` map a returned error (no thread-local last error) | `cudaGetErrorName` / `cudaGetErrorString` |
| `device_primary_ctx_get_state` is flags plus always-active (no lazy retain) | `cuDevicePrimaryCtxGetState` |
| `device_primary_ctx_set_flags` is always Invalid (primary ctx already seeded) | `cuDevicePrimaryCtxSetFlags` |
| `ctx_get_id` is `cuCtxGetId` for the seeded primary context of an explicit device | query; legal during capture |
| `ctx_get_api_version` is CUDA 13.0 for that primary context | `cuCtxGetApiVersion` |
| `ctx_get_flags` wraps `get_device_flags` for that primary context | `cuCtxGetFlags` |
| `ctx_get_cache_config` wraps `get_cache_config` for that primary context | `cuCtxGetCacheConfig` |
| `ctx_get_stream_priority_range` wraps `device_get_stream_priority_range` (H100 `(0, -5)`) | `cuCtxGetStreamPriorityRange` |
| `ctx_get_limit` wraps `get_limit` for a `DeviceLimit` | `cuCtxGetLimit` |
| `ctx_synchronize` waits every stream on one GPU (other GPUs keep running) | `cuCtxSynchronize` |
| `ctx_get_shared_mem_config` wraps `get_shared_mem_config` for that primary context | `cuCtxGetSharedMemConfig` |
| `func_get_name` is empty until a compiled kernel exists | `cudaFuncGetName` / `cuFuncGetName` |
| `func_get_param_info` is Invalid until a compiled kernel exists | `cuFuncGetParamInfo` |
| access-policy windows align to `cudaLimitMaxL2FetchGranularity` (default 128; `expertvm sim --l2-fetch N` sets 32/64/128) | exact |
| `AccessPolicyWindow.hit_ratio_permille` is CUDA `hitRatio` (`expertvm sim --l2-ratio N`; unset 1000) | exact |
| `AccessPolicyWindow.hit` Streaming skips persist fill (`expertvm sim --l2-streaming`; needs persist) | exact |
| `malloc_pitch` charges `pitch * height`; pitch is `align_up(width, 512)` | `cudaMallocPitch` |
| `malloc_pitch_with_element_size` is `cuMemAllocPitch`; `ElementSizeBytes` 4/8/16; pitch still 512-aligned | `cuMemAllocPitch` |
| `MemcpyOp` height/pitches bill `width * height` (not pitch padding); origin is srcPos / dstPos; LOD must be 0 | `cudaMemcpy2DAsync` / `CUDA_MEMCPY3D` |
| `MemsetOp` height/pitch bill `width * height` (not pitch padding) | `cudaMemset2DAsync` |
| `MemsetOp` `element_size` is 1/2/4 (`cudaMemset` stays 1; offset/width/pitch must divide it) | `cudaMemsetNodeParams::elementSize` |
| `malloc_3d` charges `pitch * height * depth` | `cudaMalloc3D` |
| `MemcpyOp` depth/slice heights bill `width * height * depth` | `cudaMemcpy3DAsync` |
| `MemsetOp` depth/ysize bill `width * height * depth` | `cudaMemset3DAsync` |
| graph-mem used is live graph allocs; reserved holds unused until trim | `cudaDeviceGetGraphMemAttribute` / `GraphMemTrim` |
| stream[i+1].start ≥ stream[i].finish (`Operation` timestamps) | queue wait vs run |
| numerically lower `set_stream_priority` starts first under contention (`cudaDeviceGetStreamPriorityRange`) | launch overhead |
| `cudaGetCurrentGraphExec` (`current_graph_exec`) is the DeviceLaunch exec in flight; host `launch_graph` does not count | query |
| `cudaStreamGraphFireAndForget` / `cudaStreamGraphTailLaunch` (`GRAPH_FIRE_AND_FORGET` / `GRAPH_TAIL_LAUNCH`) nest `device_launch_graph` while a host-issued DeviceLaunch is in flight; host `launch_graph` cannot use those ids | `graph_launch_ns` |
| `cudaStreamGraphFireAndForgetAsSibling` (`GRAPH_FIRE_AND_FORGET_AS_SIBLING`) nested DeviceLaunch does not keep the parent instance alive | `graph_launch_ns` |
| `cudaStreamDestroy` (`destroy_stream`) returns immediately; in-flight work still completes; NULL is Invalid; recreate while unfinished is `"stream in flight"` | 1 ns |
| `compute_slots>=2` overlaps independent kernels at full issue rate | kernel duration (not SM-partition) |
| `cudaLaunchCooperativeKernel` occupies every compute slot (`cooperative_kernel`) | leftover kernels cannot Hyper-Q overlap |
| `cudaLaunchAttributeCooperative` (`KernelNodeAttr::Cooperative`) | same occupancy; `graph_exec_kernel_set_params` still refuses a mismatch |
| programmatic dependent launch: wait kernel starts after the previous same-stream trigger (`pdl_trigger_permille`) | overlap needs `compute_slots >= 2` |
| `cudaLaunchAttributeAccessPolicyWindow` persisting hits (`kernel_access_policy`) | HBM discount after `set_persisting_l2_cache_size`; CUDA default size is 0 |
| `cudaStreamAttributeAccessPolicyWindow` inherited by `kernel` / `kernel_bufs` | same persist billing as `kernel_access_policy`; `kernel_with` / graph replay use launch / node |
| `cudaLaunchAttributeMemSyncDomain` fence isolation (`kernel_with` / allreduce Remote) | `same_domain_fence_permille` of leftover same-domain traffic; tax default 0 |
| `cudaLaunchAttributeMemSyncDomainMap` (`set_stream_mem_sync_domain_map`) | Hopper identity remote→1; collapse `{default:0, remote:0}` restores same-domain fence; `expertvm sim --mem-sync-map collapse` (needs `--mem-sync-domain remote`) |
| `cudaLaunchAttributeMemSyncDomain` launch Remote (`kernel_with` / graph node) | leftover prefill inherit-Default joins decode Remote; restores same-domain fence; `expertvm sim --mem-sync-launch` (needs `--mem-sync-domain remote`) |
| `cudaLaunchAttributeMemSyncDomainMap` launch collapse (`kernel_with` / graph node) | leftover prefill inherit-identity maps Default→0 with decode Remote→0; restores same-domain fence; `expertvm sim --mem-sync-launch-map` (needs `--mem-sync-domain remote`) |
| `cudaLaunchAttributeClusterDimension` (`kernel_with` cluster) | occupies `min(blocks, compute_slots)`; Hopper portable max 8 |
| `cudaLaunchAttributeClusterSchedulingPolicyPreference` Spread | occupies every Hyper-Q slot; Default uses `set_func_cluster_policy` (`cudaFuncAttributeClusterSchedulingPolicyPreference`; `expertvm sim --func-cluster-spread` sets Spread; unset never occupies extra slots without `--cluster` >= 2); LoadBalancing matches Default occupancy and overrides function Spread (`expertvm sim --cluster-load-balance`, needs `--func-cluster-spread`) |
| `cudaLaunchAttributePreferredClusterDimension` | occupies preferred size when it fits in `compute_slots` |
| `cudaFuncAttributeNonPortableClusterSizeAllowed` | sizes above `portable_cluster_size` until the SKU `max_blocks_per_cluster` |
| `cudaFuncAttributeClusterDimMustBeSet` / `RequiredClusterWidth` / Height / Depth | no cluster is Invalid; a nonzero required axis must match the launch; `expertvm sim --cluster-must-set` sets ClusterDimMustBeSet (needs `--cluster`; occupancy matches `--cluster`); `expertvm sim --required-cluster N` sets RequiredClusterWidth (needs `--cluster`; must match; occupancy matches `--cluster`) |
| `cudaLaunchAttributeSynchronizationPolicy` (stream plus graph kernel nodes) | host-wait tax on `synchronize_stream` / `synchronize_event`; kernel-node non-Auto taxes that node's launch-completion / programmatic event; Auto inherits stream / `set_device_flags` (unset / profile default 0) |
| `cudaLaunchAttributeDeviceUpdatableKernelNode` | graphs-only; `graph_exec_kernel_set_params` keeps the exec uploaded; device-launch graphs allow it |
| `cudaLaunchAttributeProgrammaticEvent` / `LaunchCompletionEvent` flags | `cudaEventRecordExternal` is Invalid; interprocess / IPC-imported events are Invalid |
| `cudaLaunchAttributePreferredSharedMemoryCarveout` | MaxShared occupies every Hyper-Q slot; Default uses `set_func_carveout` (`cudaFuncAttributePreferredSharedMemoryCarveout`; `expertvm sim --func-max-shared` sets MaxShared; unset never occupies); MaxL1 matches Default occupancy and overrides function MaxShared (`expertvm sim --max-l1`, needs `--func-max-shared`) |
| `cudaLaunchAttributeSharedMemoryMode` | Default uses function `set_func_shared_mem_config` then device `set_shared_mem_config` (unset never scales); FourByte / EightByte scale duration by `1000 / shared_mem_*_permille` (default 1000) |
| `cudaLaunchAttributePortableClusterSizeMode` | Default uses the function attr; RequirePortable always refuses oversize; AllowNonPortable allows up to SKU max |
| CUDA 13 `cudaLaunchAttributeSharedMemoryMode` (`PortableSharedMode`) | Default uses `MaxDynamicSharedMemorySize`; RequirePortable refuses oversize; AllowNonPortable allows up to opt-in max |
| `cudaLaunchKernel` `sharedMemBytes` | `0` is identity; above `max_shared_mem_per_block` needs the function attr or AllowNonPortable |
| `cudaLaunchAttributeNvlinkUtilCentricScheduling` | hint `0`/`1`; occupies every Hyper-Q slot when the profile has NVLink |
| `cudaLaunchAttributePriority` (`kernel_with` / `KernelAttrs::priority`) | `None` inherits stream create priority; `Some` overrides that kernel; numerically lower starts first under contention; stream Get/SetPriority clamp to `device_get_stream_priority_range` |
| `set_stream_sm_permille` is a duration-only SM fraction (‰) | compute-bound kernels scale; memory-bound keep full HBM; does not partition Hyper-Q |
| CUDA green contexts (`cuGreenCtxCreate`) | complementary SM spans may overlap kernels even when `compute_slots` is 1; same-span contexts share exclusive compute |
| `cuDevSmResourceSplit` (`dev_sm_resource_split`) | explicit ‰ groups; `smCount` 0 is remaining; coschedule counts / BACKFILL stay Invalid |
| `cuGreenCtxRecordEvent` / `cuGreenCtxWaitEvent` | record joins every bound stream; wait holds later work on the ctx (including streams bound after the wait); not a per-stream record/wait |
| `cudaExecutionCtxSynchronize` (`green_ctx_synchronize`) | CPU waits that green ctx; other ctxs on the same GPU keep running |
| `cuStreamGetDevResource` (`stream_get_dev_resource`) | bound stream returns that ctx's SM span; unbound is a full chip; query during capture |
| `cuGreenCtxGetId` (`green_ctx_get_id`) | unique id for a live green ctx; not `GreenCtxId` / `stream_get_id`; query during capture |
| `cudaExecutionCtxGetDevice` (`green_ctx_get_device`) | device passed to `green_ctx_create`; query during capture |
| `CUDA_KERNEL_NODE_PARAMS.ctx` (`KernelNodeParams::ctx`) | pins graph kernel duration plus SM occupancy; `None` inherits the launch stream |
| `CUDA_KERNEL_NODE_PARAMS.sharedMemBytes` (`KernelNodeParams::shared_mem_bytes`) | graph kernel dynamic shared; typed `graph_add_kernel` stays 0; CopyAttributes does not copy it |
| `CUDA_BATCH_MEM_OP_NODE_PARAMS.ctx` (`BatchMemOpNodeParams::ctx`) | graph batch-mem-op green ctx; wait/write/flush duration unchanged; typed `graph_add_batch_mem_op` stays `None` |
| `CUDA_CONDITIONAL_NODE_PARAMS.ctx` (`GraphNodeParams::If` ctx) | must match handle create ctx; conditionals do not occupy SMs; typed `graph_add_if` copies the handle |
| `cuGraphAddMemcpyNode` ctx (`MemcpyNodeParams::ctx`) | graph memcpy green ctx; copy-engine duration unchanged; typed `graph_add_memcpy` stays `None` |
| `cuGraphAddMemsetNode` ctx (`MemsetNodeParams::ctx`) | graph memset green ctx; copy-engine duration unchanged; typed `graph_add_memset` stays `None` |
| `CUgraphChildGraphNodeOwnership` (`ChildGraphNodeParams::ownership`) | clone is instantiated child without mem/conditionals; move owns an uninstantiated child that may have mem nodes; parent AutoFreeOnLaunch is inherited by MOVE children |
| `memset` / `memset_buf` needs the filled span resident (not mapped host); `memset_op` height/pitch is 2D | HBM write of payload + launch overhead |
| `cudaMemset` / `2D` / `3D` (`memset_sync` / `memset_op_sync`) wait the stream | host-synchronous; capture refused |
| peer D2D needs topology + `enable_peer` (`enable_peer_with_flags` must be 0) | link bandwidth |
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
`set_stream_sm_permille` is duration-only: compute-bound
kernels scale as `1000 / permille`; memory-bound keep full HBM. Default
unset is a full chip (`1000`). It does not partition Hyper-Q occupancy.
CUDA green contexts (`cuDeviceGetDevResource` / `cuDevSmResourceSplit` /
`cuDevSmResourceSplitByCount` /
`cuDevResourceGenerateDesc` / `cuGreenCtxCreate`) split the chip in ‰ (not
occupancy SM counts). `cuDevSmResourceSplit` takes explicit ‰ group sizes
(`smCount` `0` is remaining). Coschedule counts must be 0. Complementary spans may overlap kernels even when
`compute_slots` is 1; same-span contexts still share exclusive compute.
`cuGreenCtxRecordEvent` joins work already submitted on every bound stream;
`cuGreenCtxWaitEvent` holds later submits on that ctx (including streams
bound after the wait). Distinct from `cudaEventRecord` / `cudaStreamWaitEvent`.
`cudaExecutionCtxSynchronize` (`green_ctx_synchronize`) waits one green ctx;
other ctxs on the same GPU keep running. Distinct from `cudaDeviceSynchronize`.
`cuGreenCtxGetId` (`green_ctx_get_id`) is a unique id for a live green ctx
(`cudaExecutionCtxGetId`). Distinct from `GreenCtxId` and `stream_get_id`.
Query; legal during capture.
`cudaExecutionCtxGetDevice` (`green_ctx_get_device`) returns the create
device. Distinct from `green_ctx_get_id`. Query; legal during capture.
`CUDA_KERNEL_NODE_PARAMS.ctx` (`KernelNodeParams::ctx`) pins a graph kernel
to a live green context (duration plus SM occupancy). `None` inherits the
launch stream. Typed `graph_add_kernel` stays `None`. No Engine `--kernel-ctx`
and no `cuCtxFromGreenCtx`.
`CUDA_KERNEL_NODE_PARAMS.sharedMemBytes` (`KernelNodeParams::shared_mem_bytes`)
is dynamic shared on a graph kernel node. Typed `graph_add_kernel` stays `0`.
Get/SetAttribute stays. CopyAttributes does not copy it. Oversize without
func attr / AllowNonPortable is Invalid. Duration follows bank width, not
byte count. No Engine `--kernel-shared`.
`CUDA_BATCH_MEM_OP_NODE_PARAMS.ctx` (`BatchMemOpNodeParams::ctx`) pins a
graph batch-mem-op node to a live green context. Wait/write/flush do not
occupy SMs, so duration is unchanged. Typed `graph_add_batch_mem_op` stays
`None`. No Engine `--batch-ctx`.
`CUDA_CONDITIONAL_NODE_PARAMS.ctx` (`GraphNodeParams::If` / `IfElse` /
`While` / `Switch` ctx) must match the handle from
`graph_conditional_create_with_ctx`. Typed `graph_add_if` copies the
handle. Mismatch is Invalid `"conditional ctx"`. Conditionals do not
occupy SMs, so duration is unchanged. This VM does not invent an Engine
flag for conditional ctx.
`cuGraphAddMemcpyNode` ctx (`MemcpyNodeParams::ctx`) pins a graph memcpy
node to a live green context. Copies use copy engines, so duration is
unchanged. Typed `graph_add_memcpy` stays `None`. Typed
`graph_memcpy_set_params` does not clear ctx. This VM does not invent an
Engine flag for memcpy ctx.
`cuGraphAddMemsetNode` ctx (`MemsetNodeParams::ctx`) pins a graph memset
node to a live green context. Fills use copy engines, so duration is
unchanged. Typed `graph_add_memset` stays `None`. Typed
`graph_memset_set_params` does not clear ctx. This VM does not invent an
Engine flag for memset ctx.
`CUgraphChildGraphNodeOwnership` (`ChildGraphNodeParams::ownership`) is clone
versus move of a child graph. Typed `graph_add_child` stays clone of an
instantiated child without mem or conditional nodes. Move lets a parent own
an uninstantiated child that may contain mem alloc/free. GetParams of a
moved node reports `INVALID`. Instantiating the parent instantiates the
moved child and inherits `AutoFreeOnLaunch`. A parked in-flight-destroyed
exec used as a child-graph handle is `"unknown graph"`; a live exec as
child stays. This VM does not invent an Engine flag for child-graph
ownership.
Copy engines still overlap compute. Profile knobs `gemm_util_permille` (achieved/peak) and `grouped_moe_permille`
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

CUDA graphs: `begin_capture` / `begin_capture_to_graph` /
`begin_recapture_to_graph` / `end_capture` /
`instantiate_graph` /
`update_graph` / `clone_graph` / `destroy_graph` / `launch_graph`. Capture does
not advance the virtual clock. Independent streams stay live. A stream that
`wait_event`s an event recorded in this capture joins (CUDA forked capture);
`record_event_external` / `wait_event_external` (`cudaEventRecordExternal` /
`cudaEventWaitExternal`) do not join, so a live waiter can overlap graph launch.
`record_event_with_flags` / `wait_event_with_flags` / `create_event_with_flags`
are the flags-parameter twins (`EventRecordFlags::EXTERNAL` /
`EventWaitFlags::EXTERNAL` / `EventCreateFlags::DISABLE_TIMING` /
`EventCreateFlags::INTERPROCESS` / `EventCreateFlags::BLOCKING_SYNC`). Unknown bits are Invalid. Interprocess
requires DisableTiming. BlockingSync taxes `synchronize_event` with
`host_sync_blocking_ns`. Typed helpers
stay (`create_event_interprocess` / `create_event_blocking_sync`).
`expertvm store --event-blocking-sync` creates timing copy events with
BlockingSync (implies `--timing-events`; distinct from `--sync-policy blocking`).
`ipc_get_event` / `ipc_open_event` are `cudaIpcGetEventHandle` /
`cudaIpcOpenEventHandle`: the import aliases the source record. Destroy of
the source while imports are live is Invalid. Capture cannot include event IPC.
`launch_graph` remaps origin-stream nodes onto the launch stream so copy and
compute can overlap. Query or `synchronize_stream` of a capturing stream, and
node `synchronize`, are `Invalid`. `launch_graph` during capture records a
child-graph node if the child is already instantiated; parent launch expands
it. Independent streams still launch live. `cudaMallocAsync` / `cudaFreeAsync`
(`alloc` / `free`) during capture are graph mem alloc/free nodes.
`graph_add_alloc` / `graph_add_free` are `cudaGraphAddMemAllocNode` /
`cudaGraphAddMemFreeNode` (same reuse / AutoFreeOnLaunch rules).
`graph_add_alloc_with_access` is `accessDescs` (empty is `graph_add_alloc`;
peer PROT_READ / PROT_READ_WRITE without dest HBM after launch).
`GraphNodeParams::Alloc` is the `cudaGraphAddNode` path (same `accessDescs`;
empty is identity with `graph_add_alloc`). Graph-memory
pool stays Invalid for `pool_set_access`. `graph_alloc_get_access` /
`graph_exec_alloc_get_access` are typed GetParams accessDescs;
`graph_node_get_params` returns them on Alloc. SetParams of Alloc
stays Invalid. No Engine `--graph-alloc-access`.
`graph_add_node` is `cudaGraphAddNode` (`GraphNodeParams` plus dependency
indices in the same call; duplicate `deps` are Invalid `"graph dependency"`).
Typed `graph_add_*` stay (empty deps).
`graph_add_node_with_data` is `cuGraphAddNode_v2` (`dependencyData`; Default
type with ports 0 is identity; length mismatch Invalid).
`GraphKernelNodePort::LAUNCH_COMPLETION` waits for a source kernel to start.
Programmatic type stays Invalid. No Engine `--graph-add-node-data`.
`GraphNodeParams::If` / `IfElse` / `While` / `Switch` fill `GraphAddNode` bodies.
Typed helpers stay (`graph_add_if`, `graph_add_if_else`, `graph_add_while`,
`graph_add_switch`).
`graph_add_set_conditional` is the graph-build analog of captured
`set_conditional` (`GraphNodeParams::SetConditional`; handle is topology,
`value` is a parameter).
`graph_node_set_params` / `graph_exec_node_set_params` are
`cudaGraphNodeSetParams` / `cudaGraphExecNodeSetParams` (typed SetParams;
exec memcpy is 1D-only; exec memset 2D/3D address only; Alloc would resize HBM; Empty has no params).
`graph_node_get_params` / `graph_exec_node_get_params` are
`cudaGraphNodeGetParams` on the definition / exec snapshot (query; Empty
returns `GraphNodeParams::Empty`; Alloc is bytes plus accessDescs;
If, IfElse, and While are handle-only; Switch is handle plus branch count).
`graph_add_empty` is `cudaGraphAddEmptyNode` (1 ns; no compute/copy occupancy).
`graph_add_write_value64` / `graph_add_write_value64_with_flags` /
`graph_add_write_value32_with_flags` / `graph_add_wait_value64` /
`graph_add_wait_value64_with_flags` / `graph_add_wait_value32_with_flags` /
`graph_add_batch_mem_op` are `cudaGraphAddBatchMemOpNode` (`cuStreamWaitValue` /
`WriteValue` / `CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES`; a multi-item batch is
**one** node holding the vector; wait flags are `WaitValueFlags` (`FLUSH`
is a stream-ordered RDMA flush); write flags are `WriteValueFlags`
(NO_MEMORY_BARRIER Invalid).
Typed helpers stay.
`batch_mem_op` is live `cuStreamBatchMemOp`.
`batch_mem_op_with_flags` / `graph_add_batch_mem_op_with_flags` are the
CUDA flags word (`BatchMemOpFlags`; must be 0). Typed helpers stay.
`graph_add_dependencies` is `cudaGraphAddDependencies` (independent nodes
may Hyper-Q overlap at launch; capture records same-stream edges).
`expertvm --graph-build-deps` adds those edges. `graph_add_host_func` is
`cudaGraphAddHostNode`. `expertvm --graph-host` inserts host nodes BETWEEN
`--graph-build` combo children (serialize through `host_func_ns`; not a JOIN
after overlap).
`expertvm --graph-memset` inserts `graph_add_memset` BETWEEN `--graph-mem`
scratch alloc and the GEMM kernel (HBM-write tax; needs `--graph-mem`).
`expertvm --graph-memcpy` inserts `graph_add_memcpy_1d` BETWEEN `--graph-mem`
scratch alloc and the GEMM kernel (H2D copy-engine PCIe tax; needs
`--graph-mem`; legal with `--graph-memset`).
`graph_add_dependencies_n` / `graph_remove_dependencies_n` are the same
APIs with `numDependencies` from/to pairs (all-or-nothing).
`graph_add_dependencies_n_with_data` is `cudaGraphAddDependencies` v2
(`GraphEdgeData`; Default type with ports 0 is identity; Programmatic type
is Invalid; `GraphKernelNodePort::LAUNCH_COMPLETION` waits for a source
kernel to start). An existing `(from, to)` cannot change stored
`GraphEdgeData` (`"graph edge data"`). Incoming Default stays a no-op
(PLAN 182). `graph_edges_with_data` is `cudaGraphGetEdges` v2 (stored
edge data; Default ports 0 when unset). `graph_node_deps_with_data` /
`graph_node_dependents_with_data` are `cudaGraphNodeGetDependencies` /
`GetDependentNodes` v2 (stored edge data). Query;
legal during capture. A parked in-flight-destroyed exec is
`"unknown graph"` on GetDependencies plus GetDependentNodes; a live exec
stays. CUDA v1 `graph_edges` / `graph_node_deps` /
`graph_node_dependents` (`edgeData` NULL) are Invalid `"lossy query"`
when any reported edge has non-default stored `GraphEdgeData`
(`cudaErrorLossyQuery`). Default-only edges stay. Debug-dot ExtraTopoInfo
still dumps ports (not a GetEdges query).
`graph_remove_dependencies` is `cudaGraphRemoveDependencies` (illegal on an
exec and during capture). `graph_remove_dependencies_n_with_data` is
`cudaGraphRemoveDependencies` v2 (`GraphEdgeData`; a matching
`(from, to, data)` is removed; a missing matching edge is Invalid
`"graph dependency"`). Distinct from v1 missing-remove no-op (PLAN 182).
v1 still removes a launch-completion edge (it ignores stored data).
`graph_destroy_node` is `cudaGraphDestroyNode`
(incident edges dropped; remaining indices stay valid; illegal on an exec
and during capture; definition destroy does not retarget exec).
`begin_capture_to_graph` is
`cudaStreamBeginCaptureToGraph`: append captured nodes onto an existing
uninstantiated graph; capture roots additionally depend on the given node
indices (empty `deps` means extra roots, so they may Hyper-Q overlap).
A parked in-flight-destroyed exec is `"unknown graph"` first; a live exec
stays `"graph instantiated"`. Capture-to-graph of the definition stays.
`begin_recapture_to_graph` is `cudaStreamBeginRecaptureToGraph`: recapture
into an existing graph in place (not append). Topology and alloc/free
mismatches fail immediately (`"graph recapture"` / `"graph recapture alloc"`).
Other node parameter mismatches update the original node unless a callback
returns failure. A `None` callback still applies those updates. Failure
leaves the graph undefined (`"undefined graph"`); `destroy_graph` stays
legal. User objects on the graph are released before recapture. Matching
recapture `cudaMallocAsync` returns the existing graph-mem pointer. A
parked exec is `"unknown graph"`; a live exec stays `"graph instantiated"`.
`graph_nodes` / `graph_root_nodes` / `graph_edges` / `graph_node_dependents` /
`graph_debug_dot` / `graph_debug_dot_with_flags` / `graph_get_id` / `graph_node_get_local_id` / `graph_node_get_tools_id` are `cudaGraphGetNodes` /
`GetRootNodes` / `GetEdges` /
`NodeGetDependentNodes` plus `cudaGraphDebugDotPrint` / `cudaGraphGetId` / `cuGraphNodeGetLocalId` plus `cuGraphNodeGetToolsId` (live nodes; flags `0`
is kinds and edges; `GraphDebugDotFlags::RUNTIME_TYPES` prints
`cudaGraphNodeType*` names; `GraphDebugDotFlags::EXTRA_TOPO_INFO` numbers
existing edges; `GraphDebugDotFlags::VERBOSE` prints modeled params and ExtraTopoInfo). A parked in-flight-destroyed exec is `"unknown graph"` on GetLocalId plus GetToolsId; a live exec stays. `graph_node_get_containing_graph` is `cuGraphNodeGetContainingGraph` (the graph that owns the node). A child-graph node still lives in the parent; the nested graph is `graph_child_get_graph`. A parked in-flight-destroyed exec is `"unknown graph"`; a live exec stays. Host-sync
`malloc` / `free_sync` / `memcpy_sync` / `synchronize_device` / VMM / mempool
create cannot be captured. A graph that allocates without a matching free
reuses the pointer on later launches (no second HBM charge) unless
`instantiate_graph_auto_free` (`cudaGraphInstantiateFlagAutoFreeOnLaunch`)
stream-ordered-frees those allocs before relaunch. `clone_graph`
forks those ids. `destroy_graph` refunds remaining graph mem. `update_graph`
of mem nodes is `Invalid`.
Instantiate, update, and upload are host-synchronous and cannot run during capture.
`instantiate_graph_with_flags` is `cudaGraphInstantiateWithFlags`
(`GraphInstantiateFlags::UPLOAD` host-sync uploads during instantiate;
`USE_NODE_PRIORITY` schedules recorded kernels with the add/capture
priority; `DEVICE_LAUNCH` enables `device_launch_graph` after upload —
host `launch_graph` stays legal. The graph cannot be empty and must
contain at least one kernel, memcpy, or memset node (`"device launch
empty"`). Mem alloc/free, events, child graphs, conditionals, host,
empty, and batch-mem nodes are Invalid. Memcpy `Place::Device` must match
the graph origin device (`Place::HostPinned` stays). Memset dest and
kernel buffers must be that device, pinned mapped host, or managed.
Exec memcpy/memset/kernel SetParams re-apply those dest rules. Mixed node green ctx is
`MultipleDevicesNotSupported` (`"graph multiple ctx"`); exec SetParams
re-apply that mixed-ctx rule; exec SetAttribute cannot attach
programmatic or launch-completion events; host updates of an in-flight
device-launch exec are Invalid (`"device launch in flight"`; getters stay;
destroy does not abort the launch; `upload_graph` /
`upload_graph_async` also refuse); `update_graph` of a
device-launch exec is Invalid; cannot combine with `AUTO_FREE_ON_LAUNCH`
(`"device launch auto free"`). `current_graph_exec` is
`cudaGetCurrentGraphExec` (the DeviceLaunch exec in flight on that
device; host `launch_graph` does not count). Query; legal during capture.
`GRAPH_FIRE_AND_FORGET` / `GRAPH_TAIL_LAUNCH` are
`cudaStreamGraphFireAndForget` / `cudaStreamGraphTailLaunch` for nested
`device_launch_graph` while a host-issued DeviceLaunch is in flight. Host
`launch_graph` cannot use those ids. `GRAPH_FIRE_AND_FORGET_AS_SIBLING` is
`cudaStreamGraphFireAndForgetAsSibling` (parent instance does not wait).
Unused conditional handles are
`ConditionalHandleUnused` (`"conditional handle unused"`).
`instantiate_graph_with_params` is `cudaGraphInstantiateWithParams`
(`GraphInstantiateParams` result, err node, and `hUploadStream`).
`graph_exec_get_flags` is `cudaGraphExecGetFlags` (`GraphInstantiateFlags::UPLOAD`
is omitted; it does not affect the executable graph).
Instantiate returns a new exec id (`cudaGraphExec_t`); the source graph
stays a definition. `launch_graph` of a definition uses the primary exec.
`clone_graph` is `cudaGraphClone` (`graph_clone_ns`): an independent
uninstantiated copy; child-graph nodes are cloned recursively (a diamond
of shared children becomes one cloned child). Destroying the original
child still breaks a parent that names it; a recursive clone of that
parent keeps working. `destroy_graph` is `cudaGraphDestroy` / `cudaGraphExecDestroy` (1 ns;
later launch is unknown; remaining graph mem is refunded; user-object
refs held by the graph are released; destroying an in-flight exec does
not abort the launch, whether the work came from `device_launch_graph`
or host `launch_graph` (concurrent host launches all finish);
queries of that exec handle are unknown immediately;
clone and instantiate of that exec handle are unknown immediately;
the definition may still be cloned or instantiated as a new exec;
definition mutators of that exec handle are unknown before
`"graph instantiated"`;
destroying an exec with an in-flight
`upload_graph_async` does not abort the upload).
`user_object_create` is `cudaUserObjectCreate`
(`UserObjectFlags::NO_DESTRUCTOR_SYNC`; initial refs `1..=i32::MAX`; last
ref records `destroy_fn`).
`graph_retain_user_object` / `graph_release_user_object` are
`cudaGraphRetainUserObject` / `ReleaseUserObject` on a definition
(`GraphUserObjectFlags::MOVE` transfers one caller ref). Clone does not
copy retains. Instantiate copies current retains onto the exec. Capture cannot include them. First launch instantiates if needed (`graph_instantiate_ns` once)
then uploads if needed (`graph_upload_ns`). `upload_graph` is `cudaGraphUpload`
(host-sync). `upload_graph_async` is `cudaGraphUpload` on a stream
(Solo `graph_upload_ns`; uploaded when the op completes;
`GraphInstantiateParams::upload_stream` uses it).
`update_graph` copies source steps into the exec snapshot when the
device, stream, op kinds, dependency edges, and `GraphEdgeData` ports match
(`graph_update_ns`; in-flight host launch or stream upload of that exec
is Invalid `"exec in flight"`; SetParams stay);
IF / WHILE / SWITCH handles are parameters (bodies stay topology). A
topology mismatch is `Invalid`. `cudaGraphInstantiateFlagUseNodePriority`
forbids kernel-node priority changes (`AttributesChanged`); matching
priorities still update. Default instantiate copies priority as a
parameter. `graph_exec_kernel_node_set_priority` stays legal. 2D and 3D
memset nodes may change address only (`ParametersChanged` for geometry);
1D memset may change dimensions. `graph_exec_memset_set_params_2d` stays
legal. Memcpy source and destination `Place` (memory type) cannot change;
alloc and size may. `graph_exec_memcpy_set_params` stays legal. CUDA
arrays stay uninvented. Graphs with mem alloc/free nodes cannot be updated.
`update_graph_with_info` is `cudaGraphExecUpdate` with
`cudaGraphExecUpdateResultInfo` (filled even on `Err`: node type, deps,
edge ports, UseNodePriority priority, 2D memset geometry, memcpy memory type, mem nodes, device-launch). A parked in-flight-destroyed exec used as the update source is `"unknown graph"`; a live exec as source stays. `update_graph` uses that path and keeps the
same `why` strings.
`graph_kernel_set_params` / `graph_memcpy_set_params` /
`graph_memcpy_set_params_1d` / `graph_memcpy_set_params_2d` / `graph_memcpy_set_params_3d` / `graph_memset_set_params` / `graph_memset_set_params_2d` / `graph_memset_set_params_3d` /
`graph_batch_mem_op_set_params` /
`graph_batch_mem_ops_set_params` / `graph_event_record_set_event` /
`graph_event_wait_set_event` / `graph_child_set_params` are
`cudaGraphKernelNodeSetParams` /
`MemcpyNodeSetParams` / `MemcpyNodeSetParams1D` / `MemsetNodeSetParams` /
`BatchMemOpNodeSetParams` /
`EventRecordNodeSetEvent` / `EventWaitNodeSetEvent` /
`ChildGraphNodeSetParams` on the graph
definition (do not retarget an already-instantiated exec). A parked
in-flight-destroyed exec is `"unknown graph"` on SetParams; a live exec
stays. Child-graph
definition SetParams may change nested topology; exec SetParams still
require matching topology. Event External flags stay topology.
`graph_*_get_params` / `graph_exec_*_get_params` are
`cudaGraph*NodeGetParams` / `cudaGraphExec*NodeGetParams`
(query; no clock tick; capture is legal). Graph GetParams reads the
definition; Exec GetParams reads the snapshot (`as_exec`; uninstantiated
is Invalid). A parked in-flight-destroyed exec is `"unknown graph"` on
definition GetParams; a live exec stays. Unique-node helpers (`graph_unique_kernel`, …) still use
the launched/primary snapshot.
`graph_exec_kernel_set_params` / `graph_exec_memcpy_set_params` /
`graph_exec_memcpy_set_params_1d` / `graph_exec_memcpy_set_params_2d` / `graph_exec_memcpy_set_params_3d` / `graph_exec_memset_set_params` / `graph_exec_memset_set_params_2d` / `graph_exec_memset_set_params_3d` /
`graph_exec_batch_mem_op_set_params` /
`graph_exec_batch_mem_ops_set_params` are
`cudaGraphExecKernelNodeSetParams` / `cudaGraphExecMemcpyNodeSetParams`
(1D-only: instantiated node and new `MemcpyOp` must be 1D) /
`cudaGraphExecMemcpyNodeSetParams1D` (may convert 2D/3D) / extra 2D/3D
helpers / `cudaGraphExecMemsetNodeSetParams` (2D/3D address only) plus extra
memset 2D/3D helpers plus
`cudaGraphExecBatchMemOpNodeSetParams`
(`graph_set_params_ns`; mem nodes legal; pageable memcpy stays illegal;
device-launch execs re-apply instantiate dest and mixed-ctx rules;
host updates of an in-flight device-launch exec are Invalid
(`"device launch in flight"`);
a `GpuOp::BatchMem` item list is a parameter; kernel SetParams keeps the
exec uploaded when the node is device-updatable).
`graph_node_set_enabled` is `cudaGraphNodeSetEnabled` (skip a node at launch;
mem alloc/free cannot be disabled). `update_graph` plus ExecSetParams leave
enable unchanged.
`graph_exec_child_set_params` is `cudaGraphExecChildGraphNodeSetParams`
(swap the nested graph; nested topology must match; child ids are topology
for `update_graph`; mem nodes legal). Definition-side
`graph_child_set_params` stores the child id as passed (same as
`graph_add_child`). A parked in-flight-destroyed exec used as that child
handle is `"unknown graph"`; a live exec as child stays.
`graph_child_get_graph` is `cudaGraphChildGraphNodeGetGraph`.
`graph_exec_child_get_graph` is the exec-snapshot GetParams twin
(uninstantiated graphs are Invalid).
`graph_event_record_get_event` / `graph_event_wait_get_event` are
`cudaGraphEventRecordNodeGetEvent` / `WaitNodeGetEvent`.
`graph_exec_event_record_get_event` / `graph_exec_event_wait_get_event`
are the exec-snapshot twins (uninstantiated graphs are Invalid).
`graph_alloc_get_params` is `cudaGraphMemAllocNodeGetParams` (id and bytes).
`graph_free_get_params` is `cudaGraphMemFreeNodeGetParams` (stored id).
`graph_exec_alloc_get_params` / `graph_exec_free_get_params` are the
exec-snapshot twins (uninstantiated graphs are Invalid).
`graph_free_set_params` / `graph_exec_free_set_params` are
`cudaGraphMemFreeNodeSetParams` / `cudaGraphExecMemFreeNodeSetParams`
(definition SetParams does not retarget an exec; `graph_allocs` stays
alloc-node ids).
`graph_exec_event_record_set_event` / `graph_exec_event_wait_set_event` are
`cudaGraphExecEventRecordNodeSetEvent` / `WaitNodeSetEvent` (event id is the
parameter; External is topology).
`stream_update_capture_dependencies` is `cudaStreamUpdateCaptureDependencies`
(extra deps for the next captured node, in addition to stream-order; `Set`
replaces, `Add` unions). `stream_is_capturing` / `stream_capture_info` are
`cudaStreamIsCapturing` / `GetCaptureInfo` (includes capture mode).
`StreamCaptureInfo::dependencies` is `GetCaptureInfo_v2` (last same-stream
captured node union extra pending deps). `StreamCaptureInfo::edge_data` is
`GetCaptureInfo_v3` (Default `GraphEdgeData`, ports 0). `StreamCaptureInfo::id` is
`id_out` (unique per begin-capture sequence; forked streams share it).
`begin_capture_with_mode` is `cudaStreamBeginCapture` with
`StreamCaptureMode` (default Relaxed: independent streams stay live; a wait
of a captured record still joins. ThreadLocal/Global refuse uncaptured-stream
submits). `thread_exchange_stream_capture_mode` is
`cudaThreadExchangeStreamCaptureMode`. `graph_node_kind` is
`cudaGraphNodeGetType`.
`graph_conditional_create` / `graph_conditional_create_with_flags` /
`graph_conditional_create_with_ctx` /
`graph_add_if` / `graph_add_if_else` are `cudaGraphConditionalHandleCreate`
and an IF node (`cudaGraphCondTypeIf` size 1 / size 2). Then-body ops skip
at start when the handle is `0`; size 2 runs the else-body instead.
Instantiate requires each handle on a live IF / WHILE / SWITCH
(`ConditionalHandleUnused`). `ASSIGN_DEFAULT` is
identity with the unflagged create (each launch resets to the create-time
default). Flags `0` keeps the handle across launches. `with_ctx` pins
`CUDA_CONDITIONAL_NODE_PARAMS.ctx`; typed create stays `None`. `set_conditional`
is device `cudaGraphSetConditional` (`ASSIGN_DEFAULT` launches reset to
the create-time default).
`graph_add_set_conditional` is the graph-build analog (handle topology;
`value` is `graph_exec_set_conditional_params`).
`expertvm --graph-if` wraps `--graph-build` combo children in
`graph_add_if` + `graph_add_set_conditional` and skips extras with exec
SetParams (clears upload; distinct from `--graph-enable` SetEnabled).
`graph_add_while` / `graph_while_nodes` / `graph_add_switch` /
`graph_switch_nodes` are WHILE / SWITCH (WHILE caps at
64 iterations; SWITCH runs body `i` when the handle equals `i`).
`graph_if_else_nodes` lists size-2 IF else-bodies (`graph_if_nodes` stays
then-body).
`graph_node_set_params` / `graph_exec_node_set_params` retarget the IF /
IfElse / WHILE / SWITCH handle (type, size, and bodies stay topology).
`graph_node_find_in_clone` is `cudaGraphNodeFindInClone` (same index on a
graph produced by `clone_graph` of that original).
`expertvm --graph-set-params` parks a leaf and retargets the unique kernel
(and a unique memcpy or memset if present). `expertvm --graph-update` parks a leaf GEMM on
evict and updates the next miss instead of instantiate. `--graph-clone`
copies the capture (`cudaGraphClone`) before instantiate. `--graph-clone-parent`
clones combo parents (recursive children). `--graph-build` is
`cudaGraphCreate` / `cudaGraphAdd*` (no idle stream; combo children may
Hyper-Q overlap unless `graph_add_dependencies` chains them).
`create_graph_with_flags` is the CUDA flags word (0 only).
`graph_conditional_create_with_flags` is `cudaGraphCondAssignDefault` or
0 (persist). No Engine `--graph-cond-flags`.
`expertvm --graph-build-deps` adds those edges. `--graph-host` inserts
`graph_add_host_func` BETWEEN those children (`host_func_ns`; not a JOIN).
`--graph-if` wraps those children in `graph_add_if` + `graph_add_set_conditional`
(exec SetParams skips extras and re-uploads; not a second SetEnabled).
`--graph-piecewise`
is `cudaStreamBeginCaptureToGraph` combo parents (independent child roots).
`expertvm --graph-capture-deps` chains those fragments (`numDependencies > 0`).
`expertvm --graph-capture-host` inserts captured `host_func` BETWEEN those
fragments (`host_func_ns`; not a JOIN).
`--graph-mem` is in-graph
scratch (`graph_add_alloc` / capture `alloc`). `--graph-memset` memsets that
scratch BETWEEN alloc and GEMM (`graph_add_memset` / capture `memset`; needs
`--graph-mem`). `--graph-memcpy` H2Ds that scratch BETWEEN alloc and GEMM
(`graph_add_memcpy_1d` / capture `memcpy_pinned_to_device`; needs `--graph-mem`;
copy-engine PCIe; legal with `--graph-memset`). `--graph-leaf-host` inserts
`graph_add_host_func` / captured `host_func` BEFORE the leaf GEMM (`host_func_ns`;
not with `--device-launch`). `--graph-auto-free` is
AutoFreeOnLaunch (relaunch recharges HBM; not with `--graph-mem`).
`cooperative_kernel` / `graph_add_cooperative_kernel` are
`cudaLaunchCooperativeKernel` (occupy every Hyper-Q slot; capture allowed).
`graph_kernel_node_set_cooperative` is `cudaLaunchAttributeCooperative` on
graph kernel nodes (`CopyAttributes` copies it). Launch pays `graph_launch_ns`
once; recorded kernels skip per-kernel launch overhead.
`memset` is an HBM-write kernel on a resident alloc. `memset_sync` /
`memset_op_sync` are host-synchronous `cudaMemset` / `2D` / `3D` (capture
refused). Typed `memset` / `memset_op` stay Async. `memset_d16_async` /
`memset_d16` are `cuMemsetD16Async` / `cuMemsetD16` (`count` is CUDA `N`).
`memset_d32_async` / `memset_d32` are `cuMemsetD32Async` / `cuMemsetD32`.
Typed `memset` stays byte-counted. Fill value is not modeled. No Engine
`--memset-d16`. `memset_d2d16_async` / `memset_d2d16` are
`cuMemsetD2D16Async` / `cuMemsetD2D16` (`width` is CUDA `Width`).
`memset_d2d32_async` / `memset_d2d32` are `cuMemsetD2D32Async` /
`cuMemsetD2D32`. `memset_d2d8_async` / `memset_d2d8` are
`cuMemsetD2D8Async` / `cuMemsetD2D8` (`width` is CUDA `Width`).
`memset_2d_async` stays byte-width. No Engine `--memset-d2d`. `memset_2d` /
`memset_2d_async` are `cudaMemset2D` / `cudaMemset2DAsync` (`MemsetOp` must
be 2D). `memset_3d` / `memset_3d_async` are `cudaMemset3D` /
`cudaMemset3DAsync` (`MemsetOp` must be 3D). `host_func` is
`cudaLaunchHostFunc`: stream-ordered host work that does not occupy compute
or copy engines (other streams may GEMM). Unnamed callback by default;
`host_func_params` records `HostNodeParams`. `stream_add_callback` is
`cudaStreamAddCallback` (same enqueue; capture refused).
`stream_add_callback_with_flags` is the CUDA flags word (`StreamCallbackFlags`;
must be 0). `stream_add_callback_params` records `HostNodeParams`.
`write_value64` / `wait_value64`
are `cuStreamWriteValue64` / `WaitValue64` (mailbox; no occupancy).
`batch_mem_op` is `cuStreamBatchMemOp`. Peer D2D requires a
topology link **and** directed `enable_peer` (seeded on for every GPU↔GPU
link; `disable_peer` → `PeerDisabled`). `enable_peer_with_flags` is
`cudaDeviceEnablePeerAccess` (`flags` must be 0). `memcpy_peer` /
`memcpy_peer_async` are `cudaMemcpyPeer` / `cudaMemcpyPeerAsync` (replica
copy; Peer is host-synchronous). `memcpy_peer_3d` / `memcpy_peer_3d_async`
are `cudaMemcpy3DPeer` / `cudaMemcpy3DPeerAsync`
(`MemcpyOp` must be 3D). `memcpy_peer_2d` /
`memcpy_peer_2d_async` are `cudaMemcpy2DPeer` / `cudaMemcpy2DPeerAsync`
(`MemcpyOp` must be 2D). `memcpy_2d` /
`memcpy_2d_async` are `cudaMemcpy2D` / `cudaMemcpy2DAsync` (`MemcpyOp` must
be 2D). `memcpy_2d_unaligned` is `cuMemcpy2DUnaligned` (identity with
`memcpy_2d`; this VM does not require 2D alignment; host-sync; CUDA has no
Async Unaligned). No Engine `--memcpy-unaligned`. `memcpy_htod` /
`memcpy_dtoh` are `cuMemcpyHtoD` / `cuMemcpyDtoH` (host-synchronous pinned;
capture refused). `memcpy_pinned_to_device` / `memcpy_device_to_pinned`
stay Async. No Engine `--memcpy-htod`. `MemcpyOp` `src_x` / `src_y` /
`src_z` / `dst_x` / `dst_y` / `dst_z` are CUDA_MEMCPY2D and CUDA_MEMCPY3D
srcPos / dstPos. Default 0. 1D origin or 2D z origin is `"memcpy origin"`.
No Engine `--memcpy-origin`. `MemcpyOp` `src_lod` / `dst_lod` are
CUDA_MEMCPY3D `srcLOD` / `dstLOD` and must be 0 (CUDA arrays are not
modeled). Nonzero is `"memcpy lod"`. No Engine `--memcpy-lod`. `memcpy_3d` / `memcpy_3d_async` are `cudaMemcpy3D` /
`cudaMemcpy3DAsync` (`MemcpyOp` must be 3D). `memcpy_3d_unaligned` is
`cuMemcpy3DUnaligned` (identity with `memcpy_3d`; this VM does not require
3D alignment; host-sync; CUDA has no Async Unaligned). No Engine
`--memcpy-3d-unaligned`. `memcpy_batch_async` is
`cudaMemcpyBatchAsync` (1D pointer-to-pointer; copies in one batch do not
wait for each other; 2D/3D use `memcpy_3d_batch_async`; capture
cannot include it). `memcpy_with_attributes` is
`cudaMemcpyWithAttributesAsync` (Stream is `memcpy`; DuringApiCall waits
those copies). `expertvm sim --memcpy-during` / `gguf_gemv engine --expert-sim --memcpy-during`
is DuringApiCall on batched prefetch (needs `--memcpy-batch`; identity stays
Stream). `expertvm sim --memcpy-any` / `gguf_gemv engine --expert-sim --memcpy-any`
is Any on batched prefetch (needs `--memcpy-batch`; empty deps; no API wait).
`expertvm sim --memcpy-attr` / `gguf_gemv engine --expert-sim --memcpy-attr`
is DuringApiCall on demand pinned/VMM miss H2D (the API waits that copy;
identity stays sequential `memcpy_pinned_to_device`; does not imply
`--memcpy-batch`).
`expertvm sim --d2h-evict` / `gguf_gemv engine --expert-sim --d2h-evict`
is `cudaMemcpyAsync` Device→HostPinned before pinned/VMM LRU free (extra
PCIe; next miss still fills from staging; not mapped/managed; distinct from
`--prefetch-host`).
`expertvm sim --d2h-pageable` / `gguf_gemv engine --expert-sim --d2h-pageable`
is `cudaMemcpyAsync` Device→Host (pageable bounce-buffer) before that free
(host-synchronous; implies `--pageable`; not mapped/managed/host-register/
`--d2h-evict`).
`expertvm sim --host-unregister` / `gguf_gemv engine --expert-sim --host-unregister`
is `cudaHostUnregister` after each miss DMA (implies `--host-register`; pin
refunded between misses; `synchronize`; identity keeps staging registered).
`expertvm sim --ipc` / `gguf_gemv engine --expert-sim --ipc`
is `cudaIpcGetMemHandle` / `OpenMemHandle` of each miss `cudaMalloc` (implies
`--sync-alloc`; alias shares source HBM; close before free; not with
`--shareable` / mapped / managed / vmm; identity GEMMs on the `cudaMalloc`
pointer).
`expertvm sim --share-ptr` / `gguf_gemv engine --expert-sim --share-ptr`
is `cudaMemPoolExportPointer` / `ImportPointer` of each miss `cudaMallocAsync`
(implies `--shareable`; alias shares source HBM; `cudaFreeAsync` import
before source; not with `--ipc` / mapped / managed / vmm / `--sync-alloc`;
identity is pool-level `--shareable`).
`expertvm sim --memset-fill` / `gguf_gemv engine --expert-sim --memset-fill`
is `cudaMemsetAsync` of pinned/VMM miss pages (HBM write, compute occupancy;
not mapped/managed/pageable/memcpy-batch; distinct from `--graph-memset`
scratch). `expertvm sim --copy-host` / `gguf_gemv engine --expert-sim --copy-host`
is `cudaLaunchHostFunc` after miss DMA / prefetch (`host_func_ns` before
copy-ready; does not imply `--host-func`; mapped misses are a no-op).
`memcpy_3d_batch_async`
is `cudaMemcpy3DBatchAsync` (3D pointer-to-pointer; `flags` must be 0; CUDA
arrays are not modeled; capture cannot include it).
`memcpy_3d_with_attributes` is `cudaMemcpy3DWithAttributesAsync` (Stream is
`memcpy_3d_async`). Typed `memcpy` stays. [`StreamId::NULL`] is the CUDA null
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
`event_get_flags` is `cudaEventGetFlags` (the create flags word; query;
legal during capture).
`event_get_id` is `cuEventGetId` / `cudaEventGetId` (unique per event
handle; not the caller-chosen `EventId`; query; legal during capture).
`pool_get_id` is `cuMemPoolGetId` (unique per pool handle; not the
`PoolId`; graph-memory pools are legal; query; legal during capture).
`query_event` is `cudaEventQuery` (unknown id is semantic; incomplete is
`Ok(false)`).
`destroy_event` is `cudaEventDestroy` (waits a recorded incomplete event;
never-recorded returns immediately; capture refused; the id may be created
again).
`query_stream` is `cudaStreamQuery` (unknown device is semantic; a busy
stream is `Ok(false)`; the clock does not advance).
`mem_info` is `cudaMemGetInfo` `(free, total)` HBM bytes.
`pointer_get_attributes` is `cudaPointerGetAttributes`.
`pointer_set_attribute` / `pointer_get_attribute` are
`cuPointerSetAttribute` / `GetAttribute` (`PointerAttr`: SyncMemops is
settable; MemoryType / DevicePointer / HostPointer / IsManaged /
RangeSize / Mapped / MemPoolHandle / DeviceOrdinal / RangeStartAddr /
BufferId / IsLegacyCudaIpcCapable / IsGpuDirectRdmaCapable /
AllowedHandleTypes / MappingBaseAddr / MappingSize /
IsHwDecompressCapable / MemoryBlockId are query-only;
VMM mapping size is the `cuMemMap` span at offset 0, not the reserved
VA; hardware decompress is always 0; memory-block id is the
`MemHandleId` covering offset 0). Set is capture-refused; Get is a query.
`pointer_get_access_flags` is `CU_POINTER_ATTRIBUTE_ACCESS_FLAGS` for an
explicit device (no TLS current device; not a `PointerAttr`). Flags are
`MemAccessFlags` from this VM's kernel residency: local / mapped host /
pool ReadWrite / VMM write access are ProtReadWrite; pool ProtRead /
VMM `va_set_access` / managed SetAccessedBy are ProtRead; `enable_peer`
does not grant kernel access to remote `cudaMalloc`. Query; legal
during capture. No Engine `--pointer-access`.
`pointer_get_attribute_n` is `cuPointerGetAttributes` (batch
`cuPointerGetAttribute`; distinct from the `cudaPointerGetAttributes`
struct). Empty is `Ok([])` after a live alloc. All-or-nothing. Query;
legal during capture. No Engine `--pointer-attrs`.
`expertvm sim --sync-memops` / `gguf_gemv engine --expert-sim --sync-memops`
sets `PointerAttr::SyncMemops` on miss device pages so H2D / managed
prefetch is host-synchronous (identity stays async pinned DMA).
`expertvm sim --device-sync-memops` / `gguf_gemv engine --expert-sim --device-sync-memops`
is `cudaSetDeviceFlags(cudaDeviceSyncMemops)` so every memcpy/memset on
that GPU is host-synchronous (distinct from per-page `--sync-memops`).
`mem_get_address_range` is `cudaMemGetAddressRange` (base is the alloc id;
interior offsets are not modeled). Query; legal during capture.
`host_get_device_pointer` is `cudaHostGetDevicePointer` (mapped host;
`host_get_device_pointer_with_flags` requires `flags == 0`).
`device_get_attribute` is `cudaDeviceGetAttribute` (modeled caps only;
`TotalGlobalMem` is HBM, `AsyncEngineCount` is copy engines,
`CanMapHostMemory` / `ManagedMemory` are always 1).
`ComputeCapabilityMajor` and `ComputeCapabilityMinor` are Hopper 9.0 on
example H100 (`cudaDeviceProp` major and minor). Distinct from occupancy
SM counts. No Engine flag for compute capability.
`device_compute_capability` is `cuDeviceComputeCapability` of those same
major and minor values.
`MaxThreadsPerBlock` is 1024. `MaxBlockDimX` / `Y` / `Z` are 1024, 1024,
64. `MaxGridDimX` / `Y` / `Z` are `i32::MAX`, 65535, 65535. Example H100
launch-geometry caps. `MaxRegistersPerBlock` is 65536 (`cudaDeviceProp`
regsPerBlock). This VM does not model a thread-block launch or a
register file and does not invent occupancy SM counts. No Engine flag
for max threads or max registers.
`GlobalMemoryBusWidth` is 5120 bits on example H100 and 6144 on example
H200 (`cudaDeviceProp` memoryBusWidth). Profile key
`global_memory_bus_width_bits`. Distinct from `hbm_bps` and from memory
clock rates. No Engine flag for bus width.
`SingleToDoublePrecisionPerfRatio` is 1 on example H100
(`cudaDeviceProp` singleToDoublePrecisionPerfRatio). Hopper FP32 and
FP64 peaks match. Distinct from occupancy SM counts and from clock
rates. This VM does not scale kernel duration from this ratio. No
Engine flag for the ratio.
`device_get_exec_affinity_support` is `cuDeviceGetExecAffinitySupport`
(`SM_COUNT` is 0; this VM uses permille green-context spans, not
occupancy SM counts). Other type ids are Invalid.
`ClusterLaunch` is `max_blocks_per_cluster > 0`. `HostRegisterSupported` /
`IpcEventSupport` / `CanUseHostPointerForRegisteredMem` are always 1.
`MemoryPoolSupportedHandleTypes` is POSIX-FD (`MemHandleType`).
`GpuDirectRdmaSupported` / `CanFlushRemoteWrites` are a GPU↔GPU RDMA link.
`flush_gpu_direct_rdma_writes` is `cudaDeviceFlushGPUDirectRDMAWrites` (1 ns
host-sync barrier; capture refused; write-ordering options are not modeled).
`HostRegisterReadOnlySupported` /
`PageableMemoryAccess` / `ConcurrentManagedAccess` /
`DirectManagedMemAccessFromHost` /
`PageableMemoryAccessUsesHostPageTables` are always 0 (ReadOnly host register is Invalid;
pageable is bounce-buffer; host cannot touch managed while a kernel runs).
`HostNativeAtomicSupported` / `CooperativeMultiDeviceLaunch` / `Integrated`
are always 0 (host-mapped atomics and multi-device cooperative are not
modeled; example SKUs are discrete). `SparseCudaArraySupported` /
`DeferredMappingCudaArraySupported` / `DmaBufSupported` are always 0
(CUDA arrays and dma-buf are not modeled). `va_get_handle_for_address_range`
is `cuMemGetHandleForAddressRange` (always Invalid `"dma-buf not modeled"`;
`MemRangeHandleType::DMA_BUF_FD` only). Distinct from `ipc_get` and
`create_shareable_pool`. Query; legal during capture. No Engine `--dma-buf`.
`va_export_to_shareable_handle` is `cuMemExportToShareableHandle` (always
Invalid `"not shareable"`; VMM create-time handle types are none). POSIX-FD
only; flags 0. Distinct from `ipc_get`, `pool_export`, and
`va_get_handle_for_address_range`. Capture cannot include it. No Engine
`--vmm-export`.
`MulticastSupported` is a
GPU↔GPU NVLink on that device (PCIe P2P and RDMA are not NVLS).
`VirtualMemoryManagementSupported` is always 1 (this VM has
`cuMemAddressReserve`). `HandleTypePosixFileDescriptorSupported` is always 1
(this VM has POSIX-FD shareable pools).
`GpuDirectRdmaFlushWritesOptions` is Host (`1`) on an RDMA SKU (MemOps is
never reported). `GpuDirectRdmaWritesOrdering` is always None (native
write visibility is not modeled; flush is never a no-op). Distinct from
FlushWritesOptions. `GpuDirectRdmaWithCudaVMMSupported` is the same RDMA SKU
bit (VMM is always on). `GenericCompressionSupported` is always 0
(compression is not modeled).
`HandleTypeWin32HandleSupported` / `HandleTypeWin32KmtHandleSupported` /
`HandleTypeFabricSupported` are always 0 (POSIX-FD shareable pools;
fabric handles are not modeled).
`D3D12CigSupported` is always 0 (D3D12 CUDA-in-graphics is not modeled).
`VulkanCigSupported` is always 0 (Vulkan CUDA-in-graphics is not modeled).
`HostMemoryPoolsSupported` is always 0 (pools are device-only).
`IsMultiGpuBoard` / `MultiGpuBoardGroupID` are always 0 (example SKUs
are discrete single-GPU packages). `ComputeMode` is always Default
(exclusive process / prohibited are not modeled). `MpsEnabled` is always
0 (CUDA Multi-Process Service is not modeled). `TccDriver` is always
0 (example SKUs are not Windows TCC). `KernelExecTimeout` is always 0
(example SKUs have no display watchdog). `CanUse64BitStreamMemOps` is
always 1 (`wait_value64` / `write_value64`). `CanUseStreamMemOps` is
always 1 (`wait_value32` / `write_value32`; CUDA deprecated this in
favor of `CanUse64BitStreamMemOps`). `CanUseStreamWaitValueNor`
is always 1 (`WaitValueCmp::Nor`). `TensorMapAccessSupported` is always
0 (`CUtensorMap` / TMA is not modeled). `UnifiedFunctionPointers` is
always 0 (device-side function pointers are not modeled).
`TimelineSemaphoreInteropSupported` is always 0 (NVSci / timeline
semaphore interop is not modeled). `device_get_nvscisync_attributes` is
`cudaDeviceGetNvSciSyncAttributes` (always Invalid `"nvscisync not modeled"`;
`NvSciSyncAttrFlags::SIGNAL` / `WAIT`). Query; legal during capture. No
Engine `--nvscisync`. `MemDecompressAlgorithmMask` /
`MemDecompressMaximumLength` are always 0 (hardware decompress is not
modeled). `HostNumaVirtualMemoryManagementSupported` is always 0 (host
NUMA VMM is not modeled; `va_create_with_prop` refuses host location).
`HostNumaMemoryPoolsSupported` is always 0 (host NUMA pools are not
modeled; `create_pool_with_props` refuses host location).
`HostNumaMultinodeIpcSupported` is always 0 (this VM's IPC is same-node;
`ipc_open` requires the dest GPU already in the allocation).
`NumaConfig` is always None (GPU memory NUMA nodes are not modeled;
do not invent `cudaDevAttrNumaId`).
`OnlyPartialHostNativeAtomicSupported` is always 0 (host-mapped atomics
are not modeled; distinct from `HostNativeAtomicSupported`).
`MaxAccessPolicyWindowSize` is L2 bytes (same as `MaxPersistingL2CacheSize`).
`DeviceProperties.persisting_l2_cache_max_size` is
`cudaDeviceProp::persistingL2CacheMaxSize` (that max; distinct from the
current `persisting_l2_cache_size` limit).
`GlobalL1CacheSupported` is always 0 (this VM does not model L1 caches;
distinct from `L2CacheSize`). `LocalL1CacheSupported` is always 0 (this
VM does not model L1 caches; distinct from `GlobalL1CacheSupported`).
`ComputePreemptionSupported` is always 0 (kernel preemption is not
modeled; distinct from `KernelExecTimeout`).
`EccEnabled` is always 0 (ECC is not modeled; distinct from `TccDriver`).
`ReservedSharedMemoryPerBlock` is always 0 (driver-reserved shared memory
is not modeled; distinct from `MaxSharedMemoryPerBlock`).
`MaxSharedMemoryPerMultiprocessor` matches `MaxSharedMemoryPerBlockOptin`
(reserved shared memory is 0; not occupancy SM counts; not
`MaxRegistersPerMultiprocessor`).
`TotalConstantMemory` is always 0 (`__constant__` memory is not modeled;
distinct from `TotalGlobalMem`).
`TextureAlignment` is always 0 (CUDA arrays / textures are not modeled;
distinct from `SparseCudaArraySupported`).
`SurfaceAlignment` is always 0 (CUDA surfaces are not modeled; distinct
from `TextureAlignment`).
`TexturePitchAlignment` is always 0 (CUDA textures are not modeled;
distinct from `TextureAlignment` and from `MemcpyOp` 2D pitches).
`MaxTexture1DWidth` is always 0 (CUDA arrays / textures are not modeled;
distinct from `TextureAlignment`).
`MaxTexture2DWidth` / `MaxTexture2DHeight` / `MaxTexture3DWidth` /
`MaxTexture3DHeight` / `MaxTexture3DDepth` are always 0 (CUDA arrays /
textures are not modeled; distinct from `MaxTexture1DWidth`).
`MaxTexture1DLinearWidth`, `MaxTexture2DLinearWidth`,
`MaxTexture2DLinearHeight`, and `MaxTexture2DLinearPitch` are always 0
(CUDA linear textures are not modeled; distinct from `MaxTexture1DWidth`
and from `TexturePitchAlignment`).
`MaxTexture2DGatherWidth` and `MaxTexture2DGatherHeight` are always 0
(CUDA texture gather is not modeled; distinct from `MaxTexture2DWidth`).
`MaxTexture1DMipmappedWidth`, `MaxTexture2DMipmappedWidth`, and
`MaxTexture2DMipmappedHeight` are always 0 (CUDA mipmapped textures are
not modeled; distinct from `MaxTexture1DWidth`).
`MaxTextureCubemapWidth` is always 0 (CUDA cubemap textures are not
modeled; distinct from `MaxTexture2DWidth`).
`MaxTexture1DLayeredWidth`, `MaxTexture1DLayeredLayers`,
`MaxTexture2DLayeredWidth`, `MaxTexture2DLayeredHeight`, and
`MaxTexture2DLayeredLayers` are always 0 (CUDA layered textures are not
modeled; distinct from `MaxTexture1DWidth` and from `MaxTextureCubemapWidth`).
`MaxTextureCubemapLayeredWidth` and `MaxTextureCubemapLayeredLayers` are
always 0 (CUDA cubemap layered textures are not modeled; distinct from
`MaxTextureCubemapWidth`).
`MaxSurface1DWidth`, `MaxSurface2DWidth`, `MaxSurface2DHeight`,
`MaxSurface3DWidth`, `MaxSurface3DHeight`, and `MaxSurface3DDepth` are
always 0 (CUDA surfaces are not modeled; distinct from `SurfaceAlignment`).
`MaxSurface1DLayeredWidth`, `MaxSurface1DLayeredLayers`,
`MaxSurface2DLayeredWidth`, `MaxSurface2DLayeredHeight`, and
`MaxSurface2DLayeredLayers` are always 0 (CUDA layered surfaces are not
modeled; distinct from `MaxSurface1DWidth`).
`MaxSurfaceCubemapWidth`, `MaxSurfaceCubemapLayeredWidth`, and
`MaxSurfaceCubemapLayeredLayers` are always 0 (CUDA cubemap surfaces are
not modeled; distinct from `MaxSurface2DWidth`).
`MaxPitch` is `DeviceAttr::MAX_PITCH` (`i32::MAX`; this VM does not cap
2D memcpy / `cudaMallocPitch` pitch; distinct from `TexturePitchAlignment`).
`ComputeCapabilityMajor` / `ComputeCapabilityMinor` are Hopper 9.0 on
example H100. Profile keys `compute_capability_major` /
`compute_capability_minor`. This VM does not invent occupancy SM counts
from compute capability.
`device_compute_capability` is `cuDeviceComputeCapability` of those same
major and minor values.
`MaxThreadsPerBlock` is 1024. Block dims are 1024, 1024, 64. Grid dims
are `i32::MAX`, 65535, 65535. Distinct from occupancy SM counts. This VM
does not model a thread-block launch.
`MaxRegistersPerBlock` is 65536. Distinct from occupancy SM counts. This
VM does not model a register file.
`GlobalMemoryBusWidth` is 5120 bits on example H100 and 6144 on example
H200. Profile key `global_memory_bus_width_bits`. Distinct from `hbm_bps`
and from memory clock rates.
`SingleToDoublePrecisionPerfRatio` is 1 on example H100. Distinct from
occupancy SM counts and from clock rates. This VM does not scale kernel
duration from this ratio.
`DeviceP2pAttr::OnlyPartialNativeAtomicSupported` is always 0 (P2P native
atomics are not modeled; distinct from `NativeAtomicSupported` and from
`OnlyPartialHostNativeAtomicSupported`). This VM does not invent
`cudaDeviceGetP2PAtomicCapabilities`.
`StreamPrioritiesSupported` /
`UnifiedAddressing` are always 1. `GpuOverlap` is `copy_engines > 0`.
`device_get_properties` is `cudaGetDeviceProperties` of those same fields
(compute capability major/minor included; launch-geometry caps included;
MaxRegistersPerBlock included; MaxSharedMemoryPerMultiprocessor matches optin; GlobalMemoryBusWidth included; SingleToDoublePrecisionPerfRatio included; texture 2D/3D
dims included; linear texture 1D/2D dims included; texture 2D gather dims included; mipmapped texture 1D/2D dims included; cubemap texture width included; layered texture 1D/2D dims included; cubemap layered texture dims included; surface 1D/2D/3D dims included; layered surface 1D/2D dims included; cubemap surface dims included; texture 3D alt dims included as 0; MpsEnabled included as 0; D3D12CigSupported included as 0; VulkanCigSupported included as 0; pciSubSystemID included as 0; GpuPciDeviceId included as 0; GpuPciSubsystemId included as 0; luid included as 0; no occupancy SM count or clock). `device_get_name` is `cudaDeviceGetName` (the
profile name). `device_compute_capability` is `cuDeviceComputeCapability`
(Hopper 9.0 on example H100; also `DeviceProperties` major and minor).
`device_get_uuid` is `cuDeviceGetUuid` (synthetic 16-octet
id; also `DeviceProperties.uuid`). `device_get_by_uuid` is
`cuDeviceGetByUuid` (inverse). `device_get_luid` is `cuDeviceGetLuid`
(always-zero Windows LUID plus node mask; also `DeviceProperties.luid`
and `luidDeviceNodeMask`). `device_get_texture_1d_linear_max_width` is
`cuDeviceGetTexture1DLinearMaxWidth` (always 0; CUDA linear textures
are not modeled). `device_get_pci_bus_id` is
`cudaDeviceGetPciBusId` (synthetic PCI string; also `DeviceProperties`
PCI ids; `pciSubSystemID` is always 0; `GpuPciDeviceId` is always 0;
`GpuPciSubsystemId` is always 0; `luid` is always 0). `device_get_by_pci_bus_id` is `cudaDeviceGetByPCIBusId`
(inverse). `device_total_mem` is `cuDeviceTotalMem` (HBM bytes).
`driver_get_version` is `cudaDriverGetVersion` / `cuDriverGetVersion` (CUDA
13.0). `driver_init` is `cuInit` (flags 0; already initialized at construct;
1 ns no-op; capture cannot include it; distinct from `init_device`).
`runtime_get_version` is `cudaRuntimeGetVersion` (same toolkit). Query;
legal during capture.
`func_get_attributes` is `cudaFuncGetAttributes`
of modeled per-device function attrs (`maxDynamicSharedSizeBytes`,
`nonPortableClusterSizeAllowed`, `preferredShmemCarveout`, cluster-dim
must-be-set, required cluster width/height/depth, and
`clusterSchedulingPolicyPreference`; not per kernel). Compiler-emitted
`sharedSizeBytes`, `constSizeBytes`, `localSizeBytes`, `maxThreadsPerBlock`,
`ptxVersion`, `binaryVersion`, and `cacheModeCA` are always 0 until a
compiled kernel exists. Distinct from device `MaxThreadsPerBlock`.
`numRegs` is not modeled this slice. `func_get_name` is `cudaFuncGetName` /
`cuFuncGetName` (empty until a compiled kernel exists; distinct from
`device_get_name`). `func_get_param_info` is `cuFuncGetParamInfo` (Invalid
`"unknown function"` until a compiled kernel exists). `func_set_attribute` /
`func_get_attribute` are `cudaFuncSetAttribute` / `GetAttribute` (`FuncAttr`).
Typed setters stay. `stream_get_flags` is `cudaStreamGetFlags`
(`0` `cudaStreamDefault` / `1` `cudaStreamNonBlocking`; NULL follows
`set_legacy_null_stream`). `stream_get_priority` is `cudaStreamGetPriority`.
`stream_get_id` is `cudaStreamGetId` (unique per device/stream; not the
caller-chosen `StreamId`). `stream_get_device` is `cudaStreamGetDevice` /
`cuStreamGetDevice` (the device of the stream; green-ctx streams return
the ctx create device). Query; legal during capture. Distinct from
`stream_get_id` and `green_ctx_get_device`. `ctx_get_id` is `cuCtxGetId` for the seeded
primary context (not `green_ctx_get_id`). `ctx_get_api_version` is
`cuCtxGetApiVersion` for that same primary context (CUDA 13.0; same
encoding as `driver_get_version`; distinct from Hopper SM version).
`ctx_get_flags` is `cuCtxGetFlags` for that same primary context (same
flags as `get_device_flags`; distinct from `device_primary_ctx_get_state`).
`ctx_get_cache_config` is `cuCtxGetCacheConfig` for that same primary
context (same as `get_cache_config`; distinct from `get_func_cache_config`).
`ctx_get_stream_priority_range` is `cuCtxGetStreamPriorityRange` for that
same primary context (same as `device_get_stream_priority_range`; example
H100 is `(0, -5)`).
`ctx_get_limit` is `cuCtxGetLimit` for that same primary context (same as
`get_limit` for a `DeviceLimit`).
`ctx_synchronize` is `cuCtxSynchronize` for that same primary context
(same wait as `synchronize_device`; capture cannot include it; other GPUs
keep running).
`ctx_get_shared_mem_config` is `cuCtxGetSharedMemConfig` for that same
primary context (same as `get_shared_mem_config`; distinct from
`get_func_shared_mem_config`).
`green_ctx_get_id` is
`cuGreenCtxGetId` / `cudaExecutionCtxGetId` (unique per live green ctx;
not `GreenCtxId`).
`green_ctx_get_device` is `cudaExecutionCtxGetDevice` (create device).
`stream_get_attribute` / `stream_set_attribute` are `cudaStreamGetAttribute` /
`SetAttribute` of existing stream state (`StreamAttr`: priority, synchronization
policy, mem-sync domain/map, NVLink-util-centric, access-policy window).
Green-context SM permille is not a CUDA stream attribute. Type mismatch is
Invalid `"stream attr"`. Get is a query (capture-legal).
`set_stream_access_policy` is `cudaStreamAttributeAccessPolicyWindow`:
`kernel` / `kernel_bufs` inherit it; `kernel_with` and graph replay use the
launch / node window. Set `None` clears. This VM does not cap stream-priority
range.
`set_limit` / `get_limit` are `cudaDeviceSetLimit` / `GetLimit`.
`DevRuntimePendingLaunchCount` caps in-flight `device_launch_graph` (default
2048; queued tail does not occupy a slot). Host `launch_graph` does not count.
`set_shared_mem_config` / `get_shared_mem_config` are
`cudaDeviceSetSharedMemConfig` / `GetSharedMemConfig` (Default kernels
inherit the function config, then this; unset is unscaled).
`set_func_shared_mem_config` / `get_func_shared_mem_config` are
`cudaFuncSetSharedMemConfig` / `GetSharedMemConfig` (per device).
`set_cache_config` / `get_cache_config` are `cudaDeviceSetCacheConfig` /
`GetCacheConfig` (`FuncCache`; PreferNone default). PreferShared /
PreferL1 / PreferEqual are stored; L1 is not modeled, so kernel duration
does not change. `set_func_cache_config` / `get_func_cache_config` are
`cudaFuncSetCacheConfig` (per device; CUDA has no FuncGet). Distinct from
shared-mem carveout and bank width. No Engine `--cache-config`.
`set_device_flags` / `get_device_flags` are
`cudaSetDeviceFlags` / `GetDeviceFlags` (`DeviceFlags` schedule plus stored
MapHost / LmemResizeToMax; `SYNC_MEMOPS` waits memcpy/memset like pointer
SyncMemops; Auto streams inherit the tax). This VM does not model `cudaErrorSetOnActiveProcess`.
`init_device` / `init_device_with_flags` are `cudaInitDevice` (primary ctx
already seeded; does not make a thread-current device).
`InitDeviceFlags::FLAGS_ARE_VALID` applies `deviceFlags` like
`set_device_flags`; without that bit they are ignored. No Engine
`--init-device`.
`reset_device` is `cudaDeviceReset`. Waits that GPU, then frees
`cudaMalloc` / `cudaMallocPitch` / `cudaMalloc3D`. `cudaMallocAsync` stays.
User streams except NULL become unknown until create. Device flags and
limits return to CUDA defaults. Peer pairs involving that GPU return to
the profile seed. Host / managed allocs and events stay. Graphs stay.
`ctx_get_id` stays. Capture cannot include it. No `cuDevicePrimaryCtxReset`.
No Engine flag for device reset.
`SimError::error_name` / `error_string` are `cudaGetErrorName` /
`cudaGetErrorString`. Query on the error already returned (no thread-local
last error). `Display` stays the detailed reason. No Engine flag for error
name.
`device_primary_ctx_get_state` is `cuDevicePrimaryCtxGetState` (flags match
`get_device_flags`; active is always true). No `cuDevicePrimaryCtxRetain`.
`device_primary_ctx_set_flags` is `cuDevicePrimaryCtxSetFlags` (always
Invalid `"primary context active"`; this VM seeds a primary context at
construct). Distinct from `set_device_flags`. Capture cannot include it.
No Engine `--primary-ctx-flags`.
`ctx_get_id` is `cuCtxGetId` for the seeded primary context of an explicit
device (no TLS current device). Distinct from `green_ctx_get_id`. Query;
legal during capture. No Engine `--ctx-id`.
`expertvm sim --device-sync-memops` sets `DeviceFlags::SYNC_MEMOPS`.
`expertvm sim --device-sync-policy blocking` sets `DeviceFlags::SCHEDULE_BLOCKING_SYNC`
(Auto streams inherit host-wait tax; explicit `--sync-policy` wins).
Persisting L2 is `cudaLimitPersistingL2CacheSize`. Access-policy windows
must align to `cudaLimitMaxL2FetchGranularity` (SM 8.0+ default 128).
`malloc_pitch` is `cudaMallocPitch`. `malloc_pitch_with_element_size` is
`cuMemAllocPitch` (`ElementSizeBytes` 4 / 8 / 16; pitch still 512-aligned).
No Engine `--malloc-pitch-element`. `MemcpyOp` `height` / pitches are
`cudaMemcpy2DAsync` (payload `width * height`). Origin fields are srcPos /
dstPos (default 0). No Engine `--memcpy-origin`. `MemcpyOp` `src_lod` /
`dst_lod` are CUDA_MEMCPY3D `srcLOD` / `dstLOD` (must be 0). No Engine
`--memcpy-lod`. `MemsetOp` `height` / `pitch`
are `cudaMemset2DAsync` (payload `width * height`; padding is not written).
`MemsetOp` `element_size` is `cudaMemsetNodeParams::elementSize` (`1` / `2` /
`4`; typed `memset` stays `1`). Offset, width, and nonzero pitch must divide
that size. No Engine `--memset-element`. The fill value is not modeled.
`malloc_3d` is `cudaMalloc3D`. `MemcpyOp` `depth` / slice heights are
`cudaMemcpy3DAsync` (payload `width * height * depth`). `memcpy_3d_unaligned`
is `cuMemcpy3DUnaligned` (identity with `memcpy_3d`). No Engine
`--memcpy-3d-unaligned`. `MemsetOp` `depth` /
`ysize` are `cudaMemset3DAsync` (payload `width * height * depth`).
`graph_mem_get` / `graph_mem_set` / `graph_mem_trim` are
`cudaDeviceGetGraphMemAttribute` / `SetGraphMemAttribute` / `GraphMemTrim`
(device graph-memory pool only; unused reserved bytes return on trim).
Default `cudaMallocAsync` uses the device mempool with release threshold
`0` (unused bytes return to the OS when the stream-ordered free
completes). `create_pool` / `create_pool_with_props` / `alloc_from_pool` /
`set_pool_release_threshold` / `set_pool_max_size` / `pool_trim_to` /
`pool_get_attribute` / `pool_set_attribute` / `pool_get_id` are
`cudaMemPoolCreate` / `Create`+`MemPoolProps` /
`cudaMallocFromPoolAsync` / `cudaMemPoolAttrReleaseThreshold` /
`cudaMemPoolAttrMaxPoolSize` / `cudaMemPoolTrimTo` /
`cudaMemPoolGetAttribute` / `SetAttribute` / `cuMemPoolGetId`.
`MemPoolProps` is pinned alloc type, NONE or POSIX-FD handles, a device
location, `max_size` (`0` unlimited; otherwise reserved cannot grow
past it), and `usage` (`MemHandleUsage::NONE` only; HW decompress is not
modeled). Typed `create_pool` / `create_shareable_pool` stay.
`MemPoolAttr` is ReleaseThreshold / UsedMemCurrent / UsedMemHigh /
ReservedMemCurrent / ReservedMemHigh / MaxPoolSize plus reuse flags
(default 1). AllocationType is always PINNED. ExportHandleTypes is POSIX-FD
on shareable exporters and NONE on imported (cannot re-export). Only
`ReuseAllowOpportunistic=0` skips cache reuse (OS alloc; unused cached
bytes stay reserved). FollowEvent / Internal do not insert event waits
or extra sync. High-water Set `0` resets to current; graph mem stays
`GraphMemAttr`.
`u64::MAX` holds unused bytes so `malloc` can OOM
until trim. `expertvm sim --mempool-trim` is `pool_trim_to(device_mempool, 0)`
after score (hold during the run, return cache at idle).
`expertvm sim --mempool-no-reuse` is `ReuseAllowOpportunistic=0` (OS alloc;
leftover cache stays reserved). `expertvm sim --mempool-max N` is
`cudaMemPoolCreate` + `MemPoolProps::max_size` then `cudaDeviceSetMemPool`
(reserved `live+cached` cannot grow past `N`; `0` unset is unlimited).
`MemPoolAttr::MaxPoolSize` / `set_pool_max_size` is
`cudaMemPoolAttrMaxPoolSize` (Set after create; Get reports the stored
cap; this VM does not round up for alignment). CLI stays create plus
SetMemPool. `MemPoolAttr::AllocationType` /
`MemPoolAttr::ExportHandleTypes` are Get-only
(`cudaMemPoolAttrAllocationType` always PINNED;
`cudaMemPoolAttrExportHandleTypes` is POSIX-FD on shareable exporters and
NONE on imported). Destroying a user pool (`destroy_pool` / `cudaMemPoolDestroy`)
returns unused cache to the OS; outstanding allocs stay valid; the default
pool cannot be destroyed; destroying the current pool rebinds GetMemPool
to GetDefaultMemPool. Capture cannot include pool create/trim/set-attribute
/destroy.
`pool_get_access` is `cudaMemPoolGetAccess` (owner ReadWrite by default;
peers need `pool_set_access` / `pool_set_access_read`). `pool_set_access_with_flags` is the flags
word (`PROT_READ_WRITE` / `PROT_READ` / `PROT_NONE`). Typed helpers stay.
`ipc_get` / `ipc_open` / `ipc_close` are `cudaIpcGetMemHandle` /
`cudaIpcOpenMemHandle` / `cudaIpcCloseMemHandle`: the import aliases the
source physicals (no extra HBM). `ipc_open_with_flags` accepts
`cudaIpcMemLazyEnablePeerAccess` as a no-op (dest must already hold the
source; cross-GPU lazy peer is not modeled). Free of the source while imports are live
is Invalid. `ipc_get` of a mempool alloc is Invalid. Capture cannot include IPC.
`expertvm sim --ipc` / `gguf_gemv engine --expert-sim --ipc` export each miss
`cudaMalloc` then open an alias (implies `--sync-alloc`; GEMM stays on the
source pointer; close before free).
`ipc_get_event` / `ipc_open_event` are `cudaIpcGetEventHandle` /
`cudaIpcOpenEventHandle` (interprocess event alias; destroy of the source
while imports are live is Invalid).
`create_shareable_pool` is `cudaMemPoolCreate` with a POSIX-FD handle type.
`pool_export` / `pool_import` are `cudaMemPoolExportToShareableHandle` /
`ImportFromShareableHandle`: the import is a new pool id that shares
live/cached/threshold with the exporter. `pool_export_with_type` /
`pool_import_with_type` take POSIX-FD and flags 0 (`MemPoolExportFlags`).
Typed helpers stay. `pool_export_ptr` /
`pool_import_ptr` are `cudaMemPoolExportPointer` / `ImportPointer` (alias,
no extra HBM). `expertvm sim --share-ptr` / `gguf_gemv engine --expert-sim --share-ptr`
export each miss `cudaMallocAsync` then import an alias (implies `--shareable`;
GEMM stays on the source pointer; `cudaFreeAsync` import before source). `set_device_mempool` is `cudaDeviceSetMemPool`. `default_pool` is
`cudaDeviceGetDefaultMemPool` (seeded; SetMemPool does not replace it).
`device_mempool` is `cudaDeviceGetMemPool` (`alloc` draws from it). Default and
`create_pool` pools cannot be exported. Capture cannot include shareable
export/import.
`alloc_host` is pageable; `host_register` / `host_register_mapped` are
`cudaHostRegister` (host-synchronous). `host_register_with_size` is the CUDA
`size` argument (`size` must equal the allocation; partial register is not
modeled). `alloc_host_mapped` is
`cudaHostAllocMapped`: a kernel may read it with no H2D, billed at host
PCIe, and it does not charge HBM. `alloc_host_with_flags` /
`host_register_with_flags` store `PORTABLE` / `WRITE_COMBINED` (alloc) and
`PORTABLE` (register); those bits do not change DMA. IoMemory / ReadOnly
are Invalid. Capture cannot include host
alloc/register. `expertvm sim --host-register` / `gguf_gemv engine --expert-sim --host-register`
register pageable staging then DMA H2D (identity stays `cudaMallocHost`).
`expertvm sim --host-unregister` / `gguf_gemv engine --expert-sim --host-unregister`
unregister that staging after each miss DMA (`synchronize`; next miss
re-registers; identity keeps staging registered).
`expertvm sim --host-register-mapped` / `gguf_gemv engine --expert-sim --host-register-mapped`
is `cudaHostRegisterMapped` on expert pages (`alloc_host` then register+map;
implies `--mapped`; identity mapped stays `cudaHostAllocMapped`).
`expertvm sim --sync-memops` / `gguf_gemv engine --expert-sim --sync-memops`
sets `PointerAttr::SyncMemops` on miss device pages so H2D / managed
prefetch is host-synchronous (illegal with `--mapped` / `--memcpy-batch`;
identity stays async pinned DMA).
`expertvm sim --device-sync-memops` / `gguf_gemv engine --expert-sim --device-sync-memops`
is `cudaSetDeviceFlags(cudaDeviceSyncMemops)` so every memcpy/memset on
that GPU is host-synchronous (illegal with `--mapped` / `--memcpy-batch`;
identity stays async pinned DMA; distinct from per-page `--sync-memops`).
`alloc_managed` is `cudaMallocManaged` (no HBM until
`prefetch` / first-touch at kernel start). Default attach is Global.
`alloc_managed_host` is `cudaMemAttachHost`. `alloc_managed_with_flags` is
`cudaMallocManaged` (`MemAttachFlags::GLOBAL` / `HOST`; Single is Invalid). `stream_attach` is
`cudaStreamAttachMemAsync` (stream-ordered; Host and other-stream Single
fail device kernels / memset / device prefetch; Single cannot use the NULL
stream; capture is refused). `stream_attach_with_flags` maps
`MemAttachFlags::{GLOBAL, HOST, SINGLE}` then typed `stream_attach`
(other bits Invalid `"stream attach flags"`). `stream_attach_with_size`
is the CUDA `length` argument (`0` is the entire allocation; a nonzero
`size` must equal the allocation; partial attach is not modeled). Typed
`stream_attach` stays.
`expertvm sim --stream-attach` / `gguf_gemv engine --expert-sim --stream-attach`
attach managed experts to the compute stream and prefetch there (identity
stays Global + copy-stream prefetch).
`expertvm sim --managed-host` / `gguf_gemv engine --expert-sim --managed-host`
allocate `cudaMemAttachHost` then Global-attach on the copy stream before
prefetch (identity stays Global at alloc).
`expertvm sim --prefetch-host` / `gguf_gemv engine --expert-sim --prefetch-host`
evict managed pages with `cudaMemPrefetchAsync(..., cudaCpuDeviceId)` and
restore by prefetching the same alloc back (identity stays `cudaFree`).
`mem_advise` is `cudaMemAdvise` (host-sync).
`mem_advise_with_size` is the CUDA `count` argument (`size` must equal
the allocation; partial advise is not modeled).
`mem_advise_with_location` is `cudaMemAdvise_v2` (`Place` location;
AccessedBy requires a device place; host preferred is
`SetPreferredLocationHost`). Typed `mem_advise` stays.
`SetReadMostly` makes prefetch replicate; `expertvm --no-read-mostly` is
`UnsetReadMostly` so dest prefetch moves. `SetAccessedBy` lets a kernel
read without migrating. `SetPreferredLocation` keeps a page already at
that GPU there on a remote read (writes still migrate; host preferred
does not skip kernel first-touch). `expertvm --no-preferred` is
`UnsetPreferredLocation` so a remote GEMM first-touches.
`expertvm --no-mem-prefetch` skips fill `cudaMemPrefetchAsync` so the
kernel first-touches instead of copy-engine prefetch. `mem_range_get_attribute` /
`mem_range_get_attributes` are `cudaMemRangeGetAttribute` /
`GetAttributes` of modeled per-alloc advice (`MemRangeAttr`;
not per byte range). `mem_range_get_attribute_with_size` /
`mem_range_get_attributes_with_size` are the CUDA `count` argument
(`size` must equal the allocation; partial range queries are not
modeled). `mem_range_get_attribute_with_data_size` /
`mem_range_get_attributes_with_data_sizes` are the CUDA `dataSize` /
`dataSizes` arguments (4-byte ints; AccessedBy includes a terminator). Last-prefetch is the dest of `prefetch` /
`prefetch_host`. Preferred/last-prefetch location type (`0` Invalid /
`1` Device / `2` Host) and id (device ordinal, else `0`) wrap that
`Place`. Host NUMA is not modeled. Query; legal during
capture. `prefetch` / `prefetch_host` / `prefetch_with_flags` /
`prefetch_with_size` / `prefetch_host_with_size` are
`cudaMemPrefetchAsync` and **move** unless ReadMostly.
`prefetch_with_flags` requires `flags == 0` (`PrefetchFlags::DEFAULT`) and
a `Place` dest. `prefetch_with_size` / `prefetch_host_with_size` are the
CUDA `count` argument (`size` must equal the allocation; partial prefetch
is not modeled). Typed helpers stay. `prefetch_batch_async` /
`discard_batch_async` / `discard_and_prefetch_batch_async` are
`cudaMemPrefetchBatchAsync` / `cudaMemDiscardBatchAsync` /
`cudaMemDiscardAndPrefetchBatchAsync`: they require
`ConcurrentManagedAccess` on every GPU. This VM reports 0, so they are
Invalid `"concurrent managed access"`. Discard contents are not modeled.
Capture of
`alloc_managed` / `mem_advise` / `stream_attach` is refused; a graph must record prefetch
before the kernel unless AccessedBy or PreferredLocation covers that GPU.
`va_reserve` / `va_map` / `va_unmap` / `va_free` are CUDA virtual memory.
`va_unmap_with_size` is `cuMemUnmap` with the reservation size (must
match; partial unmap is `va_unmap_range`).
`va_free_with_size` is `cuMemAddressFree` with the reservation size
(must match; partial free is not modeled).
`va_reserve_with_flags` is `cuMemAddressReserve` alignment / addr / flags
(flags 0; nonzero addr Invalid; nonzero alignment must be a power of two
that divides size).
`va_granularity_bytes` is `cuMemGetAllocationGranularity` (`0`/`1` accepts
any size; a 2 MiB profile rejects unaligned reserve/map).
`va_map_range` / `va_unmap_range` map sparse physicals (HBM is the mapped
span). `va_create` is `cuMemCreate` (HBM, no VA). `va_create_with_prop` is
the prop + flags word (pinned device; flags 0; `MemHandleType::NONE` only;
`compression` 0; `usage` `MemHandleUsage::NONE`).
`va_map_handle` is `cuMemMap`
of that handle (no second HBM charge; two VAs may share it).
`va_map_handle_with_flags` is the flags word (0).
`va_map_handle_with_size` is the CUDA size argument (must match the
handle). `va_retain_handle`
is `cuMemRetainAllocationHandle` (combined `va_map` spans are promoted).
`va_release_handle` is `cuMemRelease` while mapped; HBM refunds when refs and
maps are 0. `va_get_allocation_properties` is
`cuMemGetAllocationPropertiesFromHandle` (`MemAllocationProp`; pinned device
location; handle types always none; RDMA capable wraps the SKU).
`va_get_allocation_granularity` is `cuMemGetAllocationGranularity`
(minimum and recommended are the same profile value; `0`/`1` → `1`).
Both are queries; legal during capture. `va_map` still Create+Maps in one call.
`multicast_create` / `multicast_add_device` / `multicast_bind_mem` /
`multicast_bind_addr` / `multicast_unbind` / `multicast_destroy` /
`va_map_multicast` are NVLS multicast (NVLink clique).
`multicast_create_with_prop` is `cuMulticastCreate` with
`CUmulticastObjectProp` (handle types none; flags 0). Typed
`multicast_create` stays.
`va_map_multicast_with_flags` is `cuMemMap` flags of a multicast handle
(0; typed `va_map_multicast` stays).
`va_map_multicast_with_size` is the CUDA size argument (must match the
multicast object).
`multicast_unbind_with_size` is `cuMulticastUnbind` with the object size
(must match; `mcOffset` 0; partial unbind is not modeled).
`multicast_bind_mem_with_flags` / `multicast_bind_addr_with_flags` require
flags 0 (`MulticastBindFlags::DEFAULT`). Typed helpers stay.
`multicast_bind_addr_with_size` is the CUDA size argument (must match the
reserved VA; `mcOffset` 0; partial bind is not modeled).
`multicast_bind_mem_with_size` is the CUDA size argument (must match the
handle; `mcOffset` / `memOffset` 0; partial bind is not modeled).
`multicast_get_granularity` is `cuMulticastGetGranularity` (minimum and
recommended are the same profile value; `0`/`1` → `1`). Query; legal
during capture. `multicast_get_granularity_with_prop` takes
`CUmulticastObjectProp` (handle types none; flags 0; size and team size
are not validated). Typed `multicast_get_granularity` stays.
`va_set_access` is `cuMemSetAccess` PROT_READ on a peer (no dest HBM;
writes still need a local map). `va_set_access_write` is PROT_READWRITE
(peer writes, no dest HBM). `va_set_access_with_flags` is the flags word
(`PROT_READ` / `PROT_READ_WRITE` / `PROT_NONE`). Typed helpers stay.
`va_set_access_with_size` is the CUDA size argument (must match the
reserved VA; partial SetAccess is not modeled). `va_set_access_n` is the
CUDA descriptor array (all-or-nothing; empty is a no-op after the size
check). Typed helpers stay.
`va_get_access` is `cuMemGetAccess` (local map ReadWrite; peer Read /
ReadWrite / None). Query; legal during capture.
`pool_set_access` is `cudaMemPoolSetAccess`
ReadWrite on a peer (no dest HBM; kernels may write). `pool_set_access_read` is
ProtRead (peer reads, no dest HBM; writes stay `NotResident`). `pool_set_access_n`
is the CUDA descriptor array (all-or-nothing; empty is a no-op after
pool checks). Typed helpers stay. `kernel()` needs the whole VA covered; `kernel_bufs`, `memset_buf`, and
`MemcpyOp::offset` touch a mapped page (paged KV). `va_acquire` remaps an idle VA of the same
size (or reserves); `va_reserve_idle` is the map-less twin (split create/map is
`va_create` then `va_map_handle`); `va_acquire_paged` maps KV-block physicals covering the VA;
`va_release` unmaps into that pool. Capture cannot
include them.
`host_func` is `cudaLaunchHostFunc` (stream-ordered; other streams can compute;
unnamed callback). `host_func_params` / `graph_add_host_func_params` record
`HostNodeParams`. `stream_add_callback` is `cudaStreamAddCallback` (same
host enqueue; cannot be captured). `stream_add_callback_with_flags` is the
CUDA flags word (`StreamCallbackFlags`; must be 0).
`stream_add_callback_params` records `HostNodeParams`.
`graph_host_set_params` is `cudaGraphHostNodeSetParams`
(definition; does not retarget an exec). `graph_exec_host_set_params` is
`cudaGraphExecHostNodeSetParams`.
`write_value64` / `wait_value64` are `cuStreamWriteValue64` /
`cuStreamWaitValue64` (mailbox on complete; unwritten locations read as 0;
kernel/memset/memcpy stores are not modeled; no compute/copy occupancy).
`expertvm sim --wait-value` / `gguf_gemv engine --expert-sim --wait-value`
use an 8-byte `cudaMallocAsync` mailbox (copy stream waited before H2D)
so compute waits Eq instead of a copy event (decode identity stays
events; GEMM graphs stay kernel-only).
`write_value64_with_flags` / `write_value32_with_flags` are the CUDA flags
word (`WriteValueFlags`; NO_MEMORY_BARRIER Invalid). Typed helpers stay.
`wait_value64_with_flags` / `wait_value32_with_flags` are the CUDA flags
word (`WaitValueFlags`; `FLUSH` is a stream-ordered RDMA flush after the
wait). Typed helpers stay.
`batch_mem_op` is `cuStreamBatchMemOp` (one stream op; a wait sees earlier
writes in that vector). `BatchMemOp::FlushRemoteWrites` is
`CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES` (1 ns Solo on an RDMA GPU; not
host-sync; `flush_gpu_direct_rdma_writes` stays the device-wide host-sync
barrier). `batch_mem_op_with_flags` is the CUDA flags word
(`BatchMemOpFlags`; must be 0). Typed helper stays.
`set_stream_blocking` is `cudaStreamCreate` vs `cudaStreamNonBlocking`
(NULL serializes with blocking streams; created streams default to
non-blocking). `stream_create_with_flags` / `stream_create_with_priority`
are the CUDA flag-word twins (`StreamCreateFlags::NON_BLOCKING`; unknown
bits Invalid; capture refused). `set_legacy_null_stream` is the CUDA legacy default
stream (NULL serializes with every stream).
`host_pin_bytes` caps page-locked host (`cudaMallocHost` / `cudaHostRegister`);
overflow is `PinOom`. Example default is unlimited.
`set_stream_priority` is the priority-only helper.
`stream_create_with_priority` is `cudaStreamCreateWithPriority` (flags plus
priority; clamped to `device_get_stream_priority_range`; numerically lower
first when compute contends). `destroy_stream` is `cudaStreamDestroy`
(returns immediately; in-flight work still completes; NULL is Invalid).
`device_get_stream_priority_range` is
`cudaDeviceGetStreamPriorityRange` (example H100 least `0`, greatest `-5`).
`current_graph_exec` is `cudaGetCurrentGraphExec` (DeviceLaunch in
flight; host `launch_graph` does not count). Query; legal during capture.
`GRAPH_FIRE_AND_FORGET` / `GRAPH_TAIL_LAUNCH` are
`cudaStreamGraphFireAndForget` / `cudaStreamGraphTailLaunch` (nested
`device_launch_graph`; host `launch_graph` cannot use those ids).
`GRAPH_FIRE_AND_FORGET_AS_SIBLING` is `cudaStreamGraphFireAndForgetAsSibling`
(parent instance does not wait).
`destroy_stream` is `cudaStreamDestroy` (returns immediately; in-flight
work still completes; NULL is Invalid; recreate while unfinished is
`"stream in flight"`). Capture cannot include it.
`KernelAttrs::priority` is `cudaLaunchAttributePriority`
(`None` inherits the stream; `Some` overrides that kernel only).
`stream_copy_attributes` is `cudaStreamCopyAttributes`
(priority, SM permille, mem-sync domain/map, synchronization policy,
NVLink-util-centric scheduling, and access-policy window).
`set_stream_sync_policy` is `cudaLaunchAttributeSynchronizationPolicy`
on streams. `graph_kernel_node_set_sync_policy` is the CUDA 13 graph
kernel-node twin (not `KernelAttrs`; not valid for host launches). Auto tax 0.
`synchronize_stream` / `synchronize_event` add
`host_sync_spin_ns` / `yield` / `blocking` after the GPU drain (default 0).
`synchronize` / `synchronize_device` do not take that tax.
`graph_kernel_node_get_priority` /
`set_priority` / `copy_attributes` are `cudaGraphKernelNodeGetAttribute` /
`SetAttribute` / `CopyAttributes` for priority (`cudaLaunchAttributePriority`),
cooperative launch (`cudaLaunchAttributeCooperative`),
programmatic dependent launch (`ProgrammaticLaunch`), programmatic event (`ProgrammaticEvent`),
access-policy window (`AccessPolicyWindow`), mem-sync domain/map,
cluster, preferred cluster, shared-memory carveout,
device-updatable kernel node (`cudaLaunchAttributeDeviceUpdatableKernelNode`),
shared-memory bank mode (`cudaLaunchAttributeSharedMemoryMode`), and
portable-cluster size mode (`cudaLaunchAttributePortableClusterSizeMode`),
CUDA 13 portable-shared mode (`cudaLaunchAttributeSharedMemoryMode` /
`PortableSharedMode`), NVLink-util-centric scheduling
(`cudaLaunchAttributeNvlinkUtilCentricScheduling`), and synchronization
policy (`cudaLaunchAttributeSynchronizationPolicy` on graph kernel nodes;
not `KernelAttrs`).
`graph_kernel_node_get_attribute` / `graph_exec_kernel_node_get_attribute` /
`graph_kernel_node_set_attribute` / `graph_exec_kernel_node_set_attribute`
are the generic `cudaGraphKernelNodeGetAttribute` / `SetAttribute`
(`KernelNodeAttr`). Typed getters stay. Definition Set does not retarget
exec. Attr/value mismatch is Invalid `"kernel node attr"`. A parked
in-flight-destroyed exec is `"unknown graph"` on SetAttribute; a live exec
stays. A parked in-flight-destroyed exec is `"unknown graph"` on
GetAttribute; a live exec stays. Query; capture is legal.
`graph_exec_kernel_node_copy_attributes` is the exec-snapshot CopyAttributes
twin (uninstantiated graphs are Invalid). A parked in-flight-destroyed exec
used as CopyAttributes src or dst is `"unknown graph"`; a live exec as either
end stays.
`kernel_pdl` is `cudaLaunchKernelEx` PDL:
a wait kernel may start after the previous same-stream kernel's trigger
(`pdl_trigger_permille`) instead of its completion. Overlap needs
`compute_slots >= 2`. A later `cudaFreeAsync` on that stream still waits
for the overlapped primary (all preceding work). `kernel_pdl_event` is
`cudaLaunchAttributeProgrammaticEvent`: other streams may `wait_event`
at the trigger instead of kernel completion.
`ProgrammaticEvent::trigger_at_block_start` is CUDA `triggerAtBlockStart`:
the event records when the kernel starts (this VM does not invent an Engine
flag for programmatic-event block-start). `ProgrammaticEvent::external`
(`cudaEventRecordExternal`) is Invalid; interprocess / IPC-imported events
are Invalid. The same flags / interprocess rules apply to
`LaunchCompletionEvent`. `kernel_with` also accepts
`KernelAttrs::programmatic_event`. `expertvm sim --programmatic-event` /
`gguf_gemv engine --expert-sim --programmatic-event` attach it to grouped
GEMMs; store `pin_hot` replica D2D waits the PDL trigger.
`kernel_launch_completion` is
`cudaLaunchAttributeLaunchCompletionEvent`: the event records when the
kernel starts. `expertvm sim --pdl` / `gguf_gemv engine --expert-sim --pdl`
launch grouped expert GEMMs that way. `kernel_with` also accepts
`KernelAttrs::launch_completion` so captured expert GEMMs carry the
attribute. `expertvm sim --launch-completion` /
`gguf_gemv engine --expert-sim --launch-completion` attach it to grouped
GEMMs; store `pin_hot` replica D2D waits kernel start. `kernel_access_policy` is
`cudaLaunchAttributeAccessPolicyWindow`: persisting hits reduce billed HBM
after `set_persisting_l2_cache_size` (CUDA default is 0).
`set_stream_access_policy` is `cudaStreamAttributeAccessPolicyWindow`;
`kernel` inherits it. `expertvm sim --l2-persist`
enables the persist limit and attaches a window to expert GEMMs.
`--l2-reset` is `cudaCtxResetPersistingL2Cache` after each GEMM (implies
`--l2-persist`; live; cannot capture). `--l2-fetch N` is
`cudaLimitMaxL2FetchGranularity` (`32`/`64`/`128`; implies `--l2-persist`).
`--l2-ratio N` is CUDA `hitRatio` as ‰ (`1..=1000`; implies `--l2-persist`;
unset is 1000). `--l2-streaming` is `cudaAccessPropertyStreaming` for persist
GEMM window hits (needs `--l2-persist`; a reused expert bills full HBM).
`kernel_with` also accepts `cudaLaunchAttributeMemSyncDomain` /
`MemSyncDomainMap`: a completing kernel waits `same_domain_fence_permille` of
leftover same-physical-domain traffic (default tax 0). Remote (and allreduce)
isolates communication. `expertvm sim --mem-sync-domain remote` /
`gguf_gemv engine --expert-sim --mem-sync-domain remote` put decode GEMMs
on Remote (prefill stays Default). `expertvm sim --mem-sync-map collapse` /
`gguf_gemv engine --expert-sim --mem-sync-map collapse` maps remote→0 on
that decode stream so leftover prefill fence tax returns (needs
`--mem-sync-domain remote`; Hopper identity is remote→1). `expertvm sim --mem-sync-launch` /
`gguf_gemv engine --expert-sim --mem-sync-launch` put Remote on grouped GEMMs
so leftover prefill inherit-Default joins decode Remote (needs
`--mem-sync-domain remote`). `expertvm sim --mem-sync-launch-map` /
`gguf_gemv engine --expert-sim --mem-sync-launch-map` put collapse
`{default: 0, remote: 0}` on grouped GEMMs so leftover prefill maps
Default→0 with decode Remote→0 (needs `--mem-sync-domain remote`).
`ClusterDim` is `cudaLaunchAttributeClusterDimension`:
the launch occupies `min(blocks, compute_slots)` Hyper-Q slots (Hopper portable
max 8). `ClusterSchedulingPolicy::Spread` occupies every slot.
Launch Default uses `set_func_cluster_policy`
(`cudaFuncAttributeClusterSchedulingPolicyPreference`).
`expertvm sim --func-cluster-spread` sets Spread so launch Default occupies
every Hyper-Q slot when `--cluster` is at least 2 (distinct from
`--cluster-spread`).
`expertvm sim --cluster-must-set` is `cudaFuncAttributeClusterDimMustBeSet`
(needs `--cluster`; occupancy matches `--cluster`; SetAttribute is +1 ns).
`preferred_cluster` is used when that size fits in `compute_slots`.
`SharedMemCarveout::MaxShared` occupies every slot (`cudaLaunchAttributePreferredSharedMemoryCarveout`).
Default uses `set_func_carveout` (`cudaFuncAttributePreferredSharedMemoryCarveout`).
`cudaLaunchAttributeDeviceUpdatableKernelNode` lets
`graph_exec_kernel_set_params` keep the exec uploaded so
`device_launch_graph` needs no host re-upload (device-launch graphs allow it).
Once true, the node cannot opt out, be destroyed, or take part in
CopyAttributes; the graph cannot be instantiated twice or passed to
`cudaGraphExecUpdate`. A non-capturing `kernel_with` with that attr is Invalid
(graphs-only).
`expertvm sim --device-launch` / `--device-updatable` instantiate leaf GEMM
graphs with `DEVICE_LAUNCH` and skip re-upload after set-params.
`expertvm sim --kernel-priority N` is `cudaLaunchAttributePriority`.
`SharedMemoryMode` is `cudaLaunchAttributeSharedMemoryMode`: Default uses
`set_func_shared_mem_config` then `set_shared_mem_config`
(`cudaFuncSetSharedMemConfig` / `cudaDeviceSetSharedMemConfig`; unset never
scales); FourByte / EightByte scale by
`1000 / shared_mem_*_permille` (profile default 1000 is identity).
`expertvm sim --func-shared-mem eight` sets function EightByte so launch
Default inherits that duration scale (distinct from `--shared-mem`).
`expertvm sim --device-shared-mem eight` sets device EightByte so launch
Default inherits when function config is also Default.
`PortableClusterMode` is `cudaLaunchAttributePortableClusterSizeMode`: Default
uses the function attribute; RequirePortable always refuses oversize;
AllowNonPortable allows up to the SKU max. `set_non_portable_cluster_size_allowed` is
`cudaFuncAttributeNonPortableClusterSizeAllowed` (default disallowed).
`PortableSharedMode` is CUDA 13 `cudaLaunchAttributeSharedMemoryMode`
(`cudaSharedMemoryMode`), distinct from bank-width `SharedMemoryMode`.
`dynamic_shared` is `cudaLaunchKernel` `sharedMemBytes`. Default uses
`set_max_dynamic_shared_memory` (`cudaFuncAttributeMaxDynamicSharedMemorySize`;
`0` = portable `max_shared_mem_per_block`). Example H100 keeps portable ==
opt-in. `expertvm sim --cluster N` / `gguf_gemv engine --expert-sim --cluster N`
launch grouped expert GEMMs that way. `--preferred-cluster N` occupies the
preferred size when it fits (needs `--cluster`). `--cluster-spread` is Spread
scheduling (occupies every Hyper-Q slot). `--func-cluster-spread` is
`cudaFuncSetAttribute` ClusterSchedulingPolicyPreference Spread (launch
Default inherits; occupies every Hyper-Q slot when `--cluster` >= 2). `--cluster-must-set` is
`cudaFuncSetAttribute` ClusterDimMustBeSet (needs `--cluster`; occupancy matches
`--cluster`). `--max-shared` is MaxShared
carveout (occupies every Hyper-Q slot). `--func-max-shared` is
`cudaFuncSetAttribute` PreferredSharedMemoryCarveout MaxShared (launch
Default inherits; occupies every Hyper-Q slot). `--non-portable-cluster` is
`cudaFuncAttributeNonPortableClusterSizeAllowed`. `--sync-policy auto|spin|yield|blocking`
is `cudaLaunchAttributeSynchronizationPolicy` on created streams (host-wait
tax on `synchronize_stream`; Auto inherits `set_device_flags`, unset tax 0). `--device-sync-policy auto|spin|yield|blocking`
is `cudaSetDeviceFlags` SCHEDULE_* (Auto streams inherit that tax; explicit
`--sync-policy` wins). `--shared-mem default|four|eight` is
`cudaLaunchAttributeSharedMemoryMode` on grouped expert GEMMs (Default uses
function then device shared-mem config; unset never scales duration). `--portable-cluster default|portable|non-portable` is
`cudaLaunchAttributePortableClusterSizeMode` on grouped expert GEMMs (Default
uses the function attribute; `portable` always refuses oversize; `non-portable`
allows up to the SKU max). `--optin-shared` is
`cudaFuncAttributeMaxDynamicSharedMemorySize`. `--dynamic-shared N` is
`cudaLaunchKernel` `sharedMemBytes`. `--portable-shared default|portable|non-portable`
is CUDA 13 `cudaLaunchAttributeSharedMemoryMode`. `--nvlink-util` is
`cudaLaunchAttributeNvlinkUtilCentricScheduling` (`0`/`1`; occupies every
Hyper-Q slot when the profile has NVLink). `cudaLaunchAttributePriority` is
a per-kernel override of stream create priority (capture snapshots it;
default instantiate still uses the launch stream unless `USE_NODE_PRIORITY`).
`--device-launch` is
`cudaGraphInstantiateFlagDeviceLaunch` plus `device_launch_graph`
(gpu-sim named device-graph streams `GRAPH_FIRE_AND_FORGET` /
`GRAPH_TAIL_LAUNCH` / `GRAPH_FIRE_AND_FORGET_AS_SIBLING` have no Engine flag).
`--device-updatable` is `cudaLaunchAttributeDeviceUpdatableKernelNode`.
`--kernel-priority N` is `cudaLaunchAttributePriority`.
Decode identity stays `kernel`. `USE_NODE_PRIORITY` at
instantiate schedules those node priorities instead of the launch stream. `set_created_streams_priority` assigns created streams
`-id` (then clamps). `set_stream_sm_permille` is a duration-only SM fraction
(compute-bound kernels scale; memory-bound keep full HBM; default unset is
a full chip; does not partition Hyper-Q). CUDA green contexts bind an SM
span so complementary streams may overlap under exclusive compute.
`green_ctx_get_id` is `cuGreenCtxGetId` (not `GreenCtxId`).
`Operation` carries `submit_ns` / `start_ns` / `done_ns`
so stream[i+1].start ≥ stream[i].finish is inspectable. `GpuOp` /
`Operation` is the compiled submit DAG (`Sim::operations`).

In-flight ops are not cancelled. `gpu-profile capture` is refused in this
crate: someone with a GPU writes a `key=value` file; agents `parse` it.
