//! Deterministic GPU-systems VM for inference.
//!
//! Mechanical invariants (streams, events, HBM vs host-pinned residency, OOM,
//! topology, CUDA-graph capture / instantiate / update / replay) are exact.
//! Operation durations come from a [`HardwareProfile`], so agents cannot invent
//! hardware numbers inside policy code.
//!
//! This crate does not model warps, L1 caches, or Tensor Core pipelines.
//! Submitted work is a [`GpuOp`] compiled into an [`Operation`] DAG
//! ([`Sim::operations`]). [`Sim::synchronize_stream`] waits one stream
//! (`cudaStreamSynchronize`); an already-idle stream returns without starting
//! leftover kernels on other streams. [`GpuProfile::compute_slots`] is Hyper-Q
//! occupancy (`1` exclusive; `>=2` concurrent kernels at full issue rate).
//! [`Sim::set_stream_sync_policy`] is `cudaLaunchAttributeSynchronizationPolicy`
//! (stream-only; [`SynchronizationPolicy::Auto`] inherits
//! [`set_device_flags`](Sim::set_device_flags), unset tax 0). Host-wait tax on
//! [`synchronize_stream`](Sim::synchronize_stream) /
//! [`synchronize_event`](Sim::synchronize_event) comes from
//! [`GpuProfile::host_sync_spin_ns`] / `yield` / `blocking` (default 0).
//! [`Sim::synchronize`] / [`synchronize_device`](Sim::synchronize_device) do not
//! take that tax.
//! [`Sim::cooperative_kernel`] is `cudaLaunchCooperativeKernel`: it occupies
//! every compute slot so leftover kernels cannot Hyper-Q overlap it. Capture is
//! allowed (CUDA 11+). [`GpuProfile::cooperative_launch`] is
//! `cudaDevAttrCooperativeLaunch` (example H100 is true).
//! [`Sim::set_stream_sm_permille`] is a green-context SM fraction (compute-bound
//! kernels scale; memory-bound keep full HBM). Default unset is a full chip.
//! [`Sim::synchronize_device`] is `cudaDeviceSynchronize` (one GPU).
//! [`Sim::synchronize_event`] is `cudaEventSynchronize`.
//! [`Sim::alloc`] / [`memcpy`](Sim::memcpy) / [`free`](Sim::free) are
//! stream-ordered (`cudaMallocAsync` / `cudaMemcpyAsync` / `cudaFreeAsync`)
//! except pageable [`Place::Host`] copies, which wait the stream (CUDA bounce
//! buffer). [`PointerAttr::SyncMemops`] on the copy alloc also waits the stream.
//! [`Sim::memcpy_pinned_to_device`] is the overlapping DMA path.
//! [`Sim::malloc`] / [`memcpy_sync`](Sim::memcpy_sync) / [`free_sync`](Sim::free_sync)
//! / [`memset_sync`](Sim::memset_sync) are host-synchronous (`cudaMalloc` /
//! `cudaMemcpy` / `cudaFree` / `cudaMemset`). [`memset_op_sync`](Sim::memset_op_sync)
//! is `cudaMemset2D` / `cudaMemset3D`. Typed [`memset`](Sim::memset) /
//! [`memset_op`](Sim::memset_op) stay Async.
//! [`Sim::ipc_get`] / [`ipc_open`](Sim::ipc_open) / [`ipc_close`](Sim::ipc_close)
//! are `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` / `cudaIpcCloseMemHandle`:
//! the import is an alias of the same physicals (no extra HBM). Free of the
//! source while imports are live is Invalid. Capture cannot include IPC.
//! [`ipc_open_with_flags`](Sim::ipc_open_with_flags) accepts
//! [`IpcMemFlags::LAZY_ENABLE_PEER_ACCESS`] as a no-op (dest must already hold
//! the source; cross-GPU lazy peer is not modeled). Typed helper stays.
//! [`Sim::ipc_get`] of a `cudaMallocAsync` pointer is Invalid (`not device ipc`).
//! [`Sim::ipc_get_event`] / [`ipc_open_event`](Sim::ipc_open_event) are
//! `cudaIpcGetEventHandle` / `cudaIpcOpenEventHandle` (interprocess events).
//! [`Sim::create_shareable_pool`] is `cudaMemPoolCreate` with a POSIX-FD handle
//! type. [`Sim::pool_export`] / [`pool_import`](Sim::pool_import) are
//! `cudaMemPoolExportToShareableHandle` / `ImportFromShareableHandle`: the
//! import is a new [`PoolId`] that shares live/cached/threshold with the
//! exporter (no extra HBM). [`Sim::pool_export_ptr`] /
//! [`pool_import_ptr`](Sim::pool_import_ptr) are `cudaMemPoolExportPointer` /
//! `ImportPointer` (alias, no extra HBM). [`Sim::set_device_mempool`] is
//! `cudaDeviceSetMemPool`. [`default_pool`](Sim::default_pool) is
//! `cudaDeviceGetDefaultMemPool` (seeded; SetMemPool does not replace it).
//! [`device_mempool`](Sim::device_mempool) is `cudaDeviceGetMemPool`. Default and [`create_pool`](Sim::create_pool) pools
//! cannot be exported. Capture cannot include shareable export/import.
//! [`Sim::alloc`] draws from [`device_mempool`](Sim::device_mempool) (`cudaMallocAsync`).
//! [`Sim::create_pool`] / [`create_pool_with_props`](Sim::create_pool_with_props) /
//! [`alloc_from_pool`](Sim::alloc_from_pool) /
//! [`set_pool_release_threshold`](Sim::set_pool_release_threshold) /
//! [`pool_trim_to`](Sim::pool_trim_to) / [`pool_get_attribute`](Sim::pool_get_attribute) /
//! [`pool_set_attribute`](Sim::pool_set_attribute) / [`destroy_pool`](Sim::destroy_pool)
//! are `cudaMemPoolCreate` / `cudaMemPoolCreate`+[`MemPoolProps`] /
//! `cudaMallocFromPoolAsync` / `cudaMemPoolAttrReleaseThreshold` /
//! `cudaMemPoolTrimTo` / `cudaMemPoolGetAttribute` / `SetAttribute` /
//! `cudaMemPoolDestroy`.
//! [`MemPoolAttr`] is ReleaseThreshold / UsedMemCurrent / ReservedMemCurrent
//! plus reuse flags (default 1; only [`MemPoolAttr::ReuseAllowOpportunistic`]
//! `0` skips cache reuse — OS alloc, unused cached bytes stay reserved).
//! FollowEvent / Internal do not insert event waits or extra sync (no invented
//! pool high-water; graph mem stays [`GraphMemAttr`]). Unused
//! pool bytes stay in `cudaMemGetInfo` used until trim when the release
//! threshold is high (`u64::MAX`, vLLM-style). Destroying a user pool returns
//! unused cache to the OS; outstanding allocs stay valid; the default pool
//! cannot be destroyed; destroying the current pool rebinds GetMemPool to
//! GetDefaultMemPool.
//! [`Sim::pool_set_access`] / [`pool_unset_access`](Sim::pool_unset_access) /
//! [`pool_get_access`](Sim::pool_get_access) are `cudaMemPoolSetAccess`
//! ReadWrite / ProtNone and `cudaMemPoolGetAccess` (owner is ReadWrite by
//! default; peers need SetAccess). [`pool_set_access_with_flags`](Sim::pool_set_access_with_flags)
//! is the flags word ([`MemAccessFlags::PROT_READ_WRITE`] / [`PROT_NONE`](MemAccessFlags::PROT_NONE);
//! [`PROT_READ`](MemAccessFlags::PROT_READ) is Invalid `"pool prot read"`).
//! Typed helpers stay. A kernel on a peer may read
//! **and write** pool allocations without dest HBM (interconnect). Applies to
//! existing and later allocs from that pool. [`Sim::malloc`] cannot consume another pool's cache.
//! [`Sim::alloc_host`] is pageable; [`Sim::host_register`] / [`host_register_mapped`](Sim::host_register_mapped)
//! are `cudaHostRegister` (host-synchronous mlock). [`Sim::alloc_host_mapped`] is
//! `cudaHostAllocMapped`: a kernel may read it without H2D, billed at host PCIe.
//! [`Sim::alloc_managed`] is `cudaMallocManaged` (no HBM until first-touch or
//! [`Sim::prefetch`] / [`prefetch_host`](Sim::prefetch_host)). Default attach is
//! [`MemAttach::Global`]. [`Sim::alloc_managed_host`] is `cudaMemAttachHost`.
//! [`alloc_managed_with_flags`](Sim::alloc_managed_with_flags) is
//! `cudaMallocManaged` ([`MemAttachFlags::GLOBAL`] / [`HOST`](MemAttachFlags::HOST);
//! Single is Invalid). Typed helpers stay.
//! [`Sim::stream_attach`] is `cudaStreamAttachMemAsync` (stream-ordered;
//! Host and other-stream Single fail device kernels / memset / device prefetch
//! with `not attached`; Single cannot use [`StreamId::NULL`]; capture is
//! refused). [`stream_attach_with_flags`](Sim::stream_attach_with_flags) is
//! the flags word ([`MemAttachFlags`]). Typed helper stays. Prefetch migrates;
//! it does not replicate unless [`Sim::mem_advise`] [`MemAdvise::SetReadMostly`].
//! [`mem_advise_with_location`](Sim::mem_advise_with_location) is
//! `cudaMemAdvise_v2` ([`Place`] location; AccessedBy requires
//! [`Place::Device`]; host preferred is [`MemAdvise::SetPreferredLocationHost`]).
//! Typed [`mem_advise`](Sim::mem_advise) stays.
//! [`prefetch_with_flags`](Sim::prefetch_with_flags) is
//! `cudaMemPrefetchAsync` / `cuMemPrefetchAsync_v2` ([`PrefetchFlags::DEFAULT`]
//! only; [`Place::Device`] / host dest). Typed helpers stay.
//! [`MemAdvise::SetAccessedBy`] maps a GPU so a kernel can read without
//! migrating (billed on the interconnect, not local HBM). A kernel
//! page-faults managed memory onto that GPU when the kernel *starts*
//! (after stream deps) unless AccessedBy covers it.
//! [`MemAdvise::SetPreferredLocation`] keeps a page already at that GPU
//! there on a remote read (same interconnect billing; writes still migrate).
//! Host preferred does not skip a kernel first-touch.
//! [`Sim::mem_range_get_attribute`] is `cudaMemRangeGetAttribute` of modeled
//! per-alloc advice ([`MemRangeAttr`]; not per byte range; location type/id
//! wrap preferred and last-prefetch [`Place`]). Query; legal during capture.
//! [`mem_range_get_attributes`](Sim::mem_range_get_attributes) is
//! `cudaMemRangeGetAttributes` (batch of those attrs; all-or-nothing).
//! [`Sim::drop_managed_copy`] refunds one ReadMostly GPU copy (dest eviction).
//! [`Sim::va_reserve`] / [`va_map`](Sim::va_map) / [`va_unmap`](Sim::va_unmap) /
//! [`va_free`](Sim::va_free) are `cuMemAddressReserve` / `cuMemMap` /
//! `cuMemUnmap` / `cuMemAddressFree`. [`va_unmap_with_size`](Sim::va_unmap_with_size)
//! is the CUDA size argument (must match the reservation). [`va_free_with_size`](Sim::va_free_with_size)
//! is the CUDA size argument (must match the reservation). [`va_reserve_with_flags`](Sim::va_reserve_with_flags)
//! is alignment / addr / flags (flags 0; nonzero addr Invalid; nonzero
//! alignment must be a power of two that divides size). Size and map offsets must be multiples of
//! [`HardwareProfile::va_granularity_bytes`] (`0`/`1` = any size; example
//! default `1` keeps 4096-byte expert pages legal). [`Sim::va_map_range`] / [`va_unmap_range`](Sim::va_unmap_range)
//! map sparse physicals (vLLM KV-block analog); HBM is the mapped span.
//! [`Sim::va_create`] is `cuMemCreate` (charges HBM). [`va_create_with_prop`](Sim::va_create_with_prop)
//! is the prop + flags word (pinned device location; flags 0;
//! [`MemHandleType::NONE`] only). [`Sim::va_map_handle`] is
//! `cuMemMap` of that handle (no second HBM charge; two VAs may share it).
//! [`va_map_handle_with_flags`](Sim::va_map_handle_with_flags) is the flags
//! word (0). [`va_map_handle_with_size`](Sim::va_map_handle_with_size) is the
//! CUDA size argument (must match the handle). [`Sim::va_retain_handle`] is `cuMemRetainAllocationHandle` (handle refs;
//! combined `va_map` spans are promoted). [`Sim::va_release_handle`] is
//! `cuMemRelease` (allowed while mapped; HBM refunds when refs and maps are 0).
//! [`va_get_allocation_properties`](Sim::va_get_allocation_properties) is
//! `cuMemGetAllocationPropertiesFromHandle` ([`MemAllocationProp`]; pinned
//! device location; handle types always none; RDMA capable wraps the SKU).
//! [`Sim::va_get_allocation_granularity`] is
//! `cuMemGetAllocationGranularity` ([`MemAllocationGranularity::MINIMUM`] /
//! [`RECOMMENDED`](MemAllocationGranularity::RECOMMENDED) are the same
//! profile value; `0`/`1` → `1`). Both are queries; legal during capture.
//! [`Sim::va_map`] still Create+Maps in one call.
//! [`Sim::multicast_create`] / [`multicast_add_device`](Sim::multicast_add_device) /
//! [`multicast_bind_mem`](Sim::multicast_bind_mem) / [`multicast_bind_addr`](Sim::multicast_bind_addr) /
//! [`multicast_unbind`](Sim::multicast_unbind) /
//! [`multicast_destroy`](Sim::multicast_destroy) / [`va_map_multicast`](Sim::va_map_multicast)
//! are `cuMulticastCreate` / `AddDevice` / `BindMem` / `BindAddr` / `Unbind` /
//! `cuMemRelease` / `cuMemMap` of a multicast handle. [`va_map_multicast_with_flags`](Sim::va_map_multicast_with_flags)
//! requires flags 0 ([`MemMapFlags::DEFAULT`]). Typed helper stays.
//! [`va_map_multicast_with_size`](Sim::va_map_multicast_with_size) is the
//! CUDA size argument (must match the multicast object). [`multicast_unbind_with_size`](Sim::multicast_unbind_with_size)
//! is the CUDA size argument (must match the object; `mcOffset` 0). [`multicast_bind_mem_with_flags`](Sim::multicast_bind_mem_with_flags)
//! / [`multicast_bind_addr_with_flags`](Sim::multicast_bind_addr_with_flags)
//! require flags 0 ([`MulticastBindFlags::DEFAULT`]). Typed helpers stay.
//! [`multicast_bind_addr_with_size`](Sim::multicast_bind_addr_with_size)
//! is the CUDA size argument (must match the reserved VA; `mcOffset` 0).
//! [`multicast_bind_mem_with_size`](Sim::multicast_bind_mem_with_size)
//! is the CUDA size argument (must match the handle; `mcOffset` / `memOffset` 0).
//! [`multicast_get_granularity`](Sim::multicast_get_granularity)
//! is `cuMulticastGetGranularity` (minimum and recommended are the same
//! profile value; `0`/`1` → `1`). Query; legal during capture. The team must be an NVLink clique (PCIe P2P
//! and RDMA refuse). BindAddr ([`Sim::multicast_bind_addr`]) retains the
//! mapped VA's [`MemHandleId`] then BindMem (dest HBM is already charged).
//! A kernel write to the multicast VA is one NVLS hop on
//! compute, not N sequential copy-engine D2Ds. Capture cannot include
//! create/add/bind/unbind/destroy/map. [`Sim::multicast_store`] binds whole-VA maps of one
//! alloc and enqueues that kernel.
//! [`Sim::va_set_access`] is `cuMemSetAccess` PROT_READ on a peer (no dest HBM;
//! interconnect). [`va_set_access_write`](Sim::va_set_access_write) is
//! PROT_READWRITE (peer writes, no dest HBM). [`Sim::va_unset_access`] drops it.
//! [`va_set_access_with_flags`](Sim::va_set_access_with_flags) is the flags
//! word ([`MemAccessFlags`]). Typed helpers stay.
//! [`va_get_access`](Sim::va_get_access) is `cuMemGetAccess` (local map
//! ReadWrite; peer Read / ReadWrite / None). Query; legal during capture.
//! [`Sim::va_acquire`] remaps an idle VA of the same size (or reserves);
//! [`va_acquire_paged`](Sim::va_acquire_paged) maps it in KV-block spans;
//! [`va_release`](Sim::va_release) unmaps into that pool instead of freeing the VA.
//! [`Sim::kernel`] still needs the whole VA mapped; [`Sim::kernel_bufs`],
//! [`Sim::memset_buf`], and [`MemcpyOp::offset`] touch a mapped span (paged KV).
//! [`Sim::is_range_resident`] is that span check.
//! [`Sim::host_func`] is `cudaLaunchHostFunc` (stream-ordered host work; no GPU occupancy;
//! unnamed callback). [`host_func_params`](Sim::host_func_params) records
//! [`HostNodeParams`] (`cudaHostFn_t` / `userData`).
//! [`Sim::write_value64`] / [`write_value32`](Sim::write_value32) are
//! `cuStreamWriteValue64` / `WriteValue32` (mailbox updates on complete; no
//! compute/copy occupancy). [`wait_value64`](Sim::wait_value64) /
//! [`wait_value32`](Sim::wait_value32) are `cuStreamWaitValue64` / `WaitValue32`
//! ([`WaitValueCmp`]; unwritten locations read as 0; unsatisfied wait plus
//! [`Sim::synchronize`] deadlocks). [`batch_mem_op`](Sim::batch_mem_op) is
//! `cuStreamBatchMemOp` (one stream op; a wait sees earlier writes in that
//! vector). Kernel / memset / memcpy stores to the
//! mailbox address are not modeled. Device-resident and mapped-host are legal;
//! remote AccessedBy maps are not. Capture records a batch-mem-op node.
//! [`Sim::set_stream_blocking`] is `cudaStreamCreate` vs `cudaStreamNonBlocking`.
//! [`Sim::stream_create_with_flags`] / [`stream_create_with_priority`](Sim::stream_create_with_priority)
//! are `cudaStreamCreateWithFlags` / `CreateWithPriority`
//! ([`StreamCreateFlags::NON_BLOCKING`]; unknown bits Invalid). Capture cannot
//! include them. Typed [`set_stream_blocking`](Sim::set_stream_blocking) /
//! [`set_stream_priority`](Sim::set_stream_priority) stay.
//! [`HardwareProfile::host_pin_bytes`] caps `cudaMallocHost` / `cudaHostRegister`.
//! [`Sim::idle_until`] drains, then jumps the virtual clock (open-loop arrivals).
//! [`Sim::event_elapsed_ns`] is `cudaEventElapsedTime` in nanoseconds.
//! [`Sim::create_event_disable_timing`] is `cudaEventDisableTiming` (elapsed fails).
//! [`Sim::create_event_with_flags`] is `cudaEventCreateWithFlags`
//! ([`EventCreateFlags::DISABLE_TIMING`] / [`EventCreateFlags::INTERPROCESS`] /
//! [`EventCreateFlags::BLOCKING_SYNC`]; Interprocess requires DisableTiming).
//! [`Sim::create_event_blocking_sync`] is the BlockingSync helper
//! ([`synchronize_event`](Sim::synchronize_event) pays
//! [`GpuProfile::host_sync_blocking_ns`]).
//! [`Sim::create_event_interprocess`] is the Interprocess+DisableTiming helper.
//! [`Sim::ipc_get_event`] / [`ipc_open_event`](Sim::ipc_open_event) are
//! `cudaIpcGetEventHandle` / `cudaIpcOpenEventHandle`: the import aliases the
//! source record. Destroy of the source while imports are live is Invalid.
//! Capture cannot include event IPC.
//! [`Sim::record_event_with_flags`] / [`wait_event_with_flags`](Sim::wait_event_with_flags)
//! are `cudaEventRecordWithFlags` / `cudaStreamWaitEvent` flags
//! ([`EventRecordFlags::EXTERNAL`] / [`EventWaitFlags::EXTERNAL`]). Typed
//! helpers stay.
//! [`Sim::destroy_event`] is `cudaEventDestroy` (waits a recorded incomplete
//! event; never-recorded returns immediately; capture refused).
//! [`Sim::query_event`] is `cudaEventQuery` (no wait).
//! [`Sim::query_stream`] is `cudaStreamQuery` (no wait).
//! [`Sim::mem_info`] is `cudaMemGetInfo` `(free, total)`.
//! [`Sim::pointer_get_attributes`] is `cudaPointerGetAttributes`.
//! [`pointer_set_attribute`](Sim::pointer_set_attribute) /
//! [`pointer_get_attribute`](Sim::pointer_get_attribute) are
//! `cuPointerSetAttribute` / `GetAttribute` ([`PointerAttr`]: SyncMemops is
//! settable; MemoryType / DevicePointer / HostPointer / IsManaged /
//! RangeSize / Mapped / MemPoolHandle / DeviceOrdinal / RangeStartAddr /
//! BufferId / IsLegacyCudaIpcCapable / IsGpuDirectRdmaCapable /
//! AllowedHandleTypes / MappingBaseAddr / MappingSize /
//! IsHwDecompressCapable / MemoryBlockId are query-only
//! wrappers of existing pointer state; VMM mapping size is the
//! `cuMemMap` span at offset 0, not the reserved VA; hardware decompress
//! is always 0; memory-block id is the [`MemHandleId`] covering offset 0). Set is
//! capture-refused; Get is a query.
//! [`Sim::mem_get_address_range`] is `cudaMemGetAddressRange` (base is the
//! alloc id; interior offsets are not modeled). Query; legal during capture.
//! [`Sim::host_get_device_pointer`] is `cudaHostGetDevicePointer` (mapped host).
//! [`host_get_device_pointer_with_flags`](Sim::host_get_device_pointer_with_flags)
//! requires [`HostGetDevicePointerFlags::DEFAULT`] (`0`). Typed helper stays.
//! [`Sim::host_get_flags`] is `cudaHostGetFlags` (the stored
//! [`HostAllocFlags`] word).
//! [`Sim::alloc_host_with_flags`] / [`host_register_with_flags`](Sim::host_register_with_flags)
//! are `cudaHostAlloc` / `cudaHostRegister` flag words
//! ([`HostAllocFlags::MAPPED`] / [`HostAllocFlags::PORTABLE`] /
//! [`HostAllocFlags::WRITE_COMBINED`] on alloc; register
//! accepts Mapped / Portable). Portable / WriteCombined are stored (no DMA
//! change). IoMemory / ReadOnly are Invalid. Typed helpers stay.
//! [`Sim::device_get_attribute`] is `cudaDeviceGetAttribute` ([`DeviceAttr`]).
//! [`Sim::device_get_properties`] is `cudaGetDeviceProperties` ([`DeviceProperties`];
//! modeled caps only — no SM count or clock). [`Sim::device_get_name`] is
//! `cudaDeviceGetName` (the profile name). [`device_total_mem`](Sim::device_total_mem)
//! is `cuDeviceTotalMem` (HBM bytes). [`DeviceAttr::CanMapHostMemory`]
//! / [`DeviceAttr::ManagedMemory`] are always 1 (this VM has mapped host and
//! UM). [`DeviceAttr::ClusterLaunch`] is `max_blocks_per_cluster > 0`.
//! [`DeviceAttr::HostRegisterSupported`] / [`IpcEventSupport`](DeviceAttr::IpcEventSupport) /
//! [`CanUseHostPointerForRegisteredMem`](DeviceAttr::CanUseHostPointerForRegisteredMem)
//! are always 1. [`DeviceAttr::MemoryPoolSupportedHandleTypes`] is
//! [`MemHandleType::POSIX_FILE_DESCRIPTOR`]. [`DeviceAttr::GpuDirectRdmaSupported`]
//! / [`CanFlushRemoteWrites`](DeviceAttr::CanFlushRemoteWrites) are a GPU↔GPU
//! [`crate::LinkKind::Rdma`] link. [`Sim::flush_gpu_direct_rdma_writes`] is
//! `cudaDeviceFlushGPUDirectRDMAWrites` (1 ns host-sync barrier; capture refused;
//! write-ordering options are not modeled). [`DeviceAttr::HostRegisterReadOnlySupported`] /
//! [`PageableMemoryAccess`](DeviceAttr::PageableMemoryAccess) /
//! [`ConcurrentManagedAccess`](DeviceAttr::ConcurrentManagedAccess) /
//! [`DirectManagedMemAccessFromHost`](DeviceAttr::DirectManagedMemAccessFromHost) /
//! [`PageableMemoryAccessUsesHostPageTables`](DeviceAttr::PageableMemoryAccessUsesHostPageTables)
//! are always 0
//! (ReadOnly host register is Invalid; pageable is bounce-buffer; host cannot
//! touch managed while a kernel runs). [`DeviceAttr::HostNativeAtomicSupported`] /
//! [`CooperativeMultiDeviceLaunch`](DeviceAttr::CooperativeMultiDeviceLaunch) /
//! [`Integrated`](DeviceAttr::Integrated) are always 0 (host-mapped atomics and
//! multi-device cooperative are not modeled; example SKUs are discrete).
//! [`DeviceAttr::SparseCudaArraySupported`] /
//! [`DeferredMappingCudaArraySupported`](DeviceAttr::DeferredMappingCudaArraySupported) /
//! [`DmaBufSupported`](DeviceAttr::DmaBufSupported) are always 0 (CUDA arrays
//! and dma-buf are not modeled).
//! [`DeviceAttr::MulticastSupported`] is a GPU↔GPU [`crate::LinkKind::Nvlink`]
//! link on that device (PCIe P2P and RDMA are not NVLS).
//! [`DeviceAttr::VirtualMemoryManagementSupported`] is always 1 (this VM has
//! `cuMemAddressReserve`).
//! [`DeviceAttr::HandleTypePosixFileDescriptorSupported`] is always 1 (this VM
//! has POSIX-FD shareable pools).
//! [`DeviceAttr::GpuDirectRdmaFlushWritesOptions`] is
//! [`FlushGpuDirectRdmaWritesOptions::HOST`] on an RDMA SKU (MemOps is never
//! reported). [`DeviceAttr::GpuDirectRdmaWithCudaVMMSupported`] is the same
//! RDMA SKU bit (VMM is always on). [`DeviceAttr::GenericCompressionSupported`]
//! is always 0 (compression is not modeled).
//! [`DeviceAttr::HandleTypeWin32HandleSupported`] /
//! [`HandleTypeWin32KmtHandleSupported`](DeviceAttr::HandleTypeWin32KmtHandleSupported) /
//! [`HandleTypeFabricSupported`](DeviceAttr::HandleTypeFabricSupported) are
//! always 0 (this VM has POSIX-FD shareable pools; fabric handles are not
//! modeled).
//! [`DeviceAttr::HostMemoryPoolsSupported`] is always 0 (pools are
//! device-only; host location is Invalid).
//! [`DeviceAttr::IsMultiGpuBoard`] / [`MultiGpuBoardGroupID`](DeviceAttr::MultiGpuBoardGroupID)
//! are always 0 (example SKUs are discrete single-GPU packages).
//! [`DeviceAttr::StreamPrioritiesSupported`] / [`UnifiedAddressing`](DeviceAttr::UnifiedAddressing)
//! are always 1. [`DeviceAttr::GpuOverlap`] is `copy_engines > 0`. [`Sim::func_get_attributes`] is `cudaFuncGetAttributes` of modeled
//! per-device function attrs ([`FuncAttributes`]; not per kernel).
//! [`func_set_attribute`](Sim::func_set_attribute) /
//! [`func_get_attribute`](Sim::func_get_attribute) are `cudaFuncSetAttribute` /
//! `GetAttribute` ([`FuncAttr`]). Typed setters stay. Get is a query
//! (capture-legal); Set is host-side like those setters.
//! [`Sim::stream_get_flags`] is `cudaStreamGetFlags` (`0` blocking / `1`
//! NonBlocking; NULL follows [`legacy_null_stream`](Sim::legacy_null_stream)).
//! [`Sim::stream_get_priority`] is `cudaStreamGetPriority`.
//! [`Sim::stream_get_id`] is `cudaStreamGetId` (unique per device/stream;
//! not the caller-chosen [`StreamId`]).
//! [`stream_get_attribute`](Sim::stream_get_attribute) /
//! [`stream_set_attribute`](Sim::stream_set_attribute) are
//! `cudaStreamGetAttribute` / `SetAttribute` of existing stream state
//! ([`StreamAttr`]: priority, synchronization policy, mem-sync domain/map,
//! NVLink-util-centric). Green-context SM permille is not a CUDA stream
//! attribute. Type mismatch is Invalid `"stream attr"`. Get is a query
//! (capture-legal); Set is host-side like the dedicated setters.
//! [`Sim::device_count`] is `cudaGetDeviceCount`.
//! [`Sim::device_can_access_peer`] / [`device_get_p2p_attribute`](Sim::device_get_p2p_attribute)
//! are `cudaDeviceCanAccessPeer` / `cudaDeviceGetP2PAttribute` (topology links;
//! [`DeviceP2pAttr::AccessSupported`] and [`PerformanceRank`](DeviceP2pAttr::PerformanceRank)
//! from GPU↔GPU `bps`; [`DeviceP2pAttr::NativeAtomicSupported`] /
//! [`CudaArrayAccessFromDevice`](DeviceP2pAttr::CudaArrayAccessFromDevice)
//! are always 0).
//! [`enable_peer_with_flags`](Sim::enable_peer_with_flags) is
//! `cudaDeviceEnablePeerAccess` ([`PeerAccessFlags::DEFAULT`] only; typed
//! [`enable_peer`](Sim::enable_peer) stays; capture is legal).
//! [`memcpy_peer_async`](Sim::memcpy_peer_async) / [`memcpy_peer`](Sim::memcpy_peer)
//! are `cudaMemcpyPeerAsync` / `cudaMemcpyPeer` (hot replica; typed
//! [`memcpy_device_to_device`](Sim::memcpy_device_to_device) stays; Peer is
//! host-synchronous and capture-illegal).
//! [`memcpy_peer_3d_async`](Sim::memcpy_peer_3d_async) / [`memcpy_peer_3d`](Sim::memcpy_peer_3d)
//! are `cudaMemcpy3DPeerAsync` / `cudaMemcpy3DPeer` ([`MemcpyOp`] height/depth).
//! [`memcpy_peer_2d_async`](Sim::memcpy_peer_2d_async) / [`memcpy_peer_2d`](Sim::memcpy_peer_2d)
//! are `cudaMemcpy2DPeerAsync` / `cudaMemcpy2DPeer` ([`MemcpyOp`] height/pitches).
//! [`Sim::set_limit`] / [`get_limit`](Sim::get_limit) are `cudaDeviceSetLimit` /
//! `GetLimit`. [`DeviceLimit::PersistingL2CacheSize`] wraps
//! [`set_persisting_l2_cache_size`](Sim::set_persisting_l2_cache_size).
//! [`DeviceLimit::MaxL2FetchGranularity`] aligns access-policy windows (CUDA
//! default 128 on SM 8.0+). [`set_shared_mem_config`](Sim::set_shared_mem_config) /
//! [`get_shared_mem_config`](Sim::get_shared_mem_config) are
//! `cudaDeviceSetSharedMemConfig` / `GetSharedMemConfig` (Default kernels
//! inherit the function config, then this; unset is unscaled).
//! [`set_func_shared_mem_config`](Sim::set_func_shared_mem_config) /
//! [`get_func_shared_mem_config`](Sim::get_func_shared_mem_config) are
//! `cudaFuncSetSharedMemConfig` / `GetSharedMemConfig` (per device).
//! [`set_device_flags`](Sim::set_device_flags) /
//! [`get_device_flags`](Sim::get_device_flags) are `cudaSetDeviceFlags` /
//! `GetDeviceFlags` ([`DeviceFlags`] schedule plus stored MapHost /
//! LmemResizeToMax; [`DeviceFlags::SYNC_MEMOPS`] waits memcpy/memset like
//! pointer SyncMemops; Auto streams inherit the tax).
//! [`Sim::malloc_pitch`] is `cudaMallocPitch`.
//! [`MemcpyOp`] `height` / pitches are `cudaMemcpy2DAsync` (payload `width *
//! height`, not pitch padding). [`MemsetOp`] `height` / `pitch` are
//! `cudaMemset2DAsync` (payload `width * height`; padding is not written).
//! [`Sim::malloc_3d`] is `cudaMalloc3D`. [`MemcpyOp`] `depth` / slice heights
//! are `cudaMemcpy3DAsync` (payload `width * height * depth`). [`MemsetOp`]
//! `depth` / `ysize` are `cudaMemset3DAsync` (payload `width * height * depth`).
//! [`graph_mem_get`](Sim::graph_mem_get) / [`graph_mem_set`](Sim::graph_mem_set) /
//! [`graph_mem_trim`](Sim::graph_mem_trim) are `cudaDeviceGetGraphMemAttribute` /
//! `SetGraphMemAttribute` / `GraphMemTrim` (graph-memory pool only; unused
//! reserved bytes return on trim). [`graph_pool`](Sim::graph_pool) is that pool.
//! [`Sim::set_stream_priority`] is the priority-only helper;
//! [`stream_create_with_priority`](Sim::stream_create_with_priority) is
//! `cudaStreamCreateWithPriority` (flags + priority).
//! [`stream_copy_attributes`](Sim::stream_copy_attributes) is
//! `cudaStreamCopyAttributes` (priority, SM permille, mem-sync domain/map,
//! synchronization policy, and NVLink-util-centric scheduling).
//! [`Sim::set_created_streams_priority`] assigns created streams their id.
//! [`Sim::instantiate_graph`] is `cudaGraphInstantiate` (host-sync; returns a
//! new exec id; first [`launch_graph`](Sim::launch_graph) of a definition
//! creates a primary exec). [`Sim::instantiate_graph_auto_free`]
//! is `cudaGraphInstantiateFlagAutoFreeOnLaunch`.
//! [`instantiate_graph_with_flags`](Sim::instantiate_graph_with_flags) is
//! `cudaGraphInstantiateWithFlags` ([`GraphInstantiateFlags::UPLOAD`] uploads
//! during instantiate; [`GraphInstantiateFlags::USE_NODE_PRIORITY`] schedules
//! recorded kernels with the add/capture priority;
//! [`GraphInstantiateFlags::DEVICE_LAUNCH`] enables
//! [`device_launch_graph`](Sim::device_launch_graph) after upload — host
//! [`launch_graph`](Sim::launch_graph) stays legal; mem alloc/free, events,
//! child graphs, conditionals, and host nodes are Invalid).
//! [`instantiate_graph_with_params`](Sim::instantiate_graph_with_params) is
//! `cudaGraphInstantiateWithParams` ([`GraphInstantiateParams`] result and
//! err node). [`graph_exec_get_flags`](Sim::graph_exec_get_flags) is
//! `cudaGraphExecGetFlags`. Instantiate returns a new exec id (`cudaGraphExec_t`);
//! the source graph stays a definition. [`launch_graph`](Sim::launch_graph) of a
//! definition uses the primary exec. `cudaGraphExec*SetParams` accept either id.
//! [`graph_kernel_set_params`](Sim::graph_kernel_set_params) /
//! [`graph_memcpy_set_params`](Sim::graph_memcpy_set_params) /
//! [`graph_memcpy_set_params_1d`](Sim::graph_memcpy_set_params_1d) /
//! [`graph_memset_set_params`](Sim::graph_memset_set_params) /
//! [`graph_host_set_params`](Sim::graph_host_set_params) /
//! [`graph_batch_mem_op_set_params`](Sim::graph_batch_mem_op_set_params) /
//! [`graph_batch_mem_ops_set_params`](Sim::graph_batch_mem_ops_set_params) /
//! [`graph_event_record_set_event`](Sim::graph_event_record_set_event) /
//! [`graph_event_wait_set_event`](Sim::graph_event_wait_set_event) /
//! [`graph_child_set_params`](Sim::graph_child_set_params) are
//! `cudaGraphKernelNodeSetParams` / `MemcpyNodeSetParams` /
//! `MemcpyNodeSetParams1D` / `MemsetNodeSetParams`
//! / `HostNodeSetParams` / `BatchMemOpNodeSetParams` /
//! `EventRecordNodeSetEvent` / `EventWaitNodeSetEvent` /
//! `ChildGraphNodeSetParams`
//! on the graph and do not retarget an already-instantiated exec.
//! Child-graph definition SetParams may change nested topology; exec
//! SetParams still require matching topology. Event External flags stay
//! topology.
//! [`graph_kernel_node_get_priority`](Sim::graph_kernel_node_get_priority) /
//! [`graph_kernel_node_set_priority`](Sim::graph_kernel_node_set_priority) /
//! [`graph_kernel_node_copy_attributes`](Sim::graph_kernel_node_copy_attributes)
//! are `cudaGraphKernelNodeGetAttribute` / `SetAttribute` / `CopyAttributes`
//! for priority (`cudaLaunchAttributePriority` / `cudaKernelNodeAttributePriority`),
//! programmatic dependent launch ([`ProgrammaticLaunch`]),
//! programmatic event ([`ProgrammaticEvent`]), access-policy window,
//! mem-sync domain/map, cluster dimension, cluster scheduling policy,
//! preferred cluster dimension, shared-memory carveout,
//! device-updatable kernel node, shared-memory bank mode, and portable-cluster
//! size mode.
//! [`graph_kernel_node_get_attribute`](Sim::graph_kernel_node_get_attribute) /
//! [`graph_exec_kernel_node_get_attribute`](Sim::graph_exec_kernel_node_get_attribute) /
//! [`graph_kernel_node_set_attribute`](Sim::graph_kernel_node_set_attribute) /
//! [`graph_exec_kernel_node_set_attribute`](Sim::graph_exec_kernel_node_set_attribute)
//! are the generic `cudaGraphKernelNodeGetAttribute` / `SetAttribute`
//! ([`KernelNodeAttr`]). Typed getters stay. Definition Set does not retarget
//! exec. Attr/value mismatch is Invalid `"kernel node attr"`. Get is a query
//! (capture-legal); Set cannot include capture.
//! [`kernel_pdl`](Sim::kernel_pdl) is `cudaLaunchKernelEx` PDL: a wait kernel
//! may start after the previous same-stream kernel's trigger
//! (`GpuProfile::pdl_trigger_permille`) instead of its completion. Overlap
//! needs `compute_slots >= 2`. [`kernel_pdl_event`](Sim::kernel_pdl_event) is
//! `cudaLaunchAttributeProgrammaticEvent`: other streams may wait that event
//! at the trigger instead of kernel completion. [`kernel_launch_completion`](Sim::kernel_launch_completion)
//! is `cudaLaunchAttributeLaunchCompletionEvent`: the event records when the
//! kernel *starts*. [`kernel_access_policy`](Sim::kernel_access_policy) is
//! `cudaLaunchAttributeAccessPolicyWindow`: persisting hits reduce billed HBM
//! after [`set_persisting_l2_cache_size`](Sim::set_persisting_l2_cache_size)
//! (CUDA default size is 0). [`kernel_with`](Sim::kernel_with) also accepts
//! [`MemSyncDomain`] / [`MemSyncDomainMap`] (`cudaLaunchAttributeMemSyncDomain`):
//! a completing kernel's implicit fence waits
//! `GpuProfile::same_domain_fence_permille` of leftover traffic from another
//! same-physical-domain kernel. [`MemSyncDomain::Remote`] (and allreduce, which
//! tags Remote like NCCL) isolates that traffic. Example H100
//! `mem_sync_domain_count` is 4; tax default is 0 (identity).
//! [`ClusterDim`] (`cudaLaunchAttributeClusterDimension`) occupies
//! `min(blocks, compute_slots)` Hyper-Q slots (Hopper portable max 8).
//! [`ClusterSchedulingPolicy::Spread`] occupies every slot.
//! [`ClusterSchedulingPolicy::Default`] uses [`set_func_cluster_policy`](Sim::set_func_cluster_policy)
//! (`cudaFuncAttributeClusterSchedulingPolicyPreference`).
//! [`KernelAttrs::preferred_cluster`] is used when that size fits in
//! `compute_slots`. [`SharedMemCarveout::MaxShared`] occupies every slot.
//! [`SharedMemCarveout::Default`] uses [`set_func_carveout`](Sim::set_func_carveout)
//! (`cudaFuncAttributePreferredSharedMemoryCarveout`).
//! Sizes above [`GpuProfile::portable_cluster_size`] need
//! [`set_non_portable_cluster_size_allowed`](Sim::set_non_portable_cluster_size_allowed)
//! or [`PortableClusterMode::AllowNonPortable`].
//! [`KernelAttrs::device_updatable`] is `cudaLaunchAttributeDeviceUpdatableKernelNode`
//! (graphs-only; a non-capturing launch is Invalid). When true,
//! [`graph_exec_kernel_set_params`](Sim::graph_exec_kernel_set_params) keeps the
//! exec uploaded so [`device_launch_graph`](Sim::device_launch_graph) needs no host
//! re-upload (device-launch graphs allow it).
//! [`SharedMemoryMode`] is `cudaLaunchAttributeSharedMemoryMode`: Default uses
//! [`set_func_shared_mem_config`](Sim::set_func_shared_mem_config) then
//! [`set_shared_mem_config`](Sim::set_shared_mem_config)
//! (`cudaFuncSetSharedMemConfig` / `cudaDeviceSetSharedMemConfig`; unset
//! never scales); FourByte / EightByte scale by
//! `1000 / GpuProfile::shared_mem_*_permille` (profile default `1000`).
//! [`PortableClusterMode`] is `cudaLaunchAttributePortableClusterSizeMode`:
//! Default uses the function attribute; RequirePortable always refuses a
//! non-portable size; AllowNonPortable allows up to `max_blocks_per_cluster`.
//! [`KernelAttrs::dynamic_shared`] is `cudaLaunchKernel` `sharedMemBytes`.
//! Sizes above [`GpuProfile::max_shared_mem_per_block`] need
//! [`set_max_dynamic_shared_memory`](Sim::set_max_dynamic_shared_memory) or
//! [`PortableSharedMode::AllowNonPortable`]. [`PortableSharedMode`] is CUDA 13
//! `cudaLaunchAttributeSharedMemoryMode` (`cudaSharedMemoryMode`), distinct
//! from bank-width [`SharedMemoryMode`].
//! [`KernelAttrs::nvlink_util_centric`] is
//! `cudaLaunchAttributeNvlinkUtilCentricScheduling` (`0`/`1`). CUDA treats it
//! as a hint; this VM occupies every Hyper-Q slot when the profile has NVLink.
//! Decode identity stays disabled. Stream SetAttribute is inherited by
//! [`Sim::kernel`]. Graph CopyAttributes copies it. Device-launch graphs allow
//! it.
//! [`KernelAttrs::priority`] is `cudaLaunchAttributePriority`: `None` inherits
//! the stream (`cudaStreamCreateWithPriority`); `Some` overrides for that kernel
//! when compute contends (higher first). Capture snapshots the effective value.
//! Default instantiate still uses the launch stream unless
//! [`GraphInstantiateFlags::USE_NODE_PRIORITY`]. Device-launch graphs allow it.
//! Decode identity stays [`Sim::kernel`] (inherit stream). [`graph_exec_kernel_node_get_priority`](Sim::graph_exec_kernel_node_get_priority) /
//! [`graph_exec_kernel_node_set_priority`](Sim::graph_exec_kernel_node_set_priority)
//! are the exec-snapshot attributes. [`Sim::upload_graph`] is
//! `cudaGraphUpload` (host-sync; first launch after instantiate calls it).
//! [`Sim::update_graph`] is
//! `cudaGraphExecUpdate` when device, stream, and op kinds match.
//! [`update_graph_with_info`](Sim::update_graph_with_info) fills
//! [`GraphExecUpdateResultInfo`] even on `Err` (node type, deps, mem nodes,
//! device-launch).
//! [`user_object_create`](Sim::user_object_create) is `cudaUserObjectCreate`
//! ([`UserObjectFlags::NO_DESTRUCTOR_SYNC`]). [`graph_retain_user_object`](Sim::graph_retain_user_object) /
//! [`graph_release_user_object`](Sim::graph_release_user_object) are
//! `cudaGraphRetainUserObject` / `ReleaseUserObject` on a definition.
//! [`GraphUserObjectFlags::MOVE`] transfers one caller ref. Destroy of the
//! last reference records [`user_object_destructors`](Sim::user_object_destructors).
//! Clone does not copy retains. Capture cannot include it.
//! [`Sim::graph_exec_kernel_set_params`] is `cudaGraphExecKernelNodeSetParams`:
//! patch one instantiated kernel node's pointers / [`KernelKind`] without a
//! second graph (`graph_set_params_ns`). Cooperative flag and edges stay.
//! Clears the upload flag unless the node is device-updatable
//! (`cudaLaunchAttributeDeviceUpdatableKernelNode`). Works on graphs with mem
//! alloc/free nodes (CUDA cannot `cudaGraphExecUpdate` those). Capture cannot
//! include it.
//! [`graph_exec_memcpy_set_params`](Sim::graph_exec_memcpy_set_params) /
//! [`graph_exec_memcpy_set_params_1d`](Sim::graph_exec_memcpy_set_params_1d)
//! are `cudaGraphExecMemcpyNodeSetParams` / `SetParams1D` (same
//! `graph_set_params_ns`; pageable still illegal; mem nodes legal; 1D may
//! convert a 2D/3D node). [`graph_unique_memcpy`](Sim::graph_unique_memcpy)
//! / [`graph_try_unique_memcpy`](Sim::graph_try_unique_memcpy) find that node.
//! [`graph_exec_memset_set_params`](Sim::graph_exec_memset_set_params) is
//! `cudaGraphExecMemsetNodeSetParams` (same cost; zero-byte still illegal).
//! [`graph_unique_memset`](Sim::graph_unique_memset) /
//! [`graph_try_unique_memset`](Sim::graph_try_unique_memset) find that node.
//! [`graph_exec_host_set_params`](Sim::graph_exec_host_set_params) is
//! `cudaGraphExecHostNodeSetParams` (same cost; [`HostNodeParams`] are
//! parameters). [`graph_unique_host`](Sim::graph_unique_host) /
//! [`graph_try_unique_host`](Sim::graph_try_unique_host) find that node.
//! [`graph_kernel_get_params`](Sim::graph_kernel_get_params) /
//! [`graph_memcpy_get_params`](Sim::graph_memcpy_get_params) /
//! [`graph_memset_get_params`](Sim::graph_memset_get_params) /
//! [`graph_host_get_params`](Sim::graph_host_get_params) /
//! [`graph_batch_mem_ops_get_params`](Sim::graph_batch_mem_ops_get_params) are
//! `cudaGraph*NodeGetParams` on the definition.
//! [`graph_exec_kernel_get_params`](Sim::graph_exec_kernel_get_params) /
//! [`graph_exec_memcpy_get_params`](Sim::graph_exec_memcpy_get_params) /
//! [`graph_exec_memset_get_params`](Sim::graph_exec_memset_get_params) /
//! [`graph_exec_host_get_params`](Sim::graph_exec_host_get_params) /
//! [`graph_exec_batch_mem_ops_get_params`](Sim::graph_exec_batch_mem_ops_get_params)
//! read the exec snapshot.
//! [`graph_exec_batch_mem_op_set_params`](Sim::graph_exec_batch_mem_op_set_params)
//! is `cudaGraphExecBatchMemOpNodeSetParams` (id/offset/value; wait vs write,
//! `bits32`, and compare stay on wait/write nodes;
//! [`graph_exec_batch_mem_ops_set_params`](Sim::graph_exec_batch_mem_ops_set_params)
//! replaces a [`GpuOp::BatchMem`] item list). [`graph_unique_write_value`](Sim::graph_unique_write_value)
//! / [`graph_unique_wait_value`](Sim::graph_unique_wait_value) find those nodes.
//! [`graph_exec_child_set_params`](Sim::graph_exec_child_set_params) is
//! `cudaGraphExecChildGraphNodeSetParams`: swap the nested graph of one
//! instantiated child-graph node (`graph_set_params_ns`; nested topology must
//! match; nested child ids are parameters; mem nodes legal). `cudaGraphExecUpdate`
//! treats child ids as topology. [`graph_child_nodes`](Sim::graph_child_nodes) /
//! [`graph_unique_child`](Sim::graph_unique_child) /
//! [`graph_try_unique_child`](Sim::graph_try_unique_child) find those nodes.
//! [`graph_child_get_graph`](Sim::graph_child_get_graph) is
//! `cudaGraphChildGraphNodeGetGraph`. [`graph_event_record_get_event`](Sim::graph_event_record_get_event) /
//! [`graph_event_wait_get_event`](Sim::graph_event_wait_get_event) are
//! `cudaGraphEventRecordNodeGetEvent` / `WaitNodeGetEvent`.
//! [`graph_alloc_get_params`](Sim::graph_alloc_get_params) is
//! `cudaGraphMemAllocNodeGetParams` (stored id and bytes).
//! [`graph_free_get_params`](Sim::graph_free_get_params) is
//! `cudaGraphMemFreeNodeGetParams` (stored [`AllocId`]).
//! [`graph_free_set_params`](Sim::graph_free_set_params) /
//! [`graph_exec_free_set_params`](Sim::graph_exec_free_set_params) are
//! `cudaGraphMemFreeNodeSetParams` / `cudaGraphExecMemFreeNodeSetParams`
//! (definition SetParams does not retarget an exec; `graph_allocs` stays
//! alloc-node ids).
//! [`graph_exec_event_record_set_event`](Sim::graph_exec_event_record_set_event) /
//! [`graph_exec_event_wait_set_event`](Sim::graph_exec_event_wait_set_event) are
//! `cudaGraphExecEventRecordNodeSetEvent` / `WaitNodeSetEvent` (event id is the
//! parameter; External flag is topology).
//! [`graph_node_set_enabled`](Sim::graph_node_set_enabled) /
//! [`graph_node_get_enabled`](Sim::graph_node_get_enabled) are
//! `cudaGraphNodeSetEnabled` / `GetEnabled` (skip launch; mem nodes illegal).
//! [`Sim::clone_graph`] is `cudaGraphClone` (independent, not instantiated;
//! child-graph nodes are cloned recursively).
//! [`Sim::create_graph`] is `cudaGraphCreate` (empty, not instantiated).
//! [`Sim::graph_add_kernel`] / [`graph_add_memcpy`](Sim::graph_add_memcpy) /
//! [`graph_add_memcpy_1d`](Sim::graph_add_memcpy_1d) /
//! [`graph_add_memset`](Sim::graph_add_memset) /
//! [`graph_add_memset_op`](Sim::graph_add_memset_op) /
//! [`graph_add_host_func`](Sim::graph_add_host_func) /
//! [`graph_add_empty`](Sim::graph_add_empty) /
//! [`graph_add_event_record`](Sim::graph_add_event_record) /
//! [`graph_add_event_wait`](Sim::graph_add_event_wait) /
//! [`graph_add_write_value64`](Sim::graph_add_write_value64) /
//! [`graph_add_wait_value64`](Sim::graph_add_wait_value64) /
//! [`graph_add_batch_mem_op`](Sim::graph_add_batch_mem_op) /
//! [`batch_mem_op`](Sim::batch_mem_op) /
//! [`graph_add_child`](Sim::graph_add_child) /
//! [`graph_add_alloc`](Sim::graph_add_alloc) /
//! [`graph_add_free`](Sim::graph_add_free) /
//! [`graph_add_cooperative_kernel`](Sim::graph_add_cooperative_kernel) /
//! [`graph_add_dependencies`](Sim::graph_add_dependencies) are
//! `cudaGraphAdd*` on that id.
//! [`graph_add_node`](Sim::graph_add_node) is `cudaGraphAddNode`
//! ([`GraphNodeParams`] plus dependency indices in the same call). Typed
//! `graph_add_*` stay (empty deps). IF/WHILE/SWITCH stay
//! [`graph_add_if`](Sim::graph_add_if) / `graph_add_while` / `graph_add_switch`.
//! Add is illegal on an instantiated exec and during capture.
//! [`graph_node_set_params`](Sim::graph_node_set_params) /
//! [`graph_exec_node_set_params`](Sim::graph_exec_node_set_params) are
//! `cudaGraphNodeSetParams` / `cudaGraphExecNodeSetParams` (dispatch to the
//! typed SetParams; Alloc would resize HBM; Empty has no params).
//! [`graph_node_get_params`](Sim::graph_node_get_params) /
//! [`graph_exec_node_get_params`](Sim::graph_exec_node_get_params) are
//! `cudaGraphNodeGetParams` on the definition / exec snapshot (query; Empty
//! returns [`GraphNodeParams::Empty`]; Alloc is bytes only).
//! [`Sim::graph_add_alloc`] / [`graph_add_free`](Sim::graph_add_free) are
//! `cudaGraphAddMemAllocNode` / `cudaGraphAddMemFreeNode` (same reuse /
//! AutoFreeOnLaunch rules as captured `cudaMallocAsync`).
//! [`Sim::graph_add_dependencies`] is `cudaGraphAddDependencies` (node indices;
//! independent nodes may Hyper-Q overlap at launch; capture records same-stream
//! edges). [`graph_add_dependencies_n`](Sim::graph_add_dependencies_n) /
//! [`graph_remove_dependencies_n`](Sim::graph_remove_dependencies_n) are the
//! same APIs with `numDependencies` from/to pairs (all-or-nothing). [`graph_remove_dependencies`](Sim::graph_remove_dependencies) is
//! `cudaGraphRemoveDependencies` (illegal on an exec and during capture).
//! [`graph_destroy_node`](Sim::graph_destroy_node) is `cudaGraphDestroyNode`
//! (drops incident edges; remaining indices stay valid; illegal on an exec and
//! during capture; does not retarget an already-instantiated exec).
//! `cudaGraphExecUpdate` treats those edges as topology.
//! [`graph_nodes`](Sim::graph_nodes) / [`graph_root_nodes`](Sim::graph_root_nodes) /
//! [`graph_edges`](Sim::graph_edges) /
//! [`graph_node_dependents`](Sim::graph_node_dependents) /
//! [`graph_debug_dot`](Sim::graph_debug_dot) /
//! [`graph_debug_dot_with_flags`](Sim::graph_debug_dot_with_flags) are
//! `cudaGraphGetNodes` / `GetRootNodes` / `GetEdges` / `NodeGetDependentNodes`
//! / `cudaGraphDebugDotPrint` (live nodes; flags `0` is kinds and edges;
//! [`GraphDebugDotFlags::VERBOSE`] prints modeled params). Destroyed slots are
//! omitted. [`begin_capture_to_graph`](Sim::begin_capture_to_graph) is
//! `cudaStreamBeginCaptureToGraph`: append captured nodes onto an existing
//! uninstantiated graph; capture roots additionally depend on the given node
//! indices (empty means extra roots). [`Sim::end_capture`] returns that graph.
//! [`stream_update_capture_dependencies`](Sim::stream_update_capture_dependencies)
//! is `cudaStreamUpdateCaptureDependencies`: extra deps for the next captured
//! node **in addition to** stream-order (`Set` replaces, `Add` unions).
//! [`stream_is_capturing`](Sim::stream_is_capturing) /
//! [`stream_capture_info`](Sim::stream_capture_info) are `cudaStreamIsCapturing`
//! / `cudaStreamGetCaptureInfo`. [`StreamCaptureInfo::dependencies`] is the v2
//! array (last same-stream captured node union extra pending deps).
//! [`StreamCaptureInfo::id`] is `id_out` (unique per begin-capture sequence;
//! forked streams share it). [`begin_capture_with_mode`](Sim::begin_capture_with_mode)
//! is `cudaStreamBeginCapture` with [`StreamCaptureMode`] (default
//! [`StreamCaptureMode::Relaxed`]: independent streams stay live; a wait of a
//! captured record still joins. [`StreamCaptureMode::ThreadLocal`] /
//! [`StreamCaptureMode::Global`] refuse uncaptured-stream submits).
//! [`thread_exchange_stream_capture_mode`](Sim::thread_exchange_stream_capture_mode)
//! is `cudaThreadExchangeStreamCaptureMode`. [`graph_node_kind`](Sim::graph_node_kind) is
//! `cudaGraphNodeGetType`.
//! [`graph_conditional_create`](Sim::graph_conditional_create) /
//! [`graph_add_if`](Sim::graph_add_if) are `cudaGraphConditionalHandleCreate`
//! and an IF node (`cudaGraphCondTypeIf`). Body ops skip at start when the
//! handle is `0`. [`set_conditional`](Sim::set_conditional) is device
//! `cudaGraphSetConditional` (capture allowed; each launch resets to the
//! create-time default first). [`graph_add_while`](Sim::graph_add_while) /
//! [`graph_add_switch`](Sim::graph_add_switch) are `cudaGraphCondTypeWhile`
//! / `Switch` (WHILE caps at 64 iterations; SWITCH branch `i` runs when the
//! handle equals `i`). [`graph_if_nodes`](Sim::graph_if_nodes) /
//! [`graph_while_nodes`](Sim::graph_while_nodes) /
//! [`graph_switch_nodes`](Sim::graph_switch_nodes) list those nodes.
//! [`graph_node_find_in_clone`](Sim::graph_node_find_in_clone) is
//! `cudaGraphNodeFindInClone` (same index on a graph produced by
//! [`clone_graph`](Sim::clone_graph) of that original).
//! [`Sim::destroy_graph`] is `cudaGraphDestroy` / `cudaGraphExecDestroy`.
//! Capture records every stream that [`wait_event`](Sim::wait_event)s an
//! event recorded in this capture (CUDA forked capture). [`record_event_external`](Sim::record_event_external)
//! / [`wait_event_external`](Sim::wait_event_external) do not join.
//! Independent streams stay live. Launch remaps origin-stream nodes onto the launch stream.
//! [`launch_graph`](Sim::launch_graph) during capture records a child graph
//! ([`GpuOp::ChildGraph`]) if the child is already instantiated.
//! [`Sim::alloc`] / [`free`](Sim::free) during capture are graph mem alloc/free
//! nodes (`cudaMallocAsync` / `cudaFreeAsync`). Host-sync [`malloc`](Sim::malloc)
//! / [`free_sync`](Sim::free_sync) / VMM / [`create_pool`](Sim::create_pool) cannot
//! be captured. A graph that allocates without a matching free reuses the pointer
//! on later launches (no second HBM charge) unless instantiated with
//! [`instantiate_graph_auto_free`](Sim::instantiate_graph_auto_free)
//! (`cudaGraphInstantiateFlagAutoFreeOnLaunch`). [`clone_graph`](Sim::clone_graph)
//! forks those ids. [`destroy_graph`](Sim::destroy_graph) returns remaining graph
//! mem to the device graph-memory pool (unused reserved stays until
//! [`graph_mem_trim`](Sim::graph_mem_trim)). [`update_graph`](Sim::update_graph) of mem nodes is Invalid.
//! [`graph_exec_kernel_set_params`](Sim::graph_exec_kernel_set_params) may still
//! retarget a kernel node in those graphs.

#![cfg_attr(not(test), deny(missing_docs))]

mod error;
mod ids;
mod ops;
mod probe;
mod profile;
mod score;
mod sim;

pub use error::SimError;
pub use ids::{
    AllocId, CondId, DeviceId, EventId, GraphId, IpcEventHandleId, IpcHandleId, LinkId,
    MemHandleId, MulticastId, OpId, PoolId, PtrExportId, ShareableHandleId, StreamId, UserObjectId,
};
pub use ops::{
    parse_nvlink_util_centric, AccessPolicyWindow, AccessProperty, BatchMemOp, CaptureDepOp,
    ClusterDim, ClusterSchedulingPolicy, DType, DeviceAttr, DeviceFlags, DeviceLimit,
    DeviceP2pAttr, DeviceProperties, EventCreateFlags, EventRecordFlags, EventWaitFlags,
    FlushGpuDirectRdmaScope, FlushGpuDirectRdmaTarget, FlushGpuDirectRdmaWritesOptions, FuncAttr,
    FuncAttributes, GpuOp, GraphAddNode, GraphDebugDotFlags, GraphExecUpdateResult,
    GraphExecUpdateResultInfo, GraphInstantiateFlags, GraphInstantiateParams,
    GraphInstantiateResult, GraphMemAttr, GraphNodeKind, GraphNodeParams, GraphUserObjectFlags,
    HostAllocFlags, HostGetDevicePointerFlags, HostNodeParams, IpcMemFlags, KernelAttrs, KernelBuf,
    KernelKind, KernelNodeAttr, KernelNodeAttrValue, KernelNodeParams, LaunchCompletionEvent,
    MemAccessFlags, MemAdvise, MemAllocationGranularity, MemAllocationProp, MemAllocationType,
    MemAttach, MemAttachFlags, MemCreateFlags, MemHandleType, MemLocationType, MemMapFlags,
    MemPoolAttr, MemPoolProps, MemRangeAttr, MemRangeAttrValue, MemReserveFlags, MemSyncDomain,
    MemSyncDomainMap, MemcpyOp, MemoryType, MemsetOp, MulticastBindFlags, MulticastGranularity,
    Operation, PdlLaunch, PeerAccessFlags, Place, PointerAttr, PointerAttributes,
    PortableClusterMode, PortableSharedMode, PrefetchFlags, ProgrammaticEvent, ProgrammaticLaunch,
    SharedMemCarveout, SharedMemoryMode, StreamAttr, StreamAttrValue, StreamCaptureInfo,
    StreamCaptureMode, StreamCreateFlags, SynchronizationPolicy, UserObjectFlags, WaitValueCmp,
};
pub use probe::{probe_topology, P2pProbe, TopologyProbe};
pub use profile::{
    align_up, ns_for_bytes, scale_ns_permille, GpuProfile, HardwareProfile, LinkKind, LinkProfile,
};
pub use score::Score;
pub use sim::Sim;

#[cfg(test)]
mod tests {
    use super::*;

    fn h100() -> HardwareProfile {
        HardwareProfile::example_h100_sxm()
    }

    fn enq(r: Result<OpId, SimError>) {
        assert!(r.expect("enqueue").0 >= 1);
    }

    #[test]
    fn alloc_kernel_sync_advances_clock() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 1 << 20, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 20, 1 << 20), &[a], &[a], s));
        assert_eq!(sim.op_stream(OpId(1)), Some(s));
        sim.synchronize().unwrap();
        assert!(sim.clock_ns() > 0);
        assert!(sim.is_resident(a, d).unwrap());
    }

    #[test]
    fn oom_is_semantic_failure() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let huge = sim.profile().gpu(d).unwrap().hbm_bytes.saturating_add(1);
        let err = sim.alloc(d, huge, s).unwrap();
        let out = sim.synchronize();
        assert!(out.is_err(), "id={err:?} {out:?}");
        match out.unwrap_err() {
            SimError::Oom { .. } => {}
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn kernel_before_residency_fails() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s0 = StreamId(0);
        let s1 = StreamId(1);
        let a = sim.alloc(d, 4096, s0).unwrap();
        // Kernel on a different stream, no event: may race. Force the race by
        // submitting the kernel first on s1 against an alloc that has not started.
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s1));
        let err = sim.synchronize();
        assert!(err.is_err());
    }

    #[test]
    fn event_orders_streams() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s0 = StreamId(0);
        let s1 = StreamId(1);
        let ev = EventId(1);
        let a = sim.alloc(d, 4096, s0).unwrap();
        enq(sim.record_event(d, ev, s0));
        enq(sim.wait_event(d, ev, s1));
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s1));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(a, d).unwrap());
    }

    #[test]
    fn free_while_leased_fails() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
        enq(sim.record_event(d, EventId(1), StreamId(0)));
        enq(sim.kernel(d, KernelKind::other(1 << 40, 4096), &[a], &[a], StreamId(0)));
        enq(sim.wait_event(d, EventId(1), StreamId(1)));
        sim.free(d, a, StreamId(1)).unwrap();
        let err = sim.synchronize();
        match err {
            Err(SimError::Leased { .. }) => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn h2d_then_kernel_hides_nothing_without_overlap() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let a = sim.alloc(d, bytes, s).unwrap();
        enq(sim.memcpy_host_to_device(d, a, bytes, s));
        enq(sim.kernel(d, KernelKind::other(1 << 20, bytes), &[a], &[a], s));
        sim.synchronize().unwrap();
        assert!(sim.bytes_moved() >= bytes);
        assert!(sim.clock_ns() > 0);
    }

    #[test]
    fn copy_overlaps_compute_on_two_streams() {
        let mut serial = Sim::new(h100());
        let mut overlapped = Sim::new(h100());
        let d = DeviceId(0);
        let bytes = 64u64 << 20;
        let k = KernelKind::GroupedMoeGemm {
            experts: 4,
            tokens_per_expert: 8,
            hidden: 1024,
            ff: 1024,
            dtype: DType::Fp16,
        };

        let a = serial.alloc(d, bytes, StreamId(0)).unwrap();
        enq(serial.memcpy_pinned_to_device(d, a, bytes, StreamId(0)));
        enq(serial.kernel(d, k.clone(), &[a], &[a], StreamId(0)));
        serial.synchronize().unwrap();

        let b = overlapped.alloc(d, bytes, StreamId(0)).unwrap();
        enq(overlapped.record_event(d, EventId(1), StreamId(0)));
        enq(overlapped.memcpy_pinned_to_device(d, b, bytes, StreamId(0)));
        // Same stream still serializes copy then compute. True overlap needs
        // the kernel on a buffer already resident while a *different* buffer copies.
        let c = overlapped.alloc(d, bytes, StreamId(1)).unwrap();
        enq(overlapped.wait_event(d, EventId(1), StreamId(1)));
        enq(overlapped.kernel(d, k, &[b], &[b], StreamId(1)));
        enq(overlapped.memcpy_pinned_to_device(d, c, bytes, StreamId(0)));
        overlapped.synchronize().unwrap();
        assert!(overlapped.clock_ns() > 0);
        assert!(serial.clock_ns() > 0);
        assert!(!Score::from_sim(&serial).line().is_empty());
        assert!(!Score::from_sim(&overlapped).line().is_empty());
    }

    #[test]
    fn tiny_copies_cannot_beat_one_large_copy() {
        let mut big = Sim::new(h100());
        let mut tiny = Sim::new(h100());
        let d = DeviceId(0);
        let total = 4u64 << 20;
        let a = big.alloc(d, total, StreamId(0)).unwrap();
        enq(big.memcpy_host_to_device(d, a, total, StreamId(0)));
        big.synchronize().unwrap();

        let b = tiny.alloc(d, total, StreamId(0)).unwrap();
        let chunk = 64u64 << 10;
        let n = total / chunk;
        for _ in 0..n {
            enq(tiny.memcpy_host_to_device(d, b, chunk, StreamId(0)));
        }
        tiny.synchronize().unwrap();
        assert!(
            tiny.clock_ns() > big.clock_ns(),
            "tiny={} big={}",
            tiny.clock_ns(),
            big.clock_ns()
        );
    }

    #[test]
    fn concurrent_h2d_on_two_streams_share_pcie() {
        let d = DeviceId(0);
        let bytes = 32u64 << 20;
        let mut one = Sim::new(h100());
        let a = one.alloc(d, bytes, StreamId(0)).unwrap();
        enq(one.memcpy_pinned_to_device(d, a, bytes, StreamId(0)));
        one.synchronize().unwrap();
        let t1 = one.clock_ns();

        let mut two = Sim::new(h100());
        let b = two.alloc(d, bytes, StreamId(0)).unwrap();
        let c = two.alloc(d, bytes, StreamId(1)).unwrap();
        enq(two.memcpy_pinned_to_device(d, b, bytes, StreamId(0)));
        enq(two.memcpy_pinned_to_device(d, c, bytes, StreamId(1)));
        two.synchronize().unwrap();
        let t2 = two.clock_ns();
        assert!(
            t2 > t1,
            "two concurrent H2D must not finish in one-copy time (shared PCIe); t1={t1} t2={t2}"
        );
    }

    #[test]
    fn determinism() {
        let run = || {
            let mut sim = Sim::new(h100());
            let d = DeviceId(0);
            let a = sim.alloc(d, 1 << 20, StreamId(0)).unwrap();
            enq(sim.memcpy_host_to_device(d, a, 1 << 20, StreamId(0)));
            enq(sim.kernel(
                d,
                KernelKind::other(1 << 24, 1 << 20),
                &[a],
                &[a],
                StreamId(0),
            ));
            sim.synchronize().unwrap();
            sim.clock_ns()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn eight_gpu_profile_has_nvlink() {
        let p = HardwareProfile::example_8xh100_nvlink();
        assert_eq!(p.n_gpus(), 8);
        assert!(p.link(Some(DeviceId(0)), Some(DeviceId(7))).is_ok());
    }

    #[test]
    fn score_line_is_stable() {
        let s = Score {
            wall_ns: 10,
            hbm_peak: 20,
            bytes_moved: 30,
            ns_per_token: None,
            energy_uj: 7,
            ttft_ns: None,
            itl_ns: None,
            rent_usd_micros_per_hour: 0,
            usd_micros_per_m_tokens: None,
        };
        assert_eq!(
            s.line(),
            "wall_ns=10 hbm_peak=20 bytes_moved=30 energy_uj=7"
        );
        assert_eq!(
            s.clone().with_tokens(2).line(),
            "wall_ns=10 hbm_peak=20 bytes_moved=30 energy_uj=7 ns_per_token=5"
        );
        assert_eq!(
            s.with_latencies(4, Some(3)).line(),
            "wall_ns=10 hbm_peak=20 bytes_moved=30 energy_uj=7 ttft_ns=4 itl_ns=3"
        );
    }

    #[test]
    fn energy_scales_with_profile_tdp_not_dollars() {
        let bytes = 8u64 << 20;
        let run = |p: HardwareProfile| {
            let mut sim = Sim::new(p);
            let d = DeviceId(0);
            let s = StreamId(0);
            let a = sim.alloc(d, bytes, s).unwrap();
            enq(sim.memcpy_host_to_device(d, a, bytes, s));
            sim.synchronize().unwrap();
            Score::from_sim(&sim)
        };
        let h100 = run(HardwareProfile::example_h100_sxm());
        let cheap = run(HardwareProfile::example_cheap_48gb());
        assert!(h100.energy_uj > 0);
        assert!(h100.energy_uj > cheap.energy_uj);
        assert!(h100.line().contains("energy_uj="));
        assert!(!h100.line().contains('$'));
        assert!(!h100
            .with_tokens(8)
            .line()
            .contains("usd_micros_per_m_tokens"));
    }

    #[test]
    fn rent_times_wall_fills_usd_per_m_tokens() {
        let p = HardwareProfile::example_h100_sxm().with_rent_usd_micros_per_hour(2_000_000);
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_host_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        let scored = Score::from_sim(&sim).with_tokens(1000);
        assert!(scored.usd_micros_per_m_tokens.is_some());
        assert!(scored.line().contains("usd_micros_per_m_tokens="));
        assert!(!scored.line().contains('$'));
    }

    #[test]
    fn grouped_moe_permille_lengthens_kernel() {
        let kind = KernelKind::GroupedMoeGemm {
            experts: 4,
            tokens_per_expert: 8,
            hidden: 4096,
            ff: 14336,
            dtype: DType::Fp16,
        };
        let ns = |pen: u16| {
            let mut p = HardwareProfile::example_h100_sxm();
            for g in &mut p.gpus {
                g.grouped_moe_permille = pen;
            }
            let mut sim = Sim::new(p);
            let d = DeviceId(0);
            let s = StreamId(0);
            let a = sim.alloc(d, 1 << 20, s).unwrap();
            enq(sim.memcpy_host_to_device(d, a, 1 << 20, s));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, kind.clone(), &[a], &[a], s));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        assert!(ns(2000) > ns(1000));
    }

    #[test]
    fn gemm_util_below_peak_lengthens_dense_kernel() {
        let ns = |util: u16| {
            let mut p = HardwareProfile::example_h100_sxm();
            for g in &mut p.gpus {
                g.gemm_util_permille = util;
            }
            let mut sim = Sim::new(p);
            let d = DeviceId(0);
            let s = StreamId(0);
            let a = sim.alloc(d, 1 << 20, s).unwrap();
            enq(sim.memcpy_host_to_device(d, a, 1 << 20, s));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, KernelKind::other(1 << 40, 1 << 20), &[a], &[a], s));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        assert!(ns(500) > ns(1000));
    }

    #[test]
    fn d2d_replica_charges_peer_hbm_and_survives_src_free() {
        let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s0 = StreamId(0);
        let bytes = 8u64 << 20;
        let a = sim.alloc(d0, bytes, s0).unwrap();
        enq(sim.memcpy_host_to_device(d0, a, bytes, s0));
        enq(sim.memcpy_device_to_device(d0, d1, a, bytes, s0));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(a, d0).unwrap());
        assert!(sim.is_resident(a, d1).unwrap());
        let used0 = sim.hbm_used(d0).unwrap();
        let used1 = sim.hbm_used(d1).unwrap();
        assert!(used0 >= bytes);
        assert!(used1 >= bytes);
        sim.free(d1, a, s0).unwrap();
        sim.synchronize().unwrap();
        assert!(sim.is_resident(a, d0).unwrap());
        assert!(!sim.is_resident(a, d1).unwrap());
        enq(sim.kernel(d0, KernelKind::other(8, bytes), &[a], &[a], s0));
        sim.synchronize().unwrap();
    }

    #[test]
    fn gpu_unavailable_rejects_submit() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        sim.set_unavailable(d, true).unwrap();
        assert!(sim.is_unavailable(d));
        match sim.alloc(d, 4096, StreamId(0)) {
            Err(SimError::Unavailable { device }) => assert_eq!(device, d),
            other => panic!("{other:?}"),
        }
        sim.set_unavailable(d, false).unwrap();
        assert!(sim.alloc(d, 4096, StreamId(0)).is_ok());
    }

    #[test]
    fn fail_next_memcpy_is_expert_load_failure() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        sim.fail_next_memcpy();
        match sim.memcpy_host_to_device(d, a, 4096, s) {
            Err(SimError::TransferFailed { alloc }) => assert_eq!(alloc, a),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn extra_transfer_ns_delays_h2d() {
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let mut fast = Sim::new(h100());
        let a = fast.alloc(d, bytes, s).unwrap();
        enq(fast.memcpy_host_to_device(d, a, bytes, s));
        fast.synchronize().unwrap();
        let t0 = fast.clock_ns();

        let mut slow = Sim::new(h100());
        slow.set_extra_transfer_ns(1_000_000);
        let b = slow.alloc(d, bytes, s).unwrap();
        enq(slow.memcpy_host_to_device(d, b, bytes, s));
        slow.synchronize().unwrap();
        assert!(
            slow.clock_ns() >= t0.saturating_add(1_000_000),
            "fast={t0} delayed={}",
            slow.clock_ns()
        );
    }

    #[test]
    fn cancel_stream_skips_queued_kernel() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_host_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 40, 4096), &[a], &[a], s));
        enq(sim.kernel(d, KernelKind::other(8, 4096), &[a], &[a], s));
        sim.start_ready().unwrap();
        let n = sim.cancel_stream(d, s).unwrap();
        assert!(n >= 1);
        match sim.synchronize() {
            Err(SimError::Cancelled { n: skipped, .. }) => assert!(skipped >= 1),
            other => panic!("{other:?}"),
        }
        assert!(sim.cancelled_count() >= 1);
    }

    #[test]
    fn two_gpu_pcie_d2d_is_slower_than_nvlink() {
        let bytes = 32u64 << 20;
        let pcie = probe_topology(HardwareProfile::example_2xh100_pcie(), bytes).unwrap();
        let nv = probe_topology(HardwareProfile::example_8xh100_nvlink(), bytes).unwrap();
        let pcie_d2d = pcie.p2p[0].ns.expect("pcie p2p");
        let nv_d2d = nv.p2p[0].ns.expect("nvlink p2p");
        assert!(pcie_d2d > nv_d2d, "pcie={pcie_d2d} nvlink={nv_d2d}");
        assert_eq!(pcie.n_gpus, 2);
        assert_eq!(nv.n_gpus, 8);
    }

    #[test]
    fn bad_numa_far_gpu_h2d_is_slower() {
        let p = probe_topology(HardwareProfile::example_bad_numa(), 16u64 << 20).unwrap();
        assert_eq!(p.h2d_ns.len(), 2);
        assert!(
            p.h2d_ns[1] > p.h2d_ns[0],
            "near={} far={}",
            p.h2d_ns[0],
            p.h2d_ns[1]
        );
        assert!(p.p2p[0].ns.is_none());
    }

    #[test]
    fn rdma_peer_exists_and_asymmetric_omits_02() {
        let rdma = probe_topology(HardwareProfile::example_2node_rdma(), 8u64 << 20).unwrap();
        assert!(rdma.p2p[0].ns.is_some());
        let kind = HardwareProfile::example_2node_rdma()
            .link(Some(DeviceId(0)), Some(DeviceId(1)))
            .unwrap()
            .kind;
        assert_eq!(kind, LinkKind::Rdma);
        let line = probe_topology(HardwareProfile::example_asymmetric_links(), 4u64 << 20).unwrap();
        assert_eq!(line.n_gpus, 3);
        let hop02 = line
            .p2p
            .iter()
            .find(|h| h.src == DeviceId(0) && h.dst == DeviceId(2))
            .expect("0-2");
        assert!(hop02.ns.is_none());
        let hop01 = line
            .p2p
            .iter()
            .find(|h| h.src == DeviceId(0) && h.dst == DeviceId(1))
            .expect("0-1");
        assert!(hop01.ns.is_some());
        assert!(!line.line().is_empty());
    }

    #[test]
    fn allreduce_needs_peer_link() {
        let mut ok = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let a = ok.alloc(DeviceId(0), bytes, s).unwrap();
        enq(ok.memcpy_host_to_device(DeviceId(0), a, bytes, s));
        enq(ok.memcpy_device_to_device(DeviceId(0), DeviceId(1), a, bytes, s));
        ok.synchronize().unwrap();
        enq(ok.allreduce(&[(DeviceId(0), a), (DeviceId(1), a)], bytes, s));
        ok.synchronize().unwrap();

        let mut bad = Sim::new(HardwareProfile::example_asymmetric_links());
        let b = bad.alloc(DeviceId(0), bytes, s).unwrap();
        enq(bad.memcpy_host_to_device(DeviceId(0), b, bytes, s));
        enq(bad.memcpy_device_to_device(DeviceId(0), DeviceId(1), b, bytes, s));
        bad.synchronize().unwrap();
        enq(bad.memcpy_device_to_device(DeviceId(1), DeviceId(2), b, bytes, s));
        bad.synchronize().unwrap();
        enq(bad.allreduce(
            &[(DeviceId(0), b), (DeviceId(1), b), (DeviceId(2), b)],
            bytes,
            s,
        ));
        match bad.synchronize() {
            Err(SimError::NoPeer { src, dst }) => {
                assert!(src == DeviceId(2) || dst == DeviceId(0) || src == DeviceId(0));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_capture_does_not_run_until_launch() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_host_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        let t0 = sim.clock_ns();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 30, 4096), &[a], &[a], s));
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.clock_ns(), t0);
        assert_eq!(sim.graph_len(g).unwrap(), 1);
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        let t1 = sim.clock_ns();
        assert!(t1 > t0);
        let n2 = sim.launch_graph(g, s).unwrap();
        assert_eq!(n2, 1);
        sim.synchronize().unwrap();
        assert!(sim.clock_ns() > t1);
    }

    #[test]
    fn graph_cannot_capture_sync_malloc() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        match sim.malloc(d, 4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        let a = sim.alloc(d, 4096, s).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 1);
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(a, d).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
    }

    #[test]
    fn graph_captures_malloc_async_kernel_and_free() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        let a = sim.alloc(d, 4096, s).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        sim.free(d, a, s).unwrap();
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 3);
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 3);
        sim.synchronize().unwrap();
        assert!(!sim.is_resident(a, d).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        assert_eq!(
            sim.graph_mem_get(d, GraphMemAttr::UsedMemCurrent).unwrap(),
            0
        );
        assert_eq!(
            sim.graph_mem_get(d, GraphMemAttr::ReservedMemCurrent)
                .unwrap(),
            4096
        );
        let n2 = sim.launch_graph(g, s).unwrap();
        assert_eq!(n2, 3);
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        assert_eq!(sim.hbm_peak(), 4096);
        sim.graph_mem_trim(d).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
    }

    #[test]
    fn graph_mem_alloc_reuses_hbm_on_relaunch() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let g = sim.end_capture().unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(a, d).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        let n2 = sim.launch_graph(g, s).unwrap();
        assert_eq!(n2, 2);
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        assert_eq!(sim.hbm_peak(), 4096);
    }

    #[test]
    fn graph_auto_free_on_launch_recharges_hbm() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.alloc_overhead_ns = 50_000;
            g.pool_reuse_ns = 1;
            g.graph_launch_ns = 1;
            g.graph_instantiate_ns = 1;
            g.graph_upload_ns = 1;
        }
        let d = DeviceId(0);
        let s = StreamId(0);
        let mut reuse = Sim::new(p.clone());
        reuse.begin_capture(d, s).unwrap();
        let a = reuse.alloc(d, 4096, s).unwrap();
        enq(reuse.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let g = reuse.end_capture().unwrap();
        let _ = reuse.instantiate_graph(g).unwrap();
        let n = reuse.launch_graph(g, s).unwrap();
        assert_eq!(n, 2);
        reuse.synchronize().unwrap();
        let t_reuse0 = reuse.clock_ns();
        let n2 = reuse.launch_graph(g, s).unwrap();
        assert_eq!(n2, 2);
        reuse.synchronize().unwrap();
        let reuse_ns = reuse.clock_ns().saturating_sub(t_reuse0);

        let mut af = Sim::new(p);
        af.begin_capture(d, s).unwrap();
        let b = af.alloc(d, 4096, s).unwrap();
        enq(af.kernel(d, KernelKind::other(8, 8), &[b], &[b], s));
        let h = af.end_capture().unwrap();
        let _ = af.instantiate_graph_auto_free(h).unwrap();
        assert!(af.graph_auto_free_on_launch(h).unwrap());
        let n = af.launch_graph(h, s).unwrap();
        assert_eq!(n, 2);
        af.synchronize().unwrap();
        assert_eq!(af.hbm_used(d).unwrap(), 4096);
        let t0 = af.clock_ns();
        let n2 = af.launch_graph(h, s).unwrap();
        assert_eq!(n2, 2);
        af.synchronize().unwrap();
        let auto_ns = af.clock_ns().saturating_sub(t0);
        assert_eq!(af.hbm_used(d).unwrap(), 4096);
        assert_eq!(af.hbm_peak(), 4096);
        assert!(
            auto_ns > reuse_ns,
            "auto-free relaunch must re-alloc, auto={auto_ns} reuse={reuse_ns}"
        );
        af.destroy_graph(h).unwrap();
        assert_eq!(
            af.graph_mem_get(d, GraphMemAttr::UsedMemCurrent).unwrap(),
            0
        );
        assert_eq!(
            af.graph_mem_get(d, GraphMemAttr::ReservedMemCurrent)
                .unwrap(),
            4096
        );
        assert_eq!(af.hbm_used(d).unwrap(), 4096);
        af.graph_mem_trim(d).unwrap();
        assert_eq!(af.hbm_used(d).unwrap(), 0);
    }

    #[test]
    fn graph_auto_free_rejects_free_nodes_and_flag_change() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        sim.free(d, a, s).unwrap();
        let with_free = sim.end_capture().unwrap();
        let err = sim.instantiate_graph_auto_free(with_free).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("mem free"), "{why}"),
            other => panic!("{other:?}"),
        }

        sim.begin_capture(d, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[b], &[b], s));
        let g = sim.end_capture().unwrap();
        let exec = sim.instantiate_graph(g).unwrap();
        assert_ne!(exec, g);
        assert!(!sim.graph_auto_free_on_launch(g).unwrap());
        let err = sim.instantiate_graph_auto_free(exec).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("flags"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_exec_get_flags_and_instantiate_with_flags() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_instantiate_ns = 20_000;
            g.graph_upload_ns = 7_000;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let g = sim.end_capture().unwrap();
        let err = sim.graph_exec_get_flags(g).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not instantiated"), "{why}"),
            e => panic!("{e:?}"),
        }
        let err = sim.instantiate_graph_with_flags(g, 1 << 16).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("instantiate flags"), "{why}"),
            e => panic!("{e:?}"),
        }
        let t0 = sim.clock_ns();
        let exec = sim
            .instantiate_graph_with_flags(g, GraphInstantiateFlags::UPLOAD)
            .unwrap();
        assert_ne!(exec, g);
        assert_eq!(sim.clock_ns(), t0.saturating_add(27_000));
        assert!(sim.graph_uploaded(g).unwrap());
        assert_eq!(
            sim.graph_exec_get_flags(g).unwrap(),
            GraphInstantiateFlags::UPLOAD
        );
        let t1 = sim.clock_ns();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        assert_eq!(sim.clock_ns(), t1);
        sim.synchronize().unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        let err = sim
            .instantiate_graph_with_flags(exec, GraphInstantiateFlags::AUTO_FREE_ON_LAUNCH)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("flags"), "{why}"),
            e => panic!("{e:?}"),
        }
        let extra = sim
            .instantiate_graph_with_flags(g, GraphInstantiateFlags::AUTO_FREE_ON_LAUNCH)
            .unwrap();
        assert_ne!(extra, exec);
        assert_eq!(
            sim.graph_exec_get_flags(g).unwrap(),
            GraphInstantiateFlags::UPLOAD
        );
        assert_eq!(
            sim.graph_exec_get_flags(extra).unwrap(),
            GraphInstantiateFlags::AUTO_FREE_ON_LAUNCH
        );

        let h = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(h, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim.instantiate_graph_auto_free(h).unwrap();
        assert_eq!(
            sim.graph_exec_get_flags(h).unwrap(),
            GraphInstantiateFlags::AUTO_FREE_ON_LAUNCH
        );
        let both = GraphInstantiateFlags::AUTO_FREE_ON_LAUNCH | GraphInstantiateFlags::UPLOAD;
        let k = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(k, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim.instantiate_graph_with_flags(k, both).unwrap();
        assert_eq!(sim.graph_exec_get_flags(k).unwrap(), both);
        assert!(sim.graph_uploaded(k).unwrap());
        assert!(sim.graph_auto_free_on_launch(k).unwrap());
    }

    #[test]
    fn graph_clone_mem_alloc_is_independent_hbm() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let src = sim.end_capture().unwrap();
        let clone = sim.clone_graph(src).unwrap();
        let n = sim.launch_graph(src, s).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        let n2 = sim.launch_graph(clone, s).unwrap();
        assert_eq!(n2, 2);
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 8192);
        sim.destroy_graph(src).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 8192);
        assert_eq!(
            sim.graph_mem_get(d, GraphMemAttr::UsedMemCurrent).unwrap(),
            4096
        );
        assert_eq!(
            sim.graph_mem_get(d, GraphMemAttr::ReservedMemCurrent)
                .unwrap(),
            8192
        );
        sim.graph_mem_trim(d).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        let n3 = sim.launch_graph(clone, s).unwrap();
        assert_eq!(n3, 2);
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
    }

    #[test]
    fn graph_destroy_refunds_mem_alloc() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let g = sim.end_capture().unwrap();
        sim.destroy_graph(g).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);

        sim.begin_capture(d, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[b], &[b], s));
        let live = sim.end_capture().unwrap();
        let n = sim.launch_graph(live, s).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        sim.destroy_graph(live).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        assert_eq!(
            sim.graph_mem_get(d, GraphMemAttr::ReservedMemCurrent)
                .unwrap(),
            4096
        );
        assert!(!sim.is_resident(b, d).unwrap());
        sim.graph_mem_trim(d).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
    }

    #[test]
    fn graph_mem_attr_counts_graph_allocs_not_malloc() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let host = sim.malloc(d, 8192).unwrap();
        assert_eq!(
            sim.graph_mem_get(d, GraphMemAttr::UsedMemCurrent).unwrap(),
            0
        );
        assert!(sim.hbm_used(d).unwrap() >= 8192);
        sim.begin_capture(d, s).unwrap();
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let g = sim.end_capture().unwrap();
        let _ = sim.instantiate_graph(g).unwrap();
        assert_eq!(
            sim.graph_mem_get(d, GraphMemAttr::UsedMemCurrent).unwrap(),
            0
        );
        let n = sim.launch_graph(g, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(n, 2);
        assert_eq!(
            sim.graph_mem_get(d, GraphMemAttr::UsedMemCurrent).unwrap(),
            4096
        );
        assert_eq!(
            sim.graph_mem_get(d, GraphMemAttr::ReservedMemCurrent)
                .unwrap(),
            4096
        );
        assert_eq!(
            sim.graph_mem_get(d, GraphMemAttr::UsedMemHigh).unwrap(),
            4096
        );
        let (free_before, _) = sim.mem_info(d).unwrap();
        sim.graph_mem_trim(d).unwrap();
        let (free_after, _) = sim.mem_info(d).unwrap();
        assert_eq!(free_before, free_after);
        sim.graph_mem_set(d, GraphMemAttr::UsedMemHigh, 0).unwrap();
        assert_eq!(
            sim.graph_mem_get(d, GraphMemAttr::UsedMemHigh).unwrap(),
            4096
        );
        let err = sim
            .graph_mem_set(d, GraphMemAttr::UsedMemCurrent, 0)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("graph mem attribute"), "{why}"),
            e => panic!("{e:?}"),
        }
        sim.destroy_graph(g).unwrap();
        assert_eq!(
            sim.graph_mem_get(d, GraphMemAttr::UsedMemCurrent).unwrap(),
            0
        );
        assert_eq!(
            sim.graph_mem_get(d, GraphMemAttr::ReservedMemCurrent)
                .unwrap(),
            4096
        );
        assert_eq!(
            sim.graph_mem_get(d, GraphMemAttr::UsedMemHigh).unwrap(),
            4096
        );
        let (free_held, _) = sim.mem_info(d).unwrap();
        sim.graph_mem_trim(d).unwrap();
        let (free_trimmed, _) = sim.mem_info(d).unwrap();
        assert_eq!(free_trimmed, free_held.saturating_add(4096));
        assert_eq!(
            sim.graph_mem_get(d, GraphMemAttr::ReservedMemCurrent)
                .unwrap(),
            0
        );
        sim.free_sync(host).unwrap();
        sim.begin_capture(d, s).unwrap();
        let err = sim.graph_mem_trim(d).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            e => panic!("{e:?}"),
        }
        let _end = sim.end_capture().unwrap();
    }

    #[test]
    fn graph_mem_pool_holds_reserved_until_trim() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let gp = sim.graph_pool(d).unwrap();
        let dp = sim.default_pool(d).unwrap();
        assert_ne!(gp, dp);
        match sim.alloc_from_pool(d, gp, 4096, s) {
            Err(SimError::Invalid { why }) => assert!(why.contains("graph mem pool"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.set_device_mempool(d, gp) {
            Err(SimError::Invalid { why }) => assert!(why.contains("graph mem pool"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.set_pool_release_threshold(gp, 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("graph mem pool"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let g = sim.end_capture().unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
        assert_eq!(sim.pool_live(gp).unwrap(), 4096);
        assert_eq!(sim.pool_cached(gp).unwrap(), 0);
        assert_eq!(sim.pool_live(dp).unwrap(), 0);
        sim.destroy_graph(g).unwrap();
        assert_eq!(sim.pool_live(gp).unwrap(), 0);
        assert_eq!(sim.pool_cached(gp).unwrap(), 4096);
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        sim.graph_mem_trim(d).unwrap();
        assert_eq!(sim.pool_cached(gp).unwrap(), 0);
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
    }

    #[test]
    fn update_graph_rejects_mem_nodes() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let exec = sim.end_capture().unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        sim.begin_capture(d, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[b], &[b], s));
        let src = sim.end_capture().unwrap();
        let err = sim.update_graph(exec, src).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("mem"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn pinned_h2d_is_faster_than_pageable() {
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let mut pageable = Sim::new(h100());
        let a = pageable.alloc(d, bytes, s).unwrap();
        enq(pageable.memcpy_host_to_device(d, a, bytes, s));
        pageable.synchronize().unwrap();
        let mut pinned = Sim::new(h100());
        let b = pinned.alloc(d, bytes, s).unwrap();
        enq(pinned.memcpy_pinned_to_device(d, b, bytes, s));
        pinned.synchronize().unwrap();
        assert!(
            pageable.clock_ns() > pinned.clock_ns(),
            "pageable={} pinned={}",
            pageable.clock_ns(),
            pinned.clock_ns()
        );
    }

    #[test]
    fn host_pinned_alloc_does_not_charge_hbm() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let h = sim.alloc_host_pinned(8 << 20).unwrap();
        assert!(sim.is_host_pinned(h).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        assert_eq!(sim.hbm_peak(), 0);
        enq(sim.kernel(d, KernelKind::other(8, 8), &[h], &[h], StreamId(0)));
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, h);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn host_pinned_copy_onto_device_then_kernel() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let h = sim.alloc_host_pinned(bytes).unwrap();
        enq(sim.memcpy_pinned_to_device(d, h, bytes, s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(h, d).unwrap());
        assert!(sim.is_host_pinned(h).unwrap());
        assert!(sim.hbm_used(d).unwrap() >= bytes);
        enq(sim.kernel(d, KernelKind::other(8, bytes), &[h], &[h], s));
        sim.synchronize().unwrap();
        sim.free(d, h, s).unwrap();
        sim.synchronize().unwrap();
        assert!(!sim.is_resident(h, d).unwrap());
        assert!(sim.is_host_pinned(h).unwrap());
        sim.free_host_pinned(h).unwrap();
        assert!(!sim.is_host_pinned(h).unwrap());
    }

    #[test]
    fn device_to_pinned_keeps_hbm_and_marks_host() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let a = sim.alloc(d, bytes, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, bytes, s));
        sim.synchronize().unwrap();
        let used = sim.hbm_used(d).unwrap();
        enq(sim.memcpy_device_to_pinned(d, a, bytes, s));
        sim.synchronize().unwrap();
        assert!(sim.is_host_pinned(a).unwrap());
        assert!(sim.is_resident(a, d).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), used);
    }

    #[test]
    fn graph_cannot_capture_host_pinned_alloc() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        match sim.alloc_host_pinned(4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("alloc")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_launch_amortizes_kernel_overhead() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.launch_overhead_ns = 50_000;
            g.graph_launch_ns = 5_000;
        }
        let d = DeviceId(0);
        let s = StreamId(0);
        let kind = KernelKind::other(8, 8);
        let n = 4u32;
        let kernels = |sim: &mut Sim, a: AllocId| {
            for _ in 0..n {
                enq(sim.kernel(d, kind.clone(), &[a], &[a], s));
            }
        };
        let mut serial = Sim::new(p.clone());
        let a = serial.alloc(d, 4096, s).unwrap();
        enq(serial.memcpy_pinned_to_device(d, a, 4096, s));
        serial.synchronize().unwrap();
        let t0 = serial.clock_ns();
        kernels(&mut serial, a);
        serial.synchronize().unwrap();
        let serial_ns = serial.clock_ns().saturating_sub(t0);

        let mut gsim = Sim::new(p);
        let b = gsim.alloc(d, 4096, s).unwrap();
        enq(gsim.memcpy_pinned_to_device(d, b, 4096, s));
        gsim.synchronize().unwrap();
        gsim.begin_capture(d, s).unwrap();
        kernels(&mut gsim, b);
        let g = gsim.end_capture().unwrap();
        let t1 = gsim.clock_ns();
        let launched = gsim.launch_graph(g, s).unwrap();
        assert_eq!(launched, n);
        gsim.synchronize().unwrap();
        let graph_ns = gsim.clock_ns().saturating_sub(t1);
        assert!(graph_ns < serial_ns, "graph={graph_ns} serial={serial_ns}");
    }

    #[test]
    fn graph_capture_replays_memcpy() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let a = sim.alloc(d, bytes, s).unwrap();
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, bytes, s));
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.bytes_moved(), 0);
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert!(sim.bytes_moved() >= bytes);
        assert!(sim.is_resident(a, d).unwrap());
    }

    #[test]
    fn first_graph_launch_instantiates_once() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_instantiate_ns = 25_000;
            g.graph_launch_ns = 1_000;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let g = sim.end_capture().unwrap();
        assert!(!sim.graph_instantiated(g).unwrap());
        let t0 = sim.clock_ns();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        assert!(sim.graph_instantiated(g).unwrap());
        assert!(sim.clock_ns() >= t0.saturating_add(25_000));
        sim.synchronize().unwrap();
        let first = sim.clock_ns().saturating_sub(t0);
        let t1 = sim.clock_ns();
        let n2 = sim.launch_graph(g, s).unwrap();
        assert_eq!(n2, 1);
        assert_eq!(sim.clock_ns(), t1);
        sim.synchronize().unwrap();
        let second = sim.clock_ns().saturating_sub(t1);
        assert!(second < first, "first={first} second={second}");
    }

    #[test]
    fn instantiate_graph_is_host_sync_and_idempotent() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_instantiate_ns = 40_000;
            g.graph_upload_ns = 12_000;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let g = sim.end_capture().unwrap();
        let t0 = sim.clock_ns();
        let exec = sim.instantiate_graph(g).unwrap();
        assert_ne!(exec, g);
        assert_eq!(sim.clock_ns(), t0.saturating_add(40_000));
        assert!(sim.graph_instantiated(g).unwrap());
        assert!(!sim.graph_uploaded(g).unwrap());
        let _ = sim.instantiate_graph(exec).unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(40_000));
        let exec2 = sim.instantiate_graph(g).unwrap();
        assert_ne!(exec2, exec);
        assert_eq!(sim.clock_ns(), t0.saturating_add(80_000));
        sim.upload_graph(g).unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(92_000));
        assert!(sim.graph_uploaded(g).unwrap());
        sim.upload_graph(g).unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(92_000));
        let t1 = sim.clock_ns();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        assert_eq!(sim.clock_ns(), t1);
        sim.synchronize().unwrap();
    }

    #[test]
    fn instantiate_returns_separate_exec_ids() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
        sim.synchronize().unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let e1 = sim.instantiate_graph(g).unwrap();
        let e2 = sim.instantiate_graph(g).unwrap();
        assert_ne!(e1, g);
        assert_ne!(e2, g);
        assert_ne!(e1, e2);
        let params = KernelNodeParams {
            kind: KernelKind::other(8, 8),
            reads: vec![KernelBuf::whole(b)],
            writes: vec![KernelBuf::whole(b)],
            cooperative: false,
        };
        sim.graph_exec_kernel_set_params(e2, 0, &params).unwrap();
        let n = sim.launch_graph(e2, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(b, d).unwrap());
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(e1, s).unwrap();
        assert_eq!(n, 1);
        let err = sim.synchronize().unwrap_err();
        match err {
            SimError::NotResident { alloc, device } => {
                assert_eq!(alloc, a);
                assert_eq!(device, d);
            }
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn instantiate_with_params_reports_success_and_err_node() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let mut params = GraphInstantiateParams::default();
        let exec = sim.instantiate_graph_with_params(g, &mut params).unwrap();
        assert_ne!(exec, g);
        assert_eq!(params.result, GraphInstantiateResult::Success);
        assert_eq!(params.err_node, None);
        let host = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(host, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_host_func(host).unwrap();
        let mut params = GraphInstantiateParams {
            flags: GraphInstantiateFlags::DEVICE_LAUNCH,
            ..GraphInstantiateParams::default()
        };
        let err = sim
            .instantiate_graph_with_params(host, &mut params)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("device launch"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(
            params.result,
            GraphInstantiateResult::NodeOperationNotSupported
        );
        assert_eq!(params.err_node, Some(1));
        let mem = sim.create_graph(d, s).unwrap();
        let id = sim.graph_add_alloc(mem, 64).unwrap();
        sim.graph_add_free(mem, id).unwrap();
        let mut params = GraphInstantiateParams {
            flags: GraphInstantiateFlags::AUTO_FREE_ON_LAUNCH,
            ..GraphInstantiateParams::default()
        };
        let err = sim
            .instantiate_graph_with_params(mem, &mut params)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("auto free"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(params.result, GraphInstantiateResult::InvalidStructure);
        assert_eq!(params.err_node, Some(1));
        sim.begin_capture(d, s).unwrap();
        let mut params = GraphInstantiateParams::default();
        let err = sim
            .instantiate_graph_with_params(g, &mut params)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(params.result, GraphInstantiateResult::Error);
        assert_eq!(params.err_node, None);
        let _cap = sim.end_capture().unwrap();
    }

    #[test]
    fn destroy_definition_leaves_exec_launchable() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let exec = sim.instantiate_graph(g).unwrap();
        sim.destroy_graph(g).unwrap();
        let err = sim.launch_graph(g, s).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("unknown graph"), "{why}"),
            e => panic!("{e:?}"),
        }
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
    }

    #[test]
    fn graph_add_on_definition_does_not_retarget_exec() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
        sim.synchronize().unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let exec = sim.instantiate_graph(g).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[b], &[b])
            .unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 2);
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        let err = sim.synchronize().unwrap_err();
        match err {
            SimError::NotResident { alloc, .. } => assert_eq!(alloc, a),
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn second_instantiate_of_mem_graph_is_invalid() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let g = sim.create_graph(d, s).unwrap();
        let _a = sim.graph_add_alloc(g, 4096).unwrap();
        let exec = sim.instantiate_graph(g).unwrap();
        assert_ne!(exec, g);
        let err = sim.instantiate_graph(g).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("mem exec"), "{why}"),
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn upload_graph_rejects_uninstantiated() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let g = sim.end_capture().unwrap();
        let err = sim.upload_graph(g).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _ = sim.instantiate_graph(g).unwrap();
        sim.upload_graph(g).unwrap();
        assert!(sim.graph_uploaded(g).unwrap());
    }

    #[test]
    fn first_graph_launch_uploads_after_instantiate() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_instantiate_ns = 25_000;
            g.graph_upload_ns = 9_000;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let g = sim.end_capture().unwrap();
        let _ = sim.instantiate_graph(g).unwrap();
        let t0 = sim.clock_ns();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        assert_eq!(sim.clock_ns(), t0.saturating_add(9_000));
        assert!(sim.graph_uploaded(g).unwrap());
        sim.synchronize().unwrap();
        let t1 = sim.clock_ns();
        let n2 = sim.launch_graph(g, s).unwrap();
        assert_eq!(n2, 1);
        assert_eq!(sim.clock_ns(), t1);
    }

    #[test]
    fn update_graph_swaps_allocs_when_topology_matches() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_update_ns = 7_000;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let exec = sim.end_capture().unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        sim.upload_graph(exec).unwrap();
        assert!(sim.graph_uploaded(exec).unwrap());
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[b], &[b], s));
        let src = sim.end_capture().unwrap();
        let t0 = sim.clock_ns();
        sim.update_graph(exec, src).unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(7_000));
        assert!(!sim.graph_uploaded(exec).unwrap());
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(b, d).unwrap());
    }

    #[test]
    fn graph_exec_kernel_set_params_swaps_allocs_without_second_graph() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_set_params_ns = 400;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let exec = sim.end_capture().unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        sim.upload_graph(exec).unwrap();
        assert!(sim.graph_uploaded(exec).unwrap());
        let (node, mut params) = sim.graph_unique_kernel(exec).unwrap();
        assert_eq!(node, 0);
        params.reads = vec![KernelBuf::whole(b)];
        params.writes = vec![KernelBuf::whole(b)];
        let t0 = sim.clock_ns();
        sim.graph_exec_kernel_set_params(exec, node, &params)
            .unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(400));
        assert!(!sim.graph_uploaded(exec).unwrap());
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(b, d).unwrap());
    }

    #[test]
    fn graph_exec_kernel_set_params_beats_exec_update_wall() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_instantiate_ns = 80_000;
            g.graph_update_ns = 9_000;
            g.graph_set_params_ns = 300;
            g.graph_upload_ns = 1_000;
            g.graph_launch_ns = 1_000;
        }
        let d = DeviceId(0);
        let s = StreamId(0);
        let run_update = {
            let mut sim = Sim::new(p.clone());
            let a = sim.alloc(d, 4096, s).unwrap();
            let b = sim.alloc(d, 4096, s).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
            enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
            sim.synchronize().unwrap();
            sim.begin_capture(d, s).unwrap();
            enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
            let exec = sim.end_capture().unwrap();
            let _ = sim.instantiate_graph(exec).unwrap();
            sim.upload_graph(exec).unwrap();
            sim.begin_capture(d, s).unwrap();
            enq(sim.kernel(d, KernelKind::other(8, 8), &[b], &[b], s));
            let src = sim.end_capture().unwrap();
            let t0 = sim.clock_ns();
            sim.update_graph(exec, src).unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let run_set = {
            let mut sim = Sim::new(p);
            let a = sim.alloc(d, 4096, s).unwrap();
            let b = sim.alloc(d, 4096, s).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
            enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
            sim.synchronize().unwrap();
            sim.begin_capture(d, s).unwrap();
            enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
            let exec = sim.end_capture().unwrap();
            let _ = sim.instantiate_graph(exec).unwrap();
            sim.upload_graph(exec).unwrap();
            let (node, mut params) = sim.graph_unique_kernel(exec).unwrap();
            params.reads = vec![KernelBuf::whole(b)];
            params.writes = vec![KernelBuf::whole(b)];
            let t0 = sim.clock_ns();
            sim.graph_exec_kernel_set_params(exec, node, &params)
                .unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        assert!(
            run_set < run_update,
            "set_params={run_set} update={run_update}"
        );
    }

    #[test]
    fn graph_exec_kernel_set_params_allows_mem_nodes() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
        sim.synchronize().unwrap();
        let exec = sim.create_graph(d, s).unwrap();
        let scratch = sim.graph_add_alloc(exec, 4096).unwrap();
        sim.graph_add_kernel(exec, KernelKind::other(8, 8), &[a], &[scratch])
            .unwrap();
        sim.graph_add_dependencies(exec, 0, 1).unwrap();
        sim.graph_add_free(exec, scratch).unwrap();
        sim.graph_add_dependencies(exec, 1, 2).unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        let src = sim.create_graph(d, s).unwrap();
        let scratch2 = sim.graph_add_alloc(src, 4096).unwrap();
        sim.graph_add_kernel(src, KernelKind::other(8, 8), &[b], &[scratch2])
            .unwrap();
        sim.graph_add_dependencies(src, 0, 1).unwrap();
        sim.graph_add_free(src, scratch2).unwrap();
        sim.graph_add_dependencies(src, 1, 2).unwrap();
        let err = sim.update_graph(exec, src).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("mem"), "{why}"),
            other => panic!("{other:?}"),
        }
        let (node, mut params) = sim.graph_unique_kernel(exec).unwrap();
        assert_eq!(node, 1);
        let owned = sim.graph_mem_allocs(exec).unwrap();
        assert_eq!(owned, vec![scratch]);
        params.reads = vec![KernelBuf::whole(b)];
        sim.graph_exec_kernel_set_params(exec, node, &params)
            .unwrap();
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 3);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(b, d).unwrap());
    }

    #[test]
    fn graph_exec_kernel_set_params_rejects_uninstantiated_and_topology() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let exec = sim.end_capture().unwrap();
        let params = KernelNodeParams {
            kind: KernelKind::other(8, 8),
            reads: vec![KernelBuf::whole(a)],
            writes: vec![KernelBuf::whole(a)],
            cooperative: false,
        };
        let err = sim
            .graph_exec_kernel_set_params(exec, 0, &params)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _ = sim.instantiate_graph(exec).unwrap();
        let err = sim
            .graph_exec_kernel_set_params(
                exec,
                0,
                &KernelNodeParams {
                    cooperative: true,
                    ..params.clone()
                },
            )
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("cooperative"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        let memcpy_g = sim.end_capture().unwrap();
        let _ = sim.instantiate_graph(memcpy_g).unwrap();
        let err = sim
            .graph_exec_kernel_set_params(memcpy_g, 0, &params)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not a kernel"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        let err = sim
            .graph_exec_kernel_set_params(exec, 0, &params)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn graph_memcpy_1d_apis_pack_and_convert_2d_nodes() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_memcpy_1d(g, Place::HostPinned, Place::Device(d), a, 4096)
            .unwrap();
        let packed = MemcpyOp::packed_1d(Place::HostPinned, Place::Device(d), a, 4096);
        let got = sim.graph_memcpy_get_params(g, 0).unwrap();
        assert_eq!(got, packed);
        assert!(got.is_1d());
        assert!(!got.is_2d());
        assert!(!got.is_3d());
        let err = sim
            .graph_add_memcpy_1d(g, Place::Host, Place::Device(d), a, 64)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("pageable"), "{why}"),
            other => panic!("{other:?}"),
        }
        let g2 = sim.create_graph(d, s).unwrap();
        sim.graph_add_memcpy(
            g2,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d),
                alloc: a,
                bytes: 256,
                height: 8,
                src_pitch: 256,
                dst_pitch: 512,
                ..MemcpyOp::default()
            },
        )
        .unwrap();
        assert!(sim.graph_memcpy_get_params(g2, 0).unwrap().is_2d());
        sim.graph_memcpy_set_params_1d(g2, 0, Place::HostPinned, Place::Device(d), a, 2048)
            .unwrap();
        let one = sim.graph_memcpy_get_params(g2, 0).unwrap();
        assert!(one.is_1d());
        assert_eq!(one.bytes, 2048);
        assert_eq!(one.height, 0);
        assert_eq!(one.src_pitch, 0);
        let exec = sim.instantiate_graph(g2).unwrap();
        sim.graph_memcpy_set_params_1d(g2, 0, Place::HostPinned, Place::Device(d), a, 1024)
            .unwrap();
        assert_eq!(
            sim.graph_exec_memcpy_get_params(exec, 0).unwrap().bytes,
            2048
        );
        assert_eq!(sim.graph_memcpy_get_params(g2, 0).unwrap().bytes, 1024);
        sim.graph_exec_memcpy_set_params_1d(exec, 0, Place::HostPinned, Place::Device(d), a, 512)
            .unwrap();
        assert_eq!(
            sim.graph_exec_memcpy_get_params(exec, 0).unwrap().bytes,
            512
        );
        sim.begin_capture(d, s).unwrap();
        let err = sim
            .graph_add_memcpy_1d(g, Place::HostPinned, Place::Device(d), a, 8)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let err = sim
            .graph_memcpy_set_params_1d(g2, 0, Place::HostPinned, Place::Device(d), a, 8)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let err = sim
            .graph_exec_memcpy_set_params_1d(exec, 0, Place::HostPinned, Place::Device(d), a, 8)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _cap = sim.end_capture().unwrap();
        let err = sim
            .graph_exec_memcpy_set_params_1d(exec, 0, Place::Host, Place::Device(d), a, 8)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("pageable"), "{why}"),
            other => panic!("{other:?}"),
        }
        let kern = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(kern, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let err = sim
            .graph_memcpy_set_params_1d(kern, 0, Place::HostPinned, Place::Device(d), a, 8)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not a memcpy"), "{why}"),
            other => panic!("{other:?}"),
        }
        let inst = sim.instantiate_graph(g).unwrap();
        let err = sim
            .graph_add_memcpy_1d(inst, Place::HostPinned, Place::Device(d), a, 8)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_exec_memcpy_set_params_retargets_without_second_graph() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        let exec = sim.create_graph(d, s).unwrap();
        sim.graph_add_memcpy(
            exec,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d),
                alloc: a,
                bytes: 4096,
                offset: 0,
                ..MemcpyOp::default()
            },
        )
        .unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        let moved0 = sim.bytes_moved();
        let (node, mut op) = sim.graph_unique_memcpy(exec).unwrap();
        assert_eq!(node, 0);
        assert_eq!(op.alloc, a);
        op.alloc = b;
        sim.graph_exec_memcpy_set_params(exec, node, &op).unwrap();
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert!(sim.bytes_moved() > moved0);
        assert!(sim.is_resident(b, d).unwrap());
    }

    #[test]
    fn graph_exec_memcpy_set_params_beats_exec_update_wall() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_instantiate_ns = 80_000;
            g.graph_update_ns = 9_000;
            g.graph_set_params_ns = 300;
            g.graph_upload_ns = 1_000;
            g.graph_launch_ns = 1_000;
        }
        let d = DeviceId(0);
        let s = StreamId(0);
        let op = |alloc: AllocId| MemcpyOp {
            src: Place::HostPinned,
            dst: Place::Device(d),
            alloc,
            bytes: 4096,
            offset: 0,
            ..MemcpyOp::default()
        };
        let run_update = {
            let mut sim = Sim::new(p.clone());
            let a = sim.malloc(d, 4096).unwrap();
            let b = sim.malloc(d, 4096).unwrap();
            let exec = sim.create_graph(d, s).unwrap();
            sim.graph_add_memcpy(exec, op(a)).unwrap();
            let _ = sim.instantiate_graph(exec).unwrap();
            sim.upload_graph(exec).unwrap();
            let src = sim.create_graph(d, s).unwrap();
            sim.graph_add_memcpy(src, op(b)).unwrap();
            let t0 = sim.clock_ns();
            sim.update_graph(exec, src).unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let run_set = {
            let mut sim = Sim::new(p);
            let a = sim.malloc(d, 4096).unwrap();
            let b = sim.malloc(d, 4096).unwrap();
            let exec = sim.create_graph(d, s).unwrap();
            sim.graph_add_memcpy(exec, op(a)).unwrap();
            let _ = sim.instantiate_graph(exec).unwrap();
            sim.upload_graph(exec).unwrap();
            let (node, mut m) = sim.graph_unique_memcpy(exec).unwrap();
            m.alloc = b;
            let t0 = sim.clock_ns();
            sim.graph_exec_memcpy_set_params(exec, node, &m).unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        assert!(
            run_set < run_update,
            "set_params={run_set} update={run_update}"
        );
        assert_eq!(run_set, 300);
        assert_eq!(run_update, 9_000);
    }

    #[test]
    fn graph_exec_memcpy_set_params_allows_mem_nodes() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        let exec = sim.create_graph(d, s).unwrap();
        let scratch = sim.graph_add_alloc(exec, 4096).unwrap();
        sim.graph_add_memcpy(
            exec,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d),
                alloc: a,
                bytes: 4096,
                offset: 0,
                ..MemcpyOp::default()
            },
        )
        .unwrap();
        sim.graph_add_dependencies(exec, 0, 1).unwrap();
        sim.graph_add_free(exec, scratch).unwrap();
        sim.graph_add_dependencies(exec, 1, 2).unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        let src = sim.create_graph(d, s).unwrap();
        let scratch2 = sim.graph_add_alloc(src, 4096).unwrap();
        sim.graph_add_memcpy(
            src,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d),
                alloc: b,
                bytes: 4096,
                offset: 0,
                ..MemcpyOp::default()
            },
        )
        .unwrap();
        sim.graph_add_dependencies(src, 0, 1).unwrap();
        sim.graph_add_free(src, scratch2).unwrap();
        sim.graph_add_dependencies(src, 1, 2).unwrap();
        let err = sim.update_graph(exec, src).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("mem"), "{why}"),
            other => panic!("{other:?}"),
        }
        let (node, mut op) = sim.graph_unique_memcpy(exec).unwrap();
        assert_eq!(node, 1);
        op.alloc = b;
        sim.graph_exec_memcpy_set_params(exec, node, &op).unwrap();
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 3);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(b, d).unwrap());
        assert!(sim.bytes_moved() >= 4096);
    }

    #[test]
    fn graph_exec_memcpy_set_params_rejects_uninstantiated_and_kernel() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let op = MemcpyOp {
            src: Place::HostPinned,
            dst: Place::Device(d),
            alloc: a,
            bytes: 4096,
            offset: 0,
            ..MemcpyOp::default()
        };
        let exec = sim.create_graph(d, s).unwrap();
        sim.graph_add_memcpy(exec, op.clone()).unwrap();
        let err = sim.graph_exec_memcpy_set_params(exec, 0, &op).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _ = sim.instantiate_graph(exec).unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let kern = sim.end_capture().unwrap();
        let _ = sim.instantiate_graph(kern).unwrap();
        let err = sim.graph_exec_memcpy_set_params(kern, 0, &op).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not a memcpy"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        let err = sim.graph_exec_memcpy_set_params(exec, 0, &op).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        match sim.graph_try_unique_memcpy(kern) {
            Ok(None) => {}
            other => panic!("{other:?}"),
        }
        match sim.graph_unique_memcpy(kern) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not a memcpy"), "{why}"),
            other => panic!("{other:?}"),
        }
        let mut pageable = op.clone();
        pageable.src = Place::Host;
        let err = sim
            .graph_exec_memcpy_set_params(exec, 0, &pageable)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("pageable"), "{why}"),
            other => panic!("{other:?}"),
        }
        let two = sim.create_graph(d, s).unwrap();
        sim.graph_add_memcpy(two, op.clone()).unwrap();
        sim.graph_add_memcpy(two, op).unwrap();
        let _ = sim.instantiate_graph(two).unwrap();
        match sim.graph_try_unique_memcpy(two) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not unique"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_exec_memset_set_params_retargets_without_second_graph() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        let exec = sim.create_graph(d, s).unwrap();
        sim.graph_add_memset(exec, KernelBuf::whole(a)).unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        let t0 = sim.clock_ns();
        let (node, mut buf) = sim.graph_unique_memset(exec).unwrap();
        assert_eq!(node, 0);
        assert_eq!(buf.id, a);
        buf.id = b;
        sim.graph_exec_memset_set_params(exec, node, buf).unwrap();
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert!(sim.clock_ns() > t0);
        assert!(sim.is_resident(b, d).unwrap());
    }

    #[test]
    fn graph_exec_memset_set_params_beats_exec_update_wall() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_instantiate_ns = 80_000;
            g.graph_update_ns = 9_000;
            g.graph_set_params_ns = 300;
            g.graph_upload_ns = 1_000;
            g.graph_launch_ns = 1_000;
        }
        let d = DeviceId(0);
        let s = StreamId(0);
        let run_update = {
            let mut sim = Sim::new(p.clone());
            let a = sim.malloc(d, 4096).unwrap();
            let b = sim.malloc(d, 4096).unwrap();
            let exec = sim.create_graph(d, s).unwrap();
            sim.graph_add_memset(exec, KernelBuf::whole(a)).unwrap();
            let _ = sim.instantiate_graph(exec).unwrap();
            sim.upload_graph(exec).unwrap();
            let src = sim.create_graph(d, s).unwrap();
            sim.graph_add_memset(src, KernelBuf::whole(b)).unwrap();
            let t0 = sim.clock_ns();
            sim.update_graph(exec, src).unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let run_set = {
            let mut sim = Sim::new(p);
            let a = sim.malloc(d, 4096).unwrap();
            let b = sim.malloc(d, 4096).unwrap();
            let exec = sim.create_graph(d, s).unwrap();
            sim.graph_add_memset(exec, KernelBuf::whole(a)).unwrap();
            let _ = sim.instantiate_graph(exec).unwrap();
            sim.upload_graph(exec).unwrap();
            let (node, mut buf) = sim.graph_unique_memset(exec).unwrap();
            buf.id = b;
            let t0 = sim.clock_ns();
            sim.graph_exec_memset_set_params(exec, node, buf).unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        assert!(
            run_set < run_update,
            "set_params={run_set} update={run_update}"
        );
        assert_eq!(run_set, 300);
        assert_eq!(run_update, 9_000);
    }

    #[test]
    fn graph_exec_memset_set_params_allows_mem_nodes() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        let exec = sim.create_graph(d, s).unwrap();
        let scratch = sim.graph_add_alloc(exec, 4096).unwrap();
        sim.graph_add_memset(exec, KernelBuf::whole(a)).unwrap();
        sim.graph_add_dependencies(exec, 0, 1).unwrap();
        sim.graph_add_free(exec, scratch).unwrap();
        sim.graph_add_dependencies(exec, 1, 2).unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        let src = sim.create_graph(d, s).unwrap();
        let scratch2 = sim.graph_add_alloc(src, 4096).unwrap();
        sim.graph_add_memset(src, KernelBuf::whole(b)).unwrap();
        sim.graph_add_dependencies(src, 0, 1).unwrap();
        sim.graph_add_free(src, scratch2).unwrap();
        sim.graph_add_dependencies(src, 1, 2).unwrap();
        let err = sim.update_graph(exec, src).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("mem"), "{why}"),
            other => panic!("{other:?}"),
        }
        let (node, mut buf) = sim.graph_unique_memset(exec).unwrap();
        assert_eq!(node, 1);
        buf.id = b;
        sim.graph_exec_memset_set_params(exec, node, buf).unwrap();
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 3);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(b, d).unwrap());
    }

    #[test]
    fn graph_exec_memset_set_params_rejects_uninstantiated_and_kernel() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let buf = KernelBuf::whole(a);
        let exec = sim.create_graph(d, s).unwrap();
        sim.graph_add_memset(exec, buf).unwrap();
        let err = sim.graph_exec_memset_set_params(exec, 0, buf).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _ = sim.instantiate_graph(exec).unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let kern = sim.end_capture().unwrap();
        let _ = sim.instantiate_graph(kern).unwrap();
        let err = sim.graph_exec_memset_set_params(kern, 0, buf).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not a memset"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        let err = sim.graph_exec_memset_set_params(exec, 0, buf).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        match sim.graph_try_unique_memset(kern) {
            Ok(None) => {}
            other => panic!("{other:?}"),
        }
        match sim.graph_unique_memset(kern) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not a memset"), "{why}"),
            other => panic!("{other:?}"),
        }
        let two = sim.create_graph(d, s).unwrap();
        sim.graph_add_memset(two, buf).unwrap();
        sim.graph_add_memset(two, buf).unwrap();
        let _ = sim.instantiate_graph(two).unwrap();
        match sim.graph_try_unique_memset(two) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not unique"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_node_set_enabled_skips_launch_without_second_graph() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let exec = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(exec, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_kernel(exec, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        assert!(sim.graph_node_get_enabled(exec, 1).unwrap());
        sim.graph_node_set_enabled(exec, 1, false).unwrap();
        assert!(!sim.graph_node_get_enabled(exec, 1).unwrap());
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        sim.graph_node_set_enabled(exec, 1, true).unwrap();
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
    }

    #[test]
    fn graph_node_set_enabled_beats_exec_update_wall() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_instantiate_ns = 80_000;
            g.graph_update_ns = 9_000;
            g.graph_set_params_ns = 300;
            g.graph_upload_ns = 1_000;
            g.graph_launch_ns = 1_000;
        }
        let d = DeviceId(0);
        let s = StreamId(0);
        let run_update = {
            let mut sim = Sim::new(p.clone());
            let a = sim.malloc(d, 4096).unwrap();
            let exec = sim.create_graph(d, s).unwrap();
            sim.graph_add_kernel(exec, KernelKind::other(8, 8), &[a], &[a])
                .unwrap();
            sim.graph_add_kernel(exec, KernelKind::other(8, 8), &[a], &[a])
                .unwrap();
            let _ = sim.instantiate_graph(exec).unwrap();
            sim.upload_graph(exec).unwrap();
            let src = sim.create_graph(d, s).unwrap();
            sim.graph_add_kernel(src, KernelKind::other(8, 8), &[a], &[a])
                .unwrap();
            sim.graph_add_kernel(src, KernelKind::other(8, 8), &[a], &[a])
                .unwrap();
            let t0 = sim.clock_ns();
            sim.update_graph(exec, src).unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let run_set = {
            let mut sim = Sim::new(p);
            let a = sim.malloc(d, 4096).unwrap();
            let exec = sim.create_graph(d, s).unwrap();
            sim.graph_add_kernel(exec, KernelKind::other(8, 8), &[a], &[a])
                .unwrap();
            sim.graph_add_kernel(exec, KernelKind::other(8, 8), &[a], &[a])
                .unwrap();
            let _ = sim.instantiate_graph(exec).unwrap();
            sim.upload_graph(exec).unwrap();
            let t0 = sim.clock_ns();
            sim.graph_node_set_enabled(exec, 1, false).unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        assert!(
            run_set < run_update,
            "set_enabled={run_set} update={run_update}"
        );
        assert_eq!(run_set, 300);
        assert_eq!(run_update, 9_000);
    }

    #[test]
    fn graph_node_set_enabled_keeps_mem_chain_and_skips_child() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let exec = sim.create_graph(d, s).unwrap();
        let scratch = sim.graph_add_alloc(exec, 4096).unwrap();
        sim.graph_add_kernel(exec, KernelKind::other(8, 8), &[scratch], &[scratch])
            .unwrap();
        sim.graph_add_dependencies(exec, 0, 1).unwrap();
        sim.graph_add_free(exec, scratch).unwrap();
        sim.graph_add_dependencies(exec, 1, 2).unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        sim.graph_node_set_enabled(exec, 1, false).unwrap();
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
        assert!(!sim.is_resident(scratch, d).unwrap());
        let leaf0 = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(leaf0, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim.instantiate_graph(leaf0).unwrap();
        let leaf1 = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(leaf1, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim.instantiate_graph(leaf1).unwrap();
        let parent = sim.create_graph(d, s).unwrap();
        sim.graph_add_child(parent, leaf0).unwrap();
        sim.graph_add_child(parent, leaf1).unwrap();
        let _ = sim.instantiate_graph(parent).unwrap();
        sim.graph_node_set_enabled(parent, 1, false).unwrap();
        let n = sim.launch_graph(parent, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
    }

    #[test]
    fn graph_node_set_enabled_rejects_uninstantiated_mem_and_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let exec = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(exec, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let err = sim.graph_node_set_enabled(exec, 0, false).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        let err = sim.graph_node_get_enabled(exec, 0).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _ = sim.instantiate_graph(exec).unwrap();
        let mem = sim.create_graph(d, s).unwrap();
        let scratch = sim.graph_add_alloc(mem, 4096).unwrap();
        sim.graph_add_free(mem, scratch).unwrap();
        sim.graph_add_dependencies(mem, 0, 1).unwrap();
        let _ = sim.instantiate_graph(mem).unwrap();
        let err = sim.graph_node_set_enabled(mem, 0, false).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("mem"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        let err = sim.graph_node_set_enabled(exec, 0, false).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn graph_exec_child_set_params_swaps_without_second_graph() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_set_params_ns = 400;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        let leaf_a = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(leaf_a, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let leaf_a_exec = sim.instantiate_graph(leaf_a).unwrap();
        let leaf_b = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(leaf_b, KernelKind::other(8, 8), &[b], &[b])
            .unwrap();
        let leaf_b_exec = sim.instantiate_graph(leaf_b).unwrap();
        let exec = sim.create_graph(d, s).unwrap();
        sim.graph_add_child(exec, leaf_a).unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        sim.upload_graph(exec).unwrap();
        assert!(sim.graph_uploaded(exec).unwrap());
        let (node, nested) = sim.graph_unique_child(exec).unwrap();
        assert_eq!(node, 0);
        assert_eq!(nested, leaf_a_exec);
        let src = sim.create_graph(d, s).unwrap();
        sim.graph_add_child(src, leaf_b).unwrap();
        let err = sim.update_graph(exec, src).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("topology"), "{why}"),
            other => panic!("{other:?}"),
        }
        let t0 = sim.clock_ns();
        sim.graph_exec_child_set_params(exec, node, leaf_b).unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(400));
        assert!(!sim.graph_uploaded(exec).unwrap());
        let (_, nested) = sim.graph_unique_child(exec).unwrap();
        assert_eq!(nested, leaf_b_exec);
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(b, d).unwrap());
    }

    #[test]
    fn graph_exec_child_set_params_beats_instantiate_wall() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_instantiate_ns = 80_000;
            g.graph_update_ns = 9_000;
            g.graph_set_params_ns = 300;
            g.graph_upload_ns = 1_000;
            g.graph_launch_ns = 1_000;
        }
        let d = DeviceId(0);
        let s = StreamId(0);
        let run_inst = {
            let mut sim = Sim::new(p.clone());
            let a = sim.malloc(d, 4096).unwrap();
            let b = sim.malloc(d, 4096).unwrap();
            let leaf_a = sim.create_graph(d, s).unwrap();
            sim.graph_add_kernel(leaf_a, KernelKind::other(8, 8), &[a], &[a])
                .unwrap();
            let _ = sim.instantiate_graph(leaf_a).unwrap();
            let leaf_b = sim.create_graph(d, s).unwrap();
            sim.graph_add_kernel(leaf_b, KernelKind::other(8, 8), &[b], &[b])
                .unwrap();
            let _ = sim.instantiate_graph(leaf_b).unwrap();
            let exec = sim.create_graph(d, s).unwrap();
            sim.graph_add_child(exec, leaf_a).unwrap();
            let _ = sim.instantiate_graph(exec).unwrap();
            sim.upload_graph(exec).unwrap();
            let t0 = sim.clock_ns();
            sim.destroy_graph(exec).unwrap();
            let next = sim.create_graph(d, s).unwrap();
            sim.graph_add_child(next, leaf_b).unwrap();
            let _ = sim.instantiate_graph(next).unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let run_set = {
            let mut sim = Sim::new(p);
            let a = sim.malloc(d, 4096).unwrap();
            let b = sim.malloc(d, 4096).unwrap();
            let leaf_a = sim.create_graph(d, s).unwrap();
            sim.graph_add_kernel(leaf_a, KernelKind::other(8, 8), &[a], &[a])
                .unwrap();
            let _ = sim.instantiate_graph(leaf_a).unwrap();
            let leaf_b = sim.create_graph(d, s).unwrap();
            sim.graph_add_kernel(leaf_b, KernelKind::other(8, 8), &[b], &[b])
                .unwrap();
            let _ = sim.instantiate_graph(leaf_b).unwrap();
            let exec = sim.create_graph(d, s).unwrap();
            sim.graph_add_child(exec, leaf_a).unwrap();
            let _ = sim.instantiate_graph(exec).unwrap();
            sim.upload_graph(exec).unwrap();
            let t0 = sim.clock_ns();
            sim.graph_exec_child_set_params(exec, 0, leaf_b).unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        assert!(
            run_set < run_inst,
            "set_params={run_set} instantiate={run_inst}"
        );
        assert_eq!(run_set, 300);
    }

    #[test]
    fn graph_exec_child_set_params_allows_mem_nodes() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        let leaf_a = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(leaf_a, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim.instantiate_graph(leaf_a).unwrap();
        let leaf_b = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(leaf_b, KernelKind::other(8, 8), &[b], &[b])
            .unwrap();
        let _ = sim.instantiate_graph(leaf_b).unwrap();
        let exec = sim.create_graph(d, s).unwrap();
        let scratch = sim.graph_add_alloc(exec, 4096).unwrap();
        sim.graph_add_child(exec, leaf_a).unwrap();
        sim.graph_add_dependencies(exec, 0, 1).unwrap();
        sim.graph_add_free(exec, scratch).unwrap();
        sim.graph_add_dependencies(exec, 1, 2).unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        let src = sim.create_graph(d, s).unwrap();
        let scratch2 = sim.graph_add_alloc(src, 4096).unwrap();
        sim.graph_add_child(src, leaf_a).unwrap();
        sim.graph_add_dependencies(src, 0, 1).unwrap();
        sim.graph_add_free(src, scratch2).unwrap();
        sim.graph_add_dependencies(src, 1, 2).unwrap();
        let err = sim.update_graph(exec, src).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("mem"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.graph_exec_child_set_params(exec, 1, leaf_b).unwrap();
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 3);
        sim.synchronize().unwrap();
        assert!(!sim.is_resident(scratch, d).unwrap());
        assert!(sim.is_resident(b, d).unwrap());
    }

    #[test]
    fn graph_exec_child_set_params_rejects_uninstantiated_topology_and_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        let leaf_a = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(leaf_a, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim.instantiate_graph(leaf_a).unwrap();
        let leaf_b = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(leaf_b, KernelKind::other(8, 8), &[b], &[b])
            .unwrap();
        let _ = sim.instantiate_graph(leaf_b).unwrap();
        let memcpy = sim.create_graph(d, s).unwrap();
        sim.graph_add_memcpy(
            memcpy,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d),
                alloc: b,
                bytes: 4096,
                offset: 0,
                ..MemcpyOp::default()
            },
        )
        .unwrap();
        let _ = sim.instantiate_graph(memcpy).unwrap();
        let exec = sim.create_graph(d, s).unwrap();
        sim.graph_add_child(exec, leaf_a).unwrap();
        let err = sim
            .graph_exec_child_set_params(exec, 0, leaf_b)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _ = sim.instantiate_graph(exec).unwrap();
        let raw = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(raw, KernelKind::other(8, 8), &[b], &[b])
            .unwrap();
        let err = sim.graph_exec_child_set_params(exec, 0, raw).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        let kern = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(kern, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim.instantiate_graph(kern).unwrap();
        let err = sim
            .graph_exec_child_set_params(kern, 0, leaf_b)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not a child"), "{why}"),
            other => panic!("{other:?}"),
        }
        let err = sim
            .graph_exec_child_set_params(exec, 0, memcpy)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("topology"), "{why}"),
            other => panic!("{other:?}"),
        }
        let err = sim.graph_exec_child_set_params(exec, 0, exec).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("self"), "{why}"),
            other => panic!("{other:?}"),
        }
        let mid = sim.create_graph(d, s).unwrap();
        sim.graph_add_child(mid, leaf_a).unwrap();
        let _ = sim.instantiate_graph(mid).unwrap();
        let outer = sim.create_graph(d, s).unwrap();
        sim.graph_add_child(outer, mid).unwrap();
        let _ = sim.instantiate_graph(outer).unwrap();
        let wrap = sim.create_graph(d, s).unwrap();
        sim.graph_add_child(wrap, outer).unwrap();
        let _ = sim.instantiate_graph(wrap).unwrap();
        let err = sim.graph_exec_child_set_params(outer, 0, wrap).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("cyclic"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        let err = sim
            .graph_exec_child_set_params(exec, 0, leaf_b)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        let two = sim.create_graph(d, s).unwrap();
        sim.graph_add_child(two, leaf_a).unwrap();
        sim.graph_add_child(two, leaf_b).unwrap();
        match sim.graph_try_unique_child(two) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not unique"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert!(sim.graph_try_unique_child(leaf_a).unwrap().is_none());
        let err = sim.graph_unique_child(leaf_a).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not a child"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_exec_child_set_params_rejects_gpu_mismatch() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let a = sim.malloc(d0, 4096).unwrap();
        let b = sim.malloc(d1, 4096).unwrap();
        let leaf0 = sim.create_graph(d0, s).unwrap();
        sim.graph_add_kernel(leaf0, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim.instantiate_graph(leaf0).unwrap();
        let leaf1 = sim.create_graph(d1, s).unwrap();
        sim.graph_add_kernel(leaf1, KernelKind::other(8, 8), &[b], &[b])
            .unwrap();
        let _ = sim.instantiate_graph(leaf1).unwrap();
        let exec = sim.create_graph(d0, s).unwrap();
        sim.graph_add_child(exec, leaf0).unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        let err = sim.graph_exec_child_set_params(exec, 0, leaf1).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("gpu"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_exec_event_record_set_event_retargets_without_second_graph() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_set_params_ns = 400;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let e1 = EventId(1);
        let e2 = EventId(2);
        sim.create_event(e1).unwrap();
        sim.create_event(e2).unwrap();
        let exec = sim.create_graph(d, s).unwrap();
        sim.graph_add_event_record(exec, e1, false).unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        sim.upload_graph(exec).unwrap();
        let (node, ev) = sim.graph_unique_event_record(exec).unwrap();
        assert_eq!(node, 0);
        assert_eq!(ev, e1);
        let t0 = sim.clock_ns();
        sim.graph_exec_event_record_set_event(exec, node, e2)
            .unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(400));
        assert!(!sim.graph_uploaded(exec).unwrap());
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert!(sim.query_event(e2).unwrap());
        assert!(!sim.query_event(e1).unwrap());
    }

    #[test]
    fn graph_exec_event_wait_set_event_retargets_live_record() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let e1 = EventId(1);
        let e2 = EventId(2);
        sim.create_event(e1).unwrap();
        sim.create_event(e2).unwrap();
        enq(sim.record_event(d, e2, s));
        sim.synchronize().unwrap();
        let exec = sim.create_graph(d, s).unwrap();
        sim.graph_add_event_wait(exec, e1, false).unwrap();
        sim.graph_add_kernel(exec, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_dependencies(exec, 0, 1).unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        let (node, ev) = sim.graph_unique_event_wait(exec).unwrap();
        assert_eq!(ev, e1);
        sim.graph_exec_event_wait_set_event(exec, node, e2).unwrap();
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(a, d).unwrap());
    }

    #[test]
    fn graph_exec_event_set_event_beats_exec_update_wall() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_instantiate_ns = 80_000;
            g.graph_update_ns = 9_000;
            g.graph_set_params_ns = 300;
            g.graph_upload_ns = 1_000;
            g.graph_launch_ns = 1_000;
        }
        let d = DeviceId(0);
        let s = StreamId(0);
        let run_update = {
            let mut sim = Sim::new(p.clone());
            sim.create_event(EventId(1)).unwrap();
            sim.create_event(EventId(2)).unwrap();
            let exec = sim.create_graph(d, s).unwrap();
            sim.graph_add_event_record(exec, EventId(1), false).unwrap();
            let _ = sim.instantiate_graph(exec).unwrap();
            sim.upload_graph(exec).unwrap();
            let src = sim.create_graph(d, s).unwrap();
            sim.graph_add_event_record(src, EventId(2), false).unwrap();
            let t0 = sim.clock_ns();
            sim.update_graph(exec, src).unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let run_set = {
            let mut sim = Sim::new(p);
            sim.create_event(EventId(1)).unwrap();
            sim.create_event(EventId(2)).unwrap();
            let exec = sim.create_graph(d, s).unwrap();
            sim.graph_add_event_record(exec, EventId(1), false).unwrap();
            let _ = sim.instantiate_graph(exec).unwrap();
            sim.upload_graph(exec).unwrap();
            let t0 = sim.clock_ns();
            sim.graph_exec_event_record_set_event(exec, 0, EventId(2))
                .unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        assert!(
            run_set < run_update,
            "set_params={run_set} update={run_update}"
        );
        assert_eq!(run_set, 300);
        assert_eq!(run_update, 9_000);
    }

    #[test]
    fn graph_exec_event_set_event_allows_mem_nodes() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.create_event(EventId(1)).unwrap();
        sim.create_event(EventId(2)).unwrap();
        let exec = sim.create_graph(d, s).unwrap();
        let scratch = sim.graph_add_alloc(exec, 4096).unwrap();
        sim.graph_add_event_record(exec, EventId(1), false).unwrap();
        sim.graph_add_dependencies(exec, 0, 1).unwrap();
        sim.graph_add_free(exec, scratch).unwrap();
        sim.graph_add_dependencies(exec, 1, 2).unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        let src = sim.create_graph(d, s).unwrap();
        let scratch2 = sim.graph_add_alloc(src, 4096).unwrap();
        sim.graph_add_event_record(src, EventId(1), false).unwrap();
        sim.graph_add_dependencies(src, 0, 1).unwrap();
        sim.graph_add_free(src, scratch2).unwrap();
        sim.graph_add_dependencies(src, 1, 2).unwrap();
        let err = sim.update_graph(exec, src).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("mem"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.graph_exec_event_record_set_event(exec, 1, EventId(2))
            .unwrap();
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 3);
        sim.synchronize().unwrap();
        assert!(sim.query_event(EventId(2)).unwrap());
        assert!(!sim.is_resident(scratch, d).unwrap());
    }

    #[test]
    fn graph_exec_event_set_event_rejects_uninstantiated_and_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        sim.create_event(EventId(1)).unwrap();
        sim.create_event(EventId(2)).unwrap();
        let exec = sim.create_graph(d, s).unwrap();
        sim.graph_add_event_record(exec, EventId(1), false).unwrap();
        let err = sim
            .graph_exec_event_record_set_event(exec, 0, EventId(2))
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _ = sim.instantiate_graph(exec).unwrap();
        let kern = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(kern, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim.instantiate_graph(kern).unwrap();
        let err = sim
            .graph_exec_event_record_set_event(kern, 0, EventId(2))
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not an event record"), "{why}"),
            other => panic!("{other:?}"),
        }
        let err = sim
            .graph_exec_event_wait_set_event(exec, 0, EventId(2))
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not an event wait"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.graph_exec_event_record_set_event(exec, 0, EventId(9)) {
            Err(SimError::UnknownEvent { event: 9 }) => {}
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        let err = sim
            .graph_exec_event_record_set_event(exec, 0, EventId(2))
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        assert!(sim.graph_try_unique_event_wait(exec).unwrap().is_none());
    }

    #[test]
    fn update_graph_rejects_topology_and_uninstantiated() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let exec = sim.end_capture().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        let memcpy_src = sim.end_capture().unwrap();
        let err = sim.update_graph(exec, memcpy_src).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _ = sim.instantiate_graph(exec).unwrap();
        let err = sim.update_graph(exec, memcpy_src).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("topology"), "{why}"),
            other => panic!("{other:?}"),
        }
        let err = sim.update_graph(exec, exec).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("same id"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn update_graph_with_info_reports_success_and_node_type() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_update_ns = 7_000;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
        sim.synchronize().unwrap();
        let exec = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(exec, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        let src = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(src, KernelKind::other(8, 8), &[b], &[b])
            .unwrap();
        let t0 = sim.clock_ns();
        let mut info = GraphExecUpdateResultInfo::default();
        sim.update_graph_with_info(exec, src, &mut info).unwrap();
        assert_eq!(info.result, GraphExecUpdateResult::Success);
        assert_eq!(info.error_node, None);
        assert_eq!(info.error_from_node, None);
        assert_eq!(sim.clock_ns(), t0.saturating_add(7_000));
        let memcpy_src = sim.create_graph(d, s).unwrap();
        sim.graph_add_memcpy(
            memcpy_src,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d),
                alloc: a,
                bytes: 4096,
                offset: 0,
                ..MemcpyOp::default()
            },
        )
        .unwrap();
        let err = sim
            .update_graph_with_info(exec, memcpy_src, &mut info)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("topology"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(info.result, GraphExecUpdateResult::NodeTypeChanged);
        assert_eq!(info.error_node, Some(0));
        assert_eq!(info.error_from_node, Some(0));
        let extra = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(extra, KernelKind::other(8, 8), &[b], &[b])
            .unwrap();
        sim.graph_add_kernel(extra, KernelKind::other(8, 8), &[b], &[b])
            .unwrap();
        let err = sim
            .update_graph_with_info(exec, extra, &mut info)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("topology"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(info.result, GraphExecUpdateResult::TopologyChanged);
        assert_eq!(info.error_node, Some(1));
        assert_eq!(info.error_from_node, None);
    }

    #[test]
    fn update_graph_with_info_classifies_deps_mem_and_params() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
        sim.synchronize().unwrap();
        let exec = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(exec, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_kernel(exec, KernelKind::other(8, 8), &[b], &[b])
            .unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        let src = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(src, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_kernel(src, KernelKind::other(8, 8), &[b], &[b])
            .unwrap();
        sim.graph_add_dependencies(src, 0, 1).unwrap();
        let mut info = GraphExecUpdateResultInfo::default();
        let err = sim
            .update_graph_with_info(exec, src, &mut info)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("topology"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(info.result, GraphExecUpdateResult::DependenciesChanged);
        assert_eq!(info.error_node, Some(1));
        assert_eq!(info.error_from_node, Some(0));
        let coop = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(coop, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_cooperative_kernel(coop, KernelKind::other(8, 8), &[b], &[b])
            .unwrap();
        let err = sim
            .update_graph_with_info(exec, coop, &mut info)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("topology"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(info.result, GraphExecUpdateResult::ParametersChanged);
        assert_eq!(info.error_node, Some(1));
        assert_eq!(info.error_from_node, Some(1));
        let mem = sim.create_graph(d, s).unwrap();
        let scratch = sim.graph_add_alloc(mem, 4096).unwrap();
        sim.graph_add_kernel(mem, KernelKind::other(8, 8), &[a], &[scratch])
            .unwrap();
        sim.graph_add_dependencies(mem, 0, 1).unwrap();
        let mem_exec = sim.create_graph(d, s).unwrap();
        let scratch2 = sim.graph_add_alloc(mem_exec, 4096).unwrap();
        sim.graph_add_kernel(mem_exec, KernelKind::other(8, 8), &[a], &[scratch2])
            .unwrap();
        sim.graph_add_dependencies(mem_exec, 0, 1).unwrap();
        let _ = sim.instantiate_graph(mem_exec).unwrap();
        let err = sim
            .update_graph_with_info(mem_exec, mem, &mut info)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("mem"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(info.result, GraphExecUpdateResult::NotSupported);
        assert_eq!(info.error_node, Some(0));
        let dl = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(dl, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim
            .instantiate_graph_with_flags(dl, GraphInstantiateFlags::DEVICE_LAUNCH)
            .unwrap();
        let dl_src = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(dl_src, KernelKind::other(8, 8), &[b], &[b])
            .unwrap();
        let err = sim
            .update_graph_with_info(dl, dl_src, &mut info)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("device launch"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(info.result, GraphExecUpdateResult::NotSupported);
        assert_eq!(info.error_node, None);
        sim.begin_capture(d, s).unwrap();
        let err = sim
            .update_graph_with_info(exec, src, &mut info)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(info.result, GraphExecUpdateResult::Error);
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn update_graph_with_info_attributes_and_child_id() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.create_event(EventId(1)).unwrap();
        let exec = sim.create_graph(d, s).unwrap();
        sim.graph_add_event_record(exec, EventId(1), false).unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        let src = sim.create_graph(d, s).unwrap();
        sim.graph_add_event_record(src, EventId(1), true).unwrap();
        let mut info = GraphExecUpdateResultInfo::default();
        let err = sim
            .update_graph_with_info(exec, src, &mut info)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("topology"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(info.result, GraphExecUpdateResult::AttributesChanged);
        assert_eq!(info.error_node, Some(0));
        assert_eq!(info.error_from_node, Some(0));
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let child_a = sim.end_capture().unwrap();
        let _ = sim.instantiate_graph(child_a).unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let child_b = sim.end_capture().unwrap();
        let _ = sim.instantiate_graph(child_b).unwrap();
        let parent = sim.create_graph(d, s).unwrap();
        sim.graph_add_child(parent, child_a).unwrap();
        let _ = sim.instantiate_graph(parent).unwrap();
        let parent_src = sim.create_graph(d, s).unwrap();
        sim.graph_add_child(parent_src, child_b).unwrap();
        let err = sim
            .update_graph_with_info(parent, parent_src, &mut info)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("topology"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(info.result, GraphExecUpdateResult::TopologyChanged);
        assert_eq!(info.error_node, Some(0));
        assert_eq!(info.error_from_node, Some(0));
    }

    #[test]
    fn user_object_release_fires_destructor() {
        let mut sim = Sim::new(h100());
        let t0 = sim.clock_ns();
        let obj = sim
            .user_object_create(7, 1, UserObjectFlags::NO_DESTRUCTOR_SYNC)
            .unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(1));
        assert_eq!(sim.user_object_refs(obj).unwrap(), 1);
        sim.user_object_retain(obj, 1).unwrap();
        assert_eq!(sim.user_object_refs(obj).unwrap(), 2);
        sim.user_object_release(obj, 1).unwrap();
        assert_eq!(sim.user_object_refs(obj).unwrap(), 1);
        assert!(sim.user_object_destructors().is_empty());
        sim.user_object_release(obj, 1).unwrap();
        assert_eq!(sim.user_object_destructors(), &[(obj, 7)]);
        match sim.user_object_refs(obj).unwrap_err() {
            SimError::Invalid { why } => assert!(why.contains("unknown user object"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim
            .user_object_create(1, 0, UserObjectFlags::NO_DESTRUCTOR_SYNC)
            .unwrap_err()
        {
            SimError::Invalid { why } => assert!(why.contains("initial"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.user_object_create(1, 1, 0).unwrap_err() {
            SimError::Invalid { why } => assert!(why.contains("flags"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(DeviceId(0), StreamId(0)).unwrap();
        match sim
            .user_object_create(1, 1, UserObjectFlags::NO_DESTRUCTOR_SYNC)
            .unwrap_err()
        {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn graph_retain_user_object_move_and_clone() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let g = sim.create_graph(d, s).unwrap();
        let obj = sim
            .user_object_create(3, 1, UserObjectFlags::NO_DESTRUCTOR_SYNC)
            .unwrap();
        sim.graph_retain_user_object(g, obj, 1, GraphUserObjectFlags::MOVE)
            .unwrap();
        assert_eq!(sim.user_object_refs(obj).unwrap(), 1);
        assert_eq!(sim.user_object_graph_refs(g, obj).unwrap(), 1);
        match sim.user_object_release(obj, 1).unwrap_err() {
            SimError::Invalid { why } => assert!(why.contains("user object refs"), "{why}"),
            other => panic!("{other:?}"),
        }
        let clone = sim.clone_graph(g).unwrap();
        assert_eq!(sim.user_object_graph_refs(clone, obj).unwrap(), 0);
        sim.destroy_graph(g).unwrap();
        assert_eq!(sim.user_object_destructors(), &[(obj, 3)]);
        match sim.user_object_graph_refs(clone, obj).unwrap_err() {
            SimError::Invalid { why } => assert!(why.contains("unknown user object"), "{why}"),
            other => panic!("{other:?}"),
        }
        let g2 = sim.create_graph(d, s).unwrap();
        let obj2 = sim
            .user_object_create(9, 1, UserObjectFlags::NO_DESTRUCTOR_SYNC)
            .unwrap();
        sim.graph_retain_user_object(g2, obj2, 1, 0).unwrap();
        assert_eq!(sim.user_object_refs(obj2).unwrap(), 2);
        sim.destroy_graph(g2).unwrap();
        assert_eq!(sim.user_object_refs(obj2).unwrap(), 1);
        sim.user_object_release(obj2, 1).unwrap();
        assert_eq!(sim.user_object_destructors(), &[(obj, 3), (obj2, 9)]);
        let g3 = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g3, KernelKind::other(8, 8), &[], &[])
            .unwrap();
        let exec = sim.instantiate_graph(g3).unwrap();
        let obj3 = sim
            .user_object_create(1, 1, UserObjectFlags::NO_DESTRUCTOR_SYNC)
            .unwrap();
        match sim.graph_retain_user_object(exec, obj3, 1, 0).unwrap_err() {
            SimError::Invalid { why } => assert!(why.contains("instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.graph_retain_user_object(g3, obj3, 1, GraphUserObjectFlags::MOVE)
            .unwrap();
        sim.destroy_graph(exec).unwrap();
        assert_eq!(sim.user_object_refs(obj3).unwrap(), 1);
        sim.destroy_graph(g3).unwrap();
        assert!(sim
            .user_object_destructors()
            .iter()
            .any(|(id, fn_id)| *id == obj3 && *fn_id == 1));
    }

    #[test]
    fn instantiate_and_update_cannot_run_during_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let exec = sim.end_capture().unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let src = sim.end_capture().unwrap();
        sim.begin_capture(d, s).unwrap();
        let inst = sim.instantiate_graph(exec).unwrap_err();
        match inst {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let upd = sim.update_graph(exec, src).unwrap_err();
        match upd {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let cl = sim.clone_graph(exec).unwrap_err();
        match cl {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let up = sim.upload_graph(exec).unwrap_err();
        match up {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn clone_graph_is_independent_and_not_instantiated() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_clone_ns = 9_000;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let src = sim.end_capture().unwrap();
        let _ = sim.instantiate_graph(src).unwrap();
        let t0 = sim.clock_ns();
        let clone = sim.clone_graph(src).unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(9_000));
        assert_ne!(clone, src);
        assert_eq!(sim.graph_len(clone).unwrap(), 1);
        assert!(!sim.graph_instantiated(clone).unwrap());
        assert!(sim.graph_instantiated(src).unwrap());
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[b], &[b], s));
        let alt = sim.end_capture().unwrap();
        sim.update_graph(src, alt).unwrap();
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(src, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert!(!sim.graph_instantiated(clone).unwrap());
        let n2 = sim.launch_graph(clone, s).unwrap();
        assert_eq!(n2, 1);
        assert!(sim.graph_instantiated(clone).unwrap());
        let err = sim.synchronize().unwrap_err();
        match err {
            SimError::NotResident { alloc, device } => {
                assert_eq!(alloc, a);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn destroy_graph_forbids_later_launch_and_spares_clones() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let g = sim.end_capture().unwrap();
        let clone = sim.clone_graph(g).unwrap();
        sim.destroy_graph(g).unwrap();
        let err = sim.launch_graph(g, s).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("unknown graph"), "{why}"),
            other => panic!("{other:?}"),
        }
        let n = sim.launch_graph(clone, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        let cap = sim.destroy_graph(clone).unwrap_err();
        match cap {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _end = sim.end_capture().unwrap();
        sim.destroy_graph(clone).unwrap();
        let gone = sim.graph_len(clone).unwrap_err();
        match gone {
            SimError::Invalid { why } => assert!(why.contains("unknown graph"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn memset_requires_residency_then_advances_clock() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let h = sim.alloc_host_pinned(4096).unwrap();
        enq(sim.memset(d, h, 4096, s));
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, h);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
        let mut sim = Sim::new(h100());
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        let t0 = sim.clock_ns();
        enq(sim.memset(d, a, 4096, s));
        sim.synchronize().unwrap();
        assert!(sim.clock_ns() > t0);
    }

    #[test]
    fn disable_peer_blocks_d2d_when_link_exists() {
        let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let a = sim.alloc(d0, bytes, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d0, a, bytes, s));
        sim.synchronize().unwrap();
        assert!(sim.peer_access(d0, d1));
        sim.disable_peer(d0, d1).unwrap();
        assert!(!sim.peer_access(d0, d1));
        enq(sim.memcpy_device_to_device(d0, d1, a, bytes, s));
        match sim.synchronize() {
            Err(SimError::PeerDisabled { src, dst }) => {
                assert_eq!(src, d0);
                assert_eq!(dst, d1);
            }
            other => panic!("{other:?}"),
        }
        sim.enable_peer(d0, d1).unwrap();
        enq(sim.memcpy_device_to_device(d0, d1, a, bytes, s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(a, d1).unwrap());
    }

    #[test]
    fn enable_peer_with_flags_requires_zero() {
        let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        sim.disable_peer(d0, d1).unwrap();
        assert!(!sim.peer_access(d0, d1));
        match sim.enable_peer_with_flags(d0, d1, 1) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("peer access flags"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        assert!(!sim.peer_access(d0, d1));
        sim.enable_peer_with_flags(d0, d1, PeerAccessFlags::DEFAULT)
            .unwrap();
        assert!(sim.peer_access(d0, d1));
        sim.disable_peer(d0, d1).unwrap();
        sim.begin_capture(d0, StreamId(0)).unwrap();
        sim.enable_peer_with_flags(d0, d1, PeerAccessFlags::DEFAULT)
            .unwrap();
        assert!(sim.peer_access(d0, d1));
        match sim.enable_peer_with_flags(d0, d1, 1) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("peer access flags"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        sim.enable_peer(d0, d1).unwrap();
    }

    #[test]
    fn memcpy_peer_is_host_sync_cuda_memcpy_peer() {
        let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let a = sim.alloc(d0, bytes, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d0, a, bytes, s));
        sim.synchronize().unwrap();
        let t0 = sim.clock_ns();
        let id = sim.memcpy_peer(d0, d1, a, bytes, s).unwrap();
        assert!(sim.stream_is_idle(d0, s).unwrap());
        assert!(sim.is_resident(a, d1).unwrap());
        assert!(sim.clock_ns() > t0);
        assert_eq!(sim.op_stream(id), Some(s));
        sim.begin_capture(d0, s).unwrap();
        match sim.memcpy_peer(d0, d1, a, bytes, s) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        enq(sim.memcpy_peer_async(d0, d1, a, bytes, s));
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 1);
    }

    #[test]
    fn memcpy_peer_3d_is_host_sync_cuda_memcpy3d_peer() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let (a, pitch) = sim.malloc_3d(d0, 256, 4, 4).unwrap();
        enq(sim.memcpy(
            d0,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d0),
                alloc: a,
                bytes: 256,
                height: 4,
                src_pitch: 256,
                dst_pitch: pitch,
                depth: 4,
                src_height: 4,
                dst_height: 4,
                ..MemcpyOp::default()
            },
            s,
        ));
        sim.synchronize().unwrap();
        let moved0 = sim.bytes_moved();
        let t0 = sim.clock_ns();
        let op = MemcpyOp {
            alloc: a,
            bytes: 256,
            height: 4,
            src_pitch: pitch,
            dst_pitch: pitch,
            depth: 4,
            src_height: 4,
            dst_height: 4,
            ..MemcpyOp::default()
        };
        let id = sim.memcpy_peer_3d(d0, d1, op.clone(), s).unwrap();
        assert!(sim.stream_is_idle(d0, s).unwrap());
        assert!(sim.is_resident(a, d1).unwrap());
        assert!(sim.clock_ns() > t0);
        assert_eq!(sim.bytes_moved(), moved0 + 4096);
        assert_eq!(sim.op_stream(id), Some(s));
        sim.begin_capture(d0, s).unwrap();
        match sim.memcpy_peer_3d(d0, d1, op.clone(), s) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        enq(sim.memcpy_peer_3d_async(d0, d1, op, s));
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 1);
    }

    #[test]
    fn memcpy_peer_2d_is_host_sync_cuda_memcpy2d_peer() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let (a, pitch) = sim.malloc_pitch(d0, 256, 8).unwrap();
        enq(sim.memcpy(
            d0,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d0),
                alloc: a,
                bytes: 256,
                height: 8,
                src_pitch: 256,
                dst_pitch: pitch,
                ..MemcpyOp::default()
            },
            s,
        ));
        sim.synchronize().unwrap();
        let moved0 = sim.bytes_moved();
        let t0 = sim.clock_ns();
        let op = MemcpyOp {
            alloc: a,
            bytes: 256,
            height: 8,
            src_pitch: pitch,
            dst_pitch: pitch,
            ..MemcpyOp::default()
        };
        let id = sim.memcpy_peer_2d(d0, d1, op.clone(), s).unwrap();
        assert!(sim.stream_is_idle(d0, s).unwrap());
        assert!(sim.is_resident(a, d1).unwrap());
        assert!(sim.clock_ns() > t0);
        assert_eq!(sim.bytes_moved(), moved0 + 2048);
        assert_eq!(sim.op_stream(id), Some(s));
        sim.begin_capture(d0, s).unwrap();
        match sim.memcpy_peer_2d(d0, d1, op.clone(), s) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        enq(sim.memcpy_peer_2d_async(d0, d1, op, s));
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 1);
    }

    #[test]
    fn legacy_null_stream_serializes_copy_with_compute() {
        let d = DeviceId(0);
        let bytes = 32u64 << 20;
        let k = KernelKind::GroupedMoeGemm {
            experts: 8,
            tokens_per_expert: 16,
            hidden: 2048,
            ff: 2048,
            dtype: DType::Fp16,
        };
        let run = |legacy: bool| {
            let mut sim = Sim::new(h100());
            sim.set_legacy_null_stream(legacy);
            let w = sim.alloc(d, bytes, StreamId(0)).unwrap();
            let c = sim.alloc(d, bytes, StreamId(1)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, w, bytes, StreamId(0)));
            enq(sim.memcpy_pinned_to_device(d, c, bytes, StreamId(1)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, k.clone(), &[w], &[w], StreamId::NULL));
            enq(sim.memcpy_pinned_to_device(d, c, bytes, StreamId(1)));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let serial = run(true);
        let overlap = run(false);
        assert!(
            serial > overlap,
            "legacy null stream must serialize; serial={serial} overlap={overlap}"
        );
        assert!(Sim::new(h100()).stream_is_idle(d, StreamId(0)).unwrap());
    }

    #[test]
    fn third_copy_waits_when_two_engines() {
        let p = HardwareProfile::parse("gpus=1\ncopy_engines=2\n").unwrap();
        let d = DeviceId(0);
        let bytes = 16u64 << 20;
        let n_copies = |n: u16| {
            let mut sim = Sim::new(p.clone());
            for i in 0..n {
                let s = StreamId(i);
                let a = sim.alloc(d, bytes, s).unwrap();
                enq(sim.memcpy_pinned_to_device(d, a, bytes, s));
            }
            sim.synchronize().unwrap();
            sim.clock_ns()
        };
        let two = n_copies(2);
        let three = n_copies(3);
        assert!(
            three > two,
            "third H2D must wait for a copy engine; two={two} three={three}"
        );
    }

    #[test]
    fn event_complete_after_record_sync() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let ev = EventId(7);
        assert!(!sim.event_complete(ev));
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.record_event(d, ev, s));
        assert!(!sim.event_complete(ev));
        sim.synchronize().unwrap();
        assert!(sim.event_complete(ev));
        assert!(sim.stream_is_idle(d, s).unwrap());
    }

    #[test]
    fn operations_are_a_stream_ordered_dag() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.kernel(d, KernelKind::other(8, 4096), &[a], &[a], s));
        sim.synchronize().unwrap();
        let ops: Vec<Operation> = sim.operations().collect();
        assert!(ops.iter().any(|o| matches!(o.kind, GpuOp::Alloc { .. })));
        assert!(ops.iter().any(|o| matches!(o.kind, GpuOp::Memcpy(_))));
        let kernel = ops
            .iter()
            .find(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .expect("kernel");
        assert!(kernel.done);
        assert!(!kernel.deps.is_empty());
        assert_eq!(sim.operation(kernel.id).unwrap().kind, kernel.kind);
        assert!(kernel.start_ns.is_some());
        assert!(kernel.done_ns.is_some());
        assert!(kernel.duration_ns().is_some());
    }

    #[test]
    fn same_stream_next_op_starts_after_previous_finishes() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.kernel(d, KernelKind::other(1 << 20, 4096), &[a], &[a], s));
        enq(sim.kernel(d, KernelKind::other(1 << 20, 4096), &[a], &[a], s));
        sim.synchronize().unwrap();
        let kernels: Vec<Operation> = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .collect();
        assert_eq!(kernels.len(), 2);
        let a0 = kernels[0].start_ns.expect("k0 start");
        let d0 = kernels[0].done_ns.expect("k0 done");
        let a1 = kernels[1].start_ns.expect("k1 start");
        assert!(
            a1 >= d0,
            "stream[i+1].start >= stream[i].finish; start1={a1} done0={d0} start0={a0}"
        );
    }

    #[test]
    fn pdl_same_stream_overlaps_after_trigger() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.compute_slots = 2;
            g.pdl_trigger_permille = 250;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
        sim.synchronize().unwrap();
        let k = KernelKind::other(1 << 30, 4096);
        enq(sim.kernel_pdl(
            d,
            k.clone(),
            &[a],
            &[a],
            s,
            ProgrammaticLaunch {
                wait: false,
                trigger: true,
            },
        ));
        enq(sim.kernel_pdl(
            d,
            k,
            &[b],
            &[b],
            s,
            ProgrammaticLaunch {
                wait: true,
                trigger: false,
            },
        ));
        sim.synchronize().unwrap();
        let kernels: Vec<Operation> = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .collect();
        assert_eq!(kernels.len(), 2);
        let a0 = kernels[0].start_ns.expect("k0 start");
        let d0 = kernels[0].done_ns.expect("k0 done");
        let a1 = kernels[1].start_ns.expect("k1 start");
        assert!(
            a1 < d0,
            "PDL secondary must start before primary done; start1={a1} done0={d0}"
        );
        assert!(
            a1 >= a0,
            "secondary cannot start before primary; start1={a1} start0={a0}"
        );
    }

    #[test]
    fn pdl_wait_without_trigger_stays_serial() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.compute_slots = 2;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        let k = KernelKind::other(1 << 20, 4096);
        enq(sim.kernel(d, k.clone(), &[a], &[a], s));
        enq(sim.kernel_pdl(
            d,
            k,
            &[a],
            &[a],
            s,
            ProgrammaticLaunch {
                wait: true,
                trigger: false,
            },
        ));
        sim.synchronize().unwrap();
        let kernels: Vec<Operation> = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .collect();
        let d0 = kernels[0].done_ns.expect("k0 done");
        let a1 = kernels[1].start_ns.expect("k1 start");
        assert!(
            a1 >= d0,
            "wait without trigger is serial; start1={a1} done0={d0}"
        );
    }

    #[test]
    fn graph_pdl_attributes_overlap_dependent_kernels() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.compute_slots = 2;
            g.pdl_trigger_permille = 250;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
        sim.synchronize().unwrap();
        let k = KernelKind::other(1 << 30, 4096);
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, k.clone(), &[a], &[a]).unwrap();
        sim.graph_add_kernel(g, k, &[b], &[b]).unwrap();
        sim.graph_add_dependencies(g, 0, 1).unwrap();
        sim.graph_kernel_node_set_pdl(
            g,
            0,
            ProgrammaticLaunch {
                wait: false,
                trigger: true,
            },
        )
        .unwrap();
        sim.graph_kernel_node_set_pdl(
            g,
            1,
            ProgrammaticLaunch {
                wait: true,
                trigger: false,
            },
        )
        .unwrap();
        assert!(sim.graph_kernel_node_get_pdl(g, 0).unwrap().trigger);
        assert!(sim.graph_kernel_node_get_pdl(g, 1).unwrap().wait);
        let _ = sim.instantiate_graph(g).unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
        let kernels: Vec<Operation> = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .collect();
        assert_eq!(kernels.len(), 2);
        let d0 = kernels[0].done_ns.expect("k0 done");
        let a1 = kernels[1].start_ns.expect("k1 start");
        assert!(a1 < d0, "graph PDL must overlap; start1={a1} done0={d0}");
    }

    #[test]
    fn pdl_stream_free_waits_for_overlapped_primary() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.compute_slots = 2;
            g.pdl_trigger_permille = 250;
            g.graph_launch_ns = 100_000;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
        sim.synchronize().unwrap();
        let k = KernelKind::other(1 << 20, 4096);
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, k.clone(), &[a], &[a]).unwrap();
        sim.graph_add_kernel(g, k, &[b], &[b]).unwrap();
        sim.graph_add_dependencies(g, 0, 1).unwrap();
        let both = ProgrammaticLaunch {
            wait: true,
            trigger: true,
        };
        sim.graph_kernel_node_set_pdl(g, 0, both).unwrap();
        sim.graph_kernel_node_set_pdl(g, 1, both).unwrap();
        let _ = sim.instantiate_graph(g).unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 2);
        sim.free(d, a, s)
            .expect("free waits for overlapped primary");
        sim.synchronize().unwrap();
    }

    fn pdl_two_slot() -> HardwareProfile {
        let mut p = h100();
        for g in &mut p.gpus {
            g.compute_slots = 2;
            g.pdl_trigger_permille = 250;
        }
        p
    }

    #[test]
    fn programmatic_event_cross_stream_starts_at_trigger() {
        let mut sim = Sim::new(pdl_two_slot());
        let d = DeviceId(0);
        let s0 = StreamId(0);
        let s1 = StreamId(1);
        let a = sim.alloc(d, 4096, s0).unwrap();
        let b = sim.alloc(d, 4096, s1).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s0));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s1));
        sim.synchronize().unwrap();
        let ev = EventId(1);
        let k = KernelKind::other(1 << 30, 4096);
        enq(sim.kernel_pdl_event(d, k.clone(), &[a], &[a], s0, PdlLaunch::trigger_event(ev)));
        enq(sim.wait_event(d, ev, s1));
        enq(sim.kernel(d, k, &[b], &[b], s1));
        sim.synchronize().unwrap();
        let k0 = sim
            .operations()
            .find(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.stream == s0)
            .expect("primary");
        let k1 = sim
            .operations()
            .find(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.stream == s1)
            .expect("waiter");
        let d0 = k0.done_ns.expect("k0 done");
        let a1 = k1.start_ns.expect("k1 start");
        assert!(
            a1 < d0,
            "programmatic event must unblock at trigger; start1={a1} done0={d0}"
        );
    }

    #[test]
    fn programmatic_event_without_trigger_waits_for_completion() {
        let mut sim = Sim::new(pdl_two_slot());
        let d = DeviceId(0);
        let s0 = StreamId(0);
        let s1 = StreamId(1);
        let a = sim.alloc(d, 4096, s0).unwrap();
        let b = sim.alloc(d, 4096, s1).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s0));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s1));
        sim.synchronize().unwrap();
        let ev = EventId(2);
        let k = KernelKind::other(1 << 30, 4096);
        enq(sim.kernel_pdl_event(
            d,
            k.clone(),
            &[a],
            &[a],
            s0,
            PdlLaunch {
                pdl: ProgrammaticLaunch {
                    wait: false,
                    trigger: false,
                },
                event: Some(ProgrammaticEvent {
                    event: ev,
                    external: false,
                }),
            },
        ));
        enq(sim.wait_event(d, ev, s1));
        enq(sim.kernel(d, k, &[b], &[b], s1));
        sim.synchronize().unwrap();
        let k0 = sim
            .operations()
            .find(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.stream == s0)
            .expect("primary");
        let k1 = sim
            .operations()
            .find(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.stream == s1)
            .expect("waiter");
        let d0 = k0.done_ns.expect("k0 done");
        let a1 = k1.start_ns.expect("k1 start");
        assert!(
            a1 >= d0,
            "no trigger means the event records at completion; start1={a1} done0={d0}"
        );
    }

    #[test]
    fn programmatic_event_query_fires_before_kernel_done() {
        let mut sim = Sim::new(pdl_two_slot());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        let ev = EventId(3);
        enq(sim.kernel_pdl_event(
            d,
            KernelKind::other(1 << 30, 4096),
            &[a],
            &[a],
            s,
            PdlLaunch::trigger_event(ev),
        ));
        sim.synchronize_event(ev).unwrap();
        assert!(sim.query_event(ev).unwrap());
        let k0 = sim
            .operations()
            .find(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .expect("kernel");
        assert!(!k0.done, "query at trigger must precede kernel completion");
        assert!(!sim.stream_is_idle(d, s).unwrap());
        sim.synchronize().unwrap();
        let k0 = sim
            .operations()
            .find(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .expect("kernel done");
        assert!(k0.done);
    }

    #[test]
    fn programmatic_event_elapsed_uses_trigger() {
        let mut sim = Sim::new(pdl_two_slot());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        let start = EventId(4);
        let end = EventId(5);
        enq(sim.record_event(d, start, s));
        enq(sim.kernel_pdl_event(
            d,
            KernelKind::other(1 << 30, 4096),
            &[a],
            &[a],
            s,
            PdlLaunch::trigger_event(end),
        ));
        sim.synchronize().unwrap();
        let elapsed = sim.event_elapsed_ns(start, end).unwrap();
        let k0 = sim
            .operations()
            .find(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .expect("kernel");
        let start_ns = k0.start_ns.expect("start");
        let done_ns = k0.done_ns.expect("done");
        let dur = done_ns.saturating_sub(start_ns);
        assert!(
            elapsed < dur,
            "elapsed through programmatic event must be the trigger, not completion; elapsed={elapsed} dur={dur}"
        );
        assert!(elapsed > 0, "trigger must be after record");
    }

    #[test]
    fn graph_programmatic_event_cross_stream_at_trigger() {
        let mut sim = Sim::new(pdl_two_slot());
        let d = DeviceId(0);
        let s0 = StreamId(0);
        let s1 = StreamId(1);
        let a = sim.alloc(d, 4096, s0).unwrap();
        let b = sim.alloc(d, 4096, s1).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s0));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s1));
        sim.synchronize().unwrap();
        let ev = EventId(6);
        let k = KernelKind::other(1 << 30, 4096);
        let g = sim.create_graph(d, s0).unwrap();
        sim.graph_add_kernel(g, k.clone(), &[a], &[a]).unwrap();
        sim.graph_kernel_node_set_pdl(
            g,
            0,
            ProgrammaticLaunch {
                wait: false,
                trigger: true,
            },
        )
        .unwrap();
        sim.graph_kernel_node_set_programmatic_event(
            g,
            0,
            Some(ProgrammaticEvent {
                event: ev,
                external: false,
            }),
        )
        .unwrap();
        assert_eq!(
            sim.graph_kernel_node_get_programmatic_event(g, 0)
                .unwrap()
                .map(|p| p.event),
            Some(ev)
        );
        let _ = sim.instantiate_graph(g).unwrap();
        let n = sim.launch_graph(g, s0).unwrap();
        assert_eq!(n, 1);
        enq(sim.wait_event(d, ev, s1));
        enq(sim.kernel(d, k, &[b], &[b], s1));
        sim.synchronize().unwrap();
        let k0 = sim
            .operations()
            .find(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.stream == s0)
            .expect("primary");
        let k1 = sim
            .operations()
            .find(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.stream == s1)
            .expect("waiter");
        let d0 = k0.done_ns.expect("k0 done");
        let a1 = k1.start_ns.expect("k1 start");
        assert!(
            a1 < d0,
            "graph programmatic event must unblock at trigger; start1={a1} done0={d0}"
        );
    }

    #[test]
    fn graph_kernel_node_copy_attributes_copies_programmatic_event() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        let ev = EventId(7);
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_kernel_node_set_programmatic_event(
            g,
            0,
            Some(ProgrammaticEvent {
                event: ev,
                external: true,
            }),
        )
        .unwrap();
        let h = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(h, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        assert!(sim
            .graph_kernel_node_get_programmatic_event(h, 0)
            .unwrap()
            .is_none());
        sim.graph_kernel_node_copy_attributes(h, 0, g, 0).unwrap();
        let got = sim
            .graph_kernel_node_get_programmatic_event(h, 0)
            .unwrap()
            .expect("copied");
        assert_eq!(got.event, ev);
        assert!(got.external);
        sim.graph_kernel_node_set_programmatic_event(h, 0, None)
            .unwrap();
        assert!(sim
            .graph_kernel_node_get_programmatic_event(h, 0)
            .unwrap()
            .is_none());
    }

    #[test]
    fn device_launch_rejects_programmatic_event() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 64).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_kernel_node_set_programmatic_event(
            g,
            0,
            Some(ProgrammaticEvent {
                event: EventId(8),
                external: false,
            }),
        )
        .unwrap();
        let err = sim
            .instantiate_graph_with_flags(g, GraphInstantiateFlags::DEVICE_LAUNCH)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("device launch"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn launch_completion_event_unblocks_copy_at_kernel_start() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s0 = StreamId(0);
        let s1 = StreamId(1);
        let a = sim.alloc(d, 4096, s0).unwrap();
        let b = sim.alloc(d, 64 << 20, s1).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s0));
        sim.synchronize().unwrap();
        let ev = EventId(9);
        enq(sim.kernel_launch_completion(
            d,
            KernelKind::other(1 << 30, 4096),
            &[a],
            &[a],
            s0,
            LaunchCompletionEvent {
                event: ev,
                external: false,
            },
        ));
        enq(sim.wait_event(d, ev, s1));
        enq(sim.memcpy_pinned_to_device(d, b, 64 << 20, s1));
        sim.synchronize().unwrap();
        let k0 = sim
            .operations()
            .find(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .expect("kernel");
        let copy = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Memcpy(_)))
            .last()
            .expect("copy");
        let start_k = k0.start_ns.expect("k start");
        let done_k = k0.done_ns.expect("k done");
        let start_c = copy.start_ns.expect("copy start");
        assert!(
            start_c >= start_k,
            "copy must wait for launch completion; copy={start_c} kstart={start_k}"
        );
        assert!(
            start_c < done_k,
            "copy must overlap leftover kernel; copy={start_c} kdone={done_k}"
        );
    }

    #[test]
    fn launch_completion_query_fires_at_kernel_start() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        let ev = EventId(10);
        enq(sim.kernel_launch_completion(
            d,
            KernelKind::other(1 << 30, 4096),
            &[a],
            &[a],
            s,
            LaunchCompletionEvent {
                event: ev,
                external: false,
            },
        ));
        sim.synchronize_event(ev).unwrap();
        assert!(sim.query_event(ev).unwrap());
        let k0 = sim
            .operations()
            .find(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .expect("kernel");
        assert!(!k0.done, "launch completion must precede kernel done");
        assert!(k0.start_ns.is_some());
        sim.synchronize().unwrap();
    }

    #[test]
    fn graph_launch_completion_copies_and_device_launch_refuses() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 64).unwrap();
        let ev = EventId(11);
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_kernel_node_set_launch_completion(
            g,
            0,
            Some(LaunchCompletionEvent {
                event: ev,
                external: true,
            }),
        )
        .unwrap();
        let h = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(h, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_kernel_node_copy_attributes(h, 0, g, 0).unwrap();
        let got = sim
            .graph_kernel_node_get_launch_completion(h, 0)
            .unwrap()
            .expect("copied");
        assert_eq!(got.event, ev);
        assert!(got.external);
        let err = sim
            .instantiate_graph_with_flags(g, GraphInstantiateFlags::DEVICE_LAUNCH)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("device launch"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    fn persist_mem_profile() -> HardwareProfile {
        let mut p = h100();
        let g = p.gpus.first_mut().expect("gpu0");
        g.launch_overhead_ns = 1;
        g.graph_launch_ns = 1;
        g.l2_persist_hit_permille = 1000;
        g.hbm_bps = 1_000_000_000;
        p
    }

    #[test]
    fn persist_limit_zero_window_is_noop() {
        let mut sim = Sim::new(persist_mem_profile());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let a = sim.alloc(d, bytes, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, bytes, s));
        sim.synchronize().unwrap();
        let kind = KernelKind::other(1, bytes);
        let w = AccessPolicyWindow::persisting(KernelBuf::whole(a));
        enq(sim.kernel_access_policy(d, kind.clone(), &[a], &[a], s, w));
        enq(sim.kernel_access_policy(d, kind, &[a], &[a], s, w));
        sim.synchronize().unwrap();
        let ks: Vec<_> = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .collect();
        let d0 = ks[0].duration_ns().expect("k0");
        let d1 = ks[1].duration_ns().expect("k1");
        assert_eq!(d0, d1, "persist limit 0 must not discount; d0={d0} d1={d1}");
        assert_eq!(sim.persisting_l2_cache_size(d).unwrap(), 0);
    }

    #[test]
    fn second_kernel_hits_persisting_l2() {
        let mut sim = Sim::new(persist_mem_profile());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.enable_persisting_l2().unwrap();
        let bytes = 8u64 << 20;
        let a = sim.alloc(d, bytes, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, bytes, s));
        sim.synchronize().unwrap();
        let kind = KernelKind::other(1, bytes);
        let w = AccessPolicyWindow::persisting(KernelBuf::whole(a));
        enq(sim.kernel_access_policy(d, kind.clone(), &[a], &[a], s, w));
        enq(sim.kernel_access_policy(d, kind, &[a], &[a], s, w));
        sim.synchronize().unwrap();
        let ks: Vec<_> = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .collect();
        let d0 = ks[0].duration_ns().expect("k0");
        let d1 = ks[1].duration_ns().expect("k1");
        assert!(
            d1 < d0 / 2,
            "warm persist must cut HBM; cold={d0} warm={d1}"
        );
    }

    #[test]
    fn reset_persisting_l2_cache_colds_next_kernel() {
        let mut sim = Sim::new(persist_mem_profile());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.enable_persisting_l2().unwrap();
        let bytes = 8u64 << 20;
        let a = sim.alloc(d, bytes, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, bytes, s));
        sim.synchronize().unwrap();
        let kind = KernelKind::other(1, bytes);
        let w = AccessPolicyWindow::persisting(KernelBuf::whole(a));
        enq(sim.kernel_access_policy(d, kind.clone(), &[a], &[a], s, w));
        sim.synchronize().unwrap();
        sim.reset_persisting_l2_cache(d).unwrap();
        enq(sim.kernel_access_policy(d, kind, &[a], &[a], s, w));
        sim.synchronize().unwrap();
        let ks: Vec<_> = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .collect();
        let d0 = ks[0].duration_ns().expect("k0");
        let d1 = ks[1].duration_ns().expect("k1");
        assert_eq!(d0, d1, "reset must refill; d0={d0} d1={d1}");
    }

    #[test]
    fn set_persisting_l2_rejects_over_l2_and_miss_persisting() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let cap = sim.profile().gpu(d).unwrap().l2_bytes;
        let err = sim
            .set_persisting_l2_cache_size(d, cap.saturating_add(1))
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("persisting L2"), "{why}"),
            other => panic!("{other:?}"),
        }
        let s = StreamId(0);
        let a = sim.malloc(d, 64).unwrap();
        let bad = AccessPolicyWindow {
            buf: KernelBuf::whole(a),
            hit_ratio_permille: 1000,
            hit: AccessProperty::Persisting,
            miss: AccessProperty::Persisting,
        };
        let err = sim
            .kernel_access_policy(d, KernelKind::other(8, 8), &[a], &[a], s, bad)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("miss persisting"), "{why}"),
            other => panic!("{other:?}"),
        }
        let ratio = AccessPolicyWindow {
            buf: KernelBuf::whole(a),
            hit_ratio_permille: 1001,
            hit: AccessProperty::Persisting,
            miss: AccessProperty::Streaming,
        };
        let err = sim
            .kernel_access_policy(d, KernelKind::other(8, 8), &[a], &[a], s, ratio)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("hit ratio"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn streaming_hit_does_not_fill_persisting_l2() {
        let mut sim = Sim::new(persist_mem_profile());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.enable_persisting_l2().unwrap();
        let bytes = 8u64 << 20;
        let a = sim.alloc(d, bytes, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, bytes, s));
        sim.synchronize().unwrap();
        let kind = KernelKind::other(1, bytes);
        let w = AccessPolicyWindow {
            buf: KernelBuf::whole(a),
            hit_ratio_permille: 1000,
            hit: AccessProperty::Streaming,
            miss: AccessProperty::Streaming,
        };
        enq(sim.kernel_access_policy(d, kind.clone(), &[a], &[a], s, w));
        enq(sim.kernel_access_policy(d, kind, &[a], &[a], s, w));
        sim.synchronize().unwrap();
        let ks: Vec<_> = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .collect();
        assert_eq!(
            ks[0].duration_ns().expect("k0"),
            ks[1].duration_ns().expect("k1")
        );
    }

    #[test]
    fn graph_access_policy_copies_and_device_launch_allows() {
        let mut sim = Sim::new(persist_mem_profile());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.enable_persisting_l2().unwrap();
        let a = sim.malloc(d, 8 << 20).unwrap();
        let w = AccessPolicyWindow::persisting(KernelBuf::whole(a));
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(1, 8 << 20), &[a], &[a])
            .unwrap();
        sim.graph_kernel_node_set_access_policy(g, 0, Some(w))
            .unwrap();
        let h = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(h, KernelKind::other(1, 8 << 20), &[a], &[a])
            .unwrap();
        assert!(sim
            .graph_kernel_node_get_access_policy(h, 0)
            .unwrap()
            .is_none());
        sim.graph_kernel_node_copy_attributes(h, 0, g, 0).unwrap();
        let got = sim
            .graph_kernel_node_get_access_policy(h, 0)
            .unwrap()
            .expect("copied");
        assert_eq!(got, w);
        let exec = sim
            .instantiate_graph_with_flags(g, GraphInstantiateFlags::DEVICE_LAUNCH)
            .expect("device-launch allows access policy");
        sim.upload_graph(exec).unwrap();
        assert!(sim
            .graph_exec_kernel_node_get_access_policy(exec, 0)
            .unwrap()
            .is_some());
    }

    #[test]
    fn graph_replay_second_launch_hits_persisting_l2() {
        let mut sim = Sim::new(persist_mem_profile());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.enable_persisting_l2().unwrap();
        let bytes = 8u64 << 20;
        let a = sim.alloc(d, bytes, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, bytes, s));
        sim.synchronize().unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(1, bytes), &[a], &[a])
            .unwrap();
        sim.graph_kernel_node_set_access_policy(
            g,
            0,
            Some(AccessPolicyWindow::persisting(KernelBuf::whole(a))),
        )
        .unwrap();
        let exec = sim.instantiate_graph(g).unwrap();
        let _n = sim.launch_graph(exec, s).unwrap();
        sim.synchronize().unwrap();
        let _n = sim.launch_graph(exec, s).unwrap();
        sim.synchronize().unwrap();
        let ks: Vec<_> = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .collect();
        let d0 = ks[0].duration_ns().expect("k0");
        let d1 = ks[1].duration_ns().expect("k1");
        assert!(
            d1 < d0 / 2,
            "graph relaunch must hit persist; cold={d0} warm={d1}"
        );
    }

    fn fence_profile(tax: u16) -> HardwareProfile {
        let mut p = h100();
        let g = p.gpus.first_mut().expect("gpu0");
        g.compute_slots = 2;
        g.launch_overhead_ns = 1;
        g.same_domain_fence_permille = tax;
        g.mem_sync_domain_count = 4;
        p
    }

    fn short_long_overlap(
        tax: u16,
        long_domain: MemSyncDomain,
        map: Option<MemSyncDomainMap>,
    ) -> (u64, u64) {
        let mut sim = Sim::new(fence_profile(tax));
        let d = DeviceId(0);
        let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
        sim.synchronize().unwrap();
        if let Some(map) = map {
            sim.set_stream_mem_sync_domain_map(d, StreamId(1), map)
                .unwrap();
            sim.set_stream_mem_sync_domain_map(d, StreamId(2), map)
                .unwrap();
        }
        let short = KernelKind::other(1, 4096);
        let long = KernelKind::other(1 << 40, 4096);
        enq(sim.kernel(d, short, &[a], &[a], StreamId(1)));
        enq(sim.kernel_with(
            d,
            long,
            &[a],
            &[a],
            StreamId(2),
            KernelAttrs {
                mem_sync_domain: Some(long_domain),
                ..KernelAttrs::default()
            },
        ));
        sim.synchronize().unwrap();
        let mut ks: Vec<_> = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .collect();
        ks.sort_by_key(|o| o.stream.0);
        (
            ks[0].duration_ns().expect("short"),
            ks[1].duration_ns().expect("long"),
        )
    }

    #[test]
    fn same_domain_fence_waits_on_peer_leftover() {
        let (short0, long0) = short_long_overlap(0, MemSyncDomain::Default, None);
        let (short_tax, long_tax) = short_long_overlap(1000, MemSyncDomain::Default, None);
        assert!(
            short0 * 4 < long0,
            "short kernel must finish first without tax; short={short0} long={long0}"
        );
        assert!(
            short_tax + 1 >= long_tax,
            "same-domain tax 1000 must wait leftover; short={short_tax} long={long_tax}"
        );
        assert_eq!(long0, long_tax, "peer duration is unchanged");
    }

    #[test]
    fn remote_domain_skips_same_domain_fence() {
        let (short_iso, long_iso) = short_long_overlap(1000, MemSyncDomain::Remote, None);
        let (short_same, _) = short_long_overlap(1000, MemSyncDomain::Default, None);
        assert!(
            short_iso * 4 < long_iso,
            "remote domain must isolate; short={short_iso} long={long_iso}"
        );
        assert!(
            short_iso < short_same / 2,
            "isolated short must beat same-domain tax; iso={short_iso} same={short_same}"
        );
    }

    #[test]
    fn mem_sync_map_can_collapse_remote_onto_default() {
        let collide = MemSyncDomainMap {
            default: 0,
            remote: 0,
        };
        let (short_col, long_col) = short_long_overlap(1000, MemSyncDomain::Remote, Some(collide));
        assert!(
            short_col + 1 >= long_col,
            "collapsed map must tax; short={short_col} long={long_col}"
        );
    }

    #[test]
    fn mem_sync_map_rejects_id_past_count() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        assert_eq!(sim.mem_sync_domain_count(d).unwrap(), 4);
        let err = sim
            .set_stream_mem_sync_domain_map(
                d,
                StreamId(0),
                MemSyncDomainMap {
                    default: 0,
                    remote: 4,
                },
            )
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("mem sync domain"), "{why}"),
            other => panic!("{other:?}"),
        }
        let mut one = h100();
        one.gpus.first_mut().expect("gpu0").mem_sync_domain_count = 1;
        let mut sim = Sim::new(one);
        let err = sim
            .set_stream_mem_sync_domain_map(d, StreamId(0), MemSyncDomainMap::identity(2))
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("mem sync domain"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.set_stream_mem_sync_domain_map(d, StreamId(0), MemSyncDomainMap::identity(1))
            .unwrap();
        assert_eq!(
            sim.stream_mem_sync_domain_map(d, StreamId(0))
                .unwrap()
                .remote,
            0
        );
    }

    #[test]
    fn graph_mem_sync_copies_and_device_launch_allows() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 8).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        assert_eq!(
            sim.graph_kernel_node_get_mem_sync_domain(g, 0).unwrap(),
            MemSyncDomain::Default
        );
        sim.graph_kernel_node_set_mem_sync_domain(g, 0, MemSyncDomain::Remote)
            .unwrap();
        let map = MemSyncDomainMap {
            default: 2,
            remote: 3,
        };
        sim.graph_kernel_node_set_mem_sync_domain_map(g, 0, map)
            .unwrap();
        let h = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(h, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_kernel_node_copy_attributes(h, 0, g, 0).unwrap();
        assert_eq!(
            sim.graph_kernel_node_get_mem_sync_domain(h, 0).unwrap(),
            MemSyncDomain::Remote
        );
        assert_eq!(
            sim.graph_kernel_node_get_mem_sync_domain_map(h, 0).unwrap(),
            map
        );
        let exec = sim
            .instantiate_graph_with_flags(g, GraphInstantiateFlags::DEVICE_LAUNCH)
            .expect("device-launch allows mem sync");
        assert_eq!(
            sim.graph_exec_kernel_node_get_mem_sync_domain(exec, 0)
                .unwrap(),
            MemSyncDomain::Remote
        );
    }

    #[test]
    fn graph_replay_ignores_launch_stream_mem_sync_map() {
        let mut sim = Sim::new(fence_profile(1000));
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(1, 4096), &[a], &[a])
            .unwrap();
        sim.graph_kernel_node_set_mem_sync_domain(g, 0, MemSyncDomain::Remote)
            .unwrap();
        let exec = sim.instantiate_graph(g).unwrap();
        sim.set_stream_mem_sync_domain_map(
            d,
            s,
            MemSyncDomainMap {
                default: 0,
                remote: 0,
            },
        )
        .unwrap();
        let long = KernelKind::other(1 << 40, 4096);
        enq(sim.kernel(d, long, &[a], &[a], StreamId(1)));
        let _n = sim.launch_graph(exec, s).unwrap();
        sim.synchronize().unwrap();
        let mut ks: Vec<_> = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .collect();
        ks.sort_by_key(|o| o.duration_ns().unwrap_or(u64::MAX));
        let gd = ks[0].duration_ns().expect("graph short");
        let ld = ks[1].duration_ns().expect("live long");
        assert!(
            gd * 4 < ld,
            "graph node Remote must isolate despite launch-stream map 0; graph={gd} live={ld}"
        );
    }

    #[test]
    fn allreduce_remote_isolates_from_default_kernel() {
        let mut p = HardwareProfile::example_8xh100_nvlink();
        for g in &mut p.gpus {
            g.compute_slots = 2;
            g.launch_overhead_ns = 1;
            g.same_domain_fence_permille = 1000;
        }
        let mut sim = Sim::new(p);
        let s0 = StreamId(0);
        let s1 = StreamId(1);
        let bytes = 8u64 << 20;
        let a = sim.alloc(DeviceId(0), bytes, s0).unwrap();
        enq(sim.memcpy_pinned_to_device(DeviceId(0), a, bytes, s0));
        enq(sim.memcpy_device_to_device(DeviceId(0), DeviceId(1), a, bytes, s0));
        sim.synchronize().unwrap();
        let short = KernelKind::other(1, 4096);
        enq(sim.kernel(DeviceId(0), short, &[a], &[a], s1));
        enq(sim.allreduce(&[(DeviceId(0), a), (DeviceId(1), a)], bytes, s0));
        sim.synchronize().unwrap();
        let k = sim
            .operations()
            .find(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .expect("kernel");
        let ar = sim
            .operations()
            .find(|o| matches!(o.kind, GpuOp::AllReduce { .. }))
            .expect("allreduce");
        let kd = k.duration_ns().expect("k");
        let ad = ar.duration_ns().expect("ar");
        assert!(
            kd * 4 < ad,
            "NCCL-style Remote allreduce must not tax default GEMM; k={kd} ar={ad}"
        );
    }

    #[test]
    fn cluster_kernel_occupies_min_blocks_compute_slots() {
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |cluster: Option<ClusterDim>| {
            let mut sim = Sim::new(h100().with_compute_slots(2));
            let d = DeviceId(0);
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(1)));
            enq(sim.kernel_with(
                d,
                kind.clone(),
                &[a],
                &[a],
                StreamId(2),
                KernelAttrs {
                    cluster,
                    ..KernelAttrs::default()
                },
            ));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let overlap = run(None);
        let clustered = run(Some(ClusterDim::x(2)));
        assert!(
            overlap < clustered,
            "cluster of 2 must occupy both Hyper-Q slots; overlap={overlap} cluster={clustered}"
        );
    }

    #[test]
    fn cluster_rejects_zero_dim_and_over_max() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 8).unwrap();
        let err = sim
            .kernel_with(
                d,
                KernelKind::other(8, 8),
                &[a],
                &[a],
                s,
                KernelAttrs {
                    cluster: Some(ClusterDim { x: 2, y: 0, z: 1 }),
                    ..KernelAttrs::default()
                },
            )
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("cluster dimension"), "{why}"),
            other => panic!("{other:?}"),
        }
        let err = sim
            .kernel_with(
                d,
                KernelKind::other(8, 8),
                &[a],
                &[a],
                s,
                KernelAttrs {
                    cluster: Some(ClusterDim::x(9)),
                    ..KernelAttrs::default()
                },
            )
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("cluster size"), "{why}"),
            other => panic!("{other:?}"),
        }
        let mut one = h100();
        one.gpus.first_mut().expect("gpu0").max_blocks_per_cluster = 1;
        let mut sim = Sim::new(one);
        let a = sim.malloc(d, 8).unwrap();
        enq(sim.kernel_with(
            d,
            KernelKind::other(8, 8),
            &[a],
            &[a],
            s,
            KernelAttrs {
                cluster: Some(ClusterDim::x(1)),
                ..KernelAttrs::default()
            },
        ));
        let err = sim
            .kernel_with(
                d,
                KernelKind::other(8, 8),
                &[a],
                &[a],
                s,
                KernelAttrs {
                    cluster: Some(ClusterDim::x(2)),
                    ..KernelAttrs::default()
                },
            )
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("cluster size"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn cluster_dim_must_be_set_and_required_size() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 8).unwrap();
        assert!(!sim.cluster_dim_must_be_set(d).unwrap());
        sim.set_cluster_dim_must_be_set(d, true).unwrap();
        match sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("cluster dim must be set"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        enq(sim.kernel_with(
            d,
            KernelKind::other(8, 8),
            &[a],
            &[a],
            s,
            KernelAttrs {
                cluster: Some(ClusterDim::x(2)),
                ..KernelAttrs::default()
            },
        ));
        sim.set_required_cluster_width(d, 2).unwrap();
        enq(sim.kernel_with(
            d,
            KernelKind::other(8, 8),
            &[a],
            &[a],
            s,
            KernelAttrs {
                cluster: Some(ClusterDim::x(2)),
                ..KernelAttrs::default()
            },
        ));
        match sim.kernel_with(
            d,
            KernelKind::other(8, 8),
            &[a],
            &[a],
            s,
            KernelAttrs {
                cluster: Some(ClusterDim::x(4)),
                ..KernelAttrs::default()
            },
        ) {
            Err(SimError::Invalid { why }) => assert!(why.contains("required cluster"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.set_cluster_dim_must_be_set(d, false).unwrap();
        sim.set_required_cluster_width(d, 0).unwrap();
        sim.set_required_cluster_height(d, 2).unwrap();
        match sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s) {
            Err(SimError::Invalid { why }) => assert!(why.contains("required cluster"), "{why}"),
            other => panic!("{other:?}"),
        }
        enq(sim.kernel_with(
            d,
            KernelKind::other(8, 8),
            &[a],
            &[a],
            s,
            KernelAttrs {
                cluster: Some(ClusterDim { x: 1, y: 2, z: 1 }),
                ..KernelAttrs::default()
            },
        ));
        match sim.set_required_cluster_width(d, 9) {
            Err(SimError::Invalid { why }) => assert!(why.contains("cluster size"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(
            sim.func_get_attribute(d, FuncAttr::RequiredClusterHeight)
                .unwrap(),
            2
        );
        let attrs = sim.func_get_attributes(d).unwrap();
        assert!(!attrs.cluster_dim_must_be_set);
        assert_eq!(attrs.required_cluster_height, 2);
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        sim.func_set_attribute(d, FuncAttr::ClusterDimMustBeSet, 1)
            .unwrap();
        assert!(sim.cluster_dim_must_be_set(d).unwrap());
        let _g = sim.end_capture().unwrap();
        match sim.func_set_attribute(d, FuncAttr::ClusterDimMustBeSet, 2) {
            Err(SimError::Invalid { why }) => assert!(why.contains("func attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.func_set_attribute(d, FuncAttr::RequiredClusterDepth, -1) {
            Err(SimError::Invalid { why }) => assert!(why.contains("func attr"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn non_portable_cluster_size_requires_func_attribute() {
        let p = HardwareProfile::parse("gpus=1\nmax_blocks_per_cluster=16\n").unwrap();
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 8).unwrap();
        let err = sim
            .kernel_with(
                d,
                KernelKind::other(8, 8),
                &[a],
                &[a],
                s,
                KernelAttrs {
                    cluster: Some(ClusterDim::x(16)),
                    ..KernelAttrs::default()
                },
            )
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("non-portable cluster"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.set_non_portable_cluster_size_allowed(d, true).unwrap();
        assert!(sim.non_portable_cluster_size_allowed(d));
        enq(sim.kernel_with(
            d,
            KernelKind::other(8, 8),
            &[a],
            &[a],
            s,
            KernelAttrs {
                cluster: Some(ClusterDim::x(16)),
                ..KernelAttrs::default()
            },
        ));
        let err = sim
            .kernel_with(
                d,
                KernelKind::other(8, 8),
                &[a],
                &[a],
                s,
                KernelAttrs {
                    cluster: Some(ClusterDim::x(17)),
                    ..KernelAttrs::default()
                },
            )
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("cluster size"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn cluster_spread_occupies_all_compute_slots() {
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |policy: ClusterSchedulingPolicy| {
            let mut sim = Sim::new(h100().with_compute_slots(4));
            let d = DeviceId(0);
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(1)));
            enq(sim.kernel_with(
                d,
                kind.clone(),
                &[a],
                &[a],
                StreamId(2),
                KernelAttrs {
                    cluster: Some(ClusterDim::x(2)),
                    cluster_policy: policy,
                    ..KernelAttrs::default()
                },
            ));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let packed = run(ClusterSchedulingPolicy::Default);
        let spread = run(ClusterSchedulingPolicy::Spread);
        assert!(
            packed < spread,
            "spread cluster of 2 must occupy all 4 Hyper-Q slots; packed={packed} spread={spread}"
        );
        let balanced = run(ClusterSchedulingPolicy::LoadBalancing);
        assert_eq!(packed, balanced, "load-balancing matches default occupancy");
    }

    #[test]
    fn func_cluster_policy_inherits_into_default_launch() {
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |func: ClusterSchedulingPolicy, launch: ClusterSchedulingPolicy| {
            let mut sim = Sim::new(h100().with_compute_slots(4));
            let d = DeviceId(0);
            sim.set_func_cluster_policy(d, func).unwrap();
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(1)));
            enq(sim.kernel_with(
                d,
                kind.clone(),
                &[a],
                &[a],
                StreamId(2),
                KernelAttrs {
                    cluster: Some(ClusterDim::x(2)),
                    cluster_policy: launch,
                    ..KernelAttrs::default()
                },
            ));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let packed = run(
            ClusterSchedulingPolicy::Default,
            ClusterSchedulingPolicy::Default,
        );
        let func_spread = run(
            ClusterSchedulingPolicy::Spread,
            ClusterSchedulingPolicy::Default,
        );
        let launch_spread = run(
            ClusterSchedulingPolicy::Default,
            ClusterSchedulingPolicy::Spread,
        );
        assert!(
            packed < func_spread,
            "func Spread must occupy all slots; packed={packed} func={func_spread}"
        );
        assert_eq!(
            func_spread, launch_spread,
            "func inherit matches launch Spread"
        );
        let launch_balanced = run(
            ClusterSchedulingPolicy::Spread,
            ClusterSchedulingPolicy::LoadBalancing,
        );
        assert_eq!(
            launch_balanced, packed,
            "launch LoadBalancing overrides func Spread"
        );
        let func_balanced = run(
            ClusterSchedulingPolicy::LoadBalancing,
            ClusterSchedulingPolicy::Default,
        );
        assert_eq!(
            func_balanced, packed,
            "func LoadBalancing matches default occupancy"
        );
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        assert_eq!(
            sim.get_func_cluster_policy(d).unwrap(),
            ClusterSchedulingPolicy::Default
        );
        assert_eq!(
            sim.func_get_attribute(d, FuncAttr::ClusterSchedulingPolicyPreference)
                .unwrap(),
            0
        );
        sim.func_set_attribute(d, FuncAttr::ClusterSchedulingPolicyPreference, 1)
            .unwrap();
        assert_eq!(
            sim.get_func_cluster_policy(d).unwrap(),
            ClusterSchedulingPolicy::Spread
        );
        assert_eq!(
            sim.func_get_attributes(d)
                .unwrap()
                .cluster_scheduling_policy_preference,
            ClusterSchedulingPolicy::Spread
        );
        match sim.func_set_attribute(d, FuncAttr::ClusterSchedulingPolicyPreference, 3) {
            Err(SimError::Invalid { why }) => assert!(why.contains("func attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, StreamId(0)).unwrap();
        sim.set_func_cluster_policy(d, ClusterSchedulingPolicy::LoadBalancing)
            .unwrap();
        assert_eq!(
            sim.func_get_attribute(d, FuncAttr::ClusterSchedulingPolicyPreference)
                .unwrap(),
            2
        );
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn preferred_cluster_occupies_preferred_when_it_fits() {
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |preferred: Option<ClusterDim>| {
            let mut sim = Sim::new(h100().with_compute_slots(4));
            let d = DeviceId(0);
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(1)));
            enq(sim.kernel_with(
                d,
                kind.clone(),
                &[a],
                &[a],
                StreamId(2),
                KernelAttrs {
                    cluster: Some(ClusterDim::x(2)),
                    preferred_cluster: preferred,
                    ..KernelAttrs::default()
                },
            ));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let required = run(None);
        let preferred = run(Some(ClusterDim::x(4)));
        assert!(
            required < preferred,
            "preferred cluster of 4 must occupy all 4 slots; required={required} preferred={preferred}"
        );
        let mut sim = Sim::new(h100().with_compute_slots(4));
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 8).unwrap();
        let err = sim
            .kernel_with(
                d,
                KernelKind::other(8, 8),
                &[a],
                &[a],
                s,
                KernelAttrs {
                    preferred_cluster: Some(ClusterDim::x(2)),
                    ..KernelAttrs::default()
                },
            )
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("preferred cluster"), "{why}"),
            other => panic!("{other:?}"),
        }
        let err = sim
            .kernel_with(
                d,
                KernelKind::other(8, 8),
                &[a],
                &[a],
                s,
                KernelAttrs {
                    cluster: Some(ClusterDim::x(2)),
                    preferred_cluster: Some(ClusterDim { x: 3, y: 1, z: 1 }),
                    ..KernelAttrs::default()
                },
            )
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("preferred cluster"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_cluster_policy_and_preferred_copy_and_device_launch() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 8).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let dim = ClusterDim::x(2);
        sim.graph_kernel_node_set_cluster(g, 0, Some(dim)).unwrap();
        sim.graph_kernel_node_set_cluster_policy(g, 0, ClusterSchedulingPolicy::Spread)
            .unwrap();
        sim.graph_kernel_node_set_preferred_cluster(g, 0, Some(ClusterDim::x(4)))
            .unwrap();
        let h = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(h, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_kernel_node_copy_attributes(h, 0, g, 0).unwrap();
        assert_eq!(sim.graph_kernel_node_get_cluster(h, 0).unwrap(), Some(dim));
        assert_eq!(
            sim.graph_kernel_node_get_cluster_policy(h, 0).unwrap(),
            ClusterSchedulingPolicy::Spread
        );
        assert_eq!(
            sim.graph_kernel_node_get_preferred_cluster(h, 0).unwrap(),
            Some(ClusterDim::x(4))
        );
        let exec = sim
            .instantiate_graph_with_flags(g, GraphInstantiateFlags::DEVICE_LAUNCH)
            .expect("device-launch allows cluster policy");
        assert_eq!(
            sim.graph_exec_kernel_node_get_cluster_policy(exec, 0)
                .unwrap(),
            ClusterSchedulingPolicy::Spread
        );
        assert_eq!(
            sim.graph_exec_kernel_node_get_preferred_cluster(exec, 0)
                .unwrap(),
            Some(ClusterDim::x(4))
        );
    }

    #[test]
    fn max_shared_carveout_occupies_all_compute_slots() {
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |carveout: SharedMemCarveout| {
            let mut sim = Sim::new(h100().with_compute_slots(2));
            let d = DeviceId(0);
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(1)));
            enq(sim.kernel_with(
                d,
                kind.clone(),
                &[a],
                &[a],
                StreamId(2),
                KernelAttrs {
                    carveout,
                    ..KernelAttrs::default()
                },
            ));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let overlap = run(SharedMemCarveout::Default);
        let serial = run(SharedMemCarveout::MaxShared);
        assert!(
            overlap < serial,
            "MaxShared carveout must occupy all Hyper-Q slots; overlap={overlap} serial={serial}"
        );
        let l1 = run(SharedMemCarveout::MaxL1);
        assert_eq!(overlap, l1, "MaxL1 matches Default occupancy");
    }

    #[test]
    fn func_carveout_inherits_into_default_launch() {
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |func: SharedMemCarveout, launch: SharedMemCarveout| {
            let mut sim = Sim::new(h100().with_compute_slots(2));
            let d = DeviceId(0);
            sim.set_func_carveout(d, func).unwrap();
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel_with(
                d,
                kind.clone(),
                &[a],
                &[a],
                StreamId(1),
                KernelAttrs {
                    carveout: SharedMemCarveout::MaxL1,
                    ..KernelAttrs::default()
                },
            ));
            enq(sim.kernel_with(
                d,
                kind.clone(),
                &[a],
                &[a],
                StreamId(2),
                KernelAttrs {
                    carveout: launch,
                    ..KernelAttrs::default()
                },
            ));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let overlap = run(SharedMemCarveout::Default, SharedMemCarveout::Default);
        let func_serial = run(SharedMemCarveout::MaxShared, SharedMemCarveout::Default);
        let launch_serial = run(SharedMemCarveout::Default, SharedMemCarveout::MaxShared);
        assert!(
            overlap < func_serial,
            "func MaxShared must occupy all slots; overlap={overlap} func={func_serial}"
        );
        assert_eq!(
            func_serial, launch_serial,
            "func inherit matches launch MaxShared"
        );
        let launch_l1 = run(SharedMemCarveout::MaxShared, SharedMemCarveout::MaxL1);
        assert_eq!(launch_l1, overlap, "launch MaxL1 overrides func MaxShared");
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        assert_eq!(
            sim.get_func_carveout(d).unwrap(),
            SharedMemCarveout::Default
        );
        assert_eq!(
            sim.func_get_attribute(d, FuncAttr::PreferredSharedMemoryCarveout)
                .unwrap(),
            -1
        );
        sim.func_set_attribute(d, FuncAttr::PreferredSharedMemoryCarveout, 100)
            .unwrap();
        assert_eq!(
            sim.get_func_carveout(d).unwrap(),
            SharedMemCarveout::MaxShared
        );
        assert_eq!(
            sim.func_get_attributes(d).unwrap().preferred_shmem_carveout,
            SharedMemCarveout::MaxShared
        );
        match sim.func_set_attribute(d, FuncAttr::PreferredSharedMemoryCarveout, 50) {
            Err(SimError::Invalid { why }) => assert!(why.contains("func attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, StreamId(0)).unwrap();
        sim.set_func_carveout(d, SharedMemCarveout::MaxL1).unwrap();
        assert_eq!(
            sim.func_get_attribute(d, FuncAttr::PreferredSharedMemoryCarveout)
                .unwrap(),
            0
        );
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn shared_mem_mode_scales_kernel_duration() {
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |mode: SharedMemoryMode, four: u16, eight: u16| {
            let mut p = h100();
            for g in &mut p.gpus {
                g.shared_mem_four_byte_permille = four;
                g.shared_mem_eight_byte_permille = eight;
            }
            let mut sim = Sim::new(p);
            let d = DeviceId(0);
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel_with(
                d,
                kind.clone(),
                &[a],
                &[a],
                StreamId(1),
                KernelAttrs {
                    shared_mem: mode,
                    ..KernelAttrs::default()
                },
            ));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let def = run(SharedMemoryMode::Default, 1000, 1000);
        let def_ignore = run(SharedMemoryMode::Default, 500, 500);
        assert_eq!(def, def_ignore, "Default ignores shared_mem_*_permille");
        let four_id = run(SharedMemoryMode::FourByte, 1000, 1000);
        let eight_id = run(SharedMemoryMode::EightByte, 1000, 1000);
        assert_eq!(def, four_id, "FourByte at 1000 is identity");
        assert_eq!(def, eight_id, "EightByte at 1000 is identity");
        let eight_slow = run(SharedMemoryMode::EightByte, 1000, 500);
        let four_slow = run(SharedMemoryMode::FourByte, 500, 1000);
        assert!(
            eight_slow > def,
            "EightByte at 500 must lengthen; def={def} eight={eight_slow}"
        );
        assert!(
            four_slow > def,
            "FourByte at 500 must lengthen; def={def} four={four_slow}"
        );
        assert_eq!(four_slow, eight_slow, "same permille, same scale");
    }

    #[test]
    fn device_shared_mem_config_inherits_into_default() {
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |device: SharedMemoryMode, launch: SharedMemoryMode, four: u16| {
            let mut p = h100();
            for g in &mut p.gpus {
                g.shared_mem_four_byte_permille = four;
            }
            let mut sim = Sim::new(p);
            let d = DeviceId(0);
            sim.set_shared_mem_config(d, device).unwrap();
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel_with(
                d,
                kind.clone(),
                &[a],
                &[a],
                StreamId(1),
                KernelAttrs {
                    shared_mem: launch,
                    ..KernelAttrs::default()
                },
            ));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let unset = run(SharedMemoryMode::Default, SharedMemoryMode::Default, 500);
        let inherit = run(SharedMemoryMode::FourByte, SharedMemoryMode::Default, 500);
        let explicit = run(SharedMemoryMode::Default, SharedMemoryMode::FourByte, 500);
        assert!(
            inherit > unset,
            "device FourByte must scale Default kernels; unset={unset} inherit={inherit}"
        );
        assert_eq!(inherit, explicit, "inherit matches launch FourByte");
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        assert_eq!(
            sim.get_shared_mem_config(d).unwrap(),
            SharedMemoryMode::Default
        );
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(
            sim.get_shared_mem_config(d).unwrap(),
            SharedMemoryMode::Default
        );
        match sim.set_shared_mem_config(d, SharedMemoryMode::FourByte) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn func_shared_mem_config_inherits_before_device() {
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |func: SharedMemoryMode,
                   device: SharedMemoryMode,
                   launch: SharedMemoryMode,
                   four: u16,
                   eight: u16| {
            let mut p = h100();
            for g in &mut p.gpus {
                g.shared_mem_four_byte_permille = four;
                g.shared_mem_eight_byte_permille = eight;
            }
            let mut sim = Sim::new(p);
            let d = DeviceId(0);
            sim.set_shared_mem_config(d, device).unwrap();
            sim.set_func_shared_mem_config(d, func).unwrap();
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel_with(
                d,
                kind.clone(),
                &[a],
                &[a],
                StreamId(1),
                KernelAttrs {
                    shared_mem: launch,
                    ..KernelAttrs::default()
                },
            ));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let unset = run(
            SharedMemoryMode::Default,
            SharedMemoryMode::Default,
            SharedMemoryMode::Default,
            500,
            1000,
        );
        let func_four = run(
            SharedMemoryMode::FourByte,
            SharedMemoryMode::Default,
            SharedMemoryMode::Default,
            500,
            1000,
        );
        let launch_four = run(
            SharedMemoryMode::Default,
            SharedMemoryMode::Default,
            SharedMemoryMode::FourByte,
            500,
            1000,
        );
        assert!(
            func_four > unset,
            "func FourByte must scale Default kernels; unset={unset} func={func_four}"
        );
        assert_eq!(
            func_four, launch_four,
            "func inherit matches launch FourByte"
        );
        let device_four = run(
            SharedMemoryMode::Default,
            SharedMemoryMode::FourByte,
            SharedMemoryMode::Default,
            500,
            1000,
        );
        assert_eq!(device_four, launch_four, "device FourByte still scales");
        let launch_eight = run(
            SharedMemoryMode::FourByte,
            SharedMemoryMode::Default,
            SharedMemoryMode::EightByte,
            500,
            250,
        );
        let eight = run(
            SharedMemoryMode::Default,
            SharedMemoryMode::Default,
            SharedMemoryMode::EightByte,
            500,
            250,
        );
        assert_eq!(launch_eight, eight, "launch EightByte overrides func");
        let func_wins = run(
            SharedMemoryMode::FourByte,
            SharedMemoryMode::EightByte,
            SharedMemoryMode::Default,
            500,
            250,
        );
        assert_eq!(func_wins, func_four, "func config wins over device");
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        assert_eq!(
            sim.get_func_shared_mem_config(d).unwrap(),
            SharedMemoryMode::Default
        );
        sim.set_func_shared_mem_config(d, SharedMemoryMode::FourByte)
            .unwrap();
        assert_eq!(
            sim.get_func_shared_mem_config(d).unwrap(),
            SharedMemoryMode::FourByte
        );
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(
            sim.get_func_shared_mem_config(d).unwrap(),
            SharedMemoryMode::FourByte
        );
        match sim.set_func_shared_mem_config(d, SharedMemoryMode::EightByte) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn kernel_with_capture_records_shared_mem() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel_with(
            d,
            KernelKind::other(8, 8),
            &[a],
            &[a],
            s,
            KernelAttrs {
                shared_mem: SharedMemoryMode::FourByte,
                ..KernelAttrs::default()
            },
        ));
        let g = sim.end_capture().unwrap();
        assert_eq!(
            sim.graph_kernel_node_get_shared_mem(g, 0).unwrap(),
            SharedMemoryMode::FourByte
        );
    }

    #[test]
    fn graph_carveout_copy_and_device_launch() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 8).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        assert_eq!(
            sim.graph_kernel_node_get_carveout(g, 0).unwrap(),
            SharedMemCarveout::Default
        );
        sim.graph_kernel_node_set_carveout(g, 0, SharedMemCarveout::MaxShared)
            .unwrap();
        let h = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(h, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_kernel_node_copy_attributes(h, 0, g, 0).unwrap();
        assert_eq!(
            sim.graph_kernel_node_get_carveout(h, 0).unwrap(),
            SharedMemCarveout::MaxShared
        );
        let exec = sim
            .instantiate_graph_with_flags(g, GraphInstantiateFlags::DEVICE_LAUNCH)
            .expect("device-launch allows carveout");
        assert_eq!(
            sim.graph_exec_kernel_node_get_carveout(exec, 0).unwrap(),
            SharedMemCarveout::MaxShared
        );
        sim.graph_exec_kernel_node_set_carveout(exec, 0, SharedMemCarveout::MaxL1)
            .unwrap();
        assert_eq!(
            sim.graph_exec_kernel_node_get_carveout(exec, 0).unwrap(),
            SharedMemCarveout::MaxL1
        );
    }

    #[test]
    fn graph_device_updatable_copy_and_device_launch() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 8).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        assert!(!sim.graph_kernel_node_get_device_updatable(g, 0).unwrap());
        sim.graph_kernel_node_set_device_updatable(g, 0, true)
            .unwrap();
        let h = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(h, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_kernel_node_copy_attributes(h, 0, g, 0).unwrap();
        assert!(sim.graph_kernel_node_get_device_updatable(h, 0).unwrap());
        let exec = sim
            .instantiate_graph_with_flags(g, GraphInstantiateFlags::DEVICE_LAUNCH)
            .expect("device-launch allows device-updatable");
        assert!(sim
            .graph_exec_kernel_node_get_device_updatable(exec, 0)
            .unwrap());
        sim.graph_exec_kernel_node_set_device_updatable(exec, 0, false)
            .unwrap();
        assert!(!sim
            .graph_exec_kernel_node_get_device_updatable(exec, 0)
            .unwrap());
        let empty = sim.create_graph(d, s).unwrap();
        sim.graph_add_empty(empty).unwrap();
        let err = sim
            .graph_kernel_node_get_device_updatable(empty, 0)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not a kernel node"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_shared_mem_copy_and_device_launch() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 8).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        assert_eq!(
            sim.graph_kernel_node_get_shared_mem(g, 0).unwrap(),
            SharedMemoryMode::Default
        );
        sim.graph_kernel_node_set_shared_mem(g, 0, SharedMemoryMode::EightByte)
            .unwrap();
        let h = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(h, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_kernel_node_copy_attributes(h, 0, g, 0).unwrap();
        assert_eq!(
            sim.graph_kernel_node_get_shared_mem(h, 0).unwrap(),
            SharedMemoryMode::EightByte
        );
        let exec = sim
            .instantiate_graph_with_flags(g, GraphInstantiateFlags::DEVICE_LAUNCH)
            .expect("device-launch allows shared-mem");
        assert_eq!(
            sim.graph_exec_kernel_node_get_shared_mem(exec, 0).unwrap(),
            SharedMemoryMode::EightByte
        );
        sim.graph_exec_kernel_node_set_shared_mem(exec, 0, SharedMemoryMode::FourByte)
            .unwrap();
        assert_eq!(
            sim.graph_exec_kernel_node_get_shared_mem(exec, 0).unwrap(),
            SharedMemoryMode::FourByte
        );
    }

    #[test]
    fn portable_cluster_mode_overrides_func_attribute() {
        let p = HardwareProfile::parse("gpus=1\nmax_blocks_per_cluster=16\n").unwrap();
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 8).unwrap();
        let err = sim
            .kernel_with(
                d,
                KernelKind::other(8, 8),
                &[a],
                &[a],
                s,
                KernelAttrs {
                    cluster: Some(ClusterDim::x(16)),
                    ..KernelAttrs::default()
                },
            )
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("non-portable cluster"), "{why}"),
            other => panic!("{other:?}"),
        }
        enq(sim.kernel_with(
            d,
            KernelKind::other(8, 8),
            &[a],
            &[a],
            s,
            KernelAttrs {
                cluster: Some(ClusterDim::x(16)),
                portable_cluster: PortableClusterMode::AllowNonPortable,
                ..KernelAttrs::default()
            },
        ));
        sim.set_non_portable_cluster_size_allowed(d, true).unwrap();
        let err = sim
            .kernel_with(
                d,
                KernelKind::other(8, 8),
                &[a],
                &[a],
                s,
                KernelAttrs {
                    cluster: Some(ClusterDim::x(16)),
                    portable_cluster: PortableClusterMode::RequirePortable,
                    ..KernelAttrs::default()
                },
            )
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("non-portable cluster"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn default_portable_cluster_uses_func_attr_at_launch() {
        let p = HardwareProfile::parse("gpus=1\nmax_blocks_per_cluster=16\n").unwrap();
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 8).unwrap();
        sim.set_non_portable_cluster_size_allowed(d, true).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_kernel_node_set_cluster(g, 0, Some(ClusterDim::x(16)))
            .unwrap();
        let exec = sim.instantiate_graph(g).unwrap();
        sim.set_non_portable_cluster_size_allowed(d, false).unwrap();
        let err = sim.launch_graph(exec, s).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("non-portable cluster"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.graph_exec_kernel_node_set_portable_cluster(
            exec,
            0,
            PortableClusterMode::AllowNonPortable,
        )
        .unwrap();
        let n = sim.launch_graph(exec, s).unwrap();
        assert!(n >= 1);
    }

    #[test]
    fn graph_portable_cluster_copy_and_device_launch() {
        let p = HardwareProfile::parse("gpus=1\nmax_blocks_per_cluster=16\n").unwrap();
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 8).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        assert_eq!(
            sim.graph_kernel_node_get_portable_cluster(g, 0).unwrap(),
            PortableClusterMode::Default
        );
        sim.graph_kernel_node_set_portable_cluster(g, 0, PortableClusterMode::AllowNonPortable)
            .unwrap();
        sim.graph_kernel_node_set_cluster(g, 0, Some(ClusterDim::x(16)))
            .unwrap();
        let h = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(h, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_kernel_node_copy_attributes(h, 0, g, 0).unwrap();
        assert_eq!(
            sim.graph_kernel_node_get_portable_cluster(h, 0).unwrap(),
            PortableClusterMode::AllowNonPortable
        );
        assert_eq!(
            sim.graph_kernel_node_get_cluster(h, 0).unwrap(),
            Some(ClusterDim::x(16))
        );
        let exec = sim
            .instantiate_graph_with_flags(g, GraphInstantiateFlags::DEVICE_LAUNCH)
            .expect("device-launch allows portable-cluster");
        assert_eq!(
            sim.graph_exec_kernel_node_get_portable_cluster(exec, 0)
                .unwrap(),
            PortableClusterMode::AllowNonPortable
        );
        let err = sim
            .graph_exec_kernel_node_set_portable_cluster(
                exec,
                0,
                PortableClusterMode::RequirePortable,
            )
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("non-portable cluster"), "{why}"),
            other => panic!("{other:?}"),
        }
        let empty = sim.create_graph(d, s).unwrap();
        sim.graph_add_empty(empty).unwrap();
        let err = sim
            .graph_kernel_node_get_portable_cluster(empty, 0)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not a kernel node"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn kernel_with_capture_records_portable_cluster() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel_with(
            d,
            KernelKind::other(8, 8),
            &[a],
            &[a],
            s,
            KernelAttrs {
                portable_cluster: PortableClusterMode::AllowNonPortable,
                ..KernelAttrs::default()
            },
        ));
        let g = sim.end_capture().unwrap();
        assert_eq!(
            sim.graph_kernel_node_get_portable_cluster(g, 0).unwrap(),
            PortableClusterMode::AllowNonPortable
        );
    }

    fn open_shared_profile() -> HardwareProfile {
        HardwareProfile::parse("gpus=1\nmax_shared_mem_per_block_optin=232448\n").unwrap()
    }

    #[test]
    fn portable_shared_mode_overrides_func_attribute() {
        let mut sim = Sim::new(open_shared_profile());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 8).unwrap();
        let over = KernelAttrs {
            dynamic_shared: 65_536,
            ..KernelAttrs::default()
        };
        let err = sim
            .kernel_with(d, KernelKind::other(8, 8), &[a], &[a], s, over)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("non-portable shared"), "{why}"),
            other => panic!("{other:?}"),
        }
        enq(sim.kernel_with(
            d,
            KernelKind::other(8, 8),
            &[a],
            &[a],
            s,
            KernelAttrs {
                dynamic_shared: 65_536,
                portable_shared: PortableSharedMode::AllowNonPortable,
                ..KernelAttrs::default()
            },
        ));
        sim.set_max_dynamic_shared_memory(d, 65_536).unwrap();
        let err = sim
            .kernel_with(
                d,
                KernelKind::other(8, 8),
                &[a],
                &[a],
                s,
                KernelAttrs {
                    dynamic_shared: 65_536,
                    portable_shared: PortableSharedMode::RequirePortable,
                    ..KernelAttrs::default()
                },
            )
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("non-portable shared"), "{why}"),
            other => panic!("{other:?}"),
        }
        let err = sim
            .kernel_with(
                d,
                KernelKind::other(8, 8),
                &[a],
                &[a],
                s,
                KernelAttrs {
                    dynamic_shared: 300_000,
                    portable_shared: PortableSharedMode::AllowNonPortable,
                    ..KernelAttrs::default()
                },
            )
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("dynamic shared"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn default_portable_shared_uses_func_attr_at_launch() {
        let mut sim = Sim::new(open_shared_profile());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 8).unwrap();
        sim.set_max_dynamic_shared_memory(d, 65_536).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_kernel_node_set_dynamic_shared(g, 0, 65_536)
            .unwrap();
        let exec = sim.instantiate_graph(g).unwrap();
        sim.set_max_dynamic_shared_memory(d, 0).unwrap();
        let err = sim.launch_graph(exec, s).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("non-portable shared"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.set_max_dynamic_shared_memory(d, 65_536).unwrap();
        sim.graph_exec_kernel_node_set_portable_shared(
            exec,
            0,
            PortableSharedMode::AllowNonPortable,
        )
        .unwrap();
        sim.set_max_dynamic_shared_memory(d, 0).unwrap();
        let n = sim.launch_graph(exec, s).unwrap();
        assert!(n >= 1);
    }

    #[test]
    fn graph_portable_shared_copy_and_device_launch() {
        let mut sim = Sim::new(open_shared_profile());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 8).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        assert_eq!(
            sim.graph_kernel_node_get_portable_shared(g, 0).unwrap(),
            PortableSharedMode::Default
        );
        assert_eq!(sim.graph_kernel_node_get_dynamic_shared(g, 0).unwrap(), 0);
        sim.graph_kernel_node_set_portable_shared(g, 0, PortableSharedMode::AllowNonPortable)
            .unwrap();
        sim.graph_kernel_node_set_dynamic_shared(g, 0, 65_536)
            .unwrap();
        let h = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(h, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_kernel_node_copy_attributes(h, 0, g, 0).unwrap();
        assert_eq!(
            sim.graph_kernel_node_get_portable_shared(h, 0).unwrap(),
            PortableSharedMode::AllowNonPortable
        );
        assert_eq!(
            sim.graph_kernel_node_get_dynamic_shared(h, 0).unwrap(),
            0,
            "sharedMemBytes is KernelNodeParams, not CopyAttributes"
        );
        sim.graph_kernel_node_set_dynamic_shared(h, 0, 65_536)
            .unwrap();
        let exec = sim
            .instantiate_graph_with_flags(g, GraphInstantiateFlags::DEVICE_LAUNCH)
            .expect("device-launch allows portable-shared");
        assert_eq!(
            sim.graph_exec_kernel_node_get_portable_shared(exec, 0)
                .unwrap(),
            PortableSharedMode::AllowNonPortable
        );
        assert_eq!(
            sim.graph_exec_kernel_node_get_dynamic_shared(exec, 0)
                .unwrap(),
            65_536
        );
        let err = sim
            .graph_exec_kernel_node_set_portable_shared(
                exec,
                0,
                PortableSharedMode::RequirePortable,
            )
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("non-portable shared"), "{why}"),
            other => panic!("{other:?}"),
        }
        let empty = sim.create_graph(d, s).unwrap();
        sim.graph_add_empty(empty).unwrap();
        let err = sim
            .graph_kernel_node_get_portable_shared(empty, 0)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not a kernel node"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn kernel_with_capture_records_portable_shared() {
        let mut sim = Sim::new(open_shared_profile());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel_with(
            d,
            KernelKind::other(8, 8),
            &[a],
            &[a],
            s,
            KernelAttrs {
                dynamic_shared: 65_536,
                portable_shared: PortableSharedMode::AllowNonPortable,
                ..KernelAttrs::default()
            },
        ));
        let g = sim.end_capture().unwrap();
        assert_eq!(
            sim.graph_kernel_node_get_portable_shared(g, 0).unwrap(),
            PortableSharedMode::AllowNonPortable
        );
        assert_eq!(
            sim.graph_kernel_node_get_dynamic_shared(g, 0).unwrap(),
            65_536
        );
    }

    #[test]
    fn nvlink_util_centric_occupies_all_slots_when_nvlink() {
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |profile: HardwareProfile, nvlink: bool| {
            let mut sim = Sim::new(profile.with_compute_slots(2));
            let d = DeviceId(0);
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(1)));
            enq(sim.kernel_with(
                d,
                kind.clone(),
                &[a],
                &[a],
                StreamId(2),
                KernelAttrs {
                    nvlink_util_centric: nvlink,
                    ..KernelAttrs::default()
                },
            ));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let h100_off = run(h100(), false);
        let h100_on = run(h100(), true);
        assert_eq!(
            h100_off, h100_on,
            "without NVLink the hint must not change occupancy; off={h100_off} on={h100_on}"
        );
        let nv_off = run(HardwareProfile::example_8xh100_nvlink(), false);
        let nv_on = run(HardwareProfile::example_8xh100_nvlink(), true);
        assert!(
            nv_off < nv_on,
            "NVLink-util-centric must occupy all Hyper-Q slots; overlap={nv_off} serial={nv_on}"
        );
    }

    #[test]
    fn stream_nvlink_util_centric_inherits_on_kernel() {
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |inherit: bool| {
            let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink().with_compute_slots(2));
            let d = DeviceId(0);
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            if inherit {
                sim.set_stream_nvlink_util_centric(d, StreamId(2), true)
                    .unwrap();
            }
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(1)));
            enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(2)));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let overlap = run(false);
        let serial = run(true);
        assert!(
            overlap < serial,
            "stream NVLink-util-centric must inherit onto kernel(); overlap={overlap} serial={serial}"
        );
    }

    #[test]
    fn graph_nvlink_util_centric_copy_and_device_launch() {
        let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 8).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        assert!(!sim.graph_kernel_node_get_nvlink_util_centric(g, 0).unwrap());
        sim.graph_kernel_node_set_nvlink_util_centric(g, 0, true)
            .unwrap();
        let h = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(h, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_kernel_node_copy_attributes(h, 0, g, 0).unwrap();
        assert!(sim.graph_kernel_node_get_nvlink_util_centric(h, 0).unwrap());
        let exec = sim
            .instantiate_graph_with_flags(g, GraphInstantiateFlags::DEVICE_LAUNCH)
            .expect("device-launch allows nvlink-util");
        assert!(sim
            .graph_exec_kernel_node_get_nvlink_util_centric(exec, 0)
            .unwrap());
        sim.graph_exec_kernel_node_set_nvlink_util_centric(exec, 0, false)
            .unwrap();
        assert!(!sim
            .graph_exec_kernel_node_get_nvlink_util_centric(exec, 0)
            .unwrap());
        let empty = sim.create_graph(d, s).unwrap();
        sim.graph_add_empty(empty).unwrap();
        let err = sim
            .graph_kernel_node_get_nvlink_util_centric(empty, 0)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not a kernel node"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn kernel_with_capture_records_nvlink_util_centric() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel_with(
            d,
            KernelKind::other(8, 8),
            &[a],
            &[a],
            s,
            KernelAttrs {
                nvlink_util_centric: true,
                ..KernelAttrs::default()
            },
        ));
        let g = sim.end_capture().unwrap();
        assert!(sim.graph_kernel_node_get_nvlink_util_centric(g, 0).unwrap());
    }

    #[test]
    fn stream_copy_attributes_copies_nvlink_util_centric() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        sim.set_stream_nvlink_util_centric(d, StreamId(1), true)
            .unwrap();
        sim.stream_copy_attributes(d, StreamId(2), d, StreamId(1))
            .unwrap();
        assert!(sim.stream_nvlink_util_centric(d, StreamId(2)));
        sim.set_stream_nvlink_util_centric(d, StreamId(1), false)
            .unwrap();
        sim.stream_copy_attributes(d, StreamId(2), d, StreamId(1))
            .unwrap();
        assert!(!sim.stream_nvlink_util_centric(d, StreamId(2)));
    }

    #[test]
    fn launch_attribute_priority_beats_stream_priority() {
        let d = DeviceId(0);
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |hi_first: bool| {
            let mut sim = Sim::new(h100());
            sim.set_stream_priority(d, StreamId(0), 0).unwrap();
            sim.set_stream_priority(d, StreamId(1), 0).unwrap();
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            let hi = KernelAttrs {
                priority: Some(1),
                ..KernelAttrs::default()
            };
            if hi_first {
                enq(sim.kernel_with(d, kind.clone(), &[a], &[a], StreamId(1), hi));
                enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(0)));
            } else {
                enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(0)));
                enq(sim.kernel_with(d, kind.clone(), &[a], &[a], StreamId(1), hi));
            }
            sim.start_ready().unwrap();
            let started: Vec<StreamId> = sim
                .operations()
                .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.start_ns.is_some())
                .map(|o| o.stream)
                .collect();
            started
        };
        let low_submitted_first = run(false);
        assert_eq!(low_submitted_first, vec![StreamId(1)]);
        let high_submitted_first = run(true);
        assert_eq!(high_submitted_first, vec![StreamId(1)]);
    }

    #[test]
    fn kernel_with_none_inherits_stream_priority() {
        let d = DeviceId(0);
        let kind = KernelKind::other(1 << 40, 4096);
        let mut sim = Sim::new(h100());
        sim.set_stream_priority(d, StreamId(0), 0).unwrap();
        sim.set_stream_priority(d, StreamId(1), 0).unwrap();
        sim.set_stream_priority(d, StreamId(2), 5).unwrap();
        let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
        sim.synchronize().unwrap();
        enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(1)));
        enq(sim.kernel_with(d, kind, &[a], &[a], StreamId(2), KernelAttrs::default()));
        sim.start_ready().unwrap();
        let started: Vec<StreamId> = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.start_ns.is_some())
            .map(|o| o.stream)
            .collect();
        assert_eq!(
            started,
            vec![StreamId(2)],
            "default KernelAttrs must inherit stream create priority; {started:?}"
        );
    }

    #[test]
    fn kernel_with_priority_zero_overrides_high_stream() {
        let d = DeviceId(0);
        let kind = KernelKind::other(1 << 40, 4096);
        let mut sim = Sim::new(h100());
        sim.set_stream_priority(d, StreamId(1), 0).unwrap();
        sim.set_stream_priority(d, StreamId(2), 5).unwrap();
        let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
        sim.synchronize().unwrap();
        enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(1)));
        enq(sim.kernel_with(
            d,
            kind,
            &[a],
            &[a],
            StreamId(2),
            KernelAttrs {
                priority: Some(0),
                ..KernelAttrs::default()
            },
        ));
        sim.start_ready().unwrap();
        let started: Vec<StreamId> = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.start_ns.is_some())
            .map(|o| o.stream)
            .collect();
        assert_eq!(
            started,
            vec![StreamId(1)],
            "explicit launch priority 0 must not keep the stream's 5; {started:?}"
        );
    }

    #[test]
    fn kernel_with_capture_records_launch_priority() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.set_stream_priority(d, s, 0).unwrap();
        let a = sim.malloc(d, 4096).unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel_with(
            d,
            KernelKind::other(8, 8),
            &[a],
            &[a],
            s,
            KernelAttrs {
                priority: Some(7),
                ..KernelAttrs::default()
            },
        ));
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_kernel_node_get_priority(g, 0).unwrap(), 7);
        let exec = sim
            .instantiate_graph_with_flags(g, GraphInstantiateFlags::DEVICE_LAUNCH)
            .expect("device-launch allows launch-attribute priority");
        assert_eq!(sim.graph_exec_kernel_node_get_priority(exec, 0).unwrap(), 7);
    }

    #[test]
    fn use_node_priority_uses_captured_launch_attr() {
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |node_pri: bool| {
            let mut sim = Sim::new(h100().with_compute_slots(1));
            let d = DeviceId(0);
            sim.set_stream_priority(d, StreamId(0), 0).unwrap();
            sim.set_stream_priority(d, StreamId(1), 0).unwrap();
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            let b = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            enq(sim.memcpy_pinned_to_device(d, b, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            sim.begin_capture(d, StreamId(0)).unwrap();
            enq(sim.kernel_with(
                d,
                kind.clone(),
                &[a],
                &[a],
                StreamId(0),
                KernelAttrs {
                    priority: Some(5),
                    ..KernelAttrs::default()
                },
            ));
            let g = sim.end_capture().unwrap();
            assert_eq!(sim.graph_kernel_node_get_priority(g, 0).unwrap(), 5);
            let flags = if node_pri {
                GraphInstantiateFlags::USE_NODE_PRIORITY
            } else {
                0
            };
            let _ = sim.instantiate_graph_with_flags(g, flags).unwrap();
            enq(sim.kernel(d, kind.clone(), &[b], &[b], StreamId(1)));
            let n = sim.launch_graph(g, StreamId(0)).unwrap();
            assert_eq!(n, 1);
            sim.start_ready().unwrap();
            sim.operations()
                .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.start_ns.is_some())
                .map(|o| o.stream)
                .collect::<Vec<StreamId>>()
        };
        let with_flag = run(true);
        assert_eq!(
            with_flag,
            vec![StreamId(0)],
            "UseNodePriority must honor captured launch-attribute priority; {with_flag:?}"
        );
        let without = run(false);
        assert_eq!(
            without,
            vec![StreamId(1)],
            "default instantiate must ignore captured launch-attribute priority; {without:?}"
        );
    }

    #[test]
    fn graph_cluster_copies_and_device_launch_allows() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 8).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        assert!(sim.graph_kernel_node_get_cluster(g, 0).unwrap().is_none());
        let dim = ClusterDim::x(4);
        sim.graph_kernel_node_set_cluster(g, 0, Some(dim)).unwrap();
        let h = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(h, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_kernel_node_copy_attributes(h, 0, g, 0).unwrap();
        assert_eq!(sim.graph_kernel_node_get_cluster(h, 0).unwrap(), Some(dim));
        let exec = sim
            .instantiate_graph_with_flags(g, GraphInstantiateFlags::DEVICE_LAUNCH)
            .expect("device-launch allows cluster");
        assert_eq!(
            sim.graph_exec_kernel_node_get_cluster(exec, 0).unwrap(),
            Some(dim)
        );
    }

    #[test]
    fn higher_stream_priority_starts_first_when_compute_contends() {
        let d = DeviceId(0);
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |hi_first: bool| {
            let mut sim = Sim::new(h100());
            sim.set_stream_priority(d, StreamId(0), 0).unwrap();
            sim.set_stream_priority(d, StreamId(1), 1).unwrap();
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            if hi_first {
                enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(1)));
                enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(0)));
            } else {
                enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(0)));
                enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(1)));
            }
            sim.start_ready().unwrap();
            let started: Vec<StreamId> = sim
                .operations()
                .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.start_ns.is_some())
                .map(|o| o.stream)
                .collect();
            started
        };
        let low_submitted_first = run(false);
        assert_eq!(low_submitted_first, vec![StreamId(1)]);
        let high_submitted_first = run(true);
        assert_eq!(high_submitted_first, vec![StreamId(1)]);
    }

    #[test]
    fn stream_copy_attributes_copies_priority_and_sm() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        sim.set_stream_priority(d, StreamId(1), 7).unwrap();
        sim.set_stream_sm_permille(d, StreamId(1), 250).unwrap();
        sim.set_stream_mem_sync_domain(d, StreamId(1), MemSyncDomain::Remote)
            .unwrap();
        let map = MemSyncDomainMap {
            default: 0,
            remote: 2,
        };
        sim.set_stream_mem_sync_domain_map(d, StreamId(1), map)
            .unwrap();
        sim.stream_copy_attributes(d, StreamId(2), d, StreamId(1))
            .unwrap();
        assert_eq!(sim.stream_priority(d, StreamId(2)), 7);
        assert_eq!(sim.stream_sm_permille(d, StreamId(2)), 250);
        assert_eq!(
            sim.stream_mem_sync_domain(d, StreamId(2)),
            MemSyncDomain::Remote
        );
        assert_eq!(sim.stream_mem_sync_domain_map(d, StreamId(2)).unwrap(), map);
        assert_eq!(
            sim.stream_sync_policy(d, StreamId(2)),
            SynchronizationPolicy::Auto
        );
        sim.set_stream_sync_policy(d, StreamId(1), SynchronizationPolicy::BlockingSync)
            .unwrap();
        sim.stream_copy_attributes(d, StreamId(2), d, StreamId(1))
            .unwrap();
        assert_eq!(
            sim.stream_sync_policy(d, StreamId(2)),
            SynchronizationPolicy::BlockingSync
        );
        let err = sim
            .stream_copy_attributes(DeviceId(0), StreamId(0), DeviceId(1), StreamId(0))
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("device mismatch"), "{why}"),
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn use_node_priority_beats_launch_stream() {
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |node_pri: bool| {
            let mut sim = Sim::new(h100().with_compute_slots(1));
            let d = DeviceId(0);
            sim.set_stream_priority(d, StreamId(0), 0).unwrap();
            sim.set_stream_priority(d, StreamId(1), 0).unwrap();
            sim.set_stream_priority(d, StreamId(2), 5).unwrap();
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            let b = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            enq(sim.memcpy_pinned_to_device(d, b, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            sim.begin_capture(d, StreamId(2)).unwrap();
            enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(2)));
            let g = sim.end_capture().unwrap();
            assert_eq!(sim.graph_kernel_node_get_priority(g, 0).unwrap(), 5);
            let flags = if node_pri {
                GraphInstantiateFlags::USE_NODE_PRIORITY
            } else {
                0
            };
            let _ = sim.instantiate_graph_with_flags(g, flags).unwrap();
            enq(sim.kernel(d, kind.clone(), &[b], &[b], StreamId(1)));
            let n = sim.launch_graph(g, StreamId(0)).unwrap();
            assert_eq!(n, 1);
            sim.start_ready().unwrap();
            let started: Vec<StreamId> = sim
                .operations()
                .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.start_ns.is_some())
                .map(|o| o.stream)
                .collect();
            started
        };
        let with_flag = run(true);
        assert_eq!(
            with_flag,
            vec![StreamId(0)],
            "UseNodePriority must start the captured kernel on the launch stream; {with_flag:?}"
        );
        let without = run(false);
        assert_eq!(
            without,
            vec![StreamId(1)],
            "default instantiate must use launch-stream priority 0 so the live kernel wins by submit order; {without:?}"
        );
    }

    #[test]
    fn graph_kernel_node_copy_attributes_retargets_priority() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.set_stream_priority(d, s, 3).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.set_stream_priority(d, s, 9).unwrap();
        let h = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(h, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        assert_eq!(sim.graph_kernel_node_get_priority(g, 0).unwrap(), 3);
        assert_eq!(sim.graph_kernel_node_get_priority(h, 0).unwrap(), 9);
        sim.graph_kernel_node_copy_attributes(g, 0, h, 0).unwrap();
        assert_eq!(sim.graph_kernel_node_get_priority(g, 0).unwrap(), 9);
        sim.graph_kernel_node_set_priority(g, 0, 1).unwrap();
        assert_eq!(sim.graph_kernel_node_get_priority(g, 0).unwrap(), 1);
        let exec = sim.instantiate_graph(g).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let err = sim.graph_add_kernel(exec, KernelKind::other(8, 8), &[a], &[a]);
        match err {
            Err(SimError::Invalid { why }) => assert!(why.contains("instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        let cap = sim.graph_kernel_node_set_priority(h, 0, 0).unwrap_err();
        match cap {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            e => panic!("{e:?}"),
        }
        let _end = sim.end_capture().unwrap();
    }

    #[test]
    fn graph_kernel_node_get_set_attribute_dispatches_without_retargeting_exec() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_empty(g).unwrap();
        assert_eq!(
            sim.graph_kernel_node_get_attribute(g, 0, KernelNodeAttr::Priority)
                .unwrap(),
            KernelNodeAttrValue::Priority(0)
        );
        sim.graph_kernel_node_set_attribute(
            g,
            0,
            KernelNodeAttr::Priority,
            KernelNodeAttrValue::Priority(7),
        )
        .unwrap();
        assert_eq!(sim.graph_kernel_node_get_priority(g, 0).unwrap(), 7);
        sim.graph_kernel_node_set_attribute(
            g,
            0,
            KernelNodeAttr::NvlinkUtilCentric,
            KernelNodeAttrValue::NvlinkUtilCentric(true),
        )
        .unwrap();
        assert!(sim.graph_kernel_node_get_nvlink_util_centric(g, 0).unwrap());
        match sim.graph_kernel_node_get_attribute(g, 1, KernelNodeAttr::Priority) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not a kernel"), "{why}"),
            other => panic!("{other:?}"),
        }
        let exec = sim.instantiate_graph(g).unwrap();
        sim.graph_kernel_node_set_attribute(
            g,
            0,
            KernelNodeAttr::Priority,
            KernelNodeAttrValue::Priority(3),
        )
        .unwrap();
        assert_eq!(
            sim.graph_exec_kernel_node_get_attribute(exec, 0, KernelNodeAttr::Priority)
                .unwrap(),
            KernelNodeAttrValue::Priority(7)
        );
        sim.graph_exec_kernel_node_set_attribute(
            exec,
            0,
            KernelNodeAttr::Priority,
            KernelNodeAttrValue::Priority(1),
        )
        .unwrap();
        assert_eq!(sim.graph_exec_kernel_node_get_priority(exec, 0).unwrap(), 1);
        assert_eq!(sim.graph_kernel_node_get_priority(g, 0).unwrap(), 3);
        match sim.graph_kernel_node_set_attribute(
            g,
            0,
            KernelNodeAttr::Priority,
            KernelNodeAttrValue::NvlinkUtilCentric(false),
        ) {
            Err(SimError::Invalid { why }) => assert!(why.contains("kernel node attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        match sim.graph_kernel_node_set_attribute(
            g,
            0,
            KernelNodeAttr::Priority,
            KernelNodeAttrValue::Priority(0),
        ) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(
            sim.graph_kernel_node_get_attribute(g, 0, KernelNodeAttr::Priority)
                .unwrap(),
            KernelNodeAttrValue::Priority(3)
        );
        let _end = sim.end_capture().unwrap();
    }

    #[test]
    fn graph_kernel_set_params_does_not_retarget_instantiated_exec() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
        sim.synchronize().unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim.instantiate_graph(g).unwrap();
        let params = KernelNodeParams {
            kind: KernelKind::other(8, 8),
            reads: vec![KernelBuf::whole(b)],
            writes: vec![KernelBuf::whole(b)],
            cooperative: false,
        };
        sim.graph_kernel_set_params(g, 0, &params).unwrap();
        let (_, now) = sim.graph_unique_kernel(g).unwrap();
        assert_eq!(
            now.reads[0].id, a,
            "unique kernel on an instantiated id is the exec snapshot"
        );
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        let err = sim.synchronize().unwrap_err();
        match err {
            SimError::NotResident { alloc, device } => {
                assert_eq!(alloc, a);
                assert_eq!(device, d);
            }
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn graph_get_params_reads_definition_exec_reads_snapshot() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
        sim.synchronize().unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let got = sim.graph_kernel_get_params(g, 0).unwrap();
        assert_eq!(got.reads[0].id, a);
        assert_eq!(got.kind, KernelKind::other(8, 8));
        let err = sim.graph_exec_kernel_get_params(g, 0).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        let exec = sim.instantiate_graph(g).unwrap();
        let patched = KernelNodeParams {
            kind: KernelKind::other(16, 16),
            reads: vec![KernelBuf::whole(b)],
            writes: vec![KernelBuf::whole(b)],
            cooperative: false,
        };
        sim.graph_kernel_set_params(g, 0, &patched).unwrap();
        let def = sim.graph_kernel_get_params(g, 0).unwrap();
        assert_eq!(def.reads[0].id, b);
        assert_eq!(def.kind, KernelKind::other(16, 16));
        let snap = sim.graph_exec_kernel_get_params(g, 0).unwrap();
        assert_eq!(snap.reads[0].id, a, "exec GetParams is the snapshot");
        assert_eq!(snap.kind, KernelKind::other(8, 8));
        let snap2 = sim.graph_exec_kernel_get_params(exec, 0).unwrap();
        assert_eq!(snap2.reads[0].id, a);
        let err = sim.graph_memcpy_get_params(g, 0).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not a memcpy"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        let during = sim.graph_kernel_get_params(g, 0).unwrap();
        assert_eq!(during.reads[0].id, b);
        let _cap = sim.end_capture().unwrap();
    }

    #[test]
    fn graph_host_get_params_round_trips_set_params() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_host_func(g).unwrap();
        assert_eq!(
            sim.graph_host_get_params(g, 0).unwrap(),
            HostNodeParams::default()
        );
        let next = HostNodeParams {
            fn_id: 4,
            user_data: 40,
        };
        sim.graph_host_set_params(g, 0, next).unwrap();
        assert_eq!(sim.graph_host_get_params(g, 0).unwrap(), next);
        let exec = sim.instantiate_graph(g).unwrap();
        let later = HostNodeParams {
            fn_id: 5,
            user_data: 50,
        };
        sim.graph_exec_host_set_params(exec, 0, later).unwrap();
        assert_eq!(sim.graph_exec_host_get_params(exec, 0).unwrap(), later);
        assert_eq!(sim.graph_host_get_params(g, 0).unwrap(), next);
    }

    #[test]
    fn graph_memcpy_set_params_does_not_retarget_instantiated_exec() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        let op_a = MemcpyOp {
            src: Place::HostPinned,
            dst: Place::Device(d),
            alloc: a,
            bytes: 4096,
            offset: 0,
            ..MemcpyOp::default()
        };
        sim.graph_add_memcpy(g, op_a.clone()).unwrap();
        let _ = sim.instantiate_graph(g).unwrap();
        let mut op_b = op_a;
        op_b.alloc = b;
        sim.graph_memcpy_set_params(g, 0, &op_b).unwrap();
        let (_, now) = sim.graph_unique_memcpy(g).unwrap();
        assert_eq!(now.alloc, a, "unique memcpy is the exec snapshot");
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        let err = sim.synchronize().unwrap_err();
        match err {
            SimError::UnknownAlloc { alloc } | SimError::NotResident { alloc, .. } => {
                assert_eq!(alloc, a);
            }
            e => panic!("{e:?}"),
        }
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_memset(g, KernelBuf::whole(a)).unwrap();
        let _ = sim.instantiate_graph(g).unwrap();
        sim.graph_memset_set_params(g, 0, KernelBuf::whole(b))
            .unwrap();
        let (_, now) = sim.graph_unique_memset(g).unwrap();
        assert_eq!(now.id, a, "unique memset is the exec snapshot");
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        let err = sim.synchronize().unwrap_err();
        match err {
            SimError::NotResident { alloc, device } => {
                assert_eq!(alloc, a);
                assert_eq!(device, d);
            }
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn graph_kernel_set_params_before_instantiate_is_snapshotted() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
        sim.synchronize().unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let params = KernelNodeParams {
            kind: KernelKind::other(8, 8),
            reads: vec![KernelBuf::whole(b)],
            writes: vec![KernelBuf::whole(b)],
            cooperative: false,
        };
        sim.graph_kernel_set_params(g, 0, &params).unwrap();
        let _ = sim.instantiate_graph(g).unwrap();
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(b, d).unwrap());
    }

    #[test]
    fn graph_kernel_node_set_priority_does_not_retarget_instantiated_exec() {
        let kind = KernelKind::other(1 << 40, 4096);
        let mut sim = Sim::new(h100().with_compute_slots(1));
        let d = DeviceId(0);
        sim.set_stream_priority(d, StreamId(0), 0).unwrap();
        sim.set_stream_priority(d, StreamId(1), 0).unwrap();
        sim.set_stream_priority(d, StreamId(2), 5).unwrap();
        let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
        let b = sim.alloc(d, 4096, StreamId(0)).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, StreamId(0)));
        sim.synchronize().unwrap();
        sim.begin_capture(d, StreamId(2)).unwrap();
        enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(2)));
        let g = sim.end_capture().unwrap();
        let _ = sim
            .instantiate_graph_with_flags(g, GraphInstantiateFlags::USE_NODE_PRIORITY)
            .unwrap();
        assert_eq!(sim.graph_kernel_node_get_priority(g, 0).unwrap(), 5);
        assert_eq!(sim.graph_exec_kernel_node_get_priority(g, 0).unwrap(), 5);
        sim.graph_kernel_node_set_priority(g, 0, 0).unwrap();
        assert_eq!(sim.graph_kernel_node_get_priority(g, 0).unwrap(), 0);
        assert_eq!(sim.graph_exec_kernel_node_get_priority(g, 0).unwrap(), 5);
        enq(sim.kernel(d, kind.clone(), &[b], &[b], StreamId(1)));
        let n = sim.launch_graph(g, StreamId(0)).unwrap();
        assert_eq!(n, 1);
        sim.start_ready().unwrap();
        let started: Vec<StreamId> = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.start_ns.is_some())
            .map(|o| o.stream)
            .collect();
        assert_eq!(
            started,
            vec![StreamId(0)],
            "graph-side SetAttribute must not drop the exec snapshot priority; {started:?}"
        );
    }

    #[test]
    fn graph_exec_kernel_node_set_priority_retargets_launch() {
        let kind = KernelKind::other(1 << 40, 4096);
        let mut sim = Sim::new(h100().with_compute_slots(1));
        let d = DeviceId(0);
        sim.set_stream_priority(d, StreamId(0), 0).unwrap();
        sim.set_stream_priority(d, StreamId(1), 0).unwrap();
        sim.set_stream_priority(d, StreamId(2), 5).unwrap();
        let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
        let b = sim.alloc(d, 4096, StreamId(0)).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, StreamId(0)));
        sim.synchronize().unwrap();
        sim.begin_capture(d, StreamId(2)).unwrap();
        enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(2)));
        let g = sim.end_capture().unwrap();
        let _ = sim
            .instantiate_graph_with_flags(g, GraphInstantiateFlags::USE_NODE_PRIORITY)
            .unwrap();
        sim.graph_exec_kernel_node_set_priority(g, 0, 0).unwrap();
        assert_eq!(sim.graph_kernel_node_get_priority(g, 0).unwrap(), 5);
        assert_eq!(sim.graph_exec_kernel_node_get_priority(g, 0).unwrap(), 0);
        enq(sim.kernel(d, kind, &[b], &[b], StreamId(1)));
        let n = sim.launch_graph(g, StreamId(0)).unwrap();
        assert_eq!(n, 1);
        sim.start_ready().unwrap();
        let started: Vec<StreamId> = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.start_ns.is_some())
            .map(|o| o.stream)
            .collect();
        assert_eq!(
            started,
            vec![StreamId(1)],
            "exec SetAttribute must drop node priority so the live kernel wins; {started:?}"
        );
    }

    #[test]
    fn created_stream_priority_helper_matches_manual() {
        let mut sim = Sim::new(h100());
        sim.set_created_streams_priority(2).unwrap();
        assert_eq!(sim.stream_priority(DeviceId(0), StreamId(0)), 0);
        assert_eq!(sim.stream_priority(DeviceId(0), StreamId(1)), 1);
    }

    fn sync_policy_pair() -> (Sim, Sim, DeviceId, StreamId) {
        let p = HardwareProfile::parse("gpus=1\nhost_sync_blocking_ns=10000\nfp16_flops=1000000\n")
            .expect("host-sync profile");
        let mut auto = Sim::new(p.clone());
        let mut block = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = auto.alloc(d, 1 << 20, s).unwrap();
        let b = block.alloc(d, 1 << 20, s).unwrap();
        enq(auto.kernel(d, KernelKind::other(1 << 20, 1 << 20), &[a], &[a], s));
        enq(block.kernel(d, KernelKind::other(1 << 20, 1 << 20), &[b], &[b], s));
        block
            .set_stream_sync_policy(d, s, SynchronizationPolicy::BlockingSync)
            .unwrap();
        (auto, block, d, s)
    }

    #[test]
    fn stream_sync_policy_blocking_taxes_stream_wait() {
        let (mut auto, mut block, d, s) = sync_policy_pair();
        auto.synchronize_stream(d, s).unwrap();
        block.synchronize_stream(d, s).unwrap();
        assert_eq!(block.clock_ns(), auto.clock_ns().saturating_add(10_000));
    }

    #[test]
    fn stream_sync_policy_does_not_tax_device_sync() {
        let (mut auto, mut block, d, _) = sync_policy_pair();
        auto.synchronize().unwrap();
        block.synchronize().unwrap();
        assert_eq!(block.clock_ns(), auto.clock_ns());
        let p = HardwareProfile::parse("gpus=1\nhost_sync_blocking_ns=10000\n").unwrap();
        let mut device = Sim::new(p.clone());
        let mut auto_dev = Sim::new(p);
        device
            .set_stream_sync_policy(d, StreamId(0), SynchronizationPolicy::BlockingSync)
            .unwrap();
        let a = device.alloc(d, 1 << 20, StreamId(0)).unwrap();
        let b = auto_dev.alloc(d, 1 << 20, StreamId(0)).unwrap();
        enq(device.kernel(
            d,
            KernelKind::other(1 << 20, 1 << 20),
            &[a],
            &[a],
            StreamId(0),
        ));
        enq(auto_dev.kernel(
            d,
            KernelKind::other(1 << 20, 1 << 20),
            &[b],
            &[b],
            StreamId(0),
        ));
        device.synchronize_device(d).unwrap();
        auto_dev.synchronize_device(d).unwrap();
        assert_eq!(device.clock_ns(), auto_dev.clock_ns());
    }

    #[test]
    fn stream_sync_policy_taxes_event_wait_on_recording_stream() {
        let (mut auto, mut block, d, s) = sync_policy_pair();
        auto.create_event(EventId(1)).unwrap();
        block.create_event(EventId(1)).unwrap();
        enq(auto.record_event(d, EventId(1), s));
        enq(block.record_event(d, EventId(1), s));
        auto.synchronize_event(EventId(1)).unwrap();
        block.synchronize_event(EventId(1)).unwrap();
        assert_eq!(block.clock_ns(), auto.clock_ns().saturating_add(10_000));
    }

    #[test]
    fn stream_sync_policy_default_profile_tax_is_zero() {
        let mut ident = Sim::new(h100());
        let mut ident_auto = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        ident
            .set_stream_sync_policy(d, s, SynchronizationPolicy::BlockingSync)
            .unwrap();
        let c = ident.alloc(d, 4096, s).unwrap();
        let e = ident_auto.alloc(d, 4096, s).unwrap();
        enq(ident.kernel(d, KernelKind::other(8, 8), &[c], &[c], s));
        enq(ident_auto.kernel(d, KernelKind::other(8, 8), &[e], &[e], s));
        ident.synchronize_stream(d, s).unwrap();
        ident_auto.synchronize_stream(d, s).unwrap();
        assert_eq!(ident.clock_ns(), ident_auto.clock_ns());
    }

    #[test]
    fn synchronization_policy_parse() {
        assert_eq!(
            SynchronizationPolicy::parse("auto").unwrap(),
            SynchronizationPolicy::Auto
        );
        assert_eq!(
            SynchronizationPolicy::parse("spin").unwrap(),
            SynchronizationPolicy::Spin
        );
        assert_eq!(
            SynchronizationPolicy::parse("yield").unwrap(),
            SynchronizationPolicy::Yield
        );
        assert_eq!(
            SynchronizationPolicy::parse("blocking").unwrap(),
            SynchronizationPolicy::BlockingSync
        );
        let err = SynchronizationPolicy::parse("bogus").unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("unknown sync-policy"), "{why}"),
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn shared_memory_mode_parse() {
        assert_eq!(
            SharedMemoryMode::parse("default").unwrap(),
            SharedMemoryMode::Default
        );
        assert_eq!(
            SharedMemoryMode::parse("four").unwrap(),
            SharedMemoryMode::FourByte
        );
        assert_eq!(
            SharedMemoryMode::parse("eight").unwrap(),
            SharedMemoryMode::EightByte
        );
        let err = SharedMemoryMode::parse("bogus").unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("unknown shared-mem"), "{why}"),
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn portable_cluster_mode_parse() {
        assert_eq!(
            PortableClusterMode::parse("default").unwrap(),
            PortableClusterMode::Default
        );
        assert_eq!(
            PortableClusterMode::parse("portable").unwrap(),
            PortableClusterMode::RequirePortable
        );
        assert_eq!(
            PortableClusterMode::parse("non-portable").unwrap(),
            PortableClusterMode::AllowNonPortable
        );
        let err = PortableClusterMode::parse("bogus").unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("unknown portable-cluster"), "{why}"),
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn portable_shared_mode_parse() {
        assert_eq!(
            PortableSharedMode::parse("default").unwrap(),
            PortableSharedMode::Default
        );
        assert_eq!(
            PortableSharedMode::parse("portable").unwrap(),
            PortableSharedMode::RequirePortable
        );
        assert_eq!(
            PortableSharedMode::parse("non-portable").unwrap(),
            PortableSharedMode::AllowNonPortable
        );
        let err = PortableSharedMode::parse("bogus").unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("unknown portable-shared"), "{why}"),
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn nvlink_util_centric_parse() {
        assert!(!parse_nvlink_util_centric("0").unwrap());
        assert!(parse_nvlink_util_centric("1").unwrap());
        let err = parse_nvlink_util_centric("bogus").unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("unknown nvlink-util"), "{why}"),
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn stream_sync_policy_spin_and_yield_differ() {
        let p = HardwareProfile::parse(
            "gpus=1\nhost_sync_spin_ns=1000\nhost_sync_yield_ns=5000\nhost_sync_blocking_ns=9000\n",
        )
        .unwrap();
        let run = |policy: SynchronizationPolicy| {
            let mut sim = Sim::new(p.clone());
            let d = DeviceId(0);
            let s = StreamId(0);
            sim.set_stream_sync_policy(d, s, policy).unwrap();
            let a = sim.alloc(d, 4096, s).unwrap();
            enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
            sim.synchronize_stream(d, s).unwrap();
            sim.clock_ns()
        };
        let auto = run(SynchronizationPolicy::Auto);
        let spin = run(SynchronizationPolicy::Spin);
        let yield_p = run(SynchronizationPolicy::Yield);
        let block = run(SynchronizationPolicy::BlockingSync);
        assert_eq!(spin, auto.saturating_add(1_000));
        assert_eq!(yield_p, auto.saturating_add(5_000));
        assert_eq!(block, auto.saturating_add(9_000));
    }

    #[test]
    fn idle_stream_sync_still_pays_blocking_tax() {
        let p = HardwareProfile::parse("gpus=1\nhost_sync_blocking_ns=10000\n").unwrap();
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.set_stream_sync_policy(d, s, SynchronizationPolicy::BlockingSync)
            .unwrap();
        let t0 = sim.clock_ns();
        sim.synchronize_stream(d, s).unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(10_000));
        sim.synchronize().unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(10_000));
    }

    #[test]
    fn device_schedule_flags_are_auto_fallback() {
        let p =
            HardwareProfile::parse("gpus=1\nhost_sync_spin_ns=1000\nhost_sync_blocking_ns=9000\n")
                .unwrap();
        let d = DeviceId(0);
        let s = StreamId(0);
        let mut spin = Sim::new(p.clone());
        assert_eq!(
            spin.get_device_flags(d).unwrap(),
            DeviceFlags::SCHEDULE_AUTO
        );
        spin.set_device_flags(d, DeviceFlags::SCHEDULE_SPIN)
            .unwrap();
        assert_eq!(
            spin.get_device_flags(d).unwrap(),
            DeviceFlags::SCHEDULE_SPIN
        );
        let t0 = spin.clock_ns();
        spin.synchronize_stream(d, s).unwrap();
        assert_eq!(spin.clock_ns(), t0.saturating_add(1_000));
        spin.set_device_flags(d, DeviceFlags::SCHEDULE_SPIN | DeviceFlags::MAP_HOST)
            .unwrap();
        let t_map = spin.clock_ns();
        spin.synchronize_stream(d, s).unwrap();
        assert_eq!(spin.clock_ns(), t_map.saturating_add(1_000));
        let mut override_p = Sim::new(p);
        override_p
            .set_device_flags(d, DeviceFlags::SCHEDULE_SPIN)
            .unwrap();
        override_p
            .set_stream_sync_policy(d, s, SynchronizationPolicy::BlockingSync)
            .unwrap();
        let t1 = override_p.clock_ns();
        override_p.synchronize_stream(d, s).unwrap();
        assert_eq!(override_p.clock_ns(), t1.saturating_add(9_000));
        let mut sim = Sim::new(h100());
        match sim.set_device_flags(d, 3) {
            Err(SimError::Invalid { why }) => assert!(why.contains("device schedule"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.set_device_flags(d, DeviceFlags::MAP_HOST).unwrap();
        assert_eq!(sim.get_device_flags(d).unwrap(), DeviceFlags::MAP_HOST);
        sim.set_device_flags(d, DeviceFlags::LMEM_RESIZE_TO_MAX)
            .unwrap();
        assert_eq!(
            sim.get_device_flags(d).unwrap(),
            DeviceFlags::LMEM_RESIZE_TO_MAX
        );
        match sim.set_device_flags(d, 0x20) {
            Err(SimError::Invalid { why }) => assert!(why.contains("device flags"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.set_device_flags(d, DeviceFlags::SCHEDULE_AUTO).unwrap();
        sim.begin_capture(d, s).unwrap();
        assert_eq!(sim.get_device_flags(d).unwrap(), DeviceFlags::SCHEDULE_AUTO);
        match sim.set_device_flags(d, DeviceFlags::SCHEDULE_SPIN) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn query_stream_does_not_pay_sync_policy_tax() {
        let p = HardwareProfile::parse("gpus=1\nhost_sync_blocking_ns=10000\n").unwrap();
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.set_stream_sync_policy(d, s, SynchronizationPolicy::BlockingSync)
            .unwrap();
        let t0 = sim.clock_ns();
        assert!(sim.query_stream(d, s).unwrap());
        assert_eq!(sim.clock_ns(), t0);
    }

    #[test]
    fn set_created_streams_sync_policy_skips_null() {
        let mut sim = Sim::new(h100());
        sim.set_created_streams_sync_policy(2, SynchronizationPolicy::Spin)
            .unwrap();
        assert_eq!(
            sim.stream_sync_policy(DeviceId(0), StreamId(0)),
            SynchronizationPolicy::Auto
        );
        assert_eq!(
            sim.stream_sync_policy(DeviceId(0), StreamId(1)),
            SynchronizationPolicy::Spin
        );
        let err = sim
            .set_stream_sync_policy(DeviceId(9), StreamId(0), SynchronizationPolicy::Auto)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("device not in profile"), "{why}"),
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn synchronize_stream_does_not_wait_for_other_streams() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let bytes = 64u64 << 20;
        let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
        sim.synchronize().unwrap();
        let t0 = sim.clock_ns();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], StreamId(0)));
        let b = sim.alloc(d, bytes, StreamId(1)).unwrap();
        enq(sim.memcpy_pinned_to_device(d, b, bytes, StreamId(1)));
        sim.synchronize_stream(d, StreamId(0)).unwrap();
        let partial = sim.clock_ns().saturating_sub(t0);
        assert!(sim.stream_is_idle(d, StreamId(0)).unwrap());
        assert!(!sim.stream_is_idle(d, StreamId(1)).unwrap());
        sim.synchronize().unwrap();
        let full = sim.clock_ns().saturating_sub(t0);
        assert!(
            partial < full,
            "stream-0 sync must leave the long H2D running; partial={partial} full={full}"
        );
    }

    #[test]
    fn synchronize_idle_stream_does_not_start_other_kernels() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
        sim.synchronize().unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 40, 4096), &[a], &[a], StreamId(1)));
        sim.synchronize_stream(d, StreamId(0)).unwrap();
        assert!(sim.stream_is_idle(d, StreamId(0)).unwrap());
        let started = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.start_ns.is_some())
            .count();
        assert_eq!(
            started, 0,
            "idle stream-0 sync must not start the leftover kernel"
        );
        assert!(!sim.stream_is_idle(d, StreamId(1)).unwrap());
        sim.synchronize().unwrap();
        assert!(sim.stream_is_idle(d, StreamId(1)).unwrap());
    }

    #[test]
    fn two_compute_slots_overlap_kernels_on_two_streams() {
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |slots: u8| {
            let mut sim = Sim::new(h100().with_compute_slots(slots));
            let d = DeviceId(0);
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(1)));
            enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(2)));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let serial = run(1);
        let overlap = run(2);
        assert!(
            overlap < serial,
            "two Hyper-Q slots must overlap independent kernels; overlap={overlap} serial={serial}"
        );
        let same_stream = {
            let mut sim = Sim::new(h100().with_compute_slots(2));
            let d = DeviceId(0);
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(1)));
            enq(sim.kernel(d, kind, &[a], &[a], StreamId(1)));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        assert!(
            overlap < same_stream,
            "Hyper-Q must not overlap kernels on one stream; overlap={overlap} same={same_stream}"
        );
    }

    #[test]
    fn two_compute_slots_start_both_ready_kernels() {
        let d = DeviceId(0);
        let kind = KernelKind::other(1 << 40, 4096);
        let mut sim = Sim::new(h100().with_compute_slots(2));
        let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
        sim.synchronize().unwrap();
        enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(1)));
        enq(sim.kernel(d, kind, &[a], &[a], StreamId(2)));
        sim.start_ready().unwrap();
        let started: Vec<StreamId> = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.start_ns.is_some())
            .map(|o| o.stream)
            .collect();
        assert_eq!(
            started.len(),
            2,
            "two Hyper-Q slots must start both ready kernels; started={started:?}"
        );
    }

    #[test]
    fn cooperative_kernel_occupies_all_compute_slots() {
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |coop_second: bool| {
            let mut sim = Sim::new(h100().with_compute_slots(2));
            let d = DeviceId(0);
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(1)));
            if coop_second {
                enq(sim.cooperative_kernel(d, kind.clone(), &[a], &[a], StreamId(2)));
            } else {
                enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(2)));
            }
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let overlap = run(false);
        let serial = run(true);
        assert!(
            overlap < serial,
            "cooperative must not Hyper-Q overlap leftover; overlap={overlap} serial={serial}"
        );
    }

    #[test]
    fn cooperative_kernel_rejects_unsupported_device() {
        let mut sim = Sim::new(h100().with_cooperative_launch(false));
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        match sim.cooperative_kernel(d, KernelKind::other(8, 8), &[a], &[a], s) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("cooperative launch not supported"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        let g = sim.create_graph(d, s).unwrap();
        match sim.graph_add_cooperative_kernel(g, KernelKind::other(8, 8), &[a], &[a]) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("cooperative launch not supported"), "{why}");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn cooperative_kernel_capture_and_graph_add() {
        let kind = KernelKind::other(8, 8);
        let d = DeviceId(0);
        let s = StreamId(0);
        let mut cap = Sim::new(h100());
        let a = cap.alloc(d, 4096, s).unwrap();
        enq(cap.memcpy_pinned_to_device(d, a, 4096, s));
        cap.synchronize().unwrap();
        cap.begin_capture(d, s).unwrap();
        enq(cap.cooperative_kernel(d, kind.clone(), &[a], &[], s));
        let g = cap.end_capture().unwrap();
        let n = cap.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        cap.synchronize().unwrap();
        let coop = cap.operations().any(|o| {
            matches!(
                o.kind,
                GpuOp::Kernel {
                    cooperative: true,
                    ..
                }
            )
        });
        assert!(coop, "captured cooperative kernel");

        let mut built = Sim::new(h100());
        let b = built.alloc(d, 4096, s).unwrap();
        enq(built.memcpy_pinned_to_device(d, b, 4096, s));
        built.synchronize().unwrap();
        let h = built.create_graph(d, s).unwrap();
        built
            .graph_add_cooperative_kernel(h, kind, &[b], &[])
            .unwrap();
        let _ = built.instantiate_graph(h).unwrap();
        let n = built.launch_graph(h, s).unwrap();
        assert_eq!(n, 1);
        built.synchronize().unwrap();
    }

    #[test]
    fn stream_sm_permille_slows_compute_bound_kernel() {
        let d = DeviceId(0);
        let kind = KernelKind::other(1 << 40, 8);
        let run = |permille: u16| {
            let mut sim = Sim::new(h100());
            sim.set_stream_sm_permille(d, StreamId(1), permille)
                .unwrap();
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(1)));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let full = run(1000);
        let quarter = run(250);
        assert!(
            quarter > full,
            "250‰ SMs must slow a compute-bound kernel; quarter={quarter} full={full}"
        );
        let err = Sim::new(h100())
            .set_stream_sm_permille(d, StreamId(1), 0)
            .unwrap_err();
        assert!(
            format!("{err:?}").contains("sm permille must be 1..=1000"),
            "{err:?}"
        );
        let err = Sim::new(h100())
            .set_stream_sm_permille(d, StreamId(1), 1001)
            .unwrap_err();
        assert!(
            format!("{err:?}").contains("sm permille must be 1..=1000"),
            "{err:?}"
        );
        let mem = |permille: u16| {
            let mut sim = Sim::new(h100());
            sim.set_stream_sm_permille(d, StreamId(1), permille)
                .unwrap();
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, KernelKind::other(8, 1 << 40), &[a], &[a], StreamId(1)));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let mem_full = mem(1000);
        let mem_quarter = mem(250);
        assert_eq!(
            mem_quarter, mem_full,
            "250‰ SMs must keep full HBM for a memory-bound kernel; quarter={mem_quarter} full={mem_full}"
        );
    }

    #[test]
    fn synchronize_stream_ignores_cancel_on_other_stream() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
        sim.synchronize().unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 40, 4096), &[a], &[a], StreamId(1)));
        enq(sim.kernel(d, KernelKind::other(8, 4096), &[a], &[a], StreamId(1)));
        sim.start_ready().unwrap();
        let n = sim.cancel_stream(d, StreamId(1)).unwrap();
        assert!(n >= 1);
        sim.synchronize_stream(d, StreamId(0)).unwrap();
        match sim.synchronize() {
            Err(SimError::Cancelled { n: skipped, .. }) => assert!(skipped >= 1),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn synchronize_event_does_not_drain_the_rest_of_the_stream() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let ev = EventId(3);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        let t0 = sim.clock_ns();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        enq(sim.record_event(d, ev, s));
        let bytes = 64u64 << 20;
        enq(sim.memcpy_pinned_to_device(d, a, bytes, s));
        sim.synchronize_event(ev).unwrap();
        let partial = sim.clock_ns().saturating_sub(t0);
        assert!(sim.event_complete(ev));
        assert!(!sim.stream_is_idle(d, s).unwrap());
        sim.synchronize().unwrap();
        let full = sim.clock_ns().saturating_sub(t0);
        assert!(
            partial < full,
            "event sync must not wait for the later H2D; partial={partial} full={full}"
        );
    }

    #[test]
    fn synchronize_event_unknown_is_semantic() {
        let mut sim = Sim::new(h100());
        match sim.synchronize_event(EventId(99)) {
            Err(SimError::UnknownEvent { event: 99 }) => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn idle_until_jumps_clock_after_drain() {
        let mut sim = Sim::new(h100());
        let jumped = sim.idle_until(1_000_000).unwrap();
        assert_eq!(jumped, 1_000_000);
        assert_eq!(sim.clock_ns(), 1_000_000);
        let again = sim.idle_until(500_000).unwrap();
        assert_eq!(again, 0);
        assert_eq!(sim.clock_ns(), 1_000_000);
    }

    #[test]
    fn idle_until_drains_in_flight_work() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let k = sim
            .kernel(d, KernelKind::other(8, 8), &[a], &[a], s)
            .unwrap();
        assert!(!sim.operation(k).unwrap().done);
        let jumped = sim.idle_until(0).unwrap();
        assert_eq!(jumped, 0);
        assert!(sim.operation(k).unwrap().done);
        assert!(sim.clock_ns() > 0);
    }

    #[test]
    fn event_elapsed_ns_is_record_done_delta() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let start = EventId(1);
        let end = EventId(2);
        enq(sim.record_event(d, start, s));
        enq(sim.kernel(d, KernelKind::other(1 << 20, 4096), &[a], &[a], s));
        enq(sim.record_event(d, end, s));
        match sim.event_elapsed_ns(start, end) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not complete"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.synchronize().unwrap();
        let ns = sim.event_elapsed_ns(start, end).unwrap();
        assert!(ns > 0, "elapsed={ns}");
        match sim.event_elapsed_ns(end, start) {
            Err(SimError::Invalid { why }) => assert!(why.contains("end before start"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.event_elapsed_ns(EventId(99), end) {
            Err(SimError::UnknownEvent { event: 99 }) => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn query_event_does_not_wait() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let ev = EventId(4);
        match sim.query_event(ev) {
            Err(SimError::UnknownEvent { event: 4 }) => {}
            other => panic!("{other:?}"),
        }
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 20, 4096), &[a], &[a], s));
        enq(sim.record_event(d, ev, s));
        assert!(!sim.query_event(ev).unwrap());
        sim.synchronize().unwrap();
        assert!(sim.query_event(ev).unwrap());
    }

    #[test]
    fn query_stream_does_not_wait() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        match sim.query_stream(DeviceId(99), s) {
            Err(SimError::Invalid { why }) => assert!(why.contains("device"), "{why}"),
            other => panic!("{other:?}"),
        }
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 20, 4096), &[a], &[a], s));
        let t0 = sim.clock_ns();
        assert!(!sim.query_stream(d, s).unwrap());
        assert_eq!(sim.clock_ns(), t0);
        sim.synchronize().unwrap();
        assert!(sim.query_stream(d, s).unwrap());
    }

    #[test]
    fn mem_info_tracks_free_and_total() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let (free0, total) = sim.mem_info(d).unwrap();
        assert_eq!(total, sim.profile().gpu(d).unwrap().hbm_bytes);
        assert_eq!(free0, total);
        let bytes = 1u64 << 20;
        let a = sim.alloc(d, bytes, s).unwrap();
        sim.synchronize().unwrap();
        let (free1, total1) = sim.mem_info(d).unwrap();
        assert_eq!(total1, total);
        assert_eq!(free1, total.saturating_sub(sim.hbm_used(d).unwrap()));
        assert!(free1 < free0);
        sim.free(d, a, s).unwrap();
        sim.synchronize().unwrap();
        let (free2, _) = sim.mem_info(d).unwrap();
        assert_eq!(free2, total);
    }

    #[test]
    fn malloc_is_resident_without_stream_sync() {
        let mut queued = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let a = queued.alloc(d, bytes, s).unwrap();
        assert!(!queued.is_resident(a, d).unwrap());
        let (free0, total) = queued.mem_info(d).unwrap();
        assert_eq!(free0, total);

        let mut sim = Sim::new(h100());
        let b = sim.malloc(d, bytes).unwrap();
        assert!(sim.is_resident(b, d).unwrap());
        let (free, total) = sim.mem_info(d).unwrap();
        assert_eq!(free, total.saturating_sub(bytes));
        enq(sim.kernel(d, KernelKind::other(1 << 20, bytes), &[b], &[b], s));
        sim.synchronize_stream(d, s).unwrap();
        assert!(sim.is_resident(b, d).unwrap());
    }

    #[test]
    fn ipc_open_kernel_shares_hbm() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let a = sim.malloc(d, bytes).unwrap();
        let h = sim.ipc_get(a).unwrap();
        assert_eq!(sim.ipc_get(a).unwrap(), h);
        let imp = sim.ipc_open(d, h).unwrap();
        assert!(sim.is_ipc_import(imp).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        enq(sim.kernel(d, KernelKind::other(1 << 20, bytes), &[imp], &[imp], s));
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        sim.ipc_close(imp).unwrap();
        assert!(!sim.is_ipc_import(imp).unwrap());
        sim.free_sync(a).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
    }

    #[test]
    fn ipc_open_with_flags_lazy_peer_is_noop() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let bytes = 4096u64;
        let a = sim.malloc(d, bytes).unwrap();
        let h = sim.ipc_get(a).unwrap();
        let imp = sim
            .ipc_open_with_flags(d, h, IpcMemFlags::LAZY_ENABLE_PEER_ACCESS)
            .unwrap();
        assert!(sim.is_ipc_import(imp).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        sim.ipc_close(imp).unwrap();
        let imp0 = sim.ipc_open_with_flags(d, h, IpcMemFlags::DEFAULT).unwrap();
        assert!(sim.is_ipc_import(imp0).unwrap());
        sim.ipc_close(imp0).unwrap();
        match sim.ipc_open_with_flags(d, h, 2) {
            Err(SimError::Invalid { why }) => assert!(why.contains("ipc open flags"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, StreamId(0)).unwrap();
        match sim.ipc_open_with_flags(d, h, 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        sim.free_sync(a).unwrap();
    }

    #[test]
    fn ipc_free_source_while_mapped_and_close_via_free() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let bytes = 4096u64;
        let a = sim.malloc(d, bytes).unwrap();
        let h = sim.ipc_get(a).unwrap();
        let imp = sim.ipc_open(d, h).unwrap();
        match sim.free_sync(a) {
            Err(SimError::Invalid { why }) => assert!(why.contains("ipc mapped")),
            other => panic!("{other:?}"),
        }
        sim.free_sync(imp).unwrap();
        sim.free_sync(a).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        match sim.ipc_open(d, h) {
            Err(SimError::Invalid { why }) => assert!(why.contains("freed")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn ipc_rejects_managed_host_and_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let m = sim.alloc_managed(4096).unwrap();
        match sim.ipc_get(m) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not device ipc")),
            other => panic!("{other:?}"),
        }
        let host = sim.alloc_host_pinned(4096).unwrap();
        match sim.ipc_get(host) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not device ipc")),
            other => panic!("{other:?}"),
        }
        let a = sim.malloc(d, 4096).unwrap();
        sim.begin_capture(d, s).unwrap();
        match sim.ipc_get(a) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        sim.free_sync(a).unwrap();
        sim.free_sync(m).unwrap();
        sim.free_host_pinned(host).unwrap();
    }

    #[test]
    fn malloc_waits_for_other_streams_alloc_async_does_not() {
        let mut async_sim = Sim::new(h100());
        let d = DeviceId(0);
        let s0 = StreamId(0);
        let s1 = StreamId(1);
        let a = async_sim.alloc(d, 4096, s0).unwrap();
        async_sim.synchronize_stream(d, s0).unwrap();
        enq(async_sim.kernel(d, KernelKind::other(1 << 30, 4096), &[a], &[a], s0));
        let t0 = async_sim.clock_ns();
        let b = async_sim.alloc(d, 4096, s1).unwrap();
        async_sim.synchronize_stream(d, s1).unwrap();
        let async_ns = async_sim.clock_ns().saturating_sub(t0);
        assert!(async_sim.is_resident(b, d).unwrap());

        let mut sync_sim = Sim::new(h100());
        let c = sync_sim.alloc(d, 4096, s0).unwrap();
        sync_sim.synchronize_stream(d, s0).unwrap();
        enq(sync_sim.kernel(d, KernelKind::other(1 << 30, 4096), &[c], &[c], s0));
        let t1 = sync_sim.clock_ns();
        let e = sync_sim.malloc(d, 4096).unwrap();
        let malloc_ns = sync_sim.clock_ns().saturating_sub(t1);
        assert!(sync_sim.is_resident(e, d).unwrap());
        assert!(
            malloc_ns > async_ns,
            "cudaMalloc must wait for the other stream; malloc={malloc_ns} async={async_ns}"
        );
    }

    #[test]
    fn malloc_oom_is_returned_at_the_call() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let cap = sim.profile().gpu(d).unwrap().hbm_bytes;
        let _full = sim.malloc(d, cap).unwrap();
        let err = sim.malloc(d, 1).unwrap_err();
        match err {
            SimError::Oom { need: 1, .. } => {}
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn free_sync_waits_then_releases_hbm() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let a = sim.malloc(d, bytes).unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 20, bytes), &[a], &[a], s));
        sim.free_sync(a).unwrap();
        assert!(!sim.is_resident(a, d).unwrap());
        let (free, total) = sim.mem_info(d).unwrap();
        assert_eq!(free, total);
        let b = sim.malloc(d, total).unwrap();
        assert!(sim.is_resident(b, d).unwrap());
    }

    #[test]
    fn memcpy_sync_leaves_the_stream_idle() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 4096u64;
        let a = sim.malloc(d, bytes).unwrap();
        let op = sim
            .memcpy_sync(
                d,
                MemcpyOp {
                    src: Place::HostPinned,
                    dst: Place::Device(d),
                    alloc: a,
                    bytes,
                    offset: 0,
                    ..MemcpyOp::default()
                },
                s,
            )
            .unwrap();
        assert!(op.0 >= 1);
        assert!(sim.query_stream(d, s).unwrap());
        assert!(sim.is_resident(a, d).unwrap());
    }

    #[test]
    fn memset_sync_is_cuda_memset() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 4096u64;
        let a = sim.malloc(d, bytes).unwrap();
        let op = sim.memset_sync(d, a, bytes, s).unwrap();
        assert!(op.0 >= 1);
        assert!(sim.query_stream(d, s).unwrap());
        let pitched = sim
            .memset_op_sync(
                d,
                MemsetOp {
                    id: a,
                    offset: 0,
                    bytes: 256,
                    height: 8,
                    pitch: 512,
                    ..MemsetOp::default()
                },
                s,
            )
            .unwrap();
        assert!(pitched.0 >= 1);
        assert!(sim.query_stream(d, s).unwrap());
        sim.begin_capture(d, s).unwrap();
        match sim.memset_sync(d, a, bytes, s) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.memset_op_sync(d, MemsetOp::from(KernelBuf::whole(a)), s) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn pointer_get_attributes_classifies_host_device_managed() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let dev = sim.pointer_get_attributes(a).unwrap();
        assert_eq!(dev.kind, MemoryType::Device);
        assert_eq!(dev.device, Some(d));
        assert!(dev.device_pointer);
        assert!(!dev.host_pointer);
        sim.free_sync(a).unwrap();
        let gone = sim.pointer_get_attributes(a).unwrap();
        assert_eq!(gone.kind, MemoryType::Unregistered);
        let pin = sim.alloc_host_pinned(64).unwrap();
        let host = sim.pointer_get_attributes(pin).unwrap();
        assert_eq!(host.kind, MemoryType::Host);
        assert!(!host.device_pointer);
        assert!(host.host_pointer);
        let mapped = sim.alloc_host_mapped(64).unwrap();
        let m = sim.pointer_get_attributes(mapped).unwrap();
        assert_eq!(m.kind, MemoryType::Host);
        assert!(m.device_pointer);
        assert!(m.host_pointer);
        let um = sim.alloc_managed(64).unwrap();
        let u = sim.pointer_get_attributes(um).unwrap();
        assert_eq!(u.kind, MemoryType::Managed);
        assert!(u.device_pointer);
        assert!(u.host_pointer);
        match sim.pointer_get_attributes(AllocId(u64::MAX)) {
            Err(SimError::UnknownAlloc { .. }) => {}
            other => panic!("{other:?}"),
        }
        let live = sim.malloc(d, 4096).unwrap();
        assert_eq!(sim.mem_get_address_range(live).unwrap(), (live, 4096));
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(sim.mem_get_address_range(live).unwrap(), (live, 4096));
        let _cap = sim.end_capture().unwrap();
        sim.free_sync(live).unwrap();
        match sim.mem_get_address_range(live) {
            Err(SimError::Invalid { why }) => assert!(why.contains("address range"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.mem_get_address_range(AllocId(u64::MAX)) {
            Err(SimError::UnknownAlloc { .. }) => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn pointer_get_attribute_wraps_type_mapped_pool_and_range() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let a = sim.malloc(d, 4096).unwrap();
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::MemoryType)
                .unwrap(),
            MemoryType::Device.to_cuda()
        );
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::DevicePointer)
                .unwrap(),
            1
        );
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::HostPointer)
                .unwrap(),
            0
        );
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::IsManaged)
                .unwrap(),
            0
        );
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::RangeSize)
                .unwrap(),
            4096
        );
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::Mapped).unwrap(),
            0
        );
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::MemPoolHandle)
                .unwrap(),
            0
        );
        let async_a = sim.alloc(d, 256, StreamId(0)).unwrap();
        sim.synchronize_stream(d, StreamId(0)).unwrap();
        let pool = sim.default_pool(d).unwrap();
        assert_eq!(
            sim.pointer_get_attribute(async_a, PointerAttr::MemPoolHandle)
                .unwrap(),
            u64::from(pool.0)
        );
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::DeviceOrdinal)
                .unwrap(),
            u64::from(d.0)
        );
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::RangeStartAddr)
                .unwrap(),
            a.0
        );
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::BufferId).unwrap(),
            a.0
        );
        let pin = sim.alloc_host_pinned(64).unwrap();
        match sim.pointer_get_attribute(pin, PointerAttr::DeviceOrdinal) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pointer attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        let mut dual = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d1 = DeviceId(1);
        let b = dual.malloc(d1, 8).unwrap();
        assert_eq!(
            dual.pointer_get_attribute(b, PointerAttr::DeviceOrdinal)
                .unwrap(),
            u64::from(d1.0)
        );
        let mapped = sim.alloc_host_mapped(64).unwrap();
        assert_eq!(
            sim.pointer_get_attribute(mapped, PointerAttr::MemoryType)
                .unwrap(),
            MemoryType::Host.to_cuda()
        );
        assert_eq!(
            sim.pointer_get_attribute(mapped, PointerAttr::Mapped)
                .unwrap(),
            1
        );
        assert_eq!(
            sim.pointer_get_attribute(mapped, PointerAttr::DevicePointer)
                .unwrap(),
            1
        );
        assert_eq!(
            sim.pointer_get_attribute(mapped, PointerAttr::HostPointer)
                .unwrap(),
            1
        );
        let um = sim.alloc_managed(64).unwrap();
        assert_eq!(
            sim.pointer_get_attribute(um, PointerAttr::IsManaged)
                .unwrap(),
            1
        );
        assert_eq!(
            sim.pointer_get_attribute(um, PointerAttr::MemoryType)
                .unwrap(),
            MemoryType::Managed.to_cuda()
        );
        match sim.pointer_set_attribute(a, PointerAttr::RangeSize, 1) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pointer attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::RangeSize)
                .unwrap(),
            4096
        );
        match sim.pointer_set_attribute(a, PointerAttr::MemoryType, 2) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pointer attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn pointer_get_attribute_wraps_ipc_rdma_and_handle_types() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 64).unwrap();
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::IsLegacyCudaIpcCapable)
                .unwrap(),
            1
        );
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::IsGpuDirectRdmaCapable)
                .unwrap(),
            0
        );
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::AllowedHandleTypes)
                .unwrap(),
            MemHandleType::NONE
        );
        let pin = sim.alloc_host_pinned(64).unwrap();
        assert_eq!(
            sim.pointer_get_attribute(pin, PointerAttr::IsLegacyCudaIpcCapable)
                .unwrap(),
            0
        );
        assert_eq!(
            sim.pointer_get_attribute(pin, PointerAttr::IsGpuDirectRdmaCapable)
                .unwrap(),
            0
        );
        let async_a = sim.alloc(d, 64, s).unwrap();
        sim.synchronize_stream(d, s).unwrap();
        assert_eq!(
            sim.pointer_get_attribute(async_a, PointerAttr::IsLegacyCudaIpcCapable)
                .unwrap(),
            0
        );
        assert_eq!(
            sim.pointer_get_attribute(async_a, PointerAttr::AllowedHandleTypes)
                .unwrap(),
            MemHandleType::NONE
        );
        let p = sim.create_shareable_pool(d).unwrap();
        let sp = sim.alloc_from_pool(d, p, 64, s).unwrap();
        sim.synchronize_stream(d, s).unwrap();
        assert_eq!(
            sim.pointer_get_attribute(sp, PointerAttr::AllowedHandleTypes)
                .unwrap(),
            MemHandleType::POSIX_FILE_DESCRIPTOR
        );
        assert_eq!(
            sim.pointer_get_attribute(sp, PointerAttr::IsLegacyCudaIpcCapable)
                .unwrap(),
            0
        );
        let mut rdma = Sim::new(HardwareProfile::example_2node_rdma());
        let r = rdma.malloc(DeviceId(0), 64).unwrap();
        assert_eq!(
            rdma.pointer_get_attribute(r, PointerAttr::IsGpuDirectRdmaCapable)
                .unwrap(),
            1
        );
        assert_eq!(
            rdma.pointer_get_attribute(r, PointerAttr::IsLegacyCudaIpcCapable)
                .unwrap(),
            1
        );
        match sim.pointer_set_attribute(a, PointerAttr::IsLegacyCudaIpcCapable, 1) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pointer attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::IsLegacyCudaIpcCapable)
                .unwrap(),
            1
        );
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn pointer_get_attribute_wraps_vmm_mapping_base_and_size() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let a = sim.malloc(d, 4096).unwrap();
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::MappingBaseAddr)
                .unwrap(),
            a.0
        );
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::MappingSize)
                .unwrap(),
            4096
        );
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::RangeStartAddr)
                .unwrap(),
            a.0
        );
        match sim.pointer_set_attribute(a, PointerAttr::MappingSize, 1) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pointer attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        let va = sim.va_reserve(16_384).unwrap();
        assert_eq!(
            sim.pointer_get_attribute(va, PointerAttr::RangeSize)
                .unwrap(),
            16_384
        );
        match sim.pointer_get_attribute(va, PointerAttr::MappingSize) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pointer attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.pointer_get_attribute(va, PointerAttr::MappingBaseAddr) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pointer attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_map_range(va, d, 4096, 4096).unwrap();
        match sim.pointer_get_attribute(va, PointerAttr::MappingSize) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pointer attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_map_range(va, d, 0, 4096).unwrap();
        assert_eq!(
            sim.pointer_get_attribute(va, PointerAttr::MappingBaseAddr)
                .unwrap(),
            va.0
        );
        assert_eq!(
            sim.pointer_get_attribute(va, PointerAttr::MappingSize)
                .unwrap(),
            4096
        );
        assert_eq!(
            sim.pointer_get_attribute(va, PointerAttr::RangeSize)
                .unwrap(),
            16_384
        );
        sim.va_unmap(va).unwrap();
        match sim.pointer_get_attribute(va, PointerAttr::MappingSize) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pointer attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_free(va).unwrap();
        let whole = sim.va_reserve(8192).unwrap();
        sim.va_map(whole, d).unwrap();
        assert_eq!(
            sim.pointer_get_attribute(whole, PointerAttr::MappingSize)
                .unwrap(),
            8192
        );
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(
            sim.pointer_get_attribute(whole, PointerAttr::MappingBaseAddr)
                .unwrap(),
            whole.0
        );
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn pointer_get_attribute_hw_decompress_is_unsupported() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let a = sim.malloc(d, 64).unwrap();
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::IsHwDecompressCapable)
                .unwrap(),
            0
        );
        match sim.pointer_set_attribute(a, PointerAttr::IsHwDecompressCapable, 1) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pointer attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::IsHwDecompressCapable)
                .unwrap(),
            0
        );
        let _g = sim.end_capture().unwrap();
        sim.free_sync(a).unwrap();
    }

    #[test]
    fn pointer_get_attribute_wraps_vmm_memory_block_id() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let a = sim.malloc(d, 4096).unwrap();
        match sim.pointer_get_attribute(a, PointerAttr::MemoryBlockId) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pointer attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.pointer_set_attribute(a, PointerAttr::MemoryBlockId, 1) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pointer attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        let va = sim.va_reserve(16_384).unwrap();
        match sim.pointer_get_attribute(va, PointerAttr::MemoryBlockId) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pointer attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_map_range(va, d, 4096, 4096).unwrap();
        match sim.pointer_get_attribute(va, PointerAttr::MemoryBlockId) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pointer attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_map_range(va, d, 0, 4096).unwrap();
        match sim.pointer_get_attribute(va, PointerAttr::MemoryBlockId) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pointer attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        let h = sim.va_retain_handle(va, d, 0).unwrap();
        assert_eq!(
            sim.pointer_get_attribute(va, PointerAttr::MemoryBlockId)
                .unwrap(),
            h.0
        );
        let created = sim.va_create(d, 4096).unwrap();
        let mapped = sim.va_reserve(4096).unwrap();
        sim.va_map_handle(mapped, d, 0, created).unwrap();
        assert_eq!(
            sim.pointer_get_attribute(mapped, PointerAttr::MemoryBlockId)
                .unwrap(),
            created.0
        );
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(
            sim.pointer_get_attribute(mapped, PointerAttr::MemoryBlockId)
                .unwrap(),
            created.0
        );
        let _g = sim.end_capture().unwrap();
        sim.va_unmap(va).unwrap();
        sim.va_unmap(mapped).unwrap();
        sim.va_free(va).unwrap();
        sim.va_free(mapped).unwrap();
        sim.va_release_handle(h).unwrap();
        sim.va_release_handle(created).unwrap();
        sim.free_sync(a).unwrap();
    }

    #[test]
    fn host_get_flags_reports_mapped_bit() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let pin = sim.alloc_host_pinned(64).unwrap();
        assert_eq!(sim.host_get_flags(pin).unwrap(), HostAllocFlags::DEFAULT);
        let mapped = sim.alloc_host_mapped(64).unwrap();
        assert_eq!(sim.host_get_flags(mapped).unwrap(), HostAllocFlags::MAPPED);
        let pageable = sim.alloc_host(64).unwrap();
        match sim.host_get_flags(pageable) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not host alloc"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.host_register(pageable).unwrap();
        assert_eq!(
            sim.host_get_flags(pageable).unwrap(),
            HostAllocFlags::DEFAULT
        );
        sim.host_unregister(pageable).unwrap();
        sim.host_register_mapped(pageable).unwrap();
        assert_eq!(
            sim.host_get_flags(pageable).unwrap(),
            HostAllocFlags::MAPPED
        );
        let a = sim.malloc(d, 64).unwrap();
        match sim.host_get_flags(a) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not host alloc"), "{why}"),
            other => panic!("{other:?}"),
        }
        let um = sim.alloc_managed(64).unwrap();
        match sim.host_get_flags(um) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not host alloc"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(sim.host_get_flags(mapped).unwrap(), HostAllocFlags::MAPPED);
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn host_alloc_and_register_with_flags() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let pin = sim
            .alloc_host_with_flags(64, HostAllocFlags::DEFAULT)
            .unwrap();
        assert_eq!(sim.host_get_flags(pin).unwrap(), HostAllocFlags::DEFAULT);
        let mapped = sim
            .alloc_host_with_flags(64, HostAllocFlags::MAPPED)
            .unwrap();
        assert_eq!(sim.host_get_flags(mapped).unwrap(), HostAllocFlags::MAPPED);
        let portable = sim
            .alloc_host_with_flags(64, HostAllocFlags::PORTABLE)
            .unwrap();
        assert_eq!(
            sim.host_get_flags(portable).unwrap(),
            HostAllocFlags::PORTABLE
        );
        let wc = sim
            .alloc_host_with_flags(64, HostAllocFlags::WRITE_COMBINED)
            .unwrap();
        assert_eq!(
            sim.host_get_flags(wc).unwrap(),
            HostAllocFlags::WRITE_COMBINED
        );
        match sim.alloc_host_with_flags(64, 8) {
            Err(SimError::Invalid { why }) => assert!(why.contains("host alloc flags"), "{why}"),
            other => panic!("{other:?}"),
        }
        let pageable = sim.alloc_host(64).unwrap();
        sim.host_register_with_flags(pageable, HostAllocFlags::DEFAULT)
            .unwrap();
        assert_eq!(
            sim.host_get_flags(pageable).unwrap(),
            HostAllocFlags::DEFAULT
        );
        sim.host_unregister(pageable).unwrap();
        sim.host_register_with_flags(pageable, HostAllocFlags::MAPPED)
            .unwrap();
        assert_eq!(
            sim.host_get_flags(pageable).unwrap(),
            HostAllocFlags::MAPPED
        );
        sim.host_unregister(pageable).unwrap();
        sim.host_register_with_flags(pageable, HostAllocFlags::PORTABLE)
            .unwrap();
        assert_eq!(
            sim.host_get_flags(pageable).unwrap(),
            HostAllocFlags::PORTABLE
        );
        sim.host_unregister(pageable).unwrap();
        match sim.host_register_with_flags(pageable, 8) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("host register flags"), "{why}")
            }
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, StreamId(0)).unwrap();
        match sim.alloc_host_with_flags(64, 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn host_get_device_pointer_and_device_get_attribute() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let mapped = sim.alloc_host_mapped(64).unwrap();
        assert_eq!(sim.host_get_device_pointer(mapped).unwrap(), mapped);
        assert_eq!(
            sim.host_get_device_pointer_with_flags(mapped, HostGetDevicePointerFlags::DEFAULT)
                .unwrap(),
            mapped
        );
        match sim.host_get_device_pointer_with_flags(mapped, 1) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("host get device pointer flags"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        let pin = sim.alloc_host_pinned(64).unwrap();
        match sim.host_get_device_pointer(pin) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not mapped"), "{why}"),
            other => panic!("{other:?}"),
        }
        let a = sim.malloc(d, 64).unwrap();
        match sim.host_get_device_pointer(a) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not mapped"), "{why}"),
            other => panic!("{other:?}"),
        }
        let gpu = sim.profile().gpu(d).unwrap();
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::CooperativeLaunch)
                .unwrap(),
            u64::from(gpu.cooperative_launch)
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::ConcurrentKernels)
                .unwrap(),
            0,
            "example H100 compute_slots is 1"
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::L2CacheSize)
                .unwrap(),
            gpu.l2_bytes
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::MaxPersistingL2CacheSize)
                .unwrap(),
            gpu.l2_bytes
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::MemoryPoolsSupported)
                .unwrap(),
            1
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::MaxSharedMemoryPerBlock)
                .unwrap(),
            u64::from(gpu.max_shared_mem_per_block)
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::TotalGlobalMem)
                .unwrap(),
            gpu.hbm_bytes
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::AsyncEngineCount)
                .unwrap(),
            u64::from(gpu.copy_engines)
        );
        let props = sim.device_get_properties(d).unwrap();
        assert_eq!(props.name, "example-h100-sxm");
        assert_eq!(props.total_global_mem, gpu.hbm_bytes);
        assert_eq!(props.shared_mem_per_block, gpu.max_shared_mem_per_block);
        assert_eq!(
            props.shared_mem_per_block_optin,
            gpu.max_shared_mem_per_block_optin
        );
        assert_eq!(props.l2_cache_size, gpu.l2_bytes);
        assert_eq!(props.async_engine_count, u32::from(gpu.copy_engines));
        assert!(!props.concurrent_kernels);
        assert_eq!(props.cooperative_launch, gpu.cooperative_launch);
        assert_eq!(
            props.max_blocks_per_cluster,
            u32::from(gpu.max_blocks_per_cluster)
        );
        assert_eq!(
            props.portable_cluster_size,
            u32::from(gpu.portable_cluster_size)
        );
        assert_eq!(
            props.mem_sync_domain_count,
            u32::from(gpu.mem_sync_domain_count)
        );
        assert!(props.memory_pools_supported);
        assert!(props.can_map_host_memory);
        assert!(props.managed_memory);
        assert_eq!(props.cluster_launch, gpu.max_blocks_per_cluster > 0);
        assert!(props.host_register_supported);
        assert!(props.ipc_event_support);
        assert!(props.can_use_host_pointer_for_registered_mem);
        assert_eq!(
            props.memory_pool_supported_handle_types,
            MemHandleType::POSIX_FILE_DESCRIPTOR
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::CanMapHostMemory)
                .unwrap(),
            1
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::ManagedMemory)
                .unwrap(),
            1
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::ClusterLaunch)
                .unwrap(),
            u64::from(gpu.max_blocks_per_cluster > 0)
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::HostRegisterSupported)
                .unwrap(),
            1
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::IpcEventSupport)
                .unwrap(),
            1
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::CanUseHostPointerForRegisteredMem)
                .unwrap(),
            1
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::MemoryPoolSupportedHandleTypes)
                .unwrap(),
            MemHandleType::POSIX_FILE_DESCRIPTOR
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::GpuDirectRdmaSupported)
                .unwrap(),
            0
        );
        assert!(!props.gpu_direct_rdma_supported);
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::HostRegisterReadOnlySupported)
                .unwrap(),
            0
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::PageableMemoryAccess)
                .unwrap(),
            0
        );
        assert!(!props.host_register_read_only_supported);
        assert!(!props.pageable_memory_access);
        assert!(props.stream_priorities_supported);
        assert!(props.gpu_overlap);
        assert!(props.unified_addressing);
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::StreamPrioritiesSupported)
                .unwrap(),
            1
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::GpuOverlap).unwrap(),
            1
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::UnifiedAddressing)
                .unwrap(),
            1
        );
        let no_copy = HardwareProfile::parse("gpus=1\ncopy_engines=0\n").unwrap();
        let mut overlap = Sim::new(no_copy);
        assert_eq!(
            overlap
                .device_get_attribute(d, DeviceAttr::GpuOverlap)
                .unwrap(),
            0
        );
        assert!(!overlap.device_get_properties(d).unwrap().gpu_overlap);
        overlap.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(
            overlap
                .device_get_attribute(d, DeviceAttr::UnifiedAddressing)
                .unwrap(),
            1
        );
        let _cap = overlap.end_capture().unwrap();
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(
            sim.host_get_device_pointer_with_flags(mapped, 0).unwrap(),
            mapped
        );
        let _g = sim.end_capture().unwrap();
        match sim.device_get_properties(DeviceId(1)) {
            Err(SimError::Invalid { why }) => assert!(why.contains("device"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn stream_get_flags_and_priority_wrap_create_state() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        assert_eq!(sim.stream_get_flags(d, StreamId::NULL).unwrap(), 1);
        assert_eq!(sim.stream_get_priority(d, StreamId::NULL).unwrap(), 0);
        sim.set_legacy_null_stream(true);
        assert_eq!(sim.stream_get_flags(d, StreamId::NULL).unwrap(), 0);
        sim.set_stream_blocking(d, StreamId(1), true).unwrap();
        assert_eq!(sim.stream_get_flags(d, StreamId(1)).unwrap(), 0);
        sim.set_stream_blocking(d, StreamId(1), false).unwrap();
        assert_eq!(sim.stream_get_flags(d, StreamId(1)).unwrap(), 1);
        sim.set_stream_priority(d, StreamId(2), 7).unwrap();
        assert_eq!(sim.stream_get_priority(d, StreamId(2)).unwrap(), 7);
        match sim.stream_get_flags(DeviceId(1), StreamId(0)) {
            Err(SimError::Invalid { why }) => assert!(why.contains("device"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(sim.stream_get_flags(d, StreamId(1)).unwrap(), 1);
        assert_eq!(sim.device_get_properties(d).unwrap().async_engine_count, 2);
    }

    #[test]
    fn stream_create_with_flags_and_priority() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(1);
        sim.stream_create_with_flags(d, s, StreamCreateFlags::DEFAULT)
            .unwrap();
        assert!(sim.stream_is_blocking(d, s));
        assert_eq!(sim.stream_get_flags(d, s).unwrap(), 0);
        sim.stream_create_with_flags(d, s, StreamCreateFlags::NON_BLOCKING)
            .unwrap();
        assert!(!sim.stream_is_blocking(d, s));
        assert_eq!(sim.stream_get_flags(d, s).unwrap(), 1);
        sim.stream_create_with_priority(d, s, StreamCreateFlags::DEFAULT, 7)
            .unwrap();
        assert!(sim.stream_is_blocking(d, s));
        assert_eq!(sim.stream_get_flags(d, s).unwrap(), 0);
        assert_eq!(sim.stream_get_priority(d, s).unwrap(), 7);
        match sim.stream_create_with_flags(d, s, 2) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("stream create flags"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        match sim.stream_create_with_flags(d, StreamId::NULL, 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("null stream"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, StreamId(0)).unwrap();
        match sim.stream_create_with_flags(d, s, 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.stream_create_with_priority(d, s, 0, 1) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        sim.set_stream_blocking(d, s, true).unwrap();
        assert!(sim.stream_is_blocking(d, s));
    }

    #[test]
    fn stream_get_id_is_unique_per_device_stream() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let null = sim.stream_get_id(d, StreamId::NULL).unwrap();
        let created = sim.stream_get_id(d, StreamId(1)).unwrap();
        assert_ne!(null, created);
        assert_eq!(null, sim.stream_get_id(d, StreamId::NULL).unwrap());
        match sim.stream_get_id(DeviceId(1), StreamId(0)) {
            Err(SimError::Invalid { why }) => assert!(why.contains("device"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(sim.stream_get_id(d, StreamId(1)).unwrap(), created);
        let _g = sim.end_capture().unwrap();
        let mut eight = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let gpu0 = eight.stream_get_id(DeviceId(0), StreamId::NULL).unwrap();
        let gpu1 = eight.stream_get_id(DeviceId(1), StreamId::NULL).unwrap();
        assert_ne!(gpu0, gpu1);
        eight.begin_capture(DeviceId(1), StreamId(0)).unwrap();
        assert_eq!(
            eight.stream_get_id(DeviceId(1), StreamId::NULL).unwrap(),
            gpu1
        );
        let _g = eight.end_capture().unwrap();
    }

    #[test]
    fn stream_get_set_attribute_wraps_existing_state() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(1);
        assert_eq!(
            sim.stream_get_attribute(d, s, StreamAttr::Priority)
                .unwrap(),
            StreamAttrValue::Priority(0)
        );
        sim.stream_set_attribute(d, s, StreamAttr::Priority, StreamAttrValue::Priority(7))
            .unwrap();
        assert_eq!(sim.stream_get_priority(d, s).unwrap(), 7);
        assert_eq!(
            sim.stream_get_attribute(d, s, StreamAttr::Priority)
                .unwrap(),
            StreamAttrValue::Priority(7)
        );
        sim.stream_set_attribute(
            d,
            s,
            StreamAttr::SynchronizationPolicy,
            StreamAttrValue::SynchronizationPolicy(SynchronizationPolicy::BlockingSync),
        )
        .unwrap();
        assert_eq!(
            sim.stream_sync_policy(d, s),
            SynchronizationPolicy::BlockingSync
        );
        sim.stream_set_attribute(
            d,
            s,
            StreamAttr::MemSyncDomain,
            StreamAttrValue::MemSyncDomain(MemSyncDomain::Remote),
        )
        .unwrap();
        assert_eq!(sim.stream_mem_sync_domain(d, s), MemSyncDomain::Remote);
        let map = MemSyncDomainMap {
            default: 1,
            remote: 0,
        };
        sim.stream_set_attribute(
            d,
            s,
            StreamAttr::MemSyncDomainMap,
            StreamAttrValue::MemSyncDomainMap(map),
        )
        .unwrap();
        assert_eq!(sim.stream_mem_sync_domain_map(d, s).unwrap(), map);
        sim.stream_set_attribute(
            d,
            s,
            StreamAttr::NvlinkUtilCentric,
            StreamAttrValue::NvlinkUtilCentric(true),
        )
        .unwrap();
        assert!(sim.stream_nvlink_util_centric(d, s));
        match sim.stream_set_attribute(
            d,
            s,
            StreamAttr::Priority,
            StreamAttrValue::NvlinkUtilCentric(false),
        ) {
            Err(SimError::Invalid { why }) => assert!(why.contains("stream attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.stream_get_attribute(DeviceId(1), s, StreamAttr::Priority) {
            Err(SimError::Invalid { why }) => assert!(why.contains("device"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(
            sim.stream_get_attribute(d, s, StreamAttr::Priority)
                .unwrap(),
            StreamAttrValue::Priority(7)
        );
        sim.stream_set_attribute(d, s, StreamAttr::Priority, StreamAttrValue::Priority(1))
            .unwrap();
        assert_eq!(sim.stream_get_priority(d, s).unwrap(), 1);
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn graph_nodes_are_creation_order_and_capture_legal() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        match sim.graph_nodes(GraphId(99)) {
            Err(SimError::Invalid { why }) => assert!(why.contains("unknown graph"), "{why}"),
            other => panic!("{other:?}"),
        }
        let g = sim.create_graph(d, s).unwrap();
        assert!(sim.graph_nodes(g).unwrap().is_empty());
        let a = sim.malloc(d, 4096).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_empty(g).unwrap();
        assert_eq!(sim.graph_nodes(g).unwrap(), vec![0, 1]);
        assert_eq!(sim.graph_root_nodes(g).unwrap(), vec![0, 1]);
        sim.begin_capture_to_graph(d, s, g, &[]).unwrap();
        assert_eq!(sim.graph_nodes(g).unwrap(), vec![0, 1]);
        let _end = sim.end_capture().unwrap();
        assert_eq!(sim.graph_nodes(g).unwrap(), vec![0, 1]);
    }

    #[test]
    fn graph_node_getters_wrap_child_event_and_alloc() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let leaf = sim.create_graph(d, s).unwrap();
        sim.graph_add_empty(leaf).unwrap();
        let _ = sim.instantiate_graph(leaf).unwrap();
        let parent = sim.create_graph(d, s).unwrap();
        sim.graph_add_child(parent, leaf).unwrap();
        assert_eq!(sim.graph_child_get_graph(parent, 0).unwrap(), leaf);
        match sim.graph_child_get_graph(leaf, 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("child graph"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.create_event(EventId(1)).unwrap();
        sim.create_event(EventId(2)).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_event_record(g, EventId(1), false).unwrap();
        sim.graph_add_event_wait(g, EventId(2), false).unwrap();
        assert_eq!(sim.graph_event_record_get_event(g, 0).unwrap(), EventId(1));
        assert_eq!(sim.graph_event_wait_get_event(g, 1).unwrap(), EventId(2));
        match sim.graph_event_record_get_event(g, 1) {
            Err(SimError::Invalid { why }) => assert!(why.contains("event record"), "{why}"),
            other => panic!("{other:?}"),
        }
        let mem = sim.create_graph(d, s).unwrap();
        let id = sim.graph_add_alloc(mem, 4096).unwrap();
        assert_eq!(sim.graph_alloc_get_params(mem, 0).unwrap(), (id, 4096));
        match sim.graph_alloc_get_params(g, 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("mem alloc"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        assert_eq!(sim.graph_child_get_graph(parent, 0).unwrap(), leaf);
        assert_eq!(sim.graph_event_record_get_event(g, 0).unwrap(), EventId(1));
        let _end = sim.end_capture().unwrap();
        match sim.graph_child_get_graph(GraphId(99), 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("unknown graph"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_free_get_set_params_wraps_stored_id() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_free(g, a).unwrap();
        assert_eq!(sim.graph_free_get_params(g, 0).unwrap(), a);
        sim.graph_free_set_params(g, 0, b).unwrap();
        assert_eq!(sim.graph_free_get_params(g, 0).unwrap(), b);
        match sim.graph_alloc_get_params(g, 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("mem alloc"), "{why}"),
            other => panic!("{other:?}"),
        }
        let kern = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(kern, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        match sim.graph_free_get_params(kern, 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("mem free"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.graph_free_set_params(kern, 0, b) {
            Err(SimError::Invalid { why }) => assert!(why.contains("mem free"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.graph_free_set_params(g, 0, AllocId(99)) {
            Err(SimError::UnknownAlloc { alloc }) => assert_eq!(alloc, AllocId(99)),
            other => panic!("{other:?}"),
        }
        let _ = sim.instantiate_graph(g).unwrap();
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert!(!sim.is_resident(b, d).unwrap());
        sim.begin_capture(d, s).unwrap();
        assert_eq!(sim.graph_free_get_params(g, 0).unwrap(), b);
        match sim.graph_free_set_params(g, 0, b) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _end = sim.end_capture().unwrap();
    }

    #[test]
    fn graph_exec_free_set_params_retargets_exec_not_definition() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_free(g, a).unwrap();
        let err = sim.graph_exec_free_set_params(g, 0, b).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        let exec = sim.instantiate_graph(g).unwrap();
        sim.graph_free_set_params(g, 0, b).unwrap();
        assert_eq!(sim.graph_free_get_params(exec, 0).unwrap(), a);
        let t0 = sim.clock_ns();
        sim.graph_exec_free_set_params(exec, 0, b).unwrap();
        assert_eq!(
            sim.clock_ns().saturating_sub(t0),
            h100().gpu(d).unwrap().graph_set_params_ns.max(1)
        );
        assert_eq!(sim.graph_free_get_params(exec, 0).unwrap(), b);
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(a, d).unwrap());
        assert!(!sim.is_resident(b, d).unwrap());
        match sim.graph_exec_free_set_params(exec, 0, AllocId(99)) {
            Err(SimError::UnknownAlloc { alloc }) => assert_eq!(alloc, AllocId(99)),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        match sim.graph_exec_free_set_params(exec, 0, a) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _end = sim.end_capture().unwrap();
    }

    #[test]
    fn graph_debug_dot_prints_kinds_and_edges() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        assert_eq!(sim.graph_debug_dot(g).unwrap(), "digraph {\n}\n");
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_empty(g).unwrap();
        sim.graph_add_dependencies(g, 0, 1).unwrap();
        let dot = sim.graph_debug_dot(g).unwrap();
        assert!(dot.contains("n0 [label=\"0 Kernel\"]"), "{dot}");
        assert!(dot.contains("n1 [label=\"1 Empty\"]"), "{dot}");
        assert!(dot.contains("n0 -> n1;"), "{dot}");
        sim.begin_capture(d, s).unwrap();
        assert!(sim.graph_debug_dot(g).unwrap().contains("n0 -> n1;"));
        let _end = sim.end_capture().unwrap();
        match sim.graph_debug_dot(GraphId(99)) {
            Err(SimError::Invalid { why }) => assert!(why.contains("unknown graph"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_debug_dot_with_flags_dumps_kernel_params() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_memcpy(
            g,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d),
                alloc: a,
                bytes: 4096,
                ..MemcpyOp::default()
            },
        )
        .unwrap();
        sim.graph_kernel_node_set_priority(g, 0, 7).unwrap();
        let plain = sim.graph_debug_dot(g).unwrap();
        assert!(plain.contains("n0 [label=\"0 Kernel\"]"), "{plain}");
        assert!(!plain.contains("coop="), "{plain}");
        let kern = sim
            .graph_debug_dot_with_flags(g, GraphDebugDotFlags::KERNEL_NODE_PARAMS)
            .unwrap();
        assert!(kern.contains("coop=0"), "{kern}");
        assert!(kern.contains(&format!("r={}", a.0)), "{kern}");
        let attrs = sim
            .graph_debug_dot_with_flags(g, GraphDebugDotFlags::KERNEL_NODE_ATTRIBUTES)
            .unwrap();
        assert!(attrs.contains("pri=7"), "{attrs}");
        let copy = sim
            .graph_debug_dot_with_flags(g, GraphDebugDotFlags::MEMCPY_NODE_PARAMS)
            .unwrap();
        assert!(copy.contains("bytes=4096"), "{copy}");
        assert!(copy.contains("src=HostPinned"), "{copy}");
        let verbose = sim
            .graph_debug_dot_with_flags(g, GraphDebugDotFlags::VERBOSE)
            .unwrap();
        assert!(
            verbose.starts_with(&format!("digraph g{} {{", g.0)),
            "{verbose}"
        );
        assert!(verbose.contains("coop=0"), "{verbose}");
        assert!(verbose.contains("pri=7"), "{verbose}");
        assert!(verbose.contains("bytes=4096"), "{verbose}");
        match sim.graph_debug_dot_with_flags(g, 1 << 7) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("graph debug dot flags"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        assert!(sim
            .graph_debug_dot_with_flags(g, GraphDebugDotFlags::KERNEL_NODE_PARAMS)
            .unwrap()
            .contains("coop=0"));
        let _end = sim.end_capture().unwrap();
    }

    #[test]
    fn graph_event_set_event_does_not_retarget_instantiated_exec() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.create_event(EventId(1)).unwrap();
        sim.create_event(EventId(2)).unwrap();
        sim.create_event(EventId(3)).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_event_record(g, EventId(1), false).unwrap();
        sim.graph_add_event_wait(g, EventId(1), false).unwrap();
        sim.graph_event_record_set_event(g, 0, EventId(2)).unwrap();
        sim.graph_event_wait_set_event(g, 1, EventId(2)).unwrap();
        assert_eq!(sim.graph_event_record_get_event(g, 0).unwrap(), EventId(2));
        assert_eq!(sim.graph_event_wait_get_event(g, 1).unwrap(), EventId(2));
        let kern = sim.create_graph(d, s).unwrap();
        let a = sim.malloc(d, 4096).unwrap();
        sim.graph_add_kernel(kern, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        match sim.graph_event_record_set_event(kern, 0, EventId(2)) {
            Err(SimError::Invalid { why }) => assert!(why.contains("event record"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.graph_event_record_set_event(g, 0, EventId(9)) {
            Err(SimError::UnknownEvent { event: 9 }) => {}
            other => panic!("{other:?}"),
        }
        let exec = sim.instantiate_graph(g).unwrap();
        sim.graph_event_record_set_event(g, 0, EventId(3)).unwrap();
        assert_eq!(
            sim.graph_event_record_get_event(exec, 0).unwrap(),
            EventId(2)
        );
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
        assert!(sim.query_event(EventId(2)).unwrap());
        assert!(!sim.query_event(EventId(3)).unwrap());
        sim.begin_capture(d, s).unwrap();
        match sim.graph_event_record_set_event(g, 0, EventId(1)) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _end = sim.end_capture().unwrap();
    }

    #[test]
    fn graph_child_set_params_allows_topology_change_on_definition() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        let kern = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(kern, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let kern_exec = sim.instantiate_graph(kern).unwrap();
        let copy = sim.create_graph(d, s).unwrap();
        sim.graph_add_memcpy(
            copy,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d),
                alloc: b,
                bytes: 4096,
                offset: 0,
                ..MemcpyOp::default()
            },
        )
        .unwrap();
        let copy_exec = sim.instantiate_graph(copy).unwrap();
        let parent = sim.create_graph(d, s).unwrap();
        sim.graph_add_child(parent, kern).unwrap();
        sim.graph_child_set_params(parent, 0, copy).unwrap();
        assert_eq!(sim.graph_child_get_graph(parent, 0).unwrap(), copy);
        let exec = sim.instantiate_graph(parent).unwrap();
        assert_eq!(sim.graph_child_get_graph(exec, 0).unwrap(), copy_exec);
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(b, d).unwrap());
        let parent2 = sim.create_graph(d, s).unwrap();
        sim.graph_add_child(parent2, kern).unwrap();
        let exec2 = sim.instantiate_graph(parent2).unwrap();
        assert_eq!(sim.graph_child_get_graph(exec2, 0).unwrap(), kern_exec);
        sim.graph_child_set_params(parent2, 0, copy).unwrap();
        assert_eq!(sim.graph_child_get_graph(exec2, 0).unwrap(), kern_exec);
        let exec3 = sim.instantiate_graph(parent2).unwrap();
        assert_eq!(sim.graph_child_get_graph(exec3, 0).unwrap(), copy_exec);
        match sim.graph_child_set_params(kern, 0, copy) {
            Err(SimError::Invalid { why }) => assert!(why.contains("child graph"), "{why}"),
            other => panic!("{other:?}"),
        }
        let raw = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(raw, KernelKind::other(8, 8), &[b], &[b])
            .unwrap();
        match sim.graph_child_set_params(parent, 0, raw) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        match sim.graph_child_set_params(parent, 0, copy) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _end = sim.end_capture().unwrap();
    }

    #[test]
    fn graph_add_node_binds_deps_and_alloc_id() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        let empty = sim.graph_add_node(g, &[], GraphNodeParams::Empty).unwrap();
        assert_eq!(empty.node, 0);
        assert!(empty.alloc.is_none());
        let kern = sim
            .graph_add_node(
                g,
                &[0],
                GraphNodeParams::Kernel(KernelNodeParams {
                    kind: KernelKind::other(8, 8),
                    reads: vec![KernelBuf::whole(a)],
                    writes: vec![KernelBuf::whole(a)],
                    cooperative: false,
                }),
            )
            .unwrap();
        assert_eq!(kern.node, 1);
        assert_eq!(sim.graph_node_deps(g, 1).unwrap(), vec![0]);
        assert_eq!(sim.graph_edges(g).unwrap(), vec![(0, 1)]);
        assert_eq!(sim.graph_node_kind(g, 1).unwrap(), GraphNodeKind::Kernel);
        let mem = sim
            .graph_add_node(g, &[1], GraphNodeParams::Alloc { bytes: 4096 })
            .unwrap();
        assert_eq!(mem.node, 2);
        let id = mem.alloc.expect("alloc node id");
        assert_eq!(sim.graph_alloc_get_params(g, 2).unwrap(), (id, 4096));
        assert_eq!(sim.graph_node_deps(g, 2).unwrap(), vec![1]);
        match sim.graph_add_node(g, &[99], GraphNodeParams::Empty) {
            Err(SimError::Invalid { why }) => assert!(why.contains("dependency"), "{why}"),
            other => panic!("{other:?}"),
        }
        let exec = sim.instantiate_graph(g).unwrap();
        match sim.graph_add_node(exec, &[], GraphNodeParams::Empty) {
            Err(SimError::Invalid { why }) => assert!(why.contains("instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 3);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(id, d).unwrap());
        let g2 = sim.create_graph(d, s).unwrap();
        sim.begin_capture(d, s).unwrap();
        match sim.graph_add_node(g2, &[], GraphNodeParams::Empty) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _end = sim.end_capture().unwrap();
    }

    #[test]
    fn graph_node_set_params_dispatches_without_retargeting_exec() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let exec = sim.instantiate_graph(g).unwrap();
        let patched = GraphNodeParams::Kernel(KernelNodeParams {
            kind: KernelKind::other(8, 8),
            reads: vec![KernelBuf::whole(b)],
            writes: vec![KernelBuf::whole(b)],
            cooperative: false,
        });
        sim.graph_node_set_params(g, 0, patched.clone()).unwrap();
        let def = sim.graph_kernel_get_params(g, 0).unwrap();
        assert_eq!(def.reads[0].id, b);
        let snap = sim.graph_exec_kernel_get_params(exec, 0).unwrap();
        assert_eq!(snap.reads[0].id, a);
        sim.graph_exec_node_set_params(exec, 0, patched).unwrap();
        let snap = sim.graph_exec_kernel_get_params(exec, 0).unwrap();
        assert_eq!(snap.reads[0].id, b);
        match sim.graph_node_set_params(g, 0, GraphNodeParams::Empty) {
            Err(SimError::Invalid { why }) => assert!(why.contains("empty"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.graph_node_set_params(g, 0, GraphNodeParams::Alloc { bytes: 4096 }) {
            Err(SimError::Invalid { why }) => assert!(why.contains("alloc"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.graph_node_set_params(g, 0, GraphNodeParams::Host(HostNodeParams::default())) {
            Err(SimError::Invalid { why }) => assert!(why.contains("host"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        match sim.graph_node_set_params(
            g,
            0,
            GraphNodeParams::Kernel(KernelNodeParams {
                kind: KernelKind::other(8, 8),
                reads: vec![KernelBuf::whole(a)],
                writes: vec![KernelBuf::whole(a)],
                cooperative: false,
            }),
        ) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _end = sim.end_capture().unwrap();
    }

    #[test]
    fn graph_node_get_params_reads_definition_not_exec() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_empty(g).unwrap();
        assert_eq!(
            sim.graph_node_get_params(g, 1).unwrap(),
            GraphNodeParams::Empty
        );
        let exec = sim.instantiate_graph(g).unwrap();
        sim.graph_node_set_params(
            g,
            0,
            GraphNodeParams::Kernel(KernelNodeParams {
                kind: KernelKind::other(8, 8),
                reads: vec![KernelBuf::whole(b)],
                writes: vec![KernelBuf::whole(b)],
                cooperative: false,
            }),
        )
        .unwrap();
        match sim.graph_node_get_params(g, 0).unwrap() {
            GraphNodeParams::Kernel(p) => assert_eq!(p.reads[0].id, b),
            other => panic!("{other:?}"),
        }
        match sim.graph_exec_node_get_params(exec, 0).unwrap() {
            GraphNodeParams::Kernel(p) => assert_eq!(p.reads[0].id, a),
            other => panic!("{other:?}"),
        }
        let mem = sim.create_graph(d, s).unwrap();
        let added = sim
            .graph_add_node(mem, &[], GraphNodeParams::Alloc { bytes: 4096 })
            .unwrap();
        assert_eq!(
            sim.graph_node_get_params(mem, added.node).unwrap(),
            GraphNodeParams::Alloc { bytes: 4096 }
        );
        let cond = sim.create_graph(d, s).unwrap();
        let h = sim.graph_conditional_create(cond, 0).unwrap();
        let _body = sim.graph_add_if(cond, h).unwrap();
        match sim.graph_node_get_params(cond, 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("params kind"), "{why}"),
            other => panic!("{other:?}"),
        }
        let raw = sim.create_graph(d, s).unwrap();
        sim.graph_add_empty(raw).unwrap();
        match sim.graph_exec_node_get_params(raw, 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        assert_eq!(
            sim.graph_node_get_params(g, 1).unwrap(),
            GraphNodeParams::Empty
        );
        let _end = sim.end_capture().unwrap();
        match sim.graph_node_get_params(g, 9) {
            Err(SimError::Invalid { why }) => assert!(why.contains("unknown graph node"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn pool_get_set_attribute_wraps_live_cached_threshold() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let p = sim.default_pool(d).unwrap();
        assert_eq!(
            sim.pool_get_attribute(p, MemPoolAttr::ReleaseThreshold)
                .unwrap(),
            0
        );
        assert_eq!(
            sim.pool_get_attribute(p, MemPoolAttr::UsedMemCurrent)
                .unwrap(),
            0
        );
        assert_eq!(
            sim.pool_get_attribute(p, MemPoolAttr::ReservedMemCurrent)
                .unwrap(),
            0
        );
        assert_eq!(
            sim.pool_get_access(p, d).unwrap(),
            MemAccessFlags::PROT_READ_WRITE
        );
        let a = sim.alloc(d, 256, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(
            sim.pool_get_attribute(p, MemPoolAttr::UsedMemCurrent)
                .unwrap(),
            256
        );
        assert_eq!(
            sim.pool_get_attribute(p, MemPoolAttr::ReservedMemCurrent)
                .unwrap(),
            256
        );
        sim.free(d, a, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(
            sim.pool_get_attribute(p, MemPoolAttr::UsedMemCurrent)
                .unwrap(),
            0
        );
        assert_eq!(
            sim.pool_get_attribute(p, MemPoolAttr::ReservedMemCurrent)
                .unwrap(),
            0
        );
        sim.pool_set_attribute(p, MemPoolAttr::ReleaseThreshold, u64::MAX)
            .unwrap();
        assert_eq!(
            sim.pool_get_attribute(p, MemPoolAttr::ReleaseThreshold)
                .unwrap(),
            u64::MAX
        );
        let b = sim.alloc(d, 256, s).unwrap();
        sim.synchronize().unwrap();
        sim.free(d, b, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(
            sim.pool_get_attribute(p, MemPoolAttr::UsedMemCurrent)
                .unwrap(),
            0
        );
        assert_eq!(
            sim.pool_get_attribute(p, MemPoolAttr::ReservedMemCurrent)
                .unwrap(),
            256
        );
        match sim.pool_set_attribute(p, MemPoolAttr::UsedMemCurrent, 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("read-only"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.pool_set_attribute(p, MemPoolAttr::ReservedMemCurrent, 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("read-only"), "{why}"),
            other => panic!("{other:?}"),
        }
        let gp = sim.graph_pool(d).unwrap();
        match sim.pool_get_attribute(gp, MemPoolAttr::UsedMemCurrent) {
            Err(SimError::Invalid { why }) => assert!(why.contains("graph mem"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.pool_get_access(gp, d) {
            Err(SimError::Invalid { why }) => assert!(why.contains("graph mem"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.pool_get_attribute(PoolId(9999), MemPoolAttr::UsedMemCurrent) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pool"), "{why}"),
            other => panic!("{other:?}"),
        }
        let sp = sim.create_shareable_pool(d).unwrap();
        sim.set_pool_release_threshold(sp, u64::MAX).unwrap();
        let h = sim.pool_export(sp).unwrap();
        let imp = sim.pool_import(d, h).unwrap();
        let c = sim.alloc_from_pool(d, sp, 128, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(
            sim.pool_get_attribute(imp, MemPoolAttr::UsedMemCurrent)
                .unwrap(),
            128
        );
        assert_eq!(
            sim.pool_get_attribute(imp, MemPoolAttr::ReleaseThreshold)
                .unwrap(),
            u64::MAX
        );
        sim.free(d, c, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(
            sim.pool_get_attribute(imp, MemPoolAttr::ReservedMemCurrent)
                .unwrap(),
            128
        );
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(
            sim.pool_get_attribute(p, MemPoolAttr::UsedMemCurrent)
                .unwrap(),
            0
        );
        match sim.pool_set_attribute(p, MemPoolAttr::ReleaseThreshold, 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn pool_get_access_owner_and_peer() {
        let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let p = sim.default_pool(d0).unwrap();
        assert_eq!(
            sim.pool_get_access(p, d0).unwrap(),
            MemAccessFlags::PROT_READ_WRITE
        );
        assert_eq!(
            sim.pool_get_access(p, d1).unwrap(),
            MemAccessFlags::PROT_NONE
        );
        sim.pool_set_access(p, d1).unwrap();
        assert_eq!(
            sim.pool_get_access(p, d1).unwrap(),
            MemAccessFlags::PROT_READ_WRITE
        );
        sim.pool_unset_access(p, d1).unwrap();
        assert_eq!(
            sim.pool_get_access(p, d1).unwrap(),
            MemAccessFlags::PROT_NONE
        );
        match sim.pool_get_access(p, DeviceId(9)) {
            Err(SimError::Invalid { why }) => assert!(why.contains("device"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d0, StreamId(0)).unwrap();
        assert_eq!(
            sim.pool_get_access(p, d0).unwrap(),
            MemAccessFlags::PROT_READ_WRITE
        );
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn func_get_attributes_wraps_per_device_setters() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let a0 = sim.func_get_attributes(d).unwrap();
        assert_eq!(a0.max_dynamic_shared_size_bytes, 0);
        assert!(!a0.non_portable_cluster_size_allowed);
        assert_eq!(a0.preferred_shmem_carveout, SharedMemCarveout::Default);
        assert_eq!(
            a0.cluster_scheduling_policy_preference,
            ClusterSchedulingPolicy::Default
        );
        sim.set_max_dynamic_shared_memory(d, 49_152).unwrap();
        sim.set_non_portable_cluster_size_allowed(d, true).unwrap();
        sim.set_func_carveout(d, SharedMemCarveout::MaxShared)
            .unwrap();
        sim.set_func_cluster_policy(d, ClusterSchedulingPolicy::Spread)
            .unwrap();
        let a1 = sim.func_get_attributes(d).unwrap();
        assert_eq!(a1.max_dynamic_shared_size_bytes, 49_152);
        assert!(a1.non_portable_cluster_size_allowed);
        assert_eq!(a1.preferred_shmem_carveout, SharedMemCarveout::MaxShared);
        assert_eq!(
            a1.cluster_scheduling_policy_preference,
            ClusterSchedulingPolicy::Spread
        );
        match sim.func_get_attributes(DeviceId(1)) {
            Err(SimError::Invalid { why }) => assert!(why.contains("device"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(
            sim.func_get_attributes(d)
                .unwrap()
                .max_dynamic_shared_size_bytes,
            49_152
        );
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn func_set_get_attribute_dispatch() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        assert_eq!(
            sim.func_get_attribute(d, FuncAttr::MaxDynamicSharedMemorySize)
                .unwrap(),
            0
        );
        assert_eq!(
            sim.func_get_attribute(d, FuncAttr::NonPortableClusterSizeAllowed)
                .unwrap(),
            0
        );
        assert_eq!(
            sim.func_get_attribute(d, FuncAttr::PreferredSharedMemoryCarveout)
                .unwrap(),
            -1
        );
        sim.func_set_attribute(d, FuncAttr::MaxDynamicSharedMemorySize, 49_152)
            .unwrap();
        sim.func_set_attribute(d, FuncAttr::NonPortableClusterSizeAllowed, 1)
            .unwrap();
        assert_eq!(
            sim.func_get_attribute(d, FuncAttr::MaxDynamicSharedMemorySize)
                .unwrap(),
            49_152
        );
        assert_eq!(
            sim.func_get_attribute(d, FuncAttr::NonPortableClusterSizeAllowed)
                .unwrap(),
            1
        );
        let a = sim.func_get_attributes(d).unwrap();
        assert_eq!(a.max_dynamic_shared_size_bytes, 49_152);
        assert!(a.non_portable_cluster_size_allowed);
        match sim.func_set_attribute(d, FuncAttr::MaxDynamicSharedMemorySize, -1) {
            Err(SimError::Invalid { why }) => assert!(why.contains("func attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.func_set_attribute(d, FuncAttr::NonPortableClusterSizeAllowed, 2) {
            Err(SimError::Invalid { why }) => assert!(why.contains("func attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.func_get_attribute(DeviceId(1), FuncAttr::MaxDynamicSharedMemorySize) {
            Err(SimError::Invalid { why }) => assert!(why.contains("device"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(
            sim.func_get_attribute(d, FuncAttr::MaxDynamicSharedMemorySize)
                .unwrap(),
            49_152
        );
        sim.func_set_attribute(d, FuncAttr::NonPortableClusterSizeAllowed, 0)
            .unwrap();
        let _g = sim.end_capture().unwrap();
        assert!(!sim.non_portable_cluster_size_allowed(d));
    }

    #[test]
    fn destroy_pool_returns_cache_keeps_live_and_rebinds_current() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        match sim.destroy_pool(sim.default_pool(d).unwrap()) {
            Err(SimError::Invalid { why }) => assert!(why.contains("default"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.destroy_pool(sim.graph_pool(d).unwrap()) {
            Err(SimError::Invalid { why }) => assert!(why.contains("graph mem"), "{why}"),
            other => panic!("{other:?}"),
        }
        let p = sim.create_pool(d).unwrap();
        sim.pool_set_attribute(p, MemPoolAttr::ReleaseThreshold, u64::MAX)
            .unwrap();
        let a = sim.alloc_from_pool(d, p, 256, s).unwrap();
        sim.synchronize().unwrap();
        sim.free(d, a, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 256);
        sim.destroy_pool(p).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        match sim.alloc_from_pool(d, p, 256, s) {
            Err(SimError::Invalid { why }) => assert!(why.contains("destroyed"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.pool_get_attribute(p, MemPoolAttr::UsedMemCurrent) {
            Err(SimError::Invalid { why }) => assert!(why.contains("destroyed"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.destroy_pool(p) {
            Err(SimError::Invalid { why }) => assert!(why.contains("destroyed"), "{why}"),
            other => panic!("{other:?}"),
        }
        let q = sim.create_pool(d).unwrap();
        let b = sim.alloc_from_pool(d, q, 128, s).unwrap();
        sim.synchronize().unwrap();
        sim.destroy_pool(q).unwrap();
        assert!(sim.is_resident(b, d).unwrap());
        match sim.alloc_from_pool(d, q, 64, s) {
            Err(SimError::Invalid { why }) => assert!(why.contains("destroyed"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.free(d, b, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        let cur = sim.create_pool(d).unwrap();
        sim.set_device_mempool(d, cur).unwrap();
        assert_eq!(sim.device_mempool(d).unwrap(), cur);
        sim.destroy_pool(cur).unwrap();
        assert_eq!(sim.device_mempool(d).unwrap(), sim.default_pool(d).unwrap());
        let c = sim.alloc(d, 64, s).unwrap();
        sim.synchronize().unwrap();
        sim.free(d, c, s).unwrap();
        sim.synchronize().unwrap();
        let cap = sim.create_pool(d).unwrap();
        sim.begin_capture(d, s).unwrap();
        match sim.destroy_pool(cap) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn destroy_pool_shareable_import_and_exporter() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let p = sim.create_shareable_pool(d).unwrap();
        let h = sim.pool_export(p).unwrap();
        let imp = sim.pool_import(d, h).unwrap();
        let a = sim.alloc_from_pool(d, p, 64, s).unwrap();
        sim.synchronize().unwrap();
        sim.destroy_pool(imp).unwrap();
        assert!(sim.is_resident(a, d).unwrap());
        let b = sim.alloc_from_pool(d, p, 32, s).unwrap();
        sim.synchronize().unwrap();
        sim.destroy_pool(p).unwrap();
        match sim.pool_export(p) {
            Err(SimError::Invalid { why }) => assert!(why.contains("destroyed"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.pool_import(d, h) {
            Err(SimError::Invalid { why }) => assert!(
                why.contains("shareable") || why.contains("destroyed"),
                "{why}"
            ),
            other => panic!("{other:?}"),
        }
        sim.free(d, a, s).unwrap();
        sim.free(d, b, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
    }

    #[test]
    fn device_count_and_p2p_attribute_are_topology() {
        let sim = Sim::new(h100());
        assert_eq!(sim.device_count(), 1);
        assert!(!sim
            .device_can_access_peer(DeviceId(0), DeviceId(0))
            .unwrap());
        match sim.device_can_access_peer(DeviceId(0), DeviceId(1)) {
            Err(SimError::Invalid { why }) => assert!(why.contains("device"), "{why}"),
            other => panic!("{other:?}"),
        }
        let mut nv = Sim::new(HardwareProfile::example_8xh100_nvlink());
        assert_eq!(nv.device_count(), 8);
        assert!(nv.device_can_access_peer(DeviceId(0), DeviceId(1)).unwrap());
        assert_eq!(
            nv.device_get_p2p_attribute(DeviceId(0), DeviceId(1), DeviceP2pAttr::AccessSupported)
                .unwrap(),
            1
        );
        assert_eq!(
            nv.device_get_p2p_attribute(DeviceId(0), DeviceId(1), DeviceP2pAttr::PerformanceRank)
                .unwrap(),
            0
        );
        assert_eq!(
            nv.device_get_p2p_attribute(DeviceId(0), DeviceId(0), DeviceP2pAttr::PerformanceRank)
                .unwrap(),
            0
        );
        assert_eq!(
            nv.device_get_p2p_attribute(
                DeviceId(0),
                DeviceId(1),
                DeviceP2pAttr::NativeAtomicSupported
            )
            .unwrap(),
            0
        );
        assert_eq!(
            nv.device_get_p2p_attribute(
                DeviceId(0),
                DeviceId(1),
                DeviceP2pAttr::CudaArrayAccessFromDevice
            )
            .unwrap(),
            0
        );
        nv.begin_capture(DeviceId(0), StreamId(0)).unwrap();
        assert_eq!(
            nv.device_get_p2p_attribute(
                DeviceId(0),
                DeviceId(1),
                DeviceP2pAttr::NativeAtomicSupported
            )
            .unwrap(),
            0
        );
        let _g = nv.end_capture().unwrap();
        nv.disable_peer(DeviceId(0), DeviceId(1)).unwrap();
        assert!(!nv.peer_access(DeviceId(0), DeviceId(1)));
        assert!(nv.device_can_access_peer(DeviceId(0), DeviceId(1)).unwrap());
        let asym = Sim::new(HardwareProfile::example_asymmetric_links());
        assert_eq!(asym.device_count(), 3);
        assert!(asym
            .device_can_access_peer(DeviceId(0), DeviceId(1))
            .unwrap());
        assert!(!asym
            .device_can_access_peer(DeviceId(0), DeviceId(2))
            .unwrap());
        assert_eq!(
            asym.device_get_p2p_attribute(DeviceId(0), DeviceId(2), DeviceP2pAttr::PerformanceRank)
                .unwrap(),
            0
        );
        assert_eq!(
            nv.device_get_attribute(DeviceId(0), DeviceAttr::GpuDirectRdmaSupported)
                .unwrap(),
            0
        );
        assert_eq!(
            nv.device_get_attribute(DeviceId(0), DeviceAttr::CanFlushRemoteWrites)
                .unwrap(),
            0
        );
        let rdma = Sim::new(HardwareProfile::example_2node_rdma());
        assert_eq!(
            rdma.device_get_attribute(DeviceId(0), DeviceAttr::GpuDirectRdmaSupported)
                .unwrap(),
            1
        );
        assert_eq!(
            rdma.device_get_attribute(DeviceId(0), DeviceAttr::CanFlushRemoteWrites)
                .unwrap(),
            1
        );
        let rdma_props = rdma.device_get_properties(DeviceId(0)).unwrap();
        assert!(rdma_props.gpu_direct_rdma_supported);
        assert!(rdma_props.can_flush_remote_writes);
        let mut mixed = HardwareProfile::example_8xh100_nvlink();
        for l in &mut mixed.links {
            if l.connects(Some(DeviceId(0)), Some(DeviceId(1))) {
                l.bps = 16u64.saturating_mul(1_000_000_000);
            }
        }
        let mut mixed = Sim::new(mixed);
        assert_eq!(
            mixed
                .device_get_p2p_attribute(DeviceId(0), DeviceId(2), DeviceP2pAttr::PerformanceRank)
                .unwrap(),
            0
        );
        assert_eq!(
            mixed
                .device_get_p2p_attribute(DeviceId(0), DeviceId(1), DeviceP2pAttr::PerformanceRank)
                .unwrap(),
            1
        );
        mixed.begin_capture(DeviceId(0), StreamId(0)).unwrap();
        assert_eq!(
            mixed
                .device_get_p2p_attribute(DeviceId(0), DeviceId(1), DeviceP2pAttr::PerformanceRank)
                .unwrap(),
            1
        );
        let _g = mixed.end_capture().unwrap();
    }

    #[test]
    fn flush_gpu_direct_rdma_writes_is_host_sync_barrier() {
        let d = DeviceId(0);
        let mut h100 = Sim::new(h100());
        let hp = h100.device_get_properties(d).unwrap();
        assert!(!hp.concurrent_managed_access);
        assert!(!hp.direct_managed_mem_access_from_host);
        assert!(!hp.pageable_memory_access_uses_host_page_tables);
        assert!(!hp.can_flush_remote_writes);
        assert_eq!(
            h100.device_get_attribute(d, DeviceAttr::ConcurrentManagedAccess)
                .unwrap(),
            0
        );
        assert_eq!(
            h100.device_get_attribute(d, DeviceAttr::DirectManagedMemAccessFromHost)
                .unwrap(),
            0
        );
        assert_eq!(
            h100.device_get_attribute(d, DeviceAttr::PageableMemoryAccessUsesHostPageTables)
                .unwrap(),
            0
        );
        assert!(!hp.host_native_atomic_supported);
        assert!(!hp.cooperative_multi_device_launch);
        assert!(!hp.integrated);
        assert_eq!(
            h100.device_get_attribute(d, DeviceAttr::HostNativeAtomicSupported)
                .unwrap(),
            0
        );
        assert_eq!(
            h100.device_get_attribute(d, DeviceAttr::CooperativeMultiDeviceLaunch)
                .unwrap(),
            0
        );
        assert_eq!(
            h100.device_get_attribute(d, DeviceAttr::Integrated)
                .unwrap(),
            0
        );
        assert!(!hp.sparse_cuda_array_supported);
        assert!(!hp.deferred_mapping_cuda_array_supported);
        assert!(!hp.dma_buf_supported);
        assert_eq!(
            h100.device_get_attribute(d, DeviceAttr::SparseCudaArraySupported)
                .unwrap(),
            0
        );
        assert_eq!(
            h100.device_get_attribute(d, DeviceAttr::DeferredMappingCudaArraySupported)
                .unwrap(),
            0
        );
        assert_eq!(
            h100.device_get_attribute(d, DeviceAttr::DmaBufSupported)
                .unwrap(),
            0
        );
        match h100.flush_gpu_direct_rdma_writes(
            d,
            FlushGpuDirectRdmaTarget::CURRENT_DEVICE,
            FlushGpuDirectRdmaScope::TO_OWNER,
        ) {
            Err(SimError::Invalid { why }) => assert!(why.contains("gpu direct rdma"), "{why}"),
            other => panic!("{other:?}"),
        }
        let mut sim = Sim::new(HardwareProfile::example_2node_rdma());
        let t0 = sim.clock_ns();
        sim.flush_gpu_direct_rdma_writes(
            d,
            FlushGpuDirectRdmaTarget::CURRENT_DEVICE,
            FlushGpuDirectRdmaScope::TO_OWNER,
        )
        .unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(1));
        sim.flush_gpu_direct_rdma_writes(
            d,
            FlushGpuDirectRdmaTarget::CURRENT_DEVICE,
            FlushGpuDirectRdmaScope::TO_ALL_DEVICES,
        )
        .unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(2));
        match sim.flush_gpu_direct_rdma_writes(d, 1, FlushGpuDirectRdmaScope::TO_OWNER) {
            Err(SimError::Invalid { why }) => assert!(why.contains("flush rdma target"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.flush_gpu_direct_rdma_writes(d, FlushGpuDirectRdmaTarget::CURRENT_DEVICE, 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("flush rdma scope"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::CanFlushRemoteWrites)
                .unwrap(),
            1
        );
        match sim.flush_gpu_direct_rdma_writes(
            d,
            FlushGpuDirectRdmaTarget::CURRENT_DEVICE,
            FlushGpuDirectRdmaScope::TO_OWNER,
        ) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn device_get_attribute_multicast_supported() {
        let mut h100 = Sim::new(h100());
        let d = DeviceId(0);
        assert!(!h100.device_get_properties(d).unwrap().multicast_supported);
        assert_eq!(
            h100.device_get_attribute(d, DeviceAttr::MulticastSupported)
                .unwrap(),
            0
        );
        let pcie = Sim::new(HardwareProfile::example_2xh100_pcie());
        assert_eq!(
            pcie.device_get_attribute(DeviceId(0), DeviceAttr::MulticastSupported)
                .unwrap(),
            0
        );
        let rdma = Sim::new(HardwareProfile::example_2node_rdma());
        assert_eq!(
            rdma.device_get_attribute(DeviceId(0), DeviceAttr::MulticastSupported)
                .unwrap(),
            0
        );
        let mut nv = Sim::new(HardwareProfile::example_8xh100_nvlink());
        assert_eq!(
            nv.device_get_attribute(DeviceId(0), DeviceAttr::MulticastSupported)
                .unwrap(),
            1
        );
        assert_eq!(
            nv.device_get_attribute(DeviceId(7), DeviceAttr::MulticastSupported)
                .unwrap(),
            1
        );
        assert!(
            nv.device_get_properties(DeviceId(7))
                .unwrap()
                .multicast_supported
        );
        nv.begin_capture(DeviceId(0), StreamId(0)).unwrap();
        assert_eq!(
            nv.device_get_attribute(DeviceId(0), DeviceAttr::MulticastSupported)
                .unwrap(),
            1
        );
        let _g = nv.end_capture().unwrap();
        h100.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(
            h100.device_get_attribute(d, DeviceAttr::MulticastSupported)
                .unwrap(),
            0
        );
        let _h = h100.end_capture().unwrap();
    }

    #[test]
    fn device_get_attribute_virtual_memory_management_supported() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        assert!(
            sim.device_get_properties(d)
                .unwrap()
                .virtual_memory_management_supported
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::VirtualMemoryManagementSupported)
                .unwrap(),
            1
        );
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::VirtualMemoryManagementSupported)
                .unwrap(),
            1
        );
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn device_get_attribute_posix_fd_handle_type_supported() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        assert!(
            sim.device_get_properties(d)
                .unwrap()
                .handle_type_posix_file_descriptor_supported
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::HandleTypePosixFileDescriptorSupported)
                .unwrap(),
            1
        );
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::HandleTypePosixFileDescriptorSupported)
                .unwrap(),
            1
        );
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn device_get_attribute_rdma_flush_options_and_compression() {
        let mut h100 = Sim::new(h100());
        let d = DeviceId(0);
        let hp = h100.device_get_properties(d).unwrap();
        assert_eq!(hp.gpu_direct_rdma_flush_writes_options, 0);
        assert!(!hp.gpu_direct_rdma_with_cuda_vmm_supported);
        assert!(!hp.generic_compression_supported);
        assert_eq!(
            h100.device_get_attribute(d, DeviceAttr::GpuDirectRdmaFlushWritesOptions)
                .unwrap(),
            0
        );
        assert_eq!(
            h100.device_get_attribute(d, DeviceAttr::GpuDirectRdmaWithCudaVMMSupported)
                .unwrap(),
            0
        );
        assert_eq!(
            h100.device_get_attribute(d, DeviceAttr::GenericCompressionSupported)
                .unwrap(),
            0
        );
        let nv = Sim::new(HardwareProfile::example_8xh100_nvlink());
        assert_eq!(
            nv.device_get_attribute(DeviceId(0), DeviceAttr::GpuDirectRdmaFlushWritesOptions)
                .unwrap(),
            0
        );
        assert_eq!(
            nv.device_get_attribute(DeviceId(0), DeviceAttr::GpuDirectRdmaWithCudaVMMSupported)
                .unwrap(),
            0
        );
        let rdma = Sim::new(HardwareProfile::example_2node_rdma());
        assert_eq!(
            rdma.device_get_attribute(DeviceId(0), DeviceAttr::GpuDirectRdmaFlushWritesOptions)
                .unwrap(),
            FlushGpuDirectRdmaWritesOptions::HOST
        );
        assert_eq!(
            rdma.device_get_attribute(DeviceId(0), DeviceAttr::GpuDirectRdmaWithCudaVMMSupported)
                .unwrap(),
            1
        );
        assert_eq!(
            rdma.device_get_attribute(DeviceId(0), DeviceAttr::GenericCompressionSupported)
                .unwrap(),
            0
        );
        let rp = rdma.device_get_properties(DeviceId(0)).unwrap();
        assert_eq!(
            rp.gpu_direct_rdma_flush_writes_options,
            FlushGpuDirectRdmaWritesOptions::HOST
        );
        assert!(rp.gpu_direct_rdma_with_cuda_vmm_supported);
        assert!(!rp.generic_compression_supported);
        h100.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(
            h100.device_get_attribute(d, DeviceAttr::GpuDirectRdmaFlushWritesOptions)
                .unwrap(),
            0
        );
        assert_eq!(
            h100.device_get_attribute(d, DeviceAttr::GenericCompressionSupported)
                .unwrap(),
            0
        );
        let _g = h100.end_capture().unwrap();
    }

    #[test]
    fn device_get_attribute_win32_and_fabric_handle_types() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let hp = sim.device_get_properties(d).unwrap();
        assert!(!hp.handle_type_win32_handle_supported);
        assert!(!hp.handle_type_win32_kmt_handle_supported);
        assert!(!hp.handle_type_fabric_supported);
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::HandleTypeWin32HandleSupported)
                .unwrap(),
            0
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::HandleTypeWin32KmtHandleSupported)
                .unwrap(),
            0
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::HandleTypeFabricSupported)
                .unwrap(),
            0
        );
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::HandleTypeFabricSupported)
                .unwrap(),
            0
        );
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn device_get_attribute_host_memory_pools_unsupported() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let hp = sim.device_get_properties(d).unwrap();
        assert!(!hp.host_memory_pools_supported);
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::HostMemoryPoolsSupported)
                .unwrap(),
            0
        );
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::HostMemoryPoolsSupported)
                .unwrap(),
            0
        );
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn device_get_attribute_is_not_multi_gpu_board() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let hp = sim.device_get_properties(d).unwrap();
        assert!(!hp.is_multi_gpu_board);
        assert_eq!(hp.multi_gpu_board_group_id, 0);
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::IsMultiGpuBoard)
                .unwrap(),
            0
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::MultiGpuBoardGroupID)
                .unwrap(),
            0
        );
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::IsMultiGpuBoard)
                .unwrap(),
            0
        );
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::MultiGpuBoardGroupID)
                .unwrap(),
            0
        );
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn device_get_name_wraps_profile_name() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        assert_eq!(sim.device_get_name(d).unwrap(), "example-h100-sxm");
        assert_eq!(
            sim.device_get_name(d).unwrap(),
            sim.device_get_properties(d).unwrap().name
        );
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(sim.device_get_name(d).unwrap(), "example-h100-sxm");
        let _g = sim.end_capture().unwrap();
        match sim.device_get_name(DeviceId(9)) {
            Err(SimError::Invalid { .. }) => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn device_total_mem_wraps_hbm_bytes() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let hbm = sim.device_get_properties(d).unwrap().total_global_mem;
        assert_eq!(sim.device_total_mem(d).unwrap(), hbm);
        assert_eq!(
            sim.device_get_attribute(d, DeviceAttr::TotalGlobalMem)
                .unwrap(),
            hbm
        );
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(sim.device_total_mem(d).unwrap(), hbm);
        let _g = sim.end_capture().unwrap();
        match sim.device_total_mem(DeviceId(9)) {
            Err(SimError::Invalid { .. }) => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn set_limit_wraps_persisting_l2_and_fetch_granularity() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        assert_eq!(
            sim.get_limit(d, DeviceLimit::PersistingL2CacheSize)
                .unwrap(),
            0
        );
        assert_eq!(
            sim.get_limit(d, DeviceLimit::MaxL2FetchGranularity)
                .unwrap(),
            128
        );
        assert_eq!(sim.get_limit(d, DeviceLimit::StackSize).unwrap(), 1024);
        assert_eq!(
            sim.get_limit(d, DeviceLimit::MallocHeapSize).unwrap(),
            8 << 20
        );
        let cap = sim.profile().gpu(d).unwrap().l2_bytes;
        sim.set_limit(d, DeviceLimit::PersistingL2CacheSize, cap)
            .unwrap();
        assert_eq!(sim.persisting_l2_cache_size(d).unwrap(), cap);
        sim.set_limit(d, DeviceLimit::MaxL2FetchGranularity, 32)
            .unwrap();
        assert_eq!(
            sim.get_limit(d, DeviceLimit::MaxL2FetchGranularity)
                .unwrap(),
            32
        );
        match sim.set_limit(d, DeviceLimit::MaxL2FetchGranularity, 96) {
            Err(SimError::Invalid { why }) => assert!(why.contains("l2 fetch"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.set_limit(d, DeviceLimit::StackSize, 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("device limit"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn access_policy_window_must_align_to_l2_fetch() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 64).unwrap();
        let w = AccessPolicyWindow::persisting(KernelBuf::whole(a));
        match sim.kernel_access_policy(d, KernelKind::other(8, 8), &[a], &[a], s, w) {
            Err(SimError::Invalid { why }) => assert!(why.contains("L2 fetch"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.set_limit(d, DeviceLimit::MaxL2FetchGranularity, 32)
            .unwrap();
        enq(sim.kernel_access_policy(d, KernelKind::other(8, 8), &[a], &[a], s, w));
    }

    #[test]
    fn malloc_pitch_and_memcpy2d_bills_payload_not_padding() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let (a, pitch) = sim.malloc_pitch(d, 256, 8).unwrap();
        assert_eq!(pitch, 512);
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        enq(sim.memcpy(
            d,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d),
                alloc: a,
                bytes: 256,
                offset: 0,
                height: 8,
                src_pitch: 256,
                dst_pitch: pitch,
                ..MemcpyOp::default()
            },
            s,
        ));
        sim.synchronize().unwrap();
        assert_eq!(sim.bytes_moved(), 2048);
        match sim.malloc_pitch(d, 0, 8) {
            Err(SimError::Invalid { why }) => assert!(why.contains("malloc pitch"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.memcpy(
            d,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d),
                alloc: a,
                bytes: 300,
                offset: 0,
                height: 8,
                src_pitch: 256,
                dst_pitch: pitch,
                ..MemcpyOp::default()
            },
            s,
        ) {
            Err(SimError::Invalid { why }) => assert!(why.contains("memcpy2d pitch"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn malloc_pitch_and_memset2d_bills_payload_not_padding() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let (a, pitch) = sim.malloc_pitch(d, 256, 8).unwrap();
        assert_eq!(pitch, 512);
        let t0 = sim.clock_ns();
        enq(sim.memset_op(
            d,
            MemsetOp {
                id: a,
                offset: 0,
                bytes: 256,
                height: 8,
                pitch,
                ..MemsetOp::default()
            },
            s,
        ));
        sim.synchronize().unwrap();
        let pitched_ns = sim.clock_ns().saturating_sub(t0);
        let mut packed = Sim::new(h100());
        let b = packed.malloc(d, 4096).unwrap();
        let t1 = packed.clock_ns();
        enq(packed.memset(d, b, 4096, s));
        packed.synchronize().unwrap();
        let packed_ns = packed.clock_ns().saturating_sub(t1);
        assert!(
            pitched_ns < packed_ns,
            "pitched={pitched_ns} packed={packed_ns}"
        );
        assert_eq!(sim.bytes_moved(), 0);
        match sim.memset_op(
            d,
            MemsetOp {
                id: a,
                offset: 0,
                bytes: 600,
                height: 8,
                pitch,
                ..MemsetOp::default()
            },
            s,
        ) {
            Err(SimError::Invalid { why }) => assert!(why.contains("memset2d pitch"), "{why}"),
            other => panic!("{other:?}"),
        }
        let small = sim.malloc(d, 2048).unwrap();
        match sim.memset_op(
            d,
            MemsetOp {
                id: small,
                offset: 0,
                bytes: 256,
                height: 8,
                pitch: 512,
                ..MemsetOp::default()
            },
            s,
        ) {
            Err(SimError::Invalid { why }) => assert!(why.contains("memset range"), "{why}"),
            other => panic!("{other:?}"),
        }
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_memset_op(
            g,
            MemsetOp {
                id: a,
                offset: 0,
                bytes: 256,
                height: 8,
                pitch,
                ..MemsetOp::default()
            },
        )
        .unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        let got = sim.graph_memset_get_params(g, 0).unwrap();
        assert_eq!(got.height, 8);
        assert_eq!(got.pitch, pitch);
        assert_eq!(got.payload_bytes(), 2048);
        assert_eq!(got.extent_bytes(), 7 * 512 + 256);
    }

    #[test]
    fn malloc_3d_and_memcpy3d_bills_payload_not_padding() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let (a, pitch) = sim.malloc_3d(d, 256, 4, 4).unwrap();
        assert_eq!(pitch, 512);
        assert_eq!(sim.hbm_used(d).unwrap(), 8192);
        enq(sim.memcpy(
            d,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d),
                alloc: a,
                bytes: 256,
                offset: 0,
                height: 4,
                src_pitch: 256,
                dst_pitch: pitch,
                depth: 4,
                src_height: 4,
                dst_height: 4,
            },
            s,
        ));
        sim.synchronize().unwrap();
        assert_eq!(sim.bytes_moved(), 4096);
        match sim.malloc_3d(d, 256, 4, 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("malloc 3d"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.memcpy(
            d,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d),
                alloc: a,
                bytes: 600,
                offset: 0,
                height: 4,
                src_pitch: 256,
                dst_pitch: pitch,
                depth: 4,
                src_height: 4,
                dst_height: 4,
            },
            s,
        ) {
            Err(SimError::Invalid { why }) => assert!(why.contains("memcpy3d pitch"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.memcpy(
            d,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d),
                alloc: a,
                bytes: 256,
                offset: 0,
                height: 4,
                src_pitch: 256,
                dst_pitch: pitch,
                depth: 4,
                src_height: 2,
                dst_height: 4,
            },
            s,
        ) {
            Err(SimError::Invalid { why }) => assert!(why.contains("memcpy3d height"), "{why}"),
            other => panic!("{other:?}"),
        }
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_memcpy(
            g,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d),
                alloc: a,
                bytes: 256,
                offset: 0,
                height: 4,
                src_pitch: 256,
                dst_pitch: pitch,
                depth: 4,
                src_height: 4,
                dst_height: 4,
            },
        )
        .unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert_eq!(sim.bytes_moved(), 8192);
        let got = sim.graph_memcpy_get_params(g, 0).unwrap();
        assert_eq!(got.depth, 4);
        assert_eq!(got.payload_bytes(), 4096);
        assert_eq!(got.extent_bytes(), 3 * 512 * 4 + 3 * 512 + 256);
    }

    #[test]
    fn malloc_3d_and_memset3d_bills_payload_not_padding() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let (a, pitch) = sim.malloc_3d(d, 256, 4, 4).unwrap();
        assert_eq!(pitch, 512);
        let t0 = sim.clock_ns();
        enq(sim.memset_op(
            d,
            MemsetOp {
                id: a,
                offset: 0,
                bytes: 256,
                height: 4,
                pitch,
                depth: 4,
                ysize: 4,
            },
            s,
        ));
        sim.synchronize().unwrap();
        let pitched_ns = sim.clock_ns().saturating_sub(t0);
        let mut packed = Sim::new(h100());
        let b = packed.malloc(d, 8192).unwrap();
        let t1 = packed.clock_ns();
        enq(packed.memset(d, b, 8192, s));
        packed.synchronize().unwrap();
        let packed_ns = packed.clock_ns().saturating_sub(t1);
        assert!(
            pitched_ns < packed_ns,
            "pitched={pitched_ns} packed={packed_ns}"
        );
        assert_eq!(sim.bytes_moved(), 0);
        match sim.memset_op(
            d,
            MemsetOp {
                id: a,
                offset: 0,
                bytes: 600,
                height: 4,
                pitch,
                depth: 4,
                ysize: 4,
            },
            s,
        ) {
            Err(SimError::Invalid { why }) => assert!(why.contains("memset3d pitch"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.memset_op(
            d,
            MemsetOp {
                id: a,
                offset: 0,
                bytes: 256,
                height: 4,
                pitch,
                depth: 4,
                ysize: 2,
            },
            s,
        ) {
            Err(SimError::Invalid { why }) => assert!(why.contains("memset3d height"), "{why}"),
            other => panic!("{other:?}"),
        }
        let got = {
            let g = sim.create_graph(d, s).unwrap();
            sim.graph_add_memset_op(
                g,
                MemsetOp {
                    id: a,
                    offset: 0,
                    bytes: 256,
                    height: 4,
                    pitch,
                    depth: 4,
                    ysize: 4,
                },
            )
            .unwrap();
            let n = sim.launch_graph(g, s).unwrap();
            assert_eq!(n, 1);
            sim.synchronize().unwrap();
            sim.graph_memset_get_params(g, 0).unwrap()
        };
        assert_eq!(got.depth, 4);
        assert_eq!(got.payload_bytes(), 4096);
        assert_eq!(got.extent_bytes(), 3 * 512 * 4 + 3 * 512 + 256);
    }

    #[test]
    fn malloc_does_not_wait_peer_gpu() {
        let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let a1 = sim.malloc(d1, 4096).unwrap();
        enq(sim.kernel(d1, KernelKind::other(1 << 30, 4096), &[a1], &[a1], s));
        let t0 = sim.clock_ns();
        let a0 = sim.malloc(d0, 4096).unwrap();
        let malloc_ns = sim.clock_ns().saturating_sub(t0);
        assert!(sim.is_resident(a0, d0).unwrap());
        assert!(
            malloc_ns < 10_000,
            "cudaMalloc on GPU0 must not drain GPU1; ns={malloc_ns}"
        );
        assert!(!sim.query_stream(d1, s).unwrap());
        sim.synchronize_device(d1).unwrap();
        assert!(sim.query_stream(d1, s).unwrap());
        assert!(sim.clock_ns().saturating_sub(t0) > malloc_ns);
    }

    #[test]
    fn synchronize_device_waits_every_stream_on_one_gpu() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s0 = StreamId(0);
        let s1 = StreamId(1);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 20, 4096), &[a], &[a], s0));
        enq(sim.kernel(d, KernelKind::other(1 << 20, 4096), &[b], &[b], s1));
        sim.synchronize_device(d).unwrap();
        assert!(sim.query_stream(d, s0).unwrap());
        assert!(sim.query_stream(d, s1).unwrap());
    }

    #[test]
    fn host_sync_memory_cannot_be_captured() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        sim.begin_capture(d, s).unwrap();
        match sim.malloc(d, 4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.free_sync(a) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.synchronize_device(d) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.memcpy_sync(
            d,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d),
                alloc: a,
                bytes: 4096,
                offset: 0,
                ..MemcpyOp::default()
            },
            s,
        ) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.memcpy_host_to_device(d, a, 4096, s) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.create_pool(d) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        let pool = sim.default_pool(d).unwrap();
        match sim.set_pool_release_threshold(pool, u64::MAX) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.pool_trim_to(pool, 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.pool_set_access(pool, d) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.alloc_host(4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.host_register(a) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.alloc_managed(4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.va_reserve(4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.va_create(d, 4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn pageable_h2d_is_host_synchronous_pinned_is_not() {
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let mut pinned = Sim::new(h100());
        let a = pinned.alloc(d, bytes, s).unwrap();
        pinned.synchronize_stream(d, s).unwrap();
        let t0 = pinned.clock_ns();
        enq(pinned.memcpy_pinned_to_device(d, a, bytes, s));
        assert!(!pinned.query_stream(d, s).unwrap());
        assert_eq!(pinned.clock_ns(), t0);
        pinned.synchronize_stream(d, s).unwrap();
        assert!(pinned.is_resident(a, d).unwrap());

        let mut pageable = Sim::new(h100());
        let b = pageable.alloc(d, bytes, s).unwrap();
        pageable.synchronize_stream(d, s).unwrap();
        let t1 = pageable.clock_ns();
        enq(pageable.memcpy_host_to_device(d, b, bytes, s));
        assert!(pageable.query_stream(d, s).unwrap());
        assert!(pageable.clock_ns() > t1);
        assert!(pageable.is_resident(b, d).unwrap());
    }

    #[test]
    fn pointer_sync_memops_makes_memcpy_host_synchronous() {
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let mut sim = Sim::new(h100());
        let a = sim.alloc(d, bytes, s).unwrap();
        sim.synchronize_stream(d, s).unwrap();
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::SyncMemops)
                .unwrap(),
            0
        );
        let t0 = sim.clock_ns();
        enq(sim.memcpy_pinned_to_device(d, a, bytes, s));
        assert!(!sim.query_stream(d, s).unwrap());
        assert_eq!(sim.clock_ns(), t0);
        sim.synchronize_stream(d, s).unwrap();
        sim.pointer_set_attribute(a, PointerAttr::SyncMemops, 1)
            .unwrap();
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::SyncMemops)
                .unwrap(),
            1
        );
        let t1 = sim.clock_ns();
        enq(sim.memcpy_pinned_to_device(d, a, bytes, s));
        assert!(sim.query_stream(d, s).unwrap());
        assert!(sim.clock_ns() > t1);
        match sim.pointer_set_attribute(a, PointerAttr::SyncMemops, 2) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pointer attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        match sim.memcpy_pinned_to_device(d, a, bytes, s) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("sync memops memcpy"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        match sim.memcpy_host_to_device(d, a, bytes, s) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("pageable memcpy"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        match sim.memset(d, a, 64, s) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("sync memops memset"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        match sim.pointer_set_attribute(a, PointerAttr::SyncMemops, 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pointer attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(
            sim.pointer_get_attribute(a, PointerAttr::SyncMemops)
                .unwrap(),
            1
        );
        let _g = sim.end_capture().unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_memset(g, KernelBuf::whole(a)).unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        sim.free_sync(a).unwrap();
        match sim.pointer_set_attribute(a, PointerAttr::SyncMemops, 1) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pointer attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.pointer_set_attribute(AllocId(u64::MAX), PointerAttr::SyncMemops, 1) {
            Err(SimError::UnknownAlloc { .. }) => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn device_sync_memops_makes_memcpy_host_synchronous() {
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let mut sim = Sim::new(h100());
        let a = sim.alloc(d, bytes, s).unwrap();
        sim.synchronize_stream(d, s).unwrap();
        assert_eq!(sim.get_device_flags(d).unwrap(), DeviceFlags::SCHEDULE_AUTO);
        let t0 = sim.clock_ns();
        enq(sim.memcpy_pinned_to_device(d, a, bytes, s));
        assert!(!sim.query_stream(d, s).unwrap());
        assert_eq!(sim.clock_ns(), t0);
        sim.synchronize_stream(d, s).unwrap();
        sim.set_device_flags(d, DeviceFlags::SYNC_MEMOPS).unwrap();
        assert_eq!(sim.get_device_flags(d).unwrap(), DeviceFlags::SYNC_MEMOPS);
        let t1 = sim.clock_ns();
        enq(sim.memcpy_pinned_to_device(d, a, bytes, s));
        assert!(sim.query_stream(d, s).unwrap());
        assert!(sim.clock_ns() > t1);
        sim.begin_capture(d, s).unwrap();
        match sim.memcpy_pinned_to_device(d, a, bytes, s) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("sync memops memcpy"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        match sim.memcpy_host_to_device(d, a, bytes, s) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("pageable memcpy"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        match sim.memset(d, a, 64, s) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("sync memops memset"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_memset(g, KernelBuf::whole(a)).unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
    }

    #[test]
    fn pageable_h2d_cannot_share_pcie_the_way_pinned_can() {
        let d = DeviceId(0);
        let bytes = 32u64 << 20;
        let mut pinned = Sim::new(h100());
        let a = pinned.alloc(d, bytes, StreamId(0)).unwrap();
        let b = pinned.alloc(d, bytes, StreamId(1)).unwrap();
        enq(pinned.memcpy_pinned_to_device(d, a, bytes, StreamId(0)));
        enq(pinned.memcpy_pinned_to_device(d, b, bytes, StreamId(1)));
        pinned.synchronize().unwrap();
        let pin_ns = pinned.clock_ns();

        let mut pageable = Sim::new(h100());
        let c = pageable.alloc(d, bytes, StreamId(0)).unwrap();
        let e = pageable.alloc(d, bytes, StreamId(1)).unwrap();
        enq(pageable.memcpy_host_to_device(d, c, bytes, StreamId(0)));
        enq(pageable.memcpy_host_to_device(d, e, bytes, StreamId(1)));
        pageable.synchronize().unwrap();
        let page_ns = pageable.clock_ns();
        assert!(
            page_ns > pin_ns,
            "pageable cudaMemcpyAsync is host-sync so two copies cannot DMA together; pageable={page_ns} pinned={pin_ns}"
        );
    }

    #[test]
    fn pageable_d2h_is_host_synchronous() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 4096u64;
        let a = sim.malloc(d, bytes).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, bytes, s));
        sim.synchronize_stream(d, s).unwrap();
        let t0 = sim.clock_ns();
        enq(sim.memcpy_device_to_host(d, a, bytes, s));
        assert!(sim.query_stream(d, s).unwrap());
        assert!(sim.clock_ns() > t0);
        assert!(sim.is_resident(a, d).unwrap());
    }

    #[test]
    fn free_sync_rejects_host_pin() {
        let mut sim = Sim::new(h100());
        let h = sim.alloc_host_pinned(4096).unwrap();
        match sim.free_sync(h) {
            Err(SimError::UnknownAlloc { alloc }) => assert_eq!(alloc, h),
            other => panic!("{other:?}"),
        }
        sim.free_host_pinned(h).unwrap();
    }

    #[test]
    fn default_pool_threshold_zero_releases_hbm() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let p = sim.default_pool(d).unwrap();
        let a = sim.alloc(d, 1 << 20, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.pool_live(p).unwrap(), 1 << 20);
        sim.free(d, a, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        assert_eq!(sim.pool_cached(p).unwrap(), 0);
        assert_eq!(sim.pool_live(p).unwrap(), 0);
    }

    #[test]
    fn high_release_threshold_holds_hbm_until_trim() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let p = sim.default_pool(d).unwrap();
        sim.set_pool_release_threshold(p, u64::MAX).unwrap();
        let bytes = 1u64 << 20;
        let a = sim.alloc(d, bytes, s).unwrap();
        sim.synchronize().unwrap();
        let used = sim.hbm_used(d).unwrap();
        sim.free(d, a, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), used);
        assert_eq!(sim.pool_cached(p).unwrap(), bytes);
        let b = sim.alloc(d, bytes, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), used);
        assert_eq!(sim.pool_cached(p).unwrap(), 0);
        sim.free(d, b, s).unwrap();
        sim.synchronize().unwrap();
        let dropped = sim.pool_trim_to(p, 0).unwrap();
        assert_eq!(dropped, bytes);
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        assert_eq!(sim.pool_cached(p).unwrap(), 0);
    }

    #[test]
    fn malloc_cannot_consume_pool_cache_until_trim() {
        let mut sim = Sim::new(HardwareProfile::parse("gpus=1\nhbm_bytes=1048576\n").unwrap());
        let d = DeviceId(0);
        let s = StreamId(0);
        let p = sim.default_pool(d).unwrap();
        sim.set_pool_release_threshold(p, u64::MAX).unwrap();
        let a = sim.alloc(d, 1 << 20, s).unwrap();
        sim.synchronize().unwrap();
        sim.free(d, a, s).unwrap();
        sim.synchronize().unwrap();
        match sim.malloc(d, 4096) {
            Err(SimError::Oom { .. }) => {}
            other => panic!("{other:?}"),
        }
        let _n = sim.pool_trim_to(p, 0).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        assert!(sim.is_resident(b, d).unwrap());
    }

    #[test]
    fn two_pools_do_not_share_cache() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let p1 = sim.create_pool(d).unwrap();
        let p2 = sim.create_pool(d).unwrap();
        sim.set_pool_release_threshold(p1, u64::MAX).unwrap();
        sim.set_pool_release_threshold(p2, u64::MAX).unwrap();
        let a = sim.alloc_from_pool(d, p1, 4096, s).unwrap();
        sim.synchronize().unwrap();
        sim.free(d, a, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.pool_cached(p1).unwrap(), 4096);
        assert_eq!(sim.pool_cached(p2).unwrap(), 0);
        let b = sim.alloc_from_pool(d, p2, 4096, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.pool_cached(p1).unwrap(), 4096);
        assert_eq!(sim.pool_cached(p2).unwrap(), 0);
        assert!(sim.is_resident(b, d).unwrap());
    }

    #[test]
    fn shareable_import_shares_pool_cache() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let p = sim.create_shareable_pool(d).unwrap();
        sim.set_pool_release_threshold(p, u64::MAX).unwrap();
        let h = sim.pool_export(p).unwrap();
        assert_eq!(sim.pool_export(p).unwrap(), h);
        let imp = sim.pool_import(d, h).unwrap();
        assert!(sim.is_pool_shareable(p).unwrap());
        assert!(!sim.is_pool_shareable(imp).unwrap());
        assert!(sim.is_pool_imported(imp).unwrap());
        let a = sim.alloc_from_pool(d, p, 4096, s).unwrap();
        sim.synchronize().unwrap();
        sim.free(d, a, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.pool_cached(p).unwrap(), 4096);
        assert_eq!(sim.pool_cached(imp).unwrap(), 4096);
        let used0 = sim.hbm_used(d).unwrap();
        let b = sim.alloc_from_pool(d, imp, 4096, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), used0);
        assert_eq!(sim.pool_cached(p).unwrap(), 0);
        assert!(sim.is_resident(b, d).unwrap());
    }

    #[test]
    fn shareable_ptr_import_kernel_shares_hbm() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let p = sim.create_shareable_pool(d).unwrap();
        let h = sim.pool_export(p).unwrap();
        let imp = sim.pool_import(d, h).unwrap();
        let a = sim.alloc_from_pool(d, p, bytes, s).unwrap();
        sim.synchronize().unwrap();
        let e = sim.pool_export_ptr(a).unwrap();
        assert_eq!(sim.pool_export_ptr(a).unwrap(), e);
        let alias = sim.pool_import_ptr(imp, e).unwrap();
        assert!(sim.is_share_import(alias).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        enq(sim.kernel(d, KernelKind::other(1 << 20, bytes), &[alias], &[alias], s));
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        sim.free_sync(alias).unwrap();
        sim.free_sync(a).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
    }

    #[test]
    fn shareable_free_source_while_mapped_and_rejects() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let p = sim.create_shareable_pool(d).unwrap();
        let h = sim.pool_export(p).unwrap();
        let imp = sim.pool_import(d, h).unwrap();
        let a = sim.alloc_from_pool(d, p, 4096, s).unwrap();
        sim.synchronize().unwrap();
        let e = sim.pool_export_ptr(a).unwrap();
        let alias = sim.pool_import_ptr(imp, e).unwrap();
        match sim.free_sync(a) {
            Err(SimError::Invalid { why }) => assert!(why.contains("share mapped")),
            other => panic!("{other:?}"),
        }
        sim.free_sync(alias).unwrap();
        sim.free_sync(a).unwrap();
        match sim.pool_import_ptr(imp, e) {
            Err(SimError::Invalid { why }) => assert!(why.contains("freed")),
            other => panic!("{other:?}"),
        }
        match sim.pool_export(sim.default_pool(d).unwrap()) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not shareable")),
            other => panic!("{other:?}"),
        }
        let none = sim.create_pool(d).unwrap();
        match sim.pool_export(none) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not shareable")),
            other => panic!("{other:?}"),
        }
        match sim.pool_import_ptr(p, e) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not imported pool")),
            other => panic!("{other:?}"),
        }
        sim.set_device_mempool(d, p).unwrap();
        let async_id = sim.alloc(d, 4096, s).unwrap();
        sim.synchronize().unwrap();
        match sim.ipc_get(async_id) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not device ipc")),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        match sim.pool_export(p) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn shareable_set_device_mempool_redirects_alloc() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let p = sim.create_shareable_pool(d).unwrap();
        sim.set_pool_release_threshold(p, u64::MAX).unwrap();
        sim.set_device_mempool(d, p).unwrap();
        assert_eq!(sim.device_mempool(d).unwrap(), p);
        assert_ne!(sim.default_pool(d).unwrap(), p);
        let a = sim.alloc(d, 4096, s).unwrap();
        sim.synchronize().unwrap();
        let e = sim.pool_export_ptr(a).unwrap();
        let h = sim.pool_export(p).unwrap();
        let imp = sim.pool_import(d, h).unwrap();
        let alias = sim.pool_import_ptr(imp, e).unwrap();
        assert!(sim.is_share_import(alias).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
    }

    #[test]
    fn set_device_mempool_keeps_get_default_mempool() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let orig = sim.default_pool(d).unwrap();
        assert_eq!(sim.device_mempool(d).unwrap(), orig);
        let p = sim.create_pool(d).unwrap();
        sim.set_pool_release_threshold(p, u64::MAX).unwrap();
        sim.set_device_mempool(d, p).unwrap();
        assert_eq!(sim.default_pool(d).unwrap(), orig);
        assert_eq!(sim.device_mempool(d).unwrap(), p);
        sim.begin_capture(d, s).unwrap();
        assert_eq!(sim.default_pool(d).unwrap(), orig);
        assert_eq!(sim.device_mempool(d).unwrap(), p);
        let _g = sim.end_capture().unwrap();
        let a = sim.alloc(d, 4096, s).unwrap();
        sim.synchronize().unwrap();
        sim.free(d, a, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.pool_cached(p).unwrap(), 4096);
        assert_eq!(sim.pool_cached(orig).unwrap(), 0);
    }

    #[test]
    fn pool_reuse_is_cheaper_than_os_alloc() {
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let run = |hold: bool| {
            let mut sim = Sim::new(h100());
            if hold {
                sim.set_default_pool_release_threshold(u64::MAX).unwrap();
            }
            let a = sim.alloc(d, bytes, s).unwrap();
            sim.synchronize().unwrap();
            sim.free(d, a, s).unwrap();
            let b = sim.alloc(d, bytes, s).unwrap();
            sim.synchronize().unwrap();
            assert!(sim.is_resident(b, d).unwrap());
            sim.clock_ns()
        };
        assert!(
            run(true) < run(false),
            "cached cudaMallocFromPoolAsync must beat first-touch; hold={} release={}",
            run(true),
            run(false)
        );
    }

    #[test]
    fn pool_reuse_attr_skips_opportunistic_cache() {
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let mut sim = Sim::new(h100());
        let p = sim.default_pool(d).unwrap();
        assert_eq!(
            sim.pool_get_attribute(p, MemPoolAttr::ReuseFollowEventDependencies)
                .unwrap(),
            1
        );
        assert_eq!(
            sim.pool_get_attribute(p, MemPoolAttr::ReuseAllowOpportunistic)
                .unwrap(),
            1
        );
        assert_eq!(
            sim.pool_get_attribute(p, MemPoolAttr::ReuseAllowInternalDependencies)
                .unwrap(),
            1
        );
        sim.pool_set_attribute(p, MemPoolAttr::ReleaseThreshold, u64::MAX)
            .unwrap();
        let a = sim.alloc(d, bytes, s).unwrap();
        sim.synchronize().unwrap();
        sim.free(d, a, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        sim.pool_set_attribute(p, MemPoolAttr::ReuseFollowEventDependencies, 0)
            .unwrap();
        sim.pool_set_attribute(p, MemPoolAttr::ReuseAllowInternalDependencies, 0)
            .unwrap();
        let b = sim.alloc(d, bytes, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        sim.free(d, b, s).unwrap();
        sim.synchronize().unwrap();
        sim.pool_set_attribute(p, MemPoolAttr::ReuseAllowOpportunistic, 0)
            .unwrap();
        assert_eq!(
            sim.pool_get_attribute(p, MemPoolAttr::ReuseAllowOpportunistic)
                .unwrap(),
            0
        );
        let c = sim.alloc(d, bytes, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), bytes.saturating_mul(2));
        assert_eq!(
            sim.pool_get_attribute(p, MemPoolAttr::UsedMemCurrent)
                .unwrap(),
            bytes
        );
        assert_eq!(
            sim.pool_get_attribute(p, MemPoolAttr::ReservedMemCurrent)
                .unwrap(),
            bytes.saturating_mul(2)
        );
        match sim.pool_set_attribute(p, MemPoolAttr::ReuseAllowOpportunistic, 2) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pool reuse attr"), "{why}"),
            other => panic!("{other:?}"),
        }
        let gp = sim.graph_pool(d).unwrap();
        match sim.pool_get_attribute(gp, MemPoolAttr::ReuseAllowOpportunistic) {
            Err(SimError::Invalid { why }) => assert!(why.contains("graph mem"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.free(d, c, s).unwrap();
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        assert_eq!(
            sim.pool_get_attribute(p, MemPoolAttr::ReuseAllowOpportunistic)
                .unwrap(),
            0
        );
        match sim.pool_set_attribute(p, MemPoolAttr::ReuseAllowOpportunistic, 1) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        {
            let mut capped = Sim::new(h100());
            let q = capped
                .create_pool_with_props(MemPoolProps {
                    max_size: 4096,
                    ..MemPoolProps::default()
                })
                .unwrap();
            capped
                .pool_set_attribute(q, MemPoolAttr::ReleaseThreshold, u64::MAX)
                .unwrap();
            let x = capped.alloc_from_pool(d, q, 4096, s).unwrap();
            capped.synchronize().unwrap();
            capped.free(d, x, s).unwrap();
            capped.synchronize().unwrap();
            capped
                .pool_set_attribute(q, MemPoolAttr::ReuseAllowOpportunistic, 0)
                .unwrap();
            let _over = capped.alloc_from_pool(d, q, 4096, s).unwrap();
            match capped.synchronize() {
                Err(SimError::Oom { need, free, .. }) => {
                    assert_eq!(need, 4096);
                    assert_eq!(free, 0);
                }
                other => panic!("{other:?}"),
            }
        }
        let sp = sim.create_shareable_pool(d).unwrap();
        let h = sim.pool_export(sp).unwrap();
        let imp = sim.pool_import(d, h).unwrap();
        sim.pool_set_attribute(imp, MemPoolAttr::ReuseAllowOpportunistic, 0)
            .unwrap();
        assert_eq!(
            sim.pool_get_attribute(sp, MemPoolAttr::ReuseAllowOpportunistic)
                .unwrap(),
            0
        );
    }

    #[test]
    fn alloc_from_pool_rejects_wrong_device() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let p0 = sim.default_pool(DeviceId(0)).unwrap();
        match sim.alloc_from_pool(DeviceId(1), p0, 4096, StreamId(0)) {
            Err(SimError::Invalid { why }) => assert!(why.contains("mismatch")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn create_pool_with_props_is_cuda_mempool_create() {
        let d = DeviceId(0);
        let s = StreamId(0);
        {
            let mut sim = Sim::new(h100());
            let p = sim
                .create_pool_with_props(MemPoolProps {
                    max_size: 4096,
                    ..MemPoolProps::default()
                })
                .unwrap();
            let a = sim.alloc_from_pool(d, p, 4096, s).unwrap();
            sim.synchronize().unwrap();
            assert!(sim.is_resident(a, d).unwrap());
            let _over = sim.alloc_from_pool(d, p, 4096, s).unwrap();
            match sim.synchronize() {
                Err(SimError::Oom { need, free, .. }) => {
                    assert_eq!(need, 4096);
                    assert_eq!(free, 0);
                }
                other => panic!("{other:?}"),
            }
        }
        {
            let mut sim = Sim::new(h100());
            match sim.create_pool_with_props(MemPoolProps {
                alloc_type: 0,
                ..MemPoolProps::default()
            }) {
                Err(SimError::Invalid { why }) => assert!(why.contains("pool alloc type"), "{why}"),
                other => panic!("{other:?}"),
            }
            match sim.create_pool_with_props(MemPoolProps {
                location: Place::Host,
                ..MemPoolProps::default()
            }) {
                Err(SimError::Invalid { why }) => assert!(why.contains("pool location"), "{why}"),
                other => panic!("{other:?}"),
            }
            match sim.create_pool_with_props(MemPoolProps {
                handle_types: 2,
                ..MemPoolProps::default()
            }) {
                Err(SimError::Invalid { why }) => {
                    assert!(why.contains("pool handle types"), "{why}");
                }
                other => panic!("{other:?}"),
            }
            let share = sim
                .create_pool_with_props(MemPoolProps {
                    handle_types: MemHandleType::POSIX_FILE_DESCRIPTOR,
                    ..MemPoolProps::default()
                })
                .unwrap();
            let h = sim.pool_export(share).unwrap();
            let _imp = sim.pool_import(d, h).unwrap();
            let p = sim
                .create_pool_with_props(MemPoolProps {
                    max_size: 4096,
                    ..MemPoolProps::default()
                })
                .unwrap();
            sim.set_pool_release_threshold(p, u64::MAX).unwrap();
            let a = sim.alloc_from_pool(d, p, 4096, s).unwrap();
            sim.synchronize().unwrap();
            sim.free(d, a, s).unwrap();
            sim.synchronize().unwrap();
            let reuse = sim.alloc_from_pool(d, p, 4096, s).unwrap();
            sim.synchronize().unwrap();
            assert!(sim.is_resident(reuse, d).unwrap());
            sim.begin_capture(d, s).unwrap();
            match sim.create_pool_with_props(MemPoolProps::default()) {
                Err(SimError::Invalid { why }) => assert!(why.contains("mempool"), "{why}"),
                other => panic!("{other:?}"),
            }
            let _g = sim.end_capture().unwrap();
        }
    }

    #[test]
    fn replica_free_does_not_fill_dest_pool() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let p1 = sim.default_pool(d1).unwrap();
        sim.set_pool_release_threshold(p1, u64::MAX).unwrap();
        let a = sim.alloc(d0, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d0, a, 4096, s));
        enq(sim.memcpy_device_to_device(d0, d1, a, 4096, s));
        sim.synchronize().unwrap();
        sim.free(d1, a, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.pool_cached(p1).unwrap(), 0);
        assert_eq!(sim.hbm_used(d1).unwrap(), 0);
    }

    #[test]
    fn host_register_pins_pageable_for_dma() {
        let mut sim = Sim::new(h100());
        let bytes = 8u64 << 20;
        let h = sim.alloc_host(bytes).unwrap();
        assert!(sim.is_host_pageable(h).unwrap());
        assert!(!sim.is_host_pinned(h).unwrap());
        sim.host_register(h).unwrap();
        assert!(sim.is_host_pinned(h).unwrap());
        assert!(!sim.is_host_pageable(h).unwrap());
        let err = sim.free_host(h).unwrap_err();
        match err {
            SimError::UnknownAlloc { alloc } => assert_eq!(alloc, h),
            other => panic!("{other:?}"),
        }
        let err = sim.free_host_pinned(h).unwrap_err();
        match err {
            SimError::UnknownAlloc { alloc } => assert_eq!(alloc, h),
            other => panic!("{other:?}"),
        }
        sim.host_unregister(h).unwrap();
        sim.free_host(h).unwrap();
    }

    #[test]
    fn pageable_host_kernel_is_not_resident() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let h = sim.alloc_host(4096).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[h], &[h], StreamId(0)));
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, h);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn host_register_rejects_already_pinned() {
        let mut sim = Sim::new(h100());
        let h = sim.alloc_host_pinned(4096).unwrap();
        match sim.host_register(h) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pageable")),
            other => panic!("{other:?}"),
        }
        match sim.host_unregister(h) {
            Err(SimError::Invalid { why }) => assert!(why.contains("registered")),
            other => panic!("{other:?}"),
        }
        sim.free_host_pinned(h).unwrap();
    }

    #[test]
    fn mapped_host_kernel_skips_h2d_and_hbm() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let h = sim.alloc_host_mapped(bytes).unwrap();
        assert!(sim.is_host_mapped(h).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        enq(sim.kernel(d, KernelKind::other(1 << 20, bytes), &[h], &[h], s));
        sim.synchronize().unwrap();
        assert!(sim.clock_ns() > 0);
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        assert!(!sim.is_resident(h, d).unwrap());
        sim.free_host_pinned(h).unwrap();
    }

    #[test]
    fn mapped_host_kernel_is_slower_than_hbm_resident() {
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 32u64 << 20;
        let mut mapped = Sim::new(h100());
        let h = mapped.alloc_host_mapped(bytes).unwrap();
        enq(mapped.kernel(d, KernelKind::other(1 << 20, bytes), &[h], &[h], s));
        mapped.synchronize().unwrap();
        let map_ns = mapped.clock_ns();

        let mut hbm = Sim::new(h100());
        let a = hbm.malloc(d, bytes).unwrap();
        enq(hbm.kernel(d, KernelKind::other(1 << 20, bytes), &[a], &[a], s));
        hbm.synchronize().unwrap();
        let hbm_ns = hbm.clock_ns();
        assert!(
            map_ns > hbm_ns,
            "mapped zero-copy is PCIe; HBM-resident is faster; mapped={map_ns} hbm={hbm_ns}"
        );
    }

    #[test]
    fn host_register_mapped_then_kernel() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let h = sim.alloc_host(4096).unwrap();
        sim.host_register_mapped(h).unwrap();
        assert!(sim.is_host_mapped(h).unwrap());
        enq(sim.kernel(d, KernelKind::other(8, 4096), &[h], &[h], s));
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        sim.host_unregister(h).unwrap();
        sim.free_host(h).unwrap();
    }

    #[test]
    fn free_host_pinned_waits_queued_mapped_kernel() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let h = sim.alloc_host_mapped(bytes).unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 20, bytes), &[h], &[h], s));
        assert!(!sim.query_stream(d, s).unwrap());
        sim.free_host_pinned(h).unwrap();
        assert!(sim.query_stream(d, s).unwrap());
        assert!(sim.clock_ns() > 0);
        assert!(!sim.is_host_mapped(h).unwrap());
    }

    #[test]
    fn alloc_managed_does_not_charge_hbm() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let m = sim.alloc_managed(8 << 20).unwrap();
        assert!(sim.is_managed(m).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        assert!(!sim.is_resident(m, d).unwrap());
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn alloc_managed_with_flags_is_cuda_malloc_managed() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let g = sim
            .alloc_managed_with_flags(4096, MemAttachFlags::GLOBAL)
            .unwrap();
        assert_eq!(sim.mem_attach(g).unwrap(), MemAttach::Global);
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        let h = sim
            .alloc_managed_with_flags(4096, MemAttachFlags::HOST)
            .unwrap();
        assert_eq!(sim.mem_attach(h).unwrap(), MemAttach::Host);
        match sim.alloc_managed_with_flags(4096, MemAttachFlags::SINGLE) {
            Err(SimError::Invalid { why }) => assert!(why.contains("managed flags"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.alloc_managed_with_flags(4096, 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("managed flags"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, StreamId(0)).unwrap();
        match sim.alloc_managed_with_flags(4096, MemAttachFlags::GLOBAL) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _cap = sim.end_capture().unwrap();
    }

    #[test]
    fn managed_kernel_fault_migrates_and_charges_hbm() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 20, bytes), &[m], &[m], s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(m, d).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        assert_eq!(sim.bytes_moved(), bytes);
        sim.free_sync(m).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
    }

    #[test]
    fn attach_single_blocks_other_stream() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s0 = StreamId(1);
        let s1 = StreamId(2);
        let bytes = 8u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.prefetch(d, m, s0));
        enq(sim.stream_attach(d, m, s0, MemAttach::Single));
        sim.synchronize().unwrap();
        assert_eq!(sim.mem_attach(m).unwrap(), MemAttach::Single);
        assert!(sim.is_attached_to(m, s0).unwrap());
        assert!(!sim.is_attached_to(m, s1).unwrap());
        enq(sim.kernel(d, KernelKind::other(1 << 20, bytes), &[m], &[m], s1));
        match sim.synchronize() {
            Err(SimError::Invalid { why }) => assert!(why.contains("not attached")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn stream_attach_with_flags_is_cuda_stream_attach_mem_async() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s0 = StreamId(1);
        let bytes = 4096u64;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.prefetch(d, m, s0));
        enq(sim.stream_attach_with_flags(d, m, s0, MemAttachFlags::SINGLE));
        sim.synchronize().unwrap();
        assert_eq!(sim.mem_attach(m).unwrap(), MemAttach::Single);
        enq(sim.stream_attach_with_flags(d, m, s0, MemAttachFlags::GLOBAL));
        sim.synchronize().unwrap();
        assert_eq!(sim.mem_attach(m).unwrap(), MemAttach::Global);
        enq(sim.stream_attach_with_flags(d, m, s0, MemAttachFlags::HOST));
        sim.synchronize().unwrap();
        assert_eq!(sim.mem_attach(m).unwrap(), MemAttach::Host);
        match sim.stream_attach_with_flags(d, m, s0, 0) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("stream attach flags"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, StreamId(0)).unwrap();
        match sim.stream_attach_with_flags(d, m, s0, MemAttachFlags::GLOBAL) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn attach_single_same_stream_kernel_ok() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(1);
        let bytes = 8u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.prefetch(d, m, s));
        enq(sim.stream_attach(d, m, s, MemAttach::Single));
        enq(sim.kernel(d, KernelKind::other(1 << 20, bytes), &[m], &[m], s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(m, d).unwrap());
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn attach_host_blocks_kernel_and_device_prefetch() {
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        {
            let mut sim = Sim::new(h100());
            let m = sim.alloc_managed(bytes).unwrap();
            enq(sim.prefetch(d, m, s));
            enq(sim.stream_attach(d, m, s, MemAttach::Host));
            sim.synchronize().unwrap();
            assert_eq!(sim.mem_attach(m).unwrap(), MemAttach::Host);
            enq(sim.kernel(d, KernelKind::other(1 << 20, bytes), &[m], &[m], s));
            match sim.synchronize() {
                Err(SimError::Invalid { why }) => assert!(why.contains("not attached")),
                other => panic!("{other:?}"),
            }
        }
        {
            let mut sim = Sim::new(h100());
            let m = sim.alloc_managed(bytes).unwrap();
            enq(sim.prefetch(d, m, s));
            enq(sim.stream_attach(d, m, s, MemAttach::Host));
            sim.synchronize().unwrap();
            enq(sim.memset(d, m, bytes, s));
            match sim.synchronize() {
                Err(SimError::Invalid { why }) => assert!(why.contains("not attached")),
                other => panic!("{other:?}"),
            }
        }
        {
            let mut sim = Sim::new(h100());
            let m = sim.alloc_managed_host(bytes).unwrap();
            assert_eq!(sim.mem_attach(m).unwrap(), MemAttach::Host);
            enq(sim.prefetch(d, m, s));
            match sim.synchronize() {
                Err(SimError::Invalid { why }) => assert!(why.contains("not attached")),
                other => panic!("{other:?}"),
            }
        }
    }

    #[test]
    fn attach_host_then_global_kernel_on_stream() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let m = sim.alloc_managed_host(bytes).unwrap();
        enq(sim.stream_attach(d, m, s, MemAttach::Global));
        enq(sim.kernel(d, KernelKind::other(1 << 20, bytes), &[m], &[m], s));
        sim.synchronize().unwrap();
        assert_eq!(sim.mem_attach(m).unwrap(), MemAttach::Global);
        assert!(sim.is_resident(m, d).unwrap());
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn attach_host_allows_prefetch_host() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.prefetch(d, m, s));
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        enq(sim.stream_attach(d, m, s, MemAttach::Host));
        enq(sim.prefetch_host(d, m, s));
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn stream_attach_rejects_null_single_device_alloc_and_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        match sim.stream_attach(d, a, s, MemAttach::Global) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not managed")),
            other => panic!("{other:?}"),
        }
        let m = sim.alloc_managed(4096).unwrap();
        match sim.stream_attach(d, m, StreamId::NULL, MemAttach::Single) {
            Err(SimError::Invalid { why }) => assert!(why.contains("null stream")),
            other => panic!("{other:?}"),
        }
        enq(sim.stream_attach(d, m, StreamId::NULL, MemAttach::Global));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        match sim.stream_attach(d, m, s, MemAttach::Host) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 0);
        sim.free_sync(m).unwrap();
        sim.free_sync(a).unwrap();
    }

    #[test]
    fn kernel_start_fault_sees_waited_prefetch() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let home = DeviceId(0);
        let compute = DeviceId(1);
        let copy = StreamId(0);
        let gemm = StreamId(1);
        let bytes = 8u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        sim.mem_advise(m, MemAdvise::SetReadMostly, home).unwrap();
        sim.mem_advise(m, MemAdvise::SetPreferredLocation, home)
            .unwrap();
        enq(sim.prefetch(home, m, copy));
        sim.create_event_disable_timing(EventId(1)).unwrap();
        enq(sim.record_event(home, EventId(1), copy));
        enq(sim.wait_event(compute, EventId(1), gemm));
        enq(sim.kernel(compute, KernelKind::other(8, bytes), &[m], &[], gemm));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(m, home).unwrap());
        assert!(!sim.is_resident(m, compute).unwrap());
        assert_eq!(sim.bytes_moved(), bytes);
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn malloc_can_fill_hbm_until_managed_prefetch() {
        let mut sim = Sim::new(h100().restrict_hbm(4096));
        let d = DeviceId(0);
        let s = StreamId(0);
        let m = sim.alloc_managed(4096).unwrap();
        let a = sim.malloc(d, 4096).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        assert!(!sim.is_resident(m, d).unwrap());
        enq(sim.prefetch(d, m, s));
        match sim.synchronize() {
            Err(SimError::Oom { device, need, free }) => {
                assert_eq!(device, d);
                assert_eq!(need, 4096);
                assert_eq!(free, 0);
            }
            other => panic!("{other:?}"),
        }
        assert!(sim.is_resident(a, d).unwrap());
    }

    #[test]
    fn prefetch_host_refunds_hbm() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.prefetch(d, m, s));
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        enq(sim.prefetch_host(d, m, s));
        sim.synchronize().unwrap();
        assert!(!sim.is_resident(m, d).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        assert!(sim.is_managed(m).unwrap());
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn prefetch_with_flags_is_cuda_mem_prefetch_async() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.prefetch_with_flags(d, m, Place::Device(d), PrefetchFlags::DEFAULT, s));
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        enq(sim.prefetch_with_flags(d, m, Place::Host, PrefetchFlags::DEFAULT, s));
        sim.synchronize().unwrap();
        assert!(!sim.is_resident(m, d).unwrap());
        match sim.prefetch_with_flags(d, m, Place::Device(d), 1, s) {
            Err(SimError::Invalid { why }) => assert!(why.contains("prefetch flags"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn managed_d2d_prefetch_moves_not_replicas() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.prefetch(d0, m, s));
        sim.synchronize().unwrap();
        enq(sim.prefetch(d1, m, s));
        sim.synchronize().unwrap();
        assert!(!sim.is_resident(m, d0).unwrap());
        assert!(sim.is_resident(m, d1).unwrap());
        assert_eq!(sim.hbm_used(d0).unwrap(), 0);
        assert_eq!(sim.hbm_used(d1).unwrap(), bytes);
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn read_mostly_prefetch_replicates() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        sim.mem_advise(m, MemAdvise::SetReadMostly, d0).unwrap();
        assert!(sim.is_read_mostly(m).unwrap());
        enq(sim.prefetch(d0, m, s));
        sim.synchronize().unwrap();
        enq(sim.prefetch(d1, m, s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(m, d0).unwrap());
        assert!(sim.is_resident(m, d1).unwrap());
        assert_eq!(sim.hbm_used(d0).unwrap(), bytes);
        assert_eq!(sim.hbm_used(d1).unwrap(), bytes);
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn drop_managed_copy_refunds_one_gpu() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        sim.mem_advise(m, MemAdvise::SetReadMostly, d0).unwrap();
        enq(sim.prefetch(d0, m, s));
        sim.synchronize().unwrap();
        enq(sim.prefetch(d1, m, s));
        sim.synchronize().unwrap();
        sim.drop_managed_copy(m, d1).unwrap();
        assert!(sim.is_resident(m, d0).unwrap());
        assert!(!sim.is_resident(m, d1).unwrap());
        assert_eq!(sim.hbm_used(d0).unwrap(), bytes);
        assert_eq!(sim.hbm_used(d1).unwrap(), 0);
        enq(sim.kernel(d0, KernelKind::other(8, bytes), &[m], &[], s));
        sim.synchronize().unwrap();
        let last = sim.drop_managed_copy(m, d0).unwrap_err();
        match last {
            SimError::Invalid { why } => assert!(why.contains("last"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn drop_managed_copy_rejects_unmanaged_and_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let err = sim.drop_managed_copy(a, d).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("managed"), "{why}"),
            other => panic!("{other:?}"),
        }
        let m = sim.alloc_managed(4096).unwrap();
        sim.mem_advise(m, MemAdvise::SetReadMostly, d).unwrap();
        enq(sim.prefetch(d, m, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        let cap = sim.drop_managed_copy(m, d).unwrap_err();
        match cap {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn unset_read_mostly_then_prefetch_moves() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        sim.mem_advise(m, MemAdvise::SetReadMostly, d0).unwrap();
        enq(sim.prefetch(d0, m, s));
        sim.synchronize().unwrap();
        enq(sim.prefetch(d1, m, s));
        sim.synchronize().unwrap();
        sim.mem_advise(m, MemAdvise::UnsetReadMostly, d0).unwrap();
        assert!(!sim.is_read_mostly(m).unwrap());
        assert!(sim.is_resident(m, d0).unwrap());
        enq(sim.prefetch(d0, m, s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(m, d0).unwrap());
        assert!(!sim.is_resident(m, d1).unwrap());
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn write_kernel_invalidates_read_mostly_copies() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        sim.mem_advise(m, MemAdvise::SetReadMostly, d0).unwrap();
        enq(sim.prefetch(d0, m, s));
        enq(sim.prefetch(d1, m, s));
        sim.synchronize().unwrap();
        enq(sim.kernel(d1, KernelKind::other(8, 8), &[m], &[m], s));
        sim.synchronize().unwrap();
        assert!(!sim.is_resident(m, d0).unwrap());
        assert!(sim.is_resident(m, d1).unwrap());
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn accessed_by_kernel_reads_without_migrating() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 32u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.prefetch(d0, m, s));
        sim.synchronize().unwrap();
        sim.mem_advise(m, MemAdvise::SetAccessedBy, d1).unwrap();
        assert!(sim.is_accessed_by(m, d1).unwrap());
        let t0 = sim.clock_ns();
        enq(sim.kernel(d1, KernelKind::other(8, bytes), &[m], &[], s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(m, d0).unwrap());
        assert!(!sim.is_resident(m, d1).unwrap());
        assert_eq!(sim.hbm_used(d1).unwrap(), 0);
        let remote = sim.clock_ns().saturating_sub(t0);
        let mut local = Sim::new(HardwareProfile::example_2xh100_pcie());
        let a = local.alloc_managed(bytes).unwrap();
        enq(local.prefetch(d0, a, s));
        local.synchronize().unwrap();
        let t1 = local.clock_ns();
        enq(local.kernel(d0, KernelKind::other(8, bytes), &[a], &[], s));
        local.synchronize().unwrap();
        let hbm = local.clock_ns().saturating_sub(t1);
        assert!(remote > hbm, "remote={remote} hbm={hbm}");
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn va_set_access_kernel_reads_without_dest_hbm() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 32u64 << 20;
        let va = sim.va_reserve(bytes).unwrap();
        sim.va_map(va, d0).unwrap();
        enq(sim.memcpy_pinned_to_device(d0, va, bytes, s));
        sim.synchronize().unwrap();
        sim.va_set_access(va, d1).unwrap();
        assert!(sim.is_accessed_by(va, d1).unwrap());
        assert!(!sim.is_resident(va, d1).unwrap());
        let t0 = sim.clock_ns();
        enq(sim.kernel(d1, KernelKind::other(8, bytes), &[va], &[], s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(va, d0).unwrap());
        assert!(!sim.is_resident(va, d1).unwrap());
        assert_eq!(sim.hbm_used(d1).unwrap(), 0);
        let remote = sim.clock_ns().saturating_sub(t0);
        let mut local = Sim::new(HardwareProfile::example_2xh100_pcie());
        let a = local.va_reserve(bytes).unwrap();
        local.va_map(a, d0).unwrap();
        enq(local.memcpy_pinned_to_device(d0, a, bytes, s));
        local.synchronize().unwrap();
        let t1 = local.clock_ns();
        enq(local.kernel(d0, KernelKind::other(8, bytes), &[a], &[], s));
        local.synchronize().unwrap();
        let hbm = local.clock_ns().saturating_sub(t1);
        assert!(remote > hbm, "remote={remote} hbm={hbm}");
        sim.va_unmap(va).unwrap();
        assert!(!sim.is_accessed_by(va, d1).unwrap());
        sim.va_free(va).unwrap();
    }

    #[test]
    fn va_set_access_rejects_unmapped_peer_and_capture() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let va = sim.va_reserve(4096).unwrap();
        let unmapped = sim.va_set_access(va, d1).unwrap_err();
        match unmapped {
            SimError::Invalid { why } => assert!(why.contains("mapped"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_map(va, d0).unwrap();
        sim.disable_peer(d0, d1).unwrap();
        match sim.va_set_access(va, d1).unwrap_err() {
            SimError::PeerDisabled { src, dst } => {
                assert_eq!(src, d0);
                assert_eq!(dst, d1);
            }
            other => panic!("{other:?}"),
        }
        sim.enable_peer(d0, d1).unwrap();
        sim.va_set_access(va, d1).unwrap();
        sim.va_unset_access(va, d1).unwrap();
        assert!(!sim.is_accessed_by(va, d1).unwrap());
        sim.va_set_access_with_flags(va, d1, MemAccessFlags::PROT_READ)
            .unwrap();
        assert!(sim.is_accessed_by(va, d1).unwrap());
        assert!(!sim.is_va_write_accessed_by(va, d1).unwrap());
        sim.va_set_access_with_flags(va, d1, MemAccessFlags::PROT_READ_WRITE)
            .unwrap();
        assert!(sim.is_va_write_accessed_by(va, d1).unwrap());
        sim.va_set_access_with_flags(va, d1, MemAccessFlags::PROT_NONE)
            .unwrap();
        assert!(!sim.is_accessed_by(va, d1).unwrap());
        match sim.va_set_access_with_flags(va, d1, 2) {
            Err(SimError::Invalid { why }) => assert!(why.contains("va access flags"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d0, s).unwrap();
        match sim.va_set_access(va, d1).unwrap_err() {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        let mut chain = Sim::new(HardwareProfile::example_asymmetric_links());
        let far = chain.va_reserve(4096).unwrap();
        chain.va_map(far, DeviceId(0)).unwrap();
        match chain.va_set_access(far, DeviceId(2)).unwrap_err() {
            SimError::NoPeer { src, dst } => {
                assert_eq!(src, DeviceId(0));
                assert_eq!(dst, DeviceId(2));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn va_get_access_reports_local_peer_read_and_write() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let va = sim.va_reserve(4096).unwrap();
        match sim.va_get_access(va, d0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("mapped"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_map(va, d0).unwrap();
        assert_eq!(
            sim.va_get_access(va, d0).unwrap(),
            MemAccessFlags::PROT_READ_WRITE
        );
        assert_eq!(
            sim.va_get_access(va, d1).unwrap(),
            MemAccessFlags::PROT_NONE
        );
        sim.va_set_access(va, d1).unwrap();
        assert_eq!(
            sim.va_get_access(va, d1).unwrap(),
            MemAccessFlags::PROT_READ
        );
        sim.va_set_access_write(va, d1).unwrap();
        assert_eq!(
            sim.va_get_access(va, d1).unwrap(),
            MemAccessFlags::PROT_READ_WRITE
        );
        sim.va_unset_access(va, d1).unwrap();
        assert_eq!(
            sim.va_get_access(va, d1).unwrap(),
            MemAccessFlags::PROT_NONE
        );
        let a = sim.malloc(d0, 64).unwrap();
        match sim.va_get_access(a, d0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not a VA"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.va_get_access(va, DeviceId(9)) {
            Err(SimError::Invalid { why }) => assert!(why.contains("device"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d0, StreamId(0)).unwrap();
        assert_eq!(
            sim.va_get_access(va, d0).unwrap(),
            MemAccessFlags::PROT_READ_WRITE
        );
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn va_set_access_write_is_not_resident() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let va = sim.va_reserve(4096).unwrap();
        sim.va_map(va, d0).unwrap();
        enq(sim.memcpy_pinned_to_device(d0, va, 4096, s));
        sim.synchronize().unwrap();
        sim.va_set_access(va, d1).unwrap();
        enq(sim.kernel(d1, KernelKind::other(8, 8), &[va], &[va], s));
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, va);
                assert_eq!(device, d1);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn va_set_access_write_kernel_skips_dest_hbm() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 32u64 << 20;
        let va = sim.va_reserve(bytes).unwrap();
        sim.va_map(va, d0).unwrap();
        enq(sim.memcpy_pinned_to_device(d0, va, bytes, s));
        sim.synchronize().unwrap();
        sim.va_set_access_write(va, d1).unwrap();
        assert!(sim.is_accessed_by(va, d1).unwrap());
        assert!(sim.is_va_write_accessed_by(va, d1).unwrap());
        assert!(!sim.is_resident(va, d1).unwrap());
        let t0 = sim.clock_ns();
        enq(sim.kernel(d1, KernelKind::other(8, bytes), &[va], &[va], s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(va, d0).unwrap());
        assert!(!sim.is_resident(va, d1).unwrap());
        assert_eq!(sim.hbm_used(d1).unwrap(), 0);
        let remote = sim.clock_ns().saturating_sub(t0);
        enq(sim.memset(d1, va, 4096, s));
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d1).unwrap(), 0);
        sim.va_set_access(va, d1).unwrap();
        assert!(!sim.is_va_write_accessed_by(va, d1).unwrap());
        enq(sim.kernel(d1, KernelKind::other(8, 8), &[va], &[va], s));
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, va);
                assert_eq!(device, d1);
            }
            other => panic!("{other:?}"),
        }
        let mut local = Sim::new(HardwareProfile::example_2xh100_pcie());
        let a = local.va_reserve(bytes).unwrap();
        local.va_map(a, d0).unwrap();
        enq(local.memcpy_pinned_to_device(d0, a, bytes, s));
        local.synchronize().unwrap();
        let t1 = local.clock_ns();
        enq(local.kernel(d0, KernelKind::other(8, bytes), &[a], &[a], s));
        local.synchronize().unwrap();
        let hbm = local.clock_ns().saturating_sub(t1);
        assert!(remote > hbm, "remote={remote} hbm={hbm}");
    }

    #[test]
    fn pool_set_access_kernel_reads_and_writes_without_dest_hbm() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 32u64 << 20;
        let p0 = sim.default_pool(d0).unwrap();
        let a = sim.alloc(d0, bytes, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d0, a, bytes, s));
        sim.synchronize().unwrap();
        sim.pool_set_access(p0, d1).unwrap();
        assert!(sim.is_pool_accessed_by(p0, d1).unwrap());
        assert!(sim.is_accessed_by(a, d1).unwrap());
        assert!(!sim.is_resident(a, d1).unwrap());
        let t0 = sim.clock_ns();
        enq(sim.kernel(d1, KernelKind::other(8, bytes), &[a], &[a], s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(a, d0).unwrap());
        assert!(!sim.is_resident(a, d1).unwrap());
        assert_eq!(sim.hbm_used(d1).unwrap(), 0);
        let remote = sim.clock_ns().saturating_sub(t0);
        let mut local = Sim::new(HardwareProfile::example_2xh100_pcie());
        let b = local.alloc(d0, bytes, s).unwrap();
        enq(local.memcpy_pinned_to_device(d0, b, bytes, s));
        local.synchronize().unwrap();
        let t1 = local.clock_ns();
        enq(local.kernel(d0, KernelKind::other(8, bytes), &[b], &[b], s));
        local.synchronize().unwrap();
        let hbm = local.clock_ns().saturating_sub(t1);
        assert!(remote > hbm, "remote={remote} hbm={hbm}");
        sim.free(d0, a, s).unwrap();
        sim.synchronize().unwrap();
    }

    #[test]
    fn pool_set_access_covers_existing_and_later_allocs() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let p0 = sim.default_pool(d0).unwrap();
        let first = sim.alloc(d0, 4096, s).unwrap();
        sim.synchronize().unwrap();
        sim.pool_set_access(p0, d1).unwrap();
        let later = sim.alloc(d0, 4096, s).unwrap();
        sim.synchronize().unwrap();
        enq(sim.kernel(d1, KernelKind::other(8, 8), &[first], &[], s));
        enq(sim.kernel(d1, KernelKind::other(8, 8), &[later], &[], s));
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d1).unwrap(), 0);
    }

    #[test]
    fn pool_set_access_rejects_peer_disabled_malloc_and_capture() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let p0 = sim.default_pool(d0).unwrap();
        sim.disable_peer(d0, d1).unwrap();
        match sim.pool_set_access(p0, d1).unwrap_err() {
            SimError::PeerDisabled { src, dst } => {
                assert_eq!(src, d0);
                assert_eq!(dst, d1);
            }
            other => panic!("{other:?}"),
        }
        sim.enable_peer(d0, d1).unwrap();
        sim.pool_set_access(p0, d1).unwrap();
        sim.pool_unset_access(p0, d1).unwrap();
        assert!(!sim.is_pool_accessed_by(p0, d1).unwrap());
        let m = sim.malloc(d0, 4096).unwrap();
        sim.pool_set_access(p0, d1).unwrap();
        enq(sim.kernel(d1, KernelKind::other(8, 8), &[m], &[], s));
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, m);
                assert_eq!(device, d1);
            }
            other => panic!("{other:?}"),
        }
        let mut cap = Sim::new(HardwareProfile::example_2xh100_pcie());
        let p_cap = cap.default_pool(d0).unwrap();
        cap.begin_capture(d0, s).unwrap();
        match cap.pool_set_access(p_cap, d1).unwrap_err() {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = cap.end_capture().unwrap();
        let mut flags = Sim::new(HardwareProfile::example_2xh100_pcie());
        let pf = flags.default_pool(d0).unwrap();
        flags
            .pool_set_access_with_flags(pf, d1, MemAccessFlags::PROT_READ_WRITE)
            .unwrap();
        assert!(flags.is_pool_accessed_by(pf, d1).unwrap());
        flags
            .pool_set_access_with_flags(pf, d1, MemAccessFlags::PROT_NONE)
            .unwrap();
        assert!(!flags.is_pool_accessed_by(pf, d1).unwrap());
        match flags.pool_set_access_with_flags(pf, d1, MemAccessFlags::PROT_READ) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pool prot read"), "{why}"),
            other => panic!("{other:?}"),
        }
        match flags.pool_set_access_with_flags(pf, d1, 2) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pool access flags"), "{why}"),
            other => panic!("{other:?}"),
        }
        let mut chain = Sim::new(HardwareProfile::example_asymmetric_links());
        let far = chain.default_pool(DeviceId(0)).unwrap();
        match chain.pool_set_access(far, DeviceId(2)).unwrap_err() {
            SimError::NoPeer { src, dst } => {
                assert_eq!(src, DeviceId(0));
                assert_eq!(dst, DeviceId(2));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn pool_set_access_memset_peer_skips_dest_hbm() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let p0 = sim.default_pool(d0).unwrap();
        let a = sim.alloc(d0, bytes, s).unwrap();
        sim.synchronize().unwrap();
        sim.pool_set_access(p0, d1).unwrap();
        enq(sim.memset(d1, a, bytes, s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(a, d0).unwrap());
        assert!(!sim.is_resident(a, d1).unwrap());
        assert_eq!(sim.hbm_used(d1).unwrap(), 0);
    }

    #[test]
    fn accessed_by_write_still_migrates() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.prefetch(d0, m, s));
        sim.synchronize().unwrap();
        sim.mem_advise(m, MemAdvise::SetAccessedBy, d1).unwrap();
        enq(sim.kernel(d1, KernelKind::other(8, 8), &[m], &[m], s));
        sim.synchronize().unwrap();
        assert!(!sim.is_resident(m, d0).unwrap());
        assert!(sim.is_resident(m, d1).unwrap());
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn preferred_location_kernel_reads_without_migrating() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 32u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.prefetch(d0, m, s));
        sim.synchronize().unwrap();
        sim.mem_advise(m, MemAdvise::SetPreferredLocation, d0)
            .unwrap();
        assert!(sim.is_preferred_location(m, d0).unwrap());
        assert!(!sim.is_preferred_location(m, d1).unwrap());
        let t0 = sim.clock_ns();
        enq(sim.kernel(d1, KernelKind::other(8, bytes), &[m], &[], s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(m, d0).unwrap());
        assert!(!sim.is_resident(m, d1).unwrap());
        assert_eq!(sim.hbm_used(d1).unwrap(), 0);
        let remote = sim.clock_ns().saturating_sub(t0);
        let mut local = Sim::new(HardwareProfile::example_2xh100_pcie());
        let a = local.alloc_managed(bytes).unwrap();
        enq(local.prefetch(d0, a, s));
        local.synchronize().unwrap();
        let t1 = local.clock_ns();
        enq(local.kernel(d0, KernelKind::other(8, bytes), &[a], &[], s));
        local.synchronize().unwrap();
        let hbm = local.clock_ns().saturating_sub(t1);
        assert!(remote > hbm, "remote={remote} hbm={hbm}");
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn preferred_location_write_still_migrates() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.prefetch(d0, m, s));
        sim.synchronize().unwrap();
        sim.mem_advise(m, MemAdvise::SetPreferredLocation, d0)
            .unwrap();
        enq(sim.kernel(d1, KernelKind::other(8, 8), &[m], &[m], s));
        sim.synchronize().unwrap();
        assert!(!sim.is_resident(m, d0).unwrap());
        assert!(sim.is_resident(m, d1).unwrap());
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn unset_preferred_location_then_kernel_migrates() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.prefetch(d0, m, s));
        sim.synchronize().unwrap();
        sim.mem_advise(m, MemAdvise::SetPreferredLocation, d0)
            .unwrap();
        sim.mem_advise(m, MemAdvise::UnsetPreferredLocation, d0)
            .unwrap();
        assert!(!sim.is_preferred_location(m, d0).unwrap());
        enq(sim.kernel(d1, KernelKind::other(8, 8), &[m], &[], s));
        sim.synchronize().unwrap();
        assert!(!sim.is_resident(m, d0).unwrap());
        assert!(sim.is_resident(m, d1).unwrap());
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn preferred_location_host_kernel_first_touch_still_migrates() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        sim.mem_advise(m, MemAdvise::SetPreferredLocationHost, d)
            .unwrap();
        assert!(sim.is_preferred_host(m).unwrap());
        enq(sim.kernel(d, KernelKind::other(8, 8), &[m], &[], s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(m, d).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn preferred_location_without_residency_still_faults() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        sim.mem_advise(m, MemAdvise::SetPreferredLocation, d0)
            .unwrap();
        enq(sim.kernel(d1, KernelKind::other(8, 8), &[m], &[], s));
        sim.synchronize().unwrap();
        assert!(!sim.is_resident(m, d0).unwrap());
        assert!(sim.is_resident(m, d1).unwrap());
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn mem_advise_rejects_unmanaged_and_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let err = sim.mem_advise(a, MemAdvise::SetReadMostly, d).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("managed"), "{why}"),
            other => panic!("{other:?}"),
        }
        let pref = sim
            .mem_advise(a, MemAdvise::SetPreferredLocation, d)
            .unwrap_err();
        match pref {
            SimError::Invalid { why } => assert!(why.contains("managed"), "{why}"),
            other => panic!("{other:?}"),
        }
        let m = sim.alloc_managed(4096).unwrap();
        sim.begin_capture(d, s).unwrap();
        let cap = sim.mem_advise(m, MemAdvise::SetReadMostly, d).unwrap_err();
        match cap {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let cap_pref = sim
            .mem_advise(m, MemAdvise::SetPreferredLocation, d)
            .unwrap_err();
        match cap_pref {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn mem_advise_with_location_maps_host_places() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let m = sim.alloc_managed(4096).unwrap();
        sim.mem_advise_with_location(m, MemAdvise::SetReadMostly, Place::Host)
            .unwrap();
        assert!(sim.is_read_mostly(m).unwrap());
        sim.mem_advise_with_location(m, MemAdvise::SetPreferredLocation, Place::Host)
            .unwrap();
        assert!(sim.is_preferred_host(m).unwrap());
        sim.mem_advise_with_location(m, MemAdvise::SetPreferredLocation, Place::Device(d))
            .unwrap();
        assert!(sim.is_preferred_location(m, d).unwrap());
        sim.mem_advise_with_location(m, MemAdvise::SetAccessedBy, Place::Device(d))
            .unwrap();
        assert!(sim.is_accessed_by(m, d).unwrap());
        match sim.mem_advise_with_location(m, MemAdvise::SetAccessedBy, Place::Host) {
            Err(SimError::Invalid { why }) => assert!(why.contains("advise location"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.mem_advise_with_location(m, MemAdvise::UnsetAccessedBy, Place::HostPinned) {
            Err(SimError::Invalid { why }) => assert!(why.contains("advise location"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, StreamId(0)).unwrap();
        match sim.mem_advise_with_location(m, MemAdvise::SetReadMostly, Place::Device(d)) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn mem_range_get_attribute_reads_managed_advice() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        match sim.mem_range_get_attribute(a, MemRangeAttr::ReadMostly) {
            Err(SimError::Invalid { why }) => assert!(why.contains("managed"), "{why}"),
            other => panic!("{other:?}"),
        }
        let m = sim.alloc_managed(4096).unwrap();
        assert_eq!(
            sim.mem_range_get_attribute(m, MemRangeAttr::ReadMostly)
                .unwrap(),
            MemRangeAttrValue::ReadMostly(false)
        );
        assert_eq!(
            sim.mem_range_get_attribute(m, MemRangeAttr::PreferredLocation)
                .unwrap(),
            MemRangeAttrValue::PreferredLocation(None)
        );
        assert_eq!(
            sim.mem_range_get_attribute(m, MemRangeAttr::AccessedBy)
                .unwrap(),
            MemRangeAttrValue::AccessedBy(Vec::new())
        );
        assert_eq!(
            sim.mem_range_get_attribute(m, MemRangeAttr::LastPrefetchLocation)
                .unwrap(),
            MemRangeAttrValue::LastPrefetchLocation(None)
        );
        enq(sim.prefetch(d, m, s));
        sim.synchronize().unwrap();
        assert_eq!(
            sim.mem_range_get_attribute(m, MemRangeAttr::LastPrefetchLocation)
                .unwrap(),
            MemRangeAttrValue::LastPrefetchLocation(Some(Place::Device(d)))
        );
        enq(sim.prefetch_host(d, m, s));
        sim.synchronize().unwrap();
        assert_eq!(
            sim.mem_range_get_attribute(m, MemRangeAttr::LastPrefetchLocation)
                .unwrap(),
            MemRangeAttrValue::LastPrefetchLocation(Some(Place::Host))
        );
        sim.mem_advise(m, MemAdvise::SetReadMostly, d).unwrap();
        sim.mem_advise(m, MemAdvise::SetPreferredLocation, d)
            .unwrap();
        sim.mem_advise(m, MemAdvise::SetAccessedBy, d).unwrap();
        assert_eq!(
            sim.mem_range_get_attribute(m, MemRangeAttr::ReadMostly)
                .unwrap(),
            MemRangeAttrValue::ReadMostly(true)
        );
        assert_eq!(
            sim.mem_range_get_attribute(m, MemRangeAttr::PreferredLocation)
                .unwrap(),
            MemRangeAttrValue::PreferredLocation(Some(Place::Device(d)))
        );
        assert_eq!(
            sim.mem_range_get_attribute(m, MemRangeAttr::AccessedBy)
                .unwrap(),
            MemRangeAttrValue::AccessedBy(vec![d])
        );
        sim.mem_advise(m, MemAdvise::SetPreferredLocationHost, d)
            .unwrap();
        assert_eq!(
            sim.mem_range_get_attribute(m, MemRangeAttr::PreferredLocation)
                .unwrap(),
            MemRangeAttrValue::PreferredLocation(Some(Place::Host))
        );
        sim.begin_capture(d, s).unwrap();
        assert_eq!(
            sim.mem_range_get_attribute(m, MemRangeAttr::ReadMostly)
                .unwrap(),
            MemRangeAttrValue::ReadMostly(true)
        );
        assert_eq!(
            sim.mem_range_get_attributes(
                m,
                &[
                    MemRangeAttr::ReadMostly,
                    MemRangeAttr::PreferredLocation,
                    MemRangeAttr::AccessedBy,
                    MemRangeAttr::LastPrefetchLocation,
                    MemRangeAttr::PreferredLocationType,
                    MemRangeAttr::LastPrefetchLocationType,
                ]
            )
            .unwrap(),
            vec![
                MemRangeAttrValue::ReadMostly(true),
                MemRangeAttrValue::PreferredLocation(Some(Place::Host)),
                MemRangeAttrValue::AccessedBy(vec![d]),
                MemRangeAttrValue::LastPrefetchLocation(Some(Place::Host)),
                MemRangeAttrValue::PreferredLocationType(MemLocationType::Host),
                MemRangeAttrValue::LastPrefetchLocationType(MemLocationType::Host),
            ]
        );
        assert!(sim.mem_range_get_attributes(m, &[]).unwrap().is_empty());
        match sim.mem_range_get_attributes(a, &[MemRangeAttr::ReadMostly]) {
            Err(SimError::Invalid { why }) => assert!(why.contains("managed"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn mem_range_get_attribute_location_type_and_id() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let m = sim.alloc_managed(64).unwrap();
        assert_eq!(
            sim.mem_range_get_attribute(m, MemRangeAttr::PreferredLocationType)
                .unwrap(),
            MemRangeAttrValue::PreferredLocationType(MemLocationType::Invalid)
        );
        assert_eq!(
            sim.mem_range_get_attribute(m, MemRangeAttr::PreferredLocationId)
                .unwrap(),
            MemRangeAttrValue::PreferredLocationId(0)
        );
        assert_eq!(
            sim.mem_range_get_attribute(m, MemRangeAttr::LastPrefetchLocationType)
                .unwrap(),
            MemRangeAttrValue::LastPrefetchLocationType(MemLocationType::Invalid)
        );
        sim.mem_advise(m, MemAdvise::SetPreferredLocation, d)
            .unwrap();
        assert_eq!(
            sim.mem_range_get_attribute(m, MemRangeAttr::PreferredLocationType)
                .unwrap(),
            MemRangeAttrValue::PreferredLocationType(MemLocationType::Device)
        );
        assert_eq!(
            sim.mem_range_get_attribute(m, MemRangeAttr::PreferredLocationId)
                .unwrap(),
            MemRangeAttrValue::PreferredLocationId(u32::from(d.0))
        );
        enq(sim.prefetch(d, m, s));
        sim.synchronize().unwrap();
        assert_eq!(
            sim.mem_range_get_attribute(m, MemRangeAttr::LastPrefetchLocationType)
                .unwrap(),
            MemRangeAttrValue::LastPrefetchLocationType(MemLocationType::Device)
        );
        assert_eq!(
            sim.mem_range_get_attribute(m, MemRangeAttr::LastPrefetchLocationId)
                .unwrap(),
            MemRangeAttrValue::LastPrefetchLocationId(u32::from(d.0))
        );
        sim.mem_advise(m, MemAdvise::SetPreferredLocationHost, d)
            .unwrap();
        enq(sim.prefetch_host(d, m, s));
        sim.synchronize().unwrap();
        assert_eq!(
            sim.mem_range_get_attribute(m, MemRangeAttr::PreferredLocationType)
                .unwrap(),
            MemRangeAttrValue::PreferredLocationType(MemLocationType::Host)
        );
        assert_eq!(
            sim.mem_range_get_attribute(m, MemRangeAttr::LastPrefetchLocationType)
                .unwrap(),
            MemRangeAttrValue::LastPrefetchLocationType(MemLocationType::Host)
        );
        assert_eq!(
            sim.mem_range_get_attribute(m, MemRangeAttr::LastPrefetchLocationId)
                .unwrap(),
            MemRangeAttrValue::LastPrefetchLocationId(0)
        );
        assert_eq!(MemLocationType::Device.to_cuda(), 1);
        assert_eq!(MemLocationType::Host.to_cuda(), 2);
        let mut dual = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d1 = DeviceId(1);
        let n = dual.alloc_managed(64).unwrap();
        dual.mem_advise(n, MemAdvise::SetPreferredLocation, d1)
            .unwrap();
        enq(dual.prefetch(d1, n, StreamId(0)));
        dual.synchronize().unwrap();
        assert_eq!(
            dual.mem_range_get_attribute(n, MemRangeAttr::PreferredLocationId)
                .unwrap(),
            MemRangeAttrValue::PreferredLocationId(u32::from(d1.0))
        );
        assert_eq!(
            dual.mem_range_get_attribute(n, MemRangeAttr::LastPrefetchLocationId)
                .unwrap(),
            MemRangeAttrValue::LastPrefetchLocationId(u32::from(d1.0))
        );
        dual.begin_capture(d1, StreamId(0)).unwrap();
        assert_eq!(
            dual.mem_range_get_attribute(n, MemRangeAttr::PreferredLocationType)
                .unwrap(),
            MemRangeAttrValue::PreferredLocationType(MemLocationType::Device)
        );
        let _g = dual.end_capture().unwrap();
    }

    #[test]
    fn already_local_prefetch_does_not_count_bytes() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.prefetch(d, m, s));
        sim.synchronize().unwrap();
        let moved = sim.bytes_moved();
        let t0 = sim.clock_ns();
        enq(sim.prefetch(d, m, s));
        sim.synchronize().unwrap();
        assert_eq!(sim.bytes_moved(), moved);
        assert!(sim.clock_ns() > t0);
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn managed_after_prefetch_is_faster_than_mapped() {
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 32u64 << 20;
        let mut mapped = Sim::new(h100());
        let h = mapped.alloc_host_mapped(bytes).unwrap();
        enq(mapped.kernel(d, KernelKind::other(1 << 20, bytes), &[h], &[h], s));
        mapped.synchronize().unwrap();
        let map_ns = mapped.clock_ns();

        let mut um = Sim::new(h100());
        let m = um.alloc_managed(bytes).unwrap();
        enq(um.prefetch(d, m, s));
        um.synchronize().unwrap();
        let t0 = um.clock_ns();
        enq(um.kernel(d, KernelKind::other(1 << 20, bytes), &[m], &[m], s));
        um.synchronize().unwrap();
        let um_ns = um.clock_ns().saturating_sub(t0);
        assert!(
            map_ns > um_ns,
            "mapped PCIe kernel vs managed-on-HBM; mapped={map_ns} um={um_ns}"
        );
    }

    #[test]
    fn graph_records_managed_prefetch_then_kernel() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let m = sim.alloc_managed(4096).unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.prefetch(d, m, s));
        enq(sim.kernel(d, KernelKind::other(8, 4096), &[m], &[m], s));
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 2);
        assert!(!sim.is_resident(m, d).unwrap());
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(m, d).unwrap());
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn graph_kernel_without_managed_prefetch_is_not_resident() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let m = sim.alloc_managed(4096).unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 4096), &[m], &[m], s));
        let g = sim.end_capture().unwrap();
        let _n = sim.launch_graph(g, s).unwrap();
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, m);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn prefetch_rejects_unmanaged() {
        let mut sim = Sim::new(h100());
        let a = sim.malloc(DeviceId(0), 4096).unwrap();
        match sim.prefetch(DeviceId(0), a, StreamId(0)) {
            Err(SimError::Invalid { why }) => assert!(why.contains("managed")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn va_reserve_does_not_charge_hbm() {
        let mut sim = Sim::new(h100().restrict_hbm(4096));
        let d = DeviceId(0);
        let va = sim.va_reserve(4096).unwrap();
        assert!(sim.is_vmm(va).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        let a = sim.malloc(d, 4096).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        match sim.va_map(va, d) {
            Err(SimError::Oom { device, need, free }) => {
                assert_eq!(device, d);
                assert_eq!(need, 4096);
                assert_eq!(free, 0);
            }
            other => panic!("{other:?}"),
        }
        sim.free_sync(a).unwrap();
        sim.va_map(va, d).unwrap();
        assert!(sim.is_resident(va, d).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn va_reserve_with_flags_is_cu_mem_address_reserve() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let va = sim
            .va_reserve_with_flags(4096, 0, 0, MemReserveFlags::DEFAULT)
            .unwrap();
        assert!(sim.is_vmm(va).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        let aligned = sim
            .va_reserve_with_flags(4096, 4096, 0, MemReserveFlags::DEFAULT)
            .unwrap();
        assert!(sim.is_vmm(aligned).unwrap());
        match sim.va_reserve_with_flags(4096, 0, 0, 1) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("mem reserve flags"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        match sim.va_reserve_with_flags(4096, 0, 1, MemReserveFlags::DEFAULT) {
            Err(SimError::Invalid { why }) => assert!(why.contains("reserve addr"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.va_reserve_with_flags(4096, 3, 0, MemReserveFlags::DEFAULT) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("reserve alignment"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        match sim.va_reserve_with_flags(4096, 8192, 0, MemReserveFlags::DEFAULT) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("reserve alignment"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, StreamId(0)).unwrap();
        match sim.va_reserve_with_flags(4096, 0, 0, MemReserveFlags::DEFAULT) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn va_unmap_keeps_pointer_for_remap() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let va = sim.va_reserve(bytes).unwrap();
        sim.va_map(va, d).unwrap();
        enq(sim.memcpy_pinned_to_device(d, va, bytes, s));
        enq(sim.kernel(d, KernelKind::other(1 << 20, bytes), &[va], &[va], s));
        sim.synchronize().unwrap();
        sim.va_unmap(va).unwrap();
        assert!(!sim.is_resident(va, d).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        assert!(sim.is_vmm(va).unwrap());
        sim.va_map(va, d).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[va], &[va], s));
        sim.synchronize().unwrap();
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn va_unmap_with_size_is_cu_mem_unmap() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let va = sim.va_reserve(4096).unwrap();
        sim.va_map(va, d).unwrap();
        match sim.va_unmap_with_size(va, 2048) {
            Err(SimError::Invalid { why }) => assert!(why.contains("unmap size"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert!(sim.is_resident(va, d).unwrap());
        sim.va_unmap_with_size(va, 4096).unwrap();
        assert!(!sim.is_resident(va, d).unwrap());
        match sim.va_unmap_with_size(va, 4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not mapped"), "{why}"),
            other => panic!("{other:?}"),
        }
        let malloc = sim.malloc(d, 4096).unwrap();
        match sim.va_unmap_with_size(malloc, 4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not a VA"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_map(va, d).unwrap();
        sim.begin_capture(d, StreamId(0)).unwrap();
        match sim.va_unmap_with_size(va, 4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
        sim.free_sync(malloc).unwrap();
    }

    #[test]
    fn kernel_on_unmapped_va_is_not_resident() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(4096).unwrap();
        sim.va_map(va, d).unwrap();
        sim.va_unmap(va).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[va], &[va], s));
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, va);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn va_free_rejects_mapped_and_cuda_free() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let va = sim.va_reserve(4096).unwrap();
        sim.va_map(va, d).unwrap();
        match sim.va_free(va) {
            Err(SimError::Invalid { why }) => assert!(why.contains("mapped")),
            other => panic!("{other:?}"),
        }
        match sim.free_sync(va) {
            Err(SimError::UnknownAlloc { alloc }) => assert_eq!(alloc, va),
            other => panic!("{other:?}"),
        }
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn va_free_with_size_is_cu_mem_address_free() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let va = sim.va_reserve(4096).unwrap();
        sim.va_free_with_size(va, 4096).unwrap();
        assert!(!sim.is_vmm(va).unwrap());
        let va = sim.va_reserve(4096).unwrap();
        match sim.va_free_with_size(va, 2048) {
            Err(SimError::Invalid { why }) => assert!(why.contains("free size"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_map(va, d).unwrap();
        match sim.va_free_with_size(va, 4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("mapped"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_unmap(va).unwrap();
        let malloc = sim.malloc(d, 4096).unwrap();
        match sim.va_free_with_size(malloc, 4096) {
            Err(SimError::UnknownAlloc { alloc }) => assert_eq!(alloc, malloc),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, StreamId(0)).unwrap();
        match sim.va_free_with_size(va, 4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        sim.va_free_with_size(va, 4096).unwrap();
    }

    #[test]
    fn va_map_rejects_already_mapped() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let va = sim.va_reserve(4096).unwrap();
        sim.va_map(va, d).unwrap();
        match sim.va_map(va, d) {
            Err(SimError::Invalid { why }) => assert!(why.contains("already")),
            other => panic!("{other:?}"),
        }
        match sim.va_unmap(va) {
            Ok(()) => {}
            other => panic!("{other:?}"),
        }
        match sim.va_unmap(va) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not mapped")),
            other => panic!("{other:?}"),
        }
        sim.va_free(va).unwrap();
    }

    #[test]
    fn va_create_map_handle_shares_hbm_across_vas() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 4096u64;
        let h = sim.va_create(d, bytes).unwrap();
        assert!(sim.is_handle_live(h).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        let a = sim.va_reserve(bytes).unwrap();
        let b = sim.va_reserve(bytes).unwrap();
        sim.va_map_handle(a, d, 0, h).unwrap();
        sim.va_map_handle(b, d, 0, h).unwrap();
        assert_eq!(sim.handle_maps(h).unwrap(), 2);
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        assert!(sim.is_resident(a, d).unwrap());
        assert!(sim.is_resident(b, d).unwrap());
        enq(sim.memcpy_pinned_to_device(d, a, bytes, s));
        enq(sim.kernel(d, KernelKind::other(8, 8), &[b], &[b], s));
        sim.synchronize().unwrap();
        sim.va_unmap(a).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        assert_eq!(sim.handle_maps(h).unwrap(), 1);
        sim.va_unmap(b).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        assert_eq!(sim.handle_maps(h).unwrap(), 0);
        assert_eq!(sim.handle_refs(h).unwrap(), 1);
        sim.va_release_handle(h).unwrap();
        assert!(!sim.is_handle_live(h).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        sim.va_free(a).unwrap();
        sim.va_free(b).unwrap();
    }

    #[test]
    fn va_get_allocation_properties_wraps_handle_device_and_rdma() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let h = sim.va_create(d, 4096).unwrap();
        let p = sim.va_get_allocation_properties(h).unwrap();
        assert_eq!(p.alloc_type, MemAllocationType::PINNED);
        assert_eq!(p.handle_types, MemHandleType::NONE);
        assert_eq!(p.location, Place::Device(d));
        assert!(!p.gpu_direct_rdma_capable);
        match sim.va_get_allocation_properties(MemHandleId(u64::MAX)) {
            Err(SimError::Invalid { why }) => assert!(why.contains("handle"), "{why}"),
            other => panic!("{other:?}"),
        }
        let mut rdma = Sim::new(HardwareProfile::example_2node_rdma());
        let r = rdma.va_create(DeviceId(0), 4096).unwrap();
        let rp = rdma.va_get_allocation_properties(r).unwrap();
        assert_eq!(rp.location, Place::Device(DeviceId(0)));
        assert!(rp.gpu_direct_rdma_capable);
        sim.begin_capture(d, StreamId(0)).unwrap();
        assert_eq!(
            sim.va_get_allocation_properties(h).unwrap().location,
            Place::Device(d)
        );
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn va_create_with_prop_is_cu_mem_create() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let bytes = 4096u64;
        let h = sim
            .va_create_with_prop(
                bytes,
                MemAllocationProp {
                    location: Place::Device(d),
                    ..MemAllocationProp::default()
                },
                MemCreateFlags::DEFAULT,
            )
            .unwrap();
        assert!(sim.is_handle_live(h).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        let p = sim.va_get_allocation_properties(h).unwrap();
        assert_eq!(p.handle_types, MemHandleType::NONE);
        assert!(!p.gpu_direct_rdma_capable);
        match sim.va_create_with_prop(
            bytes,
            MemAllocationProp {
                location: Place::Device(d),
                ..MemAllocationProp::default()
            },
            1,
        ) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("mem create flags"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        match sim.va_create_with_prop(
            bytes,
            MemAllocationProp {
                alloc_type: 0,
                location: Place::Device(d),
                ..MemAllocationProp::default()
            },
            MemCreateFlags::DEFAULT,
        ) {
            Err(SimError::Invalid { why }) => assert!(why.contains("alloc type"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.va_create_with_prop(
            bytes,
            MemAllocationProp {
                location: Place::Host,
                ..MemAllocationProp::default()
            },
            MemCreateFlags::DEFAULT,
        ) {
            Err(SimError::Invalid { why }) => assert!(why.contains("create location"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.va_create_with_prop(
            bytes,
            MemAllocationProp {
                handle_types: MemHandleType::POSIX_FILE_DESCRIPTOR,
                location: Place::Device(d),
                ..MemAllocationProp::default()
            },
            MemCreateFlags::DEFAULT,
        ) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("vmm handle types"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        let rdma_req = sim
            .va_create_with_prop(
                bytes,
                MemAllocationProp {
                    location: Place::Device(d),
                    gpu_direct_rdma_capable: true,
                    ..MemAllocationProp::default()
                },
                MemCreateFlags::DEFAULT,
            )
            .unwrap();
        assert!(
            !sim.va_get_allocation_properties(rdma_req)
                .unwrap()
                .gpu_direct_rdma_capable
        );
        sim.begin_capture(d, StreamId(0)).unwrap();
        match sim.va_create_with_prop(
            bytes,
            MemAllocationProp {
                location: Place::Device(d),
                ..MemAllocationProp::default()
            },
            MemCreateFlags::DEFAULT,
        ) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn va_get_allocation_granularity_wraps_profile() {
        let mut sim = Sim::new(h100());
        let prop = MemAllocationProp::default();
        assert_eq!(
            sim.va_get_allocation_granularity(prop, MemAllocationGranularity::MINIMUM)
                .unwrap(),
            1
        );
        assert_eq!(
            sim.va_get_allocation_granularity(prop, MemAllocationGranularity::RECOMMENDED)
                .unwrap(),
            1
        );
        match sim.va_get_allocation_granularity(prop, 2) {
            Err(SimError::Invalid { why }) => assert!(why.contains("granularity flags"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.va_get_allocation_granularity(
            MemAllocationProp {
                alloc_type: 0,
                ..MemAllocationProp::default()
            },
            MemAllocationGranularity::MINIMUM,
        ) {
            Err(SimError::Invalid { why }) => assert!(why.contains("alloc type"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.va_get_allocation_granularity(
            MemAllocationProp {
                location: Place::Host,
                ..MemAllocationProp::default()
            },
            MemAllocationGranularity::MINIMUM,
        ) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("granularity location"), "{why}")
            }
            other => panic!("{other:?}"),
        }
        match sim.va_get_allocation_granularity(
            MemAllocationProp {
                location: Place::Device(DeviceId(9)),
                ..MemAllocationProp::default()
            },
            MemAllocationGranularity::MINIMUM,
        ) {
            Err(SimError::Invalid { why }) => assert!(why.contains("device"), "{why}"),
            other => panic!("{other:?}"),
        }
        let aligned = Sim::new(h100().with_va_granularity(2u64 << 20));
        assert_eq!(
            aligned
                .va_get_allocation_granularity(prop, MemAllocationGranularity::MINIMUM)
                .unwrap(),
            2u64 << 20
        );
        sim.begin_capture(DeviceId(0), StreamId(0)).unwrap();
        assert_eq!(
            sim.va_get_allocation_granularity(prop, MemAllocationGranularity::MINIMUM)
                .unwrap(),
            1
        );
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn va_map_handle_rejects_mismatch_mapped_and_release() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let bytes = 4096u64;
        let h = sim.va_create(d0, bytes).unwrap();
        let va = sim.va_reserve(bytes).unwrap();
        match sim.va_map_handle(va, d1, 0, h).unwrap_err() {
            SimError::Invalid { why } => assert!(why.contains("device"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_map_handle(va, d0, 0, h).unwrap();
        sim.va_release_handle(h).unwrap();
        assert!(!sim.is_handle_live(h).unwrap());
        assert_eq!(sim.hbm_used(d0).unwrap(), bytes);
        let va2 = sim.va_reserve(bytes).unwrap();
        match sim.va_map_handle(va2, d0, 0, h).unwrap_err() {
            SimError::Invalid { why } => assert!(why.contains("released"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_unmap(va).unwrap();
        assert_eq!(sim.hbm_used(d0).unwrap(), 0);
        sim.va_free(va).unwrap();
        sim.va_free(va2).unwrap();
        match sim.va_create(d0, 0).unwrap_err() {
            SimError::Invalid { why } => assert!(why.contains("zero"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn va_map_handle_with_flags_is_cu_mem_map() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let bytes = 4096u64;
        let h = sim.va_create(d, bytes).unwrap();
        let va = sim.va_reserve(bytes).unwrap();
        sim.va_map_handle_with_flags(va, d, 0, h, MemMapFlags::DEFAULT)
            .unwrap();
        assert!(sim.is_resident(va, d).unwrap());
        let va2 = sim.va_reserve(bytes).unwrap();
        match sim.va_map_handle_with_flags(va2, d, 0, h, 1) {
            Err(SimError::Invalid { why }) => assert!(why.contains("mem map flags"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, StreamId(0)).unwrap();
        match sim.va_map_handle_with_flags(va2, d, 0, h, MemMapFlags::DEFAULT) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
        sim.va_free(va2).unwrap();
        sim.va_release_handle(h).unwrap();
    }

    #[test]
    fn va_map_handle_with_size_is_cu_mem_map() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let bytes = 4096u64;
        let h = sim.va_create(d, bytes).unwrap();
        let va = sim.va_reserve(bytes).unwrap();
        let va2 = sim.va_reserve(bytes).unwrap();
        match sim.va_map_handle_with_size(va, d, 0, h, 2048, MemMapFlags::DEFAULT) {
            Err(SimError::Invalid { why }) => assert!(why.contains("mem map size"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_map_handle_with_size(va, d, 0, h, bytes, MemMapFlags::DEFAULT)
            .unwrap();
        assert!(sim.is_resident(va, d).unwrap());
        sim.begin_capture(d, StreamId(0)).unwrap();
        match sim.va_map_handle_with_size(va2, d, 0, h, bytes, MemMapFlags::DEFAULT) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
        sim.va_free(va2).unwrap();
        sim.va_release_handle(h).unwrap();
    }

    #[test]
    fn va_retain_handle_aliases_combined_map_without_extra_hbm() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 4096u64;
        let a = sim.va_reserve(bytes).unwrap();
        let b = sim.va_reserve(bytes).unwrap();
        sim.va_map(a, d).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        let h = sim.va_retain_handle(a, d, 0).unwrap();
        assert_eq!(sim.handle_refs(h).unwrap(), 1);
        assert_eq!(sim.handle_maps(h).unwrap(), 1);
        sim.va_map_handle(b, d, 0, h).unwrap();
        assert_eq!(sim.handle_maps(h).unwrap(), 2);
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        enq(sim.kernel(d, KernelKind::other(8, 8), &[b], &[b], s));
        sim.synchronize().unwrap();
        sim.va_unmap(a).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        sim.va_unmap(b).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        sim.va_release_handle(h).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        sim.va_free(a).unwrap();
        sim.va_free(b).unwrap();
    }

    #[test]
    fn va_retain_restores_released_mapped_handle() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let bytes = 4096u64;
        let h = sim.va_create(d, bytes).unwrap();
        let a = sim.va_reserve(bytes).unwrap();
        let b = sim.va_reserve(bytes).unwrap();
        sim.va_map_handle(a, d, 0, h).unwrap();
        sim.va_release_handle(h).unwrap();
        assert!(!sim.is_handle_live(h).unwrap());
        let h2 = sim.va_retain_handle(a, d, 0).unwrap();
        assert_eq!(h2, h);
        assert_eq!(sim.handle_refs(h).unwrap(), 1);
        sim.va_map_handle(b, d, 0, h2).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        sim.va_unmap(a).unwrap();
        sim.va_unmap(b).unwrap();
        sim.va_release_handle(h2).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        sim.va_free(a).unwrap();
        sim.va_free(b).unwrap();
    }

    #[test]
    fn va_retain_handle_rejects_unmapped_and_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(4096).unwrap();
        match sim.va_retain_handle(va, d, 0).unwrap_err() {
            SimError::Invalid { why } => assert!(why.contains("no such map"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_map(va, d).unwrap();
        sim.begin_capture(d, s).unwrap();
        match sim.va_retain_handle(va, d, 0).unwrap_err() {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn va_granularity_rejects_unaligned_reserve_and_map() {
        let gran = 2u64 << 20;
        let mut sim = Sim::new(h100().with_va_granularity(gran));
        let d = DeviceId(0);
        match sim.va_reserve(4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("unaligned"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.va_create(d, 4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("unaligned"), "{why}"),
            other => panic!("{other:?}"),
        }
        let va = sim.va_reserve(gran.saturating_mul(2)).unwrap();
        match sim.va_map_range(va, d, 4096, gran) {
            Err(SimError::Invalid { why }) => assert!(why.contains("unaligned"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.va_map_range(va, d, 0, 4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("unaligned"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_map_range(va, d, 0, gran).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), gran);
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn va_granularity_default_allows_4096() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let va = sim.va_reserve(4096).unwrap();
        sim.va_map(va, d).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn va_map_range_charges_only_that_span() {
        let mut sim = Sim::new(h100().restrict_hbm(8192));
        let d = DeviceId(0);
        let va = sim.va_reserve(16_384).unwrap();
        sim.va_map_range(va, d, 0, 4096).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        assert_eq!(sim.vmm_mapped_bytes(va, d).unwrap(), 4096);
        assert!(!sim.is_resident(va, d).unwrap());
        sim.va_map_range(va, d, 4096, 4096).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 8192);
        match sim.va_map_range(va, d, 8192, 4096) {
            Err(SimError::Oom { device, need, free }) => {
                assert_eq!(device, d);
                assert_eq!(need, 4096);
                assert_eq!(free, 0);
            }
            other => panic!("{other:?}"),
        }
        sim.va_unmap_range(va, d, 0, 4096).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        assert_eq!(sim.vmm_mapped_bytes(va, d).unwrap(), 4096);
        sim.va_unmap(va).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        sim.va_free(va).unwrap();
    }

    #[test]
    fn adjacent_ranges_cover_the_va_for_kernels() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 0, 4096).unwrap();
        sim.va_map_range(va, d, 4096, 4096).unwrap();
        assert!(sim.is_resident(va, d).unwrap());
        enq(sim.kernel(d, KernelKind::other(8, 8), &[va], &[va], s));
        sim.synchronize().unwrap();
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn kernel_on_partial_va_is_not_resident() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 0, 4096).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[va], &[va], s));
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, va);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn kernel_bufs_on_mapped_span_is_ok() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 4096, 4096).unwrap();
        assert!(!sim.is_resident(va, d).unwrap());
        assert!(sim.is_range_resident(va, d, 4096, 4096).unwrap());
        assert!(!sim.is_range_resident(va, d, 0, 4096).unwrap());
        let buf = KernelBuf::span(va, 4096, 4096);
        enq(sim.kernel_bufs(d, KernelKind::other(8, 8), &[buf], &[buf], s));
        sim.synchronize().unwrap();
        sim.va_unmap_range(va, d, 4096, 4096).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn kernel_bufs_on_unmapped_span_is_not_resident() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 0, 4096).unwrap();
        let buf = KernelBuf::span(va, 4096, 4096);
        enq(sim.kernel_bufs(d, KernelKind::other(8, 8), &[buf], &[buf], s));
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, va);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn kernel_bufs_past_alloc_is_invalid() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(4096).unwrap();
        sim.va_map(va, d).unwrap();
        let buf = KernelBuf::span(va, 0, 8192);
        match sim.kernel_bufs(d, KernelKind::other(8, 8), &[buf], &[buf], s) {
            Err(SimError::Invalid { why }) => assert!(why.contains("past")),
            other => panic!("{other:?}"),
        }
        match sim.is_range_resident(va, d, 0, 8192) {
            Err(SimError::Invalid { why }) => assert!(why.contains("past")),
            other => panic!("{other:?}"),
        }
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn h2d_into_interior_mapped_page_is_ok() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 4096, 4096).unwrap();
        enq(sim.memcpy(
            d,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d),
                alloc: va,
                bytes: 4096,
                offset: 4096,
                ..MemcpyOp::default()
            },
            s,
        ));
        sim.synchronize().unwrap();
        sim.va_unmap_range(va, d, 4096, 4096).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn h2d_into_interior_unmapped_page_is_not_resident() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 0, 4096).unwrap();
        enq(sim.memcpy(
            d,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d),
                alloc: va,
                bytes: 4096,
                offset: 4096,
                ..MemcpyOp::default()
            },
            s,
        ));
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, va);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_replays_kernel_bufs_span() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 0, 4096).unwrap();
        sim.begin_capture(d, s).unwrap();
        let buf = KernelBuf::span(va, 0, 4096);
        enq(sim.kernel_bufs(d, KernelKind::other(8, 8), &[buf], &[buf], s));
        let g = sim.end_capture().unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        sim.va_unmap_range(va, d, 0, 4096).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn memset_buf_on_mapped_span_is_ok() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 4096, 4096).unwrap();
        assert!(!sim.is_resident(va, d).unwrap());
        enq(sim.memset_buf(d, KernelBuf::span(va, 4096, 4096), s));
        sim.synchronize().unwrap();
        sim.va_unmap_range(va, d, 4096, 4096).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn memset_whole_va_on_partial_map_is_not_resident() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 0, 4096).unwrap();
        enq(sim.memset(d, va, 8192, s));
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, va);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn memset_first_page_of_partial_va_is_ok() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 0, 4096).unwrap();
        enq(sim.memset(d, va, 4096, s));
        sim.synchronize().unwrap();
        sim.va_unmap_range(va, d, 0, 4096).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn overlapping_va_map_range_is_already_mapped() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 0, 4096).unwrap();
        match sim.va_map_range(va, d, 2048, 4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("already")),
            other => panic!("{other:?}"),
        }
        match sim.va_map_range(va, d, 8192, 4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("past")),
            other => panic!("{other:?}"),
        }
        sim.va_unmap_range(va, d, 0, 4096).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn h2d_into_unmapped_span_is_not_resident() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 0, 4096).unwrap();
        enq(sim.memcpy_pinned_to_device(d, va, 8192, s));
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, va);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn h2d_into_mapped_span_is_ok() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 0, 4096).unwrap();
        enq(sim.memcpy_pinned_to_device(d, va, 4096, s));
        sim.synchronize().unwrap();
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn va_acquire_reuses_released_pointer() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let a = sim.va_acquire(d, 4096).unwrap();
        assert_eq!(sim.vmm_idle_len(), 0);
        sim.va_release(a).unwrap();
        assert_eq!(sim.vmm_idle_len(), 1);
        sim.va_release(a).unwrap();
        assert_eq!(sim.vmm_idle_len(), 1);
        let b = sim.va_acquire(d, 4096).unwrap();
        assert_eq!(a, b);
        assert_eq!(sim.vmm_idle_len(), 0);
        assert!(sim.is_resident(a, d).unwrap());
        sim.va_release(a).unwrap();
        sim.va_free(a).unwrap();
        assert_eq!(sim.vmm_idle_len(), 0);
        let c = sim.va_acquire(d, 4096).unwrap();
        assert_ne!(a, c);
        sim.va_release(c).unwrap();
    }

    #[test]
    fn va_acquire_reuse_skips_reserve_overhead() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let a = sim.va_acquire(d, 4096).unwrap();
        let fresh = sim.clock_ns();
        sim.va_release(a).unwrap();
        let after_rel = sim.clock_ns();
        let b = sim.va_acquire(d, 4096).unwrap();
        assert_eq!(a, b);
        let reuse = sim.clock_ns().saturating_sub(after_rel);
        let map_ns = h100().gpu(d).unwrap().alloc_overhead_ns.max(1);
        assert_eq!(reuse, map_ns);
        assert!(reuse < fresh);
        sim.va_release(a).unwrap();
    }

    #[test]
    fn va_acquire_does_not_share_sizes() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let a = sim.va_acquire(d, 4096).unwrap();
        sim.va_release(a).unwrap();
        let b = sim.va_acquire(d, 8192).unwrap();
        assert_ne!(a, b);
        assert_eq!(sim.vmm_idle_len(), 1);
        sim.va_release(b).unwrap();
        let c = sim.va_acquire(d, 4096).unwrap();
        assert_eq!(a, c);
        assert_eq!(sim.vmm_idle_len(), 1);
        sim.va_release(c).unwrap();
        sim.va_free(b).unwrap();
        sim.va_free(c).unwrap();
    }

    #[test]
    fn va_acquire_oom_parks_reserved_va() {
        let mut sim = Sim::new(h100().restrict_hbm(4096));
        let d = DeviceId(0);
        let blocker = sim.malloc(d, 4096).unwrap();
        match sim.va_acquire(d, 4096) {
            Err(SimError::Oom { device, need, free }) => {
                assert_eq!(device, d);
                assert_eq!(need, 4096);
                assert_eq!(free, 0);
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(sim.vmm_idle_len(), 1);
        sim.free_sync(blocker).unwrap();
        let va = sim.va_acquire(d, 4096).unwrap();
        assert_eq!(sim.vmm_idle_len(), 0);
        sim.va_release(va).unwrap();
    }

    #[test]
    fn va_acquire_rejects_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        match sim.va_acquire(d, 4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn va_release_rejects_malloc() {
        let mut sim = Sim::new(h100());
        let a = sim.malloc(DeviceId(0), 4096).unwrap();
        match sim.va_release(a) {
            Err(SimError::Invalid { why }) => assert!(why.contains("VA")),
            other => panic!("{other:?}"),
        }
        sim.free_sync(a).unwrap();
    }

    #[test]
    fn va_acquire_paged_covers_the_va_in_spans() {
        let mut one = Sim::new(h100());
        let mut paged = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 16u64 << 10;
        let page = bytes / 4;
        let a = one.va_acquire(d, bytes).unwrap();
        let t1 = one.clock_ns();
        let b = paged.va_acquire_paged(d, bytes, page).unwrap();
        let t4 = paged.clock_ns();
        assert!(t4 > t1, "paged={t4} one={t1}");
        assert_eq!(paged.vmm_mapped_bytes(b, d).unwrap(), bytes);
        assert!(paged.is_resident(b, d).unwrap());
        enq(paged.kernel(d, KernelKind::other(8, 8), &[b], &[b], s));
        paged.synchronize().unwrap();
        one.va_release(a).unwrap();
        paged.va_release(b).unwrap();
        let c = paged.va_acquire_paged(d, bytes, page).unwrap();
        assert_eq!(c, b);
        paged.va_release(c).unwrap();
        paged.va_free(c).unwrap();
    }

    #[test]
    fn host_func_duration_matches_profile() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let t0 = sim.clock_ns();
        enq(sim.host_func(d, StreamId(0)));
        sim.synchronize().unwrap();
        assert_eq!(
            sim.clock_ns().saturating_sub(t0),
            h100().gpu(d).unwrap().host_func_ns.max(1)
        );
        assert!(sim
            .operations()
            .any(|o| matches!(o.kind, GpuOp::HostFunc { .. })));
    }

    #[test]
    fn host_func_does_not_occupy_compute() {
        let d = DeviceId(0);
        let bytes = 8u64 << 20;
        let k = KernelKind::GroupedMoeGemm {
            experts: 8,
            tokens_per_expert: 16,
            hidden: 2048,
            ff: 2048,
            dtype: DType::Fp16,
        };
        let wall = |same: bool| {
            let mut sim = Sim::new(h100());
            let a = sim.alloc(d, bytes, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, bytes, StreamId(0)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, k.clone(), &[a], &[a], StreamId(0)));
            let hs = if same { StreamId(0) } else { StreamId(1) };
            enq(sim.host_func(d, hs));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let serial = wall(true);
        let overlap = wall(false);
        assert!(
            serial > overlap,
            "host callback must not take the compute engine; serial={serial} overlap={overlap}"
        );
    }

    #[test]
    fn host_func_waits_for_prior_kernel_on_same_stream() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 20, 4096), &[a], &[a], s));
        enq(sim.host_func(d, s));
        sim.synchronize().unwrap();
        let ops: Vec<Operation> = sim.operations().collect();
        let k = ops
            .iter()
            .find(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.start_ns.is_some())
            .expect("kernel");
        let h = ops
            .iter()
            .find(|o| matches!(o.kind, GpuOp::HostFunc { .. }))
            .expect("host");
        assert!(k.done_ns.unwrap() <= h.start_ns.unwrap());
    }

    #[test]
    fn graph_can_capture_host_func() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        enq(sim.host_func(d, s));
        let g = sim.end_capture().unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert!(sim
            .operations()
            .any(|o| matches!(o.kind, GpuOp::HostFunc { .. }) && o.done));
    }

    #[test]
    fn host_func_params_are_recorded_on_the_op() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let params = HostNodeParams {
            fn_id: 7,
            user_data: 0xfeed_face,
        };
        enq(sim.host_func_params(d, s, params));
        sim.synchronize().unwrap();
        let op = sim
            .operations()
            .find(|o| matches!(o.kind, GpuOp::HostFunc { .. }))
            .expect("host");
        match op.kind {
            GpuOp::HostFunc { fn_id, user_data } => {
                assert_eq!(fn_id, 7);
                assert_eq!(user_data, 0xfeed_face);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_host_set_params_does_not_retarget_instantiated_exec() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_host_func_params(
            g,
            HostNodeParams {
                fn_id: 1,
                user_data: 10,
            },
        )
        .unwrap();
        let _ = sim.instantiate_graph(g).unwrap();
        sim.graph_host_set_params(
            g,
            0,
            HostNodeParams {
                fn_id: 2,
                user_data: 20,
            },
        )
        .unwrap();
        let (_, now) = sim.graph_unique_host(g).unwrap();
        assert_eq!(
            now.fn_id, 1,
            "unique host on a definition is the exec snapshot"
        );
        assert_eq!(now.user_data, 10);
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        let launched = sim
            .operations()
            .find(|o| matches!(o.kind, GpuOp::HostFunc { .. }) && o.done)
            .expect("host");
        match launched.kind {
            GpuOp::HostFunc { fn_id, user_data } => {
                assert_eq!(fn_id, 1);
                assert_eq!(user_data, 10);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_exec_host_set_params_retargets_without_second_graph() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_host_func(g).unwrap();
        let exec = sim.instantiate_graph(g).unwrap();
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        let (node, params) = sim.graph_unique_host(exec).unwrap();
        assert_eq!(node, 0);
        assert_eq!(params, HostNodeParams::default());
        let next = HostNodeParams {
            fn_id: 3,
            user_data: 99,
        };
        sim.graph_exec_host_set_params(exec, node, next).unwrap();
        let (_, now) = sim.graph_unique_host(g).unwrap();
        assert_eq!(
            now, next,
            "unique host on the definition forwards to the exec"
        );
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        let launched = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::HostFunc { .. }) && o.done)
            .last()
            .expect("host");
        match launched.kind {
            GpuOp::HostFunc { fn_id, user_data } => {
                assert_eq!(fn_id, 3);
                assert_eq!(user_data, 99);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_host_set_params_before_instantiate_is_snapshotted() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_host_func(g).unwrap();
        let patched = HostNodeParams {
            fn_id: 4,
            user_data: 40,
        };
        sim.graph_host_set_params(g, 0, patched).unwrap();
        let _ = sim.instantiate_graph(g).unwrap();
        let (_, now) = sim.graph_unique_host(g).unwrap();
        assert_eq!(now, patched);
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        let launched = sim
            .operations()
            .find(|o| matches!(o.kind, GpuOp::HostFunc { .. }) && o.done)
            .expect("host");
        match launched.kind {
            GpuOp::HostFunc { fn_id, user_data } => {
                assert_eq!(fn_id, 4);
                assert_eq!(user_data, 40);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_exec_host_set_params_rejects_uninstantiated_and_kernel() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let params = HostNodeParams {
            fn_id: 1,
            user_data: 1,
        };
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_host_func(g).unwrap();
        let err = sim.graph_exec_host_set_params(g, 0, params).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        let exec = sim.instantiate_graph(g).unwrap();
        let kern = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(kern, KernelKind::other(8, 8), &[], &[])
            .unwrap();
        let _ = sim.instantiate_graph(kern).unwrap();
        let err = sim.graph_exec_host_set_params(kern, 0, params).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not a host"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        let err = sim.graph_exec_host_set_params(exec, 0, params).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let err = sim.graph_host_set_params(g, 0, params).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _cap = sim.end_capture().unwrap();
    }

    #[test]
    fn update_graph_treats_host_params_as_not_topology() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let exec_src = sim.create_graph(d, s).unwrap();
        sim.graph_add_host_func_params(
            exec_src,
            HostNodeParams {
                fn_id: 1,
                user_data: 1,
            },
        )
        .unwrap();
        let exec = sim.instantiate_graph(exec_src).unwrap();
        let src = sim.create_graph(d, s).unwrap();
        sim.graph_add_host_func_params(
            src,
            HostNodeParams {
                fn_id: 9,
                user_data: 9,
            },
        )
        .unwrap();
        sim.update_graph(exec, src).unwrap();
        let (_, now) = sim.graph_unique_host(exec).unwrap();
        assert_eq!(
            now,
            HostNodeParams {
                fn_id: 9,
                user_data: 9
            }
        );
    }

    #[test]
    fn graph_exec_host_set_params_beats_exec_update_wall() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_instantiate_ns = 80_000;
            g.graph_update_ns = 9_000;
            g.graph_set_params_ns = 300;
            g.graph_upload_ns = 1_000;
            g.graph_launch_ns = 1_000;
        }
        let d = DeviceId(0);
        let s = StreamId(0);
        let run_update = {
            let mut sim = Sim::new(p.clone());
            let def = sim.create_graph(d, s).unwrap();
            sim.graph_add_host_func(def).unwrap();
            let exec = sim.instantiate_graph(def).unwrap();
            sim.upload_graph(exec).unwrap();
            let src = sim.create_graph(d, s).unwrap();
            sim.graph_add_host_func_params(
                src,
                HostNodeParams {
                    fn_id: 2,
                    user_data: 2,
                },
            )
            .unwrap();
            let t0 = sim.clock_ns();
            sim.update_graph(exec, src).unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let run_set = {
            let mut sim = Sim::new(p);
            let def = sim.create_graph(d, s).unwrap();
            sim.graph_add_host_func(def).unwrap();
            let exec = sim.instantiate_graph(def).unwrap();
            sim.upload_graph(exec).unwrap();
            let t0 = sim.clock_ns();
            sim.graph_exec_host_set_params(
                exec,
                0,
                HostNodeParams {
                    fn_id: 2,
                    user_data: 2,
                },
            )
            .unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        assert!(
            run_set < run_update,
            "SetParams must beat ExecUpdate; set={run_set} update={run_update}"
        );
    }

    #[test]
    fn capture_host_func_params_records_payload() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let params = HostNodeParams {
            fn_id: 5,
            user_data: 50,
        };
        sim.begin_capture(d, s).unwrap();
        enq(sim.host_func_params(d, s, params));
        let g = sim.end_capture().unwrap();
        let (_, now) = sim.graph_unique_host(g).unwrap();
        assert_eq!(now, params);
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        let launched = sim
            .operations()
            .find(|o| matches!(o.kind, GpuOp::HostFunc { .. }) && o.done)
            .expect("host");
        match launched.kind {
            GpuOp::HostFunc { fn_id, user_data } => {
                assert_eq!(fn_id, 5);
                assert_eq!(user_data, 50);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn disable_timing_event_cannot_elapsed() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let start = EventId(1);
        let end = EventId(2);
        sim.create_event_disable_timing(start).unwrap();
        sim.create_event(end).unwrap();
        assert!(!sim.event_timing(start).unwrap());
        assert!(sim.event_timing(end).unwrap());
        enq(sim.record_event(d, start, s));
        enq(sim.record_event(d, end, s));
        sim.synchronize().unwrap();
        match sim.event_elapsed_ns(start, end) {
            Err(SimError::Invalid { why }) => assert!(why.contains("disable timing"), "{why}"),
            other => panic!("{other:?}"),
        }
        let ns = sim.event_elapsed_ns(end, end).unwrap();
        assert_eq!(ns, 0);
        assert!(sim.query_event(start).unwrap());
        enq(sim.wait_event(d, start, StreamId(1)));
        sim.synchronize().unwrap();
    }

    #[test]
    fn create_event_rejects_duplicate_and_capture() {
        let mut sim = Sim::new(h100());
        let ev = EventId(8);
        sim.create_event(ev).unwrap();
        let err = sim.create_event(ev).unwrap_err();
        assert!(matches!(err, SimError::Invalid { .. }), "{err:?}");
        let err = sim.create_event_disable_timing(ev).unwrap_err();
        assert!(matches!(err, SimError::Invalid { .. }), "{err:?}");
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        let err = sim.create_event(EventId(9)).unwrap_err();
        assert!(matches!(err, SimError::Invalid { .. }), "{err:?}");
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn destroy_event_waits_recorded_and_allows_recreate() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let ev = EventId(3);
        match sim.destroy_event(ev) {
            Err(SimError::UnknownEvent { event }) => assert_eq!(event, ev.0),
            other => panic!("{other:?}"),
        }
        sim.create_event(ev).unwrap();
        sim.destroy_event(ev).unwrap();
        match sim.query_event(ev) {
            Err(SimError::UnknownEvent { event }) => assert_eq!(event, ev.0),
            other => panic!("{other:?}"),
        }
        sim.create_event(ev).unwrap();
        let a = sim.alloc(d, 256, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 256), &[a], &[a], s));
        enq(sim.record_event(d, ev, s));
        assert!(!sim.query_event(ev).unwrap());
        sim.destroy_event(ev).unwrap();
        assert!(sim.query_event(ev).is_err());
        sim.create_event_disable_timing(ev).unwrap();
        sim.destroy_event(ev).unwrap();
        sim.create_event(ev).unwrap();
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        match sim.destroy_event(ev) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn blocking_stream_serializes_with_null() {
        let d = DeviceId(0);
        let bytes = 32u64 << 20;
        let k = KernelKind::GroupedMoeGemm {
            experts: 8,
            tokens_per_expert: 16,
            hidden: 2048,
            ff: 2048,
            dtype: DType::Fp16,
        };
        let run = |blocking: bool| {
            let mut sim = Sim::new(h100());
            if blocking {
                sim.set_stream_blocking(d, StreamId(1), true).unwrap();
            }
            let w = sim.alloc(d, bytes, StreamId(0)).unwrap();
            let c = sim.alloc(d, bytes, StreamId(1)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, w, bytes, StreamId(0)));
            enq(sim.memcpy_pinned_to_device(d, c, bytes, StreamId(1)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, k.clone(), &[w], &[w], StreamId::NULL));
            enq(sim.memcpy_pinned_to_device(d, c, bytes, StreamId(1)));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let serial = run(true);
        let overlap = run(false);
        assert!(
            serial > overlap,
            "cudaStreamCreate must serialize with NULL; serial={serial} overlap={overlap}"
        );
    }

    #[test]
    fn set_stream_blocking_rejects_null() {
        let mut sim = Sim::new(h100());
        let err = sim
            .set_stream_blocking(DeviceId(0), StreamId::NULL, true)
            .unwrap_err();
        assert!(matches!(err, SimError::Invalid { .. }), "{err:?}");
        assert!(!sim.stream_is_blocking(DeviceId(0), StreamId::NULL));
    }

    #[test]
    fn set_created_streams_blocking_skips_null() {
        let mut sim = Sim::new(h100());
        sim.set_created_streams_blocking(2).unwrap();
        assert!(!sim.stream_is_blocking(DeviceId(0), StreamId::NULL));
        assert!(sim.stream_is_blocking(DeviceId(0), StreamId(1)));
        sim.set_created_streams_blocking(1).unwrap();
        assert!(sim.stream_is_blocking(DeviceId(0), StreamId(1)));
        sim.set_stream_blocking(DeviceId(0), StreamId(1), false)
            .unwrap();
        assert!(!sim.stream_is_blocking(DeviceId(0), StreamId(1)));
    }

    #[test]
    fn nonblocking_stream_still_overlaps_null_when_peer_is_blocking() {
        let d = DeviceId(0);
        let bytes = 32u64 << 20;
        let k = KernelKind::GroupedMoeGemm {
            experts: 8,
            tokens_per_expert: 16,
            hidden: 2048,
            ff: 2048,
            dtype: DType::Fp16,
        };
        let run = |copy_on: StreamId| {
            let mut sim = Sim::new(h100());
            sim.set_stream_blocking(d, StreamId(1), true).unwrap();
            let w = sim.alloc(d, bytes, StreamId(0)).unwrap();
            let c = sim.alloc(d, bytes, copy_on).unwrap();
            enq(sim.memcpy_pinned_to_device(d, w, bytes, StreamId(0)));
            enq(sim.memcpy_pinned_to_device(d, c, bytes, copy_on));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, k.clone(), &[w], &[w], StreamId::NULL));
            enq(sim.memcpy_pinned_to_device(d, c, bytes, copy_on));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let serial = run(StreamId(1));
        let overlap = run(StreamId(2));
        assert!(
            serial > overlap,
            "non-blocking stream 2 must still overlap NULL; block1={serial} nb2={overlap}"
        );
    }

    #[test]
    fn host_pin_budget_rejects_second_mapped_alloc() {
        let mut sim = Sim::new(h100().restrict_pin(4096));
        let a = sim.alloc_host_mapped(4096).unwrap();
        assert_eq!(sim.pin_used(), 4096);
        match sim.alloc_host_mapped(4096) {
            Err(SimError::PinOom { need, free }) => {
                assert_eq!(need, 4096);
                assert_eq!(free, 0);
            }
            other => panic!("{other:?}"),
        }
        sim.free_host_pinned(a).unwrap();
        assert_eq!(sim.pin_used(), 0);
        let b = sim.alloc_host_mapped(4096).unwrap();
        sim.free_host_pinned(b).unwrap();
    }

    #[test]
    fn pageable_host_does_not_charge_pin_budget() {
        let mut sim = Sim::new(h100().restrict_pin(4096));
        let h = sim.alloc_host(8 << 20).unwrap();
        assert_eq!(sim.pin_used(), 0);
        match sim.host_register(h) {
            Err(SimError::PinOom { need, free }) => {
                assert_eq!(need, 8 << 20);
                assert_eq!(free, 4096);
            }
            other => panic!("{other:?}"),
        }
        sim.free_host(h).unwrap();
    }

    #[test]
    fn independent_stream_runs_during_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let cap = StreamId(0);
        let live = StreamId(1);
        let a = sim.alloc(d, 4096, cap).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, cap));
        sim.synchronize().unwrap();
        sim.begin_capture(d, cap).unwrap();
        let t0 = sim.clock_ns();
        enq(sim.kernel(d, KernelKind::other(1 << 20, 4096), &[a], &[a], live));
        sim.synchronize_stream(d, live).unwrap();
        assert!(sim.clock_ns() > t0);
        assert!(sim.query_stream(d, live).unwrap());
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 0);
    }

    #[test]
    fn thread_local_capture_refuses_independent_stream() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let cap = StreamId(0);
        let live = StreamId(1);
        let a = sim.alloc(d, 4096, cap).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, cap));
        sim.synchronize().unwrap();
        sim.begin_capture_with_mode(d, cap, StreamCaptureMode::ThreadLocal)
            .unwrap();
        let err = sim
            .kernel(d, KernelKind::other(1 << 20, 4096), &[a], &[a], live)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not capturing"), "{why}"),
            other => panic!("{other:?}"),
        }
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 0);
        sim.begin_capture_with_mode(d, cap, StreamCaptureMode::Global)
            .unwrap();
        let err = sim
            .kernel(d, KernelKind::other(8, 8), &[a], &[a], live)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not capturing"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn thread_local_capture_still_forks() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let copy = StreamId(0);
        let compute = StreamId(1);
        let ev = EventId(1);
        let a = sim.alloc(d, 4096, copy).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, copy));
        sim.synchronize().unwrap();
        sim.create_event(ev).unwrap();
        sim.begin_capture_with_mode(d, copy, StreamCaptureMode::ThreadLocal)
            .unwrap();
        enq(sim.record_event(d, ev, copy));
        enq(sim.wait_event(d, ev, compute));
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], compute));
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 3);
        let info_mode = {
            sim.begin_capture_with_mode(d, copy, StreamCaptureMode::ThreadLocal)
                .unwrap();
            let mode = sim.stream_capture_info(d, copy).expect("cap").mode;
            let _g = sim.end_capture().unwrap();
            mode
        };
        assert_eq!(info_mode, StreamCaptureMode::ThreadLocal);
    }

    #[test]
    fn thread_local_fork_requires_idle_stream() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let copy = StreamId(0);
        let compute = StreamId(1);
        let ev = EventId(1);
        let a = sim.alloc(d, 4096, copy).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, copy));
        sim.synchronize().unwrap();
        sim.create_event(ev).unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 40, 4096), &[a], &[a], compute));
        sim.begin_capture_with_mode(d, copy, StreamCaptureMode::ThreadLocal)
            .unwrap();
        enq(sim.record_event(d, ev, copy));
        let err = sim.wait_event(d, ev, compute).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("idle"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        sim.synchronize().unwrap();
    }

    #[test]
    fn thread_exchange_capture_mode_applies_to_next_begin() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let cap = StreamId(0);
        let live = StreamId(1);
        let a = sim.alloc(d, 4096, cap).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, cap));
        sim.synchronize().unwrap();
        assert_eq!(sim.stream_capture_mode(), StreamCaptureMode::Relaxed);
        sim.begin_capture(d, cap).unwrap();
        let prev = sim.thread_exchange_stream_capture_mode(StreamCaptureMode::ThreadLocal);
        assert_eq!(prev, StreamCaptureMode::Relaxed);
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], live));
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 0);
        assert_eq!(sim.stream_capture_mode(), StreamCaptureMode::ThreadLocal);
        sim.begin_capture(d, cap).unwrap();
        assert_eq!(
            sim.stream_capture_info(d, cap).expect("cap").mode,
            StreamCaptureMode::ThreadLocal
        );
        let err = sim
            .kernel(d, KernelKind::other(8, 8), &[a], &[a], live)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not capturing"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        let prev = sim.thread_exchange_stream_capture_mode(StreamCaptureMode::Relaxed);
        assert_eq!(prev, StreamCaptureMode::ThreadLocal);
    }

    #[test]
    fn capturing_stream_cannot_query_or_sync() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        let q = sim.query_stream(d, s).unwrap_err();
        match q {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let sy = sim.synchronize_stream(d, s).unwrap_err();
        match sy {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let node = sim.synchronize().unwrap_err();
        match node {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn event_wait_joins_capture_on_idle_stream() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let copy = StreamId(0);
        let compute = StreamId(1);
        let ev = EventId(1);
        let bytes = 4096u64;
        let resident = sim.alloc(d, bytes, copy).unwrap();
        let extra = sim.alloc(d, bytes, copy).unwrap();
        enq(sim.memcpy_pinned_to_device(d, resident, bytes, copy));
        sim.synchronize().unwrap();
        sim.create_event(ev).unwrap();
        sim.begin_capture(d, copy).unwrap();
        enq(sim.memcpy_pinned_to_device(d, extra, bytes, copy));
        enq(sim.record_event(d, ev, copy));
        enq(sim.wait_event(d, ev, compute));
        enq(sim.kernel(
            d,
            KernelKind::other(8, bytes),
            &[resident],
            &[resident],
            compute,
        ));
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 4);
        let _ = sim.instantiate_graph(g).unwrap();
        let n = sim.launch_graph(g, copy).unwrap();
        assert_eq!(n, 4);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(extra, d).unwrap());
    }

    #[test]
    fn capture_fork_requires_idle_stream() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let copy = StreamId(0);
        let compute = StreamId(1);
        let ev = EventId(1);
        let a = sim.alloc(d, 4096, copy).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, copy));
        sim.synchronize().unwrap();
        sim.create_event(ev).unwrap();
        sim.begin_capture(d, copy).unwrap();
        enq(sim.record_event(d, ev, copy));
        enq(sim.kernel(d, KernelKind::other(1 << 30, 4096), &[a], &[a], compute));
        let err = sim.wait_event(d, ev, compute).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("idle"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        sim.synchronize().unwrap();
    }

    #[test]
    fn wait_on_live_event_does_not_join_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let copy = StreamId(0);
        let live = StreamId(1);
        let ev = EventId(7);
        sim.create_event(ev).unwrap();
        enq(sim.record_event(d, ev, live));
        sim.synchronize().unwrap();
        sim.begin_capture(d, copy).unwrap();
        enq(sim.wait_event(d, ev, live));
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 0);
    }

    #[test]
    fn record_external_does_not_fork_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let copy = StreamId(0);
        let compute = StreamId(1);
        let ev = EventId(3);
        let a = sim.alloc(d, 4096, copy).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, copy));
        sim.synchronize().unwrap();
        sim.create_event(ev).unwrap();
        sim.begin_capture(d, copy).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], copy));
        enq(sim.record_event_external(d, ev, copy));
        enq(sim.wait_event(d, ev, compute));
        assert!(!sim.query_stream(d, compute).unwrap());
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 2);
        let n = sim.launch_graph(g, copy).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
        assert!(sim.event_complete(ev));
        assert!(sim.query_stream(d, compute).unwrap());
    }

    #[test]
    fn event_record_wait_create_with_flags_dispatch() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let copy = StreamId(0);
        let compute = StreamId(1);
        let ev = EventId(3);
        let a = sim.alloc(d, 4096, copy).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, copy));
        sim.synchronize().unwrap();
        sim.create_event_with_flags(ev, EventCreateFlags::DEFAULT)
            .unwrap();
        assert!(sim.event_timing(ev).unwrap());
        sim.begin_capture(d, copy).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], copy));
        enq(sim.record_event_with_flags(d, ev, copy, EventRecordFlags::EXTERNAL));
        enq(sim.wait_event_with_flags(d, ev, compute, EventWaitFlags::DEFAULT));
        assert!(!sim.query_stream(d, compute).unwrap());
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 2);
        match sim.record_event_with_flags(d, ev, copy, 2) {
            Err(SimError::Invalid { why }) => assert!(why.contains("event record flags"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.wait_event_with_flags(d, ev, copy, 2) {
            Err(SimError::Invalid { why }) => assert!(why.contains("event wait flags"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.create_event_with_flags(EventId(9), 8) {
            Err(SimError::Invalid { why }) => assert!(why.contains("event create flags"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.create_event_with_flags(EventId(8), EventCreateFlags::DISABLE_TIMING)
            .unwrap();
        assert!(!sim.event_timing(EventId(8)).unwrap());
        sim.begin_capture(d, copy).unwrap();
        match sim.create_event_with_flags(EventId(10), 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn blocking_sync_event_pays_host_wait_tax() {
        let p = HardwareProfile::parse("gpus=1\nhost_sync_blocking_ns=10000\n").unwrap();
        let d = DeviceId(0);
        let s = StreamId(0);
        let mut def = Sim::new(p.clone());
        def.create_event(EventId(1)).unwrap();
        let a = def.alloc(d, 4096, s).unwrap();
        enq(def.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        enq(def.record_event(d, EventId(1), s));
        def.synchronize_event(EventId(1)).unwrap();
        let t_def = def.clock_ns();
        let mut blk = Sim::new(p);
        blk.create_event_blocking_sync(EventId(1)).unwrap();
        let b = blk.alloc(d, 4096, s).unwrap();
        enq(blk.kernel(d, KernelKind::other(8, 8), &[b], &[b], s));
        enq(blk.record_event(d, EventId(1), s));
        blk.synchronize_event(EventId(1)).unwrap();
        assert_eq!(blk.clock_ns(), t_def.saturating_add(10_000));
    }

    #[test]
    fn ipc_event_handle_aliases_record() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let copy = StreamId(0);
        let compute = StreamId(1);
        let ev = EventId(3);
        match sim.create_event_with_flags(ev, EventCreateFlags::INTERPROCESS) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("interprocess timing"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        sim.create_event_interprocess(ev).unwrap();
        assert!(!sim.event_timing(ev).unwrap());
        match sim.ipc_get_event(EventId(99)) {
            Err(SimError::UnknownEvent { event }) => assert_eq!(event, 99),
            other => panic!("{other:?}"),
        }
        let h = sim.ipc_get_event(ev).unwrap();
        assert_eq!(sim.ipc_get_event(ev).unwrap(), h);
        let imp = sim.ipc_open_event(h).unwrap();
        assert_ne!(imp, ev);
        assert!(sim.is_ipc_event_import(imp).unwrap());
        assert!(!sim.is_ipc_event_import(ev).unwrap());
        match sim.ipc_get_event(imp) {
            Err(SimError::Invalid { why }) => assert!(why.contains("ipc event import"), "{why}"),
            other => panic!("{other:?}"),
        }
        let a = sim.alloc(d, 4096, copy).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, copy));
        sim.synchronize().unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], copy));
        enq(sim.record_event(d, ev, copy));
        enq(sim.wait_event(d, imp, compute));
        assert!(!sim.query_event(imp).unwrap());
        sim.synchronize().unwrap();
        assert!(sim.query_event(imp).unwrap());
        assert!(sim.query_event(ev).unwrap());
        match sim.event_elapsed_ns(ev, imp) {
            Err(SimError::Invalid { why }) => assert!(why.contains("disable timing"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.destroy_event(ev) {
            Err(SimError::Invalid { why }) => assert!(why.contains("ipc mapped"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.destroy_event(imp).unwrap();
        match sim.is_ipc_event_import(imp) {
            Err(SimError::UnknownEvent { event }) => assert_eq!(event, imp.0),
            other => panic!("{other:?}"),
        }
        sim.destroy_event(ev).unwrap();
    }

    #[test]
    fn ipc_event_record_on_import_wakes_source_wait() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let copy = StreamId(0);
        let compute = StreamId(1);
        let ev = EventId(4);
        sim.create_event_interprocess(ev).unwrap();
        let h = sim.ipc_get_event(ev).unwrap();
        let imp = sim.ipc_open_event(h).unwrap();
        let a = sim.alloc(d, 4096, copy).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, copy));
        sim.synchronize().unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], copy));
        enq(sim.record_event(d, imp, copy));
        enq(sim.wait_event(d, ev, compute));
        assert!(!sim.query_event(ev).unwrap());
        sim.synchronize().unwrap();
        assert!(sim.query_event(ev).unwrap());
        assert!(sim.query_event(imp).unwrap());
    }

    #[test]
    fn ipc_event_capture_refused_and_non_interprocess() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.create_event(EventId(1)).unwrap();
        match sim.ipc_get_event(EventId(1)) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not interprocess"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.create_event_disable_timing(EventId(2)).unwrap();
        match sim.ipc_get_event(EventId(2)) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not interprocess"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.ipc_open_event(IpcEventHandleId(99)) {
            Err(SimError::Invalid { why }) => assert!(why.contains("unknown ipc event"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.create_event_interprocess(EventId(3)).unwrap();
        sim.begin_capture(d, s).unwrap();
        match sim.ipc_get_event(EventId(3)) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.ipc_open_event(IpcEventHandleId(1)) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn ipc_event_capture_fork_joins_import_wait() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let copy = StreamId(0);
        let compute = StreamId(1);
        let ev = EventId(3);
        sim.create_event_interprocess(ev).unwrap();
        let h = sim.ipc_get_event(ev).unwrap();
        let imp = sim.ipc_open_event(h).unwrap();
        let a = sim.alloc(d, 4096, copy).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, copy));
        sim.synchronize().unwrap();
        sim.begin_capture(d, copy).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], copy));
        enq(sim.record_event(d, ev, copy));
        enq(sim.wait_event(d, imp, compute));
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 3);
        let n = sim.launch_graph(g, copy).unwrap();
        assert_eq!(n, 3);
        sim.synchronize().unwrap();
        assert!(sim.event_complete(ev));
        assert!(sim.event_complete(imp));
    }

    #[test]
    fn wait_external_does_not_fork_default_record() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let copy = StreamId(0);
        let compute = StreamId(1);
        let ev = EventId(4);
        let a = sim.alloc(d, 4096, copy).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, copy));
        sim.synchronize().unwrap();
        sim.create_event(ev).unwrap();
        sim.begin_capture(d, copy).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], copy));
        enq(sim.record_event(d, ev, copy));
        enq(sim.wait_event_external(d, ev, compute));
        assert!(!sim.query_stream(d, compute).unwrap());
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 2);
        let n = sim.launch_graph(g, copy).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
        assert!(sim.query_stream(d, compute).unwrap());
    }

    #[test]
    fn wait_external_graph_waits_for_live_record() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let copy = StreamId(0);
        let compute = StreamId(1);
        let ev = EventId(5);
        let bytes = 8u64 << 20;
        let a = sim.alloc(d, bytes, copy).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, bytes, copy));
        sim.synchronize().unwrap();
        sim.create_event(ev).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, bytes, copy));
        enq(sim.record_event(d, ev, copy));
        sim.begin_capture(d, compute).unwrap();
        enq(sim.wait_event_external(d, ev, compute));
        enq(sim.kernel(d, KernelKind::other(1 << 20, bytes), &[a], &[a], compute));
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 2);
        let n = sim.launch_graph(g, compute).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
        let rec = sim
            .operations()
            .find(|o| {
                matches!(
                    o.kind,
                    GpuOp::EventRecord {
                        external: false,
                        ..
                    }
                )
            })
            .and_then(|o| o.done_ns)
            .expect("live record");
        let kern = sim
            .operations()
            .find(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.start_ns.is_some())
            .and_then(|o| o.start_ns)
            .expect("kernel");
        assert!(kern >= rec, "kernel={kern} record={rec}");
    }

    #[test]
    fn capture_fork_rejects_other_device() {
        let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let ev = EventId(1);
        sim.create_event(ev).unwrap();
        sim.begin_capture(d0, s).unwrap();
        enq(sim.record_event(d0, ev, s));
        let err = sim.wait_event(d1, ev, s).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("same device"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn forked_graph_launch_overlaps_copy_and_compute() {
        let d = DeviceId(0);
        let copy = StreamId(0);
        let compute = StreamId(1);
        let bytes = 32u64 << 20;
        let k = KernelKind::other(1 << 40, bytes);
        let serial = {
            let mut sim = Sim::new(h100());
            let resident = sim.alloc(d, bytes, copy).unwrap();
            let extra = sim.alloc(d, bytes, copy).unwrap();
            let extra2 = sim.alloc(d, bytes, copy).unwrap();
            enq(sim.memcpy_pinned_to_device(d, resident, bytes, copy));
            sim.synchronize().unwrap();
            sim.begin_capture(d, copy).unwrap();
            enq(sim.memcpy_pinned_to_device(d, extra, bytes, copy));
            enq(sim.kernel(d, k.clone(), &[resident], &[resident], copy));
            enq(sim.memcpy_pinned_to_device(d, extra2, bytes, copy));
            let g = sim.end_capture().unwrap();
            let _ = sim.instantiate_graph(g).unwrap();
            let t0 = sim.clock_ns();
            let n = sim.launch_graph(g, copy).unwrap();
            assert_eq!(n, 3);
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let forked = {
            let mut sim = Sim::new(h100());
            let resident = sim.alloc(d, bytes, copy).unwrap();
            let extra = sim.alloc(d, bytes, copy).unwrap();
            let extra2 = sim.alloc(d, bytes, copy).unwrap();
            enq(sim.memcpy_pinned_to_device(d, resident, bytes, copy));
            sim.synchronize().unwrap();
            sim.create_event(EventId(1)).unwrap();
            sim.begin_capture(d, copy).unwrap();
            enq(sim.memcpy_pinned_to_device(d, extra, bytes, copy));
            enq(sim.record_event(d, EventId(1), copy));
            enq(sim.wait_event(d, EventId(1), compute));
            enq(sim.kernel(d, k, &[resident], &[resident], compute));
            enq(sim.memcpy_pinned_to_device(d, extra2, bytes, copy));
            let g = sim.end_capture().unwrap();
            let _ = sim.instantiate_graph(g).unwrap();
            let t0 = sim.clock_ns();
            let n = sim.launch_graph(g, copy).unwrap();
            assert_eq!(n, 5);
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        assert!(
            serial > forked,
            "forked graph must overlap copy+compute; serial={serial} forked={forked}"
        );
    }

    #[test]
    fn update_graph_treats_stream_as_topology() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s0 = StreamId(0);
        let s1 = StreamId(1);
        let a = sim.alloc(d, 4096, s0).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s0));
        sim.synchronize().unwrap();
        sim.create_event(EventId(1)).unwrap();
        sim.begin_capture(d, s0).unwrap();
        enq(sim.record_event(d, EventId(1), s0));
        enq(sim.wait_event(d, EventId(1), s1));
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s1));
        let exec = sim.end_capture().unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        sim.begin_capture(d, s0).unwrap();
        enq(sim.record_event(d, EventId(1), s0));
        enq(sim.wait_event(d, EventId(1), s0));
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s0));
        let src = sim.end_capture().unwrap();
        let err = sim.update_graph(exec, src).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("topology"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn child_graph_launch_during_capture_expands_on_parent_launch() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let child = sim.end_capture().unwrap();
        let _ = sim.instantiate_graph(child).unwrap();
        sim.begin_capture(d, s).unwrap();
        let recorded = sim.launch_graph(child, s).unwrap();
        assert_eq!(recorded, 1);
        let parent = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(parent).unwrap(), 1);
        let _ = sim.instantiate_graph(parent).unwrap();
        let t0 = sim.clock_ns();
        let n = sim.launch_graph(parent, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert!(sim.clock_ns() > t0);
    }

    #[test]
    fn child_graph_must_already_be_instantiated() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let child = sim.end_capture().unwrap();
        sim.begin_capture(d, s).unwrap();
        let err = sim.launch_graph(child, s).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn independent_stream_launches_child_live_during_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let cap = StreamId(0);
        let live = StreamId(1);
        let a = sim.alloc(d, 4096, cap).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, cap));
        sim.synchronize().unwrap();
        sim.begin_capture(d, cap).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], cap));
        let child = sim.end_capture().unwrap();
        let _ = sim.instantiate_graph(child).unwrap();
        sim.begin_capture(d, cap).unwrap();
        let t0 = sim.clock_ns();
        let n = sim.launch_graph(child, live).unwrap();
        assert_eq!(n, 1);
        sim.synchronize_stream(d, live).unwrap();
        assert!(sim.clock_ns() > t0);
        let parent = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(parent).unwrap(), 0);
    }

    #[test]
    fn destroy_child_graph_breaks_parent_launch() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let child = sim.end_capture().unwrap();
        let _ = sim.instantiate_graph(child).unwrap();
        sim.begin_capture(d, s).unwrap();
        let _n = sim.launch_graph(child, s).unwrap();
        let parent = sim.end_capture().unwrap();
        sim.destroy_graph(child).unwrap();
        let err = sim.launch_graph(parent, s).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("unknown graph"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn clone_graph_recursively_clones_child_graphs() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_clone_ns = 9_000;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let child = sim.end_capture().unwrap();
        let _ = sim.instantiate_graph(child).unwrap();
        sim.begin_capture(d, s).unwrap();
        let _once = sim.launch_graph(child, s).unwrap();
        let _twice = sim.launch_graph(child, s).unwrap();
        let parent = sim.end_capture().unwrap();
        let t0 = sim.clock_ns();
        let cloned = sim.clone_graph(parent).unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(18_000));
        assert_eq!(sim.graph_len(cloned).unwrap(), 2);
        assert!(!sim.graph_instantiated(cloned).unwrap());
        sim.destroy_graph(child).unwrap();
        let err = sim.launch_graph(parent, s).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("unknown graph"), "{why}"),
            other => panic!("{other:?}"),
        }
        let n = sim.launch_graph(cloned, s).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
    }

    #[test]
    fn update_graph_child_id_is_topology() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let child_a = sim.end_capture().unwrap();
        let _ = sim.instantiate_graph(child_a).unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[b], &[b], s));
        let child_b = sim.end_capture().unwrap();
        let _ = sim.instantiate_graph(child_b).unwrap();
        sim.begin_capture(d, s).unwrap();
        let _n = sim.launch_graph(child_a, s).unwrap();
        let exec = sim.end_capture().unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        sim.begin_capture(d, s).unwrap();
        let _n = sim.launch_graph(child_b, s).unwrap();
        let src = sim.end_capture().unwrap();
        let err = sim.update_graph(exec, src).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("topology"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn create_graph_add_kernel_launch_matches_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let captured = sim.end_capture().unwrap();
        let _ = sim.instantiate_graph(captured).unwrap();
        sim.upload_graph(captured).unwrap();
        let built = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(built, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim.instantiate_graph(built).unwrap();
        sim.upload_graph(built).unwrap();
        assert_eq!(
            sim.graph_len(captured).unwrap(),
            sim.graph_len(built).unwrap()
        );
        let t0 = sim.clock_ns();
        let n0 = sim.launch_graph(captured, s).unwrap();
        sim.synchronize().unwrap();
        let cap_ns = sim.clock_ns().saturating_sub(t0);
        let t1 = sim.clock_ns();
        let n1 = sim.launch_graph(built, s).unwrap();
        sim.synchronize().unwrap();
        let bld_ns = sim.clock_ns().saturating_sub(t1);
        assert_eq!(n0, 1);
        assert_eq!(n1, 1);
        assert_eq!(cap_ns, bld_ns);
    }

    #[test]
    fn graph_add_after_instantiate_is_invalid() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        let g = sim.create_graph(d, s).unwrap();
        let exec = sim.instantiate_graph(g).unwrap();
        assert_ne!(exec, g);
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let err = sim
            .graph_add_kernel(exec, KernelKind::other(8, 8), &[a], &[a])
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn empty_created_graph_launch_is_zero_ops() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let g = sim.create_graph(d, s).unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 0);
        assert!(!sim.graph_instantiated(g).unwrap());
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 0);
        assert!(sim.graph_instantiated(g).unwrap());
    }

    #[test]
    fn create_graph_during_capture_is_invalid() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let g = sim.create_graph(d, s).unwrap();
        sim.begin_capture(d, s).unwrap();
        let err = sim.create_graph(d, s).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let err = sim.graph_add_host_func(g).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _cap = sim.end_capture().unwrap();
    }

    #[test]
    fn graph_add_memcpy_then_kernel_moves_bytes() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        sim.synchronize().unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_memcpy(
            g,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d),
                alloc: a,
                bytes: 4096,
                offset: 0,
                ..MemcpyOp::default()
            },
        )
        .unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(a, d).unwrap());
        assert!(sim.bytes_moved() >= 4096);
        let g2 = sim.create_graph(d, s).unwrap();
        let err = sim
            .graph_add_memcpy(
                g2,
                MemcpyOp {
                    src: Place::Host,
                    dst: Place::Device(d),
                    alloc: a,
                    bytes: 4096,
                    offset: 0,
                    ..MemcpyOp::default()
                },
            )
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("pageable"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_add_memset_host_and_event_nodes() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.create_event(EventId(1)).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_memset(g, KernelBuf::whole(a)).unwrap();
        sim.graph_add_host_func(g).unwrap();
        sim.graph_add_event_record(g, EventId(1), false).unwrap();
        sim.graph_add_event_wait(g, EventId(1), false).unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 4);
        sim.synchronize().unwrap();
    }

    #[test]
    fn update_graph_from_explicit_create() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
        sim.synchronize().unwrap();
        let exec = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(exec, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        let src = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(src, KernelKind::other(8, 8), &[b], &[b])
            .unwrap();
        sim.update_graph(exec, src).unwrap();
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
    }

    #[test]
    fn graph_add_child_expands_at_parent_launch() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        let leaf = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(leaf, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim.instantiate_graph(leaf).unwrap();
        let parent = sim.create_graph(d, s).unwrap();
        sim.graph_add_child(parent, leaf).unwrap();
        let err = sim.graph_add_child(parent, parent).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("self"), "{why}"),
            other => panic!("{other:?}"),
        }
        let raw = sim.create_graph(d, s).unwrap();
        let err = sim.graph_add_child(parent, raw).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _ = sim.instantiate_graph(parent).unwrap();
        let n = sim.launch_graph(parent, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
    }

    #[test]
    fn create_graph_does_not_require_idle_stream() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 20, 4096), &[a], &[a], s));
        assert!(!sim.query_stream(d, s).unwrap());
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim.instantiate_graph(g).unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
    }

    #[test]
    fn graph_add_alloc_kernel_free_matches_capture() {
        let mut cap = Sim::new(h100());
        let mut bld = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        cap.begin_capture(d, s).unwrap();
        let a = cap.alloc(d, 4096, s).unwrap();
        enq(cap.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        cap.free(d, a, s).unwrap();
        let captured = cap.end_capture().unwrap();
        let _ = cap.instantiate_graph(captured).unwrap();
        cap.upload_graph(captured).unwrap();
        let built = bld.create_graph(d, s).unwrap();
        let b = bld.graph_add_alloc(built, 4096).unwrap();
        bld.graph_add_kernel(built, KernelKind::other(8, 8), &[b], &[b])
            .unwrap();
        bld.graph_add_dependencies(built, 0, 1).unwrap();
        bld.graph_add_free(built, b).unwrap();
        bld.graph_add_dependencies(built, 1, 2).unwrap();
        let _ = bld.instantiate_graph(built).unwrap();
        bld.upload_graph(built).unwrap();
        assert_eq!(cap.graph_len(captured).unwrap(), 3);
        assert_eq!(bld.graph_len(built).unwrap(), 3);
        let n0 = cap.launch_graph(captured, s).unwrap();
        cap.synchronize().unwrap();
        let n1 = bld.launch_graph(built, s).unwrap();
        bld.synchronize().unwrap();
        assert_eq!(n0, 3);
        assert_eq!(n1, 3);
        assert_eq!(cap.hbm_used(d).unwrap(), 4096);
        assert_eq!(bld.hbm_used(d).unwrap(), 4096);
        assert_eq!(
            cap.graph_mem_get(d, GraphMemAttr::UsedMemCurrent).unwrap(),
            0
        );
        assert_eq!(
            bld.graph_mem_get(d, GraphMemAttr::ReservedMemCurrent)
                .unwrap(),
            4096
        );
        assert_eq!(cap.hbm_peak(), 4096);
        assert_eq!(bld.hbm_peak(), 4096);
        cap.graph_mem_trim(d).unwrap();
        bld.graph_mem_trim(d).unwrap();
        assert_eq!(cap.hbm_used(d).unwrap(), 0);
        assert_eq!(bld.hbm_used(d).unwrap(), 0);
    }

    #[test]
    fn graph_add_alloc_reuses_hbm_on_relaunch() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let g = sim.create_graph(d, s).unwrap();
        let a = sim.graph_add_alloc(g, 4096).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_dependencies(g, 0, 1).unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(a, d).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        let n2 = sim.launch_graph(g, s).unwrap();
        assert_eq!(n2, 2);
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        assert_eq!(sim.hbm_peak(), 4096);
    }

    #[test]
    fn update_graph_rejects_explicit_mem_nodes() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let exec = sim.create_graph(d, s).unwrap();
        let a = sim.graph_add_alloc(exec, 4096).unwrap();
        sim.graph_add_kernel(exec, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        let src = sim.create_graph(d, s).unwrap();
        let b = sim.graph_add_alloc(src, 4096).unwrap();
        sim.graph_add_kernel(src, KernelKind::other(8, 8), &[b], &[b])
            .unwrap();
        let err = sim.update_graph(exec, src).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("mem"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_add_alloc_zero_and_after_instantiate() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let g = sim.create_graph(d, s).unwrap();
        let err = sim.graph_add_alloc(g, 0).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("zero-byte"), "{why}"),
            other => panic!("{other:?}"),
        }
        let exec = sim.instantiate_graph(g).unwrap();
        let err = sim.graph_add_alloc(exec, 4096).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_add_dependencies_orders_alloc_before_kernel() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let g = sim.create_graph(d, s).unwrap();
        let a = sim.graph_add_alloc(g, 4096).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_dependencies(g, 0, 1).unwrap();
        assert_eq!(sim.graph_node_deps(g, 1).unwrap(), vec![0]);
        let exec = sim.instantiate_graph(g).unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(a, d).unwrap());
        let err = sim.graph_add_dependencies(exec, 0, 1).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_add_dependencies_rejects_cycles() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_dependencies(g, 0, 1).unwrap();
        let err = sim.graph_add_dependencies(g, 1, 0).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("cyclic"), "{why}"),
            other => panic!("{other:?}"),
        }
        let err = sim.graph_add_dependencies(g, 0, 0).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("graph dependency"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_destroy_node_drops_edges_and_keeps_handles() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_dependencies(g, 0, 1).unwrap();
        sim.graph_add_dependencies(g, 1, 2).unwrap();
        sim.graph_destroy_node(g, 1).unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 3);
        assert_eq!(sim.graph_nodes(g).unwrap(), vec![0, 2]);
        assert!(sim.graph_edges(g).unwrap().is_empty());
        assert!(sim.graph_node_deps(g, 2).unwrap().is_empty());
        assert_eq!(sim.graph_node_kind(g, 2).unwrap(), GraphNodeKind::Kernel);
        match sim.graph_node_kind(g, 1) {
            Err(SimError::Invalid { why }) => assert!(why.contains("unknown graph node"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.graph_kernel_get_params(g, 1) {
            Err(SimError::Invalid { why }) => assert!(why.contains("unknown graph node"), "{why}"),
            other => panic!("{other:?}"),
        }
        let err = sim.graph_add_dependencies(g, 0, 1).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("graph dependency"), "{why}"),
            other => panic!("{other:?}"),
        }
        let dot = sim.graph_debug_dot(g).unwrap();
        assert!(dot.contains("n0"));
        assert!(dot.contains("n2"));
        assert!(!dot.contains("n1 ["));
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
        let err = sim.graph_destroy_node(g, 1).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("unknown graph node"), "{why}"),
            other => panic!("{other:?}"),
        }
        let exec = sim.instantiate_graph(g).unwrap();
        let err = sim.graph_destroy_node(exec, 0).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        let err = sim.graph_destroy_node(g, 0).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _cap = sim.end_capture().unwrap();
        let g2 = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g2, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_kernel(g2, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let exec2 = sim.instantiate_graph(g2).unwrap();
        sim.graph_destroy_node(g2, 0).unwrap();
        assert_eq!(sim.graph_nodes(g2).unwrap(), vec![1]);
        let n = sim.launch_graph(exec2, s).unwrap();
        assert_eq!(n, 2, "definition destroy does not retarget exec");
        sim.synchronize().unwrap();
        let err = sim.update_graph(exec2, g2).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("topology"), "{why}"),
            other => panic!("{other:?}"),
        }
        let mem = sim.create_graph(d, s).unwrap();
        let id = sim.graph_add_alloc(mem, 4096).unwrap();
        assert_eq!(sim.graph_mem_allocs(mem).unwrap(), vec![id]);
        sim.graph_destroy_node(mem, 0).unwrap();
        assert!(sim.graph_mem_allocs(mem).unwrap().is_empty());
        let n = sim.launch_graph(mem, s).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn graph_add_dependencies_n_is_all_or_nothing() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_empty(g).unwrap();
        sim.graph_add_dependencies_n(g, &[]).unwrap();
        assert!(sim.graph_edges(g).unwrap().is_empty());
        sim.graph_add_dependencies_n(g, &[(0, 1), (1, 2)]).unwrap();
        assert_eq!(sim.graph_edges(g).unwrap(), vec![(0, 1), (1, 2)]);
        match sim.graph_add_dependencies_n(g, &[(2, 0)]) {
            Err(SimError::Invalid { why }) => assert!(why.contains("cyclic"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(sim.graph_edges(g).unwrap(), vec![(0, 1), (1, 2)]);
        let cycle = sim.create_graph(d, s).unwrap();
        sim.graph_add_empty(cycle).unwrap();
        sim.graph_add_empty(cycle).unwrap();
        match sim.graph_add_dependencies_n(cycle, &[(0, 1), (1, 0)]) {
            Err(SimError::Invalid { why }) => assert!(why.contains("cyclic"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert!(sim.graph_edges(cycle).unwrap().is_empty());
        sim.graph_remove_dependencies_n(g, &[(0, 1), (1, 2)])
            .unwrap();
        assert!(sim.graph_edges(g).unwrap().is_empty());
        match sim.graph_remove_dependencies_n(g, &[(9, 0)]) {
            Err(SimError::Invalid { why }) => assert!(why.contains("dependency"), "{why}"),
            other => panic!("{other:?}"),
        }
        let exec = sim.instantiate_graph(g).unwrap();
        match sim.graph_add_dependencies_n(exec, &[(0, 1)]) {
            Err(SimError::Invalid { why }) => assert!(why.contains("instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_remove_dependencies_restores_hyperq_overlap() {
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |remove: bool| {
            let mut sim = Sim::new(h100().with_compute_slots(2));
            let d = DeviceId(0);
            let s = StreamId(0);
            let a = sim.alloc(d, 4096, s).unwrap();
            let b = sim.alloc(d, 4096, s).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
            enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
            sim.synchronize().unwrap();
            let g = sim.create_graph(d, s).unwrap();
            sim.graph_add_kernel(g, kind.clone(), &[a], &[a]).unwrap();
            sim.graph_add_kernel(g, kind.clone(), &[b], &[b]).unwrap();
            sim.graph_add_dependencies(g, 0, 1).unwrap();
            assert_eq!(sim.graph_node_deps(g, 1).unwrap(), vec![0]);
            if remove {
                sim.graph_remove_dependencies(g, 0, 1).unwrap();
                assert!(sim.graph_node_deps(g, 1).unwrap().is_empty());
            }
            let _ = sim.instantiate_graph(g).unwrap();
            sim.upload_graph(g).unwrap();
            let t0 = sim.clock_ns();
            let n = sim.launch_graph(g, s).unwrap();
            sim.synchronize().unwrap();
            assert_eq!(n, 2);
            sim.clock_ns().saturating_sub(t0)
        };
        let chained = run(false);
        let freed = run(true);
        assert!(
            freed < chained,
            "remove_dependencies must restore Hyper-Q overlap; freed={freed} chained={chained}"
        );
    }

    #[test]
    fn graph_remove_dependencies_rejects_instantiated_and_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_dependencies(g, 0, 1).unwrap();
        sim.graph_remove_dependencies(g, 0, 1).unwrap();
        sim.graph_remove_dependencies(g, 0, 1).unwrap();
        let err = sim.graph_remove_dependencies(g, 0, 0).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("graph dependency"), "{why}"),
            other => panic!("{other:?}"),
        }
        let exec = sim.instantiate_graph(g).unwrap();
        let err = sim.graph_remove_dependencies(exec, 0, 1).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        let err = sim.graph_remove_dependencies(g, 0, 1).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _end = sim.end_capture().unwrap();
    }

    #[test]
    fn capture_to_graph_extra_deps_serialize_hyperq() {
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |chain: bool| {
            let mut sim = Sim::new(h100().with_compute_slots(2));
            let d = DeviceId(0);
            let s = StreamId(0);
            let a = sim.alloc(d, 4096, s).unwrap();
            let b = sim.alloc(d, 4096, s).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
            enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
            sim.synchronize().unwrap();
            let g = sim.create_graph(d, s).unwrap();
            sim.graph_add_kernel(g, kind.clone(), &[a], &[a]).unwrap();
            let extra: &[usize] = if chain { &[0] } else { &[] };
            sim.begin_capture_to_graph(d, s, g, extra).unwrap();
            enq(sim.kernel(d, kind.clone(), &[b], &[b], s));
            assert_eq!(sim.end_capture().unwrap(), g);
            if chain {
                assert_eq!(sim.graph_root_nodes(g).unwrap(), vec![0]);
                assert_eq!(sim.graph_edges(g).unwrap(), vec![(0, 1)]);
                assert_eq!(sim.graph_node_dependents(g, 0).unwrap(), vec![1]);
            } else {
                assert_eq!(sim.graph_root_nodes(g).unwrap(), vec![0, 1]);
                assert!(sim.graph_edges(g).unwrap().is_empty());
            }
            let _ = sim.instantiate_graph(g).unwrap();
            sim.upload_graph(g).unwrap();
            let t0 = sim.clock_ns();
            let n = sim.launch_graph(g, s).unwrap();
            sim.synchronize().unwrap();
            assert_eq!(n, 2);
            sim.clock_ns().saturating_sub(t0)
        };
        let chained = run(true);
        let free = run(false);
        assert!(
            free < chained,
            "empty capture-to-graph deps must Hyper-Q overlap; free={free} chained={chained}"
        );
    }

    #[test]
    fn capture_to_graph_piecewise_sessions_are_independent_roots() {
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |piecewise: bool| {
            let mut sim = Sim::new(h100().with_compute_slots(2));
            let d = DeviceId(0);
            let s = StreamId(0);
            let a = sim.alloc(d, 4096, s).unwrap();
            let b = sim.alloc(d, 4096, s).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
            enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
            sim.synchronize().unwrap();
            let g = if piecewise {
                let g = sim.create_graph(d, s).unwrap();
                sim.begin_capture_to_graph(d, s, g, &[]).unwrap();
                enq(sim.kernel(d, kind.clone(), &[a], &[a], s));
                assert_eq!(sim.end_capture().unwrap(), g);
                sim.begin_capture_to_graph(d, s, g, &[]).unwrap();
                enq(sim.kernel(d, kind.clone(), &[b], &[b], s));
                assert_eq!(sim.end_capture().unwrap(), g);
                g
            } else {
                sim.begin_capture(d, s).unwrap();
                enq(sim.kernel(d, kind.clone(), &[a], &[a], s));
                enq(sim.kernel(d, kind.clone(), &[b], &[b], s));
                sim.end_capture().unwrap()
            };
            let _ = sim.instantiate_graph(g).unwrap();
            sim.upload_graph(g).unwrap();
            let t0 = sim.clock_ns();
            let n = sim.launch_graph(g, s).unwrap();
            sim.synchronize().unwrap();
            assert_eq!(n, 2);
            sim.clock_ns().saturating_sub(t0)
        };
        let serial = run(false);
        let piece = run(true);
        assert!(
            piece < serial,
            "piecewise capture-to-graph roots must overlap serial capture; piece={piece} serial={serial}"
        );
    }

    #[test]
    fn capture_to_graph_rejects_instantiated_nested_and_mismatch() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let err = sim.begin_capture_to_graph(d, s, g, &[1]).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("graph dependency"), "{why}"),
            other => panic!("{other:?}"),
        }
        let exec = sim.instantiate_graph(g).unwrap();
        let err = sim.begin_capture_to_graph(d, s, exec, &[0]).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        let err = sim.begin_capture_to_graph(d, s, g, &[]).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("nested"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _end = sim.end_capture().unwrap();
        let mut dual = Sim::new(HardwareProfile::example_2xh100_pcie());
        let g0 = dual.create_graph(d, s).unwrap();
        let err = dual
            .begin_capture_to_graph(DeviceId(1), s, g0, &[])
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("gpu mismatch"), "{why}"),
            other => panic!("{other:?}"),
        }
        let err = dual
            .begin_capture_to_graph(d, s, GraphId(99), &[])
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("unknown graph"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn capture_to_graph_appends_mem_alloc_nodes() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.begin_capture_to_graph(d, s, g, &[0]).unwrap();
        let scratch = sim.alloc(d, 64, s).unwrap();
        let id = sim.end_capture().unwrap();
        assert_eq!(id, g);
        assert_eq!(sim.graph_len(g).unwrap(), 2);
        assert_eq!(sim.graph_mem_allocs(g).unwrap(), vec![scratch]);
        assert_eq!(sim.graph_node_deps(g, 1).unwrap(), vec![0]);
    }

    #[test]
    fn graph_add_empty_does_not_occupy_compute() {
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |empty: bool| {
            let mut sim = Sim::new(h100().with_compute_slots(2));
            let d = DeviceId(0);
            let s = StreamId(0);
            let a = sim.alloc(d, 4096, s).unwrap();
            let b = sim.alloc(d, 4096, s).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
            enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
            sim.synchronize().unwrap();
            let g = sim.create_graph(d, s).unwrap();
            sim.graph_add_kernel(g, kind.clone(), &[a], &[a]).unwrap();
            sim.graph_add_kernel(g, kind.clone(), &[b], &[b]).unwrap();
            if empty {
                sim.graph_add_empty(g).unwrap();
            } else {
                sim.graph_add_kernel(g, kind.clone(), &[a], &[a]).unwrap();
            }
            let _ = sim.instantiate_graph(g).unwrap();
            sim.upload_graph(g).unwrap();
            let t0 = sim.clock_ns();
            let n = sim.launch_graph(g, s).unwrap();
            sim.synchronize().unwrap();
            assert_eq!(n, 3);
            sim.clock_ns().saturating_sub(t0)
        };
        let with_empty = run(true);
        let three_gemm = run(false);
        assert!(
            with_empty < three_gemm,
            "empty must not take a Hyper-Q slot; empty={with_empty} three={three_gemm}"
        );
    }

    #[test]
    fn graph_add_empty_join_then_capture_to_graph() {
        let kind = KernelKind::other(1 << 40, 4096);
        let mut sim = Sim::new(h100().with_compute_slots(2));
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
        sim.synchronize().unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, kind.clone(), &[a], &[a]).unwrap();
        sim.graph_add_kernel(g, kind.clone(), &[b], &[b]).unwrap();
        sim.graph_add_empty(g).unwrap();
        sim.graph_add_dependencies(g, 0, 2).unwrap();
        sim.graph_add_dependencies(g, 1, 2).unwrap();
        sim.begin_capture_to_graph(d, s, g, &[2]).unwrap();
        enq(sim.kernel(d, kind, &[a], &[a], s));
        assert_eq!(sim.end_capture().unwrap(), g);
        assert_eq!(sim.graph_root_nodes(g).unwrap(), vec![0, 1]);
        assert_eq!(sim.graph_node_deps(g, 3).unwrap(), vec![2]);
        let _ = sim.instantiate_graph(g).unwrap();
        sim.graph_node_set_enabled(g, 2, false).unwrap();
        assert!(!sim.graph_node_get_enabled(g, 2).unwrap());
    }

    #[test]
    fn graph_add_empty_rejects_instantiated_and_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_empty(g).unwrap();
        let exec = sim.instantiate_graph(g).unwrap();
        let err = sim.graph_add_empty(exec).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        let err = sim.graph_add_empty(g).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _end = sim.end_capture().unwrap();
    }

    #[test]
    fn update_capture_deps_serialize_next_node() {
        let long = KernelKind::other(1 << 40, 4096);
        let tiny = KernelKind::other(8, 4096);
        let run = |update: bool| {
            let mut sim = Sim::new(h100().with_compute_slots(2));
            let d = DeviceId(0);
            let s = StreamId(0);
            let a = sim.alloc(d, 4096, s).unwrap();
            let b = sim.alloc(d, 4096, s).unwrap();
            let c = sim.alloc(d, 4096, s).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
            enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
            enq(sim.memcpy_pinned_to_device(d, c, 4096, s));
            sim.synchronize().unwrap();
            let g = sim.create_graph(d, s).unwrap();
            sim.graph_add_kernel(g, long.clone(), &[a], &[a]).unwrap();
            sim.begin_capture_to_graph(d, s, g, &[]).unwrap();
            enq(sim.kernel(d, tiny.clone(), &[b], &[b], s));
            if update {
                sim.stream_update_capture_dependencies(d, s, &[0], CaptureDepOp::Set)
                    .unwrap();
            }
            enq(sim.kernel(d, long.clone(), &[c], &[c], s));
            assert_eq!(sim.end_capture().unwrap(), g);
            if update {
                assert_eq!(sim.graph_node_deps(g, 2).unwrap(), vec![0, 1]);
            } else {
                assert_eq!(sim.graph_node_deps(g, 2).unwrap(), vec![1]);
            }
            let _ = sim.instantiate_graph(g).unwrap();
            sim.upload_graph(g).unwrap();
            let t0 = sim.clock_ns();
            let n = sim.launch_graph(g, s).unwrap();
            sim.synchronize().unwrap();
            assert_eq!(n, 3);
            sim.clock_ns().saturating_sub(t0)
        };
        let with = run(true);
        let without = run(false);
        assert!(
            without < with,
            "update deps must wait for A; with={with} without={without}"
        );
    }

    #[test]
    fn update_capture_deps_rejects_idle_and_bad_index() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let err = sim
            .stream_update_capture_dependencies(d, s, &[0], CaptureDepOp::Set)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not capturing"), "{why}"),
            other => panic!("{other:?}"),
        }
        let g = sim.create_graph(d, s).unwrap();
        sim.begin_capture_to_graph(d, s, g, &[]).unwrap();
        let err = sim
            .stream_update_capture_dependencies(d, s, &[0], CaptureDepOp::Set)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("graph dependency"), "{why}"),
            other => panic!("{other:?}"),
        }
        let err = sim
            .stream_update_capture_dependencies(d, StreamId(1), &[], CaptureDepOp::Set)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not capturing"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _end = sim.end_capture().unwrap();
    }

    #[test]
    fn stream_capture_info_during_and_after() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        assert!(!sim.stream_is_capturing(d, s));
        assert!(sim.stream_capture_info(d, s).is_none());
        let g = sim.create_graph(d, s).unwrap();
        sim.begin_capture_to_graph(d, s, g, &[]).unwrap();
        assert!(sim.stream_is_capturing(d, s));
        let info = sim.stream_capture_info(d, s).expect("capturing");
        assert_eq!(info.graph, g);
        assert_eq!(info.origin, (d, s));
        assert!(info.pending_deps.is_empty());
        assert!(info.dependencies.is_empty());
        assert_eq!(info.mode, StreamCaptureMode::Relaxed);
        assert_eq!(sim.graph_len(g).unwrap(), 0);
        let _end = sim.end_capture().unwrap();
        assert!(!sim.stream_is_capturing(d, s));
        assert!(sim.stream_capture_info(d, s).is_none());
        let a = sim.malloc(d, 4096).unwrap();
        sim.begin_capture(d, s).unwrap();
        let info = sim.stream_capture_info(d, s).expect("vanilla capture");
        assert_eq!(info.mode, StreamCaptureMode::Relaxed);
        assert!(info.dependencies.is_empty());
        assert_eq!(sim.graph_len(info.graph).unwrap(), 0);
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let info = sim.stream_capture_info(d, s).expect("after kernel");
        assert_eq!(info.dependencies, vec![0]);
        assert!(info.pending_deps.is_empty());
        let _end = sim.end_capture().unwrap();
        assert!(!sim.stream_is_capturing(d, s));
    }

    #[test]
    fn update_capture_deps_add_unions_set_replaces() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.begin_capture_to_graph(d, s, g, &[]).unwrap();
        sim.stream_update_capture_dependencies(d, s, &[0], CaptureDepOp::Set)
            .unwrap();
        sim.stream_update_capture_dependencies(d, s, &[1], CaptureDepOp::Add)
            .unwrap();
        let info = sim.stream_capture_info(d, s).expect("pending");
        assert_eq!(info.pending_deps, vec![0, 1]);
        assert_eq!(info.dependencies, vec![0, 1]);
        sim.stream_update_capture_dependencies(d, s, &[1], CaptureDepOp::Set)
            .unwrap();
        let info = sim.stream_capture_info(d, s).expect("set");
        assert_eq!(info.pending_deps, vec![1]);
        assert_eq!(info.dependencies, vec![1]);
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let info = sim.stream_capture_info(d, s).expect("consumed");
        assert!(info.pending_deps.is_empty());
        assert_eq!(info.dependencies, vec![2]);
        assert_eq!(sim.end_capture().unwrap(), g);
        assert_eq!(sim.graph_node_deps(g, 2).unwrap(), vec![1]);
    }

    #[test]
    fn stream_capture_info_id_is_unique_per_sequence() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        sim.begin_capture(d, s).unwrap();
        let first = sim.stream_capture_info(d, s).expect("first");
        assert_eq!(first.id, 1);
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let after = sim.stream_capture_info(d, s).expect("after kernel");
        assert_eq!(after.id, first.id);
        let _g1 = sim.end_capture().unwrap();
        sim.begin_capture(d, s).unwrap();
        let second = sim.stream_capture_info(d, s).expect("second");
        assert_eq!(second.id, 2);
        assert_ne!(second.graph, first.graph);
        let _g2 = sim.end_capture().unwrap();
        let copy = StreamId(0);
        let compute = StreamId(1);
        let ev = EventId(1);
        sim.create_event(ev).unwrap();
        sim.begin_capture(d, copy).unwrap();
        let origin = sim.stream_capture_info(d, copy).expect("origin");
        assert_eq!(origin.id, 3);
        enq(sim.record_event(d, ev, copy));
        enq(sim.wait_event(d, ev, compute));
        let fork = sim.stream_capture_info(d, compute).expect("fork");
        assert_eq!(fork.id, origin.id);
        assert_eq!(fork.graph, origin.graph);
        assert_eq!(fork.origin, origin.origin);
        let still = sim.stream_capture_info(d, copy).expect("still");
        assert_eq!(still.id, origin.id);
        let _g3 = sim.end_capture().unwrap();
    }

    #[test]
    fn graph_node_kind_empty_and_kernel() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_empty(g).unwrap();
        assert_eq!(sim.graph_node_kind(g, 0).unwrap(), GraphNodeKind::Kernel);
        assert_eq!(sim.graph_node_kind(g, 1).unwrap(), GraphNodeKind::Empty);
        let err = sim.graph_node_kind(g, 2).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("graph dependency"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_if_default_skips_or_runs_body() {
        let long = KernelKind::other(1 << 40, 4096);
        let run = |default: u32| {
            let mut sim = Sim::new(h100());
            let d = DeviceId(0);
            let s = StreamId(0);
            let a = sim.alloc(d, 4096, s).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
            sim.synchronize().unwrap();
            let g = sim.create_graph(d, s).unwrap();
            let h = sim.graph_conditional_create(g, default).unwrap();
            let body = sim.graph_add_if(g, h).unwrap();
            sim.graph_add_kernel(body, long.clone(), &[a], &[a])
                .unwrap();
            assert_eq!(sim.graph_node_kind(g, 0).unwrap(), GraphNodeKind::If);
            let _ = sim.instantiate_graph(g).unwrap();
            assert!(sim.graph_instantiated(body).unwrap());
            sim.upload_graph(g).unwrap();
            let t0 = sim.clock_ns();
            let n = sim.launch_graph(g, s).unwrap();
            sim.synchronize().unwrap();
            assert_eq!(n, 1);
            sim.clock_ns().saturating_sub(t0)
        };
        let skipped = run(0);
        let ran = run(1);
        assert!(
            skipped < ran,
            "IF default 0 must skip the body GEMM; skip={skipped} run={ran}"
        );
    }

    #[test]
    fn graph_set_conditional_enables_if_body() {
        let long = KernelKind::other(1 << 40, 4096);
        let run = |write: bool| {
            let mut sim = Sim::new(h100());
            let d = DeviceId(0);
            let s = StreamId(0);
            let a = sim.alloc(d, 4096, s).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
            sim.synchronize().unwrap();
            let g = sim.create_graph(d, s).unwrap();
            let h = sim.graph_conditional_create(g, 0).unwrap();
            sim.begin_capture_to_graph(d, s, g, &[]).unwrap();
            if write {
                enq(sim.set_conditional(d, h, 1, s));
            } else {
                enq(sim.kernel(d, KernelKind::other(8, 4096), &[a], &[a], s));
            }
            assert_eq!(sim.end_capture().unwrap(), g);
            let body = sim.graph_add_if(g, h).unwrap();
            sim.graph_add_kernel(body, long.clone(), &[a], &[a])
                .unwrap();
            sim.graph_add_dependencies(g, 0, 1).unwrap();
            let _ = sim.instantiate_graph(g).unwrap();
            sim.upload_graph(g).unwrap();
            let t0 = sim.clock_ns();
            let n = sim.launch_graph(g, s).unwrap();
            sim.synchronize().unwrap();
            assert_eq!(n, 2);
            sim.clock_ns().saturating_sub(t0)
        };
        let enabled = run(true);
        let skipped = run(false);
        assert!(
            skipped < enabled,
            "device set_conditional(1) must run the IF body; set={enabled} skip={skipped}"
        );
    }

    #[test]
    fn graph_add_if_rejects_instantiated_capture_and_mismatch() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let g = sim.create_graph(d, s).unwrap();
        let other = sim.create_graph(d, s).unwrap();
        let h = sim.graph_conditional_create(g, 0).unwrap();
        let err = sim.graph_add_if(other, h).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("mismatch"), "{why}"),
            e => panic!("{e:?}"),
        }
        let body = sim.graph_add_if(g, h).unwrap();
        assert_eq!(sim.graph_if_nodes(g).unwrap().len(), 1);
        let exec = sim.instantiate_graph(g).unwrap();
        let err = sim.graph_add_if(exec, h).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("instantiated"), "{why}"),
            e => panic!("{e:?}"),
        }
        let err = sim.graph_conditional_create(exec, 1).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("instantiated"), "{why}"),
            e => panic!("{e:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        let err = sim.graph_conditional_create(body, 0).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            e => panic!("{e:?}"),
        }
        let _end = sim.end_capture().unwrap();
    }

    #[test]
    fn graph_while_one_iter_then_clear() {
        let long = KernelKind::other(1 << 40, 4096);
        let run = |default: u32| {
            let mut sim = Sim::new(h100());
            let d = DeviceId(0);
            let s = StreamId(0);
            let a = sim.alloc(d, 4096, s).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
            sim.synchronize().unwrap();
            let g = sim.create_graph(d, s).unwrap();
            let h = sim.graph_conditional_create(g, default).unwrap();
            let body = sim.graph_add_while(g, h).unwrap();
            sim.graph_add_kernel(body, long.clone(), &[a], &[a])
                .unwrap();
            sim.begin_capture_to_graph(d, s, body, &[0]).unwrap();
            enq(sim.set_conditional(d, h, 0, s));
            assert_eq!(sim.end_capture().unwrap(), body);
            assert_eq!(sim.graph_node_kind(g, 0).unwrap(), GraphNodeKind::While);
            let _ = sim.instantiate_graph(g).unwrap();
            sim.upload_graph(g).unwrap();
            let t0 = sim.clock_ns();
            let _n = sim.launch_graph(g, s).unwrap();
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let skipped = run(0);
        let once = run(1);
        assert!(
            skipped < once,
            "WHILE default 0 must skip the body; skip={skipped} once={once}"
        );
    }

    #[test]
    fn graph_while_unclear_hits_cap() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        let g = sim.create_graph(d, s).unwrap();
        let h = sim.graph_conditional_create(g, 1).unwrap();
        let body = sim.graph_add_while(g, h).unwrap();
        sim.graph_add_kernel(body, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim.instantiate_graph(g).unwrap();
        let _n = sim.launch_graph(g, s).unwrap();
        let err = sim.synchronize().unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("while iteration cap"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_switch_selects_branch() {
        let long = KernelKind::other(1 << 40, 4096);
        let tiny = KernelKind::other(1 << 30, 4096);
        let run = |default: u32| {
            let mut sim = Sim::new(h100());
            let d = DeviceId(0);
            let s = StreamId(0);
            let a = sim.alloc(d, 4096, s).unwrap();
            let b = sim.alloc(d, 4096, s).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
            enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
            sim.synchronize().unwrap();
            let g = sim.create_graph(d, s).unwrap();
            let h = sim.graph_conditional_create(g, default).unwrap();
            let bodies = sim.graph_add_switch(g, h, 2).unwrap();
            sim.graph_add_kernel(bodies[0], long.clone(), &[a], &[a])
                .unwrap();
            sim.graph_add_kernel(bodies[1], tiny.clone(), &[b], &[b])
                .unwrap();
            assert_eq!(sim.graph_node_kind(g, 0).unwrap(), GraphNodeKind::Switch);
            let _ = sim.instantiate_graph(g).unwrap();
            sim.upload_graph(g).unwrap();
            let t0 = sim.clock_ns();
            let n = sim.launch_graph(g, s).unwrap();
            sim.synchronize().unwrap();
            assert_eq!(n, 2);
            sim.clock_ns().saturating_sub(t0)
        };
        let branch0 = run(0);
        let branch1 = run(1);
        let none = run(2);
        assert!(
            branch1 < branch0,
            "SWITCH 1 must run the tiny body; b0={branch0} b1={branch1}"
        );
        assert!(
            none < branch1,
            "SWITCH out of range must skip both bodies; none={none} b1={branch1}"
        );
    }

    #[test]
    fn graph_add_switch_rejects_zero_branches() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let g = sim.create_graph(d, s).unwrap();
        let h = sim.graph_conditional_create(g, 0).unwrap();
        let err = sim.graph_add_switch(g, h, 0).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("switch branches"), "{why}"),
            other => panic!("{other:?}"),
        }
        let err = sim.graph_add_switch(g, h, 65).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("switch branches"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_add_while_rejects_instantiated_capture_and_mismatch() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let g = sim.create_graph(d, s).unwrap();
        let other = sim.create_graph(d, s).unwrap();
        let h = sim.graph_conditional_create(g, 0).unwrap();
        let err = sim.graph_add_while(other, h).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("mismatch"), "{why}"),
            e => panic!("{e:?}"),
        }
        let body = sim.graph_add_while(g, h).unwrap();
        assert_eq!(sim.graph_while_nodes(g).unwrap().len(), 1);
        let exec = sim.instantiate_graph(g).unwrap();
        let err = sim.graph_add_while(exec, h).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("instantiated"), "{why}"),
            e => panic!("{e:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        let err = sim.graph_add_while(body, h).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            e => panic!("{e:?}"),
        }
        let _end = sim.end_capture().unwrap();
    }

    #[test]
    fn clone_graph_remaps_while_and_switch_handles() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        let g = sim.create_graph(d, s).unwrap();
        let hw = sim.graph_conditional_create(g, 0).unwrap();
        let hs = sim.graph_conditional_create(g, 1).unwrap();
        let wbody = sim.graph_add_while(g, hw).unwrap();
        sim.graph_add_kernel(wbody, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let bodies = sim.graph_add_switch(g, hs, 2).unwrap();
        sim.graph_add_kernel(bodies[0], KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_kernel(bodies[1], KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let orig_w = sim.graph_while_nodes(g).unwrap();
        let orig_s = sim.graph_switch_nodes(g).unwrap();
        let cloned = sim.clone_graph(g).unwrap();
        sim.destroy_graph(g).unwrap();
        let cw = sim.graph_while_nodes(cloned).unwrap();
        let cs = sim.graph_switch_nodes(cloned).unwrap();
        assert_eq!(cw.len(), 1);
        assert_eq!(cs.len(), 1);
        assert_ne!(cw[0].1, orig_w[0].1);
        assert_ne!(cw[0].2, orig_w[0].2);
        assert_ne!(cs[0].1, orig_s[0].1);
        assert_ne!(cs[0].2, orig_s[0].2);
        let _ = sim.instantiate_graph(cloned).unwrap();
        let n = sim.launch_graph(cloned, s).unwrap();
        sim.synchronize().unwrap();
        assert!(
            n >= 2,
            "clone must expand WHILE skip + SWITCH branch; n={n}"
        );
    }

    #[test]
    fn graph_node_find_in_clone_maps_kernel_and_rejects() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let orig = sim.end_capture().unwrap();
        let clone = sim.clone_graph(orig).unwrap();
        let again = sim.clone_graph(clone).unwrap();
        sim.begin_capture(d, s).unwrap();
        assert_eq!(sim.graph_node_find_in_clone(orig, 0, clone).unwrap(), 0);
        let _end = sim.end_capture().unwrap();
        let err = sim.graph_node_find_in_clone(orig, 0, again).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not a clone"), "{why}"),
            e => panic!("{e:?}"),
        }
        assert_eq!(sim.graph_node_find_in_clone(clone, 0, again).unwrap(), 0);
        let err = sim.graph_node_find_in_clone(orig, 0, orig).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not a clone"), "{why}"),
            e => panic!("{e:?}"),
        }
        let err = sim.graph_node_find_in_clone(orig, 1, clone).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("unknown graph node"), "{why}"),
            e => panic!("{e:?}"),
        }
        let other = sim.create_graph(d, s).unwrap();
        let err = sim.graph_node_find_in_clone(orig, 0, other).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not a clone"), "{why}"),
            e => panic!("{e:?}"),
        }
        let _ = sim.instantiate_graph(clone).unwrap();
        let node = sim.graph_node_find_in_clone(orig, 0, clone).unwrap();
        let (_, mut params) = sim.graph_unique_kernel(clone).unwrap();
        params.reads = vec![KernelBuf::whole(b)];
        params.writes = vec![KernelBuf::whole(b)];
        sim.graph_exec_kernel_set_params(clone, node, &params)
            .unwrap();
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(clone, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        let _ = sim.instantiate_graph(orig).unwrap();
        let _n = sim.launch_graph(orig, s).unwrap();
        let err = sim.synchronize().unwrap_err();
        match err {
            SimError::NotResident { alloc, device } => {
                assert_eq!(alloc, a);
                assert_eq!(device, d);
            }
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn graph_node_find_in_clone_nested_child() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let child = sim.end_capture().unwrap();
        let _ = sim.instantiate_graph(child).unwrap();
        let parent = sim.create_graph(d, s).unwrap();
        sim.graph_add_child(parent, child).unwrap();
        let cloned_parent = sim.clone_graph(parent).unwrap();
        let kids = sim.graph_child_nodes(cloned_parent).unwrap();
        assert_eq!(kids.len(), 1);
        let cloned_child = kids[0].1;
        assert_eq!(
            sim.graph_node_find_in_clone(parent, 0, cloned_parent)
                .unwrap(),
            0
        );
        assert_eq!(
            sim.graph_node_find_in_clone(child, 0, cloned_child)
                .unwrap(),
            0
        );
        let err = sim
            .graph_node_find_in_clone(child, 0, cloned_parent)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not a clone"), "{why}"),
            e => panic!("{e:?}"),
        }
        sim.destroy_graph(parent).unwrap();
        let err = sim
            .graph_node_find_in_clone(parent, 0, cloned_parent)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("unknown graph"), "{why}"),
            e => panic!("{e:?}"),
        }
        let n = sim.launch_graph(cloned_parent, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
    }

    #[test]
    fn graph_add_independent_kernels_hyperq_overlap() {
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |slots: u8, chain: bool| {
            let mut sim = Sim::new(h100().with_compute_slots(slots));
            let d = DeviceId(0);
            let s = StreamId(0);
            let a = sim.alloc(d, 4096, s).unwrap();
            let b = sim.alloc(d, 4096, s).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
            enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
            sim.synchronize().unwrap();
            let g = sim.create_graph(d, s).unwrap();
            sim.graph_add_kernel(g, kind.clone(), &[a], &[a]).unwrap();
            sim.graph_add_kernel(g, kind.clone(), &[b], &[b]).unwrap();
            if chain {
                sim.graph_add_dependencies(g, 0, 1).unwrap();
            }
            let _ = sim.instantiate_graph(g).unwrap();
            sim.upload_graph(g).unwrap();
            let t0 = sim.clock_ns();
            let n = sim.launch_graph(g, s).unwrap();
            sim.synchronize().unwrap();
            assert_eq!(n, 2);
            sim.clock_ns().saturating_sub(t0)
        };
        let serial = run(2, true);
        let overlap = run(2, false);
        assert!(
            overlap < serial,
            "independent graph nodes must Hyper-Q overlap; overlap={overlap} serial={serial}"
        );
        let exclusive = run(1, false);
        assert!(
            exclusive > overlap,
            "one compute slot must serialize independent nodes; exclusive={exclusive} overlap={overlap}"
        );
    }

    #[test]
    fn graph_add_independent_children_overlap_capture_chain() {
        let kind = KernelKind::other(1 << 40, 4096);
        let mut cap = Sim::new(h100().with_compute_slots(2));
        let mut bld = Sim::new(h100().with_compute_slots(2));
        let d = DeviceId(0);
        let s = StreamId(0);
        let setup = |sim: &mut Sim| {
            let a = sim.alloc(d, 4096, s).unwrap();
            let b = sim.alloc(d, 4096, s).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
            enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
            sim.synchronize().unwrap();
            (a, b)
        };
        let (ca, cb) = setup(&mut cap);
        let leaf0 = {
            cap.begin_capture(d, s).unwrap();
            enq(cap.kernel(d, kind.clone(), &[ca], &[ca], s));
            let g = cap.end_capture().unwrap();
            let _ = cap.instantiate_graph(g).unwrap();
            g
        };
        let leaf1 = {
            cap.begin_capture(d, s).unwrap();
            enq(cap.kernel(d, kind.clone(), &[cb], &[cb], s));
            let g = cap.end_capture().unwrap();
            let _ = cap.instantiate_graph(g).unwrap();
            g
        };
        cap.begin_capture(d, s).unwrap();
        let _n0 = cap.launch_graph(leaf0, s).unwrap();
        let _n1 = cap.launch_graph(leaf1, s).unwrap();
        let captured = cap.end_capture().unwrap();
        let _ = cap.instantiate_graph(captured).unwrap();
        cap.upload_graph(captured).unwrap();

        let (ba, bb) = setup(&mut bld);
        let l0 = bld.create_graph(d, s).unwrap();
        bld.graph_add_kernel(l0, kind.clone(), &[ba], &[ba])
            .unwrap();
        let _ = bld.instantiate_graph(l0).unwrap();
        let l1 = bld.create_graph(d, s).unwrap();
        bld.graph_add_kernel(l1, kind, &[bb], &[bb]).unwrap();
        let _ = bld.instantiate_graph(l1).unwrap();
        let parent = bld.create_graph(d, s).unwrap();
        bld.graph_add_child(parent, l0).unwrap();
        bld.graph_add_child(parent, l1).unwrap();
        let _ = bld.instantiate_graph(parent).unwrap();
        bld.upload_graph(parent).unwrap();

        let t0 = cap.clock_ns();
        let n0 = cap.launch_graph(captured, s).unwrap();
        cap.synchronize().unwrap();
        let cap_ns = cap.clock_ns().saturating_sub(t0);
        let t1 = bld.clock_ns();
        let n1 = bld.launch_graph(parent, s).unwrap();
        bld.synchronize().unwrap();
        let bld_ns = bld.clock_ns().saturating_sub(t1);
        assert_eq!(n0, 2);
        assert_eq!(n1, 2);
        assert!(
            bld_ns < cap_ns,
            "graph_add_child without deps must overlap; build={bld_ns} capture={cap_ns}"
        );
    }

    #[test]
    fn update_graph_treats_dependencies_as_topology() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
        sim.synchronize().unwrap();
        let exec = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(exec, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_kernel(exec, KernelKind::other(8, 8), &[b], &[b])
            .unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        let src = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(src, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_kernel(src, KernelKind::other(8, 8), &[b], &[b])
            .unwrap();
        sim.graph_add_dependencies(src, 0, 1).unwrap();
        let err = sim.update_graph(exec, src).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("topology"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    fn map_whole(sim: &mut Sim, device: DeviceId, bytes: u64) -> AllocId {
        let va = sim.va_reserve(bytes).unwrap();
        sim.va_map(va, device).unwrap();
        va
    }

    #[test]
    fn multicast_pcie_team_is_invalid() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let bytes = 4096u64;
        let mc = sim.multicast_create(bytes, 2).unwrap();
        sim.multicast_add_device(mc, DeviceId(0)).unwrap();
        let err = sim.multicast_add_device(mc, DeviceId(1)).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("NVLink"), "{why}"),
            other => panic!("{other:?}"),
        }
        let mut one = Sim::new(h100());
        let mc = one.multicast_create(bytes, 2).unwrap();
        one.multicast_add_device(mc, DeviceId(0)).unwrap();
        let err = one.multicast_add_device(mc, DeviceId(1)).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("device"), "{why}"),
            other => panic!("{other:?}"),
        }
        let err = one.multicast_create(bytes, 1).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("NVLink"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn multicast_granularity_and_capture_refused() {
        let mut sim =
            Sim::new(HardwareProfile::example_8xh100_nvlink().with_multicast_granularity(1 << 21));
        assert_eq!(
            sim.multicast_get_granularity(MulticastGranularity::MINIMUM)
                .unwrap(),
            1 << 21
        );
        assert_eq!(
            sim.multicast_get_granularity(MulticastGranularity::RECOMMENDED)
                .unwrap(),
            1 << 21
        );
        match sim.multicast_get_granularity(2) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("multicast granularity flags"), "{why}")
            }
            other => panic!("{other:?}"),
        }
        let err = sim.multicast_create(4096, 2).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("unaligned"), "{why}"),
            other => panic!("{other:?}"),
        }
        let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink());
        assert_eq!(
            sim.multicast_get_granularity(MulticastGranularity::MINIMUM)
                .unwrap(),
            1
        );
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        assert_eq!(
            sim.multicast_get_granularity(MulticastGranularity::MINIMUM)
                .unwrap(),
            1
        );
        let err = sim.multicast_create(4096, 2).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn multicast_unbind_drops_bind_until_remap() {
        let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let bytes = 4096u64;
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let mc = sim.multicast_create(bytes, 2).unwrap();
        sim.multicast_add_device(mc, d0).unwrap();
        sim.multicast_add_device(mc, d1).unwrap();
        let h0 = sim.va_create(d0, bytes).unwrap();
        let h1 = sim.va_create(d1, bytes).unwrap();
        sim.multicast_bind_mem(mc, d0, h0).unwrap();
        sim.multicast_bind_mem(mc, d1, h1).unwrap();
        assert_eq!(sim.multicast_binds(mc).unwrap(), 2);
        sim.multicast_unbind(mc, d1).unwrap();
        assert_eq!(sim.multicast_binds(mc).unwrap(), 1);
        match sim.multicast_unbind(mc, d1) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not bound"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.multicast_bind_mem(mc, d1, h1).unwrap();
        assert_eq!(sim.multicast_binds(mc).unwrap(), 2);
        let va = sim.va_reserve(bytes).unwrap();
        sim.va_map_multicast(va, d0, 0, mc).unwrap();
        match sim.multicast_unbind(mc, d0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("mapped"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_unmap(va).unwrap();
        sim.multicast_unbind(mc, d0).unwrap();
        assert_eq!(sim.multicast_binds(mc).unwrap(), 1);
        sim.begin_capture(d0, StreamId(0)).unwrap();
        match sim.multicast_unbind(mc, d1) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn multicast_unbind_with_size_is_cu_multicast_unbind() {
        let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let bytes = 4096u64;
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let mc = sim.multicast_create(bytes, 2).unwrap();
        sim.multicast_add_device(mc, d0).unwrap();
        sim.multicast_add_device(mc, d1).unwrap();
        let h0 = sim.va_create(d0, bytes).unwrap();
        let h1 = sim.va_create(d1, bytes).unwrap();
        sim.multicast_bind_mem(mc, d0, h0).unwrap();
        sim.multicast_bind_mem(mc, d1, h1).unwrap();
        match sim.multicast_unbind_with_size(mc, d1, 2048) {
            Err(SimError::Invalid { why }) => assert!(why.contains("unbind size"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(sim.multicast_binds(mc).unwrap(), 2);
        sim.multicast_unbind_with_size(mc, d1, bytes).unwrap();
        assert_eq!(sim.multicast_binds(mc).unwrap(), 1);
        sim.begin_capture(d0, StreamId(0)).unwrap();
        match sim.multicast_unbind_with_size(mc, d0, bytes) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn multicast_destroy_releases_object() {
        let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let bytes = 4096u64;
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let mc = sim.multicast_create(bytes, 2).unwrap();
        sim.multicast_add_device(mc, d0).unwrap();
        sim.multicast_add_device(mc, d1).unwrap();
        let h0 = sim.va_create(d0, bytes).unwrap();
        let h1 = sim.va_create(d1, bytes).unwrap();
        sim.multicast_bind_mem(mc, d0, h0).unwrap();
        sim.multicast_bind_mem(mc, d1, h1).unwrap();
        sim.multicast_destroy(mc).unwrap();
        match sim.multicast_binds(mc) {
            Err(SimError::Invalid { why }) => assert!(why.contains("unknown multicast"), "{why}"),
            other => panic!("{other:?}"),
        }
        let mc = sim.multicast_create(bytes, 2).unwrap();
        sim.multicast_add_device(mc, d0).unwrap();
        sim.multicast_add_device(mc, d1).unwrap();
        sim.multicast_bind_mem(mc, d0, h0).unwrap();
        sim.multicast_bind_mem(mc, d1, h1).unwrap();
        let va = sim.va_reserve(bytes).unwrap();
        sim.va_map_multicast(va, d0, 0, mc).unwrap();
        match sim.multicast_destroy(mc) {
            Err(SimError::Invalid { why }) => assert!(why.contains("mapped"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_unmap(va).unwrap();
        sim.multicast_destroy(mc).unwrap();
        let mc = sim.multicast_create(bytes, 2).unwrap();
        sim.begin_capture(d0, StreamId(0)).unwrap();
        match sim.multicast_destroy(mc) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn va_map_multicast_with_flags_is_cu_mem_map() {
        let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let bytes = 4096u64;
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let mc = sim.multicast_create(bytes, 2).unwrap();
        sim.multicast_add_device(mc, d0).unwrap();
        sim.multicast_add_device(mc, d1).unwrap();
        let h0 = sim.va_create(d0, bytes).unwrap();
        let h1 = sim.va_create(d1, bytes).unwrap();
        sim.multicast_bind_mem(mc, d0, h0).unwrap();
        sim.multicast_bind_mem(mc, d1, h1).unwrap();
        let va = sim.va_reserve(bytes).unwrap();
        let va2 = sim.va_reserve(bytes).unwrap();
        match sim.va_map_multicast_with_flags(va, d0, 0, mc, 1) {
            Err(SimError::Invalid { why }) => assert!(why.contains("mem map flags"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_map_multicast_with_flags(va, d0, 0, mc, MemMapFlags::DEFAULT)
            .unwrap();
        assert!(sim.is_multicast_va(va));
        sim.begin_capture(d0, StreamId(0)).unwrap();
        match sim.va_map_multicast_with_flags(va2, d0, 0, mc, MemMapFlags::DEFAULT) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
        sim.va_free(va2).unwrap();
    }

    #[test]
    fn va_map_multicast_with_size_is_cu_mem_map() {
        let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let bytes = 4096u64;
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let mc = sim.multicast_create(bytes, 2).unwrap();
        sim.multicast_add_device(mc, d0).unwrap();
        sim.multicast_add_device(mc, d1).unwrap();
        let h0 = sim.va_create(d0, bytes).unwrap();
        let h1 = sim.va_create(d1, bytes).unwrap();
        sim.multicast_bind_mem(mc, d0, h0).unwrap();
        sim.multicast_bind_mem(mc, d1, h1).unwrap();
        let va = sim.va_reserve(bytes).unwrap();
        let va2 = sim.va_reserve(bytes).unwrap();
        match sim.va_map_multicast_with_size(va, d0, 0, mc, 2048, MemMapFlags::DEFAULT) {
            Err(SimError::Invalid { why }) => assert!(why.contains("mem map size"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_map_multicast_with_size(va, d0, 0, mc, bytes, MemMapFlags::DEFAULT)
            .unwrap();
        assert!(sim.is_multicast_va(va));
        sim.begin_capture(d0, StreamId(0)).unwrap();
        match sim.va_map_multicast_with_size(va2, d0, 0, mc, bytes, MemMapFlags::DEFAULT) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
        sim.va_free(va2).unwrap();
    }

    #[test]
    fn multicast_bind_mem_with_flags_is_cu_multicast_bind_mem() {
        let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let bytes = 4096u64;
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let mc = sim.multicast_create(bytes, 2).unwrap();
        sim.multicast_add_device(mc, d0).unwrap();
        sim.multicast_add_device(mc, d1).unwrap();
        let h0 = sim.va_create(d0, bytes).unwrap();
        let h1 = sim.va_create(d1, bytes).unwrap();
        sim.multicast_bind_mem_with_flags(mc, d0, h0, MulticastBindFlags::DEFAULT)
            .unwrap();
        sim.multicast_bind_mem(mc, d1, h1).unwrap();
        assert_eq!(sim.multicast_binds(mc).unwrap(), 2);
        match sim.multicast_bind_mem_with_flags(mc, d0, h0, 1) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("multicast bind flags"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d0, StreamId(0)).unwrap();
        match sim.multicast_bind_mem_with_flags(mc, d0, h0, MulticastBindFlags::DEFAULT) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn multicast_bind_mem_with_size_is_cu_multicast_bind_mem() {
        let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let bytes = 4096u64;
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let mc = sim.multicast_create(bytes, 2).unwrap();
        sim.multicast_add_device(mc, d0).unwrap();
        sim.multicast_add_device(mc, d1).unwrap();
        let h0 = sim.va_create(d0, bytes).unwrap();
        let h1 = sim.va_create(d1, bytes).unwrap();
        match sim.multicast_bind_mem_with_size(mc, d0, h0, 2048, MulticastBindFlags::DEFAULT) {
            Err(SimError::Invalid { why }) => assert!(why.contains("bind size"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(sim.multicast_binds(mc).unwrap(), 0);
        sim.multicast_bind_mem_with_size(mc, d0, h0, bytes, MulticastBindFlags::DEFAULT)
            .unwrap();
        sim.multicast_bind_mem(mc, d1, h1).unwrap();
        assert_eq!(sim.multicast_binds(mc).unwrap(), 2);
        sim.begin_capture(d0, StreamId(0)).unwrap();
        match sim.multicast_bind_mem_with_size(mc, d0, h0, bytes, MulticastBindFlags::DEFAULT) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn multicast_bind_addr_retains_mapped_va() {
        let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let bytes = 4096u64;
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let va = sim.va_reserve(bytes).unwrap();
        sim.va_map(va, d0).unwrap();
        sim.va_map(va, d1).unwrap();
        let mc = sim.multicast_create(bytes, 2).unwrap();
        sim.multicast_add_device(mc, d0).unwrap();
        sim.multicast_add_device(mc, d1).unwrap();
        sim.multicast_bind_addr(mc, d0, va).unwrap();
        sim.multicast_bind_addr(mc, d1, va).unwrap();
        assert_eq!(sim.multicast_binds(mc).unwrap(), 2);
        match sim.multicast_bind_addr_with_flags(mc, d0, va, 1) {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("multicast bind flags"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        let malloc = sim.malloc(d0, bytes).unwrap();
        match sim.multicast_bind_addr(mc, d0, malloc) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not a VA"), "{why}"),
            other => panic!("{other:?}"),
        }
        let hole = sim.va_reserve(bytes).unwrap();
        match sim.multicast_bind_addr(mc, d0, hole) {
            Err(SimError::Invalid { why }) => assert!(why.contains("no such map"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d0, StreamId(0)).unwrap();
        match sim.multicast_bind_addr(mc, d0, va) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn multicast_bind_addr_with_size_is_cu_multicast_bind_addr() {
        let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let bytes = 4096u64;
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let va = sim.va_reserve(bytes).unwrap();
        sim.va_map(va, d0).unwrap();
        sim.va_map(va, d1).unwrap();
        let mc = sim.multicast_create(bytes, 2).unwrap();
        sim.multicast_add_device(mc, d0).unwrap();
        sim.multicast_add_device(mc, d1).unwrap();
        match sim.multicast_bind_addr_with_size(mc, d0, va, 2048, MulticastBindFlags::DEFAULT) {
            Err(SimError::Invalid { why }) => assert!(why.contains("bind size"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(sim.multicast_binds(mc).unwrap(), 0);
        sim.multicast_bind_addr_with_size(mc, d0, va, bytes, MulticastBindFlags::DEFAULT)
            .unwrap();
        sim.multicast_bind_addr(mc, d1, va).unwrap();
        assert_eq!(sim.multicast_binds(mc).unwrap(), 2);
        sim.begin_capture(d0, StreamId(0)).unwrap();
        match sim.multicast_bind_addr_with_size(mc, d0, va, bytes, MulticastBindFlags::DEFAULT) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn multicast_bind_map_kernel_is_one_nvls_hop() {
        let p = HardwareProfile::example_8xh100_nvlink();
        let bytes = 1u64 << 20;
        let src = DeviceId(0);
        let s = StreamId(0);
        let dests: Vec<DeviceId> = (1..8u16).map(DeviceId).collect();
        let mut d2d = Sim::new(p.clone());
        let mut nvls = Sim::new(p);
        let setup = |sim: &mut Sim| {
            let va = map_whole(sim, src, bytes);
            for &d in &dests {
                sim.va_map(va, d).unwrap();
            }
            enq(sim.memcpy_pinned_to_device(src, va, bytes, s));
            sim.synchronize().unwrap();
            va
        };
        let a = setup(&mut d2d);
        let b = setup(&mut nvls);
        let t0 = d2d.clock_ns();
        let moved0 = d2d.bytes_moved();
        for &d in &dests {
            enq(d2d.memcpy_device_to_device(src, d, a, bytes, s));
        }
        d2d.synchronize().unwrap();
        let d2d_ns = d2d.clock_ns().saturating_sub(t0);
        let d2d_moved = d2d.bytes_moved().saturating_sub(moved0);
        let t1 = nvls.clock_ns();
        let moved1 = nvls.bytes_moved();
        enq(nvls.multicast_store(src, b, &dests, s));
        nvls.synchronize().unwrap();
        let nvls_ns = nvls.clock_ns().saturating_sub(t1);
        let nvls_moved = nvls.bytes_moved().saturating_sub(moved1);
        assert!(
            nvls_ns < d2d_ns,
            "nvls={nvls_ns} d2d={d2d_ns} (NVLS must beat 7 sequential D2Ds)"
        );
        assert_eq!(d2d_moved, bytes.saturating_mul(7));
        assert_eq!(nvls_moved, bytes);
        assert!(nvls.operations().any(|o| matches!(
            o.kind,
            GpuOp::Kernel {
                kind: KernelKind::Other { flops: 0, bytes: n },
                ..
            } if n == bytes
        )));
        assert_eq!(nvls.hbm_used(DeviceId(7)).unwrap(), bytes);
        assert_eq!(d2d.hbm_used(DeviceId(7)).unwrap(), bytes);
    }

    #[test]
    fn multicast_kernel_blocks_compute_unlike_d2d() {
        let p = HardwareProfile::parse(
            "gpus=2\nfp16_flops=1000000\nhbm_bps=1000000000000\ncompute_slots=1\nlaunch_overhead_ns=1\n",
        )
        .unwrap();
        assert!(p.has_nvlink());
        let bytes = 4096u64;
        let src = DeviceId(0);
        let dst = DeviceId(1);
        let copy = StreamId(0);
        let compute = StreamId(1);
        let gemm = KernelKind::other(1_000_000_000, 8);
        let run = |nvls: bool| {
            let mut sim = Sim::new(p.clone());
            let va = map_whole(&mut sim, src, bytes);
            sim.va_map(va, dst).unwrap();
            enq(sim.memcpy_pinned_to_device(src, va, bytes, copy));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(src, gemm.clone(), &[va], &[va], compute));
            if nvls {
                enq(sim.multicast_store(src, va, &[dst], copy));
            } else {
                enq(sim.memcpy_device_to_device(src, dst, va, bytes, copy));
            }
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let d2d_ns = run(false);
        let nvls_ns = run(true);
        assert!(
            nvls_ns > d2d_ns,
            "exclusive compute: NVLS kernel must serialize with GEMM, nvls={nvls_ns} d2d={d2d_ns}"
        );
    }

    fn wait_op(sim: &Sim) -> Operation {
        sim.operations()
            .find(|o| matches!(o.kind, GpuOp::WaitValue { .. }))
            .expect("wait-value op")
    }

    fn write_op(sim: &Sim) -> Operation {
        sim.operations()
            .find(|o| matches!(o.kind, GpuOp::WriteValue { .. }))
            .expect("write-value op")
    }

    #[test]
    fn wait_eq_unsatisfied_deadlocks_synchronize() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 64).unwrap();
        enq(sim.wait_value64(d, a, 0, 1, WaitValueCmp::Eq, s));
        let err = sim.synchronize().unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("deadlock"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn wait_eq_unblocks_after_write_on_other_stream_without_event() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let wait_s = StreamId(0);
        let write_s = StreamId(1);
        let a = sim.malloc(d, 64).unwrap();
        enq(sim.wait_value64(d, a, 0, 1, WaitValueCmp::Eq, wait_s));
        enq(sim.write_value64(d, a, 0, 1, write_s));
        sim.synchronize().unwrap();
        let wait = wait_op(&sim);
        let write = write_op(&sim);
        assert!(wait.done);
        assert!(write.done);
        assert!(wait.start_ns.unwrap() >= write.done_ns.unwrap());
    }

    #[test]
    fn wait_then_write_same_stream_deadlocks() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 64).unwrap();
        enq(sim.wait_value64(d, a, 0, 1, WaitValueCmp::Eq, s));
        enq(sim.write_value64(d, a, 0, 1, s));
        let err = sim.synchronize().unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("deadlock"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn wait_geq_and_nor_need_later_write() {
        let d = DeviceId(0);
        let wait_s = StreamId(0);
        let write_s = StreamId(1);
        let run = |cmp: WaitValueCmp, first: u64, second: u64, want: u64| {
            let mut sim = Sim::new(h100());
            let a = sim.malloc(d, 64).unwrap();
            enq(sim.wait_value64(d, a, 0, want, cmp, wait_s));
            enq(sim.write_value64(d, a, 0, first, write_s));
            sim.synchronize_stream(d, write_s).unwrap();
            let err = sim.synchronize_stream(d, wait_s);
            match err {
                Err(SimError::Invalid { why }) => assert!(why.contains("deadlock"), "{why}"),
                other => panic!("first write must not unblock: {other:?}"),
            }
            let mut sim = Sim::new(h100());
            let a = sim.malloc(d, 64).unwrap();
            enq(sim.wait_value64(d, a, 0, want, cmp, wait_s));
            enq(sim.write_value64(d, a, 0, first, write_s));
            enq(sim.write_value64(d, a, 0, second, write_s));
            sim.synchronize().unwrap();
        };
        run(WaitValueCmp::Geq, 3, 5, 5);
        run(WaitValueCmp::And, 1, 4, 4);
        run(WaitValueCmp::Nor, 1, 3, 3);
    }

    #[test]
    fn write32_masks_high_bits_for_wait64() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let wait_s = StreamId(0);
        let write_s = StreamId(1);
        let a = sim.malloc(d, 64).unwrap();
        enq(sim.wait_value64(d, a, 0, 0x1_0000_0001, WaitValueCmp::Eq, wait_s));
        enq(sim.write_value32(d, a, 0, 0x1_0000_0001, write_s));
        let err = sim.synchronize().unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("deadlock"), "{why}"),
            other => panic!("{other:?}"),
        }
        let mut sim = Sim::new(h100());
        let a = sim.malloc(d, 64).unwrap();
        enq(sim.wait_value32(d, a, 0, 1, WaitValueCmp::Eq, wait_s));
        enq(sim.write_value32(d, a, 0, 0x1_0000_0001, write_s));
        sim.synchronize().unwrap();
    }

    #[test]
    fn wait_value_alignment_and_span() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 16).unwrap();
        let err = sim
            .wait_value64(d, a, 1, 1, WaitValueCmp::Eq, s)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("alignment"), "{why}"),
            other => panic!("{other:?}"),
        }
        let err = sim
            .wait_value64(d, a, 16, 1, WaitValueCmp::Eq, s)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("span"), "{why}"),
            other => panic!("{other:?}"),
        }
        let err = sim.write_value32(d, a, 2, 1, s).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("alignment"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn wait_on_pageable_host_is_not_resident() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let h = sim.alloc_host(64).unwrap();
        enq(sim.wait_value64(d, h, 0, 1, WaitValueCmp::Eq, s));
        let err = sim.synchronize().unwrap_err();
        match err {
            SimError::NotResident { alloc, device } => {
                assert_eq!(alloc, h);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn memset_does_not_unblock_wait() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let wait_s = StreamId(0);
        let fill_s = StreamId(1);
        let a = sim.malloc(d, 64).unwrap();
        enq(sim.wait_value64(d, a, 0, 1, WaitValueCmp::Eq, wait_s));
        enq(sim.memset(d, a, 64, fill_s));
        let err = sim.synchronize().unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("deadlock"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn mapped_host_wait_value_is_legal() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let wait_s = StreamId(0);
        let write_s = StreamId(1);
        let h = sim.alloc_host_mapped(64).unwrap();
        enq(sim.wait_value64(d, h, 0, 7, WaitValueCmp::Eq, wait_s));
        enq(sim.write_value64(d, h, 0, 7, write_s));
        sim.synchronize().unwrap();
    }

    #[test]
    fn wait_does_not_occupy_compute() {
        let mut sim = Sim::new(h100().with_compute_slots(1));
        let d = DeviceId(0);
        let gemm_s = StreamId(0);
        let wait_s = StreamId(1);
        let write_s = StreamId(2);
        let a = sim.malloc(d, 64).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 40, 4096), &[b], &[b], gemm_s));
        enq(sim.wait_value64(d, a, 0, 1, WaitValueCmp::Eq, wait_s));
        enq(sim.write_value64(d, a, 0, 1, write_s));
        sim.synchronize().unwrap();
        let gemm = sim
            .operations()
            .find(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .expect("kernel");
        let wait = wait_op(&sim);
        assert!(wait.start_ns.unwrap() < gemm.done_ns.unwrap());
        assert!(wait.start_ns.unwrap() >= write_op(&sim).done_ns.unwrap());
    }

    #[test]
    fn graph_add_wait_write_launch() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 64).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_write_value64(g, a, 0, 1).unwrap();
        sim.graph_add_wait_value64(g, a, 0, 1, WaitValueCmp::Eq)
            .unwrap();
        sim.graph_add_dependencies(g, 0, 1).unwrap();
        assert_eq!(
            sim.graph_node_kind(g, 0).unwrap(),
            GraphNodeKind::BatchMemOp
        );
        assert_eq!(
            sim.graph_node_kind(g, 1).unwrap(),
            GraphNodeKind::BatchMemOp
        );
        let _ = sim.instantiate_graph(g).unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
    }

    #[test]
    fn graph_add_batch_mem_op_is_one_node() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 64).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_batch_mem_op(
            g,
            &[
                BatchMemOp::Write {
                    id: a,
                    offset: 0,
                    value: 1,
                    bits32: false,
                },
                BatchMemOp::Wait {
                    id: a,
                    offset: 0,
                    value: 1,
                    bits32: false,
                    cmp: WaitValueCmp::Eq,
                },
            ],
        )
        .unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 1);
        assert_eq!(
            sim.graph_node_kind(g, 0).unwrap(),
            GraphNodeKind::BatchMemOp
        );
        assert!(sim.graph_node_deps(g, 0).unwrap().is_empty());
        let _ = sim.instantiate_graph(g).unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        let err = sim.graph_add_batch_mem_op(g, &[]).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("empty"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn live_batch_write_then_wait_is_one_op() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 64).unwrap();
        let t0 = sim.clock_ns();
        enq(sim.batch_mem_op(
            d,
            s,
            &[
                BatchMemOp::Write {
                    id: a,
                    offset: 0,
                    value: 9,
                    bits32: false,
                },
                BatchMemOp::Wait {
                    id: a,
                    offset: 0,
                    value: 9,
                    bits32: false,
                    cmp: WaitValueCmp::Eq,
                },
            ],
        ));
        sim.synchronize().unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(1));
        let n = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::BatchMem { .. }))
            .count();
        assert_eq!(n, 1);
        let mut seq = Sim::new(h100());
        let a2 = seq.malloc(d, 64).unwrap();
        let t1 = seq.clock_ns();
        enq(seq.write_value64(d, a2, 0, 9, s));
        enq(seq.wait_value64(d, a2, 0, 9, WaitValueCmp::Eq, s));
        seq.synchronize().unwrap();
        assert_eq!(seq.clock_ns(), t1.saturating_add(2));
    }

    #[test]
    fn batch_wait_then_write_does_not_see_later_write() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 64).unwrap();
        enq(sim.batch_mem_op(
            d,
            s,
            &[
                BatchMemOp::Wait {
                    id: a,
                    offset: 0,
                    value: 1,
                    bits32: false,
                    cmp: WaitValueCmp::Eq,
                },
                BatchMemOp::Write {
                    id: a,
                    offset: 0,
                    value: 1,
                    bits32: false,
                },
            ],
        ));
        let err = sim.synchronize().unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("deadlock"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_batch_mem_disable_skips_writes() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let wait_s = StreamId(1);
        let a = sim.malloc(d, 64).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_batch_mem_op(
            g,
            &[BatchMemOp::Write {
                id: a,
                offset: 0,
                value: 1,
                bits32: false,
            }],
        )
        .unwrap();
        let _ = sim.instantiate_graph(g).unwrap();
        sim.graph_node_set_enabled(g, 0, false).unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 0);
        enq(sim.wait_value64(d, a, 0, 1, WaitValueCmp::Eq, wait_s));
        let err = sim.synchronize().unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("deadlock"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_exec_batch_mem_ops_set_params_retargets_vector() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 64).unwrap();
        let b = sim.malloc(d, 64).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_batch_mem_op(
            g,
            &[
                BatchMemOp::Write {
                    id: a,
                    offset: 0,
                    value: 4,
                    bits32: false,
                },
                BatchMemOp::Wait {
                    id: a,
                    offset: 0,
                    value: 4,
                    bits32: false,
                    cmp: WaitValueCmp::Eq,
                },
            ],
        )
        .unwrap();
        let _ = sim.instantiate_graph(g).unwrap();
        sim.graph_batch_mem_ops_set_params(
            g,
            0,
            &[BatchMemOp::Write {
                id: a,
                offset: 0,
                value: 99,
                bits32: false,
            }],
        )
        .unwrap();
        assert_eq!(sim.graph_batch_mem_ops(g, 0).unwrap().len(), 2);
        sim.graph_exec_batch_mem_ops_set_params(
            g,
            0,
            &[
                BatchMemOp::Write {
                    id: b,
                    offset: 0,
                    value: 4,
                    bits32: false,
                },
                BatchMemOp::Wait {
                    id: b,
                    offset: 0,
                    value: 4,
                    bits32: false,
                    cmp: WaitValueCmp::Eq,
                },
            ],
        )
        .unwrap();
        assert_eq!(
            sim.graph_batch_mem_ops(g, 0)
                .unwrap()
                .first()
                .and_then(|op| match *op {
                    BatchMemOp::Write { id, .. } => Some(id),
                    _ => None,
                }),
            Some(b)
        );
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
    }

    #[test]
    fn update_graph_treats_batch_mem_items_as_params() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 64).unwrap();
        let b = sim.malloc(d, 64).unwrap();
        let exec = sim.create_graph(d, s).unwrap();
        sim.graph_add_batch_mem_op(
            exec,
            &[BatchMemOp::Write {
                id: a,
                offset: 0,
                value: 1,
                bits32: false,
            }],
        )
        .unwrap();
        let _ = sim.instantiate_graph(exec).unwrap();
        let src = sim.create_graph(d, s).unwrap();
        sim.graph_add_batch_mem_op(
            src,
            &[
                BatchMemOp::Write {
                    id: b,
                    offset: 0,
                    value: 2,
                    bits32: false,
                },
                BatchMemOp::Wait {
                    id: b,
                    offset: 0,
                    value: 2,
                    bits32: false,
                    cmp: WaitValueCmp::Eq,
                },
            ],
        )
        .unwrap();
        sim.update_graph(exec, src).unwrap();
        assert_eq!(sim.graph_batch_mem_ops(exec, 0).unwrap().len(), 2);
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
    }

    #[test]
    fn capture_batch_mem_op_is_one_node() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 64).unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.batch_mem_op(
            d,
            s,
            &[
                BatchMemOp::Write {
                    id: a,
                    offset: 0,
                    value: 3,
                    bits32: false,
                },
                BatchMemOp::Wait {
                    id: a,
                    offset: 0,
                    value: 3,
                    bits32: false,
                    cmp: WaitValueCmp::Eq,
                },
            ],
        ));
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 1);
        let _ = sim.instantiate_graph(g).unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
    }

    #[test]
    fn capture_write_then_wait_replays() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 64).unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.write_value64(d, a, 0, 9, s));
        enq(sim.wait_value64(d, a, 0, 9, WaitValueCmp::Eq, s));
        let g = sim.end_capture().unwrap();
        assert_eq!(
            sim.graph_node_kind(g, 0).unwrap(),
            GraphNodeKind::BatchMemOp
        );
        let _ = sim.instantiate_graph(g).unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
    }

    #[test]
    fn graph_add_wait_rejects_instantiated_and_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 64).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_wait_value64(g, a, 0, 1, WaitValueCmp::Eq)
            .unwrap();
        let exec = sim.instantiate_graph(g).unwrap();
        let err = sim.graph_add_write_value64(exec, a, 0, 1).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        let err = sim
            .graph_add_wait_value64(g, a, 0, 1, WaitValueCmp::Eq)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _end = sim.end_capture().unwrap();
    }

    #[test]
    fn graph_batch_mem_op_set_params_does_not_retarget_exec() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 64).unwrap();
        let b = sim.malloc(d, 64).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_wait_value64(g, a, 0, 1, WaitValueCmp::Eq)
            .unwrap();
        let _ = sim.instantiate_graph(g).unwrap();
        sim.graph_batch_mem_op_set_params(
            g,
            0,
            BatchMemOp::Wait {
                id: b,
                offset: 0,
                value: 1,
                bits32: false,
                cmp: WaitValueCmp::Eq,
            },
        )
        .unwrap();
        let (_, now) = sim.graph_unique_wait_value(g).unwrap();
        match now {
            BatchMemOp::Wait { id, .. } => assert_eq!(id, a, "unique wait is the exec snapshot"),
            other => panic!("{other:?}"),
        }
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        let err = sim.synchronize().unwrap_err();
        match err {
            SimError::NotResident { alloc, device } => {
                assert_eq!(alloc, a);
                assert_eq!(device, d);
            }
            SimError::Invalid { why } if why.contains("deadlock") => {
                panic!("wait should fail residency on freed A, not hang: {why}");
            }
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn graph_exec_batch_mem_op_set_params_retargets() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let write_s = StreamId(1);
        let a = sim.malloc(d, 64).unwrap();
        let b = sim.malloc(d, 64).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_wait_value64(g, a, 0, 1, WaitValueCmp::Eq)
            .unwrap();
        let _ = sim.instantiate_graph(g).unwrap();
        let (node, mut op) = sim.graph_unique_wait_value(g).unwrap();
        match &mut op {
            BatchMemOp::Wait { id, .. } => *id = b,
            other => panic!("{other:?}"),
        }
        sim.graph_exec_batch_mem_op_set_params(g, node, op).unwrap();
        let (_, now) = sim.graph_unique_wait_value(g).unwrap();
        match now {
            BatchMemOp::Wait { id, .. } => assert_eq!(id, b),
            other => panic!("{other:?}"),
        }
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        enq(sim.write_value64(d, b, 0, 1, write_s));
        sim.synchronize().unwrap();
    }

    #[test]
    fn free_mailbox_alloc_while_waiting_is_not_resident() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let wait_s = StreamId(0);
        let free_s = StreamId(1);
        let a = sim.alloc(d, 64, wait_s).unwrap();
        sim.synchronize().unwrap();
        enq(sim.wait_value64(d, a, 0, 1, WaitValueCmp::Eq, wait_s));
        sim.free(d, a, free_s).unwrap();
        let err = sim.synchronize().unwrap_err();
        match err {
            SimError::NotResident { alloc, device } => {
                assert_eq!(alloc, a);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn device_launch_flag_requires_upload() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_launch_ns = 5_000;
            g.graph_upload_ns = 7_000;
            g.graph_instantiate_ns = 1_000;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim
            .instantiate_graph_with_flags(g, GraphInstantiateFlags::DEVICE_LAUNCH)
            .unwrap();
        assert_eq!(
            sim.graph_exec_get_flags(g).unwrap(),
            GraphInstantiateFlags::DEVICE_LAUNCH
        );
        let err = sim.device_launch_graph(g, s).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not uploaded"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.upload_graph(g).unwrap();
        enq(sim.device_launch_graph(g, s));
        sim.synchronize().unwrap();
        assert!(sim
            .operations()
            .any(|o| matches!(o.kind, GpuOp::DeviceLaunch { .. })));
    }

    #[test]
    fn device_launch_rejects_mem_event_child_and_host() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 64).unwrap();
        let mem = sim.create_graph(d, s).unwrap();
        let _id = sim.graph_add_alloc(mem, 64).unwrap();
        let err = sim
            .instantiate_graph_with_flags(mem, GraphInstantiateFlags::DEVICE_LAUNCH)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("device launch"), "{why}"),
            other => panic!("{other:?}"),
        }
        let ev = sim.create_graph(d, s).unwrap();
        sim.create_event(EventId(1)).unwrap();
        sim.graph_add_event_record(ev, EventId(1), false).unwrap();
        let err = sim
            .instantiate_graph_with_flags(ev, GraphInstantiateFlags::DEVICE_LAUNCH)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("device launch"), "{why}"),
            other => panic!("{other:?}"),
        }
        let host = sim.create_graph(d, s).unwrap();
        sim.graph_add_host_func(host).unwrap();
        let err = sim
            .instantiate_graph_with_flags(host, GraphInstantiateFlags::DEVICE_LAUNCH)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("device launch"), "{why}"),
            other => panic!("{other:?}"),
        }
        let leaf = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(leaf, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim.instantiate_graph(leaf).unwrap();
        let parent = sim.create_graph(d, s).unwrap();
        sim.graph_add_child(parent, leaf).unwrap();
        let err = sim
            .instantiate_graph_with_flags(parent, GraphInstantiateFlags::DEVICE_LAUNCH)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("device launch"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn host_launch_of_device_launch_exec_auto_uploads() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim
            .instantiate_graph_with_flags(g, GraphInstantiateFlags::DEVICE_LAUNCH)
            .unwrap();
        assert!(!sim.graph_uploaded(g).unwrap());
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        assert!(sim.graph_uploaded(g).unwrap());
        sim.synchronize().unwrap();
    }

    #[test]
    fn update_graph_rejects_device_launch_exec() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        let exec = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(exec, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim
            .instantiate_graph_with_flags(exec, GraphInstantiateFlags::DEVICE_LAUNCH)
            .unwrap();
        let src = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(src, KernelKind::other(8, 8), &[b], &[b])
            .unwrap();
        let err = sim.update_graph(exec, src).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("device launch"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn device_launch_in_flight_and_capture_refused() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(1 << 40, 4096), &[a], &[a])
            .unwrap();
        let _ = sim
            .instantiate_graph_with_flags(
                g,
                GraphInstantiateFlags::DEVICE_LAUNCH | GraphInstantiateFlags::UPLOAD,
            )
            .unwrap();
        enq(sim.device_launch_graph(g, s));
        let err = sim.device_launch_graph(g, s).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("in flight"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.synchronize().unwrap();
        enq(sim.device_launch_graph(g, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        let err = sim.device_launch_graph(g, s).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _end = sim.end_capture().unwrap();
    }

    #[test]
    fn device_launch_occupies_compute() {
        let mut p = h100().with_compute_slots(1);
        for g in &mut p.gpus {
            g.graph_launch_ns = 50_000;
            g.launch_overhead_ns = 1;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let gemm_s = StreamId(0);
        let launch_s = StreamId(1);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 40, 4096), &[a], &[a], gemm_s));
        let g = sim.create_graph(d, launch_s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[b], &[b])
            .unwrap();
        let _ = sim
            .instantiate_graph_with_flags(
                g,
                GraphInstantiateFlags::DEVICE_LAUNCH | GraphInstantiateFlags::UPLOAD,
            )
            .unwrap();
        enq(sim.device_launch_graph(g, launch_s));
        sim.synchronize().unwrap();
        let leftover = sim
            .operations()
            .find(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.stream == gemm_s)
            .expect("leftover");
        let launcher = sim
            .operations()
            .find(|o| matches!(o.kind, GpuOp::DeviceLaunch { .. }))
            .expect("launcher");
        assert!(launcher.start_ns.unwrap() >= leftover.done_ns.unwrap());
    }

    #[test]
    fn device_launch_set_params_requires_reupload() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim
            .instantiate_graph_with_flags(
                g,
                GraphInstantiateFlags::DEVICE_LAUNCH | GraphInstantiateFlags::UPLOAD,
            )
            .unwrap();
        let (node, mut params) = sim.graph_unique_kernel(g).unwrap();
        params.reads = vec![KernelBuf::whole(b)];
        params.writes = vec![KernelBuf::whole(b)];
        sim.graph_exec_kernel_set_params(g, node, &params).unwrap();
        let err = sim.device_launch_graph(g, s).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not uploaded"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.upload_graph(g).unwrap();
        sim.free_sync(a).unwrap();
        enq(sim.device_launch_graph(g, s));
        sim.synchronize().unwrap();
    }

    #[test]
    fn device_launch_device_updatable_set_params_skips_reupload() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_kernel_node_set_device_updatable(g, 0, true)
            .unwrap();
        let _ = sim
            .instantiate_graph_with_flags(
                g,
                GraphInstantiateFlags::DEVICE_LAUNCH | GraphInstantiateFlags::UPLOAD,
            )
            .unwrap();
        assert!(sim.graph_uploaded(g).unwrap());
        let (node, mut params) = sim.graph_unique_kernel(g).unwrap();
        params.reads = vec![KernelBuf::whole(b)];
        params.writes = vec![KernelBuf::whole(b)];
        let t0 = sim.clock_ns();
        sim.graph_exec_kernel_set_params(g, node, &params).unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(1_000));
        assert!(sim.graph_uploaded(g).unwrap());
        sim.free_sync(a).unwrap();
        enq(sim.device_launch_graph(g, s));
        sim.synchronize().unwrap();
    }

    #[test]
    fn device_updatable_mixed_nodes_set_params_clears_upload() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        sim.graph_kernel_node_set_device_updatable(g, 0, true)
            .unwrap();
        let _ = sim
            .instantiate_graph_with_flags(
                g,
                GraphInstantiateFlags::DEVICE_LAUNCH | GraphInstantiateFlags::UPLOAD,
            )
            .unwrap();
        let mut params = sim.graph_exec_kernel_get_params(g, 0).unwrap();
        params.reads = vec![KernelBuf::whole(b)];
        params.writes = vec![KernelBuf::whole(b)];
        sim.graph_exec_kernel_set_params(g, 0, &params).unwrap();
        assert!(sim.graph_uploaded(g).unwrap());
        let mut params = sim.graph_exec_kernel_get_params(g, 1).unwrap();
        params.reads = vec![KernelBuf::whole(b)];
        params.writes = vec![KernelBuf::whole(b)];
        sim.graph_exec_kernel_set_params(g, 1, &params).unwrap();
        assert!(!sim.graph_uploaded(g).unwrap());
        let err = sim.device_launch_graph(g, s).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not uploaded"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_exec_set_device_updatable_then_set_params_skips_reupload() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim
            .instantiate_graph_with_flags(
                g,
                GraphInstantiateFlags::DEVICE_LAUNCH | GraphInstantiateFlags::UPLOAD,
            )
            .unwrap();
        sim.graph_exec_kernel_node_set_device_updatable(g, 0, true)
            .unwrap();
        let (node, mut params) = sim.graph_unique_kernel(g).unwrap();
        params.reads = vec![KernelBuf::whole(b)];
        params.writes = vec![KernelBuf::whole(b)];
        sim.graph_exec_kernel_set_params(g, node, &params).unwrap();
        assert!(sim.graph_uploaded(g).unwrap());
        sim.free_sync(a).unwrap();
        enq(sim.device_launch_graph(g, s));
        sim.synchronize().unwrap();
    }

    #[test]
    fn device_updatable_after_set_params_does_not_restore_upload() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim
            .instantiate_graph_with_flags(
                g,
                GraphInstantiateFlags::DEVICE_LAUNCH | GraphInstantiateFlags::UPLOAD,
            )
            .unwrap();
        let (node, mut params) = sim.graph_unique_kernel(g).unwrap();
        params.reads = vec![KernelBuf::whole(b)];
        params.writes = vec![KernelBuf::whole(b)];
        sim.graph_exec_kernel_set_params(g, node, &params).unwrap();
        assert!(!sim.graph_uploaded(g).unwrap());
        sim.graph_exec_kernel_node_set_device_updatable(g, 0, true)
            .unwrap();
        let err = sim.device_launch_graph(g, s).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not uploaded"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn kernel_with_capture_records_device_updatable() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel_with(
            d,
            KernelKind::other(8, 8),
            &[a],
            &[a],
            s,
            KernelAttrs {
                device_updatable: true,
                ..KernelAttrs::default()
            },
        ));
        let g = sim.end_capture().unwrap();
        assert!(sim.graph_kernel_node_get_device_updatable(g, 0).unwrap());
        let live = sim.kernel_with(
            d,
            KernelKind::other(8, 8),
            &[a],
            &[a],
            s,
            KernelAttrs {
                device_updatable: true,
                ..KernelAttrs::default()
            },
        );
        match live {
            Err(SimError::Invalid { why }) => {
                assert!(why.contains("graphs-only"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        sim.begin_capture(d, s).unwrap();
        let err = sim
            .graph_kernel_node_set_device_updatable(g, 0, false)
            .unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("cannot capture"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn device_launch_without_flag_is_invalid() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let g = sim.create_graph(d, s).unwrap();
        sim.graph_add_kernel(g, KernelKind::other(8, 8), &[a], &[a])
            .unwrap();
        let _ = sim.instantiate_graph(g).unwrap();
        sim.upload_graph(g).unwrap();
        let err = sim.device_launch_graph(g, s).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not device launch"), "{why}"),
            other => panic!("{other:?}"),
        }
    }
}
