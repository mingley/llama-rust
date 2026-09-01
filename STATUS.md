# STATUS

Live research plan: [PLAN.md](PLAN.md).
Visible five-turn extract: [docs/chatgpt-share-6a920fe1.md](docs/chatgpt-share-6a920fe1.md).
Complete share-API extract: [docs/chatgpt-share-6a920fe1/](docs/chatgpt-share-6a920fe1/).
Work lands on `main`. No PRs.

## Shipped 2026-09-01 — CUDA `cuCtxGetStreamPriorityRange`

`gpu-sim` `Sim::ctx_get_stream_priority_range` is
`cuCtxGetStreamPriorityRange` for the seeded primary context of an
explicit device. Same as `device_get_stream_priority_range`. Example
H100 is `(0, -5)`. Query; legal during capture. Distinct from
`device_get_stream_priority_range` (runtime) and from `stream_get_priority`.
This VM does not invent Engine `--ctx-priority-range`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cuCtxGetCacheConfig`

`gpu-sim` `Sim::ctx_get_cache_config` is `cuCtxGetCacheConfig` for the
seeded primary context of an explicit device. Same as `get_cache_config`.
Query; legal during capture. Distinct from `get_cache_config` (runtime)
and from `get_func_cache_config`. This VM does not invent
`cuCtxSetCacheConfig` or Engine `--ctx-cache-config`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cuCtxGetFlags`

`gpu-sim` `Sim::ctx_get_flags` is `cuCtxGetFlags` for the seeded
primary context of an explicit device. Same flags as `get_device_flags`.
Query; legal during capture. Distinct from `get_device_flags` (runtime)
and from `device_primary_ctx_get_state` (also reports active). This VM
does not invent `cuCtxSetFlags` or Engine `--ctx-flags`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cuCtxGetApiVersion`

`gpu-sim` `Sim::ctx_get_api_version` is `cuCtxGetApiVersion` for the
seeded primary context of an explicit device. CUDA 13.0. Query; legal
during capture. Distinct from `driver_get_version` (same encoding, no
device) and from `device_compute_capability` (Hopper SM version). This
VM does not invent occupancy SM counts or Engine `--ctx-api-version`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cuDeviceComputeCapability`

`gpu-sim` `Sim::device_compute_capability` is `cuDeviceComputeCapability`.
Example H100 is Hopper 9.0. Query; legal during capture. Distinct from
`device_get_attribute` of `ComputeCapabilityMajor` / `Minor` (same
values). This VM does not invent occupancy SM counts or Engine `--cu-compute-capability`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaDevAttrGpuPciSubsystemId`

`gpu-sim` `DeviceAttr::GpuPciSubsystemId` is always 0. This VM has no
NVIDIA PCI subsystem id. Query; legal during capture. Distinct from
`GpuPciDeviceId`. Also `DeviceProperties::pci_subsystem_id`. This VM
does not invent `DeviceAttr::PciSubSystemId` or Engine `--gpu-pci-subsystem`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaDevAttrGpuPciDeviceId`

`gpu-sim` `DeviceAttr::GpuPciDeviceId` is always 0. This VM has no
NVIDIA PCI vendor/device id. Query; legal during capture. Distinct from
`PciDeviceId`. This VM does not invent `DeviceAttr::PciSubSystemId`,
`cudaDevAttrGpuPciSubsystemId`, or Engine `--gpu-pci-device`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaDevAttrMaxSharedMemoryPerMultiprocessor`

`gpu-sim` `DeviceAttr::MaxSharedMemoryPerMultiprocessor` is
`cudaDevAttrMaxSharedMemoryPerMultiprocessor`. Same bytes as
`MaxSharedMemoryPerBlockOptin` (`ReservedSharedMemoryPerBlock` is 0).
Query; legal during capture. Distinct from `MaxSharedMemoryPerBlock`.
This VM does not invent `MaxRegistersPerMultiprocessor` or Engine `--shared-per-mp`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cuDeviceGetTexture1DLinearMaxWidth`

`gpu-sim` `Sim::device_get_texture_1d_linear_max_width` is
`cuDeviceGetTexture1DLinearMaxWidth`. Always 0 (CUDA linear textures
are not modeled). Query; legal during capture. Distinct from
`MaxTexture1DLinearWidth`. This VM does not invent Engine `--texture-1d-linear-max`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaDevAttrVulkanCigSupported`

`gpu-sim` `DeviceAttr::VulkanCigSupported` is always 0. Vulkan
CUDA-in-graphics is not modeled. Query; legal during capture. Distinct
from `D3D12CigSupported`. This VM does not invent Engine `--vulkan-cig`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaDevAttrD3D12CigSupported`

`gpu-sim` `DeviceAttr::D3D12CigSupported` is always 0. D3D12
CUDA-in-graphics is not modeled. Query; legal during capture. Distinct
from `HandleTypeWin32HandleSupported`. This VM does not invent Engine `--d3d12-cig`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaDevAttrMpsEnabled`

`gpu-sim` `DeviceAttr::MpsEnabled` is always 0. CUDA Multi-Process
Service is not modeled. Query; legal during capture. Distinct from
`ComputeMode`. This VM does not invent Engine `--mps-enabled`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA alternate texture 3D dimension DeviceAttr caps

`gpu-sim` `DeviceAttr::MaxTexture3DWidthAlt` /
`MaxTexture3DHeightAlt` / `MaxTexture3DDepthAlt` are always 0. CUDA
alternate 3D texture dims are not modeled. Query; legal during capture.
Distinct from `MaxTexture3DWidth`. This VM does not invent Engine `--texture-3d-alt`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cuDeviceGetLuid`

`gpu-sim` `Sim::device_get_luid` is `cuDeviceGetLuid`. Always zeros
(Windows LUID is not modeled). Query; legal during capture. Distinct
from `device_get_uuid`. Also `DeviceProperties::luid` /
`luid_device_node_mask`. This VM does not invent Engine `--get-luid`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaDeviceProp::luid`

`gpu-sim` `DeviceProperties::luid` and `luid_device_node_mask` are
`cudaDeviceProp::luid` and `luidDeviceNodeMask`. Always 0 (Windows LUID
is not modeled). Query via `device_get_properties`; legal during capture.
Distinct from `uuid`. This VM does not invent Engine `--luid`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaDeviceProp::pciSubSystemID`

`gpu-sim` `DeviceProperties::pci_subsystem_id` is
`cudaDeviceProp::pciSubSystemID`. Always 0 (synthetic PCI has no
subsystem id). Query via `device_get_properties`; legal during capture.
Distinct from `PciDeviceId`. This VM does not invent
`DeviceAttr::PciSubSystemId` or Engine `--pci-subsystem`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA cubemap surface dimension DeviceAttr caps

`gpu-sim` `DeviceAttr::MaxSurfaceCubemapWidth` /
`MaxSurfaceCubemapLayeredWidth` / `MaxSurfaceCubemapLayeredLayers` are
always 0. CUDA cubemap surfaces are not modeled. Query; legal during
capture. Distinct from `MaxSurface2DWidth`. This VM does not invent
Engine `--surface-cubemap`. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA layered surface 1D/2D dimension DeviceAttr caps

`gpu-sim` `DeviceAttr::MaxSurface1DLayeredWidth` /
`MaxSurface1DLayeredLayers` / `MaxSurface2DLayeredWidth` /
`MaxSurface2DLayeredHeight` / `MaxSurface2DLayeredLayers` are always 0.
CUDA layered surfaces are not modeled. Query; legal during capture.
Distinct from `MaxSurface1DWidth`. This VM does not invent Engine
`--surface-layered` or cubemap surface attrs. `gpu-profile capture` is
still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA cubemap layered texture dimension DeviceAttr caps

`gpu-sim` `DeviceAttr::MaxTextureCubemapLayeredWidth` /
`MaxTextureCubemapLayeredLayers` are always 0. CUDA cubemap layered
textures are not modeled. Query; legal during capture. Distinct from
`MaxTextureCubemapWidth`. This VM does not invent Engine
`--texture-cubemap-layered` or layered surface attrs.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA layered texture 1D/2D dimension DeviceAttr caps

`gpu-sim` `DeviceAttr::MaxTexture1DLayeredWidth` /
`MaxTexture1DLayeredLayers` / `MaxTexture2DLayeredWidth` /
`MaxTexture2DLayeredHeight` / `MaxTexture2DLayeredLayers` are always 0.
CUDA layered textures are not modeled. Query; legal during capture.
Distinct from `MaxTexture1DWidth` and from `MaxTextureCubemapWidth`.
This VM does not invent Engine `--texture-layered` or cubemap-layered
texture attrs. `gpu-profile capture` is still refused. Dual score still
has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA cubemap texture dimension DeviceAttr cap

`gpu-sim` `DeviceAttr::MaxTextureCubemapWidth` is always 0. CUDA cubemap
textures are not modeled. Query; legal during capture. Distinct from
`MaxTexture2DWidth`. This VM does not invent Engine `--texture-cubemap`
or layered texture attrs. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA mipmapped texture 1D/2D dimension DeviceAttr caps

`gpu-sim` `DeviceAttr::MaxTexture1DMipmappedWidth` /
`MaxTexture2DMipmappedWidth` / `MaxTexture2DMipmappedHeight` are always
0. CUDA mipmapped textures are not modeled. Query; legal during capture.
Distinct from `MaxTexture1DWidth` and from linear and gather dims. This
VM does not invent Engine `--texture-mipmap` or layered/cubemap texture
attrs. `gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-09-01 — CUDA texture 2D gather dimension DeviceAttr caps

`gpu-sim` `DeviceAttr::MaxTexture2DGatherWidth` / `MaxTexture2DGatherHeight`
are always 0. CUDA texture gather is not modeled. Query; legal during
capture. Distinct from `MaxTexture2DWidth` and from linear texture dims.
This VM does not invent Engine `--texture-gather` or layered/cubemap
texture attrs. `gpu-profile capture` is still refused. Dual score still
has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA linear texture 1D/2D dimension DeviceAttr caps

`gpu-sim` `DeviceAttr::MaxTexture1DLinearWidth` / `MaxTexture2DLinearWidth` /
`MaxTexture2DLinearHeight` / `MaxTexture2DLinearPitch` are always 0. CUDA
linear textures are not modeled. Query; legal during capture. Distinct
from `MaxTexture1DWidth`, `MaxTexture2DWidth`, and `TexturePitchAlignment`.
This VM does not invent Engine `--texture-linear` or layered/cubemap
texture attrs. `gpu-profile capture` is still refused. Dual score still
has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaFuncGetAttributes` compiler fields

`gpu-sim` `FuncAttributes` compiler-emitted `cudaFuncGetAttributes` fields
`sharedSizeBytes`, `constSizeBytes`, `localSizeBytes`, `maxThreadsPerBlock`,
`ptxVersion`, `binaryVersion`, and `cacheModeCA` are always 0 until a
compiled kernel exists. Query; legal during capture. Distinct from
`DeviceAttr::MaxThreadsPerBlock`, `TotalConstantMemory`, and `FuncCache`.
`numRegs` stays unmodeled this slice. This VM does not invent Engine
`--func-attrs` or a PTX compiler. `gpu-profile capture` is still refused.
Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaDevAttrSingleToDoublePrecisionPerfRatio`

`gpu-sim` `DeviceAttr::SingleToDoublePrecisionPerfRatio` is
`cudaDevAttrSingleToDoublePrecisionPerfRatio`
(`cudaDeviceProp` singleToDoublePrecisionPerfRatio). Example H100 is 1
(Hopper FP32 and FP64 peaks match). Query; legal during capture. This
VM does not invent Engine `--fp64-ratio`, clock rates, or kernel
duration from this ratio. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA surface 1D/2D/3D dimension DeviceAttr caps

`gpu-sim` `DeviceAttr::MaxSurface1DWidth` / `MaxSurface2DWidth` /
`MaxSurface2DHeight` / `MaxSurface3DWidth` / `MaxSurface3DHeight` /
`MaxSurface3DDepth` are always 0. CUDA surfaces are not modeled. Query;
legal during capture. Distinct from `SurfaceAlignment` and from texture
dim caps. This VM does not invent Engine `--max-surface` or
layered/cubemap surface attrs.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA texture 2D/3D dimension DeviceAttr caps

`gpu-sim` `DeviceAttr::MaxTexture2DWidth` / `MaxTexture2DHeight` /
`MaxTexture3DWidth` / `MaxTexture3DHeight` / `MaxTexture3DDepth` are
always 0. CUDA arrays / textures are not modeled. Query; legal during
capture. Distinct from `MaxTexture1DWidth`. This VM does not invent
Engine `--max-texture` or layered/cubemap texture attrs.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaDevAttrGlobalMemoryBusWidth`

`gpu-sim` `DeviceAttr::GlobalMemoryBusWidth` is
`cudaDevAttrGlobalMemoryBusWidth` (`cudaDeviceProp` memoryBusWidth).
Example H100 is 5120 bits. Example H200 is 6144 bits. Query; legal
during capture. Profile key `global_memory_bus_width_bits`. This VM
does not invent Engine `--bus-width`, memory clock rates, or HBM
duration from bus width. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaDevAttrMaxRegistersPerBlock`

`gpu-sim` `DeviceAttr::MaxRegistersPerBlock` is
`cudaDevAttrMaxRegistersPerBlock` (`cudaDeviceProp` regsPerBlock).
Example H100 is 65536. Query; legal during capture. This VM does not
model a register file, invent occupancy SM counts, or invent Engine
`--max-registers`. `gpu-profile capture` is still refused. Dual score
still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA launch-geometry DeviceAttr caps

`gpu-sim` `DeviceAttr::MaxThreadsPerBlock` / `MaxBlockDimX` / `Y` / `Z` /
`MaxGridDimX` / `Y` / `Z` are CUDA launch-geometry caps. Example H100 is
1024 threads per block; block 1024, 1024, 64; grid `i32::MAX`, 65535,
65535. Query; legal during capture. This VM does not model a thread-block
launch, invent occupancy SM counts, or invent Engine `--max-threads`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaDevAttrComputeCapabilityMajor` / `Minor`

`gpu-sim` `DeviceAttr::ComputeCapabilityMajor` / `ComputeCapabilityMinor`
are `cudaDevAttrComputeCapabilityMajor` and `cudaDevAttrComputeCapabilityMinor`
(`cudaDeviceProp` major and minor). Example H100 is Hopper 9.0. Query;
legal during capture. Profile keys `compute_capability_major` /
`compute_capability_minor`. This VM does not invent Engine
`--compute-capability`, occupancy SM counts, clock rates, or
`cudaChooseDevice`. `gpu-profile capture` is still refused. Dual score
still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaGetErrorName` / `cudaGetErrorString`

`gpu-sim` `SimError::error_name` / `error_string` are `cudaGetErrorName` /
`cudaGetErrorString`. Query on the error already returned (no thread-local
last error). `Invalid` names `cudaErrorInvalidValue`; the modeled `why`
stays on Display. This VM does not invent Engine `--error-name` or
`cudaGetLastError` / `cudaPeekAtLastError`. `gpu-profile capture` is still
refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaDeviceReset`

`gpu-sim` `reset_device` is `cudaDeviceReset`. Waits outstanding work on that
GPU (unlike `destroy_stream`), then frees `cudaMalloc` / `cudaMallocPitch` /
`cudaMalloc3D`. `cudaMallocAsync` stays. User streams except NULL become
unknown until create. Device flags and limits return to CUDA defaults. Peer
pairs involving that GPU return to the profile seed. Host / managed allocs,
events, and graphs stay. `ctx_get_id` stays. Capture cannot include it. This
VM does not invent Engine `--device-reset` or a public `cuDevicePrimaryCtxReset`
(no `CUcontext` object). `gpu-profile capture` is still refused. Dual score
still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaLimitDevRuntimePendingLaunchCount`

`gpu-sim` `DeviceLimit::DevRuntimePendingLaunchCount` caps in-flight
`device_launch_graph` (host, fire-and-forget, sibling, and flushed tail).
A queued tail does not occupy a slot. Default 2048. Exceeding is Invalid
`"pending launch count"`. Host `launch_graph` does not count. This VM does
not invent Engine `--pending-launch` or make `DevRuntimeSyncDepth`
mechanical (no device-side `cudaDeviceSynchronize`). Heap still does not
charge HBM. `gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaStreamGraphFireAndForgetAsSibling`

`gpu-sim` `GRAPH_FIRE_AND_FORGET_AS_SIBLING` is `cudaStreamGraphFireAndForgetAsSibling`
on `device_launch_graph`. Same nested DeviceLaunch as fire-and-forget, but the
parent instance does not wait: no join on the parent host stream, and tail
launch does not wait for the sibling. Requires a host-issued DeviceLaunch in
flight. Host `launch_graph` of that id is Invalid `"device launch stream"`.
This VM does not invent Engine `--graph-faf-sibling`. `current_graph_exec`
still returns the lowest in-flight DeviceLaunch id. `gpu-profile capture` is
still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaStreamGraphFireAndForget` / `cudaStreamGraphTailLaunch`

`gpu-sim` named device-graph streams are `cudaStreamGraphFireAndForget` /
`cudaStreamGraphTailLaunch` (`StreamId::GRAPH_FIRE_AND_FORGET` /
`GRAPH_TAIL_LAUNCH`) on `device_launch_graph`. They require a host-issued
DeviceLaunch in flight (`"no current graph exec"`). Fire-and-forget runs on
an internal stream; the parent host stream waits for the FAF body. Tail
launch waits for the parent instance plus those FAF children (one tail per
instance; `"device launch tail"`). Self tail-relaunch of the in-flight exec
is legal. Host `launch_graph` / `kernel` of those ids is Invalid
`"device launch stream"`. This VM does not invent
`cudaStreamGraphFireAndForgetAsSibling` or Engine `--graph-tail-launch`.
`current_graph_exec` still returns the lowest in-flight DeviceLaunch id.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaStreamDestroy`

`gpu-sim` `destroy_stream` is `cudaStreamDestroy`. Returns immediately;
in-flight work still completes. Capture cannot include it. NULL is Invalid
`"null stream"`. A destroyed handle is Invalid `"unknown stream"` for new
work and queries until `stream_create_with_flags` / `set_stream_blocking`.
Recreate while that stream still has unfinished ops is Invalid
`"stream in flight"` (this VM reuses caller-chosen ids). Distinct from
`destroy_event` (which waits). Device `synchronize` still drains parked
work. This VM does not invent Engine `--stream-destroy`. `gpu-profile
capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaGetCurrentGraphExec`

`gpu-sim` `current_graph_exec` is `cudaGetCurrentGraphExec`. Query; legal
during capture. Returns the DeviceLaunch executable in flight on that
device, if any. Host `launch_graph` does not count. Concurrent in-flight
DeviceLaunch execs return the lowest `GraphId`. A parked in-flight destroy
still reports that exec until the launch tail completes. This VM does not
invent Engine `--current-graph-exec`, `CUgraphDeviceNode`, or public
`Sim::advance`. `gpu-profile capture` is still refused. Dual score still
has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaDeviceGetStreamPriorityRange`

`gpu-sim` `device_get_stream_priority_range` is `cudaDeviceGetStreamPriorityRange`.
Query; legal during capture. Example H100 is least `0`, greatest `-5`. Stream
create / SetPriority / SetAttribute clamp out of range. GetPriority reports the
clamped value. Numerically lower priorities start first under contention.
Graph kernel-node SetPriority stays unclamped for ExecUpdate (PLAN 506).
`set_created_streams_priority` assigns `-stream_id` then clamps. This VM does
not invent Engine `--stream-priority-range` or `cudaStreamDestroy`. `gpu-profile
capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaStreamBeginRecaptureToGraph`

`gpu-sim` `begin_recapture_to_graph` is `cudaStreamBeginRecaptureToGraph` /
`cuStreamBeginRecaptureToGraph`. Recapture updates an existing graph in place
(not append). Topology and alloc/free mismatches fail immediately. Other node
parameter mismatches update the original node unless a callback returns
failure. `None` callback still applies those updates. Failure leaves the graph
undefined (`"undefined graph"`); `destroy_graph` stays legal. User objects are
released first. Matching recapture malloc returns the existing graph-mem
pointer. This VM does not invent Engine `--graph-recapture`. `gpu-profile
capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaDriverGetVersion` / `cudaRuntimeGetVersion`

`gpu-sim` `driver_get_version` / `runtime_get_version` are `cudaDriverGetVersion`
/ `cuDriverGetVersion` / `cudaRuntimeGetVersion` (CUDA 13.0). Query; legal
during capture. Same toolkit value. Distinct from `device_count`. This VM does
not invent Engine `--driver-version` or `cudaGetLastError`. `gpu-profile
capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaGraphRemoveDependencies` v2

`gpu-sim` `graph_remove_dependencies_with_data` /
`graph_remove_dependencies_n_with_data` is `cudaGraphRemoveDependencies` v2
(`GraphEdgeData`). A matching `(from, to, data)` is removed; a missing
matching edge is Invalid `"graph dependency"`. Distinct from v1 missing-remove
no-op (PLAN 182). v1 still removes a launch-completion edge (it ignores stored
data). Capture cannot include it. Illegal on an instantiated exec. This VM
does not invent Engine `--graph-edge-data`. `gpu-profile capture` is still
refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaStreamGetDevice`

`gpu-sim` `stream_get_device` is `cudaStreamGetDevice` / `cuStreamGetDevice`
(the device of the stream; green-ctx streams return the ctx create device).
Query; legal during capture. Distinct from `stream_get_id` and
`green_ctx_get_device`. This VM does not invent Engine `--stream-device`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cuGraphNodeGetContainingGraph`

`gpu-sim` `graph_node_get_containing_graph` is `cuGraphNodeGetContainingGraph`
(the graph that owns the node). A child-graph node still lives in the parent;
the nested graph is `graph_child_get_graph`. Query; capture is legal. A parked
in-flight-destroyed exec is Invalid `"unknown graph"`. Live exec
GetContainingGraph stays. Definition GetContainingGraph stays. This VM does
not invent Engine `--graph-containing`. `gpu-profile capture` is still
refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — Parked exec GetAttribute is unknown

`gpu-sim` CUDA `cudaGraphKernelNodeGetAttribute` of an exec whose handle
was destroyed while a launch was in flight is Invalid `"unknown graph"`
(typed getters plus generic GetAttribute). Query; capture is legal. Live
exec GetAttribute stays. Definition GetAttribute stays. Exec GetAttribute
of a parked dest already unknown via `as_exec`. This VM does not invent
Engine `--graph-get-attr-gone`. `gpu-profile capture` is still refused.
Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — Parked exec GetLocalId is unknown

`gpu-sim` CUDA `cuGraphNodeGetLocalId` / `cuGraphNodeGetToolsId` of an
exec whose handle was destroyed while a launch was in flight is Invalid
`"unknown graph"`. Query; capture is legal. Live exec GetLocalId stays.
Definition GetLocalId stays. GetToolsId of a parked dest already unknown
via `graph_get_id`. This VM does not invent Engine `--graph-local-id-gone`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — Parked exec GetDependencies is unknown

`gpu-sim` CUDA `cudaGraphNodeGetDependencies` /
`cudaGraphNodeGetDependentNodes` of an exec whose handle was destroyed
while a launch was in flight is Invalid `"unknown graph"` (v1 and v2).
Query; capture is legal. Live exec GetDependencies stays. Definition
GetDependencies stays. This VM does not invent Engine `--graph-deps-gone`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — Parked exec GetParams is unknown

`gpu-sim` CUDA definition `cudaGraph*NodeGetParams` / `cudaGraphNodeGetParams`
of an exec whose handle was destroyed while a launch was in flight is Invalid
`"unknown graph"`. Query; capture is legal. Live exec GetParams stays.
Definition GetParams stays. Exec GetParams of a parked dest stays unknown
via `as_exec`. This VM does not invent Engine `--graph-get-params-gone`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — Parked exec as child graph is unknown

`gpu-sim` CUDA `cudaGraphAddChildGraphNode` /
`cudaGraphChildGraphNodeSetParams` / `cudaGraphExecChildGraphNodeSetParams`
whose child handle was destroyed while a launch was in flight is Invalid
`"unknown graph"`. Capture is still reported first. Live exec as child
stays. Add-child of the definition after that exec is parked stays. This
VM does not invent Engine `--graph-child-gone`. `gpu-profile capture` is
still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — Parked exec SetParams is unknown

`gpu-sim` CUDA definition `cudaGraph*NodeSetParams` / `cudaGraphNodeSetParams`
of an exec whose handle was destroyed while a launch was in flight is Invalid
`"unknown graph"` (kernel, memcpy, memset, host, child, event SetEvent, free,
batch-mem, conditional). Capture is still reported first. Live exec SetParams
stays. Definition SetParams stays. Exec SetParams of a parked dest stays
unknown via `as_exec_for_update`. This VM does not invent Engine
`--graph-set-params-gone`. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — Parked exec SetAttribute is unknown

`gpu-sim` CUDA `cudaGraphKernelNodeSetAttribute` of an exec whose handle
was destroyed while a launch was in flight is Invalid `"unknown graph"`.
Capture is still reported first. Live exec SetAttribute stays. Definition
SetAttribute stays. This VM does not invent Engine
`--graph-set-attr-gone`. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — Parked exec CopyAttributes is unknown

`gpu-sim` CUDA `cudaGraphKernelNodeCopyAttributes` of an exec whose
handle was destroyed while a launch was in flight is Invalid
`"unknown graph"` (source or dest). Capture is still reported first.
Live exec as either end stays. Definition CopyAttributes stays. This VM
does not invent Engine `--graph-copy-attr-gone`. `gpu-profile capture`
is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — Parked ExecUpdate source is unknown

`gpu-sim` CUDA `cudaGraphExecUpdate` whose source graph handle was
destroyed while a launch was in flight is Invalid `"unknown graph"`.
Capture is still reported first. Live exec as source stays. Definition
as source stays. Dest in-flight update stays `"exec in flight"`. This
VM does not invent Engine `--graph-update-src-gone`. `gpu-profile
capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — Parked exec capture-to-graph is unknown first

`gpu-sim` CUDA `begin_capture_to_graph` of an exec whose handle was
destroyed while a launch was in flight is Invalid `"unknown graph"`
before `"graph instantiated"`. Capture is still reported first. Live
exec still `"graph instantiated"`. Capture-to-graph of the definition
stays. The parked launch still finishes. This VM does not invent Engine
`--graph-capture-gone`. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — Parked exec definition mutators are unknown first

`gpu-sim` CUDA definition mutators of an exec whose handle was destroyed
while a launch was in flight are Invalid `"unknown graph"` before
`"graph instantiated"` (add node, user-object retain/release, destroy
node, remove deps, conditional create). Capture is still reported first.
Live exec still `"graph instantiated"`. Definition mutators stay. This
VM does not invent Engine `--graph-def-gone`. `gpu-profile capture` is
still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — Parked exec clone and instantiate are unknown

`gpu-sim` CUDA `clone_graph` / `instantiate_graph` of an exec whose
handle was destroyed while a launch was in flight are Invalid
`"unknown graph"`. Capture is still reported first. Clone of a live exec
stays. Re-instantiate of a live exec stays a no-op. Clone and instantiate
of the definition stay (instantiate creates a new exec). The parked
launch still finishes. This VM does not invent Engine
`--graph-clone-gone`. `gpu-profile capture` is still refused. Dual score
still has no `$/M tokens`.

## Shipped 2026-09-01 — Parked exec queries are unknown

`gpu-sim` CUDA queries of an exec whose handle was destroyed while a
launch was in flight are Invalid `"unknown graph"` (`graph_len` /
`graph_instantiated` / `graph_get_id` / `graph_nodes` /
`graph_debug_dot`). The launch still finishes. Capture is still
reported first. Getters before destroy stay. Definition queries stay.
This VM does not invent Engine `--graph-query-gone`. `gpu-profile
capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — Concurrent host launches pin every destroy tail

`gpu-sim` CUDA `cudaGraphExecDestroy` of an exec with concurrent in-flight
host `launch_graph`s waits for every launch tail, not only the last.
The handle is unknown immediately; user-object exec refs are held until
all those tails complete. Capture is still reported first. Empty host
launches do not pin destroy. Host SetParams stay. This VM does not invent
Engine `--graph-exec-multi-launch`. `gpu-profile capture` is still
refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — ExecUpdate of an in-flight exec is Invalid

`gpu-sim` CUDA `cudaGraphExecUpdate` of an exec with an in-flight host
`launch_graph` or `upload_graph_async` is Invalid `"exec in flight"`.
Capture is still reported first. After synchronize, update stays. Host
SetParams of that exec stay. DeviceLaunch update stays NotSupported.
This VM does not invent Engine `--graph-exec-update-inflight`.
`gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-09-01 — DeviceLaunch in-flight upload is Invalid

`gpu-sim` CUDA host `upload_graph` / `upload_graph_async` of a
DeviceLaunch exec while `device_launch_graph` is in flight is Invalid
`"device launch in flight"`. Capture is still reported first. After
synchronize, upload stays. Getters stay. Destroy stays. Host
`launch_graph` in-flight upload stays. This VM does not invent Engine
`--device-launch-upload`. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — In-flight GraphUpload exec destroy parks

`gpu-sim` CUDA `cudaGraphExecDestroy` of an exec with an in-flight
`upload_graph_async` does not abort the upload (the handle is unknown
immediately; the upload still completes and user-object exec refs are
held until that op finishes). Capture is still reported first. Idle
exec destroy still drops immediately. Host-sync `upload_graph` then
destroy stays immediate. Host `launch_graph` and DeviceLaunch
in-flight destroy still park. This VM does not invent Engine
`--graph-upload-destroy`. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — Host launch_graph in-flight exec destroy parks

`gpu-sim` CUDA `cudaGraphExecDestroy` of an in-flight host
`launch_graph` does not abort the launch (any instantiated exec, not
only DeviceLaunch). The handle is unknown immediately; user-object exec
refs are held until that launch's tail completes. Capture is still
reported first. Idle exec destroy still drops immediately. An empty host
launch does not pin destroy to unrelated prior stream work. Host
`launch_graph` in-flight SetParams stay. DeviceLaunch
`device_launch_graph` in-flight destroy still parks. This VM does not
invent Engine `--graph-exec-destroy-inflight`. `gpu-profile capture` is
still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — Instantiate copies user-object retains onto the exec

`gpu-sim` CUDA instantiate copies user-object retains from the
definition onto the exec. Retain on an exec stays illegal. Clone still
does not copy retains. Retains added after instantiate stay on the
definition. In-flight DeviceLaunch exec destroy still holds those exec
refs until the launch completes. Capture is still reported first. This
VM does not invent Engine `--user-object-instantiate`. `gpu-profile
capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — DeviceLaunch destroy does not abort in-flight launch

`gpu-sim` CUDA `cudaGraphExecDestroy` of an in-flight DeviceLaunch exec
does not abort the launch (the handle is unknown immediately; the body
still runs). Capture is still reported first. Idle exec destroy still
drops immediately. Host `launch_graph` in-flight destroy of a
DeviceLaunch exec still drops immediately this slice. This VM does not
invent Engine `--device-launch-destroy`. `gpu-profile capture` is still
refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — DeviceLaunch in-flight host updates are Invalid

`gpu-sim` CUDA host updates of a DeviceLaunch exec while
`device_launch_graph` is in flight are Invalid `"device launch in
flight"` (SetParams / SetAttribute / SetEnabled / CopyAttributes).
Capture is still reported first. After synchronize, SetParams stay.
Getters stay. Destroy stays. Host-instantiated exec SetParams during a
regular `launch_graph` stay. This VM does not invent Engine
`--device-launch-in-flight`. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — DeviceLaunch kernel buffers stay on-device

`gpu-sim` CUDA DeviceLaunch kernel buffers must be origin-device memory
or pinned/mapped/managed host (off-device malloc is `"device launch
instantiate flag"`). Exec kernel SetParams re-apply those dest rules.
Definition SetParams still defer to instantiate. Host instantiate stays.
Capture is still reported first. This VM does not invent Engine
`--device-launch-kernel`. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — DeviceLaunch exec cannot attach event attrs

`gpu-sim` CUDA DeviceLaunch exec SetAttribute cannot attach
programmatic or launch-completion events (`"device launch instantiate
flag"`). Clearing stays. External/interprocess Invalid stays first.
Definition SetAttribute still defers to instantiate. Host-instantiated
exec attach stays. Capture is still reported first. This VM does not
invent Engine `--device-launch-event-attr`. `gpu-profile capture` is
still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — DeviceLaunch exec SetParams mixed ctx

`gpu-sim` CUDA DeviceLaunch exec SetParams re-apply mixed green-ctx
rules (`"graph multiple ctx"`). Same-ctx DeviceLaunch SetParams stays.
Single-node ctx SetParams stays. Definition SetParams still defer to
instantiate. Host-instantiated exec mixed-ctx SetParams stays. Capture
is still reported first. This VM does not invent Engine
`--device-launch-set-ctx` or a second MultipleCtxs result for SetParams.
`gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-09-01 — DeviceLaunch exec SetParams re-check dest

`gpu-sim` CUDA DeviceLaunch exec memcpy/memset SetParams re-apply
instantiate dest rules (`"device launch instantiate flag"`). Definition
SetParams still defer to instantiate. Host-instantiated exec SetParams
of off-device dest stays. Capture is still reported first. Legal
DeviceLaunch SetParams to HostPinned / origin Device / managed memset
dest stay. This VM does not invent Engine `--device-launch-set-params`.
`gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-09-01 — DeviceLaunch memset allows managed dest

`gpu-sim` CUDA DeviceLaunch memset dest may be managed
(`cudaMallocManaged`). Origin-device malloc, pinned, and mapped stay.
Pageable and off-device malloc stay `"device launch instantiate flag"`.
Host instantiate stays. Capture is still reported first. This VM does
not invent Engine `--device-launch-managed`. `gpu-profile capture` is
still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — Unused conditional handles fail instantiate

`gpu-sim` CUDA unused conditional handles fail instantiate
(`ConditionalHandleUnused`; `"conditional handle unused"`).
Set-conditional does not associate. Live IF / WHILE / SWITCH does.
DeviceLaunch type refuse still beats unused. Capture is still reported
first. This VM does not invent Engine `--graph-cond-unused` or a
double-association error. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — DeviceLaunch mixed green ctx is MultipleDevices

`gpu-sim` CUDA DeviceLaunch mixed node green ctx is
`MultipleDevicesNotSupported` (driver MultipleCtxs; `"graph multiple
ctx"`). Same-ctx DeviceLaunch stays. Host instantiate of mixed ctx stays.
Type refuse still beats ctx. Capture is still reported first. This VM
does not invent Engine `--device-launch-ctx` or a second instantiate
result variant. `gpu-profile capture` is still refused. Dual score still
has no `$/M tokens`.

## Shipped 2026-09-01 — DeviceLaunch memset stays on the graph device

`gpu-sim` CUDA DeviceLaunch memset dest must be origin-device memory or
pinned/mapped host (off-device, pageable, or unknown dest is `"device
launch instantiate flag"`). Host instantiate of that memset stays.
Capture is still reported first. Origin-device malloc, pinned, and
mapped DeviceLaunch memset stay legal. This VM does not invent Engine
`--device-launch-memset`. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — DeviceLaunch memcpy stays on the graph device

`gpu-sim` CUDA DeviceLaunch memcpy `Place::Device` must match the graph
origin device (`HostPinned` stays; off-device Device is `"device launch
instantiate flag"`). Host instantiate of that memcpy stays. Capture is
still reported first. Kernel, HostPinned memcpy, and same-device D2D
DeviceLaunch stay legal. This VM does not invent Engine
`--device-launch-memcpy` or CUDA-array memcpy. `gpu-profile capture` is
still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — DeviceLaunch cannot instantiate an empty graph

`gpu-sim` CUDA `cudaGraphInstantiateFlagDeviceLaunch` cannot instantiate
an empty graph and requires at least one kernel, memcpy, or memset node
(`"device launch empty"`; `NodeOperationNotSupported`, `err_node =
None`). Empty, batch-mem, wait-value, and write-value nodes stay
`"device launch instantiate flag"`. Child graphs stay refused. Kernel,
memcpy, and memset-only graphs stay legal. `DEVICE_LAUNCH` plus `UPLOAD`
stays. Capture is still reported first. This VM does not invent Engine
`--device-launch-empty` or reverse DeviceLaunch child-graph refuse.
`gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-09-01 — User-object count is 1 through INT_MAX

`gpu-sim` CUDA user-object `count` / `initialRefcount` is `1..=i32::MAX`
(`INT_MAX`; `user_object_create` / `user_object_retain` /
`user_object_release` / `graph_retain_user_object` /
`graph_release_user_object`). `GraphUserObjectFlags::MOVE` still ignores
`count`. Capture is still reported first. This VM does not invent Engine
`--user-object-int-max`. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — AddNode rejects duplicate dependencies

`gpu-sim` CUDA `cudaGraphAddNode` `pDependencies` may not have duplicate
entries (`graph_add_node` / `graph_add_node_with_data`; `"graph
dependency"`). All-or-nothing: the node is not created. Capture is still
reported first. Distinct from PLAN 182 AddDependencies duplicate-add
no-op. This VM does not invent Engine `--graph-add-node-dup` or reverse
PLAN 182. `gpu-profile capture` is still refused. Dual score still has
no `$/M tokens`.

## Shipped 2026-09-01 — GetFlags omits instantiate Upload

`gpu-sim` CUDA `cudaGraphExecGetFlags` (`graph_exec_get_flags`) omits
`GraphInstantiateFlags::UPLOAD` (it does not affect the resulting
executable graph). Instantiate with UPLOAD still host-sync uploads.
`AUTO_FREE` plus `UPLOAD` and `DEVICE_LAUNCH` plus `UPLOAD` stay legal;
GetFlags returns the other bits. Capture is a query. This VM does not
invent Engine `--graph-exec-flags` or omit other instantiate bits.
`gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-09-01 — AddDependencies cannot change existing edge data

`gpu-sim` CUDA `cudaGraphAddDependencies` cannot change `GraphEdgeData`
of an existing `(from, to)` (`"graph edge data"`). Incoming Default is
the v1 no-op (PLAN 182) and does not overwrite stored launch-completion
data. Same non-default is idempotent. Intra-batch conflicting data is
the same Invalid. Capture is still reported first. New launch-completion
pairs stay. This VM does not invent a second existing-edge data check or
reverse PLAN 182. `gpu-profile capture` is still refused. Dual score
still has no `$/M tokens`.

## Shipped 2026-09-01 — DeviceLaunch forbids AutoFreeOnLaunch

`gpu-sim` CUDA `cudaGraphInstantiateFlagDeviceLaunch` cannot combine
with `cudaGraphInstantiateFlagAutoFreeOnLaunch` (`"device launch auto
free"`). `instantiate_graph_with_params` reports Error with `err_node =
None`. Capture is still reported first. `AUTO_FREE` plus `UPLOAD` and
`DEVICE_LAUNCH` plus `UPLOAD` stay. This VM does not invent Engine
`--device-launch-auto-free`. `gpu-profile capture` is still refused.
Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA GetEdges LossyQuery

`gpu-sim` CUDA v1 `cudaGraphGetEdges` / `cudaGraphNodeGetDependencies` /
`cudaGraphNodeGetDependentNodes` (`edgeData` NULL) are Invalid
`"lossy query"` when any reported edge has non-default stored
`GraphEdgeData` (`cudaErrorLossyQuery`). Default-only edges stay. v2
`graph_edges_with_data` / `graph_node_deps_with_data` /
`graph_node_dependents_with_data` stay lossless. Per-node queries only
inspect that node's edges. Debug-dot ExtraTopoInfo still dumps ports.
This VM does not invent Engine `--graph-lossy` or LossyQuery on
debug-dot. `gpu-profile capture` is still refused. Dual score still has
no `$/M tokens`.

## Shipped 2026-09-01 — ExecUpdate preserves node enable

`gpu-sim` CUDA `cudaGraphExecUpdate` leaves `graph_node_set_enabled` state
unchanged. Disabled nodes stay skipped after a successful update. Typed
ExecSetParams also leave enable unchanged. This VM does not invent Engine
`--graph-enable-update` or copy enable from the source definition.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA ExecMemset SetParams 2D/3D address-only

`gpu-sim` CUDA `cudaGraphExecMemsetNodeSetParams` of a 2D/3D node may
change address only (`"memset dims"` for width, height, pitch, depth,
ysize, plus elementSize). 1D nodes may change dimensions.
`graph_exec_node_set_params` of memset uses the same check. Extra
`graph_exec_memset_set_params_2d` plus `_3d` stay legal. Definition
SetParams still accepts geometry changes. This VM does not invent Engine
`--graph-memset-dims` or 1D memset work-resource mapping failures.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA ExecMemcpy SetParams 1D-only

`gpu-sim` CUDA `cudaGraphExecMemcpyNodeSetParams` is 1-dimensional only:
both the instantiated node and new `MemcpyOp` must be `is_1d` (`"memcpy
1d"`). `graph_exec_node_set_params` of memcpy uses the same check.
`graph_exec_memcpy_set_params_1d` may still convert a 2D/3D node. Extra
`graph_exec_memcpy_set_params_2d` plus `_3d` stay. Definition SetParams still
accepts 2D/3D. This VM does not invent Engine `--graph-memcpy-1d` or
ExecUpdate 2D memcpy geometry as ParametersChanged. `gpu-profile capture`
is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — ExecUpdate kernel function variant

`gpu-sim` CUDA `cudaGraphExecUpdate` of a different kernel-node function
(`KernelKind` variant: Other, Matmul, or GroupedMoeGemm) is
`GraphExecUpdateResult::FunctionChanged`. Matching variants still update
(work sizes may change). `graph_exec_kernel_set_params` stays legal. Child
SetParams nested function stays a parameter. This VM does not invent Engine
`--graph-function-update` or `UnsupportedFunctionChange` (no CUfunction
signature or destroy). `gpu-profile capture` is still refused. Dual score
still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA mempool AllocationType and ExportHandleTypes

`gpu-sim` CUDA `cudaMemPoolAttrAllocationType` /
`cudaMemPoolAttrExportHandleTypes` are Get-only
(`pool_get_attribute`; Set is `"read-only pool attr"`). AllocationType is
always `MemAllocationType::PINNED`. ExportHandleTypes is POSIX-FD on
shareable exporters and `MemHandleType::NONE` on default, `create_pool`,
and imported handles (imported cannot be re-exported). Graph-memory pool
stays Invalid. Capture: Get legal, Set Invalid. This VM does not invent
NT/fabric handle types, Engine `--pool-alloc-type`, or Set of these
attrs. `gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-09-01 — CUDA mempool MaxPoolSize attribute

`gpu-sim` CUDA `cudaMemPoolAttrMaxPoolSize` is `MemPoolAttr::MaxPoolSize`
(`pool_get_attribute` / `pool_set_attribute`). Typed `set_pool_max_size`
stays. Get reports stored `MemPoolProps::max_size` (`0` unlimited). Set
updates the cap for later `alloc_from_pool` (reserved live plus cached).
Does not free live allocs if the new cap is below current reserved.
Graph-memory pool stays Invalid. Capture: Get legal, Set Invalid.
Imported pools Get/Set the exporter. `expertvm sim --mempool-max` stays
`create_pool_with_props` plus `set_device_mempool`. This VM does not
invent a second MaxPoolSize CLI, Engine `--mempool-max` as
SetAttribute-only, alignment rounding, `cudaMemPoolAttrHwDecompressEnabled`,
or pool LocationType / LocationId. `gpu-profile capture` is still refused.
Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — ExecUpdate memcpy memory type

`gpu-sim` CUDA `cudaGraphExecUpdate` memcpy source and destination
memory types (`CU_MEMORYTYPE_*` / `Place`) cannot change
(`GraphExecUpdateResult::ParametersChanged`). Alloc and size may.
`graph_exec_memcpy_set_params` stays legal. CUDA arrays stay uninvented.
This VM does not invent Engine `--memcpy-update`. `gpu-profile capture`
is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — ExecUpdate 2D memset geometry

`gpu-sim` CUDA `cudaGraphExecUpdate` of 2D and 3D memset nodes may
change address only (id / offset; fill value is not modeled). Width,
height, pitch, depth, ysize, and elementSize are
`GraphExecUpdateResult::ParametersChanged`. 1D memset may change
dimensions. `graph_exec_memset_set_params_2d` stays legal. This VM does
not invent Engine `--memset-update` or 1D work-resource mapping
failures. `gpu-profile capture` is still refused. Dual score still has
no `$/M tokens`.

## Shipped 2026-09-01 — ExecUpdate UseNodePriority kernel priority

`gpu-sim` CUDA `cudaGraphExecUpdate` forbids kernel-node priority
changes when the exec was instantiated with
`cudaGraphInstantiateFlagUseNodePriority`
(`GraphExecUpdateResult::AttributesChanged`). Matching priorities still
update. Default instantiate copies priority as a parameter.
`graph_exec_kernel_node_set_priority` stays legal. This VM does not
invent Engine `--graph-node-priority-update`, child SetParams nested
priority, or `cudaDeviceGetStreamPriorityRange` clamping. `gpu-profile
capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — ExecUpdate launch-completion ports are topology

`gpu-sim` CUDA `cudaGraphExecUpdate` treats `GraphEdgeData` ports as
topology: Default completion vs `cudaGraphKernelNodePortLaunchCompletion`
is `GraphExecUpdateResult::DependenciesChanged`. Same `(from, to)` with
matching ports still updates. Child-graph SetParams nested topology
includes ports. Typed `graph_add_dependencies` stays Default ports 0.
This VM does not invent Engine `--graph-edge-port`, programmatic edges,
or `cudaGraphRemoveDependencies` edgeData. `gpu-profile capture` is still
refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA ExtraTopoInfo edge numbering

`gpu-sim` `GraphDebugDotFlags::EXTRA_TOPO_INFO` is
`cudaGraphDebugDotFlagsExtraTopoInfo` (`1 << 14`). Debug-dot numbers
existing edges (`label="0"`). Launch-completion edges also dump
`from_port=2`. `VERBOSE` includes ExtraTopoInfo. Flags `0` stays
unlabeled edges. Query; legal during capture. This VM does not invent
extra conditional edges or Engine `--graph-debug-dot`. `gpu-profile
capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — graph launch-completion edge ports

`gpu-sim` CUDA `cudaGraphKernelNodePortLaunchCompletion` on
`GraphEdgeData::from_port`: a Default-type graph edge with that port waits
for the source kernel to start, not finish. Typed `graph_add_dependencies`
stays Default ports 0. Programmatic type and port 1 stay Invalid. Source
must be a kernel. GetEdges / GetDependencies v2 report stored ports. This
VM does not invent an Engine flag for edge ports or nonzero `to_port`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — launch-attribute event flags and interprocess

`gpu-sim` CUDA `cudaLaunchAttributeProgrammaticEvent` /
`cudaLaunchAttributeLaunchCompletionEvent` flags: `external`
(`cudaEventRecordExternal`) is Invalid. An interprocess or IPC-imported
event is Invalid. The `external` field stays the CUDA flags word (always
false). Decode identity stays no launch-attribute event. This VM does not
invent a `flags: u32` field, disable-timing as a requirement, NvSciSync /
interop event kinds, or an Engine flag for External. `gpu-profile capture`
is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — device-updatable kernel-node restrictions

`gpu-sim` CUDA `cudaLaunchAttributeDeviceUpdatableKernelNode` restrictions:
once true, SetAttribute to false is Invalid. The node cannot be destroyed.
CopyAttributes cannot involve it. A definition with such a node cannot be
instantiated twice. `cudaGraphExecUpdate` of an exec or source that has
one is Invalid. Opt-in and skip-reupload stay. This VM does not invent
`CUgraphDeviceNode` / device-side kernel updates. `gpu-profile capture`
is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — programmatic event triggerAtBlockStart

`gpu-sim` CUDA `cudaLaunchAttributeProgrammaticEvent.triggerAtBlockStart`:
`ProgrammaticEvent::trigger_at_block_start` records the event when the
kernel starts (this VM does not model per-block begins). Default `false`
keeps the PDL-trigger / kernel-completion identity.
`PdlLaunch::trigger_event` and `expertvm sim --programmatic-event` stay
false. Other streams may `wait_event` earlier than the PDL trigger.
Get/Set/CopyAttributes and capture carry the field. Debug-dot dumps
`pde-block-start` when set. Kernel duration is unchanged. This VM does
not invent an Engine flag for programmatic-event block-start.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — graph kernel-node SynchronizationPolicy

`gpu-sim` CUDA `cudaLaunchAttributeSynchronizationPolicy` on graph kernel
nodes: `KernelNodeAttr::SynchronizationPolicy` is CUDA 13
`CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY`. Valid for streams (already)
and graph nodes, not host launches (`KernelAttrs` stays without it). Typed
`graph_add_kernel` and capture snapshot the stream policy. `Auto` inherits
the recording stream at `cudaEventSynchronize` of that node's
launch-completion or programmatic event. An explicit Spin / Yield /
BlockingSync taxes that event wait even when the launch stream is Auto.
Kernel duration is unchanged. Get/Set/CopyAttributes. Capture cannot
include Set. This VM does not invent an Engine flag for kernel-node sync
policy. `gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-09-01 — store GEMM graphs inherit launch-stream green ctx

Capture from a green-ctx stream snapshots `CUDA_KERNEL_NODE_PARAMS.ctx`
(CUDA). Store per-leaf GEMM graphs clear that ctx before instantiate so
occupancy follows the launch stream. An expert captured during prefill
still uses the decode partition when leftover decode replays it, so
complementary `--green-ctx` spans can overlap under exclusive compute.
Walker combo parents with several kernels skip the unique-kernel patch.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — expertvm combo graph-mem MOVE children

Walker `--graph-build` combo parents with `--graph-mem` / `--graph-auto-free`
MOVE an uninstantiated `cudaGraphClone` of each GraphBank leaf. CUDA clone
ownership cannot name a child with mem alloc/free. `clone_graph` of an
instantiated exec forks the definition's mem ids. Parent
`AutoFreeOnLaunch` is inherited by MOVE children so auto-free combo
scratch still frees. Destroying a parent exec destroys those moved
child definitions. Kernel-only combos stay typed `graph_add_child`
(clone). `gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-09-01 — CUgraphChildGraphNodeOwnership

`gpu-sim` CUDA `CUgraphChildGraphNodeOwnership`:
`ChildGraphNodeParams::ownership` is clone versus move. Typed
`graph_add_child` stays clone of an instantiated child without mem
alloc/free or conditional nodes. Move lets a parent own an uninstantiated
child that may contain mem nodes. GetParams of a moved node reports
`INVALID`. After the move the child cannot be independently instantiated,
launched, destroyed, cloned, updated, or added as a child of another
parent. This VM does not invent an Engine flag for child-graph ownership.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — cuGraphAddMemsetNode ctx

`gpu-sim` CUDA `cuGraphAddMemsetNode` ctx: `MemsetNodeParams::ctx` is the
extra driver `CUcontext` argument on `cuGraphAddMemsetNode` /
`cuGraphExecMemsetNodeSetParams`. Stored on the graph step, not
`Kind::Memset` or `MemsetOp`. Typed `graph_add_memset` stays `None`. Typed
`graph_memset_set_params` does not clear ctx. Fills use copy engines, so
duration is unchanged. Unknown or destroyed is Invalid `"unknown green
ctx"`. Device mismatch is Invalid `"green ctx device"`. Parameter, not
topology. Capture from a green-ctx stream snapshots that ctx. This VM
does not invent an Engine flag for memset ctx. `gpu-profile capture` is
still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — cuGraphAddMemcpyNode ctx

`gpu-sim` CUDA `cuGraphAddMemcpyNode` ctx: `MemcpyNodeParams::ctx` is the
extra driver `CUcontext` argument on `cuGraphAddMemcpyNode` /
`cuGraphExecMemcpyNodeSetParams`. Stored on the graph step, not
`Kind::Memcpy` or `MemcpyOp`. Typed `graph_add_memcpy` stays `None`. Typed
`graph_memcpy_set_params` does not clear ctx. Copies use copy engines, so
duration is unchanged. Unknown or destroyed is Invalid `"unknown green
ctx"`. Device mismatch is Invalid `"green ctx device"`. Parameter, not
topology. Capture from a green-ctx stream snapshots that ctx. This VM
does not invent an Engine flag for memcpy ctx. `gpu-profile capture` is
still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA_CONDITIONAL_NODE_PARAMS.ctx

`gpu-sim` CUDA `CUDA_CONDITIONAL_NODE_PARAMS.ctx`:
`GraphNodeParams::If` / `IfElse` / `While` / `Switch` `ctx` is
`CUDA_CONDITIONAL_NODE_PARAMS.ctx`. `graph_conditional_create_with_ctx`
is `cuGraphConditionalHandleCreate`'s ctx argument. Node ctx must match
the handle (Invalid `"conditional ctx"`). Stored on the graph step, not
`Kind::If`. Unknown or destroyed is Invalid `"unknown green ctx"`. Device
mismatch is Invalid `"green ctx device"`. Parameter, not topology.
Conditionals do not occupy SMs, so duration is unchanged. Typed
`graph_add_if` copies the handle ctx. Typed create stays `None`. This VM
does not invent an Engine flag for conditional ctx. `gpu-profile capture`
is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA_BATCH_MEM_OP_NODE_PARAMS.ctx

`gpu-sim` CUDA `CUDA_BATCH_MEM_OP_NODE_PARAMS.ctx`:
`BatchMemOpNodeParams::ctx` is `CUDA_BATCH_MEM_OP_NODE_PARAMS.ctx`.
Stored on the graph step, not `Kind::BatchMem`. `None` inherits the launch
stream. Wait/write/flush do not occupy SMs, so duration is unchanged.
Unknown or destroyed is Invalid `"unknown green ctx"`. Device mismatch is
Invalid `"green ctx device"`. Parameter, not topology. Capture from a
green-ctx stream snapshots that ctx. Typed `graph_add_batch_mem_op` stays
`None`. Item-list SetParams does not clear ctx. No Engine `--batch-ctx`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA_KERNEL_NODE_PARAMS.sharedMemBytes

`gpu-sim` CUDA `CUDA_KERNEL_NODE_PARAMS.sharedMemBytes`:
`KernelNodeParams::shared_mem_bytes` is `sharedMemBytes`. Stored on the
graph step, not `Kind::Kernel`. GetParams / SetParams / AddNode carry the
field. Typed `graph_add_kernel` stays `0`. `KernelNodeAttr::DynamicShared`
Get/SetAttribute stays. CopyAttributes does not copy it. Oversize without
func attr / AllowNonPortable is Invalid `"dynamic shared"` /
`"non-portable shared"`. Parameter, not topology. Duration is bank width,
not byte count. No Engine `--kernel-shared`. `gpu-profile capture` is
still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA_KERNEL_NODE_PARAMS.ctx

`gpu-sim` CUDA 13 `CUDA_KERNEL_NODE_PARAMS.ctx`: `KernelNodeParams::ctx`
is `CUDA_KERNEL_NODE_PARAMS.ctx` / `cudaKernelNodeParamsV2.ctx`. Stored
on the graph step, not `Kind::Kernel`. `None` inherits the launch stream.
`Some` pins duration and SM occupancy to that live green context.
Unknown or destroyed is Invalid `"unknown green ctx"`. Device mismatch
is Invalid `"green ctx device"`. Parameter, not topology. Capture from a
green-ctx stream snapshots that ctx. Typed `graph_add_kernel` stays
`None`. No Engine `--kernel-ctx`. This VM does not invent
`cuCtxFromGreenCtx`. `gpu-profile capture` is still refused. Dual score
still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA cuDevSmResourceSplit

`gpu-sim` `cuDevSmResourceSplit`: `dev_sm_resource_split` is
`cuDevSmResourceSplit`. `DevSmResourceGroupParams` `smCount` is ‰ of the
chip (same unit as `dev_sm_resource_split_by_count`). `0` is discovery
(remaining ‰). `coscheduledSmCount` / `preferredCoscheduledSmCount` must
be 0. Group flags must be 0 (`BACKFILL` is Invalid). API flags must be
0. Unequal groups; leftover is remaining. Query; legal during capture.
Typed `dev_sm_resource_split_by_count` stays. No Engine `--sm-split`.
`gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-09-01 — CUDA_MEMCPY3D srcLOD / dstLOD

`gpu-sim` CUDA_MEMCPY3D `srcLOD` / `dstLOD`: `MemcpyOp` `src_lod` /
`dst_lod` must be 0. CUDA arrays are not modeled. Nonzero is Invalid
`"memcpy lod"`. Default 0 is identical. No Engine `--memcpy-lod`.
`gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-09-01 — CUDA cuMemcpy3DUnaligned

`gpu-sim` `cuMemcpy3DUnaligned`: `memcpy_3d_unaligned` is
`cuMemcpy3DUnaligned`. Identity with `memcpy_3d`: this VM does not require
CUDA 3D pitch/offset alignment. Host-synchronous; capture cannot include
it. CUDA has no Async Unaligned; `memcpy_3d_async` stays. No Engine
`--memcpy-3d-unaligned`. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA_MEMCPY2D / 3D srcPos / dstPos

`gpu-sim` CUDA_MEMCPY2D / CUDA_MEMCPY3D srcPos / dstPos: `MemcpyOp`
`src_x` / `src_y` / `src_z` / `dst_x` / `dst_y` / `dst_z` are
`srcXInBytes` / `srcY` / `srcZ` / `dstXInBytes` / `dstY` / `dstZ`.
Default 0 is origin (0,0[,0]). 1D with any origin, or 2D with a z
origin, is Invalid `"memcpy origin"`. 2D/3D `x + width` vs pitch stays
`"memcpy2d pitch"` / `"memcpy3d pitch"`. Oversized 2D `srcY` is
`"memcpy range past alloc"`. Capture GetParams preserves origin. No
Engine `--memcpy-origin`. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA cuMemAllocPitch

`gpu-sim` `cuMemAllocPitch`: `malloc_pitch_with_element_size` is
`cuMemAllocPitch`. `element_size` is CUDA `ElementSizeBytes` and must be
4, 8, or 16. Pitch stays 512-aligned. `malloc_pitch` stays
`cudaMallocPitch`. Host-sync; capture cannot include it. No Engine
`--malloc-pitch-element`. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA cuMemcpyHtoD / DtoH

`gpu-sim` `cuMemcpyHtoD` / `cuMemcpyDtoH`: `memcpy_htod` is `cuMemcpyHtoD`
(host-synchronous pinned H2D). `memcpy_dtoh` is `cuMemcpyDtoH`
(host-synchronous Device→HostPinned). Capture cannot include them.
`memcpy_pinned_to_device` / `memcpy_device_to_pinned` stay
`cuMemcpyHtoDAsync` / `cuMemcpyDtoHAsync`. Pageable helpers stay.
`memcpy_sync` stays generic `cudaMemcpy`. No Engine `--memcpy-htod`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA VMM allocation prop usage

`gpu-sim` `CUmemAllocationProp` usage / compression:
`MemAllocationProp::usage` must be `MemHandleUsage::NONE`;
`MemAllocationProp::compression` must be 0. `va_create_with_prop` rejects
HW decompress and nonzero compression. Get always reports none. Distinct
from pool `MemPoolProps::usage`. Typed `va_create` stays. No Engine
`--vmm-usage`. `gpu-profile capture` is still refused. Dual score still
has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA mempool props usage

`gpu-sim` `CUmemPoolProps::usage`: `MemPoolProps::usage` is
`cudaMemPoolProps::usage`. Must be `MemHandleUsage::NONE`.
`MemHandleUsage::HW_DECOMPRESS` is Invalid (hardware decompress is not
modeled). Typed `create_pool` stays. No Engine `--pool-usage`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA cuMemcpy2DUnaligned

`gpu-sim` `cuMemcpy2DUnaligned`: `memcpy_2d_unaligned` is
`cuMemcpy2DUnaligned`. Identity with `memcpy_2d`: this VM does not require
CUDA 2D pitch/offset alignment. Host-synchronous; capture cannot include
it. CUDA has no Async Unaligned; `memcpy_2d_async` stays. No Engine
`--memcpy-unaligned`. `gpu-profile capture` is still refused. Dual score
still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA cuMemsetD2D8

`gpu-sim` `cuMemsetD2D8`: `memset_d2d8_async` / `memset_d2d8` are
`cuMemsetD2D8Async` / `cuMemsetD2D8`. `width` is CUDA `Width`; row payload
is `width` bytes. `pitch` is bytes. `memset_2d_async` stays byte-width.
Capture of Async is legal; host-sync is refused. Fill value is not modeled.
No Engine `--memset-d2d`. `gpu-profile capture` is still refused. Dual score
still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA graph AddNode mem-alloc accessDescs

`gpu-sim` `cudaGraphAddNode` mem-alloc accessDescs: `GraphNodeParams::Alloc`
carries `accessDescs` (`bytes` plus `MemAccessDesc` list). `graph_add_node`
with empty access is `graph_add_alloc`; peer PROT_READ / PROT_READ_WRITE
matches `graph_add_alloc_with_access` (kernel without dest HBM). GetParams
`graph_node_get_params` / `graph_exec_node_get_params` return stored
accessDescs. Typed `graph_add_alloc_with_access` stays. SetParams of Alloc
stays Invalid. No Engine `--graph-alloc-access`. `gpu-profile capture` is
still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA cuMemsetD2D16 / D2D32

`gpu-sim` `cuMemsetD2D16` / `cuMemsetD2D32`: `memset_d2d16_async` /
`memset_d2d16` are `cuMemsetD2D16Async` / `cuMemsetD2D16`.
`memset_d2d32_async` / `memset_d2d32` are `cuMemsetD2D32Async` /
`cuMemsetD2D32`. `width` is CUDA `Width`; row payload is `width * 2` /
`width * 4`. `pitch` is bytes. `memset_2d_async` stays byte-width. Capture
of Async is legal; host-sync is refused. Fill value is not modeled. No
Engine `--memset-d2d`. `gpu-profile capture` is still refused. Dual score
still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA cuMemsetD16 / D32

`gpu-sim` `cuMemsetD16` / `cuMemsetD32`: `memset_d16_async` / `memset_d16`
are `cuMemsetD16Async` / `cuMemsetD16`. `memset_d32_async` / `memset_d32`
are `cuMemsetD32Async` / `cuMemsetD32`. `count` is CUDA `N`; payload is
`count * 2` / `count * 4`. Overflow is Invalid. Typed `memset` stays
byte-counted. Capture of Async is legal; host-sync is refused. Fill value
is not modeled. No Engine `--memset-d16`. `gpu-profile capture` is still
refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA graph mem-alloc accessDescs

`gpu-sim` `cudaGraphAddMemAllocNode` accessDescs:
`graph_add_alloc_with_access` is `cudaMemAllocNodeParams::accessDescs`.
Empty is `graph_add_alloc`. Peer PROT_READ / PROT_READ_WRITE lets a live
kernel on that GPU use the graph alloc without dest HBM (graph-memory pool
stays Invalid for `pool_set_access`). Host location and unknown flags are
Invalid. Applied when the alloc is live, not at add. Last descriptor for a
device wins. All-or-nothing before the node is created. Capture cannot
include it. `graph_alloc_get_access` / `graph_exec_alloc_get_access` are
GetParams accessDescs. SetParams of Alloc stays Invalid. No Engine
`--graph-alloc-access`. `gpu-profile capture` is still refused. Dual score
still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA graph AddNode v2 edge data

`gpu-sim` `cuGraphAddNode_v2`: `graph_add_node_with_data` is
`cudaGraphAddNode` with `dependencyData`. `deps` and `data` must match.
Default type with ports 0 is `graph_add_node`. Programmatic type stays
Invalid. Edge checks run before the node is created. Capture cannot
include it. No Engine `--graph-add-node-data`. `gpu-profile capture` is
still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA primary-context GetId

`gpu-sim` `cuCtxGetId`: `ctx_get_id` is `cuCtxGetId` for the seeded
primary context of an explicit `DeviceId` (no TLS current device / no
`CUcontext` object). Distinct from `green_ctx_get_id` / `stream_get_id` /
`event_get_id`. Unknown devices are Invalid. Query; legal during capture.
No Engine `--ctx-id`. `gpu-profile capture` is still refused. Dual score
still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA wait-value FLUSH on RDMA

`gpu-sim` `CU_STREAM_WAIT_VALUE_FLUSH`: `WaitValueFlags::FLUSH` follows a
wait with a stream-ordered remote write flush on an RDMA SKU (same rule
as `BatchMemOp::FlushRemoteWrites`). Stored on the wait; parameter, not
topology. Unknown bits stay Invalid. Non-RDMA is Invalid. Capture legal.
No Engine `--wait-flush`. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA stream-ordered RDMA flush

`gpu-sim` `CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES`:
`BatchMemOp::FlushRemoteWrites` is stream-ordered 1 ns Solo on an RDMA
SKU (`cuStreamBatchMemOp`). Distinct from host-sync
`flush_gpu_direct_rdma_writes` (capture refused). Write visibility is
not modeled; flush is never a no-op. Non-RDMA is Invalid. Capture
records a batch-mem-op node. No Engine `--flush-remote`. `gpu-profile
capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA batch pointer GetAttributes

`gpu-sim` `cuPointerGetAttributes`: `pointer_get_attribute_n` is a batch
`pointer_get_attribute`. Distinct from `pointer_get_attributes`
(`cudaPointerGetAttributes` struct). Empty is `Ok([])` after a live
alloc. All-or-nothing. ACCESS_FLAGS stays `pointer_get_access_flags`.
Query; legal during capture. No Engine `--pointer-attrs`. `gpu-profile
capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA memset elementSize 1/2/4

`gpu-sim` `cudaMemsetNodeParams::elementSize`: `MemsetOp::element_size`
is `1` / `2` / `4` (`cuMemsetD8` / `D16` / `D32`). Typed `memset` stays
`1`. Offset, width, and nonzero pitch must divide that size. The fill
value is not modeled. No Engine `--memset-element`. `gpu-profile capture`
is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA pointer AccessFlags on an explicit device

`gpu-sim` `CU_POINTER_ATTRIBUTE_ACCESS_FLAGS`:
`pointer_get_access_flags` reports `MemAccessFlags` for an explicit
`DeviceId` (no TLS current device). Flags are this VM's kernel residency,
not `cudaDeviceEnablePeerAccess` (D2D memcpy only). Mapped host is
ReadWrite; pool ProtRead / VMM `va_set_access` / managed SetAccessedBy
are Read. Not a `PointerAttr`. Query; legal during capture. No Engine
`--pointer-access`. CONTEXT / P2P tokens stay unmodeled. `gpu-profile
capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA ExecUpdate copies conditional handles

`gpu-sim` `cudaGraphExecUpdate` copies IF / IfElse / WHILE / SWITCH
handles when bodies match. Handle is a parameter (same as SetParams).
Type, size, and bodies stay topology. No Engine `--graph-cond-update`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA conditional-node handle SetParams

`gpu-sim` `cudaGraphConditionalNodeSetParams` /
`cudaGraphExecConditionalNodeSetParams` retarget the IF / IfElse / WHILE /
SWITCH handle. Type, size, and bodies stay topology. Definition SetParams
does not retarget the exec. Capture cannot include it. No Engine
`--graph-cond-set`. `gpu-profile capture` is still refused. Dual score
still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA size-2 IF else-body list

`gpu-sim` `graph_if_else_nodes` lists size-2 IF nodes as `(index,
handle, then, else)`. `graph_if_nodes` stays then-body. Query; legal
during capture. Instantiated ids use the exec snapshot. No Engine
`--graph-if-else`. `gpu-profile capture` is still refused. Dual score
still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA exec kernel CopyAttributes

`gpu-sim` `graph_exec_kernel_node_copy_attributes` is exec-snapshot
`cudaGraphKernelNodeCopyAttributes`. Uninstantiated graphs are Invalid.
After instantiate this copies the launched attributes. Definition
CopyAttributes does not retarget the exec. `sharedMemBytes` stays
params, not CopyAttributes. Capture cannot include it. No Engine
`--graph-exec-copy-attrs`. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA exec mem-node GetParams

`gpu-sim` `graph_exec_alloc_get_params` /
`graph_exec_free_get_params` are exec-snapshot
`cudaGraphMemAllocNodeGetParams` / `MemFreeNodeGetParams`. Uninstantiated
graphs are Invalid. After instantiate this is the launched node. View
GetParams stays. Query; legal during capture. No Engine
`--graph-exec-mem`. Alloc SetParams stays parked. `gpu-profile capture`
is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA exec event GetEvent

`gpu-sim` `graph_exec_event_record_get_event` /
`graph_exec_event_wait_get_event` are exec-snapshot
`cudaGraphEventRecordNodeGetEvent` / `WaitNodeGetEvent`. Uninstantiated
graphs are Invalid. After instantiate this is the launched event. View
GetEvent stays. Query; legal during capture. No Engine
`--graph-exec-event`. `gpu-profile capture` is still refused. Dual score
still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA exec child-graph GetParams

`gpu-sim` `graph_exec_child_get_graph` is
`cudaGraphExecChildGraphNodeGetParams`. Uninstantiated graphs are
Invalid. After instantiate this is the launched child.
`graph_child_get_graph` stays a view. Query; legal during capture. No
Engine `--graph-exec-child`. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaGraphAddNode` SWITCH params

`gpu-sim` `GraphNodeParams::Switch` is `cudaGraphAddNode` SWITCH.
`GraphAddNode::switch_bodies` is Copy `phGraph_out` (`n` is `1..=64`).
Typed `graph_add_switch` stays. GetParams is handle plus branch count.
SetParams is Invalid `"conditional node params"`. `gpu-profile capture`
is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaGraphAddNode` IF/WHILE params

`gpu-sim` `GraphNodeParams::If` / `IfElse` / `While` are
`cudaGraphAddNode` conditional params. `GraphAddNode::body` /
`else_body` are `phGraph_out`. Typed helpers stay. SWITCH stays
`graph_add_switch`. GetParams is handle-only. SetParams is Invalid
`"conditional node params"`. No Engine `--graph-if-else`. `gpu-profile
capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaGraphCondTypeIf` size 2

`gpu-sim` `graph_add_if_else` is `cudaGraphCondTypeIf` size 2. Then-body
skips when the handle is `0`; the else-body runs instead. `graph_add_if`
stays size 1. Capture cannot include it. Illegal on an instantiated exec.
No Engine `--graph-if-else`. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaGraphInstantiateWithParams` `hUploadStream`

`gpu-sim` `GraphInstantiateParams::upload_stream` is
`cudaGraphInstantiateWithParams` `hUploadStream`. With
`GraphInstantiateFlags::UPLOAD`, `Some` enqueues `upload_graph_async`
(uploaded when the op completes). `None` stays host-sync
`upload_graph`. Ignored when UPLOAD is unset.
`instantiate_graph_with_flags` stays host-sync. No Engine
`--graph-upload-stream`. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA stream-ordered `cudaGraphUpload`

`gpu-sim` `upload_graph_async` is `cudaGraphUpload` on a stream.
Stream-ordered Solo `graph_upload_ns`; uploaded when the op completes.
Already uploaded at start is 1 ns. Capture cannot include it. Typed
`upload_graph` stays host-sync. `launch_graph` waits an in-flight
upload instead of a second host-sync upload. No Engine
`--graph-upload-stream`. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaMemRangeGetAttribute` dataSize

`gpu-sim` `mem_range_get_attribute_with_data_size` /
`mem_range_get_attributes_with_data_sizes` are
`cudaMemRangeGetAttribute` / `GetAttributes` `dataSize` / `dataSizes`.
Scalar attrs need 4 bytes. AccessedBy needs 4 bytes per device plus a
terminator. Smaller Invalid `"range data size"`. Length mismatch
Invalid `"range data sizes"`. Typed helpers stay. Query; legal during
capture. No Engine `--range-data-size`. `gpu-profile capture` is still
refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaHostRegister` size

`gpu-sim` `host_register_with_size` is `cudaHostRegister` with the CUDA
`size` argument. `size` must equal the allocation bytes. Other sizes
Invalid `"register size"`. Partial register is not modeled. Typed
helpers stay. Capture cannot include it. No Engine `--register-size`.
`gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaMemRangeGetAttribute` count

`gpu-sim` `mem_range_get_attribute_with_size` /
`mem_range_get_attributes_with_size` are `cudaMemRangeGetAttribute` /
`GetAttributes` with the CUDA `count` argument. `size` must equal the
allocation bytes. Other sizes Invalid `"range size"`. Partial range
queries are not modeled. Typed helpers stay. Query; legal during
capture. No Engine `--range-size`. `gpu-profile capture` is still
refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaStreamAttachMemAsync` length

`gpu-sim` `stream_attach_with_size` is `cudaStreamAttachMemAsync` with
the CUDA `length` argument. `size` `0` is the entire allocation. A
nonzero `size` must equal the allocation bytes. Other sizes Invalid
`"attach size"`. Partial attach is not modeled. Typed `stream_attach`
stays (`length` 0). Capture cannot include it. No Engine `--attach-size`.
`gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaMemPrefetchAsync` count

`gpu-sim` `prefetch_with_size` / `prefetch_host_with_size` are
`cudaMemPrefetchAsync` with the CUDA `count` argument. `size` must equal
the allocation bytes. Other sizes Invalid `"prefetch size"`. Partial
prefetch is not modeled. Typed `prefetch` / `prefetch_host` stay.
Capture may record the memcpy. No Engine `--prefetch-size`.
`gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaMemAdvise` count

`gpu-sim` `mem_advise_with_size` is `cudaMemAdvise` with the CUDA `count`
argument. `size` must equal the allocation bytes. Other sizes Invalid
`"advise size"`. Partial advise is not modeled. Typed `mem_advise` stays.
Capture cannot include it. No Engine `--advise-size`. `gpu-profile
capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaMemPoolSetAccess` ProtRead

`gpu-sim` `pool_set_access_read` is `cudaMemPoolSetAccess`
`cudaMemAccessFlagsProtRead`. Peer kernels may read without dest HBM;
writes and memset stay `NotResident` until ReadWrite.
`pool_set_access_with_flags` / `pool_set_access_n` accept `PROT_READ`.
`pool_get_access` returns `PROT_READ` on those peers. Owner stays
ReadWrite. Capture cannot include it. Graph-memory pools stay Invalid.
No Engine `--pool-prot-read`. `gpu-profile capture` is still refused.
Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cuDeviceGetExecAffinitySupport`

`gpu-sim` `device_get_exec_affinity_support` is
`cuDeviceGetExecAffinitySupport`. `SM_COUNT` is 0 (this VM uses permille
green-context spans, not occupancy SM counts). Other type ids Invalid
`"exec affinity type"`. Query; legal during capture. Do not invent
`cuCtxSetExecAffinity`. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cuMemPoolGetId`

`gpu-sim` `pool_get_id` is `cuMemPoolGetId`. Unique per `PoolId` handle
(`PoolId + 1`). Distinct from the handle, `event_get_id`, and
`stream_get_id`. Query; legal during capture. Graph-memory pools are
legal (distinct from `pool_get_attribute`). Destroyed is Invalid.
Imported shareable pools differ from the exporter. Recreate after destroy
returns a new id. `gpu-profile capture` is still refused. Dual score
still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaGraphConditionalHandleCreate` flags

`gpu-sim` `graph_conditional_create_with_flags` is
`cudaGraphConditionalHandleCreate` with the CUDA flags word.
`GraphCondFlags::ASSIGN_DEFAULT` (`cudaGraphCondAssignDefault`) is
identity with `graph_conditional_create` (each launch resets to the
create-time default). Flags 0 keeps the handle across launches. Unknown
bits Invalid `"graph cond flags"`. Capture cannot include it. No Engine
`--graph-cond-flags`. `gpu-profile capture` is still refused. Dual score
still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cuEventGetId`

`gpu-sim` `event_get_id` is `cuEventGetId` / `cudaEventGetId`. Unique per
`EventId` handle (`EventId + 1`). Distinct from the caller-chosen
handle, `stream_get_id`, and `event_get_flags`. Query; legal during
capture. Recreate after destroy returns the same id. `gpu-profile
capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaGraphCreate` flags

`gpu-sim` `create_graph_with_flags` is `cudaGraphCreate` with the CUDA
flags word. Flags 0 is identity with `create_graph`. Unknown bits
Invalid `"graph create flags"`. Capture cannot include it. Distinct
from `GraphInstantiateFlags`. No Engine `--graph-create-flags`.
`gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaDeviceProp::persistingL2CacheMaxSize`

`gpu-sim` `DeviceProperties::persisting_l2_cache_max_size` is
`cudaDeviceProp::persistingL2CacheMaxSize` /
`DeviceAttr::MaxPersistingL2CacheSize` (L2 bytes). Distinct from
`access_policy_max_window_size` and from the current
`persisting_l2_cache_size` limit (default 0). Query; legal during
capture. `gpu-profile capture` is still refused. Dual score still has
no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaStreamGetCaptureInfo_v3`

`gpu-sim` `StreamCaptureInfo::edge_data` is
`cudaStreamGetCaptureInfo_v3` (`GraphEdgeData` parallel to
`dependencies`). Existing capture deps are Default, ports 0. Query
during capture. Distinct from `graph_edges_with_data` and
`graph_node_deps_with_data`. No Engine `--graph-edge-data`. Do not
invent `cudaStreamUpdateCaptureDependencies` v2 edgeData.
`gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaGraphNodeGetDependencies` v2

`gpu-sim` `graph_node_deps_with_data` / `graph_node_dependents_with_data`
are `cudaGraphNodeGetDependencies` / `GetDependentNodes` v2
(`GraphEdgeData`). Existing edges are Default, ports 0 (identity with
`graph_node_deps` / `graph_node_dependents`). Query; legal during
capture. Distinct from `graph_edges_with_data`. No Engine
`--graph-edge-data`. `gpu-profile capture` is still refused. Dual score
still has no `$/M tokens`.

## Shipped 2026-09-01 — CUDA `cudaGraphDebugDotFlagsRuntimeTypes`

`gpu-sim` `GraphDebugDotFlags::RUNTIME_TYPES` is
`cudaGraphDebugDotFlagsRuntimeTypes` (`1 << 1`). Debug-dot labels use
CUDA runtime `cudaGraphNodeType*` names. Flags `0` stays `GraphNodeKind`
Debug names. VM-only kinds keep Debug names. `VERBOSE` includes
RuntimeTypes. Query; legal during capture. Distinct from param-class
dumps and HANDLES. No Engine `--graph-debug-dot`. `gpu-profile capture`
is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaDeviceGetNvSciSyncAttributes`

`gpu-sim` `device_get_nvscisync_attributes` is
`cudaDeviceGetNvSciSyncAttributes`. Always Invalid `"nvscisync not modeled"`
(`TimelineSemaphoreInteropSupported` is 0). SIGNAL / WAIT flags. Distinct
from `ipc_get_event` and `wait_value32`. Query; legal during capture. No
Engine `--nvscisync`. `gpu-profile capture` is still refused. Dual score
still has no `$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaGraphAddDependencies` v2

`gpu-sim` `graph_add_dependencies_n_with_data` is
`cudaGraphAddDependencies` v2 (`GraphEdgeData`). Default type with ports
0 is identity. Programmatic type is Invalid. `graph_edges_with_data` is
`cudaGraphGetEdges` v2. Query; legal during capture. Capture cannot
include Add. Distinct from kernel-node PDL. No Engine `--graph-edge-data`.
`gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-31 — CUDA `cuMemExportToShareableHandle`

`gpu-sim` `va_export_to_shareable_handle` is
`cuMemExportToShareableHandle`. Always Invalid `"not shareable"`
(VMM create-time handle types are none). POSIX-FD only. Distinct from
`ipc_get`, `pool_export`, and dma-buf `va_get_handle_for_address_range`.
Capture cannot include it. No Engine `--vmm-export`.
`gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-31 — CUDA `cuMemGetHandleForAddressRange`

`gpu-sim` `va_get_handle_for_address_range` is
`cuMemGetHandleForAddressRange`. Always Invalid `"dma-buf not modeled"`
(`DmaBufSupported` is 0). DMA_BUF_FD only. Distinct from `ipc_get` and
shareable pools. Query; legal during capture. No Engine `--dma-buf`.
`gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaDevAttrMaxPitch`

`gpu-sim` `DeviceAttr::MaxPitch` is `DeviceAttr::MAX_PITCH` (`i32::MAX`).
This VM does not cap 2D memcpy / `cudaMallocPitch` pitch. Distinct from
`TexturePitchAlignment`. Also `DeviceProperties.mem_pitch`. Query; legal
during capture. `gpu-profile capture` is still refused. Dual score still
has no `$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaInitDevice`

`gpu-sim` `init_device` / `init_device_with_flags` are `cudaInitDevice`.
Primary ctx is already seeded; this does not make a thread-current
device. `FLAGS_ARE_VALID` applies `deviceFlags` like `set_device_flags`;
without that bit they are ignored. Host-sync 1 ns. Capture cannot include
it. No Engine `--init-device`. `gpu-profile capture` is still refused.
Dual score still has no `$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaDevP2PAttrOnlyPartialNativeAtomicSupported`

`gpu-sim` `DeviceP2pAttr::OnlyPartialNativeAtomicSupported` is always 0.
P2P native atomics are not modeled. Distinct from `NativeAtomicSupported`
and from `OnlyPartialHostNativeAtomicSupported`. Query; legal during
capture. Do not invent `cudaDeviceGetP2PAtomicCapabilities`.
`gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaStreamAddCallback`

`gpu-sim` `stream_add_callback` / `stream_add_callback_params` /
`stream_add_callback_with_flags` are `cudaStreamAddCallback`. Same
stream-ordered host enqueue as `host_func` (`cudaLaunchHostFunc`):
bills `host_func_ns`, does not occupy compute or copy. Cannot be
captured. Flags must be 0. Distinct from capturable `host_func`. No
Engine `--stream-callback`. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaDevAttrMaxTexture1DWidth`

`gpu-sim` `DeviceAttr::MaxTexture1DWidth` is always 0. CUDA arrays /
textures are not modeled. Distinct from `TextureAlignment`. Also
`DeviceProperties.max_texture_1d_width`. Query; legal during capture.
`gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaDevAttrTexturePitchAlignment`

`gpu-sim` `DeviceAttr::TexturePitchAlignment` is always 0. CUDA textures
are not modeled. Distinct from `TextureAlignment` and from `MemcpyOp` 2D
pitches (which this VM does model). Also
`DeviceProperties.texture_pitch_alignment`. Query; legal during capture.
`gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaDevAttrSurfaceAlignment`

`gpu-sim` `DeviceAttr::SurfaceAlignment` is always 0. CUDA surfaces are
not modeled. Distinct from `TextureAlignment`. Also
`DeviceProperties.surface_alignment`. Query; legal during capture.
`gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaDevAttrTextureAlignment`

`gpu-sim` `DeviceAttr::TextureAlignment` is always 0. CUDA arrays /
textures are not modeled. Distinct from `SparseCudaArraySupported`. Also
`DeviceProperties.texture_alignment`. Query; legal during capture.
`gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaDevAttrTotalConstantMemory`

`gpu-sim` `DeviceAttr::TotalConstantMemory` is always 0. Constant memory
is not modeled. Distinct from `TotalGlobalMem` (HBM). Also
`DeviceProperties.total_constant_memory`. Query; legal during capture.
`gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaDevAttrReservedSharedMemoryPerBlock`

`gpu-sim` `DeviceAttr::ReservedSharedMemoryPerBlock` is always 0.
Driver-reserved shared memory is not modeled. Distinct from
`MaxSharedMemoryPerBlock` (profile bytes). Also
`DeviceProperties.reserved_shared_mem_per_block`. Query; legal during
capture. `gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaDeviceSetCacheConfig`

`gpu-sim` `set_cache_config` / `get_cache_config` are
`cudaDeviceSetCacheConfig` / `GetCacheConfig`. `set_func_cache_config` /
`get_func_cache_config` are `cudaFuncSetCacheConfig` (per device; CUDA has
no FuncGet). Default PreferNone. PreferShared / PreferL1 / PreferEqual
are stored; L1 is not modeled, so kernel duration does not change.
Host-sync 1 ns on Set. Capture cannot include Set. Get is a query.
Distinct from SharedMemCarveout and SharedMemoryMode. No Engine
`--cache-config`. `gpu-profile capture` is still refused. Dual score still
has no `$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaDevAttrEccEnabled`

`gpu-sim` `DeviceAttr::EccEnabled` is always 0. ECC is not modeled.
Distinct from `TccDriver`. Also `DeviceProperties.ecc_enabled`. Query;
legal during capture. `gpu-profile capture` is still refused. Dual score
still has no `$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaDevAttrComputePreemptionSupported`

`gpu-sim` `DeviceAttr::ComputePreemptionSupported` is always 0. Kernel
preemption is not modeled. Distinct from `KernelExecTimeout`. Also
`DeviceProperties.compute_preemption_supported`. Query; legal during
capture. `gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaDevAttrLocalL1CacheSupported`

`gpu-sim` `DeviceAttr::LocalL1CacheSupported` is always 0. This VM does
not model L1 caches. Distinct from `GlobalL1CacheSupported`. Also
`DeviceProperties.local_l1_cache_supported`. Query; legal during
capture. `gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaDevAttrGlobalL1CacheSupported`

`gpu-sim` `DeviceAttr::GlobalL1CacheSupported` is always 0. This VM does
not model L1 caches. Distinct from `L2CacheSize`. Also
`DeviceProperties.global_l1_cache_supported`. Query; legal during
capture. `gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaDevAttrGPUDirectRDMAWritesOrdering`

`gpu-sim` `DeviceAttr::GpuDirectRdmaWritesOrdering` is always None. Native
write visibility is not modeled, so `flush_gpu_direct_rdma_writes` is never
a no-op. Distinct from `GpuDirectRdmaFlushWritesOptions` (Host on an RDMA
SKU). Also `DeviceProperties.gpu_direct_rdma_writes_ordering`. Query; legal
during capture. `gpu-profile capture` is still refused. Dual score still
has no `$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaDevAttrMaxAccessPolicyWindowSize`

`gpu-sim` `DeviceAttr::MaxAccessPolicyWindowSize` is L2 bytes, same as
`MaxPersistingL2CacheSize`. Also `DeviceProperties.access_policy_max_window_size`.
Query; legal during capture. `gpu-profile capture` is still refused. Dual
score still has no `$/M tokens`.

## Shipped 2026-08-31 — CUDA `cuDevicePrimaryCtxGetState`

`gpu-sim` `device_primary_ctx_get_state` is `cuDevicePrimaryCtxGetState`.
Flags match `get_device_flags`. Active is always true (this VM seeds a
primary context at construct; no lazy retain). Query; legal during
capture. Unknown is Invalid. No `cuDevicePrimaryCtxRetain` / `Release` /
`Reset`. `gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaDeviceGetByPCIBusId`

`gpu-sim` `device_get_by_pci_bus_id` is `cudaDeviceGetByPCIBusId`. A
matching synthetic PCI bus id from `device_get_pci_bus_id` returns that
device. Unknown or malformed is Invalid. Query; legal during capture.
Distinct from `device_get_by_uuid`. No `cuDeviceGetLuid`. `gpu-profile
capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaDeviceGetPciBusId`

`gpu-sim` `device_get_pci_bus_id` is `cudaDeviceGetPciBusId` /
`cuDeviceGetPCIBusId`. A live device returns a synthetic
`domain:bus:device.function` string. Distinct from `device_get_uuid` and
`device_get_name`. Same ordinal on two profiles share a bus id. Also
`DeviceProperties` PCI ids and `DeviceAttr::PciDomainId` / `PciBusId` /
`PciDeviceId`. Query; legal during capture. Unknown is Invalid. Not a
real PCI topology. No `cudaDeviceGetByPCIBusId` this slice. No
`cuDeviceGetLuid`. `gpu-profile capture` is still refused. Dual score
still has no `$/M tokens`.

## Shipped 2026-08-31 — CUDA `cuDeviceGetByUuid`

`gpu-sim` `device_get_by_uuid` is `cuDeviceGetByUuid`. A matching
synthetic UUID from `device_get_uuid` returns that device. Unknown UUID
is Invalid. Query; legal during capture. Distinct from `device_get`
(ordinal). No `cuDeviceGetLuid`. `gpu-profile capture` is still refused.
Dual score still has no `$/M tokens`.

## Shipped 2026-08-31 — CUDA `cuDeviceGetUuid`

`gpu-sim` `device_get_uuid` is `cuDeviceGetUuid` / `cudaDeviceGetUuid`. A
live device returns a synthetic 16-octet UUID for `(profile.name, device)`.
Distinct from `device_get_name` and `DeviceId`. Two `Sim`s with the same
profile agree; GPUs on an 8×H100 profile differ. Also
`DeviceProperties.uuid`. Query; legal during capture. Unknown is Invalid.
Not a real NVIDIA UUID. No `cuDeviceGetLuid`. `gpu-profile capture` is
still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-08-31 — CUDA `cuGraphNodeGetToolsId`

`gpu-sim` `graph_node_get_tools_id` is `cuGraphNodeGetToolsId`. A live graph
node returns a unique tools id (CUPTI-style). Distinct from
`graph_node_get_local_id` (debug-dot `n0`) and `graph_get_id`. A
definition, instantiate exec, and clone each assign different tools ids.
Query; legal during capture. Destroyed or unknown is Invalid.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-08-31 — CUDA `cuGraphNodeGetLocalId`

`gpu-sim` `graph_node_get_local_id` is `cuGraphNodeGetLocalId`. A live graph
node returns the id printed by debug-dot (`n0`). Together with
`graph_get_id` the pair uniquely identifies a node. Query; legal during
capture. Destroyed or unknown is Invalid. `gpu-profile capture` is still
refused. Dual score still has no `$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaGraphGetId`

`gpu-sim` `graph_get_id` is `cudaGraphGetId` / `cudaGraphExecGetId`. A live
graph or exec returns the unique id printed by debug-dot HANDLES. Distinct
from node indices. A definition, instantiate exec, and clone each have
their own id. Query; legal during capture. Unknown or destroyed is
Invalid. `gpu-profile capture` is still refused. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaEventGetFlags`

`gpu-sim` `event_get_flags` is `cudaEventGetFlags`. A live event returns
the create flags word (`DisableTiming` / `Interprocess` / `BlockingSync`).
Distinct from `event_timing` / `event_blocking_sync`. Query; legal during
capture. Unknown or destroyed is UnknownEvent. Implicit first-record
events are Default. IPC imports report Interprocess plus DisableTiming.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-08-31 — CUDA `cudaExecutionCtxGetDevice`

`gpu-sim` `green_ctx_get_device` is `cudaExecutionCtxGetDevice`. A live
green context returns the device passed to `green_ctx_create`. Distinct
from `green_ctx_get_id`, `stream_get_green_ctx`, and
`device_get_dev_resource`. Query; legal during capture. Unknown or
destroyed is Invalid. No `cuCtxFromGreenCtx`. No second `--green-ctx`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-08-31 — CUDA `cuGreenCtxGetId`

`gpu-sim` `green_ctx_get_id` is `cuGreenCtxGetId` / `cudaExecutionCtxGetId`.
A live green context returns a unique id for the life of the `Sim`. Distinct
from `GreenCtxId` (the VM handle), `stream_get_id`, and
`stream_get_green_ctx`. Query; legal during capture. Unknown or destroyed
is Invalid. No NULL-means-current. No `cuCtxFromGreenCtx`. No second
`--green-ctx`. `gpu-profile capture` is still refused. Dual score still
has no `$/M tokens`.

## Shipped 2026-08-31 — official shared-expert `{1}` `ffn_up_shexp.scale`

llama.cpp `build_ffn` applies `ggml_mul` after the shared-expert up GEMM
(`TENSOR_NOT_REQUIRED`; missing is `1.0`) before the gated product. Decode
loads `blk.{i}.ffn_up_shexp.scale` only when `ffn_up_shexp.weight` is present
(qwen2moe / qwen3next / llama4). Writer-tiny
`tiny-qwen2moe-ffn-up-shexp-s` is `tiny-qwen2moe` plus scale `2.75`.
Llama / qwen3moe files that contain `ffn_up_shexp.scale` are ignored (not
loaded). Dense `ffn_up.scale` and shared-expert `ffn_gate_shexp.scale` /
`ffn_down_shexp.scale` stay those tensors. Do not invent
`output.input_scale` this slice.

## Shipped 2026-08-31 — official shared-expert `{1}` `ffn_gate_shexp.scale`

llama.cpp `build_ffn` applies `ggml_mul` after the shared-expert gate GEMM
(`TENSOR_NOT_REQUIRED`; missing is `1.0`) before SiLU. Decode loads
`blk.{i}.ffn_gate_shexp.scale` only when `ffn_gate_shexp.weight` is present
(qwen2moe / qwen3next / llama4). Writer-tiny
`tiny-qwen2moe-ffn-gate-shexp-s` is `tiny-qwen2moe` plus scale `2.25`.
Llama / qwen3moe files that contain `ffn_gate_shexp.scale` are ignored
(not loaded). Dense `ffn_gate.scale` and shared-expert
`ffn_down_shexp.scale` stay those tensors. Do not invent
`output.input_scale` this slice.

## Shipped 2026-08-31 — official shared-expert `{1}` `ffn_down_shexp.scale`

llama.cpp `build_ffn` applies `ggml_mul` after the shared-expert down GEMM
(`TENSOR_NOT_REQUIRED`; missing is `1.0`) before `ffn_gate_inp_shexp`
sigmoid. Decode loads `blk.{i}.ffn_down_shexp.scale` only when
`ffn_down_shexp.weight` is present (qwen2moe / qwen3next / llama4).
Writer-tiny `tiny-qwen2moe-ffn-down-shexp-s` is `tiny-qwen2moe` plus
scale `1.75`. Llama / qwen3moe files that contain `ffn_down_shexp.scale`
are ignored (not loaded). Dense `ffn_down.scale` and per-expert
`ffn_down_exps.scale` stay those tensors. Do not invent
`output.input_scale` this slice.

## Shipped 2026-08-31 — official fused `{1}` `attn_qkv.scale`

llama.cpp `build_qkv` applies `wqkv_s` after the fused QKV GEMM, before
QKV bias (`TENSOR_NOT_REQUIRED`; missing is `1.0`). Decode loads
`blk.{i}.attn_qkv.scale` only when fused `attn_qkv.weight` is present
(bloom). Writer-tiny `tiny-bloom-attn-qkv-s` is `tiny-bloom` plus scale
`3.0`. Llama files that contain `attn_qkv.scale` are ignored (not
loaded). Split `attn_q.scale` / `attn_k.scale` / `attn_v.scale` stay the
llama tensors. Do not invent `output.input_scale` this slice.

## Shipped 2026-08-31 — CUDA `cuStreamGetDevResource`

`gpu-sim` `stream_get_dev_resource` is `cuStreamGetDevResource`. A stream
bound to a green context returns that ctx's SM span (‰ of the chip). An
unbound stream returns a full chip. Query; legal during capture. Distinct
from `stream_get_green_ctx` (ctx id) and `device_get_dev_resource` (always
full chip). No second `--green-ctx`. `gpu-profile capture` is still
refused. Dual score still has no `$/M tokens`.

## Shipped 2026-08-31 — CUDA green-context sync (`cudaExecutionCtxSynchronize`)

`gpu-sim` `green_ctx_synchronize` is `cudaExecutionCtxSynchronize` for a
green context: CPU waits every bound stream (and that ctx's
`green_ctx_wait_event` ops). Other green contexts on the same GPU keep
running. Distinct from `synchronize_stream` (one stream) and
`synchronize_device` (whole GPU). Capture is refused when a bound stream
is capturing. No second `--green-ctx`. `gpu-profile capture` is still
refused. Dual score still has no `$/M tokens`.

## Shipped 2026-08-31 — CUDA green-context events (`cuGreenCtxRecordEvent`)

`gpu-sim` `green_ctx_record_event` / `green_ctx_wait_event` are
`cuGreenCtxRecordEvent` / `cuGreenCtxWaitEvent`. Record joins every
stream bound to that ctx (later work does not join). Wait holds later
submits on that ctx, including streams bound after the wait. Distinct
from `cudaEventRecord` / `cudaStreamWaitEvent` (one stream). Capture is
refused when a bound stream is capturing. The record is not inserted
into a bound stream's timeline. No second `--green-ctx`.
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-08-31 — official NVFP4 `{1}` `output.scale`

llama.cpp creates `output.scale` shape `{1}` only when `output.weight`
is `GGML_TYPE_NVFP4` (`TENSOR_NOT_REQUIRED`; missing is `1.0`). Decode
follows `build_lora_mm`: `ggml_mul` after the LM-head GEMM, before bias
/ tanh softcap. Writer-tiny `tiny-nvfp4-output-s` is `tiny-nvfp4` plus
scale `4.0`. F32 files that contain `output.scale` are ignored (not
loaded). Tied embeddings (absent `output.weight`) do not load the
scale. Do not invent `output.input_scale` this slice.

## Shipped 2026-08-31 — official dense `{1}` `ffn_up.scale`

Generic llama.cpp post-load `blk.{i}.ffn_up.scale` shape `{1}`
(`TENSOR_NOT_REQUIRED`; missing is `1.0`). Decode follows `build_ffn`:
`ggml_mul` after the up GEMM, before the gated product. Writer-tiny
`tiny-llama-ffn-up-s` is `tiny-llama` plus scale `3.5` (distinct from
dense `ffn_down.scale` `0.5` and `ffn_gate.scale` `1.25`). Phi2/Bloom stay
on `Phi2Ffn` (no up-gate split). Per-expert `ffn_up_exps.scale` stays the
gemma4 MoE tensor.

## Shipped 2026-08-31 — official dense `{1}` `ffn_gate.scale`

Generic llama.cpp post-load `blk.{i}.ffn_gate.scale` shape `{1}`
(`TENSOR_NOT_REQUIRED`; missing is `1.0`). Decode follows `build_ffn`:
`ggml_mul` after the gate GEMM, before SiLU / GELU. Writer-tiny
`tiny-llama-ffn-gate-s` is `tiny-llama` plus scale `1.25` (distinct from
dense `ffn_down.scale` `0.5`). SiLU is nonlinear, so the walk stays
observable even where `post_ffw_norm` would cancel a down scale. Phi2/Bloom
stay on `Phi2Ffn` (no gate). Per-expert `ffn_gate_exps.scale` stays the
gemma4 MoE tensor.

## Shipped 2026-08-31 — official dense `{1}` `attn_v.scale`

Generic llama.cpp post-load `blk.{i}.attn_v.scale` shape `{1}`
(`TENSOR_NOT_REQUIRED`; missing is `1.0`). Decode follows `build_lora_mm`:
`ggml_mul` after the V GEMM, before V RMSNorm. Writer-tiny
`tiny-llama-attn-v-s` is `tiny-llama` plus scale `2.5` (distinct from
dense `ffn_down.scale` `0.5`, `attn_q.scale` `2.0`, `attn_output.scale`
`1.5`, and `attn_k.scale` `0.75`). Gemma3/4 unweighted V RMSNorm would
cancel a positive scalar, so the observable fixture is llama, not a
second gemma4 family.

## Shipped 2026-08-31 — official dense `{1}` `attn_k.scale`

Generic llama.cpp post-load `blk.{i}.attn_k.scale` shape `{1}`
(`TENSOR_NOT_REQUIRED`; missing is `1.0`). Decode follows `build_lora_mm`:
`ggml_mul` after the K GEMM, before K-norm / omitted-`wv` copy. Writer-tiny
`tiny-llama-attn-k-s` is `tiny-llama` plus scale `0.75` (distinct from
dense `ffn_down.scale` `0.5`, `attn_q.scale` `2.0`, and `attn_output.scale`
`1.5`). Gemma3/4 K-norm would cancel a positive scalar, so the observable
fixture is llama, not a second gemma4 family.

## Shipped 2026-08-31 — official dense `{1}` `attn_output.scale`

Generic llama.cpp post-load `blk.{i}.attn_output.scale` shape `{1}`
(`TENSOR_NOT_REQUIRED`; missing is `1.0`). Decode follows `build_lora_mm`:
`ggml_mul` after the output GEMM, before `post_attention_norm`. Writer-tiny
`tiny-llama-attn-out-s` is `tiny-llama` plus scale `1.5` (distinct from
dense `ffn_down.scale` `0.5` and `attn_q.scale` `2.0`). Gemma2/3/4
`post_attention_norm` RMSNorm would cancel a positive scalar, so the
observable fixture is llama, not a second gemma4 family.

## Shipped 2026-08-31 — official dense `{1}` `attn_q.scale`

Generic llama.cpp post-load `blk.{i}.attn_q.scale` shape `{1}`
(`TENSOR_NOT_REQUIRED`; missing is `1.0`). Decode follows `build_lora_mm`:
`ggml_mul` after the Q GEMM, before QK-Norm. Writer-tiny
`tiny-llama-attn-q-s` is `tiny-llama` plus scale `2.0` (distinct from
dense `ffn_down.scale` `0.5`). Gemma3/4 QK-Norm would cancel a positive
scalar, so the observable fixture is llama, not a second gemma4 family.
Do not invent F32 `output.scale` unless NVFP4.

## Shipped 2026-08-31 — official dense `{1}` `ffn_down.scale`

Generic llama.cpp post-load `blk.{i}.ffn_down.scale` shape `{1}`
(`TENSOR_NOT_REQUIRED`; missing is `1.0`). Decode follows `build_ffn`:
`ggml_mul` after the down GEMM on every `DenseFfn` (llama / gemma / qwen
dense and gemma4 shared MLP). Writer-tiny `tiny-llama-ffn-down-s` is
`tiny-llama` plus scale `0.5`. Gemma2/3/4 `post_ffw_norm` RMSNorm would
cancel a positive scalar, so the observable fixture is llama, not a
second gemma4 family. Phi2/Bloom stay on `Phi2Ffn` (no load this item).
Per-expert `ffn_*_exps.scale` stay the gemma4 MoE tensors.

## Shipped 2026-08-31 — official gemma4 per-expert gate/up scale

`general.architecture=gemma4` with `blk.{i}.ffn_gate_exps.scale` and
`blk.{i}.ffn_up_exps.scale` (same arch as dense/MoE/PLE/fused/shared-KV/
mixed-hd/SWA-base/no-wv/out-scale/rope-freqs/down-s gemma4, not a second
family). Decode follows llama.cpp `build_lora_mm_id`: after the gate/up
expert GEMM, `ggml_mul` broadcasts `{n_expert}` onto that expert's output
(`TENSOR_NOT_REQUIRED`; missing is `1.0`). Fused `ffn_gate_up_exps`
applies `up_exps_s` to both halves and ignores `gate_exps_s`. Writer-tiny
`tiny-gemma4-moe-gate-up-s` is split `tiny-gemma4-moe` plus scales
`[1.5, 0.5, 3.0, 0.75]` / `[0.75, 3.0, 0.5, 1.5]`. Writer-tiny zeros
`ffn_gate_inp` so softmax is uniform and packs split experts as F32
(Q4_K GELU(gate) is ~0 on the tiny, and `ffn_post_norm_2` RMSNorm would
cancel a one-hot mix). Omits `ffn_down_exps.scale`.
Fused/PLE/shared-KV/mixed-hd omit the tensors.
`mixtral` stays rejected.

## Shipped 2026-08-31 — CUDA green contexts (`cuGreenCtxCreate`)

`gpu-sim` `device_get_dev_resource` / `dev_sm_resource_split_by_count` /
`dev_resource_generate_desc` / `green_ctx_create` / `green_ctx_stream_create` /
`green_ctx_set_stream` / `stream_get_green_ctx` are CUDA green contexts.
SM resources are ‰ of the chip, not occupancy SM counts.
`set_stream_sm_permille` stays duration-only (does not partition Hyper-Q).
Complementary spans may overlap kernels even when `compute_slots` is 1;
same-span contexts still share exclusive compute. Capture cannot include
create / desc / bind / destroy. Flags 0 only.
`expertvm sim` / `schedule` / `store` / Engine / serve `--green-ctx` binds
decode vs leftover prefill (implies `--decode-priority`; default 500/500;
`--decode-sms 1000` refused). Distinct from `--decode-sms` (duration scale,
exclusive occupancy still serializes leftover prefill). Off by default
keeps full-chip exclusive compute (decode identity).
`gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

## Shipped 2026-08-31 — official gemma4 per-expert `ffn_down_exps.scale`

`general.architecture=gemma4` with `blk.{i}.ffn_down_exps.scale` (same arch
as dense/MoE/PLE/fused/shared-KV/mixed-hd/SWA-base/no-wv/out-scale/rope-freqs
gemma4, not a second family). Decode follows llama.cpp `build_lora_mm_id`:
after the down expert GEMM, `ggml_mul` broadcasts `{n_expert}` onto that
expert's output (`TENSOR_NOT_REQUIRED`; missing is `1.0`). Writer-tiny
`tiny-gemma4-moe-down-s` is split `tiny-gemma4-moe` plus scales
`[0.5, 2.0, 0.25, 4.0]`. Writer-tiny zeros `ffn_gate_inp` so softmax is
uniform and packs split experts as F32 (Q4_K GELU(gate) is ~0 on the
tiny, and `ffn_post_norm_2` RMSNorm would cancel a one-hot mix).
Fused/PLE/shared-KV/mixed-hd omit the tensor.
`mixtral` stays rejected.

## Shipped 2026-08-31 — official gemma4 proportional `rope_freqs`

`general.architecture=gemma4` with `rope_freqs.weight` (same arch as
dense/MoE/PLE/fused/shared-KV/mixed-hd/SWA-base/no-wv/out-scale gemma4,
not a second family). Decode follows llama.cpp: non-SWA layers pass the
tensor into `ggml_rope_ext`; ggml uses `theta/ff`. SWA keeps
`freq_factors` null. Writer-tiny `tiny-gemma4-rope-freqs` is mixed-hd
plus convert packing (`1` then `1e30` for unrotated NeoX pairs).
Mixed-hd / SWA-base / no-wv / out-scale omit the tensor. `mixtral` stays
rejected.

## Shipped 2026-08-31 — official gemma4 layer output scale

`general.architecture=gemma4` with `blk.{i}.layer_output_scale` (same arch
as dense/MoE/PLE/fused/shared-KV/mixed-hd/SWA-base/no-wv gemma4, not a
second family). Decode follows llama.cpp: `out_scale` is
`TENSOR_NOT_REQUIRED`; when present, `ggml_mul` broadcasts onto the
residual after FFN (and PLE inject). Writer-tiny `tiny-gemma4-out-scale`
is dense `tiny-gemma4` plus scale `0.5`. Optional `rope_freqs` stay
omitted. `mixtral` stays rejected.

## Shipped 2026-08-31 — official gemma4 omitted `wv` (K as V)

`general.architecture=gemma4` without `attn_v` (same arch as
dense/MoE/PLE/fused/shared-KV/mixed-hd/SWA-base gemma4, not a second
family). Decode follows llama.cpp: `wv` is `TENSOR_NOT_REQUIRED`; when
missing, `Vcur = Kcur` before the unweighted V RMSNorm. Writer-tiny
`tiny-gemma4-no-wv` is dense `tiny-gemma4` without `attn_v`. Optional
`out_scale` / `rope_freqs` stay omitted. `mixtral` stays rejected.

## Shipped 2026-08-31 — official gemma4 SWA RoPE base

`general.architecture=gemma4` with `rope.freq_base_swa` (same arch as
dense/MoE/PLE/fused/shared-KV/mixed-hd gemma4, not a second family).
Decode follows llama.cpp `get_rope_freq_base(il)`: SWA uses
`rope.freq_base_swa` (default `10000` when omitted), global uses
`rope.freq_base`. Writer-tiny `tiny-gemma4-swa-base` is mixed-hd tensors
with E2B/E4B convert values (`1000000` vs `10000`). Mixed-hd global
QK-Norm is `0.1` so attn-scale `1.0` does not saturate softmax. Optional
`wv` / `out_scale` / `rope_freqs` stay omitted. `mixtral` stays rejected.

## Shipped 2026-08-31 — official gemma4 mixed SWA/global head dims

`general.architecture=gemma4` with `key_length_swa != key_length` (same
arch as dense/MoE/PLE/fused/shared-KV gemma4, not a second family).
Decode follows llama.cpp `n_embd_head_k(il)`: SWA uses
`attention.key_length_swa` / `rope.dimension_count_swa`, global uses
`attention.key_length`. KV cache stride is the max head dim. Writer-tiny
`tiny-gemma4-mixed-hd` is two dense layers, SWA then global,
`key_length_swa` half of `key_length`. 1-layer tinies stay equal-hd.
`k_swa != v_swa` still refused with named keys. Optional `wv` /
`out_scale` / `rope_freqs` / `rope.freq_base_swa` stay omitted.
`mixtral` stays rejected.

## Shipped 2026-08-31 — official gemma4 shared KV

`general.architecture=gemma4` with `attention.shared_kv_layers > 0` (same
arch as dense/MoE/PLE/fused gemma4, not a second family). Decode follows
llama.cpp `has_kv(il)` plus KV-cache reuse: shared layers skip K/V and
attend against the donor `n_layer_kv_from_start` minus 2 (SWA) or minus 1
(global). Writer-tiny `tiny-gemma4-shared-kv` is three dense all-SWA
layers with `shared_kv_layers=1` (layer 2 reuses layer 0). 1-layer tinies
stay full-KV. Mixed SWA/global head dims stay refused with named keys.
`mixtral` stays rejected.

## Shipped 2026-08-31 — official gemma4 fused plus PLE

`general.architecture=gemma4` with fused `ffn_gate_up_exps` and
`embedding_length_per_layer_input > 0` (production E2B/E4B packing, same
arch as fused-only and split MoE plus PLE, not a second family). Decode is
fused GELU experts then PLE inject after the FFN residual. Writer-tiny
`tiny-gemma4-moe-fused-ple` (`n_expert=4`, `n_expert_used=2`,
`n_embd_per_layer=64`). Split `tiny-gemma4-moe-ple` and fused-only
`tiny-gemma4-moe-fused` stay. Shared KV and mixed SWA/global head dims stay
refused with named keys. `mixtral` stays rejected.

## Shipped 2026-08-31 — official gemma4 fused `ffn_gate_up_exps`

`general.architecture=gemma4` with fused `ffn_gate_up_exps` instead of
separate `ffn_gate_exps` / `ffn_up_exps` (same arch as dense/MoE/PLE
gemma4, not a second family). Decode follows `src/models/gemma4.cpp`
`build_moe_ffn`: one GEMV of twice `n_ff` rows, then split gate then up,
then GELU. DirectStore catalogs fused bytes in `ExpertParts.gate` with
empty `up`. Writer-tiny `tiny-gemma4-moe-fused` (`n_expert=4`,
`n_expert_used=2`). Split `tiny-gemma4-moe`, PLE `tiny-gemma4-ple`, and
MoE plus PLE `tiny-gemma4-moe-ple` stay. Shared KV and mixed SWA/global
head dims stay refused with named keys. `mixtral` stays rejected.

## Shipped 2026-08-31 — official gemma4 MoE+PLE

`general.architecture=gemma4` with `ffn_gate_inp` and
`embedding_length_per_layer_input > 0` (production E2B/E4B shape, same arch
as dense/MoE/PLE-only gemma4, not a second family). Decode is shared GeGLU
plus custom-router GELU experts, then PLE inject after the FFN residual.
Writer-tiny `tiny-gemma4-moe-ple` (`n_expert=4`, `n_expert_used=2`,
`n_embd_per_layer=64`). Dense `tiny-gemma4`, MoE `tiny-gemma4-moe`, and PLE
`tiny-gemma4-ple` stay. Fused `ffn_gate_up_exps`, shared KV, and mixed
SWA/global head dims stay refused with named keys. `mixtral` stays rejected.

## Shipped 2026-08-31 — official gemma4 PLE

`general.architecture=gemma4` with `embedding_length_per_layer_input > 0`
(same arch as dense/MoE gemma4, not a second family). Decode follows
`src/models/gemma4.cpp` PLE: `build_inp_per_layer` plus
`project_per_layer_inputs`, then after the FFN residual
`gelu(mm(inp_gate)) * per-layer slice`, `mm(proj)`, RMSNorm `post_norm`,
residual add. No AltUp / Laurel. Writer-tiny `tiny-gemma4-ple`
(`n_embd_per_layer=64`). Dense `tiny-gemma4` and MoE `tiny-gemma4-moe`
keep `n_embd_per_layer=0`. Shared KV and mixed SWA/global head dims stay
refused with named keys. `mixtral` stays rejected.

## Shipped 2026-08-31 — official gemma4 MoE

`general.architecture=gemma4` with `ffn_gate_inp` (same arch as dense
gemma4, not a second family). Decode follows `src/models/gemma4.cpp` MoE:
shared dense GeGLU plus `ffn_post_norm_1`, expert `ffn_pre_norm_2`, custom
router on `attn_out` (unweighted RMSNorm, `1/sqrt(n_embd)`,
`ffn_gate_inp.scale`, then `ffn_gate_inp`), `build_moe_ffn` softmax then
top-k with `norm_w` clamp `2^-14` and GELU experts, then `ffn_post_norm_2`
and add. Writer-tiny `tiny-gemma4-moe` (`n_expert=4`, `n_expert_used=2`).
Dense `tiny-gemma4` is unchanged. Fused `ffn_gate_up_exps`, per-layer
embeddings, shared KV, and mixed SWA/global head dims stay refused with
named keys. `mixtral` stays rejected.

## Shipped 2026-08-31 — official gemma4

`general.architecture=gemma4` with convert-shaped `gemma4.*` KV. Decode
follows `src/models/gemma4.cpp`: Gemma embed-scale + GeGLU, QK-Norm before
RoPE, unweighted RMSNorm on V, attention scale `1.0`, RMSNorm
`post_attention_norm` / `ffn_norm` / `post_ffw_norm`, required
`attention.sliding_window`, convert `attention.sliding_window_pattern` as a
per-layer bool array, required `attention.key_length_swa` /
`value_length_swa` and `embedding_length_per_layer_input`. Writer-tiny is
dense (no MoE), `embedding_length_per_layer_input = 0`, no shared KV, equal
SWA/global head dims, `attention.sliding_window = 2`. Gemma4 MoE, per-layer
embeddings, shared KV, and mixed SWA/global head dims stay refused with
named keys. `mixtral` stays rejected.

## Shipped 2026-08-31 — `cuMemCreate` plus `cuMemMap` instead of combined `va_acquire`

`GpuStoreCfg::vmm_handle` / `SimCfg::vmm_handle` idle-or-reserve a VA
(`va_reserve_idle`) then `cuMemCreate` plus `cuMemMap` of that handle at
offset 0 so the page holds a live handle and `va_release_handle` before
`va_release` (`alloc_overhead_ns` on create and map). Hits/misses stay the
same. Distinct from `--vmm` (combined `va_map` without a handle) and
`--vmm-retain` (promote after acquire). Implies `--vmm`. Illegal with
`--mapped` / `--managed` / `--vmm-retain`. `--vmm-handle` is off by default
(decode identity: `--vmm` without create plus map). Store and walker (not
walker-only). infer-bench schedule gets `--vmm-handle` (implies `--vmm`; no
standalone `--vmm`). `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cuMemRetainAllocationHandle` after each VMM miss map

`GpuStoreCfg::vmm_retain` / `SimCfg::vmm_retain` call
`cuMemRetainAllocationHandle` at offset 0 after each VMM miss `va_acquire`
/ `va_acquire_paged` so the page holds a live handle and `va_release_handle`
before `va_release` (`alloc_overhead_ns` on every retain). Hits/misses stay
the same. Distinct from `--vmm` (combined `va_map` without a handle) and
`--vmm-page` (split maps; retain still one handle at offset 0). Implies
`--vmm`. Illegal with `--mapped` / `--managed`. `--vmm-retain` is off by
default (decode identity: `--vmm` without a handle handshake). Store and
walker (not walker-only). infer-bench schedule gets `--vmm-retain` (implies
`--vmm`; no standalone `--vmm`). `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaMemPoolExportPointer` of each miss `cudaMallocAsync`

`GpuStoreCfg::share_ptr` / `SimCfg::share_ptr` call `cudaMemPoolExportPointer`
then `cudaMemPoolImportPointer` after each pinned miss fill so the page
holds a live alias (no extra HBM) and `cudaFreeAsync` of the import before
the source (`alloc_overhead_ns` on first export and every import).
Hits/misses stay the same. Distinct from `--shareable` (POSIX-FD pool-level
IPC) and `--ipc` (`cudaIpcGetMemHandle` of `cudaMalloc`). Implies
`--shareable` (which implies `--mempool`). Illegal with `--ipc` /
`--mapped` / `--managed` / `--vmm` / `--sync-alloc`. `--share-ptr` is off
by default (decode identity: pool-level `--shareable` without a per-page
pointer handshake). Store and walker (not walker-only). infer-bench has no
shareable pool expert fill of this form, so it does not get `--share-ptr`.
`gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaIpcGetMemHandle` of each miss `cudaMalloc`

`GpuStoreCfg::ipc` / `SimCfg::ipc` call `cudaIpcGetMemHandle` then
`cudaIpcOpenMemHandle` after each pinned miss fill so the page holds a live
alias (no extra HBM) and `cudaIpcCloseMemHandle` before `cudaFree`
(`alloc_overhead_ns` on first get and every open). Hits/misses stay the
same. Distinct from `--shareable` (POSIX-FD mempool IPC). Implies
`--sync-alloc`. Illegal with `--mapped` / `--managed` / `--vmm` /
`--shareable`. `--ipc` is off by default (decode identity: GEMM on the
`cudaMalloc` pointer with no IPC handshake). Store and walker (not
walker-only). infer-bench has no `cudaMalloc` expert fill of this form, so
it does not get `--ipc`. `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaHostUnregister` after miss DMA

`GpuStoreCfg::host_unregister` / `SimCfg::host_unregister` call
`cudaHostUnregister` on pageable staging after each miss DMA so the next
miss re-registers (`synchronize` plus `first_alloc_ns`; pin refunded
between misses). Hits/misses stay the same. Distinct from `--host-register`
(keep registered for lifetime). Implies `--host-register` (which implies
`--pageable`). Illegal with `--mapped` / `--managed` /
`--host-register-mapped` / `--d2h-pageable`. `--host-unregister` is off by
default (decode identity: keep staging registered). Store and walker (not
walker-only). infer-bench has no pageable staging of this form, so it does
not get `--host-unregister`. `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — Device→Host pageable memcpy on pinned/VMM LRU evict

`GpuStoreCfg::d2h_pageable` / `SimCfg::d2h_pageable` use `cudaMemcpyAsync`
Device→Host (pageable bounce-buffer) on the copy stream before
`cudaFreeAsync` / `va_release` / `free_sync` so evict pays extra
host-synchronous PCIe. Hits/misses stay the same; the next miss still fills
from catalog staging. Distinct from `--d2h-evict` (Device→HostPinned
overlapping DMA) and `--prefetch-host` (managed keep-alloc). Implies
`--pageable`. Illegal with `--mapped` / `--managed` / `--host-register` /
`--d2h-evict`. `--d2h-pageable` is off by default (decode identity: free
with no D2H). Store and walker (not walker-only). infer-bench has no
pinned/VMM expert evict of this form, so it does not get `--d2h-pageable`.
`gpu-profile capture` is still refused.

## Shipped 2026-08-31 — Device→HostPinned memcpy on pinned/VMM LRU evict

`GpuStoreCfg::d2h_evict` / `SimCfg::d2h_evict` use `cudaMemcpyAsync`
Device→HostPinned on the copy stream before `cudaFreeAsync` / `va_release`
/ `free_sync` so evict pays extra PCIe. Hits/misses stay the same; the next
miss still fills from catalog staging. Distinct from `--prefetch-host`
(managed keep-alloc). Illegal with `--mapped` / `--managed`. `--d2h-evict`
is off by default (decode identity: free with no D2H). Store and walker
(not walker-only). infer-bench has no pinned/VMM expert evict of this form,
so it does not get `--d2h-evict`. `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaMemcpyWithAttributesAsync` demand miss fill

`GpuStoreCfg::memcpy_attr` / `SimCfg::memcpy_attr` use
`cudaMemcpyWithAttributesAsync` DuringApiCall for demand pinned/VMM miss H2D
so the API waits that copy instead of leaving `memcpy_pinned_to_device` in
flight. Hits/misses stay the same. Does not imply `--memcpy-batch`. Distinct
from `--memcpy-during` (batch prefetch DuringApiCall) and `--memcpy-any`
(empty deps, no API wait). Keep replica `pin_hot` D2D as memcpy. Illegal
with `--mapped` / `--managed` / `--pageable` / `--memset-fill` /
`--sync-alloc` / `--sync-memops` / `--device-sync-memops`. `--memcpy-attr`
is off by default (decode identity: sequential pinned H2D). Store and walker
(not walker-only). infer-bench has no expert H2D fill, so it does not get
`--memcpy-attr`. `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — skip fill `cudaMemPrefetchAsync`

`GpuStoreCfg::no_mem_prefetch` / `SimCfg::no_mem_prefetch` skip
`cudaMemPrefetchAsync` at managed miss fill so the first GEMM first-touches
on compute instead of copy-engine prefetch overlapping leftover GEMM.
Implies `--managed`. Hits/misses stay the same. Distinct from `--prefetch
none` (predictor) and `--prefetch-host` (evict to host). Keep replica dest
prefetch and host-restore prefetch. Keep SetReadMostly /
SetPreferredLocation unless those flags are also set. `--no-mem-prefetch`
is off by default (decode identity: fill prefetch). Store and walker (not
walker-only). infer-bench has managed fill via `--managed-host`, so it gets
`--no-mem-prefetch`. `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — skip `cudaMemAdviseSetPreferredLocation`

`GpuStoreCfg::no_preferred` / `SimCfg::no_preferred` call
`UnsetPreferredLocation` at managed fill instead of SetPreferredLocation so
a remote GEMM first-touches (weight migrate onto the compute GPU) instead
of staying on home. Implies `--managed`. Hits/misses stay the same.
Distinct from `--accessed-by` (dest GEMM without migrating) and
`--no-read-mostly` (prefetch move vs replicate). Keep SetReadMostly unless
`--no-read-mostly` is also set. `--no-preferred` is off by default (decode
identity: SetPreferredLocation). Store and walker (not walker-only).
infer-bench has managed fill via `--managed-host` and `--place remote`, so
it gets `--no-preferred`. Do not invent `SetPreferredLocationHost` as a
decode-path skip of kernel first-touch. `gpu-profile capture` is still
refused.

## Shipped 2026-08-31 — skip `cudaMemAdviseSetReadMostly`

`GpuStoreCfg::no_read_mostly` / `SimCfg::no_read_mostly` call
`UnsetReadMostly` at managed fill instead of SetReadMostly so dest prefetch
**moves** (one GPU copy; lower `hbm_peak`) instead of replicating. Implies
`--managed`. Hits/misses stay the same. Distinct from `--accessed-by` (dest
GEMM without a second copy). Keep SetPreferredLocation. `--no-read-mostly`
is off by default (decode identity: SetReadMostly). Store and walker (not
walker-only). infer-bench has managed fill via `--managed-host`, so it gets
`--no-read-mostly`. Dest replica drop is a no-op when the page is not
ReadMostly. `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaGraphAddIf` combo reuse

`GpuStoreCfg::graph_if` / `SimCfg::graph_if` wrap walker combo children in
`graph_add_if` + `graph_add_set_conditional` (handle default 0; node value 1)
so a later subset retargets extras with `graph_exec_set_conditional_params`
instead of instantiating a new parent. Pays `graph_set_params_ns` and clears
upload (next launch re-uploads). Needs `--graph-build`. Does not imply
graph-build. Illegal with `--device-launch` and `--graph-enable`. Hits/misses
stay the same. Distinct from `--graph-enable` (SetEnabled does not clear
upload). `--graph-if` is off by default (decode identity: no IF nodes). Store
GEMM stays per-leaf. infer-bench has no combo graph construction, so it does
not get `--graph-if`. `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — graph-build `cudaGraphSetConditional` + exec SetParams

`Sim::graph_add_set_conditional` is the graph-build analog of captured
`cudaGraphSetConditional`. Handle is topology; `value` is
`graph_exec_set_conditional_params` / `GraphNodeParams::SetConditional`.
Device-launch instantiate refuses the node. Decode identity does not add
set-conditional nodes. `--graph-if` is the Engine twin (graph-build combo
IF wrap + exec SetParams).

## Shipped 2026-08-31 — `cudaLaunchHostFunc` after miss DMA

`GpuStoreCfg::copy_host` / `SimCfg::copy_host` enqueue `cudaLaunchHostFunc`
on the DMA stream after pinned/VMM H2D, `cudaMemsetAsync` miss fill, or
managed prefetch, before copy-ready. Bills `host_func_ns` so GEMM waits
include the host callback. Does not imply `--host-func`. Hits/misses stay
the same. Distinct from `--host-func` (after every token GEMM, including
hits). Mapped misses have no device fill (no-op). Replica `pin_hot` D2D
stays memcpy-only. `--copy-host` is off by default (decode identity: no
host after copy). Store and walker (not walker-only). infer-bench has no
expert miss DMA, so it does not get `--copy-host`. `gpu-profile capture`
is still refused.

## Shipped 2026-08-31 — `cudaMemsetAsync` miss fill

`GpuStoreCfg::memset_fill` / `SimCfg::memset_fill` fill pinned/VMM miss
pages with `cudaMemsetAsync` (HBM write, compute occupancy) instead of
pinned H2D (copy-engine PCIe). Does not imply `--cuda-graphs`. Hits/misses
stay the same. Distinct from `--graph-memset` (in-graph scratch vs miss
page). Illegal with `--mapped` / `--managed` / `--pageable` /
`--memcpy-batch`. Replica `pin_hot` D2D stays memcpy. `--memset-fill` is
off by default (decode identity: pinned H2D; inner ExpertStore still holds
real weights). Store and walker (not walker-only). infer-bench has no
expert H2D fill, so it does not get `--memset-fill`. `gpu-profile capture`
is still refused.

## Shipped 2026-08-31 — `cudaGraphClone` of combo parents

`GpuStoreCfg::graph_clone_parent` / `SimCfg::graph_clone_parent` clone walker
combo parents (`cudaGraphClone`, recursive children) before instantiate so
the parent tree is independent of GraphBank leaves. Does not imply
`--graph-clone` or `--cuda-graphs`. Hits/misses stay the same. Distinct from
`--graph-clone` (leaves vs combo parents). `--graph-clone-parent` is off by
default (decode identity: instantiate-in-place). Store GEMM stays per-leaf.
infer-bench has no combo graph construction, so it does not get
`--graph-clone-parent`. `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaGraphAddHostNode` before each leaf GEMM

`GpuStoreCfg::graph_leaf_host` / `SimCfg::graph_leaf_host` insert
`cudaGraphAddHostNode` / captured `cudaLaunchHostFunc` BEFORE the leaf GEMM
kernel so each launch bills `host_func_ns` (`host → kernel`). Implies
`--cuda-graphs`. Does not imply `--host-func`, `--graph-host`, or
`--graph-capture-host`. Hits/misses stay the same. Distinct from live
`--host-func` (after the token GEMM), `--graph-host` (BETWEEN graph-build combo
children), and `--graph-capture-host` (BETWEEN piecewise fragments). Illegal
with `--device-launch`. `--graph-leaf-host` is off by default (decode identity:
kernel-only graphs). Store leaf GEMMs host too (not walker-only). infer-bench
has no CUDA-graph leaf construction, so it does not get `--graph-leaf-host`.
`gpu-profile capture` is still refused.

## Shipped 2026-08-31 — captured `cudaLaunchHostFunc` between piecewise fragments

`GpuStoreCfg::graph_capture_host` / `SimCfg::graph_capture_host` insert
captured `cudaLaunchHostFunc` BETWEEN `--graph-piecewise` combo fragments so
sibling expert GEMMs serialize through `host_func_ns` (`child → host → child`).
Needs `--graph-piecewise`. Does not imply piecewise or `--host-func`.
Hits/misses stay the same. Distinct from `--graph-piecewise` (overlap vs
serial+tax), `--graph-capture-deps` (serialize without host tax), and
`--graph-host` (`cudaGraphAddHostNode` on graph-build). `--graph-capture-host`
is off by default (decode identity: no host nodes in combo parents). Store
GEMM stays per-leaf. infer-bench has no `--graph-piecewise`, so it does not
get `--graph-capture-host`. `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaGraphAddMemcpyNode` H2D of graph-mem scratch

`GpuStoreCfg::graph_memcpy` / `SimCfg::graph_memcpy` insert
`cudaGraphAddMemcpyNode` / `cudaMemcpyAsync` H2D BETWEEN `--graph-mem` scratch
alloc and the GEMM kernel so launch bills copy-engine PCIe. Needs
`--graph-mem`. Does not imply graph-mem or `--graph-memset`. Hits/misses stay
the same. Distinct from `--graph-mem` (scratch alloc+free vs extra H2D of that
scratch) and `--graph-memset` (compute HBM write vs copy-engine). Does not
work with `--graph-auto-free` alone. `--graph-memcpy` is off by default
(decode identity: kernel-only graphs). Store leaf GEMMs memcpy too (not
walker-only). infer-bench has no `--graph-mem`, so it does not get
`--graph-memcpy`. `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaGraphAddMemsetNode` of graph-mem scratch

`GpuStoreCfg::graph_memset` / `SimCfg::graph_memset` insert
`cudaGraphAddMemsetNode` / `cudaMemsetAsync` BETWEEN `--graph-mem` scratch
alloc and the GEMM kernel so launch bills an extra HBM write. Needs
`--graph-mem`. Does not imply graph-mem. Hits/misses stay the same.
Distinct from `--graph-mem` (scratch alloc+free vs extra memset of that
scratch). Does not work with `--graph-auto-free` alone. `--graph-memset`
is off by default (decode identity: kernel-only graphs). Store leaf GEMMs
memset too (not walker-only). infer-bench has no `--graph-mem`, so it does
not get `--graph-memset`. `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaGraphAddHostNode` between combo children

`GpuStoreCfg::graph_host` / `SimCfg::graph_host` insert
`cudaGraphAddHostNode` BETWEEN `--graph-build` combo children so sibling
expert GEMMs serialize through `host_func_ns` (`child → host → child`).
Needs `--graph-build`. Does not imply graph-build or `--host-func`.
Hits/misses stay the same. Distinct from `--graph-build` (overlap vs
serial+tax), `--graph-build-deps` (serialize without host tax), and live
`--host-func` (after the token's GEMMs). `--graph-host` is off by default
(decode identity: no host nodes in combo parents). Store GEMM stays
per-leaf. infer-bench has no `--graph-build`, so it does not get
`--graph-host`. `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaGraphAddDependencies` combo edges

`GpuStoreCfg::graph_build_deps` / `SimCfg::graph_build_deps` call
`cudaGraphAddDependencies` on `--graph-build` combo parents so later
`graph_add_child` nodes chain to the previous child and sibling expert
GEMMs serialize. Needs `--graph-build`. Does not imply graph-build.
Hits/misses stay the same. Distinct from `--graph-build` (no edges vs
chained) and `--graph-capture-deps` (`cudaStreamBeginCaptureToGraph`
deps). `--graph-build-deps` is off by default (decode identity: no
edges). Store GEMM stays per-leaf. infer-bench has no `--graph-build`,
so it does not get `--graph-build-deps`. `gpu-profile capture` is still
refused.

## Shipped 2026-08-31 — `cudaStreamBeginCaptureToGraph` deps

`GpuStoreCfg::graph_capture_deps` / `SimCfg::graph_capture_deps` pass a
non-empty `cudaStreamBeginCaptureToGraph` dependency array on
`--graph-piecewise` combo parents so later fragments chain to the previous
last node and sibling expert GEMMs serialize. Needs `--graph-piecewise`.
Does not imply piecewise. Hits/misses stay the same. Distinct from
`--graph-piecewise` (empty deps / extra roots vs chained deps) and
`--graph-build` (no `graph_add_dependencies` edges). `--graph-capture-deps`
is off by default (decode identity: empty deps). Store GEMM stays
per-leaf. infer-bench has no `--graph-piecewise`, so it does not get
`--graph-capture-deps`. `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaAccessPropertyStreaming` persist-window hits

`GpuStoreCfg::l2_streaming` / `SimCfg::l2_streaming` set
`cudaAccessPropertyStreaming` for `--l2-persist` GEMM window hits so a reused
expert bills full HBM (no persist fill). Needs persist (`--l2-persist` /
reset / fetch / ratio). Does not imply persist. Hits/misses stay the same.
Distinct from `--l2-ratio` (ratio of Persisting window vs hit property) and
`--l2-reset` (fill then clear). `--l2-streaming` is off by default (decode
identity: Persisting hits). infer-bench has `--l2-persist`, so it gets
`--l2-streaming`. `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaMemcpySrcAccessOrderAny`

`GpuStoreCfg::memcpy_any` / `SimCfg::memcpy_any` put
`cudaMemcpySrcAccessOrderAny` on `--memcpy-batch` H2D so intra-batch copies
have empty deps and the API does not wait (copies stay in flight). Needs
`--memcpy-batch`. Exclusive with `--memcpy-during`. Hits/misses stay the
same. Distinct from `--memcpy-during` (wait vs no wait) and Stream order
(empty deps vs stream-order snapshot). `--memcpy-any` is off by default
(decode identity: Stream order, or sequential H2D). infer-bench has no
`--memcpy-batch`, so it does not get `--memcpy-any`. `gpu-profile capture`
is still refused.

## Shipped 2026-08-31 — `cudaMemcpySrcAccessOrderDuringApiCall`

`GpuStoreCfg::memcpy_during` / `SimCfg::memcpy_during` put
`cudaMemcpySrcAccessOrderDuringApiCall` on `--memcpy-batch` H2D so the batch
API waits those copies before return (not the whole stream). Needs
`--memcpy-batch`. Hits/misses stay the same. Distinct from `--memcpy-batch`
(batch vs sequential H2D vs During wait). `--memcpy-during` is off by default
(decode identity: Stream order, or sequential H2D). infer-bench has no
`--memcpy-batch`, so it does not get `--memcpy-during`. `gpu-profile capture`
is still refused.

## Shipped 2026-08-31 — `cudaMemPoolProps::maxSize`

`GpuStoreCfg::mempool_max` / `SimCfg::mempool_max` `cudaMemPoolCreate` with
`cudaMemPoolProps::maxSize` then `cudaDeviceSetMemPool` so `cudaMallocAsync`
draws from the capped pool. Implies `--mempool` (`u64::MAX` hold). Illegal
with `--sync-alloc`. Hits/misses stay the same when `N` fits. Reserved
`live+cached` cannot grow past `N` (leftover cache plus
`--mempool-no-reuse` OS alloc OOMs). Distinct from `--mempool` (hold),
`--mempool-trim` (idle `TrimTo(0)`), and `--mempool-no-reuse` (skip
opportunistic reuse). `--mempool-max` is unset (`0`, unlimited) by default
(decode identity: device default pool). infer-bench has no `--mempool`, so
it does not get `--mempool-max`. `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaLaunchAttributeMemSyncDomainMap` launch collapse

`GpuStoreCfg::mem_sync_launch_map` / `SimCfg::mem_sync_launch_map` put
`cudaLaunchAttributeMemSyncDomainMap` collapse `{default: 0, remote: 0}` on
grouped GEMMs. Needs `--mem-sync-domain remote`. Overrides prefill
inherit-identity so leftover prefill maps Default→0 with decode Remote→0 and
`same_domain_fence_permille` returns (Hopper identity tax is 0; SetAttribute
does not tick the clock). Hits/misses stay the same. Distinct from
`--mem-sync-launch` (physical mapping vs logical domain) and `--mem-sync-map`
(launch vs stream). `--mem-sync-launch-map` is off by default (decode
identity: inherit-stream). infer-bench has no `--mem-sync-domain`, so it
does not get `--mem-sync-launch-map`. `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaLaunchAttributeMemSyncDomain` launch Remote

`GpuStoreCfg::mem_sync_launch` / `SimCfg::mem_sync_launch` put
`cudaLaunchAttributeMemSyncDomain` Remote on grouped GEMMs. Needs
`--mem-sync-domain remote`. Overrides prefill inherit-Default so leftover
prefill shares the decode Remote domain and `same_domain_fence_permille`
returns (Hopper identity tax is 0; SetAttribute does not tick the clock).
Hits/misses stay the same. Distinct from `--mem-sync-map` (logical launch
domain vs physical mapping). `--mem-sync-launch` is off by default (decode
identity: inherit-stream). infer-bench has no `--mem-sync-domain`, so it
does not get `--mem-sync-launch`. `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaAccessPolicyWindow.hitRatio`

`GpuStoreCfg::l2_ratio` / `SimCfg::l2_ratio` set launch `hitRatio` as ‰.
Implies `--l2-persist`. `1..=1000` (unset is 1000). A partial ratio bills more
HBM than full persist on a reused expert (no extra construction tick).
Hits/misses stay the same. `--l2-ratio` is unset by default (decode identity:
1000). infer-bench has `--l2-persist`, so it gets `--l2-ratio`.
`gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaLaunchAttributeClusterSchedulingPolicyPreference` LoadBalancing

`GpuStoreCfg::cluster_load_balance` / `SimCfg::cluster_load_balance` set
launch LoadBalancing. Needs `--func-cluster-spread`. Overrides function
Spread so leftover kernels can Hyper-Q overlap again. Exclusive with
`--cluster-spread`. Hits/misses stay the same. `--cluster-load-balance` is
off by default (decode identity: Default policy). `gpu-profile capture` is
still refused.

## Shipped 2026-08-31 — `cudaLaunchAttributePreferredSharedMemoryCarveout` MaxL1

`GpuStoreCfg::max_l1` / `SimCfg::max_l1` set launch MaxL1 carveout. Needs
`--func-max-shared`. Overrides function MaxShared so leftover kernels can
Hyper-Q overlap again. Exclusive with `--max-shared`. Hits/misses stay the
same. `--max-l1` is off by default (decode identity: Default carveout).
`gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaLaunchAttributeMemSyncDomainMap` collapse

`GpuStoreCfg::mem_sync_collapse` / `SimCfg::mem_sync_collapse` call
`cudaStreamSetAttribute(..., cudaLaunchAttributeMemSyncDomainMap)` on the
decode stream (`{default: 0, remote: 0}`). Needs `--mem-sync-domain
remote`. Restores leftover prefill `same_domain_fence_permille` (Hopper
identity is remote→1; SetAttribute does not tick the clock). Hits/misses
stay the same. Distinct from `--mem-sync-domain` (logical domain vs
physical mapping). `--mem-sync-map` is identity by default (decode
identity: Hopper remote→1). infer-bench has no `--mem-sync-domain`, so it
does not get `--mem-sync-map`. `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaFuncAttributeRequiredClusterWidth`

`GpuStoreCfg::required_cluster` / `SimCfg::required_cluster` call
`cudaFuncSetAttribute(..., cudaFuncAttributeRequiredClusterWidth)` after
`Sim::new`. Needs `--cluster` and must equal `--cluster` X. Occupancy matches
`--cluster` (SetAttribute is +1 ns). Hits/misses stay the same. Distinct from
`--cluster-must-set` (bool: some cluster dim must be present) and
`--preferred-cluster`. A mismatched width is refused at construction
(`required-cluster must match cluster`), not at the first Invalid
`"required cluster"`. `--required-cluster` is unset by default (decode
identity: `0`). `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaSetDeviceFlags` schedule

`GpuStoreCfg::device_sync_policy` / `SimCfg::device_sync_policy` call
`cudaSetDeviceFlags` SCHEDULE_SPIN / YIELD / BLOCKING_SYNC after `Sim::new`.
Auto streams inherit that host-wait tax. Explicit `--sync-policy` wins.
ORs with `--device-sync-memops` (`SYNC_MEMOPS` stays). Hits/misses stay
the same. `--device-sync-policy` is Auto by default (decode identity: skip
`set_device_flags`). `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaLimitMaxL2FetchGranularity`

`GpuStoreCfg::l2_fetch` / `SimCfg::l2_fetch` call
`cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity)` after `Sim::new`.
Implies `--l2-persist`. `32` / `64` / `128` only. Access-policy windows must
align (64-byte persist is Invalid until `--l2-fetch 32`). Hits/misses stay
the same for 4096-byte experts. `--l2-fetch` is unset by default (decode
identity: 128). `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaFuncAttributeClusterDimMustBeSet`

`GpuStoreCfg::cluster_must_set` / `SimCfg::cluster_must_set` call
`cudaFuncSetAttribute(..., cudaFuncAttributeClusterDimMustBeSet)` after
`Sim::new`. Needs `--cluster`. Occupancy matches `--cluster` (SetAttribute
is +1 ns). Hits/misses stay the same. A grouped GEMM without `--cluster`
is refused at construction (`cluster-must-set needs cluster`), not at the
first Invalid `"cluster dim must be set"`. `--cluster-must-set` is off by
default (decode identity: unset). `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaEventBlockingSync` copy events

`GpuStoreCfg::event_blocking_sync` creates copy start/end events with
`cudaEventBlockingSync`. Implies `--timing-events`. `synchronize_event` pays
`host_sync_blocking_ns` instead of the recording stream's sync policy.
Hits/misses stay the same. Distinct from `--sync-policy blocking` (that taxes
`synchronize_stream`). `--event-blocking-sync` is off by default (decode
identity: `cudaEventDisableTiming`). `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaDeviceSetSharedMemConfig` device bank width

`GpuStoreCfg::device_shared_mem` / `SimCfg::device_shared_mem` call
`cudaDeviceSetSharedMemConfig` after `Sim::new`. Launch Default inherits that
bank-width duration scale when function config is also Default. Hits/misses
stay the same. Distinct from `--func-shared-mem` and launch-attribute
`--shared-mem`. `--device-shared-mem` is Default by default (decode identity:
unscaled). `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaFuncSetAttribute` cluster Spread

`GpuStoreCfg::func_cluster_spread` / `SimCfg::func_cluster_spread` call
`cudaFuncSetAttribute(..., cudaFuncAttributeClusterSchedulingPolicyPreference)`
Spread after `Sim::new`. Launch Default inherits that occupancy so leftover
kernels cannot Hyper-Q overlap when `--cluster` is at least 2. Hits/misses
stay the same. Distinct from launch-attribute `--cluster-spread`.
`--func-cluster-spread` is off by default (decode identity: Default function
policy). `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaFuncSetSharedMemConfig` function bank width

`GpuStoreCfg::func_shared_mem` / `SimCfg::func_shared_mem` call
`cudaFuncSetSharedMemConfig` after `Sim::new`. Launch Default inherits that
bank-width duration scale. Hits/misses stay the same. Distinct from
launch-attribute `--shared-mem`. `--func-shared-mem` is Default by default
(decode identity: unscaled). `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaCtxResetPersistingL2Cache` after each GEMM

`GpuStoreCfg::l2_reset` / `SimCfg::l2_reset` call `cudaCtxResetPersistingL2Cache`
after each live grouped GEMM so a reused expert does not keep persisting L2
lines. Implies `--l2-persist`. Hits/misses stay the same. Live (cannot
capture); kernel-only graphs stay legal. `--l2-reset` is off by default
(decode identity: no reset). `gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaFuncSetAttribute` MaxShared carveout

`GpuStoreCfg::func_max_shared` / `SimCfg::func_max_shared` call
`cudaFuncSetAttribute(..., cudaFuncAttributePreferredSharedMemoryCarveout)`
MaxShared after `Sim::new`. Launch Default inherits that occupancy so leftover
kernels cannot Hyper-Q overlap. Hits/misses stay the same. Distinct from
launch-attribute `--max-shared`. `--func-max-shared` is off by default
(decode identity: Default function carveout). `gpu-profile capture` is still
refused.

## Shipped 2026-08-31 — `cudaSetDeviceFlags` SyncMemops

`GpuStoreCfg::device_sync_memops` / `SimCfg::device_sync_memops` call
`cudaSetDeviceFlags(cudaDeviceSyncMemops)` after `Sim::new` so every
runtime memcpy/memset on that GPU is host-synchronous (including unmarked
pointers). Hits/misses stay the same. Distinct from per-page `--sync-memops`.
Illegal with `--mapped` / `--memcpy-batch`. `--device-sync-memops` is off by
default (decode identity: async `cudaMemcpyAsync` pinned H2D).
`gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cuPointerSetAttribute` SyncMemops on miss pages

`GpuStoreCfg::sync_memops` / `SimCfg::sync_memops` call
`cuPointerSetAttribute(CU_POINTER_ATTRIBUTE_SYNC_MEMOPS)` after device
alloc so miss H2D and managed prefetch are host-synchronous. Hits/misses
stay the same. Illegal with `--mapped` / `--memcpy-batch`. `--sync-memops`
is off by default (decode identity: async `cudaMemcpyAsync` pinned H2D).
`gpu-profile capture` is still refused.

## Shipped 2026-08-31 — `cudaHostRegisterMapped` expert pages

`GpuStoreCfg::host_register_mapped` / `SimCfg::host_register_mapped` allocate
pageable host (`cudaMalloc`) then `cudaHostRegisterMapped` so miss pages are
pin-and-map zero-copy (PCIe GEMM, `hbm_peak` 0) instead of `cudaHostAllocMapped`.
Implies `--mapped`. Illegal with `--host-register`. Hits/misses stay the same.
Evict is `cudaHostUnregister` then `free_host`. `--host-register-mapped` is off
by default (decode identity: `cudaHostAllocMapped`). `gpu-profile capture` is
still refused.

## Shipped 2026-08-31 — `cudaMemPrefetchAsync` to host on managed evict

`GpuStoreCfg::prefetch_host` / `SimCfg::prefetch_host` call
`cudaMemPrefetchAsync(..., cudaCpuDeviceId)` on managed LRU evict so the
`cudaMallocManaged` allocation stays live on the host. The next miss
prefetches the same pointer back (no second alloc). Implies `--managed`.
Hits/misses stay the same. `--prefetch-host` is off by default (decode
identity: `cudaFree` / `free_sync` on evict). `gpu-profile capture` is
still refused.

## Shipped 2026-08-31 — `cudaHostRegister` pageable staging then pinned DMA

`GpuStoreCfg::host_register` / `SimCfg::host_register` allocate pageable
staging (`cudaMalloc`) then `cudaHostRegister` so miss H2D is pinned DMA
instead of host-sync pageable copies. Implies `--pageable`. Illegal with
`--mapped` / `--managed`. Hits/misses stay the same. `--host-register` is
off by default (decode identity: `cudaMallocHost` staging). `gpu-profile
capture` is still refused.

## Shipped 2026-08-30 — `cudaMallocManaged(..., cudaMemAttachHost)` then Global attach

`GpuStoreCfg::managed_host` / `SimCfg::managed_host` allocate managed experts
with `cudaMemAttachHost`, then `cudaStreamAttachMemAsync` Global on the copy
stream so device prefetch is legal. Identity managed is Global at alloc (no
Attach op). Implies `--managed`. Prefetch stays on the copy stream and still
overlaps leftover compute unless `--stream-attach`. `--managed-host` is off
by default (decode identity: Global alloc + copy-stream prefetch).
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaStreamAttachMemAsync` Single on managed experts

`GpuStoreCfg::stream_attach` / `SimCfg::stream_attach` call
`cudaStreamAttachMemAsync(..., cudaMemAttachSingle)` on the compute
stream after managed alloc+advise, then prefetch on that stream so GEMM
stays legal under Single. Identity managed prefetch stays on the copy
stream (overlaps leftover compute). Implies `--managed`. Illegal with
`--seq-streams`. `--stream-attach` is off by default (decode identity:
Global attach + copy-stream prefetch). `gpu-profile capture` is still
refused.

## Shipped 2026-08-30 — `cudaLaunchAttributeProgrammaticEvent` replica overlap

`GpuStoreCfg::programmatic_event` / `SimCfg::programmatic_event` record
`cudaLaunchAttributeProgrammaticEvent` on grouped expert GEMMs so other
streams may `wait_event` at the PDL trigger (`pdl_trigger_permille`)
instead of kernel completion. Store `pin_hot` replica D2D on `n_gpus >= 2`
waits that event on the copy stream instead of draining the GEMM, so
leftover compute overlaps the replica. Implies a PDL trigger on those
GEMMs (same-stream PDL wait stays `--pdl`). Illegal with `--device-launch`.
`--programmatic-event` is off by default (decode identity).
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaMemPoolReuseAllowOpportunistic=0`

`GpuStoreCfg::mempool_no_reuse` / `SimCfg::mempool_no_reuse` set
`cudaMemPoolReuseAllowOpportunistic=0` on every GPU's current device
mempool after the `u64::MAX` hold. Leftover cache stays reserved; the next
miss is an OS alloc and charges extra HBM. Implies `--mempool`. Illegal
with `--sync-alloc`. Hits/misses stay the same. `--mempool-no-reuse` is
off by default (decode identity: CUDA default opportunistic reuse).
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaMemPoolTrimTo` after score

`GpuStoreCfg::mempool_trim` / `SimCfg::mempool_trim` hold unused
`cudaMallocAsync` bytes (`u64::MAX` release threshold) then
`cudaMemPoolTrimTo(0)` on every GPU's current device mempool after
`score()` / walker / schedule finish. Implies `--mempool`. Illegal with
`--sync-alloc`. Hits/misses and `hbm_peak` stay the same. Token ITL
(`clock_ns`) does not trim. `--mempool-trim` is off by default (decode
identity: CUDA default threshold 0 already returns on free).
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cuStreamWaitValue64` copy-ready

`GpuStoreCfg::wait_value` / `SimCfg::wait_value` allocate an 8-byte device
mailbox per expert page (`cudaMallocAsync` on the copy stream, waited
before H2D so compute `wait_value64` is resident during DMA) and
`cuStreamWriteValue64` after H2D so compute `cuStreamWaitValue64` Eq waits
that mailbox instead of a copy event. Replica D2D waits the mailbox on the
copy stream. Walker wait/write are live stream ops (GEMM graphs stay
kernel-only). `--wait-value` is off by default (decode identity).
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaLaunchAttributeLaunchCompletionEvent` replica overlap

`GpuStoreCfg::launch_completion` / `SimCfg::launch_completion` record
`cudaLaunchAttributeLaunchCompletionEvent` on grouped expert GEMMs so
other streams may `wait_event` at kernel start. Store `pin_hot` replica
D2D on `n_gpus >= 2` waits that event on the copy stream instead of
draining the GEMM, so leftover compute overlaps the replica. Illegal with
`--device-launch`. `--launch-completion` is off by default (decode
identity). `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaLaunchAttributeMemSyncDomain` decode isolation

`GpuStoreCfg::mem_sync_domain` / `SimCfg::mem_sync_domain` set
`cudaStreamSetAttribute` MemSyncDomain on the decode compute stream
(prefill stays Default). `MemSyncDomain::Remote` isolates leftover
prefill `same_domain_fence_permille` when `--decode-priority` puts
decode GEMMs on a second stream. Engine `--mem-sync-domain remote`
implies `--decode-priority`. Walker does not. Store leaf graph replay
SetAttributes the exec node to the launch stream when they disagree.
`--mem-sync-domain` is Default by default (decode identity).
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaGraphNodeSetEnabled` combo reuse

`GpuStoreCfg::graph_enable` / `SimCfg::graph_enable` capture a wide walker
combo parent of resident expert leaves, then `cudaGraphNodeSetEnabled`
skips extra children on a later subset instead of instantiating a new
parent. Child index is not cover order. Implies `--cuda-graphs`. Illegal
with `--device-launch`. Store GEMM stays per-leaf. `--graph-enable` is
off by default (decode identity). `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — memcpy-batch expert prefetch

`GpuStoreCfg::memcpy_batch` / `SimCfg::memcpy_batch` fill a multi-expert
pinned/VMM prefetch window with `cudaMemcpyBatchAsync`. Sibling H2D
copies share one stream-order snapshot. Demand acquire stays sequential.
Illegal with pageable, host-sync, mapped, or managed fills.
`--memcpy-batch` is off by default (decode identity). A same-stream
`cudaMallocAsync` pointer may be enqueued in the batch without a host
sync (the copy waits the alloc via stream order). `gpu-profile capture`
is still refused.

## Shipped 2026-08-30 — `cudaLaunchAttributeCooperative`

`KernelNodeAttr::Cooperative` is `cudaGraphKernelNodeGetAttribute` /
`SetAttribute` / `CopyAttributes` for cooperative launch. Typed
`graph_kernel_node_get_cooperative` / `set_cooperative` stay. Setting
`true` requires `cooperative_launch` and occupies every Hyper-Q slot at
launch. Definition Set does not retarget exec.
`graph_exec_kernel_set_params` still refuses a cooperative mismatch.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — managed batch prefetch/discard require CMA

`prefetch_batch_async` / `discard_batch_async` /
`discard_and_prefetch_batch_async` are `cudaMemPrefetchBatchAsync` /
`cudaMemDiscardBatchAsync` / `cudaMemDiscardAndPrefetchBatchAsync`. They
require `ConcurrentManagedAccess` on every GPU. This VM reports 0, so
they are Invalid `"concurrent managed access"`. Typed `prefetch` stays.
Discard contents are not modeled. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaMemcpy3DBatchAsync`

`memcpy_3d_batch_async` is pointer-to-pointer `cudaMemcpy3DBatchAsync`
(`MemcpyOp` must be 3D). `flags` must be 0. CUDA arrays are not modeled.
Sibling copies share a stream-order snapshot or empty DuringApiCall/Any
deps like `memcpy_batch_async`. `memcpy_3d_with_attributes` Stream is
`memcpy_3d_async`. Capture cannot include a 3D batch. `gpu-profile
capture` is still refused.

## Shipped 2026-08-30 — `cudaMemcpyBatchAsync`

`memcpy_batch_async` is 1D pointer-to-pointer `cudaMemcpyBatchAsync`.
Copies in one batch share a snapshotted stream-order predecessor list
(or empty DuringApiCall/Any deps) so they occupy copy engines together;
later submits wait for the whole batch. DuringApiCall waits those copies
before return. Any does not. `memcpy_with_attributes` Stream is `memcpy`.
`PreferOverlapWithCompute` is ignored (discrete). Location hints are
omitted (`ConcurrentManagedAccess` / `PageableMemoryAccess` are 0).
Capture cannot include a batch. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — official gemma3n

`general.architecture=gemma3n` with convert-shaped `gemma3n.*` KV. Decode
follows `src/models/gemma3n.cpp`: Gemma3 QK-Norm + post-norms + GeGLU,
attention scale `1.0`, unweighted V RMSNorm, AltUp (4 streams) + Laurel +
per-layer inputs, gaussian_topk on the first 10 layers, SWA default
period 5, final tanh softcap (default 30). Writer-tiny
`attention.sliding_window = 2`. `gemma4` stays rejected.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — official gemma3

`general.architecture=gemma3` with convert-shaped `gemma3.*` KV. Decode
follows `src/models/gemma3.cpp`: Gemma2 post-norms + GeGLU, QK-Norm before
RoPE, no attn logit softcap, optional final tanh softcap, SWA default
period 6. Writer-tiny `attention.sliding_window = 2`. `gemma3n` stays
rejected. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaStreamAttributeAccessPolicyWindow`

`set_stream_access_policy` / `StreamAttr::AccessPolicy` store a persisting
L2 window on the stream. `kernel` / `kernel_bufs` inherit it so billing
matches `kernel_access_policy`. `kernel_with` and graph replay use the
launch / node window. Set `None` clears. `stream_copy_attributes` copies
or clears. Decode identity stays `cudaLaunchKernel` with no stream window.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — official bloom

`general.architecture=bloom` with convert-shaped `bloom.*` KV. Decode
follows `src/models/bloom.cpp`: `token_embd_norm`, fused `attn_qkv`,
ALiBi (hardcoded bias 8, no RoPE), sequential LayerNorm GELU-seq.
`gemma2` stays rejected. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — mempool UsedMemHigh / ReservedMemHigh

`MemPoolAttr::UsedMemHigh` / `ReservedMemHigh` are ordinary-pool
high-water. Set `0` resets to current. Graph mem stays `GraphMemAttr`.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — graph MemsetNodeSetParams 3D helper

`graph_memset_set_params_3d` / `graph_exec_memset_set_params_3d` require
a 3D `MemsetOp` (`depth > 1`). Typed SetParams stays. `gpu-profile
capture` is still refused.

## Shipped 2026-08-30 — graph MemsetNodeSetParams 2D helper

`graph_memset_set_params_2d` / `graph_exec_memset_set_params_2d` require
a 2D `MemsetOp` (`height > 1`, not 3D). Typed SetParams stays.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — graph MemcpyNodeSetParams 3D helper

`graph_memcpy_set_params_3d` / `graph_exec_memcpy_set_params_3d` require
a 3D `MemcpyOp` (`depth > 1`). Typed SetParams stays. `gpu-profile
capture` is still refused.

## Shipped 2026-08-30 — graph MemcpyNodeSetParams 2D helper

`graph_memcpy_set_params_2d` / `graph_exec_memcpy_set_params_2d` require
a 2D `MemcpyOp` (`height > 1`, not 3D). Typed SetParams stays.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — graph `AddMemsetNode` 3D helper

`graph_add_memset_3d` requires a 3D `MemsetOp` (`depth > 1`). Typed
`graph_add_memset_op` stays. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — graph `AddMemsetNode` 2D helper

`graph_add_memset_2d` requires a 2D `MemsetOp` (`height > 1`, not 3D).
Typed `graph_add_memset_op` stays. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — graph `AddMemcpyNode` 3D helper

`graph_add_memcpy_3d` requires a 3D `MemcpyOp` (`depth > 1`). Typed
`graph_add_memcpy` stays. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — graph `AddMemcpyNode` 2D helper

`graph_add_memcpy_2d` requires a 2D `MemcpyOp` (`height > 1`, not 3D).
Typed `graph_add_memcpy` stays. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — expertvm KV pitched fill uses `memcpy_2d_async`

Pitched `kv_paged` miss fills call `memcpy_2d_async` / `memset_2d_async`
when the op is 2D. Packed 1D stays `memcpy` / `memset_op`. `gpu-profile
capture` is still refused.

## Shipped 2026-08-30 — `cudaMemcpy3DPeer` requires `is_3d`

`memcpy_peer_3d` / `memcpy_peer_3d_async` require a 3D `MemcpyOp`
(`depth > 1`). Typed `memcpy_peer` stays. `gpu-profile capture` is still
refused.

## Shipped 2026-08-30 — `cudaMemcpy2DPeer` requires `is_2d`

`memcpy_peer_2d` / `memcpy_peer_2d_async` require a 2D `MemcpyOp`
(`height > 1`, not 3D). Typed `memcpy_peer` stays. `gpu-profile capture`
is still refused.

## Shipped 2026-08-30 — `cudaMemset3D` / `Memset3DAsync`

`memset_3d` / `memset_3d_async` require a 3D `MemsetOp` (`depth > 1`).
Typed `memset_op` stays. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaMemset2D` / `Memset2DAsync`

`memset_2d` / `memset_2d_async` require a 2D `MemsetOp` (`height > 1`,
not 3D). Typed `memset_op` stays. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaMemcpy3D` / `Memcpy3DAsync`

`memcpy_3d` / `memcpy_3d_async` require a 3D `MemcpyOp` (`depth > 1`).
Typed `memcpy` stays. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaMemcpy2D` / `Memcpy2DAsync`

`memcpy_2d` / `memcpy_2d_async` require a 2D `MemcpyOp` (`height > 1`,
not 3D). Typed `memcpy` stays. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — mempool export/import handle type

`pool_export_with_type` / `pool_import_with_type` take POSIX-FD and
flags 0 (`MemPoolExportFlags`). Typed `pool_export` / `pool_import`
stay. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — stream batch memop flags

`batch_mem_op_with_flags` / `graph_add_batch_mem_op_with_flags` take
`cuStreamBatchMemOp` flags (`BatchMemOpFlags`; must be 0). Typed
`batch_mem_op` stays. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — stream write-value flags

`write_value32_with_flags` / `write_value64_with_flags` take
`CU_STREAM_WRITE_VALUE_*` (`WriteValueFlags`). NO_MEMORY_BARRIER is
Invalid. Typed `write_value32` stays. Graph twins
`graph_add_write_value32_with_flags` / `graph_add_write_value64_with_flags`.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — stream wait-value flags

`wait_value32_with_flags` / `wait_value64_with_flags` take
`CU_STREAM_WAIT_VALUE_*` (`WaitValueFlags`). FLUSH is Invalid. Typed
`wait_value32` stays. Graph twins `graph_add_wait_value32_with_flags` /
`graph_add_wait_value64_with_flags`. `gpu-profile capture` is still
refused.

## Shipped 2026-08-30 — partial host-native atomics unsupported

`DeviceAttr::OnlyPartialHostNativeAtomicSupported` is always 0
(host-mapped atomics are not modeled). Distinct from
`HostNativeAtomicSupported` (already 0). Query; capture-legal.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — NUMA config None

`DeviceAttr::NumaConfig` is always `DeviceNumaConfig::NONE` (GPU memory
NUMA nodes are not modeled). Do not invent `cudaDevAttrNumaId`. Query;
capture-legal. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — host NUMA multi-node IPC unsupported

`DeviceAttr::HostNumaMultinodeIpcSupported` is always 0 (this VM's IPC
is same-node; `ipc_open` requires the dest GPU already in the
allocation). Distinct from `IpcEventSupport` (always 1). Query;
capture-legal. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — host NUMA memory pools unsupported

`DeviceAttr::HostNumaMemoryPoolsSupported` is always 0 (host NUMA
pools are not modeled; `create_pool_with_props` refuses host location).
Distinct from `HostMemoryPoolsSupported` (already 0). Query;
capture-legal. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — 32-bit stream memops

`DeviceAttr::CanUseStreamMemOps` is always 1 (`wait_value32` /
`write_value32`). CUDA deprecated this in favor of
`CanUse64BitStreamMemOps`. Query; capture-legal. `gpu-profile capture`
is still refused.

## Shipped 2026-08-30 — host NUMA VMM unsupported

`DeviceAttr::HostNumaVirtualMemoryManagementSupported` is always 0
(host NUMA VMM is not modeled; `va_create_with_prop` refuses host
location). Distinct from device VMM (always 1). Query; capture-legal.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaMemPoolSetAccess` descriptor array

`pool_set_access_n` takes `cudaMemAccessDesc` (`descList`, `count`).
All-or-nothing. Empty is a no-op after pool checks. Typed
`pool_set_access` stays. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cuMemSetAccess` descriptor array

`va_set_access_n` takes `CUmemAccessDesc` (`desc`, `count`). All-or-nothing.
Empty is a no-op after the size check. Typed `va_set_access` stays.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — mem decompress max length unsupported

`DeviceAttr::MemDecompressMaximumLength` is always 0 (hardware
decompress is not modeled). Query; capture-legal. `gpu-profile capture`
is still refused.

## Shipped 2026-08-30 — mem decompress unsupported

`DeviceAttr::MemDecompressAlgorithmMask` is always 0 (hardware decompress
is not modeled). Query; capture-legal. `gpu-profile capture` is still
refused.

## Shipped 2026-08-30 — timeline semaphore interop unsupported

`DeviceAttr::TimelineSemaphoreInteropSupported` is always 0 (NVSci /
timeline semaphore interop is not modeled). Query; capture-legal.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — unified function pointers unsupported

`DeviceAttr::UnifiedFunctionPointers` is always 0 (device-side function
pointers are not modeled). Query; capture-legal. `gpu-profile capture`
is still refused.

## Shipped 2026-08-30 — tensor-map access unsupported

`DeviceAttr::TensorMapAccessSupported` is always 0 (`CUtensorMap` / TMA
is not modeled). Query; capture-legal. `gpu-profile capture` is still
refused.

## Shipped 2026-08-30 — stream wait-value NOR

`DeviceAttr::CanUseStreamWaitValueNor` is always 1 (`WaitValueCmp::Nor`).
Query; capture-legal. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — 64-bit stream memops

`DeviceAttr::CanUse64BitStreamMemOps` is always 1 (`wait_value64` /
`write_value64`). Query; capture-legal. `gpu-profile capture` is still
refused.

## Shipped 2026-08-30 — kernel exec timeout off

`DeviceAttr::KernelExecTimeout` is always 0 (example SKUs have no
display watchdog). Query; capture-legal. `gpu-profile capture` is still
refused.

## Shipped 2026-08-30 — TCC driver unsupported

`DeviceAttr::TccDriver` is always 0 (example SKUs are not Windows TCC).
Query; capture-legal. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — compute mode Default

`DeviceAttr::ComputeMode` is always `ComputeMode::DEFAULT`. Exclusive
process is not modeled. Query; capture-legal. `gpu-profile capture` is
still refused.

## Shipped 2026-08-30 — `cuMulticastGetGranularity` prop

`multicast_get_granularity_with_prop` takes `CUmulticastObjectProp`
(handle types none; flags 0; size and team size are not validated).
Typed `multicast_get_granularity` stays. Query; capture-legal.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cuDeviceGet`

`device_get` is the ordinal in `0 .. device_count`. Query; capture-legal.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cuMulticastCreate` prop

`multicast_create_with_prop` is `CUmulticastObjectProp` (handle types
none; flags 0). Typed `multicast_create` stays. `gpu-profile capture` is
still refused.

## Shipped 2026-08-30 — `cuMemSetAccess` size

`va_set_access_with_size` requires the CUDA reserved-VA size. Partial
SetAccess is not modeled. Typed `va_set_access` stays. `gpu-profile
capture` is still refused.

## Shipped 2026-08-30 — `cuDeviceTotalMem`

`device_total_mem` is HBM bytes (same as TotalGlobalMem). Query;
capture-legal. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaDeviceGetName`

`device_get_name` is the profile name (same as `DeviceProperties::name`).
Query; capture-legal. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cuMemMap` size

`va_map_handle_with_size` requires the CUDA handle size. Typed
`va_map_handle` stays. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — multicast `cuMemMap` size

`va_map_multicast_with_size` requires the CUDA multicast object size.
Typed `va_map_multicast` stays. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — multicast `cuMemMap` flags

`va_map_multicast_with_flags` requires flags 0. Typed `va_map_multicast`
stays. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cuMulticastBindMem` size

`multicast_bind_mem_with_size` requires the CUDA handle size. Partial
bind is not modeled (`mcOffset` / `memOffset` 0). Typed
`multicast_bind_mem` stays. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cuMulticastBindAddr` size

`multicast_bind_addr_with_size` requires the CUDA reserved-VA size.
Partial bind is not modeled (`mcOffset` 0). Typed `multicast_bind_addr`
stays. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cuMulticastUnbind` size

`multicast_unbind_with_size` requires the CUDA multicast object size.
Partial unbind is not modeled (`mcOffset` 0). Typed `multicast_unbind`
stays. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cuMemUnmap` size

`va_unmap_with_size` requires the CUDA reservation size. Partial unmap
stays `va_unmap_range`. Typed `va_unmap` stays. `gpu-profile capture` is
still refused.

## Shipped 2026-08-30 — PointerAttr memory-block id

`pointer_get_attribute` reports MemoryBlockId as the VMM `MemHandleId`
covering offset 0. `cudaMalloc` and combined `va_map` without retain stay
Invalid. Set stays SyncMemops only. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — multi-GPU board attributes

`device_get_attribute` / `device_get_properties` report IsMultiGpuBoard
and MultiGpuBoardGroupID always 0 (example SKUs are discrete single-GPU
packages). `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cuMulticastBindMem` flags

`multicast_bind_mem_with_flags` requires flags 0. Typed `multicast_bind_mem`
stays. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — host memory pools unsupported

`device_get_attribute` / `device_get_properties` report
HostMemoryPoolsSupported always 0 (pools are device-only).
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cuMemMap` flags

`va_map_handle_with_flags` requires flags 0. Typed `va_map_handle` stays.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cuMemAddressFree` size

`va_free_with_size` requires the CUDA reservation size. Partial free is
not modeled. Typed `va_free` stays. `gpu-profile capture` is still
refused.

## Shipped 2026-08-30 — `cuMemAddressReserve` flags

`va_reserve_with_flags` is alignment / addr / flags (flags 0; fixed VA
is not modeled; alignment must divide size). Typed `va_reserve` stays.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cuMemCreate` prop + flags

`va_create_with_prop` is `cuMemCreate` with `CUmemAllocationProp` and
flags (0; pinned device; VMM POSIX-FD export is not modeled).
Typed `va_create` stays. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cuMulticastBindAddr`

`multicast_bind_addr` binds a mapped VMM VA (retain + BindMem).
`multicast_store` uses BindAddr. Flags must be 0.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — multicast destroy

`multicast_destroy` is `cuMemRelease` of a `cuMulticastCreate` handle.
Live multicast VA maps are Invalid; leftover binds are dropped.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cuMulticastUnbind`

`multicast_unbind` drops a whole-handle bind. Live multicast VA maps are
Invalid. `va_unmap` decrements the map count so unbind can proceed.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — PointerAttr hardware decompress

`pointer_get_attribute` reports IsHwDecompressCapable always 0
(compression is not modeled). Set stays SyncMemops only.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — Win32 / fabric handle-type device attributes

`device_get_attribute` / `device_get_properties` report Win32, Win32 KMT,
and fabric handle types unsupported (always 0; this VM has POSIX-FD
shareable pools). `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — RDMA flush-options / VMM / compression attrs

`device_get_attribute` / `device_get_properties` report
GPUDirectRDMAFlushWritesOptions (Host on an RDMA SKU; MemOps never),
GPUDirectRDMAWithCudaVMMSupported (same RDMA SKU; VMM always on), and
GenericCompressionSupported (always 0). Write-ordering stays unmodeled.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaDevAttrHandleTypePosixFileDescriptorSupported`

`device_get_attribute` / `device_get_properties` report POSIX-FD handle
types supported (always 1; this VM has shareable pools).
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaDevAttrVirtualMemoryManagementSupported`

`device_get_attribute` / `device_get_properties` report VMM supported
(always 1; this VM has `va_reserve`). `gpu-profile capture` is still
refused.

## Shipped 2026-08-30 — `cudaDevAttrMulticastSupported`

`device_get_attribute` / `device_get_properties` report MulticastSupported
from a GPU↔GPU NVLink on that device. PCIe P2P and RDMA stay 0.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cuMulticastGetGranularity`

`multicast_get_granularity` reports the profile multicast granularity
(minimum and recommended are the same; `0`/`1` → `1`).
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cuMemGetAllocationGranularity`

`va_get_allocation_granularity` reports the profile VA granularity
(minimum and recommended are the same; `0`/`1` → `1`).
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cuMemGetAllocationPropertiesFromHandle`

`va_get_allocation_properties` reports pinned device location and RDMA
capability for a `cuMemCreate` handle. Handle types stay none.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cuMemGetAccess`

`va_get_access` reports VMM ProtReadWrite / ProtRead / ProtNone from
local maps and `va_set_access` / `va_set_access_write`. Unmapped VA is
Invalid. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — PointerAttr VMM mapping queries

`pointer_get_attribute` reports MappingBaseAddr / MappingSize (the
`cuMemMap` span at offset 0, not the reserved VA). Unmapped VMM is
Invalid. Set stays SyncMemops only. `gpu-profile capture` is still
refused.

## Shipped 2026-08-30 — Managed location type and id

`mem_range_get_attribute` reports PreferredLocationType / Id and
LastPrefetchLocationType / Id (`0` Invalid / `1` Device / `2` Host).
Host NUMA is not modeled. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — Last-prefetch range attribute

`mem_range_get_attribute` reports `LastPrefetchLocation` from
`prefetch` / `prefetch_host` (unset until the first prefetch).
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — PointerAttr capability queries

`pointer_get_attribute` reports IsLegacyCudaIpcCapable /
IsGpuDirectRdmaCapable / AllowedHandleTypes. Set stays SyncMemops only.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — PointerAttr identity queries

`pointer_get_attribute` reports DeviceOrdinal / RangeStartAddr / BufferId.
Unmapped host has no ordinal. Set stays SyncMemops only.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — Func cluster scheduling policy preference

`set_func_cluster_policy` / `get_func_cluster_policy` are
`cudaFuncAttributeClusterSchedulingPolicyPreference` (per device; Default
launches inherit occupancy). CUDA ints `0`/`1`/`2`.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — Required cluster dim function attributes

`set_cluster_dim_must_be_set` and `set_required_cluster_width` / height /
depth are `cudaFuncAttributeClusterDimMustBeSet` /
`RequiredClusterWidth` / Height / Depth. A missing or mismatched cluster
is Invalid. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — PointerAttr query wrappers

`pointer_get_attribute` reports MemoryType / DevicePointer / HostPointer /
IsManaged / RangeSize / Mapped / MemPoolHandle. Set stays SyncMemops only.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — Func preferred shared-memory carveout

`set_func_carveout` / `get_func_carveout` are
`cudaFuncAttributePreferredSharedMemoryCarveout` (per device; Default
launches inherit occupancy). CUDA ints `-1`/`0`/`100` only.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — Func shared-mem config, remaining device flags, SyncMemops

`set_func_shared_mem_config` / `get_func_shared_mem_config` are
`cudaFuncSetSharedMemConfig` / `GetSharedMemConfig` (per device; Default
kernels inherit function then device). `DeviceFlags::MAP_HOST` /
`LMEM_RESIZE_TO_MAX` are stored. `SYNC_MEMOPS` makes runtime memcpy/memset
wait the stream like pointer SyncMemops (graph-add is not refused).
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — HostAlloc Portable / WriteCombined stored flags

`alloc_host_with_flags` stores `PORTABLE` / `WRITE_COMBINED` (no DMA
change). `host_register_with_flags` accepts Portable. `host_get_flags`
returns the stored word. IoMemory / ReadOnly stay Invalid.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — SharedMemConfig, event BlockingSync, device schedule flags

`set_shared_mem_config` / `get_shared_mem_config` are
`cudaDeviceSetSharedMemConfig` / `GetSharedMemConfig` (Default kernels
inherit; unset is unscaled). `EventCreateFlags::BLOCKING_SYNC` taxes
`synchronize_event` with `host_sync_blocking_ns`. `set_device_flags` /
`get_device_flags` are the schedule mask (`cudaSetDeviceFlags`); Auto
streams inherit the tax. MapHost / Lmem / SyncMemops stay Invalid.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — Advise location, SyncMemops, CUDA-array/dma-buf DeviceAttr

`mem_advise_with_location` is `cudaMemAdvise_v2` (`Place` location;
AccessedBy requires a device place). `pointer_set_attribute` /
`pointer_get_attribute` model `CU_POINTER_ATTRIBUTE_SYNC_MEMOPS`
(memcpy/memset wait the stream like pageable; capture of those copies
is refused; graph-add is not). `SparseCudaArraySupported` /
`DeferredMappingCudaArraySupported` / `DmaBufSupported` are always 0.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — VMM access flags and cudaMemGetAddressRange

`va_set_access_with_flags` maps PROT_READ / PROT_READWRITE / PROT_NONE
onto typed VMM helpers. `mem_get_address_range` returns `(alloc, bytes)`
for a live id (interior offsets not modeled). Query; capture-legal.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — Pool access flags and discrete/unmodeled DeviceAttr zeros

`pool_set_access_with_flags` maps `PROT_READ_WRITE` / `PROT_NONE` onto
typed helpers. `PROT_READ` is Invalid (pool ProtRead is not modeled).
`HostNativeAtomicSupported` / `CooperativeMultiDeviceLaunch` /
`Integrated` are always 0. Query; capture-legal. `gpu-profile capture`
is still refused.

## Shipped 2026-08-30 — MemPool reuse attrs, managed/flush DeviceAttr, RDMA flush

`MemPoolAttr` reuse flags default to 1. Only `ReuseAllowOpportunistic=0`
skips cache reuse (OS alloc; cached bytes stay reserved). FollowEvent /
Internal are stored and do not insert waits. ConcurrentManagedAccess /
DirectManagedMemAccessFromHost / PageableMemoryAccessUsesHostPageTables
are always 0. `CanFlushRemoteWrites` follows GPUDirect RDMA.
`flush_gpu_direct_rdma_writes` is a 1 ns host-sync barrier (capture
refused; no write-visibility). `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — Prefetch flags and P2P NativeAtomic/CudaArray = 0

`prefetch_with_flags` requires `PrefetchFlags::DEFAULT` (`0`). Device dest
is typed `prefetch`; host dest is `prefetch_host`. Other bits Invalid.
`DeviceP2pAttr::NativeAtomicSupported` and `CudaArrayAccessFromDevice`
are always 0. Query; capture-legal. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaMemPoolCreate` props

`create_pool_with_props` is `cudaMemPoolCreate` + `cudaMemPoolProps`.
Pinned alloc type only. NONE / POSIX-FD handle types map to typed
`create_pool` / `create_shareable_pool`. `max_size` caps reserved bytes
(`0` unlimited). Host location and other handle bits are Invalid. Typed
helpers stay. Capture refused. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — StreamPriorities / GpuOverlap / UnifiedAddressing

`DeviceAttr::StreamPrioritiesSupported` and `UnifiedAddressing` are 1.
`GpuOverlap` is `copy_engines > 0`. Query; legal during capture.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaMemset` host-sync

`memset_sync` / `memset_op_sync` are `cudaMemset` / `2D` / `3D`
(host-synchronous; capture refused). Typed `memset` / `memset_op` stay
Async. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaStreamAttachMemAsync` flags

`stream_attach_with_flags` maps `MemAttachFlags::{GLOBAL, HOST, SINGLE}`
onto `MemAttach` then typed `stream_attach` (capture refuse / Single+NULL
/ stream-match stay). Other bits Invalid `"stream attach flags"`. Typed
`stream_attach` stays. Query; capture is refused by the typed helper.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaIpcOpenMemHandle` flags

`ipc_open_with_flags` accepts `IpcMemFlags::LAZY_ENABLE_PEER_ACCESS` as a
no-op (dest must already hold the source). Cross-GPU lazy peer is not
modeled. Typed `ipc_open` stays. Capture refused. `gpu-profile capture`
is still refused.

## Shipped 2026-08-30 — `cudaMallocManaged` flags

`alloc_managed_with_flags` is Global / Host. Single and other bits are
Invalid `"managed flags"`. Typed helpers stay. Capture refused.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaHostGetDevicePointer` flags

`host_get_device_pointer_with_flags` requires
`HostGetDevicePointerFlags::DEFAULT` (`0`). Unknown bits are Invalid.
Typed `host_get_device_pointer` stays. Query; legal during capture.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — HostRegisterReadOnly / PageableMemoryAccess are 0

`DeviceAttr::HostRegisterReadOnlySupported` and `PageableMemoryAccess`
are 0. ReadOnly host register is Invalid; pageable is bounce-buffer.
Do not report coherent pageable access. Query; legal during capture.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaMemcpy2DPeer` / `cudaMemcpy2DPeerAsync`

`memcpy_peer_2d` is host-synchronous; capture refused.
`memcpy_peer_2d_async` bills 2D payload (not pitch padding). Places are
forced to `src`/`dst`. Typed `memcpy` stays. `gpu-profile capture` is
still refused.

## Shipped 2026-08-30 — `cudaMemcpy3DPeer` / `cudaMemcpy3DPeerAsync`

`memcpy_peer_3d` is host-synchronous; capture refused.
`memcpy_peer_3d_async` bills 3D payload (not pitch padding). Places are
forced to `src`/`dst`. Typed `memcpy` stays. `gpu-profile capture` is
still refused.

## Shipped 2026-08-30 — `cudaDevAttrGPUDirectRDMASupported`

`DeviceAttr::GpuDirectRdmaSupported` is a GPU↔GPU `LinkKind::Rdma` link.
Flush/write-ordering are not modeled. Query; legal during capture.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaMemRangeGetAttributes`

`mem_range_get_attributes` batches modeled per-alloc managed advice
(all-or-nothing). Last-prefetch is not modeled. Query; legal during
capture. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaMemcpyPeer` / `cudaMemcpyPeerAsync`

`memcpy_peer` is host-synchronous (`cudaMemcpyPeer`); capture refused.
`memcpy_peer_async` is the stream-ordered replica copy (typed
`memcpy_device_to_device` stays). `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaDeviceEnablePeerAccess` flags

`enable_peer_with_flags` requires `PeerAccessFlags::DEFAULT` (`0`).
Unknown bits are Invalid `"peer access flags"`. Typed `enable_peer`
stays. Capture is legal. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaDevP2PAttrPerformanceRank`

`device_get_p2p_attribute(..., PerformanceRank)` is unique GPU↔GPU link
`bps` descending (lower is better). Same device or no link is 0. Native
atomics are not modeled. Query; legal during capture. `gpu-profile
capture` is still refused.

## Shipped 2026-08-30 — DeviceAttr cluster / host-register / IPC / POSIX pool

`DeviceAttr::ClusterLaunch` is `max_blocks_per_cluster > 0`.
`HostRegisterSupported` / `IpcEventSupport` /
`CanUseHostPointerForRegisteredMem` are always 1.
`MemoryPoolSupportedHandleTypes` is POSIX-FD. Query; legal during capture.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaMemRangeGetAttribute`

`Sim::mem_range_get_attribute` returns modeled per-alloc managed advice
(ReadMostly / PreferredLocation / AccessedBy). Not per-byte; last-prefetch
is not modeled. Query; legal during capture. `gpu-profile capture` is
still refused.

## Shipped 2026-08-30 — `cudaFuncSetAttribute` / `GetAttribute`

`Sim::func_set_attribute` / `func_get_attribute` dispatch `FuncAttr` onto
the typed per-device setters. Negative max-dynamic-shared or a non-0/1
non-portable-cluster value is Invalid. Typed helpers stay. `gpu-profile
capture` is still refused.

## Shipped 2026-08-30 — `cudaStreamCreateWithFlags` / `CreateWithPriority`

`Sim::stream_create_with_flags` / `stream_create_with_priority` take
`StreamCreateFlags`. Known bit: NonBlocking. Unknown bits are Invalid.
NULL is Invalid. Capture cannot include them. Typed `set_stream_blocking`
/ `set_stream_priority` stay. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaIpcGetEventHandle` / `OpenEventHandle`

`Sim::ipc_get_event` / `ipc_open_event` export an Interprocess event and
import an alias of the source record. Interprocess requires DisableTiming.
`create_event_interprocess` is the typed helper. Destroy of the source
while imports are live is Invalid. Capture cannot include event IPC.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaHostAlloc` / `cudaHostRegister` flags

`Sim::alloc_host_with_flags` / `host_register_with_flags` take
`HostAllocFlags`. Known bit: MAPPED. Portable / WriteCombined / IoMemory /
ReadOnly are Invalid. Typed helpers stay. `gpu-profile capture` is still
refused.

## Shipped 2026-08-30 — `cudaHostGetFlags`

`Sim::host_get_flags` returns `HostAllocFlags::MAPPED` or `0`. Device,
managed, and unregistered pageable pointers are Invalid. Query; legal
during capture. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaGraphDestroyNode`

`Sim::graph_destroy_node` drops a definition node and incident edges.
Remaining indices stay valid. Illegal on an exec and during capture.
Definition destroy does not retarget exec. `gpu-profile capture` is still
refused.

## Shipped 2026-08-30 — `cudaGraphAddMemcpyNode1D` / SetParams1D

`Sim::graph_add_memcpy_1d` / `graph_memcpy_set_params_1d` /
`graph_exec_memcpy_set_params_1d` pack `MemcpyOp::packed_1d`. SetParams1D
may convert a 2D/3D node to 1D. Pageable copies stay illegal. Definition
Set does not retarget exec. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaGraphDebugDotPrint` flags

`Sim::graph_debug_dot_with_flags` takes `GraphDebugDotFlags` (CUDA bits).
Flags `0` stays kinds and edges. `VERBOSE` dumps modeled params.
External-semaphore bits are Invalid. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — event record/wait/create with flags

`Sim::record_event_with_flags` / `wait_event_with_flags` /
`create_event_with_flags` are `cudaEventRecordWithFlags` /
`cudaStreamWaitEvent` flags / `cudaEventCreateWithFlags`. Known bits:
External and DisableTiming. BlockingSync / Interprocess are Invalid.
Typed helpers stay. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaGraphKernelNodeGetAttribute` / `SetAttribute`

`Sim::graph_kernel_node_get_attribute` / `set_attribute` (and exec twins)
dispatch `KernelNodeAttr` onto the typed kernel-node getters/setters.
Definition Set does not retarget exec. Typed helpers stay.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaStreamGetId`

`Sim::stream_get_id` is unique per `(device, stream)` (not the
caller-chosen `StreamId`). Query; legal during capture. Unknown devices
are Invalid. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaStreamGetCaptureInfo` `id_out`

`StreamCaptureInfo::id` is unique per begin-capture sequence (starts at 1).
Forked streams share it. Query. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaStreamGetCaptureInfo_v2` dependencies

`StreamCaptureInfo::dependencies` is the last same-stream captured node
union extra `pending_deps`. Query. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaGraphNodeGetParams`

`Sim::graph_node_get_params` / `graph_exec_node_get_params` return
`GraphNodeParams` from the definition / exec snapshot. Query; Empty
returns `Empty`; Alloc is bytes only. IF/WHILE/SWITCH are Invalid.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — bulk graph Add/RemoveDependencies

`Sim::graph_add_dependencies_n` / `graph_remove_dependencies_n` are
`cudaGraphAddDependencies` / `RemoveDependencies` of N from/to pairs
(all-or-nothing). Pairwise helpers call them. `gpu-profile capture` is
still refused.

## Shipped 2026-08-30 — `cudaGraphNodeSetParams` / `ExecNodeSetParams`

`Sim::graph_node_set_params` / `graph_exec_node_set_params` dispatch
`GraphNodeParams` onto the typed SetParams. Alloc and Empty are Invalid.
Definition does not retarget exec. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaGraphAddNode`

`Sim::graph_add_node` is `cudaGraphAddNode` (`GraphNodeParams` plus
dependency indices). Typed `graph_add_*` stay. Alloc fills
`GraphAddNode::alloc`. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — graph event/child definition SetParams

`Sim::graph_event_record_set_event` / `graph_event_wait_set_event` are
`cudaGraphEventRecordNodeSetEvent` / `WaitNodeSetEvent` on the graph
definition (External flag stays topology; does not retarget exec).
`graph_child_set_params` is `cudaGraphChildGraphNodeSetParams` on the
definition (nested topology may change; stores the child id as passed).
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — OpenAI `n` must be 1

`gguf_gemv serve` accepts `"n":1` and rejects any other completion count.
Omitted `n` is still one greedy completion. `gpu-profile capture` is still
refused.

## Shipped 2026-08-30 — `cudaGraphDebugDotPrint`

`Sim::graph_debug_dot` prints stored node kinds and edges as DOT. Query;
legal during capture. No verbose kernel-param flags. `gpu-profile capture`
is still refused.

## Shipped 2026-08-30 — graph mem-free Get/SetParams; serve `echo` / `add_special_tokens`

`Sim::graph_free_get_params` is `cudaGraphMemFreeNodeGetParams`.
`graph_free_set_params` / `graph_exec_free_set_params` are
`cudaGraphMemFreeNodeSetParams` / `cudaGraphExecMemFreeNodeSetParams`
(definition SetParams does not retarget an exec). `POST /v1/completions`
`echo` returns prompt plus completion. `POST /tokenize`
`add_special_tokens` (default true) is BOS via `prompt_ids`.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — graph node GetGraph / GetEvent / GetParams

`Sim::graph_child_get_graph` is `cudaGraphChildGraphNodeGetGraph`.
`graph_event_record_get_event` / `graph_event_wait_get_event` are
`cudaGraphEventRecordNodeGetEvent` / `WaitNodeGetEvent`.
`graph_alloc_get_params` is `cudaGraphMemAllocNodeGetParams` of stored
id and bytes. Query; legal during capture. `gpu-profile capture` is
still refused.

## Shipped 2026-08-30 — `POST /tokenize` / `POST /detokenize`

`gguf_gemv serve` (default and `--engine`) maps `POST /tokenize` onto the
same prompt / messages path as generate and returns `{"tokens","count"}`
of those ids. `POST /detokenize` takes `{"tokens":[...]}` and returns
`{"text"}`. GET is 405. `--engine` does not admit a sequence.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaGraphGetNodes` / `cudaStreamGetAttribute`

`Sim::graph_nodes` is `cudaGraphGetNodes` (`0 .. graph_len` in creation
order). Query; legal during capture. `stream_get_attribute` /
`stream_set_attribute` are `cudaStreamGetAttribute` / `SetAttribute` of
existing stream state (priority, synchronization policy, mem-sync
domain/map, NVLink-util-centric, access-policy window). Green-context SM
permille is not a CUDA stream attribute. Type mismatch is Invalid
`"stream attr"`. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaEventDestroy`

`Sim::destroy_event` is `cudaEventDestroy`. A recorded incomplete event
waits like `synchronize_event`. A never-recorded event returns immediately.
Unknown ids are `UnknownEvent`. Capture cannot include it. The id may be
created again. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaMemPoolDestroy`

`Sim::destroy_pool` is `cudaMemPoolDestroy`. Unused cached bytes return to
the OS. Outstanding allocs stay valid until freed. Destroying the current
device mempool rebinds GetMemPool to GetDefaultMemPool. Default and
graph-memory pools cannot be destroyed. Capture cannot include it.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaMemPoolGetAttribute` / `cudaFuncGetAttributes` / GetAccess

`Sim::pool_get_attribute` / `pool_set_attribute` are `cudaMemPoolGetAttribute`
/ `SetAttribute` of existing live, cached, and release-threshold state.
Used/Reserved are read-only. No invented ordinary-pool high-water (graph
mem stays `graph_mem_get`). Graph-memory pool is Invalid. Imported pools
report the exporter. `func_get_attributes` is `cudaFuncGetAttributes` of
per-device `maxDynamicSharedSizeBytes` and `nonPortableClusterSizeAllowed`
(not per kernel). `DeviceAttr::CanMapHostMemory` / `ManagedMemory` are
always 1. `pool_get_access` is `cudaMemPoolGetAccess` (owner ReadWrite;
peers after SetAccess). Query gets are capture-legal; SetAttribute cannot
capture. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaDeviceGetMemPool` vs GetDefaultMemPool

`Sim::default_pool` is `cudaDeviceGetDefaultMemPool` (seeded at construct).
`device_mempool` is `cudaDeviceGetMemPool`. `cudaDeviceSetMemPool` rebinds
`alloc` without replacing GetDefaultMemPool. ExpertStore SetAccess and
cached-byte helpers follow GetMemPool. Query; legal during capture.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaGetDeviceProperties` / stream flags / serve GET

`Sim::device_get_properties` is `cudaGetDeviceProperties` of modeled SKU
caps only (HBM, shared mem, L2, copy engines, Hyper-Q, cooperative launch,
cluster sizes, mem-sync domains, mempools). No SM count or clock.
`DeviceAttr::TotalGlobalMem` / `AsyncEngineCount` wrap `hbm_bytes` /
`copy_engines`. `stream_get_flags` is `cudaStreamGetFlags` (`0` blocking /
`1` NonBlocking; NULL follows `set_legacy_null_stream`).
`stream_get_priority` is `cudaStreamGetPriority`. `gguf_gemv serve`
`GET /v1/models` / `GET /v1/models/{id}` list/retrieve `--model-id`
(default GGUF stem). Completions/chat include `"model"`. `GET /health` is
`{"status":"ok"}`. `GET /metrics` is idle `{"engine":false}` or Engine
counters with `--engine`. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaGetDeviceCount` / P2P attributes

`Sim::device_count` is `cudaGetDeviceCount`. `device_can_access_peer` /
`device_get_p2p_attribute` expose `cudaDevP2PAttrAccessSupported` from the
profile topology (a device–device link). Same device is 0. Missing links
are 0. Independent of `enable_peer`. Query APIs; legal during capture.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaMemset3DAsync`

`MemsetOp` depth/ysize is `cudaMemset3DAsync`: billed HBM write is
`width * height * depth` (row and slice padding are not written). Packed
1D/2D (`depth` 0/1) is unchanged. Decode identity stays packed 1D.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaMalloc3D` / `cudaMemcpy3DAsync`

`Sim::malloc_3d` is `cudaMalloc3D` (pitch `align_up(width, 512)`, charge
`pitch * height * depth`). `MemcpyOp` depth/slice heights are
`cudaMemcpy3DAsync` (payload `width * height * depth`; row and slice padding
are not billed). Packed 1D/2D (`depth` 0/1) is unchanged. Decode identity
stays packed 1D. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaMemset2DAsync`

`MemsetOp` height/pitch is `cudaMemset2DAsync`: billed HBM write is
`width * height` (pitch padding is not written). The mapped span is the 2D
extent. `Sim::memset_op` / `graph_add_memset_op` take that struct. Packed 1D
(`height` 0/1) is unchanged. `expertvm kv --fill memset --row-width W --pitch P`
uses the 2D fill. Decode identity stays packed 1D. `gpu-profile capture` is
still refused.

## Shipped 2026-08-30 — `cudaHostGetDevicePointer` / `cudaDeviceGetAttribute`

`Sim::host_get_device_pointer` of mapped host returns the same id. Unmapped
host and device allocs are Invalid. `Sim::device_get_attribute` exposes
cooperative launch, concurrent kernels (`compute_slots > 1`), shared memory,
L2 / persisting L2 max, max blocks per cluster, mem-sync domain count, and
memory-pool support. Example H100 `compute_slots` is 1 so concurrent kernels
is 0. Query APIs; legal during capture. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaPointerGetAttributes` / `cudaDeviceGetLimit` / `cudaMemcpy2D`

`Sim::pointer_get_attributes` classifies Unregistered / Host / Device / Managed
(CUDA 11+ freed pointers are Unregistered). `Sim::set_limit` / `get_limit` wrap
`cudaDeviceSetLimit` / `GetLimit`: persisting L2 is the existing API,
`MaxL2FetchGranularity` is 32/64/128 (SM 8.0+ default 128) and access-policy
windows must align to it. `Sim::malloc_pitch` is `cudaMallocPitch`.
`MemcpyOp` height/pitches are `cudaMemcpy2DAsync` (payload `width * height`).
`expertvm kv --row-width W --pitch P` uses that 2D fill. Decode identity stays
packed 1D / persist limit 0. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — OpenAI `/v1/completions` and `/v1/chat/completions`

`gguf_gemv serve` (default and `--engine`) maps `POST /v1/completions` and
`POST /v1/chat/completions` onto the same greedy path as `POST /generate`.
`max_tokens` aliases `n_predict`. The OpenAI `choices` envelope returns the
completion only (`text` / `message.content`), not the prompt+decode string.
`--engine` `"stream": true` on those routes is chunked SSE (`data:` lines,
then `data: [DONE]`). Native `/generate` is unchanged. Keep-alive still
applies. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — expertvm `--graph-mem-trim`

`expertvm sim` / `schedule` / `store`, `gguf_gemv engine` / `serve`, and
`infer-bench` inherit `GpuStoreCfg::graph_mem_trim` /
`SimCfg::graph_mem_trim`. After the walk, `cudaDeviceGraphMemTrim` returns
unused reserved graph-mem so live `hbm_used` is expert pages, not abandoned
scratch. `hbm_peak` still includes scratch during launch. Decode identity
stays off. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — HTTP/1.1 keep-alive

`gguf_gemv serve` (default and `--engine`) keeps the TCP connection open after
`POST /generate` unless the client sends `Connection: close` or HTTP/1.0.
Pipelined requests reuse the persistent KV / Engine. OpenAI `/v1` routes
share the same keep-alive. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — dedicated graph-memory pool

Captured `cudaMallocAsync` and `cudaGraphAddMemAllocNode` draw from a per-device
graph-memory pool (`u64::MAX` release threshold), not the default mempool.
`UsedMemCurrent` is live graph allocs. `ReservedMemCurrent` is live plus unused
cached bytes. `cudaDeviceGraphMemTrim` returns unused reserved so
`cudaMemGetInfo` free grows. Destroy of a definition parks remaining graph mem
until trim. User `alloc_from_pool` / `set_device_mempool` refuse that pool.
Decode identity stays kernel-only graphs. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — expertvm `--kernel-priority N`

`expertvm sim` / `schedule` / `store`, `gguf_gemv engine` / `serve`, and
`infer-bench schedule` take `--kernel-priority N`. Grouped expert GEMMs launch
with `cudaLaunchAttributePriority`. `None` inherits stream create priority.
`N` (including `0`) overrides that kernel when compute contends. Distinct from
`--stream-priority`. Graph instantiate uses `UseNodePriority` so captured node
values are used at replay. Legal with `--pdl` and `--cooperative`. Decode identity
stays inherit-stream. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaLaunchAttributePriority`

Launch-time override of `cudaStreamCreateWithPriority` (`KernelAttrs::priority`).
`None` inherits the stream. `Some` schedules that kernel at the given priority
when compute contends (higher first). Capture snapshots the effective value.
Default instantiate still uses the launch stream unless
`cudaGraphInstantiateFlagUseNodePriority`. Device-launch graphs allow it.
Decode identity stays inherit-stream. Distinct from `--stream-priority`.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — expertvm `--device-launch` / `--device-updatable`

`expertvm sim` / `schedule` / `store`, `gguf_gemv engine` / `serve`, and
`infer-bench schedule` take `--device-launch` and `--device-updatable`.
Leaf GEMM graphs instantiate with `cudaGraphInstantiateFlagDeviceLaunch`
and launch via `device_launch_graph`. `--device-updatable` is
`cudaLaunchAttributeDeviceUpdatableKernelNode` so `--graph-set-params`
keeps the exec uploaded. Illegal with `--graph-update` and with
`--graph-mem` / `--graph-auto-free`. Combo parents stay per-leaf launches.
Legal with `--pdl` and `--cooperative`. Decode identity stays host launch /
not device-updatable. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — device-updatable is graphs-only

A non-capturing `kernel_with` with `cudaLaunchAttributeDeviceUpdatableKernelNode`
is Invalid. Capture still records the attr. Decode identity stays disabled.

## Shipped 2026-08-30 — Llama NORM real-model fixture

`tests/reference/llama-3.2-1b-instruct-q4_k_m.json` is a llama.cpp capture of
`Llama-3.2-1B-Instruct-Q4_K_M.gguf` (architecture `llama`, NORM RoPE). Token
ids (including BOS 128000), argmax, and a 24-token greedy continuation match.
`real_llama_norm_matches_llama_cpp_reference` fail-louds when
`LLAMA_RUST_REAL_MODEL_DIR` is set. Does not download Hugging Face in CI.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — expertvm `--nvlink-util`

`expertvm sim` / `schedule` / `store`, `gguf_gemv engine` / `serve`, and
`infer-bench schedule` take `--nvlink-util`. Grouped expert GEMMs launch with
`cudaLaunchAttributeNvlinkUtilCentricScheduling`. Occupies every Hyper-Q slot
when the profile has NVLink so leftover prefill cannot overlap decode even
with `--compute-slots 2`. Without NVLink occupancy is unchanged. Legal with
`--pdl` and `--cooperative`. Decode identity stays disabled.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaLaunchAttributeNvlinkUtilCentricScheduling`

Launch / stream / graph-node flag (`0` disabled / `1` enabled). CUDA treats it
as a hint; this VM occupies every Hyper-Q slot when the profile has NVLink.
Without NVLink the flag is stored and occupancy is unchanged. Stream
SetAttribute is inherited by `kernel`. Graph CopyAttributes copies it.
Device-launch graphs allow it. Decode identity stays disabled.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — expertvm `--dynamic-shared` / `--optin-shared` / `--portable-shared`

`expertvm sim` / `schedule` / `store`, `gguf_gemv engine` / `serve`, and
`infer-bench schedule` take `--dynamic-shared N`, `--optin-shared`, and
`--portable-shared default|portable|non-portable`. Grouped expert GEMMs launch
with `cudaLaunchKernel` `sharedMemBytes` and CUDA 13
`cudaLaunchAttributeSharedMemoryMode`. `--optin-shared` sets
`cudaFuncAttributeMaxDynamicSharedMemorySize` to the SKU opt-in max. Default
uses the function attribute. RequirePortable always refuses oversize.
AllowNonPortable allows up to the opt-in max even when `--optin-shared` is off.
Legal with `--pdl` and `--cooperative`. Decode identity stays 0 bytes / Default.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — CUDA 13 portable dynamic shared memory

`PortableSharedMode` is CUDA 13 `cudaLaunchAttributeSharedMemoryMode`
(`cudaSharedMemoryMode`), distinct from bank-width `SharedMemoryMode`.
`KernelAttrs::dynamic_shared` is `cudaLaunchKernel` `sharedMemBytes`.
Default uses `cudaFuncAttributeMaxDynamicSharedMemorySize` (`0` = portable
`max_shared_mem_per_block`). RequirePortable always refuses oversize.
AllowNonPortable allows up to `max_shared_mem_per_block_optin`. Example
H100 keeps portable == optin so decode identity stays 0 bytes / Default.
Device-launch graphs allow it. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — expertvm `--portable-cluster`

`expertvm sim` / `schedule` / `store`, `gguf_gemv engine` / `serve`, and
`infer-bench schedule` take `--portable-cluster default|portable|non-portable`.
Grouped expert GEMMs launch with `cudaLaunchAttributePortableClusterSizeMode`.
Default uses the function attribute. RequirePortable always refuses oversize.
AllowNonPortable allows up to the SKU max even when `--non-portable-cluster`
is off. Legal with `--pdl` and `--cooperative`. Decode identity stays Default.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaLaunchAttributePortableClusterSizeMode`

Launch-time override of `cudaFuncAttributeNonPortableClusterSizeAllowed`
(SetAttribute / CopyAttributes / `KernelAttrs`). Default uses the current
function attribute (resolved at launch). RequirePortable always refuses a
size above `portable_cluster_size`. AllowNonPortable allows up to the SKU
`max_blocks_per_cluster`. Device-launch graphs allow it. Decode identity
stays Default (function attr disallowed). `gpu-profile capture` is still
refused.

## Shipped 2026-08-30 — expertvm `--shared-mem`

`expertvm sim` / `schedule` / `store`, `gguf_gemv engine` / `serve`, and
`infer-bench schedule` take `--shared-mem default|four|eight`. Grouped expert
GEMMs launch with `cudaLaunchAttributeSharedMemoryMode`. Default never scales
duration. FourByte / EightByte scale kernel time by
`1000 / shared_mem_*_permille` (profile default 1000 is identity). Legal with
`--pdl` and `--cooperative`. Decode identity stays Default.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaLaunchAttributeSharedMemoryMode`

Kernel-node bank width Default / FourByte / EightByte (SetAttribute /
CopyAttributes / `KernelAttrs`). Default never scales duration. FourByte /
EightByte scale kernel time by `1000 / shared_mem_*_permille` (profile
default 1000 is identity). Device-launch graphs allow it. Decode identity
stays Default. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaLaunchAttributeDeviceUpdatableKernelNode`

Kernel-node flag (graph SetAttribute / CopyAttributes / `KernelAttrs`).
Default false. `graph_exec_kernel_set_params` keeps the exec uploaded so a
later `device_launch_graph` needs no host re-upload. Control without the
flag still requires `upload_graph`. Device-launch graphs allow it.
Decode identity stays not device-updatable. `gpu-profile capture` is still
refused.

## Shipped 2026-08-30 — expertvm `--sync-policy`

`expertvm sim` / `schedule` / `store`, `gguf_gemv engine` / `serve`, and
`infer-bench schedule` take `--sync-policy auto|spin|yield|blocking`.
Created streams get `cudaLaunchAttributeSynchronizationPolicy`. Decode-stream
ITL (`--decode-priority`) pays `host_sync_*_ns` on `synchronize_stream`.
Auto tax is 0. Legal with `--pdl` and `--cooperative`. Decode identity stays
Auto. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaLaunchAttributeSynchronizationPolicy`

Stream-only (`cudaStreamSetAttribute` / CopyAttributes). Auto / Spin / Yield /
BlockingSync. Host-wait tax on `synchronize_stream` and `synchronize_event`
(recording stream) after the GPU drain. Profile `host_sync_*_ns` default 0
(Auto always 0) so decode identity and existing timing tests stay green.
`synchronize` / `synchronize_device` do not take the tax. Not a kernel
launch attribute. Decode identity stays Auto. `gpu-profile capture` is
still refused.

## Shipped 2026-08-30 — expertvm `--non-portable-cluster`

`expertvm sim` / `schedule` / `store`, `gguf_gemv engine` / `serve`, and
`infer-bench schedule` take `--non-portable-cluster`. Grouped expert GEMMs
may launch a cluster larger than `portable_cluster_size` up to the SKU
`max_blocks_per_cluster` (`cudaFuncAttributeNonPortableClusterSizeAllowed`).
Example H100 keeps both at 8. Legal with `--pdl` and `--cooperative`.
Decode identity stays disallowed. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — expertvm `--preferred-cluster`

`expertvm sim` / `schedule` / `store`, `gguf_gemv engine` / `serve`, and
`infer-bench schedule` take `--preferred-cluster N`. Grouped expert GEMMs
launch with Hopper preferred cluster dimension so occupancy uses that size
when it fits in `--compute-slots`, else the required `--cluster`. Needs
`--cluster`; `N` must be a multiple of it. `N==0` is refused at parse.
Legal with `--pdl` and `--cooperative`. Decode identity stays no preferred
dim. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — expertvm `--max-shared`

`expertvm sim` / `schedule` / `store`, `gguf_gemv engine` / `serve`, and
`infer-bench schedule` take `--max-shared`. Grouped expert GEMMs launch with
MaxShared L1/shared carveout so the launch occupies every Hyper-Q slot.
Legal with `--pdl` and `--cooperative`. Decode identity stays Default.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaLaunchAttributePreferredSharedMemoryCarveout`

MaxShared occupies every Hyper-Q slot. Default and MaxL1 keep current
occupancy. Graph SetAttribute / CopyAttributes carry it. Device-launch
graphs allow it. Decode identity stays Default. `gpu-profile capture` is
still refused.

## Shipped 2026-08-30 — expertvm `--cluster-spread`

`expertvm sim` / `schedule` / `store`, `gguf_gemv engine` / `serve`, and
`infer-bench schedule` take `--cluster-spread`. Grouped expert GEMMs launch
with Hopper cluster scheduling Spread so the launch occupies every Hyper-Q
slot even when `--cluster N` is smaller than `--compute-slots`. A no-op
without `--cluster` of at least 2. Legal with `--pdl` and `--cooperative`.
Decode identity stays Default. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — cluster scheduling, preferred dim, non-portable size

`cudaLaunchAttributeClusterSchedulingPolicyPreference` Spread occupies every
Hyper-Q slot. Default and LoadBalancing keep `min(blocks, compute_slots)`.
`cudaLaunchAttributePreferredClusterDimension` occupies the preferred size when
it fits. `cudaFuncAttributeNonPortableClusterSizeAllowed` is required for a
cluster larger than `portable_cluster_size` (Hopper 8) up to the SKU
`max_blocks_per_cluster`. Decode identity stays no cluster / Default policy /
disallowed non-portable. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — expertvm `--cluster`

`expertvm sim` / `schedule` / `store`, `gguf_gemv engine` / `serve`, and
`infer-bench schedule` take `--cluster N`. Grouped expert GEMMs launch as a
Hopper thread-block cluster of X size `N` so the launch occupies
`min(N, compute_slots)` Hyper-Q slots. A cluster that fills the cap cannot
overlap leftover kernels. `N==0` is refused at parse. Legal with `--pdl`
and `--cooperative`. Decode identity stays `cudaLaunchKernel` (no cluster).
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaLaunchAttributeClusterDimension`

`kernel_with` / graph SetAttribute launch a Hopper thread-block cluster.
The product of `{x,y,z}` occupies `min(blocks, compute_slots)` Hyper-Q slots.
Hopper portable max is 8 (`max_blocks_per_cluster`). Zero dims and oversize
clusters are Invalid. Device-launch graphs allow it. Decode identity stays
`cudaLaunchKernel` (no cluster). `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaLaunchAttributeMemSyncDomain`

`kernel_with` / stream and graph SetAttribute select Default vs Remote and a
physical map (`cudaDevAttrMemSyncDomainCount`; example H100 is 4). A completing
kernel's implicit fence waits `same_domain_fence_permille` of leftover
same-domain kernel or allreduce traffic. Default tax is 0 (identity). Allreduce
tags Remote like NCCL. Graph replay uses the node's map, not the launch stream.
Device-launch graphs allow it. Decode identity stays `cudaLaunchKernel`.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaLaunchAttributeAccessPolicyWindow`

`kernel_access_policy` applies a persisting L2 window so a reused expert
GEMM bills less HBM after `set_persisting_l2_cache_size` (CUDA default 0).
The first kernel fills the persist cache; `reset_persisting_l2_cache`
colds it. Graph SetAttribute / CopyAttributes carry the window.
Device-launch graphs allow it. `expertvm sim --l2-persist` / Engine
`--expert-sim --l2-persist` enable the limit and attach a window to expert
GEMMs. Decode identity stays `cudaLaunchKernel` with persist limit 0.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaLaunchAttributeLaunchCompletionEvent`

`kernel_launch_completion` records an event when the kernel grid is
launched (`start_ns`), so other streams can `wait_event` it while the
primary is still running. Graph SetAttribute / CopyAttributes carry it.
Device-launch graphs refuse it. Decode identity stays `cudaLaunchKernel`.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaLaunchAttributeProgrammaticEvent`

`kernel_pdl_event` records an event at the PDL trigger so other streams
can `wait_event` it before the primary kernel finishes. Without trigger
the event records at completion. `query_event` / `synchronize_event` /
`event_elapsed_ns` use the trigger. Graph SetAttribute / CopyAttributes
carry it. Device-launch graphs refuse it. Decode identity stays
`cudaLaunchKernel`. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — expertvm `--pdl`

`expertvm sim` / `schedule` / `store`, `gguf_gemv engine` / `serve`, and
`infer-bench schedule` take `--pdl`. Consecutive same-stream expert GEMMs
launch with wait+trigger so they may overlap after the previous kernel's
PDL trigger when `--compute-slots` is `>=2`. Illegal with `--cooperative`.
`cudaFreeAsync` on that stream still waits for the overlapped primary
(all preceding work). Decode identity stays `cudaLaunchKernel`.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — programmatic dependent launch

`kernel_pdl` / `graph_kernel_node_set_pdl` model CUDA PDL. A wait kernel
may start after the previous same-stream kernel's trigger
(`pdl_trigger_permille`) instead of its completion. Overlap needs
`compute_slots >= 2`. Decode identity stays `kernel`. `gpu-profile capture`
is still refused.

## Shipped 2026-08-30 — `cudaUserObjectCreate` / graph retain

`user_object_create` is `cudaUserObjectCreate` (`NO_DESTRUCTOR_SYNC`).
Graphs retain refs with `graph_retain_user_object` (`MOVE` transfers one
caller ref). Last remaining ref records `user_object_destructors`. Clone
does not copy retains. Exec ids and capture are Invalid. Decode identity
does not create user objects. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaGraphExecUpdateResultInfo`

`update_graph_with_info` fills `GraphExecUpdateResultInfo` even on `Err`.
Node-type, dependency, mem-node, device-launch, and cooperative mismatches
are classified. `update_graph` uses this path and keeps the same `why`
strings. Decode identity stays `update_graph`. `gpu-profile capture` is
still refused.

## Shipped 2026-08-30 — graph/exec GetParams

`graph_kernel_get_params` / `graph_memcpy_get_params` /
`graph_memset_get_params` / `graph_host_get_params` /
`graph_batch_mem_ops_get_params` are `cudaGraph*NodeGetParams` on the
definition. `graph_exec_*_get_params` reads the exec snapshot.
Unique helpers still use the launched snapshot. Decode identity stays
kernel-only graphs. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaGraphInstantiateWithParams`

`instantiate_graph_with_params` fills `GraphInstantiateParams` result and
err node even on `Err`. Device-launch of a forbidden node is
`NodeOperationNotSupported` with that index. Auto-free plus a mem-free node
is `InvalidStructure`. `instantiate_graph_with_flags` uses this path.
Decode identity stays `instantiate_graph`. `gpu-profile capture` is still
refused.

## Shipped 2026-08-30 — `cudaStreamCaptureMode`

`begin_capture` defaults to Relaxed: independent streams stay live, and a
wait of a captured record still joins an idle stream. `begin_capture_with_mode`
accepts Global / ThreadLocal / Relaxed. Global and ThreadLocal refuse
uncaptured-stream submits (`stream not capturing`) except a joining wait.
`thread_exchange_stream_capture_mode` is the thread default for the next
`begin_capture`. Decode identity stays Relaxed. `gpu-profile capture` is
still refused.

## Shipped 2026-08-30 — host-node SetParams

`HostNodeParams` (`fn_id` / `user_data`) is `cudaHostNodeParams` on
`GpuOp::HostFunc`. `host_func` / `graph_add_host_func` stay the unnamed
default. `graph_host_set_params` patches the definition and does not retarget
an exec. `graph_exec_host_set_params` patches the snapshot
(`graph_set_params_ns`). Capture records the payload. Device-launch graphs
still refuse host nodes. Decode identity stays kernel-only graphs.
`gpu-profile capture` is still refused.

## Shipped 2026-08-30 — separate `cudaGraphExec_t` handles

`instantiate_graph` / `instantiate_graph_with_flags` return a new exec id.
The source graph stays a definition. A second instantiate of a kernel graph
creates another exec; mem alloc/free graphs cannot. `launch_graph` of a
definition uses the primary exec. `graph_add_*` is legal on the definition
after instantiate and Invalid on the exec. Destroying the definition refunds
graph mem and leaves execs launchable. Decode identity still launches the
definition. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — true `cudaGraphAddBatchMemOpNode` / `cuStreamBatchMemOp`

A multi-item batch is one `GpuOp::BatchMem` node, not sequential wait/write
nodes with deps. `batch_mem_op` is live `cuStreamBatchMemOp`; a single item
still submits `WriteValue` / `WaitValue`. A wait sees earlier writes in that
vector and does not see later ones. Writes commit on complete. Disabling the
node skips the whole batch. Item lists are parameters for `update_graph` and
`graph_exec_batch_mem_ops_set_params`. Decode identity stays kernel-only
graphs. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — `cudaGraphInstantiateFlagDeviceLaunch`

`instantiate_graph_with_flags(DEVICE_LAUNCH)` on kernel/memcpy/memset
graphs. `device_launch_graph` is device-side `cudaGraphLaunch` after
upload (host `launch_graph` still auto-uploads). The launcher occupies
one compute slot for `graph_launch_ns`; the body enqueues when it
completes. Mem alloc/free, events, child graphs, conditionals, and host
nodes are Invalid. `update_graph` of a device-launch exec is Invalid.
Overlapping device launches of the same exec are Invalid. Decode identity
stays default flags. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — stream wait/write-value (`cuStreamWaitValue`)

`wait_value64` / `write_value64` (and 32-bit) are `cuStreamWaitValue*` /
`cuStreamWriteValue*`. The mailbox updates when write **completes**;
unwritten locations read as 0. Kernel / memset / memcpy stores to that
address are not modeled. Wait stays pending until the compare matches
(no compute or copy occupancy). A write on another stream unblocks wait
without an event. Unsatisfied wait plus `synchronize` is deadlock.
`graph_add_wait_value64` / `graph_add_write_value64` /
`graph_add_batch_mem_op` are `cudaGraphAddBatchMemOpNode`. Graph vs exec
SetParams match memcpy. Decode identity stays kernel-only graphs.
`gpu-profile capture` is still refused. Device-launch is still Invalid.

## Shipped 2026-08-30 — graph-side memcpy/memset SetParams

`graph_memcpy_set_params` / `graph_memset_set_params` are
`cudaGraphMemcpyNodeSetParams` / `MemsetNodeSetParams` on the graph
definition. After instantiate they do not retarget the exec snapshot.
Decode identity stays ExecSetParams. `gpu-profile capture` is still refused.

## Shipped 2026-08-30 — dual-score `$/M tokens` from profile rent

`HardwareProfile::rent_usd_micros_per_hour` is an example list-price knob
(`$2.00/hr` is `2_000_000`). `0` omits dollars; example profiles stay `0`.
`Score::with_tokens` fills `usd_micros_per_m_tokens` from rent × wall /
tokens. Not a capture. `gpu-profile capture` is still refused. Device-launch
is still Invalid.

## Shipped 2026-08-30 — graph vs exec snapshot

`instantiate_graph` clones graph steps into an exec snapshot on the same
`GraphId`. Launch, `cudaGraphExec*SetParams`, and `cudaGraphNodeSetEnabled`
use the snapshot. `graph_kernel_set_params` is
`cudaGraphKernelNodeSetParams` on the graph and does not retarget an
already-instantiated exec. `graph_exec_kernel_node_set_priority` is the
exec attribute. `update_graph` replaces the snapshot. Graph and exec still
share one id (no separate exec handle). Device-launch is still Invalid.
Decode identity stays destroy+instantiate. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-30 — `cudaGraphInstantiateFlagUseNodePriority`

`instantiate_graph_with_flags(USE_NODE_PRIORITY)` schedules recorded
kernels with the add/capture priority instead of the launch stream.
`graph_kernel_node_get_priority` / `set_priority` / `copy_attributes` are
kernel-node Get/Set/CopyAttributes. `stream_copy_attributes` is
`cudaStreamCopyAttributes`. Device-launch is still Invalid. Decode
identity stays default flags. Dual score still has no `$/M tokens`.

## Shipped 2026-08-30 — `cudaDeviceGetGraphMemAttribute`

`Sim::graph_mem_get` counts live graph-mem allocs, not ordinary
`malloc` / `alloc`. Reserved equals used. `graph_mem_set` resets a High
attr at `0`. `graph_mem_trim` is host-sync and does not change
`mem_info`. Decode identity stays kernel-only graphs. Dual score still
has no `$/M tokens`.

## Shipped 2026-08-30 — `cudaGraphInstantiateWithFlags` / `GetFlags`

`Sim::instantiate_graph_with_flags` is `cudaGraphInstantiateWithFlags`:
`AUTO_FREE_ON_LAUNCH` and `UPLOAD` (upload during instantiate). Device-launch
and node-priority bits are Invalid. `graph_exec_get_flags` is
`cudaGraphExecGetFlags`. Decode identity stays default flags. Dual score
still has no `$/M tokens`.

## Shipped 2026-08-30 — `cudaGraphNodeFindInClone`

`Sim::graph_node_find_in_clone` is `cudaGraphNodeFindInClone`: the node
index on a graph produced by `clone_graph` of that original (nested
graphs cloned in that call count). A second clone of the clone does not
map the first original. Decode identity stays kernel-only graphs. Dual
score still has no `$/M tokens`.

## Shipped 2026-08-30 — conditional WHILE and SWITCH

`Sim::graph_add_while` is `cudaGraphCondTypeWhile`: the body repeats while
the handle is non-zero (Invalid after 64 iterations). `graph_add_switch`
is `cudaGraphCondTypeSwitch`: branch `i` runs when the handle equals `i`;
out of range skips every body. Decode identity stays kernel-only graphs.
Dual score still has no `$/M tokens`.

## Shipped 2026-08-30 — conditional IF graphs

`Sim::graph_conditional_create` is `cudaGraphConditionalHandleCreate`.
`graph_add_if` is an IF node whose body skips at start when the handle is
`0`. `set_conditional` is device `cudaGraphSetConditional` (capture
allowed; each launch resets to the create-time default). Decode identity
stays kernel-only graphs. Dual score still has no `$/M tokens`.

## Shipped 2026-08-30 — `cudaStreamUpdateCaptureDependencies`

`Sim::stream_update_capture_dependencies` is
`cudaStreamUpdateCaptureDependencies`: extra deps for the next captured
node, **in addition to** stream-order (`Set` replaces, `Add` unions).
`stream_is_capturing` / `stream_capture_info` are `cudaStreamIsCapturing`
/ `GetCaptureInfo`. `graph_node_kind` is `cudaGraphNodeGetType`. Decode
identity stays stream-capture edges. Dual score still has no `$/M tokens`.

## Shipped 2026-08-30 — `cudaGraphAddEmptyNode`

`Sim::graph_add_empty` is `cudaGraphAddEmptyNode`: a join/fork with no
work. Completes in 1 ns and does not occupy compute or copy engines, so
leftover kernels may Hyper-Q overlap it. Illegal after instantiate and
during capture. May be disabled. Capture-to-graph can name it as a
dependency anchor. Decode identity stays kernel-only graphs. Dual score
still has no `$/M tokens`.

## Shipped 2026-08-30 — `cudaStreamBeginCaptureToGraph`

`Sim::begin_capture_to_graph` is `cudaStreamBeginCaptureToGraph`: capture
into an existing uninstantiated graph. Capture roots additionally depend
on the given node indices; empty `deps` makes extra roots so independent
nodes may Hyper-Q overlap. `end_capture` returns that graph.
`graph_root_nodes` / `graph_edges` / `graph_node_dependents` query
topology. `--graph-piecewise` captures combo parents as independent child
roots (illegal with `--graph-build`). Decode identity stays a single
`begin_capture` of child launches. Dual score still has no `$/M tokens`.

## Shipped 2026-08-30 — `cudaGraphRemoveDependencies`

`Sim::graph_remove_dependencies` is `cudaGraphRemoveDependencies`: drop a
predecessor edge on an uninstantiated graph. Illegal after instantiate and
during capture. Missing edges are a no-op. Independent nodes may Hyper-Q
overlap at launch. Decode identity stays stream-capture edges. Dual score
still has no `$/M tokens`.

## Shipped 2026-08-30 — `cudaGraphExecEvent*NodeSetEvent`

`Sim::graph_exec_event_record_set_event` /
`graph_exec_event_wait_set_event` are `cudaGraphExecEventRecordNodeSetEvent`
/ `WaitNodeSetEvent`: retarget the event on an instantiated record or wait
node without a second graph. The External flag stays (topology). Cheaper
than `cudaGraphExecUpdate`. Legal with mem alloc/free nodes. Decode
identity stays kernel-only graphs. Dual score still has no `$/M tokens`.

## Shipped 2026-08-30 — `cudaGraphExecChildGraphNodeSetParams`

`Sim::graph_exec_child_set_params` is `cudaGraphExecChildGraphNodeSetParams`:
swap the nested graph of one instantiated child-graph node without a
second parent. Nested topology must match; child ids are topology for
`cudaGraphExecUpdate`. Cheaper than instantiate. Legal with mem alloc/free
nodes. `--graph-set-params` parks combo parents and retargets nested
leaves. Decode identity stays destroy+instantiate. Dual score still has
no `$/M tokens`.

## Shipped 2026-08-30 — Engine `--bench` infer-bench report

`gguf_gemv engine --bench` records the Engine's batched MoE traces and
prints `expertvm::report` (the same policy table as `infer-bench trace`;
with `--expert-sim`, the same sim scorecard). `--capacity N` is the replay
cache size (default `--expert-slots`, or 8). llama-rust is not an
infer-bench dependency. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — `cudaGraphNodeSetEnabled`

`Sim::graph_node_set_enabled` / `graph_node_get_enabled` are
`cudaGraphNodeSetEnabled` / `GetEnabled`: skip an instantiated node at
launch without rebuilding. Dependents wait for the disabled node's
predecessors. Memory alloc/free nodes cannot be disabled. Decode identity
stays every node enabled. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — `cudaGraphExecMemsetNodeSetParams`

`Sim::graph_exec_memset_set_params` is `cudaGraphExecMemsetNodeSetParams`:
patch one instantiated memset node's destination span without a second
graph. Cheaper than `cudaGraphExecUpdate`. Legal with mem alloc/free
nodes. `--graph-set-params` retargets a unique memset if the parked leaf
has one. Decode identity stays destroy+instantiate. Dual score still has
no `$/M tokens`.

## Shipped 2026-08-29 — `cudaGraphExecMemcpyNodeSetParams`

`Sim::graph_exec_memcpy_set_params` is `cudaGraphExecMemcpyNodeSetParams`:
patch one instantiated memcpy node's `MemcpyOp` without a second graph.
Cheaper than `cudaGraphExecUpdate`. Legal with mem alloc/free nodes.
Pageable copies stay illegal. `--graph-set-params` retargets a unique
memcpy if the parked leaf has one (copy stays off the compute GEMM graph).
Decode identity stays destroy+instantiate. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — mempool shareable-handle IPC

`Sim::create_shareable_pool` is `cudaMemPoolCreate` with a POSIX-FD handle
type. `pool_export` / `pool_import` are `cudaMemPoolExportToShareableHandle`
/ `ImportFromShareableHandle`: the import shares live/cached/threshold
(no extra HBM). `pool_export_ptr` / `pool_import_ptr` alias a live pool
alloc. `set_device_mempool` is `cudaDeviceSetMemPool`. Default and
`create_pool` pools cannot be exported. `ipc_get` of a mempool alloc is
Invalid. `--shareable` implies `--mempool` and rebinds `cudaMallocAsync`.
Decode identity stays the device default pool. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — `cudaGraphExecKernelNodeSetParams`

`Sim::graph_exec_kernel_set_params` is `cudaGraphExecKernelNodeSetParams`:
patch one instantiated kernel node's pointers / kind without a second graph.
Cheaper than `cudaGraphExecUpdate`. Legal with mem alloc/free nodes.
`--graph-set-params` parks leaf execs on evict and retargets the unique kernel.
Illegal with `--graph-update`. Decode identity stays destroy+instantiate.
Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — Hopper `cuMulticastCreate` NVLS replica fanout

`Sim::multicast_create` / `multicast_add_device` / `multicast_bind_mem` /
`va_map_multicast` are `cuMulticastCreate` / `cuMulticastAddDevice` /
`cuMulticastBindMem` / `cuMemMap` of a multicast handle. The team must be
an NVLink clique. Bind uses existing VMM physicals (dest HBM already
charged). A kernel write to the multicast VA is one NVLS hop on compute,
not N sequential copy-engine D2Ds. `--multicast` implies `--vmm` and
replaces `--place replicas` / `pin_hot` D2D with that kernel. Decode
identity stays copy-engine D2D. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — `cudaGraphAddDependencies`

`Sim::graph_add_dependencies` is `cudaGraphAddDependencies`. Independent
graph nodes (empty deps) run on internal streams so Hyper-Q can overlap
them; the launch stream still waits for the whole graph. Stream capture
records same-stream edges. `--graph-build` combo parents leave sibling
`graph_add_child` nodes independent. `cudaGraphExecUpdate` treats those
edges as topology. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — `cudaLaunchCooperativeKernel`

`Sim::cooperative_kernel` / `graph_add_cooperative_kernel` are
`cudaLaunchCooperativeKernel`. A cooperative grid occupies every Hyper-Q
compute slot, so leftover prefill cannot overlap decode the way `--compute-slots
2` can. Capture is allowed (CUDA 11+). `--cooperative` is opt-in on the walker,
Engine, serve, and infer-bench schedule. Decode identity stays
`cudaLaunchKernel`. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — `--graph-auto-free` (AutoFreeOnLaunch)

`--graph-auto-free` records leaf GEMM scratch without a matching free and
instantiates with `cudaGraphInstantiateFlagAutoFreeOnLaunch`, so relaunch
recharges HBM. Illegal with `--graph-mem`. `--graph-update` is skipped.
Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — `cudaGraphAddMemAllocNode` / `--graph-mem`

`Sim::graph_add_alloc` / `graph_add_free` are `cudaGraphAddMemAllocNode` /
`cudaGraphAddMemFreeNode`. `--graph-mem` records leaf GEMM graphs with
in-graph scratch workspace. Hits/misses stay the same; HBM peak includes
the scratch. `--graph-update` is skipped (CUDA cannot
`cudaGraphExecUpdate` mem nodes). Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — `cudaGraphCreate` / `cudaGraphAdd*`

`Sim::create_graph` is `cudaGraphCreate` (empty, uninstantiated).
`graph_add_kernel` / `graph_add_memcpy` / `graph_add_memset` /
`graph_add_host_func` / `graph_add_event_record` / `graph_add_event_wait` /
`graph_add_child` are `cudaGraphAdd*` (illegal after instantiate and
during capture). `--graph-build` uses this path in SimulatedGpuStore and
the `--cuda-graphs` walker. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — CUDA IPC mem handles

`Sim::ipc_get` / `ipc_open` / `ipc_close` are `cudaIpcGetMemHandle` /
`cudaIpcOpenMemHandle` / `cudaIpcCloseMemHandle`: the import aliases source
physicals (no extra HBM). Free of the source while imports are live is
Invalid. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — CUDA event RecordExternal / WaitExternal

`Sim::record_event_external` / `wait_event_external` are
`cudaEventRecordExternal` / `cudaEventWaitExternal`: captured without
forked-capture join so a live waiter can overlap graph launch. Dual score
still has no `$/M tokens`.

## Shipped 2026-08-29 — `cudaStreamAttachMemAsync`

`Sim::stream_attach` is stream-ordered managed-memory visibility
(`MemAttach::{Global,Host,Single}`). Default `alloc_managed` is Global.
`alloc_managed_host` is `cudaMemAttachHost`. Device kernels, memset, and
device prefetch fail when Host-attached or Single-attached to another
stream. Single cannot use the NULL stream. Capture is refused. Dual score
still has no `$/M tokens`.

## Shipped 2026-08-29 — VMM `cuMemSetAccess` PROT_READWRITE

`Sim::va_set_access_write` is PROT_READWRITE: a kernel on a peer may read
and write home VMM physicals without dest HBM (interconnect billed).
Default `va_set_access` stays PROT_READ. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — CUDA-graph AutoFreeOnLaunch

`Sim::instantiate_graph_auto_free` is `cudaGraphInstantiateFlagAutoFreeOnLaunch`:
graph mem allocs are `cudaFreeAsync`'d on the launch stream before a later
launch's alloc nodes, so relaunch recharges HBM. Default instantiate still
reuses the pointer. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — CUDA-graph mem alloc/free nodes

`cudaMallocAsync` / `cudaFreeAsync` (`Sim::alloc` / `free`) during stream
capture record graph mem nodes. Host-sync `malloc` / `free_sync` / VMM /
mempool create still cannot be captured. Relaunch without a matching free
reuses the pointer (no second HBM charge). `clone_graph` forks those ids.
`destroy_graph` refunds remaining graph mem. `update_graph` of mem nodes is
Invalid. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — VMM `cuMemRetainAllocationHandle`

`Sim::va_retain_handle` is `cuMemRetainAllocationHandle` (handle refs;
combined `va_map` spans are promoted so two VAs share one physical).
`va_release_handle` is `cuMemRelease` while mapped; HBM refunds when refs
and maps are both 0. `expertvm kv --sequences N` and Engine `--kv-sim`
interned blocks are `cuMemCreate` + `cuMemMap`. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — VMM `cuMemCreate` / `cuMemMap` split

`Sim::va_create` is `cuMemCreate` (HBM, no VA). `va_map_handle` is `cuMemMap`
of that handle so two reserved VAs share one physical without dest/extra
HBM. `va_release_handle` refunds when no maps remain. Combined `va_map`
stays Create+Map. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — Mempool `cudaMemPoolSetAccess` peer maps

`Sim::pool_set_access` is `cudaMemPoolSetAccess` ReadWrite on a mapped
mempool so a kernel on a peer can read **and write** home physicals
without dest HBM (interconnect billed). `--accessed-by` on pinned
`cudaMallocAsync` applies it to every default pool; pin/migrate/`--place
replicas` skip dest D2D. `cudaMalloc` / `--sync-alloc` still D2Ds. Dual
score still has no `$/M tokens`.

## Shipped 2026-08-29 — Trace-walker VMM `--place replicas`

`expertvm schedule --vmm --place replicas` maps dest then D2D like
SimulatedGpuStore pin (`va_unmap_range` on dest eviction). `--accessed-by`
skips dest HBM (`va_set_access` at fill). Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — VMM allocation granularity

`HardwareProfile::va_granularity_bytes` is `cuMemGetAllocationGranularity`.
Example default is `1` (any size), so 4096-byte expert VMM pages stay legal.
A 2 MiB profile (`with_va_granularity` / `.profile` `va_granularity_bytes=`)
rejects unaligned `va_reserve` / `va_map_range`. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — VMM `cuMemSetAccess` peer maps

`Sim::va_set_access` is `cuMemSetAccess` PROT_READ on a mapped VMM VA so a
kernel on a peer GPU can read physicals that live on the home GPU without
dest HBM (interconnect billed). Writes still need a local map. Capture is
refused. `GpuStoreCfg::accessed_by` / `SimCfg::accessed_by` apply at VMM
fills (`va_set_access` on every GPU), pin (skip dest map+D2D), and migrate
(retarget GEMM, keep home physicals). Managed `SetAccessedBy` is unchanged.
Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — Trace-walker `--decode-priority` ITL

`expertvm sim` / `schedule` / `store` and `infer-bench schedule` take the
same decode-stream ITL knob as Engine. Token 0 stays on the prefill stream;
later tokens GEMM on `StreamId(n_copy + 1)` at higher CUDA priority (CLI
implies `--stream-priority`). Token-boundary ITL samples that decode stream
so leftover prefill does not inflate it. Walker `--decode-sms` does not
imply `--decode-priority` (token 0 is prefill). Mixed leftover-prefill ITL
is strictly shorter than a full-device sample on a GEMM-bound profile.
Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — Trace-walker `--compute-slots` / `--decode-sms`

`expertvm sim` / `schedule` / `store` and `infer-bench schedule` take the
same Hyper-Q occupancy and green-context SM knobs as Engine. Independent
`--seq-streams` GEMMs overlap when `--compute-slots N` (`N>=2`). `--decode-sms
N` caps every replay stream (compute-bound kernels scale; memory-bound keep
full HBM). Default unset keeps exclusive compute and a full chip. Dual score
still has no `$/M tokens`.

## Shipped 2026-08-29 — Engine `--decode-sms` green-context SMs

`--expert-sim --decode-sms N` (`1..=1000`) reserves that permille of peak
FLOP/s for decode GEMMs (`Sim::set_stream_sm_permille`). Leftover prefill
gets the remainder. Implies `--decode-priority`. Compute-bound kernels
scale; memory-bound keep full HBM. Default unset is a full chip, so decode
identity stays. Mixed leftover-prefill ITL is strictly longer at 250‰ than
at a full chip on a GEMM-bound profile; greedy identity stays. Dual score
still has no `$/M tokens`.

## Shipped 2026-08-29 — Engine `--compute-slots` Hyper-Q

`--expert-sim --compute-slots N` (`N>=2`, with `--decode-priority`) lets
leftover prefill and decode GEMMs overlap at full issue rate on two
compute streams (Hyper-Q occupancy, not SM-partition). Default profile
occupancy is exclusive (`1`), so stream-priority contention and decode
identity stay. Mixed leftover-prefill `wall_ns` is strictly shorter than
one slot; greedy identity stays. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — Engine `--decode-priority` ITL

`--expert-sim --decode-priority` token-boundary ITL samples the decode
compute stream (`token_clock_ns`) so leftover prefill on the lower-priority
stream does not inflate it. Mixed leftover-prefill ITL is strictly shorter
than a full-device sample; greedy identity stays. An already-idle
`cudaStreamSynchronize` does not start leftover kernels on other streams.
1-GPU `pin_hot` skips the GEMM-lease wait (no replica). Default
`--expert-sim` keeps one compute stream and a full-device clock. Dual score
still has no `$/M tokens`.

## Shipped 2026-08-29 — Engine `--decode-priority`

`--expert-sim --decode-priority` runs decode GEMMs on a second compute
stream at higher CUDA priority than leftover prefill (implies
`--stream-priority`). Prefill/replay stays on the existing compute
stream. Default `--expert-sim` keeps one compute stream. Dual score
still has no `$/M tokens`.

## Shipped 2026-08-29 — Engine `--kv-sim`

`--expert-sim --kv-sim` maps Engine interned KV blocks onto the same
gpu-sim clock as expert H2D (`va_reserve` + memset on fault, kernel on
intern hit). Distinct from `expertvm kv`. Default `--expert-sim` stays
expert-only identity. `--kv-bytes` overrides intern page size so TTFT/ITL
include KV traffic (`kv_misses>=1`; shared-prefix intern `kv_hits>0`;
1MiB pages strictly lengthen `wall_ns`). Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — Engine place_hot fail-loud

Managed/VMM `place_hot` waits that expert page's GEMM and copy streams
before replica prefetch / `drop_managed_copy` / `va_unmap`. Engine
propagates pin and migrate errors (`allocation N is still leased` used
to be swallowed). 8×H100 managed and pinned two-sequence runs with
64-byte pages D2D onto GPU0 (`migrates>=1`) and keep greedy identity.
Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — Engine `--seq-streams`

`--expert-sim --seq-streams` maps each Engine sequence onto a copy stream
(`sequence % copy_engines.max(2)`) so concurrent H2D can overlap — the
real-KV analog of `expertvm sim --seq-streams`. Grouped expert GEMM stays
on one compute stream. Default `--expert-sim` keeps copy on NULL and
compute on stream 1. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — Engine predictor planner

`--prefetch none|copy-forward|markov|both` / `--plan-window N` /
`--plan-threshold N` park the same Stay vs Fetch predictor as
`expertvm sim` on Engine serving (`gguf_gemv engine` and
`serve --engine`). Upcoming keys are the online predicted list, not
JSONL future events. Default `both` / window `0` / threshold `500`
keeps today's copy-forward ∪ lookback-2 decode policy.
`--prefetch none` is demand paging (`prefetches=0`). Dual score still
has no `$/M tokens`.

## Shipped 2026-08-29 — Engine GpuStoreCfg CUDA knobs

`--expert-sim --host-func` / `--blocking-streams` / `--sync-alloc` /
`--mempool` / `--vmm-page N` / `--pageable` / `--accessed-by` /
`--legacy-null` / `--stream-priority` park the same `GpuStoreCfg` as
`expertvm sim` on Engine serving (`gguf_gemv engine` and
`serve --engine`). `--vmm-page N` with `N>0` implies `--vmm`. Default
pinned async stays decode identity. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — Engine GpuFill mapped / managed / vmm

`--expert-sim --mapped` / `--managed` / `--vmm` park SimulatedGpuStore
miss pages as `cudaHostAllocMapped`, `cudaMallocManaged`, or `va_acquire`
on the Engine serving path. Default stays pinned H2D identity. Dual score
still has no `$/M tokens`.

## Shipped 2026-08-29 — Engine CUDA graphs, ITL SLO, shared-prefix intern

`--expert-sim` captures per-page GEMM graphs on the Engine path
(`graph_launches=`). `--graph-update` parks a leaf on evict and
`cudaGraphExecUpdate`s the next miss; `--graph-clone` clones before
instantiate; `--timing-events` records copy elapsed. `--cuda-graphs` is
the discoverable no-op (graphs are on by default). `--itl-slo-ns` counts
later-token misses (`itl_slo_miss=`; does not drop). Batch-128 sequences
that share a `[1, 2]` first KV page intern more than disjoint 3-grams.
Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — Engine `TieredStore`

Engine parks `LiveStore::Tiered` the same way as CachedStore. Two
sequences and batch-128 (1-layer and 2-layer Qwen3MoE tinies) keep
greedy identity, acquire from the paging store, and GEMM together.
`WeightStorage::mmap` stays parked. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — Engine `slo_reject`

Waiting Engine sequences whose gpu-sim queue wait already meets
`ttft_slo_ns` are dropped (`Engine::rejected`), matching
`expertvm schedule --slo-reject`. `gguf_gemv engine --slo-reject
--ttft-slo-ns N --expert-sim` and `serve --engine --expert-sim
--slo-reject` wire the knobs. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — Engine gpu-sim TTFT / ITL

SimulatedGpuStore Engine runs sample the virtual clock at each newly
generated token. `expert_store_score` fills `ttft_ns`, `itl_ns`, and
`ns_per_token` (still no `$/M tokens`). On the two-layer Qwen3MoE tiny,
`decode_first` shortens the decode sequence's ITL while a leftover
prefill waits — the same policy as `expertvm schedule --decode-first`.

## Shipped 2026-08-29 — Engine `decode_first` + batch-128 Cached/GPU stores

`EngineCfg::decode_first` holds leftover prefill while any live sequence
is already decoding (same policy as `expertvm schedule --decode-first`).
Greedy ids still match independent decode. `gguf_gemv engine --decode-first`
and `serve --engine --decode-first` wire the knob. Engine batch-128 on
writer-tiny Qwen3MoE (1-layer and 2-layer) now also runs CachedStore
(full catalog) and SimulatedGpuStore (example H100, 4096-byte pages)
with identity, store hits, and gpu-sim `wall_ns`. Dual score still has
no `$/M tokens`.

## Shipped 2026-08-29 — Engine batch-128 MoE DirectStore

128 Engine prompts on writer-tiny Qwen3MoE (1-layer and 2-layer) with
`max_seqs=8` and DirectStore match independent greedy and acquire from
the store. Dense llama batch-128 stays on the blob FFN. Dual score still
has no `$/M tokens`.

## Shipped 2026-08-29 — Engine batch-128 waiting queue

128 Engine prompts with `max_seqs=8` wait, then finish with greedy ids
matching independent `greedy_generate_cache`. The 8 in-flight sequences
still GEMM together. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — batch-1 vs batch-128 + 2-layer GPU store

Adversarial `Workload::{Batch1, Batch, Batch128}` is 1 / 8 / 128 concurrent
sequences. `schedule-all` beats `schedule-1` on batch-128. Engine
SimulatedGpuStore on the two-layer tiny keeps greedy identity, bills
`wall_ns`, `plan_placement`s L0 and L+1, and copy-forward-prefetches more
than the 1-layer tiny. `infer-bench` / `expertvm schedule` replay
`tests/traces/tiny-qwen3moe-2layer.jsonl` with copy-forward L+1 hits.
Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — multi-layer seq persist + 2-layer JSONL

Layer-major MoE JSONL pairs `seq_persist‰` and lookback-2 Markov on the
**same layer**, not adjacent lines (`L0,t → L1,t → L0,t+1`). Copy-forward
and layer-forward Markov still train on L→L+1. Checked-in
`tests/traces/tiny-qwen3moe-2layer.jsonl` (`ab` + 8 tokens) reports
`layer_persist‰` and `seq_persist‰` both > 0. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — two-layer Qwen3MoE tiny + multi-layer oracle

`tiny_qwen3moe_2layer_gguf` / `gguf_gemv write-tiny-qwen3moe-2layer` writes
`qwen3moe.block_count=2` with `blk.1.*` cloned from `blk.0.*` so ExpertStore
copy-forward L+1 keys exist (1-layer tinies skip those as unknown). The
independent oracle walks every `block_count` layer; 2-layer logits match
and differ from the 1-layer tiny. CachedStore prefetches more on 2-layer
than on 1-layer. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — serve `--engine` stream / prefill-chunk / trace-out

`"stream": true` on `--engine` `POST /generate` is HTTP/1.1 chunked NDJSON
token lines then a final `generated` object (greedy identity). Concurrent
streams still GEMM together. Default serve ignores `stream`.
`--prefill-chunk N` is the same Engine knob as `gguf_gemv engine`.
`--trace-out FILE` appends batched MoE JSONL as sequences finish.
Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — Markov prefetch before grouped expert GEMM

Copy-forward ∪ lookback-2 prefetch runs after the router GEMM and
**before** grouped expert GEMM so H2D of L+1 can overlap this layer's
compute. `CachedStore` skips unknown catalog keys. Dual score still has
no `$/M tokens`.

## Shipped 2026-08-29 — serve `--engine` intern `page_hits`

`--engine` JSON `page_hits` is how many intern hits that sequence took
on the shared `PagedKvPool` (not a persistent per-connection cache).
`prefix_hit` stays 0. A later identical prompt intern-hits completed
pages after `Engine::take`. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — serve `--engine` ExpertStore

`gguf_gemv serve --engine --expert-slots N` parks DirectStore (`0`) or
CachedStore on the HTTP Engine. `--expert-sim` / `--expert-8gpu` /
`--expert-bytes` attach SimulatedGpuStore the same way as
`gguf_gemv engine`. Concurrent writer-tiny Qwen3MoE posts acquire from
the store and still GEMM together. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — serve `--engine` continuous-batch HTTP

`gguf_gemv serve --engine` admits concurrent loopback `POST /generate`
onto one `Engine` so prefills and decodes GEMM together (`gemm_peak`).
`--max-seqs N` is the in-flight cap (default 4). `--n-ctx` defaults to
64 and `--kv-page` to 16. JSON `generated` matches
`greedy_generate_ctx`. `prefix_hit` stays 0; `page_hits` is the intern
delta on the shared pool. Default serve without
`--engine` is still one HTTP request at a time. No Tokio, no
OpenAI-compat. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — pin_hot replicates onto the next GPU

`SimulatedGpuStore::pin_hot` D2Ds a replica onto `(home + 1) % n_gpus`
instead of always GPU1, so a striped expert on GPU3 copies to GPU4.
`StoreMetrics::replicates` counts those peer copies. Evict frees that
dest. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — engine `--expert-8gpu` / `--expert-bytes`

`gguf_gemv engine --expert-sim --expert-8gpu` attaches SimulatedGpuStore
to the example 8×H100 NVLink profile so `plan_placement` can D2D or
dispatch. `--expert-bytes N` sets the simulated page size (default
4096). Default `--expert-sim` stays 1×H100. Store metrics print
`migrates=` / `dispatches=`. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — Engine plan_placement (move vs dispatch)

After each GEMM, Engine sticky-pins last-used ∪ Markov experts and then
asks `plan_placement` whether those pins should D2D onto GPU0
(`MoveWeights`) or stay on the striped home (`DispatchActivations`).
The crossover is `DECODE_ACTIVATION_BYTES * fan_in * reuse` on the
GPU0↔GPU1 hop. `reuse` is how many GEMMs have keep-hot'd that key so
far (online, no JSONL future leak). `fan_in` is the last router event's
expert count. `SimulatedGpuStore::migrate` stays unconditional.
`StoreMetrics::dispatches` counts the leave-in-place choice. Writer-tiny
Qwen3MoE greedy ids still match the blob Engine. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — batched router GEMM + Engine pin_hot

`ffn_gate_inp` is one GEMM of every token in the layer, then per-row
softmax (Llama4: sigmoid top-k on the GEMM rows). Softmax MoE walks and
Llama4 stay bit-equal to the serial GEMV router.

`CachedStore::pin_hot` is a sticky pin, not an in-flight lease: decode
`release` cannot drop keep-hot. Engine parks last-used ∪ Markov keys
after each GEMM, at most `slots - 1` (`slots == 1` pins nothing so a
tight cache can still demand-page). SimulatedGpuStore `pin_hot` still
NVLink-replicates. `StoreMetrics::pins` counts sticky inserts.
After the router GEMM, unique experts that fit in `slots` prefetch
before the grouped expert GEMM (H2D can start before compute on
SimulatedGpuStore). A multi-GPU SimulatedGpuStore then
`plan_placement`s pinned experts (`StoreMetrics::migrates` /
`dispatches`). Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — grouped expert GEMM

Routed experts in a multi-token forward gather tokens that selected the
same expert and run one gate/up/down GEMM (one store acquire per expert
per layer). Softmax MoE walks and Llama4 `weight_before_ffn` logits
bit-match the serial GEMV path. Engine DirectStore hits stay below one
acquire per token-expert. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — Engine batched MoE traces

`Llama::prefill_batch` / `forward_batch` record `ExpertAccess` with
per-row sequence, token, and prefix hash so two paged sequences GEMM
together while tracing. Events scatter back onto the cache that owns
the sequence (an untraced first cache does not keep the GEMM-local
log). `Engine::enable_moe_trace` / `take_moe_trace` banks traces on
retire and `take`. `gguf_gemv engine --trace-out FILE` writes JSONL for
every sequence. Writer-tiny Qwen3MoE events bit-match sequential
traced prefill/forward and dense `greedy_generate_cache`. `gemm_peak`
stays on the batched path. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — Engine ExpertStore on batched GEMM

`Engine::attach_expert_store` parks one `LiveStore` on the first cache of
each `prefill_batch` / `forward_batch` so MoE serving stays on the
shared-pool GEMM (not a sequential fallback). DirectStore and CachedStore
greedy ids match the blob Engine on writer-tiny Qwen3MoE. Two per-cache
stores still fall back sequential. `gguf_gemv engine --expert-slots N`
(`0` = DirectStore, `N>0` = CachedStore) prints `StoreMetrics::line()`.
Markov prefetch state is parked with the store so a tight CachedStore
can prefetch across GEMMs (copy-forward ∪ lookback-2).
`Engine::expert_store_score` / `gguf_gemv engine --expert-sim` runs the
same scheduler through SimulatedGpuStore and prints the gpu-sim score
line (`wall_ns`, HBM, bytes, `energy_uj`). Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — Engine GEMM stats

`EngineStats` counts scheduler steps, tokens that ran in a
cross-sequence GEMM, the peak GEMM width, and one-at-a-time fallback
tokens. `gguf_gemv engine` prints `gemm_tokens` / `gemm_peak` /
`serial_tokens` next to intern hits. Not wall-clock tok/s. Dual score
still has no `$/M tokens`.

## Shipped 2026-08-29 — mixed prefill+replay GEMM

`Engine` packs ready prefill chunks and unforwarded replay tokens into
one `Llama::prefill_batch` when two or more jobs share the intern pool
(ragged lengths included). New greedy tokens still sample and
`forward_batch` afterward. Greedy ids still match
`greedy_generate_cache` on writer tinies, including recompute
preemption. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — batched prefill GEMM

`Llama::prefill_batch` runs Q/K/V, FFN, and lm_head as one GEMM of all
tokens in the step when the caches share a `PagedKvPool`, including
ragged chunk lengths. Attention stays per sequence. `Engine` prefills
that are ready together use this path after per-sequence intern bind.
Logits bit-match sequential `prefill` on writer tinies. Dual score
still has no `$/M tokens`.

## Shipped 2026-08-29 — batched decode GEMM

`Llama::forward_batch` runs Q/K/V, FFN, and lm_head as one GEMM of N
decode tokens when the caches share a `PagedKvPool`. Attention stays
per sequence (its `n_past` and block table). `Engine` decode uses this
path for replay and sampled tokens. Logits bit-match sequential
`forward` on writer tinies. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — Engine waiting queue

`Engine::add` beyond `max_seqs` parks the prompt. Each `step` retires
finished slots (KV Drop frees unique blocks) and admits waiters. Greedy
ids still match `greedy_generate_cache`. `gguf_gemv engine --max-seqs N`
may list more `--prompt`s than N. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — paged-KV Drop and `gguf_gemv engine`

`KvPages` Drop rewinds the block table so unique refs return to the
intern pool (`Engine::take` can admit the next sequence on a tight cap).
Intern pins stay. `gguf_gemv engine` runs several `--prompt`s through
`Engine` (chunked prefill, intern hits, recompute preemption) and prints
`intern_hits` / `preempts`. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — Engine recompute preemption

When `PagedKvPool` alloc returns `kv page cap`, `Engine` drops unique KV
for the live sequence with the most tokens (`KvCache::preempt` / rewind
to 0). Interned prefixes stay pinned (`refs` may remain 1) so intern
eviction can reclaim them. The victim re-prefills and **replays**
already sampled greedy ids; it does not resample. Greedy output still
matches `greedy_generate_cache`. A single sequence that cannot fit is a
hard `kv page cap` error. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — continuous batching on paged KV

`Engine` admits several sequences onto one `PagedKvPool`. Each `step`
prefills at most `prefill_chunk` tokens (chunked prefill) then samples one
greedy token per ready sequence. A sequence may `add` while another is
already decoding. Greedy ids match `greedy_generate_cache` on writer
tinies. Intern hits count across sequences. Distinct from `expertvm schedule`
(trace JSONL, not decode KV). Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — shared paged-KV intern pool

`PagedKvPool` is a cloneable interned-block arena. Two `KvCache`s built with
`Llama::new_paged_cache_on` / `Model::session_on_pool` intern-hit each
other's completed prefixes (vLLM-style prefix cache across sequences).
`Llama::new_paged_cache` still owns a private pool. Logits bit-match dense.
Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — paged KV on the decode engine

`Llama::new_paged_cache` / `Model::session_paged` store K/V in fixed-size
blocks with a sequence block table. Completed blocks are interned by
`expertvm::prefix_hash` of the token prefix so a later prompt on the same
cache can hit them after a rewind (vLLM Automatic Prefix Caching + block
intern). Writing a block with `refs > 1` copy-on-writes. Default
`Llama::new_cache` stays dense. Logits and greedy text bit-match dense on
writer tinies (llama / qwen3moe / llama4). `gguf_gemv serve|chat --kv-page N`
opts in; serve JSON includes `page_hits`. Distinct from `expertvm kv`
(simulated VMM pages, not decode blocks). Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — prefix KV reuse on the decode engine

`KvCache::reuse_prefix` rewinds `n_past` to the longest common prefix of
the new token ids and the ids already in KV. `Llama::prompt` prefills
only the suffix (a full hit recomputes the last prompt token so logits
are of that token). `forward` / `prefill` still append. `Session::prompt`,
`gguf_gemv serve`, and `gguf_gemv chat` keep one cache across requests /
turns. Serve JSON includes `prefix_hit`. Greedy text and logits match a
cold prefill. Distinct from JSONL `"p"` / `expertvm schedule --prefix-cache`
(that skips expert GEMMs on a hash, not KV). Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — timing copy events and cudaEventQuery on the store

`GpuStoreCfg::timing_events` creates timing-on copy start/end events and
sums `event_elapsed_ns` (`cudaEventElapsedTime`) after
`cudaEventSynchronize`. Default stays `cudaEventDisableTiming`. Phase and
copy waits use `query_event` (`cudaEventQuery`); capture uses
`query_stream` (`cudaStreamQuery`). `expertvm store --timing-events` opts
in. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — cudaGraphClone before instantiate

`GpuStoreCfg::graph_clone` / `SimCfg::graph_clone` clones a leaf capture
(`graph_clone_ns`), destroys the src, then instantiates the copy so the
graph and exec are distinct ids. Parent combo graphs still instantiate
in place. `expertvm sim|schedule|store --graph-clone` opts in; decode
identity stays instantiate-in-place. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — cudaGraphExecUpdate on store and walker

`GpuStoreCfg::graph_update` / `SimCfg::graph_update` parks a captured
leaf GEMM on evict and `Sim::update_graph`s the next miss on that GPU
(or `(device, stream)` in `sim_replay`). Pays `graph_update_ns` instead
of instantiate. Parent combo graphs still destroy (child ids are
topology). `expertvm sim|schedule|store --graph-update` opts in; decode
identity stays destroy+instantiate. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — seq-stream CUDA priority

`Sim::set_created_streams_priority` is `cudaStreamCreateWithPriority`
for streams `1 .. n` (priority = id). `SimCfg::stream_priority` /
`GpuStoreCfg::stream_priority` and `expertvm --stream-priority` opt in
so a later sequence wins when compute contends. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — AccessedBy and legacy NULL on the store seam

`GpuStoreCfg::accessed_by` / `SimCfg::accessed_by` is
`cudaMemAdviseSetAccessedBy` on every GPU at a managed fill, or
`cuMemSetAccess` PROT_READ at a VMM fill. Expert GEMMs
are reads-only, so migrate retargets compute without dest prefetch /
VMM map+D2D or `drop_managed_copy` (home residency, dest HBM 0). Pin /
`--place replicas` skip the dest copy. `legacy_null` is `set_legacy_null_stream` (copy NULL
serializes with compute). `SimCfg::pageable` is the walker H2D path.
`expertvm sim|schedule|store` take `--accessed-by`, `--legacy-null`, and
`--pageable`. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — store_replay prefetch and pageable H2D

`store_replay_cfg` runs copy-forward / Markov / `plan_window` on
`SimulatedGpuStore` (same predictors as `sim_replay`, no JSONL future
leak except the Stay vs Fetch window). `GpuStoreCfg::pageable` is
host-sync `memcpy_host_to_device` (slower than pinned DMA). `expertvm store`
takes `--prefetch` and `--pageable`. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — `expertvm store` demand-pages SimulatedGpuStore

`store_replay` / `expertvm store` walks a JSONL trace on
`SimulatedGpuStore` (`--mapped` / `--managed` / `--vmm` plus `with_cfg`
knobs) and prints hits plus the dual score. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — SimulatedGpuStore paged VMM

`GpuStoreCfg::vmm_page` uses `va_acquire_paged` so a KV-sized physical
pays map overhead per block. Hits/misses/HBM match whole-VA `with_vmm`.
Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — SimulatedGpuStore mapped pin occupancy

`with_mapped` caps cache slots at `host_pin_bytes / expert_bytes` so a
pin budget of one expert pages one expert instead of `PinOom` on the
second. Zero fit still `PinOom`s the first mapped alloc. Mapped fill
uses a pageable staging object so construction does not steal the last
mlock. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — SimulatedGpuStore SimCfg knobs

`SimulatedGpuStore::with_cfg` takes `GpuFill` plus `GpuStoreCfg`:
`host_func` after each acquire GEMM, blocking compute vs NULL copy,
host-sync `malloc`/`memcpy_sync`/`free_sync`, and default-pool
`u64::MAX` hold. `new` / `with_managed` / `with_mapped` / `with_vmm`
keep decode identity (async, non-blocking, no callback, threshold 0).
Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — SimulatedGpuStore mapped host and VMM

`with_mapped` is `cudaHostAllocMapped` (PCIe kernel, no H2D, `hbm_peak`
0). `with_vmm` is `va_acquire` + pinned H2D; evict `va_release`s.
Migrate mapped retargets GEMM; VMM maps dest then D2D. Default `new`
stays pinned H2D. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — Managed page faults at kernel start

Live kernels prefetch managed pages when they start, after stream deps.
A waited home prefetch is visible then, so a PreferredLocation remote
read does not copy twice. Graph replay still omits implicit faults
(`NotResident` if the graph skipped prefetch). Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — SimulatedGpuStore managed miss path

`SimulatedGpuStore::with_managed` places experts with `cudaMallocManaged`
+ ReadMostly + PreferredLocation + prefetch. Default `new` stays pinned
H2D (decode identity). Pin/migrate prefetch dest copies and drop extras
with `drop_managed_copy`. Evict is `free_sync`. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — cudaGraphClone recursively clones child graphs

`Sim::clone_graph` walks child-graph nodes and clones each unique graph
(`graph_clone_ns` per id). A diamond of shared children becomes one
cloned child. Cycles fail. The original parent still names the original
child; destroying that child breaks parent launch, not a recursive clone.
Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — cudaGraphUpload after instantiate

`Sim::upload_graph` is `cudaGraphUpload` (host-sync, `graph_upload_ns`).
The exec must already be instantiated. First `launch_graph` uploads if
needed. `update_graph` clears the flag. Capture refused.
`sim_replay` / `SimulatedGpuStore` upload after instantiate so replay
launches skip the host-sync. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — Remote managed GEMM reads PreferredLocation

Expert GEMMs treat weight pages as kernel reads. `--managed --place remote`
prefetches onto home with `SetPreferredLocation` and GEMMs on GPU0 over
the interconnect (no dest HBM copy, no weight D2D). Dual score still has
no `$/M tokens`.

## Shipped 2026-08-29 — PreferredLocation and grouped child graphs

`MemAdvise::SetPreferredLocation` keeps a managed page already at that
GPU there on a remote kernel read (interconnect, not dest HBM). Writes
still migrate. Host preferred does not skip first-touch. `--cuda-graphs`
captures a leaf GEMM per expert, instantiates it, then a parent of
child-graph nodes so grouped launches reuse leaves. Dual score still
has no `$/M tokens`.

## Shipped 2026-08-29 — Managed ReadMostly replicas

`--place replicas` with `--managed` `prefetch`s hot keys onto dest GPUs
(ReadMostly keeps the home copy). Dest eviction is `Sim::drop_managed_copy`
(one GPU's HBM, allocation stays). Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — CUDA child graphs from launch during capture

`launch_graph` on a captured stream records `GpuOp::ChildGraph`. The
child must already be instantiated. Parent launch expands the nested
exec (GraphHead once). Independent streams still launch live. Destroying
the child makes parent launch unknown. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — CUDA graph capture forks on event wait

Independent streams stay live during `begin_capture`. A stream that
`wait_event`s an event recorded in this capture joins (CUDA forked
capture). `launch_graph` remaps origin-stream nodes onto the launch
stream so copy and compute overlap. Query/sync of a capturing stream
is Invalid. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — CUDA graph destroy frees the id

`Sim::destroy_graph` is `cudaGraphDestroy` / `cudaGraphExecDestroy`
(host-sync, 1 ns). Capture refused. `sim_replay` / `SimulatedGpuStore`
destroy captured GEMM graphs when a page is evicted or migrated.
Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — CUDA graph clone is an independent copy

`Sim::clone_graph` is `cudaGraphClone` (host-sync, `graph_clone_ns`).
The clone is not instantiated; instantiate/update of one id does not
change the other. Capture refused. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — cudaMemAdvise ReadMostly and AccessedBy

`Sim::mem_advise` is `cudaMemAdvise` (host-sync; capture refused).
`SetReadMostly`: prefetch onto a second GPU keeps the first copy; a
kernel write invalidates extras. `SetAccessedBy`: a kernel may read
without migrating (interconnect, not local HBM); writes still migrate.
`--managed` sets ReadMostly on expert pages (weights are read-only).
Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — CUDA graph instantiate and exec update

`Sim::instantiate_graph` is `cudaGraphInstantiate` (host-sync,
`graph_instantiate_ns`). First `launch_graph` instantiates if needed;
later launches skip it. `Sim::update_graph` is `cudaGraphExecUpdate`:
same device sequence and op kinds, different KernelBuf / memcpy sizes.
Topology mismatch / uninstantiated exec / same id / capture are
`Invalid`. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — memset of a mapped VMM span (new KV block)

`Sim::memset` fills `[0, bytes)`. `Sim::memset_buf` names an interior
page. A hole is `NotResident`; mapped host is not a memset dest.
`expertvm kv --fill memset` zeros a miss in HBM (no PCIe) so `bytes_moved=0`
and wall is shorter than `--fill h2d`. Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — kernel/copy of a mapped VMM span (paged KV)

`Sim::kernel` still needs the whole VA covered. `Sim::kernel_bufs` /
`KernelBuf` and `MemcpyOp::offset` touch `[offset, offset+bytes)` so a
reserved KV pointer can keep only the working-set pages mapped.
`is_range_resident` is that span check. `expertvm kv` LRU-pages a VA;
`hbm_peak` is `slots * page_bytes`, not the reservation. Dual score still
has no `$/M tokens`.

## Shipped 2026-08-29 — paged VMM maps experts in KV-block physicals

`Sim::va_acquire_paged` reserves (or remaps an idle VA) then `va_map_range`s
`page`-byte spans that cover the pointer. Hits/misses match a single
`va_map`; wall pays `alloc_overhead_ns` per block. `expertvm sim --vmm-page N`
implies `--vmm`. `expertvm bench` `sim-vmmpage`. Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — cudaEventDisableTiming forbids elapsed

`Sim::create_event` / `create_event_disable_timing` are `cudaEventCreate`
and `cudaEventCreateWithFlags(..., cudaEventDisableTiming)`. Timing-off
events still record, wait, and query; `event_elapsed_ns` is `Invalid`.
Capture refuses create. Implicit create on first record stays
timing-on. `SimulatedGpuStore` and remote D2D waits use disable-timing
sync events (vLLM-style). Dual score still has no `$/M tokens`.

## Shipped 2026-08-29 — mapped occupancy respects the pin budget

`--mapped` walker slots are `min(capacity, host_pin_bytes / expert_bytes)`
so a pin cap of one expert pages one expert instead of `PinOom` on the
second. Zero fit (pin smaller than one expert) is still `PinOom`. Dual
score still has no `$/M tokens`.

## Shipped 2026-08-29 — cudaStreamCreate serializes with the default stream

`Sim::set_stream_blocking` / `set_created_streams_blocking` model
`cudaStreamCreate` (blocking) vs `cudaStreamCreateWithFlags(...,
cudaStreamNonBlocking)`. Blocking streams wait on / for
`StreamId::NULL`. The legacy default stream still serializes with
*every* stream. Created streams default to non-blocking so
`--seq-streams` can overlap. `expertvm sim --seq-streams --blocking-streams`
/ `expertvm bench` `sim-blockstrm`. `SimulatedGpuStore` stays
non-blocking. Dual score still has no `$/M tokens`.

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
replay now caps occupancy at `pin / expert_bytes` (see the mapped
occupancy note). `SimulatedGpuStore::with_mapped` uses the same cap.
Dual score still has no `$/M tokens`.

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
share PCIe. Capture refuses pageable copies. `GpuStoreCfg::pageable` /
`expertvm store --pageable` uses that path; `SimulatedGpuStore::new`
stays pinned. Dual score still has no `$/M tokens`.

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
replays them with `launch_graph`; first launch instantiates
(`graph_instantiate_ns`), then each launch pays `graph_launch_ns` once
instead of per-kernel `launch_overhead_ns`. `SimulatedGpuStore` does the same
per resident page after a drain (completed copy event, idle compute stream).
Replay lines report `prefetch_hits` / `prefetch_waste` / `graph_launches`.
gpu-sim also models `memset`, directed `enable_peer` / `disable_peer`, and the
legacy CUDA null stream (`set_legacy_null_stream`). Dual score still has no
`$/M tokens`.

## Shipped 2026-08-29 — real-model sidecar JSON drives the llama.cpp check

`tests/reference/qwen2.5-0.5b-instruct-q4_k_m.json` is the source of tokens,
greedy ids, and the max-logit band (not hardcoded). The NORM Llama control is
`llama-3.2-1b-instruct-q4_k_m.json`. A set `LLAMA_RUST_REAL_MODEL_DIR` must
contain both GGUFs (fail-loud). Tests still do not download Hugging Face in CI. Dual score still
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
Adversarial suite is fourteen named workloads (coding/chat/long-context,
prefill-heavy, decode-heavy, batch-1 / batch-8 / batch-128, prefill-batch)
plus the original four. gpu-sim
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
  onto `(home + 1) % n_gpus` when the profile has `n_gpus >= 2`.

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
- Decode: RMSNorm, RoPE, GQA+KV, SwiGLU (llama/qwen2/mistral/phi3/qwen3/llama4/qwen2moe/qwen3moe/qwen2vl/qwen3vl/qwen3next/qwen35) or Gemma GeGLU or official Phi2 sequential GELU or official Bloom sequential GELU, lm_head, greedy sample by default. Gemma scales token embeds by `sqrt(n_embd)` (`ggml_scale`). Official Qwen3 applies per-head QK-Norm (`attn_q_norm` / `attn_k_norm` RMSNorm on Q and K after projection, before RoPE). Official Qwen3MoE applies the same QK-Norm plus `build_moe_ffn` softmax then top-k with `norm_w` clamp `2^-14` and no shared expert. Official Llama4 text applies iRoPE/NoPE, unweighted QK-Norm after RoPE, and expert FFN on MoE layers. Official llama MoE (`architecture=llama` with `n_expert>0`) applies `build_moe_ffn` softmax then top-k, SwiGLU, weights after the expert with `norm_w` clamp `2^-14`. Official Qwen2MoE (`architecture=qwen2moe`) applies softmax then top-k without `norm_w`, SwiGLU experts, and a shared expert gated by `silu(x)/x` on `ffn_gate_inp_shexp`. Official Qwen2VL applies the Qwen2 language walk plus m-RoPE (`ggml_rope_multi` / `LLAMA_ROPE_TYPE_MROPE`, `rope.dimension_sections`, text `n_pos_per_embd=4` with `[t,h,w,e]=[p,p,p,0]`). Official Qwen3VL applies Qwen3 QK-Norm plus interleaved m-RoPE (`ggml_rope_multi` / `LLAMA_ROPE_TYPE_IMROPE`, required `rope.dimension_sections`, text `n_pos_per_embd=4` with `[t,h,w,e]=[p,p,p,0]`). Official Qwen3Next applies gated full attention (joint Q+gate, QK-Norm, sigmoid after attn), official `post_attention_norm`, and `build_moe_ffn` softmax then top-k with `norm_w` clamp `2^-14` plus a shared expert gated by sigmoid. Official Qwen35 applies gated full attention (joint Q+gate, QK-Norm, IMROPE, sigmoid after attn), official `post_attention_norm`, and dense SwiGLU. Official Phi2 applies LayerNorm, NEOX RoPE, Q-scale then attn scale `1.0`, and parallel GELU-seq FFN. Official Bloom applies `token_embd_norm` LayerNorm, fused `attn_qkv`, ALiBi (hardcoded `f_max_alibi_bias = 8`, no RoPE), sequential LayerNorm GELU-seq FFN. Official Gemma2 applies post-norms, SWA, tanh logit softcap. Official Gemma3 applies Gemma2 post-norms + GeGLU, QK-Norm before RoPE, no attn softcap, optional final tanh softcap, SWA default period 6. Official Gemma3n applies Gemma3 plus attention scale `1.0`, unweighted V RMSNorm, AltUp, Laurel, per-layer inputs, gaussian_topk, SWA period 5, and final softcap 30. Official Gemma4 applies Gemma3 QK-Norm + post-norms + GeGLU, attention scale `1.0`, unweighted V RMSNorm, convert bool-array SWA; dense layers stay GeGLU; MoE layers add a shared dense MLP plus routed GELU experts (custom router on `attn_out`). Writer-tiny Gemma4 is dense (`tiny-gemma4`), MoE (`tiny-gemma4-moe`), PLE (`tiny-gemma4-ple`), MoE plus PLE (`tiny-gemma4-moe-ple`), fused experts (`tiny-gemma4-moe-fused`), fused plus PLE (`tiny-gemma4-moe-fused-ple`), or shared KV (`tiny-gemma4-shared-kv`) or mixed SWA/global head dims (`tiny-gemma4-mixed-hd`) or SWA RoPE base (`tiny-gemma4-swa-base`) or omitted `wv` (`tiny-gemma4-no-wv`) or layer output scale (`tiny-gemma4-out-scale`) or proportional RoPE (`tiny-gemma4-rope-freqs`) or per-expert down scale (`tiny-gemma4-moe-down-s`) or per-expert gate/up scale (`tiny-gemma4-moe-gate-up-s`) on the same arch. `k_swa != v_swa` still refused with named keys. Linear-attn / gated-delta layers are refused. RMSNorm +1 is a convert-hf bake on GGUF `norm.weight` bytes; decode uses `LLM_NORM_RMS` as-is.
- **Gemma architecture.** `general.architecture=gemma` with `gemma.*` KV (same prefix pattern as mistral/phi3/qwen2). Same `blk.{i}.*` tensor names as llama. Writer-built `tiny-gemma` loads and greedy-generates. Load/GEMV/GEMM/embed logits match an independent scalar of the official llama.cpp Gemma walk (embed scale + `LLM_FFN_GELU`/`ggml_gelu` tanh approx). Not `gemma2`. Not a dtype. No tok/s.
- **Gemma2 architecture.** `general.architecture=gemma2` (gguf-py `MODEL_ARCH_NAMES[GEMMA2] = "gemma2"`, llama.cpp `LLM_ARCH_GEMMA2 = "gemma2"`, `src/models/gemma2.cpp`) with `gemma2.*` KV. Official convert writes `general.architecture=gemma2` (`Gemma2ForCausalLM` → `"gemma2"`). Official named differences vs gemma, measured from `src/models/gemma2.cpp` plus `conversion/gemma.py` `Gemma2Model`: RMSNorm `blk.{i}.post_attention_norm` on attn out before residual; `blk.{i}.ffn_norm` before GeGLU; RMSNorm `blk.{i}.post_ffw_norm` on FFN out before residual; sliding-window attention (`set_swa_pattern(2)`, `LLAMA_SWA_TYPE_STANDARD`, `is_masked_swa`: `p1 - p0 >= n_swa`); attn/final tanh logit softcap (`cap * tanh(x/cap)`). Convert skips `lm_head.weight` (tied), omits `rope.dimension_count` / `sliding_window_pattern`, writes `attn_logit_softcapping` / `final_logit_softcapping` / `attention.sliding_window` / `attention.key_length` / `attention.value_length` / `context_length`. Writer-tiny uses `attention.sliding_window = 2` so short-seq tests clip. Convert-hf bakes `norm.weight + 1`; runtime does not add 1 again. `{arch}.context_length` is still unused for KV sizing. Writer-built `tiny-gemma2` loads and greedy-generates. Load/GEMV/GEMM/embed logits match an independent scalar of the official Gemma2 walk on those GGUF bytes. `mixtral` stays rejected as unknown architecture. Not a dtype. No tok/s.
- **Gemma3 architecture.** `general.architecture=gemma3` (gguf-py `MODEL_ARCH_NAMES[GEMMA3] = "gemma3"`, llama.cpp `LLM_ARCH_GEMMA3 = "gemma3"`, `src/models/gemma3.cpp`) with `gemma3.*` KV. Official convert writes `general.architecture=gemma3` (`Gemma3ForCausalLM` → `"gemma3"`). Official named differences vs gemma2, measured from `src/models/gemma3.cpp` plus `conversion/gemma.py` `Gemma3Model`: same Gemma embed-scale + GeGLU + RMSNorm `post_attention_norm` / `ffn_norm` / `post_ffw_norm`; QK-Norm (`blk.{i}.attn_q_norm` / `attn_k_norm`, `LLM_NORM_RMS` on Q and K after projection / before RoPE); no attn logit softcap (convert asserts `attn_logit_softcapping` is None); optional final tanh logit softcap (`cap * tanh(x/cap)`, default 0); SWA only when `attention.sliding_window` is present and `> 0` (`set_swa_pattern` default period 6, `LLAMA_SWA_TYPE_STANDARD`). Convert skips `lm_head.weight` when tied, omits `attn_logit_softcapping` / `rope.dimension_count` / `sliding_window_pattern`, writes `attention.sliding_window` when `sliding_window_pattern != 1`, plus `attention.key_length` / `value_length` / `context_length`. Writer-tiny uses `attention.sliding_window = 2` so short-seq tests clip and omits final softcap (default 0). Convert-hf bakes `norm.weight + 1`; runtime does not add 1 again. `{arch}.context_length` is still unused for KV sizing. Writer-built `tiny-gemma3` loads and greedy-generates. Load/GEMV/GEMM/embed logits match an independent scalar of the official Gemma3 walk on those GGUF bytes. `mixtral` stays rejected as unknown architecture. Not a dtype. No tok/s.
- **Gemma3n architecture.** `general.architecture=gemma3n` (gguf-py `MODEL_ARCH_NAMES[GEMMA3N] = "gemma3n"`, llama.cpp `LLM_ARCH_GEMMA3N = "gemma3n"`, `src/models/gemma3n.cpp`) with `gemma3n.*` KV. Official convert writes `general.architecture=gemma3n` (`Gemma3nForCausalLM` → `"gemma3n"`). Official named differences vs gemma3, measured from `src/models/gemma3n.cpp` plus `conversion/gemma.py` `Gemma3NModel`: Gemma embed-scale + GeGLU + QK-Norm + RMSNorm `post_attention_norm` / `ffn_norm` / `post_ffw_norm`; unweighted RMSNorm on V (`ggml_rms_norm` over `n_embd_head` before KV store); attention scale `1.0` (not `1/sqrt(head)`); AltUp (4 residual streams, `altup_proj` / `altup_unembd_proj`, predict/correct mix, per-layer inject); Laurel (`laurel_l` / `laurel_r` / `laurel_post_norm`); per-layer token embeddings (`per_layer_token_embd` / `per_layer_model_proj`); gaussian_topk on the first 10 layers (`n_layer_sparsity` hardcoded, Bessel `n_ff-1`); SWA required (`set_swa_pattern` default period 5); final tanh logit softcap (default 30); no attn logit softcap. Convert skips `lm_head.weight` when tied, omits `attn_logit_softcapping` / `rope.dimension_count` / `sliding_window_pattern`, writes `gemma3n.altup.*` / `embedding_length_per_layer_input` / `attention.shared_kv_layers`. Writer-tiny uses `attention.sliding_window = 2` so short-seq tests clip. Convert `norm_shift` is 0. `{arch}.context_length` is still unused for KV sizing. `n_layer_kv_from_start = 20` is written but unused on the 1-layer tiny (full KV branch only). Writer-built `tiny-gemma3n` loads and greedy-generates. Load/GEMV/GEMM/embed logits match an independent scalar of the official Gemma3n walk on those GGUF bytes. `mixtral` stays rejected as unknown architecture. Not a dtype. No tok/s.
- **Gemma4 architecture.** `general.architecture=gemma4` (gguf-py `MODEL_ARCH_NAMES[GEMMA4] = "gemma4"`, llama.cpp `LLM_ARCH_GEMMA4 = "gemma4"`, `src/models/gemma4.cpp`) with `gemma4.*` KV. Official convert writes `general.architecture=gemma4` (`Gemma4ForCausalLM` → `"gemma4"`). Official named differences vs gemma3 / gemma3n, measured from `src/models/gemma4.cpp`: Gemma embed-scale + GeGLU + QK-Norm (`blk.{i}.attn_q_norm` / `attn_k_norm` after projection / before RoPE) + RMSNorm `post_attention_norm` / `ffn_norm` / `post_ffw_norm`; unweighted RMSNorm on V (`ggml_rms_norm` over `n_embd_head` before KV store); attention scale `1.0` (not `1/sqrt(head)`); required `attention.sliding_window`; convert `attention.sliding_window_pattern` as a per-layer bool array (`get_key_or_arr`, not gemma2/3/3n `set_swa_pattern` period); required `attention.key_length_swa` / `value_length_swa` and `embedding_length_per_layer_input`; optional final tanh logit softcap (default 0); no attn logit softcap; no AltUp / Laurel / gaussian_topk. Convert skips `lm_head.weight` when tied, omits `attn_logit_softcapping` / `rope.dimension_count`. Writer-tiny dense has no `ffn_gate_inp`. Writer-tiny MoE (`tiny-gemma4-moe`) is the same `architecture=gemma4` with `expert_count` / `expert_used_count` / `expert_feed_forward_length`: shared dense GeGLU plus `ffn_post_norm_1`, expert `ffn_pre_norm_2`, custom router on `attn_out` (unweighted RMSNorm, `1/sqrt(n_embd)`, `ffn_gate_inp.scale`, then `ffn_gate_inp`), `build_moe_ffn` softmax then top-k with `norm_w` clamp `2^-14` and GELU experts, then `ffn_post_norm_2` and add. Writer-tiny fused (`tiny-gemma4-moe-fused`) packs `ffn_gate_up_exps` (twice `n_ff` rows, gate then up) instead of separate gate/up; DirectStore catalogs fused bytes in `ExpertParts.gate` with empty `up`. Writer-tiny shared KV (`tiny-gemma4-shared-kv`) is three dense all-SWA layers with `shared_kv_layers=1` (layer 2 reuses layer 0). Writer-tiny mixed SWA/global head dims (`tiny-gemma4-mixed-hd`) is two dense layers, SWA then global, `key_length_swa` half of `key_length`. Writer-tiny SWA RoPE base (`tiny-gemma4-swa-base`) is those mixed-hd tensors with `rope.freq_base` vs `rope.freq_base_swa` (`1000000` vs `10000`); SWA uses `freq_base_swa` (default `10000` when omitted), global uses `freq_base`. Mixed-hd global QK-Norm is `0.1` so attn-scale `1.0` does not saturate softmax. Writer-tiny omitted `wv` (`tiny-gemma4-no-wv`) is dense `tiny-gemma4` without `attn_v`; decode uses K as V before the unweighted V RMSNorm. Writer-tiny layer output scale (`tiny-gemma4-out-scale`) is dense `tiny-gemma4` plus `blk.0.layer_output_scale` at `0.5`; `ggml_mul` broadcasts after the FFN residual. Writer-tiny proportional RoPE (`tiny-gemma4-rope-freqs`) is mixed-hd plus `rope_freqs.weight` with convert packing (`1` then `1e30` for unrotated NeoX pairs); SWA keeps `freq_factors` null, global uses `theta/ff`. Writer-tiny per-expert down scale (`tiny-gemma4-moe-down-s`) is split `tiny-gemma4-moe` plus `blk.0.ffn_down_exps.scale` (`{n_expert}`); `ggml_mul` after the down expert GEMM. Writer-tiny per-expert gate/up scale (`tiny-gemma4-moe-gate-up-s`) is split `tiny-gemma4-moe` plus `blk.0.ffn_gate_exps.scale` / `ffn_up_exps.scale` (`{n_expert}`); `ggml_mul` after the gate/up expert GEMM (fused applies `up_exps_s` to both halves and ignores `gate_exps_s`). `k_swa != v_swa` still refused with named keys. Writer-tiny PLE (`tiny-gemma4-ple`) is the same `architecture=gemma4` with `embedding_length_per_layer_input > 0`: `per_layer_token_embd` / `per_layer_model_proj` / `per_layer_proj_norm` plus per-layer `inp_gate` / `proj` / `post_norm`; inject after the FFN residual; no AltUp / Laurel. Writer-tiny MoE+PLE (`tiny-gemma4-moe-ple`) is the production E2B/E4B shape on the same arch (`ffn_gate_inp` and `n_embd_per_layer > 0`). Writer-tiny fused plus PLE (`tiny-gemma4-moe-fused-ple`) is the production packing (fused `ffn_gate_up_exps` and `n_embd_per_layer > 0`). Writer-built `tiny-gemma4`, `tiny-gemma4-moe`, `tiny-gemma4-ple`, `tiny-gemma4-moe-ple`, `tiny-gemma4-moe-fused`, `tiny-gemma4-moe-fused-ple`, `tiny-gemma4-shared-kv`, `tiny-gemma4-mixed-hd`, `tiny-gemma4-swa-base`, `tiny-gemma4-no-wv`, `tiny-gemma4-out-scale`, `tiny-gemma4-rope-freqs`, and `tiny-gemma4-moe-down-s`, `tiny-gemma4-moe-gate-up-s` load and greedy-generate. Load/GEMV/GEMM/embed logits match an independent scalar of the official Gemma4 dense, MoE, PLE, MoE+PLE, fused-expert, fused plus PLE, shared-KV, mixed-hd, SWA-base, omitted-`wv`, out-scale, proportional-`rope_freqs`, or per-expert-`ffn_down_exps.scale` / `ffn_gate_exps.scale` / `ffn_up_exps.scale` walk on those GGUF bytes. `k_swa != v_swa` still refused with named keys. `mixtral` stays rejected as unknown architecture. Not a dtype. No tok/s. Do not invent a second gemma4.
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
- **Bloom architecture.** `general.architecture=bloom` (gguf-py `MODEL_ARCH_NAMES[BLOOM] = "bloom"`, llama.cpp `LLM_ARCH_BLOOM = "bloom"`, `src/models/bloom.cpp`) with `bloom.*` KV. Official convert writes `general.architecture=bloom` (`BloomForCausalLM` / `BloomModel` → `"bloom"`). Official named differences vs llama / phi2 / gemma, measured from `src/models/bloom.cpp` plus `conversion/bloom.py` / `ggml_soft_max_ext`: `token_embd_norm` LayerNorm; fused `attn_qkv` (convert restacks HF interleaved QKV to concatenated Q/K/V); `LLM_NORM` on attn/ffn/output (including `ffn_norm.bias`); sequential residual; `LLM_FFN_GELU`/`LLM_FFN_SEQ` with biases; ALiBi (`hparams.f_max_alibi_bias = 8.0f` hardcoded, not a GGUF KV; no RoPE). Official convert writes `attention.layer_norm_epsilon` (not rms), `feed_forward_length = 4 * n_embed`, `head_count_kv = n_head`, `context_length` (`seq_length` else `n_embed`), and `add_bos_token=false`. No `rope.dimension_count` / `rope.freq_base`. No `output.bias`. Writer-built `tiny-bloom` uses that convert shape (`n_ff=1024`, fused `attn_qkv`, `context_length=n_embed`). Tied `output.weight` reuse is allowed. `{arch}.context_length` is still unused for KV sizing. `qwen3vlmoe` stays rejected. Not Mixtral, not a phi2 redo, not linear-attn, no invented extra norms. Writer-built `tiny-bloom` loads and greedy-generates. Load/GEMV/GEMM/embed logits match an independent scalar of the official Bloom language-model walk on those GGUF bytes. `mixtral` stays rejected as unknown architecture. Not a dtype. No tok/s.
- **Sampling.** Seedless greedy (`temperature <= 0`, argmax, first index on ties) is still the `infer` / `greedy_generate` path. `SampleParams` + `generate` add temperature, top-k, top-p, and unique-id repeat penalty (`logit > 0` then `/=`, else `*=`). Stochastic draws use SplitMix64 and require a seed. No CLI sampling flags.
- Prefill GEMM. Prompt tokens are one causal pass. A single token stays GEMV.
- Architectures: `llama` (dense `n_expert==0`, or official llama MoE when `n_expert>0`), `qwen2`, `mistral`, `phi3`, `gemma`, `gemma2`, `gemma3`, `gemma3n`, `gemma4`, `qwen3`, `llama4`, `qwen2moe`, `qwen3moe`, `qwen2vl`, `qwen3vl`, `qwen3next`, `qwen35`, `phi2`, `bloom` `{arch}.*` KV. `mixtral` stays rejected. Official Mixtral GGUF is `architecture=llama` with `n_expert>0`, not a `mixtral` arch. `qwen3vlmoe` stays rejected. A separate `qwen25vl` language arch stays rejected (official Qwen2.5-VL text is `architecture=qwen2vl`). Later official families still in `LLM_ARCH_NAMES` stay rejected.
- Q4_K_M shape that common OSS files actually have:
  - quantized `token_embd.weight` (Q1_0 / Q2_0 / TQ1_0 / TQ2_0 / Q2_K / Q3_K / Q4_1 / Q4_K / Q5_0 / Q5_1 / Q5_K / Q6_K / Q8_1 / IQ1_M / IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ4_NL / IQ4_XS / MXFP4 / NVFP4 / F32) or F16 / BF16
  - missing `{arch}.rope.dimension_count` derived from `embedding_length / head_count`
  - optional F32 or F16 `attn_{q,k,v}.bias`
  - `tokenizer.ggml.add_bos_token=false` honored
- Load/decode errors name tensor, ggml type id, and/or KV key. ggml-removed type ids are named as removed.
- CLI: `gguf_gemv infer <path> [--prompt TEXT] [--n-predict N] [--n-ctx N]`. Seedless greedy. Defaults remain `ab` / 2 so the shipped two-run command still works.
- **MoE traces.** `gguf_gemv trace <path> --out FILE [--capacity N]`. Same greedy as `infer`, writes JSONL, prints the measured expertvm hit-rate table. Identity vs untraced greedy.
- **Serving.** Local `gguf_gemv serve <path> [--n-predict N] [--n-ctx N] [--bind HOST:PORT] [--model-id ID] [--engine] [--max-seqs N] [--expert-slots N] [--expert-sim]`. Std `TcpListener` on `127.0.0.1` (default `:8080`; `localhost` allowed). Default: one HTTP/1.1 request at a time, `POST /generate` JSON `{"prompt"}` optional `n_predict` → `{"generated"}`, persistent KV prefix reuse, HTTP/1.1 keep-alive unless `Connection: close`. `POST /v1/completions` and `POST /v1/chat/completions` use an OpenAI `choices` envelope. `POST /tokenize` / `POST /detokenize` encode and decode without generating. `GET /v1/models` lists `--model-id` (GGUF stem by default). `GET /health` / `GET /metrics` are probes. `--engine` admits concurrent requests onto one `Engine` so they GEMM together. `--expert-slots` / `--expert-sim` park DirectStore / CachedStore / SimulatedGpuStore. Seedless greedy. Missing file and empty prompt fail cleanly. No tok/s. Not a production inference server. Kernel Integrity has not signed it.
- One file blob. `load_gguf_owned(Vec<u8>)` keeps the file bytes. Tensor payloads are ranges of that blob. mmap is still forbidden (`unsafe` or a crate).
- **Tokenizer.** `token_id` / merge rank are `HashMap` lookups, not a linear scan of the vocab.
  - `tokenizer.ggml.model=gpt2` (and vocabs that contain `Ġ` / `Ċ`): UTF-8 bytes → GPT-2 bytes-to-unicode → BPE. Decode maps `Ċ` → `\n` and `Ġ` → space. The recorded Qwen `generated=abĊĊ` was two newline pieces printed raw.
  - `tokenizer.ggml.model=llama` (and vocabs with `▁` / `<0xHH>`): space → `▁`; unknown UTF-8 → `<0xHH>` when those tokens exist. Decode maps both.
  - Writer-built tiny vocabs stay character-then-merge (`encode("ab")=[3]`, `decode([1,2])="ab"`).
  - `tokenizer.chat_template` is stored on `Tokenizer` and rendered by
    `apply_chat_template` (in-tree Jinja subset, no crates.io Jinja).
    `gguf_gemv chat` / `serve` `{"messages"}` use that path.
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
2. **Dtypes / arches still rejected.** Remaining official vision families, remaining MoE families. `gemma` `{arch}.*` KV loads (writer-built tiny). `qwen3` `{arch}.*` KV loads (writer-built tiny; official QK-Norm). `llama4` `{arch}.*` KV loads (writer-built tiny; official iRoPE/NoPE, unweighted QK-Norm after RoPE, expert FFN). Official llama MoE (`architecture=llama` with `n_expert>0`) loads (writer-built tiny; official `build_moe_ffn` softmax then top-k, `norm_w` clamp `2^-14`). `qwen2moe` `{arch}.*` KV loads (writer-built tiny; official softmax then top-k without `norm_w`, shared expert gated by `silu(x)/x`). `qwen3moe` `{arch}.*` KV loads (writer-built tiny; official Qwen3 QK-Norm plus softmax then top-k with `norm_w` clamp `2^-14`, no shared expert). `qwen2vl` `{arch}.*` KV loads (writer-built tiny; official Qwen2 plus m-RoPE / `ggml_rope_multi`). `qwen3vl` `{arch}.*` KV loads (writer-built tiny; official Qwen3 QK-Norm plus interleaved m-RoPE / `LLAMA_ROPE_TYPE_IMROPE`). `qwen3next` `{arch}.*` KV loads (writer-built tiny; official gated full attention, `post_attention_norm`, MoE `norm_w` plus sigmoid-gated shared expert). `qwen35` `{arch}.*` KV loads (writer-built tiny; official gated full attention, IMROPE, `post_attention_norm`, dense SwiGLU; linear-attn / gated-delta refused). `phi2` `{arch}.*` KV loads (writer-built tiny; official LayerNorm, NEOX RoPE, Q-scale, parallel GELU-seq FFN). `bloom` `{arch}.*` KV loads (writer-built tiny; official `token_embd_norm`, fused QKV, ALiBi, sequential GELU-seq FFN). `gemma2` `{arch}.*` KV loads (writer-built tiny; official post-norms, SWA, tanh softcap). `gemma3` `{arch}.*` KV loads (writer-built tiny; official QK-Norm, post-norms, SWA period 6, no attn softcap). `gemma3n` `{arch}.*` KV loads (writer-built tiny; official AltUp, Laurel, V RMSNorm, attn scale 1.0, gaussian_topk, SWA period 5, final softcap 30). `gemma4` `{arch}.*` KV loads (writer-built tiny; official V RMSNorm, attn scale 1.0, convert bool-array SWA; dense GeGLU or MoE shared-MLP plus GELU experts; PLE when `embedding_length_per_layer_input > 0`; MoE+PLE writer-tiny; fused `ffn_gate_up_exps`; fused plus PLE writer-tiny; shared KV writer-tiny; mixed-hd writer-tiny; optional `wv` / `out_scale` / `rope_freqs` / `rope.freq_base_swa` / `ffn_down_exps.scale` / `ffn_gate_exps.scale` / `ffn_up_exps.scale` omitted). `mixtral` is still rejected (`mixtral` is not an official arch; official Mixtral GGUF is `architecture=llama` with `n_expert>0`). `qwen3vlmoe` is still rejected. 1-D F16 norms/bias load (attn/ffn/`output_norm`, official Qwen3 `attn_{q,k}_norm`, official phi2 LayerNorm/FFN/output bias, official bloom `token_embd_norm` / `ffn_norm.bias` / `attn_qkv.bias`, and optional `attn_{q,k,v}.bias`). Tied `output.weight` reuses `token_embd.weight` when absent. Common OSS IQ* 2-D, BF16 2-D, Q2_K 2-D, Q3_K 2-D, Q4_1 2-D, Q5_0 2-D, Q5_1 2-D, MXFP4 2-D, NVFP4 2-D, Q1_0 2-D, Q2_0 2-D, Q8_1 2-D, TQ1_0 2-D, and TQ2_0 2-D are loaded (IQ1_M / IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ4_NL / IQ4_XS / BF16 / Q2_K / Q3_K / Q4_1 / Q5_0 / Q5_1 / MXFP4 / NVFP4 / Q1_0 / Q2_0 / Q8_1 / TQ1_0 / TQ2_0). IQ4_NL_4_4/4_8/8_8 (36..=38) are ggml-removed, not a remaining hole. Next remaining live rejected ggml weight type id: none. Official phi2 loaded. Official bloom loaded (`token_embd_norm`, fused `attn_qkv`, ALiBi, sequential GELU-seq). Official gemma2 loaded (`post_attention_norm` / `post_ffw_norm`, SWA, tanh softcap). Official gemma3 loaded (QK-Norm, post-norms, SWA period 6, no attn softcap). Official gemma3n loaded (AltUp, Laurel, V RMSNorm, attn scale 1.0, gaussian_topk, SWA period 5, final softcap 30). Official gemma4 loaded (QK-Norm, post-norms, V RMSNorm, attn scale 1.0, convert bool-array SWA; dense GeGLU or MoE shared-MLP plus GELU experts; PLE when `embedding_length_per_layer_input > 0`; MoE+PLE writer-tiny; fused `ffn_gate_up_exps`; fused plus PLE writer-tiny; shared KV writer-tiny; mixed-hd writer-tiny; SWA-base writer-tiny; omitted-`wv` writer-tiny; out-scale writer-tiny; rope_freqs writer-tiny; down_s writer-tiny; gate/up_s writer-tiny). `mixtral` is still rejected (`mixtral` is not an official arch; official Mixtral GGUF is `architecture=llama` with `n_expert>0`). `qwen3vlmoe` is still rejected. Do not invent an arch. Do not invent a dtype. Do not list mixtral or qwen3vlmoe as accepted.
3. **KV cache** sized to prompt+predict is the default; `{arch}.context_length` is still unused. `--n-ctx` is an override only.
4. **crates.io** unpublished. Linux proof is GHA tiny/oracle tests only (2GB GGUF is gitignored).
5. **Chat template apply.** Shipped: in-tree Jinja subset, special-token
   split, Unicode BPE pre-tokenizer, `Tokenizer::apply_chat_template`,
   `gguf_gemv chat`. Remaining: not a full Jinja engine.
6. **Serving beyond local.** `gguf_gemv engine` and `gguf_gemv serve --engine`
   continuous-batch on one interned pool (HTTP `POST /generate` GEMMs
   together, HTTP/1.1 keep-alive). Default `serve` is still one HTTP request
   at a time with keep-alive. OpenAI `POST /v1/completions` /
   `POST /v1/chat/completions` plus `GET /v1/models` / `GET /health` /
   `GET /metrics`. `POST /tokenize` / `POST /detokenize` encode and decode
   without generating (`add_special_tokens` on tokenize; `echo` on
   `/v1/completions`). Not a production inference server.

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
./target/release/expertvm workload batch-1
./target/release/expertvm workload batch-128 --tokens 8
./target/release/expertvm schedule tests/traces/tiny-qwen3moe-2layer.jsonl --capacity 2 --seq-streams --compute-slots 2 --decode-sms 250
./target/release/expertvm schedule tests/traces/tiny-qwen3moe-2layer.jsonl --capacity 2 --prefill-chunk 1 --decode-priority --compute-slots 2
./target/release/expertvm schedule tests/traces/cycling.jsonl --capacity 8 --place striped --profile 8xh100 --expert-bytes 1048576
./target/release/expertvm schedule tests/traces/cycling.jsonl --capacity 8 --place replicas --profile 8xh100 --expert-bytes 1048576
./target/release/expertvm schedule tests/traces/cycling.jsonl --capacity 8 --place remote --profile 2node-rdma --expert-bytes 1048576 --prefetch copy-forward
./target/release/expertvm workload prefill-batch
./target/release/gpu-profile probe bad-numa --bytes 1048576
cargo run -p llama-rust --example session
```

Next code change is PLAN systems depth after item 259 (`HostMemoryPoolsSupported`).
`gguf_gemv serve --engine`
streams NDJSON, chunks prefill, and appends MoE JSONL on the same
Engine scheduler. Phase 0 leftover
is a Llama NORM real-model fixture when a GGUF is on disk. Physical
Phase 4 stays parked. Do not add crates.io
runtime deps. Do not start Metal-in-crate on Linux. Do not invent a
`block_iq4_nl_4_4` dequant. Do not invent an arch. Do not list mixtral
or qwen3vlmoe as an accepted arch. Do not invent `$/M tokens`.
