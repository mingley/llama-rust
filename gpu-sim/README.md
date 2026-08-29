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
| `cudaHostRegister` pins pageable host for DMA (`host_register`) | `alloc_overhead_ns` (mlock, host-sync) |
| `cudaHostAllocMapped` / `host_register_mapped`: kernel may read host with no H2D | host PCIe vs HBM |
| `cudaMallocManaged` (`alloc_managed`) does not charge HBM until migrate | `alloc_overhead_ns` (VA reserve at the call) |
| `cudaMemAdviseSetReadMostly`: prefetch replicates | same DMA as a move |
| `drop_managed_copy`: dest eviction of one ReadMostly GPU | other copies stay |
| `cudaMemAdviseSetAccessedBy`: kernel may read without migrating | interconnect, not local HBM |
| `cudaMemAdviseSetPreferredLocation`: stay if already there | interconnect on remote read; writes migrate |
| `cudaMemPrefetchAsync` (`prefetch` / `prefetch_host`) **moves** unless ReadMostly | PCIe / NVLink (1 ns if already local) |
| `cuMemAddressReserve` (`va_reserve`) is a VA with no physical pages | `alloc_overhead_ns` (at the call) |
| `cuMemMap` (`va_map`) charges HBM; `va_unmap` refunds; the VA is reusable | `alloc_overhead_ns` (map) |
| `cuMemCreate` (`va_create`) charges HBM with no VA; `va_map_handle` maps it without a second charge; `va_release_handle` refunds when maps are 0 | `alloc_overhead_ns` (create and map) |
| `cuMemGetAllocationGranularity` (`va_granularity_bytes`): reserve/map sizes align (`0`/`1` = any) | not timed |
| `cuMemSetAccess` (`va_set_access`) PROT_READ on a peer; dest HBM stays 0 | interconnect, not local HBM |
| `va_map_range` / `va_unmap_range` map a span; `kernel()` needs the whole VA; `kernel_bufs` / `memset_buf` / `MemcpyOp::offset` touch a mapped span | HBM = mapped bytes |
| `va_release` parks an unmapped VA; `va_acquire` remaps same size | map only on reuse (no second reserve) |
| `va_acquire_paged` maps the VA in `page` physicals | `alloc_overhead_ns` per block |
| `cudaLaunchHostFunc` (`host_func`) is stream-ordered host work | `host_func_ns` (no compute / copy occupancy) |
| `cudaStreamCreate` (`set_stream_blocking`) serializes with NULL | copy/compute overlap vs NULL |
| host pin / `mlock` budget (`host_pin_bytes`) | `SimError::PinOom` |
| `cudaFree` (`free_sync`) waits owning GPU(s), then every copy is gone | stream-ordered `free` refunds when that stream runs |
| `cudaMemcpyAsync` of pageable host memory is host-synchronous | `pageable_permille` (bounce + DMA) |
| `cudaMemcpyAsync` of pinned / device memory is stream-ordered | PCIe / NVLink bandwidth |
| `cudaMemcpy` (`memcpy_sync`) waits that stream | pinned `memcpy` does not |
| `synchronize_device` waits one GPU | other GPUs keep running |
| stream order, event dependencies | memcpy microseconds |
| residency: a kernel may only read **device**, **mapped-host**, VMM peer `va_set_access` (reads), or mempool peer `pool_set_access` (read/write) allocations; managed first-touch at kernel start | PCIe / NVLink / HBM bandwidth |
| HBM vs host-pinned: `alloc_host_pinned` does not charge HBM | pageable vs pinned H2D (`pageable_permille`) |
| copy-engine occupancy | launch overhead |
| peer accessibility | size-dependent efficiency |
| graph capture does not execute; launch replays | GEMM util / grouped-MoE ‰ |
| forked capture: `wait_event` on a captured record joins that stream | copy/compute overlap inside one launch |
| `launch_graph` during capture is a child-graph node | nested exec expanded at parent launch |
| independent streams stay live during capture | query/sync of a capturing stream is Invalid |
| graph instantiate is host-sync; first launch pays it once | `graph_instantiate_ns` |
| graph upload is host-sync after instantiate; first launch pays it once | `graph_upload_ns` |
| graph update replaces steps when topology matches (device, stream, kind) | `graph_update_ns` |
| graph clone is an independent uninstantiated copy; child graphs cloned recursively | `graph_clone_ns` |
| graph destroy drops the id (`cudaGraphDestroy`) | 1 ns host-sync |
| graph launch amortizes per-kernel launch overhead | `graph_launch_ns` |
| `synchronize_stream` waits one stream only | other streams keep running |
| `synchronize_event` waits the record only | later ops on that stream keep running |
| `idle_until` drains, then jumps the clock | GPU idle until the next arrival |
| `event_elapsed_ns` is record-to-record delta | `cudaEventElapsedTime` (ns) |
| `cudaEventDisableTiming` forbids elapsed | wait / query still work |
| `query_event` is non-blocking | `cudaEventQuery` |
| `query_stream` is non-blocking | `cudaStreamQuery` |
| `mem_info` is `(free, total)` HBM | `cudaMemGetInfo` |
| stream[i+1].start ≥ stream[i].finish (`Operation` timestamps) | queue wait vs run |
| higher `set_stream_priority` starts first under contention | launch overhead |
| `compute_slots>=2` overlaps independent kernels at full issue rate | kernel duration (not SM-partition) |
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
  `ns_per_token`, `ttft_ns`, `itl_ns`

`Score::with_tokens(n)` fills `ns_per_token = wall_ns / n`.
`Score::with_latencies` attaches TTFT / mean ITL when a caller samples the
virtual clock at token boundaries (`expertvm::sim_replay` does this).
`energy_uj` is profile board TDP × virtual wall (`mW × ns / 1e6`). There is
no invented `$/M tokens` field. Device-to-device replica copies charge the
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

CUDA graphs: `begin_capture` / `end_capture` / `instantiate_graph` /
`update_graph` / `clone_graph` / `destroy_graph` / `launch_graph`. Capture does
not advance the virtual clock. Independent streams stay live. A stream that
`wait_event`s an event recorded in this capture joins (CUDA forked capture);
`launch_graph` remaps origin-stream nodes onto the launch stream so copy and
compute can overlap. Query or `synchronize_stream` of a capturing stream, and
node `synchronize`, are `Invalid`. `launch_graph` during capture records a
child-graph node if the child is already instantiated; parent launch expands
it. Independent streams still launch live. Alloc/free cannot be captured, including
host-sync `malloc` / `free_sync` / `memcpy_sync` / `synchronize_device`.
Instantiate, update, and upload are host-synchronous and cannot run during capture.
`clone_graph` is `cudaGraphClone` (`graph_clone_ns`): an independent
uninstantiated copy; child-graph nodes are cloned recursively (a diamond
of shared children becomes one cloned child). Destroying the original
child still breaks a parent that names it; a recursive clone of that
parent keeps working. `destroy_graph` is `cudaGraphDestroy` (1 ns;
later launch is unknown). First launch instantiates if needed (`graph_instantiate_ns` once)
then uploads if needed (`graph_upload_ns`). `upload_graph` is `cudaGraphUpload`.
`update_graph` copies source steps into an instantiated exec when the
device, stream, and op kinds match (`graph_update_ns`); a topology
mismatch is `Invalid`. `expertvm --graph-update` parks a leaf GEMM on
evict and updates the next miss instead of instantiate. `--graph-clone`
copies the capture (`cudaGraphClone`) before instantiate. Launch pays `graph_launch_ns` once; recorded
kernels skip per-kernel launch overhead.
`memset` is an HBM-write kernel on a resident alloc. `host_func` is
`cudaLaunchHostFunc`: stream-ordered host work that does not occupy compute
or copy engines (other streams may GEMM). Peer D2D requires a
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
Default `cudaMallocAsync` uses the device mempool with release threshold
`0` (unused bytes return to the OS when the stream-ordered free
completes). `create_pool` / `alloc_from_pool` /
`set_pool_release_threshold` / `pool_trim_to` are `cudaMemPoolCreate` /
`cudaMallocFromPoolAsync` / `cudaMemPoolAttrReleaseThreshold` /
`cudaMemPoolTrimTo`. `u64::MAX` holds unused bytes so `malloc` can OOM
until trim. Capture cannot include pool create/trim/set-attribute.
`alloc_host` is pageable; `host_register` / `host_register_mapped` are
`cudaHostRegister` (host-synchronous). `alloc_host_mapped` is
`cudaHostAllocMapped`: a kernel may read it with no H2D, billed at host
PCIe, and it does not charge HBM. Capture cannot include host
alloc/register. `alloc_managed` is `cudaMallocManaged` (no HBM until
`prefetch` / first-touch at kernel start). `mem_advise` is `cudaMemAdvise` (host-sync).
`SetReadMostly` makes prefetch replicate; `SetAccessedBy` lets a kernel
read without migrating. `SetPreferredLocation` keeps a page already at
that GPU there on a remote read (writes still migrate; host preferred
does not skip kernel first-touch). `prefetch` / `prefetch_host` are
`cudaMemPrefetchAsync` and **move** unless ReadMostly. Capture of
`alloc_managed` / `mem_advise` is refused; a graph must record prefetch
before the kernel unless AccessedBy or PreferredLocation covers that GPU.
`va_reserve` / `va_map` / `va_unmap` / `va_free` are CUDA virtual memory.
`va_granularity_bytes` is `cuMemGetAllocationGranularity` (`0`/`1` accepts
any size; a 2 MiB profile rejects unaligned reserve/map).
`va_map_range` / `va_unmap_range` map sparse physicals (HBM is the mapped
span). `va_create` is `cuMemCreate` (HBM, no VA). `va_map_handle` is `cuMemMap`
of that handle (no second HBM charge; two VAs may share it). `va_release_handle`
is `cuMemRelease` when no maps remain. `va_map` still Create+Maps in one call.
`va_set_access` is `cuMemSetAccess` PROT_READ on a peer (no dest HBM;
writes still need a local map). `pool_set_access` is `cudaMemPoolSetAccess`
ReadWrite on a peer (no dest HBM; kernels may write). `kernel()` needs the whole VA covered; `kernel_bufs`, `memset_buf`, and
`MemcpyOp::offset` touch a mapped page (paged KV). `va_acquire` remaps an idle VA of the same
size (or reserves); `va_acquire_paged` maps KV-block physicals covering the VA;
`va_release` unmaps into that pool. Capture cannot
include them.
`host_func` is `cudaLaunchHostFunc` (stream-ordered; other streams can compute).
`set_stream_blocking` is `cudaStreamCreate` vs `cudaStreamNonBlocking`
(NULL serializes with blocking streams; created streams default to
non-blocking). `set_legacy_null_stream` is the CUDA legacy default
stream (NULL serializes with every stream).
`host_pin_bytes` caps page-locked host (`cudaMallocHost` / `cudaHostRegister`);
overflow is `PinOom`. Example default is unlimited.
`set_stream_priority` is `cudaStreamCreateWithPriority` (higher first when
compute contends). `set_created_streams_priority` assigns created streams
their id. `set_stream_sm_permille` is a green-context SM fraction
(compute-bound kernels scale; memory-bound keep full HBM; default unset is
a full chip). `Operation` carries `submit_ns` / `start_ns` / `done_ns`
so stream[i+1].start ≥ stream[i].finish is inspectable. `GpuOp` /
`Operation` is the compiled submit DAG (`Sim::operations`).

In-flight ops are not cancelled. `gpu-profile capture` is refused in this
crate: someone with a GPU writes a `key=value` file; agents `parse` it.
