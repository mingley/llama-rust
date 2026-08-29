//! Discrete-event GPU-systems simulator.

use std::cmp::Reverse;
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};

use crate::error::SimError;
use crate::ids::{
    AllocId, DeviceId, EventId, GraphId, IpcHandleId, MemHandleId, MulticastId, OpId, PoolId,
    PtrExportId, ShareableHandleId, StreamId,
};
use crate::ops::{
    GpuOp as Kind, KernelBuf, KernelKind, KernelNodeParams, MemAdvise, MemAttach, MemcpyOp,
    Operation, Place,
};
use crate::profile::{ns_for_bytes, HardwareProfile, LinkKind};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Preferred {
    None,
    Host,
    Gpu(DeviceId),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Attach {
    Global,
    Host,
    Single(StreamId),
}

struct Alloc {
    bytes: u64,
    devices: Vec<DeviceId>,
    leases: u32,
    live: bool,
    host_pinned: bool,
    /// `cudaHostAlloc` / `cudaHostRegister` of pageable host memory (not yet pinned).
    host_pageable: bool,
    /// Mapped into the device VA (`cudaHostAllocMapped` / `cudaHostRegisterMapped`).
    host_mapped: bool,
    /// Came from `cudaHostRegister`, not `cudaMallocHost`.
    host_registered: bool,
    /// `cudaMallocManaged`: one location; [`Sim::prefetch`] migrates it.
    managed: bool,
    /// Stream visibility for managed memory ([`Sim::stream_attach`]).
    attach: Attach,
    /// `cudaMemAdviseSetReadMostly`: prefetch replicates instead of moving.
    read_mostly: bool,
    /// GPUs that may read this alloc without a local copy (`SetAccessedBy` /
    /// VMM `cuMemSetAccess` PROT_READ). Mempool peer access lives on [`Pool`].
    accessed_by: BTreeSet<DeviceId>,
    /// VMM `cuMemSetAccess` PROT_READWRITE peers (writes without a local map).
    vmm_write_by: BTreeSet<DeviceId>,
    /// `cudaMemAdviseSetPreferredLocation` (host or one GPU).
    preferred: Preferred,
    /// `cuMemAddressReserve` VA. HBM is charged only while mapped (possibly sparse).
    vmm: bool,
    /// Physical maps `(device, offset, bytes)` into a VMM VA.
    vmm_maps: Vec<(DeviceId, u64, u64)>,
    /// `None` is `cudaMalloc` / host-pinned. `Some` is `cudaMallocAsync` from that pool.
    pool: Option<PoolId>,
    /// Import from [`Sim::ipc_open`] (`cudaIpcOpenMemHandle`). Physicals stay on the source.
    ipc_src: Option<AllocId>,
    /// Live [`Sim::ipc_open`] mappings of this allocation.
    ipc_opens: u32,
    /// Import from [`Sim::pool_import_ptr`] (`cudaMemPoolImportPointer`).
    share_src: Option<AllocId>,
    /// Live [`Sim::pool_import_ptr`] mappings of this allocation.
    share_opens: u32,
}

impl Alloc {
    fn remote_read_ok(&self, device: DeviceId) -> bool {
        if !self.live {
            return false;
        }
        if self.managed {
            if self.accessed_by.contains(&device) {
                return true;
            }
            return match self.preferred {
                Preferred::Gpu(p) => self.devices.contains(&p) && p != device,
                Preferred::None | Preferred::Host => false,
            };
        }
        self.vmm && self.accessed_by.contains(&device) && !self.vmm_maps.is_empty()
    }

    fn vmm_home(&self) -> Option<DeviceId> {
        self.vmm_maps.first().map(|(d, _, _)| *d)
    }

    fn device_attach_ok(&self, stream: StreamId) -> bool {
        if !self.managed {
            return true;
        }
        match self.attach {
            Attach::Global => true,
            Attach::Host => false,
            Attach::Single(s) => s == stream,
        }
    }
}

struct Pool {
    device: DeviceId,
    live: u64,
    cached: u64,
    release_threshold: u64,
    /// GPUs granted [`Sim::pool_set_access`] (`cudaMemPoolSetAccess` ReadWrite).
    accessed_by: BTreeSet<DeviceId>,
    /// Created with `cudaMemAllocationHandleTypePosixFileDescriptor`.
    shareable: bool,
    /// Imported pools share live/cached/threshold with this root.
    share_root: Option<PoolId>,
}

impl Pool {
    fn new(device: DeviceId) -> Self {
        Self {
            device,
            live: 0,
            cached: 0,
            release_threshold: 0,
            accessed_by: BTreeSet::new(),
            shareable: false,
            share_root: None,
        }
    }
}

struct MemHandle {
    device: DeviceId,
    bytes: u64,
    /// `cuMemCreate` / retain refs. `0` is released (`cuMemRelease`).
    refs: u32,
    maps: u32,
    /// HBM still charged for this physical.
    charged: bool,
}

struct Multicast {
    bytes: u64,
    n_dev: u32,
    devices: Vec<DeviceId>,
    binds: BTreeMap<DeviceId, MemHandleId>,
    maps: u32,
}

struct Op {
    device: DeviceId,
    stream: StreamId,
    kind: Kind,
    deps: Vec<OpId>,
    done: bool,
    cancelled: bool,
    launch: LaunchCost,
    submit_ns: u64,
    start_ns: Option<u64>,
    done_ns: Option<u64>,
}

/// How a submitted op pays kernel/graph launch overhead.
#[derive(Clone, Copy)]
enum LaunchCost {
    /// Standalone kernel: profile `launch_overhead_ns`.
    Kernel,
    /// First recorded op of a graph launch: profile `graph_launch_ns`.
    GraphHead,
    /// Later recorded ops of a graph launch: no extra launch overhead.
    GraphBody,
}

struct Running {
    op: OpId,
    remaining_ns: u64,
    /// Occupancy group: copies on the same link index, else unique.
    share: Share,
}

enum Share {
    Solo,
    Link(usize),
}

struct GpuRt {
    used: u64,
    /// In-flight kernels / memset / allreduce occupying Hyper-Q slots.
    compute: u8,
    copies: u8,
}

struct Ev {
    recorded_by: Option<OpId>,
    /// `false` is `cudaEventDisableTiming`: [`Sim::event_elapsed_ns`] fails.
    timing: bool,
}

struct Capture {
    origin: (DeviceId, StreamId),
    streams: BTreeSet<(DeviceId, StreamId)>,
    events: BTreeSet<EventId>,
    /// `cudaMallocAsync` ids recorded as graph mem alloc nodes.
    mem_allocs: Vec<AllocId>,
}

/// One `cudaGraphAdd*` / captured node plus [`Sim::graph_add_dependencies`] edges.
#[derive(Clone, Debug)]
struct GraphStep {
    device: DeviceId,
    stream: StreamId,
    kind: Kind,
    /// Predecessor node indices (`cudaGraphAddDependencies`). Empty is independent.
    deps: Vec<usize>,
}

struct Graph {
    steps: Vec<GraphStep>,
    origin: (DeviceId, StreamId),
    /// `cudaGraphInstantiate` has run (explicit or first launch).
    instantiated: bool,
    /// `cudaGraphUpload` has run (explicit or first launch after instantiate).
    uploaded: bool,
    /// `cudaGraphInstantiateFlagAutoFreeOnLaunch`: free graph mem before relaunch.
    auto_free_on_launch: bool,
}

/// DFS state for [`Sim::clone_graph`]: unique graphs in post-order, ancestor stack.
struct CloneWalk {
    order: Vec<GraphId>,
    seen: BTreeSet<GraphId>,
    stack: Vec<GraphId>,
}

/// Deterministic GPU node.
pub struct Sim {
    profile: HardwareProfile,
    clock: u64,
    next_op: u64,
    next_alloc: u64,
    next_graph: u32,
    allocs: BTreeMap<AllocId, Alloc>,
    ops: BTreeMap<OpId, Op>,
    tail: BTreeMap<(DeviceId, StreamId), OpId>,
    events: BTreeMap<EventId, Ev>,
    graphs: BTreeMap<GraphId, Graph>,
    capturing: Option<Capture>,
    capture_buf: Vec<GraphStep>,
    /// Graph-owned mem alloc node ids (`cudaMallocAsync` captured into a graph).
    graph_allocs: BTreeMap<GraphId, Vec<AllocId>>,
    /// Worker-stream ops that [`cudaGraphLaunch`] folded into the launch stream.
    ///
    /// Independent graph nodes run on internal streams (Hyper-Q). The launch
    /// stream still waits for them (`cudaStreamSynchronize` / later submits).
    graph_joins: BTreeMap<(DeviceId, StreamId), Vec<OpId>>,
    running: Vec<Running>,
    gpus: BTreeMap<DeviceId, GpuRt>,
    bytes_moved: u64,
    hbm_peak: u64,
    unavailable: BTreeSet<DeviceId>,
    fail_next_memcpy: bool,
    extra_transfer_ns: u64,
    peer_enabled: BTreeSet<(DeviceId, DeviceId)>,
    legacy_null_stream: bool,
    priority: BTreeMap<(DeviceId, StreamId), i32>,
    /// `cudaStreamCreate` (blocking) streams. They serialize with [`StreamId::NULL`].
    blocking: BTreeSet<(DeviceId, StreamId)>,
    /// Green-context SM fraction per stream (‰). Missing is full chip (`1000`).
    sm_permille: BTreeMap<(DeviceId, StreamId), u16>,
    next_pool: u32,
    pools: BTreeMap<PoolId, Pool>,
    default_pools: BTreeMap<DeviceId, PoolId>,
    next_handle: u64,
    mem_handles: BTreeMap<MemHandleId, MemHandle>,
    next_ipc: u64,
    ipc_handles: BTreeMap<IpcHandleId, AllocId>,
    next_share: u64,
    share_handles: BTreeMap<ShareableHandleId, PoolId>,
    next_ptr: u64,
    ptr_exports: BTreeMap<PtrExportId, AllocId>,
    /// Explicit [`Sim::va_map_handle`] maps. Missing is combined Create+Map.
    vmm_handle_at: BTreeMap<(AllocId, DeviceId, u64, u64), MemHandleId>,
    next_mc: u32,
    multicasts: BTreeMap<MulticastId, Multicast>,
    /// Reserved VAs mapped with [`Sim::va_map_multicast`].
    mc_vas: BTreeMap<AllocId, MulticastId>,
    pinned_used: u64,
    /// Unmapped live VAs waiting for [`Self::va_acquire`].
    vmm_idle: Vec<AllocId>,
}

impl Sim {
    /// Idle node at t = 0.
    #[must_use]
    pub fn new(profile: HardwareProfile) -> Self {
        let mut gpus = BTreeMap::new();
        for g in &profile.gpus {
            let replaced = gpus.insert(
                g.id,
                GpuRt {
                    used: 0,
                    compute: 0,
                    copies: 0,
                },
            );
            let _dup = replaced.is_some();
        }
        let peer_enabled = seed_peers(&profile);
        let (next_pool, pools, default_pools) = seed_pools(&profile);
        Self {
            profile,
            clock: 0,
            next_op: 1,
            next_alloc: 1,
            next_graph: 1,
            allocs: BTreeMap::new(),
            ops: BTreeMap::new(),
            tail: BTreeMap::new(),
            events: BTreeMap::new(),
            graphs: BTreeMap::new(),
            capturing: None,
            capture_buf: Vec::new(),
            graph_allocs: BTreeMap::new(),
            graph_joins: BTreeMap::new(),
            running: Vec::new(),
            gpus,
            bytes_moved: 0,
            hbm_peak: 0,
            unavailable: BTreeSet::new(),
            fail_next_memcpy: false,
            extra_transfer_ns: 0,
            peer_enabled,
            legacy_null_stream: false,
            priority: BTreeMap::new(),
            blocking: BTreeSet::new(),
            sm_permille: BTreeMap::new(),
            next_pool,
            pools,
            default_pools,
            next_handle: 1,
            mem_handles: BTreeMap::new(),
            next_ipc: 1,
            ipc_handles: BTreeMap::new(),
            next_share: 1,
            share_handles: BTreeMap::new(),
            next_ptr: 1,
            ptr_exports: BTreeMap::new(),
            vmm_handle_at: BTreeMap::new(),
            next_mc: 1,
            multicasts: BTreeMap::new(),
            mc_vas: BTreeMap::new(),
            pinned_used: 0,
            vmm_idle: Vec::new(),
        }
    }

    /// Virtual clock, nanoseconds.
    #[must_use]
    pub fn clock_ns(&self) -> u64 {
        self.clock
    }

    /// Payload bytes moved by completed memcpys.
    #[must_use]
    pub fn bytes_moved(&self) -> u64 {
        self.bytes_moved
    }

    /// High-water HBM bytes on the greediest GPU.
    #[must_use]
    pub fn hbm_peak(&self) -> u64 {
        self.hbm_peak
    }

    /// Page-locked host bytes currently charged against [`HardwareProfile::host_pin_bytes`].
    #[must_use]
    pub fn pin_used(&self) -> u64 {
        self.pinned_used
    }

    /// Pin budget from the profile (`u64::MAX` is unlimited).
    #[must_use]
    pub fn pin_budget(&self) -> u64 {
        self.profile.host_pin_bytes
    }

    /// Borrow the profile.
    #[must_use]
    pub fn profile(&self) -> &HardwareProfile {
        &self.profile
    }

    /// Stream an op was submitted on.
    #[must_use]
    pub fn op_stream(&self, id: OpId) -> Option<StreamId> {
        self.ops.get(&id).map(|o| o.stream)
    }

    /// Compiled DAG node for a submitted op. Capture-only ids are absent until launch.
    #[must_use]
    pub fn operation(&self, id: OpId) -> Option<Operation> {
        self.ops.get(&id).map(|o| snapshot_op(id, o))
    }

    /// Submitted ops in id order (the dependency DAG plus completion flags).
    pub fn operations(&self) -> impl Iterator<Item = Operation> + '_ {
        self.ops.iter().map(|(id, o)| snapshot_op(*id, o))
    }

    /// Bytes currently reserved on `device`.
    pub fn hbm_used(&self, device: DeviceId) -> Result<u64, SimError> {
        Ok(self.gpu_rt(device)?.used)
    }

    /// `cudaMemGetInfo`: `(free, total)` HBM bytes on `device`.
    pub fn mem_info(&self, device: DeviceId) -> Result<(u64, u64), SimError> {
        let total = self.profile.gpu(device)?.hbm_bytes;
        let used = self.hbm_used(device)?;
        Ok((total.saturating_sub(used), total))
    }

    /// Whether `alloc` currently has a copy on `device`.
    ///
    /// A VMM VA is resident only when mapped physicals cover the whole reserved
    /// size (a hole is not [`Self::kernel`]-readable). [`Self::kernel_bufs`]
    /// of a mapped span uses [`Self::is_range_resident`].
    pub fn is_resident(&self, alloc: AllocId, device: DeviceId) -> Result<bool, SimError> {
        let a = self.alloc_ref(alloc)?;
        if a.vmm {
            Ok(a.live && vmm_covers(&a.vmm_maps, device, 0, a.bytes))
        } else {
            Ok(a.live && a.devices.contains(&device))
        }
    }

    /// Whether `[offset, offset+bytes)` of `alloc` is mapped/resident on `device`.
    ///
    /// Non-VMM allocations ignore the span (the object is on the device or not).
    /// A range past the reservation is [`SimError::Invalid`].
    pub fn is_range_resident(
        &self,
        alloc: AllocId,
        device: DeviceId,
        offset: u64,
        bytes: u64,
    ) -> Result<bool, SimError> {
        let a = self.alloc_ref(alloc)?;
        if offset.saturating_add(bytes) > a.bytes {
            return Err(SimError::Invalid {
                why: "range past alloc",
            });
        }
        if a.vmm {
            Ok(a.live && vmm_covers(&a.vmm_maps, device, offset, bytes))
        } else {
            Ok(a.live && a.devices.contains(&device))
        }
    }

    /// Refuse new work (and starting queued work) on `device`.
    pub fn set_unavailable(&mut self, device: DeviceId, yes: bool) -> Result<(), SimError> {
        let _gpu = self.profile.gpu(device)?;
        if yes {
            let _was = self.unavailable.insert(device);
        } else {
            let _was = self.unavailable.remove(&device);
        }
        Ok(())
    }

    /// Whether [`Self::set_unavailable`] is set for `device`.
    #[must_use]
    pub fn is_unavailable(&self, device: DeviceId) -> bool {
        self.unavailable.contains(&device)
    }

    /// Next memcpy that *starts* fails with [`SimError::TransferFailed`] (expert load failure).
    pub fn fail_next_memcpy(&mut self) {
        self.fail_next_memcpy = true;
    }

    /// Extra nanoseconds added to every memcpy and allreduce (injected transfer delay).
    pub fn set_extra_transfer_ns(&mut self, ns: u64) {
        self.extra_transfer_ns = ns;
    }

    /// CUDA stream priority (`cudaStreamCreateWithPriority`). Higher runs first
    /// when multiple ops are ready for the same resource. Default `0`.
    pub fn set_stream_priority(
        &mut self,
        device: DeviceId,
        stream: StreamId,
        priority: i32,
    ) -> Result<(), SimError> {
        let _gpu = self.profile.gpu(device)?;
        let _prev = self.priority.insert((device, stream), priority);
        Ok(())
    }

    /// Current priority for `(device, stream)`, or `0` if unset.
    #[must_use]
    pub fn stream_priority(&self, device: DeviceId, stream: StreamId) -> i32 {
        self.priority.get(&(device, stream)).copied().unwrap_or(0)
    }

    /// Reserve a green-context SM fraction for `(device, stream)` (‰ of peak FLOP/s).
    ///
    /// `1000` is a full chip. Compute-bound kernels scale as `1000 / permille`;
    /// memory-bound kernels keep full HBM. Default (unset) is `1000`. `0` is
    /// [`SimError::Invalid`].
    pub fn set_stream_sm_permille(
        &mut self,
        device: DeviceId,
        stream: StreamId,
        permille: u16,
    ) -> Result<(), SimError> {
        let _gpu = self.profile.gpu(device)?;
        if permille == 0 || permille > 1000 {
            return Err(SimError::Invalid {
                why: "sm permille must be 1..=1000",
            });
        }
        let _prev = self.sm_permille.insert((device, stream), permille);
        Ok(())
    }

    /// SM fraction for `(device, stream)`, or `1000` if unset.
    #[must_use]
    pub fn stream_sm_permille(&self, device: DeviceId, stream: StreamId) -> u16 {
        self.sm_permille
            .get(&(device, stream))
            .copied()
            .unwrap_or(1000)
            .max(1)
    }

    /// CUDA legacy null stream: [`StreamId::NULL`] serializes with every other stream
    /// on that device. Off by default (`cudaStreamNonBlocking` created streams).
    pub fn set_legacy_null_stream(&mut self, yes: bool) {
        self.legacy_null_stream = yes;
    }

    /// Whether [`Self::set_legacy_null_stream`] is on.
    #[must_use]
    pub fn legacy_null_stream(&self) -> bool {
        self.legacy_null_stream
    }

    /// `cudaStreamCreate` (`yes`) vs `cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)`.
    ///
    /// Blocking streams serialize with [`StreamId::NULL`] even when legacy null
    /// is off. Created streams default to non-blocking (vLLM-style). The null
    /// stream's flags are [`Self::set_legacy_null_stream`], not this call.
    pub fn set_stream_blocking(
        &mut self,
        device: DeviceId,
        stream: StreamId,
        yes: bool,
    ) -> Result<(), SimError> {
        let _gpu = self.profile.gpu(device)?;
        if stream == StreamId::NULL {
            return Err(SimError::Invalid {
                why: "null stream uses set_legacy_null_stream",
            });
        }
        if yes {
            let _was = self.blocking.insert((device, stream));
        } else {
            let _was = self.blocking.remove(&(device, stream));
        }
        Ok(())
    }

    /// Whether `stream` is a blocking `cudaStreamCreate` stream on `device`.
    #[must_use]
    pub fn stream_is_blocking(&self, device: DeviceId, stream: StreamId) -> bool {
        self.blocking.contains(&(device, stream))
    }

    /// Mark streams `1 .. n_streams` blocking on every GPU (`cudaStreamCreate`).
    ///
    /// [`StreamId::NULL`] stays the default stream. `n_streams <= 1` is a no-op.
    pub fn set_created_streams_blocking(&mut self, n_streams: u8) -> Result<(), SimError> {
        let devices: Vec<DeviceId> = self.profile.gpus.iter().map(|g| g.id).collect();
        for d in devices {
            for s in 1..n_streams {
                self.set_stream_blocking(d, StreamId(u16::from(s)), true)?;
            }
        }
        Ok(())
    }

    /// `cudaStreamCreateWithPriority` for streams `1 .. n_streams` on every GPU.
    ///
    /// Priority equals the stream id (higher id runs first when compute
    /// contends). [`StreamId::NULL`] stays `0`. `n_streams <= 1` is a no-op.
    pub fn set_created_streams_priority(&mut self, n_streams: u8) -> Result<(), SimError> {
        let devices: Vec<DeviceId> = self.profile.gpus.iter().map(|g| g.id).collect();
        for d in devices {
            for s in 1..n_streams {
                self.set_stream_priority(d, StreamId(u16::from(s)), i32::from(s))?;
            }
        }
        Ok(())
    }

    /// Take one Hyper-Q compute slot, or `false` if [`GpuProfile::compute_slots`] are full.
    fn take_compute(&mut self, device: DeviceId) -> Result<bool, SimError> {
        self.take_compute_n(device, 1)
    }

    /// Take `n` Hyper-Q slots (`cudaLaunchCooperativeKernel` takes every slot).
    fn take_compute_n(&mut self, device: DeviceId, n: u8) -> Result<bool, SimError> {
        let cap = self.profile.gpu(device)?.compute_slots.max(1);
        let need = n.max(1);
        let rt = self.gpu_rt_mut(device)?;
        if rt.compute.saturating_add(need) > cap {
            return Ok(false);
        }
        rt.compute = rt.compute.saturating_add(need);
        Ok(true)
    }

    /// Release one Hyper-Q compute slot.
    fn drop_compute(&mut self, device: DeviceId) -> Result<(), SimError> {
        self.drop_compute_n(device, 1)
    }

    /// Release `n` Hyper-Q slots.
    fn drop_compute_n(&mut self, device: DeviceId, n: u8) -> Result<(), SimError> {
        let rt = self.gpu_rt_mut(device)?;
        rt.compute = rt.compute.saturating_sub(n.max(1));
        Ok(())
    }

    /// Slots a kernel occupies: cooperative grids need the whole GPU.
    fn kernel_slots(&self, device: DeviceId, cooperative: bool) -> Result<u8, SimError> {
        if cooperative {
            Ok(self.profile.gpu(device)?.compute_slots.max(1))
        } else {
            Ok(1)
        }
    }

    /// `cudaDevAttrCooperativeLaunch` must be set on `device`.
    fn require_cooperative(&self, device: DeviceId) -> Result<(), SimError> {
        if self.profile.gpu(device)?.cooperative_launch {
            Ok(())
        } else {
            Err(SimError::Invalid {
                why: "cooperative launch not supported",
            })
        }
    }

    /// `cudaDeviceEnablePeerAccess(dst)` from `src`. No-op if `src == dst`.
    pub fn enable_peer(&mut self, src: DeviceId, dst: DeviceId) -> Result<(), SimError> {
        if src == dst {
            return Ok(());
        }
        let _link = self.profile.link(Some(src), Some(dst))?;
        let _was = self.peer_enabled.insert((src, dst));
        Ok(())
    }

    /// `cudaDeviceDisablePeerAccess(dst)` from `src`. Later D2D is [`SimError::PeerDisabled`].
    pub fn disable_peer(&mut self, src: DeviceId, dst: DeviceId) -> Result<(), SimError> {
        if src == dst {
            return Ok(());
        }
        let _gpu_s = self.profile.gpu(src)?;
        let _gpu_d = self.profile.gpu(dst)?;
        let _was = self.peer_enabled.remove(&(src, dst));
        Ok(())
    }

    /// Whether `src` may D2D-read `dst` (directed, like CUDA peer access).
    #[must_use]
    pub fn peer_access(&self, src: DeviceId, dst: DeviceId) -> bool {
        src == dst || self.peer_enabled.contains(&(src, dst))
    }

    /// No unfinished ops on `(device, stream)`, including in-flight.
    ///
    /// Same as [`Self::query_stream`]: a capturing stream is [`SimError::Invalid`].
    pub fn stream_is_idle(&self, device: DeviceId, stream: StreamId) -> Result<bool, SimError> {
        self.query_stream(device, stream)
    }

    /// `cudaStreamQuery`: whether `(device, stream)` has no unfinished ops. Does not wait.
    ///
    /// Unknown devices are [`SimError::Invalid`]. A busy stream is `Ok(false)`.
    /// A stream in an active graph capture is [`SimError::Invalid`].
    pub fn query_stream(&self, device: DeviceId, stream: StreamId) -> Result<bool, SimError> {
        let _gpu = self.profile.gpu(device)?;
        if self.in_capture(device, stream) {
            return Err(SimError::Invalid {
                why: "cannot query stream during capture",
            });
        }
        Ok(self.stream_idle(device, stream))
    }

    /// Recorded event whose record op has completed.
    #[must_use]
    pub fn event_complete(&self, event: EventId) -> bool {
        self.events
            .get(&event)
            .and_then(|e| e.recorded_by)
            .is_some_and(|id| self.op_done(id))
    }

    /// `cudaEventQuery`: whether `event` is recorded and complete. Does not wait.
    ///
    /// Unknown ids are [`SimError::UnknownEvent`]. Incomplete records are `Ok(false)`.
    pub fn query_event(&self, event: EventId) -> Result<bool, SimError> {
        if !self.events.contains_key(&event) {
            return Err(SimError::UnknownEvent { event: event.0 });
        }
        Ok(self.event_complete(event))
    }

    /// `cudaEventCreate` (timing enabled). Implicit on first [`Self::record_event`]
    /// if the id was never created.
    pub fn create_event(&mut self, event: EventId) -> Result<(), SimError> {
        self.insert_event(event, true)
    }

    /// `cudaEventCreateWithFlags(..., cudaEventDisableTiming)`.
    ///
    /// Record / wait / query still work. [`Self::event_elapsed_ns`] is
    /// [`SimError::Invalid`].
    pub fn create_event_disable_timing(&mut self, event: EventId) -> Result<(), SimError> {
        self.insert_event(event, false)
    }

    /// Whether `event` was created with timing enabled (`cudaEventDefault`).
    pub fn event_timing(&self, event: EventId) -> Result<bool, SimError> {
        self.events
            .get(&event)
            .map(|e| e.timing)
            .ok_or(SimError::UnknownEvent { event: event.0 })
    }

    fn insert_event(&mut self, event: EventId, timing: bool) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture event create")?;
        if self.events.contains_key(&event) {
            return Err(SimError::Invalid {
                why: "event already created",
            });
        }
        let _prev = self.events.insert(
            event,
            Ev {
                recorded_by: None,
                timing,
            },
        );
        Ok(())
    }

    /// Drop not-yet-started ops on `(device, stream)`. In-flight ops still complete.
    pub fn cancel_stream(&mut self, device: DeviceId, stream: StreamId) -> Result<u32, SimError> {
        let _gpu = self.profile.gpu(device)?;
        let running: BTreeSet<OpId> = self.running.iter().map(|r| r.op).collect();
        let mut n = 0u32;
        let ids: Vec<OpId> = self
            .ops
            .iter()
            .filter(|(id, o)| {
                o.device == device && o.stream == stream && !o.done && !running.contains(id)
            })
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            if let Some(op) = self.ops.get_mut(&id) {
                op.cancelled = true;
                op.done = true;
                op.done_ns = Some(self.clock);
                n = n.saturating_add(1);
            }
        }
        Ok(n)
    }

    /// How many submitted ops were skipped by cancel or a failed transfer.
    #[must_use]
    pub fn cancelled_count(&self) -> u32 {
        let n = self.ops.values().filter(|o| o.cancelled).count();
        u32::try_from(n).unwrap_or(u32::MAX)
    }

    /// Start every currently ready op without advancing the virtual clock.
    pub fn start_ready(&mut self) -> Result<(), SimError> {
        self.schedule()
    }

    /// Ring allreduce among `parts`. Each alloc must already be resident on its device.
    pub fn allreduce(
        &mut self,
        parts: &[(DeviceId, AllocId)],
        bytes: u64,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        if parts.len() < 2 {
            return Err(SimError::Invalid {
                why: "allreduce needs >= 2 ranks",
            });
        }
        let device = parts
            .first()
            .ok_or(SimError::Invalid {
                why: "allreduce needs >= 2 ranks",
            })?
            .0;
        self.submit(
            device,
            stream,
            Kind::AllReduce {
                parts: parts.to_vec(),
                bytes,
            },
        )
    }

    /// Start recording later submits on `(device, stream)`. Recorded ops do not run.
    ///
    /// Other streams keep running. A stream that [`Self::wait_event`]s an
    /// event recorded in this capture **joins** (CUDA forked capture) so copy
    /// and compute can overlap inside one [`Self::launch_graph`].
    /// [`Self::record_event_external`] / [`Self::wait_event_external`] do not
    /// join (`cudaEventRecordExternal` / `cudaEventWaitExternal`).
    pub fn begin_capture(&mut self, device: DeviceId, stream: StreamId) -> Result<(), SimError> {
        let _gpu = self.profile.gpu(device)?;
        if self.capturing.is_some() {
            return Err(SimError::Invalid {
                why: "nested graph capture",
            });
        }
        if !self.stream_idle(device, stream) {
            return Err(SimError::Invalid {
                why: "capture requires idle stream",
            });
        }
        let mut streams = BTreeSet::new();
        let _ins = streams.insert((device, stream));
        self.capturing = Some(Capture {
            origin: (device, stream),
            streams,
            events: BTreeSet::new(),
            mem_allocs: Vec::new(),
        });
        self.capture_buf.clear();
        Ok(())
    }

    /// Finish capture. The graph is empty of side effects until [`Self::launch_graph`].
    pub fn end_capture(&mut self) -> Result<GraphId, SimError> {
        let Some(cap) = self.capturing.take() else {
            return Err(SimError::Invalid {
                why: "end_capture without begin_capture",
            });
        };
        let id = GraphId(self.next_graph);
        self.next_graph = self.next_graph.saturating_add(1);
        let steps = core::mem::take(&mut self.capture_buf);
        let mem_allocs = cap.mem_allocs;
        let _prev = self.graphs.insert(
            id,
            Graph {
                steps,
                origin: cap.origin,
                instantiated: false,
                uploaded: false,
                auto_free_on_launch: false,
            },
        );
        let _old = self.graph_allocs.insert(id, mem_allocs);
        Ok(id)
    }

    /// Enqueue every recorded op. Origin-stream nodes use `stream`; forked
    /// streams keep the ids they joined with, so copy and compute can overlap.
    /// Independent nodes (empty [`GraphStep::deps`]) use internal streams so
    /// Hyper-Q can overlap them; the launch stream still waits for the whole
    /// graph (`cudaGraphLaunch`).
    ///
    /// First launch [`Self::instantiate_graph`]s if needed (`cudaGraphInstantiate`
    /// then [`Self::upload_graph`] then `cudaGraphLaunch`). Later launches skip
    /// both. [`Self::instantiate_graph_auto_free`] frees graph mem on the launch
    /// stream before a later launch's alloc nodes (`AutoFreeOnLaunch`).
    /// During capture on a captured stream this records a child-graph node (the
    /// child must already be instantiated). Independent streams still launch live.
    /// [`Self::upload_graph`] is skipped while any stream is capturing (host-sync
    /// upload cannot run during capture); the live launch still enqueues.
    pub fn launch_graph(&mut self, graph: GraphId, stream: StreamId) -> Result<u32, SimError> {
        let (origin, instantiated, uploaded) = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            (g.origin, g.instantiated, g.uploaded)
        };
        if self.in_capture(origin.0, stream) {
            return self.capture_child_graph(graph, origin.0, stream, instantiated);
        }
        if !instantiated {
            self.instantiate_graph(graph)?;
        }
        if !uploaded && self.capturing.is_none() {
            self.upload_graph(graph)?;
        }
        let mut stack = BTreeSet::new();
        self.enqueue_graph(graph, stream, true, &mut stack, &[])
    }

    fn capture_child_graph(
        &mut self,
        graph: GraphId,
        device: DeviceId,
        stream: StreamId,
        instantiated: bool,
    ) -> Result<u32, SimError> {
        if !instantiated {
            return Err(SimError::Invalid {
                why: "child graph not instantiated",
            });
        }
        let _id = self.submit_captured(device, stream, Kind::ChildGraph { graph })?;
        Ok(1)
    }

    /// Stream-ordered `cudaFreeAsync` of live graph mem before recorded steps
    /// (`cudaGraphInstantiateFlagAutoFreeOnLaunch`). Not counted in launch size.
    fn enqueue_auto_frees(&mut self, graph: GraphId, stream: StreamId) -> Result<(), SimError> {
        let (device, ids) = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            if !g.auto_free_on_launch {
                return Ok(());
            }
            let ids = self.graph_allocs.get(&graph).cloned().unwrap_or_default();
            (g.origin.0, ids)
        };
        for id in ids {
            let live = self
                .alloc_ref(id)
                .is_ok_and(|a| a.live && a.devices.contains(&device));
            if !live {
                continue;
            }
            let _op =
                self.submit_launch(device, stream, Kind::Free { id }, LaunchCost::GraphBody)?;
        }
        Ok(())
    }

    fn enqueue_graph(
        &mut self,
        graph: GraphId,
        stream: StreamId,
        head: bool,
        stack: &mut BTreeSet<GraphId>,
        extra_wait: &[OpId],
    ) -> Result<u32, SimError> {
        if !stack.insert(graph) {
            return Err(SimError::Invalid {
                why: "cyclic child graph",
            });
        }
        self.enqueue_auto_frees(graph, stream)?;
        let (origin, steps) = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            (g.origin, g.steps.clone())
        };
        let order = graph_topo_order(&steps)?;
        let launch_tail = self.tail.get(&(origin.0, stream)).copied();
        let mut n = 0u32;
        let mut head = head;
        let mut rec_ops: BTreeMap<EventId, OpId> = BTreeMap::new();
        let mut node_ops: Vec<Vec<OpId>> = vec![Vec::new(); steps.len()];
        let mut node_stream: Vec<Option<StreamId>> = vec![None; steps.len()];
        let mut worker = 0u16;
        let mut pending_joins = Vec::new();
        for idx in order {
            let step = steps.get(idx).ok_or(SimError::Invalid {
                why: "graph dependency",
            })?;
            let wait = graph_node_waits(step, extra_wait, launch_tail, &node_ops)?;
            let s = self.graph_exec_stream(origin, stream, step, &node_stream, &mut worker);
            if let Some(slot) = node_stream.get_mut(idx) {
                *slot = Some(s);
            }
            if let Kind::ChildGraph { graph: child } = &step.kind {
                let add = self.enqueue_graph(*child, s, head, stack, &wait)?;
                head = false;
                n = n.saturating_add(add);
                if let Some(id) = self.tail.get(&(step.device, s)).copied() {
                    if let Some(ops) = node_ops.get_mut(idx) {
                        ops.push(id);
                    }
                    if s != stream {
                        pending_joins.push(id);
                    }
                }
                if s != stream {
                    if let Some(ids) = self.graph_joins.get(&(step.device, s)).cloned() {
                        pending_joins.extend(ids);
                    }
                }
                continue;
            }
            let rec = match &step.kind {
                Kind::EventRecord { event, .. } => Some(*event),
                _ => None,
            };
            let wait_ev = match &step.kind {
                Kind::EventWait { event, external } => Some((*event, *external)),
                _ => None,
            };
            let launch = if head {
                LaunchCost::GraphHead
            } else {
                LaunchCost::GraphBody
            };
            head = false;
            let id = self.submit_launch(step.device, s, step.kind.clone(), launch)?;
            for dep in wait {
                self.add_op_dep(id, dep);
            }
            if let Some(event) = rec {
                let _prev = rec_ops.insert(event, id);
            }
            if let Some((event, external)) = wait_ev {
                if external {
                    if let Some(rec_id) = self.events.get(&event).and_then(|e| e.recorded_by) {
                        self.add_op_dep(id, rec_id);
                    }
                } else if let Some(rec_id) = rec_ops.get(&event).copied() {
                    self.add_op_dep(id, rec_id);
                }
            }
            if let Some(ops) = node_ops.get_mut(idx) {
                ops.push(id);
            }
            if s != stream {
                pending_joins.push(id);
            }
            n = n.saturating_add(1);
        }
        if !pending_joins.is_empty() {
            self.graph_joins
                .entry((origin.0, stream))
                .or_default()
                .extend(pending_joins);
        }
        let _gone = stack.remove(&graph);
        Ok(n)
    }

    /// Internal stream for an origin-stream graph node. Chains stay on the
    /// predecessor stream; independent nodes get a Hyper-Q worker.
    fn graph_exec_stream(
        &mut self,
        origin: (DeviceId, StreamId),
        launch: StreamId,
        step: &GraphStep,
        node_stream: &[Option<StreamId>],
        worker: &mut u16,
    ) -> StreamId {
        if (step.device, step.stream) != origin {
            return step.stream;
        }
        if step.deps.len() == 1 {
            if let Some(pred) = step.deps.first().copied() {
                if let Some(s) = node_stream.get(pred).copied().flatten() {
                    self.bind_graph_worker(step.device, launch, s);
                    return s;
                }
            }
        }
        let taken = node_stream.iter().any(Option::is_some);
        let s = if !taken && step.deps.is_empty() {
            launch
        } else {
            alloc_graph_worker(launch, worker)
        };
        self.bind_graph_worker(step.device, launch, s);
        s
    }

    fn bind_graph_worker(&mut self, device: DeviceId, launch: StreamId, worker: StreamId) {
        if worker == launch {
            return;
        }
        if let Some(p) = self.priority.get(&(device, launch)).copied() {
            let _prev = self.priority.insert((device, worker), p);
        }
        if let Some(sm) = self.sm_permille.get(&(device, launch)).copied() {
            let _prev = self.sm_permille.insert((device, worker), sm);
        }
    }

    fn add_op_dep(&mut self, id: OpId, dep: OpId) {
        if let Some(op) = self.ops.get_mut(&id) {
            if !op.deps.contains(&dep) {
                op.deps.push(dep);
            }
        }
    }

    /// Recorded op count.
    pub fn graph_len(&self, graph: GraphId) -> Result<usize, SimError> {
        self.graphs
            .get(&graph)
            .map(|g| g.steps.len())
            .ok_or(SimError::Invalid {
                why: "unknown graph",
            })
    }

    /// Whether [`Self::instantiate_graph`] (or a first launch) has run.
    pub fn graph_instantiated(&self, graph: GraphId) -> Result<bool, SimError> {
        self.graphs
            .get(&graph)
            .map(|g| g.instantiated)
            .ok_or(SimError::Invalid {
                why: "unknown graph",
            })
    }

    /// Whether [`Self::upload_graph`] (or a first launch after instantiate) has run.
    pub fn graph_uploaded(&self, graph: GraphId) -> Result<bool, SimError> {
        self.graphs
            .get(&graph)
            .map(|g| g.uploaded)
            .ok_or(SimError::Invalid {
                why: "unknown graph",
            })
    }

    /// Whether [`Self::instantiate_graph_auto_free`] was used
    /// (`cudaGraphInstantiateFlagAutoFreeOnLaunch`).
    pub fn graph_auto_free_on_launch(&self, graph: GraphId) -> Result<bool, SimError> {
        self.graphs
            .get(&graph)
            .map(|g| g.auto_free_on_launch)
            .ok_or(SimError::Invalid {
                why: "unknown graph",
            })
    }

    /// `cudaGraphInstantiate`. Host-synchronous. Capture cannot include it.
    ///
    /// Already-instantiated ids are a no-op. The first [`Self::launch_graph`]
    /// calls this when needed (default flags: graph mem allocs without a
    /// matching free are reused on relaunch).
    pub fn instantiate_graph(&mut self, graph: GraphId) -> Result<(), SimError> {
        self.instantiate_graph_inner(graph, false)
    }

    /// `cudaGraphInstantiate` with `cudaGraphInstantiateFlagAutoFreeOnLaunch`.
    ///
    /// Host-synchronous. Capture cannot include it. Graph mem allocs are
    /// `cudaFreeAsync`'d on the launch stream before a later launch's alloc
    /// nodes run, so relaunch recharges HBM instead of reusing the pointer.
    /// Illegal when the graph has mem free nodes. Illegal after a default
    /// [`Self::instantiate_graph`].
    pub fn instantiate_graph_auto_free(&mut self, graph: GraphId) -> Result<(), SimError> {
        self.instantiate_graph_inner(graph, true)
    }

    fn instantiate_graph_inner(&mut self, graph: GraphId, auto_free: bool) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture graph instantiate")?;
        let (device, already, current_auto, has_free) = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let device = g.steps.first().map(|s| s.device).unwrap_or(DeviceId(0));
            let has_free = g.steps.iter().any(|s| matches!(&s.kind, Kind::Free { .. }));
            (device, g.instantiated, g.auto_free_on_launch, has_free)
        };
        if already {
            if auto_free && !current_auto {
                return Err(SimError::Invalid {
                    why: "graph instantiate flags",
                });
            }
            return Ok(());
        }
        if auto_free && has_free {
            return Err(SimError::Invalid {
                why: "auto free with mem free nodes",
            });
        }
        let ns = self.profile.gpu(device)?.graph_instantiate_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        g.instantiated = true;
        g.auto_free_on_launch = auto_free;
        Ok(())
    }

    /// `cudaGraphUpload`. Host-synchronous. Capture cannot include it.
    ///
    /// The exec must already be instantiated. Already-uploaded ids are a no-op.
    /// The first [`Self::launch_graph`] calls this when needed. [`Self::update_graph`]
    /// clears the flag so the next launch uploads again.
    pub fn upload_graph(&mut self, graph: GraphId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture graph upload")?;
        let (device, instantiated, already) = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let device = g.steps.first().map(|s| s.device).unwrap_or(DeviceId(0));
            (device, g.instantiated, g.uploaded)
        };
        if !instantiated {
            return Err(SimError::Invalid {
                why: "graph not instantiated",
            });
        }
        if already {
            return Ok(());
        }
        let ns = self.profile.gpu(device)?.graph_upload_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        self.graphs
            .get_mut(&graph)
            .ok_or(SimError::Invalid {
                why: "unknown graph",
            })?
            .uploaded = true;
        Ok(())
    }

    /// `cudaGraphExecUpdate`: replace `exec` steps with `src` when topology matches.
    ///
    /// Same device, stream, op kinds, cooperative flag, and dependency edges;
    /// KernelBuf / memcpy sizes may change. Pays `graph_update_ns`. Recapture
    /// if topology differs. Capture cannot include it. `exec` must already be
    /// instantiated. Graphs with mem alloc or mem free nodes cannot be updated
    /// (`cudaGraphExecUpdate` of mem nodes).
    pub fn update_graph(&mut self, exec: GraphId, src: GraphId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture graph update")?;
        if exec == src {
            return Err(SimError::Invalid {
                why: "graph update same id",
            });
        }
        let (instantiated, exec_steps, src_steps, device) = {
            let e = self.graphs.get(&exec).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let s = self.graphs.get(&src).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let device = e.steps.first().map(|s| s.device).unwrap_or(DeviceId(0));
            (e.instantiated, e.steps.clone(), s.steps.clone(), device)
        };
        if !instantiated {
            return Err(SimError::Invalid {
                why: "graph not instantiated",
            });
        }
        if !graph_topology_eq(&exec_steps, &src_steps) {
            return Err(SimError::Invalid {
                why: "graph update topology",
            });
        }
        if self.graph_has_mem_nodes(exec) || self.graph_has_mem_nodes(src) {
            return Err(SimError::Invalid {
                why: "cannot update graph mem nodes",
            });
        }
        let ns = self.profile.gpu(device)?.graph_update_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        let exec = self.graphs.get_mut(&exec).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        exec.steps = src_steps;
        exec.uploaded = false;
        Ok(())
    }

    /// `cudaGraphExecKernelNodeSetParams` on an instantiated exec.
    ///
    /// Node `node` must already be a kernel. [`KernelNodeParams::cooperative`]
    /// must match the existing node (cooperative vs `cudaLaunchKernel` is
    /// topology). Pointers and [`KernelKind`] may change. Pays
    /// `graph_set_params_ns` and clears the upload flag. Capture cannot include
    /// it. Graphs with mem alloc/free nodes are legal (unlike
    /// [`Self::update_graph`]).
    pub fn graph_exec_kernel_set_params(
        &mut self,
        exec: GraphId,
        node: usize,
        params: &KernelNodeParams,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel set params")?;
        let (instantiated, device, cooperative) = {
            let g = self.graphs.get(&exec).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let step = g.steps.get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?;
            let Kind::Kernel { cooperative, .. } = &step.kind else {
                return Err(SimError::Invalid {
                    why: "not a kernel node",
                });
            };
            if *cooperative != params.cooperative {
                return Err(SimError::Invalid {
                    why: "cooperative is topology",
                });
            }
            (g.instantiated, step.device, *cooperative)
        };
        if !instantiated {
            return Err(SimError::Invalid {
                why: "graph not instantiated",
            });
        }
        let reads = self.resolve_bufs(&params.reads)?;
        let writes = self.resolve_bufs(&params.writes)?;
        let ns = self.profile.gpu(device)?.graph_set_params_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        let g = self.graphs.get_mut(&exec).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = g.steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?;
        step.kind = Kind::Kernel {
            kind: params.kind.clone(),
            reads,
            writes,
            cooperative,
        };
        g.uploaded = false;
        Ok(())
    }

    /// Unique kernel node on `graph` plus its current [`KernelNodeParams`].
    ///
    /// Zero or more than one kernel node is Invalid. Used by
    /// `cudaGraphExecKernelNodeSetParams` callers that parked a leaf GEMM.
    pub fn graph_unique_kernel(
        &self,
        graph: GraphId,
    ) -> Result<(usize, KernelNodeParams), SimError> {
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let mut found = None;
        for (i, step) in g.steps.iter().enumerate() {
            let Kind::Kernel {
                kind,
                reads,
                writes,
                cooperative,
            } = &step.kind
            else {
                continue;
            };
            if found.is_some() {
                return Err(SimError::Invalid {
                    why: "not unique kernel node",
                });
            }
            found = Some((
                i,
                KernelNodeParams {
                    kind: kind.clone(),
                    reads: reads.clone(),
                    writes: writes.clone(),
                    cooperative: *cooperative,
                },
            ));
        }
        found.ok_or(SimError::Invalid {
            why: "not a kernel node",
        })
    }

    /// Graph mem alloc node ids (`cudaMallocAsync` / `cudaGraphAddMemAllocNode`).
    pub fn graph_mem_allocs(&self, graph: GraphId) -> Result<Vec<AllocId>, SimError> {
        if !self.graphs.contains_key(&graph) {
            return Err(SimError::Invalid {
                why: "unknown graph",
            });
        }
        Ok(self.graph_allocs.get(&graph).cloned().unwrap_or_default())
    }

    /// `cudaGraphClone`. Host-synchronous. Capture cannot include it.
    ///
    /// The clone is an independent graph (`instantiated = false`). Child-graph
    /// nodes are cloned recursively so the copy names new ids; a diamond of
    /// shared children becomes one cloned child. Graph mem alloc nodes get new
    /// `cudaMallocAsync` ids (independent HBM). Instantiating or updating one
    /// id does not change the other. Cycles among child ids fail.
    pub fn clone_graph(&mut self, graph: GraphId) -> Result<GraphId, SimError> {
        self.fail_if_capturing("cannot capture graph clone")?;
        let mut walk = CloneWalk {
            order: Vec::new(),
            seen: BTreeSet::new(),
            stack: Vec::new(),
        };
        self.collect_clone_tree(graph, &mut walk)?;
        let mut remap = BTreeMap::new();
        for &src in &walk.order {
            let id = GraphId(self.next_graph);
            self.next_graph = self.next_graph.saturating_add(1);
            let _prev = remap.insert(src, id);
        }
        let mut built = Vec::new();
        for &src in &walk.order {
            let (origin, raw) = {
                let g = self.graphs.get(&src).ok_or(SimError::Invalid {
                    why: "unknown graph",
                })?;
                (g.origin, g.steps.clone())
            };
            let steps = remap_child_graphs(&raw, &remap)?;
            let cloned = remap.get(&src).copied().ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let steps = self.clone_mem_alloc_nodes(src, cloned, steps)?;
            built.push((
                cloned,
                Graph {
                    steps,
                    origin,
                    instantiated: false,
                    uploaded: false,
                    auto_free_on_launch: false,
                },
                origin.0,
            ));
        }
        for (id, cloned, device) in built {
            let ns = self.profile.gpu(device)?.graph_clone_ns.max(1);
            self.clock = self.clock.saturating_add(ns);
            let _prev = self.graphs.insert(id, cloned);
        }
        remap.get(&graph).copied().ok_or(SimError::Invalid {
            why: "unknown graph",
        })
    }

    /// Post-order unique graphs in `graph`'s child-graph tree. Diamonds reuse
    /// `seen`; an ancestor appearing again is a cycle.
    fn collect_clone_tree(&self, graph: GraphId, walk: &mut CloneWalk) -> Result<(), SimError> {
        if walk.seen.contains(&graph) {
            return Ok(());
        }
        if walk.stack.contains(&graph) {
            return Err(SimError::Invalid {
                why: "cyclic child graph",
            });
        }
        let steps = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            g.steps.clone()
        };
        walk.stack.push(graph);
        for step in &steps {
            if let Kind::ChildGraph { graph: child } = &step.kind {
                self.collect_clone_tree(*child, walk)?;
            }
        }
        let _popped = walk.stack.pop();
        let _fresh = walk.seen.insert(graph);
        walk.order.push(graph);
        Ok(())
    }

    fn graph_has_mem_nodes(&self, graph: GraphId) -> bool {
        if self.graph_allocs.get(&graph).is_some_and(|v| !v.is_empty()) {
            return true;
        }
        self.graphs.get(&graph).is_some_and(|g| {
            g.steps
                .iter()
                .any(|s| matches!(&s.kind, Kind::Alloc { .. } | Kind::Free { .. }))
        })
    }

    fn fork_alloc(&mut self, src: AllocId) -> Result<AllocId, SimError> {
        let (bytes, pool) = {
            let a = self.alloc_ref(src)?;
            (a.bytes, a.pool)
        };
        let id = AllocId(self.next_alloc);
        self.next_alloc = self.next_alloc.saturating_add(1);
        let _prev = self.allocs.insert(
            id,
            Alloc {
                bytes,
                devices: Vec::new(),
                leases: 0,
                live: false,
                host_pinned: false,
                host_pageable: false,
                host_mapped: false,
                host_registered: false,
                managed: false,
                attach: Attach::Global,
                read_mostly: false,
                accessed_by: BTreeSet::new(),
                vmm_write_by: BTreeSet::new(),
                preferred: Preferred::None,
                vmm: false,
                vmm_maps: Vec::new(),
                pool,
                ipc_src: None,
                ipc_opens: 0,
                share_src: None,
                share_opens: 0,
            },
        );
        Ok(id)
    }

    fn clone_mem_alloc_nodes(
        &mut self,
        src: GraphId,
        dst: GraphId,
        steps: Vec<GraphStep>,
    ) -> Result<Vec<GraphStep>, SimError> {
        let src_ids = self.graph_allocs.get(&src).cloned().unwrap_or_default();
        if src_ids.is_empty() {
            return Ok(steps);
        }
        let mut map = BTreeMap::new();
        let mut dst_ids = Vec::new();
        for old in src_ids {
            let new = self.fork_alloc(old)?;
            let _prev = map.insert(old, new);
            dst_ids.push(new);
        }
        let _old = self.graph_allocs.insert(dst, dst_ids);
        Ok(steps
            .into_iter()
            .map(|mut step| {
                step.kind = remap_alloc_kind(step.kind, &map);
                step
            })
            .collect())
    }

    fn release_graph_allocs(&mut self, graph: GraphId) -> Result<(), SimError> {
        let ids = self.graph_allocs.remove(&graph).unwrap_or_default();
        for id in ids {
            let (live, bytes, devices) = {
                let Ok(a) = self.alloc_ref(id) else {
                    continue;
                };
                (a.live, a.bytes, a.devices.clone())
            };
            if !live {
                continue;
            }
            for d in devices {
                self.refund_device(d, id, bytes)?;
            }
            let a = self.alloc_mut(id)?;
            a.devices.clear();
            a.live = false;
        }
        Ok(())
    }

    /// `cudaGraphDestroy` / `cudaGraphExecDestroy`. Host-synchronous.
    ///
    /// Capture cannot include it. Later [`Self::launch_graph`] of this id is
    /// `unknown graph`. Clones are independent. Remaining graph mem allocs are
    /// freed (`cudaGraphDestroy` of a graph with mem nodes).
    pub fn destroy_graph(&mut self, graph: GraphId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture graph destroy")?;
        let g = self.graphs.remove(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let device = g.origin.0;
        let _gpu = self.profile.gpu(device)?;
        self.release_graph_allocs(graph)?;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphCreate`. Host-synchronous empty graph (`instantiated = false`).
    ///
    /// Capture cannot include it. The origin `(device, stream)` is the capture
    /// analog: [`Self::launch_graph`] remaps those nodes onto the launch stream.
    /// Add nodes with [`Self::graph_add_kernel`] and friends, then instantiate.
    pub fn create_graph(
        &mut self,
        device: DeviceId,
        stream: StreamId,
    ) -> Result<GraphId, SimError> {
        self.fail_if_capturing("cannot create graph during capture")?;
        let _gpu = self.profile.gpu(device)?;
        let id = GraphId(self.next_graph);
        self.next_graph = self.next_graph.saturating_add(1);
        let _prev = self.graphs.insert(
            id,
            Graph {
                steps: Vec::new(),
                origin: (device, stream),
                instantiated: false,
                uploaded: false,
                auto_free_on_launch: false,
            },
        );
        let _old = self.graph_allocs.insert(id, Vec::new());
        self.clock = self.clock.saturating_add(1);
        Ok(id)
    }

    /// `cudaGraphAddKernelNode` on an uninstantiated [`Self::create_graph`] id.
    ///
    /// Nodes start with no dependencies (CUDA). Use
    /// [`Self::graph_add_dependencies`] so a later node waits. Independent
    /// kernels may Hyper-Q overlap at [`Self::launch_graph`]. Capture cannot
    /// include it. Illegal after [`Self::instantiate_graph`]. Does not run the
    /// kernel; [`Self::launch_graph`] does.
    pub fn graph_add_kernel(
        &mut self,
        graph: GraphId,
        kind: KernelKind,
        reads: &[AllocId],
        writes: &[AllocId],
    ) -> Result<(), SimError> {
        self.graph_add_kernel_node(graph, kind, reads, writes, false)
    }

    /// `cudaGraphAddKernelNode` for a [`Self::cooperative_kernel`] launch.
    ///
    /// Occupies every Hyper-Q slot at launch. Capture cannot include it.
    /// Illegal after instantiate. Device must advertise
    /// [`crate::GpuProfile::cooperative_launch`].
    pub fn graph_add_cooperative_kernel(
        &mut self,
        graph: GraphId,
        kind: KernelKind,
        reads: &[AllocId],
        writes: &[AllocId],
    ) -> Result<(), SimError> {
        self.graph_add_kernel_node(graph, kind, reads, writes, true)
    }

    fn graph_add_kernel_node(
        &mut self,
        graph: GraphId,
        kind: KernelKind,
        reads: &[AllocId],
        writes: &[AllocId],
        cooperative: bool,
    ) -> Result<(), SimError> {
        let (device, stream) = self.graph_origin_for_add(graph)?;
        if cooperative {
            self.require_cooperative(device)?;
        }
        let reads: Vec<KernelBuf> = reads.iter().copied().map(KernelBuf::whole).collect();
        let writes: Vec<KernelBuf> = writes.iter().copied().map(KernelBuf::whole).collect();
        let reads = self.resolve_bufs(&reads)?;
        let writes = self.resolve_bufs(&writes)?;
        self.graph_push(
            graph,
            device,
            stream,
            Kind::Kernel {
                kind,
                reads,
                writes,
                cooperative,
            },
        )
    }

    /// `cudaGraphAddMemcpyNode`. Pageable copies cannot be graph nodes.
    pub fn graph_add_memcpy(&mut self, graph: GraphId, op: MemcpyOp) -> Result<(), SimError> {
        let (device, stream) = self.graph_origin_for_add(graph)?;
        let _a = self.alloc_ref(op.alloc)?;
        if op.src.is_pageable() || op.dst.is_pageable() {
            return Err(SimError::Invalid {
                why: "cannot add pageable memcpy",
            });
        }
        self.graph_push(graph, device, stream, Kind::Memcpy(op))
    }

    /// `cudaGraphAddMemsetNode` of a [`KernelBuf`] span.
    pub fn graph_add_memset(&mut self, graph: GraphId, buf: KernelBuf) -> Result<(), SimError> {
        let (device, stream) = self.graph_origin_for_add(graph)?;
        let total = self.alloc_ref(buf.id)?.bytes;
        let (offset, bytes) = kernel_span(total, &buf)?;
        if bytes == 0 {
            return Err(SimError::Invalid {
                why: "zero-byte memset",
            });
        }
        self.graph_push(
            graph,
            device,
            stream,
            Kind::Memset {
                id: buf.id,
                offset,
                bytes,
            },
        )
    }

    /// `cudaGraphAddHostNode` (`cudaLaunchHostFunc`).
    pub fn graph_add_host_func(&mut self, graph: GraphId) -> Result<(), SimError> {
        let (device, stream) = self.graph_origin_for_add(graph)?;
        self.graph_push(graph, device, stream, Kind::HostFunc)
    }

    /// `cudaGraphAddEventRecordNode`. `external` is `cudaEventRecordExternal`.
    pub fn graph_add_event_record(
        &mut self,
        graph: GraphId,
        event: EventId,
        external: bool,
    ) -> Result<(), SimError> {
        let (device, stream) = self.graph_origin_for_add(graph)?;
        if !self.events.contains_key(&event) {
            return Err(SimError::UnknownEvent { event: event.0 });
        }
        self.graph_push(graph, device, stream, Kind::EventRecord { event, external })
    }

    /// `cudaGraphAddEventWaitNode`. `external` is `cudaEventWaitExternal`.
    pub fn graph_add_event_wait(
        &mut self,
        graph: GraphId,
        event: EventId,
        external: bool,
    ) -> Result<(), SimError> {
        let (device, stream) = self.graph_origin_for_add(graph)?;
        if !self.events.contains_key(&event) {
            return Err(SimError::UnknownEvent { event: event.0 });
        }
        self.graph_push(graph, device, stream, Kind::EventWait { event, external })
    }

    /// `cudaGraphAddChildGraphNode`. `child` must already be instantiated.
    ///
    /// Sibling children have no dependency until [`Self::graph_add_dependencies`].
    /// Independent children may Hyper-Q overlap at parent launch.
    pub fn graph_add_child(&mut self, graph: GraphId, child: GraphId) -> Result<(), SimError> {
        let (device, stream) = self.graph_origin_for_add(graph)?;
        if child == graph {
            return Err(SimError::Invalid {
                why: "graph child is self",
            });
        }
        let (instantiated, origin) = {
            let c = self.graphs.get(&child).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            (c.instantiated, c.origin.0)
        };
        if !instantiated {
            return Err(SimError::Invalid {
                why: "child graph not instantiated",
            });
        }
        if origin != device {
            return Err(SimError::Invalid {
                why: "graph child gpu mismatch",
            });
        }
        self.graph_push(graph, device, stream, Kind::ChildGraph { graph: child })
    }

    /// `cudaGraphAddMemAllocNode`. Returns a pending `cudaMallocAsync` id.
    ///
    /// The pointer is not resident until [`Self::launch_graph`]. Capture cannot
    /// include it (use [`Self::alloc`] during stream capture). Illegal after
    /// instantiate. [`Self::update_graph`] of mem nodes is Invalid.
    pub fn graph_add_alloc(&mut self, graph: GraphId, bytes: u64) -> Result<AllocId, SimError> {
        let (device, stream) = self.graph_origin_for_add(graph)?;
        if bytes == 0 {
            return Err(SimError::Invalid {
                why: "zero-byte alloc",
            });
        }
        let pool = self.default_pool(device)?;
        let id = self.insert_pool_alloc(pool, bytes)?;
        self.graph_push(graph, device, stream, Kind::Alloc { id, bytes })?;
        self.graph_allocs.entry(graph).or_default().push(id);
        Ok(id)
    }

    /// `cudaGraphAddMemFreeNode` of a pending or live allocation.
    pub fn graph_add_free(&mut self, graph: GraphId, id: AllocId) -> Result<(), SimError> {
        let (device, stream) = self.graph_origin_for_add(graph)?;
        let _a = self.alloc_ref(id)?;
        self.graph_push(graph, device, stream, Kind::Free { id })
    }

    /// `cudaGraphAddDependencies`: `from` must complete before `to` starts.
    ///
    /// Capture cannot include it. Illegal after instantiate. Indices are
    /// 0-based in add order. A cycle is Invalid. Independent nodes (no edge)
    /// may Hyper-Q overlap at [`Self::launch_graph`].
    pub fn graph_add_dependencies(
        &mut self,
        graph: GraphId,
        from: usize,
        to: usize,
    ) -> Result<(), SimError> {
        let _origin = self.graph_origin_for_add(graph)?;
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let n = g.steps.len();
        if from == to || from >= n || to >= n {
            return Err(SimError::Invalid {
                why: "graph dependency",
            });
        }
        if graph_reaches(&g.steps, to, from) {
            return Err(SimError::Invalid {
                why: "cyclic graph dependencies",
            });
        }
        let step = g.steps.get_mut(to).ok_or(SimError::Invalid {
            why: "graph dependency",
        })?;
        if !step.deps.contains(&from) {
            step.deps.push(from);
            step.deps.sort_unstable();
        }
        Ok(())
    }

    /// Predecessor indices of node `i` (`cudaGraphAddDependencies`).
    pub fn graph_node_deps(&self, graph: GraphId, i: usize) -> Result<Vec<usize>, SimError> {
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        g.steps
            .get(i)
            .map(|s| s.deps.clone())
            .ok_or(SimError::Invalid {
                why: "graph dependency",
            })
    }

    fn graph_origin_for_add(&self, graph: GraphId) -> Result<(DeviceId, StreamId), SimError> {
        self.fail_if_capturing("cannot add graph node during capture")?;
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        if g.instantiated {
            return Err(SimError::Invalid {
                why: "graph instantiated",
            });
        }
        Ok(g.origin)
    }

    fn graph_push(
        &mut self,
        graph: GraphId,
        device: DeviceId,
        stream: StreamId,
        kind: Kind,
    ) -> Result<(), SimError> {
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        g.steps.push(GraphStep {
            device,
            stream,
            kind,
            deps: Vec::new(),
        });
        Ok(())
    }

    /// Stream-ordered allocation (`cudaMallocAsync`) from the device default pool.
    ///
    /// Capacity is reserved when the op starts. The pointer is not usable until
    /// this stream catches up. Capture records a graph mem alloc node
    /// (`cudaMallocAsync` during stream capture). [`Self::malloc`] is
    /// host-synchronous `cudaMalloc` and cannot be captured.
    pub fn alloc(
        &mut self,
        device: DeviceId,
        bytes: u64,
        stream: StreamId,
    ) -> Result<AllocId, SimError> {
        let pool = self.default_pool(device)?;
        self.alloc_from_pool(device, pool, bytes, stream)
    }

    /// Device default mempool (`cudaDeviceGetDefaultMemPool`).
    pub fn default_pool(&self, device: DeviceId) -> Result<PoolId, SimError> {
        let _gpu = self.profile.gpu(device)?;
        self.default_pools
            .get(&device)
            .copied()
            .ok_or(SimError::Invalid {
                why: "default pool missing",
            })
    }

    /// `cudaDeviceSetMemPool`. Later [`Self::alloc`] draws from `pool`.
    ///
    /// Capture cannot include it. `pool` must belong to `device` (an imported
    /// sibling is legal). Does not change live/cached bytes.
    pub fn set_device_mempool(&mut self, device: DeviceId, pool: PoolId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture mempool")?;
        let _gpu = self.profile.gpu(device)?;
        if self.pool_ref(pool)?.device != device {
            return Err(SimError::Invalid {
                why: "pool device mismatch",
            });
        }
        let _prev = self.default_pools.insert(device, pool);
        self.clock = self.clock.saturating_add(self.first_alloc_ns().max(1));
        Ok(())
    }

    /// `cudaMemPoolCreate` for `device`. Release threshold starts at 0.
    ///
    /// Not shareable (`cudaMemHandleTypeNone`). Use
    /// [`Self::create_shareable_pool`] for POSIX-FD export.
    pub fn create_pool(&mut self, device: DeviceId) -> Result<PoolId, SimError> {
        self.insert_pool(device, false)
    }

    /// `cudaMemPoolCreate` with `cudaMemAllocationHandleTypePosixFileDescriptor`.
    pub fn create_shareable_pool(&mut self, device: DeviceId) -> Result<PoolId, SimError> {
        self.insert_pool(device, true)
    }

    fn insert_pool(&mut self, device: DeviceId, shareable: bool) -> Result<PoolId, SimError> {
        self.fail_if_capturing("cannot capture mempool")?;
        let _gpu = self.profile.gpu(device)?;
        let id = PoolId(self.next_pool);
        self.next_pool = self.next_pool.saturating_add(1);
        let mut p = Pool::new(device);
        p.shareable = shareable;
        let _prev = self.pools.insert(id, p);
        Ok(id)
    }

    /// `cudaMallocFromPoolAsync`. `pool` must belong to `device`.
    pub fn alloc_from_pool(
        &mut self,
        device: DeviceId,
        pool: PoolId,
        bytes: u64,
        stream: StreamId,
    ) -> Result<AllocId, SimError> {
        if bytes == 0 {
            return Err(SimError::Invalid {
                why: "zero-byte alloc",
            });
        }
        if self.pool_ref(pool)?.device != device {
            return Err(SimError::Invalid {
                why: "pool device mismatch",
            });
        }
        let id = self.insert_pool_alloc(pool, bytes)?;
        let _op = self.submit(device, stream, Kind::Alloc { id, bytes })?;
        if self.in_capture(device, stream) {
            if let Some(cap) = self.capturing.as_mut() {
                cap.mem_allocs.push(id);
            }
        }
        Ok(id)
    }

    fn insert_pool_alloc(&mut self, pool: PoolId, bytes: u64) -> Result<AllocId, SimError> {
        let id = AllocId(self.next_alloc);
        self.next_alloc = self.next_alloc.saturating_add(1);
        let _prev = self.allocs.insert(
            id,
            Alloc {
                bytes,
                devices: Vec::new(),
                leases: 0,
                live: false,
                host_pinned: false,
                host_pageable: false,
                host_mapped: false,
                host_registered: false,
                managed: false,
                attach: Attach::Global,
                read_mostly: false,
                accessed_by: BTreeSet::new(),
                vmm_write_by: BTreeSet::new(),
                preferred: Preferred::None,
                vmm: false,
                vmm_maps: Vec::new(),
                pool: Some(pool),
                ipc_src: None,
                ipc_opens: 0,
                share_src: None,
                share_opens: 0,
            },
        );
        Ok(id)
    }

    /// `cudaMemPoolAttrReleaseThreshold`. Does not trim; later frees apply it.
    ///
    /// `0` (CUDA default) returns unused bytes to the OS when the stream-ordered
    /// free completes. `u64::MAX` holds them so [`Self::mem_info`] still counts
    /// them used until [`Self::pool_trim_to`].
    pub fn set_pool_release_threshold(&mut self, pool: PoolId, bytes: u64) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture mempool")?;
        let root = self.pool_root(pool)?;
        self.pool_mut(root)?.release_threshold = bytes;
        Ok(())
    }

    /// Set every device default pool's release threshold.
    pub fn set_default_pool_release_threshold(&mut self, bytes: u64) -> Result<(), SimError> {
        let ids: Vec<PoolId> = self.default_pools.values().copied().collect();
        for id in ids {
            self.set_pool_release_threshold(id, bytes)?;
        }
        Ok(())
    }

    /// `cudaMemPoolTrimTo`: return cached bytes above `min_bytes` to the OS.
    ///
    /// Only completed frees are cached; in-flight [`Self::free`] has not entered
    /// the pool yet. Applied immediately to `mem_info` (no extra device sync).
    pub fn pool_trim_to(&mut self, pool: PoolId, min_bytes: u64) -> Result<u64, SimError> {
        self.fail_if_capturing("cannot capture mempool")?;
        let root = self.pool_root(pool)?;
        let (device, cached) = {
            let p = self.pool_ref(root)?;
            (p.device, p.cached)
        };
        let drop = cached.saturating_sub(min_bytes);
        if drop == 0 {
            return Ok(0);
        }
        self.pool_mut(root)?.cached = cached.saturating_sub(drop);
        let used = self.gpu_rt(device)?.used;
        self.gpu_rt_mut(device)?.used = used.saturating_sub(drop);
        Ok(drop)
    }

    /// Unused bytes held by `pool` (`cudaMemGetInfo` still counts them used).
    ///
    /// An imported pool reports the exporter's cache.
    pub fn pool_cached(&self, pool: PoolId) -> Result<u64, SimError> {
        Ok(self.pool_ref(self.pool_root(pool)?)?.cached)
    }

    /// Live bytes allocated from `pool` and not yet freed.
    ///
    /// An imported pool reports the exporter's live bytes.
    pub fn pool_live(&self, pool: PoolId) -> Result<u64, SimError> {
        Ok(self.pool_ref(self.pool_root(pool)?)?.live)
    }

    /// `cudaMemPoolSetAccess` ReadWrite on `device` for allocations from `pool`.
    ///
    /// Host-synchronous. Does not charge dest HBM. A kernel on `device` may
    /// read **and write** pointers whose physicals live on the pool's GPU
    /// (interconnect). Capture cannot include it. Needs a topology link and
    /// directed peer access from the pool GPU, same as D2D. Same-device is a
    /// no-op that still records access. Applies to existing and later allocs.
    pub fn pool_set_access(&mut self, pool: PoolId, device: DeviceId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture mempool")?;
        let _gpu = self.profile.gpu(device)?;
        let owner = self.pool_ref(pool)?.device;
        if owner != device {
            let _link = self.profile.link(Some(owner), Some(device))?;
            if !self.peer_access(owner, device) {
                return Err(SimError::PeerDisabled {
                    src: owner,
                    dst: device,
                });
            }
        }
        let _ins = self.pool_mut(pool)?.accessed_by.insert(device);
        self.clock = self.clock.saturating_add(self.first_alloc_ns().max(1));
        Ok(())
    }

    /// Drop [`Self::pool_set_access`] for `device` (`cudaMemAccessFlagsProtNone`).
    pub fn pool_unset_access(&mut self, pool: PoolId, device: DeviceId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture mempool")?;
        let _gpu = self.profile.gpu(device)?;
        let _p = self.pool_ref(pool)?;
        let _was = self.pool_mut(pool)?.accessed_by.remove(&device);
        self.clock = self.clock.saturating_add(self.first_alloc_ns().max(1));
        Ok(())
    }

    /// Whether `device` has [`Self::pool_set_access`] on `pool`.
    pub fn is_pool_accessed_by(&self, pool: PoolId, device: DeviceId) -> Result<bool, SimError> {
        Ok(self.pool_ref(pool)?.accessed_by.contains(&device))
    }

    /// Whether `pool` was created with a POSIX-FD shareable handle type.
    pub fn is_pool_shareable(&self, pool: PoolId) -> Result<bool, SimError> {
        Ok(self.pool_ref(pool)?.shareable)
    }

    /// Whether `pool` came from [`Self::pool_import`].
    pub fn is_pool_imported(&self, pool: PoolId) -> Result<bool, SimError> {
        Ok(self.pool_ref(pool)?.share_root.is_some())
    }

    /// `cudaMemPoolExportToShareableHandle`. Host-synchronous.
    ///
    /// Only [`Self::create_shareable_pool`] pools export. Default and
    /// [`Self::create_pool`] pools are `not shareable`. The same pool returns
    /// the same handle. Capture cannot include it.
    pub fn pool_export(&mut self, pool: PoolId) -> Result<ShareableHandleId, SimError> {
        self.fail_if_capturing("cannot capture mempool")?;
        let (device, shareable) = {
            let p = self.pool_ref(pool)?;
            (p.device, p.shareable)
        };
        if !shareable {
            return Err(SimError::Invalid {
                why: "not shareable",
            });
        }
        if let Some(h) = self
            .share_handles
            .iter()
            .find_map(|(&h, src)| (*src == pool).then_some(h))
        {
            return Ok(h);
        }
        let ns = self.profile.gpu(device)?.alloc_overhead_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        let h = ShareableHandleId(self.next_share);
        self.next_share = self.next_share.saturating_add(1);
        let _prev = self.share_handles.insert(h, pool);
        Ok(h)
    }

    /// `cudaMemPoolImportFromShareableHandle` on `device`.
    ///
    /// Returns a new [`PoolId`] that shares live/cached/threshold with the
    /// exporter (no extra HBM). `device` must match the exporter. Capture
    /// cannot include it.
    pub fn pool_import(
        &mut self,
        device: DeviceId,
        handle: ShareableHandleId,
    ) -> Result<PoolId, SimError> {
        self.fail_if_capturing("cannot capture mempool")?;
        let src = *self.share_handles.get(&handle).ok_or(SimError::Invalid {
            why: "unknown shareable",
        })?;
        let root = self.pool_root(src)?;
        if self.pool_ref(root)?.device != device {
            return Err(SimError::Invalid {
                why: "pool device mismatch",
            });
        }
        let ns = self.profile.gpu(device)?.alloc_overhead_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        let id = PoolId(self.next_pool);
        self.next_pool = self.next_pool.saturating_add(1);
        let mut p = Pool::new(device);
        p.share_root = Some(root);
        let _prev = self.pools.insert(id, p);
        Ok(id)
    }

    /// `cudaMemPoolExportPointer` of a live allocation from a shareable pool.
    ///
    /// The same alloc returns the same handle. [`Self::ipc_get`] of a pool
    /// alloc is Invalid. Capture cannot include it.
    pub fn pool_export_ptr(&mut self, id: AllocId) -> Result<PtrExportId, SimError> {
        self.fail_if_capturing("cannot capture mempool")?;
        let (pool, device) = {
            let a = self.alloc_ref(id)?;
            if !a.live
                || a.managed
                || a.vmm
                || a.host_pinned
                || a.host_pageable
                || a.ipc_src.is_some()
                || a.share_src.is_some()
                || a.devices.is_empty()
            {
                return Err(SimError::Invalid {
                    why: "not shareable ptr",
                });
            }
            let pool = a.pool.ok_or(SimError::Invalid {
                why: "not shareable ptr",
            })?;
            let d = a.devices.first().copied().ok_or(SimError::Invalid {
                why: "not shareable ptr",
            })?;
            (pool, d)
        };
        if !self.pool_ref(self.pool_root(pool)?)?.shareable {
            return Err(SimError::Invalid {
                why: "not shareable ptr",
            });
        }
        if let Some(h) = self
            .ptr_exports
            .iter()
            .find_map(|(&h, src)| (*src == id).then_some(h))
        {
            return Ok(h);
        }
        let ns = self.profile.gpu(device)?.alloc_overhead_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        let h = PtrExportId(self.next_ptr);
        self.next_ptr = self.next_ptr.saturating_add(1);
        let _prev = self.ptr_exports.insert(h, id);
        Ok(h)
    }

    /// `cudaMemPoolImportPointer` into an imported pool. Alias shares the
    /// source physicals (no extra HBM). Capture cannot include it.
    pub fn pool_import_ptr(
        &mut self,
        pool: PoolId,
        export: PtrExportId,
    ) -> Result<AllocId, SimError> {
        self.fail_if_capturing("cannot capture mempool")?;
        let src = *self.ptr_exports.get(&export).ok_or(SimError::Invalid {
            why: "unknown ptr export",
        })?;
        let root = self.pool_root(pool)?;
        if self.pool_ref(pool)?.share_root.is_none() {
            return Err(SimError::Invalid {
                why: "not imported pool",
            });
        }
        let (bytes, devices, opens, src_root, device) = {
            let a = self.alloc_ref(src)?;
            if !a.live {
                return Err(SimError::Invalid { why: "freed" });
            }
            let src_pool = a.pool.ok_or(SimError::Invalid {
                why: "not shareable ptr",
            })?;
            let src_root = self.pool_root(src_pool)?;
            let d = a.devices.first().copied().ok_or(SimError::Invalid {
                why: "not shareable ptr",
            })?;
            (a.bytes, a.devices.clone(), a.share_opens, src_root, d)
        };
        if src_root != root {
            return Err(SimError::Invalid {
                why: "pool mismatch",
            });
        }
        let ns = self.profile.gpu(device)?.alloc_overhead_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        self.alloc_mut(src)?.share_opens = opens.saturating_add(1);
        let id = AllocId(self.next_alloc);
        self.next_alloc = self.next_alloc.saturating_add(1);
        let _prev = self.allocs.insert(
            id,
            Alloc {
                bytes,
                devices,
                leases: 0,
                live: true,
                host_pinned: false,
                host_pageable: false,
                host_mapped: false,
                host_registered: false,
                managed: false,
                attach: Attach::Global,
                read_mostly: false,
                accessed_by: BTreeSet::new(),
                vmm_write_by: BTreeSet::new(),
                preferred: Preferred::None,
                vmm: false,
                vmm_maps: Vec::new(),
                pool: Some(pool),
                ipc_src: None,
                ipc_opens: 0,
                share_src: Some(src),
                share_opens: 0,
            },
        );
        Ok(id)
    }

    /// Whether `id` is a live [`Self::pool_import_ptr`] alias.
    pub fn is_share_import(&self, id: AllocId) -> Result<bool, SimError> {
        let a = self.alloc_ref(id)?;
        Ok(a.live && a.share_src.is_some())
    }

    /// `cudaMalloc`: [`Self::synchronize_device`] then the pointer is usable.
    ///
    /// OOM is returned at this call, not later at [`Self::synchronize`]. Capture
    /// cannot include it. [`Self::alloc`] is `cudaMallocAsync`.
    pub fn malloc(&mut self, device: DeviceId, bytes: u64) -> Result<AllocId, SimError> {
        if bytes == 0 {
            return Err(SimError::Invalid {
                why: "zero-byte alloc",
            });
        }
        if self.capturing.is_some() {
            return Err(SimError::Invalid {
                why: "cannot capture alloc/free",
            });
        }
        let _gpu = self.profile.gpu(device)?;
        self.synchronize_device(device)?;
        self.reserve_now(device, bytes)
    }

    /// Immediate page-locked host allocation. Does not charge HBM.
    ///
    /// A kernel may not read this object until a copy has placed it on a
    /// device (or [`Self::alloc_host_mapped`] / [`Self::host_register_mapped`]).
    /// Capture cannot include host alloc.
    pub fn alloc_host_pinned(&mut self, bytes: u64) -> Result<AllocId, SimError> {
        self.insert_host(bytes, false, true, false, false)
    }

    /// Pageable host allocation (`malloc`). Pin it with [`Self::host_register`].
    pub fn alloc_host(&mut self, bytes: u64) -> Result<AllocId, SimError> {
        self.insert_host(bytes, true, false, false, false)
    }

    /// `cudaHostAllocMapped`: pinned, mapped, no HBM. A kernel may read it
    /// immediately; the memory term is host PCIe, not HBM.
    pub fn alloc_host_mapped(&mut self, bytes: u64) -> Result<AllocId, SimError> {
        self.insert_host(bytes, false, true, true, false)
    }

    /// `cudaMallocManaged`: pointer is live immediately, no HBM until a
    /// device first-touch or [`Self::prefetch`]. Default attach is
    /// [`MemAttach::Global`]. [`Self::alloc_managed_host`] is Host.
    ///
    /// Does not [`Self::synchronize_device`] (`cudaMalloc` does). Capture
    /// cannot include it. [`Self::free_sync`] is `cudaFree`.
    pub fn alloc_managed(&mut self, bytes: u64) -> Result<AllocId, SimError> {
        if bytes == 0 {
            return Err(SimError::Invalid {
                why: "zero-byte alloc",
            });
        }
        self.fail_if_capturing("cannot capture alloc/free")?;
        self.clock = self.clock.saturating_add(self.first_alloc_ns());
        let id = AllocId(self.next_alloc);
        self.next_alloc = self.next_alloc.saturating_add(1);
        let _prev = self.allocs.insert(
            id,
            Alloc {
                bytes,
                devices: Vec::new(),
                leases: 0,
                live: true,
                host_pinned: false,
                host_pageable: false,
                host_mapped: false,
                host_registered: false,
                managed: true,
                attach: Attach::Global,
                read_mostly: false,
                accessed_by: BTreeSet::new(),
                vmm_write_by: BTreeSet::new(),
                preferred: Preferred::None,
                vmm: false,
                vmm_maps: Vec::new(),
                pool: None,
                ipc_src: None,
                ipc_opens: 0,
                share_src: None,
                share_opens: 0,
            },
        );
        Ok(id)
    }

    /// `cudaMallocManaged(..., cudaMemAttachHost)`: CPU-exclusive until a later
    /// [`Self::stream_attach`].
    pub fn alloc_managed_host(&mut self, bytes: u64) -> Result<AllocId, SimError> {
        let id = self.alloc_managed(bytes)?;
        self.alloc_mut(id)?.attach = Attach::Host;
        Ok(id)
    }

    /// Current `cudaMemAttach*` visibility of a live managed allocation.
    pub fn mem_attach(&self, id: AllocId) -> Result<MemAttach, SimError> {
        let a = self.alloc_ref(id)?;
        if !a.live {
            return Err(SimError::Invalid { why: "freed" });
        }
        if !a.managed {
            return Err(SimError::Invalid { why: "not managed" });
        }
        Ok(match a.attach {
            Attach::Global => MemAttach::Global,
            Attach::Host => MemAttach::Host,
            Attach::Single(_) => MemAttach::Single,
        })
    }

    /// Whether a kernel on `stream` may touch `id` under the current attach.
    pub fn is_attached_to(&self, id: AllocId, stream: StreamId) -> Result<bool, SimError> {
        Ok(self.alloc_ref(id)?.device_attach_ok(stream))
    }

    /// Whether `alloc` is live unified memory (`cudaMallocManaged`).
    pub fn is_managed(&self, alloc: AllocId) -> Result<bool, SimError> {
        let a = self.alloc_ref(alloc)?;
        Ok(a.live && a.managed)
    }

    /// `cudaMemAdvise`. Host-synchronous. Capture cannot include it.
    ///
    /// [`MemAdvise::SetReadMostly`]: later [`Self::prefetch`] onto a second GPU
    /// keeps the first copy. A kernel write invalidates the extra copies.
    /// [`MemAdvise::SetAccessedBy`]: a kernel on `device` may read without
    /// migrating (interconnect, not local HBM). Writes still migrate.
    /// [`MemAdvise::SetPreferredLocation`]: a page already at that GPU stays
    /// there on a remote read (same interconnect billing; writes still migrate).
    /// [`MemAdvise::SetPreferredLocationHost`] does not skip kernel first-touch.
    pub fn mem_advise(
        &mut self,
        alloc: AllocId,
        advice: MemAdvise,
        device: DeviceId,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture mem advise")?;
        let a = self.alloc_ref(alloc)?;
        if !a.live || !a.managed {
            return Err(SimError::Invalid { why: "not managed" });
        }
        match advice {
            MemAdvise::SetReadMostly => {
                self.alloc_mut(alloc)?.read_mostly = true;
            }
            MemAdvise::UnsetReadMostly => {
                self.alloc_mut(alloc)?.read_mostly = false;
            }
            MemAdvise::SetAccessedBy => {
                let _gpu = self.profile.gpu(device)?;
                let _ins = self.alloc_mut(alloc)?.accessed_by.insert(device);
            }
            MemAdvise::UnsetAccessedBy => {
                let _gpu = self.profile.gpu(device)?;
                let _was = self.alloc_mut(alloc)?.accessed_by.remove(&device);
            }
            MemAdvise::SetPreferredLocation => {
                let _gpu = self.profile.gpu(device)?;
                self.alloc_mut(alloc)?.preferred = Preferred::Gpu(device);
            }
            MemAdvise::SetPreferredLocationHost => {
                self.alloc_mut(alloc)?.preferred = Preferred::Host;
            }
            MemAdvise::UnsetPreferredLocation => {
                self.alloc_mut(alloc)?.preferred = Preferred::None;
            }
        }
        self.clock = self.clock.saturating_add(self.first_alloc_ns().max(1));
        Ok(())
    }

    /// Whether [`MemAdvise::SetReadMostly`] is set.
    pub fn is_read_mostly(&self, alloc: AllocId) -> Result<bool, SimError> {
        let a = self.alloc_ref(alloc)?;
        Ok(a.live && a.managed && a.read_mostly)
    }

    /// Whether `device` has [`MemAdvise::SetAccessedBy`], [`Self::va_set_access`],
    /// or [`Self::pool_set_access`] on this alloc's mempool.
    pub fn is_accessed_by(&self, alloc: AllocId, device: DeviceId) -> Result<bool, SimError> {
        let a = self.alloc_ref(alloc)?;
        if !a.live {
            return Ok(false);
        }
        if a.accessed_by.contains(&device) && (a.managed || a.vmm) {
            return Ok(true);
        }
        Ok(self.pool_peer_ok(a, device))
    }

    /// Whether [`MemAdvise::SetPreferredLocation`] names `device`.
    pub fn is_preferred_location(
        &self,
        alloc: AllocId,
        device: DeviceId,
    ) -> Result<bool, SimError> {
        let a = self.alloc_ref(alloc)?;
        Ok(a.live && a.managed && matches!(a.preferred, Preferred::Gpu(p) if p == device))
    }

    /// Whether [`MemAdvise::SetPreferredLocationHost`] is set.
    pub fn is_preferred_host(&self, alloc: AllocId) -> Result<bool, SimError> {
        let a = self.alloc_ref(alloc)?;
        Ok(a.live && a.managed && matches!(a.preferred, Preferred::Host))
    }

    /// Drop one GPU's copy of a [`MemAdvise::SetReadMostly`] managed alloc.
    ///
    /// Host-synchronous. Capture cannot include it. The allocation stays live
    /// on every other device that still holds a copy. The last copy cannot be
    /// dropped this way ([`Self::free_sync`] / [`Self::prefetch_host`]).
    pub fn drop_managed_copy(&mut self, alloc: AllocId, device: DeviceId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture alloc/free")?;
        let _gpu = self.profile.gpu(device)?;
        let a = self.alloc_ref(alloc)?;
        if !a.live || !a.managed {
            return Err(SimError::Invalid { why: "not managed" });
        }
        if !a.read_mostly {
            return Err(SimError::Invalid {
                why: "not read-mostly",
            });
        }
        if a.leases > 0 {
            return Err(SimError::Leased { alloc });
        }
        if !a.devices.contains(&device) {
            return Err(SimError::NotResident { alloc, device });
        }
        if a.devices.len() < 2 {
            return Err(SimError::Invalid {
                why: "last managed copy",
            });
        }
        let bytes = a.bytes;
        self.refund_device(device, alloc, bytes)?;
        self.alloc_mut(alloc)?.devices.retain(|x| *x != device);
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cuMemAddressReserve`: a VA with no physical pages. Does not charge HBM.
    ///
    /// [`Self::va_map`] maps the whole VA; [`Self::va_map_range`] maps a span.
    /// Size and map offsets must be multiples of
    /// [`HardwareProfile::va_granularity_bytes`] (`0`/`1` = any size).
    /// Capture cannot include it.
    pub fn va_reserve(&mut self, bytes: u64) -> Result<AllocId, SimError> {
        if bytes == 0 {
            return Err(SimError::Invalid {
                why: "zero-byte alloc",
            });
        }
        self.check_va_align(bytes)?;
        self.fail_if_capturing("cannot capture alloc/free")?;
        self.clock = self.clock.saturating_add(self.first_alloc_ns());
        let id = AllocId(self.next_alloc);
        self.next_alloc = self.next_alloc.saturating_add(1);
        let _prev = self.allocs.insert(
            id,
            Alloc {
                bytes,
                devices: Vec::new(),
                leases: 0,
                live: true,
                host_pinned: false,
                host_pageable: false,
                host_mapped: false,
                host_registered: false,
                managed: false,
                attach: Attach::Global,
                read_mostly: false,
                accessed_by: BTreeSet::new(),
                vmm_write_by: BTreeSet::new(),
                preferred: Preferred::None,
                vmm: true,
                vmm_maps: Vec::new(),
                pool: None,
                ipc_src: None,
                ipc_opens: 0,
                share_src: None,
                share_opens: 0,
            },
        );
        Ok(id)
    }

    /// Whether `alloc` is a live reserved VA (`cuMemAddressReserve`).
    pub fn is_vmm(&self, alloc: AllocId) -> Result<bool, SimError> {
        let a = self.alloc_ref(alloc)?;
        Ok(a.live && a.vmm)
    }

    /// `cuMemCreate` + `cuMemMap` for the whole VA (local ReadWrite).
    ///
    /// Peer [`Self::va_set_access`] / [`Self::va_set_access_write`] is
    /// `cuMemSetAccess` PROT_READ / PROT_READWRITE.
    /// Equivalent to [`Self::va_map_range`] of `[0, bytes)`. Capture cannot include it.
    /// Split create/map is [`Self::va_create`] then [`Self::va_map_handle`].
    pub fn va_map(&mut self, id: AllocId, device: DeviceId) -> Result<(), SimError> {
        let bytes = self.alloc_ref(id)?.bytes;
        self.va_map_range(id, device, 0, bytes)
    }

    /// `cuMemCreate`: a device physical. Charges HBM. Does not map a VA.
    ///
    /// Host-synchronous. Capture cannot include it. Size must be granularity-aligned.
    /// Starts with one handle ref. [`Self::va_map_handle`] maps this handle into a
    /// reserved VA without a second HBM charge. [`Self::va_release_handle`] is
    /// `cuMemRelease` (allowed while mapped). HBM refunds when refs and maps are
    /// both 0.
    pub fn va_create(&mut self, device: DeviceId, bytes: u64) -> Result<MemHandleId, SimError> {
        if bytes == 0 {
            return Err(SimError::Invalid {
                why: "zero-byte alloc",
            });
        }
        self.check_va_align(bytes)?;
        self.fail_if_capturing("cannot capture alloc/free")?;
        let _gpu = self.profile.gpu(device)?;
        self.reserve_hbm(device, bytes)?;
        let ns = self.profile.gpu(device)?.alloc_overhead_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        let id = MemHandleId(self.next_handle);
        self.next_handle = self.next_handle.saturating_add(1);
        let _prev = self.mem_handles.insert(
            id,
            MemHandle {
                device,
                bytes,
                refs: 1,
                maps: 0,
                charged: true,
            },
        );
        Ok(id)
    }

    /// `cuMemMap` of an existing [`Self::va_create`] handle into a reserved VA.
    ///
    /// Host-synchronous. Does not charge HBM (the handle already holds the
    /// physicals). `device` must be the handle's device. The handle must still
    /// have a ref ([`Self::va_release_handle`] while mapped forbids further
    /// maps). Capture cannot include it.
    pub fn va_map_handle(
        &mut self,
        id: AllocId,
        device: DeviceId,
        offset: u64,
        handle: MemHandleId,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture alloc/free")?;
        let h = self.handle_ref(handle)?;
        if h.refs == 0 {
            return Err(SimError::Invalid {
                why: "handle released",
            });
        }
        if h.device != device {
            return Err(SimError::Invalid {
                why: "handle device mismatch",
            });
        }
        let bytes = h.bytes;
        let a = self.alloc_ref(id)?;
        if a.leases > 0 {
            return Err(SimError::Leased { alloc: id });
        }
        if !a.live || !a.vmm {
            return Err(SimError::Invalid { why: "not a VA" });
        }
        let end = offset.saturating_add(bytes);
        if end > a.bytes {
            return Err(SimError::Invalid {
                why: "range past VA",
            });
        }
        self.check_va_align(offset)?;
        if vmm_overlap(&a.vmm_maps, device, offset, bytes) {
            return Err(SimError::Invalid {
                why: "already mapped",
            });
        }
        let ns = self.profile.gpu(device)?.alloc_overhead_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        let a = self.alloc_mut(id)?;
        a.vmm_maps.push((device, offset, bytes));
        if !a.devices.contains(&device) {
            a.devices.push(device);
        }
        self.vmm_idle.retain(|&x| x != id);
        {
            let h = self.handle_mut(handle)?;
            h.maps = h.maps.saturating_add(1);
        }
        let _prev = self
            .vmm_handle_at
            .insert((id, device, offset, bytes), handle);
        Ok(())
    }

    /// `cuMemRelease`. Allowed while the physical is still mapped.
    ///
    /// Drops one handle ref. HBM refunds when refs and maps are both 0.
    /// Capture cannot include it. A released handle cannot be mapped again;
    /// [`Self::va_retain_handle`] on a still-mapped VA restores a ref.
    pub fn va_release_handle(&mut self, handle: MemHandleId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture alloc/free")?;
        let refs = self.handle_ref(handle)?.refs;
        if refs == 0 {
            return Err(SimError::Invalid {
                why: "handle released",
            });
        }
        self.handle_mut(handle)?.refs = refs.saturating_sub(1);
        self.clock = self.clock.saturating_add(1);
        self.maybe_refund_handle(handle)
    }

    /// `cuMemRetainAllocationHandle` at a mapped `(device, offset)` span.
    ///
    /// Host-synchronous. Capture cannot include it. An explicit handle's ref
    /// count increments (a released-but-mapped handle is restored to one ref).
    /// A combined [`Self::va_map`] / [`Self::va_map_range`] span is promoted
    /// so later unmaps do not refund until [`Self::va_release_handle`].
    pub fn va_retain_handle(
        &mut self,
        id: AllocId,
        device: DeviceId,
        offset: u64,
    ) -> Result<MemHandleId, SimError> {
        self.fail_if_capturing("cannot capture alloc/free")?;
        let _gpu = self.profile.gpu(device)?;
        let a = self.alloc_ref(id)?;
        if !a.live || !a.vmm {
            return Err(SimError::Invalid { why: "not a VA" });
        }
        let bytes = a
            .vmm_maps
            .iter()
            .find(|&&(d, o, _)| d == device && o == offset)
            .map(|&(_, _, n)| n)
            .ok_or(SimError::Invalid { why: "no such map" })?;
        let key = (id, device, offset, bytes);
        if let Some(&h) = self.vmm_handle_at.get(&key) {
            let refs = self.handle_ref(h)?.refs;
            self.handle_mut(h)?.refs = refs.saturating_add(1);
            let ns = self.profile.gpu(device)?.alloc_overhead_ns.max(1);
            self.clock = self.clock.saturating_add(ns);
            return Ok(h);
        }
        let ns = self.profile.gpu(device)?.alloc_overhead_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        let h = MemHandleId(self.next_handle);
        self.next_handle = self.next_handle.saturating_add(1);
        let _prev = self.mem_handles.insert(
            h,
            MemHandle {
                device,
                bytes,
                refs: 1,
                maps: 1,
                charged: true,
            },
        );
        let _old = self.vmm_handle_at.insert(key, h);
        Ok(h)
    }

    /// Whether `handle` still has a `cuMemCreate` / retain ref.
    pub fn is_handle_live(&self, handle: MemHandleId) -> Result<bool, SimError> {
        Ok(self.handle_ref(handle)?.refs > 0)
    }

    /// How many VA maps currently hold `handle`.
    pub fn handle_maps(&self, handle: MemHandleId) -> Result<u32, SimError> {
        Ok(self.handle_ref(handle)?.maps)
    }

    /// Outstanding `cuMemCreate` / [`Self::va_retain_handle`] refs.
    pub fn handle_refs(&self, handle: MemHandleId) -> Result<u32, SimError> {
        Ok(self.handle_ref(handle)?.refs)
    }

    /// `cuMulticastCreate`: an NVLS multicast object. Does not charge HBM.
    ///
    /// Host-synchronous. Capture cannot include it. `bytes` must be
    /// [`HardwareProfile::multicast_aligned`]. `num_devices` is the team size
    /// (`cuMulticastAddDevice` must fill it before bind/map). PCIe-only and
    /// 1-GPU profiles still create; bind/map fail without an NVLink clique.
    pub fn multicast_create(
        &mut self,
        bytes: u64,
        num_devices: u32,
    ) -> Result<MulticastId, SimError> {
        if bytes == 0 {
            return Err(SimError::Invalid {
                why: "zero-byte alloc",
            });
        }
        if num_devices < 2 {
            return Err(SimError::Invalid {
                why: "multicast needs NVLink",
            });
        }
        self.check_mc_align(bytes)?;
        self.fail_if_capturing("cannot capture alloc/free")?;
        self.clock = self.clock.saturating_add(self.first_alloc_ns());
        let id = MulticastId(self.next_mc);
        self.next_mc = self.next_mc.saturating_add(1);
        let _prev = self.multicasts.insert(
            id,
            Multicast {
                bytes,
                n_dev: num_devices,
                devices: Vec::new(),
                binds: BTreeMap::new(),
                maps: 0,
            },
        );
        Ok(id)
    }

    /// `cuMulticastAddDevice`. Host-synchronous. Capture cannot include it.
    ///
    /// Must run before bind/map. Duplicate add is Invalid. The completed team
    /// must be an NVLink clique.
    pub fn multicast_add_device(
        &mut self,
        mc: MulticastId,
        device: DeviceId,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture alloc/free")?;
        let _gpu = self.profile.gpu(device)?;
        let done = {
            let obj = self.mc_mut(mc)?;
            if !obj.binds.is_empty() || obj.maps > 0 {
                return Err(SimError::Invalid {
                    why: "add all devices first",
                });
            }
            if obj.devices.contains(&device) {
                return Err(SimError::Invalid {
                    why: "already added",
                });
            }
            if u32::try_from(obj.devices.len()).unwrap_or(u32::MAX) >= obj.n_dev {
                return Err(SimError::Invalid { why: "team full" });
            }
            obj.devices.push(device);
            u32::try_from(obj.devices.len()).unwrap_or(0) == obj.n_dev
        };
        self.clock = self.clock.saturating_add(1);
        if done {
            let team = self.mc_ref(mc)?.devices.clone();
            nvlink_clique(&self.profile, &team)?;
        }
        Ok(())
    }

    /// `cuMulticastBindMem` of a [`Self::va_create`] handle on `device`.
    ///
    /// Host-synchronous. Capture cannot include it. The handle's device and
    /// size must match. All devices must already be added. Dest HBM is the
    /// handle (already charged); bind does not charge again.
    pub fn multicast_bind_mem(
        &mut self,
        mc: MulticastId,
        device: DeviceId,
        handle: MemHandleId,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture alloc/free")?;
        let h = self.handle_ref(handle)?;
        if h.refs == 0 {
            return Err(SimError::Invalid {
                why: "handle released",
            });
        }
        if h.device != device {
            return Err(SimError::Invalid {
                why: "handle device mismatch",
            });
        }
        let h_bytes = h.bytes;
        let (n_dev, n_added, in_team, already, size, team) = {
            let obj = self.mc_ref(mc)?;
            (
                obj.n_dev,
                u32::try_from(obj.devices.len()).unwrap_or(0),
                obj.devices.contains(&device),
                obj.binds.contains_key(&device),
                obj.bytes,
                obj.devices.clone(),
            )
        };
        if n_added != n_dev {
            return Err(SimError::Invalid {
                why: "add all devices first",
            });
        }
        if !in_team {
            return Err(SimError::Invalid {
                why: "device not in team",
            });
        }
        if already {
            return Err(SimError::Invalid {
                why: "already bound",
            });
        }
        if h_bytes != size {
            return Err(SimError::Invalid {
                why: "handle size mismatch",
            });
        }
        nvlink_clique(&self.profile, &team)?;
        let _prev = self.mc_mut(mc)?.binds.insert(device, handle);
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cuMemMap` of a multicast object into a reserved VA (no extra HBM).
    ///
    /// Host-synchronous. Capture cannot include it. Every team device must
    /// already be bound. Kernel writes to this VA are billed as one NVLS hop
    /// and occupy compute (not a copy engine).
    pub fn va_map_multicast(
        &mut self,
        id: AllocId,
        device: DeviceId,
        offset: u64,
        mc: MulticastId,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture alloc/free")?;
        let (bytes, n_dev, n_binds, in_team, team) = {
            let obj = self.mc_ref(mc)?;
            (
                obj.bytes,
                obj.n_dev,
                u32::try_from(obj.binds.len()).unwrap_or(0),
                obj.devices.contains(&device),
                obj.devices.clone(),
            )
        };
        if n_binds != n_dev {
            return Err(SimError::Invalid {
                why: "bind all devices first",
            });
        }
        if !in_team {
            return Err(SimError::Invalid {
                why: "device not in team",
            });
        }
        nvlink_clique(&self.profile, &team)?;
        let a = self.alloc_ref(id)?;
        if a.leases > 0 {
            return Err(SimError::Leased { alloc: id });
        }
        if !a.live || !a.vmm {
            return Err(SimError::Invalid { why: "not a VA" });
        }
        let end = offset.saturating_add(bytes);
        if end > a.bytes {
            return Err(SimError::Invalid {
                why: "range past VA",
            });
        }
        self.check_va_align(offset)?;
        if vmm_overlap(&a.vmm_maps, device, offset, bytes) {
            return Err(SimError::Invalid {
                why: "already mapped",
            });
        }
        if self.mc_vas.contains_key(&id) {
            return Err(SimError::Invalid {
                why: "already mapped",
            });
        }
        let ns = self.profile.gpu(device)?.alloc_overhead_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        let a = self.alloc_mut(id)?;
        a.vmm_maps.push((device, offset, bytes));
        if !a.devices.contains(&device) {
            a.devices.push(device);
        }
        self.vmm_idle.retain(|&x| x != id);
        let maps = self.mc_ref(mc)?.maps;
        self.mc_mut(mc)?.maps = maps.saturating_add(1);
        let _prev = self.mc_vas.insert(id, mc);
        Ok(())
    }

    /// Whether `id` is a reserved VA mapped with [`Self::va_map_multicast`].
    #[must_use]
    pub fn is_multicast_va(&self, id: AllocId) -> bool {
        self.mc_vas.contains_key(&id)
    }

    /// How many devices currently have [`Self::multicast_bind_mem`] on `mc`.
    pub fn multicast_binds(&self, mc: MulticastId) -> Result<u32, SimError> {
        let n = self.mc_ref(mc)?.binds.len();
        Ok(u32::try_from(n).unwrap_or(u32::MAX))
    }

    /// NVLS kernel store: bind `id`'s VMM maps on `src` and `dests`, then write.
    ///
    /// Each device must already have a whole-VA [`Self::va_map`] / handle map
    /// (dest HBM is that physical). Enqueues a compute kernel whose duration is
    /// one NVLink hop of `id`'s bytes, not `dests.len()` sequential D2Ds.
    /// Capture cannot include the create/bind/map; the kernel may be captured
    /// later. `dests` must be nonempty and not include `src`.
    pub fn multicast_store(
        &mut self,
        src: DeviceId,
        id: AllocId,
        dests: &[DeviceId],
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        self.fail_if_capturing("cannot capture alloc/free")?;
        let bytes = self.alloc_ref(id)?.bytes;
        let team = multicast_team(src, dests)?;
        nvlink_clique(&self.profile, &team)?;
        self.require_whole_maps(id, &team)?;
        let n = u32::try_from(team.len()).unwrap_or(0);
        let mc = self.multicast_create(bytes, n)?;
        for d in &team {
            self.multicast_add_device(mc, *d)?;
        }
        for d in team {
            let h = self.handle_for_bind(id, d)?;
            self.multicast_bind_mem(mc, d, h)?;
        }
        let va = self.va_reserve(bytes)?;
        self.va_map_multicast(va, src, 0, mc)?;
        self.kernel(src, KernelKind::other(0, bytes), &[id], &[va], stream)
    }

    fn require_whole_maps(&self, id: AllocId, team: &[DeviceId]) -> Result<(), SimError> {
        let a = self.alloc_ref(id)?;
        if !a.live || !a.vmm {
            return Err(SimError::Invalid { why: "not a VA" });
        }
        let bytes = a.bytes;
        for &d in team {
            if !vmm_covers(&a.vmm_maps, d, 0, bytes) {
                return Err(SimError::NotResident {
                    alloc: id,
                    device: d,
                });
            }
            let page = a
                .vmm_maps
                .iter()
                .find(|&&(dev, off, _)| dev == d && off == 0)
                .map(|&(_, _, n)| n);
            if page != Some(bytes) {
                return Err(SimError::Invalid {
                    why: "multicast needs whole-VA maps",
                });
            }
        }
        Ok(())
    }

    fn handle_for_bind(&mut self, id: AllocId, device: DeviceId) -> Result<MemHandleId, SimError> {
        let bytes = self.alloc_ref(id)?.bytes;
        let key = (id, device, 0, bytes);
        if let Some(&h) = self.vmm_handle_at.get(&key) {
            return Ok(h);
        }
        self.va_retain_handle(id, device, 0)
    }

    /// Map `[offset, offset+bytes)` of a reserved VA onto `device`.
    ///
    /// Host-synchronous. Charges `bytes` of HBM. Overlapping maps on the same
    /// device fail. Offset and `bytes` must be granularity-aligned.
    /// [`Self::kernel`] needs the full VA covered; [`Self::kernel_bufs`]
    /// may run on this span. A hole is [`SimError::NotResident`] for that API.
    /// Capture cannot include it.
    pub fn va_map_range(
        &mut self,
        id: AllocId,
        device: DeviceId,
        offset: u64,
        bytes: u64,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture alloc/free")?;
        let _gpu = self.profile.gpu(device)?;
        let a = self.alloc_ref(id)?;
        if a.leases > 0 {
            return Err(SimError::Leased { alloc: id });
        }
        if !a.live || !a.vmm {
            return Err(SimError::Invalid { why: "not a VA" });
        }
        if bytes == 0 {
            return Err(SimError::Invalid {
                why: "zero-byte alloc",
            });
        }
        let end = offset.saturating_add(bytes);
        if end > a.bytes {
            return Err(SimError::Invalid {
                why: "range past VA",
            });
        }
        self.check_va_align(offset)?;
        self.check_va_align(bytes)?;
        if vmm_overlap(&a.vmm_maps, device, offset, bytes) {
            return Err(SimError::Invalid {
                why: "already mapped",
            });
        }
        self.reserve_hbm(device, bytes)?;
        let ns = self.profile.gpu(device)?.alloc_overhead_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        let a = self.alloc_mut(id)?;
        a.vmm_maps.push((device, offset, bytes));
        if !a.devices.contains(&device) {
            a.devices.push(device);
        }
        self.vmm_idle.retain(|&x| x != id);
        Ok(())
    }

    /// `cuMemUnmap` + `cuMemRelease` for every physical on this VA.
    ///
    /// Host-synchronous: in-flight kernels using this pointer complete first.
    pub fn va_unmap(&mut self, id: AllocId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture alloc/free")?;
        self.synchronize()?;
        let a = self.alloc_ref(id)?;
        if a.leases > 0 {
            return Err(SimError::Leased { alloc: id });
        }
        if !a.live || !a.vmm {
            return Err(SimError::Invalid { why: "not a VA" });
        }
        let maps = a.vmm_maps.clone();
        if maps.is_empty() {
            return Err(SimError::Invalid { why: "not mapped" });
        }
        for (d, o, n) in maps {
            self.drop_vmm_physical(id, d, o, n)?;
        }
        let a = self.alloc_mut(id)?;
        a.vmm_maps.clear();
        a.devices.clear();
        a.accessed_by.clear();
        a.vmm_write_by.clear();
        let _gone = self.mc_vas.remove(&id);
        Ok(())
    }

    /// Unmap one exact `(device, offset, bytes)` physical. The VA stays reserved.
    pub fn va_unmap_range(
        &mut self,
        id: AllocId,
        device: DeviceId,
        offset: u64,
        bytes: u64,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture alloc/free")?;
        self.synchronize()?;
        let a = self.alloc_ref(id)?;
        if a.leases > 0 {
            return Err(SimError::Leased { alloc: id });
        }
        if !a.live || !a.vmm {
            return Err(SimError::Invalid { why: "not a VA" });
        }
        let pos = a
            .vmm_maps
            .iter()
            .position(|&(d, o, n)| d == device && o == offset && n == bytes);
        let Some(i) = pos else {
            return Err(SimError::Invalid { why: "no such map" });
        };
        self.drop_vmm_physical(id, device, offset, bytes)?;
        let a = self.alloc_mut(id)?;
        let _gone = a.vmm_maps.remove(i);
        if !a.vmm_maps.iter().any(|&(d, _, _)| d == device) {
            a.devices.retain(|d| *d != device);
        }
        if a.vmm_maps.is_empty() {
            a.accessed_by.clear();
            a.vmm_write_by.clear();
            let _gone = self.mc_vas.remove(&id);
        }
        Ok(())
    }

    /// `cuMemSetAccess` PROT_READ on `device` for a mapped VMM VA.
    ///
    /// Host-synchronous. Does not charge dest HBM. A kernel on `device` may
    /// read physicals that live on another GPU (interconnect). Writes still
    /// need a local map unless [`Self::va_set_access_write`]. Capture cannot
    /// include it. Needs a topology link and directed peer access from the
    /// home GPU, same as D2D. Downgrades a prior PROT_READWRITE on `device`.
    pub fn va_set_access(&mut self, id: AllocId, device: DeviceId) -> Result<(), SimError> {
        self.va_set_access_inner(id, device, false)
    }

    /// `cuMemSetAccess` PROT_READWRITE on `device` for a mapped VMM VA.
    ///
    /// Host-synchronous. Does not charge dest HBM. A kernel on `device` may
    /// read **and write** home physicals (interconnect), same class as
    /// [`Self::pool_set_access`]. Capture cannot include it.
    pub fn va_set_access_write(&mut self, id: AllocId, device: DeviceId) -> Result<(), SimError> {
        self.va_set_access_inner(id, device, true)
    }

    fn va_set_access_inner(
        &mut self,
        id: AllocId,
        device: DeviceId,
        write: bool,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture alloc/free")?;
        let _gpu = self.profile.gpu(device)?;
        let a = self.alloc_ref(id)?;
        if !a.live || !a.vmm {
            return Err(SimError::Invalid { why: "not a VA" });
        }
        let Some(owner) = a.vmm_home() else {
            return Err(SimError::Invalid { why: "not mapped" });
        };
        if owner != device {
            let _link = self.profile.link(Some(owner), Some(device))?;
            if !self.peer_access(owner, device) {
                return Err(SimError::PeerDisabled {
                    src: owner,
                    dst: device,
                });
            }
        }
        {
            let a = self.alloc_mut(id)?;
            let _ins = a.accessed_by.insert(device);
            if write {
                let _w = a.vmm_write_by.insert(device);
            } else {
                let _gone = a.vmm_write_by.remove(&device);
            }
        }
        self.clock = self.clock.saturating_add(self.first_alloc_ns().max(1));
        Ok(())
    }

    /// Drop [`Self::va_set_access`] / [`Self::va_set_access_write`] for `device`.
    /// Host-synchronous.
    pub fn va_unset_access(&mut self, id: AllocId, device: DeviceId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture alloc/free")?;
        let _gpu = self.profile.gpu(device)?;
        let a = self.alloc_ref(id)?;
        if !a.live || !a.vmm {
            return Err(SimError::Invalid { why: "not a VA" });
        }
        {
            let a = self.alloc_mut(id)?;
            let _was = a.accessed_by.remove(&device);
            let _w = a.vmm_write_by.remove(&device);
        }
        self.clock = self.clock.saturating_add(self.first_alloc_ns().max(1));
        Ok(())
    }

    /// Whether `device` has [`Self::va_set_access_write`] on this VMM VA.
    pub fn is_va_write_accessed_by(
        &self,
        alloc: AllocId,
        device: DeviceId,
    ) -> Result<bool, SimError> {
        let a = self.alloc_ref(alloc)?;
        Ok(a.live && a.vmm && a.vmm_write_by.contains(&device))
    }

    /// Mapped bytes of `alloc` currently charged on `device`.
    pub fn vmm_mapped_bytes(&self, alloc: AllocId, device: DeviceId) -> Result<u64, SimError> {
        let a = self.alloc_ref(alloc)?;
        if !a.live || !a.vmm {
            return Err(SimError::Invalid { why: "not a VA" });
        }
        Ok(a.vmm_maps
            .iter()
            .filter(|(d, _, _)| *d == device)
            .fold(0u64, |acc, (_, _, n)| acc.saturating_add(*n)))
    }

    /// `cuMemAddressFree`. Must already be unmapped and not leased.
    pub fn va_free(&mut self, id: AllocId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture alloc/free")?;
        let a = self.alloc_ref(id)?;
        if a.leases > 0 {
            return Err(SimError::Leased { alloc: id });
        }
        if !a.live || !a.vmm {
            return Err(SimError::UnknownAlloc { alloc: id });
        }
        if !a.devices.is_empty() {
            return Err(SimError::Invalid {
                why: "VA still mapped",
            });
        }
        self.vmm_idle.retain(|&x| x != id);
        self.alloc_mut(id)?.live = false;
        Ok(())
    }

    fn take_idle_va(&mut self, bytes: u64) -> Option<AllocId> {
        self.vmm_idle.retain(|&id| {
            self.allocs
                .get(&id)
                .is_some_and(|a| a.live && a.vmm && a.devices.is_empty())
        });
        let pos = self
            .vmm_idle
            .iter()
            .position(|&id| self.allocs.get(&id).is_some_and(|a| a.bytes == bytes))?;
        Some(self.vmm_idle.remove(pos))
    }

    fn park_idle_va(&mut self, id: AllocId) {
        if !self.vmm_idle.contains(&id) {
            self.vmm_idle.push(id);
        }
    }

    /// Map a previously [`Self::va_release`]d VA of `bytes`, or [`Self::va_reserve`].
    ///
    /// Unmap keeps the pointer (vLLM-style). The next miss remaps without another
    /// `cuMemAddressReserve`. Capture cannot include it.
    pub fn va_acquire(&mut self, device: DeviceId, bytes: u64) -> Result<AllocId, SimError> {
        if let Some(id) = self.take_idle_va(bytes) {
            if let Err(e) = self.va_map(id, device) {
                self.park_idle_va(id);
                return Err(e);
            }
            return Ok(id);
        }
        let id = self.va_reserve(bytes)?;
        if let Err(e) = self.va_map(id, device) {
            self.park_idle_va(id);
            return Err(e);
        }
        Ok(id)
    }

    /// [`Self::va_acquire`] mapping `page` physicals that cover the VA (vLLM KV analog).
    ///
    /// `page >= bytes` is [`Self::va_acquire`]. [`Self::kernel`] still needs the
    /// whole VA covered; this splits `cuMemMap` so each block pays
    /// `alloc_overhead_ns`. A paged working set that leaves holes uses
    /// [`Self::va_map_range`] plus [`Self::kernel_bufs`].
    pub fn va_acquire_paged(
        &mut self,
        device: DeviceId,
        bytes: u64,
        page: u64,
    ) -> Result<AllocId, SimError> {
        let page = page.max(1);
        if page >= bytes {
            return self.va_acquire(device, bytes);
        }
        let id = match self.take_idle_va(bytes) {
            Some(id) => id,
            None => self.va_reserve(bytes)?,
        };
        if let Err(e) = self.map_va_pages(id, device, page) {
            self.park_idle_va(id);
            return Err(e);
        }
        Ok(id)
    }

    fn map_va_pages(&mut self, id: AllocId, device: DeviceId, page: u64) -> Result<(), SimError> {
        let total = self.alloc_ref(id)?.bytes;
        let mut off = 0u64;
        while off < total {
            let n = page.min(total.saturating_sub(off));
            if let Err(e) = self.va_map_range(id, device, off, n) {
                if off > 0 {
                    match self.va_unmap(id) {
                        Ok(()) => {}
                        Err(_u) => {}
                    }
                }
                return Err(e);
            }
            off = off.saturating_add(n);
        }
        Ok(())
    }

    /// [`Self::va_unmap`] then keep the VA for [`Self::va_acquire`]. Does not
    /// [`Self::va_free`]. Already-idle ids are a no-op. Capture cannot include it.
    pub fn va_release(&mut self, id: AllocId) -> Result<(), SimError> {
        if self.vmm_idle.contains(&id) {
            return Ok(());
        }
        let a = self.alloc_ref(id)?;
        if !a.live || !a.vmm {
            return Err(SimError::Invalid { why: "not a VA" });
        }
        if !a.devices.is_empty() {
            self.va_unmap(id)?;
        }
        self.park_idle_va(id);
        Ok(())
    }

    /// Unmapped live VAs waiting for [`Self::va_acquire`].
    #[must_use]
    pub fn vmm_idle_len(&self) -> usize {
        self.vmm_idle.len()
    }

    fn first_alloc_ns(&self) -> u64 {
        self.profile
            .gpus
            .first()
            .map(|g| g.alloc_overhead_ns)
            .unwrap_or(1)
            .max(1)
    }

    fn check_va_align(&self, n: u64) -> Result<(), SimError> {
        if self.profile.va_aligned(n) {
            Ok(())
        } else {
            Err(SimError::Invalid {
                why: "unaligned VA",
            })
        }
    }

    fn check_mc_align(&self, n: u64) -> Result<(), SimError> {
        if self.profile.multicast_aligned(n) {
            Ok(())
        } else {
            Err(SimError::Invalid {
                why: "unaligned multicast",
            })
        }
    }

    /// `cudaMemPrefetchAsync` onto `device`. Stream-ordered; migrates, does not
    /// replicate unless [`MemAdvise::SetReadMostly`]. Already-local pages pay
    /// 1 ns and skip the copy engine.
    ///
    /// Capture may record it (it is a memcpy). A kernel that first-touches
    /// managed memory calls this on the same stream before the GEMM.
    pub fn prefetch(
        &mut self,
        device: DeviceId,
        alloc: AllocId,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        let (bytes, src) = self.managed_move_src(alloc, Some(device))?;
        self.memcpy(
            device,
            MemcpyOp {
                src,
                dst: Place::Device(device),
                alloc,
                bytes,
                offset: 0,
            },
            stream,
        )
    }

    /// `cudaMemPrefetchAsync(..., cudaCpuDeviceId)`. Pages leave HBM.
    ///
    /// Submit on `device`'s `stream` (the stream that owns the work). Already
    /// on the host is a 1 ns no-op on that stream.
    pub fn prefetch_host(
        &mut self,
        device: DeviceId,
        alloc: AllocId,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        let (bytes, src) = self.managed_move_src(alloc, None)?;
        let submit = match src {
            Place::Device(d) => d,
            Place::Host | Place::HostPinned => device,
        };
        self.memcpy(
            submit,
            MemcpyOp {
                src,
                dst: Place::HostPinned,
                alloc,
                bytes,
                offset: 0,
            },
            stream,
        )
    }

    /// Whether `alloc` is live in page-locked host memory.
    pub fn is_host_pinned(&self, alloc: AllocId) -> Result<bool, SimError> {
        let a = self.alloc_ref(alloc)?;
        Ok(a.live && a.host_pinned)
    }

    /// Whether `alloc` is live pageable host memory (not yet [`Self::host_register`]).
    pub fn is_host_pageable(&self, alloc: AllocId) -> Result<bool, SimError> {
        let a = self.alloc_ref(alloc)?;
        Ok(a.live && a.host_pageable)
    }

    /// Whether `alloc` is mapped into the device VA (zero-copy host).
    pub fn is_host_mapped(&self, alloc: AllocId) -> Result<bool, SimError> {
        let a = self.alloc_ref(alloc)?;
        Ok(a.live && a.host_mapped)
    }

    /// `cudaHostRegister`: pin pageable host for DMA. Host-synchronous (mlock).
    ///
    /// Capture cannot include it. Already-pinned ids fail. [`Self::host_register_mapped`]
    /// also maps the pointer so a kernel can read it without H2D.
    pub fn host_register(&mut self, id: AllocId) -> Result<(), SimError> {
        self.host_register_flags(id, false)
    }

    /// `cudaHostRegisterMapped`: pin and map pageable host. Kernels may read it
    /// over PCIe without a device copy.
    pub fn host_register_mapped(&mut self, id: AllocId) -> Result<(), SimError> {
        self.host_register_flags(id, true)
    }

    /// `cudaHostUnregister`. Only ids from [`Self::host_register`]. Must not be leased
    /// or resident on a device.
    pub fn host_unregister(&mut self, id: AllocId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture host register")?;
        self.synchronize()?;
        let a = self.alloc_ref(id)?;
        if a.leases > 0 {
            return Err(SimError::Leased { alloc: id });
        }
        if !a.live || !a.host_registered {
            return Err(SimError::Invalid {
                why: "not registered",
            });
        }
        if !a.devices.is_empty() {
            return Err(SimError::Invalid {
                why: "host-registered still resident on a device",
            });
        }
        let bytes = a.bytes;
        self.refund_pin(bytes);
        let a = self.alloc_mut(id)?;
        a.host_pinned = false;
        a.host_mapped = false;
        a.host_registered = false;
        a.host_pageable = true;
        Ok(())
    }

    /// Drop pageable host memory from [`Self::alloc_host`]. Unregister first if pinned.
    pub fn free_host(&mut self, id: AllocId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture alloc/free")?;
        self.synchronize()?;
        let a = self.alloc_ref(id)?;
        if a.leases > 0 {
            return Err(SimError::Leased { alloc: id });
        }
        if !a.live || !a.host_pageable || a.host_pinned {
            return Err(SimError::UnknownAlloc { alloc: id });
        }
        if !a.devices.is_empty() {
            return Err(SimError::Invalid {
                why: "host still resident on a device",
            });
        }
        let a = self.alloc_mut(id)?;
        a.live = false;
        a.host_pageable = false;
        Ok(())
    }

    /// Drop a host-pinned allocation that is not resident on any GPU.
    ///
    /// Host-synchronous (`cudaFreeHost`): in-flight kernels using this pointer
    /// complete first. Registered ids need [`Self::host_unregister`].
    pub fn free_host_pinned(&mut self, id: AllocId) -> Result<(), SimError> {
        if self.capturing.is_some() {
            return Err(SimError::Invalid {
                why: "cannot capture alloc/free",
            });
        }
        self.synchronize()?;
        let a = self.alloc_ref(id)?;
        if a.leases > 0 {
            return Err(SimError::Leased { alloc: id });
        }
        if !a.live || !a.host_pinned || a.host_registered {
            return Err(SimError::UnknownAlloc { alloc: id });
        }
        if !a.devices.is_empty() {
            return Err(SimError::Invalid {
                why: "host-pinned still resident on a device",
            });
        }
        let bytes = a.bytes;
        self.refund_pin(bytes);
        let a = self.alloc_mut(id)?;
        a.live = false;
        a.host_pinned = false;
        a.host_mapped = false;
        a.host_pageable = false;
        Ok(())
    }

    /// Stream-ordered free (`cudaFreeAsync`). Illegal while a kernel lease is held.
    ///
    /// Capture records a graph mem free node.
    pub fn free(
        &mut self,
        device: DeviceId,
        id: AllocId,
        stream: StreamId,
    ) -> Result<(), SimError> {
        let _op = self.submit(device, stream, Kind::Free { id })?;
        Ok(())
    }

    /// `cudaIpcGetMemHandle` of a live device allocation. Host-synchronous.
    /// Capture cannot include it. The same alloc returns the same handle.
    pub fn ipc_get(&mut self, id: AllocId) -> Result<IpcHandleId, SimError> {
        self.fail_if_capturing("cannot capture ipc")?;
        let d = {
            let a = self.alloc_ref(id)?;
            if !a.live
                || a.managed
                || a.vmm
                || a.host_pinned
                || a.host_pageable
                || a.ipc_src.is_some()
                || a.share_src.is_some()
                || a.pool.is_some()
                || a.devices.is_empty()
            {
                return Err(SimError::Invalid {
                    why: "not device ipc",
                });
            }
            a.devices.first().copied().ok_or(SimError::Invalid {
                why: "not device ipc",
            })?
        };
        if let Some(h) = self
            .ipc_handles
            .iter()
            .find_map(|(&h, src)| (*src == id).then_some(h))
        {
            return Ok(h);
        }
        let ns = self.profile.gpu(d)?.alloc_overhead_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        let h = IpcHandleId(self.next_ipc);
        self.next_ipc = self.next_ipc.saturating_add(1);
        let _prev = self.ipc_handles.insert(h, id);
        Ok(h)
    }

    /// `cudaIpcOpenMemHandle` on `device`. Alias shares the source physicals
    /// (no extra HBM). `device` must already hold the source.
    pub fn ipc_open(&mut self, device: DeviceId, handle: IpcHandleId) -> Result<AllocId, SimError> {
        self.fail_if_capturing("cannot capture ipc")?;
        let src = *self
            .ipc_handles
            .get(&handle)
            .ok_or(SimError::Invalid { why: "unknown ipc" })?;
        let (bytes, devices, opens) = {
            let a = self.alloc_ref(src)?;
            if !a.live {
                return Err(SimError::Invalid { why: "freed" });
            }
            if !a.devices.contains(&device) {
                return Err(SimError::Invalid {
                    why: "ipc device mismatch",
                });
            }
            (a.bytes, a.devices.clone(), a.ipc_opens)
        };
        let ns = self.profile.gpu(device)?.alloc_overhead_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        self.alloc_mut(src)?.ipc_opens = opens.saturating_add(1);
        let id = AllocId(self.next_alloc);
        self.next_alloc = self.next_alloc.saturating_add(1);
        let _prev = self.allocs.insert(
            id,
            Alloc {
                bytes,
                devices,
                leases: 0,
                live: true,
                host_pinned: false,
                host_pageable: false,
                host_mapped: false,
                host_registered: false,
                managed: false,
                attach: Attach::Global,
                read_mostly: false,
                accessed_by: BTreeSet::new(),
                vmm_write_by: BTreeSet::new(),
                preferred: Preferred::None,
                vmm: false,
                vmm_maps: Vec::new(),
                pool: None,
                ipc_src: Some(src),
                ipc_opens: 0,
                share_src: None,
                share_opens: 0,
            },
        );
        Ok(id)
    }

    /// `cudaIpcCloseMemHandle`. Does not refund source HBM.
    pub fn ipc_close(&mut self, id: AllocId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture ipc")?;
        self.drop_ipc_import(id)
    }

    /// Whether `id` is a live [`Self::ipc_open`] alias.
    pub fn is_ipc_import(&self, id: AllocId) -> Result<bool, SimError> {
        let a = self.alloc_ref(id)?;
        Ok(a.live && a.ipc_src.is_some())
    }

    /// `cudaFree`: wait every GPU that holds the pointer, then it is gone on
    /// all of them. [`Self::free`] is `cudaFreeAsync`. Host-pinned ids are
    /// [`SimError::UnknownAlloc`] (`cudaFreeHost` is [`Self::free_host_pinned`]).
    pub fn free_sync(&mut self, id: AllocId) -> Result<(), SimError> {
        if self.capturing.is_some() {
            return Err(SimError::Invalid {
                why: "cannot capture alloc/free",
            });
        }
        let a = self.alloc_ref(id)?;
        if a.host_pinned || a.host_pageable || a.vmm {
            return Err(SimError::UnknownAlloc { alloc: id });
        }
        let holders = a.devices.clone();
        if holders.is_empty() {
            // Stream-ordered `alloc` may not have started; drain so OOM/residency
            // is resolved before the host returns.
            self.synchronize()?;
        } else {
            for d in holders {
                self.synchronize_device(d)?;
            }
        }
        let a = self.alloc_ref(id)?;
        if a.leases > 0 {
            return Err(SimError::Leased { alloc: id });
        }
        if a.ipc_src.is_some() {
            return self.drop_ipc_import(id);
        }
        if a.share_src.is_some() {
            return self.drop_share_import(id);
        }
        if a.ipc_opens > 0 {
            return Err(SimError::Invalid { why: "ipc mapped" });
        }
        if a.share_opens > 0 {
            return Err(SimError::Invalid {
                why: "share mapped",
            });
        }
        if !a.live || a.host_pinned {
            return Err(SimError::UnknownAlloc { alloc: id });
        }
        let devices = a.devices.clone();
        let bytes = a.bytes;
        for d in &devices {
            self.refund_device(*d, id, bytes)?;
        }
        let a = self.alloc_mut(id)?;
        a.devices.clear();
        a.live = false;
        Ok(())
    }

    /// Asynchronous copy (`cudaMemcpyAsync`) when both ends are device or pinned.
    ///
    /// Pageable host (`Place::Host`) is host-synchronous: the driver bounces
    /// through pinned staging, so this call waits [`Self::synchronize_stream`]
    /// before returning. Capture cannot include a pageable copy. Pinned DMA
    /// ([`Self::memcpy_pinned_to_device`]) stays stream-ordered.
    pub fn memcpy(
        &mut self,
        device: DeviceId,
        op: MemcpyOp,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        let pageable = op.src.is_pageable() || op.dst.is_pageable();
        if pageable && self.in_capture(device, stream) {
            return Err(SimError::Invalid {
                why: "cannot capture pageable memcpy",
            });
        }
        let id = self.submit(device, stream, Kind::Memcpy(op))?;
        if pageable {
            self.synchronize_stream(device, stream)?;
        }
        Ok(id)
    }

    /// `cudaMemcpy`: enqueue then wait for that stream (host-synchronous).
    ///
    /// Capture cannot include it. [`Self::memcpy`] is `cudaMemcpyAsync`.
    pub fn memcpy_sync(
        &mut self,
        device: DeviceId,
        op: MemcpyOp,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        if self.capturing.is_some() {
            return Err(SimError::Invalid {
                why: "cannot capture host-sync memcpy",
            });
        }
        let id = self.memcpy(device, op, stream)?;
        self.synchronize_stream(device, stream)?;
        Ok(id)
    }

    /// Pageable host → `device`. Host-synchronous and slower than pinned DMA.
    ///
    /// Real `cudaMemcpyAsync` of pageable memory bounces through a driver
    /// staging buffer; the host does not return until this stream has finished
    /// the copy. [`Self::memcpy_pinned_to_device`] is the overlapping path.
    pub fn memcpy_host_to_device(
        &mut self,
        device: DeviceId,
        alloc: AllocId,
        bytes: u64,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        self.memcpy(
            device,
            MemcpyOp {
                src: Place::Host,
                dst: Place::Device(device),
                alloc,
                bytes,
                offset: 0,
            },
            stream,
        )
    }

    /// Page-locked host → `device` (CUDA DMA / `cudaMallocHost` source).
    pub fn memcpy_pinned_to_device(
        &mut self,
        device: DeviceId,
        alloc: AllocId,
        bytes: u64,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        self.memcpy(
            device,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(device),
                alloc,
                bytes,
                offset: 0,
            },
            stream,
        )
    }

    /// `device` → pageable host. Host-synchronous; source HBM residency is kept.
    pub fn memcpy_device_to_host(
        &mut self,
        device: DeviceId,
        alloc: AllocId,
        bytes: u64,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        self.memcpy(
            device,
            MemcpyOp {
                src: Place::Device(device),
                dst: Place::Host,
                alloc,
                bytes,
                offset: 0,
            },
            stream,
        )
    }

    /// `device` → page-locked host. Source HBM residency is kept (copy).
    pub fn memcpy_device_to_pinned(
        &mut self,
        device: DeviceId,
        alloc: AllocId,
        bytes: u64,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        self.memcpy(
            device,
            MemcpyOp {
                src: Place::Device(device),
                dst: Place::HostPinned,
                alloc,
                bytes,
                offset: 0,
            },
            stream,
        )
    }

    /// Peer copy `src` → `dst` of an existing allocation (hot replica).
    ///
    /// Submitted on `src`'s copy engine so it is stream-ordered with the
    /// producing alloc/H2D. Completion adds `dst` to the object's residency
    /// set and charges `dst` HBM; it does not drop `src`.
    pub fn memcpy_device_to_device(
        &mut self,
        src: DeviceId,
        dst: DeviceId,
        alloc: AllocId,
        bytes: u64,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        self.memcpy(
            src,
            MemcpyOp {
                src: Place::Device(src),
                dst: Place::Device(dst),
                alloc,
                bytes,
                offset: 0,
            },
            stream,
        )
    }

    /// Enqueue a kernel on whole allocations. Reads/writes are leased until it completes.
    ///
    /// A VMM VA must be fully mapped ([`Self::is_resident`]) or peer-readable
    /// via [`Self::va_set_access`] (reads) / [`Self::va_set_access_write`]
    /// (read and write). A mempool alloc may be read **and written** on a
    /// peer after [`Self::pool_set_access`]. For a mapped page of a larger
    /// VA, use [`Self::kernel_bufs`].
    ///
    /// A managed allocation not yet on `device` is [`Self::prefetch`]'d when
    /// the kernel starts (page fault after stream deps). Capture does not
    /// record that migrate; graph replay fails [`SimError::NotResident`] if
    /// the graph omitted it. Prefetch before [`Self::begin_capture`], or
    /// record [`Self::prefetch`] in the graph. Host attach and Single attach
    /// on another stream fail [`SimError::Invalid`] (`not attached`) instead
    /// of paging.
    pub fn kernel(
        &mut self,
        device: DeviceId,
        kind: KernelKind,
        reads: &[AllocId],
        writes: &[AllocId],
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        let reads: Vec<KernelBuf> = reads.iter().copied().map(KernelBuf::whole).collect();
        let writes: Vec<KernelBuf> = writes.iter().copied().map(KernelBuf::whole).collect();
        self.kernel_bufs(device, kind, &reads, &writes, stream)
    }

    /// Enqueue a kernel on explicit buffer spans (vLLM paged-KV analog).
    ///
    /// Each [`KernelBuf`] must be mapped-host, device-resident, a VMM span
    /// covered by [`Self::va_map_range`], a VMM peer [`Self::va_set_access`]
    /// read, a VMM peer [`Self::va_set_access_write`] (read and write), or a
    /// mempool peer [`Self::pool_set_access`] (read and write).
    /// `bytes == 0` means from `offset` to
    /// the end of the allocation. A range past the reservation is `Invalid`.
    /// A live kernel (not a graph replay) page-faults managed memory when it
    /// *starts*, after stream deps, so a waited prefetch is visible.
    /// Host attach and Single attach on another stream fail `not attached`.
    pub fn kernel_bufs(
        &mut self,
        device: DeviceId,
        kind: KernelKind,
        reads: &[KernelBuf],
        writes: &[KernelBuf],
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        self.submit_kernel(device, kind, reads, writes, stream, false)
    }

    /// `cudaLaunchCooperativeKernel` on whole allocations.
    ///
    /// Same lease / residency rules as [`Self::kernel`]. The grid occupies
    /// every [`crate::GpuProfile::compute_slots`] so leftover kernels on other
    /// streams cannot Hyper-Q overlap it. Fails `cooperative launch not
    /// supported` unless [`crate::GpuProfile::cooperative_launch`]. Capture is
    /// allowed (CUDA 11+).
    pub fn cooperative_kernel(
        &mut self,
        device: DeviceId,
        kind: KernelKind,
        reads: &[AllocId],
        writes: &[AllocId],
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        let reads: Vec<KernelBuf> = reads.iter().copied().map(KernelBuf::whole).collect();
        let writes: Vec<KernelBuf> = writes.iter().copied().map(KernelBuf::whole).collect();
        self.cooperative_kernel_bufs(device, kind, &reads, &writes, stream)
    }

    /// `cudaLaunchCooperativeKernel` on explicit buffer spans.
    pub fn cooperative_kernel_bufs(
        &mut self,
        device: DeviceId,
        kind: KernelKind,
        reads: &[KernelBuf],
        writes: &[KernelBuf],
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        self.require_cooperative(device)?;
        self.submit_kernel(device, kind, reads, writes, stream, true)
    }

    fn submit_kernel(
        &mut self,
        device: DeviceId,
        kind: KernelKind,
        reads: &[KernelBuf],
        writes: &[KernelBuf],
        stream: StreamId,
        cooperative: bool,
    ) -> Result<OpId, SimError> {
        let reads = self.resolve_bufs(reads)?;
        let writes = self.resolve_bufs(writes)?;
        self.submit(
            device,
            stream,
            Kind::Kernel {
                kind,
                reads,
                writes,
                cooperative,
            },
        )
    }

    /// Device-side fill (`cudaMemsetAsync`) of `[0, bytes)`.
    ///
    /// A VMM destination must have that span mapped ([`Self::is_range_resident`]).
    /// [`Self::kernel`] still needs the whole VA. [`Self::memset_buf`] names an
    /// interior page. Capture is allowed. Host-sync `cudaMalloc` / VMM / mempool
    /// create still cannot be captured; [`Self::alloc`] may be a mem alloc node.
    pub fn memset(
        &mut self,
        device: DeviceId,
        alloc: AllocId,
        bytes: u64,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        if bytes == 0 {
            return Err(SimError::Invalid {
                why: "zero-byte memset",
            });
        }
        self.memset_buf(device, KernelBuf::span(alloc, 0, bytes), stream)
    }

    /// `cudaMemsetAsync` of a [`KernelBuf`] span (vLLM new-KV-block analog).
    ///
    /// `bytes == 0` means from `offset` to the end of the allocation. A range
    /// past the reservation is `Invalid`. Mapped host is not a memset dest.
    pub fn memset_buf(
        &mut self,
        device: DeviceId,
        buf: KernelBuf,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        let total = self.alloc_ref(buf.id)?.bytes;
        let (offset, bytes) = kernel_span(total, &buf)?;
        if bytes == 0 {
            return Err(SimError::Invalid {
                why: "zero-byte memset",
            });
        }
        self.submit(
            device,
            stream,
            Kind::Memset {
                id: buf.id,
                offset,
                bytes,
            },
        )
    }

    /// `cudaLaunchHostFunc`. Stream-ordered host work; does not occupy compute
    /// or copy engines. Other streams can run GPU kernels at the same virtual
    /// time. Capture records a graph host node.
    pub fn host_func(&mut self, device: DeviceId, stream: StreamId) -> Result<OpId, SimError> {
        self.submit(device, stream, Kind::HostFunc)
    }

    /// `cudaStreamAttachMemAsync`. Stream-ordered visibility change; later ops
    /// on `stream` see the new attach. Illegal under stream capture (CUDA
    /// `cudaErrorStreamCaptureUnsupported`). `MemAttach::Single` cannot use the
    /// NULL stream.
    pub fn stream_attach(
        &mut self,
        device: DeviceId,
        id: AllocId,
        stream: StreamId,
        flags: MemAttach,
    ) -> Result<OpId, SimError> {
        self.fail_if_capturing("cannot capture stream attach")?;
        let a = self.alloc_ref(id)?;
        if !a.live {
            return Err(SimError::Invalid { why: "freed" });
        }
        if !a.managed {
            return Err(SimError::Invalid { why: "not managed" });
        }
        if flags == MemAttach::Single && stream == StreamId::NULL {
            return Err(SimError::Invalid {
                why: "cannot attach single to null stream",
            });
        }
        self.submit(device, stream, Kind::Attach { id, flags })
    }

    /// Record `event` after prior ops on `stream` (`cudaEventRecord`).
    pub fn record_event(
        &mut self,
        device: DeviceId,
        event: EventId,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        self.record_event_flags(device, event, stream, false)
    }

    /// `cudaEventRecordWithFlags(..., cudaEventRecordExternal)`.
    ///
    /// During capture this is a record node that does **not** put `event` in the
    /// forked-capture join set. A later [`Self::wait_event`] on another stream
    /// stays live. Live (non-capturing) this matches [`Self::record_event`].
    pub fn record_event_external(
        &mut self,
        device: DeviceId,
        event: EventId,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        self.record_event_flags(device, event, stream, true)
    }

    fn record_event_flags(
        &mut self,
        device: DeviceId,
        event: EventId,
        stream: StreamId,
        external: bool,
    ) -> Result<OpId, SimError> {
        let _ev = self.events.entry(event).or_insert(Ev {
            recorded_by: None,
            timing: true,
        });
        self.submit(device, stream, Kind::EventRecord { event, external })
    }

    /// Make later ops on `stream` wait until `event` is recorded and complete.
    pub fn wait_event(
        &mut self,
        device: DeviceId,
        event: EventId,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        self.wait_event_flags(device, event, stream, false)
    }

    /// `cudaStreamWaitEvent(..., cudaEventWaitExternal)`.
    ///
    /// During capture this is a wait node that does **not** join the waiter into
    /// the graph. Graph replay waits for a live record of `event`, not a
    /// [`Self::record_event_external`] node in the same graph. Live this matches
    /// [`Self::wait_event`].
    pub fn wait_event_external(
        &mut self,
        device: DeviceId,
        event: EventId,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        self.wait_event_flags(device, event, stream, true)
    }

    fn wait_event_flags(
        &mut self,
        device: DeviceId,
        event: EventId,
        stream: StreamId,
        external: bool,
    ) -> Result<OpId, SimError> {
        if let Entry::Vacant(slot) = self.events.entry(event) {
            let _ev = slot.insert(Ev {
                recorded_by: None,
                timing: true,
            });
        }
        self.submit(device, stream, Kind::EventWait { event, external })
    }

    /// Run until every submitted op is complete (`cudaDeviceSynchronize` on every GPU).
    ///
    /// Capture cannot include it. [`Self::synchronize_device`] waits one GPU.
    /// [`Self::synchronize_stream`] can still wait a stream that is not in the
    /// capture.
    pub fn synchronize(&mut self) -> Result<(), SimError> {
        self.fail_if_capturing("cannot synchronize during capture")?;
        self.drive_until(|sim| sim.running.is_empty() && sim.ops.values().all(|o| o.done))?;
        if self.running.is_empty() && !self.ops.values().all(|o| o.done) {
            return Err(SimError::Invalid {
                why: "deadlock: waiting ops but nothing running",
            });
        }
        self.sync_outcome()
    }

    /// `cudaDeviceSynchronize`: wait until every stream on `device` is idle.
    ///
    /// Other GPUs keep running. Capture cannot include it.
    /// [`Self::synchronize`] waits the whole node; [`Self::synchronize_stream`]
    /// waits one stream.
    pub fn synchronize_device(&mut self, device: DeviceId) -> Result<(), SimError> {
        if self.capturing.is_some() {
            return Err(SimError::Invalid {
                why: "cannot synchronize device during capture",
            });
        }
        let _gpu = self.profile.gpu(device)?;
        self.drive_until(|sim| sim.device_idle(device))?;
        if !self.device_idle(device) {
            return Err(SimError::Invalid {
                why: "deadlock: device busy but nothing running",
            });
        }
        self.device_sync_outcome(device)
    }

    /// Drain in-flight work, then jump the virtual clock to `ns` if that is still in the future.
    ///
    /// Models a GPU sitting idle until the next request arrives. Jumping the clock
    /// while work is queued would skip in-flight ops, so this always
    /// [`synchronize`](Self::synchronize)s first. Returns how many nanoseconds the
    /// clock jumped (0 if `ns` is already behind).
    pub fn idle_until(&mut self, ns: u64) -> Result<u64, SimError> {
        self.synchronize()?;
        if ns > self.clock {
            let jumped = ns.saturating_sub(self.clock);
            self.clock = ns;
            return Ok(jumped);
        }
        Ok(0)
    }

    /// `cudaStreamSynchronize`: advance the virtual clock until `stream` is idle.
    ///
    /// Other streams keep running. An already-idle stream returns without
    /// starting leftover kernels on other streams. A stream in an active graph
    /// capture is [`SimError::Invalid`]. Cancelled ops on *this* stream fail;
    /// cancelled work on other streams is left for a later [`Self::synchronize`].
    pub fn synchronize_stream(
        &mut self,
        device: DeviceId,
        stream: StreamId,
    ) -> Result<(), SimError> {
        let _gpu = self.profile.gpu(device)?;
        if self.in_capture(device, stream) {
            return Err(SimError::Invalid {
                why: "cannot synchronize stream during capture",
            });
        }
        self.drive_until(|sim| sim.stream_idle(device, stream))?;
        if !self.stream_idle(device, stream) {
            return Err(SimError::Invalid {
                why: "deadlock: stream busy but nothing running",
            });
        }
        self.stream_sync_outcome(device, stream)
    }

    /// `cudaEventSynchronize`: wait until `event` is recorded and complete.
    ///
    /// Work after the record on the same stream, and work on other streams,
    /// keeps running. An event with no record and no running ops is a deadlock.
    pub fn synchronize_event(&mut self, event: EventId) -> Result<(), SimError> {
        if !self.events.contains_key(&event) {
            return Err(SimError::UnknownEvent { event: event.0 });
        }
        self.drive_until(|sim| sim.event_complete(event))?;
        if !self.event_complete(event) {
            return Err(SimError::Invalid {
                why: "deadlock: event not complete but nothing running",
            });
        }
        let rec = self.events.get(&event).and_then(|e| e.recorded_by);
        if let Some(id) = rec {
            if self.ops.get(&id).is_some_and(|o| o.cancelled) {
                let stream = self.ops.get(&id).map(|o| o.stream).unwrap_or(StreamId(0));
                return Err(SimError::Cancelled { stream, n: 1 });
            }
        }
        Ok(())
    }

    /// `cudaEventElapsedTime` in nanoseconds (this crate is ns, not milliseconds).
    ///
    /// Both events must be recorded, complete, and created with timing enabled.
    /// [`Self::create_event_disable_timing`] events are [`SimError::Invalid`].
    /// Returns `end.done_ns - start.done_ns`. An unknown event is
    /// [`SimError::UnknownEvent`]. If `end` finished first, [`SimError::Invalid`].
    pub fn event_elapsed_ns(&self, start: EventId, end: EventId) -> Result<u64, SimError> {
        self.require_event_timing(start)?;
        self.require_event_timing(end)?;
        let start_ns = self.event_done_ns(start)?;
        let end_ns = self.event_done_ns(end)?;
        end_ns.checked_sub(start_ns).ok_or(SimError::Invalid {
            why: "event elapsed: end before start",
        })
    }

    fn require_event_timing(&self, event: EventId) -> Result<(), SimError> {
        match self.events.get(&event) {
            None => Err(SimError::UnknownEvent { event: event.0 }),
            Some(ev) if ev.timing => Ok(()),
            Some(_) => Err(SimError::Invalid {
                why: "event elapsed: disable timing",
            }),
        }
    }

    fn event_done_ns(&self, event: EventId) -> Result<u64, SimError> {
        if !self.events.contains_key(&event) {
            return Err(SimError::UnknownEvent { event: event.0 });
        }
        if !self.event_complete(event) {
            return Err(SimError::Invalid {
                why: "event elapsed: not complete",
            });
        }
        let rec = self.events.get(&event).and_then(|e| e.recorded_by);
        let Some(id) = rec else {
            return Err(SimError::Invalid {
                why: "event elapsed: not recorded",
            });
        };
        let op = self.ops.get(&id).ok_or(SimError::Invalid {
            why: "event elapsed: missing record op",
        })?;
        op.done_ns.ok_or(SimError::Invalid {
            why: "event elapsed: record has no done_ns",
        })
    }

    fn drive_until(&mut self, idle: impl Fn(&Self) -> bool) -> Result<(), SimError> {
        let mut steps = 0u32;
        loop {
            // An already-idle waited stream must not start leftover work on
            // other streams (`cudaStreamSynchronize` returns immediately).
            if idle(self) {
                return Ok(());
            }
            self.schedule()?;
            if idle(self) {
                return Ok(());
            }
            if self.running.is_empty() {
                return Ok(());
            }
            self.advance_to_next_completion()?;
            steps = steps.saturating_add(1);
            if steps > 10_000_000 {
                return Err(SimError::Invalid {
                    why: "simulator step limit",
                });
            }
        }
    }

    fn sync_outcome(&self) -> Result<(), SimError> {
        let mut n = 0u32;
        let mut stream = StreamId(0);
        for o in self.ops.values() {
            if o.cancelled {
                n = n.saturating_add(1);
                stream = o.stream;
            }
        }
        if n > 0 {
            return Err(SimError::Cancelled { stream, n });
        }
        Ok(())
    }

    fn stream_sync_outcome(&self, device: DeviceId, stream: StreamId) -> Result<(), SimError> {
        let mut n = 0u32;
        for o in self.ops.values() {
            if o.cancelled && o.device == device && o.stream == stream {
                n = n.saturating_add(1);
            }
        }
        if n > 0 {
            return Err(SimError::Cancelled { stream, n });
        }
        Ok(())
    }

    fn submit(&mut self, device: DeviceId, stream: StreamId, kind: Kind) -> Result<OpId, SimError> {
        self.submit_launch(device, stream, kind, LaunchCost::Kernel)
    }

    fn submit_launch(
        &mut self,
        device: DeviceId,
        stream: StreamId,
        kind: Kind,
        launch: LaunchCost,
    ) -> Result<OpId, SimError> {
        if self.unavailable.contains(&device) {
            return Err(SimError::Unavailable { device });
        }
        if self.capturing.is_some() {
            return self.submit_captured(device, stream, kind);
        }
        self.submit_live(device, stream, kind, launch)
    }

    fn in_capture(&self, device: DeviceId, stream: StreamId) -> bool {
        self.capturing
            .as_ref()
            .is_some_and(|c| c.streams.contains(&(device, stream)))
    }

    fn submit_captured(
        &mut self,
        device: DeviceId,
        stream: StreamId,
        kind: Kind,
    ) -> Result<OpId, SimError> {
        if let Kind::EventWait { event, external } = &kind {
            let join = !*external
                && self
                    .capturing
                    .as_ref()
                    .is_some_and(|c| c.events.contains(event));
            if join && !self.in_capture(device, stream) {
                let same = self
                    .capturing
                    .as_ref()
                    .is_some_and(|c| c.origin.0 == device);
                if !same {
                    return Err(SimError::Invalid {
                        why: "capture fork requires same device",
                    });
                }
                if !self.stream_idle(device, stream) {
                    return Err(SimError::Invalid {
                        why: "capture fork requires idle stream",
                    });
                }
                if let Some(cap) = self.capturing.as_mut() {
                    let _ins = cap.streams.insert((device, stream));
                }
            }
        }
        if !self.in_capture(device, stream) {
            return self.submit_live(device, stream, kind, LaunchCost::Kernel);
        }
        if let Kind::EventRecord { event, external } = &kind {
            if !*external {
                if let Some(cap) = self.capturing.as_mut() {
                    let _ins = cap.events.insert(*event);
                }
            }
        }
        let deps = capture_step_deps(&self.capture_buf, device, stream, &kind);
        self.capture_buf.push(GraphStep {
            device,
            stream,
            kind,
            deps,
        });
        let id = OpId(self.next_op);
        self.next_op = self.next_op.saturating_add(1);
        Ok(id)
    }

    fn submit_live(
        &mut self,
        device: DeviceId,
        stream: StreamId,
        kind: Kind,
        launch: LaunchCost,
    ) -> Result<OpId, SimError> {
        let _gpu = self.profile.gpu(device)?;
        let id = OpId(self.next_op);
        self.next_op = self.next_op.saturating_add(1);
        let mut deps = self.stream_order_deps(device, stream);
        if let Kind::EventWait { event, .. } = &kind {
            if let Some(ev) = self.events.get(event) {
                if let Some(rec) = ev.recorded_by {
                    deps.push(rec);
                }
            }
        }
        let _prev_op = self.ops.insert(
            id,
            Op {
                device,
                stream,
                kind,
                deps,
                done: false,
                cancelled: false,
                launch,
                submit_ns: self.clock,
                start_ns: None,
                done_ns: None,
            },
        );
        let _prev_tail = self.tail.insert((device, stream), id);
        Ok(id)
    }

    fn stream_order_deps(&self, device: DeviceId, stream: StreamId) -> Vec<OpId> {
        let mut deps = Vec::new();
        if let Some(prev) = self.tail.get(&(device, stream)) {
            deps.push(*prev);
        }
        if let Some(joins) = self.graph_joins.get(&(device, stream)) {
            for id in joins {
                if !deps.contains(id) {
                    deps.push(*id);
                }
            }
        }
        self.add_null_stream_deps(device, stream, &mut deps);
        deps
    }

    fn add_null_stream_deps(&self, device: DeviceId, stream: StreamId, deps: &mut Vec<OpId>) {
        if self.legacy_null_stream {
            if stream == StreamId::NULL {
                for ((d, s), tail) in &self.tail {
                    if *d == device && *s != stream {
                        deps.push(*tail);
                    }
                }
            } else if let Some(tail) = self.tail.get(&(device, StreamId::NULL)) {
                deps.push(*tail);
            }
            return;
        }
        if stream == StreamId::NULL {
            for ((d, s), tail) in &self.tail {
                if *d == device && self.blocking.contains(&(*d, *s)) {
                    deps.push(*tail);
                }
            }
        } else if self.blocking.contains(&(device, stream)) {
            if let Some(tail) = self.tail.get(&(device, StreamId::NULL)) {
                deps.push(*tail);
            }
        }
    }

    fn device_idle(&self, device: DeviceId) -> bool {
        !self.ops.values().any(|o| o.device == device && !o.done)
            && !self
                .running
                .iter()
                .any(|r| self.ops.get(&r.op).is_some_and(|o| o.device == device))
    }

    fn device_sync_outcome(&self, device: DeviceId) -> Result<(), SimError> {
        let mut n = 0u32;
        let mut stream = StreamId(0);
        for o in self.ops.values() {
            if o.cancelled && o.device == device {
                n = n.saturating_add(1);
                stream = o.stream;
            }
        }
        if n > 0 {
            return Err(SimError::Cancelled { stream, n });
        }
        Ok(())
    }

    fn stream_idle(&self, device: DeviceId, stream: StreamId) -> bool {
        let local = !self
            .ops
            .values()
            .any(|o| o.device == device && o.stream == stream && !o.done)
            && !self.running.iter().any(|r| {
                self.ops
                    .get(&r.op)
                    .is_some_and(|o| o.device == device && o.stream == stream)
            });
        if !local {
            return false;
        }
        self.graph_joins
            .get(&(device, stream))
            .is_none_or(|ids| ids.iter().all(|id| self.op_done(*id)))
    }

    fn gpu_rt(&self, device: DeviceId) -> Result<&GpuRt, SimError> {
        self.gpus.get(&device).ok_or(SimError::Invalid {
            why: "device runtime missing",
        })
    }

    fn gpu_rt_mut(&mut self, device: DeviceId) -> Result<&mut GpuRt, SimError> {
        self.gpus.get_mut(&device).ok_or(SimError::Invalid {
            why: "device runtime missing",
        })
    }

    fn alloc_ref(&self, id: AllocId) -> Result<&Alloc, SimError> {
        self.allocs
            .get(&id)
            .ok_or(SimError::UnknownAlloc { alloc: id })
    }

    fn alloc_mut(&mut self, id: AllocId) -> Result<&mut Alloc, SimError> {
        self.allocs
            .get_mut(&id)
            .ok_or(SimError::UnknownAlloc { alloc: id })
    }

    fn fail_if_capturing(&self, why: &'static str) -> Result<(), SimError> {
        if self.capturing.is_some() {
            Err(SimError::Invalid { why })
        } else {
            Ok(())
        }
    }

    fn insert_host(
        &mut self,
        bytes: u64,
        pageable: bool,
        pinned: bool,
        mapped: bool,
        registered: bool,
    ) -> Result<AllocId, SimError> {
        if bytes == 0 {
            return Err(SimError::Invalid {
                why: "zero-byte alloc",
            });
        }
        self.fail_if_capturing("cannot capture alloc/free")?;
        if pinned {
            self.charge_pin(bytes)?;
        }
        let id = AllocId(self.next_alloc);
        self.next_alloc = self.next_alloc.saturating_add(1);
        let _prev = self.allocs.insert(
            id,
            Alloc {
                bytes,
                devices: Vec::new(),
                leases: 0,
                live: true,
                host_pinned: pinned,
                host_pageable: pageable,
                host_mapped: mapped,
                host_registered: registered,
                managed: false,
                attach: Attach::Global,
                read_mostly: false,
                accessed_by: BTreeSet::new(),
                vmm_write_by: BTreeSet::new(),
                preferred: Preferred::None,
                vmm: false,
                vmm_maps: Vec::new(),
                pool: None,
                ipc_src: None,
                ipc_opens: 0,
                share_src: None,
                share_opens: 0,
            },
        );
        Ok(id)
    }

    fn charge_pin(&mut self, bytes: u64) -> Result<(), SimError> {
        let cap = self.profile.host_pin_bytes;
        let used = self.pinned_used;
        let free = cap.saturating_sub(used);
        if bytes > free {
            return Err(SimError::PinOom { need: bytes, free });
        }
        self.pinned_used = used.saturating_add(bytes);
        Ok(())
    }

    fn refund_pin(&mut self, bytes: u64) {
        self.pinned_used = self.pinned_used.saturating_sub(bytes);
    }

    fn host_register_flags(&mut self, id: AllocId, mapped: bool) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture host register")?;
        let a = self.alloc_ref(id)?;
        if a.leases > 0 {
            return Err(SimError::Leased { alloc: id });
        }
        if !a.live || !a.host_pageable || a.host_pinned {
            return Err(SimError::Invalid {
                why: "not pageable host",
            });
        }
        let bytes = a.bytes;
        self.charge_pin(bytes)?;
        self.clock = self.clock.saturating_add(self.first_alloc_ns());
        let a = self.alloc_mut(id)?;
        a.host_pageable = false;
        a.host_pinned = true;
        a.host_mapped = mapped;
        a.host_registered = true;
        Ok(())
    }

    fn managed_move_src(
        &self,
        alloc: AllocId,
        dest: Option<DeviceId>,
    ) -> Result<(u64, Place), SimError> {
        let a = self.alloc_ref(alloc)?;
        if !a.live || !a.managed {
            return Err(SimError::Invalid { why: "not managed" });
        }
        let src = match dest {
            Some(d) if a.devices.contains(&d) => Place::HostPinned,
            Some(_) => a
                .devices
                .first()
                .copied()
                .map(Place::Device)
                .unwrap_or(Place::HostPinned),
            None if a.devices.is_empty() => Place::HostPinned,
            None => a
                .devices
                .first()
                .copied()
                .map(Place::Device)
                .unwrap_or(Place::HostPinned),
        };
        Ok((a.bytes, src))
    }

    fn require_device_attach(&self, id: AllocId, stream: StreamId) -> Result<(), SimError> {
        if self.alloc_ref(id)?.device_attach_ok(stream) {
            Ok(())
        } else {
            Err(SimError::Invalid {
                why: "not attached",
            })
        }
    }

    fn require_bufs_attached(
        &self,
        stream: StreamId,
        reads: &[KernelBuf],
        writes: &[KernelBuf],
    ) -> Result<(), SimError> {
        for b in reads.iter().chain(writes.iter()) {
            self.require_device_attach(b.id, stream)?;
        }
        Ok(())
    }

    fn require_memcpy_attach(&self, stream: StreamId, m: &MemcpyOp) -> Result<(), SimError> {
        match m.dst {
            Place::Device(_) => self.require_device_attach(m.alloc, stream),
            Place::Host | Place::HostPinned => Ok(()),
        }
    }

    /// Live kernels page-fault at start. Returns true if a prefetch was
    /// inserted ahead of `kernel` (caller must reschedule; the kernel is not
    /// running yet).
    fn inject_managed_faults(
        &mut self,
        kernel: OpId,
        device: DeviceId,
        reads: &[KernelBuf],
        writes: &[KernelBuf],
    ) -> Result<bool, SimError> {
        let wait = self.managed_fault_ids(device, reads, writes)?;
        if wait.is_empty() {
            return Ok(false);
        }
        let kdeps = self
            .ops
            .get(&kernel)
            .ok_or(SimError::Invalid { why: "unknown op" })?
            .deps
            .clone();
        for alloc in wait {
            self.inject_fault_memcpy(kernel, device, alloc, &kdeps)?;
        }
        Ok(true)
    }

    fn managed_fault_ids(
        &self,
        device: DeviceId,
        reads: &[KernelBuf],
        writes: &[KernelBuf],
    ) -> Result<BTreeSet<AllocId>, SimError> {
        let mut wait = BTreeSet::new();
        for b in reads {
            let a = self.alloc_ref(b.id)?;
            if a.live && a.managed && !a.devices.contains(&device) && !a.remote_read_ok(device) {
                let _ins = wait.insert(b.id);
            }
        }
        for b in writes {
            let a = self.alloc_ref(b.id)?;
            if a.live && a.managed && !a.devices.contains(&device) {
                let _ins = wait.insert(b.id);
            }
        }
        Ok(wait)
    }

    fn inject_fault_memcpy(
        &mut self,
        kernel: OpId,
        device: DeviceId,
        alloc: AllocId,
        deps: &[OpId],
    ) -> Result<(), SimError> {
        let stream = self
            .ops
            .get(&kernel)
            .ok_or(SimError::Invalid { why: "unknown op" })?
            .stream;
        let (bytes, src) = self.managed_move_src(alloc, Some(device))?;
        let id = OpId(self.next_op);
        self.next_op = self.next_op.saturating_add(1);
        let _prev = self.ops.insert(
            id,
            Op {
                device,
                stream,
                kind: Kind::Memcpy(MemcpyOp {
                    src,
                    dst: Place::Device(device),
                    alloc,
                    bytes,
                    offset: 0,
                }),
                deps: deps.to_vec(),
                done: false,
                cancelled: false,
                launch: LaunchCost::Kernel,
                submit_ns: self.clock,
                start_ns: None,
                done_ns: None,
            },
        );
        self.add_op_dep(kernel, id);
        Ok(())
    }

    fn start_kernel(&mut self, id: OpId) -> Result<bool, SimError> {
        let (device, stream, launch, reads, writes, kind, cooperative) = {
            let op = self
                .ops
                .get(&id)
                .ok_or(SimError::Invalid { why: "unknown op" })?;
            match &op.kind {
                Kind::Kernel {
                    reads,
                    writes,
                    kind,
                    cooperative,
                } => (
                    op.device,
                    op.stream,
                    op.launch,
                    reads.clone(),
                    writes.clone(),
                    kind.clone(),
                    *cooperative,
                ),
                _ => {
                    return Err(SimError::Invalid {
                        why: "not a kernel",
                    });
                }
            }
        };
        self.require_bufs_attached(stream, &reads, &writes)?;
        if matches!(launch, LaunchCost::Kernel)
            && self.inject_managed_faults(id, device, &reads, &writes)?
        {
            return Ok(true);
        }
        let slots = self.kernel_slots(device, cooperative)?;
        if !self.take_compute_n(device, slots)? {
            return Ok(false);
        }
        let mem_bps = self.kernel_mem_bps(device, &reads, &writes)?;
        if let Err(e) = self.lease_kernel(device, &reads, &writes, true) {
            self.drop_compute_n(device, slots)?;
            return Err(e);
        }
        if let Err(e) = self.invalidate_read_mostly_writes(device, &writes) {
            self.drop_compute_n(device, slots)?;
            return Err(e);
        }
        let ns = match self.kernel_ns(device, stream, &kind, launch, mem_bps) {
            Ok(n) => n,
            Err(e) => {
                self.drop_compute_n(device, slots)?;
                return Err(e);
            }
        };
        self.running.push(Running {
            op: id,
            remaining_ns: ns.max(1),
            share: Share::Solo,
        });
        Ok(true)
    }

    fn start_memset(&mut self, id: OpId) -> Result<bool, SimError> {
        let (device, stream, launch, alloc, offset, bytes) = {
            let op = self
                .ops
                .get(&id)
                .ok_or(SimError::Invalid { why: "unknown op" })?;
            match &op.kind {
                Kind::Memset {
                    id: alloc,
                    offset,
                    bytes,
                } => (op.device, op.stream, op.launch, *alloc, *offset, *bytes),
                _ => {
                    return Err(SimError::Invalid {
                        why: "not a memset",
                    });
                }
            }
        };
        self.require_device_attach(alloc, stream)?;
        if !self.take_compute(device)? {
            return Ok(false);
        }
        let writes = [KernelBuf::span(alloc, offset, bytes)];
        if let Err(e) = self.lease_kernel(device, &[], &writes, false) {
            self.drop_compute(device)?;
            return Err(e);
        }
        if let Err(e) = self.invalidate_read_mostly_writes(device, &writes) {
            self.drop_compute(device)?;
            return Err(e);
        }
        let mem_bps = match self.kernel_mem_bps(device, &[], &writes) {
            Ok(b) => b,
            Err(e) => {
                self.drop_compute(device)?;
                return Err(e);
            }
        };
        let ns = match self.memset_ns(device, bytes, launch, mem_bps) {
            Ok(n) => n,
            Err(e) => {
                self.drop_compute(device)?;
                return Err(e);
            }
        };
        self.running.push(Running {
            op: id,
            remaining_ns: ns.max(1),
            share: Share::Solo,
        });
        Ok(true)
    }

    fn resolve_bufs(&self, bufs: &[KernelBuf]) -> Result<Vec<KernelBuf>, SimError> {
        let mut out = Vec::new();
        for b in bufs {
            let total = self.alloc_ref(b.id)?.bytes;
            let (offset, bytes) = kernel_span(total, b)?;
            out.push(KernelBuf {
                id: b.id,
                offset,
                bytes,
            });
        }
        Ok(out)
    }

    fn managed_local_copy(&self, m: &MemcpyOp) -> Result<bool, SimError> {
        let a = self.alloc_ref(m.alloc)?;
        if !a.managed {
            return Ok(false);
        }
        match m.dst {
            Place::Device(d) => Ok(a.devices.contains(&d)),
            Place::Host | Place::HostPinned => Ok(a.devices.is_empty()),
        }
    }

    fn migrate_off_except(&mut self, alloc: AllocId, keep: DeviceId) -> Result<(), SimError> {
        let a = self.alloc_ref(alloc)?;
        let bytes = a.bytes;
        let others: Vec<DeviceId> = a.devices.iter().copied().filter(|d| *d != keep).collect();
        for d in others {
            self.refund_device(d, alloc, bytes)?;
            self.alloc_mut(alloc)?.devices.retain(|x| *x != d);
        }
        Ok(())
    }

    fn migrate_off_all(&mut self, alloc: AllocId) -> Result<(), SimError> {
        let a = self.alloc_ref(alloc)?;
        let bytes = a.bytes;
        let holders = a.devices.clone();
        for d in holders {
            self.refund_device(d, alloc, bytes)?;
        }
        self.alloc_mut(alloc)?.devices.clear();
        Ok(())
    }

    fn finish_memcpy(&mut self, device: DeviceId, m: MemcpyOp, dma: bool) -> Result<(), SimError> {
        if dma {
            self.gpu_rt_mut(device)?.copies = self.gpu_rt(device)?.copies.saturating_sub(1);
            self.bytes_moved = self.bytes_moved.saturating_add(m.bytes);
        }
        let managed = self.alloc_ref(m.alloc)?.managed;
        let read_mostly = self.alloc_ref(m.alloc)?.read_mostly;
        if let Place::Device(dst) = m.dst {
            let a = self.alloc_mut(m.alloc)?;
            if !a.devices.contains(&dst) {
                a.devices.push(dst);
            }
            if managed && !read_mostly {
                self.migrate_off_except(m.alloc, dst)?;
            }
        } else if managed {
            self.migrate_off_all(m.alloc)?;
        } else if matches!(m.dst, Place::HostPinned) {
            self.alloc_mut(m.alloc)?.host_pinned = true;
        }
        Ok(())
    }

    fn pool_ref(&self, id: PoolId) -> Result<&Pool, SimError> {
        self.pools.get(&id).ok_or(SimError::Invalid {
            why: "unknown pool",
        })
    }

    fn pool_mut(&mut self, id: PoolId) -> Result<&mut Pool, SimError> {
        self.pools.get_mut(&id).ok_or(SimError::Invalid {
            why: "unknown pool",
        })
    }

    fn pool_root(&self, pool: PoolId) -> Result<PoolId, SimError> {
        Ok(self.pool_ref(pool)?.share_root.unwrap_or(pool))
    }

    fn handle_ref(&self, id: MemHandleId) -> Result<&MemHandle, SimError> {
        self.mem_handles.get(&id).ok_or(SimError::Invalid {
            why: "unknown handle",
        })
    }

    fn handle_mut(&mut self, id: MemHandleId) -> Result<&mut MemHandle, SimError> {
        self.mem_handles.get_mut(&id).ok_or(SimError::Invalid {
            why: "unknown handle",
        })
    }

    fn mc_ref(&self, id: MulticastId) -> Result<&Multicast, SimError> {
        self.multicasts.get(&id).ok_or(SimError::Invalid {
            why: "unknown multicast",
        })
    }

    fn mc_mut(&mut self, id: MulticastId) -> Result<&mut Multicast, SimError> {
        self.multicasts.get_mut(&id).ok_or(SimError::Invalid {
            why: "unknown multicast",
        })
    }

    /// Unmap one VMM span: refund HBM unless it is an explicit [`Self::va_map_handle`].
    fn drop_vmm_physical(
        &mut self,
        id: AllocId,
        device: DeviceId,
        offset: u64,
        bytes: u64,
    ) -> Result<(), SimError> {
        if let Some(h) = self.vmm_handle_at.remove(&(id, device, offset, bytes)) {
            let maps = self.handle_ref(h)?.maps;
            self.handle_mut(h)?.maps = maps.saturating_sub(1);
            return self.maybe_refund_handle(h);
        }
        self.refund_device(device, id, bytes)
    }

    /// Free HBM when a handle has no refs and no maps.
    fn maybe_refund_handle(&mut self, handle: MemHandleId) -> Result<(), SimError> {
        let (device, bytes, refund) = {
            let h = self.handle_ref(handle)?;
            (h.device, h.bytes, h.refs == 0 && h.maps == 0 && h.charged)
        };
        if refund {
            let used = self.gpu_rt(device)?.used;
            self.gpu_rt_mut(device)?.used = used.saturating_sub(bytes);
            self.handle_mut(handle)?.charged = false;
        }
        Ok(())
    }

    fn bump_hbm_peak(&mut self, device: DeviceId) -> Result<(), SimError> {
        let peak = self.gpu_rt(device)?.used;
        if peak > self.hbm_peak {
            self.hbm_peak = peak;
        }
        Ok(())
    }

    fn reserve_hbm(&mut self, device: DeviceId, bytes: u64) -> Result<(), SimError> {
        let cap = self.profile.gpu(device)?.hbm_bytes;
        let used = self.gpu_rt(device)?.used;
        let free = cap.saturating_sub(used);
        if bytes > free {
            return Err(SimError::Oom {
                device,
                need: bytes,
                free,
            });
        }
        self.gpu_rt_mut(device)?.used = used.saturating_add(bytes);
        self.bump_hbm_peak(device)
    }

    fn pool_acquire(&mut self, pool: PoolId, bytes: u64) -> Result<u64, SimError> {
        let pool = self.pool_root(pool)?;
        let (device, cached) = {
            let p = self.pool_ref(pool)?;
            (p.device, p.cached)
        };
        let first = self.profile.gpu(device)?.alloc_overhead_ns;
        let reuse = self.profile.gpu(device)?.pool_reuse_ns;
        if cached >= bytes {
            let p = self.pool_mut(pool)?;
            p.cached = cached.saturating_sub(bytes);
            p.live = p.live.saturating_add(bytes);
            return Ok(reuse.max(1));
        }
        let extra = bytes.saturating_sub(cached);
        self.reserve_hbm(device, extra)?;
        let p = self.pool_mut(pool)?;
        p.cached = 0;
        p.live = p.live.saturating_add(bytes);
        Ok(first.max(1))
    }

    fn pool_release(&mut self, pool: PoolId, bytes: u64) -> Result<(), SimError> {
        let pool = self.pool_root(pool)?;
        let (device, threshold, cached, live) = {
            let p = self.pool_ref(pool)?;
            (p.device, p.release_threshold, p.cached, p.live)
        };
        let live = live.saturating_sub(bytes);
        let cached = cached.saturating_add(bytes);
        let (cached, drop) = if cached > threshold {
            (threshold, cached.saturating_sub(threshold))
        } else {
            (cached, 0)
        };
        {
            let p = self.pool_mut(pool)?;
            p.live = live;
            p.cached = cached;
        }
        if drop > 0 {
            let used = self.gpu_rt(device)?.used;
            self.gpu_rt_mut(device)?.used = used.saturating_sub(drop);
        }
        Ok(())
    }

    fn refund_device(
        &mut self,
        device: DeviceId,
        alloc: AllocId,
        bytes: u64,
    ) -> Result<(), SimError> {
        let pool = self.alloc_ref(alloc)?.pool;
        if let Some(p) = pool {
            if self.pool_ref(p)?.device == device {
                return self.pool_release(p, bytes);
            }
        }
        let used = self.gpu_rt(device)?.used;
        self.gpu_rt_mut(device)?.used = used.saturating_sub(bytes);
        Ok(())
    }

    fn start_alloc(
        &mut self,
        op: OpId,
        device: DeviceId,
        alloc: AllocId,
        bytes: u64,
    ) -> Result<bool, SimError> {
        let pool = self.alloc_ref(alloc)?.pool;
        let already = {
            let a = self.alloc_ref(alloc)?;
            a.live && a.devices.contains(&device)
        };
        let ns = if already {
            match pool {
                Some(_) => self.profile.gpu(device)?.pool_reuse_ns.max(1),
                None => 1,
            }
        } else if let Some(p) = pool {
            if self.pool_ref(p)?.device != device {
                return Err(SimError::Invalid {
                    why: "pool device mismatch",
                });
            }
            self.pool_acquire(p, bytes)?
        } else {
            self.reserve_hbm(device, bytes)?;
            self.profile.gpu(device)?.alloc_overhead_ns.max(1)
        };
        self.running.push(Running {
            op,
            remaining_ns: ns.max(1),
            share: Share::Solo,
        });
        Ok(true)
    }

    fn start_free(&mut self, op: OpId, device: DeviceId, alloc: AllocId) -> Result<bool, SimError> {
        let a = self.alloc_ref(alloc)?;
        if a.leases > 0 {
            return Err(SimError::Leased { alloc });
        }
        if a.ipc_src.is_some() {
            self.drop_ipc_import(alloc)?;
            self.running.push(Running {
                op,
                remaining_ns: 1,
                share: Share::Solo,
            });
            return Ok(true);
        }
        if a.share_src.is_some() {
            self.drop_share_import(alloc)?;
            self.running.push(Running {
                op,
                remaining_ns: 1,
                share: Share::Solo,
            });
            return Ok(true);
        }
        if a.ipc_opens > 0 {
            return Err(SimError::Invalid { why: "ipc mapped" });
        }
        if a.share_opens > 0 {
            return Err(SimError::Invalid {
                why: "share mapped",
            });
        }
        if !a.live || !a.devices.contains(&device) {
            return Err(SimError::UnknownAlloc { alloc });
        }
        let bytes = a.bytes;
        self.refund_device(device, alloc, bytes)?;
        let a = self.alloc_mut(alloc)?;
        a.devices.retain(|d| *d != device);
        if a.devices.is_empty() && !a.host_pinned {
            a.live = false;
        }
        self.running.push(Running {
            op,
            remaining_ns: 1,
            share: Share::Solo,
        });
        Ok(true)
    }

    fn drop_ipc_import(&mut self, id: AllocId) -> Result<(), SimError> {
        let (src, leases) = {
            let a = self.alloc_ref(id)?;
            (
                a.ipc_src.ok_or(SimError::Invalid {
                    why: "not ipc import",
                })?,
                a.leases,
            )
        };
        if leases > 0 {
            return Err(SimError::Leased { alloc: id });
        }
        let opens = self.alloc_ref(src)?.ipc_opens;
        self.alloc_mut(src)?.ipc_opens = opens.saturating_sub(1);
        let a = self.alloc_mut(id)?;
        a.live = false;
        a.devices.clear();
        Ok(())
    }

    fn drop_share_import(&mut self, id: AllocId) -> Result<(), SimError> {
        let (src, leases) = {
            let a = self.alloc_ref(id)?;
            (
                a.share_src.ok_or(SimError::Invalid {
                    why: "not share import",
                })?,
                a.leases,
            )
        };
        if leases > 0 {
            return Err(SimError::Leased { alloc: id });
        }
        let opens = self.alloc_ref(src)?.share_opens;
        self.alloc_mut(src)?.share_opens = opens.saturating_sub(1);
        let a = self.alloc_mut(id)?;
        a.live = false;
        a.devices.clear();
        Ok(())
    }

    fn reserve_now(&mut self, device: DeviceId, bytes: u64) -> Result<AllocId, SimError> {
        self.reserve_hbm(device, bytes)?;
        let overhead = self.profile.gpu(device)?.alloc_overhead_ns.max(1);
        self.clock = self.clock.saturating_add(overhead);
        let id = AllocId(self.next_alloc);
        self.next_alloc = self.next_alloc.saturating_add(1);
        let _prev = self.allocs.insert(
            id,
            Alloc {
                bytes,
                devices: vec![device],
                leases: 0,
                live: true,
                host_pinned: false,
                host_pageable: false,
                host_mapped: false,
                host_registered: false,
                managed: false,
                attach: Attach::Global,
                read_mostly: false,
                accessed_by: BTreeSet::new(),
                vmm_write_by: BTreeSet::new(),
                preferred: Preferred::None,
                vmm: false,
                vmm_maps: Vec::new(),
                pool: None,
                ipc_src: None,
                ipc_opens: 0,
                share_src: None,
                share_opens: 0,
            },
        );
        Ok(id)
    }

    fn op_done(&self, id: OpId) -> bool {
        self.ops.get(&id).is_some_and(|o| o.done)
    }

    fn deps_ready(&self, op: &Op) -> bool {
        op.deps.iter().all(|d| self.op_done(*d))
    }

    fn is_running(&self, id: OpId) -> bool {
        self.running.iter().any(|r| r.op == id)
    }

    fn schedule(&mut self) -> Result<(), SimError> {
        loop {
            let mut started = false;
            let mut candidates: Vec<(i32, u64, OpId)> = Vec::new();
            for (id, o) in &self.ops {
                if o.done || self.is_running(*id) || !self.deps_ready(o) {
                    continue;
                }
                let pri = self.stream_priority(o.device, o.stream);
                candidates.push((pri, id.0, *id));
            }
            candidates.sort_by_key(|&(pri, oid, _)| (Reverse(pri), oid));
            for (_, _, id) in candidates {
                if self.try_start(id)? {
                    if self.is_running(id) {
                        if let Some(op) = self.ops.get_mut(&id) {
                            if op.start_ns.is_none() && !op.cancelled {
                                op.start_ns = Some(self.clock);
                            }
                        }
                    }
                    started = true;
                }
            }
            if !started {
                return Ok(());
            }
        }
    }

    fn try_start(&mut self, id: OpId) -> Result<bool, SimError> {
        let Some(op) = self.ops.get(&id) else {
            return Err(SimError::Invalid { why: "unknown op" });
        };
        let device = op.device;
        let launch = op.launch;
        let stream = op.stream;
        if self.unavailable.contains(&device) {
            return Err(SimError::Unavailable { device });
        }
        match &op.kind {
            Kind::Alloc { id: alloc, bytes } => self.start_alloc(id, device, *alloc, *bytes),
            Kind::Free { id: alloc } => self.start_free(id, device, *alloc),
            Kind::Kernel { .. } => self.start_kernel(id),
            Kind::Memset { .. } => self.start_memset(id),
            Kind::HostFunc => {
                let ns = self.host_func_ns(device, launch)?;
                self.running.push(Running {
                    op: id,
                    remaining_ns: ns.max(1),
                    share: Share::Solo,
                });
                Ok(true)
            }
            Kind::Attach { .. } => {
                self.running.push(Running {
                    op: id,
                    remaining_ns: 1,
                    share: Share::Solo,
                });
                Ok(true)
            }
            Kind::Memcpy(m) => {
                let m = m.clone();
                if self.fail_next_memcpy {
                    self.fail_next_memcpy = false;
                    if let Some(op) = self.ops.get_mut(&id) {
                        op.cancelled = true;
                        op.done = true;
                        op.done_ns = Some(self.clock);
                    }
                    return Err(SimError::TransferFailed { alloc: m.alloc });
                }
                self.require_memcpy_attach(stream, &m)?;
                self.memcpy_precheck(&m)?;
                if self.managed_local_copy(&m)? {
                    self.running.push(Running {
                        op: id,
                        remaining_ns: 1,
                        share: Share::Solo,
                    });
                    return Ok(true);
                }
                let gp = self.profile.gpu(device)?;
                if self.gpu_rt(device)?.copies >= gp.copy_engines {
                    return Ok(false);
                }
                self.charge_replica_hbm(&m)?;
                let (ns, link_idx) = self.memcpy_ns(&m)?;
                let ns = ns.saturating_add(self.graph_head_ns(device, launch)?);
                self.gpu_rt_mut(device)?.copies = self.gpu_rt(device)?.copies.saturating_add(1);
                self.running.push(Running {
                    op: id,
                    remaining_ns: ns.max(1),
                    share: Share::Link(link_idx),
                });
                Ok(true)
            }
            Kind::EventRecord { event, .. } => {
                let event = *event;
                if let Some(ev) = self.events.get_mut(&event) {
                    ev.recorded_by = Some(id);
                }
                self.running.push(Running {
                    op: id,
                    remaining_ns: 1,
                    share: Share::Solo,
                });
                Ok(true)
            }
            Kind::EventWait { event, external } => {
                let skip_graph =
                    *external && matches!(launch, LaunchCost::GraphHead | LaunchCost::GraphBody);
                if self.event_wait_gate(*event, skip_graph)?.is_none() {
                    return Ok(false);
                }
                self.running.push(Running {
                    op: id,
                    remaining_ns: 1,
                    share: Share::Solo,
                });
                Ok(true)
            }
            Kind::AllReduce { parts, bytes } => {
                let parts = parts.clone();
                let bytes = *bytes;
                self.start_allreduce(id, device, &parts, bytes)
            }
            Kind::ChildGraph { .. } => Err(SimError::Invalid {
                why: "child graph must be expanded",
            }),
        }
    }

    fn invalidate_read_mostly_writes(
        &mut self,
        device: DeviceId,
        writes: &[KernelBuf],
    ) -> Result<(), SimError> {
        let mut ids = BTreeSet::new();
        for b in writes {
            let a = self.alloc_ref(b.id)?;
            if a.live && a.managed && a.read_mostly {
                let _ins = ids.insert(b.id);
            }
        }
        for id in ids {
            self.migrate_off_except(id, device)?;
        }
        Ok(())
    }

    fn lease_kernel(
        &mut self,
        device: DeviceId,
        reads: &[KernelBuf],
        writes: &[KernelBuf],
        mapped_ok: bool,
    ) -> Result<(), SimError> {
        for b in reads {
            if !self.buf_on_device(b, device, mapped_ok, true)? {
                return Err(SimError::NotResident {
                    alloc: b.id,
                    device,
                });
            }
        }
        for b in writes {
            if !self.buf_on_device(b, device, mapped_ok, false)? {
                return Err(SimError::NotResident {
                    alloc: b.id,
                    device,
                });
            }
        }
        for b in reads.iter().chain(writes.iter()) {
            let a = self.alloc_mut(b.id)?;
            a.leases = a.leases.saturating_add(1);
        }
        Ok(())
    }

    fn buf_on_device(
        &self,
        buf: &KernelBuf,
        device: DeviceId,
        mapped_ok: bool,
        allow_remote: bool,
    ) -> Result<bool, SimError> {
        let root = {
            let a = self.alloc_ref(buf.id)?;
            a.share_src.or(a.ipc_src).unwrap_or(buf.id)
        };
        let a = self.alloc_ref(root)?;
        let (off, n) = kernel_span(a.bytes, buf)?;
        let on_device = if a.vmm {
            a.live && vmm_covers(&a.vmm_maps, device, off, n)
        } else {
            a.live && a.devices.contains(&device)
        };
        let mapped = mapped_ok && a.live && a.host_mapped;
        let home_span = a
            .vmm_home()
            .is_some_and(|h| vmm_covers(&a.vmm_maps, h, off, n));
        let accessed =
            mapped_ok && allow_remote && a.remote_read_ok(device) && (!a.vmm || home_span);
        let pool_ok = self.pool_peer_ok(a, device);
        let vmm_rw = a.vmm && a.vmm_write_by.contains(&device) && home_span;
        Ok(on_device || mapped || accessed || pool_ok || vmm_rw)
    }

    /// Peer ReadWrite via [`Self::pool_set_access`]. Physicals stay on the pool GPU.
    fn pool_peer_ok(&self, a: &Alloc, device: DeviceId) -> bool {
        if !a.live {
            return false;
        }
        let Some(pid) = a.pool else {
            return false;
        };
        let Ok(root_id) = self.pool_root(pid) else {
            return false;
        };
        let Some(root) = self.pools.get(&root_id) else {
            return false;
        };
        if root.device == device || !a.devices.contains(&root.device) {
            return false;
        }
        let local_ok = self
            .pools
            .get(&pid)
            .is_some_and(|p| p.accessed_by.contains(&device));
        local_ok || root.accessed_by.contains(&device)
    }

    fn peer_or_host_bps(&self, src: DeviceId, dst: DeviceId) -> Result<u64, SimError> {
        if let Ok(link) = self.profile.link(Some(src), Some(dst)) {
            return Ok(link.bps);
        }
        Ok(self.profile.link(None, Some(dst))?.bps)
    }

    fn kernel_mem_bps(
        &self,
        device: DeviceId,
        reads: &[KernelBuf],
        writes: &[KernelBuf],
    ) -> Result<u64, SimError> {
        let hbm = self.profile.gpu(device)?.hbm_bps;
        let mut bps = hbm;
        let mut remote = false;
        for b in reads.iter().chain(writes.iter()) {
            if let Some(&mc) = self.mc_vas.get(&b.id) {
                bps = bps.min(self.nvls_bps(device, mc)?);
                remote = true;
                continue;
            }
            let a = self.alloc_ref(b.id)?;
            if a.devices.contains(&device) {
                continue;
            }
            if a.host_mapped {
                let pcie = self.profile.link(None, Some(device))?.bps;
                bps = bps.min(pcie);
                remote = true;
                continue;
            }
            if a.remote_read_ok(device)
                || self.pool_peer_ok(a, device)
                || (a.vmm && a.vmm_write_by.contains(&device))
            {
                let src = if a.vmm {
                    a.vmm_home()
                } else if self.pool_peer_ok(a, device) {
                    a.pool
                        .and_then(|p| self.pools.get(&p).map(|pool| pool.device))
                } else {
                    match a.preferred {
                        Preferred::Gpu(p) if a.devices.contains(&p) => Some(p),
                        _ => a.devices.first().copied(),
                    }
                };
                let link = if let Some(src) = src {
                    self.peer_or_host_bps(src, device)?
                } else {
                    self.profile.link(None, Some(device))?.bps
                };
                bps = bps.min(link);
                remote = true;
            }
        }
        Ok(if remote { bps } else { hbm })
    }

    fn nvls_bps(&self, src: DeviceId, mc: MulticastId) -> Result<u64, SimError> {
        let team = self.mc_ref(mc)?;
        let mut bps = u64::MAX;
        let mut any = false;
        for &d in &team.devices {
            if d == src {
                continue;
            }
            any = true;
            bps = bps.min(self.peer_or_host_bps(src, d)?);
        }
        if !any {
            return Ok(self.profile.gpu(src)?.hbm_bps);
        }
        Ok(bps)
    }

    fn multicast_write_bytes(&self, writes: &[KernelBuf]) -> u64 {
        let mut n = 0u64;
        for w in writes {
            if !self.mc_vas.contains_key(&w.id) {
                continue;
            }
            let Ok(a) = self.alloc_ref(w.id) else {
                continue;
            };
            let Ok((_, m)) = kernel_span(a.bytes, w) else {
                continue;
            };
            n = n.saturating_add(m);
        }
        n
    }

    fn kernel_ns(
        &self,
        device: DeviceId,
        stream: StreamId,
        kind: &KernelKind,
        launch: LaunchCost,
        mem_bps: u64,
    ) -> Result<u64, SimError> {
        let g = self.profile.gpu(device)?;
        let (flops, bytes) = kind.flops_and_bytes();
        let sm = u64::from(self.stream_sm_permille(device, stream));
        let peak = g
            .flops(kind.dtype())
            .saturating_mul(sm)
            .checked_div(1000)
            .unwrap_or(1)
            .max(1);
        let compute = ns_for_bytes(flops, peak);
        let memory = ns_for_bytes(bytes, mem_bps.max(1));
        let overhead = match launch {
            LaunchCost::Kernel => g.launch_overhead_ns,
            LaunchCost::GraphHead => g.graph_launch_ns,
            LaunchCost::GraphBody => 0,
        };
        let mut ns = overhead.saturating_add(compute.max(memory));
        let util = u64::from(g.gemm_util_permille.max(1));
        ns = ns
            .saturating_mul(1000)
            .checked_div(util)
            .unwrap_or(u64::MAX);
        if matches!(kind, KernelKind::GroupedMoeGemm { .. }) {
            let pen = u64::from(g.grouped_moe_permille.max(1));
            ns = ns.saturating_mul(pen).checked_div(1000).unwrap_or(u64::MAX);
        }
        Ok(ns)
    }

    fn memset_ns(
        &self,
        device: DeviceId,
        bytes: u64,
        launch: LaunchCost,
        mem_bps: u64,
    ) -> Result<u64, SimError> {
        let g = self.profile.gpu(device)?;
        let overhead = match launch {
            LaunchCost::Kernel => g.launch_overhead_ns,
            LaunchCost::GraphHead => g.graph_launch_ns,
            LaunchCost::GraphBody => 0,
        };
        Ok(overhead.saturating_add(ns_for_bytes(bytes, mem_bps.max(1))))
    }

    fn host_func_ns(&self, device: DeviceId, launch: LaunchCost) -> Result<u64, SimError> {
        let g = self.profile.gpu(device)?;
        let head = match launch {
            LaunchCost::GraphHead => g.graph_launch_ns,
            LaunchCost::Kernel | LaunchCost::GraphBody => 0,
        };
        Ok(head.saturating_add(g.host_func_ns.max(1)))
    }

    fn graph_head_ns(&self, device: DeviceId, launch: LaunchCost) -> Result<u64, SimError> {
        match launch {
            LaunchCost::GraphHead => Ok(self.profile.gpu(device)?.graph_launch_ns),
            LaunchCost::Kernel | LaunchCost::GraphBody => Ok(0),
        }
    }

    fn event_wait_gate(&self, event: EventId, skip_graph: bool) -> Result<Option<OpId>, SimError> {
        if let Some(rec) = self.events.get(&event).and_then(|e| e.recorded_by) {
            let graph_rec = self.ops.get(&rec).is_some_and(|o| {
                skip_graph && matches!(o.launch, LaunchCost::GraphHead | LaunchCost::GraphBody)
            });
            if !graph_rec {
                if self.ops.get(&rec).is_some_and(|o| o.cancelled) {
                    let stream = self.ops.get(&rec).map(|o| o.stream).unwrap_or(StreamId(0));
                    return Err(SimError::Cancelled { stream, n: 1 });
                }
                if !self.op_done(rec) {
                    return Ok(None);
                }
                return Ok(Some(rec));
            }
        }
        let mut pending = false;
        let mut cancelled_n = 0u32;
        let mut stream = StreamId(0);
        for op in self.ops.values() {
            if let Kind::EventRecord { event: ev, .. } = &op.kind {
                if *ev == event {
                    if skip_graph
                        && matches!(op.launch, LaunchCost::GraphHead | LaunchCost::GraphBody)
                    {
                        continue;
                    }
                    if op.cancelled {
                        cancelled_n = cancelled_n.saturating_add(1);
                        stream = op.stream;
                    } else if !op.done {
                        pending = true;
                    }
                }
            }
        }
        if pending {
            return Ok(None);
        }
        if cancelled_n > 0 {
            return Err(SimError::Cancelled {
                stream,
                n: cancelled_n,
            });
        }
        Ok(None)
    }

    fn start_allreduce(
        &mut self,
        id: OpId,
        device: DeviceId,
        parts: &[(DeviceId, AllocId)],
        bytes: u64,
    ) -> Result<bool, SimError> {
        for (d, a) in parts {
            let alloc = self.alloc_ref(*a)?;
            if !alloc.live || !alloc.devices.contains(d) {
                return Err(SimError::NotResident {
                    alloc: *a,
                    device: *d,
                });
            }
        }
        let ns = self.allreduce_ns(parts, bytes)?;
        if !self.take_compute(device)? {
            return Ok(false);
        }
        for (_, a) in parts {
            let alloc = self.alloc_mut(*a)?;
            alloc.leases = alloc.leases.saturating_add(1);
        }
        self.running.push(Running {
            op: id,
            remaining_ns: ns.max(1),
            share: Share::Solo,
        });
        Ok(true)
    }

    fn allreduce_ns(&self, parts: &[(DeviceId, AllocId)], bytes: u64) -> Result<u64, SimError> {
        let n = parts.len();
        if n < 2 {
            return Err(SimError::Invalid {
                why: "allreduce needs >= 2 ranks",
            });
        }
        let mut worst = 0u64;
        for i in 0..n {
            let src = parts
                .get(i)
                .ok_or(SimError::Invalid {
                    why: "allreduce rank",
                })?
                .0;
            let j = if i.saturating_add(1) >= n {
                0
            } else {
                i.saturating_add(1)
            };
            let dst = parts
                .get(j)
                .ok_or(SimError::Invalid {
                    why: "allreduce rank",
                })?
                .0;
            let hop = self.profile.link(Some(src), Some(dst))?.copy_ns(bytes);
            if hop > worst {
                worst = hop;
            }
        }
        let hops = u64::try_from(n.saturating_sub(1)).unwrap_or(u64::MAX);
        Ok(worst
            .saturating_mul(hops)
            .saturating_add(self.extra_transfer_ns))
    }

    fn memcpy_precheck(&self, m: &MemcpyOp) -> Result<(), SimError> {
        let a = self.alloc_ref(m.alloc)?;
        if !a.live {
            return Err(SimError::UnknownAlloc { alloc: m.alloc });
        }
        if (a.vmm || m.offset > 0) && m.offset.saturating_add(m.bytes) > a.bytes {
            return Err(SimError::Invalid {
                why: "memcpy range past alloc",
            });
        }
        if a.managed && a.leases > 0 {
            let staying = match m.dst {
                Place::Device(d) => a.devices.contains(&d),
                Place::Host | Place::HostPinned => a.devices.is_empty(),
            };
            if !staying {
                return Err(SimError::Leased { alloc: m.alloc });
            }
        }
        if let (Place::Device(src), Place::Device(dst)) = (m.src, m.dst) {
            if src != dst
                && !self.peer_enabled.contains(&(src, dst))
                && self.profile.link(Some(src), Some(dst)).is_ok()
            {
                return Err(SimError::PeerDisabled { src, dst });
            }
        }
        match m.src {
            Place::Host | Place::HostPinned => {}
            Place::Device(d) => {
                let src_ok = if a.vmm {
                    vmm_covers(&a.vmm_maps, d, m.offset, m.bytes)
                } else {
                    a.devices.contains(&d)
                };
                if !src_ok {
                    return Err(SimError::NotResident {
                        alloc: m.alloc,
                        device: d,
                    });
                }
            }
        }
        if let Place::Device(d) = m.dst {
            if a.vmm && !vmm_covers(&a.vmm_maps, d, m.offset, m.bytes) {
                return Err(SimError::NotResident {
                    alloc: m.alloc,
                    device: d,
                });
            }
        }
        Ok(())
    }

    /// Replication onto a GPU that does not yet hold `m.alloc` charges that GPU's HBM.
    fn charge_replica_hbm(&mut self, m: &MemcpyOp) -> Result<(), SimError> {
        let Place::Device(dst) = m.dst else {
            return Ok(());
        };
        let a = self.alloc_ref(m.alloc)?;
        if a.devices.contains(&dst) {
            return Ok(());
        }
        let bytes = a.bytes;
        let cap = self.profile.gpu(dst)?.hbm_bytes;
        let used = self.gpu_rt(dst)?.used;
        let free = cap.saturating_sub(used);
        if bytes > free {
            return Err(SimError::Oom {
                device: dst,
                need: bytes,
                free,
            });
        }
        self.gpu_rt_mut(dst)?.used = used.saturating_add(bytes);
        let peak = self.gpu_rt(dst)?.used;
        if peak > self.hbm_peak {
            self.hbm_peak = peak;
        }
        Ok(())
    }

    fn memcpy_ns(&self, m: &MemcpyOp) -> Result<(u64, usize), SimError> {
        let src = m.src.device();
        let dst = m.dst.device();
        let idx = self
            .profile
            .links
            .iter()
            .position(|l| l.connects(src, dst))
            .ok_or(SimError::NoPeer {
                src: src.unwrap_or(DeviceId(u16::MAX)),
                dst: dst.unwrap_or(DeviceId(u16::MAX)),
            })?;
        let link = self
            .profile
            .links
            .get(idx)
            .ok_or(SimError::Invalid { why: "link index" })?;
        let copy = if m.src.is_pageable() || m.dst.is_pageable() {
            link.pageable_copy_ns(m.bytes)
        } else {
            link.copy_ns(m.bytes)
        };
        Ok((copy.saturating_add(self.extra_transfer_ns), idx))
    }

    fn share_n(&self, share: &Share) -> u64 {
        match *share {
            Share::Solo => 1,
            Share::Link(idx) => {
                let n = self
                    .running
                    .iter()
                    .filter(|r| matches!(r.share, Share::Link(i) if i == idx))
                    .count();
                u64::try_from(n.max(1)).unwrap_or(1)
            }
        }
    }

    fn advance_to_next_completion(&mut self) -> Result<(), SimError> {
        if self.running.is_empty() {
            return Ok(());
        }
        let mut min_dt = u64::MAX;
        for r in &self.running {
            let n = self.share_n(&r.share).max(1);
            let dt = r.remaining_ns.saturating_mul(n);
            if dt < min_dt {
                min_dt = dt;
            }
        }
        if min_dt == 0 || min_dt == u64::MAX {
            min_dt = 1;
        }
        self.clock = self.clock.saturating_add(min_dt);
        let shares: Vec<u64> = self
            .running
            .iter()
            .map(|r| self.share_n(&r.share).max(1))
            .collect();
        let mut finished = Vec::new();
        for (r, n) in self.running.iter_mut().zip(shares.iter()) {
            let drained = min_dt / (*n).max(1);
            r.remaining_ns = r.remaining_ns.saturating_sub(drained);
            if r.remaining_ns == 0 {
                finished.push(r.op);
            }
        }
        for id in finished {
            self.complete(id)?;
        }
        Ok(())
    }

    fn complete(&mut self, id: OpId) -> Result<(), SimError> {
        let dma = self
            .running
            .iter()
            .any(|r| r.op == id && matches!(r.share, Share::Link(_)));
        self.running.retain(|r| r.op != id);
        let device = self
            .ops
            .get(&id)
            .ok_or(SimError::Invalid {
                why: "complete unknown op",
            })?
            .device;
        let alloc_done = self.ops.get(&id).and_then(|op| match &op.kind {
            Kind::Alloc { id: alloc, .. } => Some((*alloc, device)),
            _ => None,
        });
        if let Some((alloc, device)) = alloc_done {
            let a = self.alloc_mut(alloc)?;
            a.live = true;
            if !a.devices.contains(&device) {
                a.devices.push(device);
            }
        }
        let kernel_work: Option<(Vec<AllocId>, bool)> = self.ops.get(&id).and_then(|op| match &op
            .kind
        {
            Kind::Kernel {
                reads,
                writes,
                cooperative,
                ..
            } => Some((
                reads.iter().chain(writes.iter()).map(|b| b.id).collect(),
                *cooperative,
            )),
            Kind::Memset { id: alloc, .. } => Some((vec![*alloc], false)),
            Kind::AllReduce { parts, .. } => Some((parts.iter().map(|(_, a)| *a).collect(), false)),
            _ => None,
        });
        let memcpy = self.ops.get(&id).and_then(|op| match &op.kind {
            Kind::Memcpy(m) => Some(m.clone()),
            _ => None,
        });
        let attach = self.ops.get(&id).and_then(|op| match &op.kind {
            Kind::Attach { id: alloc, flags } => Some((*alloc, *flags, op.stream)),
            _ => None,
        });
        let mc_bytes = self.ops.get(&id).and_then(|op| match &op.kind {
            Kind::Kernel { writes, .. } => Some(self.multicast_write_bytes(writes)),
            _ => None,
        });
        if let Some((alloc, flags, stream)) = attach {
            self.alloc_mut(alloc)?.attach = attach_state(flags, stream);
        }
        if let Some((ids, cooperative)) = kernel_work {
            let n = self.kernel_slots(device, cooperative)?;
            self.drop_compute_n(device, n)?;
            for a in ids {
                let cur = self.alloc_mut(a)?;
                cur.leases = cur.leases.saturating_sub(1);
            }
        }
        if let Some(n) = mc_bytes {
            self.bytes_moved = self.bytes_moved.saturating_add(n);
        }
        if let Some(m) = memcpy {
            self.finish_memcpy(device, m, dma)?;
        }
        if let Some(op) = self.ops.get_mut(&id) {
            op.done = true;
            op.done_ns = Some(self.clock);
        }
        Ok(())
    }
}

fn kernel_span(total: u64, buf: &KernelBuf) -> Result<(u64, u64), SimError> {
    if buf.offset > total {
        return Err(SimError::Invalid {
            why: "kernel range past alloc",
        });
    }
    let n = if buf.bytes == 0 {
        total.saturating_sub(buf.offset)
    } else {
        buf.bytes
    };
    if n == 0 || buf.offset.saturating_add(n) > total {
        return Err(SimError::Invalid {
            why: "kernel range past alloc",
        });
    }
    Ok((buf.offset, n))
}

fn nvlink_clique(profile: &HardwareProfile, devices: &[DeviceId]) -> Result<(), SimError> {
    if devices.len() < 2 {
        return Err(SimError::Invalid {
            why: "multicast needs NVLink",
        });
    }
    for (i, a) in devices.iter().enumerate() {
        for b in devices.iter().skip(i.saturating_add(1)) {
            match profile.link(Some(*a), Some(*b)) {
                Ok(l) if l.kind == LinkKind::Nvlink => {}
                _ => {
                    return Err(SimError::Invalid {
                        why: "multicast needs NVLink",
                    });
                }
            }
        }
    }
    Ok(())
}

fn multicast_team(src: DeviceId, dests: &[DeviceId]) -> Result<Vec<DeviceId>, SimError> {
    if dests.is_empty() {
        return Err(SimError::Invalid {
            why: "multicast needs NVLink",
        });
    }
    let mut team = vec![src];
    for &d in dests {
        if d == src {
            return Err(SimError::Invalid {
                why: "multicast src is dest",
            });
        }
        if team.contains(&d) {
            continue;
        }
        team.push(d);
    }
    if team.len() < 2 {
        return Err(SimError::Invalid {
            why: "multicast needs NVLink",
        });
    }
    Ok(team)
}

fn snapshot_op(id: OpId, o: &Op) -> Operation {
    Operation {
        id,
        device: o.device,
        stream: o.stream,
        kind: o.kind.clone(),
        deps: o.deps.clone(),
        done: o.done,
        cancelled: o.cancelled,
        submit_ns: o.submit_ns,
        start_ns: o.start_ns,
        done_ns: o.done_ns,
    }
}

fn vmm_overlap(maps: &[(DeviceId, u64, u64)], device: DeviceId, offset: u64, bytes: u64) -> bool {
    let end = offset.saturating_add(bytes);
    maps.iter()
        .any(|&(d, o, n)| d == device && offset < o.saturating_add(n) && o < end)
}

fn vmm_covers(maps: &[(DeviceId, u64, u64)], device: DeviceId, offset: u64, bytes: u64) -> bool {
    if bytes == 0 {
        return true;
    }
    let end = offset.saturating_add(bytes);
    let mut segs: Vec<(u64, u64)> = maps
        .iter()
        .filter(|(d, _, _)| *d == device)
        .map(|(_, o, n)| (*o, o.saturating_add(*n)))
        .collect();
    segs.sort_by_key(|s| s.0);
    let mut cur = offset;
    for (s, e) in segs {
        if s > cur {
            return false;
        }
        if e > cur {
            cur = e;
        }
        if cur >= end {
            return true;
        }
    }
    cur >= end
}

fn seed_peers(profile: &HardwareProfile) -> BTreeSet<(DeviceId, DeviceId)> {
    let mut out = BTreeSet::new();
    for link in &profile.links {
        let (Some(a), Some(b)) = (link.a, link.b) else {
            continue;
        };
        let _ab = out.insert((a, b));
        let _ba = out.insert((b, a));
    }
    out
}

fn remap_alloc_id(id: AllocId, map: &BTreeMap<AllocId, AllocId>) -> AllocId {
    map.get(&id).copied().unwrap_or(id)
}

fn remap_buf(buf: KernelBuf, map: &BTreeMap<AllocId, AllocId>) -> KernelBuf {
    KernelBuf {
        id: remap_alloc_id(buf.id, map),
        offset: buf.offset,
        bytes: buf.bytes,
    }
}

/// Rewrite alloc ids in captured ops after [`Sim::clone_graph`].
/// Unmapped ids stay (external pointers, not mem-alloc nodes).
fn remap_alloc_kind(kind: Kind, map: &BTreeMap<AllocId, AllocId>) -> Kind {
    match kind {
        Kind::Alloc { id, bytes } => Kind::Alloc {
            id: remap_alloc_id(id, map),
            bytes,
        },
        Kind::Free { id } => Kind::Free {
            id: remap_alloc_id(id, map),
        },
        Kind::Memcpy(op) => Kind::Memcpy(MemcpyOp {
            alloc: remap_alloc_id(op.alloc, map),
            src: op.src,
            dst: op.dst,
            bytes: op.bytes,
            offset: op.offset,
        }),
        Kind::Kernel {
            kind,
            reads,
            writes,
            cooperative,
        } => Kind::Kernel {
            kind,
            reads: reads.into_iter().map(|b| remap_buf(b, map)).collect(),
            writes: writes.into_iter().map(|b| remap_buf(b, map)).collect(),
            cooperative,
        },
        Kind::Memset { id, offset, bytes } => Kind::Memset {
            id: remap_alloc_id(id, map),
            offset,
            bytes,
        },
        Kind::Attach { id, flags } => Kind::Attach {
            id: remap_alloc_id(id, map),
            flags,
        },
        Kind::AllReduce { bytes, parts } => Kind::AllReduce {
            bytes,
            parts: parts
                .into_iter()
                .map(|(d, a)| (d, remap_alloc_id(a, map)))
                .collect(),
        },
        other => other,
    }
}

fn remap_child_graphs(
    steps: &[GraphStep],
    remap: &BTreeMap<GraphId, GraphId>,
) -> Result<Vec<GraphStep>, SimError> {
    let mut out = Vec::with_capacity(steps.len());
    for step in steps {
        let kind = match &step.kind {
            Kind::ChildGraph { graph } => {
                let cloned = remap.get(graph).copied().ok_or(SimError::Invalid {
                    why: "unknown graph",
                })?;
                Kind::ChildGraph { graph: cloned }
            }
            other => other.clone(),
        };
        out.push(GraphStep {
            device: step.device,
            stream: step.stream,
            kind,
            deps: step.deps.clone(),
        });
    }
    Ok(out)
}

fn graph_topology_eq(a: &[GraphStep], b: &[GraphStep]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| {
        x.device == y.device && x.stream == y.stream && op_eq(&x.kind, &y.kind) && x.deps == y.deps
    })
}

fn capture_step_deps(
    buf: &[GraphStep],
    device: DeviceId,
    stream: StreamId,
    kind: &Kind,
) -> Vec<usize> {
    let mut deps = Vec::new();
    if let Some(i) = buf
        .iter()
        .rposition(|s| s.device == device && s.stream == stream)
    {
        deps.push(i);
    }
    if let Kind::EventWait { event, external } = kind {
        if !*external {
            if let Some(i) = buf.iter().rposition(|s| {
                matches!(
                    &s.kind,
                    Kind::EventRecord {
                        event: e,
                        external: false
                    } if e == event
                )
            }) {
                if !deps.contains(&i) {
                    deps.push(i);
                }
            }
        }
    }
    deps
}

fn graph_topo_order(steps: &[GraphStep]) -> Result<Vec<usize>, SimError> {
    let n = steps.len();
    let mut indeg = vec![0u32; n];
    for (i, step) in steps.iter().enumerate() {
        let mut seen = BTreeSet::new();
        for d in &step.deps {
            if *d >= n || *d == i || !seen.insert(*d) {
                return Err(SimError::Invalid {
                    why: "graph dependency",
                });
            }
            let slot = indeg.get_mut(i).ok_or(SimError::Invalid {
                why: "graph dependency",
            })?;
            *slot = slot.saturating_add(1);
        }
    }
    let mut ready: BTreeSet<usize> = (0..n)
        .filter(|i| indeg.get(*i).copied() == Some(0))
        .collect();
    let mut order = Vec::with_capacity(n);
    while let Some(i) = ready.pop_first() {
        order.push(i);
        for (j, step) in steps.iter().enumerate() {
            if step.deps.contains(&i) {
                let slot = indeg.get_mut(j).ok_or(SimError::Invalid {
                    why: "graph dependency",
                })?;
                *slot = slot.saturating_sub(1);
                if *slot == 0 {
                    let _ins = ready.insert(j);
                }
            }
        }
    }
    if order.len() != n {
        return Err(SimError::Invalid {
            why: "cyclic graph dependencies",
        });
    }
    Ok(order)
}

fn graph_node_waits(
    step: &GraphStep,
    extra_wait: &[OpId],
    launch_tail: Option<OpId>,
    node_ops: &[Vec<OpId>],
) -> Result<Vec<OpId>, SimError> {
    let mut wait = Vec::new();
    if step.deps.is_empty() {
        wait.extend(extra_wait.iter().copied());
        if let Some(t) = launch_tail {
            wait.push(t);
        }
    }
    for d in &step.deps {
        let ops = node_ops.get(*d).ok_or(SimError::Invalid {
            why: "graph dependency",
        })?;
        wait.extend(ops.iter().copied());
    }
    Ok(wait)
}

fn graph_reaches(steps: &[GraphStep], start: usize, goal: usize) -> bool {
    let mut seen = BTreeSet::new();
    let mut stack = vec![start];
    while let Some(i) = stack.pop() {
        if i == goal {
            return true;
        }
        if !seen.insert(i) {
            continue;
        }
        for (j, step) in steps.iter().enumerate() {
            if step.deps.contains(&i) {
                stack.push(j);
            }
        }
    }
    false
}

fn alloc_graph_worker(launch: StreamId, worker: &mut u16) -> StreamId {
    loop {
        let s = StreamId(u16::MAX.saturating_sub(*worker));
        *worker = worker.saturating_add(1);
        if s != launch {
            return s;
        }
    }
}

fn op_eq(a: &Kind, b: &Kind) -> bool {
    match (a, b) {
        (Kind::ChildGraph { graph: x }, Kind::ChildGraph { graph: y }) => x == y,
        (Kind::EventRecord { external: x, .. }, Kind::EventRecord { external: y, .. }) => x == y,
        (Kind::EventWait { external: x, .. }, Kind::EventWait { external: y, .. }) => x == y,
        (Kind::Kernel { cooperative: x, .. }, Kind::Kernel { cooperative: y, .. }) => x == y,
        _ => op_tag(a) == op_tag(b),
    }
}

fn op_tag(k: &Kind) -> u8 {
    match k {
        Kind::Alloc { .. } => 0,
        Kind::Free { .. } => 1,
        Kind::Memcpy(_) => 2,
        Kind::Kernel { .. } => 3,
        Kind::Memset { .. } => 4,
        Kind::HostFunc => 5,
        Kind::EventRecord { .. } => 6,
        Kind::EventWait { .. } => 7,
        Kind::AllReduce { .. } => 8,
        Kind::ChildGraph { .. } => 9,
        Kind::Attach { .. } => 10,
    }
}

fn attach_state(flags: MemAttach, stream: StreamId) -> Attach {
    match flags {
        MemAttach::Global => Attach::Global,
        MemAttach::Host => Attach::Host,
        MemAttach::Single => Attach::Single(stream),
    }
}

fn seed_pools(
    profile: &HardwareProfile,
) -> (u32, BTreeMap<PoolId, Pool>, BTreeMap<DeviceId, PoolId>) {
    let mut pools = BTreeMap::new();
    let mut default_pools = BTreeMap::new();
    let mut next_pool = 1u32;
    for g in &profile.gpus {
        let id = PoolId(next_pool);
        next_pool = next_pool.saturating_add(1);
        let replaced = pools.insert(id, Pool::new(g.id));
        let _dup = replaced.is_some();
        let replaced = default_pools.insert(g.id, id);
        let _dup = replaced.is_some();
    }
    (next_pool, pools, default_pools)
}
