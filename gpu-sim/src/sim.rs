//! Discrete-event GPU-systems simulator.

use std::cmp::Reverse;
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};

use crate::error::SimError;
use crate::ids::{AllocId, DeviceId, EventId, GraphId, OpId, PoolId, StreamId};
use crate::ops::{GpuOp as Kind, KernelBuf, KernelKind, MemAdvise, MemcpyOp, Operation, Place};
use crate::profile::{ns_for_bytes, HardwareProfile};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Preferred {
    None,
    Host,
    Gpu(DeviceId),
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
    /// `cudaMemAdviseSetReadMostly`: prefetch replicates instead of moving.
    read_mostly: bool,
    /// GPUs that may map this managed alloc without migrating (`SetAccessedBy`).
    accessed_by: BTreeSet<DeviceId>,
    /// `cudaMemAdviseSetPreferredLocation` (host or one GPU).
    preferred: Preferred,
    /// `cuMemAddressReserve` VA. HBM is charged only while mapped (possibly sparse).
    vmm: bool,
    /// Physical maps `(device, offset, bytes)` into a VMM VA.
    vmm_maps: Vec<(DeviceId, u64, u64)>,
    /// `None` is `cudaMalloc` / host-pinned. `Some` is `cudaMallocAsync` from that pool.
    pool: Option<PoolId>,
}

impl Alloc {
    fn remote_read_ok(&self, device: DeviceId) -> bool {
        if !self.live || !self.managed {
            return false;
        }
        if self.accessed_by.contains(&device) {
            return true;
        }
        match self.preferred {
            Preferred::Gpu(p) => self.devices.contains(&p) && p != device,
            Preferred::None | Preferred::Host => false,
        }
    }
}

struct Pool {
    device: DeviceId,
    live: u64,
    cached: u64,
    release_threshold: u64,
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
    compute_busy: bool,
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
}

struct Graph {
    steps: Vec<(DeviceId, StreamId, Kind)>,
    origin: (DeviceId, StreamId),
    /// `cudaGraphInstantiate` has run (explicit or first launch).
    instantiated: bool,
    /// `cudaGraphUpload` has run (explicit or first launch after instantiate).
    uploaded: bool,
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
    capture_buf: Vec<(DeviceId, StreamId, Kind)>,
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
    next_pool: u32,
    pools: BTreeMap<PoolId, Pool>,
    default_pools: BTreeMap<DeviceId, PoolId>,
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
                    compute_busy: false,
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
            next_pool,
            pools,
            default_pools,
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
        let _prev = self.graphs.insert(
            id,
            Graph {
                steps,
                origin: cap.origin,
                instantiated: false,
                uploaded: false,
            },
        );
        Ok(id)
    }

    /// Enqueue every recorded op. Origin-stream nodes use `stream`; forked
    /// streams keep the ids they joined with, so copy and compute can overlap.
    ///
    /// First launch [`Self::instantiate_graph`]s if needed (`cudaGraphInstantiate`
    /// then [`Self::upload_graph`] then `cudaGraphLaunch`). Later launches skip
    /// both. During capture
    /// on a captured stream this records a child-graph node (the child must
    /// already be instantiated). Independent streams still launch live.
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
        self.enqueue_graph(graph, stream, true, &mut stack)
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

    fn enqueue_graph(
        &mut self,
        graph: GraphId,
        stream: StreamId,
        head: bool,
        stack: &mut BTreeSet<GraphId>,
    ) -> Result<u32, SimError> {
        if !stack.insert(graph) {
            return Err(SimError::Invalid {
                why: "cyclic child graph",
            });
        }
        let (origin, steps) = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            (g.origin, g.steps.clone())
        };
        let launch_tail = self.tail.get(&(origin.0, stream)).copied();
        let mut n = 0u32;
        let mut head = head;
        let mut seen = BTreeSet::new();
        let mut rec_ops: BTreeMap<EventId, OpId> = BTreeMap::new();
        for (device, rec_stream, kind) in steps {
            let s = if (device, rec_stream) == origin {
                stream
            } else {
                rec_stream
            };
            if let Kind::ChildGraph { graph: child } = kind {
                n = n.saturating_add(self.enqueue_graph(child, s, head, stack)?);
                head = false;
                continue;
            }
            let rec = match &kind {
                Kind::EventRecord { event } => Some(*event),
                _ => None,
            };
            let wait = match &kind {
                Kind::EventWait { event } => Some(*event),
                _ => None,
            };
            let launch = if head {
                LaunchCost::GraphHead
            } else {
                LaunchCost::GraphBody
            };
            head = false;
            let id = self.submit_launch(device, s, kind, launch)?;
            if let Some(event) = rec {
                let _prev = rec_ops.insert(event, id);
            }
            if let Some(event) = wait {
                if let Some(rec_id) = rec_ops.get(&event).copied() {
                    self.add_op_dep(id, rec_id);
                }
            }
            if seen.insert((device, s)) {
                if let Some(tail) = launch_tail {
                    self.add_op_dep(id, tail);
                }
            }
            n = n.saturating_add(1);
        }
        let _gone = stack.remove(&graph);
        Ok(n)
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

    /// `cudaGraphInstantiate`. Host-synchronous. Capture cannot include it.
    ///
    /// Already-instantiated ids are a no-op. The first [`Self::launch_graph`]
    /// calls this when needed.
    pub fn instantiate_graph(&mut self, graph: GraphId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture graph instantiate")?;
        let (device, already) = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let device = g.steps.first().map(|(d, _, _)| *d).unwrap_or(DeviceId(0));
            (device, g.instantiated)
        };
        if already {
            return Ok(());
        }
        let ns = self.profile.gpu(device)?.graph_instantiate_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        self.graphs
            .get_mut(&graph)
            .ok_or(SimError::Invalid {
                why: "unknown graph",
            })?
            .instantiated = true;
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
            let device = g.steps.first().map(|(d, _, _)| *d).unwrap_or(DeviceId(0));
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
    /// Same device, stream, and op kinds; KernelBuf / memcpy sizes may change.
    /// Pays `graph_update_ns`. Recapture if topology differs. Capture cannot
    /// include it. `exec` must already be instantiated.
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
            let device = e.steps.first().map(|(d, _, _)| *d).unwrap_or(DeviceId(0));
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
        let ns = self.profile.gpu(device)?.graph_update_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        let exec = self.graphs.get_mut(&exec).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        exec.steps = src_steps;
        exec.uploaded = false;
        Ok(())
    }

    /// `cudaGraphClone`. Host-synchronous. Capture cannot include it.
    ///
    /// The clone is an independent graph (`instantiated = false`). Child-graph
    /// nodes are cloned recursively so the copy names new ids; a diamond of
    /// shared children becomes one cloned child. Instantiating or updating one
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
            let g = self.graphs.get(&src).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let steps = remap_child_graphs(&g.steps, &remap)?;
            let cloned = remap.get(&src).copied().ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            built.push((
                cloned,
                Graph {
                    steps,
                    origin: g.origin,
                    instantiated: false,
                    uploaded: false,
                },
                g.origin.0,
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
        for (_, _, kind) in &steps {
            if let Kind::ChildGraph { graph: child } = kind {
                self.collect_clone_tree(*child, walk)?;
            }
        }
        let _popped = walk.stack.pop();
        let _fresh = walk.seen.insert(graph);
        walk.order.push(graph);
        Ok(())
    }

    /// `cudaGraphDestroy` / `cudaGraphExecDestroy`. Host-synchronous.
    ///
    /// Capture cannot include it. Later [`Self::launch_graph`] of this id is
    /// `unknown graph`. Clones are independent.
    pub fn destroy_graph(&mut self, graph: GraphId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture graph destroy")?;
        let g = self.graphs.remove(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let device = g.origin.0;
        let _gpu = self.profile.gpu(device)?;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// Stream-ordered allocation (`cudaMallocAsync`) from the device default pool.
    ///
    /// Capacity is reserved when the op starts. The pointer is not usable until
    /// this stream catches up. [`Self::malloc`] is host-synchronous `cudaMalloc`.
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

    /// `cudaMemPoolCreate` for `device`. Release threshold starts at 0.
    pub fn create_pool(&mut self, device: DeviceId) -> Result<PoolId, SimError> {
        self.fail_if_capturing("cannot capture mempool")?;
        let _gpu = self.profile.gpu(device)?;
        let id = PoolId(self.next_pool);
        self.next_pool = self.next_pool.saturating_add(1);
        let _prev = self.pools.insert(
            id,
            Pool {
                device,
                live: 0,
                cached: 0,
                release_threshold: 0,
            },
        );
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
                read_mostly: false,
                accessed_by: BTreeSet::new(),
                preferred: Preferred::None,
                vmm: false,
                vmm_maps: Vec::new(),
                pool: Some(pool),
            },
        );
        let _op = self.submit(device, stream, Kind::Alloc { id, bytes })?;
        Ok(id)
    }

    /// `cudaMemPoolAttrReleaseThreshold`. Does not trim; later frees apply it.
    ///
    /// `0` (CUDA default) returns unused bytes to the OS when the stream-ordered
    /// free completes. `u64::MAX` holds them so [`Self::mem_info`] still counts
    /// them used until [`Self::pool_trim_to`].
    pub fn set_pool_release_threshold(&mut self, pool: PoolId, bytes: u64) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture mempool")?;
        self.pool_mut(pool)?.release_threshold = bytes;
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
        let (device, cached) = {
            let p = self.pool_ref(pool)?;
            (p.device, p.cached)
        };
        let drop = cached.saturating_sub(min_bytes);
        if drop == 0 {
            return Ok(0);
        }
        self.pool_mut(pool)?.cached = cached.saturating_sub(drop);
        let used = self.gpu_rt(device)?.used;
        self.gpu_rt_mut(device)?.used = used.saturating_sub(drop);
        Ok(drop)
    }

    /// Unused bytes held by `pool` (`cudaMemGetInfo` still counts them used).
    pub fn pool_cached(&self, pool: PoolId) -> Result<u64, SimError> {
        Ok(self.pool_ref(pool)?.cached)
    }

    /// Live bytes allocated from `pool` and not yet freed.
    pub fn pool_live(&self, pool: PoolId) -> Result<u64, SimError> {
        Ok(self.pool_ref(pool)?.live)
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
    /// device first-touch or [`Self::prefetch`].
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
                read_mostly: false,
                accessed_by: BTreeSet::new(),
                preferred: Preferred::None,
                vmm: false,
                vmm_maps: Vec::new(),
                pool: None,
            },
        );
        Ok(id)
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

    /// Whether `device` has [`MemAdvise::SetAccessedBy`] on `alloc`.
    pub fn is_accessed_by(&self, alloc: AllocId, device: DeviceId) -> Result<bool, SimError> {
        let a = self.alloc_ref(alloc)?;
        Ok(a.live && a.managed && a.accessed_by.contains(&device))
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
    /// Capture cannot include it.
    pub fn va_reserve(&mut self, bytes: u64) -> Result<AllocId, SimError> {
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
                managed: false,
                read_mostly: false,
                accessed_by: BTreeSet::new(),
                preferred: Preferred::None,
                vmm: true,
                vmm_maps: Vec::new(),
                pool: None,
            },
        );
        Ok(id)
    }

    /// Whether `alloc` is a live reserved VA (`cuMemAddressReserve`).
    pub fn is_vmm(&self, alloc: AllocId) -> Result<bool, SimError> {
        let a = self.alloc_ref(alloc)?;
        Ok(a.live && a.vmm)
    }

    /// `cuMemCreate` + `cuMemMap` + `cuMemSetAccess` for the whole VA.
    ///
    /// Equivalent to [`Self::va_map_range`] of `[0, bytes)`. Capture cannot include it.
    pub fn va_map(&mut self, id: AllocId, device: DeviceId) -> Result<(), SimError> {
        let bytes = self.alloc_ref(id)?.bytes;
        self.va_map_range(id, device, 0, bytes)
    }

    /// Map `[offset, offset+bytes)` of a reserved VA onto `device`.
    ///
    /// Host-synchronous. Charges `bytes` of HBM. Overlapping maps on the same
    /// device fail. [`Self::kernel`] needs the full VA covered; [`Self::kernel_bufs`]
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
        for (d, _, n) in maps {
            self.refund_device(d, id, n)?;
        }
        let a = self.alloc_mut(id)?;
        a.vmm_maps.clear();
        a.devices.clear();
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
        self.refund_device(device, id, bytes)?;
        let a = self.alloc_mut(id)?;
        let _gone = a.vmm_maps.remove(i);
        if !a.vmm_maps.iter().any(|&(d, _, _)| d == device) {
            a.devices.retain(|d| *d != device);
        }
        Ok(())
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
    pub fn free(
        &mut self,
        device: DeviceId,
        id: AllocId,
        stream: StreamId,
    ) -> Result<(), SimError> {
        let _op = self.submit(device, stream, Kind::Free { id })?;
        Ok(())
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
    /// A VMM VA must be fully mapped ([`Self::is_resident`]). For a mapped
    /// page of a larger VA, use [`Self::kernel_bufs`].
    ///
    /// A managed allocation not yet on `device` is [`Self::prefetch`]'d when
    /// the kernel starts (page fault after stream deps). Capture does not
    /// record that migrate; graph replay fails [`SimError::NotResident`] if
    /// the graph omitted it. Prefetch before [`Self::begin_capture`], or
    /// record [`Self::prefetch`] in the graph.
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
    /// Each [`KernelBuf`] must be mapped-host, device-resident, or a VMM span
    /// covered by [`Self::va_map_range`]. `bytes == 0` means from `offset` to
    /// the end of the allocation. A range past the reservation is `Invalid`.
    /// A live kernel (not a graph replay) page-faults managed memory when it
    /// *starts*, after stream deps, so a waited prefetch is visible.
    pub fn kernel_bufs(
        &mut self,
        device: DeviceId,
        kind: KernelKind,
        reads: &[KernelBuf],
        writes: &[KernelBuf],
        stream: StreamId,
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
            },
        )
    }

    /// Device-side fill (`cudaMemsetAsync`) of `[0, bytes)`.
    ///
    /// A VMM destination must have that span mapped ([`Self::is_range_resident`]).
    /// [`Self::kernel`] still needs the whole VA. [`Self::memset_buf`] names an
    /// interior page. Capture is allowed; alloc/free still are not.
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

    /// Record `event` after prior ops on `stream`.
    pub fn record_event(
        &mut self,
        device: DeviceId,
        event: EventId,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        let _ev = self.events.entry(event).or_insert(Ev {
            recorded_by: None,
            timing: true,
        });
        self.submit(device, stream, Kind::EventRecord { event })
    }

    /// Make later ops on `stream` wait until `event` is recorded and complete.
    pub fn wait_event(
        &mut self,
        device: DeviceId,
        event: EventId,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        if let Entry::Vacant(slot) = self.events.entry(event) {
            let _ev = slot.insert(Ev {
                recorded_by: None,
                timing: true,
            });
        }
        self.submit(device, stream, Kind::EventWait { event })
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
    /// Other streams keep running. A stream in an active graph capture is
    /// [`SimError::Invalid`]. Cancelled ops on *this* stream fail; cancelled
    /// work on other streams is left for a later [`Self::synchronize`].
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
        if let Kind::EventWait { event } = &kind {
            let join = self
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
        if matches!(kind, Kind::Alloc { .. } | Kind::Free { .. }) {
            return Err(SimError::Invalid {
                why: "cannot capture alloc/free",
            });
        }
        if let Kind::EventRecord { event } = &kind {
            if let Some(cap) = self.capturing.as_mut() {
                let _ins = cap.events.insert(*event);
            }
        }
        self.capture_buf.push((device, stream, kind));
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
        if let Kind::EventWait { event } = &kind {
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
        !self
            .ops
            .values()
            .any(|o| o.device == device && o.stream == stream && !o.done)
            && !self.running.iter().any(|r| {
                self.ops
                    .get(&r.op)
                    .is_some_and(|o| o.device == device && o.stream == stream)
            })
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
                read_mostly: false,
                accessed_by: BTreeSet::new(),
                preferred: Preferred::None,
                vmm: false,
                vmm_maps: Vec::new(),
                pool: None,
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
        let (device, launch, reads, writes, kind) = {
            let op = self
                .ops
                .get(&id)
                .ok_or(SimError::Invalid { why: "unknown op" })?;
            match &op.kind {
                Kind::Kernel {
                    reads,
                    writes,
                    kind,
                } => (
                    op.device,
                    op.launch,
                    reads.clone(),
                    writes.clone(),
                    kind.clone(),
                ),
                _ => {
                    return Err(SimError::Invalid {
                        why: "not a kernel",
                    });
                }
            }
        };
        if matches!(launch, LaunchCost::Kernel)
            && self.inject_managed_faults(id, device, &reads, &writes)?
        {
            return Ok(true);
        }
        if self.gpu_rt(device)?.compute_busy {
            return Ok(false);
        }
        let mem_bps = self.kernel_mem_bps(device, &reads, &writes)?;
        self.lease_kernel(device, &reads, &writes, true)?;
        self.invalidate_read_mostly_writes(device, &writes)?;
        let ns = self.kernel_ns(device, &kind, launch, mem_bps)?;
        self.gpu_rt_mut(device)?.compute_busy = true;
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
        let ns = if let Some(p) = pool {
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
                read_mostly: false,
                accessed_by: BTreeSet::new(),
                preferred: Preferred::None,
                vmm: false,
                vmm_maps: Vec::new(),
                pool: None,
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
        if self.unavailable.contains(&device) {
            return Err(SimError::Unavailable { device });
        }
        match &op.kind {
            Kind::Alloc { id: alloc, bytes } => self.start_alloc(id, device, *alloc, *bytes),
            Kind::Free { id: alloc } => self.start_free(id, device, *alloc),
            Kind::Kernel { .. } => self.start_kernel(id),
            Kind::Memset {
                id: alloc,
                offset,
                bytes,
            } => {
                if self.gpu_rt(device)?.compute_busy {
                    return Ok(false);
                }
                let alloc = *alloc;
                let offset = *offset;
                let bytes = *bytes;
                let writes = [KernelBuf::span(alloc, offset, bytes)];
                self.lease_kernel(device, &[], &writes, false)?;
                self.invalidate_read_mostly_writes(device, &writes)?;
                let ns = self.memset_ns(device, bytes, launch)?;
                self.gpu_rt_mut(device)?.compute_busy = true;
                self.running.push(Running {
                    op: id,
                    remaining_ns: ns.max(1),
                    share: Share::Solo,
                });
                Ok(true)
            }
            Kind::HostFunc => {
                let ns = self.host_func_ns(device, launch)?;
                self.running.push(Running {
                    op: id,
                    remaining_ns: ns.max(1),
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
            Kind::EventRecord { event } => {
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
            Kind::EventWait { event } => {
                let event = *event;
                if self.event_wait_gate(event)?.is_none() {
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
                if self.gpu_rt(device)?.compute_busy {
                    return Ok(false);
                }
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
        for b in reads.iter().chain(writes.iter()) {
            if !self.buf_on_device(b, device, mapped_ok)? {
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
    ) -> Result<bool, SimError> {
        let a = self.alloc_ref(buf.id)?;
        let (off, n) = kernel_span(a.bytes, buf)?;
        let on_device = if a.vmm {
            a.live && vmm_covers(&a.vmm_maps, device, off, n)
        } else {
            a.live && a.devices.contains(&device)
        };
        let mapped = mapped_ok && a.live && a.host_mapped;
        let accessed = mapped_ok && a.remote_read_ok(device);
        Ok(on_device || mapped || accessed)
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
            if a.managed && a.remote_read_ok(device) {
                let src = match a.preferred {
                    Preferred::Gpu(p) if a.devices.contains(&p) => Some(p),
                    _ => a.devices.first().copied(),
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

    fn kernel_ns(
        &self,
        device: DeviceId,
        kind: &KernelKind,
        launch: LaunchCost,
        mem_bps: u64,
    ) -> Result<u64, SimError> {
        let g = self.profile.gpu(device)?;
        let (flops, bytes) = kind.flops_and_bytes();
        let compute = ns_for_bytes(flops, g.flops(kind.dtype()));
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

    fn memset_ns(&self, device: DeviceId, bytes: u64, launch: LaunchCost) -> Result<u64, SimError> {
        let g = self.profile.gpu(device)?;
        let overhead = match launch {
            LaunchCost::Kernel => g.launch_overhead_ns,
            LaunchCost::GraphHead => g.graph_launch_ns,
            LaunchCost::GraphBody => 0,
        };
        Ok(overhead.saturating_add(ns_for_bytes(bytes, g.hbm_bps)))
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

    fn event_wait_gate(&self, event: EventId) -> Result<Option<OpId>, SimError> {
        if let Some(rec) = self.events.get(&event).and_then(|e| e.recorded_by) {
            if self.ops.get(&rec).is_some_and(|o| o.cancelled) {
                let stream = self.ops.get(&rec).map(|o| o.stream).unwrap_or(StreamId(0));
                return Err(SimError::Cancelled { stream, n: 1 });
            }
            if !self.op_done(rec) {
                return Ok(None);
            }
            return Ok(Some(rec));
        }
        let mut pending = false;
        let mut cancelled_n = 0u32;
        let mut stream = StreamId(0);
        for op in self.ops.values() {
            if let Kind::EventRecord { event: ev } = &op.kind {
                if *ev == event {
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
        for (_, a) in parts {
            let alloc = self.alloc_mut(*a)?;
            alloc.leases = alloc.leases.saturating_add(1);
        }
        self.gpu_rt_mut(device)?.compute_busy = true;
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
        let kernel_ids: Option<Vec<AllocId>> = self.ops.get(&id).and_then(|op| match &op.kind {
            Kind::Kernel { reads, writes, .. } => {
                Some(reads.iter().chain(writes.iter()).map(|b| b.id).collect())
            }
            Kind::Memset { id: alloc, .. } => Some(vec![*alloc]),
            Kind::AllReduce { parts, .. } => Some(parts.iter().map(|(_, a)| *a).collect()),
            _ => None,
        });
        let memcpy = self.ops.get(&id).and_then(|op| match &op.kind {
            Kind::Memcpy(m) => Some(m.clone()),
            _ => None,
        });
        if let Some(ids) = kernel_ids {
            self.gpu_rt_mut(device)?.compute_busy = false;
            for a in ids {
                let cur = self.alloc_mut(a)?;
                cur.leases = cur.leases.saturating_sub(1);
            }
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

fn remap_child_graphs(
    steps: &[(DeviceId, StreamId, Kind)],
    remap: &BTreeMap<GraphId, GraphId>,
) -> Result<Vec<(DeviceId, StreamId, Kind)>, SimError> {
    let mut out = Vec::with_capacity(steps.len());
    for (device, stream, kind) in steps {
        let kind = match kind {
            Kind::ChildGraph { graph } => {
                let cloned = remap.get(graph).copied().ok_or(SimError::Invalid {
                    why: "unknown graph",
                })?;
                Kind::ChildGraph { graph: cloned }
            }
            other => other.clone(),
        };
        out.push((*device, *stream, kind));
    }
    Ok(out)
}

fn graph_topology_eq(a: &[(DeviceId, StreamId, Kind)], b: &[(DeviceId, StreamId, Kind)]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|((d0, s0, k0), (d1, s1, k1))| *d0 == *d1 && *s0 == *s1 && op_eq(k0, k1))
}

fn op_eq(a: &Kind, b: &Kind) -> bool {
    match (a, b) {
        (Kind::ChildGraph { graph: x }, Kind::ChildGraph { graph: y }) => x == y,
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
        let replaced = pools.insert(
            id,
            Pool {
                device: g.id,
                live: 0,
                cached: 0,
                release_threshold: 0,
            },
        );
        let _dup = replaced.is_some();
        let replaced = default_pools.insert(g.id, id);
        let _dup = replaced.is_some();
    }
    (next_pool, pools, default_pools)
}
