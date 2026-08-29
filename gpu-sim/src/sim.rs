//! Discrete-event GPU-systems simulator.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};

use crate::error::SimError;
use crate::ids::{AllocId, DeviceId, EventId, GraphId, OpId, StreamId};
use crate::ops::{KernelKind, MemcpyOp, Place};
use crate::profile::{ns_for_bytes, HardwareProfile};

struct Alloc {
    bytes: u64,
    devices: Vec<DeviceId>,
    leases: u32,
    live: bool,
}

struct Op {
    device: DeviceId,
    stream: StreamId,
    kind: Kind,
    deps: Vec<OpId>,
    done: bool,
    cancelled: bool,
}

#[derive(Clone)]
enum Kind {
    Alloc {
        id: AllocId,
        bytes: u64,
    },
    Free {
        id: AllocId,
    },
    Memcpy(MemcpyOp),
    Kernel {
        kind: KernelKind,
        reads: Vec<AllocId>,
        writes: Vec<AllocId>,
    },
    EventRecord {
        event: EventId,
    },
    EventWait {
        event: EventId,
    },
    AllReduce {
        parts: Vec<(DeviceId, AllocId)>,
        bytes: u64,
    },
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
}

struct Graph {
    steps: Vec<(DeviceId, Kind)>,
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
    capturing: Option<(DeviceId, StreamId)>,
    capture_buf: Vec<(DeviceId, Kind)>,
    running: Vec<Running>,
    gpus: BTreeMap<DeviceId, GpuRt>,
    bytes_moved: u64,
    hbm_peak: u64,
    unavailable: BTreeSet<DeviceId>,
    fail_next_memcpy: bool,
    extra_transfer_ns: u64,
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

    /// Bytes currently reserved on `device`.
    pub fn hbm_used(&self, device: DeviceId) -> Result<u64, SimError> {
        Ok(self.gpu_rt(device)?.used)
    }

    /// Whether `alloc` currently has a copy on `device`.
    pub fn is_resident(&self, alloc: AllocId, device: DeviceId) -> Result<bool, SimError> {
        let a = self.alloc_ref(alloc)?;
        Ok(a.live && a.devices.contains(&device))
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
    pub fn begin_capture(&mut self, device: DeviceId, stream: StreamId) -> Result<(), SimError> {
        let _gpu = self.profile.gpu(device)?;
        if self.capturing.is_some() {
            return Err(SimError::Invalid {
                why: "nested graph capture",
            });
        }
        let busy = self
            .ops
            .values()
            .any(|o| o.device == device && o.stream == stream && !o.done)
            || self.running.iter().any(|r| {
                self.ops
                    .get(&r.op)
                    .is_some_and(|o| o.device == device && o.stream == stream)
            });
        if busy {
            return Err(SimError::Invalid {
                why: "capture requires idle stream",
            });
        }
        self.capturing = Some((device, stream));
        self.capture_buf.clear();
        Ok(())
    }

    /// Finish capture. The graph is empty of side effects until [`Self::launch_graph`].
    pub fn end_capture(&mut self) -> Result<GraphId, SimError> {
        let Some(_) = self.capturing.take() else {
            return Err(SimError::Invalid {
                why: "end_capture without begin_capture",
            });
        };
        let id = GraphId(self.next_graph);
        self.next_graph = self.next_graph.saturating_add(1);
        let steps = core::mem::take(&mut self.capture_buf);
        let _prev = self.graphs.insert(id, Graph { steps });
        Ok(id)
    }

    /// Enqueue every recorded op, stream-ordered, on `stream` of each step's device.
    pub fn launch_graph(&mut self, graph: GraphId, stream: StreamId) -> Result<u32, SimError> {
        if self.capturing.is_some() {
            return Err(SimError::Invalid {
                why: "cannot launch during capture",
            });
        }
        let steps = self
            .graphs
            .get(&graph)
            .ok_or(SimError::Invalid {
                why: "unknown graph",
            })?
            .steps
            .clone();
        let mut n = 0u32;
        for (device, kind) in steps {
            let _op = self.submit(device, stream, kind)?;
            n = n.saturating_add(1);
        }
        Ok(n)
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

    /// Stream-ordered allocation. Capacity is reserved when the op starts.
    pub fn alloc(
        &mut self,
        device: DeviceId,
        bytes: u64,
        stream: StreamId,
    ) -> Result<AllocId, SimError> {
        if bytes == 0 {
            return Err(SimError::Invalid {
                why: "zero-byte alloc",
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
            },
        );
        let _op = self.submit(device, stream, Kind::Alloc { id, bytes })?;
        Ok(id)
    }

    /// Stream-ordered free. Illegal while a kernel lease is held.
    pub fn free(
        &mut self,
        device: DeviceId,
        id: AllocId,
        stream: StreamId,
    ) -> Result<(), SimError> {
        let _op = self.submit(device, stream, Kind::Free { id })?;
        Ok(())
    }

    /// Asynchronous copy. Completion moves/replicates residency.
    pub fn memcpy(
        &mut self,
        device: DeviceId,
        op: MemcpyOp,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        self.submit(device, stream, Kind::Memcpy(op))
    }

    /// Host → `device` convenience.
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
            },
            stream,
        )
    }

    /// Enqueue a kernel. Reads/writes are leased until it completes.
    pub fn kernel(
        &mut self,
        device: DeviceId,
        kind: KernelKind,
        reads: &[AllocId],
        writes: &[AllocId],
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        self.submit(
            device,
            stream,
            Kind::Kernel {
                kind,
                reads: reads.to_vec(),
                writes: writes.to_vec(),
            },
        )
    }

    /// Record `event` after prior ops on `stream`.
    pub fn record_event(
        &mut self,
        device: DeviceId,
        event: EventId,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        let _ev = self.events.entry(event).or_insert(Ev { recorded_by: None });
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
            let _ev = slot.insert(Ev { recorded_by: None });
        }
        self.submit(device, stream, Kind::EventWait { event })
    }

    /// Run until every submitted op is complete.
    pub fn synchronize(&mut self) -> Result<(), SimError> {
        let mut steps = 0u32;
        loop {
            self.schedule()?;
            if self.running.is_empty() {
                if self.ops.values().all(|o| o.done) {
                    return self.sync_outcome();
                }
                return Err(SimError::Invalid {
                    why: "deadlock: waiting ops but nothing running",
                });
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

    fn submit(&mut self, device: DeviceId, stream: StreamId, kind: Kind) -> Result<OpId, SimError> {
        if self.unavailable.contains(&device) {
            return Err(SimError::Unavailable { device });
        }
        if let Some((cd, cs)) = self.capturing {
            if device == cd && stream != cs {
                return Err(SimError::Invalid {
                    why: "other stream active during capture",
                });
            }
            if device == cd && stream == cs {
                if matches!(kind, Kind::Alloc { .. } | Kind::Free { .. }) {
                    return Err(SimError::Invalid {
                        why: "cannot capture alloc/free",
                    });
                }
                self.capture_buf.push((device, kind));
                let id = OpId(self.next_op);
                self.next_op = self.next_op.saturating_add(1);
                return Ok(id);
            }
        }
        let _gpu = self.profile.gpu(device)?;
        let id = OpId(self.next_op);
        self.next_op = self.next_op.saturating_add(1);
        let mut deps = Vec::new();
        if let Some(prev) = self.tail.get(&(device, stream)) {
            deps.push(*prev);
        }
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
            },
        );
        let _prev_tail = self.tail.insert((device, stream), id);
        Ok(id)
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
            let candidates: Vec<OpId> = self
                .ops
                .iter()
                .filter(|(id, o)| !o.done && !self.is_running(**id) && self.deps_ready(o))
                .map(|(id, _)| *id)
                .collect();
            for id in candidates {
                if self.try_start(id)? {
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
        if self.unavailable.contains(&device) {
            return Err(SimError::Unavailable { device });
        }
        match &op.kind {
            Kind::Alloc { bytes, .. } => {
                let bytes = *bytes;
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
                let ns = self.profile.gpu(device)?.alloc_overhead_ns;
                self.gpu_rt_mut(device)?.used = used.saturating_add(bytes);
                let peak = self.gpu_rt(device)?.used;
                if peak > self.hbm_peak {
                    self.hbm_peak = peak;
                }
                self.running.push(Running {
                    op: id,
                    remaining_ns: ns.max(1),
                    share: Share::Solo,
                });
                Ok(true)
            }
            Kind::Free { id: alloc } => {
                let alloc = *alloc;
                let a = self.alloc_ref(alloc)?;
                if a.leases > 0 {
                    return Err(SimError::Leased { alloc });
                }
                if !a.live || !a.devices.contains(&device) {
                    return Err(SimError::UnknownAlloc { alloc });
                }
                let bytes = a.bytes;
                self.gpu_rt_mut(device)?.used = self.gpu_rt(device)?.used.saturating_sub(bytes);
                let a = self.alloc_mut(alloc)?;
                a.devices.retain(|d| *d != device);
                if a.devices.is_empty() {
                    a.live = false;
                }
                self.running.push(Running {
                    op: id,
                    remaining_ns: 1,
                    share: Share::Solo,
                });
                Ok(true)
            }
            Kind::Kernel {
                reads,
                writes,
                kind,
            } => {
                if self.gpu_rt(device)?.compute_busy {
                    return Ok(false);
                }
                let reads = reads.clone();
                let writes = writes.clone();
                let kind = kind.clone();
                self.lease_kernel(device, &reads, &writes)?;
                let ns = self.kernel_ns(device, &kind)?;
                self.gpu_rt_mut(device)?.compute_busy = true;
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
                    }
                    return Err(SimError::TransferFailed { alloc: m.alloc });
                }
                let gp = self.profile.gpu(device)?;
                if self.gpu_rt(device)?.copies >= gp.copy_engines {
                    return Ok(false);
                }
                self.memcpy_precheck(&m)?;
                self.charge_replica_hbm(&m)?;
                let (ns, link_idx) = self.memcpy_ns(&m)?;
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
        }
    }

    fn lease_kernel(
        &mut self,
        device: DeviceId,
        reads: &[AllocId],
        writes: &[AllocId],
    ) -> Result<(), SimError> {
        for id in reads.iter().chain(writes.iter()) {
            let a = self.alloc_ref(*id)?;
            if !a.live || !a.devices.contains(&device) {
                return Err(SimError::NotResident { alloc: *id, device });
            }
        }
        for id in reads.iter().chain(writes.iter()) {
            let a = self.alloc_mut(*id)?;
            a.leases = a.leases.saturating_add(1);
        }
        Ok(())
    }

    fn kernel_ns(&self, device: DeviceId, kind: &KernelKind) -> Result<u64, SimError> {
        let g = self.profile.gpu(device)?;
        let (flops, bytes) = kind.flops_and_bytes();
        let compute = ns_for_bytes(flops, g.flops(kind.dtype()));
        let memory = ns_for_bytes(bytes, g.hbm_bps);
        Ok(g.launch_overhead_ns.saturating_add(compute.max(memory)))
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
        match m.src {
            Place::Host => Ok(()),
            Place::Device(d) => {
                if a.devices.contains(&d) {
                    Ok(())
                } else {
                    Err(SimError::NotResident {
                        alloc: m.alloc,
                        device: d,
                    })
                }
            }
        }
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
        let src = match m.src {
            Place::Host => None,
            Place::Device(d) => Some(d),
        };
        let dst = match m.dst {
            Place::Host => None,
            Place::Device(d) => Some(d),
        };
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
        Ok((
            link.copy_ns(m.bytes).saturating_add(self.extra_transfer_ns),
            idx,
        ))
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
                Some(reads.iter().chain(writes.iter()).copied().collect())
            }
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
            self.gpu_rt_mut(device)?.copies = self.gpu_rt(device)?.copies.saturating_sub(1);
            self.bytes_moved = self.bytes_moved.saturating_add(m.bytes);
            if let Place::Device(dst) = m.dst {
                let a = self.alloc_mut(m.alloc)?;
                if !a.devices.contains(&dst) {
                    a.devices.push(dst);
                }
            }
        }
        if let Some(op) = self.ops.get_mut(&id) {
            op.done = true;
        }
        Ok(())
    }
}
