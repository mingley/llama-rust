//! Discrete-event GPU-systems simulator.

use std::cmp::Reverse;
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};

use crate::error::SimError;
use crate::ids::{
    AllocId, CondId, DeviceId, EventId, GraphId, IpcEventHandleId, IpcHandleId, MemHandleId,
    MulticastId, OpId, PoolId, PtrExportId, ShareableHandleId, StreamId, UserObjectId,
};
use crate::ops::{
    AccessPolicyWindow, AccessProperty, BatchMemOp, CaptureDepOp, ClusterDim,
    ClusterSchedulingPolicy, DeviceAttr, DeviceLimit, DeviceP2pAttr, DeviceProperties,
    EventCreateFlags, EventRecordFlags, EventWaitFlags, FuncAttr, FuncAttributes, GpuOp as Kind,
    GraphAddNode, GraphDebugDotFlags, GraphExecUpdateResult, GraphExecUpdateResultInfo,
    GraphInstantiateFlags, GraphInstantiateParams, GraphInstantiateResult, GraphMemAttr,
    GraphNodeKind, GraphNodeParams, GraphUserObjectFlags, HostAllocFlags, HostNodeParams,
    KernelAttrs, KernelBuf, KernelKind, KernelNodeAttr, KernelNodeAttrValue, KernelNodeParams,
    LaunchCompletionEvent, MemAccessFlags, MemAdvise, MemAttach, MemHandleType, MemPoolAttr,
    MemRangeAttr, MemRangeAttrValue, MemSyncDomain, MemSyncDomainMap, MemcpyOp, MemoryType,
    MemsetOp, Operation, PdlLaunch, PeerAccessFlags, Place, PointerAttributes, PortableClusterMode,
    PortableSharedMode, ProgrammaticEvent, ProgrammaticLaunch, SharedMemCarveout, SharedMemoryMode,
    StreamAttr, StreamAttrValue, StreamCaptureInfo, StreamCaptureMode, StreamCreateFlags,
    SynchronizationPolicy, UserObjectFlags, WaitValueCmp,
};
use crate::profile::{align_up, ns_for_bytes, scale_ns_permille, HardwareProfile, LinkKind};

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
    /// Device graph-memory pool (`cudaDeviceGetGraphMemAttribute` backing).
    ///
    /// Capture [`Sim::alloc`] and [`Sim::graph_add_alloc`] draw from this pool.
    /// Release threshold is `u64::MAX` so unused bytes stay reserved until
    /// [`Sim::graph_mem_trim`].
    graph: bool,
    /// `cudaMemPoolDestroy`: handle is invalid; outstanding allocs stay valid.
    destroyed: bool,
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
            graph: false,
            destroyed: false,
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
    /// Conditional handles that skip this op at start when any predicate fails.
    preds: Vec<CondPred>,
    /// Predicates skipped this op; completion has no alloc/kernel/memcpy effects.
    skipped: bool,
    /// Scheduling priority (`cudaStreamCreateWithPriority`,
    /// `cudaLaunchAttributePriority`, or a kernel-node attribute when the exec
    /// used `cudaGraphInstantiateFlagUseNodePriority`).
    priority: i32,
    /// Programmatic dependent launch flags for this op.
    pdl: ProgrammaticLaunch,
    /// Clock when a [`ProgrammaticLaunch::trigger`] kernel signals completion.
    pdl_trigger_ns: Option<u64>,
    /// `cudaLaunchAttributeProgrammaticEvent` on this kernel, if any.
    programmatic_event: Option<ProgrammaticEvent>,
    /// `cudaLaunchAttributeLaunchCompletionEvent` on this kernel, if any.
    launch_completion: Option<LaunchCompletionEvent>,
    /// `cudaLaunchAttributeAccessPolicyWindow` on this kernel, if any.
    access_policy: Option<AccessPolicyWindow>,
    /// Resolved physical mem-sync domain (`cudaLaunchAttributeMemSyncDomain`).
    mem_sync_physical: u8,
    /// Completion fence already waited on same-domain leftover traffic.
    domain_fence_paid: bool,
    /// `cudaLaunchAttributeClusterDimension` on this kernel, if any.
    cluster: Option<ClusterDim>,
    /// `cudaLaunchAttributeClusterSchedulingPolicyPreference`.
    cluster_policy: ClusterSchedulingPolicy,
    /// `cudaLaunchAttributePreferredClusterDimension`.
    preferred_cluster: Option<ClusterDim>,
    /// `cudaLaunchAttributePreferredSharedMemoryCarveout`.
    carveout: SharedMemCarveout,
    /// `cudaLaunchAttributeSharedMemoryMode`.
    shared_mem: SharedMemoryMode,
    /// `cudaLaunchAttributeNvlinkUtilCentricScheduling`. Occupies every Hyper-Q
    /// slot when the profile has NVLink.
    nvlink_util_centric: bool,
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
    /// High-water of live graph-mem alloc bytes on this GPU.
    graph_used_high: u64,
    /// High-water of reserved graph-mem bytes (live + unused cached).
    graph_reserved_high: u64,
    /// `cudaLimitPersistingL2CacheSize`. CUDA default is 0.
    persist_limit: u64,
    /// Filled persisting-L2 ranges (insertion order is LRU).
    persist_lines: Vec<PersistLine>,
    /// `cudaDeviceGetLimit` values other than persisting L2.
    limits: DeviceLimits,
}

/// CUDA `cudaDeviceGetLimit` defaults (SM 8.0+ fetch granularity).
struct DeviceLimits {
    stack_size: u64,
    printf_fifo: u64,
    malloc_heap: u64,
    sync_depth: u64,
    pending_launch: u64,
    l2_fetch: u64,
}

impl DeviceLimits {
    fn sm80() -> Self {
        Self {
            stack_size: 1024,
            printf_fifo: 1 << 20,
            malloc_heap: 8 << 20,
            sync_depth: 2,
            pending_launch: 2048,
            l2_fetch: 128,
        }
    }
}

struct PersistLine {
    id: AllocId,
    offset: u64,
    bytes: u64,
}

struct Ev {
    recorded_by: Option<OpId>,
    /// `false` is `cudaEventDisableTiming`: [`Sim::event_elapsed_ns`] fails.
    timing: bool,
    /// `cudaEventInterprocess`. Required for [`Sim::ipc_get_event`].
    interprocess: bool,
    /// Import from [`Sim::ipc_open_event`]: record/wait follow this source.
    ipc_src: Option<EventId>,
    /// Outstanding [`Sim::ipc_open_event`] aliases. Destroy of the source is Invalid while > 0.
    ipc_opens: u32,
}

impl Ev {
    fn new(timing: bool) -> Self {
        Self {
            recorded_by: None,
            timing,
            interprocess: false,
            ipc_src: None,
            ipc_opens: 0,
        }
    }
}

struct Capture {
    origin: (DeviceId, StreamId),
    streams: BTreeSet<(DeviceId, StreamId)>,
    events: BTreeSet<EventId>,
    /// `cudaMallocAsync` ids recorded as graph mem alloc nodes.
    mem_allocs: Vec<AllocId>,
    /// Graph being captured into (created at [`Sim::begin_capture`] or passed
    /// to [`Sim::begin_capture_to_graph`]).
    into: CaptureInto,
    /// Per-stream extra deps for the next captured node
    /// (`cudaStreamUpdateCaptureDependencies`).
    pending: BTreeMap<(DeviceId, StreamId), Vec<usize>>,
    /// `capture_buf` index → extra deps in destination-graph index space.
    extra_abs: BTreeMap<usize, Vec<usize>>,
    /// Mode this session started with.
    mode: StreamCaptureMode,
    /// `cudaStreamGetCaptureInfo` `id_out` (unique per begin-capture sequence).
    id: u64,
}

/// Existing graph plus extra root deps for [`Capture::into`].
struct CaptureInto {
    graph: GraphId,
    /// Existing node indices that capture roots additionally depend on.
    deps: Vec<usize>,
}

/// One `cudaGraphAdd*` / captured node plus [`Sim::graph_add_dependencies`] edges.
#[derive(Clone, Debug)]
struct GraphStep {
    device: DeviceId,
    stream: StreamId,
    kind: Kind,
    /// Predecessor node indices (`cudaGraphAddDependencies`). Empty is independent.
    deps: Vec<usize>,
    /// `cudaGraphNodeSetEnabled`. Disabled nodes skip launch and complete immediately.
    enabled: bool,
    /// `cudaGraphDestroyNode`. Remaining node indices stay valid (CUDA handles).
    destroyed: bool,
    /// Stream or launch-attribute priority snapshotted at add/capture
    /// (`cudaKernelNodeAttributePriority` / `cudaLaunchAttributePriority`).
    priority: i32,
    /// Programmatic dependent launch (`cudaLaunchAttributeProgrammaticStreamSerialization`).
    pdl: ProgrammaticLaunch,
    /// `cudaLaunchAttributeProgrammaticEvent` on this kernel node, if any.
    programmatic_event: Option<ProgrammaticEvent>,
    /// `cudaLaunchAttributeLaunchCompletionEvent` on this kernel node, if any.
    launch_completion: Option<LaunchCompletionEvent>,
    /// `cudaLaunchAttributeAccessPolicyWindow` on this kernel node, if any.
    access_policy: Option<AccessPolicyWindow>,
    /// `cudaLaunchAttributeMemSyncDomain` on this kernel node.
    mem_sync_domain: MemSyncDomain,
    /// `cudaLaunchAttributeMemSyncDomainMap` on this kernel node.
    mem_sync_map: MemSyncDomainMap,
    /// `cudaLaunchAttributeClusterDimension` on this kernel node.
    cluster: Option<ClusterDim>,
    /// `cudaLaunchAttributeClusterSchedulingPolicyPreference`.
    cluster_policy: ClusterSchedulingPolicy,
    /// `cudaLaunchAttributePreferredClusterDimension`.
    preferred_cluster: Option<ClusterDim>,
    /// `cudaLaunchAttributePreferredSharedMemoryCarveout`.
    carveout: SharedMemCarveout,
    /// `cudaLaunchAttributeDeviceUpdatableKernelNode`.
    device_updatable: bool,
    /// `cudaLaunchAttributeSharedMemoryMode`.
    shared_mem: SharedMemoryMode,
    /// `cudaLaunchAttributePortableClusterSizeMode`.
    portable_cluster: PortableClusterMode,
    /// `cudaLaunchKernel` / `cudaKernelNodeParams::sharedMemBytes`.
    dynamic_shared: u32,
    /// CUDA 13 `cudaLaunchAttributeSharedMemoryMode` (`cudaSharedMemoryMode`).
    portable_shared: PortableSharedMode,
    /// `cudaLaunchAttributeNvlinkUtilCentricScheduling`.
    nvlink_util_centric: bool,
}

struct Graph {
    /// `cudaGraph_t` definition. [`Sim::graph_add_*`] / graph-side SetParams.
    steps: Vec<GraphStep>,
    /// `cudaGraphExec_t` snapshot, cloned at instantiate. Launch and
    /// `cudaGraphExec*SetParams` use this.
    exec: Option<Vec<GraphStep>>,
    origin: (DeviceId, StreamId),
    /// This id is a `cudaGraphExec_t` (not the `cudaGraph_t` definition).
    instantiated: bool,
    /// `cudaGraphUpload` has run (explicit or first launch after instantiate).
    uploaded: bool,
    /// `cudaGraphInstantiateFlagAutoFreeOnLaunch`: free graph mem before relaunch.
    auto_free_on_launch: bool,
    /// Flags passed to [`Sim::instantiate_graph_with_flags`] (`cudaGraphExecGetFlags`).
    instantiate_flags: u32,
    /// Last op of an in-flight [`Sim::device_launch_graph`] (launcher or body tail).
    device_launch_tail: Option<OpId>,
    /// First exec created from this definition. `None` on exec ids.
    primary_exec: Option<GraphId>,
    /// Definition this exec was instantiated from. `None` on definitions.
    src: Option<GraphId>,
}

impl Graph {
    fn view(&self) -> &[GraphStep] {
        self.exec.as_deref().unwrap_or(&self.steps)
    }

    fn exec_mut(&mut self) -> Result<&mut Vec<GraphStep>, SimError> {
        if !self.instantiated {
            return Err(SimError::Invalid {
                why: "graph not instantiated",
            });
        }
        self.exec.as_mut().ok_or(SimError::Invalid {
            why: "graph not instantiated",
        })
    }

    /// Exec id, or a definition that already has a primary exec.
    fn ready(&self) -> bool {
        self.instantiated || self.primary_exec.is_some()
    }
}

impl Sim {
    /// Exec id for `id`: itself when instantiated, else the primary exec.
    fn as_exec(&self, id: GraphId) -> Result<GraphId, SimError> {
        let g = self.graphs.get(&id).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        if g.instantiated {
            return Ok(id);
        }
        g.primary_exec.ok_or(SimError::Invalid {
            why: "graph not instantiated",
        })
    }

    fn remap_kind_to_exec(&self, kind: Kind) -> Result<Kind, SimError> {
        Ok(match kind {
            Kind::ChildGraph { graph } => Kind::ChildGraph {
                graph: self.resolved_graph(graph)?,
            },
            Kind::If { handle, body } => Kind::If {
                handle,
                body: self.resolved_graph(body)?,
            },
            Kind::While { handle, body } => Kind::While {
                handle,
                body: self.resolved_graph(body)?,
            },
            Kind::Switch { handle, bodies } => {
                let mut out = Vec::with_capacity(bodies.len());
                for b in bodies {
                    out.push(self.resolved_graph(b)?);
                }
                Kind::Switch {
                    handle,
                    bodies: out,
                }
            }
            other => other,
        })
    }

    /// Exec snapshot for `id`, or `id` itself when it is still a definition.
    fn resolved_graph(&self, graph: GraphId) -> Result<GraphId, SimError> {
        match self.as_exec(graph) {
            Ok(id) => Ok(id),
            Err(SimError::Invalid {
                why: "graph not instantiated",
            }) => Ok(graph),
            Err(e) => Err(e),
        }
    }

    fn def_id(&self, id: GraphId) -> GraphId {
        self.graphs.get(&id).and_then(|g| g.src).unwrap_or(id)
    }

    fn kind_def_ids(&self, kind: Kind) -> Kind {
        match kind {
            Kind::ChildGraph { graph } => Kind::ChildGraph {
                graph: self.def_id(graph),
            },
            Kind::If { handle, body } => Kind::If {
                handle,
                body: self.def_id(body),
            },
            Kind::While { handle, body } => Kind::While {
                handle,
                body: self.def_id(body),
            },
            Kind::Switch { handle, bodies } => Kind::Switch {
                handle,
                bodies: bodies.into_iter().map(|b| self.def_id(b)).collect(),
            },
            other => other,
        }
    }

    fn steps_def_ids(&self, steps: Vec<GraphStep>) -> Vec<GraphStep> {
        steps
            .into_iter()
            .map(|mut step| {
                step.kind = self.kind_def_ids(step.kind);
                step
            })
            .collect()
    }

    fn graph_mem_ids(&self, graph: GraphId) -> Vec<AllocId> {
        if let Some(v) = self.graph_allocs.get(&graph) {
            if !v.is_empty() {
                return v.clone();
            }
        }
        let src = self.graphs.get(&graph).and_then(|g| g.src);
        src.and_then(|s| self.graph_allocs.get(&s).cloned())
            .unwrap_or_default()
    }
}

/// Record vs wait for event-node SetEvent (definition and exec).
#[derive(Clone, Copy)]
enum EventSetKind {
    Record,
    Wait,
}

/// `cudaGraphConditionalHandle`.
struct Cond {
    graph: GraphId,
    default: u32,
    value: u32,
}

/// `cudaUserObject_t` refcounts. Destroy callback fires at zero.
struct UserObject {
    destroy_fn: u64,
    /// References held by the creating thread (`cudaUserObjectRetain`).
    caller: u32,
    /// References held by graph definitions (`cudaGraphRetainUserObject`).
    graphs: BTreeMap<GraphId, u32>,
}

/// Skip predicate for IF / WHILE / SWITCH body ops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CondPred {
    /// Skip when the handle is `0`.
    Nonzero(CondId),
    /// Skip when the handle is not `branch`.
    Equals(CondId, u32),
}

/// DFS state for [`Sim::clone_graph`]: unique graphs in post-order, ancestor stack.
struct CloneWalk {
    order: Vec<GraphId>,
    seen: BTreeSet<GraphId>,
    stack: Vec<GraphId>,
}

/// Fields copied out of a definition before [`Sim::instantiate_graph_inner`] mutates.
struct InstantiateSnap {
    device: DeviceId,
    already: bool,
    current_flags: u32,
    has_free: bool,
    device_launch_err: Option<usize>,
    has_mem: bool,
    mem_node: Option<usize>,
    free_node: Option<usize>,
    multi_dev: Option<usize>,
    primary: Option<GraphId>,
    origin: (DeviceId, StreamId),
    bodies: Vec<GraphId>,
    steps: Vec<GraphStep>,
}

/// Deterministic GPU node.
pub struct Sim {
    profile: HardwareProfile,
    clock: u64,
    next_op: u64,
    next_alloc: u64,
    next_graph: u32,
    /// Next `cudaStreamGetCaptureInfo` `id_out` (starts at 1).
    next_capture_id: u64,
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
    /// `cudaLaunchAttributeMemSyncDomain` per stream. Missing is Default.
    stream_mem_sync_domain: BTreeMap<(DeviceId, StreamId), MemSyncDomain>,
    /// `cudaLaunchAttributeMemSyncDomainMap` per stream. Missing is identity.
    stream_mem_sync_map: BTreeMap<(DeviceId, StreamId), MemSyncDomainMap>,
    /// `cudaLaunchAttributeSynchronizationPolicy` per stream. Missing is Auto.
    stream_sync_policy: BTreeMap<(DeviceId, StreamId), SynchronizationPolicy>,
    next_pool: u32,
    pools: BTreeMap<PoolId, Pool>,
    /// `cudaDeviceGetDefaultMemPool`. Seeded at construct; [`Self::set_device_mempool`]
    /// does not replace it.
    default_pools: BTreeMap<DeviceId, PoolId>,
    /// `cudaDeviceGetMemPool`. [`Self::alloc`] draws from this.
    current_pools: BTreeMap<DeviceId, PoolId>,
    /// Per-device graph memory pool (`cudaDeviceGraphMemTrim` backing).
    graph_pools: BTreeMap<DeviceId, PoolId>,
    next_handle: u64,
    mem_handles: BTreeMap<MemHandleId, MemHandle>,
    next_ipc: u64,
    ipc_handles: BTreeMap<IpcHandleId, AllocId>,
    next_ipc_event: u64,
    ipc_event_handles: BTreeMap<IpcEventHandleId, EventId>,
    next_imported_event: u32,
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
    next_cond: u32,
    conds: BTreeMap<CondId, Cond>,
    /// IF/WHILE/SWITCH body predicates applied to ops submitted by [`Self::enqueue_graph`].
    enqueue_preds: Vec<CondPred>,
    /// `cudaGraphClone` provenance: clone id → source id.
    clone_of: BTreeMap<GraphId, GraphId>,
    /// Override [`Op::priority`] for graph replay (`UseNodePriority`) or
    /// [`Self::kernel_with`] (`cudaLaunchAttributePriority`).
    enqueue_priority: Option<i32>,
    /// PDL flags for the next submit ([`Self::kernel_pdl`] / graph replay).
    enqueue_pdl: ProgrammaticLaunch,
    /// Programmatic event for the next kernel submit / graph replay.
    enqueue_programmatic_event: Option<ProgrammaticEvent>,
    /// Launch-completion event for the next kernel submit / graph replay.
    enqueue_launch_completion: Option<LaunchCompletionEvent>,
    /// Access-policy window for the next kernel submit / graph replay.
    enqueue_access_policy: Option<AccessPolicyWindow>,
    /// Mem-sync domain for the next kernel submit / graph replay.
    enqueue_mem_sync_domain: Option<MemSyncDomain>,
    /// Mem-sync map for the next kernel submit / graph replay.
    enqueue_mem_sync_map: Option<MemSyncDomainMap>,
    /// Cluster dimension for the next kernel submit / graph replay.
    enqueue_cluster: Option<ClusterDim>,
    /// Cluster scheduling policy for the next kernel submit / graph replay.
    enqueue_cluster_policy: ClusterSchedulingPolicy,
    /// Preferred cluster dimension for the next kernel submit / graph replay.
    enqueue_preferred_cluster: Option<ClusterDim>,
    /// Shared-memory carveout for the next kernel submit / graph replay.
    enqueue_carveout: SharedMemCarveout,
    /// Device-updatable kernel node for the next submit / graph replay.
    enqueue_device_updatable: bool,
    /// Shared-memory bank mode for the next submit / graph replay.
    enqueue_shared_mem: SharedMemoryMode,
    /// Portable-cluster mode for the next submit / graph replay.
    enqueue_portable_cluster: PortableClusterMode,
    /// Dynamic shared bytes for the next submit / graph replay.
    enqueue_dynamic_shared: u32,
    /// Portable-shared mode for the next submit / graph replay.
    enqueue_portable_shared: PortableSharedMode,
    /// NVLink-util-centric scheduling for the next submit / graph replay.
    enqueue_nvlink_util_centric: bool,
    /// Devices with `cudaFuncAttributeNonPortableClusterSizeAllowed`.
    non_portable_cluster: BTreeSet<DeviceId>,
    /// `cudaFuncAttributeMaxDynamicSharedMemorySize` per device (`0` = portable).
    max_dynamic_shared: BTreeMap<DeviceId, u32>,
    /// Streams with `cudaLaunchAttributeNvlinkUtilCentricScheduling` enabled.
    stream_nvlink_util_centric: BTreeSet<(DeviceId, StreamId)>,
    /// Wait/write-value mailbox: `(alloc, offset) → word`. Missing is `0`.
    mailbox: BTreeMap<(AllocId, u64), u64>,
    /// `cudaThreadExchangeStreamCaptureMode` default for [`Self::begin_capture`].
    capture_mode: StreamCaptureMode,
    next_user_object: u32,
    user_objects: BTreeMap<UserObjectId, UserObject>,
    /// `(id, destroy_fn)` in fire order. Handle is unknown after the last ref.
    user_object_dtors: Vec<(UserObjectId, u64)>,
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
                    graph_used_high: 0,
                    graph_reserved_high: 0,
                    persist_limit: 0,
                    persist_lines: Vec::new(),
                    limits: DeviceLimits::sm80(),
                },
            );
            let _dup = replaced.is_some();
        }
        let peer_enabled = seed_peers(&profile);
        let (mut next_pool, mut pools, default_pools) = seed_pools(&profile);
        let current_pools = default_pools.clone();
        let mut graph_pools = BTreeMap::new();
        for g in &profile.gpus {
            let gid = PoolId(next_pool);
            next_pool = next_pool.saturating_add(1);
            let mut gp = Pool::new(g.id);
            gp.release_threshold = u64::MAX;
            gp.graph = true;
            let replaced = pools.insert(gid, gp);
            let _dup = replaced.is_some();
            let replaced = graph_pools.insert(g.id, gid);
            let _dup = replaced.is_some();
        }
        Self {
            profile,
            clock: 0,
            next_op: 1,
            next_alloc: 1,
            next_graph: 1,
            next_capture_id: 1,
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
            stream_mem_sync_domain: BTreeMap::new(),
            stream_mem_sync_map: BTreeMap::new(),
            stream_sync_policy: BTreeMap::new(),
            next_pool,
            pools,
            default_pools,
            current_pools,
            graph_pools,
            next_handle: 1,
            mem_handles: BTreeMap::new(),
            next_ipc: 1,
            ipc_handles: BTreeMap::new(),
            next_ipc_event: 1,
            ipc_event_handles: BTreeMap::new(),
            next_imported_event: 1,
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
            next_cond: 1,
            conds: BTreeMap::new(),
            enqueue_preds: Vec::new(),
            clone_of: BTreeMap::new(),
            enqueue_priority: None,
            enqueue_pdl: ProgrammaticLaunch::default(),
            enqueue_programmatic_event: None,
            enqueue_launch_completion: None,
            enqueue_access_policy: None,
            enqueue_mem_sync_domain: None,
            enqueue_mem_sync_map: None,
            enqueue_cluster: None,
            enqueue_cluster_policy: ClusterSchedulingPolicy::Default,
            enqueue_preferred_cluster: None,
            enqueue_carveout: SharedMemCarveout::Default,
            enqueue_device_updatable: false,
            enqueue_shared_mem: SharedMemoryMode::Default,
            enqueue_portable_cluster: PortableClusterMode::Default,
            enqueue_dynamic_shared: 0,
            enqueue_portable_shared: PortableSharedMode::Default,
            enqueue_nvlink_util_centric: false,
            non_portable_cluster: BTreeSet::new(),
            max_dynamic_shared: BTreeMap::new(),
            stream_nvlink_util_centric: BTreeSet::new(),
            mailbox: BTreeMap::new(),
            capture_mode: StreamCaptureMode::Relaxed,
            next_user_object: 1,
            user_objects: BTreeMap::new(),
            user_object_dtors: Vec::new(),
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

    /// `cudaDeviceGetGraphMemAttribute` for the device graph-memory pool.
    ///
    /// Counts [`Self::graph_add_alloc`] / captured `cudaMallocAsync` from that
    /// pool, not ordinary [`Self::malloc`] / live [`Self::alloc`]. Used is live
    /// graph allocs. Reserved is live plus unused cached bytes held until
    /// [`Self::graph_mem_trim`]. Capture is allowed (query).
    pub fn graph_mem_get(&self, device: DeviceId, attr: GraphMemAttr) -> Result<u64, SimError> {
        let (used, reserved) = self.graph_mem_used_reserved(device)?;
        let rt = self.gpu_rt(device)?;
        Ok(match attr {
            GraphMemAttr::UsedMemCurrent => used,
            GraphMemAttr::ReservedMemCurrent => reserved,
            GraphMemAttr::UsedMemHigh => rt.graph_used_high.max(used),
            GraphMemAttr::ReservedMemHigh => rt.graph_reserved_high.max(reserved),
        })
    }

    /// `cudaDeviceSetGraphMemAttribute`. Only the High attrs; `value` must be `0`
    /// (reset that high-water to the current used/reserved). Host-synchronous.
    /// Capture cannot include it.
    pub fn graph_mem_set(
        &mut self,
        device: DeviceId,
        attr: GraphMemAttr,
        value: u64,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture graph mem set")?;
        let _gpu = self.profile.gpu(device)?;
        if value != 0 {
            return Err(SimError::Invalid {
                why: "graph mem attribute value",
            });
        }
        let (used, reserved) = self.graph_mem_used_reserved(device)?;
        let rt = self.gpu_rt_mut(device)?;
        match attr {
            GraphMemAttr::UsedMemHigh => rt.graph_used_high = used,
            GraphMemAttr::ReservedMemHigh => rt.graph_reserved_high = reserved,
            GraphMemAttr::UsedMemCurrent | GraphMemAttr::ReservedMemCurrent => {
                return Err(SimError::Invalid {
                    why: "graph mem attribute",
                });
            }
        }
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaDeviceGraphMemTrim`. Host-synchronous. Capture cannot include it.
    ///
    /// Returns unused reserved graph-mem bytes (cached after a graph free or
    /// [`Self::destroy_graph`]) to the OS so [`Self::mem_info`] free grows.
    /// Live graph allocs are not trimmed.
    pub fn graph_mem_trim(&mut self, device: DeviceId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture graph mem trim")?;
        let pool = self.graph_pool(device)?;
        let _dropped = self.pool_trim_to(pool, 0)?;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    fn graph_mem_used_reserved(&self, device: DeviceId) -> Result<(u64, u64), SimError> {
        let pool = self.graph_pool(device)?;
        let p = self.pool_ref(pool)?;
        let used = p.live;
        Ok((used, used.saturating_add(p.cached)))
    }

    fn is_graph_alloc(&self, id: AllocId) -> bool {
        self.graph_allocs.values().any(|v| v.contains(&id))
    }

    fn bump_graph_mem_high(&mut self, device: DeviceId) -> Result<(), SimError> {
        let (used, reserved) = self.graph_mem_used_reserved(device)?;
        let rt = self.gpu_rt_mut(device)?;
        rt.graph_used_high = rt.graph_used_high.max(used);
        rt.graph_reserved_high = rt.graph_reserved_high.max(reserved);
        Ok(())
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

    /// [`KernelAttrs::priority`] if set, else [`Self::stream_priority`].
    fn snap_priority(&self, device: DeviceId, stream: StreamId) -> i32 {
        self.enqueue_priority
            .unwrap_or_else(|| self.stream_priority(device, stream))
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

    /// `cudaDevAttrMemSyncDomainCount`.
    pub fn mem_sync_domain_count(&self, device: DeviceId) -> Result<u8, SimError> {
        Ok(self.profile.gpu(device)?.mem_sync_domain_count.max(1))
    }

    /// `cudaStreamSetAttribute` for `cudaLaunchAttributeMemSyncDomain`.
    pub fn set_stream_mem_sync_domain(
        &mut self,
        device: DeviceId,
        stream: StreamId,
        domain: MemSyncDomain,
    ) -> Result<(), SimError> {
        let _gpu = self.profile.gpu(device)?;
        let _prev = self.stream_mem_sync_domain.insert((device, stream), domain);
        Ok(())
    }

    /// Stream mem-sync domain, or [`MemSyncDomain::Default`] if unset.
    #[must_use]
    pub fn stream_mem_sync_domain(&self, device: DeviceId, stream: StreamId) -> MemSyncDomain {
        self.stream_mem_sync_domain
            .get(&(device, stream))
            .copied()
            .unwrap_or(MemSyncDomain::Default)
    }

    /// `cudaStreamSetAttribute` for `cudaLaunchAttributeMemSyncDomainMap`.
    pub fn set_stream_mem_sync_domain_map(
        &mut self,
        device: DeviceId,
        stream: StreamId,
        map: MemSyncDomainMap,
    ) -> Result<(), SimError> {
        self.validate_mem_sync_map(device, map)?;
        let _prev = self.stream_mem_sync_map.insert((device, stream), map);
        Ok(())
    }

    /// Stream mem-sync map, or CUDA identity for this device's domain count.
    pub fn stream_mem_sync_domain_map(
        &self,
        device: DeviceId,
        stream: StreamId,
    ) -> Result<MemSyncDomainMap, SimError> {
        if let Some(map) = self.stream_mem_sync_map.get(&(device, stream)).copied() {
            return Ok(map);
        }
        Ok(MemSyncDomainMap::identity(
            self.mem_sync_domain_count(device)?,
        ))
    }

    /// `cudaStreamSetAttribute` for `cudaLaunchAttributeSynchronizationPolicy`.
    ///
    /// Host-wait tax for [`Self::synchronize_stream`] / [`Self::synchronize_event`].
    /// Missing is [`SynchronizationPolicy::Auto`] (tax 0).
    pub fn set_stream_sync_policy(
        &mut self,
        device: DeviceId,
        stream: StreamId,
        policy: SynchronizationPolicy,
    ) -> Result<(), SimError> {
        let _gpu = self.profile.gpu(device)?;
        let _prev = self.stream_sync_policy.insert((device, stream), policy);
        Ok(())
    }

    /// Stream synchronization policy, or [`SynchronizationPolicy::Auto`] if unset.
    #[must_use]
    pub fn stream_sync_policy(&self, device: DeviceId, stream: StreamId) -> SynchronizationPolicy {
        self.stream_sync_policy
            .get(&(device, stream))
            .copied()
            .unwrap_or(SynchronizationPolicy::Auto)
    }

    /// `cudaStreamSetAttribute` for `cudaLaunchAttributeNvlinkUtilCentricScheduling`.
    ///
    /// Inherited by [`Self::kernel`] / [`Self::kernel_bufs`] on this stream.
    /// [`Self::kernel_with`] and graph replay use the launch / node value.
    /// Decode identity stays disabled.
    pub fn set_stream_nvlink_util_centric(
        &mut self,
        device: DeviceId,
        stream: StreamId,
        enabled: bool,
    ) -> Result<(), SimError> {
        let _gpu = self.profile.gpu(device)?;
        if enabled {
            let _ins = self.stream_nvlink_util_centric.insert((device, stream));
        } else {
            let _rm = self.stream_nvlink_util_centric.remove(&(device, stream));
        }
        Ok(())
    }

    /// Stream NVLink-util-centric flag, or `false` if unset.
    #[must_use]
    pub fn stream_nvlink_util_centric(&self, device: DeviceId, stream: StreamId) -> bool {
        self.stream_nvlink_util_centric.contains(&(device, stream))
    }

    /// `cudaStreamCopyAttributes`: copy priority, SM permille, mem-sync
    /// domain/map, synchronization policy, and NVLink-util-centric scheduling
    /// from `src` to `dst`.
    ///
    /// Same device required. Capture is allowed (host-side, not a graph node).
    pub fn stream_copy_attributes(
        &mut self,
        dst_device: DeviceId,
        dst: StreamId,
        src_device: DeviceId,
        src: StreamId,
    ) -> Result<(), SimError> {
        if dst_device != src_device {
            return Err(SimError::Invalid {
                why: "stream attribute device mismatch",
            });
        }
        let _gpu = self.profile.gpu(src_device)?;
        let pri = self.stream_priority(src_device, src);
        let sm = self.sm_permille.get(&(src_device, src)).copied();
        let domain = self.stream_mem_sync_domain.get(&(src_device, src)).copied();
        let map = self.stream_mem_sync_map.get(&(src_device, src)).copied();
        let sync = self.stream_sync_policy.get(&(src_device, src)).copied();
        let nvlink = self.stream_nvlink_util_centric.contains(&(src_device, src));
        self.set_stream_priority(dst_device, dst, pri)?;
        if let Some(sm) = sm {
            self.set_stream_sm_permille(dst_device, dst, sm)?;
        } else {
            let _gone = self.sm_permille.remove(&(dst_device, dst));
        }
        if let Some(domain) = domain {
            self.set_stream_mem_sync_domain(dst_device, dst, domain)?;
        } else {
            let _gone = self.stream_mem_sync_domain.remove(&(dst_device, dst));
        }
        if let Some(map) = map {
            self.set_stream_mem_sync_domain_map(dst_device, dst, map)?;
        } else {
            let _gone = self.stream_mem_sync_map.remove(&(dst_device, dst));
        }
        if let Some(sync) = sync {
            self.set_stream_sync_policy(dst_device, dst, sync)?;
        } else {
            let _gone = self.stream_sync_policy.remove(&(dst_device, dst));
        }
        self.set_stream_nvlink_util_centric(dst_device, dst, nvlink)?;
        Ok(())
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

    /// `cudaStreamCreateWithFlags`. Capture cannot include it.
    ///
    /// Known bit: [`StreamCreateFlags::NON_BLOCKING`]. Other bits are Invalid
    /// `"stream create flags"`. [`StreamId::NULL`] is Invalid (use
    /// [`Self::set_legacy_null_stream`]). Typed [`Self::set_stream_blocking`]
    /// stays. Created streams still default to non-blocking until this is
    /// called with [`StreamCreateFlags::DEFAULT`].
    pub fn stream_create_with_flags(
        &mut self,
        device: DeviceId,
        stream: StreamId,
        flags: u32,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture stream create")?;
        const KNOWN: u32 = StreamCreateFlags::NON_BLOCKING;
        if flags & !KNOWN != 0 {
            return Err(SimError::Invalid {
                why: "stream create flags",
            });
        }
        self.set_stream_blocking(device, stream, flags & StreamCreateFlags::NON_BLOCKING == 0)
    }

    /// `cudaStreamCreateWithPriority`. Capture cannot include it.
    ///
    /// Flags are [`Self::stream_create_with_flags`]. Priority is
    /// [`Self::set_stream_priority`]. This VM does not cap the range
    /// (`cudaDeviceGetStreamPriorityRange` is not modeled).
    pub fn stream_create_with_priority(
        &mut self,
        device: DeviceId,
        stream: StreamId,
        flags: u32,
        priority: i32,
    ) -> Result<(), SimError> {
        self.stream_create_with_flags(device, stream, flags)?;
        self.set_stream_priority(device, stream, priority)
    }

    /// Whether `stream` is a blocking `cudaStreamCreate` stream on `device`.
    #[must_use]
    pub fn stream_is_blocking(&self, device: DeviceId, stream: StreamId) -> bool {
        self.blocking.contains(&(device, stream))
    }

    /// `cudaStreamGetFlags`. Query; legal during capture.
    ///
    /// `0` is `cudaStreamDefault` (blocking). `1` is `cudaStreamNonBlocking`.
    /// [`StreamId::NULL`] uses [`Self::legacy_null_stream`] (off → NonBlocking).
    /// Unknown devices are Invalid. Any other stream id is legal (created
    /// streams default to NonBlocking).
    pub fn stream_get_flags(&self, device: DeviceId, stream: StreamId) -> Result<u32, SimError> {
        let _gpu = self.profile.gpu(device)?;
        let blocking = if stream == StreamId::NULL {
            self.legacy_null_stream
        } else {
            self.stream_is_blocking(device, stream)
        };
        Ok(u32::from(!blocking))
    }

    /// `cudaStreamGetPriority`. Query; legal during capture.
    ///
    /// Unset streams are `0`. Unknown devices are Invalid. This VM does not
    /// cap the range (`cudaDeviceGetStreamPriorityRange` is not modeled).
    pub fn stream_get_priority(&self, device: DeviceId, stream: StreamId) -> Result<i32, SimError> {
        let _gpu = self.profile.gpu(device)?;
        Ok(self.stream_priority(device, stream))
    }

    /// `cudaStreamGetId`. Query; legal during capture.
    ///
    /// Unique per `(device, stream)` for this VM. [`StreamId`] stays
    /// caller-chosen; this is not that handle and not a capture-sequence id.
    /// Unknown devices are Invalid. This VM does not invent
    /// `cudaStreamDestroy`.
    pub fn stream_get_id(&self, device: DeviceId, stream: StreamId) -> Result<u64, SimError> {
        let _gpu = self.profile.gpu(device)?;
        Ok((u64::from(device.0) << 16)
            .saturating_add(u64::from(stream.0))
            .saturating_add(1))
    }

    /// `cudaStreamGetAttribute`. Query; legal during capture.
    ///
    /// Wraps existing stream state only. Green-context SM permille is not a
    /// CUDA stream attribute.
    pub fn stream_get_attribute(
        &self,
        device: DeviceId,
        stream: StreamId,
        attr: StreamAttr,
    ) -> Result<StreamAttrValue, SimError> {
        let _gpu = self.profile.gpu(device)?;
        Ok(match attr {
            StreamAttr::Priority => StreamAttrValue::Priority(self.stream_priority(device, stream)),
            StreamAttr::SynchronizationPolicy => {
                StreamAttrValue::SynchronizationPolicy(self.stream_sync_policy(device, stream))
            }
            StreamAttr::MemSyncDomain => {
                StreamAttrValue::MemSyncDomain(self.stream_mem_sync_domain(device, stream))
            }
            StreamAttr::MemSyncDomainMap => {
                StreamAttrValue::MemSyncDomainMap(self.stream_mem_sync_domain_map(device, stream)?)
            }
            StreamAttr::NvlinkUtilCentric => {
                StreamAttrValue::NvlinkUtilCentric(self.stream_nvlink_util_centric(device, stream))
            }
        })
    }

    /// `cudaStreamSetAttribute`. Host-side; not a graph node.
    ///
    /// Same capture rule as the dedicated setters (legal during capture).
    /// Attr/value type mismatch is Invalid `"stream attr"`.
    pub fn stream_set_attribute(
        &mut self,
        device: DeviceId,
        stream: StreamId,
        attr: StreamAttr,
        value: StreamAttrValue,
    ) -> Result<(), SimError> {
        match (attr, value) {
            (StreamAttr::Priority, StreamAttrValue::Priority(p)) => {
                self.set_stream_priority(device, stream, p)
            }
            (StreamAttr::SynchronizationPolicy, StreamAttrValue::SynchronizationPolicy(policy)) => {
                self.set_stream_sync_policy(device, stream, policy)
            }
            (StreamAttr::MemSyncDomain, StreamAttrValue::MemSyncDomain(domain)) => {
                self.set_stream_mem_sync_domain(device, stream, domain)
            }
            (StreamAttr::MemSyncDomainMap, StreamAttrValue::MemSyncDomainMap(map)) => {
                self.set_stream_mem_sync_domain_map(device, stream, map)
            }
            (StreamAttr::NvlinkUtilCentric, StreamAttrValue::NvlinkUtilCentric(yes)) => {
                self.set_stream_nvlink_util_centric(device, stream, yes)
            }
            _ => Err(SimError::Invalid { why: "stream attr" }),
        }
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

    /// `cudaStreamSetAttribute` SynchronizationPolicy on streams `1 .. n_streams`.
    ///
    /// [`StreamId::NULL`] stays [`SynchronizationPolicy::Auto`]. `n_streams <= 1`
    /// is a no-op.
    pub fn set_created_streams_sync_policy(
        &mut self,
        n_streams: u8,
        policy: SynchronizationPolicy,
    ) -> Result<(), SimError> {
        let devices: Vec<DeviceId> = self.profile.gpus.iter().map(|g| g.id).collect();
        for d in devices {
            for s in 1..n_streams {
                self.set_stream_sync_policy(d, StreamId(u16::from(s)), policy)?;
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

    /// Slots a kernel occupies: cooperative grids and MaxShared carveout need
    /// the whole GPU. A Default/LoadBalancing cluster occupies
    /// `min(blocks, compute_slots)`. Spread occupies every slot. Preferred
    /// cluster size is used when it fits in `compute_slots`.
    fn kernel_slots(
        &self,
        device: DeviceId,
        cooperative: bool,
        cluster: Option<ClusterDim>,
        preferred: Option<ClusterDim>,
        policy: ClusterSchedulingPolicy,
        carveout: SharedMemCarveout,
    ) -> Result<u8, SimError> {
        let cap = self.profile.gpu(device)?.compute_slots.max(1);
        if cooperative || carveout.occupies_all_slots() {
            return Ok(cap);
        }
        let blocks = self.effective_cluster_blocks(device, cluster, preferred)?;
        if blocks <= 1 {
            return Ok(1);
        }
        if policy == ClusterSchedulingPolicy::Spread {
            return Ok(cap);
        }
        let n = blocks.min(u32::from(cap));
        Ok(u8::try_from(n).unwrap_or(cap).max(1))
    }

    fn op_kernel_slots(&self, op: &Op) -> Result<u8, SimError> {
        let cooperative = match &op.kind {
            Kind::Kernel { cooperative, .. } => *cooperative,
            _ => return Ok(1),
        };
        let slots = self.kernel_slots(
            op.device,
            cooperative,
            op.cluster,
            op.preferred_cluster,
            op.cluster_policy,
            op.carveout,
        )?;
        if op.nvlink_util_centric && self.profile.has_nvlink() {
            return Ok(self.profile.gpu(op.device)?.compute_slots.max(1));
        }
        Ok(slots)
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

    /// `cudaDeviceEnablePeerAccess` with a flags word.
    ///
    /// CUDA requires `flags == 0` ([`PeerAccessFlags::DEFAULT`]). Other bits
    /// are Invalid `"peer access flags"`. Typed [`Self::enable_peer`] stays.
    /// Capture is legal (same as the typed helper).
    pub fn enable_peer_with_flags(
        &mut self,
        src: DeviceId,
        dst: DeviceId,
        flags: u32,
    ) -> Result<(), SimError> {
        if flags != PeerAccessFlags::DEFAULT {
            return Err(SimError::Invalid {
                why: "peer access flags",
            });
        }
        self.enable_peer(src, dst)
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

    /// Recorded event that has fired ([`Self::query_event`] / wait).
    ///
    /// A [`ProgrammaticEvent`] with [`ProgrammaticLaunch::trigger`] fires at
    /// the PDL trigger, before the kernel completes. A normal
    /// [`Self::record_event`] fires when that record op completes.
    #[must_use]
    pub fn event_complete(&self, event: EventId) -> bool {
        self.event_is_recorded(event)
    }

    /// `cudaEventQuery`: whether `event` is recorded and complete. Does not wait.
    ///
    /// Unknown ids are [`SimError::UnknownEvent`]. Incomplete records are `Ok(false)`.
    /// An [`Self::ipc_open_event`] alias follows the source record.
    pub fn query_event(&self, event: EventId) -> Result<bool, SimError> {
        if !self.events.contains_key(&event) {
            return Err(SimError::UnknownEvent { event: event.0 });
        }
        Ok(self.event_complete(event))
    }

    fn event_root(&self, event: EventId) -> EventId {
        event_root_of(&self.events, event)
    }

    fn event_recorded_by(&self, event: EventId) -> Option<OpId> {
        self.events
            .get(&self.event_root(event))
            .and_then(|e| e.recorded_by)
    }

    fn event_is_recorded(&self, event: EventId) -> bool {
        let root = self.event_root(event);
        let Some(id) = self.event_recorded_by(root) else {
            return false;
        };
        let Some(op) = self.ops.get(&id) else {
            return false;
        };
        if op.cancelled {
            return false;
        }
        if op.done {
            return true;
        }
        if op
            .programmatic_event
            .is_some_and(|p| self.event_root(p.event) == root)
            && op.pdl.trigger
            && op.pdl_trigger_ns.is_some_and(|t| self.clock >= t)
        {
            return true;
        }
        op.launch_completion
            .is_some_and(|p| self.event_root(p.event) == root)
            && op.start_ns.is_some()
    }

    /// `cudaEventCreate` (timing enabled). Implicit on first [`Self::record_event`]
    /// if the id was never created.
    pub fn create_event(&mut self, event: EventId) -> Result<(), SimError> {
        self.create_event_with_flags(event, EventCreateFlags::DEFAULT)
    }

    /// `cudaEventCreateWithFlags(..., cudaEventDisableTiming)`.
    ///
    /// Record / wait / query still work. [`Self::event_elapsed_ns`] is
    /// [`SimError::Invalid`].
    pub fn create_event_disable_timing(&mut self, event: EventId) -> Result<(), SimError> {
        self.create_event_with_flags(event, EventCreateFlags::DISABLE_TIMING)
    }

    /// `cudaEventCreateWithFlags(..., cudaEventInterprocess | cudaEventDisableTiming)`.
    ///
    /// Required for [`Self::ipc_get_event`]. Timing is disabled (CUDA: Interprocess
    /// requires DisableTiming).
    pub fn create_event_interprocess(&mut self, event: EventId) -> Result<(), SimError> {
        self.create_event_with_flags(
            event,
            EventCreateFlags::DISABLE_TIMING | EventCreateFlags::INTERPROCESS,
        )
    }

    /// `cudaEventCreateWithFlags`. Capture cannot include it.
    ///
    /// Known bits: [`EventCreateFlags::DISABLE_TIMING`] /
    /// [`EventCreateFlags::INTERPROCESS`]. [`INTERPROCESS`](EventCreateFlags::INTERPROCESS)
    /// requires disable-timing (Invalid `"interprocess timing"` otherwise).
    /// Other bits (`cudaEventBlockingSync`) are Invalid `"event create flags"`.
    pub fn create_event_with_flags(&mut self, event: EventId, flags: u32) -> Result<(), SimError> {
        const KNOWN: u32 = EventCreateFlags::DISABLE_TIMING | EventCreateFlags::INTERPROCESS;
        if flags & !KNOWN != 0 {
            return Err(SimError::Invalid {
                why: "event create flags",
            });
        }
        let disable = flags & EventCreateFlags::DISABLE_TIMING != 0;
        let interprocess = flags & EventCreateFlags::INTERPROCESS != 0;
        if interprocess && !disable {
            return Err(SimError::Invalid {
                why: "interprocess timing",
            });
        }
        self.insert_event(event, !disable, interprocess)
    }

    /// `cudaEventDestroy`. Host-synchronous. Capture cannot include it.
    ///
    /// An event that was recorded and is not yet complete waits like
    /// [`Self::synchronize_event`]. A never-recorded event returns immediately.
    /// Unknown ids are [`SimError::UnknownEvent`]. The id may be created again.
    /// Destroy of an [`Self::ipc_open_event`] alias does not destroy the source.
    /// Destroy of a source with live imports is Invalid `"ipc mapped"`.
    pub fn destroy_event(&mut self, event: EventId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture event destroy")?;
        let (ipc_src, ipc_opens) = {
            let ev = self
                .events
                .get(&event)
                .ok_or(SimError::UnknownEvent { event: event.0 })?;
            (ev.ipc_src, ev.ipc_opens)
        };
        if ipc_opens > 0 {
            return Err(SimError::Invalid { why: "ipc mapped" });
        }
        if self.event_recorded_by(event).is_some() {
            self.synchronize_event(event)?;
        }
        if let Some(src) = ipc_src {
            let _gone = self.events.remove(&event);
            if let Some(s) = self.events.get_mut(&src) {
                s.ipc_opens = s.ipc_opens.saturating_sub(1);
            }
            return Ok(());
        }
        self.ipc_event_handles.retain(|_, e| *e != event);
        let _gone = self.events.remove(&event);
        Ok(())
    }

    /// Whether `event` was created with timing enabled (`cudaEventDefault`).
    pub fn event_timing(&self, event: EventId) -> Result<bool, SimError> {
        self.events
            .get(&event)
            .map(|e| e.timing)
            .ok_or(SimError::UnknownEvent { event: event.0 })
    }

    fn insert_event(
        &mut self,
        event: EventId,
        timing: bool,
        interprocess: bool,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture event create")?;
        if self.events.contains_key(&event) {
            return Err(SimError::Invalid {
                why: "event already created",
            });
        }
        let mut ev = Ev::new(timing);
        ev.interprocess = interprocess;
        let _prev = self.events.insert(event, ev);
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
    /// Default mode is [`StreamCaptureMode::Relaxed`] (or the last
    /// [`Self::thread_exchange_stream_capture_mode`]). Independent streams keep
    /// running. A stream that [`Self::wait_event`]s an event recorded in this
    /// capture **joins** (CUDA forked capture) so copy and compute can overlap
    /// inside one [`Self::launch_graph`]. [`Self::record_event_external`] /
    /// [`Self::wait_event_external`] do not join. Creates an empty graph (no
    /// clock tick); [`Self::end_capture`] appends recorded nodes and returns
    /// that id. For an existing graph see [`Self::begin_capture_to_graph`].
    /// [`Self::begin_capture_with_mode`] picks the mode for this capture only.
    pub fn begin_capture(&mut self, device: DeviceId, stream: StreamId) -> Result<(), SimError> {
        if self.capturing.is_some() {
            return Err(SimError::Invalid {
                why: "nested graph capture",
            });
        }
        let _gpu = self.profile.gpu(device)?;
        if !self.stream_idle(device, stream) {
            return Err(SimError::Invalid {
                why: "capture requires idle stream",
            });
        }
        let graph = self.insert_graph(device, stream);
        self.begin_capture_inner(device, stream, graph, &[], self.capture_mode)
    }

    /// `cudaStreamBeginCapture` with an explicit [`StreamCaptureMode`].
    ///
    /// Does not change the thread default ([`Self::thread_exchange_stream_capture_mode`]).
    pub fn begin_capture_with_mode(
        &mut self,
        device: DeviceId,
        stream: StreamId,
        mode: StreamCaptureMode,
    ) -> Result<(), SimError> {
        if self.capturing.is_some() {
            return Err(SimError::Invalid {
                why: "nested graph capture",
            });
        }
        let _gpu = self.profile.gpu(device)?;
        if !self.stream_idle(device, stream) {
            return Err(SimError::Invalid {
                why: "capture requires idle stream",
            });
        }
        let graph = self.insert_graph(device, stream);
        self.begin_capture_inner(device, stream, graph, &[], mode)
    }

    /// `cudaStreamBeginCaptureToGraph`: record later submits into `graph`.
    ///
    /// `graph` must already exist and must not be instantiated. Capture roots
    /// (nodes with no predecessors in this fragment) additionally depend on
    /// `deps` (existing node indices). Empty `deps` makes those nodes extra
    /// roots, so they may Hyper-Q overlap prior nodes at launch. Same device
    /// as `graph`'s origin. Nested capture is Invalid. Capture does not run
    /// recorded ops. [`Self::end_capture`] returns `graph`.
    pub fn begin_capture_to_graph(
        &mut self,
        device: DeviceId,
        stream: StreamId,
        graph: GraphId,
        deps: &[usize],
    ) -> Result<(), SimError> {
        self.begin_capture_inner(device, stream, graph, deps, self.capture_mode)
    }

    /// `cudaStreamBeginCaptureToGraph` with an explicit [`StreamCaptureMode`].
    pub fn begin_capture_to_graph_with_mode(
        &mut self,
        device: DeviceId,
        stream: StreamId,
        graph: GraphId,
        deps: &[usize],
        mode: StreamCaptureMode,
    ) -> Result<(), SimError> {
        self.begin_capture_inner(device, stream, graph, deps, mode)
    }

    fn begin_capture_inner(
        &mut self,
        device: DeviceId,
        stream: StreamId,
        graph: GraphId,
        deps: &[usize],
        mode: StreamCaptureMode,
    ) -> Result<(), SimError> {
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
        let into = self.capture_into(device, graph, deps)?;
        let mut streams = BTreeSet::new();
        let _ins = streams.insert((device, stream));
        let id = self.next_capture_id;
        self.next_capture_id = self.next_capture_id.saturating_add(1);
        self.capturing = Some(Capture {
            origin: (device, stream),
            streams,
            events: BTreeSet::new(),
            mem_allocs: Vec::new(),
            into,
            pending: BTreeMap::new(),
            extra_abs: BTreeMap::new(),
            mode,
            id,
        });
        self.capture_buf.clear();
        Ok(())
    }

    fn capture_into(
        &self,
        device: DeviceId,
        graph: GraphId,
        deps: &[usize],
    ) -> Result<CaptureInto, SimError> {
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        if g.instantiated {
            return Err(SimError::Invalid {
                why: "graph instantiated",
            });
        }
        if g.origin.0 != device {
            return Err(SimError::Invalid {
                why: "capture gpu mismatch",
            });
        }
        let n = g.steps.len();
        let mut extra = Vec::new();
        for &d in deps {
            if d >= n {
                return Err(SimError::Invalid {
                    why: "graph dependency",
                });
            }
            if !extra.contains(&d) {
                extra.push(d);
            }
        }
        extra.sort_unstable();
        Ok(CaptureInto { graph, deps: extra })
    }

    /// Finish capture. The graph is empty of side effects until [`Self::launch_graph`].
    ///
    /// Appends recorded nodes onto the graph from [`Self::begin_capture`] /
    /// [`Self::begin_capture_to_graph`] and returns that id.
    pub fn end_capture(&mut self) -> Result<GraphId, SimError> {
        let Some(cap) = self.capturing.take() else {
            return Err(SimError::Invalid {
                why: "end_capture without begin_capture",
            });
        };
        let steps = core::mem::take(&mut self.capture_buf);
        self.append_captured(cap.into, steps, cap.mem_allocs, cap.extra_abs)
    }

    /// `cudaStreamUpdateCaptureDependencies`: extra deps for the next captured
    /// node on this stream, **in addition to** stream-order (not instead of).
    ///
    /// `deps` are destination-graph indices: existing nodes `0..graph_len-1`,
    /// then this-session nodes at `graph_len + i`. [`Self::graph_len`] during
    /// capture does not include the session buffer. [`CaptureDepOp::Set`]
    /// replaces the pending set; [`CaptureDepOp::Add`] unions. The pending set
    /// is consumed by the next captured submit on this stream. The stream must
    /// be in the capture set. Same-stream independent children still need
    /// separate [`Self::begin_capture_to_graph`] sessions.
    pub fn stream_update_capture_dependencies(
        &mut self,
        device: DeviceId,
        stream: StreamId,
        deps: &[usize],
        mode: CaptureDepOp,
    ) -> Result<(), SimError> {
        let Some(cap) = self.capturing.as_ref() else {
            return Err(SimError::Invalid {
                why: "not capturing",
            });
        };
        if !cap.streams.contains(&(device, stream)) {
            return Err(SimError::Invalid {
                why: "stream not capturing",
            });
        }
        let existing = self
            .graphs
            .get(&cap.into.graph)
            .map_or(0, |g| g.steps.len());
        let hi = existing.saturating_add(self.capture_buf.len());
        for &p in deps {
            if p >= hi {
                return Err(SimError::Invalid {
                    why: "graph dependency",
                });
            }
        }
        let mut named = deps.to_vec();
        named.sort_unstable();
        named.dedup();
        let Some(cap) = self.capturing.as_mut() else {
            return Err(SimError::Invalid {
                why: "not capturing",
            });
        };
        match mode {
            CaptureDepOp::Set => {
                let _old = cap.pending.insert((device, stream), named);
            }
            CaptureDepOp::Add => {
                let slot = cap.pending.entry((device, stream)).or_default();
                slot.extend(named);
                slot.sort_unstable();
                slot.dedup();
            }
        }
        Ok(())
    }

    /// `cudaStreamIsCapturing`.
    #[must_use]
    pub fn stream_is_capturing(&self, device: DeviceId, stream: StreamId) -> bool {
        self.in_capture(device, stream)
    }

    /// `cudaStreamGetCaptureInfo`. `None` if this stream is not capturing.
    ///
    /// `pending_deps` are extra [`Self::stream_update_capture_dependencies`]
    /// indices not yet consumed (not stream-order predecessors).
    /// `dependencies` is the v2 array: last same-stream captured node union
    /// those extras (destination-graph indices). `id` is `id_out` (unique per
    /// sequence; forked streams share it). [`Self::graph_len`] of `info.graph`
    /// during capture excludes this session's buffer until
    /// [`Self::end_capture`].
    #[must_use]
    pub fn stream_capture_info(
        &self,
        device: DeviceId,
        stream: StreamId,
    ) -> Option<StreamCaptureInfo> {
        let cap = self.capturing.as_ref()?;
        if !cap.streams.contains(&(device, stream)) {
            return None;
        }
        let pending_deps = cap
            .pending
            .get(&(device, stream))
            .cloned()
            .unwrap_or_default();
        let existing = self
            .graphs
            .get(&cap.into.graph)
            .map_or(0, |g| g.steps.len());
        let mut dependencies = pending_deps.clone();
        if let Some(i) = self
            .capture_buf
            .iter()
            .rposition(|s| s.device == device && s.stream == stream)
        {
            dependencies.push(existing.saturating_add(i));
        }
        dependencies.sort_unstable();
        dependencies.dedup();
        Some(StreamCaptureInfo {
            graph: cap.into.graph,
            origin: cap.origin,
            pending_deps,
            dependencies,
            id: cap.id,
            mode: cap.mode,
        })
    }

    /// `cudaThreadExchangeStreamCaptureMode`. Returns the previous default.
    ///
    /// The next [`Self::begin_capture`] / [`Self::begin_capture_to_graph`] uses
    /// `mode`. An in-flight capture keeps the mode it started with.
    pub fn thread_exchange_stream_capture_mode(
        &mut self,
        mode: StreamCaptureMode,
    ) -> StreamCaptureMode {
        let prev = self.capture_mode;
        self.capture_mode = mode;
        prev
    }

    /// Thread default [`StreamCaptureMode`] for [`Self::begin_capture`].
    #[must_use]
    pub fn stream_capture_mode(&self) -> StreamCaptureMode {
        self.capture_mode
    }

    fn append_captured(
        &mut self,
        into: CaptureInto,
        steps: Vec<GraphStep>,
        mem_allocs: Vec<AllocId>,
        extra_abs: BTreeMap<usize, Vec<usize>>,
    ) -> Result<GraphId, SimError> {
        self.fail_capture_child_cycles(into.graph, &steps)?;
        let g = self.graphs.get_mut(&into.graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        if g.instantiated {
            return Err(SimError::Invalid {
                why: "graph instantiated",
            });
        }
        let offset = g.steps.len();
        for (i, mut step) in steps.into_iter().enumerate() {
            let was_root =
                step.deps.is_empty() && extra_abs.get(&i).is_none_or(|abs| abs.is_empty());
            for d in &mut step.deps {
                *d = d.saturating_add(offset);
            }
            if let Some(abs) = extra_abs.get(&i) {
                for extra in abs {
                    if !step.deps.contains(extra) {
                        step.deps.push(*extra);
                    }
                }
            }
            if was_root {
                for extra in &into.deps {
                    if !step.deps.contains(extra) {
                        step.deps.push(*extra);
                    }
                }
            }
            step.deps.sort_unstable();
            g.steps.push(step);
        }
        match self.graph_allocs.entry(into.graph) {
            Entry::Occupied(mut e) => e.get_mut().extend(mem_allocs),
            Entry::Vacant(e) => {
                let _prev = e.insert(mem_allocs);
            }
        }
        Ok(into.graph)
    }

    fn fail_capture_child_cycles(
        &self,
        parent: GraphId,
        steps: &[GraphStep],
    ) -> Result<(), SimError> {
        for step in steps {
            for child in nested_graphs(&step.kind) {
                if child == parent || self.graph_tree_contains(child, parent)? {
                    return Err(SimError::Invalid {
                        why: "cyclic child graph",
                    });
                }
            }
        }
        Ok(())
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
        let (origin, ready) = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            (g.origin, g.ready())
        };
        if self.in_capture(origin.0, stream) {
            return self.capture_child_graph(graph, origin.0, stream, ready);
        }
        let exec = self.ensure_exec(graph)?;
        let uploaded = self
            .graphs
            .get(&exec)
            .ok_or(SimError::Invalid {
                why: "unknown graph",
            })?
            .uploaded;
        if !uploaded && self.capturing.is_none() {
            self.upload_graph(exec)?;
        }
        self.reset_graph_tree_conds(exec)?;
        let mut stack = BTreeSet::new();
        self.enqueue_graph(exec, stream, true, &mut stack, &[])
    }

    /// Device-side `cudaGraphLaunch` (`cudaGraphInstantiateFlagDeviceLaunch`).
    ///
    /// The exec must already be instantiated with [`GraphInstantiateFlags::DEVICE_LAUNCH`]
    /// and uploaded ([`Self::upload_graph`]; host [`Self::launch_graph`] still
    /// auto-uploads). Submits a launcher kernel that occupies one compute slot
    /// for `graph_launch_ns`. When it completes the body is enqueued on
    /// `stream`. A second device launch of the same exec before that work
    /// finishes is Invalid. Capture cannot include it.
    /// [`Self::update_graph`] of a device-launch exec is Invalid.
    pub fn device_launch_graph(
        &mut self,
        graph: GraphId,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        self.fail_if_capturing("cannot capture device launch")?;
        let exec = self.as_exec(graph)?;
        let (device, flags, uploaded, tail) = {
            let g = self.graphs.get(&exec).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            (
                g.origin.0,
                g.instantiate_flags,
                g.uploaded,
                g.device_launch_tail,
            )
        };
        if flags & GraphInstantiateFlags::DEVICE_LAUNCH == 0 {
            return Err(SimError::Invalid {
                why: "graph not device launch",
            });
        }
        if !uploaded {
            return Err(SimError::Invalid {
                why: "graph not uploaded",
            });
        }
        if tail.is_some_and(|id| !self.op_done(id)) {
            return Err(SimError::Invalid {
                why: "device launch in flight",
            });
        }
        let id = self.submit(device, stream, Kind::DeviceLaunch { graph: exec })?;
        self.graphs
            .get_mut(&exec)
            .ok_or(SimError::Invalid {
                why: "unknown graph",
            })?
            .device_launch_tail = Some(id);
        Ok(id)
    }

    fn reset_graph_tree_conds(&mut self, root: GraphId) -> Result<(), SimError> {
        let mut stack = vec![root];
        let mut seen = BTreeSet::new();
        while let Some(g) = stack.pop() {
            if !seen.insert(g) {
                continue;
            }
            let src = self.graphs.get(&g).and_then(|gr| gr.src);
            for c in self.conds.values_mut() {
                if c.graph == g || src == Some(c.graph) {
                    c.value = c.default;
                }
            }
            let steps = self
                .graphs
                .get(&g)
                .map(|x| x.view().to_vec())
                .unwrap_or_default();
            for step in steps {
                stack.extend(nested_graphs(&step.kind));
            }
        }
        Ok(())
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
            let ids = self.graph_mem_ids(graph);
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
            (g.origin, g.view().to_vec())
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
            if self.graph_uses_node_priority(graph) {
                self.enqueue_priority = Some(step.priority);
            } else {
                self.enqueue_priority = None;
            }
            self.enqueue_pdl = step.pdl;
            self.enqueue_programmatic_event = step.programmatic_event;
            self.enqueue_launch_completion = step.launch_completion;
            self.enqueue_access_policy = step.access_policy;
            self.enqueue_mem_sync_domain = Some(step.mem_sync_domain);
            self.enqueue_mem_sync_map = Some(step.mem_sync_map);
            self.enqueue_cluster = step.cluster;
            self.enqueue_cluster_policy = step.cluster_policy;
            self.enqueue_preferred_cluster = step.preferred_cluster;
            self.enqueue_carveout = step.carveout;
            self.enqueue_device_updatable = step.device_updatable;
            self.enqueue_shared_mem = step.shared_mem;
            self.enqueue_portable_cluster = step.portable_cluster;
            self.enqueue_dynamic_shared = step.dynamic_shared;
            self.enqueue_portable_shared = step.portable_shared;
            self.enqueue_nvlink_util_centric = step.nvlink_util_centric;
            let wait = graph_node_waits(step, extra_wait, launch_tail, &node_ops)?;
            let s = self.graph_exec_stream(origin, stream, step, &node_stream, &mut worker);
            if let Some(slot) = node_stream.get_mut(idx) {
                *slot = Some(s);
            }
            if step.destroyed {
                continue;
            }
            if !step.enabled {
                if let Some(ops) = node_ops.get_mut(idx) {
                    ops.extend(wait);
                }
                continue;
            }
            if let Kind::ChildGraph { graph: child } = &step.kind {
                let child = self.resolved_graph(*child)?;
                let add = self.enqueue_graph(child, s, head, stack, &wait)?;
                head = false;
                n = n.saturating_add(add);
                self.note_nested_tail(
                    step.device,
                    s,
                    stream,
                    idx,
                    &mut node_ops,
                    &mut pending_joins,
                );
                continue;
            }
            if let Kind::If { handle, body } = &step.kind {
                let add = self.enqueue_pred_graph(
                    CondPred::Nonzero(*handle),
                    self.resolved_graph(*body)?,
                    s,
                    head,
                    stack,
                    &wait,
                )?;
                head = false;
                n = n.saturating_add(add);
                self.note_nested_tail(
                    step.device,
                    s,
                    stream,
                    idx,
                    &mut node_ops,
                    &mut pending_joins,
                );
                continue;
            }
            if let Kind::While { handle, body } = &step.kind {
                let handle = *handle;
                let body = self.resolved_graph(*body)?;
                let add = self.enqueue_pred_graph(
                    CondPred::Nonzero(handle),
                    body,
                    s,
                    head,
                    stack,
                    &wait,
                )?;
                head = false;
                n = n.saturating_add(add);
                let body_tail = self.tail.get(&(step.device, s)).copied();
                self.enqueue_preds.push(CondPred::Nonzero(handle));
                let tick = self.submit_launch(
                    step.device,
                    s,
                    Kind::WhileTick {
                        handle,
                        body,
                        iter: 1,
                    },
                    LaunchCost::GraphBody,
                );
                let _pop = self.enqueue_preds.pop();
                let tick = tick?;
                for dep in &wait {
                    self.add_op_dep(tick, *dep);
                }
                if let Some(t) = body_tail {
                    if t != tick {
                        self.add_op_dep(tick, t);
                    }
                }
                n = n.saturating_add(1);
                self.note_nested_tail(
                    step.device,
                    s,
                    stream,
                    idx,
                    &mut node_ops,
                    &mut pending_joins,
                );
                continue;
            }
            if let Kind::Switch { handle, bodies } = &step.kind {
                let handle = *handle;
                let bodies = bodies.clone();
                for (i, body) in bodies.into_iter().enumerate() {
                    let idx_u = u32::try_from(i).map_err(|_| SimError::Invalid {
                        why: "switch branches",
                    })?;
                    let add = self.enqueue_pred_graph(
                        CondPred::Equals(handle, idx_u),
                        self.resolved_graph(body)?,
                        s,
                        head,
                        stack,
                        &wait,
                    )?;
                    head = false;
                    n = n.saturating_add(add);
                }
                self.note_nested_tail(
                    step.device,
                    s,
                    stream,
                    idx,
                    &mut node_ops,
                    &mut pending_joins,
                );
                continue;
            }
            let rec = match &step.kind {
                Kind::EventRecord { event, .. } => Some(*event),
                _ => step
                    .programmatic_event
                    .map(|p| p.event)
                    .or_else(|| step.launch_completion.map(|p| p.event)),
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
                let _prev = rec_ops.insert(self.event_root(event), id);
            }
            if let Some((event, external)) = wait_ev {
                let root = self.event_root(event);
                if external {
                    if let Some(rec_id) = self.event_recorded_by(root) {
                        self.add_op_dep(id, rec_id);
                    }
                } else if let Some(rec_id) = rec_ops.get(&root).copied() {
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
        self.enqueue_priority = None;
        self.enqueue_pdl = ProgrammaticLaunch::default();
        self.enqueue_programmatic_event = None;
        self.enqueue_launch_completion = None;
        self.enqueue_access_policy = None;
        self.enqueue_mem_sync_domain = None;
        self.enqueue_mem_sync_map = None;
        self.enqueue_cluster = None;
        self.enqueue_cluster_policy = ClusterSchedulingPolicy::Default;
        self.enqueue_preferred_cluster = None;
        self.enqueue_carveout = SharedMemCarveout::Default;
        self.enqueue_device_updatable = false;
        self.enqueue_shared_mem = SharedMemoryMode::Default;
        self.enqueue_portable_cluster = PortableClusterMode::Default;
        self.enqueue_dynamic_shared = 0;
        self.enqueue_portable_shared = PortableSharedMode::Default;
        self.enqueue_nvlink_util_centric = false;
        Ok(n)
    }

    fn graph_uses_node_priority(&self, graph: GraphId) -> bool {
        self.graphs
            .get(&graph)
            .is_some_and(|g| g.instantiate_flags & GraphInstantiateFlags::USE_NODE_PRIORITY != 0)
    }

    fn enqueue_pred_graph(
        &mut self,
        pred: CondPred,
        body: GraphId,
        stream: StreamId,
        head: bool,
        stack: &mut BTreeSet<GraphId>,
        wait: &[OpId],
    ) -> Result<u32, SimError> {
        self.enqueue_preds.push(pred);
        let nested = self.enqueue_graph(body, stream, head, stack, wait);
        let _pop = self.enqueue_preds.pop();
        nested
    }

    fn note_nested_tail(
        &mut self,
        device: DeviceId,
        s: StreamId,
        stream: StreamId,
        idx: usize,
        node_ops: &mut [Vec<OpId>],
        pending_joins: &mut Vec<OpId>,
    ) {
        if let Some(id) = self.tail.get(&(device, s)).copied() {
            if let Some(ops) = node_ops.get_mut(idx) {
                ops.push(id);
            }
            if s != stream {
                pending_joins.push(id);
            }
        }
        if s != stream {
            if let Some(ids) = self.graph_joins.get(&(device, s)).cloned() {
                pending_joins.extend(ids);
            }
        }
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

    /// Add-order slot count (including [`Self::graph_destroy_node`] tombstones).
    ///
    /// During capture this is the destination graph only; this session's
    /// buffer is not included until [`Self::end_capture`]. Live nodes are
    /// [`Self::graph_nodes`].
    pub fn graph_len(&self, graph: GraphId) -> Result<usize, SimError> {
        self.graphs
            .get(&graph)
            .map(|g| g.steps.len())
            .ok_or(SimError::Invalid {
                why: "unknown graph",
            })
    }

    /// `cudaGraphGetNodes` — live node indices in creation order.
    ///
    /// Query; legal during capture. During capture this is the destination
    /// graph only. [`Self::graph_destroy_node`] tombstones a slot, so this may
    /// skip indices; [`Self::graph_len`] stays the add-order bound.
    pub fn graph_nodes(&self, graph: GraphId) -> Result<Vec<usize>, SimError> {
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        Ok(g.steps
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.destroyed)
            .map(|(i, _)| i)
            .collect())
    }

    /// Whether [`Self::instantiate_graph`] (or a first launch) has created an exec.
    ///
    /// True for an exec id, or a definition that has a primary exec.
    pub fn graph_instantiated(&self, graph: GraphId) -> Result<bool, SimError> {
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        Ok(g.instantiated || g.primary_exec.is_some())
    }

    /// Whether [`Self::upload_graph`] (or a first launch after instantiate) has run.
    pub fn graph_uploaded(&self, graph: GraphId) -> Result<bool, SimError> {
        let exec = self.as_exec(graph)?;
        self.graphs
            .get(&exec)
            .map(|g| g.uploaded)
            .ok_or(SimError::Invalid {
                why: "unknown graph",
            })
    }

    /// Whether [`Self::instantiate_graph_auto_free`] was used
    /// (`cudaGraphInstantiateFlagAutoFreeOnLaunch`).
    pub fn graph_auto_free_on_launch(&self, graph: GraphId) -> Result<bool, SimError> {
        let exec = self.as_exec(graph)?;
        self.graphs
            .get(&exec)
            .map(|g| g.auto_free_on_launch)
            .ok_or(SimError::Invalid {
                why: "unknown graph",
            })
    }

    /// `cudaGraphInstantiate`. Host-synchronous. Capture cannot include it.
    ///
    /// Returns a new exec id (`cudaGraphExec_t`). The source graph stays a
    /// definition and may be instantiated again (kernel/memcpy graphs). The
    /// first [`Self::launch_graph`] of the definition creates a primary exec
    /// if needed. Already-instantiated **exec** ids are a no-op.
    pub fn instantiate_graph(&mut self, graph: GraphId) -> Result<GraphId, SimError> {
        self.instantiate_graph_with_flags(graph, 0)
    }

    /// `cudaGraphInstantiate` with `cudaGraphInstantiateFlagAutoFreeOnLaunch`.
    ///
    /// Host-synchronous. Capture cannot include it. Graph mem allocs are
    /// `cudaFreeAsync`'d on the launch stream before a later launch's alloc
    /// nodes run, so relaunch recharges HBM instead of reusing the pointer.
    /// Illegal when the graph has mem free nodes. A second instantiate of a
    /// definition that has mem nodes is Invalid (execs would need independent
    /// pointers).
    pub fn instantiate_graph_auto_free(&mut self, graph: GraphId) -> Result<GraphId, SimError> {
        self.instantiate_graph_with_flags(graph, GraphInstantiateFlags::AUTO_FREE_ON_LAUNCH)
    }

    /// `cudaGraphInstantiateWithFlags`. Host-synchronous. Capture cannot include it.
    ///
    /// Returns a new exec id. [`GraphInstantiateFlags::AUTO_FREE_ON_LAUNCH`]
    /// matches [`Self::instantiate_graph_auto_free`]. [`GraphInstantiateFlags::UPLOAD`]
    /// host-sync uploads during instantiate so the first launch skips
    /// [`Self::upload_graph`]. [`GraphInstantiateFlags::USE_NODE_PRIORITY`]
    /// schedules recorded kernels with the priority snapshotted at add/capture
    /// instead of the launch stream. [`GraphInstantiateFlags::DEVICE_LAUNCH`]
    /// enables [`Self::device_launch_graph`] after upload (host
    /// [`Self::launch_graph`] stays legal). Mem alloc/free, events, child
    /// graphs, conditionals, and host nodes are Invalid for device-launch.
    /// Instantiating an exec id is a no-op when `flags` adds no new bits.
    pub fn instantiate_graph_with_flags(
        &mut self,
        graph: GraphId,
        flags: u32,
    ) -> Result<GraphId, SimError> {
        let mut params = GraphInstantiateParams {
            flags,
            ..GraphInstantiateParams::default()
        };
        self.instantiate_graph_with_params(graph, &mut params)
    }

    /// `cudaGraphInstantiateWithParams`. Host-synchronous. Capture cannot include it.
    ///
    /// Fills [`GraphInstantiateParams::result`] and
    /// [`GraphInstantiateParams::err_node`] even when this returns `Err`.
    /// Success is [`GraphInstantiateResult::Success`] with `err_node = None`.
    /// [`GraphInstantiateParams::flags`] are the instantiate flags (same as
    /// [`Self::instantiate_graph_with_flags`]).
    pub fn instantiate_graph_with_params(
        &mut self,
        graph: GraphId,
        params: &mut GraphInstantiateParams,
    ) -> Result<GraphId, SimError> {
        let flags = params.flags;
        let exec = self.instantiate_graph_inner(graph, flags, Some(params))?;
        if flags & GraphInstantiateFlags::UPLOAD != 0 {
            self.upload_graph(exec)?;
        }
        Ok(exec)
    }

    /// `cudaGraphExecGetFlags` on an instantiated exec (or a definition's primary).
    ///
    /// Capture is allowed. Uninstantiated graphs are Invalid.
    pub fn graph_exec_get_flags(&self, exec: GraphId) -> Result<u32, SimError> {
        let exec = self.as_exec(exec)?;
        let g = self.graphs.get(&exec).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        Ok(g.instantiate_flags)
    }

    fn check_instantiate_flags(flags: u32) -> Result<(), SimError> {
        const KNOWN: u32 = GraphInstantiateFlags::AUTO_FREE_ON_LAUNCH
            | GraphInstantiateFlags::UPLOAD
            | GraphInstantiateFlags::DEVICE_LAUNCH
            | GraphInstantiateFlags::USE_NODE_PRIORITY;
        if flags & !KNOWN != 0 {
            return Err(SimError::Invalid {
                why: "instantiate flags",
            });
        }
        Ok(())
    }

    fn instantiate_report(
        out: Option<&mut GraphInstantiateParams>,
        result: GraphInstantiateResult,
        err_node: Option<usize>,
        why: &'static str,
    ) -> Result<GraphId, SimError> {
        if let Some(p) = out {
            p.result = result;
            p.err_node = err_node;
        }
        Err(SimError::Invalid { why })
    }

    fn instantiate_graph_inner(
        &mut self,
        graph: GraphId,
        flags: u32,
        mut out: Option<&mut GraphInstantiateParams>,
    ) -> Result<GraphId, SimError> {
        if let Err(e) = self.fail_if_capturing("cannot capture graph instantiate") {
            if let Some(p) = out.as_mut() {
                p.result = GraphInstantiateResult::Error;
                p.err_node = None;
            }
            return Err(e);
        }
        if let Err(e) = Self::check_instantiate_flags(flags) {
            if let Some(p) = out.as_mut() {
                p.result = GraphInstantiateResult::Error;
                p.err_node = None;
            }
            return Err(e);
        }
        let auto_free = flags & GraphInstantiateFlags::AUTO_FREE_ON_LAUNCH != 0;
        let device_launch = flags & GraphInstantiateFlags::DEVICE_LAUNCH != 0;
        let snapshot = match self.instantiate_snapshot(graph, device_launch) {
            Ok(s) => s,
            Err(e) => {
                if let Some(p) = out.as_mut() {
                    p.result = GraphInstantiateResult::Error;
                    p.err_node = None;
                }
                return Err(e);
            }
        };
        if snapshot.already {
            if flags & !snapshot.current_flags != 0 {
                return Self::instantiate_report(
                    out,
                    GraphInstantiateResult::Error,
                    None,
                    "graph instantiate flags",
                );
            }
            if let Some(p) = out.as_mut() {
                p.result = GraphInstantiateResult::Success;
                p.err_node = None;
            }
            return Ok(graph);
        }
        if auto_free && snapshot.has_free {
            return Self::instantiate_report(
                out,
                GraphInstantiateResult::InvalidStructure,
                snapshot.free_node,
                "auto free with mem free nodes",
            );
        }
        if let Some(node) = snapshot.device_launch_err {
            return Self::instantiate_report(
                out,
                GraphInstantiateResult::NodeOperationNotSupported,
                Some(node),
                "device launch instantiate flag",
            );
        }
        if snapshot.primary.is_some() && snapshot.has_mem {
            return Self::instantiate_report(
                out,
                GraphInstantiateResult::InvalidStructure,
                snapshot.mem_node,
                "graph mem exec",
            );
        }
        if let Some(node) = snapshot.multi_dev {
            return Self::instantiate_report(
                out,
                GraphInstantiateResult::MultipleDevicesNotSupported,
                Some(node),
                "graph multiple devices",
            );
        }
        let ns = match self.profile.gpu(snapshot.device) {
            Ok(g) => g.graph_instantiate_ns.max(1),
            Err(e) => {
                if let Some(p) = out.as_mut() {
                    p.result = GraphInstantiateResult::Error;
                    p.err_node = None;
                }
                return Err(e);
            }
        };
        self.clock = self.clock.saturating_add(ns);
        for body in snapshot.bodies {
            let _exec = self.instantiate_graph_inner(body, 0, None)?;
        }
        let exec_id = GraphId(self.next_graph);
        self.next_graph = self.next_graph.saturating_add(1);
        let mut snap = Vec::with_capacity(snapshot.steps.len());
        for mut node in snapshot.steps {
            node.kind = self.remap_kind_to_exec(node.kind)?;
            snap.push(node);
        }
        let _prev = self.graphs.insert(
            exec_id,
            Graph {
                steps: snap.clone(),
                exec: Some(snap),
                origin: snapshot.origin,
                instantiated: true,
                uploaded: false,
                auto_free_on_launch: auto_free,
                instantiate_flags: flags,
                device_launch_tail: None,
                primary_exec: None,
                src: Some(graph),
            },
        );
        let def = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        if def.primary_exec.is_none() {
            def.primary_exec = Some(exec_id);
        }
        if let Some(p) = out.as_mut() {
            p.result = GraphInstantiateResult::Success;
            p.err_node = None;
        }
        Ok(exec_id)
    }

    fn instantiate_snapshot(
        &self,
        graph: GraphId,
        device_launch: bool,
    ) -> Result<InstantiateSnap, SimError> {
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let origin_dev = g.origin.0;
        let device = g.steps.first().map(|s| s.device).unwrap_or(origin_dev);
        let free_node = g
            .steps
            .iter()
            .position(|s| matches!(&s.kind, Kind::Free { .. }));
        let mem_node = g
            .steps
            .iter()
            .position(|s| matches!(&s.kind, Kind::Alloc { .. } | Kind::Free { .. }));
        let has_free = free_node.is_some();
        let has_mem =
            mem_node.is_some() || self.graph_allocs.get(&graph).is_some_and(|v| !v.is_empty());
        let device_launch_err = if device_launch {
            g.steps.iter().position(|s| {
                device_launch_refused(&s.kind)
                    || s.programmatic_event.is_some()
                    || s.launch_completion.is_some()
            })
        } else {
            None
        };
        let multi_dev = g.steps.iter().position(|s| s.device != origin_dev);
        Ok(InstantiateSnap {
            device,
            already: g.instantiated,
            current_flags: g.instantiate_flags,
            has_free,
            device_launch_err,
            has_mem,
            mem_node,
            free_node,
            multi_dev,
            primary: g.primary_exec,
            origin: g.origin,
            bodies: g
                .steps
                .iter()
                .flat_map(|s| cond_body_graphs(&s.kind))
                .collect(),
            steps: g.steps.clone(),
        })
    }

    fn ensure_exec(&mut self, graph: GraphId) -> Result<GraphId, SimError> {
        let (instantiated, primary) = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            (g.instantiated, g.primary_exec)
        };
        if instantiated {
            return Ok(graph);
        }
        if let Some(p) = primary {
            return Ok(p);
        }
        self.instantiate_graph(graph)
    }

    /// `cudaGraphUpload`. Host-synchronous. Capture cannot include it.
    ///
    /// The exec must already be instantiated. Already-uploaded ids are a no-op.
    /// The first [`Self::launch_graph`] calls this when needed. [`Self::update_graph`]
    /// clears the flag so the next launch uploads again.
    pub fn upload_graph(&mut self, graph: GraphId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture graph upload")?;
        let exec = self.as_exec(graph)?;
        let (device, already) = {
            let g = self.graphs.get(&exec).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let device = g.steps.first().map(|s| s.device).unwrap_or(DeviceId(0));
            (device, g.uploaded)
        };
        if already {
            return Ok(());
        }
        let ns = self.profile.gpu(device)?.graph_upload_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        self.graphs
            .get_mut(&exec)
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
        let mut info = GraphExecUpdateResultInfo::default();
        self.update_graph_with_info(exec, src, &mut info)
    }

    /// `cudaGraphExecUpdate` with [`GraphExecUpdateResultInfo`].
    ///
    /// Fills `info` even when this returns `Err`. Success is
    /// [`GraphExecUpdateResult::Success`] with both node fields `None`.
    /// [`Self::update_graph`] keeps the same `why` strings.
    pub fn update_graph_with_info(
        &mut self,
        exec: GraphId,
        src: GraphId,
        info: &mut GraphExecUpdateResultInfo,
    ) -> Result<(), SimError> {
        *info = GraphExecUpdateResultInfo::default();
        if let Err(e) = self.fail_if_capturing("cannot capture graph update") {
            info.result = GraphExecUpdateResult::Error;
            return Err(e);
        }
        if exec == src {
            return update_report(
                info,
                GraphExecUpdateResult::Error,
                None,
                None,
                "graph update same id",
            );
        }
        let exec = match self.as_exec(exec) {
            Ok(id) => id,
            Err(e) => {
                info.result = GraphExecUpdateResult::Error;
                return Err(e);
            }
        };
        if exec == src {
            return update_report(
                info,
                GraphExecUpdateResult::Error,
                None,
                None,
                "graph update same id",
            );
        }
        let (exec_steps, src_steps, device, exec_flags) = match self.update_graph_pair(exec, src) {
            Ok(pair) => pair,
            Err(e) => {
                info.result = GraphExecUpdateResult::Error;
                return Err(e);
            }
        };
        if exec_flags & GraphInstantiateFlags::DEVICE_LAUNCH != 0 {
            return update_report(
                info,
                GraphExecUpdateResult::NotSupported,
                None,
                None,
                "device launch graph update",
            );
        }
        let exec_norm = self.steps_def_ids(exec_steps);
        let src_norm = self.steps_def_ids(src_steps.clone());
        if let Some(diff) = graph_topology_diff(&exec_norm, &src_norm) {
            return update_report(
                info,
                diff.result,
                diff.error_node,
                diff.error_from_node,
                "graph update topology",
            );
        }
        if self.graph_has_mem_nodes(exec) || self.graph_has_mem_nodes(src) {
            let node = first_mem_node(&src_norm).or_else(|| first_mem_node(&exec_norm));
            return update_report(
                info,
                GraphExecUpdateResult::NotSupported,
                node,
                None,
                "cannot update graph mem nodes",
            );
        }
        let ns = self.profile.gpu(device)?.graph_update_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        let exec = self.graphs.get_mut(&exec).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        exec.exec = Some(src_steps);
        exec.uploaded = false;
        info.result = GraphExecUpdateResult::Success;
        Ok(())
    }

    fn update_graph_pair(
        &self,
        exec: GraphId,
        src: GraphId,
    ) -> Result<(Vec<GraphStep>, Vec<GraphStep>, DeviceId, u32), SimError> {
        let e = self.graphs.get(&exec).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let s = self.graphs.get(&src).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let device = e
            .steps
            .first()
            .map(|step| step.device)
            .unwrap_or(DeviceId(0));
        Ok((
            e.view().to_vec(),
            s.steps.clone(),
            device,
            e.instantiate_flags,
        ))
    }

    /// `cudaGraphExecKernelNodeSetParams` on an instantiated exec.
    ///
    /// Node `node` must already be a kernel. [`KernelNodeParams::cooperative`]
    /// must match the existing node (cooperative vs `cudaLaunchKernel` is
    /// topology). Pointers and [`KernelKind`] may change. Pays
    /// `graph_set_params_ns`. Clears the upload flag unless the node is
    /// device-updatable (`cudaLaunchAttributeDeviceUpdatableKernelNode`), so a
    /// later [`Self::device_launch_graph`] needs no host re-upload. Capture
    /// cannot include it. Graphs with mem alloc/free nodes are legal (unlike
    /// [`Self::update_graph`]).
    pub fn graph_exec_kernel_set_params(
        &mut self,
        exec: GraphId,
        node: usize,
        params: &KernelNodeParams,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel set params")?;
        let exec = self.as_exec(exec)?;
        let (device, cooperative, device_updatable) = {
            let g = self.graphs.get(&exec).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let step = live_ok(g.view().get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
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
            (step.device, *cooperative, step.device_updatable)
        };
        let reads = self.resolve_bufs(&params.reads)?;
        let writes = self.resolve_bufs(&params.writes)?;
        let ns = self.profile.gpu(device)?.graph_set_params_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        let g = self.graphs.get_mut(&exec).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.exec_mut()?.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.kind = Kind::Kernel {
            kind: params.kind.clone(),
            reads,
            writes,
            cooperative,
        };
        if !device_updatable {
            g.uploaded = false;
        }
        Ok(())
    }

    /// `cudaGraphKernelNodeSetParams` on the graph definition.
    ///
    /// After [`Self::instantiate_graph`], this does not retarget the exec
    /// snapshot; use [`Self::graph_exec_kernel_set_params`]. Cooperative flag
    /// must match (topology). Capture cannot include it. Host-sync 1 ns.
    pub fn graph_kernel_set_params(
        &mut self,
        graph: GraphId,
        node: usize,
        params: &KernelNodeParams,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set params")?;
        let (device, cooperative) = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let step = live_ok(g.steps.get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
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
            (step.device, *cooperative)
        };
        let reads = self.resolve_bufs(&params.reads)?;
        let writes = self.resolve_bufs(&params.writes)?;
        let _gpu = self.profile.gpu(device)?;
        self.clock = self.clock.saturating_add(1);
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.kind = Kind::Kernel {
            kind: params.kind.clone(),
            reads,
            writes,
            cooperative,
        };
        Ok(())
    }

    /// `cudaGraphMemcpyNodeSetParams` on the graph definition.
    ///
    /// After instantiate this does not retarget the exec; use
    /// [`Self::graph_exec_memcpy_set_params`]. Pageable copies stay illegal.
    /// Capture cannot include it. Host-sync 1 ns. [`Self::graph_memcpy_set_params_1d`]
    /// is `cudaGraphMemcpyNodeSetParams1D` (packed 1D, including converting a
    /// 2D/3D node).
    pub fn graph_memcpy_set_params(
        &mut self,
        graph: GraphId,
        node: usize,
        op: &MemcpyOp,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture memcpy node set params")?;
        if op.src.is_pageable() || op.dst.is_pageable() {
            return Err(SimError::Invalid {
                why: "cannot add pageable memcpy",
            });
        }
        let device = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let step = live_ok(g.steps.get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            if !matches!(step.kind, Kind::Memcpy(_)) {
                return Err(SimError::Invalid {
                    why: "not a memcpy node",
                });
            }
            step.device
        };
        self.memcpy_precheck(op)?;
        let _gpu = self.profile.gpu(device)?;
        self.clock = self.clock.saturating_add(1);
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.kind = Kind::Memcpy(op.clone());
        Ok(())
    }

    /// `cudaGraphMemcpyNodeSetParams1D` on the graph definition.
    ///
    /// Packs a 1D [`MemcpyOp`] ([`MemcpyOp::packed_1d`]). A 2D/3D node may
    /// become 1D. After instantiate this does not retarget the exec; use
    /// [`Self::graph_exec_memcpy_set_params_1d`]. Pageable copies stay illegal.
    /// Capture cannot include it. Host-sync 1 ns.
    pub fn graph_memcpy_set_params_1d(
        &mut self,
        graph: GraphId,
        node: usize,
        src: Place,
        dst: Place,
        alloc: AllocId,
        bytes: u64,
    ) -> Result<(), SimError> {
        self.graph_memcpy_set_params(graph, node, &MemcpyOp::packed_1d(src, dst, alloc, bytes))
    }

    /// `cudaGraphMemsetNodeSetParams` on the graph definition.
    ///
    /// After instantiate this does not retarget the exec; use
    /// [`Self::graph_exec_memset_set_params`]. Zero-byte fills stay illegal.
    /// Capture cannot include it. Host-sync 1 ns. [`KernelBuf`] converts to a
    /// packed 1D [`MemsetOp`].
    pub fn graph_memset_set_params(
        &mut self,
        graph: GraphId,
        node: usize,
        op: impl Into<MemsetOp>,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture memset node set params")?;
        let device = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let step = live_ok(g.steps.get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            if !matches!(step.kind, Kind::Memset(_)) {
                return Err(SimError::Invalid {
                    why: "not a memset node",
                });
            }
            step.device
        };
        let op = self.resolve_memset_op(op.into())?;
        let _gpu = self.profile.gpu(device)?;
        self.clock = self.clock.saturating_add(1);
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.kind = Kind::Memset(op);
        Ok(())
    }

    /// `cudaGraphHostNodeSetParams` on the graph definition.
    ///
    /// After instantiate this does not retarget the exec; use
    /// [`Self::graph_exec_host_set_params`]. [`HostNodeParams::fn_id`] /
    /// [`HostNodeParams::user_data`] are parameters. Capture cannot include it.
    /// Host-sync 1 ns.
    pub fn graph_host_set_params(
        &mut self,
        graph: GraphId,
        node: usize,
        params: HostNodeParams,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture host node set params")?;
        let device = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let step = live_ok(g.steps.get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            if !matches!(step.kind, Kind::HostFunc { .. }) {
                return Err(SimError::Invalid {
                    why: "not a host node",
                });
            }
            step.device
        };
        let _gpu = self.profile.gpu(device)?;
        self.clock = self.clock.saturating_add(1);
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.kind = Kind::HostFunc {
            fn_id: params.fn_id,
            user_data: params.user_data,
        };
        Ok(())
    }

    /// `cudaGraphMemFreeNodeSetParams` on the graph definition.
    ///
    /// After [`Self::instantiate_graph`], this does not retarget the exec
    /// snapshot; use [`Self::graph_exec_free_set_params`]. Capture cannot
    /// include it. Host-sync 1 ns. The node must already be a mem free node.
    /// [`Sim::graph_allocs`] stays the alloc-node ids.
    pub fn graph_free_set_params(
        &mut self,
        graph: GraphId,
        node: usize,
        id: AllocId,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture mem free node set params")?;
        let _a = self.alloc_ref(id)?;
        let device = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let step = live_ok(g.steps.get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            if !matches!(step.kind, Kind::Free { .. }) {
                return Err(SimError::Invalid {
                    why: "not a mem free node",
                });
            }
            step.device
        };
        let _gpu = self.profile.gpu(device)?;
        self.clock = self.clock.saturating_add(1);
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.kind = Kind::Free { id };
        Ok(())
    }

    /// `cudaGraphEventRecordNodeSetEvent` on the graph definition.
    ///
    /// After [`Self::instantiate_graph`], this does not retarget the exec
    /// snapshot; use [`Self::graph_exec_event_record_set_event`]. The External
    /// flag stays (topology). Capture cannot include it. Host-sync 1 ns.
    pub fn graph_event_record_set_event(
        &mut self,
        graph: GraphId,
        node: usize,
        event: EventId,
    ) -> Result<(), SimError> {
        self.graph_def_event_set(graph, node, event, EventSetKind::Record)
    }

    /// `cudaGraphEventWaitNodeSetEvent` on the graph definition.
    ///
    /// After instantiate this does not retarget the exec; use
    /// [`Self::graph_exec_event_wait_set_event`]. The External flag stays
    /// (topology). Capture cannot include it. Host-sync 1 ns.
    pub fn graph_event_wait_set_event(
        &mut self,
        graph: GraphId,
        node: usize,
        event: EventId,
    ) -> Result<(), SimError> {
        self.graph_def_event_set(graph, node, event, EventSetKind::Wait)
    }

    fn graph_def_event_set(
        &mut self,
        graph: GraphId,
        node: usize,
        event: EventId,
        kind: EventSetKind,
    ) -> Result<(), SimError> {
        let capture = match kind {
            EventSetKind::Record => "cannot capture event record set event",
            EventSetKind::Wait => "cannot capture event wait set event",
        };
        self.fail_if_capturing(capture)?;
        if !self.events.contains_key(&event) {
            return Err(SimError::UnknownEvent { event: event.0 });
        }
        let (device, external) = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let step = live_ok(g.steps.get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            let external = match (kind, &step.kind) {
                (EventSetKind::Record, Kind::EventRecord { external, .. }) => *external,
                (EventSetKind::Wait, Kind::EventWait { external, .. }) => *external,
                (EventSetKind::Record, _) => {
                    return Err(SimError::Invalid {
                        why: "not an event record node",
                    });
                }
                (EventSetKind::Wait, _) => {
                    return Err(SimError::Invalid {
                        why: "not an event wait node",
                    });
                }
            };
            (step.device, external)
        };
        let _gpu = self.profile.gpu(device)?;
        self.clock = self.clock.saturating_add(1);
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.kind = match kind {
            EventSetKind::Record => Kind::EventRecord { event, external },
            EventSetKind::Wait => Kind::EventWait { event, external },
        };
        Ok(())
    }

    /// `cudaGraphChildGraphNodeSetParams` on the graph definition.
    ///
    /// After instantiate this does not retarget the exec; use
    /// [`Self::graph_exec_child_set_params`]. `child` must already be
    /// instantiated, on the same GPU. Nested topology may change (unlike
    /// ExecSetParams). Capture cannot include it. Host-sync 1 ns.
    pub fn graph_child_set_params(
        &mut self,
        graph: GraphId,
        node: usize,
        child: GraphId,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture child graph node set params")?;
        if child == graph {
            return Err(SimError::Invalid {
                why: "graph child is self",
            });
        }
        let device = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let step = live_ok(g.steps.get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            if !matches!(step.kind, Kind::ChildGraph { .. }) {
                return Err(SimError::Invalid {
                    why: "not a child graph node",
                });
            }
            step.device
        };
        let (ready, origin) = {
            let c = self.graphs.get(&child).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            (c.ready(), c.origin.0)
        };
        if !ready {
            return Err(SimError::Invalid {
                why: "child graph not instantiated",
            });
        }
        if origin != device {
            return Err(SimError::Invalid {
                why: "graph child gpu mismatch",
            });
        }
        if self.graph_tree_contains(child, graph)? {
            return Err(SimError::Invalid {
                why: "cyclic child graph",
            });
        }
        let _gpu = self.profile.gpu(device)?;
        self.clock = self.clock.saturating_add(1);
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.kind = Kind::ChildGraph { graph: child };
        Ok(())
    }

    /// `cudaGraphNodeSetParams` on the graph definition.
    ///
    /// Dispatches to the typed SetParams. [`GraphNodeParams::Alloc`] is Invalid
    /// (would resize HBM). [`GraphNodeParams::Empty`] has no params. Event
    /// External flags are not rewritten (topology). After instantiate this
    /// does not retarget the exec; use [`Self::graph_exec_node_set_params`].
    /// Capture cannot include it.
    pub fn graph_node_set_params(
        &mut self,
        graph: GraphId,
        node: usize,
        params: GraphNodeParams,
    ) -> Result<(), SimError> {
        self.set_node_params(graph, node, params, false)
    }

    /// `cudaGraphExecNodeSetParams` on an instantiated exec.
    ///
    /// Dispatches to the typed ExecSetParams. [`GraphNodeParams::Alloc`] is
    /// Invalid. [`GraphNodeParams::Empty`] has no params. Capture cannot
    /// include it.
    pub fn graph_exec_node_set_params(
        &mut self,
        exec: GraphId,
        node: usize,
        params: GraphNodeParams,
    ) -> Result<(), SimError> {
        self.set_node_params(exec, node, params, true)
    }

    fn set_node_params(
        &mut self,
        graph: GraphId,
        node: usize,
        params: GraphNodeParams,
        exec: bool,
    ) -> Result<(), SimError> {
        match params {
            GraphNodeParams::Kernel(p) => {
                if exec {
                    self.graph_exec_kernel_set_params(graph, node, &p)
                } else {
                    self.graph_kernel_set_params(graph, node, &p)
                }
            }
            GraphNodeParams::Memcpy(op) => {
                if exec {
                    self.graph_exec_memcpy_set_params(graph, node, &op)
                } else {
                    self.graph_memcpy_set_params(graph, node, &op)
                }
            }
            GraphNodeParams::Memset(op) => {
                if exec {
                    self.graph_exec_memset_set_params(graph, node, op)
                } else {
                    self.graph_memset_set_params(graph, node, op)
                }
            }
            GraphNodeParams::Host(p) => {
                if exec {
                    self.graph_exec_host_set_params(graph, node, p)
                } else {
                    self.graph_host_set_params(graph, node, p)
                }
            }
            GraphNodeParams::Empty => Err(SimError::Invalid {
                why: "empty node has no params",
            }),
            GraphNodeParams::EventRecord { event, .. } => {
                if exec {
                    self.graph_exec_event_record_set_event(graph, node, event)
                } else {
                    self.graph_event_record_set_event(graph, node, event)
                }
            }
            GraphNodeParams::EventWait { event, .. } => {
                if exec {
                    self.graph_exec_event_wait_set_event(graph, node, event)
                } else {
                    self.graph_event_wait_set_event(graph, node, event)
                }
            }
            GraphNodeParams::ChildGraph(child) => {
                if exec {
                    self.graph_exec_child_set_params(graph, node, child)
                } else {
                    self.graph_child_set_params(graph, node, child)
                }
            }
            GraphNodeParams::Alloc { .. } => Err(SimError::Invalid {
                why: "cannot set mem alloc node params",
            }),
            GraphNodeParams::Free(id) => {
                if exec {
                    self.graph_exec_free_set_params(graph, node, id)
                } else {
                    self.graph_free_set_params(graph, node, id)
                }
            }
            GraphNodeParams::BatchMemOp(ops) => {
                if exec {
                    self.graph_exec_batch_mem_ops_set_params(graph, node, &ops)
                } else {
                    self.graph_batch_mem_ops_set_params(graph, node, &ops)
                }
            }
        }
    }

    /// `cudaGraphNodeGetParams` on the graph definition.
    ///
    /// Query; legal during capture. Typed GetParams stay. IF/WHILE/SWITCH stay
    /// [`Self::graph_if_nodes`] / `graph_while_nodes` / `graph_switch_nodes`.
    /// [`GraphNodeParams::Alloc`] is bytes only; the pointer is
    /// [`Self::graph_alloc_get_params`]. Empty returns [`GraphNodeParams::Empty`].
    pub fn graph_node_get_params(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<GraphNodeParams, SimError> {
        node_params_of(&self.graph_def_step(graph, node)?.kind)
    }

    /// Exec-snapshot [`Self::graph_node_get_params`].
    ///
    /// Uninstantiated graphs are Invalid. After instantiate this is the
    /// launched node; [`Self::graph_node_get_params`] stays on the definition.
    pub fn graph_exec_node_get_params(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<GraphNodeParams, SimError> {
        node_params_of(&self.graph_exec_step(exec, node)?.kind)
    }

    /// `cudaGraphBatchMemOpNodeSetParams` on the graph definition.
    ///
    /// After instantiate this does not retarget the exec; use
    /// [`Self::graph_exec_batch_mem_op_set_params`]. A wait-value / write-value
    /// node keeps wait vs write, `bits32`, and compare (topology). A
    /// [`crate::GpuOp::BatchMem`] node treats the item list as parameters
    /// (length may change). Capture cannot include it. Host-sync 1 ns.
    pub fn graph_batch_mem_op_set_params(
        &mut self,
        graph: GraphId,
        node: usize,
        op: BatchMemOp,
    ) -> Result<(), SimError> {
        self.set_batch_mem_ops(graph, node, &[op], false)
    }

    /// Replace the item list of a [`crate::GpuOp::BatchMem`] graph node.
    ///
    /// Empty is Invalid. Wait vs write mix and length are parameters. After
    /// instantiate this does not retarget the exec. Capture cannot include it.
    /// Host-sync 1 ns.
    pub fn graph_batch_mem_ops_set_params(
        &mut self,
        graph: GraphId,
        node: usize,
        ops: &[BatchMemOp],
    ) -> Result<(), SimError> {
        self.set_batch_mem_ops(graph, node, ops, false)
    }

    /// `cudaGraphExecMemcpyNodeSetParams` on an instantiated exec.
    ///
    /// Node `node` must already be a memcpy. [`MemcpyOp`] src/dst/alloc/bytes
    /// may change. Pageable copies stay illegal. Pays `graph_set_params_ns`
    /// and clears the upload flag. Capture cannot include it. Graphs with
    /// mem alloc/free nodes are legal (unlike [`Self::update_graph`]).
    /// [`Self::graph_exec_memcpy_set_params_1d`] is
    /// `cudaGraphExecMemcpyNodeSetParams1D`.
    pub fn graph_exec_memcpy_set_params(
        &mut self,
        exec: GraphId,
        node: usize,
        op: &MemcpyOp,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture memcpy set params")?;
        if op.src.is_pageable() || op.dst.is_pageable() {
            return Err(SimError::Invalid {
                why: "cannot add pageable memcpy",
            });
        }
        let exec = self.as_exec(exec)?;
        let device = {
            let g = self.graphs.get(&exec).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let step = live_ok(g.view().get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            if !matches!(step.kind, Kind::Memcpy(_)) {
                return Err(SimError::Invalid {
                    why: "not a memcpy node",
                });
            }
            step.device
        };
        self.memcpy_precheck(op)?;
        let ns = self.profile.gpu(device)?.graph_set_params_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        let g = self.graphs.get_mut(&exec).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.exec_mut()?.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.kind = Kind::Memcpy(op.clone());
        g.uploaded = false;
        Ok(())
    }

    /// `cudaGraphExecMemcpyNodeSetParams1D` on an instantiated exec.
    ///
    /// Packs a 1D [`MemcpyOp`]. A 2D/3D node may become 1D. Pageable copies
    /// stay illegal. Pays `graph_set_params_ns`. Capture cannot include it.
    pub fn graph_exec_memcpy_set_params_1d(
        &mut self,
        exec: GraphId,
        node: usize,
        src: Place,
        dst: Place,
        alloc: AllocId,
        bytes: u64,
    ) -> Result<(), SimError> {
        self.graph_exec_memcpy_set_params(exec, node, &MemcpyOp::packed_1d(src, dst, alloc, bytes))
    }

    /// `cudaGraphExecMemsetNodeSetParams` on an instantiated exec.
    ///
    /// Node `node` must already be a memset. Dest/span/pitch may change.
    /// Zero-byte fills stay illegal. Pays `graph_set_params_ns` and clears the
    /// upload flag. Capture cannot include it. Graphs with mem alloc/free nodes
    /// are legal (unlike [`Self::update_graph`]). [`KernelBuf`] converts to a
    /// packed 1D [`MemsetOp`].
    pub fn graph_exec_memset_set_params(
        &mut self,
        exec: GraphId,
        node: usize,
        op: impl Into<MemsetOp>,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture memset set params")?;
        let exec = self.as_exec(exec)?;
        let device = {
            let g = self.graphs.get(&exec).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let step = live_ok(g.view().get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            if !matches!(step.kind, Kind::Memset(_)) {
                return Err(SimError::Invalid {
                    why: "not a memset node",
                });
            }
            step.device
        };
        let op = self.resolve_memset_op(op.into())?;
        let ns = self.profile.gpu(device)?.graph_set_params_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        let g = self.graphs.get_mut(&exec).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.exec_mut()?.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.kind = Kind::Memset(op);
        g.uploaded = false;
        Ok(())
    }

    /// `cudaGraphExecHostNodeSetParams` on an instantiated exec.
    ///
    /// Node `node` must already be a host node. [`HostNodeParams::fn_id`] /
    /// [`HostNodeParams::user_data`] may change. Pays `graph_set_params_ns` and
    /// clears the upload flag. Capture cannot include it. Graphs with mem
    /// alloc/free nodes are legal (unlike [`Self::update_graph`]).
    pub fn graph_exec_host_set_params(
        &mut self,
        exec: GraphId,
        node: usize,
        params: HostNodeParams,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture host set params")?;
        let exec = self.as_exec(exec)?;
        let device = {
            let g = self.graphs.get(&exec).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let step = live_ok(g.view().get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            if !matches!(step.kind, Kind::HostFunc { .. }) {
                return Err(SimError::Invalid {
                    why: "not a host node",
                });
            }
            step.device
        };
        let ns = self.profile.gpu(device)?.graph_set_params_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        let g = self.graphs.get_mut(&exec).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.exec_mut()?.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.kind = Kind::HostFunc {
            fn_id: params.fn_id,
            user_data: params.user_data,
        };
        g.uploaded = false;
        Ok(())
    }

    /// `cudaGraphExecBatchMemOpNodeSetParams` on an instantiated exec.
    ///
    /// A wait-value / write-value node may change id / offset / value; wait vs
    /// write, `bits32`, and compare stay. A [`crate::GpuOp::BatchMem`] node
    /// replaces the item list (length may change). Pays `graph_set_params_ns`
    /// and clears the upload flag. Capture cannot include it.
    pub fn graph_exec_batch_mem_op_set_params(
        &mut self,
        exec: GraphId,
        node: usize,
        op: BatchMemOp,
    ) -> Result<(), SimError> {
        self.set_batch_mem_ops(exec, node, &[op], true)
    }

    /// Exec-side item-list SetParams for a [`crate::GpuOp::BatchMem`] node.
    pub fn graph_exec_batch_mem_ops_set_params(
        &mut self,
        exec: GraphId,
        node: usize,
        ops: &[BatchMemOp],
    ) -> Result<(), SimError> {
        self.set_batch_mem_ops(exec, node, ops, true)
    }

    fn set_batch_mem_ops(
        &mut self,
        graph: GraphId,
        node: usize,
        ops: &[BatchMemOp],
        exec: bool,
    ) -> Result<(), SimError> {
        self.fail_if_capturing(if exec {
            "cannot capture batch mem op set params"
        } else {
            "cannot capture batch mem op node set params"
        })?;
        self.check_batch_mem_ops(ops)?;
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let (device, next) = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let step = live_ok(
                if exec {
                    g.view().get(node)
                } else {
                    g.steps.get(node)
                }
                .ok_or(SimError::Invalid {
                    why: "unknown graph node",
                })?,
            )?;
            let next = batch_ops_set_params_kind(&step.kind, ops)?;
            (step.device, next)
        };
        if exec {
            let ns = self.profile.gpu(device)?.graph_set_params_ns.max(1);
            self.clock = self.clock.saturating_add(ns);
        } else {
            let _gpu = self.profile.gpu(device)?;
            self.clock = self.clock.saturating_add(1);
        }
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(
            if exec {
                g.exec_mut()?.get_mut(node)
            } else {
                g.steps.get_mut(node)
            }
            .ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?,
        )?;
        step.kind = next;
        if exec {
            g.uploaded = false;
        }
        Ok(())
    }

    /// `cudaGraphExecChildGraphNodeSetParams` on an instantiated exec.
    ///
    /// Node `node` must already be a child-graph node. `child` must already be
    /// instantiated, on the same GPU, and have matching topology with the
    /// current nested graph (device, stream, op kinds, cooperative flag, deps).
    /// Nested child-graph ids may differ (the nested graph is the
    /// parameter). Pays `graph_set_params_ns` and clears the upload flag.
    /// Capture cannot include it. Graphs with mem alloc/free nodes are legal
    /// (unlike [`Self::update_graph`], which treats child ids as topology).
    /// Nesting `exec` under itself, or a child whose tree already names `exec`,
    /// is Invalid.
    pub fn graph_exec_child_set_params(
        &mut self,
        exec: GraphId,
        node: usize,
        child: GraphId,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture child set params")?;
        if child == exec {
            return Err(SimError::Invalid {
                why: "graph child is self",
            });
        }
        let exec = self.as_exec(exec)?;
        if child == exec {
            return Err(SimError::Invalid {
                why: "graph child is self",
            });
        }
        let (device, old_child) = {
            let g = self.graphs.get(&exec).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let step = live_ok(g.view().get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            let Kind::ChildGraph { graph } = &step.kind else {
                return Err(SimError::Invalid {
                    why: "not a child graph node",
                });
            };
            (step.device, *graph)
        };
        let (child_ok, child_gpu, child_steps) = {
            let c = self.graphs.get(&child).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            (c.ready(), c.origin.0, c.steps.clone())
        };
        if !child_ok {
            return Err(SimError::Invalid {
                why: "child graph not instantiated",
            });
        }
        if child_gpu != device {
            return Err(SimError::Invalid {
                why: "graph child gpu mismatch",
            });
        }
        let old_steps = {
            let old = self.graphs.get(&old_child).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            old.steps.clone()
        };
        if !child_param_topology_eq(&old_steps, &child_steps) {
            return Err(SimError::Invalid {
                why: "child graph topology",
            });
        }
        if self.graph_tree_contains(child, exec)? {
            return Err(SimError::Invalid {
                why: "cyclic child graph",
            });
        }
        let child_exec = self.as_exec(child)?;
        if child_exec == exec {
            return Err(SimError::Invalid {
                why: "graph child is self",
            });
        }
        let ns = self.profile.gpu(device)?.graph_set_params_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        let g = self.graphs.get_mut(&exec).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.exec_mut()?.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.kind = Kind::ChildGraph { graph: child_exec };
        g.uploaded = false;
        Ok(())
    }

    /// `cudaGraphExecEventRecordNodeSetEvent` on an instantiated exec.
    ///
    /// Node `node` must already be an event-record node. The event id may
    /// change; the External flag stays (topology). Pays `graph_set_params_ns`
    /// and clears the upload flag. Capture cannot include it. Graphs with mem
    /// alloc/free nodes are legal.
    pub fn graph_exec_event_record_set_event(
        &mut self,
        exec: GraphId,
        node: usize,
        event: EventId,
    ) -> Result<(), SimError> {
        self.graph_exec_event_set(exec, node, event, EventSetKind::Record)
    }

    /// `cudaGraphExecEventWaitNodeSetEvent` on an instantiated exec.
    ///
    /// Node `node` must already be an event-wait node. The event id may change;
    /// the External flag stays (topology). Pays `graph_set_params_ns` and
    /// clears the upload flag. Capture cannot include it.
    pub fn graph_exec_event_wait_set_event(
        &mut self,
        exec: GraphId,
        node: usize,
        event: EventId,
    ) -> Result<(), SimError> {
        self.graph_exec_event_set(exec, node, event, EventSetKind::Wait)
    }

    fn graph_exec_event_set(
        &mut self,
        exec: GraphId,
        node: usize,
        event: EventId,
        kind: EventSetKind,
    ) -> Result<(), SimError> {
        let capture = match kind {
            EventSetKind::Record => "cannot capture event record set event",
            EventSetKind::Wait => "cannot capture event wait set event",
        };
        self.fail_if_capturing(capture)?;
        if !self.events.contains_key(&event) {
            return Err(SimError::UnknownEvent { event: event.0 });
        }
        let exec = self.as_exec(exec)?;
        let (device, external) = {
            let g = self.graphs.get(&exec).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let step = live_ok(g.view().get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            let external = match (kind, &step.kind) {
                (EventSetKind::Record, Kind::EventRecord { external, .. }) => *external,
                (EventSetKind::Wait, Kind::EventWait { external, .. }) => *external,
                (EventSetKind::Record, _) => {
                    return Err(SimError::Invalid {
                        why: "not an event record node",
                    });
                }
                (EventSetKind::Wait, _) => {
                    return Err(SimError::Invalid {
                        why: "not an event wait node",
                    });
                }
            };
            (step.device, external)
        };
        let ns = self.profile.gpu(device)?.graph_set_params_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        let g = self.graphs.get_mut(&exec).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.exec_mut()?.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.kind = match kind {
            EventSetKind::Record => Kind::EventRecord { event, external },
            EventSetKind::Wait => Kind::EventWait { event, external },
        };
        g.uploaded = false;
        Ok(())
    }

    /// `cudaGraphExecMemFreeNodeSetParams` on an instantiated exec.
    ///
    /// Node `node` must already be a mem free node. The freed id may change.
    /// Pays `graph_set_params_ns` and clears the upload flag. Capture cannot
    /// include it. Graphs with mem alloc/free nodes are legal (unlike
    /// [`Self::update_graph`]). [`Sim::graph_allocs`] stays the alloc-node ids.
    pub fn graph_exec_free_set_params(
        &mut self,
        exec: GraphId,
        node: usize,
        id: AllocId,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture mem free node set params")?;
        let _a = self.alloc_ref(id)?;
        let exec = self.as_exec(exec)?;
        let device = {
            let g = self.graphs.get(&exec).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let step = live_ok(g.view().get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            if !matches!(step.kind, Kind::Free { .. }) {
                return Err(SimError::Invalid {
                    why: "not a mem free node",
                });
            }
            step.device
        };
        let ns = self.profile.gpu(device)?.graph_set_params_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        let g = self.graphs.get_mut(&exec).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.exec_mut()?.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.kind = Kind::Free { id };
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
        let graph = self.resolved_graph(graph)?;
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let mut found = None;
        for (i, step) in g.view().iter().enumerate() {
            if step.destroyed {
                continue;
            }
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

    /// Unique memcpy node on `graph` plus its current [`MemcpyOp`].
    ///
    /// Zero memcpy nodes is Invalid (`not a memcpy node`). More than one is
    /// `not unique memcpy node`.
    pub fn graph_unique_memcpy(&self, graph: GraphId) -> Result<(usize, MemcpyOp), SimError> {
        match self.graph_try_unique_memcpy(graph)? {
            Some(v) => Ok(v),
            None => Err(SimError::Invalid {
                why: "not a memcpy node",
            }),
        }
    }

    /// Unique memcpy node, or `None` when the graph has no memcpy.
    ///
    /// More than one memcpy node is Invalid.
    pub fn graph_try_unique_memcpy(
        &self,
        graph: GraphId,
    ) -> Result<Option<(usize, MemcpyOp)>, SimError> {
        let graph = self.resolved_graph(graph)?;
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let mut found = None;
        for (i, step) in g.view().iter().enumerate() {
            if step.destroyed {
                continue;
            }
            let Kind::Memcpy(op) = &step.kind else {
                continue;
            };
            if found.is_some() {
                return Err(SimError::Invalid {
                    why: "not unique memcpy node",
                });
            }
            found = Some((i, op.clone()));
        }
        Ok(found)
    }

    /// Unique memset node on `graph` plus its current [`MemsetOp`].
    ///
    /// Zero memset nodes is Invalid (`not a memset node`). More than one is
    /// `not unique memset node`.
    pub fn graph_unique_memset(&self, graph: GraphId) -> Result<(usize, MemsetOp), SimError> {
        match self.graph_try_unique_memset(graph)? {
            Some(v) => Ok(v),
            None => Err(SimError::Invalid {
                why: "not a memset node",
            }),
        }
    }

    /// Unique memset node, or `None` when the graph has no memset.
    ///
    /// More than one memset node is Invalid.
    pub fn graph_try_unique_memset(
        &self,
        graph: GraphId,
    ) -> Result<Option<(usize, MemsetOp)>, SimError> {
        let graph = self.resolved_graph(graph)?;
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let mut found = None;
        for (i, step) in g.view().iter().enumerate() {
            if step.destroyed {
                continue;
            }
            let Kind::Memset(op) = &step.kind else {
                continue;
            };
            if found.is_some() {
                return Err(SimError::Invalid {
                    why: "not unique memset node",
                });
            }
            found = Some((i, *op));
        }
        Ok(found)
    }

    /// Unique host node on `graph` plus its current [`HostNodeParams`].
    ///
    /// Zero host nodes is Invalid (`not a host node`). More than one is
    /// `not unique host node`.
    pub fn graph_unique_host(&self, graph: GraphId) -> Result<(usize, HostNodeParams), SimError> {
        match self.graph_try_unique_host(graph)? {
            Some(v) => Ok(v),
            None => Err(SimError::Invalid {
                why: "not a host node",
            }),
        }
    }

    /// Unique host node, or `None` when the graph has no host callback.
    ///
    /// More than one host node is Invalid.
    pub fn graph_try_unique_host(
        &self,
        graph: GraphId,
    ) -> Result<Option<(usize, HostNodeParams)>, SimError> {
        let graph = self.resolved_graph(graph)?;
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let mut found = None;
        for (i, step) in g.view().iter().enumerate() {
            if step.destroyed {
                continue;
            }
            let Kind::HostFunc { fn_id, user_data } = &step.kind else {
                continue;
            };
            if found.is_some() {
                return Err(SimError::Invalid {
                    why: "not unique host node",
                });
            }
            found = Some((
                i,
                HostNodeParams {
                    fn_id: *fn_id,
                    user_data: *user_data,
                },
            ));
        }
        Ok(found)
    }

    fn graph_def_step(&self, graph: GraphId, node: usize) -> Result<&GraphStep, SimError> {
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        live_ok(g.steps.get(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)
    }

    fn graph_view_step(&self, graph: GraphId, node: usize) -> Result<&GraphStep, SimError> {
        let graph = self.resolved_graph(graph)?;
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        live_ok(g.view().get(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)
    }

    fn graph_exec_step(&self, exec: GraphId, node: usize) -> Result<&GraphStep, SimError> {
        let exec = self.as_exec(exec)?;
        let g = self.graphs.get(&exec).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        live_ok(g.view().get(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)
    }

    /// `cudaGraphKernelNodeGetParams` on the graph definition.
    pub fn graph_kernel_get_params(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<KernelNodeParams, SimError> {
        kernel_params_of(&self.graph_def_step(graph, node)?.kind)
    }

    /// `cudaGraphKernelNodeGetParams` of the exec snapshot.
    ///
    /// Uninstantiated graphs are Invalid. After instantiate this is the
    /// launched kernel; [`Self::graph_kernel_get_params`] stays on the
    /// definition.
    pub fn graph_exec_kernel_get_params(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<KernelNodeParams, SimError> {
        kernel_params_of(&self.graph_exec_step(exec, node)?.kind)
    }

    /// `cudaGraphMemcpyNodeGetParams` on the graph definition.
    pub fn graph_memcpy_get_params(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<MemcpyOp, SimError> {
        memcpy_params_of(&self.graph_def_step(graph, node)?.kind)
    }

    /// Exec-snapshot memcpy params. Uninstantiated graphs are Invalid.
    pub fn graph_exec_memcpy_get_params(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<MemcpyOp, SimError> {
        memcpy_params_of(&self.graph_exec_step(exec, node)?.kind)
    }

    /// `cudaGraphMemsetNodeGetParams` on the graph definition.
    pub fn graph_memset_get_params(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<MemsetOp, SimError> {
        memset_params_of(&self.graph_def_step(graph, node)?.kind)
    }

    /// Exec-snapshot memset params. Uninstantiated graphs are Invalid.
    pub fn graph_exec_memset_get_params(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<MemsetOp, SimError> {
        memset_params_of(&self.graph_exec_step(exec, node)?.kind)
    }

    /// `cudaGraphHostNodeGetParams` on the graph definition.
    pub fn graph_host_get_params(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<HostNodeParams, SimError> {
        host_params_of(&self.graph_def_step(graph, node)?.kind)
    }

    /// Exec-snapshot host params. Uninstantiated graphs are Invalid.
    pub fn graph_exec_host_get_params(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<HostNodeParams, SimError> {
        host_params_of(&self.graph_exec_step(exec, node)?.kind)
    }

    /// `cudaGraphBatchMemOpNodeGetParams` on the graph definition.
    ///
    /// Wait-value / write-value nodes are a one-item list.
    pub fn graph_batch_mem_ops_get_params(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<Vec<BatchMemOp>, SimError> {
        batch_items(&self.graph_def_step(graph, node)?.kind).ok_or(SimError::Invalid {
            why: "not a batch mem op node",
        })
    }

    /// Exec-snapshot batch-mem-op items. Uninstantiated graphs are Invalid.
    pub fn graph_exec_batch_mem_ops_get_params(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<Vec<BatchMemOp>, SimError> {
        batch_items(&self.graph_exec_step(exec, node)?.kind).ok_or(SimError::Invalid {
            why: "not a batch mem op node",
        })
    }

    /// Child-graph nodes on `graph` as `(index, nested GraphId)` in add order.
    pub fn graph_child_nodes(&self, graph: GraphId) -> Result<Vec<(usize, GraphId)>, SimError> {
        let graph = self.resolved_graph(graph)?;
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let mut out = Vec::new();
        for (i, step) in g.view().iter().enumerate() {
            if step.destroyed {
                continue;
            }
            if let Kind::ChildGraph { graph: child } = step.kind {
                out.push((i, child));
            }
        }
        Ok(out)
    }

    /// `cudaGraphChildGraphNodeGetGraph`. Query; legal during capture.
    ///
    /// Instantiated ids use the exec snapshot (same as [`Self::graph_child_nodes`]).
    pub fn graph_child_get_graph(&self, graph: GraphId, node: usize) -> Result<GraphId, SimError> {
        match &self.graph_view_step(graph, node)?.kind {
            Kind::ChildGraph { graph: child } => Ok(*child),
            _ => Err(SimError::Invalid {
                why: "not a child graph node",
            }),
        }
    }

    /// `cudaGraphEventRecordNodeGetEvent`. Query; legal during capture.
    pub fn graph_event_record_get_event(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<EventId, SimError> {
        match &self.graph_view_step(graph, node)?.kind {
            Kind::EventRecord { event, .. } => Ok(*event),
            _ => Err(SimError::Invalid {
                why: "not an event record node",
            }),
        }
    }

    /// `cudaGraphEventWaitNodeGetEvent`. Query; legal during capture.
    pub fn graph_event_wait_get_event(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<EventId, SimError> {
        match &self.graph_view_step(graph, node)?.kind {
            Kind::EventWait { event, .. } => Ok(*event),
            _ => Err(SimError::Invalid {
                why: "not an event wait node",
            }),
        }
    }

    /// `cudaGraphMemAllocNodeGetParams` of stored id and bytes.
    ///
    /// Query; legal during capture. Pool identity stays the graph-memory pool.
    pub fn graph_alloc_get_params(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<(AllocId, u64), SimError> {
        match &self.graph_view_step(graph, node)?.kind {
            Kind::Alloc { id, bytes } => Ok((*id, *bytes)),
            _ => Err(SimError::Invalid {
                why: "not a mem alloc node",
            }),
        }
    }

    /// `cudaGraphMemFreeNodeGetParams` of the stored [`AllocId`].
    ///
    /// Query; legal during capture. Instantiated ids use the exec snapshot
    /// (same as [`Self::graph_alloc_get_params`]). [`Sim::graph_allocs`] is
    /// alloc-node ids for AutoFree / destroy refund, not this free target.
    pub fn graph_free_get_params(&self, graph: GraphId, node: usize) -> Result<AllocId, SimError> {
        match &self.graph_view_step(graph, node)?.kind {
            Kind::Free { id } => Ok(*id),
            _ => Err(SimError::Invalid {
                why: "not a mem free node",
            }),
        }
    }

    /// Unique child-graph node on `graph`.
    ///
    /// Zero child nodes is Invalid (`not a child graph node`). More than one is
    /// `not unique child graph node`.
    pub fn graph_unique_child(&self, graph: GraphId) -> Result<(usize, GraphId), SimError> {
        match self.graph_try_unique_child(graph)? {
            Some(v) => Ok(v),
            None => Err(SimError::Invalid {
                why: "not a child graph node",
            }),
        }
    }

    /// Unique child-graph node, or `None` when the graph has no child.
    ///
    /// More than one child-graph node is Invalid.
    pub fn graph_try_unique_child(
        &self,
        graph: GraphId,
    ) -> Result<Option<(usize, GraphId)>, SimError> {
        let nodes = self.graph_child_nodes(graph)?;
        match nodes.len() {
            0 => Ok(None),
            1 => Ok(nodes.into_iter().next()),
            _ => Err(SimError::Invalid {
                why: "not unique child graph node",
            }),
        }
    }

    /// Unique event-record node on `graph` plus its current [`EventId`].
    pub fn graph_unique_event_record(&self, graph: GraphId) -> Result<(usize, EventId), SimError> {
        match self.graph_try_unique_event_record(graph)? {
            Some(v) => Ok(v),
            None => Err(SimError::Invalid {
                why: "not an event record node",
            }),
        }
    }

    /// Unique event-record node, or `None` when the graph has no record node.
    pub fn graph_try_unique_event_record(
        &self,
        graph: GraphId,
    ) -> Result<Option<(usize, EventId)>, SimError> {
        self.graph_try_unique_event(graph, true)
    }

    /// Unique event-wait node on `graph` plus its current [`EventId`].
    pub fn graph_unique_event_wait(&self, graph: GraphId) -> Result<(usize, EventId), SimError> {
        match self.graph_try_unique_event_wait(graph)? {
            Some(v) => Ok(v),
            None => Err(SimError::Invalid {
                why: "not an event wait node",
            }),
        }
    }

    /// Unique event-wait node, or `None` when the graph has no wait node.
    pub fn graph_try_unique_event_wait(
        &self,
        graph: GraphId,
    ) -> Result<Option<(usize, EventId)>, SimError> {
        self.graph_try_unique_event(graph, false)
    }

    fn graph_try_unique_event(
        &self,
        graph: GraphId,
        record: bool,
    ) -> Result<Option<(usize, EventId)>, SimError> {
        let graph = self.resolved_graph(graph)?;
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let mut found = None;
        for (i, step) in g.view().iter().enumerate() {
            if step.destroyed {
                continue;
            }
            let ev = if record {
                match &step.kind {
                    Kind::EventRecord { event, .. } => Some(*event),
                    _ => None,
                }
            } else {
                match &step.kind {
                    Kind::EventWait { event, .. } => Some(*event),
                    _ => None,
                }
            };
            let Some(event) = ev else {
                continue;
            };
            if found.is_some() {
                return Err(SimError::Invalid {
                    why: if record {
                        "not unique event record node"
                    } else {
                        "not unique event wait node"
                    },
                });
            }
            found = Some((i, event));
        }
        Ok(found)
    }

    /// Unique write-value node on `graph` plus its current [`BatchMemOp`].
    pub fn graph_unique_write_value(
        &self,
        graph: GraphId,
    ) -> Result<(usize, BatchMemOp), SimError> {
        match self.graph_try_unique_write_value(graph)? {
            Some(v) => Ok(v),
            None => Err(SimError::Invalid {
                why: "not a write-value node",
            }),
        }
    }

    /// Unique write-value node, or `None` when the graph has no write-value.
    pub fn graph_try_unique_write_value(
        &self,
        graph: GraphId,
    ) -> Result<Option<(usize, BatchMemOp)>, SimError> {
        self.graph_try_unique_batch_mem(graph, true)
    }

    /// Unique wait-value node on `graph` plus its current [`BatchMemOp`].
    pub fn graph_unique_wait_value(&self, graph: GraphId) -> Result<(usize, BatchMemOp), SimError> {
        match self.graph_try_unique_wait_value(graph)? {
            Some(v) => Ok(v),
            None => Err(SimError::Invalid {
                why: "not a wait-value node",
            }),
        }
    }

    /// Unique wait-value node, or `None` when the graph has no wait-value.
    pub fn graph_try_unique_wait_value(
        &self,
        graph: GraphId,
    ) -> Result<Option<(usize, BatchMemOp)>, SimError> {
        self.graph_try_unique_batch_mem(graph, false)
    }

    /// Wait/write items of a batch-mem-op node on `graph` (exec snapshot when
    /// instantiated). Wait-value / write-value nodes are a one-item list.
    pub fn graph_batch_mem_ops(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<Vec<BatchMemOp>, SimError> {
        let graph = self.resolved_graph(graph)?;
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok(g.view().get(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        batch_items(&step.kind).ok_or(SimError::Invalid {
            why: "not a batch mem op node",
        })
    }

    fn graph_try_unique_batch_mem(
        &self,
        graph: GraphId,
        write: bool,
    ) -> Result<Option<(usize, BatchMemOp)>, SimError> {
        let graph = self.resolved_graph(graph)?;
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let mut found = None;
        for (i, step) in g.view().iter().enumerate() {
            if step.destroyed {
                continue;
            }
            let hit = match (&step.kind, write) {
                (Kind::WriteValue { .. }, true) | (Kind::WaitValue { .. }, false) => {
                    batch_from_kind(&step.kind)
                }
                _ => None,
            };
            let Some(op) = hit else {
                continue;
            };
            if found.is_some() {
                return Err(SimError::Invalid {
                    why: if write {
                        "not unique write-value node"
                    } else {
                        "not unique wait-value node"
                    },
                });
            }
            found = Some((i, op));
        }
        Ok(found)
    }

    /// `cudaGraphNodeSetEnabled` on an instantiated exec.
    ///
    /// Disabled nodes are not launched; dependents treat them as already
    /// complete (wait for the disabled node's predecessors). Memory alloc/free
    /// nodes cannot be disabled. Pays `graph_set_params_ns`. Capture cannot
    /// include it. Does not clear the upload flag (topology unchanged).
    pub fn graph_node_set_enabled(
        &mut self,
        exec: GraphId,
        node: usize,
        enabled: bool,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture node set enabled")?;
        let exec = self.as_exec(exec)?;
        let (device, mem) = {
            let g = self.graphs.get(&exec).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let step = live_ok(g.view().get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            let mem = matches!(step.kind, Kind::Alloc { .. } | Kind::Free { .. });
            (step.device, mem)
        };
        if mem {
            return Err(SimError::Invalid {
                why: "cannot disable mem node",
            });
        }
        let ns = self.profile.gpu(device)?.graph_set_params_ns.max(1);
        self.clock = self.clock.saturating_add(ns);
        let g = self.graphs.get_mut(&exec).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.exec_mut()?.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.enabled = enabled;
        Ok(())
    }

    /// `cudaGraphNodeGetEnabled` on an instantiated exec.
    pub fn graph_node_get_enabled(&self, exec: GraphId, node: usize) -> Result<bool, SimError> {
        let exec = self.as_exec(exec)?;
        let g = self.graphs.get(&exec).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok(g.view().get(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        Ok(step.enabled)
    }

    /// Graph mem alloc node ids (`cudaMallocAsync` / `cudaGraphAddMemAllocNode`).
    pub fn graph_mem_allocs(&self, graph: GraphId) -> Result<Vec<AllocId>, SimError> {
        if !self.graphs.contains_key(&graph) {
            return Err(SimError::Invalid {
                why: "unknown graph",
            });
        }
        Ok(self.graph_mem_ids(graph))
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
            let steps = remap_nested_graphs(&raw, &remap)?;
            let cloned = remap.get(&src).copied().ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let steps = self.clone_mem_alloc_nodes(src, cloned, steps)?;
            built.push((
                cloned,
                Graph {
                    steps,
                    exec: None,
                    origin,
                    instantiated: false,
                    uploaded: false,
                    auto_free_on_launch: false,
                    instantiate_flags: 0,
                    device_launch_tail: None,
                    primary_exec: None,
                    src: None,
                },
                origin.0,
            ));
        }
        for (id, cloned, device) in built {
            let ns = self.profile.gpu(device)?.graph_clone_ns.max(1);
            self.clock = self.clock.saturating_add(ns);
            let _prev = self.graphs.insert(id, cloned);
        }
        for (src, cloned) in &remap {
            let _prev = self.clone_of.insert(*cloned, *src);
        }
        self.clone_conditionals(&remap)?;
        remap.get(&graph).copied().ok_or(SimError::Invalid {
            why: "unknown graph",
        })
    }

    fn clone_conditionals(&mut self, remap: &BTreeMap<GraphId, GraphId>) -> Result<(), SimError> {
        let mut cond_remap = BTreeMap::new();
        let src: Vec<(CondId, Cond)> = self
            .conds
            .iter()
            .filter(|(_, c)| remap.contains_key(&c.graph))
            .map(|(id, c)| {
                (
                    *id,
                    Cond {
                        graph: c.graph,
                        default: c.default,
                        value: c.default,
                    },
                )
            })
            .collect();
        for (old, mut cond) in src {
            let new_g = remap.get(&cond.graph).copied().ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            cond.graph = new_g;
            let id = CondId(self.next_cond);
            self.next_cond = self.next_cond.saturating_add(1);
            let _prev = cond_remap.insert(old, id);
            let _ins = self.conds.insert(id, cond);
        }
        if cond_remap.is_empty() {
            return Ok(());
        }
        for dst in remap.values() {
            let g = self.graphs.get_mut(dst).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            for step in &mut g.steps {
                match &mut step.kind {
                    Kind::If { handle, .. }
                    | Kind::While { handle, .. }
                    | Kind::Switch { handle, .. }
                    | Kind::SetConditional { handle, .. }
                    | Kind::WhileTick { handle, .. } => {
                        if let Some(h) = cond_remap.get(handle) {
                            *handle = *h;
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
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
            for child in nested_graphs(&step.kind) {
                self.collect_clone_tree(child, walk)?;
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
            g.view()
                .iter()
                .any(|s| matches!(&s.kind, Kind::Alloc { .. } | Kind::Free { .. }))
        })
    }

    /// True when `target` is `root` or nested under it via child-graph nodes.
    fn graph_tree_contains(&self, root: GraphId, target: GraphId) -> Result<bool, SimError> {
        let target_def = self.def_id(target);
        let mut seen = BTreeSet::new();
        let mut stack = vec![root];
        while let Some(g) = stack.pop() {
            if g == target || self.def_id(g) == target_def {
                return Ok(true);
            }
            if !seen.insert(g) {
                continue;
            }
            let steps = {
                let gr = self.graphs.get(&g).ok_or(SimError::Invalid {
                    why: "unknown graph",
                })?;
                gr.view().to_vec()
            };
            for step in steps {
                for child in nested_graphs(&step.kind) {
                    stack.push(child);
                }
            }
        }
        Ok(false)
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
            {
                let a = self.alloc_mut(id)?;
                a.devices.clear();
                a.live = false;
            }
            self.clear_mailbox(id);
        }
        Ok(())
    }

    /// `cudaGraphDestroy` / `cudaGraphExecDestroy`. Host-synchronous.
    ///
    /// Capture cannot include it. Later [`Self::launch_graph`] of this id is
    /// `unknown graph`. Destroying a definition returns remaining graph mem to
    /// the device graph-memory pool (`cudaGraphDestroy` of a graph with mem
    /// nodes). Unused reserved bytes stay charged until [`Self::graph_mem_trim`].
    /// Destroying an exec does not; a later [`Self::launch_graph`] of that exec
    /// is unknown, but the definition (and other execs) stay. Clones are
    /// independent.
    pub fn destroy_graph(&mut self, graph: GraphId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture graph destroy")?;
        let g = self.graphs.remove(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let device = g.origin.0;
        let _gpu = self.profile.gpu(device)?;
        if g.instantiated {
            if let Some(src) = g.src {
                let next = self
                    .graphs
                    .iter()
                    .find_map(|(id, x)| (x.src == Some(src)).then_some(*id));
                if let Some(def) = self.graphs.get_mut(&src) {
                    if def.primary_exec == Some(graph) {
                        def.primary_exec = next;
                    }
                }
            }
        } else {
            self.release_graph_allocs(graph)?;
        }
        self.release_user_objects_for_graph(graph);
        let _src = self.clone_of.remove(&graph);
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
        let id = self.insert_graph(device, stream);
        self.clock = self.clock.saturating_add(1);
        Ok(id)
    }

    /// `cudaUserObjectCreate`. Host-synchronous. Capture cannot include it.
    ///
    /// `flags` must be [`UserObjectFlags::NO_DESTRUCTOR_SYNC`]. `initial_refcount`
    /// must be non-zero. `destroy_fn` is the host callback id recorded when the
    /// last reference is released (no Rust callback). Decode identity does not
    /// create user objects.
    pub fn user_object_create(
        &mut self,
        destroy_fn: u64,
        initial_refcount: u32,
        flags: u32,
    ) -> Result<UserObjectId, SimError> {
        self.fail_if_capturing("cannot capture user object")?;
        if flags != UserObjectFlags::NO_DESTRUCTOR_SYNC {
            return Err(SimError::Invalid {
                why: "user object flags",
            });
        }
        if initial_refcount == 0 {
            return Err(SimError::Invalid {
                why: "user object initial refs",
            });
        }
        let id = UserObjectId(self.next_user_object);
        self.next_user_object = self.next_user_object.saturating_add(1);
        let _prev = self.user_objects.insert(
            id,
            UserObject {
                destroy_fn,
                caller: initial_refcount,
                graphs: BTreeMap::new(),
            },
        );
        self.clock = self.clock.saturating_add(1);
        Ok(id)
    }

    /// `cudaUserObjectRetain`. Host-synchronous. Capture cannot include it.
    pub fn user_object_retain(&mut self, object: UserObjectId, count: u32) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture user object")?;
        if count == 0 {
            return Err(SimError::Invalid {
                why: "user object count",
            });
        }
        let obj = self.user_object_mut(object)?;
        obj.caller = obj.caller.checked_add(count).ok_or(SimError::Invalid {
            why: "user object refs overflow",
        })?;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaUserObjectRelease`. Host-synchronous. Capture cannot include it.
    ///
    /// Releasing the last reference records [`Self::user_object_destructors`].
    pub fn user_object_release(
        &mut self,
        object: UserObjectId,
        count: u32,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture user object")?;
        if count == 0 {
            return Err(SimError::Invalid {
                why: "user object count",
            });
        }
        {
            let obj = self.user_object_mut(object)?;
            if count > obj.caller {
                return Err(SimError::Invalid {
                    why: "user object refs",
                });
            }
            obj.caller = obj.caller.saturating_sub(count);
        }
        self.maybe_destroy_user_object(object);
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphRetainUserObject` on a definition. Host-synchronous.
    ///
    /// Capture cannot include it. Illegal on an instantiated exec. Clone does
    /// not copy retains. [`GraphUserObjectFlags::MOVE`] transfers one caller
    /// reference (`count` ignored); otherwise the graph takes `count` extra refs.
    pub fn graph_retain_user_object(
        &mut self,
        graph: GraphId,
        object: UserObjectId,
        count: u32,
        flags: u32,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture user object")?;
        self.require_user_object_graph(graph)?;
        if flags & !GraphUserObjectFlags::MOVE != 0 {
            return Err(SimError::Invalid {
                why: "user object graph flags",
            });
        }
        let mv = flags & GraphUserObjectFlags::MOVE != 0;
        if !mv && count == 0 {
            return Err(SimError::Invalid {
                why: "user object count",
            });
        }
        let add = if mv { 1 } else { count };
        let obj = self.user_object_mut(object)?;
        if mv {
            if obj.caller == 0 {
                return Err(SimError::Invalid {
                    why: "user object refs",
                });
            }
            obj.caller = obj.caller.saturating_sub(1);
        }
        let held = obj.graphs.get(&graph).copied().unwrap_or(0);
        let next = held.checked_add(add).ok_or(SimError::Invalid {
            why: "user object refs overflow",
        })?;
        let _prev = obj.graphs.insert(graph, next);
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphReleaseUserObject` on a definition. Host-synchronous.
    ///
    /// Capture cannot include it. Illegal on an instantiated exec.
    pub fn graph_release_user_object(
        &mut self,
        graph: GraphId,
        object: UserObjectId,
        count: u32,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture user object")?;
        self.require_user_object_graph(graph)?;
        if count == 0 {
            return Err(SimError::Invalid {
                why: "user object count",
            });
        }
        self.graph_release_user_object_inner(graph, object, count)?;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// Total remaining references (caller plus graphs).
    pub fn user_object_refs(&self, object: UserObjectId) -> Result<u32, SimError> {
        let obj = self.user_object(object)?;
        Ok(obj
            .caller
            .saturating_add(obj.graphs.values().copied().fold(0, u32::saturating_add)))
    }

    /// References held by `graph` (`0` if the graph holds none).
    pub fn user_object_graph_refs(
        &self,
        graph: GraphId,
        object: UserObjectId,
    ) -> Result<u32, SimError> {
        let _g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        Ok(self
            .user_object(object)?
            .graphs
            .get(&graph)
            .copied()
            .unwrap_or(0))
    }

    /// Destroy callbacks that have run, in order (`id`, `destroy_fn`).
    #[must_use]
    pub fn user_object_destructors(&self) -> &[(UserObjectId, u64)] {
        &self.user_object_dtors
    }

    fn user_object(&self, object: UserObjectId) -> Result<&UserObject, SimError> {
        self.user_objects.get(&object).ok_or(SimError::Invalid {
            why: "unknown user object",
        })
    }

    fn user_object_mut(&mut self, object: UserObjectId) -> Result<&mut UserObject, SimError> {
        self.user_objects.get_mut(&object).ok_or(SimError::Invalid {
            why: "unknown user object",
        })
    }

    fn require_user_object_graph(&self, graph: GraphId) -> Result<(), SimError> {
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        if g.instantiated {
            return Err(SimError::Invalid {
                why: "graph instantiated",
            });
        }
        Ok(())
    }

    fn graph_release_user_object_inner(
        &mut self,
        graph: GraphId,
        object: UserObjectId,
        count: u32,
    ) -> Result<(), SimError> {
        let obj = self.user_object_mut(object)?;
        let held = obj.graphs.get(&graph).copied().unwrap_or(0);
        if count > held {
            return Err(SimError::Invalid {
                why: "user object graph refs",
            });
        }
        let next = held.saturating_sub(count);
        if next == 0 {
            let _gone = obj.graphs.remove(&graph);
        } else {
            let _prev = obj.graphs.insert(graph, next);
        }
        self.maybe_destroy_user_object(object);
        Ok(())
    }

    fn release_user_objects_for_graph(&mut self, graph: GraphId) {
        let ids: Vec<(UserObjectId, u32)> = self
            .user_objects
            .iter()
            .filter_map(|(id, obj)| obj.graphs.get(&graph).copied().map(|n| (*id, n)))
            .collect();
        for (id, count) in ids {
            drop(self.graph_release_user_object_inner(graph, id, count));
        }
    }

    fn maybe_destroy_user_object(&mut self, object: UserObjectId) {
        let Some(obj) = self.user_objects.get(&object) else {
            return;
        };
        if obj.caller != 0 || !obj.graphs.is_empty() {
            return;
        }
        if let Some(obj) = self.user_objects.remove(&object) {
            self.user_object_dtors.push((object, obj.destroy_fn));
        }
    }

    fn insert_graph(&mut self, device: DeviceId, stream: StreamId) -> GraphId {
        let id = GraphId(self.next_graph);
        self.next_graph = self.next_graph.saturating_add(1);
        let _prev = self.graphs.insert(
            id,
            Graph {
                steps: Vec::new(),
                exec: None,
                origin: (device, stream),
                instantiated: false,
                uploaded: false,
                auto_free_on_launch: false,
                instantiate_flags: 0,
                device_launch_tail: None,
                primary_exec: None,
                src: None,
            },
        );
        let _old = self.graph_allocs.insert(id, Vec::new());
        id
    }

    /// `cudaGraphAddNode` (`cudaGraphNodeParams` plus dependency indices).
    ///
    /// Typed [`Self::graph_add_kernel`] / `graph_add_memcpy` / … stay; they
    /// start with no dependencies. This call binds `deps` in the same step (all
    /// indices must already exist). IF/WHILE/SWITCH stay
    /// [`Self::graph_add_if`] / `graph_add_while` / `graph_add_switch`. Capture
    /// cannot include it. Illegal on an instantiated exec.
    /// [`GraphNodeParams::Alloc`] fills [`GraphAddNode::alloc`].
    pub fn graph_add_node(
        &mut self,
        graph: GraphId,
        deps: &[usize],
        params: GraphNodeParams,
    ) -> Result<GraphAddNode, SimError> {
        let n = self.graph_len(graph)?;
        for &from in deps {
            if from >= n {
                return Err(SimError::Invalid {
                    why: "graph dependency",
                });
            }
        }
        let alloc = self.graph_add_node_kind(graph, params)?;
        let node = self.graph_len(graph)?.saturating_sub(1);
        self.graph_bind_new_deps(graph, node, deps)?;
        Ok(GraphAddNode { node, alloc })
    }

    fn graph_add_node_kind(
        &mut self,
        graph: GraphId,
        params: GraphNodeParams,
    ) -> Result<Option<AllocId>, SimError> {
        match params {
            GraphNodeParams::Kernel(p) => {
                self.graph_add_kernel_params(graph, p)?;
                Ok(None)
            }
            GraphNodeParams::Memcpy(op) => {
                self.graph_add_memcpy(graph, op)?;
                Ok(None)
            }
            GraphNodeParams::Memset(op) => {
                self.graph_add_memset_op(graph, op)?;
                Ok(None)
            }
            GraphNodeParams::Host(p) => {
                self.graph_add_host_func_params(graph, p)?;
                Ok(None)
            }
            GraphNodeParams::Empty => {
                self.graph_add_empty(graph)?;
                Ok(None)
            }
            GraphNodeParams::EventRecord { event, external } => {
                self.graph_add_event_record(graph, event, external)?;
                Ok(None)
            }
            GraphNodeParams::EventWait { event, external } => {
                self.graph_add_event_wait(graph, event, external)?;
                Ok(None)
            }
            GraphNodeParams::ChildGraph(child) => {
                self.graph_add_child(graph, child)?;
                Ok(None)
            }
            GraphNodeParams::Alloc { bytes } => Ok(Some(self.graph_add_alloc(graph, bytes)?)),
            GraphNodeParams::Free(id) => {
                self.graph_add_free(graph, id)?;
                Ok(None)
            }
            GraphNodeParams::BatchMemOp(ops) => {
                self.graph_add_batch_mem_op(graph, &ops)?;
                Ok(None)
            }
        }
    }

    fn graph_bind_new_deps(
        &mut self,
        graph: GraphId,
        node: usize,
        deps: &[usize],
    ) -> Result<(), SimError> {
        if deps.is_empty() {
            return Ok(());
        }
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let n = g.steps.len();
        for &from in deps {
            if from == node || from >= n {
                return Err(SimError::Invalid {
                    why: "graph dependency",
                });
            }
            if g.steps.get(from).is_some_and(|s| s.destroyed) {
                return Err(SimError::Invalid {
                    why: "graph dependency",
                });
            }
        }
        let step = live_ok_mut(g.steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        for &from in deps {
            if !step.deps.contains(&from) {
                step.deps.push(from);
            }
        }
        step.deps.sort_unstable();
        Ok(())
    }

    fn graph_add_kernel_params(
        &mut self,
        graph: GraphId,
        params: KernelNodeParams,
    ) -> Result<(), SimError> {
        let (device, stream) = self.graph_origin_for_add(graph)?;
        if params.cooperative {
            self.require_cooperative(device)?;
        }
        let reads = self.resolve_bufs(&params.reads)?;
        let writes = self.resolve_bufs(&params.writes)?;
        self.graph_push(
            graph,
            device,
            stream,
            Kind::Kernel {
                kind: params.kind,
                reads,
                writes,
                cooperative: params.cooperative,
            },
        )
    }

    /// `cudaGraphAddKernelNode` on a [`Self::create_graph`] definition.
    ///
    /// Nodes start with no dependencies (CUDA). Use
    /// [`Self::graph_add_dependencies`] so a later node waits. Independent
    /// kernels may Hyper-Q overlap at [`Self::launch_graph`]. Capture cannot
    /// include it. Illegal on an instantiated exec. Does not run the
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
    /// Illegal on an instantiated exec. Device must advertise
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
    /// [`Self::graph_add_memcpy_1d`] is `cudaGraphAddMemcpyNode1D`.
    pub fn graph_add_memcpy(&mut self, graph: GraphId, op: MemcpyOp) -> Result<(), SimError> {
        let (device, stream) = self.graph_origin_for_add(graph)?;
        let _a = self.alloc_ref(op.alloc)?;
        if op.src.is_pageable() || op.dst.is_pageable() {
            return Err(SimError::Invalid {
                why: "cannot add pageable memcpy",
            });
        }
        memcpy_2d_check(&op)?;
        self.graph_push(graph, device, stream, Kind::Memcpy(op))
    }

    /// `cudaGraphAddMemcpyNode1D`. Pageable copies cannot be graph nodes.
    pub fn graph_add_memcpy_1d(
        &mut self,
        graph: GraphId,
        src: Place,
        dst: Place,
        alloc: AllocId,
        bytes: u64,
    ) -> Result<(), SimError> {
        self.graph_add_memcpy(graph, MemcpyOp::packed_1d(src, dst, alloc, bytes))
    }

    /// `cudaGraphAddMemsetNode` of a [`KernelBuf`] span (packed 1D).
    pub fn graph_add_memset(&mut self, graph: GraphId, buf: KernelBuf) -> Result<(), SimError> {
        self.graph_add_memset_op(graph, MemsetOp::from(buf))
    }

    /// `cudaGraphAddMemsetNode` / `cudaMemset2D` params ([`MemsetOp`]).
    pub fn graph_add_memset_op(&mut self, graph: GraphId, op: MemsetOp) -> Result<(), SimError> {
        let (device, stream) = self.graph_origin_for_add(graph)?;
        let op = self.resolve_memset_op(op)?;
        self.graph_push(graph, device, stream, Kind::Memset(op))
    }

    /// `cudaGraphAddHostNode` (`cudaLaunchHostFunc`) with the unnamed callback.
    pub fn graph_add_host_func(&mut self, graph: GraphId) -> Result<(), SimError> {
        self.graph_add_host_func_params(graph, HostNodeParams::default())
    }

    /// `cudaGraphAddHostNode` with [`HostNodeParams`] (`cudaHostFn_t` / `userData`).
    pub fn graph_add_host_func_params(
        &mut self,
        graph: GraphId,
        params: HostNodeParams,
    ) -> Result<(), SimError> {
        let (device, stream) = self.graph_origin_for_add(graph)?;
        self.graph_push(
            graph,
            device,
            stream,
            Kind::HostFunc {
                fn_id: params.fn_id,
                user_data: params.user_data,
            },
        )
    }

    /// `cudaGraphAddEmptyNode`: join/fork with no work.
    ///
    /// Completes in 1 ns and does not occupy compute or copy engines, so
    /// leftover kernels may Hyper-Q overlap it. Capture cannot include it.
    /// Illegal on an instantiated exec.
    pub fn graph_add_empty(&mut self, graph: GraphId) -> Result<(), SimError> {
        let (device, stream) = self.graph_origin_for_add(graph)?;
        self.graph_push(graph, device, stream, Kind::Empty)
    }

    /// `cudaGraphConditionalHandleCreate` on an uninstantiated graph.
    ///
    /// `default` is applied at each [`Self::launch_graph`] of that graph tree
    /// (`cudaGraphCondAssignDefault`). Capture cannot include it. Illegal on
    /// an instantiated exec.
    pub fn graph_conditional_create(
        &mut self,
        graph: GraphId,
        default: u32,
    ) -> Result<CondId, SimError> {
        self.fail_if_capturing("cannot capture conditional create")?;
        let origin = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            if g.instantiated {
                return Err(SimError::Invalid {
                    why: "graph instantiated",
                });
            }
            g.origin.0
        };
        let _gpu = self.profile.gpu(origin)?;
        let id = CondId(self.next_cond);
        self.next_cond = self.next_cond.saturating_add(1);
        let _prev = self.conds.insert(
            id,
            Cond {
                graph,
                default,
                value: default,
            },
        );
        self.clock = self.clock.saturating_add(1);
        Ok(id)
    }

    /// `cudaGraphAddNode` IF (`cudaGraphCondTypeIf`). Returns the body graph.
    ///
    /// Add nodes to the body, then instantiate the parent. Body ops skip at
    /// start when `handle` is `0`. `handle` must have been created on `graph`.
    /// Capture cannot include it. Illegal on an instantiated exec.
    pub fn graph_add_if(&mut self, graph: GraphId, handle: CondId) -> Result<GraphId, SimError> {
        let (device, stream) = self.graph_origin_for_add(graph)?;
        self.require_cond_on_graph(handle, graph)?;
        let body = self.insert_graph(device, stream);
        self.graph_push(graph, device, stream, Kind::If { handle, body })?;
        Ok(body)
    }

    /// `cudaGraphAddNode` WHILE (`cudaGraphCondTypeWhile`). Returns the body.
    ///
    /// Each iteration skips at start when `handle` is `0`. A body that leaves
    /// the handle non-zero is Invalid after 64 iterations. Capture cannot
    /// include it. Illegal on an instantiated exec.
    pub fn graph_add_while(&mut self, graph: GraphId, handle: CondId) -> Result<GraphId, SimError> {
        let (device, stream) = self.graph_origin_for_add(graph)?;
        self.require_cond_on_graph(handle, graph)?;
        let body = self.insert_graph(device, stream);
        self.graph_push(graph, device, stream, Kind::While { handle, body })?;
        Ok(body)
    }

    /// `cudaGraphAddNode` SWITCH (`cudaGraphCondTypeSwitch`). Returns `n` bodies.
    ///
    /// Branch `i` runs when the handle equals `i`. Out of range skips every
    /// body. `n` must be `1..=64`. Capture cannot include it. Illegal on an
    /// instantiated exec.
    pub fn graph_add_switch(
        &mut self,
        graph: GraphId,
        handle: CondId,
        n: u32,
    ) -> Result<Vec<GraphId>, SimError> {
        if n == 0 || n > 64 {
            return Err(SimError::Invalid {
                why: "switch branches",
            });
        }
        let (device, stream) = self.graph_origin_for_add(graph)?;
        self.require_cond_on_graph(handle, graph)?;
        let mut bodies = Vec::new();
        for _ in 0..n {
            bodies.push(self.insert_graph(device, stream));
        }
        self.graph_push(
            graph,
            device,
            stream,
            Kind::Switch {
                handle,
                bodies: bodies.clone(),
            },
        )?;
        Ok(bodies)
    }

    fn require_cond_on_graph(&self, handle: CondId, graph: GraphId) -> Result<(), SimError> {
        let cond_graph = self
            .conds
            .get(&handle)
            .ok_or(SimError::Invalid {
                why: "unknown conditional",
            })?
            .graph;
        if cond_graph != graph {
            return Err(SimError::Invalid {
                why: "conditional graph mismatch",
            });
        }
        Ok(())
    }

    /// Device `cudaGraphSetConditional`: write `handle` when this op starts.
    ///
    /// Capture is allowed. Does not occupy compute or copy engines. A later IF
    /// / WHILE / SWITCH node waits for this op if it depends on it. Each
    /// [`Self::launch_graph`] resets handles to their create-time default first,
    /// so a live set before launch is wiped.
    pub fn set_conditional(
        &mut self,
        device: DeviceId,
        handle: CondId,
        value: u32,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        if !self.conds.contains_key(&handle) {
            return Err(SimError::Invalid {
                why: "unknown conditional",
            });
        }
        self.submit(device, stream, Kind::SetConditional { handle, value })
    }

    /// IF nodes on `graph` as `(index, handle, body)` in add order.
    pub fn graph_if_nodes(
        &self,
        graph: GraphId,
    ) -> Result<Vec<(usize, CondId, GraphId)>, SimError> {
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let mut out = Vec::new();
        for (i, step) in g.steps.iter().enumerate() {
            if step.destroyed {
                continue;
            }
            if let Kind::If { handle, body } = step.kind {
                out.push((i, handle, body));
            }
        }
        Ok(out)
    }

    /// WHILE nodes on `graph` as `(index, handle, body)` in add order.
    pub fn graph_while_nodes(
        &self,
        graph: GraphId,
    ) -> Result<Vec<(usize, CondId, GraphId)>, SimError> {
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let mut out = Vec::new();
        for (i, step) in g.steps.iter().enumerate() {
            if step.destroyed {
                continue;
            }
            if let Kind::While { handle, body } = step.kind {
                out.push((i, handle, body));
            }
        }
        Ok(out)
    }

    /// SWITCH nodes on `graph` as `(index, handle, bodies)` in add order.
    pub fn graph_switch_nodes(
        &self,
        graph: GraphId,
    ) -> Result<Vec<(usize, CondId, Vec<GraphId>)>, SimError> {
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let mut out = Vec::new();
        for (i, step) in g.steps.iter().enumerate() {
            if step.destroyed {
                continue;
            }
            if let Kind::Switch { handle, bodies } = &step.kind {
                out.push((i, *handle, bodies.clone()));
            }
        }
        Ok(out)
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

    /// `cuStreamWriteValue64` as a `cudaGraphAddBatchMemOpNode`.
    ///
    /// Capture cannot include it (use [`Self::write_value64`] during capture).
    /// Illegal on an instantiated exec.
    pub fn graph_add_write_value64(
        &mut self,
        graph: GraphId,
        id: AllocId,
        offset: u64,
        value: u64,
    ) -> Result<(), SimError> {
        self.graph_add_batch_item(
            graph,
            BatchMemOp::Write {
                id,
                offset,
                value,
                bits32: false,
            },
        )
    }

    /// `cuStreamWriteValue32` as a `cudaGraphAddBatchMemOpNode`.
    pub fn graph_add_write_value32(
        &mut self,
        graph: GraphId,
        id: AllocId,
        offset: u64,
        value: u64,
    ) -> Result<(), SimError> {
        self.graph_add_batch_item(
            graph,
            BatchMemOp::Write {
                id,
                offset,
                value,
                bits32: true,
            },
        )
    }

    /// `cuStreamWaitValue64` as a `cudaGraphAddBatchMemOpNode`.
    pub fn graph_add_wait_value64(
        &mut self,
        graph: GraphId,
        id: AllocId,
        offset: u64,
        value: u64,
        cmp: WaitValueCmp,
    ) -> Result<(), SimError> {
        self.graph_add_batch_item(
            graph,
            BatchMemOp::Wait {
                id,
                offset,
                value,
                bits32: false,
                cmp,
            },
        )
    }

    /// `cuStreamWaitValue32` as a `cudaGraphAddBatchMemOpNode`.
    pub fn graph_add_wait_value32(
        &mut self,
        graph: GraphId,
        id: AllocId,
        offset: u64,
        value: u64,
        cmp: WaitValueCmp,
    ) -> Result<(), SimError> {
        self.graph_add_batch_item(
            graph,
            BatchMemOp::Wait {
                id,
                offset,
                value,
                bits32: true,
                cmp,
            },
        )
    }

    /// `cudaGraphAddBatchMemOpNode`: one node holding the wait/write vector.
    ///
    /// Empty is Invalid. Items run in order inside the node at launch (a wait
    /// sees earlier writes in this vector; it does not see later ones). Capture
    /// cannot include it (use [`Self::batch_mem_op`] during capture). Illegal
    /// after instantiate.
    pub fn graph_add_batch_mem_op(
        &mut self,
        graph: GraphId,
        ops: &[BatchMemOp],
    ) -> Result<(), SimError> {
        self.check_batch_mem_ops(ops)?;
        let (device, stream) = self.graph_origin_for_add(graph)?;
        self.graph_push(graph, device, stream, Kind::BatchMem { ops: ops.to_vec() })
    }

    fn graph_add_batch_item(&mut self, graph: GraphId, op: BatchMemOp) -> Result<(), SimError> {
        let (device, stream) = self.graph_origin_for_add(graph)?;
        self.check_batch_mem(op)?;
        self.graph_push(graph, device, stream, kind_from_batch(op))
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
        let (ready, origin) = {
            let c = self.graphs.get(&child).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            (c.ready(), c.origin.0)
        };
        if !ready {
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
    /// include it (use [`Self::alloc`] during stream capture). Illegal on an
    /// instantiated exec. [`Self::update_graph`] of mem nodes is Invalid.
    pub fn graph_add_alloc(&mut self, graph: GraphId, bytes: u64) -> Result<AllocId, SimError> {
        let (device, stream) = self.graph_origin_for_add(graph)?;
        if bytes == 0 {
            return Err(SimError::Invalid {
                why: "zero-byte alloc",
            });
        }
        let pool = self.graph_pool(device)?;
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
    /// Capture cannot include it. Illegal on an instantiated exec. Indices are
    /// 0-based in add order. A cycle is Invalid. Independent nodes (no edge)
    /// may Hyper-Q overlap at [`Self::launch_graph`].
    pub fn graph_add_dependencies(
        &mut self,
        graph: GraphId,
        from: usize,
        to: usize,
    ) -> Result<(), SimError> {
        self.graph_add_dependencies_n(graph, &[(from, to)])
    }

    /// `cudaGraphAddDependencies` of `numDependencies` from/to pairs.
    ///
    /// All-or-nothing: a cycle or out-of-range index adds nothing. Duplicate
    /// edges are a no-op. Empty `edges` is success. Capture cannot include it.
    /// Illegal on an instantiated exec.
    pub fn graph_add_dependencies_n(
        &mut self,
        graph: GraphId,
        edges: &[(usize, usize)],
    ) -> Result<(), SimError> {
        let _origin = self.graph_origin_for_add(graph)?;
        if edges.is_empty() {
            return Ok(());
        }
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let n = g.steps.len();
        let mut steps = g.steps.clone();
        for &(from, to) in edges {
            if from == to || from >= n || to >= n {
                return Err(SimError::Invalid {
                    why: "graph dependency",
                });
            }
            let from_live = steps.get(from).is_some_and(|s| !s.destroyed);
            let to_live = steps.get(to).is_some_and(|s| !s.destroyed);
            if !from_live || !to_live {
                return Err(SimError::Invalid {
                    why: "graph dependency",
                });
            }
            if graph_reaches(&steps, to, from) {
                return Err(SimError::Invalid {
                    why: "cyclic graph dependencies",
                });
            }
            let step = steps.get_mut(to).ok_or(SimError::Invalid {
                why: "graph dependency",
            })?;
            if !step.deps.contains(&from) {
                step.deps.push(from);
                step.deps.sort_unstable();
            }
        }
        for (dst, src) in g.steps.iter_mut().zip(steps) {
            dst.deps = src.deps;
        }
        Ok(())
    }

    /// `cudaGraphRemoveDependencies`: drop the `from` → `to` edge.
    ///
    /// Capture cannot include it. Illegal on an instantiated exec. Missing edges are
    /// a no-op. Independent nodes (no remaining edge) may Hyper-Q overlap at
    /// [`Self::launch_graph`].
    pub fn graph_remove_dependencies(
        &mut self,
        graph: GraphId,
        from: usize,
        to: usize,
    ) -> Result<(), SimError> {
        self.graph_remove_dependencies_n(graph, &[(from, to)])
    }

    /// `cudaGraphRemoveDependencies` of `numDependencies` from/to pairs.
    ///
    /// All-or-nothing on out-of-range indices (nothing is removed). Missing
    /// edges are a no-op. Empty `edges` is success. Capture cannot include it.
    /// Illegal on an instantiated exec.
    pub fn graph_remove_dependencies_n(
        &mut self,
        graph: GraphId,
        edges: &[(usize, usize)],
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot remove graph dependencies during capture")?;
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        if g.instantiated {
            return Err(SimError::Invalid {
                why: "graph instantiated",
            });
        }
        if edges.is_empty() {
            return Ok(());
        }
        let n = g.steps.len();
        for &(from, to) in edges {
            if from == to || from >= n || to >= n {
                return Err(SimError::Invalid {
                    why: "graph dependency",
                });
            }
            let from_live = g.steps.get(from).is_some_and(|s| !s.destroyed);
            let to_live = g.steps.get(to).is_some_and(|s| !s.destroyed);
            if !from_live || !to_live {
                return Err(SimError::Invalid {
                    why: "graph dependency",
                });
            }
        }
        for &(from, to) in edges {
            let step = g.steps.get_mut(to).ok_or(SimError::Invalid {
                why: "graph dependency",
            })?;
            step.deps.retain(|d| *d != from);
        }
        Ok(())
    }

    /// `cudaGraphDestroyNode` on a graph definition.
    ///
    /// Drops the node and incident edges. Remaining indices stay valid (CUDA
    /// handles). Capture cannot include it. Illegal on an instantiated exec.
    /// Does not retarget an already-instantiated exec. Destroying a mem alloc
    /// node unlinks it from [`Self::graph_mem_allocs`]. Nested child-graph
    /// objects are not destroyed.
    pub fn graph_destroy_node(&mut self, graph: GraphId, node: usize) -> Result<(), SimError> {
        self.fail_if_capturing("cannot destroy graph node during capture")?;
        let alloc = {
            let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            if g.instantiated {
                return Err(SimError::Invalid {
                    why: "graph instantiated",
                });
            }
            let alloc = {
                let step = live_ok(g.steps.get(node).ok_or(SimError::Invalid {
                    why: "unknown graph node",
                })?)?;
                match &step.kind {
                    Kind::Alloc { id, .. } => Some(*id),
                    _ => None,
                }
            };
            for s in &mut g.steps {
                s.deps.retain(|d| *d != node);
            }
            let step = g.steps.get_mut(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?;
            step.destroyed = true;
            step.deps.clear();
            alloc
        };
        if let Some(id) = alloc {
            if let Some(v) = self.graph_allocs.get_mut(&graph) {
                v.retain(|x| *x != id);
            }
        }
        Ok(())
    }

    /// Predecessor indices of node `i` (`cudaGraphNodeGetDependencies`).
    pub fn graph_node_deps(&self, graph: GraphId, i: usize) -> Result<Vec<usize>, SimError> {
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        g.steps
            .get(i)
            .ok_or(SimError::Invalid {
                why: "graph dependency",
            })
            .and_then(live_ok)
            .map(|s| s.deps.clone())
    }

    /// Root node indices (`cudaGraphGetRootNodes`): nodes with no predecessors.
    pub fn graph_root_nodes(&self, graph: GraphId) -> Result<Vec<usize>, SimError> {
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        Ok(g.steps
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.destroyed && s.deps.is_empty())
            .map(|(i, _)| i)
            .collect())
    }

    /// Edges (`cudaGraphGetEdges`): `(from, to)` in node-add order.
    pub fn graph_edges(&self, graph: GraphId) -> Result<Vec<(usize, usize)>, SimError> {
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let mut edges = Vec::new();
        for (to, step) in g.steps.iter().enumerate() {
            if step.destroyed {
                continue;
            }
            for &from in &step.deps {
                edges.push((from, to));
            }
        }
        Ok(edges)
    }

    /// `cudaGraphDebugDotPrint` of stored node kinds and edges.
    ///
    /// Query; legal during capture. Destination graph only during capture
    /// (same as [`Self::graph_len`]). Flags `0` prints kinds and edges only.
    pub fn graph_debug_dot(&self, graph: GraphId) -> Result<String, SimError> {
        self.graph_debug_dot_with_flags(graph, 0)
    }

    /// `cudaGraphDebugDotPrint` with [`GraphDebugDotFlags`].
    ///
    /// Query; legal during capture. Unknown bits (including external-semaphore
    /// and extra-conditional-edge flags) are Invalid `"graph debug dot flags"`.
    /// [`GraphDebugDotFlags::VERBOSE`] dumps every modeled param class.
    pub fn graph_debug_dot_with_flags(
        &self,
        graph: GraphId,
        flags: u32,
    ) -> Result<String, SimError> {
        const KNOWN: u32 = GraphDebugDotFlags::VERBOSE
            | GraphDebugDotFlags::KERNEL_NODE_PARAMS
            | GraphDebugDotFlags::MEMCPY_NODE_PARAMS
            | GraphDebugDotFlags::MEMSET_NODE_PARAMS
            | GraphDebugDotFlags::HOST_NODE_PARAMS
            | GraphDebugDotFlags::EVENT_NODE_PARAMS
            | GraphDebugDotFlags::KERNEL_NODE_ATTRIBUTES
            | GraphDebugDotFlags::HANDLES
            | GraphDebugDotFlags::MEM_ALLOC_NODE_PARAMS
            | GraphDebugDotFlags::MEM_FREE_NODE_PARAMS
            | GraphDebugDotFlags::BATCH_MEM_OP_NODE_PARAMS
            | GraphDebugDotFlags::CONDITIONAL_NODE_PARAMS;
        if flags & !KNOWN != 0 {
            return Err(SimError::Invalid {
                why: "graph debug dot flags",
            });
        }
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let dump = if flags & GraphDebugDotFlags::VERBOSE != 0 {
            KNOWN
        } else {
            flags
        };
        let mut out = if dump & GraphDebugDotFlags::HANDLES != 0 {
            let mut s = String::from("digraph g");
            s.push_str(&graph.0.to_string());
            s.push_str(" {\n");
            s
        } else {
            String::from("digraph {\n")
        };
        for (i, step) in g.steps.iter().enumerate() {
            if step.destroyed {
                continue;
            }
            out.push_str("  n");
            out.push_str(&i.to_string());
            out.push_str(" [label=\"");
            out.push_str(&debug_dot_label(i, step, dump));
            out.push_str("\"];\n");
        }
        for (to, step) in g.steps.iter().enumerate() {
            if step.destroyed {
                continue;
            }
            for from in &step.deps {
                out.push_str("  n");
                out.push_str(&from.to_string());
                out.push_str(" -> n");
                out.push_str(&to.to_string());
                out.push_str(";\n");
            }
        }
        out.push_str("}\n");
        Ok(out)
    }

    /// Successors of node `i` (`cudaGraphNodeGetDependentNodes`).
    pub fn graph_node_dependents(&self, graph: GraphId, i: usize) -> Result<Vec<usize>, SimError> {
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        if i >= g.steps.len() {
            return Err(SimError::Invalid {
                why: "graph dependency",
            });
        }
        if g.steps.get(i).is_some_and(|s| s.destroyed) {
            return Err(SimError::Invalid {
                why: "unknown graph node",
            });
        }
        Ok(g.steps
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.destroyed && s.deps.contains(&i))
            .map(|(j, _)| j)
            .collect())
    }

    /// `cudaGraphNodeGetType` for node `i`.
    pub fn graph_node_kind(&self, graph: GraphId, i: usize) -> Result<GraphNodeKind, SimError> {
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let st = live_ok(g.steps.get(i).ok_or(SimError::Invalid {
            why: "graph dependency",
        })?)?;
        Ok(node_kind(&st.kind))
    }

    /// `cudaGraphKernelNodeGetAttribute` for priority on the graph definition.
    pub fn graph_kernel_node_get_priority(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<i32, SimError> {
        self.kernel_node_priority(graph, node, false)
    }

    /// `cudaGraphExecKernelNodeGetAttribute` for priority on the exec snapshot.
    pub fn graph_exec_kernel_node_get_priority(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<i32, SimError> {
        self.kernel_node_priority(exec, node, true)
    }

    fn kernel_node_priority(
        &self,
        graph: GraphId,
        node: usize,
        exec: bool,
    ) -> Result<i32, SimError> {
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.view() } else { &g.steps };
        let step = live_ok(steps.get(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        Ok(step.priority)
    }

    /// `cudaGraphKernelNodeSetAttribute` for priority on the graph definition.
    ///
    /// After instantiate this does not retarget the exec; use
    /// [`Self::graph_exec_kernel_node_set_priority`]. Capture cannot include it.
    pub fn graph_kernel_node_set_priority(
        &mut self,
        graph: GraphId,
        node: usize,
        priority: i32,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set attribute")?;
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        step.priority = priority;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphExecKernelNodeSetAttribute` for priority on the exec snapshot.
    pub fn graph_exec_kernel_node_set_priority(
        &mut self,
        exec: GraphId,
        node: usize,
        priority: i32,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set attribute")?;
        let exec = self.as_exec(exec)?;
        let g = self.graphs.get_mut(&exec).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.exec_mut()?.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        step.priority = priority;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphKernelNodeGetAttribute` for programmatic stream serialization.
    pub fn graph_kernel_node_get_pdl(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<ProgrammaticLaunch, SimError> {
        self.kernel_node_pdl(graph, node, false)
    }

    /// `cudaGraphExecKernelNodeGetAttribute` for PDL on the exec snapshot.
    pub fn graph_exec_kernel_node_get_pdl(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<ProgrammaticLaunch, SimError> {
        self.kernel_node_pdl(exec, node, true)
    }

    fn kernel_node_pdl(
        &self,
        graph: GraphId,
        node: usize,
        exec: bool,
    ) -> Result<ProgrammaticLaunch, SimError> {
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.view() } else { &g.steps };
        let step = live_ok(steps.get(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        Ok(step.pdl)
    }

    /// `cudaGraphKernelNodeSetAttribute` for PDL on the graph definition.
    pub fn graph_kernel_node_set_pdl(
        &mut self,
        graph: GraphId,
        node: usize,
        pdl: ProgrammaticLaunch,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set attribute")?;
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        step.pdl = pdl;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphExecKernelNodeSetAttribute` for PDL on the exec snapshot.
    pub fn graph_exec_kernel_node_set_pdl(
        &mut self,
        exec: GraphId,
        node: usize,
        pdl: ProgrammaticLaunch,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set attribute")?;
        let exec = self.as_exec(exec)?;
        let g = self.graphs.get_mut(&exec).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.exec_mut()?.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        step.pdl = pdl;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphKernelNodeGetAttribute` for programmatic event on the definition.
    pub fn graph_kernel_node_get_programmatic_event(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<Option<ProgrammaticEvent>, SimError> {
        self.kernel_node_programmatic_event(graph, node, false)
    }

    /// `cudaGraphExecKernelNodeGetAttribute` for programmatic event on the exec.
    pub fn graph_exec_kernel_node_get_programmatic_event(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<Option<ProgrammaticEvent>, SimError> {
        self.kernel_node_programmatic_event(exec, node, true)
    }

    fn kernel_node_programmatic_event(
        &self,
        graph: GraphId,
        node: usize,
        exec: bool,
    ) -> Result<Option<ProgrammaticEvent>, SimError> {
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.view() } else { &g.steps };
        let step = live_ok(steps.get(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        Ok(step.programmatic_event)
    }

    /// `cudaGraphKernelNodeSetAttribute` for programmatic event on the definition.
    pub fn graph_kernel_node_set_programmatic_event(
        &mut self,
        graph: GraphId,
        node: usize,
        event: Option<ProgrammaticEvent>,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set attribute")?;
        {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let step = live_ok(g.steps.get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            if !matches!(step.kind, Kind::Kernel { .. }) {
                return Err(SimError::Invalid {
                    why: "not a kernel node",
                });
            }
        }
        if let Some(pe) = event {
            let _ev = self.events.entry(pe.event).or_insert(Ev::new(true));
        }
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.programmatic_event = event;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphExecKernelNodeSetAttribute` for programmatic event on the exec.
    pub fn graph_exec_kernel_node_set_programmatic_event(
        &mut self,
        exec: GraphId,
        node: usize,
        event: Option<ProgrammaticEvent>,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set attribute")?;
        let exec = self.as_exec(exec)?;
        if let Some(pe) = event {
            let _ev = self.events.entry(pe.event).or_insert(Ev::new(true));
        }
        let g = self.graphs.get_mut(&exec).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.exec_mut()?.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        step.programmatic_event = event;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphKernelNodeGetAttribute` for launch-completion event on the definition.
    pub fn graph_kernel_node_get_launch_completion(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<Option<LaunchCompletionEvent>, SimError> {
        self.kernel_node_launch_completion(graph, node, false)
    }

    /// `cudaGraphExecKernelNodeGetAttribute` for launch-completion event on the exec.
    pub fn graph_exec_kernel_node_get_launch_completion(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<Option<LaunchCompletionEvent>, SimError> {
        self.kernel_node_launch_completion(exec, node, true)
    }

    fn kernel_node_launch_completion(
        &self,
        graph: GraphId,
        node: usize,
        exec: bool,
    ) -> Result<Option<LaunchCompletionEvent>, SimError> {
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.view() } else { &g.steps };
        let step = live_ok(steps.get(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        Ok(step.launch_completion)
    }

    /// `cudaGraphKernelNodeSetAttribute` for launch-completion event on the definition.
    pub fn graph_kernel_node_set_launch_completion(
        &mut self,
        graph: GraphId,
        node: usize,
        event: Option<LaunchCompletionEvent>,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set attribute")?;
        {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let step = live_ok(g.steps.get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            if !matches!(step.kind, Kind::Kernel { .. }) {
                return Err(SimError::Invalid {
                    why: "not a kernel node",
                });
            }
        }
        if let Some(lc) = event {
            let _ev = self.events.entry(lc.event).or_insert(Ev::new(true));
        }
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.launch_completion = event;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphExecKernelNodeSetAttribute` for launch-completion event on the exec.
    pub fn graph_exec_kernel_node_set_launch_completion(
        &mut self,
        exec: GraphId,
        node: usize,
        event: Option<LaunchCompletionEvent>,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set attribute")?;
        let exec = self.as_exec(exec)?;
        if let Some(lc) = event {
            let _ev = self.events.entry(lc.event).or_insert(Ev::new(true));
        }
        let g = self.graphs.get_mut(&exec).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.exec_mut()?.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        step.launch_completion = event;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphKernelNodeGetAttribute` for access-policy window on the definition.
    pub fn graph_kernel_node_get_access_policy(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<Option<AccessPolicyWindow>, SimError> {
        self.kernel_node_access_policy(graph, node, false)
    }

    /// `cudaGraphExecKernelNodeGetAttribute` for access-policy window on the exec.
    pub fn graph_exec_kernel_node_get_access_policy(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<Option<AccessPolicyWindow>, SimError> {
        self.kernel_node_access_policy(exec, node, true)
    }

    fn kernel_node_access_policy(
        &self,
        graph: GraphId,
        node: usize,
        exec: bool,
    ) -> Result<Option<AccessPolicyWindow>, SimError> {
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.view() } else { &g.steps };
        let step = live_ok(steps.get(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        Ok(step.access_policy)
    }

    /// `cudaGraphKernelNodeSetAttribute` for access-policy window on the definition.
    pub fn graph_kernel_node_set_access_policy(
        &mut self,
        graph: GraphId,
        node: usize,
        window: Option<AccessPolicyWindow>,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set attribute")?;
        let device = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let step = live_ok(g.steps.get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            if !matches!(step.kind, Kind::Kernel { .. }) {
                return Err(SimError::Invalid {
                    why: "not a kernel node",
                });
            }
            step.device
        };
        if let Some(w) = window {
            self.validate_access_policy_window(device, w)?;
        }
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        step.access_policy = window;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphExecKernelNodeSetAttribute` for access-policy window on the exec.
    pub fn graph_exec_kernel_node_set_access_policy(
        &mut self,
        exec: GraphId,
        node: usize,
        window: Option<AccessPolicyWindow>,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set attribute")?;
        let exec = self.as_exec(exec)?;
        let device = {
            let g = self.graphs.get(&exec).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let step = live_ok(g.view().get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            if !matches!(step.kind, Kind::Kernel { .. }) {
                return Err(SimError::Invalid {
                    why: "not a kernel node",
                });
            }
            step.device
        };
        if let Some(w) = window {
            self.validate_access_policy_window(device, w)?;
        }
        let g = self.graphs.get_mut(&exec).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let step = live_ok_mut(g.exec_mut()?.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        step.access_policy = window;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphKernelNodeGetAttribute` for mem-sync domain on the definition.
    pub fn graph_kernel_node_get_mem_sync_domain(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<MemSyncDomain, SimError> {
        Ok(self.kernel_node_mem_sync(graph, node, false)?.0)
    }

    /// `cudaGraphExecKernelNodeGetAttribute` for mem-sync domain on the exec.
    pub fn graph_exec_kernel_node_get_mem_sync_domain(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<MemSyncDomain, SimError> {
        Ok(self.kernel_node_mem_sync(exec, node, true)?.0)
    }

    /// `cudaGraphKernelNodeGetAttribute` for mem-sync map on the definition.
    pub fn graph_kernel_node_get_mem_sync_domain_map(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<MemSyncDomainMap, SimError> {
        Ok(self.kernel_node_mem_sync(graph, node, false)?.1)
    }

    /// `cudaGraphExecKernelNodeGetAttribute` for mem-sync map on the exec.
    pub fn graph_exec_kernel_node_get_mem_sync_domain_map(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<MemSyncDomainMap, SimError> {
        Ok(self.kernel_node_mem_sync(exec, node, true)?.1)
    }

    fn kernel_node_mem_sync(
        &self,
        graph: GraphId,
        node: usize,
        exec: bool,
    ) -> Result<(MemSyncDomain, MemSyncDomainMap), SimError> {
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.view() } else { &g.steps };
        let step = live_ok(steps.get(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        Ok((step.mem_sync_domain, step.mem_sync_map))
    }

    /// `cudaGraphKernelNodeSetAttribute` for mem-sync domain on the definition.
    pub fn graph_kernel_node_set_mem_sync_domain(
        &mut self,
        graph: GraphId,
        node: usize,
        domain: MemSyncDomain,
    ) -> Result<(), SimError> {
        self.set_kernel_node_mem_sync_domain(graph, node, false, domain)
    }

    /// `cudaGraphExecKernelNodeSetAttribute` for mem-sync domain on the exec.
    pub fn graph_exec_kernel_node_set_mem_sync_domain(
        &mut self,
        exec: GraphId,
        node: usize,
        domain: MemSyncDomain,
    ) -> Result<(), SimError> {
        self.set_kernel_node_mem_sync_domain(exec, node, true, domain)
    }

    fn set_kernel_node_mem_sync_domain(
        &mut self,
        graph: GraphId,
        node: usize,
        exec: bool,
        domain: MemSyncDomain,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set attribute")?;
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.exec_mut()? } else { &mut g.steps };
        let step = live_ok_mut(steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        step.mem_sync_domain = domain;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphKernelNodeSetAttribute` for mem-sync map on the definition.
    pub fn graph_kernel_node_set_mem_sync_domain_map(
        &mut self,
        graph: GraphId,
        node: usize,
        map: MemSyncDomainMap,
    ) -> Result<(), SimError> {
        self.set_kernel_node_mem_sync_map(graph, node, false, map)
    }

    /// `cudaGraphExecKernelNodeSetAttribute` for mem-sync map on the exec.
    pub fn graph_exec_kernel_node_set_mem_sync_domain_map(
        &mut self,
        exec: GraphId,
        node: usize,
        map: MemSyncDomainMap,
    ) -> Result<(), SimError> {
        self.set_kernel_node_mem_sync_map(exec, node, true, map)
    }

    fn set_kernel_node_mem_sync_map(
        &mut self,
        graph: GraphId,
        node: usize,
        exec: bool,
        map: MemSyncDomainMap,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set attribute")?;
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let device = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let steps = if exec { g.view() } else { &g.steps };
            let step = live_ok(steps.get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            if !matches!(step.kind, Kind::Kernel { .. }) {
                return Err(SimError::Invalid {
                    why: "not a kernel node",
                });
            }
            step.device
        };
        self.validate_mem_sync_map(device, map)?;
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.exec_mut()? } else { &mut g.steps };
        let step = live_ok_mut(steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.mem_sync_map = map;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphKernelNodeGetAttribute` for cluster dimension on the definition.
    pub fn graph_kernel_node_get_cluster(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<Option<ClusterDim>, SimError> {
        self.kernel_node_cluster(graph, node, false)
    }

    /// `cudaGraphExecKernelNodeGetAttribute` for cluster dimension on the exec.
    pub fn graph_exec_kernel_node_get_cluster(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<Option<ClusterDim>, SimError> {
        self.kernel_node_cluster(exec, node, true)
    }

    fn kernel_node_cluster(
        &self,
        graph: GraphId,
        node: usize,
        exec: bool,
    ) -> Result<Option<ClusterDim>, SimError> {
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.view() } else { &g.steps };
        let step = live_ok(steps.get(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        Ok(step.cluster)
    }

    /// `cudaGraphKernelNodeSetAttribute` for cluster dimension on the definition.
    pub fn graph_kernel_node_set_cluster(
        &mut self,
        graph: GraphId,
        node: usize,
        cluster: Option<ClusterDim>,
    ) -> Result<(), SimError> {
        self.set_kernel_node_cluster(graph, node, false, cluster)
    }

    /// `cudaGraphExecKernelNodeSetAttribute` for cluster dimension on the exec.
    pub fn graph_exec_kernel_node_set_cluster(
        &mut self,
        exec: GraphId,
        node: usize,
        cluster: Option<ClusterDim>,
    ) -> Result<(), SimError> {
        self.set_kernel_node_cluster(exec, node, true, cluster)
    }

    fn set_kernel_node_cluster(
        &mut self,
        graph: GraphId,
        node: usize,
        exec: bool,
        cluster: Option<ClusterDim>,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set attribute")?;
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let (device, mode) = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let steps = if exec { g.view() } else { &g.steps };
            let step = live_ok(steps.get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            if !matches!(step.kind, Kind::Kernel { .. }) {
                return Err(SimError::Invalid {
                    why: "not a kernel node",
                });
            }
            (step.device, step.portable_cluster)
        };
        if let Some(c) = cluster {
            let _n = self.validate_cluster(device, c, mode)?;
        }
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.exec_mut()? } else { &mut g.steps };
        let step = live_ok_mut(steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.cluster = cluster;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphKernelNodeGetAttribute` for cluster scheduling policy.
    pub fn graph_kernel_node_get_cluster_policy(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<ClusterSchedulingPolicy, SimError> {
        self.kernel_node_cluster_policy(graph, node, false)
    }

    /// `cudaGraphExecKernelNodeGetAttribute` for cluster scheduling policy.
    pub fn graph_exec_kernel_node_get_cluster_policy(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<ClusterSchedulingPolicy, SimError> {
        self.kernel_node_cluster_policy(exec, node, true)
    }

    fn kernel_node_cluster_policy(
        &self,
        graph: GraphId,
        node: usize,
        exec: bool,
    ) -> Result<ClusterSchedulingPolicy, SimError> {
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.view() } else { &g.steps };
        let step = live_ok(steps.get(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        Ok(step.cluster_policy)
    }

    /// `cudaGraphKernelNodeSetAttribute` for cluster scheduling policy.
    pub fn graph_kernel_node_set_cluster_policy(
        &mut self,
        graph: GraphId,
        node: usize,
        policy: ClusterSchedulingPolicy,
    ) -> Result<(), SimError> {
        self.set_kernel_node_cluster_policy(graph, node, false, policy)
    }

    /// `cudaGraphExecKernelNodeSetAttribute` for cluster scheduling policy.
    pub fn graph_exec_kernel_node_set_cluster_policy(
        &mut self,
        exec: GraphId,
        node: usize,
        policy: ClusterSchedulingPolicy,
    ) -> Result<(), SimError> {
        self.set_kernel_node_cluster_policy(exec, node, true, policy)
    }

    fn set_kernel_node_cluster_policy(
        &mut self,
        graph: GraphId,
        node: usize,
        exec: bool,
        policy: ClusterSchedulingPolicy,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set attribute")?;
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.exec_mut()? } else { &mut g.steps };
        let step = live_ok_mut(steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        step.cluster_policy = policy;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphKernelNodeGetAttribute` for preferred cluster dimension.
    pub fn graph_kernel_node_get_preferred_cluster(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<Option<ClusterDim>, SimError> {
        self.kernel_node_preferred_cluster(graph, node, false)
    }

    /// `cudaGraphExecKernelNodeGetAttribute` for preferred cluster dimension.
    pub fn graph_exec_kernel_node_get_preferred_cluster(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<Option<ClusterDim>, SimError> {
        self.kernel_node_preferred_cluster(exec, node, true)
    }

    fn kernel_node_preferred_cluster(
        &self,
        graph: GraphId,
        node: usize,
        exec: bool,
    ) -> Result<Option<ClusterDim>, SimError> {
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.view() } else { &g.steps };
        let step = live_ok(steps.get(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        Ok(step.preferred_cluster)
    }

    /// `cudaGraphKernelNodeSetAttribute` for preferred cluster dimension.
    pub fn graph_kernel_node_set_preferred_cluster(
        &mut self,
        graph: GraphId,
        node: usize,
        preferred: Option<ClusterDim>,
    ) -> Result<(), SimError> {
        self.set_kernel_node_preferred_cluster(graph, node, false, preferred)
    }

    /// `cudaGraphExecKernelNodeSetAttribute` for preferred cluster dimension.
    pub fn graph_exec_kernel_node_set_preferred_cluster(
        &mut self,
        exec: GraphId,
        node: usize,
        preferred: Option<ClusterDim>,
    ) -> Result<(), SimError> {
        self.set_kernel_node_preferred_cluster(exec, node, true, preferred)
    }

    fn set_kernel_node_preferred_cluster(
        &mut self,
        graph: GraphId,
        node: usize,
        exec: bool,
        preferred: Option<ClusterDim>,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set attribute")?;
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let (device, cluster, mode) = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let steps = if exec { g.view() } else { &g.steps };
            let step = live_ok(steps.get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            if !matches!(step.kind, Kind::Kernel { .. }) {
                return Err(SimError::Invalid {
                    why: "not a kernel node",
                });
            }
            (step.device, step.cluster, step.portable_cluster)
        };
        self.validate_cluster_attrs(device, cluster, preferred, mode)?;
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.exec_mut()? } else { &mut g.steps };
        let step = live_ok_mut(steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.preferred_cluster = preferred;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphKernelNodeGetAttribute` for shared-memory carveout.
    pub fn graph_kernel_node_get_carveout(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<SharedMemCarveout, SimError> {
        self.kernel_node_carveout(graph, node, false)
    }

    /// `cudaGraphExecKernelNodeGetAttribute` for shared-memory carveout.
    pub fn graph_exec_kernel_node_get_carveout(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<SharedMemCarveout, SimError> {
        self.kernel_node_carveout(exec, node, true)
    }

    fn kernel_node_carveout(
        &self,
        graph: GraphId,
        node: usize,
        exec: bool,
    ) -> Result<SharedMemCarveout, SimError> {
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.view() } else { &g.steps };
        let step = live_ok(steps.get(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        Ok(step.carveout)
    }

    /// `cudaGraphKernelNodeSetAttribute` for shared-memory carveout.
    pub fn graph_kernel_node_set_carveout(
        &mut self,
        graph: GraphId,
        node: usize,
        carveout: SharedMemCarveout,
    ) -> Result<(), SimError> {
        self.set_kernel_node_carveout(graph, node, false, carveout)
    }

    /// `cudaGraphExecKernelNodeSetAttribute` for shared-memory carveout.
    pub fn graph_exec_kernel_node_set_carveout(
        &mut self,
        exec: GraphId,
        node: usize,
        carveout: SharedMemCarveout,
    ) -> Result<(), SimError> {
        self.set_kernel_node_carveout(exec, node, true, carveout)
    }

    fn set_kernel_node_carveout(
        &mut self,
        graph: GraphId,
        node: usize,
        exec: bool,
        carveout: SharedMemCarveout,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set attribute")?;
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.exec_mut()? } else { &mut g.steps };
        let step = live_ok_mut(steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        step.carveout = carveout;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphKernelNodeGetAttribute` for device-updatable kernel node.
    pub fn graph_kernel_node_get_device_updatable(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<bool, SimError> {
        self.kernel_node_device_updatable(graph, node, false)
    }

    /// `cudaGraphExecKernelNodeGetAttribute` for device-updatable kernel node.
    pub fn graph_exec_kernel_node_get_device_updatable(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<bool, SimError> {
        self.kernel_node_device_updatable(exec, node, true)
    }

    fn kernel_node_device_updatable(
        &self,
        graph: GraphId,
        node: usize,
        exec: bool,
    ) -> Result<bool, SimError> {
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.view() } else { &g.steps };
        let step = live_ok(steps.get(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        Ok(step.device_updatable)
    }

    /// `cudaGraphKernelNodeSetAttribute` for device-updatable kernel node.
    pub fn graph_kernel_node_set_device_updatable(
        &mut self,
        graph: GraphId,
        node: usize,
        device_updatable: bool,
    ) -> Result<(), SimError> {
        self.set_kernel_node_device_updatable(graph, node, false, device_updatable)
    }

    /// `cudaGraphExecKernelNodeSetAttribute` for device-updatable kernel node.
    pub fn graph_exec_kernel_node_set_device_updatable(
        &mut self,
        exec: GraphId,
        node: usize,
        device_updatable: bool,
    ) -> Result<(), SimError> {
        self.set_kernel_node_device_updatable(exec, node, true, device_updatable)
    }

    fn set_kernel_node_device_updatable(
        &mut self,
        graph: GraphId,
        node: usize,
        exec: bool,
        device_updatable: bool,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set attribute")?;
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.exec_mut()? } else { &mut g.steps };
        let step = live_ok_mut(steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        step.device_updatable = device_updatable;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphKernelNodeGetAttribute` for shared-memory bank mode.
    pub fn graph_kernel_node_get_shared_mem(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<SharedMemoryMode, SimError> {
        self.kernel_node_shared_mem(graph, node, false)
    }

    /// `cudaGraphExecKernelNodeGetAttribute` for shared-memory bank mode.
    pub fn graph_exec_kernel_node_get_shared_mem(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<SharedMemoryMode, SimError> {
        self.kernel_node_shared_mem(exec, node, true)
    }

    fn kernel_node_shared_mem(
        &self,
        graph: GraphId,
        node: usize,
        exec: bool,
    ) -> Result<SharedMemoryMode, SimError> {
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.view() } else { &g.steps };
        let step = live_ok(steps.get(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        Ok(step.shared_mem)
    }

    /// `cudaGraphKernelNodeSetAttribute` for shared-memory bank mode.
    pub fn graph_kernel_node_set_shared_mem(
        &mut self,
        graph: GraphId,
        node: usize,
        shared_mem: SharedMemoryMode,
    ) -> Result<(), SimError> {
        self.set_kernel_node_shared_mem(graph, node, false, shared_mem)
    }

    /// `cudaGraphExecKernelNodeSetAttribute` for shared-memory bank mode.
    pub fn graph_exec_kernel_node_set_shared_mem(
        &mut self,
        exec: GraphId,
        node: usize,
        shared_mem: SharedMemoryMode,
    ) -> Result<(), SimError> {
        self.set_kernel_node_shared_mem(exec, node, true, shared_mem)
    }

    fn set_kernel_node_shared_mem(
        &mut self,
        graph: GraphId,
        node: usize,
        exec: bool,
        shared_mem: SharedMemoryMode,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set attribute")?;
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.exec_mut()? } else { &mut g.steps };
        let step = live_ok_mut(steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        step.shared_mem = shared_mem;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphKernelNodeGetAttribute` for portable-cluster size mode.
    pub fn graph_kernel_node_get_portable_cluster(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<PortableClusterMode, SimError> {
        self.kernel_node_portable_cluster(graph, node, false)
    }

    /// `cudaGraphExecKernelNodeGetAttribute` for portable-cluster size mode.
    pub fn graph_exec_kernel_node_get_portable_cluster(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<PortableClusterMode, SimError> {
        self.kernel_node_portable_cluster(exec, node, true)
    }

    fn kernel_node_portable_cluster(
        &self,
        graph: GraphId,
        node: usize,
        exec: bool,
    ) -> Result<PortableClusterMode, SimError> {
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.view() } else { &g.steps };
        let step = live_ok(steps.get(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        Ok(step.portable_cluster)
    }

    /// `cudaGraphKernelNodeSetAttribute` for portable-cluster size mode.
    pub fn graph_kernel_node_set_portable_cluster(
        &mut self,
        graph: GraphId,
        node: usize,
        mode: PortableClusterMode,
    ) -> Result<(), SimError> {
        self.set_kernel_node_portable_cluster(graph, node, false, mode)
    }

    /// `cudaGraphExecKernelNodeSetAttribute` for portable-cluster size mode.
    pub fn graph_exec_kernel_node_set_portable_cluster(
        &mut self,
        exec: GraphId,
        node: usize,
        mode: PortableClusterMode,
    ) -> Result<(), SimError> {
        self.set_kernel_node_portable_cluster(exec, node, true, mode)
    }

    fn set_kernel_node_portable_cluster(
        &mut self,
        graph: GraphId,
        node: usize,
        exec: bool,
        mode: PortableClusterMode,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set attribute")?;
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let (device, cluster, preferred) = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let steps = if exec { g.view() } else { &g.steps };
            let step = live_ok(steps.get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            if !matches!(step.kind, Kind::Kernel { .. }) {
                return Err(SimError::Invalid {
                    why: "not a kernel node",
                });
            }
            (step.device, step.cluster, step.preferred_cluster)
        };
        self.validate_cluster_attrs(device, cluster, preferred, mode)?;
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.exec_mut()? } else { &mut g.steps };
        let step = live_ok_mut(steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.portable_cluster = mode;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphKernelNodeGetAttribute` for CUDA 13 portable-shared mode.
    pub fn graph_kernel_node_get_portable_shared(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<PortableSharedMode, SimError> {
        self.kernel_node_portable_shared(graph, node, false)
    }

    /// `cudaGraphExecKernelNodeGetAttribute` for CUDA 13 portable-shared mode.
    pub fn graph_exec_kernel_node_get_portable_shared(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<PortableSharedMode, SimError> {
        self.kernel_node_portable_shared(exec, node, true)
    }

    fn kernel_node_portable_shared(
        &self,
        graph: GraphId,
        node: usize,
        exec: bool,
    ) -> Result<PortableSharedMode, SimError> {
        Ok(self.kernel_node_shared_launch(graph, node, exec)?.1)
    }

    /// `cudaKernelNodeParams::sharedMemBytes` on the graph definition.
    pub fn graph_kernel_node_get_dynamic_shared(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<u32, SimError> {
        self.kernel_node_dynamic_shared(graph, node, false)
    }

    /// `cudaKernelNodeParams::sharedMemBytes` on the exec snapshot.
    pub fn graph_exec_kernel_node_get_dynamic_shared(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<u32, SimError> {
        self.kernel_node_dynamic_shared(exec, node, true)
    }

    fn kernel_node_dynamic_shared(
        &self,
        graph: GraphId,
        node: usize,
        exec: bool,
    ) -> Result<u32, SimError> {
        Ok(self.kernel_node_shared_launch(graph, node, exec)?.0)
    }

    fn kernel_node_shared_launch(
        &self,
        graph: GraphId,
        node: usize,
        exec: bool,
    ) -> Result<(u32, PortableSharedMode), SimError> {
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.view() } else { &g.steps };
        let step = live_ok(steps.get(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        Ok((step.dynamic_shared, step.portable_shared))
    }

    /// `cudaGraphKernelNodeSetAttribute` for CUDA 13 portable-shared mode.
    pub fn graph_kernel_node_set_portable_shared(
        &mut self,
        graph: GraphId,
        node: usize,
        mode: PortableSharedMode,
    ) -> Result<(), SimError> {
        self.set_kernel_node_portable_shared(graph, node, false, mode)
    }

    /// `cudaGraphExecKernelNodeSetAttribute` for CUDA 13 portable-shared mode.
    pub fn graph_exec_kernel_node_set_portable_shared(
        &mut self,
        exec: GraphId,
        node: usize,
        mode: PortableSharedMode,
    ) -> Result<(), SimError> {
        self.set_kernel_node_portable_shared(exec, node, true, mode)
    }

    fn set_kernel_node_portable_shared(
        &mut self,
        graph: GraphId,
        node: usize,
        exec: bool,
        mode: PortableSharedMode,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set attribute")?;
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let (device, bytes) = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let steps = if exec { g.view() } else { &g.steps };
            let step = live_ok(steps.get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            if !matches!(step.kind, Kind::Kernel { .. }) {
                return Err(SimError::Invalid {
                    why: "not a kernel node",
                });
            }
            (step.device, step.dynamic_shared)
        };
        self.validate_dynamic_shared(device, bytes, mode)?;
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.exec_mut()? } else { &mut g.steps };
        let step = live_ok_mut(steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.portable_shared = mode;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaKernelNodeParams::sharedMemBytes` on the graph definition.
    pub fn graph_kernel_node_set_dynamic_shared(
        &mut self,
        graph: GraphId,
        node: usize,
        bytes: u32,
    ) -> Result<(), SimError> {
        self.set_kernel_node_dynamic_shared(graph, node, false, bytes)
    }

    /// `cudaKernelNodeParams::sharedMemBytes` on the exec snapshot.
    pub fn graph_exec_kernel_node_set_dynamic_shared(
        &mut self,
        exec: GraphId,
        node: usize,
        bytes: u32,
    ) -> Result<(), SimError> {
        self.set_kernel_node_dynamic_shared(exec, node, true, bytes)
    }

    fn set_kernel_node_dynamic_shared(
        &mut self,
        graph: GraphId,
        node: usize,
        exec: bool,
        bytes: u32,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set attribute")?;
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let (device, mode) = {
            let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
                why: "unknown graph",
            })?;
            let steps = if exec { g.view() } else { &g.steps };
            let step = live_ok(steps.get(node).ok_or(SimError::Invalid {
                why: "unknown graph node",
            })?)?;
            if !matches!(step.kind, Kind::Kernel { .. }) {
                return Err(SimError::Invalid {
                    why: "not a kernel node",
                });
            }
            (step.device, step.portable_shared)
        };
        self.validate_dynamic_shared(device, bytes, mode)?;
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.exec_mut()? } else { &mut g.steps };
        let step = live_ok_mut(steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        step.dynamic_shared = bytes;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphKernelNodeGetAttribute` for NVLink-util-centric scheduling.
    pub fn graph_kernel_node_get_nvlink_util_centric(
        &self,
        graph: GraphId,
        node: usize,
    ) -> Result<bool, SimError> {
        self.kernel_node_nvlink_util_centric(graph, node, false)
    }

    /// `cudaGraphExecKernelNodeGetAttribute` for NVLink-util-centric scheduling.
    pub fn graph_exec_kernel_node_get_nvlink_util_centric(
        &self,
        exec: GraphId,
        node: usize,
    ) -> Result<bool, SimError> {
        self.kernel_node_nvlink_util_centric(exec, node, true)
    }

    fn kernel_node_nvlink_util_centric(
        &self,
        graph: GraphId,
        node: usize,
        exec: bool,
    ) -> Result<bool, SimError> {
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let g = self.graphs.get(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.view() } else { &g.steps };
        let step = live_ok(steps.get(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        Ok(step.nvlink_util_centric)
    }

    /// `cudaGraphKernelNodeSetAttribute` for NVLink-util-centric scheduling.
    pub fn graph_kernel_node_set_nvlink_util_centric(
        &mut self,
        graph: GraphId,
        node: usize,
        enabled: bool,
    ) -> Result<(), SimError> {
        self.set_kernel_node_nvlink_util_centric(graph, node, false, enabled)
    }

    /// `cudaGraphExecKernelNodeSetAttribute` for NVLink-util-centric scheduling.
    pub fn graph_exec_kernel_node_set_nvlink_util_centric(
        &mut self,
        exec: GraphId,
        node: usize,
        enabled: bool,
    ) -> Result<(), SimError> {
        self.set_kernel_node_nvlink_util_centric(exec, node, true, enabled)
    }

    fn set_kernel_node_nvlink_util_centric(
        &mut self,
        graph: GraphId,
        node: usize,
        exec: bool,
        enabled: bool,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node set attribute")?;
        let graph = if exec { self.as_exec(graph)? } else { graph };
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let steps = if exec { g.exec_mut()? } else { &mut g.steps };
        let step = live_ok_mut(steps.get_mut(node).ok_or(SimError::Invalid {
            why: "unknown graph node",
        })?)?;
        if !matches!(step.kind, Kind::Kernel { .. }) {
            return Err(SimError::Invalid {
                why: "not a kernel node",
            });
        }
        step.nvlink_util_centric = enabled;
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaGraphKernelNodeCopyAttributes`: copy priority, PDL, programmatic
    /// event, launch-completion event, access-policy window, mem-sync
    /// domain/map, cluster dimension, cluster scheduling policy, preferred
    /// cluster dimension, portable-cluster mode, shared-memory carveout,
    /// device-updatable kernel node, shared-memory bank mode, CUDA 13
    /// portable-shared mode, and NVLink-util-centric scheduling from `src`
    /// to `dst`.
    ///
    /// Both nodes must be kernels. Capture cannot include it.
    pub fn graph_kernel_node_copy_attributes(
        &mut self,
        dst_graph: GraphId,
        dst: usize,
        src_graph: GraphId,
        src: usize,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture kernel node copy attributes")?;
        let pri = self.graph_kernel_node_get_priority(src_graph, src)?;
        let pdl = self.graph_kernel_node_get_pdl(src_graph, src)?;
        let pde = self.graph_kernel_node_get_programmatic_event(src_graph, src)?;
        let lce = self.graph_kernel_node_get_launch_completion(src_graph, src)?;
        let apw = self.graph_kernel_node_get_access_policy(src_graph, src)?;
        self.graph_kernel_node_set_priority(dst_graph, dst, pri)?;
        self.graph_kernel_node_set_pdl(dst_graph, dst, pdl)?;
        self.graph_kernel_node_set_programmatic_event(dst_graph, dst, pde)?;
        self.graph_kernel_node_set_launch_completion(dst_graph, dst, lce)?;
        self.graph_kernel_node_set_access_policy(dst_graph, dst, apw)?;
        let domain = self.graph_kernel_node_get_mem_sync_domain(src_graph, src)?;
        let map = self.graph_kernel_node_get_mem_sync_domain_map(src_graph, src)?;
        self.graph_kernel_node_set_mem_sync_domain(dst_graph, dst, domain)?;
        self.graph_kernel_node_set_mem_sync_domain_map(dst_graph, dst, map)?;
        let pc = self.graph_kernel_node_get_portable_cluster(src_graph, src)?;
        self.graph_kernel_node_set_portable_cluster(dst_graph, dst, pc)?;
        let cluster = self.graph_kernel_node_get_cluster(src_graph, src)?;
        self.graph_kernel_node_set_cluster(dst_graph, dst, cluster)?;
        let policy = self.graph_kernel_node_get_cluster_policy(src_graph, src)?;
        self.graph_kernel_node_set_cluster_policy(dst_graph, dst, policy)?;
        let preferred = self.graph_kernel_node_get_preferred_cluster(src_graph, src)?;
        self.graph_kernel_node_set_preferred_cluster(dst_graph, dst, preferred)?;
        let carveout = self.graph_kernel_node_get_carveout(src_graph, src)?;
        self.graph_kernel_node_set_carveout(dst_graph, dst, carveout)?;
        let upd = self.graph_kernel_node_get_device_updatable(src_graph, src)?;
        self.graph_kernel_node_set_device_updatable(dst_graph, dst, upd)?;
        let sm = self.graph_kernel_node_get_shared_mem(src_graph, src)?;
        self.graph_kernel_node_set_shared_mem(dst_graph, dst, sm)?;
        let ps = self.graph_kernel_node_get_portable_shared(src_graph, src)?;
        self.graph_kernel_node_set_portable_shared(dst_graph, dst, ps)?;
        let nv = self.graph_kernel_node_get_nvlink_util_centric(src_graph, src)?;
        self.graph_kernel_node_set_nvlink_util_centric(dst_graph, dst, nv)
    }

    /// `cudaGraphKernelNodeGetAttribute` on the graph definition.
    ///
    /// Query; legal during capture. Typed getters stay. Attr/value type
    /// mismatch is Invalid `"kernel node attr"`.
    pub fn graph_kernel_node_get_attribute(
        &self,
        graph: GraphId,
        node: usize,
        attr: KernelNodeAttr,
    ) -> Result<KernelNodeAttrValue, SimError> {
        self.kernel_node_attribute(graph, node, attr, false)
    }

    /// `cudaGraphExecKernelNodeGetAttribute` on the exec snapshot.
    ///
    /// Query; legal during capture. Uninstantiated exec is Invalid.
    pub fn graph_exec_kernel_node_get_attribute(
        &self,
        exec: GraphId,
        node: usize,
        attr: KernelNodeAttr,
    ) -> Result<KernelNodeAttrValue, SimError> {
        self.kernel_node_attribute(exec, node, attr, true)
    }

    fn kernel_node_attribute(
        &self,
        graph: GraphId,
        node: usize,
        attr: KernelNodeAttr,
        exec: bool,
    ) -> Result<KernelNodeAttrValue, SimError> {
        Ok(match attr {
            KernelNodeAttr::Priority => {
                KernelNodeAttrValue::Priority(self.kernel_node_priority(graph, node, exec)?)
            }
            KernelNodeAttr::Pdl => {
                KernelNodeAttrValue::Pdl(self.kernel_node_pdl(graph, node, exec)?)
            }
            KernelNodeAttr::ProgrammaticEvent => KernelNodeAttrValue::ProgrammaticEvent(
                self.kernel_node_programmatic_event(graph, node, exec)?,
            ),
            KernelNodeAttr::LaunchCompletion => KernelNodeAttrValue::LaunchCompletion(
                self.kernel_node_launch_completion(graph, node, exec)?,
            ),
            KernelNodeAttr::AccessPolicy => KernelNodeAttrValue::AccessPolicy(
                self.kernel_node_access_policy(graph, node, exec)?,
            ),
            KernelNodeAttr::MemSyncDomain => {
                KernelNodeAttrValue::MemSyncDomain(self.kernel_node_mem_sync(graph, node, exec)?.0)
            }
            KernelNodeAttr::MemSyncDomainMap => KernelNodeAttrValue::MemSyncDomainMap(
                self.kernel_node_mem_sync(graph, node, exec)?.1,
            ),
            KernelNodeAttr::Cluster => {
                KernelNodeAttrValue::Cluster(self.kernel_node_cluster(graph, node, exec)?)
            }
            KernelNodeAttr::ClusterPolicy => KernelNodeAttrValue::ClusterPolicy(
                self.kernel_node_cluster_policy(graph, node, exec)?,
            ),
            KernelNodeAttr::PreferredCluster => KernelNodeAttrValue::PreferredCluster(
                self.kernel_node_preferred_cluster(graph, node, exec)?,
            ),
            KernelNodeAttr::Carveout => {
                KernelNodeAttrValue::Carveout(self.kernel_node_carveout(graph, node, exec)?)
            }
            KernelNodeAttr::DeviceUpdatable => KernelNodeAttrValue::DeviceUpdatable(
                self.kernel_node_device_updatable(graph, node, exec)?,
            ),
            KernelNodeAttr::SharedMem => {
                KernelNodeAttrValue::SharedMem(self.kernel_node_shared_mem(graph, node, exec)?)
            }
            KernelNodeAttr::PortableCluster => KernelNodeAttrValue::PortableCluster(
                self.kernel_node_portable_cluster(graph, node, exec)?,
            ),
            KernelNodeAttr::PortableShared => KernelNodeAttrValue::PortableShared(
                self.kernel_node_portable_shared(graph, node, exec)?,
            ),
            KernelNodeAttr::DynamicShared => KernelNodeAttrValue::DynamicShared(
                self.kernel_node_dynamic_shared(graph, node, exec)?,
            ),
            KernelNodeAttr::NvlinkUtilCentric => KernelNodeAttrValue::NvlinkUtilCentric(
                self.kernel_node_nvlink_util_centric(graph, node, exec)?,
            ),
        })
    }

    /// `cudaGraphKernelNodeSetAttribute` on the graph definition.
    ///
    /// Dispatches to the typed setters. After instantiate this does not
    /// retarget the exec; use [`Self::graph_exec_kernel_node_set_attribute`].
    /// Capture cannot include it. Attr/value type mismatch is Invalid
    /// `"kernel node attr"`.
    pub fn graph_kernel_node_set_attribute(
        &mut self,
        graph: GraphId,
        node: usize,
        attr: KernelNodeAttr,
        value: KernelNodeAttrValue,
    ) -> Result<(), SimError> {
        self.set_kernel_node_attribute(graph, node, attr, value, false)
    }

    /// `cudaGraphExecKernelNodeSetAttribute` on the exec snapshot.
    pub fn graph_exec_kernel_node_set_attribute(
        &mut self,
        exec: GraphId,
        node: usize,
        attr: KernelNodeAttr,
        value: KernelNodeAttrValue,
    ) -> Result<(), SimError> {
        self.set_kernel_node_attribute(exec, node, attr, value, true)
    }

    fn set_kernel_node_attribute(
        &mut self,
        graph: GraphId,
        node: usize,
        attr: KernelNodeAttr,
        value: KernelNodeAttrValue,
        exec: bool,
    ) -> Result<(), SimError> {
        match (attr, value) {
            (KernelNodeAttr::Priority, KernelNodeAttrValue::Priority(p)) => {
                if exec {
                    self.graph_exec_kernel_node_set_priority(graph, node, p)
                } else {
                    self.graph_kernel_node_set_priority(graph, node, p)
                }
            }
            (KernelNodeAttr::Pdl, KernelNodeAttrValue::Pdl(pdl)) => {
                if exec {
                    self.graph_exec_kernel_node_set_pdl(graph, node, pdl)
                } else {
                    self.graph_kernel_node_set_pdl(graph, node, pdl)
                }
            }
            (KernelNodeAttr::ProgrammaticEvent, KernelNodeAttrValue::ProgrammaticEvent(ev)) => {
                if exec {
                    self.graph_exec_kernel_node_set_programmatic_event(graph, node, ev)
                } else {
                    self.graph_kernel_node_set_programmatic_event(graph, node, ev)
                }
            }
            (KernelNodeAttr::LaunchCompletion, KernelNodeAttrValue::LaunchCompletion(ev)) => {
                if exec {
                    self.graph_exec_kernel_node_set_launch_completion(graph, node, ev)
                } else {
                    self.graph_kernel_node_set_launch_completion(graph, node, ev)
                }
            }
            (KernelNodeAttr::AccessPolicy, KernelNodeAttrValue::AccessPolicy(w)) => {
                if exec {
                    self.graph_exec_kernel_node_set_access_policy(graph, node, w)
                } else {
                    self.graph_kernel_node_set_access_policy(graph, node, w)
                }
            }
            (KernelNodeAttr::MemSyncDomain, KernelNodeAttrValue::MemSyncDomain(d)) => {
                if exec {
                    self.graph_exec_kernel_node_set_mem_sync_domain(graph, node, d)
                } else {
                    self.graph_kernel_node_set_mem_sync_domain(graph, node, d)
                }
            }
            (KernelNodeAttr::MemSyncDomainMap, KernelNodeAttrValue::MemSyncDomainMap(m)) => {
                if exec {
                    self.graph_exec_kernel_node_set_mem_sync_domain_map(graph, node, m)
                } else {
                    self.graph_kernel_node_set_mem_sync_domain_map(graph, node, m)
                }
            }
            (KernelNodeAttr::Cluster, KernelNodeAttrValue::Cluster(c)) => {
                if exec {
                    self.graph_exec_kernel_node_set_cluster(graph, node, c)
                } else {
                    self.graph_kernel_node_set_cluster(graph, node, c)
                }
            }
            (KernelNodeAttr::ClusterPolicy, KernelNodeAttrValue::ClusterPolicy(p)) => {
                if exec {
                    self.graph_exec_kernel_node_set_cluster_policy(graph, node, p)
                } else {
                    self.graph_kernel_node_set_cluster_policy(graph, node, p)
                }
            }
            (KernelNodeAttr::PreferredCluster, KernelNodeAttrValue::PreferredCluster(c)) => {
                if exec {
                    self.graph_exec_kernel_node_set_preferred_cluster(graph, node, c)
                } else {
                    self.graph_kernel_node_set_preferred_cluster(graph, node, c)
                }
            }
            (KernelNodeAttr::Carveout, KernelNodeAttrValue::Carveout(c)) => {
                if exec {
                    self.graph_exec_kernel_node_set_carveout(graph, node, c)
                } else {
                    self.graph_kernel_node_set_carveout(graph, node, c)
                }
            }
            (KernelNodeAttr::DeviceUpdatable, KernelNodeAttrValue::DeviceUpdatable(yes)) => {
                if exec {
                    self.graph_exec_kernel_node_set_device_updatable(graph, node, yes)
                } else {
                    self.graph_kernel_node_set_device_updatable(graph, node, yes)
                }
            }
            (KernelNodeAttr::SharedMem, KernelNodeAttrValue::SharedMem(m)) => {
                if exec {
                    self.graph_exec_kernel_node_set_shared_mem(graph, node, m)
                } else {
                    self.graph_kernel_node_set_shared_mem(graph, node, m)
                }
            }
            (KernelNodeAttr::PortableCluster, KernelNodeAttrValue::PortableCluster(m)) => {
                if exec {
                    self.graph_exec_kernel_node_set_portable_cluster(graph, node, m)
                } else {
                    self.graph_kernel_node_set_portable_cluster(graph, node, m)
                }
            }
            (KernelNodeAttr::PortableShared, KernelNodeAttrValue::PortableShared(m)) => {
                if exec {
                    self.graph_exec_kernel_node_set_portable_shared(graph, node, m)
                } else {
                    self.graph_kernel_node_set_portable_shared(graph, node, m)
                }
            }
            (KernelNodeAttr::DynamicShared, KernelNodeAttrValue::DynamicShared(n)) => {
                if exec {
                    self.graph_exec_kernel_node_set_dynamic_shared(graph, node, n)
                } else {
                    self.graph_kernel_node_set_dynamic_shared(graph, node, n)
                }
            }
            (KernelNodeAttr::NvlinkUtilCentric, KernelNodeAttrValue::NvlinkUtilCentric(yes)) => {
                if exec {
                    self.graph_exec_kernel_node_set_nvlink_util_centric(graph, node, yes)
                } else {
                    self.graph_kernel_node_set_nvlink_util_centric(graph, node, yes)
                }
            }
            _ => Err(SimError::Invalid {
                why: "kernel node attr",
            }),
        }
    }

    /// `cudaGraphNodeFindInClone`: index in `cloned` of the node that was `node`
    /// on `original`.
    ///
    /// `cloned` must have been produced by [`Self::clone_graph`] of `original`
    /// (a nested graph cloned in that same call counts). Capture is allowed.
    /// Add order is preserved, so the index is unchanged. A second clone of the
    /// clone does not map nodes from the first original.
    pub fn graph_node_find_in_clone(
        &self,
        original: GraphId,
        node: usize,
        cloned: GraphId,
    ) -> Result<usize, SimError> {
        if !self.graphs.contains_key(&original) || !self.graphs.contains_key(&cloned) {
            return Err(SimError::Invalid {
                why: "unknown graph",
            });
        }
        if original == cloned {
            return Err(SimError::Invalid { why: "not a clone" });
        }
        let src = self
            .clone_of
            .get(&cloned)
            .copied()
            .ok_or(SimError::Invalid { why: "not a clone" })?;
        if src != original {
            return Err(SimError::Invalid { why: "not a clone" });
        }
        let orig = self.graphs.get(&original).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        let clone = self.graphs.get(&cloned).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        if orig.steps.get(node).is_none_or(|s| s.destroyed)
            || clone.steps.get(node).is_none_or(|s| s.destroyed)
        {
            return Err(SimError::Invalid {
                why: "unknown graph node",
            });
        }
        Ok(node)
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
        let priority = self.snap_priority(device, stream);
        let (mem_sync_domain, mem_sync_map) = self.snap_mem_sync(device, stream, &kind)?;
        let g = self.graphs.get_mut(&graph).ok_or(SimError::Invalid {
            why: "unknown graph",
        })?;
        g.steps.push(GraphStep {
            device,
            stream,
            kind,
            deps: Vec::new(),
            enabled: true,
            destroyed: false,
            priority,
            pdl: self.enqueue_pdl,
            programmatic_event: self.enqueue_programmatic_event,
            launch_completion: self.enqueue_launch_completion,
            access_policy: self.enqueue_access_policy,
            mem_sync_domain,
            mem_sync_map,
            cluster: self.enqueue_cluster,
            cluster_policy: self.enqueue_cluster_policy,
            preferred_cluster: self.enqueue_preferred_cluster,
            carveout: self.enqueue_carveout,
            device_updatable: self.enqueue_device_updatable,
            shared_mem: self.enqueue_shared_mem,
            portable_cluster: self.enqueue_portable_cluster,
            dynamic_shared: self.enqueue_dynamic_shared,
            portable_shared: self.enqueue_portable_shared,
            nvlink_util_centric: self.enqueue_nvlink_util_centric,
        });
        Ok(())
    }

    /// Stream-ordered allocation (`cudaMallocAsync`) from the device default pool.
    ///
    /// Capacity is reserved when the op starts. The pointer is not usable until
    /// this stream catches up. Capture records a graph mem alloc node
    /// (`cudaMallocAsync` during stream capture) and draws from the device
    /// graph-memory pool, not the current default mempool. [`Self::malloc`] is
    /// host-synchronous `cudaMalloc` and cannot be captured.
    pub fn alloc(
        &mut self,
        device: DeviceId,
        bytes: u64,
        stream: StreamId,
    ) -> Result<AllocId, SimError> {
        let pool = if self.in_capture(device, stream) {
            self.graph_pool(device)?
        } else {
            self.device_mempool(device)?
        };
        self.alloc_from_pool_inner(device, pool, bytes, stream)
    }

    /// `cudaDeviceGetDefaultMemPool`. Query; legal during capture.
    ///
    /// Seeded at construct. [`Self::set_device_mempool`] does not replace it.
    pub fn default_pool(&self, device: DeviceId) -> Result<PoolId, SimError> {
        let _gpu = self.profile.gpu(device)?;
        self.default_pools
            .get(&device)
            .copied()
            .ok_or(SimError::Invalid {
                why: "default pool missing",
            })
    }

    /// `cudaDeviceGetMemPool`. Query; legal during capture.
    ///
    /// [`Self::alloc`] (`cudaMallocAsync`) draws from this. Starts as
    /// [`Self::default_pool`]; [`Self::set_device_mempool`] rebinds it.
    pub fn device_mempool(&self, device: DeviceId) -> Result<PoolId, SimError> {
        let _gpu = self.profile.gpu(device)?;
        self.current_pools
            .get(&device)
            .copied()
            .ok_or(SimError::Invalid {
                why: "device mempool missing",
            })
    }

    /// Device graph-memory pool (`cudaDeviceGetGraphMemAttribute` backing).
    ///
    /// Not the default mempool. [`Self::alloc_from_pool`] / [`Self::set_device_mempool`]
    /// / [`Self::set_pool_release_threshold`] / [`Self::pool_set_access`] /
    /// [`Self::pool_get_attribute`] / [`Self::pool_set_attribute`] /
    /// [`Self::pool_get_access`] / [`Self::destroy_pool`] refuse it.
    pub fn graph_pool(&self, device: DeviceId) -> Result<PoolId, SimError> {
        let _gpu = self.profile.gpu(device)?;
        self.graph_pools
            .get(&device)
            .copied()
            .ok_or(SimError::Invalid {
                why: "graph mem pool missing",
            })
    }

    fn refuse_graph_pool(&self, pool: PoolId) -> Result<(), SimError> {
        if self.pool_ref(self.pool_root(pool)?)?.graph {
            return Err(SimError::Invalid {
                why: "graph mem pool",
            });
        }
        Ok(())
    }

    fn refuse_destroyed_pool(&self, pool: PoolId) -> Result<(), SimError> {
        if self.pool_ref(pool)?.destroyed || self.pool_ref(self.pool_root(pool)?)?.destroyed {
            return Err(SimError::Invalid {
                why: "destroyed pool",
            });
        }
        Ok(())
    }

    fn rebind_current_pools(&mut self, pool: PoolId) {
        let rebound: Vec<DeviceId> = self
            .current_pools
            .iter()
            .filter(|(_, p)| **p == pool)
            .map(|(d, _)| *d)
            .collect();
        for d in rebound {
            if let Some(&def) = self.default_pools.get(&d) {
                let _prev = self.current_pools.insert(d, def);
            }
        }
    }

    /// `cudaDeviceSetMemPool`. Later [`Self::alloc`] draws from `pool`.
    ///
    /// Capture cannot include it. `pool` must belong to `device` (an imported
    /// sibling is legal). Does not change live/cached bytes. The graph-memory
    /// pool is not a valid device mempool. [`Self::default_pool`] stays the
    /// seeded default.
    pub fn set_device_mempool(&mut self, device: DeviceId, pool: PoolId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture mempool")?;
        self.refuse_graph_pool(pool)?;
        self.refuse_destroyed_pool(pool)?;
        let _gpu = self.profile.gpu(device)?;
        if self.pool_ref(pool)?.device != device {
            return Err(SimError::Invalid {
                why: "pool device mismatch",
            });
        }
        let _prev = self.current_pools.insert(device, pool);
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

    /// `cudaMemPoolDestroy`. Host-synchronous. Capture cannot include it.
    ///
    /// Returns immediately. Unused cached bytes return to the OS. Outstanding
    /// allocations stay valid until freed (later frees do not re-cache).
    /// Destroying the current device mempool rebinds [`Self::device_mempool`]
    /// to [`Self::default_pool`]. The default and graph-memory pools cannot be
    /// destroyed. A destroyed handle is Invalid for alloc/export/get/set.
    pub fn destroy_pool(&mut self, pool: PoolId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture mempool")?;
        self.refuse_graph_pool(pool)?;
        if self.pool_ref(pool)?.destroyed {
            return Err(SimError::Invalid {
                why: "destroyed pool",
            });
        }
        if self.default_pools.values().any(|&p| p == pool) {
            return Err(SimError::Invalid {
                why: "default pool",
            });
        }
        self.rebind_current_pools(pool);
        if self.pool_ref(pool)?.share_root.is_some() {
            self.pool_mut(pool)?.destroyed = true;
            self.clock = self.clock.saturating_add(self.first_alloc_ns().max(1));
            return Ok(());
        }
        let _dropped = self.pool_trim_to(pool, 0)?;
        {
            let p = self.pool_mut(pool)?;
            p.release_threshold = 0;
            p.destroyed = true;
        }
        self.share_handles.retain(|_, src| *src != pool);
        self.clock = self.clock.saturating_add(self.first_alloc_ns().max(1));
        Ok(())
    }

    /// `cudaMallocFromPoolAsync`. `pool` must belong to `device`.
    ///
    /// The graph-memory pool is not a user mempool; use [`Self::alloc`] during
    /// capture or [`Self::graph_add_alloc`].
    pub fn alloc_from_pool(
        &mut self,
        device: DeviceId,
        pool: PoolId,
        bytes: u64,
        stream: StreamId,
    ) -> Result<AllocId, SimError> {
        self.refuse_graph_pool(pool)?;
        self.refuse_destroyed_pool(pool)?;
        self.alloc_from_pool_inner(device, pool, bytes, stream)
    }

    fn alloc_from_pool_inner(
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
    /// them used until [`Self::pool_trim_to`]. Also
    /// [`Self::pool_set_attribute`] [`MemPoolAttr::ReleaseThreshold`].
    pub fn set_pool_release_threshold(&mut self, pool: PoolId, bytes: u64) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture mempool")?;
        let root = self.pool_root(pool)?;
        self.refuse_graph_pool(root)?;
        self.refuse_destroyed_pool(root)?;
        self.pool_mut(root)?.release_threshold = bytes;
        Ok(())
    }

    /// `cudaMemPoolGetAttribute`. Query; legal during capture.
    ///
    /// [`MemPoolAttr::UsedMemCurrent`] is [`Self::pool_live`]. Reserved is live
    /// plus [`Self::pool_cached`]. An imported pool reports the exporter.
    /// The graph-memory pool is Invalid (use [`Self::graph_mem_get`]).
    pub fn pool_get_attribute(&self, pool: PoolId, attr: MemPoolAttr) -> Result<u64, SimError> {
        self.refuse_graph_pool(pool)?;
        self.refuse_destroyed_pool(pool)?;
        match attr {
            MemPoolAttr::ReleaseThreshold => {
                Ok(self.pool_ref(self.pool_root(pool)?)?.release_threshold)
            }
            MemPoolAttr::UsedMemCurrent => self.pool_live(pool),
            MemPoolAttr::ReservedMemCurrent => Ok(self
                .pool_live(pool)?
                .saturating_add(self.pool_cached(pool)?)),
        }
    }

    /// `cudaMemPoolSetAttribute`. Host-synchronous. Capture cannot include it.
    ///
    /// Only [`MemPoolAttr::ReleaseThreshold`] is writable (same as
    /// [`Self::set_pool_release_threshold`]). Used/Reserved are read-only.
    /// The graph-memory pool is Invalid (use [`Self::graph_mem_set`]).
    pub fn pool_set_attribute(
        &mut self,
        pool: PoolId,
        attr: MemPoolAttr,
        value: u64,
    ) -> Result<(), SimError> {
        match attr {
            MemPoolAttr::ReleaseThreshold => self.set_pool_release_threshold(pool, value),
            MemPoolAttr::UsedMemCurrent | MemPoolAttr::ReservedMemCurrent => {
                self.fail_if_capturing("cannot capture mempool")?;
                self.refuse_graph_pool(pool)?;
                self.refuse_destroyed_pool(pool)?;
                Err(SimError::Invalid {
                    why: "read-only pool attr",
                })
            }
        }
    }

    /// Set every device's current mempool release threshold (`cudaDeviceGetMemPool`).
    pub fn set_default_pool_release_threshold(&mut self, bytes: u64) -> Result<(), SimError> {
        let ids: Vec<PoolId> = self.current_pools.values().copied().collect();
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
        self.refuse_destroyed_pool(root)?;
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
        self.refuse_graph_pool(pool)?;
        self.refuse_destroyed_pool(pool)?;
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
        self.refuse_destroyed_pool(pool)?;
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

    /// `cudaMemPoolGetAccess`. Query; legal during capture.
    ///
    /// [`MemAccessFlags::PROT_READ_WRITE`] (`3`) on the owning device (default
    /// accessibility) and on peers after [`Self::pool_set_access`]. Otherwise
    /// [`MemAccessFlags::PROT_NONE`] (`0`). This VM does not model ProtRead.
    /// The graph-memory pool is Invalid.
    pub fn pool_get_access(&self, pool: PoolId, device: DeviceId) -> Result<u32, SimError> {
        let _gpu = self.profile.gpu(device)?;
        self.refuse_graph_pool(pool)?;
        self.refuse_destroyed_pool(pool)?;
        let owner = self.pool_ref(pool)?.device;
        if owner == device || self.is_pool_accessed_by(pool, device)? {
            Ok(MemAccessFlags::PROT_READ_WRITE)
        } else {
            Ok(MemAccessFlags::PROT_NONE)
        }
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
        self.refuse_destroyed_pool(pool)?;
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
        self.refuse_destroyed_pool(root)?;
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
        self.alloc_host_with_flags(bytes, HostAllocFlags::DEFAULT)
    }

    /// Pageable host allocation (`malloc`). Pin it with [`Self::host_register`].
    pub fn alloc_host(&mut self, bytes: u64) -> Result<AllocId, SimError> {
        self.insert_host(bytes, true, false, false, false)
    }

    /// `cudaHostAllocMapped`: pinned, mapped, no HBM. A kernel may read it
    /// immediately; the memory term is host PCIe, not HBM.
    pub fn alloc_host_mapped(&mut self, bytes: u64) -> Result<AllocId, SimError> {
        self.alloc_host_with_flags(bytes, HostAllocFlags::MAPPED)
    }

    /// `cudaHostAlloc` with [`HostAllocFlags`].
    ///
    /// Known bits: [`HostAllocFlags::MAPPED`]. Portable / WriteCombined are
    /// Invalid `"host alloc flags"`. Typed helpers stay. Capture cannot include
    /// host alloc.
    pub fn alloc_host_with_flags(&mut self, bytes: u64, flags: u32) -> Result<AllocId, SimError> {
        const KNOWN: u32 = HostAllocFlags::MAPPED;
        if flags & !KNOWN != 0 {
            return Err(SimError::Invalid {
                why: "host alloc flags",
            });
        }
        let mapped = flags & HostAllocFlags::MAPPED != 0;
        self.insert_host(bytes, false, true, mapped, false)
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

    /// `cudaMemRangeGetAttribute`. Query; legal during capture.
    ///
    /// This VM tracks advice per live managed allocation, not per byte range.
    /// Non-managed pointers are Invalid `"not managed"`. Last-prefetch location
    /// is not modeled.
    pub fn mem_range_get_attribute(
        &self,
        alloc: AllocId,
        attr: MemRangeAttr,
    ) -> Result<MemRangeAttrValue, SimError> {
        let a = self.alloc_ref(alloc)?;
        if !a.live || !a.managed {
            return Err(SimError::Invalid { why: "not managed" });
        }
        Ok(match attr {
            MemRangeAttr::ReadMostly => MemRangeAttrValue::ReadMostly(a.read_mostly),
            MemRangeAttr::PreferredLocation => {
                let loc = match a.preferred {
                    Preferred::None => None,
                    Preferred::Host => Some(Place::Host),
                    Preferred::Gpu(d) => Some(Place::Device(d)),
                };
                MemRangeAttrValue::PreferredLocation(loc)
            }
            MemRangeAttr::AccessedBy => {
                MemRangeAttrValue::AccessedBy(a.accessed_by.iter().copied().collect())
            }
        })
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
                ..MemcpyOp::default()
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
                ..MemcpyOp::default()
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
        self.host_register_with_flags(id, HostAllocFlags::DEFAULT)
    }

    /// `cudaHostRegisterMapped`: pin and map pageable host. Kernels may read it
    /// over PCIe without a device copy.
    pub fn host_register_mapped(&mut self, id: AllocId) -> Result<(), SimError> {
        self.host_register_with_flags(id, HostAllocFlags::MAPPED)
    }

    /// `cudaHostRegister` with [`HostAllocFlags`].
    ///
    /// Known bits: [`HostAllocFlags::MAPPED`]. Portable / IoMemory / ReadOnly
    /// are Invalid `"host register flags"`. Typed helpers stay.
    pub fn host_register_with_flags(&mut self, id: AllocId, flags: u32) -> Result<(), SimError> {
        const KNOWN: u32 = HostAllocFlags::MAPPED;
        if flags & !KNOWN != 0 {
            return Err(SimError::Invalid {
                why: "host register flags",
            });
        }
        self.host_register_flags(id, flags & HostAllocFlags::MAPPED != 0)
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
        {
            let a = self.alloc_mut(id)?;
            a.live = false;
            a.host_pageable = false;
        }
        self.clear_mailbox(id);
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
        {
            let a = self.alloc_mut(id)?;
            a.live = false;
            a.host_pinned = false;
            a.host_mapped = false;
            a.host_pageable = false;
        }
        self.clear_mailbox(id);
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

    /// `cudaIpcGetEventHandle`. Host-synchronous. Capture cannot include it.
    ///
    /// The event must be [`EventCreateFlags::INTERPROCESS`] (and therefore
    /// disable-timing). The same event returns the same handle. An
    /// [`Self::ipc_open_event`] alias cannot export.
    pub fn ipc_get_event(&mut self, event: EventId) -> Result<IpcEventHandleId, SimError> {
        self.fail_if_capturing("cannot capture ipc")?;
        let ev = self
            .events
            .get(&event)
            .ok_or(SimError::UnknownEvent { event: event.0 })?;
        if ev.ipc_src.is_some() {
            return Err(SimError::Invalid {
                why: "ipc event import",
            });
        }
        if !ev.interprocess {
            return Err(SimError::Invalid {
                why: "not interprocess",
            });
        }
        if let Some(h) = self
            .ipc_event_handles
            .iter()
            .find_map(|(&h, &src)| (src == event).then_some(h))
        {
            return Ok(h);
        }
        self.clock = self.clock.saturating_add(1);
        let h = IpcEventHandleId(self.next_ipc_event);
        self.next_ipc_event = self.next_ipc_event.saturating_add(1);
        let _prev = self.ipc_event_handles.insert(h, event);
        Ok(h)
    }

    /// `cudaIpcOpenEventHandle`. Alias shares the source record (no extra event).
    /// Capture cannot include it. The returned id is simulator-chosen.
    pub fn ipc_open_event(&mut self, handle: IpcEventHandleId) -> Result<EventId, SimError> {
        self.fail_if_capturing("cannot capture ipc")?;
        let src = *self
            .ipc_event_handles
            .get(&handle)
            .ok_or(SimError::Invalid {
                why: "unknown ipc event",
            })?;
        {
            let ev = self
                .events
                .get(&src)
                .ok_or(SimError::UnknownEvent { event: src.0 })?;
            if ev.ipc_src.is_some() {
                return Err(SimError::Invalid {
                    why: "ipc event import",
                });
            }
            if !ev.interprocess {
                return Err(SimError::Invalid {
                    why: "not interprocess",
                });
            }
        }
        self.clock = self.clock.saturating_add(1);
        let id = self.alloc_imported_event()?;
        if let Some(ev) = self.events.get_mut(&src) {
            ev.ipc_opens = ev.ipc_opens.saturating_add(1);
        }
        let mut ev = Ev::new(false);
        ev.interprocess = true;
        ev.ipc_src = Some(src);
        let _prev = self.events.insert(id, ev);
        Ok(id)
    }

    /// Whether `event` is a live [`Self::ipc_open_event`] alias.
    pub fn is_ipc_event_import(&self, event: EventId) -> Result<bool, SimError> {
        let ev = self
            .events
            .get(&event)
            .ok_or(SimError::UnknownEvent { event: event.0 })?;
        Ok(ev.ipc_src.is_some())
    }

    fn alloc_imported_event(&mut self) -> Result<EventId, SimError> {
        let start = if self.next_imported_event == 0 {
            1
        } else {
            self.next_imported_event
        };
        let mut n = start;
        loop {
            let id = EventId(n);
            if !self.events.contains_key(&id) {
                self.next_imported_event = n.checked_add(1).unwrap_or(1);
                return Ok(id);
            }
            n = n.checked_add(1).unwrap_or(1);
            if n == start {
                return Err(SimError::Invalid {
                    why: "event id space",
                });
            }
        }
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
        {
            let a = self.alloc_mut(id)?;
            a.devices.clear();
            a.live = false;
        }
        self.clear_mailbox(id);
        Ok(())
    }

    /// Asynchronous copy (`cudaMemcpyAsync`) when both ends are device or pinned.
    ///
    /// Pageable host (`Place::Host`) is host-synchronous: the driver bounces
    /// through pinned staging, so this call waits [`Self::synchronize_stream`]
    /// before returning. Capture cannot include a pageable copy. Pinned DMA
    /// ([`Self::memcpy_pinned_to_device`]) stays stream-ordered.
    /// [`MemcpyOp::height`] `> 1` is `cudaMemcpy2DAsync`: billed bytes are
    /// `width * height`, not pitch padding. [`MemcpyOp::depth`] `> 1` is
    /// `cudaMemcpy3DAsync`: billed bytes are `width * height * depth`, not
    /// row or slice padding.
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
        memcpy_2d_check(&op)?;
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
                ..MemcpyOp::default()
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
                ..MemcpyOp::default()
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
                ..MemcpyOp::default()
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
                ..MemcpyOp::default()
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
                ..MemcpyOp::default()
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
        let prev = self.enqueue_nvlink_util_centric;
        self.enqueue_nvlink_util_centric = self.stream_nvlink_util_centric(device, stream);
        let out = self.submit_kernel(device, kind, reads, writes, stream, false);
        self.enqueue_nvlink_util_centric = prev;
        out
    }

    /// Same as [`Self::kernel`] with [`ProgrammaticLaunch`] (CUDA PDL).
    ///
    /// [`ProgrammaticLaunch::wait`] lets this kernel start after the previous
    /// same-stream kernel's programmatic trigger instead of its completion.
    /// [`ProgrammaticLaunch::trigger`] fires that trigger at
    /// [`crate::GpuProfile::pdl_trigger_permille`]. Overlap needs
    /// [`crate::GpuProfile::compute_slots`] `>= 2`. Capture records the flags.
    /// Decode identity stays [`Self::kernel`] (both flags false).
    pub fn kernel_pdl(
        &mut self,
        device: DeviceId,
        kind: KernelKind,
        reads: &[AllocId],
        writes: &[AllocId],
        stream: StreamId,
        pdl: ProgrammaticLaunch,
    ) -> Result<OpId, SimError> {
        let reads: Vec<KernelBuf> = reads.iter().copied().map(KernelBuf::whole).collect();
        let writes: Vec<KernelBuf> = writes.iter().copied().map(KernelBuf::whole).collect();
        self.kernel_pdl_bufs(device, kind, &reads, &writes, stream, pdl)
    }

    /// [`Self::kernel_bufs`] with [`ProgrammaticLaunch`].
    pub fn kernel_pdl_bufs(
        &mut self,
        device: DeviceId,
        kind: KernelKind,
        reads: &[KernelBuf],
        writes: &[KernelBuf],
        stream: StreamId,
        pdl: ProgrammaticLaunch,
    ) -> Result<OpId, SimError> {
        let prev = self.enqueue_pdl;
        self.enqueue_pdl = pdl;
        let out = self.submit_kernel(device, kind, reads, writes, stream, false);
        self.enqueue_pdl = prev;
        out
    }

    /// [`Self::kernel_pdl`] plus [`ProgrammaticEvent`] (`cudaLaunchAttributeProgrammaticEvent`).
    ///
    /// Other streams may [`Self::wait_event`] the event after the PDL trigger
    /// when [`ProgrammaticLaunch::trigger`] is set. Without trigger the event
    /// records at kernel completion. Decode identity stays [`Self::kernel`].
    pub fn kernel_pdl_event(
        &mut self,
        device: DeviceId,
        kind: KernelKind,
        reads: &[AllocId],
        writes: &[AllocId],
        stream: StreamId,
        launch: PdlLaunch,
    ) -> Result<OpId, SimError> {
        let reads: Vec<KernelBuf> = reads.iter().copied().map(KernelBuf::whole).collect();
        let writes: Vec<KernelBuf> = writes.iter().copied().map(KernelBuf::whole).collect();
        self.kernel_pdl_event_bufs(device, kind, &reads, &writes, stream, launch)
    }

    /// [`Self::kernel_pdl_bufs`] plus [`ProgrammaticEvent`].
    pub fn kernel_pdl_event_bufs(
        &mut self,
        device: DeviceId,
        kind: KernelKind,
        reads: &[KernelBuf],
        writes: &[KernelBuf],
        stream: StreamId,
        launch: PdlLaunch,
    ) -> Result<OpId, SimError> {
        let prev_pdl = self.enqueue_pdl;
        let prev_ev = self.enqueue_programmatic_event;
        self.enqueue_pdl = launch.pdl;
        self.enqueue_programmatic_event = launch.event;
        let out = self.submit_kernel(device, kind, reads, writes, stream, false);
        self.enqueue_pdl = prev_pdl;
        self.enqueue_programmatic_event = prev_ev;
        out
    }

    /// [`Self::kernel`] plus [`LaunchCompletionEvent`] (`cudaLaunchAttributeLaunchCompletionEvent`).
    ///
    /// Other streams may [`Self::wait_event`] the event when this kernel
    /// *starts*, not when it finishes. Decode identity stays [`Self::kernel`].
    pub fn kernel_launch_completion(
        &mut self,
        device: DeviceId,
        kind: KernelKind,
        reads: &[AllocId],
        writes: &[AllocId],
        stream: StreamId,
        event: LaunchCompletionEvent,
    ) -> Result<OpId, SimError> {
        let reads: Vec<KernelBuf> = reads.iter().copied().map(KernelBuf::whole).collect();
        let writes: Vec<KernelBuf> = writes.iter().copied().map(KernelBuf::whole).collect();
        self.kernel_launch_completion_bufs(device, kind, &reads, &writes, stream, event)
    }

    /// [`Self::kernel_bufs`] plus [`LaunchCompletionEvent`].
    pub fn kernel_launch_completion_bufs(
        &mut self,
        device: DeviceId,
        kind: KernelKind,
        reads: &[KernelBuf],
        writes: &[KernelBuf],
        stream: StreamId,
        event: LaunchCompletionEvent,
    ) -> Result<OpId, SimError> {
        let prev = self.enqueue_launch_completion;
        self.enqueue_launch_completion = Some(event);
        let out = self.submit_kernel(device, kind, reads, writes, stream, false);
        self.enqueue_launch_completion = prev;
        out
    }

    /// [`Self::kernel`] plus packed [`KernelAttrs`] (`cudaLaunchKernelEx`).
    ///
    /// Combines cooperative, PDL, an access-policy window, mem-sync
    /// domain/map, cluster, shared-memory carveout, device-updatable kernel
    /// node, shared-memory bank mode, and launch-attribute priority on one
    /// submit. Decode identity stays [`Self::kernel`] ([`KernelAttrs::default`]).
    pub fn kernel_with(
        &mut self,
        device: DeviceId,
        kind: KernelKind,
        reads: &[AllocId],
        writes: &[AllocId],
        stream: StreamId,
        attrs: KernelAttrs,
    ) -> Result<OpId, SimError> {
        let reads: Vec<KernelBuf> = reads.iter().copied().map(KernelBuf::whole).collect();
        let writes: Vec<KernelBuf> = writes.iter().copied().map(KernelBuf::whole).collect();
        self.kernel_bufs_with(device, kind, &reads, &writes, stream, attrs)
    }

    /// [`Self::kernel_bufs`] plus packed [`KernelAttrs`].
    pub fn kernel_bufs_with(
        &mut self,
        device: DeviceId,
        kind: KernelKind,
        reads: &[KernelBuf],
        writes: &[KernelBuf],
        stream: StreamId,
        attrs: KernelAttrs,
    ) -> Result<OpId, SimError> {
        if let Some(w) = attrs.access_policy {
            self.validate_access_policy_window(device, w)?;
        }
        if let Some(map) = attrs.mem_sync_map {
            self.validate_mem_sync_map(device, map)?;
        }
        self.validate_cluster_attrs(
            device,
            attrs.cluster,
            attrs.preferred_cluster,
            attrs.portable_cluster,
        )?;
        self.validate_dynamic_shared(device, attrs.dynamic_shared, attrs.portable_shared)?;
        if attrs.cooperative {
            self.require_cooperative(device)?;
        }
        if attrs.device_updatable && self.capturing.is_none() {
            return Err(SimError::Invalid {
                why: "device-updatable is graphs-only",
            });
        }
        let prev_pdl = self.enqueue_pdl;
        let prev_win = self.enqueue_access_policy;
        let prev_dom = self.enqueue_mem_sync_domain;
        let prev_map = self.enqueue_mem_sync_map;
        let prev_cl = self.enqueue_cluster;
        let prev_pol = self.enqueue_cluster_policy;
        let prev_pref = self.enqueue_preferred_cluster;
        let prev_carve = self.enqueue_carveout;
        let prev_upd = self.enqueue_device_updatable;
        let prev_sm = self.enqueue_shared_mem;
        let prev_pc = self.enqueue_portable_cluster;
        let prev_ds = self.enqueue_dynamic_shared;
        let prev_ps = self.enqueue_portable_shared;
        let prev_nv = self.enqueue_nvlink_util_centric;
        let prev_pri = self.enqueue_priority;
        self.enqueue_pdl = attrs.pdl;
        self.enqueue_access_policy = attrs.access_policy;
        self.enqueue_mem_sync_domain = attrs.mem_sync_domain;
        self.enqueue_mem_sync_map = attrs.mem_sync_map;
        self.enqueue_cluster = attrs.cluster;
        self.enqueue_cluster_policy = attrs.cluster_policy;
        self.enqueue_preferred_cluster = attrs.preferred_cluster;
        self.enqueue_carveout = attrs.carveout;
        self.enqueue_device_updatable = attrs.device_updatable;
        self.enqueue_shared_mem = attrs.shared_mem;
        self.enqueue_portable_cluster = attrs.portable_cluster;
        self.enqueue_dynamic_shared = attrs.dynamic_shared;
        self.enqueue_portable_shared = attrs.portable_shared;
        self.enqueue_nvlink_util_centric = attrs.nvlink_util_centric;
        self.enqueue_priority = attrs.priority;
        let out = self.submit_kernel(device, kind, reads, writes, stream, attrs.cooperative);
        self.enqueue_pdl = prev_pdl;
        self.enqueue_access_policy = prev_win;
        self.enqueue_mem_sync_domain = prev_dom;
        self.enqueue_mem_sync_map = prev_map;
        self.enqueue_cluster = prev_cl;
        self.enqueue_cluster_policy = prev_pol;
        self.enqueue_preferred_cluster = prev_pref;
        self.enqueue_carveout = prev_carve;
        self.enqueue_device_updatable = prev_upd;
        self.enqueue_shared_mem = prev_sm;
        self.enqueue_portable_cluster = prev_pc;
        self.enqueue_dynamic_shared = prev_ds;
        self.enqueue_portable_shared = prev_ps;
        self.enqueue_nvlink_util_centric = prev_nv;
        self.enqueue_priority = prev_pri;
        out
    }

    /// [`Self::kernel`] plus [`AccessPolicyWindow`] (`cudaLaunchAttributeAccessPolicyWindow`).
    ///
    /// Persisting hits reduce billed HBM after
    /// [`Self::set_persisting_l2_cache_size`]. Decode identity stays [`Self::kernel`].
    pub fn kernel_access_policy(
        &mut self,
        device: DeviceId,
        kind: KernelKind,
        reads: &[AllocId],
        writes: &[AllocId],
        stream: StreamId,
        window: AccessPolicyWindow,
    ) -> Result<OpId, SimError> {
        self.kernel_with(
            device,
            kind,
            reads,
            writes,
            stream,
            KernelAttrs {
                access_policy: Some(window),
                ..KernelAttrs::default()
            },
        )
    }

    /// [`Self::kernel_bufs`] plus [`AccessPolicyWindow`].
    pub fn kernel_access_policy_bufs(
        &mut self,
        device: DeviceId,
        kind: KernelKind,
        reads: &[KernelBuf],
        writes: &[KernelBuf],
        stream: StreamId,
        window: AccessPolicyWindow,
    ) -> Result<OpId, SimError> {
        self.kernel_bufs_with(
            device,
            kind,
            reads,
            writes,
            stream,
            KernelAttrs {
                access_policy: Some(window),
                ..KernelAttrs::default()
            },
        )
    }

    /// `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize)`.
    ///
    /// Host-synchronous. Capture cannot include it. `bytes` must be `<=`
    /// [`crate::GpuProfile::l2_bytes`]. CUDA default is 0 (windows are a no-op
    /// until this is set). Shrinking evicts oldest persisting lines.
    pub fn set_persisting_l2_cache_size(
        &mut self,
        device: DeviceId,
        bytes: u64,
    ) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture persisting L2")?;
        let cap = self.profile.gpu(device)?.l2_bytes;
        if bytes > cap {
            return Err(SimError::Invalid {
                why: "persisting L2 cache size",
            });
        }
        let rt = self.gpus.get_mut(&device).ok_or(SimError::Invalid {
            why: "unknown device",
        })?;
        rt.persist_limit = bytes;
        persist_trim(rt);
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaDeviceGetLimit(cudaLimitPersistingL2CacheSize)`.
    pub fn persisting_l2_cache_size(&self, device: DeviceId) -> Result<u64, SimError> {
        let _gpu = self.profile.gpu(device)?;
        Ok(self
            .gpus
            .get(&device)
            .ok_or(SimError::Invalid {
                why: "unknown device",
            })?
            .persist_limit)
    }

    /// `cudaCtxResetPersistingL2Cache`.
    ///
    /// Host-synchronous. Drops filled persisting lines; the limit stays.
    /// Capture cannot include it. The next persisting kernel refills (cold).
    pub fn reset_persisting_l2_cache(&mut self, device: DeviceId) -> Result<(), SimError> {
        self.fail_if_capturing("cannot capture persisting L2")?;
        let _gpu = self.profile.gpu(device)?;
        let rt = self.gpus.get_mut(&device).ok_or(SimError::Invalid {
            why: "unknown device",
        })?;
        rt.persist_lines.clear();
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// Set each GPU's persisting L2 limit to [`crate::GpuProfile::l2_bytes`].
    ///
    /// Fails `l2 persist needs l2_bytes` when a GPU reports 0. Decode identity
    /// does not call this (limit stays 0).
    pub fn enable_persisting_l2(&mut self) -> Result<(), SimError> {
        let ids: Vec<(DeviceId, u64)> = self
            .profile
            .gpus
            .iter()
            .map(|g| (g.id, g.l2_bytes))
            .collect();
        for (id, bytes) in ids {
            if bytes == 0 {
                return Err(SimError::Invalid {
                    why: "l2 persist needs l2_bytes",
                });
            }
            self.set_persisting_l2_cache_size(id, bytes)?;
        }
        Ok(())
    }

    /// `cudaDeviceSetLimit`. Host-synchronous. Capture cannot include it.
    ///
    /// [`DeviceLimit::PersistingL2CacheSize`] is [`Self::set_persisting_l2_cache_size`].
    /// [`DeviceLimit::MaxL2FetchGranularity`] must be 32, 64, or 128 (CUDA SM 8.0+
    /// default 128). Access-policy windows must align to the current value.
    /// Heap / stack / printf / CDP limits are stored; heap does not charge HBM.
    pub fn set_limit(
        &mut self,
        device: DeviceId,
        limit: DeviceLimit,
        value: u64,
    ) -> Result<(), SimError> {
        if let DeviceLimit::PersistingL2CacheSize = limit {
            return self.set_persisting_l2_cache_size(device, value);
        }
        self.fail_if_capturing("cannot capture set limit")?;
        let _gpu = self.profile.gpu(device)?;
        let rt = self.gpu_rt_mut(device)?;
        match limit {
            DeviceLimit::StackSize if value > 0 => rt.limits.stack_size = value,
            DeviceLimit::PrintfFifoSize if value > 0 => rt.limits.printf_fifo = value,
            DeviceLimit::MallocHeapSize => rt.limits.malloc_heap = value,
            DeviceLimit::DevRuntimeSyncDepth if value >= 2 => rt.limits.sync_depth = value,
            DeviceLimit::DevRuntimePendingLaunchCount if value > 0 => {
                rt.limits.pending_launch = value;
            }
            DeviceLimit::MaxL2FetchGranularity if value == 32 || value == 64 || value == 128 => {
                rt.limits.l2_fetch = value;
            }
            DeviceLimit::MaxL2FetchGranularity => {
                return Err(SimError::Invalid {
                    why: "l2 fetch granularity",
                });
            }
            DeviceLimit::PersistingL2CacheSize => {}
            DeviceLimit::StackSize
            | DeviceLimit::PrintfFifoSize
            | DeviceLimit::DevRuntimeSyncDepth
            | DeviceLimit::DevRuntimePendingLaunchCount => {
                return Err(SimError::Invalid {
                    why: "device limit",
                });
            }
        }
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// `cudaDeviceGetLimit`. Query; legal during capture.
    pub fn get_limit(&self, device: DeviceId, limit: DeviceLimit) -> Result<u64, SimError> {
        if limit == DeviceLimit::PersistingL2CacheSize {
            return self.persisting_l2_cache_size(device);
        }
        let rt = self.gpu_rt(device)?;
        Ok(match limit {
            DeviceLimit::StackSize => rt.limits.stack_size,
            DeviceLimit::PrintfFifoSize => rt.limits.printf_fifo,
            DeviceLimit::MallocHeapSize => rt.limits.malloc_heap,
            DeviceLimit::DevRuntimeSyncDepth => rt.limits.sync_depth,
            DeviceLimit::DevRuntimePendingLaunchCount => rt.limits.pending_launch,
            DeviceLimit::MaxL2FetchGranularity => rt.limits.l2_fetch,
            DeviceLimit::PersistingL2CacheSize => rt.persist_limit,
        })
    }

    /// `cudaPointerGetAttributes`. Query; legal during capture.
    ///
    /// A never-created id is [`SimError::UnknownAlloc`]. A freed id is
    /// [`MemoryType::Unregistered`] (CUDA 11+).
    pub fn pointer_get_attributes(&self, id: AllocId) -> Result<PointerAttributes, SimError> {
        let a = self.alloc_ref(id)?;
        if !a.live {
            return Ok(PointerAttributes {
                kind: MemoryType::Unregistered,
                device: None,
                device_pointer: false,
                host_pointer: false,
            });
        }
        if a.managed {
            return Ok(PointerAttributes {
                kind: MemoryType::Managed,
                device: a.devices.first().copied(),
                device_pointer: true,
                host_pointer: true,
            });
        }
        if a.host_pageable || a.host_pinned {
            let mapped = a.host_mapped;
            return Ok(PointerAttributes {
                kind: MemoryType::Host,
                device: if mapped {
                    self.profile.gpus.first().map(|g| g.id)
                } else {
                    None
                },
                device_pointer: mapped,
                host_pointer: true,
            });
        }
        Ok(PointerAttributes {
            kind: MemoryType::Device,
            device: a
                .devices
                .first()
                .copied()
                .or_else(|| a.vmm_maps.first().map(|m| m.0)),
            device_pointer: true,
            host_pointer: false,
        })
    }

    /// `cudaHostGetDevicePointer`. Query; legal during capture.
    ///
    /// Mapped host (`cudaHostAllocMapped` / `cudaHostRegisterMapped`) returns
    /// the same id (this VM has one pointer space). Unmapped host and device
    /// allocs are Invalid `not mapped`.
    pub fn host_get_device_pointer(&self, id: AllocId) -> Result<AllocId, SimError> {
        let a = self.alloc_ref(id)?;
        if !a.live {
            return Err(SimError::UnknownAlloc { alloc: id });
        }
        if a.host_mapped {
            return Ok(id);
        }
        Err(SimError::Invalid { why: "not mapped" })
    }

    /// `cudaHostGetFlags`. Query; legal during capture.
    ///
    /// Returns [`HostAllocFlags::MAPPED`] when the pointer is
    /// `cudaHostAllocMapped` / `cudaHostRegisterMapped`, else `0` for pinned
    /// or registered host. Device, managed, VMM, and unregistered pageable
    /// pointers are Invalid `"not host alloc"`. Portable / WriteCombined are
    /// not modeled.
    pub fn host_get_flags(&self, id: AllocId) -> Result<u32, SimError> {
        let a = self.alloc_ref(id)?;
        if !a.live {
            return Err(SimError::UnknownAlloc { alloc: id });
        }
        if a.managed || a.vmm || !(a.host_pinned || a.host_registered) {
            return Err(SimError::Invalid {
                why: "not host alloc",
            });
        }
        if a.host_mapped {
            Ok(HostAllocFlags::MAPPED)
        } else {
            Ok(HostAllocFlags::DEFAULT)
        }
    }

    /// `cudaDeviceGetAttribute`. Query; legal during capture.
    ///
    /// Only attributes this VM already models ([`DeviceAttr`]).
    pub fn device_get_attribute(
        &self,
        device: DeviceId,
        attr: DeviceAttr,
    ) -> Result<u64, SimError> {
        let gpu = self.profile.gpu(device)?;
        Ok(match attr {
            DeviceAttr::CooperativeLaunch => u64::from(gpu.cooperative_launch),
            DeviceAttr::ConcurrentKernels => u64::from(gpu.compute_slots > 1),
            DeviceAttr::MaxSharedMemoryPerBlock => u64::from(gpu.max_shared_mem_per_block),
            DeviceAttr::MaxSharedMemoryPerBlockOptin => {
                u64::from(gpu.max_shared_mem_per_block_optin)
            }
            DeviceAttr::L2CacheSize | DeviceAttr::MaxPersistingL2CacheSize => gpu.l2_bytes,
            DeviceAttr::MaxBlocksPerCluster => u64::from(gpu.max_blocks_per_cluster),
            DeviceAttr::MemSyncDomainCount => u64::from(gpu.mem_sync_domain_count),
            DeviceAttr::MemoryPoolsSupported => 1,
            DeviceAttr::CanMapHostMemory | DeviceAttr::ManagedMemory => 1,
            DeviceAttr::TotalGlobalMem => gpu.hbm_bytes,
            DeviceAttr::AsyncEngineCount => u64::from(gpu.copy_engines),
            DeviceAttr::ClusterLaunch => u64::from(gpu.max_blocks_per_cluster > 0),
            DeviceAttr::HostRegisterSupported
            | DeviceAttr::IpcEventSupport
            | DeviceAttr::CanUseHostPointerForRegisteredMem => 1,
            DeviceAttr::MemoryPoolSupportedHandleTypes => MemHandleType::POSIX_FILE_DESCRIPTOR,
        })
    }

    /// `cudaGetDeviceProperties`. Query; legal during capture.
    ///
    /// Only fields this VM already models ([`DeviceProperties`]). Unknown
    /// devices are Invalid.
    pub fn device_get_properties(&self, device: DeviceId) -> Result<DeviceProperties, SimError> {
        let gpu = self.profile.gpu(device)?;
        Ok(DeviceProperties {
            name: self.profile.name.clone(),
            total_global_mem: gpu.hbm_bytes,
            shared_mem_per_block: gpu.max_shared_mem_per_block,
            shared_mem_per_block_optin: gpu.max_shared_mem_per_block_optin,
            l2_cache_size: gpu.l2_bytes,
            async_engine_count: u32::from(gpu.copy_engines),
            concurrent_kernels: gpu.compute_slots > 1,
            cooperative_launch: gpu.cooperative_launch,
            max_blocks_per_cluster: u32::from(gpu.max_blocks_per_cluster),
            portable_cluster_size: u32::from(gpu.portable_cluster_size),
            mem_sync_domain_count: u32::from(gpu.mem_sync_domain_count),
            memory_pools_supported: true,
            can_map_host_memory: true,
            managed_memory: true,
            cluster_launch: gpu.max_blocks_per_cluster > 0,
            host_register_supported: true,
            ipc_event_support: true,
            can_use_host_pointer_for_registered_mem: true,
            memory_pool_supported_handle_types: MemHandleType::POSIX_FILE_DESCRIPTOR,
        })
    }

    /// `cudaGetDeviceCount`. Query; legal during capture.
    #[must_use]
    pub fn device_count(&self) -> u32 {
        u32::try_from(self.profile.gpus.len()).unwrap_or(u32::MAX)
    }

    /// `cudaDeviceCanAccessPeer`. Query; legal during capture.
    ///
    /// Hardware topology only (a profile link). Same device is false.
    /// [`Self::enable_peer`] is still required before D2D.
    pub fn device_can_access_peer(
        &self,
        device: DeviceId,
        peer: DeviceId,
    ) -> Result<bool, SimError> {
        Ok(self.device_get_p2p_attribute(device, peer, DeviceP2pAttr::AccessSupported)? != 0)
    }

    /// `cudaDeviceGetP2PAttribute`. Query; legal during capture.
    ///
    /// [`DeviceP2pAttr::AccessSupported`] is a profile device–device link.
    /// [`DeviceP2pAttr::PerformanceRank`] is unique GPU↔GPU link `bps`
    /// descending (lower is better). Same device is 0. Missing links are 0,
    /// not [`SimError::NoPeer`]. Unknown devices are Invalid. Native atomics
    /// are not modeled.
    pub fn device_get_p2p_attribute(
        &self,
        src: DeviceId,
        dst: DeviceId,
        attr: DeviceP2pAttr,
    ) -> Result<u64, SimError> {
        let _src = self.profile.gpu(src)?;
        let _dst = self.profile.gpu(dst)?;
        Ok(match attr {
            DeviceP2pAttr::AccessSupported => {
                if src == dst || self.profile.link(Some(src), Some(dst)).is_err() {
                    0
                } else {
                    1
                }
            }
            DeviceP2pAttr::PerformanceRank => self.profile.p2p_performance_rank(src, dst),
        })
    }

    /// `cudaMallocPitch`: aligned 2D allocation. Returns `(ptr, pitch)`.
    ///
    /// Pitch is `align_up(width, 512)`. Size charged is `pitch * height`.
    /// Host-synchronous like [`Self::malloc`]. Capture cannot include it.
    pub fn malloc_pitch(
        &mut self,
        device: DeviceId,
        width: u64,
        height: u64,
    ) -> Result<(AllocId, u64), SimError> {
        if width == 0 || height == 0 {
            return Err(SimError::Invalid {
                why: "malloc pitch",
            });
        }
        let pitch = align_up(width, 512);
        let bytes = pitch.saturating_mul(height);
        let id = self.malloc(device, bytes)?;
        Ok((id, pitch))
    }

    /// `cudaMalloc3D`: aligned 3D allocation. Returns `(ptr, pitch)`.
    ///
    /// Pitch is `align_up(width, 512)`. Size charged is `pitch * height * depth`.
    /// Host-synchronous like [`Self::malloc`]. Capture cannot include it.
    pub fn malloc_3d(
        &mut self,
        device: DeviceId,
        width: u64,
        height: u64,
        depth: u64,
    ) -> Result<(AllocId, u64), SimError> {
        if width == 0 || height == 0 || depth == 0 {
            return Err(SimError::Invalid { why: "malloc 3d" });
        }
        let pitch = align_up(width, 512);
        let bytes = pitch.saturating_mul(height).saturating_mul(depth);
        let id = self.malloc(device, bytes)?;
        Ok((id, pitch))
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
    /// [`Self::memset_op`] is `cudaMemset2DAsync` when [`MemsetOp::height`] `> 1`.
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
        self.memset_op(device, MemsetOp::from(buf), stream)
    }

    /// `cudaMemsetAsync` / `cudaMemset2DAsync`.
    ///
    /// [`MemsetOp::height`] `> 1` bills `width * height` as an HBM write (pitch
    /// padding is not written). [`MemsetOp::depth`] `> 1` is `cudaMemset3DAsync`
    /// (`width * height * depth`). The mapped span is the 2D/3D extent. Capture
    /// is allowed. Mapped host is not a memset dest.
    pub fn memset_op(
        &mut self,
        device: DeviceId,
        op: MemsetOp,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        let op = self.resolve_memset_op(op)?;
        self.submit(device, stream, Kind::Memset(op))
    }

    /// `cudaLaunchHostFunc`. Stream-ordered host work; does not occupy compute
    /// or copy engines. Other streams can run GPU kernels at the same virtual
    /// time. Capture records a graph host node. Unnamed callback (`fn_id = 0`,
    /// `user_data = 0`).
    pub fn host_func(&mut self, device: DeviceId, stream: StreamId) -> Result<OpId, SimError> {
        self.host_func_params(device, stream, HostNodeParams::default())
    }

    /// `cudaLaunchHostFunc` with [`HostNodeParams`].
    ///
    /// Capture records those params on the host node. Does not occupy compute
    /// or copy engines.
    pub fn host_func_params(
        &mut self,
        device: DeviceId,
        stream: StreamId,
        params: HostNodeParams,
    ) -> Result<OpId, SimError> {
        self.submit(
            device,
            stream,
            Kind::HostFunc {
                fn_id: params.fn_id,
                user_data: params.user_data,
            },
        )
    }

    /// `cuStreamWriteValue64`. The mailbox updates when this op completes.
    ///
    /// Does not occupy compute or copy engines. Capture records a batch-mem-op
    /// node. Alignment is 8 bytes; the span must fit the allocation.
    pub fn write_value64(
        &mut self,
        device: DeviceId,
        id: AllocId,
        offset: u64,
        value: u64,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        self.submit_batch_mem(
            device,
            stream,
            BatchMemOp::Write {
                id,
                offset,
                value,
                bits32: false,
            },
        )
    }

    /// `cuStreamWriteValue32`. Stores the low 32 bits; high bits of a prior
    /// 64-bit write at the same offset stay.
    pub fn write_value32(
        &mut self,
        device: DeviceId,
        id: AllocId,
        offset: u64,
        value: u64,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        self.submit_batch_mem(
            device,
            stream,
            BatchMemOp::Write {
                id,
                offset,
                value,
                bits32: true,
            },
        )
    }

    /// `cuStreamWaitValue64`. Stays pending until the mailbox compare matches.
    ///
    /// Unwritten locations read as 0. Does not occupy compute or copy engines.
    /// Capture records a batch-mem-op node. An unsatisfied wait plus
    /// [`Self::synchronize`] is deadlock if nothing else is running.
    pub fn wait_value64(
        &mut self,
        device: DeviceId,
        id: AllocId,
        offset: u64,
        value: u64,
        cmp: WaitValueCmp,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        self.submit_batch_mem(
            device,
            stream,
            BatchMemOp::Wait {
                id,
                offset,
                value,
                bits32: false,
                cmp,
            },
        )
    }

    /// `cuStreamWaitValue32`. Compares the low 32 bits of the mailbox word.
    pub fn wait_value32(
        &mut self,
        device: DeviceId,
        id: AllocId,
        offset: u64,
        value: u64,
        cmp: WaitValueCmp,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        self.submit_batch_mem(
            device,
            stream,
            BatchMemOp::Wait {
                id,
                offset,
                value,
                bits32: true,
                cmp,
            },
        )
    }

    /// `cuStreamBatchMemOp`. One stream op for the wait/write vector.
    ///
    /// Empty is Invalid. A single item is [`Self::write_value64`] /
    /// [`Self::wait_value64`] (same [`crate::GpuOp`] as those APIs). Two or more
    /// items are [`crate::GpuOp::BatchMem`]. Capture records one graph node.
    /// Writes commit on complete. A wait sees earlier writes in this vector.
    pub fn batch_mem_op(
        &mut self,
        device: DeviceId,
        stream: StreamId,
        ops: &[BatchMemOp],
    ) -> Result<OpId, SimError> {
        self.check_batch_mem_ops(ops)?;
        if let [op] = ops {
            return self.submit(device, stream, kind_from_batch(*op));
        }
        self.submit(device, stream, Kind::BatchMem { ops: ops.to_vec() })
    }

    fn submit_batch_mem(
        &mut self,
        device: DeviceId,
        stream: StreamId,
        op: BatchMemOp,
    ) -> Result<OpId, SimError> {
        self.check_batch_mem(op)?;
        self.submit(device, stream, kind_from_batch(op))
    }

    fn check_batch_mem_ops(&self, ops: &[BatchMemOp]) -> Result<(), SimError> {
        if ops.is_empty() {
            return Err(SimError::Invalid {
                why: "empty batch mem op",
            });
        }
        for op in ops {
            self.check_batch_mem(*op)?;
        }
        Ok(())
    }

    fn check_batch_mem(&self, op: BatchMemOp) -> Result<(), SimError> {
        let (id, offset, bits32) = match op {
            BatchMemOp::Write {
                id, offset, bits32, ..
            }
            | BatchMemOp::Wait {
                id, offset, bits32, ..
            } => (id, offset, bits32),
        };
        let total = self.alloc_ref(id)?.bytes;
        let _width = wait_value_span(total, offset, bits32)?;
        Ok(())
    }

    fn apply_mailbox_write(&mut self, id: AllocId, offset: u64, value: u64, bits32: bool) {
        let mask = wait_value_mask(bits32);
        let prev = self.mailbox.get(&(id, offset)).copied().unwrap_or(0);
        let stored = (prev & !mask) | (value & mask);
        let _old = self.mailbox.insert((id, offset), stored);
    }

    fn clear_mailbox(&mut self, id: AllocId) {
        self.mailbox.retain(|(a, _), _| *a != id);
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
        self.record_event_with_flags(device, event, stream, EventRecordFlags::DEFAULT)
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
        self.record_event_with_flags(device, event, stream, EventRecordFlags::EXTERNAL)
    }

    /// `cudaEventRecordWithFlags`. Unknown bits are Invalid `"event record flags"`.
    pub fn record_event_with_flags(
        &mut self,
        device: DeviceId,
        event: EventId,
        stream: StreamId,
        flags: u32,
    ) -> Result<OpId, SimError> {
        const KNOWN: u32 = EventRecordFlags::EXTERNAL;
        if flags & !KNOWN != 0 {
            return Err(SimError::Invalid {
                why: "event record flags",
            });
        }
        self.record_event_flags(
            device,
            event,
            stream,
            flags & EventRecordFlags::EXTERNAL != 0,
        )
    }

    fn record_event_flags(
        &mut self,
        device: DeviceId,
        event: EventId,
        stream: StreamId,
        external: bool,
    ) -> Result<OpId, SimError> {
        let _ev = self.events.entry(event).or_insert(Ev::new(true));
        self.submit(device, stream, Kind::EventRecord { event, external })
    }

    /// Make later ops on `stream` wait until `event` is recorded and complete.
    pub fn wait_event(
        &mut self,
        device: DeviceId,
        event: EventId,
        stream: StreamId,
    ) -> Result<OpId, SimError> {
        self.wait_event_with_flags(device, event, stream, EventWaitFlags::DEFAULT)
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
        self.wait_event_with_flags(device, event, stream, EventWaitFlags::EXTERNAL)
    }

    /// `cudaStreamWaitEvent` with flags. Unknown bits are Invalid
    /// `"event wait flags"`.
    pub fn wait_event_with_flags(
        &mut self,
        device: DeviceId,
        event: EventId,
        stream: StreamId,
        flags: u32,
    ) -> Result<OpId, SimError> {
        const KNOWN: u32 = EventWaitFlags::EXTERNAL;
        if flags & !KNOWN != 0 {
            return Err(SimError::Invalid {
                why: "event wait flags",
            });
        }
        self.wait_event_flags(device, event, stream, flags & EventWaitFlags::EXTERNAL != 0)
    }

    fn wait_event_flags(
        &mut self,
        device: DeviceId,
        event: EventId,
        stream: StreamId,
        external: bool,
    ) -> Result<OpId, SimError> {
        if let Entry::Vacant(slot) = self.events.entry(event) {
            let _ev = slot.insert(Ev::new(true));
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
    /// After the GPU drain, the host-wait tax from
    /// [`Self::stream_sync_policy`] is added (`Auto` / profile `0` is identity).
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
        self.stream_sync_outcome(device, stream)?;
        self.apply_stream_sync_policy_tax(device, stream);
        Ok(())
    }

    /// `cudaEventSynchronize`: wait until `event` is recorded and complete.
    ///
    /// Work after the record on the same stream, and work on other streams,
    /// keeps running. An event with no record and no running ops is a deadlock.
    /// After the GPU drain, the host-wait tax from the recording stream's
    /// [`Self::stream_sync_policy`] is added.
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
        let rec = self.event_recorded_by(event);
        if let Some(id) = rec {
            if self.ops.get(&id).is_some_and(|o| o.cancelled) {
                let stream = self.ops.get(&id).map(|o| o.stream).unwrap_or(StreamId(0));
                return Err(SimError::Cancelled { stream, n: 1 });
            }
        }
        self.apply_event_sync_policy_tax(event);
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
        let rec = self.event_recorded_by(event);
        let Some(id) = rec else {
            return Err(SimError::Invalid {
                why: "event elapsed: not recorded",
            });
        };
        let root = self.event_root(event);
        let op = self.ops.get(&id).ok_or(SimError::Invalid {
            why: "event elapsed: missing record op",
        })?;
        if op
            .programmatic_event
            .is_some_and(|p| event_root_of(&self.events, p.event) == root)
            && op.pdl.trigger
        {
            return op.pdl_trigger_ns.ok_or(SimError::Invalid {
                why: "event elapsed: programmatic trigger missing",
            });
        }
        if op
            .launch_completion
            .is_some_and(|p| event_root_of(&self.events, p.event) == root)
        {
            return op.start_ns.ok_or(SimError::Invalid {
                why: "event elapsed: launch completion missing start",
            });
        }
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

    fn host_sync_policy_tax_ns(&self, device: DeviceId, policy: SynchronizationPolicy) -> u64 {
        let Ok(g) = self.profile.gpu(device) else {
            return 0;
        };
        match policy {
            SynchronizationPolicy::Auto => 0,
            SynchronizationPolicy::Spin => g.host_sync_spin_ns,
            SynchronizationPolicy::Yield => g.host_sync_yield_ns,
            SynchronizationPolicy::BlockingSync => g.host_sync_blocking_ns,
        }
    }

    fn apply_stream_sync_policy_tax(&mut self, device: DeviceId, stream: StreamId) {
        let policy = self.stream_sync_policy(device, stream);
        let tax = self.host_sync_policy_tax_ns(device, policy);
        self.clock = self.clock.saturating_add(tax);
    }

    fn apply_event_sync_policy_tax(&mut self, event: EventId) {
        let Some(op) = self.event_recorded_by(event) else {
            return;
        };
        let Some(row) = self.ops.get(&op) else {
            return;
        };
        let device = row.device;
        let stream = row.stream;
        self.apply_stream_sync_policy_tax(device, stream);
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
            let root = self.event_root(*event);
            let join = !*external
                && self
                    .capturing
                    .as_ref()
                    .is_some_and(|c| c.events.contains(&root));
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
            let live = self
                .capturing
                .as_ref()
                .is_some_and(|c| c.mode.live_uncaptured());
            if live {
                return self.submit_live(device, stream, kind, LaunchCost::Kernel);
            }
            return Err(SimError::Invalid {
                why: "stream not capturing",
            });
        }
        if let Kind::EventRecord { event, external } = &kind {
            if !*external {
                let root = self.event_root(*event);
                if let Some(cap) = self.capturing.as_mut() {
                    let _ins = cap.events.insert(root);
                }
            }
        }
        if let Some(pe) = self.enqueue_programmatic_event {
            let _ev = self.events.entry(pe.event).or_insert(Ev::new(true));
            if !pe.external {
                let root = self.event_root(pe.event);
                if let Some(cap) = self.capturing.as_mut() {
                    let _ins = cap.events.insert(root);
                }
            }
        }
        if let Some(lc) = self.enqueue_launch_completion {
            let _ev = self.events.entry(lc.event).or_insert(Ev::new(true));
            if !lc.external {
                let root = self.event_root(lc.event);
                if let Some(cap) = self.capturing.as_mut() {
                    let _ins = cap.events.insert(root);
                }
            }
        }
        let mut deps = capture_step_deps(&self.capture_buf, device, stream, &kind, &self.events);
        self.merge_capture_pending(device, stream, &mut deps);
        let priority = self.snap_priority(device, stream);
        let (mem_sync_domain, mem_sync_map) = self.snap_mem_sync(device, stream, &kind)?;
        self.capture_buf.push(GraphStep {
            device,
            stream,
            kind,
            deps,
            enabled: true,
            destroyed: false,
            priority,
            pdl: self.enqueue_pdl,
            programmatic_event: self.enqueue_programmatic_event,
            launch_completion: self.enqueue_launch_completion,
            access_policy: self.enqueue_access_policy,
            mem_sync_domain,
            mem_sync_map,
            cluster: self.enqueue_cluster,
            cluster_policy: self.enqueue_cluster_policy,
            preferred_cluster: self.enqueue_preferred_cluster,
            carveout: self.enqueue_carveout,
            device_updatable: self.enqueue_device_updatable,
            shared_mem: self.enqueue_shared_mem,
            portable_cluster: self.enqueue_portable_cluster,
            dynamic_shared: self.enqueue_dynamic_shared,
            portable_shared: self.enqueue_portable_shared,
            nvlink_util_centric: self.enqueue_nvlink_util_centric,
        });
        let id = OpId(self.next_op);
        self.next_op = self.next_op.saturating_add(1);
        Ok(id)
    }

    fn merge_capture_pending(&mut self, device: DeviceId, stream: StreamId, deps: &mut Vec<usize>) {
        let extra = self
            .capturing
            .as_mut()
            .map(|c| c.pending.remove(&(device, stream)).unwrap_or_default())
            .unwrap_or_default();
        if extra.is_empty() {
            return;
        }
        let graph = self.capturing.as_ref().map(|c| c.into.graph);
        let existing = graph
            .and_then(|g| self.graphs.get(&g))
            .map_or(0, |g| g.steps.len());
        let buf_i = self.capture_buf.len();
        let mut extra_abs = Vec::new();
        for p in extra {
            if p < existing {
                extra_abs.push(p);
            } else {
                let rel = p.saturating_sub(existing);
                if rel < buf_i && !deps.contains(&rel) {
                    deps.push(rel);
                }
            }
        }
        deps.sort_unstable();
        extra_abs.sort_unstable();
        extra_abs.dedup();
        if extra_abs.is_empty() {
            return;
        }
        if let Some(cap) = self.capturing.as_mut() {
            let _prev = cap.extra_abs.insert(buf_i, extra_abs);
        }
    }

    fn submit_live(
        &mut self,
        device: DeviceId,
        stream: StreamId,
        kind: Kind,
        launch: LaunchCost,
    ) -> Result<OpId, SimError> {
        let _gpu = self.profile.gpu(device)?;
        if matches!(kind, Kind::Kernel { .. }) {
            self.validate_cluster_attrs(
                device,
                self.enqueue_cluster,
                self.enqueue_preferred_cluster,
                self.enqueue_portable_cluster,
            )?;
            self.validate_dynamic_shared(
                device,
                self.enqueue_dynamic_shared,
                self.enqueue_portable_shared,
            )?;
        }
        let id = OpId(self.next_op);
        self.next_op = self.next_op.saturating_add(1);
        let pde = self.enqueue_programmatic_event;
        let lce = self.enqueue_launch_completion;
        if let Some(pe) = pde {
            let _ev = self.events.entry(pe.event).or_insert(Ev::new(true));
        }
        if let Some(lc) = lce {
            let _ev = self.events.entry(lc.event).or_insert(Ev::new(true));
        }
        let mut deps = self.stream_order_deps(device, stream);
        if let Kind::EventWait { event, .. } = &kind {
            if let Some(rec) = self.event_recorded_by(*event) {
                deps.push(rec);
            }
        }
        let priority = self.snap_priority(device, stream);
        let (mem_sync_domain, mem_sync_map) = self.snap_mem_sync(device, stream, &kind)?;
        let mem_sync_physical = mem_sync_map.physical(mem_sync_domain);
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
                preds: self.enqueue_preds.clone(),
                skipped: false,
                priority,
                pdl: self.enqueue_pdl,
                pdl_trigger_ns: None,
                programmatic_event: pde,
                launch_completion: lce,
                access_policy: self.enqueue_access_policy,
                mem_sync_physical,
                domain_fence_paid: false,
                cluster: self.enqueue_cluster,
                cluster_policy: self.enqueue_cluster_policy,
                preferred_cluster: self.enqueue_preferred_cluster,
                carveout: self.enqueue_carveout,
                shared_mem: self.enqueue_shared_mem,
                nvlink_util_centric: self.enqueue_nvlink_util_centric,
            },
        );
        if let Some(pe) = pde {
            let root = self.event_root(pe.event);
            if let Some(ev) = self.events.get_mut(&root) {
                ev.recorded_by = Some(id);
            }
        }
        if let Some(lc) = lce {
            let root = self.event_root(lc.event);
            if let Some(ev) = self.events.get_mut(&root) {
                ev.recorded_by = Some(id);
            }
        }
        let _prev_tail = self.tail.insert((device, stream), id);
        Ok(id)
    }

    fn stream_order_deps(&self, device: DeviceId, stream: StreamId) -> Vec<OpId> {
        let mut deps = Vec::new();
        if let Some(prev) = self.tail.get(&(device, stream)) {
            deps.push(*prev);
        }
        // PDL can finish a later wait kernel while an earlier trigger kernel
        // is still running. `cudaFreeAsync` and other non-wait submits still
        // wait for every preceding op on the stream (CUDA: all preceding work).
        for (id, o) in &self.ops {
            if o.device == device
                && o.stream == stream
                && !o.done
                && !o.cancelled
                && !deps.contains(id)
            {
                deps.push(*id);
            }
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
        let priority = self
            .ops
            .get(&kernel)
            .ok_or(SimError::Invalid { why: "unknown op" })?
            .priority;
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
                    ..MemcpyOp::default()
                }),
                deps: deps.to_vec(),
                done: false,
                cancelled: false,
                launch: LaunchCost::Kernel,
                submit_ns: self.clock,
                start_ns: None,
                done_ns: None,
                preds: Vec::new(),
                skipped: false,
                priority,
                pdl: ProgrammaticLaunch::default(),
                pdl_trigger_ns: None,
                programmatic_event: None,
                launch_completion: None,
                access_policy: None,
                mem_sync_physical: 0,
                domain_fence_paid: false,
                cluster: None,
                cluster_policy: ClusterSchedulingPolicy::Default,
                preferred_cluster: None,
                carveout: SharedMemCarveout::Default,
                shared_mem: SharedMemoryMode::Default,
                nvlink_util_centric: false,
            },
        );
        self.add_op_dep(kernel, id);
        Ok(())
    }

    fn start_kernel(&mut self, id: OpId) -> Result<bool, SimError> {
        let (device, stream, launch, reads, writes, kind, shared_mem) = {
            let op = self
                .ops
                .get(&id)
                .ok_or(SimError::Invalid { why: "unknown op" })?;
            match &op.kind {
                Kind::Kernel {
                    reads,
                    writes,
                    kind,
                    ..
                } => (
                    op.device,
                    op.stream,
                    op.launch,
                    reads.clone(),
                    writes.clone(),
                    kind.clone(),
                    op.shared_mem,
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
        let slots = {
            let op = self
                .ops
                .get(&id)
                .ok_or(SimError::Invalid { why: "unknown op" })?;
            self.op_kernel_slots(op)?
        };
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
        let window = self.ops.get(&id).and_then(|o| o.access_policy);
        let billed = match self.persist_kernel_bytes(device, window, &kind, &reads, &writes) {
            Ok(n) => n,
            Err(e) => {
                self.drop_compute_n(device, slots)?;
                return Err(e);
            }
        };
        let ns = match self.kernel_ns(device, stream, &kind, launch, mem_bps, billed) {
            Ok(n) => n,
            Err(e) => {
                self.drop_compute_n(device, slots)?;
                return Err(e);
            }
        };
        let ns = match self.shared_mem_ns(device, shared_mem, ns) {
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
        if let Some(op) = self.ops.get_mut(&id) {
            if op.pdl.trigger {
                let permille = u64::from(
                    self.profile
                        .gpu(device)
                        .map(|g| g.pdl_trigger_permille.min(1000))
                        .unwrap_or(1000),
                );
                let delay = ns.saturating_mul(permille) / 1000;
                op.pdl_trigger_ns = Some(self.clock.saturating_add(delay));
            }
        }
        Ok(true)
    }

    fn start_memset(&mut self, id: OpId) -> Result<bool, SimError> {
        let (device, stream, launch, op) = {
            let row = self
                .ops
                .get(&id)
                .ok_or(SimError::Invalid { why: "unknown op" })?;
            match &row.kind {
                Kind::Memset(op) => (row.device, row.stream, row.launch, *op),
                _ => {
                    return Err(SimError::Invalid {
                        why: "not a memset",
                    });
                }
            }
        };
        self.require_device_attach(op.id, stream)?;
        if !self.take_compute(device)? {
            return Ok(false);
        }
        let writes = [KernelBuf::span(op.id, op.offset, op.extent_bytes())];
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
        let ns = match self.memset_ns(device, op.payload_bytes(), launch, mem_bps) {
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
            self.bytes_moved = self.bytes_moved.saturating_add(m.payload_bytes());
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
        let gone = {
            let a = self.alloc_mut(alloc)?;
            a.devices.retain(|d| *d != device);
            if a.devices.is_empty() && !a.host_pinned {
                a.live = false;
                true
            } else {
                false
            }
        };
        if gone {
            self.clear_mailbox(alloc);
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
        {
            let a = self.alloc_mut(id)?;
            a.live = false;
            a.devices.clear();
        }
        self.clear_mailbox(id);
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
        {
            let a = self.alloc_mut(id)?;
            a.live = false;
            a.devices.clear();
        }
        self.clear_mailbox(id);
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
        op.deps.iter().all(|d| self.dep_satisfied(op, *d))
    }

    fn dep_satisfied(&self, op: &Op, dep: OpId) -> bool {
        if self.op_done(dep) {
            return true;
        }
        let Some(prev) = self.ops.get(&dep) else {
            return false;
        };
        if op.pdl.wait
            && prev.pdl.trigger
            && prev.device == op.device
            && prev.stream == op.stream
            && prev.pdl_trigger_ns.is_some_and(|t| self.clock >= t)
        {
            return true;
        }
        if let Kind::EventWait { event, .. } = &op.kind {
            let root = event_root_of(&self.events, *event);
            if prev
                .programmatic_event
                .is_some_and(|p| event_root_of(&self.events, p.event) == root)
                && prev.pdl.trigger
                && prev.pdl_trigger_ns.is_some_and(|t| self.clock >= t)
            {
                return true;
            }
            if prev
                .launch_completion
                .is_some_and(|p| event_root_of(&self.events, p.event) == root)
                && prev.start_ns.is_some()
            {
                return true;
            }
        }
        false
    }

    fn next_pdl_wake(&self) -> Option<u64> {
        let mut soonest = None;
        for o in self.ops.values() {
            if o.done || o.cancelled {
                continue;
            }
            let Some(t) = o.pdl_trigger_ns else {
                continue;
            };
            if t <= self.clock {
                continue;
            }
            soonest = Some(soonest.map_or(t, |s: u64| s.min(t)));
        }
        soonest
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
                let pri = o.priority;
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
        let (device, launch, stream, preds) = {
            let Some(op) = self.ops.get(&id) else {
                return Err(SimError::Invalid { why: "unknown op" });
            };
            (op.device, op.launch, op.stream, op.preds.clone())
        };
        if self.unavailable.contains(&device) {
            return Err(SimError::Unavailable { device });
        }
        if self.cond_skip(&preds) {
            if let Some(op) = self.ops.get_mut(&id) {
                op.skipped = true;
            }
            self.running.push(Running {
                op: id,
                remaining_ns: 1,
                share: Share::Solo,
            });
            return Ok(true);
        }
        let Some(op) = self.ops.get(&id) else {
            return Err(SimError::Invalid { why: "unknown op" });
        };
        match &op.kind {
            Kind::Alloc { id: alloc, bytes } => self.start_alloc(id, device, *alloc, *bytes),
            Kind::Free { id: alloc } => self.start_free(id, device, *alloc),
            Kind::Kernel { .. } => self.start_kernel(id),
            Kind::Memset(_) => self.start_memset(id),
            Kind::HostFunc { .. } => {
                let ns = self.host_func_ns(device, launch)?;
                self.running.push(Running {
                    op: id,
                    remaining_ns: ns.max(1),
                    share: Share::Solo,
                });
                Ok(true)
            }
            Kind::Empty => {
                self.running.push(Running {
                    op: id,
                    remaining_ns: 1,
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
                let root = self.event_root(*event);
                if let Some(ev) = self.events.get_mut(&root) {
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
            Kind::If { .. } => Err(SimError::Invalid {
                why: "conditional if must be expanded",
            }),
            Kind::While { .. } => Err(SimError::Invalid {
                why: "conditional while must be expanded",
            }),
            Kind::Switch { .. } => Err(SimError::Invalid {
                why: "conditional switch must be expanded",
            }),
            Kind::WhileTick { .. } => {
                self.running.push(Running {
                    op: id,
                    remaining_ns: 1,
                    share: Share::Solo,
                });
                Ok(true)
            }
            Kind::SetConditional { handle, value } => {
                let handle = *handle;
                let value = *value;
                let c = self.conds.get_mut(&handle).ok_or(SimError::Invalid {
                    why: "unknown conditional",
                })?;
                c.value = value;
                self.running.push(Running {
                    op: id,
                    remaining_ns: 1,
                    share: Share::Solo,
                });
                Ok(true)
            }
            Kind::WriteValue { .. } | Kind::WaitValue { .. } => {
                let batch = batch_from_kind(&op.kind).ok_or(SimError::Invalid {
                    why: "batch mem op",
                })?;
                self.start_batch_mem_ops(id, device, &[batch])
            }
            Kind::BatchMem { ops } => {
                let ops = ops.clone();
                self.start_batch_mem_ops(id, device, &ops)
            }
            Kind::DeviceLaunch { .. } => self.start_device_launch(id, device),
        }
    }

    fn start_batch_mem_ops(
        &mut self,
        op: OpId,
        device: DeviceId,
        ops: &[BatchMemOp],
    ) -> Result<bool, SimError> {
        if !self.batch_mem_ready(device, ops)? {
            return Ok(false);
        }
        self.running.push(Running {
            op,
            remaining_ns: 1,
            share: Share::Solo,
        });
        Ok(true)
    }

    fn batch_mem_ready(&self, device: DeviceId, ops: &[BatchMemOp]) -> Result<bool, SimError> {
        let mut overlay: BTreeMap<(AllocId, u64), u64> = BTreeMap::new();
        for item in ops {
            let (alloc, offset, bits32) = match *item {
                BatchMemOp::Write {
                    id, offset, bits32, ..
                }
                | BatchMemOp::Wait {
                    id, offset, bits32, ..
                } => (id, offset, bits32),
            };
            self.require_wait_value_resident(device, alloc, offset, bits32)?;
            match *item {
                BatchMemOp::Write {
                    id,
                    offset,
                    value,
                    bits32,
                } => {
                    let mask = wait_value_mask(bits32);
                    let prev = overlay
                        .get(&(id, offset))
                        .copied()
                        .or_else(|| self.mailbox.get(&(id, offset)).copied())
                        .unwrap_or(0);
                    let _old = overlay.insert((id, offset), (prev & !mask) | (value & mask));
                }
                BatchMemOp::Wait {
                    id,
                    offset,
                    value,
                    bits32,
                    cmp,
                } => {
                    let mask = wait_value_mask(bits32);
                    let loc = overlay
                        .get(&(id, offset))
                        .copied()
                        .or_else(|| self.mailbox.get(&(id, offset)).copied())
                        .unwrap_or(0)
                        & mask;
                    if !cmp.matches(loc, value & mask) {
                        return Ok(false);
                    }
                }
            }
        }
        Ok(true)
    }

    fn require_wait_value_resident(
        &self,
        device: DeviceId,
        alloc: AllocId,
        offset: u64,
        bits32: bool,
    ) -> Result<(), SimError> {
        let width = wait_value_span(self.alloc_ref(alloc)?.bytes, offset, bits32)?;
        let buf = KernelBuf {
            id: alloc,
            offset,
            bytes: width,
        };
        if !self.buf_on_device(&buf, device, true, false)? {
            return Err(SimError::NotResident { alloc, device });
        }
        Ok(())
    }

    fn start_device_launch(&mut self, op: OpId, device: DeviceId) -> Result<bool, SimError> {
        if !self.take_compute(device)? {
            return Ok(false);
        }
        let ns = self.profile.gpu(device)?.graph_launch_ns.max(1);
        self.running.push(Running {
            op,
            remaining_ns: ns,
            share: Share::Solo,
        });
        Ok(true)
    }

    fn cond_skip(&self, preds: &[CondPred]) -> bool {
        preds.iter().any(|p| match p {
            CondPred::Nonzero(h) => self.conds.get(h).is_some_and(|c| c.value == 0),
            CondPred::Equals(h, branch) => self.conds.get(h).is_none_or(|c| c.value != *branch),
        })
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

    fn persist_kernel_bytes(
        &mut self,
        device: DeviceId,
        window: Option<AccessPolicyWindow>,
        kind: &KernelKind,
        reads: &[KernelBuf],
        writes: &[KernelBuf],
    ) -> Result<u64, SimError> {
        let (_, kind_bytes) = kind.flops_and_bytes();
        let Some(window) = window else {
            return Ok(kind_bytes);
        };
        self.validate_access_policy_window(device, window)?;
        if window.hit != AccessProperty::Persisting {
            return Ok(kind_bytes);
        }
        let persist_hit = u64::from(self.profile.gpu(device)?.l2_persist_hit_permille.min(1000));
        let total = self.alloc_ref(window.buf.id)?.bytes;
        let (off, n) = kernel_span(total, &window.buf)?;
        let touch = self.kernel_touch_spans(reads, writes)?;
        let overlap = span_overlap_with(window.buf.id, off, n, &touch);
        if overlap == 0 {
            return Ok(kind_bytes);
        }
        let rt = self.gpus.get_mut(&device).ok_or(SimError::Invalid {
            why: "unknown device",
        })?;
        if rt.persist_limit == 0 {
            return Ok(kind_bytes);
        }
        let want = overlap.saturating_mul(u64::from(window.hit_ratio_permille.min(1000))) / 1000;
        let cap = want.min(rt.persist_limit);
        let cached = persist_cached(rt, window.buf.id, off, n).min(overlap);
        let hit = cached.min(cap);
        let touch_bytes = touch
            .iter()
            .fold(0u64, |acc, s| acc.saturating_add(s.bytes));
        let billed = persist_discount(kind_bytes, hit, touch_bytes, persist_hit);
        persist_fill(rt, window.buf.id, off, cap);
        Ok(billed)
    }

    fn kernel_touch_spans(
        &self,
        reads: &[KernelBuf],
        writes: &[KernelBuf],
    ) -> Result<Vec<PersistLine>, SimError> {
        let mut spans = Vec::new();
        for b in reads.iter().chain(writes.iter()) {
            let total = self.alloc_ref(b.id)?.bytes;
            let (offset, bytes) = kernel_span(total, b)?;
            spans.push(PersistLine {
                id: b.id,
                offset,
                bytes,
            });
        }
        merge_persist_spans(&mut spans);
        Ok(spans)
    }

    fn kernel_ns(
        &self,
        device: DeviceId,
        stream: StreamId,
        kind: &KernelKind,
        launch: LaunchCost,
        mem_bps: u64,
        billed_bytes: u64,
    ) -> Result<u64, SimError> {
        let g = self.profile.gpu(device)?;
        let (flops, _) = kind.flops_and_bytes();
        let sm = u64::from(self.stream_sm_permille(device, stream));
        let peak = g
            .flops(kind.dtype())
            .saturating_mul(sm)
            .checked_div(1000)
            .unwrap_or(1)
            .max(1);
        let compute = ns_for_bytes(flops, peak);
        let memory = ns_for_bytes(billed_bytes, mem_bps.max(1));
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

    fn shared_mem_ns(
        &self,
        device: DeviceId,
        mode: SharedMemoryMode,
        ns: u64,
    ) -> Result<u64, SimError> {
        let g = self.profile.gpu(device)?;
        let permille = match mode {
            SharedMemoryMode::Default => return Ok(ns),
            SharedMemoryMode::FourByte => g.shared_mem_four_byte_permille,
            SharedMemoryMode::EightByte => g.shared_mem_eight_byte_permille,
        };
        Ok(scale_ns_permille(ns, permille))
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
        let root = self.event_root(event);
        if let Some(rec) = self.event_recorded_by(root) {
            let graph_rec = self.ops.get(&rec).is_some_and(|o| {
                skip_graph && matches!(o.launch, LaunchCost::GraphHead | LaunchCost::GraphBody)
            });
            if !graph_rec {
                if self.ops.get(&rec).is_some_and(|o| o.cancelled) {
                    let stream = self.ops.get(&rec).map(|o| o.stream).unwrap_or(StreamId(0));
                    return Err(SimError::Cancelled { stream, n: 1 });
                }
                if self.event_is_recorded(root) {
                    return Ok(Some(rec));
                }
                return Ok(None);
            }
        }
        let mut pending = false;
        let mut cancelled_n = 0u32;
        let mut stream = StreamId(0);
        for op in self.ops.values() {
            if let Kind::EventRecord { event: ev, .. } = &op.kind {
                if event_root_of(&self.events, *ev) == root {
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

    fn resolve_memset_op(&self, op: MemsetOp) -> Result<MemsetOp, SimError> {
        memset_2d_check(&op)?;
        let total = self.alloc_ref(op.id)?.bytes;
        if op.is_2d() || op.is_3d() {
            let span = op.extent_bytes();
            if span == 0 {
                return Err(SimError::Invalid {
                    why: "zero-byte memset",
                });
            }
            if op.offset.saturating_add(span) > total {
                return Err(SimError::Invalid {
                    why: "memset range past alloc",
                });
            }
            return Ok(op);
        }
        let buf = KernelBuf {
            id: op.id,
            offset: op.offset,
            bytes: op.bytes,
        };
        let (offset, bytes) = kernel_span(total, &buf)?;
        if bytes == 0 {
            return Err(SimError::Invalid {
                why: "zero-byte memset",
            });
        }
        Ok(MemsetOp {
            id: op.id,
            offset,
            bytes,
            ..MemsetOp::default()
        })
    }

    fn memcpy_precheck(&self, m: &MemcpyOp) -> Result<(), SimError> {
        memcpy_2d_check(m)?;
        let a = self.alloc_ref(m.alloc)?;
        if !a.live {
            return Err(SimError::UnknownAlloc { alloc: m.alloc });
        }
        let span = m.extent_bytes();
        if (a.vmm || m.offset > 0 || m.is_2d() || m.is_3d())
            && m.offset.saturating_add(span) > a.bytes
        {
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
                    vmm_covers(&a.vmm_maps, d, m.offset, span)
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
            if a.vmm && !vmm_covers(&a.vmm_maps, d, m.offset, span) {
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
            link.pageable_copy_ns(m.payload_bytes())
        } else {
            link.copy_ns(m.payload_bytes())
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

    fn validate_access_policy_window(
        &self,
        device: DeviceId,
        window: AccessPolicyWindow,
    ) -> Result<(), SimError> {
        validate_access_policy(window)?;
        let gran = self.gpu_rt(device)?.limits.l2_fetch;
        if gran <= 1 {
            return Ok(());
        }
        let total = self.alloc_ref(window.buf.id)?.bytes;
        let (off, n) = kernel_span(total, &window.buf)?;
        if !off.is_multiple_of(gran) || !n.is_multiple_of(gran) {
            return Err(SimError::Invalid {
                why: "access policy L2 fetch",
            });
        }
        Ok(())
    }

    fn validate_mem_sync_map(
        &self,
        device: DeviceId,
        map: MemSyncDomainMap,
    ) -> Result<(), SimError> {
        let count = self.mem_sync_domain_count(device)?;
        if map.default >= count || map.remote >= count {
            Err(SimError::Invalid {
                why: "mem sync domain",
            })
        } else {
            Ok(())
        }
    }

    fn snap_mem_sync(
        &self,
        device: DeviceId,
        stream: StreamId,
        kind: &Kind,
    ) -> Result<(MemSyncDomain, MemSyncDomainMap), SimError> {
        let map = match self.enqueue_mem_sync_map {
            Some(m) => m,
            None => self.stream_mem_sync_domain_map(device, stream)?,
        };
        self.validate_mem_sync_map(device, map)?;
        let domain =
            if matches!(kind, Kind::AllReduce { .. }) && self.enqueue_mem_sync_domain.is_none() {
                MemSyncDomain::Remote
            } else {
                self.enqueue_mem_sync_domain
                    .unwrap_or_else(|| self.stream_mem_sync_domain(device, stream))
            };
        Ok((domain, map))
    }

    fn apply_domain_fences(&mut self) {
        let finishing: Vec<OpId> = self
            .running
            .iter()
            .filter(|r| r.remaining_ns == 0)
            .map(|r| r.op)
            .collect();
        for id in finishing {
            let extra = self.same_domain_fence_extra(id);
            if extra == 0 {
                continue;
            }
            if let Some(r) = self.running.iter_mut().find(|r| r.op == id) {
                r.remaining_ns = extra;
            }
            if let Some(op) = self.ops.get_mut(&id) {
                op.domain_fence_paid = true;
            }
        }
    }

    fn same_domain_fence_extra(&self, id: OpId) -> u64 {
        let Some(op) = self.ops.get(&id) else {
            return 0;
        };
        if op.domain_fence_paid || !mem_sync_kind(&op.kind) {
            return 0;
        }
        let Ok(g) = self.profile.gpu(op.device) else {
            return 0;
        };
        let tax = u64::from(g.same_domain_fence_permille);
        if tax == 0 {
            return 0;
        }
        let phys = op.mem_sync_physical;
        let device = op.device;
        let mut leftover = 0u64;
        for r in &self.running {
            if r.op == id || r.remaining_ns == 0 {
                continue;
            }
            let Some(other) = self.ops.get(&r.op) else {
                continue;
            };
            if other.device != device || !mem_sync_kind(&other.kind) {
                continue;
            }
            if other.mem_sync_physical != phys {
                continue;
            }
            leftover = leftover.max(r.remaining_ns);
        }
        leftover.saturating_mul(tax) / 1000
    }

    fn cluster_block_count(&self, device: DeviceId, dim: ClusterDim) -> Result<u32, SimError> {
        let n = dim.blocks().ok_or(SimError::Invalid {
            why: "cluster dimension",
        })?;
        let gpu = self.profile.gpu(device)?;
        let max = u32::from(gpu.max_blocks_per_cluster.max(1));
        if n > max {
            return Err(SimError::Invalid {
                why: "cluster size",
            });
        }
        Ok(n)
    }

    fn validate_cluster(
        &self,
        device: DeviceId,
        dim: ClusterDim,
        mode: PortableClusterMode,
    ) -> Result<u32, SimError> {
        let n = self.cluster_block_count(device, dim)?;
        let portable = u32::from(self.profile.gpu(device)?.portable_cluster_size.max(1));
        let func = self.non_portable_cluster.contains(&device);
        if n > portable && !mode.allows_non_portable(func) {
            return Err(SimError::Invalid {
                why: "non-portable cluster",
            });
        }
        Ok(n)
    }

    fn validate_cluster_attrs(
        &self,
        device: DeviceId,
        cluster: Option<ClusterDim>,
        preferred: Option<ClusterDim>,
        mode: PortableClusterMode,
    ) -> Result<(), SimError> {
        if let Some(c) = cluster {
            let _n = self.validate_cluster(device, c, mode)?;
        }
        if let Some(p) = preferred {
            let Some(c) = cluster else {
                return Err(SimError::Invalid {
                    why: "preferred cluster",
                });
            };
            if !p.multiple_of(c) {
                return Err(SimError::Invalid {
                    why: "preferred cluster",
                });
            }
            let _n = self.validate_cluster(device, p, mode)?;
        }
        Ok(())
    }

    fn effective_cluster_blocks(
        &self,
        device: DeviceId,
        cluster: Option<ClusterDim>,
        preferred: Option<ClusterDim>,
    ) -> Result<u32, SimError> {
        if let Some(p) = preferred {
            let Some(c) = cluster else {
                return Err(SimError::Invalid {
                    why: "preferred cluster",
                });
            };
            if !p.multiple_of(c) {
                return Err(SimError::Invalid {
                    why: "preferred cluster",
                });
            }
        }
        let Some(c) = cluster else {
            return Ok(0);
        };
        let required = self.cluster_block_count(device, c)?;
        let Some(p) = preferred else {
            return Ok(required);
        };
        let preferred_n = self.cluster_block_count(device, p)?;
        let cap = u32::from(self.profile.gpu(device)?.compute_slots.max(1));
        if preferred_n <= cap {
            Ok(preferred_n)
        } else {
            Ok(required)
        }
    }

    /// `cudaFuncSetAttribute(..., cudaFuncAttributeNonPortableClusterSizeAllowed)`.
    ///
    /// Default is disallowed. A cluster larger than
    /// [`crate::GpuProfile::portable_cluster_size`] is Invalid until this is
    /// true, unless the launch uses [`PortableClusterMode::AllowNonPortable`].
    /// Decode identity stays disallowed.
    pub fn set_non_portable_cluster_size_allowed(
        &mut self,
        device: DeviceId,
        allowed: bool,
    ) -> Result<(), SimError> {
        let _gpu = self.profile.gpu(device)?;
        if allowed {
            let _ins = self.non_portable_cluster.insert(device);
        } else {
            let _rm = self.non_portable_cluster.remove(&device);
        }
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// Current [`Self::set_non_portable_cluster_size_allowed`] for `device`.
    #[must_use]
    pub fn non_portable_cluster_size_allowed(&self, device: DeviceId) -> bool {
        self.non_portable_cluster.contains(&device)
    }

    fn validate_dynamic_shared(
        &self,
        device: DeviceId,
        bytes: u32,
        mode: PortableSharedMode,
    ) -> Result<(), SimError> {
        if bytes == 0 {
            return Ok(());
        }
        let gpu = self.profile.gpu(device)?;
        let portable = gpu.max_shared_mem_per_block.max(1);
        let optin = gpu.max_shared_mem_per_block_optin.max(portable);
        if bytes > optin {
            return Err(SimError::Invalid {
                why: "dynamic shared",
            });
        }
        let func = self.max_dynamic_shared.get(&device).copied().unwrap_or(0);
        if !mode.allows_oversize(func, bytes, portable) {
            return Err(SimError::Invalid {
                why: "non-portable shared",
            });
        }
        Ok(())
    }

    /// `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize)`.
    ///
    /// Default `0` allows only [`crate::GpuProfile::max_shared_mem_per_block`].
    /// `bytes` above [`crate::GpuProfile::max_shared_mem_per_block_optin`] is
    /// Invalid. Decode identity stays `0`.
    pub fn set_max_dynamic_shared_memory(
        &mut self,
        device: DeviceId,
        bytes: u32,
    ) -> Result<(), SimError> {
        let optin = self
            .profile
            .gpu(device)?
            .max_shared_mem_per_block_optin
            .max(1);
        if bytes > optin {
            return Err(SimError::Invalid {
                why: "max dynamic shared",
            });
        }
        if bytes == 0 {
            let _rm = self.max_dynamic_shared.remove(&device);
        } else {
            let _prev = self.max_dynamic_shared.insert(device, bytes);
        }
        self.clock = self.clock.saturating_add(1);
        Ok(())
    }

    /// Current [`Self::set_max_dynamic_shared_memory`] for `device`.
    #[must_use]
    pub fn max_dynamic_shared_memory(&self, device: DeviceId) -> u32 {
        self.max_dynamic_shared.get(&device).copied().unwrap_or(0)
    }

    /// `cudaFuncGetAttributes` of modeled per-device function attrs.
    ///
    /// Query; legal during capture. Unknown devices are Invalid. This VM has
    /// one function-attr set per device, not per kernel function.
    pub fn func_get_attributes(&self, device: DeviceId) -> Result<FuncAttributes, SimError> {
        let _gpu = self.profile.gpu(device)?;
        Ok(FuncAttributes {
            max_dynamic_shared_size_bytes: self.max_dynamic_shared_memory(device),
            non_portable_cluster_size_allowed: self.non_portable_cluster_size_allowed(device),
        })
    }

    /// `cudaFuncSetAttribute`. Host-side; not a graph node.
    ///
    /// Dispatches [`FuncAttr`] onto the typed setters. Capture-legal like those
    /// setters. Negative [`FuncAttr::MaxDynamicSharedMemorySize`] or a
    /// non-0/1 [`FuncAttr::NonPortableClusterSizeAllowed`] is Invalid
    /// `"func attr"`. Typed helpers stay. Decode identity stays `0` / disallowed.
    pub fn func_set_attribute(
        &mut self,
        device: DeviceId,
        attr: FuncAttr,
        value: i32,
    ) -> Result<(), SimError> {
        match attr {
            FuncAttr::MaxDynamicSharedMemorySize => {
                let bytes =
                    u32::try_from(value).map_err(|_| SimError::Invalid { why: "func attr" })?;
                self.set_max_dynamic_shared_memory(device, bytes)
            }
            FuncAttr::NonPortableClusterSizeAllowed => {
                if value != 0 && value != 1 {
                    return Err(SimError::Invalid { why: "func attr" });
                }
                self.set_non_portable_cluster_size_allowed(device, value != 0)
            }
        }
    }

    /// `cudaFuncGetAttribute`. Query; legal during capture.
    ///
    /// Unknown devices are Invalid. This VM has one function-attr set per
    /// device, not per kernel function.
    pub fn func_get_attribute(&self, device: DeviceId, attr: FuncAttr) -> Result<i32, SimError> {
        let _gpu = self.profile.gpu(device)?;
        match attr {
            FuncAttr::MaxDynamicSharedMemorySize => {
                i32::try_from(self.max_dynamic_shared_memory(device))
                    .map_err(|_| SimError::Invalid { why: "func attr" })
            }
            FuncAttr::NonPortableClusterSizeAllowed => {
                Ok(i32::from(self.non_portable_cluster_size_allowed(device)))
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
        if let Some(wake) = self.next_pdl_wake() {
            let dt = wake.saturating_sub(self.clock);
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
        }
        self.apply_domain_fences();
        for r in &self.running {
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
        if self.ops.get(&id).is_some_and(|o| o.skipped) {
            if let Some(op) = self.ops.get_mut(&id) {
                op.done = true;
                op.done_ns = Some(self.clock);
            }
            return Ok(());
        }
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
            Kind::Memset(op) => Some((vec![op.id], false)),
            Kind::AllReduce { parts, .. } => Some((parts.iter().map(|(_, a)| *a).collect(), false)),
            Kind::DeviceLaunch { .. } => Some((Vec::new(), false)),
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
        if let Some((ids, _)) = kernel_work {
            let n = {
                let op = self
                    .ops
                    .get(&id)
                    .ok_or(SimError::Invalid { why: "unknown op" })?;
                self.op_kernel_slots(op)?
            };
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
        let writes: Vec<(AllocId, u64, u64, bool)> = self
            .ops
            .get(&id)
            .map(|op| mailbox_writes(&op.kind))
            .unwrap_or_default();
        for (alloc, offset, value, bits32) in writes {
            self.apply_mailbox_write(alloc, offset, value, bits32);
        }
        if let Some(op) = self.ops.get_mut(&id) {
            op.done = true;
            op.done_ns = Some(self.clock);
        }
        self.bump_graph_mem_from_op(id)?;
        self.finish_device_launch(id)?;
        self.continue_while(id)?;
        Ok(())
    }

    fn finish_device_launch(&mut self, id: OpId) -> Result<(), SimError> {
        let (graph, stream) = {
            let Some(op) = self.ops.get(&id) else {
                return Ok(());
            };
            let Kind::DeviceLaunch { graph } = &op.kind else {
                return Ok(());
            };
            (*graph, op.stream)
        };
        self.reset_graph_tree_conds(graph)?;
        let mut stack = BTreeSet::new();
        let wait = [id];
        let n = self.enqueue_graph(graph, stream, false, &mut stack, &wait)?;
        let tail = if n == 0 {
            id
        } else {
            let device = self
                .ops
                .get(&id)
                .ok_or(SimError::Invalid { why: "unknown op" })?
                .device;
            self.tail.get(&(device, stream)).copied().unwrap_or(id)
        };
        self.graphs
            .get_mut(&graph)
            .ok_or(SimError::Invalid {
                why: "unknown graph",
            })?
            .device_launch_tail = Some(tail);
        Ok(())
    }

    fn bump_graph_mem_from_op(&mut self, id: OpId) -> Result<(), SimError> {
        let Some(op) = self.ops.get(&id) else {
            return Ok(());
        };
        let alloc = match &op.kind {
            Kind::Alloc { id, .. } | Kind::Free { id } => *id,
            _ => return Ok(()),
        };
        if !self.is_graph_alloc(alloc) {
            return Ok(());
        }
        self.bump_graph_mem_high(op.device)
    }

    fn continue_while(&mut self, id: OpId) -> Result<(), SimError> {
        let (handle, body, iter, device, stream) = {
            let Some(op) = self.ops.get(&id) else {
                return Ok(());
            };
            let Kind::WhileTick { handle, body, iter } = &op.kind else {
                return Ok(());
            };
            (*handle, *body, *iter, op.device, op.stream)
        };
        let value = self
            .conds
            .get(&handle)
            .ok_or(SimError::Invalid {
                why: "unknown conditional",
            })?
            .value;
        if value == 0 {
            return Ok(());
        }
        if iter >= 64 {
            return Err(SimError::Invalid {
                why: "while iteration cap",
            });
        }
        let wait = [id];
        let add = self.enqueue_pred_graph(
            CondPred::Nonzero(handle),
            body,
            stream,
            false,
            &mut BTreeSet::new(),
            &wait,
        )?;
        let body_tail = self.tail.get(&(device, stream)).copied();
        self.enqueue_preds.push(CondPred::Nonzero(handle));
        let tick = self.submit_launch(
            device,
            stream,
            Kind::WhileTick {
                handle,
                body,
                iter: iter.saturating_add(1),
            },
            LaunchCost::GraphBody,
        );
        let _pop = self.enqueue_preds.pop();
        let tick = tick?;
        self.add_op_dep(tick, id);
        if let Some(t) = body_tail {
            if t != tick {
                self.add_op_dep(tick, t);
            }
        }
        let _n = add;
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

fn memcpy_2d_check(m: &MemcpyOp) -> Result<(), SimError> {
    if m.is_3d() {
        if m.bytes == 0 {
            return Err(SimError::Invalid {
                why: "memcpy3d width",
            });
        }
        if m.height == 0 {
            return Err(SimError::Invalid {
                why: "memcpy3d height",
            });
        }
        if m.bytes > m.src_pitch_or_width() || m.bytes > m.dst_pitch_or_width() {
            return Err(SimError::Invalid {
                why: "memcpy3d pitch",
            });
        }
        if m.height > m.src_height_or_extent() || m.height > m.dst_height_or_extent() {
            return Err(SimError::Invalid {
                why: "memcpy3d height",
            });
        }
        return Ok(());
    }
    if !m.is_2d() {
        return Ok(());
    }
    if m.bytes == 0 {
        return Err(SimError::Invalid {
            why: "memcpy2d width",
        });
    }
    if m.bytes > m.src_pitch_or_width() || m.bytes > m.dst_pitch_or_width() {
        return Err(SimError::Invalid {
            why: "memcpy2d pitch",
        });
    }
    Ok(())
}

fn memset_2d_check(op: &MemsetOp) -> Result<(), SimError> {
    if op.is_3d() {
        if op.bytes == 0 {
            return Err(SimError::Invalid {
                why: "memset3d width",
            });
        }
        if op.height == 0 {
            return Err(SimError::Invalid {
                why: "memset3d height",
            });
        }
        if op.bytes > op.pitch_or_width() {
            return Err(SimError::Invalid {
                why: "memset3d pitch",
            });
        }
        if op.height > op.ysize_or_extent() {
            return Err(SimError::Invalid {
                why: "memset3d height",
            });
        }
        return Ok(());
    }
    if !op.is_2d() {
        return Ok(());
    }
    if op.bytes == 0 {
        return Err(SimError::Invalid {
            why: "memset2d width",
        });
    }
    if op.bytes > op.pitch_or_width() {
        return Err(SimError::Invalid {
            why: "memset2d pitch",
        });
    }
    Ok(())
}

fn validate_access_policy(window: AccessPolicyWindow) -> Result<(), SimError> {
    if window.hit_ratio_permille > 1000 {
        return Err(SimError::Invalid {
            why: "access policy hit ratio",
        });
    }
    if window.miss == AccessProperty::Persisting {
        return Err(SimError::Invalid {
            why: "access policy miss persisting",
        });
    }
    Ok(())
}

fn range_overlap(a0: u64, an: u64, b0: u64, bn: u64) -> u64 {
    let a1 = a0.saturating_add(an);
    let b1 = b0.saturating_add(bn);
    let lo = a0.max(b0);
    let hi = a1.min(b1);
    hi.saturating_sub(lo)
}

fn span_overlap_with(id: AllocId, off: u64, n: u64, spans: &[PersistLine]) -> u64 {
    spans.iter().filter(|s| s.id == id).fold(0u64, |acc, s| {
        acc.saturating_add(range_overlap(off, n, s.offset, s.bytes))
    })
}

fn persist_cached(rt: &GpuRt, id: AllocId, off: u64, n: u64) -> u64 {
    span_overlap_with(id, off, n, &rt.persist_lines)
}

fn persist_used(rt: &GpuRt) -> u64 {
    rt.persist_lines
        .iter()
        .fold(0u64, |acc, l| acc.saturating_add(l.bytes))
}

fn persist_trim(rt: &mut GpuRt) {
    while persist_used(rt) > rt.persist_limit && !rt.persist_lines.is_empty() {
        let _gone = rt.persist_lines.remove(0);
    }
}

fn persist_fill(rt: &mut GpuRt, id: AllocId, off: u64, n: u64) {
    if n == 0 || rt.persist_limit == 0 {
        return;
    }
    let n = n.min(rt.persist_limit);
    rt.persist_lines
        .retain(|l| l.id != id || range_overlap(off, n, l.offset, l.bytes) == 0);
    persist_trim(rt);
    while persist_used(rt).saturating_add(n) > rt.persist_limit && !rt.persist_lines.is_empty() {
        let _gone = rt.persist_lines.remove(0);
    }
    if persist_used(rt).saturating_add(n) <= rt.persist_limit {
        rt.persist_lines.push(PersistLine {
            id,
            offset: off,
            bytes: n,
        });
    }
}

fn persist_discount(kind_bytes: u64, hit: u64, touch: u64, persist_permille: u64) -> u64 {
    if hit == 0 || touch == 0 {
        return kind_bytes;
    }
    let frac = hit.saturating_mul(1000) / touch;
    let discount = frac.saturating_mul(persist_permille) / 1000;
    kind_bytes.saturating_mul(1000u64.saturating_sub(discount.min(1000))) / 1000
}

fn merge_persist_spans(spans: &mut Vec<PersistLine>) {
    spans.sort_by_key(|s| (s.id.0, s.offset));
    let mut out: Vec<PersistLine> = Vec::new();
    for s in spans.drain(..) {
        if let Some(last) = out.last_mut() {
            if last.id == s.id && last.offset.saturating_add(last.bytes) >= s.offset {
                let end = last
                    .offset
                    .saturating_add(last.bytes)
                    .max(s.offset.saturating_add(s.bytes));
                last.bytes = end.saturating_sub(last.offset);
                continue;
            }
        }
        out.push(s);
    }
    *spans = out;
}

fn wait_value_mask(bits32: bool) -> u64 {
    if bits32 {
        0xFFFF_FFFF
    } else {
        u64::MAX
    }
}

fn wait_value_span(total: u64, offset: u64, bits32: bool) -> Result<u64, SimError> {
    let width = if bits32 { 4 } else { 8 };
    if !offset.is_multiple_of(width) {
        return Err(SimError::Invalid {
            why: "wait-value alignment",
        });
    }
    if offset.saturating_add(width) > total {
        return Err(SimError::Invalid {
            why: "wait-value span",
        });
    }
    Ok(width)
}

fn mem_sync_kind(kind: &Kind) -> bool {
    matches!(kind, Kind::Kernel { .. } | Kind::AllReduce { .. })
}

fn device_launch_refused(kind: &Kind) -> bool {
    matches!(
        kind,
        Kind::Alloc { .. }
            | Kind::Free { .. }
            | Kind::EventRecord { .. }
            | Kind::EventWait { .. }
            | Kind::ChildGraph { .. }
            | Kind::If { .. }
            | Kind::While { .. }
            | Kind::Switch { .. }
            | Kind::WhileTick { .. }
            | Kind::Attach { .. }
            | Kind::AllReduce { .. }
            | Kind::HostFunc { .. }
            | Kind::DeviceLaunch { .. }
    )
}

fn kind_from_batch(op: BatchMemOp) -> Kind {
    match op {
        BatchMemOp::Write {
            id,
            offset,
            value,
            bits32,
        } => Kind::WriteValue {
            id,
            offset,
            value,
            bits32,
        },
        BatchMemOp::Wait {
            id,
            offset,
            value,
            bits32,
            cmp,
        } => Kind::WaitValue {
            id,
            offset,
            value,
            bits32,
            cmp,
        },
    }
}

fn batch_from_kind(kind: &Kind) -> Option<BatchMemOp> {
    match *kind {
        Kind::WriteValue {
            id,
            offset,
            value,
            bits32,
        } => Some(BatchMemOp::Write {
            id,
            offset,
            value,
            bits32,
        }),
        Kind::WaitValue {
            id,
            offset,
            value,
            bits32,
            cmp,
        } => Some(BatchMemOp::Wait {
            id,
            offset,
            value,
            bits32,
            cmp,
        }),
        _ => None,
    }
}

fn batch_set_params_ok(step: &Kind, op: BatchMemOp) -> Result<(), SimError> {
    match (step, op) {
        (Kind::WriteValue { bits32: a, .. }, BatchMemOp::Write { bits32: b, .. }) if *a == b => {
            Ok(())
        }
        (
            Kind::WaitValue {
                bits32: a, cmp: c, ..
            },
            BatchMemOp::Wait {
                bits32: b, cmp: d, ..
            },
        ) if *a == b && *c == d => Ok(()),
        (Kind::WriteValue { .. }, _) => Err(SimError::Invalid {
            why: "not a write-value node",
        }),
        (Kind::WaitValue { .. }, _) => Err(SimError::Invalid {
            why: "not a wait-value node",
        }),
        _ => Err(SimError::Invalid {
            why: "not a batch mem op node",
        }),
    }
}

fn batch_ops_set_params_kind(step: &Kind, ops: &[BatchMemOp]) -> Result<Kind, SimError> {
    match step {
        Kind::BatchMem { .. } => Ok(Kind::BatchMem { ops: ops.to_vec() }),
        Kind::WriteValue { .. } | Kind::WaitValue { .. } => {
            if ops.len() != 1 {
                return Err(SimError::Invalid {
                    why: "batch mem op length",
                });
            }
            let op = ops.first().copied().ok_or(SimError::Invalid {
                why: "empty batch mem op",
            })?;
            batch_set_params_ok(step, op)?;
            Ok(kind_from_batch(op))
        }
        _ => Err(SimError::Invalid {
            why: "not a batch mem op node",
        }),
    }
}

fn batch_items(kind: &Kind) -> Option<Vec<BatchMemOp>> {
    match kind {
        Kind::BatchMem { ops } => Some(ops.clone()),
        Kind::WriteValue { .. } | Kind::WaitValue { .. } => {
            batch_from_kind(kind).map(|op| vec![op])
        }
        _ => None,
    }
}

fn kernel_params_of(kind: &Kind) -> Result<KernelNodeParams, SimError> {
    let Kind::Kernel {
        kind,
        reads,
        writes,
        cooperative,
    } = kind
    else {
        return Err(SimError::Invalid {
            why: "not a kernel node",
        });
    };
    Ok(KernelNodeParams {
        kind: kind.clone(),
        reads: reads.clone(),
        writes: writes.clone(),
        cooperative: *cooperative,
    })
}

fn memcpy_params_of(kind: &Kind) -> Result<MemcpyOp, SimError> {
    let Kind::Memcpy(op) = kind else {
        return Err(SimError::Invalid {
            why: "not a memcpy node",
        });
    };
    Ok(op.clone())
}

fn memset_params_of(kind: &Kind) -> Result<MemsetOp, SimError> {
    let Kind::Memset(op) = kind else {
        return Err(SimError::Invalid {
            why: "not a memset node",
        });
    };
    Ok(*op)
}

fn host_params_of(kind: &Kind) -> Result<HostNodeParams, SimError> {
    let Kind::HostFunc { fn_id, user_data } = kind else {
        return Err(SimError::Invalid {
            why: "not a host node",
        });
    };
    Ok(HostNodeParams {
        fn_id: *fn_id,
        user_data: *user_data,
    })
}

fn node_params_of(kind: &Kind) -> Result<GraphNodeParams, SimError> {
    Ok(match kind {
        Kind::Kernel { .. } => GraphNodeParams::Kernel(kernel_params_of(kind)?),
        Kind::Memcpy(_) => GraphNodeParams::Memcpy(memcpy_params_of(kind)?),
        Kind::Memset(_) => GraphNodeParams::Memset(memset_params_of(kind)?),
        Kind::HostFunc { .. } => GraphNodeParams::Host(host_params_of(kind)?),
        Kind::Empty => GraphNodeParams::Empty,
        Kind::EventRecord { event, external } => GraphNodeParams::EventRecord {
            event: *event,
            external: *external,
        },
        Kind::EventWait { event, external } => GraphNodeParams::EventWait {
            event: *event,
            external: *external,
        },
        Kind::ChildGraph { graph } => GraphNodeParams::ChildGraph(*graph),
        Kind::Alloc { bytes, .. } => GraphNodeParams::Alloc { bytes: *bytes },
        Kind::Free { id } => GraphNodeParams::Free(*id),
        Kind::BatchMem { .. } | Kind::WriteValue { .. } | Kind::WaitValue { .. } => {
            GraphNodeParams::BatchMemOp(batch_items(kind).ok_or(SimError::Invalid {
                why: "not a batch mem op node",
            })?)
        }
        Kind::If { .. }
        | Kind::While { .. }
        | Kind::Switch { .. }
        | Kind::SetConditional { .. }
        | Kind::WhileTick { .. }
        | Kind::Attach { .. }
        | Kind::AllReduce { .. }
        | Kind::DeviceLaunch { .. } => {
            return Err(SimError::Invalid {
                why: "not a graph node params kind",
            });
        }
    })
}

fn mailbox_writes(kind: &Kind) -> Vec<(AllocId, u64, u64, bool)> {
    match kind {
        Kind::WriteValue {
            id,
            offset,
            value,
            bits32,
        } => vec![(*id, *offset, *value, *bits32)],
        Kind::BatchMem { ops } => ops
            .iter()
            .filter_map(|item| match *item {
                BatchMemOp::Write {
                    id,
                    offset,
                    value,
                    bits32,
                } => Some((id, offset, value, bits32)),
                BatchMemOp::Wait { .. } => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn remap_batch_op(op: BatchMemOp, map: &BTreeMap<AllocId, AllocId>) -> BatchMemOp {
    match op {
        BatchMemOp::Write {
            id,
            offset,
            value,
            bits32,
        } => BatchMemOp::Write {
            id: remap_alloc_id(id, map),
            offset,
            value,
            bits32,
        },
        BatchMemOp::Wait {
            id,
            offset,
            value,
            bits32,
            cmp,
        } => BatchMemOp::Wait {
            id: remap_alloc_id(id, map),
            offset,
            value,
            bits32,
            cmp,
        },
    }
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
            height: op.height,
            src_pitch: op.src_pitch,
            dst_pitch: op.dst_pitch,
            depth: op.depth,
            src_height: op.src_height,
            dst_height: op.dst_height,
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
        Kind::Memset(op) => Kind::Memset(MemsetOp {
            id: remap_alloc_id(op.id, map),
            offset: op.offset,
            bytes: op.bytes,
            height: op.height,
            pitch: op.pitch,
            depth: op.depth,
            ysize: op.ysize,
        }),
        Kind::Attach { id, flags } => Kind::Attach {
            id: remap_alloc_id(id, map),
            flags,
        },
        Kind::WriteValue {
            id,
            offset,
            value,
            bits32,
        } => Kind::WriteValue {
            id: remap_alloc_id(id, map),
            offset,
            value,
            bits32,
        },
        Kind::WaitValue {
            id,
            offset,
            value,
            bits32,
            cmp,
        } => Kind::WaitValue {
            id: remap_alloc_id(id, map),
            offset,
            value,
            bits32,
            cmp,
        },
        Kind::BatchMem { ops } => Kind::BatchMem {
            ops: ops.into_iter().map(|op| remap_batch_op(op, map)).collect(),
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

fn remap_nested_graphs(
    steps: &[GraphStep],
    remap: &BTreeMap<GraphId, GraphId>,
) -> Result<Vec<GraphStep>, SimError> {
    let mut out = Vec::with_capacity(steps.len());
    for step in steps {
        let kind = match &step.kind {
            Kind::ChildGraph { graph: child } => {
                let cloned = remap.get(child).copied().ok_or(SimError::Invalid {
                    why: "unknown graph",
                })?;
                Kind::ChildGraph { graph: cloned }
            }
            Kind::If { handle, body } => {
                let cloned = remap.get(body).copied().ok_or(SimError::Invalid {
                    why: "unknown graph",
                })?;
                Kind::If {
                    handle: *handle,
                    body: cloned,
                }
            }
            Kind::While { handle, body } => {
                let cloned = remap.get(body).copied().ok_or(SimError::Invalid {
                    why: "unknown graph",
                })?;
                Kind::While {
                    handle: *handle,
                    body: cloned,
                }
            }
            Kind::Switch { handle, bodies } => {
                let mut cloned = Vec::with_capacity(bodies.len());
                for b in bodies {
                    cloned.push(remap.get(b).copied().ok_or(SimError::Invalid {
                        why: "unknown graph",
                    })?);
                }
                Kind::Switch {
                    handle: *handle,
                    bodies: cloned,
                }
            }
            other => other.clone(),
        };
        out.push(GraphStep {
            device: step.device,
            stream: step.stream,
            kind,
            deps: step.deps.clone(),
            enabled: step.enabled,
            destroyed: step.destroyed,
            priority: step.priority,
            pdl: step.pdl,
            programmatic_event: step.programmatic_event,
            launch_completion: step.launch_completion,
            access_policy: step.access_policy,
            mem_sync_domain: step.mem_sync_domain,
            mem_sync_map: step.mem_sync_map,
            cluster: step.cluster,
            cluster_policy: step.cluster_policy,
            preferred_cluster: step.preferred_cluster,
            carveout: step.carveout,
            device_updatable: step.device_updatable,
            shared_mem: step.shared_mem,
            portable_cluster: step.portable_cluster,
            dynamic_shared: step.dynamic_shared,
            portable_shared: step.portable_shared,
            nvlink_util_centric: step.nvlink_util_centric,
        });
    }
    Ok(out)
}

fn live_ok(step: &GraphStep) -> Result<&GraphStep, SimError> {
    if step.destroyed {
        Err(SimError::Invalid {
            why: "unknown graph node",
        })
    } else {
        Ok(step)
    }
}

fn live_ok_mut(step: &mut GraphStep) -> Result<&mut GraphStep, SimError> {
    if step.destroyed {
        Err(SimError::Invalid {
            why: "unknown graph node",
        })
    } else {
        Ok(step)
    }
}

struct GraphTopologyDiff {
    result: GraphExecUpdateResult,
    error_node: Option<usize>,
    error_from_node: Option<usize>,
}

fn graph_topology_diff(exec: &[GraphStep], src: &[GraphStep]) -> Option<GraphTopologyDiff> {
    if exec.len() != src.len() {
        let extra = exec.len().min(src.len());
        return Some(GraphTopologyDiff {
            result: GraphExecUpdateResult::TopologyChanged,
            error_node: (src.len() > exec.len()).then_some(extra),
            error_from_node: (exec.len() > src.len()).then_some(extra),
        });
    }
    for (i, (x, y)) in exec.iter().zip(src.iter()).enumerate() {
        if x.destroyed != y.destroyed {
            return Some(GraphTopologyDiff {
                result: GraphExecUpdateResult::TopologyChanged,
                error_node: Some(i),
                error_from_node: Some(i),
            });
        }
        if x.destroyed {
            continue;
        }
        if node_kind(&x.kind) != node_kind(&y.kind) {
            return Some(GraphTopologyDiff {
                result: GraphExecUpdateResult::NodeTypeChanged,
                error_node: Some(i),
                error_from_node: Some(i),
            });
        }
        if x.device != y.device || x.stream != y.stream {
            return Some(GraphTopologyDiff {
                result: GraphExecUpdateResult::TopologyChanged,
                error_node: Some(i),
                error_from_node: Some(i),
            });
        }
        if x.deps != y.deps {
            return Some(dep_mismatch(&x.deps, &y.deps, i));
        }
        if !op_eq(&x.kind, &y.kind) {
            return Some(op_mismatch(&x.kind, &y.kind, i));
        }
    }
    None
}

fn dep_mismatch(exec_deps: &[usize], src_deps: &[usize], to: usize) -> GraphTopologyDiff {
    let n = exec_deps.len().max(src_deps.len());
    for k in 0..n {
        let e = exec_deps.get(k);
        let s = src_deps.get(k);
        if e != s {
            return GraphTopologyDiff {
                result: GraphExecUpdateResult::DependenciesChanged,
                error_node: Some(to),
                error_from_node: s.or(e).copied(),
            };
        }
    }
    GraphTopologyDiff {
        result: GraphExecUpdateResult::DependenciesChanged,
        error_node: Some(to),
        error_from_node: None,
    }
}

fn op_mismatch(exec_kind: &Kind, src_kind: &Kind, i: usize) -> GraphTopologyDiff {
    let result = match (exec_kind, src_kind) {
        (Kind::Kernel { cooperative: a, .. }, Kind::Kernel { cooperative: b, .. }) if a != b => {
            GraphExecUpdateResult::ParametersChanged
        }
        (Kind::ChildGraph { .. }, Kind::ChildGraph { .. })
        | (Kind::If { .. }, Kind::If { .. })
        | (Kind::While { .. }, Kind::While { .. })
        | (Kind::Switch { .. }, Kind::Switch { .. })
        | (Kind::SetConditional { .. }, Kind::SetConditional { .. })
        | (Kind::WhileTick { .. }, Kind::WhileTick { .. }) => {
            GraphExecUpdateResult::TopologyChanged
        }
        (Kind::EventRecord { .. }, Kind::EventRecord { .. })
        | (Kind::EventWait { .. }, Kind::EventWait { .. }) => {
            GraphExecUpdateResult::AttributesChanged
        }
        _ => GraphExecUpdateResult::ParametersChanged,
    };
    GraphTopologyDiff {
        result,
        error_node: Some(i),
        error_from_node: Some(i),
    }
}

fn first_mem_node(steps: &[GraphStep]) -> Option<usize> {
    steps
        .iter()
        .position(|s| matches!(s.kind, Kind::Alloc { .. } | Kind::Free { .. }))
}

fn update_report(
    info: &mut GraphExecUpdateResultInfo,
    result: GraphExecUpdateResult,
    error_node: Option<usize>,
    error_from_node: Option<usize>,
    why: &'static str,
) -> Result<(), SimError> {
    info.result = result;
    info.error_node = error_node;
    info.error_from_node = error_from_node;
    Err(SimError::Invalid { why })
}

/// Topology for [`Sim::graph_exec_child_set_params`]: nested child-graph ids are
/// parameters, not topology (unlike [`graph_topology_diff`] / `cudaGraphExecUpdate`).
fn child_param_topology_eq(a: &[GraphStep], b: &[GraphStep]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| {
        x.device == y.device
            && x.stream == y.stream
            && child_param_op_eq(&x.kind, &y.kind)
            && x.deps == y.deps
    })
}

fn child_param_op_eq(a: &Kind, b: &Kind) -> bool {
    if matches!((a, b), (Kind::ChildGraph { .. }, Kind::ChildGraph { .. })) {
        return true;
    }
    op_eq(a, b)
}

fn debug_dot_place(p: Place) -> String {
    match p {
        Place::Host => String::from("Host"),
        Place::HostPinned => String::from("HostPinned"),
        Place::Device(d) => {
            let mut s = String::from("D");
            s.push_str(&d.0.to_string());
            s
        }
    }
}

fn debug_dot_bufs(bufs: &[KernelBuf]) -> String {
    let mut s = String::new();
    for (i, b) in bufs.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&b.id.0.to_string());
    }
    s
}

fn debug_dot_label(i: usize, step: &GraphStep, flags: u32) -> String {
    let mut label = i.to_string();
    label.push(' ');
    label.push_str(&format!("{:?}", node_kind(&step.kind)));
    match &step.kind {
        Kind::Kernel {
            cooperative,
            reads,
            writes,
            ..
        } => {
            if flags & GraphDebugDotFlags::KERNEL_NODE_PARAMS != 0 {
                label.push_str(" coop=");
                label.push_str(&u8::from(*cooperative).to_string());
                label.push_str(" r=");
                label.push_str(&debug_dot_bufs(reads));
                label.push_str(" w=");
                label.push_str(&debug_dot_bufs(writes));
            }
            if flags & GraphDebugDotFlags::KERNEL_NODE_ATTRIBUTES != 0 && step.priority != 0 {
                label.push_str(" pri=");
                label.push_str(&step.priority.to_string());
            }
        }
        Kind::Memcpy(op) => {
            if flags & GraphDebugDotFlags::MEMCPY_NODE_PARAMS != 0 {
                label.push_str(" bytes=");
                label.push_str(&op.bytes.to_string());
                label.push_str(" src=");
                label.push_str(&debug_dot_place(op.src));
                label.push_str(" dst=");
                label.push_str(&debug_dot_place(op.dst));
            }
        }
        Kind::Memset(op) => {
            if flags & GraphDebugDotFlags::MEMSET_NODE_PARAMS != 0 {
                label.push_str(" bytes=");
                label.push_str(&op.bytes.to_string());
                label.push_str(" id=");
                label.push_str(&op.id.0.to_string());
            }
        }
        Kind::HostFunc { fn_id, user_data } => {
            if flags & GraphDebugDotFlags::HOST_NODE_PARAMS != 0 {
                label.push_str(" fn=");
                label.push_str(&fn_id.to_string());
                label.push_str(" data=");
                label.push_str(&user_data.to_string());
            }
        }
        Kind::EventRecord { event, external } | Kind::EventWait { event, external } => {
            if flags & GraphDebugDotFlags::EVENT_NODE_PARAMS != 0 {
                label.push_str(" ev=");
                label.push_str(&event.0.to_string());
                if *external {
                    label.push_str(" ext");
                }
            }
        }
        Kind::Alloc { id, bytes } => {
            if flags & GraphDebugDotFlags::MEM_ALLOC_NODE_PARAMS != 0 {
                label.push_str(" id=");
                label.push_str(&id.0.to_string());
                label.push_str(" bytes=");
                label.push_str(&bytes.to_string());
            }
        }
        Kind::Free { id } => {
            if flags & GraphDebugDotFlags::MEM_FREE_NODE_PARAMS != 0 {
                label.push_str(" id=");
                label.push_str(&id.0.to_string());
            }
        }
        Kind::WriteValue { id, .. } | Kind::WaitValue { id, .. } => {
            if flags & GraphDebugDotFlags::BATCH_MEM_OP_NODE_PARAMS != 0 {
                label.push_str(" id=");
                label.push_str(&id.0.to_string());
                label.push_str(" n=1");
            }
        }
        Kind::BatchMem { ops } => {
            if flags & GraphDebugDotFlags::BATCH_MEM_OP_NODE_PARAMS != 0 {
                label.push_str(" n=");
                label.push_str(&ops.len().to_string());
            }
        }
        Kind::If { handle, body } | Kind::While { handle, body } => {
            if flags & GraphDebugDotFlags::CONDITIONAL_NODE_PARAMS != 0 {
                label.push_str(" h=");
                label.push_str(&handle.0.to_string());
                label.push_str(" body=");
                label.push_str(&body.0.to_string());
            }
        }
        Kind::Switch { handle, bodies } => {
            if flags & GraphDebugDotFlags::CONDITIONAL_NODE_PARAMS != 0 {
                label.push_str(" h=");
                label.push_str(&handle.0.to_string());
                label.push_str(" n=");
                label.push_str(&bodies.len().to_string());
            }
        }
        Kind::SetConditional { handle, value } => {
            if flags & GraphDebugDotFlags::CONDITIONAL_NODE_PARAMS != 0 {
                label.push_str(" h=");
                label.push_str(&handle.0.to_string());
                label.push_str(" v=");
                label.push_str(&value.to_string());
            }
        }
        Kind::ChildGraph { graph } if flags & GraphDebugDotFlags::HANDLES != 0 => {
            label.push_str(" g=");
            label.push_str(&graph.0.to_string());
        }
        _ => {}
    }
    label
}

fn node_kind(kind: &Kind) -> GraphNodeKind {
    match kind {
        Kind::Kernel { .. } => GraphNodeKind::Kernel,
        Kind::Memcpy(_) => GraphNodeKind::Memcpy,
        Kind::Memset(_) => GraphNodeKind::Memset,
        Kind::HostFunc { .. } => GraphNodeKind::Host,
        Kind::Empty => GraphNodeKind::Empty,
        Kind::EventRecord { .. } => GraphNodeKind::EventRecord,
        Kind::EventWait { .. } => GraphNodeKind::EventWait,
        Kind::ChildGraph { .. } => GraphNodeKind::ChildGraph,
        Kind::Alloc { .. } => GraphNodeKind::Alloc,
        Kind::Free { .. } => GraphNodeKind::Free,
        Kind::AllReduce { .. } => GraphNodeKind::AllReduce,
        Kind::Attach { .. } => GraphNodeKind::Attach,
        Kind::If { .. } => GraphNodeKind::If,
        Kind::SetConditional { .. } => GraphNodeKind::SetConditional,
        Kind::While { .. } => GraphNodeKind::While,
        Kind::WhileTick { .. } => GraphNodeKind::WhileTick,
        Kind::Switch { .. } => GraphNodeKind::Switch,
        Kind::WriteValue { .. } | Kind::WaitValue { .. } | Kind::BatchMem { .. } => {
            GraphNodeKind::BatchMemOp
        }
        Kind::DeviceLaunch { .. } => GraphNodeKind::DeviceLaunch,
    }
}

fn nested_graphs(kind: &Kind) -> Vec<GraphId> {
    match kind {
        Kind::ChildGraph { graph }
        | Kind::If { body: graph, .. }
        | Kind::While { body: graph, .. } => {
            vec![*graph]
        }
        Kind::Switch { bodies, .. } => bodies.clone(),
        _ => Vec::new(),
    }
}

fn cond_body_graphs(kind: &Kind) -> Vec<GraphId> {
    match kind {
        Kind::If { body, .. } | Kind::While { body, .. } => vec![*body],
        Kind::Switch { bodies, .. } => bodies.clone(),
        _ => Vec::new(),
    }
}

fn event_root_of(events: &BTreeMap<EventId, Ev>, event: EventId) -> EventId {
    events.get(&event).and_then(|e| e.ipc_src).unwrap_or(event)
}

fn capture_step_deps(
    buf: &[GraphStep],
    device: DeviceId,
    stream: StreamId,
    kind: &Kind,
    events: &BTreeMap<EventId, Ev>,
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
            let wait = event_root_of(events, *event);
            if let Some(i) = buf.iter().rposition(|s| {
                matches!(
                    &s.kind,
                    Kind::EventRecord {
                        event: e,
                        external: false
                    } if event_root_of(events, *e) == wait
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
        (
            Kind::If {
                handle: hx,
                body: bx,
            },
            Kind::If {
                handle: hy,
                body: by,
            },
        ) => hx == hy && bx == by,
        (Kind::SetConditional { handle: x, .. }, Kind::SetConditional { handle: y, .. }) => x == y,
        (
            Kind::While {
                handle: hx,
                body: bx,
            },
            Kind::While {
                handle: hy,
                body: by,
            },
        ) => hx == hy && bx == by,
        (
            Kind::Switch {
                handle: hx,
                bodies: bx,
            },
            Kind::Switch {
                handle: hy,
                bodies: by,
            },
        ) => hx == hy && bx == by,
        (Kind::WhileTick { handle: x, .. }, Kind::WhileTick { handle: y, .. }) => x == y,
        (Kind::EventRecord { external: x, .. }, Kind::EventRecord { external: y, .. }) => x == y,
        (Kind::EventWait { external: x, .. }, Kind::EventWait { external: y, .. }) => x == y,
        (Kind::Kernel { cooperative: x, .. }, Kind::Kernel { cooperative: y, .. }) => x == y,
        (Kind::WriteValue { bits32: x, .. }, Kind::WriteValue { bits32: y, .. }) => x == y,
        (
            Kind::WaitValue {
                bits32: x, cmp: cx, ..
            },
            Kind::WaitValue {
                bits32: y, cmp: cy, ..
            },
        ) => x == y && cx == cy,
        (Kind::BatchMem { .. }, Kind::BatchMem { .. }) => true,
        (Kind::HostFunc { .. }, Kind::HostFunc { .. }) => true,
        _ => op_tag(a) == op_tag(b),
    }
}

fn op_tag(k: &Kind) -> u8 {
    match k {
        Kind::Alloc { .. } => 0,
        Kind::Free { .. } => 1,
        Kind::Memcpy(_) => 2,
        Kind::Kernel { .. } => 3,
        Kind::Memset(_) => 4,
        Kind::HostFunc { .. } => 5,
        Kind::Empty => 11,
        Kind::EventRecord { .. } => 6,
        Kind::EventWait { .. } => 7,
        Kind::AllReduce { .. } => 8,
        Kind::ChildGraph { .. } => 9,
        Kind::Attach { .. } => 10,
        Kind::If { .. } => 12,
        Kind::SetConditional { .. } => 13,
        Kind::While { .. } => 14,
        Kind::WhileTick { .. } => 15,
        Kind::Switch { .. } => 16,
        Kind::WriteValue { .. } => 17,
        Kind::WaitValue { .. } => 18,
        Kind::DeviceLaunch { .. } => 19,
        Kind::BatchMem { .. } => 20,
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
