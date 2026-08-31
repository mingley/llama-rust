//! Replay a trace through [`gpu_sim`] with pinned H2D fills on miss.

/// In-graph scratch workspace for [`SimCfg::graph_mem`] / [`SimCfg::graph_auto_free`].
pub(crate) const GRAPH_SCRATCH_BYTES: u64 = 4096;

/// How a leaf GEMM graph owns its scratch workspace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LeafMem {
    /// Kernel-only (decode identity).
    None,
    /// `cudaMallocAsync` + matching `cudaFreeAsync` in the graph.
    Free,
    /// Alloc without free; [`gpu_sim::Sim::instantiate_graph_auto_free`].
    AutoFree,
}

impl LeafMem {
    /// Exclusive scratch mode from `--graph-mem` / `--graph-auto-free`.
    pub(crate) fn from_flags(mem: bool, auto_free: bool) -> Result<Self, Error> {
        match (mem, auto_free) {
            (true, true) => Err(Error::Store("choose one of graph-mem, graph-auto-free")),
            (true, false) => Ok(Self::Free),
            (false, true) => Ok(Self::AutoFree),
            (false, false) => Ok(Self::None),
        }
    }
}

/// Cooperative vs programmatic-dependent-launch flags for grouped GEMMs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GemmFlags {
    /// `cudaLaunchCooperativeKernel` (exclusive compute).
    pub cooperative: bool,
    /// Same-stream PDL wait+trigger (`cudaLaunchKernelEx`).
    pub pdl: bool,
    /// `cudaLaunchAttributeAccessPolicyWindow` over the expert page.
    pub l2_persist: bool,
    /// Hopper cluster X size (`cudaLaunchAttributeClusterDimension`). `0` is off.
    pub cluster: u8,
    /// Preferred cluster X (`cudaLaunchAttributePreferredClusterDimension`). `0` is off.
    pub preferred_cluster: u8,
    /// `cudaLaunchAttributeClusterSchedulingPolicyPreference` Spread.
    pub cluster_spread: bool,
    /// `cudaLaunchAttributePreferredSharedMemoryCarveout` MaxShared.
    pub max_shared: bool,
    /// `cudaLaunchAttributeSharedMemoryMode`. Decode identity stays Default.
    pub shared_mem: gpu_sim::SharedMemoryMode,
    /// `cudaLaunchAttributePortableClusterSizeMode`. Decode identity stays Default.
    pub portable_cluster: gpu_sim::PortableClusterMode,
    /// `cudaLaunchKernel` `sharedMemBytes`. Decode identity stays `0`.
    pub dynamic_shared: u32,
    /// CUDA 13 `cudaLaunchAttributeSharedMemoryMode`. Decode identity stays Default.
    pub portable_shared: gpu_sim::PortableSharedMode,
    /// `cudaLaunchAttributeNvlinkUtilCentricScheduling`. Occupies every Hyper-Q
    /// slot when the profile has NVLink. Decode identity stays disabled.
    pub nvlink_util_centric: bool,
    /// `cudaLaunchAttributeDeviceUpdatableKernelNode`. Graphs-only.
    /// Decode identity stays disabled.
    pub device_updatable: bool,
    /// `cudaLaunchAttributePriority`. [`None`] inherits the stream.
    /// Decode identity stays inherit-stream.
    pub priority: Option<i32>,
    /// `cudaLaunchAttributeLaunchCompletionEvent`. [`None`] records nothing.
    /// Decode identity stays no launch-completion event.
    pub launch_completion: Option<LaunchCompletionEvent>,
    /// `cudaLaunchAttributeProgrammaticEvent`. [`None`] records nothing.
    ///
    /// Other streams may `wait_event` at the PDL trigger. Decode identity stays
    /// no programmatic event.
    pub programmatic_event: Option<ProgrammaticEvent>,
}

impl GemmFlags {
    pub(crate) fn pdl_attr(self) -> Option<ProgrammaticLaunch> {
        if self.pdl && !self.cooperative {
            return Some(ProgrammaticLaunch {
                wait: true,
                trigger: true,
            });
        }
        // Programmatic-event overlap needs `cudaTriggerProgrammaticLaunchCompletion`
        // so the event fires at `pdl_trigger_permille`, not kernel completion.
        self.programmatic_event
            .is_some()
            .then_some(ProgrammaticLaunch {
                wait: false,
                trigger: true,
            })
    }

    pub(crate) fn persist_window(self, id: AllocId) -> Option<AccessPolicyWindow> {
        self.l2_persist
            .then(|| AccessPolicyWindow::persisting(KernelBuf::whole(id)))
    }

    pub(crate) fn cluster_dim(self) -> Option<gpu_sim::ClusterDim> {
        (self.cluster >= 1).then_some(gpu_sim::ClusterDim::x(u32::from(self.cluster)))
    }

    pub(crate) fn preferred_cluster_dim(self) -> Option<gpu_sim::ClusterDim> {
        (self.preferred_cluster >= 1)
            .then_some(gpu_sim::ClusterDim::x(u32::from(self.preferred_cluster)))
    }

    pub(crate) fn cluster_policy(self) -> ClusterSchedulingPolicy {
        if self.cluster_spread {
            ClusterSchedulingPolicy::Spread
        } else {
            ClusterSchedulingPolicy::Default
        }
    }

    pub(crate) fn carveout(self) -> SharedMemCarveout {
        if self.max_shared {
            SharedMemCarveout::MaxShared
        } else {
            SharedMemCarveout::Default
        }
    }

    pub(crate) fn kernel_attrs(self, id: AllocId) -> KernelAttrs {
        KernelAttrs {
            cooperative: self.cooperative,
            pdl: self.pdl_attr().unwrap_or_default(),
            access_policy: self.persist_window(id),
            cluster: self.cluster_dim(),
            preferred_cluster: self.preferred_cluster_dim(),
            cluster_policy: self.cluster_policy(),
            carveout: self.carveout(),
            shared_mem: self.shared_mem,
            portable_cluster: self.portable_cluster,
            dynamic_shared: self.dynamic_shared,
            portable_shared: self.portable_shared,
            nvlink_util_centric: self.nvlink_util_centric,
            device_updatable: self.device_updatable,
            priority: self.priority,
            programmatic_event: self.programmatic_event,
            launch_completion: self.launch_completion,
            ..KernelAttrs::default()
        }
    }

    /// Stream launch cannot carry the graphs-only device-updatable attr.
    pub(crate) fn for_stream(self) -> Self {
        Self {
            device_updatable: false,
            ..self
        }
    }
}

/// Preferred cluster needs a required dim that it is a multiple of (CUDA).
pub(crate) fn check_cluster_preferred(cluster: u8, preferred: u8) -> Result<(), Error> {
    if preferred == 0 {
        return Ok(());
    }
    if cluster == 0 {
        return Err(Error::Store("preferred-cluster needs cluster"));
    }
    if !preferred.is_multiple_of(cluster) {
        return Err(Error::Store(
            "preferred cluster must be a multiple of cluster",
        ));
    }
    Ok(())
}

/// Device-launch graphs refuse mem nodes and `cudaGraphExecUpdate`.
/// Device-updatable nodes also cannot `cudaGraphExecUpdate`.
pub(crate) fn check_device_graph_flags(
    device_launch: bool,
    device_updatable: bool,
    graph_update: bool,
    graph_mem: bool,
    graph_auto_free: bool,
) -> Result<(), Error> {
    if graph_update && device_launch {
        return Err(Error::Store("choose one of graph-update, device-launch"));
    }
    if graph_update && device_updatable {
        return Err(Error::Store("choose one of graph-update, device-updatable"));
    }
    if device_launch && (graph_mem || graph_auto_free) {
        return Err(Error::Store("device-launch cannot graph-mem"));
    }
    Ok(())
}

/// One `cudaEventCreate` for [`SimCfg::launch_completion`] / store GEMMs.
pub(crate) fn alloc_launch_completion(
    sim: &mut Sim,
    on: bool,
    next_event: &mut u32,
) -> Result<Option<LaunchCompletionEvent>, Error> {
    if !on {
        return Ok(None);
    }
    let ev = EventId(*next_event);
    *next_event = next_event.saturating_add(1);
    sim.create_event_disable_timing(ev)?;
    Ok(Some(LaunchCompletionEvent {
        event: ev,
        external: false,
    }))
}

/// One `cudaEventCreate` for [`SimCfg::programmatic_event`] / store GEMMs.
pub(crate) fn alloc_programmatic_event(
    sim: &mut Sim,
    on: bool,
    next_event: &mut u32,
) -> Result<Option<ProgrammaticEvent>, Error> {
    if !on {
        return Ok(None);
    }
    let ev = EventId(*next_event);
    *next_event = next_event.saturating_add(1);
    sim.create_event_disable_timing(ev)?;
    Ok(Some(ProgrammaticEvent {
        event: ev,
        external: false,
    }))
}

/// 8-byte device mailbox for [`SimCfg::wait_value`] / store copy-ready.
pub(crate) const MAILBOX_BYTES: u64 = 8;

/// `cudaMallocAsync` (or host-sync `cudaMalloc`) for a wait/write-value mailbox.
///
/// The pointer is not usable for [`wait_copy_ready`] / [`signal_copy_ready`] on
/// another stream until this stream catches up ([`gpu_sim::Sim::alloc`]).
/// Store [`crate::SimulatedGpuStore`] waits the copy stream after this alloc
/// *before* H2D so compute `wait_value64` is legal during DMA.
/// [`gpu_sim::Sim::malloc`] would `synchronize_device` and drain leftover
/// prefill.
pub(crate) fn alloc_copy_mailbox(
    sim: &mut Sim,
    device: DeviceId,
    stream: StreamId,
    sync: bool,
) -> Result<AllocId, Error> {
    hbm_alloc(sim, device, MAILBOX_BYTES, stream, sync)
}

/// [`alloc_copy_mailbox`] then wait `stream` so the pointer is device-resident.
///
/// No-op extra wait when `sync` (`cudaMalloc` is already live). Does not
/// [`gpu_sim::Sim::synchronize_device`].
pub(crate) fn alloc_resident_copy_mailbox(
    sim: &mut Sim,
    device: DeviceId,
    stream: StreamId,
    sync: bool,
) -> Result<AllocId, Error> {
    let id = alloc_copy_mailbox(sim, device, stream, sync)?;
    if !sync {
        sim.synchronize_stream(device, stream)?;
    }
    Ok(id)
}

/// `cuStreamWriteValue64` after H2D / prefetch so compute can wait without an event.
pub(crate) fn signal_copy_ready(
    sim: &mut Sim,
    device: DeviceId,
    mailbox: AllocId,
    gen: u64,
    stream: StreamId,
) -> Result<(), Error> {
    let _w = sim.write_value64(device, mailbox, 0, gen, stream)?;
    Ok(())
}

/// `cuStreamWaitValue64` Eq until [`signal_copy_ready`] completes on another stream.
pub(crate) fn wait_copy_ready(
    sim: &mut Sim,
    device: DeviceId,
    mailbox: AllocId,
    gen: u64,
    stream: StreamId,
) -> Result<(), Error> {
    let _w = sim.wait_value64(device, mailbox, 0, gen, WaitValueCmp::Eq, stream)?;
    Ok(())
}

/// Free a copy-ready mailbox. Host-sync `cudaFree` when `sync`.
pub(crate) fn free_copy_mailbox(
    sim: &mut Sim,
    device: DeviceId,
    mailbox: AllocId,
    stream: StreamId,
    sync: bool,
) -> Result<(), Error> {
    if sync {
        sim.free_sync(mailbox)?;
    } else {
        sim.free(device, mailbox, stream)?;
    }
    Ok(())
}

use crate::access::{ExpertAccess, ExpertKey, Trace};
use crate::error::Error;
use crate::place::PlaceMap;
use crate::planner::{
    plan_placement, plan_window, predicted_keys, window_keys, ChainState, Markov, Placement, Plan,
    Prefetch,
};
use crate::policy::Policy;
use crate::replay::{Touch, Walker};
use gpu_sim::{
    AccessPolicyWindow, AllocId, ClusterSchedulingPolicy, DType, DeviceFlags, DeviceId, EventId,
    GraphId, GraphInstantiateFlags, HardwareProfile, KernelAttrs, KernelBuf, KernelKind,
    LaunchCompletionEvent, MemAttach, MemPoolAttr, MemcpyAttributes, MemcpyOp, Place, PointerAttr,
    PoolId, PortableClusterMode, PortableSharedMode, ProgrammaticEvent, ProgrammaticLaunch, Score,
    SharedMemCarveout, SharedMemoryMode, Sim, StreamId, SynchronizationPolicy, WaitValueCmp,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

/// Simulated residency result: semantic score plus cache stats.
#[derive(Clone, Debug)]
pub struct SimReplay {
    /// Simulated nanoseconds (same as [`Score::wall_ns`]).
    pub sim_ns: u64,
    /// Host↔device bytes moved.
    pub bytes_moved: u64,
    /// Peak HBM live bytes.
    pub hbm_peak: u64,
    /// Profile TDP × wall, microjoules.
    pub energy_uj: u64,
    /// Microdollars per million tokens when the profile has rent.
    pub usd_micros_per_m_tokens: Option<u64>,
    /// Clock after the first token's last layer, when the trace has tokens.
    pub ttft_ns: Option<u64>,
    /// Mean later-token delta, when the trace has at least two tokens.
    pub itl_ns: Option<u64>,
    /// Cache hits.
    pub hits: u64,
    /// Cache misses.
    pub misses: u64,
    /// Prefetch fills that were not already resident.
    pub prefetches: u64,
    /// Demand hits on a key that was last filled by prefetch (not a demand miss).
    pub prefetch_hits: u64,
    /// Prefetched keys evicted before a demand acquire.
    pub prefetch_waste: u64,
    /// [`gpu_sim::Sim::launch_graph`] calls (0 unless [`SimCfg::cuda_graphs`]).
    pub graph_launches: u64,
    /// Grouped captures that recorded a parent of leaf child graphs.
    pub child_graphs: u64,
    /// [`gpu_sim::Sim::update_graph`] calls that reused a parked leaf exec.
    pub graph_updates: u64,
    /// [`gpu_sim::Sim::clone_graph`] calls that copied a leaf before instantiate.
    pub graph_clones: u64,
    /// [`gpu_sim::Sim::graph_exec_kernel_set_params`] calls that reused a parked leaf.
    pub graph_set_params: u64,
}

impl SimReplay {
    /// Single-line agent / CLI log.
    #[must_use]
    pub fn line(&self) -> String {
        let mut s = format!(
            "sim_ns={} bytes_moved={} hbm_peak={} energy_uj={}",
            self.sim_ns, self.bytes_moved, self.hbm_peak, self.energy_uj
        );
        if let Some(n) = self.usd_micros_per_m_tokens {
            let _w = write!(s, " usd_micros_per_m_tokens={n}");
        }
        if let Some(n) = self.ttft_ns {
            let _w = write!(s, " ttft_ns={n}");
        }
        if let Some(n) = self.itl_ns {
            let _w = write!(s, " itl_ns={n}");
        }
        let _w = write!(
            s,
            " hits={} misses={} prefetches={} prefetch_hits={} prefetch_waste={} graph_launches={} child_graphs={} graph_updates={} graph_clones={} graph_set_params={}",
            self.hits,
            self.misses,
            self.prefetches,
            self.prefetch_hits,
            self.prefetch_waste,
            self.graph_launches,
            self.child_graphs,
            self.graph_updates,
            self.graph_clones,
            self.graph_set_params
        );
        s
    }
}

/// Cache size, policy, expert payload, lookahead, and prefetch mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimCfg {
    /// Resident expert slots per home GPU (`GPU0` when unplaced).
    ///
    /// [`crate::schedule_placed`] / [`crate::schedule_remote`] keep one walker
    /// per home so a miss cannot evict a peer GPU's resident expert. When
    /// `restrict_hbm` holds fewer pages than `slots`, the scheduler still
    /// evicts so the next alloc cannot OOM. [`crate::schedule_remote`] uses the
    /// same page budget on the expert home. `--mapped` also caps occupancy at
    /// `host_pin_bytes / expert_bytes` so a pin budget of one expert pages
    /// one expert instead of `PinOom` on the second.
    pub slots: usize,
    /// Victim policy.
    pub policy: Policy,
    /// Bytes per expert H2D.
    pub bytes_per_expert: u64,
    /// Lookahead window for layer-ahead / oracle.
    pub lookahead: usize,
    /// Prefetch before the next router event.
    pub prefetch: Prefetch,
    /// Map `sequence % n_streams` onto CUDA streams so a batch can overlap.
    ///
    /// `n_streams` is GPU0 `copy_engines.max(2)`. Token-boundary sync ignores
    /// sequence changes so interleaved sequences at the same token stay concurrent.
    pub seq_streams: bool,
    /// Capture grouped expert GEMMs and replay them with [`gpu_sim::Sim::launch_graph`].
    ///
    /// Capture requires an idle stream (CUDA). After a token drain, a sticky
    /// resident set launches one parent graph. Each expert alloc is a
    /// captured leaf; a multi-expert launch records those leaves as child
    /// graphs so a later combo can reuse them. Leaves and parents are
    /// instantiated and uploaded before the first launch.
    pub cuda_graphs: bool,
    /// Upcoming-event window for [`plan_window`]. `0` leaves prefetch ungated.
    pub plan_window: usize,
    /// Stay vs Fetch threshold, permille of upcoming unique keys already resident.
    pub plan_threshold: u32,
    /// Sequences admitted per engine iteration at a token. `0` admits every
    /// sequence that shares the current token (one drain at the token boundary).
    pub max_batch: usize,
    /// Host-synchronous `cudaMalloc` / `cudaMemcpy` / `cudaFree` on every miss.
    ///
    /// Default is stream-ordered `alloc` / `memcpy` / `free` (`cudaMallocAsync`).
    /// A naive engine that uses the sync path cannot overlap a miss with other
    /// streams on that GPU. [`crate::SimulatedGpuStore::new`] stays async;
    /// [`crate::SimulatedGpuStore::with_cfg`] with [`crate::GpuStoreCfg::sync_alloc`]
    /// uses this path.
    pub sync_alloc: bool,
    /// Hold unused `cudaMallocAsync` bytes in the default mempool (`u64::MAX`
    /// release threshold) until [`gpu_sim::Sim::pool_trim_to`].
    ///
    /// Default `false` matches CUDA's default pool (`threshold = 0`: free
    /// returns HBM when the stream-ordered free completes). Serving engines
    /// raise the threshold so `cudaMalloc` can OOM while the pool still holds
    /// cache. Hits/misses stay the same; reuse pays `pool_reuse_ns`.
    /// [`crate::SimulatedGpuStore::new`] stays on threshold 0;
    /// [`crate::SimulatedGpuStore::with_cfg`] with [`crate::GpuStoreCfg::mempool`]
    /// raises it.
    pub mempool: bool,
    /// `cudaMemPoolTrimTo(0)` unused cached bytes after the walk.
    ///
    /// Implies [`Self::mempool`]. Illegal with [`Self::sync_alloc`]. Hits stay
    /// the same; peak HBM is unchanged. Decode identity stays no trim.
    /// [`crate::GpuStoreCfg::mempool_trim`] is the store path.
    pub mempool_trim: bool,
    /// `cudaMemPoolReuseAllowOpportunistic=0` after the mempool hold.
    ///
    /// Implies [`Self::mempool`]. Illegal with [`Self::sync_alloc`]. Hits stay
    /// the same; leftover cache stays reserved. Decode identity stays reuse-on.
    /// [`crate::GpuStoreCfg::mempool_no_reuse`] is the store path.
    pub mempool_no_reuse: bool,
    /// POSIX-FD shareable mempool IPC (`cudaMemPoolExportToShareableHandle`).
    ///
    /// Creates a shareable pool, exports it, imports a sibling that shares
    /// live/cached, and `cudaDeviceSetMemPool`s so `cudaMallocAsync` draws
    /// from it. Implies [`Self::mempool`] (`u64::MAX` hold). Illegal with
    /// [`Self::sync_alloc`], [`Self::mapped`], [`Self::managed`], or
    /// [`Self::vmm`]. Decode identity stays the device default pool.
    /// [`crate::GpuStoreCfg::shareable`] is the store path.
    pub shareable: bool,
    /// `cudaHostAllocMapped`: miss pages are mapped host, not HBM. Kernels run
    /// over PCIe with no H2D. Hits/misses follow the same walker; `hbm_peak`
    /// stays near zero. [`Self::host_register_mapped`] is `alloc_host` then
    /// `cudaHostRegisterMapped` (evict unregisters). Decode identity stays
    /// [`crate::SimulatedGpuStore::with_mapped`]. [`crate::SimulatedGpuStore::new`]
    /// stays on the H2D path; [`crate::SimulatedGpuStore::with_mapped`] uses this
    /// path.
    pub mapped: bool,
    /// `cudaMallocManaged` + `cudaMemAdviseSetReadMostly` + prefetch on miss.
    /// Alloc does not charge HBM; prefetch migrates (and replicates if a
    /// second GPU later prefetches the same page). Home also sets
    /// [`gpu_sim::MemAdvise::SetPreferredLocation`] so a remote read can
    /// keep the page on that GPU. `--place remote` GEMMs on GPU0 without a
    /// dest HBM copy. `--place replicas` uses
    /// that dest prefetch; dest eviction is `drop_managed_copy`. [`Self::accessed_by`]
    /// maps every GPU at fill so dest GEMMs read without a second copy.
    /// Hits/misses match H2D. [`crate::SimulatedGpuStore::new`] stays on pinned
    /// H2D; [`crate::SimulatedGpuStore::with_managed`] uses this path.
    pub managed: bool,
    /// `va_acquire` on miss (reuse an unmapped VA, else reserve+map), then
    /// pinned H2D. Evict [`gpu_sim::Sim::va_release`]s so the pointer stays.
    /// Hits/misses match H2D. [`crate::SimulatedGpuStore::new`] stays on pinned
    /// H2D; [`crate::SimulatedGpuStore::with_vmm`] uses this path. [`Self::vmm_page`] splits each map into KV-sized
    /// physicals (`0` is one `cuMemMap` for the whole expert). [`Self::accessed_by`]
    /// is `va_set_access` on every GPU at fill (peer read, no dest HBM).
    /// `--place replicas` maps dest then D2D unless AccessedBy; dest eviction
    /// is `va_unmap_range`.
    pub vmm: bool,
    /// Page size for [`Self::vmm`]. `0` maps the whole expert in one physical.
    /// [`crate::SimulatedGpuStore::with_vmm`] stays whole-VA;
    /// [`crate::GpuStoreCfg::vmm_page`] is the store path.
    pub vmm_page: u64,
    /// `cudaLaunchHostFunc` after each event's GEMMs (CPU scheduler roundtrip).
    ///
    /// Does not change hits/misses. Lengthens wall by `host_func_ns` per
    /// stream that ran a GEMM. [`crate::SimulatedGpuStore::new`] does not
    /// enqueue it; [`crate::SimulatedGpuStore::with_cfg`] with
    /// [`crate::GpuStoreCfg::host_func`] does.
    pub host_func: bool,
    /// `cudaStreamCreate` (blocking) for streams `1 .. n_streams`.
    ///
    /// They serialize with [`gpu_sim::StreamId::NULL`]. Default is
    /// `cudaStreamNonBlocking` (vLLM-style overlap). A no-op unless
    /// [`Self::seq_streams`] creates extra streams. [`crate::SimulatedGpuStore::new`]
    /// stays non-blocking; [`crate::SimulatedGpuStore::with_cfg`] with
    /// [`crate::GpuStoreCfg::blocking_streams`] marks the compute stream blocking.
    pub blocking_streams: bool,
    /// Pageable `cudaMemcpyAsync` (`memcpy_host_to_device`) instead of pinned DMA.
    ///
    /// Host-synchronous (`pageable_permille`). [`crate::SimulatedGpuStore::new`]
    /// stays pinned; [`crate::GpuStoreCfg::pageable`] is the store path.
    pub pageable: bool,
    /// `cudaHostRegister` on pageable staging so miss H2D is pinned DMA.
    ///
    /// Walker keeps a registered host buffer for the pin tax. Implies
    /// [`Self::pageable`]. Illegal with [`Self::mapped`] / [`Self::managed`].
    /// [`crate::GpuStoreCfg::host_register`] is the store path.
    pub host_register: bool,
    /// `cudaHostRegisterMapped` on mapped expert pages (`alloc_host` then register).
    ///
    /// Implies [`Self::mapped`]. Illegal with [`Self::host_register`]. Evict is
    /// `host_unregister` then `free_host`. Decode identity stays
    /// `cudaHostAllocMapped`. [`crate::GpuStoreCfg::host_register_mapped`] is
    /// the store path.
    pub host_register_mapped: bool,
    /// `cuPointerSetAttribute` SyncMemops on miss device pages.
    ///
    /// After alloc, before H2D / managed prefetch: memcpy of that pointer is
    /// host-synchronous. Illegal with [`Self::mapped`] and [`Self::memcpy_batch`].
    /// Decode identity stays async pinned H2D. [`crate::GpuStoreCfg::sync_memops`]
    /// is the store path.
    pub sync_memops: bool,
    /// `cudaSetDeviceFlags(cudaDeviceSyncMemops)` on every GPU.
    ///
    /// All runtime memcpy/memset on that device are host-synchronous, including
    /// unmarked pointers (replica D2D, KV). Distinct from per-page
    /// [`Self::sync_memops`]. Illegal with [`Self::mapped`] and
    /// [`Self::memcpy_batch`]. Does not imply pageable or [`Self::sync_alloc`].
    /// Decode identity stays async pinned H2D.
    /// [`crate::GpuStoreCfg::device_sync_memops`] is the store path.
    pub device_sync_memops: bool,
    /// `cudaMemcpyBatchAsync` for a multi-expert pinned/VMM prefetch window.
    ///
    /// Sibling H2D copies share one stream-order snapshot. Illegal with
    /// pageable, host-sync, mapped, managed, [`Self::sync_memops`], or
    /// [`Self::device_sync_memops`] fills.
    /// Decode identity stays sequential `memcpy_pinned_to_device`.
    /// [`crate::GpuStoreCfg::memcpy_batch`] is the store path.
    pub memcpy_batch: bool,
    /// Peer map without dest HBM at a managed or VMM fill.
    ///
    /// Dest GEMMs may read without migrating or charging dest HBM. `--place replicas`
    /// skips dest prefetch / VMM dest map+D2D / pinned D2D (no extra HBM). No-op unless
    /// [`Self::managed`], [`Self::vmm`], or pinned-async (`cudaMallocAsync`). Host-sync
    /// [`Self::sync_alloc`] (`cudaMalloc`) still D2Ds. [`crate::GpuStoreCfg::accessed_by`]
    /// is the store path.
    pub accessed_by: bool,
    /// CUDA legacy null stream (`set_legacy_null_stream`): NULL serializes
    /// with every other stream. Off by default. [`crate::GpuStoreCfg::legacy_null`]
    /// is the store path (copy NULL vs compute `StreamId(1)`).
    pub legacy_null: bool,
    /// `cudaStreamCreateWithPriority` for seq-streams (`set_created_streams_priority`).
    ///
    /// Created streams get priority equal to their id, so a later sequence
    /// wins when compute contends. A no-op unless [`Self::seq_streams`].
    /// [`crate::GpuStoreCfg::stream_priority`] marks the store compute stream.
    pub stream_priority: bool,
    /// `cudaGraphExecUpdate` a parked leaf exec onto the next miss alloc.
    ///
    /// Evict parks a one-expert graph instead of `destroy_graph`. The next
    /// leaf capture on that `(device, stream)` pays `graph_update_ns` instead
    /// of instantiate. Parent combo graphs still destroy (child ids are
    /// topology). A no-op unless [`Self::cuda_graphs`]. Decode identity stays
    /// destroy+instantiate. [`crate::GpuStoreCfg::graph_update`] is the store
    /// path (always captures when compute is idle).
    pub graph_update: bool,
    /// `cudaGraphExecKernelNodeSetParams` a parked leaf onto the next miss alloc.
    ///
    /// Evict parks the instantiated GEMM graph. The next miss retargets the
    /// unique kernel node (`graph_set_params_ns`) without a second capture,
    /// and a unique memcpy or memset if the leaf has one. Combo parents park
    /// too: [`gpu_sim::Sim::graph_exec_child_set_params`] swaps nested leaf
    /// ids (`cudaGraphExecChildGraphNodeSetParams`; child ids are topology for
    /// [`gpu_sim::Sim::update_graph`]). Works with
    /// [`Self::graph_mem`] / [`Self::graph_auto_free`] (CUDA cannot
    /// `cudaGraphExecUpdate` mem nodes). Illegal with [`Self::graph_update`].
    /// Implies [`Self::cuda_graphs`]. Decode identity stays destroy+instantiate.
    /// [`crate::GpuStoreCfg::graph_set_params`] is the store path.
    pub graph_set_params: bool,
    /// `cudaGraphClone` a leaf capture before instantiate (graph vs exec).
    ///
    /// Parent combo graphs still instantiate in place so child ids stay the
    /// GraphBank leaves. A no-op unless [`Self::cuda_graphs`]. Decode identity
    /// stays instantiate-in-place. [`crate::GpuStoreCfg::graph_clone`] is the
    /// store path.
    pub graph_clone: bool,
    /// `cudaGraphCreate` / `cudaGraphAdd*` instead of stream capture.
    ///
    /// Leaves and parents are built with [`gpu_sim::Sim::create_graph`] and
    /// [`gpu_sim::Sim::graph_add_kernel`] / [`gpu_sim::Sim::graph_add_child`].
    /// Combo children have no [`gpu_sim::Sim::graph_add_dependencies`] edge, so
    /// independent expert GEMMs may Hyper-Q overlap (`compute_slots >= 2`).
    /// Does not require an idle stream. Implies [`Self::cuda_graphs`]. Decode
    /// identity stays stream capture. [`crate::GpuStoreCfg::graph_build`] is
    /// the store path. Illegal with [`Self::graph_piecewise`].
    pub graph_build: bool,
    /// `cudaStreamBeginCaptureToGraph` combo parents (independent child roots).
    ///
    /// Each instantiated leaf is captured into one parent as an extra root
    /// (`numDependencies = 0`), so sibling expert GEMMs may Hyper-Q overlap
    /// (`compute_slots >= 2`). Leaves still use stream capture. Implies
    /// [`Self::cuda_graphs`]. Illegal with [`Self::graph_build`]. Decode
    /// identity stays a single `begin_capture` of child launches (same-stream
    /// edges serialize). [`crate::GpuStoreCfg::graph_piecewise`] is the store
    /// path (leaf capture-to-graph into `create_graph`).
    pub graph_piecewise: bool,
    /// `cudaGraphNodeSetEnabled` on a wide combo parent instead of recapture.
    ///
    /// A later token that GEMMs a subset of a stored combo (or of currently
    /// resident experts on that origin) skips extra child graphs instead of
    /// instantiating a new parent. Implies [`Self::cuda_graphs`]. Illegal with
    /// [`Self::device_launch`] (that path already splits combos to singles).
    /// Decode identity stays exact combo recapture.
    /// [`crate::GpuStoreCfg::graph_enable`] is the store CLI field (store GEMM
    /// stays per-leaf).
    pub graph_enable: bool,
    /// Leaf GEMM graphs include a scratch `cudaMallocAsync` + free.
    ///
    /// Models in-graph workspace (`cudaGraphAddMemAllocNode`). Hits/misses
    /// stay the same; HBM peak includes the scratch during launch. CUDA cannot
    /// `cudaGraphExecUpdate` mem nodes, so [`Self::graph_update`] is skipped.
    /// Implies [`Self::cuda_graphs`]. Decode identity stays kernel-only graphs.
    /// [`crate::GpuStoreCfg::graph_mem`] is the store path.
    pub graph_mem: bool,
    /// Leaf GEMM graphs alloc scratch without a matching free.
    ///
    /// Instantiates with [`gpu_sim::Sim::instantiate_graph_auto_free`] so
    /// relaunch recharges HBM. Illegal with [`Self::graph_mem`] (CUDA cannot
    /// AutoFreeOnLaunch a graph that has mem free nodes). CUDA cannot
    /// `cudaGraphExecUpdate` mem nodes, so [`Self::graph_update`] is skipped.
    /// Implies [`Self::cuda_graphs`]. Decode identity stays kernel-only graphs.
    /// [`crate::GpuStoreCfg::graph_auto_free`] is the store path.
    pub graph_auto_free: bool,
    /// `cudaDeviceGraphMemTrim` unused reserved graph-mem after the walk.
    ///
    /// Peak HBM is unchanged. Live graph allocs stay. Decode identity stays
    /// off. [`crate::GpuStoreCfg::graph_mem_trim`] is the store path.
    pub graph_mem_trim: bool,
    /// `cudaLaunchCooperativeKernel` for grouped GEMMs.
    ///
    /// Occupies every Hyper-Q slot so leftover prefill cannot overlap decode
    /// even when [`Self::compute_slots`] is `>=2`. Decode identity stays
    /// `cudaLaunchKernel`. [`crate::GpuStoreCfg::cooperative`] is the store
    /// path.
    pub cooperative: bool,
    /// Same-stream programmatic dependent launch for grouped GEMMs.
    ///
    /// Consecutive expert kernels on one stream may overlap after the
    /// previous kernel's PDL trigger (`pdl_trigger_permille`) when
    /// [`Self::compute_slots`] is `>=2`. Illegal with [`Self::cooperative`].
    /// Decode identity stays `cudaLaunchKernel`. [`crate::GpuStoreCfg::pdl`]
    /// is the store path.
    pub pdl: bool,
    /// `cudaLaunchAttributeAccessPolicyWindow` over each expert page.
    ///
    /// Sets [`gpu_sim::Sim::enable_persisting_l2`] and launches GEMMs with a
    /// persisting window so a reused expert bills less HBM after the first
    /// fill. Decode identity stays `cudaLaunchKernel` with persist limit 0.
    /// [`crate::GpuStoreCfg::l2_persist`] is the store path.
    pub l2_persist: bool,
    /// `cudaCtxResetPersistingL2Cache` after each grouped GEMM.
    ///
    /// Implies [`Self::l2_persist`]. Live (cannot capture). Hits stay the same;
    /// a reused expert does not keep persisting L2 lines. Decode identity stays
    /// no reset. [`crate::GpuStoreCfg::l2_reset`] is the store path.
    pub l2_reset: bool,
    /// Hopper cluster X size (`cudaLaunchAttributeClusterDimension`). `0` is off.
    ///
    /// Occupies `min(N, compute_slots)` Hyper-Q slots so leftover kernels
    /// cannot overlap a cluster that fills the cap. Decode identity stays
    /// `cudaLaunchKernel` (no cluster). [`crate::GpuStoreCfg::cluster`] is the
    /// store path.
    pub cluster: u8,
    /// Hopper preferred cluster X (`cudaLaunchAttributePreferredClusterDimension`).
    /// `0` is off.
    ///
    /// Occupancy uses this size when it fits in [`Self::compute_slots`], else
    /// [`Self::cluster`]. Requires [`Self::cluster`]. Must be an integer
    /// multiple of the required size. Decode identity stays no preferred dim.
    /// [`crate::GpuStoreCfg::preferred_cluster`] is the store path.
    pub preferred_cluster: u8,
    /// Spread cluster scheduling (`cudaLaunchAttributeClusterSchedulingPolicyPreference`).
    ///
    /// Occupies every Hyper-Q slot so leftover kernels cannot overlap even
    /// when [`Self::cluster`] is smaller than [`Self::compute_slots`]. A no-op
    /// unless cluster blocks `> 1`. Decode identity stays Default.
    /// [`crate::GpuStoreCfg::cluster_spread`] is the store path.
    pub cluster_spread: bool,
    /// Function Spread cluster scheduling (`cudaFuncSetAttribute` ClusterSchedulingPolicyPreference).
    ///
    /// Launch Default inherits this occupancy so leftover kernels cannot
    /// overlap even when [`Self::cluster`] is smaller than [`Self::compute_slots`].
    /// Distinct from launch-attribute [`Self::cluster_spread`]. A no-op unless
    /// cluster blocks `> 1`. Decode identity stays Default.
    /// [`crate::GpuStoreCfg::func_cluster_spread`] is the store path.
    pub func_cluster_spread: bool,
    /// Max-shared carveout (`cudaLaunchAttributePreferredSharedMemoryCarveout`).
    ///
    /// Occupies every Hyper-Q slot so leftover kernels cannot overlap.
    /// Decode identity stays Default. [`crate::GpuStoreCfg::max_shared`] is
    /// the store path.
    pub max_shared: bool,
    /// Function MaxShared carveout (`cudaFuncSetAttribute` PreferredSharedMemoryCarveout).
    ///
    /// Launch Default inherits this occupancy so leftover kernels cannot
    /// overlap. Distinct from launch-attribute [`Self::max_shared`]. Decode
    /// identity stays Default. [`crate::GpuStoreCfg::func_max_shared`] is the
    /// store path.
    pub func_max_shared: bool,
    /// `cudaFuncAttributeNonPortableClusterSizeAllowed`. Default disallowed.
    ///
    /// Lets [`Self::cluster`] exceed `portable_cluster_size` up to
    /// `max_blocks_per_cluster`. Decode identity stays disallowed.
    /// [`crate::GpuStoreCfg::non_portable_cluster`] is the store path.
    pub non_portable_cluster: bool,
    /// Stream host-wait policy (`cudaLaunchAttributeSynchronizationPolicy`).
    ///
    /// Decode-token `sync_work` pays `host_sync_*_ns` on `synchronize_stream`.
    /// Decode identity stays Auto. [`crate::GpuStoreCfg::sync_policy`] is the
    /// store path.
    pub sync_policy: gpu_sim::SynchronizationPolicy,
    /// `cudaLaunchAttributeMemSyncDomain` on the decode compute stream.
    ///
    /// [`gpu_sim::MemSyncDomain::Remote`] isolates leftover prefill fence tax
    /// (`same_domain_fence_permille`) when [`Self::decode_priority`] puts
    /// decode GEMMs on a second stream. [`gpu_sim::MemSyncDomain::Default`]
    /// is decode identity. [`crate::GpuStoreCfg::mem_sync_domain`] is the
    /// store path.
    pub mem_sync_domain: gpu_sim::MemSyncDomain,
    /// Shared-memory bank width (`cudaLaunchAttributeSharedMemoryMode`).
    ///
    /// Default never scales duration. FourByte / EightByte scale by
    /// `1000 / GpuProfile::shared_mem_*_permille` (profile default 1000).
    /// Decode identity stays Default. [`crate::GpuStoreCfg::shared_mem`] is
    /// the store path.
    pub shared_mem: gpu_sim::SharedMemoryMode,
    /// Function shared-mem bank width (`cudaFuncSetSharedMemConfig`).
    ///
    /// Launch Default inherits this duration scale. Distinct from launch-attribute
    /// [`Self::shared_mem`]. Decode identity stays Default.
    /// [`crate::GpuStoreCfg::func_shared_mem`] is the store path.
    pub func_shared_mem: gpu_sim::SharedMemoryMode,
    /// Device shared-mem bank width (`cudaDeviceSetSharedMemConfig`).
    ///
    /// Launch Default inherits this duration scale when the function config is
    /// also Default. Distinct from [`Self::func_shared_mem`] and launch-attribute
    /// [`Self::shared_mem`]. Decode identity stays Default.
    /// [`crate::GpuStoreCfg::device_shared_mem`] is the store path.
    pub device_shared_mem: gpu_sim::SharedMemoryMode,
    /// Portable-cluster size mode (`cudaLaunchAttributePortableClusterSizeMode`).
    ///
    /// Default uses the current function attribute. RequirePortable always
    /// refuses a cluster larger than `portable_cluster_size`. AllowNonPortable
    /// allows up to `max_blocks_per_cluster` even when
    /// [`Self::non_portable_cluster`] is off. Decode identity stays Default.
    /// [`crate::GpuStoreCfg::portable_cluster`] is the store path.
    pub portable_cluster: gpu_sim::PortableClusterMode,
    /// `cudaFuncAttributeMaxDynamicSharedMemorySize` to the SKU opt-in max.
    ///
    /// Lets [`Self::dynamic_shared`] exceed `max_shared_mem_per_block`.
    /// Decode identity stays `0` (portable only).
    /// [`crate::GpuStoreCfg::optin_shared`] is the store path.
    pub optin_shared: bool,
    /// `cudaLaunchKernel` `sharedMemBytes` on grouped expert GEMMs. `0` is off.
    ///
    /// Decode identity stays `0`. [`crate::GpuStoreCfg::dynamic_shared`] is
    /// the store path.
    pub dynamic_shared: u32,
    /// CUDA 13 portable-shared mode (`cudaLaunchAttributeSharedMemoryMode`).
    ///
    /// Default uses the function attribute. RequirePortable always refuses
    /// oversize. AllowNonPortable allows up to `max_shared_mem_per_block_optin`
    /// even when [`Self::optin_shared`] is off. Decode identity stays Default.
    /// [`crate::GpuStoreCfg::portable_shared`] is the store path.
    pub portable_shared: gpu_sim::PortableSharedMode,
    /// `cudaLaunchAttributeNvlinkUtilCentricScheduling`.
    ///
    /// Occupies every Hyper-Q slot when the profile has NVLink so leftover
    /// kernels cannot overlap even when [`Self::compute_slots`] is `>=2`.
    /// Without NVLink the flag is stored and occupancy is unchanged.
    /// Decode identity stays disabled. [`crate::GpuStoreCfg::nvlink_util_centric`]
    /// is the store path.
    pub nvlink_util_centric: bool,
    /// `cudaLaunchAttributeDeviceUpdatableKernelNode` on grouped expert GEMMs.
    ///
    /// [`gpu_sim::Sim::graph_exec_kernel_set_params`] keeps the exec uploaded.
    /// Illegal with [`Self::graph_update`]. Decode identity stays disabled.
    /// [`crate::GpuStoreCfg::device_updatable`] is the store path.
    pub device_updatable: bool,
    /// `cudaLaunchAttributePriority` on grouped expert GEMMs.
    ///
    /// [`None`] inherits `cudaStreamCreateWithPriority`. [`Some`] overrides
    /// that kernel (memcpy stays on the stream). Higher first when compute
    /// contends. Decode identity stays inherit-stream. Instantiates with
    /// `cudaGraphInstantiateFlagUseNodePriority` so captured node values are
    /// used at replay.
    /// [`crate::GpuStoreCfg::kernel_priority`] is the store path.
    pub kernel_priority: Option<i32>,
    /// `cudaGraphInstantiateFlagDeviceLaunch` + [`gpu_sim::Sim::device_launch_graph`].
    ///
    /// Leaf GEMM graphs only (no combo-parent child graphs, no mem nodes).
    /// Illegal with [`Self::graph_update`] / [`Self::graph_mem`] /
    /// [`Self::graph_auto_free`]. Decode identity stays host `launch_graph`.
    /// [`crate::GpuStoreCfg::device_launch`] is the store path.
    pub device_launch: bool,
    /// `cudaLaunchAttributeLaunchCompletionEvent` on grouped expert GEMMs.
    ///
    /// Other streams may `wait_event` at kernel *start* instead of completion.
    /// Store [`crate::SimulatedGpuStore::pin_hot`] replica D2D on `n_gpus >= 2`
    /// waits that event on the copy stream so leftover GEMM overlaps the copy.
    /// Illegal with [`Self::device_launch`]. Decode identity stays no event.
    /// [`crate::GpuStoreCfg::launch_completion`] is the store path.
    pub launch_completion: bool,
    /// `cudaLaunchAttributeProgrammaticEvent` on grouped expert GEMMs.
    ///
    /// Other streams may `wait_event` at the PDL trigger (`pdl_trigger_permille`)
    /// instead of kernel completion. Store [`crate::SimulatedGpuStore::pin_hot`]
    /// replica D2D on `n_gpus >= 2` waits that event on the copy stream so
    /// leftover GEMM overlaps the copy. Implies a PDL trigger on those GEMMs
    /// (same-stream PDL wait stays [`Self::pdl`]). Illegal with
    /// [`Self::device_launch`]. Decode identity stays no event.
    /// [`crate::GpuStoreCfg::programmatic_event`] is the store path.
    pub programmatic_event: bool,
    /// `cudaStreamAttachMemAsync(..., cudaMemAttachSingle)` on managed experts.
    ///
    /// After alloc+advise, attach the page to the work stream and prefetch
    /// there so GEMM stays legal under Single. NULL work becomes `StreamId(1)`
    /// (Single cannot use the NULL stream). Identity stays Global + prefetch
    /// on the work stream as today. Implies [`Self::managed`]. Illegal with
    /// [`Self::seq_streams`]. [`crate::GpuStoreCfg::stream_attach`] is the
    /// store path.
    pub stream_attach: bool,
    /// `cudaMallocManaged(..., cudaMemAttachHost)` then Global attach.
    ///
    /// After alloc+advise, attach Global on the work stream so device prefetch
    /// is legal. Identity managed is Global at alloc (no Attach op). Implies
    /// [`Self::managed`]. Prefetch stays on the work stream. Combined with
    /// [`Self::stream_attach`], alloc is Host then Single (no extra Global).
    /// [`crate::GpuStoreCfg::managed_host`] is the store path.
    pub managed_host: bool,
    /// `cudaMemPrefetchAsync(..., cudaCpuDeviceId)` on managed LRU evict.
    ///
    /// Keeps the managed allocation instead of `cudaFree`. The next miss
    /// prefetches the same pointer back. Implies [`Self::managed`]. Hits stay
    /// the same; extra host↔device bytes move on thrash. Decode identity stays
    /// `free_sync`. [`crate::GpuStoreCfg::prefetch_host`] is the store path.
    pub prefetch_host: bool,
    /// `cuStreamWaitValue64` / `cuStreamWriteValue64` copy-ready handshake.
    ///
    /// After H2D / prefetch, write a generation into an 8-byte device mailbox
    /// instead of `record_event`. The mailbox is `cudaMallocAsync` on the copy
    /// stream and that stream is waited before H2D so compute wait is resident
    /// during DMA. GEMM waits Eq on the compute stream. Decode identity stays
    /// CUDA events. Graphs stay kernel-only (wait/write are live stream ops).
    /// [`crate::GpuStoreCfg::wait_value`] is the store path.
    pub wait_value: bool,
    /// Hopper NVLS replica fanout (`cuMulticastCreate` / bind / kernel store).
    ///
    /// `--place replicas` maps dest VMM physicals then one NVLS kernel instead
    /// of N sequential D2Ds. Occupies compute, not a copy engine. Implies
    /// [`Self::vmm`]. Illegal with [`Self::accessed_by`], [`Self::mapped`],
    /// [`Self::managed`], or [`Self::vmm_page`]. Needs an NVLink clique.
    /// Decode identity stays copy-engine D2D.
    pub multicast: bool,
    /// Hyper-Q occupancy override (`0` keeps the profile).
    ///
    /// `1` is exclusive compute. `>=2` lets independent sequence GEMMs overlap
    /// at full issue rate when [`Self::seq_streams`] puts them on different
    /// streams. Default `0` keeps decode identity.
    pub compute_slots: u8,
    /// Green-context SM fraction (‰) on every replay stream. `0` keeps a full
    /// chip. Compute-bound kernels scale; memory-bound keep full HBM.
    ///
    /// With [`Self::decode_priority`], this caps the decode stream; leftover
    /// prefill gets the remainder. Walker `--decode-sms` does not imply
    /// decode-priority (token 0 is prefill).
    pub decode_sm_permille: u16,
    /// Decode GEMMs on a second compute stream (`StreamId(n_copy + 1)`).
    ///
    /// Token 0 stays on the prefill stream. Token-boundary ITL samples the
    /// decode stream so leftover prefill does not inflate it. Does not imply
    /// [`Self::stream_priority`] (the walker CLI does). Default off: every
    /// event uses `sequence % n_copy` / NULL.
    pub decode_priority: bool,
}

impl SimCfg {
    /// LRU demand paging: no prefetch, graphs, planner, or seq-streams.
    #[must_use]
    pub fn lru(slots: usize, bytes_per_expert: u64, lookahead: usize) -> Self {
        Self {
            slots,
            policy: Policy::Lru,
            bytes_per_expert,
            lookahead,
            prefetch: Prefetch::None,
            seq_streams: false,
            cuda_graphs: false,
            plan_window: 0,
            plan_threshold: 500,
            max_batch: 0,
            sync_alloc: false,
            mempool: false,
            mempool_trim: false,
            mempool_no_reuse: false,
            shareable: false,
            mapped: false,
            managed: false,
            vmm: false,
            vmm_page: 0,
            host_func: false,
            blocking_streams: false,
            pageable: false,
            host_register: false,
            host_register_mapped: false,
            sync_memops: false,
            device_sync_memops: false,
            accessed_by: false,
            legacy_null: false,
            stream_priority: false,
            graph_update: false,
            graph_set_params: false,
            graph_clone: false,
            graph_build: false,
            graph_piecewise: false,
            graph_enable: false,
            graph_mem: false,
            graph_auto_free: false,
            graph_mem_trim: false,
            cooperative: false,
            pdl: false,
            l2_persist: false,
            l2_reset: false,
            cluster: 0,
            preferred_cluster: 0,
            cluster_spread: false,
            func_cluster_spread: false,
            max_shared: false,
            func_max_shared: false,
            non_portable_cluster: false,
            sync_policy: gpu_sim::SynchronizationPolicy::Auto,
            mem_sync_domain: gpu_sim::MemSyncDomain::Default,
            shared_mem: gpu_sim::SharedMemoryMode::Default,
            func_shared_mem: gpu_sim::SharedMemoryMode::Default,
            device_shared_mem: gpu_sim::SharedMemoryMode::Default,
            portable_cluster: gpu_sim::PortableClusterMode::Default,
            optin_shared: false,
            dynamic_shared: 0,
            portable_shared: gpu_sim::PortableSharedMode::Default,
            nvlink_util_centric: false,
            device_updatable: false,
            kernel_priority: None,
            device_launch: false,
            launch_completion: false,
            programmatic_event: false,
            stream_attach: false,
            managed_host: false,
            prefetch_host: false,
            wait_value: false,
            multicast: false,
            compute_slots: 0,
            decode_sm_permille: 0,
            decode_priority: false,
            memcpy_batch: false,
        }
    }
}

pub(crate) fn validate_sim_cfg(cfg: &SimCfg, profile: &HardwareProfile) -> Result<(), Error> {
    check_cluster_preferred(cfg.cluster, cfg.preferred_cluster)?;
    if cfg.graph_update && cfg.graph_set_params {
        return Err(Error::Store("choose one of graph-update, graph-set-params"));
    }
    check_device_graph_flags(
        cfg.device_launch,
        cfg.device_updatable,
        cfg.graph_update,
        cfg.graph_mem,
        cfg.graph_auto_free,
    )?;
    if cfg.graph_build && cfg.graph_piecewise {
        return Err(Error::Store("choose one of graph-build, graph-piecewise"));
    }
    if cfg.graph_enable && cfg.device_launch {
        return Err(Error::Store("graph-enable cannot device-launch"));
    }
    if cfg.launch_completion && cfg.device_launch {
        return Err(Error::Store("launch-completion cannot device-launch"));
    }
    if cfg.programmatic_event && cfg.device_launch {
        return Err(Error::Store("programmatic-event cannot device-launch"));
    }
    if cfg.stream_attach && !cfg.managed {
        return Err(Error::Store("stream-attach needs managed"));
    }
    if cfg.stream_attach && cfg.seq_streams {
        return Err(Error::Store("stream-attach cannot seq-streams"));
    }
    if cfg.managed_host && !cfg.managed {
        return Err(Error::Store("managed-host needs managed"));
    }
    if cfg.prefetch_host && !cfg.managed {
        return Err(Error::Store("prefetch-host needs managed"));
    }
    if cfg.pdl && cfg.cooperative {
        return Err(Error::Store("choose one of pdl, cooperative"));
    }
    if cfg.shareable && (cfg.sync_alloc || cfg.mapped || cfg.managed || cfg.vmm) {
        return Err(Error::Store("shareable needs cudaMallocAsync"));
    }
    if cfg.mempool_trim && cfg.sync_alloc {
        return Err(Error::Store("mempool-trim needs cudaMallocAsync"));
    }
    if cfg.mempool_no_reuse && cfg.sync_alloc {
        return Err(Error::Store("mempool-no-reuse needs cudaMallocAsync"));
    }
    if cfg.memcpy_batch
        && (cfg.pageable
            || cfg.sync_alloc
            || cfg.mapped
            || cfg.managed
            || cfg.host_register_mapped
            || cfg.sync_memops
            || cfg.device_sync_memops)
    {
        return Err(Error::Store("memcpy-batch needs async pinned/vmm H2D"));
    }
    if cfg.sync_memops && cfg.mapped {
        return Err(Error::Store("sync-memops needs device memcpy"));
    }
    if cfg.device_sync_memops && cfg.mapped {
        return Err(Error::Store("device-sync-memops needs device memcpy"));
    }
    if cfg.host_register_mapped && cfg.host_register {
        return Err(Error::Store(
            "choose one of host-register, host-register-mapped",
        ));
    }
    if cfg.host_register && !cfg.pageable {
        return Err(Error::Store("host-register needs pageable"));
    }
    if cfg.host_register && (cfg.mapped || cfg.managed) {
        return Err(Error::Store("host-register needs pinned/vmm H2D"));
    }
    if cfg.host_register_mapped && !cfg.mapped {
        return Err(Error::Store("host-register-mapped needs mapped"));
    }
    if !cfg.multicast {
        return Ok(());
    }
    if !cfg.vmm || cfg.mapped || cfg.managed {
        return Err(Error::Store("multicast requires vmm"));
    }
    if cfg.accessed_by {
        return Err(Error::Store("choose one of multicast, accessed-by"));
    }
    if cfg.vmm_page > 0 {
        return Err(Error::Store("multicast needs whole-VA maps"));
    }
    if profile.n_gpus() < 2 || !profile.has_nvlink() {
        return Err(Error::Store("multicast needs NVLink"));
    }
    Ok(())
}

/// Replay `trace` on `profile` with a `slots`-entry expert cache.
///
/// Each miss copies `bytes_per_expert` pinned-host→device, then a grouped GEMM runs
/// on the same stream (stream-ordered; no invented overlap). Hits skip the copy.
/// The clock is sampled after each token so TTFT / ITL are real token boundaries.
pub fn sim_replay(
    trace: &Trace,
    profile: HardwareProfile,
    slots: usize,
    policy: Policy,
    bytes_per_expert: u64,
    lookahead: usize,
) -> Result<SimReplay, Error> {
    sim_replay_cfg(
        trace,
        profile,
        SimCfg {
            policy,
            ..SimCfg::lru(slots, bytes_per_expert, lookahead)
        },
    )
}

/// [`sim_replay`] with an explicit [`Prefetch`] mode.
pub fn sim_replay_cfg(
    trace: &Trace,
    profile: HardwareProfile,
    cfg: SimCfg,
) -> Result<SimReplay, Error> {
    validate_sim_cfg(&cfg, &profile)?;
    let keys = trace.keys();
    let mut sim = Sim::new(sim_profile(profile, &cfg));
    apply_device_sync_memops(&mut sim, cfg.device_sync_memops)?;
    apply_func_max_shared(&mut sim, cfg.func_max_shared)?;
    apply_func_cluster_spread(&mut sim, cfg.func_cluster_spread)?;
    apply_func_shared_mem(&mut sim, cfg.func_shared_mem)?;
    apply_device_shared_mem(&mut sim, cfg.device_shared_mem)?;
    if cfg.shareable {
        let _imported = bind_shareable_mempools(&mut sim)?;
    }
    if cfg.mempool || cfg.shareable || cfg.mempool_trim || cfg.mempool_no_reuse {
        sim.set_default_pool_release_threshold(u64::MAX)?;
    }
    if cfg.mempool_no_reuse {
        disable_pool_opportunistic_reuse(&mut sim)?;
    }
    advise_pool_access_if_pinned(&mut sim, &cfg)?;
    let d = DeviceId(0);
    let s = StreamId(0);
    let bytes = cfg.bytes_per_expert.max(1);
    pin_pageable_staging(&mut sim, &cfg, bytes)?;
    let slots = occupancy_slots(&cfg, sim.pin_budget());
    let mut handles: BTreeMap<ExpertKey, PageHandle> = BTreeMap::new();
    let mut w = Walker::new(&keys, slots, cfg.policy, cfg.lookahead);
    let plan = StreamPlan::new(sim.profile(), cfg.seq_streams, cfg.decode_priority);
    if cfg.blocking_streams {
        sim.set_created_streams_blocking(plan.mark)?;
    }
    if cfg.legacy_null {
        sim.set_legacy_null_stream(true);
    }
    if cfg.stream_priority {
        sim.set_created_streams_priority(plan.mark)?;
    }
    apply_stream_sms(&mut sim, plan, cfg.decode_sm_permille)?;
    apply_stream_sync_policy(&mut sim, plan, cfg.sync_policy)?;
    apply_stream_mem_sync_domain(&mut sim, plan, cfg.mem_sync_domain)?;
    if cfg.l2_persist || cfg.l2_reset {
        sim.enable_persisting_l2()?;
    }
    allow_non_portable_cluster_if(&mut sim, cfg.non_portable_cluster)?;
    allow_optin_shared_if(&mut sim, cfg.optin_shared)?;
    let mut args = TouchArgs {
        d,
        s,
        bytes,
        slots,
        sync_alloc: cfg.sync_alloc,
        mapped: cfg.mapped,
        managed: cfg.managed,
        vmm: cfg.vmm,
        vmm_page: cfg.vmm_page,
        pageable: cfg.pageable,
        host_register: cfg.host_register,
        host_register_mapped: cfg.host_register_mapped,
        sync_memops: cfg.sync_memops,
        memcpy_batch: cfg.memcpy_batch,
        accessed_by: cfg.accessed_by,
        wait_value: cfg.wait_value,
        stream_attach: cfg.stream_attach,
        managed_host: cfg.managed_host,
        prefetch_host: cfg.prefetch_host,
    };
    let mut token_ends: Vec<u64> = Vec::new();
    let mut ctr = ReplayCounters::default();
    let mut markov = Markov::new();
    let mut chain = ChainState::new();
    let mut prefetched: BTreeSet<ExpertKey> = BTreeSet::new();
    let leaf = LeafMem::from_flags(cfg.graph_mem, cfg.graph_auto_free)?;
    let mut next_event = 1u32;
    let launch_completion =
        alloc_launch_completion(&mut sim, cfg.launch_completion, &mut next_event)?;
    let programmatic_event =
        alloc_programmatic_event(&mut sim, cfg.programmatic_event, &mut next_event)?;
    let mut graphs = GraphBank::new(cfg.graph_update, cfg.graph_clone, cfg.graph_build, leaf)
        .with_cooperative(cfg.cooperative)
        .with_pdl(cfg.pdl)
        .with_l2_persist(cfg.l2_persist || cfg.l2_reset)
        .with_l2_reset(cfg.l2_reset)
        .with_cluster(cfg.cluster)
        .with_preferred_cluster(cfg.preferred_cluster)
        .with_cluster_spread(cfg.cluster_spread)
        .with_max_shared(cfg.max_shared)
        .with_shared_mem(cfg.shared_mem)
        .with_portable_cluster(cfg.portable_cluster)
        .with_dynamic_shared(cfg.dynamic_shared)
        .with_portable_shared(cfg.portable_shared)
        .with_nvlink_util(cfg.nvlink_util_centric)
        .with_device_updatable(cfg.device_updatable)
        .with_kernel_priority(cfg.kernel_priority)
        .with_device_launch(cfg.device_launch)
        .with_launch_completion(launch_completion)
        .with_programmatic_event(programmatic_event)
        .with_set_params(cfg.graph_set_params)
        .with_piecewise(cfg.graph_piecewise)
        .with_enable(cfg.graph_enable);
    let mut admitted: BTreeSet<u64> = BTreeSet::new();
    for (i, event) in trace.events.iter().enumerate() {
        args.s = bump_null_for_attach(plan.work(event.sequence, event.token), cfg.stream_attach);
        let ek = event.keys();
        for key in &ek {
            let (got, touch) = w.next_touch().ok_or(Error::Store("short walker"))?;
            if got != *key {
                return Err(Error::Store("walker key mismatch"));
            }
            note_touch(&mut ctr, &mut prefetched, *key, touch);
            apply_touch(
                &mut sim,
                &mut handles,
                &mut graphs,
                args,
                *key,
                touch,
                &mut next_event,
            )?;
        }
        gemm_keys(
            &mut sim,
            &mut handles,
            &mut graphs,
            &ek,
            cfg.cuda_graphs,
            &mut ctr,
            cfg.decode_priority.then_some(args.s),
        )?;
        if cfg.host_func {
            host_callbacks(
                &mut sim,
                &handles,
                &ek,
                cfg.decode_priority.then_some(args.s),
            )?;
        }
        if should_prefetch(cfg, &handles, trace, i) {
            let predicted = predicted_keys(cfg.prefetch, &markov, chain.predecessor(event), &ek);
            let planned = if cfg.plan_window > 0 {
                window_keys(trace, i.saturating_add(1), cfg.plan_window)
            } else {
                Vec::new()
            };
            let fill = args;
            let mut misses = Vec::new();
            for key in predicted.into_iter().chain(planned) {
                match w.prefetch_touch(key) {
                    Touch::Hit => {}
                    miss @ Touch::Miss { .. } => {
                        ctr.prefetches = ctr.prefetches.saturating_add(1);
                        let _ins = prefetched.insert(key);
                        misses.push((key, miss));
                    }
                }
            }
            apply_misses(
                &mut sim,
                &mut handles,
                &mut graphs,
                fill,
                &misses,
                &mut next_event,
            )?;
        }
        chain.observe(&mut markov, event);
        let _ins = admitted.insert(event.sequence);
        if engine_step(&trace.events, i, cfg.max_batch, admitted.len()) {
            sync_work(&mut sim, 1, plan, event.token > 0)?;
            if last_of_token(&trace.events, i) {
                token_ends.push(sim.clock_ns());
            }
            admitted.clear();
        }
    }
    if token_ends.is_empty() {
        sim.synchronize()?;
    }
    ctr.graph_updates = graphs.updates;
    ctr.graph_clones = graphs.clones;
    ctr.graph_set_params = graphs.kernel_sets;
    if cfg.graph_mem_trim {
        trim_graph_pools(&mut sim)?;
    }
    if cfg.mempool_trim {
        trim_device_pools(&mut sim)?;
    }
    Ok(finish(&sim, &token_ends, ctr))
}

#[derive(Clone, Copy)]
pub(crate) struct TouchArgs {
    pub d: DeviceId,
    pub s: StreamId,
    pub bytes: u64,
    pub slots: usize,
    /// [`SimCfg::sync_alloc`]: `malloc` / `memcpy_sync` / `free_sync`.
    pub sync_alloc: bool,
    /// [`SimCfg::mapped`]: `alloc_host_mapped`, no H2D.
    pub mapped: bool,
    /// [`SimCfg::managed`]: `alloc_managed` + ReadMostly + prefetch.
    pub managed: bool,
    /// [`SimCfg::vmm`]: `va_acquire` / `va_acquire_paged` + H2D.
    pub vmm: bool,
    /// [`SimCfg::vmm_page`]: physical span for paged VMM (`0` = whole expert).
    pub vmm_page: u64,
    /// [`SimCfg::pageable`]: host-sync pageable H2D.
    pub pageable: bool,
    /// [`SimCfg::host_register`]: `cudaHostRegister` then pinned DMA H2D.
    pub host_register: bool,
    /// [`SimCfg::host_register_mapped`]: `alloc_host` then `cudaHostRegisterMapped`.
    pub host_register_mapped: bool,
    /// [`SimCfg::sync_memops`]: `cuPointerSetAttribute` SyncMemops after device alloc.
    pub sync_memops: bool,
    /// [`SimCfg::memcpy_batch`]: multi-expert prefetch uses `cudaMemcpyBatchAsync`.
    pub memcpy_batch: bool,
    /// [`SimCfg::accessed_by`]: SetAccessedBy / VMM SetAccess / mempool SetAccess
    /// on every GPU (fill or default pools).
    pub accessed_by: bool,
    /// [`SimCfg::wait_value`]: per-page 8-byte mailbox + live wait/write-value.
    pub wait_value: bool,
    /// [`SimCfg::stream_attach`]: `cudaStreamAttachMemAsync` Single then prefetch
    /// on this work stream (`StreamId(1)` when the plan would use NULL).
    pub stream_attach: bool,
    /// [`SimCfg::managed_host`]: `cudaMallocManaged` Host attach, then Global
    /// attach on this work stream before prefetch (unless stream-attach Single).
    pub managed_host: bool,
    /// [`SimCfg::prefetch_host`]: evict `cudaMemPrefetchAsync` to host; restore
    /// prefetches the same managed alloc back (no second `cudaMallocManaged`).
    pub prefetch_host: bool,
}

fn hbm_alloc(
    sim: &mut Sim,
    device: DeviceId,
    bytes: u64,
    stream: StreamId,
    sync: bool,
) -> Result<AllocId, Error> {
    if sync {
        Ok(sim.malloc(device, bytes)?)
    } else {
        Ok(sim.alloc(device, bytes, stream)?)
    }
}

fn hbm_h2d_pinned(
    sim: &mut Sim,
    device: DeviceId,
    alloc: AllocId,
    bytes: u64,
    stream: StreamId,
    sync: bool,
) -> Result<(), Error> {
    if sync {
        let _id = sim.memcpy_sync(
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
        )?;
    } else {
        let _c = sim.memcpy_pinned_to_device(device, alloc, bytes, stream)?;
    }
    Ok(())
}

fn hbm_h2d_many(sim: &mut Sim, args: TouchArgs, allocs: &[AllocId]) -> Result<(), Error> {
    if allocs.is_empty() {
        return Ok(());
    }
    let batch = args.memcpy_batch
        && !args.pageable
        && !args.sync_alloc
        && !args.mapped
        && !args.managed
        && allocs.len() >= 2;
    if batch {
        let ops: Vec<MemcpyOp> = allocs
            .iter()
            .map(|id| {
                MemcpyOp::packed_1d(Place::HostPinned, Place::Device(args.d), *id, args.bytes)
            })
            .collect();
        let attr = MemcpyAttributes::default();
        let _ids =
            sim.memcpy_batch_async(args.d, &ops, std::slice::from_ref(&attr), &[0], args.s)?;
        return Ok(());
    }
    for id in allocs {
        if args.pageable && !args.host_register {
            let _id = sim.memcpy_host_to_device(args.d, *id, args.bytes, args.s)?;
        } else {
            hbm_h2d_pinned(sim, args.d, *id, args.bytes, args.s, args.sync_alloc)?;
        }
    }
    Ok(())
}

/// `cudaMemAdviseSetAccessedBy` on every GPU so a remote read does not migrate.
pub(crate) fn advise_accessed_by(sim: &mut Sim, id: AllocId) -> Result<(), Error> {
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    for g in 0..n {
        sim.mem_advise(id, gpu_sim::MemAdvise::SetAccessedBy, DeviceId(g))?;
    }
    Ok(())
}

/// `cuMemSetAccess` PROT_READ on every GPU so a remote VMM read skips dest HBM.
pub(crate) fn advise_vmm_access(sim: &mut Sim, id: AllocId) -> Result<(), Error> {
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    for g in 0..n {
        sim.va_set_access(id, DeviceId(g))?;
    }
    Ok(())
}

/// POSIX-FD shareable pool per GPU, export, import sibling, `cudaDeviceSetMemPool`.
pub(crate) fn bind_shareable_mempools(sim: &mut Sim) -> Result<BTreeMap<DeviceId, PoolId>, Error> {
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    let mut imported = BTreeMap::new();
    for g in 0..n {
        let d = DeviceId(g);
        let pool = sim.create_shareable_pool(d)?;
        let h = sim.pool_export(pool)?;
        let imp = sim.pool_import(d, h)?;
        sim.set_device_mempool(d, pool)?;
        let _prev = imported.insert(d, imp);
    }
    Ok(imported)
}

/// `cudaMemPoolSetAccess` ReadWrite on every current device mempool.
pub(crate) fn advise_pool_access(sim: &mut Sim) -> Result<(), Error> {
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    for g in 0..n {
        let home = DeviceId(g);
        let pool = sim.device_mempool(home)?;
        for d in 0..n {
            sim.pool_set_access(pool, DeviceId(d))?;
        }
    }
    Ok(())
}

/// Pinned `cudaMallocAsync` + `--accessed-by` (not mapped/managed/VMM/`cudaMalloc`).
pub(crate) fn advise_pool_access_if_pinned(sim: &mut Sim, cfg: &SimCfg) -> Result<(), Error> {
    if cfg.accessed_by && !cfg.mapped && !cfg.managed && !cfg.vmm && !cfg.sync_alloc {
        advise_pool_access(sim)?;
    }
    Ok(())
}

/// `cudaFuncSetAttribute` NonPortableClusterSizeAllowed on every GPU.
pub(crate) fn allow_non_portable_cluster(sim: &mut Sim) -> Result<(), Error> {
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    for g in 0..n {
        sim.set_non_portable_cluster_size_allowed(DeviceId(g), true)?;
    }
    Ok(())
}

pub(crate) fn allow_non_portable_cluster_if(sim: &mut Sim, yes: bool) -> Result<(), Error> {
    if yes {
        allow_non_portable_cluster(sim)?;
    }
    Ok(())
}

/// `cudaFuncSetAttribute` MaxDynamicSharedMemorySize to the SKU opt-in max.
pub(crate) fn allow_optin_shared(sim: &mut Sim) -> Result<(), Error> {
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    for g in 0..n {
        let d = DeviceId(g);
        let optin = sim.profile().gpu(d)?.max_shared_mem_per_block_optin;
        sim.set_max_dynamic_shared_memory(d, optin)?;
    }
    Ok(())
}

pub(crate) fn allow_optin_shared_if(sim: &mut Sim, yes: bool) -> Result<(), Error> {
    if yes {
        allow_optin_shared(sim)?;
    }
    Ok(())
}

/// `cudaFuncSetAttribute(..., cudaFuncAttributePreferredSharedMemoryCarveout)` MaxShared.
///
/// Call after [`Sim::new`]. Launch Default inherits this occupancy (Hyper-Q
/// exclusive). Capture-legal; kernel-only graphs stay legal.
pub(crate) fn apply_func_max_shared(sim: &mut Sim, on: bool) -> Result<(), Error> {
    if !on {
        return Ok(());
    }
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    for g in 0..n {
        sim.set_func_carveout(DeviceId(g), SharedMemCarveout::MaxShared)?;
    }
    Ok(())
}

/// `cudaFuncSetAttribute(..., cudaFuncAttributeClusterSchedulingPolicyPreference)` Spread.
///
/// Call after [`Sim::new`]. Launch Default inherits this occupancy (Hyper-Q
/// exclusive when cluster blocks `> 1`). Capture-legal; kernel-only graphs
/// stay legal.
pub(crate) fn apply_func_cluster_spread(sim: &mut Sim, on: bool) -> Result<(), Error> {
    if !on {
        return Ok(());
    }
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    for g in 0..n {
        sim.set_func_cluster_policy(DeviceId(g), ClusterSchedulingPolicy::Spread)?;
    }
    Ok(())
}

/// `cudaFuncSetSharedMemConfig` so launch Default inherits bank-width duration.
///
/// Call after [`Sim::new`]. Capture cannot include it; construction is live.
/// Launch FourByte / EightByte still override.
pub(crate) fn apply_func_shared_mem(sim: &mut Sim, mode: SharedMemoryMode) -> Result<(), Error> {
    if mode == SharedMemoryMode::Default {
        return Ok(());
    }
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    for g in 0..n {
        sim.set_func_shared_mem_config(DeviceId(g), mode)?;
    }
    Ok(())
}

/// `cudaDeviceSetSharedMemConfig` so launch Default inherits when function is Default.
///
/// Call after [`Sim::new`]. Capture cannot include it; construction is live.
/// Function FourByte / EightByte still override. Launch FourByte / EightByte
/// still override.
pub(crate) fn apply_device_shared_mem(sim: &mut Sim, mode: SharedMemoryMode) -> Result<(), Error> {
    if mode == SharedMemoryMode::Default {
        return Ok(());
    }
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    for g in 0..n {
        sim.set_shared_mem_config(DeviceId(g), mode)?;
    }
    Ok(())
}

/// `cudaCtxResetPersistingL2Cache` on `device` after a live GEMM.
///
/// Capture cannot include it; call after `launch_graph` / stream kernel.
pub(crate) fn reset_persisting_l2_if(
    sim: &mut Sim,
    device: DeviceId,
    on: bool,
) -> Result<(), Error> {
    if !on {
        return Ok(());
    }
    sim.reset_persisting_l2_cache(device)?;
    Ok(())
}

#[derive(Clone, Copy, Default)]
pub(crate) struct ReplayCounters {
    pub hits: u64,
    pub misses: u64,
    pub prefetches: u64,
    pub prefetch_hits: u64,
    pub prefetch_waste: u64,
    pub graph_launches: u64,
    pub child_graphs: u64,
    pub graph_updates: u64,
    pub graph_clones: u64,
    pub graph_set_params: u64,
}

/// Instantiated CUDA graph execs, optionally parked for `update_graph`.
pub(crate) struct GraphBank {
    graphs: BTreeMap<Vec<AllocId>, (GraphId, (DeviceId, StreamId))>,
    idle: BTreeMap<(DeviceId, StreamId), Vec<GraphId>>,
    update: bool,
    clone: bool,
    build: bool,
    piecewise: bool,
    mem: LeafMem,
    cooperative: bool,
    pdl: bool,
    l2_persist: bool,
    l2_reset: bool,
    cluster: u8,
    preferred_cluster: u8,
    cluster_spread: bool,
    max_shared: bool,
    shared_mem: gpu_sim::SharedMemoryMode,
    portable_cluster: gpu_sim::PortableClusterMode,
    dynamic_shared: u32,
    portable_shared: gpu_sim::PortableSharedMode,
    nvlink_util_centric: bool,
    device_updatable: bool,
    kernel_priority: Option<i32>,
    device_launch: bool,
    launch_completion: Option<LaunchCompletionEvent>,
    programmatic_event: Option<ProgrammaticEvent>,
    set_params: bool,
    enable: bool,
    pub updates: u64,
    pub clones: u64,
    pub kernel_sets: u64,
}

impl GraphBank {
    pub(crate) fn new(update: bool, clone: bool, build: bool, mem: LeafMem) -> Self {
        Self {
            graphs: BTreeMap::new(),
            idle: BTreeMap::new(),
            update,
            clone,
            build,
            piecewise: false,
            mem,
            cooperative: false,
            pdl: false,
            l2_persist: false,
            l2_reset: false,
            cluster: 0,
            preferred_cluster: 0,
            cluster_spread: false,
            max_shared: false,
            shared_mem: gpu_sim::SharedMemoryMode::Default,
            portable_cluster: gpu_sim::PortableClusterMode::Default,
            dynamic_shared: 0,
            portable_shared: gpu_sim::PortableSharedMode::Default,
            nvlink_util_centric: false,
            device_updatable: false,
            kernel_priority: None,
            device_launch: false,
            launch_completion: None,
            programmatic_event: None,
            set_params: false,
            enable: false,
            updates: 0,
            clones: 0,
            kernel_sets: 0,
        }
    }

    pub(crate) fn with_cooperative(mut self, yes: bool) -> Self {
        self.cooperative = yes;
        self
    }

    pub(crate) fn with_pdl(mut self, yes: bool) -> Self {
        self.pdl = yes;
        self
    }

    pub(crate) fn with_l2_persist(mut self, yes: bool) -> Self {
        self.l2_persist = yes;
        self
    }

    pub(crate) fn with_l2_reset(mut self, yes: bool) -> Self {
        self.l2_reset = yes;
        self
    }

    pub(crate) fn with_cluster(mut self, n: u8) -> Self {
        self.cluster = n;
        self
    }

    pub(crate) fn with_preferred_cluster(mut self, n: u8) -> Self {
        self.preferred_cluster = n;
        self
    }

    pub(crate) fn with_cluster_spread(mut self, yes: bool) -> Self {
        self.cluster_spread = yes;
        self
    }

    pub(crate) fn with_max_shared(mut self, yes: bool) -> Self {
        self.max_shared = yes;
        self
    }

    pub(crate) fn with_shared_mem(mut self, mode: SharedMemoryMode) -> Self {
        self.shared_mem = mode;
        self
    }

    pub(crate) fn with_portable_cluster(mut self, mode: PortableClusterMode) -> Self {
        self.portable_cluster = mode;
        self
    }

    pub(crate) fn with_dynamic_shared(mut self, bytes: u32) -> Self {
        self.dynamic_shared = bytes;
        self
    }

    pub(crate) fn with_portable_shared(mut self, mode: PortableSharedMode) -> Self {
        self.portable_shared = mode;
        self
    }

    pub(crate) fn with_nvlink_util(mut self, yes: bool) -> Self {
        self.nvlink_util_centric = yes;
        self
    }

    pub(crate) fn with_device_updatable(mut self, yes: bool) -> Self {
        self.device_updatable = yes;
        self
    }

    pub(crate) fn with_device_launch(mut self, yes: bool) -> Self {
        self.device_launch = yes;
        self
    }

    pub(crate) fn with_launch_completion(mut self, ev: Option<LaunchCompletionEvent>) -> Self {
        self.launch_completion = ev;
        self
    }

    pub(crate) fn with_programmatic_event(mut self, ev: Option<ProgrammaticEvent>) -> Self {
        self.programmatic_event = ev;
        self
    }

    fn gemm_flags(&self) -> GemmFlags {
        GemmFlags {
            cooperative: self.cooperative,
            pdl: self.pdl,
            l2_persist: self.l2_persist,
            cluster: self.cluster,
            preferred_cluster: self.preferred_cluster,
            cluster_spread: self.cluster_spread,
            max_shared: self.max_shared,
            shared_mem: self.shared_mem,
            portable_cluster: self.portable_cluster,
            dynamic_shared: self.dynamic_shared,
            portable_shared: self.portable_shared,
            nvlink_util_centric: self.nvlink_util_centric,
            device_updatable: self.device_updatable,
            priority: self.kernel_priority,
            launch_completion: self.launch_completion,
            programmatic_event: self.programmatic_event,
        }
    }

    pub(crate) fn with_kernel_priority(mut self, pri: Option<i32>) -> Self {
        self.kernel_priority = pri;
        self
    }

    pub(crate) fn with_set_params(mut self, yes: bool) -> Self {
        self.set_params = yes;
        self
    }

    pub(crate) fn with_piecewise(mut self, yes: bool) -> Self {
        self.piecewise = yes;
        self
    }

    pub(crate) fn with_enable(mut self, yes: bool) -> Self {
        self.enable = yes;
        self
    }

    pub(crate) fn get(&self, ids: &[AllocId]) -> Option<GraphId> {
        self.graphs.get(ids).map(|(g, _)| *g)
    }

    fn alloc_of(&self, gid: GraphId) -> Option<AllocId> {
        self.graphs.iter().find_map(|(ids, (g, _))| {
            if *g == gid && ids.len() == 1 {
                ids.first().copied()
            } else {
                None
            }
        })
    }

    /// Smallest stored combo (`len >= 2`, same origin) that is a set superset of `want`.
    fn find_cover(&self, origin: (DeviceId, StreamId), want: &[AllocId]) -> Option<GraphId> {
        let want: BTreeSet<AllocId> = want.iter().copied().collect();
        if want.is_empty() {
            return None;
        }
        let mut best: Option<(&Vec<AllocId>, GraphId)> = None;
        for (ids, (gid, o)) in &self.graphs {
            if *o != origin || ids.len() < 2 {
                continue;
            }
            if !want.iter().all(|id| ids.contains(id)) {
                continue;
            }
            let better = match best {
                None => true,
                Some((b, _)) if ids.len() < b.len() => true,
                Some((b, _)) if ids.len() == b.len() && ids.as_slice() < b.as_slice() => true,
                _ => false,
            };
            if better {
                best = Some((ids, *gid));
            }
        }
        best.map(|(_, g)| g)
    }

    /// Enable child graphs whose leaf alloc is in `want`; skip the rest.
    ///
    /// Child index is not cover order: first-token expert order may differ
    /// from a later sorted resident cover. Only bills `graph_set_params_ns`
    /// when [`gpu_sim::Sim::graph_node_get_enabled`] disagrees.
    fn apply_child_enable(
        &self,
        sim: &mut Sim,
        exec: GraphId,
        want: &[AllocId],
    ) -> Result<(), Error> {
        if !self.enable {
            return Ok(());
        }
        let want: BTreeSet<AllocId> = want.iter().copied().collect();
        let children = sim.graph_child_nodes(exec)?;
        for (node, child) in children {
            let Some(alloc) = self.alloc_of(child) else {
                return Err(Error::Store("combo child alloc"));
            };
            let on = want.contains(&alloc);
            if sim.graph_node_get_enabled(exec, node)? != on {
                sim.graph_node_set_enabled(exec, node, on)?;
            }
        }
        Ok(())
    }

    /// Instantiate `src`, or `update_graph` / SetParams a parked leaf on `origin`.
    pub(crate) fn bind(
        &mut self,
        sim: &mut Sim,
        origin: (DeviceId, StreamId),
        ids: Vec<AllocId>,
        src: GraphId,
    ) -> Result<GraphId, Error> {
        if let Some(gid) = self.get(&ids) {
            sim.destroy_graph(src)?;
            return Ok(gid);
        }
        if self.set_params {
            if let Some(exec) = self.pop_matching(sim, origin, ids.len())? {
                if ids.len() == 1 {
                    let id = ids
                        .first()
                        .copied()
                        .ok_or(Error::Store("empty graph bind"))?;
                    retarget_parked_kernel(sim, exec, id)?;
                } else {
                    retarget_parked_children(sim, exec, src)?;
                }
                sim.destroy_graph(src)?;
                self.kernel_sets = self.kernel_sets.saturating_add(1);
                upload_after_set_params(sim, exec, self.device_updatable)?;
                let _prev = self.graphs.insert(ids, (exec, origin));
                return Ok(exec);
            }
        }
        let gid = if self.update && self.mem == LeafMem::None && ids.len() == 1 {
            if let Some(exec) = self.idle.entry(origin).or_default().pop() {
                sim.update_graph(exec, src)?;
                sim.destroy_graph(src)?;
                self.updates = self.updates.saturating_add(1);
                sim.upload_graph(exec)?;
                exec
            } else {
                self.instantiate_leaf(sim, &ids, src)?
            }
        } else {
            self.instantiate_leaf(sim, &ids, src)?
        };
        let _prev = self.graphs.insert(ids, (gid, origin));
        Ok(gid)
    }

    /// Retarget a parked leaf without capturing a second graph.
    pub(crate) fn try_retarget(
        &mut self,
        sim: &mut Sim,
        origin: (DeviceId, StreamId),
        id: AllocId,
    ) -> Result<Option<GraphId>, Error> {
        if !self.set_params {
            return Ok(None);
        }
        let Some(exec) = self.pop_matching(sim, origin, 1)? else {
            return Ok(None);
        };
        retarget_parked_kernel(sim, exec, id)?;
        self.kernel_sets = self.kernel_sets.saturating_add(1);
        upload_after_set_params(sim, exec, self.device_updatable)?;
        let _prev = self.graphs.insert(vec![id], (exec, origin));
        Ok(Some(exec))
    }

    fn instantiate_leaf(
        &mut self,
        sim: &mut Sim,
        ids: &[AllocId],
        src: GraphId,
    ) -> Result<GraphId, Error> {
        let exec = if self.clone && ids.len() == 1 {
            let cloned = sim.clone_graph(src)?;
            sim.destroy_graph(src)?;
            self.clones = self.clones.saturating_add(1);
            cloned
        } else {
            src
        };
        instantiate_exec(
            sim,
            exec,
            self.mem == LeafMem::AutoFree,
            self.device_launch,
            self.kernel_priority.is_some(),
        )
    }

    fn parks(&self, n_ids: usize) -> bool {
        self.set_params || (n_ids == 1 && self.update && self.mem == LeafMem::None)
    }

    /// Pop a parked exec whose child-graph count matches a leaf (`n_ids == 1`)
    /// or a combo parent (`n_ids` children).
    fn pop_matching(
        &mut self,
        sim: &Sim,
        origin: (DeviceId, StreamId),
        n_ids: usize,
    ) -> Result<Option<GraphId>, Error> {
        let idle = self.idle.entry(origin).or_default();
        let mut skipped = Vec::new();
        let mut found = None;
        while let Some(exec) = idle.pop() {
            let n_child = sim.graph_child_nodes(exec)?.len();
            let ok = if n_ids == 1 {
                n_child == 0
            } else {
                n_child == n_ids
            };
            if ok {
                found = Some(exec);
                break;
            }
            skipped.push(exec);
        }
        for e in skipped.into_iter().rev() {
            idle.push(e);
        }
        Ok(found)
    }

    pub(crate) fn drop_alloc(&mut self, sim: &mut Sim, id: AllocId) -> Result<(), Error> {
        let victims: Vec<(Vec<AllocId>, GraphId, (DeviceId, StreamId))> = self
            .graphs
            .iter()
            .filter(|(ids, _)| ids.contains(&id))
            .map(|(ids, (g, o))| (ids.clone(), *g, *o))
            .collect();
        for (ids, gid, origin) in victims {
            let _gone = self.graphs.remove(&ids);
            if self.parks(ids.len()) {
                self.idle.entry(origin).or_default().push(gid);
            } else {
                sim.destroy_graph(gid)?;
            }
        }
        Ok(())
    }
}

pub(crate) fn instantiate_exec(
    sim: &mut Sim,
    src: GraphId,
    auto_free: bool,
    device_launch: bool,
    use_node_priority: bool,
) -> Result<GraphId, Error> {
    let mut flags = 0u32;
    if auto_free {
        flags |= GraphInstantiateFlags::AUTO_FREE_ON_LAUNCH;
    }
    if device_launch {
        flags |= GraphInstantiateFlags::DEVICE_LAUNCH;
    }
    if use_node_priority {
        flags |= GraphInstantiateFlags::USE_NODE_PRIORITY;
    }
    let exec = if flags == 0 {
        sim.instantiate_graph(src)?
    } else {
        sim.instantiate_graph_with_flags(src, flags)?
    };
    sim.upload_graph(exec)?;
    Ok(exec)
}

pub(crate) fn upload_after_set_params(
    sim: &mut Sim,
    exec: GraphId,
    device_updatable: bool,
) -> Result<(), Error> {
    if device_updatable && sim.graph_uploaded(exec)? {
        return Ok(());
    }
    sim.upload_graph(exec)?;
    Ok(())
}

pub(crate) fn replay_exec(
    sim: &mut Sim,
    exec: GraphId,
    device: DeviceId,
    stream: StreamId,
    device_launch: bool,
) -> Result<(), Error> {
    if device_launch {
        if !sim.query_stream(device, stream)? {
            sim.synchronize_stream(device, stream)?;
        }
        let _n = sim.device_launch_graph(exec, stream)?;
    } else {
        let _n = sim.launch_graph(exec, stream)?;
    }
    Ok(())
}

/// Patch a parked leaf GEMM so it reads/writes `expert` instead of the evicted alloc.
///
/// Also retargets a unique memcpy node's [`gpu_sim::MemcpyOp::alloc`] or a
/// unique memset node's [`gpu_sim::MemsetOp`] when the leaf has one (not the
/// default compute GEMM capture).
pub(crate) fn retarget_parked_kernel(
    sim: &mut Sim,
    exec: GraphId,
    expert: AllocId,
) -> Result<(), Error> {
    let (node, mut params) = sim.graph_unique_kernel(exec)?;
    let owned = sim.graph_mem_allocs(exec)?;
    for buf in params.reads.iter_mut().chain(params.writes.iter_mut()) {
        if !owned.contains(&buf.id) {
            buf.id = expert;
        }
    }
    sim.graph_exec_kernel_set_params(exec, node, &params)?;
    if let Some((mnode, mut mop)) = sim.graph_try_unique_memcpy(exec)? {
        if !owned.contains(&mop.alloc) {
            mop.alloc = expert;
        }
        sim.graph_exec_memcpy_set_params(exec, mnode, &mop)?;
    }
    if let Some((znode, mut zbuf)) = sim.graph_try_unique_memset(exec)? {
        if !owned.contains(&zbuf.id) {
            zbuf.id = expert;
        }
        sim.graph_exec_memset_set_params(exec, znode, zbuf)?;
    }
    Ok(())
}

/// Patch a parked combo parent so its child-graph nodes name `src`'s children.
fn retarget_parked_children(sim: &mut Sim, exec: GraphId, src: GraphId) -> Result<(), Error> {
    let parked = sim.graph_child_nodes(exec)?;
    let fresh = sim.graph_child_nodes(src)?;
    if parked.len() != fresh.len() || parked.is_empty() {
        return Err(Error::Sim("child graph topology".into()));
    }
    for ((node, _), (_, child)) in parked.iter().zip(fresh.iter()) {
        sim.graph_exec_child_set_params(exec, *node, *child)?;
    }
    Ok(())
}

pub(crate) struct PageHandle {
    pub(crate) id: AllocId,
    pub(crate) stream: StreamId,
    pub(crate) device: DeviceId,
    /// Extra devices that hold a D2D / VMM-map replica of `id`.
    pub(crate) replicas: Vec<DeviceId>,
    /// 8-byte device mailbox when [`TouchArgs::wait_value`].
    pub(crate) mailbox: Option<AllocId>,
    /// Generation [`signal_copy_ready`] wrote; [`None`] after GEMM waited.
    pub(crate) ready_gen: Option<u64>,
    /// [`TouchArgs::prefetch_host`]: alloc is live on the host, not in HBM.
    pub(crate) host_resident: bool,
}

pub(crate) fn note_touch(
    ctr: &mut ReplayCounters,
    prefetched: &mut BTreeSet<ExpertKey>,
    key: ExpertKey,
    touch: Touch,
) {
    match touch {
        Touch::Hit => {
            ctr.hits = ctr.hits.saturating_add(1);
            if prefetched.remove(&key) {
                ctr.prefetch_hits = ctr.prefetch_hits.saturating_add(1);
            }
        }
        Touch::Miss { evicted } => {
            ctr.misses = ctr.misses.saturating_add(1);
            if let Some(v) = evicted {
                if prefetched.remove(&v) {
                    ctr.prefetch_waste = ctr.prefetch_waste.saturating_add(1);
                }
            }
            let _gone = prefetched.remove(&key);
        }
    }
}

fn should_prefetch(
    cfg: SimCfg,
    handles: &BTreeMap<ExpertKey, PageHandle>,
    trace: &Trace,
    i: usize,
) -> bool {
    if cfg.plan_window == 0 {
        return true;
    }
    let resident: BTreeSet<ExpertKey> = handles
        .iter()
        .filter(|(_, p)| !p.host_resident)
        .map(|(k, _)| *k)
        .collect();
    !matches!(
        plan_window(
            &resident,
            trace,
            i.saturating_add(1),
            cfg.plan_window,
            cfg.plan_threshold,
        ),
        Plan::Stay
    )
}

pub(crate) fn apply_touch(
    sim: &mut Sim,
    handles: &mut BTreeMap<ExpertKey, PageHandle>,
    graphs: &mut GraphBank,
    args: TouchArgs,
    key: ExpertKey,
    touch: Touch,
    next_event: &mut u32,
) -> Result<(), Error> {
    apply_misses(sim, handles, graphs, args, &[(key, touch)], next_event)
}

pub(crate) fn apply_misses(
    sim: &mut Sim,
    handles: &mut BTreeMap<ExpertKey, PageHandle>,
    graphs: &mut GraphBank,
    args: TouchArgs,
    misses: &[(ExpertKey, Touch)],
    next_event: &mut u32,
) -> Result<(), Error> {
    let mut pending: Vec<(ExpertKey, Option<AllocId>)> = Vec::new();
    for (key, touch) in misses {
        match touch {
            Touch::Hit => {}
            Touch::Miss { evicted } => {
                if let Some(v) = evicted {
                    reclaim_victim(sim, handles, graphs, args, *v, next_event)?;
                }
                if args.slots == 0 {
                    continue;
                }
                if args.prefetch_host && args.managed {
                    if let Some(page) = handles.get_mut(key) {
                        if page.host_resident {
                            restore_managed_from_host(sim, page, args)?;
                            continue;
                        }
                    }
                }
                let mailbox = if args.wait_value {
                    Some(alloc_copy_mailbox(sim, args.d, args.s, args.sync_alloc)?)
                } else {
                    None
                };
                pending.push((*key, mailbox));
            }
        }
    }
    if args.wait_value && !args.sync_alloc && pending.iter().any(|(_, mb)| mb.is_some()) {
        // Pointer must be resident before a later compute-stream wait
        // (`decode_priority` GEMM stream ≠ H2D stream). Do this before H2D
        // so the host wait is the 8-byte alloc, not the DMA.
        sim.synchronize_stream(args.d, args.s)?;
    }
    let mut filled: Vec<(ExpertKey, AllocId, Option<AllocId>)> = Vec::new();
    for (key, mailbox) in pending {
        let id = alloc_touch_page(sim, args)?;
        filled.push((key, id, mailbox));
    }
    let h2d: Vec<AllocId> = filled
        .iter()
        .filter_map(|(_, id, _)| {
            if args.mapped || args.managed {
                None
            } else {
                Some(*id)
            }
        })
        .collect();
    if args.managed {
        for (_, id, _) in &filled {
            let _p = sim.prefetch(args.d, *id, args.s)?;
        }
    } else if !args.mapped {
        hbm_h2d_many(sim, args, &h2d)?;
    }
    for (key, id, mailbox) in filled {
        if args.vmm && args.accessed_by {
            advise_vmm_access(sim, id)?;
        }
        let (mailbox, ready_gen) = if let Some(mb) = mailbox {
            signal_copy_ready(sim, args.d, mb, 1, args.s)?;
            (Some(mb), Some(1))
        } else {
            (None, None)
        };
        let _prev = handles.insert(
            key,
            PageHandle {
                id,
                stream: args.s,
                device: args.d,
                replicas: Vec::new(),
                mailbox,
                ready_gen,
                host_resident: false,
            },
        );
    }
    Ok(())
}

fn alloc_touch_page(sim: &mut Sim, args: TouchArgs) -> Result<AllocId, Error> {
    if args.mapped {
        if args.host_register_mapped {
            let h = sim.alloc_host(args.bytes)?;
            sim.host_register_mapped(h)?;
            return Ok(h);
        }
        return Ok(sim.alloc_host_mapped(args.bytes)?);
    }
    let id = if args.managed {
        let id = if args.managed_host {
            sim.alloc_managed_host(args.bytes)?
        } else {
            sim.alloc_managed(args.bytes)?
        };
        sim.mem_advise(id, gpu_sim::MemAdvise::SetReadMostly, args.d)?;
        sim.mem_advise(id, gpu_sim::MemAdvise::SetPreferredLocation, args.d)?;
        if args.accessed_by {
            advise_accessed_by(sim, id)?;
        }
        attach_managed_visibility(
            sim,
            args.d,
            id,
            args.s,
            args.stream_attach,
            args.managed_host,
        )?;
        id
    } else if args.vmm {
        if args.vmm_page > 0 && args.vmm_page < args.bytes {
            sim.va_acquire_paged(args.d, args.bytes, args.vmm_page)?
        } else {
            sim.va_acquire(args.d, args.bytes)?
        }
    } else {
        hbm_alloc(sim, args.d, args.bytes, args.s, args.sync_alloc)?
    };
    mark_sync_memops(sim, id, args.d, args.s, args.sync_memops)?;
    Ok(id)
}

/// `cuPointerSetAttribute` SyncMemops on a miss page before H2D / prefetch.
///
/// `cudaMallocAsync` pointers are not live until the alloc op completes, so
/// this waits `stream` when GetAttribute is not yet legal (copy stream only;
/// leftover compute is not drained).
pub(crate) fn mark_sync_memops(
    sim: &mut Sim,
    id: AllocId,
    device: DeviceId,
    stream: StreamId,
    on: bool,
) -> Result<(), Error> {
    if !on {
        return Ok(());
    }
    if sim
        .pointer_get_attribute(id, PointerAttr::SyncMemops)
        .is_err()
    {
        sim.synchronize_stream(device, stream)?;
    }
    sim.pointer_set_attribute(id, PointerAttr::SyncMemops, 1)?;
    Ok(())
}

/// `cudaSetDeviceFlags(cudaDeviceSyncMemops)` on every GPU.
///
/// Call after [`Sim::new`] so memcpy/memset (including unmarked pointers)
/// host-wait the submitting stream. Capture of those copies is refused;
/// kernel-only graphs stay legal.
pub(crate) fn apply_device_sync_memops(sim: &mut Sim, on: bool) -> Result<(), Error> {
    if !on {
        return Ok(());
    }
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    for g in 0..n {
        let d = DeviceId(g);
        let flags = sim.get_device_flags(d)?;
        sim.set_device_flags(d, flags | DeviceFlags::SYNC_MEMOPS)?;
    }
    Ok(())
}

/// Free `victim` on `device` only (replica) or the whole page (home).
pub(crate) fn reclaim_victim(
    sim: &mut Sim,
    handles: &mut BTreeMap<ExpertKey, PageHandle>,
    graphs: &mut GraphBank,
    args: TouchArgs,
    victim: ExpertKey,
    next_event: &mut u32,
) -> Result<(), Error> {
    let home = handles.get(&victim).map(|p| p.device);
    match home {
        None => Ok(()),
        Some(h) if h == args.d => {
            if args.prefetch_host && args.managed {
                if let Some(page) = handles.get_mut(&victim) {
                    if !page.host_resident {
                        evict_managed_to_host(sim, page)?;
                    }
                    return Ok(());
                }
            }
            let Some(page) = handles.remove(&victim) else {
                return Ok(());
            };
            drop_handle(sim, graphs, page, next_event, args.sync_alloc)
        }
        Some(_) => {
            let Some(page) = handles.get_mut(&victim) else {
                return Ok(());
            };
            drop_replica(sim, page, args.d, next_event, args.bytes)
        }
    }
}

fn drop_handle(
    sim: &mut Sim,
    graphs: &mut GraphBank,
    page: PageHandle,
    next_event: &mut u32,
    sync: bool,
) -> Result<(), Error> {
    graphs.drop_alloc(sim, page.id)?;
    if let Some(mb) = page.mailbox {
        free_copy_mailbox(sim, page.device, mb, page.stream, sync)?;
    }
    if page_is_mapped(sim, page.id) {
        // cudaFreeHost waits GPU work on this pointer, then the mapping is gone.
        // Registered mapped pages must cudaHostUnregister then free pageable host.
        sim.synchronize_stream(page.device, page.stream)?;
        free_mapped_host(sim, page.id)?;
        return Ok(());
    }
    if page_is_managed(sim, page.id) {
        sim.free_sync(page.id)?;
        return Ok(());
    }
    if page_is_vmm(sim, page.id) {
        sim.va_release(page.id)?;
        return Ok(());
    }
    if sync {
        // cudaFree: one host call, every device copy is gone.
        sim.free_sync(page.id)?;
        return Ok(());
    }
    for dst in &page.replicas {
        if *dst == page.device {
            continue;
        }
        wait_peer(sim, page.device, *dst, page.stream, next_event)?;
        sim.free(*dst, page.id, page.stream)?;
    }
    sim.free(page.device, page.id, page.stream)?;
    Ok(())
}

fn evict_managed_to_host(sim: &mut Sim, page: &mut PageHandle) -> Result<(), Error> {
    sim.synchronize_device(page.device)?;
    page.replicas.clear();
    let _p = sim.prefetch_host(page.device, page.id, page.stream)?;
    sim.synchronize_stream(page.device, page.stream)?;
    page.host_resident = true;
    page.ready_gen = None;
    Ok(())
}

fn restore_managed_from_host(
    sim: &mut Sim,
    page: &mut PageHandle,
    args: TouchArgs,
) -> Result<(), Error> {
    let stream = bump_null_for_attach(args.s, args.stream_attach);
    ensure_single_attach(sim, args.d, page.id, stream)?;
    let _p = sim.prefetch(args.d, page.id, stream)?;
    page.host_resident = false;
    page.device = args.d;
    page.stream = stream;
    if args.wait_value {
        if let Some(mb) = page.mailbox {
            signal_copy_ready(sim, args.d, mb, 1, stream)?;
            page.ready_gen = Some(1);
        }
    }
    Ok(())
}

fn drop_replica(
    sim: &mut Sim,
    page: &mut PageHandle,
    dst: DeviceId,
    next_event: &mut u32,
    bytes: u64,
) -> Result<(), Error> {
    if !page.replicas.contains(&dst) {
        return Ok(());
    }
    wait_peer(sim, page.device, dst, page.stream, next_event)?;
    if page_is_managed(sim, page.id) {
        sim.drop_managed_copy(page.id, dst)?;
    } else if page_is_vmm(sim, page.id) {
        sim.va_unmap_range(page.id, dst, 0, bytes)?;
    } else {
        sim.free(dst, page.id, page.stream)?;
    }
    page.replicas.retain(|d| *d != dst);
    Ok(())
}

fn page_is_mapped(sim: &Sim, id: AllocId) -> bool {
    sim.is_host_mapped(id).unwrap_or(false)
}

/// Drop a mapped host page: `cudaFreeHost`, or unregister then `free_host`
/// when the id came from `cudaHostRegisterMapped`.
pub(crate) fn free_mapped_host(sim: &mut Sim, id: AllocId) -> Result<(), Error> {
    if sim.is_host_registered(id)? {
        sim.host_unregister(id)?;
        sim.free_host(id)?;
    } else {
        sim.free_host_pinned(id)?;
    }
    Ok(())
}

fn page_is_managed(sim: &Sim, id: AllocId) -> bool {
    sim.is_managed(id).unwrap_or(false)
}

fn page_is_vmm(sim: &Sim, id: AllocId) -> bool {
    sim.is_vmm(id).unwrap_or(false)
}

/// Single attach cannot use the NULL stream; store compute is `StreamId(1)`.
pub(crate) fn bump_null_for_attach(s: StreamId, stream_attach: bool) -> StreamId {
    if stream_attach && s == StreamId::NULL {
        StreamId(1)
    } else {
        s
    }
}

/// Host-attached managed pages need a later Global/Single attach before
/// device prefetch. Stream-attach Single wins over a Host→Global attach.
fn attach_managed_visibility(
    sim: &mut Sim,
    device: DeviceId,
    id: AllocId,
    stream: StreamId,
    stream_attach: bool,
    managed_host: bool,
) -> Result<(), Error> {
    if stream_attach {
        let _a = sim.stream_attach(device, id, stream, MemAttach::Single)?;
    } else if managed_host {
        let _a = sim.stream_attach(device, id, stream, MemAttach::Global)?;
    }
    Ok(())
}

/// Live `cudaStreamAttachMemAsync` Single onto `stream` when the page is already
/// Single-attached elsewhere (prefill vs decode). Capture cannot include attach.
pub(crate) fn ensure_single_attach(
    sim: &mut Sim,
    device: DeviceId,
    id: AllocId,
    stream: StreamId,
) -> Result<(), Error> {
    if !sim.is_managed(id)? {
        return Ok(());
    }
    if sim.mem_attach(id)? != MemAttach::Single {
        return Ok(());
    }
    if sim.is_attached_to(id, stream)? {
        return Ok(());
    }
    let _a = sim.stream_attach(device, id, stream, MemAttach::Single)?;
    Ok(())
}

pub(crate) fn gemm_keys(
    sim: &mut Sim,
    handles: &mut BTreeMap<ExpertKey, PageHandle>,
    graphs: &mut GraphBank,
    keys: &[ExpertKey],
    cuda_graphs: bool,
    ctr: &mut ReplayCounters,
    work: Option<StreamId>,
) -> Result<(), Error> {
    for key in keys {
        let Some(page) = handles.get_mut(key) else {
            continue;
        };
        if page.host_resident {
            continue;
        }
        if let (Some(mb), Some(gen)) = (page.mailbox, page.ready_gen.take()) {
            let stream = work.unwrap_or(page.stream);
            wait_copy_ready(sim, page.device, mb, gen, stream)?;
        }
    }
    for key in keys {
        let Some(page) = handles.get(key) else {
            continue;
        };
        if page.host_resident {
            continue;
        }
        let stream = work.unwrap_or(page.stream);
        ensure_single_attach(sim, page.device, page.id, stream)?;
    }
    let mut by_dev: BTreeMap<(DeviceId, StreamId), Vec<AllocId>> = BTreeMap::new();
    for key in keys {
        let Some(page) = handles.get(key) else {
            continue;
        };
        if page.host_resident {
            continue;
        }
        let stream = work.unwrap_or(page.stream);
        by_dev
            .entry((page.device, stream))
            .or_default()
            .push(page.id);
    }
    for ((d, stream), ids) in by_dev {
        let cover = if graphs.enable {
            resident_cover(handles, d, stream, work)
        } else {
            Vec::new()
        };
        gemm_ids(sim, graphs, (d, stream), ids, cover, cuda_graphs, ctr)?;
        reset_persisting_l2_if(sim, d, graphs.l2_reset)?;
    }
    Ok(())
}

fn resident_cover(
    handles: &BTreeMap<ExpertKey, PageHandle>,
    d: DeviceId,
    stream: StreamId,
    work: Option<StreamId>,
) -> Vec<AllocId> {
    let mut ids: Vec<AllocId> = handles
        .values()
        .filter(|page| {
            !page.host_resident && page.device == d && work.unwrap_or(page.stream) == stream
        })
        .map(|page| page.id)
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

fn is_strict_superset(cover: &[AllocId], want: &[AllocId]) -> bool {
    let cover: BTreeSet<AllocId> = cover.iter().copied().collect();
    let want: BTreeSet<AllocId> = want.iter().copied().collect();
    !want.is_empty() && want.iter().all(|id| cover.contains(id)) && cover.len() > want.len()
}

pub(crate) fn host_callbacks(
    sim: &mut Sim,
    handles: &BTreeMap<ExpertKey, PageHandle>,
    keys: &[ExpertKey],
    work: Option<StreamId>,
) -> Result<(), Error> {
    let mut seen: BTreeSet<(DeviceId, StreamId)> = BTreeSet::new();
    for key in keys {
        let Some(page) = handles.get(key) else {
            continue;
        };
        let stream = work.unwrap_or(page.stream);
        if seen.insert((page.device, stream)) {
            let _id = sim.host_func(page.device, stream)?;
        }
    }
    Ok(())
}

fn gemm_ids(
    sim: &mut Sim,
    graphs: &mut GraphBank,
    origin: (DeviceId, StreamId),
    ids: Vec<AllocId>,
    cover: Vec<AllocId>,
    cuda_graphs: bool,
    ctr: &mut ReplayCounters,
) -> Result<(), Error> {
    if ids.is_empty() {
        return Ok(());
    }
    let (d, stream) = origin;
    if graphs.device_launch && ids.len() > 1 {
        for id in ids {
            gemm_ids(sim, graphs, origin, vec![id], Vec::new(), cuda_graphs, ctr)?;
        }
        return Ok(());
    }
    if let Some(g) = graphs.get(&ids) {
        graphs.apply_child_enable(sim, g, &ids)?;
        replay_exec(sim, g, d, stream, graphs.device_launch)?;
        ctr.graph_launches = ctr.graph_launches.saturating_add(1);
        return Ok(());
    }
    if graphs.enable {
        if let Some(g) = graphs.find_cover(origin, &ids) {
            graphs.apply_child_enable(sim, g, &ids)?;
            replay_exec(sim, g, d, stream, graphs.device_launch)?;
            ctr.graph_launches = ctr.graph_launches.saturating_add(1);
            return Ok(());
        }
    }
    if ids.len() == 1 {
        if let Some(id) = ids.first().copied() {
            if let Some(g) = graphs.try_retarget(sim, origin, id)? {
                replay_exec(sim, g, d, stream, graphs.device_launch)?;
                ctr.graph_launches = ctr.graph_launches.saturating_add(1);
                return Ok(());
            }
        }
    }
    let capture_ids = if graphs.enable && cover.len() > 1 && is_strict_superset(&cover, &ids) {
        cover
    } else {
        ids.clone()
    };
    if cuda_graphs
        || graphs.build
        || graphs.piecewise
        || graphs.mem != LeafMem::None
        || graphs.set_params
        || graphs.device_launch
        || graphs.device_updatable
        || graphs.enable
    {
        if let Some(g) = capture_expert_graph(sim, graphs, d, stream, &capture_ids)? {
            if capture_ids.len() > 1 {
                ctr.child_graphs = ctr.child_graphs.saturating_add(1);
            }
            graphs.apply_child_enable(sim, g, &ids)?;
            replay_exec(sim, g, d, stream, graphs.device_launch)?;
            ctr.graph_launches = ctr.graph_launches.saturating_add(1);
            return Ok(());
        }
    }
    for id in ids {
        kernel_leaf(
            sim,
            d,
            stream,
            id,
            LeafMem::None,
            graphs.gemm_flags().for_stream(),
        )?;
    }
    Ok(())
}

fn capture_expert_graph(
    sim: &mut Sim,
    graphs: &mut GraphBank,
    d: DeviceId,
    stream: StreamId,
    ids: &[AllocId],
) -> Result<Option<GraphId>, Error> {
    if graphs.build {
        return build_expert_graph(sim, graphs, d, stream, ids);
    }
    if !sim.stream_is_idle(d, stream)? {
        sim.synchronize_stream(d, stream)?;
    }
    if !sim.stream_is_idle(d, stream)? {
        return Ok(None);
    }
    let origin = (d, stream);
    let mut leaves = Vec::new();
    for id in ids {
        let key = vec![*id];
        if let Some(g) = graphs.get(&key) {
            leaves.push(g);
            continue;
        }
        if let Some(g) = graphs.try_retarget(sim, origin, *id)? {
            leaves.push(g);
            continue;
        }
        sim.begin_capture(d, stream)?;
        kernel_leaf(sim, d, stream, *id, graphs.mem, graphs.gemm_flags())?;
        let src = sim.end_capture()?;
        leaves.push(graphs.bind(sim, origin, key, src)?);
    }
    if ids.len() == 1 {
        return Ok(leaves.first().copied());
    }
    if graphs.piecewise {
        return piecewise_expert_graph(sim, graphs, d, stream, ids, &leaves);
    }
    sim.begin_capture(d, stream)?;
    for g in leaves {
        let _n = sim.launch_graph(g, stream)?;
    }
    let src = sim.end_capture()?;
    Ok(Some(graphs.bind(sim, origin, ids.to_vec(), src)?))
}

fn piecewise_expert_graph(
    sim: &mut Sim,
    graphs: &mut GraphBank,
    d: DeviceId,
    stream: StreamId,
    ids: &[AllocId],
    leaves: &[GraphId],
) -> Result<Option<GraphId>, Error> {
    let parent = sim.create_graph(d, stream)?;
    for g in leaves {
        sim.begin_capture_to_graph(d, stream, parent, &[])?;
        let _n = sim.launch_graph(*g, stream)?;
        let ended = sim.end_capture()?;
        if ended != parent {
            return Err(Error::Store("capture-to-graph id"));
        }
    }
    Ok(Some(graphs.bind(sim, (d, stream), ids.to_vec(), parent)?))
}

fn build_expert_graph(
    sim: &mut Sim,
    graphs: &mut GraphBank,
    d: DeviceId,
    stream: StreamId,
    ids: &[AllocId],
) -> Result<Option<GraphId>, Error> {
    let origin = (d, stream);
    let mut leaves = Vec::new();
    for id in ids {
        let key = vec![*id];
        if let Some(g) = graphs.get(&key) {
            leaves.push(g);
            continue;
        }
        if let Some(g) = graphs.try_retarget(sim, origin, *id)? {
            leaves.push(g);
            continue;
        }
        let src = sim.create_graph(d, stream)?;
        add_leaf_gemm(sim, src, *id, graphs.mem, graphs.gemm_flags())?;
        leaves.push(graphs.bind(sim, origin, key, src)?);
    }
    if ids.len() == 1 {
        return Ok(leaves.first().copied());
    }
    let parent = sim.create_graph(d, stream)?;
    for g in leaves {
        sim.graph_add_child(parent, g)?;
    }
    Ok(Some(graphs.bind(sim, origin, ids.to_vec(), parent)?))
}

fn finish(sim: &Sim, token_ends: &[u64], ctr: ReplayCounters) -> SimReplay {
    let n = u64::try_from(token_ends.len()).unwrap_or(0);
    replay_from_sim(
        sim,
        n,
        token_ends.first().copied(),
        itl_from_ends(token_ends),
        ctr,
    )
}

/// `cudaDeviceGraphMemTrim` on every GPU. Unused reserved returns to the OS.
pub(crate) fn trim_graph_pools(sim: &mut Sim) -> Result<(), Error> {
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    for g in 0..n {
        sim.graph_mem_trim(DeviceId(g))?;
    }
    Ok(())
}

/// `cudaMemPoolTrimTo(0)` on every GPU's current device mempool.
pub(crate) fn trim_device_pools(sim: &mut Sim) -> Result<(), Error> {
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    for g in 0..n {
        let d = DeviceId(g);
        let pool = sim.device_mempool(d)?;
        let _dropped = sim.pool_trim_to(pool, 0)?;
    }
    Ok(())
}

/// `cudaMemPoolReuseAllowOpportunistic=0` on every GPU's current device mempool.
pub(crate) fn disable_pool_opportunistic_reuse(sim: &mut Sim) -> Result<(), Error> {
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    for g in 0..n {
        let d = DeviceId(g);
        let pool = sim.device_mempool(d)?;
        sim.pool_set_attribute(pool, MemPoolAttr::ReuseAllowOpportunistic, 0)?;
    }
    Ok(())
}

pub(crate) fn replay_from_sim(
    sim: &Sim,
    n_tokens: u64,
    ttft: Option<u64>,
    itl: Option<u64>,
    ctr: ReplayCounters,
) -> SimReplay {
    let mut score = Score::from_sim(sim);
    if n_tokens > 0 {
        score = score.with_tokens(n_tokens);
    }
    if let Some(t) = ttft {
        score = score.with_latencies(t, itl);
    }
    SimReplay {
        sim_ns: score.wall_ns,
        bytes_moved: score.bytes_moved,
        hbm_peak: score.hbm_peak,
        energy_uj: score.energy_uj,
        usd_micros_per_m_tokens: score.usd_micros_per_m_tokens,
        ttft_ns: score.ttft_ns,
        itl_ns: score.itl_ns,
        hits: ctr.hits,
        misses: ctr.misses,
        prefetches: ctr.prefetches,
        prefetch_hits: ctr.prefetch_hits,
        prefetch_waste: ctr.prefetch_waste,
        graph_launches: ctr.graph_launches,
        child_graphs: ctr.child_graphs,
        graph_updates: ctr.graph_updates,
        graph_clones: ctr.graph_clones,
        graph_set_params: ctr.graph_set_params,
    }
}

fn last_of_token(events: &[ExpertAccess], i: usize) -> bool {
    let Some(cur) = events.get(i) else {
        return true;
    };
    match events.get(i.saturating_add(1)) {
        Some(n) => n.token != cur.token,
        None => true,
    }
}

fn engine_step(events: &[ExpertAccess], i: usize, max_batch: usize, admitted: usize) -> bool {
    if last_of_token(events, i) {
        return true;
    }
    if max_batch == 0 || admitted < max_batch {
        return false;
    }
    sequence_done(events, i)
}

fn sequence_done(events: &[ExpertAccess], i: usize) -> bool {
    let Some(cur) = events.get(i) else {
        return true;
    };
    match events.get(i.saturating_add(1)) {
        Some(n) => n.token != cur.token || n.sequence != cur.sequence,
        None => true,
    }
}

/// `cudaHostRegister` a pageable staging buffer so walker H2D can DMA.
///
/// Kept live for the pin tax; dropped with the Sim.
pub(crate) fn pin_pageable_staging(sim: &mut Sim, cfg: &SimCfg, bytes: u64) -> Result<(), Error> {
    if !cfg.host_register {
        return Ok(());
    }
    let h = sim.alloc_host(bytes)?;
    sim.host_register(h)?;
    Ok(())
}

/// `--mapped` occupancy: `min(slots, pin / expert_bytes)`.
///
/// When the pin budget cannot hold one expert (`fit == 0`), returns the
/// requested slots so the first `alloc_host_mapped` is [`gpu_sim::SimError::PinOom`].
pub(crate) fn occupancy_slots(cfg: &SimCfg, pin_bytes: u64) -> usize {
    if !cfg.mapped {
        return cfg.slots;
    }
    let bytes = cfg.bytes_per_expert.max(1);
    let fit = usize::try_from(pin_bytes / bytes).unwrap_or(usize::MAX);
    if fit == 0 {
        cfg.slots
    } else {
        cfg.slots.min(fit)
    }
}

pub(crate) fn sim_profile(profile: HardwareProfile, cfg: &SimCfg) -> HardwareProfile {
    if cfg.compute_slots > 0 {
        profile.with_compute_slots(cfg.compute_slots)
    } else {
        profile
    }
}

pub(crate) fn apply_stream_sms(
    sim: &mut Sim,
    plan: StreamPlan,
    permille: u16,
) -> Result<(), Error> {
    if permille == 0 {
        return Ok(());
    }
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    if plan.decode_priority {
        let dec = permille.min(1000);
        let pre = 1000u16.saturating_sub(dec).max(1);
        for g in 0..n {
            let d = DeviceId(g);
            sim.set_stream_sm_permille(d, plan.decode, dec)?;
            if plan.prefill != plan.decode {
                sim.set_stream_sm_permille(d, plan.prefill, pre)?;
            }
        }
        return Ok(());
    }
    for g in 0..n {
        for s in 0..plan.n_copy.max(1) {
            sim.set_stream_sm_permille(DeviceId(g), StreamId(u16::from(s)), permille)?;
        }
    }
    Ok(())
}

/// `cudaStreamSetAttribute` SynchronizationPolicy on copy/prefill/decode streams.
pub(crate) fn apply_stream_sync_policy(
    sim: &mut Sim,
    plan: StreamPlan,
    policy: SynchronizationPolicy,
) -> Result<(), Error> {
    if policy == SynchronizationPolicy::Auto {
        return Ok(());
    }
    sim.set_created_streams_sync_policy(plan.mark, policy)?;
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    for g in 0..n {
        let d = DeviceId(g);
        sim.set_stream_sync_policy(d, StreamId::NULL, policy)?;
        sim.set_stream_sync_policy(d, plan.prefill, policy)?;
        sim.set_stream_sync_policy(d, plan.decode, policy)?;
        for s in 0..plan.n_copy.max(1) {
            sim.set_stream_sync_policy(d, StreamId(u16::from(s)), policy)?;
        }
    }
    Ok(())
}

/// `cudaStreamSetAttribute` MemSyncDomain on the decode stream (prefill stays Default).
pub(crate) fn apply_stream_mem_sync_domain(
    sim: &mut Sim,
    plan: StreamPlan,
    domain: gpu_sim::MemSyncDomain,
) -> Result<(), Error> {
    if domain == gpu_sim::MemSyncDomain::Default {
        return Ok(());
    }
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    for g in 0..n {
        let d = DeviceId(g);
        if plan.decode_priority && plan.prefill != plan.decode {
            sim.set_stream_mem_sync_domain(d, plan.decode, domain)?;
        } else {
            sim.set_stream_mem_sync_domain(d, plan.prefill, domain)?;
        }
    }
    Ok(())
}

/// `cudaGraphExecKernelNodeSetAttribute` MemSyncDomain to the launch stream.
///
/// CUDA graphs bake capture-time domain. Store leaves are keyed by alloc, so a
/// prefill capture would otherwise replay Default on the decode stream.
/// Walker [`GraphBank`] is already per-stream. Combo parents are not unique
/// kernels; they recapture on the decode origin instead.
pub(crate) fn apply_exec_mem_sync_domain(
    sim: &mut Sim,
    exec: GraphId,
    device: DeviceId,
    stream: StreamId,
) -> Result<(), Error> {
    let want = sim.stream_mem_sync_domain(device, stream);
    let (node, _) = sim.graph_unique_kernel(exec)?;
    let have = sim.graph_exec_kernel_node_get_mem_sync_domain(exec, node)?;
    if have != want {
        sim.graph_exec_kernel_node_set_mem_sync_domain(exec, node, want)?;
    }
    Ok(())
}

/// Token-boundary drain: decode stream only when leftover prefill may still run.
pub(crate) fn sync_work(
    sim: &mut Sim,
    n_gpus: u16,
    plan: StreamPlan,
    decode_token: bool,
) -> Result<(), Error> {
    if plan.decode_priority && decode_token {
        for g in 0..n_gpus {
            sim.synchronize_stream(DeviceId(g), plan.decode)?;
        }
    } else {
        sim.synchronize()?;
    }
    Ok(())
}

/// Copy-engine count plus optional prefill/decode compute streams.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StreamPlan {
    n_copy: u8,
    /// Prefill compute stream (`n_copy` when decode-priority, else NULL).
    pub(crate) prefill: StreamId,
    /// Decode compute stream (`n_copy + 1` when decode-priority).
    pub(crate) decode: StreamId,
    /// Exclusive upper bound for `set_created_streams_*` (`1 .. mark`).
    pub(crate) mark: u8,
    decode_priority: bool,
}

impl StreamPlan {
    /// Prefill/decode streams when `decode_priority`, else seq-stream NULL mapping.
    pub(crate) fn new(profile: &HardwareProfile, seq_streams: bool, decode_priority: bool) -> Self {
        let n_copy = replay_streams(profile, seq_streams);
        if decode_priority {
            let prefill = StreamId(u16::from(n_copy));
            let decode = StreamId(u16::from(n_copy).saturating_add(1));
            Self {
                n_copy,
                prefill,
                decode,
                mark: n_copy.saturating_add(2),
                decode_priority: true,
            }
        } else {
            Self {
                n_copy,
                prefill: StreamId(0),
                decode: StreamId(0),
                mark: n_copy,
                decode_priority: false,
            }
        }
    }

    /// Work stream for this event: prefill vs decode, or `sequence % n_copy`.
    pub(crate) fn work(self, sequence: u64, token: u32) -> StreamId {
        if self.decode_priority {
            if token == 0 {
                self.prefill
            } else {
                self.decode
            }
        } else {
            stream_of(sequence, self.n_copy)
        }
    }
}

pub(crate) fn replay_streams(profile: &HardwareProfile, seq_streams: bool) -> u8 {
    if !seq_streams {
        return 1;
    }
    profile
        .gpus
        .first()
        .map(|g| g.copy_engines.max(2))
        .unwrap_or(2)
}

pub(crate) fn stream_of(sequence: u64, n_streams: u8) -> StreamId {
    if n_streams <= 1 {
        return StreamId(0);
    }
    let n = u64::from(n_streams);
    let id = sequence % n;
    StreamId(u16::try_from(id).unwrap_or(0))
}

fn itl_from_ends(ends: &[u64]) -> Option<u64> {
    if ends.len() < 2 {
        return None;
    }
    let first = *ends.first()?;
    let last = *ends.last()?;
    let n = u64::try_from(ends.len().saturating_sub(1)).ok()?;
    last.saturating_sub(first).checked_div(n.max(1))
}

fn gemm_kind() -> KernelKind {
    KernelKind::GroupedMoeGemm {
        experts: 1,
        tokens_per_expert: 1,
        hidden: 64,
        ff: 64,
        dtype: DType::Fp16,
    }
}

fn kernel(sim: &mut Sim, d: DeviceId, s: StreamId, id: AllocId) -> Result<(), Error> {
    ensure_single_attach(sim, d, id, s)?;
    kernel_leaf(sim, d, s, id, LeafMem::None, GemmFlags::default())
}

pub(crate) fn kernel_leaf(
    sim: &mut Sim,
    d: DeviceId,
    s: StreamId,
    id: AllocId,
    mem: LeafMem,
    flags: GemmFlags,
) -> Result<(), Error> {
    if mem == LeafMem::None {
        launch_gemm_kernel(sim, d, s, id, &[], flags)?;
        return Ok(());
    }
    let scratch = sim.alloc(d, GRAPH_SCRATCH_BYTES, s)?;
    launch_gemm_kernel(sim, d, s, id, &[scratch], flags)?;
    if mem == LeafMem::Free {
        sim.free(d, scratch, s)?;
    }
    Ok(())
}

fn launch_gemm_kernel(
    sim: &mut Sim,
    d: DeviceId,
    s: StreamId,
    id: AllocId,
    writes: &[gpu_sim::AllocId],
    flags: GemmFlags,
) -> Result<(), Error> {
    let _k = sim.kernel_with(d, gemm_kind(), &[id], writes, s, flags.kernel_attrs(id))?;
    Ok(())
}

pub(crate) fn add_leaf_gemm(
    sim: &mut Sim,
    graph: GraphId,
    id: AllocId,
    mem: LeafMem,
    flags: GemmFlags,
) -> Result<(), Error> {
    if mem == LeafMem::None {
        return add_gemm_kernel(sim, graph, id, &[], flags);
    }
    let scratch = sim.graph_add_alloc(graph, GRAPH_SCRATCH_BYTES)?;
    add_gemm_kernel(sim, graph, id, &[scratch], flags)?;
    sim.graph_add_dependencies(graph, 0, 1)?;
    if mem == LeafMem::Free {
        sim.graph_add_free(graph, scratch)?;
        sim.graph_add_dependencies(graph, 1, 2)?;
    }
    Ok(())
}

fn add_gemm_kernel(
    sim: &mut Sim,
    graph: GraphId,
    id: AllocId,
    writes: &[gpu_sim::AllocId],
    flags: GemmFlags,
) -> Result<(), Error> {
    if flags.cooperative {
        sim.graph_add_cooperative_kernel(graph, gemm_kind(), &[id], writes)?;
    } else {
        sim.graph_add_kernel(graph, gemm_kind(), &[id], writes)?;
        if let Some(pdl) = flags.pdl_attr() {
            let node = usize::from(!writes.is_empty());
            sim.graph_kernel_node_set_pdl(graph, node, pdl)?;
        }
    }
    if let Some(w) = flags.persist_window(id) {
        let node = usize::from(!writes.is_empty());
        sim.graph_kernel_node_set_access_policy(graph, node, Some(w))?;
    }
    if let Some(c) = flags.cluster_dim() {
        let node = usize::from(!writes.is_empty());
        sim.graph_kernel_node_set_cluster(graph, node, Some(c))?;
    }
    if let Some(p) = flags.preferred_cluster_dim() {
        let node = usize::from(!writes.is_empty());
        sim.graph_kernel_node_set_preferred_cluster(graph, node, Some(p))?;
    }
    if flags.cluster_spread {
        let node = usize::from(!writes.is_empty());
        sim.graph_kernel_node_set_cluster_policy(graph, node, ClusterSchedulingPolicy::Spread)?;
    }
    if flags.max_shared {
        let node = usize::from(!writes.is_empty());
        sim.graph_kernel_node_set_carveout(graph, node, SharedMemCarveout::MaxShared)?;
    }
    if flags.shared_mem != SharedMemoryMode::Default {
        let node = usize::from(!writes.is_empty());
        sim.graph_kernel_node_set_shared_mem(graph, node, flags.shared_mem)?;
    }
    if flags.portable_cluster != PortableClusterMode::Default {
        let node = usize::from(!writes.is_empty());
        sim.graph_kernel_node_set_portable_cluster(graph, node, flags.portable_cluster)?;
    }
    if flags.portable_shared != PortableSharedMode::Default {
        let node = usize::from(!writes.is_empty());
        sim.graph_kernel_node_set_portable_shared(graph, node, flags.portable_shared)?;
    }
    if flags.nvlink_util_centric {
        let node = usize::from(!writes.is_empty());
        sim.graph_kernel_node_set_nvlink_util_centric(graph, node, true)?;
    }
    if flags.device_updatable {
        let node = usize::from(!writes.is_empty());
        sim.graph_kernel_node_set_device_updatable(graph, node, true)?;
    }
    if flags.dynamic_shared > 0 {
        let node = usize::from(!writes.is_empty());
        sim.graph_kernel_node_set_dynamic_shared(graph, node, flags.dynamic_shared)?;
    }
    if let Some(ev) = flags.launch_completion {
        let node = usize::from(!writes.is_empty());
        sim.graph_kernel_node_set_launch_completion(graph, node, Some(ev))?;
    }
    if let Some(ev) = flags.programmatic_event {
        let node = usize::from(!writes.is_empty());
        sim.graph_kernel_node_set_programmatic_event(graph, node, Some(ev))?;
    }
    Ok(())
}

pub(crate) use crate::planner::DECODE_ACTIVATION_BYTES;

/// Place each expert per `map` (home H2D, replica D2D). HBM is the only cap.
pub fn sim_placed(
    trace: &Trace,
    profile: HardwareProfile,
    bytes_per_expert: u64,
    map: &PlaceMap,
) -> Result<SimReplay, Error> {
    let n_gpus = u16::try_from(profile.n_gpus()).unwrap_or(1).max(1);
    let mut sim = Sim::new(profile);
    let s = StreamId(0);
    let mut handles: BTreeMap<ExpertKey, AllocId> = BTreeMap::new();
    let bytes = bytes_per_expert.max(1);
    let mut hits = 0u64;
    let mut misses = 0u64;
    let mut token_ends: Vec<u64> = Vec::new();
    for (i, event) in trace.events.iter().enumerate() {
        for key in event.keys() {
            let d = map.home_of(key, n_gpus);
            if let Some(id) = handles.get(&key).copied() {
                hits = hits.saturating_add(1);
                kernel(&mut sim, d, s, id)?;
            } else {
                misses = misses.saturating_add(1);
                let id = sim.alloc(d, bytes, s)?;
                let _c = sim.memcpy_pinned_to_device(d, id, bytes, s)?;
                if let Some(reps) = map.replicas.get(&key) {
                    for dst in reps {
                        let _c = sim.memcpy_device_to_device(d, *dst, id, bytes, s)?;
                    }
                }
                kernel(&mut sim, d, s, id)?;
                let _prev = handles.insert(key, id);
            }
        }
        if last_of_token(&trace.events, i) {
            sim.synchronize()?;
            token_ends.push(sim.clock_ns());
        }
    }
    if token_ends.is_empty() {
        sim.synchronize()?;
    }
    Ok(finish(
        &sim,
        &token_ends,
        ReplayCounters {
            hits,
            misses,
            ..ReplayCounters::default()
        },
    ))
}

/// Compute on GPU0; experts live on `map` homes. Miss: pinned H2D to home, then
/// [`plan_placement`] chooses D2D of weights onto GPU0 vs shipping activations
/// to home (GEMM on home, small result D2D back). Online reuse is how many
/// times this key has been seen so far (no future leak).
///
/// Homes that are already GPU0 skip the peer hop. Hits GEMM where the first
/// fetch left the weights.
pub fn sim_remote_home(
    trace: &Trace,
    profile: HardwareProfile,
    bytes_per_expert: u64,
    map: &PlaceMap,
) -> Result<SimReplay, Error> {
    sim_remote_home_cfg(
        trace,
        profile,
        bytes_per_expert,
        DECODE_ACTIVATION_BYTES,
        map,
    )
}

/// [`sim_remote_home`] with an explicit activation payload size.
pub fn sim_remote_home_cfg(
    trace: &Trace,
    profile: HardwareProfile,
    bytes_per_expert: u64,
    activation_bytes: u64,
    map: &PlaceMap,
) -> Result<SimReplay, Error> {
    let n_gpus = u16::try_from(profile.n_gpus()).unwrap_or(1).max(1);
    let mut sim = Sim::new(profile);
    let compute = DeviceId(0);
    let s = StreamId(0);
    let mut pages: BTreeMap<ExpertKey, RemotePage> = BTreeMap::new();
    let mut seen: BTreeMap<ExpertKey, u64> = BTreeMap::new();
    let bytes = bytes_per_expert.max(1);
    let act = activation_bytes.max(1);
    let mut hits = 0u64;
    let mut misses = 0u64;
    let mut next_event = 1u32;
    let mut token_ends: Vec<u64> = Vec::new();
    for (i, event) in trace.events.iter().enumerate() {
        let fan_in = u64::try_from(event.experts.len()).unwrap_or(1).max(1);
        for key in event.keys() {
            let n = seen.entry(key).or_insert(0);
            *n = n.saturating_add(1);
            let reuse = *n;
            if let Some(page) = pages.get(&key).copied() {
                hits = hits.saturating_add(1);
                let page = remote_hit(&mut sim, page, compute, act, s, &mut next_event, false)?;
                let _prev = pages.insert(key, page);
            } else {
                misses = misses.saturating_add(1);
                let home = map.home_of(key, n_gpus);
                let page = fetch_remote(
                    &mut sim,
                    RemoteFetch {
                        home,
                        compute,
                        expert_bytes: bytes,
                        act_bytes: act,
                        stream: s,
                        sync_alloc: false,
                        managed: false,
                        accessed_by: false,
                        stream_attach: false,
                        managed_host: false,
                    },
                    reuse,
                    fan_in,
                    &mut next_event,
                )?;
                let _prev = pages.insert(key, page);
            }
        }
        if last_of_token(&trace.events, i) {
            sim.synchronize()?;
            token_ends.push(sim.clock_ns());
        }
    }
    if token_ends.is_empty() {
        sim.synchronize()?;
    }
    Ok(finish(
        &sim,
        &token_ends,
        ReplayCounters {
            hits,
            misses,
            ..ReplayCounters::default()
        },
    ))
}

#[derive(Clone, Copy)]
pub(crate) struct RemotePage {
    pub(crate) id: AllocId,
    pub(crate) gemm: DeviceId,
    pub(crate) home: DeviceId,
    pub(crate) act: Option<AllocId>,
}

pub(crate) struct RemoteFetch {
    pub(crate) home: DeviceId,
    pub(crate) compute: DeviceId,
    pub(crate) expert_bytes: u64,
    pub(crate) act_bytes: u64,
    pub(crate) stream: StreamId,
    pub(crate) sync_alloc: bool,
    /// `cudaMallocManaged` + PreferredLocation on home; compute GEMM reads remotely.
    pub(crate) managed: bool,
    /// [`SimCfg::accessed_by`]: map compute without a dest migrate (managed, VMM, or mempool).
    pub(crate) accessed_by: bool,
    /// [`SimCfg::stream_attach`]: Single-attach then prefetch on `stream`.
    pub(crate) stream_attach: bool,
    /// [`SimCfg::managed_host`]: Host alloc then Global attach before prefetch.
    pub(crate) managed_host: bool,
}

pub(crate) fn remote_hit(
    sim: &mut Sim,
    page: RemotePage,
    compute: DeviceId,
    act_bytes: u64,
    stream: StreamId,
    next_event: &mut u32,
    sync: bool,
) -> Result<RemotePage, Error> {
    if page.home != compute && page.gemm == page.home {
        let act = match page.act {
            Some(a) => a,
            None => hbm_alloc(sim, compute, act_bytes, stream, sync)?,
        };
        ship_act(sim, compute, page.home, act, act_bytes, stream, next_event)?;
        kernel(sim, page.home, stream, page.id)?;
        ship_act(sim, page.home, compute, act, act_bytes, stream, next_event)?;
        return Ok(RemotePage {
            act: Some(act),
            ..page
        });
    }
    kernel(sim, page.gemm, stream, page.id)?;
    Ok(page)
}

/// Pin weights on home (and D2D them onto compute when
/// [`Placement::MoveWeights`], unless [`RemoteFetch::accessed_by`] already
/// maps compute). [`RemoteFetch::managed`] prefetches a
/// PreferredLocation page and leaves GEMM on compute as a remote read.
/// No GEMM — that is [`remote_hit`] / demand.
pub(crate) fn fill_remote(
    sim: &mut Sim,
    fetch: RemoteFetch,
    reuse: u64,
    fan_in: u64,
    next_event: &mut u32,
) -> Result<RemotePage, Error> {
    if fetch.managed {
        return fill_remote_managed(sim, fetch, next_event);
    }
    let id = hbm_alloc(
        sim,
        fetch.home,
        fetch.expert_bytes,
        fetch.stream,
        fetch.sync_alloc,
    )?;
    hbm_h2d_pinned(
        sim,
        fetch.home,
        id,
        fetch.expert_bytes,
        fetch.stream,
        fetch.sync_alloc,
    )?;
    if fetch.home == fetch.compute {
        return Ok(RemotePage {
            id,
            gemm: fetch.compute,
            home: fetch.home,
            act: None,
        });
    }
    let bps = sim
        .profile()
        .link(Some(fetch.home), Some(fetch.compute))?
        .bps;
    match plan_placement(fetch.expert_bytes, fetch.act_bytes, fan_in, reuse, bps) {
        Placement::MoveWeights => {
            if fetch.accessed_by && !fetch.sync_alloc {
                wait_peer(sim, fetch.home, fetch.compute, fetch.stream, next_event)?;
                return Ok(RemotePage {
                    id,
                    gemm: fetch.compute,
                    home: fetch.home,
                    act: None,
                });
            }
            let _c = sim.memcpy_device_to_device(
                fetch.home,
                fetch.compute,
                id,
                fetch.expert_bytes,
                fetch.stream,
            )?;
            wait_peer(sim, fetch.home, fetch.compute, fetch.stream, next_event)?;
            Ok(RemotePage {
                id,
                gemm: fetch.compute,
                home: fetch.home,
                act: None,
            })
        }
        Placement::DispatchActivations => Ok(RemotePage {
            id,
            gemm: fetch.home,
            home: fetch.home,
            act: None,
        }),
    }
}

fn fill_remote_managed(
    sim: &mut Sim,
    fetch: RemoteFetch,
    next_event: &mut u32,
) -> Result<RemotePage, Error> {
    let id = if fetch.managed_host {
        sim.alloc_managed_host(fetch.expert_bytes)?
    } else {
        sim.alloc_managed(fetch.expert_bytes)?
    };
    sim.mem_advise(id, gpu_sim::MemAdvise::SetReadMostly, fetch.home)?;
    sim.mem_advise(id, gpu_sim::MemAdvise::SetPreferredLocation, fetch.home)?;
    if fetch.accessed_by {
        advise_accessed_by(sim, id)?;
    }
    let stream = bump_null_for_attach(fetch.stream, fetch.stream_attach);
    attach_managed_visibility(
        sim,
        fetch.home,
        id,
        stream,
        fetch.stream_attach,
        fetch.managed_host,
    )?;
    let _p = sim.prefetch(fetch.home, id, stream)?;
    if fetch.home != fetch.compute {
        wait_peer(sim, fetch.home, fetch.compute, stream, next_event)?;
    }
    Ok(RemotePage {
        id,
        gemm: fetch.compute,
        home: fetch.home,
        act: None,
    })
}

pub(crate) fn fetch_remote(
    sim: &mut Sim,
    fetch: RemoteFetch,
    reuse: u64,
    fan_in: u64,
    next_event: &mut u32,
) -> Result<RemotePage, Error> {
    let compute = fetch.compute;
    let act = fetch.act_bytes;
    let stream = fetch.stream;
    let sync = fetch.sync_alloc;
    let page = fill_remote(sim, fetch, reuse, fan_in, next_event)?;
    remote_hit(sim, page, compute, act, stream, next_event, sync)
}

/// Free a remote expert page (weights on home/compute, optional act).
pub(crate) fn drop_remote(
    sim: &mut Sim,
    page: RemotePage,
    compute: DeviceId,
    stream: StreamId,
    sync: bool,
) -> Result<(), Error> {
    if page_is_managed(sim, page.id) {
        if page.gemm != page.home {
            sim.synchronize_stream(page.gemm, stream)?;
        }
        sim.free_sync(page.id)?;
        if let Some(act) = page.act {
            if sync {
                sim.free_sync(act)?;
            } else {
                if page.home != compute {
                    sim.free(page.home, act, stream)?;
                }
                sim.free(compute, act, stream)?;
            }
        }
        return Ok(());
    }
    if sync {
        sim.free_sync(page.id)?;
        if let Some(act) = page.act {
            sim.free_sync(act)?;
        }
        return Ok(());
    }
    if page.gemm != page.home {
        sim.free(page.gemm, page.id, stream)?;
    }
    sim.free(page.home, page.id, stream)?;
    if let Some(act) = page.act {
        if page.home != compute {
            sim.free(page.home, act, stream)?;
        }
        sim.free(compute, act, stream)?;
    }
    Ok(())
}

fn ship_act(
    sim: &mut Sim,
    src: DeviceId,
    dst: DeviceId,
    act: AllocId,
    bytes: u64,
    stream: StreamId,
    next_event: &mut u32,
) -> Result<(), Error> {
    if src == dst {
        return Ok(());
    }
    let _c = sim.memcpy_device_to_device(src, dst, act, bytes, stream)?;
    wait_peer(sim, src, dst, stream, next_event)
}

fn wait_peer(
    sim: &mut Sim,
    src: DeviceId,
    dst: DeviceId,
    stream: StreamId,
    next_event: &mut u32,
) -> Result<(), Error> {
    let ev = EventId(*next_event);
    *next_event = next_event.saturating_add(1);
    sim.create_event_disable_timing(ev)?;
    let _r = sim.record_event(src, ev, stream)?;
    let _w = sim.wait_event(dst, ev, stream)?;
    Ok(())
}

/// Cached LRU on GPU0 versus static EP across the profile's GPUs.
#[derive(Clone, Debug)]
pub struct EpCompare {
    /// [`sim_replay`] with a bounded GPU0 cache (evicts).
    pub cached: SimReplay,
    /// Static placement. `Err` when a home GPU OOMs (illegal under that HBM).
    pub static_ep: Result<SimReplay, Error>,
}

impl EpCompare {
    /// One line for CLI / benches.
    #[must_use]
    pub fn line(&self) -> String {
        match &self.static_ep {
            Ok(s) => format!("cached {} | static {}", self.cached.line(), s.line()),
            Err(e) => format!("cached {} | static err={e}", self.cached.line()),
        }
    }
}

/// Run LRU-on-GPU0 and static EP on the same trace and profile.
pub fn compare_ep(
    trace: &Trace,
    profile: HardwareProfile,
    slots: usize,
    bytes_per_expert: u64,
    lookahead: usize,
) -> Result<EpCompare, Error> {
    let cached = sim_replay(
        trace,
        profile.clone(),
        slots,
        Policy::Lru,
        bytes_per_expert,
        lookahead,
    )?;
    let static_ep = sim_static_ep(trace, profile, bytes_per_expert);
    Ok(EpCompare { cached, static_ep })
}

/// Place each expert on `home_gpu` and leave it there. HBM is the only cap.
pub fn sim_static_ep(
    trace: &Trace,
    profile: HardwareProfile,
    bytes_per_expert: u64,
) -> Result<SimReplay, Error> {
    let n = u16::try_from(profile.n_gpus()).unwrap_or(1).max(1);
    sim_placed(
        trace,
        profile,
        bytes_per_expert,
        &crate::place::striped(trace, n),
    )
}
