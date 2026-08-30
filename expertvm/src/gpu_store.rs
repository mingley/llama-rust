//! [`ExpertStore`] backed by [`gpu_sim`]: H2D, mapped host, managed, or VMM on miss.

use crate::access::{ExpertKey, Trace};
use crate::error::Error;
use crate::kv::KvSimOp;
use crate::place::home_gpu;
use crate::planner::{
    plan_placement, plan_window, predicted_keys, window_keys, ChainState, Markov, Placement, Plan,
    Prefetch, DECODE_ACTIVATION_BYTES,
};
use crate::sim_replay::{
    bind_shareable_mempools, check_cluster_preferred, replay_streams, retarget_parked_kernel,
    stream_of, GemmFlags, LeafMem, GRAPH_SCRATCH_BYTES,
};
use crate::store::{CachedStore, DirectStore, ExpertParts, ExpertPhase, ExpertStore, StoreMetrics};
use gpu_sim::{
    AllocId, DType, DeviceId, EventId, GraphId, HardwareProfile, KernelBuf, KernelKind, MemAdvise,
    MemHandleId, MemcpyOp, Place, PoolId, Score, Sim, StreamId,
};
use std::collections::{BTreeMap, BTreeSet};

/// How [`SimulatedGpuStore`] places a miss page.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GpuFill {
    /// `cudaMallocAsync` + pinned H2D (decode identity).
    #[default]
    Pinned,
    /// `cudaMallocManaged` + ReadMostly + PreferredLocation + prefetch.
    Managed,
    /// `cudaHostAllocMapped` (PCIe kernel, no H2D).
    Mapped,
    /// `va_acquire` + pinned H2D; evict [`gpu_sim::Sim::va_release`].
    Vmm,
}

impl GpuFill {
    /// Pinned when every flag is off; otherwise exactly one of mapped/managed/vmm.
    pub fn from_flags(mapped: bool, managed: bool, vmm: bool) -> Result<Self, Error> {
        match (mapped, managed, vmm) {
            (false, false, false) => Ok(Self::Pinned),
            (true, false, false) => Ok(Self::Mapped),
            (false, true, false) => Ok(Self::Managed),
            (false, false, true) => Ok(Self::Vmm),
            _ => Err(Error::Store("choose one of mapped, managed, vmm")),
        }
    }
}

/// CUDA knobs [`SimulatedGpuStore::new`] leaves off so decode identity stays async.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuStoreCfg {
    /// [`gpu_sim::Sim::host_func`] after each acquire GEMM (CPU scheduler roundtrip).
    pub host_func: bool,
    /// Blocking `cudaStreamCreate` for created streams (`1 .. n`).
    ///
    /// Default off (decode identity): copy is NULL, compute is `StreamId(1)`, so
    /// `n = 2`. [`Self::seq_streams`] marks copy streams `1 .. n_copy-1` and
    /// compute `StreamId(n_copy)` (`n = n_copy + 1`).
    pub blocking_streams: bool,
    /// Host-sync `cudaMalloc` / `cudaMemcpy` / `cudaFree` on pinned/VMM misses.
    pub sync_alloc: bool,
    /// Hold unused `cudaMallocAsync` bytes in the default pool (`u64::MAX` threshold).
    pub mempool: bool,
    /// POSIX-FD shareable mempool IPC (`cudaMemPoolExportToShareableHandle`).
    ///
    /// Creates a shareable pool, exports it, imports a sibling that shares
    /// live/cached, and `cudaDeviceSetMemPool`s so miss pages draw from it.
    /// Implies [`Self::mempool`]. Illegal with [`Self::sync_alloc`] or
    /// mapped/managed/vmm fills. Decode identity stays the device default pool.
    pub shareable: bool,
    /// Physical span for [`GpuFill::Vmm`]. `0` maps the whole expert (`va_acquire`).
    pub vmm_page: u64,
    /// Pageable `cudaMemcpyAsync` (`memcpy_host_to_device`) instead of pinned DMA.
    ///
    /// Host-synchronous and slower (`pageable_permille`). Decode identity stays
    /// on pinned H2D.
    pub pageable: bool,
    /// Peer map without dest HBM: managed [`gpu_sim::MemAdvise::SetAccessedBy`],
    /// VMM [`gpu_sim::Sim::va_set_access`], or pinned-async
    /// [`gpu_sim::Sim::pool_set_access`] on every GPU.
    ///
    /// Expert GEMMs are reads-only, so a dest acquire does not migrate or
    /// charge dest HBM. Migrate retargets compute (no dest prefetch / VMM
    /// map+D2D / pinned D2D, no `drop_managed_copy` / `va_unmap` of home). Pin
    /// skips the replica prefetch, VMM dest map, or pinned D2D. No-op for
    /// [`GpuFill::Mapped`] or host-sync [`Self::sync_alloc`] (`cudaMalloc` is
    /// not a mempool). Decode identity stays off.
    pub accessed_by: bool,
    /// CUDA legacy null stream: copy (`StreamId(0)`) serializes with compute.
    ///
    /// Off by default (`cudaStreamNonBlocking` compute). Decode identity stays
    /// overlapping.
    pub legacy_null: bool,
    /// `cudaStreamCreateWithPriority` on created streams (priority = stream id).
    ///
    /// Default off: copy stays NULL at priority 0, compute is `StreamId(1)`.
    /// [`Self::seq_streams`] also marks the extra copy streams and compute
    /// `StreamId(n_copy)`. Decode identity stays default priority.
    pub stream_priority: bool,
    /// `cudaGraphExecUpdate` a parked exec onto the next miss alloc.
    ///
    /// Evict parks the instantiated GEMM graph instead of `destroy_graph`.
    /// The next capture on that GPU updates it (`graph_update_ns`) instead of
    /// instantiate. Decode identity stays destroy+instantiate.
    pub graph_update: bool,
    /// `cudaGraphExecKernelNodeSetParams` a parked exec onto the next miss.
    ///
    /// Evict parks the instantiated GEMM graph. The next miss retargets the
    /// unique kernel node (`graph_set_params_ns`) without recapture, and a
    /// unique memcpy or memset if the leaf has one. Walker combo parents
    /// additionally use `cudaGraphExecChildGraphNodeSetParams`. Works with
    /// [`Self::graph_mem`] / [`Self::graph_auto_free`]. Illegal with
    /// [`Self::graph_update`]. Decode identity stays destroy+instantiate.
    pub graph_set_params: bool,
    /// `cudaGraphClone` the capture before instantiate (graph vs exec).
    ///
    /// Clone, destroy the src, then instantiate the copy. A parked-exec
    /// update skips clone. Decode identity stays instantiate-in-place.
    pub graph_clone: bool,
    /// `cudaGraphCreate` / `cudaGraphAddKernelNode` instead of stream capture.
    ///
    /// Does not require an idle compute stream (`cudaStreamBeginCapture` does).
    /// Combo parents add children without [`gpu_sim::Sim::graph_add_dependencies`]
    /// so independent expert GEMMs may Hyper-Q overlap. Decode identity stays
    /// `begin_capture` / `end_capture`. Illegal with [`Self::graph_piecewise`].
    pub graph_build: bool,
    /// `cudaStreamBeginCaptureToGraph` combo parents (independent child roots).
    ///
    /// Walker combo parents capture each instantiated leaf into one parent as
    /// an extra root. Sibling GEMMs may Hyper-Q overlap. Store leaves capture
    /// into `create_graph`. Illegal with [`Self::graph_build`]. Decode identity
    /// stays `begin_capture` / `end_capture`.
    pub graph_piecewise: bool,
    /// Leaf GEMM graphs include a scratch `cudaMallocAsync` + free.
    ///
    /// CUDA cannot `cudaGraphExecUpdate` mem nodes, so [`Self::graph_update`]
    /// is skipped. [`Self::graph_set_params`] still retargets the kernel.
    /// Decode identity stays kernel-only graphs.
    pub graph_mem: bool,
    /// Scratch alloc without a matching free; AutoFreeOnLaunch instantiate.
    ///
    /// Illegal with [`Self::graph_mem`]. `--graph-update` is skipped;
    /// [`Self::graph_set_params`] still retargets. Decode
    /// identity stays kernel-only graphs.
    pub graph_auto_free: bool,
    /// `cudaLaunchCooperativeKernel` for grouped GEMMs.
    ///
    /// Occupies every Hyper-Q slot so leftover prefill cannot overlap decode
    /// even when [`Self::compute_slots`] is `>=2`. Decode identity stays
    /// `cudaLaunchKernel`.
    pub cooperative: bool,
    /// Same-stream programmatic dependent launch for grouped GEMMs.
    ///
    /// Consecutive expert kernels on one compute stream may overlap after the
    /// previous kernel's PDL trigger when [`Self::compute_slots`] is `>=2`.
    /// Illegal with [`Self::cooperative`]. Decode identity stays
    /// `cudaLaunchKernel`.
    pub pdl: bool,
    /// `cudaLaunchAttributeAccessPolicyWindow` over each expert page.
    ///
    /// [`SimulatedGpuStore::with_cfg`] calls [`gpu_sim::Sim::enable_persisting_l2`]
    /// and launches GEMMs with a persisting window. Decode identity stays
    /// persist limit 0.
    pub l2_persist: bool,
    /// Hopper cluster X size (`cudaLaunchAttributeClusterDimension`). `0` is off.
    ///
    /// Occupies `min(N, compute_slots)` Hyper-Q slots. Decode identity stays
    /// `cudaLaunchKernel` (no cluster).
    pub cluster: u8,
    /// Hopper preferred cluster X (`cudaLaunchAttributePreferredClusterDimension`).
    /// `0` is off.
    ///
    /// Occupancy uses this size when it fits in [`Self::compute_slots`], else
    /// [`Self::cluster`]. Requires [`Self::cluster`]. Must be an integer
    /// multiple of the required size. Decode identity stays no preferred dim.
    pub preferred_cluster: u8,
    /// Spread cluster scheduling (`cudaLaunchAttributeClusterSchedulingPolicyPreference`).
    ///
    /// Occupies every Hyper-Q slot so leftover kernels cannot overlap even
    /// when [`Self::cluster`] is smaller than [`Self::compute_slots`]. A no-op
    /// unless cluster blocks `> 1`. Decode identity stays Default.
    pub cluster_spread: bool,
    /// Max-shared carveout (`cudaLaunchAttributePreferredSharedMemoryCarveout`).
    ///
    /// Occupies every Hyper-Q slot so leftover kernels cannot overlap.
    /// Decode identity stays Default.
    pub max_shared: bool,
    /// Hopper NVLS replica fanout (`cuMulticastCreate` / bind / kernel store).
    ///
    /// [`Self::pin_hot`] and walker `--place replicas` map dest VMM physicals
    /// then one NVLS kernel instead of sequential D2D. Occupies compute, not a
    /// copy engine. Requires [`GpuFill::Vmm`]. Illegal with [`Self::accessed_by`]
    /// or [`Self::vmm_page`]. Needs NVLink. Decode identity stays D2D.
    pub multicast: bool,
    /// Timing-on copy events (`cudaEventCreate`) and [`gpu_sim::Sim::event_elapsed_ns`].
    ///
    /// Default is `cudaEventDisableTiming` (vLLM wait events). Decode identity
    /// stays disable-timing; elapsed is not recorded.
    pub timing_events: bool,
    /// Per-sequence copy streams so concurrent H2D can overlap.
    ///
    /// `n_copy` is GPU0 `copy_engines.max(2)`; copy for sequence `s` is
    /// `s % n_copy`. Grouped GEMM stays on `StreamId(n_copy)` (not a
    /// per-sequence compute stream). Default off: copy is NULL (`StreamId(0)`),
    /// compute is `StreamId(1)`. Decode identity stays serial copies.
    pub seq_streams: bool,
    /// Map Engine interned KV blocks onto this Sim (`cuMemCreate` + `cuMemMap`).
    ///
    /// Default off (decode identity): scores bill expert H2D/GEMM only.
    /// `--kv-sim` reserves a VA of `pool_blocks` pages so TTFT/ITL include
    /// KV traffic on the same clock. Interned blocks share one physical
    /// handle. Distinct from `expertvm kv`.
    pub kv_sim: bool,
    /// Second compute stream at higher CUDA priority for decode GEMMs.
    ///
    /// Prefill stays on the existing compute stream. [`Self::stream_priority`]
    /// must also be on so decode actually preempts leftover prefill (priority
    /// equals stream id). Engine token-boundary ITL then
    /// [`SimulatedGpuStore::token_clock_ns`] (decode stream only) so leftover
    /// prefill does not inflate ITL. Default off: one compute stream (decode
    /// identity); ITL still [`SimulatedGpuStore::clock_ns`] (full drain).
    pub decode_priority: bool,
    /// Hyper-Q compute occupancy override (`0` keeps the profile).
    ///
    /// `1` is exclusive compute. `>=2` lets leftover prefill and decode GEMMs
    /// overlap at full issue rate when they sit on different streams
    /// ([`Self::decode_priority`]). Default `0` (profile `compute_slots`,
    /// example H100 is `1`) keeps decode identity.
    pub compute_slots: u8,
    /// Decode-stream green-context SM fraction (‰). `0` keeps a full chip.
    ///
    /// `1..=1000` calls [`gpu_sim::Sim::set_stream_sm_permille`] on the decode
    /// stream; leftover prefill gets the remainder when
    /// [`Self::decode_priority`] is on. Default `0` keeps decode identity.
    pub decode_sm_permille: u16,
}

#[derive(Clone, Copy)]
struct GpuPage {
    id: AllocId,
    device: DeviceId,
    /// Copy-stream event the compute stream must wait on, if not yet consumed.
    ready: Option<EventId>,
    /// Timing-on record before the copy, when [`GpuStoreCfg::timing_events`].
    start: Option<EventId>,
}

/// Bounded cache whose misses pay pinned H2D, mapped host, managed prefetch, or VMM.
pub struct SimulatedGpuStore {
    cache: CachedStore,
    sim: Sim,
    device: DeviceId,
    copy: StreamId,
    compute: StreamId,
    prefill: StreamId,
    decode: StreamId,
    decode_priority: bool,
    cooperative: bool,
    pdl: bool,
    l2_persist: bool,
    cluster: u8,
    preferred_cluster: u8,
    cluster_spread: bool,
    max_shared: bool,
    multicast: bool,
    next_event: u32,
    pages: BTreeMap<ExpertKey, GpuPage>,
    /// Peer copy dest per pinned key (`(home + 1) % n_gpus`).
    replicas: BTreeMap<ExpertKey, DeviceId>,
    evicting: BTreeMap<ExpertKey, GpuPage>,
    bytes_per_expert: u64,
    staging: AllocId,
    graphs: BTreeMap<AllocId, GraphId>,
    /// Instantiated GEMM execs parked on evict, keyed by the capture GPU.
    idle_execs: BTreeMap<DeviceId, Vec<GraphId>>,
    graph_launches: u64,
    graph_updates: u64,
    graph_clones: u64,
    graph_set_params_n: u64,
    graph_update: bool,
    graph_set_params: bool,
    graph_clone: bool,
    graph_build: bool,
    graph_piecewise: bool,
    leaf: LeafMem,
    timing_events: bool,
    copy_elapsed_ns: u64,
    mode: GpuFill,
    host_func: bool,
    sync_alloc: bool,
    /// [`GpuStoreCfg::vmm_page`]: KV-sized physicals when [`GpuFill::Vmm`].
    vmm_page: u64,
    /// Pageable H2D (`memcpy_host_to_device`) instead of pinned DMA.
    pageable: bool,
    /// [`GpuStoreCfg::accessed_by`]: managed/VMM/mempool pages stay on the home GPU.
    accessed_by: bool,
    /// Successful D2D [`Self::migrate`] calls (source device ≠ dest).
    migrates: u64,
    /// [`plan_placement`] chose [`Placement::DispatchActivations`] (no D2D).
    dispatches: u64,
    /// Successful peer replica copies from [`Self::pin_hot`].
    replicates: u64,
    /// [`GpuStoreCfg::seq_streams`]: `bind_sequence` retargets [`Self::copy`].
    seq_streams: bool,
    /// [`GpuStoreCfg::kv_sim`]: Engine interned KV on this Sim.
    kv_sim: bool,
    kv: Option<KvGpu>,
    /// Imported sibling of GPU0's shareable device mempool.
    share_import: Option<PoolId>,
}

struct KvGpu {
    va: AllocId,
    page_bytes: u64,
    n_pages: u32,
    mapped: BTreeSet<u32>,
    handles: BTreeMap<u32, MemHandleId>,
    hits: u64,
    misses: u64,
}

impl SimulatedGpuStore {
    /// `slots` HBM expert pages of `bytes_per_expert` each (pinned H2D).
    pub fn new(
        inner: DirectStore,
        slots: usize,
        profile: HardwareProfile,
        bytes_per_expert: u64,
    ) -> Result<Self, Error> {
        Self::with_cfg(
            inner,
            slots,
            profile,
            bytes_per_expert,
            GpuFill::Pinned,
            GpuStoreCfg::default(),
        )
    }

    /// Same as [`Self::new`], but miss pages are `cudaMallocManaged`.
    ///
    /// ReadMostly + PreferredLocation on the striped home, then prefetch
    /// (no pinned H2D). Decode identity stays on [`Self::new`].
    pub fn with_managed(
        inner: DirectStore,
        slots: usize,
        profile: HardwareProfile,
        bytes_per_expert: u64,
    ) -> Result<Self, Error> {
        Self::with_cfg(
            inner,
            slots,
            profile,
            bytes_per_expert,
            GpuFill::Managed,
            GpuStoreCfg::default(),
        )
    }

    /// Same as [`Self::new`], but miss pages are `cudaHostAllocMapped`.
    ///
    /// Kernels read over PCIe with no H2D and no HBM. Decode identity stays
    /// on [`Self::new`].
    pub fn with_mapped(
        inner: DirectStore,
        slots: usize,
        profile: HardwareProfile,
        bytes_per_expert: u64,
    ) -> Result<Self, Error> {
        Self::with_cfg(
            inner,
            slots,
            profile,
            bytes_per_expert,
            GpuFill::Mapped,
            GpuStoreCfg::default(),
        )
    }

    /// Same as [`Self::new`], but miss pages are `va_acquire` then pinned H2D.
    ///
    /// Evict [`gpu_sim::Sim::va_release`]s so the VA can be remapped. Decode
    /// identity stays on [`Self::new`].
    pub fn with_vmm(
        inner: DirectStore,
        slots: usize,
        profile: HardwareProfile,
        bytes_per_expert: u64,
    ) -> Result<Self, Error> {
        Self::with_cfg(
            inner,
            slots,
            profile,
            bytes_per_expert,
            GpuFill::Vmm,
            GpuStoreCfg::default(),
        )
    }

    /// Construct with an explicit fill and [`GpuStoreCfg`] (`SimCfg` subset).
    ///
    /// Defaults keep decode identity: async alloc, overlapping copy/compute,
    /// no `host_func`, default pool threshold 0, no AccessedBy, per-thread NULL,
    /// destroy+instantiate graphs (stream capture; [`GpuStoreCfg::graph_build`]
    /// is `cudaGraphCreate` / `cudaGraphAdd*`; [`GpuStoreCfg::graph_mem`]
    /// records in-graph scratch with a matching free;
    /// [`GpuStoreCfg::graph_auto_free`] is AutoFreeOnLaunch without a free;
    /// [`GpuStoreCfg::graph_set_params`] retargets a parked kernel node),
    /// disable-timing copy events.
    /// [`GpuStoreCfg::cooperative`] launches GEMMs with
    /// `cudaLaunchCooperativeKernel` (exclusive compute).
    /// [`GpuStoreCfg::pdl`] is same-stream programmatic dependent launch
    /// (illegal with cooperative).
    /// [`GpuStoreCfg::l2_persist`] is `cudaLaunchAttributeAccessPolicyWindow`
    /// over expert pages (persisting L2 after the first fill).
    /// [`GpuStoreCfg::cluster`] / [`GpuStoreCfg::preferred_cluster`] are Hopper
    /// thread-block cluster dims.
    /// [`GpuStoreCfg::multicast`] is Hopper NVLS replica fanout (requires
    /// [`GpuFill::Vmm`] and NVLink).
    /// [`GpuStoreCfg::compute_slots`] `0` keeps the profile (example H100 is
    /// exclusive compute). [`GpuStoreCfg::decode_sm_permille`] `0` keeps a
    /// full chip. `1..=1000` without [`GpuStoreCfg::decode_priority`] caps the
    /// single compute stream.
    pub fn with_cfg(
        inner: DirectStore,
        slots: usize,
        profile: HardwareProfile,
        bytes_per_expert: u64,
        fill: GpuFill,
        cfg: GpuStoreCfg,
    ) -> Result<Self, Error> {
        if cfg.graph_update && cfg.graph_set_params {
            return Err(Error::Store("choose one of graph-update, graph-set-params"));
        }
        if cfg.graph_build && cfg.graph_piecewise {
            return Err(Error::Store("choose one of graph-build, graph-piecewise"));
        }
        if cfg.pdl && cfg.cooperative {
            return Err(Error::Store("choose one of pdl, cooperative"));
        }
        check_cluster_preferred(cfg.cluster, cfg.preferred_cluster)?;
        if cfg.shareable && (cfg.sync_alloc || fill != GpuFill::Pinned) {
            return Err(Error::Store("shareable needs cudaMallocAsync"));
        }
        if cfg.multicast {
            if fill != GpuFill::Vmm {
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
        }
        let leaf = LeafMem::from_flags(cfg.graph_mem, cfg.graph_auto_free)?;
        let bytes = bytes_per_expert.max(1);
        let (copy, prefill, decode, mark) =
            copy_compute_streams(&profile, cfg.seq_streams, cfg.decode_priority);
        let profile = if cfg.compute_slots > 0 {
            profile.with_compute_slots(cfg.compute_slots)
        } else {
            profile
        };
        let mut sim = Sim::new(profile);
        if cfg.l2_persist {
            sim.enable_persisting_l2()?;
        }
        let share_import = if cfg.shareable {
            bind_shareable_mempools(&mut sim)?
                .get(&DeviceId(0))
                .copied()
        } else {
            None
        };
        if cfg.mempool || cfg.shareable {
            sim.set_default_pool_release_threshold(u64::MAX)?;
        }
        if cfg.accessed_by && fill == GpuFill::Pinned && !cfg.sync_alloc {
            advise_pool_access(&mut sim)?;
        }
        if cfg.blocking_streams {
            sim.set_created_streams_blocking(mark)?;
        }
        if cfg.legacy_null {
            sim.set_legacy_null_stream(true);
        }
        if cfg.stream_priority {
            sim.set_created_streams_priority(mark)?;
        }
        if cfg.decode_sm_permille > 0 {
            let dec = cfg.decode_sm_permille.min(1000);
            let pre = 1000u16.saturating_sub(dec).max(1);
            let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
            for g in 0..n {
                let d = DeviceId(g);
                sim.set_stream_sm_permille(d, decode, dec)?;
                if prefill != decode {
                    sim.set_stream_sm_permille(d, prefill, pre)?;
                }
            }
        }
        let cache_slots = mapped_occupancy(slots, fill, sim.pin_budget(), bytes);
        // Mapped expert pages already charge the pin budget. Pageable H2D
        // does not need a second mlock either.
        let staging = if fill == GpuFill::Mapped || cfg.pageable {
            sim.alloc_host(bytes)?
        } else {
            sim.alloc_host_pinned(bytes)?
        };
        Ok(Self {
            cache: CachedStore::new(inner, cache_slots)?,
            sim,
            device: DeviceId(0),
            copy,
            compute: prefill,
            prefill,
            decode,
            decode_priority: cfg.decode_priority,
            cooperative: cfg.cooperative,
            pdl: cfg.pdl,
            l2_persist: cfg.l2_persist,
            cluster: cfg.cluster,
            preferred_cluster: cfg.preferred_cluster,
            cluster_spread: cfg.cluster_spread,
            max_shared: cfg.max_shared,
            multicast: cfg.multicast,
            next_event: 1,
            pages: BTreeMap::new(),
            replicas: BTreeMap::new(),
            evicting: BTreeMap::new(),
            bytes_per_expert: bytes,
            staging,
            graphs: BTreeMap::new(),
            idle_execs: BTreeMap::new(),
            graph_launches: 0,
            graph_updates: 0,
            graph_clones: 0,
            graph_set_params_n: 0,
            graph_update: cfg.graph_update,
            graph_set_params: cfg.graph_set_params,
            graph_clone: cfg.graph_clone,
            graph_build: cfg.graph_build,
            graph_piecewise: cfg.graph_piecewise,
            leaf,
            timing_events: cfg.timing_events,
            copy_elapsed_ns: 0,
            mode: fill,
            host_func: cfg.host_func,
            sync_alloc: cfg.sync_alloc,
            vmm_page: cfg.vmm_page,
            pageable: cfg.pageable,
            accessed_by: cfg.accessed_by,
            migrates: 0,
            dispatches: 0,
            replicates: 0,
            seq_streams: cfg.seq_streams,
            kv_sim: cfg.kv_sim,
            kv: None,
            share_import,
        })
    }

    /// Whether miss pages use unified memory (`cudaMallocManaged`).
    #[must_use]
    pub fn uses_managed(&self) -> bool {
        matches!(self.mode, GpuFill::Managed)
    }

    /// Whether miss pages use mapped host (`cudaHostAllocMapped`).
    #[must_use]
    pub fn uses_mapped(&self) -> bool {
        matches!(self.mode, GpuFill::Mapped)
    }

    /// Whether miss pages use CUDA VMM (`va_acquire`).
    #[must_use]
    pub fn uses_vmm(&self) -> bool {
        matches!(self.mode, GpuFill::Vmm)
    }

    /// Page-locked staging buffer from construction; does not count toward HBM.
    #[must_use]
    pub fn staging_is_pinned(&self) -> bool {
        self.sim.is_host_pinned(self.staging).unwrap_or(false)
    }

    /// Unused bytes in GPU0's default mempool after a device drain.
    pub fn default_pool_cached(&mut self) -> Result<u64, Error> {
        self.sim.synchronize()?;
        self.sweep_evicts();
        Ok(self.sim.pool_cached(self.sim.default_pool(self.device)?)?)
    }

    /// Imported sibling of the shareable device mempool (`None` unless
    /// [`GpuStoreCfg::shareable`]).
    #[must_use]
    pub fn share_imported_pool(&self) -> Option<PoolId> {
        self.share_import
    }

    /// `cudaMallocFromPoolAsync` from the imported shareable sibling.
    ///
    /// After an evict with mempool hold, this reuses cached bytes (no extra
    /// HBM). Illegal unless [`GpuStoreCfg::shareable`].
    pub fn alloc_from_imported_pool(&mut self, bytes: u64) -> Result<AllocId, Error> {
        let pool = self
            .share_import
            .ok_or(Error::Store("no shareable import"))?;
        let id = self
            .sim
            .alloc_from_pool(self.device, pool, bytes, self.copy)?;
        self.sim.synchronize_stream(self.device, self.copy)?;
        Ok(id)
    }

    /// Bind H2D / alloc / free to the copy stream for `sequence`.
    ///
    /// No-op unless [`GpuStoreCfg::seq_streams`]. Grouped GEMM stays on
    /// [`Self::compute_stream`].
    pub fn bind_sequence(&mut self, sequence: u64) {
        if !self.seq_streams {
            return;
        }
        // Prefill id is `n_copy` even after `bind_decode_compute` retargets GEMM.
        let n = u8::try_from(self.prefill.0).unwrap_or(1);
        self.copy = stream_of(sequence, n);
    }

    /// Compute stream for grouped expert GEMM (not a per-sequence stream).
    #[must_use]
    pub fn compute_stream(&self) -> StreamId {
        self.compute
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
        }
    }

    /// Prefill grouped-GEMM stream (same as [`Self::compute_stream`] unless decode-priority).
    #[must_use]
    pub fn prefill_stream(&self) -> StreamId {
        self.prefill
    }

    /// Decode grouped-GEMM stream (same as prefill unless [`GpuStoreCfg::decode_priority`]).
    #[must_use]
    pub fn decode_stream(&self) -> StreamId {
        self.decode
    }

    /// Retarget grouped GEMM to the decode or prefill compute stream.
    ///
    /// No-op unless [`GpuStoreCfg::decode_priority`]. Engine `forward_batch`
    /// binds decode; prefill/replay binds prefill.
    pub fn bind_decode_compute(&mut self, decode: bool) {
        if !self.decode_priority {
            return;
        }
        self.compute = if decode { self.decode } else { self.prefill };
    }

    /// Copy stream for the currently bound sequence (NULL when seq-streams is off).
    #[must_use]
    pub fn copy_stream(&self) -> StreamId {
        self.copy
    }

    /// Drain the simulator and return its performance vector.
    pub fn score(&mut self) -> Result<gpu_sim::Score, Error> {
        self.sim.synchronize()?;
        self.sweep_evicts();
        Ok(gpu_sim::Score::from_sim(&self.sim))
    }

    /// Drain and return the virtual clock (token-boundary sample).
    pub fn clock_ns(&mut self) -> Result<u64, Error> {
        self.sim.synchronize()?;
        self.sweep_evicts();
        Ok(self.sim.clock_ns())
    }

    /// Clock after the decode compute stream is idle (leftover prefill may run).
    ///
    /// When [`GpuStoreCfg::decode_priority`] is off this is [`Self::clock_ns`].
    /// Engine ITL uses this so a mixed leftover-prefill step does not wait the
    /// prefill stream. `score()` / waiter arrival still drain the whole node.
    pub fn token_clock_ns(&mut self) -> Result<u64, Error> {
        if !self.decode_priority {
            return self.clock_ns();
        }
        self.sync_decode()?;
        self.sweep_evicts();
        Ok(self.sim.clock_ns())
    }

    fn sync_decode(&mut self) -> Result<(), Error> {
        let n = u16::try_from(self.sim.profile().n_gpus()).unwrap_or(1);
        for g in 0..n {
            self.sim.synchronize_stream(DeviceId(g), self.decode)?;
        }
        Ok(())
    }

    /// Next H2D that starts fails ([`gpu_sim::SimError::TransferFailed`]).
    pub fn fail_next_transfer(&mut self) {
        self.sim.fail_next_memcpy();
    }

    /// Mark the home GPU unavailable (new submits fail).
    pub fn set_gpu_unavailable(&mut self, yes: bool) -> Result<(), Error> {
        self.sim.set_unavailable(self.device, yes)?;
        Ok(())
    }

    /// Injected extra nanoseconds on every memcpy (transfer delay fault).
    pub fn set_transfer_delay_ns(&mut self, ns: u64) {
        self.sim.set_extra_transfer_ns(ns);
    }

    /// Cancel queued copy-stream ops. In-flight copies still complete.
    pub fn cancel_copy_stream(&mut self) -> Result<u32, Error> {
        Ok(self.sim.cancel_stream(self.device, self.copy)?)
    }

    /// Fault `keys` in (H2D or managed prefetch, no GEMM). Unknown catalog keys are skipped.
    pub fn prefetch(&mut self, keys: &[ExpertKey]) -> Result<u64, Error> {
        self.sweep_evicts();
        let n = self.cache.prefetch(keys)?;
        for key in keys {
            if self.cache.is_resident(*key) && !self.pages.contains_key(key) {
                self.place(*key)?;
            }
        }
        Ok(n)
    }

    /// Pin against eviction (sticky, survives compute `release`) and, on
    /// multi-GPU profiles, NVLink-replicate onto `(home + 1) % n_gpus`.
    ///
    /// Waits this page's GEMM lease before the replica copy so managed
    /// `prefetch` is not [`gpu_sim::SimError::Leased`]. Managed +
    /// `cudaMemAdviseSetAccessedBy` / VMM `cuMemSetAccess` / pinned
    /// `cudaMemPoolSetAccess` maps the dest without a prefetch or D2D.
    /// [`GpuStoreCfg::multicast`] maps dest VMM physicals then one NVLS kernel
    /// (`cuMulticastCreate`) instead of copy-engine D2D.
    /// 1-GPU profiles skip that wait so leftover prefill GEMMs can overlap
    /// decode-priority ITL samples.
    pub fn pin_hot(&mut self, keys: &[ExpertKey]) -> Result<(), Error> {
        for key in keys {
            if !self.cache.contains_catalog(*key) {
                continue;
            }
            if !self.cache.is_resident(*key) {
                let _n = self.cache.prefetch(&[*key])?;
            }
            if !self.pages.contains_key(key) {
                self.place(*key)?;
            }
            self.wait_copy(*key)?;
            // Replica prefetch / D2D cannot start while a GEMM still leases the
            // page. 1-GPU sticky pin does not copy, so leftover prefill GEMMs
            // stay in flight for decode-priority ITL.
            if self.sim.profile().n_gpus() >= 2 {
                self.wait_page_idle(*key)?;
            }
            self.cache.pin_hot(&[*key])?;
            self.replicate(*key)?;
        }
        Ok(())
    }

    /// Move `key` onto `dst`. Pinned/VMM pages D2D unless [`GpuStoreCfg::accessed_by`]
    /// (retarget GEMM, keep home physicals / mempool); managed pages prefetch then drop the source copy unless AccessedBy (retarget GEMM, keep home residency); mapped pages retarget GEMM.
    ///
    /// Source HBM is released after the copy is stream-ordered; destination HBM
    /// is charged by the peer memcpy. Dest compute can overlap other GPUs.
    pub fn migrate(&mut self, key: ExpertKey, dst: DeviceId) -> Result<(), Error> {
        if self.sim.profile().n_gpus() < 2 {
            return Err(Error::Store("no peer"));
        }
        let _gpu = self.sim.profile().gpu(dst)?;
        if !self.pages.contains_key(&key) {
            if !self.cache.is_resident(key) {
                return Err(Error::Store("not resident"));
            }
            self.place(key)?;
        }
        let (id, src) = {
            let page = self.pages.get(&key).ok_or(Error::Store("missing handle"))?;
            (page.id, page.device)
        };
        if src == dst {
            return Ok(());
        }
        // Host-sync managed/VMM drops, and stream-ordered pinned free, must not
        // race a live GEMM lease. Drain this page (and its replica) only — do
        // not `synchronize()` the whole device, or overlapping H2D of other
        // experts would collapse into the migrate.
        self.wait_page_idle(key)?;
        if let Some(g) = self.graphs.remove(&id) {
            self.sim.destroy_graph(g)?;
        }
        match self.mode {
            GpuFill::Managed => return self.migrate_managed(key, id, src, dst),
            GpuFill::Mapped => return self.migrate_mapped(key, dst),
            GpuFill::Vmm => return self.migrate_vmm(key, id, src, dst),
            GpuFill::Pinned if self.accessed_by && !self.sync_alloc => {
                return self.migrate_pool_peer(key, dst);
            }
            GpuFill::Pinned => {}
        }
        let already = self.sim.is_resident(id, dst)?;
        if !already {
            let _c =
                self.sim
                    .memcpy_device_to_device(src, dst, id, self.bytes_per_expert, self.copy)?;
        }
        let ev_copy = EventId(self.next_event);
        self.next_event = self.next_event.saturating_add(1);
        self.sim.create_event_disable_timing(ev_copy)?;
        let _r = self.sim.record_event(src, ev_copy, self.copy)?;
        // Copy-engine free must not race a compute-stream lease on src.
        let ev_compute = EventId(self.next_event);
        self.next_event = self.next_event.saturating_add(1);
        self.sim.create_event_disable_timing(ev_compute)?;
        let _r2 = self.sim.record_event(src, ev_compute, self.compute)?;
        let _w = self.sim.wait_event(src, ev_compute, self.copy)?;
        self.sim.free(src, id, self.copy)?;
        let _gone = self.replicas.remove(&key);
        if let Some(page) = self.pages.get_mut(&key) {
            page.device = dst;
            page.ready = Some(ev_copy);
        }
        self.note_migrate();
        Ok(())
    }

    /// GEMM retarget only — [`gpu_sim::Sim::pool_set_access`] already maps `dst`.
    fn migrate_pool_peer(&mut self, key: ExpertKey, dst: DeviceId) -> Result<(), Error> {
        let _gone = self.replicas.remove(&key);
        if let Some(page) = self.pages.get_mut(&key) {
            page.device = dst;
            page.ready = None;
        }
        self.note_migrate();
        Ok(())
    }

    /// GPU0↔GPU1 link bandwidth, or a 32 GB/s PCIe-order fallback.
    #[must_use]
    pub fn peer_bps(&self) -> u64 {
        self.sim
            .profile()
            .link(Some(DeviceId(0)), Some(DeviceId(1)))
            .map_or(32_000_000_000, |l| l.bps)
    }

    /// Payload bytes billed per expert page.
    #[must_use]
    pub fn expert_bytes(&self) -> u64 {
        self.bytes_per_expert
    }

    /// [`plan_placement`] on the GPU0↔GPU1 hop: D2D weights onto GPU0, or count a dispatch.
    ///
    /// 1-GPU profiles skip (`Ok`). [`Self::migrate`] stays unconditional. `reuse` is
    /// how many times this key has been selected for keep-hot so far (online).
    /// Fail-loud: a leased managed/VMM drop is [`Error::Sim`], not a swallowed
    /// no-op (Engine serving used to ignore that `Result`).
    pub fn place_hot(&mut self, key: ExpertKey, reuse: u64, fan_in: u64) -> Result<(), Error> {
        if self.sim.profile().n_gpus() < 2 {
            return Ok(());
        }
        match plan_placement(
            self.bytes_per_expert,
            DECODE_ACTIVATION_BYTES,
            fan_in,
            reuse,
            self.peer_bps(),
        ) {
            Placement::MoveWeights => self.migrate(key, DeviceId(0)),
            Placement::DispatchActivations => {
                self.dispatches = self.dispatches.saturating_add(1);
                Ok(())
            }
        }
    }

    fn note_migrate(&mut self) {
        self.migrates = self.migrates.saturating_add(1);
    }

    /// Prefetch `dst` (ReadMostly keeps extras) then drop the source copy.
    ///
    /// [`Self::accessed_by`]: GEMM retarget only — AccessedBy already maps `dst`.
    /// [`Self::wait_page_idle`] already drained this alloc's kernel leases.
    fn migrate_managed(
        &mut self,
        key: ExpertKey,
        id: AllocId,
        src: DeviceId,
        dst: DeviceId,
    ) -> Result<(), Error> {
        if self.accessed_by {
            let _gone = self.replicas.remove(&key);
            if let Some(page) = self.pages.get_mut(&key) {
                page.device = dst;
                page.ready = None;
            }
            self.note_migrate();
            return Ok(());
        }
        if !self.sim.is_resident(id, dst)? {
            let _p = self.sim.prefetch(dst, id, self.copy)?;
            self.sim.synchronize_stream(dst, self.copy)?;
        }
        if self.sim.is_resident(id, src)? {
            self.sim.drop_managed_copy(id, src)?;
        }
        let _gone = self.replicas.remove(&key);
        if let Some(page) = self.pages.get_mut(&key) {
            page.device = dst;
            page.ready = None;
        }
        self.note_migrate();
        Ok(())
    }

    /// Mapped host is already kernel-readable on every GPU; only the GEMM device changes.
    fn migrate_mapped(&mut self, key: ExpertKey, dst: DeviceId) -> Result<(), Error> {
        let _gone = self.replicas.remove(&key);
        if let Some(page) = self.pages.get_mut(&key) {
            page.device = dst;
            page.ready = None;
        }
        self.note_migrate();
        Ok(())
    }

    /// Map `dst`, D2D, then unmap the source physicals.
    ///
    /// [`Self::accessed_by`]: GEMM retarget only — `va_set_access` already maps `dst`.
    fn migrate_vmm(
        &mut self,
        key: ExpertKey,
        id: AllocId,
        src: DeviceId,
        dst: DeviceId,
    ) -> Result<(), Error> {
        if self.accessed_by {
            let _gone = self.replicas.remove(&key);
            if let Some(page) = self.pages.get_mut(&key) {
                page.device = dst;
                page.ready = None;
            }
            self.note_migrate();
            return Ok(());
        }
        self.sim.synchronize_stream(src, self.prefill)?;
        if self.decode != self.prefill {
            self.sim.synchronize_stream(src, self.decode)?;
        }
        if !self.sim.is_resident(id, dst)? {
            self.sim.va_map(id, dst)?;
            let _c =
                self.sim
                    .memcpy_device_to_device(src, dst, id, self.bytes_per_expert, self.copy)?;
            self.sim.synchronize_stream(src, self.copy)?;
        }
        if self.sim.is_resident(id, src)? {
            self.sim.va_unmap_range(id, src, 0, self.bytes_per_expert)?;
        }
        let _gone = self.replicas.remove(&key);
        if let Some(page) = self.pages.get_mut(&key) {
            page.device = dst;
            page.ready = None;
        }
        self.note_migrate();
        Ok(())
    }

    /// GPU that currently holds `key`, if it has been placed.
    #[must_use]
    pub fn device_of(&self, key: ExpertKey) -> Option<DeviceId> {
        self.pages.get(&key).map(|p| p.device)
    }

    /// Live HBM bytes on `device` (does not drain).
    pub fn hbm_used(&self, device: DeviceId) -> Result<u64, Error> {
        Ok(self.sim.hbm_used(device)?)
    }

    /// Whether the GPU page for `key` is resident on `device`.
    #[must_use]
    pub fn page_resident(&self, key: ExpertKey, device: DeviceId) -> bool {
        self.pages
            .get(&key)
            .and_then(|p| self.sim.is_resident(p.id, device).ok())
            .unwrap_or(false)
    }

    /// Whether `device` has SetAccessedBy / VMM `va_set_access` / mempool
    /// `pool_set_access` on `key`.
    #[must_use]
    pub fn page_accessed_by(&self, key: ExpertKey, device: DeviceId) -> bool {
        self.pages
            .get(&key)
            .and_then(|p| self.sim.is_accessed_by(p.id, device).ok())
            .unwrap_or(false)
    }

    /// True when any placed page is SetAccessedBy / VMM `va_set_access` /
    /// mempool `pool_set_access` on `device`.
    #[must_use]
    pub fn any_page_accessed_by(&self, device: DeviceId) -> bool {
        self.pages.keys().any(|k| self.page_accessed_by(*k, device))
    }

    /// [`gpu_sim::Sim::stream_priority`] for `(device, stream)`.
    #[must_use]
    pub fn stream_priority(&self, device: DeviceId, stream: StreamId) -> i32 {
        self.sim.stream_priority(device, stream)
    }

    /// How many times a captured GEMM graph was launched.
    #[must_use]
    pub fn graph_launches(&self) -> u64 {
        self.graph_launches
    }

    /// How many times [`gpu_sim::Sim::update_graph`] reused a parked exec.
    #[must_use]
    pub fn graph_updates(&self) -> u64 {
        self.graph_updates
    }

    /// How many times [`gpu_sim::Sim::clone_graph`] copied a capture before instantiate.
    #[must_use]
    pub fn graph_clones(&self) -> u64 {
        self.graph_clones
    }

    /// How many times [`gpu_sim::Sim::graph_exec_kernel_set_params`] reused a parked exec.
    #[must_use]
    pub fn graph_set_params(&self) -> u64 {
        self.graph_set_params_n
    }

    /// Sum of [`gpu_sim::Sim::event_elapsed_ns`] on copy start/end events.
    ///
    /// Zero unless [`GpuStoreCfg::timing_events`] and a copy was waited.
    #[must_use]
    pub fn copy_elapsed_ns(&self) -> u64 {
        self.copy_elapsed_ns
    }

    /// Whether `key` is in the fast CPU tier.
    #[must_use]
    pub fn is_resident(&self, key: ExpertKey) -> bool {
        self.cache.is_resident(key)
    }

    /// Fast-tier capacity (may be smaller than the constructor `slots` for mapped pages).
    #[must_use]
    pub fn slots(&self) -> usize {
        self.cache.slots()
    }

    /// GPUs in the attached hardware profile.
    #[must_use]
    pub fn n_gpus(&self) -> usize {
        self.sim.profile().n_gpus()
    }

    /// Sticky keep-hot budget: leave one slot for demand paging.
    #[must_use]
    pub fn pin_budget(&self) -> usize {
        self.cache.pin_budget()
    }

    /// Whether `key` has a sticky [`Self::pin_hot`] pin.
    #[must_use]
    pub fn is_pinned(&self, key: ExpertKey) -> bool {
        self.cache.is_pinned(key)
    }

    /// Drop every sticky pin. In-flight leases are unchanged.
    pub fn unpin_all(&mut self) {
        self.cache.unpin_all();
    }

    /// PLAN state: GPU copies are Transferring until the copy-stream event completes.
    #[must_use]
    pub fn phase(&self, key: ExpertKey) -> ExpertPhase {
        if let Some(page) = self.evicting.get(&key) {
            if self.sim.is_resident(page.id, page.device).unwrap_or(false) {
                return ExpertPhase::Evicting;
            }
            return ExpertPhase::Cold;
        }
        if let Some(page) = self.pages.get(&key) {
            if let Some(ev) = page.ready {
                if !self.sim.query_event(ev).unwrap_or(false) {
                    return ExpertPhase::Transferring;
                }
            }
        }
        ExpertPhase::cpu(self.cache.is_resident(key), self.cache.is_held(key))
    }

    /// Drop `key` from HBM. Illegal while leased. Stays [`ExpertPhase::Evicting`]
    /// until the stream-ordered free completes.
    pub fn evict(&mut self, key: ExpertKey) -> Result<(), Error> {
        self.sweep_evicts();
        self.cache.evict(key)?;
        self.drop_gpu(key)
    }

    /// Drain GEMM and copy on this page's device (and replica), not the whole sim.
    fn wait_page_idle(&mut self, key: ExpertKey) -> Result<(), Error> {
        let (device, replica) = {
            let Some(page) = self.pages.get(&key) else {
                return Ok(());
            };
            (page.device, self.replicas.get(&key).copied())
        };
        self.wait_compute(device)?;
        self.sim.synchronize_stream(device, self.copy)?;
        if let Some(dst) = replica {
            if dst != device {
                self.wait_compute(dst)?;
                self.sim.synchronize_stream(dst, self.copy)?;
            }
        }
        Ok(())
    }

    fn wait_compute(&mut self, device: DeviceId) -> Result<(), Error> {
        self.sim.synchronize_stream(device, self.prefill)?;
        if self.decode != self.prefill {
            self.sim.synchronize_stream(device, self.decode)?;
        }
        Ok(())
    }

    fn wait_copy(&mut self, key: ExpertKey) -> Result<(), Error> {
        let (device, ready, start) = {
            let page = self.pages.get(&key).ok_or(Error::Store("missing handle"))?;
            (page.device, page.ready, page.start)
        };
        if let Some(ev) = ready {
            if !self.sim.query_event(ev)? {
                // Wait the DMA stream only. Syncing compute here would drain
                // leftover prefill GEMMs before decode-priority can overlap them.
                self.sim.synchronize_stream(device, self.copy)?;
            }
            self.note_copy_elapsed(start, ev)?;
            if let Some(page) = self.pages.get_mut(&key) {
                page.ready = None;
                page.start = None;
            }
        }
        Ok(())
    }

    fn place(&mut self, key: ExpertKey) -> Result<(), Error> {
        if let Some(v) = self.cache.take_victim() {
            self.drop_gpu(v)?;
        }
        if self.pages.contains_key(&key) {
            return Ok(());
        }
        let d = self.home(key);
        let start = self.record_copy_start(d)?;
        let id = self.fill_page(d)?;
        let ev = self.create_copy_event()?;
        let _r = self.sim.record_event(d, ev, self.copy)?;
        let _prev = self.pages.insert(
            key,
            GpuPage {
                id,
                device: d,
                ready: Some(ev),
                start,
            },
        );
        Ok(())
    }

    fn create_copy_event(&mut self) -> Result<EventId, Error> {
        let ev = EventId(self.next_event);
        self.next_event = self.next_event.saturating_add(1);
        if self.timing_events {
            self.sim.create_event(ev)?;
        } else {
            self.sim.create_event_disable_timing(ev)?;
        }
        Ok(ev)
    }

    fn record_copy_start(&mut self, d: DeviceId) -> Result<Option<EventId>, Error> {
        if !self.timing_events || self.mode == GpuFill::Mapped {
            return Ok(None);
        }
        let ev = self.create_copy_event()?;
        let _r = self.sim.record_event(d, ev, self.copy)?;
        Ok(Some(ev))
    }

    fn note_copy_elapsed(&mut self, start: Option<EventId>, end: EventId) -> Result<(), Error> {
        if !self.timing_events {
            return Ok(());
        }
        let Some(st) = start else {
            return Ok(());
        };
        self.sim.synchronize_event(st)?;
        self.sim.synchronize_event(end)?;
        let ns = self.sim.event_elapsed_ns(st, end)?;
        self.copy_elapsed_ns = self.copy_elapsed_ns.saturating_add(ns);
        Ok(())
    }

    fn fill_page(&mut self, d: DeviceId) -> Result<AllocId, Error> {
        let bytes = self.bytes_per_expert;
        match self.mode {
            GpuFill::Managed => {
                let id = self.sim.alloc_managed(bytes)?;
                self.sim.mem_advise(id, MemAdvise::SetReadMostly, d)?;
                self.sim
                    .mem_advise(id, MemAdvise::SetPreferredLocation, d)?;
                if self.accessed_by {
                    advise_accessed_by(&mut self.sim, id)?;
                }
                let _p = self.sim.prefetch(d, id, self.copy)?;
                if self.sync_alloc {
                    self.sim.synchronize_stream(d, self.copy)?;
                }
                Ok(id)
            }
            GpuFill::Mapped => Ok(self.sim.alloc_host_mapped(bytes)?),
            GpuFill::Vmm => {
                let id = self.vmm_alloc(d)?;
                self.fill_hbm(d, id)?;
                if self.accessed_by {
                    advise_vmm_access(&mut self.sim, id)?;
                }
                Ok(id)
            }
            GpuFill::Pinned => {
                let id = self.hbm_alloc(d)?;
                self.fill_hbm(d, id)?;
                Ok(id)
            }
        }
    }

    fn vmm_alloc(&mut self, d: DeviceId) -> Result<AllocId, Error> {
        let bytes = self.bytes_per_expert;
        if self.vmm_page > 0 && self.vmm_page < bytes {
            Ok(self.sim.va_acquire_paged(d, bytes, self.vmm_page)?)
        } else {
            Ok(self.sim.va_acquire(d, bytes)?)
        }
    }

    fn hbm_alloc(&mut self, d: DeviceId) -> Result<AllocId, Error> {
        if self.sync_alloc {
            Ok(self.sim.malloc(d, self.bytes_per_expert)?)
        } else {
            // Stream-ordered `cudaMallocAsync`. `malloc` would device-sync.
            Ok(self.sim.alloc(d, self.bytes_per_expert, self.copy)?)
        }
    }

    fn fill_hbm(&mut self, d: DeviceId, id: AllocId) -> Result<(), Error> {
        let bytes = self.bytes_per_expert;
        if self.pageable {
            let _id = self.sim.memcpy_host_to_device(d, id, bytes, self.copy)?;
        } else if self.sync_alloc {
            let _id = self.sim.memcpy_sync(
                d,
                MemcpyOp {
                    src: Place::HostPinned,
                    dst: Place::Device(d),
                    alloc: id,
                    bytes,
                    offset: 0,
                },
                self.copy,
            )?;
        } else {
            let _c = self.sim.memcpy_pinned_to_device(d, id, bytes, self.copy)?;
        }
        Ok(())
    }

    fn home(&self, key: ExpertKey) -> DeviceId {
        let n = u16::try_from(self.sim.profile().n_gpus())
            .unwrap_or(1)
            .max(1);
        home_gpu(key, n)
    }

    fn gemm_resident(&mut self, key: ExpertKey) -> Result<(), Error> {
        let (id, device, ready, start) = {
            let page = self
                .pages
                .get_mut(&key)
                .ok_or(Error::Store("missing handle"))?;
            let ready = page.ready.take();
            let start = page.start.take();
            (page.id, page.device, ready, start)
        };
        if let Some(ev) = ready {
            if !self.sim.query_event(ev)? {
                let _w = self.sim.wait_event(device, ev, self.compute)?;
            }
            self.note_copy_elapsed(start, ev)?;
        }
        self.launch_or_gemm(device, id)?;
        if self.host_func {
            let _id = self.sim.host_func(device, self.compute)?;
        }
        Ok(())
    }

    fn launch_or_gemm(&mut self, device: DeviceId, id: AllocId) -> Result<(), Error> {
        let flags = self.gemm_flags();
        if let Some(g) = self.graphs.get(&id).copied() {
            self.graph_launches = self.graph_launches.saturating_add(1);
            let _n = self.sim.launch_graph(g, self.compute)?;
            return Ok(());
        }
        if self.graph_set_params {
            if let Some(exec) = self.idle_execs.get_mut(&device).and_then(Vec::pop) {
                retarget_parked_kernel(&mut self.sim, exec, id)?;
                self.graph_set_params_n = self.graph_set_params_n.saturating_add(1);
                self.sim.upload_graph(exec)?;
                let _prev = self.graphs.insert(id, exec);
                self.graph_launches = self.graph_launches.saturating_add(1);
                let _n = self.sim.launch_graph(exec, self.compute)?;
                return Ok(());
            }
        }
        if self.graph_build {
            let src = self.build_gemm_graph(device, id)?;
            let g = self.bind_graph(device, src)?;
            let _prev = self.graphs.insert(id, g);
            self.graph_launches = self.graph_launches.saturating_add(1);
            let _n = self.sim.launch_graph(g, self.compute)?;
            return Ok(());
        }
        if !self.sim.query_stream(device, self.compute)? {
            self.sim.synchronize_stream(device, self.compute)?;
        }
        if self.sim.query_stream(device, self.compute)? {
            if self.graph_piecewise {
                let src = self.piecewise_gemm_graph(device, id)?;
                let g = self.bind_graph(device, src)?;
                let _prev = self.graphs.insert(id, g);
                self.graph_launches = self.graph_launches.saturating_add(1);
                let _n = self.sim.launch_graph(g, self.compute)?;
                return Ok(());
            }
            self.sim.begin_capture(device, self.compute)?;
            gemm_leaf(&mut self.sim, device, self.compute, id, self.leaf, flags)?;
            let src = self.sim.end_capture()?;
            let g = self.bind_graph(device, src)?;
            let _prev = self.graphs.insert(id, g);
            self.graph_launches = self.graph_launches.saturating_add(1);
            let _n = self.sim.launch_graph(g, self.compute)?;
            return Ok(());
        }
        gemm_leaf(
            &mut self.sim,
            device,
            self.compute,
            id,
            LeafMem::None,
            flags,
        )
    }

    fn build_gemm_graph(&mut self, device: DeviceId, id: AllocId) -> Result<GraphId, Error> {
        let flags = self.gemm_flags();
        let g = self.sim.create_graph(device, self.compute)?;
        add_leaf_gemm(&mut self.sim, g, id, self.leaf, flags)?;
        Ok(g)
    }

    fn piecewise_gemm_graph(&mut self, device: DeviceId, id: AllocId) -> Result<GraphId, Error> {
        let flags = self.gemm_flags();
        let g = self.sim.create_graph(device, self.compute)?;
        self.sim
            .begin_capture_to_graph(device, self.compute, g, &[])?;
        gemm_leaf(&mut self.sim, device, self.compute, id, self.leaf, flags)?;
        let ended = self.sim.end_capture()?;
        if ended != g {
            return Err(Error::Store("capture-to-graph id"));
        }
        Ok(g)
    }

    fn bind_graph(&mut self, device: DeviceId, src: GraphId) -> Result<GraphId, Error> {
        if self.graph_update && self.leaf == LeafMem::None {
            if let Some(exec) = self.idle_execs.get_mut(&device).and_then(Vec::pop) {
                self.sim.update_graph(exec, src)?;
                self.sim.destroy_graph(src)?;
                self.graph_updates = self.graph_updates.saturating_add(1);
                self.sim.upload_graph(exec)?;
                return Ok(exec);
            }
        }
        let exec = if self.graph_clone {
            let cloned = self.sim.clone_graph(src)?;
            self.sim.destroy_graph(src)?;
            self.graph_clones = self.graph_clones.saturating_add(1);
            cloned
        } else {
            src
        };
        let exec = if self.leaf == LeafMem::AutoFree {
            self.sim.instantiate_graph_auto_free(exec)?
        } else {
            self.sim.instantiate_graph(exec)?
        };
        self.sim.upload_graph(exec)?;
        Ok(exec)
    }

    fn drop_gpu(&mut self, key: ExpertKey) -> Result<(), Error> {
        let Some(page) = self.pages.remove(&key) else {
            return Ok(());
        };
        let _prev = self.evicting.insert(key, page);
        self.finish_drop(key, page)
    }

    fn sweep_evicts(&mut self) {
        let done: Vec<ExpertKey> = self
            .evicting
            .iter()
            .filter_map(|(k, p)| {
                if self.sim.is_resident(p.id, p.device).unwrap_or(false) {
                    None
                } else {
                    Some(*k)
                }
            })
            .collect();
        for k in done {
            let _gone = self.evicting.remove(&k);
        }
    }

    fn finish_drop(&mut self, key: ExpertKey, page: GpuPage) -> Result<(), Error> {
        if let Some(g) = self.graphs.remove(&page.id) {
            if self.graph_set_params || (self.graph_update && self.leaf == LeafMem::None) {
                self.idle_execs.entry(page.device).or_default().push(g);
            } else {
                self.sim.destroy_graph(g)?;
            }
        }
        match self.mode {
            GpuFill::Managed => {
                self.wait_compute(page.device)?;
                self.sim.synchronize_stream(page.device, self.copy)?;
                if let Some(dst) = self.replicas.remove(&key) {
                    self.sim.synchronize_stream(dst, self.copy)?;
                }
                self.sim.free_sync(page.id)?;
                return Ok(());
            }
            GpuFill::Mapped => {
                self.wait_compute(page.device)?;
                self.sim.free_host_pinned(page.id)?;
                let _gone = self.replicas.remove(&key);
                return Ok(());
            }
            GpuFill::Vmm => {
                self.wait_compute(page.device)?;
                self.sim.va_release(page.id)?;
                let _gone = self.replicas.remove(&key);
                return Ok(());
            }
            GpuFill::Pinned if self.sync_alloc => {
                self.wait_compute(page.device)?;
                self.sim.free_sync(page.id)?;
                let _gone = self.replicas.remove(&key);
                return Ok(());
            }
            GpuFill::Pinned => {
                self.wait_compute(page.device)?;
            }
        }
        // Copy-engine free must not race a compute-stream lease on the same page.
        let ev = EventId(self.next_event);
        self.next_event = self.next_event.saturating_add(1);
        self.sim.create_event_disable_timing(ev)?;
        let _r = self.sim.record_event(page.device, ev, self.compute)?;
        let _w = self.sim.wait_event(page.device, ev, self.copy)?;
        if let Some(dst) = self.replicas.remove(&key) {
            if self.sim.is_resident(page.id, dst)? {
                self.sim.free(dst, page.id, self.copy)?;
            }
        }
        self.sim.free(page.device, page.id, self.copy)?;
        Ok(())
    }

    fn replicate(&mut self, key: ExpertKey) -> Result<(), Error> {
        if self.sim.profile().n_gpus() < 2 {
            return Ok(());
        }
        if self.replicas.contains_key(&key) {
            return Ok(());
        }
        let (id, src) = {
            let page = self.pages.get(&key).ok_or(Error::Store("missing handle"))?;
            (page.id, page.device)
        };
        let Some(dst) = self.replica_dst(src) else {
            return Ok(());
        };
        if src == dst {
            let _prev = self.replicas.insert(key, dst);
            return Ok(());
        }
        match self.mode {
            GpuFill::Managed => {
                if !self.accessed_by {
                    let _p = self.sim.prefetch(dst, id, self.copy)?;
                    self.note_replicate();
                }
                let _prev = self.replicas.insert(key, dst);
                return Ok(());
            }
            GpuFill::Mapped => {
                let _prev = self.replicas.insert(key, dst);
                return Ok(());
            }
            GpuFill::Vmm => {
                if self.accessed_by {
                    if !self.sim.is_accessed_by(id, dst)? {
                        self.sim.va_set_access(id, dst)?;
                    }
                } else if self.multicast {
                    if !self.sim.is_resident(id, dst)? {
                        self.sim.va_map(id, dst)?;
                    }
                    let _c = self.sim.multicast_store(src, id, &[dst], self.copy)?;
                    self.note_replicate();
                } else if !self.sim.is_resident(id, dst)? {
                    self.sim.va_map(id, dst)?;
                    let _c = self.sim.memcpy_device_to_device(
                        src,
                        dst,
                        id,
                        self.bytes_per_expert,
                        self.copy,
                    )?;
                    self.note_replicate();
                }
                let _prev = self.replicas.insert(key, dst);
                return Ok(());
            }
            GpuFill::Pinned => {
                if self.accessed_by && !self.sync_alloc {
                    let _prev = self.replicas.insert(key, dst);
                    return Ok(());
                }
            }
        }
        let _c =
            self.sim
                .memcpy_device_to_device(src, dst, id, self.bytes_per_expert, self.copy)?;
        self.note_replicate();
        let _prev = self.replicas.insert(key, dst);
        Ok(())
    }

    /// Next GPU after `src` on the profile mesh (`None` when `n_gpus < 2`).
    #[must_use]
    fn replica_dst(&self, src: DeviceId) -> Option<DeviceId> {
        let n = u16::try_from(self.sim.profile().n_gpus()).unwrap_or(1);
        if n < 2 {
            return None;
        }
        Some(DeviceId(src.0.wrapping_add(1) % n))
    }

    fn note_replicate(&mut self) {
        self.replicates = self.replicates.saturating_add(1);
    }

    /// Peer GPU that holds a [`Self::pin_hot`] replica of `key`, if any.
    #[must_use]
    pub fn replica_of(&self, key: ExpertKey) -> Option<DeviceId> {
        self.replicas.get(&key).copied()
    }

    /// Reserve a VMM VA for Engine interned KV (`pool_blocks` pages).
    ///
    /// No-op unless [`GpuStoreCfg::kv_sim`]. Idempotent after the first bind.
    /// Faults `cuMemCreate` a physical and `cuMemMap` it; drop releases the handle.
    pub fn bind_kv(&mut self, n_pages: u32, page_bytes: u64) -> Result<(), Error> {
        if !self.kv_sim {
            return Ok(());
        }
        if self.kv.is_some() {
            return Ok(());
        }
        if n_pages == 0 || page_bytes == 0 {
            return Err(Error::Store("kv sim pages"));
        }
        let va_bytes = u64::from(n_pages).saturating_mul(page_bytes);
        if va_bytes == 0 {
            return Err(Error::Store("kv sim pages"));
        }
        let va = self.sim.va_reserve(va_bytes)?;
        self.kv = Some(KvGpu {
            va,
            page_bytes,
            n_pages,
            mapped: BTreeSet::new(),
            handles: BTreeMap::new(),
            hits: 0,
            misses: 0,
        });
        Ok(())
    }

    /// Replay Engine intern/alloc events onto the reserved KV VA.
    pub fn apply_kv_ops(&mut self, ops: &[KvSimOp]) -> Result<(), Error> {
        if self.kv.is_none() {
            return Ok(());
        }
        for op in ops {
            match *op {
                KvSimOp::Fault(id) => self.kv_fault(id)?,
                KvSimOp::Hit(id) => self.kv_hit(id)?,
                KvSimOp::Cow { dst, .. } => self.kv_fault(dst)?,
                KvSimOp::Drop(id) => self.kv_drop(id)?,
            }
        }
        Ok(())
    }

    /// Intern-hit kernels billed on the KV VA.
    #[must_use]
    pub fn kv_hits(&self) -> u64 {
        self.kv.as_ref().map_or(0, |k| k.hits)
    }

    /// Map+memset fills billed on the KV VA.
    #[must_use]
    pub fn kv_misses(&self) -> u64 {
        self.kv.as_ref().map_or(0, |k| k.misses)
    }

    fn kv_off(&self, id: u32) -> Result<(AllocId, u64, u64), Error> {
        let kv = self.kv.as_ref().ok_or(Error::Store("kv sim unbound"))?;
        if id >= kv.n_pages {
            return Err(Error::Store("kv page id"));
        }
        Ok((
            kv.va,
            u64::from(id).saturating_mul(kv.page_bytes),
            kv.page_bytes,
        ))
    }

    fn kv_map(&mut self, id: u32) -> Result<(AllocId, u64, u64), Error> {
        let (va, off, bytes) = self.kv_off(id)?;
        let already = self.kv.as_ref().is_some_and(|k| k.mapped.contains(&id));
        if !already {
            self.sim.synchronize_stream(self.device, self.compute)?;
            let existing = self.kv.as_ref().and_then(|k| k.handles.get(&id).copied());
            if let Some(h) = existing {
                self.sim.va_map_handle(va, self.device, off, h)?;
            } else {
                let h = self.sim.va_create(self.device, bytes)?;
                if let Some(k) = self.kv.as_mut() {
                    let _prev = k.handles.insert(id, h);
                }
                self.sim.va_map_handle(va, self.device, off, h)?;
            }
            if let Some(k) = self.kv.as_mut() {
                let _ins = k.mapped.insert(id);
            }
        }
        Ok((va, off, bytes))
    }

    fn kv_fault(&mut self, id: u32) -> Result<(), Error> {
        let (va, off, bytes) = self.kv_map(id)?;
        let buf = KernelBuf::span(va, off, bytes);
        let _op = self.sim.memset_buf(self.device, buf, self.compute)?;
        if let Some(k) = self.kv.as_mut() {
            k.misses = k.misses.saturating_add(1);
        }
        Ok(())
    }

    fn kv_hit(&mut self, id: u32) -> Result<(), Error> {
        let (va, off, bytes) = self.kv_map(id)?;
        let buf = KernelBuf::span(va, off, bytes);
        let _op = self.sim.kernel_bufs(
            self.device,
            KernelKind::other(8, bytes),
            &[buf],
            &[buf],
            self.compute,
        )?;
        if let Some(k) = self.kv.as_mut() {
            k.hits = k.hits.saturating_add(1);
        }
        Ok(())
    }

    fn kv_drop(&mut self, id: u32) -> Result<(), Error> {
        let mapped = self.kv.as_ref().is_some_and(|k| k.mapped.contains(&id));
        if !mapped {
            return Ok(());
        }
        let (va, off, bytes) = self.kv_off(id)?;
        self.sim.va_unmap_range(va, self.device, off, bytes)?;
        let h = self.kv.as_ref().and_then(|k| k.handles.get(&id).copied());
        if let Some(h) = h {
            if self.sim.handle_maps(h)? == 0 {
                self.sim.va_release_handle(h)?;
            }
        }
        if let Some(k) = self.kv.as_mut() {
            let _gone = k.mapped.remove(&id);
            let _h = k.handles.remove(&id);
        }
        Ok(())
    }
}

impl ExpertStore for SimulatedGpuStore {
    fn acquire(&mut self, key: ExpertKey) -> Result<ExpertParts, Error> {
        self.sweep_evicts();
        let hit = self.cache.is_resident(key);
        let parts = self.cache.acquire(key)?;
        if !hit && !self.pages.contains_key(&key) {
            self.place(key)?;
        }
        self.gemm_resident(key)?;
        Ok(parts)
    }

    fn lease(&mut self, key: ExpertKey) -> Result<(), Error> {
        match self.phase(key) {
            ExpertPhase::Resident | ExpertPhase::Leased => self.cache.lease(key),
            ExpertPhase::Transferring => Err(Error::Store("lease of transferring expert")),
            ExpertPhase::Evicting => Err(Error::Store("lease of evicting expert")),
            ExpertPhase::Cold => Err(Error::Store("lease of non-resident expert")),
        }
    }

    fn release(&mut self, key: ExpertKey) {
        self.cache.release(key);
    }

    fn metrics(&self) -> StoreMetrics {
        let mut m = self.cache.metrics();
        m.bytes_moved = self.sim.bytes_moved();
        m.migrates = self.migrates;
        m.dispatches = self.dispatches;
        m.replicates = self.replicates;
        m
    }
}

/// Demand-page a trace through [`SimulatedGpuStore`] and drain the virtual clock.
pub struct StoreReplay {
    /// Cache hits/misses/evicts after every routed expert.
    pub metrics: StoreMetrics,
    /// Virtual wall after a device drain. No `$/M tokens`.
    pub score: Score,
    /// [`SimulatedGpuStore::graph_updates`] after the walk.
    pub graph_updates: u64,
    /// [`SimulatedGpuStore::graph_clones`] after the walk.
    pub graph_clones: u64,
    /// [`SimulatedGpuStore::graph_set_params`] after the walk.
    pub graph_set_params: u64,
    /// [`SimulatedGpuStore::copy_elapsed_ns`] after the walk.
    pub copy_elapsed_ns: u64,
}

impl StoreReplay {
    /// Single-line agent / CLI log.
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "store {} {} graph_updates={} graph_clones={} graph_set_params={} copy_elapsed_ns={}",
            self.metrics.line(),
            self.score.line(),
            self.graph_updates,
            self.graph_clones,
            self.graph_set_params,
            self.copy_elapsed_ns
        )
    }
}

/// Planner + CUDA knobs for [`store_replay_cfg`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoreReplayCfg {
    /// Cache slots (mapped occupancy may shrink this).
    pub slots: usize,
    /// Payload bytes per expert page.
    pub bytes_per_expert: u64,
    /// Miss fill path.
    pub fill: GpuFill,
    /// `SimCfg` subset on the store.
    pub gpu: GpuStoreCfg,
    /// Online prefetch (no JSONL future leak except [`Self::plan_window`]).
    pub prefetch: Prefetch,
    /// Upcoming-event window for [`plan_window`]. `0` leaves prefetch ungated.
    pub plan_window: usize,
    /// Stay vs Fetch threshold, permille of upcoming unique keys already resident.
    pub plan_threshold: u32,
}

impl StoreReplayCfg {
    /// Demand paging, pinned async H2D, no planner.
    #[must_use]
    pub fn demand(slots: usize, bytes_per_expert: u64, fill: GpuFill) -> Self {
        Self {
            slots,
            bytes_per_expert,
            fill,
            gpu: GpuStoreCfg::default(),
            prefetch: Prefetch::None,
            plan_window: 0,
            plan_threshold: 500,
        }
    }
}

/// Walk `trace.keys()` on a store, then [`SimulatedGpuStore::score`].
pub fn store_replay(
    trace: &Trace,
    profile: HardwareProfile,
    slots: usize,
    bytes_per_expert: u64,
    fill: GpuFill,
    cfg: GpuStoreCfg,
) -> Result<StoreReplay, Error> {
    store_replay_cfg(
        trace,
        profile,
        StoreReplayCfg {
            slots,
            bytes_per_expert,
            fill,
            gpu: cfg,
            prefetch: Prefetch::None,
            plan_window: 0,
            plan_threshold: 500,
        },
    )
}

/// [`store_replay`] plus copy-forward / Markov / `plan_window` prefetch.
pub fn store_replay_cfg(
    trace: &Trace,
    profile: HardwareProfile,
    run: StoreReplayCfg,
) -> Result<StoreReplay, Error> {
    let inner = DirectStore::from_trace(trace);
    let mut store = SimulatedGpuStore::with_cfg(
        inner,
        run.slots,
        profile,
        run.bytes_per_expert,
        run.fill,
        run.gpu,
    )?;
    let catalog: BTreeSet<ExpertKey> = trace.keys().into_iter().collect();
    let mut markov = Markov::new();
    let mut chain = ChainState::new();
    for (i, event) in trace.events.iter().enumerate() {
        store.bind_sequence(event.sequence);
        store.bind_decode_compute(event.token > 0);
        let ek = event.keys();
        for key in &ek {
            let _p = store.acquire(*key)?;
            store.release(*key);
        }
        if store_should_prefetch(&store, &catalog, &run, trace, i) {
            let predicted = predicted_keys(run.prefetch, &markov, chain.predecessor(event), &ek);
            let planned = if run.plan_window > 0 {
                window_keys(trace, i.saturating_add(1), run.plan_window)
            } else {
                Vec::new()
            };
            let fill: Vec<ExpertKey> = predicted.into_iter().chain(planned).collect();
            let _n = store.prefetch(&fill)?;
        }
        chain.observe(&mut markov, event);
    }
    let n = u64::try_from(trace.events.len()).unwrap_or(1);
    Ok(StoreReplay {
        metrics: store.metrics(),
        graph_updates: store.graph_updates(),
        graph_clones: store.graph_clones(),
        graph_set_params: store.graph_set_params(),
        copy_elapsed_ns: store.copy_elapsed_ns(),
        score: store.score()?.with_tokens(n),
    })
}

fn store_should_prefetch(
    store: &SimulatedGpuStore,
    catalog: &BTreeSet<ExpertKey>,
    run: &StoreReplayCfg,
    trace: &Trace,
    i: usize,
) -> bool {
    if run.plan_window == 0 {
        return true;
    }
    let resident: BTreeSet<ExpertKey> = catalog
        .iter()
        .copied()
        .filter(|k| store.is_resident(*k))
        .collect();
    !matches!(
        plan_window(
            &resident,
            trace,
            i.saturating_add(1),
            run.plan_window,
            run.plan_threshold,
        ),
        Plan::Stay
    )
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

fn gemm_leaf(
    sim: &mut Sim,
    d: DeviceId,
    s: StreamId,
    id: AllocId,
    mem: LeafMem,
    flags: GemmFlags,
) -> Result<(), Error> {
    if mem == LeafMem::None {
        launch_store_gemm(sim, d, s, id, &[], flags)?;
        return Ok(());
    }
    let scratch = sim.alloc(d, GRAPH_SCRATCH_BYTES, s)?;
    launch_store_gemm(sim, d, s, id, &[scratch], flags)?;
    if mem == LeafMem::Free {
        sim.free(d, scratch, s)?;
    }
    Ok(())
}

fn launch_store_gemm(
    sim: &mut Sim,
    d: DeviceId,
    s: StreamId,
    id: AllocId,
    writes: &[AllocId],
    flags: GemmFlags,
) -> Result<(), Error> {
    let _k = sim.kernel_with(d, gemm_kind(), &[id], writes, s, flags.kernel_attrs(id))?;
    Ok(())
}

fn add_leaf_gemm(
    sim: &mut Sim,
    graph: GraphId,
    id: AllocId,
    mem: LeafMem,
    flags: GemmFlags,
) -> Result<(), Error> {
    if mem == LeafMem::None {
        return add_store_gemm(sim, graph, id, &[], flags);
    }
    let scratch = sim.graph_add_alloc(graph, GRAPH_SCRATCH_BYTES)?;
    add_store_gemm(sim, graph, id, &[scratch], flags)?;
    sim.graph_add_dependencies(graph, 0, 1)?;
    if mem == LeafMem::Free {
        sim.graph_add_free(graph, scratch)?;
        sim.graph_add_dependencies(graph, 1, 2)?;
    }
    Ok(())
}

fn add_store_gemm(
    sim: &mut Sim,
    graph: GraphId,
    id: AllocId,
    writes: &[AllocId],
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
    Ok(())
}

/// Copy is NULL; prefill compute is stream 1. Seq-streams: copy `0 .. n_copy-1`,
/// prefill `StreamId(n_copy)`. Decode-priority adds `n_copy+1` at higher id.
fn copy_compute_streams(
    profile: &HardwareProfile,
    seq_streams: bool,
    decode_priority: bool,
) -> (StreamId, StreamId, StreamId, u8) {
    let (copy, prefill, mut mark) = if seq_streams {
        let n_copy = replay_streams(profile, true);
        let compute = StreamId(u16::from(n_copy));
        let mark = u8::try_from(u16::from(n_copy).saturating_add(1)).unwrap_or(u8::MAX);
        (StreamId(0), compute, mark)
    } else {
        (StreamId(0), StreamId(1), 2)
    };
    let decode = if decode_priority {
        mark = mark.saturating_add(1);
        StreamId(prefill.0.saturating_add(1))
    } else {
        prefill
    };
    (copy, prefill, decode, mark)
}

/// `cudaMemAdviseSetAccessedBy` on every GPU so a remote read does not migrate.
fn advise_accessed_by(sim: &mut Sim, id: AllocId) -> Result<(), Error> {
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    for g in 0..n {
        sim.mem_advise(id, MemAdvise::SetAccessedBy, DeviceId(g))?;
    }
    Ok(())
}

/// `cuMemSetAccess` PROT_READ on every GPU so a remote VMM read skips dest HBM.
fn advise_vmm_access(sim: &mut Sim, id: AllocId) -> Result<(), Error> {
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    for g in 0..n {
        sim.va_set_access(id, DeviceId(g))?;
    }
    Ok(())
}

/// `cudaMemPoolSetAccess` ReadWrite on every default pool for every GPU.
fn advise_pool_access(sim: &mut Sim) -> Result<(), Error> {
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    for g in 0..n {
        let home = DeviceId(g);
        let pool = sim.default_pool(home)?;
        for d in 0..n {
            sim.pool_set_access(pool, DeviceId(d))?;
        }
    }
    Ok(())
}

/// `--mapped` occupancy: `min(slots, pin / expert_bytes)`.
///
/// `fit == 0` keeps the requested slots so the first `alloc_host_mapped` is
/// [`gpu_sim::SimError::PinOom`] (same walker rule as `sim_replay --mapped`).
fn mapped_occupancy(slots: usize, fill: GpuFill, pin_bytes: u64, expert_bytes: u64) -> usize {
    if fill != GpuFill::Mapped {
        return slots;
    }
    let bytes = expert_bytes.max(1);
    let fit = usize::try_from(pin_bytes / bytes).unwrap_or(usize::MAX);
    if fit == 0 {
        slots
    } else {
        slots.min(fit)
    }
}
