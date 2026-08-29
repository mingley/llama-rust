//! Replay a trace through [`gpu_sim`] with pinned H2D fills on miss.

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
    AllocId, DType, DeviceId, EventId, GraphId, HardwareProfile, KernelKind, MemcpyOp, Place,
    Score, Sim, StreamId,
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
}

impl SimReplay {
    /// Single-line agent / CLI log.
    #[must_use]
    pub fn line(&self) -> String {
        let mut s = format!(
            "sim_ns={} bytes_moved={} hbm_peak={} energy_uj={}",
            self.sim_ns, self.bytes_moved, self.hbm_peak, self.energy_uj
        );
        if let Some(n) = self.ttft_ns {
            let _w = write!(s, " ttft_ns={n}");
        }
        if let Some(n) = self.itl_ns {
            let _w = write!(s, " itl_ns={n}");
        }
        let _w = write!(
            s,
            " hits={} misses={} prefetches={} prefetch_hits={} prefetch_waste={} graph_launches={} child_graphs={} graph_updates={} graph_clones={}",
            self.hits,
            self.misses,
            self.prefetches,
            self.prefetch_hits,
            self.prefetch_waste,
            self.graph_launches,
            self.child_graphs,
            self.graph_updates,
            self.graph_clones
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
    /// `cudaHostAllocMapped`: miss pages are mapped host, not HBM. Kernels run
    /// over PCIe with no H2D. Hits/misses follow the same walker; `hbm_peak`
    /// stays near zero. [`crate::SimulatedGpuStore::new`] stays on the H2D path;
    /// [`crate::SimulatedGpuStore::with_mapped`] uses this path.
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
    /// Does not require an idle stream. Implies [`Self::cuda_graphs`]. Decode
    /// identity stays stream capture. [`crate::GpuStoreCfg::graph_build`] is
    /// the store path.
    pub graph_build: bool,
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
            mapped: false,
            managed: false,
            vmm: false,
            vmm_page: 0,
            host_func: false,
            blocking_streams: false,
            pageable: false,
            accessed_by: false,
            legacy_null: false,
            stream_priority: false,
            graph_update: false,
            graph_clone: false,
            graph_build: false,
            compute_slots: 0,
            decode_sm_permille: 0,
            decode_priority: false,
        }
    }
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
    let keys = trace.keys();
    let mut sim = Sim::new(sim_profile(profile, &cfg));
    if cfg.mempool {
        sim.set_default_pool_release_threshold(u64::MAX)?;
    }
    advise_pool_access_if_pinned(&mut sim, &cfg)?;
    let d = DeviceId(0);
    let s = StreamId(0);
    let bytes = cfg.bytes_per_expert.max(1);
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
        accessed_by: cfg.accessed_by,
    };
    let mut token_ends: Vec<u64> = Vec::new();
    let mut ctr = ReplayCounters::default();
    let mut markov = Markov::new();
    let mut chain = ChainState::new();
    let mut prefetched: BTreeSet<ExpertKey> = BTreeSet::new();
    let mut graphs = GraphBank::new(cfg.graph_update, cfg.graph_clone, cfg.graph_build);
    let mut admitted: BTreeSet<u64> = BTreeSet::new();
    let mut next_event = 1u32;
    for (i, event) in trace.events.iter().enumerate() {
        args.s = plan.work(event.sequence, event.token);
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
            &handles,
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
            for key in predicted.into_iter().chain(planned) {
                match w.prefetch_touch(key) {
                    Touch::Hit => {}
                    miss @ Touch::Miss { .. } => {
                        ctr.prefetches = ctr.prefetches.saturating_add(1);
                        let _ins = prefetched.insert(key);
                        apply_touch(
                            &mut sim,
                            &mut handles,
                            &mut graphs,
                            fill,
                            key,
                            miss,
                            &mut next_event,
                        )?;
                    }
                }
            }
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
    /// [`SimCfg::accessed_by`]: SetAccessedBy / VMM SetAccess / mempool SetAccess
    /// on every GPU (fill or default pools).
    pub accessed_by: bool,
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
            },
            stream,
        )?;
    } else {
        let _c = sim.memcpy_pinned_to_device(device, alloc, bytes, stream)?;
    }
    Ok(())
}

fn hbm_h2d(sim: &mut Sim, args: TouchArgs, alloc: AllocId) -> Result<(), Error> {
    if args.pageable {
        let _id = sim.memcpy_host_to_device(args.d, alloc, args.bytes, args.s)?;
        return Ok(());
    }
    hbm_h2d_pinned(sim, args.d, alloc, args.bytes, args.s, args.sync_alloc)
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

/// `cudaMemPoolSetAccess` ReadWrite on every default pool for every GPU.
pub(crate) fn advise_pool_access(sim: &mut Sim) -> Result<(), Error> {
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

/// Pinned `cudaMallocAsync` + `--accessed-by` (not mapped/managed/VMM/`cudaMalloc`).
pub(crate) fn advise_pool_access_if_pinned(sim: &mut Sim, cfg: &SimCfg) -> Result<(), Error> {
    if cfg.accessed_by && !cfg.mapped && !cfg.managed && !cfg.vmm && !cfg.sync_alloc {
        advise_pool_access(sim)?;
    }
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
}

/// Instantiated CUDA graph execs, optionally parked for `update_graph`.
pub(crate) struct GraphBank {
    graphs: BTreeMap<Vec<AllocId>, (GraphId, (DeviceId, StreamId))>,
    idle: BTreeMap<(DeviceId, StreamId), Vec<GraphId>>,
    update: bool,
    clone: bool,
    build: bool,
    pub updates: u64,
    pub clones: u64,
}

impl GraphBank {
    pub(crate) fn new(update: bool, clone: bool, build: bool) -> Self {
        Self {
            graphs: BTreeMap::new(),
            idle: BTreeMap::new(),
            update,
            clone,
            build,
            updates: 0,
            clones: 0,
        }
    }

    pub(crate) fn get(&self, ids: &[AllocId]) -> Option<GraphId> {
        self.graphs.get(ids).map(|(g, _)| *g)
    }

    /// Instantiate `src`, or `update_graph` a parked leaf on `origin`.
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
        let gid = if self.update && ids.len() == 1 {
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
        instantiate_src(sim, exec)
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
            if self.update && ids.len() == 1 {
                self.idle.entry(origin).or_default().push(gid);
            } else {
                sim.destroy_graph(gid)?;
            }
        }
        Ok(())
    }
}

fn instantiate_src(sim: &mut Sim, src: GraphId) -> Result<GraphId, Error> {
    sim.instantiate_graph(src)?;
    sim.upload_graph(src)?;
    Ok(src)
}

pub(crate) struct PageHandle {
    pub(crate) id: AllocId,
    pub(crate) stream: StreamId,
    pub(crate) device: DeviceId,
    /// Extra devices that hold a D2D / VMM-map replica of `id`.
    pub(crate) replicas: Vec<DeviceId>,
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
    let resident: BTreeSet<ExpertKey> = handles.keys().copied().collect();
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
    match touch {
        Touch::Hit => Ok(()),
        Touch::Miss { evicted } => {
            if let Some(v) = evicted {
                reclaim_victim(sim, handles, graphs, args, v, next_event)?;
            }
            if args.slots == 0 {
                return Ok(());
            }
            let id = if args.mapped {
                sim.alloc_host_mapped(args.bytes)?
            } else if args.managed {
                let id = sim.alloc_managed(args.bytes)?;
                sim.mem_advise(id, gpu_sim::MemAdvise::SetReadMostly, args.d)?;
                sim.mem_advise(id, gpu_sim::MemAdvise::SetPreferredLocation, args.d)?;
                if args.accessed_by {
                    advise_accessed_by(sim, id)?;
                }
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
            match (args.mapped, args.managed) {
                (true, _) => {}
                (false, true) => {
                    let _p = sim.prefetch(args.d, id, args.s)?;
                }
                (false, false) => {
                    hbm_h2d(sim, args, id)?;
                }
            }
            if args.vmm && args.accessed_by {
                advise_vmm_access(sim, id)?;
            }
            let _prev = handles.insert(
                key,
                PageHandle {
                    id,
                    stream: args.s,
                    device: args.d,
                    replicas: Vec::new(),
                },
            );
            Ok(())
        }
    }
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
    if page_is_mapped(sim, page.id) {
        // cudaFreeHost waits GPU work on this pointer, then the mapping is gone.
        sim.synchronize_stream(page.device, page.stream)?;
        sim.free_host_pinned(page.id)?;
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

fn page_is_managed(sim: &Sim, id: AllocId) -> bool {
    sim.is_managed(id).unwrap_or(false)
}

fn page_is_vmm(sim: &Sim, id: AllocId) -> bool {
    sim.is_vmm(id).unwrap_or(false)
}

pub(crate) fn gemm_keys(
    sim: &mut Sim,
    handles: &BTreeMap<ExpertKey, PageHandle>,
    graphs: &mut GraphBank,
    keys: &[ExpertKey],
    cuda_graphs: bool,
    ctr: &mut ReplayCounters,
    work: Option<StreamId>,
) -> Result<(), Error> {
    let mut by_dev: BTreeMap<(DeviceId, StreamId), Vec<AllocId>> = BTreeMap::new();
    for key in keys {
        let Some(page) = handles.get(key) else {
            continue;
        };
        let stream = work.unwrap_or(page.stream);
        by_dev
            .entry((page.device, stream))
            .or_default()
            .push(page.id);
    }
    for ((d, stream), ids) in by_dev {
        gemm_ids(sim, graphs, d, stream, ids, cuda_graphs, ctr)?;
    }
    Ok(())
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
    d: DeviceId,
    stream: StreamId,
    ids: Vec<AllocId>,
    cuda_graphs: bool,
    ctr: &mut ReplayCounters,
) -> Result<(), Error> {
    if ids.is_empty() {
        return Ok(());
    }
    if let Some(g) = graphs.get(&ids) {
        let _n = sim.launch_graph(g, stream)?;
        ctr.graph_launches = ctr.graph_launches.saturating_add(1);
        return Ok(());
    }
    if cuda_graphs || graphs.build {
        if let Some(g) = capture_expert_graph(sim, graphs, d, stream, &ids)? {
            if ids.len() > 1 {
                ctr.child_graphs = ctr.child_graphs.saturating_add(1);
            }
            let _n = sim.launch_graph(g, stream)?;
            ctr.graph_launches = ctr.graph_launches.saturating_add(1);
            return Ok(());
        }
    }
    for id in ids {
        kernel(sim, d, stream, id)?;
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
        sim.begin_capture(d, stream)?;
        kernel(sim, d, stream, *id)?;
        let src = sim.end_capture()?;
        leaves.push(graphs.bind(sim, origin, key, src)?);
    }
    if ids.len() == 1 {
        return Ok(leaves.first().copied());
    }
    sim.begin_capture(d, stream)?;
    for g in leaves {
        let _n = sim.launch_graph(g, stream)?;
    }
    let src = sim.end_capture()?;
    Ok(Some(graphs.bind(sim, origin, ids.to_vec(), src)?))
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
        let src = sim.create_graph(d, stream)?;
        sim.graph_add_kernel(src, gemm_kind(), &[*id], &[])?;
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
    let _k = sim.kernel(d, gemm_kind(), &[id], &[], s)?;
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
    let id = sim.alloc_managed(fetch.expert_bytes)?;
    sim.mem_advise(id, gpu_sim::MemAdvise::SetReadMostly, fetch.home)?;
    sim.mem_advise(id, gpu_sim::MemAdvise::SetPreferredLocation, fetch.home)?;
    if fetch.accessed_by {
        advise_accessed_by(sim, id)?;
    }
    let _p = sim.prefetch(fetch.home, id, fetch.stream)?;
    if fetch.home != fetch.compute {
        wait_peer(sim, fetch.home, fetch.compute, fetch.stream, next_event)?;
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
