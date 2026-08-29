//! Replay a trace through [`gpu_sim`] with pinned H2D fills on miss.

use crate::access::{ExpertAccess, ExpertKey, Trace};
use crate::error::Error;
use crate::place::PlaceMap;
use crate::planner::{
    copy_forward, observe_chain, plan_placement, plan_window, prefetch_keys_ctx, window_keys,
    Markov, Placement, Plan, Prefetch,
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
            " hits={} misses={} prefetches={} prefetch_hits={} prefetch_waste={} graph_launches={}",
            self.hits,
            self.misses,
            self.prefetches,
            self.prefetch_hits,
            self.prefetch_waste,
            self.graph_launches
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
    /// same page budget on the expert home.
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
    /// resident set launches one graph instead of N kernel launches.
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
    /// streams on that GPU. [`crate::SimulatedGpuStore`] stays on the async path.
    pub sync_alloc: bool,
    /// Hold unused `cudaMallocAsync` bytes in the default mempool (`u64::MAX`
    /// release threshold) until [`gpu_sim::Sim::pool_trim_to`].
    ///
    /// Default `false` matches CUDA's default pool (`threshold = 0`: free
    /// returns HBM when the stream-ordered free completes). Serving engines
    /// raise the threshold so `cudaMalloc` can OOM while the pool still holds
    /// cache. Hits/misses stay the same; reuse pays `pool_reuse_ns`.
    pub mempool: bool,
    /// `cudaHostAllocMapped`: miss pages are mapped host, not HBM. Kernels run
    /// over PCIe with no H2D. Hits/misses follow the same walker; `hbm_peak`
    /// stays near zero. [`crate::SimulatedGpuStore`] stays on the H2D path.
    pub mapped: bool,
    /// `cudaMallocManaged` + `cudaMemPrefetchAsync` on miss. Alloc does not
    /// charge HBM; prefetch migrates the page. Hits/misses match H2D.
    /// [`crate::SimulatedGpuStore`] stays on pinned H2D.
    pub managed: bool,
    /// `va_acquire` on miss (reuse an unmapped VA, else reserve+map), then
    /// pinned H2D. Evict [`gpu_sim::Sim::va_release`]s so the pointer stays.
    /// Hits/misses match H2D.
    pub vmm: bool,
    /// `cudaLaunchHostFunc` after each event's GEMMs (CPU scheduler roundtrip).
    ///
    /// Does not change hits/misses. Lengthens wall by `host_func_ns` per
    /// stream that ran a GEMM. [`crate::SimulatedGpuStore`] does not enqueue it.
    pub host_func: bool,
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
            host_func: false,
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
    let mut sim = Sim::new(profile);
    if cfg.mempool {
        sim.set_default_pool_release_threshold(u64::MAX)?;
    }
    let d = DeviceId(0);
    let s = StreamId(0);
    let mut handles: BTreeMap<ExpertKey, PageHandle> = BTreeMap::new();
    let mut w = Walker::new(&keys, cfg.slots, cfg.policy, cfg.lookahead);
    let bytes = cfg.bytes_per_expert.max(1);
    let n_streams = replay_streams(sim.profile(), cfg.seq_streams);
    let mut args = TouchArgs {
        d,
        s,
        bytes,
        slots: cfg.slots,
        sync_alloc: cfg.sync_alloc,
        mapped: cfg.mapped,
        managed: cfg.managed,
        vmm: cfg.vmm,
    };
    let mut token_ends: Vec<u64> = Vec::new();
    let mut ctr = ReplayCounters::default();
    let mut markov = Markov::new();
    let mut prev: Option<&ExpertAccess> = None;
    let mut prev2: Option<&ExpertAccess> = None;
    let mut prefetched: BTreeSet<ExpertKey> = BTreeSet::new();
    let mut graphs: BTreeMap<Vec<AllocId>, GraphId> = BTreeMap::new();
    let mut admitted: BTreeSet<u64> = BTreeSet::new();
    let mut next_event = 1u32;
    for (i, event) in trace.events.iter().enumerate() {
        args.s = stream_of(event.sequence, n_streams);
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
        )?;
        if cfg.host_func {
            host_callbacks(&mut sim, &handles, &ek)?;
        }
        if should_prefetch(cfg, &handles, trace, i) {
            let predicted = predicted_keys(cfg.prefetch, &markov, prev, &ek);
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
        observe_chain(&mut markov, prev2, prev, event);
        prev2 = prev;
        prev = Some(event);
        let _ins = admitted.insert(event.sequence);
        if engine_step(&trace.events, i, cfg.max_batch, admitted.len()) {
            sim.synchronize()?;
            if last_of_token(&trace.events, i) {
                token_ends.push(sim.clock_ns());
            }
            admitted.clear();
        }
    }
    if token_ends.is_empty() {
        sim.synchronize()?;
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
    /// [`SimCfg::managed`]: `alloc_managed` + [`gpu_sim::Sim::prefetch`].
    pub managed: bool,
    /// [`SimCfg::vmm`]: `va_acquire` + H2D.
    pub vmm: bool,
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
            },
            stream,
        )?;
    } else {
        let _c = sim.memcpy_pinned_to_device(device, alloc, bytes, stream)?;
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
}

pub(crate) struct PageHandle {
    pub(crate) id: AllocId,
    pub(crate) stream: StreamId,
    pub(crate) device: DeviceId,
    /// Extra devices that hold a D2D replica of `id`.
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
    graphs: &mut BTreeMap<Vec<AllocId>, GraphId>,
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
                sim.alloc_managed(args.bytes)?
            } else if args.vmm {
                sim.va_acquire(args.d, args.bytes)?
            } else {
                hbm_alloc(sim, args.d, args.bytes, args.s, args.sync_alloc)?
            };
            match (args.mapped, args.managed) {
                (true, _) => {}
                (false, true) => {
                    let _p = sim.prefetch(args.d, id, args.s)?;
                }
                (false, false) => {
                    hbm_h2d_pinned(sim, args.d, id, args.bytes, args.s, args.sync_alloc)?;
                }
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
    graphs: &mut BTreeMap<Vec<AllocId>, GraphId>,
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
            drop_replica(sim, page, args.d, next_event)
        }
    }
}

fn drop_handle(
    sim: &mut Sim,
    graphs: &mut BTreeMap<Vec<AllocId>, GraphId>,
    page: PageHandle,
    next_event: &mut u32,
    sync: bool,
) -> Result<(), Error> {
    drop_graphs(graphs, page.id);
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
) -> Result<(), Error> {
    if !page.replicas.contains(&dst) {
        return Ok(());
    }
    wait_peer(sim, page.device, dst, page.stream, next_event)?;
    sim.free(dst, page.id, page.stream)?;
    page.replicas.retain(|d| *d != dst);
    Ok(())
}

fn drop_graphs(graphs: &mut BTreeMap<Vec<AllocId>, GraphId>, id: AllocId) {
    graphs.retain(|ids, _| !ids.contains(&id));
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
    graphs: &mut BTreeMap<Vec<AllocId>, GraphId>,
    keys: &[ExpertKey],
    cuda_graphs: bool,
    ctr: &mut ReplayCounters,
) -> Result<(), Error> {
    let mut by_dev: BTreeMap<(DeviceId, StreamId), Vec<AllocId>> = BTreeMap::new();
    for key in keys {
        let Some(page) = handles.get(key) else {
            continue;
        };
        by_dev
            .entry((page.device, page.stream))
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
) -> Result<(), Error> {
    let mut seen: BTreeSet<(DeviceId, StreamId)> = BTreeSet::new();
    for key in keys {
        let Some(page) = handles.get(key) else {
            continue;
        };
        if seen.insert((page.device, page.stream)) {
            let _id = sim.host_func(page.device, page.stream)?;
        }
    }
    Ok(())
}

fn gemm_ids(
    sim: &mut Sim,
    graphs: &mut BTreeMap<Vec<AllocId>, GraphId>,
    d: DeviceId,
    stream: StreamId,
    ids: Vec<AllocId>,
    cuda_graphs: bool,
    ctr: &mut ReplayCounters,
) -> Result<(), Error> {
    if ids.is_empty() {
        return Ok(());
    }
    if let Some(g) = graphs.get(&ids).copied() {
        let _n = sim.launch_graph(g, stream)?;
        ctr.graph_launches = ctr.graph_launches.saturating_add(1);
        return Ok(());
    }
    if cuda_graphs {
        if !sim.stream_is_idle(d, stream)? {
            sim.synchronize_stream(d, stream)?;
        }
        if sim.stream_is_idle(d, stream)? {
            sim.begin_capture(d, stream)?;
            for id in &ids {
                kernel(sim, d, stream, *id)?;
            }
            let g = sim.end_capture()?;
            let _prev = graphs.insert(ids.clone(), g);
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

pub(crate) fn predicted_keys(
    prefetch: Prefetch,
    markov: &Markov,
    prev: Option<&ExpertAccess>,
    ek: &[ExpertKey],
) -> Vec<ExpertKey> {
    match prefetch {
        Prefetch::None => Vec::new(),
        Prefetch::CopyForward => copy_forward(ek),
        Prefetch::Markov => {
            let k = ek.len().max(1);
            match prev {
                Some(p) => markov.predict_ctx(&p.keys(), ek, k),
                None => markov.predict(ek, k),
            }
        }
        Prefetch::Both => match prev {
            Some(p) => prefetch_keys_ctx(markov, &p.keys(), ek),
            None => prefetch_keys_ctx(markov, &[], ek),
        },
    }
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

fn kernel(sim: &mut Sim, d: DeviceId, s: StreamId, id: AllocId) -> Result<(), Error> {
    let _k = sim.kernel(
        d,
        KernelKind::GroupedMoeGemm {
            experts: 1,
            tokens_per_expert: 1,
            hidden: 64,
            ff: 64,
            dtype: DType::Fp16,
        },
        &[id],
        &[id],
        s,
    )?;
    Ok(())
}

/// Decode-shaped activation payload used by [`sim_remote_home`] (hidden=64 fp16).
pub const DECODE_ACTIVATION_BYTES: u64 = 128;

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
/// [`Placement::MoveWeights`]). No GEMM — that is [`remote_hit`] / demand.
pub(crate) fn fill_remote(
    sim: &mut Sim,
    fetch: RemoteFetch,
    reuse: u64,
    fan_in: u64,
    next_event: &mut u32,
) -> Result<RemotePage, Error> {
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
