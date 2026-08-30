//! Open-loop continuous batching over [`ExpertAccess`] traces.

use crate::access::{ExpertAccess, ExpertKey, Trace};
use crate::error::Error;
use crate::place::PlaceMap;
use crate::planner::{plan_keys, predicted_keys, ChainState, Markov, Plan};
use crate::replay::{Touch, Walker};
use crate::sim_replay::{
    advise_pool_access_if_pinned, apply_stream_sms, apply_touch, bind_shareable_mempools,
    drop_remote, fetch_remote, fill_remote, gemm_keys, host_callbacks, note_touch, occupancy_slots,
    reclaim_victim, remote_hit, replay_from_sim, sim_profile, sync_work, validate_sim_cfg,
    GraphBank, LeafMem, PageHandle, RemoteFetch, RemotePage, ReplayCounters, SimCfg, SimReplay,
    StreamPlan, TouchArgs,
};
use gpu_sim::{DeviceId, HardwareProfile, Sim, StreamId};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Write;

/// Continuous-batching knobs. `sim_replay` `max_batch` is closed-loop (one token
/// barrier); this is the open-loop running set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchedCfg {
    /// Max sequences in the running set. `0` = unlimited.
    pub max_batch: usize,
    /// Sequence `s` arrives at `s * interarrival_ns`. `0` = all at t=0.
    pub interarrival_ns: u64,
    /// Optional TTFT SLO. Misses increment [`SchedReplay::ttft_slo_miss`].
    pub ttft_slo_ns: Option<u64>,
    /// Optional ITL SLO. Misses increment [`SchedReplay::itl_slo_miss`].
    pub itl_slo_ns: Option<u64>,
    /// Prefill chunk: max layer-events of a sequence's first token per engine
    /// step. `0` runs the whole token (decode-shaped). `N > 0` lets other
    /// sequences' decode tokens complete while a long prefill is still in flight.
    pub prefill_chunk_layers: usize,
    /// When mixed with in-flight decode, skip prefill sequences this iteration
    /// so decode ITL does not wait on the rest of a long first token.
    pub decode_first: bool,
    /// Drop a waiting sequence instead of admitting it when queue wait already
    /// meets [`Self::ttft_slo_ns`] (Mooncake-style early rejection).
    pub slo_reject: bool,
    /// Skip GPU work for a token whose content-addressed prefix hash already
    /// completed on another sequence. Inserts `p` only after that token
    /// finishes, so in-flight layers of the computing sequence still run.
    pub prefix_cache: bool,
}

impl SchedCfg {
    /// All sequences arrive at t=0. Whole first token per step. `max_batch` `0`
    /// admits everyone.
    #[must_use]
    pub fn closed(max_batch: usize) -> Self {
        Self {
            max_batch,
            interarrival_ns: 0,
            ttft_slo_ns: None,
            itl_slo_ns: None,
            prefill_chunk_layers: 0,
            decode_first: false,
            slo_reject: false,
            prefix_cache: false,
        }
    }

    /// [`Self::closed`] plus a prefill chunk of `layers` events.
    #[must_use]
    pub fn chunked(max_batch: usize, layers: usize) -> Self {
        Self {
            prefill_chunk_layers: layers,
            ..Self::closed(max_batch)
        }
    }
}

/// One scheduled replay. TTFT/ITL are from each sequence's arrival, not t=0.
#[derive(Clone, Debug)]
pub struct SchedReplay {
    /// Hits/misses/wall from the same GPU loop as [`crate::sim_replay()`].
    pub replay: SimReplay,
    /// Sequences that finished every token.
    pub completed: u64,
    /// Waiting sequences dropped by [`SchedCfg::slo_reject`].
    pub rejected: u64,
    /// First-token latency samples that missed [`SchedCfg::ttft_slo_ns`].
    pub ttft_slo_miss: u64,
    /// Later-token gaps that missed [`SchedCfg::itl_slo_ns`].
    pub itl_slo_miss: u64,
    /// Nanoseconds the virtual clock jumped while waiting for arrivals.
    pub idle_ns: u64,
    /// Mean first-token queue wait (`iteration_start - arrival`) when sampled.
    pub queue_ns: Option<u64>,
    /// Layer-events skipped because [`SchedCfg::prefix_cache`] found a completed
    /// content-addressed prefix.
    pub prefix_hits: u64,
}

impl SchedReplay {
    /// Single-line agent / CLI log.
    #[must_use]
    pub fn line(&self) -> String {
        let mut s = self.replay.line();
        let _w = write!(
            s,
            " completed={} rejected={} prefix_hits={} ttft_slo_miss={} itl_slo_miss={} idle_ns={}",
            self.completed,
            self.rejected,
            self.prefix_hits,
            self.ttft_slo_miss,
            self.itl_slo_miss,
            self.idle_ns
        );
        if let Some(n) = self.queue_ns {
            let _w = write!(s, " queue_ns={n}");
        }
        s
    }
}

/// Iteration-level continuous batching: admit FCFS up to `max_batch`, run one
/// next chunk (layer-major across the running set) per engine step, retire
/// finished sequences, idle the GPU until the next arrival when the running
/// set is empty.
///
/// A chunk is the whole next token unless [`SchedCfg::prefill_chunk_layers`]
/// is set: then a sequence's first token advances at most that many
/// layer-events so a short decode in the same batch is not stuck behind a
/// long prefill. [`SchedCfg::decode_first`] further holds prefill while any
/// running sequence is already in decode. Cache order is demand paging (no
/// JSONL future leak). Prefetch Stay vs Fetch uses remaining keys of
/// *running* sequences only. [`SchedCfg::slo_reject`] drops a waiting
/// sequence whose queue wait already meets the TTFT SLO so a later arrival
/// is not stuck behind hopeless FCFS head-of-line work.
/// [`schedule_placed`] H2Ds a miss onto the expert's [`PlaceMap`] home so a
/// wide token can use every GPU's copy engines; [`schedule_replay`] is GPU0.
/// Capacity is per home GPU. [`schedule_remote`] keeps compute on GPU0 and uses [`crate::plan_placement`]
/// on the home hop (move weights vs dispatch activations).
/// [`SchedCfg::prefix_cache`] skips GPU work for a token whose `"p"` hash
/// already completed on another sequence (insert after the computing token
/// finishes; a hit consumes the whole remaining token, not one prefill chunk).
/// Remote prefetch fills home-GPU weight pages without mixing local handles
/// into the remote map.
pub fn schedule_replay(
    trace: &Trace,
    profile: HardwareProfile,
    cfg: SimCfg,
    sched: SchedCfg,
) -> Result<SchedReplay, Error> {
    schedule_placed(trace, profile, cfg, sched, None)
}

/// [`schedule_replay`] with expert-parallel homes. `None` is GPU0 (same as
/// [`schedule_replay`]). A miss copies onto `map.home_of`; GEMM runs there.
/// `--capacity` is slots **on that home**, not a cluster-wide LRU.
pub fn schedule_placed(
    trace: &Trace,
    profile: HardwareProfile,
    cfg: SimCfg,
    sched: SchedCfg,
    place: Option<&PlaceMap>,
) -> Result<SchedReplay, Error> {
    schedule_run(trace, profile, cfg, sched, place.cloned(), None)
}

/// [`schedule_placed`] with compute pinned on GPU0. A miss H2Ds onto the
/// expert home, then [`crate::plan_placement`] either D2Ds weights onto GPU0
/// or ships a small activation payload to home. Hits GEMM where the first
/// fetch left the weights. Demand paging (no JSONL future leak). Prefetch
/// fills remote weight pages only (no GEMM, no local handles mixed in).
pub fn schedule_remote(
    trace: &Trace,
    profile: HardwareProfile,
    cfg: SimCfg,
    sched: SchedCfg,
    map: &PlaceMap,
    act_bytes: u64,
) -> Result<SchedReplay, Error> {
    schedule_run(
        trace,
        profile,
        cfg,
        sched,
        Some(map.clone()),
        Some(act_bytes.max(1)),
    )
}

fn schedule_run(
    trace: &Trace,
    profile: HardwareProfile,
    cfg: SimCfg,
    sched: SchedCfg,
    place: Option<PlaceMap>,
    remote_act: Option<u64>,
) -> Result<SchedReplay, Error> {
    let mut pending = group_jobs(&trace.events, sched.interarrival_ns);
    let mut running: Vec<Job> = Vec::new();
    let mut rt = SchedRt::new(profile, cfg, place, remote_act)?;
    let mut rec = Rec::default();
    let cap = batch_cap(sched.max_batch);
    loop {
        admit(
            &mut pending,
            &mut running,
            cap,
            rt.sim.clock_ns(),
            sched,
            &mut rec,
        );
        if running.is_empty() {
            let Some(next) = pending.front() else {
                break;
            };
            let jumped = rt.sim.idle_until(next.arrival)?;
            rt.idle_ns = rt.idle_ns.saturating_add(jumped);
            continue;
        }
        note_queue(&mut running, rt.sim.clock_ns(), &mut rec);
        let consumed = execute_iteration(&mut rt, &running, sched)?;
        let now = rt.sim.clock_ns();
        retire(
            &mut running,
            &consumed,
            now,
            sched,
            &mut rec,
            &mut rt.prefixes,
        );
    }
    finish_sched(rt, rec)
}

struct Job {
    seq: u64,
    tokens: Vec<(u32, Vec<ExpertAccess>)>,
    arrival: u64,
    first_end: Option<u64>,
    prev_end: Option<u64>,
    queued: bool,
}

#[derive(Default)]
struct Rec {
    ttfts: Vec<u64>,
    itls: Vec<u64>,
    ttft_slo_miss: u64,
    itl_slo_miss: u64,
    completed: u64,
    tokens_done: u64,
    queues: Vec<u64>,
    rejected: u64,
}

struct SchedRt {
    sim: Sim,
    handles: BTreeMap<ExpertKey, PageHandle>,
    graphs: GraphBank,
    args: TouchArgs,
    ctr: ReplayCounters,
    prefetched: BTreeSet<ExpertKey>,
    markov: Markov,
    chain: ChainState,
    cfg: SimCfg,
    idle_ns: u64,
    place: Option<PlaceMap>,
    n_gpus: u16,
    plan: StreamPlan,
    remote_act: Option<u64>,
    remotes: BTreeMap<ExpertKey, RemotePage>,
    seen: BTreeMap<ExpertKey, u64>,
    next_event: u32,
    /// One demand walker per home GPU. `--capacity` is slots on that device,
    /// not a cluster-wide LRU that can evict a peer's resident expert.
    walkers: BTreeMap<DeviceId, Walker>,
    prefixes: BTreeSet<u64>,
    prefix_hits: u64,
}

impl SchedRt {
    fn new(
        profile: HardwareProfile,
        cfg: SimCfg,
        place: Option<PlaceMap>,
        remote_act: Option<u64>,
    ) -> Result<Self, Error> {
        validate_sim_cfg(&cfg, &profile)?;
        let mut sim = Sim::new(sim_profile(profile, &cfg));
        if cfg.shareable {
            let _imported = bind_shareable_mempools(&mut sim)?;
        }
        if cfg.mempool || cfg.shareable {
            sim.set_default_pool_release_threshold(u64::MAX)?;
        }
        advise_pool_access_if_pinned(&mut sim, &cfg)?;
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
        if cfg.l2_persist {
            sim.enable_persisting_l2()?;
        }
        let n_gpus = u16::try_from(sim.profile().n_gpus()).unwrap_or(1).max(1);
        let bytes = cfg.bytes_per_expert.max(1);
        let mut cfg = cfg;
        cfg.slots = occupancy_slots(&cfg, sim.pin_budget());
        Ok(Self {
            walkers: BTreeMap::new(),
            args: TouchArgs {
                d: DeviceId(0),
                s: StreamId(0),
                bytes,
                slots: cfg.slots,
                sync_alloc: cfg.sync_alloc,
                mapped: cfg.mapped,
                managed: cfg.managed,
                vmm: cfg.vmm,
                vmm_page: cfg.vmm_page,
                pageable: cfg.pageable,
                accessed_by: cfg.accessed_by,
            },
            sim,
            handles: BTreeMap::new(),
            graphs: GraphBank::new(
                cfg.graph_update,
                cfg.graph_clone,
                cfg.graph_build,
                LeafMem::from_flags(cfg.graph_mem, cfg.graph_auto_free)?,
            )
            .with_cooperative(cfg.cooperative)
            .with_pdl(cfg.pdl)
            .with_l2_persist(cfg.l2_persist)
            .with_set_params(cfg.graph_set_params)
            .with_piecewise(cfg.graph_piecewise),
            ctr: ReplayCounters::default(),
            prefetched: BTreeSet::new(),
            markov: Markov::new(),
            chain: ChainState::new(),
            cfg,
            idle_ns: 0,
            place,
            n_gpus,
            plan,
            remote_act,
            remotes: BTreeMap::new(),
            seen: BTreeMap::new(),
            next_event: 1,
            prefixes: BTreeSet::new(),
            prefix_hits: 0,
        })
    }

    fn home(&self, key: ExpertKey) -> DeviceId {
        self.place
            .as_ref()
            .map(|m| m.home_of(key, self.n_gpus))
            .unwrap_or(DeviceId(0))
    }

    fn args_for(&self, key: ExpertKey) -> TouchArgs {
        let mut a = self.args;
        a.d = self.home(key);
        a
    }

    fn args_on(&self, device: DeviceId) -> TouchArgs {
        let mut a = self.args;
        a.d = device;
        a
    }

    fn walker_mut(&mut self, device: DeviceId) -> &mut Walker {
        let slots = self.cfg.slots;
        let policy = self.cfg.policy;
        let lookahead = self.cfg.lookahead;
        self.walkers
            .entry(device)
            .or_insert_with(|| Walker::demand(slots, policy, lookahead))
    }

    fn touch_event(&mut self, ev: &ExpertAccess) -> Result<(), Error> {
        if self.remote_act.is_some() {
            self.touch_remote(ev)?;
            if self.cfg.host_func {
                let _id = self.sim.host_func(DeviceId(0), self.args.s)?;
            }
            return Ok(());
        }
        self.args.s = self.plan.work(ev.sequence, ev.token);
        let ek = ev.keys();
        for key in &ek {
            let home = self.home(*key);
            let touch = self.walker_mut(home).demand_touch(*key);
            note_touch(&mut self.ctr, &mut self.prefetched, *key, touch);
            if matches!(touch, Touch::Miss { .. }) {
                self.make_room(home, self.args.bytes)?;
            }
            let args = self.args_for(*key);
            apply_touch(
                &mut self.sim,
                &mut self.handles,
                &mut self.graphs,
                args,
                *key,
                touch,
                &mut self.next_event,
            )?;
            if let Touch::Miss { evicted: Some(v) } = touch {
                self.forget_peer_if_home_dropped(v);
            }
            if matches!(touch, Touch::Miss { .. }) && !self.cfg.sync_alloc {
                self.replicate_key(*key)?;
            }
        }
        gemm_keys(
            &mut self.sim,
            &self.handles,
            &mut self.graphs,
            &ek,
            self.cfg.cuda_graphs,
            &mut self.ctr,
            self.cfg.decode_priority.then_some(self.args.s),
        )?;
        if self.cfg.host_func {
            host_callbacks(
                &mut self.sim,
                &self.handles,
                &ek,
                self.cfg.decode_priority.then_some(self.args.s),
            )?;
        }
        Ok(())
    }

    fn prefetch_event(&mut self, ev: &ExpertAccess, running: &[Job]) -> Result<(), Error> {
        let resident: BTreeSet<ExpertKey> = if self.remote_act.is_some() {
            self.remotes.keys().copied().collect()
        } else {
            self.handles.keys().copied().collect()
        };
        if !want_prefetch(self.cfg, &resident, running) {
            return Ok(());
        }
        if self.remote_act.is_some() {
            return self.prefetch_remote(ev, running);
        }
        let ek = ev.keys();
        let predicted = predicted_keys(
            self.cfg.prefetch,
            &self.markov,
            self.chain.predecessor(ev),
            &ek,
        );
        let planned = if self.cfg.plan_window > 0 {
            remaining_window(running, self.cfg.plan_window)
        } else {
            Vec::new()
        };
        for key in predicted.into_iter().chain(planned) {
            let home = self.home(key);
            match self.walker_mut(home).prefetch_touch(key) {
                Touch::Hit => {}
                miss @ Touch::Miss { .. } => {
                    self.ctr.prefetches = self.ctr.prefetches.saturating_add(1);
                    let _ins = self.prefetched.insert(key);
                    self.make_room(home, self.args.bytes)?;
                    let args = self.args_for(key);
                    apply_touch(
                        &mut self.sim,
                        &mut self.handles,
                        &mut self.graphs,
                        args,
                        key,
                        miss,
                        &mut self.next_event,
                    )?;
                    if let Touch::Miss { evicted: Some(v) } = miss {
                        self.forget_peer_if_home_dropped(v);
                    }
                    if !self.cfg.sync_alloc {
                        self.replicate_key(key)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn prefetch_remote(&mut self, ev: &ExpertAccess, running: &[Job]) -> Result<(), Error> {
        let ek = ev.keys();
        let predicted = predicted_keys(
            self.cfg.prefetch,
            &self.markov,
            self.chain.predecessor(ev),
            &ek,
        );
        let planned = if self.cfg.plan_window > 0 {
            remaining_window(running, self.cfg.plan_window)
        } else {
            Vec::new()
        };
        let compute = DeviceId(0);
        let act = self.remote_act.unwrap_or(1);
        let fan_in = u64::try_from(ev.experts.len()).unwrap_or(1).max(1);
        for key in predicted.into_iter().chain(planned) {
            let home = self.home(key);
            match self.walker_mut(home).prefetch_touch(key) {
                Touch::Hit => {}
                Touch::Miss { evicted } => {
                    self.ctr.prefetches = self.ctr.prefetches.saturating_add(1);
                    let _ins = self.prefetched.insert(key);
                    if let Some(v) = evicted {
                        if let Some(page) = self.remotes.remove(&v) {
                            drop_remote(
                                &mut self.sim,
                                page,
                                compute,
                                self.args.s,
                                self.cfg.sync_alloc,
                            )?;
                        }
                    }
                    if self.args.slots == 0 {
                        continue;
                    }
                    let stream = self.args.s;
                    let bytes = self.args.bytes;
                    self.make_room_remote(home, bytes, compute, stream)?;
                    let reuse = self.seen.get(&key).copied().unwrap_or(1);
                    let page = fill_remote(
                        &mut self.sim,
                        RemoteFetch {
                            home,
                            compute,
                            expert_bytes: bytes,
                            act_bytes: act,
                            stream,
                            sync_alloc: self.cfg.sync_alloc,
                            managed: self.cfg.managed,
                            accessed_by: self.cfg.accessed_by,
                        },
                        reuse,
                        fan_in,
                        &mut self.next_event,
                    )?;
                    let _prev = self.remotes.insert(key, page);
                }
            }
        }
        Ok(())
    }

    fn touch_remote(&mut self, ev: &ExpertAccess) -> Result<(), Error> {
        self.args.s = self.plan.work(ev.sequence, ev.token);
        let ek = ev.keys();
        let fan_in = u64::try_from(ev.experts.len()).unwrap_or(1).max(1);
        let act = self.remote_act.unwrap_or(1);
        let compute = DeviceId(0);
        for key in &ek {
            let n = self.seen.entry(*key).or_insert(0);
            *n = n.saturating_add(1);
            let reuse = *n;
            let home = self.home(*key);
            let touch = self.walker_mut(home).demand_touch(*key);
            note_touch(&mut self.ctr, &mut self.prefetched, *key, touch);
            match touch {
                Touch::Hit => {
                    let page = self
                        .remotes
                        .get(key)
                        .copied()
                        .ok_or(Error::Store("remote hit without page"))?;
                    let page = remote_hit(
                        &mut self.sim,
                        page,
                        compute,
                        act,
                        self.args.s,
                        &mut self.next_event,
                        self.cfg.sync_alloc,
                    )?;
                    let _prev = self.remotes.insert(*key, page);
                }
                Touch::Miss { evicted } => {
                    if let Some(v) = evicted {
                        if let Some(page) = self.remotes.remove(&v) {
                            drop_remote(
                                &mut self.sim,
                                page,
                                compute,
                                self.args.s,
                                self.cfg.sync_alloc,
                            )?;
                        }
                    }
                    if self.args.slots == 0 {
                        continue;
                    }
                    let stream = self.args.s;
                    let bytes = self.args.bytes;
                    self.make_room_remote(home, bytes, compute, stream)?;
                    let page = fetch_remote(
                        &mut self.sim,
                        RemoteFetch {
                            home,
                            compute,
                            expert_bytes: bytes,
                            act_bytes: act,
                            stream,
                            sync_alloc: self.cfg.sync_alloc,
                            managed: self.cfg.managed,
                            accessed_by: self.cfg.accessed_by,
                        },
                        reuse,
                        fan_in,
                        &mut self.next_event,
                    )?;
                    let _prev = self.remotes.insert(*key, page);
                }
            }
        }
        Ok(())
    }

    fn replicate_key(&mut self, key: ExpertKey) -> Result<(), Error> {
        if self.args.mapped {
            return Ok(());
        }
        if self.cfg.accessed_by && (self.args.managed || self.args.vmm || !self.args.sync_alloc) {
            // SetAccessedBy / va_set_access / pool_set_access already maps dest.
            return Ok(());
        }
        let Some(map) = &self.place else {
            return Ok(());
        };
        let Some(dsts) = map.replicas.get(&key).cloned() else {
            return Ok(());
        };
        let Some(page) = self.handles.get(&key) else {
            return Ok(());
        };
        let src = page.device;
        let id = page.id;
        let stream = page.stream;
        let bytes = self.args.bytes;
        let mut nvls = Vec::new();
        for dst in dsts {
            if dst == src {
                continue;
            }
            if self
                .handles
                .get(&key)
                .is_some_and(|p| p.replicas.contains(&dst))
            {
                continue;
            }
            let touch = self.walker_mut(dst).prefetch_touch(key);
            let args = self.args_on(dst);
            if let Touch::Miss { evicted: Some(v) } = touch {
                reclaim_victim(
                    &mut self.sim,
                    &mut self.handles,
                    &mut self.graphs,
                    args,
                    v,
                    &mut self.next_event,
                )?;
                self.forget_peer_if_home_dropped(v);
            }
            self.make_room(dst, bytes)?;
            if self.args.vmm {
                if !self.sim.is_resident(id, dst)? {
                    self.sim.va_map(id, dst)?;
                    if self.cfg.multicast {
                        nvls.push(dst);
                    } else {
                        let _c = self
                            .sim
                            .memcpy_device_to_device(src, dst, id, bytes, stream)?;
                    }
                }
            } else if self.args.managed {
                let _p = self.sim.prefetch(dst, id, stream)?;
            } else {
                let _c = self
                    .sim
                    .memcpy_device_to_device(src, dst, id, bytes, stream)?;
            }
            if let Some(held) = self.handles.get_mut(&key) {
                held.replicas.push(dst);
            }
        }
        if !nvls.is_empty() {
            let _c = self.sim.multicast_store(src, id, &nvls, stream)?;
        }
        Ok(())
    }

    fn forget_peer_if_home_dropped(&mut self, key: ExpertKey) {
        if self.handles.contains_key(&key) {
            return;
        }
        let home = self.home(key);
        let devices: Vec<DeviceId> = self
            .walkers
            .keys()
            .copied()
            .filter(|d| *d != home)
            .collect();
        for d in devices {
            if let Some(w) = self.walkers.get_mut(&d) {
                w.forget(key);
            }
        }
    }

    fn make_room(&mut self, device: DeviceId, need: u64) -> Result<(), Error> {
        if need == 0 {
            return Ok(());
        }
        let total = self.sim.profile().gpu(device)?.hbm_bytes;
        let max_pages = usize::try_from(total / need).unwrap_or(0);
        let mut steps = 0usize;
        let cap = self.cfg.slots.saturating_add(1);
        loop {
            let n = self
                .walkers
                .get(&device)
                .map(Walker::resident_len)
                .unwrap_or(0);
            if n <= max_pages {
                return Ok(());
            }
            let Some(v) = self.walker_mut(device).evict_one() else {
                return Ok(());
            };
            let args = self.args_on(device);
            reclaim_victim(
                &mut self.sim,
                &mut self.handles,
                &mut self.graphs,
                args,
                v,
                &mut self.next_event,
            )?;
            self.forget_peer_if_home_dropped(v);
            steps = steps.saturating_add(1);
            if steps > cap {
                return Ok(());
            }
        }
    }

    fn make_room_remote(
        &mut self,
        device: DeviceId,
        need: u64,
        compute: DeviceId,
        stream: StreamId,
    ) -> Result<(), Error> {
        if need == 0 {
            return Ok(());
        }
        let total = self.sim.profile().gpu(device)?.hbm_bytes;
        let max_pages = usize::try_from(total / need).unwrap_or(0);
        let mut steps = 0usize;
        let cap = self.cfg.slots.saturating_add(1);
        loop {
            let n = self
                .walkers
                .get(&device)
                .map(Walker::resident_len)
                .unwrap_or(0);
            if n <= max_pages {
                return Ok(());
            }
            let Some(v) = self.walker_mut(device).evict_one() else {
                return Ok(());
            };
            if let Some(page) = self.remotes.remove(&v) {
                drop_remote(&mut self.sim, page, compute, stream, self.cfg.sync_alloc)?;
            }
            steps = steps.saturating_add(1);
            if steps > cap {
                return Ok(());
            }
        }
    }

    fn observe(&mut self, ev: &ExpertAccess) {
        self.chain.observe(&mut self.markov, ev);
    }
}

fn group_jobs(events: &[ExpertAccess], interarrival_ns: u64) -> VecDeque<Job> {
    let mut by_seq: BTreeMap<u64, BTreeMap<u32, Vec<ExpertAccess>>> = BTreeMap::new();
    for ev in events {
        by_seq
            .entry(ev.sequence)
            .or_default()
            .entry(ev.token)
            .or_default()
            .push(ev.clone());
    }
    let mut jobs: Vec<Job> = by_seq
        .into_iter()
        .map(|(seq, tokens)| Job {
            seq,
            tokens: tokens.into_iter().collect(),
            arrival: seq.saturating_mul(interarrival_ns),
            first_end: None,
            prev_end: None,
            queued: false,
        })
        .collect();
    jobs.sort_by_key(|j| (j.arrival, j.seq));
    jobs.into()
}

fn batch_cap(max_batch: usize) -> usize {
    if max_batch == 0 {
        usize::MAX
    } else {
        max_batch
    }
}

fn admit(
    pending: &mut VecDeque<Job>,
    running: &mut Vec<Job>,
    cap: usize,
    now: u64,
    sched: SchedCfg,
    rec: &mut Rec,
) {
    while running.len() < cap {
        let Some(front) = pending.front() else {
            break;
        };
        if front.arrival > now {
            break;
        }
        if late_for_ttft_slo(front, now, sched) {
            let _drop = pending.pop_front();
            rec.rejected = rec.rejected.saturating_add(1);
            continue;
        }
        if let Some(job) = pending.pop_front() {
            running.push(job);
        }
    }
}

fn note_queue(running: &mut [Job], now: u64, rec: &mut Rec) {
    for job in running {
        if job.first_end.is_none() && !job.queued {
            rec.queues.push(now.saturating_sub(job.arrival));
            job.queued = true;
        }
    }
}

fn late_for_ttft_slo(job: &Job, now: u64, sched: SchedCfg) -> bool {
    match (sched.slo_reject, sched.ttft_slo_ns) {
        (true, Some(slo)) => now.saturating_sub(job.arrival) >= slo,
        _ => false,
    }
}

fn hold_prefill(running: &[Job], job: &Job, sched: SchedCfg) -> bool {
    sched.decode_first
        && job.first_end.is_none()
        && running
            .iter()
            .any(|j| j.first_end.is_some() && !j.tokens.is_empty())
}

fn chunk_events(job: &Job, chunk: usize) -> Vec<ExpertAccess> {
    let Some((_, evs)) = job.tokens.first() else {
        return Vec::new();
    };
    let prefill = job.first_end.is_none();
    if chunk == 0 || !prefill {
        return evs.clone();
    }
    evs.iter().take(chunk).cloned().collect()
}

fn execute_iteration(
    rt: &mut SchedRt,
    running: &[Job],
    sched: SchedCfg,
) -> Result<Vec<usize>, Error> {
    let mut consumed = Vec::new();
    let mut gpu: Vec<ExpertAccess> = Vec::new();
    for job in running {
        if let Some(evs) = cached_token(job, sched, &rt.prefixes) {
            let n = evs.len();
            for ev in &evs {
                rt.observe(ev);
            }
            rt.prefix_hits = rt.prefix_hits.saturating_add(u64::try_from(n).unwrap_or(0));
            consumed.push(n);
            continue;
        }
        if hold_prefill(running, job, sched) {
            consumed.push(0);
            continue;
        }
        let ch = chunk_events(job, sched.prefill_chunk_layers);
        consumed.push(ch.len());
        gpu.extend(ch);
    }
    let decode_token = gpu.iter().any(|e| e.token > 0);
    let mut by_layer: BTreeMap<u32, Vec<ExpertAccess>> = BTreeMap::new();
    for ev in gpu {
        by_layer.entry(ev.layer).or_default().push(ev);
    }
    for (_layer, batch) in by_layer {
        for ev in &batch {
            rt.touch_event(ev)?;
            rt.prefetch_event(ev, running)?;
            rt.observe(ev);
        }
    }
    sync_work(&mut rt.sim, rt.n_gpus, rt.plan, decode_token)?;
    Ok(consumed)
}

fn token_prefix(evs: &[ExpertAccess]) -> Option<u64> {
    let first = evs.first()?;
    let p = first.prefix?;
    let tok = first.token;
    evs.iter()
        .all(|e| e.token == tok && e.prefix == Some(p))
        .then_some(p)
}

fn cached_token(job: &Job, sched: SchedCfg, prefixes: &BTreeSet<u64>) -> Option<Vec<ExpertAccess>> {
    if !sched.prefix_cache {
        return None;
    }
    let (_, evs) = job.tokens.first()?;
    let p = token_prefix(evs)?;
    prefixes.contains(&p).then(|| evs.clone())
}

fn consume_prefix(job: &mut Job, n: usize) -> (bool, Option<u64>) {
    let Some((_, evs)) = job.tokens.first_mut() else {
        return (false, None);
    };
    let p = token_prefix(evs);
    let n = n.min(evs.len());
    let keep: Vec<ExpertAccess> = evs.iter().skip(n).cloned().collect();
    if keep.is_empty() {
        let _tok = job.tokens.remove(0);
        (true, p)
    } else {
        *evs = keep;
        (false, None)
    }
}

fn retire(
    running: &mut Vec<Job>,
    consumed: &[usize],
    now: u64,
    sched: SchedCfg,
    rec: &mut Rec,
    prefixes: &mut BTreeSet<u64>,
) {
    let mut keep = Vec::new();
    for (i, mut job) in running.drain(..).enumerate() {
        let n = consumed.get(i).copied().unwrap_or(0);
        if n == 0 {
            keep.push(job);
            continue;
        }
        let (finished, prefix) = consume_prefix(&mut job, n);
        if finished {
            rec.tokens_done = rec.tokens_done.saturating_add(1);
            record_latency(&mut job, now, sched, rec);
            if sched.prefix_cache {
                if let Some(p) = prefix {
                    let _ins = prefixes.insert(p);
                }
            }
        }
        if job.tokens.is_empty() {
            rec.completed = rec.completed.saturating_add(1);
        } else {
            keep.push(job);
        }
    }
    *running = keep;
}

fn record_latency(job: &mut Job, now: u64, sched: SchedCfg, rec: &mut Rec) {
    if job.first_end.is_none() {
        job.first_end = Some(now);
        let ttft = now.saturating_sub(job.arrival);
        rec.ttfts.push(ttft);
        if sched.ttft_slo_ns.is_some_and(|slo| ttft > slo) {
            rec.ttft_slo_miss = rec.ttft_slo_miss.saturating_add(1);
        }
    } else if let Some(prev) = job.prev_end {
        let d = now.saturating_sub(prev);
        rec.itls.push(d);
        if sched.itl_slo_ns.is_some_and(|slo| d > slo) {
            rec.itl_slo_miss = rec.itl_slo_miss.saturating_add(1);
        }
    }
    job.prev_end = Some(now);
}

fn want_prefetch(cfg: SimCfg, resident: &BTreeSet<ExpertKey>, running: &[Job]) -> bool {
    if cfg.plan_window == 0 {
        return true;
    }
    let upcoming = remaining_window(running, cfg.plan_window);
    !matches!(
        plan_keys(resident, &upcoming, cfg.plan_threshold),
        Plan::Stay
    )
}

fn remaining_window(running: &[Job], n: usize) -> Vec<ExpertKey> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    let mut taken = 0usize;
    if n == 0 {
        return out;
    }
    for job in running {
        for (_, evs) in &job.tokens {
            for ev in evs {
                if taken >= n {
                    return out;
                }
                taken = taken.saturating_add(1);
                for k in ev.keys() {
                    if seen.insert(k) {
                        out.push(k);
                    }
                }
            }
        }
    }
    out
}

fn mean_u64(xs: &[u64]) -> Option<u64> {
    if xs.is_empty() {
        return None;
    }
    let n = u64::try_from(xs.len()).ok()?;
    let sum = xs.iter().copied().fold(0u64, u64::saturating_add);
    Some(sum / n.max(1))
}

fn finish_sched(mut rt: SchedRt, rec: Rec) -> Result<SchedReplay, Error> {
    rt.sim.synchronize()?;
    rt.ctr.graph_updates = rt.graphs.updates;
    rt.ctr.graph_clones = rt.graphs.clones;
    rt.ctr.graph_set_params = rt.graphs.kernel_sets;
    let replay = replay_from_sim(
        &rt.sim,
        rec.tokens_done,
        mean_u64(&rec.ttfts),
        mean_u64(&rec.itls),
        rt.ctr,
    );
    Ok(SchedReplay {
        replay,
        completed: rec.completed,
        rejected: rec.rejected,
        ttft_slo_miss: rec.ttft_slo_miss,
        itl_slo_miss: rec.itl_slo_miss,
        idle_ns: rt.idle_ns,
        queue_ns: mean_u64(&rec.queues),
        prefix_hits: rt.prefix_hits,
    })
}
