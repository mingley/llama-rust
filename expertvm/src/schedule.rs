//! Open-loop continuous batching over [`ExpertAccess`] traces.

use crate::access::{ExpertAccess, ExpertKey, Trace};
use crate::error::Error;
use crate::planner::{observe_chain, plan_keys, Markov, Plan};
use crate::replay::{Touch, Walker};
use crate::sim_replay::{
    apply_touch, gemm_keys, note_touch, predicted_keys, replay_from_sim, replay_streams, stream_of,
    PageHandle, ReplayCounters, SimCfg, SimReplay, TouchArgs,
};
use gpu_sim::{AllocId, DeviceId, GraphId, HardwareProfile, Sim, StreamId};
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
}

impl SchedReplay {
    /// Single-line agent / CLI log.
    #[must_use]
    pub fn line(&self) -> String {
        let mut s = self.replay.line();
        let _w = write!(
            s,
            " completed={} rejected={} ttft_slo_miss={} itl_slo_miss={} idle_ns={}",
            self.completed, self.rejected, self.ttft_slo_miss, self.itl_slo_miss, self.idle_ns
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
pub fn schedule_replay(
    trace: &Trace,
    profile: HardwareProfile,
    cfg: SimCfg,
    sched: SchedCfg,
) -> Result<SchedReplay, Error> {
    let mut pending = group_jobs(&trace.events, sched.interarrival_ns);
    let mut running: Vec<Job> = Vec::new();
    let mut rt = SchedRt::new(profile, cfg)?;
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
        retire(&mut running, &consumed, rt.sim.clock_ns(), sched, &mut rec);
    }
    Ok(finish_sched(rt, rec))
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
    graphs: BTreeMap<Vec<AllocId>, GraphId>,
    args: TouchArgs,
    ctr: ReplayCounters,
    prefetched: BTreeSet<ExpertKey>,
    markov: Markov,
    prev: Option<ExpertAccess>,
    prev2: Option<ExpertAccess>,
    walker: Walker,
    n_streams: u8,
    cfg: SimCfg,
    idle_ns: u64,
}

impl SchedRt {
    fn new(profile: HardwareProfile, cfg: SimCfg) -> Result<Self, Error> {
        let sim = Sim::new(profile);
        let n_streams = replay_streams(sim.profile(), cfg.seq_streams);
        let bytes = cfg.bytes_per_expert.max(1);
        Ok(Self {
            n_streams,
            walker: Walker::demand(cfg.slots, cfg.policy, cfg.lookahead),
            args: TouchArgs {
                d: DeviceId(0),
                s: StreamId(0),
                bytes,
                slots: cfg.slots,
            },
            sim,
            handles: BTreeMap::new(),
            graphs: BTreeMap::new(),
            ctr: ReplayCounters::default(),
            prefetched: BTreeSet::new(),
            markov: Markov::new(),
            prev: None,
            prev2: None,
            cfg,
            idle_ns: 0,
        })
    }

    fn touch_event(&mut self, ev: &ExpertAccess) -> Result<(), Error> {
        self.args.s = stream_of(ev.sequence, self.n_streams);
        let ek = ev.keys();
        for key in &ek {
            let touch = self.walker.demand_touch(*key);
            note_touch(&mut self.ctr, &mut self.prefetched, *key, touch);
            apply_touch(
                &mut self.sim,
                &mut self.handles,
                &mut self.graphs,
                self.args,
                *key,
                touch,
            )?;
        }
        gemm_keys(
            &mut self.sim,
            &self.handles,
            &mut self.graphs,
            self.args.d,
            &ek,
            self.cfg.cuda_graphs,
            &mut self.ctr,
        )
    }

    fn prefetch_event(&mut self, ev: &ExpertAccess, running: &[Job]) -> Result<(), Error> {
        if !want_prefetch(self.cfg, &self.handles, running) {
            return Ok(());
        }
        let ek = ev.keys();
        let predicted = predicted_keys(self.cfg.prefetch, &self.markov, self.prev.as_ref(), &ek);
        let planned = if self.cfg.plan_window > 0 {
            remaining_window(running, self.cfg.plan_window)
        } else {
            Vec::new()
        };
        let fill = self.args;
        for key in predicted.into_iter().chain(planned) {
            match self.walker.prefetch_touch(key) {
                Touch::Hit => {}
                miss @ Touch::Miss { .. } => {
                    self.ctr.prefetches = self.ctr.prefetches.saturating_add(1);
                    let _ins = self.prefetched.insert(key);
                    apply_touch(
                        &mut self.sim,
                        &mut self.handles,
                        &mut self.graphs,
                        fill,
                        key,
                        miss,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn observe(&mut self, ev: &ExpertAccess) {
        observe_chain(
            &mut self.markov,
            self.prev2.as_ref(),
            self.prev.as_ref(),
            ev,
        );
        self.prev2 = self.prev.clone();
        self.prev = Some(ev.clone());
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
    let chunks: Vec<Vec<ExpertAccess>> = running
        .iter()
        .map(|j| {
            if hold_prefill(running, j, sched) {
                Vec::new()
            } else {
                chunk_events(j, sched.prefill_chunk_layers)
            }
        })
        .collect();
    let consumed: Vec<usize> = chunks.iter().map(Vec::len).collect();
    let mut by_layer: BTreeMap<u32, Vec<ExpertAccess>> = BTreeMap::new();
    for ch in &chunks {
        for ev in ch {
            by_layer.entry(ev.layer).or_default().push(ev.clone());
        }
    }
    for (_layer, batch) in by_layer {
        for ev in &batch {
            rt.touch_event(ev)?;
            rt.prefetch_event(ev, running)?;
            rt.observe(ev);
        }
    }
    rt.sim.synchronize()?;
    Ok(consumed)
}

fn consume_prefix(job: &mut Job, n: usize) -> bool {
    let Some((_, evs)) = job.tokens.first_mut() else {
        return false;
    };
    let n = n.min(evs.len());
    let keep: Vec<ExpertAccess> = evs.iter().skip(n).cloned().collect();
    if keep.is_empty() {
        let _tok = job.tokens.remove(0);
        true
    } else {
        *evs = keep;
        false
    }
}

fn retire(running: &mut Vec<Job>, consumed: &[usize], now: u64, sched: SchedCfg, rec: &mut Rec) {
    let mut keep = Vec::new();
    for (i, mut job) in running.drain(..).enumerate() {
        let n = consumed.get(i).copied().unwrap_or(0);
        if n == 0 {
            keep.push(job);
            continue;
        }
        if consume_prefix(&mut job, n) {
            rec.tokens_done = rec.tokens_done.saturating_add(1);
            record_latency(&mut job, now, sched, rec);
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

fn want_prefetch(cfg: SimCfg, handles: &BTreeMap<ExpertKey, PageHandle>, running: &[Job]) -> bool {
    if cfg.plan_window == 0 {
        return true;
    }
    let resident: BTreeSet<ExpertKey> = handles.keys().copied().collect();
    let upcoming = remaining_window(running, cfg.plan_window);
    !matches!(
        plan_keys(&resident, &upcoming, cfg.plan_threshold),
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

fn finish_sched(rt: SchedRt, rec: Rec) -> SchedReplay {
    let replay = replay_from_sim(
        &rt.sim,
        rec.tokens_done,
        mean_u64(&rec.ttfts),
        mean_u64(&rec.itls),
        rt.ctr,
    );
    SchedReplay {
        replay,
        completed: rec.completed,
        rejected: rec.rejected,
        ttft_slo_miss: rec.ttft_slo_miss,
        itl_slo_miss: rec.itl_slo_miss,
        idle_ns: rt.idle_ns,
        queue_ns: mean_u64(&rec.queues),
    }
}
