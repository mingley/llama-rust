//! Replay a trace through [`gpu_sim`] with H2D fills on miss.

use crate::access::{ExpertAccess, ExpertKey, Trace};
use crate::error::Error;
use crate::place::PlaceMap;
use crate::planner::{copy_forward, transition_pair, Markov, Prefetch};
use crate::policy::Policy;
use crate::replay::{Touch, Walker};
use gpu_sim::{
    AllocId, DType, DeviceId, EventId, HardwareProfile, KernelKind, Score, Sim, StreamId,
};
use std::collections::BTreeMap;
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
            " hits={} misses={} prefetches={}",
            self.hits, self.misses, self.prefetches
        );
        s
    }
}

/// Cache size, policy, expert payload, lookahead, and prefetch mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimCfg {
    /// Resident expert slots on GPU0.
    pub slots: usize,
    /// Victim policy.
    pub policy: Policy,
    /// Bytes per expert H2D.
    pub bytes_per_expert: u64,
    /// Lookahead window for layer-ahead / oracle.
    pub lookahead: usize,
    /// Prefetch before the next router event.
    pub prefetch: Prefetch,
}

/// Replay `trace` on `profile` with a `slots`-entry expert cache.
///
/// Each miss copies `bytes_per_expert` host→device, then a grouped GEMM runs
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
            slots,
            policy,
            bytes_per_expert,
            lookahead,
            prefetch: Prefetch::None,
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
    let d = DeviceId(0);
    let s = StreamId(0);
    let mut handles: BTreeMap<ExpertKey, AllocId> = BTreeMap::new();
    let mut w = Walker::new(&keys, cfg.slots, cfg.policy, cfg.lookahead);
    let bytes = cfg.bytes_per_expert.max(1);
    let args = TouchArgs {
        d,
        s,
        bytes,
        slots: cfg.slots,
        kernel: true,
    };
    let mut token_ends: Vec<u64> = Vec::new();
    let mut hits = 0u64;
    let mut misses = 0u64;
    let mut prefetches = 0u64;
    let mut markov = Markov::new();
    let mut prev: Option<&ExpertAccess> = None;
    for (i, event) in trace.events.iter().enumerate() {
        let ek = event.keys();
        for key in &ek {
            let (got, touch) = w.next_touch().ok_or(Error::Store("short walker"))?;
            if got != *key {
                return Err(Error::Store("walker key mismatch"));
            }
            match touch {
                Touch::Hit => hits = hits.saturating_add(1),
                Touch::Miss { .. } => misses = misses.saturating_add(1),
            }
            apply_touch(&mut sim, &mut handles, args, *key, touch)?;
        }
        let predicted = match cfg.prefetch {
            Prefetch::None => Vec::new(),
            Prefetch::CopyForward => copy_forward(&ek),
            Prefetch::Markov => markov.predict(&ek, ek.len().max(1)),
        };
        let mut fill = args;
        fill.kernel = false;
        for key in predicted {
            match w.prefetch_touch(key) {
                Touch::Hit => {}
                miss @ Touch::Miss { .. } => {
                    prefetches = prefetches.saturating_add(1);
                    apply_touch(&mut sim, &mut handles, fill, key, miss)?;
                }
            }
        }
        if let Some(p) = prev {
            if transition_pair(p, event) {
                markov.observe(&p.keys(), &ek);
            }
        }
        prev = Some(event);
        if last_of_token(&trace.events, i) {
            sim.synchronize()?;
            token_ends.push(sim.clock_ns());
        }
    }
    if token_ends.is_empty() {
        sim.synchronize()?;
    }
    Ok(finish(&sim, &token_ends, hits, misses, prefetches))
}

#[derive(Clone, Copy)]
struct TouchArgs {
    d: DeviceId,
    s: StreamId,
    bytes: u64,
    slots: usize,
    kernel: bool,
}

fn apply_touch(
    sim: &mut Sim,
    handles: &mut BTreeMap<ExpertKey, AllocId>,
    args: TouchArgs,
    key: ExpertKey,
    touch: Touch,
) -> Result<(), Error> {
    match touch {
        Touch::Hit => {
            if !args.kernel {
                return Ok(());
            }
            let id = *handles.get(&key).ok_or(Error::Store("missing handle"))?;
            kernel(sim, args.d, args.s, id)
        }
        Touch::Miss { evicted } => {
            if let Some(v) = evicted {
                if let Some(id) = handles.remove(&v) {
                    sim.free(args.d, id, args.s)?;
                }
            }
            if args.slots == 0 {
                return Ok(());
            }
            let id = sim.alloc(args.d, args.bytes, args.s)?;
            let _c = sim.memcpy_host_to_device(args.d, id, args.bytes, args.s)?;
            if args.kernel {
                kernel(sim, args.d, args.s, id)?;
            }
            let _prev = handles.insert(key, id);
            Ok(())
        }
    }
}

fn finish(sim: &Sim, token_ends: &[u64], hits: u64, misses: u64, prefetches: u64) -> SimReplay {
    let score = serving_score(sim, token_ends);
    SimReplay {
        sim_ns: score.wall_ns,
        bytes_moved: score.bytes_moved,
        hbm_peak: score.hbm_peak,
        energy_uj: score.energy_uj,
        ttft_ns: score.ttft_ns,
        itl_ns: score.itl_ns,
        hits,
        misses,
        prefetches,
    }
}

fn serving_score(sim: &Sim, token_ends: &[u64]) -> Score {
    let n = u64::try_from(token_ends.len()).unwrap_or(0);
    let mut score = Score::from_sim(sim);
    if n > 0 {
        score = score.with_tokens(n);
    }
    let Some(ttft) = token_ends.first().copied() else {
        return score;
    };
    score.with_latencies(ttft, itl_from_ends(token_ends))
}

fn last_of_token(events: &[ExpertAccess], i: usize) -> bool {
    let Some(cur) = events.get(i) else {
        return true;
    };
    match events.get(i.saturating_add(1)) {
        Some(n) => n.sequence != cur.sequence || n.token != cur.token,
        None => true,
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
                let _c = sim.memcpy_host_to_device(d, id, bytes, s)?;
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
    Ok(finish(&sim, &token_ends, hits, misses, 0))
}

/// Compute on GPU0; experts live on `map` homes. Miss: H2D to home, then peer
/// copy (NVLink / PCIe P2P / RDMA) onto GPU0 before the GEMM.
///
/// Homes that are already GPU0 skip the peer hop (local H2D only). Hits GEMM
/// on GPU0 against the replica left by the first fetch.
pub fn sim_remote_home(
    trace: &Trace,
    profile: HardwareProfile,
    bytes_per_expert: u64,
    map: &PlaceMap,
) -> Result<SimReplay, Error> {
    let n_gpus = u16::try_from(profile.n_gpus()).unwrap_or(1).max(1);
    let mut sim = Sim::new(profile);
    let compute = DeviceId(0);
    let s = StreamId(0);
    let mut handles: BTreeMap<ExpertKey, AllocId> = BTreeMap::new();
    let bytes = bytes_per_expert.max(1);
    let mut hits = 0u64;
    let mut misses = 0u64;
    let mut next_event = 1u32;
    let mut token_ends: Vec<u64> = Vec::new();
    for (i, event) in trace.events.iter().enumerate() {
        for key in event.keys() {
            let home = map.home_of(key, n_gpus);
            if let Some(id) = handles.get(&key).copied() {
                hits = hits.saturating_add(1);
                kernel(&mut sim, compute, s, id)?;
            } else {
                misses = misses.saturating_add(1);
                let id = sim.alloc(home, bytes, s)?;
                let _c = sim.memcpy_host_to_device(home, id, bytes, s)?;
                if home != compute {
                    let _c = sim.memcpy_device_to_device(home, compute, id, bytes, s)?;
                    let ev = EventId(next_event);
                    next_event = next_event.saturating_add(1);
                    let _r = sim.record_event(home, ev, s)?;
                    let _w = sim.wait_event(compute, ev, s)?;
                }
                kernel(&mut sim, compute, s, id)?;
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
    Ok(finish(&sim, &token_ends, hits, misses, 0))
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
