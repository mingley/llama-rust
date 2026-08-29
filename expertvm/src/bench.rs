//! CPU vs GPU-sim scorecard for a trace. No invented dollar figures.

use crate::access::Trace;
use crate::error::Error;
use crate::place::striped;
use crate::replay::{compare, format_table};
use crate::schedule::{schedule_placed, schedule_remote, schedule_replay, SchedCfg};
use crate::sim_replay::{sim_replay_cfg, SimCfg, DECODE_ACTIVATION_BYTES};
use crate::workload::{generate, Workload};
use gpu_sim::HardwareProfile;
use std::collections::BTreeMap;

/// One measured line: policy table plus simulated LRU cost.
#[derive(Clone, Debug)]
pub struct BenchReport {
    /// Workload or file name.
    pub name: String,
    /// Cache slots.
    pub capacity: usize,
    /// `expertvm replay` table.
    pub table: String,
    /// `expertvm sim` LRU line, if run.
    pub sim: Option<String>,
    /// Serial vs `--seq-streams` when the trace has more than one sequence.
    pub overlap: Option<String>,
    /// Serial vs `--cuda-graphs` on the same LRU config.
    pub graphs: Option<String>,
    /// Stream-ordered `alloc` vs host-sync `malloc` on the same LRU config.
    pub malloc: Option<String>,
    /// Default-pool release vs `u64::MAX` mempool hold on the same LRU config.
    pub mempool: Option<String>,
    /// Pinned H2D vs `cudaHostAllocMapped` zero-copy on the same LRU config.
    pub mapped: Option<String>,
    /// Pinned H2D vs `cudaMallocManaged` + prefetch on the same LRU config.
    pub managed: Option<String>,
    /// Pinned H2D vs `cuMemMap` expert pages on the same LRU config.
    pub vmm: Option<String>,
    /// Pinned H2D vs a `cudaLaunchHostFunc` after each event's GEMMs.
    pub host_func: Option<String>,
    /// `--seq-streams` non-blocking vs `cudaStreamCreate` blocking streams.
    pub blocking_streams: Option<String>,
    /// Closed-loop `schedule-all` vs `schedule-1` when the trace has >1 sequence.
    pub schedule: Option<String>,
    /// Unchunked vs `--prefill-chunk 1` when a first token has more than one layer.
    pub chunk: Option<String>,
    /// Chunked vs `--decode-first` when a first token has more than one layer
    /// and a later token exists in the trace.
    pub decode: Option<String>,
    /// GPU0 vs striped homes when the profile has more than one GPU.
    pub ep: Option<String>,
    /// `schedule-1` vs `--prefix-cache` when the trace has a `"p"` hash and
    /// more than one sequence.
    pub prefix: Option<String>,
}

impl BenchReport {
    /// Multi-line text.
    #[must_use]
    pub fn render(&self) -> String {
        let mut s = format!("# {} capacity={}\n{}", self.name, self.capacity, self.table);
        if let Some(sim) = &self.sim {
            s.push_str(sim);
            s.push('\n');
        }
        if let Some(ov) = &self.overlap {
            s.push_str(ov);
            s.push('\n');
        }
        if let Some(g) = &self.graphs {
            s.push_str(g);
            s.push('\n');
        }
        if let Some(m) = &self.malloc {
            s.push_str(m);
            s.push('\n');
        }
        if let Some(mp) = &self.mempool {
            s.push_str(mp);
            s.push('\n');
        }
        if let Some(md) = &self.mapped {
            s.push_str(md);
            s.push('\n');
        }
        if let Some(um) = &self.managed {
            s.push_str(um);
            s.push('\n');
        }
        if let Some(vmm) = &self.vmm {
            s.push_str(vmm);
            s.push('\n');
        }
        if let Some(hf) = &self.host_func {
            s.push_str(hf);
            s.push('\n');
        }
        if let Some(bs) = &self.blocking_streams {
            s.push_str(bs);
            s.push('\n');
        }
        if let Some(sch) = &self.schedule {
            s.push_str(sch);
            s.push('\n');
        }
        if let Some(ch) = &self.chunk {
            s.push_str(ch);
            s.push('\n');
        }
        if let Some(df) = &self.decode {
            s.push_str(df);
            s.push('\n');
        }
        if let Some(ep) = &self.ep {
            s.push_str(ep);
            s.push('\n');
        }
        if let Some(px) = &self.prefix {
            s.push_str(px);
            s.push('\n');
        }
        s
    }
}

/// Replay `trace` and optionally time LRU on `profile`.
pub fn report(
    name: &str,
    trace: &Trace,
    capacity: usize,
    lookahead: usize,
    profile: Option<HardwareProfile>,
    expert_bytes: u64,
) -> Result<BenchReport, Error> {
    let table = format_table(&compare(trace, capacity, lookahead));
    let mut sim = None;
    let mut overlap = None;
    let mut graphs = None;
    let mut malloc = None;
    let mut mempool = None;
    let mut mapped = None;
    let mut managed = None;
    let mut vmm = None;
    let mut host_func = None;
    let mut blocking_streams = None;
    let mut schedule = None;
    let mut chunk = None;
    let mut decode = None;
    let mut ep = None;
    let mut prefix = None;
    if let Some(p) = profile {
        let lines = sim_lines(trace, p, capacity, lookahead, expert_bytes)?;
        sim = Some(lines.serial);
        overlap = lines.overlap;
        graphs = lines.graphs;
        malloc = lines.malloc;
        mempool = lines.mempool;
        mapped = lines.mapped;
        managed = lines.managed;
        vmm = lines.vmm;
        host_func = lines.host_func;
        blocking_streams = lines.blocking_streams;
        schedule = lines.schedule;
        chunk = lines.chunk;
        decode = lines.decode;
        ep = lines.ep;
        prefix = lines.prefix;
    }
    Ok(BenchReport {
        name: name.to_string(),
        capacity,
        table,
        sim,
        overlap,
        graphs,
        malloc,
        mempool,
        mapped,
        managed,
        vmm,
        host_func,
        blocking_streams,
        schedule,
        chunk,
        decode,
        ep,
        prefix,
    })
}

struct SimLines {
    serial: String,
    overlap: Option<String>,
    graphs: Option<String>,
    malloc: Option<String>,
    mempool: Option<String>,
    mapped: Option<String>,
    managed: Option<String>,
    vmm: Option<String>,
    host_func: Option<String>,
    blocking_streams: Option<String>,
    schedule: Option<String>,
    chunk: Option<String>,
    decode: Option<String>,
    ep: Option<String>,
    prefix: Option<String>,
}

fn sim_lines(
    trace: &Trace,
    profile: HardwareProfile,
    capacity: usize,
    lookahead: usize,
    expert_bytes: u64,
) -> Result<SimLines, Error> {
    let base = SimCfg::lru(capacity, expert_bytes, lookahead);
    let serial = sim_replay_cfg(trace, profile.clone(), base)?;
    let (overlap, blocking_streams) = if trace.n_sequences() > 1 {
        let mut streamed = base;
        streamed.seq_streams = true;
        let ov = sim_replay_cfg(trace, profile.clone(), streamed)?;
        let mut blk = streamed;
        blk.blocking_streams = true;
        let bl = sim_replay_cfg(trace, profile.clone(), blk)?;
        (
            Some(format!("serial {} | overlap {}", serial.line(), ov.line())),
            Some(format!(
                "sim-overlap {} | sim-blockstrm {}",
                ov.line(),
                bl.line()
            )),
        )
    } else {
        (None, None)
    };
    let lines = schedule_compare(trace, profile.clone(), base)?;
    let mut graphed = base;
    graphed.cuda_graphs = true;
    let g = sim_replay_cfg(trace, profile.clone(), graphed)?;
    let mut malloced = base;
    malloced.sync_alloc = true;
    let mal = sim_replay_cfg(trace, profile.clone(), malloced)?;
    let mut pooled = base;
    pooled.mempool = true;
    let mp = sim_replay_cfg(trace, profile.clone(), pooled)?;
    let mut mapped = base;
    mapped.mapped = true;
    let md = sim_replay_cfg(trace, profile.clone(), mapped)?;
    let mut um = base;
    um.managed = true;
    let um_row = sim_replay_cfg(trace, profile.clone(), um)?;
    let mut vmm = base;
    vmm.vmm = true;
    let vmm_row = sim_replay_cfg(trace, profile.clone(), vmm)?;
    let mut hf = base;
    hf.host_func = true;
    let hf_row = sim_replay_cfg(trace, profile, hf)?;
    Ok(SimLines {
        serial: serial.line(),
        overlap,
        graphs: Some(format!("serial {} | graphs {}", serial.line(), g.line())),
        malloc: Some(format!(
            "sim-async {} | sim-malloc {}",
            serial.line(),
            mal.line()
        )),
        mempool: Some(format!(
            "sim-async {} | sim-pool {}",
            serial.line(),
            mp.line()
        )),
        mapped: Some(format!(
            "sim-async {} | sim-mapped {}",
            serial.line(),
            md.line()
        )),
        managed: Some(format!(
            "sim-async {} | sim-managed {}",
            serial.line(),
            um_row.line()
        )),
        vmm: Some(format!(
            "sim-async {} | sim-vmm {}",
            serial.line(),
            vmm_row.line()
        )),
        host_func: Some(format!(
            "sim-async {} | sim-hostfn {}",
            serial.line(),
            hf_row.line()
        )),
        blocking_streams,
        schedule: lines.schedule,
        chunk: lines.chunk,
        decode: lines.decode,
        ep: lines.ep,
        prefix: lines.prefix,
    })
}

struct ScheduleLines {
    schedule: Option<String>,
    chunk: Option<String>,
    decode: Option<String>,
    ep: Option<String>,
    prefix: Option<String>,
}

fn schedule_compare(
    trace: &Trace,
    profile: HardwareProfile,
    base: SimCfg,
) -> Result<ScheduleLines, Error> {
    let ep = ep_line(trace, profile.clone(), base)?;
    if trace.n_sequences() <= 1 {
        return Ok(ScheduleLines {
            schedule: None,
            chunk: None,
            decode: None,
            ep,
            prefix: None,
        });
    }
    let all = schedule_replay(trace, profile.clone(), base, SchedCfg::closed(0))?;
    let one = schedule_replay(trace, profile.clone(), base, SchedCfg::closed(1))?;
    let schedule = Some(format!(
        "schedule-all {} | schedule-1 {}",
        all.line(),
        one.line()
    ));
    let wide = first_token_events(trace) > 1;
    let chunked = if wide {
        Some(schedule_replay(
            trace,
            profile.clone(),
            base,
            SchedCfg::chunked(0, 1),
        )?)
    } else {
        None
    };
    let chunk = chunked.as_ref().map(|ch| {
        format!(
            "schedule-all {} | schedule-chunk1 {}",
            all.line(),
            ch.line()
        )
    });
    let decode = match (has_later_token(trace), chunked.as_ref()) {
        (true, Some(ch)) => {
            let df = schedule_replay(
                trace,
                profile.clone(),
                base,
                SchedCfg {
                    decode_first: true,
                    ..SchedCfg::chunked(0, 1)
                },
            )?;
            Some(format!(
                "schedule-chunk1 {} | schedule-decode-first {}",
                ch.line(),
                df.line()
            ))
        }
        _ => None,
    };
    let prefix = if has_prefix(trace) {
        let cached = schedule_replay(
            trace,
            profile,
            base,
            SchedCfg {
                prefix_cache: true,
                ..SchedCfg::closed(1)
            },
        )?;
        Some(format!(
            "schedule-1 {} | schedule-prefix {}",
            one.line(),
            cached.line()
        ))
    } else {
        None
    };
    Ok(ScheduleLines {
        schedule,
        chunk,
        decode,
        ep,
        prefix,
    })
}

fn has_prefix(trace: &Trace) -> bool {
    trace.events.iter().any(|e| e.prefix.is_some())
}

fn ep_line(trace: &Trace, profile: HardwareProfile, base: SimCfg) -> Result<Option<String>, Error> {
    if profile.n_gpus() <= 1 {
        return Ok(None);
    }
    let n = u16::try_from(profile.n_gpus()).unwrap_or(1).max(1);
    let map = striped(trace, n);
    let gpu0 = schedule_replay(trace, profile.clone(), base, SchedCfg::closed(0))?;
    let placed = schedule_placed(
        trace,
        profile.clone(),
        base,
        SchedCfg::closed(0),
        Some(&map),
    )?;
    let remote = schedule_remote(
        trace,
        profile,
        base,
        SchedCfg::closed(0),
        &map,
        DECODE_ACTIVATION_BYTES,
    )?;
    Ok(Some(format!(
        "schedule-gpu0 {} | schedule-striped {} | schedule-remote {}",
        gpu0.line(),
        placed.line(),
        remote.line()
    )))
}

fn first_token_events(trace: &Trace) -> usize {
    let mut first: BTreeMap<u64, u32> = BTreeMap::new();
    let mut n: BTreeMap<u64, usize> = BTreeMap::new();
    for e in &trace.events {
        match first.get(&e.sequence).copied() {
            None => {
                let _f = first.insert(e.sequence, e.token);
                let _c = n.insert(e.sequence, 1);
            }
            Some(t) if t == e.token => {
                if let Some(c) = n.get_mut(&e.sequence) {
                    *c = c.saturating_add(1);
                }
            }
            Some(_) => {}
        }
    }
    n.values().copied().max().unwrap_or(0)
}

fn has_later_token(trace: &Trace) -> bool {
    let mut first: BTreeMap<u64, u32> = BTreeMap::new();
    for e in &trace.events {
        match first.get(&e.sequence).copied() {
            None => {
                let _f = first.insert(e.sequence, e.token);
            }
            Some(t) if e.token > t => return true,
            Some(_) => {}
        }
    }
    false
}

/// Run every named adversarial workload at `capacity`.
pub fn adversarial_suite(
    n_tokens: u32,
    n_experts: u32,
    capacity: usize,
    profile: HardwareProfile,
) -> Result<Vec<BenchReport>, Error> {
    let mut out = Vec::new();
    for kind in Workload::ALL {
        let trace = generate(kind, n_tokens, n_experts, 1, 1);
        out.push(report(
            kind.name(),
            &trace,
            capacity,
            8,
            Some(profile.clone()),
            4096,
        )?);
    }
    Ok(out)
}

/// Probe every named example topology. Same payload, different meshes.
pub fn topology_suite(bytes: u64) -> Result<Vec<gpu_sim::TopologyProbe>, Error> {
    let mut out = Vec::new();
    for name in HardwareProfile::example_names() {
        let profile = HardwareProfile::by_name(name)?;
        out.push(gpu_sim::probe_topology(profile, bytes)?);
    }
    Ok(out)
}
