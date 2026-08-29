//! CPU vs GPU-sim scorecard for a trace. No invented dollar figures.

use crate::access::Trace;
use crate::error::Error;
use crate::replay::{compare, format_table};
use crate::schedule::{schedule_replay, SchedCfg};
use crate::sim_replay::{sim_replay_cfg, SimCfg};
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
    /// Closed-loop `schedule-all` vs `schedule-1` when the trace has >1 sequence.
    pub schedule: Option<String>,
    /// Unchunked vs `--prefill-chunk 1` when a first token has more than one layer.
    pub chunk: Option<String>,
    /// Chunked vs `--decode-first` when a first token has more than one layer
    /// and a later token exists in the trace.
    pub decode: Option<String>,
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
    let mut schedule = None;
    let mut chunk = None;
    let mut decode = None;
    if let Some(p) = profile {
        let lines = sim_lines(trace, p, capacity, lookahead, expert_bytes)?;
        sim = Some(lines.serial);
        overlap = lines.overlap;
        graphs = lines.graphs;
        schedule = lines.schedule;
        chunk = lines.chunk;
        decode = lines.decode;
    }
    Ok(BenchReport {
        name: name.to_string(),
        capacity,
        table,
        sim,
        overlap,
        graphs,
        schedule,
        chunk,
        decode,
    })
}

struct SimLines {
    serial: String,
    overlap: Option<String>,
    graphs: Option<String>,
    schedule: Option<String>,
    chunk: Option<String>,
    decode: Option<String>,
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
    let overlap = if trace.n_sequences() > 1 {
        let mut streamed = base;
        streamed.seq_streams = true;
        let ov = sim_replay_cfg(trace, profile.clone(), streamed)?;
        Some(format!("serial {} | overlap {}", serial.line(), ov.line()))
    } else {
        None
    };
    let lines = schedule_compare(trace, profile.clone(), base)?;
    let mut graphed = base;
    graphed.cuda_graphs = true;
    let g = sim_replay_cfg(trace, profile, graphed)?;
    Ok(SimLines {
        serial: serial.line(),
        overlap,
        graphs: Some(format!("serial {} | graphs {}", serial.line(), g.line())),
        schedule: lines.schedule,
        chunk: lines.chunk,
        decode: lines.decode,
    })
}

struct ScheduleLines {
    schedule: Option<String>,
    chunk: Option<String>,
    decode: Option<String>,
}

fn schedule_compare(
    trace: &Trace,
    profile: HardwareProfile,
    base: SimCfg,
) -> Result<ScheduleLines, Error> {
    if trace.n_sequences() <= 1 {
        return Ok(ScheduleLines {
            schedule: None,
            chunk: None,
            decode: None,
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
                profile,
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
    Ok(ScheduleLines {
        schedule,
        chunk,
        decode,
    })
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
