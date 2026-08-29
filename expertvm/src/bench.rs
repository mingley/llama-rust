//! CPU vs GPU-sim scorecard for a trace. No invented dollar figures.

use crate::access::Trace;
use crate::error::Error;
use crate::policy::Policy;
use crate::replay::{compare, format_table};
use crate::sim_replay::{sim_replay_cfg, SimCfg};
use crate::workload::{generate, Workload};
use crate::Prefetch;
use gpu_sim::HardwareProfile;

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
    let (sim, overlap) = match profile {
        Some(p) => sim_lines(trace, p, capacity, lookahead, expert_bytes)?,
        None => (None, None),
    };
    Ok(BenchReport {
        name: name.to_string(),
        capacity,
        table,
        sim,
        overlap,
    })
}

fn sim_lines(
    trace: &Trace,
    profile: HardwareProfile,
    capacity: usize,
    lookahead: usize,
    expert_bytes: u64,
) -> Result<(Option<String>, Option<String>), Error> {
    let base = SimCfg {
        slots: capacity,
        policy: Policy::Lru,
        bytes_per_expert: expert_bytes,
        lookahead,
        prefetch: Prefetch::None,
        seq_streams: false,
    };
    let serial = sim_replay_cfg(trace, profile.clone(), base)?;
    let overlap = if trace.n_sequences() > 1 {
        let mut streamed = base;
        streamed.seq_streams = true;
        let ov = sim_replay_cfg(trace, profile, streamed)?;
        Some(format!("serial {} | overlap {}", serial.line(), ov.line()))
    } else {
        None
    };
    Ok((Some(serial.line()), overlap))
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
