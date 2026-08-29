//! CPU vs GPU-sim scorecard for a trace. No invented dollar figures.

use crate::access::Trace;
use crate::error::Error;
use crate::policy::Policy;
use crate::replay::{compare, format_table};
use crate::sim_replay::sim_replay;
use crate::workload::{generate, Workload};
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
}

impl BenchReport {
    /// Multi-line text.
    #[must_use]
    pub fn render(&self) -> String {
        match &self.sim {
            Some(s) => format!(
                "# {} capacity={}\n{}{}\n",
                self.name, self.capacity, self.table, s
            ),
            None => format!("# {} capacity={}\n{}", self.name, self.capacity, self.table),
        }
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
    let sim = match profile {
        Some(p) => {
            let row = sim_replay(trace, p, capacity, Policy::Lru, expert_bytes, lookahead)?;
            Some(row.line())
        }
        None => None,
    };
    Ok(BenchReport {
        name: name.to_string(),
        capacity,
        table,
        sim,
    })
}

/// Run the four adversarial workloads at `capacity`.
pub fn adversarial_suite(
    n_tokens: u32,
    n_experts: u32,
    capacity: usize,
    profile: HardwareProfile,
) -> Result<Vec<BenchReport>, Error> {
    let kinds = [
        Workload::Uniform,
        Workload::Hotset,
        Workload::ShiftingHotset,
        Workload::Thrash,
    ];
    let mut out = Vec::new();
    for kind in kinds {
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
