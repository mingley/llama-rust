//! Performance vector. Semantic failure is [`crate::SimError`], not a field here.
//!
//! There is no `$/M tokens` field. Energy is `profile TDP × virtual wall`, not
//! a rental price.

use crate::sim::Sim;

/// Continuous scores. Compare two policies only if both were semantically `Ok`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Score {
    /// Virtual nanoseconds to drain the submitted graph.
    pub wall_ns: u64,
    /// Peak HBM bytes on any GPU.
    pub hbm_peak: u64,
    /// Payload bytes moved by memcpy.
    pub bytes_moved: u64,
    /// `wall_ns / n_tokens` when the caller knows the token count.
    pub ns_per_token: Option<u64>,
    /// Microjoules: `node_tdp_mw * wall_ns / 1_000_000`. Profile TDP, not a bill.
    pub energy_uj: u64,
}

impl Score {
    /// Snapshot after [`Sim::synchronize`].
    #[must_use]
    pub fn from_sim(sim: &Sim) -> Self {
        let wall_ns = sim.clock_ns();
        Self {
            wall_ns,
            hbm_peak: sim.hbm_peak(),
            bytes_moved: sim.bytes_moved(),
            ns_per_token: None,
            energy_uj: energy_uj(sim.profile().node_tdp_mw(), wall_ns),
        }
    }

    /// Attach a per-token rate. Does not invent a dollar figure.
    #[must_use]
    pub fn with_tokens(mut self, n_tokens: u64) -> Self {
        self.ns_per_token = self.wall_ns.checked_div(n_tokens.max(1));
        self
    }

    /// Format as a single line for agent logs.
    #[must_use]
    pub fn line(&self) -> String {
        match self.ns_per_token {
            Some(n) => format!(
                "wall_ns={} hbm_peak={} bytes_moved={} energy_uj={} ns_per_token={}",
                self.wall_ns, self.hbm_peak, self.bytes_moved, self.energy_uj, n
            ),
            None => format!(
                "wall_ns={} hbm_peak={} bytes_moved={} energy_uj={}",
                self.wall_ns, self.hbm_peak, self.bytes_moved, self.energy_uj
            ),
        }
    }
}

/// `tdp_mw * wall_ns / 1e6` microjoules.
fn energy_uj(tdp_mw: u64, wall_ns: u64) -> u64 {
    let n = u128::from(tdp_mw)
        .saturating_mul(u128::from(wall_ns))
        .checked_div(1_000_000)
        .unwrap_or(u128::MAX);
    u64::try_from(n).unwrap_or(u64::MAX)
}
