//! Performance vector. Semantic failure is [`crate::SimError`], not a field here.

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
}

impl Score {
    /// Snapshot after [`Sim::synchronize`].
    #[must_use]
    pub fn from_sim(sim: &Sim) -> Self {
        Self {
            wall_ns: sim.clock_ns(),
            hbm_peak: sim.hbm_peak(),
            bytes_moved: sim.bytes_moved(),
            ns_per_token: None,
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
                "wall_ns={} hbm_peak={} bytes_moved={} ns_per_token={}",
                self.wall_ns, self.hbm_peak, self.bytes_moved, n
            ),
            None => format!(
                "wall_ns={} hbm_peak={} bytes_moved={}",
                self.wall_ns, self.hbm_peak, self.bytes_moved
            ),
        }
    }
}
