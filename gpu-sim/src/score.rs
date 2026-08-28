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
}

impl Score {
    /// Snapshot after [`Sim::synchronize`].
    #[must_use]
    pub fn from_sim(sim: &Sim) -> Self {
        Self {
            wall_ns: sim.clock_ns(),
            hbm_peak: sim.hbm_peak(),
            bytes_moved: sim.bytes_moved(),
        }
    }

    /// Format as a single line for agent logs.
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "wall_ns={} hbm_peak={} bytes_moved={}",
            self.wall_ns, self.hbm_peak, self.bytes_moved
        )
    }
}
