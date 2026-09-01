//! Performance vector. Semantic failure is [`crate::SimError`], not a field here.
//!
//! Energy is `profile TDP × virtual wall`, not a rental price. `$/M tokens` is
//! optional: profile `rent_usd_micros_per_hour` × wall / tokens. `0` rent
//! omits dollars (example profiles). TTFT / ITL are optional; the simulator
//! core has no tokens.

use std::fmt::Write;

use crate::sim::Sim;

/// One hour in virtual nanoseconds.
const HOUR_NS: u128 = 3_600_000_000_000;

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
    /// Time to first token, when the caller samples the clock at token 0.
    pub ttft_ns: Option<u64>,
    /// Mean inter-token latency after TTFT (`(wall - ttft) / (n-1)`).
    pub itl_ns: Option<u64>,
    /// Profile list-price rent used for [`Self::usd_micros_per_m_tokens`].
    pub rent_usd_micros_per_hour: u64,
    /// Microdollars per million tokens when rent and a token count are known.
    ///
    /// `rent_usd_micros_per_hour * wall_ns * 1e6 / (hour_ns * n_tokens)`.
    /// Example profiles leave rent at `0`, so this stays `None`.
    pub usd_micros_per_m_tokens: Option<u64>,
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
            ttft_ns: None,
            itl_ns: None,
            rent_usd_micros_per_hour: sim.profile().rent_usd_micros_per_hour,
            usd_micros_per_m_tokens: None,
        }
    }

    /// Attach a per-token rate. Fills `$/M tokens` when the profile has rent.
    #[must_use]
    pub fn with_tokens(mut self, n_tokens: u64) -> Self {
        self.ns_per_token = self.wall_ns.checked_div(n_tokens.max(1));
        self.usd_micros_per_m_tokens =
            usd_micros_per_m_tokens(self.rent_usd_micros_per_hour, self.wall_ns, n_tokens);
        self
    }

    /// Attach serving latencies measured at token boundaries.
    #[must_use]
    pub fn with_latencies(mut self, ttft_ns: u64, itl_ns: Option<u64>) -> Self {
        self.ttft_ns = Some(ttft_ns);
        self.itl_ns = itl_ns;
        self
    }

    /// Format as a single line for agent logs.
    #[must_use]
    pub fn line(&self) -> String {
        let mut s = format!(
            "wall_ns={} hbm_peak={} bytes_moved={} energy_uj={}",
            self.wall_ns, self.hbm_peak, self.bytes_moved, self.energy_uj
        );
        if let Some(n) = self.ns_per_token {
            let _w = write!(s, " ns_per_token={n}");
        }
        if let Some(n) = self.ttft_ns {
            let _w = write!(s, " ttft_ns={n}");
        }
        if let Some(n) = self.itl_ns {
            let _w = write!(s, " itl_ns={n}");
        }
        if let Some(n) = self.usd_micros_per_m_tokens {
            let _w = write!(s, " usd_micros_per_m_tokens={n}");
        }
        s
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

/// `rent * wall_ns * 1e6 / (hour_ns * n_tokens)` microdollars per million tokens.
fn usd_micros_per_m_tokens(
    rent_usd_micros_per_hour: u64,
    wall_ns: u64,
    n_tokens: u64,
) -> Option<u64> {
    if rent_usd_micros_per_hour == 0 || n_tokens == 0 {
        return None;
    }
    let n = u128::from(rent_usd_micros_per_hour)
        .saturating_mul(u128::from(wall_ns))
        .saturating_mul(1_000_000)
        .checked_div(HOUR_NS.saturating_mul(u128::from(n_tokens)))
        .unwrap_or(u128::MAX);
    Some(u64::try_from(n).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::usd_micros_per_m_tokens;

    #[test]
    fn two_dollar_hour_one_second_thousand_tokens() {
        // $2.00/hr, 1 s wall, 1000 tokens → $2 / 3.6 M tokens/hr = $0.555…/M
        assert_eq!(
            usd_micros_per_m_tokens(2_000_000, 1_000_000_000, 1000),
            Some(555_555)
        );
        assert_eq!(usd_micros_per_m_tokens(0, 1_000_000_000, 1000), None);
        assert_eq!(usd_micros_per_m_tokens(2_000_000, 1_000_000_000, 0), None);
        let twice_wall = usd_micros_per_m_tokens(2_000_000, 2_000_000_000, 1000);
        let half_tokens = usd_micros_per_m_tokens(2_000_000, 1_000_000_000, 500);
        assert_eq!(twice_wall, half_tokens);
        assert!(twice_wall.unwrap() > 555_555);
    }
}
