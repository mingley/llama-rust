//! Move-weights vs already-resident: a cheap planner over a trace window.

use crate::access::{ExpertAccess, ExpertKey, Trace};
use gpu_sim::ns_for_bytes;
use std::collections::{BTreeMap, BTreeSet};

/// Online counts for `P(to | from)` over paired routing events.
#[derive(Clone, Debug, Default)]
pub struct Markov {
    counts: BTreeMap<ExpertKey, BTreeMap<ExpertKey, u64>>,
}

impl Markov {
    /// Empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add every from×to pair (same token next layer, or next token same layer).
    pub fn observe(&mut self, from: &[ExpertKey], to: &[ExpertKey]) {
        for a in from {
            let row = self.counts.entry(*a).or_default();
            for b in to {
                let slot = row.entry(*b).or_insert(0);
                *slot = slot.saturating_add(1);
            }
        }
    }

    /// Top-`k` destinations by summed count from `from`. Stable on ties.
    #[must_use]
    pub fn predict(&self, from: &[ExpertKey], k: usize) -> Vec<ExpertKey> {
        let mut score: BTreeMap<ExpertKey, u64> = BTreeMap::new();
        for a in from {
            let Some(row) = self.counts.get(a) else {
                continue;
            };
            for (b, c) in row {
                let slot = score.entry(*b).or_insert(0);
                *slot = slot.saturating_add(*c);
            }
        }
        let mut pairs: Vec<(ExpertKey, u64)> = score.into_iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        pairs.into_iter().take(k).map(|(key, _)| key).collect()
    }
}

/// Same sequence, and either next layer this token or next token this layer.
#[must_use]
pub fn transition_pair(prev: &ExpertAccess, next: &ExpertAccess) -> bool {
    if prev.sequence != next.sequence {
        return false;
    }
    let next_layer = prev.token == next.token && next.layer == prev.layer.saturating_add(1);
    let next_tok = next.token == prev.token.saturating_add(1) && next.layer == prev.layer;
    next_layer || next_tok
}

/// Prefetch policy for [`crate::sim_replay::sim_replay_cfg`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prefetch {
    /// Demand paging only.
    None,
    /// Same expert ids one layer ahead.
    CopyForward,
    /// Online [`Markov`] table fitted on observed pairs only (no future leak).
    Markov,
}

impl Prefetch {
    /// CLI name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::CopyForward => "copy-forward",
            Self::Markov => "markov",
        }
    }

    /// Parse a CLI name.
    pub fn parse(name: &str) -> Result<Self, crate::Error> {
        match name {
            "none" => Ok(Self::None),
            "copy-forward" => Ok(Self::CopyForward),
            "markov" => Ok(Self::Markov),
            _ => Err(crate::Error::Trace("unknown prefetch")),
        }
    }
}

/// Move the expert weights, or ship activations to a resident copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Copy expert weights onto the compute device (pay `expert_bytes` once).
    MoveWeights,
    /// Leave the expert where it lives; ship activations each use.
    DispatchActivations,
}

/// Volume-crossover: move once vs dispatch `reuse * fan_in` activation payloads.
///
/// `link_bps` is the hop that would carry either transfer (PCIe, NVLink, RDMA).
/// Equal bandwidth means this is a byte-volume compare; different hops are the
/// caller's job (pass that hop's `bps`). Not a dollar figure.
#[must_use]
pub fn plan_placement(
    expert_bytes: u64,
    activation_bytes: u64,
    fan_in: u64,
    reuse: u64,
    link_bps: u64,
) -> Placement {
    let move_ns = ns_for_bytes(expert_bytes, link_bps);
    let dispatch_bytes = activation_bytes
        .saturating_mul(fan_in.max(1))
        .saturating_mul(reuse.max(1));
    let dispatch_ns = ns_for_bytes(dispatch_bytes, link_bps);
    if move_ns <= dispatch_ns {
        Placement::MoveWeights
    } else {
        Placement::DispatchActivations
    }
}

/// Decision for one window of future expert uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plan {
    /// Keep current residents; next uses are already on device.
    Stay,
    /// Fetch the listed keys (move weights).
    Fetch,
}

/// If more than `threshold_permille` of the next `window` unique keys are
/// already in `resident`, stay; otherwise fetch.
#[must_use]
pub fn plan_window(
    resident: &BTreeSet<ExpertKey>,
    trace: &Trace,
    from: usize,
    window: usize,
    threshold_permille: u32,
) -> Plan {
    let end = from.saturating_add(window);
    let slice = match trace.events.get(from..end) {
        Some(s) => s,
        None => return Plan::Stay,
    };
    if slice.is_empty() {
        return Plan::Stay;
    }
    let mut upcoming = BTreeSet::new();
    for a in slice {
        for k in a.keys() {
            let _ins = upcoming.insert(k);
        }
    }
    if upcoming.is_empty() {
        return Plan::Stay;
    }
    let hits = upcoming.iter().filter(|k| resident.contains(k)).count();
    let n = upcoming.len();
    let n64 = u64::try_from(n).unwrap_or(1);
    let hits64 = u64::try_from(hits).unwrap_or(0);
    let permille = hits64
        .saturating_mul(1000)
        .checked_div(n64.max(1))
        .unwrap_or(0);
    if permille >= u64::from(threshold_permille) {
        Plan::Stay
    } else {
        Plan::Fetch
    }
}

/// Unique keys in `trace[from .. from+window]`.
#[must_use]
pub fn window_keys(trace: &Trace, from: usize, window: usize) -> Vec<ExpertKey> {
    let end = from.saturating_add(window);
    let slice = match trace.events.get(from..end) {
        Some(s) => s,
        None => return Vec::new(),
    };
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for a in slice {
        for k in a.keys() {
            if seen.insert(k) {
                out.push(k);
            }
        }
    }
    out
}

/// Copy-forward: same expert ids one layer ahead (predictor prefetch).
#[must_use]
pub fn copy_forward(keys: &[ExpertKey]) -> Vec<ExpertKey> {
    let mut out = Vec::with_capacity(keys.len());
    for k in keys {
        out.push(ExpertKey::new(k.layer.saturating_add(1), k.expert));
    }
    out
}

/// Copy-forward union online [`Markov`] destinations (no future leak).
#[must_use]
pub fn prefetch_keys(markov: &Markov, keys: &[ExpertKey]) -> Vec<ExpertKey> {
    let mut out = copy_forward(keys);
    let k = keys.len().max(1);
    for extra in markov.predict(keys, k) {
        if !out.contains(&extra) {
            out.push(extra);
        }
    }
    out
}

/// Hottest `n` keys by acquire count (stable on ties).
#[must_use]
pub fn hot_keys(trace: &Trace, n: usize) -> Vec<ExpertKey> {
    let mut freq: BTreeMap<ExpertKey, u64> = BTreeMap::new();
    for k in trace.keys() {
        let slot = freq.entry(k).or_insert(0);
        *slot = slot.saturating_add(1);
    }
    let mut pairs: Vec<(ExpertKey, u64)> = freq.into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    pairs.into_iter().take(n).map(|(k, _)| k).collect()
}
