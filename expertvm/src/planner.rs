//! Move-weights vs already-resident: a cheap planner over a trace window.

use crate::access::{ExpertAccess, ExpertKey, Trace};
use gpu_sim::ns_for_bytes;
use std::collections::{BTreeMap, BTreeSet};

/// Online counts for `P(to | from)` and lookback-2 `P(to | from, from_prev)`.
///
/// Order-2 backs off to order-1 when a pair has never been seen. No prompt-class
/// labels (those are not in the JSONL).
#[derive(Clone, Debug, Default)]
pub struct Markov {
    order1: BTreeMap<ExpertKey, BTreeMap<ExpertKey, u64>>,
    order2: BTreeMap<(ExpertKey, ExpertKey), BTreeMap<ExpertKey, u64>>,
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
            let row = self.order1.entry(*a).or_default();
            for b in to {
                let slot = row.entry(*b).or_insert(0);
                *slot = slot.saturating_add(1);
            }
        }
    }

    /// Record `prev2 → prev1 → to` plus the order-1 `prev1 → to` edge.
    pub fn observe_ctx(&mut self, prev2: &[ExpertKey], prev1: &[ExpertKey], to: &[ExpertKey]) {
        self.observe(prev1, to);
        for a in prev2 {
            for b in prev1 {
                let row = self.order2.entry((*a, *b)).or_default();
                for c in to {
                    let slot = row.entry(*c).or_insert(0);
                    *slot = slot.saturating_add(1);
                }
            }
        }
    }

    /// Top-`k` destinations by summed count from `from`. Stable on ties.
    #[must_use]
    pub fn predict(&self, from: &[ExpertKey], k: usize) -> Vec<ExpertKey> {
        rank_scores(self.order1_scores(from), k)
    }

    /// `P(to | prev2, prev1)` with order-1 mixed in as backoff. Falls back to
    /// [`Self::predict`] when the pair has no counts.
    #[must_use]
    pub fn predict_ctx(
        &self,
        prev2: &[ExpertKey],
        prev1: &[ExpertKey],
        k: usize,
    ) -> Vec<ExpertKey> {
        let mut score = self.order2_scores(prev2, prev1);
        if score.is_empty() {
            return self.predict(prev1, k);
        }
        for (key, n) in self.order1_scores(prev1) {
            let slot = score.entry(key).or_insert(0);
            *slot = slot.saturating_add(n);
        }
        rank_scores(score, k)
    }

    fn order1_scores(&self, from: &[ExpertKey]) -> BTreeMap<ExpertKey, u64> {
        let mut score: BTreeMap<ExpertKey, u64> = BTreeMap::new();
        for a in from {
            let Some(row) = self.order1.get(a) else {
                continue;
            };
            for (b, c) in row {
                let slot = score.entry(*b).or_insert(0);
                *slot = slot.saturating_add(*c);
            }
        }
        score
    }

    fn order2_scores(&self, prev2: &[ExpertKey], prev1: &[ExpertKey]) -> BTreeMap<ExpertKey, u64> {
        let mut score: BTreeMap<ExpertKey, u64> = BTreeMap::new();
        for a in prev2 {
            for b in prev1 {
                let Some(row) = self.order2.get(&(*a, *b)) else {
                    continue;
                };
                for (c, n) in row {
                    let slot = score.entry(*c).or_insert(0);
                    *slot = slot.saturating_add(*n);
                }
            }
        }
        score
    }
}

fn rank_scores(score: BTreeMap<ExpertKey, u64>, k: usize) -> Vec<ExpertKey> {
    let mut pairs: Vec<(ExpertKey, u64)> = score.into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    pairs.into_iter().take(k).map(|(key, _)| key).collect()
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

/// Fit order-2 when `prev2 → prev → event` are consecutive transitions; else order-1.
pub fn observe_chain(
    markov: &mut Markov,
    prev2: Option<&ExpertAccess>,
    prev: Option<&ExpertAccess>,
    event: &ExpertAccess,
) {
    let Some(p) = prev else {
        return;
    };
    if !transition_pair(p, event) {
        return;
    }
    let to = event.keys();
    if let Some(p2) = prev2 {
        if transition_pair(p2, p) {
            markov.observe_ctx(&p2.keys(), &p.keys(), &to);
            return;
        }
    }
    markov.observe(&p.keys(), &to);
}

/// Prefetch policy for [`crate::sim_replay::sim_replay_cfg`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prefetch {
    /// Demand paging only.
    None,
    /// Same expert ids one layer ahead.
    CopyForward,
    /// Online [`Markov`] table: lookback-2 `P(to|from, from_prev)` with order-1 backoff.
    Markov,
    /// Copy-forward ∪ Markov (decode's attached-store policy).
    Both,
}

impl Prefetch {
    /// CLI name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::CopyForward => "copy-forward",
            Self::Markov => "markov",
            Self::Both => "both",
        }
    }

    /// Parse a CLI name.
    pub fn parse(name: &str) -> Result<Self, crate::Error> {
        match name {
            "none" => Ok(Self::None),
            "copy-forward" => Ok(Self::CopyForward),
            "markov" => Ok(Self::Markov),
            "both" => Ok(Self::Both),
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
    prefetch_keys_ctx(markov, &[], keys)
}

/// Copy-forward union lookback-2 Markov destinations (order-1 backoff).
#[must_use]
pub fn prefetch_keys_ctx(
    markov: &Markov,
    prev: &[ExpertKey],
    keys: &[ExpertKey],
) -> Vec<ExpertKey> {
    let mut out = copy_forward(keys);
    let k = keys.len().max(1);
    for extra in markov.predict_ctx(prev, keys, k) {
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
