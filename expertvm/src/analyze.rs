//! Locality statistics on a real MoE access trace.

use std::collections::BTreeMap;
use std::fmt::Write;

use crate::access::{ExpertAccess, ExpertKey, Trace};

/// Working-set and predictability summary. All rates are in parts per thousand
/// so the crate stays integer-only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceStats {
    /// Routing events.
    pub n_events: u64,
    /// Expert acquires.
    pub n_acquires: u64,
    /// Distinct (layer, expert) keys.
    pub n_unique: u64,
    /// Fraction of acquires covered by the hottest 20% of keys, ‰.
    pub top20_share_pt: u64,
    /// Next-layer expert is in the previous layer's selected set, ‰.
    pub layer_persist_pt: u64,
    /// Next-token same-layer expert is in the previous token's set, ‰.
    pub seq_persist_pt: u64,
    /// Non-first acquires whose previous use was ≤ 8 acquires ago, ‰.
    pub reuse8_pt: u64,
    /// Smallest key count covering 90% of acquires.
    pub ws90: u64,
    /// Distinct unordered co-activation pairs (same event, top-k ≥ 2).
    pub coact_pairs: u64,
    /// Sum of recorded [`crate::ExpertAccess::weight_pt`] (0 if the trace has no `w`).
    pub mass_pt: u64,
    /// Hottest 20% of keys by router mass, ‰ of [`Self::mass_pt`]. 0 if no `w`.
    pub top20_mass_pt: u64,
}

impl TraceStats {
    /// Human-readable block for CLI / docs.
    #[must_use]
    pub fn report(&self) -> String {
        let mut s = format!(
            "events={} acquires={} unique={} top20‰={} layer_persist‰={} seq_persist‰={} reuse8‰={} ws90={} coact_pairs={}",
            self.n_events,
            self.n_acquires,
            self.n_unique,
            self.top20_share_pt,
            self.layer_persist_pt,
            self.seq_persist_pt,
            self.reuse8_pt,
            self.ws90,
            self.coact_pairs
        );
        if self.mass_pt > 0 {
            let _w = write!(
                s,
                " mass‰={} top20_mass‰={}",
                self.mass_pt, self.top20_mass_pt
            );
        }
        s
    }
}

/// Analyze a trace. Does not invent hit rates; those come from [`crate::replay`].
#[must_use]
pub fn analyze(trace: &Trace) -> TraceStats {
    let n_events = u64::try_from(trace.events.len()).unwrap_or(0);
    let n_acquires = trace.n_acquires();
    let freq = freq_table(trace);
    let n_unique = u64::try_from(freq.len()).unwrap_or(0);
    let mass = mass_table(trace);
    let mass_pt = mass.values().fold(0u64, |a, v| a.saturating_add(*v));
    TraceStats {
        n_events,
        n_acquires,
        n_unique,
        top20_share_pt: top_share_pt(&freq, n_acquires, 200),
        layer_persist_pt: layer_persist_pt(trace),
        seq_persist_pt: seq_persist_pt(trace),
        reuse8_pt: reuse_within_pt(trace, 8),
        ws90: working_set_cover(&freq, n_acquires, 900),
        coact_pairs: u64::try_from(coactivation_counts(trace).len()).unwrap_or(0),
        mass_pt,
        top20_mass_pt: if mass_pt == 0 {
            0
        } else {
            top_share_pt(&mass, mass_pt, 200)
        },
    }
}

/// Acquire counts per key.
#[must_use]
pub fn freq_table(trace: &Trace) -> BTreeMap<ExpertKey, u64> {
    let mut freq: BTreeMap<ExpertKey, u64> = BTreeMap::new();
    for k in trace.keys() {
        let slot = freq.entry(k).or_insert(0);
        *slot = slot.saturating_add(1);
    }
    freq
}

/// Sum of router permille per key. Events with empty `weight_pt` contribute nothing.
#[must_use]
pub fn mass_table(trace: &Trace) -> BTreeMap<ExpertKey, u64> {
    let mut mass: BTreeMap<ExpertKey, u64> = BTreeMap::new();
    for e in &trace.events {
        if e.weight_pt.is_empty() {
            continue;
        }
        for (i, id) in e.experts.iter().enumerate() {
            let w = u64::from(e.weight_pt.get(i).copied().unwrap_or(0));
            let slot = mass.entry(ExpertKey::new(e.layer, *id)).or_insert(0);
            *slot = slot.saturating_add(w);
        }
    }
    mass
}

/// Unordered co-activation pair counts (same routing event).
#[must_use]
pub fn coactivation_counts(trace: &Trace) -> BTreeMap<(ExpertKey, ExpertKey), u64> {
    let mut pairs: BTreeMap<(ExpertKey, ExpertKey), u64> = BTreeMap::new();
    for e in &trace.events {
        let keys = e.keys();
        for (i, a) in keys.iter().enumerate() {
            for b in keys.iter().skip(i.saturating_add(1)) {
                let pair = if a < b { (*a, *b) } else { (*b, *a) };
                let slot = pairs.entry(pair).or_insert(0);
                *slot = slot.saturating_add(1);
            }
        }
    }
    pairs
}

fn top_share_pt(freq: &BTreeMap<ExpertKey, u64>, n_acquires: u64, cover_pt: u64) -> u64 {
    let mut counts: Vec<u64> = freq.values().copied().collect();
    counts.sort_unstable();
    counts.reverse();
    let n = counts.len();
    let hot_n = n
        .saturating_mul(usize::try_from(cover_pt).unwrap_or(0))
        .checked_div(1000)
        .unwrap_or(0)
        .max(1);
    let mut hot = 0u64;
    for c in counts.iter().take(hot_n) {
        hot = hot.saturating_add(*c);
    }
    hot.saturating_mul(1000)
        .checked_div(n_acquires)
        .unwrap_or(0)
}

fn working_set_cover(freq: &BTreeMap<ExpertKey, u64>, n_acquires: u64, cover_pt: u64) -> u64 {
    let mut counts: Vec<u64> = freq.values().copied().collect();
    counts.sort_unstable();
    counts.reverse();
    let need = n_acquires
        .saturating_mul(cover_pt)
        .checked_div(1000)
        .unwrap_or(n_acquires);
    let mut acc = 0u64;
    let mut n = 0u64;
    for c in counts {
        acc = acc.saturating_add(c);
        n = n.saturating_add(1);
        if acc >= need {
            return n;
        }
    }
    n
}

fn layer_persist_pt(trace: &Trace) -> u64 {
    persist_pt(trace, |p, e| {
        p.sequence == e.sequence && p.token == e.token && e.layer == p.layer.saturating_add(1)
    })
}

fn seq_persist_pt(trace: &Trace) -> u64 {
    persist_pt(trace, |p, e| {
        p.sequence == e.sequence && e.token == p.token.saturating_add(1) && e.layer == p.layer
    })
}

fn persist_pt(trace: &Trace, paired: fn(&ExpertAccess, &ExpertAccess) -> bool) -> u64 {
    let mut persist_hit = 0u64;
    let mut persist_n = 0u64;
    let mut prev: Option<&ExpertAccess> = None;
    for e in &trace.events {
        if let Some(p) = prev {
            if paired(p, e) {
                persist_n = persist_n.saturating_add(u64::try_from(e.experts.len()).unwrap_or(0));
                for x in &e.experts {
                    if p.experts.contains(x) {
                        persist_hit = persist_hit.saturating_add(1);
                    }
                }
            }
        }
        prev = Some(e);
    }
    persist_hit
        .saturating_mul(1000)
        .checked_div(persist_n)
        .unwrap_or(0)
}

fn reuse_within_pt(trace: &Trace, window: usize) -> u64 {
    let mut last: BTreeMap<ExpertKey, usize> = BTreeMap::new();
    let mut hit = 0u64;
    let mut n = 0u64;
    for (i, k) in trace.keys().into_iter().enumerate() {
        if let Some(p) = last.get(&k).copied() {
            n = n.saturating_add(1);
            if i.saturating_sub(p) <= window {
                hit = hit.saturating_add(1);
            }
        }
        let _prev = last.insert(k, i);
    }
    hit.saturating_mul(1000).checked_div(n).unwrap_or(0)
}
