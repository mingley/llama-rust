//! Locality statistics on a real MoE access trace.

use std::collections::BTreeMap;

use crate::access::{ExpertKey, Trace};

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
}

impl TraceStats {
    /// Human-readable block for CLI / docs.
    #[must_use]
    pub fn report(&self) -> String {
        format!(
            "events={} acquires={} unique={} top20‰={} layer_persist‰={}",
            self.n_events,
            self.n_acquires,
            self.n_unique,
            self.top20_share_pt,
            self.layer_persist_pt
        )
    }
}

/// Analyze a trace. Does not invent hit rates; those come from [`crate::replay`].
#[must_use]
pub fn analyze(trace: &Trace) -> TraceStats {
    let n_events = u64::try_from(trace.events.len()).unwrap_or(0);
    let n_acquires = trace.n_acquires();
    let mut freq: BTreeMap<ExpertKey, u64> = BTreeMap::new();
    for e in &trace.events {
        for k in e.keys() {
            let slot = freq.entry(k).or_insert(0);
            *slot = slot.saturating_add(1);
        }
    }
    let n_unique = u64::try_from(freq.len()).unwrap_or(0);
    let mut counts: Vec<u64> = freq.values().copied().collect();
    counts.sort_unstable();
    counts.reverse();
    let hot_n = (counts.len() / 5).max(1);
    let mut hot = 0u64;
    for c in counts.iter().take(hot_n) {
        hot = hot.saturating_add(*c);
    }
    let top20_share_pt = if n_acquires == 0 {
        0
    } else {
        hot.saturating_mul(1000) / n_acquires
    };
    let mut persist_hit = 0u64;
    let mut persist_n = 0u64;
    let mut prev: Option<&crate::access::ExpertAccess> = None;
    for e in &trace.events {
        if let Some(p) = prev {
            if p.sequence == e.sequence
                && p.token == e.token
                && e.layer == p.layer.saturating_add(1)
            {
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
    let layer_persist_pt = if persist_n == 0 {
        0
    } else {
        persist_hit.saturating_mul(1000) / persist_n
    };
    TraceStats {
        n_events,
        n_acquires,
        n_unique,
        top20_share_pt,
        layer_persist_pt,
    }
}
