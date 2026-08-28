//! Move-weights vs already-resident: a cheap planner over a trace window.

use crate::access::{ExpertKey, Trace};
use std::collections::BTreeSet;

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
