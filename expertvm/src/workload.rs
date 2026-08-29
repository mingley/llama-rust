//! Adversarial MoE access traces. Timing stays in the hardware profile.

use crate::access::{ExpertAccess, ExpertKey, Trace};
use std::collections::BTreeSet;

/// Named synthetic traces for anti-Goodhart experiments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Workload {
    /// Uniform expert ids, no locality.
    Uniform,
    /// 80% of acquires hit a small hot set.
    Hotset,
    /// Hot set slides every `shift` tokens.
    ShiftingHotset,
    /// Cyclic thrash (Belady's anomaly case).
    Thrash,
}

impl Workload {
    /// Name used in benches and CLI.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Hotset => "hotset",
            Self::ShiftingHotset => "shifting-hotset",
            Self::Thrash => "thrash",
        }
    }
}

/// Build `n_tokens` events, one layer, `top_k` experts from `n_experts`.
#[must_use]
pub fn generate(kind: Workload, n_tokens: u32, n_experts: u32, top_k: u32, seed: u64) -> Trace {
    let mut rng = seed | 1;
    let mut events = Vec::new();
    let k = top_k.max(1);
    let n_ex = n_experts.max(k);
    for tok in 0..n_tokens {
        let experts = match kind {
            Workload::Uniform => pick_uniform(&mut rng, n_ex, k),
            Workload::Hotset => pick_hotset(&mut rng, n_ex, k, 80),
            Workload::ShiftingHotset => {
                let shift = (tok / 16) % n_ex;
                pick_shifted(&mut rng, n_ex, k, shift)
            }
            Workload::Thrash => {
                let e = tok % n_ex.max(1);
                vec![e]
            }
        };
        events.push(ExpertAccess {
            sequence: 0,
            token: tok,
            layer: 0,
            experts,
        });
    }
    Trace { events }
}

fn next_u32(rng: &mut u64) -> u32 {
    *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
    u32::try_from(*rng >> 33).unwrap_or(0)
}

fn pick_uniform(rng: &mut u64, n_ex: u32, k: u32) -> Vec<u32> {
    let mut out = Vec::new();
    for _ in 0..k {
        let e = next_u32(rng) % n_ex.max(1);
        if !out.contains(&e) {
            out.push(e);
        }
    }
    if out.is_empty() {
        out.push(0);
    }
    out
}

fn pick_hotset(rng: &mut u64, n_ex: u32, k: u32, hot_pt: u32) -> Vec<u32> {
    let hot_n = (n_ex / 5).max(1);
    let mut out = Vec::new();
    for _ in 0..k {
        let roll = next_u32(rng) % 100;
        let e = if roll < hot_pt {
            next_u32(rng) % hot_n
        } else {
            hot_n + (next_u32(rng) % (n_ex.saturating_sub(hot_n).max(1)))
        };
        let e = e.min(n_ex.saturating_sub(1));
        if !out.contains(&e) {
            out.push(e);
        }
    }
    if out.is_empty() {
        out.push(0);
    }
    out
}

fn pick_shifted(rng: &mut u64, n_ex: u32, k: u32, shift: u32) -> Vec<u32> {
    let mut out = pick_hotset(rng, n_ex, k, 80);
    for e in &mut out {
        *e = (*e).saturating_add(shift) % n_ex.max(1);
    }
    out
}

/// Distinct keys in `trace`, decode order of first appearance.
#[must_use]
pub fn unique_keys(trace: &Trace) -> Vec<ExpertKey> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for k in trace.keys() {
        if seen.insert(k) {
            out.push(k);
        }
    }
    out
}
