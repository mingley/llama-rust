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
    /// Copilot-shaped: tiny hot set, ~95% reuse.
    Coding,
    /// Chat-shaped: medium hot set, ~70% reuse.
    Chat,
    /// Long context: expert id walks slowly with token.
    LongContext,
    /// Prefill: each token touches four layers.
    PrefillHeavy,
    /// Decode: one layer, one expert, cycling.
    DecodeHeavy,
    /// Eight interleaved sequences (batch > 1).
    Batch,
}

impl Workload {
    /// Every named workload, in CLI order.
    pub const ALL: [Self; 10] = [
        Self::Uniform,
        Self::Hotset,
        Self::ShiftingHotset,
        Self::Thrash,
        Self::Coding,
        Self::Chat,
        Self::LongContext,
        Self::PrefillHeavy,
        Self::DecodeHeavy,
        Self::Batch,
    ];

    /// Name used in benches and CLI.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Hotset => "hotset",
            Self::ShiftingHotset => "shifting-hotset",
            Self::Thrash => "thrash",
            Self::Coding => "coding",
            Self::Chat => "chat",
            Self::LongContext => "long-context",
            Self::PrefillHeavy => "prefill-heavy",
            Self::DecodeHeavy => "decode-heavy",
            Self::Batch => "batch",
        }
    }

    /// Parse a CLI workload name.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|w| w.name() == name)
    }
}

/// Build `n_tokens` (per sequence) events from `n_experts`.
#[must_use]
pub fn generate(kind: Workload, n_tokens: u32, n_experts: u32, top_k: u32, seed: u64) -> Trace {
    let mut rng = seed | 1;
    let mut events = Vec::new();
    let k = top_k.max(1);
    let n_ex = n_experts.max(k);
    for tok in 0..n_tokens {
        match kind {
            Workload::PrefillHeavy => {
                for layer in 0..4u32 {
                    events.push(ev(0, tok, layer, pick_uniform(&mut rng, n_ex, k)));
                }
            }
            Workload::Batch => {
                for seq in 0..8u64 {
                    events.push(ev(seq, tok, 0, pick_uniform(&mut rng, n_ex, k)));
                }
            }
            Workload::Uniform
            | Workload::Hotset
            | Workload::ShiftingHotset
            | Workload::Thrash
            | Workload::Coding
            | Workload::Chat
            | Workload::LongContext
            | Workload::DecodeHeavy => {
                let experts = match kind {
                    Workload::Uniform => pick_uniform(&mut rng, n_ex, k),
                    Workload::Hotset => pick_hotset(&mut rng, n_ex, k, 80, n_ex / 5),
                    Workload::ShiftingHotset => {
                        let shift = (tok / 16) % n_ex;
                        pick_shifted(&mut rng, n_ex, k, shift)
                    }
                    Workload::Thrash | Workload::DecodeHeavy => vec![tok % n_ex.max(1)],
                    Workload::Coding => pick_hotset(&mut rng, n_ex, k, 95, 2),
                    Workload::Chat => pick_hotset(&mut rng, n_ex, k, 70, (n_ex / 4).max(1)),
                    Workload::LongContext => vec![(tok / 4) % n_ex.max(1)],
                    Workload::PrefillHeavy | Workload::Batch => vec![0],
                };
                events.push(ev(0, tok, 0, experts));
            }
        }
    }
    Trace { events }
}

fn ev(sequence: u64, token: u32, layer: u32, experts: Vec<u32>) -> ExpertAccess {
    ExpertAccess {
        sequence,
        token,
        layer,
        experts,
        weight_pt: Vec::new(),
    }
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

fn pick_hotset(rng: &mut u64, n_ex: u32, k: u32, hot_pt: u32, hot_n: u32) -> Vec<u32> {
    let hot_n = hot_n.max(1).min(n_ex.max(1));
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
    let mut out = pick_hotset(rng, n_ex, k, 80, n_ex / 5);
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
