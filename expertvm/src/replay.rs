//! Cache replay: leases, hits, misses, evictions.

use crate::access::{ExpertKey, Trace};
use crate::policy::Policy;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// One row of a policy comparison table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayRow {
    /// Policy.
    pub policy: Policy,
    /// Cache hits.
    pub hits: u64,
    /// Cache misses.
    pub misses: u64,
    /// Evictions.
    pub evicts: u64,
    /// Hits in parts per thousand of acquires.
    pub hits_permille: u32,
}

impl ReplayRow {
    fn finish(mut self, n: u64) -> Self {
        self.hits_permille = match self.hits.saturating_mul(1000).checked_div(n) {
            Some(v) => u32::try_from(v).unwrap_or(u32::MAX),
            None => 0,
        };
        self
    }
}

/// Replay `trace` with a bounded cache of `slots` expert tensors.
#[must_use]
pub fn replay(trace: &Trace, slots: usize, policy: Policy, lookahead: usize) -> ReplayRow {
    let keys = trace.keys();
    replay_keys(&keys, slots, policy, lookahead)
}

/// Compare every [`Policy`] on the same cache size.
#[must_use]
pub fn compare(trace: &Trace, slots: usize, lookahead: usize) -> Vec<ReplayRow> {
    Policy::all()
        .iter()
        .copied()
        .map(|p| replay(trace, slots, p, lookahead))
        .collect()
}

/// Markdown-ish table for CLI and docs. Numbers are measured, not invented.
#[must_use]
pub fn format_table(rows: &[ReplayRow]) -> String {
    let mut out = String::from("policy        hits  misses  evicts  hits‰\n");
    for r in rows {
        out.push_str(&format!(
            "{:<12} {:>5} {:>7} {:>7} {:>6}\n",
            r.policy.name(),
            r.hits,
            r.misses,
            r.evicts,
            r.hits_permille
        ));
    }
    out
}

pub(crate) fn replay_keys(
    keys: &[ExpertKey],
    slots: usize,
    policy: Policy,
    lookahead: usize,
) -> ReplayRow {
    let n = u64::try_from(keys.len()).unwrap_or(u64::MAX);
    let mut w = Walker::new(keys, slots, policy, lookahead);
    let mut hits = 0u64;
    let mut misses = 0u64;
    let mut evicts = 0u64;
    while let Some((_key, touch)) = w.next_touch() {
        match touch {
            Touch::Hit => hits = hits.saturating_add(1),
            Touch::Miss { evicted } => {
                misses = misses.saturating_add(1);
                if evicted.is_some() {
                    evicts = evicts.saturating_add(1);
                }
            }
        }
    }
    ReplayRow {
        policy,
        hits,
        misses,
        evicts,
        hits_permille: 0,
    }
    .finish(n)
}

/// Hit or miss (with optional victim) for one acquire.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Touch {
    Hit,
    Miss { evicted: Option<ExpertKey> },
}

/// Demand-paging walker used by replay and gpu-sim.
pub(crate) struct Walker {
    keys: Vec<ExpertKey>,
    i: usize,
    slots: usize,
    policy: Policy,
    lookahead: usize,
    resident: BTreeSet<ExpertKey>,
    recency: VecDeque<ExpertKey>,
    freq: BTreeMap<ExpertKey, u32>,
    next_use: BTreeMap<ExpertKey, VecDeque<usize>>,
    last: Option<ExpertKey>,
    rng: u64,
}

impl Walker {
    pub(crate) fn new(keys: &[ExpertKey], slots: usize, policy: Policy, lookahead: usize) -> Self {
        let mut next_use: BTreeMap<ExpertKey, VecDeque<usize>> = BTreeMap::new();
        for (i, key) in keys.iter().enumerate() {
            next_use.entry(*key).or_default().push_back(i);
        }
        Self {
            keys: keys.to_vec(),
            i: 0,
            slots,
            policy,
            lookahead,
            resident: BTreeSet::new(),
            recency: VecDeque::new(),
            freq: BTreeMap::new(),
            next_use,
            last: None,
            rng: 0x9e37_79b9_7f4a_7c15,
        }
    }

    /// Demand paging with no future key list (open-loop schedule order).
    ///
    /// Oracle / layer-ahead see only keys already demanded; Belady without a
    /// future use is furthest-next-use = never (`usize::MAX`).
    pub(crate) fn demand(slots: usize, policy: Policy, lookahead: usize) -> Self {
        Self::new(&[], slots, policy, lookahead)
    }

    pub(crate) fn next_touch(&mut self) -> Option<(ExpertKey, Touch)> {
        let key = *self.keys.get(self.i)?;
        if let Some(q) = self.next_use.get_mut(&key) {
            let _used = q.pop_front();
        }
        let c = self.freq.entry(key).or_insert(0);
        *c = c.saturating_add(1);
        let touch = self.fault_in(key);
        self.last = Some(key);
        self.i = self.i.saturating_add(1);
        Some((key, touch))
    }

    /// Demand-fault `key` without a precomputed key list (no future leak).
    pub(crate) fn demand_touch(&mut self, key: ExpertKey) -> Touch {
        self.keys.push(key);
        match self.next_touch() {
            Some((_, touch)) => touch,
            None => Touch::Miss { evicted: None },
        }
    }

    /// Fill `key` without consuming the demand stream (prefetch).
    pub(crate) fn prefetch_touch(&mut self, key: ExpertKey) -> Touch {
        self.fault_in(key)
    }

    /// Drop `key` from this device's resident set without a demand acquire.
    ///
    /// Used when a home eviction frees peer replicas that this walker still
    /// counted as occupying a slot.
    pub(crate) fn forget(&mut self, key: ExpertKey) {
        let _r = self.resident.remove(&key);
        self.recency.retain(|k| *k != key);
    }

    fn fault_in(&mut self, key: ExpertKey) -> Touch {
        if self.resident.contains(&key) {
            self.touch_recency(key);
            Touch::Hit
        } else if self.slots == 0 {
            Touch::Miss { evicted: None }
        } else {
            let evicted = if self.resident.len() >= self.slots {
                let v = self.pick_victim();
                let _removed = self.resident.remove(&v);
                self.recency.retain(|k| *k != v);
                Some(v)
            } else {
                None
            };
            let _inserted = self.resident.insert(key);
            self.recency.push_back(key);
            Touch::Miss { evicted }
        }
    }

    fn touch_recency(&mut self, key: ExpertKey) {
        self.recency.retain(|k| *k != key);
        self.recency.push_back(key);
    }

    fn pick_victim(&mut self) -> ExpertKey {
        match self.policy {
            Policy::Lru => self.recency.front().copied().unwrap_or(self.first()),
            Policy::Lfu => self
                .resident
                .iter()
                .copied()
                .min_by_key(|k| self.freq.get(k).copied().unwrap_or(0))
                .unwrap_or(self.first()),
            Policy::Random => {
                self.rng = self.rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let n = self.resident.len().max(1);
                let n64 = u64::try_from(n).unwrap_or(1);
                let skip = usize::try_from(self.rng % n64).unwrap_or(0);
                self.resident
                    .iter()
                    .copied()
                    .nth(skip)
                    .unwrap_or(self.first())
            }
            Policy::LayerAhead => self.ahead_victim(),
            Policy::Predictor => self.predictor_victim(),
            Policy::Oracle => self.belady_victim(),
        }
    }

    fn ahead_victim(&self) -> ExpertKey {
        let start = self.i.saturating_add(1);
        let end = start.saturating_add(self.lookahead.max(1));
        let window = self.keys.get(start..end).unwrap_or(&[]);
        let upcoming: BTreeSet<ExpertKey> = window.iter().copied().collect();
        self.resident
            .iter()
            .copied()
            .find(|k| !upcoming.contains(k))
            .or_else(|| self.recency.front().copied())
            .unwrap_or(self.first())
    }

    fn predictor_victim(&self) -> ExpertKey {
        let pred = self
            .last
            .map(|k| ExpertKey::new(k.layer.saturating_add(1), k.expert));
        self.resident
            .iter()
            .copied()
            .find(|k| pred != Some(*k))
            .or_else(|| self.recency.front().copied())
            .unwrap_or(self.first())
    }

    fn belady_victim(&self) -> ExpertKey {
        self.resident
            .iter()
            .copied()
            .max_by_key(|k| {
                self.next_use
                    .get(k)
                    .and_then(|q| q.front().copied())
                    .unwrap_or(usize::MAX)
            })
            .unwrap_or(self.first())
    }

    fn first(&self) -> ExpertKey {
        self.resident
            .iter()
            .copied()
            .next()
            .unwrap_or(ExpertKey::new(0, 0))
    }
}
