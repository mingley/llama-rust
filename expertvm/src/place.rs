//! Traffic-aware expert homes and hot replicas.

use crate::access::{ExpertKey, Trace};
use crate::analyze::{coactivation_counts, freq_table};
use gpu_sim::DeviceId;
use std::collections::BTreeMap;

/// Static expert-parallel home: `expert_id % n_gpus`. No eviction, no migration.
#[must_use]
pub fn home_gpu(key: ExpertKey, n_gpus: u16) -> DeviceId {
    let n = u32::from(n_gpus.max(1));
    let idx = key.expert.checked_rem(n).unwrap_or(0);
    DeviceId(u16::try_from(idx).unwrap_or(0))
}

/// Static homes plus optional extra GPU copies.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlaceMap {
    /// Compute home for each key.
    pub home: BTreeMap<ExpertKey, DeviceId>,
    /// Extra residencies (not the home).
    pub replicas: BTreeMap<ExpertKey, Vec<DeviceId>>,
}

impl PlaceMap {
    /// Compact log: `L0E1->gpu0(+gpu1) …`.
    #[must_use]
    pub fn line(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        for (k, d) in &self.home {
            let extra = self
                .replicas
                .get(k)
                .map(|v| {
                    let mut s = String::new();
                    for r in v {
                        s.push_str(&format!("+{r}"));
                    }
                    s
                })
                .unwrap_or_default();
            parts.push(format!("{k}->{d}{extra}"));
        }
        parts.join(" ")
    }

    /// Home, or striped fallback.
    #[must_use]
    pub fn home_of(&self, key: ExpertKey, n_gpus: u16) -> DeviceId {
        self.home
            .get(&key)
            .copied()
            .unwrap_or_else(|| home_gpu(key, n_gpus))
    }
}

/// `expert_id % n_gpus` for every key in `trace`.
#[must_use]
pub fn striped(trace: &Trace, n_gpus: u16) -> PlaceMap {
    let mut home = BTreeMap::new();
    for k in trace.keys() {
        let _p = home.insert(k, home_gpu(k, n_gpus));
    }
    PlaceMap {
        home,
        replicas: BTreeMap::new(),
    }
}

/// Colocate co-activated experts on the same GPU; leftover keys fill min-load GPUs.
#[must_use]
pub fn colocated(trace: &Trace, n_gpus: u16) -> PlaceMap {
    let n = n_gpus.max(1);
    let freq = freq_table(trace);
    let mut assigned: BTreeMap<ExpertKey, DeviceId> = BTreeMap::new();
    let mut load: BTreeMap<DeviceId, u64> = BTreeMap::new();
    for i in 0..n {
        let _p = load.insert(DeviceId(i), 0);
    }
    let mut pairs: Vec<((ExpertKey, ExpertKey), u64)> =
        coactivation_counts(trace).into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for ((a, b), _) in pairs {
        place_pair(&mut assigned, &mut load, &freq, n, a, b);
    }
    let mut rest: Vec<(ExpertKey, u64)> = freq.into_iter().collect();
    rest.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (k, f) in rest {
        if assigned.contains_key(&k) {
            continue;
        }
        let d = min_load(&load, n);
        add_load(&mut assigned, &mut load, k, d, f);
    }
    PlaceMap {
        home: assigned,
        replicas: BTreeMap::new(),
    }
}

/// Copy hot keys (share ≥ `hot_pt` ‰ of acquires) onto the next GPU.
#[must_use]
pub fn with_hot_replicas(mut map: PlaceMap, trace: &Trace, n_gpus: u16, hot_pt: u32) -> PlaceMap {
    let n = n_gpus.max(1);
    if n < 2 {
        return map;
    }
    let freq = freq_table(trace);
    let total = trace.n_acquires().max(1);
    let thresh = u64::from(hot_pt);
    for (k, f) in freq {
        let share = f.saturating_mul(1000).checked_div(total).unwrap_or(0);
        if share < thresh {
            continue;
        }
        let home = map.home_of(k, n);
        let nxt = DeviceId(home.0.saturating_add(1) % n);
        if nxt == home {
            continue;
        }
        let slot = map.replicas.entry(k).or_default();
        if !slot.contains(&nxt) {
            slot.push(nxt);
        }
    }
    map
}

fn place_pair(
    assigned: &mut BTreeMap<ExpertKey, DeviceId>,
    load: &mut BTreeMap<DeviceId, u64>,
    freq: &BTreeMap<ExpertKey, u64>,
    n: u16,
    a: ExpertKey,
    b: ExpertKey,
) {
    let fa = freq.get(&a).copied().unwrap_or(0);
    let fb = freq.get(&b).copied().unwrap_or(0);
    match (assigned.get(&a).copied(), assigned.get(&b).copied()) {
        (None, None) => {
            let d = min_load(load, n);
            add_load(assigned, load, a, d, fa);
            add_load(assigned, load, b, d, fb);
        }
        (Some(d), None) => add_load(assigned, load, b, d, fb),
        (None, Some(d)) => add_load(assigned, load, a, d, fa),
        (Some(_), Some(_)) => {}
    }
}

fn min_load(load: &BTreeMap<DeviceId, u64>, n: u16) -> DeviceId {
    let mut best = DeviceId(0);
    let mut best_n = u64::MAX;
    for i in 0..n {
        let d = DeviceId(i);
        let v = load.get(&d).copied().unwrap_or(0);
        if v < best_n {
            best_n = v;
            best = d;
        }
    }
    best
}

fn add_load(
    assigned: &mut BTreeMap<ExpertKey, DeviceId>,
    load: &mut BTreeMap<DeviceId, u64>,
    key: ExpertKey,
    d: DeviceId,
    freq: u64,
) {
    if assigned.contains_key(&key) {
        return;
    }
    let _p = assigned.insert(key, d);
    let slot = load.entry(d).or_insert(0);
    *slot = slot.saturating_add(freq);
}
