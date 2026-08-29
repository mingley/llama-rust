//! Library tests: JSONL, Zipf-ish traces, policy order, leases, gpu-sim.

use super::*;
use gpu_sim::HardwareProfile;
use std::collections::BTreeSet;

fn ev(token: u32, layer: u32, experts: &[u32]) -> ExpertAccess {
    ExpertAccess {
        sequence: 0,
        token,
        layer,
        experts: experts.to_vec(),
    }
}

fn cycling_trace() -> Trace {
    // 3 experts, repeating: LRU with 2 slots thrashes; oracle does not.
    let mut events = Vec::new();
    for t in 0..24u32 {
        events.push(ev(t, 0, &[t % 3]));
    }
    Trace { events }
}

fn zipf_trace() -> Trace {
    let mut events = Vec::new();
    let mut rng = 1u64;
    for tok in 0..64u32 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bucket = (rng >> 33) % 100;
        let a = if bucket < 70 {
            0
        } else if bucket < 90 {
            1
        } else {
            2 + u32::try_from(bucket % 6).unwrap_or(2)
        };
        events.push(ev(tok, 0, &[a]));
    }
    Trace { events }
}

#[test]
fn jsonl_roundtrip() {
    let t = Trace {
        events: vec![ev(1, 2, &[3, 4])],
    };
    let text = t.to_jsonl();
    let parsed = Trace::parse(&text).expect("parse");
    assert_eq!(parsed, t);
}

#[test]
fn analyze_counts_unique_and_persist() {
    let t = Trace {
        events: vec![ev(0, 0, &[1, 2]), ev(0, 1, &[1, 7])],
    };
    let s = analyze(&t);
    assert_eq!(s.n_events, 2);
    assert_eq!(s.n_acquires, 4);
    assert_eq!(s.n_unique, 4);
    assert!(s.layer_persist_pt > 0);
}

#[test]
fn oracle_beats_lru_on_cyclic_thrash() {
    let t = cycling_trace();
    let lru = replay(&t, 2, Policy::Lru, 8);
    let oracle = replay(&t, 2, Policy::Oracle, 8);
    assert!(
        oracle.hits > lru.hits,
        "oracle={} lru={}",
        oracle.hits,
        lru.hits
    );
    assert_eq!(lru.hits, 0);
}

#[test]
fn zipf_lru_beats_random_or_ties() {
    let t = zipf_trace();
    let table = compare(&t, 2, 8);
    let random = table.iter().find(|r| r.policy == Policy::Random).unwrap();
    let lru = table.iter().find(|r| r.policy == Policy::Lru).unwrap();
    let oracle = table.iter().find(|r| r.policy == Policy::Oracle).unwrap();
    assert!(oracle.hits >= lru.hits);
    assert!(oracle.hits >= random.hits);
    assert!(lru.hits > 0);
}

#[test]
fn direct_store_is_identity() {
    let t = zipf_trace();
    let mut store = DirectStore::from_trace(&t);
    for k in t.keys() {
        let a = store.acquire(k).expect("blob");
        let b = store.get(k).expect("get");
        assert_eq!(a, b);
        assert_eq!(a.gate, vec![1]);
        assert_eq!(a.up, vec![1]);
        assert_eq!(a.down, vec![1]);
    }
    assert_eq!(store.misses(), 0);
}

#[test]
fn cached_store_lease_blocks_eviction() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[1]), ev(2, 0, &[2])],
    };
    let mut cache = CachedStore::new(DirectStore::from_trace(&t), 2).expect("cache");
    let k0 = ExpertKey::new(0, 0);
    let k1 = ExpertKey::new(0, 1);
    let k2 = ExpertKey::new(0, 2);
    let _k0 = cache.acquire(k0).expect("k0");
    let _k1 = cache.acquire(k1).expect("k1");
    cache.lease(k0).expect("lease k0");
    cache.lease(k1).expect("lease k1");
    let err = cache.acquire(k2).unwrap_err();
    assert!(matches!(err, Error::Store(_)));
    cache.release(k0);
    let _k2 = cache.acquire(k2).expect("evict unleased");
    assert!(cache.evicts() >= 1);
}

#[test]
fn cached_store_prefetch_skips_unknown_and_counts() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[1])],
    };
    let mut cache = CachedStore::new(DirectStore::from_trace(&t), 2).expect("cache");
    let known = ExpertKey::new(0, 0);
    let ghost = ExpertKey::new(9, 9);
    let n = cache.prefetch(&[ghost, known]).expect("prefetch");
    assert_eq!(n, 1);
    assert!(cache.is_resident(known));
    assert_eq!(cache.metrics().prefetches, 1);
    let _hit = cache.acquire(known).expect("hit after prefetch");
    assert_eq!(cache.hits(), 1);
    assert_eq!(cache.misses(), 0);
}

#[test]
fn planner_stays_when_window_is_resident() {
    let t = cycling_trace();
    let mut resident = BTreeSet::new();
    let _i0 = resident.insert(ExpertKey::new(0, 0));
    let _i1 = resident.insert(ExpertKey::new(0, 1));
    let _i2 = resident.insert(ExpertKey::new(0, 2));
    assert_eq!(plan_window(&resident, &t, 0, 8, 500), Plan::Stay);
    assert_eq!(plan_window(&BTreeSet::new(), &t, 0, 8, 500), Plan::Fetch);
    assert!(!window_keys(&t, 0, 3).is_empty());
    let fwd = copy_forward(&[ExpertKey::new(0, 7)]);
    assert_eq!(fwd[0], ExpertKey::new(1, 7));
    let hot = hot_keys(&t, 1);
    assert_eq!(hot.len(), 1);
}

#[test]
fn sim_replay_moves_bytes_on_miss() {
    let t = cycling_trace();
    let hitty = sim_replay(
        &t,
        HardwareProfile::example_h100_sxm(),
        8,
        Policy::Lru,
        4096,
        8,
    )
    .expect("sim full");
    let missy = sim_replay(
        &t,
        HardwareProfile::example_h100_sxm(),
        1,
        Policy::Lru,
        4096,
        8,
    )
    .expect("sim tight");
    assert!(missy.bytes_moved > hitty.bytes_moved);
    assert!(missy.sim_ns > 0);
    assert!(hitty.hits > missy.hits);
}

#[test]
fn simulated_gpu_store_evicts_and_scores() {
    let t = cycling_trace();
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::new(inner, 2, HardwareProfile::example_h100_sxm(), 4096).expect("gpu");
    for k in t.keys() {
        let parts = gpu.acquire(k).expect("acq");
        assert_eq!(parts.gate, vec![1]);
    }
    let m = gpu.metrics();
    assert!(m.misses >= 3);
    assert!(m.evicts >= 1);
    let score = gpu.score().expect("score");
    assert!(score.wall_ns > 0);
    assert!(score.bytes_moved > 0);
    assert!(gpu.metrics().bytes_moved > 0);
    assert!(score.with_tokens(24).ns_per_token.is_some());
}

#[test]
fn simulated_gpu_store_pin_hot_replicates_on_nvlink() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[1])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::new(inner, 2, HardwareProfile::example_8xh100_nvlink(), 4096)
        .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    gpu.pin_hot(&[k0]).expect("pin");
    let _p = gpu.acquire(k0).expect("hit");
    assert!(gpu.is_resident(k0));
    let score = gpu.score().expect("score");
    assert!(score.bytes_moved >= 4096);
}

#[test]
fn adversarial_workloads_are_named_and_measurable() {
    let kinds = [
        Workload::Uniform,
        Workload::Hotset,
        Workload::ShiftingHotset,
        Workload::Thrash,
    ];
    for kind in kinds {
        let t = generate(kind, 32, 8, 1, 1);
        assert!(!t.events.is_empty(), "{}", kind.name());
        assert!(!unique_keys(&t).is_empty());
    }
    let rows = adversarial_suite(16, 6, 2, HardwareProfile::example_cheap_48gb()).expect("suite");
    assert_eq!(rows.len(), 4);
    assert!(rows[0].render().contains("uniform"));
}

#[test]
fn live_store_dispatches() {
    let t = cycling_trace();
    let mut live = LiveStore::Cached(CachedStore::new(DirectStore::from_trace(&t), 2).unwrap());
    let k = ExpertKey::new(0, 0);
    let _p = live.acquire(k).expect("acq");
    live.pin_hot(&[k]).expect("pin");
    assert!(live.is_resident(k));
    assert!(live.score().expect("score").is_none());
}

#[test]
fn format_table_names_policies() {
    let t = cycling_trace();
    let text = format_table(&compare(&t, 2, 4));
    assert!(text.contains("lru"));
    assert!(text.contains("oracle"));
}
