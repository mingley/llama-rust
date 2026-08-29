//! Library tests: JSONL, Zipf-ish traces, policy order, leases, gpu-sim.

use super::*;
use gpu_sim::{DeviceId, HardwareProfile};
use std::collections::BTreeSet;

fn ev(token: u32, layer: u32, experts: &[u32]) -> ExpertAccess {
    ExpertAccess {
        sequence: 0,
        token,
        layer,
        experts: experts.to_vec(),
        weight_pt: Vec::new(),
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
fn jsonl_omits_w_and_parses_legacy() {
    let t = Trace {
        events: vec![ev(1, 2, &[3, 4])],
    };
    assert!(!t.to_jsonl().contains("\"w\""));
    let parsed = Trace::parse("{\"sequence\":0,\"token\":1,\"layer\":2,\"experts\":[3,4]}\n")
        .expect("legacy");
    let e = parsed.events.first().expect("one");
    assert!(e.weight_pt.is_empty());
    assert_eq!(e.experts, vec![3, 4]);
}

#[test]
fn jsonl_roundtrip_weight_pt() {
    let mut e = ev(1, 2, &[3, 4]);
    e.weight_pt = vec![500, 500];
    let t = Trace { events: vec![e] };
    let parsed = Trace::parse(&t.to_jsonl()).expect("parse");
    assert_eq!(parsed, t);
    assert!(t.to_jsonl().contains("\"w\":[500,500]"));
}

#[test]
fn weight_permille_floors_without_cast() {
    assert_eq!(weight_permille(0.0), 0);
    assert_eq!(weight_permille(0.5), 500);
    assert_eq!(weight_permille(1.0), 1000);
    assert_eq!(weight_permille(-1.0), 0);
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
    assert_eq!(s.coact_pairs, 2);
    assert_eq!(s.ws90, 3);
    assert_eq!(s.mass_pt, 0);
    assert!(!s.report().contains("mass‰="));
}

#[test]
fn seq_persist_and_reuse_on_repeated_token_expert() {
    let t = Trace {
        events: vec![ev(0, 0, &[4]), ev(1, 0, &[4]), ev(2, 0, &[4])],
    };
    let s = analyze(&t);
    assert_eq!(s.seq_persist_pt, 1000);
    assert!(s.reuse8_pt > 0);
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
    let empty = prefetch_keys(&Markov::new(), &[ExpertKey::new(0, 7)]);
    assert_eq!(empty, fwd);
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
    assert!(missy.energy_uj > 0);
    let ttft = missy.ttft_ns.expect("first token");
    assert!(ttft > 0);
    assert!(ttft <= missy.sim_ns);
    let itl = missy.itl_ns.expect("24 tokens");
    assert!(itl > 0);
    assert!(hitty.hits > missy.hits);
    assert!(missy.line().contains("energy_uj="));
    assert!(missy.line().contains("ttft_ns="));
    assert!(missy.line().contains("itl_ns="));
}

#[test]
fn placement_moves_when_reuse_beats_expert_bytes() {
    let expert = 188u64 << 20;
    let act = 4096u64;
    let pcie = 32u64.saturating_mul(1_000_000_000);
    assert_eq!(
        plan_placement(expert, act, 1, 1, pcie),
        Placement::DispatchActivations
    );
    assert_eq!(
        plan_placement(expert, act, 1, 64_000, pcie),
        Placement::MoveWeights
    );
}

#[test]
fn home_gpu_is_expert_mod_n() {
    assert_eq!(home_gpu(ExpertKey::new(3, 0), 8), DeviceId(0));
    assert_eq!(home_gpu(ExpertKey::new(3, 8), 8), DeviceId(0));
    assert_eq!(home_gpu(ExpertKey::new(0, 1), 8), DeviceId(1));
}

#[test]
fn static_ep_ooms_when_home_cannot_hold_two_experts() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 8])],
    };
    let p = HardwareProfile::example_8xh100_nvlink().restrict_hbm(4096);
    let bytes = 4096u64;
    let cached = sim_replay(&t, p.clone(), 1, Policy::Lru, bytes, 8).expect("lru");
    assert!(cached.misses >= 1);
    let st = sim_static_ep(&t, p, bytes);
    assert!(st.is_err(), "{st:?}");
}

#[test]
fn static_ep_parallel_pcie_beats_serial_gpu0() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1, 2, 3, 4, 5, 6, 7])],
    };
    let p = HardwareProfile::example_8xh100_nvlink();
    let bytes = 4u64 << 20;
    let row = compare_ep(&t, p, 8, bytes, 8).expect("ep");
    let st = row.static_ep.as_ref().expect("fits");
    assert!(
        st.sim_ns < row.cached.sim_ns,
        "static={} cached={}",
        st.sim_ns,
        row.cached.sim_ns
    );
    assert!(row.line().contains("cached"));
    assert!(row.line().contains("static"));
}

#[test]
fn markov_prefetch_beats_copy_forward_when_ids_are_not_sticky() {
    let mut events = Vec::new();
    for tok in 0..16u32 {
        events.push(ev(tok, 0, &[0]));
        events.push(ev(tok, 1, &[7]));
    }
    let t = Trace { events };
    let p = HardwareProfile::example_h100_sxm();
    let cfg = |prefetch: Prefetch| SimCfg {
        slots: 1,
        policy: Policy::Lru,
        bytes_per_expert: 4096,
        lookahead: 8,
        prefetch,
    };
    let none = sim_replay_cfg(&t, p.clone(), cfg(Prefetch::None)).expect("none");
    let fwd = sim_replay_cfg(&t, p.clone(), cfg(Prefetch::CopyForward)).expect("fwd");
    let mk = sim_replay_cfg(&t, p, cfg(Prefetch::Markov)).expect("mk");
    assert!(
        mk.hits > fwd.hits,
        "markov={} copy-forward={}",
        mk.hits,
        fwd.hits
    );
    assert!(
        mk.hits > none.hits,
        "markov={} demand={}",
        mk.hits,
        none.hits
    );
    assert!(mk.prefetches > 0);
    let both =
        sim_replay_cfg(&t, HardwareProfile::example_h100_sxm(), cfg(Prefetch::Both)).expect("both");
    assert!(both.prefetches >= fwd.prefetches);
}

#[test]
fn colocated_keeps_coactivated_pair_on_one_gpu() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1]), ev(1, 0, &[0, 1]), ev(2, 0, &[0, 1])],
    };
    let map = colocated(&t, 8);
    let a = ExpertKey::new(0, 0);
    let b = ExpertKey::new(0, 1);
    assert_eq!(map.home_of(a, 8), map.home_of(b, 8));
    let stripe = striped(&t, 8);
    assert_ne!(stripe.home_of(a, 8), stripe.home_of(b, 8));
}

#[test]
fn hot_replicas_move_more_bytes_than_stripe() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0]), ev(2, 0, &[0])],
    };
    let p = HardwareProfile::example_8xh100_nvlink();
    let bytes = 1u64 << 20;
    let base = striped(&t, 8);
    let hot = with_hot_replicas(base.clone(), &t, 8, 200);
    assert!(!hot.replicas.is_empty());
    let plain = sim_placed(&t, p.clone(), bytes, &base).expect("stripe");
    let rep = sim_placed(&t, p, bytes, &hot).expect("rep");
    assert!(rep.bytes_moved > plain.bytes_moved);
}

#[test]
fn simulated_gpu_store_evicts_and_scores() {
    let t = cycling_trace();
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::new(inner, 2, HardwareProfile::example_h100_sxm(), 4096).expect("gpu");
    assert!(gpu.staging_is_pinned());
    let before = gpu.score().expect("idle");
    assert_eq!(before.hbm_peak, 0);
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
    for kind in Workload::ALL {
        let t = generate(kind, 32, 8, 1, 1);
        assert!(!t.events.is_empty(), "{}", kind.name());
        assert!(!unique_keys(&t).is_empty());
    }
    let rows = adversarial_suite(16, 6, 2, HardwareProfile::example_cheap_48gb()).expect("suite");
    assert_eq!(rows.len(), Workload::ALL.len());
    assert!(rows[0].render().contains("uniform"));
    assert!(rows.iter().any(|r| r.name == "prefill-heavy"));
    assert!(rows.iter().any(|r| r.name == "batch"));
}

#[test]
fn topology_suite_covers_named_meshes() {
    let rows = topology_suite(1u64 << 20).expect("topo");
    assert_eq!(rows.len(), HardwareProfile::example_names().len());
    assert!(rows.iter().any(|r| r.name.contains("bad-numa")));
    assert!(rows.iter().any(|r| r.line().contains("p2p=")));
}

#[test]
fn simulated_gpu_store_transfer_failure_is_load_error() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::new(inner, 1, HardwareProfile::example_h100_sxm(), 4096).expect("gpu");
    gpu.fail_next_transfer();
    let k0 = ExpertKey::new(0, 0);
    let n = gpu.prefetch(&[k0]).expect("enqueue");
    assert_eq!(n, 1);
    let err = gpu.score().unwrap_err();
    assert!(err.to_string().contains("transfer"), "{err}");
}

#[test]
fn simulated_gpu_store_unavailable_blocks_acquire() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::new(inner, 1, HardwareProfile::example_h100_sxm(), 4096).expect("gpu");
    gpu.set_gpu_unavailable(true).expect("fault");
    let err = gpu.acquire(ExpertKey::new(0, 0)).unwrap_err();
    assert!(err.to_string().contains("unavailable"), "{err}");
}

#[test]
fn simulated_gpu_store_cancel_copy_stream() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::new(inner, 1, HardwareProfile::example_h100_sxm(), 4096).expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    let n = gpu.prefetch(&[k0]).expect("enqueue");
    assert_eq!(n, 1);
    let skipped = gpu.cancel_copy_stream().expect("cancel");
    assert!(skipped >= 1);
    let err = gpu.score().unwrap_err();
    assert!(err.to_string().contains("cancelled"), "{err}");
}

#[test]
fn tiered_memory_matches_direct_and_evicts() {
    let t = cycling_trace();
    let mut direct = DirectStore::from_trace(&t);
    let mut tier = TieredStore::memory(DirectStore::from_trace(&t), 2).expect("tier");
    assert_eq!(tier.storage(), WeightStorage::InMemory);
    let k0 = ExpertKey::new(0, 0);
    let k1 = ExpertKey::new(0, 1);
    let k2 = ExpertKey::new(0, 2);
    let a = tier.acquire(k0).expect("k0");
    let b = direct.acquire(k0).expect("d0");
    assert_eq!(a, b);
    let _k1 = tier.acquire(k1).expect("k1");
    let _k2 = tier.acquire(k2).expect("k2");
    assert_eq!(tier.fast_len(), 2);
    assert!(tier.metrics().evicts >= 1);
    assert!(tier.metrics().bytes_moved > 0);
    assert!(WeightStorage::mmap().is_err());
}

#[test]
fn tiered_file_pages_identity_bytes() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[1])],
    };
    let inner = DirectStore::from_trace(&t);
    let k0 = ExpertKey::new(0, 0);
    let want = inner.get(k0).expect("want");
    let mut path = std::env::temp_dir();
    path.push(format!(
        "expertvm-tiered-{}-{}.bin",
        std::process::id(),
        inner.len()
    ));
    let mut tier = TieredStore::on_path(inner, 1, &path).expect("file store");
    assert_eq!(tier.storage(), WeightStorage::File);
    let got = tier.acquire(k0).expect("page in");
    assert_eq!(got, want);
    let k1 = ExpertKey::new(0, 1);
    let _k1 = tier.acquire(k1).expect("evict k0");
    assert!(!tier.is_resident(k0));
    let again = tier.acquire(k0).expect("page back");
    assert_eq!(again, want);
    let _removed = std::fs::remove_file(&path);
}

#[test]
fn tiered_synthetic_faults_fill_bytes() {
    let mut keys = BTreeSet::new();
    let _i0 = keys.insert(ExpertKey::new(0, 0));
    let _i1 = keys.insert(ExpertKey::new(0, 1));
    let mut tier = TieredStore::synthetic(keys, 4, 7, 1).expect("syn");
    let p = tier.acquire(ExpertKey::new(0, 0)).expect("syn acq");
    assert_eq!(p.gate, vec![7, 7, 7, 7]);
    assert_eq!(tier.storage(), WeightStorage::Synthetic);
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

#[test]
fn migrate_moves_page_to_peer_and_gemms_there() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::new(inner, 1, HardwareProfile::example_2node_rdma(), 4096).expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    let _p = gpu.acquire(k0).expect("acq");
    assert_eq!(gpu.device_of(k0), Some(DeviceId(0)));
    gpu.migrate(k0, DeviceId(1)).expect("mig");
    assert_eq!(gpu.device_of(k0), Some(DeviceId(1)));
    let _p = gpu.acquire(k0).expect("gemm dest");
    let score = gpu.score().expect("score");
    assert!(score.bytes_moved >= 8192, "{}", score.bytes_moved);
}

#[test]
fn migrate_single_gpu_is_no_peer() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::new(inner, 1, HardwareProfile::example_h100_sxm(), 4096).expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    let _p = gpu.acquire(k0).expect("acq");
    let err = gpu.migrate(k0, DeviceId(1)).unwrap_err();
    assert!(err.to_string().contains("no peer"), "{err}");
}

#[test]
fn simulated_gpu_store_places_on_striped_home() {
    let t = Trace {
        events: vec![ev(0, 0, &[1])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::new(inner, 1, HardwareProfile::example_2node_rdma(), 4096).expect("gpu");
    let k = ExpertKey::new(0, 1);
    let _p = gpu.acquire(k).expect("acq");
    assert_eq!(gpu.device_of(k), Some(DeviceId(1)));
    let score = gpu.score().expect("score");
    assert!(score.bytes_moved >= 4096);
}

#[test]
fn remote_home_on_rdma_pays_peer_copy() {
    let t = Trace {
        events: vec![ev(0, 0, &[1])],
    };
    let p = HardwareProfile::example_2node_rdma();
    let bytes = 1u64 << 20;
    let map = striped(&t, 2);
    let local = sim_placed(&t, p.clone(), bytes, &map).expect("local");
    let remote = sim_remote_home(&t, p.clone(), bytes, &map).expect("remote");
    assert!(
        remote.bytes_moved > local.bytes_moved,
        "remote={} local={}",
        remote.bytes_moved,
        local.bytes_moved
    );
    assert!(
        remote.sim_ns > local.sim_ns,
        "remote={} local={}",
        remote.sim_ns,
        local.sim_ns
    );
    assert!(
        remote.bytes_moved < bytes.saturating_mul(2),
        "dispatch should not D2D the full expert; moved={}",
        remote.bytes_moved
    );
    let moved = sim_remote_home_cfg(&t, p, bytes, bytes, &map).expect("move");
    assert!(
        moved.bytes_moved >= bytes.saturating_mul(2),
        "equal act vs expert volume must move weights; moved={}",
        moved.bytes_moved
    );
    assert!(moved.bytes_moved > remote.bytes_moved);
}

#[test]
fn live_store_migrate_is_noop_on_cached() {
    let t = cycling_trace();
    let mut live = LiveStore::Cached(CachedStore::new(DirectStore::from_trace(&t), 2).unwrap());
    let k = ExpertKey::new(0, 0);
    let _p = live.acquire(k).expect("acq");
    live.migrate(k, DeviceId(1)).expect("noop");
}

#[test]
fn mass_table_weights_hottest_key_above_count_share() {
    let mut a = ev(0, 0, &[0, 1]);
    a.weight_pt = vec![900, 100];
    let mut b = ev(1, 0, &[0, 1]);
    b.weight_pt = vec![900, 100];
    let t = Trace { events: vec![a, b] };
    let s = analyze(&t);
    assert_eq!(s.mass_pt, 2000);
    assert_eq!(s.top20_share_pt, 500);
    assert_eq!(s.top20_mass_pt, 900);
    assert!(s.report().contains("top20_mass‰=900"));
    assert_eq!(
        mass_table(&t).get(&ExpertKey::new(0, 0)).copied(),
        Some(1800)
    );
}

#[test]
fn hot_replicas_prefer_mass_when_w_is_present() {
    let mut a = ev(0, 0, &[0, 1]);
    a.weight_pt = vec![900, 100];
    let mut b = ev(1, 0, &[0, 1]);
    b.weight_pt = vec![900, 100];
    let t = Trace { events: vec![a, b] };
    let map = with_hot_replicas(striped(&t, 8), &t, 8, 500);
    assert!(map.replicas.contains_key(&ExpertKey::new(0, 0)));
    assert!(!map.replicas.contains_key(&ExpertKey::new(0, 1)));
}
