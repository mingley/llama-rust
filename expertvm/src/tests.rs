//! Library tests: JSONL, Zipf-ish traces, policy order, leases, gpu-sim.

use super::*;
use gpu_sim::{DeviceId, HardwareProfile};
use std::collections::BTreeSet;

fn ev(token: u32, layer: u32, experts: &[u32]) -> ExpertAccess {
    ev_seq(0, token, layer, experts)
}

fn ev_seq(seq: u64, token: u32, layer: u32, experts: &[u32]) -> ExpertAccess {
    ExpertAccess {
        sequence: seq,
        token,
        layer,
        experts: experts.to_vec(),
        weight_pt: Vec::new(),
        prefix: None,
    }
}

fn load_checked_in_trace(name: &str) -> Trace {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("tests")
        .join("traces")
        .join(name);
    let mut f = std::fs::File::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut text = String::new();
    let _n = std::io::Read::read_to_string(&mut f, &mut text).expect("read");
    Trace::parse(&text).expect("parse")
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
    assert!(!t.to_jsonl().contains("\"p\""));
    let parsed = Trace::parse("{\"sequence\":0,\"token\":1,\"layer\":2,\"experts\":[3,4]}\n")
        .expect("legacy");
    let e = parsed.events.first().expect("one");
    assert!(e.weight_pt.is_empty());
    assert!(e.prefix.is_none());
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
fn jsonl_roundtrip_prefix() {
    let mut e = ev(1, 2, &[3, 4]);
    e.prefix = Some(prefix_hash(&[1, 2, 3]));
    let t = Trace { events: vec![e] };
    let parsed = Trace::parse(&t.to_jsonl()).expect("parse");
    assert_eq!(parsed, t);
    assert!(t.to_jsonl().contains("\"p\":"));
}

#[test]
fn prefix_hash_is_content_addressed() {
    assert_eq!(prefix_hash(&[1]), prefix_hash(&[1]));
    assert_ne!(prefix_hash(&[1]), prefix_hash(&[1, 2]));
    assert_ne!(prefix_hash(&[1, 2]), prefix_hash(&[1, 3]));
    assert_eq!(prefix_hash(&[]), prefix_hash(&[]));
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
    assert_eq!(s.order2_persist_pt, 0);
    assert!(!s.report().contains("mass‰="));
    assert!(s.report().contains("order2_persist‰="));
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
fn two_layer_jsonl_seq_persist_skips_interleaved_layers() {
    let t = Trace {
        events: vec![
            ev(0, 0, &[1]),
            ev(0, 1, &[2]),
            ev(1, 0, &[1]),
            ev(1, 1, &[2]),
            ev(2, 0, &[1]),
            ev(2, 1, &[2]),
        ],
    };
    let s = analyze(&t);
    assert_eq!(s.n_events, 6);
    assert_eq!(
        s.layer_persist_pt, 0,
        "ids 1→2 at the next layer must not count as layer persist"
    );
    assert_eq!(
        s.seq_persist_pt, 1000,
        "same-layer next-token pairs are not adjacent in layer-major JSONL, seq_persist‰={}",
        s.seq_persist_pt
    );
    assert_eq!(
        s.order2_persist_pt, 1000,
        "same-layer lookback-2 must train across interleaved layers, order2_persist‰={}",
        s.order2_persist_pt
    );
}

#[test]
fn two_layer_layer_persist_same_ids() {
    let t = Trace {
        events: vec![
            ev(0, 0, &[3]),
            ev(0, 1, &[3]),
            ev(1, 0, &[3]),
            ev(1, 1, &[3]),
        ],
    };
    let s = analyze(&t);
    assert_eq!(s.layer_persist_pt, 1000);
    assert_eq!(s.seq_persist_pt, 1000);
}

#[test]
fn checked_in_two_layer_qwen3moe_has_honest_persist() {
    let t = load_checked_in_trace("tiny-qwen3moe-2layer.jsonl");
    assert!(t.events.iter().any(|e| e.layer == 0));
    assert!(t.events.iter().any(|e| e.layer == 1));
    let s = analyze(&t);
    assert!(
        s.seq_persist_pt > 0,
        "seq_persist must see same-layer next-token pairs, {}",
        s.report()
    );
    assert!(
        s.layer_persist_pt > 0,
        "layer_persist must see L→L+1 pairs, {}",
        s.report()
    );
    assert!(
        s.order2_persist_pt > 0,
        "order2 must train on same-layer lookback-2, {}",
        s.report()
    );
}

#[test]
fn order2_persist_predicts_a_cycle() {
    let t = cycling_trace();
    let s = analyze(&t);
    assert!(
        s.order2_persist_pt > 500,
        "order2_persist‰={}",
        s.order2_persist_pt
    );
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
fn cached_store_phase_is_cold_resident_leased() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let mut cache = CachedStore::new(DirectStore::from_trace(&t), 1).expect("cache");
    let k = ExpertKey::new(0, 0);
    assert_eq!(cache.phase(k), ExpertPhase::Cold);
    let _p = cache.acquire(k).expect("acq");
    assert_eq!(cache.phase(k), ExpertPhase::Resident);
    cache.lease(k).expect("lease");
    assert_eq!(cache.phase(k), ExpertPhase::Leased);
    assert!(cache.is_leased(k));
    cache.release(k);
    assert_eq!(cache.phase(k), ExpertPhase::Resident);
    cache.evict(k).expect("evict");
    assert_eq!(cache.phase(k), ExpertPhase::Cold);
    let err = cache.lease(k).unwrap_err();
    assert!(matches!(err, Error::Store(_)));
}

#[test]
fn cached_store_pin_hot_survives_release() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(0, 0, &[1]), ev(0, 0, &[2])],
    };
    let mut cache = CachedStore::new(DirectStore::from_trace(&t), 2).expect("cache");
    let k0 = ExpertKey::new(0, 0);
    let k1 = ExpertKey::new(0, 1);
    let k2 = ExpertKey::new(0, 2);
    let _p0 = cache.acquire(k0).expect("k0");
    cache.pin_hot(&[k0]).expect("pin");
    assert!(cache.is_pinned(k0));
    assert!(!cache.is_leased(k0));
    assert_eq!(cache.phase(k0), ExpertPhase::Leased);
    cache.lease(k0).expect("compute lease");
    cache.release(k0);
    assert!(cache.is_pinned(k0));
    assert_eq!(cache.phase(k0), ExpertPhase::Leased);
    let _p1 = cache.acquire(k1).expect("k1");
    let _p2 = cache.acquire(k2).expect("evict unpinned");
    assert!(cache.is_resident(k0));
    assert!(cache.is_resident(k2));
    assert!(!cache.is_resident(k1));
    assert_eq!(cache.metrics().pins, 1);
}

#[test]
fn cached_store_unpin_all_frees_demand_slot() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(0, 0, &[1])],
    };
    let mut cache = CachedStore::new(DirectStore::from_trace(&t), 1).expect("cache");
    let k0 = ExpertKey::new(0, 0);
    let k1 = ExpertKey::new(0, 1);
    cache.pin_hot(&[k0]).expect("pin");
    let err = cache.acquire(k1).unwrap_err();
    assert!(matches!(err, Error::Store(_)));
    cache.unpin_all();
    assert!(!cache.is_pinned(k0));
    let _p1 = cache.acquire(k1).expect("evict after unpin");
    assert!(cache.is_resident(k1));
    assert!(!cache.is_resident(k0));
    assert_eq!(cache.pin_budget(), 0);
}

#[test]
fn cached_store_evict_of_leased_is_fatal() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let mut cache = CachedStore::new(DirectStore::from_trace(&t), 1).expect("cache");
    let k = ExpertKey::new(0, 0);
    let _p = cache.acquire(k).expect("acq");
    cache.lease(k).expect("lease");
    let err = cache.evict(k).unwrap_err();
    assert!(matches!(err, Error::Store(_)));
    assert_eq!(cache.phase(k), ExpertPhase::Leased);
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
        prefetch,
        ..SimCfg::lru(1, 4096, 8)
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
    assert!(
        mk.prefetch_hits > 0,
        "useful prefetch must show up as demand hits; line={}",
        mk.line()
    );
    let both =
        sim_replay_cfg(&t, HardwareProfile::example_h100_sxm(), cfg(Prefetch::Both)).expect("both");
    assert!(both.prefetches >= fwd.prefetches);
}

#[test]
fn markov_order2_breaks_order1_tie() {
    let mut m = Markov::new();
    let l0e5 = ExpertKey::new(0, 5);
    let l0e6 = ExpertKey::new(0, 6);
    let l1e0 = ExpertKey::new(1, 0);
    let l2e1 = ExpertKey::new(2, 1);
    let l2e2 = ExpertKey::new(2, 2);
    for _ in 0..8 {
        m.observe_ctx(&[l0e5], &[l1e0], &[l2e1]);
        m.observe_ctx(&[l0e6], &[l1e0], &[l2e2]);
    }
    let from0 = [l1e0];
    assert_eq!(m.predict(&from0, 1), vec![l2e1]);
    assert_eq!(m.predict_ctx(&[l0e6], &from0, 1), vec![l2e2]);
    assert_eq!(m.predict_ctx(&[l0e5], &from0, 1), vec![l2e1]);
    let ctx = prefetch_keys_ctx(&m, &[l0e6], &from0);
    assert!(ctx.contains(&l2e2));
}

#[test]
fn markov_order2_prefetch_beats_copy_forward_on_tied_ids() {
    let mut events = Vec::new();
    for tok in 0..16u32 {
        if tok % 2 == 0 {
            events.push(ev(tok, 0, &[5]));
            events.push(ev(tok, 1, &[0]));
            events.push(ev(tok, 2, &[1]));
        } else {
            events.push(ev(tok, 0, &[6]));
            events.push(ev(tok, 1, &[0]));
            events.push(ev(tok, 2, &[2]));
        }
    }
    let t = Trace { events };
    let p = HardwareProfile::example_h100_sxm();
    let cfg = |prefetch: Prefetch| SimCfg {
        slots: 1,
        prefetch,
        ..SimCfg::lru(1, 4096, 8)
    };
    let fwd = sim_replay_cfg(&t, p.clone(), cfg(Prefetch::CopyForward)).expect("fwd");
    let mk = sim_replay_cfg(&t, p, cfg(Prefetch::Markov)).expect("mk");
    assert!(
        mk.hits > fwd.hits,
        "order2 markov={} copy-forward={}",
        mk.hits,
        fwd.hits
    );
}

#[test]
fn seq_streams_overlap_beats_serial_on_batch_token() {
    let t = Trace {
        events: vec![
            ExpertAccess {
                sequence: 0,
                token: 0,
                layer: 0,
                experts: vec![0],
                weight_pt: Vec::new(),
                prefix: None,
            },
            ExpertAccess {
                sequence: 1,
                token: 0,
                layer: 0,
                experts: vec![1],
                weight_pt: Vec::new(),
                prefix: None,
            },
        ],
    };
    let p = HardwareProfile::parse("gpus=1\ncopy_engines=1\n").expect("profile");
    let cfg = |seq_streams: bool| SimCfg {
        slots: 2,
        bytes_per_expert: 32u64 << 20,
        lookahead: 0,
        seq_streams,
        ..SimCfg::lru(2, 32u64 << 20, 0)
    };
    let serial = sim_replay_cfg(&t, p.clone(), cfg(false)).expect("serial");
    let overlap = sim_replay_cfg(&t, p, cfg(true)).expect("overlap");
    assert!(
        overlap.sim_ns < serial.sim_ns,
        "overlap={} serial={}",
        overlap.sim_ns,
        serial.sim_ns
    );
}

#[test]
fn sync_alloc_cannot_overlap_misses_across_streams() {
    let t = Trace {
        events: vec![
            ExpertAccess {
                sequence: 0,
                token: 0,
                layer: 0,
                experts: vec![0],
                weight_pt: Vec::new(),
                prefix: None,
            },
            ExpertAccess {
                sequence: 1,
                token: 0,
                layer: 0,
                experts: vec![1],
                weight_pt: Vec::new(),
                prefix: None,
            },
        ],
    };
    let p = HardwareProfile::parse("gpus=1\ncopy_engines=1\n").expect("profile");
    let cfg = |sync_alloc: bool| SimCfg {
        slots: 2,
        bytes_per_expert: 32u64 << 20,
        lookahead: 0,
        seq_streams: true,
        sync_alloc,
        ..SimCfg::lru(2, 32u64 << 20, 0)
    };
    let async_row = sim_replay_cfg(&t, p.clone(), cfg(false)).expect("async");
    let sync_row = sim_replay_cfg(&t, p, cfg(true)).expect("sync");
    assert_eq!(async_row.hits, sync_row.hits);
    assert_eq!(async_row.misses, sync_row.misses);
    assert!(
        sync_row.sim_ns > async_row.sim_ns,
        "cudaMalloc must serialize the two-stream miss; sync={} async={}",
        sync_row.sim_ns,
        async_row.sim_ns
    );
}

#[test]
fn blocking_streams_cannot_overlap_misses_across_streams() {
    let t = Trace {
        events: vec![
            ExpertAccess {
                sequence: 0,
                token: 0,
                layer: 0,
                experts: vec![0],
                weight_pt: Vec::new(),
                prefix: None,
            },
            ExpertAccess {
                sequence: 1,
                token: 0,
                layer: 0,
                experts: vec![1],
                weight_pt: Vec::new(),
                prefix: None,
            },
        ],
    };
    let p = HardwareProfile::parse("gpus=1\ncopy_engines=1\n").expect("profile");
    let cfg = |blocking_streams: bool| SimCfg {
        slots: 2,
        bytes_per_expert: 32u64 << 20,
        lookahead: 0,
        seq_streams: true,
        blocking_streams,
        ..SimCfg::lru(2, 32u64 << 20, 0)
    };
    let nonblock = sim_replay_cfg(&t, p.clone(), cfg(false)).expect("nb");
    let block = sim_replay_cfg(&t, p.clone(), cfg(true)).expect("block");
    assert_eq!(nonblock.hits, block.hits);
    assert_eq!(nonblock.misses, block.misses);
    assert!(
        block.sim_ns > nonblock.sim_ns,
        "cudaStreamCreate must serialize the two-stream miss; block={} nb={}",
        block.sim_ns,
        nonblock.sim_ns
    );
    let sched_nb = schedule_replay(&t, p.clone(), cfg(false), SchedCfg::closed(0)).expect("snb");
    let sched_b = schedule_replay(&t, p, cfg(true), SchedCfg::closed(0)).expect("sb");
    assert_eq!(sched_nb.replay.hits, sched_b.replay.hits);
    assert!(
        sched_b.replay.sim_ns > sched_nb.replay.sim_ns,
        "schedule blocking={} nb={}",
        sched_b.replay.sim_ns,
        sched_nb.replay.sim_ns
    );
}

#[test]
fn mempool_reuse_beats_first_touch_on_thrash() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let cfg = |mempool: bool| SimCfg {
        slots: 1,
        mempool,
        ..SimCfg::lru(1, 4096, 0)
    };
    let released = sim_replay_cfg(&t, p.clone(), cfg(false)).expect("release");
    let held = sim_replay_cfg(&t, p, cfg(true)).expect("hold");
    assert_eq!(released.hits, held.hits);
    assert_eq!(released.misses, held.misses);
    assert!(
        held.sim_ns < released.sim_ns,
        "cached cudaMallocFromPoolAsync must beat first-touch; pool={} release={}",
        held.sim_ns,
        released.sim_ns
    );
}

#[test]
fn mapped_host_skips_hbm_and_h2d() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let cfg = |mapped: bool| SimCfg {
        slots: 1,
        mapped,
        ..SimCfg::lru(1, 1u64 << 20, 0)
    };
    let h2d = sim_replay_cfg(&t, p.clone(), cfg(false)).expect("h2d");
    let mapped = sim_replay_cfg(&t, p, cfg(true)).expect("mapped");
    assert_eq!(h2d.hits, mapped.hits);
    assert_eq!(h2d.misses, mapped.misses);
    assert_eq!(mapped.hbm_peak, 0);
    assert_eq!(mapped.bytes_moved, 0);
    assert!(h2d.hbm_peak > 0);
    assert!(h2d.bytes_moved > 0);
}

#[test]
fn managed_prefetch_matches_h2d_hits() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let cfg = |managed: bool| SimCfg {
        slots: 1,
        managed,
        ..SimCfg::lru(1, 1u64 << 20, 0)
    };
    let h2d = sim_replay_cfg(&t, p.clone(), cfg(false)).expect("h2d");
    let um = sim_replay_cfg(&t, p, cfg(true)).expect("managed");
    assert_eq!(h2d.hits, um.hits);
    assert_eq!(h2d.misses, um.misses);
    assert!(um.hbm_peak > 0);
    assert!(um.bytes_moved > 0);
    assert_eq!(um.hbm_peak, h2d.hbm_peak);
}

#[test]
fn vmm_map_matches_h2d_hits() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let cfg = |vmm: bool| SimCfg {
        slots: 1,
        vmm,
        ..SimCfg::lru(1, 1u64 << 20, 0)
    };
    let h2d = sim_replay_cfg(&t, p.clone(), cfg(false)).expect("h2d");
    let vmm = sim_replay_cfg(&t, p, cfg(true)).expect("vmm");
    assert_eq!(h2d.hits, vmm.hits);
    assert_eq!(h2d.misses, vmm.misses);
    assert!(vmm.hbm_peak > 0);
    assert!(vmm.bytes_moved > 0);
    assert_eq!(vmm.hbm_peak, h2d.hbm_peak);
}

#[test]
fn vmm_paged_matches_full_vmm_hits() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let bytes = 1u64 << 20;
    let full = SimCfg {
        slots: 1,
        vmm: true,
        ..SimCfg::lru(1, bytes, 0)
    };
    let paged = SimCfg {
        vmm_page: bytes / 4,
        ..full
    };
    let one = sim_replay_cfg(&t, p.clone(), full).expect("full");
    let many = sim_replay_cfg(&t, p, paged).expect("paged");
    assert_eq!(one.hits, many.hits);
    assert_eq!(one.misses, many.misses);
    assert_eq!(one.hbm_peak, many.hbm_peak);
    assert!(
        many.sim_ns > one.sim_ns,
        "paged maps must pay per-block map overhead; paged={} full={}",
        many.sim_ns,
        one.sim_ns
    );
}

#[test]
fn vmm_evict_reacquires_same_va() {
    use crate::replay::Touch;
    use crate::sim_replay::{apply_touch, GraphBank, PageHandle, TouchArgs};
    use gpu_sim::{Sim, StreamId};
    use std::collections::BTreeMap;

    let mut sim = Sim::new(HardwareProfile::example_h100_sxm());
    let mut handles: BTreeMap<ExpertKey, PageHandle> = BTreeMap::new();
    let mut graphs = GraphBank::new(false, false, false, crate::sim_replay::LeafMem::None);
    let args = TouchArgs {
        d: DeviceId(0),
        s: StreamId(0),
        bytes: 4096,
        slots: 1,
        sync_alloc: false,
        mapped: false,
        managed: false,
        vmm: true,
        vmm_page: 0,
        pageable: false,
        accessed_by: false,
    };
    let mut next_event = 1u32;
    let k0 = ExpertKey::new(0, 0);
    let k1 = ExpertKey::new(0, 1);
    apply_touch(
        &mut sim,
        &mut handles,
        &mut graphs,
        args,
        k0,
        Touch::Miss { evicted: None },
        &mut next_event,
    )
    .expect("k0");
    let id0 = handles.get(&k0).expect("h0").id;
    apply_touch(
        &mut sim,
        &mut handles,
        &mut graphs,
        args,
        k1,
        Touch::Miss { evicted: Some(k0) },
        &mut next_event,
    )
    .expect("k1");
    assert_eq!(handles.get(&k1).expect("h1").id, id0);
    assert_eq!(sim.vmm_idle_len(), 0);
    apply_touch(
        &mut sim,
        &mut handles,
        &mut graphs,
        args,
        k0,
        Touch::Miss { evicted: Some(k1) },
        &mut next_event,
    )
    .expect("k0 again");
    assert_eq!(handles.get(&k0).expect("h0b").id, id0);
}

#[test]
fn host_func_lengthens_wall_not_hits() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let base = SimCfg::lru(2, 4096, 0);
    let plain = sim_replay_cfg(&t, p.clone(), base).expect("plain");
    let cb = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            host_func: true,
            ..base
        },
    )
    .expect("host");
    assert_eq!(plain.hits, cb.hits);
    assert_eq!(plain.misses, cb.misses);
    assert!(
        cb.sim_ns > plain.sim_ns,
        "host={} plain={}",
        cb.sim_ns,
        plain.sim_ns
    );
}

#[test]
fn mapped_pin_budget_caps_occupancy() {
    let mut events = Vec::new();
    for tok in 0..16u32 {
        events.push(ev(tok, 0, &[tok % 2]));
    }
    let t = Trace { events };
    let bytes = 1u64 << 20;
    let tight = HardwareProfile::example_h100_sxm().restrict_pin(bytes);
    let open = HardwareProfile::example_h100_sxm();
    let two = SimCfg {
        slots: 2,
        mapped: true,
        ..SimCfg::lru(2, bytes, 0)
    };
    let one = SimCfg {
        slots: 1,
        mapped: true,
        ..SimCfg::lru(1, bytes, 0)
    };
    let capped = sim_replay_cfg(&t, tight.clone(), two).expect("cap");
    let slim = sim_replay_cfg(&t, open.clone(), one).expect("one");
    let fat = sim_replay_cfg(&t, open, two).expect("two");
    assert_eq!(capped.hits, slim.hits);
    assert_eq!(capped.misses, slim.misses);
    assert!(
        fat.hits > capped.hits,
        "uncapped two-slot mapped must hit; fat={} cap={}",
        fat.hits,
        capped.hits
    );
    let sched_capped = schedule_replay(&t, tight, two, SchedCfg::closed(0)).expect("sched cap");
    assert_eq!(sched_capped.replay.hits, capped.hits);
}

#[test]
fn mapped_pin_budget_zero_fit_is_pin_oom() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm().restrict_pin(1);
    let cfg = SimCfg {
        slots: 1,
        mapped: true,
        ..SimCfg::lru(1, 1u64 << 20, 0)
    };
    let err = sim_replay_cfg(&t, p, cfg).unwrap_err();
    assert!(err.to_string().contains("pin"), "{err}");
}

#[test]
fn max_batch_serializes_sequences_at_a_token() {
    let t = Trace {
        events: vec![
            ExpertAccess {
                sequence: 0,
                token: 0,
                layer: 0,
                experts: vec![0],
                weight_pt: Vec::new(),
                prefix: None,
            },
            ExpertAccess {
                sequence: 1,
                token: 0,
                layer: 0,
                experts: vec![1],
                weight_pt: Vec::new(),
                prefix: None,
            },
        ],
    };
    let p = HardwareProfile::parse("gpus=1\ncopy_engines=1\n").expect("profile");
    let cfg = |max_batch: usize| SimCfg {
        slots: 2,
        bytes_per_expert: 32u64 << 20,
        lookahead: 0,
        seq_streams: true,
        max_batch,
        ..SimCfg::lru(2, 32u64 << 20, 0)
    };
    let all = sim_replay_cfg(&t, p.clone(), cfg(0)).expect("admit all");
    let one = sim_replay_cfg(&t, p, cfg(1)).expect("admit one");
    assert_eq!(all.hits, one.hits);
    assert_eq!(all.misses, one.misses);
    assert_eq!(all.ttft_ns.is_some(), one.ttft_ns.is_some());
    assert!(all.itl_ns.is_none());
    assert!(one.itl_ns.is_none());
    assert!(
        one.sim_ns > all.sim_ns,
        "max_batch=1 must drain before the next sequence; one={} all={}",
        one.sim_ns,
        all.sim_ns
    );
}

#[test]
fn demand_oracle_cannot_beat_belady() {
    let t = cycling_trace();
    let full = replay(&t, 2, Policy::Oracle, 0);
    let mut w = crate::replay::Walker::demand(2, Policy::Oracle, 0);
    let mut hits = 0u64;
    for k in t.keys() {
        if matches!(w.demand_touch(k), crate::replay::Touch::Hit) {
            hits = hits.saturating_add(1);
        }
    }
    assert!(
        hits <= full.hits,
        "online oracle hits={hits} belady={}",
        full.hits
    );
}

#[test]
fn schedule_closed_loop_matches_sim_replay_hits() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0]), ev_seq(1, 0, 0, &[1])],
    };
    let p = HardwareProfile::parse("gpus=1\ncopy_engines=1\n").expect("profile");
    let cfg = SimCfg::lru(2, 4096, 0);
    let sim = sim_replay_cfg(&t, p.clone(), cfg).expect("sim");
    let sched = schedule_replay(&t, p, cfg, SchedCfg::closed(0)).expect("sched");
    assert_eq!(sched.replay.hits, sim.hits);
    assert_eq!(sched.replay.misses, sim.misses);
    assert_eq!(sched.completed, 2);
    assert_eq!(sched.queue_ns, Some(0));
}

#[test]
fn schedule_open_loop_idles_until_arrival() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0]), ev_seq(1, 0, 0, &[1])],
    };
    let p = HardwareProfile::parse("gpus=1\ncopy_engines=1\n").expect("profile");
    let gap = 50_000_000u64;
    let row = schedule_replay(
        &t,
        p,
        SimCfg::lru(2, 4096, 0),
        SchedCfg {
            interarrival_ns: gap,
            ..SchedCfg::closed(1)
        },
    )
    .expect("sched");
    assert!(
        row.replay.sim_ns >= gap,
        "wall={} gap={gap}",
        row.replay.sim_ns
    );
    assert!(row.idle_ns > 0);
    let ttft = row.replay.ttft_ns.expect("ttft");
    assert!(
        ttft < gap / 2,
        "TTFT must be from arrival, not t=0; ttft={ttft} gap={gap}"
    );
    assert_eq!(row.completed, 2);
    assert_eq!(row.queue_ns, Some(0));
}

#[test]
fn schedule_max_batch_serializes_running_set() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0]), ev_seq(1, 0, 0, &[1])],
    };
    let p = HardwareProfile::parse("gpus=1\ncopy_engines=1\n").expect("profile");
    let cfg = SimCfg {
        seq_streams: true,
        bytes_per_expert: 32u64 << 20,
        ..SimCfg::lru(2, 32u64 << 20, 0)
    };
    let all = schedule_replay(&t, p.clone(), cfg, SchedCfg::closed(0)).expect("all");
    let one = schedule_replay(&t, p, cfg, SchedCfg::closed(1)).expect("one");
    assert_eq!(all.replay.hits, one.replay.hits);
    assert!(
        one.replay.sim_ns > all.replay.sim_ns,
        "max_batch=1 must run sequences one at a time; one={} all={}",
        one.replay.sim_ns,
        all.replay.sim_ns
    );
    assert_eq!(all.queue_ns, Some(0));
    assert!(
        one.queue_ns.expect("queue") > 0,
        "max_batch=1 must queue the second sequence; queue={:?}",
        one.queue_ns
    );
}

#[test]
fn schedule_ttft_slo_misses_when_tight() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0]), ev_seq(0, 1, 0, &[1])],
    };
    let p = HardwareProfile::parse("gpus=1\ncopy_engines=1\n").expect("profile");
    let row = schedule_replay(
        &t,
        p,
        SimCfg::lru(2, 32u64 << 20, 0),
        SchedCfg {
            ttft_slo_ns: Some(1),
            itl_slo_ns: Some(1),
            ..SchedCfg::closed(0)
        },
    )
    .expect("sched");
    assert!(row.ttft_slo_miss > 0);
    assert!(row.itl_slo_miss > 0);
    assert_eq!(row.completed, 1);
}

#[test]
fn schedule_chunked_prefill_unblocks_short_decode() {
    let mut events = Vec::new();
    for layer in 0..8u32 {
        events.push(ev_seq(0, 0, layer, &[0]));
    }
    events.push(ev_seq(1, 0, 0, &[1]));
    events.push(ev_seq(1, 1, 0, &[1]));
    let t = Trace { events };
    let p = HardwareProfile::parse("gpus=1\ncopy_engines=1\n").expect("profile");
    let cfg = SimCfg {
        seq_streams: true,
        bytes_per_expert: 8u64 << 20,
        ..SimCfg::lru(4, 8u64 << 20, 0)
    };
    let fat = schedule_replay(&t, p.clone(), cfg, SchedCfg::closed(0)).expect("fat");
    let chunk = schedule_replay(&t, p, cfg, SchedCfg::chunked(0, 1)).expect("chunk");
    assert_eq!(fat.completed, 2);
    assert_eq!(chunk.completed, 2);
    let fat_ttft = fat.replay.ttft_ns.expect("fat ttft");
    let chunk_ttft = chunk.replay.ttft_ns.expect("chunk ttft");
    assert!(
        chunk_ttft < fat_ttft,
        "chunked prefill must let seq1 finish first token earlier; chunk={chunk_ttft} fat={fat_ttft}"
    );
}

#[test]
fn schedule_decode_first_shortens_mixed_itl() {
    let mut events = Vec::new();
    for layer in 0..16u32 {
        events.push(ev_seq(0, 0, layer, &[0]));
    }
    events.push(ev_seq(1, 0, 0, &[1]));
    for token in 1..5u32 {
        events.push(ev_seq(1, token, 0, &[1]));
    }
    let t = Trace { events };
    let p = HardwareProfile::parse("gpus=1\ncopy_engines=1\n").expect("profile");
    let cfg = SimCfg {
        seq_streams: true,
        bytes_per_expert: 8u64 << 20,
        ..SimCfg::lru(4, 8u64 << 20, 0)
    };
    let mixed = schedule_replay(&t, p.clone(), cfg, SchedCfg::chunked(0, 1)).expect("mixed");
    let prefer = schedule_replay(
        &t,
        p,
        cfg,
        SchedCfg {
            decode_first: true,
            ..SchedCfg::chunked(0, 1)
        },
    )
    .expect("prefer");
    assert_eq!(mixed.completed, 2);
    assert_eq!(prefer.completed, 2);
    let mixed_itl = mixed.replay.itl_ns.expect("mixed itl");
    let prefer_itl = prefer.replay.itl_ns.expect("prefer itl");
    assert!(
        prefer_itl < mixed_itl,
        "decode-first must not wait ITL on leftover prefill; prefer={prefer_itl} mixed={mixed_itl}"
    );
}

#[test]
fn schedule_decode_priority_shortens_mixed_itl() {
    let mut events = Vec::new();
    for layer in 0..16u32 {
        events.push(ev_seq(0, 0, layer, &[0]));
    }
    events.push(ev_seq(1, 0, 0, &[1]));
    for token in 1..5u32 {
        events.push(ev_seq(1, token, 0, &[1]));
    }
    let t = Trace { events };
    // Slow GEMM so leftover/decode compute is the critical path (example H100
    // hides those kernels under H2D, so decode-priority cannot shorten ITL).
    let p = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let off = SimCfg {
        seq_streams: true,
        compute_slots: 2,
        ..SimCfg::lru(4, 4096, 0)
    };
    let on = SimCfg {
        decode_priority: true,
        stream_priority: true,
        ..off
    };
    let mixed = schedule_replay(&t, p.clone(), off, SchedCfg::chunked(0, 1)).expect("mixed");
    let prefer = schedule_replay(&t, p, on, SchedCfg::chunked(0, 1)).expect("prefer");
    assert_eq!(mixed.completed, 2);
    assert_eq!(prefer.completed, 2);
    let mixed_itl = mixed.replay.itl_ns.expect("mixed itl");
    let prefer_itl = prefer.replay.itl_ns.expect("prefer itl");
    assert!(
        prefer_itl < mixed_itl,
        "decode-priority ITL must not wait leftover prefill; prefer={prefer_itl} mixed={mixed_itl}"
    );
}

#[test]
fn schedule_striped_homes_beat_gpu0_on_wide_token() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1, 2, 3, 4, 5, 6, 7])],
    };
    let p = HardwareProfile::example_8xh100_nvlink();
    let bytes = 4u64 << 20;
    let cfg = SimCfg::lru(8, bytes, 0);
    let gpu0 = schedule_replay(&t, p.clone(), cfg, SchedCfg::closed(0)).expect("gpu0");
    let map = striped(&t, 8);
    let ep = schedule_placed(&t, p, cfg, SchedCfg::closed(0), Some(&map)).expect("ep");
    assert_eq!(gpu0.completed, 1);
    assert_eq!(ep.completed, 1);
    assert!(
        ep.replay.sim_ns < gpu0.replay.sim_ns,
        "striped={} gpu0={}",
        ep.replay.sim_ns,
        gpu0.replay.sim_ns
    );
}

#[test]
fn schedule_placed_evicts_per_home_not_cluster() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[1]), ev(2, 0, &[0])],
    };
    let p = HardwareProfile::example_2node_rdma();
    let cfg = SimCfg::lru(1, 4096, 0);
    let map = striped(&t, 2);
    let gpu0 = schedule_replay(&t, p.clone(), cfg, SchedCfg::closed(0)).expect("gpu0");
    let ep = schedule_placed(&t, p.clone(), cfg, SchedCfg::closed(0), Some(&map)).expect("ep");
    let remote = schedule_remote(
        &t,
        p,
        cfg,
        SchedCfg::closed(0),
        &map,
        DECODE_ACTIVATION_BYTES,
    )
    .expect("remote");
    assert_eq!(gpu0.replay.hits, 0);
    assert_eq!(gpu0.replay.misses, 3);
    assert_eq!(ep.replay.hits, 1);
    assert_eq!(ep.replay.misses, 2);
    assert_eq!(remote.replay.hits, 1);
    assert_eq!(remote.replay.misses, 2);
}

#[test]
fn schedule_hot_replicas_move_more_bytes_than_stripe() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0]), ev(2, 0, &[0])],
    };
    let p = HardwareProfile::example_8xh100_nvlink();
    let bytes = 1u64 << 20;
    let cfg = SimCfg::lru(4, bytes, 0);
    let stripe = striped(&t, 8);
    let hot = with_hot_replicas(stripe.clone(), &t, 8, 200);
    let a =
        schedule_placed(&t, p.clone(), cfg, SchedCfg::closed(0), Some(&stripe)).expect("stripe");
    let b = schedule_placed(&t, p, cfg, SchedCfg::closed(0), Some(&hot)).expect("hot");
    assert!(
        b.replay.bytes_moved > a.replay.bytes_moved,
        "hot={} stripe={}",
        b.replay.bytes_moved,
        a.replay.bytes_moved
    );
}

#[test]
fn schedule_managed_hot_replicas_prefetch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0]), ev(2, 0, &[0])],
    };
    let p = HardwareProfile::example_8xh100_nvlink();
    let bytes = 1u64 << 20;
    let cfg = SimCfg {
        managed: true,
        ..SimCfg::lru(4, bytes, 0)
    };
    let stripe = striped(&t, 8);
    let hot = with_hot_replicas(stripe.clone(), &t, 8, 200);
    let a =
        schedule_placed(&t, p.clone(), cfg, SchedCfg::closed(0), Some(&stripe)).expect("stripe");
    let b = schedule_placed(&t, p, cfg, SchedCfg::closed(0), Some(&hot)).expect("hot");
    assert!(
        b.replay.bytes_moved > a.replay.bytes_moved,
        "managed hot={} stripe={}",
        b.replay.bytes_moved,
        a.replay.bytes_moved
    );
}

#[test]
fn schedule_replica_evict_frees_peer_hbm() {
    let t = Trace {
        events: vec![
            ev(0, 0, &[0]),
            ev(1, 0, &[0]),
            ev(2, 0, &[0]),
            ev(3, 0, &[0]),
            ev(4, 0, &[1]),
        ],
    };
    let p = HardwareProfile::example_2xh100_pcie().restrict_hbm(4096);
    let cfg = SimCfg::lru(1, 4096, 0);
    let stripe = striped(&t, 2);
    let hot = with_hot_replicas(stripe, &t, 2, 250);
    assert!(
        hot.replicas.contains_key(&ExpertKey::new(0, 0)),
        "{}",
        hot.line()
    );
    let row = schedule_placed(&t, p, cfg, SchedCfg::closed(0), Some(&hot)).expect("hbm");
    assert_eq!(row.completed, 1);
    assert!(row.replay.misses >= 2);
}

#[test]
fn schedule_vmm_replica_evict_frees_peer_hbm() {
    let t = Trace {
        events: vec![
            ev(0, 0, &[0]),
            ev(1, 0, &[0]),
            ev(2, 0, &[0]),
            ev(3, 0, &[0]),
            ev(4, 0, &[1]),
        ],
    };
    let p = HardwareProfile::example_2xh100_pcie().restrict_hbm(4096);
    let cfg = SimCfg {
        vmm: true,
        ..SimCfg::lru(1, 4096, 0)
    };
    let stripe = striped(&t, 2);
    let hot = with_hot_replicas(stripe, &t, 2, 250);
    assert!(
        hot.replicas.contains_key(&ExpertKey::new(0, 0)),
        "{}",
        hot.line()
    );
    let row = schedule_placed(&t, p, cfg, SchedCfg::closed(0), Some(&hot)).expect("hbm");
    assert_eq!(row.completed, 1);
    assert!(row.replay.misses >= 2);
}

#[test]
fn schedule_hbm_evicts_when_slots_are_loose() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[2]), ev(2, 0, &[0])],
    };
    let p = HardwareProfile::example_2xh100_pcie().restrict_hbm(4096);
    let cfg = SimCfg::lru(8, 4096, 0);
    let map = striped(&t, 2);
    let row = schedule_placed(&t, p, cfg, SchedCfg::closed(0), Some(&map)).expect("hbm");
    assert_eq!(row.completed, 1);
    assert_eq!(row.replay.misses, 3);
    assert_eq!(row.replay.hits, 0);
}

#[test]
fn schedule_remote_hbm_evicts_when_slots_are_loose() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[2]), ev(2, 0, &[0])],
    };
    let p = HardwareProfile::example_2xh100_pcie().restrict_hbm(4096);
    let cfg = SimCfg::lru(8, 4096, 0);
    let map = striped(&t, 2);
    let row = schedule_remote(
        &t,
        p,
        cfg,
        SchedCfg::closed(0),
        &map,
        DECODE_ACTIVATION_BYTES,
    )
    .expect("hbm");
    assert_eq!(row.completed, 1);
    assert_eq!(row.replay.misses, 3);
}

#[test]
fn schedule_managed_remote_hbm_evicts_when_slots_are_loose() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[2]), ev(2, 0, &[0])],
    };
    let p = HardwareProfile::example_2xh100_pcie().restrict_hbm(4096);
    let cfg = SimCfg {
        managed: true,
        ..SimCfg::lru(8, 4096, 0)
    };
    let map = striped(&t, 2);
    let row = schedule_remote(
        &t,
        p,
        cfg,
        SchedCfg::closed(0),
        &map,
        DECODE_ACTIVATION_BYTES,
    )
    .expect("hbm");
    assert_eq!(row.completed, 1);
    assert_eq!(row.replay.misses, 3);
}

#[test]
fn schedule_remote_pays_peer_copy_on_rdma() {
    let t = Trace {
        events: vec![ev(0, 0, &[1])],
    };
    let p = HardwareProfile::example_2node_rdma();
    let bytes = 1u64 << 20;
    let cfg = SimCfg::lru(4, bytes, 0);
    let map = striped(&t, 2);
    let local =
        schedule_placed(&t, p.clone(), cfg, SchedCfg::closed(0), Some(&map)).expect("local");
    let remote = schedule_remote(
        &t,
        p.clone(),
        cfg,
        SchedCfg::closed(0),
        &map,
        DECODE_ACTIVATION_BYTES,
    )
    .expect("remote");
    assert_eq!(local.completed, 1);
    assert_eq!(remote.completed, 1);
    assert!(
        remote.replay.bytes_moved > local.replay.bytes_moved,
        "remote={} local={}",
        remote.replay.bytes_moved,
        local.replay.bytes_moved
    );
    assert!(
        remote.replay.sim_ns > local.replay.sim_ns,
        "remote={} local={}",
        remote.replay.sim_ns,
        local.replay.sim_ns
    );
    assert!(
        remote.replay.bytes_moved < bytes.saturating_mul(2),
        "dispatch should not D2D the full expert; moved={}",
        remote.replay.bytes_moved
    );
    let moved = schedule_remote(&t, p, cfg, SchedCfg::closed(0), &map, bytes).expect("move");
    assert!(
        moved.replay.bytes_moved >= bytes.saturating_mul(2),
        "equal act vs expert volume must move weights; moved={}",
        moved.replay.bytes_moved
    );
    assert!(moved.replay.bytes_moved > remote.replay.bytes_moved);
}

#[test]
fn schedule_managed_remote_reads_without_weight_d2d() {
    let t = Trace {
        events: vec![ev(0, 0, &[1])],
    };
    let p = HardwareProfile::example_2xh100_pcie();
    let bytes = 1u64 << 20;
    let map = striped(&t, 2);
    let pin = SimCfg::lru(4, bytes, 0);
    let um = SimCfg {
        managed: true,
        ..pin
    };
    let moved = schedule_remote(&t, p.clone(), pin, SchedCfg::closed(0), &map, bytes).expect("d2d");
    let remote = schedule_remote(&t, p, um, SchedCfg::closed(0), &map, bytes).expect("um");
    assert_eq!(moved.completed, 1);
    assert_eq!(remote.completed, 1);
    assert_eq!(moved.replay.misses, remote.replay.misses);
    assert!(
        remote.replay.bytes_moved < moved.replay.bytes_moved,
        "managed={} d2d={}",
        remote.replay.bytes_moved,
        moved.replay.bytes_moved
    );
    assert!(
        remote.replay.bytes_moved < bytes.saturating_mul(2),
        "preferred-location GEMM must not D2D the expert; moved={}",
        remote.replay.bytes_moved
    );
}

#[test]
fn schedule_remote_prefetch_hits_copy_forward_layer() {
    let t = Trace {
        events: vec![ev(0, 0, &[1]), ev(0, 1, &[1])],
    };
    let p = HardwareProfile::example_2node_rdma();
    let bytes = 1u64 << 20;
    let map = striped(&t, 2);
    let none = schedule_remote(
        &t,
        p.clone(),
        SimCfg::lru(4, bytes, 0),
        SchedCfg::closed(0),
        &map,
        DECODE_ACTIVATION_BYTES,
    )
    .expect("none");
    let fwd = schedule_remote(
        &t,
        p,
        SimCfg {
            prefetch: Prefetch::CopyForward,
            ..SimCfg::lru(4, bytes, 0)
        },
        SchedCfg::closed(0),
        &map,
        DECODE_ACTIVATION_BYTES,
    )
    .expect("fwd");
    assert_eq!(none.replay.prefetches, 0);
    assert_eq!(none.replay.misses, 2);
    assert!(fwd.replay.prefetches >= 1, "{}", fwd.replay.line());
    assert!(fwd.replay.prefetch_hits >= 1, "{}", fwd.replay.line());
    assert!(
        fwd.replay.misses < none.replay.misses,
        "fwd misses={} none={}",
        fwd.replay.misses,
        none.replay.misses
    );
}

#[test]
fn schedule_remote_hit_reuses_resident_page() {
    let t = Trace {
        events: vec![ev(0, 0, &[1]), ev(1, 0, &[1])],
    };
    let p = HardwareProfile::example_2node_rdma();
    let bytes = 1u64 << 20;
    let cfg = SimCfg::lru(4, bytes, 0);
    let map = striped(&t, 2);
    let row = schedule_remote(
        &t,
        p,
        cfg,
        SchedCfg::closed(0),
        &map,
        DECODE_ACTIVATION_BYTES,
    )
    .expect("remote");
    assert_eq!(row.completed, 1);
    assert_eq!(row.replay.misses, 1);
    assert_eq!(row.replay.hits, 1);
}

#[test]
fn schedule_remote_evicts_when_slots_are_tight() {
    let t = cycling_trace();
    let p = HardwareProfile::example_2node_rdma();
    let map = striped(&t, 2);
    let row = schedule_remote(
        &t,
        p,
        SimCfg::lru(1, 4096, 0),
        SchedCfg::closed(0),
        &map,
        DECODE_ACTIVATION_BYTES,
    )
    .expect("remote");
    assert_eq!(row.completed, 1);
    assert!(row.replay.misses >= 3);
}

#[test]
fn schedule_slo_reject_drops_late_head_of_line() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0]), ev_seq(1, 0, 0, &[1])],
    };
    let p = HardwareProfile::parse("gpus=1\ncopy_engines=1\n").expect("profile");
    let cfg = SimCfg::lru(2, 8u64 << 20, 0);
    let keep = schedule_replay(
        &t,
        p.clone(),
        cfg,
        SchedCfg {
            ttft_slo_ns: Some(1),
            ..SchedCfg::closed(1)
        },
    )
    .expect("keep");
    let drop = schedule_replay(
        &t,
        p,
        cfg,
        SchedCfg {
            ttft_slo_ns: Some(1),
            slo_reject: true,
            ..SchedCfg::closed(1)
        },
    )
    .expect("drop");
    assert_eq!(keep.completed, 2);
    assert_eq!(keep.rejected, 0);
    assert!(keep.ttft_slo_miss > 0);
    assert_eq!(drop.completed, 1);
    assert_eq!(drop.rejected, 1);
}

#[test]
fn schedule_retires_finished_sequences() {
    let t = Trace {
        events: vec![
            ev_seq(0, 0, 0, &[0]),
            ev_seq(0, 1, 0, &[0]),
            ev_seq(1, 0, 0, &[1]),
        ],
    };
    let p = HardwareProfile::parse("gpus=1\ncopy_engines=1\n").expect("profile");
    let row = schedule_replay(&t, p, SimCfg::lru(2, 4096, 0), SchedCfg::closed(1)).expect("sched");
    assert_eq!(row.completed, 2);
    assert_eq!(row.replay.hits.saturating_add(row.replay.misses), 3);
}

fn shared_prefix_two_seq(layers: u32) -> Trace {
    let pfx = prefix_hash(&[1]);
    let mut events = Vec::new();
    for seq in 0..2u64 {
        for _ in 0..layers {
            events.push(ExpertAccess {
                sequence: seq,
                token: 0,
                layer: 0,
                experts: vec![u32::try_from(seq).unwrap_or(0)],
                weight_pt: Vec::new(),
                prefix: Some(pfx),
            });
        }
    }
    Trace { events }
}

#[test]
fn schedule_prefix_cache_hits_after_first_sequence() {
    let t = shared_prefix_two_seq(4);
    let p = HardwareProfile::parse("gpus=1\ncopy_engines=1\n").expect("profile");
    let cfg = SimCfg::lru(1, 4096, 0);
    let off = schedule_replay(&t, p.clone(), cfg, SchedCfg::closed(1)).expect("off");
    assert_eq!(off.prefix_hits, 0);
    assert_eq!(off.replay.misses, 2);
    let on = schedule_replay(
        &t,
        p.clone(),
        cfg,
        SchedCfg {
            prefix_cache: true,
            ..SchedCfg::closed(1)
        },
    )
    .expect("on");
    assert_eq!(on.prefix_hits, 4);
    assert_eq!(on.replay.misses, 1);
    let concurrent = schedule_replay(
        &t,
        p.clone(),
        cfg,
        SchedCfg {
            prefix_cache: true,
            ..SchedCfg::closed(0)
        },
    )
    .expect("concurrent");
    assert_eq!(concurrent.prefix_hits, 0);
    assert_eq!(concurrent.replay.misses, 2);
    let chunked = schedule_replay(
        &t,
        p,
        cfg,
        SchedCfg {
            prefix_cache: true,
            prefill_chunk_layers: 1,
            ..SchedCfg::closed(1)
        },
    )
    .expect("chunk");
    assert_eq!(chunked.prefix_hits, 4);
    assert_eq!(chunked.replay.misses, 1);
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
fn simulated_gpu_store_managed_prefetches_on_miss() {
    let t = cycling_trace();
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::with_managed(inner, 2, HardwareProfile::example_h100_sxm(), 4096)
            .expect("gpu");
    assert!(gpu.uses_managed());
    assert!(gpu.staging_is_pinned());
    let k0 = ExpertKey::new(0, 0);
    let parts = gpu.acquire(k0).expect("acq");
    assert_eq!(parts.gate, vec![1]);
    let score = gpu.score().expect("score");
    assert!(score.bytes_moved >= 4096);
    assert!(score.hbm_peak >= 4096);
    gpu.evict(k0).expect("evict");
    assert!(!gpu.is_resident(k0));
    assert_eq!(gpu.phase(k0), ExpertPhase::Cold);
}

#[test]
fn simulated_gpu_store_managed_pin_replicates_by_prefetch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[1])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::with_managed(inner, 2, HardwareProfile::example_8xh100_nvlink(), 4096)
            .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    gpu.pin_hot(&[k0]).expect("pin");
    let _p = gpu.acquire(k0).expect("hit");
    let score = gpu.score().expect("score");
    assert!(score.bytes_moved >= 8192, "{}", score.bytes_moved);
}

#[test]
fn simulated_gpu_store_managed_migrate_drops_source_copy() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::with_managed(inner, 1, HardwareProfile::example_2node_rdma(), 4096)
            .expect("gpu");
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
fn simulated_gpu_store_mapped_skips_hbm_and_h2d() {
    let t = cycling_trace();
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::with_mapped(inner, 2, HardwareProfile::example_h100_sxm(), 4096)
            .expect("gpu");
    assert!(gpu.uses_mapped());
    let k0 = ExpertKey::new(0, 0);
    let parts = gpu.acquire(k0).expect("acq");
    assert_eq!(parts.gate, vec![1]);
    let score = gpu.score().expect("score");
    assert_eq!(score.bytes_moved, 0);
    assert_eq!(score.hbm_peak, 0);
    gpu.evict(k0).expect("evict");
    assert!(!gpu.is_resident(k0));
}

#[test]
fn simulated_gpu_store_vmm_h2d_and_release() {
    let t = cycling_trace();
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_vmm(inner, 2, HardwareProfile::example_h100_sxm(), 4096)
        .expect("gpu");
    assert!(gpu.uses_vmm());
    let k0 = ExpertKey::new(0, 0);
    let parts = gpu.acquire(k0).expect("acq");
    assert_eq!(parts.gate, vec![1]);
    let score = gpu.score().expect("score");
    assert!(score.bytes_moved >= 4096);
    assert!(score.hbm_peak >= 4096);
    gpu.evict(k0).expect("evict");
    assert!(!gpu.is_resident(k0));
    assert_eq!(gpu.phase(k0), ExpertPhase::Cold);
}

#[test]
fn simulated_gpu_store_mapped_migrate_retargets_gemm() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::with_mapped(inner, 1, HardwareProfile::example_2node_rdma(), 4096)
            .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    let _p = gpu.acquire(k0).expect("acq");
    assert_eq!(gpu.device_of(k0), Some(DeviceId(0)));
    gpu.migrate(k0, DeviceId(1)).expect("mig");
    assert_eq!(gpu.device_of(k0), Some(DeviceId(1)));
    let _p = gpu.acquire(k0).expect("gemm dest");
    let score = gpu.score().expect("score");
    assert_eq!(score.bytes_moved, 0);
    assert_eq!(score.hbm_peak, 0);
}

#[test]
fn simulated_gpu_store_vmm_migrate_maps_dest() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::with_vmm(inner, 1, HardwareProfile::example_2node_rdma(), 4096)
            .expect("gpu");
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
    gpu.release(k0);
    assert!(gpu.is_pinned(k0));
    assert_eq!(gpu.phase(k0), ExpertPhase::Leased);
    assert!(gpu.is_resident(k0));
    assert_eq!(gpu.replica_of(k0), Some(DeviceId(1)));
    let score = gpu.score().expect("score");
    assert!(gpu.page_resident(k0, DeviceId(1)));
    assert_eq!(gpu.metrics().replicates, 1);
    assert!(score.bytes_moved >= 4096);
}

#[test]
fn simulated_gpu_store_multicast_pin_hot_nvls() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_cfg(
        inner,
        2,
        HardwareProfile::example_8xh100_nvlink(),
        4096,
        GpuFill::Vmm,
        GpuStoreCfg {
            multicast: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    gpu.pin_hot(&[k0]).expect("pin");
    assert_eq!(gpu.replica_of(k0), Some(DeviceId(1)));
    assert!(gpu.page_resident(k0, DeviceId(1)));
    assert_eq!(gpu.metrics().replicates, 1);
    let score = gpu.score().expect("score");
    assert!(score.bytes_moved >= 4096, "{}", score.bytes_moved);
}

#[test]
fn simulated_gpu_store_pin_hot_replicates_onto_next_home() {
    let t = Trace {
        events: vec![ev(0, 0, &[3])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::new(inner, 2, HardwareProfile::example_8xh100_nvlink(), 4096)
        .expect("gpu");
    let k = ExpertKey::new(0, 3);
    gpu.pin_hot(&[k]).expect("pin");
    assert_eq!(gpu.device_of(k), Some(DeviceId(3)));
    assert_eq!(gpu.replica_of(k), Some(DeviceId(4)));
    assert_eq!(gpu.metrics().replicates, 1);
    let score = gpu.score().expect("score");
    assert!(gpu.page_resident(k, DeviceId(4)));
    assert!(score.bytes_moved >= 8192, "{}", score.bytes_moved);
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
    assert!(rows.iter().any(|r| r.name == "batch-1"));
    assert!(rows.iter().any(|r| r.name == "batch-128"));
    let batch1 = rows.iter().find(|r| r.name == "batch-1").unwrap();
    assert!(
        batch1.overlap.is_none(),
        "batch-1 is one sequence, {}",
        batch1.render()
    );
    let batch128 = rows.iter().find(|r| r.name == "batch-128").unwrap();
    assert!(batch128.overlap.is_some(), "{}", batch128.render());
    assert!(batch128.schedule.is_some(), "{}", batch128.render());
    assert!(batch128.render().contains("schedule-all"));
    assert!(batch128.render().contains("schedule-1"));
    let batch = rows.iter().find(|r| r.name == "batch").unwrap();
    assert!(batch.overlap.is_some(), "{}", batch.render());
    assert!(batch.render().contains("overlap"));
    assert!(batch.graphs.is_some(), "{}", batch.render());
    assert!(batch.render().contains("graphs"));
    assert!(batch.schedule.is_some(), "{}", batch.render());
    assert!(batch.render().contains("schedule-all"));
    assert!(batch.render().contains("schedule-1"));
    assert!(batch.render().contains("sim-managed"), "{}", batch.render());
    assert!(batch.render().contains("sim-vmm"), "{}", batch.render());
    assert!(batch.render().contains("sim-vmmpage"), "{}", batch.render());
    assert!(batch.render().contains("sim-hostfn"), "{}", batch.render());
    assert!(
        batch.render().contains("sim-blockstrm"),
        "{}",
        batch.render()
    );
    let mixed = rows.iter().find(|r| r.name == "prefill-batch").unwrap();
    assert!(mixed.chunk.is_some(), "{}", mixed.render());
    assert!(mixed.render().contains("schedule-chunk1"));
    let shared = rows.iter().find(|r| r.name == "shared-prefix").unwrap();
    assert!(
        shared.render().contains("schedule-prefix"),
        "{}",
        shared.render()
    );
}

#[test]
fn batch_1_vs_128_widths_and_schedule() {
    let b1 = generate(Workload::Batch1, 4, 8, 1, 1);
    let b8 = generate(Workload::Batch, 4, 8, 1, 1);
    let b128 = generate(Workload::Batch128, 4, 8, 1, 1);
    assert_eq!(Workload::Batch1.concurrent_seqs(), 1);
    assert_eq!(Workload::Batch128.concurrent_seqs(), 128);
    assert_eq!(b1.n_sequences(), 1);
    assert_eq!(b8.n_sequences(), 8);
    assert_eq!(b128.n_sequences(), 128);
    assert_eq!(b1.events.len(), 4);
    assert_eq!(b8.events.len(), 32);
    assert_eq!(b128.events.len(), 512);
    let p = HardwareProfile::example_cheap_48gb();
    let cfg = SimCfg::lru(4, 4096, 0);
    let all = schedule_replay(&b128, p.clone(), cfg, SchedCfg::closed(0)).expect("all");
    let one = schedule_replay(&b128, p.clone(), cfg, SchedCfg::closed(1)).expect("one");
    assert_eq!(all.completed, 128);
    assert_eq!(one.completed, 128);
    assert!(all.replay.sim_ns > 0, "{}", all.replay.line());
    assert!(one.replay.sim_ns > 0, "{}", one.replay.line());
    let row1 = report("batch-1", &b1, 4, 8, Some(p.clone()), 4096).expect("r1");
    assert!(
        row1.schedule.is_none(),
        "batch-1 is one sequence so schedule-all vs schedule-1 is skipped, {}",
        row1.render()
    );
    let row128 = report("batch-128", &b128, 4, 8, Some(p), 4096).expect("r128");
    assert!(row128.schedule.is_some(), "{}", row128.render());
    assert!(row128.render().contains("schedule-all"));
    assert!(row128.render().contains("schedule-1"));
}

#[test]
fn checked_in_two_layer_schedule_and_copy_forward() {
    let t = load_checked_in_trace("tiny-qwen3moe-2layer.jsonl");
    assert!(t.events.iter().any(|e| e.layer == 1));
    assert_eq!(t.n_sequences(), 1);
    let p = HardwareProfile::example_cheap_48gb();
    let sch = schedule_replay(&t, p.clone(), SimCfg::lru(4, 4096, 8), SchedCfg::closed(0))
        .expect("schedule");
    assert_eq!(sch.completed, 1);
    assert!(sch.replay.sim_ns > 0, "{}", sch.replay.line());
    let cfg = |prefetch: Prefetch| SimCfg {
        prefetch,
        ..SimCfg::lru(4, 4096, 8)
    };
    let none = sim_replay_cfg(&t, p.clone(), cfg(Prefetch::None)).expect("none");
    let fwd = sim_replay_cfg(&t, p.clone(), cfg(Prefetch::CopyForward)).expect("fwd");
    assert!(
        fwd.prefetches > none.prefetches,
        "2-layer copy-forward must fill L+1, fwd={} none={}",
        fwd.prefetches,
        none.prefetches
    );
    assert!(
        fwd.prefetch_hits > 0,
        "L0 [1,2] → L1 [1,2] must hit a copy-forward fill, {}",
        fwd.line()
    );
    let store = store_replay_cfg(
        &t,
        p,
        StoreReplayCfg {
            prefetch: Prefetch::CopyForward,
            ..StoreReplayCfg::demand(2, 4096, GpuFill::Pinned)
        },
    )
    .expect("store");
    assert!(
        store.metrics.prefetches > 0,
        "slots=2 must evict L+1 so copy-forward refills, {}",
        store.line()
    );
    assert!(store.score.wall_ns > 0, "{}", store.line());
}

#[test]
fn shared_prefix_workload_reuses_token0_hash() {
    let t = generate(Workload::SharedPrefix, 4, 8, 1, 1);
    let p0: Vec<Option<u64>> = t
        .events
        .iter()
        .filter(|e| e.token == 0)
        .map(|e| e.prefix)
        .collect();
    assert!(p0.len() >= 8);
    let first = p0.first().copied().flatten().expect("token0 p");
    assert!(p0.iter().all(|p| *p == Some(first)));
    let later = t.events.iter().find(|e| e.token > 0).expect("decode");
    assert_ne!(later.prefix, Some(first));
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
    assert!(live.is_pinned(k));
    live.release(k);
    assert!(live.is_pinned(k));
    live.place_hot(k, 1, 2).expect("cpu place");
    assert_eq!(live.metrics().dispatches, 0);
    assert_eq!(live.metrics().migrates, 0);
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
    assert_eq!(gpu.metrics().migrates, 1);
    gpu.migrate(k0, DeviceId(1)).expect("already dest");
    assert_eq!(gpu.metrics().migrates, 1);
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
fn place_hot_dispatches_when_activations_are_cheaper() {
    let t = Trace {
        events: vec![ev(0, 0, &[1])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::new(inner, 1, HardwareProfile::example_2node_rdma(), 4096).expect("gpu");
    let k = ExpertKey::new(0, 1);
    let _p = gpu.acquire(k).expect("acq");
    gpu.pin_hot(&[k]).expect("pin");
    assert_eq!(gpu.device_of(k), Some(DeviceId(1)));
    gpu.place_hot(k, 1, 2).expect("place");
    assert_eq!(gpu.metrics().dispatches, 1);
    assert_eq!(gpu.metrics().migrates, 0);
    assert_eq!(gpu.device_of(k), Some(DeviceId(1)));
}

#[test]
fn place_hot_migrates_when_weights_are_cheaper() {
    let t = Trace {
        events: vec![ev(0, 0, &[1])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::new(inner, 1, HardwareProfile::example_2node_rdma(), 64).expect("gpu");
    let k = ExpertKey::new(0, 1);
    let _p = gpu.acquire(k).expect("acq");
    gpu.pin_hot(&[k]).expect("pin");
    gpu.place_hot(k, 1, 2).expect("place");
    assert_eq!(gpu.metrics().migrates, 1);
    assert_eq!(gpu.metrics().dispatches, 0);
    assert_eq!(gpu.device_of(k), Some(DeviceId(0)));
}

#[test]
fn place_hot_migrates_when_reuse_crosses_over() {
    let t = Trace {
        events: vec![ev(0, 0, &[1])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::new(inner, 1, HardwareProfile::example_2node_rdma(), 4096).expect("gpu");
    let k = ExpertKey::new(0, 1);
    let _p = gpu.acquire(k).expect("acq");
    gpu.pin_hot(&[k]).expect("pin");
    gpu.place_hot(k, 16, 2).expect("place");
    assert_eq!(gpu.metrics().migrates, 1);
    assert_eq!(gpu.metrics().dispatches, 0);
    assert_eq!(gpu.device_of(k), Some(DeviceId(0)));
}

#[test]
fn place_hot_single_gpu_is_noop() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::new(inner, 1, HardwareProfile::example_h100_sxm(), 64).expect("gpu");
    let k = ExpertKey::new(0, 0);
    let _p = gpu.acquire(k).expect("acq");
    gpu.place_hot(k, 1, 2).expect("place");
    assert_eq!(gpu.metrics().dispatches, 0);
    assert_eq!(gpu.metrics().migrates, 0);
}

#[test]
fn managed_pin_after_gemm_replicates() {
    let t = Trace {
        events: vec![ev(0, 0, &[1])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::with_managed(inner, 2, HardwareProfile::example_8xh100_nvlink(), 4096)
            .expect("gpu");
    let k = ExpertKey::new(0, 1);
    let _p = gpu.acquire(k).expect("acq");
    assert_eq!(gpu.device_of(k), Some(DeviceId(1)));
    gpu.pin_hot(&[k]).expect("pin after gemm");
    assert!(
        gpu.replica_of(k).is_some(),
        "replica after GEMM lease drain"
    );
}

#[test]
fn managed_place_hot_migrates_after_gemm() {
    let t = Trace {
        events: vec![ev(0, 0, &[1])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::with_managed(inner, 2, HardwareProfile::example_8xh100_nvlink(), 64)
            .expect("gpu");
    let k = ExpertKey::new(0, 1);
    let _p = gpu.acquire(k).expect("acq");
    gpu.pin_hot(&[k]).expect("pin");
    gpu.place_hot(k, 1, 2).expect("place");
    assert_eq!(gpu.metrics().migrates, 1);
    assert_eq!(gpu.device_of(k), Some(DeviceId(0)));
}

#[test]
fn vmm_place_hot_migrates_after_gemm() {
    let t = Trace {
        events: vec![ev(0, 0, &[1])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::with_vmm(inner, 2, HardwareProfile::example_8xh100_nvlink(), 64)
            .expect("gpu");
    let k = ExpertKey::new(0, 1);
    let _p = gpu.acquire(k).expect("acq");
    gpu.pin_hot(&[k]).expect("pin");
    gpu.place_hot(k, 1, 2).expect("place");
    assert_eq!(gpu.metrics().migrates, 1);
    assert_eq!(gpu.device_of(k), Some(DeviceId(0)));
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

#[test]
fn plan_window_stay_skips_harmful_copy_forward() {
    let mut events = Vec::new();
    for tok in 0..16u32 {
        events.push(ev(tok, 0, &[0]));
        events.push(ev(tok, 1, &[1]));
    }
    let t = Trace { events };
    let p = HardwareProfile::example_h100_sxm();
    let fwd = sim_replay_cfg(
        &t,
        p.clone(),
        SimCfg {
            slots: 2,
            prefetch: Prefetch::CopyForward,
            ..SimCfg::lru(2, 4096, 8)
        },
    )
    .expect("fwd");
    let planned = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            slots: 2,
            prefetch: Prefetch::CopyForward,
            plan_window: 8,
            plan_threshold: 500,
            ..SimCfg::lru(2, 4096, 8)
        },
    )
    .expect("plan");
    assert!(
        planned.hits > fwd.hits,
        "stay planner={} copy-forward={}",
        planned.hits,
        fwd.hits
    );
    assert!(
        planned.prefetches < fwd.prefetches,
        "stay prefetches={} copy-forward={}",
        planned.prefetches,
        fwd.prefetches
    );
}

#[test]
fn cuda_graphs_amortize_repeated_expert_gemms() {
    let mut events = Vec::new();
    for tok in 0..12u32 {
        events.push(ev(tok, 0, &[0, 1, 2, 3]));
    }
    let t = Trace { events };
    let p = HardwareProfile::parse(
        "gpus=1\nlaunch_overhead_ns=50000\ngraph_launch_ns=4000\ncopy_engines=2\n",
    )
    .expect("profile");
    let base = SimCfg {
        slots: 4,
        bytes_per_expert: 4096,
        lookahead: 0,
        ..SimCfg::lru(4, 4096, 0)
    };
    let serial = sim_replay_cfg(&t, p.clone(), base).expect("serial");
    let mut graphs = base;
    graphs.cuda_graphs = true;
    let g = sim_replay_cfg(&t, p, graphs).expect("graphs");
    assert!(
        g.graph_launches > 0,
        "expected captured launches; line={}",
        g.line()
    );
    assert!(
        g.sim_ns < serial.sim_ns,
        "graphs={} serial={}",
        g.sim_ns,
        serial.sim_ns
    );
    assert!(g.child_graphs > 0, "expected parent child-graph capture");
    assert!(g.line().contains("graph_launches="));
    assert!(g.line().contains("child_graphs="));
}

#[test]
fn cuda_graphs_reuse_leaf_graphs_across_combos() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1]), ev(1, 0, &[0, 2])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nlaunch_overhead_ns=50000\ngraph_launch_ns=4000\ncopy_engines=2\n",
    )
    .expect("profile");
    let mut cfg = SimCfg::lru(4, 4096, 0);
    cfg.cuda_graphs = true;
    let g = sim_replay_cfg(&t, p, cfg).expect("graphs");
    assert_eq!(g.graph_launches, 2);
    assert_eq!(g.child_graphs, 2);
}

#[test]
fn cuda_graphs_graph_update_reuses_parked_leaves() {
    let t = Trace {
        events: vec![
            ev(0, 0, &[0]),
            ev(1, 0, &[1]),
            ev(2, 0, &[0]),
            ev(3, 0, &[1]),
        ],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=100000\ngraph_update_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let base = SimCfg {
        cuda_graphs: true,
        ..SimCfg::lru(1, 4096, 0)
    };
    let inst = sim_replay_cfg(&t, p.clone(), base).expect("inst");
    let upd = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            graph_update: true,
            ..base
        },
    )
    .expect("upd");
    assert_eq!(inst.hits, upd.hits);
    assert_eq!(inst.misses, upd.misses);
    assert_eq!(inst.graph_updates, 0);
    assert_eq!(upd.graph_updates, 3);
    assert!(
        upd.sim_ns < inst.sim_ns,
        "update={} instantiate={}",
        upd.sim_ns,
        inst.sim_ns
    );
    assert!(upd.line().contains("graph_updates="));
}

#[test]
fn retarget_parked_kernel_patches_unique_memcpy() {
    use crate::sim_replay::retarget_parked_kernel;
    use gpu_sim::{KernelKind, MemcpyOp, Place, Sim, StreamId};

    let mut sim = Sim::new(HardwareProfile::example_h100_sxm());
    let d = DeviceId(0);
    let s = StreamId(0);
    let a = sim.malloc(d, 4096).expect("a");
    let b = sim.malloc(d, 4096).expect("b");
    let exec = sim.create_graph(d, s).expect("g");
    sim.graph_add_kernel(exec, KernelKind::other(8, 8), &[a], &[a])
        .expect("k");
    sim.graph_add_memcpy(
        exec,
        MemcpyOp {
            src: Place::HostPinned,
            dst: Place::Device(d),
            alloc: a,
            bytes: 4096,
            offset: 0,
        },
    )
    .expect("m");
    sim.instantiate_graph(exec).expect("i");
    retarget_parked_kernel(&mut sim, exec, b).expect("retarget");
    let (_, params) = sim.graph_unique_kernel(exec).expect("k2");
    let read = params.reads.first().expect("read");
    let write = params.writes.first().expect("write");
    assert_eq!(read.id, b);
    assert_eq!(write.id, b);
    let (_, mop) = sim.graph_unique_memcpy(exec).expect("m2");
    assert_eq!(mop.alloc, b);
}

#[test]
fn retarget_parked_kernel_patches_unique_memset() {
    use crate::sim_replay::retarget_parked_kernel;
    use gpu_sim::{KernelBuf, KernelKind, Sim, StreamId};

    let mut sim = Sim::new(HardwareProfile::example_h100_sxm());
    let d = DeviceId(0);
    let s = StreamId(0);
    let a = sim.malloc(d, 4096).expect("a");
    let b = sim.malloc(d, 4096).expect("b");
    let exec = sim.create_graph(d, s).expect("g");
    sim.graph_add_kernel(exec, KernelKind::other(8, 8), &[a], &[a])
        .expect("k");
    sim.graph_add_memset(exec, KernelBuf::whole(a)).expect("z");
    sim.instantiate_graph(exec).expect("i");
    retarget_parked_kernel(&mut sim, exec, b).expect("retarget");
    let (_, params) = sim.graph_unique_kernel(exec).expect("k2");
    let read = params.reads.first().expect("read");
    let write = params.writes.first().expect("write");
    assert_eq!(read.id, b);
    assert_eq!(write.id, b);
    let (_, zbuf) = sim.graph_unique_memset(exec).expect("z2");
    assert_eq!(zbuf.id, b);
}

#[test]
fn cuda_graphs_graph_set_params_reuses_parked_leaves() {
    let t = Trace {
        events: vec![
            ev(0, 0, &[0]),
            ev(1, 0, &[1]),
            ev(2, 0, &[0]),
            ev(3, 0, &[1]),
        ],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=100000\ngraph_update_ns=1000\ngraph_set_params_ns=100\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let base = SimCfg {
        cuda_graphs: true,
        ..SimCfg::lru(1, 4096, 0)
    };
    let inst = sim_replay_cfg(&t, p.clone(), base).expect("inst");
    let upd = sim_replay_cfg(
        &t,
        p.clone(),
        SimCfg {
            graph_update: true,
            ..base
        },
    )
    .expect("upd");
    let set = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            graph_set_params: true,
            ..base
        },
    )
    .expect("set");
    assert_eq!(inst.hits, set.hits);
    assert_eq!(inst.misses, set.misses);
    assert_eq!(inst.graph_set_params, 0);
    assert_eq!(set.graph_updates, 0);
    assert_eq!(set.graph_set_params, 3);
    assert!(
        set.sim_ns < upd.sim_ns,
        "set_params={} update={}",
        set.sim_ns,
        upd.sim_ns
    );
    assert!(set.line().contains("graph_set_params="));
}

#[test]
fn cuda_graphs_graph_set_params_works_with_mem_nodes() {
    let t = Trace {
        events: vec![
            ev(0, 0, &[0]),
            ev(1, 0, &[1]),
            ev(2, 0, &[0]),
            ev(3, 0, &[1]),
        ],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=100000\ngraph_update_ns=1000\ngraph_set_params_ns=100\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let set = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            cuda_graphs: true,
            graph_set_params: true,
            graph_mem: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    )
    .expect("set");
    assert_eq!(set.graph_updates, 0);
    assert_eq!(set.graph_set_params, 3);
    assert!(set.graph_launches > 0, "line={}", set.line());
}

#[test]
fn graph_update_and_graph_set_params_conflict() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let err = sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            cuda_graphs: true,
            graph_update: true,
            graph_set_params: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    )
    .expect_err("conflict");
    assert!(
        err.to_string()
            .contains("choose one of graph-update, graph-set-params"),
        "{err}"
    );
}

#[test]
fn simulated_gpu_store_graph_update_reuses_parked_exec() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[1])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=100000\ngraph_update_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let run = |graph_update: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            1,
            p.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                graph_update,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        let k0 = ExpertKey::new(0, 0);
        let k1 = ExpertKey::new(0, 1);
        let _n = gpu.prefetch(&[k0]).expect("p0");
        let _s = gpu.score().expect("d0");
        let _a = gpu.acquire(k0).expect("a0");
        let _s = gpu.score().expect("drain0");
        gpu.evict(k0).expect("evict");
        let _s = gpu.score().expect("free");
        let _n = gpu.prefetch(&[k1]).expect("p1");
        let _s = gpu.score().expect("d1");
        let _b = gpu.acquire(k1).expect("a1");
        let score = gpu.score().expect("final");
        (
            gpu.graph_updates(),
            gpu.metrics().hits,
            gpu.metrics().misses,
            score.wall_ns,
        )
    };
    let (upd, h0, m0, wall_u) = run(true);
    let (none, h1, m1, wall_i) = run(false);
    assert_eq!(upd, 1);
    assert_eq!(none, 0);
    assert_eq!(h0, h1);
    assert_eq!(m0, m1);
    assert!(wall_u < wall_i, "update={wall_u} instantiate={wall_i}");
}

#[test]
fn simulated_gpu_store_graph_set_params_reuses_parked_exec() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[1])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=100000\ngraph_update_ns=1000\ngraph_set_params_ns=100\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let run = |graph_set_params: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            1,
            p.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                graph_set_params,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        let k0 = ExpertKey::new(0, 0);
        let k1 = ExpertKey::new(0, 1);
        let _n = gpu.prefetch(&[k0]).expect("p0");
        let _s = gpu.score().expect("d0");
        let _a = gpu.acquire(k0).expect("a0");
        let _s = gpu.score().expect("drain0");
        gpu.evict(k0).expect("evict");
        let _s = gpu.score().expect("free");
        let _n = gpu.prefetch(&[k1]).expect("p1");
        let _s = gpu.score().expect("d1");
        let _b = gpu.acquire(k1).expect("a1");
        let score = gpu.score().expect("final");
        (
            gpu.graph_set_params(),
            gpu.graph_updates(),
            gpu.metrics().hits,
            gpu.metrics().misses,
            score.wall_ns,
        )
    };
    let (sets, upd, h0, m0, wall_s) = run(true);
    let (none, none_u, h1, m1, wall_i) = run(false);
    assert_eq!(sets, 1);
    assert_eq!(upd, 0);
    assert_eq!(none, 0);
    assert_eq!(none_u, 0);
    assert_eq!(h0, h1);
    assert_eq!(m0, m1);
    assert!(wall_s < wall_i, "set_params={wall_s} instantiate={wall_i}");
}

#[test]
fn simulated_gpu_store_graph_set_params_with_mem() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[1])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=100000\ngraph_update_ns=1000\ngraph_set_params_ns=100\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_cfg(
        inner,
        1,
        p,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_set_params: true,
            graph_mem: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    let k1 = ExpertKey::new(0, 1);
    let _n = gpu.prefetch(&[k0]).expect("p0");
    let _s = gpu.score().expect("d0");
    let _a = gpu.acquire(k0).expect("a0");
    let _s = gpu.score().expect("drain0");
    gpu.evict(k0).expect("evict");
    let _s = gpu.score().expect("free");
    let _n = gpu.prefetch(&[k1]).expect("p1");
    let _s = gpu.score().expect("d1");
    let _b = gpu.acquire(k1).expect("a1");
    let _s = gpu.score().expect("final");
    assert_eq!(gpu.graph_updates(), 0);
    assert_eq!(gpu.graph_set_params(), 1);
}

#[test]
fn simulated_gpu_store_graph_update_and_set_params_conflict() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let err = match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_update: true,
            graph_set_params: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("both flags"),
        Err(e) => e,
    };
    assert!(
        err.to_string()
            .contains("choose one of graph-update, graph-set-params"),
        "{err}"
    );
}

#[test]
fn cuda_graphs_graph_clone_copies_leaf_before_instantiate() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_clone_ns=80000\ngraph_instantiate_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let base = SimCfg {
        cuda_graphs: true,
        ..SimCfg::lru(1, 4096, 0)
    };
    let plain = sim_replay_cfg(&t, p.clone(), base).expect("plain");
    let cloned = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            graph_clone: true,
            ..base
        },
    )
    .expect("clone");
    assert_eq!(plain.hits, cloned.hits);
    assert_eq!(plain.misses, cloned.misses);
    assert_eq!(plain.graph_clones, 0);
    assert_eq!(cloned.graph_clones, 1);
    assert!(
        cloned.sim_ns > plain.sim_ns,
        "clone={} plain={}",
        cloned.sim_ns,
        plain.sim_ns
    );
    assert!(cloned.line().contains("graph_clones="));
}

#[test]
fn cuda_graphs_graph_build_matches_capture_hits() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1]), ev(1, 0, &[0, 2])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let base = SimCfg {
        cuda_graphs: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let cap = sim_replay_cfg(&t, p.clone(), base).expect("cap");
    let bld = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            graph_build: true,
            ..base
        },
    )
    .expect("bld");
    assert_eq!(cap.hits, bld.hits);
    assert_eq!(cap.misses, bld.misses);
    assert_eq!(cap.graph_launches, bld.graph_launches);
    assert_eq!(cap.child_graphs, bld.child_graphs);
    assert!(bld.graph_launches > 0, "line={}", bld.line());
    assert!(bld.child_graphs > 0, "line={}", bld.line());
}

#[test]
fn cuda_graphs_graph_build_independent_children_overlap() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000\nhbm_bps=1000000000000\ngraph_instantiate_ns=1\ngraph_upload_ns=1\ngraph_launch_ns=1\nlaunch_overhead_ns=1\ncopy_engines=2\n",
    )
    .expect("profile");
    let run = |graph_build: bool| {
        sim_replay_cfg(
            &t,
            p.clone(),
            SimCfg {
                cuda_graphs: true,
                graph_build,
                compute_slots: 2,
                ..SimCfg::lru(2, 4096, 0)
            },
        )
        .expect("replay")
        .sim_ns
    };
    let cap = run(false);
    let bld = run(true);
    assert!(
        bld < cap,
        "graph-build combo children must Hyper-Q overlap capture; build={bld} capture={cap}"
    );
}

#[test]
fn simulated_gpu_store_graph_build_launches() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let run = |graph_build: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            1,
            p.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                graph_build,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        let k0 = ExpertKey::new(0, 0);
        let _n = gpu.prefetch(&[k0]).expect("prefetch");
        let _s = gpu.score().expect("drain");
        let _a = gpu.acquire(k0).expect("acq");
        let score = gpu.score().expect("final");
        (gpu.graph_launches(), gpu.metrics().hits, score.wall_ns)
    };
    let (n_build, h0, _wall_b) = run(true);
    let (n_cap, h1, _wall_c) = run(false);
    assert!(n_build > 0);
    assert_eq!(n_build, n_cap);
    assert_eq!(h0, h1);
}

#[test]
fn simulated_gpu_store_graph_build_update_reuses_parked_exec() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[1])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=100000\ngraph_update_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_cfg(
        inner,
        1,
        p,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_update: true,
            graph_build: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    let k1 = ExpertKey::new(0, 1);
    let _n = gpu.prefetch(&[k0]).expect("p0");
    let _s = gpu.score().expect("d0");
    let _a = gpu.acquire(k0).expect("a0");
    let _s = gpu.score().expect("drain0");
    gpu.evict(k0).expect("evict");
    let _s = gpu.score().expect("free");
    let _n = gpu.prefetch(&[k1]).expect("p1");
    let _s = gpu.score().expect("d1");
    let _b = gpu.acquire(k1).expect("a1");
    let _score = gpu.score().expect("final");
    assert_eq!(gpu.graph_updates(), 1);
    assert!(gpu.graph_launches() >= 2);
}

#[test]
fn cuda_graphs_graph_mem_matches_capture_hits() {
    use crate::sim_replay::GRAPH_SCRATCH_BYTES;
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let base = SimCfg {
        cuda_graphs: true,
        ..SimCfg::lru(1, 4096, 0)
    };
    let cap = sim_replay_cfg(&t, p.clone(), base).expect("cap");
    let mem = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            graph_mem: true,
            ..base
        },
    )
    .expect("mem");
    assert_eq!(cap.hits, mem.hits);
    assert_eq!(cap.misses, mem.misses);
    assert_eq!(cap.graph_launches, mem.graph_launches);
    assert!(mem.graph_launches > 0, "line={}", mem.line());
    assert_eq!(cap.hbm_peak, 4096);
    assert_eq!(mem.hbm_peak, 4096 + GRAPH_SCRATCH_BYTES);
}

#[test]
fn graph_mem_implies_cuda_graphs() {
    use crate::sim_replay::GRAPH_SCRATCH_BYTES;
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let mem = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            graph_mem: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    )
    .expect("mem");
    assert!(mem.graph_launches > 0, "line={}", mem.line());
    assert_eq!(mem.hbm_peak, 4096 + GRAPH_SCRATCH_BYTES);
}

#[test]
fn cuda_graphs_graph_mem_build_matches_capture() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1]), ev(1, 0, &[0, 2])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let cap = sim_replay_cfg(
        &t,
        p.clone(),
        SimCfg {
            cuda_graphs: true,
            graph_mem: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    )
    .expect("cap");
    let bld = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            graph_build: true,
            graph_mem: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    )
    .expect("bld");
    assert_eq!(cap.hits, bld.hits);
    assert_eq!(cap.misses, bld.misses);
    assert_eq!(cap.graph_launches, bld.graph_launches);
    assert_eq!(cap.child_graphs, bld.child_graphs);
    assert!(
        bld.hbm_peak > cap.hbm_peak,
        "independent graph-mem children overlap scratch; build={} capture={}",
        bld.hbm_peak,
        cap.hbm_peak
    );
    assert!(bld.child_graphs > 0, "line={}", bld.line());
}

#[test]
fn cuda_graphs_graph_mem_skips_update() {
    let t = Trace {
        events: vec![
            ev(0, 0, &[0]),
            ev(1, 0, &[1]),
            ev(2, 0, &[0]),
            ev(3, 0, &[1]),
        ],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=100000\ngraph_update_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let mem = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            cuda_graphs: true,
            graph_update: true,
            graph_mem: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    )
    .expect("mem");
    assert_eq!(mem.graph_updates, 0);
    assert!(mem.graph_launches > 0, "line={}", mem.line());
}

#[test]
fn simulated_gpu_store_graph_mem_scratch_peak() {
    use crate::sim_replay::GRAPH_SCRATCH_BYTES;
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let run = |graph_mem: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            1,
            p.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                graph_mem,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        let k0 = ExpertKey::new(0, 0);
        let _n = gpu.prefetch(&[k0]).expect("prefetch");
        let _s = gpu.score().expect("drain");
        let _a = gpu.acquire(k0).expect("acq");
        let score = gpu.score().expect("final");
        (gpu.graph_launches(), gpu.metrics().hits, score.hbm_peak)
    };
    let (n_mem, h0, peak_m) = run(true);
    let (n_cap, h1, peak_c) = run(false);
    assert!(n_mem > 0);
    assert_eq!(n_mem, n_cap);
    assert_eq!(h0, h1);
    assert_eq!(peak_c, 4096);
    assert_eq!(peak_m, 4096 + GRAPH_SCRATCH_BYTES);
}

#[test]
fn simulated_gpu_store_graph_mem_build_matches_capture() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let run = |graph_build: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            1,
            p.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                graph_mem: true,
                graph_build,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        let k0 = ExpertKey::new(0, 0);
        let _n = gpu.prefetch(&[k0]).expect("prefetch");
        let _s = gpu.score().expect("drain");
        let _a = gpu.acquire(k0).expect("acq");
        let score = gpu.score().expect("final");
        (gpu.graph_launches(), gpu.metrics().hits, score.hbm_peak)
    };
    let (n_build, h0, peak_b) = run(true);
    let (n_cap, h1, peak_c) = run(false);
    assert!(n_build > 0);
    assert_eq!(n_build, n_cap);
    assert_eq!(h0, h1);
    assert_eq!(peak_b, peak_c);
}

#[test]
fn simulated_gpu_store_graph_mem_skips_update() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[1])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=100000\ngraph_update_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_cfg(
        inner,
        1,
        p,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_update: true,
            graph_mem: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    let k1 = ExpertKey::new(0, 1);
    let _n = gpu.prefetch(&[k0]).expect("p0");
    let _s = gpu.score().expect("d0");
    let _a = gpu.acquire(k0).expect("a0");
    let _s = gpu.score().expect("drain0");
    gpu.evict(k0).expect("evict");
    let _s = gpu.score().expect("free");
    let _n = gpu.prefetch(&[k1]).expect("p1");
    let _s = gpu.score().expect("d1");
    let _b = gpu.acquire(k1).expect("a1");
    let _score = gpu.score().expect("final");
    assert_eq!(gpu.graph_updates(), 0);
    assert!(gpu.graph_launches() >= 2);
}

#[test]
fn simulated_gpu_store_graph_auto_free_keeps_scratch() {
    use crate::sim_replay::GRAPH_SCRATCH_BYTES;
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let run = |graph_auto_free: bool, graph_mem: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            1,
            p.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                graph_auto_free,
                graph_mem,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        let k0 = ExpertKey::new(0, 0);
        let _n = gpu.prefetch(&[k0]).expect("prefetch");
        let _s = gpu.score().expect("drain");
        let _a = gpu.acquire(k0).expect("acq");
        let score = gpu.score().expect("final");
        (
            gpu.graph_launches(),
            gpu.hbm_used(DeviceId(0)).expect("hbm"),
            score.hbm_peak,
        )
    };
    let (n_af, used_af, peak_af) = run(true, false);
    let (n_mem, used_mem, peak_mem) = run(false, true);
    assert!(n_af > 0);
    assert_eq!(n_af, n_mem);
    assert_eq!(peak_af, 4096 + GRAPH_SCRATCH_BYTES);
    assert_eq!(peak_mem, 4096 + GRAPH_SCRATCH_BYTES);
    assert_eq!(used_mem, 4096);
    assert_eq!(used_af, 4096 + GRAPH_SCRATCH_BYTES);
}

#[test]
fn graph_auto_free_implies_cuda_graphs() {
    use crate::sim_replay::GRAPH_SCRATCH_BYTES;
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let af = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            graph_auto_free: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    )
    .expect("af");
    assert!(af.graph_launches > 0, "line={}", af.line());
    assert_eq!(af.hbm_peak, 4096 + GRAPH_SCRATCH_BYTES);
}

#[test]
fn cuda_graphs_graph_auto_free_build_matches_capture() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1]), ev(1, 0, &[0, 2])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let cap = sim_replay_cfg(
        &t,
        p.clone(),
        SimCfg {
            cuda_graphs: true,
            graph_auto_free: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    )
    .expect("cap");
    let bld = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            graph_build: true,
            graph_auto_free: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    )
    .expect("bld");
    assert_eq!(cap.hits, bld.hits);
    assert_eq!(cap.misses, bld.misses);
    assert_eq!(cap.graph_launches, bld.graph_launches);
    assert_eq!(cap.child_graphs, bld.child_graphs);
    assert_eq!(cap.hbm_peak, bld.hbm_peak);
}

#[test]
fn cuda_graphs_graph_auto_free_skips_update() {
    let t = Trace {
        events: vec![
            ev(0, 0, &[0]),
            ev(1, 0, &[1]),
            ev(2, 0, &[0]),
            ev(3, 0, &[1]),
        ],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=100000\ngraph_update_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let af = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            cuda_graphs: true,
            graph_update: true,
            graph_auto_free: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    )
    .expect("af");
    assert_eq!(af.graph_updates, 0);
    assert!(af.graph_launches > 0, "line={}", af.line());
}

#[test]
fn graph_mem_and_graph_auto_free_conflict() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let err = sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            graph_mem: true,
            graph_auto_free: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("graph-mem"), "{err}");
}

#[test]
fn simulated_gpu_store_graph_auto_free_skips_update() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[1])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=100000\ngraph_update_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_cfg(
        inner,
        1,
        p,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_update: true,
            graph_auto_free: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    let k1 = ExpertKey::new(0, 1);
    let _n = gpu.prefetch(&[k0]).expect("p0");
    let _s = gpu.score().expect("d0");
    let _a = gpu.acquire(k0).expect("a0");
    let _s = gpu.score().expect("drain0");
    gpu.evict(k0).expect("evict");
    let _s = gpu.score().expect("free");
    let _n = gpu.prefetch(&[k1]).expect("p1");
    let _s = gpu.score().expect("d1");
    let _b = gpu.acquire(k1).expect("a1");
    let _score = gpu.score().expect("final");
    assert_eq!(gpu.graph_updates(), 0);
    assert!(gpu.graph_launches() >= 2);
}

#[test]
fn simulated_gpu_store_graph_mem_auto_free_conflict() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let err = match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_mem: true,
            graph_auto_free: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("expected conflict"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("graph-mem"), "{err}");
}

#[test]
fn simulated_gpu_store_graph_clone_copies_capture() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_clone_ns=80000\ngraph_instantiate_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let run = |graph_clone: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            1,
            p.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                graph_clone,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        let k0 = ExpertKey::new(0, 0);
        let _n = gpu.prefetch(&[k0]).expect("prefetch");
        let _s = gpu.score().expect("drain");
        let _a = gpu.acquire(k0).expect("acq");
        let score = gpu.score().expect("final");
        (gpu.graph_clones(), gpu.metrics().hits, score.wall_ns)
    };
    let (n_clone, h0, wall_c) = run(true);
    let (n_plain, h1, wall_p) = run(false);
    assert_eq!(n_clone, 1);
    assert_eq!(n_plain, 0);
    assert_eq!(h0, h1);
    assert!(wall_c > wall_p, "clone={wall_c} plain={wall_p}");
}

#[test]
fn simulated_gpu_store_timing_events_score_copy_elapsed() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::example_h100_sxm();
    let run = |timing_events: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            1,
            p.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                timing_events,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        let k0 = ExpertKey::new(0, 0);
        let _p = gpu.acquire(k0).expect("acq");
        gpu.release(k0);
        let _s = gpu.score().expect("score");
        (
            gpu.copy_elapsed_ns(),
            gpu.metrics().hits,
            gpu.metrics().misses,
        )
    };
    let (elapsed, h0, m0) = run(true);
    let (none, h1, m1) = run(false);
    assert!(elapsed > 0, "elapsed={elapsed}");
    assert_eq!(none, 0);
    assert_eq!(h0, h1);
    assert_eq!(m0, m1);
}

#[test]
fn simulated_gpu_store_captures_gemm_after_drain() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::new(inner, 1, HardwareProfile::example_h100_sxm(), 4096).expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    let n = gpu.prefetch(&[k0]).expect("prefetch");
    assert_eq!(n, 1);
    let _score = gpu.score().expect("drain");
    let _a = gpu.acquire(k0).expect("first");
    let _b = gpu.acquire(k0).expect("second");
    assert!(
        gpu.graph_launches() >= 2,
        "launches={}",
        gpu.graph_launches()
    );
}

#[test]
fn simulated_gpu_store_phase_tracks_copy() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::new(inner, 1, HardwareProfile::example_h100_sxm(), 4096).expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    assert_eq!(gpu.phase(k0), ExpertPhase::Cold);
    let n = gpu.prefetch(&[k0]).expect("prefetch");
    assert_eq!(n, 1);
    assert_eq!(gpu.phase(k0), ExpertPhase::Transferring);
    let err = gpu.lease(k0).unwrap_err();
    assert!(matches!(err, Error::Store(_)));
    let _score = gpu.score().expect("drain");
    assert_eq!(gpu.phase(k0), ExpertPhase::Resident);
    gpu.lease(k0).expect("lease after copy");
    assert_eq!(gpu.phase(k0), ExpertPhase::Leased);
    gpu.release(k0);
    assert_eq!(gpu.phase(k0), ExpertPhase::Resident);
}

#[test]
fn simulated_gpu_store_evict_is_observable_until_free_completes() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[1])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::new(inner, 1, HardwareProfile::example_h100_sxm(), 4096).expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    let k1 = ExpertKey::new(0, 1);
    let _n = gpu.prefetch(&[k0]).expect("k0");
    let _score = gpu.score().expect("drain k0");
    assert_eq!(gpu.phase(k0), ExpertPhase::Resident);
    gpu.evict(k0).expect("evict");
    assert_eq!(gpu.phase(k0), ExpertPhase::Evicting);
    let err = gpu.lease(k0).unwrap_err();
    assert!(matches!(err, Error::Store(_)));
    let _score = gpu.score().expect("drain free");
    assert_eq!(gpu.phase(k0), ExpertPhase::Cold);
    let _n = gpu.prefetch(&[k0]).expect("k0 again");
    let _score = gpu.score().expect("drain");
    let _n = gpu.prefetch(&[k1]).expect("evict via prefetch");
    assert_eq!(gpu.phase(k0), ExpertPhase::Evicting);
    assert_eq!(gpu.phase(k1), ExpertPhase::Transferring);
}

#[test]
fn simulated_gpu_store_host_func_lengthens_wall() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let inner = DirectStore::from_trace(&t);
    let mut plain = SimulatedGpuStore::new(inner, 2, p.clone(), 4096).expect("plain");
    let inner = DirectStore::from_trace(&t);
    let mut cb = SimulatedGpuStore::with_cfg(
        inner,
        2,
        p,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            host_func: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("host");
    for k in t.keys() {
        let _p = plain.acquire(k).expect("plain acq");
        let _c = cb.acquire(k).expect("cb acq");
    }
    let a = plain.score().expect("plain score");
    let b = cb.score().expect("cb score");
    assert_eq!(plain.metrics().hits, cb.metrics().hits);
    assert_eq!(plain.metrics().misses, cb.metrics().misses);
    assert!(
        b.wall_ns > a.wall_ns,
        "host={} plain={}",
        b.wall_ns,
        a.wall_ns
    );
}

#[test]
fn simulated_gpu_store_blocking_streams_serialize_copy_and_compute() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[1])],
    };
    let p = HardwareProfile::example_h100_sxm();
    let bytes = 32u64 << 20;
    let run = |blocking_streams: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            2,
            p.clone(),
            bytes,
            GpuFill::Pinned,
            GpuStoreCfg {
                blocking_streams,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        let _a = gpu.acquire(ExpertKey::new(0, 0)).expect("k0");
        let _b = gpu.acquire(ExpertKey::new(0, 1)).expect("k1");
        gpu.score().expect("score")
    };
    let nb = run(false);
    let block = run(true);
    assert!(
        block.wall_ns > nb.wall_ns,
        "block={} nonblock={}",
        block.wall_ns,
        nb.wall_ns
    );
}

#[test]
fn simulated_gpu_store_sync_alloc_cannot_overlap_miss() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[1])],
    };
    let p = HardwareProfile::example_h100_sxm();
    let bytes = 32u64 << 20;
    let run = |sync_alloc: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            2,
            p.clone(),
            bytes,
            GpuFill::Pinned,
            GpuStoreCfg {
                sync_alloc,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        let _a = gpu.acquire(ExpertKey::new(0, 0)).expect("k0");
        let _b = gpu.acquire(ExpertKey::new(0, 1)).expect("k1");
        gpu.score().expect("score")
    };
    let async_row = run(false);
    let sync_row = run(true);
    assert!(
        sync_row.wall_ns > async_row.wall_ns,
        "sync={} async={}",
        sync_row.wall_ns,
        async_row.wall_ns
    );
}

#[test]
fn simulated_gpu_store_mempool_holds_after_evict() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::example_h100_sxm();
    let bytes = 4096u64;
    let inner = DirectStore::from_trace(&t);
    let mut plain = SimulatedGpuStore::new(inner, 1, p.clone(), bytes).expect("plain");
    let inner = DirectStore::from_trace(&t);
    let mut pooled = SimulatedGpuStore::with_cfg(
        inner,
        1,
        p,
        bytes,
        GpuFill::Pinned,
        GpuStoreCfg {
            mempool: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("pool");
    let k0 = ExpertKey::new(0, 0);
    let _a = plain.acquire(k0).expect("plain acq");
    let _b = pooled.acquire(k0).expect("pool acq");
    plain.evict(k0).expect("plain evict");
    pooled.evict(k0).expect("pool evict");
    assert_eq!(plain.default_pool_cached().expect("plain cached"), 0);
    assert!(
        pooled.default_pool_cached().expect("pool cached") >= bytes,
        "mempool must hold the expert page"
    );
}

#[test]
fn simulated_gpu_store_shareable_imported_pool_reuses_cache() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::example_h100_sxm();
    let bytes = 4096u64;
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_cfg(
        inner,
        1,
        p,
        bytes,
        GpuFill::Pinned,
        GpuStoreCfg {
            shareable: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("share");
    assert!(gpu.share_imported_pool().is_some());
    let k0 = ExpertKey::new(0, 0);
    let _a = gpu.acquire(k0).expect("acq");
    gpu.evict(k0).expect("evict");
    assert!(
        gpu.default_pool_cached().expect("cached") >= bytes,
        "shareable implies mempool hold"
    );
    let used0 = gpu.hbm_used(DeviceId(0)).expect("used0");
    let _imp = gpu.alloc_from_imported_pool(bytes).expect("import alloc");
    assert_eq!(gpu.hbm_used(DeviceId(0)).expect("used1"), used0);
}

#[test]
fn shareable_needs_cuda_malloc_async() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let err = sim_replay_cfg(
        &t,
        p.clone(),
        SimCfg {
            shareable: true,
            sync_alloc: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    )
    .expect_err("sync");
    assert!(
        err.to_string().contains("shareable needs cudaMallocAsync"),
        "{err}"
    );
    let inner = DirectStore::from_trace(&t);
    match SimulatedGpuStore::with_cfg(
        inner,
        2,
        p,
        4096,
        GpuFill::Vmm,
        GpuStoreCfg {
            shareable: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("vmm shareable must fail"),
        Err(e) => assert!(
            e.to_string().contains("shareable needs cudaMallocAsync"),
            "{e}"
        ),
    }
}

#[test]
fn sim_replay_shareable_holds_like_mempool() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let held = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            shareable: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    )
    .expect("share");
    assert!(held.hbm_peak >= 4096);
}

#[test]
fn simulated_gpu_store_vmm_page_pays_map_overhead() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let bytes = 1u64 << 20;
    let run = |page: u64| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            1,
            p.clone(),
            bytes,
            GpuFill::Vmm,
            GpuStoreCfg {
                vmm_page: page,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        for k in t.keys() {
            let _p = gpu.acquire(k).expect("acq");
        }
        let metrics = gpu.metrics();
        let score = gpu.score().expect("score");
        (metrics, score)
    };
    let (full_m, full_s) = run(0);
    let (paged_m, paged_s) = run(bytes / 4);
    assert_eq!(full_m.hits, paged_m.hits);
    assert_eq!(full_m.misses, paged_m.misses);
    assert_eq!(full_s.hbm_peak, paged_s.hbm_peak);
    assert!(
        paged_s.wall_ns > full_s.wall_ns,
        "paged maps must pay per-block map overhead; paged={} full={}",
        paged_s.wall_ns,
        full_s.wall_ns
    );
}

#[test]
fn simulated_gpu_store_mapped_pin_budget_caps_occupancy() {
    let mut events = Vec::new();
    for tok in 0..16u32 {
        events.push(ev(tok, 0, &[tok % 2]));
    }
    let t = Trace { events };
    let bytes = 1u64 << 20;
    let tight = HardwareProfile::example_h100_sxm().restrict_pin(bytes);
    let open = HardwareProfile::example_h100_sxm();
    let run = |slots: usize, profile: HardwareProfile| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_mapped(inner, slots, profile, bytes).expect("gpu");
        assert!(!gpu.staging_is_pinned());
        for k in t.keys() {
            let _p = gpu.acquire(k).expect("acq");
        }
        gpu.metrics()
    };
    let capped = run(2, tight);
    let slim = run(1, open.clone());
    let fat = run(2, open);
    assert_eq!(capped.hits, slim.hits);
    assert_eq!(capped.misses, slim.misses);
    assert!(
        fat.hits > capped.hits,
        "uncapped two-slot mapped must hit; fat={} cap={}",
        fat.hits,
        capped.hits
    );
}

#[test]
fn simulated_gpu_store_mapped_pin_budget_zero_fit_is_pin_oom() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::example_h100_sxm().restrict_pin(1);
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_mapped(inner, 1, p, 1u64 << 20).expect("gpu");
    let err = gpu.acquire(ExpertKey::new(0, 0)).unwrap_err();
    assert!(err.to_string().contains("pin"), "{err}");
}

#[test]
fn gpu_fill_from_flags_is_exclusive() {
    assert_eq!(
        GpuFill::from_flags(false, false, false).expect("pin"),
        GpuFill::Pinned
    );
    assert_eq!(
        GpuFill::from_flags(true, false, false).expect("map"),
        GpuFill::Mapped
    );
    assert_eq!(
        GpuFill::from_flags(false, true, false).expect("um"),
        GpuFill::Managed
    );
    assert_eq!(
        GpuFill::from_flags(false, false, true).expect("vmm"),
        GpuFill::Vmm
    );
    let err = GpuFill::from_flags(true, true, false).unwrap_err();
    assert!(matches!(err, Error::Store(_)));
}

#[test]
fn store_replay_demand_pages_the_trace() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let row =
        store_replay(&t, p, 2, 4096, GpuFill::Pinned, GpuStoreCfg::default()).expect("replay");
    assert!(row.metrics.misses >= 3);
    assert!(row.score.wall_ns > 0);
    assert!(row.line().starts_with("store "));
    assert!(row.line().contains("hits="));
    assert!(row.line().contains("wall_ns="));
}

#[test]
fn simulated_gpu_store_pageable_h2d_is_slower_than_pinned() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[1])],
    };
    let p = HardwareProfile::example_h100_sxm();
    let bytes = 32u64 << 20;
    let run = |pageable: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            2,
            p.clone(),
            bytes,
            GpuFill::Pinned,
            GpuStoreCfg {
                pageable,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        if pageable {
            assert!(!gpu.staging_is_pinned());
        }
        let _a = gpu.acquire(ExpertKey::new(0, 0)).expect("k0");
        let _b = gpu.acquire(ExpertKey::new(0, 1)).expect("k1");
        gpu.score().expect("score")
    };
    let pin = run(false);
    let page = run(true);
    assert!(
        page.wall_ns > pin.wall_ns,
        "pageable={} pinned={}",
        page.wall_ns,
        pin.wall_ns
    );
}

#[test]
fn store_replay_markov_prefetch_beats_demand() {
    let mut events = Vec::new();
    for tok in 0..16u32 {
        events.push(ev(tok, 0, &[0]));
        events.push(ev(tok, 1, &[7]));
    }
    let t = Trace { events };
    let p = HardwareProfile::example_h100_sxm();
    let run = |prefetch: Prefetch| {
        store_replay_cfg(
            &t,
            p.clone(),
            StoreReplayCfg {
                prefetch,
                ..StoreReplayCfg::demand(1, 4096, GpuFill::Pinned)
            },
        )
        .expect("store")
    };
    let none = run(Prefetch::None);
    let fwd = run(Prefetch::CopyForward);
    let mk = run(Prefetch::Markov);
    assert!(
        mk.metrics.hits > fwd.metrics.hits,
        "markov={} copy-forward={}",
        mk.metrics.hits,
        fwd.metrics.hits
    );
    assert!(
        mk.metrics.hits > none.metrics.hits,
        "markov={} demand={}",
        mk.metrics.hits,
        none.metrics.hits
    );
    assert!(mk.metrics.prefetches > 0);
}

#[test]
fn simulated_gpu_store_accessed_by_migrate_keeps_home() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::example_2node_rdma();
    let bytes = 4096u64;
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_cfg(
        inner,
        1,
        p,
        bytes,
        GpuFill::Managed,
        GpuStoreCfg {
            accessed_by: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    let _p = gpu.acquire(k0).expect("acq");
    assert!(gpu.page_accessed_by(k0, DeviceId(1)));
    gpu.migrate(k0, DeviceId(1)).expect("mig");
    assert_eq!(gpu.device_of(k0), Some(DeviceId(1)));
    let _p = gpu.acquire(k0).expect("gemm dest");
    let score = gpu.score().expect("score");
    assert_eq!(score.bytes_moved, bytes);
    assert_eq!(score.hbm_peak, bytes);
    assert!(gpu.page_resident(k0, DeviceId(0)));
    assert!(!gpu.page_resident(k0, DeviceId(1)));
    assert_eq!(gpu.hbm_used(DeviceId(1)).expect("d1"), 0);
}

#[test]
fn simulated_gpu_store_accessed_by_pin_skips_dest_hbm() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_cfg(
        inner,
        2,
        HardwareProfile::example_8xh100_nvlink(),
        4096,
        GpuFill::Managed,
        GpuStoreCfg {
            accessed_by: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    gpu.pin_hot(&[k0]).expect("pin");
    let _p = gpu.acquire(k0).expect("hit");
    let score = gpu.score().expect("score");
    assert_eq!(score.bytes_moved, 4096);
    assert_eq!(gpu.hbm_used(DeviceId(1)).expect("d1"), 0);
    assert!(gpu.page_accessed_by(k0, DeviceId(1)));
}

#[test]
fn simulated_gpu_store_vmm_accessed_by_migrate_keeps_home() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::example_2node_rdma();
    let bytes = 4096u64;
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_cfg(
        inner,
        1,
        p,
        bytes,
        GpuFill::Vmm,
        GpuStoreCfg {
            accessed_by: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    let _p = gpu.acquire(k0).expect("acq");
    assert!(gpu.page_accessed_by(k0, DeviceId(1)));
    gpu.migrate(k0, DeviceId(1)).expect("mig");
    assert_eq!(gpu.device_of(k0), Some(DeviceId(1)));
    let _p = gpu.acquire(k0).expect("gemm dest");
    let score = gpu.score().expect("score");
    assert_eq!(score.bytes_moved, bytes);
    assert_eq!(score.hbm_peak, bytes);
    assert!(gpu.page_resident(k0, DeviceId(0)));
    assert!(!gpu.page_resident(k0, DeviceId(1)));
    assert_eq!(gpu.hbm_used(DeviceId(1)).expect("d1"), 0);
}

#[test]
fn simulated_gpu_store_vmm_accessed_by_pin_skips_dest_hbm() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_cfg(
        inner,
        2,
        HardwareProfile::example_8xh100_nvlink(),
        4096,
        GpuFill::Vmm,
        GpuStoreCfg {
            accessed_by: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    gpu.pin_hot(&[k0]).expect("pin");
    let _p = gpu.acquire(k0).expect("hit");
    let score = gpu.score().expect("score");
    assert_eq!(score.bytes_moved, 4096);
    assert_eq!(gpu.hbm_used(DeviceId(1)).expect("d1"), 0);
    assert!(gpu.page_accessed_by(k0, DeviceId(1)));
}

#[test]
fn simulated_gpu_store_pool_accessed_by_migrate_keeps_home() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::example_2node_rdma();
    let bytes = 4096u64;
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_cfg(
        inner,
        1,
        p,
        bytes,
        GpuFill::Pinned,
        GpuStoreCfg {
            accessed_by: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    let _p = gpu.acquire(k0).expect("acq");
    assert!(gpu.page_accessed_by(k0, DeviceId(1)));
    gpu.migrate(k0, DeviceId(1)).expect("mig");
    assert_eq!(gpu.device_of(k0), Some(DeviceId(1)));
    let _p = gpu.acquire(k0).expect("gemm dest");
    let score = gpu.score().expect("score");
    assert_eq!(score.bytes_moved, bytes);
    assert_eq!(score.hbm_peak, bytes);
    assert!(gpu.page_resident(k0, DeviceId(0)));
    assert!(!gpu.page_resident(k0, DeviceId(1)));
    assert_eq!(gpu.hbm_used(DeviceId(1)).expect("d1"), 0);
}

#[test]
fn simulated_gpu_store_pool_accessed_by_pin_skips_dest_hbm() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_cfg(
        inner,
        2,
        HardwareProfile::example_8xh100_nvlink(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            accessed_by: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    gpu.pin_hot(&[k0]).expect("pin");
    let _p = gpu.acquire(k0).expect("hit");
    let score = gpu.score().expect("score");
    assert_eq!(score.bytes_moved, 4096);
    assert_eq!(gpu.hbm_used(DeviceId(1)).expect("d1"), 0);
    assert!(gpu.page_accessed_by(k0, DeviceId(1)));
}

#[test]
fn simulated_gpu_store_legacy_null_serializes_copy_and_compute() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[1])],
    };
    let p = HardwareProfile::example_h100_sxm();
    let bytes = 32u64 << 20;
    let run = |legacy_null: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            2,
            p.clone(),
            bytes,
            GpuFill::Pinned,
            GpuStoreCfg {
                legacy_null,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        let _a = gpu.acquire(ExpertKey::new(0, 0)).expect("k0");
        let _b = gpu.acquire(ExpertKey::new(0, 1)).expect("k1");
        gpu.score().expect("score")
    };
    let def = run(false);
    let legacy = run(true);
    assert!(
        legacy.wall_ns > def.wall_ns,
        "legacy={} default={}",
        legacy.wall_ns,
        def.wall_ns
    );
}

#[test]
fn sim_replay_accessed_by_maps_peer_without_migrating() {
    use crate::replay::Touch;
    use crate::sim_replay::{apply_touch, GraphBank, PageHandle, TouchArgs};
    use gpu_sim::{DType, KernelKind, Sim, StreamId};
    use std::collections::BTreeMap;

    let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
    let mut handles: BTreeMap<ExpertKey, PageHandle> = BTreeMap::new();
    let mut graphs = GraphBank::new(false, false, false, crate::sim_replay::LeafMem::None);
    let args = TouchArgs {
        d: DeviceId(0),
        s: StreamId(0),
        bytes: 4096,
        slots: 1,
        sync_alloc: false,
        mapped: false,
        managed: true,
        vmm: false,
        vmm_page: 0,
        pageable: false,
        accessed_by: true,
    };
    let mut next_event = 1u32;
    let k0 = ExpertKey::new(0, 0);
    apply_touch(
        &mut sim,
        &mut handles,
        &mut graphs,
        args,
        k0,
        Touch::Miss { evicted: None },
        &mut next_event,
    )
    .expect("miss");
    let id = handles.get(&k0).expect("page").id;
    sim.synchronize().expect("prefetch");
    assert!(sim.is_accessed_by(id, DeviceId(1)).expect("advise"));
    let _k = sim
        .kernel(
            DeviceId(1),
            KernelKind::GroupedMoeGemm {
                experts: 1,
                tokens_per_expert: 1,
                hidden: 64,
                ff: 64,
                dtype: DType::Fp16,
            },
            &[id],
            &[],
            StreamId(0),
        )
        .expect("remote gemm");
    sim.synchronize().expect("gemm");
    assert!(sim.is_resident(id, DeviceId(0)).expect("home"));
    assert!(!sim.is_resident(id, DeviceId(1)).expect("dest"));
    assert_eq!(sim.hbm_used(DeviceId(1)).expect("d1"), 0);
}

#[test]
fn sim_replay_vmm_accessed_by_maps_peer_without_migrating() {
    use crate::replay::Touch;
    use crate::sim_replay::{apply_touch, GraphBank, PageHandle, TouchArgs};
    use gpu_sim::{DType, KernelKind, Sim, StreamId};
    use std::collections::BTreeMap;

    let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
    let mut handles: BTreeMap<ExpertKey, PageHandle> = BTreeMap::new();
    let mut graphs = GraphBank::new(false, false, false, crate::sim_replay::LeafMem::None);
    let args = TouchArgs {
        d: DeviceId(0),
        s: StreamId(0),
        bytes: 4096,
        slots: 1,
        sync_alloc: false,
        mapped: false,
        managed: false,
        vmm: true,
        vmm_page: 0,
        pageable: false,
        accessed_by: true,
    };
    let mut next_event = 1u32;
    let k0 = ExpertKey::new(0, 0);
    apply_touch(
        &mut sim,
        &mut handles,
        &mut graphs,
        args,
        k0,
        Touch::Miss { evicted: None },
        &mut next_event,
    )
    .expect("miss");
    let id = handles.get(&k0).expect("page").id;
    sim.synchronize().expect("h2d");
    assert!(sim.is_accessed_by(id, DeviceId(1)).expect("set access"));
    let _k = sim
        .kernel(
            DeviceId(1),
            KernelKind::GroupedMoeGemm {
                experts: 1,
                tokens_per_expert: 1,
                hidden: 64,
                ff: 64,
                dtype: DType::Fp16,
            },
            &[id],
            &[],
            StreamId(0),
        )
        .expect("remote gemm");
    sim.synchronize().expect("gemm");
    assert!(sim.is_resident(id, DeviceId(0)).expect("home"));
    assert!(!sim.is_resident(id, DeviceId(1)).expect("dest"));
    assert_eq!(sim.hbm_used(DeviceId(1)).expect("d1"), 0);
}

#[test]
fn sim_replay_pool_accessed_by_maps_peer_without_migrating() {
    use crate::replay::Touch;
    use crate::sim_replay::{advise_pool_access, apply_touch, GraphBank, PageHandle, TouchArgs};
    use gpu_sim::{DType, KernelKind, Sim, StreamId};
    use std::collections::BTreeMap;

    let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
    advise_pool_access(&mut sim).expect("pool access");
    let mut handles: BTreeMap<ExpertKey, PageHandle> = BTreeMap::new();
    let mut graphs = GraphBank::new(false, false, false, crate::sim_replay::LeafMem::None);
    let args = TouchArgs {
        d: DeviceId(0),
        s: StreamId(0),
        bytes: 4096,
        slots: 1,
        sync_alloc: false,
        mapped: false,
        managed: false,
        vmm: false,
        vmm_page: 0,
        pageable: false,
        accessed_by: true,
    };
    let mut next_event = 1u32;
    let k0 = ExpertKey::new(0, 0);
    apply_touch(
        &mut sim,
        &mut handles,
        &mut graphs,
        args,
        k0,
        Touch::Miss { evicted: None },
        &mut next_event,
    )
    .expect("miss");
    let id = handles.get(&k0).expect("page").id;
    sim.synchronize().expect("h2d");
    assert!(sim.is_accessed_by(id, DeviceId(1)).expect("set access"));
    let _k = sim
        .kernel(
            DeviceId(1),
            KernelKind::GroupedMoeGemm {
                experts: 1,
                tokens_per_expert: 1,
                hidden: 64,
                ff: 64,
                dtype: DType::Fp16,
            },
            &[id],
            &[],
            StreamId(0),
        )
        .expect("remote gemm");
    sim.synchronize().expect("gemm");
    assert!(sim.is_resident(id, DeviceId(0)).expect("home"));
    assert!(!sim.is_resident(id, DeviceId(1)).expect("dest"));
    assert_eq!(sim.hbm_used(DeviceId(1)).expect("d1"), 0);
}

#[test]
fn sim_replay_pageable_h2d_is_slower_than_pinned() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[1])],
    };
    let p = HardwareProfile::example_h100_sxm();
    let bytes = 32u64 << 20;
    let run = |pageable: bool| {
        sim_replay_cfg(
            &t,
            p.clone(),
            SimCfg {
                pageable,
                ..SimCfg::lru(2, bytes, 0)
            },
        )
        .expect("sim")
    };
    let pin = run(false);
    let page = run(true);
    assert_eq!(pin.hits, page.hits);
    assert_eq!(pin.misses, page.misses);
    assert!(
        page.sim_ns > pin.sim_ns,
        "pageable={} pinned={}",
        page.sim_ns,
        pin.sim_ns
    );
}

#[test]
fn sim_replay_legacy_null_serializes_seq_streams() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0]), ev_seq(1, 0, 0, &[1])],
    };
    let p = HardwareProfile::parse("gpus=1\ncopy_engines=1\n").expect("profile");
    let cfg = |legacy_null: bool| SimCfg {
        slots: 2,
        bytes_per_expert: 32u64 << 20,
        lookahead: 0,
        seq_streams: true,
        legacy_null,
        ..SimCfg::lru(2, 32u64 << 20, 0)
    };
    let def = sim_replay_cfg(&t, p.clone(), cfg(false)).expect("def");
    let legacy = sim_replay_cfg(&t, p, cfg(true)).expect("legacy");
    assert_eq!(def.hits, legacy.hits);
    assert_eq!(def.misses, legacy.misses);
    assert!(
        legacy.sim_ns > def.sim_ns,
        "legacy={} default={}",
        legacy.sim_ns,
        def.sim_ns
    );
}

#[test]
fn schedule_managed_accessed_by_replicas_skip_dest_prefetch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0]), ev(2, 0, &[0])],
    };
    let p = HardwareProfile::example_8xh100_nvlink();
    let bytes = 1u64 << 20;
    let um = SimCfg {
        managed: true,
        ..SimCfg::lru(4, bytes, 0)
    };
    let ab = SimCfg {
        accessed_by: true,
        ..um
    };
    let stripe = striped(&t, 8);
    let hot = with_hot_replicas(stripe.clone(), &t, 8, 200);
    let plain =
        schedule_placed(&t, p.clone(), um, SchedCfg::closed(0), Some(&stripe)).expect("stripe");
    let prefetch =
        schedule_placed(&t, p.clone(), um, SchedCfg::closed(0), Some(&hot)).expect("prefetch");
    let mapped = schedule_placed(&t, p, ab, SchedCfg::closed(0), Some(&hot)).expect("accessed");
    assert!(
        prefetch.replay.bytes_moved > plain.replay.bytes_moved,
        "prefetch={} stripe={}",
        prefetch.replay.bytes_moved,
        plain.replay.bytes_moved
    );
    assert_eq!(mapped.replay.bytes_moved, plain.replay.bytes_moved);
}

#[test]
fn schedule_vmm_hot_replicas_d2d() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0]), ev(2, 0, &[0])],
    };
    let p = HardwareProfile::example_8xh100_nvlink();
    let bytes = 1u64 << 20;
    let cfg = SimCfg {
        vmm: true,
        ..SimCfg::lru(4, bytes, 0)
    };
    let stripe = striped(&t, 8);
    let hot = with_hot_replicas(stripe.clone(), &t, 8, 200);
    let a =
        schedule_placed(&t, p.clone(), cfg, SchedCfg::closed(0), Some(&stripe)).expect("stripe");
    let b = schedule_placed(&t, p, cfg, SchedCfg::closed(0), Some(&hot)).expect("hot");
    assert!(
        b.replay.bytes_moved > a.replay.bytes_moved,
        "vmm hot={} stripe={}",
        b.replay.bytes_moved,
        a.replay.bytes_moved
    );
}

#[test]
fn schedule_vmm_multicast_replicas_beat_sequential_d2d() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0]), ev(2, 0, &[0])],
    };
    let p = HardwareProfile::example_8xh100_nvlink();
    let bytes = 1u64 << 20;
    let d2d = SimCfg {
        vmm: true,
        ..SimCfg::lru(8, bytes, 0)
    };
    let nvls = SimCfg {
        multicast: true,
        ..d2d
    };
    let mut hot = striped(&t, 8);
    let _prev = hot
        .replicas
        .insert(ExpertKey::new(0, 0), (1..8u16).map(DeviceId).collect());
    let a = schedule_placed(&t, p.clone(), d2d, SchedCfg::closed(0), Some(&hot)).expect("d2d");
    let b = schedule_placed(&t, p, nvls, SchedCfg::closed(0), Some(&hot)).expect("nvls");
    assert!(
        b.replay.sim_ns < a.replay.sim_ns,
        "nvls={} d2d={}",
        b.replay.sim_ns,
        a.replay.sim_ns
    );
    assert!(
        b.replay.bytes_moved < a.replay.bytes_moved,
        "NVLS counts one fabric hop; nvls={} d2d={}",
        b.replay.bytes_moved,
        a.replay.bytes_moved
    );
}

#[test]
fn multicast_cfg_requires_vmm_and_nvlink() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let err = sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            multicast: true,
            vmm: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    )
    .expect_err("1gpu");
    assert!(err.to_string().contains("NVLink"), "{err}");
    let err = sim_replay_cfg(
        &t,
        HardwareProfile::example_8xh100_nvlink(),
        SimCfg {
            multicast: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    )
    .expect_err("no vmm");
    assert!(err.to_string().contains("vmm"), "{err}");
    let err = match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        2,
        HardwareProfile::example_8xh100_nvlink(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            multicast: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("pinned multicast"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("vmm"), "{err}");
}

#[test]
fn schedule_vmm_accessed_by_replicas_skip_dest_hbm() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0]), ev(2, 0, &[0])],
    };
    let p = HardwareProfile::example_8xh100_nvlink();
    let bytes = 1u64 << 20;
    let um = SimCfg {
        vmm: true,
        ..SimCfg::lru(4, bytes, 0)
    };
    let ab = SimCfg {
        accessed_by: true,
        ..um
    };
    let stripe = striped(&t, 8);
    let hot = with_hot_replicas(stripe.clone(), &t, 8, 200);
    let plain =
        schedule_placed(&t, p.clone(), um, SchedCfg::closed(0), Some(&stripe)).expect("stripe");
    let prefetch =
        schedule_placed(&t, p.clone(), um, SchedCfg::closed(0), Some(&hot)).expect("prefetch");
    let mapped = schedule_placed(&t, p, ab, SchedCfg::closed(0), Some(&hot)).expect("accessed");
    assert!(
        prefetch.replay.bytes_moved > plain.replay.bytes_moved,
        "prefetch={} stripe={}",
        prefetch.replay.bytes_moved,
        plain.replay.bytes_moved
    );
    assert_eq!(mapped.replay.bytes_moved, plain.replay.bytes_moved);
}

#[test]
fn schedule_pool_accessed_by_replicas_skip_dest_hbm() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0]), ev(2, 0, &[0])],
    };
    let p = HardwareProfile::example_8xh100_nvlink();
    let bytes = 1u64 << 20;
    let pin = SimCfg::lru(4, bytes, 0);
    let ab = SimCfg {
        accessed_by: true,
        ..pin
    };
    let stripe = striped(&t, 8);
    let hot = with_hot_replicas(stripe.clone(), &t, 8, 200);
    let plain =
        schedule_placed(&t, p.clone(), pin, SchedCfg::closed(0), Some(&stripe)).expect("stripe");
    let prefetch =
        schedule_placed(&t, p.clone(), pin, SchedCfg::closed(0), Some(&hot)).expect("prefetch");
    let mapped = schedule_placed(&t, p, ab, SchedCfg::closed(0), Some(&hot)).expect("accessed");
    assert!(
        prefetch.replay.bytes_moved > plain.replay.bytes_moved,
        "prefetch={} stripe={}",
        prefetch.replay.bytes_moved,
        plain.replay.bytes_moved
    );
    assert_eq!(mapped.replay.bytes_moved, plain.replay.bytes_moved);
}

#[test]
fn simulated_gpu_store_stream_priority_marks_compute() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let gpu = SimulatedGpuStore::with_cfg(
        inner,
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            stream_priority: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    assert_eq!(gpu.stream_priority(DeviceId(0), gpu_sim::StreamId(0)), 0);
    assert_eq!(gpu.stream_priority(DeviceId(0), gpu_sim::StreamId(1)), 1);
}

#[test]
fn simulated_gpu_store_seq_streams_overlaps_h2d() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0]), ev_seq(1, 0, 0, &[1])],
    };
    let p = HardwareProfile::example_h100_sxm();
    let bytes = 32u64 << 20;
    let wall = |seq_streams: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            2,
            p.clone(),
            bytes,
            GpuFill::Pinned,
            GpuStoreCfg {
                seq_streams,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        gpu.bind_sequence(0);
        let _n = gpu.prefetch(&[ExpertKey::new(0, 0)]).expect("k0");
        gpu.bind_sequence(1);
        let _n = gpu.prefetch(&[ExpertKey::new(0, 1)]).expect("k1");
        gpu.score().expect("score").wall_ns
    };
    let serial = wall(false);
    let overlap = wall(true);
    assert!(
        overlap < serial,
        "per-sequence copy streams must overlap H2D; overlap={overlap} serial={serial}"
    );
}

#[test]
fn seq_stream_priority_starts_higher_stream_first() {
    use crate::replay::Touch;
    use crate::sim_replay::{
        apply_touch, gemm_keys, GraphBank, PageHandle, ReplayCounters, TouchArgs,
    };
    use gpu_sim::{GpuOp, Sim, StreamId};
    use std::collections::BTreeMap;

    let p = HardwareProfile::parse("gpus=1\ncopy_engines=2\n").expect("profile");
    let first = |priority: bool| {
        let mut sim = Sim::new(p.clone());
        if priority {
            sim.set_created_streams_priority(2).expect("pri");
        }
        let mut handles: BTreeMap<ExpertKey, PageHandle> = BTreeMap::new();
        let mut graphs = GraphBank::new(false, false, false, crate::sim_replay::LeafMem::None);
        let mut args = TouchArgs {
            d: DeviceId(0),
            s: StreamId(0),
            bytes: 4096,
            slots: 2,
            sync_alloc: false,
            mapped: false,
            managed: false,
            vmm: false,
            vmm_page: 0,
            pageable: false,
            accessed_by: false,
        };
        let mut next_event = 1u32;
        let k0 = ExpertKey::new(0, 0);
        let k1 = ExpertKey::new(0, 1);
        apply_touch(
            &mut sim,
            &mut handles,
            &mut graphs,
            args,
            k0,
            Touch::Miss { evicted: None },
            &mut next_event,
        )
        .expect("k0");
        args.s = StreamId(1);
        apply_touch(
            &mut sim,
            &mut handles,
            &mut graphs,
            args,
            k1,
            Touch::Miss { evicted: None },
            &mut next_event,
        )
        .expect("k1");
        sim.synchronize().expect("h2d");
        let mut ctr = ReplayCounters::default();
        gemm_keys(
            &mut sim,
            &handles,
            &mut graphs,
            &[k0, k1],
            false,
            &mut ctr,
            None,
        )
        .expect("gemm");
        sim.start_ready().expect("start");
        let started: Vec<StreamId> = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.start_ns.is_some())
            .map(|o| o.stream)
            .collect();
        started.first().copied()
    };
    assert_eq!(first(false), Some(StreamId(0)));
    assert_eq!(first(true), Some(StreamId(1)));
}

#[test]
fn sim_replay_stream_priority_keeps_hits() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0]), ev_seq(1, 0, 0, &[1])],
    };
    let p = HardwareProfile::example_h100_sxm();
    let cfg = |stream_priority: bool| SimCfg {
        seq_streams: true,
        stream_priority,
        ..SimCfg::lru(2, 4096, 0)
    };
    let plain = sim_replay_cfg(&t, p.clone(), cfg(false)).expect("plain");
    let pri = sim_replay_cfg(&t, p, cfg(true)).expect("pri");
    assert_eq!(plain.hits, pri.hits);
    assert_eq!(plain.misses, pri.misses);
}

#[test]
fn simulated_gpu_store_kv_sim_bills_fault_hit_drop() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_cfg(
        inner,
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            kv_sim: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    gpu.bind_kv(4, 1 << 20).expect("bind");
    gpu.apply_kv_ops(&[KvSimOp::Fault(0)]).expect("fault");
    gpu.apply_kv_ops(&[KvSimOp::Hit(0)]).expect("hit");
    gpu.apply_kv_ops(&[KvSimOp::Drop(0)]).expect("drop");
    assert_eq!(gpu.kv_misses(), 1);
    assert_eq!(gpu.kv_hits(), 1);
    assert!(gpu.score().expect("score").wall_ns > 0);
}

#[test]
fn simulated_gpu_store_kv_sim_off_ignores_ops() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu =
        SimulatedGpuStore::new(inner, 1, HardwareProfile::example_h100_sxm(), 4096).expect("gpu");
    gpu.bind_kv(4, 4096).expect("bind");
    gpu.apply_kv_ops(&[KvSimOp::Fault(0)]).expect("op");
    assert_eq!(gpu.kv_misses(), 0);
    assert_eq!(gpu.kv_hits(), 0);
}

#[test]
fn simulated_gpu_store_decode_priority_marks_higher_stream() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_cfg(
        inner,
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            decode_priority: true,
            stream_priority: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    assert_eq!(gpu.prefill_stream(), gpu_sim::StreamId(1));
    assert_eq!(gpu.decode_stream(), gpu_sim::StreamId(2));
    assert_eq!(gpu.compute_stream(), gpu_sim::StreamId(1));
    gpu.bind_decode_compute(true);
    assert_eq!(gpu.compute_stream(), gpu_sim::StreamId(2));
    assert!(
        gpu.stream_priority(DeviceId(0), gpu_sim::StreamId(2))
            > gpu.stream_priority(DeviceId(0), gpu_sim::StreamId(1)),
        "decode stream must outrank prefill"
    );
}

#[test]
fn bind_sequence_keeps_copy_mod_n_copy_after_decode_bind() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_cfg(
        inner,
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            seq_streams: true,
            decode_priority: true,
            stream_priority: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    assert_eq!(gpu.prefill_stream(), gpu_sim::StreamId(2));
    gpu.bind_decode_compute(true);
    gpu.bind_sequence(2);
    assert_eq!(
        gpu.copy_stream(),
        gpu_sim::StreamId(0),
        "copy must stay sequence % n_copy after decode retarget"
    );
}

#[test]
fn simulated_gpu_store_token_clock_skips_leftover_prefill() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_cfg(
        inner,
        2,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            decode_priority: true,
            stream_priority: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let pre = ExpertKey::new(0, 0);
    let dec = ExpertKey::new(0, 1);
    gpu.bind_decode_compute(false);
    let _warm_pre = gpu.acquire(pre).expect("warm pre");
    gpu.release(pre);
    let _warm_dec = gpu.acquire(dec).expect("warm dec");
    gpu.release(dec);
    let _drained = gpu.clock_ns().expect("drain h2d");
    gpu.bind_decode_compute(false);
    let _prefill = gpu.acquire(pre).expect("prefill");
    gpu.release(pre);
    gpu.bind_decode_compute(true);
    let _decode = gpu.acquire(dec).expect("decode");
    gpu.release(dec);
    let token = gpu.token_clock_ns().expect("token");
    let full = gpu.clock_ns().expect("full");
    assert!(
        token < full,
        "decode-stream ITL must leave leftover prefill running; token={token} full={full}"
    );
}

#[test]
fn simulated_gpu_store_compute_slots_overlap_leftover_prefill() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let run = |slots: u8| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            2,
            HardwareProfile::example_h100_sxm(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                decode_priority: true,
                stream_priority: true,
                compute_slots: slots,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        let pre = ExpertKey::new(0, 0);
        let dec = ExpertKey::new(0, 1);
        gpu.bind_decode_compute(false);
        let _warm_pre = gpu.acquire(pre).expect("warm pre");
        gpu.release(pre);
        let _warm_dec = gpu.acquire(dec).expect("warm dec");
        gpu.release(dec);
        let t0 = gpu.clock_ns().expect("drain h2d");
        gpu.bind_decode_compute(false);
        let _prefill = gpu.acquire(pre).expect("prefill");
        gpu.release(pre);
        gpu.bind_decode_compute(true);
        let _decode = gpu.acquire(dec).expect("decode");
        gpu.release(dec);
        gpu.score().expect("score").wall_ns.saturating_sub(t0)
    };
    let serial = run(1);
    let overlap = run(2);
    assert!(
        overlap < serial,
        "two Hyper-Q slots must overlap leftover prefill with decode; overlap={overlap} serial={serial}"
    );
}

#[test]
fn simulated_gpu_store_cooperative_serializes_leftover_prefill() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let run = |cooperative: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            2,
            HardwareProfile::example_h100_sxm(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                decode_priority: true,
                stream_priority: true,
                compute_slots: 2,
                cooperative,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        let pre = ExpertKey::new(0, 0);
        let dec = ExpertKey::new(0, 1);
        gpu.bind_decode_compute(false);
        let _warm_pre = gpu.acquire(pre).expect("warm pre");
        gpu.release(pre);
        let _warm_dec = gpu.acquire(dec).expect("warm dec");
        gpu.release(dec);
        let t0 = gpu.clock_ns().expect("drain h2d");
        gpu.bind_decode_compute(false);
        let _prefill = gpu.acquire(pre).expect("prefill");
        gpu.release(pre);
        gpu.bind_decode_compute(true);
        let _decode = gpu.acquire(dec).expect("decode");
        gpu.release(dec);
        gpu.score().expect("score").wall_ns.saturating_sub(t0)
    };
    let overlap = run(false);
    let serial = run(true);
    assert!(
        overlap < serial,
        "cooperative must not overlap leftover prefill with decode; overlap={overlap} serial={serial}"
    );
}

#[test]
fn simulated_gpu_store_compute_slots_overlap_across_token_clock() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let run = |slots: u8| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            2,
            HardwareProfile::example_h100_sxm(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                decode_priority: true,
                stream_priority: true,
                compute_slots: slots,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        let pre = ExpertKey::new(0, 0);
        let dec = ExpertKey::new(0, 1);
        gpu.bind_decode_compute(false);
        let _warm_pre = gpu.acquire(pre).expect("warm pre");
        gpu.release(pre);
        let _warm_dec = gpu.acquire(dec).expect("warm dec");
        gpu.release(dec);
        let t0 = gpu.clock_ns().expect("drain h2d");
        for _ in 0..8 {
            gpu.bind_decode_compute(false);
            let _prefill = gpu.acquire(pre).expect("prefill");
            gpu.release(pre);
            let _tok = gpu.token_clock_ns().expect("token");
            gpu.bind_decode_compute(true);
            let _decode = gpu.acquire(dec).expect("decode");
            gpu.release(dec);
        }
        gpu.score().expect("score").wall_ns.saturating_sub(t0)
    };
    let serial = run(1);
    let overlap = run(2);
    assert!(
        overlap < serial,
        "Hyper-Q must overlap leftover across decode-stream ITL samples; overlap={overlap} serial={serial}"
    );
}

#[test]
fn simulated_gpu_store_decode_sms_lengthens_token_clock() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let run = |sms: u16| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            2,
            profile.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                decode_priority: true,
                stream_priority: true,
                compute_slots: 2,
                decode_sm_permille: sms,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        let pre = ExpertKey::new(0, 0);
        let dec = ExpertKey::new(0, 1);
        gpu.bind_decode_compute(false);
        let _warm_pre = gpu.acquire(pre).expect("warm pre");
        gpu.release(pre);
        let _warm_dec = gpu.acquire(dec).expect("warm dec");
        gpu.release(dec);
        let t0 = gpu.clock_ns().expect("drain h2d");
        gpu.bind_decode_compute(false);
        let _prefill = gpu.acquire(pre).expect("prefill");
        gpu.release(pre);
        gpu.bind_decode_compute(true);
        let _decode = gpu.acquire(dec).expect("decode");
        gpu.release(dec);
        gpu.token_clock_ns().expect("token").saturating_sub(t0)
    };
    let full = run(0);
    let quarter = run(250);
    assert!(
        quarter > full,
        "250‰ decode SMs must lengthen compute-bound ITL; quarter={quarter} full={full}"
    );
}

#[test]
fn sim_replay_compute_slots_overlap_seq_streams() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0]), ev_seq(1, 0, 0, &[1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let cfg = |slots: u8| SimCfg {
        seq_streams: true,
        compute_slots: slots,
        ..SimCfg::lru(2, 4096, 0)
    };
    let serial = sim_replay_cfg(&t, profile.clone(), cfg(1)).expect("serial");
    let overlap = sim_replay_cfg(&t, profile, cfg(2)).expect("overlap");
    assert_eq!(serial.hits, overlap.hits);
    assert_eq!(serial.misses, overlap.misses);
    assert!(
        overlap.sim_ns < serial.sim_ns,
        "two compute slots must overlap independent sequence GEMMs; overlap={} serial={}",
        overlap.sim_ns,
        serial.sim_ns
    );
}

#[test]
fn sim_replay_cooperative_serializes_seq_streams() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0]), ev_seq(1, 0, 0, &[1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let hyperq = SimCfg {
        seq_streams: true,
        compute_slots: 2,
        ..SimCfg::lru(2, 4096, 0)
    };
    let coop = SimCfg {
        cooperative: true,
        ..hyperq
    };
    let overlap = sim_replay_cfg(&t, profile.clone(), hyperq).expect("hyperq");
    let serial = sim_replay_cfg(&t, profile, coop).expect("coop");
    assert_eq!(overlap.hits, serial.hits);
    assert_eq!(overlap.misses, serial.misses);
    assert!(
        overlap.sim_ns < serial.sim_ns,
        "cooperative GEMMs must not Hyper-Q overlap; overlap={} serial={}",
        overlap.sim_ns,
        serial.sim_ns
    );
}

#[test]
fn sim_replay_decode_sms_lengthens_compute_bound() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let cfg = |sms: u16| SimCfg {
        decode_sm_permille: sms,
        ..SimCfg::lru(1, 4096, 0)
    };
    let full = sim_replay_cfg(&t, profile.clone(), cfg(0)).expect("full");
    let quarter = sim_replay_cfg(&t, profile, cfg(250)).expect("quarter");
    assert_eq!(full.misses, quarter.misses);
    assert!(
        quarter.sim_ns > full.sim_ns,
        "250‰ SMs must lengthen a compute-bound replay; quarter={} full={}",
        quarter.sim_ns,
        full.sim_ns
    );
}

#[test]
fn sim_replay_decode_priority_keeps_hits() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg::lru(1, 4096, 0);
    let on = SimCfg {
        decode_priority: true,
        stream_priority: true,
        ..off
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("off");
    let b = sim_replay_cfg(&t, profile, on).expect("on");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn store_replay_decode_priority_binds_later_tokens() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let mut run = StoreReplayCfg::demand(1, 4096, GpuFill::Pinned);
    run.gpu.decode_priority = true;
    run.gpu.stream_priority = true;
    let row = store_replay_cfg(&t, HardwareProfile::example_h100_sxm(), run).expect("store");
    assert_eq!(row.metrics.misses, 1);
    assert_eq!(row.metrics.hits, 1);
    assert!(row.score.wall_ns > 0);
}
