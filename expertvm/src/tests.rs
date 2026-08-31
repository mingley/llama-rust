//! Library tests: JSONL, Zipf-ish traces, policy order, leases, gpu-sim.

use super::*;
use gpu_sim::{AccessProperty, AllocId, DeviceId, EventId, GpuOp, HardwareProfile, Place};
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
        host_register: false,
        host_register_mapped: false,
        sync_memops: false,
        memcpy_batch: false,
        memcpy_during: false,
        memcpy_any: false,
        memcpy_attr: false,
        memset_fill: false,
        copy_host: false,
        accessed_by: false,
        no_read_mostly: false,
        no_preferred: false,
        no_mem_prefetch: false,
        wait_value: false,
        stream_attach: false,
        managed_host: false,
        prefetch_host: false,
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
fn copy_host_lengthens_miss_wall_not_later_hits() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0]), ev(2, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000\nhbm_bps=1000000000000\nhost_func_ns=100000\nlaunch_overhead_ns=1\ncopy_engines=2\n",
    )
    .expect("profile");
    let base = SimCfg::lru(2, 4096, 0);
    let plain = sim_replay_cfg(&t, p.clone(), base).expect("plain");
    let copy = sim_replay_cfg(
        &t,
        p.clone(),
        SimCfg {
            copy_host: true,
            ..base
        },
    )
    .expect("copy-host");
    let gemm = sim_replay_cfg(
        &t,
        p.clone(),
        SimCfg {
            host_func: true,
            ..base
        },
    )
    .expect("host-func");
    let both = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            host_func: true,
            copy_host: true,
            ..base
        },
    )
    .expect("both");
    assert_eq!(plain.hits, copy.hits);
    assert_eq!(plain.misses, copy.misses);
    assert_eq!(copy.misses, 1);
    assert!(
        copy.sim_ns > plain.sim_ns,
        "copy-host={} plain={}",
        copy.sim_ns,
        plain.sim_ns
    );
    assert!(
        gemm.sim_ns > copy.sim_ns,
        "host-func after every GEMM must beat miss-only copy-host; gemm={} copy={}",
        gemm.sim_ns,
        copy.sim_ns
    );
    assert!(
        both.sim_ns > gemm.sim_ns,
        "copy-host must stack with host-func; both={} gemm={}",
        both.sim_ns,
        gemm.sim_ns
    );
}

#[test]
fn sim_replay_copy_host_legal_with_pdl_cooperative_wait_value() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::example_h100_sxm();
    let base = SimCfg::lru(1, 4096, 0);
    let _pdl = sim_replay_cfg(
        &t,
        p.clone(),
        SimCfg {
            copy_host: true,
            pdl: true,
            compute_slots: 2,
            ..base
        },
    )
    .expect("pdl");
    let _coop = sim_replay_cfg(
        &t,
        p.clone(),
        SimCfg {
            copy_host: true,
            cooperative: true,
            ..base
        },
    )
    .expect("coop");
    let wait = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            copy_host: true,
            wait_value: true,
            ..base
        },
    )
    .expect("wait");
    assert_eq!(wait.misses, 1);
}

#[test]
fn sim_replay_copy_host_mapped_is_noop() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000\nhbm_bps=1000000000000\nhost_func_ns=100000\nlaunch_overhead_ns=1\ncopy_engines=2\n",
    )
    .expect("profile");
    let base = SimCfg {
        mapped: true,
        ..SimCfg::lru(1, 4096, 0)
    };
    let plain = sim_replay_cfg(&t, p.clone(), base).expect("mapped");
    let copy = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            copy_host: true,
            ..base
        },
    )
    .expect("mapped copy-host");
    assert_eq!(plain.hits, copy.hits);
    assert_eq!(plain.misses, copy.misses);
    assert_eq!(
        plain.sim_ns, copy.sim_ns,
        "mapped has no device fill; copy-host={} mapped={}",
        copy.sim_ns, plain.sim_ns
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
fn schedule_mem_sync_domain_isolates_leftover_prefill() {
    let mut events = Vec::new();
    for layer in 0..16u32 {
        events.push(ev_seq(0, 0, layer, &[0]));
    }
    events.push(ev_seq(1, 0, 0, &[1]));
    for token in 1..5u32 {
        events.push(ev_seq(1, token, 0, &[1]));
    }
    let t = Trace { events };
    let p = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000\ncopy_engines=2\nsame_domain_fence_permille=1000\n",
    )
    .expect("fence profile");
    let base = SimCfg {
        seq_streams: true,
        compute_slots: 2,
        decode_priority: true,
        stream_priority: true,
        ..SimCfg::lru(4, 4096, 0)
    };
    let same = schedule_replay(&t, p.clone(), base, SchedCfg::chunked(0, 1)).expect("same");
    let iso = schedule_replay(
        &t,
        p,
        SimCfg {
            mem_sync_domain: MemSyncDomain::Remote,
            ..base
        },
        SchedCfg::chunked(0, 1),
    )
    .expect("iso");
    assert_eq!(same.completed, 2);
    assert_eq!(iso.completed, 2);
    let same_itl = same.replay.itl_ns.expect("same itl");
    let iso_itl = iso.replay.itl_ns.expect("iso itl");
    assert!(
        iso_itl < same_itl,
        "Remote decode domain must skip leftover prefill fence; iso={iso_itl} same={same_itl}"
    );
}

#[test]
fn schedule_mem_sync_map_collapse_restores_leftover_prefill_fence() {
    let mut events = Vec::new();
    for layer in 0..16u32 {
        events.push(ev_seq(0, 0, layer, &[0]));
    }
    events.push(ev_seq(1, 0, 0, &[1]));
    for token in 1..5u32 {
        events.push(ev_seq(1, token, 0, &[1]));
    }
    let t = Trace { events };
    let p = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000\ncopy_engines=2\nsame_domain_fence_permille=1000\n",
    )
    .expect("fence profile");
    let base = SimCfg {
        seq_streams: true,
        compute_slots: 2,
        decode_priority: true,
        stream_priority: true,
        mem_sync_domain: MemSyncDomain::Remote,
        ..SimCfg::lru(4, 4096, 0)
    };
    let iso = schedule_replay(&t, p.clone(), base, SchedCfg::chunked(0, 1)).expect("iso");
    let collapsed = schedule_replay(
        &t,
        p,
        SimCfg {
            mem_sync_collapse: true,
            ..base
        },
        SchedCfg::chunked(0, 1),
    )
    .expect("collapse");
    assert_eq!(iso.completed, 2);
    assert_eq!(collapsed.completed, 2);
    let iso_itl = iso.replay.itl_ns.expect("iso itl");
    let col_itl = collapsed.replay.itl_ns.expect("collapse itl");
    assert!(
        col_itl > iso_itl,
        "collapse map must restore leftover prefill fence; collapse={col_itl} iso={iso_itl}"
    );
}

#[test]
fn schedule_mem_sync_launch_restores_leftover_prefill_fence() {
    let mut events = Vec::new();
    for layer in 0..16u32 {
        events.push(ev_seq(0, 0, layer, &[0]));
    }
    events.push(ev_seq(1, 0, 0, &[1]));
    for token in 1..5u32 {
        events.push(ev_seq(1, token, 0, &[1]));
    }
    let t = Trace { events };
    let p = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000\ncopy_engines=2\nsame_domain_fence_permille=1000\n",
    )
    .expect("fence profile");
    let base = SimCfg {
        seq_streams: true,
        compute_slots: 2,
        decode_priority: true,
        stream_priority: true,
        mem_sync_domain: MemSyncDomain::Remote,
        ..SimCfg::lru(4, 4096, 0)
    };
    let iso = schedule_replay(&t, p.clone(), base, SchedCfg::chunked(0, 1)).expect("iso");
    let launch = schedule_replay(
        &t,
        p,
        SimCfg {
            mem_sync_launch: true,
            ..base
        },
        SchedCfg::chunked(0, 1),
    )
    .expect("launch");
    assert_eq!(iso.completed, 2);
    assert_eq!(launch.completed, 2);
    let iso_itl = iso.replay.itl_ns.expect("iso itl");
    let launch_itl = launch.replay.itl_ns.expect("launch itl");
    assert!(
        launch_itl > iso_itl,
        "launch Remote must restore leftover prefill fence; launch={launch_itl} iso={iso_itl}"
    );
}

#[test]
fn schedule_mem_sync_launch_map_restores_leftover_prefill_fence() {
    let mut events = Vec::new();
    for layer in 0..16u32 {
        events.push(ev_seq(0, 0, layer, &[0]));
    }
    events.push(ev_seq(1, 0, 0, &[1]));
    for token in 1..5u32 {
        events.push(ev_seq(1, token, 0, &[1]));
    }
    let t = Trace { events };
    let p = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000\ncopy_engines=2\nsame_domain_fence_permille=1000\n",
    )
    .expect("fence profile");
    let base = SimCfg {
        seq_streams: true,
        compute_slots: 2,
        decode_priority: true,
        stream_priority: true,
        mem_sync_domain: MemSyncDomain::Remote,
        ..SimCfg::lru(4, 4096, 0)
    };
    let iso = schedule_replay(&t, p.clone(), base, SchedCfg::chunked(0, 1)).expect("iso");
    let launch = schedule_replay(
        &t,
        p,
        SimCfg {
            mem_sync_launch_map: true,
            ..base
        },
        SchedCfg::chunked(0, 1),
    )
    .expect("launch-map");
    assert_eq!(iso.completed, 2);
    assert_eq!(launch.completed, 2);
    let iso_itl = iso.replay.itl_ns.expect("iso itl");
    let launch_itl = launch.replay.itl_ns.expect("launch-map itl");
    assert!(
        launch_itl > iso_itl,
        "launch collapse map must restore leftover prefill fence; launch={launch_itl} iso={iso_itl}"
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
fn schedule_managed_no_read_mostly_replicas_lower_hbm() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[1]), ev(2, 0, &[0])],
    };
    let p = HardwareProfile::example_2xh100_pcie();
    let bytes = 1u64 << 20;
    let stripe = striped(&t, 2);
    let hot = with_hot_replicas(stripe, &t, 2, 200);
    let run = |no_read_mostly: bool| {
        let cfg = SimCfg {
            managed: true,
            no_read_mostly,
            ..SimCfg::lru(4, bytes, 0)
        };
        schedule_placed(&t, p.clone(), cfg, SchedCfg::closed(0), Some(&hot)).expect("sched")
    };
    let off = run(false);
    let on = run(true);
    assert_eq!(off.replay.hits, on.replay.hits);
    assert_eq!(off.replay.misses, on.replay.misses);
    assert!(
        on.replay.bytes_moved > off.replay.bytes_moved,
        "no-read-mostly must migrate back after dest prefetch; off={} on={}",
        off.replay.bytes_moved,
        on.replay.bytes_moved
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
fn schedule_managed_no_preferred_remote_migrates() {
    let t = Trace {
        events: vec![ev(0, 0, &[1])],
    };
    let p = HardwareProfile::example_2xh100_pcie();
    let bytes = 1u64 << 20;
    let map = striped(&t, 2);
    let run = |no_preferred: bool| {
        let cfg = SimCfg {
            managed: true,
            no_preferred,
            ..SimCfg::lru(4, bytes, 0)
        };
        schedule_remote(&t, p.clone(), cfg, SchedCfg::closed(0), &map, bytes).expect("sched")
    };
    let off = run(false);
    let on = run(true);
    assert_eq!(off.replay.hits, on.replay.hits);
    assert_eq!(off.replay.misses, on.replay.misses);
    assert!(
        off.replay.bytes_moved < bytes.saturating_mul(2),
        "preferred-location GEMM must not D2D the expert; moved={}",
        off.replay.bytes_moved
    );
    assert!(
        on.replay.bytes_moved > off.replay.bytes_moved,
        "no-preferred must first-touch migrate; off={} on={}",
        off.replay.bytes_moved,
        on.replay.bytes_moved
    );
}

#[test]
fn schedule_managed_no_mem_prefetch_keeps_hits() {
    let t = Trace {
        events: vec![ev(0, 0, &[1])],
    };
    let p = HardwareProfile::example_2xh100_pcie();
    let bytes = 1u64 << 20;
    let map = striped(&t, 2);
    let run = |no_mem_prefetch: bool| {
        let cfg = SimCfg {
            managed: true,
            no_mem_prefetch,
            ..SimCfg::lru(4, bytes, 0)
        };
        schedule_remote(&t, p.clone(), cfg, SchedCfg::closed(0), &map, bytes).expect("sched")
    };
    let off = run(false);
    let on = run(true);
    assert_eq!(off.replay.hits, on.replay.hits);
    assert_eq!(off.replay.misses, on.replay.misses);
    assert!(
        off.replay.bytes_moved < bytes.saturating_mul(2),
        "preferred-location fill prefetch must not D2D the expert; moved={}",
        off.replay.bytes_moved
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
fn simulated_gpu_store_no_read_mostly_pin_moves() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let k0 = ExpertKey::new(0, 0);
    let p = HardwareProfile::example_2xh100_pcie();
    let bytes = 4096u64;
    let mut off = SimulatedGpuStore::with_managed(DirectStore::from_trace(&t), 2, p.clone(), bytes)
        .expect("read-mostly");
    off.pin_hot(&[k0]).expect("pin");
    let _off_score = off.score().expect("score");
    assert!(off.page_read_mostly(k0));
    assert!(off.page_resident(k0, DeviceId(0)));
    assert!(off.page_resident(k0, DeviceId(1)));
    let off_sum =
        off.hbm_used(DeviceId(0)).expect("off0") + off.hbm_used(DeviceId(1)).expect("off1");
    assert!(off_sum >= bytes * 2, "{off_sum}");

    let mut on = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        2,
        p,
        bytes,
        GpuFill::Managed,
        GpuStoreCfg {
            no_read_mostly: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("no-read-mostly");
    on.pin_hot(&[k0]).expect("pin");
    let _on_score = on.score().expect("score");
    assert!(!on.page_read_mostly(k0));
    assert!(!on.page_resident(k0, DeviceId(0)));
    assert!(on.page_resident(k0, DeviceId(1)));
    let on_sum = on.hbm_used(DeviceId(0)).expect("on0") + on.hbm_used(DeviceId(1)).expect("on1");
    assert_eq!(on_sum, bytes);
    assert!(
        on_sum < off_sum,
        "no-read-mostly cluster={on_sum} read-mostly cluster={off_sum}"
    );
}

#[test]
fn simulated_gpu_store_no_read_mostly_pin_gemm_moves_back() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let k0 = ExpertKey::new(0, 0);
    let p = HardwareProfile::example_2xh100_pcie();
    let bytes = 4096u64;
    let run = |no_read_mostly: bool| {
        let mut gpu = SimulatedGpuStore::with_cfg(
            DirectStore::from_trace(&t),
            2,
            p.clone(),
            bytes,
            GpuFill::Managed,
            GpuStoreCfg {
                no_read_mostly,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        gpu.pin_hot(&[k0]).expect("pin");
        let _p = gpu.acquire(k0).expect("gemm");
        let score = gpu.score().expect("score");
        let cluster =
            gpu.hbm_used(DeviceId(0)).expect("h0") + gpu.hbm_used(DeviceId(1)).expect("h1");
        (
            score.bytes_moved,
            cluster,
            gpu.page_resident(k0, DeviceId(0)),
            gpu.page_resident(k0, DeviceId(1)),
        )
    };
    let off = run(false);
    let on = run(true);
    assert!(
        on.0 > off.0,
        "no-read-mostly must migrate back; off={} on={}",
        off.0,
        on.0
    );
    assert!(off.2 && off.3, "read-mostly pin keeps both copies");
    assert!(on.2 && !on.3, "no-read-mostly GEMM moves back to home");
    assert!(
        on.1 < off.1,
        "no-read-mostly must not keep two copies; off={} on={}",
        off.1,
        on.1
    );
}

#[test]
fn simulated_gpu_store_no_read_mostly_needs_managed() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            no_read_mostly: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("no-read-mostly without managed must fail"),
        Err(err) => assert!(
            err.to_string().contains("no-read-mostly needs managed"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_no_preferred_unsets_home() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let k0 = ExpertKey::new(0, 0);
    let p = HardwareProfile::example_2xh100_pcie();
    let bytes = 4096u64;
    let mut off = SimulatedGpuStore::with_managed(DirectStore::from_trace(&t), 2, p.clone(), bytes)
        .expect("preferred");
    let _p = off.acquire(k0).expect("acq");
    let _s = off.score().expect("score");
    assert!(off.page_preferred(k0, DeviceId(0)));
    assert!(off.page_read_mostly(k0));

    let mut on = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        2,
        p,
        bytes,
        GpuFill::Managed,
        GpuStoreCfg {
            no_preferred: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("no-preferred");
    let _p = on.acquire(k0).expect("acq");
    let _s = on.score().expect("score");
    assert!(!on.page_preferred(k0, DeviceId(0)));
    assert!(on.page_read_mostly(k0));
}

#[test]
fn simulated_gpu_store_no_preferred_needs_managed() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            no_preferred: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("no-preferred without managed must fail"),
        Err(err) => assert!(
            err.to_string().contains("no-preferred needs managed"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_no_mem_prefetch_needs_managed() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            no_mem_prefetch: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("no-mem-prefetch without managed must fail"),
        Err(err) => assert!(
            err.to_string().contains("no-mem-prefetch needs managed"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_no_mem_prefetch_keeps_advise() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let k0 = ExpertKey::new(0, 0);
    let mut gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Managed,
        GpuStoreCfg {
            no_mem_prefetch: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("no-mem-prefetch");
    let _p = gpu.acquire(k0).expect("acq");
    let _s = gpu.score().expect("score");
    assert!(gpu.page_preferred(k0, DeviceId(0)));
    assert!(gpu.page_read_mostly(k0));
}

#[test]
fn simulated_gpu_store_no_mem_prefetch_serializes_miss_prefetch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let profile =
        HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n").expect("slow gemm");
    let run = |no_mem_prefetch: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = match SimulatedGpuStore::with_cfg(
            inner,
            2,
            profile.clone(),
            4096,
            GpuFill::Managed,
            GpuStoreCfg {
                no_mem_prefetch,
                ..GpuStoreCfg::default()
            },
        ) {
            Ok(gpu) => gpu,
            Err(err) => panic!("gpu: {err}"),
        };
        let k0 = ExpertKey::new(0, 0);
        let k1 = ExpertKey::new(0, 1);
        let _p = match gpu.acquire(k0) {
            Ok(v) => v,
            Err(err) => panic!("warm: {err}"),
        };
        gpu.release(k0);
        let _d = gpu.clock_ns().expect("drain");
        let _p = match gpu.acquire(k0) {
            Ok(v) => v,
            Err(err) => panic!("leftover: {err}"),
        };
        gpu.release(k0);
        let _p = match gpu.acquire(k1) {
            Ok(v) => v,
            Err(err) => panic!("miss: {err}"),
        };
        gpu.release(k1);
        let score = gpu.score().expect("score");
        let kernels: Vec<_> = gpu
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .collect();
        let prefetches: Vec<_> = gpu
            .operations()
            .filter(|o| match &o.kind {
                GpuOp::Memcpy(m) => !matches!((m.src, m.dst), (Place::Device(_), Place::Device(_))),
                _ => false,
            })
            .collect();
        let leftover = kernels
            .get(kernels.len().saturating_sub(2))
            .expect("leftover gemm");
        let miss = prefetches.last().expect("miss prefetch");
        (
            score.wall_ns,
            leftover.done_ns.expect("k done"),
            miss.start_ns.expect("prefetch start"),
        )
    };
    let (off_wall, off_done, off_pf) = run(false);
    let (on_wall, on_done, on_pf) = run(true);
    assert!(
        off_pf < off_done,
        "identity managed prefetch must overlap leftover GEMM; pf={off_pf} kdone={off_done}"
    );
    assert!(
        on_pf >= on_done,
        "no-mem-prefetch first-touch must wait leftover GEMM; pf={on_pf} kdone={on_done}"
    );
    assert!(
        on_wall > off_wall,
        "no-mem-prefetch must lengthen wall; on={on_wall} off={off_wall}"
    );
}

#[test]
fn simulated_gpu_store_no_mem_prefetch_pin_still_replicates() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let mut gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        2,
        HardwareProfile::example_8xh100_nvlink(),
        4096,
        GpuFill::Managed,
        GpuStoreCfg {
            no_mem_prefetch: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    gpu.pin_hot(&[k0]).expect("pin");
    assert_eq!(gpu.replica_of(k0), Some(DeviceId(1)));
    assert_eq!(gpu.metrics().replicates, 1);
    let _s = gpu.score().expect("score");
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
fn cuda_graphs_graph_enable_skips_combo_recapture() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1, 2]), ev(1, 0, &[0, 1])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nlaunch_overhead_ns=1000\ngraph_launch_ns=1000\ngraph_instantiate_ns=100000\ngraph_set_params_ns=1000\ngraph_upload_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let base = SimCfg {
        cuda_graphs: true,
        ..SimCfg::lru(3, 4096, 0)
    };
    let recapture = sim_replay_cfg(&t, p.clone(), base).expect("recapture");
    let enable = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            graph_enable: true,
            ..base
        },
    )
    .expect("enable");
    assert_eq!(recapture.child_graphs, 2);
    assert_eq!(enable.child_graphs, 1);
    assert_eq!(enable.graph_launches, 2);
    assert_eq!(recapture.graph_launches, 2);
    assert!(
        enable.sim_ns < recapture.sim_ns,
        "enable={} recapture={}",
        enable.sim_ns,
        recapture.sim_ns
    );
}

#[test]
fn cuda_graphs_graph_enable_restores_disabled_children() {
    let t = Trace {
        events: vec![
            ev(0, 0, &[0, 1, 2]),
            ev(1, 0, &[0, 1]),
            ev(2, 0, &[0, 1, 2]),
        ],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nlaunch_overhead_ns=1000\ngraph_launch_ns=1000\ngraph_instantiate_ns=100000\ngraph_set_params_ns=1000\ngraph_upload_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let enable = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            cuda_graphs: true,
            graph_enable: true,
            ..SimCfg::lru(3, 4096, 0)
        },
    )
    .expect("enable");
    assert_eq!(enable.child_graphs, 1);
    assert_eq!(enable.graph_launches, 3);
}

#[test]
fn cuda_graphs_graph_enable_captures_resident_cover() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1]), ev(1, 0, &[1, 2]), ev(2, 0, &[0, 2])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nlaunch_overhead_ns=1000\ngraph_launch_ns=1000\ngraph_instantiate_ns=100000\ngraph_set_params_ns=1000\ngraph_upload_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let recapture = sim_replay_cfg(
        &t,
        p.clone(),
        SimCfg {
            cuda_graphs: true,
            ..SimCfg::lru(3, 4096, 0)
        },
    )
    .expect("recapture");
    let enable = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            cuda_graphs: true,
            graph_enable: true,
            ..SimCfg::lru(3, 4096, 0)
        },
    )
    .expect("enable");
    assert_eq!(recapture.child_graphs, 3);
    assert_eq!(enable.child_graphs, 2);
    assert_eq!(enable.graph_launches, 3);
}

#[test]
fn sim_cfg_graph_enable_refuses_device_launch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let err = match sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            cuda_graphs: true,
            graph_enable: true,
            device_launch: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-enable + device-launch must fail"),
        Err(e) => e,
    };
    assert!(
        err.to_string()
            .contains("graph-enable cannot device-launch"),
        "{err}"
    );
}

#[test]
fn simulated_gpu_store_graph_enable_refuses_device_launch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_enable: true,
            device_launch: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("graph-enable + device-launch must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("graph-enable cannot device-launch"),
            "{err}"
        ),
    }
}

#[test]
fn graph_enable_implies_cuda_graphs() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1]), ev(1, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nlaunch_overhead_ns=1000\ngraph_launch_ns=1000\ngraph_instantiate_ns=1000\ngraph_set_params_ns=1000\ngraph_upload_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let g = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            graph_enable: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    )
    .expect("enable");
    assert!(g.graph_launches > 0, "line={}", g.line());
    assert_eq!(g.child_graphs, 1);
}

#[test]
fn cuda_graphs_graph_if_skips_combo_recapture() {
    let t = Trace {
        events: vec![
            ev(0, 0, &[0, 1, 2, 3]),
            ev(1, 0, &[0, 1]),
            ev(2, 0, &[0, 2]),
            ev(3, 0, &[0, 3]),
            ev(4, 0, &[1, 2]),
            ev(5, 0, &[1, 3]),
        ],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nlaunch_overhead_ns=1000\ngraph_launch_ns=1000\ngraph_instantiate_ns=100000\ngraph_set_params_ns=1000\ngraph_upload_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let base = SimCfg {
        graph_build: true,
        ..SimCfg::lru(4, 4096, 0)
    };
    let recapture = sim_replay_cfg(&t, p.clone(), base).expect("recapture");
    let gated = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            graph_if: true,
            ..base
        },
    )
    .expect("graph-if");
    assert_eq!(recapture.child_graphs, 6);
    assert_eq!(gated.child_graphs, 1);
    assert_eq!(gated.graph_launches, 6);
    assert_eq!(recapture.graph_launches, 6);
    assert_eq!(gated.hits, recapture.hits);
    assert_eq!(gated.misses, recapture.misses);
    assert!(
        gated.sim_ns < recapture.sim_ns,
        "if={} recapture={}",
        gated.sim_ns,
        recapture.sim_ns
    );
}

#[test]
fn cuda_graphs_graph_if_restores_disabled_children() {
    let t = Trace {
        events: vec![
            ev(0, 0, &[0, 1, 2]),
            ev(1, 0, &[0, 1]),
            ev(2, 0, &[0, 1, 2]),
        ],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nlaunch_overhead_ns=1000\ngraph_launch_ns=1000\ngraph_instantiate_ns=100000\ngraph_set_params_ns=1000\ngraph_upload_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let gated = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            graph_build: true,
            graph_if: true,
            ..SimCfg::lru(3, 4096, 0)
        },
    )
    .expect("graph-if");
    assert_eq!(gated.child_graphs, 1);
    assert_eq!(gated.graph_launches, 3);
}

#[test]
fn cuda_graphs_graph_if_captures_resident_cover() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1]), ev(1, 0, &[1, 2]), ev(2, 0, &[0, 2])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nlaunch_overhead_ns=1000\ngraph_launch_ns=1000\ngraph_instantiate_ns=100000\ngraph_set_params_ns=1000\ngraph_upload_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let recapture = sim_replay_cfg(
        &t,
        p.clone(),
        SimCfg {
            graph_build: true,
            ..SimCfg::lru(3, 4096, 0)
        },
    )
    .expect("recapture");
    let gated = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            graph_build: true,
            graph_if: true,
            ..SimCfg::lru(3, 4096, 0)
        },
    )
    .expect("graph-if");
    assert_eq!(recapture.child_graphs, 3);
    assert_eq!(gated.child_graphs, 2);
    assert_eq!(gated.graph_launches, 3);
}

#[test]
fn cuda_graphs_graph_if_reupload_beats_graph_enable() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1, 2]), ev(1, 0, &[0, 1])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nlaunch_overhead_ns=1000\ngraph_launch_ns=1000\ngraph_instantiate_ns=100000\ngraph_set_params_ns=1000\ngraph_upload_ns=50000\ncopy_engines=2\n",
    )
    .expect("profile");
    let enable = sim_replay_cfg(
        &t,
        p.clone(),
        SimCfg {
            cuda_graphs: true,
            graph_enable: true,
            ..SimCfg::lru(3, 4096, 0)
        },
    )
    .expect("enable");
    let gated = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            graph_build: true,
            graph_if: true,
            ..SimCfg::lru(3, 4096, 0)
        },
    )
    .expect("graph-if");
    assert_eq!(enable.child_graphs, 1);
    assert_eq!(gated.child_graphs, 1);
    assert_eq!(enable.graph_launches, 2);
    assert_eq!(gated.graph_launches, 2);
    assert!(
        enable.sim_ns < gated.sim_ns,
        "enable={} if={}",
        enable.sim_ns,
        gated.sim_ns
    );
}

#[test]
fn sim_cfg_graph_if_needs_graph_build() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let err = match sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            graph_if: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-if without graph-build must fail"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("graph-if needs graph-build"),
        "{err}"
    );
}

#[test]
fn sim_cfg_graph_if_refuses_device_launch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let err = match sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            graph_build: true,
            graph_if: true,
            device_launch: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-if + device-launch must fail"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("graph-if cannot device-launch"),
        "{err}"
    );
}

#[test]
fn sim_cfg_graph_if_refuses_graph_enable() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let err = match sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            graph_build: true,
            graph_if: true,
            graph_enable: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-if + graph-enable must fail"),
        Err(e) => e,
    };
    assert!(
        err.to_string()
            .contains("choose one of graph-if, graph-enable"),
        "{err}"
    );
}

#[test]
fn simulated_gpu_store_graph_if_refuses_device_launch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_build: true,
            graph_if: true,
            device_launch: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("graph-if + device-launch must fail"),
        Err(err) => assert!(
            err.to_string().contains("graph-if cannot device-launch"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_graph_if_needs_graph_build() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_if: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("graph-if without graph-build must fail"),
        Err(err) => assert!(
            err.to_string().contains("graph-if needs graph-build"),
            "{err}"
        ),
    }
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
            ..MemcpyOp::default()
        },
    )
    .expect("m");
    let _ = sim.instantiate_graph(exec).expect("i");
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
    let _ = sim.instantiate_graph(exec).expect("i");
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
fn graph_node_set_enabled_skips_combo_child() {
    use gpu_sim::{KernelKind, Sim, StreamId};

    let mut sim = Sim::new(HardwareProfile::example_h100_sxm());
    let d = DeviceId(0);
    let s = StreamId(0);
    let a = sim.malloc(d, 4096).expect("a");
    let leaf0 = sim.create_graph(d, s).expect("l0");
    sim.graph_add_kernel(leaf0, KernelKind::other(8, 8), &[a], &[a])
        .expect("k0");
    let _ = sim.instantiate_graph(leaf0).expect("i0");
    let leaf1 = sim.create_graph(d, s).expect("l1");
    sim.graph_add_kernel(leaf1, KernelKind::other(8, 8), &[a], &[a])
        .expect("k1");
    let _ = sim.instantiate_graph(leaf1).expect("i1");
    let parent = sim.create_graph(d, s).expect("p");
    sim.graph_add_child(parent, leaf0).expect("c0");
    sim.graph_add_child(parent, leaf1).expect("c1");
    let _ = sim.instantiate_graph(parent).expect("ip");
    assert!(sim.graph_node_get_enabled(parent, 1).expect("on"));
    sim.graph_node_set_enabled(parent, 1, false).expect("off");
    let n = sim.launch_graph(parent, s).expect("launch");
    assert_eq!(n, 1);
    sim.synchronize().expect("sync");
}

#[test]
fn graph_exec_child_set_params_swaps_combo_child() {
    use gpu_sim::{KernelKind, Sim, StreamId};

    let mut sim = Sim::new(HardwareProfile::example_h100_sxm());
    let d = DeviceId(0);
    let s = StreamId(0);
    let a = sim.malloc(d, 4096).expect("a");
    let b = sim.malloc(d, 4096).expect("b");
    let leaf_a = sim.create_graph(d, s).expect("la");
    sim.graph_add_kernel(leaf_a, KernelKind::other(8, 8), &[a], &[a])
        .expect("ka");
    let leaf_a_exec = sim.instantiate_graph(leaf_a).expect("ia");
    let leaf_b = sim.create_graph(d, s).expect("lb");
    sim.graph_add_kernel(leaf_b, KernelKind::other(8, 8), &[b], &[b])
        .expect("kb");
    let _ = sim.instantiate_graph(leaf_b).expect("ib");
    let parent = sim.create_graph(d, s).expect("p");
    sim.graph_add_child(parent, leaf_a).expect("c");
    let _ = sim.instantiate_graph(parent).expect("ip");
    let (node, nested) = sim.graph_unique_child(parent).expect("u");
    assert_eq!(nested, leaf_a_exec);
    sim.graph_exec_child_set_params(parent, node, leaf_b)
        .expect("set");
    sim.free_sync(a).expect("free");
    let n = sim.launch_graph(parent, s).expect("launch");
    assert_eq!(n, 1);
    sim.synchronize().expect("sync");
}

#[test]
fn cuda_graphs_graph_set_params_retargets_combo_parent() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1]), ev(1, 0, &[2, 3])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=100000\ngraph_update_ns=1000\ngraph_set_params_ns=100\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let base = SimCfg {
        cuda_graphs: true,
        graph_build: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let inst = sim_replay_cfg(&t, p.clone(), base).expect("inst");
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
    assert_eq!(inst.child_graphs, set.child_graphs);
    assert_eq!(inst.graph_set_params, 0);
    assert_eq!(set.graph_set_params, 3);
    assert!(
        set.sim_ns < inst.sim_ns,
        "set={} instantiate={}",
        set.sim_ns,
        inst.sim_ns
    );
    assert!(set.line().contains("graph_set_params="));
}

#[test]
fn graph_exec_event_record_set_event_retargets() {
    use gpu_sim::{EventId, Sim, StreamId};

    let mut sim = Sim::new(HardwareProfile::example_h100_sxm());
    let d = DeviceId(0);
    let s = StreamId(0);
    let e1 = EventId(1);
    let e2 = EventId(2);
    sim.create_event(e1).expect("e1");
    sim.create_event(e2).expect("e2");
    let exec = sim.create_graph(d, s).expect("g");
    sim.graph_add_event_record(exec, e1, false).expect("rec");
    let _ = sim.instantiate_graph(exec).expect("i");
    let (node, ev) = sim.graph_unique_event_record(exec).expect("u");
    assert_eq!(ev, e1);
    sim.graph_exec_event_record_set_event(exec, node, e2)
        .expect("set");
    let n = sim.launch_graph(exec, s).expect("launch");
    assert_eq!(n, 1);
    sim.synchronize().expect("sync");
    assert!(sim.query_event(e2).expect("q2"));
    assert!(!sim.query_event(e1).expect("q1"));
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
fn simulated_gpu_store_graph_build_and_piecewise_conflict() {
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
            graph_build: true,
            graph_piecewise: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("both flags"),
        Err(e) => e,
    };
    assert!(
        err.to_string()
            .contains("choose one of graph-build, graph-piecewise"),
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
fn cuda_graphs_graph_clone_parent_copies_combo_before_instantiate() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_clone_ns=80000\ngraph_instantiate_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let base = SimCfg {
        cuda_graphs: true,
        compute_slots: 2,
        ..SimCfg::lru(2, 4096, 0)
    };
    let run = |graph_clone: bool, graph_clone_parent: bool| {
        sim_replay_cfg(
            &t,
            p.clone(),
            SimCfg {
                graph_clone,
                graph_clone_parent,
                ..base
            },
        )
        .expect("replay")
    };
    let plain = run(false, false);
    let leaves = run(true, false);
    let parent = run(false, true);
    let both = run(true, true);
    assert_eq!(plain.hits, parent.hits);
    assert_eq!(plain.misses, parent.misses);
    assert_eq!(leaves.hits, parent.hits);
    assert_eq!(both.hits, parent.hits);
    assert_eq!(plain.graph_clones, 0);
    assert_eq!(leaves.graph_clones, 2);
    assert_eq!(parent.graph_clones, 1);
    assert_eq!(both.graph_clones, 3);
    assert!(
        parent.sim_ns > plain.sim_ns,
        "graph-clone-parent must bill recursive clone_ns; parent={} plain={}",
        parent.sim_ns,
        plain.sim_ns
    );
    assert!(
        parent.sim_ns > leaves.sim_ns,
        "combo parent clone must exceed two leaf clones; parent={} leaves={}",
        parent.sim_ns,
        leaves.sim_ns
    );
    assert!(
        both.sim_ns > parent.sim_ns,
        "leaf+parent clone must stack; both={} parent={}",
        both.sim_ns,
        parent.sim_ns
    );
    let pdl = sim_replay_cfg(
        &t,
        p.clone(),
        SimCfg {
            graph_clone_parent: true,
            pdl: true,
            ..base
        },
    )
    .expect("pdl");
    assert_eq!(pdl.hits, parent.hits);
    assert_eq!(pdl.graph_clones, 1);
    let bld = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            graph_build: true,
            graph_clone_parent: true,
            ..base
        },
    )
    .expect("build");
    assert_eq!(bld.hits, parent.hits);
    assert_eq!(bld.graph_clones, 1);
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
fn cuda_graphs_graph_build_deps_serializes_combo_children() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000\nhbm_bps=1000000000000\ngraph_instantiate_ns=1\ngraph_upload_ns=1\ngraph_launch_ns=1\nlaunch_overhead_ns=1\ncopy_engines=2\n",
    )
    .expect("profile");
    let run = |graph_build_deps: bool| {
        sim_replay_cfg(
            &t,
            p.clone(),
            SimCfg {
                cuda_graphs: true,
                graph_build: true,
                graph_build_deps,
                compute_slots: 2,
                ..SimCfg::lru(2, 4096, 0)
            },
        )
        .expect("replay")
    };
    let bld = run(false);
    let chained = run(true);
    assert_eq!(bld.hits, chained.hits);
    assert_eq!(bld.misses, chained.misses);
    assert!(
        chained.sim_ns > bld.sim_ns,
        "graph-build-deps must serialize combo children; chained={} build={}",
        chained.sim_ns,
        bld.sim_ns
    );
}

#[test]
fn cuda_graphs_graph_host_serializes_combo_children_with_host_tax() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000\nhbm_bps=1000000000000\ngraph_instantiate_ns=1\ngraph_upload_ns=1\ngraph_launch_ns=1\nlaunch_overhead_ns=1\ncopy_engines=2\nhost_func_ns=100000\n",
    )
    .expect("profile");
    let run = |graph_build_deps: bool, graph_host: bool| {
        sim_replay_cfg(
            &t,
            p.clone(),
            SimCfg {
                cuda_graphs: true,
                graph_build: true,
                graph_build_deps,
                graph_host,
                compute_slots: 2,
                ..SimCfg::lru(2, 4096, 0)
            },
        )
        .expect("replay")
    };
    let bld = run(false, false);
    let chained = run(true, false);
    let hosted = run(false, true);
    assert_eq!(bld.hits, hosted.hits);
    assert_eq!(bld.misses, hosted.misses);
    assert_eq!(chained.hits, hosted.hits);
    assert!(
        bld.sim_ns < hosted.sim_ns,
        "graph-host must serialize combo children with host tax; host={} build={}",
        hosted.sim_ns,
        bld.sim_ns
    );
    assert!(
        chained.sim_ns < hosted.sim_ns,
        "graph-host must add host_func_ns over build-deps; host={} deps={}",
        hosted.sim_ns,
        chained.sim_ns
    );
}

#[test]
fn cuda_graphs_graph_piecewise_independent_children_overlap() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000\nhbm_bps=1000000000000\ngraph_instantiate_ns=1\ngraph_upload_ns=1\ngraph_launch_ns=1\nlaunch_overhead_ns=1\ncopy_engines=2\n",
    )
    .expect("profile");
    let run = |graph_piecewise: bool| {
        sim_replay_cfg(
            &t,
            p.clone(),
            SimCfg {
                cuda_graphs: true,
                graph_piecewise,
                compute_slots: 2,
                ..SimCfg::lru(2, 4096, 0)
            },
        )
        .expect("replay")
        .sim_ns
    };
    let cap = run(false);
    let piece = run(true);
    assert!(
        piece < cap,
        "graph-piecewise combo roots must Hyper-Q overlap capture; piecewise={piece} capture={cap}"
    );
}

#[test]
fn cuda_graphs_graph_capture_deps_serializes_piecewise_children() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000\nhbm_bps=1000000000000\ngraph_instantiate_ns=1\ngraph_upload_ns=1\ngraph_launch_ns=1\nlaunch_overhead_ns=1\ncopy_engines=2\n",
    )
    .expect("profile");
    let run = |graph_capture_deps: bool| {
        sim_replay_cfg(
            &t,
            p.clone(),
            SimCfg {
                cuda_graphs: true,
                graph_piecewise: true,
                graph_capture_deps,
                compute_slots: 2,
                ..SimCfg::lru(2, 4096, 0)
            },
        )
        .expect("replay")
    };
    let piece = run(false);
    let chained = run(true);
    assert_eq!(piece.hits, chained.hits);
    assert_eq!(piece.misses, chained.misses);
    assert!(
        chained.sim_ns > piece.sim_ns,
        "graph-capture-deps must serialize piecewise combo children; chained={} piecewise={}",
        chained.sim_ns,
        piece.sim_ns
    );
}

#[test]
fn cuda_graphs_graph_capture_host_serializes_piecewise_children_with_host_tax() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000\nhbm_bps=1000000000000\ngraph_instantiate_ns=1\ngraph_upload_ns=1\ngraph_launch_ns=1\nlaunch_overhead_ns=1\ncopy_engines=2\nhost_func_ns=100000\n",
    )
    .expect("profile");
    let run = |graph_capture_deps: bool, graph_capture_host: bool| {
        sim_replay_cfg(
            &t,
            p.clone(),
            SimCfg {
                cuda_graphs: true,
                graph_piecewise: true,
                graph_capture_deps,
                graph_capture_host,
                compute_slots: 2,
                ..SimCfg::lru(2, 4096, 0)
            },
        )
        .expect("replay")
    };
    let piece = run(false, false);
    let chained = run(true, false);
    let hosted = run(false, true);
    let both = run(true, true);
    assert_eq!(piece.hits, hosted.hits);
    assert_eq!(piece.misses, hosted.misses);
    assert_eq!(chained.hits, hosted.hits);
    assert_eq!(both.hits, hosted.hits);
    assert!(
        piece.sim_ns < hosted.sim_ns,
        "graph-capture-host must serialize piecewise children with host tax; host={} piecewise={}",
        hosted.sim_ns,
        piece.sim_ns
    );
    assert!(
        chained.sim_ns < hosted.sim_ns,
        "graph-capture-host must add host_func_ns over capture-deps; host={} deps={}",
        hosted.sim_ns,
        chained.sim_ns
    );
    assert_eq!(
        hosted.sim_ns, both.sim_ns,
        "capture-host already chains through the host node; capture-deps must not double-tax; host={} both={}",
        hosted.sim_ns, both.sim_ns
    );
}

#[test]
fn sim_cfg_graph_build_and_piecewise_conflict() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let err = sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            cuda_graphs: true,
            graph_build: true,
            graph_piecewise: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    )
    .expect_err("conflict");
    assert!(
        err.to_string()
            .contains("choose one of graph-build, graph-piecewise"),
        "{err}"
    );
}

#[test]
fn sim_replay_graph_build_deps_needs_graph_build() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    match sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            graph_build_deps: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-build-deps without graph-build must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("graph-build-deps needs graph-build"),
            "{err}"
        ),
    }
    match sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            cuda_graphs: true,
            graph_build_deps: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-build-deps with cuda-graphs still needs graph-build"),
        Err(err) => assert!(
            err.to_string()
                .contains("graph-build-deps needs graph-build"),
            "{err}"
        ),
    }
    match sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            cuda_graphs: true,
            graph_piecewise: true,
            graph_build_deps: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-build-deps with piecewise still needs graph-build"),
        Err(err) => assert!(
            err.to_string()
                .contains("graph-build-deps needs graph-build"),
            "{err}"
        ),
    }
    let _ok = sim_replay_cfg(
        &t,
        profile,
        SimCfg {
            cuda_graphs: true,
            graph_build: true,
            graph_build_deps: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    )
    .expect("graph-build arms build-deps");
}

#[test]
fn sim_replay_graph_host_needs_graph_build() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    match sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            graph_host: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-host without graph-build must fail"),
        Err(err) => assert!(
            err.to_string().contains("graph-host needs graph-build"),
            "{err}"
        ),
    }
    match sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            cuda_graphs: true,
            graph_host: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-host with cuda-graphs still needs graph-build"),
        Err(err) => assert!(
            err.to_string().contains("graph-host needs graph-build"),
            "{err}"
        ),
    }
    match sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            cuda_graphs: true,
            graph_piecewise: true,
            graph_host: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-host with piecewise still needs graph-build"),
        Err(err) => assert!(
            err.to_string().contains("graph-host needs graph-build"),
            "{err}"
        ),
    }
    let _ok = sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            cuda_graphs: true,
            graph_build: true,
            graph_host: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    )
    .expect("graph-build arms graph-host");
    let _both = sim_replay_cfg(
        &t,
        profile,
        SimCfg {
            cuda_graphs: true,
            graph_build: true,
            graph_build_deps: true,
            graph_host: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    )
    .expect("graph-host with build-deps");
}

#[test]
fn sim_replay_graph_memset_needs_graph_mem() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    match sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            graph_memset: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-memset without graph-mem must fail"),
        Err(err) => assert!(
            err.to_string().contains("graph-memset needs graph-mem"),
            "{err}"
        ),
    }
    match sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            cuda_graphs: true,
            graph_memset: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-memset with cuda-graphs still needs graph-mem"),
        Err(err) => assert!(
            err.to_string().contains("graph-memset needs graph-mem"),
            "{err}"
        ),
    }
    match sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            cuda_graphs: true,
            graph_auto_free: true,
            graph_memset: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-memset with auto-free still needs graph-mem"),
        Err(err) => assert!(
            err.to_string().contains("graph-memset needs graph-mem"),
            "{err}"
        ),
    }
    let _ok = sim_replay_cfg(
        &t,
        profile,
        SimCfg {
            cuda_graphs: true,
            graph_mem: true,
            graph_memset: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    )
    .expect("graph-mem arms graph-memset");
}

#[test]
fn sim_replay_graph_memcpy_needs_graph_mem() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    match sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            graph_memcpy: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-memcpy without graph-mem must fail"),
        Err(err) => assert!(
            err.to_string().contains("graph-memcpy needs graph-mem"),
            "{err}"
        ),
    }
    match sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            cuda_graphs: true,
            graph_memcpy: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-memcpy with cuda-graphs still needs graph-mem"),
        Err(err) => assert!(
            err.to_string().contains("graph-memcpy needs graph-mem"),
            "{err}"
        ),
    }
    match sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            cuda_graphs: true,
            graph_auto_free: true,
            graph_memcpy: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-memcpy with auto-free still needs graph-mem"),
        Err(err) => assert!(
            err.to_string().contains("graph-memcpy needs graph-mem"),
            "{err}"
        ),
    }
    let _ok = sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            cuda_graphs: true,
            graph_mem: true,
            graph_memcpy: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    )
    .expect("graph-mem arms graph-memcpy");
    let _both = sim_replay_cfg(
        &t,
        profile,
        SimCfg {
            cuda_graphs: true,
            graph_mem: true,
            graph_memset: true,
            graph_memcpy: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    )
    .expect("memset and memcpy together need graph-mem");
}

#[test]
fn sim_replay_graph_leaf_host_cannot_device_launch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    match sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            graph_leaf_host: true,
            device_launch: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-leaf-host with device-launch must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("graph-leaf-host cannot device-launch"),
            "{err}"
        ),
    }
    let _ok = sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            graph_leaf_host: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    )
    .expect("leaf-host implies graphs");
    let _mem = sim_replay_cfg(
        &t,
        profile,
        SimCfg {
            cuda_graphs: true,
            graph_mem: true,
            graph_leaf_host: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    )
    .expect("leaf-host is legal with graph-mem");
}

#[test]
fn sim_replay_graph_capture_deps_needs_graph_piecewise() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    match sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            graph_capture_deps: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-capture-deps without piecewise must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("graph-capture-deps needs graph-piecewise"),
            "{err}"
        ),
    }
    match sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            cuda_graphs: true,
            graph_capture_deps: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-capture-deps with cuda-graphs still needs piecewise"),
        Err(err) => assert!(
            err.to_string()
                .contains("graph-capture-deps needs graph-piecewise"),
            "{err}"
        ),
    }
    match sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            cuda_graphs: true,
            graph_build: true,
            graph_capture_deps: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-capture-deps with graph-build still needs piecewise"),
        Err(err) => assert!(
            err.to_string()
                .contains("graph-capture-deps needs graph-piecewise"),
            "{err}"
        ),
    }
    let _ok = sim_replay_cfg(
        &t,
        profile,
        SimCfg {
            cuda_graphs: true,
            graph_piecewise: true,
            graph_capture_deps: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    )
    .expect("piecewise arms capture-deps");
}

#[test]
fn sim_replay_graph_capture_host_needs_graph_piecewise() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    match sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            graph_capture_host: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-capture-host without piecewise must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("graph-capture-host needs graph-piecewise"),
            "{err}"
        ),
    }
    match sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            cuda_graphs: true,
            graph_capture_host: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-capture-host with cuda-graphs still needs piecewise"),
        Err(err) => assert!(
            err.to_string()
                .contains("graph-capture-host needs graph-piecewise"),
            "{err}"
        ),
    }
    match sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            cuda_graphs: true,
            graph_build: true,
            graph_capture_host: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    ) {
        Ok(_) => panic!("graph-capture-host with graph-build still needs piecewise"),
        Err(err) => assert!(
            err.to_string()
                .contains("graph-capture-host needs graph-piecewise"),
            "{err}"
        ),
    }
    let _ok = sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            cuda_graphs: true,
            graph_piecewise: true,
            graph_capture_host: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    )
    .expect("piecewise arms capture-host");
    let _both = sim_replay_cfg(
        &t,
        profile,
        SimCfg {
            cuda_graphs: true,
            graph_piecewise: true,
            graph_capture_deps: true,
            graph_capture_host: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    )
    .expect("capture-deps and capture-host together need piecewise");
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
fn simulated_gpu_store_graph_build_deps_needs_graph_build() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile.clone(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_build_deps: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("graph-build-deps without graph-build must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("graph-build-deps needs graph-build"),
            "{err}"
        ),
    }
    let mut gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_build: true,
            graph_build_deps: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("graph-build arms build-deps");
    assert!(gpu.graph_build_deps());
    let _s = gpu.score().expect("score");
}

#[test]
fn simulated_gpu_store_graph_host_needs_graph_build() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile.clone(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_host: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("graph-host without graph-build must fail"),
        Err(err) => assert!(
            err.to_string().contains("graph-host needs graph-build"),
            "{err}"
        ),
    }
    let mut gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_build: true,
            graph_host: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("graph-build arms graph-host");
    assert!(gpu.graph_host());
    let _s = gpu.score().expect("score");
}

#[test]
fn simulated_gpu_store_graph_memset_needs_graph_mem() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile.clone(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_memset: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("graph-memset without graph-mem must fail"),
        Err(err) => assert!(
            err.to_string().contains("graph-memset needs graph-mem"),
            "{err}"
        ),
    }
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile.clone(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_auto_free: true,
            graph_memset: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("graph-memset with auto-free still needs graph-mem"),
        Err(err) => assert!(
            err.to_string().contains("graph-memset needs graph-mem"),
            "{err}"
        ),
    }
    let mut gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_mem: true,
            graph_memset: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("graph-mem arms graph-memset");
    assert!(gpu.graph_memset());
    let _s = gpu.score().expect("score");
}

#[test]
fn simulated_gpu_store_graph_memcpy_needs_graph_mem() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile.clone(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_memcpy: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("graph-memcpy without graph-mem must fail"),
        Err(err) => assert!(
            err.to_string().contains("graph-memcpy needs graph-mem"),
            "{err}"
        ),
    }
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile.clone(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_auto_free: true,
            graph_memcpy: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("graph-memcpy with auto-free still needs graph-mem"),
        Err(err) => assert!(
            err.to_string().contains("graph-memcpy needs graph-mem"),
            "{err}"
        ),
    }
    let mut gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile.clone(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_mem: true,
            graph_memcpy: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("graph-mem arms graph-memcpy");
    assert!(gpu.graph_memcpy());
    let _s = gpu.score().expect("score");
    let mut both = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_mem: true,
            graph_memset: true,
            graph_memcpy: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("memset and memcpy together need graph-mem");
    assert!(both.graph_memset());
    assert!(both.graph_memcpy());
    let _s = both.score().expect("score");
}

#[test]
fn simulated_gpu_store_graph_leaf_host_cannot_device_launch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile.clone(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_leaf_host: true,
            device_launch: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("graph-leaf-host with device-launch must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("graph-leaf-host cannot device-launch"),
            "{err}"
        ),
    }
    let mut gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_leaf_host: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("leaf-host on store graphs");
    assert!(gpu.graph_leaf_host());
    let _s = gpu.score().expect("score");
}

#[test]
fn simulated_gpu_store_graph_piecewise_launches() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let run = |graph_piecewise: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            1,
            p.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                graph_piecewise,
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
    let (n_piece, h0, _wall_p) = run(true);
    let (n_cap, h1, _wall_c) = run(false);
    assert!(n_piece > 0);
    assert_eq!(n_piece, n_cap);
    assert_eq!(h0, h1);
}

#[test]
fn simulated_gpu_store_graph_capture_deps_needs_graph_piecewise() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile.clone(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_capture_deps: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("graph-capture-deps without piecewise must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("graph-capture-deps needs graph-piecewise"),
            "{err}"
        ),
    }
    let mut gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_piecewise: true,
            graph_capture_deps: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("piecewise arms capture-deps");
    assert!(gpu.graph_capture_deps());
    let _s = gpu.score().expect("score");
}

#[test]
fn simulated_gpu_store_graph_capture_host_needs_graph_piecewise() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile.clone(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_capture_host: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("graph-capture-host without piecewise must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("graph-capture-host needs graph-piecewise"),
            "{err}"
        ),
    }
    let mut gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_piecewise: true,
            graph_capture_host: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("piecewise arms capture-host");
    assert!(gpu.graph_capture_host());
    let _s = gpu.score().expect("score");
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
fn cuda_graphs_graph_memset_slows_graph_mem_scratch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000\nhbm_bps=1000000000000\ngraph_instantiate_ns=1\ngraph_upload_ns=1\ngraph_launch_ns=1\nlaunch_overhead_ns=100000\ncopy_engines=2\n",
    )
    .expect("profile");
    let run = |graph_memset: bool| {
        sim_replay_cfg(
            &t,
            p.clone(),
            SimCfg {
                cuda_graphs: true,
                graph_mem: true,
                graph_memset,
                ..SimCfg::lru(1, 4096, 0)
            },
        )
        .expect("replay")
    };
    let mem = run(false);
    let memset = run(true);
    assert_eq!(mem.hits, memset.hits);
    assert_eq!(mem.misses, memset.misses);
    assert_eq!(mem.graph_launches, memset.graph_launches);
    assert!(
        mem.sim_ns < memset.sim_ns,
        "graph-memset must add memset tax on graph-mem scratch; mem={} memset={}",
        mem.sim_ns,
        memset.sim_ns
    );
}

#[test]
fn cuda_graphs_graph_memcpy_slows_graph_mem_scratch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000\nhbm_bps=1000000000000\npcie_bps=1000000\ngraph_instantiate_ns=1\ngraph_upload_ns=1\ngraph_launch_ns=1\nlaunch_overhead_ns=1\ncopy_engines=2\n",
    )
    .expect("profile");
    let run = |graph_memcpy: bool| {
        sim_replay_cfg(
            &t,
            p.clone(),
            SimCfg {
                cuda_graphs: true,
                graph_mem: true,
                graph_memcpy,
                ..SimCfg::lru(1, 4096, 0)
            },
        )
        .expect("replay")
    };
    let mem = run(false);
    let memcpy = run(true);
    assert_eq!(mem.hits, memcpy.hits);
    assert_eq!(mem.misses, memcpy.misses);
    assert_eq!(mem.graph_launches, memcpy.graph_launches);
    assert!(
        mem.sim_ns < memcpy.sim_ns,
        "graph-memcpy must add PCIe tax on graph-mem scratch; mem={} memcpy={}",
        mem.sim_ns,
        memcpy.sim_ns
    );
}

#[test]
fn cuda_graphs_graph_leaf_host_slows_leaf_gemm() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000\nhbm_bps=1000000000000\ngraph_instantiate_ns=1\ngraph_upload_ns=1\ngraph_launch_ns=1\nlaunch_overhead_ns=1\ncopy_engines=2\nhost_func_ns=100000\n",
    )
    .expect("profile");
    let run = |graph_leaf_host: bool| {
        sim_replay_cfg(
            &t,
            p.clone(),
            SimCfg {
                cuda_graphs: true,
                graph_leaf_host,
                ..SimCfg::lru(1, 4096, 0)
            },
        )
        .expect("replay")
    };
    let graphs = run(false);
    let hosted = run(true);
    assert_eq!(graphs.hits, hosted.hits);
    assert_eq!(graphs.misses, hosted.misses);
    assert_eq!(graphs.graph_launches, hosted.graph_launches);
    assert!(
        graphs.sim_ns < hosted.sim_ns,
        "graph-leaf-host must add host_func_ns before the leaf GEMM; graphs={} host={}",
        graphs.sim_ns,
        hosted.sim_ns
    );
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
    assert_eq!(used_mem, 4096 + GRAPH_SCRATCH_BYTES);
    assert_eq!(used_af, 4096 + GRAPH_SCRATCH_BYTES);
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_cfg(
        inner,
        1,
        p.clone(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_mem: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    let _n = gpu.prefetch(&[k0]).expect("prefetch");
    let _s = gpu.score().expect("drain");
    let _a = gpu.acquire(k0).expect("acq");
    let _score = gpu.score().expect("final");
    assert_eq!(gpu.graph_mem_used(DeviceId(0)).expect("gused"), 0);
    assert_eq!(
        gpu.graph_mem_reserved(DeviceId(0)).expect("gres"),
        GRAPH_SCRATCH_BYTES
    );
    gpu.graph_mem_trim(DeviceId(0)).expect("trim");
    assert_eq!(gpu.hbm_used(DeviceId(0)).expect("trimmed"), 4096);
}

#[test]
fn graph_mem_trim_cfg_returns_reserved_on_score() {
    use crate::sim_replay::GRAPH_SCRATCH_BYTES;
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_cfg(
        inner,
        1,
        p.clone(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_mem: true,
            graph_mem_trim: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    let _n = gpu.prefetch(&[k0]).expect("prefetch");
    let _s = gpu.score().expect("drain");
    let _a = gpu.acquire(k0).expect("acq");
    let score = gpu.score().expect("final");
    assert_eq!(score.hbm_peak, 4096 + GRAPH_SCRATCH_BYTES);
    assert_eq!(gpu.graph_mem_used(DeviceId(0)).expect("gused"), 0);
    assert_eq!(gpu.graph_mem_reserved(DeviceId(0)).expect("gres"), 0);
    assert_eq!(gpu.hbm_used(DeviceId(0)).expect("hbm"), 4096);
    let replay = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            graph_mem: true,
            graph_mem_trim: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    )
    .expect("sim");
    assert_eq!(replay.hbm_peak, 4096 + GRAPH_SCRATCH_BYTES);
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
fn simulated_gpu_store_graph_clone_parent_is_walker_only() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_clone_ns=80000\ngraph_instantiate_ns=1000\ngraph_upload_ns=1000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let mut gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        2,
        p,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_clone_parent: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    assert!(gpu.graph_clone_parent());
    let k0 = ExpertKey::new(0, 0);
    let k1 = ExpertKey::new(0, 1);
    let _n = gpu.prefetch(&[k0, k1]).expect("prefetch");
    let _s = gpu.score().expect("drain");
    let _a0 = gpu.acquire(k0).expect("acq0");
    let _a1 = gpu.acquire(k1).expect("acq1");
    let _score = gpu.score().expect("final");
    assert_eq!(
        gpu.graph_clones(),
        0,
        "store GEMM stays per-leaf; clone-parent must not clone leaves"
    );
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
fn simulated_gpu_store_event_blocking_sync_pays_host_wait_tax() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nhost_sync_blocking_ns=10000\nfp16_flops=1000000\ncopy_engines=2\n",
    )
    .expect("blocking profile");
    let run =
        |event_blocking_sync: bool, timing_events: bool, sync_policy: SynchronizationPolicy| {
            let inner = DirectStore::from_trace(&t);
            let mut gpu = match SimulatedGpuStore::with_cfg(
                inner,
                1,
                p.clone(),
                4096,
                GpuFill::Pinned,
                GpuStoreCfg {
                    event_blocking_sync,
                    timing_events,
                    sync_policy,
                    ..GpuStoreCfg::default()
                },
            ) {
                Ok(gpu) => gpu,
                Err(err) => panic!("gpu: {err}"),
            };
            assert_eq!(gpu.event_blocking_sync(), event_blocking_sync);
            let k0 = ExpertKey::new(0, 0);
            match gpu.acquire(k0) {
                Ok(_) => {}
                Err(err) => panic!("acq: {err}"),
            }
            gpu.release(k0);
            let _s = gpu.score().expect("score");
            (
                gpu.copy_elapsed_ns(),
                gpu.metrics().hits,
                gpu.metrics().misses,
                gpu.clock_ns().expect("clock"),
            )
        };
    let inner = DirectStore::from_trace(&t);
    let identity = match SimulatedGpuStore::new(inner, 1, p.clone(), 4096) {
        Ok(gpu) => gpu,
        Err(err) => panic!("id: {err}"),
    };
    assert!(!identity.event_blocking_sync());
    let (elapsed_t, h0, m0, clock_t) = run(false, true, SynchronizationPolicy::Auto);
    let (elapsed_b, h1, m1, clock_b) = run(true, false, SynchronizationPolicy::Auto);
    let (elapsed_s, h2, m2, clock_s) = run(false, false, SynchronizationPolicy::BlockingSync);
    assert!(elapsed_t > 0, "timing elapsed={elapsed_t}");
    assert!(elapsed_b > 0, "blocking elapsed={elapsed_b}");
    assert_eq!(elapsed_s, 0);
    assert_eq!(h0, h1);
    assert_eq!(m0, m1);
    assert_eq!(h0, h2);
    assert_eq!(m0, m2);
    assert!(
        clock_b > clock_t,
        "BlockingSync copy events must pay host_sync_blocking_ns; timing={clock_t} blocking={clock_b}"
    );
    assert_eq!(
        clock_b.saturating_sub(clock_t),
        20_000,
        "two synchronize_event taxes; timing={clock_t} blocking={clock_b}"
    );
    assert!(
        clock_b > clock_s,
        "event BlockingSync taxes synchronize_event; stream --sync-policy blocking without timing events does not; event={clock_b} stream={clock_s}"
    );
}

#[test]
fn simulated_gpu_store_event_blocking_sync_keeps_hits() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let run = |event_blocking_sync: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = match SimulatedGpuStore::with_cfg(
            inner,
            2,
            p.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                event_blocking_sync,
                ..GpuStoreCfg::default()
            },
        ) {
            Ok(gpu) => gpu,
            Err(err) => panic!("gpu: {err}"),
        };
        for key in t.keys() {
            match gpu.acquire(key) {
                Ok(_) => {}
                Err(err) => panic!("acq: {err}"),
            }
            gpu.release(key);
        }
        let _s = gpu.score().expect("score");
        (gpu.metrics().hits, gpu.metrics().misses)
    };
    let (h0, m0) = run(false);
    let (h1, m1) = run(true);
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
fn simulated_gpu_store_copy_host_lengthens_miss_not_later_hits() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0]), ev(2, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000\nhbm_bps=1000000000000\nhost_func_ns=100000\nlaunch_overhead_ns=1\ncopy_engines=2\n",
    )
    .expect("profile");
    let run = |cfg: GpuStoreCfg| {
        let mut gpu = SimulatedGpuStore::with_cfg(
            DirectStore::from_trace(&t),
            2,
            p.clone(),
            4096,
            GpuFill::Pinned,
            cfg,
        )
        .expect("gpu");
        for k in t.keys() {
            let _p = gpu.acquire(k).expect("acq");
        }
        gpu.score().expect("score")
    };
    let plain = run(GpuStoreCfg::default());
    let mut gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        2,
        p.clone(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            copy_host: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("copy-host");
    assert!(gpu.copy_host());
    for k in t.keys() {
        let _p = gpu.acquire(k).expect("acq");
    }
    let copy = gpu.score().expect("copy score");
    let gemm = run(GpuStoreCfg {
        host_func: true,
        ..GpuStoreCfg::default()
    });
    assert!(
        copy.wall_ns > plain.wall_ns,
        "copy-host={} plain={}",
        copy.wall_ns,
        plain.wall_ns
    );
    assert!(
        gemm.wall_ns > copy.wall_ns,
        "host-func after every GEMM must beat miss-only copy-host; gemm={} copy={}",
        gemm.wall_ns,
        copy.wall_ns
    );
}

#[test]
fn simulated_gpu_store_copy_host_legal_with_pdl_cooperative() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::example_h100_sxm();
    for extra in [
        GpuStoreCfg {
            copy_host: true,
            pdl: true,
            compute_slots: 2,
            ..GpuStoreCfg::default()
        },
        GpuStoreCfg {
            copy_host: true,
            cooperative: true,
            ..GpuStoreCfg::default()
        },
        GpuStoreCfg {
            copy_host: true,
            wait_value: true,
            ..GpuStoreCfg::default()
        },
        GpuStoreCfg {
            copy_host: true,
            memset_fill: true,
            ..GpuStoreCfg::default()
        },
    ] {
        let mut gpu = SimulatedGpuStore::with_cfg(
            DirectStore::from_trace(&t),
            1,
            p.clone(),
            4096,
            GpuFill::Pinned,
            extra,
        )
        .expect("gpu");
        assert!(gpu.copy_host());
        let _n = gpu.prefetch(&[ExpertKey::new(0, 0)]).expect("prefetch");
        let _s = gpu.score().expect("drain");
    }
}

#[test]
fn simulated_gpu_store_copy_host_mapped_is_noop() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000\nhbm_bps=1000000000000\nhost_func_ns=100000\nlaunch_overhead_ns=1\ncopy_engines=2\n",
    )
    .expect("profile");
    let run = |copy_host: bool| {
        let mut gpu = SimulatedGpuStore::with_cfg(
            DirectStore::from_trace(&t),
            1,
            p.clone(),
            4096,
            GpuFill::Mapped,
            GpuStoreCfg {
                copy_host,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        assert_eq!(gpu.copy_host(), copy_host);
        let _n = gpu.prefetch(&[ExpertKey::new(0, 0)]).expect("prefetch");
        gpu.score().expect("drain")
    };
    let plain = run(false);
    let copy = run(true);
    assert_eq!(
        plain.wall_ns, copy.wall_ns,
        "mapped has no device fill; copy-host={} mapped={}",
        copy.wall_ns, plain.wall_ns
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
fn simulated_gpu_store_mempool_trim_returns_cached() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::example_h100_sxm();
    let bytes = 4096u64;
    let k0 = ExpertKey::new(0, 0);
    let mut hold = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        p.clone(),
        bytes,
        GpuFill::Pinned,
        GpuStoreCfg {
            mempool: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("hold");
    let _a = hold.acquire(k0).expect("hold acq");
    hold.evict(k0).expect("hold evict");
    let hold_score = hold.score().expect("hold score");
    assert!(
        hold.default_pool_cached().expect("hold cached") >= bytes,
        "mempool without trim must keep cache"
    );
    let mut trim = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        p,
        bytes,
        GpuFill::Pinned,
        GpuStoreCfg {
            mempool_trim: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("trim");
    let _a = trim.acquire(k0).expect("trim acq");
    trim.evict(k0).expect("trim evict");
    assert!(
        trim.default_pool_cached().expect("pre-score") >= bytes,
        "mempool-trim must hold until score"
    );
    let _clk = trim.clock_ns().expect("itl");
    assert!(
        trim.default_pool_cached().expect("after clock") >= bytes,
        "token ITL must not cudaMemPoolTrimTo"
    );
    let trim_score = trim.score().expect("trim score");
    assert_eq!(
        hold_score.hbm_peak, trim_score.hbm_peak,
        "trim must not change peak HBM"
    );
    assert_eq!(
        trim.default_pool_cached().expect("trimmed"),
        0,
        "cudaMemPoolTrimTo(0) must return cached bytes"
    );
}

#[test]
fn simulated_gpu_store_mempool_trim_refuses_sync_alloc() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            mempool_trim: true,
            sync_alloc: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("mempool-trim + sync-alloc must fail"),
        Err(e) => {
            let s = e.to_string();
            assert!(s.contains("mempool-trim"), "{s}");
        }
    }
}

#[test]
fn sim_replay_mempool_trim_keeps_hits() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg::lru(1, 4096, 0);
    let on = SimCfg {
        mempool_trim: true,
        ..off
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("off");
    let b = sim_replay_cfg(&t, profile.clone(), on).expect("on");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
    assert_eq!(a.hbm_peak, b.hbm_peak);
    let sched_off = schedule_replay(&t, profile.clone(), off, SchedCfg::closed(0)).expect("soff");
    let sched_on = schedule_replay(&t, profile, on, SchedCfg::closed(0)).expect("son");
    assert_eq!(sched_off.replay.hits, sched_on.replay.hits);
    assert_eq!(sched_off.replay.misses, sched_on.replay.misses);
    assert_eq!(sched_off.replay.hbm_peak, sched_on.replay.hbm_peak);
}

#[test]
fn simulated_gpu_store_shareable_mempool_trim_returns_cached() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let bytes = 4096u64;
    let mut gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        bytes,
        GpuFill::Pinned,
        GpuStoreCfg {
            shareable: true,
            mempool_trim: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("share trim");
    let k0 = ExpertKey::new(0, 0);
    let _a = gpu.acquire(k0).expect("acq");
    gpu.evict(k0).expect("evict");
    assert!(
        gpu.default_pool_cached().expect("pre-score") >= bytes,
        "shareable mempool-trim must hold until score"
    );
    let _s = gpu.score().expect("score");
    assert_eq!(
        gpu.default_pool_cached().expect("trimmed"),
        0,
        "trim of device_mempool must return shared cache"
    );
}

#[test]
fn sim_replay_mempool_trim_refuses_sync_alloc() {
    match sim_replay_cfg(
        &Trace {
            events: vec![ev(0, 0, &[0])],
        },
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            mempool_trim: true,
            sync_alloc: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    ) {
        Ok(_) => panic!("mempool-trim + sync-alloc must fail"),
        Err(e) => {
            let s = e.to_string();
            assert!(s.contains("mempool-trim"), "{s}");
        }
    }
}

#[test]
fn simulated_gpu_store_mempool_no_reuse_charges_extra_hbm() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::example_h100_sxm();
    let bytes = 4096u64;
    let k0 = ExpertKey::new(0, 0);
    let mut reuse = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        p.clone(),
        bytes,
        GpuFill::Pinned,
        GpuStoreCfg {
            mempool: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("reuse");
    let _a = reuse.acquire(k0).expect("reuse acq");
    reuse.evict(k0).expect("reuse evict");
    let _clk = reuse.clock_ns().expect("reuse drain");
    let used0 = reuse.hbm_used(DeviceId(0)).expect("u0");
    let _b = reuse.acquire(k0).expect("reuse acq2");
    let _clk = reuse.clock_ns().expect("reuse drain2");
    let used1 = reuse.hbm_used(DeviceId(0)).expect("u1");
    assert_eq!(
        used1, used0,
        "opportunistic reuse must not charge extra HBM"
    );
    let mut skip = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        p,
        bytes,
        GpuFill::Pinned,
        GpuStoreCfg {
            mempool_no_reuse: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("skip");
    let _a = skip.acquire(k0).expect("skip acq");
    skip.evict(k0).expect("skip evict");
    let _clk = skip.clock_ns().expect("skip drain");
    assert!(
        skip.default_pool_cached().expect("cached") >= bytes,
        "no-reuse must still hold cache"
    );
    let used_a = skip.hbm_used(DeviceId(0)).expect("ua");
    let _b = skip.acquire(k0).expect("skip acq2");
    let _clk = skip.clock_ns().expect("skip drain2");
    let used_b = skip.hbm_used(DeviceId(0)).expect("ub");
    assert!(
        used_b >= used_a.saturating_add(bytes),
        "OS alloc must charge extra HBM; a={used_a} b={used_b}"
    );
    assert!(
        skip.default_pool_cached().expect("still") >= bytes,
        "unused cache stays reserved"
    );
}

#[test]
fn simulated_gpu_store_mempool_no_reuse_refuses_sync_alloc() {
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&Trace {
            events: vec![ev(0, 0, &[0])],
        }),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            mempool_no_reuse: true,
            sync_alloc: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("mempool-no-reuse + sync-alloc must fail"),
        Err(e) => {
            let s = e.to_string();
            assert!(s.contains("mempool-no-reuse"), "{s}");
        }
    }
}

#[test]
fn sim_replay_mempool_no_reuse_keeps_hits() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg::lru(1, 4096, 0);
    let held = SimCfg {
        mempool: true,
        ..off
    };
    let skip = SimCfg {
        mempool_no_reuse: true,
        ..off
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("off");
    let b = sim_replay_cfg(&t, profile.clone(), held).expect("held");
    let c = sim_replay_cfg(&t, profile.clone(), skip).expect("skip");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
    assert_eq!(a.hits, c.hits);
    assert_eq!(a.misses, c.misses);
    assert!(
        b.sim_ns < c.sim_ns,
        "no-reuse must pay alloc_overhead; reuse={} skip={}",
        b.sim_ns,
        c.sim_ns
    );
    assert!(
        c.hbm_peak > b.hbm_peak,
        "no-reuse must keep leftover cache reserved; reuse={} skip={}",
        b.hbm_peak,
        c.hbm_peak
    );
    let sched_held = schedule_replay(&t, profile.clone(), held, SchedCfg::closed(0)).expect("sh");
    let sched_skip = schedule_replay(&t, profile, skip, SchedCfg::closed(0)).expect("ss");
    assert_eq!(sched_held.replay.hits, sched_skip.replay.hits);
    assert_eq!(sched_held.replay.misses, sched_skip.replay.misses);
}

#[test]
fn sim_replay_mempool_max_keeps_hits() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg::lru(2, 4096, 0);
    let cap = SimCfg {
        mempool_max: 1 << 30,
        ..off
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("off");
    let b = sim_replay_cfg(&t, profile.clone(), cap).expect("cap");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
    let sched_off = schedule_replay(&t, profile.clone(), off, SchedCfg::closed(0)).expect("so");
    let sched_cap = schedule_replay(&t, profile, cap, SchedCfg::closed(0)).expect("sc");
    assert_eq!(sched_off.replay.hits, sched_cap.replay.hits);
    assert_eq!(sched_off.replay.misses, sched_cap.replay.misses);
}

#[test]
fn sim_replay_mempool_max_ooms_without_reuse() {
    match sim_replay_cfg(
        &Trace {
            events: vec![ev(0, 0, &[0]), ev(1, 0, &[1])],
        },
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            mempool_no_reuse: true,
            mempool_max: 4096,
            ..SimCfg::lru(1, 4096, 0)
        },
    ) {
        Ok(_) => panic!("capped no-reuse thrash must OOM"),
        Err(e) => {
            let s = e.to_string();
            assert!(s.contains("OOM"), "{s}");
        }
    }
}

#[test]
fn sim_replay_mempool_max_needs_cuda_malloc_async() {
    match sim_replay_cfg(
        &Trace {
            events: vec![ev(0, 0, &[0])],
        },
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            mempool_max: 4096,
            sync_alloc: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    ) {
        Ok(_) => panic!("mempool-max + sync-alloc must fail"),
        Err(e) => {
            let s = e.to_string();
            assert!(s.contains("mempool-max"), "{s}");
        }
    }
}

#[test]
fn simulated_gpu_store_mempool_max_ooms_without_reuse() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::example_h100_sxm();
    let bytes = 4096u64;
    let k0 = ExpertKey::new(0, 0);
    let mut skip = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        p,
        bytes,
        GpuFill::Pinned,
        GpuStoreCfg {
            mempool_no_reuse: true,
            mempool_max: 4096,
            ..GpuStoreCfg::default()
        },
    )
    .expect("skip");
    assert_eq!(skip.mempool_max(), 4096);
    let _a = skip.acquire(k0).expect("skip acq");
    skip.evict(k0).expect("skip evict");
    let _clk = skip.clock_ns().expect("skip drain");
    match skip.acquire(k0) {
        Ok(_) => panic!("capped no-reuse leftover must OOM"),
        Err(e) => {
            let s = e.to_string();
            assert!(s.contains("OOM"), "{s}");
        }
    }
}

#[test]
fn simulated_gpu_store_mempool_max_needs_cuda_malloc_async() {
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&Trace {
            events: vec![ev(0, 0, &[0])],
        }),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            mempool_max: 4096,
            sync_alloc: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("mempool-max + sync-alloc must fail"),
        Err(e) => {
            let s = e.to_string();
            assert!(s.contains("mempool-max"), "{s}");
        }
    }
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
fn sim_cfg_host_register_needs_pageable() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let err = match sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            host_register: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    ) {
        Ok(_) => panic!("host-register without pageable must fail"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("host-register needs pageable"),
        "{err}"
    );
}

#[test]
fn simulated_gpu_store_host_register_needs_pageable() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            host_register: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("host-register without pageable must fail"),
        Err(err) => assert!(
            err.to_string().contains("host-register needs pageable"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_host_register_refuses_managed() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Managed,
        GpuStoreCfg {
            pageable: true,
            host_register: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("host-register + managed must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("host-register needs pinned/vmm H2D"),
            "{err}"
        ),
    }
}

#[test]
fn sim_replay_host_register_keeps_hits() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    let pin = SimCfg::lru(1, 4096, 0);
    let reg = SimCfg {
        pageable: true,
        host_register: true,
        ..SimCfg::lru(1, 4096, 0)
    };
    let a = sim_replay_cfg(&t, profile.clone(), pin).expect("pin");
    let b = sim_replay_cfg(&t, profile, reg).expect("reg");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn simulated_gpu_store_host_register_is_faster_than_pageable() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[1])],
    };
    let p = HardwareProfile::example_h100_sxm();
    let bytes = 32u64 << 20;
    let run = |pageable: bool, host_register: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = match SimulatedGpuStore::with_cfg(
            inner,
            2,
            p.clone(),
            bytes,
            GpuFill::Pinned,
            GpuStoreCfg {
                pageable,
                host_register,
                ..GpuStoreCfg::default()
            },
        ) {
            Ok(gpu) => gpu,
            Err(err) => panic!("gpu: {err}"),
        };
        if host_register {
            assert!(gpu.staging_is_pinned());
        } else if pageable {
            assert!(!gpu.staging_is_pinned());
        }
        let _a = gpu.acquire(ExpertKey::new(0, 0)).expect("k0");
        let _b = gpu.acquire(ExpertKey::new(0, 1)).expect("k1");
        gpu.score().expect("score")
    };
    let pin = run(false, false);
    let page = run(true, false);
    let reg = run(true, true);
    assert!(
        page.wall_ns > pin.wall_ns,
        "pageable={} pinned={}",
        page.wall_ns,
        pin.wall_ns
    );
    assert!(
        page.wall_ns > reg.wall_ns,
        "pageable={} host-register={}",
        page.wall_ns,
        reg.wall_ns
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
        host_register: false,
        host_register_mapped: false,
        sync_memops: false,
        memcpy_batch: false,
        memcpy_during: false,
        memcpy_any: false,
        memcpy_attr: false,
        memset_fill: false,
        copy_host: false,
        accessed_by: true,
        no_read_mostly: false,
        no_preferred: false,
        no_mem_prefetch: false,
        wait_value: false,
        stream_attach: false,
        managed_host: false,
        prefetch_host: false,
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
        host_register: false,
        host_register_mapped: false,
        sync_memops: false,
        memcpy_batch: false,
        memcpy_during: false,
        memcpy_any: false,
        memcpy_attr: false,
        memset_fill: false,
        copy_host: false,
        accessed_by: true,
        no_read_mostly: false,
        no_preferred: false,
        no_mem_prefetch: false,
        wait_value: false,
        stream_attach: false,
        managed_host: false,
        prefetch_host: false,
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
        host_register: false,
        host_register_mapped: false,
        sync_memops: false,
        memcpy_batch: false,
        memcpy_during: false,
        memcpy_any: false,
        memcpy_attr: false,
        memset_fill: false,
        copy_host: false,
        accessed_by: true,
        no_read_mostly: false,
        no_preferred: false,
        no_mem_prefetch: false,
        wait_value: false,
        stream_attach: false,
        managed_host: false,
        prefetch_host: false,
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
fn memcpy_batch_prefetch_siblings_share_stream_order_snapshot() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_cfg(
        inner,
        2,
        HardwareProfile::example_h100_sxm(),
        4 << 20,
        GpuFill::Pinned,
        GpuStoreCfg {
            memcpy_batch: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let _n = gpu
        .prefetch(&[ExpertKey::new(0, 0), ExpertKey::new(0, 1)])
        .expect("prefetch");
    let copies: Vec<_> = gpu.memcpy_operations();
    assert_eq!(copies.len(), 2, "{copies:?}");
    let a = &copies[0];
    let b = &copies[1];
    assert_eq!(a.deps, b.deps, "batch siblings share stream-order deps");
    assert!(!a.deps.contains(&b.id));
    assert!(!b.deps.contains(&a.id));
    let _s = gpu.score().expect("drain");
    let copies: Vec<_> = gpu.memcpy_operations();
    assert_eq!(copies[0].start_ns, copies[1].start_ns);
}

#[test]
fn memcpy_batch_demand_acquire_stays_sequential() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_cfg(
        inner,
        2,
        HardwareProfile::example_h100_sxm(),
        4 << 20,
        GpuFill::Pinned,
        GpuStoreCfg {
            memcpy_batch: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let _a = gpu.acquire(ExpertKey::new(0, 0)).expect("a");
    let _b = gpu.acquire(ExpertKey::new(0, 1)).expect("b");
    let _s = gpu.score().expect("drain");
    let copies: Vec<_> = gpu.memcpy_operations();
    assert_eq!(copies.len(), 2, "{copies:?}");
    assert_ne!(
        copies[0].start_ns, copies[1].start_ns,
        "demand H2D stays stream-ordered, not a batch snapshot"
    );
}

#[test]
fn memcpy_batch_apply_misses_siblings_share_stream_order_snapshot() {
    use crate::replay::Touch;
    use crate::sim_replay::{apply_misses, GraphBank, PageHandle, TouchArgs};
    use gpu_sim::{Sim, StreamId};
    use std::collections::BTreeMap;

    let mut sim = Sim::new(HardwareProfile::example_h100_sxm());
    let mut handles: BTreeMap<ExpertKey, PageHandle> = BTreeMap::new();
    let mut graphs = GraphBank::new(false, false, false, crate::sim_replay::LeafMem::None);
    let args = TouchArgs {
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
        host_register: false,
        host_register_mapped: false,
        sync_memops: false,
        memcpy_batch: true,
        memcpy_during: false,
        memcpy_any: false,
        memcpy_attr: false,
        memset_fill: false,
        copy_host: false,
        accessed_by: false,
        no_read_mostly: false,
        no_preferred: false,
        no_mem_prefetch: false,
        wait_value: false,
        stream_attach: false,
        managed_host: false,
        prefetch_host: false,
    };
    let mut next_event = 1u32;
    apply_misses(
        &mut sim,
        &mut handles,
        &mut graphs,
        args,
        &[
            (ExpertKey::new(0, 0), Touch::Miss { evicted: None }),
            (ExpertKey::new(0, 1), Touch::Miss { evicted: None }),
        ],
        &mut next_event,
    )
    .expect("misses");
    let copies: Vec<_> = sim
        .operations()
        .filter(|o| matches!(o.kind, gpu_sim::GpuOp::Memcpy(_)))
        .collect();
    assert_eq!(copies.len(), 2, "{copies:?}");
    assert_eq!(copies[0].deps, copies[1].deps);
    assert!(!copies[0].deps.contains(&copies[1].id));
    assert!(!copies[1].deps.contains(&copies[0].id));
    sim.synchronize().expect("drain");
    let copies: Vec<_> = sim
        .operations()
        .filter(|o| matches!(o.kind, gpu_sim::GpuOp::Memcpy(_)))
        .collect();
    assert_eq!(copies[0].start_ns, copies[1].start_ns);
}

#[test]
fn memcpy_during_apply_misses_waits_copies() {
    use crate::replay::Touch;
    use crate::sim_replay::{apply_misses, GraphBank, PageHandle, TouchArgs};
    use gpu_sim::{Sim, StreamId};
    use std::collections::BTreeMap;

    let run = |during: bool| {
        let mut sim = Sim::new(HardwareProfile::example_h100_sxm());
        let mut handles: BTreeMap<ExpertKey, PageHandle> = BTreeMap::new();
        let mut graphs = GraphBank::new(false, false, false, crate::sim_replay::LeafMem::None);
        let args = TouchArgs {
            d: DeviceId(0),
            s: StreamId(0),
            bytes: 4 << 20,
            slots: 2,
            sync_alloc: false,
            mapped: false,
            managed: false,
            vmm: false,
            vmm_page: 0,
            pageable: false,
            host_register: false,
            host_register_mapped: false,
            sync_memops: false,
            memcpy_batch: true,
            memcpy_during: during,
            memcpy_any: false,
            memcpy_attr: false,
            memset_fill: false,
            copy_host: false,
            accessed_by: false,
            no_read_mostly: false,
            no_preferred: false,
            no_mem_prefetch: false,
            wait_value: false,
            stream_attach: false,
            managed_host: false,
            prefetch_host: false,
        };
        let mut next_event = 1u32;
        apply_misses(
            &mut sim,
            &mut handles,
            &mut graphs,
            args,
            &[
                (ExpertKey::new(0, 0), Touch::Miss { evicted: None }),
                (ExpertKey::new(0, 1), Touch::Miss { evicted: None }),
            ],
            &mut next_event,
        )
        .expect("misses");
        sim.operations()
            .filter(|o| matches!(o.kind, gpu_sim::GpuOp::Memcpy(_)))
            .collect::<Vec<_>>()
    };
    let stream = run(false);
    assert_eq!(stream.len(), 2, "{stream:?}");
    assert!(
        stream.iter().all(|c| !c.done),
        "Stream copies stay in flight; {stream:?}"
    );
    let during = run(true);
    assert_eq!(during.len(), 2, "{during:?}");
    assert!(
        during.iter().all(|c| c.done),
        "DuringApiCall must wait those copies; {during:?}"
    );
}

#[test]
fn memcpy_any_apply_misses_empty_deps() {
    use crate::replay::Touch;
    use crate::sim_replay::{apply_misses, GraphBank, PageHandle, TouchArgs};
    use gpu_sim::{Sim, StreamId};
    use std::collections::BTreeMap;

    let run = |any: bool| {
        let mut sim = Sim::new(HardwareProfile::example_h100_sxm());
        let mut handles: BTreeMap<ExpertKey, PageHandle> = BTreeMap::new();
        let mut graphs = GraphBank::new(false, false, false, crate::sim_replay::LeafMem::None);
        let args = TouchArgs {
            d: DeviceId(0),
            s: StreamId(0),
            bytes: 4 << 20,
            slots: 2,
            sync_alloc: false,
            mapped: false,
            managed: false,
            vmm: false,
            vmm_page: 0,
            pageable: false,
            host_register: false,
            host_register_mapped: false,
            sync_memops: false,
            memcpy_batch: true,
            memcpy_during: false,
            memcpy_any: any,
            memcpy_attr: false,
            memset_fill: false,
            copy_host: false,
            accessed_by: false,
            no_read_mostly: false,
            no_preferred: false,
            no_mem_prefetch: false,
            wait_value: false,
            stream_attach: false,
            managed_host: false,
            prefetch_host: false,
        };
        let mut next_event = 1u32;
        apply_misses(
            &mut sim,
            &mut handles,
            &mut graphs,
            args,
            &[
                (ExpertKey::new(0, 0), Touch::Miss { evicted: None }),
                (ExpertKey::new(0, 1), Touch::Miss { evicted: None }),
            ],
            &mut next_event,
        )
        .expect("misses");
        sim.operations()
            .filter(|o| matches!(o.kind, gpu_sim::GpuOp::Memcpy(_)))
            .collect::<Vec<_>>()
    };
    let stream = run(false);
    assert_eq!(stream.len(), 2, "{stream:?}");
    assert!(
        stream.iter().all(|c| !c.done),
        "Stream copies stay in flight; {stream:?}"
    );
    assert!(
        stream.iter().any(|c| !c.deps.is_empty()),
        "Stream snapshot has predecessors; {stream:?}"
    );
    let any = run(true);
    assert_eq!(any.len(), 2, "{any:?}");
    assert!(
        any.iter().all(|c| !c.done),
        "Any copies stay in flight; {any:?}"
    );
    assert!(
        any.iter().all(|c| c.deps.is_empty()),
        "Any copies have empty deps; {any:?}"
    );
}

#[test]
fn memcpy_attr_apply_misses_waits_copies() {
    use crate::replay::Touch;
    use crate::sim_replay::{apply_misses, GraphBank, PageHandle, TouchArgs};
    use gpu_sim::{Sim, StreamId};
    use std::collections::BTreeMap;

    let run = |attr: bool| {
        let mut sim = Sim::new(HardwareProfile::example_h100_sxm());
        let mut handles: BTreeMap<ExpertKey, PageHandle> = BTreeMap::new();
        let mut graphs = GraphBank::new(false, false, false, crate::sim_replay::LeafMem::None);
        let args = TouchArgs {
            d: DeviceId(0),
            s: StreamId(0),
            bytes: 4 << 20,
            slots: 1,
            sync_alloc: false,
            mapped: false,
            managed: false,
            vmm: false,
            vmm_page: 0,
            pageable: false,
            host_register: false,
            host_register_mapped: false,
            sync_memops: false,
            memcpy_batch: false,
            memcpy_during: false,
            memcpy_any: false,
            memcpy_attr: attr,
            memset_fill: false,
            copy_host: false,
            accessed_by: false,
            no_read_mostly: false,
            no_preferred: false,
            no_mem_prefetch: false,
            wait_value: false,
            stream_attach: false,
            managed_host: false,
            prefetch_host: false,
        };
        let mut next_event = 1u32;
        apply_misses(
            &mut sim,
            &mut handles,
            &mut graphs,
            args,
            &[(ExpertKey::new(0, 0), Touch::Miss { evicted: None })],
            &mut next_event,
        )
        .expect("misses");
        sim.operations()
            .filter(|o| matches!(o.kind, gpu_sim::GpuOp::Memcpy(_)))
            .collect::<Vec<_>>()
    };
    let stream = run(false);
    assert_eq!(stream.len(), 1, "{stream:?}");
    assert!(
        stream.iter().all(|c| !c.done),
        "Stream copies stay in flight; {stream:?}"
    );
    let attr = run(true);
    assert_eq!(attr.len(), 1, "{attr:?}");
    assert!(
        attr.iter().all(|c| c.done),
        "DuringApiCall must wait those copies; {attr:?}"
    );
}

#[test]
fn memcpy_batch_sim_replay_copy_forward() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let row = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            slots: 2,
            prefetch: Prefetch::CopyForward,
            memcpy_batch: true,
            ..SimCfg::lru(2, 4096, 8)
        },
    )
    .expect("batch");
    assert!(row.prefetches > 0, "{}", row.line());
}

#[test]
fn memcpy_batch_rejects_pageable_and_managed() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let refuse = |fill: GpuFill, cfg: GpuStoreCfg| match SimulatedGpuStore::with_cfg(
        inner.clone(),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        fill,
        cfg,
    ) {
        Ok(_) => panic!("memcpy-batch should refuse {fill:?} {cfg:?}"),
        Err(e) => e,
    };
    for err in [
        refuse(
            GpuFill::Pinned,
            GpuStoreCfg {
                memcpy_batch: true,
                pageable: true,
                ..GpuStoreCfg::default()
            },
        ),
        refuse(
            GpuFill::Pinned,
            GpuStoreCfg {
                memcpy_batch: true,
                sync_alloc: true,
                ..GpuStoreCfg::default()
            },
        ),
        refuse(
            GpuFill::Mapped,
            GpuStoreCfg {
                memcpy_batch: true,
                ..GpuStoreCfg::default()
            },
        ),
        refuse(
            GpuFill::Managed,
            GpuStoreCfg {
                memcpy_batch: true,
                ..GpuStoreCfg::default()
            },
        ),
        refuse(
            GpuFill::Pinned,
            GpuStoreCfg {
                memcpy_batch: true,
                sync_memops: true,
                ..GpuStoreCfg::default()
            },
        ),
    ] {
        assert!(
            err.to_string()
                .contains("memcpy-batch needs async pinned/vmm H2D"),
            "{err}"
        );
    }
}

#[test]
fn simulated_gpu_store_memcpy_during_waits_prefetch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let inner = DirectStore::from_trace(&t);
    let prefetch = |during: bool| {
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner.clone(),
            2,
            HardwareProfile::example_h100_sxm(),
            4 << 20,
            GpuFill::Pinned,
            GpuStoreCfg {
                memcpy_batch: true,
                memcpy_during: during,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        let _n = gpu
            .prefetch(&[ExpertKey::new(0, 0), ExpertKey::new(0, 1)])
            .expect("prefetch");
        assert_eq!(gpu.memcpy_during(), during);
        gpu.memcpy_operations()
    };
    let stream = prefetch(false);
    assert_eq!(stream.len(), 2, "{stream:?}");
    assert!(
        stream.iter().all(|c| !c.done),
        "Stream copies stay in flight; {stream:?}"
    );
    let during = prefetch(true);
    assert_eq!(during.len(), 2, "{during:?}");
    assert!(
        during.iter().all(|c| c.done),
        "DuringApiCall must wait those copies; {during:?}"
    );
}

#[test]
fn simulated_gpu_store_memcpy_during_needs_memcpy_batch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            memcpy_during: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("memcpy-during without memcpy-batch must fail"),
        Err(e) => {
            let s = e.to_string();
            assert!(s.contains("memcpy-during"), "{s}");
        }
    }
}

#[test]
fn sim_replay_memcpy_during_keeps_hits() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let batch = SimCfg {
        prefetch: Prefetch::CopyForward,
        memcpy_batch: true,
        ..SimCfg::lru(2, 4096, 8)
    };
    let during = SimCfg {
        memcpy_during: true,
        ..batch
    };
    let a = sim_replay_cfg(&t, p.clone(), batch).expect("batch");
    let b = sim_replay_cfg(&t, p.clone(), during).expect("during");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
    let sched_a = schedule_replay(&t, p.clone(), batch, SchedCfg::closed(0)).expect("sa");
    let sched_b = schedule_replay(&t, p, during, SchedCfg::closed(0)).expect("sb");
    assert_eq!(sched_a.replay.hits, sched_b.replay.hits);
    assert_eq!(sched_a.replay.misses, sched_b.replay.misses);
}

#[test]
fn sim_replay_memcpy_during_needs_memcpy_batch() {
    match sim_replay_cfg(
        &Trace {
            events: vec![ev(0, 0, &[0])],
        },
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            memcpy_during: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    ) {
        Ok(_) => panic!("memcpy-during without memcpy-batch must fail"),
        Err(e) => {
            let s = e.to_string();
            assert!(s.contains("memcpy-during"), "{s}");
        }
    }
}

#[test]
fn simulated_gpu_store_memcpy_any_empty_deps() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let inner = DirectStore::from_trace(&t);
    let prefetch = |any: bool| {
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner.clone(),
            2,
            HardwareProfile::example_h100_sxm(),
            4 << 20,
            GpuFill::Pinned,
            GpuStoreCfg {
                memcpy_batch: true,
                memcpy_any: any,
                memcpy_attr: false,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        let _n = gpu
            .prefetch(&[ExpertKey::new(0, 0), ExpertKey::new(0, 1)])
            .expect("prefetch");
        assert_eq!(gpu.memcpy_any(), any);
        gpu.memcpy_operations()
    };
    let stream = prefetch(false);
    assert_eq!(stream.len(), 2, "{stream:?}");
    assert!(
        stream.iter().all(|c| !c.done),
        "Stream copies stay in flight; {stream:?}"
    );
    assert!(
        stream.iter().any(|c| !c.deps.is_empty()),
        "Stream snapshot has predecessors; {stream:?}"
    );
    let any = prefetch(true);
    assert_eq!(any.len(), 2, "{any:?}");
    assert!(
        any.iter().all(|c| !c.done),
        "Any copies stay in flight; {any:?}"
    );
    assert!(
        any.iter().all(|c| c.deps.is_empty()),
        "Any copies have empty deps; {any:?}"
    );
}

#[test]
fn simulated_gpu_store_memcpy_any_needs_memcpy_batch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            memcpy_any: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("memcpy-any without memcpy-batch must fail"),
        Err(e) => {
            let s = e.to_string();
            assert!(s.contains("memcpy-any"), "{s}");
        }
    }
}

#[test]
fn simulated_gpu_store_memcpy_any_refuses_during() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            memcpy_batch: true,
            memcpy_during: true,
            memcpy_any: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("memcpy-any + memcpy-during must fail"),
        Err(e) => {
            let s = e.to_string();
            assert!(s.contains("memcpy-any"), "{s}");
            assert!(s.contains("memcpy-during"), "{s}");
        }
    }
}

#[test]
fn simulated_gpu_store_memset_fill_is_faster_than_pcie_h2d() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000000000000\nhbm_bps=1000000000000\npcie_bps=1000\nlaunch_overhead_ns=1\ncopy_engines=2\n",
    )
    .expect("profile");
    let run = |memset_fill: bool, fill: GpuFill, extra: GpuStoreCfg| {
        let mut cfg = extra;
        cfg.memset_fill = memset_fill;
        let mut gpu =
            SimulatedGpuStore::with_cfg(DirectStore::from_trace(&t), 1, p.clone(), 4096, fill, cfg)
                .expect("gpu");
        assert_eq!(gpu.memset_fill(), memset_fill);
        let _n = gpu.prefetch(&[ExpertKey::new(0, 0)]).expect("prefetch");
        let score = gpu.score().expect("drain");
        (gpu.metrics().hits, gpu.metrics().misses, score.wall_ns)
    };
    let h2d = run(false, GpuFill::Pinned, GpuStoreCfg::default());
    let memset = run(true, GpuFill::Pinned, GpuStoreCfg::default());
    assert_eq!(h2d.0, memset.0);
    assert_eq!(h2d.1, memset.1);
    assert!(
        memset.2 < h2d.2,
        "memset-fill must beat slow PCIe H2D; memset={} h2d={}",
        memset.2,
        h2d.2
    );
    let vmm = run(true, GpuFill::Vmm, GpuStoreCfg::default());
    assert_eq!(vmm.0, memset.0);
    assert!(vmm.2 < h2d.2, "vmm memset={} h2d={}", vmm.2, h2d.2);
    let _pdl = run(
        true,
        GpuFill::Pinned,
        GpuStoreCfg {
            pdl: true,
            compute_slots: 2,
            ..GpuStoreCfg::default()
        },
    );
    let _coop = run(
        true,
        GpuFill::Pinned,
        GpuStoreCfg {
            cooperative: true,
            ..GpuStoreCfg::default()
        },
    );
    let _sync = run(
        true,
        GpuFill::Pinned,
        GpuStoreCfg {
            sync_alloc: true,
            ..GpuStoreCfg::default()
        },
    );
    let _gb = run(
        true,
        GpuFill::Pinned,
        GpuStoreCfg {
            graph_build: true,
            ..GpuStoreCfg::default()
        },
    );
}

#[test]
fn simulated_gpu_store_memset_fill_cannot_mapped_managed_pageable_batch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::example_h100_sxm();
    let inner = DirectStore::from_trace(&t);
    let refuse = |fill: GpuFill, cfg: GpuStoreCfg, needle: &str| match SimulatedGpuStore::with_cfg(
        inner.clone(),
        1,
        p.clone(),
        4096,
        fill,
        cfg,
    ) {
        Ok(_) => panic!("memset-fill must refuse {needle}"),
        Err(e) => {
            let s = e.to_string();
            assert!(s.contains(needle), "{s}");
        }
    };
    refuse(
        GpuFill::Mapped,
        GpuStoreCfg {
            memset_fill: true,
            ..GpuStoreCfg::default()
        },
        "memset-fill cannot mapped",
    );
    refuse(
        GpuFill::Managed,
        GpuStoreCfg {
            memset_fill: true,
            ..GpuStoreCfg::default()
        },
        "memset-fill cannot managed",
    );
    refuse(
        GpuFill::Pinned,
        GpuStoreCfg {
            memset_fill: true,
            pageable: true,
            ..GpuStoreCfg::default()
        },
        "memset-fill cannot pageable",
    );
    refuse(
        GpuFill::Pinned,
        GpuStoreCfg {
            memset_fill: true,
            memcpy_batch: true,
            ..GpuStoreCfg::default()
        },
        "memset-fill cannot memcpy-batch",
    );
}

#[test]
fn sim_replay_memcpy_any_keeps_hits() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let batch = SimCfg {
        prefetch: Prefetch::CopyForward,
        memcpy_batch: true,
        ..SimCfg::lru(2, 4096, 8)
    };
    let any = SimCfg {
        memcpy_any: true,
        ..batch
    };
    let a = sim_replay_cfg(&t, p.clone(), batch).expect("batch");
    let b = sim_replay_cfg(&t, p.clone(), any).expect("any");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
    let sched_a = schedule_replay(&t, p.clone(), batch, SchedCfg::closed(0)).expect("sa");
    let sched_b = schedule_replay(&t, p, any, SchedCfg::closed(0)).expect("sb");
    assert_eq!(sched_a.replay.hits, sched_b.replay.hits);
    assert_eq!(sched_a.replay.misses, sched_b.replay.misses);
}

#[test]
fn sim_replay_memcpy_any_needs_memcpy_batch() {
    match sim_replay_cfg(
        &Trace {
            events: vec![ev(0, 0, &[0])],
        },
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            memcpy_any: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    ) {
        Ok(_) => panic!("memcpy-any without memcpy-batch must fail"),
        Err(e) => {
            let s = e.to_string();
            assert!(s.contains("memcpy-any"), "{s}");
        }
    }
}

#[test]
fn sim_replay_memcpy_any_refuses_during() {
    match sim_replay_cfg(
        &Trace {
            events: vec![ev(0, 0, &[0])],
        },
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            memcpy_batch: true,
            memcpy_during: true,
            memcpy_any: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    ) {
        Ok(_) => panic!("memcpy-any + memcpy-during must fail"),
        Err(e) => {
            let s = e.to_string();
            assert!(s.contains("memcpy-any"), "{s}");
            assert!(s.contains("memcpy-during"), "{s}");
        }
    }
}

#[test]
fn simulated_gpu_store_memcpy_attr_waits_demand_copy() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let fill = |attr: bool| {
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner.clone(),
            1,
            HardwareProfile::example_h100_sxm(),
            4 << 20,
            GpuFill::Pinned,
            GpuStoreCfg {
                memcpy_attr: attr,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        let _n = gpu.prefetch(&[ExpertKey::new(0, 0)]).expect("prefetch");
        assert_eq!(gpu.memcpy_attr(), attr);
        gpu.memcpy_operations()
    };
    let stream = fill(false);
    assert_eq!(stream.len(), 1, "{stream:?}");
    assert!(
        stream.iter().all(|c| !c.done),
        "Stream copies stay in flight; {stream:?}"
    );
    let attr = fill(true);
    assert_eq!(attr.len(), 1, "{attr:?}");
    assert!(
        attr.iter().all(|c| c.done),
        "DuringApiCall must wait those copies; {attr:?}"
    );
}

#[test]
fn simulated_gpu_store_memcpy_attr_needs_async_pinned() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let p = HardwareProfile::example_h100_sxm();
    let refuse = |fill: GpuFill, cfg: GpuStoreCfg| match SimulatedGpuStore::with_cfg(
        inner.clone(),
        1,
        p.clone(),
        4096,
        fill,
        cfg,
    ) {
        Ok(_) => panic!("memcpy-attr must need async pinned/vmm H2D"),
        Err(e) => {
            let s = e.to_string();
            assert!(s.contains("memcpy-attr needs async pinned/vmm H2D"), "{s}");
        }
    };
    refuse(
        GpuFill::Managed,
        GpuStoreCfg {
            memcpy_attr: true,
            ..GpuStoreCfg::default()
        },
    );
    refuse(
        GpuFill::Mapped,
        GpuStoreCfg {
            memcpy_attr: true,
            ..GpuStoreCfg::default()
        },
    );
    refuse(
        GpuFill::Pinned,
        GpuStoreCfg {
            memcpy_attr: true,
            pageable: true,
            ..GpuStoreCfg::default()
        },
    );
    refuse(
        GpuFill::Pinned,
        GpuStoreCfg {
            memcpy_attr: true,
            sync_alloc: true,
            ..GpuStoreCfg::default()
        },
    );
    match SimulatedGpuStore::with_cfg(
        inner,
        1,
        p,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            memcpy_attr: true,
            memset_fill: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("memset-fill cannot memcpy-attr"),
        Err(e) => {
            let s = e.to_string();
            assert!(s.contains("memset-fill cannot memcpy-attr"), "{s}");
        }
    }
}

#[test]
fn sim_replay_memcpy_attr_keeps_hits() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let base = SimCfg::lru(2, 4096, 8);
    let attr = SimCfg {
        memcpy_attr: true,
        ..base
    };
    let a = sim_replay_cfg(&t, p.clone(), base).expect("base");
    let b = sim_replay_cfg(&t, p.clone(), attr).expect("attr");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
    let sched_a = schedule_replay(&t, p.clone(), base, SchedCfg::closed(0)).expect("sa");
    let sched_b = schedule_replay(&t, p, attr, SchedCfg::closed(0)).expect("sb");
    assert_eq!(sched_a.replay.hits, sched_b.replay.hits);
    assert_eq!(sched_a.replay.misses, sched_b.replay.misses);
}

#[test]
fn sim_replay_memcpy_attr_needs_async_pinned() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::example_h100_sxm();
    let refuse = |cfg: SimCfg| match sim_replay_cfg(&t, p.clone(), cfg) {
        Ok(_) => panic!("memcpy-attr must need async pinned/vmm H2D"),
        Err(e) => {
            let s = e.to_string();
            assert!(s.contains("memcpy-attr needs async pinned/vmm H2D"), "{s}");
        }
    };
    refuse(SimCfg {
        memcpy_attr: true,
        managed: true,
        ..SimCfg::lru(1, 4096, 0)
    });
    refuse(SimCfg {
        memcpy_attr: true,
        mapped: true,
        ..SimCfg::lru(1, 4096, 0)
    });
    refuse(SimCfg {
        memcpy_attr: true,
        pageable: true,
        ..SimCfg::lru(1, 4096, 0)
    });
    refuse(SimCfg {
        memcpy_attr: true,
        sync_alloc: true,
        ..SimCfg::lru(1, 4096, 0)
    });
    match sim_replay_cfg(
        &t,
        p,
        SimCfg {
            memcpy_attr: true,
            memset_fill: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    ) {
        Ok(_) => panic!("memset-fill cannot memcpy-attr"),
        Err(e) => {
            let s = e.to_string();
            assert!(s.contains("memset-fill cannot memcpy-attr"), "{s}");
        }
    }
}

#[test]
fn sim_replay_memset_fill_is_faster_than_pcie_h2d() {
    let t = cycling_trace();
    let p = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000000000000\nhbm_bps=1000000000000\npcie_bps=1000\nlaunch_overhead_ns=1\ncopy_engines=2\n",
    )
    .expect("profile");
    let base = SimCfg::lru(1, 4096, 0);
    let h2d = sim_replay_cfg(&t, p.clone(), base).expect("h2d");
    let memset = sim_replay_cfg(
        &t,
        p.clone(),
        SimCfg {
            memset_fill: true,
            ..base
        },
    )
    .expect("memset");
    assert_eq!(h2d.hits, memset.hits);
    assert_eq!(h2d.misses, memset.misses);
    assert!(
        memset.sim_ns < h2d.sim_ns,
        "memset-fill must beat slow PCIe H2D; memset={} h2d={}",
        memset.sim_ns,
        h2d.sim_ns
    );
    let sched_h = schedule_replay(&t, p.clone(), base, SchedCfg::closed(0)).expect("sh");
    let sched_m = schedule_replay(
        &t,
        p.clone(),
        SimCfg {
            memset_fill: true,
            ..base
        },
        SchedCfg::closed(0),
    )
    .expect("sm");
    assert_eq!(sched_h.replay.hits, sched_m.replay.hits);
    assert_eq!(sched_h.replay.misses, sched_m.replay.misses);
    let _pdl = sim_replay_cfg(
        &t,
        p.clone(),
        SimCfg {
            memset_fill: true,
            pdl: true,
            compute_slots: 2,
            ..base
        },
    )
    .expect("pdl");
    let _coop = sim_replay_cfg(
        &t,
        p.clone(),
        SimCfg {
            memset_fill: true,
            cooperative: true,
            ..base
        },
    )
    .expect("coop");
    let _vmm = sim_replay_cfg(
        &t,
        p.clone(),
        SimCfg {
            memset_fill: true,
            vmm: true,
            ..base
        },
    )
    .expect("vmm");
    let _sync = sim_replay_cfg(
        &t,
        p.clone(),
        SimCfg {
            memset_fill: true,
            sync_alloc: true,
            ..base
        },
    )
    .expect("sync");
    let _gb = sim_replay_cfg(
        &t,
        p,
        SimCfg {
            memset_fill: true,
            graph_build: true,
            cuda_graphs: true,
            ..base
        },
    )
    .expect("graph-build");
}

#[test]
fn sim_replay_memset_fill_cannot_mapped_managed_pageable_batch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::example_h100_sxm();
    let refuse = |cfg: SimCfg, needle: &str| match sim_replay_cfg(&t, p.clone(), cfg) {
        Ok(_) => panic!("memset-fill must refuse {needle}"),
        Err(e) => {
            let s = e.to_string();
            assert!(s.contains(needle), "{s}");
        }
    };
    refuse(
        SimCfg {
            memset_fill: true,
            mapped: true,
            ..SimCfg::lru(1, 4096, 0)
        },
        "memset-fill cannot mapped",
    );
    refuse(
        SimCfg {
            memset_fill: true,
            managed: true,
            ..SimCfg::lru(1, 4096, 0)
        },
        "memset-fill cannot managed",
    );
    refuse(
        SimCfg {
            memset_fill: true,
            pageable: true,
            ..SimCfg::lru(1, 4096, 0)
        },
        "memset-fill cannot pageable",
    );
    refuse(
        SimCfg {
            memset_fill: true,
            memcpy_batch: true,
            ..SimCfg::lru(1, 4096, 0)
        },
        "memset-fill cannot memcpy-batch",
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
            host_register: false,
            host_register_mapped: false,
            sync_memops: false,
            memcpy_batch: false,
            memcpy_during: false,
            memcpy_any: false,
            memcpy_attr: false,
            memset_fill: false,
            copy_host: false,
            accessed_by: false,
            no_read_mostly: false,
            no_preferred: false,
            no_mem_prefetch: false,
            wait_value: false,
            stream_attach: false,
            managed_host: false,
            prefetch_host: false,
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
            &mut handles,
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
fn simulated_gpu_store_cluster_serializes_leftover_prefill() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let run = |cluster: u8| {
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
                cluster,
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
    let overlap = run(0);
    let serial = run(2);
    assert!(
        overlap < serial,
        "cluster of 2 must not overlap leftover prefill with decode; overlap={overlap} serial={serial}"
    );
}

#[test]
fn simulated_gpu_store_cluster_spread_serializes_leftover_prefill() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let run = |spread: bool| {
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
                compute_slots: 4,
                cluster: 2,
                cluster_spread: spread,
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
        "cluster Spread must not overlap leftover prefill with decode; overlap={overlap} serial={serial}"
    );
}

#[test]
fn simulated_gpu_store_func_cluster_spread_serializes_leftover_prefill() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let run = |func_cluster_spread: bool| {
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
                compute_slots: 4,
                cluster: 2,
                func_cluster_spread,
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
        "func cluster Spread must not overlap leftover prefill with decode; overlap={overlap} serial={serial}"
    );
}

#[test]
fn simulated_gpu_store_cluster_load_balance_restores_leftover_prefill_overlap() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let run = |cluster_load_balance: bool| {
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
                compute_slots: 4,
                cluster: 2,
                func_cluster_spread: true,
                cluster_load_balance,
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
    let serial = run(false);
    let restored = run(true);
    assert!(
        restored < serial,
        "launch LoadBalancing must restore leftover prefill overlap; restored={restored} serial={serial}"
    );
}

#[test]
fn simulated_gpu_store_cluster_must_set_matches_cluster_occupancy() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let run = |cluster_must_set: bool| {
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
                compute_slots: 4,
                cluster: 2,
                cluster_must_set,
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
    let cluster = run(false);
    let must = run(true);
    assert_eq!(
        cluster, must,
        "ClusterDimMustBeSet occupancy matches --cluster; cluster={cluster} must={must}"
    );
}

#[test]
fn simulated_gpu_store_cluster_must_set_needs_cluster() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            cluster_must_set: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("cluster-must-set without cluster must fail"),
        Err(err) => assert!(
            err.to_string().contains("cluster-must-set needs cluster"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_required_cluster_matches_cluster_occupancy() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let run = |required_cluster: u8| {
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
                compute_slots: 4,
                cluster: 2,
                required_cluster,
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
    let cluster = run(0);
    let required = run(2);
    assert_eq!(
        cluster, required,
        "RequiredClusterWidth occupancy matches --cluster; cluster={cluster} required={required}"
    );
}

#[test]
fn simulated_gpu_store_required_cluster_needs_cluster() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            required_cluster: 2,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("required-cluster without cluster must fail"),
        Err(err) => assert!(
            err.to_string().contains("required-cluster needs cluster"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_required_cluster_must_match() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            cluster: 2,
            required_cluster: 4,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("required-cluster mismatch must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("required-cluster must match cluster"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_preferred_cluster_serializes_leftover_prefill() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let run = |preferred: u8| {
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
                compute_slots: 4,
                cluster: 2,
                preferred_cluster: preferred,
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
    let overlap = run(0);
    let serial = run(4);
    assert!(
        overlap < serial,
        "preferred cluster of 4 must not overlap leftover prefill with decode; overlap={overlap} serial={serial}"
    );
}

#[test]
fn simulated_gpu_store_preferred_cluster_needs_cluster() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            preferred_cluster: 4,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("preferred without cluster must fail"),
        Err(err) => assert!(
            err.to_string().contains("preferred-cluster needs cluster"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_max_shared_serializes_leftover_prefill() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let run = |max_shared: bool| {
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
                max_shared,
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
        "MaxShared carveout must not overlap leftover prefill with decode; overlap={overlap} serial={serial}"
    );
}

#[test]
fn simulated_gpu_store_func_max_shared_serializes_leftover_prefill() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let run = |func_max_shared: bool| {
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
                func_max_shared,
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
        "func MaxShared carveout must not overlap leftover prefill with decode; overlap={overlap} serial={serial}"
    );
}

#[test]
fn simulated_gpu_store_max_l1_restores_leftover_prefill_overlap() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let run = |max_l1: bool| {
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
                func_max_shared: true,
                max_l1,
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
    let serial = run(false);
    let restored = run(true);
    assert!(
        restored < serial,
        "launch MaxL1 must restore leftover prefill overlap; restored={restored} serial={serial}"
    );
}

#[test]
fn simulated_gpu_store_nvlink_util_serializes_leftover_prefill() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 2])],
    };
    let nv = HardwareProfile::parse("gpus=2\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("nvlink slow gemm");
    assert!(nv.has_nvlink());
    let run = |profile: HardwareProfile, nvlink: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = match SimulatedGpuStore::with_cfg(
            inner,
            2,
            profile,
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                decode_priority: true,
                stream_priority: true,
                compute_slots: 2,
                nvlink_util_centric: nvlink,
                ..GpuStoreCfg::default()
            },
        ) {
            Ok(gpu) => gpu,
            Err(err) => panic!("gpu: {err}"),
        };
        let pre = ExpertKey::new(0, 0);
        let dec = ExpertKey::new(0, 2);
        gpu.bind_decode_compute(false);
        let _warm_pre = match gpu.acquire(pre) {
            Ok(v) => v,
            Err(err) => panic!("warm pre: {err}"),
        };
        gpu.release(pre);
        let _warm_dec = match gpu.acquire(dec) {
            Ok(v) => v,
            Err(err) => panic!("warm dec: {err}"),
        };
        gpu.release(dec);
        let t0 = gpu.clock_ns().expect("drain h2d");
        gpu.bind_decode_compute(false);
        let _prefill = match gpu.acquire(pre) {
            Ok(v) => v,
            Err(err) => panic!("prefill: {err}"),
        };
        gpu.release(pre);
        gpu.bind_decode_compute(true);
        let _decode = match gpu.acquire(dec) {
            Ok(v) => v,
            Err(err) => panic!("decode: {err}"),
        };
        gpu.release(dec);
        gpu.score().expect("score").wall_ns.saturating_sub(t0)
    };
    let overlap = run(nv.clone(), false);
    let serial = run(nv, true);
    assert!(
        overlap < serial,
        "NVLink-util-centric must not overlap leftover prefill; overlap={overlap} serial={serial}"
    );
    let h100 =
        HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n").expect("no nvlink");
    assert!(!h100.has_nvlink());
    let off = run(h100.clone(), false);
    let on = run(h100, true);
    assert_eq!(
        off, on,
        "without NVLink the hint must not change occupancy; off={off} on={on}"
    );
}

#[test]
fn gemm_flags_priority_is_launch_attr() {
    use crate::sim_replay::GemmFlags;
    let id = AllocId(1);
    let none = GemmFlags::default().kernel_attrs(id);
    assert_eq!(none.priority, None);
    let some = GemmFlags {
        priority: Some(-5),
        ..GemmFlags::default()
    }
    .kernel_attrs(id);
    assert_eq!(some.priority, Some(-5));
}

#[test]
fn gemm_flags_launch_completion_is_launch_attr() {
    use crate::sim_replay::GemmFlags;
    use gpu_sim::LaunchCompletionEvent;
    let id = AllocId(1);
    let none = GemmFlags::default().kernel_attrs(id);
    assert_eq!(none.launch_completion, None);
    let ev = LaunchCompletionEvent {
        event: EventId(7),
        external: false,
    };
    let some = GemmFlags {
        launch_completion: Some(ev),
        ..GemmFlags::default()
    }
    .kernel_attrs(id);
    assert_eq!(some.launch_completion, Some(ev));
}

#[test]
fn gemm_flags_programmatic_event_is_launch_attr() {
    use crate::sim_replay::GemmFlags;
    use gpu_sim::{ProgrammaticEvent, ProgrammaticLaunch};
    let id = AllocId(1);
    let none = GemmFlags::default().kernel_attrs(id);
    assert_eq!(none.programmatic_event, None);
    assert!(!none.pdl.trigger);
    let ev = ProgrammaticEvent {
        event: EventId(8),
        external: false,
    };
    let some = GemmFlags {
        programmatic_event: Some(ev),
        ..GemmFlags::default()
    }
    .kernel_attrs(id);
    assert_eq!(some.programmatic_event, Some(ev));
    assert_eq!(
        some.pdl,
        ProgrammaticLaunch {
            wait: false,
            trigger: true,
        }
    );
}

#[test]
fn gemm_flags_persist_ratio_is_launch_attr() {
    use crate::sim_replay::GemmFlags;
    let id = AllocId(1);
    let none = GemmFlags::default().kernel_attrs(id);
    assert!(none.access_policy.is_none());
    let full = GemmFlags {
        l2_persist: true,
        ..GemmFlags::default()
    }
    .kernel_attrs(id);
    assert_eq!(full.access_policy.expect("window").hit_ratio_permille, 1000);
    let half = GemmFlags {
        l2_persist: true,
        l2_ratio: 500,
        ..GemmFlags::default()
    }
    .kernel_attrs(id);
    assert_eq!(half.access_policy.expect("window").hit_ratio_permille, 500);
}

#[test]
fn gemm_flags_l2_streaming_is_streaming_hit() {
    use crate::sim_replay::GemmFlags;
    let id = AllocId(1);
    let persist = GemmFlags {
        l2_persist: true,
        ..GemmFlags::default()
    }
    .kernel_attrs(id);
    assert_eq!(
        persist.access_policy.expect("window").hit,
        AccessProperty::Persisting
    );
    let streaming = GemmFlags {
        l2_persist: true,
        l2_streaming: true,
        ..GemmFlags::default()
    }
    .kernel_attrs(id);
    let window = streaming.access_policy.expect("window");
    assert_eq!(window.hit, AccessProperty::Streaming);
    assert_eq!(window.miss, AccessProperty::Streaming);
    assert_eq!(window.hit_ratio_permille, 1000);
    let none = GemmFlags {
        l2_streaming: true,
        ..GemmFlags::default()
    }
    .kernel_attrs(id);
    assert!(none.access_policy.is_none());
}

#[test]
fn gemm_flags_mem_sync_launch_is_launch_attr() {
    use crate::sim_replay::GemmFlags;
    let id = AllocId(1);
    let none = GemmFlags::default().kernel_attrs(id);
    assert_eq!(none.mem_sync_domain, None);
    let some = GemmFlags {
        mem_sync_launch: true,
        ..GemmFlags::default()
    }
    .kernel_attrs(id);
    assert_eq!(some.mem_sync_domain, Some(MemSyncDomain::Remote));
}

#[test]
fn gemm_flags_mem_sync_launch_map_is_launch_attr() {
    use crate::sim_replay::GemmFlags;
    let id = AllocId(1);
    let none = GemmFlags::default().kernel_attrs(id);
    assert_eq!(none.mem_sync_map, None);
    let some = GemmFlags {
        mem_sync_launch_map: true,
        ..GemmFlags::default()
    }
    .kernel_attrs(id);
    assert_eq!(
        some.mem_sync_map,
        Some(MemSyncDomainMap {
            default: 0,
            remote: 0,
        })
    );
}

#[test]
fn simulated_gpu_store_device_updatable_skips_reupload() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[1])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\ngraph_instantiate_ns=1000\ngraph_set_params_ns=100\ngraph_upload_ns=50000\ngraph_launch_ns=1000\nlaunch_overhead_ns=1000\ncopy_engines=2\n",
    )
    .expect("profile");
    let run = |device_updatable: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = match SimulatedGpuStore::with_cfg(
            inner,
            1,
            p.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                graph_set_params: true,
                device_launch: true,
                device_updatable,
                ..GpuStoreCfg::default()
            },
        ) {
            Ok(gpu) => gpu,
            Err(err) => panic!("gpu: {err}"),
        };
        let k0 = ExpertKey::new(0, 0);
        let k1 = ExpertKey::new(0, 1);
        let _n = match gpu.prefetch(&[k0]) {
            Ok(v) => v,
            Err(err) => panic!("p0: {err}"),
        };
        let _s = gpu.score().expect("d0");
        let _a = match gpu.acquire(k0) {
            Ok(v) => v,
            Err(err) => panic!("a0: {err}"),
        };
        let _s = gpu.score().expect("drain0");
        gpu.evict(k0).expect("evict");
        let _s = gpu.score().expect("free");
        let _n = match gpu.prefetch(&[k1]) {
            Ok(v) => v,
            Err(err) => panic!("p1: {err}"),
        };
        let _s = gpu.score().expect("d1");
        let _b = match gpu.acquire(k1) {
            Ok(v) => v,
            Err(err) => panic!("a1: {err}"),
        };
        gpu.score().expect("final").wall_ns
    };
    let skip = run(true);
    let reupload = run(false);
    assert!(
        skip < reupload,
        "device-updatable set-params must skip re-upload; skip={skip} reupload={reupload}"
    );
}

#[test]
fn simulated_gpu_store_device_launch_refuses_graph_update() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            device_launch: true,
            graph_update: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("device-launch + graph-update must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("choose one of graph-update, device-launch"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_non_portable_cluster_allows_oversize() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::parse("gpus=1\nmax_blocks_per_cluster=16\n")
        .expect("open cluster profile");
    let key = ExpertKey::new(0, 0);
    let mut blocked = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile.clone(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            cluster: 16,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    match blocked.acquire(key) {
        Ok(_) => panic!("cluster 16 without non-portable must fail"),
        Err(err) => assert!(err.to_string().contains("non-portable cluster"), "{err}"),
    }
    let mut allowed = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            cluster: 16,
            non_portable_cluster: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let _lease = allowed.acquire(key).expect("allowed");
    allowed.release(key);
}

#[test]
fn simulated_gpu_store_portable_cluster_overrides_func_attribute() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::parse("gpus=1\nmax_blocks_per_cluster=16\n")
        .expect("open cluster profile");
    let key = ExpertKey::new(0, 0);
    let mut allowed = match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile.clone(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            cluster: 16,
            portable_cluster: PortableClusterMode::AllowNonPortable,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(gpu) => gpu,
        Err(err) => panic!("gpu: {err}"),
    };
    match allowed.acquire(key) {
        Ok(_) => {}
        Err(err) => panic!("AllowNonPortable must launch oversize without func attr: {err}"),
    }
    allowed.release(key);
    let mut refused = match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            cluster: 16,
            non_portable_cluster: true,
            portable_cluster: PortableClusterMode::RequirePortable,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(gpu) => gpu,
        Err(err) => panic!("gpu: {err}"),
    };
    match refused.acquire(key) {
        Ok(_) => panic!("RequirePortable must refuse oversize even with func attr"),
        Err(err) => assert!(err.to_string().contains("non-portable cluster"), "{err}"),
    }
}

#[test]
fn simulated_gpu_store_optin_shared_allows_oversize() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::parse("gpus=1\nmax_shared_mem_per_block_optin=232448\n")
        .expect("open shared profile");
    let key = ExpertKey::new(0, 0);
    let mut blocked = match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile.clone(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            dynamic_shared: 65_536,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(gpu) => gpu,
        Err(err) => panic!("gpu: {err}"),
    };
    match blocked.acquire(key) {
        Ok(_) => panic!("oversize without permission must fail"),
        Err(err) => assert!(err.to_string().contains("non-portable shared"), "{err}"),
    }
    let mut allowed = match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            dynamic_shared: 65_536,
            optin_shared: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(gpu) => gpu,
        Err(err) => panic!("gpu: {err}"),
    };
    match allowed.acquire(key) {
        Ok(_) => {}
        Err(err) => panic!("optin must launch oversize: {err}"),
    }
    allowed.release(key);
}

#[test]
fn simulated_gpu_store_portable_shared_overrides_func_attribute() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::parse("gpus=1\nmax_shared_mem_per_block_optin=232448\n")
        .expect("open shared profile");
    let key = ExpertKey::new(0, 0);
    let mut allowed = match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile.clone(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            dynamic_shared: 65_536,
            portable_shared: PortableSharedMode::AllowNonPortable,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(gpu) => gpu,
        Err(err) => panic!("gpu: {err}"),
    };
    match allowed.acquire(key) {
        Ok(_) => {}
        Err(err) => panic!("AllowNonPortable must launch oversize without func attr: {err}"),
    }
    allowed.release(key);
    let mut refused = match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            dynamic_shared: 65_536,
            optin_shared: true,
            portable_shared: PortableSharedMode::RequirePortable,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(gpu) => gpu,
        Err(err) => panic!("gpu: {err}"),
    };
    match refused.acquire(key) {
        Ok(_) => panic!("RequirePortable must refuse oversize even with func attr"),
        Err(err) => assert!(err.to_string().contains("non-portable shared"), "{err}"),
    }
}

#[test]
fn simulated_gpu_store_sync_policy_taxes_decode_stream() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile =
        HardwareProfile::parse("gpus=1\nhost_sync_blocking_ns=10000\nfp16_flops=1000000\n")
            .expect("host-sync profile");
    let run = |policy: SynchronizationPolicy| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            1,
            profile.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                decode_priority: true,
                stream_priority: true,
                sync_policy: policy,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        let key = ExpertKey::new(0, 0);
        gpu.bind_decode_compute(true);
        let _warm = gpu.acquire(key).expect("warm");
        gpu.release(key);
        let t0 = gpu.clock_ns().expect("drain");
        gpu.bind_decode_compute(true);
        let _hit = gpu.acquire(key).expect("hit");
        gpu.release(key);
        gpu.token_clock_ns().expect("token").saturating_sub(t0)
    };
    let auto = run(SynchronizationPolicy::Auto);
    let block = run(SynchronizationPolicy::BlockingSync);
    assert_eq!(
        block,
        auto.saturating_add(10_000),
        "blocking stream wait must add host-sync tax; auto={auto} block={block}"
    );
}

#[test]
fn simulated_gpu_store_device_sync_policy_taxes_decode_stream() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile =
        HardwareProfile::parse("gpus=1\nhost_sync_blocking_ns=10000\nfp16_flops=1000000\n")
            .expect("host-sync profile");
    let run = |policy: SynchronizationPolicy| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            1,
            profile.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                decode_priority: true,
                stream_priority: true,
                device_sync_policy: policy,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        assert_eq!(gpu.device_sync_policy(), policy);
        let key = ExpertKey::new(0, 0);
        gpu.bind_decode_compute(true);
        let _warm = gpu.acquire(key).expect("warm");
        gpu.release(key);
        let t0 = gpu.clock_ns().expect("drain");
        gpu.bind_decode_compute(true);
        let _hit = gpu.acquire(key).expect("hit");
        gpu.release(key);
        gpu.token_clock_ns().expect("token").saturating_sub(t0)
    };
    let auto = run(SynchronizationPolicy::Auto);
    let block = run(SynchronizationPolicy::BlockingSync);
    assert_eq!(
        block,
        auto.saturating_add(10_000),
        "device BlockingSync must tax Auto streams; auto={auto} block={block}"
    );
}

#[test]
fn simulated_gpu_store_device_sync_policy_stream_wins() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::parse(
        "gpus=1\nhost_sync_spin_ns=1000\nhost_sync_blocking_ns=10000\nfp16_flops=1000000\n",
    )
    .expect("host-sync profile");
    let run = |device: SynchronizationPolicy, stream: SynchronizationPolicy| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            1,
            profile.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                decode_priority: true,
                stream_priority: true,
                device_sync_policy: device,
                sync_policy: stream,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        assert_eq!(gpu.device_sync_policy(), device);
        let key = ExpertKey::new(0, 0);
        gpu.bind_decode_compute(true);
        let _warm = gpu.acquire(key).expect("warm");
        gpu.release(key);
        let t0 = gpu.clock_ns().expect("drain");
        gpu.bind_decode_compute(true);
        let _hit = gpu.acquire(key).expect("hit");
        gpu.release(key);
        gpu.token_clock_ns().expect("token").saturating_sub(t0)
    };
    let device_block = run(
        SynchronizationPolicy::BlockingSync,
        SynchronizationPolicy::Auto,
    );
    let stream_spin = run(
        SynchronizationPolicy::BlockingSync,
        SynchronizationPolicy::Spin,
    );
    assert_eq!(
        stream_spin,
        device_block.saturating_sub(9_000),
        "explicit stream Spin must beat device BlockingSync; device={device_block} stream={stream_spin}"
    );
}

#[test]
fn simulated_gpu_store_device_sync_policy_ors_sync_memops() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let inner = DirectStore::from_trace(&t);
    let mut gpu = match SimulatedGpuStore::with_cfg(
        inner,
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            device_sync_memops: true,
            device_sync_policy: SynchronizationPolicy::BlockingSync,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(gpu) => gpu,
        Err(err) => panic!("gpu: {err}"),
    };
    assert!(gpu.device_sync_memops());
    assert_eq!(
        gpu.device_sync_policy(),
        SynchronizationPolicy::BlockingSync
    );
    let identity = SimulatedGpuStore::new(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
    )
    .expect("id");
    assert!(!identity.device_sync_memops());
    assert_eq!(identity.device_sync_policy(), SynchronizationPolicy::Auto);
    let _s = gpu.score().expect("drain");
}

#[test]
fn simulated_gpu_store_shared_mem_eight_lengthens_gemm() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile =
        HardwareProfile::parse("gpus=1\nshared_mem_eight_byte_permille=500\nfp16_flops=1000000\n")
            .expect("shared-mem profile");
    let run = |mode: SharedMemoryMode| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = match SimulatedGpuStore::with_cfg(
            inner,
            1,
            profile.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                shared_mem: mode,
                ..GpuStoreCfg::default()
            },
        ) {
            Ok(gpu) => gpu,
            Err(err) => panic!("gpu: {err}"),
        };
        let key = ExpertKey::new(0, 0);
        match gpu.acquire(key) {
            Ok(_) => {}
            Err(err) => panic!("warm: {err}"),
        }
        gpu.release(key);
        let t0 = gpu.clock_ns().expect("drain");
        match gpu.acquire(key) {
            Ok(_) => {}
            Err(err) => panic!("hit: {err}"),
        }
        gpu.release(key);
        gpu.clock_ns().expect("done").saturating_sub(t0)
    };
    let def = run(SharedMemoryMode::Default);
    let eight = run(SharedMemoryMode::EightByte);
    assert!(
        eight > def,
        "EightByte at permille 500 must lengthen hit GEMM; default={def} eight={eight}"
    );
}

#[test]
fn simulated_gpu_store_func_shared_mem_lengthens_gemm() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile =
        HardwareProfile::parse("gpus=1\nshared_mem_eight_byte_permille=500\nfp16_flops=1000000\n")
            .expect("shared-mem profile");
    let run = |func: SharedMemoryMode, launch: SharedMemoryMode| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = match SimulatedGpuStore::with_cfg(
            inner,
            1,
            profile.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                func_shared_mem: func,
                shared_mem: launch,
                ..GpuStoreCfg::default()
            },
        ) {
            Ok(gpu) => gpu,
            Err(err) => panic!("gpu: {err}"),
        };
        assert_eq!(gpu.func_shared_mem(), func);
        let key = ExpertKey::new(0, 0);
        match gpu.acquire(key) {
            Ok(_) => {}
            Err(err) => panic!("warm: {err}"),
        }
        gpu.release(key);
        let t0 = gpu.clock_ns().expect("drain");
        match gpu.acquire(key) {
            Ok(_) => {}
            Err(err) => panic!("hit: {err}"),
        }
        gpu.release(key);
        gpu.clock_ns().expect("done").saturating_sub(t0)
    };
    let inner = DirectStore::from_trace(&t);
    let mut identity = SimulatedGpuStore::new(inner, 1, profile.clone(), 4096).expect("id");
    let k0 = ExpertKey::new(0, 0);
    let _a = identity.acquire(k0).expect("id acq");
    assert_eq!(identity.func_shared_mem(), SharedMemoryMode::Default);
    let _s = identity.score().expect("id score");
    let def = run(SharedMemoryMode::Default, SharedMemoryMode::Default);
    let eight = run(SharedMemoryMode::EightByte, SharedMemoryMode::Default);
    let overridden = run(SharedMemoryMode::EightByte, SharedMemoryMode::FourByte);
    assert!(
        eight > def,
        "func EightByte at permille 500 must lengthen hit GEMM; default={def} eight={eight}"
    );
    assert!(
        overridden < eight,
        "launch FourByte must override func EightByte; override={overridden} eight={eight}"
    );
}

#[test]
fn simulated_gpu_store_device_shared_mem_lengthens_gemm() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile =
        HardwareProfile::parse("gpus=1\nshared_mem_eight_byte_permille=500\nfp16_flops=1000000\n")
            .expect("shared-mem profile");
    let run = |device: SharedMemoryMode, func: SharedMemoryMode, launch: SharedMemoryMode| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = match SimulatedGpuStore::with_cfg(
            inner,
            1,
            profile.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                device_shared_mem: device,
                func_shared_mem: func,
                shared_mem: launch,
                ..GpuStoreCfg::default()
            },
        ) {
            Ok(gpu) => gpu,
            Err(err) => panic!("gpu: {err}"),
        };
        assert_eq!(gpu.device_shared_mem(), device);
        let key = ExpertKey::new(0, 0);
        match gpu.acquire(key) {
            Ok(_) => {}
            Err(err) => panic!("warm: {err}"),
        }
        gpu.release(key);
        let t0 = gpu.clock_ns().expect("drain");
        match gpu.acquire(key) {
            Ok(_) => {}
            Err(err) => panic!("hit: {err}"),
        }
        gpu.release(key);
        gpu.clock_ns().expect("done").saturating_sub(t0)
    };
    let inner = DirectStore::from_trace(&t);
    let mut identity = SimulatedGpuStore::new(inner, 1, profile.clone(), 4096).expect("id");
    let k0 = ExpertKey::new(0, 0);
    let _a = identity.acquire(k0).expect("id acq");
    assert_eq!(identity.device_shared_mem(), SharedMemoryMode::Default);
    let _s = identity.score().expect("id score");
    let def = run(
        SharedMemoryMode::Default,
        SharedMemoryMode::Default,
        SharedMemoryMode::Default,
    );
    let eight = run(
        SharedMemoryMode::EightByte,
        SharedMemoryMode::Default,
        SharedMemoryMode::Default,
    );
    let overridden = run(
        SharedMemoryMode::EightByte,
        SharedMemoryMode::Default,
        SharedMemoryMode::FourByte,
    );
    let func_wins = run(
        SharedMemoryMode::FourByte,
        SharedMemoryMode::EightByte,
        SharedMemoryMode::Default,
    );
    assert!(
        eight > def,
        "device EightByte at permille 500 must lengthen hit GEMM; default={def} eight={eight}"
    );
    assert!(
        overridden < eight,
        "launch FourByte must override device EightByte; override={overridden} eight={eight}"
    );
    assert!(
        func_wins > def,
        "func EightByte must override device FourByte; default={def} func={func_wins}"
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
fn sim_replay_cluster_serializes_seq_streams() {
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
    let clustered = SimCfg {
        cluster: 2,
        ..hyperq
    };
    let overlap = sim_replay_cfg(&t, profile.clone(), hyperq).expect("hyperq");
    let serial = sim_replay_cfg(&t, profile, clustered).expect("cluster");
    assert_eq!(overlap.hits, serial.hits);
    assert_eq!(overlap.misses, serial.misses);
    assert!(
        overlap.sim_ns < serial.sim_ns,
        "cluster of 2 must not Hyper-Q overlap; overlap={} serial={}",
        overlap.sim_ns,
        serial.sim_ns
    );
}

#[test]
fn sim_replay_cluster_spread_serializes_seq_streams() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0]), ev_seq(1, 0, 0, &[1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let packed = SimCfg {
        seq_streams: true,
        compute_slots: 4,
        cluster: 2,
        ..SimCfg::lru(2, 4096, 0)
    };
    let spread = SimCfg {
        cluster_spread: true,
        ..packed
    };
    let overlap = sim_replay_cfg(&t, profile.clone(), packed).expect("packed");
    let serial = sim_replay_cfg(&t, profile, spread).expect("spread");
    assert_eq!(overlap.hits, serial.hits);
    assert_eq!(overlap.misses, serial.misses);
    assert!(
        overlap.sim_ns < serial.sim_ns,
        "cluster Spread must not Hyper-Q overlap leftover seq GEMMs; overlap={} serial={}",
        overlap.sim_ns,
        serial.sim_ns
    );
}

#[test]
fn sim_replay_func_cluster_spread_serializes_seq_streams() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0]), ev_seq(1, 0, 0, &[1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let packed = SimCfg {
        seq_streams: true,
        compute_slots: 4,
        cluster: 2,
        ..SimCfg::lru(2, 4096, 0)
    };
    let spread = SimCfg {
        func_cluster_spread: true,
        ..packed
    };
    let overlap = sim_replay_cfg(&t, profile.clone(), packed).expect("packed");
    let serial = sim_replay_cfg(&t, profile, spread).expect("func-cluster-spread");
    assert_eq!(overlap.hits, serial.hits);
    assert_eq!(overlap.misses, serial.misses);
    assert!(
        overlap.sim_ns < serial.sim_ns,
        "func cluster Spread must not Hyper-Q overlap leftover seq GEMMs; overlap={} serial={}",
        overlap.sim_ns,
        serial.sim_ns
    );
}

#[test]
fn sim_replay_cluster_load_balance_restores_seq_stream_overlap() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0]), ev_seq(1, 0, 0, &[1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let packed = SimCfg {
        seq_streams: true,
        compute_slots: 4,
        cluster: 2,
        ..SimCfg::lru(2, 4096, 0)
    };
    let serial = SimCfg {
        func_cluster_spread: true,
        ..packed
    };
    let restored = SimCfg {
        cluster_load_balance: true,
        ..serial
    };
    let overlap = sim_replay_cfg(&t, profile.clone(), packed).expect("packed");
    let serial = sim_replay_cfg(&t, profile.clone(), serial).expect("func-cluster-spread");
    let restored = sim_replay_cfg(&t, profile, restored).expect("cluster-load-balance");
    assert_eq!(overlap.hits, restored.hits);
    assert_eq!(overlap.misses, restored.misses);
    assert!(
        restored.sim_ns < serial.sim_ns,
        "launch LoadBalancing must restore Hyper-Q overlap; restored={} serial={}",
        restored.sim_ns,
        serial.sim_ns
    );
}

#[test]
fn sim_replay_func_cluster_spread_serializes_seq_streams_graphs() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0]), ev_seq(1, 0, 0, &[1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let packed = SimCfg {
        seq_streams: true,
        compute_slots: 4,
        cluster: 2,
        cuda_graphs: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let spread = SimCfg {
        func_cluster_spread: true,
        ..packed
    };
    let overlap = sim_replay_cfg(&t, profile.clone(), packed).expect("packed graphs");
    let serial = sim_replay_cfg(&t, profile, spread).expect("func-cluster-spread graphs");
    assert_eq!(overlap.hits, serial.hits);
    assert_eq!(overlap.misses, serial.misses);
    assert!(
        overlap.sim_ns < serial.sim_ns,
        "func cluster Spread graphs must inherit occupancy; overlap={} serial={}",
        overlap.sim_ns,
        serial.sim_ns
    );
}

#[test]
fn sim_replay_cluster_must_set_matches_cluster_occupancy() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0]), ev_seq(1, 0, 0, &[1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let packed = SimCfg {
        seq_streams: true,
        compute_slots: 4,
        cluster: 2,
        ..SimCfg::lru(2, 4096, 0)
    };
    let must = SimCfg {
        cluster_must_set: true,
        ..packed
    };
    let overlap = sim_replay_cfg(&t, profile.clone(), packed).expect("cluster");
    let serial = sim_replay_cfg(&t, profile, must).expect("cluster-must-set");
    assert_eq!(overlap.hits, serial.hits);
    assert_eq!(overlap.misses, serial.misses);
    assert_eq!(
        overlap.sim_ns.saturating_add(1),
        serial.sim_ns,
        "ClusterDimMustBeSet occupancy matches --cluster; SetAttribute is +1 ns; cluster={} must={}",
        overlap.sim_ns,
        serial.sim_ns
    );
}

#[test]
fn sim_replay_cluster_must_set_matches_cluster_occupancy_graphs() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0]), ev_seq(1, 0, 0, &[1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let packed = SimCfg {
        seq_streams: true,
        compute_slots: 4,
        cluster: 2,
        cuda_graphs: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let must = SimCfg {
        cluster_must_set: true,
        ..packed
    };
    let overlap = sim_replay_cfg(&t, profile.clone(), packed).expect("cluster graphs");
    let serial = sim_replay_cfg(&t, profile, must).expect("cluster-must-set graphs");
    assert_eq!(overlap.hits, serial.hits);
    assert_eq!(overlap.misses, serial.misses);
    assert_eq!(
        overlap.sim_ns.saturating_add(1),
        serial.sim_ns,
        "ClusterDimMustBeSet graphs inherit occupancy; SetAttribute is +1 ns; cluster={} must={}",
        overlap.sim_ns,
        serial.sim_ns
    );
}

#[test]
fn sim_replay_required_cluster_matches_cluster_occupancy() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0]), ev_seq(1, 0, 0, &[1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let packed = SimCfg {
        seq_streams: true,
        compute_slots: 4,
        cluster: 2,
        ..SimCfg::lru(2, 4096, 0)
    };
    let required = SimCfg {
        required_cluster: 2,
        ..packed
    };
    let overlap = sim_replay_cfg(&t, profile.clone(), packed).expect("cluster");
    let serial = sim_replay_cfg(&t, profile, required).expect("required-cluster");
    assert_eq!(overlap.hits, serial.hits);
    assert_eq!(overlap.misses, serial.misses);
    assert_eq!(
        overlap.sim_ns.saturating_add(1),
        serial.sim_ns,
        "RequiredClusterWidth occupancy matches --cluster; SetAttribute is +1 ns; cluster={} required={}",
        overlap.sim_ns,
        serial.sim_ns
    );
}

#[test]
fn sim_replay_required_cluster_matches_cluster_occupancy_graphs() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0]), ev_seq(1, 0, 0, &[1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let packed = SimCfg {
        seq_streams: true,
        compute_slots: 4,
        cluster: 2,
        cuda_graphs: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let required = SimCfg {
        required_cluster: 2,
        ..packed
    };
    let overlap = sim_replay_cfg(&t, profile.clone(), packed).expect("cluster graphs");
    let serial = sim_replay_cfg(&t, profile, required).expect("required-cluster graphs");
    assert_eq!(overlap.hits, serial.hits);
    assert_eq!(overlap.misses, serial.misses);
    assert_eq!(
        overlap.sim_ns.saturating_add(1),
        serial.sim_ns,
        "RequiredClusterWidth graphs inherit occupancy; SetAttribute is +1 ns; cluster={} required={}",
        overlap.sim_ns,
        serial.sim_ns
    );
}

#[test]
fn sim_replay_preferred_cluster_serializes_seq_streams() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0]), ev_seq(1, 0, 0, &[1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let packed = SimCfg {
        seq_streams: true,
        compute_slots: 4,
        cluster: 2,
        ..SimCfg::lru(2, 4096, 0)
    };
    let preferred = SimCfg {
        preferred_cluster: 4,
        ..packed
    };
    let overlap = sim_replay_cfg(&t, profile.clone(), packed).expect("packed");
    let serial = sim_replay_cfg(&t, profile, preferred).expect("preferred");
    assert_eq!(overlap.hits, serial.hits);
    assert_eq!(overlap.misses, serial.misses);
    assert!(
        overlap.sim_ns < serial.sim_ns,
        "preferred cluster of 4 must not Hyper-Q overlap leftover seq GEMMs; overlap={} serial={}",
        overlap.sim_ns,
        serial.sim_ns
    );
}

#[test]
fn sim_replay_max_shared_serializes_seq_streams() {
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
    let max_shared = SimCfg {
        max_shared: true,
        ..hyperq
    };
    let overlap = sim_replay_cfg(&t, profile.clone(), hyperq).expect("hyperq");
    let serial = sim_replay_cfg(&t, profile, max_shared).expect("max-shared");
    assert_eq!(overlap.hits, serial.hits);
    assert_eq!(overlap.misses, serial.misses);
    assert!(
        overlap.sim_ns < serial.sim_ns,
        "MaxShared carveout must not Hyper-Q overlap; overlap={} serial={}",
        overlap.sim_ns,
        serial.sim_ns
    );
}

#[test]
fn sim_replay_func_max_shared_serializes_seq_streams() {
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
    let func_max_shared = SimCfg {
        func_max_shared: true,
        ..hyperq
    };
    let overlap = sim_replay_cfg(&t, profile.clone(), hyperq).expect("hyperq");
    let serial = sim_replay_cfg(&t, profile, func_max_shared).expect("func-max-shared");
    assert_eq!(overlap.hits, serial.hits);
    assert_eq!(overlap.misses, serial.misses);
    assert!(
        overlap.sim_ns < serial.sim_ns,
        "func MaxShared carveout must not Hyper-Q overlap; overlap={} serial={}",
        overlap.sim_ns,
        serial.sim_ns
    );
}

#[test]
fn sim_replay_max_l1_restores_seq_stream_overlap() {
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
    let serial = SimCfg {
        func_max_shared: true,
        ..hyperq
    };
    let restored = SimCfg {
        max_l1: true,
        ..serial
    };
    let overlap = sim_replay_cfg(&t, profile.clone(), hyperq).expect("hyperq");
    let serial = sim_replay_cfg(&t, profile.clone(), serial).expect("func-max-shared");
    let restored = sim_replay_cfg(&t, profile, restored).expect("max-l1");
    assert_eq!(overlap.hits, restored.hits);
    assert_eq!(overlap.misses, restored.misses);
    assert!(
        restored.sim_ns < serial.sim_ns,
        "launch MaxL1 must restore Hyper-Q overlap; restored={} serial={}",
        restored.sim_ns,
        serial.sim_ns
    );
}

#[test]
fn sim_replay_nvlink_util_serializes_seq_streams() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0]), ev_seq(1, 0, 0, &[1])],
    };
    let nv = HardwareProfile::parse("gpus=2\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("nvlink slow gemm");
    assert!(nv.has_nvlink());
    let hyperq = SimCfg {
        seq_streams: true,
        compute_slots: 2,
        ..SimCfg::lru(2, 4096, 0)
    };
    let flagged = SimCfg {
        nvlink_util_centric: true,
        ..hyperq
    };
    let overlap = match sim_replay_cfg(&t, nv.clone(), hyperq) {
        Ok(row) => row,
        Err(err) => panic!("hyperq: {err}"),
    };
    let serial = match sim_replay_cfg(&t, nv, flagged) {
        Ok(row) => row,
        Err(err) => panic!("nvlink-util: {err}"),
    };
    assert_eq!(overlap.hits, serial.hits);
    assert_eq!(overlap.misses, serial.misses);
    assert!(
        overlap.sim_ns < serial.sim_ns,
        "NVLink-util-centric must not Hyper-Q overlap; overlap={} serial={}",
        overlap.sim_ns,
        serial.sim_ns
    );
    let h100 =
        HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n").expect("no nvlink");
    assert!(!h100.has_nvlink());
    let off = match sim_replay_cfg(&t, h100.clone(), hyperq) {
        Ok(row) => row,
        Err(err) => panic!("h100 off: {err}"),
    };
    let on = match sim_replay_cfg(&t, h100, flagged) {
        Ok(row) => row,
        Err(err) => panic!("h100 on: {err}"),
    };
    assert_eq!(
        off.sim_ns, on.sim_ns,
        "without NVLink the hint must not change occupancy; off={} on={}",
        off.sim_ns, on.sim_ns
    );
}

#[test]
fn sim_replay_device_launch_keeps_hits() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let p = HardwareProfile::example_h100_sxm();
    let base = SimCfg {
        cuda_graphs: true,
        ..SimCfg::lru(1, 4096, 0)
    };
    let host = match sim_replay_cfg(&t, p.clone(), base) {
        Ok(row) => row,
        Err(err) => panic!("host: {err}"),
    };
    let device = match sim_replay_cfg(
        &t,
        p,
        SimCfg {
            device_launch: true,
            ..base
        },
    ) {
        Ok(row) => row,
        Err(err) => panic!("device-launch: {err}"),
    };
    assert_eq!(host.hits, device.hits);
    assert_eq!(host.misses, device.misses);
    assert!(
        device.graph_launches > 0,
        "device-launch must replay GEMM graphs"
    );
}

#[test]
fn sim_replay_device_launch_refuses_graph_mem() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let err = sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            device_launch: true,
            graph_mem: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    );
    match err {
        Err(e) => assert!(
            e.to_string().contains("device-launch cannot graph-mem"),
            "{e}"
        ),
        Ok(_) => panic!("device-launch + graph-mem must fail"),
    }
}

#[test]
fn sim_replay_non_portable_cluster_allows_oversize() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::parse("gpus=1\nmax_blocks_per_cluster=16\n")
        .expect("open cluster profile");
    let oversize = SimCfg {
        cluster: 16,
        ..SimCfg::lru(1, 4096, 0)
    };
    let err = sim_replay_cfg(&t, profile.clone(), oversize).expect_err("portable");
    assert!(err.to_string().contains("non-portable cluster"), "{err}");
    let allowed = SimCfg {
        non_portable_cluster: true,
        ..oversize
    };
    let ok = sim_replay_cfg(&t, profile.clone(), allowed).expect("allowed");
    assert!(ok.misses > 0, "{}", ok.line());
    let too_big = SimCfg {
        cluster: 17,
        non_portable_cluster: true,
        ..SimCfg::lru(1, 4096, 0)
    };
    let err = sim_replay_cfg(&t, profile, too_big).expect_err("max");
    assert!(err.to_string().contains("cluster size"), "{err}");
}

#[test]
fn sim_replay_portable_cluster_overrides_func_attribute() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::parse("gpus=1\nmax_blocks_per_cluster=16\n")
        .expect("open cluster profile");
    let oversize = SimCfg {
        cluster: 16,
        ..SimCfg::lru(1, 4096, 0)
    };
    let allowed = SimCfg {
        portable_cluster: PortableClusterMode::AllowNonPortable,
        ..oversize
    };
    let ok = match sim_replay_cfg(&t, profile.clone(), allowed) {
        Ok(row) => row,
        Err(err) => panic!("AllowNonPortable: {err}"),
    };
    assert!(ok.misses > 0, "{}", ok.line());
    let refused = SimCfg {
        non_portable_cluster: true,
        portable_cluster: PortableClusterMode::RequirePortable,
        ..oversize
    };
    let err = match sim_replay_cfg(&t, profile, refused) {
        Ok(_) => panic!("RequirePortable must refuse oversize even with func attr"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("non-portable cluster"), "{err}");
}

#[test]
fn sim_replay_optin_shared_allows_oversize() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::parse("gpus=1\nmax_shared_mem_per_block_optin=232448\n")
        .expect("open shared profile");
    let oversize = SimCfg {
        dynamic_shared: 65_536,
        ..SimCfg::lru(1, 4096, 0)
    };
    let err = match sim_replay_cfg(&t, profile.clone(), oversize) {
        Ok(_) => panic!("oversize without permission must fail"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("non-portable shared"), "{err}");
    let allowed = SimCfg {
        optin_shared: true,
        ..oversize
    };
    let ok = match sim_replay_cfg(&t, profile, allowed) {
        Ok(row) => row,
        Err(err) => panic!("optin: {err}"),
    };
    assert!(ok.misses > 0, "{}", ok.line());
}

#[test]
fn sim_replay_portable_shared_overrides_func_attribute() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::parse("gpus=1\nmax_shared_mem_per_block_optin=232448\n")
        .expect("open shared profile");
    let oversize = SimCfg {
        dynamic_shared: 65_536,
        ..SimCfg::lru(1, 4096, 0)
    };
    let allowed = SimCfg {
        portable_shared: PortableSharedMode::AllowNonPortable,
        ..oversize
    };
    let ok = match sim_replay_cfg(&t, profile.clone(), allowed) {
        Ok(row) => row,
        Err(err) => panic!("AllowNonPortable: {err}"),
    };
    assert!(ok.misses > 0, "{}", ok.line());
    let refused = SimCfg {
        optin_shared: true,
        portable_shared: PortableSharedMode::RequirePortable,
        ..oversize
    };
    let err = match sim_replay_cfg(&t, profile, refused) {
        Ok(_) => panic!("RequirePortable must refuse oversize even with func attr"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("non-portable shared"), "{err}");
}

#[test]
fn sim_replay_sync_policy_taxes_decode_priority_itl() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let profile = HardwareProfile::parse(
        "gpus=1\nhost_sync_blocking_ns=10000\nfp16_flops=1000000\ncopy_engines=2\n",
    )
    .expect("host-sync profile");
    let base = SimCfg {
        decode_priority: true,
        stream_priority: true,
        ..SimCfg::lru(1, 4096, 0)
    };
    let auto = sim_replay_cfg(&t, profile.clone(), base).expect("auto");
    let block = sim_replay_cfg(
        &t,
        profile,
        SimCfg {
            sync_policy: SynchronizationPolicy::BlockingSync,
            ..base
        },
    )
    .expect("block");
    assert_eq!(auto.hits, block.hits);
    let auto_itl = auto.itl_ns.expect("auto itl");
    let block_itl = block.itl_ns.expect("block itl");
    assert!(
        block_itl > auto_itl,
        "blocking host wait must inflate decode ITL; auto={auto_itl} block={block_itl}"
    );
}

#[test]
fn sim_replay_device_sync_policy_taxes_decode_priority_itl() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let profile = HardwareProfile::parse(
        "gpus=1\nhost_sync_blocking_ns=10000\nfp16_flops=1000000\ncopy_engines=2\n",
    )
    .expect("host-sync profile");
    let base = SimCfg {
        decode_priority: true,
        stream_priority: true,
        ..SimCfg::lru(1, 4096, 0)
    };
    let auto = sim_replay_cfg(&t, profile.clone(), base).expect("auto");
    let block = sim_replay_cfg(
        &t,
        profile,
        SimCfg {
            device_sync_policy: SynchronizationPolicy::BlockingSync,
            ..base
        },
    )
    .expect("block");
    assert_eq!(auto.hits, block.hits);
    let auto_itl = auto.itl_ns.expect("auto itl");
    let block_itl = block.itl_ns.expect("block itl");
    assert!(
        block_itl > auto_itl,
        "device BlockingSync must inflate decode ITL; auto={auto_itl} block={block_itl}"
    );
}

#[test]
fn sim_replay_shared_mem_eight_lengthens() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0]), ev(2, 0, &[0])],
    };
    let profile = HardwareProfile::parse(
        "gpus=1\nshared_mem_eight_byte_permille=500\nfp16_flops=1000000\ncopy_engines=2\n",
    )
    .expect("shared-mem profile");
    let base = SimCfg::lru(1, 4096, 0);
    let def = sim_replay_cfg(&t, profile.clone(), base).expect("default");
    let eight = sim_replay_cfg(
        &t,
        profile,
        SimCfg {
            shared_mem: SharedMemoryMode::EightByte,
            ..base
        },
    )
    .expect("eight");
    assert_eq!(def.hits, eight.hits);
    assert_eq!(def.misses, eight.misses);
    assert!(
        eight.sim_ns > def.sim_ns,
        "EightByte at permille 500 must lengthen replay GEMMs; default={} eight={}",
        def.sim_ns,
        eight.sim_ns
    );
}

#[test]
fn sim_replay_func_shared_mem_eight_lengthens() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0]), ev(2, 0, &[0])],
    };
    let profile = HardwareProfile::parse(
        "gpus=1\nshared_mem_eight_byte_permille=500\nfp16_flops=1000000\ncopy_engines=2\n",
    )
    .expect("shared-mem profile");
    let base = SimCfg::lru(1, 4096, 0);
    let def = sim_replay_cfg(&t, profile.clone(), base).expect("default");
    let eight = sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            func_shared_mem: SharedMemoryMode::EightByte,
            ..base
        },
    )
    .expect("func eight");
    let overridden = sim_replay_cfg(
        &t,
        profile,
        SimCfg {
            func_shared_mem: SharedMemoryMode::EightByte,
            shared_mem: SharedMemoryMode::FourByte,
            ..base
        },
    )
    .expect("launch four overrides");
    assert_eq!(def.hits, eight.hits);
    assert_eq!(def.misses, eight.misses);
    assert!(
        eight.sim_ns > def.sim_ns,
        "func EightByte at permille 500 must lengthen replay GEMMs; default={} eight={}",
        def.sim_ns,
        eight.sim_ns
    );
    assert!(
        overridden.sim_ns < eight.sim_ns,
        "launch FourByte must override func EightByte; override={} eight={}",
        overridden.sim_ns,
        eight.sim_ns
    );
}

#[test]
fn sim_replay_device_shared_mem_eight_lengthens() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0]), ev(2, 0, &[0])],
    };
    let profile = HardwareProfile::parse(
        "gpus=1\nshared_mem_eight_byte_permille=500\nfp16_flops=1000000\ncopy_engines=2\n",
    )
    .expect("shared-mem profile");
    let base = SimCfg::lru(1, 4096, 0);
    let def = sim_replay_cfg(&t, profile.clone(), base).expect("default");
    let eight = sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            device_shared_mem: SharedMemoryMode::EightByte,
            ..base
        },
    )
    .expect("device eight");
    let overridden = sim_replay_cfg(
        &t,
        profile.clone(),
        SimCfg {
            device_shared_mem: SharedMemoryMode::EightByte,
            shared_mem: SharedMemoryMode::FourByte,
            ..base
        },
    )
    .expect("launch four overrides");
    let func_wins = sim_replay_cfg(
        &t,
        profile,
        SimCfg {
            device_shared_mem: SharedMemoryMode::FourByte,
            func_shared_mem: SharedMemoryMode::EightByte,
            ..base
        },
    )
    .expect("func eight overrides device");
    assert_eq!(def.hits, eight.hits);
    assert_eq!(def.misses, eight.misses);
    assert!(
        eight.sim_ns > def.sim_ns,
        "device EightByte at permille 500 must lengthen replay GEMMs; default={} eight={}",
        def.sim_ns,
        eight.sim_ns
    );
    assert!(
        overridden.sim_ns < eight.sim_ns,
        "launch FourByte must override device EightByte; override={} eight={}",
        overridden.sim_ns,
        eight.sim_ns
    );
    assert!(
        func_wins.sim_ns > def.sim_ns,
        "func EightByte must override device FourByte; default={} func={}",
        def.sim_ns,
        func_wins.sim_ns
    );
}

#[test]
fn sim_replay_pdl_overlaps_same_stream_gemms() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0, 1]), ev_seq(0, 1, 0, &[0, 1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let serial = SimCfg {
        compute_slots: 2,
        ..SimCfg::lru(2, 4096, 0)
    };
    let pdl = SimCfg {
        pdl: true,
        ..serial
    };
    let off = sim_replay_cfg(&t, profile.clone(), serial).expect("serial");
    let on = sim_replay_cfg(&t, profile, pdl).expect("pdl");
    assert_eq!(off.hits, on.hits);
    assert_eq!(off.misses, on.misses);
    assert!(
        on.sim_ns < off.sim_ns,
        "PDL must overlap consecutive same-stream GEMMs; pdl={} serial={}",
        on.sim_ns,
        off.sim_ns
    );
}

#[test]
fn sim_replay_pdl_cuda_graphs_overlaps_same_stream() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0, 1]), ev_seq(0, 1, 0, &[0, 1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let serial = SimCfg {
        compute_slots: 2,
        cuda_graphs: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let pdl = SimCfg {
        pdl: true,
        ..serial
    };
    let off = sim_replay_cfg(&t, profile.clone(), serial).expect("serial");
    let on = sim_replay_cfg(&t, profile, pdl).expect("pdl");
    assert_eq!(off.hits, on.hits);
    assert_eq!(off.misses, on.misses);
    assert!(
        on.sim_ns < off.sim_ns,
        "PDL graph leaves must overlap consecutive same-stream GEMMs; pdl={} serial={}",
        on.sim_ns,
        off.sim_ns
    );
}

#[test]
fn sim_replay_pdl_cuda_graphs_evicts_without_lease() {
    let t = load_checked_in_trace("tiny-qwen3moe-2layer.jsonl");
    let row = sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            compute_slots: 2,
            cuda_graphs: true,
            pdl: true,
            ..SimCfg::lru(2, 4096, 0)
        },
    )
    .expect("pdl graphs");
    assert_eq!(row.misses, 36);
    assert!(row.graph_launches > 0);
}

#[test]
fn sim_replay_pdl_rejects_cooperative() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let err = sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            pdl: true,
            cooperative: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    )
    .expect_err("conflict");
    assert!(
        err.to_string().contains("choose one of pdl, cooperative"),
        "{err}"
    );
}

#[test]
fn sim_replay_l2_persist_reused_expert_is_faster() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let profile = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000000000000\nlaunch_overhead_ns=1\nhbm_bps=1000000000\nl2_persist_hit_permille=1000\n",
    )
    .expect("memory-bound persist profile");
    let cold = SimCfg::lru(2, 4096, 0);
    let warm = SimCfg {
        l2_persist: true,
        ..cold
    };
    let off = sim_replay_cfg(&t, profile.clone(), cold).expect("cold");
    let on = sim_replay_cfg(&t, profile, warm).expect("persist");
    assert_eq!(off.hits, on.hits);
    assert_eq!(off.misses, on.misses);
    assert!(
        on.sim_ns < off.sim_ns,
        "persisting L2 must speed a reused expert GEMM; persist={} cold={}",
        on.sim_ns,
        off.sim_ns
    );
}

#[test]
fn sim_replay_l2_ratio_partial_is_slower_than_full() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let profile = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000000000000\nlaunch_overhead_ns=1\nhbm_bps=1000000000\nl2_persist_hit_permille=1000\n",
    )
    .expect("memory-bound persist profile");
    let persist = SimCfg {
        l2_persist: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let half = SimCfg {
        l2_ratio: 500,
        ..SimCfg::lru(2, 4096, 0)
    };
    let full_flag = SimCfg {
        l2_persist: true,
        l2_ratio: 1000,
        ..SimCfg::lru(2, 4096, 0)
    };
    let a = sim_replay_cfg(&t, profile.clone(), persist).expect("persist");
    let b = sim_replay_cfg(&t, profile.clone(), half).expect("ratio-500");
    let c = sim_replay_cfg(&t, profile, full_flag).expect("ratio-1000");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
    assert_eq!(a.hits, c.hits);
    assert_eq!(a.misses, c.misses);
    assert_eq!(
        a.sim_ns, c.sim_ns,
        "hitRatio 1000 must match persist; persist={} ratio1000={}",
        a.sim_ns, c.sim_ns
    );
    assert!(
        b.sim_ns > a.sim_ns,
        "hitRatio 500 must bill more HBM than full persist; half={} full={}",
        b.sim_ns,
        a.sim_ns
    );
}

#[test]
fn sim_replay_l2_ratio_refuses_over_1000() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let on = SimCfg {
        l2_ratio: 1001,
        ..SimCfg::lru(2, 4096, 0)
    };
    match sim_replay_cfg(&t, profile, on) {
        Ok(_) => panic!("l2-ratio 1001 must fail"),
        Err(err) => assert!(
            err.to_string().contains("l2-ratio must be 1..=1000"),
            "{err}"
        ),
    }
}

#[test]
fn sim_replay_l2_streaming_keeps_hits() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let profile = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000000000000\nlaunch_overhead_ns=1\nhbm_bps=1000000000\nl2_persist_hit_permille=1000\n",
    )
    .expect("memory-bound persist profile");
    let persist = SimCfg {
        l2_persist: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let streaming = SimCfg {
        l2_persist: true,
        l2_streaming: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let a = sim_replay_cfg(&t, profile.clone(), persist).expect("persist");
    let b = sim_replay_cfg(&t, profile, streaming).expect("streaming");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
    assert!(
        b.sim_ns > a.sim_ns,
        "streaming hits must bill more HBM than persist; streaming={} persist={}",
        b.sim_ns,
        a.sim_ns
    );
}

#[test]
fn sim_replay_l2_streaming_needs_l2_persist() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let on = SimCfg {
        l2_streaming: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    match sim_replay_cfg(&t, profile.clone(), on) {
        Ok(_) => panic!("l2-streaming without persist must fail"),
        Err(err) => assert!(
            err.to_string().contains("l2-streaming needs l2-persist"),
            "{err}"
        ),
    }
    let ratio = SimCfg {
        l2_ratio: 500,
        l2_streaming: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let _ok = sim_replay_cfg(&t, profile, ratio).expect("ratio arms persist");
}

#[test]
fn sim_replay_l2_reset_colds_reused_expert() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let profile = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000000000000\nlaunch_overhead_ns=1\nhbm_bps=1000000000\nl2_persist_hit_permille=1000\n",
    )
    .expect("memory-bound persist profile");
    let persist = SimCfg {
        l2_persist: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let reset = SimCfg {
        l2_persist: true,
        l2_reset: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let only_reset = SimCfg {
        l2_reset: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let warm = sim_replay_cfg(&t, profile.clone(), persist).expect("persist");
    let cold = sim_replay_cfg(&t, profile.clone(), reset).expect("persist+reset");
    let implied = sim_replay_cfg(&t, profile, only_reset).expect("reset-implies-persist");
    assert_eq!(warm.hits, cold.hits);
    assert_eq!(warm.misses, cold.misses);
    assert_eq!(implied.hits, cold.hits);
    assert_eq!(implied.misses, cold.misses);
    assert!(
        warm.sim_ns < cold.sim_ns,
        "reset must cold reused persist lines; persist={} reset={}",
        warm.sim_ns,
        cold.sim_ns
    );
    assert_eq!(
        implied.sim_ns, cold.sim_ns,
        "bare --l2-reset must imply persist windows; implied={} reset={}",
        implied.sim_ns, cold.sim_ns
    );
}

#[test]
fn sim_replay_l2_reset_with_graphs_keeps_hits() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let persist = SimCfg {
        cuda_graphs: true,
        l2_persist: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let reset = SimCfg {
        cuda_graphs: true,
        l2_persist: true,
        l2_reset: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let a = sim_replay_cfg(&t, profile.clone(), persist).expect("persist graphs");
    let b = sim_replay_cfg(&t, profile, reset).expect("reset graphs");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn simulated_gpu_store_pdl_overlaps_same_stream_gemms() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let run = |pdl: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            2,
            profile.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                compute_slots: 2,
                pdl,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        let k0 = ExpertKey::new(0, 0);
        let k1 = ExpertKey::new(0, 1);
        let _w0 = gpu.acquire(k0).expect("warm 0");
        gpu.release(k0);
        let _w1 = gpu.acquire(k1).expect("warm 1");
        gpu.release(k1);
        let t0 = gpu.clock_ns().expect("drain");
        let _a = gpu.acquire(k0).expect("hit 0");
        gpu.release(k0);
        let _b = gpu.acquire(k1).expect("hit 1");
        gpu.release(k1);
        gpu.score().expect("score").wall_ns.saturating_sub(t0)
    };
    let serial = run(false);
    let overlap = run(true);
    assert!(
        overlap < serial,
        "PDL must overlap consecutive same-stream store GEMMs; pdl={overlap} serial={serial}"
    );
}

#[test]
fn simulated_gpu_store_pdl_rejects_cooperative() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let err = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            pdl: true,
            cooperative: true,
            ..GpuStoreCfg::default()
        },
    )
    .err()
    .expect("conflict");
    assert!(
        err.to_string().contains("choose one of pdl, cooperative"),
        "{err}"
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
fn sim_replay_mem_sync_domain_keeps_hits() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg {
        decode_priority: true,
        stream_priority: true,
        ..SimCfg::lru(1, 4096, 0)
    };
    let on = SimCfg {
        mem_sync_domain: MemSyncDomain::Remote,
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

#[test]
fn store_replay_mem_sync_domain_binds_later_tokens() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let mut run = StoreReplayCfg::demand(1, 4096, GpuFill::Pinned);
    run.gpu.decode_priority = true;
    run.gpu.stream_priority = true;
    run.gpu.mem_sync_domain = MemSyncDomain::Remote;
    let row = store_replay_cfg(&t, HardwareProfile::example_h100_sxm(), run).expect("store");
    assert_eq!(row.metrics.misses, 1);
    assert_eq!(row.metrics.hits, 1);
    assert!(row.score.wall_ns > 0);
}

#[test]
fn simulated_gpu_store_mem_sync_domain_marks_decode_stream() {
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
            decode_priority: true,
            stream_priority: true,
            mem_sync_domain: MemSyncDomain::Remote,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let d = DeviceId(0);
    assert_eq!(
        gpu.stream_mem_sync_domain(d, gpu.decode_stream()),
        MemSyncDomain::Remote
    );
    assert_eq!(
        gpu.stream_mem_sync_domain(d, gpu.prefill_stream()),
        MemSyncDomain::Default
    );
}

#[test]
fn simulated_gpu_store_mem_sync_map_collapse_marks_decode_stream() {
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
            decode_priority: true,
            stream_priority: true,
            mem_sync_domain: MemSyncDomain::Remote,
            mem_sync_collapse: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let d = DeviceId(0);
    assert_eq!(
        gpu.stream_mem_sync_domain_map(d, gpu.decode_stream())
            .expect("decode map")
            .remote,
        0
    );
    assert_eq!(
        gpu.stream_mem_sync_domain_map(d, gpu.prefill_stream())
            .expect("prefill map")
            .remote,
        1
    );
    let inner = DirectStore::from_trace(&t);
    let identity =
        SimulatedGpuStore::new(inner, 1, HardwareProfile::example_h100_sxm(), 4096).expect("id");
    assert_eq!(
        identity
            .stream_mem_sync_domain_map(d, identity.decode_stream())
            .expect("id map")
            .remote,
        1
    );
}

#[test]
fn simulated_gpu_store_mem_sync_map_collapse_needs_remote() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            mem_sync_collapse: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("collapse without remote must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("mem-sync-map collapse needs mem-sync-domain remote"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_mem_sync_launch_marks_gemm() {
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
            decode_priority: true,
            stream_priority: true,
            mem_sync_domain: MemSyncDomain::Remote,
            mem_sync_launch: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    assert!(gpu.mem_sync_launch());
    let inner = DirectStore::from_trace(&t);
    let identity =
        SimulatedGpuStore::new(inner, 1, HardwareProfile::example_h100_sxm(), 4096).expect("id");
    assert!(!identity.mem_sync_launch());
}

#[test]
fn simulated_gpu_store_mem_sync_launch_needs_remote() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            mem_sync_launch: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("launch without remote must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("mem-sync-launch needs mem-sync-domain remote"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_mem_sync_launch_map_marks_gemm() {
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
            decode_priority: true,
            stream_priority: true,
            mem_sync_domain: MemSyncDomain::Remote,
            mem_sync_launch_map: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    assert!(gpu.mem_sync_launch_map());
    let inner = DirectStore::from_trace(&t);
    let identity =
        SimulatedGpuStore::new(inner, 1, HardwareProfile::example_h100_sxm(), 4096).expect("id");
    assert!(!identity.mem_sync_launch_map());
}

#[test]
fn simulated_gpu_store_mem_sync_launch_map_needs_remote() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            mem_sync_launch_map: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("launch-map without remote must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("mem-sync-launch-map needs mem-sync-domain remote"),
            "{err}"
        ),
    }
}

#[test]
fn sim_replay_mem_sync_map_collapse_keeps_hits() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg {
        decode_priority: true,
        stream_priority: true,
        mem_sync_domain: MemSyncDomain::Remote,
        ..SimCfg::lru(1, 4096, 0)
    };
    let on = SimCfg {
        mem_sync_collapse: true,
        ..off
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("off");
    let b = sim_replay_cfg(&t, profile, on).expect("collapse");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn sim_replay_mem_sync_map_collapse_needs_remote() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let on = SimCfg {
        mem_sync_collapse: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    match sim_replay_cfg(&t, profile, on) {
        Ok(_) => panic!("collapse without remote must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("mem-sync-map collapse needs mem-sync-domain remote"),
            "{err}"
        ),
    }
}

#[test]
fn sim_replay_mem_sync_launch_keeps_hits() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg {
        decode_priority: true,
        stream_priority: true,
        mem_sync_domain: MemSyncDomain::Remote,
        ..SimCfg::lru(1, 4096, 0)
    };
    let on = SimCfg {
        mem_sync_launch: true,
        ..off
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("off");
    let b = sim_replay_cfg(&t, profile, on).expect("launch");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn sim_replay_mem_sync_launch_needs_remote() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let on = SimCfg {
        mem_sync_launch: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    match sim_replay_cfg(&t, profile, on) {
        Ok(_) => panic!("launch without remote must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("mem-sync-launch needs mem-sync-domain remote"),
            "{err}"
        ),
    }
}

#[test]
fn sim_replay_mem_sync_launch_map_keeps_hits() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg {
        decode_priority: true,
        stream_priority: true,
        mem_sync_domain: MemSyncDomain::Remote,
        ..SimCfg::lru(1, 4096, 0)
    };
    let on = SimCfg {
        mem_sync_launch_map: true,
        ..off
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("off");
    let b = sim_replay_cfg(&t, profile, on).expect("launch-map");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn sim_replay_mem_sync_launch_map_needs_remote() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let on = SimCfg {
        mem_sync_launch_map: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    match sim_replay_cfg(&t, profile, on) {
        Ok(_) => panic!("launch-map without remote must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("mem-sync-launch-map needs mem-sync-domain remote"),
            "{err}"
        ),
    }
}

#[test]
fn sim_cfg_launch_completion_refuses_device_launch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let err = match sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            launch_completion: true,
            device_launch: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    ) {
        Ok(_) => panic!("launch-completion + device-launch must fail"),
        Err(e) => e,
    };
    assert!(
        err.to_string()
            .contains("launch-completion cannot device-launch"),
        "{err}"
    );
}

#[test]
fn simulated_gpu_store_launch_completion_refuses_device_launch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            launch_completion: true,
            device_launch: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("launch-completion + device-launch must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("launch-completion cannot device-launch"),
            "{err}"
        ),
    }
}

#[test]
fn sim_replay_launch_completion_keeps_hits() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg::lru(1, 4096, 0);
    let on = SimCfg {
        launch_completion: true,
        ..off
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("off");
    let b = sim_replay_cfg(&t, profile, on).expect("on");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn simulated_gpu_store_launch_completion_pin_before_acquire_still_replicates() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let mut gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        2,
        HardwareProfile::example_8xh100_nvlink(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            launch_completion: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    gpu.pin_hot(&[k0]).expect("pin");
    assert_eq!(gpu.replica_of(k0), Some(DeviceId(1)));
    assert_eq!(gpu.metrics().replicates, 1);
    let _s = gpu.score().expect("score");
}

#[test]
fn simulated_gpu_store_launch_completion_overlaps_replica() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::parse("gpus=2\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("nvlink slow gemm");
    assert!(profile.has_nvlink());
    let run = |launch_completion: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = match SimulatedGpuStore::with_cfg(
            inner,
            2,
            profile.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                launch_completion,
                ..GpuStoreCfg::default()
            },
        ) {
            Ok(gpu) => gpu,
            Err(err) => panic!("gpu: {err}"),
        };
        let k0 = ExpertKey::new(0, 0);
        let _p = match gpu.acquire(k0) {
            Ok(v) => v,
            Err(err) => panic!("acquire: {err}"),
        };
        gpu.pin_hot(&[k0]).expect("pin");
        let score = gpu.score().expect("score");
        let gemm = gpu
            .operations()
            .find(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .expect("kernel");
        let copy = gpu
            .operations()
            .filter(|o| match &o.kind {
                GpuOp::Memcpy(m) => {
                    matches!((m.src, m.dst), (Place::Device(_), Place::Device(_)))
                }
                _ => false,
            })
            .last()
            .expect("d2d");
        (
            score.wall_ns,
            gpu.metrics().replicates,
            gemm.start_ns.expect("k start"),
            gemm.done_ns.expect("k done"),
            copy.start_ns.expect("copy start"),
        )
    };
    let (off_wall, off_rep, _, off_done, off_copy) = run(false);
    let (on_wall, on_rep, on_start, on_done, on_copy) = run(true);
    assert_eq!(off_rep, 1);
    assert_eq!(on_rep, 1);
    assert!(
        off_copy >= off_done,
        "identity pin_hot must wait GEMM done; copy={off_copy} kdone={off_done}"
    );
    assert!(
        on_copy >= on_start,
        "launch-completion copy must wait kernel start; copy={on_copy} kstart={on_start}"
    );
    assert!(
        on_copy < on_done,
        "launch-completion replica must overlap leftover GEMM; copy={on_copy} kdone={on_done}"
    );
    assert!(
        on_wall < off_wall,
        "launch-completion replica overlap must shorten wall; on={on_wall} off={off_wall}"
    );
}

#[test]
fn sim_cfg_programmatic_event_refuses_device_launch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let err = match sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            programmatic_event: true,
            device_launch: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    ) {
        Ok(_) => panic!("programmatic-event + device-launch must fail"),
        Err(e) => e,
    };
    assert!(
        err.to_string()
            .contains("programmatic-event cannot device-launch"),
        "{err}"
    );
}

#[test]
fn simulated_gpu_store_programmatic_event_refuses_device_launch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            programmatic_event: true,
            device_launch: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("programmatic-event + device-launch must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("programmatic-event cannot device-launch"),
            "{err}"
        ),
    }
}

#[test]
fn sim_replay_programmatic_event_keeps_hits() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg::lru(1, 4096, 0);
    let on = SimCfg {
        programmatic_event: true,
        ..off
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("off");
    let b = sim_replay_cfg(&t, profile, on).expect("on");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn simulated_gpu_store_programmatic_event_pin_before_acquire_still_replicates() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let mut gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        2,
        HardwareProfile::example_8xh100_nvlink(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            programmatic_event: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    gpu.pin_hot(&[k0]).expect("pin");
    assert_eq!(gpu.replica_of(k0), Some(DeviceId(1)));
    assert_eq!(gpu.metrics().replicates, 1);
    let _s = gpu.score().expect("score");
}

#[test]
fn simulated_gpu_store_programmatic_event_overlaps_replica() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::parse("gpus=2\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("nvlink slow gemm");
    assert!(profile.has_nvlink());
    let run = |programmatic_event: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = match SimulatedGpuStore::with_cfg(
            inner,
            2,
            profile.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                programmatic_event,
                ..GpuStoreCfg::default()
            },
        ) {
            Ok(gpu) => gpu,
            Err(err) => panic!("gpu: {err}"),
        };
        let k0 = ExpertKey::new(0, 0);
        let _p = match gpu.acquire(k0) {
            Ok(v) => v,
            Err(err) => panic!("acquire: {err}"),
        };
        gpu.pin_hot(&[k0]).expect("pin");
        let score = gpu.score().expect("score");
        let gemm = gpu
            .operations()
            .find(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .expect("kernel");
        let copy = gpu
            .operations()
            .filter(|o| match &o.kind {
                GpuOp::Memcpy(m) => {
                    matches!((m.src, m.dst), (Place::Device(_), Place::Device(_)))
                }
                _ => false,
            })
            .last()
            .expect("d2d");
        (
            score.wall_ns,
            gpu.metrics().replicates,
            gemm.start_ns.expect("k start"),
            gemm.done_ns.expect("k done"),
            copy.start_ns.expect("copy start"),
        )
    };
    let (off_wall, off_rep, _, off_done, off_copy) = run(false);
    let (on_wall, on_rep, on_start, on_done, on_copy) = run(true);
    assert_eq!(off_rep, 1);
    assert_eq!(on_rep, 1);
    assert!(
        off_copy >= off_done,
        "identity pin_hot must wait GEMM done; copy={off_copy} kdone={off_done}"
    );
    assert!(
        on_copy > on_start,
        "programmatic-event copy must wait PDL trigger; copy={on_copy} kstart={on_start}"
    );
    assert!(
        on_copy < on_done,
        "programmatic-event replica must overlap leftover GEMM; copy={on_copy} kdone={on_done}"
    );
    assert!(
        on_wall < off_wall,
        "programmatic-event replica overlap must shorten wall; on={on_wall} off={off_wall}"
    );
}

#[test]
fn sim_replay_wait_value_keeps_hits() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg::lru(1, 4096, 0);
    let on = SimCfg {
        wait_value: true,
        ..off
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("off");
    let b = sim_replay_cfg(&t, profile, on).expect("on");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn sim_replay_wait_value_decode_priority_keeps_hits() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 1, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg {
        decode_priority: true,
        ..SimCfg::lru(1, 4096, 0)
    };
    let on = SimCfg {
        wait_value: true,
        ..off
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("off");
    let b = sim_replay_cfg(&t, profile, on).expect("on");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn simulated_gpu_store_wait_value_compute_waits_copy_write() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let mut gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            wait_value: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    let _p = gpu.acquire(k0).expect("acquire");
    let _s = gpu.score().expect("score");
    let write = gpu
        .operations()
        .find(|o| matches!(o.kind, GpuOp::WriteValue { .. }))
        .expect("write-value");
    let wait = gpu
        .operations()
        .find(|o| matches!(o.kind, GpuOp::WaitValue { .. }))
        .expect("wait-value");
    let gemm = gpu
        .operations()
        .find(|o| matches!(o.kind, GpuOp::Kernel { .. }))
        .expect("kernel");
    assert_eq!(write.stream, gpu.copy_stream());
    assert_eq!(wait.stream, gpu.compute_stream());
    let write_done = write.done_ns.expect("write done");
    let wait_start = wait.start_ns.expect("wait start");
    let gemm_start = gemm.start_ns.expect("gemm start");
    assert!(
        wait_start >= write_done,
        "wait must see the copy-stream write; wait={wait_start} write={write_done}"
    );
    assert!(
        gemm_start >= wait_start,
        "GEMM must wait the mailbox; gemm={gemm_start} wait={wait_start}"
    );
    let mb = match write.kind {
        GpuOp::WriteValue { id, .. } => id,
        other => panic!("write-value: {other:?}"),
    };
    let expert = match &gemm.kind {
        GpuOp::Kernel { reads, .. } => reads.first().expect("read").id,
        other => panic!("kernel: {other:?}"),
    };
    assert_ne!(mb, expert, "mailbox must not be the expert page");
}

#[test]
fn simulated_gpu_store_wait_value_does_not_drain_leftover_prefill() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let mut gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        2,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            wait_value: true,
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
        "wait-value must not cudaMalloc-drain leftover prefill; token={token} full={full}"
    );
}

#[test]
fn simulated_gpu_store_wait_value_pin_hot_still_replicates() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let mut gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        2,
        HardwareProfile::example_8xh100_nvlink(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            wait_value: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    let _p = gpu.acquire(k0).expect("acquire");
    gpu.pin_hot(&[k0]).expect("pin");
    assert_eq!(gpu.replica_of(k0), Some(DeviceId(1)));
    assert_eq!(gpu.metrics().replicates, 1);
    let _s = gpu.score().expect("score");
}

#[test]
fn sim_cfg_stream_attach_needs_managed() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let err = match sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            stream_attach: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    ) {
        Ok(_) => panic!("stream-attach without managed must fail"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("stream-attach needs managed"),
        "{err}"
    );
}

#[test]
fn simulated_gpu_store_stream_attach_needs_managed() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            stream_attach: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("stream-attach without managed must fail"),
        Err(err) => assert!(
            err.to_string().contains("stream-attach needs managed"),
            "{err}"
        ),
    }
}

#[test]
fn sim_cfg_stream_attach_refuses_seq_streams() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let err = match sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            managed: true,
            stream_attach: true,
            seq_streams: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    ) {
        Ok(_) => panic!("stream-attach + seq-streams must fail"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("stream-attach cannot seq-streams"),
        "{err}"
    );
}

#[test]
fn simulated_gpu_store_stream_attach_refuses_seq_streams() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Managed,
        GpuStoreCfg {
            stream_attach: true,
            seq_streams: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("stream-attach + seq-streams must fail"),
        Err(err) => assert!(
            err.to_string().contains("stream-attach cannot seq-streams"),
            "{err}"
        ),
    }
}

#[test]
fn sim_replay_stream_attach_keeps_hits() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg {
        managed: true,
        ..SimCfg::lru(1, 4096, 0)
    };
    let on = SimCfg {
        managed: true,
        stream_attach: true,
        ..SimCfg::lru(1, 4096, 0)
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("off");
    let b = sim_replay_cfg(&t, profile, on).expect("on");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn simulated_gpu_store_stream_attach_pin_before_acquire_still_replicates() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let mut gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        2,
        HardwareProfile::example_8xh100_nvlink(),
        4096,
        GpuFill::Managed,
        GpuStoreCfg {
            stream_attach: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    gpu.pin_hot(&[k0]).expect("pin");
    assert_eq!(gpu.replica_of(k0), Some(DeviceId(1)));
    assert_eq!(gpu.metrics().replicates, 1);
    let _s = gpu.score().expect("score");
}

#[test]
fn simulated_gpu_store_stream_attach_serializes_miss_prefetch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let profile =
        HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n").expect("slow gemm");
    let run = |stream_attach: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = match SimulatedGpuStore::with_cfg(
            inner,
            2,
            profile.clone(),
            4096,
            GpuFill::Managed,
            GpuStoreCfg {
                stream_attach,
                ..GpuStoreCfg::default()
            },
        ) {
            Ok(gpu) => gpu,
            Err(err) => panic!("gpu: {err}"),
        };
        let k0 = ExpertKey::new(0, 0);
        let k1 = ExpertKey::new(0, 1);
        let _p = match gpu.acquire(k0) {
            Ok(v) => v,
            Err(err) => panic!("warm: {err}"),
        };
        gpu.release(k0);
        let _d = gpu.clock_ns().expect("drain");
        let _p = match gpu.acquire(k0) {
            Ok(v) => v,
            Err(err) => panic!("leftover: {err}"),
        };
        gpu.release(k0);
        let _p = match gpu.acquire(k1) {
            Ok(v) => v,
            Err(err) => panic!("miss: {err}"),
        };
        gpu.release(k1);
        let score = gpu.score().expect("score");
        let kernels: Vec<_> = gpu
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .collect();
        let prefetches: Vec<_> = gpu
            .operations()
            .filter(|o| match &o.kind {
                GpuOp::Memcpy(m) => !matches!((m.src, m.dst), (Place::Device(_), Place::Device(_))),
                _ => false,
            })
            .collect();
        let attaches = gpu
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Attach { .. }))
            .count();
        let leftover = kernels
            .get(kernels.len().saturating_sub(2))
            .expect("leftover gemm");
        let miss = prefetches.last().expect("miss prefetch");
        (
            score.wall_ns,
            leftover.done_ns.expect("k done"),
            miss.start_ns.expect("prefetch start"),
            attaches,
        )
    };
    let (off_wall, off_done, off_pf, off_att) = run(false);
    let (on_wall, on_done, on_pf, on_att) = run(true);
    assert_eq!(off_att, 0);
    assert!(on_att >= 1, "stream-attach must submit Attach; n={on_att}");
    assert!(
        off_pf < off_done,
        "identity managed prefetch must overlap leftover GEMM; pf={off_pf} kdone={off_done}"
    );
    assert!(
        on_pf >= on_done,
        "stream-attach prefetch must wait leftover GEMM; pf={on_pf} kdone={on_done}"
    );
    assert!(
        on_wall > off_wall,
        "stream-attach must lengthen wall; on={on_wall} off={off_wall}"
    );
}

#[test]
fn sim_cfg_managed_host_needs_managed() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let err = match sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            managed_host: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    ) {
        Ok(_) => panic!("managed-host without managed must fail"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("managed-host needs managed"),
        "{err}"
    );
}

#[test]
fn simulated_gpu_store_managed_host_needs_managed() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            managed_host: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("managed-host without managed must fail"),
        Err(err) => assert!(
            err.to_string().contains("managed-host needs managed"),
            "{err}"
        ),
    }
}

#[test]
fn sim_replay_managed_host_keeps_hits() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg {
        managed: true,
        ..SimCfg::lru(1, 4096, 0)
    };
    let on = SimCfg {
        managed: true,
        managed_host: true,
        ..SimCfg::lru(1, 4096, 0)
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("off");
    let b = sim_replay_cfg(&t, profile, on).expect("on");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn simulated_gpu_store_managed_host_pin_before_acquire_still_replicates() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let mut gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        2,
        HardwareProfile::example_8xh100_nvlink(),
        4096,
        GpuFill::Managed,
        GpuStoreCfg {
            managed_host: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    gpu.pin_hot(&[k0]).expect("pin");
    assert_eq!(gpu.replica_of(k0), Some(DeviceId(1)));
    assert_eq!(gpu.metrics().replicates, 1);
    let _s = gpu.score().expect("score");
}

#[test]
fn simulated_gpu_store_managed_host_attach_overlaps_leftover() {
    let t = Trace {
        events: vec![ev(0, 0, &[0, 1])],
    };
    let profile =
        HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n").expect("slow gemm");
    let run = |managed_host: bool, stream_attach: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = match SimulatedGpuStore::with_cfg(
            inner,
            2,
            profile.clone(),
            4096,
            GpuFill::Managed,
            GpuStoreCfg {
                managed_host,
                stream_attach,
                ..GpuStoreCfg::default()
            },
        ) {
            Ok(gpu) => gpu,
            Err(err) => panic!("gpu: {err}"),
        };
        let k0 = ExpertKey::new(0, 0);
        let k1 = ExpertKey::new(0, 1);
        let _p = match gpu.acquire(k0) {
            Ok(v) => v,
            Err(err) => panic!("warm: {err}"),
        };
        gpu.release(k0);
        let _d = gpu.clock_ns().expect("drain");
        let _p = match gpu.acquire(k0) {
            Ok(v) => v,
            Err(err) => panic!("leftover: {err}"),
        };
        gpu.release(k0);
        let _p = match gpu.acquire(k1) {
            Ok(v) => v,
            Err(err) => panic!("miss: {err}"),
        };
        gpu.release(k1);
        let score = gpu.score().expect("score");
        let kernels: Vec<_> = gpu
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .collect();
        let prefetches: Vec<_> = gpu
            .operations()
            .filter(|o| match &o.kind {
                GpuOp::Memcpy(m) => !matches!((m.src, m.dst), (Place::Device(_), Place::Device(_))),
                _ => false,
            })
            .collect();
        let attaches: Vec<_> = gpu
            .operations()
            .filter_map(|o| match o.kind {
                GpuOp::Attach { flags, .. } => Some(flags),
                _ => None,
            })
            .collect();
        let leftover = kernels
            .get(kernels.len().saturating_sub(2))
            .expect("leftover gemm");
        let miss = prefetches.last().expect("miss prefetch");
        (
            score.wall_ns,
            leftover.done_ns.expect("k done"),
            miss.start_ns.expect("prefetch start"),
            attaches,
        )
    };
    let (off_wall, off_done, off_pf, off_att) = run(false, false);
    let (host_wall, host_done, host_pf, host_att) = run(true, false);
    let (_both_wall, both_done, both_pf, both_att) = run(true, true);
    assert!(
        off_att.is_empty(),
        "identity managed has no Attach; {off_att:?}"
    );
    assert!(
        host_att.contains(&gpu_sim::MemAttach::Global),
        "managed-host must Global-attach; {host_att:?}"
    );
    assert!(
        both_att.contains(&gpu_sim::MemAttach::Single),
        "managed-host + stream-attach must Single-attach; {both_att:?}"
    );
    assert!(
        off_pf < off_done,
        "identity managed prefetch must overlap leftover GEMM; pf={off_pf} kdone={off_done}"
    );
    assert!(
        host_pf < host_done,
        "managed-host prefetch must still overlap leftover GEMM; pf={host_pf} kdone={host_done}"
    );
    assert!(
        both_pf >= both_done,
        "managed-host + stream-attach prefetch must wait leftover GEMM; pf={both_pf} kdone={both_done}"
    );
    assert!(
        host_wall >= off_wall,
        "managed-host Attach must not shorten wall; host={host_wall} off={off_wall}"
    );
}

fn count_host_prefetch(gpu: &SimulatedGpuStore) -> usize {
    gpu.operations()
        .filter(|o| match &o.kind {
            GpuOp::Memcpy(m) => {
                matches!(m.dst, Place::Host | Place::HostPinned)
                    && matches!(m.src, Place::Device(_))
            }
            _ => false,
        })
        .count()
}

#[test]
fn sim_cfg_prefetch_host_needs_managed() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let err = match sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            prefetch_host: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    ) {
        Ok(_) => panic!("prefetch-host without managed must fail"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("prefetch-host needs managed"),
        "{err}"
    );
}

#[test]
fn sim_cfg_no_read_mostly_needs_managed() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let err = match sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            no_read_mostly: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    ) {
        Ok(_) => panic!("no-read-mostly without managed must fail"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("no-read-mostly needs managed"),
        "{err}"
    );
}

#[test]
fn sim_cfg_no_preferred_needs_managed() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let err = match sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            no_preferred: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    ) {
        Ok(_) => panic!("no-preferred without managed must fail"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("no-preferred needs managed"),
        "{err}"
    );
}

#[test]
fn sim_cfg_no_mem_prefetch_needs_managed() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let err = match sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            no_mem_prefetch: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    ) {
        Ok(_) => panic!("no-mem-prefetch without managed must fail"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("no-mem-prefetch needs managed"),
        "{err}"
    );
}

#[test]
fn sim_replay_no_mem_prefetch_keeps_hits() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg {
        managed: true,
        ..SimCfg::lru(1, 4096, 0)
    };
    let on = SimCfg {
        managed: true,
        no_mem_prefetch: true,
        ..SimCfg::lru(1, 4096, 0)
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("off");
    let b = sim_replay_cfg(&t, profile, on).expect("on");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn simulated_gpu_store_prefetch_host_needs_managed() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            prefetch_host: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("prefetch-host without managed must fail"),
        Err(err) => assert!(
            err.to_string().contains("prefetch-host needs managed"),
            "{err}"
        ),
    }
}

#[test]
fn sim_replay_prefetch_host_keeps_hits() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg {
        managed: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let on = SimCfg {
        managed: true,
        prefetch_host: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("off");
    let b = sim_replay_cfg(&t, profile, on).expect("on");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
    assert!(
        b.bytes_moved > a.bytes_moved,
        "prefetch-host must add host copies; off={} on={}",
        a.bytes_moved,
        b.bytes_moved
    );
}

#[test]
fn simulated_gpu_store_prefetch_host_pin_before_acquire_still_replicates() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let mut gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        2,
        HardwareProfile::example_8xh100_nvlink(),
        4096,
        GpuFill::Managed,
        GpuStoreCfg {
            prefetch_host: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    let k0 = ExpertKey::new(0, 0);
    gpu.pin_hot(&[k0]).expect("pin");
    assert_eq!(gpu.replica_of(k0), Some(DeviceId(1)));
    assert_eq!(gpu.metrics().replicates, 1);
    let _s = gpu.score().expect("score");
}

#[test]
fn simulated_gpu_store_prefetch_host_evicts_without_free() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let run = |prefetch_host: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = match SimulatedGpuStore::with_cfg(
            inner,
            2,
            p.clone(),
            4096,
            GpuFill::Managed,
            GpuStoreCfg {
                prefetch_host,
                ..GpuStoreCfg::default()
            },
        ) {
            Ok(gpu) => gpu,
            Err(err) => panic!("gpu: {err}"),
        };
        for key in t.keys() {
            let _p = gpu.acquire(key).expect("acquire");
            gpu.release(key);
        }
        let host = count_host_prefetch(&gpu);
        let metrics = gpu.metrics();
        let _s = gpu.score().expect("score");
        (metrics.hits, metrics.misses, host)
    };
    let (off_hits, off_misses, off_host) = run(false);
    let (on_hits, on_misses, on_host) = run(true);
    assert_eq!(off_hits, on_hits);
    assert_eq!(off_misses, on_misses);
    assert_eq!(off_host, 0, "identity managed must not prefetch to host");
    assert!(
        on_host >= 1,
        "prefetch-host must submit Device→Host memcpy; n={on_host}"
    );
}

#[test]
fn sim_cfg_host_register_mapped_needs_mapped() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let err = match sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            host_register_mapped: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    ) {
        Ok(_) => panic!("host-register-mapped without mapped must fail"),
        Err(e) => e,
    };
    assert!(
        err.to_string()
            .contains("host-register-mapped needs mapped"),
        "{err}"
    );
}

#[test]
fn simulated_gpu_store_host_register_mapped_needs_mapped() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            host_register_mapped: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("host-register-mapped without mapped must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("host-register-mapped needs mapped"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_host_register_mapped_refuses_host_register() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Mapped,
        GpuStoreCfg {
            pageable: true,
            host_register: true,
            host_register_mapped: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("host-register + host-register-mapped must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("choose one of host-register, host-register-mapped"),
            "{err}"
        ),
    }
}

#[test]
fn sim_replay_host_register_mapped_keeps_hits() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let host_alloc = SimCfg {
        mapped: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let registered = SimCfg {
        mapped: true,
        host_register_mapped: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let a = sim_replay_cfg(&t, profile.clone(), host_alloc).expect("mapped");
    let b = sim_replay_cfg(&t, profile, registered).expect("register-mapped");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
    assert_eq!(a.hbm_peak, 0);
    assert_eq!(b.hbm_peak, 0);
}

#[test]
fn simulated_gpu_store_host_register_mapped_is_registered_not_host_alloc() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let run = |host_register_mapped: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = match SimulatedGpuStore::with_cfg(
            inner,
            2,
            p.clone(),
            4096,
            GpuFill::Mapped,
            GpuStoreCfg {
                host_register_mapped,
                ..GpuStoreCfg::default()
            },
        ) {
            Ok(gpu) => gpu,
            Err(err) => panic!("gpu: {err}"),
        };
        let k0 = ExpertKey::new(0, 0);
        let _a = gpu.acquire(k0).expect("acq");
        assert!(gpu.uses_mapped());
        assert!(gpu.page_is_host_mapped(k0));
        assert_eq!(gpu.page_is_host_registered(k0), host_register_mapped);
        gpu.release(k0);
        for key in t.keys() {
            let _p = gpu.acquire(key).expect("acquire");
            gpu.release(key);
        }
        let metrics = gpu.metrics();
        let score = gpu.score().expect("score");
        assert_eq!(score.hbm_peak, 0);
        assert_eq!(score.bytes_moved, 0);
        (metrics.hits, metrics.misses)
    };
    let mut identity =
        SimulatedGpuStore::with_mapped(DirectStore::from_trace(&t), 2, p.clone(), 4096)
            .expect("identity mapped");
    let k0 = ExpertKey::new(0, 0);
    let _a = identity.acquire(k0).expect("id acq");
    assert!(identity.page_is_host_mapped(k0));
    assert!(!identity.page_is_host_registered(k0));
    let _s = identity.score().expect("id score");
    let host_alloc = run(false);
    let registered = run(true);
    assert_eq!(host_alloc.0, registered.0);
    assert_eq!(host_alloc.1, registered.1);
}

#[test]
fn sim_cfg_sync_memops_refuses_mapped() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let err = match sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            mapped: true,
            sync_memops: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    ) {
        Ok(_) => panic!("sync-memops + mapped must fail"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("sync-memops needs device memcpy"),
        "{err}"
    );
}

#[test]
fn simulated_gpu_store_sync_memops_refuses_mapped() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Mapped,
        GpuStoreCfg {
            sync_memops: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("sync-memops + mapped must fail"),
        Err(err) => assert!(
            err.to_string().contains("sync-memops needs device memcpy"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_sync_memops_refuses_memcpy_batch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            memcpy_batch: true,
            sync_memops: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("sync-memops + memcpy-batch must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("memcpy-batch needs async pinned/vmm H2D"),
            "{err}"
        ),
    }
}

#[test]
fn sim_replay_sync_memops_keeps_hits() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg::lru(2, 4096, 0);
    let on = SimCfg {
        sync_memops: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("async");
    let b = sim_replay_cfg(&t, profile, on).expect("sync-memops");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn simulated_gpu_store_sync_memops_marks_pointer() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let run = |sync_memops: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = match SimulatedGpuStore::with_cfg(
            inner,
            2,
            p.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                sync_memops,
                ..GpuStoreCfg::default()
            },
        ) {
            Ok(gpu) => gpu,
            Err(err) => panic!("gpu: {err}"),
        };
        let k0 = ExpertKey::new(0, 0);
        let _a = gpu.acquire(k0).expect("acq");
        assert_eq!(gpu.page_sync_memops(k0), sync_memops);
        gpu.release(k0);
        for key in t.keys() {
            let _p = gpu.acquire(key).expect("acquire");
            gpu.release(key);
        }
        let metrics = gpu.metrics();
        let _s = gpu.score().expect("score");
        (metrics.hits, metrics.misses)
    };
    let mut identity =
        SimulatedGpuStore::new(DirectStore::from_trace(&t), 2, p.clone(), 4096).expect("identity");
    let k0 = ExpertKey::new(0, 0);
    let _a = identity.acquire(k0).expect("id acq");
    assert!(!identity.page_sync_memops(k0));
    let _s = identity.score().expect("id score");
    let off = run(false);
    let on = run(true);
    assert_eq!(off.0, on.0);
    assert_eq!(off.1, on.1);
}

#[test]
fn sim_replay_sync_memops_h2d_host_sync() {
    use crate::replay::Touch;
    use crate::sim_replay::{apply_touch, GraphBank, PageHandle, TouchArgs};
    use gpu_sim::{PointerAttr, Sim, StreamId};
    use std::collections::BTreeMap;

    let bytes = 8u64 << 20;
    let run = |sync_memops: bool| {
        let mut sim = Sim::new(HardwareProfile::example_h100_sxm());
        let mut handles: BTreeMap<ExpertKey, PageHandle> = BTreeMap::new();
        let mut graphs = GraphBank::new(false, false, false, crate::sim_replay::LeafMem::None);
        let args = TouchArgs {
            d: DeviceId(0),
            s: StreamId(0),
            bytes,
            slots: 1,
            sync_alloc: false,
            mapped: false,
            managed: false,
            vmm: false,
            vmm_page: 0,
            pageable: false,
            host_register: false,
            host_register_mapped: false,
            sync_memops,
            memcpy_batch: false,
            memcpy_during: false,
            memcpy_any: false,
            memcpy_attr: false,
            memset_fill: false,
            copy_host: false,
            accessed_by: false,
            no_read_mostly: false,
            no_preferred: false,
            no_mem_prefetch: false,
            wait_value: false,
            stream_attach: false,
            managed_host: false,
            prefetch_host: false,
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
        let idle = sim.query_stream(DeviceId(0), StreamId(0)).expect("query");
        let attr = if sync_memops {
            Some(
                sim.pointer_get_attribute(id, PointerAttr::SyncMemops)
                    .expect("attr"),
            )
        } else {
            None
        };
        (attr, idle)
    };
    let (off_attr, off_idle) = run(false);
    let (on_attr, on_idle) = run(true);
    assert_eq!(off_attr, None);
    assert!(
        !off_idle,
        "async pinned H2D must leave the copy stream busy"
    );
    assert_eq!(on_attr, Some(1));
    assert!(
        on_idle,
        "SyncMemops H2D must host-wait the copy stream before return"
    );
}

#[test]
fn sim_cfg_device_sync_memops_refuses_mapped() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let err = match sim_replay_cfg(
        &t,
        HardwareProfile::example_h100_sxm(),
        SimCfg {
            mapped: true,
            device_sync_memops: true,
            ..SimCfg::lru(1, 4096, 0)
        },
    ) {
        Ok(_) => panic!("device-sync-memops + mapped must fail"),
        Err(e) => e,
    };
    assert!(
        err.to_string()
            .contains("device-sync-memops needs device memcpy"),
        "{err}"
    );
}

#[test]
fn simulated_gpu_store_device_sync_memops_refuses_mapped() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Mapped,
        GpuStoreCfg {
            device_sync_memops: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("device-sync-memops + mapped must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("device-sync-memops needs device memcpy"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_device_sync_memops_refuses_memcpy_batch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            device_sync_memops: true,
            memcpy_batch: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("device-sync-memops + memcpy-batch must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("memcpy-batch needs async pinned/vmm H2D"),
            "{err}"
        ),
    }
}

#[test]
fn sim_replay_device_sync_memops_keeps_hits() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg::lru(2, 4096, 0);
    let on = SimCfg {
        device_sync_memops: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("async");
    let b = sim_replay_cfg(&t, profile, on).expect("device-sync-memops");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn simulated_gpu_store_device_sync_memops_sets_flags() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let run = |device_sync_memops: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = match SimulatedGpuStore::with_cfg(
            inner,
            2,
            p.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                device_sync_memops,
                ..GpuStoreCfg::default()
            },
        ) {
            Ok(gpu) => gpu,
            Err(err) => panic!("gpu: {err}"),
        };
        assert_eq!(gpu.device_sync_memops(), device_sync_memops);
        let k0 = ExpertKey::new(0, 0);
        let _a = gpu.acquire(k0).expect("acq");
        assert!(!gpu.page_sync_memops(k0));
        gpu.release(k0);
        for key in t.keys() {
            let _p = gpu.acquire(key).expect("acquire");
            gpu.release(key);
        }
        let metrics = gpu.metrics();
        let _s = gpu.score().expect("score");
        (metrics.hits, metrics.misses)
    };
    let inner = DirectStore::from_trace(&t);
    let mut identity = SimulatedGpuStore::new(inner, 2, p.clone(), 4096).expect("id");
    let k0 = ExpertKey::new(0, 0);
    let _a = identity.acquire(k0).expect("id acq");
    assert!(!identity.device_sync_memops());
    assert!(!identity.page_sync_memops(k0));
    let _s = identity.score().expect("id score");
    let off = run(false);
    let on = run(true);
    assert_eq!(off.0, on.0);
    assert_eq!(off.1, on.1);
}

#[test]
fn sim_replay_device_sync_memops_h2d_host_sync() {
    use crate::replay::Touch;
    use crate::sim_replay::{
        apply_device_sync_memops, apply_touch, GraphBank, PageHandle, TouchArgs,
    };
    use gpu_sim::{DeviceFlags, Sim, StreamId};
    use std::collections::BTreeMap;

    let bytes = 8u64 << 20;
    let run = |device_sync_memops: bool| {
        let mut sim = Sim::new(HardwareProfile::example_h100_sxm());
        apply_device_sync_memops(&mut sim, device_sync_memops).expect("flags");
        let mut handles: BTreeMap<ExpertKey, PageHandle> = BTreeMap::new();
        let mut graphs = GraphBank::new(false, false, false, crate::sim_replay::LeafMem::None);
        let args = TouchArgs {
            d: DeviceId(0),
            s: StreamId(0),
            bytes,
            slots: 1,
            sync_alloc: false,
            mapped: false,
            managed: false,
            vmm: false,
            vmm_page: 0,
            pageable: false,
            host_register: false,
            host_register_mapped: false,
            sync_memops: false,
            memcpy_batch: false,
            memcpy_during: false,
            memcpy_any: false,
            memcpy_attr: false,
            memset_fill: false,
            copy_host: false,
            accessed_by: false,
            no_read_mostly: false,
            no_preferred: false,
            no_mem_prefetch: false,
            wait_value: false,
            stream_attach: false,
            managed_host: false,
            prefetch_host: false,
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
        let idle = sim.query_stream(DeviceId(0), StreamId(0)).expect("query");
        let flags = sim.get_device_flags(DeviceId(0)).expect("get flags");
        (flags & DeviceFlags::SYNC_MEMOPS != 0, idle)
    };
    let (off_flag, off_idle) = run(false);
    let (on_flag, on_idle) = run(true);
    assert!(!off_flag);
    assert!(
        !off_idle,
        "async pinned H2D must leave the copy stream busy"
    );
    assert!(on_flag);
    assert!(
        on_idle,
        "device SyncMemops H2D must host-wait the copy stream before return"
    );
}

#[test]
fn sim_replay_func_max_shared_keeps_hits() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg::lru(2, 4096, 0);
    let on = SimCfg {
        func_max_shared: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("identity");
    let b = sim_replay_cfg(&t, profile, on).expect("func-max-shared");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn sim_replay_max_l1_keeps_hits() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg {
        func_max_shared: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let on = SimCfg {
        max_l1: true,
        ..off
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("func-max-shared");
    let b = sim_replay_cfg(&t, profile, on).expect("max-l1");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn sim_replay_max_l1_needs_func_max_shared() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let on = SimCfg {
        max_l1: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    match sim_replay_cfg(&t, profile, on) {
        Ok(_) => panic!("max-l1 without func-max-shared must fail"),
        Err(err) => assert!(
            err.to_string().contains("max-l1 needs func-max-shared"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_max_l1_needs_func_max_shared() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            max_l1: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("max-l1 without func-max-shared must fail"),
        Err(err) => assert!(
            err.to_string().contains("max-l1 needs func-max-shared"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_max_l1_refuses_max_shared() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            func_max_shared: true,
            max_shared: true,
            max_l1: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("max-l1 with max-shared must fail"),
        Err(err) => assert!(
            err.to_string().contains("choose one of max-l1, max-shared"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_max_l1_marks_launch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            func_max_shared: true,
            max_l1: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    assert!(gpu.max_l1());
    assert!(gpu.func_max_shared());
    let identity = SimulatedGpuStore::new(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
    )
    .expect("id");
    assert!(!identity.max_l1());
}

#[test]
fn simulated_gpu_store_func_max_shared_sets_func_carveout() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let run = |func_max_shared: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            2,
            p.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                func_max_shared,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        assert_eq!(gpu.func_max_shared(), func_max_shared);
        for key in t.keys() {
            let _p = gpu.acquire(key).expect("acquire");
            gpu.release(key);
        }
        let metrics = gpu.metrics();
        let _s = gpu.score().expect("score");
        (metrics.hits, metrics.misses)
    };
    let inner = DirectStore::from_trace(&t);
    let mut identity = SimulatedGpuStore::new(inner, 2, p.clone(), 4096).expect("id");
    let k0 = ExpertKey::new(0, 0);
    let _a = identity.acquire(k0).expect("id acq");
    assert!(!identity.func_max_shared());
    let _s = identity.score().expect("id score");
    let off = run(false);
    let on = run(true);
    assert_eq!(off.0, on.0);
    assert_eq!(off.1, on.1);
}

#[test]
fn sim_replay_func_cluster_spread_keeps_hits() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg {
        cluster: 2,
        ..SimCfg::lru(2, 4096, 0)
    };
    let on = SimCfg {
        cluster: 2,
        func_cluster_spread: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("identity");
    let b = sim_replay_cfg(&t, profile, on).expect("func-cluster-spread");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn sim_replay_cluster_load_balance_keeps_hits() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg {
        cluster: 2,
        func_cluster_spread: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let on = SimCfg {
        cluster_load_balance: true,
        ..off
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("func-cluster-spread");
    let b = sim_replay_cfg(&t, profile, on).expect("cluster-load-balance");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn sim_replay_cluster_load_balance_needs_func_cluster_spread() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let on = SimCfg {
        cluster: 2,
        cluster_load_balance: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    match sim_replay_cfg(&t, profile, on) {
        Ok(_) => panic!("cluster-load-balance without func-cluster-spread must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("cluster-load-balance needs func-cluster-spread"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_cluster_load_balance_needs_func_cluster_spread() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            cluster: 2,
            cluster_load_balance: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("cluster-load-balance without func-cluster-spread must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("cluster-load-balance needs func-cluster-spread"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_cluster_load_balance_refuses_cluster_spread() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            cluster: 2,
            func_cluster_spread: true,
            cluster_spread: true,
            cluster_load_balance: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("cluster-load-balance with cluster-spread must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("choose one of cluster-load-balance, cluster-spread"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_cluster_load_balance_marks_launch() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            cluster: 2,
            func_cluster_spread: true,
            cluster_load_balance: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    assert!(gpu.cluster_load_balance());
    assert!(gpu.func_cluster_spread());
    let identity = SimulatedGpuStore::new(
        DirectStore::from_trace(&t),
        1,
        HardwareProfile::example_h100_sxm(),
        4096,
    )
    .expect("id");
    assert!(!identity.cluster_load_balance());
}

#[test]
fn simulated_gpu_store_func_cluster_spread_sets_func_policy() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let run = |func_cluster_spread: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            2,
            p.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                cluster: 2,
                func_cluster_spread,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        assert_eq!(gpu.func_cluster_spread(), func_cluster_spread);
        for key in t.keys() {
            let _p = gpu.acquire(key).expect("acquire");
            gpu.release(key);
        }
        let metrics = gpu.metrics();
        let _s = gpu.score().expect("score");
        (metrics.hits, metrics.misses)
    };
    let inner = DirectStore::from_trace(&t);
    let mut identity = SimulatedGpuStore::new(inner, 2, p.clone(), 4096).expect("id");
    let k0 = ExpertKey::new(0, 0);
    let _a = identity.acquire(k0).expect("id acq");
    assert!(!identity.func_cluster_spread());
    let _s = identity.score().expect("id score");
    let off = run(false);
    let on = run(true);
    assert_eq!(off.0, on.0);
    assert_eq!(off.1, on.1);
}

#[test]
fn sim_replay_func_cluster_spread_matches_launch_spread() {
    let t = Trace {
        events: vec![ev_seq(0, 0, 0, &[0]), ev_seq(1, 0, 0, &[1])],
    };
    let profile = HardwareProfile::parse("gpus=1\nfp16_flops=1000000\ncopy_engines=2\n")
        .expect("slow gemm profile");
    let base = SimCfg {
        seq_streams: true,
        compute_slots: 4,
        cluster: 2,
        ..SimCfg::lru(2, 4096, 0)
    };
    let launch = SimCfg {
        cluster_spread: true,
        ..base
    };
    let func = SimCfg {
        func_cluster_spread: true,
        ..base
    };
    let both = SimCfg {
        cluster_spread: true,
        func_cluster_spread: true,
        ..base
    };
    let a = sim_replay_cfg(&t, profile.clone(), launch).expect("launch");
    let b = sim_replay_cfg(&t, profile.clone(), func).expect("func");
    let c = sim_replay_cfg(&t, profile, both).expect("both");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
    assert_eq!(b.sim_ns, c.sim_ns);
    assert_eq!(
        a.sim_ns.saturating_add(1),
        b.sim_ns,
        "func SetAttribute is +1 ns; occupancy matches launch Spread; launch={} func={} both={}",
        a.sim_ns,
        b.sim_ns,
        c.sim_ns
    );
}

#[test]
fn sim_replay_cluster_must_set_keeps_hits() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg {
        cluster: 2,
        ..SimCfg::lru(2, 4096, 0)
    };
    let on = SimCfg {
        cluster: 2,
        cluster_must_set: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("identity");
    let b = sim_replay_cfg(&t, profile, on).expect("cluster-must-set");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn simulated_gpu_store_cluster_must_set_sets_func_attr() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let run = |cluster_must_set: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            2,
            p.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                cluster: 2,
                cluster_must_set,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        assert_eq!(gpu.cluster_must_set(), cluster_must_set);
        for key in t.keys() {
            let _p = gpu.acquire(key).expect("acquire");
            gpu.release(key);
        }
        let metrics = gpu.metrics();
        let _s = gpu.score().expect("score");
        (metrics.hits, metrics.misses)
    };
    let inner = DirectStore::from_trace(&t);
    let mut identity = SimulatedGpuStore::new(inner, 2, p.clone(), 4096).expect("id");
    let k0 = ExpertKey::new(0, 0);
    let _a = identity.acquire(k0).expect("id acq");
    assert!(!identity.cluster_must_set());
    let _s = identity.score().expect("id score");
    let off = run(false);
    let on = run(true);
    assert_eq!(off.0, on.0);
    assert_eq!(off.1, on.1);
}

#[test]
fn sim_replay_cluster_must_set_needs_cluster() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let on = SimCfg {
        cluster_must_set: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    match sim_replay_cfg(&t, profile, on) {
        Ok(_) => panic!("cluster-must-set without cluster must fail"),
        Err(err) => assert!(
            err.to_string().contains("cluster-must-set needs cluster"),
            "{err}"
        ),
    }
}

#[test]
fn sim_replay_required_cluster_keeps_hits() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg {
        cluster: 2,
        ..SimCfg::lru(2, 4096, 0)
    };
    let on = SimCfg {
        cluster: 2,
        required_cluster: 2,
        ..SimCfg::lru(2, 4096, 0)
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("identity");
    let b = sim_replay_cfg(&t, profile, on).expect("required-cluster");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn simulated_gpu_store_required_cluster_sets_func_attr() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let run = |required_cluster: u8| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            2,
            p.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                cluster: 2,
                required_cluster,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        assert_eq!(gpu.required_cluster(), required_cluster);
        for key in t.keys() {
            let _p = gpu.acquire(key).expect("acquire");
            gpu.release(key);
        }
        let metrics = gpu.metrics();
        let _s = gpu.score().expect("score");
        (metrics.hits, metrics.misses)
    };
    let inner = DirectStore::from_trace(&t);
    let mut identity = SimulatedGpuStore::new(inner, 2, p.clone(), 4096).expect("id");
    let k0 = ExpertKey::new(0, 0);
    let _a = identity.acquire(k0).expect("id acq");
    assert_eq!(identity.required_cluster(), 0);
    let _s = identity.score().expect("id score");
    let off = run(0);
    let on = run(2);
    assert_eq!(off.0, on.0);
    assert_eq!(off.1, on.1);
}

#[test]
fn sim_replay_required_cluster_needs_cluster() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let on = SimCfg {
        required_cluster: 2,
        ..SimCfg::lru(2, 4096, 0)
    };
    match sim_replay_cfg(&t, profile, on) {
        Ok(_) => panic!("required-cluster without cluster must fail"),
        Err(err) => assert!(
            err.to_string().contains("required-cluster needs cluster"),
            "{err}"
        ),
    }
}

#[test]
fn sim_replay_required_cluster_must_match() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let on = SimCfg {
        cluster: 2,
        required_cluster: 4,
        ..SimCfg::lru(2, 4096, 0)
    };
    match sim_replay_cfg(&t, profile, on) {
        Ok(_) => panic!("required-cluster mismatch must fail"),
        Err(err) => assert!(
            err.to_string()
                .contains("required-cluster must match cluster"),
            "{err}"
        ),
    }
}

#[test]
fn sim_replay_l2_reset_keeps_hits() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg::lru(2, 4096, 0);
    let on = SimCfg {
        l2_reset: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("identity");
    let b = sim_replay_cfg(&t, profile, on).expect("l2-reset");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn sim_replay_l2_fetch_keeps_hits() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg {
        l2_persist: true,
        ..SimCfg::lru(2, 4096, 0)
    };
    let on = SimCfg {
        l2_persist: true,
        l2_fetch: 32,
        ..SimCfg::lru(2, 4096, 0)
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("persist");
    let b = sim_replay_cfg(&t, profile, on).expect("l2-fetch");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
    assert_eq!(
        a.sim_ns.saturating_add(1),
        b.sim_ns,
        "l2-fetch SetLimit is +1 ns; persist={} fetch={}",
        a.sim_ns,
        b.sim_ns
    );
}

#[test]
fn sim_replay_l2_fetch_allows_unaligned_persist() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    let persist = SimCfg {
        l2_persist: true,
        ..SimCfg::lru(2, 64, 0)
    };
    match sim_replay_cfg(&t, profile.clone(), persist) {
        Ok(_) => panic!("64-byte persist must fail default L2 fetch 128"),
        Err(err) => assert!(err.to_string().contains("L2 fetch"), "{err}"),
    }
    let fetch = SimCfg {
        l2_persist: true,
        l2_fetch: 32,
        ..SimCfg::lru(2, 64, 0)
    };
    let row = sim_replay_cfg(&t, profile, fetch).expect("aligned");
    assert_eq!(row.misses, 1);
}

#[test]
fn simulated_gpu_store_l2_fetch_sets_limit() {
    let t = cycling_trace();
    let p = HardwareProfile::example_h100_sxm();
    let run = |l2_fetch: u64| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            2,
            p.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                l2_persist: true,
                l2_fetch,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        assert_eq!(gpu.l2_fetch(), if l2_fetch == 0 { 128 } else { l2_fetch });
        for key in t.keys() {
            let _p = gpu.acquire(key).expect("acquire");
            gpu.release(key);
        }
        let metrics = gpu.metrics();
        let _s = gpu.score().expect("score");
        (metrics.hits, metrics.misses)
    };
    let inner = DirectStore::from_trace(&t);
    let mut identity = SimulatedGpuStore::new(inner, 2, p.clone(), 4096).expect("id");
    let k0 = ExpertKey::new(0, 0);
    let _a = identity.acquire(k0).expect("id acq");
    assert_eq!(identity.l2_fetch(), 128);
    let _s = identity.score().expect("id score");
    let off = run(0);
    let on = run(32);
    assert_eq!(off.0, on.0);
    assert_eq!(off.1, on.1);
}

#[test]
fn simulated_gpu_store_l2_fetch_allows_unaligned_persist() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let p = HardwareProfile::example_h100_sxm();
    let k = ExpertKey::new(0, 0);
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_cfg(
        inner,
        1,
        p.clone(),
        64,
        GpuFill::Pinned,
        GpuStoreCfg {
            l2_persist: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    match gpu.acquire(k) {
        Ok(_) => panic!("64-byte persist must fail default L2 fetch 128"),
        Err(err) => assert!(err.to_string().contains("L2 fetch"), "{err}"),
    }
    let inner = DirectStore::from_trace(&t);
    let mut gpu = SimulatedGpuStore::with_cfg(
        inner,
        1,
        p,
        64,
        GpuFill::Pinned,
        GpuStoreCfg {
            l2_persist: true,
            l2_fetch: 32,
            ..GpuStoreCfg::default()
        },
    )
    .expect("gpu");
    assert_eq!(gpu.l2_fetch(), 32);
    let _p = gpu.acquire(k).expect("aligned");
    gpu.release(k);
    let _s = gpu.score().expect("score");
}

#[test]
fn simulated_gpu_store_l2_fetch_refuses_invalid() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            l2_fetch: 96,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("l2-fetch 96 must fail"),
        Err(err) => assert!(
            err.to_string().contains("l2-fetch must be 32, 64, or 128"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_l2_ratio_partial_is_slower_than_full() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000000000000\nlaunch_overhead_ns=1\nhbm_bps=1000000000\nl2_persist_hit_permille=1000\n",
    )
    .expect("memory-bound persist profile");
    let run = |l2_persist: bool, l2_ratio: u16| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            2,
            p.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                l2_persist,
                l2_ratio,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        assert_eq!(gpu.l2_ratio(), if l2_ratio == 0 { 1000 } else { l2_ratio });
        for key in t.keys() {
            let _p = gpu.acquire(key).expect("acquire");
            gpu.release(key);
        }
        let metrics = gpu.metrics();
        let score = gpu.score().expect("score");
        (metrics.hits, metrics.misses, score.wall_ns)
    };
    let inner = DirectStore::from_trace(&t);
    let mut identity = SimulatedGpuStore::new(inner, 2, p.clone(), 4096).expect("id");
    let k0 = ExpertKey::new(0, 0);
    let _a = identity.acquire(k0).expect("id acq");
    assert_eq!(identity.l2_ratio(), 1000);
    let _s = identity.score().expect("id score");
    let persist = run(true, 0);
    let half = run(false, 500);
    let full = run(true, 1000);
    assert_eq!(persist.0, half.0);
    assert_eq!(persist.1, half.1);
    assert_eq!(persist.0, full.0);
    assert_eq!(persist.1, full.1);
    assert_eq!(
        persist.2, full.2,
        "hitRatio 1000 must match persist; persist={} ratio1000={}",
        persist.2, full.2
    );
    assert!(
        half.2 > persist.2,
        "hitRatio 500 must bill more HBM than full persist; half={} full={}",
        half.2,
        persist.2
    );
}

#[test]
fn simulated_gpu_store_l2_ratio_refuses_over_1000() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            l2_ratio: 1001,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("l2-ratio 1001 must fail"),
        Err(err) => assert!(
            err.to_string().contains("l2-ratio must be 1..=1000"),
            "{err}"
        ),
    }
}

#[test]
fn simulated_gpu_store_l2_streaming_is_slower_than_persist() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000000000000\nlaunch_overhead_ns=1\nhbm_bps=1000000000\nl2_persist_hit_permille=1000\n",
    )
    .expect("memory-bound persist profile");
    let run = |l2_persist: bool, l2_streaming: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            2,
            p.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                l2_persist,
                l2_streaming,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        assert_eq!(gpu.l2_streaming(), l2_streaming);
        for key in t.keys() {
            let _p = gpu.acquire(key).expect("acquire");
            gpu.release(key);
        }
        let metrics = gpu.metrics();
        let score = gpu.score().expect("score");
        (metrics.hits, metrics.misses, score.wall_ns)
    };
    let persist = run(true, false);
    let streaming = run(true, true);
    assert_eq!(persist.0, streaming.0);
    assert_eq!(persist.1, streaming.1);
    assert!(
        streaming.2 > persist.2,
        "streaming hits must bill more HBM than persist; streaming={} persist={}",
        streaming.2,
        persist.2
    );
}

#[test]
fn simulated_gpu_store_l2_streaming_needs_l2_persist() {
    let t = Trace {
        events: vec![ev(0, 0, &[0])],
    };
    let profile = HardwareProfile::example_h100_sxm();
    match SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile.clone(),
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            l2_streaming: true,
            ..GpuStoreCfg::default()
        },
    ) {
        Ok(_) => panic!("l2-streaming without persist must fail"),
        Err(err) => assert!(
            err.to_string().contains("l2-streaming needs l2-persist"),
            "{err}"
        ),
    }
    let mut gpu = SimulatedGpuStore::with_cfg(
        DirectStore::from_trace(&t),
        1,
        profile,
        4096,
        GpuFill::Pinned,
        GpuStoreCfg {
            l2_ratio: 500,
            l2_streaming: true,
            ..GpuStoreCfg::default()
        },
    )
    .expect("ratio arms persist");
    assert!(gpu.l2_streaming());
    let _s = gpu.score().expect("score");
}

#[test]
fn simulated_gpu_store_l2_reset_colds_reused_expert() {
    let t = Trace {
        events: vec![ev(0, 0, &[0]), ev(1, 0, &[0])],
    };
    let p = HardwareProfile::parse(
        "gpus=1\nfp16_flops=1000000000000000\nlaunch_overhead_ns=1\nhbm_bps=1000000000\nl2_persist_hit_permille=1000\n",
    )
    .expect("memory-bound persist profile");
    let run = |l2_persist: bool, l2_reset: bool| {
        let inner = DirectStore::from_trace(&t);
        let mut gpu = SimulatedGpuStore::with_cfg(
            inner,
            2,
            p.clone(),
            4096,
            GpuFill::Pinned,
            GpuStoreCfg {
                l2_persist,
                l2_reset,
                ..GpuStoreCfg::default()
            },
        )
        .expect("gpu");
        assert_eq!(gpu.l2_reset(), l2_reset);
        for key in t.keys() {
            let _p = gpu.acquire(key).expect("acquire");
            gpu.release(key);
        }
        let metrics = gpu.metrics();
        let score = gpu.score().expect("score");
        (metrics.hits, metrics.misses, score.wall_ns)
    };
    let inner = DirectStore::from_trace(&t);
    let mut identity = SimulatedGpuStore::new(inner, 2, p.clone(), 4096).expect("id");
    let k0 = ExpertKey::new(0, 0);
    let _a = identity.acquire(k0).expect("id acq");
    assert!(!identity.l2_reset());
    let _s = identity.score().expect("id score");
    let persist = run(true, false);
    let reset = run(true, true);
    let implied = run(false, true);
    assert_eq!(persist.0, reset.0);
    assert_eq!(persist.1, reset.1);
    assert_eq!(implied.0, reset.0);
    assert_eq!(implied.1, reset.1);
    assert!(
        persist.2 < reset.2,
        "reset must cold reused persist lines; persist={} reset={}",
        persist.2,
        reset.2
    );
    assert_eq!(
        implied.2, reset.2,
        "bare l2_reset must imply persist windows; implied={} reset={}",
        implied.2, reset.2
    );
}

#[test]
fn sim_replay_func_shared_mem_keeps_hits() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg::lru(2, 4096, 0);
    let on = SimCfg {
        func_shared_mem: SharedMemoryMode::EightByte,
        ..SimCfg::lru(2, 4096, 0)
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("identity");
    let b = sim_replay_cfg(&t, profile, on).expect("func-shared-mem");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}

#[test]
fn sim_replay_device_shared_mem_keeps_hits() {
    let t = cycling_trace();
    let profile = HardwareProfile::example_h100_sxm();
    let off = SimCfg::lru(2, 4096, 0);
    let on = SimCfg {
        device_shared_mem: SharedMemoryMode::EightByte,
        ..SimCfg::lru(2, 4096, 0)
    };
    let a = sim_replay_cfg(&t, profile.clone(), off).expect("identity");
    let b = sim_replay_cfg(&t, profile, on).expect("device-shared-mem");
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.misses, b.misses);
}
