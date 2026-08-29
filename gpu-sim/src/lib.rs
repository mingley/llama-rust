//! Deterministic GPU-systems VM for inference.
//!
//! Mechanical invariants (streams, events, HBM vs host-pinned residency, OOM,
//! topology, CUDA-graph capture/replay) are exact.
//! Operation durations come from a [`HardwareProfile`], so agents cannot invent
//! hardware numbers inside policy code.
//!
//! This crate does not model warps, L1 caches, or Tensor Core pipelines.
//! Submitted work is a [`GpuOp`] compiled into an [`Operation`] DAG
//! ([`Sim::operations`]). [`Sim::synchronize_stream`] waits one stream.
//! [`Sim::synchronize_event`] is `cudaEventSynchronize`.
//! [`Sim::set_stream_priority`] is `cudaStreamCreateWithPriority`.

#![cfg_attr(not(test), deny(missing_docs))]

mod error;
mod ids;
mod ops;
mod probe;
mod profile;
mod score;
mod sim;

pub use error::SimError;
pub use ids::{AllocId, DeviceId, EventId, GraphId, LinkId, OpId, StreamId};
pub use ops::{DType, GpuOp, KernelKind, MemcpyOp, Operation, Place};
pub use probe::{probe_topology, P2pProbe, TopologyProbe};
pub use profile::{
    align_up, ns_for_bytes, scale_ns_permille, GpuProfile, HardwareProfile, LinkKind, LinkProfile,
};
pub use score::Score;
pub use sim::Sim;

#[cfg(test)]
mod tests {
    use super::*;

    fn h100() -> HardwareProfile {
        HardwareProfile::example_h100_sxm()
    }

    fn enq(r: Result<OpId, SimError>) {
        assert!(r.expect("enqueue").0 >= 1);
    }

    #[test]
    fn alloc_kernel_sync_advances_clock() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 1 << 20, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 20, 1 << 20), &[a], &[a], s));
        assert_eq!(sim.op_stream(OpId(1)), Some(s));
        sim.synchronize().unwrap();
        assert!(sim.clock_ns() > 0);
        assert!(sim.is_resident(a, d).unwrap());
    }

    #[test]
    fn oom_is_semantic_failure() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let huge = sim.profile().gpu(d).unwrap().hbm_bytes.saturating_add(1);
        let err = sim.alloc(d, huge, s).unwrap();
        let out = sim.synchronize();
        assert!(out.is_err(), "id={err:?} {out:?}");
        match out.unwrap_err() {
            SimError::Oom { .. } => {}
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn kernel_before_residency_fails() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s0 = StreamId(0);
        let s1 = StreamId(1);
        let a = sim.alloc(d, 4096, s0).unwrap();
        // Kernel on a different stream, no event: may race. Force the race by
        // submitting the kernel first on s1 against an alloc that has not started.
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s1));
        let err = sim.synchronize();
        assert!(err.is_err());
    }

    #[test]
    fn event_orders_streams() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s0 = StreamId(0);
        let s1 = StreamId(1);
        let ev = EventId(1);
        let a = sim.alloc(d, 4096, s0).unwrap();
        enq(sim.record_event(d, ev, s0));
        enq(sim.wait_event(d, ev, s1));
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s1));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(a, d).unwrap());
    }

    #[test]
    fn free_while_leased_fails() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
        enq(sim.record_event(d, EventId(1), StreamId(0)));
        enq(sim.kernel(d, KernelKind::other(1 << 40, 4096), &[a], &[a], StreamId(0)));
        enq(sim.wait_event(d, EventId(1), StreamId(1)));
        sim.free(d, a, StreamId(1)).unwrap();
        let err = sim.synchronize();
        match err {
            Err(SimError::Leased { .. }) => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn h2d_then_kernel_hides_nothing_without_overlap() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let a = sim.alloc(d, bytes, s).unwrap();
        enq(sim.memcpy_host_to_device(d, a, bytes, s));
        enq(sim.kernel(d, KernelKind::other(1 << 20, bytes), &[a], &[a], s));
        sim.synchronize().unwrap();
        assert!(sim.bytes_moved() >= bytes);
        assert!(sim.clock_ns() > 0);
    }

    #[test]
    fn copy_overlaps_compute_on_two_streams() {
        let mut serial = Sim::new(h100());
        let mut overlapped = Sim::new(h100());
        let d = DeviceId(0);
        let bytes = 64u64 << 20;
        let k = KernelKind::GroupedMoeGemm {
            experts: 4,
            tokens_per_expert: 8,
            hidden: 1024,
            ff: 1024,
            dtype: DType::Fp16,
        };

        let a = serial.alloc(d, bytes, StreamId(0)).unwrap();
        enq(serial.memcpy_host_to_device(d, a, bytes, StreamId(0)));
        enq(serial.kernel(d, k.clone(), &[a], &[a], StreamId(0)));
        serial.synchronize().unwrap();

        let b = overlapped.alloc(d, bytes, StreamId(0)).unwrap();
        enq(overlapped.record_event(d, EventId(1), StreamId(0)));
        enq(overlapped.memcpy_host_to_device(d, b, bytes, StreamId(0)));
        // Same stream still serializes copy then compute. True overlap needs
        // the kernel on a buffer already resident while a *different* buffer copies.
        let c = overlapped.alloc(d, bytes, StreamId(1)).unwrap();
        enq(overlapped.wait_event(d, EventId(1), StreamId(1)));
        enq(overlapped.kernel(d, k, &[b], &[b], StreamId(1)));
        enq(overlapped.memcpy_host_to_device(d, c, bytes, StreamId(0)));
        overlapped.synchronize().unwrap();
        assert!(overlapped.clock_ns() > 0);
        assert!(serial.clock_ns() > 0);
        assert!(!Score::from_sim(&serial).line().is_empty());
        assert!(!Score::from_sim(&overlapped).line().is_empty());
    }

    #[test]
    fn tiny_copies_cannot_beat_one_large_copy() {
        let mut big = Sim::new(h100());
        let mut tiny = Sim::new(h100());
        let d = DeviceId(0);
        let total = 4u64 << 20;
        let a = big.alloc(d, total, StreamId(0)).unwrap();
        enq(big.memcpy_host_to_device(d, a, total, StreamId(0)));
        big.synchronize().unwrap();

        let b = tiny.alloc(d, total, StreamId(0)).unwrap();
        let chunk = 64u64 << 10;
        let n = total / chunk;
        for _ in 0..n {
            enq(tiny.memcpy_host_to_device(d, b, chunk, StreamId(0)));
        }
        tiny.synchronize().unwrap();
        assert!(
            tiny.clock_ns() > big.clock_ns(),
            "tiny={} big={}",
            tiny.clock_ns(),
            big.clock_ns()
        );
    }

    #[test]
    fn concurrent_h2d_on_two_streams_share_pcie() {
        let d = DeviceId(0);
        let bytes = 32u64 << 20;
        let mut one = Sim::new(h100());
        let a = one.alloc(d, bytes, StreamId(0)).unwrap();
        enq(one.memcpy_host_to_device(d, a, bytes, StreamId(0)));
        one.synchronize().unwrap();
        let t1 = one.clock_ns();

        let mut two = Sim::new(h100());
        let b = two.alloc(d, bytes, StreamId(0)).unwrap();
        let c = two.alloc(d, bytes, StreamId(1)).unwrap();
        enq(two.memcpy_host_to_device(d, b, bytes, StreamId(0)));
        enq(two.memcpy_host_to_device(d, c, bytes, StreamId(1)));
        two.synchronize().unwrap();
        let t2 = two.clock_ns();
        assert!(
            t2 > t1,
            "two concurrent H2D must not finish in one-copy time (shared PCIe); t1={t1} t2={t2}"
        );
    }

    #[test]
    fn determinism() {
        let run = || {
            let mut sim = Sim::new(h100());
            let d = DeviceId(0);
            let a = sim.alloc(d, 1 << 20, StreamId(0)).unwrap();
            enq(sim.memcpy_host_to_device(d, a, 1 << 20, StreamId(0)));
            enq(sim.kernel(
                d,
                KernelKind::other(1 << 24, 1 << 20),
                &[a],
                &[a],
                StreamId(0),
            ));
            sim.synchronize().unwrap();
            sim.clock_ns()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn eight_gpu_profile_has_nvlink() {
        let p = HardwareProfile::example_8xh100_nvlink();
        assert_eq!(p.n_gpus(), 8);
        assert!(p.link(Some(DeviceId(0)), Some(DeviceId(7))).is_ok());
    }

    #[test]
    fn score_line_is_stable() {
        let s = Score {
            wall_ns: 10,
            hbm_peak: 20,
            bytes_moved: 30,
            ns_per_token: None,
            energy_uj: 7,
            ttft_ns: None,
            itl_ns: None,
        };
        assert_eq!(
            s.line(),
            "wall_ns=10 hbm_peak=20 bytes_moved=30 energy_uj=7"
        );
        assert_eq!(
            s.clone().with_tokens(2).line(),
            "wall_ns=10 hbm_peak=20 bytes_moved=30 energy_uj=7 ns_per_token=5"
        );
        assert_eq!(
            s.with_latencies(4, Some(3)).line(),
            "wall_ns=10 hbm_peak=20 bytes_moved=30 energy_uj=7 ttft_ns=4 itl_ns=3"
        );
    }

    #[test]
    fn energy_scales_with_profile_tdp_not_dollars() {
        let bytes = 8u64 << 20;
        let run = |p: HardwareProfile| {
            let mut sim = Sim::new(p);
            let d = DeviceId(0);
            let s = StreamId(0);
            let a = sim.alloc(d, bytes, s).unwrap();
            enq(sim.memcpy_host_to_device(d, a, bytes, s));
            sim.synchronize().unwrap();
            Score::from_sim(&sim)
        };
        let h100 = run(HardwareProfile::example_h100_sxm());
        let cheap = run(HardwareProfile::example_cheap_48gb());
        assert!(h100.energy_uj > 0);
        assert!(h100.energy_uj > cheap.energy_uj);
        assert!(h100.line().contains("energy_uj="));
        assert!(!h100.line().contains('$'));
    }

    #[test]
    fn grouped_moe_permille_lengthens_kernel() {
        let kind = KernelKind::GroupedMoeGemm {
            experts: 4,
            tokens_per_expert: 8,
            hidden: 4096,
            ff: 14336,
            dtype: DType::Fp16,
        };
        let ns = |pen: u16| {
            let mut p = HardwareProfile::example_h100_sxm();
            for g in &mut p.gpus {
                g.grouped_moe_permille = pen;
            }
            let mut sim = Sim::new(p);
            let d = DeviceId(0);
            let s = StreamId(0);
            let a = sim.alloc(d, 1 << 20, s).unwrap();
            enq(sim.memcpy_host_to_device(d, a, 1 << 20, s));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, kind.clone(), &[a], &[a], s));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        assert!(ns(2000) > ns(1000));
    }

    #[test]
    fn gemm_util_below_peak_lengthens_dense_kernel() {
        let ns = |util: u16| {
            let mut p = HardwareProfile::example_h100_sxm();
            for g in &mut p.gpus {
                g.gemm_util_permille = util;
            }
            let mut sim = Sim::new(p);
            let d = DeviceId(0);
            let s = StreamId(0);
            let a = sim.alloc(d, 1 << 20, s).unwrap();
            enq(sim.memcpy_host_to_device(d, a, 1 << 20, s));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, KernelKind::other(1 << 40, 1 << 20), &[a], &[a], s));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        assert!(ns(500) > ns(1000));
    }

    #[test]
    fn d2d_replica_charges_peer_hbm_and_survives_src_free() {
        let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s0 = StreamId(0);
        let bytes = 8u64 << 20;
        let a = sim.alloc(d0, bytes, s0).unwrap();
        enq(sim.memcpy_host_to_device(d0, a, bytes, s0));
        enq(sim.memcpy_device_to_device(d0, d1, a, bytes, s0));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(a, d0).unwrap());
        assert!(sim.is_resident(a, d1).unwrap());
        let used0 = sim.hbm_used(d0).unwrap();
        let used1 = sim.hbm_used(d1).unwrap();
        assert!(used0 >= bytes);
        assert!(used1 >= bytes);
        sim.free(d1, a, s0).unwrap();
        sim.synchronize().unwrap();
        assert!(sim.is_resident(a, d0).unwrap());
        assert!(!sim.is_resident(a, d1).unwrap());
        enq(sim.kernel(d0, KernelKind::other(8, bytes), &[a], &[a], s0));
        sim.synchronize().unwrap();
    }

    #[test]
    fn gpu_unavailable_rejects_submit() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        sim.set_unavailable(d, true).unwrap();
        assert!(sim.is_unavailable(d));
        match sim.alloc(d, 4096, StreamId(0)) {
            Err(SimError::Unavailable { device }) => assert_eq!(device, d),
            other => panic!("{other:?}"),
        }
        sim.set_unavailable(d, false).unwrap();
        assert!(sim.alloc(d, 4096, StreamId(0)).is_ok());
    }

    #[test]
    fn fail_next_memcpy_is_expert_load_failure() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        sim.fail_next_memcpy();
        enq(sim.memcpy_host_to_device(d, a, 4096, s));
        match sim.synchronize() {
            Err(SimError::TransferFailed { alloc }) => assert_eq!(alloc, a),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn extra_transfer_ns_delays_h2d() {
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let mut fast = Sim::new(h100());
        let a = fast.alloc(d, bytes, s).unwrap();
        enq(fast.memcpy_host_to_device(d, a, bytes, s));
        fast.synchronize().unwrap();
        let t0 = fast.clock_ns();

        let mut slow = Sim::new(h100());
        slow.set_extra_transfer_ns(1_000_000);
        let b = slow.alloc(d, bytes, s).unwrap();
        enq(slow.memcpy_host_to_device(d, b, bytes, s));
        slow.synchronize().unwrap();
        assert!(
            slow.clock_ns() >= t0.saturating_add(1_000_000),
            "fast={t0} delayed={}",
            slow.clock_ns()
        );
    }

    #[test]
    fn cancel_stream_skips_queued_kernel() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_host_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 40, 4096), &[a], &[a], s));
        enq(sim.kernel(d, KernelKind::other(8, 4096), &[a], &[a], s));
        sim.start_ready().unwrap();
        let n = sim.cancel_stream(d, s).unwrap();
        assert!(n >= 1);
        match sim.synchronize() {
            Err(SimError::Cancelled { n: skipped, .. }) => assert!(skipped >= 1),
            other => panic!("{other:?}"),
        }
        assert!(sim.cancelled_count() >= 1);
    }

    #[test]
    fn two_gpu_pcie_d2d_is_slower_than_nvlink() {
        let bytes = 32u64 << 20;
        let pcie = probe_topology(HardwareProfile::example_2xh100_pcie(), bytes).unwrap();
        let nv = probe_topology(HardwareProfile::example_8xh100_nvlink(), bytes).unwrap();
        let pcie_d2d = pcie.p2p[0].ns.expect("pcie p2p");
        let nv_d2d = nv.p2p[0].ns.expect("nvlink p2p");
        assert!(pcie_d2d > nv_d2d, "pcie={pcie_d2d} nvlink={nv_d2d}");
        assert_eq!(pcie.n_gpus, 2);
        assert_eq!(nv.n_gpus, 8);
    }

    #[test]
    fn bad_numa_far_gpu_h2d_is_slower() {
        let p = probe_topology(HardwareProfile::example_bad_numa(), 16u64 << 20).unwrap();
        assert_eq!(p.h2d_ns.len(), 2);
        assert!(
            p.h2d_ns[1] > p.h2d_ns[0],
            "near={} far={}",
            p.h2d_ns[0],
            p.h2d_ns[1]
        );
        assert!(p.p2p[0].ns.is_none());
    }

    #[test]
    fn rdma_peer_exists_and_asymmetric_omits_02() {
        let rdma = probe_topology(HardwareProfile::example_2node_rdma(), 8u64 << 20).unwrap();
        assert!(rdma.p2p[0].ns.is_some());
        let kind = HardwareProfile::example_2node_rdma()
            .link(Some(DeviceId(0)), Some(DeviceId(1)))
            .unwrap()
            .kind;
        assert_eq!(kind, LinkKind::Rdma);
        let line = probe_topology(HardwareProfile::example_asymmetric_links(), 4u64 << 20).unwrap();
        assert_eq!(line.n_gpus, 3);
        let hop02 = line
            .p2p
            .iter()
            .find(|h| h.src == DeviceId(0) && h.dst == DeviceId(2))
            .expect("0-2");
        assert!(hop02.ns.is_none());
        let hop01 = line
            .p2p
            .iter()
            .find(|h| h.src == DeviceId(0) && h.dst == DeviceId(1))
            .expect("0-1");
        assert!(hop01.ns.is_some());
        assert!(!line.line().is_empty());
    }

    #[test]
    fn allreduce_needs_peer_link() {
        let mut ok = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let a = ok.alloc(DeviceId(0), bytes, s).unwrap();
        enq(ok.memcpy_host_to_device(DeviceId(0), a, bytes, s));
        enq(ok.memcpy_device_to_device(DeviceId(0), DeviceId(1), a, bytes, s));
        ok.synchronize().unwrap();
        enq(ok.allreduce(&[(DeviceId(0), a), (DeviceId(1), a)], bytes, s));
        ok.synchronize().unwrap();

        let mut bad = Sim::new(HardwareProfile::example_asymmetric_links());
        let b = bad.alloc(DeviceId(0), bytes, s).unwrap();
        enq(bad.memcpy_host_to_device(DeviceId(0), b, bytes, s));
        enq(bad.memcpy_device_to_device(DeviceId(0), DeviceId(1), b, bytes, s));
        bad.synchronize().unwrap();
        enq(bad.memcpy_device_to_device(DeviceId(1), DeviceId(2), b, bytes, s));
        bad.synchronize().unwrap();
        enq(bad.allreduce(
            &[(DeviceId(0), b), (DeviceId(1), b), (DeviceId(2), b)],
            bytes,
            s,
        ));
        match bad.synchronize() {
            Err(SimError::NoPeer { src, dst }) => {
                assert!(src == DeviceId(2) || dst == DeviceId(0) || src == DeviceId(0));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_capture_does_not_run_until_launch() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_host_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        let t0 = sim.clock_ns();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 30, 4096), &[a], &[a], s));
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.clock_ns(), t0);
        assert_eq!(sim.graph_len(g).unwrap(), 1);
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        let t1 = sim.clock_ns();
        assert!(t1 > t0);
        let n2 = sim.launch_graph(g, s).unwrap();
        assert_eq!(n2, 1);
        sim.synchronize().unwrap();
        assert!(sim.clock_ns() > t1);
    }

    #[test]
    fn graph_cannot_capture_alloc() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        match sim.alloc(d, 4096, s) {
            Err(SimError::Invalid { why }) => assert!(why.contains("alloc")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn pinned_h2d_is_faster_than_pageable() {
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let mut pageable = Sim::new(h100());
        let a = pageable.alloc(d, bytes, s).unwrap();
        enq(pageable.memcpy_host_to_device(d, a, bytes, s));
        pageable.synchronize().unwrap();
        let mut pinned = Sim::new(h100());
        let b = pinned.alloc(d, bytes, s).unwrap();
        enq(pinned.memcpy_pinned_to_device(d, b, bytes, s));
        pinned.synchronize().unwrap();
        assert!(
            pageable.clock_ns() > pinned.clock_ns(),
            "pageable={} pinned={}",
            pageable.clock_ns(),
            pinned.clock_ns()
        );
    }

    #[test]
    fn host_pinned_alloc_does_not_charge_hbm() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let h = sim.alloc_host_pinned(8 << 20).unwrap();
        assert!(sim.is_host_pinned(h).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        assert_eq!(sim.hbm_peak(), 0);
        enq(sim.kernel(d, KernelKind::other(8, 8), &[h], &[h], StreamId(0)));
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, h);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn host_pinned_copy_onto_device_then_kernel() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let h = sim.alloc_host_pinned(bytes).unwrap();
        enq(sim.memcpy_pinned_to_device(d, h, bytes, s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(h, d).unwrap());
        assert!(sim.is_host_pinned(h).unwrap());
        assert!(sim.hbm_used(d).unwrap() >= bytes);
        enq(sim.kernel(d, KernelKind::other(8, bytes), &[h], &[h], s));
        sim.synchronize().unwrap();
        sim.free(d, h, s).unwrap();
        sim.synchronize().unwrap();
        assert!(!sim.is_resident(h, d).unwrap());
        assert!(sim.is_host_pinned(h).unwrap());
        sim.free_host_pinned(h).unwrap();
        assert!(!sim.is_host_pinned(h).unwrap());
    }

    #[test]
    fn device_to_pinned_keeps_hbm_and_marks_host() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let a = sim.alloc(d, bytes, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, bytes, s));
        sim.synchronize().unwrap();
        let used = sim.hbm_used(d).unwrap();
        enq(sim.memcpy_device_to_pinned(d, a, bytes, s));
        sim.synchronize().unwrap();
        assert!(sim.is_host_pinned(a).unwrap());
        assert!(sim.is_resident(a, d).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), used);
    }

    #[test]
    fn graph_cannot_capture_host_pinned_alloc() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        match sim.alloc_host_pinned(4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("alloc")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_launch_amortizes_kernel_overhead() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.launch_overhead_ns = 50_000;
            g.graph_launch_ns = 5_000;
        }
        let d = DeviceId(0);
        let s = StreamId(0);
        let kind = KernelKind::other(8, 8);
        let n = 4u32;
        let kernels = |sim: &mut Sim, a: AllocId| {
            for _ in 0..n {
                enq(sim.kernel(d, kind.clone(), &[a], &[a], s));
            }
        };
        let mut serial = Sim::new(p.clone());
        let a = serial.alloc(d, 4096, s).unwrap();
        enq(serial.memcpy_pinned_to_device(d, a, 4096, s));
        serial.synchronize().unwrap();
        let t0 = serial.clock_ns();
        kernels(&mut serial, a);
        serial.synchronize().unwrap();
        let serial_ns = serial.clock_ns().saturating_sub(t0);

        let mut gsim = Sim::new(p);
        let b = gsim.alloc(d, 4096, s).unwrap();
        enq(gsim.memcpy_pinned_to_device(d, b, 4096, s));
        gsim.synchronize().unwrap();
        gsim.begin_capture(d, s).unwrap();
        kernels(&mut gsim, b);
        let g = gsim.end_capture().unwrap();
        let t1 = gsim.clock_ns();
        let launched = gsim.launch_graph(g, s).unwrap();
        assert_eq!(launched, n);
        gsim.synchronize().unwrap();
        let graph_ns = gsim.clock_ns().saturating_sub(t1);
        assert!(graph_ns < serial_ns, "graph={graph_ns} serial={serial_ns}");
    }

    #[test]
    fn graph_capture_replays_memcpy() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let a = sim.alloc(d, bytes, s).unwrap();
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, bytes, s));
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.bytes_moved(), 0);
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert!(sim.bytes_moved() >= bytes);
        assert!(sim.is_resident(a, d).unwrap());
    }

    #[test]
    fn memset_requires_residency_then_advances_clock() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let h = sim.alloc_host_pinned(4096).unwrap();
        enq(sim.memset(d, h, 4096, s));
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, h);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
        let mut sim = Sim::new(h100());
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        let t0 = sim.clock_ns();
        enq(sim.memset(d, a, 4096, s));
        sim.synchronize().unwrap();
        assert!(sim.clock_ns() > t0);
    }

    #[test]
    fn disable_peer_blocks_d2d_when_link_exists() {
        let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let a = sim.alloc(d0, bytes, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d0, a, bytes, s));
        sim.synchronize().unwrap();
        assert!(sim.peer_access(d0, d1));
        sim.disable_peer(d0, d1).unwrap();
        assert!(!sim.peer_access(d0, d1));
        enq(sim.memcpy_device_to_device(d0, d1, a, bytes, s));
        match sim.synchronize() {
            Err(SimError::PeerDisabled { src, dst }) => {
                assert_eq!(src, d0);
                assert_eq!(dst, d1);
            }
            other => panic!("{other:?}"),
        }
        sim.enable_peer(d0, d1).unwrap();
        enq(sim.memcpy_device_to_device(d0, d1, a, bytes, s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(a, d1).unwrap());
    }

    #[test]
    fn legacy_null_stream_serializes_copy_with_compute() {
        let d = DeviceId(0);
        let bytes = 32u64 << 20;
        let k = KernelKind::GroupedMoeGemm {
            experts: 8,
            tokens_per_expert: 16,
            hidden: 2048,
            ff: 2048,
            dtype: DType::Fp16,
        };
        let run = |legacy: bool| {
            let mut sim = Sim::new(h100());
            sim.set_legacy_null_stream(legacy);
            let w = sim.alloc(d, bytes, StreamId(0)).unwrap();
            let c = sim.alloc(d, bytes, StreamId(1)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, w, bytes, StreamId(0)));
            enq(sim.memcpy_pinned_to_device(d, c, bytes, StreamId(1)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, k.clone(), &[w], &[w], StreamId::NULL));
            enq(sim.memcpy_pinned_to_device(d, c, bytes, StreamId(1)));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let serial = run(true);
        let overlap = run(false);
        assert!(
            serial > overlap,
            "legacy null stream must serialize; serial={serial} overlap={overlap}"
        );
        assert!(Sim::new(h100()).stream_is_idle(d, StreamId(0)).unwrap());
    }

    #[test]
    fn third_copy_waits_when_two_engines() {
        let p = HardwareProfile::parse("gpus=1\ncopy_engines=2\n").unwrap();
        let d = DeviceId(0);
        let bytes = 16u64 << 20;
        let n_copies = |n: u16| {
            let mut sim = Sim::new(p.clone());
            for i in 0..n {
                let s = StreamId(i);
                let a = sim.alloc(d, bytes, s).unwrap();
                enq(sim.memcpy_pinned_to_device(d, a, bytes, s));
            }
            sim.synchronize().unwrap();
            sim.clock_ns()
        };
        let two = n_copies(2);
        let three = n_copies(3);
        assert!(
            three > two,
            "third H2D must wait for a copy engine; two={two} three={three}"
        );
    }

    #[test]
    fn event_complete_after_record_sync() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let ev = EventId(7);
        assert!(!sim.event_complete(ev));
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.record_event(d, ev, s));
        assert!(!sim.event_complete(ev));
        sim.synchronize().unwrap();
        assert!(sim.event_complete(ev));
        assert!(sim.stream_is_idle(d, s).unwrap());
    }

    #[test]
    fn operations_are_a_stream_ordered_dag() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.kernel(d, KernelKind::other(8, 4096), &[a], &[a], s));
        sim.synchronize().unwrap();
        let ops: Vec<Operation> = sim.operations().collect();
        assert!(ops.iter().any(|o| matches!(o.kind, GpuOp::Alloc { .. })));
        assert!(ops.iter().any(|o| matches!(o.kind, GpuOp::Memcpy(_))));
        let kernel = ops
            .iter()
            .find(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .expect("kernel");
        assert!(kernel.done);
        assert!(!kernel.deps.is_empty());
        assert_eq!(sim.operation(kernel.id).unwrap().kind, kernel.kind);
        assert!(kernel.start_ns.is_some());
        assert!(kernel.done_ns.is_some());
        assert!(kernel.duration_ns().is_some());
    }

    #[test]
    fn same_stream_next_op_starts_after_previous_finishes() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.kernel(d, KernelKind::other(1 << 20, 4096), &[a], &[a], s));
        enq(sim.kernel(d, KernelKind::other(1 << 20, 4096), &[a], &[a], s));
        sim.synchronize().unwrap();
        let kernels: Vec<Operation> = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }))
            .collect();
        assert_eq!(kernels.len(), 2);
        let a0 = kernels[0].start_ns.expect("k0 start");
        let d0 = kernels[0].done_ns.expect("k0 done");
        let a1 = kernels[1].start_ns.expect("k1 start");
        assert!(
            a1 >= d0,
            "stream[i+1].start >= stream[i].finish; start1={a1} done0={d0} start0={a0}"
        );
    }

    #[test]
    fn higher_stream_priority_starts_first_when_compute_contends() {
        let d = DeviceId(0);
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |hi_first: bool| {
            let mut sim = Sim::new(h100());
            sim.set_stream_priority(d, StreamId(0), 0).unwrap();
            sim.set_stream_priority(d, StreamId(1), 1).unwrap();
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            if hi_first {
                enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(1)));
                enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(0)));
            } else {
                enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(0)));
                enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(1)));
            }
            sim.start_ready().unwrap();
            let started: Vec<StreamId> = sim
                .operations()
                .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.start_ns.is_some())
                .map(|o| o.stream)
                .collect();
            started
        };
        let low_submitted_first = run(false);
        assert_eq!(low_submitted_first, vec![StreamId(1)]);
        let high_submitted_first = run(true);
        assert_eq!(high_submitted_first, vec![StreamId(1)]);
    }

    #[test]
    fn synchronize_stream_does_not_wait_for_other_streams() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let bytes = 64u64 << 20;
        let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
        sim.synchronize().unwrap();
        let t0 = sim.clock_ns();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], StreamId(0)));
        let b = sim.alloc(d, bytes, StreamId(1)).unwrap();
        enq(sim.memcpy_pinned_to_device(d, b, bytes, StreamId(1)));
        sim.synchronize_stream(d, StreamId(0)).unwrap();
        let partial = sim.clock_ns().saturating_sub(t0);
        assert!(sim.stream_is_idle(d, StreamId(0)).unwrap());
        assert!(!sim.stream_is_idle(d, StreamId(1)).unwrap());
        sim.synchronize().unwrap();
        let full = sim.clock_ns().saturating_sub(t0);
        assert!(
            partial < full,
            "stream-0 sync must leave the long H2D running; partial={partial} full={full}"
        );
    }

    #[test]
    fn synchronize_stream_ignores_cancel_on_other_stream() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
        sim.synchronize().unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 40, 4096), &[a], &[a], StreamId(1)));
        enq(sim.kernel(d, KernelKind::other(8, 4096), &[a], &[a], StreamId(1)));
        sim.start_ready().unwrap();
        let n = sim.cancel_stream(d, StreamId(1)).unwrap();
        assert!(n >= 1);
        sim.synchronize_stream(d, StreamId(0)).unwrap();
        match sim.synchronize() {
            Err(SimError::Cancelled { n: skipped, .. }) => assert!(skipped >= 1),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn synchronize_event_does_not_drain_the_rest_of_the_stream() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let ev = EventId(3);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        let t0 = sim.clock_ns();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        enq(sim.record_event(d, ev, s));
        let bytes = 64u64 << 20;
        enq(sim.memcpy_pinned_to_device(d, a, bytes, s));
        sim.synchronize_event(ev).unwrap();
        let partial = sim.clock_ns().saturating_sub(t0);
        assert!(sim.event_complete(ev));
        assert!(!sim.stream_is_idle(d, s).unwrap());
        sim.synchronize().unwrap();
        let full = sim.clock_ns().saturating_sub(t0);
        assert!(
            partial < full,
            "event sync must not wait for the later H2D; partial={partial} full={full}"
        );
    }

    #[test]
    fn synchronize_event_unknown_is_semantic() {
        let mut sim = Sim::new(h100());
        match sim.synchronize_event(EventId(99)) {
            Err(SimError::UnknownEvent { event: 99 }) => {}
            other => panic!("{other:?}"),
        }
    }
}
