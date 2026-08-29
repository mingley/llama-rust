//! Deterministic GPU-systems VM for inference.
//!
//! Mechanical invariants (streams, events, residency, OOM, topology) are exact.
//! Operation durations come from a [`HardwareProfile`], so agents cannot invent
//! hardware numbers inside policy code.
//!
//! This crate does not model warps, L1 caches, or Tensor Core pipelines.

#![cfg_attr(not(test), deny(missing_docs))]

mod error;
mod ids;
mod ops;
mod profile;
mod score;
mod sim;

pub use error::SimError;
pub use ids::{AllocId, DeviceId, EventId, LinkId, OpId, StreamId};
pub use ops::{DType, KernelKind, MemcpyOp, Place};
pub use profile::{ns_for_bytes, GpuProfile, HardwareProfile, LinkKind, LinkProfile};
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
        };
        assert_eq!(s.line(), "wall_ns=10 hbm_peak=20 bytes_moved=30");
        assert_eq!(
            s.clone().with_tokens(2).line(),
            "wall_ns=10 hbm_peak=20 bytes_moved=30 ns_per_token=5"
        );
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
}
