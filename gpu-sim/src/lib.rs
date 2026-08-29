//! Deterministic GPU-systems VM for inference.
//!
//! Mechanical invariants (streams, events, HBM vs host-pinned residency, OOM,
//! topology, CUDA-graph capture / instantiate / update / replay) are exact.
//! Operation durations come from a [`HardwareProfile`], so agents cannot invent
//! hardware numbers inside policy code.
//!
//! This crate does not model warps, L1 caches, or Tensor Core pipelines.
//! Submitted work is a [`GpuOp`] compiled into an [`Operation`] DAG
//! ([`Sim::operations`]). [`Sim::synchronize_stream`] waits one stream
//! (`cudaStreamSynchronize`); an already-idle stream returns without starting
//! leftover kernels on other streams. [`GpuProfile::compute_slots`] is Hyper-Q
//! occupancy (`1` exclusive; `>=2` concurrent kernels at full issue rate).
//! [`Sim::set_stream_sm_permille`] is a green-context SM fraction (compute-bound
//! kernels scale; memory-bound keep full HBM). Default unset is a full chip.
//! [`Sim::synchronize_device`] is `cudaDeviceSynchronize` (one GPU).
//! [`Sim::synchronize_event`] is `cudaEventSynchronize`.
//! [`Sim::alloc`] / [`memcpy`](Sim::memcpy) / [`free`](Sim::free) are
//! stream-ordered (`cudaMallocAsync` / `cudaMemcpyAsync` / `cudaFreeAsync`)
//! except pageable [`Place::Host`] copies, which wait the stream (CUDA bounce
//! buffer). [`Sim::memcpy_pinned_to_device`] is the overlapping DMA path.
//! [`Sim::malloc`] / [`memcpy_sync`](Sim::memcpy_sync) / [`free_sync`](Sim::free_sync)
//! are host-synchronous (`cudaMalloc` / `cudaMemcpy` / `cudaFree`).
//! [`Sim::alloc`] draws from the device default mempool (`cudaMallocAsync`).
//! [`Sim::create_pool`] / [`alloc_from_pool`](Sim::alloc_from_pool) /
//! [`set_pool_release_threshold`](Sim::set_pool_release_threshold) /
//! [`pool_trim_to`](Sim::pool_trim_to) are `cudaMemPoolCreate` /
//! `cudaMallocFromPoolAsync` / `cudaMemPoolAttrReleaseThreshold` /
//! `cudaMemPoolTrimTo`. Unused pool bytes stay in `cudaMemGetInfo` used until
//! trim when the release threshold is high (`u64::MAX`, vLLM-style).
//! [`Sim::pool_set_access`] / [`pool_unset_access`](Sim::pool_unset_access) are
//! `cudaMemPoolSetAccess` ReadWrite / ProtNone: a kernel on a peer may read
//! **and write** pool allocations without dest HBM (interconnect). Applies to
//! existing and later allocs from that pool. [`Sim::malloc`] cannot consume another pool's cache.
//! [`Sim::alloc_host`] is pageable; [`Sim::host_register`] / [`host_register_mapped`](Sim::host_register_mapped)
//! are `cudaHostRegister` (host-synchronous mlock). [`Sim::alloc_host_mapped`] is
//! `cudaHostAllocMapped`: a kernel may read it without H2D, billed at host PCIe.
//! [`Sim::alloc_managed`] is `cudaMallocManaged` (no HBM until first-touch or
//! [`Sim::prefetch`] / [`prefetch_host`](Sim::prefetch_host)). Prefetch migrates;
//! it does not replicate unless [`Sim::mem_advise`] [`MemAdvise::SetReadMostly`].
//! [`MemAdvise::SetAccessedBy`] maps a GPU so a kernel can read without
//! migrating (billed on the interconnect, not local HBM). A kernel
//! page-faults managed memory onto that GPU when the kernel *starts*
//! (after stream deps) unless AccessedBy covers it.
//! [`MemAdvise::SetPreferredLocation`] keeps a page already at that GPU
//! there on a remote read (same interconnect billing; writes still migrate).
//! Host preferred does not skip a kernel first-touch.
//! [`Sim::drop_managed_copy`] refunds one ReadMostly GPU copy (dest eviction).
//! [`Sim::va_reserve`] / [`va_map`](Sim::va_map) / [`va_unmap`](Sim::va_unmap) /
//! [`va_free`](Sim::va_free) are `cuMemAddressReserve` / `cuMemMap` /
//! `cuMemUnmap` / `cuMemAddressFree`. Size and map offsets must be multiples of
//! [`HardwareProfile::va_granularity_bytes`] (`0`/`1` = any size; example
//! default `1` keeps 4096-byte expert pages legal). [`Sim::va_map_range`] / [`va_unmap_range`](Sim::va_unmap_range)
//! map sparse physicals (vLLM KV-block analog); HBM is the mapped span.
//! [`Sim::va_create`] is `cuMemCreate` (charges HBM). [`Sim::va_map_handle`] is
//! `cuMemMap` of that handle (no second HBM charge; two VAs may share it).
//! [`Sim::va_retain_handle`] is `cuMemRetainAllocationHandle` (handle refs;
//! combined `va_map` spans are promoted). [`Sim::va_release_handle`] is
//! `cuMemRelease` (allowed while mapped; HBM refunds when refs and maps are 0).
//! [`Sim::va_map`] still Create+Maps in one call.
//! [`Sim::va_set_access`] is `cuMemSetAccess` PROT_READ on a peer (no dest HBM;
//! interconnect). Writes still need a local map. [`Sim::va_unset_access`] drops it.
//! [`Sim::va_acquire`] remaps an idle VA of the same size (or reserves);
//! [`va_acquire_paged`](Sim::va_acquire_paged) maps it in KV-block spans;
//! [`va_release`](Sim::va_release) unmaps into that pool instead of freeing the VA.
//! [`Sim::kernel`] still needs the whole VA mapped; [`Sim::kernel_bufs`],
//! [`Sim::memset_buf`], and [`MemcpyOp::offset`] touch a mapped span (paged KV).
//! [`Sim::is_range_resident`] is that span check.
//! [`Sim::host_func`] is `cudaLaunchHostFunc` (stream-ordered host work; no GPU occupancy).
//! [`Sim::set_stream_blocking`] is `cudaStreamCreate` vs `cudaStreamNonBlocking`.
//! [`HardwareProfile::host_pin_bytes`] caps `cudaMallocHost` / `cudaHostRegister`.
//! [`Sim::idle_until`] drains, then jumps the virtual clock (open-loop arrivals).
//! [`Sim::event_elapsed_ns`] is `cudaEventElapsedTime` in nanoseconds.
//! [`Sim::create_event_disable_timing`] is `cudaEventDisableTiming` (elapsed fails).
//! [`Sim::query_event`] is `cudaEventQuery` (no wait).
//! [`Sim::query_stream`] is `cudaStreamQuery` (no wait).
//! [`Sim::mem_info`] is `cudaMemGetInfo` `(free, total)`.
//! [`Sim::set_stream_priority`] is `cudaStreamCreateWithPriority`.
//! [`Sim::set_created_streams_priority`] assigns created streams their id.
//! [`Sim::instantiate_graph`] is `cudaGraphInstantiate` (host-sync; first
//! [`launch_graph`](Sim::launch_graph) calls it). [`Sim::instantiate_graph_auto_free`]
//! is `cudaGraphInstantiateFlagAutoFreeOnLaunch`. [`Sim::upload_graph`] is
//! `cudaGraphUpload` (host-sync; first launch after instantiate calls it).
//! [`Sim::update_graph`] is
//! `cudaGraphExecUpdate` when device, stream, and op kinds match.
//! [`Sim::clone_graph`] is `cudaGraphClone` (independent, not instantiated;
//! child-graph nodes are cloned recursively).
//! [`Sim::destroy_graph`] is `cudaGraphDestroy` / `cudaGraphExecDestroy`.
//! Capture records every stream that [`wait_event`](Sim::wait_event)s an
//! event recorded in this capture (CUDA forked capture). Independent streams
//! stay live. Launch remaps origin-stream nodes onto the launch stream.
//! [`launch_graph`](Sim::launch_graph) during capture records a child graph
//! ([`GpuOp::ChildGraph`]) if the child is already instantiated.
//! [`Sim::alloc`] / [`free`](Sim::free) during capture are graph mem alloc/free
//! nodes (`cudaMallocAsync` / `cudaFreeAsync`). Host-sync [`malloc`](Sim::malloc)
//! / [`free_sync`](Sim::free_sync) / VMM / [`create_pool`](Sim::create_pool) cannot
//! be captured. A graph that allocates without a matching free reuses the pointer
//! on later launches (no second HBM charge) unless instantiated with
//! [`instantiate_graph_auto_free`](Sim::instantiate_graph_auto_free)
//! (`cudaGraphInstantiateFlagAutoFreeOnLaunch`). [`clone_graph`](Sim::clone_graph)
//! forks those ids. [`destroy_graph`](Sim::destroy_graph) refunds remaining graph
//! mem. [`update_graph`](Sim::update_graph) of mem nodes is Invalid.

#![cfg_attr(not(test), deny(missing_docs))]

mod error;
mod ids;
mod ops;
mod probe;
mod profile;
mod score;
mod sim;

pub use error::SimError;
pub use ids::{AllocId, DeviceId, EventId, GraphId, LinkId, MemHandleId, OpId, PoolId, StreamId};
pub use ops::{DType, GpuOp, KernelBuf, KernelKind, MemAdvise, MemcpyOp, Operation, Place};
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
        enq(serial.memcpy_pinned_to_device(d, a, bytes, StreamId(0)));
        enq(serial.kernel(d, k.clone(), &[a], &[a], StreamId(0)));
        serial.synchronize().unwrap();

        let b = overlapped.alloc(d, bytes, StreamId(0)).unwrap();
        enq(overlapped.record_event(d, EventId(1), StreamId(0)));
        enq(overlapped.memcpy_pinned_to_device(d, b, bytes, StreamId(0)));
        // Same stream still serializes copy then compute. True overlap needs
        // the kernel on a buffer already resident while a *different* buffer copies.
        let c = overlapped.alloc(d, bytes, StreamId(1)).unwrap();
        enq(overlapped.wait_event(d, EventId(1), StreamId(1)));
        enq(overlapped.kernel(d, k, &[b], &[b], StreamId(1)));
        enq(overlapped.memcpy_pinned_to_device(d, c, bytes, StreamId(0)));
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
        enq(one.memcpy_pinned_to_device(d, a, bytes, StreamId(0)));
        one.synchronize().unwrap();
        let t1 = one.clock_ns();

        let mut two = Sim::new(h100());
        let b = two.alloc(d, bytes, StreamId(0)).unwrap();
        let c = two.alloc(d, bytes, StreamId(1)).unwrap();
        enq(two.memcpy_pinned_to_device(d, b, bytes, StreamId(0)));
        enq(two.memcpy_pinned_to_device(d, c, bytes, StreamId(1)));
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
        match sim.memcpy_host_to_device(d, a, 4096, s) {
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
    fn graph_cannot_capture_sync_malloc() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        match sim.malloc(d, 4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        let a = sim.alloc(d, 4096, s).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 1);
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(a, d).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
    }

    #[test]
    fn graph_captures_malloc_async_kernel_and_free() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        let a = sim.alloc(d, 4096, s).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        sim.free(d, a, s).unwrap();
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 3);
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 3);
        sim.synchronize().unwrap();
        assert!(!sim.is_resident(a, d).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        let n2 = sim.launch_graph(g, s).unwrap();
        assert_eq!(n2, 3);
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        assert_eq!(sim.hbm_peak(), 4096);
    }

    #[test]
    fn graph_mem_alloc_reuses_hbm_on_relaunch() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let g = sim.end_capture().unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(a, d).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        let n2 = sim.launch_graph(g, s).unwrap();
        assert_eq!(n2, 2);
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        assert_eq!(sim.hbm_peak(), 4096);
    }

    #[test]
    fn graph_auto_free_on_launch_recharges_hbm() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.alloc_overhead_ns = 50_000;
            g.pool_reuse_ns = 1;
            g.graph_launch_ns = 1;
            g.graph_instantiate_ns = 1;
            g.graph_upload_ns = 1;
        }
        let d = DeviceId(0);
        let s = StreamId(0);
        let mut reuse = Sim::new(p.clone());
        reuse.begin_capture(d, s).unwrap();
        let a = reuse.alloc(d, 4096, s).unwrap();
        enq(reuse.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let g = reuse.end_capture().unwrap();
        reuse.instantiate_graph(g).unwrap();
        let n = reuse.launch_graph(g, s).unwrap();
        assert_eq!(n, 2);
        reuse.synchronize().unwrap();
        let t_reuse0 = reuse.clock_ns();
        let n2 = reuse.launch_graph(g, s).unwrap();
        assert_eq!(n2, 2);
        reuse.synchronize().unwrap();
        let reuse_ns = reuse.clock_ns().saturating_sub(t_reuse0);

        let mut af = Sim::new(p);
        af.begin_capture(d, s).unwrap();
        let b = af.alloc(d, 4096, s).unwrap();
        enq(af.kernel(d, KernelKind::other(8, 8), &[b], &[b], s));
        let h = af.end_capture().unwrap();
        af.instantiate_graph_auto_free(h).unwrap();
        assert!(af.graph_auto_free_on_launch(h).unwrap());
        let n = af.launch_graph(h, s).unwrap();
        assert_eq!(n, 2);
        af.synchronize().unwrap();
        assert_eq!(af.hbm_used(d).unwrap(), 4096);
        let t0 = af.clock_ns();
        let n2 = af.launch_graph(h, s).unwrap();
        assert_eq!(n2, 2);
        af.synchronize().unwrap();
        let auto_ns = af.clock_ns().saturating_sub(t0);
        assert_eq!(af.hbm_used(d).unwrap(), 4096);
        assert_eq!(af.hbm_peak(), 4096);
        assert!(
            auto_ns > reuse_ns,
            "auto-free relaunch must re-alloc, auto={auto_ns} reuse={reuse_ns}"
        );
        af.destroy_graph(h).unwrap();
        assert_eq!(af.hbm_used(d).unwrap(), 0);
    }

    #[test]
    fn graph_auto_free_rejects_free_nodes_and_flag_change() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        sim.free(d, a, s).unwrap();
        let with_free = sim.end_capture().unwrap();
        let err = sim.instantiate_graph_auto_free(with_free).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("mem free"), "{why}"),
            other => panic!("{other:?}"),
        }

        sim.begin_capture(d, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[b], &[b], s));
        let g = sim.end_capture().unwrap();
        sim.instantiate_graph(g).unwrap();
        assert!(!sim.graph_auto_free_on_launch(g).unwrap());
        let err = sim.instantiate_graph_auto_free(g).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("flags"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_clone_mem_alloc_is_independent_hbm() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let src = sim.end_capture().unwrap();
        let clone = sim.clone_graph(src).unwrap();
        let n = sim.launch_graph(src, s).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        let n2 = sim.launch_graph(clone, s).unwrap();
        assert_eq!(n2, 2);
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 8192);
        sim.destroy_graph(src).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        let n3 = sim.launch_graph(clone, s).unwrap();
        assert_eq!(n3, 2);
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
    }

    #[test]
    fn graph_destroy_refunds_mem_alloc() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let g = sim.end_capture().unwrap();
        sim.destroy_graph(g).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);

        sim.begin_capture(d, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[b], &[b], s));
        let live = sim.end_capture().unwrap();
        let n = sim.launch_graph(live, s).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        sim.destroy_graph(live).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        assert!(!sim.is_resident(b, d).unwrap());
    }

    #[test]
    fn update_graph_rejects_mem_nodes() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let exec = sim.end_capture().unwrap();
        sim.instantiate_graph(exec).unwrap();
        sim.begin_capture(d, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[b], &[b], s));
        let src = sim.end_capture().unwrap();
        let err = sim.update_graph(exec, src).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("mem"), "{why}"),
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
    fn first_graph_launch_instantiates_once() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_instantiate_ns = 25_000;
            g.graph_launch_ns = 1_000;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let g = sim.end_capture().unwrap();
        assert!(!sim.graph_instantiated(g).unwrap());
        let t0 = sim.clock_ns();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        assert!(sim.graph_instantiated(g).unwrap());
        assert!(sim.clock_ns() >= t0.saturating_add(25_000));
        sim.synchronize().unwrap();
        let first = sim.clock_ns().saturating_sub(t0);
        let t1 = sim.clock_ns();
        let n2 = sim.launch_graph(g, s).unwrap();
        assert_eq!(n2, 1);
        assert_eq!(sim.clock_ns(), t1);
        sim.synchronize().unwrap();
        let second = sim.clock_ns().saturating_sub(t1);
        assert!(second < first, "first={first} second={second}");
    }

    #[test]
    fn instantiate_graph_is_host_sync_and_idempotent() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_instantiate_ns = 40_000;
            g.graph_upload_ns = 12_000;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let g = sim.end_capture().unwrap();
        let t0 = sim.clock_ns();
        sim.instantiate_graph(g).unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(40_000));
        assert!(sim.graph_instantiated(g).unwrap());
        assert!(!sim.graph_uploaded(g).unwrap());
        sim.instantiate_graph(g).unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(40_000));
        sim.upload_graph(g).unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(52_000));
        assert!(sim.graph_uploaded(g).unwrap());
        sim.upload_graph(g).unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(52_000));
        let t1 = sim.clock_ns();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        assert_eq!(sim.clock_ns(), t1);
        sim.synchronize().unwrap();
    }

    #[test]
    fn upload_graph_rejects_uninstantiated() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let g = sim.end_capture().unwrap();
        let err = sim.upload_graph(g).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.instantiate_graph(g).unwrap();
        sim.upload_graph(g).unwrap();
        assert!(sim.graph_uploaded(g).unwrap());
    }

    #[test]
    fn first_graph_launch_uploads_after_instantiate() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_instantiate_ns = 25_000;
            g.graph_upload_ns = 9_000;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let g = sim.end_capture().unwrap();
        sim.instantiate_graph(g).unwrap();
        let t0 = sim.clock_ns();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        assert_eq!(sim.clock_ns(), t0.saturating_add(9_000));
        assert!(sim.graph_uploaded(g).unwrap());
        sim.synchronize().unwrap();
        let t1 = sim.clock_ns();
        let n2 = sim.launch_graph(g, s).unwrap();
        assert_eq!(n2, 1);
        assert_eq!(sim.clock_ns(), t1);
    }

    #[test]
    fn update_graph_swaps_allocs_when_topology_matches() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_update_ns = 7_000;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let exec = sim.end_capture().unwrap();
        sim.instantiate_graph(exec).unwrap();
        sim.upload_graph(exec).unwrap();
        assert!(sim.graph_uploaded(exec).unwrap());
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[b], &[b], s));
        let src = sim.end_capture().unwrap();
        let t0 = sim.clock_ns();
        sim.update_graph(exec, src).unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(7_000));
        assert!(!sim.graph_uploaded(exec).unwrap());
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(exec, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(b, d).unwrap());
    }

    #[test]
    fn update_graph_rejects_topology_and_uninstantiated() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let exec = sim.end_capture().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        let memcpy_src = sim.end_capture().unwrap();
        let err = sim.update_graph(exec, memcpy_src).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.instantiate_graph(exec).unwrap();
        let err = sim.update_graph(exec, memcpy_src).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("topology"), "{why}"),
            other => panic!("{other:?}"),
        }
        let err = sim.update_graph(exec, exec).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("same id"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn instantiate_and_update_cannot_run_during_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let exec = sim.end_capture().unwrap();
        sim.instantiate_graph(exec).unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let src = sim.end_capture().unwrap();
        sim.begin_capture(d, s).unwrap();
        let inst = sim.instantiate_graph(exec).unwrap_err();
        match inst {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let upd = sim.update_graph(exec, src).unwrap_err();
        match upd {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let cl = sim.clone_graph(exec).unwrap_err();
        match cl {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let up = sim.upload_graph(exec).unwrap_err();
        match up {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn clone_graph_is_independent_and_not_instantiated() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_clone_ns = 9_000;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let src = sim.end_capture().unwrap();
        sim.instantiate_graph(src).unwrap();
        let t0 = sim.clock_ns();
        let clone = sim.clone_graph(src).unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(9_000));
        assert_ne!(clone, src);
        assert_eq!(sim.graph_len(clone).unwrap(), 1);
        assert!(!sim.graph_instantiated(clone).unwrap());
        assert!(sim.graph_instantiated(src).unwrap());
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[b], &[b], s));
        let alt = sim.end_capture().unwrap();
        sim.update_graph(src, alt).unwrap();
        sim.free_sync(a).unwrap();
        let n = sim.launch_graph(src, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert!(!sim.graph_instantiated(clone).unwrap());
        let n2 = sim.launch_graph(clone, s).unwrap();
        assert_eq!(n2, 1);
        assert!(sim.graph_instantiated(clone).unwrap());
        let err = sim.synchronize().unwrap_err();
        match err {
            SimError::NotResident { alloc, device } => {
                assert_eq!(alloc, a);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn destroy_graph_forbids_later_launch_and_spares_clones() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let g = sim.end_capture().unwrap();
        let clone = sim.clone_graph(g).unwrap();
        sim.destroy_graph(g).unwrap();
        let err = sim.launch_graph(g, s).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("unknown graph"), "{why}"),
            other => panic!("{other:?}"),
        }
        let n = sim.launch_graph(clone, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        let cap = sim.destroy_graph(clone).unwrap_err();
        match cap {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _end = sim.end_capture().unwrap();
        sim.destroy_graph(clone).unwrap();
        let gone = sim.graph_len(clone).unwrap_err();
        match gone {
            SimError::Invalid { why } => assert!(why.contains("unknown graph"), "{why}"),
            other => panic!("{other:?}"),
        }
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
    fn created_stream_priority_helper_matches_manual() {
        let mut sim = Sim::new(h100());
        sim.set_created_streams_priority(2).unwrap();
        assert_eq!(sim.stream_priority(DeviceId(0), StreamId(0)), 0);
        assert_eq!(sim.stream_priority(DeviceId(0), StreamId(1)), 1);
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
    fn synchronize_idle_stream_does_not_start_other_kernels() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
        sim.synchronize().unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 40, 4096), &[a], &[a], StreamId(1)));
        sim.synchronize_stream(d, StreamId(0)).unwrap();
        assert!(sim.stream_is_idle(d, StreamId(0)).unwrap());
        let started = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.start_ns.is_some())
            .count();
        assert_eq!(
            started, 0,
            "idle stream-0 sync must not start the leftover kernel"
        );
        assert!(!sim.stream_is_idle(d, StreamId(1)).unwrap());
        sim.synchronize().unwrap();
        assert!(sim.stream_is_idle(d, StreamId(1)).unwrap());
    }

    #[test]
    fn two_compute_slots_overlap_kernels_on_two_streams() {
        let kind = KernelKind::other(1 << 40, 4096);
        let run = |slots: u8| {
            let mut sim = Sim::new(h100().with_compute_slots(slots));
            let d = DeviceId(0);
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(1)));
            enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(2)));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let serial = run(1);
        let overlap = run(2);
        assert!(
            overlap < serial,
            "two Hyper-Q slots must overlap independent kernels; overlap={overlap} serial={serial}"
        );
        let same_stream = {
            let mut sim = Sim::new(h100().with_compute_slots(2));
            let d = DeviceId(0);
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(1)));
            enq(sim.kernel(d, kind, &[a], &[a], StreamId(1)));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        assert!(
            overlap < same_stream,
            "Hyper-Q must not overlap kernels on one stream; overlap={overlap} same={same_stream}"
        );
    }

    #[test]
    fn two_compute_slots_start_both_ready_kernels() {
        let d = DeviceId(0);
        let kind = KernelKind::other(1 << 40, 4096);
        let mut sim = Sim::new(h100().with_compute_slots(2));
        let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
        sim.synchronize().unwrap();
        enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(1)));
        enq(sim.kernel(d, kind, &[a], &[a], StreamId(2)));
        sim.start_ready().unwrap();
        let started: Vec<StreamId> = sim
            .operations()
            .filter(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.start_ns.is_some())
            .map(|o| o.stream)
            .collect();
        assert_eq!(
            started.len(),
            2,
            "two Hyper-Q slots must start both ready kernels; started={started:?}"
        );
    }

    #[test]
    fn stream_sm_permille_slows_compute_bound_kernel() {
        let d = DeviceId(0);
        let kind = KernelKind::other(1 << 40, 8);
        let run = |permille: u16| {
            let mut sim = Sim::new(h100());
            sim.set_stream_sm_permille(d, StreamId(1), permille)
                .unwrap();
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, kind.clone(), &[a], &[a], StreamId(1)));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let full = run(1000);
        let quarter = run(250);
        assert!(
            quarter > full,
            "250‰ SMs must slow a compute-bound kernel; quarter={quarter} full={full}"
        );
        let err = Sim::new(h100())
            .set_stream_sm_permille(d, StreamId(1), 0)
            .unwrap_err();
        assert!(
            format!("{err:?}").contains("sm permille must be 1..=1000"),
            "{err:?}"
        );
        let err = Sim::new(h100())
            .set_stream_sm_permille(d, StreamId(1), 1001)
            .unwrap_err();
        assert!(
            format!("{err:?}").contains("sm permille must be 1..=1000"),
            "{err:?}"
        );
        let mem = |permille: u16| {
            let mut sim = Sim::new(h100());
            sim.set_stream_sm_permille(d, StreamId(1), permille)
                .unwrap();
            let a = sim.alloc(d, 4096, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, 4096, StreamId(0)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, KernelKind::other(8, 1 << 40), &[a], &[a], StreamId(1)));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let mem_full = mem(1000);
        let mem_quarter = mem(250);
        assert_eq!(
            mem_quarter, mem_full,
            "250‰ SMs must keep full HBM for a memory-bound kernel; quarter={mem_quarter} full={mem_full}"
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

    #[test]
    fn idle_until_jumps_clock_after_drain() {
        let mut sim = Sim::new(h100());
        let jumped = sim.idle_until(1_000_000).unwrap();
        assert_eq!(jumped, 1_000_000);
        assert_eq!(sim.clock_ns(), 1_000_000);
        let again = sim.idle_until(500_000).unwrap();
        assert_eq!(again, 0);
        assert_eq!(sim.clock_ns(), 1_000_000);
    }

    #[test]
    fn idle_until_drains_in_flight_work() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let k = sim
            .kernel(d, KernelKind::other(8, 8), &[a], &[a], s)
            .unwrap();
        assert!(!sim.operation(k).unwrap().done);
        let jumped = sim.idle_until(0).unwrap();
        assert_eq!(jumped, 0);
        assert!(sim.operation(k).unwrap().done);
        assert!(sim.clock_ns() > 0);
    }

    #[test]
    fn event_elapsed_ns_is_record_done_delta() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let start = EventId(1);
        let end = EventId(2);
        enq(sim.record_event(d, start, s));
        enq(sim.kernel(d, KernelKind::other(1 << 20, 4096), &[a], &[a], s));
        enq(sim.record_event(d, end, s));
        match sim.event_elapsed_ns(start, end) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not complete"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.synchronize().unwrap();
        let ns = sim.event_elapsed_ns(start, end).unwrap();
        assert!(ns > 0, "elapsed={ns}");
        match sim.event_elapsed_ns(end, start) {
            Err(SimError::Invalid { why }) => assert!(why.contains("end before start"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.event_elapsed_ns(EventId(99), end) {
            Err(SimError::UnknownEvent { event: 99 }) => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn query_event_does_not_wait() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let ev = EventId(4);
        match sim.query_event(ev) {
            Err(SimError::UnknownEvent { event: 4 }) => {}
            other => panic!("{other:?}"),
        }
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 20, 4096), &[a], &[a], s));
        enq(sim.record_event(d, ev, s));
        assert!(!sim.query_event(ev).unwrap());
        sim.synchronize().unwrap();
        assert!(sim.query_event(ev).unwrap());
    }

    #[test]
    fn query_stream_does_not_wait() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        match sim.query_stream(DeviceId(99), s) {
            Err(SimError::Invalid { why }) => assert!(why.contains("device"), "{why}"),
            other => panic!("{other:?}"),
        }
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 20, 4096), &[a], &[a], s));
        let t0 = sim.clock_ns();
        assert!(!sim.query_stream(d, s).unwrap());
        assert_eq!(sim.clock_ns(), t0);
        sim.synchronize().unwrap();
        assert!(sim.query_stream(d, s).unwrap());
    }

    #[test]
    fn mem_info_tracks_free_and_total() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let (free0, total) = sim.mem_info(d).unwrap();
        assert_eq!(total, sim.profile().gpu(d).unwrap().hbm_bytes);
        assert_eq!(free0, total);
        let bytes = 1u64 << 20;
        let a = sim.alloc(d, bytes, s).unwrap();
        sim.synchronize().unwrap();
        let (free1, total1) = sim.mem_info(d).unwrap();
        assert_eq!(total1, total);
        assert_eq!(free1, total.saturating_sub(sim.hbm_used(d).unwrap()));
        assert!(free1 < free0);
        sim.free(d, a, s).unwrap();
        sim.synchronize().unwrap();
        let (free2, _) = sim.mem_info(d).unwrap();
        assert_eq!(free2, total);
    }

    #[test]
    fn malloc_is_resident_without_stream_sync() {
        let mut queued = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let a = queued.alloc(d, bytes, s).unwrap();
        assert!(!queued.is_resident(a, d).unwrap());
        let (free0, total) = queued.mem_info(d).unwrap();
        assert_eq!(free0, total);

        let mut sim = Sim::new(h100());
        let b = sim.malloc(d, bytes).unwrap();
        assert!(sim.is_resident(b, d).unwrap());
        let (free, total) = sim.mem_info(d).unwrap();
        assert_eq!(free, total.saturating_sub(bytes));
        enq(sim.kernel(d, KernelKind::other(1 << 20, bytes), &[b], &[b], s));
        sim.synchronize_stream(d, s).unwrap();
        assert!(sim.is_resident(b, d).unwrap());
    }

    #[test]
    fn malloc_waits_for_other_streams_alloc_async_does_not() {
        let mut async_sim = Sim::new(h100());
        let d = DeviceId(0);
        let s0 = StreamId(0);
        let s1 = StreamId(1);
        let a = async_sim.alloc(d, 4096, s0).unwrap();
        async_sim.synchronize_stream(d, s0).unwrap();
        enq(async_sim.kernel(d, KernelKind::other(1 << 30, 4096), &[a], &[a], s0));
        let t0 = async_sim.clock_ns();
        let b = async_sim.alloc(d, 4096, s1).unwrap();
        async_sim.synchronize_stream(d, s1).unwrap();
        let async_ns = async_sim.clock_ns().saturating_sub(t0);
        assert!(async_sim.is_resident(b, d).unwrap());

        let mut sync_sim = Sim::new(h100());
        let c = sync_sim.alloc(d, 4096, s0).unwrap();
        sync_sim.synchronize_stream(d, s0).unwrap();
        enq(sync_sim.kernel(d, KernelKind::other(1 << 30, 4096), &[c], &[c], s0));
        let t1 = sync_sim.clock_ns();
        let e = sync_sim.malloc(d, 4096).unwrap();
        let malloc_ns = sync_sim.clock_ns().saturating_sub(t1);
        assert!(sync_sim.is_resident(e, d).unwrap());
        assert!(
            malloc_ns > async_ns,
            "cudaMalloc must wait for the other stream; malloc={malloc_ns} async={async_ns}"
        );
    }

    #[test]
    fn malloc_oom_is_returned_at_the_call() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let cap = sim.profile().gpu(d).unwrap().hbm_bytes;
        let _full = sim.malloc(d, cap).unwrap();
        let err = sim.malloc(d, 1).unwrap_err();
        match err {
            SimError::Oom { need: 1, .. } => {}
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn free_sync_waits_then_releases_hbm() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let a = sim.malloc(d, bytes).unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 20, bytes), &[a], &[a], s));
        sim.free_sync(a).unwrap();
        assert!(!sim.is_resident(a, d).unwrap());
        let (free, total) = sim.mem_info(d).unwrap();
        assert_eq!(free, total);
        let b = sim.malloc(d, total).unwrap();
        assert!(sim.is_resident(b, d).unwrap());
    }

    #[test]
    fn memcpy_sync_leaves_the_stream_idle() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 4096u64;
        let a = sim.malloc(d, bytes).unwrap();
        let op = sim
            .memcpy_sync(
                d,
                MemcpyOp {
                    src: Place::HostPinned,
                    dst: Place::Device(d),
                    alloc: a,
                    bytes,
                    offset: 0,
                },
                s,
            )
            .unwrap();
        assert!(op.0 >= 1);
        assert!(sim.query_stream(d, s).unwrap());
        assert!(sim.is_resident(a, d).unwrap());
    }

    #[test]
    fn malloc_does_not_wait_peer_gpu() {
        let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let a1 = sim.malloc(d1, 4096).unwrap();
        enq(sim.kernel(d1, KernelKind::other(1 << 30, 4096), &[a1], &[a1], s));
        let t0 = sim.clock_ns();
        let a0 = sim.malloc(d0, 4096).unwrap();
        let malloc_ns = sim.clock_ns().saturating_sub(t0);
        assert!(sim.is_resident(a0, d0).unwrap());
        assert!(
            malloc_ns < 10_000,
            "cudaMalloc on GPU0 must not drain GPU1; ns={malloc_ns}"
        );
        assert!(!sim.query_stream(d1, s).unwrap());
        sim.synchronize_device(d1).unwrap();
        assert!(sim.query_stream(d1, s).unwrap());
        assert!(sim.clock_ns().saturating_sub(t0) > malloc_ns);
    }

    #[test]
    fn synchronize_device_waits_every_stream_on_one_gpu() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s0 = StreamId(0);
        let s1 = StreamId(1);
        let a = sim.malloc(d, 4096).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 20, 4096), &[a], &[a], s0));
        enq(sim.kernel(d, KernelKind::other(1 << 20, 4096), &[b], &[b], s1));
        sim.synchronize_device(d).unwrap();
        assert!(sim.query_stream(d, s0).unwrap());
        assert!(sim.query_stream(d, s1).unwrap());
    }

    #[test]
    fn host_sync_memory_cannot_be_captured() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        sim.begin_capture(d, s).unwrap();
        match sim.malloc(d, 4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.free_sync(a) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.synchronize_device(d) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.memcpy_sync(
            d,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d),
                alloc: a,
                bytes: 4096,
                offset: 0,
            },
            s,
        ) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.memcpy_host_to_device(d, a, 4096, s) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.create_pool(d) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        let pool = sim.default_pool(d).unwrap();
        match sim.set_pool_release_threshold(pool, u64::MAX) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.pool_trim_to(pool, 0) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.pool_set_access(pool, d) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.alloc_host(4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.host_register(a) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.alloc_managed(4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.va_reserve(4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
        match sim.va_create(d, 4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn pageable_h2d_is_host_synchronous_pinned_is_not() {
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let mut pinned = Sim::new(h100());
        let a = pinned.alloc(d, bytes, s).unwrap();
        pinned.synchronize_stream(d, s).unwrap();
        let t0 = pinned.clock_ns();
        enq(pinned.memcpy_pinned_to_device(d, a, bytes, s));
        assert!(!pinned.query_stream(d, s).unwrap());
        assert_eq!(pinned.clock_ns(), t0);
        pinned.synchronize_stream(d, s).unwrap();
        assert!(pinned.is_resident(a, d).unwrap());

        let mut pageable = Sim::new(h100());
        let b = pageable.alloc(d, bytes, s).unwrap();
        pageable.synchronize_stream(d, s).unwrap();
        let t1 = pageable.clock_ns();
        enq(pageable.memcpy_host_to_device(d, b, bytes, s));
        assert!(pageable.query_stream(d, s).unwrap());
        assert!(pageable.clock_ns() > t1);
        assert!(pageable.is_resident(b, d).unwrap());
    }

    #[test]
    fn pageable_h2d_cannot_share_pcie_the_way_pinned_can() {
        let d = DeviceId(0);
        let bytes = 32u64 << 20;
        let mut pinned = Sim::new(h100());
        let a = pinned.alloc(d, bytes, StreamId(0)).unwrap();
        let b = pinned.alloc(d, bytes, StreamId(1)).unwrap();
        enq(pinned.memcpy_pinned_to_device(d, a, bytes, StreamId(0)));
        enq(pinned.memcpy_pinned_to_device(d, b, bytes, StreamId(1)));
        pinned.synchronize().unwrap();
        let pin_ns = pinned.clock_ns();

        let mut pageable = Sim::new(h100());
        let c = pageable.alloc(d, bytes, StreamId(0)).unwrap();
        let e = pageable.alloc(d, bytes, StreamId(1)).unwrap();
        enq(pageable.memcpy_host_to_device(d, c, bytes, StreamId(0)));
        enq(pageable.memcpy_host_to_device(d, e, bytes, StreamId(1)));
        pageable.synchronize().unwrap();
        let page_ns = pageable.clock_ns();
        assert!(
            page_ns > pin_ns,
            "pageable cudaMemcpyAsync is host-sync so two copies cannot DMA together; pageable={page_ns} pinned={pin_ns}"
        );
    }

    #[test]
    fn pageable_d2h_is_host_synchronous() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 4096u64;
        let a = sim.malloc(d, bytes).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, bytes, s));
        sim.synchronize_stream(d, s).unwrap();
        let t0 = sim.clock_ns();
        enq(sim.memcpy_device_to_host(d, a, bytes, s));
        assert!(sim.query_stream(d, s).unwrap());
        assert!(sim.clock_ns() > t0);
        assert!(sim.is_resident(a, d).unwrap());
    }

    #[test]
    fn free_sync_rejects_host_pin() {
        let mut sim = Sim::new(h100());
        let h = sim.alloc_host_pinned(4096).unwrap();
        match sim.free_sync(h) {
            Err(SimError::UnknownAlloc { alloc }) => assert_eq!(alloc, h),
            other => panic!("{other:?}"),
        }
        sim.free_host_pinned(h).unwrap();
    }

    #[test]
    fn default_pool_threshold_zero_releases_hbm() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let p = sim.default_pool(d).unwrap();
        let a = sim.alloc(d, 1 << 20, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.pool_live(p).unwrap(), 1 << 20);
        sim.free(d, a, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        assert_eq!(sim.pool_cached(p).unwrap(), 0);
        assert_eq!(sim.pool_live(p).unwrap(), 0);
    }

    #[test]
    fn high_release_threshold_holds_hbm_until_trim() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let p = sim.default_pool(d).unwrap();
        sim.set_pool_release_threshold(p, u64::MAX).unwrap();
        let bytes = 1u64 << 20;
        let a = sim.alloc(d, bytes, s).unwrap();
        sim.synchronize().unwrap();
        let used = sim.hbm_used(d).unwrap();
        sim.free(d, a, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), used);
        assert_eq!(sim.pool_cached(p).unwrap(), bytes);
        let b = sim.alloc(d, bytes, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), used);
        assert_eq!(sim.pool_cached(p).unwrap(), 0);
        sim.free(d, b, s).unwrap();
        sim.synchronize().unwrap();
        let dropped = sim.pool_trim_to(p, 0).unwrap();
        assert_eq!(dropped, bytes);
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        assert_eq!(sim.pool_cached(p).unwrap(), 0);
    }

    #[test]
    fn malloc_cannot_consume_pool_cache_until_trim() {
        let mut sim = Sim::new(HardwareProfile::parse("gpus=1\nhbm_bytes=1048576\n").unwrap());
        let d = DeviceId(0);
        let s = StreamId(0);
        let p = sim.default_pool(d).unwrap();
        sim.set_pool_release_threshold(p, u64::MAX).unwrap();
        let a = sim.alloc(d, 1 << 20, s).unwrap();
        sim.synchronize().unwrap();
        sim.free(d, a, s).unwrap();
        sim.synchronize().unwrap();
        match sim.malloc(d, 4096) {
            Err(SimError::Oom { .. }) => {}
            other => panic!("{other:?}"),
        }
        let _n = sim.pool_trim_to(p, 0).unwrap();
        let b = sim.malloc(d, 4096).unwrap();
        assert!(sim.is_resident(b, d).unwrap());
    }

    #[test]
    fn two_pools_do_not_share_cache() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let p1 = sim.create_pool(d).unwrap();
        let p2 = sim.create_pool(d).unwrap();
        sim.set_pool_release_threshold(p1, u64::MAX).unwrap();
        sim.set_pool_release_threshold(p2, u64::MAX).unwrap();
        let a = sim.alloc_from_pool(d, p1, 4096, s).unwrap();
        sim.synchronize().unwrap();
        sim.free(d, a, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.pool_cached(p1).unwrap(), 4096);
        assert_eq!(sim.pool_cached(p2).unwrap(), 0);
        let b = sim.alloc_from_pool(d, p2, 4096, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.pool_cached(p1).unwrap(), 4096);
        assert_eq!(sim.pool_cached(p2).unwrap(), 0);
        assert!(sim.is_resident(b, d).unwrap());
    }

    #[test]
    fn pool_reuse_is_cheaper_than_os_alloc() {
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let run = |hold: bool| {
            let mut sim = Sim::new(h100());
            if hold {
                sim.set_default_pool_release_threshold(u64::MAX).unwrap();
            }
            let a = sim.alloc(d, bytes, s).unwrap();
            sim.synchronize().unwrap();
            sim.free(d, a, s).unwrap();
            let b = sim.alloc(d, bytes, s).unwrap();
            sim.synchronize().unwrap();
            assert!(sim.is_resident(b, d).unwrap());
            sim.clock_ns()
        };
        assert!(
            run(true) < run(false),
            "cached cudaMallocFromPoolAsync must beat first-touch; hold={} release={}",
            run(true),
            run(false)
        );
    }

    #[test]
    fn alloc_from_pool_rejects_wrong_device() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let p0 = sim.default_pool(DeviceId(0)).unwrap();
        match sim.alloc_from_pool(DeviceId(1), p0, 4096, StreamId(0)) {
            Err(SimError::Invalid { why }) => assert!(why.contains("mismatch")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn replica_free_does_not_fill_dest_pool() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let p1 = sim.default_pool(d1).unwrap();
        sim.set_pool_release_threshold(p1, u64::MAX).unwrap();
        let a = sim.alloc(d0, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d0, a, 4096, s));
        enq(sim.memcpy_device_to_device(d0, d1, a, 4096, s));
        sim.synchronize().unwrap();
        sim.free(d1, a, s).unwrap();
        sim.synchronize().unwrap();
        assert_eq!(sim.pool_cached(p1).unwrap(), 0);
        assert_eq!(sim.hbm_used(d1).unwrap(), 0);
    }

    #[test]
    fn host_register_pins_pageable_for_dma() {
        let mut sim = Sim::new(h100());
        let bytes = 8u64 << 20;
        let h = sim.alloc_host(bytes).unwrap();
        assert!(sim.is_host_pageable(h).unwrap());
        assert!(!sim.is_host_pinned(h).unwrap());
        sim.host_register(h).unwrap();
        assert!(sim.is_host_pinned(h).unwrap());
        assert!(!sim.is_host_pageable(h).unwrap());
        let err = sim.free_host(h).unwrap_err();
        match err {
            SimError::UnknownAlloc { alloc } => assert_eq!(alloc, h),
            other => panic!("{other:?}"),
        }
        let err = sim.free_host_pinned(h).unwrap_err();
        match err {
            SimError::UnknownAlloc { alloc } => assert_eq!(alloc, h),
            other => panic!("{other:?}"),
        }
        sim.host_unregister(h).unwrap();
        sim.free_host(h).unwrap();
    }

    #[test]
    fn pageable_host_kernel_is_not_resident() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let h = sim.alloc_host(4096).unwrap();
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
    fn host_register_rejects_already_pinned() {
        let mut sim = Sim::new(h100());
        let h = sim.alloc_host_pinned(4096).unwrap();
        match sim.host_register(h) {
            Err(SimError::Invalid { why }) => assert!(why.contains("pageable")),
            other => panic!("{other:?}"),
        }
        match sim.host_unregister(h) {
            Err(SimError::Invalid { why }) => assert!(why.contains("registered")),
            other => panic!("{other:?}"),
        }
        sim.free_host_pinned(h).unwrap();
    }

    #[test]
    fn mapped_host_kernel_skips_h2d_and_hbm() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let h = sim.alloc_host_mapped(bytes).unwrap();
        assert!(sim.is_host_mapped(h).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        enq(sim.kernel(d, KernelKind::other(1 << 20, bytes), &[h], &[h], s));
        sim.synchronize().unwrap();
        assert!(sim.clock_ns() > 0);
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        assert!(!sim.is_resident(h, d).unwrap());
        sim.free_host_pinned(h).unwrap();
    }

    #[test]
    fn mapped_host_kernel_is_slower_than_hbm_resident() {
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 32u64 << 20;
        let mut mapped = Sim::new(h100());
        let h = mapped.alloc_host_mapped(bytes).unwrap();
        enq(mapped.kernel(d, KernelKind::other(1 << 20, bytes), &[h], &[h], s));
        mapped.synchronize().unwrap();
        let map_ns = mapped.clock_ns();

        let mut hbm = Sim::new(h100());
        let a = hbm.malloc(d, bytes).unwrap();
        enq(hbm.kernel(d, KernelKind::other(1 << 20, bytes), &[a], &[a], s));
        hbm.synchronize().unwrap();
        let hbm_ns = hbm.clock_ns();
        assert!(
            map_ns > hbm_ns,
            "mapped zero-copy is PCIe; HBM-resident is faster; mapped={map_ns} hbm={hbm_ns}"
        );
    }

    #[test]
    fn host_register_mapped_then_kernel() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let h = sim.alloc_host(4096).unwrap();
        sim.host_register_mapped(h).unwrap();
        assert!(sim.is_host_mapped(h).unwrap());
        enq(sim.kernel(d, KernelKind::other(8, 4096), &[h], &[h], s));
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        sim.host_unregister(h).unwrap();
        sim.free_host(h).unwrap();
    }

    #[test]
    fn free_host_pinned_waits_queued_mapped_kernel() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let h = sim.alloc_host_mapped(bytes).unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 20, bytes), &[h], &[h], s));
        assert!(!sim.query_stream(d, s).unwrap());
        sim.free_host_pinned(h).unwrap();
        assert!(sim.query_stream(d, s).unwrap());
        assert!(sim.clock_ns() > 0);
        assert!(!sim.is_host_mapped(h).unwrap());
    }

    #[test]
    fn alloc_managed_does_not_charge_hbm() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let m = sim.alloc_managed(8 << 20).unwrap();
        assert!(sim.is_managed(m).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        assert!(!sim.is_resident(m, d).unwrap());
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn managed_kernel_fault_migrates_and_charges_hbm() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 20, bytes), &[m], &[m], s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(m, d).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        assert_eq!(sim.bytes_moved(), bytes);
        sim.free_sync(m).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
    }

    #[test]
    fn kernel_start_fault_sees_waited_prefetch() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let home = DeviceId(0);
        let compute = DeviceId(1);
        let copy = StreamId(0);
        let gemm = StreamId(1);
        let bytes = 8u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        sim.mem_advise(m, MemAdvise::SetReadMostly, home).unwrap();
        sim.mem_advise(m, MemAdvise::SetPreferredLocation, home)
            .unwrap();
        enq(sim.prefetch(home, m, copy));
        sim.create_event_disable_timing(EventId(1)).unwrap();
        enq(sim.record_event(home, EventId(1), copy));
        enq(sim.wait_event(compute, EventId(1), gemm));
        enq(sim.kernel(compute, KernelKind::other(8, bytes), &[m], &[], gemm));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(m, home).unwrap());
        assert!(!sim.is_resident(m, compute).unwrap());
        assert_eq!(sim.bytes_moved(), bytes);
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn malloc_can_fill_hbm_until_managed_prefetch() {
        let mut sim = Sim::new(h100().restrict_hbm(4096));
        let d = DeviceId(0);
        let s = StreamId(0);
        let m = sim.alloc_managed(4096).unwrap();
        let a = sim.malloc(d, 4096).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        assert!(!sim.is_resident(m, d).unwrap());
        enq(sim.prefetch(d, m, s));
        match sim.synchronize() {
            Err(SimError::Oom { device, need, free }) => {
                assert_eq!(device, d);
                assert_eq!(need, 4096);
                assert_eq!(free, 0);
            }
            other => panic!("{other:?}"),
        }
        assert!(sim.is_resident(a, d).unwrap());
    }

    #[test]
    fn prefetch_host_refunds_hbm() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.prefetch(d, m, s));
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        enq(sim.prefetch_host(d, m, s));
        sim.synchronize().unwrap();
        assert!(!sim.is_resident(m, d).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        assert!(sim.is_managed(m).unwrap());
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn managed_d2d_prefetch_moves_not_replicas() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.prefetch(d0, m, s));
        sim.synchronize().unwrap();
        enq(sim.prefetch(d1, m, s));
        sim.synchronize().unwrap();
        assert!(!sim.is_resident(m, d0).unwrap());
        assert!(sim.is_resident(m, d1).unwrap());
        assert_eq!(sim.hbm_used(d0).unwrap(), 0);
        assert_eq!(sim.hbm_used(d1).unwrap(), bytes);
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn read_mostly_prefetch_replicates() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        sim.mem_advise(m, MemAdvise::SetReadMostly, d0).unwrap();
        assert!(sim.is_read_mostly(m).unwrap());
        enq(sim.prefetch(d0, m, s));
        sim.synchronize().unwrap();
        enq(sim.prefetch(d1, m, s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(m, d0).unwrap());
        assert!(sim.is_resident(m, d1).unwrap());
        assert_eq!(sim.hbm_used(d0).unwrap(), bytes);
        assert_eq!(sim.hbm_used(d1).unwrap(), bytes);
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn drop_managed_copy_refunds_one_gpu() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        sim.mem_advise(m, MemAdvise::SetReadMostly, d0).unwrap();
        enq(sim.prefetch(d0, m, s));
        sim.synchronize().unwrap();
        enq(sim.prefetch(d1, m, s));
        sim.synchronize().unwrap();
        sim.drop_managed_copy(m, d1).unwrap();
        assert!(sim.is_resident(m, d0).unwrap());
        assert!(!sim.is_resident(m, d1).unwrap());
        assert_eq!(sim.hbm_used(d0).unwrap(), bytes);
        assert_eq!(sim.hbm_used(d1).unwrap(), 0);
        enq(sim.kernel(d0, KernelKind::other(8, bytes), &[m], &[], s));
        sim.synchronize().unwrap();
        let last = sim.drop_managed_copy(m, d0).unwrap_err();
        match last {
            SimError::Invalid { why } => assert!(why.contains("last"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn drop_managed_copy_rejects_unmanaged_and_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let err = sim.drop_managed_copy(a, d).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("managed"), "{why}"),
            other => panic!("{other:?}"),
        }
        let m = sim.alloc_managed(4096).unwrap();
        sim.mem_advise(m, MemAdvise::SetReadMostly, d).unwrap();
        enq(sim.prefetch(d, m, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        let cap = sim.drop_managed_copy(m, d).unwrap_err();
        match cap {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn unset_read_mostly_then_prefetch_moves() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        sim.mem_advise(m, MemAdvise::SetReadMostly, d0).unwrap();
        enq(sim.prefetch(d0, m, s));
        sim.synchronize().unwrap();
        enq(sim.prefetch(d1, m, s));
        sim.synchronize().unwrap();
        sim.mem_advise(m, MemAdvise::UnsetReadMostly, d0).unwrap();
        assert!(!sim.is_read_mostly(m).unwrap());
        assert!(sim.is_resident(m, d0).unwrap());
        enq(sim.prefetch(d0, m, s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(m, d0).unwrap());
        assert!(!sim.is_resident(m, d1).unwrap());
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn write_kernel_invalidates_read_mostly_copies() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        sim.mem_advise(m, MemAdvise::SetReadMostly, d0).unwrap();
        enq(sim.prefetch(d0, m, s));
        enq(sim.prefetch(d1, m, s));
        sim.synchronize().unwrap();
        enq(sim.kernel(d1, KernelKind::other(8, 8), &[m], &[m], s));
        sim.synchronize().unwrap();
        assert!(!sim.is_resident(m, d0).unwrap());
        assert!(sim.is_resident(m, d1).unwrap());
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn accessed_by_kernel_reads_without_migrating() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 32u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.prefetch(d0, m, s));
        sim.synchronize().unwrap();
        sim.mem_advise(m, MemAdvise::SetAccessedBy, d1).unwrap();
        assert!(sim.is_accessed_by(m, d1).unwrap());
        let t0 = sim.clock_ns();
        enq(sim.kernel(d1, KernelKind::other(8, bytes), &[m], &[], s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(m, d0).unwrap());
        assert!(!sim.is_resident(m, d1).unwrap());
        assert_eq!(sim.hbm_used(d1).unwrap(), 0);
        let remote = sim.clock_ns().saturating_sub(t0);
        let mut local = Sim::new(HardwareProfile::example_2xh100_pcie());
        let a = local.alloc_managed(bytes).unwrap();
        enq(local.prefetch(d0, a, s));
        local.synchronize().unwrap();
        let t1 = local.clock_ns();
        enq(local.kernel(d0, KernelKind::other(8, bytes), &[a], &[], s));
        local.synchronize().unwrap();
        let hbm = local.clock_ns().saturating_sub(t1);
        assert!(remote > hbm, "remote={remote} hbm={hbm}");
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn va_set_access_kernel_reads_without_dest_hbm() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 32u64 << 20;
        let va = sim.va_reserve(bytes).unwrap();
        sim.va_map(va, d0).unwrap();
        enq(sim.memcpy_pinned_to_device(d0, va, bytes, s));
        sim.synchronize().unwrap();
        sim.va_set_access(va, d1).unwrap();
        assert!(sim.is_accessed_by(va, d1).unwrap());
        assert!(!sim.is_resident(va, d1).unwrap());
        let t0 = sim.clock_ns();
        enq(sim.kernel(d1, KernelKind::other(8, bytes), &[va], &[], s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(va, d0).unwrap());
        assert!(!sim.is_resident(va, d1).unwrap());
        assert_eq!(sim.hbm_used(d1).unwrap(), 0);
        let remote = sim.clock_ns().saturating_sub(t0);
        let mut local = Sim::new(HardwareProfile::example_2xh100_pcie());
        let a = local.va_reserve(bytes).unwrap();
        local.va_map(a, d0).unwrap();
        enq(local.memcpy_pinned_to_device(d0, a, bytes, s));
        local.synchronize().unwrap();
        let t1 = local.clock_ns();
        enq(local.kernel(d0, KernelKind::other(8, bytes), &[a], &[], s));
        local.synchronize().unwrap();
        let hbm = local.clock_ns().saturating_sub(t1);
        assert!(remote > hbm, "remote={remote} hbm={hbm}");
        sim.va_unmap(va).unwrap();
        assert!(!sim.is_accessed_by(va, d1).unwrap());
        sim.va_free(va).unwrap();
    }

    #[test]
    fn va_set_access_rejects_unmapped_peer_and_capture() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let va = sim.va_reserve(4096).unwrap();
        let unmapped = sim.va_set_access(va, d1).unwrap_err();
        match unmapped {
            SimError::Invalid { why } => assert!(why.contains("mapped"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_map(va, d0).unwrap();
        sim.disable_peer(d0, d1).unwrap();
        match sim.va_set_access(va, d1).unwrap_err() {
            SimError::PeerDisabled { src, dst } => {
                assert_eq!(src, d0);
                assert_eq!(dst, d1);
            }
            other => panic!("{other:?}"),
        }
        sim.enable_peer(d0, d1).unwrap();
        sim.va_set_access(va, d1).unwrap();
        sim.va_unset_access(va, d1).unwrap();
        assert!(!sim.is_accessed_by(va, d1).unwrap());
        sim.begin_capture(d0, s).unwrap();
        match sim.va_set_access(va, d1).unwrap_err() {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        let mut chain = Sim::new(HardwareProfile::example_asymmetric_links());
        let far = chain.va_reserve(4096).unwrap();
        chain.va_map(far, DeviceId(0)).unwrap();
        match chain.va_set_access(far, DeviceId(2)).unwrap_err() {
            SimError::NoPeer { src, dst } => {
                assert_eq!(src, DeviceId(0));
                assert_eq!(dst, DeviceId(2));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn va_set_access_write_is_not_resident() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let va = sim.va_reserve(4096).unwrap();
        sim.va_map(va, d0).unwrap();
        enq(sim.memcpy_pinned_to_device(d0, va, 4096, s));
        sim.synchronize().unwrap();
        sim.va_set_access(va, d1).unwrap();
        enq(sim.kernel(d1, KernelKind::other(8, 8), &[va], &[va], s));
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, va);
                assert_eq!(device, d1);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn pool_set_access_kernel_reads_and_writes_without_dest_hbm() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 32u64 << 20;
        let p0 = sim.default_pool(d0).unwrap();
        let a = sim.alloc(d0, bytes, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d0, a, bytes, s));
        sim.synchronize().unwrap();
        sim.pool_set_access(p0, d1).unwrap();
        assert!(sim.is_pool_accessed_by(p0, d1).unwrap());
        assert!(sim.is_accessed_by(a, d1).unwrap());
        assert!(!sim.is_resident(a, d1).unwrap());
        let t0 = sim.clock_ns();
        enq(sim.kernel(d1, KernelKind::other(8, bytes), &[a], &[a], s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(a, d0).unwrap());
        assert!(!sim.is_resident(a, d1).unwrap());
        assert_eq!(sim.hbm_used(d1).unwrap(), 0);
        let remote = sim.clock_ns().saturating_sub(t0);
        let mut local = Sim::new(HardwareProfile::example_2xh100_pcie());
        let b = local.alloc(d0, bytes, s).unwrap();
        enq(local.memcpy_pinned_to_device(d0, b, bytes, s));
        local.synchronize().unwrap();
        let t1 = local.clock_ns();
        enq(local.kernel(d0, KernelKind::other(8, bytes), &[b], &[b], s));
        local.synchronize().unwrap();
        let hbm = local.clock_ns().saturating_sub(t1);
        assert!(remote > hbm, "remote={remote} hbm={hbm}");
        sim.free(d0, a, s).unwrap();
        sim.synchronize().unwrap();
    }

    #[test]
    fn pool_set_access_covers_existing_and_later_allocs() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let p0 = sim.default_pool(d0).unwrap();
        let first = sim.alloc(d0, 4096, s).unwrap();
        sim.synchronize().unwrap();
        sim.pool_set_access(p0, d1).unwrap();
        let later = sim.alloc(d0, 4096, s).unwrap();
        sim.synchronize().unwrap();
        enq(sim.kernel(d1, KernelKind::other(8, 8), &[first], &[], s));
        enq(sim.kernel(d1, KernelKind::other(8, 8), &[later], &[], s));
        sim.synchronize().unwrap();
        assert_eq!(sim.hbm_used(d1).unwrap(), 0);
    }

    #[test]
    fn pool_set_access_rejects_peer_disabled_malloc_and_capture() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let p0 = sim.default_pool(d0).unwrap();
        sim.disable_peer(d0, d1).unwrap();
        match sim.pool_set_access(p0, d1).unwrap_err() {
            SimError::PeerDisabled { src, dst } => {
                assert_eq!(src, d0);
                assert_eq!(dst, d1);
            }
            other => panic!("{other:?}"),
        }
        sim.enable_peer(d0, d1).unwrap();
        sim.pool_set_access(p0, d1).unwrap();
        sim.pool_unset_access(p0, d1).unwrap();
        assert!(!sim.is_pool_accessed_by(p0, d1).unwrap());
        let m = sim.malloc(d0, 4096).unwrap();
        sim.pool_set_access(p0, d1).unwrap();
        enq(sim.kernel(d1, KernelKind::other(8, 8), &[m], &[], s));
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, m);
                assert_eq!(device, d1);
            }
            other => panic!("{other:?}"),
        }
        let mut cap = Sim::new(HardwareProfile::example_2xh100_pcie());
        let p_cap = cap.default_pool(d0).unwrap();
        cap.begin_capture(d0, s).unwrap();
        match cap.pool_set_access(p_cap, d1).unwrap_err() {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = cap.end_capture().unwrap();
        let mut chain = Sim::new(HardwareProfile::example_asymmetric_links());
        let far = chain.default_pool(DeviceId(0)).unwrap();
        match chain.pool_set_access(far, DeviceId(2)).unwrap_err() {
            SimError::NoPeer { src, dst } => {
                assert_eq!(src, DeviceId(0));
                assert_eq!(dst, DeviceId(2));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn pool_set_access_memset_peer_skips_dest_hbm() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let p0 = sim.default_pool(d0).unwrap();
        let a = sim.alloc(d0, bytes, s).unwrap();
        sim.synchronize().unwrap();
        sim.pool_set_access(p0, d1).unwrap();
        enq(sim.memset(d1, a, bytes, s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(a, d0).unwrap());
        assert!(!sim.is_resident(a, d1).unwrap());
        assert_eq!(sim.hbm_used(d1).unwrap(), 0);
    }

    #[test]
    fn accessed_by_write_still_migrates() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.prefetch(d0, m, s));
        sim.synchronize().unwrap();
        sim.mem_advise(m, MemAdvise::SetAccessedBy, d1).unwrap();
        enq(sim.kernel(d1, KernelKind::other(8, 8), &[m], &[m], s));
        sim.synchronize().unwrap();
        assert!(!sim.is_resident(m, d0).unwrap());
        assert!(sim.is_resident(m, d1).unwrap());
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn preferred_location_kernel_reads_without_migrating() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 32u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.prefetch(d0, m, s));
        sim.synchronize().unwrap();
        sim.mem_advise(m, MemAdvise::SetPreferredLocation, d0)
            .unwrap();
        assert!(sim.is_preferred_location(m, d0).unwrap());
        assert!(!sim.is_preferred_location(m, d1).unwrap());
        let t0 = sim.clock_ns();
        enq(sim.kernel(d1, KernelKind::other(8, bytes), &[m], &[], s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(m, d0).unwrap());
        assert!(!sim.is_resident(m, d1).unwrap());
        assert_eq!(sim.hbm_used(d1).unwrap(), 0);
        let remote = sim.clock_ns().saturating_sub(t0);
        let mut local = Sim::new(HardwareProfile::example_2xh100_pcie());
        let a = local.alloc_managed(bytes).unwrap();
        enq(local.prefetch(d0, a, s));
        local.synchronize().unwrap();
        let t1 = local.clock_ns();
        enq(local.kernel(d0, KernelKind::other(8, bytes), &[a], &[], s));
        local.synchronize().unwrap();
        let hbm = local.clock_ns().saturating_sub(t1);
        assert!(remote > hbm, "remote={remote} hbm={hbm}");
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn preferred_location_write_still_migrates() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.prefetch(d0, m, s));
        sim.synchronize().unwrap();
        sim.mem_advise(m, MemAdvise::SetPreferredLocation, d0)
            .unwrap();
        enq(sim.kernel(d1, KernelKind::other(8, 8), &[m], &[m], s));
        sim.synchronize().unwrap();
        assert!(!sim.is_resident(m, d0).unwrap());
        assert!(sim.is_resident(m, d1).unwrap());
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn unset_preferred_location_then_kernel_migrates() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.prefetch(d0, m, s));
        sim.synchronize().unwrap();
        sim.mem_advise(m, MemAdvise::SetPreferredLocation, d0)
            .unwrap();
        sim.mem_advise(m, MemAdvise::UnsetPreferredLocation, d0)
            .unwrap();
        assert!(!sim.is_preferred_location(m, d0).unwrap());
        enq(sim.kernel(d1, KernelKind::other(8, 8), &[m], &[], s));
        sim.synchronize().unwrap();
        assert!(!sim.is_resident(m, d0).unwrap());
        assert!(sim.is_resident(m, d1).unwrap());
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn preferred_location_host_kernel_first_touch_still_migrates() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        sim.mem_advise(m, MemAdvise::SetPreferredLocationHost, d)
            .unwrap();
        assert!(sim.is_preferred_host(m).unwrap());
        enq(sim.kernel(d, KernelKind::other(8, 8), &[m], &[], s));
        sim.synchronize().unwrap();
        assert!(sim.is_resident(m, d).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn preferred_location_without_residency_still_faults() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        sim.mem_advise(m, MemAdvise::SetPreferredLocation, d0)
            .unwrap();
        enq(sim.kernel(d1, KernelKind::other(8, 8), &[m], &[], s));
        sim.synchronize().unwrap();
        assert!(!sim.is_resident(m, d0).unwrap());
        assert!(sim.is_resident(m, d1).unwrap());
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn mem_advise_rejects_unmanaged_and_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.malloc(d, 4096).unwrap();
        let err = sim.mem_advise(a, MemAdvise::SetReadMostly, d).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("managed"), "{why}"),
            other => panic!("{other:?}"),
        }
        let pref = sim
            .mem_advise(a, MemAdvise::SetPreferredLocation, d)
            .unwrap_err();
        match pref {
            SimError::Invalid { why } => assert!(why.contains("managed"), "{why}"),
            other => panic!("{other:?}"),
        }
        let m = sim.alloc_managed(4096).unwrap();
        sim.begin_capture(d, s).unwrap();
        let cap = sim.mem_advise(m, MemAdvise::SetReadMostly, d).unwrap_err();
        match cap {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let cap_pref = sim
            .mem_advise(m, MemAdvise::SetPreferredLocation, d)
            .unwrap_err();
        match cap_pref {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn already_local_prefetch_does_not_count_bytes() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 8u64 << 20;
        let m = sim.alloc_managed(bytes).unwrap();
        enq(sim.prefetch(d, m, s));
        sim.synchronize().unwrap();
        let moved = sim.bytes_moved();
        let t0 = sim.clock_ns();
        enq(sim.prefetch(d, m, s));
        sim.synchronize().unwrap();
        assert_eq!(sim.bytes_moved(), moved);
        assert!(sim.clock_ns() > t0);
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn managed_after_prefetch_is_faster_than_mapped() {
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 32u64 << 20;
        let mut mapped = Sim::new(h100());
        let h = mapped.alloc_host_mapped(bytes).unwrap();
        enq(mapped.kernel(d, KernelKind::other(1 << 20, bytes), &[h], &[h], s));
        mapped.synchronize().unwrap();
        let map_ns = mapped.clock_ns();

        let mut um = Sim::new(h100());
        let m = um.alloc_managed(bytes).unwrap();
        enq(um.prefetch(d, m, s));
        um.synchronize().unwrap();
        let t0 = um.clock_ns();
        enq(um.kernel(d, KernelKind::other(1 << 20, bytes), &[m], &[m], s));
        um.synchronize().unwrap();
        let um_ns = um.clock_ns().saturating_sub(t0);
        assert!(
            map_ns > um_ns,
            "mapped PCIe kernel vs managed-on-HBM; mapped={map_ns} um={um_ns}"
        );
    }

    #[test]
    fn graph_records_managed_prefetch_then_kernel() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let m = sim.alloc_managed(4096).unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.prefetch(d, m, s));
        enq(sim.kernel(d, KernelKind::other(8, 4096), &[m], &[m], s));
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 2);
        assert!(!sim.is_resident(m, d).unwrap());
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(m, d).unwrap());
        sim.free_sync(m).unwrap();
    }

    #[test]
    fn graph_kernel_without_managed_prefetch_is_not_resident() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let m = sim.alloc_managed(4096).unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 4096), &[m], &[m], s));
        let g = sim.end_capture().unwrap();
        let _n = sim.launch_graph(g, s).unwrap();
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, m);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn prefetch_rejects_unmanaged() {
        let mut sim = Sim::new(h100());
        let a = sim.malloc(DeviceId(0), 4096).unwrap();
        match sim.prefetch(DeviceId(0), a, StreamId(0)) {
            Err(SimError::Invalid { why }) => assert!(why.contains("managed")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn va_reserve_does_not_charge_hbm() {
        let mut sim = Sim::new(h100().restrict_hbm(4096));
        let d = DeviceId(0);
        let va = sim.va_reserve(4096).unwrap();
        assert!(sim.is_vmm(va).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        let a = sim.malloc(d, 4096).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        match sim.va_map(va, d) {
            Err(SimError::Oom { device, need, free }) => {
                assert_eq!(device, d);
                assert_eq!(need, 4096);
                assert_eq!(free, 0);
            }
            other => panic!("{other:?}"),
        }
        sim.free_sync(a).unwrap();
        sim.va_map(va, d).unwrap();
        assert!(sim.is_resident(va, d).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn va_unmap_keeps_pointer_for_remap() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 1u64 << 20;
        let va = sim.va_reserve(bytes).unwrap();
        sim.va_map(va, d).unwrap();
        enq(sim.memcpy_pinned_to_device(d, va, bytes, s));
        enq(sim.kernel(d, KernelKind::other(1 << 20, bytes), &[va], &[va], s));
        sim.synchronize().unwrap();
        sim.va_unmap(va).unwrap();
        assert!(!sim.is_resident(va, d).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        assert!(sim.is_vmm(va).unwrap());
        sim.va_map(va, d).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[va], &[va], s));
        sim.synchronize().unwrap();
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn kernel_on_unmapped_va_is_not_resident() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(4096).unwrap();
        sim.va_map(va, d).unwrap();
        sim.va_unmap(va).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[va], &[va], s));
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, va);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn va_free_rejects_mapped_and_cuda_free() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let va = sim.va_reserve(4096).unwrap();
        sim.va_map(va, d).unwrap();
        match sim.va_free(va) {
            Err(SimError::Invalid { why }) => assert!(why.contains("mapped")),
            other => panic!("{other:?}"),
        }
        match sim.free_sync(va) {
            Err(SimError::UnknownAlloc { alloc }) => assert_eq!(alloc, va),
            other => panic!("{other:?}"),
        }
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn va_map_rejects_already_mapped() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let va = sim.va_reserve(4096).unwrap();
        sim.va_map(va, d).unwrap();
        match sim.va_map(va, d) {
            Err(SimError::Invalid { why }) => assert!(why.contains("already")),
            other => panic!("{other:?}"),
        }
        match sim.va_unmap(va) {
            Ok(()) => {}
            other => panic!("{other:?}"),
        }
        match sim.va_unmap(va) {
            Err(SimError::Invalid { why }) => assert!(why.contains("not mapped")),
            other => panic!("{other:?}"),
        }
        sim.va_free(va).unwrap();
    }

    #[test]
    fn va_create_map_handle_shares_hbm_across_vas() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 4096u64;
        let h = sim.va_create(d, bytes).unwrap();
        assert!(sim.is_handle_live(h).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        let a = sim.va_reserve(bytes).unwrap();
        let b = sim.va_reserve(bytes).unwrap();
        sim.va_map_handle(a, d, 0, h).unwrap();
        sim.va_map_handle(b, d, 0, h).unwrap();
        assert_eq!(sim.handle_maps(h).unwrap(), 2);
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        assert!(sim.is_resident(a, d).unwrap());
        assert!(sim.is_resident(b, d).unwrap());
        enq(sim.memcpy_pinned_to_device(d, a, bytes, s));
        enq(sim.kernel(d, KernelKind::other(8, 8), &[b], &[b], s));
        sim.synchronize().unwrap();
        sim.va_unmap(a).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        assert_eq!(sim.handle_maps(h).unwrap(), 1);
        sim.va_unmap(b).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        assert_eq!(sim.handle_maps(h).unwrap(), 0);
        assert_eq!(sim.handle_refs(h).unwrap(), 1);
        sim.va_release_handle(h).unwrap();
        assert!(!sim.is_handle_live(h).unwrap());
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        sim.va_free(a).unwrap();
        sim.va_free(b).unwrap();
    }

    #[test]
    fn va_map_handle_rejects_mismatch_mapped_and_release() {
        let mut sim = Sim::new(HardwareProfile::example_2xh100_pcie());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let bytes = 4096u64;
        let h = sim.va_create(d0, bytes).unwrap();
        let va = sim.va_reserve(bytes).unwrap();
        match sim.va_map_handle(va, d1, 0, h).unwrap_err() {
            SimError::Invalid { why } => assert!(why.contains("device"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_map_handle(va, d0, 0, h).unwrap();
        sim.va_release_handle(h).unwrap();
        assert!(!sim.is_handle_live(h).unwrap());
        assert_eq!(sim.hbm_used(d0).unwrap(), bytes);
        let va2 = sim.va_reserve(bytes).unwrap();
        match sim.va_map_handle(va2, d0, 0, h).unwrap_err() {
            SimError::Invalid { why } => assert!(why.contains("released"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_unmap(va).unwrap();
        assert_eq!(sim.hbm_used(d0).unwrap(), 0);
        sim.va_free(va).unwrap();
        sim.va_free(va2).unwrap();
        match sim.va_create(d0, 0).unwrap_err() {
            SimError::Invalid { why } => assert!(why.contains("zero"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn va_retain_handle_aliases_combined_map_without_extra_hbm() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 4096u64;
        let a = sim.va_reserve(bytes).unwrap();
        let b = sim.va_reserve(bytes).unwrap();
        sim.va_map(a, d).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        let h = sim.va_retain_handle(a, d, 0).unwrap();
        assert_eq!(sim.handle_refs(h).unwrap(), 1);
        assert_eq!(sim.handle_maps(h).unwrap(), 1);
        sim.va_map_handle(b, d, 0, h).unwrap();
        assert_eq!(sim.handle_maps(h).unwrap(), 2);
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        enq(sim.kernel(d, KernelKind::other(8, 8), &[b], &[b], s));
        sim.synchronize().unwrap();
        sim.va_unmap(a).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        sim.va_unmap(b).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        sim.va_release_handle(h).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        sim.va_free(a).unwrap();
        sim.va_free(b).unwrap();
    }

    #[test]
    fn va_retain_restores_released_mapped_handle() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let bytes = 4096u64;
        let h = sim.va_create(d, bytes).unwrap();
        let a = sim.va_reserve(bytes).unwrap();
        let b = sim.va_reserve(bytes).unwrap();
        sim.va_map_handle(a, d, 0, h).unwrap();
        sim.va_release_handle(h).unwrap();
        assert!(!sim.is_handle_live(h).unwrap());
        let h2 = sim.va_retain_handle(a, d, 0).unwrap();
        assert_eq!(h2, h);
        assert_eq!(sim.handle_refs(h).unwrap(), 1);
        sim.va_map_handle(b, d, 0, h2).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), bytes);
        sim.va_unmap(a).unwrap();
        sim.va_unmap(b).unwrap();
        sim.va_release_handle(h2).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        sim.va_free(a).unwrap();
        sim.va_free(b).unwrap();
    }

    #[test]
    fn va_retain_handle_rejects_unmapped_and_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(4096).unwrap();
        match sim.va_retain_handle(va, d, 0).unwrap_err() {
            SimError::Invalid { why } => assert!(why.contains("no such map"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_map(va, d).unwrap();
        sim.begin_capture(d, s).unwrap();
        match sim.va_retain_handle(va, d, 0).unwrap_err() {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn va_granularity_rejects_unaligned_reserve_and_map() {
        let gran = 2u64 << 20;
        let mut sim = Sim::new(h100().with_va_granularity(gran));
        let d = DeviceId(0);
        match sim.va_reserve(4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("unaligned"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.va_create(d, 4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("unaligned"), "{why}"),
            other => panic!("{other:?}"),
        }
        let va = sim.va_reserve(gran.saturating_mul(2)).unwrap();
        match sim.va_map_range(va, d, 4096, gran) {
            Err(SimError::Invalid { why }) => assert!(why.contains("unaligned"), "{why}"),
            other => panic!("{other:?}"),
        }
        match sim.va_map_range(va, d, 0, 4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("unaligned"), "{why}"),
            other => panic!("{other:?}"),
        }
        sim.va_map_range(va, d, 0, gran).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), gran);
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn va_granularity_default_allows_4096() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let va = sim.va_reserve(4096).unwrap();
        sim.va_map(va, d).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn va_map_range_charges_only_that_span() {
        let mut sim = Sim::new(h100().restrict_hbm(8192));
        let d = DeviceId(0);
        let va = sim.va_reserve(16_384).unwrap();
        sim.va_map_range(va, d, 0, 4096).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        assert_eq!(sim.vmm_mapped_bytes(va, d).unwrap(), 4096);
        assert!(!sim.is_resident(va, d).unwrap());
        sim.va_map_range(va, d, 4096, 4096).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 8192);
        match sim.va_map_range(va, d, 8192, 4096) {
            Err(SimError::Oom { device, need, free }) => {
                assert_eq!(device, d);
                assert_eq!(need, 4096);
                assert_eq!(free, 0);
            }
            other => panic!("{other:?}"),
        }
        sim.va_unmap_range(va, d, 0, 4096).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 4096);
        assert_eq!(sim.vmm_mapped_bytes(va, d).unwrap(), 4096);
        sim.va_unmap(va).unwrap();
        assert_eq!(sim.hbm_used(d).unwrap(), 0);
        sim.va_free(va).unwrap();
    }

    #[test]
    fn adjacent_ranges_cover_the_va_for_kernels() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 0, 4096).unwrap();
        sim.va_map_range(va, d, 4096, 4096).unwrap();
        assert!(sim.is_resident(va, d).unwrap());
        enq(sim.kernel(d, KernelKind::other(8, 8), &[va], &[va], s));
        sim.synchronize().unwrap();
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn kernel_on_partial_va_is_not_resident() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 0, 4096).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[va], &[va], s));
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, va);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn kernel_bufs_on_mapped_span_is_ok() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 4096, 4096).unwrap();
        assert!(!sim.is_resident(va, d).unwrap());
        assert!(sim.is_range_resident(va, d, 4096, 4096).unwrap());
        assert!(!sim.is_range_resident(va, d, 0, 4096).unwrap());
        let buf = KernelBuf::span(va, 4096, 4096);
        enq(sim.kernel_bufs(d, KernelKind::other(8, 8), &[buf], &[buf], s));
        sim.synchronize().unwrap();
        sim.va_unmap_range(va, d, 4096, 4096).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn kernel_bufs_on_unmapped_span_is_not_resident() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 0, 4096).unwrap();
        let buf = KernelBuf::span(va, 4096, 4096);
        enq(sim.kernel_bufs(d, KernelKind::other(8, 8), &[buf], &[buf], s));
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, va);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn kernel_bufs_past_alloc_is_invalid() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(4096).unwrap();
        sim.va_map(va, d).unwrap();
        let buf = KernelBuf::span(va, 0, 8192);
        match sim.kernel_bufs(d, KernelKind::other(8, 8), &[buf], &[buf], s) {
            Err(SimError::Invalid { why }) => assert!(why.contains("past")),
            other => panic!("{other:?}"),
        }
        match sim.is_range_resident(va, d, 0, 8192) {
            Err(SimError::Invalid { why }) => assert!(why.contains("past")),
            other => panic!("{other:?}"),
        }
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn h2d_into_interior_mapped_page_is_ok() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 4096, 4096).unwrap();
        enq(sim.memcpy(
            d,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d),
                alloc: va,
                bytes: 4096,
                offset: 4096,
            },
            s,
        ));
        sim.synchronize().unwrap();
        sim.va_unmap_range(va, d, 4096, 4096).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn h2d_into_interior_unmapped_page_is_not_resident() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 0, 4096).unwrap();
        enq(sim.memcpy(
            d,
            MemcpyOp {
                src: Place::HostPinned,
                dst: Place::Device(d),
                alloc: va,
                bytes: 4096,
                offset: 4096,
            },
            s,
        ));
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, va);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn graph_replays_kernel_bufs_span() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 0, 4096).unwrap();
        sim.begin_capture(d, s).unwrap();
        let buf = KernelBuf::span(va, 0, 4096);
        enq(sim.kernel_bufs(d, KernelKind::other(8, 8), &[buf], &[buf], s));
        let g = sim.end_capture().unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        sim.va_unmap_range(va, d, 0, 4096).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn memset_buf_on_mapped_span_is_ok() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 4096, 4096).unwrap();
        assert!(!sim.is_resident(va, d).unwrap());
        enq(sim.memset_buf(d, KernelBuf::span(va, 4096, 4096), s));
        sim.synchronize().unwrap();
        sim.va_unmap_range(va, d, 4096, 4096).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn memset_whole_va_on_partial_map_is_not_resident() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 0, 4096).unwrap();
        enq(sim.memset(d, va, 8192, s));
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, va);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn memset_first_page_of_partial_va_is_ok() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 0, 4096).unwrap();
        enq(sim.memset(d, va, 4096, s));
        sim.synchronize().unwrap();
        sim.va_unmap_range(va, d, 0, 4096).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn overlapping_va_map_range_is_already_mapped() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 0, 4096).unwrap();
        match sim.va_map_range(va, d, 2048, 4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("already")),
            other => panic!("{other:?}"),
        }
        match sim.va_map_range(va, d, 8192, 4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("past")),
            other => panic!("{other:?}"),
        }
        sim.va_unmap_range(va, d, 0, 4096).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn h2d_into_unmapped_span_is_not_resident() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 0, 4096).unwrap();
        enq(sim.memcpy_pinned_to_device(d, va, 8192, s));
        match sim.synchronize() {
            Err(SimError::NotResident { alloc, device }) => {
                assert_eq!(alloc, va);
                assert_eq!(device, d);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn h2d_into_mapped_span_is_ok() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let va = sim.va_reserve(8192).unwrap();
        sim.va_map_range(va, d, 0, 4096).unwrap();
        enq(sim.memcpy_pinned_to_device(d, va, 4096, s));
        sim.synchronize().unwrap();
        sim.va_unmap(va).unwrap();
        sim.va_free(va).unwrap();
    }

    #[test]
    fn va_acquire_reuses_released_pointer() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let a = sim.va_acquire(d, 4096).unwrap();
        assert_eq!(sim.vmm_idle_len(), 0);
        sim.va_release(a).unwrap();
        assert_eq!(sim.vmm_idle_len(), 1);
        sim.va_release(a).unwrap();
        assert_eq!(sim.vmm_idle_len(), 1);
        let b = sim.va_acquire(d, 4096).unwrap();
        assert_eq!(a, b);
        assert_eq!(sim.vmm_idle_len(), 0);
        assert!(sim.is_resident(a, d).unwrap());
        sim.va_release(a).unwrap();
        sim.va_free(a).unwrap();
        assert_eq!(sim.vmm_idle_len(), 0);
        let c = sim.va_acquire(d, 4096).unwrap();
        assert_ne!(a, c);
        sim.va_release(c).unwrap();
    }

    #[test]
    fn va_acquire_reuse_skips_reserve_overhead() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let a = sim.va_acquire(d, 4096).unwrap();
        let fresh = sim.clock_ns();
        sim.va_release(a).unwrap();
        let after_rel = sim.clock_ns();
        let b = sim.va_acquire(d, 4096).unwrap();
        assert_eq!(a, b);
        let reuse = sim.clock_ns().saturating_sub(after_rel);
        let map_ns = h100().gpu(d).unwrap().alloc_overhead_ns.max(1);
        assert_eq!(reuse, map_ns);
        assert!(reuse < fresh);
        sim.va_release(a).unwrap();
    }

    #[test]
    fn va_acquire_does_not_share_sizes() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let a = sim.va_acquire(d, 4096).unwrap();
        sim.va_release(a).unwrap();
        let b = sim.va_acquire(d, 8192).unwrap();
        assert_ne!(a, b);
        assert_eq!(sim.vmm_idle_len(), 1);
        sim.va_release(b).unwrap();
        let c = sim.va_acquire(d, 4096).unwrap();
        assert_eq!(a, c);
        assert_eq!(sim.vmm_idle_len(), 1);
        sim.va_release(c).unwrap();
        sim.va_free(b).unwrap();
        sim.va_free(c).unwrap();
    }

    #[test]
    fn va_acquire_oom_parks_reserved_va() {
        let mut sim = Sim::new(h100().restrict_hbm(4096));
        let d = DeviceId(0);
        let blocker = sim.malloc(d, 4096).unwrap();
        match sim.va_acquire(d, 4096) {
            Err(SimError::Oom { device, need, free }) => {
                assert_eq!(device, d);
                assert_eq!(need, 4096);
                assert_eq!(free, 0);
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(sim.vmm_idle_len(), 1);
        sim.free_sync(blocker).unwrap();
        let va = sim.va_acquire(d, 4096).unwrap();
        assert_eq!(sim.vmm_idle_len(), 0);
        sim.va_release(va).unwrap();
    }

    #[test]
    fn va_acquire_rejects_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        match sim.va_acquire(d, 4096) {
            Err(SimError::Invalid { why }) => assert!(why.contains("capture")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn va_release_rejects_malloc() {
        let mut sim = Sim::new(h100());
        let a = sim.malloc(DeviceId(0), 4096).unwrap();
        match sim.va_release(a) {
            Err(SimError::Invalid { why }) => assert!(why.contains("VA")),
            other => panic!("{other:?}"),
        }
        sim.free_sync(a).unwrap();
    }

    #[test]
    fn va_acquire_paged_covers_the_va_in_spans() {
        let mut one = Sim::new(h100());
        let mut paged = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let bytes = 16u64 << 10;
        let page = bytes / 4;
        let a = one.va_acquire(d, bytes).unwrap();
        let t1 = one.clock_ns();
        let b = paged.va_acquire_paged(d, bytes, page).unwrap();
        let t4 = paged.clock_ns();
        assert!(t4 > t1, "paged={t4} one={t1}");
        assert_eq!(paged.vmm_mapped_bytes(b, d).unwrap(), bytes);
        assert!(paged.is_resident(b, d).unwrap());
        enq(paged.kernel(d, KernelKind::other(8, 8), &[b], &[b], s));
        paged.synchronize().unwrap();
        one.va_release(a).unwrap();
        paged.va_release(b).unwrap();
        let c = paged.va_acquire_paged(d, bytes, page).unwrap();
        assert_eq!(c, b);
        paged.va_release(c).unwrap();
        paged.va_free(c).unwrap();
    }

    #[test]
    fn host_func_duration_matches_profile() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let t0 = sim.clock_ns();
        enq(sim.host_func(d, StreamId(0)));
        sim.synchronize().unwrap();
        assert_eq!(
            sim.clock_ns().saturating_sub(t0),
            h100().gpu(d).unwrap().host_func_ns.max(1)
        );
        assert!(sim.operations().any(|o| matches!(o.kind, GpuOp::HostFunc)));
    }

    #[test]
    fn host_func_does_not_occupy_compute() {
        let d = DeviceId(0);
        let bytes = 8u64 << 20;
        let k = KernelKind::GroupedMoeGemm {
            experts: 8,
            tokens_per_expert: 16,
            hidden: 2048,
            ff: 2048,
            dtype: DType::Fp16,
        };
        let wall = |same: bool| {
            let mut sim = Sim::new(h100());
            let a = sim.alloc(d, bytes, StreamId(0)).unwrap();
            enq(sim.memcpy_pinned_to_device(d, a, bytes, StreamId(0)));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, k.clone(), &[a], &[a], StreamId(0)));
            let hs = if same { StreamId(0) } else { StreamId(1) };
            enq(sim.host_func(d, hs));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let serial = wall(true);
        let overlap = wall(false);
        assert!(
            serial > overlap,
            "host callback must not take the compute engine; serial={serial} overlap={overlap}"
        );
    }

    #[test]
    fn host_func_waits_for_prior_kernel_on_same_stream() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        enq(sim.kernel(d, KernelKind::other(1 << 20, 4096), &[a], &[a], s));
        enq(sim.host_func(d, s));
        sim.synchronize().unwrap();
        let ops: Vec<Operation> = sim.operations().collect();
        let k = ops
            .iter()
            .find(|o| matches!(o.kind, GpuOp::Kernel { .. }) && o.start_ns.is_some())
            .expect("kernel");
        let h = ops
            .iter()
            .find(|o| matches!(o.kind, GpuOp::HostFunc))
            .expect("host");
        assert!(k.done_ns.unwrap() <= h.start_ns.unwrap());
    }

    #[test]
    fn graph_can_capture_host_func() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        enq(sim.host_func(d, s));
        let g = sim.end_capture().unwrap();
        let n = sim.launch_graph(g, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert!(sim
            .operations()
            .any(|o| matches!(o.kind, GpuOp::HostFunc) && o.done));
    }

    #[test]
    fn disable_timing_event_cannot_elapsed() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let start = EventId(1);
        let end = EventId(2);
        sim.create_event_disable_timing(start).unwrap();
        sim.create_event(end).unwrap();
        assert!(!sim.event_timing(start).unwrap());
        assert!(sim.event_timing(end).unwrap());
        enq(sim.record_event(d, start, s));
        enq(sim.record_event(d, end, s));
        sim.synchronize().unwrap();
        match sim.event_elapsed_ns(start, end) {
            Err(SimError::Invalid { why }) => assert!(why.contains("disable timing"), "{why}"),
            other => panic!("{other:?}"),
        }
        let ns = sim.event_elapsed_ns(end, end).unwrap();
        assert_eq!(ns, 0);
        assert!(sim.query_event(start).unwrap());
        enq(sim.wait_event(d, start, StreamId(1)));
        sim.synchronize().unwrap();
    }

    #[test]
    fn create_event_rejects_duplicate_and_capture() {
        let mut sim = Sim::new(h100());
        let ev = EventId(8);
        sim.create_event(ev).unwrap();
        let err = sim.create_event(ev).unwrap_err();
        assert!(matches!(err, SimError::Invalid { .. }), "{err:?}");
        let err = sim.create_event_disable_timing(ev).unwrap_err();
        assert!(matches!(err, SimError::Invalid { .. }), "{err:?}");
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        let err = sim.create_event(EventId(9)).unwrap_err();
        assert!(matches!(err, SimError::Invalid { .. }), "{err:?}");
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn blocking_stream_serializes_with_null() {
        let d = DeviceId(0);
        let bytes = 32u64 << 20;
        let k = KernelKind::GroupedMoeGemm {
            experts: 8,
            tokens_per_expert: 16,
            hidden: 2048,
            ff: 2048,
            dtype: DType::Fp16,
        };
        let run = |blocking: bool| {
            let mut sim = Sim::new(h100());
            if blocking {
                sim.set_stream_blocking(d, StreamId(1), true).unwrap();
            }
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
            "cudaStreamCreate must serialize with NULL; serial={serial} overlap={overlap}"
        );
    }

    #[test]
    fn set_stream_blocking_rejects_null() {
        let mut sim = Sim::new(h100());
        let err = sim
            .set_stream_blocking(DeviceId(0), StreamId::NULL, true)
            .unwrap_err();
        assert!(matches!(err, SimError::Invalid { .. }), "{err:?}");
        assert!(!sim.stream_is_blocking(DeviceId(0), StreamId::NULL));
    }

    #[test]
    fn set_created_streams_blocking_skips_null() {
        let mut sim = Sim::new(h100());
        sim.set_created_streams_blocking(2).unwrap();
        assert!(!sim.stream_is_blocking(DeviceId(0), StreamId::NULL));
        assert!(sim.stream_is_blocking(DeviceId(0), StreamId(1)));
        sim.set_created_streams_blocking(1).unwrap();
        assert!(sim.stream_is_blocking(DeviceId(0), StreamId(1)));
        sim.set_stream_blocking(DeviceId(0), StreamId(1), false)
            .unwrap();
        assert!(!sim.stream_is_blocking(DeviceId(0), StreamId(1)));
    }

    #[test]
    fn nonblocking_stream_still_overlaps_null_when_peer_is_blocking() {
        let d = DeviceId(0);
        let bytes = 32u64 << 20;
        let k = KernelKind::GroupedMoeGemm {
            experts: 8,
            tokens_per_expert: 16,
            hidden: 2048,
            ff: 2048,
            dtype: DType::Fp16,
        };
        let run = |copy_on: StreamId| {
            let mut sim = Sim::new(h100());
            sim.set_stream_blocking(d, StreamId(1), true).unwrap();
            let w = sim.alloc(d, bytes, StreamId(0)).unwrap();
            let c = sim.alloc(d, bytes, copy_on).unwrap();
            enq(sim.memcpy_pinned_to_device(d, w, bytes, StreamId(0)));
            enq(sim.memcpy_pinned_to_device(d, c, bytes, copy_on));
            sim.synchronize().unwrap();
            let t0 = sim.clock_ns();
            enq(sim.kernel(d, k.clone(), &[w], &[w], StreamId::NULL));
            enq(sim.memcpy_pinned_to_device(d, c, bytes, copy_on));
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let serial = run(StreamId(1));
        let overlap = run(StreamId(2));
        assert!(
            serial > overlap,
            "non-blocking stream 2 must still overlap NULL; block1={serial} nb2={overlap}"
        );
    }

    #[test]
    fn host_pin_budget_rejects_second_mapped_alloc() {
        let mut sim = Sim::new(h100().restrict_pin(4096));
        let a = sim.alloc_host_mapped(4096).unwrap();
        assert_eq!(sim.pin_used(), 4096);
        match sim.alloc_host_mapped(4096) {
            Err(SimError::PinOom { need, free }) => {
                assert_eq!(need, 4096);
                assert_eq!(free, 0);
            }
            other => panic!("{other:?}"),
        }
        sim.free_host_pinned(a).unwrap();
        assert_eq!(sim.pin_used(), 0);
        let b = sim.alloc_host_mapped(4096).unwrap();
        sim.free_host_pinned(b).unwrap();
    }

    #[test]
    fn pageable_host_does_not_charge_pin_budget() {
        let mut sim = Sim::new(h100().restrict_pin(4096));
        let h = sim.alloc_host(8 << 20).unwrap();
        assert_eq!(sim.pin_used(), 0);
        match sim.host_register(h) {
            Err(SimError::PinOom { need, free }) => {
                assert_eq!(need, 8 << 20);
                assert_eq!(free, 4096);
            }
            other => panic!("{other:?}"),
        }
        sim.free_host(h).unwrap();
    }

    #[test]
    fn independent_stream_runs_during_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let cap = StreamId(0);
        let live = StreamId(1);
        let a = sim.alloc(d, 4096, cap).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, cap));
        sim.synchronize().unwrap();
        sim.begin_capture(d, cap).unwrap();
        let t0 = sim.clock_ns();
        enq(sim.kernel(d, KernelKind::other(1 << 20, 4096), &[a], &[a], live));
        sim.synchronize_stream(d, live).unwrap();
        assert!(sim.clock_ns() > t0);
        assert!(sim.query_stream(d, live).unwrap());
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 0);
    }

    #[test]
    fn capturing_stream_cannot_query_or_sync() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        sim.begin_capture(d, s).unwrap();
        let q = sim.query_stream(d, s).unwrap_err();
        match q {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let sy = sim.synchronize_stream(d, s).unwrap_err();
        match sy {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let node = sim.synchronize().unwrap_err();
        match node {
            SimError::Invalid { why } => assert!(why.contains("capture"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn event_wait_joins_capture_on_idle_stream() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let copy = StreamId(0);
        let compute = StreamId(1);
        let ev = EventId(1);
        let bytes = 4096u64;
        let resident = sim.alloc(d, bytes, copy).unwrap();
        let extra = sim.alloc(d, bytes, copy).unwrap();
        enq(sim.memcpy_pinned_to_device(d, resident, bytes, copy));
        sim.synchronize().unwrap();
        sim.create_event(ev).unwrap();
        sim.begin_capture(d, copy).unwrap();
        enq(sim.memcpy_pinned_to_device(d, extra, bytes, copy));
        enq(sim.record_event(d, ev, copy));
        enq(sim.wait_event(d, ev, compute));
        enq(sim.kernel(
            d,
            KernelKind::other(8, bytes),
            &[resident],
            &[resident],
            compute,
        ));
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 4);
        sim.instantiate_graph(g).unwrap();
        let n = sim.launch_graph(g, copy).unwrap();
        assert_eq!(n, 4);
        sim.synchronize().unwrap();
        assert!(sim.is_resident(extra, d).unwrap());
    }

    #[test]
    fn capture_fork_requires_idle_stream() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let copy = StreamId(0);
        let compute = StreamId(1);
        let ev = EventId(1);
        let a = sim.alloc(d, 4096, copy).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, copy));
        sim.synchronize().unwrap();
        sim.create_event(ev).unwrap();
        sim.begin_capture(d, copy).unwrap();
        enq(sim.record_event(d, ev, copy));
        enq(sim.kernel(d, KernelKind::other(1 << 30, 4096), &[a], &[a], compute));
        let err = sim.wait_event(d, ev, compute).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("idle"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
        sim.synchronize().unwrap();
    }

    #[test]
    fn wait_on_live_event_does_not_join_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let copy = StreamId(0);
        let live = StreamId(1);
        let ev = EventId(7);
        sim.create_event(ev).unwrap();
        enq(sim.record_event(d, ev, live));
        sim.synchronize().unwrap();
        sim.begin_capture(d, copy).unwrap();
        enq(sim.wait_event(d, ev, live));
        let g = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(g).unwrap(), 0);
    }

    #[test]
    fn capture_fork_rejects_other_device() {
        let mut sim = Sim::new(HardwareProfile::example_8xh100_nvlink());
        let d0 = DeviceId(0);
        let d1 = DeviceId(1);
        let s = StreamId(0);
        let ev = EventId(1);
        sim.create_event(ev).unwrap();
        sim.begin_capture(d0, s).unwrap();
        enq(sim.record_event(d0, ev, s));
        let err = sim.wait_event(d1, ev, s).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("same device"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn forked_graph_launch_overlaps_copy_and_compute() {
        let d = DeviceId(0);
        let copy = StreamId(0);
        let compute = StreamId(1);
        let bytes = 32u64 << 20;
        let k = KernelKind::other(1 << 40, bytes);
        let serial = {
            let mut sim = Sim::new(h100());
            let resident = sim.alloc(d, bytes, copy).unwrap();
            let extra = sim.alloc(d, bytes, copy).unwrap();
            let extra2 = sim.alloc(d, bytes, copy).unwrap();
            enq(sim.memcpy_pinned_to_device(d, resident, bytes, copy));
            sim.synchronize().unwrap();
            sim.begin_capture(d, copy).unwrap();
            enq(sim.memcpy_pinned_to_device(d, extra, bytes, copy));
            enq(sim.kernel(d, k.clone(), &[resident], &[resident], copy));
            enq(sim.memcpy_pinned_to_device(d, extra2, bytes, copy));
            let g = sim.end_capture().unwrap();
            sim.instantiate_graph(g).unwrap();
            let t0 = sim.clock_ns();
            let n = sim.launch_graph(g, copy).unwrap();
            assert_eq!(n, 3);
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        let forked = {
            let mut sim = Sim::new(h100());
            let resident = sim.alloc(d, bytes, copy).unwrap();
            let extra = sim.alloc(d, bytes, copy).unwrap();
            let extra2 = sim.alloc(d, bytes, copy).unwrap();
            enq(sim.memcpy_pinned_to_device(d, resident, bytes, copy));
            sim.synchronize().unwrap();
            sim.create_event(EventId(1)).unwrap();
            sim.begin_capture(d, copy).unwrap();
            enq(sim.memcpy_pinned_to_device(d, extra, bytes, copy));
            enq(sim.record_event(d, EventId(1), copy));
            enq(sim.wait_event(d, EventId(1), compute));
            enq(sim.kernel(d, k, &[resident], &[resident], compute));
            enq(sim.memcpy_pinned_to_device(d, extra2, bytes, copy));
            let g = sim.end_capture().unwrap();
            sim.instantiate_graph(g).unwrap();
            let t0 = sim.clock_ns();
            let n = sim.launch_graph(g, copy).unwrap();
            assert_eq!(n, 5);
            sim.synchronize().unwrap();
            sim.clock_ns().saturating_sub(t0)
        };
        assert!(
            serial > forked,
            "forked graph must overlap copy+compute; serial={serial} forked={forked}"
        );
    }

    #[test]
    fn update_graph_treats_stream_as_topology() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s0 = StreamId(0);
        let s1 = StreamId(1);
        let a = sim.alloc(d, 4096, s0).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s0));
        sim.synchronize().unwrap();
        sim.create_event(EventId(1)).unwrap();
        sim.begin_capture(d, s0).unwrap();
        enq(sim.record_event(d, EventId(1), s0));
        enq(sim.wait_event(d, EventId(1), s1));
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s1));
        let exec = sim.end_capture().unwrap();
        sim.instantiate_graph(exec).unwrap();
        sim.begin_capture(d, s0).unwrap();
        enq(sim.record_event(d, EventId(1), s0));
        enq(sim.wait_event(d, EventId(1), s0));
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s0));
        let src = sim.end_capture().unwrap();
        let err = sim.update_graph(exec, src).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("topology"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn child_graph_launch_during_capture_expands_on_parent_launch() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let child = sim.end_capture().unwrap();
        sim.instantiate_graph(child).unwrap();
        sim.begin_capture(d, s).unwrap();
        let recorded = sim.launch_graph(child, s).unwrap();
        assert_eq!(recorded, 1);
        let parent = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(parent).unwrap(), 1);
        sim.instantiate_graph(parent).unwrap();
        let t0 = sim.clock_ns();
        let n = sim.launch_graph(parent, s).unwrap();
        assert_eq!(n, 1);
        sim.synchronize().unwrap();
        assert!(sim.clock_ns() > t0);
    }

    #[test]
    fn child_graph_must_already_be_instantiated() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let child = sim.end_capture().unwrap();
        sim.begin_capture(d, s).unwrap();
        let err = sim.launch_graph(child, s).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("not instantiated"), "{why}"),
            other => panic!("{other:?}"),
        }
        let _g = sim.end_capture().unwrap();
    }

    #[test]
    fn independent_stream_launches_child_live_during_capture() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let cap = StreamId(0);
        let live = StreamId(1);
        let a = sim.alloc(d, 4096, cap).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, cap));
        sim.synchronize().unwrap();
        sim.begin_capture(d, cap).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], cap));
        let child = sim.end_capture().unwrap();
        sim.instantiate_graph(child).unwrap();
        sim.begin_capture(d, cap).unwrap();
        let t0 = sim.clock_ns();
        let n = sim.launch_graph(child, live).unwrap();
        assert_eq!(n, 1);
        sim.synchronize_stream(d, live).unwrap();
        assert!(sim.clock_ns() > t0);
        let parent = sim.end_capture().unwrap();
        assert_eq!(sim.graph_len(parent).unwrap(), 0);
    }

    #[test]
    fn destroy_child_graph_breaks_parent_launch() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let child = sim.end_capture().unwrap();
        sim.instantiate_graph(child).unwrap();
        sim.begin_capture(d, s).unwrap();
        let _n = sim.launch_graph(child, s).unwrap();
        let parent = sim.end_capture().unwrap();
        sim.destroy_graph(child).unwrap();
        let err = sim.launch_graph(parent, s).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("unknown graph"), "{why}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn clone_graph_recursively_clones_child_graphs() {
        let mut p = h100();
        for g in &mut p.gpus {
            g.graph_clone_ns = 9_000;
        }
        let mut sim = Sim::new(p);
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let child = sim.end_capture().unwrap();
        sim.instantiate_graph(child).unwrap();
        sim.begin_capture(d, s).unwrap();
        let _once = sim.launch_graph(child, s).unwrap();
        let _twice = sim.launch_graph(child, s).unwrap();
        let parent = sim.end_capture().unwrap();
        let t0 = sim.clock_ns();
        let cloned = sim.clone_graph(parent).unwrap();
        assert_eq!(sim.clock_ns(), t0.saturating_add(18_000));
        assert_eq!(sim.graph_len(cloned).unwrap(), 2);
        assert!(!sim.graph_instantiated(cloned).unwrap());
        sim.destroy_graph(child).unwrap();
        let err = sim.launch_graph(parent, s).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("unknown graph"), "{why}"),
            other => panic!("{other:?}"),
        }
        let n = sim.launch_graph(cloned, s).unwrap();
        assert_eq!(n, 2);
        sim.synchronize().unwrap();
    }

    #[test]
    fn update_graph_child_id_is_topology() {
        let mut sim = Sim::new(h100());
        let d = DeviceId(0);
        let s = StreamId(0);
        let a = sim.alloc(d, 4096, s).unwrap();
        let b = sim.alloc(d, 4096, s).unwrap();
        enq(sim.memcpy_pinned_to_device(d, a, 4096, s));
        enq(sim.memcpy_pinned_to_device(d, b, 4096, s));
        sim.synchronize().unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[a], &[a], s));
        let child_a = sim.end_capture().unwrap();
        sim.instantiate_graph(child_a).unwrap();
        sim.begin_capture(d, s).unwrap();
        enq(sim.kernel(d, KernelKind::other(8, 8), &[b], &[b], s));
        let child_b = sim.end_capture().unwrap();
        sim.instantiate_graph(child_b).unwrap();
        sim.begin_capture(d, s).unwrap();
        let _n = sim.launch_graph(child_a, s).unwrap();
        let exec = sim.end_capture().unwrap();
        sim.instantiate_graph(exec).unwrap();
        sim.begin_capture(d, s).unwrap();
        let _n = sim.launch_graph(child_b, s).unwrap();
        let src = sim.end_capture().unwrap();
        let err = sim.update_graph(exec, src).unwrap_err();
        match err {
            SimError::Invalid { why } => assert!(why.contains("topology"), "{why}"),
            other => panic!("{other:?}"),
        }
    }
}
