//! Structural GPU operations. Timing is derived from a [`crate::HardwareProfile`].

use crate::ids::{AllocId, DeviceId, EventId, OpId, StreamId};

/// Element type for roofline math. Maps onto a peak-FLOP field in the profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    /// IEEE binary16 / BF16 tensor-core class.
    Fp16,
    /// FP8 tensor-core class.
    Fp8,
    /// FP4 tensor-core class.
    Fp4,
    /// FP32 CUDA cores.
    Fp32,
}

/// Work a kernel represents. Not SASS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelKind {
    /// Dense GEMM, FLOPs = `2 * m * n * k`.
    Matmul {
        /// Rows of the output.
        m: u64,
        /// Columns of the output.
        n: u64,
        /// Inner dimension.
        k: u64,
        /// Accumulator / tensor-core flavor.
        dtype: DType,
    },
    /// Routed MoE grouped GEMM.
    GroupedMoeGemm {
        /// Experts touched this step.
        experts: u32,
        /// Tokens sent to each listed expert (uniform for the cost model).
        tokens_per_expert: u32,
        /// Model hidden size.
        hidden: u64,
        /// Expert FF inner size.
        ff: u64,
        /// GEMM dtype.
        dtype: DType,
    },
    /// Explicit FLOPs and bytes when the caller already lowered the op.
    Other {
        /// Arithmetic work.
        flops: u64,
        /// Bytes moved through HBM.
        bytes: u64,
    },
}

impl KernelKind {
    /// [`KernelKind::Other`] helper.
    #[must_use]
    pub fn other(flops: u64, bytes: u64) -> Self {
        Self::Other { flops, bytes }
    }

    /// Arithmetic intensity inputs for the roofline.
    #[must_use]
    pub fn flops_and_bytes(&self) -> (u64, u64) {
        match *self {
            Self::Matmul { m, n, k, .. } => {
                let flops = m.saturating_mul(n).saturating_mul(k).saturating_mul(2);
                let bytes = m
                    .saturating_mul(k)
                    .saturating_add(k.saturating_mul(n))
                    .saturating_add(m.saturating_mul(n))
                    .saturating_mul(2);
                (flops, bytes)
            }
            Self::GroupedMoeGemm {
                experts,
                tokens_per_expert,
                hidden,
                ff,
                ..
            } => {
                let e = u64::from(experts);
                let t = u64::from(tokens_per_expert);
                // gate, up, down: three GEMMs of (t x hidden x ff) class, 2 flop/MAC.
                let flops = e
                    .saturating_mul(t)
                    .saturating_mul(hidden)
                    .saturating_mul(ff)
                    .saturating_mul(6);
                let bytes = e
                    .saturating_mul(hidden)
                    .saturating_mul(ff)
                    .saturating_mul(6)
                    .saturating_add(e.saturating_mul(t).saturating_mul(hidden).saturating_mul(4));
                (flops, bytes)
            }
            Self::Other { flops, bytes } => (flops, bytes),
        }
    }

    /// Dtype used to pick the profile's peak FLOP rate.
    #[must_use]
    pub fn dtype(&self) -> DType {
        match *self {
            Self::Matmul { dtype, .. } | Self::GroupedMoeGemm { dtype, .. } => dtype,
            Self::Other { .. } => DType::Fp16,
        }
    }
}

/// Where a buffer lives for memcpy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    /// Pageable host memory. H2D/D2H is host-synchronous and pays
    /// [`crate::LinkProfile::pageable_permille`].
    Host,
    /// Page-locked host memory (`cudaMallocHost`). DMA at full link rate.
    /// Mapped host (`cudaHostAllocMapped`) is still this place for memcpy; a
    /// kernel may read it without a device copy.
    HostPinned,
    /// A GPU's HBM.
    Device(DeviceId),
}

impl Place {
    /// Device id when this place is HBM.
    #[must_use]
    pub fn device(self) -> Option<DeviceId> {
        match self {
            Self::Host | Self::HostPinned => None,
            Self::Device(d) => Some(d),
        }
    }

    /// Pageable host (needs a bounce through pinned staging on real hardware).
    #[must_use]
    pub fn is_pageable(self) -> bool {
        matches!(self, Self::Host)
    }
}

/// One asynchronous copy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemcpyOp {
    /// Source.
    pub src: Place,
    /// Destination.
    pub dst: Place,
    /// Object whose residency moves (or is replicated) on completion.
    pub alloc: AllocId,
    /// Payload bytes. Cost uses this, not a free full-link assumption.
    pub bytes: u64,
}

/// One submitted GPU primitive. PLAN's Kernel / Memcpy / Collective / Event /
/// Alloc / Free, plus `cudaMemsetAsync` and `cudaLaunchHostFunc`. Timing is not stored here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuOp {
    /// Stream-ordered device allocation. Capacity is reserved when the op starts.
    Alloc {
        /// Object created by this op.
        id: AllocId,
        /// Reserved bytes.
        bytes: u64,
    },
    /// Stream-ordered free. Illegal while a kernel lease is held.
    Free {
        /// Object dropped on this device.
        id: AllocId,
    },
    /// Asynchronous copy. Completion moves or replicates residency.
    Memcpy(MemcpyOp),
    /// Compute kernel. Reads/writes are leased until it completes.
    Kernel {
        /// Structural work (roofline inputs).
        kind: KernelKind,
        /// Allocations the kernel reads.
        reads: Vec<AllocId>,
        /// Allocations the kernel writes.
        writes: Vec<AllocId>,
    },
    /// Device-side fill (`cudaMemsetAsync`).
    Memset {
        /// Resident allocation to fill.
        id: AllocId,
        /// Bytes billed as an HBM write.
        bytes: u64,
    },
    /// Host callback (`cudaLaunchHostFunc`). Stream-ordered; does not occupy
    /// compute or copy engines.
    HostFunc,
    /// Record `event` after prior ops on this stream.
    EventRecord {
        /// Event id.
        event: EventId,
    },
    /// Later ops on this stream wait until `event` is recorded and complete.
    EventWait {
        /// Event id.
        event: EventId,
    },
    /// Ring allreduce (PLAN Collective). Each alloc must already be resident.
    AllReduce {
        /// Rank → allocation.
        parts: Vec<(DeviceId, AllocId)>,
        /// Payload bytes per hop.
        bytes: u64,
    },
}

/// One node in the compiled dependency DAG ([`GpuOp`] + stream + deps).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Operation {
    /// Submit id.
    pub id: OpId,
    /// Device the op was submitted on.
    pub device: DeviceId,
    /// Stream the op was submitted on.
    pub stream: StreamId,
    /// Primitive.
    pub kind: GpuOp,
    /// Stream-order and event-wait predecessors that must finish first.
    pub deps: Vec<OpId>,
    /// Whether the discrete-event engine has completed this op.
    pub done: bool,
    /// Cancelled before start, or failed transfer.
    pub cancelled: bool,
    /// Virtual time when the op was submitted (enqueued).
    pub submit_ns: u64,
    /// Virtual time when the op started. `None` if still queued or cancelled first.
    pub start_ns: Option<u64>,
    /// Virtual time when the op finished (including cancel / failed transfer).
    pub done_ns: Option<u64>,
}

impl Operation {
    /// Queue wait: `start_ns - submit_ns` once the op has started.
    #[must_use]
    pub fn queue_ns(&self) -> Option<u64> {
        Some(self.start_ns?.saturating_sub(self.submit_ns))
    }

    /// Run duration: `done_ns - start_ns` once the op has started and finished.
    #[must_use]
    pub fn duration_ns(&self) -> Option<u64> {
        Some(self.done_ns?.saturating_sub(self.start_ns?))
    }
}
