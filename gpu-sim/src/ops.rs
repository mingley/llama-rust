//! Structural GPU operations. Timing is derived from a [`crate::HardwareProfile`].

use crate::ids::{AllocId, CondId, DeviceId, EventId, GraphId, OpId, StreamId};

/// `cudaMemAdvise` hint on a [`crate::Sim::alloc_managed`] pointer.
///
/// Host-synchronous. Capture cannot include it. [`Self::SetReadMostly`] /
/// [`Self::UnsetReadMostly`] / [`Self::SetPreferredLocationHost`] ignore
/// `device` (CUDA `cudaCpuDeviceId`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemAdvise {
    /// Prefetch onto a second GPU keeps the first copy (read-only replicas).
    SetReadMostly,
    /// Later prefetch migrates again (unique location).
    UnsetReadMostly,
    /// Kernel on `device` may read without migrating (remote map).
    SetAccessedBy,
    /// Drop the remote mapping. The next kernel on `device` page-faults.
    UnsetAccessedBy,
    /// Keep the page at `device` on remote reads (`cudaMemAdviseSetPreferredLocation`).
    SetPreferredLocation,
    /// Prefer host; a kernel first-touch still migrates to the accessing GPU.
    SetPreferredLocationHost,
    /// Clear the preferred-location hint.
    UnsetPreferredLocation,
}

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
///
/// [`crate::Sim::graph_exec_memcpy_set_params`] patches this on an instantiated
/// memcpy node (`cudaGraphExecMemcpyNodeSetParams`). Pageable src/dst stay
/// illegal as graph params.
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
    /// Byte offset into [`Self::alloc`].
    ///
    /// Host-to-device into a VMM VA copies this span (`cudaMemcpy` of
    /// `ptr + offset`). `0` is the whole-object helpers
    /// ([`crate::Sim::memcpy_pinned_to_device`]). A VMM destination must have
    /// `[offset, offset+bytes)` mapped.
    pub offset: u64,
}

/// One kernel buffer: a whole allocation or a mapped VMM span.
///
/// [`Self::whole`] is `offset = 0`, `bytes = 0` (remainder of the alloc).
/// [`crate::Sim::kernel`] uses that. [`crate::Sim::kernel_bufs`] can name a
/// mapped page of a reserved VA so a paged KV working set need not cover the
/// whole pointer. [`crate::Sim::graph_exec_memset_set_params`] patches this on
/// an instantiated memset node (`cudaGraphExecMemsetNodeSetParams`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelBuf {
    /// Allocation (device, mapped-host, managed, or VMM VA).
    pub id: AllocId,
    /// Byte offset into `id`.
    pub offset: u64,
    /// Span length. `0` means from `offset` to the end of the allocation.
    pub bytes: u64,
}

impl KernelBuf {
    /// Whole allocation. [`crate::Sim::kernel`] uses this.
    #[must_use]
    pub fn whole(id: AllocId) -> Self {
        Self {
            id,
            offset: 0,
            bytes: 0,
        }
    }

    /// Mapped span `[offset, offset+bytes)` of `id` (vLLM KV-block analog).
    #[must_use]
    pub fn span(id: AllocId, offset: u64, bytes: u64) -> Self {
        Self { id, offset, bytes }
    }
}

/// `cudaHostNodeParams` for [`crate::Sim::graph_exec_host_set_params`].
///
/// Topology is "this is a host node." [`Self::fn_id`] and [`Self::user_data`]
/// are parameters (`cudaHostFn_t` / `userData`). Capture cannot include
/// SetParams. [`Default`] is the unnamed callback (`fn_id = 0`, `user_data = 0`)
/// used by [`crate::Sim::host_func`] / [`crate::Sim::graph_add_host_func`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostNodeParams {
    /// Callback identity (`cudaHostFn_t` analog). `0` is the unnamed default.
    pub fn_id: u32,
    /// Opaque `userData` pointer analog.
    pub user_data: u64,
}

/// `cudaKernelNodeParams` for [`crate::Sim::graph_exec_kernel_set_params`].
///
/// Topology (node index, cooperative flag, dependency edges) stays. Pointers
/// and [`KernelKind`] may change. Capture cannot include SetParams.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelNodeParams {
    /// Structural work (roofline inputs / kernel function analog).
    pub kind: KernelKind,
    /// Buffers the kernel reads.
    pub reads: Vec<KernelBuf>,
    /// Buffers the kernel writes.
    pub writes: Vec<KernelBuf>,
    /// Must match the existing node (`cudaLaunchCooperativeKernel` vs
    /// `cudaLaunchKernel`). Changing it is topology, not params.
    pub cooperative: bool,
}

/// `cudaLaunchAttributeProgrammaticStreamSerialization` / programmatic
/// dependent launch (PDL).
///
/// [`Self::wait`] is the secondary (`programmaticStreamSerializationAllowed`):
/// same-stream start may wait for the previous kernel's programmatic trigger
/// instead of its completion. [`Self::trigger`] means this kernel calls
/// `cudaTriggerProgrammaticLaunchCompletion` at
/// [`crate::GpuProfile::pdl_trigger_permille`] of its duration. Decode identity
/// stays both flags false. Capture records the flags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProgrammaticLaunch {
    /// Secondary: overlap the previous same-stream kernel after its PDL trigger.
    pub wait: bool,
    /// Primary: signal programmatic completion before the kernel finishes.
    pub trigger: bool,
}

/// `cudaLaunchAttributeProgrammaticEvent`.
///
/// [`crate::Sim::kernel_pdl_event`] records `event` when the kernel fires
/// [`ProgrammaticLaunch::trigger`] (or at kernel completion when trigger is
/// false). Other streams may [`crate::Sim::wait_event`] it and start before
/// the primary finishes. Same-stream later work still waits for completion
/// unless that work uses PDL wait. [`Self::external`] is
/// `cudaEventRecordExternal` (captured without forked-capture join). Decode
/// identity stays [`crate::Sim::kernel`] with no programmatic event. Capture
/// records the attribute on the kernel node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgrammaticEvent {
    /// Event recorded at the programmatic trigger (or kernel completion).
    pub event: EventId,
    /// `cudaEventRecordExternal` on the launch attribute.
    pub external: bool,
}

/// Live [`crate::Sim::kernel_pdl_event`] attributes: PDL plus an optional
/// programmatic event.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PdlLaunch {
    /// `cudaLaunchAttributeProgrammaticStreamSerialization`.
    pub pdl: ProgrammaticLaunch,
    /// `cudaLaunchAttributeProgrammaticEvent`, if any.
    pub event: Option<ProgrammaticEvent>,
}

impl PdlLaunch {
    /// Trigger at `pdl_trigger_permille` and record `event` (not External).
    #[must_use]
    pub fn trigger_event(event: EventId) -> Self {
        Self {
            pdl: ProgrammaticLaunch {
                wait: false,
                trigger: true,
            },
            event: Some(ProgrammaticEvent {
                event,
                external: false,
            }),
        }
    }
}

/// `cudaLaunchAttributeLaunchCompletionEvent`.
///
/// [`crate::Sim::kernel_launch_completion`] records `event` when the kernel
/// grid has been launched ([`Operation::start_ns`]), not when it finishes.
/// Other streams may [`crate::Sim::wait_event`] it and start copy or compute
/// while the primary is still running. Same-stream later work still waits for
/// completion. [`Self::external`] is `cudaEventRecordExternal`. Decode identity
/// stays [`crate::Sim::kernel`] with no launch-completion event. Capture
/// records the attribute on the kernel node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchCompletionEvent {
    /// Event recorded when the kernel starts.
    pub event: EventId,
    /// `cudaEventRecordExternal` on the launch attribute.
    pub external: bool,
}

/// `cudaAccessProperty` for [`AccessPolicyWindow`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AccessProperty {
    /// Normal L2 (no persist fill, no HBM discount).
    #[default]
    Normal,
    /// Streaming: accessed once; does not occupy persisting L2.
    Streaming,
    /// Persisting: reused lines stay in L2 until
    /// [`crate::Sim::reset_persisting_l2_cache`]. `miss` cannot be this.
    Persisting,
}

/// `cudaAccessPolicyWindow` / `cudaLaunchAttributeAccessPolicyWindow`.
///
/// [`crate::Sim::kernel_access_policy`] applies this window to one launch.
/// Persisting hits are billed at `1000 - GpuProfile::l2_persist_hit_permille`
/// of HBM after the first kernel has filled
/// [`crate::Sim::set_persisting_l2_cache_size`] (CUDA default size is 0).
/// [`KernelBuf::bytes`] `0` means the remainder of the allocation (not
/// CUDA `num_bytes = 0`; use [`None`] to clear the attribute). Decode identity
/// stays [`crate::Sim::kernel`] with no window. Capture records the attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccessPolicyWindow {
    /// VA range (`base_ptr` / `num_bytes`).
    pub buf: KernelBuf,
    /// `hitRatio` as ‰ (`1000` = the whole window). Must be `<= 1000`.
    pub hit_ratio_permille: u16,
    /// Property for expected hits.
    pub hit: AccessProperty,
    /// Property for misses. Cannot be [`AccessProperty::Persisting`].
    pub miss: AccessProperty,
}

impl AccessPolicyWindow {
    /// Persisting hits, streaming misses, full window (`hitRatio = 1.0`).
    #[must_use]
    pub fn persisting(buf: KernelBuf) -> Self {
        Self {
            buf,
            hit_ratio_permille: 1000,
            hit: AccessProperty::Persisting,
            miss: AccessProperty::Streaming,
        }
    }
}

/// `cudaLaunchMemSyncDomain`.
///
/// Logical domain for [`crate::Sim::kernel_with`]. [`Self::Default`] is CUDA
/// domain 0. [`Self::Remote`] is intended for communication kernels (NCCL
/// tags Remote). Physical id is [`MemSyncDomainMap::physical`]. Decode identity
/// stays [`Self::Default`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MemSyncDomain {
    /// `cudaLaunchMemSyncDomainDefault` (physical id from the map's `default`).
    #[default]
    Default,
    /// `cudaLaunchMemSyncDomainRemote` (physical id from the map's `remote`).
    Remote,
}

/// `cudaLaunchMemSyncDomainMap`.
///
/// Maps logical [`MemSyncDomain`] onto a physical id in
/// `0..GpuProfile::mem_sync_domain_count`. CUDA default on Hopper (`count > 1`)
/// is default→0, remote→1. Pre-Hopper `count == 1` maps both to 0.
/// [`crate::Sim::set_stream_mem_sync_domain_map`] is the stream attribute.
/// Graph replay uses the node's map, not the launch stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemSyncDomainMap {
    /// Physical id for [`MemSyncDomain::Default`].
    pub default: u8,
    /// Physical id for [`MemSyncDomain::Remote`].
    pub remote: u8,
}

impl MemSyncDomainMap {
    /// CUDA default map for `cudaDevAttrMemSyncDomainCount`.
    #[must_use]
    pub fn identity(count: u8) -> Self {
        if count <= 1 {
            Self {
                default: 0,
                remote: 0,
            }
        } else {
            Self {
                default: 0,
                remote: 1,
            }
        }
    }

    /// Physical domain id for `domain`.
    #[must_use]
    pub fn physical(self, domain: MemSyncDomain) -> u8 {
        match domain {
            MemSyncDomain::Default => self.default,
            MemSyncDomain::Remote => self.remote,
        }
    }
}

impl Default for MemSyncDomainMap {
    fn default() -> Self {
        Self::identity(2)
    }
}

/// Packed `cudaLaunchKernelEx` / graph kernel-node attributes.
///
/// [`crate::Sim::kernel_with`] applies these on one submit so PDL, an
/// access-policy window, and a mem-sync domain can share a launch (7 arguments
/// including `self`). Decode identity stays [`crate::Sim::kernel`] ([`Default`]:
/// no cooperative, no PDL, no window, inherit stream mem-sync, no cluster,
/// Default carveout).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KernelAttrs {
    /// `cudaLaunchCooperativeKernel`.
    pub cooperative: bool,
    /// `cudaLaunchAttributeProgrammaticStreamSerialization`.
    pub pdl: ProgrammaticLaunch,
    /// `cudaLaunchAttributeAccessPolicyWindow`.
    pub access_policy: Option<AccessPolicyWindow>,
    /// `cudaLaunchAttributeMemSyncDomain`. `None` inherits the stream.
    pub mem_sync_domain: Option<MemSyncDomain>,
    /// `cudaLaunchAttributeMemSyncDomainMap`. `None` inherits the stream.
    pub mem_sync_map: Option<MemSyncDomainMap>,
    /// `cudaLaunchAttributeClusterDimension`. `None` is a non-cluster launch.
    pub cluster: Option<ClusterDim>,
    /// `cudaLaunchAttributeClusterSchedulingPolicyPreference`.
    pub cluster_policy: ClusterSchedulingPolicy,
    /// `cudaLaunchAttributePreferredClusterDimension`. `None` uses [`Self::cluster`].
    pub preferred_cluster: Option<ClusterDim>,
    /// `cudaLaunchAttributePreferredSharedMemoryCarveout`.
    pub carveout: SharedMemCarveout,
}

/// `cudaFuncCache` / `cudaSharedmemCarveout` preference
/// (`cudaLaunchAttributePreferredSharedMemoryCarveout`).
///
/// [`Self::MaxShared`] occupies every Hyper-Q slot so leftover kernels cannot
/// overlap. [`Self::Default`] and [`Self::MaxL1`] keep current occupancy.
/// Decode identity stays [`Self::Default`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SharedMemCarveout {
    /// `cudaSharedmemCarveoutDefault` (`-1`).
    #[default]
    Default,
    /// `cudaSharedmemCarveoutMaxL1` (`0`): prefer L1 over shared.
    MaxL1,
    /// `cudaSharedmemCarveoutMaxShared` (`100`): prefer shared over L1.
    MaxShared,
}

impl SharedMemCarveout {
    /// Max-shared kernels occupy the whole GPU.
    #[must_use]
    pub fn occupies_all_slots(self) -> bool {
        matches!(self, Self::MaxShared)
    }
}

/// `cudaClusterSchedulingPolicy` (`cudaLaunchAttributeClusterSchedulingPolicyPreference`).
///
/// Spread occupies every Hyper-Q slot so leftover kernels cannot overlap a
/// clustered launch. Default and LoadBalancing occupy
/// `min(blocks, compute_slots)` (the current cluster occupancy). Decode
/// identity stays [`Self::Default`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClusterSchedulingPolicy {
    /// `cudaClusterSchedulingPolicyDefault`.
    #[default]
    Default,
    /// `cudaClusterSchedulingPolicySpread`: spread cluster blocks across SMs.
    Spread,
    /// `cudaClusterSchedulingPolicyLoadBalancing`: hardware may pack blocks.
    LoadBalancing,
}

/// `cudaLaunchAttributeClusterDimension` (`clusterDim`).
///
/// All three dimensions must be `>= 1`. Product is the cluster block count and
/// must be `<= GpuProfile::max_blocks_per_cluster`. Sizes above
/// [`crate::GpuProfile::portable_cluster_size`] also need
/// [`crate::Sim::set_non_portable_cluster_size_allowed`]. Decode identity stays
/// [`None`] (not a cluster). Capture records the attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClusterDim {
    /// Blocks in X.
    pub x: u32,
    /// Blocks in Y.
    pub y: u32,
    /// Blocks in Z.
    pub z: u32,
}

impl ClusterDim {
    /// One-dimensional cluster of `n` blocks (`{n,1,1}`).
    #[must_use]
    pub fn x(n: u32) -> Self {
        Self { x: n, y: 1, z: 1 }
    }

    /// Cluster block count. `None` if a dimension is 0 or the product overflows.
    #[must_use]
    pub fn blocks(self) -> Option<u32> {
        if self.x == 0 || self.y == 0 || self.z == 0 {
            return None;
        }
        self.x
            .checked_mul(self.y)
            .and_then(|p| p.checked_mul(self.z))
    }

    /// True when each axis of `self` is a positive integer multiple of `min`.
    #[must_use]
    pub fn multiple_of(self, min: Self) -> bool {
        min.x > 0
            && min.y > 0
            && min.z > 0
            && self.x.is_multiple_of(min.x)
            && self.y.is_multiple_of(min.y)
            && self.z.is_multiple_of(min.z)
    }
}

/// `cudaStreamAttachMemAsync` flags (`cudaMemAttachGlobal` / `Host` / `Single`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemAttach {
    /// Accessible from every stream (`cudaMemAttachGlobal`). Default for
    /// [`crate::Sim::alloc_managed`].
    Global,
    /// CPU-exclusive until a later attach (`cudaMemAttachHost`). Device kernels
    /// and device prefetch fail [`crate::SimError::Invalid`].
    Host,
    /// Only the attach stream may use it from the device (`cudaMemAttachSingle`).
    /// Illegal on [`crate::StreamId::NULL`].
    Single,
}

/// `CU_STREAM_WAIT_VALUE_*` compare for [`GpuOp::WaitValue`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitValueCmp {
    /// Wait until `*addr == value` (`CU_STREAM_WAIT_VALUE_EQ`).
    Eq,
    /// Unsigned `*addr >= value` (`CU_STREAM_WAIT_VALUE_GEQ`).
    Geq,
    /// Wait until `(*addr & value) != 0` (`CU_STREAM_WAIT_VALUE_AND`).
    And,
    /// Wait until `(*addr & value) == value` (`CU_STREAM_WAIT_VALUE_NOR`).
    Nor,
}

impl WaitValueCmp {
    /// Whether `loc` (already masked to 32 or 64 bits) satisfies this compare.
    #[must_use]
    pub fn matches(self, loc: u64, value: u64) -> bool {
        match self {
            Self::Eq => loc == value,
            Self::Geq => loc >= value,
            Self::And => (loc & value) != 0,
            Self::Nor => (loc & value) == value,
        }
    }
}

/// One `cuStreamBatchMemOp` item (`cudaGraphAddBatchMemOpNode`).
///
/// [`crate::Sim::graph_add_batch_mem_op`] / [`crate::Sim::batch_mem_op`] pack a
/// non-empty vector into one [`GpuOp::BatchMem`] node. Single-item live
/// [`crate::Sim::write_value64`] / [`crate::Sim::wait_value64`] stay
/// [`GpuOp::WriteValue`] / [`GpuOp::WaitValue`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchMemOp {
    /// [`GpuOp::WriteValue`].
    Write {
        /// Allocation holding the mailbox word.
        id: AllocId,
        /// Byte offset (4-byte aligned for 32-bit, 8-byte for 64-bit).
        offset: u64,
        /// Value written on complete (low 32 bits when `bits32`).
        value: u64,
        /// `cuStreamWriteValue32` when true.
        bits32: bool,
    },
    /// [`GpuOp::WaitValue`].
    Wait {
        /// Allocation holding the mailbox word.
        id: AllocId,
        /// Byte offset (4-byte aligned for 32-bit, 8-byte for 64-bit).
        offset: u64,
        /// Compare operand (low 32 bits when `bits32`).
        value: u64,
        /// `cuStreamWaitValue32` when true.
        bits32: bool,
        /// Compare mode.
        cmp: WaitValueCmp,
    },
}

/// One submitted GPU primitive. PLAN's Kernel / Memcpy / Collective / Event /
/// Alloc / Free, plus `cudaMemsetAsync`, `cudaLaunchHostFunc`, stream attach,
/// empty graph nodes, nested [`Self::ChildGraph`], conditional IF / WHILE /
/// SWITCH / [`Self::SetConditional`], wait/write-value, and multi-item
/// [`Self::BatchMem`] (`cuStreamBatchMemOp`). Timing is not stored here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuOp {
    /// Stream-ordered device allocation (`cudaMallocAsync`). Capacity is
    /// reserved when the op starts. During stream capture this is a graph mem
    /// alloc node.
    Alloc {
        /// Object created by this op.
        id: AllocId,
        /// Reserved bytes.
        bytes: u64,
    },
    /// Stream-ordered free (`cudaFreeAsync`). Illegal while a kernel lease is
    /// held. During stream capture this is a graph mem free node.
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
        /// Buffers the kernel reads (whole alloc or a mapped VMM span).
        reads: Vec<KernelBuf>,
        /// Buffers the kernel writes (whole alloc or a mapped VMM span).
        writes: Vec<KernelBuf>,
        /// `cudaLaunchCooperativeKernel`: the grid must occupy the whole GPU
        /// ([`crate::GpuProfile::compute_slots`]) so leftover kernels cannot
        /// Hyper-Q overlap it. Capture is allowed (CUDA 11+).
        cooperative: bool,
    },
    /// Device-side fill (`cudaMemsetAsync`).
    Memset {
        /// Allocation to fill.
        id: AllocId,
        /// Byte offset into `id`.
        offset: u64,
        /// Bytes billed as an HBM write and the mapped span that must be resident.
        bytes: u64,
    },
    /// Host callback (`cudaLaunchHostFunc`). Stream-ordered; does not occupy
    /// compute or copy engines. `fn_id` / `user_data` are
    /// [`HostNodeParams`] (parameters, not topology).
    HostFunc {
        /// Callback identity (`cudaHostFn_t` analog). Default `0`.
        fn_id: u32,
        /// Opaque `userData`. Default `0`.
        user_data: u64,
    },
    /// Empty graph node (`cudaGraphAddEmptyNode`). Completes immediately;
    /// does not occupy compute or copy engines. Capture cannot include it
    /// (add on an uninstantiated graph). A join/fork with no work.
    Empty,
    /// `cudaStreamAttachMemAsync`. Stream-ordered; cannot be captured.
    Attach {
        /// Managed allocation.
        id: AllocId,
        /// Global / Host / Single (Single uses this op's stream).
        flags: MemAttach,
    },
    /// Record `event` after prior ops on this stream.
    EventRecord {
        /// Event id.
        event: EventId,
        /// `cudaEventRecordExternal`: captured without putting the event in the
        /// forked-capture join set; live waiters do not join the graph.
        external: bool,
    },
    /// Later ops on this stream wait until `event` is recorded and complete.
    EventWait {
        /// Event id.
        event: EventId,
        /// `cudaEventWaitExternal`: captured without forking; graph waits do not
        /// depend on the same graph's record of this event.
        external: bool,
    },
    /// Ring allreduce (PLAN Collective). Each alloc must already be resident.
    AllReduce {
        /// Rank → allocation.
        parts: Vec<(DeviceId, AllocId)>,
        /// Payload bytes per hop.
        bytes: u64,
    },
    /// Nested graph (`cudaGraphLaunch` while capturing). Expanded at parent
    /// launch; never a live [`crate::Sim::operations`] node.
    ChildGraph {
        /// Instantiated exec launched as a child node.
        graph: GraphId,
    },
    /// Conditional IF node (`cudaGraphNodeTypeConditional` / `If`). Expanded at
    /// parent launch. Body ops skip at start when the handle is `0`.
    If {
        /// Handle created with [`crate::Sim::graph_conditional_create`].
        handle: CondId,
        /// Body graph returned by [`crate::Sim::graph_add_if`].
        body: GraphId,
    },
    /// Device `cudaGraphSetConditional` (captured or live). Writes the handle
    /// when the op starts. Does not occupy compute or copy engines.
    SetConditional {
        /// Handle to write.
        handle: CondId,
        /// Non-zero runs a later IF/WHILE body; SWITCH uses the value as a
        /// branch index (`0 .. n-1`).
        value: u32,
    },
    /// Conditional WHILE node (`cudaGraphCondTypeWhile`). Expanded at parent
    /// launch. Each iteration's body ops skip at start when the handle is `0`.
    /// A body that leaves the handle non-zero is Invalid after 64 iterations.
    While {
        /// Handle created with [`crate::Sim::graph_conditional_create`].
        handle: CondId,
        /// Body graph returned by [`crate::Sim::graph_add_while`].
        body: GraphId,
    },
    /// Runtime WHILE iteration fence. Never a `cudaGraphAdd*` node.
    WhileTick {
        /// Handle sampled after this iteration.
        handle: CondId,
        /// Body to enqueue again when the handle is still non-zero.
        body: GraphId,
        /// 1-based iteration count (cap 64).
        iter: u32,
    },
    /// Conditional SWITCH node (`cudaGraphCondTypeSwitch`). Expanded at parent
    /// launch. Branch `i` runs when the handle equals `i`; out of range skips
    /// every body.
    Switch {
        /// Handle created with [`crate::Sim::graph_conditional_create`].
        handle: CondId,
        /// Body graphs returned by [`crate::Sim::graph_add_switch`].
        bodies: Vec<GraphId>,
    },
    /// `cuStreamWriteValue32` / `WriteValue64`. The mailbox updates when this
    /// op **completes**, not when it starts. Does not occupy compute or copy
    /// engines. Capture records a batch-mem-op node. Kernel / memset / memcpy
    /// stores to this address are not modeled.
    WriteValue {
        /// Allocation holding the mailbox word.
        id: AllocId,
        /// Byte offset (aligned to 4 or 8).
        offset: u64,
        /// Value written on complete.
        value: u64,
        /// 32-bit store when true (high bits of a prior 64-bit write stay).
        bits32: bool,
    },
    /// Multi-item `cuStreamBatchMemOp` / `cudaGraphAddBatchMemOpNode`.
    ///
    /// Items run in order inside this one stream op (no compute or copy
    /// occupancy). A wait sees writes **earlier in this vector** via an
    /// overlay; it does not see later writes in the same batch. All writes
    /// commit to the mailbox when the op **completes**. Capture records one
    /// node. Empty is Invalid.
    BatchMem {
        /// Wait and write items in CUDA batch order.
        ops: Vec<BatchMemOp>,
    },
    /// `cuStreamWaitValue32` / `WaitValue64`. Stays pending until the mailbox
    /// compare matches. Unwritten locations read as 0 (flag-memory analog, not
    /// uninitialized `cudaMalloc`). No compute or copy occupancy. Capture
    /// records a batch-mem-op node.
    WaitValue {
        /// Allocation holding the mailbox word.
        id: AllocId,
        /// Byte offset (aligned to 4 or 8).
        offset: u64,
        /// Compare operand.
        value: u64,
        /// 32-bit compare when true (mask `0xFFFF_FFFF`).
        bits32: bool,
        /// `CU_STREAM_WAIT_VALUE_*` mode.
        cmp: WaitValueCmp,
    },
    /// Device-side `cudaGraphLaunch` (`cudaGraphInstantiateFlagDeviceLaunch`).
    ///
    /// Occupies one compute slot for `graph_launch_ns`. When it completes the
    /// exec body is enqueued on this stream. Never a `cudaGraphAdd*` node;
    /// capture is refused.
    DeviceLaunch {
        /// Instantiated exec launched from the device.
        graph: GraphId,
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

/// `cudaStreamUpdateCaptureDependencies` flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureDepOp {
    /// Union with the current pending set (`cudaStreamAddCaptureDependencies`).
    Add,
    /// Replace the current pending set (`cudaStreamSetCaptureDependencies`).
    Set,
}

/// `cudaStreamCaptureMode` for [`crate::Sim::begin_capture_with_mode`].
///
/// [`Self::Relaxed`] is the default for [`crate::Sim::begin_capture`]:
/// independent streams may run live work, and `wait_event` of a captured
/// record still joins an idle stream (forked capture). [`Self::Global`] and
/// [`Self::ThreadLocal`] are the same in this single-threaded VM: submits on
/// a stream not in the capture set are Invalid (`stream not capturing`),
/// except a joining wait.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StreamCaptureMode {
    /// `cudaStreamCaptureModeGlobal`. Same as [`Self::ThreadLocal`] here.
    Global,
    /// `cudaStreamCaptureModeThreadLocal`. Uncaptured-stream submits are Invalid.
    ThreadLocal,
    /// `cudaStreamCaptureModeRelaxed`. Uncaptured-stream submits run live.
    ///
    /// Default for [`crate::Sim::begin_capture`]. Forked capture still joins.
    #[default]
    Relaxed,
}

impl StreamCaptureMode {
    /// Independent streams may run live CUDA work during this capture.
    #[must_use]
    pub fn live_uncaptured(self) -> bool {
        matches!(self, Self::Relaxed)
    }
}

/// `cudaGraphInstantiateFlags` bit names (`cudaGraphExecGetFlags`).
pub struct GraphInstantiateFlags;

impl GraphInstantiateFlags {
    /// `cudaGraphInstantiateFlagAutoFreeOnLaunch`.
    pub const AUTO_FREE_ON_LAUNCH: u32 = 1;
    /// `cudaGraphInstantiateFlagUpload`: host-sync upload during instantiate.
    pub const UPLOAD: u32 = 2;
    /// `cudaGraphInstantiateFlagDeviceLaunch`: [`crate::Sim::device_launch_graph`]
    /// is legal after upload. Host [`crate::Sim::launch_graph`] stays legal.
    /// Mem alloc/free, events, child graphs, conditionals, and host nodes are
    /// Invalid.
    pub const DEVICE_LAUNCH: u32 = 4;
    /// `cudaGraphInstantiateFlagUseNodePriority`: recorded kernels keep the
    /// priority snapshotted at add/capture instead of the launch stream.
    pub const USE_NODE_PRIORITY: u32 = 8;
}

/// `cudaGraphInstantiateResult` from [`crate::Sim::instantiate_graph_with_params`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GraphInstantiateResult {
    /// `cudaGraphInstantiateSuccess`.
    #[default]
    Success,
    /// `cudaGraphInstantiateError`.
    Error,
    /// `cudaGraphInstantiateInvalidStructure`.
    InvalidStructure,
    /// `cudaGraphInstantiateNodeOperationNotSupported`.
    NodeOperationNotSupported,
    /// `cudaGraphInstantiateMultipleDevicesNotSupported`.
    MultipleDevicesNotSupported,
}

/// `cudaGraphInstantiateParams` for [`crate::Sim::instantiate_graph_with_params`].
///
/// `flags` are inputs. `err_node` / `result` are outputs (filled even on
/// `Err`). Decode identity stays [`crate::Sim::instantiate_graph`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GraphInstantiateParams {
    /// `cudaGraphInstantiateFlags`.
    pub flags: u32,
    /// First node that made instantiate fail, when known.
    pub err_node: Option<usize>,
    /// CUDA result enum.
    pub result: GraphInstantiateResult,
}

/// `cudaGraphExecUpdateResult` from [`crate::Sim::update_graph_with_info`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GraphExecUpdateResult {
    /// `cudaGraphExecUpdateSuccess`.
    #[default]
    Success,
    /// `cudaGraphExecUpdateError`.
    Error,
    /// `cudaGraphExecUpdateErrorTopologyChanged`.
    TopologyChanged,
    /// `cudaGraphExecUpdateErrorNodeTypeChanged`.
    NodeTypeChanged,
    /// `cudaGraphExecUpdateErrorFunctionChanged`.
    FunctionChanged,
    /// `cudaGraphExecUpdateErrorParametersChanged`.
    ParametersChanged,
    /// `cudaGraphExecUpdateErrorNotSupported`.
    NotSupported,
    /// `cudaGraphExecUpdateErrorUnsupportedFunctionChange`.
    UnsupportedFunctionChange,
    /// `cudaGraphExecUpdateErrorAttributesChanged`.
    AttributesChanged,
    /// `cudaGraphExecUpdateErrorDependenciesChanged` (`errorFromNode` edge).
    DependenciesChanged,
}

/// `cudaGraphExecUpdateResultInfo` for [`crate::Sim::update_graph_with_info`].
///
/// Filled even on `Err`. [`crate::Sim::update_graph`] uses this path and
/// keeps the same `why` strings. Decode identity stays
/// [`crate::Sim::update_graph`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphExecUpdateResultInfo {
    /// CUDA result enum.
    pub result: GraphExecUpdateResult,
    /// `errorNode`: source node that failed, or the to-node of a changed edge.
    pub error_node: Option<usize>,
    /// `errorFromNode`: exec index of a node-local mismatch, or the from-node
    /// of a changed dependency edge.
    pub error_from_node: Option<usize>,
}

/// `cudaUserObjectFlags` for [`crate::Sim::user_object_create`].
pub struct UserObjectFlags;

impl UserObjectFlags {
    /// `cudaUserObjectNoDestructorSync`. Required; the destroy callback is not
    /// device-synchronized.
    pub const NO_DESTRUCTOR_SYNC: u32 = 1;
}

/// `cudaGraphUserObjectRetain` flags for [`crate::Sim::graph_retain_user_object`].
pub struct GraphUserObjectFlags;

impl GraphUserObjectFlags {
    /// `cudaGraphUserObjectMove`: transfer one caller reference to the graph.
    pub const MOVE: u32 = 1;
}

/// `cudaGraphMemAttributeType` for [`crate::Sim::graph_mem_get`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphMemAttr {
    /// Live graph-mem alloc bytes on the device (`UsedMemCurrent`).
    UsedMemCurrent,
    /// High-water of [`Self::UsedMemCurrent`].
    UsedMemHigh,
    /// Reserved graph-mem bytes. Equal to used: graph allocs charge device HBM
    /// (async pool / OS) directly, so [`crate::Sim::graph_mem_trim`] has nothing
    /// unused to return.
    ReservedMemCurrent,
    /// High-water of [`Self::ReservedMemCurrent`].
    ReservedMemHigh,
}

/// Active stream capture (`cudaStreamGetCaptureInfo`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamCaptureInfo {
    /// Graph being captured into (`cudaStreamBeginCapture` / `ToGraph`).
    pub graph: GraphId,
    /// Capture origin `(device, stream)`.
    pub origin: (DeviceId, StreamId),
    /// Extra deps for the next captured node on this stream.
    ///
    /// Indices are existing graph nodes, then this-session nodes at
    /// `graph_len + i`. Empty until [`crate::Sim::stream_update_capture_dependencies`].
    pub pending_deps: Vec<usize>,
    /// Mode this capture started with (`cudaStreamGetCaptureInfo` status).
    pub mode: StreamCaptureMode,
}

/// `cudaGraphNodeGetType` tag for one graph node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphNodeKind {
    /// `cudaGraphAddKernelNode` / captured kernel.
    Kernel,
    /// `cudaGraphAddMemcpyNode` / captured memcpy.
    Memcpy,
    /// `cudaGraphAddMemsetNode` / captured memset.
    Memset,
    /// `cudaGraphAddHostNode`.
    Host,
    /// `cudaGraphAddEmptyNode`.
    Empty,
    /// `cudaGraphAddEventRecordNode`.
    EventRecord,
    /// `cudaGraphAddEventWaitNode`.
    EventWait,
    /// `cudaGraphAddChildGraphNode` / `launch_graph` during capture.
    ChildGraph,
    /// `cudaGraphAddMemAllocNode` / captured `cudaMallocAsync`.
    Alloc,
    /// `cudaGraphAddMemFreeNode` / captured `cudaFreeAsync`.
    Free,
    /// Captured ring allreduce (not a `cudaGraphAdd*` node).
    AllReduce,
    /// Captured `cudaStreamAttachMemAsync` (illegal to capture; defensive).
    Attach,
    /// Conditional IF node (`cudaGraphNodeTypeConditional`).
    If,
    /// Captured [`crate::GpuOp::SetConditional`] (`cudaGraphSetConditional`).
    SetConditional,
    /// Conditional WHILE node (`cudaGraphCondTypeWhile`).
    While,
    /// Runtime [`crate::GpuOp::WhileTick`].
    WhileTick,
    /// Conditional SWITCH node (`cudaGraphCondTypeSwitch`).
    Switch,
    /// `cudaGraphAddBatchMemOpNode` / captured wait-value or write-value.
    BatchMemOp,
    /// Live [`crate::GpuOp::DeviceLaunch`] (not a `cudaGraphAdd*` node).
    DeviceLaunch,
}
