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

/// `cudaMemRangeAttribute` for [`crate::Sim::mem_range_get_attribute`] /
/// [`crate::Sim::mem_range_get_attributes`].
///
/// This VM tracks advice per allocation, not per byte range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemRangeAttr {
    /// `cudaMemRangeAttributeReadMostly`.
    ReadMostly,
    /// `cudaMemRangeAttributePreferredLocation`.
    PreferredLocation,
    /// `cudaMemRangeAttributeAccessedBy`.
    AccessedBy,
    /// `cudaMemRangeAttributeLastPrefetchLocation`.
    LastPrefetchLocation,
    /// `cudaMemRangeAttributePreferredLocationType`.
    PreferredLocationType,
    /// `cudaMemRangeAttributePreferredLocationId`.
    PreferredLocationId,
    /// `cudaMemRangeAttributeLastPrefetchLocationType`.
    LastPrefetchLocationType,
    /// `cudaMemRangeAttributeLastPrefetchLocationId`.
    LastPrefetchLocationId,
}

/// Value of [`crate::Sim::mem_range_get_attribute`] /
/// [`crate::Sim::mem_range_get_attributes`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemRangeAttrValue {
    /// [`MemRangeAttr::ReadMostly`].
    ReadMostly(bool),
    /// [`MemRangeAttr::PreferredLocation`]. `None` is unset (`cudaInvalidDeviceId`).
    /// [`Place::Host`] is `cudaCpuDeviceId`. [`Place::Device`] is that GPU.
    PreferredLocation(Option<Place>),
    /// [`MemRangeAttr::AccessedBy`]. Device ids that may remote-read.
    AccessedBy(Vec<DeviceId>),
    /// [`MemRangeAttr::LastPrefetchLocation`]. `None` is never prefetched
    /// (`cudaInvalidDeviceId`). [`Place::Host`] is `cudaCpuDeviceId`.
    LastPrefetchLocation(Option<Place>),
    /// [`MemRangeAttr::PreferredLocationType`].
    PreferredLocationType(MemLocationType),
    /// [`MemRangeAttr::PreferredLocationId`]. Device ordinal when the type is
    /// [`MemLocationType::Device`]; `0` (ignored) otherwise. Host NUMA is not
    /// modeled.
    PreferredLocationId(u32),
    /// [`MemRangeAttr::LastPrefetchLocationType`].
    LastPrefetchLocationType(MemLocationType),
    /// [`MemRangeAttr::LastPrefetchLocationId`]. Device ordinal when the type is
    /// [`MemLocationType::Device`]; `0` (ignored) otherwise.
    LastPrefetchLocationId(u32),
}

/// `cudaMemLocationType` for [`MemRangeAttr::PreferredLocationType`] /
/// [`LastPrefetchLocationType`](MemRangeAttr::LastPrefetchLocationType).
///
/// Host NUMA / NUMA-current / Invisible are not modeled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemLocationType {
    /// `cudaMemLocationTypeInvalid` / `None` (`0`). Unset preferred location
    /// or never prefetched.
    Invalid,
    /// `cudaMemLocationTypeDevice` (`1`).
    Device,
    /// `cudaMemLocationTypeHost` (`2`).
    Host,
}

impl MemLocationType {
    /// Map a stored [`Place`]. [`None`] is [`Self::Invalid`]. Host-pinned
    /// prefetch dest is [`Self::Host`]. Host NUMA is not modeled.
    #[must_use]
    pub fn from_place(place: Option<Place>) -> Self {
        match place {
            None => Self::Invalid,
            Some(Place::Device(_)) => Self::Device,
            Some(Place::Host | Place::HostPinned) => Self::Host,
        }
    }

    /// CUDA `int`: [`Self::Invalid`] `0`, [`Self::Device`] `1`, [`Self::Host`] `2`.
    #[must_use]
    pub fn to_cuda(self) -> i32 {
        match self {
            Self::Invalid => 0,
            Self::Device => 1,
            Self::Host => 2,
        }
    }

    /// Device ordinal when `place` is [`Place::Device`]; `0` (ignored) otherwise.
    #[must_use]
    pub fn id_from_place(place: Option<Place>) -> u32 {
        match place {
            Some(Place::Device(d)) => u32::from(d.0),
            _ => 0,
        }
    }
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
/// illegal as graph params. [`Self::packed_1d`] is
/// `cudaGraphAddMemcpyNode1D` / `MemcpyNodeSetParams1D`.
///
/// [`Self::height`] `0` or `1` is `cudaMemcpyAsync` of [`Self::bytes`].
/// `height > 1` and [`Self::depth`] `<= 1` is `cudaMemcpy2DAsync`:
/// [`Self::bytes`] is the row width, billed payload is `width * height`
/// (pitch padding is not transferred). [`Self::depth`] `> 1` is
/// `cudaMemcpy3DAsync`: billed payload is `width * height * depth`
/// (row and slice padding are not transferred).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemcpyOp {
    /// Source.
    pub src: Place,
    /// Destination.
    pub dst: Place,
    /// Object whose residency moves (or is replicated) on completion.
    pub alloc: AllocId,
    /// Payload bytes for a 1D copy, or row width for [`Self::height`] `> 1`.
    pub bytes: u64,
    /// Byte offset into [`Self::alloc`].
    ///
    /// Host-to-device into a VMM VA copies this span (`cudaMemcpy` of
    /// `ptr + offset`). `0` is the whole-object helpers
    /// ([`crate::Sim::memcpy_pinned_to_device`]). A VMM destination must have
    /// `[offset, offset+extent)` mapped (`extent` is packed `bytes` or the
    /// pitched 2D rectangle).
    pub offset: u64,
    /// Row count for `cudaMemcpy2D`. `0` or `1` is a 1D copy of [`Self::bytes`].
    pub height: u64,
    /// Source pitch in bytes (`spitch`). `0` means packed (`width`).
    pub src_pitch: u64,
    /// Destination pitch in bytes (`dpitch`). `0` means packed (`width`).
    pub dst_pitch: u64,
    /// Slice count for `cudaMemcpy3D`. `0` or `1` is a 1D or 2D copy.
    pub depth: u64,
    /// Source 2D-slice height (`cudaPitchedPtr::ysize`). `0` means packed
    /// ([`Self::height`]).
    pub src_height: u64,
    /// Destination 2D-slice height (`cudaPitchedPtr::ysize`). `0` means packed
    /// ([`Self::height`]).
    pub dst_height: u64,
}

impl Default for MemcpyOp {
    fn default() -> Self {
        Self {
            src: Place::Host,
            dst: Place::Host,
            alloc: AllocId(0),
            bytes: 0,
            offset: 0,
            height: 0,
            src_pitch: 0,
            dst_pitch: 0,
            depth: 0,
            src_height: 0,
            dst_height: 0,
        }
    }
}

impl MemcpyOp {
    /// Packed `cudaMemcpy` / `cudaGraphAddMemcpyNode1D` of `bytes`.
    ///
    /// Height, depth, and pitches stay `0` ([`Self::default`]).
    #[must_use]
    pub fn packed_1d(src: Place, dst: Place, alloc: AllocId, bytes: u64) -> Self {
        Self {
            src,
            dst,
            alloc,
            bytes,
            ..Self::default()
        }
    }

    /// Packed 1D copy: neither [`Self::is_2d`] nor [`Self::is_3d`].
    #[must_use]
    pub fn is_1d(&self) -> bool {
        !self.is_2d() && !self.is_3d()
    }

    /// `cudaMemcpy2D` / `height > 1` and not [`Self::is_3d`].
    #[must_use]
    pub fn is_2d(&self) -> bool {
        self.height > 1 && self.depth <= 1
    }

    /// `cudaMemcpy3D` / `depth > 1`.
    #[must_use]
    pub fn is_3d(&self) -> bool {
        self.depth > 1
    }

    /// Bytes the copy engine moves (pitch and slice padding are not billed).
    #[must_use]
    pub fn payload_bytes(&self) -> u64 {
        if self.depth > 1 {
            self.bytes
                .saturating_mul(self.height.max(1))
                .saturating_mul(self.depth)
        } else if self.height > 1 {
            self.bytes.saturating_mul(self.height)
        } else {
            self.bytes
        }
    }

    /// Source pitch, or packed width when [`Self::src_pitch`] is `0`.
    #[must_use]
    pub fn src_pitch_or_width(&self) -> u64 {
        if self.src_pitch == 0 {
            self.bytes
        } else {
            self.src_pitch
        }
    }

    /// Destination pitch, or packed width when [`Self::dst_pitch`] is `0`.
    #[must_use]
    pub fn dst_pitch_or_width(&self) -> u64 {
        if self.dst_pitch == 0 {
            self.bytes
        } else {
            self.dst_pitch
        }
    }

    /// Source 2D-slice height, or packed [`Self::height`] when [`Self::src_height`] is `0`.
    #[must_use]
    pub fn src_height_or_extent(&self) -> u64 {
        if self.src_height == 0 {
            self.height.max(1)
        } else {
            self.src_height
        }
    }

    /// Destination 2D-slice height, or packed [`Self::height`] when [`Self::dst_height`] is `0`.
    #[must_use]
    pub fn dst_height_or_extent(&self) -> u64 {
        if self.dst_height == 0 {
            self.height.max(1)
        } else {
            self.dst_height
        }
    }

    /// Contiguous span in [`Self::alloc`] covering the 1D copy, 2D rectangle, or
    /// 3D box.
    #[must_use]
    pub fn extent_bytes(&self) -> u64 {
        if !self.is_2d() && !self.is_3d() {
            return self.bytes;
        }
        let pitch = self.device_pitch();
        let h = self.height.max(1);
        let plane = h
            .saturating_sub(1)
            .saturating_mul(pitch)
            .saturating_add(self.bytes);
        if !self.is_3d() {
            return plane;
        }
        let slice = pitch.saturating_mul(self.device_ysize());
        self.depth
            .saturating_sub(1)
            .saturating_mul(slice)
            .saturating_add(plane)
    }

    fn device_pitch(&self) -> u64 {
        let dst_dev = matches!(self.dst, Place::Device(_));
        let src_dev = matches!(self.src, Place::Device(_));
        if dst_dev && !src_dev {
            self.dst_pitch_or_width()
        } else if src_dev && !dst_dev {
            self.src_pitch_or_width()
        } else {
            self.src_pitch_or_width().max(self.dst_pitch_or_width())
        }
    }

    fn device_ysize(&self) -> u64 {
        let dst_dev = matches!(self.dst, Place::Device(_));
        let src_dev = matches!(self.src, Place::Device(_));
        if dst_dev && !src_dev {
            self.dst_height_or_extent()
        } else if src_dev && !dst_dev {
            self.src_height_or_extent()
        } else {
            self.src_height_or_extent().max(self.dst_height_or_extent())
        }
    }
}

/// `cudaMemcpySrcAccessOrder` for [`MemcpyAttributes`].
///
/// Destination access stays stream-ordered. Source access follows this enum.
/// `0` (`cudaMemcpySrcAccessOrderInvalid`) is not constructible.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MemcpySrcAccessOrder {
    /// `cudaMemcpySrcAccessOrderStream` (`1`). Source waits for prior stream work.
    #[default]
    Stream,
    /// `cudaMemcpySrcAccessOrderDuringApiCall` (`2`). Source may be read out of
    /// stream order; the API waits until those copies complete (ephemeral /
    /// stack sources). Capture cannot include it.
    DuringApiCall,
    /// `cudaMemcpySrcAccessOrderAny` (`3`). Source may be read out of stream
    /// order after the API returns (`malloc` host pointers). Capture cannot
    /// include it.
    Any,
}

/// `cudaMemcpyFlags` for [`MemcpyAttributes::flags`].
///
/// Unknown bits are Invalid `"memcpy flags"`.
/// [`Self::PREFER_OVERLAP_WITH_COMPUTE`] is a hint; this VM ignores it
/// (example H100 is discrete, not Tegra).
pub struct MemcpyFlags;

impl MemcpyFlags {
    /// `cudaMemcpyFlagDefault`.
    pub const DEFAULT: u32 = 0;
    /// `cudaMemcpyFlagPreferOverlapWithCompute`. Hint; ignored here.
    pub const PREFER_OVERLAP_WITH_COMPUTE: u32 = 1;
}

/// `cudaMemcpyAttributes` for [`crate::Sim::memcpy_batch_async`] /
/// [`crate::Sim::memcpy_with_attributes`].
///
/// Location hints (`srcLocHint` / `dstLocHint`) are omitted:
/// [`crate::DeviceAttr::ConcurrentManagedAccess`] and
/// [`crate::DeviceAttr::PageableMemoryAccess`] are 0, so CUDA ignores them.
/// Host NUMA ids are not modeled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemcpyAttributes {
    /// `cudaMemcpyAttributes::srcAccessOrder`.
    pub src_access_order: MemcpySrcAccessOrder,
    /// `cudaMemcpyAttributes::flags` ([`MemcpyFlags`]).
    pub flags: u32,
}

/// Device-side fill (`cudaMemsetAsync` / `cudaMemset2DAsync` / `cudaMemset3DAsync`).
///
/// [`Self::height`] `0` or `1` is `cudaMemsetAsync` of [`Self::bytes`].
/// `height > 1` and [`Self::depth`] `<= 1` is `cudaMemset2DAsync`: billed
/// payload is `width * height` (pitch padding is not written).
/// [`Self::depth`] `> 1` is `cudaMemset3DAsync`: billed payload is
/// `width * height * depth`. [`crate::Sim::graph_exec_memset_set_params`]
/// patches this on an instantiated memset node
/// (`cudaGraphExecMemsetNodeSetParams`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemsetOp {
    /// Allocation to fill.
    pub id: AllocId,
    /// Byte offset into [`Self::id`].
    pub offset: u64,
    /// Payload bytes for a 1D fill, or row width for [`Self::height`] `> 1`.
    pub bytes: u64,
    /// Row count for `cudaMemset2D` / `3D`. `0` or `1` is a 1D fill of [`Self::bytes`].
    pub height: u64,
    /// Destination pitch in bytes. `0` means packed (`width`).
    pub pitch: u64,
    /// Slice count for `cudaMemset3D`. `0` or `1` is a 1D or 2D fill.
    pub depth: u64,
    /// 2D-slice height (`cudaPitchedPtr::ysize`). `0` means packed ([`Self::height`]).
    pub ysize: u64,
}

impl Default for MemsetOp {
    fn default() -> Self {
        Self {
            id: AllocId(0),
            offset: 0,
            bytes: 0,
            height: 0,
            pitch: 0,
            depth: 0,
            ysize: 0,
        }
    }
}

impl From<KernelBuf> for MemsetOp {
    fn from(buf: KernelBuf) -> Self {
        Self {
            id: buf.id,
            offset: buf.offset,
            bytes: buf.bytes,
            ..Self::default()
        }
    }
}

impl MemsetOp {
    /// `cudaMemset2D` / `height > 1` and not [`Self::is_3d`].
    #[must_use]
    pub fn is_2d(&self) -> bool {
        self.height > 1 && self.depth <= 1
    }

    /// `cudaMemset3D` / `depth > 1`.
    #[must_use]
    pub fn is_3d(&self) -> bool {
        self.depth > 1
    }

    /// Bytes the fill engine writes (pitch and slice padding are not billed).
    #[must_use]
    pub fn payload_bytes(&self) -> u64 {
        if self.depth > 1 {
            self.bytes
                .saturating_mul(self.height.max(1))
                .saturating_mul(self.depth)
        } else if self.height > 1 {
            self.bytes.saturating_mul(self.height)
        } else {
            self.bytes
        }
    }

    /// Pitch, or packed width when [`Self::pitch`] is `0`.
    #[must_use]
    pub fn pitch_or_width(&self) -> u64 {
        if self.pitch == 0 {
            self.bytes
        } else {
            self.pitch
        }
    }

    /// Slice height, or packed [`Self::height`] when [`Self::ysize`] is `0`.
    #[must_use]
    pub fn ysize_or_extent(&self) -> u64 {
        if self.ysize == 0 {
            self.height.max(1)
        } else {
            self.ysize
        }
    }

    /// Contiguous span in [`Self::id`] covering the 1D fill, 2D rectangle, or
    /// 3D box.
    #[must_use]
    pub fn extent_bytes(&self) -> u64 {
        if !self.is_2d() && !self.is_3d() {
            return self.bytes;
        }
        let pitch = self.pitch_or_width();
        let h = self.height.max(1);
        let plane = h
            .saturating_sub(1)
            .saturating_mul(pitch)
            .saturating_add(self.bytes);
        if !self.is_3d() {
            return plane;
        }
        let slice = pitch.saturating_mul(self.ysize_or_extent());
        self.depth
            .saturating_sub(1)
            .saturating_mul(slice)
            .saturating_add(plane)
    }

    /// 1D [`KernelBuf`] view (payload span). Pitch is not represented.
    #[must_use]
    pub fn buf(&self) -> KernelBuf {
        KernelBuf {
            id: self.id,
            offset: self.offset,
            bytes: self.payload_bytes(),
        }
    }
}

/// `cudaLimit` for [`crate::Sim::set_limit`] / [`get_limit`](crate::Sim::get_limit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceLimit {
    /// `cudaLimitStackSize`. CUDA default 1024.
    StackSize,
    /// `cudaLimitPrintfFifoSize`. CUDA default 1 MiB.
    PrintfFifoSize,
    /// `cudaLimitMallocHeapSize`. CUDA default 8 MiB.
    ///
    /// Stored; this VM does not charge HBM (no device-side `malloc` yet).
    MallocHeapSize,
    /// `cudaLimitDevRuntimeSyncDepth`. CUDA default 2. Minimum 2.
    DevRuntimeSyncDepth,
    /// `cudaLimitDevRuntimePendingLaunchCount`. CUDA default 2048.
    DevRuntimePendingLaunchCount,
    /// `cudaLimitMaxL2FetchGranularity`. Power of two in `[32, 128]`.
    ///
    /// CUDA default on SM 8.0+ is 128. Access-policy windows must align to it.
    MaxL2FetchGranularity,
    /// `cudaLimitPersistingL2CacheSize`. CUDA default 0.
    PersistingL2CacheSize,
}

/// `cudaMemoryType` from [`crate::Sim::pointer_get_attributes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryType {
    /// Not a live allocation (`cudaMemoryTypeUnregistered`).
    Unregistered,
    /// Host pageable, pinned, or mapped (`cudaMemoryTypeHost`).
    Host,
    /// `cudaMalloc` / async / VMM (`cudaMemoryTypeDevice`).
    Device,
    /// `cudaMallocManaged` (`cudaMemoryTypeManaged`).
    Managed,
}

impl MemoryType {
    /// CUDA `cudaMemoryType` int (`0` Unregistered / `1` Host / `2` Device /
    /// `3` Managed).
    #[must_use]
    pub fn to_cuda(self) -> u64 {
        match self {
            Self::Unregistered => 0,
            Self::Host => 1,
            Self::Device => 2,
            Self::Managed => 3,
        }
    }
}

/// `cudaPointerAttributes`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PointerAttributes {
    /// `cudaPointerAttributes::type`.
    pub kind: MemoryType,
    /// Owning GPU. `None` for host-only or unregistered.
    pub device: Option<DeviceId>,
    /// Device mapping exists (`devicePointer != NULL`).
    pub device_pointer: bool,
    /// Host mapping exists (`hostPointer != NULL`).
    pub host_pointer: bool,
}

/// `cuPointerSetAttribute` / `cuPointerGetAttribute` for
/// [`crate::Sim::pointer_set_attribute`].
///
/// Only attributes this VM already models. Set is
/// [`Self::SyncMemops`] only; the rest are query-only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointerAttr {
    /// `CU_POINTER_ATTRIBUTE_SYNC_MEMOPS`. `1` makes memcpy/memset of this
    /// pointer host-synchronous (like pageable). Capture of those copies is
    /// refused.
    SyncMemops,
    /// `CU_POINTER_ATTRIBUTE_MEMORY_TYPE` ([`MemoryType::to_cuda`]).
    MemoryType,
    /// `CU_POINTER_ATTRIBUTE_DEVICE_POINTER` (`1` if a device mapping exists).
    DevicePointer,
    /// `CU_POINTER_ATTRIBUTE_HOST_POINTER` (`1` if a host mapping exists).
    HostPointer,
    /// `CU_POINTER_ATTRIBUTE_IS_MANAGED`.
    IsManaged,
    /// `CU_POINTER_ATTRIBUTE_RANGE_SIZE` (allocation bytes; interior offsets
    /// are not modeled).
    RangeSize,
    /// `CU_POINTER_ATTRIBUTE_MAPPED` (`1` if `cudaHostAllocMapped` /
    /// `cudaHostRegisterMapped`).
    Mapped,
    /// `CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE` ([`crate::PoolId`] as `u64`; `0`
    /// if not pool-backed).
    MemPoolHandle,
    /// `CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL` ([`crate::DeviceId`] as `u64`).
    /// Unmapped host (no device) is Invalid `"pointer attr"`.
    DeviceOrdinal,
    /// `CU_POINTER_ATTRIBUTE_RANGE_START_ADDR`. Interior offsets are not
    /// modeled: the base is the alloc id (same as
    /// [`crate::Sim::mem_get_address_range`]).
    RangeStartAddr,
    /// `CU_POINTER_ATTRIBUTE_BUFFER_ID`. This VM uses [`crate::AllocId`].
    BufferId,
    /// `CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE` (`1` if
    /// [`crate::Sim::ipc_get`] would succeed).
    IsLegacyCudaIpcCapable,
    /// `CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE` (`1` if legacy-IPC
    /// device memory on a [`crate::LinkKind::Rdma`] GPU).
    IsGpuDirectRdmaCapable,
    /// `CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES` (POSIX-FD for shareable
    /// pool allocs; else [`MemHandleType::NONE`]).
    AllowedHandleTypes,
    /// `CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR`. Interior offsets are not
    /// modeled: the base is the alloc id when a mapping covers offset 0.
    /// Unmapped VMM is Invalid `"pointer attr"`.
    MappingBaseAddr,
    /// `CU_POINTER_ATTRIBUTE_MAPPING_SIZE`. Non-VMM is [`Self::RangeSize`].
    /// VMM is the `cuMemMap` span at offset 0, not the reserved VA. Unmapped
    /// VMM or maps that skip offset 0 are Invalid `"pointer attr"`.
    MappingSize,
    /// `CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE` (always 0; compression
    /// is not modeled).
    IsHwDecompressCapable,
    /// `CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID`. The [`crate::MemHandleId`] of
    /// the `cuMemMap` that covers offset 0. Combined [`crate::Sim::va_map`]
    /// without [`crate::Sim::va_retain_handle`], `cudaMalloc`, unmapped VMM,
    /// and maps that skip offset 0 are Invalid `"pointer attr"`.
    MemoryBlockId,
}

/// `cudaDeviceAttr` for [`crate::Sim::device_get_attribute`].
///
/// Only attributes this VM already models. Values are the CUDA `int` as `u64`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceAttr {
    /// `cudaDevAttrCooperativeLaunch`.
    CooperativeLaunch,
    /// `cudaDevAttrConcurrentKernels` (`compute_slots > 1`).
    ConcurrentKernels,
    /// `cudaDevAttrMaxSharedMemoryPerBlock`.
    MaxSharedMemoryPerBlock,
    /// `cudaDevAttrMaxSharedMemoryPerBlockOptin`.
    MaxSharedMemoryPerBlockOptin,
    /// `cudaDevAttrL2CacheSize`.
    L2CacheSize,
    /// `cudaDevAttrMaxPersistingL2CacheSize`.
    MaxPersistingL2CacheSize,
    /// `cudaDevAttrMaxAccessPolicyWindowSize` (same bytes as
    /// [`Self::MaxPersistingL2CacheSize`] / [`crate::GpuProfile::l2_bytes`]).
    MaxAccessPolicyWindowSize,
    /// `cudaDevAttrGlobalL1CacheSupported` (always 0; this VM does not model
    /// L1 caches). Distinct from [`Self::L2CacheSize`].
    GlobalL1CacheSupported,
    /// `cudaDevAttrMaxBlocksPerCluster`.
    MaxBlocksPerCluster,
    /// `cudaDevAttrMemSyncDomainCount`.
    MemSyncDomainCount,
    /// `cudaDevAttrMemoryPoolsSupported` (always 1; this VM has mempools).
    MemoryPoolsSupported,
    /// `cudaDevAttrCanMapHostMemory` (always 1; this VM has mapped host).
    CanMapHostMemory,
    /// `cudaDevAttrManagedMemory` (always 1; this VM has `cudaMallocManaged`).
    ManagedMemory,
    /// `cudaDevAttrTotalGlobalMem` ([`crate::GpuProfile::hbm_bytes`]).
    TotalGlobalMem,
    /// `cudaDevAttrAsyncEngineCount` ([`crate::GpuProfile::copy_engines`]).
    AsyncEngineCount,
    /// `cudaDevAttrClusterLaunch` (`max_blocks_per_cluster > 0`).
    ClusterLaunch,
    /// `cudaDevAttrHostRegisterSupported` (always 1; this VM has `cudaHostRegister`).
    HostRegisterSupported,
    /// `cudaDevAttrIpcEventSupport` (always 1; this VM has event IPC).
    IpcEventSupport,
    /// `cudaDevAttrCanUseHostPointerForRegisteredMem` (always 1; mapped registered host).
    CanUseHostPointerForRegisteredMem,
    /// `cudaDevAttrMemoryPoolSupportedHandleTypes`
    /// ([`MemHandleType::POSIX_FILE_DESCRIPTOR`]).
    MemoryPoolSupportedHandleTypes,
    /// `cudaDevAttrGPUDirectRDMASupported` (a [`crate::LinkKind::Rdma`] link).
    GpuDirectRdmaSupported,
    /// `cudaDevAttrHostRegisterReadOnlySupported` (always 0; ReadOnly is Invalid).
    HostRegisterReadOnlySupported,
    /// `cudaDevAttrPageableMemoryAccess` (always 0; pageable is bounce-buffer).
    PageableMemoryAccess,
    /// `cudaDevAttrStreamPrioritiesSupported` (always 1; this VM has
    /// [`crate::Sim::set_stream_priority`]).
    StreamPrioritiesSupported,
    /// `cudaDevAttrGpuOverlap` ([`crate::GpuProfile::copy_engines`] `> 0`).
    GpuOverlap,
    /// `cudaDevAttrUnifiedAddressing` (always 1; this VM has one pointer space).
    UnifiedAddressing,
    /// `cudaDevAttrConcurrentManagedAccess` (always 0; host cannot touch managed
    /// while a kernel runs).
    ConcurrentManagedAccess,
    /// `cudaDevAttrDirectManagedMemAccessFromHost` (always 0).
    DirectManagedMemAccessFromHost,
    /// `cudaDevAttrPageableMemoryAccessUsesHostPageTables` (always 0; pageable
    /// is bounce-buffer).
    PageableMemoryAccessUsesHostPageTables,
    /// `cudaDevAttrCanFlushRemoteWrites` ([`Self::GpuDirectRdmaSupported`]).
    CanFlushRemoteWrites,
    /// `cudaDevAttrHostNativeAtomicSupported` (always 0; host-mapped atomics
    /// are not modeled).
    HostNativeAtomicSupported,
    /// `cudaDevAttrCooperativeMultiDeviceLaunch` (always 0; multi-device
    /// cooperative is not modeled).
    CooperativeMultiDeviceLaunch,
    /// `cudaDevAttrIntegrated` (always 0; example SKUs are discrete).
    Integrated,
    /// `cudaDevAttrSparseCudaArraySupported` (always 0; CUDA arrays are not
    /// modeled).
    SparseCudaArraySupported,
    /// `cudaDevAttrDeferredMappingCudaArraySupported` (always 0; CUDA arrays
    /// are not modeled).
    DeferredMappingCudaArraySupported,
    /// `cudaDevAttrDmaBufSupported` (always 0; dma-buf is not modeled).
    DmaBufSupported,
    /// `cudaDevAttrMulticastSupported` (a GPU↔GPU [`crate::LinkKind::Nvlink`]
    /// link on this device). PCIe P2P and RDMA are not NVLS.
    MulticastSupported,
    /// `cudaDevAttrVirtualMemoryManagementSupported` (always 1; this VM has
    /// [`crate::Sim::va_reserve`]).
    VirtualMemoryManagementSupported,
    /// `cudaDevAttrHandleTypePosixFileDescriptorSupported` (always 1; this VM
    /// has [`crate::Sim::create_shareable_pool`]).
    HandleTypePosixFileDescriptorSupported,
    /// `cudaDevAttrGPUDirectRDMAFlushWritesOptions`.
    ///
    /// [`FlushGpuDirectRdmaWritesOptions::HOST`] when this device has a
    /// GPU↔GPU [`crate::LinkKind::Rdma`] link ([`crate::Sim::flush_gpu_direct_rdma_writes`]
    /// is a host-sync barrier). [`FlushGpuDirectRdmaWritesOptions::MEMOPS`]
    /// is not modeled and is never reported.
    GpuDirectRdmaFlushWritesOptions,
    /// `cudaDevAttrGPUDirectRDMAWritesOrdering` (always
    /// [`GpuDirectRdmaWritesOrdering::NONE`]; native write visibility is not
    /// modeled, so [`crate::Sim::flush_gpu_direct_rdma_writes`] is never a
    /// no-op). Distinct from [`Self::GpuDirectRdmaFlushWritesOptions`].
    GpuDirectRdmaWritesOrdering,
    /// `cudaDevAttrGPUDirectRDMAWithCudaVMMSupported` (same RDMA SKU bit as
    /// [`Self::GpuDirectRdmaSupported`]; this VM always has VMM).
    GpuDirectRdmaWithCudaVMMSupported,
    /// `cudaDevAttrGenericCompressionSupported` (always 0; compression is not
    /// modeled).
    GenericCompressionSupported,
    /// `cudaDevAttrHandleTypeWin32HandleSupported` (always 0; this VM has
    /// POSIX-FD shareable pools, not Win32 handles).
    HandleTypeWin32HandleSupported,
    /// `cudaDevAttrHandleTypeWin32KmtHandleSupported` (always 0; this VM has
    /// POSIX-FD shareable pools, not Win32 KMT handles).
    HandleTypeWin32KmtHandleSupported,
    /// `cudaDevAttrHandleTypeFabricSupported` (always 0; fabric handles are not
    /// modeled).
    HandleTypeFabricSupported,
    /// `cudaDevAttrHostMemoryPoolsSupported` (always 0; this VM's pools are
    /// device-only; [`crate::Sim::create_pool_with_props`] refuses host
    /// location).
    HostMemoryPoolsSupported,
    /// `cudaDevAttrIsMultiGpuBoard` (always 0; example SKUs are discrete
    /// single-GPU packages).
    IsMultiGpuBoard,
    /// `cudaDevAttrMultiGpuBoardGroupID` (always 0; example SKUs are not on
    /// a multi-GPU board).
    MultiGpuBoardGroupID,
    /// `cudaDevAttrComputeMode` (always [`ComputeMode::DEFAULT`]; exclusive
    /// process / prohibited are not modeled).
    ComputeMode,
    /// `cudaDevAttrTccDriver` (always 0; example SKUs are not Windows TCC).
    TccDriver,
    /// `cudaDevAttrKernelExecTimeout` (always 0; example SKUs have no display
    /// watchdog).
    KernelExecTimeout,
    /// `cudaDevAttrCanUse64BitStreamMemOps` (always 1; this VM has
    /// [`crate::Sim::wait_value64`] / [`crate::Sim::write_value64`]).
    CanUse64BitStreamMemOps,
    /// `cudaDevAttrCanUseStreamMemOps` (always 1; this VM has
    /// [`crate::Sim::wait_value32`] / [`crate::Sim::write_value32`]).
    /// CUDA deprecated this in favor of [`Self::CanUse64BitStreamMemOps`].
    CanUseStreamMemOps,
    /// `cudaDevAttrCanUseStreamWaitValueNor` (always 1; this VM has
    /// [`WaitValueCmp::Nor`]).
    CanUseStreamWaitValueNor,
    /// `cudaDevAttrTensorMapAccessSupported` (always 0; `CUtensorMap` / TMA
    /// is not modeled).
    TensorMapAccessSupported,
    /// `cudaDevAttrUnifiedFunctionPointers` (always 0; device-side function
    /// pointers are not modeled).
    UnifiedFunctionPointers,
    /// `cudaDevAttrTimelineSemaphoreInteropSupported` (always 0; NVSci /
    /// timeline semaphore interop is not modeled).
    TimelineSemaphoreInteropSupported,
    /// `cudaDevAttrMemDecompressAlgorithmMask` (always 0; hardware decompress
    /// is not modeled).
    MemDecompressAlgorithmMask,
    /// `cudaDevAttrMemDecompressMaximumLength` (always 0; hardware decompress
    /// is not modeled).
    MemDecompressMaximumLength,
    /// `cudaDevAttrHostNumaVirtualMemoryManagementSupported` (always 0; host
    /// NUMA VMM is not modeled; [`crate::Sim::va_create_with_prop`] refuses
    /// host location). Distinct from [`Self::VirtualMemoryManagementSupported`].
    HostNumaVirtualMemoryManagementSupported,
    /// `cudaDevAttrHostNumaMemoryPoolsSupported` (always 0; host NUMA pools
    /// are not modeled; [`crate::Sim::create_pool_with_props`] refuses host
    /// location). Distinct from [`Self::HostMemoryPoolsSupported`].
    HostNumaMemoryPoolsSupported,
    /// `cudaDevAttrHostNumaMultinodeIpcSupported` (always 0; this VM's IPC is
    /// same-node; [`crate::Sim::ipc_open`] requires the dest GPU already in
    /// the allocation). Distinct from [`Self::IpcEventSupport`].
    HostNumaMultinodeIpcSupported,
    /// `cudaDevAttrNumaConfig` (always [`DeviceNumaConfig::NONE`]; GPU memory
    /// NUMA nodes are not modeled). Do not invent `cudaDevAttrNumaId`.
    NumaConfig,
    /// `cudaDevAttrOnlyPartialHostNativeAtomicSupported` (always 0;
    /// host-mapped atomics are not modeled). Distinct from
    /// [`Self::HostNativeAtomicSupported`].
    OnlyPartialHostNativeAtomicSupported,
    /// `cudaDevAttrPciDomainId` (synthetic; high 8 bits of [`crate::DeviceId`]).
    PciDomainId,
    /// `cudaDevAttrPciBusId` (synthetic; low 8 bits of [`crate::DeviceId`]).
    PciBusId,
    /// `cudaDevAttrPciDeviceId` (synthetic PCI device number; always 0).
    PciDeviceId,
}

/// `cudaMemAllocationHandleType` bits for
/// [`DeviceAttr::MemoryPoolSupportedHandleTypes`].
pub struct MemHandleType;

impl MemHandleType {
    /// `cudaMemHandleTypeNone`.
    pub const NONE: u64 = 0;
    /// `cudaMemHandleTypePosixFileDescriptor` ([`crate::Sim::create_shareable_pool`]).
    pub const POSIX_FILE_DESCRIPTOR: u64 = 1;
}

/// `cudaDeviceProp` fields this VM already models.
///
/// No SM count, clock rate, warp size, or `maxThreadsPerBlock` — those are
/// not in [`crate::GpuProfile`]. [`Self::name`] is [`crate::HardwareProfile::name`].
/// [`Self::uuid`] is the synthetic [`crate::Sim::device_get_uuid`] value
/// (`cudaUuid_t`), not a real NVIDIA board UUID. [`Self::pci_domain_id`] /
/// [`Self::pci_bus_id`] / [`Self::pci_device_id`] are the synthetic PCI
/// identity from [`crate::Sim::device_get_pci_bus_id`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceProperties {
    /// Profile name (`example-h100-sxm`, a capture id, …).
    pub name: String,
    /// Synthetic `cudaUuid_t` ([`crate::Sim::device_get_uuid`]). Not a real
    /// NVIDIA UUID and not parsed from a capture file.
    pub uuid: [u8; 16],
    /// Synthetic `pciDomainID` ([`DeviceAttr::PciDomainId`]).
    pub pci_domain_id: u32,
    /// Synthetic `pciBusID` ([`DeviceAttr::PciBusId`]).
    pub pci_bus_id: u32,
    /// Synthetic `pciDeviceID` ([`DeviceAttr::PciDeviceId`]). Always 0.
    pub pci_device_id: u32,
    /// [`crate::GpuProfile::hbm_bytes`] (`totalGlobalMem`).
    pub total_global_mem: u64,
    /// [`crate::GpuProfile::max_shared_mem_per_block`].
    pub shared_mem_per_block: u32,
    /// [`crate::GpuProfile::max_shared_mem_per_block_optin`].
    pub shared_mem_per_block_optin: u32,
    /// [`crate::GpuProfile::l2_bytes`].
    pub l2_cache_size: u64,
    /// `accessPolicyMaxWindowSize` ([`DeviceAttr::MaxAccessPolicyWindowSize`]).
    /// Same bytes as [`Self::l2_cache_size`].
    pub access_policy_max_window_size: u64,
    /// `cudaDevAttrGlobalL1CacheSupported` ([`DeviceAttr::GlobalL1CacheSupported`]).
    /// Always false; this VM does not model L1 caches.
    pub global_l1_cache_supported: bool,
    /// [`crate::GpuProfile::copy_engines`].
    pub async_engine_count: u32,
    /// [`crate::GpuProfile::compute_slots`] `> 1`.
    pub concurrent_kernels: bool,
    /// [`crate::GpuProfile::cooperative_launch`].
    pub cooperative_launch: bool,
    /// [`crate::GpuProfile::max_blocks_per_cluster`].
    pub max_blocks_per_cluster: u32,
    /// [`crate::GpuProfile::portable_cluster_size`].
    pub portable_cluster_size: u32,
    /// [`crate::GpuProfile::mem_sync_domain_count`].
    pub mem_sync_domain_count: u32,
    /// This VM always has mempools.
    pub memory_pools_supported: bool,
    /// This VM always maps host (`cudaHostAllocMapped`).
    pub can_map_host_memory: bool,
    /// This VM always has managed memory (`cudaMallocManaged`).
    pub managed_memory: bool,
    /// `cudaDevAttrClusterLaunch` ([`crate::GpuProfile::max_blocks_per_cluster`] `> 0`).
    pub cluster_launch: bool,
    /// This VM always has `cudaHostRegister`.
    pub host_register_supported: bool,
    /// This VM always has event IPC (`cudaIpcGetEventHandle`).
    pub ipc_event_support: bool,
    /// Mapped registered host may be used as a device pointer.
    pub can_use_host_pointer_for_registered_mem: bool,
    /// `cudaMemHandleTypePosixFileDescriptor` ([`MemHandleType::POSIX_FILE_DESCRIPTOR`]).
    pub memory_pool_supported_handle_types: u64,
    /// `cudaDevAttrGPUDirectRDMASupported` (a GPU↔GPU [`crate::LinkKind::Rdma`] link).
    pub gpu_direct_rdma_supported: bool,
    /// `cudaDevAttrHostRegisterReadOnlySupported` (ReadOnly host register is Invalid).
    pub host_register_read_only_supported: bool,
    /// `cudaDevAttrPageableMemoryAccess` (pageable H2D/D2H is bounce-buffer).
    pub pageable_memory_access: bool,
    /// `cudaDevAttrStreamPrioritiesSupported` (this VM has stream priorities).
    pub stream_priorities_supported: bool,
    /// `cudaDevAttrGpuOverlap` ([`crate::GpuProfile::copy_engines`] `> 0`).
    pub gpu_overlap: bool,
    /// `cudaDevAttrUnifiedAddressing` (one pointer space).
    pub unified_addressing: bool,
    /// `cudaDevAttrConcurrentManagedAccess` (host cannot touch managed during a kernel).
    pub concurrent_managed_access: bool,
    /// `cudaDevAttrDirectManagedMemAccessFromHost`.
    pub direct_managed_mem_access_from_host: bool,
    /// `cudaDevAttrPageableMemoryAccessUsesHostPageTables` (pageable is bounce-buffer).
    pub pageable_memory_access_uses_host_page_tables: bool,
    /// `cudaDevAttrCanFlushRemoteWrites` (a GPU↔GPU [`crate::LinkKind::Rdma`] link).
    pub can_flush_remote_writes: bool,
    /// `cudaDevAttrHostNativeAtomicSupported` (host-mapped atomics are not modeled).
    pub host_native_atomic_supported: bool,
    /// `cudaDevAttrCooperativeMultiDeviceLaunch` (multi-device cooperative is not modeled).
    pub cooperative_multi_device_launch: bool,
    /// `cudaDevAttrIntegrated` (example SKUs are discrete).
    pub integrated: bool,
    /// `cudaDevAttrSparseCudaArraySupported` (CUDA arrays are not modeled).
    pub sparse_cuda_array_supported: bool,
    /// `cudaDevAttrDeferredMappingCudaArraySupported` (CUDA arrays are not modeled).
    pub deferred_mapping_cuda_array_supported: bool,
    /// `cudaDevAttrDmaBufSupported` (dma-buf is not modeled).
    pub dma_buf_supported: bool,
    /// `cudaDevAttrMulticastSupported` (a GPU↔GPU [`crate::LinkKind::Nvlink`]
    /// link on this device).
    pub multicast_supported: bool,
    /// `cudaDevAttrVirtualMemoryManagementSupported` (this VM has
    /// `cuMemAddressReserve`).
    pub virtual_memory_management_supported: bool,
    /// `cudaDevAttrHandleTypePosixFileDescriptorSupported` (this VM has
    /// POSIX-FD shareable pools).
    pub handle_type_posix_file_descriptor_supported: bool,
    /// `cudaDevAttrGPUDirectRDMAFlushWritesOptions` ([`FlushGpuDirectRdmaWritesOptions::HOST`]
    /// on an RDMA SKU; MemOps is never set).
    pub gpu_direct_rdma_flush_writes_options: u64,
    /// `cudaDevAttrGPUDirectRDMAWritesOrdering` (always
    /// [`GpuDirectRdmaWritesOrdering::NONE`]). Distinct from
    /// [`Self::gpu_direct_rdma_flush_writes_options`].
    pub gpu_direct_rdma_writes_ordering: u64,
    /// `cudaDevAttrGPUDirectRDMAWithCudaVMMSupported` (RDMA SKU; VMM is always on).
    pub gpu_direct_rdma_with_cuda_vmm_supported: bool,
    /// `cudaDevAttrGenericCompressionSupported` (compression is not modeled).
    pub generic_compression_supported: bool,
    /// `cudaDevAttrHandleTypeWin32HandleSupported` (POSIX-FD only).
    pub handle_type_win32_handle_supported: bool,
    /// `cudaDevAttrHandleTypeWin32KmtHandleSupported` (POSIX-FD only).
    pub handle_type_win32_kmt_handle_supported: bool,
    /// `cudaDevAttrHandleTypeFabricSupported` (fabric handles are not modeled).
    pub handle_type_fabric_supported: bool,
    /// `cudaDevAttrHostMemoryPoolsSupported` (pools are device-only).
    pub host_memory_pools_supported: bool,
    /// `cudaDevAttrIsMultiGpuBoard` (example SKUs are discrete single-GPU
    /// packages).
    pub is_multi_gpu_board: bool,
    /// `cudaDevAttrMultiGpuBoardGroupID` (example SKUs are not on a multi-GPU
    /// board).
    pub multi_gpu_board_group_id: u32,
    /// `cudaDevAttrComputeMode` (always [`ComputeMode::DEFAULT`]).
    pub compute_mode: u32,
    /// `cudaDevAttrTccDriver` (example SKUs are not Windows TCC).
    pub tcc_driver: bool,
    /// `cudaDevAttrKernelExecTimeout` (example SKUs have no display watchdog).
    pub kernel_exec_timeout: bool,
    /// `cudaDevAttrCanUse64BitStreamMemOps` (this VM has
    /// [`crate::Sim::wait_value64`] / [`crate::Sim::write_value64`]).
    pub can_use_64_bit_stream_mem_ops: bool,
    /// `cudaDevAttrCanUseStreamMemOps` (this VM has
    /// [`crate::Sim::wait_value32`] / [`crate::Sim::write_value32`]).
    /// CUDA deprecated this in favor of [`Self::can_use_64_bit_stream_mem_ops`].
    pub can_use_stream_mem_ops: bool,
    /// `cudaDevAttrCanUseStreamWaitValueNor` (this VM has
    /// [`WaitValueCmp::Nor`]).
    pub can_use_stream_wait_value_nor: bool,
    /// `cudaDevAttrTensorMapAccessSupported` (`CUtensorMap` / TMA is not
    /// modeled).
    pub tensor_map_access_supported: bool,
    /// `cudaDevAttrUnifiedFunctionPointers` (device-side function pointers
    /// are not modeled).
    pub unified_function_pointers: bool,
    /// `cudaDevAttrTimelineSemaphoreInteropSupported` (NVSci / timeline
    /// semaphore interop is not modeled).
    pub timeline_semaphore_interop_supported: bool,
    /// `cudaDevAttrMemDecompressAlgorithmMask` (hardware decompress is not
    /// modeled).
    pub mem_decompress_algorithm_mask: u64,
    /// `cudaDevAttrMemDecompressMaximumLength` (hardware decompress is not
    /// modeled).
    pub mem_decompress_maximum_length: u64,
    /// `cudaDevAttrHostNumaVirtualMemoryManagementSupported` (host NUMA VMM
    /// is not modeled). Distinct from
    /// [`Self::virtual_memory_management_supported`].
    pub host_numa_virtual_memory_management_supported: bool,
    /// `cudaDevAttrHostNumaMemoryPoolsSupported` (host NUMA pools are not
    /// modeled). Distinct from [`Self::host_memory_pools_supported`].
    pub host_numa_memory_pools_supported: bool,
    /// `cudaDevAttrHostNumaMultinodeIpcSupported` (this VM's IPC is
    /// same-node). Distinct from [`Self::ipc_event_support`].
    pub host_numa_multinode_ipc_supported: bool,
    /// `cudaDevAttrNumaConfig` (always [`DeviceNumaConfig::NONE`]).
    pub numa_config: u32,
    /// `cudaDevAttrOnlyPartialHostNativeAtomicSupported` (host-mapped atomics
    /// are not modeled). Distinct from [`Self::host_native_atomic_supported`].
    pub only_partial_host_native_atomic_supported: bool,
}

/// `cudaComputeMode` for [`DeviceAttr::ComputeMode`].
pub struct ComputeMode;

impl ComputeMode {
    /// `cudaComputeModeDefault` (`0`). This VM has no exclusive-process or
    /// prohibited modes.
    pub const DEFAULT: u32 = 0;
}

/// `cudaDeviceNumaConfig` for [`DeviceAttr::NumaConfig`].
pub struct DeviceNumaConfig;

impl DeviceNumaConfig {
    /// `cudaDeviceNumaConfigNone` (`0`). This VM has no GPU-memory NUMA nodes.
    pub const NONE: u32 = 0;
}

/// `cudaFuncAttribute` for [`crate::Sim::func_set_attribute`] / `GetAttribute`.
///
/// Only attributes this VM already models. Value is CUDA's `int`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FuncAttr {
    /// `cudaFuncAttributeMaxDynamicSharedMemorySize`.
    MaxDynamicSharedMemorySize,
    /// `cudaFuncAttributeNonPortableClusterSizeAllowed`.
    NonPortableClusterSizeAllowed,
    /// `cudaFuncAttributePreferredSharedMemoryCarveout`
    /// (`-1` Default / `0` MaxL1 / `100` MaxShared). Other percentages are
    /// Invalid `"func attr"`.
    PreferredSharedMemoryCarveout,
    /// `cudaFuncAttributeClusterDimMustBeSet`. `0`/`1`.
    ClusterDimMustBeSet,
    /// `cudaFuncAttributeRequiredClusterWidth`. `0` unset.
    RequiredClusterWidth,
    /// `cudaFuncAttributeRequiredClusterHeight`. `0` unset.
    RequiredClusterHeight,
    /// `cudaFuncAttributeRequiredClusterDepth`. `0` unset.
    RequiredClusterDepth,
    /// `cudaFuncAttributeClusterSchedulingPolicyPreference`.
    /// `0` Default / `1` Spread / `2` LoadBalancing.
    ClusterSchedulingPolicyPreference,
}

/// Modeled `cudaFuncGetAttributes` / `cudaFuncGetAttribute` fields.
///
/// This VM has one function-attr set **per device**, not per kernel
/// function. No `maxThreadsPerBlock`, register count, or binary version —
/// those are not modeled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FuncAttributes {
    /// `cudaFuncAttributeMaxDynamicSharedMemorySize`.
    pub max_dynamic_shared_size_bytes: u32,
    /// `cudaFuncAttributeNonPortableClusterSizeAllowed`.
    pub non_portable_cluster_size_allowed: bool,
    /// `cudaFuncAttributePreferredSharedMemoryCarveout`.
    pub preferred_shmem_carveout: SharedMemCarveout,
    /// `cudaFuncAttributeClusterDimMustBeSet`.
    pub cluster_dim_must_be_set: bool,
    /// `cudaFuncAttributeRequiredClusterWidth` (`0` unset).
    pub required_cluster_width: u32,
    /// `cudaFuncAttributeRequiredClusterHeight` (`0` unset).
    pub required_cluster_height: u32,
    /// `cudaFuncAttributeRequiredClusterDepth` (`0` unset).
    pub required_cluster_depth: u32,
    /// `cudaFuncAttributeClusterSchedulingPolicyPreference`.
    pub cluster_scheduling_policy_preference: ClusterSchedulingPolicy,
}

/// `cudaDeviceP2PAttr` for [`crate::Sim::device_get_p2p_attribute`].
///
/// Only attributes this VM already models (topology links). Values are the
/// CUDA `int` as `u64`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceP2pAttr {
    /// `cudaDevP2PAttrAccessSupported`. 1 if the profile has a device–device
    /// link. Same device is 0. Independent of [`crate::Sim::enable_peer`].
    AccessSupported,
    /// `cudaDevP2PAttrPerformanceRank`. Lower is better. Unique GPU↔GPU
    /// [`crate::LinkProfile::bps`] values descending; this pair's index.
    /// Same device or no link is 0. Native atomics are not modeled
    /// ([`Self::NativeAtomicSupported`] is always 0).
    PerformanceRank,
    /// `cudaDevP2PAttrNativeAtomicSupported`. Always 0; native atomics are
    /// not modeled.
    NativeAtomicSupported,
    /// `cudaDevP2PAttrCudaArrayAccessFromDevice`. Always 0; CUDA arrays are
    /// not modeled.
    CudaArrayAccessFromDevice,
}

/// One kernel buffer: a whole allocation or a mapped VMM span.
///
/// [`Self::whole`] is `offset = 0`, `bytes = 0` (remainder of the alloc).
/// [`crate::Sim::kernel`] uses that. [`crate::Sim::kernel_bufs`] can name a
/// mapped page of a reserved VA so a paged KV working set need not cover the
/// whole pointer. [`crate::Sim::graph_exec_memset_set_params`] patches a
/// [`MemsetOp`] on an instantiated memset node
/// (`cudaGraphExecMemsetNodeSetParams`); [`MemsetOp::from`] builds a 1D fill
/// from this span.
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

/// `cudaAccessPolicyWindow` / `cudaLaunchAttributeAccessPolicyWindow` /
/// `cudaStreamAttributeAccessPolicyWindow`.
///
/// [`crate::Sim::kernel_access_policy`] applies this window to one launch.
/// [`crate::Sim::set_stream_access_policy`] stores it on the stream so
/// [`crate::Sim::kernel`] / [`crate::Sim::kernel_bufs`] inherit it.
/// [`crate::Sim::kernel_with`] and graph replay use the launch / node window.
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
        Self::persisting_ratio(buf, 1000)
    }

    /// Persisting hits, streaming misses, `hitRatio` as ‰ (`1000` = the whole window).
    ///
    /// CUDA `cudaAccessPolicyWindow.hitRatio`. `0` bills no persisting hits.
    /// Must be `<= 1000` at launch.
    #[must_use]
    pub fn persisting_ratio(buf: KernelBuf, hit_ratio_permille: u16) -> Self {
        Self {
            buf,
            hit_ratio_permille,
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

impl MemSyncDomain {
    /// CLI token: `default` / `remote`.
    pub fn parse(s: &str) -> Result<Self, crate::error::SimError> {
        match s {
            "default" => Ok(Self::Default),
            "remote" => Ok(Self::Remote),
            _ => Err(crate::error::SimError::Invalid {
                why: "unknown mem-sync-domain",
            }),
        }
    }
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

/// `cudaLaunchAttributeSynchronizationPolicy` / `cudaSynchronizationPolicy`.
///
/// Stream-only (`cudaStreamSetAttribute` / `cudaStreamCreateWithAttribute`).
/// Not a kernel launch attribute and not packed into [`KernelAttrs`].
/// Host-wait tax applies to [`crate::Sim::synchronize_stream`] and
/// [`crate::Sim::synchronize_event`] (the recording stream).
/// [`crate::Sim::synchronize`] / [`crate::Sim::synchronize_device`] are
/// `cudaDeviceSynchronize` and do not take this tax. [`Self::Auto`] inherits
/// [`crate::Sim::set_device_flags`] (unset tax 0). Decode identity stays
/// [`Self::Auto`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SynchronizationPolicy {
    /// `cudaSyncPolicyAuto` (1). Default; inherits device schedule (unset tax 0).
    #[default]
    Auto,
    /// `cudaSyncPolicySpin` (2). Tax is [`crate::GpuProfile::host_sync_spin_ns`].
    Spin,
    /// `cudaSyncPolicyYield` (3). Tax is [`crate::GpuProfile::host_sync_yield_ns`].
    Yield,
    /// `cudaSyncPolicyBlockingSync` (4). Tax is
    /// [`crate::GpuProfile::host_sync_blocking_ns`].
    BlockingSync,
}

/// `cudaSetDeviceFlags` bits for [`crate::Sim::set_device_flags`].
///
/// Schedule bits [`Self::SCHEDULE_AUTO`] / [`Self::SCHEDULE_SPIN`] /
/// [`Self::SCHEDULE_YIELD`] / [`Self::SCHEDULE_BLOCKING_SYNC`] are exclusive
/// (combined schedule bits Invalid `"device schedule"`). [`Self::MAP_HOST`]
/// / [`Self::LMEM_RESIZE_TO_MAX`] are stored. [`Self::SYNC_MEMOPS`] makes
/// runtime memcpy/memset wait the stream like pointer
/// [`PointerAttr::SyncMemops`]. Unknown bits Invalid `"device flags"`.
/// Default `0` is Auto (host-wait tax 0).
pub struct DeviceFlags;

impl DeviceFlags {
    /// `cudaDeviceScheduleAuto`.
    pub const SCHEDULE_AUTO: u32 = 0;
    /// `cudaDeviceScheduleSpin`.
    pub const SCHEDULE_SPIN: u32 = 1;
    /// `cudaDeviceScheduleYield`.
    pub const SCHEDULE_YIELD: u32 = 2;
    /// `cudaDeviceScheduleBlockingSync`.
    pub const SCHEDULE_BLOCKING_SYNC: u32 = 4;
    /// `cudaDeviceScheduleMask`.
    pub const SCHEDULE_MASK: u32 = 7;
    /// `cudaDeviceMapHost`. Stored; [`DeviceAttr::CanMapHostMemory`] is already 1.
    pub const MAP_HOST: u32 = 8;
    /// `cudaDeviceLmemResizeToMax`. Stored; local-memory resize is not modeled.
    pub const LMEM_RESIZE_TO_MAX: u32 = 16;
    /// `cudaDeviceSyncMemops`. Runtime memcpy/memset wait like pointer
    /// [`PointerAttr::SyncMemops`].
    pub const SYNC_MEMOPS: u32 = 128;
}

impl SynchronizationPolicy {
    /// CLI token: `auto` / `spin` / `yield` / `blocking`.
    pub fn parse(s: &str) -> Result<Self, crate::error::SimError> {
        match s {
            "auto" => Ok(Self::Auto),
            "spin" => Ok(Self::Spin),
            "yield" => Ok(Self::Yield),
            "blocking" => Ok(Self::BlockingSync),
            _ => Err(crate::error::SimError::Invalid {
                why: "unknown sync-policy",
            }),
        }
    }
}

/// `cudaStreamAttrID` for [`crate::Sim::stream_get_attribute`].
///
/// Existing stream state only. Green-context SM permille is not a CUDA stream
/// attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamAttr {
    /// `cudaStreamAttributePriority` ([`crate::Sim::stream_get_priority`]).
    Priority,
    /// `cudaStreamAttributeSynchronizationPolicy`.
    SynchronizationPolicy,
    /// `cudaStreamAttributeMemSyncDomain`.
    MemSyncDomain,
    /// `cudaStreamAttributeMemSyncDomainMap`.
    MemSyncDomainMap,
    /// `cudaLaunchAttributeNvlinkUtilCentricScheduling` as a stream attribute.
    NvlinkUtilCentric,
    /// `cudaStreamAttributeAccessPolicyWindow`.
    AccessPolicy,
}

/// `cudaStreamAttrValue` for [`crate::Sim::stream_get_attribute`] /
/// [`crate::Sim::stream_set_attribute`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamAttrValue {
    /// [`StreamAttr::Priority`].
    Priority(i32),
    /// [`StreamAttr::SynchronizationPolicy`].
    SynchronizationPolicy(SynchronizationPolicy),
    /// [`StreamAttr::MemSyncDomain`].
    MemSyncDomain(MemSyncDomain),
    /// [`StreamAttr::MemSyncDomainMap`].
    MemSyncDomainMap(MemSyncDomainMap),
    /// [`StreamAttr::NvlinkUtilCentric`].
    NvlinkUtilCentric(bool),
    /// [`StreamAttr::AccessPolicy`]. [`None`] clears.
    AccessPolicy(Option<AccessPolicyWindow>),
}

/// `cudaKernelNodeAttrID` / `cudaLaunchAttribute_t` for graph kernel nodes.
///
/// Typed [`crate::Sim::graph_kernel_node_get_priority`] helpers stay. This
/// enum is `cudaGraphKernelNodeGetAttribute` / `SetAttribute`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelNodeAttr {
    /// `cudaKernelNodeAttributePriority`.
    Priority,
    /// `cudaLaunchAttributeCooperative`.
    Cooperative,
    /// `cudaLaunchAttributeProgrammaticStreamSerialization`.
    Pdl,
    /// `cudaLaunchAttributeProgrammaticEvent`.
    ProgrammaticEvent,
    /// `cudaLaunchAttributeLaunchCompletionEvent`.
    LaunchCompletion,
    /// `cudaLaunchAttributeAccessPolicyWindow`.
    AccessPolicy,
    /// `cudaLaunchAttributeMemSyncDomain`.
    MemSyncDomain,
    /// `cudaLaunchAttributeMemSyncDomainMap`.
    MemSyncDomainMap,
    /// `cudaLaunchAttributeClusterDimension`.
    Cluster,
    /// `cudaLaunchAttributeClusterSchedulingPolicyPreference`.
    ClusterPolicy,
    /// `cudaLaunchAttributePreferredClusterDimension`.
    PreferredCluster,
    /// `cudaLaunchAttributePreferredSharedMemoryCarveout`.
    Carveout,
    /// `cudaLaunchAttributeDeviceUpdatableKernelNode`.
    DeviceUpdatable,
    /// `cudaLaunchAttributeSharedMemoryMode` (bank width).
    SharedMem,
    /// `cudaLaunchAttributePortableClusterSizeMode`.
    PortableCluster,
    /// CUDA 13 portable shared-memory mode.
    PortableShared,
    /// `cudaKernelNodeParams::sharedMemBytes`.
    DynamicShared,
    /// `cudaLaunchAttributeNvlinkUtilCentricScheduling`.
    NvlinkUtilCentric,
}

/// `cudaKernelNodeAttrValue` for [`crate::Sim::graph_kernel_node_get_attribute`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelNodeAttrValue {
    /// [`KernelNodeAttr::Priority`].
    Priority(i32),
    /// [`KernelNodeAttr::Cooperative`].
    Cooperative(bool),
    /// [`KernelNodeAttr::Pdl`].
    Pdl(ProgrammaticLaunch),
    /// [`KernelNodeAttr::ProgrammaticEvent`].
    ProgrammaticEvent(Option<ProgrammaticEvent>),
    /// [`KernelNodeAttr::LaunchCompletion`].
    LaunchCompletion(Option<LaunchCompletionEvent>),
    /// [`KernelNodeAttr::AccessPolicy`].
    AccessPolicy(Option<AccessPolicyWindow>),
    /// [`KernelNodeAttr::MemSyncDomain`].
    MemSyncDomain(MemSyncDomain),
    /// [`KernelNodeAttr::MemSyncDomainMap`].
    MemSyncDomainMap(MemSyncDomainMap),
    /// [`KernelNodeAttr::Cluster`].
    Cluster(Option<ClusterDim>),
    /// [`KernelNodeAttr::ClusterPolicy`].
    ClusterPolicy(ClusterSchedulingPolicy),
    /// [`KernelNodeAttr::PreferredCluster`].
    PreferredCluster(Option<ClusterDim>),
    /// [`KernelNodeAttr::Carveout`].
    Carveout(SharedMemCarveout),
    /// [`KernelNodeAttr::DeviceUpdatable`].
    DeviceUpdatable(bool),
    /// [`KernelNodeAttr::SharedMem`].
    SharedMem(SharedMemoryMode),
    /// [`KernelNodeAttr::PortableCluster`].
    PortableCluster(PortableClusterMode),
    /// [`KernelNodeAttr::PortableShared`].
    PortableShared(PortableSharedMode),
    /// [`KernelNodeAttr::DynamicShared`].
    DynamicShared(u32),
    /// [`KernelNodeAttr::NvlinkUtilCentric`].
    NvlinkUtilCentric(bool),
}

/// Packed `cudaLaunchKernelEx` / graph kernel-node attributes.
///
/// [`crate::Sim::kernel_with`] applies these on one submit so PDL, an
/// access-policy window, and a mem-sync domain can share a launch (7 arguments
/// including `self`). Decode identity stays [`crate::Sim::kernel`] ([`Default`]:
/// no cooperative, no PDL, no window, inherit stream mem-sync, no cluster,
/// Default carveout, not device-updatable, Default shared-memory bank mode,
/// Default portable-cluster mode, 0 dynamic shared, Default portable-shared,
/// inherit stream priority, no programmatic event).
/// [`SynchronizationPolicy`] is a stream attribute, not a field here.
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
    ///
    /// [`crate::Sim::set_cluster_dim_must_be_set`] makes `None` Invalid.
    /// Nonzero [`crate::Sim::set_required_cluster_width`] (height/depth) must
    /// match that axis.
    pub cluster: Option<ClusterDim>,
    /// `cudaLaunchAttributeClusterSchedulingPolicyPreference`.
    ///
    /// [`ClusterSchedulingPolicy::Default`] uses
    /// [`crate::Sim::set_func_cluster_policy`]
    /// (`cudaFuncAttributeClusterSchedulingPolicyPreference`).
    pub cluster_policy: ClusterSchedulingPolicy,
    /// `cudaLaunchAttributePreferredClusterDimension`. `None` uses [`Self::cluster`].
    pub preferred_cluster: Option<ClusterDim>,
    /// `cudaLaunchAttributePreferredSharedMemoryCarveout`.
    ///
    /// [`SharedMemCarveout::Default`] uses
    /// [`crate::Sim::set_func_carveout`]
    /// (`cudaFuncAttributePreferredSharedMemoryCarveout`).
    pub carveout: SharedMemCarveout,
    /// `cudaLaunchAttributeDeviceUpdatableKernelNode`.
    ///
    /// Graphs-only: a non-capturing launch is Invalid. When true,
    /// [`crate::Sim::graph_exec_kernel_set_params`] keeps the exec uploaded so a
    /// later [`crate::Sim::device_launch_graph`] needs no host re-upload.
    /// Decode identity stays `false`.
    pub device_updatable: bool,
    /// `cudaLaunchAttributeSharedMemoryMode`.
    pub shared_mem: SharedMemoryMode,
    /// `cudaLaunchAttributePortableClusterSizeMode`.
    ///
    /// [`PortableClusterMode::Default`] uses the current
    /// [`crate::Sim::set_non_portable_cluster_size_allowed`]. Decode identity
    /// stays Default (function attr disallowed).
    pub portable_cluster: PortableClusterMode,
    /// `cudaLaunchKernel` / `cudaKernelNodeParams::sharedMemBytes`.
    ///
    /// Decode identity stays `0`. Sizes above
    /// [`crate::GpuProfile::max_shared_mem_per_block`] need
    /// [`crate::Sim::set_max_dynamic_shared_memory`] or
    /// [`PortableSharedMode::AllowNonPortable`].
    pub dynamic_shared: u32,
    /// CUDA 13 `cudaLaunchAttributeSharedMemoryMode` (`cudaSharedMemoryMode`).
    ///
    /// Distinct from [`SharedMemoryMode`] (bank width / `cudaSharedMemConfig`).
    /// [`PortableSharedMode::Default`] uses the current
    /// [`crate::Sim::set_max_dynamic_shared_memory`]. Decode identity stays
    /// Default (function attr 0).
    pub portable_shared: PortableSharedMode,
    /// `cudaLaunchAttributeNvlinkUtilCentricScheduling`.
    ///
    /// `true` is enabled (`1`). Decode identity stays `false`. CUDA treats this
    /// as a hint; this VM honors it as occupancy when the profile has NVLink
    /// (every Hyper-Q slot, even NVLink traffic per block). Without NVLink the
    /// flag is stored and occupancy is unchanged.
    pub nvlink_util_centric: bool,
    /// `cudaLaunchAttributePriority`.
    ///
    /// [`None`] inherits the stream (`cudaStreamCreateWithPriority`). [`Some`]
    /// overrides for this kernel only; memcpy and other stream work stay on the
    /// stream priority. Higher values start first when compute contends.
    /// Decode identity stays [`None`]. Capture snapshots the effective value
    /// (`cudaKernelNodeAttributePriority`). Default graph replay still uses the
    /// launch stream unless `cudaGraphInstantiateFlagUseNodePriority`.
    pub priority: Option<i32>,
    /// `cudaLaunchAttributeProgrammaticEvent`. [`None`] records nothing.
    ///
    /// Other streams may [`crate::Sim::wait_event`] at the PDL trigger when
    /// [`Self::pdl`] has [`ProgrammaticLaunch::trigger`], else at kernel
    /// completion. Decode identity stays [`None`]. Capture records the attribute.
    pub programmatic_event: Option<ProgrammaticEvent>,
    /// `cudaLaunchAttributeLaunchCompletionEvent`. [`None`] records nothing.
    ///
    /// Other streams may [`crate::Sim::wait_event`] when this kernel *starts*.
    /// Decode identity stays [`None`]. Capture records the attribute.
    pub launch_completion: Option<LaunchCompletionEvent>,
}

/// `cudaSharedmemCarveout` preference
/// (`cudaLaunchAttributePreferredSharedMemoryCarveout` /
/// `cudaFuncAttributePreferredSharedMemoryCarveout`).
///
/// [`Self::MaxShared`] occupies every Hyper-Q slot so leftover kernels cannot
/// overlap. [`Self::Default`] uses the function attribute; [`Self::MaxL1`]
/// keeps current occupancy. Decode identity stays [`Self::Default`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SharedMemCarveout {
    /// `cudaSharedmemCarveoutDefault` (`-1`). Uses the function attribute.
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

    /// CUDA `int`: [`Self::Default`] `-1`, [`Self::MaxL1`] `0`,
    /// [`Self::MaxShared`] `100`. Other percentages are Invalid `"func attr"`.
    pub fn from_cuda(value: i32) -> Result<Self, crate::error::SimError> {
        match value {
            -1 => Ok(Self::Default),
            0 => Ok(Self::MaxL1),
            100 => Ok(Self::MaxShared),
            _ => Err(crate::error::SimError::Invalid { why: "func attr" }),
        }
    }

    /// Inverse of [`Self::from_cuda`].
    #[must_use]
    pub fn to_cuda(self) -> i32 {
        match self {
            Self::Default => -1,
            Self::MaxL1 => 0,
            Self::MaxShared => 100,
        }
    }
}

/// `cudaSharedMemoryConfig` (`cudaLaunchAttributeSharedMemoryMode`).
///
/// Bank width for shared-memory accesses. [`Self::Default`] uses
/// [`crate::Sim::set_func_shared_mem_config`] (`cudaFuncSetSharedMemConfig`)
/// when that is not Default, else [`crate::Sim::set_shared_mem_config`]
/// (`cudaDeviceSetSharedMemConfig`); both unset never scale (decode
/// identity). [`Self::FourByte`] / [`Self::EightByte`] scale duration by
/// `1000 / GpuProfile::shared_mem_*_permille` (profile default `1000` is
/// identity). Not occupancy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SharedMemoryMode {
    /// `cudaSharedMemoryBankSizeDefault`. Uses the function config, then
    /// the device config.
    #[default]
    Default,
    /// `cudaSharedMemoryBankSizeFourByte`. Scale by
    /// [`crate::GpuProfile::shared_mem_four_byte_permille`].
    FourByte,
    /// `cudaSharedMemoryBankSizeEightByte`. Scale by
    /// [`crate::GpuProfile::shared_mem_eight_byte_permille`].
    EightByte,
}

impl SharedMemoryMode {
    /// CLI token: `default` / `four` / `eight`.
    pub fn parse(s: &str) -> Result<Self, crate::error::SimError> {
        match s {
            "default" => Ok(Self::Default),
            "four" => Ok(Self::FourByte),
            "eight" => Ok(Self::EightByte),
            _ => Err(crate::error::SimError::Invalid {
                why: "unknown shared-mem",
            }),
        }
    }
}

/// `cudaLaunchAttributePortableClusterSizeMode`.
///
/// Launch-time override of [`crate::Sim::set_non_portable_cluster_size_allowed`].
/// [`Self::Default`] uses the current function attribute. [`Self::RequirePortable`]
/// refuses a cluster larger than [`crate::GpuProfile::portable_cluster_size`]
/// even when the function attribute allows it. [`Self::AllowNonPortable`]
/// allows up to [`crate::GpuProfile::max_blocks_per_cluster`] even when the
/// function attribute is off. Decode identity stays [`Self::Default`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PortableClusterMode {
    /// `cudaLaunchPortableClusterModeDefault`. Use the function attribute.
    #[default]
    Default,
    /// `cudaLaunchPortableClusterModeRequirePortable`.
    RequirePortable,
    /// `cudaLaunchPortableClusterModeAllowNonPortable`.
    AllowNonPortable,
}

impl PortableClusterMode {
    /// CLI token: `default` / `portable` / `non-portable`.
    pub fn parse(s: &str) -> Result<Self, crate::error::SimError> {
        match s {
            "default" => Ok(Self::Default),
            "portable" => Ok(Self::RequirePortable),
            "non-portable" => Ok(Self::AllowNonPortable),
            _ => Err(crate::error::SimError::Invalid {
                why: "unknown portable-cluster",
            }),
        }
    }

    /// Whether `mode` plus the function attribute allows a non-portable size.
    #[must_use]
    pub fn allows_non_portable(self, func_allowed: bool) -> bool {
        match self {
            Self::Default => func_allowed,
            Self::RequirePortable => false,
            Self::AllowNonPortable => true,
        }
    }
}

/// CUDA 13 `cudaLaunchAttributeSharedMemoryMode` (`cudaSharedMemoryMode`).
///
/// Launch-time override of [`crate::Sim::set_max_dynamic_shared_memory`].
/// [`Self::Default`] uses the current function attribute.
/// [`Self::RequirePortable`] refuses dynamic shared larger than
/// [`crate::GpuProfile::max_shared_mem_per_block`] even when the function
/// attribute allows it. [`Self::AllowNonPortable`] allows up to
/// [`crate::GpuProfile::max_shared_mem_per_block_optin`] even when the
/// function attribute is 0. Distinct from [`SharedMemoryMode`] (bank width).
/// Decode identity stays [`Self::Default`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PortableSharedMode {
    /// `cudaSharedMemoryModeDefault`. Use the function attribute.
    #[default]
    Default,
    /// `cudaSharedMemoryModeRequirePortable`.
    RequirePortable,
    /// `cudaSharedMemoryModeAllowNonPortable`.
    AllowNonPortable,
}

/// `cudaLaunchAttributeNvlinkUtilCentricScheduling` (`0` disabled / `1` enabled).
pub fn parse_nvlink_util_centric(s: &str) -> Result<bool, crate::error::SimError> {
    match s {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(crate::error::SimError::Invalid {
            why: "unknown nvlink-util",
        }),
    }
}

impl PortableSharedMode {
    /// CLI token: `default` / `portable` / `non-portable`.
    pub fn parse(s: &str) -> Result<Self, crate::error::SimError> {
        match s {
            "default" => Ok(Self::Default),
            "portable" => Ok(Self::RequirePortable),
            "non-portable" => Ok(Self::AllowNonPortable),
            _ => Err(crate::error::SimError::Invalid {
                why: "unknown portable-shared",
            }),
        }
    }

    /// Whether `mode` plus the function-attribute max allows `bytes` above portable.
    #[must_use]
    pub fn allows_oversize(self, func_max: u32, bytes: u32, portable: u32) -> bool {
        if bytes <= portable {
            return true;
        }
        match self {
            Self::Default => func_max >= bytes,
            Self::RequirePortable => false,
            Self::AllowNonPortable => true,
        }
    }
}

/// `cudaClusterSchedulingPolicy` (`cudaLaunchAttributeClusterSchedulingPolicyPreference`
/// / `cudaFuncAttributeClusterSchedulingPolicyPreference`).
///
/// Spread occupies every Hyper-Q slot so leftover kernels cannot overlap a
/// clustered launch. Default and LoadBalancing occupy
/// `min(blocks, compute_slots)` (the current cluster occupancy). Launch
/// [`Self::Default`] uses the function attribute. Decode identity stays
/// [`Self::Default`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClusterSchedulingPolicy {
    /// `cudaClusterSchedulingPolicyDefault` (`0`). Uses the function attribute.
    #[default]
    Default,
    /// `cudaClusterSchedulingPolicySpread` (`1`): spread cluster blocks across SMs.
    Spread,
    /// `cudaClusterSchedulingPolicyLoadBalancing` (`2`): hardware may pack blocks.
    LoadBalancing,
}

impl ClusterSchedulingPolicy {
    /// CUDA `int`: [`Self::Default`] `0`, [`Self::Spread`] `1`,
    /// [`Self::LoadBalancing`] `2`. Other values are Invalid `"func attr"`.
    pub fn from_cuda(value: i32) -> Result<Self, crate::error::SimError> {
        match value {
            0 => Ok(Self::Default),
            1 => Ok(Self::Spread),
            2 => Ok(Self::LoadBalancing),
            _ => Err(crate::error::SimError::Invalid { why: "func attr" }),
        }
    }

    /// Inverse of [`Self::from_cuda`].
    #[must_use]
    pub fn to_cuda(self) -> i32 {
        match self {
            Self::Default => 0,
            Self::Spread => 1,
            Self::LoadBalancing => 2,
        }
    }
}

/// `cudaLaunchAttributeClusterDimension` (`clusterDim`).
///
/// All three dimensions must be `>= 1`. Product is the cluster block count and
/// must be `<= GpuProfile::max_blocks_per_cluster`. Sizes above
/// [`crate::GpuProfile::portable_cluster_size`] also need
/// [`crate::Sim::set_non_portable_cluster_size_allowed`] or
/// [`PortableClusterMode::AllowNonPortable`]. Decode identity stays
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

/// `cudaMemAttach*` bits for [`crate::Sim::alloc_managed_with_flags`] /
/// [`crate::Sim::stream_attach_with_flags`].
///
/// [`Self::SINGLE`] is stream-attach only; mallocManaged of it is Invalid `"managed flags"`.
pub struct MemAttachFlags;

impl MemAttachFlags {
    /// `cudaMemAttachGlobal`.
    pub const GLOBAL: u32 = 1;
    /// `cudaMemAttachHost`.
    pub const HOST: u32 = 2;
    /// `cudaMemAttachSingle` ([`crate::Sim::stream_attach`] /
    /// [`crate::Sim::stream_attach_with_flags`] only).
    pub const SINGLE: u32 = 4;
}

/// `CU_STREAM_WAIT_VALUE_*` flags for [`crate::Sim::wait_value32_with_flags`] /
/// [`crate::Sim::wait_value64_with_flags`].
pub struct WaitValueFlags;

impl WaitValueFlags {
    /// `CU_STREAM_WAIT_VALUE_GEQ` (`0`).
    pub const GEQ: u32 = 0;
    /// `CU_STREAM_WAIT_VALUE_EQ` (`1`).
    pub const EQ: u32 = 1;
    /// `CU_STREAM_WAIT_VALUE_AND` (`2`).
    pub const AND: u32 = 2;
    /// `CU_STREAM_WAIT_VALUE_NOR` (`3`).
    pub const NOR: u32 = 3;
    /// `CU_STREAM_WAIT_VALUE_FLUSH`. Not modeled; Invalid `"wait value flags"`.
    pub const FLUSH: u32 = 1 << 30;
}

/// `CU_STREAM_WRITE_VALUE_*` flags for [`crate::Sim::write_value32_with_flags`] /
/// [`crate::Sim::write_value64_with_flags`].
pub struct WriteValueFlags;

impl WriteValueFlags {
    /// `CU_STREAM_WRITE_VALUE_DEFAULT` (`0`).
    pub const DEFAULT: u32 = 0;
    /// `CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER`. Not modeled; Invalid
    /// `"write value flags"`.
    pub const NO_MEMORY_BARRIER: u32 = 1;
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

    /// `CU_STREAM_WAIT_VALUE_*` flags word. [`WaitValueFlags::FLUSH`] and
    /// unknown bits are Invalid `"wait value flags"`.
    pub fn from_flags(flags: u32) -> Result<Self, crate::error::SimError> {
        const CMP: u32 = WaitValueFlags::NOR;
        const KNOWN: u32 = CMP | WaitValueFlags::FLUSH;
        if flags & !KNOWN != 0 || flags & WaitValueFlags::FLUSH != 0 {
            return Err(crate::error::SimError::Invalid {
                why: "wait value flags",
            });
        }
        Ok(match flags & CMP {
            WaitValueFlags::GEQ => Self::Geq,
            WaitValueFlags::EQ => Self::Eq,
            WaitValueFlags::AND => Self::And,
            WaitValueFlags::NOR => Self::Nor,
            _ => {
                return Err(crate::error::SimError::Invalid {
                    why: "wait value flags",
                });
            }
        })
    }
}

/// `cuStreamBatchMemOp` flags for [`crate::Sim::batch_mem_op_with_flags`] /
/// [`crate::Sim::graph_add_batch_mem_op_with_flags`].
pub struct BatchMemOpFlags;

impl BatchMemOpFlags {
    /// CUDA requires `0`.
    pub const DEFAULT: u32 = 0;
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
    /// Device-side fill (`cudaMemsetAsync` / `cudaMemset2DAsync`).
    Memset(MemsetOp),
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

/// `cudaEventRecordWithFlags` bits for [`crate::Sim::record_event_with_flags`].
pub struct EventRecordFlags;

impl EventRecordFlags {
    /// `cudaEventRecordDefault`.
    pub const DEFAULT: u32 = 0;
    /// `cudaEventRecordExternal` ([`crate::Sim::record_event_external`]).
    pub const EXTERNAL: u32 = 1;
}

/// `cudaStreamWaitEvent` flags for [`crate::Sim::wait_event_with_flags`].
pub struct EventWaitFlags;

impl EventWaitFlags {
    /// `cudaEventWaitDefault`.
    pub const DEFAULT: u32 = 0;
    /// `cudaEventWaitExternal` ([`crate::Sim::wait_event_external`]).
    pub const EXTERNAL: u32 = 1;
}

/// `cudaStreamCreateWithFlags` bits for [`crate::Sim::stream_create_with_flags`].
pub struct StreamCreateFlags;

impl StreamCreateFlags {
    /// `cudaStreamDefault` (blocking; serializes with NULL).
    pub const DEFAULT: u32 = 0;
    /// `cudaStreamNonBlocking`.
    pub const NON_BLOCKING: u32 = 1;
}

/// `CUdevResourceType` for [`crate::Sim::device_get_dev_resource`] /
/// [`crate::Sim::stream_get_dev_resource`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DevResourceType {
    /// `CU_DEV_RESOURCE_TYPE_SM` (‰ of the chip, not an SM count).
    Sm,
}

/// Green-context SM span in permille of peak FLOP/s.
///
/// [`Self::FULL`] is the whole chip (`start` 0, `width` 1000). This VM does not
/// invent occupancy SM counts; the unit matches
/// [`crate::Sim::set_stream_sm_permille`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmResource {
    /// Offset into the chip, `0..1000`.
    pub start: u16,
    /// Width in ‰, `1..=1000`. `start plus width` must be `<= 1000`.
    pub width: u16,
}

impl SmResource {
    /// Full-chip SM resource (`cuDeviceGetDevResource`).
    pub const FULL: Self = Self {
        start: 0,
        width: 1000,
    };

    /// Half-open `[start, start+width)` ranges overlap.
    #[must_use]
    pub fn overlaps(self, other: Self) -> bool {
        let a1 = self.start.saturating_add(self.width);
        let b1 = other.start.saturating_add(other.width);
        self.start < b1 && other.start < a1
    }

    /// `cuDevSmResourceSplitByCount`: even groups of at least `min_count` ‰.
    ///
    /// `nb_groups` is the requested count; the actual count is
    /// `min(nb_groups, width / min_count)`. Leftover ‰ after even division
    /// is [`remaining`](Self). `min_count` `0` or `nb_groups` `0` is Invalid.
    pub fn split_by_count(
        self,
        nb_groups: u32,
        min_count: u32,
    ) -> Result<(Vec<Self>, Self), crate::SimError> {
        if self.width == 0 || self.start.saturating_add(self.width) > 1000 {
            return Err(crate::SimError::Invalid { why: "sm resource" });
        }
        if nb_groups == 0 {
            return Err(crate::SimError::Invalid {
                why: "sm split groups",
            });
        }
        if min_count == 0 {
            return Err(crate::SimError::Invalid {
                why: "sm split min count",
            });
        }
        let min = u16::try_from(min_count.min(u32::from(u16::MAX))).unwrap_or(u16::MAX);
        if min > self.width {
            return Ok((Vec::new(), self));
        }
        let max_groups = u32::from(self.width / min);
        let n = nb_groups.min(max_groups).max(1);
        let n_u16 = u16::try_from(n).unwrap_or(1);
        let each = self.width / n_u16;
        let used = each.saturating_mul(n_u16);
        let mut groups = Vec::new();
        let mut start = self.start;
        for _ in 0..n_u16 {
            groups.push(Self { start, width: each });
            start = start.saturating_add(each);
        }
        Ok((
            groups,
            Self {
                start,
                width: self.width.saturating_sub(used),
            },
        ))
    }
}

impl Default for SmResource {
    fn default() -> Self {
        Self::FULL
    }
}

/// `CUdevResource` from [`crate::Sim::device_get_dev_resource`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DevResource {
    /// SM partition ([`SmResource`]).
    Sm(SmResource),
}

/// `cuGreenCtxCreate` flags for [`crate::Sim::green_ctx_create`].
///
/// Only [`Self::DEFAULT`]. `CU_GREEN_CTX_DEFAULT_STREAM` is not modeled
/// (this VM has no second NULL stream).
pub struct GreenCtxFlags;

impl GreenCtxFlags {
    /// `CU_GREEN_CTX_FLAG_DEFAULT` (`0`).
    pub const DEFAULT: u32 = 0;
}

/// `cuDevSmResourceSplitByCount` `useFlags`.
///
/// Only [`Self::DEFAULT`]. Coscheduling / max-cluster split bits are not
/// modeled.
pub struct DevSmResourceSplitFlags;

impl DevSmResourceSplitFlags {
    /// Flags `0`.
    pub const DEFAULT: u32 = 0;
}

/// `cudaFlushGPUDirectRDMAWritesOptions` bits for
/// [`DeviceAttr::GpuDirectRdmaFlushWritesOptions`].
///
/// Host (`1`) is the only option this crate models
/// ([`crate::Sim::flush_gpu_direct_rdma_writes`] is a host-sync barrier).
/// MemOps (`2`) is not modeled and is never reported.
pub struct FlushGpuDirectRdmaWritesOptions;

impl FlushGpuDirectRdmaWritesOptions {
    /// `cudaFlushGPUDirectRDMAWritesOptionHost`.
    pub const HOST: u64 = 1;
    /// `cudaFlushGPUDirectRDMAWritesOptionMemOps`. Not modeled.
    pub const MEMOPS: u64 = 2;
}

/// `cudaGPUDirectRDMAWritesOrdering` for
/// [`DeviceAttr::GpuDirectRdmaWritesOrdering`].
///
/// This VM always reports [`Self::NONE`]: remote writes are not natively
/// ordered, so [`crate::Sim::flush_gpu_direct_rdma_writes`] is never a no-op.
/// [`Self::OWNER`] / [`Self::ALL_DEVICES`] are CUDA names only (not reported).
pub struct GpuDirectRdmaWritesOrdering;

impl GpuDirectRdmaWritesOrdering {
    /// `cudaGPUDirectRDMAWritesOrderingNone`.
    pub const NONE: u64 = 0;
    /// `cudaGPUDirectRDMAWritesOrderingOwner`. Not reported.
    pub const OWNER: u64 = 100;
    /// `cudaGPUDirectRDMAWritesOrderingAllDevices`. Not reported.
    pub const ALL_DEVICES: u64 = 200;
}

/// `cudaDeviceFlushGPUDirectRDMAWrites` target for
/// [`crate::Sim::flush_gpu_direct_rdma_writes`].
pub struct FlushGpuDirectRdmaTarget;

impl FlushGpuDirectRdmaTarget {
    /// `cudaFlushGPUDirectRDMAWritesTargetCurrentDevice`.
    pub const CURRENT_DEVICE: u32 = 0;
}

/// `cudaDeviceFlushGPUDirectRDMAWrites` scope for
/// [`crate::Sim::flush_gpu_direct_rdma_writes`].
pub struct FlushGpuDirectRdmaScope;

impl FlushGpuDirectRdmaScope {
    /// `cudaFlushGPUDirectRDMAWritesToOwner`.
    pub const TO_OWNER: u32 = 100;
    /// `cudaFlushGPUDirectRDMAWritesToAllDevices`.
    pub const TO_ALL_DEVICES: u32 = 200;
}

/// `cudaDeviceEnablePeerAccess` flags for [`crate::Sim::enable_peer_with_flags`].
///
/// CUDA requires 0. Unknown bits are Invalid `"peer access flags"`.
pub struct PeerAccessFlags;

impl PeerAccessFlags {
    /// `cudaDeviceEnablePeerAccess` `flags` must be 0.
    pub const DEFAULT: u32 = 0;
}

/// `cudaMemPrefetchAsync` / `cuMemPrefetchAsync_v2` flags for
/// [`crate::Sim::prefetch_with_flags`].
///
/// CUDA requires 0. Unknown bits are Invalid `"prefetch flags"`.
pub struct PrefetchFlags;

impl PrefetchFlags {
    /// Unflagged [`crate::Sim::prefetch`] / [`crate::Sim::prefetch_host`].
    pub const DEFAULT: u32 = 0;
}

/// `cudaHostGetDevicePointer` flags for [`crate::Sim::host_get_device_pointer_with_flags`].
///
/// CUDA requires 0. Unknown bits are Invalid `"host get device pointer flags"`.
pub struct HostGetDevicePointerFlags;

impl HostGetDevicePointerFlags {
    /// `cudaHostGetDevicePointer` `flags` must be 0.
    pub const DEFAULT: u32 = 0;
}

/// `cudaIpcOpenMemHandle` flags for [`crate::Sim::ipc_open_with_flags`].
///
/// [`Self::LAZY_ENABLE_PEER_ACCESS`] is a no-op: the dest GPU must already hold the
/// source. Cross-GPU lazy peer is not modeled. Unknown bits are Invalid
/// `"ipc open flags"`.
pub struct IpcMemFlags;

impl IpcMemFlags {
    /// Unflagged [`crate::Sim::ipc_open`].
    pub const DEFAULT: u32 = 0;
    /// `cudaIpcMemLazyEnablePeerAccess`. No-op when dest already holds the source.
    pub const LAZY_ENABLE_PEER_ACCESS: u32 = 1;
}

/// `cudaEventCreateWithFlags` bits for [`crate::Sim::create_event_with_flags`].
///
/// [`BLOCKING_SYNC`](Self::BLOCKING_SYNC) taxes [`crate::Sim::synchronize_event`]
/// with [`crate::GpuProfile::host_sync_blocking_ns`]. [`INTERPROCESS`] requires
/// [`DISABLE_TIMING`] (`cudaIpcGetEventHandle`).
pub struct EventCreateFlags;

impl EventCreateFlags {
    /// `cudaEventDefault` (timing enabled).
    pub const DEFAULT: u32 = 0;
    /// `cudaEventBlockingSync` ([`crate::Sim::create_event_blocking_sync`]).
    pub const BLOCKING_SYNC: u32 = 1;
    /// `cudaEventDisableTiming` ([`crate::Sim::create_event_disable_timing`]).
    pub const DISABLE_TIMING: u32 = 2;
    /// `cudaEventInterprocess`. Requires [`DISABLE_TIMING`].
    pub const INTERPROCESS: u32 = 4;
}

/// `cudaHostAlloc*` / `cudaHostRegister*` bits for [`crate::Sim::host_get_flags`].
///
/// Portable / WriteCombined are stored (no DMA change). IoMemory / ReadOnly
/// stay Invalid.
pub struct HostAllocFlags;

impl HostAllocFlags {
    /// `cudaHostAllocDefault` / `cudaHostRegisterDefault`.
    pub const DEFAULT: u32 = 0;
    /// `cudaHostAllocPortable` / `cudaHostRegisterPortable`.
    pub const PORTABLE: u32 = 1;
    /// `cudaHostAllocMapped` / `cudaHostRegisterMapped`.
    pub const MAPPED: u32 = 2;
    /// `cudaHostAllocWriteCombined` (alloc only; register IoMemory is Invalid).
    pub const WRITE_COMBINED: u32 = 4;
}

/// `cudaGraphDebugDotFlags` for [`crate::Sim::graph_debug_dot_with_flags`].
///
/// Bit values match CUDA. External-semaphore and extra-conditional-edge flags
/// are not modeled (Invalid).
pub struct GraphDebugDotFlags;

impl GraphDebugDotFlags {
    /// `cudaGraphDebugDotFlagsVerbose` (all modeled param dumps).
    pub const VERBOSE: u32 = 1;
    /// `cudaGraphDebugDotFlagsKernelNodeParams`.
    pub const KERNEL_NODE_PARAMS: u32 = 1 << 2;
    /// `cudaGraphDebugDotFlagsMemcpyNodeParams`.
    pub const MEMCPY_NODE_PARAMS: u32 = 1 << 3;
    /// `cudaGraphDebugDotFlagsMemsetNodeParams`.
    pub const MEMSET_NODE_PARAMS: u32 = 1 << 4;
    /// `cudaGraphDebugDotFlagsHostNodeParams`.
    pub const HOST_NODE_PARAMS: u32 = 1 << 5;
    /// `cudaGraphDebugDotFlagsEventNodeParams`.
    pub const EVENT_NODE_PARAMS: u32 = 1 << 6;
    /// `cudaGraphDebugDotFlagsKernelNodeAttributes`.
    pub const KERNEL_NODE_ATTRIBUTES: u32 = 1 << 9;
    /// `cudaGraphDebugDotFlagsHandles` (graph / alloc / event ids).
    pub const HANDLES: u32 = 1 << 10;
    /// `cudaGraphDebugDotFlagsMemAllocNodeParams`.
    pub const MEM_ALLOC_NODE_PARAMS: u32 = 1 << 11;
    /// `cudaGraphDebugDotFlagsMemFreeNodeParams`.
    pub const MEM_FREE_NODE_PARAMS: u32 = 1 << 12;
    /// `cudaGraphDebugDotFlagsBatchMemOpNodeParams`.
    pub const BATCH_MEM_OP_NODE_PARAMS: u32 = 1 << 13;
    /// `cudaGraphDebugDotFlagsConditionalNodeParams`.
    pub const CONDITIONAL_NODE_PARAMS: u32 = 1 << 15;
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
    /// Reserved graph-mem bytes (live plus unused cached). Unused bytes stay
    /// charged until [`crate::Sim::graph_mem_trim`].
    ReservedMemCurrent,
    /// High-water of [`Self::ReservedMemCurrent`].
    ReservedMemHigh,
}

/// `cudaMemAllocationType` for [`MemPoolProps`].
///
/// Only [`Self::PINNED`] is modeled (`cudaMemAllocationTypePinned`).
pub struct MemAllocationType;

impl MemAllocationType {
    /// `cudaMemAllocationTypePinned`.
    pub const PINNED: u32 = 1;
}

/// `CUmemAllocationProp` for [`crate::Sim::va_get_allocation_properties`].
///
/// Compression and usage flags are not modeled. [`crate::Sim::va_create_with_prop`]
/// accepts [`MemHandleType::NONE`] only (POSIX-FD VMM export is not modeled).
/// Get always reports none. [`Self::gpu_direct_rdma_capable`] on create is
/// ignored (Get wraps the SKU).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemAllocationProp {
    /// `CUmemAllocationType` ([`MemAllocationType::PINNED`]).
    pub alloc_type: u32,
    /// `CUmemAllocationHandleType`. Create accepts
    /// [`MemHandleType::NONE`] only; Get always reports none.
    pub handle_types: u64,
    /// Handle GPU (`cudaMemLocationTypeDevice`).
    pub location: Place,
    /// `allocFlags.gpuDirectRDMACapable` (an RDMA link on that GPU).
    pub gpu_direct_rdma_capable: bool,
}

impl Default for MemAllocationProp {
    fn default() -> Self {
        Self {
            alloc_type: MemAllocationType::PINNED,
            handle_types: MemHandleType::NONE,
            location: Place::Device(DeviceId(0)),
            gpu_direct_rdma_capable: false,
        }
    }
}

/// `CUmemAllocationGranularity_flags` for
/// [`crate::Sim::va_get_allocation_granularity`].
pub struct MemAllocationGranularity;

impl MemAllocationGranularity {
    /// `CU_MEM_ALLOC_GRANULARITY_MINIMUM` (`0`).
    pub const MINIMUM: u32 = 0;
    /// `CU_MEM_ALLOC_GRANULARITY_RECOMMENDED` (`1`). This VM has one
    /// granularity; same as [`Self::MINIMUM`].
    pub const RECOMMENDED: u32 = 1;
}

/// `cuMemCreate` flags for [`crate::Sim::va_create_with_prop`].
///
/// CUDA requires 0. Unknown bits are Invalid `"mem create flags"`.
pub struct MemCreateFlags;

impl MemCreateFlags {
    /// Unflagged [`crate::Sim::va_create`].
    pub const DEFAULT: u32 = 0;
}

/// `cuMemAddressReserve` flags for [`crate::Sim::va_reserve_with_flags`].
///
/// CUDA requires 0. Unknown bits are Invalid `"mem reserve flags"`.
/// Nonzero `addr` is Invalid `"reserve addr"` (fixed VA is not modeled).
/// Nonzero `alignment` must be a power of two that divides `size`.
pub struct MemReserveFlags;

impl MemReserveFlags {
    /// Unflagged [`crate::Sim::va_reserve`].
    pub const DEFAULT: u32 = 0;
}

/// `cuMemMap` flags for [`crate::Sim::va_map_handle_with_flags`] /
/// [`crate::Sim::va_map_multicast_with_flags`].
///
/// CUDA requires 0. Unknown bits are Invalid `"mem map flags"`.
pub struct MemMapFlags;

impl MemMapFlags {
    /// Unflagged [`crate::Sim::va_map_handle`] / [`crate::Sim::va_map_multicast`].
    pub const DEFAULT: u32 = 0;
}

/// `CUmulticastGranularity_flags` for [`crate::Sim::multicast_get_granularity`]
/// / [`crate::Sim::multicast_get_granularity_with_prop`].
pub struct MulticastGranularity;

impl MulticastGranularity {
    /// `CU_MULTICAST_GRANULARITY_MINIMUM` (`0`).
    pub const MINIMUM: u32 = 0;
    /// `CU_MULTICAST_GRANULARITY_RECOMMENDED` (`1`). This VM has one
    /// granularity; same as [`Self::MINIMUM`].
    pub const RECOMMENDED: u32 = 1;
}

/// `CUmulticastObjectProp` for [`crate::Sim::multicast_create_with_prop`] /
/// [`crate::Sim::multicast_get_granularity_with_prop`].
///
/// Handle types other than [`MemHandleType::NONE`] are Invalid
/// `"multicast handle types"` (POSIX-FD multicast export is not modeled).
/// [`Self::flags`] must be [`MulticastCreateFlags::DEFAULT`].
/// Create requires [`Self::num_devices`] at least 2 and an aligned nonzero
/// size. GetGranularity does not (CUDA queries granularity before create).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MulticastObjectProp {
    /// Team size (`numDevices`). Create requires at least 2.
    pub num_devices: u32,
    /// Object bytes (`size`). Create requires a nonzero aligned size.
    pub size: u64,
    /// `CUmemAllocationHandleType`. Create accepts [`MemHandleType::NONE`] only.
    pub handle_types: u64,
    /// Create flags. CUDA requires 0.
    pub flags: u64,
}

impl Default for MulticastObjectProp {
    fn default() -> Self {
        Self {
            num_devices: 2,
            size: 0,
            handle_types: MemHandleType::NONE,
            flags: MulticastCreateFlags::DEFAULT,
        }
    }
}

/// `cuMulticastCreate` flags on [`MulticastObjectProp::flags`].
///
/// CUDA requires 0. Unknown bits are Invalid `"multicast create flags"`.
pub struct MulticastCreateFlags;

impl MulticastCreateFlags {
    /// Unflagged [`crate::Sim::multicast_create`].
    pub const DEFAULT: u64 = 0;
}

/// `cuMulticastBindAddr` / `BindMem` flags for
/// [`crate::Sim::multicast_bind_addr_with_flags`] /
/// [`crate::Sim::multicast_bind_mem_with_flags`].
///
/// CUDA requires 0. Unknown bits are Invalid `"multicast bind flags"`.
/// Partial offset/size bind is not modeled.
pub struct MulticastBindFlags;

impl MulticastBindFlags {
    /// Unflagged [`crate::Sim::multicast_bind_addr`] /
    /// [`crate::Sim::multicast_bind_mem`].
    pub const DEFAULT: u32 = 0;
}

/// `cudaMemPoolExportToShareableHandle` / `ImportFromShareableHandle` flags
/// for [`crate::Sim::pool_export_with_type`] /
/// [`crate::Sim::pool_import_with_type`].
pub struct MemPoolExportFlags;

impl MemPoolExportFlags {
    /// CUDA requires `0`.
    pub const DEFAULT: u32 = 0;
}

/// `cudaMemPoolProps` for [`crate::Sim::create_pool_with_props`].
///
/// [`Self::alloc_type`] must be [`MemAllocationType::PINNED`].
/// [`Self::handle_types`] is [`MemHandleType::NONE`] or
/// [`MemHandleType::POSIX_FILE_DESCRIPTOR`]. [`Self::location`] must be
/// [`Place::Device`]. [`Self::max_size`] `0` is unlimited; otherwise reserved
/// (`live + cached`) cannot grow past it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemPoolProps {
    /// `cudaMemAllocationType`.
    pub alloc_type: u32,
    /// `cudaMemAllocationHandleType` bits.
    pub handle_types: u64,
    /// Pool GPU (`cudaMemLocationTypeDevice`).
    pub location: Place,
    /// Maximum reserved bytes. `0` is unlimited.
    pub max_size: u64,
}

impl Default for MemPoolProps {
    fn default() -> Self {
        Self {
            alloc_type: MemAllocationType::PINNED,
            handle_types: MemHandleType::NONE,
            location: Place::Device(DeviceId(0)),
            max_size: 0,
        }
    }
}

/// `cudaMemPoolAttr` for [`crate::Sim::pool_get_attribute`].
///
/// Ordinary pools only. Graph-memory high-water stays on [`GraphMemAttr`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemPoolAttr {
    /// `cudaMemPoolAttrReleaseThreshold` ([`crate::Sim::set_pool_release_threshold`]).
    ReleaseThreshold,
    /// Live alloc bytes not yet freed (`cudaMemPoolAttrUsedMemCurrent`).
    UsedMemCurrent,
    /// High-water of [`Self::UsedMemCurrent`] (`cudaMemPoolAttrUsedMemHigh`).
    /// Set `0` resets to the current used bytes.
    UsedMemHigh,
    /// Live plus unused cached (`cudaMemPoolAttrReservedMemCurrent`).
    ReservedMemCurrent,
    /// High-water of [`Self::ReservedMemCurrent`]
    /// (`cudaMemPoolAttrReservedMemHigh`). Set `0` resets to the current
    /// reserved bytes.
    ReservedMemHigh,
    /// `cudaMemPoolReuseFollowEventDependencies`. Default 1. This VM's reuse is
    /// completion-based; 0 does not insert event waits.
    ReuseFollowEventDependencies,
    /// `cudaMemPoolReuseAllowOpportunistic`. Default 1. `0` skips cache reuse
    /// (OS alloc; unused cached bytes stay reserved).
    ReuseAllowOpportunistic,
    /// `cudaMemPoolReuseAllowInternalDependencies`. Default 1. This VM does not
    /// insert extra sync; 0 does not change opportunistic reuse.
    ReuseAllowInternalDependencies,
}

/// `cudaMemAccessFlags` for [`crate::Sim::pool_get_access`] /
/// [`crate::Sim::va_get_access`].
pub struct MemAccessFlags;

impl MemAccessFlags {
    /// `cudaMemAccessFlagsProtNone`.
    pub const PROT_NONE: u32 = 0;
    /// `cudaMemAccessFlagsProtRead`. Not modeled for pools (Invalid
    /// `"pool prot read"`). VMM uses [`crate::Sim::va_set_access`].
    pub const PROT_READ: u32 = 1;
    /// `cudaMemAccessFlagsProtReadWrite` ([`crate::Sim::pool_set_access`] /
    /// [`crate::Sim::va_set_access_write`]).
    pub const PROT_READ_WRITE: u32 = 3;
}

/// `CUmemAccessDesc` / `cudaMemAccessDesc` for [`crate::Sim::va_set_access_n`]
/// / [`crate::Sim::pool_set_access_n`].
///
/// [`Self::location`] must be [`Place::Device`]. Host is Invalid
/// `"access location"`. Flags are [`MemAccessFlags`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemAccessDesc {
    /// Device that gains or loses access (`CUmemLocation`).
    pub location: Place,
    /// [`MemAccessFlags::PROT_READ`] / [`PROT_READ_WRITE`](MemAccessFlags::PROT_READ_WRITE)
    /// / [`PROT_NONE`](MemAccessFlags::PROT_NONE).
    pub flags: u32,
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
    /// Next captured node's full dependency set (`cudaStreamGetCaptureInfo_v2`
    /// `dependencies`): last same-stream captured node (destination-graph
    /// index) union [`Self::pending_deps`].
    pub dependencies: Vec<usize>,
    /// Capture-sequence id (`cudaStreamGetCaptureInfo` `id_out`).
    ///
    /// Unique per [`crate::Sim::begin_capture`] / `begin_capture_to_graph`.
    /// Forked streams in the same session share it. Not a [`GraphId`].
    pub id: u64,
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

/// `cudaGraphNodeParams` for [`crate::Sim::graph_add_node`] /
/// [`crate::Sim::graph_node_get_params`].
///
/// IF/WHILE/SWITCH stay [`crate::Sim::graph_add_if`] / `graph_add_while` /
/// `graph_add_switch` (those return body graphs). Set-conditional is
/// [`crate::Sim::graph_add_set_conditional`] / [`Self::SetConditional`].
/// External-semaphore nodes are not modeled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphNodeParams {
    /// `cudaGraphKernelNode`.
    Kernel(KernelNodeParams),
    /// `cudaGraphMemcpyNode`. Pageable copies are Invalid.
    Memcpy(MemcpyOp),
    /// `cudaGraphMemsetNode`.
    Memset(MemsetOp),
    /// `cudaGraphHostNode`.
    Host(HostNodeParams),
    /// `cudaGraphEmptyNode`.
    Empty,
    /// `cudaGraphEventRecordNode`.
    EventRecord {
        /// Event to record.
        event: EventId,
        /// `cudaEventRecordExternal`.
        external: bool,
    },
    /// `cudaGraphEventWaitNode`.
    EventWait {
        /// Event to wait.
        event: EventId,
        /// `cudaEventWaitExternal`.
        external: bool,
    },
    /// `cudaGraphChildGraphNode`. Child must already be instantiated.
    ChildGraph(GraphId),
    /// `cudaGraphMemAllocNode`. [`GraphAddNode::alloc`] is the pending id.
    Alloc {
        /// Bytes for the pending `cudaMallocAsync`.
        bytes: u64,
    },
    /// `cudaGraphMemFreeNode`.
    Free(AllocId),
    /// `cudaGraphBatchMemOpNode` (item list; empty is Invalid).
    BatchMemOp(Vec<BatchMemOp>),
    /// Captured / graph-build [`crate::Sim::set_conditional`]
    /// (`cudaGraphSetConditional`). [`Self::SetConditional::handle`] is
    /// topology; `value` is a parameter.
    SetConditional {
        /// Handle created with [`crate::Sim::graph_conditional_create`].
        handle: CondId,
        /// Value written when this node starts.
        value: u32,
    },
}

/// Result of [`crate::Sim::graph_add_node`] (`cudaGraphAddNode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphAddNode {
    /// New node index in add order (`cudaGraphNode_t` analog).
    pub node: usize,
    /// Filled for [`GraphNodeParams::Alloc`] (`cudaMemAllocNodeParams::dptr`).
    pub alloc: Option<AllocId>,
}
