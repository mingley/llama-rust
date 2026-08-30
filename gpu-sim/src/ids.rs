//! Newtypes so a stream id cannot be passed where a device id is required.

use core::fmt;

/// One simulated GPU in the profile, `0 .. n_gpus`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceId(pub u16);

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gpu{}", self.0)
    }
}

/// CUDA-like stream on one device. Independent streams are unordered until an event.
///
/// [`StreamId::NULL`] (`0`) is the CUDA null stream. Created streams default
/// to `cudaStreamNonBlocking`. [`crate::Sim::set_stream_blocking`] models
/// `cudaStreamCreate` (serialize with NULL). [`crate::Sim::set_legacy_null_stream`]
/// makes NULL serialize with every stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamId(pub u16);

impl StreamId {
    /// CUDA null / default stream (`cudaStream_t` 0).
    pub const NULL: Self = Self(0);
}

/// Cross-stream ordering token. Record on one stream, wait on another.
///
/// [`crate::Sim::create_event_disable_timing`] is `cudaEventDisableTiming`
/// (elapsed fails; wait/query still work). Implicit create on record is
/// timing-enabled (`cudaEventDefault`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EventId(pub u32);

/// Stream-ordered allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AllocId(pub u64);

/// Submitted operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpId(pub u64);

/// Bidirectional interconnect between two places.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LinkId(pub u16);

/// Captured CUDA-like graph. Launch replays the recorded ops; capture does not execute them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphId(pub u32);

/// CUDA memory pool (`cudaMemPool_t`). [`crate::Sim::alloc`] uses the device default pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PoolId(pub u32);

impl fmt::Display for PoolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pool{}", self.0)
    }
}

/// Physical VMM allocation (`CUmemGenericAllocationHandle`). [`crate::Sim::va_create`]
/// charges HBM; [`crate::Sim::va_map_handle`] maps it into a reserved VA without a
/// second charge. [`crate::Sim::va_retain_handle`] increments handle refs.
/// [`crate::Sim::va_release_handle`] is `cuMemRelease` (allowed while mapped).
/// [`crate::Sim::va_map`] still Create+Maps in one call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemHandleId(pub u64);

impl fmt::Display for MemHandleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "handle{}", self.0)
    }
}

/// `cudaIpcMemHandle_t`. [`crate::Sim::ipc_get`] exports a device alloc;
/// [`crate::Sim::ipc_open`] imports an alias that shares the same physicals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IpcHandleId(pub u64);

impl fmt::Display for IpcHandleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ipc{}", self.0)
    }
}

/// `cudaMemPoolExportToShareableHandle` token.
///
/// [`crate::Sim::pool_export`] exports a POSIX-FD shareable pool;
/// [`crate::Sim::pool_import`] is a new [`PoolId`] that shares live/cached
/// bytes with the exporter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShareableHandleId(pub u64);

impl fmt::Display for ShareableHandleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "share{}", self.0)
    }
}

/// `cudaMemPoolExportPointer` token.
///
/// [`crate::Sim::pool_export_ptr`] exports a live pool allocation;
/// [`crate::Sim::pool_import_ptr`] imports an alias into an imported pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PtrExportId(pub u64);

impl fmt::Display for PtrExportId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ptr{}", self.0)
    }
}

/// Hopper multicast object (`CUmemGenericAllocationHandle` from
/// [`crate::Sim::multicast_create`] / `cuMulticastCreate`).
///
/// [`crate::Sim::multicast_bind_mem`] binds a [`MemHandleId`] per device.
/// [`crate::Sim::va_map_multicast`] maps the object into a reserved VA so a
/// kernel write fans out over NVLink (NVLS), not N sequential P2P copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MulticastId(pub u32);

impl fmt::Display for MulticastId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mc{}", self.0)
    }
}

/// `cudaGraphConditionalHandle`. Created on a graph; sampled by an IF node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CondId(pub u32);

impl fmt::Display for CondId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cond{}", self.0)
    }
}
