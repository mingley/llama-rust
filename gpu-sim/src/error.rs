//! Semantic failures. A policy that produces one of these is rejected.

use core::fmt;

use crate::ids::{AllocId, DeviceId, StreamId};

/// Why the simulator refused an operation or detected an illegal GPU state.
///
/// [`Self::error_name`] / [`Self::error_string`] are `cudaGetErrorName` /
/// `cudaGetErrorString`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SimError {
    /// Device HBM cannot hold this allocation.
    Oom {
        /// GPU that ran out.
        device: DeviceId,
        /// Requested bytes.
        need: u64,
        /// Bytes still free.
        free: u64,
    },
    /// Kernel or memcpy named an allocation that was never created, or already freed.
    UnknownAlloc {
        /// Missing id.
        alloc: AllocId,
    },
    /// `free` while a kernel still lists the allocation in its reads/writes (lease).
    Leased {
        /// Allocation still in use.
        alloc: AllocId,
    },
    /// Kernel read/write on a device that does not hold the allocation.
    NotResident {
        /// Allocation.
        alloc: AllocId,
        /// Kernel's device.
        device: DeviceId,
    },
    /// Peer copy between GPUs the topology does not connect.
    NoPeer {
        /// Source GPU.
        src: DeviceId,
        /// Destination GPU.
        dst: DeviceId,
    },
    /// Topology has a link, but [`crate::Sim::enable_peer`] is off (`cudaDeviceDisablePeerAccess`).
    PeerDisabled {
        /// Source GPU.
        src: DeviceId,
        /// Destination GPU.
        dst: DeviceId,
    },
    /// Event wait before any record, or unknown event id.
    UnknownEvent {
        /// Event id.
        event: u32,
    },
    /// The device was marked unavailable (fault injection / drain).
    Unavailable {
        /// GPU that will not accept new work.
        device: DeviceId,
    },
    /// Queued work on a stream was cancelled before it started.
    Cancelled {
        /// Stream whose pending ops were dropped.
        stream: StreamId,
        /// How many ops were skipped.
        n: u32,
    },
    /// Injected memcpy / expert-load failure (transfer never completed).
    TransferFailed {
        /// Allocation that did not move.
        alloc: AllocId,
    },
    /// Host pin / `mlock` budget (`cudaMallocHost` / `cudaHostRegister`).
    PinOom {
        /// Requested pin bytes.
        need: u64,
        /// Bytes still within the profile pin cap.
        free: u64,
    },
    /// Profile or submit argument that cannot occur on real hardware as modeled.
    Invalid {
        /// Human-readable reason.
        why: &'static str,
    },
}

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Oom { device, need, free } => {
                write!(f, "OOM on {device}: need {need} bytes, {free} free")
            }
            Self::UnknownAlloc { alloc } => write!(f, "unknown allocation {}", alloc.0),
            Self::Leased { alloc } => write!(f, "allocation {} is still leased", alloc.0),
            Self::NotResident { alloc, device } => {
                write!(f, "allocation {} not resident on {device}", alloc.0)
            }
            Self::NoPeer { src, dst } => write!(f, "no peer link {src} → {dst}"),
            Self::PeerDisabled { src, dst } => {
                write!(f, "peer access disabled {src} → {dst}")
            }
            Self::UnknownEvent { event } => write!(f, "unknown event {event}"),
            Self::Unavailable { device } => write!(f, "{device} unavailable"),
            Self::Cancelled { stream, n } => {
                write!(f, "cancelled {n} queued ops on stream {}", stream.0)
            }
            Self::TransferFailed { alloc } => {
                write!(f, "transfer failed for allocation {}", alloc.0)
            }
            Self::PinOom { need, free } => {
                write!(f, "host pin OOM: need {need} bytes, {free} free")
            }
            Self::Invalid { why } => write!(f, "invalid: {why}"),
        }
    }
}

impl std::error::Error for SimError {}

impl SimError {
    /// `cudaGetErrorName`. Query. No device and no capture session.
    ///
    /// This VM has no thread-local last error (`cudaGetLastError` is not
    /// modeled). Callers pass the [`SimError`] they already received.
    /// The `Display` impl stays the detailed reason.
    #[must_use]
    pub fn error_name(&self) -> &'static str {
        match self {
            Self::Oom { .. } | Self::PinOom { .. } => "cudaErrorMemoryAllocation",
            Self::UnknownAlloc { .. } => "cudaErrorInvalidDevicePointer",
            Self::Leased { .. } | Self::Invalid { .. } => "cudaErrorInvalidValue",
            Self::NotResident { .. } => "cudaErrorIllegalAddress",
            Self::NoPeer { .. } => "cudaErrorPeerAccessUnsupported",
            Self::PeerDisabled { .. } => "cudaErrorPeerAccessNotEnabled",
            Self::UnknownEvent { .. } => "cudaErrorInvalidResourceHandle",
            Self::Unavailable { .. } => "cudaErrorDevicesUnavailable",
            Self::Cancelled { .. } | Self::TransferFailed { .. } => "cudaErrorLaunchFailure",
        }
    }

    /// `cudaGetErrorString`. Query. No device and no capture session.
    ///
    /// CUDA generic sentences for named codes. [`Self::Invalid`] stays
    /// `"invalid argument"`; the modeled reason remains the `why` field
    /// and the `Display` impl.
    #[must_use]
    pub fn error_string(&self) -> &'static str {
        match self {
            Self::Oom { .. } | Self::PinOom { .. } => "out of memory",
            Self::UnknownAlloc { .. } => "invalid device pointer",
            Self::Leased { .. } | Self::Invalid { .. } => "invalid argument",
            Self::NotResident { .. } => "an illegal memory access was encountered",
            Self::NoPeer { .. } => "peer access is not supported between these two devices",
            Self::PeerDisabled { .. } => {
                "peer access has not been enabled between these two devices"
            }
            Self::UnknownEvent { .. } => "invalid resource handle",
            Self::Unavailable { .. } => "all CUDA-capable devices are busy or unavailable",
            Self::Cancelled { .. } | Self::TransferFailed { .. } => "unspecified launch failure",
        }
    }
}
