//! Semantic failures. A policy that produces one of these is rejected.

use core::fmt;

use crate::ids::{AllocId, DeviceId};

/// Why the simulator refused an operation or detected an illegal GPU state.
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
    /// Event wait before any record, or unknown event id.
    UnknownEvent {
        /// Event id.
        event: u32,
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
            Self::UnknownEvent { event } => write!(f, "unknown event {event}"),
            Self::Invalid { why } => write!(f, "invalid: {why}"),
        }
    }
}

impl std::error::Error for SimError {}
