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
/// [`StreamId::NULL`] (`0`) is the CUDA null stream. Streams are
/// `cudaStreamNonBlocking` unless [`crate::Sim::set_legacy_null_stream`] is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamId(pub u16);

impl StreamId {
    /// CUDA null / default stream (`cudaStream_t` 0).
    pub const NULL: Self = Self(0);
}

/// Cross-stream ordering token. Record on one stream, wait on another.
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
