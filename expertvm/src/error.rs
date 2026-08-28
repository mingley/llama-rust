//! Crate errors.

use core::fmt;

/// Recoverable expertvm failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// JSONL / CLI parse.
    Trace(&'static str),
    /// Store invariant (lease, capacity, unknown key).
    Store(&'static str),
    /// Wrapped gpu-sim semantic failure.
    Sim(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Trace(s) | Self::Store(s) => write!(f, "{s}"),
            Self::Sim(s) => write!(f, "sim: {s}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<gpu_sim::SimError> for Error {
    fn from(e: gpu_sim::SimError) -> Self {
        Self::Sim(e.to_string())
    }
}
