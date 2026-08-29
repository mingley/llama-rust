//! One store handle the decoder can attach: identity, cached, or simulated GPU.

use crate::access::ExpertKey;
use crate::error::Error;
use crate::gpu_store::SimulatedGpuStore;
use crate::store::{CachedStore, DirectStore, ExpertParts, ExpertStore, StoreMetrics};

/// Runtime backend for [`crate::ExpertStore`] on a decode session.
pub enum LiveStore {
    /// Always-resident identity copies.
    Direct(DirectStore),
    /// Bounded LRU + leases.
    Cached(CachedStore),
    /// LRU plus [`gpu_sim`] H2D / GEMM / optional NVLink replica.
    Simulated(Box<SimulatedGpuStore>),
}

impl LiveStore {
    /// Wrap a simulated GPU store (boxed so the enum stays small).
    #[must_use]
    pub fn simulated(store: SimulatedGpuStore) -> Self {
        Self::Simulated(Box::new(store))
    }

    /// Fault-in without counting a compute acquire. Unknown keys are skipped.
    pub fn prefetch(&mut self, keys: &[ExpertKey]) -> Result<u64, Error> {
        match self {
            Self::Direct(_) => Ok(0),
            Self::Cached(s) => s.prefetch(keys),
            Self::Simulated(s) => s.prefetch(keys),
        }
    }

    /// Keep-hot (lease) and, for the simulated GPU, replicate to a peer.
    pub fn pin_hot(&mut self, keys: &[ExpertKey]) -> Result<(), Error> {
        match self {
            Self::Direct(_) => Ok(()),
            Self::Cached(s) => s.pin_hot(keys),
            Self::Simulated(s) => s.pin_hot(keys),
        }
    }

    /// Fast-tier residency. [`DirectStore`] is always resident.
    #[must_use]
    pub fn is_resident(&self, key: ExpertKey) -> bool {
        match self {
            Self::Direct(s) => s.contains(key),
            Self::Cached(s) => s.is_resident(key),
            Self::Simulated(s) => s.is_resident(key),
        }
    }

    /// Simulated performance vector, if this is a GPU store.
    pub fn score(&mut self) -> Result<Option<gpu_sim::Score>, Error> {
        match self {
            Self::Simulated(s) => Ok(Some(s.score()?)),
            Self::Direct(_) | Self::Cached(_) => Ok(None),
        }
    }
}

impl ExpertStore for LiveStore {
    fn acquire(&mut self, key: ExpertKey) -> Result<ExpertParts, Error> {
        match self {
            Self::Direct(s) => s.acquire(key),
            Self::Cached(s) => s.acquire(key),
            Self::Simulated(s) => s.acquire(key),
        }
    }

    fn lease(&mut self, key: ExpertKey) -> Result<(), Error> {
        match self {
            Self::Direct(s) => s.lease(key),
            Self::Cached(s) => s.lease(key),
            Self::Simulated(s) => s.lease(key),
        }
    }

    fn release(&mut self, key: ExpertKey) {
        match self {
            Self::Direct(s) => s.release(key),
            Self::Cached(s) => s.release(key),
            Self::Simulated(s) => s.release(key),
        }
    }

    fn metrics(&self) -> StoreMetrics {
        match self {
            Self::Direct(s) => s.metrics(),
            Self::Cached(s) => s.metrics(),
            Self::Simulated(s) => s.metrics(),
        }
    }
}
