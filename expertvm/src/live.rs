//! One store handle the decoder can attach: identity, cached, tiered, or simulated GPU.

use crate::access::ExpertKey;
use crate::error::Error;
use crate::gpu_store::SimulatedGpuStore;
use crate::store::{CachedStore, DirectStore, ExpertParts, ExpertStore, StoreMetrics};
use crate::tiered::TieredStore;
use gpu_sim::DeviceId;

/// Runtime backend for [`crate::ExpertStore`] on a decode session.
pub enum LiveStore {
    /// Always-resident identity copies.
    Direct(DirectStore),
    /// Bounded LRU + leases.
    Cached(CachedStore),
    /// Fast RAM / slow RAM / file paging ([`TieredStore`]).
    Tiered(Box<TieredStore>),
    /// LRU plus [`gpu_sim`] H2D / GEMM / optional NVLink replica.
    Simulated(Box<SimulatedGpuStore>),
}

impl LiveStore {
    /// Wrap a simulated GPU store (boxed so the enum stays small).
    #[must_use]
    pub fn simulated(store: SimulatedGpuStore) -> Self {
        Self::Simulated(Box::new(store))
    }

    /// Wrap a tiered store.
    #[must_use]
    pub fn tiered(store: TieredStore) -> Self {
        Self::Tiered(Box::new(store))
    }

    /// Fault-in without counting a compute acquire. Unknown keys are skipped.
    pub fn prefetch(&mut self, keys: &[ExpertKey]) -> Result<u64, Error> {
        match self {
            Self::Direct(_) => Ok(0),
            Self::Cached(s) => s.prefetch(keys),
            Self::Tiered(s) => s.prefetch(keys),
            Self::Simulated(s) => s.prefetch(keys),
        }
    }

    /// Keep-hot (sticky pin) and, for the simulated GPU, replicate to a peer.
    ///
    /// Distinct from [`ExpertStore::lease`]: a decode `release` does not drop
    /// the pin. Engine serving uses [`Self::unpin_all`] before a new pin set.
    pub fn pin_hot(&mut self, keys: &[ExpertKey]) -> Result<(), Error> {
        match self {
            Self::Direct(_) => Ok(()),
            Self::Cached(s) => s.pin_hot(keys),
            Self::Tiered(s) => s.pin_hot(keys),
            Self::Simulated(s) => s.pin_hot(keys),
        }
    }

    /// Drop sticky pins. Direct catalogs are a no-op. In-flight leases stay.
    pub fn unpin_all(&mut self) {
        match self {
            Self::Direct(_) => {}
            Self::Cached(s) => s.unpin_all(),
            Self::Tiered(s) => s.unpin_all(),
            Self::Simulated(s) => s.unpin_all(),
        }
    }

    /// Fast-tier slots. [`DirectStore`] has no cache.
    #[must_use]
    pub fn slots(&self) -> Option<usize> {
        match self {
            Self::Direct(_) => None,
            Self::Cached(s) => Some(s.slots()),
            Self::Tiered(s) => Some(s.slots()),
            Self::Simulated(s) => Some(s.slots()),
        }
    }

    /// Sticky keep-hot budget: `slots.saturating_sub(1)` so demand paging can evict.
    #[must_use]
    pub fn pin_budget(&self) -> usize {
        self.slots().unwrap_or(0).saturating_sub(1)
    }

    /// Whether `key` has a sticky [`Self::pin_hot`] pin.
    #[must_use]
    pub fn is_pinned(&self, key: ExpertKey) -> bool {
        match self {
            Self::Direct(_) => false,
            Self::Cached(s) => s.is_pinned(key),
            Self::Tiered(s) => s.is_pinned(key),
            Self::Simulated(s) => s.is_pinned(key),
        }
    }

    /// Fast-tier residency. [`DirectStore`] is always resident.
    #[must_use]
    pub fn is_resident(&self, key: ExpertKey) -> bool {
        match self {
            Self::Direct(s) => s.contains(key),
            Self::Cached(s) => s.is_resident(key),
            Self::Tiered(s) => s.is_resident(key),
            Self::Simulated(s) => s.is_resident(key),
        }
    }

    /// PLAN expert state machine. Direct catalogs are Resident when present.
    #[must_use]
    pub fn phase(&self, key: ExpertKey) -> crate::ExpertPhase {
        match self {
            Self::Direct(s) => crate::ExpertPhase::cpu(s.contains(key), false),
            Self::Cached(s) => s.phase(key),
            Self::Tiered(s) => s.phase(key),
            Self::Simulated(s) => s.phase(key),
        }
    }

    /// Drop `key` from the fast tier. Direct catalogs are a no-op.
    pub fn evict(&mut self, key: ExpertKey) -> Result<(), Error> {
        match self {
            Self::Direct(_) => Ok(()),
            Self::Cached(s) => s.evict(key),
            Self::Tiered(s) => s.evict(key),
            Self::Simulated(s) => s.evict(key),
        }
    }

    /// D2D migrate on the simulated GPU; no-op for CPU stores.
    pub fn migrate(&mut self, key: ExpertKey, dst: DeviceId) -> Result<(), Error> {
        match self {
            Self::Simulated(s) => s.migrate(key, dst),
            Self::Direct(_) | Self::Cached(_) | Self::Tiered(_) => Ok(()),
        }
    }

    /// Simulated performance vector, if this is a GPU store.
    pub fn score(&mut self) -> Result<Option<gpu_sim::Score>, Error> {
        match self {
            Self::Simulated(s) => Ok(Some(s.score()?)),
            Self::Direct(_) | Self::Cached(_) | Self::Tiered(_) => Ok(None),
        }
    }
}

impl ExpertStore for LiveStore {
    fn acquire(&mut self, key: ExpertKey) -> Result<ExpertParts, Error> {
        match self {
            Self::Direct(s) => s.acquire(key),
            Self::Cached(s) => s.acquire(key),
            Self::Tiered(s) => s.acquire(key),
            Self::Simulated(s) => s.acquire(key),
        }
    }

    fn lease(&mut self, key: ExpertKey) -> Result<(), Error> {
        match self {
            Self::Direct(s) => s.lease(key),
            Self::Cached(s) => s.lease(key),
            Self::Tiered(s) => s.lease(key),
            Self::Simulated(s) => s.lease(key),
        }
    }

    fn release(&mut self, key: ExpertKey) {
        match self {
            Self::Direct(s) => s.release(key),
            Self::Cached(s) => s.release(key),
            Self::Tiered(s) => s.release(key),
            Self::Simulated(s) => s.release(key),
        }
    }

    fn metrics(&self) -> StoreMetrics {
        match self {
            Self::Direct(s) => s.metrics(),
            Self::Cached(s) => s.metrics(),
            Self::Tiered(s) => s.metrics(),
            Self::Simulated(s) => s.metrics(),
        }
    }
}
