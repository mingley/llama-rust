//! Expert weight store: identity path, bounded cache, leases, prefetch.

use crate::access::{ExpertKey, Trace};
use crate::error::Error;
use crate::policy::Policy;
use crate::replay::{replay, ReplayRow};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Gate / up / down bytes for one routed expert. DirectStore identity is
/// bit-identical copies of the GGUF `*_exps` parts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExpertParts {
    /// `ffn_gate_exps` part.
    pub gate: Vec<u8>,
    /// `ffn_up_exps` part.
    pub up: Vec<u8>,
    /// `ffn_down_exps` part.
    pub down: Vec<u8>,
}

impl ExpertParts {
    /// Payload bytes of the three tensors.
    #[must_use]
    pub fn nbytes(&self) -> u64 {
        let n = self
            .gate
            .len()
            .saturating_add(self.up.len())
            .saturating_add(self.down.len());
        u64::try_from(n).unwrap_or(u64::MAX)
    }
}

/// PLAN expert residency: Cold → Transferring → Resident → Leased → Evicting → Cold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertPhase {
    /// Not in the fast tier and not in flight.
    Cold,
    /// Copy onto the fast tier has been submitted and has not completed.
    Transferring,
    /// Fast-tier resident and not leased.
    Resident,
    /// Pinned against eviction (in-use).
    Leased,
    /// Victim copy-out / free in flight.
    Evicting,
}

impl ExpertPhase {
    /// CPU caches fault in on the calling thread: Resident or Leased, else Cold.
    #[must_use]
    pub fn cpu(resident: bool, leased: bool) -> Self {
        if leased {
            Self::Leased
        } else if resident {
            Self::Resident
        } else {
            Self::Cold
        }
    }
}

/// Hit / miss / movement counters. Hits are cache-layer, not DirectStore.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StoreMetrics {
    /// Cache hits.
    pub hits: u64,
    /// Cache misses (fault-ins).
    pub misses: u64,
    /// Evictions.
    pub evicts: u64,
    /// Prefetch fills that were not already resident.
    pub prefetches: u64,
    /// Host↔device bytes ([`crate::SimulatedGpuStore`]); 0 for CPU stores.
    pub bytes_moved: u64,
    /// Sticky [`CachedStore::pin_hot`] inserts (not in-flight [`ExpertStore::lease`]s).
    pub pins: u64,
    /// D2D [`crate::SimulatedGpuStore::migrate`] moves (src ≠ dst); 0 for CPU stores.
    pub migrates: u64,
    /// [`crate::plan_placement`] chose [`crate::Placement::DispatchActivations`]
    /// (leave weights on the striped home); 0 for CPU stores.
    pub dispatches: u64,
    /// NVLink/peer replica copies from [`crate::SimulatedGpuStore::pin_hot`]; 0 for CPU stores.
    pub replicates: u64,
}

impl StoreMetrics {
    /// Single-line log.
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "hits={} misses={} evicts={} prefetches={} bytes_moved={} pins={} migrates={} dispatches={} replicates={}",
            self.hits,
            self.misses,
            self.evicts,
            self.prefetches,
            self.bytes_moved,
            self.pins,
            self.migrates,
            self.dispatches,
            self.replicates
        )
    }
}

/// Fetch expert tensors. [`DirectStore`] is the identity oracle.
pub trait ExpertStore {
    /// Load `key`. `Ok(parts)` on hit or fill; `Err` if the key is unknown.
    fn acquire(&mut self, key: ExpertKey) -> Result<ExpertParts, Error>;
    /// Pin `key` against eviction. Must be resident ([`CachedStore`]).
    /// In-flight only: [`CachedStore::release`] drops this. Sticky keep-hot
    /// is [`CachedStore::pin_hot`].
    fn lease(&mut self, key: ExpertKey) -> Result<(), Error>;
    /// Drop a lease.
    fn release(&mut self, key: ExpertKey);
    /// Snapshot counters.
    fn metrics(&self) -> StoreMetrics;
    /// Hits observed since construction.
    fn hits(&self) -> u64 {
        self.metrics().hits
    }
    /// Misses observed since construction.
    fn misses(&self) -> u64 {
        self.metrics().misses
    }
}

/// Always-resident store. Every acquire is a hit. Used as the identity oracle.
#[derive(Clone, Debug)]
pub struct DirectStore {
    blobs: BTreeMap<ExpertKey, ExpertParts>,
    hits: u64,
}

impl DirectStore {
    /// Build from an explicit map of real GGUF parts.
    #[must_use]
    pub fn new(blobs: BTreeMap<ExpertKey, ExpertParts>) -> Self {
        Self { blobs, hits: 0 }
    }

    /// Dummy one-byte parts per unique key in `trace` (replay tests).
    #[must_use]
    pub fn from_trace(trace: &Trace) -> Self {
        let mut blobs = BTreeMap::new();
        for a in &trace.events {
            for k in a.keys() {
                let _p = blobs.entry(k).or_insert_with(|| ExpertParts {
                    gate: vec![1],
                    up: vec![1],
                    down: vec![1],
                });
            }
        }
        Self { blobs, hits: 0 }
    }

    /// Identity fetch without bumping hit counters (cache fill path).
    pub fn get(&self, key: ExpertKey) -> Result<ExpertParts, Error> {
        match self.blobs.get(&key) {
            Some(b) => Ok(b.clone()),
            None => Err(Error::Store("unknown expert")),
        }
    }

    /// Whether `key` is in the catalog.
    #[must_use]
    pub fn contains(&self, key: ExpertKey) -> bool {
        self.blobs.contains_key(&key)
    }

    /// Number of catalogued experts.
    #[must_use]
    pub fn len(&self) -> usize {
        self.blobs.len()
    }

    /// Catalog keys in sorted order.
    pub fn keys(&self) -> impl Iterator<Item = ExpertKey> + '_ {
        self.blobs.keys().copied()
    }

    /// Empty catalog.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blobs.is_empty()
    }
}

impl ExpertStore for DirectStore {
    fn acquire(&mut self, key: ExpertKey) -> Result<ExpertParts, Error> {
        let b = self.get(key)?;
        self.hits = self.hits.saturating_add(1);
        Ok(b)
    }

    fn lease(&mut self, _key: ExpertKey) -> Result<(), Error> {
        Ok(())
    }

    fn release(&mut self, _key: ExpertKey) {}

    fn metrics(&self) -> StoreMetrics {
        StoreMetrics {
            hits: self.hits,
            ..StoreMetrics::default()
        }
    }
}

/// Bounded LRU cache in front of a [`DirectStore`]. Leases pin keys until `release`.
pub struct CachedStore {
    inner: DirectStore,
    slots: usize,
    resident: BTreeSet<ExpertKey>,
    recency: VecDeque<ExpertKey>,
    leased: BTreeSet<ExpertKey>,
    pinned: BTreeSet<ExpertKey>,
    hits: u64,
    misses: u64,
    evicts: u64,
    prefetches: u64,
    pins: u64,
    last_victim: Option<ExpertKey>,
}

impl CachedStore {
    /// `slots` concurrent resident experts. Zero is rejected.
    pub fn new(inner: DirectStore, slots: usize) -> Result<Self, Error> {
        if slots == 0 {
            return Err(Error::Store("cache slots must be > 0"));
        }
        Ok(Self {
            inner,
            slots,
            resident: BTreeSet::new(),
            recency: VecDeque::new(),
            leased: BTreeSet::new(),
            pinned: BTreeSet::new(),
            hits: 0,
            misses: 0,
            evicts: 0,
            prefetches: 0,
            pins: 0,
            last_victim: None,
        })
    }

    /// Fault `keys` in without treating them as compute acquires.
    /// Unknown catalog keys are skipped (copy-forward may name a missing layer).
    pub fn prefetch(&mut self, keys: &[ExpertKey]) -> Result<u64, Error> {
        let mut n = 0u64;
        for key in keys {
            if self.resident.contains(key) || !self.inner.contains(*key) {
                continue;
            }
            self.fault_in(*key)?;
            self.prefetches = self.prefetches.saturating_add(1);
            n = n.saturating_add(1);
        }
        Ok(n)
    }

    /// Fast-tier capacity (concurrent resident experts).
    #[must_use]
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Sticky keep-hot budget: leave one slot for demand paging. `slots == 1` is 0.
    #[must_use]
    pub fn pin_budget(&self) -> usize {
        self.slots.saturating_sub(1)
    }

    /// Pin `keys` (hot replication / keep-hot). Faults in if needed.
    ///
    /// Distinct from [`ExpertStore::lease`]: a decode `release` does not drop
    /// the pin. [`Self::unpin_all`] clears sticky pins.
    pub fn pin_hot(&mut self, keys: &[ExpertKey]) -> Result<(), Error> {
        for key in keys {
            if !self.inner.contains(*key) {
                continue;
            }
            if !self.resident.contains(key) {
                self.fault_in(*key)?;
            }
            if self.pinned.insert(*key) {
                self.pins = self.pins.saturating_add(1);
            }
        }
        Ok(())
    }

    /// Drop every sticky pin. In-flight leases are unchanged.
    pub fn unpin_all(&mut self) {
        self.pinned.clear();
    }

    /// Whether `key` is in the fast tier.
    #[must_use]
    pub fn is_resident(&self, key: ExpertKey) -> bool {
        self.resident.contains(&key)
    }

    /// Whether `key` has an in-flight compute lease.
    #[must_use]
    pub fn is_leased(&self, key: ExpertKey) -> bool {
        self.leased.contains(&key)
    }

    /// Whether `key` has a sticky [`Self::pin_hot`] pin.
    #[must_use]
    pub fn is_pinned(&self, key: ExpertKey) -> bool {
        self.pinned.contains(&key)
    }

    /// In-flight lease or sticky pin: LRU must not evict.
    #[must_use]
    pub fn is_held(&self, key: ExpertKey) -> bool {
        self.leased.contains(&key) || self.pinned.contains(&key)
    }

    /// PLAN state for `key`. CPU fault-in is instantaneous (no Transferring).
    #[must_use]
    pub fn phase(&self, key: ExpertKey) -> ExpertPhase {
        ExpertPhase::cpu(self.resident.contains(&key), self.is_held(key))
    }

    /// Whether `key` exists in the backing catalog.
    #[must_use]
    pub fn contains_catalog(&self, key: ExpertKey) -> bool {
        self.inner.contains(key)
    }

    /// Last key evicted, if any. Cleared by [`Self::take_victim`].
    #[must_use]
    pub fn last_victim(&self) -> Option<ExpertKey> {
        self.last_victim
    }

    /// Take the victim of the most recent [`Self::fault_in`], if it evicted.
    pub fn take_victim(&mut self) -> Option<ExpertKey> {
        self.last_victim.take()
    }

    /// Evictions since construction.
    #[must_use]
    pub fn evicts(&self) -> u64 {
        self.evicts
    }

    /// Drop `key` from the fast tier. Illegal while leased/pinned or if not resident.
    pub fn evict(&mut self, key: ExpertKey) -> Result<(), Error> {
        if self.is_held(key) {
            return Err(Error::Store("evict of leased expert"));
        }
        if !self.resident.contains(&key) {
            return Err(Error::Store("evict of non-resident expert"));
        }
        let _removed = self.resident.remove(&key);
        self.recency.retain(|k| *k != key);
        self.evicts = self.evicts.saturating_add(1);
        self.last_victim = Some(key);
        Ok(())
    }

    fn fault_in(&mut self, key: ExpertKey) -> Result<(), Error> {
        self.last_victim = None;
        if self.resident.len() >= self.slots {
            self.evict_lru()?;
        }
        let _blob = self.inner.get(key)?;
        let _inserted = self.resident.insert(key);
        self.recency.push_back(key);
        Ok(())
    }

    fn evict_lru(&mut self) -> Result<(), Error> {
        let victim = self.recency.iter().copied().find(|k| !self.is_held(*k));
        match victim {
            Some(v) => {
                let _removed = self.resident.remove(&v);
                self.recency.retain(|k| *k != v);
                self.evicts = self.evicts.saturating_add(1);
                self.last_victim = Some(v);
                Ok(())
            }
            None => Err(Error::Store("all resident experts are leased")),
        }
    }
}

impl ExpertStore for CachedStore {
    fn acquire(&mut self, key: ExpertKey) -> Result<ExpertParts, Error> {
        if self.resident.contains(&key) {
            self.hits = self.hits.saturating_add(1);
            self.recency.retain(|k| *k != key);
            self.recency.push_back(key);
            return self.inner.get(key);
        }
        self.misses = self.misses.saturating_add(1);
        self.fault_in(key)?;
        self.inner.get(key)
    }

    fn lease(&mut self, key: ExpertKey) -> Result<(), Error> {
        if !self.resident.contains(&key) {
            return Err(Error::Store("lease of non-resident expert"));
        }
        let _leased = self.leased.insert(key);
        Ok(())
    }

    fn release(&mut self, key: ExpertKey) {
        let _released = self.leased.remove(&key);
    }

    fn metrics(&self) -> StoreMetrics {
        StoreMetrics {
            hits: self.hits,
            misses: self.misses,
            evicts: self.evicts,
            prefetches: self.prefetches,
            bytes_moved: 0,
            pins: self.pins,
            migrates: 0,
            dispatches: 0,
            replicates: 0,
        }
    }
}

/// Offline hit-rate for a trace (same as [`replay`] with lookahead 8).
#[must_use]
pub fn replay_accesses(trace: &Trace, slots: usize, policy: Policy) -> ReplayRow {
    replay(trace, slots, policy, 8)
}
