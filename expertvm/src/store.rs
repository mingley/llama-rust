//! Expert weight store: identity path and bounded cache.

use crate::access::{ExpertKey, Trace};
use crate::error::Error;
use crate::policy::Policy;
use crate::replay::{replay, ReplayRow};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Bytes returned for one expert tensor. Tests use a tiny payload.
pub type ExpertBlob = Vec<u8>;

/// Fetch expert tensors. Implementations must not change numeric identity
/// of the returned bytes for a given key (DirectStore contract).
pub trait ExpertStore {
    /// Load `key`. `Ok(blob)` on hit or fill; `Err` if the key is unknown.
    fn acquire(&mut self, key: ExpertKey) -> Result<ExpertBlob, Error>;
    /// Hits observed since construction.
    fn hits(&self) -> u64;
    /// Misses observed since construction.
    fn misses(&self) -> u64;
}

/// Always-resident store. Every acquire is a hit. Used as the identity oracle.
#[derive(Clone, Debug)]
pub struct DirectStore {
    blobs: BTreeMap<ExpertKey, ExpertBlob>,
    hits: u64,
}

impl DirectStore {
    /// Build from an explicit map.
    #[must_use]
    pub fn new(blobs: BTreeMap<ExpertKey, ExpertBlob>) -> Self {
        Self { blobs, hits: 0 }
    }

    /// One dummy byte per unique key in `trace`.
    #[must_use]
    pub fn from_trace(trace: &Trace) -> Self {
        let mut blobs = BTreeMap::new();
        for a in &trace.events {
            for k in a.keys() {
                let _blob = blobs.entry(k).or_insert_with(|| vec![1]);
            }
        }
        Self { blobs, hits: 0 }
    }

    /// Identity fetch without bumping hit counters (cache fill path).
    pub fn get(&self, key: ExpertKey) -> Result<ExpertBlob, Error> {
        match self.blobs.get(&key) {
            Some(b) => Ok(b.clone()),
            None => Err(Error::Store("unknown expert")),
        }
    }
}

impl ExpertStore for DirectStore {
    fn acquire(&mut self, key: ExpertKey) -> Result<ExpertBlob, Error> {
        let b = self.get(key)?;
        self.hits = self.hits.saturating_add(1);
        Ok(b)
    }

    fn hits(&self) -> u64 {
        self.hits
    }

    fn misses(&self) -> u64 {
        0
    }
}

/// Bounded LRU cache in front of a [`DirectStore`]. Leases pin keys until `release`.
pub struct CachedStore {
    inner: DirectStore,
    slots: usize,
    resident: BTreeSet<ExpertKey>,
    recency: VecDeque<ExpertKey>,
    leased: BTreeSet<ExpertKey>,
    hits: u64,
    misses: u64,
    evicts: u64,
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
            hits: 0,
            misses: 0,
            evicts: 0,
        })
    }

    /// Pin `key` so LRU will not evict it. Must be resident.
    pub fn lease(&mut self, key: ExpertKey) -> Result<(), Error> {
        if !self.resident.contains(&key) {
            return Err(Error::Store("lease of non-resident expert"));
        }
        let _leased = self.leased.insert(key);
        Ok(())
    }

    /// Drop a lease.
    pub fn release(&mut self, key: ExpertKey) {
        let _released = self.leased.remove(&key);
    }

    /// Evictions since construction.
    #[must_use]
    pub fn evicts(&self) -> u64 {
        self.evicts
    }

    fn evict_lru(&mut self) -> Result<(), Error> {
        let victim = self
            .recency
            .iter()
            .copied()
            .find(|k| !self.leased.contains(k));
        match victim {
            Some(v) => {
                let _removed = self.resident.remove(&v);
                self.recency.retain(|k| *k != v);
                self.evicts = self.evicts.saturating_add(1);
                Ok(())
            }
            None => Err(Error::Store("all resident experts are leased")),
        }
    }
}

impl ExpertStore for CachedStore {
    fn acquire(&mut self, key: ExpertKey) -> Result<ExpertBlob, Error> {
        if self.resident.contains(&key) {
            self.hits = self.hits.saturating_add(1);
            self.recency.retain(|k| *k != key);
            self.recency.push_back(key);
            return self.inner.get(key);
        }
        self.misses = self.misses.saturating_add(1);
        if self.resident.len() >= self.slots {
            self.evict_lru()?;
        }
        let blob = self.inner.get(key)?;
        let _inserted = self.resident.insert(key);
        self.recency.push_back(key);
        Ok(blob)
    }

    fn hits(&self) -> u64 {
        self.hits
    }

    fn misses(&self) -> u64 {
        self.misses
    }
}

/// Offline hit-rate for a trace (same as [`replay`] with lookahead 8).
#[must_use]
pub fn replay_accesses(trace: &Trace, slots: usize, policy: Policy) -> ReplayRow {
    replay(trace, slots, policy, 8)
}
