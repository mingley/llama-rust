//! Fast RAM / slow RAM / disk expert residency. mmap stays parked.

use crate::access::ExpertKey;
use crate::error::Error;
use crate::store::{DirectStore, ExpertParts, ExpertPhase, ExpertStore, StoreMetrics};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Where the catalog lives. Fast-tier RAM is always a bounded LRU of
/// [`ExpertParts`], independent of this choice.
///
/// mmap is parked: the default engine is `forbid(unsafe_code)`, and this
/// crate does not take a mmap crate. [`WeightStorage::mmap`] returns an
/// error pointing at [`Self::File`] seek+read paging, which is the same
/// residency demonstration (only `fast_slots` experts in RAM).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightStorage {
    /// Catalog is a [`DirectStore`] (slow RAM). Fault-in clones into the fast tier.
    InMemory,
    /// Catalog is a file. Fault-in seeks and reads. Not mmap.
    File,
    /// Catalog is a key set; fault-in synthesizes `fill` bytes.
    Synthetic,
}

impl WeightStorage {
    /// mmap is not compiled. Use [`Self::File`].
    pub fn mmap() -> Result<Self, Error> {
        Err(Error::Store(
            "mmap WeightStorage is parked; use File seek/read paging",
        ))
    }
}

/// Byte range of one expert in a paging file: gate, then up, then down.
#[derive(Clone, Copy, Debug)]
struct FileSpan {
    off: u64,
    gate: u64,
    up: u64,
    down: u64,
}

enum SlowTier {
    Memory(DirectStore),
    Disk {
        file: File,
        index: BTreeMap<ExpertKey, FileSpan>,
    },
    Synthetic {
        nbytes: usize,
        fill: u8,
        keys: BTreeSet<ExpertKey>,
    },
}

/// Bounded fast-RAM cache in front of slow RAM, a paging file, or synthetic bytes.
pub struct TieredStore {
    slow: SlowTier,
    slots: usize,
    fast: BTreeMap<ExpertKey, ExpertParts>,
    recency: VecDeque<ExpertKey>,
    leased: BTreeSet<ExpertKey>,
    hits: u64,
    misses: u64,
    evicts: u64,
    prefetches: u64,
    bytes_moved: u64,
}

impl TieredStore {
    /// Fast LRU in front of a RAM catalog. The catalog stays in slow RAM;
    /// only `slots` copies sit in the fast map.
    pub fn memory(inner: DirectStore, slots: usize) -> Result<Self, Error> {
        Self::new(SlowTier::Memory(inner), slots)
    }

    /// Write `inner` to `file` and page from disk. After return, only `slots`
    /// experts occupy RAM as [`ExpertParts`]; the catalog is the file index.
    ///
    /// `file` must be opened for **read and write** (see [`Self::on_path`]).
    /// `File::create` is write-only and will fail on the first fault-in.
    pub fn on_file(inner: DirectStore, slots: usize, mut file: File) -> Result<Self, Error> {
        let mut index = BTreeMap::new();
        let mut off = 0u64;
        for key in inner.keys() {
            let parts = inner.get(key)?;
            let span = FileSpan {
                off,
                gate: u64_len(&parts.gate),
                up: u64_len(&parts.up),
                down: u64_len(&parts.down),
            };
            write_all(&mut file, &parts.gate)?;
            write_all(&mut file, &parts.up)?;
            write_all(&mut file, &parts.down)?;
            off = off
                .saturating_add(span.gate)
                .saturating_add(span.up)
                .saturating_add(span.down);
            let _prev = index.insert(key, span);
        }
        file.flush().map_err(|e| Error::Io(e.to_string()))?;
        Self::new(SlowTier::Disk { file, index }, slots)
    }

    /// Create (or replace) `path` and page from it. Opens read+write.
    pub fn on_path(inner: DirectStore, slots: usize, path: &Path) -> Result<Self, Error> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| Error::Io(e.to_string()))?;
        Self::on_file(inner, slots, file)
    }

    /// Synthesize `nbytes` of `fill` per tensor on fault. No catalog RAM.
    pub fn synthetic(
        keys: BTreeSet<ExpertKey>,
        nbytes: usize,
        fill: u8,
        slots: usize,
    ) -> Result<Self, Error> {
        Self::new(SlowTier::Synthetic { nbytes, fill, keys }, slots)
    }

    fn new(slow: SlowTier, slots: usize) -> Result<Self, Error> {
        if slots == 0 {
            return Err(Error::Store("cache slots must be > 0"));
        }
        Ok(Self {
            slow,
            slots,
            fast: BTreeMap::new(),
            recency: VecDeque::new(),
            leased: BTreeSet::new(),
            hits: 0,
            misses: 0,
            evicts: 0,
            prefetches: 0,
            bytes_moved: 0,
        })
    }

    /// Slow-tier kind.
    #[must_use]
    pub fn storage(&self) -> WeightStorage {
        match self.slow {
            SlowTier::Memory(_) => WeightStorage::InMemory,
            SlowTier::Disk { .. } => WeightStorage::File,
            SlowTier::Synthetic { .. } => WeightStorage::Synthetic,
        }
    }

    /// Fast-tier occupancy.
    #[must_use]
    pub fn fast_len(&self) -> usize {
        self.fast.len()
    }

    /// Whether `key` is in the fast tier.
    #[must_use]
    pub fn is_resident(&self, key: ExpertKey) -> bool {
        self.fast.contains_key(&key)
    }

    /// PLAN state. Fault-in is a blocking read (no Transferring).
    #[must_use]
    pub fn phase(&self, key: ExpertKey) -> ExpertPhase {
        ExpertPhase::cpu(self.fast.contains_key(&key), self.leased.contains(&key))
    }

    /// Drop `key` from the fast tier. Illegal while leased or if not resident.
    pub fn evict(&mut self, key: ExpertKey) -> Result<(), Error> {
        if self.leased.contains(&key) {
            return Err(Error::Store("evict of leased expert"));
        }
        if !self.fast.contains_key(&key) {
            return Err(Error::Store("evict of non-resident expert"));
        }
        let _gone = self.fast.remove(&key);
        self.recency.retain(|k| *k != key);
        self.evicts = self.evicts.saturating_add(1);
        Ok(())
    }

    /// Whether `key` exists in the slow catalog.
    #[must_use]
    pub fn contains_catalog(&self, key: ExpertKey) -> bool {
        self.slow_has(key)
    }

    /// Fault `keys` in without treating them as compute acquires.
    pub fn prefetch(&mut self, keys: &[ExpertKey]) -> Result<u64, Error> {
        let mut n = 0u64;
        for key in keys {
            if self.fast.contains_key(key) || !self.slow_has(*key) {
                continue;
            }
            self.fault_in(*key)?;
            self.prefetches = self.prefetches.saturating_add(1);
            n = n.saturating_add(1);
        }
        Ok(n)
    }

    /// Pin `keys` against eviction. Faults in if needed.
    pub fn pin_hot(&mut self, keys: &[ExpertKey]) -> Result<(), Error> {
        for key in keys {
            if !self.slow_has(*key) {
                continue;
            }
            if !self.fast.contains_key(key) {
                self.fault_in(*key)?;
            }
            self.lease(*key)?;
        }
        Ok(())
    }

    fn slow_has(&self, key: ExpertKey) -> bool {
        match &self.slow {
            SlowTier::Memory(d) => d.contains(key),
            SlowTier::Disk { index, .. } => index.contains_key(&key),
            SlowTier::Synthetic { keys, .. } => keys.contains(&key),
        }
    }

    fn load_slow(&mut self, key: ExpertKey) -> Result<ExpertParts, Error> {
        match &mut self.slow {
            SlowTier::Memory(d) => d.get(key),
            SlowTier::Disk { file, index } => {
                let span = index
                    .get(&key)
                    .copied()
                    .ok_or(Error::Store("unknown expert"))?;
                read_parts(file, span)
            }
            SlowTier::Synthetic { nbytes, fill, keys } => {
                if !keys.contains(&key) {
                    return Err(Error::Store("unknown expert"));
                }
                Ok(ExpertParts {
                    gate: vec![*fill; *nbytes],
                    up: vec![*fill; *nbytes],
                    down: vec![*fill; *nbytes],
                })
            }
        }
    }

    fn fault_in(&mut self, key: ExpertKey) -> Result<(), Error> {
        if self.fast.len() >= self.slots {
            self.evict_lru()?;
        }
        let parts = self.load_slow(key)?;
        self.bytes_moved = self.bytes_moved.saturating_add(parts.nbytes());
        let _prev = self.fast.insert(key, parts);
        self.recency.push_back(key);
        Ok(())
    }

    fn evict_lru(&mut self) -> Result<(), Error> {
        let victim = self
            .recency
            .iter()
            .copied()
            .find(|k| !self.leased.contains(k));
        match victim {
            Some(v) => {
                let _gone = self.fast.remove(&v);
                self.recency.retain(|k| *k != v);
                self.evicts = self.evicts.saturating_add(1);
                Ok(())
            }
            None => Err(Error::Store("all resident experts are leased")),
        }
    }
}

impl ExpertStore for TieredStore {
    fn acquire(&mut self, key: ExpertKey) -> Result<ExpertParts, Error> {
        if self.fast.contains_key(&key) {
            self.hits = self.hits.saturating_add(1);
            self.recency.retain(|k| *k != key);
            self.recency.push_back(key);
            return self
                .fast
                .get(&key)
                .cloned()
                .ok_or(Error::Store("fast-tier race"));
        }
        self.misses = self.misses.saturating_add(1);
        self.fault_in(key)?;
        self.fast
            .get(&key)
            .cloned()
            .ok_or(Error::Store("fault-in left key absent"))
    }

    fn lease(&mut self, key: ExpertKey) -> Result<(), Error> {
        if !self.fast.contains_key(&key) {
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
            bytes_moved: self.bytes_moved,
        }
    }
}

fn u64_len(v: &[u8]) -> u64 {
    u64::try_from(v.len()).unwrap_or(u64::MAX)
}

fn usize_len(n: u64) -> Result<usize, Error> {
    usize::try_from(n).map_err(|_| Error::Store("expert blob larger than usize"))
}

fn write_all(file: &mut File, bytes: &[u8]) -> Result<(), Error> {
    file.write_all(bytes).map_err(|e| Error::Io(e.to_string()))
}

fn read_parts(file: &mut File, span: FileSpan) -> Result<ExpertParts, Error> {
    let _p = file
        .seek(SeekFrom::Start(span.off))
        .map_err(|e| Error::Io(e.to_string()))?;
    Ok(ExpertParts {
        gate: read_len(file, span.gate)?,
        up: read_len(file, span.up)?,
        down: read_len(file, span.down)?,
    })
}

fn read_len(file: &mut File, n: u64) -> Result<Vec<u8>, Error> {
    let len = usize_len(n)?;
    let mut buf = vec![0u8; len];
    file.read_exact(&mut buf)
        .map_err(|e| Error::Io(e.to_string()))?;
    Ok(buf)
}
