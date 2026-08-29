//! [`ExpertStore`] backed by [`gpu_sim`]: H2D on miss, GEMM on acquire, D2D replica.

use crate::access::ExpertKey;
use crate::error::Error;
use crate::store::{CachedStore, DirectStore, ExpertParts, ExpertStore, StoreMetrics};
use gpu_sim::{AllocId, DType, DeviceId, EventId, HardwareProfile, KernelKind, Sim, StreamId};
use std::collections::{BTreeMap, BTreeSet};

struct GpuPage {
    id: AllocId,
    /// Copy-stream event the compute stream must wait on, if not yet consumed.
    ready: Option<EventId>,
}

/// Bounded cache whose misses pay a simulated PCIe copy.
pub struct SimulatedGpuStore {
    cache: CachedStore,
    sim: Sim,
    device: DeviceId,
    replica: DeviceId,
    copy: StreamId,
    compute: StreamId,
    next_event: u32,
    pages: BTreeMap<ExpertKey, GpuPage>,
    replicas: BTreeSet<ExpertKey>,
    bytes_per_expert: u64,
}

impl SimulatedGpuStore {
    /// `slots` HBM expert pages of `bytes_per_expert` each.
    pub fn new(
        inner: DirectStore,
        slots: usize,
        profile: HardwareProfile,
        bytes_per_expert: u64,
    ) -> Result<Self, Error> {
        Ok(Self {
            cache: CachedStore::new(inner, slots)?,
            sim: Sim::new(profile),
            device: DeviceId(0),
            replica: DeviceId(1),
            copy: StreamId(0),
            compute: StreamId(1),
            next_event: 1,
            pages: BTreeMap::new(),
            replicas: BTreeSet::new(),
            bytes_per_expert: bytes_per_expert.max(1),
        })
    }

    /// Drain the simulator and return its performance vector.
    pub fn score(&mut self) -> Result<gpu_sim::Score, Error> {
        self.sim.synchronize()?;
        Ok(gpu_sim::Score::from_sim(&self.sim))
    }

    /// Fault `keys` in (H2D, no GEMM). Unknown catalog keys are skipped.
    pub fn prefetch(&mut self, keys: &[ExpertKey]) -> Result<u64, Error> {
        let n = self.cache.prefetch(keys)?;
        for key in keys {
            if self.cache.is_resident(*key) && !self.pages.contains_key(key) {
                self.place(*key)?;
            }
        }
        Ok(n)
    }

    /// Pin against eviction and, on multi-GPU profiles, NVLink-replicate to GPU1.
    pub fn pin_hot(&mut self, keys: &[ExpertKey]) -> Result<(), Error> {
        self.cache.pin_hot(keys)?;
        for key in keys {
            if !self.cache.is_resident(*key) {
                continue;
            }
            if !self.pages.contains_key(key) {
                self.place(*key)?;
            }
            self.replicate(*key)?;
        }
        Ok(())
    }

    /// Whether `key` is in the fast CPU tier.
    #[must_use]
    pub fn is_resident(&self, key: ExpertKey) -> bool {
        self.cache.is_resident(key)
    }

    fn place(&mut self, key: ExpertKey) -> Result<(), Error> {
        if let Some(v) = self.cache.take_victim() {
            self.drop_gpu(v)?;
        }
        if self.pages.contains_key(&key) {
            return Ok(());
        }
        let bytes = self.bytes_per_expert;
        let id = self.sim.alloc(self.device, bytes, self.copy)?;
        let _c = self
            .sim
            .memcpy_host_to_device(self.device, id, bytes, self.copy)?;
        let ev = EventId(self.next_event);
        self.next_event = self.next_event.saturating_add(1);
        let _r = self.sim.record_event(self.device, ev, self.copy)?;
        let _prev = self.pages.insert(
            key,
            GpuPage {
                id,
                ready: Some(ev),
            },
        );
        Ok(())
    }

    fn gemm_resident(&mut self, key: ExpertKey) -> Result<(), Error> {
        let page = self
            .pages
            .get_mut(&key)
            .ok_or(Error::Store("missing handle"))?;
        if let Some(ev) = page.ready.take() {
            let _w = self.sim.wait_event(self.device, ev, self.compute)?;
        }
        let id = page.id;
        gemm(&mut self.sim, self.device, self.compute, id)
    }

    fn drop_gpu(&mut self, key: ExpertKey) -> Result<(), Error> {
        let Some(page) = self.pages.remove(&key) else {
            return Ok(());
        };
        // Copy-engine free must not race a compute-stream lease on the same page.
        let ev = EventId(self.next_event);
        self.next_event = self.next_event.saturating_add(1);
        let _r = self.sim.record_event(self.device, ev, self.compute)?;
        let _w = self.sim.wait_event(self.device, ev, self.copy)?;
        if self.replicas.remove(&key) {
            self.sim.free(self.replica, page.id, self.copy)?;
        }
        self.sim.free(self.device, page.id, self.copy)?;
        Ok(())
    }

    fn replicate(&mut self, key: ExpertKey) -> Result<(), Error> {
        if self.sim.profile().n_gpus() < 2 {
            return Ok(());
        }
        if self.replicas.contains(&key) {
            return Ok(());
        }
        let id = self
            .pages
            .get(&key)
            .ok_or(Error::Store("missing handle"))?
            .id;
        let _c = self.sim.memcpy_device_to_device(
            self.device,
            self.replica,
            id,
            self.bytes_per_expert,
            self.copy,
        )?;
        let _ins = self.replicas.insert(key);
        Ok(())
    }
}

impl ExpertStore for SimulatedGpuStore {
    fn acquire(&mut self, key: ExpertKey) -> Result<ExpertParts, Error> {
        let hit = self.cache.is_resident(key);
        let parts = self.cache.acquire(key)?;
        if !hit && !self.pages.contains_key(&key) {
            self.place(key)?;
        }
        self.gemm_resident(key)?;
        Ok(parts)
    }

    fn lease(&mut self, key: ExpertKey) -> Result<(), Error> {
        self.cache.lease(key)
    }

    fn release(&mut self, key: ExpertKey) {
        self.cache.release(key);
    }

    fn metrics(&self) -> StoreMetrics {
        let mut m = self.cache.metrics();
        m.bytes_moved = self.sim.bytes_moved();
        m
    }
}

fn gemm(sim: &mut Sim, d: DeviceId, s: StreamId, id: AllocId) -> Result<(), Error> {
    let _k = sim.kernel(
        d,
        KernelKind::GroupedMoeGemm {
            experts: 1,
            tokens_per_expert: 1,
            hidden: 64,
            ff: 64,
            dtype: DType::Fp16,
        },
        &[id],
        &[id],
        s,
    )?;
    Ok(())
}
