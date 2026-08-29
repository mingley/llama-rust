//! [`ExpertStore`] backed by [`gpu_sim`]: H2D on miss, GEMM on acquire, D2D replica.

use crate::access::ExpertKey;
use crate::error::Error;
use crate::place::home_gpu;
use crate::store::{CachedStore, DirectStore, ExpertParts, ExpertStore, StoreMetrics};
use gpu_sim::{AllocId, DType, DeviceId, EventId, HardwareProfile, KernelKind, Sim, StreamId};
use std::collections::{BTreeMap, BTreeSet};

struct GpuPage {
    id: AllocId,
    device: DeviceId,
    /// Copy-stream event the compute stream must wait on, if not yet consumed.
    ready: Option<EventId>,
}

/// Bounded cache whose misses pay a simulated pinned H2D onto the striped home GPU.
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

    /// Next H2D that starts fails ([`gpu_sim::SimError::TransferFailed`]).
    pub fn fail_next_transfer(&mut self) {
        self.sim.fail_next_memcpy();
    }

    /// Mark the home GPU unavailable (new submits fail).
    pub fn set_gpu_unavailable(&mut self, yes: bool) -> Result<(), Error> {
        self.sim.set_unavailable(self.device, yes)?;
        Ok(())
    }

    /// Injected extra nanoseconds on every memcpy (transfer delay fault).
    pub fn set_transfer_delay_ns(&mut self, ns: u64) {
        self.sim.set_extra_transfer_ns(ns);
    }

    /// Cancel queued copy-stream ops. In-flight copies still complete.
    pub fn cancel_copy_stream(&mut self) -> Result<u32, Error> {
        Ok(self.sim.cancel_stream(self.device, self.copy)?)
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

    /// Async D2D move onto `dst`. Compute on `dst` waits the copy-stream event.
    ///
    /// Source HBM is released after the copy is stream-ordered; destination HBM
    /// is charged by the peer memcpy. Dest compute can overlap other GPUs.
    pub fn migrate(&mut self, key: ExpertKey, dst: DeviceId) -> Result<(), Error> {
        if self.sim.profile().n_gpus() < 2 {
            return Err(Error::Store("no peer"));
        }
        let _gpu = self.sim.profile().gpu(dst)?;
        if !self.pages.contains_key(&key) {
            if !self.cache.is_resident(key) {
                return Err(Error::Store("not resident"));
            }
            self.place(key)?;
        }
        let (id, src) = {
            let page = self.pages.get(&key).ok_or(Error::Store("missing handle"))?;
            (page.id, page.device)
        };
        if src == dst {
            return Ok(());
        }
        let already = self.sim.is_resident(id, dst)?;
        if !already {
            let _c =
                self.sim
                    .memcpy_device_to_device(src, dst, id, self.bytes_per_expert, self.copy)?;
        }
        let ev_copy = EventId(self.next_event);
        self.next_event = self.next_event.saturating_add(1);
        let _r = self.sim.record_event(src, ev_copy, self.copy)?;
        // Copy-engine free must not race a compute-stream lease on src.
        let ev_compute = EventId(self.next_event);
        self.next_event = self.next_event.saturating_add(1);
        let _r2 = self.sim.record_event(src, ev_compute, self.compute)?;
        let _w = self.sim.wait_event(src, ev_compute, self.copy)?;
        self.sim.free(src, id, self.copy)?;
        let _gone = self.replicas.remove(&key);
        if let Some(page) = self.pages.get_mut(&key) {
            page.device = dst;
            page.ready = Some(ev_copy);
        }
        Ok(())
    }

    /// GPU that currently holds `key`, if it has been placed.
    #[must_use]
    pub fn device_of(&self, key: ExpertKey) -> Option<DeviceId> {
        self.pages.get(&key).map(|p| p.device)
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
        let d = self.home(key);
        let id = self.sim.alloc(d, bytes, self.copy)?;
        let _c = self.sim.memcpy_pinned_to_device(d, id, bytes, self.copy)?;
        let ev = EventId(self.next_event);
        self.next_event = self.next_event.saturating_add(1);
        let _r = self.sim.record_event(d, ev, self.copy)?;
        let _prev = self.pages.insert(
            key,
            GpuPage {
                id,
                device: d,
                ready: Some(ev),
            },
        );
        Ok(())
    }

    fn home(&self, key: ExpertKey) -> DeviceId {
        let n = u16::try_from(self.sim.profile().n_gpus())
            .unwrap_or(1)
            .max(1);
        home_gpu(key, n)
    }

    fn gemm_resident(&mut self, key: ExpertKey) -> Result<(), Error> {
        let (id, device, ready) = {
            let page = self
                .pages
                .get_mut(&key)
                .ok_or(Error::Store("missing handle"))?;
            let ready = page.ready.take();
            (page.id, page.device, ready)
        };
        if let Some(ev) = ready {
            let _w = self.sim.wait_event(device, ev, self.compute)?;
        }
        gemm(&mut self.sim, device, self.compute, id)
    }

    fn drop_gpu(&mut self, key: ExpertKey) -> Result<(), Error> {
        let Some(page) = self.pages.remove(&key) else {
            return Ok(());
        };
        // Copy-engine free must not race a compute-stream lease on the same page.
        let ev = EventId(self.next_event);
        self.next_event = self.next_event.saturating_add(1);
        let _r = self.sim.record_event(page.device, ev, self.compute)?;
        let _w = self.sim.wait_event(page.device, ev, self.copy)?;
        if self.replicas.remove(&key) {
            self.sim.free(self.replica, page.id, self.copy)?;
        }
        self.sim.free(page.device, page.id, self.copy)?;
        Ok(())
    }

    fn replicate(&mut self, key: ExpertKey) -> Result<(), Error> {
        if self.sim.profile().n_gpus() < 2 {
            return Ok(());
        }
        if self.replicas.contains(&key) {
            return Ok(());
        }
        let page = self.pages.get(&key).ok_or(Error::Store("missing handle"))?;
        if page.device == self.replica {
            let _ins = self.replicas.insert(key);
            return Ok(());
        }
        let _c = self.sim.memcpy_device_to_device(
            page.device,
            self.replica,
            page.id,
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
