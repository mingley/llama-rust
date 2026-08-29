//! [`ExpertStore`] backed by [`gpu_sim`]: H2D on miss, GEMM on acquire, D2D replica.

use crate::access::ExpertKey;
use crate::error::Error;
use crate::place::home_gpu;
use crate::store::{CachedStore, DirectStore, ExpertParts, ExpertPhase, ExpertStore, StoreMetrics};
use gpu_sim::{
    AllocId, DType, DeviceId, EventId, GraphId, HardwareProfile, KernelKind, Sim, StreamId,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy)]
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
    evicting: BTreeMap<ExpertKey, GpuPage>,
    bytes_per_expert: u64,
    staging: AllocId,
    graphs: BTreeMap<AllocId, GraphId>,
    graph_launches: u64,
}

impl SimulatedGpuStore {
    /// `slots` HBM expert pages of `bytes_per_expert` each.
    pub fn new(
        inner: DirectStore,
        slots: usize,
        profile: HardwareProfile,
        bytes_per_expert: u64,
    ) -> Result<Self, Error> {
        let bytes = bytes_per_expert.max(1);
        let mut sim = Sim::new(profile);
        let staging = sim.alloc_host_pinned(bytes)?;
        Ok(Self {
            cache: CachedStore::new(inner, slots)?,
            sim,
            device: DeviceId(0),
            replica: DeviceId(1),
            copy: StreamId(0),
            compute: StreamId(1),
            next_event: 1,
            pages: BTreeMap::new(),
            replicas: BTreeSet::new(),
            evicting: BTreeMap::new(),
            bytes_per_expert: bytes,
            staging,
            graphs: BTreeMap::new(),
            graph_launches: 0,
        })
    }

    /// Page-locked staging buffer from construction; does not count toward HBM.
    #[must_use]
    pub fn staging_is_pinned(&self) -> bool {
        self.sim.is_host_pinned(self.staging).unwrap_or(false)
    }

    /// Drain the simulator and return its performance vector.
    pub fn score(&mut self) -> Result<gpu_sim::Score, Error> {
        self.sim.synchronize()?;
        self.sweep_evicts();
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
        self.sweep_evicts();
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
        for key in keys {
            if !self.cache.contains_catalog(*key) {
                continue;
            }
            if !self.cache.is_resident(*key) {
                let _n = self.cache.prefetch(&[*key])?;
            }
            if !self.pages.contains_key(key) {
                self.place(*key)?;
            }
            self.wait_copy(*key)?;
            self.cache.lease(*key)?;
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
        if let Some(g) = self.graphs.remove(&id) {
            self.sim.destroy_graph(g)?;
        }
        let already = self.sim.is_resident(id, dst)?;
        if !already {
            let _c =
                self.sim
                    .memcpy_device_to_device(src, dst, id, self.bytes_per_expert, self.copy)?;
        }
        let ev_copy = EventId(self.next_event);
        self.next_event = self.next_event.saturating_add(1);
        self.sim.create_event_disable_timing(ev_copy)?;
        let _r = self.sim.record_event(src, ev_copy, self.copy)?;
        // Copy-engine free must not race a compute-stream lease on src.
        let ev_compute = EventId(self.next_event);
        self.next_event = self.next_event.saturating_add(1);
        self.sim.create_event_disable_timing(ev_compute)?;
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

    /// How many times a captured GEMM graph was launched.
    #[must_use]
    pub fn graph_launches(&self) -> u64 {
        self.graph_launches
    }

    /// Whether `key` is in the fast CPU tier.
    #[must_use]
    pub fn is_resident(&self, key: ExpertKey) -> bool {
        self.cache.is_resident(key)
    }

    /// PLAN state: GPU copies are Transferring until the copy-stream event completes.
    #[must_use]
    pub fn phase(&self, key: ExpertKey) -> ExpertPhase {
        if let Some(page) = self.evicting.get(&key) {
            if self.sim.is_resident(page.id, page.device).unwrap_or(false) {
                return ExpertPhase::Evicting;
            }
            return ExpertPhase::Cold;
        }
        if let Some(page) = self.pages.get(&key) {
            if let Some(ev) = page.ready {
                if !self.sim.event_complete(ev) {
                    return ExpertPhase::Transferring;
                }
            }
        }
        ExpertPhase::cpu(self.cache.is_resident(key), self.cache.is_leased(key))
    }

    /// Drop `key` from HBM. Illegal while leased. Stays [`ExpertPhase::Evicting`]
    /// until the stream-ordered free completes.
    pub fn evict(&mut self, key: ExpertKey) -> Result<(), Error> {
        self.sweep_evicts();
        self.cache.evict(key)?;
        self.drop_gpu(key)
    }

    fn wait_copy(&mut self, key: ExpertKey) -> Result<(), Error> {
        let (device, ready) = {
            let page = self.pages.get(&key).ok_or(Error::Store("missing handle"))?;
            (page.device, page.ready)
        };
        if let Some(ev) = ready {
            if !self.sim.event_complete(ev) {
                let _w = self.sim.wait_event(device, ev, self.compute)?;
                self.sim.synchronize_stream(device, self.compute)?;
            }
            if let Some(page) = self.pages.get_mut(&key) {
                page.ready = None;
            }
        }
        Ok(())
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
        // Stream-ordered `cudaMallocAsync`. `malloc` (`cudaMalloc`) would
        // device-sync this GPU on every miss and serialize with GEMM.
        let id = self.sim.alloc(d, bytes, self.copy)?;
        // Pinned DMA. Pageable `memcpy_host_to_device` would wait this stream.
        let _c = self.sim.memcpy_pinned_to_device(d, id, bytes, self.copy)?;
        let ev = EventId(self.next_event);
        self.next_event = self.next_event.saturating_add(1);
        self.sim.create_event_disable_timing(ev)?;
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
            if !self.sim.event_complete(ev) {
                let _w = self.sim.wait_event(device, ev, self.compute)?;
            }
        }
        self.launch_or_gemm(device, id)
    }

    fn launch_or_gemm(&mut self, device: DeviceId, id: AllocId) -> Result<(), Error> {
        if let Some(g) = self.graphs.get(&id).copied() {
            self.graph_launches = self.graph_launches.saturating_add(1);
            let _n = self.sim.launch_graph(g, self.compute)?;
            return Ok(());
        }
        if !self.sim.stream_is_idle(device, self.compute)? {
            self.sim.synchronize_stream(device, self.compute)?;
        }
        if self.sim.stream_is_idle(device, self.compute)? {
            self.sim.begin_capture(device, self.compute)?;
            gemm(&mut self.sim, device, self.compute, id)?;
            let g = self.sim.end_capture()?;
            let _prev = self.graphs.insert(id, g);
            self.graph_launches = self.graph_launches.saturating_add(1);
            let _n = self.sim.launch_graph(g, self.compute)?;
            return Ok(());
        }
        gemm(&mut self.sim, device, self.compute, id)
    }

    fn drop_gpu(&mut self, key: ExpertKey) -> Result<(), Error> {
        let Some(page) = self.pages.remove(&key) else {
            return Ok(());
        };
        let _prev = self.evicting.insert(key, page);
        self.finish_drop(key, page)
    }

    fn sweep_evicts(&mut self) {
        let done: Vec<ExpertKey> = self
            .evicting
            .iter()
            .filter_map(|(k, p)| {
                if self.sim.is_resident(p.id, p.device).unwrap_or(false) {
                    None
                } else {
                    Some(*k)
                }
            })
            .collect();
        for k in done {
            let _gone = self.evicting.remove(&k);
        }
    }

    fn finish_drop(&mut self, key: ExpertKey, page: GpuPage) -> Result<(), Error> {
        if let Some(g) = self.graphs.remove(&page.id) {
            self.sim.destroy_graph(g)?;
        }
        // Copy-engine free must not race a compute-stream lease on the same page.
        let ev = EventId(self.next_event);
        self.next_event = self.next_event.saturating_add(1);
        self.sim.create_event_disable_timing(ev)?;
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
        self.sweep_evicts();
        let hit = self.cache.is_resident(key);
        let parts = self.cache.acquire(key)?;
        if !hit && !self.pages.contains_key(&key) {
            self.place(key)?;
        }
        self.gemm_resident(key)?;
        Ok(parts)
    }

    fn lease(&mut self, key: ExpertKey) -> Result<(), Error> {
        match self.phase(key) {
            ExpertPhase::Resident | ExpertPhase::Leased => self.cache.lease(key),
            ExpertPhase::Transferring => Err(Error::Store("lease of transferring expert")),
            ExpertPhase::Evicting => Err(Error::Store("lease of evicting expert")),
            ExpertPhase::Cold => Err(Error::Store("lease of non-resident expert")),
        }
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
        &[],
        s,
    )?;
    Ok(())
}
