//! [`ExpertStore`] backed by [`gpu_sim`]: H2D, mapped host, managed, or VMM on miss.

use crate::access::{ExpertAccess, ExpertKey, Trace};
use crate::error::Error;
use crate::place::home_gpu;
use crate::planner::{
    observe_chain, plan_window, predicted_keys, window_keys, Markov, Plan, Prefetch,
};
use crate::store::{CachedStore, DirectStore, ExpertParts, ExpertPhase, ExpertStore, StoreMetrics};
use gpu_sim::{
    AllocId, DType, DeviceId, EventId, GraphId, HardwareProfile, KernelKind, MemAdvise, MemcpyOp,
    Place, Score, Sim, StreamId,
};
use std::collections::{BTreeMap, BTreeSet};

/// How [`SimulatedGpuStore`] places a miss page.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GpuFill {
    /// `cudaMallocAsync` + pinned H2D (decode identity).
    #[default]
    Pinned,
    /// `cudaMallocManaged` + ReadMostly + PreferredLocation + prefetch.
    Managed,
    /// `cudaHostAllocMapped` (PCIe kernel, no H2D).
    Mapped,
    /// `va_acquire` + pinned H2D; evict [`gpu_sim::Sim::va_release`].
    Vmm,
}

impl GpuFill {
    /// Pinned when every flag is off; otherwise exactly one of mapped/managed/vmm.
    pub fn from_flags(mapped: bool, managed: bool, vmm: bool) -> Result<Self, Error> {
        match (mapped, managed, vmm) {
            (false, false, false) => Ok(Self::Pinned),
            (true, false, false) => Ok(Self::Mapped),
            (false, true, false) => Ok(Self::Managed),
            (false, false, true) => Ok(Self::Vmm),
            _ => Err(Error::Store("choose one of mapped, managed, vmm")),
        }
    }
}

/// CUDA knobs [`SimulatedGpuStore::new`] leaves off so decode identity stays async.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuStoreCfg {
    /// [`gpu_sim::Sim::host_func`] after each acquire GEMM (CPU scheduler roundtrip).
    pub host_func: bool,
    /// Blocking `cudaStreamCreate` for the compute stream (`StreamId(1)`).
    pub blocking_streams: bool,
    /// Host-sync `cudaMalloc` / `cudaMemcpy` / `cudaFree` on pinned/VMM misses.
    pub sync_alloc: bool,
    /// Hold unused `cudaMallocAsync` bytes in the default pool (`u64::MAX` threshold).
    pub mempool: bool,
    /// Physical span for [`GpuFill::Vmm`]. `0` maps the whole expert (`va_acquire`).
    pub vmm_page: u64,
    /// Pageable `cudaMemcpyAsync` (`memcpy_host_to_device`) instead of pinned DMA.
    ///
    /// Host-synchronous and slower (`pageable_permille`). Decode identity stays
    /// on pinned H2D.
    pub pageable: bool,
    /// [`gpu_sim::MemAdvise::SetAccessedBy`] on every GPU at a managed fill.
    ///
    /// Expert GEMMs are reads-only, so a dest acquire does not migrate. Migrate
    /// retargets compute (no dest prefetch, no `drop_managed_copy`). Pin skips
    /// the replica prefetch. No-op unless [`GpuFill::Managed`]. Decode identity
    /// stays off.
    pub accessed_by: bool,
    /// CUDA legacy null stream: copy (`StreamId(0)`) serializes with compute.
    ///
    /// Off by default (`cudaStreamNonBlocking` compute). Decode identity stays
    /// overlapping.
    pub legacy_null: bool,
    /// `cudaStreamCreateWithPriority` on the compute stream (`StreamId(1)`).
    ///
    /// Copy stays NULL at priority 0. Decode identity stays default priority.
    pub stream_priority: bool,
    /// `cudaGraphExecUpdate` a parked exec onto the next miss alloc.
    ///
    /// Evict parks the instantiated GEMM graph instead of `destroy_graph`.
    /// The next capture on that GPU updates it (`graph_update_ns`) instead of
    /// instantiate. Decode identity stays destroy+instantiate.
    pub graph_update: bool,
    /// `cudaGraphClone` the capture before instantiate (graph vs exec).
    ///
    /// Clone, destroy the src, then instantiate the copy. A parked-exec
    /// update skips clone. Decode identity stays instantiate-in-place.
    pub graph_clone: bool,
}

#[derive(Clone, Copy)]
struct GpuPage {
    id: AllocId,
    device: DeviceId,
    /// Copy-stream event the compute stream must wait on, if not yet consumed.
    ready: Option<EventId>,
}

/// Bounded cache whose misses pay pinned H2D, mapped host, managed prefetch, or VMM.
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
    /// Instantiated GEMM execs parked on evict, keyed by the capture GPU.
    idle_execs: BTreeMap<DeviceId, Vec<GraphId>>,
    graph_launches: u64,
    graph_updates: u64,
    graph_clones: u64,
    graph_update: bool,
    graph_clone: bool,
    mode: GpuFill,
    host_func: bool,
    sync_alloc: bool,
    /// [`GpuStoreCfg::vmm_page`]: KV-sized physicals when [`GpuFill::Vmm`].
    vmm_page: u64,
    /// Pageable H2D (`memcpy_host_to_device`) instead of pinned DMA.
    pageable: bool,
    /// [`GpuStoreCfg::accessed_by`]: managed pages stay on the home GPU.
    accessed_by: bool,
}

impl SimulatedGpuStore {
    /// `slots` HBM expert pages of `bytes_per_expert` each (pinned H2D).
    pub fn new(
        inner: DirectStore,
        slots: usize,
        profile: HardwareProfile,
        bytes_per_expert: u64,
    ) -> Result<Self, Error> {
        Self::with_cfg(
            inner,
            slots,
            profile,
            bytes_per_expert,
            GpuFill::Pinned,
            GpuStoreCfg::default(),
        )
    }

    /// Same as [`Self::new`], but miss pages are `cudaMallocManaged`.
    ///
    /// ReadMostly + PreferredLocation on the striped home, then prefetch
    /// (no pinned H2D). Decode identity stays on [`Self::new`].
    pub fn with_managed(
        inner: DirectStore,
        slots: usize,
        profile: HardwareProfile,
        bytes_per_expert: u64,
    ) -> Result<Self, Error> {
        Self::with_cfg(
            inner,
            slots,
            profile,
            bytes_per_expert,
            GpuFill::Managed,
            GpuStoreCfg::default(),
        )
    }

    /// Same as [`Self::new`], but miss pages are `cudaHostAllocMapped`.
    ///
    /// Kernels read over PCIe with no H2D and no HBM. Decode identity stays
    /// on [`Self::new`].
    pub fn with_mapped(
        inner: DirectStore,
        slots: usize,
        profile: HardwareProfile,
        bytes_per_expert: u64,
    ) -> Result<Self, Error> {
        Self::with_cfg(
            inner,
            slots,
            profile,
            bytes_per_expert,
            GpuFill::Mapped,
            GpuStoreCfg::default(),
        )
    }

    /// Same as [`Self::new`], but miss pages are `va_acquire` then pinned H2D.
    ///
    /// Evict [`gpu_sim::Sim::va_release`]s so the VA can be remapped. Decode
    /// identity stays on [`Self::new`].
    pub fn with_vmm(
        inner: DirectStore,
        slots: usize,
        profile: HardwareProfile,
        bytes_per_expert: u64,
    ) -> Result<Self, Error> {
        Self::with_cfg(
            inner,
            slots,
            profile,
            bytes_per_expert,
            GpuFill::Vmm,
            GpuStoreCfg::default(),
        )
    }

    /// Construct with an explicit fill and [`GpuStoreCfg`] (`SimCfg` subset).
    ///
    /// Defaults keep decode identity: async alloc, overlapping copy/compute,
    /// no `host_func`, default pool threshold 0, no AccessedBy, per-thread NULL,
    /// destroy+instantiate graphs.
    pub fn with_cfg(
        inner: DirectStore,
        slots: usize,
        profile: HardwareProfile,
        bytes_per_expert: u64,
        fill: GpuFill,
        cfg: GpuStoreCfg,
    ) -> Result<Self, Error> {
        let bytes = bytes_per_expert.max(1);
        let mut sim = Sim::new(profile);
        if cfg.mempool {
            sim.set_default_pool_release_threshold(u64::MAX)?;
        }
        if cfg.blocking_streams {
            // Compute is `StreamId(1)`. Copy stays NULL (`StreamId(0)`).
            sim.set_created_streams_blocking(2)?;
        }
        if cfg.legacy_null {
            sim.set_legacy_null_stream(true);
        }
        if cfg.stream_priority {
            sim.set_created_streams_priority(2)?;
        }
        let cache_slots = mapped_occupancy(slots, fill, sim.pin_budget(), bytes);
        // Mapped expert pages already charge the pin budget. Pageable H2D
        // does not need a second mlock either.
        let staging = if fill == GpuFill::Mapped || cfg.pageable {
            sim.alloc_host(bytes)?
        } else {
            sim.alloc_host_pinned(bytes)?
        };
        Ok(Self {
            cache: CachedStore::new(inner, cache_slots)?,
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
            idle_execs: BTreeMap::new(),
            graph_launches: 0,
            graph_updates: 0,
            graph_clones: 0,
            graph_update: cfg.graph_update,
            graph_clone: cfg.graph_clone,
            mode: fill,
            host_func: cfg.host_func,
            sync_alloc: cfg.sync_alloc,
            vmm_page: cfg.vmm_page,
            pageable: cfg.pageable,
            accessed_by: cfg.accessed_by,
        })
    }

    /// Whether miss pages use unified memory (`cudaMallocManaged`).
    #[must_use]
    pub fn uses_managed(&self) -> bool {
        matches!(self.mode, GpuFill::Managed)
    }

    /// Whether miss pages use mapped host (`cudaHostAllocMapped`).
    #[must_use]
    pub fn uses_mapped(&self) -> bool {
        matches!(self.mode, GpuFill::Mapped)
    }

    /// Whether miss pages use CUDA VMM (`va_acquire`).
    #[must_use]
    pub fn uses_vmm(&self) -> bool {
        matches!(self.mode, GpuFill::Vmm)
    }

    /// Page-locked staging buffer from construction; does not count toward HBM.
    #[must_use]
    pub fn staging_is_pinned(&self) -> bool {
        self.sim.is_host_pinned(self.staging).unwrap_or(false)
    }

    /// Unused bytes in GPU0's default mempool after a device drain.
    pub fn default_pool_cached(&mut self) -> Result<u64, Error> {
        self.sim.synchronize()?;
        self.sweep_evicts();
        Ok(self.sim.pool_cached(self.sim.default_pool(self.device)?)?)
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

    /// Fault `keys` in (H2D or managed prefetch, no GEMM). Unknown catalog keys are skipped.
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
    ///
    /// Managed + [`GpuStoreCfg::accessed_by`] maps GPU1 without a dest prefetch.
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

    /// Move `key` onto `dst`. Pinned/VMM pages D2D; managed pages prefetch then drop the source copy unless [`GpuStoreCfg::accessed_by`] (retarget GEMM, keep home residency); mapped pages retarget GEMM.
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
        match self.mode {
            GpuFill::Managed => return self.migrate_managed(key, id, src, dst),
            GpuFill::Mapped => return self.migrate_mapped(key, dst),
            GpuFill::Vmm => return self.migrate_vmm(key, id, src, dst),
            GpuFill::Pinned => {}
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

    /// Prefetch `dst` (ReadMostly keeps extras) then drop the source copy.
    ///
    /// [`Self::accessed_by`]: GEMM retarget only — AccessedBy already maps `dst`.
    fn migrate_managed(
        &mut self,
        key: ExpertKey,
        id: AllocId,
        src: DeviceId,
        dst: DeviceId,
    ) -> Result<(), Error> {
        self.sim.synchronize_stream(src, self.compute)?;
        if self.accessed_by {
            let _gone = self.replicas.remove(&key);
            if let Some(page) = self.pages.get_mut(&key) {
                page.device = dst;
                page.ready = None;
            }
            return Ok(());
        }
        if !self.sim.is_resident(id, dst)? {
            let _p = self.sim.prefetch(dst, id, self.copy)?;
            self.sim.synchronize_stream(dst, self.copy)?;
        }
        if self.sim.is_resident(id, src)? {
            self.sim.drop_managed_copy(id, src)?;
        }
        let _gone = self.replicas.remove(&key);
        if let Some(page) = self.pages.get_mut(&key) {
            page.device = dst;
            page.ready = None;
        }
        Ok(())
    }

    /// Mapped host is already kernel-readable on every GPU; only the GEMM device changes.
    fn migrate_mapped(&mut self, key: ExpertKey, dst: DeviceId) -> Result<(), Error> {
        let _gone = self.replicas.remove(&key);
        if let Some(page) = self.pages.get_mut(&key) {
            page.device = dst;
            page.ready = None;
        }
        Ok(())
    }

    /// Map `dst`, D2D, then unmap the source physicals.
    fn migrate_vmm(
        &mut self,
        key: ExpertKey,
        id: AllocId,
        src: DeviceId,
        dst: DeviceId,
    ) -> Result<(), Error> {
        self.sim.synchronize_stream(src, self.compute)?;
        if !self.sim.is_resident(id, dst)? {
            self.sim.va_map(id, dst)?;
            let _c =
                self.sim
                    .memcpy_device_to_device(src, dst, id, self.bytes_per_expert, self.copy)?;
            self.sim.synchronize_stream(src, self.copy)?;
        }
        if self.sim.is_resident(id, src)? {
            self.sim.va_unmap_range(id, src, 0, self.bytes_per_expert)?;
        }
        let _gone = self.replicas.remove(&key);
        if let Some(page) = self.pages.get_mut(&key) {
            page.device = dst;
            page.ready = None;
        }
        Ok(())
    }

    /// GPU that currently holds `key`, if it has been placed.
    #[must_use]
    pub fn device_of(&self, key: ExpertKey) -> Option<DeviceId> {
        self.pages.get(&key).map(|p| p.device)
    }

    /// Live HBM bytes on `device` (does not drain).
    pub fn hbm_used(&self, device: DeviceId) -> Result<u64, Error> {
        Ok(self.sim.hbm_used(device)?)
    }

    /// Whether the GPU page for `key` is resident on `device`.
    #[must_use]
    pub fn page_resident(&self, key: ExpertKey, device: DeviceId) -> bool {
        self.pages
            .get(&key)
            .and_then(|p| self.sim.is_resident(p.id, device).ok())
            .unwrap_or(false)
    }

    /// Whether `device` has [`gpu_sim::MemAdvise::SetAccessedBy`] on `key`.
    #[must_use]
    pub fn page_accessed_by(&self, key: ExpertKey, device: DeviceId) -> bool {
        self.pages
            .get(&key)
            .and_then(|p| self.sim.is_accessed_by(p.id, device).ok())
            .unwrap_or(false)
    }

    /// [`gpu_sim::Sim::stream_priority`] for `(device, stream)`.
    #[must_use]
    pub fn stream_priority(&self, device: DeviceId, stream: StreamId) -> i32 {
        self.sim.stream_priority(device, stream)
    }

    /// How many times a captured GEMM graph was launched.
    #[must_use]
    pub fn graph_launches(&self) -> u64 {
        self.graph_launches
    }

    /// How many times [`gpu_sim::Sim::update_graph`] reused a parked exec.
    #[must_use]
    pub fn graph_updates(&self) -> u64 {
        self.graph_updates
    }

    /// How many times [`gpu_sim::Sim::clone_graph`] copied a capture before instantiate.
    #[must_use]
    pub fn graph_clones(&self) -> u64 {
        self.graph_clones
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
        let d = self.home(key);
        let id = self.fill_page(d)?;
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

    fn fill_page(&mut self, d: DeviceId) -> Result<AllocId, Error> {
        let bytes = self.bytes_per_expert;
        match self.mode {
            GpuFill::Managed => {
                let id = self.sim.alloc_managed(bytes)?;
                self.sim.mem_advise(id, MemAdvise::SetReadMostly, d)?;
                self.sim
                    .mem_advise(id, MemAdvise::SetPreferredLocation, d)?;
                if self.accessed_by {
                    advise_accessed_by(&mut self.sim, id)?;
                }
                let _p = self.sim.prefetch(d, id, self.copy)?;
                if self.sync_alloc {
                    self.sim.synchronize_stream(d, self.copy)?;
                }
                Ok(id)
            }
            GpuFill::Mapped => Ok(self.sim.alloc_host_mapped(bytes)?),
            GpuFill::Vmm => {
                let id = self.vmm_alloc(d)?;
                self.fill_hbm(d, id)?;
                Ok(id)
            }
            GpuFill::Pinned => {
                let id = self.hbm_alloc(d)?;
                self.fill_hbm(d, id)?;
                Ok(id)
            }
        }
    }

    fn vmm_alloc(&mut self, d: DeviceId) -> Result<AllocId, Error> {
        let bytes = self.bytes_per_expert;
        if self.vmm_page > 0 && self.vmm_page < bytes {
            Ok(self.sim.va_acquire_paged(d, bytes, self.vmm_page)?)
        } else {
            Ok(self.sim.va_acquire(d, bytes)?)
        }
    }

    fn hbm_alloc(&mut self, d: DeviceId) -> Result<AllocId, Error> {
        if self.sync_alloc {
            Ok(self.sim.malloc(d, self.bytes_per_expert)?)
        } else {
            // Stream-ordered `cudaMallocAsync`. `malloc` would device-sync.
            Ok(self.sim.alloc(d, self.bytes_per_expert, self.copy)?)
        }
    }

    fn fill_hbm(&mut self, d: DeviceId, id: AllocId) -> Result<(), Error> {
        let bytes = self.bytes_per_expert;
        if self.pageable {
            let _id = self.sim.memcpy_host_to_device(d, id, bytes, self.copy)?;
        } else if self.sync_alloc {
            let _id = self.sim.memcpy_sync(
                d,
                MemcpyOp {
                    src: Place::HostPinned,
                    dst: Place::Device(d),
                    alloc: id,
                    bytes,
                    offset: 0,
                },
                self.copy,
            )?;
        } else {
            let _c = self.sim.memcpy_pinned_to_device(d, id, bytes, self.copy)?;
        }
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
        self.launch_or_gemm(device, id)?;
        if self.host_func {
            let _id = self.sim.host_func(device, self.compute)?;
        }
        Ok(())
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
            let src = self.sim.end_capture()?;
            let g = self.bind_graph(device, src)?;
            let _prev = self.graphs.insert(id, g);
            self.graph_launches = self.graph_launches.saturating_add(1);
            let _n = self.sim.launch_graph(g, self.compute)?;
            return Ok(());
        }
        gemm(&mut self.sim, device, self.compute, id)
    }

    fn bind_graph(&mut self, device: DeviceId, src: GraphId) -> Result<GraphId, Error> {
        if self.graph_update {
            if let Some(exec) = self.idle_execs.get_mut(&device).and_then(Vec::pop) {
                self.sim.update_graph(exec, src)?;
                self.sim.destroy_graph(src)?;
                self.graph_updates = self.graph_updates.saturating_add(1);
                self.sim.upload_graph(exec)?;
                return Ok(exec);
            }
        }
        let exec = if self.graph_clone {
            let cloned = self.sim.clone_graph(src)?;
            self.sim.destroy_graph(src)?;
            self.graph_clones = self.graph_clones.saturating_add(1);
            cloned
        } else {
            src
        };
        self.sim.instantiate_graph(exec)?;
        self.sim.upload_graph(exec)?;
        Ok(exec)
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
            if self.graph_update {
                self.idle_execs.entry(page.device).or_default().push(g);
            } else {
                self.sim.destroy_graph(g)?;
            }
        }
        match self.mode {
            GpuFill::Managed => {
                self.sim.synchronize_stream(page.device, self.compute)?;
                self.sim.synchronize_stream(page.device, self.copy)?;
                if self.replicas.remove(&key) {
                    self.sim.synchronize_stream(self.replica, self.copy)?;
                }
                self.sim.free_sync(page.id)?;
                return Ok(());
            }
            GpuFill::Mapped => {
                self.sim.synchronize_stream(page.device, self.compute)?;
                self.sim.free_host_pinned(page.id)?;
                let _gone = self.replicas.remove(&key);
                return Ok(());
            }
            GpuFill::Vmm => {
                self.sim.synchronize_stream(page.device, self.compute)?;
                self.sim.va_release(page.id)?;
                let _gone = self.replicas.remove(&key);
                return Ok(());
            }
            GpuFill::Pinned if self.sync_alloc => {
                self.sim.synchronize_stream(page.device, self.compute)?;
                self.sim.free_sync(page.id)?;
                let _gone = self.replicas.remove(&key);
                return Ok(());
            }
            GpuFill::Pinned => {}
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
        let (id, src) = {
            let page = self.pages.get(&key).ok_or(Error::Store("missing handle"))?;
            (page.id, page.device)
        };
        if src == self.replica {
            let _ins = self.replicas.insert(key);
            return Ok(());
        }
        match self.mode {
            GpuFill::Managed => {
                if !self.accessed_by {
                    let _p = self.sim.prefetch(self.replica, id, self.copy)?;
                }
                let _ins = self.replicas.insert(key);
                return Ok(());
            }
            GpuFill::Mapped => {
                let _ins = self.replicas.insert(key);
                return Ok(());
            }
            GpuFill::Vmm => {
                if !self.sim.is_resident(id, self.replica)? {
                    self.sim.va_map(id, self.replica)?;
                    let _c = self.sim.memcpy_device_to_device(
                        src,
                        self.replica,
                        id,
                        self.bytes_per_expert,
                        self.copy,
                    )?;
                }
                let _ins = self.replicas.insert(key);
                return Ok(());
            }
            GpuFill::Pinned => {}
        }
        let _c = self.sim.memcpy_device_to_device(
            src,
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

/// Demand-page a trace through [`SimulatedGpuStore`] and drain the virtual clock.
pub struct StoreReplay {
    /// Cache hits/misses/evicts after every routed expert.
    pub metrics: StoreMetrics,
    /// Virtual wall after a device drain. No `$/M tokens`.
    pub score: Score,
    /// [`SimulatedGpuStore::graph_updates`] after the walk.
    pub graph_updates: u64,
    /// [`SimulatedGpuStore::graph_clones`] after the walk.
    pub graph_clones: u64,
}

impl StoreReplay {
    /// Single-line agent / CLI log.
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "store {} {} graph_updates={} graph_clones={}",
            self.metrics.line(),
            self.score.line(),
            self.graph_updates,
            self.graph_clones
        )
    }
}

/// Planner + CUDA knobs for [`store_replay_cfg`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoreReplayCfg {
    /// Cache slots (mapped occupancy may shrink this).
    pub slots: usize,
    /// Payload bytes per expert page.
    pub bytes_per_expert: u64,
    /// Miss fill path.
    pub fill: GpuFill,
    /// `SimCfg` subset on the store.
    pub gpu: GpuStoreCfg,
    /// Online prefetch (no JSONL future leak except [`Self::plan_window`]).
    pub prefetch: Prefetch,
    /// Upcoming-event window for [`plan_window`]. `0` leaves prefetch ungated.
    pub plan_window: usize,
    /// Stay vs Fetch threshold, permille of upcoming unique keys already resident.
    pub plan_threshold: u32,
}

impl StoreReplayCfg {
    /// Demand paging, pinned async H2D, no planner.
    #[must_use]
    pub fn demand(slots: usize, bytes_per_expert: u64, fill: GpuFill) -> Self {
        Self {
            slots,
            bytes_per_expert,
            fill,
            gpu: GpuStoreCfg::default(),
            prefetch: Prefetch::None,
            plan_window: 0,
            plan_threshold: 500,
        }
    }
}

/// Walk `trace.keys()` on a store, then [`SimulatedGpuStore::score`].
pub fn store_replay(
    trace: &Trace,
    profile: HardwareProfile,
    slots: usize,
    bytes_per_expert: u64,
    fill: GpuFill,
    cfg: GpuStoreCfg,
) -> Result<StoreReplay, Error> {
    store_replay_cfg(
        trace,
        profile,
        StoreReplayCfg {
            slots,
            bytes_per_expert,
            fill,
            gpu: cfg,
            prefetch: Prefetch::None,
            plan_window: 0,
            plan_threshold: 500,
        },
    )
}

/// [`store_replay`] plus copy-forward / Markov / `plan_window` prefetch.
pub fn store_replay_cfg(
    trace: &Trace,
    profile: HardwareProfile,
    run: StoreReplayCfg,
) -> Result<StoreReplay, Error> {
    let inner = DirectStore::from_trace(trace);
    let mut store = SimulatedGpuStore::with_cfg(
        inner,
        run.slots,
        profile,
        run.bytes_per_expert,
        run.fill,
        run.gpu,
    )?;
    let catalog: BTreeSet<ExpertKey> = trace.keys().into_iter().collect();
    let mut markov = Markov::new();
    let mut prev: Option<&ExpertAccess> = None;
    let mut prev2: Option<&ExpertAccess> = None;
    for (i, event) in trace.events.iter().enumerate() {
        let ek = event.keys();
        for key in &ek {
            let _p = store.acquire(*key)?;
            store.release(*key);
        }
        if store_should_prefetch(&store, &catalog, &run, trace, i) {
            let predicted = predicted_keys(run.prefetch, &markov, prev, &ek);
            let planned = if run.plan_window > 0 {
                window_keys(trace, i.saturating_add(1), run.plan_window)
            } else {
                Vec::new()
            };
            let fill: Vec<ExpertKey> = predicted.into_iter().chain(planned).collect();
            let _n = store.prefetch(&fill)?;
        }
        observe_chain(&mut markov, prev2, prev, event);
        prev2 = prev;
        prev = Some(event);
    }
    let n = u64::try_from(trace.events.len()).unwrap_or(1);
    Ok(StoreReplay {
        metrics: store.metrics(),
        graph_updates: store.graph_updates(),
        graph_clones: store.graph_clones(),
        score: store.score()?.with_tokens(n),
    })
}

fn store_should_prefetch(
    store: &SimulatedGpuStore,
    catalog: &BTreeSet<ExpertKey>,
    run: &StoreReplayCfg,
    trace: &Trace,
    i: usize,
) -> bool {
    if run.plan_window == 0 {
        return true;
    }
    let resident: BTreeSet<ExpertKey> = catalog
        .iter()
        .copied()
        .filter(|k| store.is_resident(*k))
        .collect();
    !matches!(
        plan_window(
            &resident,
            trace,
            i.saturating_add(1),
            run.plan_window,
            run.plan_threshold,
        ),
        Plan::Stay
    )
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

/// `cudaMemAdviseSetAccessedBy` on every GPU so a remote read does not migrate.
fn advise_accessed_by(sim: &mut Sim, id: AllocId) -> Result<(), Error> {
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    for g in 0..n {
        sim.mem_advise(id, MemAdvise::SetAccessedBy, DeviceId(g))?;
    }
    Ok(())
}

/// `--mapped` occupancy: `min(slots, pin / expert_bytes)`.
///
/// `fit == 0` keeps the requested slots so the first `alloc_host_mapped` is
/// [`gpu_sim::SimError::PinOom`] (same walker rule as `sim_replay --mapped`).
fn mapped_occupancy(slots: usize, fill: GpuFill, pin_bytes: u64, expert_bytes: u64) -> usize {
    if fill != GpuFill::Mapped {
        return slots;
    }
    let bytes = expert_bytes.max(1);
    let fit = usize::try_from(pin_bytes / bytes).unwrap_or(usize::MAX);
    if fit == 0 {
        slots
    } else {
        slots.min(fit)
    }
}
