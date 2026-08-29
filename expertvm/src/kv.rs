//! Paged VMM KV working set: map only the live pages of a reserved VA.

use crate::error::Error;
use gpu_sim::{
    AllocId, DeviceId, HardwareProfile, KernelBuf, KernelKind, MemHandleId, MemcpyOp, Place, Score,
    Sim, StreamId,
};
use std::collections::BTreeMap;
use std::fmt::{self, Write};

/// Simulated paged-KV result: working-set hits plus gpu-sim scores.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvReplay {
    /// Virtual nanoseconds after drain.
    pub sim_ns: u64,
    /// Host→device bytes moved (page fills).
    pub bytes_moved: u64,
    /// Peak HBM: mapped pages, not the reserved VA.
    pub hbm_peak: u64,
    /// Profile TDP × wall, microjoules.
    pub energy_uj: u64,
    /// Pages already in the mapped working set.
    pub hits: u64,
    /// Pages that required map + H2D.
    pub misses: u64,
    /// Pages in the reserved VA (`1 + max(accesses)`).
    pub pages: u32,
    /// Mapped-page capacity (LRU slots) per sequence VA.
    pub slots: usize,
    /// How a miss fills the mapped page.
    pub fill: KvFill,
    /// Sequence VAs that shared interned physicals.
    pub sequences: u32,
}

impl KvReplay {
    /// Single-line agent / CLI log.
    #[must_use]
    pub fn line(&self) -> String {
        let mut s = format!(
            "sim_ns={} bytes_moved={} hbm_peak={} energy_uj={}",
            self.sim_ns, self.bytes_moved, self.hbm_peak, self.energy_uj
        );
        let _w = write!(
            s,
            " hits={} misses={} pages={} slots={} fill={} sequences={}",
            self.hits, self.misses, self.pages, self.slots, self.fill, self.sequences
        );
        s
    }
}

/// One intern / alloc event the Engine replays onto [`crate::SimulatedGpuStore`].
///
/// Distinct from [`kv_paged`] (a standalone VMM walker). Engine paged-KV
/// logs these so serving scores include KV map / memset / attention traffic
/// on the same virtual clock as expert H2D.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvSimOp {
    /// New physical block: `cuMemCreate` + `cuMemMap` + `cudaMemsetAsync`.
    Fault(u32),
    /// Intern hit: kernel read of an already-mapped block.
    Hit(u32),
    /// Copy-on-write dest: map + memset (CPU already copied f32 K/V).
    Cow {
        /// Shared source block (stays mapped).
        src: u32,
        /// Unique dest block.
        dst: u32,
    },
    /// Refcount hit 0: `va_unmap_range` + `cuMemRelease`.
    Drop(u32),
}

/// How [`kv_paged`] fills a newly mapped KV page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvFill {
    /// Pinned H2D (reload a page from host).
    H2d,
    /// `cudaMemsetAsync` of the mapped span (new KV block; no PCIe).
    Memset,
}

impl KvFill {
    /// CLI / agent parse. `h2d` or `memset`.
    pub fn parse(s: &str) -> Result<Self, Error> {
        match s {
            "h2d" => Ok(Self::H2d),
            "memset" => Ok(Self::Memset),
            _ => Err(Error::Trace("fill must be h2d or memset")),
        }
    }
}

impl fmt::Display for KvFill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::H2d => write!(f, "h2d"),
            Self::Memset => write!(f, "memset"),
        }
    }
}

/// Page size, occupancy, and miss fill for [`kv_paged`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvCfg {
    /// Bytes per mapped page.
    pub page_bytes: u64,
    /// Mapped-page LRU capacity.
    pub slots: usize,
    /// Miss fill.
    pub fill: KvFill,
    /// Per-sequence VAs that alias interned physicals (`cuMemCreate` + `cuMemMap`).
    ///
    /// `1` is a single reserved VA. `2+` is the vLLM intern analog: unique pages
    /// charge HBM once; each sequence maps the same handle into its own VA.
    pub sequences: u32,
}

impl KvCfg {
    /// Pinned H2D into each miss.
    #[must_use]
    pub fn h2d(page_bytes: u64, slots: usize) -> Self {
        Self {
            page_bytes,
            slots,
            fill: KvFill::H2d,
            sequences: 1,
        }
    }

    /// Alias interned physicals into `n` sequence VAs (`1` keeps a single VA).
    #[must_use]
    pub fn with_sequences(mut self, n: u32) -> Self {
        self.sequences = n;
        self
    }
}

/// Demand-page a reserved KV VA. `accesses` are page indices.
///
/// Same as [`kv_paged`] with [`KvFill::H2d`].
pub fn kv_replay(
    accesses: &[u32],
    profile: HardwareProfile,
    page_bytes: u64,
    slots: usize,
) -> Result<KvReplay, Error> {
    kv_paged(accesses, profile, KvCfg::h2d(page_bytes, slots))
}

/// Demand-page reserved KV VAs. `accesses` are interned page indices.
///
/// Reserves `n_pages * page_bytes` of VA per sequence (`n_pages = 1 + max(accesses)`).
/// Each unique page is `cuMemCreate`; sequence VAs `cuMemMap` that handle.
/// At most `slots` maps per sequence. Peak HBM is unique physicals, not
/// `sequences * slots`. Fill a first map ([`KvFill`]) and GEMM with
/// [`gpu_sim::Sim::kernel_bufs`].
pub fn kv_paged(accesses: &[u32], profile: HardwareProfile, cfg: KvCfg) -> Result<KvReplay, Error> {
    if cfg.page_bytes == 0 {
        return Err(Error::Store("page-bytes must be > 0"));
    }
    if cfg.slots == 0 {
        return Err(Error::Store("kv slots must be > 0"));
    }
    if cfg.sequences == 0 {
        return Err(Error::Store("kv sequences must be > 0"));
    }
    if accesses.is_empty() {
        return Ok(KvReplay {
            sim_ns: 0,
            bytes_moved: 0,
            hbm_peak: 0,
            energy_uj: 0,
            hits: 0,
            misses: 0,
            pages: 0,
            slots: cfg.slots,
            fill: cfg.fill,
            sequences: cfg.sequences,
        });
    }
    let nseq = usize::try_from(cfg.sequences).map_err(|_| Error::Store("kv sequences"))?;
    let max_page = accesses.iter().copied().max().unwrap_or(0);
    let n_pages = max_page.saturating_add(1);
    let va_bytes = u64::from(n_pages).saturating_mul(cfg.page_bytes);
    let mut sim = Sim::new(profile);
    let d = DeviceId(0);
    let mut vas = Vec::new();
    for _ in 0..nseq {
        vas.push(sim.va_reserve(va_bytes)?);
    }
    let mut rt = KvRt {
        vas,
        handles: BTreeMap::new(),
        orders: vec![Vec::new(); nseq],
        cfg,
        hits: 0,
        misses: 0,
    };
    for &page in accesses {
        let mut missed: Vec<usize> = Vec::new();
        for seq in 0..nseq {
            if kv_lru_hit(&mut rt, seq, page)? {
                rt.hits = rt.hits.saturating_add(1);
                let va = *rt.vas.get(seq).ok_or(Error::Store("kv sequence"))?;
                gemm_page(
                    &mut sim,
                    va,
                    page_offset(page, rt.cfg.page_bytes),
                    rt.cfg.page_bytes,
                )?;
            } else {
                rt.misses = rt.misses.saturating_add(1);
                kv_evict_if_full(&mut sim, &mut rt, seq)?;
                missed.push(seq);
            }
        }
        for seq in missed {
            kv_finish_miss(&mut sim, &mut rt, seq, page)?;
        }
    }
    sim.synchronize()?;
    let score = Score::from_sim(&sim);
    for &va in &rt.vas {
        if sim.vmm_mapped_bytes(va, d)? > 0 {
            sim.va_unmap(va)?;
        }
        sim.va_free(va)?;
    }
    for h in rt.handles.values().copied() {
        if sim.is_handle_live(h)? {
            sim.va_release_handle(h)?;
        }
    }
    Ok(KvReplay {
        sim_ns: score.wall_ns,
        bytes_moved: score.bytes_moved,
        hbm_peak: score.hbm_peak,
        energy_uj: score.energy_uj,
        hits: rt.hits,
        misses: rt.misses,
        pages: n_pages,
        slots: rt.cfg.slots,
        fill: rt.cfg.fill,
        sequences: rt.cfg.sequences,
    })
}

struct KvRt {
    vas: Vec<AllocId>,
    handles: BTreeMap<u32, MemHandleId>,
    orders: Vec<Vec<u32>>,
    cfg: KvCfg,
    hits: u64,
    misses: u64,
}

fn kv_lru_hit(rt: &mut KvRt, seq: usize, page: u32) -> Result<bool, Error> {
    let order = rt.orders.get_mut(seq).ok_or(Error::Store("kv sequence"))?;
    Ok(recency_touch(order, page))
}

fn kv_evict_if_full(sim: &mut Sim, rt: &mut KvRt, seq: usize) -> Result<(), Error> {
    let full = rt.orders.get(seq).is_some_and(|o| o.len() >= rt.cfg.slots);
    if !full {
        return Ok(());
    }
    let va = *rt.vas.get(seq).ok_or(Error::Store("kv sequence"))?;
    let victim = rt
        .orders
        .get_mut(seq)
        .ok_or(Error::Store("kv sequence"))?
        .remove(0);
    kv_drop_map(sim, &mut rt.handles, va, victim, rt.cfg.page_bytes)
}

fn kv_finish_miss(sim: &mut Sim, rt: &mut KvRt, seq: usize, page: u32) -> Result<(), Error> {
    let page_bytes = rt.cfg.page_bytes;
    let fill = rt.cfg.fill;
    let va = *rt.vas.get(seq).ok_or(Error::Store("kv sequence"))?;
    kv_bind_map(sim, &mut rt.handles, va, page, page_bytes, fill)?;
    gemm_page(sim, va, page_offset(page, page_bytes), page_bytes)?;
    rt.orders
        .get_mut(seq)
        .ok_or(Error::Store("kv sequence"))?
        .push(page);
    Ok(())
}

fn kv_drop_map(
    sim: &mut Sim,
    handles: &mut BTreeMap<u32, MemHandleId>,
    va: AllocId,
    page: u32,
    page_bytes: u64,
) -> Result<(), Error> {
    let d = DeviceId(0);
    let off = page_offset(page, page_bytes);
    sim.va_unmap_range(va, d, off, page_bytes)?;
    let Some(h) = handles.get(&page).copied() else {
        return Ok(());
    };
    if sim.handle_maps(h)? == 0 {
        sim.va_release_handle(h)?;
        let _gone = handles.remove(&page);
    }
    Ok(())
}

fn kv_bind_map(
    sim: &mut Sim,
    handles: &mut BTreeMap<u32, MemHandleId>,
    va: AllocId,
    page: u32,
    page_bytes: u64,
    fill: KvFill,
) -> Result<(), Error> {
    let d = DeviceId(0);
    let off = page_offset(page, page_bytes);
    if let Some(h) = handles.get(&page).copied() {
        sim.va_map_handle(va, d, off, h)?;
        return Ok(());
    }
    let h = sim.va_create(d, page_bytes)?;
    let _prev = handles.insert(page, h);
    sim.va_map_handle(va, d, off, h)?;
    fill_page(sim, va, off, page_bytes, fill)
}

/// Cycling page indices `0 .. pages` for `tokens` steps.
#[must_use]
pub fn cycling_pages(pages: u32, tokens: u32) -> Vec<u32> {
    if pages == 0 {
        return Vec::new();
    }
    (0..tokens).map(|t| t % pages).collect()
}

fn page_offset(page: u32, page_bytes: u64) -> u64 {
    u64::from(page).saturating_mul(page_bytes)
}

fn recency_touch(order: &mut Vec<u32>, page: u32) -> bool {
    let Some(i) = order.iter().position(|&p| p == page) else {
        return false;
    };
    let _old = order.remove(i);
    order.push(page);
    true
}

fn fill_page(
    sim: &mut Sim,
    va: AllocId,
    off: u64,
    page_bytes: u64,
    fill: KvFill,
) -> Result<(), Error> {
    match fill {
        KvFill::H2d => h2d_page(sim, va, off, page_bytes),
        KvFill::Memset => memset_page(sim, va, off, page_bytes),
    }
}

fn h2d_page(sim: &mut Sim, va: AllocId, off: u64, page_bytes: u64) -> Result<(), Error> {
    let d = DeviceId(0);
    let s = StreamId(0);
    let _id = sim.memcpy(
        d,
        MemcpyOp {
            src: Place::HostPinned,
            dst: Place::Device(d),
            alloc: va,
            bytes: page_bytes,
            offset: off,
        },
        s,
    )?;
    Ok(())
}

fn memset_page(sim: &mut Sim, va: AllocId, off: u64, page_bytes: u64) -> Result<(), Error> {
    let d = DeviceId(0);
    let s = StreamId(0);
    let _op = sim.memset_buf(d, KernelBuf::span(va, off, page_bytes), s)?;
    Ok(())
}

fn gemm_page(sim: &mut Sim, va: AllocId, off: u64, page_bytes: u64) -> Result<(), Error> {
    let d = DeviceId(0);
    let s = StreamId(0);
    let buf = KernelBuf::span(va, off, page_bytes);
    let _op = sim.kernel_bufs(d, KernelKind::other(8, page_bytes), &[buf], &[buf], s)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tight_kv_working_set_misses_more_than_full_map() {
        let p = HardwareProfile::example_h100_sxm();
        let accesses = cycling_pages(8, 64);
        let tight = kv_replay(&accesses, p.clone(), 4096, 2).expect("tight");
        let fat = kv_replay(&accesses, p, 4096, 8).expect("fat");
        assert_eq!(tight.pages, 8);
        assert_eq!(fat.pages, 8);
        assert_eq!(tight.hbm_peak, 2 * 4096);
        assert_eq!(fat.hbm_peak, 8 * 4096);
        assert!(
            tight.misses > fat.misses,
            "tight={} fat={}",
            tight.misses,
            fat.misses
        );
        assert_eq!(fat.misses, 8);
        assert_eq!(fat.hits, 56);
        assert_eq!(tight.hits, 0);
        assert_eq!(tight.misses, 64);
        assert!(tight.bytes_moved > fat.bytes_moved);
        assert!(tight.line().contains("slots=2"));
        assert!(fat.line().contains("hbm_peak=32768"));
        assert!(tight.line().contains("fill=h2d"));
        assert!(tight.line().contains("sequences=1"));
    }

    #[test]
    fn memset_fill_skips_pcie() {
        let p = HardwareProfile::example_h100_sxm();
        let accesses = cycling_pages(8, 32);
        let cfg = KvCfg {
            page_bytes: 4096,
            slots: 2,
            fill: KvFill::H2d,
            sequences: 1,
        };
        let h2d = kv_paged(&accesses, p.clone(), cfg).expect("h2d");
        let mut zero = cfg;
        zero.fill = KvFill::Memset;
        let mem = kv_paged(&accesses, p, zero).expect("memset");
        assert_eq!(mem.bytes_moved, 0);
        assert_eq!(mem.hbm_peak, h2d.hbm_peak);
        assert_eq!(mem.misses, h2d.misses);
        assert!(
            mem.sim_ns < h2d.sim_ns,
            "memset={} h2d={}",
            mem.sim_ns,
            h2d.sim_ns
        );
        assert!(mem.line().contains("fill=memset"));
    }

    #[test]
    fn kv_rejects_zero_slots_and_page_bytes() {
        let p = HardwareProfile::example_h100_sxm();
        let err = kv_replay(&[0], p.clone(), 0, 1).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
        let err = kv_replay(&[0], p.clone(), 4096, 0).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
        let err = kv_paged(&[0], p, KvCfg::h2d(4096, 1).with_sequences(0)).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    #[test]
    fn alias_sequences_share_physicals() {
        let p = HardwareProfile::example_h100_sxm();
        let accesses = cycling_pages(8, 64);
        let one = kv_paged(&accesses, p.clone(), KvCfg::h2d(4096, 2)).expect("one");
        let two = kv_paged(&accesses, p, KvCfg::h2d(4096, 2).with_sequences(2)).expect("two");
        assert_eq!(one.hbm_peak, 2 * 4096);
        assert_eq!(two.hbm_peak, one.hbm_peak);
        assert_eq!(two.sequences, 2);
        assert_eq!(two.bytes_moved, one.bytes_moved);
        assert!(
            two.misses > one.misses,
            "one={} two={}",
            one.misses,
            two.misses
        );
        assert!(two.line().contains("sequences=2"));
    }
}
