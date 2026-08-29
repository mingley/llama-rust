//! Paged VMM KV working set: map only the live pages of a reserved VA.

use crate::error::Error;
use gpu_sim::{
    AllocId, DeviceId, HardwareProfile, KernelBuf, KernelKind, MemcpyOp, Place, Score, Sim,
    StreamId,
};
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
    /// Mapped-page capacity (LRU slots).
    pub slots: usize,
    /// How a miss fills the mapped page.
    pub fill: KvFill,
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
            " hits={} misses={} pages={} slots={} fill={}",
            self.hits, self.misses, self.pages, self.slots, self.fill
        );
        s
    }
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
}

impl KvCfg {
    /// Pinned H2D into each miss.
    #[must_use]
    pub fn h2d(page_bytes: u64, slots: usize) -> Self {
        Self {
            page_bytes,
            slots,
            fill: KvFill::H2d,
        }
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

/// Demand-page a reserved KV VA. `accesses` are page indices.
///
/// Reserves `n_pages * page_bytes` of VA (`n_pages = 1 + max(accesses)`), maps
/// at most `slots` pages at a time, fills a miss ([`KvFill`]), and GEMMs with
/// [`gpu_sim::Sim::kernel_bufs`]. Peak HBM is the working set, not the VA.
pub fn kv_paged(accesses: &[u32], profile: HardwareProfile, cfg: KvCfg) -> Result<KvReplay, Error> {
    if cfg.page_bytes == 0 {
        return Err(Error::Store("page-bytes must be > 0"));
    }
    if cfg.slots == 0 {
        return Err(Error::Store("kv slots must be > 0"));
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
        });
    }
    let max_page = accesses.iter().copied().max().unwrap_or(0);
    let n_pages = max_page.saturating_add(1);
    let va_bytes = u64::from(n_pages).saturating_mul(cfg.page_bytes);
    let mut sim = Sim::new(profile);
    let va = sim.va_reserve(va_bytes)?;
    let d = DeviceId(0);
    let mut order: Vec<u32> = Vec::new();
    let mut hits = 0u64;
    let mut misses = 0u64;
    for &page in accesses {
        let off = page_offset(page, cfg.page_bytes);
        if recency_touch(&mut order, page) {
            hits = hits.saturating_add(1);
            gemm_page(&mut sim, va, off, cfg.page_bytes)?;
            continue;
        }
        misses = misses.saturating_add(1);
        if order.len() >= cfg.slots {
            let victim = order.remove(0);
            let voff = page_offset(victim, cfg.page_bytes);
            sim.va_unmap_range(va, d, voff, cfg.page_bytes)?;
        }
        sim.va_map_range(va, d, off, cfg.page_bytes)?;
        fill_page(&mut sim, va, off, cfg.page_bytes, cfg.fill)?;
        gemm_page(&mut sim, va, off, cfg.page_bytes)?;
        order.push(page);
    }
    sim.synchronize()?;
    let score = Score::from_sim(&sim);
    sim.va_unmap(va)?;
    sim.va_free(va)?;
    Ok(KvReplay {
        sim_ns: score.wall_ns,
        bytes_moved: score.bytes_moved,
        hbm_peak: score.hbm_peak,
        energy_uj: score.energy_uj,
        hits,
        misses,
        pages: n_pages,
        slots: cfg.slots,
        fill: cfg.fill,
    })
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
    }

    #[test]
    fn memset_fill_skips_pcie() {
        let p = HardwareProfile::example_h100_sxm();
        let accesses = cycling_pages(8, 32);
        let cfg = KvCfg {
            page_bytes: 4096,
            slots: 2,
            fill: KvFill::H2d,
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
        let err = kv_replay(&[0], p, 4096, 0).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }
}
