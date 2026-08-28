//! Parameterized hardware. Timing lives here, not in policy code.

use crate::error::SimError;
use crate::ids::DeviceId;
use crate::ops::DType;

/// Peak rates and capacities for one GPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuProfile {
    /// Device index matching [`DeviceId`].
    pub id: DeviceId,
    /// HBM bytes.
    pub hbm_bytes: u64,
    /// Peak HBM bandwidth, bytes/s.
    pub hbm_bps: u64,
    /// Peak tensor-core FLOP/s for [`DType::Fp16`].
    pub fp16_flops: u64,
    /// Peak FLOP/s for [`DType::Fp8`].
    pub fp8_flops: u64,
    /// Peak FLOP/s for [`DType::Fp4`].
    pub fp4_flops: u64,
    /// Peak FP32 FLOP/s.
    pub fp32_flops: u64,
    /// Concurrent copy engines.
    pub copy_engines: u8,
    /// Kernel launch overhead, nanoseconds.
    pub launch_overhead_ns: u64,
    /// Stream-ordered alloc overhead, nanoseconds.
    pub alloc_overhead_ns: u64,
}

impl GpuProfile {
    /// Peak FLOP/s for `dtype`.
    #[must_use]
    pub fn flops(&self, dtype: DType) -> u64 {
        match dtype {
            DType::Fp16 => self.fp16_flops,
            DType::Fp8 => self.fp8_flops,
            DType::Fp4 => self.fp4_flops,
            DType::Fp32 => self.fp32_flops,
        }
    }
}

/// Kind of interconnect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkKind {
    /// Host ↔ GPU.
    Pcie,
    /// GPU ↔ GPU.
    Nvlink,
}

/// One bidirectional link. Concurrent copies share its bandwidth.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinkProfile {
    /// Endpoint A. `None` is the host.
    pub a: Option<DeviceId>,
    /// Endpoint B. `None` is the host.
    pub b: Option<DeviceId>,
    /// Peak payload bytes/s.
    pub bps: u64,
    /// Fixed per-copy overhead, nanoseconds (doorbell, setup).
    pub fixed_ns: u64,
    /// Bytes added to the payload in the duration formula (efficiency ramp).
    pub ramp_bytes: u64,
    /// What the link is.
    pub kind: LinkKind,
}

impl LinkProfile {
    /// Duration of a copy of `bytes` on an otherwise idle link.
    ///
    /// `T = fixed_ns + (bytes + ramp_bytes) / bps`.
    /// Tiny copies pay `ramp_bytes` as if they were large, so fragmenting a
    /// transfer cannot increase effective bandwidth.
    #[must_use]
    pub fn copy_ns(&self, bytes: u64) -> u64 {
        ns_for_bytes(bytes.saturating_add(self.ramp_bytes), self.bps).saturating_add(self.fixed_ns)
    }

    /// Whether this link connects `src` and `dst`.
    #[must_use]
    pub fn connects(&self, src: Option<DeviceId>, dst: Option<DeviceId>) -> bool {
        (self.a == src && self.b == dst) || (self.a == dst && self.b == src)
    }
}

/// Whole node: GPUs plus the mesh of links.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HardwareProfile {
    /// Human-readable name (`h100-sxm`, `captured-h100-driver-X`, …).
    pub name: String,
    /// GPUs.
    pub gpus: Vec<GpuProfile>,
    /// Interconnects.
    pub links: Vec<LinkProfile>,
}

impl HardwareProfile {
    /// Number of GPUs.
    #[must_use]
    pub fn n_gpus(&self) -> usize {
        self.gpus.len()
    }

    /// GPU profile for `id`.
    pub fn gpu(&self, id: DeviceId) -> Result<&GpuProfile, SimError> {
        self.gpus
            .iter()
            .find(|g| g.id == id)
            .ok_or(SimError::Invalid {
                why: "device not in profile",
            })
    }

    /// Link connecting two places, if any.
    pub fn link(
        &self,
        src: Option<DeviceId>,
        dst: Option<DeviceId>,
    ) -> Result<&LinkProfile, SimError> {
        self.links
            .iter()
            .find(|l| l.connects(src, dst))
            .ok_or(SimError::NoPeer {
                src: src.unwrap_or(DeviceId(u16::MAX)),
                dst: dst.unwrap_or(DeviceId(u16::MAX)),
            })
    }

    /// Example single H100 SXM. **Not a capture.** Public-spec order of magnitude.
    #[must_use]
    pub fn example_h100_sxm() -> Self {
        one_gpu_example(
            "example-h100-sxm",
            h100_gpu(DeviceId(0)),
            pcie_host(DeviceId(0)),
        )
    }

    /// Example single H200 SXM. **Not a capture.**
    #[must_use]
    pub fn example_h200_sxm() -> Self {
        one_gpu_example(
            "example-h200-sxm",
            h200_gpu(DeviceId(0)),
            pcie_host(DeviceId(0)),
        )
    }

    /// Example 8× H100 NVLink clique + per-GPU PCIe. **Not a capture.**
    #[must_use]
    pub fn example_8xh100_nvlink() -> Self {
        let mut gpus = Vec::new();
        let mut links = Vec::new();
        for i in 0..8u16 {
            let id = DeviceId(i);
            gpus.push(h100_gpu(id));
            links.push(pcie_host(id));
        }
        for i in 0..8u16 {
            for j in (i + 1)..8u16 {
                links.push(nvlink(DeviceId(i), DeviceId(j)));
            }
        }
        Self {
            name: "example-8xh100-nvlink".into(),
            gpus,
            links,
        }
    }

    /// Hypothetical 48 GiB card for constrained-HBM experiments.
    #[must_use]
    pub fn example_cheap_48gb() -> Self {
        let mut g = h100_gpu(DeviceId(0));
        g.hbm_bytes = 48u64.saturating_mul(1 << 30);
        one_gpu_example("example-cheap-48gb", g, pcie_host(DeviceId(0)))
    }

    /// Parse a `key=value` profile. Unknown keys are errors so captures cannot silently drop fields.
    pub fn parse(text: &str) -> Result<Self, SimError> {
        parse_profile(text)
    }
}

/// `bytes / bps` in nanoseconds, saturating.
#[must_use]
pub fn ns_for_bytes(bytes: u64, bps: u64) -> u64 {
    if bps == 0 {
        return u64::MAX;
    }
    let n = u128::from(bytes)
        .saturating_mul(1_000_000_000)
        .checked_div(u128::from(bps))
        .unwrap_or(u128::MAX);
    u64::try_from(n).unwrap_or(u64::MAX)
}

fn one_gpu_example(name: &str, gpu: GpuProfile, pcie: LinkProfile) -> HardwareProfile {
    HardwareProfile {
        name: name.into(),
        gpus: vec![gpu],
        links: vec![pcie],
    }
}

fn h100_gpu(id: DeviceId) -> GpuProfile {
    GpuProfile {
        id,
        hbm_bytes: 80u64.saturating_mul(1 << 30),
        hbm_bps: 3_350u64.saturating_mul(1_000_000_000),
        fp16_flops: 989u64.saturating_mul(1_000_000_000_000),
        fp8_flops: 1_979u64.saturating_mul(1_000_000_000_000),
        fp4_flops: 3_958u64.saturating_mul(1_000_000_000_000),
        fp32_flops: 67u64.saturating_mul(1_000_000_000_000),
        copy_engines: 2,
        launch_overhead_ns: 3_000,
        alloc_overhead_ns: 2_000,
    }
}

fn h200_gpu(id: DeviceId) -> GpuProfile {
    let mut g = h100_gpu(id);
    g.hbm_bytes = 141u64.saturating_mul(1 << 30);
    g.hbm_bps = 4_800u64.saturating_mul(1_000_000_000);
    g
}

fn pcie_host(gpu: DeviceId) -> LinkProfile {
    LinkProfile {
        a: None,
        b: Some(gpu),
        bps: 32u64.saturating_mul(1_000_000_000),
        fixed_ns: 8_000,
        ramp_bytes: 256 * 1024,
        kind: LinkKind::Pcie,
    }
}

fn nvlink(a: DeviceId, b: DeviceId) -> LinkProfile {
    LinkProfile {
        a: Some(a),
        b: Some(b),
        bps: 450u64.saturating_mul(1_000_000_000),
        fixed_ns: 2_000,
        ramp_bytes: 64 * 1024,
        kind: LinkKind::Nvlink,
    }
}

fn parse_profile(text: &str) -> Result<HardwareProfile, SimError> {
    let mut name = String::from("parsed");
    let mut n_gpus: u16 = 1;
    let mut hbm_bytes = 80u64.saturating_mul(1 << 30);
    let mut hbm_bps = 3_350u64.saturating_mul(1_000_000_000);
    let mut fp16_flops = 989u64.saturating_mul(1_000_000_000_000);
    let mut pcie_bps = 32u64.saturating_mul(1_000_000_000);
    let mut nvlink_bps = 450u64.saturating_mul(1_000_000_000);
    let mut copy_engines: u8 = 2;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            return Err(SimError::Invalid {
                why: "profile line missing =",
            });
        };
        let k = k.trim();
        let v = v.trim();
        match k {
            "name" => name = v.to_string(),
            "gpus" => n_gpus = parse_u16(v)?,
            "hbm_bytes" => hbm_bytes = parse_u64(v)?,
            "hbm_bps" => hbm_bps = parse_u64(v)?,
            "fp16_flops" => fp16_flops = parse_u64(v)?,
            "pcie_bps" => pcie_bps = parse_u64(v)?,
            "nvlink_bps" => nvlink_bps = parse_u64(v)?,
            "copy_engines" => copy_engines = parse_u8(v)?,
            _ => {
                return Err(SimError::Invalid {
                    why: "unknown profile key",
                });
            }
        }
    }
    if n_gpus == 0 {
        return Err(SimError::Invalid {
            why: "gpus must be > 0",
        });
    }
    let mut gpus = Vec::new();
    let mut links = Vec::new();
    for i in 0..n_gpus {
        let id = DeviceId(i);
        let mut g = h100_gpu(id);
        g.hbm_bytes = hbm_bytes;
        g.hbm_bps = hbm_bps;
        g.fp16_flops = fp16_flops;
        g.copy_engines = copy_engines;
        gpus.push(g);
        links.push(LinkProfile {
            a: None,
            b: Some(id),
            bps: pcie_bps,
            fixed_ns: 8_000,
            ramp_bytes: 256 * 1024,
            kind: LinkKind::Pcie,
        });
    }
    if n_gpus > 1 {
        for i in 0..n_gpus {
            for j in (i + 1)..n_gpus {
                links.push(LinkProfile {
                    a: Some(DeviceId(i)),
                    b: Some(DeviceId(j)),
                    bps: nvlink_bps,
                    fixed_ns: 2_000,
                    ramp_bytes: 64 * 1024,
                    kind: LinkKind::Nvlink,
                });
            }
        }
    }
    Ok(HardwareProfile { name, gpus, links })
}

fn parse_u64(s: &str) -> Result<u64, SimError> {
    s.parse::<u64>()
        .map_err(|_| SimError::Invalid { why: "not a u64" })
}

fn parse_u16(s: &str) -> Result<u16, SimError> {
    s.parse::<u16>()
        .map_err(|_| SimError::Invalid { why: "not a u16" })
}

fn parse_u8(s: &str) -> Result<u8, SimError> {
    s.parse::<u8>()
        .map_err(|_| SimError::Invalid { why: "not a u8" })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_copy_is_not_full_pcie() {
        let pcie = pcie_host(DeviceId(0));
        let one = pcie.copy_ns(1);
        let big = pcie.copy_ns(32 * 1024 * 1024);
        assert!(one * 32_000 > big);
    }

    #[test]
    fn parse_roundtrip_name() {
        let p = HardwareProfile::parse("name=lab\ngpus=2\nhbm_bytes=1024\n").unwrap();
        assert_eq!(p.name, "lab");
        assert_eq!(p.n_gpus(), 2);
    }
}
