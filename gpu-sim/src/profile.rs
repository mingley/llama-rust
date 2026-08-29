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
    /// GPU ↔ GPU over NVLink / NVSwitch.
    Nvlink,
    /// GPU ↔ GPU over PCIe P2P (no NVLink).
    PciePeer,
    /// GPU ↔ GPU over node-to-node RDMA.
    Rdma,
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

    /// Two H100s with host PCIe plus GPU↔GPU PCIe P2P (no NVLink). **Not a capture.**
    #[must_use]
    pub fn example_2xh100_pcie() -> Self {
        Self {
            name: "example-2xh100-pcie".into(),
            gpus: vec![h100_gpu(DeviceId(0)), h100_gpu(DeviceId(1))],
            links: vec![
                pcie_host(DeviceId(0)),
                pcie_host(DeviceId(1)),
                pcie_peer(DeviceId(0), DeviceId(1)),
            ],
        }
    }

    /// Two H100s, GPU1 on a slow far-NUMA PCIe root. No GPU↔GPU link. **Not a capture.**
    #[must_use]
    pub fn example_bad_numa() -> Self {
        Self {
            name: "example-bad-numa".into(),
            gpus: vec![h100_gpu(DeviceId(0)), h100_gpu(DeviceId(1))],
            links: vec![pcie_host(DeviceId(0)), pcie_host_slow(DeviceId(1))],
        }
    }

    /// Two nodes × one GPU, GPU-direct RDMA between them. **Not a capture.**
    #[must_use]
    pub fn example_2node_rdma() -> Self {
        Self {
            name: "example-2node-rdma".into(),
            gpus: vec![h100_gpu(DeviceId(0)), h100_gpu(DeviceId(1))],
            links: vec![
                pcie_host(DeviceId(0)),
                pcie_host(DeviceId(1)),
                rdma_peer(DeviceId(0), DeviceId(1)),
            ],
        }
    }

    /// Three H100s with NVLink only on the chain 0–1–2 (no 0–2). **Not a capture.**
    #[must_use]
    pub fn example_asymmetric_links() -> Self {
        let mut gpus = Vec::new();
        let mut links = Vec::new();
        for i in 0..3u16 {
            let id = DeviceId(i);
            gpus.push(h100_gpu(id));
            links.push(pcie_host(id));
        }
        links.push(nvlink(DeviceId(0), DeviceId(1)));
        links.push(nvlink(DeviceId(1), DeviceId(2)));
        Self {
            name: "example-asymmetric".into(),
            gpus,
            links,
        }
    }

    /// Short names accepted by [`Self::by_name`] and the CLIs.
    #[must_use]
    pub fn example_names() -> &'static [&'static str] {
        &[
            "h100",
            "h200",
            "8xh100",
            "cheap",
            "2xh100-pcie",
            "bad-numa",
            "2node-rdma",
            "asymmetric",
        ]
    }

    /// Built-in example profile. Unknown names are [`SimError::Invalid`].
    pub fn by_name(name: &str) -> Result<Self, SimError> {
        match name {
            "h100" | "example-h100-sxm" => Ok(Self::example_h100_sxm()),
            "h200" | "example-h200-sxm" => Ok(Self::example_h200_sxm()),
            "8xh100" | "example-8xh100-nvlink" => Ok(Self::example_8xh100_nvlink()),
            "cheap" | "example-cheap-48gb" => Ok(Self::example_cheap_48gb()),
            "2xh100-pcie" | "example-2xh100-pcie" => Ok(Self::example_2xh100_pcie()),
            "bad-numa" | "example-bad-numa" => Ok(Self::example_bad_numa()),
            "2node-rdma" | "example-2node-rdma" => Ok(Self::example_2node_rdma()),
            "asymmetric" | "example-asymmetric" => Ok(Self::example_asymmetric_links()),
            _ => Err(SimError::Invalid {
                why: "unknown profile name",
            }),
        }
    }

    /// Lossy `key=value` dump (GPU0 rates + GPU count). Mesh shape is not round-tripped.
    #[must_use]
    pub fn to_profile_text(&self) -> String {
        let Some(g0) = self.gpus.first() else {
            return String::from("gpus=0\n");
        };
        format!(
            "name={}\ngpus={}\nhbm_bytes={}\nhbm_bps={}\nfp16_flops={}\npcie_bps={}\ncopy_engines={}\n",
            self.name,
            self.gpus.len(),
            g0.hbm_bytes,
            g0.hbm_bps,
            g0.fp16_flops,
            self.host_bps(g0.id),
            g0.copy_engines
        )
    }

    fn host_bps(&self, gpu: DeviceId) -> u64 {
        self.links
            .iter()
            .find(|l| l.kind == LinkKind::Pcie && l.connects(None, Some(gpu)))
            .map(|l| l.bps)
            .unwrap_or(0)
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

fn pcie_peer(a: DeviceId, b: DeviceId) -> LinkProfile {
    LinkProfile {
        a: Some(a),
        b: Some(b),
        bps: 16u64.saturating_mul(1_000_000_000),
        fixed_ns: 12_000,
        ramp_bytes: 256 * 1024,
        kind: LinkKind::PciePeer,
    }
}

fn rdma_peer(a: DeviceId, b: DeviceId) -> LinkProfile {
    LinkProfile {
        a: Some(a),
        b: Some(b),
        bps: 25u64.saturating_mul(1_000_000_000),
        fixed_ns: 25_000,
        ramp_bytes: 512 * 1024,
        kind: LinkKind::Rdma,
    }
}

fn pcie_host_slow(gpu: DeviceId) -> LinkProfile {
    LinkProfile {
        a: None,
        b: Some(gpu),
        bps: 4u64.saturating_mul(1_000_000_000),
        fixed_ns: 20_000,
        ramp_bytes: 256 * 1024,
        kind: LinkKind::Pcie,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MeshKind {
    NvlinkClique,
    PciePeer,
    Rdma,
    Isolated,
    AsymmetricChain,
    NumaBad,
}

fn parse_mesh(s: &str) -> Result<MeshKind, SimError> {
    match s {
        "nvlink" | "clique" => Ok(MeshKind::NvlinkClique),
        "pcie" | "pcie-peer" => Ok(MeshKind::PciePeer),
        "rdma" => Ok(MeshKind::Rdma),
        "none" | "isolated" => Ok(MeshKind::Isolated),
        "asymmetric" | "chain" => Ok(MeshKind::AsymmetricChain),
        "numa-bad" | "bad-numa" => Ok(MeshKind::NumaBad),
        _ => Err(SimError::Invalid {
            why: "unknown topology",
        }),
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
    let mut pcie_far_bps = 4u64.saturating_mul(1_000_000_000);
    let mut copy_engines: u8 = 2;
    let mut mesh = MeshKind::NvlinkClique;
    let mut mesh_set = false;
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
            "pcie_far_bps" => pcie_far_bps = parse_u64(v)?,
            "copy_engines" => copy_engines = parse_u8(v)?,
            "topology" => {
                mesh = parse_mesh(v)?;
                mesh_set = true;
            }
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
    if !mesh_set && n_gpus == 1 {
        mesh = MeshKind::Isolated;
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
        let far = mesh == MeshKind::NumaBad && i == 1;
        links.push(LinkProfile {
            a: None,
            b: Some(id),
            bps: if far { pcie_far_bps } else { pcie_bps },
            fixed_ns: if far { 20_000 } else { 8_000 },
            ramp_bytes: 256 * 1024,
            kind: LinkKind::Pcie,
        });
    }
    push_gpu_mesh(&mut links, n_gpus, mesh, nvlink_bps)?;
    Ok(HardwareProfile { name, gpus, links })
}

fn push_gpu_mesh(
    links: &mut Vec<LinkProfile>,
    n_gpus: u16,
    mesh: MeshKind,
    nvlink_bps: u64,
) -> Result<(), SimError> {
    match mesh {
        MeshKind::Isolated | MeshKind::NumaBad => {
            if mesh == MeshKind::NumaBad && n_gpus != 2 {
                return Err(SimError::Invalid {
                    why: "numa-bad topology needs 2 gpus",
                });
            }
            Ok(())
        }
        MeshKind::NvlinkClique => {
            clique_links(
                links,
                n_gpus,
                nvlink_bps,
                2_000,
                64 * 1024,
                LinkKind::Nvlink,
            );
            Ok(())
        }
        MeshKind::PciePeer => {
            clique_links(
                links,
                n_gpus,
                16u64.saturating_mul(1_000_000_000),
                12_000,
                256 * 1024,
                LinkKind::PciePeer,
            );
            Ok(())
        }
        MeshKind::Rdma => {
            clique_links(
                links,
                n_gpus,
                25u64.saturating_mul(1_000_000_000),
                25_000,
                512 * 1024,
                LinkKind::Rdma,
            );
            Ok(())
        }
        MeshKind::AsymmetricChain => {
            if n_gpus < 3 {
                return Err(SimError::Invalid {
                    why: "asymmetric topology needs >= 3 gpus",
                });
            }
            for i in 0..n_gpus.saturating_sub(1) {
                links.push(nvlink(DeviceId(i), DeviceId(i.saturating_add(1))));
            }
            Ok(())
        }
    }
}

fn clique_links(
    links: &mut Vec<LinkProfile>,
    n_gpus: u16,
    bps: u64,
    fixed_ns: u64,
    ramp_bytes: u64,
    kind: LinkKind,
) {
    for i in 0..n_gpus {
        for j in (i.saturating_add(1))..n_gpus {
            links.push(LinkProfile {
                a: Some(DeviceId(i)),
                b: Some(DeviceId(j)),
                bps,
                fixed_ns,
                ramp_bytes,
                kind,
            });
        }
    }
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
        assert!(p.link(Some(DeviceId(0)), Some(DeviceId(1))).is_ok());
    }

    #[test]
    fn parse_asymmetric_omits_02() {
        let p = HardwareProfile::parse("name=line\ngpus=3\ntopology=asymmetric\n").unwrap();
        assert!(p.link(Some(DeviceId(0)), Some(DeviceId(1))).is_ok());
        assert!(p.link(Some(DeviceId(0)), Some(DeviceId(2))).is_err());
    }

    #[test]
    fn by_name_covers_every_example() {
        for name in HardwareProfile::example_names() {
            let p = HardwareProfile::by_name(name).unwrap();
            assert!(!p.name.is_empty());
            assert!(p.n_gpus() >= 1);
        }
    }

    #[test]
    fn checked_in_profile_files_parse() {
        use std::io::Read;
        let dir = format!("{}/profiles", env!("CARGO_MANIFEST_DIR"));
        for file in [
            "h100-sxm.profile",
            "2xh100-pcie.profile",
            "bad-numa.profile",
            "2node-rdma.profile",
            "asymmetric.profile",
        ] {
            let path = format!("{dir}/{file}");
            let mut f = std::fs::File::open(&path).unwrap();
            let mut buf = String::new();
            let _n = f.read_to_string(&mut buf).unwrap();
            let p = HardwareProfile::parse(&buf).unwrap();
            assert!(p.n_gpus() >= 1, "{file}");
        }
    }
}
