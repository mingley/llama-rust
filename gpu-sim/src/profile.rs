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
    /// Concurrent compute kernels (Hyper-Q occupancy). `1` is exclusive.
    ///
    /// Ready kernels on different streams may run together up to this cap, each
    /// at full issue rate (not an SM-partition / green-context model). Default
    /// `1` keeps exclusive compute so stream priority serializes leftover
    /// prefill behind decode. Example profiles stay `1` (not a capture).
    pub compute_slots: u8,
    /// `cudaDevAttrCooperativeLaunch`. Example H100 is true (not a capture).
    ///
    /// [`crate::Sim::cooperative_kernel`] fails `cooperative launch not
    /// supported` when this is false. A cooperative kernel occupies every
    /// [`Self::compute_slots`] so it cannot Hyper-Q overlap leftover work.
    pub cooperative_launch: bool,
    /// Kernel launch overhead, nanoseconds.
    pub launch_overhead_ns: u64,
    /// `cudaGraphLaunch` overhead, nanoseconds. Paid once per graph launch, not
    /// per recorded kernel. Example default matches [`Self::launch_overhead_ns`].
    pub graph_launch_ns: u64,
    /// `cudaGraphInstantiate` overhead, nanoseconds. Host-synchronous; paid
    /// once per graph (explicit [`crate::Sim::instantiate_graph`] or first
    /// [`crate::Sim::launch_graph`]). Example default, not a capture.
    pub graph_instantiate_ns: u64,
    /// `cudaGraphExecUpdate` overhead, nanoseconds. Cheaper than recapture
    /// when topology matches. Example default, not a capture.
    pub graph_update_ns: u64,
    /// `cudaGraphExecKernelNodeSetParams` /
    /// `cudaGraphExecMemcpyNodeSetParams` /
    /// `cudaGraphExecMemsetNodeSetParams` / `cudaGraphNodeSetEnabled`
    /// overhead, nanoseconds. Cheaper than
    /// [`Self::graph_update_ns`] (no second graph / topology match). Example
    /// default, not a capture.
    pub graph_set_params_ns: u64,
    /// `cudaGraphClone` overhead, nanoseconds. Host-synchronous. Example
    /// default, not a capture.
    pub graph_clone_ns: u64,
    /// `cudaGraphUpload` overhead, nanoseconds. Host-synchronous; paid once
    /// per exec (explicit [`crate::Sim::upload_graph`] or first
    /// [`crate::Sim::launch_graph`] after instantiate). Example default, not a capture.
    pub graph_upload_ns: u64,
    /// Stream-ordered alloc overhead, nanoseconds.
    pub alloc_overhead_ns: u64,
    /// Reuse of cached pool bytes (`cudaMallocFromPoolAsync` hit), nanoseconds.
    ///
    /// Example default is cheaper than [`Self::alloc_overhead_ns`]. Not a capture.
    pub pool_reuse_ns: u64,
    /// `cudaLaunchHostFunc` duration, nanoseconds. Does not occupy the GPU.
    ///
    /// Example default, not a capture.
    pub host_func_ns: u64,
    /// Board TDP, milliwatts. Energy estimate is `tdp_mw * wall_ns / 1e6` µJ.
    pub tdp_mw: u64,
    /// Achieved GEMM / peak, ‰ (`1000` = full roofline). Duration scales `1000 / util`.
    pub gemm_util_permille: u16,
    /// Grouped-MoE duration vs dense roofline, ‰ (`1000` = no extra). Not a capture.
    pub grouped_moe_permille: u16,
    /// When a PDL primary ([`crate::ops::ProgrammaticLaunch::trigger`]) signals
    /// `cudaTriggerProgrammaticLaunchCompletion`, as ‰ of kernel duration
    /// after start (`0` = at start, `1000` = at completion). Example default
    /// `250`, not a capture. Overlap still needs [`Self::compute_slots`] `>= 2`.
    pub pdl_trigger_permille: u16,
    /// Device L2 size, bytes. Caps [`crate::Sim::set_persisting_l2_cache_size`].
    ///
    /// Example H100 is 50 MiB, not a capture. `0` refuses a non-zero persist
    /// limit.
    pub l2_bytes: u64,
    /// HBM bytes avoided on a persisting-L2 hit, ‰ (`1000` = hit is free).
    ///
    /// Applied to the fraction of kernel traffic that hits a filled
    /// [`crate::ops::AccessPolicyWindow`]. Example default `750`, not a capture.
    /// No window, or persist limit `0`, keeps full HBM billing.
    pub l2_persist_hit_permille: u16,
    /// `cudaDevAttrMemSyncDomainCount`. Hopper example is 4; `1` is pre-Hopper
    /// (both logical domains map to physical 0). Not a capture.
    pub mem_sync_domain_count: u8,
    /// Extra duration when a kernel's implicit completion fence waits on
    /// in-flight writes from another same-physical-domain kernel, ‰ of that
    /// peer's leftover (`1000` = wait for the peer). Default `0` keeps decode
    /// identity and existing Hyper-Q overlap tests. Not a capture.
    pub same_domain_fence_permille: u16,
    /// Hardware `cudaDevAttrMaxBlocksPerCluster`. Example H100 is 8. `1` refuses
    /// a cluster larger than one block. Not a capture.
    pub max_blocks_per_cluster: u8,
    /// Portable cluster size (`sm_90` is 8). A larger launch needs
    /// [`crate::Sim::set_non_portable_cluster_size_allowed`]. Not a capture.
    pub portable_cluster_size: u8,
    /// Host-side wait tax for [`crate::ops::SynchronizationPolicy::Spin`] on
    /// `cudaStreamSynchronize` / `cudaEventSynchronize`, nanoseconds.
    ///
    /// Default `0` keeps decode identity and existing stream-sync tests.
    /// Not a capture. Auto policy is always 0 regardless of this field.
    pub host_sync_spin_ns: u64,
    /// Host-side wait tax for [`crate::ops::SynchronizationPolicy::Yield`].
    /// Default `0`. Not a capture.
    pub host_sync_yield_ns: u64,
    /// Host-side wait tax for [`crate::ops::SynchronizationPolicy::BlockingSync`].
    /// Default `0`. Not a capture.
    pub host_sync_blocking_ns: u64,
    /// Achieved shared-memory throughput in FourByte bank mode, ‰
    /// (`1000` = identity duration). [`crate::ops::SharedMemoryMode::Default`]
    /// uses the device config; unset ignores this. Not a capture.
    pub shared_mem_four_byte_permille: u16,
    /// Achieved shared-memory throughput in EightByte bank mode, ‰
    /// (`1000` = identity duration). [`crate::ops::SharedMemoryMode::Default`]
    /// uses the device config; unset ignores this. Not a capture.
    pub shared_mem_eight_byte_permille: u16,
    /// Portable dynamic shared (`cudaDevAttrMaxSharedMemoryPerBlock`). Example
    /// H100 is 48 KiB. A larger launch needs
    /// [`crate::Sim::set_max_dynamic_shared_memory`] or
    /// [`crate::ops::PortableSharedMode::AllowNonPortable`]. Not a capture.
    pub max_shared_mem_per_block: u32,
    /// Opt-in dynamic shared (`cudaDevAttrMaxSharedMemoryPerBlockOptin`).
    /// Example H100 keeps this equal to [`Self::max_shared_mem_per_block`] so
    /// decode identity stays portable. Tests open it (Hopper 227 KiB).
    /// Not a capture.
    pub max_shared_mem_per_block_optin: u32,
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
    /// Pageable H2D/D2H achieved bandwidth vs pinned, ‰ (`500` = 2× duration).
    ///
    /// Example knob, not a capture. GPU↔GPU copies ignore this.
    pub pageable_permille: u16,
    /// Copy size is rounded up to this many bytes before `ramp_bytes` (0 or 1 = off).
    ///
    /// Example knob so a 1-byte DMA cannot beat a cache-line copy. Not a capture.
    pub align_bytes: u64,
}

impl LinkProfile {
    /// Duration of a copy of `bytes` on an otherwise idle link.
    ///
    /// `T = fixed_ns + (align_up(bytes) + ramp_bytes) / bps`.
    /// Tiny copies pay `ramp_bytes` as if they were large, so fragmenting a
    /// transfer cannot increase effective bandwidth. Sizes below `align_bytes`
    /// bill as one aligned beat. This is the pinned / GPU-direct rate.
    #[must_use]
    pub fn copy_ns(&self, bytes: u64) -> u64 {
        ns_for_bytes(
            align_up(bytes, self.align_bytes).saturating_add(self.ramp_bytes),
            self.bps,
        )
        .saturating_add(self.fixed_ns)
    }

    /// Pageable host copy: [`Self::copy_ns`] scaled by `1000 / pageable_permille`.
    #[must_use]
    pub fn pageable_copy_ns(&self, bytes: u64) -> u64 {
        scale_ns_permille(self.copy_ns(bytes), self.pageable_permille)
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
    /// Host page-lock budget (`mlock` / `cudaHostRegister`), bytes.
    ///
    /// Example default is `u64::MAX` (unlimited). Not a capture.
    pub host_pin_bytes: u64,
    /// `cuMemGetAllocationGranularity`. `0` or `1` accepts any size (decode
    /// identity; 4096-byte expert pages stay legal). A 2 MiB profile rejects
    /// unaligned `va_reserve` / `va_map_range`. Not a capture.
    pub va_granularity_bytes: u64,
    /// `cuMulticastGetGranularity`. `0` or `1` accepts any size (decode
    /// identity; 4096-byte expert pages stay legal). A 2 MiB profile rejects
    /// unaligned `cuMulticastCreate`. Not a capture. Example H100 is `1`.
    pub multicast_granularity_bytes: u64,
    /// Example list-price rent, microdollars per hour (`$2.00` is `2_000_000`).
    ///
    /// `0` omits [`crate::Score::usd_micros_per_m_tokens`]. Not a capture.
    /// Parse `rent_usd_micros_per_hour`. Example profiles stay `0`.
    pub rent_usd_micros_per_hour: u64,
}

impl HardwareProfile {
    /// Number of GPUs.
    #[must_use]
    pub fn n_gpus(&self) -> usize {
        self.gpus.len()
    }

    /// Cap every GPU's HBM to `bytes` (restricted-HBM experiments).
    #[must_use]
    pub fn restrict_hbm(mut self, bytes: u64) -> Self {
        for g in &mut self.gpus {
            g.hbm_bytes = bytes;
        }
        self
    }

    /// Cap page-locked host bytes (`cudaMallocHost` / `cudaHostRegister`).
    #[must_use]
    pub fn restrict_pin(mut self, bytes: u64) -> Self {
        self.host_pin_bytes = bytes;
        self
    }

    /// Example list-price rent for `$/M tokens`. `0` omits dollars.
    #[must_use]
    pub fn with_rent_usd_micros_per_hour(mut self, micros: u64) -> Self {
        self.rent_usd_micros_per_hour = micros;
        self
    }

    /// `cuMemGetAllocationGranularity` for VMM reserve/map. `0` or `1` is off.
    #[must_use]
    pub fn with_va_granularity(mut self, bytes: u64) -> Self {
        self.va_granularity_bytes = bytes;
        self
    }

    /// Whether `n` is a legal VMM size or offset for this profile.
    #[must_use]
    pub fn va_aligned(&self, n: u64) -> bool {
        let g = self.va_granularity_bytes;
        g <= 1 || n.is_multiple_of(g)
    }

    /// `cuMulticastGetGranularity` for [`crate::Sim::multicast_create`]. `0` or `1` is off.
    #[must_use]
    pub fn with_multicast_granularity(mut self, bytes: u64) -> Self {
        self.multicast_granularity_bytes = bytes;
        self
    }

    /// Whether `n` is a legal multicast object size for this profile.
    #[must_use]
    pub fn multicast_aligned(&self, n: u64) -> bool {
        let g = self.multicast_granularity_bytes;
        g <= 1 || n.is_multiple_of(g)
    }

    /// Whether any GPU↔GPU link is NVLink (`CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`).
    ///
    /// PCIe P2P and RDMA are not NVLS. A team still needs an NVLink clique
    /// among the bound devices (`cuMulticastAddDevice`).
    #[must_use]
    pub fn has_nvlink(&self) -> bool {
        self.links.iter().any(|l| l.kind == LinkKind::Nvlink)
    }

    /// `cudaDevAttrGPUDirectRDMASupported` for `device`.
    ///
    /// True when this profile has a GPU↔GPU [`LinkKind::Rdma`] incident on
    /// `device`. [`crate::Sim::flush_gpu_direct_rdma_writes`] is a 1 ns barrier;
    /// write-ordering options are not modeled.
    #[must_use]
    pub fn gpu_direct_rdma_supported(&self, device: DeviceId) -> bool {
        self.links
            .iter()
            .any(|l| l.kind == LinkKind::Rdma && (l.a == Some(device) || l.b == Some(device)))
    }

    /// Hyper-Q occupancy on every GPU (`1` is exclusive compute).
    #[must_use]
    pub fn with_compute_slots(mut self, slots: u8) -> Self {
        let n = slots.max(1);
        for g in &mut self.gpus {
            g.compute_slots = n;
        }
        self
    }

    /// `cudaDevAttrCooperativeLaunch` on every GPU.
    #[must_use]
    pub fn with_cooperative_launch(mut self, yes: bool) -> Self {
        for g in &mut self.gpus {
            g.cooperative_launch = yes;
        }
        self
    }

    /// Sum of board TDP (milliwatts). Energy uses this times virtual wall time.
    #[must_use]
    pub fn node_tdp_mw(&self) -> u64 {
        self.gpus
            .iter()
            .fold(0u64, |acc, g| acc.saturating_add(g.tdp_mw))
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

    /// `cudaDevP2PAttrPerformanceRank`. Lower is better.
    ///
    /// Unique GPU↔GPU [`LinkProfile::bps`] in this profile, sorted
    /// descending; this pair's index. Same device or no GPU↔GPU link is 0.
    /// Host links are not ranked. [`Self::link`] still fails for missing
    /// pairs; this query does not.
    #[must_use]
    pub fn p2p_performance_rank(&self, src: DeviceId, dst: DeviceId) -> u64 {
        if src == dst {
            return 0;
        }
        let Ok(link) = self.link(Some(src), Some(dst)) else {
            return 0;
        };
        if link.a.is_none() || link.b.is_none() {
            return 0;
        }
        let mut unique: Vec<u64> = self
            .links
            .iter()
            .filter(|l| l.a.is_some() && l.b.is_some())
            .map(|l| l.bps)
            .collect();
        unique.sort_unstable_by(|a, b| b.cmp(a));
        unique.dedup();
        unique
            .iter()
            .position(|&b| b == link.bps)
            .and_then(|i| u64::try_from(i).ok())
            .unwrap_or(0)
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
            host_pin_bytes: u64::MAX,
            va_granularity_bytes: 1,
            multicast_granularity_bytes: 1,
            rent_usd_micros_per_hour: 0,
        }
    }

    /// Hypothetical 48 GiB card for constrained-HBM experiments.
    #[must_use]
    pub fn example_cheap_48gb() -> Self {
        let mut g = h100_gpu(DeviceId(0));
        g.hbm_bytes = 48u64.saturating_mul(1 << 30);
        g.tdp_mw = 300_000;
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
            host_pin_bytes: u64::MAX,
            va_granularity_bytes: 1,
            multicast_granularity_bytes: 1,
            rent_usd_micros_per_hour: 0,
        }
    }

    /// Two H100s, GPU1 on a slow far-NUMA PCIe root. No GPU↔GPU link. **Not a capture.**
    #[must_use]
    pub fn example_bad_numa() -> Self {
        Self {
            name: "example-bad-numa".into(),
            gpus: vec![h100_gpu(DeviceId(0)), h100_gpu(DeviceId(1))],
            links: vec![pcie_host(DeviceId(0)), pcie_host_slow(DeviceId(1))],
            host_pin_bytes: u64::MAX,
            va_granularity_bytes: 1,
            multicast_granularity_bytes: 1,
            rent_usd_micros_per_hour: 0,
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
            host_pin_bytes: u64::MAX,
            va_granularity_bytes: 1,
            multicast_granularity_bytes: 1,
            rent_usd_micros_per_hour: 0,
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
            host_pin_bytes: u64::MAX,
            va_granularity_bytes: 1,
            multicast_granularity_bytes: 1,
            rent_usd_micros_per_hour: 0,
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
            "name={}\ngpus={}\nhbm_bytes={}\nhbm_bps={}\nfp16_flops={}\npcie_bps={}\ncopy_engines={}\ncompute_slots={}\ncooperative_launch={}\ntdp_mw={}\nlaunch_overhead_ns={}\ngraph_launch_ns={}\ngraph_instantiate_ns={}\ngraph_update_ns={}\ngraph_set_params_ns={}\ngraph_clone_ns={}\ngraph_upload_ns={}\ngemm_util_permille={}\ngrouped_moe_permille={}\npdl_trigger_permille={}\nl2_bytes={}\nl2_persist_hit_permille={}\nmem_sync_domain_count={}\nsame_domain_fence_permille={}\nmax_blocks_per_cluster={}\nportable_cluster_size={}\nhost_sync_spin_ns={}\nhost_sync_yield_ns={}\nhost_sync_blocking_ns={}\nshared_mem_four_byte_permille={}\nshared_mem_eight_byte_permille={}\nmax_shared_mem_per_block={}\nmax_shared_mem_per_block_optin={}\npageable_permille={}\nalign_bytes={}\npool_reuse_ns={}\nhost_func_ns={}\nhost_pin_bytes={}\nva_granularity_bytes={}\nmulticast_granularity_bytes={}\nrent_usd_micros_per_hour={}\n",
            self.name,
            self.gpus.len(),
            g0.hbm_bytes,
            g0.hbm_bps,
            g0.fp16_flops,
            self.host_bps(g0.id),
            g0.copy_engines,
            g0.compute_slots,
            u8::from(g0.cooperative_launch),
            g0.tdp_mw,
            g0.launch_overhead_ns,
            g0.graph_launch_ns,
            g0.graph_instantiate_ns,
            g0.graph_update_ns,
            g0.graph_set_params_ns,
            g0.graph_clone_ns,
            g0.graph_upload_ns,
            g0.gemm_util_permille,
            g0.grouped_moe_permille,
            g0.pdl_trigger_permille,
            g0.l2_bytes,
            g0.l2_persist_hit_permille,
            g0.mem_sync_domain_count,
            g0.same_domain_fence_permille,
            g0.max_blocks_per_cluster,
            g0.portable_cluster_size,
            g0.host_sync_spin_ns,
            g0.host_sync_yield_ns,
            g0.host_sync_blocking_ns,
            g0.shared_mem_four_byte_permille,
            g0.shared_mem_eight_byte_permille,
            g0.max_shared_mem_per_block,
            g0.max_shared_mem_per_block_optin,
            self.host_pageable_permille(g0.id),
            self.host_align_bytes(g0.id),
            g0.pool_reuse_ns,
            g0.host_func_ns,
            self.host_pin_bytes,
            self.va_granularity_bytes,
            self.multicast_granularity_bytes,
            self.rent_usd_micros_per_hour
        )
    }

    fn host_bps(&self, gpu: DeviceId) -> u64 {
        self.links
            .iter()
            .find(|l| l.kind == LinkKind::Pcie && l.connects(None, Some(gpu)))
            .map(|l| l.bps)
            .unwrap_or(0)
    }

    fn host_pageable_permille(&self, gpu: DeviceId) -> u16 {
        self.links
            .iter()
            .find(|l| l.kind == LinkKind::Pcie && l.connects(None, Some(gpu)))
            .map(|l| l.pageable_permille)
            .unwrap_or(500)
    }

    fn host_align_bytes(&self, gpu: DeviceId) -> u64 {
        self.links
            .iter()
            .find(|l| l.kind == LinkKind::Pcie && l.connects(None, Some(gpu)))
            .map(|l| l.align_bytes)
            .unwrap_or(128)
    }

    /// Parse a `key=value` profile. Unknown keys are errors so captures cannot silently drop fields.
    pub fn parse(text: &str) -> Result<Self, SimError> {
        parse_profile(text)
    }
}

/// Round `n` up to a multiple of `align`. `align <= 1` is a no-op.
#[must_use]
pub fn align_up(n: u64, align: u64) -> u64 {
    if align <= 1 {
        return n;
    }
    let rem = n % align;
    if rem == 0 {
        n
    } else {
        n.saturating_add(align.saturating_sub(rem))
    }
}

/// Scale a duration by inverse achieved-permille (`1000` = identity).
#[must_use]
pub fn scale_ns_permille(ns: u64, achieved_permille: u16) -> u64 {
    let p = u128::from(achieved_permille.max(1));
    let n = u128::from(ns)
        .saturating_mul(1000)
        .checked_div(p)
        .unwrap_or(u128::MAX);
    u64::try_from(n).unwrap_or(u64::MAX)
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
        host_pin_bytes: u64::MAX,
        va_granularity_bytes: 1,
        multicast_granularity_bytes: 1,
        rent_usd_micros_per_hour: 0,
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
        compute_slots: 1,
        cooperative_launch: true,
        launch_overhead_ns: 3_000,
        graph_launch_ns: 3_000,
        graph_instantiate_ns: 25_000,
        graph_update_ns: 5_000,
        graph_set_params_ns: 1_000,
        graph_clone_ns: 8_000,
        graph_upload_ns: 6_000,
        alloc_overhead_ns: 2_000,
        pool_reuse_ns: 200,
        host_func_ns: 10_000,
        tdp_mw: 700_000,
        gemm_util_permille: 1000,
        grouped_moe_permille: 1000,
        pdl_trigger_permille: 250,
        l2_bytes: 50u64.saturating_mul(1 << 20),
        l2_persist_hit_permille: 750,
        mem_sync_domain_count: 4,
        same_domain_fence_permille: 0,
        max_blocks_per_cluster: 8,
        portable_cluster_size: 8,
        host_sync_spin_ns: 0,
        host_sync_yield_ns: 0,
        host_sync_blocking_ns: 0,
        shared_mem_four_byte_permille: 1000,
        shared_mem_eight_byte_permille: 1000,
        max_shared_mem_per_block: 49_152,
        max_shared_mem_per_block_optin: 49_152,
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
        pageable_permille: 500,
        align_bytes: link_align(LinkKind::Pcie),
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
        pageable_permille: 1000,
        align_bytes: link_align(LinkKind::Nvlink),
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
        pageable_permille: 1000,
        align_bytes: link_align(LinkKind::PciePeer),
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
        pageable_permille: 1000,
        align_bytes: link_align(LinkKind::Rdma),
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
        pageable_permille: 500,
        align_bytes: link_align(LinkKind::Pcie),
    }
}

fn link_align(kind: LinkKind) -> u64 {
    match kind {
        LinkKind::Nvlink => 16,
        LinkKind::Rdma => 64,
        LinkKind::Pcie | LinkKind::PciePeer => 128,
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
    let mut compute_slots: u8 = 1;
    let mut cooperative_launch = true;
    let mut tdp_mw = 700_000u64;
    let mut launch_overhead_ns = 3_000u64;
    let mut graph_launch_ns = 3_000u64;
    let mut graph_instantiate_ns: Option<u64> = None;
    let mut graph_update_ns: Option<u64> = None;
    let mut graph_set_params_ns: Option<u64> = None;
    let mut graph_clone_ns: Option<u64> = None;
    let mut graph_upload_ns: Option<u64> = None;
    let mut gemm_util_permille: u16 = 1000;
    let mut grouped_moe_permille: u16 = 1000;
    let mut pdl_trigger_permille: u16 = 250;
    let mut l2_bytes: Option<u64> = None;
    let mut l2_persist_hit_permille: Option<u16> = None;
    let mut mem_sync_domain_count: Option<u8> = None;
    let mut same_domain_fence_permille: Option<u16> = None;
    let mut max_blocks_per_cluster: Option<u8> = None;
    let mut portable_cluster_size: Option<u8> = None;
    let mut host_sync_spin_ns: Option<u64> = None;
    let mut host_sync_yield_ns: Option<u64> = None;
    let mut host_sync_blocking_ns: Option<u64> = None;
    let mut shared_mem_four_byte_permille: Option<u16> = None;
    let mut shared_mem_eight_byte_permille: Option<u16> = None;
    let mut max_shared_mem_per_block: Option<u32> = None;
    let mut max_shared_mem_per_block_optin: Option<u32> = None;
    let mut pageable_permille: u16 = 500;
    let mut align_bytes: u64 = 128;
    let mut pool_reuse_ns: Option<u64> = None;
    let mut host_func_ns: Option<u64> = None;
    let mut host_pin_bytes = u64::MAX;
    let mut va_granularity_bytes = 1u64;
    let mut multicast_granularity_bytes = 1u64;
    let mut rent_usd_micros_per_hour = 0u64;
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
            "compute_slots" => {
                compute_slots = parse_u8(v)?;
                if compute_slots == 0 {
                    return Err(SimError::Invalid {
                        why: "compute_slots must be > 0",
                    });
                }
            }
            "cooperative_launch" => {
                let n = parse_u8(v)?;
                if n > 1 {
                    return Err(SimError::Invalid {
                        why: "cooperative_launch must be 0 or 1",
                    });
                }
                cooperative_launch = n == 1;
            }
            "tdp_mw" => tdp_mw = parse_u64(v)?,
            "launch_overhead_ns" => launch_overhead_ns = parse_u64(v)?,
            "graph_launch_ns" => graph_launch_ns = parse_u64(v)?,
            "graph_instantiate_ns" => graph_instantiate_ns = Some(parse_u64(v)?),
            "graph_update_ns" => graph_update_ns = Some(parse_u64(v)?),
            "graph_set_params_ns" => graph_set_params_ns = Some(parse_u64(v)?),
            "graph_clone_ns" => graph_clone_ns = Some(parse_u64(v)?),
            "graph_upload_ns" => graph_upload_ns = Some(parse_u64(v)?),
            "gemm_util_permille" => gemm_util_permille = parse_u16(v)?,
            "grouped_moe_permille" => grouped_moe_permille = parse_u16(v)?,
            "pdl_trigger_permille" => pdl_trigger_permille = parse_u16(v)?,
            "l2_bytes" => l2_bytes = Some(parse_u64(v)?),
            "l2_persist_hit_permille" => l2_persist_hit_permille = Some(parse_u16(v)?),
            "mem_sync_domain_count" => {
                let n = parse_u8(v)?;
                if n == 0 {
                    return Err(SimError::Invalid {
                        why: "mem_sync_domain_count must be > 0",
                    });
                }
                mem_sync_domain_count = Some(n);
            }
            "same_domain_fence_permille" => {
                let n = parse_u16(v)?;
                if n > 1000 {
                    return Err(SimError::Invalid {
                        why: "same_domain_fence_permille must be <= 1000",
                    });
                }
                same_domain_fence_permille = Some(n);
            }
            "max_blocks_per_cluster" => {
                let n = parse_u8(v)?;
                if n == 0 {
                    return Err(SimError::Invalid {
                        why: "max_blocks_per_cluster must be > 0",
                    });
                }
                max_blocks_per_cluster = Some(n);
            }
            "portable_cluster_size" => {
                let n = parse_u8(v)?;
                if n == 0 {
                    return Err(SimError::Invalid {
                        why: "portable_cluster_size must be > 0",
                    });
                }
                portable_cluster_size = Some(n);
            }
            "host_sync_spin_ns" => host_sync_spin_ns = Some(parse_u64(v)?),
            "host_sync_yield_ns" => host_sync_yield_ns = Some(parse_u64(v)?),
            "host_sync_blocking_ns" => host_sync_blocking_ns = Some(parse_u64(v)?),
            "shared_mem_four_byte_permille" => {
                let n = parse_u16(v)?;
                if n == 0 {
                    return Err(SimError::Invalid {
                        why: "shared_mem_four_byte_permille must be > 0",
                    });
                }
                shared_mem_four_byte_permille = Some(n);
            }
            "shared_mem_eight_byte_permille" => {
                let n = parse_u16(v)?;
                if n == 0 {
                    return Err(SimError::Invalid {
                        why: "shared_mem_eight_byte_permille must be > 0",
                    });
                }
                shared_mem_eight_byte_permille = Some(n);
            }
            "max_shared_mem_per_block" => {
                let n = parse_u32(v)?;
                if n == 0 {
                    return Err(SimError::Invalid {
                        why: "max_shared_mem_per_block must be > 0",
                    });
                }
                max_shared_mem_per_block = Some(n);
            }
            "max_shared_mem_per_block_optin" => {
                let n = parse_u32(v)?;
                if n == 0 {
                    return Err(SimError::Invalid {
                        why: "max_shared_mem_per_block_optin must be > 0",
                    });
                }
                max_shared_mem_per_block_optin = Some(n);
            }
            "pageable_permille" => pageable_permille = parse_u16(v)?,
            "align_bytes" => align_bytes = parse_u64(v)?,
            "pool_reuse_ns" => pool_reuse_ns = Some(parse_u64(v)?),
            "host_func_ns" => host_func_ns = Some(parse_u64(v)?),
            "host_pin_bytes" => host_pin_bytes = parse_u64(v)?,
            "va_granularity_bytes" => va_granularity_bytes = parse_u64(v)?,
            "multicast_granularity_bytes" => multicast_granularity_bytes = parse_u64(v)?,
            "rent_usd_micros_per_hour" => rent_usd_micros_per_hour = parse_u64(v)?,
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
        g.compute_slots = compute_slots;
        g.cooperative_launch = cooperative_launch;
        g.tdp_mw = tdp_mw;
        g.launch_overhead_ns = launch_overhead_ns;
        g.graph_launch_ns = graph_launch_ns;
        if let Some(ns) = graph_instantiate_ns {
            g.graph_instantiate_ns = ns;
        }
        if let Some(ns) = graph_update_ns {
            g.graph_update_ns = ns;
        }
        if let Some(ns) = graph_set_params_ns {
            g.graph_set_params_ns = ns;
        }
        if let Some(ns) = graph_clone_ns {
            g.graph_clone_ns = ns;
        }
        if let Some(ns) = graph_upload_ns {
            g.graph_upload_ns = ns;
        }
        g.gemm_util_permille = gemm_util_permille;
        g.grouped_moe_permille = grouped_moe_permille;
        g.pdl_trigger_permille = pdl_trigger_permille;
        if let Some(n) = l2_bytes {
            g.l2_bytes = n;
        }
        if let Some(n) = l2_persist_hit_permille {
            g.l2_persist_hit_permille = n;
        }
        if let Some(n) = mem_sync_domain_count {
            g.mem_sync_domain_count = n;
        }
        if let Some(n) = same_domain_fence_permille {
            g.same_domain_fence_permille = n;
        }
        if let Some(n) = max_blocks_per_cluster {
            g.max_blocks_per_cluster = n;
        }
        if let Some(n) = portable_cluster_size {
            g.portable_cluster_size = n;
        }
        if let Some(n) = host_sync_spin_ns {
            g.host_sync_spin_ns = n;
        }
        if let Some(n) = host_sync_yield_ns {
            g.host_sync_yield_ns = n;
        }
        if let Some(n) = host_sync_blocking_ns {
            g.host_sync_blocking_ns = n;
        }
        if let Some(n) = shared_mem_four_byte_permille {
            g.shared_mem_four_byte_permille = n;
        }
        if let Some(n) = shared_mem_eight_byte_permille {
            g.shared_mem_eight_byte_permille = n;
        }
        if let Some(n) = max_shared_mem_per_block {
            g.max_shared_mem_per_block = n;
        }
        if let Some(n) = max_shared_mem_per_block_optin {
            g.max_shared_mem_per_block_optin = n;
        }
        if g.max_shared_mem_per_block > g.max_shared_mem_per_block_optin {
            if max_shared_mem_per_block.is_some() && max_shared_mem_per_block_optin.is_some() {
                return Err(SimError::Invalid {
                    why: "max_shared_mem_per_block must be <= max_shared_mem_per_block_optin",
                });
            }
            if max_shared_mem_per_block_optin.is_none() {
                g.max_shared_mem_per_block_optin = g.max_shared_mem_per_block;
            } else {
                return Err(SimError::Invalid {
                    why: "max_shared_mem_per_block must be <= max_shared_mem_per_block_optin",
                });
            }
        }
        if g.portable_cluster_size > g.max_blocks_per_cluster {
            if portable_cluster_size.is_some() {
                return Err(SimError::Invalid {
                    why: "portable_cluster_size must be <= max_blocks_per_cluster",
                });
            }
            g.portable_cluster_size = g.max_blocks_per_cluster;
        }
        if let Some(ns) = pool_reuse_ns {
            g.pool_reuse_ns = ns;
        }
        if let Some(ns) = host_func_ns {
            g.host_func_ns = ns;
        }
        gpus.push(g);
        let far = mesh == MeshKind::NumaBad && i == 1;
        links.push(LinkProfile {
            a: None,
            b: Some(id),
            bps: if far { pcie_far_bps } else { pcie_bps },
            fixed_ns: if far { 20_000 } else { 8_000 },
            ramp_bytes: 256 * 1024,
            kind: LinkKind::Pcie,
            pageable_permille,
            align_bytes,
        });
    }
    push_gpu_mesh(&mut links, n_gpus, mesh, nvlink_bps)?;
    Ok(HardwareProfile {
        name,
        gpus,
        links,
        host_pin_bytes,
        va_granularity_bytes,
        multicast_granularity_bytes,
        rent_usd_micros_per_hour,
    })
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
                link_align(LinkKind::Nvlink),
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
                link_align(LinkKind::PciePeer),
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
                link_align(LinkKind::Rdma),
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
    align_bytes: u64,
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
                pageable_permille: 1000,
                align_bytes,
            });
        }
    }
}

fn parse_u64(s: &str) -> Result<u64, SimError> {
    s.parse::<u64>()
        .map_err(|_| SimError::Invalid { why: "not a u64" })
}

fn parse_u32(s: &str) -> Result<u32, SimError> {
    s.parse::<u32>()
        .map_err(|_| SimError::Invalid { why: "not a u32" })
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
        assert_eq!(p.rent_usd_micros_per_hour, 0);
    }

    #[test]
    fn parse_rent_usd_micros_per_hour() {
        let p = HardwareProfile::parse("gpus=1\nrent_usd_micros_per_hour=2500000\n").unwrap();
        assert_eq!(p.rent_usd_micros_per_hour, 2_500_000);
        assert!(p
            .to_profile_text()
            .contains("rent_usd_micros_per_hour=2500000"));
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
            assert!(p.node_tdp_mw() > 0);
        }
    }

    #[test]
    fn restrict_hbm_caps_every_gpu() {
        let p = HardwareProfile::example_8xh100_nvlink().restrict_hbm(1024);
        assert_eq!(p.n_gpus(), 8);
        assert_eq!(p.gpu(DeviceId(7)).unwrap().hbm_bytes, 1024);
    }

    #[test]
    fn parse_tdp_mw_sets_every_gpu() {
        let p = HardwareProfile::parse("gpus=2\ntopology=pcie\ntdp_mw=123000\n").unwrap();
        assert_eq!(p.node_tdp_mw(), 246_000);
        assert_eq!(p.gpu(DeviceId(0)).unwrap().tdp_mw, 123_000);
        assert_eq!(p.gpu(DeviceId(1)).unwrap().tdp_mw, 123_000);
    }

    #[test]
    fn parse_launch_and_graph_overhead() {
        let p = HardwareProfile::parse("gpus=1\nlaunch_overhead_ns=9000\ngraph_launch_ns=1000\n")
            .unwrap();
        let g = p.gpu(DeviceId(0)).unwrap();
        assert_eq!(g.launch_overhead_ns, 9000);
        assert_eq!(g.graph_launch_ns, 1000);
        assert!(p.to_profile_text().contains("graph_launch_ns=1000"));
    }

    #[test]
    fn parse_graph_instantiate_and_update() {
        let p = HardwareProfile::parse(
            "gpus=1\ngraph_instantiate_ns=111\ngraph_update_ns=22\ngraph_set_params_ns=11\ngraph_clone_ns=33\ngraph_upload_ns=44\n",
        )
        .unwrap();
        let g = p.gpu(DeviceId(0)).unwrap();
        assert_eq!(g.graph_instantiate_ns, 111);
        assert_eq!(g.graph_update_ns, 22);
        assert_eq!(g.graph_set_params_ns, 11);
        assert_eq!(g.graph_clone_ns, 33);
        assert_eq!(g.graph_upload_ns, 44);
        let text = p.to_profile_text();
        assert!(text.contains("graph_instantiate_ns=111"));
        assert!(text.contains("graph_update_ns=22"));
        assert!(text.contains("graph_set_params_ns=11"));
        assert!(text.contains("graph_clone_ns=33"));
        assert!(text.contains("graph_upload_ns=44"));
        let open = HardwareProfile::parse("gpus=1\n").unwrap();
        assert_eq!(open.gpu(DeviceId(0)).unwrap().graph_instantiate_ns, 25_000);
        assert_eq!(open.gpu(DeviceId(0)).unwrap().graph_update_ns, 5_000);
        assert_eq!(open.gpu(DeviceId(0)).unwrap().graph_set_params_ns, 1_000);
        assert_eq!(open.gpu(DeviceId(0)).unwrap().graph_clone_ns, 8_000);
        assert_eq!(open.gpu(DeviceId(0)).unwrap().graph_upload_ns, 6_000);
    }

    #[test]
    fn parse_pool_reuse_ns() {
        let p = HardwareProfile::parse("gpus=1\npool_reuse_ns=50\n").unwrap();
        assert_eq!(p.gpu(DeviceId(0)).unwrap().pool_reuse_ns, 50);
        assert!(p.to_profile_text().contains("pool_reuse_ns=50"));
    }

    #[test]
    fn parse_compute_slots() {
        let p = HardwareProfile::parse("gpus=1\ncompute_slots=2\n").unwrap();
        assert_eq!(p.gpu(DeviceId(0)).unwrap().compute_slots, 2);
        assert!(p.to_profile_text().contains("compute_slots=2"));
        let open = HardwareProfile::parse("gpus=1\n").unwrap();
        assert_eq!(open.gpu(DeviceId(0)).unwrap().compute_slots, 1);
        let err = HardwareProfile::parse("gpus=1\ncompute_slots=0\n").unwrap_err();
        assert!(
            format!("{err:?}").contains("compute_slots must be > 0"),
            "{err:?}"
        );
    }

    #[test]
    fn parse_l2_persist() {
        let p =
            HardwareProfile::parse("gpus=1\nl2_bytes=4096\nl2_persist_hit_permille=500\n").unwrap();
        assert_eq!(p.gpu(DeviceId(0)).unwrap().l2_bytes, 4096);
        assert_eq!(p.gpu(DeviceId(0)).unwrap().l2_persist_hit_permille, 500);
        let text = p.to_profile_text();
        assert!(text.contains("l2_bytes=4096"));
        assert!(text.contains("l2_persist_hit_permille=500"));
        let open = HardwareProfile::parse("gpus=1\n").unwrap();
        assert_eq!(
            open.gpu(DeviceId(0)).unwrap().l2_bytes,
            50u64.saturating_mul(1 << 20)
        );
        assert_eq!(open.gpu(DeviceId(0)).unwrap().l2_persist_hit_permille, 750);
    }

    #[test]
    fn parse_mem_sync_domain() {
        let p = HardwareProfile::parse(
            "gpus=1\nmem_sync_domain_count=1\nsame_domain_fence_permille=250\n",
        )
        .unwrap();
        assert_eq!(p.gpu(DeviceId(0)).unwrap().mem_sync_domain_count, 1);
        assert_eq!(p.gpu(DeviceId(0)).unwrap().same_domain_fence_permille, 250);
        let text = p.to_profile_text();
        assert!(text.contains("mem_sync_domain_count=1"));
        assert!(text.contains("same_domain_fence_permille=250"));
        let open = HardwareProfile::parse("gpus=1\n").unwrap();
        assert_eq!(open.gpu(DeviceId(0)).unwrap().mem_sync_domain_count, 4);
        assert_eq!(open.gpu(DeviceId(0)).unwrap().same_domain_fence_permille, 0);
        let err = HardwareProfile::parse("gpus=1\nmem_sync_domain_count=0\n").unwrap_err();
        assert!(
            format!("{err:?}").contains("mem_sync_domain_count must be > 0"),
            "{err:?}"
        );
        let err = HardwareProfile::parse("gpus=1\nsame_domain_fence_permille=1001\n").unwrap_err();
        assert!(
            format!("{err:?}").contains("same_domain_fence_permille must be <= 1000"),
            "{err:?}"
        );
    }

    #[test]
    fn parse_max_blocks_per_cluster() {
        let p = HardwareProfile::parse("gpus=1\nmax_blocks_per_cluster=16\n").unwrap();
        assert_eq!(p.gpu(DeviceId(0)).unwrap().max_blocks_per_cluster, 16);
        assert!(p.to_profile_text().contains("max_blocks_per_cluster=16"));
        let open = HardwareProfile::parse("gpus=1\n").unwrap();
        assert_eq!(open.gpu(DeviceId(0)).unwrap().max_blocks_per_cluster, 8);
        let err = HardwareProfile::parse("gpus=1\nmax_blocks_per_cluster=0\n").unwrap_err();
        assert!(
            format!("{err:?}").contains("max_blocks_per_cluster must be > 0"),
            "{err:?}"
        );
        let p = HardwareProfile::parse("gpus=1\nportable_cluster_size=4\n").unwrap();
        assert_eq!(p.gpu(DeviceId(0)).unwrap().portable_cluster_size, 4);
        assert!(p.to_profile_text().contains("portable_cluster_size=4"));
        let open = HardwareProfile::parse("gpus=1\n").unwrap();
        assert_eq!(open.gpu(DeviceId(0)).unwrap().portable_cluster_size, 8);
        let err = HardwareProfile::parse("gpus=1\nportable_cluster_size=0\n").unwrap_err();
        assert!(
            format!("{err:?}").contains("portable_cluster_size must be > 0"),
            "{err:?}"
        );
        let p = HardwareProfile::parse("gpus=1\nmax_blocks_per_cluster=4\n").unwrap();
        assert_eq!(p.gpu(DeviceId(0)).unwrap().portable_cluster_size, 4);
        let err =
            HardwareProfile::parse("gpus=1\nmax_blocks_per_cluster=4\nportable_cluster_size=8\n")
                .unwrap_err();
        assert!(
            format!("{err:?}").contains("portable_cluster_size must be <= max_blocks_per_cluster"),
            "{err:?}"
        );
    }

    #[test]
    fn parse_host_sync_ns() {
        let p = HardwareProfile::parse(
            "gpus=1\nhost_sync_spin_ns=100\nhost_sync_yield_ns=200\nhost_sync_blocking_ns=10000\n",
        )
        .unwrap();
        let g = p.gpu(DeviceId(0)).unwrap();
        assert_eq!(g.host_sync_spin_ns, 100);
        assert_eq!(g.host_sync_yield_ns, 200);
        assert_eq!(g.host_sync_blocking_ns, 10000);
        let text = p.to_profile_text();
        assert!(text.contains("host_sync_spin_ns=100"));
        assert!(text.contains("host_sync_yield_ns=200"));
        assert!(text.contains("host_sync_blocking_ns=10000"));
        let open = HardwareProfile::parse("gpus=1\n").unwrap();
        let g0 = open.gpu(DeviceId(0)).unwrap();
        assert_eq!(g0.host_sync_spin_ns, 0);
        assert_eq!(g0.host_sync_yield_ns, 0);
        assert_eq!(g0.host_sync_blocking_ns, 0);
    }

    #[test]
    fn parse_shared_mem_permille() {
        let p = HardwareProfile::parse(
            "gpus=1\nshared_mem_four_byte_permille=500\nshared_mem_eight_byte_permille=2000\n",
        )
        .unwrap();
        let g = p.gpu(DeviceId(0)).unwrap();
        assert_eq!(g.shared_mem_four_byte_permille, 500);
        assert_eq!(g.shared_mem_eight_byte_permille, 2000);
        let text = p.to_profile_text();
        assert!(text.contains("shared_mem_four_byte_permille=500"));
        assert!(text.contains("shared_mem_eight_byte_permille=2000"));
        let open = HardwareProfile::parse("gpus=1\n").unwrap();
        let g0 = open.gpu(DeviceId(0)).unwrap();
        assert_eq!(g0.shared_mem_four_byte_permille, 1000);
        assert_eq!(g0.shared_mem_eight_byte_permille, 1000);
        let err = HardwareProfile::parse("gpus=1\nshared_mem_four_byte_permille=0\n").unwrap_err();
        assert!(
            format!("{err:?}").contains("shared_mem_four_byte_permille must be > 0"),
            "{err:?}"
        );
        let err = HardwareProfile::parse("gpus=1\nshared_mem_eight_byte_permille=0\n").unwrap_err();
        assert!(
            format!("{err:?}").contains("shared_mem_eight_byte_permille must be > 0"),
            "{err:?}"
        );
    }

    #[test]
    fn parse_max_shared_mem_per_block() {
        let p = HardwareProfile::parse(
            "gpus=1\nmax_shared_mem_per_block=49152\nmax_shared_mem_per_block_optin=232448\n",
        )
        .unwrap();
        let g = p.gpu(DeviceId(0)).unwrap();
        assert_eq!(g.max_shared_mem_per_block, 49_152);
        assert_eq!(g.max_shared_mem_per_block_optin, 232_448);
        let text = p.to_profile_text();
        assert!(text.contains("max_shared_mem_per_block=49152"));
        assert!(text.contains("max_shared_mem_per_block_optin=232448"));
        let open = HardwareProfile::parse("gpus=1\n").unwrap();
        let g0 = open.gpu(DeviceId(0)).unwrap();
        assert_eq!(g0.max_shared_mem_per_block, 49_152);
        assert_eq!(g0.max_shared_mem_per_block_optin, 49_152);
        let p = HardwareProfile::parse("gpus=1\nmax_shared_mem_per_block=102400\n").unwrap();
        assert_eq!(
            p.gpu(DeviceId(0)).unwrap().max_shared_mem_per_block_optin,
            102_400
        );
        let err = HardwareProfile::parse("gpus=1\nmax_shared_mem_per_block=0\n").unwrap_err();
        assert!(
            format!("{err:?}").contains("max_shared_mem_per_block must be > 0"),
            "{err:?}"
        );
        let err = HardwareProfile::parse(
            "gpus=1\nmax_shared_mem_per_block=232448\nmax_shared_mem_per_block_optin=49152\n",
        )
        .unwrap_err();
        assert!(
            format!("{err:?}")
                .contains("max_shared_mem_per_block must be <= max_shared_mem_per_block_optin"),
            "{err:?}"
        );
    }

    #[test]
    fn parse_cooperative_launch() {
        let p = HardwareProfile::parse("gpus=1\ncooperative_launch=0\n").unwrap();
        assert!(!p.gpu(DeviceId(0)).unwrap().cooperative_launch);
        assert!(p.to_profile_text().contains("cooperative_launch=0"));
        let open = HardwareProfile::parse("gpus=1\n").unwrap();
        assert!(open.gpu(DeviceId(0)).unwrap().cooperative_launch);
        assert!(open.to_profile_text().contains("cooperative_launch=1"));
        let err = HardwareProfile::parse("gpus=1\ncooperative_launch=2\n").unwrap_err();
        assert!(
            format!("{err:?}").contains("cooperative_launch must be 0 or 1"),
            "{err:?}"
        );
    }

    #[test]
    fn parse_host_pin_bytes() {
        let p = HardwareProfile::parse("gpus=1\nhost_pin_bytes=4096\n").unwrap();
        assert_eq!(p.host_pin_bytes, 4096);
        assert!(p.to_profile_text().contains("host_pin_bytes=4096"));
        let open = HardwareProfile::parse("gpus=1\n").unwrap();
        assert_eq!(open.host_pin_bytes, u64::MAX);
    }

    #[test]
    fn parse_va_granularity_bytes() {
        let p = HardwareProfile::parse("gpus=1\nva_granularity_bytes=2097152\n").unwrap();
        assert_eq!(p.va_granularity_bytes, 2u64 << 20);
        assert!(p.to_profile_text().contains("va_granularity_bytes=2097152"));
        let open = HardwareProfile::parse("gpus=1\n").unwrap();
        assert_eq!(open.va_granularity_bytes, 1);
        assert!(open.va_aligned(4096));
        assert!(!p.va_aligned(4096));
        assert!(p.va_aligned(2u64 << 20));
    }

    #[test]
    fn parse_multicast_granularity_bytes() {
        let p = HardwareProfile::parse("gpus=2\nmulticast_granularity_bytes=2097152\n").unwrap();
        assert_eq!(p.multicast_granularity_bytes, 2u64 << 20);
        assert!(p.has_nvlink());
        assert!(p
            .to_profile_text()
            .contains("multicast_granularity_bytes=2097152"));
        let open = HardwareProfile::parse("gpus=1\n").unwrap();
        assert_eq!(open.multicast_granularity_bytes, 1);
        assert!(!open.has_nvlink());
        assert!(open.multicast_aligned(4096));
        assert!(!p.multicast_aligned(4096));
        assert!(p.multicast_aligned(2u64 << 20));
        let pcie = HardwareProfile::example_2xh100_pcie();
        assert!(!pcie.has_nvlink());
        assert!(HardwareProfile::example_8xh100_nvlink().has_nvlink());
    }

    #[test]
    fn parse_host_func_ns() {
        let p = HardwareProfile::parse("gpus=1\nhost_func_ns=42\n").unwrap();
        assert_eq!(p.gpu(DeviceId(0)).unwrap().host_func_ns, 42);
        assert!(p.to_profile_text().contains("host_func_ns=42"));
    }

    #[test]
    fn parse_kernel_curve_keys() {
        let p =
            HardwareProfile::parse("gpus=1\ngemm_util_permille=500\ngrouped_moe_permille=2000\n")
                .unwrap();
        let g = p.gpu(DeviceId(0)).unwrap();
        assert_eq!(g.gemm_util_permille, 500);
        assert_eq!(g.grouped_moe_permille, 2000);
    }

    #[test]
    fn parse_pageable_permille_on_host_pcie() {
        let p = HardwareProfile::parse("gpus=1\npageable_permille=250\n").unwrap();
        let link = p.link(None, Some(DeviceId(0))).unwrap();
        assert_eq!(link.pageable_permille, 250);
        assert!(link.pageable_copy_ns(1 << 20) > link.copy_ns(1 << 20));
    }

    #[test]
    fn aligned_copy_bills_a_full_beat() {
        let link = LinkProfile {
            a: None,
            b: Some(DeviceId(0)),
            bps: 32u64.saturating_mul(1_000_000_000),
            fixed_ns: 0,
            ramp_bytes: 0,
            kind: LinkKind::Pcie,
            pageable_permille: 1000,
            align_bytes: 128,
        };
        assert_eq!(link.copy_ns(1), link.copy_ns(128));
        assert!(link.copy_ns(129) > link.copy_ns(128));
        assert_eq!(align_up(1, 128), 128);
        assert_eq!(align_up(128, 128), 128);
        assert_eq!(align_up(129, 128), 256);
        assert_eq!(align_up(7, 1), 7);
    }

    #[test]
    fn parse_align_bytes_on_host_pcie() {
        let p = HardwareProfile::parse("gpus=1\nalign_bytes=256\n").unwrap();
        let link = p.link(None, Some(DeviceId(0))).unwrap();
        assert_eq!(link.align_bytes, 256);
        assert!(p.to_profile_text().contains("align_bytes=256"));
    }

    #[test]
    fn pageable_copy_is_slower_than_pinned_at_default() {
        let pcie = pcie_host(DeviceId(0));
        assert_eq!(pcie.pageable_permille, 500);
        assert!(pcie.pageable_copy_ns(8 << 20) > pcie.copy_ns(8 << 20));
    }

    #[test]
    fn checked_in_profile_files_parse() {
        use std::io::Read;
        let dir = format!("{}/profiles", env!("CARGO_MANIFEST_DIR"));
        for file in [
            "h100-sxm.profile",
            "h200-sxm.profile",
            "8xh100-nvlink.profile",
            "cheap-48gb.profile",
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
