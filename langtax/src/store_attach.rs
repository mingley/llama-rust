//! Shared Engine ExpertStore attach for `engine` and `serve --engine`.

use crate::decode::Llama;
use crate::engine::Engine;
use expertvm::{
    CachedStore, GpuFill, GpuStoreCfg, HardwareProfile, LiveStore, MemSyncDomain,
    PortableClusterMode, PortableSharedMode, Prefetch, SharedMemoryMode, SimulatedGpuStore,
    SynchronizationPolicy,
};

/// CLI knobs that build a [`LiveStore`] for an [`Engine`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StoreAttach {
    /// `None` keeps blob FFN. `Some(0)` is DirectStore. `Some(n)` is CachedStore.
    pub expert_slots: Option<usize>,
    /// SimulatedGpuStore instead of Direct/Cached.
    pub expert_sim: bool,
    /// Example 8×H100 NVLink profile (`expert_sim`).
    pub expert_8gpu: bool,
    /// Simulated expert page bytes (`expert_sim`). `None` is 4096.
    pub expert_bytes: Option<u64>,
    /// CUDA-like knobs for [`SimulatedGpuStore::with_cfg`]. Identity stays default.
    pub gpu_cfg: GpuStoreCfg,
    /// Miss-page placement. Default is pinned H2D.
    pub fill: GpuFill,
    /// Simulated KV page bytes when [`GpuStoreCfg::kv_sim`]. `None` uses intern geometry.
    pub kv_bytes: Option<u64>,
}

/// CLI switches for [`gpu_knobs`] / [`GpuFill`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GpuCli {
    pub cuda_graphs: bool,
    pub graph_update: bool,
    pub graph_set_params: bool,
    pub graph_clone: bool,
    pub graph_build: bool,
    pub graph_piecewise: bool,
    /// `cudaGraphNodeSetEnabled` skip extra combo children (`GpuStoreCfg::graph_enable`).
    pub graph_enable: bool,
    pub graph_mem: bool,
    pub graph_auto_free: bool,
    pub graph_mem_trim: bool,
    pub timing_events: bool,
    /// `cudaEventBlockingSync` copy events (`GpuStoreCfg::event_blocking_sync`).
    ///
    /// Implies [`Self::timing_events`]. Distinct from `--sync-policy blocking`.
    pub event_blocking_sync: bool,
    pub mapped: bool,
    pub managed: bool,
    pub vmm: bool,
    pub host_func: bool,
    pub blocking_streams: bool,
    pub sync_alloc: bool,
    pub mempool: bool,
    /// `cudaMemPoolTrimTo(0)` after score (`GpuStoreCfg::mempool_trim`). Implies mempool.
    pub mempool_trim: bool,
    /// `cudaMemPoolReuseAllowOpportunistic=0` (`GpuStoreCfg::mempool_no_reuse`). Implies mempool.
    pub mempool_no_reuse: bool,
    /// POSIX-FD shareable mempool IPC (`GpuStoreCfg::shareable`). Implies mempool.
    pub shareable: bool,
    pub pageable: bool,
    /// `cudaHostRegister` (`GpuStoreCfg::host_register`). Implies pageable.
    pub host_register: bool,
    /// `cudaHostRegisterMapped` expert pages (`GpuStoreCfg::host_register_mapped`). Implies mapped.
    pub host_register_mapped: bool,
    /// `cuPointerSetAttribute` SyncMemops (`GpuStoreCfg::sync_memops`). Host-sync H2D.
    pub sync_memops: bool,
    /// `cudaSetDeviceFlags` SyncMemops (`GpuStoreCfg::device_sync_memops`). Host-sync memcpy.
    pub device_sync_memops: bool,
    /// `cudaMemcpyBatchAsync` prefetch (`GpuStoreCfg::memcpy_batch`).
    pub memcpy_batch: bool,
    pub accessed_by: bool,
    pub legacy_null: bool,
    pub stream_priority: bool,
    /// Per-sequence copy streams (`GpuStoreCfg::seq_streams`).
    pub seq_streams: bool,
    /// Engine interned KV on the SimulatedGpuStore clock (`GpuStoreCfg::kv_sim`).
    pub kv_sim: bool,
    /// Decode GEMMs on a higher-priority compute stream (`GpuStoreCfg::decode_priority`).
    pub decode_priority: bool,
    /// `cudaLaunchCooperativeKernel` (`GpuStoreCfg::cooperative`).
    pub cooperative: bool,
    /// Same-stream PDL wait+trigger (`GpuStoreCfg::pdl`).
    pub pdl: bool,
    /// Persisting L2 access-policy window (`GpuStoreCfg::l2_persist`).
    pub l2_persist: bool,
    /// `cudaCtxResetPersistingL2Cache` after each GEMM (`GpuStoreCfg::l2_reset`).
    ///
    /// Implies [`Self::l2_persist`].
    pub l2_reset: bool,
    /// `cudaLimitMaxL2FetchGranularity` (`GpuStoreCfg::l2_fetch`). `0` is unset.
    ///
    /// Implies [`Self::l2_persist`]. `32` / `64` / `128` only.
    pub l2_fetch: u64,
    /// CUDA `hitRatio` as ‰ (`GpuStoreCfg::l2_ratio`). `0` is unset (`1000`).
    ///
    /// Implies [`Self::l2_persist`]. `1..=1000` when set.
    pub l2_ratio: u16,
    /// Hopper cluster X size (`GpuStoreCfg::cluster`). `0` is off.
    pub cluster: u8,
    /// True when `--cluster` appeared.
    pub cluster_set: bool,
    /// Hopper preferred cluster X (`GpuStoreCfg::preferred_cluster`). `0` is off.
    pub preferred_cluster: u8,
    /// True when `--preferred-cluster` appeared.
    pub preferred_cluster_set: bool,
    /// Spread cluster scheduling (`GpuStoreCfg::cluster_spread`).
    pub cluster_spread: bool,
    /// Function Spread cluster scheduling (`GpuStoreCfg::func_cluster_spread`).
    pub func_cluster_spread: bool,
    /// Launch LoadBalancing cluster scheduling (`GpuStoreCfg::cluster_load_balance`).
    pub cluster_load_balance: bool,
    /// `cudaFuncAttributeClusterDimMustBeSet` (`GpuStoreCfg::cluster_must_set`).
    pub cluster_must_set: bool,
    /// `cudaFuncAttributeRequiredClusterWidth` (`GpuStoreCfg::required_cluster`). `0` is unset.
    pub required_cluster: u8,
    /// True when `--required-cluster` appeared.
    pub required_cluster_set: bool,
    /// Max-shared carveout (`GpuStoreCfg::max_shared`).
    pub max_shared: bool,
    /// Function MaxShared carveout (`GpuStoreCfg::func_max_shared`).
    pub func_max_shared: bool,
    /// Launch MaxL1 carveout (`GpuStoreCfg::max_l1`).
    pub max_l1: bool,
    /// Non-portable cluster size (`GpuStoreCfg::non_portable_cluster`).
    pub non_portable_cluster: bool,
    /// Stream host-wait policy (`GpuStoreCfg::sync_policy`).
    pub sync_policy: SynchronizationPolicy,
    /// True when `--sync-policy` appeared.
    pub sync_policy_set: bool,
    /// Device host-wait schedule (`GpuStoreCfg::device_sync_policy`).
    pub device_sync_policy: SynchronizationPolicy,
    /// True when `--device-sync-policy` appeared.
    pub device_sync_policy_set: bool,
    /// Decode-stream mem-sync domain (`GpuStoreCfg::mem_sync_domain`).
    pub mem_sync_domain: MemSyncDomain,
    /// True when `--mem-sync-domain` appeared.
    pub mem_sync_domain_set: bool,
    /// Decode-stream mem-sync map collapse (`GpuStoreCfg::mem_sync_collapse`).
    pub mem_sync_collapse: bool,
    /// True when `--mem-sync-map` appeared.
    pub mem_sync_map_set: bool,
    /// Kernel-node bank width (`GpuStoreCfg::shared_mem`).
    pub shared_mem: SharedMemoryMode,
    /// True when `--shared-mem` appeared.
    pub shared_mem_set: bool,
    /// Function shared-mem bank width (`GpuStoreCfg::func_shared_mem`).
    pub func_shared_mem: SharedMemoryMode,
    /// True when `--func-shared-mem` appeared.
    pub func_shared_mem_set: bool,
    /// Device shared-mem bank width (`GpuStoreCfg::device_shared_mem`).
    pub device_shared_mem: SharedMemoryMode,
    /// True when `--device-shared-mem` appeared.
    pub device_shared_mem_set: bool,
    /// Portable-cluster size mode (`GpuStoreCfg::portable_cluster`).
    pub portable_cluster: PortableClusterMode,
    /// True when `--portable-cluster` appeared.
    pub portable_cluster_set: bool,
    /// `cudaFuncAttributeMaxDynamicSharedMemorySize` (`GpuStoreCfg::optin_shared`).
    pub optin_shared: bool,
    /// `cudaLaunchKernel` `sharedMemBytes` (`GpuStoreCfg::dynamic_shared`).
    pub dynamic_shared: u32,
    /// True when `--dynamic-shared` appeared.
    pub dynamic_shared_set: bool,
    /// CUDA 13 portable-shared mode (`GpuStoreCfg::portable_shared`).
    pub portable_shared: PortableSharedMode,
    /// True when `--portable-shared` appeared.
    pub portable_shared_set: bool,
    /// `cudaLaunchAttributeNvlinkUtilCentricScheduling` (`GpuStoreCfg::nvlink_util_centric`).
    pub nvlink_util: bool,
    /// `cudaLaunchAttributeDeviceUpdatableKernelNode` (`GpuStoreCfg::device_updatable`).
    pub device_updatable: bool,
    /// `cudaLaunchAttributePriority` (`GpuStoreCfg::kernel_priority`).
    pub kernel_priority: Option<i32>,
    /// `cudaGraphInstantiateFlagDeviceLaunch` (`GpuStoreCfg::device_launch`).
    pub device_launch: bool,
    /// `cudaLaunchAttributeLaunchCompletionEvent` (`GpuStoreCfg::launch_completion`).
    pub launch_completion: bool,
    /// `cudaLaunchAttributeProgrammaticEvent` (`GpuStoreCfg::programmatic_event`).
    pub programmatic_event: bool,
    /// `cudaStreamAttachMemAsync` Single (`GpuStoreCfg::stream_attach`). Implies managed.
    pub stream_attach: bool,
    /// `cudaMallocManaged` Host attach then Global (`GpuStoreCfg::managed_host`). Implies managed.
    pub managed_host: bool,
    /// `cudaMemPrefetchAsync` to host on managed evict (`GpuStoreCfg::prefetch_host`). Implies managed.
    pub prefetch_host: bool,
    /// `cuStreamWaitValue64` / `WriteValue64` copy-ready (`GpuStoreCfg::wait_value`).
    pub wait_value: bool,
    /// Hopper NVLS replica fanout (`GpuStoreCfg::multicast`). Implies vmm.
    pub multicast: bool,
    /// Hyper-Q occupancy (`GpuStoreCfg::compute_slots`). `0` keeps the profile.
    pub compute_slots: u8,
    /// True when `--compute-slots` appeared.
    pub compute_slots_set: bool,
    /// Decode-stream SM permille (`GpuStoreCfg::decode_sm_permille`).
    pub decode_sm_permille: u16,
    /// True when `--decode-sms` appeared.
    pub decode_sm_set: bool,
    /// `--kv-bytes` override. `None` uses intern K+V geometry.
    pub kv_bytes: Option<u64>,
    /// Physical span for [`GpuFill::Vmm`]. `0` maps the whole expert.
    pub vmm_page: u64,
    /// True when `--vmm-page` appeared (even if the value is `0`).
    pub vmm_page_set: bool,
}

impl GpuCli {
    /// True when `key` is a GPU switch. Inline values are refused.
    pub(crate) fn apply(&mut self, key: &str, inline: Option<&str>) -> Result<bool, String> {
        let slot = match key {
            "--cuda-graphs" => &mut self.cuda_graphs,
            "--graph-update" => &mut self.graph_update,
            "--graph-set-params" => &mut self.graph_set_params,
            "--graph-clone" => &mut self.graph_clone,
            "--graph-build" => &mut self.graph_build,
            "--graph-piecewise" => &mut self.graph_piecewise,
            "--graph-enable" => &mut self.graph_enable,
            "--graph-mem" => &mut self.graph_mem,
            "--graph-auto-free" => &mut self.graph_auto_free,
            "--graph-mem-trim" => &mut self.graph_mem_trim,
            "--timing-events" => &mut self.timing_events,
            "--event-blocking-sync" => &mut self.event_blocking_sync,
            "--mapped" => &mut self.mapped,
            "--managed" => &mut self.managed,
            "--vmm" => &mut self.vmm,
            "--host-func" => &mut self.host_func,
            "--blocking-streams" => &mut self.blocking_streams,
            "--sync-alloc" => &mut self.sync_alloc,
            "--mempool" => &mut self.mempool,
            "--mempool-trim" => &mut self.mempool_trim,
            "--mempool-no-reuse" => &mut self.mempool_no_reuse,
            "--shareable" => &mut self.shareable,
            "--pageable" => &mut self.pageable,
            "--host-register" => &mut self.host_register,
            "--host-register-mapped" => &mut self.host_register_mapped,
            "--sync-memops" => &mut self.sync_memops,
            "--device-sync-memops" => &mut self.device_sync_memops,
            "--memcpy-batch" => &mut self.memcpy_batch,
            "--accessed-by" => &mut self.accessed_by,
            "--legacy-null" => &mut self.legacy_null,
            "--stream-priority" => &mut self.stream_priority,
            "--seq-streams" => &mut self.seq_streams,
            "--kv-sim" => &mut self.kv_sim,
            "--decode-priority" => &mut self.decode_priority,
            "--cooperative" => &mut self.cooperative,
            "--pdl" => &mut self.pdl,
            "--l2-persist" => &mut self.l2_persist,
            "--l2-reset" => &mut self.l2_reset,
            "--cluster-spread" => &mut self.cluster_spread,
            "--func-cluster-spread" => &mut self.func_cluster_spread,
            "--cluster-load-balance" => &mut self.cluster_load_balance,
            "--cluster-must-set" => &mut self.cluster_must_set,
            "--max-shared" => &mut self.max_shared,
            "--func-max-shared" => &mut self.func_max_shared,
            "--max-l1" => &mut self.max_l1,
            "--non-portable-cluster" => &mut self.non_portable_cluster,
            "--optin-shared" => &mut self.optin_shared,
            "--nvlink-util" => &mut self.nvlink_util,
            "--device-updatable" => &mut self.device_updatable,
            "--device-launch" => &mut self.device_launch,
            "--launch-completion" => &mut self.launch_completion,
            "--programmatic-event" => &mut self.programmatic_event,
            "--stream-attach" => &mut self.stream_attach,
            "--managed-host" => &mut self.managed_host,
            "--prefetch-host" => &mut self.prefetch_host,
            "--wait-value" => &mut self.wait_value,
            "--multicast" => &mut self.multicast,
            _ => return Ok(false),
        };
        if inline.is_some() {
            return Err(format!("{key} does not take a value"));
        }
        *slot = true;
        Ok(true)
    }

    /// `N>0` maps experts in `N`-byte physicals (`va_acquire_paged`).
    pub(crate) fn set_vmm_page(&mut self, n: u64) {
        self.vmm_page = n;
        self.vmm_page_set = true;
    }

    /// `--vmm-page N` with `N>0` implies [`Self::vmm`]. Call after sim-flag checks.
    pub(crate) fn imply_vmm(&mut self) {
        if self.vmm_page > 0 || self.multicast {
            self.vmm = true;
        }
    }

    /// `--stream-attach` / `--managed-host` / `--prefetch-host` imply [`Self::managed`]. Call after sim-flag checks.
    pub(crate) fn imply_managed(&mut self) {
        if self.stream_attach || self.managed_host || self.prefetch_host {
            self.managed = true;
        }
    }

    /// `--host-register-mapped` implies [`Self::mapped`]. Call after sim-flag checks.
    pub(crate) fn imply_mapped(&mut self) {
        if self.host_register_mapped {
            self.mapped = true;
        }
    }

    /// `--shareable` / `--mempool-trim` / `--mempool-no-reuse` imply [`Self::mempool`]. Call after sim-flag checks.
    pub(crate) fn imply_shareable(&mut self) {
        if self.shareable || self.mempool_trim || self.mempool_no_reuse {
            self.mempool = true;
        }
    }

    /// `--host-register` implies [`Self::pageable`]. Call after sim-flag checks.
    pub(crate) fn imply_pageable(&mut self) {
        if self.host_register {
            self.pageable = true;
        }
    }

    /// `--l2-reset` / `--l2-fetch` / `--l2-ratio` imply [`Self::l2_persist`]. Call after sim-flag checks.
    pub(crate) fn imply_l2_persist(&mut self) {
        if self.l2_reset || self.l2_fetch != 0 || self.l2_ratio != 0 {
            self.l2_persist = true;
        }
    }

    /// `--event-blocking-sync` implies [`Self::timing_events`]. Call after sim-flag checks.
    pub(crate) fn imply_timing_events(&mut self) {
        if self.event_blocking_sync {
            self.timing_events = true;
        }
    }

    /// `--decode-priority` implies [`Self::stream_priority`]. `--decode-sms`
    /// and `--mem-sync-domain remote` imply both. Call after sim-flag checks.
    pub(crate) fn imply_decode_priority(&mut self) {
        if self.decode_sm_permille > 0 || self.mem_sync_domain == MemSyncDomain::Remote {
            self.decode_priority = true;
        }
        if self.decode_priority {
            self.stream_priority = true;
        }
    }

    /// Decode-stream SM permille (`--decode-sms`). `0` and `>1000` are refused.
    pub(crate) fn set_decode_sms(&mut self, n: u16) -> Result<(), String> {
        if n == 0 || n > 1000 {
            return Err("decode-sms must be 1..=1000".into());
        }
        self.decode_sm_permille = n;
        self.decode_sm_set = true;
        Ok(())
    }

    /// KV page bytes (`--kv-bytes`). `0` is refused.
    pub(crate) fn set_kv_bytes(&mut self, n: u64) -> Result<(), String> {
        if n == 0 {
            return Err("kv-bytes must be > 0".into());
        }
        self.kv_bytes = Some(n);
        Ok(())
    }

    /// Hyper-Q occupancy (`--compute-slots`). `0` is refused.
    pub(crate) fn set_compute_slots(&mut self, n: u8) -> Result<(), String> {
        if n == 0 {
            return Err("compute-slots must be > 0".into());
        }
        self.compute_slots = n;
        self.compute_slots_set = true;
        Ok(())
    }

    /// Hopper cluster X (`--cluster`). `0` is refused.
    pub(crate) fn set_cluster(&mut self, n: u8) -> Result<(), String> {
        if n == 0 {
            return Err("cluster must be > 0".into());
        }
        self.cluster = n;
        self.cluster_set = true;
        Ok(())
    }

    /// Function RequiredClusterWidth (`--required-cluster`). `0` is refused.
    pub(crate) fn set_required_cluster(&mut self, n: u8) -> Result<(), String> {
        if n == 0 {
            return Err("required-cluster must be > 0".into());
        }
        self.required_cluster = n;
        self.required_cluster_set = true;
        Ok(())
    }

    /// `cudaLimitMaxL2FetchGranularity` (`--l2-fetch`). `32` / `64` / `128` only.
    pub(crate) fn set_l2_fetch(&mut self, n: u64) -> Result<(), String> {
        if n != 32 && n != 64 && n != 128 {
            return Err("l2-fetch must be 32, 64, or 128".into());
        }
        self.l2_fetch = n;
        Ok(())
    }

    /// CUDA `hitRatio` as ‰ (`--l2-ratio`). `1..=1000`.
    pub(crate) fn set_l2_ratio(&mut self, n: u16) -> Result<(), String> {
        if n == 0 || n > 1000 {
            return Err("l2-ratio must be 1..=1000".into());
        }
        self.l2_ratio = n;
        Ok(())
    }

    /// Hopper preferred cluster X (`--preferred-cluster`). `0` is refused.
    pub(crate) fn set_preferred_cluster(&mut self, n: u8) -> Result<(), String> {
        if n == 0 {
            return Err("preferred-cluster must be > 0".into());
        }
        self.preferred_cluster = n;
        self.preferred_cluster_set = true;
        Ok(())
    }

    /// Stream host-wait policy (`--sync-policy auto|spin|yield|blocking`).
    pub(crate) fn set_sync_policy(&mut self, raw: &str) -> Result<(), String> {
        self.sync_policy =
            SynchronizationPolicy::parse(raw).map_err(|_| format!("unknown sync-policy {raw}"))?;
        self.sync_policy_set = true;
        Ok(())
    }

    /// Device host-wait schedule (`--device-sync-policy auto|spin|yield|blocking`).
    pub(crate) fn set_device_sync_policy(&mut self, raw: &str) -> Result<(), String> {
        self.device_sync_policy = SynchronizationPolicy::parse(raw)
            .map_err(|_| format!("unknown device-sync-policy {raw}"))?;
        self.device_sync_policy_set = true;
        Ok(())
    }

    /// Decode-stream mem-sync domain (`--mem-sync-domain default|remote`).
    pub(crate) fn set_mem_sync_domain(&mut self, raw: &str) -> Result<(), String> {
        self.mem_sync_domain =
            MemSyncDomain::parse(raw).map_err(|_| format!("unknown mem-sync-domain {raw}"))?;
        self.mem_sync_domain_set = true;
        Ok(())
    }

    /// Decode-stream mem-sync map (`--mem-sync-map identity|collapse`).
    pub(crate) fn set_mem_sync_map(&mut self, raw: &str) -> Result<(), String> {
        self.mem_sync_collapse = match raw {
            "identity" => false,
            "collapse" => true,
            _ => return Err(format!("unknown mem-sync-map {raw}")),
        };
        self.mem_sync_map_set = true;
        Ok(())
    }

    /// Kernel-node bank width (`--shared-mem default|four|eight`).
    pub(crate) fn set_shared_mem(&mut self, raw: &str) -> Result<(), String> {
        self.shared_mem =
            SharedMemoryMode::parse(raw).map_err(|_| format!("unknown shared-mem {raw}"))?;
        self.shared_mem_set = true;
        Ok(())
    }

    /// Function shared-mem bank width (`--func-shared-mem default|four|eight`).
    pub(crate) fn set_func_shared_mem(&mut self, raw: &str) -> Result<(), String> {
        self.func_shared_mem =
            SharedMemoryMode::parse(raw).map_err(|_| format!("unknown func-shared-mem {raw}"))?;
        self.func_shared_mem_set = true;
        Ok(())
    }

    /// Device shared-mem bank width (`--device-shared-mem default|four|eight`).
    pub(crate) fn set_device_shared_mem(&mut self, raw: &str) -> Result<(), String> {
        self.device_shared_mem =
            SharedMemoryMode::parse(raw).map_err(|_| format!("unknown device-shared-mem {raw}"))?;
        self.device_shared_mem_set = true;
        Ok(())
    }

    /// Launch-time portable cluster (`--portable-cluster default|portable|non-portable`).
    pub(crate) fn set_portable_cluster(&mut self, raw: &str) -> Result<(), String> {
        self.portable_cluster = PortableClusterMode::parse(raw)
            .map_err(|_| format!("unknown portable-cluster {raw}"))?;
        self.portable_cluster_set = true;
        Ok(())
    }

    /// `cudaLaunchKernel` `sharedMemBytes` (`--dynamic-shared`). `0` is refused.
    pub(crate) fn set_dynamic_shared(&mut self, n: u32) -> Result<(), String> {
        if n == 0 {
            return Err("dynamic-shared must be > 0".into());
        }
        self.dynamic_shared = n;
        self.dynamic_shared_set = true;
        Ok(())
    }

    /// `cudaLaunchAttributePriority` (`--kernel-priority`). `0` is a valid override.
    pub(crate) fn set_kernel_priority(&mut self, n: i32) {
        self.kernel_priority = Some(n);
    }

    /// CUDA 13 portable-shared mode (`--portable-shared default|portable|non-portable`).
    pub(crate) fn set_portable_shared(&mut self, raw: &str) -> Result<(), String> {
        self.portable_shared =
            PortableSharedMode::parse(raw).map_err(|_| format!("unknown portable-shared {raw}"))?;
        self.portable_shared_set = true;
        Ok(())
    }

    /// Preferred cluster needs a required `--cluster` that it is a multiple of.
    pub(crate) fn check_preferred_cluster(self) -> Result<(), String> {
        if !self.preferred_cluster_set {
            return Ok(());
        }
        if !self.cluster_set {
            return Err("--preferred-cluster needs --cluster".into());
        }
        if !self.preferred_cluster.is_multiple_of(self.cluster) {
            return Err("preferred-cluster must be a multiple of cluster".into());
        }
        Ok(())
    }

    /// `--cluster-must-set` needs a required `--cluster`.
    pub(crate) fn check_cluster_must_set(self) -> Result<(), String> {
        if self.cluster_must_set && !self.cluster_set {
            return Err("--cluster-must-set needs --cluster".into());
        }
        Ok(())
    }

    /// `--required-cluster N` needs `--cluster` and must equal it.
    pub(crate) fn check_required_cluster(self) -> Result<(), String> {
        if !self.required_cluster_set {
            return Ok(());
        }
        if !self.cluster_set {
            return Err("--required-cluster needs --cluster".into());
        }
        if self.required_cluster != self.cluster {
            return Err("required-cluster must match --cluster".into());
        }
        Ok(())
    }

    /// `--cluster-load-balance` needs `--func-cluster-spread` and is exclusive with `--cluster-spread`.
    pub(crate) fn check_cluster_load_balance(self) -> Result<(), String> {
        if self.cluster_load_balance && self.cluster_spread {
            return Err("choose one of --cluster-load-balance, --cluster-spread".into());
        }
        if self.cluster_load_balance && !self.func_cluster_spread {
            return Err("--cluster-load-balance needs --func-cluster-spread".into());
        }
        Ok(())
    }

    /// `--max-l1` needs `--func-max-shared` and is exclusive with `--max-shared`.
    pub(crate) fn check_max_l1(self) -> Result<(), String> {
        if self.max_l1 && self.max_shared {
            return Err("choose one of --max-l1, --max-shared".into());
        }
        if self.max_l1 && !self.func_max_shared {
            return Err("--max-l1 needs --func-max-shared".into());
        }
        Ok(())
    }

    /// `--mem-sync-map collapse` needs `--mem-sync-domain remote`.
    pub(crate) fn check_mem_sync_map(self) -> Result<(), String> {
        if self.mem_sync_collapse && self.mem_sync_domain != MemSyncDomain::Remote {
            return Err("--mem-sync-map collapse needs --mem-sync-domain remote".into());
        }
        Ok(())
    }

    /// First CUDA knob that needs `--expert-sim`, if any.
    #[must_use]
    pub(crate) fn sim_flag(self) -> Option<&'static str> {
        [
            (self.cuda_graphs, "--cuda-graphs"),
            (self.graph_update, "--graph-update"),
            (self.graph_set_params, "--graph-set-params"),
            (self.graph_clone, "--graph-clone"),
            (self.graph_build, "--graph-build"),
            (self.graph_piecewise, "--graph-piecewise"),
            (self.graph_enable, "--graph-enable"),
            (self.graph_mem, "--graph-mem"),
            (self.graph_auto_free, "--graph-auto-free"),
            (self.graph_mem_trim, "--graph-mem-trim"),
            (self.timing_events, "--timing-events"),
            (self.event_blocking_sync, "--event-blocking-sync"),
            (self.mapped, "--mapped"),
            (self.managed, "--managed"),
            (self.vmm, "--vmm"),
            (self.host_func, "--host-func"),
            (self.blocking_streams, "--blocking-streams"),
            (self.sync_alloc, "--sync-alloc"),
            (self.mempool, "--mempool"),
            (self.mempool_trim, "--mempool-trim"),
            (self.mempool_no_reuse, "--mempool-no-reuse"),
            (self.shareable, "--shareable"),
            (self.pageable, "--pageable"),
            (self.host_register, "--host-register"),
            (self.host_register_mapped, "--host-register-mapped"),
            (self.sync_memops, "--sync-memops"),
            (self.device_sync_memops, "--device-sync-memops"),
            (self.memcpy_batch, "--memcpy-batch"),
            (self.accessed_by, "--accessed-by"),
            (self.legacy_null, "--legacy-null"),
            (self.stream_priority, "--stream-priority"),
            (self.seq_streams, "--seq-streams"),
            (self.kv_sim, "--kv-sim"),
            (self.decode_priority, "--decode-priority"),
            (self.cooperative, "--cooperative"),
            (self.pdl, "--pdl"),
            (self.l2_persist, "--l2-persist"),
            (self.l2_reset, "--l2-reset"),
            (self.l2_fetch != 0, "--l2-fetch"),
            (self.l2_ratio != 0, "--l2-ratio"),
            (self.multicast, "--multicast"),
            (self.vmm_page_set, "--vmm-page"),
            (self.compute_slots_set, "--compute-slots"),
            (self.cluster_set, "--cluster"),
            (self.preferred_cluster_set, "--preferred-cluster"),
            (self.cluster_spread, "--cluster-spread"),
            (self.func_cluster_spread, "--func-cluster-spread"),
            (self.cluster_load_balance, "--cluster-load-balance"),
            (self.cluster_must_set, "--cluster-must-set"),
            (self.required_cluster_set, "--required-cluster"),
            (self.max_shared, "--max-shared"),
            (self.func_max_shared, "--func-max-shared"),
            (self.max_l1, "--max-l1"),
            (self.non_portable_cluster, "--non-portable-cluster"),
            (self.sync_policy_set, "--sync-policy"),
            (self.device_sync_policy_set, "--device-sync-policy"),
            (self.mem_sync_domain_set, "--mem-sync-domain"),
            (self.mem_sync_map_set, "--mem-sync-map"),
            (self.shared_mem_set, "--shared-mem"),
            (self.func_shared_mem_set, "--func-shared-mem"),
            (self.device_shared_mem_set, "--device-shared-mem"),
            (self.portable_cluster_set, "--portable-cluster"),
            (self.optin_shared, "--optin-shared"),
            (self.dynamic_shared_set, "--dynamic-shared"),
            (self.portable_shared_set, "--portable-shared"),
            (self.nvlink_util, "--nvlink-util"),
            (self.device_updatable, "--device-updatable"),
            (self.kernel_priority.is_some(), "--kernel-priority"),
            (self.device_launch, "--device-launch"),
            (self.launch_completion, "--launch-completion"),
            (self.programmatic_event, "--programmatic-event"),
            (self.stream_attach, "--stream-attach"),
            (self.managed_host, "--managed-host"),
            (self.prefetch_host, "--prefetch-host"),
            (self.wait_value, "--wait-value"),
            (self.decode_sm_set, "--decode-sms"),
        ]
        .into_iter()
        .find_map(|(on, name)| on.then_some(name))
    }

    /// Pinned when every fill flag is off; otherwise exactly one of mapped/managed/vmm.
    pub(crate) fn fill(self) -> Result<GpuFill, String> {
        GpuFill::from_flags(self.mapped, self.managed, self.vmm).map_err(|e| e.to_string())
    }
}

/// GPU knobs plus predictor planner (`--prefetch` / Stay vs Fetch).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlannerCli {
    /// SimulatedGpuStore CUDA / fill flags.
    pub gpu: GpuCli,
    /// Predictor mode. Default [`Prefetch::Both`].
    pub prefetch: Prefetch,
    /// Unique predicted-key Stay vs Fetch window. `0` is ungated.
    pub plan_window: usize,
    /// Stay permille of that window already resident.
    pub plan_threshold: u32,
    prefetch_set: bool,
    window_set: bool,
    threshold_set: bool,
}

impl Default for PlannerCli {
    fn default() -> Self {
        Self {
            gpu: GpuCli::default(),
            prefetch: Prefetch::Both,
            plan_window: 0,
            plan_threshold: 500,
            prefetch_set: false,
            window_set: false,
            threshold_set: false,
        }
    }
}

/// Result of [`PlannerCli::take`] classifying one dashed operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dash {
    Taken,
    Need(PlanSlot),
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlanSlot {
    VmmPage,
    KvBytes,
    ComputeSlots,
    Cluster,
    RequiredCluster,
    PreferredCluster,
    SyncPolicy,
    DeviceSyncPolicy,
    MemSyncDomain,
    MemSyncMap,
    SharedMem,
    FuncSharedMem,
    DeviceSharedMem,
    PortableCluster,
    DynamicShared,
    PortableShared,
    KernelPriority,
    DecodeSms,
    L2Fetch,
    L2Ratio,
    Prefetch,
    PlanWindow,
    PlanThreshold,
}

impl PlanSlot {
    fn name(self) -> &'static str {
        match self {
            Self::VmmPage => "vmm-page",
            Self::KvBytes => "kv-bytes",
            Self::ComputeSlots => "compute-slots",
            Self::Cluster => "cluster",
            Self::RequiredCluster => "required-cluster",
            Self::PreferredCluster => "preferred-cluster",
            Self::SyncPolicy => "sync-policy",
            Self::DeviceSyncPolicy => "device-sync-policy",
            Self::MemSyncDomain => "mem-sync-domain",
            Self::MemSyncMap => "mem-sync-map",
            Self::SharedMem => "shared-mem",
            Self::FuncSharedMem => "func-shared-mem",
            Self::DeviceSharedMem => "device-shared-mem",
            Self::PortableCluster => "portable-cluster",
            Self::DynamicShared => "dynamic-shared",
            Self::PortableShared => "portable-shared",
            Self::KernelPriority => "kernel-priority",
            Self::DecodeSms => "decode-sms",
            Self::L2Fetch => "l2-fetch",
            Self::L2Ratio => "l2-ratio",
            Self::Prefetch => "prefetch",
            Self::PlanWindow => "plan-window",
            Self::PlanThreshold => "plan-threshold",
        }
    }
}

impl PlannerCli {
    fn dash(&mut self, key: &str, inline: Option<&str>) -> Result<Dash, String> {
        if self.gpu.apply(key, inline)? {
            return Ok(Dash::Taken);
        }
        Ok(match key {
            "--vmm-page" => Dash::Need(PlanSlot::VmmPage),
            "--kv-bytes" => Dash::Need(PlanSlot::KvBytes),
            "--compute-slots" => Dash::Need(PlanSlot::ComputeSlots),
            "--cluster" => Dash::Need(PlanSlot::Cluster),
            "--required-cluster" => Dash::Need(PlanSlot::RequiredCluster),
            "--l2-fetch" => Dash::Need(PlanSlot::L2Fetch),
            "--l2-ratio" => Dash::Need(PlanSlot::L2Ratio),
            "--preferred-cluster" => Dash::Need(PlanSlot::PreferredCluster),
            "--sync-policy" => Dash::Need(PlanSlot::SyncPolicy),
            "--device-sync-policy" => Dash::Need(PlanSlot::DeviceSyncPolicy),
            "--mem-sync-domain" => Dash::Need(PlanSlot::MemSyncDomain),
            "--mem-sync-map" => Dash::Need(PlanSlot::MemSyncMap),
            "--shared-mem" => Dash::Need(PlanSlot::SharedMem),
            "--func-shared-mem" => Dash::Need(PlanSlot::FuncSharedMem),
            "--device-shared-mem" => Dash::Need(PlanSlot::DeviceSharedMem),
            "--portable-cluster" => Dash::Need(PlanSlot::PortableCluster),
            "--dynamic-shared" => Dash::Need(PlanSlot::DynamicShared),
            "--portable-shared" => Dash::Need(PlanSlot::PortableShared),
            "--kernel-priority" => Dash::Need(PlanSlot::KernelPriority),
            "--decode-sms" => Dash::Need(PlanSlot::DecodeSms),
            "--prefetch" => Dash::Need(PlanSlot::Prefetch),
            "--plan-window" => Dash::Need(PlanSlot::PlanWindow),
            "--plan-threshold" => Dash::Need(PlanSlot::PlanThreshold),
            _ => Dash::Unknown,
        })
    }

    fn set(&mut self, slot: PlanSlot, raw: &str) -> Result<(), String> {
        match slot {
            PlanSlot::VmmPage => {
                let n = raw
                    .parse::<u64>()
                    .map_err(|_| format!("invalid vmm-page {raw:?}"))?;
                self.gpu.set_vmm_page(n);
            }
            PlanSlot::KvBytes => {
                let n = raw
                    .parse::<u64>()
                    .map_err(|_| format!("invalid kv-bytes {raw:?}"))?;
                self.gpu.set_kv_bytes(n)?;
            }
            PlanSlot::ComputeSlots => {
                let n = raw
                    .parse::<u8>()
                    .map_err(|_| format!("invalid compute-slots {raw:?}"))?;
                self.gpu.set_compute_slots(n)?;
            }
            PlanSlot::Cluster => {
                let n = raw
                    .parse::<u8>()
                    .map_err(|_| format!("invalid cluster {raw:?}"))?;
                self.gpu.set_cluster(n)?;
            }
            PlanSlot::RequiredCluster => {
                let n = raw
                    .parse::<u8>()
                    .map_err(|_| format!("invalid required-cluster {raw:?}"))?;
                self.gpu.set_required_cluster(n)?;
            }
            PlanSlot::L2Fetch => {
                let n = raw
                    .parse::<u64>()
                    .map_err(|_| format!("invalid l2-fetch {raw:?}"))?;
                self.gpu.set_l2_fetch(n)?;
            }
            PlanSlot::L2Ratio => {
                let n = raw
                    .parse::<u16>()
                    .map_err(|_| format!("invalid l2-ratio {raw:?}"))?;
                self.gpu.set_l2_ratio(n)?;
            }
            PlanSlot::PreferredCluster => {
                let n = raw
                    .parse::<u8>()
                    .map_err(|_| format!("invalid preferred-cluster {raw:?}"))?;
                self.gpu.set_preferred_cluster(n)?;
            }
            PlanSlot::SyncPolicy => self.gpu.set_sync_policy(raw)?,
            PlanSlot::DeviceSyncPolicy => self.gpu.set_device_sync_policy(raw)?,
            PlanSlot::MemSyncDomain => self.gpu.set_mem_sync_domain(raw)?,
            PlanSlot::MemSyncMap => self.gpu.set_mem_sync_map(raw)?,
            PlanSlot::SharedMem => self.gpu.set_shared_mem(raw)?,
            PlanSlot::FuncSharedMem => self.gpu.set_func_shared_mem(raw)?,
            PlanSlot::DeviceSharedMem => self.gpu.set_device_shared_mem(raw)?,
            PlanSlot::PortableCluster => self.gpu.set_portable_cluster(raw)?,
            PlanSlot::DynamicShared => {
                let n = raw
                    .parse::<u32>()
                    .map_err(|_| format!("invalid dynamic-shared {raw:?}"))?;
                self.gpu.set_dynamic_shared(n)?;
            }
            PlanSlot::PortableShared => self.gpu.set_portable_shared(raw)?,
            PlanSlot::KernelPriority => {
                let n = raw
                    .parse::<i32>()
                    .map_err(|_| format!("invalid kernel-priority {raw:?}"))?;
                self.gpu.set_kernel_priority(n);
            }
            PlanSlot::DecodeSms => {
                let n = raw
                    .parse::<u16>()
                    .map_err(|_| format!("invalid decode-sms {raw:?}"))?;
                self.gpu.set_decode_sms(n)?;
            }
            PlanSlot::Prefetch => {
                self.prefetch =
                    Prefetch::parse(raw).map_err(|_| format!("unknown prefetch {raw}"))?;
                self.prefetch_set = true;
            }
            PlanSlot::PlanWindow => {
                self.plan_window = raw
                    .parse::<usize>()
                    .map_err(|_| format!("invalid plan-window {raw:?}"))?;
                self.window_set = true;
            }
            PlanSlot::PlanThreshold => {
                self.plan_threshold = raw
                    .parse::<u32>()
                    .map_err(|_| format!("invalid plan-threshold {raw:?}"))?;
                self.threshold_set = true;
            }
        }
        Ok(())
    }

    /// Consume a dashed Engine / serve flag. `false` means the key is unknown.
    pub(crate) fn take<I, S>(
        &mut self,
        key: &str,
        inline: Option<&str>,
        it: &mut I,
    ) -> Result<bool, String>
    where
        I: Iterator<Item = S>,
        S: AsRef<str>,
    {
        match self.dash(key, inline)? {
            Dash::Taken => Ok(true),
            Dash::Need(slot) => {
                let v = match inline {
                    Some(v) => v.to_string(),
                    None => match it.next() {
                        Some(s) => s.as_ref().to_string(),
                        None => return Err(format!("missing --{} value", slot.name())),
                    },
                };
                self.set(slot, &v)?;
                Ok(true)
            }
            Dash::Unknown => Ok(false),
        }
    }

    /// First planner flag that needs `--engine` on `serve`.
    #[must_use]
    pub(crate) fn serve_engine_flag(self) -> Option<&'static str> {
        [
            (self.prefetch_set, "--prefetch"),
            (self.window_set, "--plan-window"),
            (self.threshold_set, "--plan-threshold"),
        ]
        .into_iter()
        .find_map(|(on, name)| on.then_some(name))
    }
}

/// Build [`GpuStoreCfg`] from parsed Engine / serve GPU flags.
#[must_use]
pub(crate) fn gpu_knobs(gpu: GpuCli) -> GpuStoreCfg {
    GpuStoreCfg {
        host_func: gpu.host_func,
        blocking_streams: gpu.blocking_streams,
        sync_alloc: gpu.sync_alloc,
        mempool: gpu.mempool,
        mempool_trim: gpu.mempool_trim,
        mempool_no_reuse: gpu.mempool_no_reuse,
        shareable: gpu.shareable,
        vmm_page: gpu.vmm_page,
        pageable: gpu.pageable,
        host_register: gpu.host_register,
        host_register_mapped: gpu.host_register_mapped,
        sync_memops: gpu.sync_memops,
        device_sync_memops: gpu.device_sync_memops,
        memcpy_batch: gpu.memcpy_batch,
        accessed_by: gpu.accessed_by,
        legacy_null: gpu.legacy_null,
        stream_priority: gpu.stream_priority,
        graph_update: gpu.graph_update,
        graph_set_params: gpu.graph_set_params,
        graph_clone: gpu.graph_clone,
        graph_build: gpu.graph_build,
        graph_piecewise: gpu.graph_piecewise,
        graph_enable: gpu.graph_enable,
        graph_mem: gpu.graph_mem,
        graph_auto_free: gpu.graph_auto_free,
        graph_mem_trim: gpu.graph_mem_trim,
        timing_events: gpu.timing_events || gpu.event_blocking_sync,
        event_blocking_sync: gpu.event_blocking_sync,
        seq_streams: gpu.seq_streams,
        kv_sim: gpu.kv_sim,
        decode_priority: gpu.decode_priority,
        cooperative: gpu.cooperative,
        pdl: gpu.pdl,
        l2_persist: gpu.l2_persist || gpu.l2_reset || gpu.l2_fetch != 0 || gpu.l2_ratio != 0,
        l2_reset: gpu.l2_reset,
        l2_fetch: gpu.l2_fetch,
        l2_ratio: gpu.l2_ratio,
        cluster: gpu.cluster,
        preferred_cluster: gpu.preferred_cluster,
        cluster_spread: gpu.cluster_spread,
        func_cluster_spread: gpu.func_cluster_spread,
        cluster_load_balance: gpu.cluster_load_balance,
        cluster_must_set: gpu.cluster_must_set,
        required_cluster: gpu.required_cluster,
        max_shared: gpu.max_shared,
        func_max_shared: gpu.func_max_shared,
        max_l1: gpu.max_l1,
        non_portable_cluster: gpu.non_portable_cluster,
        sync_policy: gpu.sync_policy,
        device_sync_policy: gpu.device_sync_policy,
        mem_sync_domain: gpu.mem_sync_domain,
        mem_sync_collapse: gpu.mem_sync_collapse,
        shared_mem: gpu.shared_mem,
        func_shared_mem: gpu.func_shared_mem,
        device_shared_mem: gpu.device_shared_mem,
        portable_cluster: gpu.portable_cluster,
        optin_shared: gpu.optin_shared,
        dynamic_shared: gpu.dynamic_shared,
        portable_shared: gpu.portable_shared,
        nvlink_util_centric: gpu.nvlink_util,
        device_updatable: gpu.device_updatable,
        kernel_priority: gpu.kernel_priority,
        device_launch: gpu.device_launch,
        launch_completion: gpu.launch_completion,
        programmatic_event: gpu.programmatic_event,
        stream_attach: gpu.stream_attach,
        managed_host: gpu.managed_host,
        prefetch_host: gpu.prefetch_host,
        wait_value: gpu.wait_value,
        multicast: gpu.multicast,
        compute_slots: gpu.compute_slots,
        decode_sm_permille: gpu.decode_sm_permille,
    }
}

/// Park DirectStore / CachedStore / SimulatedGpuStore on `eng`.
pub(crate) fn attach_store(
    eng: &mut Engine<'_>,
    llama: &Llama,
    spec: &StoreAttach,
) -> Result<(), String> {
    if spec.expert_sim {
        let slots = match spec.expert_slots {
            Some(0) => return Err("--expert-sim needs --expert-slots > 0".into()),
            Some(n) => n,
            None => 8,
        };
        let profile = if spec.expert_8gpu {
            HardwareProfile::example_8xh100_nvlink()
        } else {
            HardwareProfile::example_h100_sxm()
        };
        let gpu = SimulatedGpuStore::with_cfg(
            llama.expert_direct_store().map_err(|e| e.to_string())?,
            slots,
            profile,
            spec.expert_bytes.unwrap_or(4096),
            spec.fill,
            spec.gpu_cfg,
        )
        .map_err(|e| e.to_string())?;
        eng.attach_expert_store(LiveStore::simulated(gpu));
        if spec.gpu_cfg.kv_sim {
            eng.enable_kv_sim(spec.kv_bytes)
                .map_err(|e| e.to_string())?;
        }
        return Ok(());
    }
    let Some(slots) = spec.expert_slots else {
        return Ok(());
    };
    let direct = llama.expert_direct_store().map_err(|e| e.to_string())?;
    let store = if slots == 0 {
        LiveStore::Direct(direct)
    } else {
        LiveStore::Cached(CachedStore::new(direct, slots).map_err(|e| e.to_string())?)
    };
    eng.attach_expert_store(store);
    Ok(())
}
