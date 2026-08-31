//! [`ExpertStore`] backed by [`gpu_sim`]: H2D, mapped host, managed, or VMM on miss.

use crate::access::{ExpertKey, Trace};
use crate::error::Error;
use crate::kv::KvSimOp;
use crate::place::home_gpu;
use crate::planner::{
    plan_placement, plan_window, predicted_keys, window_keys, ChainState, Markov, Placement, Plan,
    Prefetch, DECODE_ACTIVATION_BYTES,
};
use crate::sim_replay::{
    add_leaf_gemm, alloc_launch_completion, alloc_programmatic_event, alloc_resident_copy_mailbox,
    allow_non_portable_cluster_if, allow_optin_shared_if, apply_cluster_dim_must_be_set,
    apply_device_shared_mem, apply_device_sync_memops, apply_device_sync_policy,
    apply_exec_mem_sync_domain, apply_exec_mem_sync_map, apply_func_cluster_spread,
    apply_func_max_shared, apply_func_shared_mem, apply_l2_fetch, apply_required_cluster_width,
    apply_stream_mem_sync_domain, apply_stream_mem_sync_map, apply_stream_sync_policy,
    bind_device_mempools, check_cluster_load_balance, check_cluster_must_set,
    check_cluster_preferred, check_device_graph_flags, check_l2_fetch, check_l2_ratio,
    check_max_l1, check_mem_sync_collapse, check_mem_sync_launch, check_mem_sync_launch_map,
    check_required_cluster, collapse_mem_sync_map, ensure_single_attach, free_copy_mailbox,
    free_mapped_host, instantiate_exec, kernel_leaf, mark_sync_memops, memcpy_batch_attr,
    mempool_hold, persist_armed, replay_exec, replay_streams, reset_persisting_l2_if,
    retarget_parked_kernel, signal_copy_ready, stream_of, upload_after_set_params, wait_copy_ready,
    wait_memcpy_during_allocs, GemmFlags, LeafMem, StreamPlan,
};
use crate::store::{CachedStore, DirectStore, ExpertParts, ExpertPhase, ExpertStore, StoreMetrics};
use gpu_sim::{
    AllocId, ClusterSchedulingPolicy, DeviceFlags, DeviceId, DeviceLimit, EventId, GraphId,
    GraphMemAttr, HardwareProfile, KernelBuf, KernelKind, LaunchCompletionEvent, MemAdvise,
    MemAttach, MemHandleId, MemSyncDomain, MemcpyOp, Place, PointerAttr, PoolId, ProgrammaticEvent,
    Score, SharedMemCarveout, SharedMemoryMode, Sim, StreamId, SynchronizationPolicy,
};
use std::collections::{BTreeMap, BTreeSet};

/// How [`SimulatedGpuStore`] places a miss page.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GpuFill {
    /// `cudaMallocAsync` + pinned H2D (decode identity).
    #[default]
    Pinned,
    /// `cudaMallocManaged` + ReadMostly + PreferredLocation + prefetch.
    ///
    /// [`GpuStoreCfg::stream_attach`] attaches Single to the compute stream
    /// and prefetches there. [`GpuStoreCfg::managed_host`] is
    /// `cudaMemAttachHost` at alloc, then Global attach on the copy stream
    /// (or Single when stream-attach). [`GpuStoreCfg::prefetch_host`] evicts
    /// with `cudaMemPrefetchAsync` to the host and restores by prefetching
    /// the same alloc back. Default is Global attach at alloc +
    /// copy-stream prefetch.
    Managed,
    /// `cudaHostAllocMapped` (PCIe kernel, no H2D).
    ///
    /// [`GpuStoreCfg::host_register_mapped`] is `alloc_host` then
    /// `cudaHostRegisterMapped` instead (same PCIe GEMM / pin budget /
    /// `hbm_peak` 0; evict `host_unregister` then `free_host`). Decode identity
    /// stays `cudaHostAllocMapped`.
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
    /// Blocking `cudaStreamCreate` for created streams (`1 .. n`).
    ///
    /// Default off (decode identity): copy is NULL, compute is `StreamId(1)`, so
    /// `n = 2`. [`Self::seq_streams`] marks copy streams `1 .. n_copy-1` and
    /// compute `StreamId(n_copy)` (`n = n_copy + 1`).
    pub blocking_streams: bool,
    /// Host-sync `cudaMalloc` / `cudaMemcpy` / `cudaFree` on pinned/VMM misses.
    pub sync_alloc: bool,
    /// Hold unused `cudaMallocAsync` bytes in the default pool (`u64::MAX` threshold).
    pub mempool: bool,
    /// `cudaMemPoolTrimTo(0)` unused cached bytes after [`SimulatedGpuStore::score`].
    ///
    /// Implies [`Self::mempool`] (hold during the run, return cache at idle).
    /// Illegal with [`Self::sync_alloc`] (`cudaMalloc` is not a mempool).
    /// Hits/misses stay the same; [`gpu_sim::Score::hbm_peak`] is unchanged.
    /// Decode identity stays no trim (CUDA default threshold 0 already
    /// returns on free).
    pub mempool_trim: bool,
    /// `cudaMemPoolReuseAllowOpportunistic=0`: skip cache reuse (OS alloc).
    ///
    /// Implies [`Self::mempool`] (hold unused bytes so skip-reuse is visible).
    /// Illegal with [`Self::sync_alloc`]. Hits/misses stay the same; leftover
    /// cache stays reserved so the next miss charges extra HBM
    /// (`alloc_overhead_ns` instead of `pool_reuse_ns`). Decode identity stays
    /// opportunistic reuse (CUDA default 1).
    pub mempool_no_reuse: bool,
    /// `cudaMemPoolProps::maxSize` on a `cudaMemPoolCreate` pool rebound with
    /// `cudaDeviceSetMemPool`.
    ///
    /// `0` is unset (unlimited). Implies [`Self::mempool`]. Illegal with
    /// [`Self::sync_alloc`]. Hits/misses stay the same when `N` fits the
    /// working set; reserved `live+cached` cannot grow past `N`. Decode
    /// identity stays the device default pool.
    pub mempool_max: u64,
    /// POSIX-FD shareable mempool IPC (`cudaMemPoolExportToShareableHandle`).
    ///
    /// Creates a shareable pool, exports it, imports a sibling that shares
    /// live/cached, and `cudaDeviceSetMemPool`s so miss pages draw from it.
    /// Implies [`Self::mempool`]. Illegal with [`Self::sync_alloc`] or
    /// mapped/managed/vmm fills. Decode identity stays the device default pool.
    pub shareable: bool,
    /// Physical span for [`GpuFill::Vmm`]. `0` maps the whole expert (`va_acquire`).
    pub vmm_page: u64,
    /// Pageable `cudaMemcpyAsync` (`memcpy_host_to_device`) instead of pinned DMA.
    ///
    /// Host-synchronous and slower (`pageable_permille`). Decode identity stays
    /// on pinned H2D.
    pub pageable: bool,
    /// `cudaHostRegister` on pageable staging so miss H2D is pinned DMA.
    ///
    /// Construction is `alloc_host` then `host_register` (mlock). Implies
    /// [`Self::pageable`]. Illegal with mapped/managed. Decode identity stays
    /// `cudaMallocHost` staging (no register).
    pub host_register: bool,
    /// `cudaHostRegisterMapped` on mapped expert pages (`alloc_host` then register).
    ///
    /// Kernels still read over PCIe with no H2D (`hbm_peak` 0). Implies mapped
    /// fill. Illegal with [`Self::host_register`] (unmapped staging). Evict is
    /// `host_unregister` then `free_host`. Decode identity stays
    /// `cudaHostAllocMapped`.
    pub host_register_mapped: bool,
    /// `cuPointerSetAttribute` [`gpu_sim::PointerAttr::SyncMemops`] on miss device pages.
    ///
    /// After alloc, before H2D / managed prefetch: memcpy and memset of that
    /// pointer are host-synchronous (`synchronize_stream` after submit).
    /// `cudaMallocAsync` waits the copy stream first so the pointer is live
    /// (leftover compute is not drained). Hits stay the same; leftover compute
    /// cannot overlap that copy. Illegal with mapped fill (no device memcpy)
    /// and [`Self::memcpy_batch`] (needs async H2D). Does not imply pageable
    /// or [`Self::sync_alloc`]. Decode identity stays async
    /// `memcpy_pinned_to_device`.
    pub sync_memops: bool,
    /// `cudaSetDeviceFlags(cudaDeviceSyncMemops)` on every GPU at construction.
    ///
    /// All runtime memcpy/memset on that device are host-synchronous, including
    /// unmarked pointers (replica D2D, KV). Distinct from per-page
    /// [`Self::sync_memops`]. Hits stay the same; leftover compute cannot
    /// overlap those copies. Illegal with mapped fill (no device memcpy) and
    /// [`Self::memcpy_batch`] (needs async H2D). Does not imply pageable or
    /// [`Self::sync_alloc`]. Decode identity stays async
    /// `memcpy_pinned_to_device`.
    pub device_sync_memops: bool,
    /// `cudaMemcpyBatchAsync` for a multi-expert pinned/VMM prefetch on one stream.
    ///
    /// Sibling H2D copies share one stream-order snapshot so they can occupy
    /// copy engines together. Illegal with pageable, host-sync, mapped,
    /// managed, [`Self::sync_memops`], or [`Self::device_sync_memops`] fills.
    /// Decode identity stays sequential `memcpy_pinned_to_device`.
    pub memcpy_batch: bool,
    /// `cudaMemcpySrcAccessOrderDuringApiCall` on [`Self::memcpy_batch`].
    ///
    /// The batch API waits those copies before return (not the whole stream).
    /// Needs [`Self::memcpy_batch`]. Hits/misses stay the same. Decode identity
    /// stays Stream order (or sequential H2D).
    pub memcpy_during: bool,
    /// `cudaMemcpySrcAccessOrderAny` on [`Self::memcpy_batch`].
    ///
    /// Empty intra-batch deps and no API wait (copies stay in flight). Needs
    /// [`Self::memcpy_batch`]. Exclusive with [`Self::memcpy_during`]. Hits/misses
    /// stay the same. Decode identity stays Stream order (or sequential H2D).
    pub memcpy_any: bool,
    /// Peer map without dest HBM: managed [`gpu_sim::MemAdvise::SetAccessedBy`],
    /// VMM [`gpu_sim::Sim::va_set_access`], or pinned-async
    /// [`gpu_sim::Sim::pool_set_access`] on every GPU.
    ///
    /// Expert GEMMs are reads-only, so a dest acquire does not migrate or
    /// charge dest HBM. Migrate retargets compute (no dest prefetch / VMM
    /// map+D2D / pinned D2D, no `drop_managed_copy` / `va_unmap` of home). Pin
    /// skips the replica prefetch, VMM dest map, or pinned D2D. No-op for
    /// [`GpuFill::Mapped`] or host-sync [`Self::sync_alloc`] (`cudaMalloc` is
    /// not a mempool). Decode identity stays off.
    pub accessed_by: bool,
    /// CUDA legacy null stream: copy (`StreamId(0)`) serializes with compute.
    ///
    /// Off by default (`cudaStreamNonBlocking` compute). Decode identity stays
    /// overlapping.
    pub legacy_null: bool,
    /// `cudaStreamCreateWithPriority` on created streams (priority = stream id).
    ///
    /// Default off: copy stays NULL at priority 0, compute is `StreamId(1)`.
    /// [`Self::seq_streams`] also marks the extra copy streams and compute
    /// `StreamId(n_copy)`. Decode identity stays default priority.
    pub stream_priority: bool,
    /// `cudaGraphExecUpdate` a parked exec onto the next miss alloc.
    ///
    /// Evict parks the instantiated GEMM graph instead of `destroy_graph`.
    /// The next capture on that GPU updates it (`graph_update_ns`) instead of
    /// instantiate. Decode identity stays destroy+instantiate.
    pub graph_update: bool,
    /// `cudaGraphExecKernelNodeSetParams` a parked exec onto the next miss.
    ///
    /// Evict parks the instantiated GEMM graph. The next miss retargets the
    /// unique kernel node (`graph_set_params_ns`) without recapture, and a
    /// unique memcpy or memset if the leaf has one. Walker combo parents
    /// additionally use `cudaGraphExecChildGraphNodeSetParams`. Works with
    /// [`Self::graph_mem`] / [`Self::graph_auto_free`]. Illegal with
    /// [`Self::graph_update`]. Decode identity stays destroy+instantiate.
    pub graph_set_params: bool,
    /// `cudaGraphClone` the capture before instantiate (graph vs exec).
    ///
    /// Clone, destroy the src, then instantiate the copy. A parked-exec
    /// update skips clone. Decode identity stays instantiate-in-place.
    pub graph_clone: bool,
    /// `cudaGraphCreate` / `cudaGraphAddKernelNode` instead of stream capture.
    ///
    /// Does not require an idle compute stream (`cudaStreamBeginCapture` does).
    /// Combo parents add children without [`gpu_sim::Sim::graph_add_dependencies`]
    /// so independent expert GEMMs may Hyper-Q overlap. Decode identity stays
    /// `begin_capture` / `end_capture`. Illegal with [`Self::graph_piecewise`].
    pub graph_build: bool,
    /// `cudaStreamBeginCaptureToGraph` combo parents (independent child roots).
    ///
    /// Walker combo parents capture each instantiated leaf into one parent as
    /// an extra root. Sibling GEMMs may Hyper-Q overlap. Store leaves capture
    /// into `create_graph`. Illegal with [`Self::graph_build`]. Decode identity
    /// stays `begin_capture` / `end_capture`.
    pub graph_piecewise: bool,
    /// `cudaGraphNodeSetEnabled` on walker combo parents instead of recapture.
    ///
    /// A later token that GEMMs a subset of a stored combo skips extra child
    /// graphs (`graph_set_params_ns`) instead of instantiate. Implies CUDA
    /// graphs on the walker. Illegal with [`Self::device_launch`]. Store GEMM
    /// stays per-leaf (`gemm_resident`). Decode identity stays exact combo
    /// recapture.
    pub graph_enable: bool,
    /// Leaf GEMM graphs include a scratch `cudaMallocAsync` + free.
    ///
    /// CUDA cannot `cudaGraphExecUpdate` mem nodes, so [`Self::graph_update`]
    /// is skipped. [`Self::graph_set_params`] still retargets the kernel.
    /// Decode identity stays kernel-only graphs.
    pub graph_mem: bool,
    /// Scratch alloc without a matching free; AutoFreeOnLaunch instantiate.
    ///
    /// Illegal with [`Self::graph_mem`]. `--graph-update` is skipped;
    /// [`Self::graph_set_params`] still retargets. Decode
    /// identity stays kernel-only graphs.
    pub graph_auto_free: bool,
    /// `cudaDeviceGraphMemTrim` unused reserved graph-mem after [`SimulatedGpuStore::score`].
    ///
    /// Does not change [`gpu_sim::Score::hbm_peak`]. Live graph allocs stay.
    /// Decode identity stays off (reserved is billed until trim).
    pub graph_mem_trim: bool,
    /// `cudaLaunchCooperativeKernel` for grouped GEMMs.
    ///
    /// Occupies every Hyper-Q slot so leftover prefill cannot overlap decode
    /// even when [`Self::compute_slots`] is `>=2`. Decode identity stays
    /// `cudaLaunchKernel`.
    pub cooperative: bool,
    /// Same-stream programmatic dependent launch for grouped GEMMs.
    ///
    /// Consecutive expert kernels on one compute stream may overlap after the
    /// previous kernel's PDL trigger when [`Self::compute_slots`] is `>=2`.
    /// Illegal with [`Self::cooperative`]. Decode identity stays
    /// `cudaLaunchKernel`.
    pub pdl: bool,
    /// `cudaLaunchAttributeAccessPolicyWindow` over each expert page.
    ///
    /// [`SimulatedGpuStore::with_cfg`] calls [`gpu_sim::Sim::enable_persisting_l2`]
    /// and launches GEMMs with a persisting window. Decode identity stays
    /// persist limit 0.
    pub l2_persist: bool,
    /// `cudaCtxResetPersistingL2Cache` after each resident GEMM.
    ///
    /// Implies [`Self::l2_persist`]. Live (cannot capture). Hits stay the same;
    /// a reused expert does not keep persisting L2 lines. Decode identity stays
    /// no reset.
    pub l2_reset: bool,
    /// `cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity)`. `0` is unset (128).
    ///
    /// Access-policy windows must align to this size. Implies
    /// [`Self::l2_persist`]. `32` / `64` / `128` only. Decode identity stays
    /// 128.
    pub l2_fetch: u64,
    /// CUDA `cudaAccessPolicyWindow.hitRatio` as ‰. `0` is unset (`1000`).
    ///
    /// Implies [`Self::l2_persist`]. `1..=1000` when set. A partial ratio bills
    /// more HBM than full persist on a reused expert. Decode identity stays
    /// 1000.
    pub l2_ratio: u16,
    /// Hopper cluster X size (`cudaLaunchAttributeClusterDimension`). `0` is off.
    ///
    /// Occupies `min(N, compute_slots)` Hyper-Q slots. Decode identity stays
    /// `cudaLaunchKernel` (no cluster).
    pub cluster: u8,
    /// Hopper preferred cluster X (`cudaLaunchAttributePreferredClusterDimension`).
    /// `0` is off.
    ///
    /// Occupancy uses this size when it fits in [`Self::compute_slots`], else
    /// [`Self::cluster`]. Requires [`Self::cluster`]. Must be an integer
    /// multiple of the required size. Decode identity stays no preferred dim.
    pub preferred_cluster: u8,
    /// Spread cluster scheduling (`cudaLaunchAttributeClusterSchedulingPolicyPreference`).
    ///
    /// Occupies every Hyper-Q slot so leftover kernels cannot overlap even
    /// when [`Self::cluster`] is smaller than [`Self::compute_slots`]. A no-op
    /// unless cluster blocks `> 1`. Decode identity stays Default.
    pub cluster_spread: bool,
    /// Function Spread cluster scheduling (`cudaFuncSetAttribute` ClusterSchedulingPolicyPreference).
    ///
    /// Launch Default inherits this occupancy so leftover kernels cannot
    /// overlap even when [`Self::cluster`] is smaller than [`Self::compute_slots`].
    /// Distinct from launch-attribute [`Self::cluster_spread`]. A no-op unless
    /// cluster blocks `> 1`. Decode identity stays Default.
    pub func_cluster_spread: bool,
    /// Launch LoadBalancing cluster scheduling (`cudaLaunchAttributeClusterSchedulingPolicyPreference`).
    ///
    /// Needs [`Self::func_cluster_spread`]. Overrides function Spread so leftover
    /// kernels can Hyper-Q overlap again. Exclusive with [`Self::cluster_spread`].
    /// A no-op unless cluster blocks `> 1`. Decode identity stays Default.
    pub cluster_load_balance: bool,
    /// `cudaFuncSetAttribute` ClusterDimMustBeSet.
    ///
    /// A grouped GEMM without [`Self::cluster`] is Invalid. Needs
    /// [`Self::cluster`]. Occupancy matches `--cluster` (SetAttribute is +1 ns).
    /// Decode identity stays unset.
    pub cluster_must_set: bool,
    /// `cudaFuncSetAttribute` RequiredClusterWidth. `0` is unset.
    ///
    /// Needs [`Self::cluster`] and must equal it. Occupancy matches `--cluster`
    /// (SetAttribute is +1 ns). Distinct from [`Self::cluster_must_set`] and
    /// [`Self::preferred_cluster`]. Decode identity stays unset.
    pub required_cluster: u8,
    /// Max-shared carveout (`cudaLaunchAttributePreferredSharedMemoryCarveout`).
    ///
    /// Occupies every Hyper-Q slot so leftover kernels cannot overlap.
    /// Decode identity stays Default.
    pub max_shared: bool,
    /// Function MaxShared carveout (`cudaFuncSetAttribute` PreferredSharedMemoryCarveout).
    ///
    /// Launch Default inherits this occupancy so leftover kernels cannot
    /// overlap. Distinct from launch-attribute [`Self::max_shared`]. Decode
    /// identity stays Default.
    pub func_max_shared: bool,
    /// Launch MaxL1 carveout (`cudaLaunchAttributePreferredSharedMemoryCarveout`).
    ///
    /// Needs [`Self::func_max_shared`]. Overrides function MaxShared so leftover
    /// kernels can Hyper-Q overlap again. Exclusive with [`Self::max_shared`].
    /// Decode identity stays Default.
    pub max_l1: bool,
    /// `cudaFuncAttributeNonPortableClusterSizeAllowed`. Default disallowed.
    ///
    /// Lets [`Self::cluster`] exceed `portable_cluster_size` up to
    /// `max_blocks_per_cluster`. Decode identity stays disallowed.
    pub non_portable_cluster: bool,
    /// Stream host-wait policy (`cudaLaunchAttributeSynchronizationPolicy`).
    ///
    /// [`SimulatedGpuStore::token_clock_ns`] / walker `sync_work` pay
    /// `host_sync_*_ns` on `synchronize_stream` (decode stream when
    /// [`Self::decode_priority`]). [`gpu_sim::SynchronizationPolicy::Auto`] tax
    /// is 0. Decode identity stays Auto.
    pub sync_policy: gpu_sim::SynchronizationPolicy,
    /// Device host-wait schedule (`cudaSetDeviceFlags` SCHEDULE_*).
    ///
    /// Auto streams inherit this tax on `synchronize_stream`. Explicit
    /// [`Self::sync_policy`] wins. [`gpu_sim::SynchronizationPolicy::Auto`]
    /// skips `set_device_flags` (decode identity). ORs with
    /// [`Self::device_sync_memops`]. Distinct from stream `--sync-policy`.
    pub device_sync_policy: gpu_sim::SynchronizationPolicy,
    /// `cudaLaunchAttributeMemSyncDomain` on the decode compute stream.
    ///
    /// [`gpu_sim::MemSyncDomain::Remote`] isolates leftover prefill fence tax
    /// when [`Self::decode_priority`] puts decode GEMMs on a second stream.
    /// Leaf graph replay SetAttributes the exec node to the launch stream
    /// when they disagree (CUDA graphs bake capture-time domain). Decode
    /// identity stays Default.
    pub mem_sync_domain: gpu_sim::MemSyncDomain,
    /// `cudaLaunchAttributeMemSyncDomainMap` collapse (remote→0).
    ///
    /// Needs [`Self::mem_sync_domain`] Remote. Restores leftover prefill fence
    /// tax. Decode identity stays CUDA identity (Hopper remote→1).
    pub mem_sync_collapse: bool,
    /// `cudaLaunchAttributeMemSyncDomain` Remote on grouped expert GEMMs.
    ///
    /// Needs [`Self::mem_sync_domain`] Remote. Overrides prefill inherit-Default
    /// so leftover prefill shares the decode Remote domain and fence tax
    /// returns. Decode identity stays inherit-stream.
    pub mem_sync_launch: bool,
    /// `cudaLaunchAttributeMemSyncDomainMap` collapse on grouped expert GEMMs.
    ///
    /// Needs [`Self::mem_sync_domain`] Remote. Overrides prefill inherit-identity
    /// so leftover prefill maps Default→0 with decode Remote→0 and fence tax
    /// returns. Decode identity stays inherit-stream.
    pub mem_sync_launch_map: bool,
    /// Shared-memory bank width (`cudaLaunchAttributeSharedMemoryMode`).
    ///
    /// Default never scales duration. FourByte / EightByte scale grouped GEMM
    /// time by `1000 / GpuProfile::shared_mem_*_permille` (profile default
    /// 1000). Decode identity stays Default.
    pub shared_mem: gpu_sim::SharedMemoryMode,
    /// Function shared-mem bank width (`cudaFuncSetSharedMemConfig`).
    ///
    /// Launch Default inherits this duration scale. Distinct from launch-attribute
    /// [`Self::shared_mem`]. Decode identity stays Default.
    pub func_shared_mem: gpu_sim::SharedMemoryMode,
    /// Device shared-mem bank width (`cudaDeviceSetSharedMemConfig`).
    ///
    /// Launch Default inherits this duration scale when the function config is
    /// also Default. Distinct from [`Self::func_shared_mem`] and launch-attribute
    /// [`Self::shared_mem`]. Decode identity stays Default.
    pub device_shared_mem: gpu_sim::SharedMemoryMode,
    /// Portable-cluster size mode (`cudaLaunchAttributePortableClusterSizeMode`).
    ///
    /// Default uses the current function attribute. RequirePortable always
    /// refuses a cluster larger than `portable_cluster_size`. AllowNonPortable
    /// allows up to `max_blocks_per_cluster` even when
    /// [`Self::non_portable_cluster`] is off. Decode identity stays Default.
    pub portable_cluster: gpu_sim::PortableClusterMode,
    /// `cudaFuncAttributeMaxDynamicSharedMemorySize` to the SKU opt-in max.
    ///
    /// Lets [`Self::dynamic_shared`] exceed `max_shared_mem_per_block`.
    /// Decode identity stays `0` (portable only).
    pub optin_shared: bool,
    /// `cudaLaunchKernel` `sharedMemBytes` on grouped expert GEMMs. `0` is off.
    ///
    /// Decode identity stays `0`.
    pub dynamic_shared: u32,
    /// CUDA 13 portable-shared mode (`cudaLaunchAttributeSharedMemoryMode`).
    ///
    /// Default uses the function attribute. RequirePortable always refuses
    /// oversize. AllowNonPortable allows up to `max_shared_mem_per_block_optin`
    /// even when [`Self::optin_shared`] is off. Decode identity stays Default.
    pub portable_shared: gpu_sim::PortableSharedMode,
    /// `cudaLaunchAttributeNvlinkUtilCentricScheduling`.
    ///
    /// Occupies every Hyper-Q slot when the profile has NVLink so leftover
    /// prefill cannot overlap decode even when [`Self::compute_slots`] is `>=2`.
    /// Without NVLink the flag is stored and occupancy is unchanged.
    /// Decode identity stays disabled.
    pub nvlink_util_centric: bool,
    /// `cudaLaunchAttributeDeviceUpdatableKernelNode` on grouped expert GEMMs.
    ///
    /// [`gpu_sim::Sim::graph_exec_kernel_set_params`] keeps the exec uploaded
    /// so a later [`gpu_sim::Sim::device_launch_graph`] needs no host
    /// `upload_graph`. Illegal with [`Self::graph_update`]. Decode identity
    /// stays not device-updatable.
    pub device_updatable: bool,
    /// `cudaLaunchAttributePriority` on grouped expert GEMMs.
    ///
    /// [`None`] inherits [`Self::stream_priority`] / stream create priority.
    /// [`Some`] overrides that kernel (memcpy stays on the stream). Higher
    /// first when compute contends. Flattening leftover-prefill vs decode
    /// stream ranking is expected when this is set. Instantiates with
    /// `cudaGraphInstantiateFlagUseNodePriority` so captured node values are
    /// used at replay. Decode identity stays inherit-stream.
    pub kernel_priority: Option<i32>,
    /// `cudaGraphInstantiateFlagDeviceLaunch` + [`gpu_sim::Sim::device_launch_graph`].
    ///
    /// Host [`gpu_sim::Sim::launch_graph`] stays legal. Illegal with
    /// [`Self::graph_update`] / [`Self::graph_mem`] / [`Self::graph_auto_free`]
    /// (mem nodes and ExecUpdate). Decode identity stays host launch.
    pub device_launch: bool,
    /// `cudaLaunchAttributeLaunchCompletionEvent` on grouped expert GEMMs.
    ///
    /// Other streams may `wait_event` when the kernel *starts*. [`Self::pin_hot`]
    /// replica D2D on `n_gpus >= 2` waits that event on the copy stream instead
    /// of draining the GEMM, so leftover compute overlaps the replica copy.
    /// Illegal with [`Self::device_launch`]. Decode identity stays no event.
    pub launch_completion: bool,
    /// `cudaLaunchAttributeProgrammaticEvent` on grouped expert GEMMs.
    ///
    /// Other streams may `wait_event` at the PDL trigger
    /// (`pdl_trigger_permille`) instead of kernel completion. [`Self::pin_hot`]
    /// replica D2D on `n_gpus >= 2` waits that event on the copy stream instead
    /// of draining the GEMM, so leftover compute overlaps the replica copy.
    /// Implies a PDL trigger on those GEMMs (same-stream PDL wait stays
    /// [`Self::pdl`]). Illegal with [`Self::device_launch`]. Decode identity
    /// stays no event.
    pub programmatic_event: bool,
    /// `cudaStreamAttachMemAsync(..., cudaMemAttachSingle)` on managed experts.
    ///
    /// After alloc+advise, attach the page to the compute stream and prefetch
    /// there so GEMM stays legal under Single. Identity managed prefetch stays
    /// on the copy stream (overlaps leftover compute). Implies managed fill.
    /// Illegal with [`Self::seq_streams`] (Single is one stream; seq-streams
    /// put walker GEMMs on per-sequence streams including NULL). Decode
    /// identity stays Global attach + copy-stream prefetch.
    pub stream_attach: bool,
    /// `cudaMallocManaged(..., cudaMemAttachHost)` then Global attach.
    ///
    /// After alloc+advise, attach Global on the copy stream so device prefetch
    /// is legal (Host attach fails device prefetch / kernels with
    /// `not attached`). Identity managed is Global at alloc (no Attach op).
    /// Implies managed fill. Prefetch stays on the copy stream unless
    /// [`Self::stream_attach`] (Single on compute). Decode identity stays
    /// Global alloc + copy-stream prefetch.
    pub managed_host: bool,
    /// `cudaMemPrefetchAsync(..., cudaCpuDeviceId)` on managed LRU evict.
    ///
    /// Keeps the `cudaMallocManaged` allocation host-resident instead of
    /// `cudaFree`. The next miss prefetches the same pointer back to the
    /// home GPU (no second alloc). Implies managed fill. Hits/misses stay
    /// the same. Decode identity stays `free_sync` on evict.
    pub prefetch_host: bool,
    /// `cuStreamWaitValue64` / `cuStreamWriteValue64` instead of copy-ready events.
    ///
    /// After H2D / prefetch, write a generation into an 8-byte device mailbox
    /// (not the expert page). The mailbox is `cudaMallocAsync` on the copy
    /// stream and that stream is waited *before* H2D so the pointer is
    /// resident for a compute `wait_value64` during DMA (`cudaMalloc` would
    /// drain leftover prefill). GEMM waits Eq on the compute stream. Replica
    /// D2D waits that mailbox on the copy stream. Decode identity stays
    /// `record_event` / `wait_event`. Graphs stay kernel-only.
    pub wait_value: bool,
    /// Hopper NVLS replica fanout (`cuMulticastCreate` / bind / kernel store).
    ///
    /// [`Self::pin_hot`] and walker `--place replicas` map dest VMM physicals
    /// then one NVLS kernel instead of sequential D2D. Occupies compute, not a
    /// copy engine. Requires [`GpuFill::Vmm`]. Illegal with [`Self::accessed_by`]
    /// or [`Self::vmm_page`]. Needs NVLink. Decode identity stays D2D.
    pub multicast: bool,
    /// Timing-on copy events (`cudaEventCreate`) and [`gpu_sim::Sim::event_elapsed_ns`].
    ///
    /// Default is `cudaEventDisableTiming` (vLLM wait events). Decode identity
    /// stays disable-timing; elapsed is not recorded.
    pub timing_events: bool,
    /// `cudaEventBlockingSync` on copy start/end events.
    ///
    /// [`gpu_sim::Sim::synchronize_event`] pays
    /// [`gpu_sim::GpuProfile::host_sync_blocking_ns`] instead of the recording
    /// stream's [`Self::sync_policy`]. Implies [`Self::timing_events`]. Distinct
    /// from `--sync-policy blocking` (that taxes `synchronize_stream`). Decode
    /// identity stays disable-timing non-blocking copy events.
    pub event_blocking_sync: bool,
    /// Per-sequence copy streams so concurrent H2D can overlap.
    ///
    /// `n_copy` is GPU0 `copy_engines.max(2)`; copy for sequence `s` is
    /// `s % n_copy`. Grouped GEMM stays on `StreamId(n_copy)` (not a
    /// per-sequence compute stream). Default off: copy is NULL (`StreamId(0)`),
    /// compute is `StreamId(1)`. Decode identity stays serial copies.
    pub seq_streams: bool,
    /// Map Engine interned KV blocks onto this Sim (`cuMemCreate` + `cuMemMap`).
    ///
    /// Default off (decode identity): scores bill expert H2D/GEMM only.
    /// `--kv-sim` reserves a VA of `pool_blocks` pages so TTFT/ITL include
    /// KV traffic on the same clock. Interned blocks share one physical
    /// handle. Distinct from `expertvm kv`.
    pub kv_sim: bool,
    /// Second compute stream at higher CUDA priority for decode GEMMs.
    ///
    /// Prefill stays on the existing compute stream. [`Self::stream_priority`]
    /// must also be on so decode actually preempts leftover prefill (priority
    /// equals stream id). Engine token-boundary ITL then
    /// [`SimulatedGpuStore::token_clock_ns`] (decode stream only) so leftover
    /// prefill does not inflate ITL. Default off: one compute stream (decode
    /// identity); ITL still [`SimulatedGpuStore::clock_ns`] (full drain).
    pub decode_priority: bool,
    /// Hyper-Q compute occupancy override (`0` keeps the profile).
    ///
    /// `1` is exclusive compute. `>=2` lets leftover prefill and decode GEMMs
    /// overlap at full issue rate when they sit on different streams
    /// ([`Self::decode_priority`]). Default `0` (profile `compute_slots`,
    /// example H100 is `1`) keeps decode identity.
    pub compute_slots: u8,
    /// Decode-stream green-context SM fraction (‰). `0` keeps a full chip.
    ///
    /// `1..=1000` calls [`gpu_sim::Sim::set_stream_sm_permille`] on the decode
    /// stream; leftover prefill gets the remainder when
    /// [`Self::decode_priority`] is on. Default `0` keeps decode identity.
    pub decode_sm_permille: u16,
}

#[derive(Clone, Copy)]
struct GpuPage {
    id: AllocId,
    device: DeviceId,
    /// Copy-stream event the compute stream must wait on, if not yet consumed.
    ready: Option<EventId>,
    /// Timing-on record before the copy, when [`GpuStoreCfg::timing_events`].
    start: Option<EventId>,
    /// 8-byte device mailbox when [`GpuStoreCfg::wait_value`].
    mailbox: Option<AllocId>,
    /// Generation [`signal_copy_ready`] wrote; [`None`] after GEMM waited.
    ready_gen: Option<u64>,
}

/// Bounded cache whose misses pay pinned H2D, mapped host, managed prefetch, or VMM.
pub struct SimulatedGpuStore {
    cache: CachedStore,
    sim: Sim,
    device: DeviceId,
    copy: StreamId,
    compute: StreamId,
    prefill: StreamId,
    decode: StreamId,
    decode_priority: bool,
    cooperative: bool,
    pdl: bool,
    l2_persist: bool,
    l2_reset: bool,
    l2_ratio: u16,
    cluster: u8,
    preferred_cluster: u8,
    cluster_spread: bool,
    cluster_load_balance: bool,
    max_shared: bool,
    max_l1: bool,
    mem_sync_launch: bool,
    mem_sync_launch_map: bool,
    shared_mem: gpu_sim::SharedMemoryMode,
    portable_cluster: gpu_sim::PortableClusterMode,
    dynamic_shared: u32,
    portable_shared: gpu_sim::PortableSharedMode,
    nvlink_util_centric: bool,
    device_updatable: bool,
    kernel_priority: Option<i32>,
    device_launch: bool,
    multicast: bool,
    next_event: u32,
    pages: BTreeMap<ExpertKey, GpuPage>,
    /// Peer copy dest per pinned key (`(home + 1) % n_gpus`).
    replicas: BTreeMap<ExpertKey, DeviceId>,
    evicting: BTreeMap<ExpertKey, GpuPage>,
    bytes_per_expert: u64,
    staging: AllocId,
    graphs: BTreeMap<AllocId, GraphId>,
    /// Instantiated GEMM execs parked on evict, keyed by the capture GPU.
    idle_execs: BTreeMap<DeviceId, Vec<GraphId>>,
    graph_launches: u64,
    graph_updates: u64,
    graph_clones: u64,
    graph_set_params_n: u64,
    graph_update: bool,
    graph_set_params: bool,
    graph_clone: bool,
    graph_build: bool,
    graph_piecewise: bool,
    leaf: LeafMem,
    graph_mem_trim: bool,
    mempool_trim: bool,
    /// [`GpuStoreCfg::mempool_max`] (`0` unset / unlimited).
    mempool_max: u64,
    timing_events: bool,
    event_blocking_sync: bool,
    copy_elapsed_ns: u64,
    mode: GpuFill,
    host_func: bool,
    sync_alloc: bool,
    /// [`GpuStoreCfg::vmm_page`]: KV-sized physicals when [`GpuFill::Vmm`].
    vmm_page: u64,
    /// Pageable H2D (`memcpy_host_to_device`) instead of pinned DMA.
    pageable: bool,
    /// [`GpuStoreCfg::host_register`]: pageable staging is `cudaHostRegister`'d.
    host_register: bool,
    /// [`GpuStoreCfg::host_register_mapped`]: mapped pages are `cudaHostRegisterMapped`.
    host_register_mapped: bool,
    /// [`GpuStoreCfg::sync_memops`]: miss pages set [`PointerAttr::SyncMemops`].
    sync_memops: bool,
    /// [`GpuStoreCfg::memcpy_batch`]: prefetch uses `cudaMemcpyBatchAsync`.
    memcpy_batch: bool,
    /// [`GpuStoreCfg::memcpy_during`]: batch attrs use DuringApiCall.
    memcpy_during: bool,
    /// [`GpuStoreCfg::memcpy_any`]: batch attrs use Any (empty deps, no wait).
    memcpy_any: bool,
    /// [`GpuStoreCfg::wait_value`]: copy-ready is wait/write-value, not events.
    wait_value: bool,
    /// [`GpuStoreCfg::accessed_by`]: managed/VMM/mempool pages stay on the home GPU.
    accessed_by: bool,
    /// Successful D2D [`Self::migrate`] calls (source device ≠ dest).
    migrates: u64,
    /// [`plan_placement`] chose [`Placement::DispatchActivations`] (no D2D).
    dispatches: u64,
    /// Successful peer replica copies from [`Self::pin_hot`].
    replicates: u64,
    /// [`GpuStoreCfg::seq_streams`]: `bind_sequence` retargets [`Self::copy`].
    seq_streams: bool,
    /// [`GpuStoreCfg::kv_sim`]: Engine interned KV on this Sim.
    kv_sim: bool,
    kv: Option<KvGpu>,
    /// Imported sibling of GPU0's shareable device mempool.
    share_import: Option<PoolId>,
    /// [`GpuStoreCfg::launch_completion`]: event recorded at GEMM kernel start
    /// so replica D2D can wait start instead of completion.
    launch_completion_event: Option<EventId>,
    /// A grouped GEMM with [`Self::launch_completion_event`] has been submitted.
    launch_completion_armed: bool,
    /// [`GpuStoreCfg::programmatic_event`]: event recorded at the PDL trigger.
    programmatic_event: Option<EventId>,
    /// A grouped GEMM with [`Self::programmatic_event`] has been submitted.
    programmatic_event_armed: bool,
    /// [`GpuStoreCfg::stream_attach`]: managed pages are `MemAttach::Single`
    /// on the compute stream; miss prefetch uses that stream.
    stream_attach: bool,
    /// [`GpuStoreCfg::managed_host`]: `cudaMallocManaged` Host attach, then
    /// Global attach on the miss DMA stream (unless stream-attach Single).
    managed_host: bool,
    /// [`GpuStoreCfg::prefetch_host`]: evict prefetches to host instead of free.
    prefetch_host: bool,
    /// Managed allocs that left HBM via [`Self::prefetch_host`] (still live).
    host_pages: BTreeMap<ExpertKey, GpuPage>,
}

struct KvGpu {
    va: AllocId,
    page_bytes: u64,
    n_pages: u32,
    mapped: BTreeSet<u32>,
    handles: BTreeMap<u32, MemHandleId>,
    hits: u64,
    misses: u64,
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
    /// destroy+instantiate graphs (stream capture; [`GpuStoreCfg::graph_build`]
    /// is `cudaGraphCreate` / `cudaGraphAdd*`; [`GpuStoreCfg::graph_mem`]
    /// records in-graph scratch with a matching free;
    /// [`GpuStoreCfg::graph_auto_free`] is AutoFreeOnLaunch without a free;
    /// [`GpuStoreCfg::graph_set_params`] retargets a parked kernel node;
    /// [`GpuStoreCfg::graph_enable`] is walker combo `cudaGraphNodeSetEnabled`
    /// (store GEMM stays per-leaf; illegal with device-launch)),
    /// disable-timing copy events.
    /// [`GpuStoreCfg::cooperative`] launches GEMMs with
    /// `cudaLaunchCooperativeKernel` (exclusive compute).
    /// [`GpuStoreCfg::pdl`] is same-stream programmatic dependent launch
    /// (illegal with cooperative).
    /// [`GpuStoreCfg::l2_persist`] is `cudaLaunchAttributeAccessPolicyWindow`
    /// over expert pages (persisting L2 after the first fill).
    /// [`GpuStoreCfg::l2_reset`] is `cudaCtxResetPersistingL2Cache` after each
    /// resident GEMM (implies persist; live; cannot capture).
    /// [`GpuStoreCfg::l2_fetch`] is `cudaLimitMaxL2FetchGranularity` (implies
    /// persist; windows must align; `32`/`64`/`128`).
    /// [`GpuStoreCfg::l2_ratio`] is CUDA `hitRatio` as ‰ (implies persist;
    /// `1..=1000`; unset is 1000).
    /// [`GpuStoreCfg::cluster`] / [`GpuStoreCfg::preferred_cluster`] are Hopper
    /// thread-block cluster dims.
    /// [`GpuStoreCfg::cluster_spread`] is launch-attribute Spread.
    /// [`GpuStoreCfg::func_cluster_spread`] is `cudaFuncSetAttribute`
    /// ClusterSchedulingPolicyPreference Spread (launch Default inherits).
    /// [`GpuStoreCfg::cluster_load_balance`] is launch-attribute LoadBalancing
    /// (needs `--func-cluster-spread`; restores Hyper-Q overlap).
    /// [`GpuStoreCfg::cluster_must_set`] is `cudaFuncSetAttribute`
    /// ClusterDimMustBeSet (needs `--cluster`; occupancy matches `--cluster`).
    /// [`GpuStoreCfg::required_cluster`] is `cudaFuncSetAttribute`
    /// RequiredClusterWidth (needs `--cluster`; must match; occupancy matches
    /// `--cluster`).
    /// [`GpuStoreCfg::sync_policy`] is stream host-wait
    /// (`cudaLaunchAttributeSynchronizationPolicy`; Auto tax 0).
    /// [`GpuStoreCfg::device_sync_policy`] is device host-wait
    /// (`cudaSetDeviceFlags` SCHEDULE_*; Auto streams inherit; explicit
    /// stream policy wins).
    /// [`GpuStoreCfg::mem_sync_domain`] is decode-stream
    /// `cudaLaunchAttributeMemSyncDomain` (Default identity; Remote isolates
    /// leftover prefill fence tax).
    /// [`GpuStoreCfg::mem_sync_collapse`] is decode-stream
    /// `cudaLaunchAttributeMemSyncDomainMap` collapse (needs Remote; restores
    /// leftover prefill fence tax).
    /// [`GpuStoreCfg::mem_sync_launch`] is launch-attribute Remote on grouped
    /// GEMMs (needs Remote; restores leftover prefill fence tax).
    /// [`GpuStoreCfg::mem_sync_launch_map`] is launch-attribute collapse map on
    /// grouped GEMMs (needs Remote; restores leftover prefill fence tax).
    /// [`GpuStoreCfg::max_shared`] is launch-attribute MaxShared carveout.
    /// [`GpuStoreCfg::func_max_shared`] is `cudaFuncSetAttribute`
    /// PreferredSharedMemoryCarveout MaxShared (launch Default inherits).
    /// [`GpuStoreCfg::max_l1`] is launch-attribute MaxL1 (needs
    /// `--func-max-shared`; restores Hyper-Q overlap).
    /// [`GpuStoreCfg::shared_mem`] is kernel-node bank width
    /// (`cudaLaunchAttributeSharedMemoryMode`; Default never scales).
    /// [`GpuStoreCfg::func_shared_mem`] is `cudaFuncSetSharedMemConfig`
    /// (launch Default inherits; distinct from launch-attribute `--shared-mem`).
    /// [`GpuStoreCfg::device_shared_mem`] is `cudaDeviceSetSharedMemConfig`
    /// (launch Default inherits when function config is also Default).
    /// [`GpuStoreCfg::portable_cluster`] is launch-time portable cluster mode
    /// (`cudaLaunchAttributePortableClusterSizeMode`; Default uses the function attr).
    /// [`GpuStoreCfg::optin_shared`] is `cudaFuncAttributeMaxDynamicSharedMemorySize`
    /// to the SKU opt-in max. [`GpuStoreCfg::dynamic_shared`] is
    /// `cudaLaunchKernel` `sharedMemBytes`. [`GpuStoreCfg::portable_shared`] is
    /// CUDA 13 `cudaLaunchAttributeSharedMemoryMode`.
    /// [`GpuStoreCfg::device_updatable`] is
    /// `cudaLaunchAttributeDeviceUpdatableKernelNode`.
    /// [`GpuStoreCfg::kernel_priority`] is `cudaLaunchAttributePriority`.
    /// [`GpuStoreCfg::device_launch`] is `cudaGraphInstantiateFlagDeviceLaunch`.
    /// [`GpuStoreCfg::launch_completion`] is
    /// `cudaLaunchAttributeLaunchCompletionEvent` on grouped GEMMs (replica
    /// D2D waits kernel start; illegal with device-launch).
    /// [`GpuStoreCfg::programmatic_event`] is
    /// `cudaLaunchAttributeProgrammaticEvent` on grouped GEMMs (replica D2D
    /// waits the PDL trigger; illegal with device-launch).
    /// [`GpuStoreCfg::stream_attach`] is `cudaStreamAttachMemAsync` Single
    /// on managed experts (prefetch on compute; illegal with seq-streams).
    /// [`GpuStoreCfg::managed_host`] is `cudaMallocManaged(..., cudaMemAttachHost)`
    /// then Global attach on the copy stream so prefetch is legal (implies
    /// managed; identity stays Global at alloc).
    /// [`GpuStoreCfg::prefetch_host`] is `cudaMemPrefetchAsync` to
    /// `cudaCpuDeviceId` on managed evict (keeps the alloc; next miss
    /// prefetches back; implies managed; identity stays `free_sync`).
    /// [`GpuStoreCfg::wait_value`] is `cuStreamWaitValue64` / `WriteValue64`
    /// for the copy-ready handshake (8-byte `cudaMallocAsync` mailbox, copy
    /// stream waited before H2D; decode identity stays events).
    /// [`GpuStoreCfg::mempool_trim`] is `cudaMemPoolTrimTo(0)` after
    /// [`Self::score`] (implies mempool hold; illegal with
    /// [`GpuStoreCfg::sync_alloc`]; [`Self::clock_ns`] / token ITL do not trim).
    /// [`GpuStoreCfg::mempool_no_reuse`] is
    /// `cudaMemPoolReuseAllowOpportunistic=0` (implies mempool hold; leftover
    /// cache stays reserved; illegal with sync-alloc).
    /// [`GpuStoreCfg::multicast`] is Hopper NVLS replica fanout (requires
    /// [`GpuFill::Vmm`] and NVLink).
    /// [`GpuStoreCfg::event_blocking_sync`] is `cudaEventBlockingSync` on
    /// timing copy events (implies [`GpuStoreCfg::timing_events`];
    /// `synchronize_event` pays `host_sync_blocking_ns`; distinct from
    /// [`GpuStoreCfg::sync_policy`] Blocking).
    /// [`GpuStoreCfg::memcpy_batch`] fills a multi-expert pinned/VMM prefetch
    /// with `cudaMemcpyBatchAsync` (sibling H2D copies share one stream-order
    /// snapshot). Demand [`ExpertStore::acquire`] stays sequential. Illegal
    /// with pageable, host-sync, mapped, managed, [`GpuStoreCfg::sync_memops`],
    /// or [`GpuStoreCfg::device_sync_memops`].
    /// [`GpuStoreCfg::host_register`] is `cudaHostRegister` on pageable staging
    /// so miss H2D is pinned DMA (implies pageable; illegal with mapped/managed).
    /// [`GpuStoreCfg::host_register_mapped`] is `cudaHostRegisterMapped` on
    /// mapped expert pages (`alloc_host` then register; implies mapped; illegal
    /// with [`GpuStoreCfg::host_register`]; identity stays `cudaHostAllocMapped`).
    /// [`GpuStoreCfg::sync_memops`] is `cuPointerSetAttribute` SyncMemops on
    /// miss device pages so H2D / managed prefetch is host-synchronous
    /// (illegal with mapped or memcpy-batch; identity stays async pinned H2D).
    /// [`GpuStoreCfg::device_sync_memops`] is `cudaSetDeviceFlags` SyncMemops
    /// so every memcpy/memset on that GPU is host-synchronous (illegal with
    /// mapped or memcpy-batch; identity stays async pinned H2D).
    /// [`GpuStoreCfg::device_sync_policy`] is `cudaSetDeviceFlags` SCHEDULE_*
    /// so Auto streams inherit host-wait tax (explicit stream
    /// [`GpuStoreCfg::sync_policy`] wins; ORs with SyncMemops).
    /// [`GpuStoreCfg::compute_slots`] `0` keeps the profile (example H100 is
    /// exclusive compute). [`GpuStoreCfg::decode_sm_permille`] `0` keeps a
    /// full chip. `1..=1000` without [`GpuStoreCfg::decode_priority`] caps the
    /// single compute stream.
    pub fn with_cfg(
        inner: DirectStore,
        slots: usize,
        profile: HardwareProfile,
        bytes_per_expert: u64,
        fill: GpuFill,
        cfg: GpuStoreCfg,
    ) -> Result<Self, Error> {
        if cfg.graph_update && cfg.graph_set_params {
            return Err(Error::Store("choose one of graph-update, graph-set-params"));
        }
        check_device_graph_flags(
            cfg.device_launch,
            cfg.device_updatable,
            cfg.graph_update,
            cfg.graph_mem,
            cfg.graph_auto_free,
        )?;
        if cfg.graph_build && cfg.graph_piecewise {
            return Err(Error::Store("choose one of graph-build, graph-piecewise"));
        }
        if cfg.graph_enable && cfg.device_launch {
            return Err(Error::Store("graph-enable cannot device-launch"));
        }
        if cfg.launch_completion && cfg.device_launch {
            return Err(Error::Store("launch-completion cannot device-launch"));
        }
        if cfg.programmatic_event && cfg.device_launch {
            return Err(Error::Store("programmatic-event cannot device-launch"));
        }
        if cfg.stream_attach && fill != GpuFill::Managed {
            return Err(Error::Store("stream-attach needs managed"));
        }
        if cfg.stream_attach && cfg.seq_streams {
            return Err(Error::Store("stream-attach cannot seq-streams"));
        }
        if cfg.managed_host && fill != GpuFill::Managed {
            return Err(Error::Store("managed-host needs managed"));
        }
        if cfg.prefetch_host && fill != GpuFill::Managed {
            return Err(Error::Store("prefetch-host needs managed"));
        }
        if cfg.pdl && cfg.cooperative {
            return Err(Error::Store("choose one of pdl, cooperative"));
        }
        check_cluster_preferred(cfg.cluster, cfg.preferred_cluster)?;
        check_cluster_must_set(cfg.cluster, cfg.cluster_must_set)?;
        check_required_cluster(cfg.cluster, cfg.required_cluster)?;
        check_mem_sync_collapse(cfg.mem_sync_domain, cfg.mem_sync_collapse)?;
        check_mem_sync_launch(cfg.mem_sync_domain, cfg.mem_sync_launch)?;
        check_mem_sync_launch_map(cfg.mem_sync_domain, cfg.mem_sync_launch_map)?;
        check_cluster_load_balance(
            cfg.cluster_load_balance,
            cfg.func_cluster_spread,
            cfg.cluster_spread,
        )?;
        check_max_l1(cfg.max_l1, cfg.func_max_shared, cfg.max_shared)?;
        check_l2_fetch(cfg.l2_fetch)?;
        check_l2_ratio(cfg.l2_ratio)?;
        if cfg.shareable && (cfg.sync_alloc || fill != GpuFill::Pinned) {
            return Err(Error::Store("shareable needs cudaMallocAsync"));
        }
        if cfg.mempool_trim && cfg.sync_alloc {
            return Err(Error::Store("mempool-trim needs cudaMallocAsync"));
        }
        if cfg.mempool_no_reuse && cfg.sync_alloc {
            return Err(Error::Store("mempool-no-reuse needs cudaMallocAsync"));
        }
        if cfg.mempool_max > 0 && cfg.sync_alloc {
            return Err(Error::Store("mempool-max needs cudaMallocAsync"));
        }
        if cfg.memcpy_during && !cfg.memcpy_batch {
            return Err(Error::Store("memcpy-during needs memcpy-batch"));
        }
        if cfg.memcpy_any && !cfg.memcpy_batch {
            return Err(Error::Store("memcpy-any needs memcpy-batch"));
        }
        if cfg.memcpy_any && cfg.memcpy_during {
            return Err(Error::Store("choose one of memcpy-any, memcpy-during"));
        }
        if cfg.memcpy_batch
            && (cfg.pageable
                || cfg.sync_alloc
                || cfg.sync_memops
                || cfg.device_sync_memops
                || fill == GpuFill::Mapped
                || fill == GpuFill::Managed)
        {
            return Err(Error::Store("memcpy-batch needs async pinned/vmm H2D"));
        }
        if cfg.sync_memops && fill == GpuFill::Mapped {
            return Err(Error::Store("sync-memops needs device memcpy"));
        }
        if cfg.device_sync_memops && fill == GpuFill::Mapped {
            return Err(Error::Store("device-sync-memops needs device memcpy"));
        }
        if cfg.host_register_mapped && cfg.host_register {
            return Err(Error::Store(
                "choose one of host-register, host-register-mapped",
            ));
        }
        if cfg.host_register && !cfg.pageable {
            return Err(Error::Store("host-register needs pageable"));
        }
        if cfg.host_register && (fill == GpuFill::Mapped || fill == GpuFill::Managed) {
            return Err(Error::Store("host-register needs pinned/vmm H2D"));
        }
        if cfg.host_register_mapped && fill != GpuFill::Mapped {
            return Err(Error::Store("host-register-mapped needs mapped"));
        }
        if cfg.multicast {
            if fill != GpuFill::Vmm {
                return Err(Error::Store("multicast requires vmm"));
            }
            if cfg.accessed_by {
                return Err(Error::Store("choose one of multicast, accessed-by"));
            }
            if cfg.vmm_page > 0 {
                return Err(Error::Store("multicast needs whole-VA maps"));
            }
            if profile.n_gpus() < 2 || !profile.has_nvlink() {
                return Err(Error::Store("multicast needs NVLink"));
            }
        }
        let leaf = LeafMem::from_flags(cfg.graph_mem, cfg.graph_auto_free)?;
        let bytes = bytes_per_expert.max(1);
        let (copy, prefill, decode, mark) =
            copy_compute_streams(&profile, cfg.seq_streams, cfg.decode_priority);
        let plan = StreamPlan::new(&profile, cfg.seq_streams, cfg.decode_priority);
        let profile = if cfg.compute_slots > 0 {
            profile.with_compute_slots(cfg.compute_slots)
        } else {
            profile
        };
        let mut sim = Sim::new(profile);
        apply_device_sync_memops(&mut sim, cfg.device_sync_memops)?;
        apply_device_sync_policy(&mut sim, cfg.device_sync_policy)?;
        apply_func_max_shared(&mut sim, cfg.func_max_shared)?;
        apply_func_cluster_spread(&mut sim, cfg.func_cluster_spread)?;
        apply_cluster_dim_must_be_set(&mut sim, cfg.cluster_must_set)?;
        apply_required_cluster_width(&mut sim, cfg.required_cluster)?;
        apply_l2_fetch(&mut sim, cfg.l2_fetch)?;
        apply_func_shared_mem(&mut sim, cfg.func_shared_mem)?;
        apply_device_shared_mem(&mut sim, cfg.device_shared_mem)?;
        if persist_armed(cfg.l2_persist, cfg.l2_reset, cfg.l2_fetch, cfg.l2_ratio) {
            sim.enable_persisting_l2()?;
        }
        allow_non_portable_cluster_if(&mut sim, cfg.non_portable_cluster)?;
        allow_optin_shared_if(&mut sim, cfg.optin_shared)?;
        let imported = if cfg.shareable || cfg.mempool_max > 0 {
            bind_device_mempools(&mut sim, cfg.shareable, cfg.mempool_max)?
        } else {
            BTreeMap::new()
        };
        let share_import = if cfg.shareable {
            imported.get(&DeviceId(0)).copied()
        } else {
            None
        };
        if mempool_hold(
            cfg.mempool,
            cfg.shareable,
            cfg.mempool_trim,
            cfg.mempool_no_reuse,
            cfg.mempool_max,
        ) {
            sim.set_default_pool_release_threshold(u64::MAX)?;
        }
        if cfg.mempool_no_reuse {
            crate::sim_replay::disable_pool_opportunistic_reuse(&mut sim)?;
        }
        if cfg.accessed_by && fill == GpuFill::Pinned && !cfg.sync_alloc {
            advise_pool_access(&mut sim)?;
        }
        if cfg.blocking_streams {
            sim.set_created_streams_blocking(mark)?;
        }
        if cfg.legacy_null {
            sim.set_legacy_null_stream(true);
        }
        if cfg.stream_priority {
            sim.set_created_streams_priority(mark)?;
        }
        apply_stream_sync_policy(&mut sim, plan, cfg.sync_policy)?;
        apply_stream_mem_sync_domain(&mut sim, plan, cfg.mem_sync_domain)?;
        apply_stream_mem_sync_map(&mut sim, plan, cfg.mem_sync_collapse)?;
        if cfg.decode_sm_permille > 0 {
            let dec = cfg.decode_sm_permille.min(1000);
            let pre = 1000u16.saturating_sub(dec).max(1);
            let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
            for g in 0..n {
                let d = DeviceId(g);
                sim.set_stream_sm_permille(d, decode, dec)?;
                if prefill != decode {
                    sim.set_stream_sm_permille(d, prefill, pre)?;
                }
            }
        }
        let cache_slots = mapped_occupancy(slots, fill, sim.pin_budget(), bytes);
        // Mapped expert pages already charge the pin budget. Pageable H2D
        // does not need a second mlock either unless `--host-register`.
        let staging = if fill == GpuFill::Mapped || (cfg.pageable && !cfg.host_register) {
            sim.alloc_host(bytes)?
        } else if cfg.host_register {
            let h = sim.alloc_host(bytes)?;
            sim.host_register(h)?;
            h
        } else {
            sim.alloc_host_pinned(bytes)?
        };
        let mut next_event = 1u32;
        let launch_ev = alloc_launch_completion(&mut sim, cfg.launch_completion, &mut next_event)?;
        let pde = alloc_programmatic_event(&mut sim, cfg.programmatic_event, &mut next_event)?;
        Ok(Self {
            cache: CachedStore::new(inner, cache_slots)?,
            sim,
            device: DeviceId(0),
            copy,
            compute: prefill,
            prefill,
            decode,
            decode_priority: cfg.decode_priority,
            cooperative: cfg.cooperative,
            pdl: cfg.pdl,
            l2_persist: persist_armed(cfg.l2_persist, cfg.l2_reset, cfg.l2_fetch, cfg.l2_ratio),
            l2_reset: cfg.l2_reset,
            l2_ratio: cfg.l2_ratio,
            cluster: cfg.cluster,
            preferred_cluster: cfg.preferred_cluster,
            cluster_spread: cfg.cluster_spread,
            cluster_load_balance: cfg.cluster_load_balance,
            max_shared: cfg.max_shared,
            max_l1: cfg.max_l1,
            mem_sync_launch: cfg.mem_sync_launch,
            mem_sync_launch_map: cfg.mem_sync_launch_map,
            shared_mem: cfg.shared_mem,
            portable_cluster: cfg.portable_cluster,
            dynamic_shared: cfg.dynamic_shared,
            portable_shared: cfg.portable_shared,
            nvlink_util_centric: cfg.nvlink_util_centric,
            device_updatable: cfg.device_updatable,
            kernel_priority: cfg.kernel_priority,
            device_launch: cfg.device_launch,
            multicast: cfg.multicast,
            next_event,
            pages: BTreeMap::new(),
            replicas: BTreeMap::new(),
            evicting: BTreeMap::new(),
            bytes_per_expert: bytes,
            staging,
            graphs: BTreeMap::new(),
            idle_execs: BTreeMap::new(),
            graph_launches: 0,
            graph_updates: 0,
            graph_clones: 0,
            graph_set_params_n: 0,
            graph_update: cfg.graph_update,
            graph_set_params: cfg.graph_set_params,
            graph_clone: cfg.graph_clone,
            graph_build: cfg.graph_build,
            graph_piecewise: cfg.graph_piecewise,
            leaf,
            graph_mem_trim: cfg.graph_mem_trim,
            mempool_trim: cfg.mempool_trim,
            mempool_max: cfg.mempool_max,
            timing_events: cfg.timing_events || cfg.event_blocking_sync,
            event_blocking_sync: cfg.event_blocking_sync,
            copy_elapsed_ns: 0,
            mode: fill,
            host_func: cfg.host_func,
            sync_alloc: cfg.sync_alloc,
            vmm_page: cfg.vmm_page,
            pageable: cfg.pageable,
            host_register: cfg.host_register,
            host_register_mapped: cfg.host_register_mapped,
            sync_memops: cfg.sync_memops,
            memcpy_batch: cfg.memcpy_batch,
            memcpy_during: cfg.memcpy_during,
            memcpy_any: cfg.memcpy_any,
            wait_value: cfg.wait_value,
            accessed_by: cfg.accessed_by,
            migrates: 0,
            dispatches: 0,
            replicates: 0,
            seq_streams: cfg.seq_streams,
            kv_sim: cfg.kv_sim,
            kv: None,
            share_import,
            launch_completion_event: launch_ev.map(|e| e.event),
            launch_completion_armed: false,
            programmatic_event: pde.map(|e| e.event),
            programmatic_event_armed: false,
            stream_attach: cfg.stream_attach,
            managed_host: cfg.managed_host,
            prefetch_host: cfg.prefetch_host,
            host_pages: BTreeMap::new(),
        })
    }

    /// Whether miss pages use unified memory (`cudaMallocManaged`).
    #[must_use]
    pub fn uses_managed(&self) -> bool {
        matches!(self.mode, GpuFill::Managed)
    }

    /// Whether miss pages use mapped host (`cudaHostAllocMapped` or
    /// `cudaHostRegisterMapped`).
    #[must_use]
    pub fn uses_mapped(&self) -> bool {
        matches!(self.mode, GpuFill::Mapped)
    }

    /// Whether miss pages use CUDA VMM (`va_acquire`).
    #[must_use]
    pub fn uses_vmm(&self) -> bool {
        matches!(self.mode, GpuFill::Vmm)
    }

    /// Whether batched prefetch waits copies at the API (`DuringApiCall`).
    #[must_use]
    pub fn memcpy_during(&self) -> bool {
        self.memcpy_during
    }

    /// Whether batched prefetch uses Any access order (empty deps, no wait).
    #[must_use]
    pub fn memcpy_any(&self) -> bool {
        self.memcpy_any
    }

    /// Stream memcpy ops (H2D / D2D) currently in the attached Sim.
    #[must_use]
    pub fn memcpy_operations(&self) -> Vec<gpu_sim::Operation> {
        self.sim
            .operations()
            .filter(|o| matches!(o.kind, gpu_sim::GpuOp::Memcpy(_)))
            .collect()
    }

    /// Page-locked staging buffer from construction; does not count toward HBM.
    #[must_use]
    pub fn staging_is_pinned(&self) -> bool {
        self.sim.is_host_pinned(self.staging).unwrap_or(false)
    }

    /// Whether the resident page for `key` came from `cudaHostRegisterMapped`.
    #[must_use]
    pub fn page_is_host_registered(&self, key: ExpertKey) -> bool {
        self.pages
            .get(&key)
            .and_then(|p| self.sim.is_host_registered(p.id).ok())
            .unwrap_or(false)
    }

    /// Whether the resident page for `key` is mapped into the device VA.
    #[must_use]
    pub fn page_is_host_mapped(&self, key: ExpertKey) -> bool {
        self.pages
            .get(&key)
            .and_then(|p| self.sim.is_host_mapped(p.id).ok())
            .unwrap_or(false)
    }

    /// Whether the resident page has [`PointerAttr::SyncMemops`].
    #[must_use]
    pub fn page_sync_memops(&self, key: ExpertKey) -> bool {
        self.pages
            .get(&key)
            .and_then(|p| {
                self.sim
                    .pointer_get_attribute(p.id, PointerAttr::SyncMemops)
                    .ok()
            })
            .is_some_and(|v| v != 0)
    }

    /// Whether this store set [`DeviceFlags::SYNC_MEMOPS`] at construction.
    #[must_use]
    pub fn device_sync_memops(&self) -> bool {
        self.sim
            .get_device_flags(self.device)
            .map(|f| f & DeviceFlags::SYNC_MEMOPS != 0)
            .unwrap_or(false)
    }

    /// Device schedule inherited by Auto streams (`cudaSetDeviceFlags`).
    #[must_use]
    pub fn device_sync_policy(&self) -> SynchronizationPolicy {
        self.sim
            .get_device_flags(self.device)
            .map(crate::sim_replay::device_schedule_from_flags)
            .unwrap_or(SynchronizationPolicy::Auto)
    }

    /// Whether this store set function MaxShared carveout at construction.
    #[must_use]
    pub fn func_max_shared(&self) -> bool {
        self.sim
            .get_func_carveout(self.device)
            .map(|c| c == SharedMemCarveout::MaxShared)
            .unwrap_or(false)
    }

    /// Whether grouped GEMMs launch with MaxL1 carveout.
    #[must_use]
    pub fn max_l1(&self) -> bool {
        self.gemm_flags().carveout() == SharedMemCarveout::MaxL1
    }

    /// Whether this store set function Spread cluster policy at construction.
    #[must_use]
    pub fn func_cluster_spread(&self) -> bool {
        self.sim
            .get_func_cluster_policy(self.device)
            .map(|p| p == ClusterSchedulingPolicy::Spread)
            .unwrap_or(false)
    }

    /// Whether grouped GEMMs launch with LoadBalancing cluster policy.
    #[must_use]
    pub fn cluster_load_balance(&self) -> bool {
        self.gemm_flags().cluster_policy() == ClusterSchedulingPolicy::LoadBalancing
    }

    /// Whether grouped GEMMs launch with Remote mem-sync domain.
    #[must_use]
    pub fn mem_sync_launch(&self) -> bool {
        self.gemm_flags().mem_sync_domain() == Some(MemSyncDomain::Remote)
    }

    /// Whether grouped GEMMs launch with a collapsed mem-sync map.
    #[must_use]
    pub fn mem_sync_launch_map(&self) -> bool {
        self.gemm_flags().mem_sync_map() == Some(collapse_mem_sync_map())
    }

    /// Current `cudaLimitMaxL2FetchGranularity` (`128` when unset).
    #[must_use]
    pub fn l2_fetch(&self) -> u64 {
        self.sim
            .get_limit(self.device, DeviceLimit::MaxL2FetchGranularity)
            .unwrap_or(128)
    }

    /// CUDA `hitRatio` as ‰ (`1000` when unset).
    #[must_use]
    pub fn l2_ratio(&self) -> u16 {
        match self.gemm_flags().l2_ratio {
            0 => 1000,
            n => n,
        }
    }

    /// Whether this store set `cudaFuncAttributeClusterDimMustBeSet`.
    #[must_use]
    pub fn cluster_must_set(&self) -> bool {
        self.sim
            .cluster_dim_must_be_set(self.device)
            .unwrap_or(false)
    }

    /// Function RequiredClusterWidth set at construction (`0` unset).
    #[must_use]
    pub fn required_cluster(&self) -> u8 {
        u8::try_from(self.sim.required_cluster_width(self.device).unwrap_or(0)).unwrap_or(0)
    }

    /// Function shared-mem bank width set at construction (`cudaFuncSetSharedMemConfig`).
    #[must_use]
    pub fn func_shared_mem(&self) -> SharedMemoryMode {
        self.sim
            .get_func_shared_mem_config(self.device)
            .unwrap_or(SharedMemoryMode::Default)
    }

    /// Device shared-mem bank width set at construction (`cudaDeviceSetSharedMemConfig`).
    #[must_use]
    pub fn device_shared_mem(&self) -> SharedMemoryMode {
        self.sim
            .get_shared_mem_config(self.device)
            .unwrap_or(SharedMemoryMode::Default)
    }

    /// Whether copy events were created with `cudaEventBlockingSync`.
    #[must_use]
    pub fn event_blocking_sync(&self) -> bool {
        self.event_blocking_sync
    }

    /// Whether this store resets persisting L2 after each resident GEMM.
    #[must_use]
    pub fn l2_reset(&self) -> bool {
        self.l2_reset
    }

    /// Unused bytes in GPU0's `cudaMallocAsync` pool after a device drain.
    pub fn default_pool_cached(&mut self) -> Result<u64, Error> {
        self.sim.synchronize()?;
        self.sweep_evicts();
        Ok(self
            .sim
            .pool_cached(self.sim.device_mempool(self.device)?)?)
    }

    /// `cudaMemPoolProps::maxSize` (`0` unset / unlimited).
    #[must_use]
    pub fn mempool_max(&self) -> u64 {
        self.mempool_max
    }

    /// Imported sibling of the shareable device mempool (`None` unless
    /// [`GpuStoreCfg::shareable`]).
    #[must_use]
    pub fn share_imported_pool(&self) -> Option<PoolId> {
        self.share_import
    }

    /// `cudaMallocFromPoolAsync` from the imported shareable sibling.
    ///
    /// After an evict with mempool hold, this reuses cached bytes (no extra
    /// HBM). Illegal unless [`GpuStoreCfg::shareable`].
    pub fn alloc_from_imported_pool(&mut self, bytes: u64) -> Result<AllocId, Error> {
        let pool = self
            .share_import
            .ok_or(Error::Store("no shareable import"))?;
        let id = self
            .sim
            .alloc_from_pool(self.device, pool, bytes, self.copy)?;
        self.sim.synchronize_stream(self.device, self.copy)?;
        Ok(id)
    }

    /// Bind H2D / alloc / free to the copy stream for `sequence`.
    ///
    /// No-op unless [`GpuStoreCfg::seq_streams`]. Grouped GEMM stays on
    /// [`Self::compute_stream`].
    pub fn bind_sequence(&mut self, sequence: u64) {
        if !self.seq_streams {
            return;
        }
        // Prefill id is `n_copy` even after `bind_decode_compute` retargets GEMM.
        let n = u8::try_from(self.prefill.0).unwrap_or(1);
        self.copy = stream_of(sequence, n);
    }

    /// Compute stream for grouped expert GEMM (not a per-sequence stream).
    #[must_use]
    pub fn compute_stream(&self) -> StreamId {
        self.compute
    }

    fn gemm_flags(&self) -> GemmFlags {
        GemmFlags {
            cooperative: self.cooperative,
            pdl: self.pdl,
            l2_persist: self.l2_persist,
            l2_ratio: self.l2_ratio,
            cluster: self.cluster,
            preferred_cluster: self.preferred_cluster,
            cluster_spread: self.cluster_spread,
            cluster_load_balance: self.cluster_load_balance,
            max_shared: self.max_shared,
            max_l1: self.max_l1,
            mem_sync_launch: self.mem_sync_launch,
            mem_sync_launch_map: self.mem_sync_launch_map,
            shared_mem: self.shared_mem,
            portable_cluster: self.portable_cluster,
            dynamic_shared: self.dynamic_shared,
            portable_shared: self.portable_shared,
            nvlink_util_centric: self.nvlink_util_centric,
            device_updatable: self.device_updatable,
            priority: self.kernel_priority,
            launch_completion: self
                .launch_completion_event
                .map(|event| LaunchCompletionEvent {
                    event,
                    external: false,
                }),
            programmatic_event: self.programmatic_event.map(|event| ProgrammaticEvent {
                event,
                external: false,
            }),
        }
    }

    /// Prefill grouped-GEMM stream (same as [`Self::compute_stream`] unless decode-priority).
    #[must_use]
    pub fn prefill_stream(&self) -> StreamId {
        self.prefill
    }

    /// Decode grouped-GEMM stream (same as prefill unless [`GpuStoreCfg::decode_priority`]).
    #[must_use]
    pub fn decode_stream(&self) -> StreamId {
        self.decode
    }

    /// Retarget grouped GEMM to the decode or prefill compute stream.
    ///
    /// No-op unless [`GpuStoreCfg::decode_priority`]. Engine `forward_batch`
    /// binds decode; prefill/replay binds prefill.
    pub fn bind_decode_compute(&mut self, decode: bool) {
        if !self.decode_priority {
            return;
        }
        self.compute = if decode { self.decode } else { self.prefill };
    }

    /// Copy stream for the currently bound sequence (NULL when seq-streams is off).
    #[must_use]
    pub fn copy_stream(&self) -> StreamId {
        self.copy
    }

    /// Drain the simulator and return its performance vector.
    ///
    /// [`GpuStoreCfg::graph_mem_trim`] / [`GpuStoreCfg::mempool_trim`] return
    /// unused reserved / cached bytes here (idle), not at token ITL.
    pub fn score(&mut self) -> Result<gpu_sim::Score, Error> {
        self.sim.synchronize()?;
        self.sweep_evicts();
        if self.graph_mem_trim {
            self.trim_graph_mem()?;
        }
        if self.mempool_trim {
            crate::sim_replay::trim_device_pools(&mut self.sim)?;
        }
        Ok(gpu_sim::Score::from_sim(&self.sim))
    }

    fn trim_graph_mem(&mut self) -> Result<(), Error> {
        crate::sim_replay::trim_graph_pools(&mut self.sim)
    }

    /// Drain and return the virtual clock (token-boundary sample).
    pub fn clock_ns(&mut self) -> Result<u64, Error> {
        self.sim.synchronize()?;
        self.sweep_evicts();
        Ok(self.sim.clock_ns())
    }

    /// Clock after the decode compute stream is idle (leftover prefill may run).
    ///
    /// When [`GpuStoreCfg::decode_priority`] is off this is [`Self::clock_ns`].
    /// Engine ITL uses this so a mixed leftover-prefill step does not wait the
    /// prefill stream. `score()` / waiter arrival still drain the whole node.
    pub fn token_clock_ns(&mut self) -> Result<u64, Error> {
        if !self.decode_priority {
            return self.clock_ns();
        }
        self.sync_decode()?;
        self.sweep_evicts();
        Ok(self.sim.clock_ns())
    }

    fn sync_decode(&mut self) -> Result<(), Error> {
        let n = u16::try_from(self.sim.profile().n_gpus()).unwrap_or(1);
        for g in 0..n {
            self.sim.synchronize_stream(DeviceId(g), self.decode)?;
        }
        Ok(())
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
        if self.memcpy_batch {
            self.place_pending(keys)?;
        } else {
            for key in keys {
                if self.cache.is_resident(*key) && !self.pages.contains_key(key) {
                    self.place(*key)?;
                }
            }
        }
        Ok(n)
    }

    /// Pin against eviction (sticky, survives compute `release`) and, on
    /// multi-GPU profiles, NVLink-replicate onto `(home + 1) % n_gpus`.
    ///
    /// Waits this page's GEMM lease before the replica copy so managed
    /// `prefetch` is not [`gpu_sim::SimError::Leased`]. Managed +
    /// `cudaMemAdviseSetAccessedBy` / VMM `cuMemSetAccess` / pinned
    /// `cudaMemPoolSetAccess` maps the dest without a prefetch or D2D.
    /// [`GpuStoreCfg::multicast`] maps dest VMM physicals then one NVLS kernel
    /// (`cuMulticastCreate`) instead of copy-engine D2D.
    /// 1-GPU profiles skip that wait so leftover prefill GEMMs can overlap
    /// decode-priority ITL samples. [`GpuStoreCfg::launch_completion`] waits
    /// kernel start on the copy stream instead of draining the GEMM, so a
    /// pinned replica D2D can overlap leftover compute.
    /// [`GpuStoreCfg::programmatic_event`] waits the PDL trigger on that copy
    /// stream (later than launch-completion, earlier than GEMM done).
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
            // Replica prefetch / D2D cannot start while a GEMM still leases the
            // page. 1-GPU sticky pin does not copy, so leftover prefill GEMMs
            // stay in flight for decode-priority ITL. Launch-completion waits
            // kernel start; programmatic-event waits the PDL trigger. Both let
            // pinned replica D2D overlap leftover GEMM (managed prefetch still
            // needs a full lease drain).
            if self.sim.profile().n_gpus() >= 2 {
                if self.mode == GpuFill::Pinned {
                    if let Some(ev) = self.replica_overlap_event() {
                        let device = self
                            .pages
                            .get(key)
                            .ok_or(Error::Store("missing handle"))?
                            .device;
                        let _w = self.sim.wait_event(device, ev, self.copy)?;
                    } else {
                        self.wait_page_idle(*key)?;
                    }
                } else {
                    self.wait_page_idle(*key)?;
                }
            }
            self.cache.pin_hot(&[*key])?;
            self.replicate(*key)?;
        }
        Ok(())
    }

    /// Move `key` onto `dst`. Pinned/VMM pages D2D unless [`GpuStoreCfg::accessed_by`]
    /// (retarget GEMM, keep home physicals / mempool); managed pages prefetch then drop the source copy unless AccessedBy (retarget GEMM, keep home residency); mapped pages retarget GEMM.
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
        // Host-sync managed/VMM drops, and stream-ordered pinned free, must not
        // race a live GEMM lease. Drain this page (and its replica) only — do
        // not `synchronize()` the whole device, or overlapping H2D of other
        // experts would collapse into the migrate.
        self.wait_page_idle(key)?;
        if let Some(g) = self.graphs.remove(&id) {
            self.sim.destroy_graph(g)?;
        }
        match self.mode {
            GpuFill::Managed => return self.migrate_managed(key, id, src, dst),
            GpuFill::Mapped => return self.migrate_mapped(key, dst),
            GpuFill::Vmm => return self.migrate_vmm(key, id, src, dst),
            GpuFill::Pinned if self.accessed_by && !self.sync_alloc => {
                return self.migrate_pool_peer(key, dst);
            }
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
        self.rehome_mailbox(key, dst, true)?;
        if let Some(page) = self.pages.get_mut(&key) {
            page.device = dst;
            if self.wait_value && !self.timing_events {
                page.ready = None;
            } else {
                page.ready = Some(ev_copy);
            }
        }
        self.note_migrate();
        Ok(())
    }

    /// GEMM retarget only — [`gpu_sim::Sim::pool_set_access`] already maps `dst`.
    fn migrate_pool_peer(&mut self, key: ExpertKey, dst: DeviceId) -> Result<(), Error> {
        let _gone = self.replicas.remove(&key);
        self.rehome_mailbox(key, dst, false)?;
        if let Some(page) = self.pages.get_mut(&key) {
            page.device = dst;
            page.ready = None;
        }
        self.note_migrate();
        Ok(())
    }

    /// GPU0↔GPU1 link bandwidth, or a 32 GB/s PCIe-order fallback.
    #[must_use]
    pub fn peer_bps(&self) -> u64 {
        self.sim
            .profile()
            .link(Some(DeviceId(0)), Some(DeviceId(1)))
            .map_or(32_000_000_000, |l| l.bps)
    }

    /// Payload bytes billed per expert page.
    #[must_use]
    pub fn expert_bytes(&self) -> u64 {
        self.bytes_per_expert
    }

    /// [`plan_placement`] on the GPU0↔GPU1 hop: D2D weights onto GPU0, or count a dispatch.
    ///
    /// 1-GPU profiles skip (`Ok`). [`Self::migrate`] stays unconditional. `reuse` is
    /// how many times this key has been selected for keep-hot so far (online).
    /// Fail-loud: a leased managed/VMM drop is [`Error::Sim`], not a swallowed
    /// no-op (Engine serving used to ignore that `Result`).
    pub fn place_hot(&mut self, key: ExpertKey, reuse: u64, fan_in: u64) -> Result<(), Error> {
        if self.sim.profile().n_gpus() < 2 {
            return Ok(());
        }
        match plan_placement(
            self.bytes_per_expert,
            DECODE_ACTIVATION_BYTES,
            fan_in,
            reuse,
            self.peer_bps(),
        ) {
            Placement::MoveWeights => self.migrate(key, DeviceId(0)),
            Placement::DispatchActivations => {
                self.dispatches = self.dispatches.saturating_add(1);
                Ok(())
            }
        }
    }

    fn note_migrate(&mut self) {
        self.migrates = self.migrates.saturating_add(1);
    }

    /// Prefetch `dst` (ReadMostly keeps extras) then drop the source copy.
    ///
    /// [`Self::accessed_by`]: GEMM retarget only — AccessedBy already maps `dst`.
    /// [`Self::wait_page_idle`] already drained this alloc's kernel leases.
    fn migrate_managed(
        &mut self,
        key: ExpertKey,
        id: AllocId,
        src: DeviceId,
        dst: DeviceId,
    ) -> Result<(), Error> {
        if self.accessed_by {
            let _gone = self.replicas.remove(&key);
            self.rehome_mailbox(key, dst, false)?;
            if let Some(page) = self.pages.get_mut(&key) {
                page.device = dst;
                page.ready = None;
            }
            self.note_migrate();
            return Ok(());
        }
        if !self.sim.is_resident(id, dst)? {
            let _p = self.sim.prefetch(dst, id, self.dma_stream())?;
            self.sim.synchronize_stream(dst, self.dma_stream())?;
        }
        if self.sim.is_resident(id, src)? {
            self.sim.drop_managed_copy(id, src)?;
        }
        let _gone = self.replicas.remove(&key);
        self.rehome_mailbox(key, dst, false)?;
        if let Some(page) = self.pages.get_mut(&key) {
            page.device = dst;
            page.ready = None;
        }
        self.note_migrate();
        Ok(())
    }

    /// Mapped host is already kernel-readable on every GPU; only the GEMM device changes.
    fn migrate_mapped(&mut self, key: ExpertKey, dst: DeviceId) -> Result<(), Error> {
        let _gone = self.replicas.remove(&key);
        self.rehome_mailbox(key, dst, false)?;
        if let Some(page) = self.pages.get_mut(&key) {
            page.device = dst;
            page.ready = None;
        }
        self.note_migrate();
        Ok(())
    }

    /// Map `dst`, D2D, then unmap the source physicals.
    ///
    /// [`Self::accessed_by`]: GEMM retarget only — `va_set_access` already maps `dst`.
    fn migrate_vmm(
        &mut self,
        key: ExpertKey,
        id: AllocId,
        src: DeviceId,
        dst: DeviceId,
    ) -> Result<(), Error> {
        if self.accessed_by {
            let _gone = self.replicas.remove(&key);
            self.rehome_mailbox(key, dst, false)?;
            if let Some(page) = self.pages.get_mut(&key) {
                page.device = dst;
                page.ready = None;
            }
            self.note_migrate();
            return Ok(());
        }
        self.sim.synchronize_stream(src, self.prefill)?;
        if self.decode != self.prefill {
            self.sim.synchronize_stream(src, self.decode)?;
        }
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
        self.rehome_mailbox(key, dst, false)?;
        if let Some(page) = self.pages.get_mut(&key) {
            page.device = dst;
            page.ready = None;
        }
        self.note_migrate();
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

    /// `cudaDeviceGetGraphMemAttribute` UsedMemCurrent on `device`.
    pub fn graph_mem_used(&self, device: DeviceId) -> Result<u64, Error> {
        Ok(self
            .sim
            .graph_mem_get(device, GraphMemAttr::UsedMemCurrent)?)
    }

    /// `cudaDeviceGetGraphMemAttribute` ReservedMemCurrent on `device`.
    pub fn graph_mem_reserved(&self, device: DeviceId) -> Result<u64, Error> {
        Ok(self
            .sim
            .graph_mem_get(device, GraphMemAttr::ReservedMemCurrent)?)
    }

    /// `cudaDeviceGraphMemTrim` on `device`.
    pub fn graph_mem_trim(&mut self, device: DeviceId) -> Result<(), Error> {
        Ok(self.sim.graph_mem_trim(device)?)
    }

    /// Whether the GPU page for `key` is resident on `device`.
    #[must_use]
    pub fn page_resident(&self, key: ExpertKey, device: DeviceId) -> bool {
        self.pages
            .get(&key)
            .and_then(|p| self.sim.is_resident(p.id, device).ok())
            .unwrap_or(false)
    }

    /// Whether `device` has SetAccessedBy / VMM `va_set_access` / mempool
    /// `pool_set_access` on `key`.
    #[must_use]
    pub fn page_accessed_by(&self, key: ExpertKey, device: DeviceId) -> bool {
        self.pages
            .get(&key)
            .and_then(|p| self.sim.is_accessed_by(p.id, device).ok())
            .unwrap_or(false)
    }

    /// True when any placed page is SetAccessedBy / VMM `va_set_access` /
    /// mempool `pool_set_access` on `device`.
    #[must_use]
    pub fn any_page_accessed_by(&self, device: DeviceId) -> bool {
        self.pages.keys().any(|k| self.page_accessed_by(*k, device))
    }

    /// [`gpu_sim::Sim::stream_priority`] for `(device, stream)`.
    #[must_use]
    pub fn stream_priority(&self, device: DeviceId, stream: StreamId) -> i32 {
        self.sim.stream_priority(device, stream)
    }

    /// [`gpu_sim::Sim::stream_mem_sync_domain`] for `(device, stream)`.
    #[must_use]
    pub fn stream_mem_sync_domain(
        &self,
        device: DeviceId,
        stream: StreamId,
    ) -> gpu_sim::MemSyncDomain {
        self.sim.stream_mem_sync_domain(device, stream)
    }

    /// [`gpu_sim::Sim::stream_mem_sync_domain_map`] for `(device, stream)`.
    pub fn stream_mem_sync_domain_map(
        &self,
        device: DeviceId,
        stream: StreamId,
    ) -> Result<gpu_sim::MemSyncDomainMap, Error> {
        Ok(self.sim.stream_mem_sync_domain_map(device, stream)?)
    }

    /// Event recorded at grouped-GEMM kernel start when [`GpuStoreCfg::launch_completion`].
    #[must_use]
    pub fn launch_completion_event(&self) -> Option<EventId> {
        self.launch_completion_event
    }

    /// Event recorded at the PDL trigger when [`GpuStoreCfg::programmatic_event`].
    #[must_use]
    pub fn programmatic_event(&self) -> Option<EventId> {
        self.programmatic_event
    }

    fn replica_overlap_event(&self) -> Option<EventId> {
        if self.launch_completion_armed {
            return self.launch_completion_event;
        }
        if self.programmatic_event_armed {
            return self.programmatic_event;
        }
        None
    }

    /// Submitted simulator ops (GEMM vs replica overlap).
    #[cfg(test)]
    pub(crate) fn operations(&self) -> impl Iterator<Item = gpu_sim::Operation> + '_ {
        self.sim.operations()
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

    /// How many times [`gpu_sim::Sim::graph_exec_kernel_set_params`] reused a parked exec.
    #[must_use]
    pub fn graph_set_params(&self) -> u64 {
        self.graph_set_params_n
    }

    /// Sum of [`gpu_sim::Sim::event_elapsed_ns`] on copy start/end events.
    ///
    /// Zero unless [`GpuStoreCfg::timing_events`] (implied by
    /// [`GpuStoreCfg::event_blocking_sync`]) and a copy was waited.
    #[must_use]
    pub fn copy_elapsed_ns(&self) -> u64 {
        self.copy_elapsed_ns
    }

    /// Whether `key` is in the fast CPU tier.
    #[must_use]
    pub fn is_resident(&self, key: ExpertKey) -> bool {
        self.cache.is_resident(key)
    }

    /// Fast-tier capacity (may be smaller than the constructor `slots` for mapped pages).
    #[must_use]
    pub fn slots(&self) -> usize {
        self.cache.slots()
    }

    /// GPUs in the attached hardware profile.
    #[must_use]
    pub fn n_gpus(&self) -> usize {
        self.sim.profile().n_gpus()
    }

    /// Sticky keep-hot budget: leave one slot for demand paging.
    #[must_use]
    pub fn pin_budget(&self) -> usize {
        self.cache.pin_budget()
    }

    /// Whether `key` has a sticky [`Self::pin_hot`] pin.
    #[must_use]
    pub fn is_pinned(&self, key: ExpertKey) -> bool {
        self.cache.is_pinned(key)
    }

    /// Drop every sticky pin. In-flight leases are unchanged.
    pub fn unpin_all(&mut self) {
        self.cache.unpin_all();
    }

    /// PLAN state: GPU copies are Transferring until the copy-stream event
    /// completes, or until the wait-value mailbox generation is consumed.
    #[must_use]
    pub fn phase(&self, key: ExpertKey) -> ExpertPhase {
        if let Some(page) = self.evicting.get(&key) {
            if self.sim.is_resident(page.id, page.device).unwrap_or(false) {
                return ExpertPhase::Evicting;
            }
            return ExpertPhase::Cold;
        }
        if let Some(page) = self.pages.get(&key) {
            if page.ready_gen.is_some() {
                return ExpertPhase::Transferring;
            }
            if let Some(ev) = page.ready {
                if !self.sim.query_event(ev).unwrap_or(false) {
                    return ExpertPhase::Transferring;
                }
            }
        }
        ExpertPhase::cpu(self.cache.is_resident(key), self.cache.is_held(key))
    }

    /// Drop `key` from HBM. Illegal while leased. Stays [`ExpertPhase::Evicting`]
    /// until the stream-ordered free completes.
    pub fn evict(&mut self, key: ExpertKey) -> Result<(), Error> {
        self.sweep_evicts();
        self.cache.evict(key)?;
        self.drop_gpu(key)
    }

    /// Drain GEMM and copy on this page's device (and replica), not the whole sim.
    fn wait_page_idle(&mut self, key: ExpertKey) -> Result<(), Error> {
        let (device, replica) = {
            let Some(page) = self.pages.get(&key) else {
                return Ok(());
            };
            (page.device, self.replicas.get(&key).copied())
        };
        self.wait_compute(device)?;
        self.sim.synchronize_stream(device, self.copy)?;
        if let Some(dst) = replica {
            if dst != device {
                self.wait_compute(dst)?;
                self.sim.synchronize_stream(dst, self.copy)?;
            }
        }
        Ok(())
    }

    fn wait_compute(&mut self, device: DeviceId) -> Result<(), Error> {
        self.sim.synchronize_stream(device, self.prefill)?;
        if self.decode != self.prefill {
            self.sim.synchronize_stream(device, self.decode)?;
        }
        Ok(())
    }

    fn wait_copy(&mut self, key: ExpertKey) -> Result<(), Error> {
        let (device, ready, start, mailbox, ready_gen) = {
            let page = self.pages.get(&key).ok_or(Error::Store("missing handle"))?;
            (
                page.device,
                page.ready,
                page.start,
                page.mailbox,
                page.ready_gen,
            )
        };
        if let (Some(mb), Some(gen)) = (mailbox, ready_gen) {
            // Queue on the DMA stream so replica copy is stream-ordered after
            // this page's write. Do not consume `ready_gen`: GEMM still waits
            // on compute. Do not synchronize compute (leftover prefill) when
            // DMA is the copy stream.
            let dma = self.dma_stream();
            wait_copy_ready(&mut self.sim, device, mb, gen, dma)?;
            if let Some(ev) = ready {
                if self.timing_events {
                    if !self.sim.query_event(ev)? {
                        self.sim.synchronize_stream(device, dma)?;
                    }
                    self.note_copy_elapsed(start, ev)?;
                    if let Some(page) = self.pages.get_mut(&key) {
                        page.ready = None;
                        page.start = None;
                    }
                }
            }
            return Ok(());
        }
        if let Some(ev) = ready {
            if !self.sim.query_event(ev)? {
                // Wait the DMA stream only. Syncing compute here would drain
                // leftover prefill GEMMs before decode-priority can overlap them
                // (identity copy-stream prefetch). `--stream-attach` DMA is
                // already on compute.
                let dma = self.dma_stream();
                self.sim.synchronize_stream(device, dma)?;
            }
            self.note_copy_elapsed(start, ev)?;
            if let Some(page) = self.pages.get_mut(&key) {
                page.ready = None;
                page.start = None;
            }
        }
        Ok(())
    }

    fn alloc_resident_mailbox(&mut self, d: DeviceId) -> Result<Option<AllocId>, Error> {
        if !self.wait_value {
            return Ok(None);
        }
        let dma = self.dma_stream();
        Ok(Some(alloc_resident_copy_mailbox(
            &mut self.sim,
            d,
            dma,
            self.sync_alloc,
        )?))
    }

    fn insert_page(
        &mut self,
        key: ExpertKey,
        id: AllocId,
        d: DeviceId,
        start: Option<EventId>,
        mailbox: Option<AllocId>,
    ) -> Result<(), Error> {
        let dma = self.dma_stream();
        let (ready, mailbox, ready_gen) = if self.wait_value {
            let mb = mailbox.ok_or(Error::Store("missing mailbox"))?;
            signal_copy_ready(&mut self.sim, d, mb, 1, dma)?;
            let ev = if self.timing_events {
                let ev = self.create_copy_event()?;
                let _r = self.sim.record_event(d, ev, dma)?;
                Some(ev)
            } else {
                None
            };
            (ev, Some(mb), Some(1))
        } else {
            let ev = self.create_copy_event()?;
            let _r = self.sim.record_event(d, ev, dma)?;
            (Some(ev), None, None)
        };
        let _prev = self.pages.insert(
            key,
            GpuPage {
                id,
                device: d,
                ready,
                start,
                mailbox,
                ready_gen,
            },
        );
        Ok(())
    }

    fn free_page_mailbox(
        &mut self,
        device: DeviceId,
        mailbox: Option<AllocId>,
    ) -> Result<(), Error> {
        let Some(mb) = mailbox else {
            return Ok(());
        };
        let dma = self.dma_stream();
        free_copy_mailbox(
            &mut self.sim,
            device,
            mb,
            dma,
            self.sync_alloc || self.mode != GpuFill::Pinned,
        )
    }

    fn rehome_mailbox(&mut self, key: ExpertKey, dst: DeviceId, signal: bool) -> Result<(), Error> {
        let (old_dev, old_mb, old_gen) = {
            let page = self.pages.get(&key).ok_or(Error::Store("missing handle"))?;
            (page.device, page.mailbox, page.ready_gen)
        };
        self.free_page_mailbox(old_dev, old_mb)?;
        if !self.wait_value {
            if let Some(page) = self.pages.get_mut(&key) {
                page.mailbox = None;
                page.ready_gen = None;
            }
            return Ok(());
        }
        let dma = self.dma_stream();
        let mb = alloc_resident_copy_mailbox(&mut self.sim, dst, dma, self.sync_alloc)?;
        let gen = if signal {
            let g = old_gen.unwrap_or(0).saturating_add(1).max(1);
            signal_copy_ready(&mut self.sim, dst, mb, g, dma)?;
            Some(g)
        } else {
            None
        };
        if let Some(page) = self.pages.get_mut(&key) {
            page.mailbox = Some(mb);
            page.ready_gen = gen;
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
        if let Some(page) = self.host_pages.remove(&key) {
            return self.restore_host_page(key, page);
        }
        let d = self.home(key);
        let mailbox = self.alloc_resident_mailbox(d)?;
        let start = self.record_copy_start(d)?;
        let id = self.fill_page(d)?;
        self.insert_page(key, id, d, start, mailbox)
    }

    fn place_pending(&mut self, keys: &[ExpertKey]) -> Result<(), Error> {
        let mut by_dev: BTreeMap<DeviceId, Vec<ExpertKey>> = BTreeMap::new();
        for key in keys {
            if self.cache.is_resident(*key) && !self.pages.contains_key(key) {
                by_dev.entry(self.home(*key)).or_default().push(*key);
            }
        }
        for (d, group) in by_dev {
            self.place_group(d, &group)?;
        }
        Ok(())
    }

    fn place_group(&mut self, d: DeviceId, keys: &[ExpertKey]) -> Result<(), Error> {
        if keys.len() < 2 {
            for key in keys {
                self.place(*key)?;
            }
            return Ok(());
        }
        let mut pending: Vec<(ExpertKey, AllocId, Option<EventId>, Option<AllocId>)> = Vec::new();
        for key in keys {
            if self.pages.contains_key(key) {
                continue;
            }
            if let Some(v) = self.cache.take_victim() {
                self.drop_gpu(v)?;
            }
            let mailbox = self.alloc_resident_mailbox(d)?;
            let start = self.record_copy_start(d)?;
            let id = self.alloc_page_no_fill(d)?;
            pending.push((*key, id, start, mailbox));
        }
        if pending.len() < 2 {
            for (key, id, start, mailbox) in pending {
                self.fill_hbm(d, id)?;
                if self.mode == GpuFill::Vmm && self.accessed_by {
                    advise_vmm_access(&mut self.sim, id)?;
                }
                self.insert_page(key, id, d, start, mailbox)?;
            }
            return Ok(());
        }
        let bytes = self.bytes_per_expert;
        let ops: Vec<MemcpyOp> = pending
            .iter()
            .map(|(_, id, _, _)| {
                MemcpyOp::packed_1d(Place::HostPinned, Place::Device(d), *id, bytes)
            })
            .collect();
        wait_memcpy_during_allocs(
            &mut self.sim,
            d,
            self.copy,
            self.memcpy_during || self.memcpy_any,
        )?;
        let attr = memcpy_batch_attr(self.memcpy_during, self.memcpy_any);
        let _ids =
            self.sim
                .memcpy_batch_async(d, &ops, std::slice::from_ref(&attr), &[0], self.copy)?;
        for (key, id, start, mailbox) in pending {
            if self.mode == GpuFill::Vmm && self.accessed_by {
                advise_vmm_access(&mut self.sim, id)?;
            }
            self.insert_page(key, id, d, start, mailbox)?;
        }
        Ok(())
    }

    fn alloc_page_no_fill(&mut self, d: DeviceId) -> Result<AllocId, Error> {
        match self.mode {
            GpuFill::Vmm => Ok(self.vmm_alloc(d)?),
            GpuFill::Pinned => self.hbm_alloc(d),
            GpuFill::Mapped | GpuFill::Managed => {
                Err(Error::Store("memcpy-batch needs async pinned/vmm H2D"))
            }
        }
    }

    fn create_copy_event(&mut self) -> Result<EventId, Error> {
        let ev = EventId(self.next_event);
        self.next_event = self.next_event.saturating_add(1);
        if self.event_blocking_sync {
            self.sim.create_event_blocking_sync(ev)?;
        } else if self.timing_events {
            self.sim.create_event(ev)?;
        } else {
            self.sim.create_event_disable_timing(ev)?;
        }
        Ok(ev)
    }

    fn record_copy_start(&mut self, d: DeviceId) -> Result<Option<EventId>, Error> {
        if !self.timing_events || self.mode == GpuFill::Mapped {
            return Ok(None);
        }
        let ev = self.create_copy_event()?;
        let _r = self.sim.record_event(d, ev, self.dma_stream())?;
        Ok(Some(ev))
    }

    fn note_copy_elapsed(&mut self, start: Option<EventId>, end: EventId) -> Result<(), Error> {
        if !self.timing_events {
            return Ok(());
        }
        let Some(st) = start else {
            return Ok(());
        };
        self.sim.synchronize_event(st)?;
        self.sim.synchronize_event(end)?;
        let ns = self.sim.event_elapsed_ns(st, end)?;
        self.copy_elapsed_ns = self.copy_elapsed_ns.saturating_add(ns);
        Ok(())
    }

    fn fill_page(&mut self, d: DeviceId) -> Result<AllocId, Error> {
        let bytes = self.bytes_per_expert;
        match self.mode {
            GpuFill::Managed => {
                let id = if self.managed_host {
                    self.sim.alloc_managed_host(bytes)?
                } else {
                    self.sim.alloc_managed(bytes)?
                };
                self.sim.mem_advise(id, MemAdvise::SetReadMostly, d)?;
                self.sim
                    .mem_advise(id, MemAdvise::SetPreferredLocation, d)?;
                if self.accessed_by {
                    advise_accessed_by(&mut self.sim, id)?;
                }
                let dma = self.dma_stream();
                if self.stream_attach {
                    let _a = self.sim.stream_attach(d, id, dma, MemAttach::Single)?;
                } else if self.managed_host {
                    let _a = self.sim.stream_attach(d, id, dma, MemAttach::Global)?;
                }
                mark_sync_memops(&mut self.sim, id, d, dma, self.sync_memops)?;
                let _p = self.sim.prefetch(d, id, dma)?;
                if self.sync_alloc {
                    self.sim.synchronize_stream(d, dma)?;
                }
                Ok(id)
            }
            GpuFill::Mapped => {
                if self.host_register_mapped {
                    let h = self.sim.alloc_host(bytes)?;
                    self.sim.host_register_mapped(h)?;
                    Ok(h)
                } else {
                    Ok(self.sim.alloc_host_mapped(bytes)?)
                }
            }
            GpuFill::Vmm => {
                let id = self.vmm_alloc(d)?;
                mark_sync_memops(&mut self.sim, id, d, self.copy, self.sync_memops)?;
                self.fill_hbm(d, id)?;
                if self.accessed_by {
                    advise_vmm_access(&mut self.sim, id)?;
                }
                Ok(id)
            }
            GpuFill::Pinned => {
                let id = self.hbm_alloc(d)?;
                mark_sync_memops(&mut self.sim, id, d, self.copy, self.sync_memops)?;
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
        if self.pageable && !self.host_register {
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
                    ..MemcpyOp::default()
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

    /// Miss DMA stream: compute when [`Self::stream_attach`], else copy.
    fn dma_stream(&self) -> StreamId {
        if self.stream_attach {
            self.compute
        } else {
            self.copy
        }
    }

    fn gemm_resident(&mut self, key: ExpertKey) -> Result<(), Error> {
        let (id, device, ready, start, mailbox, ready_gen) = {
            let page = self
                .pages
                .get_mut(&key)
                .ok_or(Error::Store("missing handle"))?;
            let ready = page.ready.take();
            let start = page.start.take();
            let ready_gen = page.ready_gen.take();
            (page.id, page.device, ready, start, page.mailbox, ready_gen)
        };
        if let (Some(mb), Some(gen)) = (mailbox, ready_gen) {
            wait_copy_ready(&mut self.sim, device, mb, gen, self.compute)?;
            if let Some(ev) = ready {
                self.note_copy_elapsed(start, ev)?;
            }
        } else if let Some(ev) = ready {
            if !self.sim.query_event(ev)? {
                let _w = self.sim.wait_event(device, ev, self.compute)?;
            }
            self.note_copy_elapsed(start, ev)?;
        }
        ensure_single_attach(&mut self.sim, device, id, self.compute)?;
        self.launch_or_gemm(device, id)?;
        reset_persisting_l2_if(&mut self.sim, device, self.l2_reset)?;
        if self.host_func {
            let _id = self.sim.host_func(device, self.compute)?;
        }
        Ok(())
    }

    fn launch_graph_exec(&mut self, g: GraphId, device: DeviceId) -> Result<(), Error> {
        let flags = self.gemm_flags();
        let launch = flags.mem_sync_domain();
        let launch_map = flags.mem_sync_map();
        apply_exec_mem_sync_domain(&mut self.sim, g, device, self.compute, launch)?;
        apply_exec_mem_sync_map(&mut self.sim, g, device, self.compute, launch_map)?;
        self.graph_launches = self.graph_launches.saturating_add(1);
        replay_exec(&mut self.sim, g, device, self.compute, self.device_launch)
    }

    fn launch_or_gemm(&mut self, device: DeviceId, id: AllocId) -> Result<(), Error> {
        if self.launch_completion_event.is_some() {
            self.launch_completion_armed = true;
        }
        if self.programmatic_event.is_some() {
            self.programmatic_event_armed = true;
        }
        let flags = self.gemm_flags();
        if let Some(g) = self.graphs.get(&id).copied() {
            return self.launch_graph_exec(g, device);
        }
        if self.graph_set_params {
            if let Some(exec) = self.idle_execs.get_mut(&device).and_then(Vec::pop) {
                retarget_parked_kernel(&mut self.sim, exec, id)?;
                self.graph_set_params_n = self.graph_set_params_n.saturating_add(1);
                upload_after_set_params(&mut self.sim, exec, self.device_updatable)?;
                let _prev = self.graphs.insert(id, exec);
                return self.launch_graph_exec(exec, device);
            }
        }
        if self.graph_build {
            let src = self.build_gemm_graph(device, id)?;
            let g = self.bind_graph(device, src)?;
            let _prev = self.graphs.insert(id, g);
            return self.launch_graph_exec(g, device);
        }
        if !self.sim.query_stream(device, self.compute)? {
            self.sim.synchronize_stream(device, self.compute)?;
        }
        if self.sim.query_stream(device, self.compute)? {
            if self.graph_piecewise {
                let src = self.piecewise_gemm_graph(device, id)?;
                let g = self.bind_graph(device, src)?;
                let _prev = self.graphs.insert(id, g);
                return self.launch_graph_exec(g, device);
            }
            self.sim.begin_capture(device, self.compute)?;
            kernel_leaf(&mut self.sim, device, self.compute, id, self.leaf, flags)?;
            let src = self.sim.end_capture()?;
            let g = self.bind_graph(device, src)?;
            let _prev = self.graphs.insert(id, g);
            return self.launch_graph_exec(g, device);
        }
        kernel_leaf(
            &mut self.sim,
            device,
            self.compute,
            id,
            LeafMem::None,
            flags.for_stream(),
        )
    }

    fn build_gemm_graph(&mut self, device: DeviceId, id: AllocId) -> Result<GraphId, Error> {
        let flags = self.gemm_flags();
        let g = self.sim.create_graph(device, self.compute)?;
        add_leaf_gemm(&mut self.sim, g, id, self.leaf, flags)?;
        Ok(g)
    }

    fn piecewise_gemm_graph(&mut self, device: DeviceId, id: AllocId) -> Result<GraphId, Error> {
        let flags = self.gemm_flags();
        let g = self.sim.create_graph(device, self.compute)?;
        self.sim
            .begin_capture_to_graph(device, self.compute, g, &[])?;
        kernel_leaf(&mut self.sim, device, self.compute, id, self.leaf, flags)?;
        let ended = self.sim.end_capture()?;
        if ended != g {
            return Err(Error::Store("capture-to-graph id"));
        }
        Ok(g)
    }

    fn bind_graph(&mut self, device: DeviceId, src: GraphId) -> Result<GraphId, Error> {
        if self.graph_update && self.leaf == LeafMem::None {
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
        instantiate_exec(
            &mut self.sim,
            exec,
            self.leaf == LeafMem::AutoFree,
            self.device_launch,
            self.kernel_priority.is_some(),
        )
    }

    fn drop_gpu(&mut self, key: ExpertKey) -> Result<(), Error> {
        if self.prefetch_host && self.mode == GpuFill::Managed {
            return self.evict_managed_to_host(key);
        }
        let Some(page) = self.pages.remove(&key) else {
            return Ok(());
        };
        let _prev = self.evicting.insert(key, page);
        self.finish_drop(key, page)
    }

    /// `cudaMemPrefetchAsync(..., cudaCpuDeviceId)` then keep the managed alloc.
    fn evict_managed_to_host(&mut self, key: ExpertKey) -> Result<(), Error> {
        let Some(page) = self.pages.remove(&key) else {
            return Ok(());
        };
        self.wait_compute(page.device)?;
        let dma = self.dma_stream();
        self.sim.synchronize_stream(page.device, dma)?;
        if let Some(dst) = self.replicas.remove(&key) {
            if dst != page.device {
                self.sim.synchronize_stream(dst, dma)?;
            }
        }
        let _p = self.sim.prefetch_host(page.device, page.id, dma)?;
        self.sim.synchronize_stream(page.device, dma)?;
        self.free_page_mailbox(page.device, page.mailbox)?;
        let _prev = self.host_pages.insert(
            key,
            GpuPage {
                mailbox: None,
                ready: None,
                start: None,
                ready_gen: None,
                ..page
            },
        );
        Ok(())
    }

    /// Prefetch a host-resident managed page back onto its home GPU.
    fn restore_host_page(&mut self, key: ExpertKey, page: GpuPage) -> Result<(), Error> {
        let d = page.device;
        let mailbox = self.alloc_resident_mailbox(d)?;
        let start = self.record_copy_start(d)?;
        let dma = self.dma_stream();
        if self.stream_attach {
            ensure_single_attach(&mut self.sim, d, page.id, dma)?;
        }
        let _p = self.sim.prefetch(d, page.id, dma)?;
        if self.sync_alloc {
            self.sim.synchronize_stream(d, dma)?;
        }
        self.insert_page(key, page.id, d, start, mailbox)
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
            if self.graph_set_params || (self.graph_update && self.leaf == LeafMem::None) {
                self.idle_execs.entry(page.device).or_default().push(g);
            } else {
                self.sim.destroy_graph(g)?;
            }
        }
        match self.mode {
            GpuFill::Managed => {
                self.wait_compute(page.device)?;
                self.sim.synchronize_stream(page.device, self.copy)?;
                if let Some(dst) = self.replicas.remove(&key) {
                    self.sim.synchronize_stream(dst, self.copy)?;
                }
                self.free_page_mailbox(page.device, page.mailbox)?;
                self.sim.free_sync(page.id)?;
                return Ok(());
            }
            GpuFill::Mapped => {
                self.wait_compute(page.device)?;
                self.free_page_mailbox(page.device, page.mailbox)?;
                free_mapped_host(&mut self.sim, page.id)?;
                let _gone = self.replicas.remove(&key);
                return Ok(());
            }
            GpuFill::Vmm => {
                self.wait_compute(page.device)?;
                self.free_page_mailbox(page.device, page.mailbox)?;
                self.sim.va_release(page.id)?;
                let _gone = self.replicas.remove(&key);
                return Ok(());
            }
            GpuFill::Pinned if self.sync_alloc => {
                self.wait_compute(page.device)?;
                self.free_page_mailbox(page.device, page.mailbox)?;
                self.sim.free_sync(page.id)?;
                let _gone = self.replicas.remove(&key);
                return Ok(());
            }
            GpuFill::Pinned => {
                self.wait_compute(page.device)?;
            }
        }
        // Copy-engine free must not race a compute-stream lease on the same page.
        let ev = EventId(self.next_event);
        self.next_event = self.next_event.saturating_add(1);
        self.sim.create_event_disable_timing(ev)?;
        let _r = self.sim.record_event(page.device, ev, self.compute)?;
        let _w = self.sim.wait_event(page.device, ev, self.copy)?;
        if let Some(dst) = self.replicas.remove(&key) {
            if self.sim.is_resident(page.id, dst)? {
                self.sim.free(dst, page.id, self.copy)?;
            }
        }
        self.free_page_mailbox(page.device, page.mailbox)?;
        self.sim.free(page.device, page.id, self.copy)?;
        Ok(())
    }

    fn replicate(&mut self, key: ExpertKey) -> Result<(), Error> {
        if self.sim.profile().n_gpus() < 2 {
            return Ok(());
        }
        if self.replicas.contains_key(&key) {
            return Ok(());
        }
        let (id, src) = {
            let page = self.pages.get(&key).ok_or(Error::Store("missing handle"))?;
            (page.id, page.device)
        };
        let Some(dst) = self.replica_dst(src) else {
            return Ok(());
        };
        if src == dst {
            let _prev = self.replicas.insert(key, dst);
            return Ok(());
        }
        match self.mode {
            GpuFill::Managed => {
                if !self.accessed_by {
                    let _p = self.sim.prefetch(dst, id, self.dma_stream())?;
                    self.note_replicate();
                }
                let _prev = self.replicas.insert(key, dst);
                return Ok(());
            }
            GpuFill::Mapped => {
                let _prev = self.replicas.insert(key, dst);
                return Ok(());
            }
            GpuFill::Vmm => {
                if self.accessed_by {
                    if !self.sim.is_accessed_by(id, dst)? {
                        self.sim.va_set_access(id, dst)?;
                    }
                } else if self.multicast {
                    if !self.sim.is_resident(id, dst)? {
                        self.sim.va_map(id, dst)?;
                    }
                    let _c = self.sim.multicast_store(src, id, &[dst], self.copy)?;
                    self.note_replicate();
                } else if !self.sim.is_resident(id, dst)? {
                    self.sim.va_map(id, dst)?;
                    let _c = self.sim.memcpy_device_to_device(
                        src,
                        dst,
                        id,
                        self.bytes_per_expert,
                        self.copy,
                    )?;
                    self.note_replicate();
                }
                let _prev = self.replicas.insert(key, dst);
                return Ok(());
            }
            GpuFill::Pinned => {
                if self.accessed_by && !self.sync_alloc {
                    let _prev = self.replicas.insert(key, dst);
                    return Ok(());
                }
            }
        }
        let _c =
            self.sim
                .memcpy_device_to_device(src, dst, id, self.bytes_per_expert, self.copy)?;
        self.note_replicate();
        let _prev = self.replicas.insert(key, dst);
        Ok(())
    }

    /// Next GPU after `src` on the profile mesh (`None` when `n_gpus < 2`).
    #[must_use]
    fn replica_dst(&self, src: DeviceId) -> Option<DeviceId> {
        let n = u16::try_from(self.sim.profile().n_gpus()).unwrap_or(1);
        if n < 2 {
            return None;
        }
        Some(DeviceId(src.0.wrapping_add(1) % n))
    }

    fn note_replicate(&mut self) {
        self.replicates = self.replicates.saturating_add(1);
    }

    /// Peer GPU that holds a [`Self::pin_hot`] replica of `key`, if any.
    #[must_use]
    pub fn replica_of(&self, key: ExpertKey) -> Option<DeviceId> {
        self.replicas.get(&key).copied()
    }

    /// Reserve a VMM VA for Engine interned KV (`pool_blocks` pages).
    ///
    /// No-op unless [`GpuStoreCfg::kv_sim`]. Idempotent after the first bind.
    /// Faults `cuMemCreate` a physical and `cuMemMap` it; drop releases the handle.
    pub fn bind_kv(&mut self, n_pages: u32, page_bytes: u64) -> Result<(), Error> {
        if !self.kv_sim {
            return Ok(());
        }
        if self.kv.is_some() {
            return Ok(());
        }
        if n_pages == 0 || page_bytes == 0 {
            return Err(Error::Store("kv sim pages"));
        }
        let va_bytes = u64::from(n_pages).saturating_mul(page_bytes);
        if va_bytes == 0 {
            return Err(Error::Store("kv sim pages"));
        }
        let va = self.sim.va_reserve(va_bytes)?;
        self.kv = Some(KvGpu {
            va,
            page_bytes,
            n_pages,
            mapped: BTreeSet::new(),
            handles: BTreeMap::new(),
            hits: 0,
            misses: 0,
        });
        Ok(())
    }

    /// Replay Engine intern/alloc events onto the reserved KV VA.
    pub fn apply_kv_ops(&mut self, ops: &[KvSimOp]) -> Result<(), Error> {
        if self.kv.is_none() {
            return Ok(());
        }
        for op in ops {
            match *op {
                KvSimOp::Fault(id) => self.kv_fault(id)?,
                KvSimOp::Hit(id) => self.kv_hit(id)?,
                KvSimOp::Cow { dst, .. } => self.kv_fault(dst)?,
                KvSimOp::Drop(id) => self.kv_drop(id)?,
            }
        }
        Ok(())
    }

    /// Intern-hit kernels billed on the KV VA.
    #[must_use]
    pub fn kv_hits(&self) -> u64 {
        self.kv.as_ref().map_or(0, |k| k.hits)
    }

    /// Map+memset fills billed on the KV VA.
    #[must_use]
    pub fn kv_misses(&self) -> u64 {
        self.kv.as_ref().map_or(0, |k| k.misses)
    }

    fn kv_off(&self, id: u32) -> Result<(AllocId, u64, u64), Error> {
        let kv = self.kv.as_ref().ok_or(Error::Store("kv sim unbound"))?;
        if id >= kv.n_pages {
            return Err(Error::Store("kv page id"));
        }
        Ok((
            kv.va,
            u64::from(id).saturating_mul(kv.page_bytes),
            kv.page_bytes,
        ))
    }

    fn kv_map(&mut self, id: u32) -> Result<(AllocId, u64, u64), Error> {
        let (va, off, bytes) = self.kv_off(id)?;
        let already = self.kv.as_ref().is_some_and(|k| k.mapped.contains(&id));
        if !already {
            self.sim.synchronize_stream(self.device, self.compute)?;
            let existing = self.kv.as_ref().and_then(|k| k.handles.get(&id).copied());
            if let Some(h) = existing {
                self.sim.va_map_handle(va, self.device, off, h)?;
            } else {
                let h = self.sim.va_create(self.device, bytes)?;
                if let Some(k) = self.kv.as_mut() {
                    let _prev = k.handles.insert(id, h);
                }
                self.sim.va_map_handle(va, self.device, off, h)?;
            }
            if let Some(k) = self.kv.as_mut() {
                let _ins = k.mapped.insert(id);
            }
        }
        Ok((va, off, bytes))
    }

    fn kv_fault(&mut self, id: u32) -> Result<(), Error> {
        let (va, off, bytes) = self.kv_map(id)?;
        let buf = KernelBuf::span(va, off, bytes);
        let _op = self.sim.memset_buf(self.device, buf, self.compute)?;
        if let Some(k) = self.kv.as_mut() {
            k.misses = k.misses.saturating_add(1);
        }
        Ok(())
    }

    fn kv_hit(&mut self, id: u32) -> Result<(), Error> {
        let (va, off, bytes) = self.kv_map(id)?;
        let buf = KernelBuf::span(va, off, bytes);
        let _op = self.sim.kernel_bufs(
            self.device,
            KernelKind::other(8, bytes),
            &[buf],
            &[buf],
            self.compute,
        )?;
        if let Some(k) = self.kv.as_mut() {
            k.hits = k.hits.saturating_add(1);
        }
        Ok(())
    }

    fn kv_drop(&mut self, id: u32) -> Result<(), Error> {
        let mapped = self.kv.as_ref().is_some_and(|k| k.mapped.contains(&id));
        if !mapped {
            return Ok(());
        }
        let (va, off, bytes) = self.kv_off(id)?;
        self.sim.va_unmap_range(va, self.device, off, bytes)?;
        let h = self.kv.as_ref().and_then(|k| k.handles.get(&id).copied());
        if let Some(h) = h {
            if self.sim.handle_maps(h)? == 0 {
                self.sim.va_release_handle(h)?;
            }
        }
        if let Some(k) = self.kv.as_mut() {
            let _gone = k.mapped.remove(&id);
            let _h = k.handles.remove(&id);
        }
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
        m.migrates = self.migrates;
        m.dispatches = self.dispatches;
        m.replicates = self.replicates;
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
    /// [`SimulatedGpuStore::graph_set_params`] after the walk.
    pub graph_set_params: u64,
    /// [`SimulatedGpuStore::copy_elapsed_ns`] after the walk.
    pub copy_elapsed_ns: u64,
}

impl StoreReplay {
    /// Single-line agent / CLI log.
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "store {} {} graph_updates={} graph_clones={} graph_set_params={} copy_elapsed_ns={}",
            self.metrics.line(),
            self.score.line(),
            self.graph_updates,
            self.graph_clones,
            self.graph_set_params,
            self.copy_elapsed_ns
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
    let mut chain = ChainState::new();
    for (i, event) in trace.events.iter().enumerate() {
        store.bind_sequence(event.sequence);
        store.bind_decode_compute(event.token > 0);
        let ek = event.keys();
        for key in &ek {
            let _p = store.acquire(*key)?;
            store.release(*key);
        }
        if store_should_prefetch(&store, &catalog, &run, trace, i) {
            let predicted = predicted_keys(run.prefetch, &markov, chain.predecessor(event), &ek);
            let planned = if run.plan_window > 0 {
                window_keys(trace, i.saturating_add(1), run.plan_window)
            } else {
                Vec::new()
            };
            let fill: Vec<ExpertKey> = predicted.into_iter().chain(planned).collect();
            let _n = store.prefetch(&fill)?;
        }
        chain.observe(&mut markov, event);
    }
    let n = u64::try_from(trace.events.len()).unwrap_or(1);
    Ok(StoreReplay {
        metrics: store.metrics(),
        graph_updates: store.graph_updates(),
        graph_clones: store.graph_clones(),
        graph_set_params: store.graph_set_params(),
        copy_elapsed_ns: store.copy_elapsed_ns(),
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

/// Copy is NULL; prefill compute is stream 1. Seq-streams: copy `0 .. n_copy-1`,
/// prefill `StreamId(n_copy)`. Decode-priority adds `n_copy+1` at higher id.
fn copy_compute_streams(
    profile: &HardwareProfile,
    seq_streams: bool,
    decode_priority: bool,
) -> (StreamId, StreamId, StreamId, u8) {
    let (copy, prefill, mut mark) = if seq_streams {
        let n_copy = replay_streams(profile, true);
        let compute = StreamId(u16::from(n_copy));
        let mark = u8::try_from(u16::from(n_copy).saturating_add(1)).unwrap_or(u8::MAX);
        (StreamId(0), compute, mark)
    } else {
        (StreamId(0), StreamId(1), 2)
    };
    let decode = if decode_priority {
        mark = mark.saturating_add(1);
        StreamId(prefill.0.saturating_add(1))
    } else {
        prefill
    };
    (copy, prefill, decode, mark)
}

/// `cudaMemAdviseSetAccessedBy` on every GPU so a remote read does not migrate.
fn advise_accessed_by(sim: &mut Sim, id: AllocId) -> Result<(), Error> {
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    for g in 0..n {
        sim.mem_advise(id, MemAdvise::SetAccessedBy, DeviceId(g))?;
    }
    Ok(())
}

/// `cuMemSetAccess` PROT_READ on every GPU so a remote VMM read skips dest HBM.
fn advise_vmm_access(sim: &mut Sim, id: AllocId) -> Result<(), Error> {
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    for g in 0..n {
        sim.va_set_access(id, DeviceId(g))?;
    }
    Ok(())
}

/// `cudaMemPoolSetAccess` ReadWrite on every current device mempool.
fn advise_pool_access(sim: &mut Sim) -> Result<(), Error> {
    let n = u16::try_from(sim.profile().n_gpus()).unwrap_or(1);
    for g in 0..n {
        let home = DeviceId(g);
        let pool = sim.device_mempool(home)?;
        for d in 0..n {
            sim.pool_set_access(pool, DeviceId(d))?;
        }
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
