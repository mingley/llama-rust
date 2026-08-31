//! Expert virtual memory: traces, policies, stores, GPU-sim replay,
//! open-loop continuous batching (`schedule_replay` / `schedule_placed` /
//! `schedule_remote`), optional prefix cache on content-addressed `"p"`,
//! paged VMM KV (`kv_replay` / `kv_paged`, interned `cuMemCreate` handles),
//! [`store_replay`] / `expertvm store`.

#![deny(missing_docs, unsafe_code)]

mod access;
mod analyze;
mod bench;
mod error;
mod gpu_store;
mod kv;
mod live;
mod place;
mod planner;
mod policy;
mod replay;
mod schedule;
mod sim_replay;
mod store;
mod tiered;
mod workload;

pub use access::{prefix_hash, weight_permille, ExpertAccess, ExpertKey, Trace};
pub use analyze::{analyze, coactivation_counts, freq_table, mass_table, TraceStats};
pub use bench::{adversarial_suite, report, topology_suite, BenchReport};
pub use error::Error;
pub use gpu_sim::{
    probe_topology, DeviceId, GpuOp, HardwareProfile, MemSyncDomain, MemSyncDomainMap, Operation,
    PoolId, PortableClusterMode, PortableSharedMode, Score, SharedMemoryMode, StreamId,
    SynchronizationPolicy, TopologyProbe,
};
pub use gpu_store::{
    store_replay, store_replay_cfg, GpuFill, GpuStoreCfg, SimulatedGpuStore, StoreReplay,
    StoreReplayCfg,
};
pub use kv::{cycling_pages, kv_paged, kv_replay, KvCfg, KvFill, KvReplay, KvSimOp};
pub use live::LiveStore;
pub use place::{colocated, home_gpu, striped, with_hot_replicas, PlaceMap};
pub use planner::{
    copy_forward, hot_keys, observe_chain, plan_keys, plan_placement, plan_window, predicted_keys,
    prefetch_keys, prefetch_keys_ctx, transition_pair, window_keys, ChainState, Markov, Placement,
    Plan, Prefetch, DECODE_ACTIVATION_BYTES,
};
pub use policy::Policy;
pub use replay::{compare, format_table, replay, ReplayRow};
pub use schedule::{schedule_placed, schedule_remote, schedule_replay, SchedCfg, SchedReplay};
pub use sim_replay::{
    compare_ep, sim_placed, sim_remote_home, sim_remote_home_cfg, sim_replay, sim_replay_cfg,
    sim_static_ep, EpCompare, SimCfg, SimReplay,
};
pub use store::{
    replay_accesses, CachedStore, DirectStore, ExpertParts, ExpertPhase, ExpertStore, StoreMetrics,
};
pub use tiered::{TieredStore, WeightStorage};
pub use workload::{generate, unique_keys, Workload};

#[cfg(test)]
mod tests;
