//! Expert virtual memory: traces, policies, stores, GPU-sim replay.

#![deny(missing_docs, unsafe_code)]

mod access;
mod analyze;
mod bench;
mod error;
mod gpu_store;
mod live;
mod place;
mod planner;
mod policy;
mod replay;
mod sim_replay;
mod store;
mod tiered;
mod workload;

pub use access::{weight_permille, ExpertAccess, ExpertKey, Trace};
pub use analyze::{analyze, coactivation_counts, freq_table, mass_table, TraceStats};
pub use bench::{adversarial_suite, report, topology_suite, BenchReport};
pub use error::Error;
pub use gpu_sim::{probe_topology, DeviceId, HardwareProfile, Score, TopologyProbe};
pub use gpu_store::SimulatedGpuStore;
pub use live::LiveStore;
pub use place::{colocated, home_gpu, striped, with_hot_replicas, PlaceMap};
pub use planner::{
    copy_forward, hot_keys, plan_placement, plan_window, prefetch_keys, transition_pair,
    window_keys, Markov, Placement, Plan, Prefetch,
};
pub use policy::Policy;
pub use replay::{compare, format_table, replay, ReplayRow};
pub use sim_replay::{
    compare_ep, sim_placed, sim_remote_home, sim_replay, sim_replay_cfg, sim_static_ep, EpCompare,
    SimCfg, SimReplay,
};
pub use store::{
    replay_accesses, CachedStore, DirectStore, ExpertParts, ExpertStore, StoreMetrics,
};
pub use tiered::{TieredStore, WeightStorage};
pub use workload::{generate, unique_keys, Workload};

#[cfg(test)]
mod tests;
