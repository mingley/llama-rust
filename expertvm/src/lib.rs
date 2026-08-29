//! Expert virtual memory: traces, policies, stores, GPU-sim replay.

#![deny(missing_docs, unsafe_code)]

mod access;
mod analyze;
mod bench;
mod error;
mod gpu_store;
mod live;
mod planner;
mod policy;
mod replay;
mod sim_replay;
mod store;
mod tiered;
mod workload;

pub use access::{ExpertAccess, ExpertKey, Trace};
pub use analyze::{analyze, TraceStats};
pub use bench::{adversarial_suite, report, topology_suite, BenchReport};
pub use error::Error;
pub use gpu_sim::{probe_topology, HardwareProfile, Score, TopologyProbe};
pub use gpu_store::SimulatedGpuStore;
pub use live::LiveStore;
pub use planner::{
    copy_forward, hot_keys, plan_placement, plan_window, window_keys, Placement, Plan,
};
pub use policy::Policy;
pub use replay::{compare, format_table, replay, ReplayRow};
pub use sim_replay::{compare_ep, home_gpu, sim_replay, sim_static_ep, EpCompare, SimReplay};
pub use store::{
    replay_accesses, CachedStore, DirectStore, ExpertParts, ExpertStore, StoreMetrics,
};
pub use tiered::{TieredStore, WeightStorage};
pub use workload::{generate, unique_keys, Workload};

#[cfg(test)]
mod tests;
