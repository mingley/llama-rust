//! Expert virtual memory: traces, policies, stores, GPU-sim replay.

#![deny(missing_docs, unsafe_code)]

mod access;
mod analyze;
mod error;
mod planner;
mod policy;
mod replay;
mod sim_replay;
mod store;

pub use access::{ExpertAccess, ExpertKey, Trace};
pub use analyze::{analyze, TraceStats};
pub use error::Error;
pub use planner::{plan_window, window_keys, Plan};
pub use policy::Policy;
pub use replay::{compare, format_table, replay, ReplayRow};
pub use sim_replay::{sim_replay, SimReplay};
pub use store::{replay_accesses, CachedStore, DirectStore, ExpertBlob, ExpertStore};

#[cfg(test)]
mod tests;
