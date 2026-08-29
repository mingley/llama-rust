//! Serving-shaped measurement: replay tables plus gpu-sim scores.
//!
//! Does not invent `$/M tokens`. Wall time is virtual nanoseconds from a
//! [`gpu_sim::HardwareProfile`].

#![deny(missing_docs, unsafe_code)]

pub use expertvm::{
    adversarial_suite, compare, format_table, generate, report, sim_replay, topology_suite,
    BenchReport, Policy, Trace, Workload,
};
pub use gpu_sim::{probe_topology, HardwareProfile, Score, TopologyProbe};

#[cfg(test)]
mod tests {
    use super::{adversarial_suite, HardwareProfile};

    #[test]
    fn adversarial_suite_covers_named_workloads() {
        let rows = adversarial_suite(8, 4, 2, HardwareProfile::example_cheap_48gb()).unwrap();
        assert_eq!(rows.len(), 10);
        assert!(rows.iter().any(|r| r.name == "thrash"));
        assert!(rows.iter().any(|r| r.name == "batch"));
    }

    #[test]
    fn topology_suite_names_every_example_profile() {
        let rows = super::topology_suite(1u64 << 20).unwrap();
        assert_eq!(rows.len(), HardwareProfile::example_names().len());
        assert!(rows.iter().any(|r| r.line().contains("h2d_ns=")));
    }
}
