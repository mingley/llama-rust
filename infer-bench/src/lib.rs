//! Serving-shaped measurement: replay tables plus gpu-sim scores.
//!
//! Does not invent `$/M tokens`. Wall time is virtual nanoseconds from a
//! [`gpu_sim::HardwareProfile`].

#![deny(missing_docs, unsafe_code)]

pub use expertvm::{
    adversarial_suite, compare, format_table, generate, report, sim_replay, BenchReport, Policy,
    Trace, Workload,
};
pub use gpu_sim::{HardwareProfile, Score};

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
}
