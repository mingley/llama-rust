//! Serving-shaped measurement: replay tables plus gpu-sim scores.
//!
//! Does not invent `$/M tokens`. Wall time is virtual nanoseconds from a
//! [`gpu_sim::HardwareProfile`].

#![deny(missing_docs, unsafe_code)]

pub use expertvm::{
    adversarial_suite, colocated, compare, format_table, generate, report, schedule_placed,
    schedule_replay, sim_placed, sim_remote_home, sim_remote_home_cfg, sim_replay, striped,
    topology_suite, with_hot_replicas, BenchReport, Policy, SchedCfg, SimCfg, Trace, Workload,
    DECODE_ACTIVATION_BYTES,
};
pub use gpu_sim::{probe_topology, HardwareProfile, Score, TopologyProbe};

#[cfg(test)]
mod tests {
    use super::{
        adversarial_suite, schedule_replay, sim_placed, sim_remote_home, striped, HardwareProfile,
        SchedCfg, SimCfg, Trace,
    };

    #[test]
    fn adversarial_suite_covers_named_workloads() {
        let rows = adversarial_suite(8, 4, 2, HardwareProfile::example_cheap_48gb()).unwrap();
        assert_eq!(rows.len(), super::Workload::ALL.len());
        assert!(rows.iter().any(|r| r.name == "thrash"));
        assert!(rows.iter().any(|r| r.name == "batch"));
    }

    #[test]
    fn topology_suite_names_every_example_profile() {
        let rows = super::topology_suite(1u64 << 20).unwrap();
        assert_eq!(rows.len(), HardwareProfile::example_names().len());
        assert!(rows.iter().any(|r| r.line().contains("h2d_ns=")));
    }

    #[test]
    fn remote_home_reexport_pays_peer_copy() {
        let t = Trace::parse("{\"sequence\":0,\"token\":0,\"layer\":0,\"experts\":[1]}\n").unwrap();
        let p = HardwareProfile::example_2node_rdma();
        let map = striped(&t, 2);
        let bytes = 1u64 << 20;
        let local = sim_placed(&t, p.clone(), bytes, &map).unwrap();
        let remote = sim_remote_home(&t, p, bytes, &map).unwrap();
        assert!(remote.bytes_moved > local.bytes_moved);
    }

    #[test]
    fn schedule_reexport_completes_sequences() {
        let t = Trace::parse(
            "{\"sequence\":0,\"token\":0,\"layer\":0,\"experts\":[0]}\n{\"sequence\":1,\"token\":0,\"layer\":0,\"experts\":[1]}\n",
        )
        .unwrap();
        let row = schedule_replay(
            &t,
            HardwareProfile::example_cheap_48gb(),
            SimCfg::lru(2, 4096, 0),
            SchedCfg::closed(0),
        )
        .unwrap();
        assert_eq!(row.completed, 2);
        assert!(row.replay.misses >= 2);
    }
}
