//! Serving-shaped measurement: replay tables plus gpu-sim scores.
//!
//! Does not invent `$/M tokens`. Wall time is virtual nanoseconds from a
//! [`gpu_sim::HardwareProfile`].

#![deny(missing_docs, unsafe_code)]

pub use expertvm::{
    adversarial_suite, colocated, compare, cycling_pages, format_table, generate, kv_paged,
    kv_replay, report, schedule_placed, schedule_remote, schedule_replay, sim_placed,
    sim_remote_home, sim_remote_home_cfg, sim_replay, striped, topology_suite, with_hot_replicas,
    BenchReport, KvCfg, KvFill, KvReplay, Policy, Prefetch, SchedCfg, SimCfg, Trace, Workload,
    DECODE_ACTIVATION_BYTES,
};
pub use gpu_sim::{probe_topology, HardwareProfile, Score, TopologyProbe};

#[cfg(test)]
mod tests {
    use super::{
        adversarial_suite, cycling_pages, kv_paged, kv_replay, schedule_replay, sim_placed,
        sim_remote_home, striped, HardwareProfile, KvCfg, KvFill, SchedCfg, SimCfg, Trace,
    };

    #[test]
    fn adversarial_suite_covers_named_workloads() {
        let rows = adversarial_suite(8, 4, 2, HardwareProfile::example_cheap_48gb()).unwrap();
        assert_eq!(rows.len(), super::Workload::ALL.len());
        assert!(rows.iter().any(|r| r.name == "thrash"));
        assert!(rows.iter().any(|r| r.name == "batch"));
        assert!(rows.iter().any(|r| r.name == "batch-1"));
        assert!(rows.iter().any(|r| r.name == "batch-128"));
        let b1 = rows.iter().find(|r| r.name == "batch-1").unwrap();
        assert!(b1.overlap.is_none(), "{}", b1.render());
        let b128 = rows.iter().find(|r| r.name == "batch-128").unwrap();
        assert!(b128.schedule.is_some(), "{}", b128.render());
        assert!(b128.render().contains("schedule-all"));
        assert!(b128.render().contains("schedule-1"));
        assert!(rows.iter().any(|r| r.name == "shared-prefix"));
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

    #[test]
    fn kv_replay_reexport_pages_working_set() {
        let row = kv_replay(
            &cycling_pages(4, 8),
            HardwareProfile::example_cheap_48gb(),
            4096,
            2,
        )
        .unwrap();
        assert_eq!(row.hbm_peak, 2 * 4096);
        assert_eq!(row.pages, 4);
        assert!(row.misses >= 4);
    }

    #[test]
    fn kv_memset_fill_moves_no_host_bytes() {
        let row = kv_paged(
            &cycling_pages(4, 8),
            HardwareProfile::example_cheap_48gb(),
            KvCfg {
                page_bytes: 4096,
                slots: 2,
                fill: KvFill::Memset,
            },
        )
        .unwrap();
        assert_eq!(row.bytes_moved, 0);
        assert_eq!(row.fill, KvFill::Memset);
        assert_eq!(row.hbm_peak, 2 * 4096);
    }

    fn load_checked_in_trace(name: &str) -> Trace {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("tests")
            .join("traces")
            .join(name);
        let mut f =
            std::fs::File::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut text = String::new();
        let _n = std::io::Read::read_to_string(&mut f, &mut text).expect("read");
        Trace::parse(&text).expect("parse")
    }

    #[test]
    fn two_layer_jsonl_report_and_schedule() {
        let t = load_checked_in_trace("tiny-qwen3moe-2layer.jsonl");
        assert!(t.events.iter().any(|e| e.layer == 0));
        assert!(t.events.iter().any(|e| e.layer == 1));
        let row = super::report(
            "tiny-qwen3moe-2layer",
            &t,
            4,
            8,
            Some(HardwareProfile::example_cheap_48gb()),
            4096,
        )
        .expect("report");
        assert!(row.sim.is_some(), "{}", row.render());
        assert!(row.render().contains("sim_ns="), "{}", row.render());
        assert!(row.render().contains("layer-ahead"), "{}", row.render());
        let sch = schedule_replay(
            &t,
            HardwareProfile::example_cheap_48gb(),
            SimCfg {
                prefetch: super::Prefetch::CopyForward,
                ..SimCfg::lru(2, 4096, 8)
            },
            SchedCfg::closed(0),
        )
        .expect("schedule");
        assert_eq!(sch.completed, 1);
        assert!(sch.replay.sim_ns > 0, "{}", sch.replay.line());
        assert!(
            sch.replay.prefetch_hits > 0,
            "slots=2 copy-forward must hit L+1, {}",
            sch.replay.line()
        );
    }
}
