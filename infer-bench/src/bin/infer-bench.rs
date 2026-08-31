//! `infer-bench adversarial | trace` — measured hit rates and sim scores.

use gpu_sim::{PortableClusterMode, PortableSharedMode, SharedMemoryMode, SynchronizationPolicy};
use infer_bench::{
    adversarial_suite, colocated, report, schedule_placed, schedule_remote, sim_placed,
    sim_remote_home_cfg, striped, topology_suite, with_hot_replicas, HardwareProfile, SchedCfg,
    SimCfg, Trace, Workload, DECODE_ACTIVATION_BYTES,
};
use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::process::ExitCode;

const USAGE: &str = "\
usage: infer-bench <command> [args]
  adversarial [--tokens N] [--experts N] [--capacity N] [--profile NAME]
  trace <trace.jsonl> [--capacity N] [--profile NAME] [--expert-bytes N]
  workload <NAME> [--tokens N] [--experts N] [--capacity N] [--profile NAME]
  topology [--bytes N]
  remote <trace.jsonl> [--expert-bytes N] [--activation-bytes N] [--profile NAME]
  schedule <trace.jsonl> [--capacity N] [--profile NAME] [--expert-bytes N] [--max-batch N] [--interarrival-ns N] [--ttft-slo-ns N] [--itl-slo-ns N] [--prefill-chunk N] [--decode-first] [--slo-reject] [--prefix-cache] [--place none|striped|colocated|replicas|remote] [--activation-bytes N] [--decode-priority] [--cooperative] [--pdl] [--l2-persist] [--l2-reset] [--l2-fetch N] [--l2-ratio N] [--l2-streaming] [--cluster N] [--preferred-cluster N] [--cluster-spread] [--func-cluster-spread] [--cluster-load-balance] [--cluster-must-set] [--required-cluster N] [--max-shared] [--func-max-shared] [--max-l1] [--non-portable-cluster] [--sync-policy auto|spin|yield|blocking] [--device-sync-policy auto|spin|yield|blocking] [--shared-mem default|four|eight] [--func-shared-mem default|four|eight] [--device-shared-mem default|four|eight] [--portable-cluster default|portable|non-portable] [--optin-shared] [--dynamic-shared N] [--portable-shared default|portable|non-portable] [--nvlink-util] [--device-launch] [--device-updatable] [--kernel-priority N] [--launch-completion] [--programmatic-event] [--stream-attach] [--managed-host] [--prefetch-host] [--no-read-mostly] [--no-preferred] [--no-mem-prefetch] [--wait-value] [--vmm-retain] [--vmm-handle] [--multicast] [--compute-slots N] [--decode-sms N]

NAME: uniform, hotset, shifting-hotset, thrash, coding, chat, long-context,
      prefill-heavy, decode-heavy, batch-1, batch, batch-128, prefill-batch,
      shared-prefix
profiles: h100 (default), h200, 8xh100, cheap, 2xh100-pcie, bad-numa,
          2node-rdma, asymmetric, or a path to a .profile file
";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let mut err = std::io::stderr();
            let _w = writeln!(err, "{e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let cmd = args.next().ok_or_else(|| USAGE.trim().to_string())?;
    match cmd.as_str() {
        "help" | "--help" | "-h" => {
            let mut out = std::io::stdout();
            out.write_all(USAGE.as_bytes()).map_err(|e| e.to_string())?;
            Ok(())
        }
        "adversarial" => {
            let cfg = parse_flags(args)?;
            let profile = load_profile(&cfg.profile)?;
            let rows = adversarial_suite(cfg.tokens, cfg.experts, cfg.capacity, profile)
                .map_err(|e| e.to_string())?;
            for row in rows {
                print!("{}", row.render());
            }
            Ok(())
        }
        "trace" => {
            let cfg = parse_flags(args)?;
            let path = cfg.path.ok_or("trace <trace.jsonl>")?;
            let trace = load_trace(&path)?;
            let profile = load_profile(&cfg.profile)?;
            let row = report(
                &path,
                &trace,
                cfg.capacity,
                8,
                Some(profile),
                cfg.expert_bytes,
            )
            .map_err(|e| e.to_string())?;
            print!("{}", row.render());
            Ok(())
        }
        "workload" => {
            let cfg = parse_flags(args)?;
            let name = cfg.path.ok_or("workload <name>")?;
            let kind = parse_workload(&name)?;
            let trace = infer_bench::generate(kind, cfg.tokens, cfg.experts, 1, 1);
            let profile = load_profile(&cfg.profile)?;
            let row = report(
                kind.name(),
                &trace,
                cfg.capacity,
                8,
                Some(profile),
                cfg.expert_bytes,
            )
            .map_err(|e| e.to_string())?;
            print!("{}", row.render());
            Ok(())
        }
        "topology" => {
            let cfg = parse_flags(args)?;
            let rows = topology_suite(cfg.expert_bytes).map_err(|e| e.to_string())?;
            for row in rows {
                println!("{}", row.line());
            }
            Ok(())
        }
        "remote" => {
            let cfg = parse_flags_profile(args, "2node-rdma")?;
            let path = cfg.path.ok_or("remote <trace.jsonl>")?;
            let trace = load_trace(&path)?;
            let profile = load_profile(&cfg.profile)?;
            let n = u16::try_from(profile.n_gpus()).unwrap_or(1).max(1);
            let map = striped(&trace, n);
            let local = sim_placed(&trace, profile.clone(), cfg.expert_bytes, &map)
                .map_err(|e| e.to_string())?;
            let remote = sim_remote_home_cfg(
                &trace,
                profile,
                cfg.expert_bytes,
                cfg.activation_bytes,
                &map,
            )
            .map_err(|e| e.to_string())?;
            println!("local {} | remote {}", local.line(), remote.line());
            Ok(())
        }
        "schedule" => {
            let cfg = parse_flags(args)?;
            let path = cfg.path.ok_or("schedule <trace.jsonl>")?;
            let trace = load_trace(&path)?;
            let profile = load_profile(&cfg.profile)?;
            let n_gpus = u16::try_from(profile.n_gpus()).unwrap_or(1).max(1);
            let mut sim_cfg = SimCfg::lru(cfg.capacity, cfg.expert_bytes, 8);
            sim_cfg.compute_slots = cfg.compute_slots;
            sim_cfg.decode_sm_permille = cfg.decode_sm_permille;
            sim_cfg.decode_priority = cfg.decode_priority;
            sim_cfg.cooperative = cfg.cooperative;
            sim_cfg.pdl = cfg.pdl;
            sim_cfg.l2_persist =
                cfg.l2_persist || cfg.l2_reset || cfg.l2_fetch != 0 || cfg.l2_ratio != 0;
            sim_cfg.l2_reset = cfg.l2_reset;
            sim_cfg.l2_fetch = cfg.l2_fetch;
            sim_cfg.l2_ratio = cfg.l2_ratio;
            sim_cfg.l2_streaming = cfg.l2_streaming;
            sim_cfg.cluster = cfg.cluster;
            sim_cfg.preferred_cluster = cfg.preferred_cluster;
            sim_cfg.cluster_spread = cfg.cluster_spread;
            sim_cfg.func_cluster_spread = cfg.func_cluster_spread;
            sim_cfg.cluster_load_balance = cfg.cluster_load_balance;
            sim_cfg.cluster_must_set = cfg.cluster_must_set;
            sim_cfg.required_cluster = cfg.required_cluster;
            sim_cfg.max_shared = cfg.max_shared;
            sim_cfg.func_max_shared = cfg.func_max_shared;
            sim_cfg.max_l1 = cfg.max_l1;
            sim_cfg.non_portable_cluster = cfg.non_portable_cluster;
            sim_cfg.sync_policy = cfg.sync_policy;
            sim_cfg.device_sync_policy = cfg.device_sync_policy;
            sim_cfg.shared_mem = cfg.shared_mem;
            sim_cfg.func_shared_mem = cfg.func_shared_mem;
            sim_cfg.device_shared_mem = cfg.device_shared_mem;
            sim_cfg.portable_cluster = cfg.portable_cluster;
            sim_cfg.optin_shared = cfg.optin_shared;
            sim_cfg.dynamic_shared = cfg.dynamic_shared;
            sim_cfg.portable_shared = cfg.portable_shared;
            sim_cfg.nvlink_util_centric = cfg.nvlink_util_centric;
            sim_cfg.device_updatable = cfg.device_updatable;
            sim_cfg.kernel_priority = cfg.kernel_priority;
            sim_cfg.device_launch = cfg.device_launch;
            sim_cfg.launch_completion = cfg.launch_completion;
            sim_cfg.programmatic_event = cfg.programmatic_event;
            sim_cfg.stream_attach = cfg.stream_attach;
            sim_cfg.managed_host = cfg.managed_host;
            sim_cfg.prefetch_host = cfg.prefetch_host;
            sim_cfg.no_read_mostly = cfg.no_read_mostly;
            sim_cfg.no_preferred = cfg.no_preferred;
            sim_cfg.no_mem_prefetch = cfg.no_mem_prefetch;
            if cfg.stream_attach
                || cfg.managed_host
                || cfg.prefetch_host
                || cfg.no_read_mostly
                || cfg.no_preferred
                || cfg.no_mem_prefetch
            {
                sim_cfg.managed = true;
            }
            sim_cfg.wait_value = cfg.wait_value;
            sim_cfg.multicast = cfg.multicast;
            sim_cfg.vmm_retain = cfg.vmm_retain;
            sim_cfg.vmm_handle = cfg.vmm_handle;
            if cfg.multicast || cfg.vmm_retain || cfg.vmm_handle {
                sim_cfg.vmm = true;
            }
            if cfg.decode_priority {
                sim_cfg.stream_priority = true;
            }
            let sched = SchedCfg {
                max_batch: cfg.max_batch,
                interarrival_ns: cfg.interarrival_ns,
                ttft_slo_ns: cfg.ttft_slo_ns,
                itl_slo_ns: cfg.itl_slo_ns,
                prefill_chunk_layers: cfg.prefill_chunk,
                decode_first: cfg.decode_first,
                slo_reject: cfg.slo_reject,
                prefix_cache: cfg.prefix_cache,
            };
            let row = if cfg.place == "remote" {
                let map = striped(&trace, n_gpus);
                schedule_remote(&trace, profile, sim_cfg, sched, &map, cfg.activation_bytes)
            } else {
                let map = match cfg.place.as_str() {
                    "none" => None,
                    "striped" => Some(striped(&trace, n_gpus)),
                    "colocated" => Some(colocated(&trace, n_gpus)),
                    "replicas" => Some(with_hot_replicas(
                        colocated(&trace, n_gpus),
                        &trace,
                        n_gpus,
                        200,
                    )),
                    other => {
                        return Err(format!(
                            "unknown --place {other} (none|striped|colocated|replicas|remote)"
                        ))
                    }
                };
                schedule_placed(&trace, profile, sim_cfg, sched, map.as_ref())
            }
            .map_err(|e| e.to_string())?;
            println!("{}", row.line());
            Ok(())
        }
        other => Err(format!("{USAGE}got {other}")),
    }
}

struct Cfg {
    path: Option<String>,
    capacity: usize,
    expert_bytes: u64,
    activation_bytes: u64,
    profile: String,
    tokens: u32,
    experts: u32,
    max_batch: usize,
    interarrival_ns: u64,
    ttft_slo_ns: Option<u64>,
    itl_slo_ns: Option<u64>,
    prefill_chunk: usize,
    decode_first: bool,
    slo_reject: bool,
    prefix_cache: bool,
    place: String,
    compute_slots: u8,
    decode_sm_permille: u16,
    decode_priority: bool,
    cooperative: bool,
    pdl: bool,
    l2_persist: bool,
    l2_reset: bool,
    l2_fetch: u64,
    l2_ratio: u16,
    l2_streaming: bool,
    cluster: u8,
    preferred_cluster: u8,
    cluster_spread: bool,
    func_cluster_spread: bool,
    cluster_load_balance: bool,
    cluster_must_set: bool,
    required_cluster: u8,
    max_shared: bool,
    func_max_shared: bool,
    max_l1: bool,
    non_portable_cluster: bool,
    sync_policy: SynchronizationPolicy,
    device_sync_policy: SynchronizationPolicy,
    shared_mem: SharedMemoryMode,
    func_shared_mem: SharedMemoryMode,
    device_shared_mem: SharedMemoryMode,
    portable_cluster: PortableClusterMode,
    optin_shared: bool,
    dynamic_shared: u32,
    portable_shared: PortableSharedMode,
    nvlink_util_centric: bool,
    device_updatable: bool,
    kernel_priority: Option<i32>,
    device_launch: bool,
    launch_completion: bool,
    programmatic_event: bool,
    stream_attach: bool,
    managed_host: bool,
    prefetch_host: bool,
    no_read_mostly: bool,
    no_preferred: bool,
    no_mem_prefetch: bool,
    wait_value: bool,
    multicast: bool,
    vmm_retain: bool,
    vmm_handle: bool,
}

fn parse_flags<I>(args: I) -> Result<Cfg, String>
where
    I: IntoIterator<Item = String>,
{
    parse_flags_profile(args, "h100")
}

fn parse_flags_profile<I>(args: I, default_profile: &str) -> Result<Cfg, String>
where
    I: IntoIterator<Item = String>,
{
    let mut path = None;
    let mut capacity = 8usize;
    let mut expert_bytes = 4096u64;
    let mut activation_bytes = DECODE_ACTIVATION_BYTES;
    let mut profile = default_profile.to_string();
    let mut tokens = 64u32;
    let mut experts = 16u32;
    let mut max_batch = 0usize;
    let mut interarrival_ns = 0u64;
    let mut ttft_slo_ns = None;
    let mut itl_slo_ns = None;
    let mut prefill_chunk = 0usize;
    let mut decode_first = false;
    let mut slo_reject = false;
    let mut prefix_cache = false;
    let mut place = String::from("none");
    let mut compute_slots = 0u8;
    let mut decode_sm_permille = 0u16;
    let mut decode_priority = false;
    let mut cooperative = false;
    let mut pdl = false;
    let mut l2_persist = false;
    let mut l2_reset = false;
    let mut l2_fetch = 0u64;
    let mut l2_ratio = 0u16;
    let mut l2_streaming = false;
    let mut cluster = 0u8;
    let mut preferred_cluster = 0u8;
    let mut cluster_spread = false;
    let mut func_cluster_spread = false;
    let mut cluster_load_balance = false;
    let mut cluster_must_set = false;
    let mut required_cluster = 0u8;
    let mut max_shared = false;
    let mut func_max_shared = false;
    let mut max_l1 = false;
    let mut non_portable_cluster = false;
    let mut sync_policy = SynchronizationPolicy::Auto;
    let mut device_sync_policy = SynchronizationPolicy::Auto;
    let mut shared_mem = SharedMemoryMode::Default;
    let mut func_shared_mem = SharedMemoryMode::Default;
    let mut device_shared_mem = SharedMemoryMode::Default;
    let mut portable_cluster = PortableClusterMode::Default;
    let mut optin_shared = false;
    let mut dynamic_shared = 0u32;
    let mut portable_shared = PortableSharedMode::Default;
    let mut nvlink_util_centric = false;
    let mut device_updatable = false;
    let mut kernel_priority = None;
    let mut device_launch = false;
    let mut launch_completion = false;
    let mut programmatic_event = false;
    let mut stream_attach = false;
    let mut managed_host = false;
    let mut prefetch_host = false;
    let mut no_read_mostly = false;
    let mut no_preferred = false;
    let mut no_mem_prefetch = false;
    let mut wait_value = false;
    let mut multicast = false;
    let mut vmm_retain = false;
    let mut vmm_handle = false;
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        let (key, inline) = match arg.split_once('=') {
            Some((k, v)) => (k.to_string(), Some(v.to_string())),
            None => (arg, None),
        };
        match key.as_str() {
            "--capacity" | "-c" => {
                capacity = parse_usize("capacity", &value("capacity", inline, &mut it)?)?
            }
            "--expert-bytes" => {
                expert_bytes = parse_u64("expert-bytes", &value("expert-bytes", inline, &mut it)?)?
            }
            "--activation-bytes" => {
                activation_bytes = parse_u64(
                    "activation-bytes",
                    &value("activation-bytes", inline, &mut it)?,
                )?
            }
            "--profile" => profile = value("profile", inline, &mut it)?,
            "--tokens" => tokens = parse_u32("tokens", &value("tokens", inline, &mut it)?)?,
            "--experts" => experts = parse_u32("experts", &value("experts", inline, &mut it)?)?,
            "--max-batch" => {
                max_batch = parse_usize("max-batch", &value("max-batch", inline, &mut it)?)?
            }
            "--interarrival-ns" => {
                interarrival_ns = parse_u64(
                    "interarrival-ns",
                    &value("interarrival-ns", inline, &mut it)?,
                )?
            }
            "--ttft-slo-ns" => {
                ttft_slo_ns = Some(parse_u64(
                    "ttft-slo-ns",
                    &value("ttft-slo-ns", inline, &mut it)?,
                )?)
            }
            "--itl-slo-ns" => {
                itl_slo_ns = Some(parse_u64(
                    "itl-slo-ns",
                    &value("itl-slo-ns", inline, &mut it)?,
                )?)
            }
            "--prefill-chunk" => {
                prefill_chunk =
                    parse_usize("prefill-chunk", &value("prefill-chunk", inline, &mut it)?)?
            }
            "--decode-first" => {
                decode_first = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--decode-priority" => {
                decode_priority = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--cooperative" => {
                cooperative = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--pdl" => {
                pdl = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--l2-persist" => {
                l2_persist = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--l2-reset" => {
                l2_reset = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--l2-fetch" => l2_fetch = parse_l2_fetch(&value("l2-fetch", inline, &mut it)?)?,
            "--l2-ratio" => l2_ratio = parse_l2_ratio(&value("l2-ratio", inline, &mut it)?)?,
            "--l2-streaming" => {
                l2_streaming = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--cluster" => cluster = parse_cluster(&value("cluster", inline, &mut it)?)?,
            "--preferred-cluster" => {
                preferred_cluster =
                    parse_preferred_cluster(&value("preferred-cluster", inline, &mut it)?)?
            }
            "--cluster-spread" => {
                cluster_spread = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--func-cluster-spread" => {
                func_cluster_spread = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--cluster-load-balance" => {
                cluster_load_balance = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--cluster-must-set" => {
                cluster_must_set = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--required-cluster" => {
                required_cluster =
                    parse_required_cluster(&value("required-cluster", inline, &mut it)?)?
            }
            "--max-shared" => {
                max_shared = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--func-max-shared" => {
                func_max_shared = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--max-l1" => {
                max_l1 = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--non-portable-cluster" => {
                non_portable_cluster = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--sync-policy" => {
                sync_policy = parse_sync_policy(&value("sync-policy", inline, &mut it)?)?
            }
            "--device-sync-policy" => {
                device_sync_policy =
                    parse_device_sync_policy(&value("device-sync-policy", inline, &mut it)?)?
            }
            "--shared-mem" => {
                shared_mem = parse_shared_mem(&value("shared-mem", inline, &mut it)?)?
            }
            "--func-shared-mem" => {
                func_shared_mem =
                    parse_func_shared_mem(&value("func-shared-mem", inline, &mut it)?)?
            }
            "--device-shared-mem" => {
                device_shared_mem =
                    parse_device_shared_mem(&value("device-shared-mem", inline, &mut it)?)?
            }
            "--portable-cluster" => {
                portable_cluster =
                    parse_portable_cluster(&value("portable-cluster", inline, &mut it)?)?
            }
            "--optin-shared" => {
                optin_shared = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--dynamic-shared" => {
                dynamic_shared = parse_dynamic_shared(&value("dynamic-shared", inline, &mut it)?)?
            }
            "--portable-shared" => {
                portable_shared =
                    parse_portable_shared(&value("portable-shared", inline, &mut it)?)?
            }
            "--nvlink-util" => {
                nvlink_util_centric = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--device-updatable" => {
                device_updatable = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--kernel-priority" => {
                kernel_priority = Some(parse_kernel_priority(&value(
                    "kernel-priority",
                    inline,
                    &mut it,
                )?)?)
            }
            "--device-launch" => {
                device_launch = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--launch-completion" => {
                launch_completion = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--programmatic-event" => {
                programmatic_event = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--stream-attach" => {
                stream_attach = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--managed-host" => {
                managed_host = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--prefetch-host" => {
                prefetch_host = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--no-read-mostly" => {
                no_read_mostly = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--no-preferred" => {
                no_preferred = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--no-mem-prefetch" => {
                no_mem_prefetch = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--wait-value" => {
                wait_value = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--multicast" => {
                multicast = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--vmm-retain" => {
                vmm_retain = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--vmm-handle" => {
                vmm_handle = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--slo-reject" => {
                slo_reject = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--prefix-cache" => {
                prefix_cache = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--place" => place = value("place", inline, &mut it)?,
            "--bytes" => expert_bytes = parse_u64("bytes", &value("bytes", inline, &mut it)?)?,
            "--compute-slots" => {
                let n = value("compute-slots", inline, &mut it)?;
                let n = n
                    .parse::<u8>()
                    .map_err(|_| format!("invalid compute-slots {n:?}"))?;
                if n == 0 {
                    return Err("compute-slots must be > 0".into());
                }
                compute_slots = n;
            }
            "--decode-sms" => {
                let n = value("decode-sms", inline, &mut it)?;
                let n = n
                    .parse::<u16>()
                    .map_err(|_| format!("invalid decode-sms {n:?}"))?;
                if n == 0 || n > 1000 {
                    return Err("decode-sms must be 1..=1000".into());
                }
                decode_sm_permille = n;
            }
            flag if flag.starts_with('-') => return Err(format!("unknown flag {flag}\n{USAGE}")),
            other => {
                if path.is_some() {
                    return Err(format!("unexpected argument {other}\n{USAGE}"));
                }
                path = Some(other.to_string());
            }
        }
    }
    if vmm_handle && vmm_retain {
        return Err("choose one of --vmm-handle, --vmm-retain".into());
    }
    if pdl && cooperative {
        return Err("choose one of --pdl, --cooperative".into());
    }
    if launch_completion && device_launch {
        return Err("launch-completion cannot device-launch".into());
    }
    if programmatic_event && device_launch {
        return Err("programmatic-event cannot device-launch".into());
    }
    if preferred_cluster != 0 && cluster == 0 {
        return Err("--preferred-cluster needs --cluster".into());
    }
    if preferred_cluster != 0 && !preferred_cluster.is_multiple_of(cluster) {
        return Err("preferred-cluster must be a multiple of cluster".into());
    }
    if cluster_must_set && cluster == 0 {
        return Err("--cluster-must-set needs --cluster".into());
    }
    if required_cluster != 0 && cluster == 0 {
        return Err("--required-cluster needs --cluster".into());
    }
    if required_cluster != 0 && required_cluster != cluster {
        return Err("required-cluster must match --cluster".into());
    }
    if cluster_load_balance && cluster_spread {
        return Err("choose one of --cluster-load-balance, --cluster-spread".into());
    }
    if cluster_load_balance && !func_cluster_spread {
        return Err("--cluster-load-balance needs --func-cluster-spread".into());
    }
    if max_l1 && max_shared {
        return Err("choose one of --max-l1, --max-shared".into());
    }
    if max_l1 && !func_max_shared {
        return Err("--max-l1 needs --func-max-shared".into());
    }
    if l2_streaming && !(l2_persist || l2_reset || l2_fetch != 0 || l2_ratio != 0) {
        return Err("--l2-streaming needs --l2-persist".into());
    }
    Ok(Cfg {
        path,
        capacity,
        expert_bytes,
        activation_bytes,
        profile,
        tokens,
        experts,
        max_batch,
        interarrival_ns,
        ttft_slo_ns,
        itl_slo_ns,
        prefill_chunk,
        decode_first,
        slo_reject,
        prefix_cache,
        place,
        compute_slots,
        decode_sm_permille,
        decode_priority,
        cooperative,
        pdl,
        l2_persist,
        l2_reset,
        l2_fetch,
        l2_ratio,
        l2_streaming,
        cluster,
        preferred_cluster,
        cluster_spread,
        func_cluster_spread,
        cluster_load_balance,
        cluster_must_set,
        required_cluster,
        max_shared,
        func_max_shared,
        max_l1,
        non_portable_cluster,
        sync_policy,
        device_sync_policy,
        shared_mem,
        func_shared_mem,
        device_shared_mem,
        portable_cluster,
        optin_shared,
        dynamic_shared,
        portable_shared,
        nvlink_util_centric,
        device_updatable,
        kernel_priority,
        device_launch,
        launch_completion,
        programmatic_event,
        stream_attach,
        managed_host,
        prefetch_host,
        no_read_mostly,
        no_preferred,
        no_mem_prefetch,
        wait_value,
        multicast,
        vmm_retain,
        vmm_handle,
    })
}

fn value<I>(name: &str, inline: Option<String>, it: &mut I) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    if let Some(v) = inline {
        return Ok(v);
    }
    it.next().ok_or_else(|| format!("missing --{name} value"))
}

fn parse_usize(name: &str, s: &str) -> Result<usize, String> {
    s.parse::<usize>()
        .map_err(|_| format!("invalid {name} {s:?}"))
}

fn parse_u64(name: &str, s: &str) -> Result<u64, String> {
    s.parse::<u64>()
        .map_err(|_| format!("invalid {name} {s:?}"))
}

fn parse_u32(name: &str, s: &str) -> Result<u32, String> {
    s.parse::<u32>()
        .map_err(|_| format!("invalid {name} {s:?}"))
}

fn parse_cluster(s: &str) -> Result<u8, String> {
    let n = s
        .parse::<u8>()
        .map_err(|_| format!("invalid cluster {s:?}"))?;
    if n == 0 {
        return Err("cluster must be > 0".into());
    }
    Ok(n)
}

fn parse_required_cluster(s: &str) -> Result<u8, String> {
    let n = s
        .parse::<u8>()
        .map_err(|_| format!("invalid required-cluster {s:?}"))?;
    if n == 0 {
        return Err("required-cluster must be > 0".into());
    }
    Ok(n)
}

fn parse_l2_fetch(s: &str) -> Result<u64, String> {
    let n = s
        .parse::<u64>()
        .map_err(|_| format!("invalid l2-fetch {s:?}"))?;
    if n != 32 && n != 64 && n != 128 {
        return Err("l2-fetch must be 32, 64, or 128".into());
    }
    Ok(n)
}

fn parse_l2_ratio(s: &str) -> Result<u16, String> {
    let n = s
        .parse::<u16>()
        .map_err(|_| format!("invalid l2-ratio {s:?}"))?;
    if n == 0 || n > 1000 {
        return Err("l2-ratio must be 1..=1000".into());
    }
    Ok(n)
}

fn parse_preferred_cluster(s: &str) -> Result<u8, String> {
    let n = s
        .parse::<u8>()
        .map_err(|_| format!("invalid preferred-cluster {s:?}"))?;
    if n == 0 {
        return Err("preferred-cluster must be > 0".into());
    }
    Ok(n)
}

fn parse_sync_policy(s: &str) -> Result<SynchronizationPolicy, String> {
    SynchronizationPolicy::parse(s).map_err(|_| format!("unknown sync-policy {s}"))
}

fn parse_device_sync_policy(s: &str) -> Result<SynchronizationPolicy, String> {
    SynchronizationPolicy::parse(s).map_err(|_| format!("unknown device-sync-policy {s}"))
}

fn parse_shared_mem(s: &str) -> Result<SharedMemoryMode, String> {
    SharedMemoryMode::parse(s).map_err(|_| format!("unknown shared-mem {s}"))
}

fn parse_func_shared_mem(s: &str) -> Result<SharedMemoryMode, String> {
    SharedMemoryMode::parse(s).map_err(|_| format!("unknown func-shared-mem {s}"))
}

fn parse_device_shared_mem(s: &str) -> Result<SharedMemoryMode, String> {
    SharedMemoryMode::parse(s).map_err(|_| format!("unknown device-shared-mem {s}"))
}

fn parse_portable_cluster(s: &str) -> Result<PortableClusterMode, String> {
    PortableClusterMode::parse(s).map_err(|_| format!("unknown portable-cluster {s}"))
}

fn parse_dynamic_shared(s: &str) -> Result<u32, String> {
    let n = s
        .parse::<u32>()
        .map_err(|_| format!("invalid dynamic-shared {s:?}"))?;
    if n == 0 {
        return Err("dynamic-shared must be > 0".into());
    }
    Ok(n)
}

fn parse_kernel_priority(s: &str) -> Result<i32, String> {
    s.parse::<i32>()
        .map_err(|_| format!("invalid kernel-priority {s:?}"))
}

fn parse_portable_shared(s: &str) -> Result<PortableSharedMode, String> {
    PortableSharedMode::parse(s).map_err(|_| format!("unknown portable-shared {s}"))
}

fn parse_workload(name: &str) -> Result<Workload, String> {
    Workload::from_name(name).ok_or_else(|| format!("unknown workload {name}"))
}

fn load_trace(path: &str) -> Result<Trace, String> {
    let mut f = File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut buf = String::new();
    let _n = f
        .read_to_string(&mut buf)
        .map_err(|e| format!("{path}: {e}"))?;
    Trace::parse(&buf).map_err(|e| e.to_string())
}

fn load_profile(name: &str) -> Result<HardwareProfile, String> {
    if let Ok(p) = HardwareProfile::by_name(name) {
        return Ok(p);
    }
    let mut f = File::open(name).map_err(|e| {
        format!(
            "unknown profile {name} ({e}); known: {}",
            HardwareProfile::example_names().join(", ")
        )
    })?;
    let mut buf = String::new();
    let _n = f
        .read_to_string(&mut buf)
        .map_err(|e| format!("{name}: {e}"))?;
    HardwareProfile::parse(&buf).map_err(|e| e.to_string())
}
