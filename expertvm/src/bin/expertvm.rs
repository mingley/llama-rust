//! `expertvm analyze | replay | sim | store` — traces in, measured tables out.

use expertvm::{
    adversarial_suite, analyze, colocated, compare, compare_ep, cycling_pages, format_table,
    generate, kv_paged, report, schedule_placed, schedule_remote, sim_placed, sim_remote_home_cfg,
    sim_replay_cfg, store_replay_cfg, striped, topology_suite, with_hot_replicas, GpuFill,
    GpuStoreCfg, KvCfg, KvFill, Policy, Prefetch, SchedCfg, SimCfg, StoreReplayCfg, Trace,
    Workload, DECODE_ACTIVATION_BYTES,
};
use gpu_sim::{
    HardwareProfile, MemSyncDomain, PortableClusterMode, PortableSharedMode, SharedMemoryMode,
    SynchronizationPolicy,
};
use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::process::ExitCode;

const USAGE: &str = "\
usage: expertvm <command> [args]
  analyze  <trace.jsonl>
  replay   <trace.jsonl> [--capacity N] [--lookahead N]
  sim      <trace.jsonl> [--capacity N] [--lookahead N] [--expert-bytes N] [--profile NAME] [--prefetch none|copy-forward|markov|both] [--seq-streams] [--cuda-graphs] [--graph-update] [--graph-set-params] [--graph-clone] [--graph-build] [--graph-build-deps] [--graph-host] [--graph-piecewise] [--graph-capture-deps] [--graph-enable] [--graph-mem] [--graph-memset] [--graph-auto-free] [--graph-mem-trim] [--plan-window N] [--plan-threshold N] [--max-batch N] [--sync-alloc] [--mempool] [--mempool-trim] [--mempool-no-reuse] [--mempool-max N] [--shareable] [--mapped] [--managed] [--vmm] [--vmm-page N] [--host-func] [--blocking-streams] [--pageable] [--host-register] [--host-register-mapped] [--sync-memops] [--device-sync-memops] [--memcpy-batch] [--memcpy-during] [--memcpy-any] [--accessed-by] [--legacy-null] [--stream-priority] [--decode-priority] [--cooperative] [--pdl] [--l2-persist] [--l2-reset] [--l2-fetch N] [--l2-ratio N] [--l2-streaming] [--cluster N] [--preferred-cluster N] [--cluster-spread] [--func-cluster-spread] [--cluster-load-balance] [--cluster-must-set] [--required-cluster N] [--max-shared] [--func-max-shared] [--max-l1] [--non-portable-cluster] [--sync-policy auto|spin|yield|blocking] [--device-sync-policy auto|spin|yield|blocking] [--mem-sync-domain default|remote] [--mem-sync-map identity|collapse] [--mem-sync-launch] [--mem-sync-launch-map] [--shared-mem default|four|eight] [--func-shared-mem default|four|eight] [--device-shared-mem default|four|eight] [--portable-cluster default|portable|non-portable] [--optin-shared] [--dynamic-shared N] [--portable-shared default|portable|non-portable] [--nvlink-util] [--device-launch] [--device-updatable] [--kernel-priority N] [--launch-completion] [--programmatic-event] [--stream-attach] [--managed-host] [--prefetch-host] [--wait-value] [--multicast] [--compute-slots N] [--decode-sms N]
  schedule <trace.jsonl> [--capacity N] [--lookahead N] [--expert-bytes N] [--profile NAME] [--prefetch none|copy-forward|markov|both] [--seq-streams] [--cuda-graphs] [--graph-update] [--graph-set-params] [--graph-clone] [--graph-build] [--graph-build-deps] [--graph-host] [--graph-piecewise] [--graph-capture-deps] [--graph-enable] [--graph-mem] [--graph-memset] [--graph-auto-free] [--graph-mem-trim] [--plan-window N] [--plan-threshold N] [--max-batch N] [--interarrival-ns N] [--ttft-slo-ns N] [--itl-slo-ns N] [--prefill-chunk N] [--decode-first] [--slo-reject] [--prefix-cache] [--place none|striped|colocated|replicas|remote] [--activation-bytes N] [--sync-alloc] [--mempool] [--mempool-trim] [--mempool-no-reuse] [--mempool-max N] [--shareable] [--mapped] [--managed] [--vmm] [--vmm-page N] [--host-func] [--blocking-streams] [--pageable] [--host-register] [--host-register-mapped] [--sync-memops] [--device-sync-memops] [--memcpy-batch] [--memcpy-during] [--memcpy-any] [--accessed-by] [--legacy-null] [--stream-priority] [--decode-priority] [--cooperative] [--pdl] [--l2-persist] [--l2-reset] [--l2-fetch N] [--l2-ratio N] [--l2-streaming] [--cluster N] [--preferred-cluster N] [--cluster-spread] [--func-cluster-spread] [--cluster-load-balance] [--cluster-must-set] [--required-cluster N] [--max-shared] [--func-max-shared] [--max-l1] [--non-portable-cluster] [--sync-policy auto|spin|yield|blocking] [--device-sync-policy auto|spin|yield|blocking] [--mem-sync-domain default|remote] [--mem-sync-map identity|collapse] [--mem-sync-launch] [--mem-sync-launch-map] [--shared-mem default|four|eight] [--func-shared-mem default|four|eight] [--device-shared-mem default|four|eight] [--portable-cluster default|portable|non-portable] [--optin-shared] [--dynamic-shared N] [--portable-shared default|portable|non-portable] [--nvlink-util] [--device-launch] [--device-updatable] [--kernel-priority N] [--launch-completion] [--programmatic-event] [--stream-attach] [--managed-host] [--prefetch-host] [--wait-value] [--multicast] [--compute-slots N] [--decode-sms N]
  bench    <trace.jsonl> [--capacity N] [--lookahead N] [--expert-bytes N] [--profile NAME]
  bench    adversarial [--tokens N] [--experts N] [--capacity N] [--profile NAME]
  workload <NAME> [--tokens N] [--experts N] [--capacity N] [--profile NAME]
  topology [--bytes N]
  ep       <trace.jsonl> [--capacity N] [--expert-bytes N] [--hbm-bytes N] [--profile NAME]
  place    <trace.jsonl> [--gpus N] [--hot-pt N]
  remote   <trace.jsonl> [--expert-bytes N] [--activation-bytes N] [--profile NAME]
  kv       [--pages N] [--page-bytes B] [--capacity C] [--tokens T] [--profile NAME] [--fill h2d|memset] [--sequences N] [--row-width W] [--pitch P]
  store    <trace.jsonl> [--capacity N] [--expert-bytes N] [--profile NAME] [--prefetch none|copy-forward|markov|both] [--plan-window N] [--plan-threshold N] [--mapped] [--managed] [--vmm] [--vmm-page N] [--sync-alloc] [--mempool] [--mempool-trim] [--mempool-no-reuse] [--mempool-max N] [--shareable] [--host-func] [--blocking-streams] [--pageable] [--host-register] [--host-register-mapped] [--sync-memops] [--device-sync-memops] [--memcpy-batch] [--memcpy-during] [--memcpy-any] [--accessed-by] [--legacy-null] [--stream-priority] [--graph-update] [--graph-set-params] [--graph-clone] [--graph-build] [--graph-build-deps] [--graph-host] [--graph-piecewise] [--graph-capture-deps] [--graph-enable] [--graph-mem] [--graph-memset] [--graph-auto-free] [--graph-mem-trim] [--timing-events] [--event-blocking-sync] [--decode-priority] [--cooperative] [--pdl] [--l2-persist] [--l2-reset] [--l2-fetch N] [--l2-ratio N] [--l2-streaming] [--cluster N] [--preferred-cluster N] [--cluster-spread] [--func-cluster-spread] [--cluster-load-balance] [--cluster-must-set] [--required-cluster N] [--max-shared] [--func-max-shared] [--max-l1] [--non-portable-cluster] [--sync-policy auto|spin|yield|blocking] [--device-sync-policy auto|spin|yield|blocking] [--mem-sync-domain default|remote] [--mem-sync-map identity|collapse] [--mem-sync-launch] [--mem-sync-launch-map] [--shared-mem default|four|eight] [--func-shared-mem default|four|eight] [--device-shared-mem default|four|eight] [--portable-cluster default|portable|non-portable] [--optin-shared] [--dynamic-shared N] [--portable-shared default|portable|non-portable] [--nvlink-util] [--device-launch] [--device-updatable] [--kernel-priority N] [--launch-completion] [--programmatic-event] [--stream-attach] [--managed-host] [--prefetch-host] [--wait-value] [--multicast] [--compute-slots N] [--decode-sms N]

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
            print_usage()?;
            Ok(())
        }
        "analyze" => {
            let path = args.next().ok_or("analyze <trace.jsonl>")?;
            let trace = load_trace(&path)?;
            println!("{}", analyze(&trace).report());
            Ok(())
        }
        "replay" => {
            let cfg = parse_cfg(args)?;
            let trace = load_trace(&cfg.path)?;
            print_replay(&trace, cfg.capacity, cfg.lookahead)?;
            Ok(())
        }
        "sim" => {
            let cfg = parse_cfg(args)?;
            let trace = load_trace(&cfg.path)?;
            print_replay(&trace, cfg.capacity, cfg.lookahead)?;
            let profile = load_profile(&cfg.profile)?;
            let prefetch = Prefetch::parse(&cfg.prefetch).map_err(|e| e.to_string())?;
            let row = sim_replay_cfg(&trace, profile, sim_cfg_from(&cfg, prefetch, cfg.max_batch))
                .map_err(|e| e.to_string())?;
            println!("{}", row.line());
            Ok(())
        }
        "schedule" => run_schedule(args),
        "bench" => run_bench(args),
        "workload" => run_workload(args),
        "topology" => run_topology(args),
        "ep" => run_ep(args),
        "place" => run_place(args),
        "remote" => run_remote(args),
        "kv" => run_kv(args),
        "store" => run_store(args),
        other => Err(format!("{USAGE}got {other}")),
    }
}

struct Cfg {
    path: String,
    capacity: usize,
    lookahead: usize,
    expert_bytes: u64,
    activation_bytes: u64,
    profile: String,
    tokens: u32,
    experts: u32,
    hbm_bytes: Option<u64>,
    prefetch: String,
    seq_streams: bool,
    cuda_graphs: bool,
    plan_window: usize,
    plan_threshold: u32,
    max_batch: usize,
    sync_alloc: bool,
    mempool: bool,
    mempool_trim: bool,
    mempool_no_reuse: bool,
    mempool_max: u64,
    shareable: bool,
    mapped: bool,
    managed: bool,
    vmm: bool,
    vmm_page: u64,
    host_func: bool,
    blocking_streams: bool,
    pageable: bool,
    host_register: bool,
    host_register_mapped: bool,
    sync_memops: bool,
    device_sync_memops: bool,
    memcpy_batch: bool,
    memcpy_during: bool,
    memcpy_any: bool,
    accessed_by: bool,
    legacy_null: bool,
    stream_priority: bool,
    graph_update: bool,
    graph_set_params: bool,
    graph_clone: bool,
    graph_build: bool,
    graph_build_deps: bool,
    graph_host: bool,
    graph_piecewise: bool,
    graph_capture_deps: bool,
    graph_enable: bool,
    graph_mem: bool,
    graph_memset: bool,
    graph_auto_free: bool,
    graph_mem_trim: bool,
    timing_events: bool,
    event_blocking_sync: bool,
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
    mem_sync_domain: MemSyncDomain,
    mem_sync_collapse: bool,
    mem_sync_launch: bool,
    mem_sync_launch_map: bool,
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
    wait_value: bool,
    multicast: bool,
    interarrival_ns: u64,
    ttft_slo_ns: Option<u64>,
    itl_slo_ns: Option<u64>,
    prefill_chunk: usize,
    decode_first: bool,
    slo_reject: bool,
    prefix_cache: bool,
    place: String,
}

fn parse_cfg<I>(args: I) -> Result<Cfg, String>
where
    I: IntoIterator<Item = String>,
{
    parse_cfg_profile(args, "h100")
}

fn parse_cfg_profile<I>(args: I, default_profile: &str) -> Result<Cfg, String>
where
    I: IntoIterator<Item = String>,
{
    let mut path = None;
    let mut capacity = 8usize;
    let mut lookahead = 8usize;
    let mut expert_bytes = 4096u64;
    let mut activation_bytes = DECODE_ACTIVATION_BYTES;
    let mut profile = default_profile.to_string();
    let mut tokens = 64u32;
    let mut experts = 16u32;
    let mut hbm_bytes = None;
    let mut prefetch = String::from("none");
    let mut seq_streams = false;
    let mut cuda_graphs = false;
    let mut sync_alloc = false;
    let mut mempool = false;
    let mut mempool_trim = false;
    let mut mempool_no_reuse = false;
    let mut mempool_max = 0u64;
    let mut shareable = false;
    let mut mapped = false;
    let mut managed = false;
    let mut vmm = false;
    let mut vmm_page = 0u64;
    let mut host_func = false;
    let mut blocking_streams = false;
    let mut pageable = false;
    let mut host_register = false;
    let mut host_register_mapped = false;
    let mut sync_memops = false;
    let mut device_sync_memops = false;
    let mut memcpy_batch = false;
    let mut memcpy_during = false;
    let mut memcpy_any = false;
    let mut accessed_by = false;
    let mut legacy_null = false;
    let mut stream_priority = false;
    let mut graph_update = false;
    let mut graph_set_params = false;
    let mut graph_clone = false;
    let mut graph_build = false;
    let mut graph_build_deps = false;
    let mut graph_host = false;
    let mut graph_piecewise = false;
    let mut graph_capture_deps = false;
    let mut graph_enable = false;
    let mut graph_mem = false;
    let mut graph_memset = false;
    let mut graph_auto_free = false;
    let mut graph_mem_trim = false;
    let mut timing_events = false;
    let mut event_blocking_sync = false;
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
    let mut mem_sync_domain = MemSyncDomain::Default;
    let mut mem_sync_collapse = false;
    let mut mem_sync_launch = false;
    let mut mem_sync_launch_map = false;
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
    let mut wait_value = false;
    let mut multicast = false;
    let mut plan_window = 0usize;
    let mut plan_threshold = 500u32;
    let mut max_batch = 0usize;
    let mut interarrival_ns = 0u64;
    let mut ttft_slo_ns = None;
    let mut itl_slo_ns = None;
    let mut prefill_chunk = 0usize;
    let mut decode_first = false;
    let mut slo_reject = false;
    let mut prefix_cache = false;
    let mut place = String::from("none");
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
            "--lookahead" => {
                lookahead = parse_usize("lookahead", &value("lookahead", inline, &mut it)?)?
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
            "--hbm-bytes" => {
                hbm_bytes = Some(parse_u64(
                    "hbm-bytes",
                    &value("hbm-bytes", inline, &mut it)?,
                )?)
            }
            "--prefetch" => prefetch = value("prefetch", inline, &mut it)?,
            "--seq-streams" => seq_streams = switch(&inline),
            "--cuda-graphs" => cuda_graphs = switch(&inline),
            "--sync-alloc" => sync_alloc = switch(&inline),
            "--mempool" => mempool = switch(&inline),
            "--mempool-trim" => mempool_trim = switch(&inline),
            "--mempool-no-reuse" => mempool_no_reuse = switch(&inline),
            "--mempool-max" => {
                mempool_max = parse_mempool_max(&value("mempool-max", inline, &mut it)?)?
            }
            "--shareable" => shareable = switch(&inline),
            "--mapped" => mapped = switch(&inline),
            "--managed" => managed = switch(&inline),
            "--vmm" => vmm = switch(&inline),
            "--vmm-page" => vmm_page = parse_u64("vmm-page", &value("vmm-page", inline, &mut it)?)?,
            "--host-func" => host_func = switch(&inline),
            "--blocking-streams" => blocking_streams = switch(&inline),
            "--pageable" => pageable = switch(&inline),
            "--host-register" => host_register = switch(&inline),
            "--host-register-mapped" => host_register_mapped = switch(&inline),
            "--sync-memops" => sync_memops = switch(&inline),
            "--device-sync-memops" => device_sync_memops = switch(&inline),
            "--memcpy-batch" => memcpy_batch = switch(&inline),
            "--memcpy-during" => memcpy_during = switch(&inline),
            "--memcpy-any" => memcpy_any = switch(&inline),
            "--accessed-by" => accessed_by = switch(&inline),
            "--legacy-null" => legacy_null = switch(&inline),
            "--stream-priority" => stream_priority = switch(&inline),
            "--decode-priority" => decode_priority = switch(&inline),
            "--cooperative" => cooperative = switch(&inline),
            "--pdl" => pdl = switch(&inline),
            "--l2-persist" => l2_persist = switch(&inline),
            "--l2-reset" => l2_reset = switch(&inline),
            "--l2-fetch" => l2_fetch = parse_l2_fetch(&value("l2-fetch", inline, &mut it)?)?,
            "--l2-ratio" => l2_ratio = parse_l2_ratio(&value("l2-ratio", inline, &mut it)?)?,
            "--l2-streaming" => l2_streaming = switch(&inline),
            "--cluster" => cluster = parse_cluster(&value("cluster", inline, &mut it)?)?,
            "--preferred-cluster" => {
                preferred_cluster =
                    parse_preferred_cluster(&value("preferred-cluster", inline, &mut it)?)?
            }
            "--cluster-spread" => cluster_spread = switch(&inline),
            "--func-cluster-spread" => func_cluster_spread = switch(&inline),
            "--cluster-load-balance" => cluster_load_balance = switch(&inline),
            "--cluster-must-set" => cluster_must_set = switch(&inline),
            "--required-cluster" => {
                required_cluster =
                    parse_required_cluster(&value("required-cluster", inline, &mut it)?)?
            }
            "--max-shared" => max_shared = switch(&inline),
            "--func-max-shared" => func_max_shared = switch(&inline),
            "--max-l1" => max_l1 = switch(&inline),
            "--non-portable-cluster" => non_portable_cluster = switch(&inline),
            "--sync-policy" => {
                sync_policy = parse_sync_policy(&value("sync-policy", inline, &mut it)?)?
            }
            "--device-sync-policy" => {
                device_sync_policy =
                    parse_device_sync_policy(&value("device-sync-policy", inline, &mut it)?)?
            }
            "--mem-sync-domain" => {
                mem_sync_domain =
                    parse_mem_sync_domain(&value("mem-sync-domain", inline, &mut it)?)?
            }
            "--mem-sync-map" => {
                mem_sync_collapse = parse_mem_sync_map(&value("mem-sync-map", inline, &mut it)?)?
            }
            "--mem-sync-launch" => mem_sync_launch = switch(&inline),
            "--mem-sync-launch-map" => mem_sync_launch_map = switch(&inline),
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
            "--optin-shared" => optin_shared = switch(&inline),
            "--dynamic-shared" => {
                dynamic_shared = parse_dynamic_shared(&value("dynamic-shared", inline, &mut it)?)?
            }
            "--portable-shared" => {
                portable_shared =
                    parse_portable_shared(&value("portable-shared", inline, &mut it)?)?
            }
            "--nvlink-util" => nvlink_util_centric = switch(&inline),
            "--device-updatable" => device_updatable = switch(&inline),
            "--kernel-priority" => {
                kernel_priority = Some(parse_kernel_priority(&value(
                    "kernel-priority",
                    inline,
                    &mut it,
                )?)?)
            }
            "--device-launch" => device_launch = switch(&inline),
            "--launch-completion" => launch_completion = switch(&inline),
            "--programmatic-event" => programmatic_event = switch(&inline),
            "--stream-attach" => stream_attach = switch(&inline),
            "--managed-host" => managed_host = switch(&inline),
            "--prefetch-host" => prefetch_host = switch(&inline),
            "--wait-value" => wait_value = switch(&inline),
            "--multicast" => multicast = switch(&inline),
            "--graph-update" => graph_update = switch(&inline),
            "--graph-set-params" => graph_set_params = switch(&inline),
            "--graph-clone" => graph_clone = switch(&inline),
            "--graph-build" => graph_build = switch(&inline),
            "--graph-build-deps" => graph_build_deps = switch(&inline),
            "--graph-host" => graph_host = switch(&inline),
            "--graph-piecewise" => graph_piecewise = switch(&inline),
            "--graph-capture-deps" => graph_capture_deps = switch(&inline),
            "--graph-enable" => graph_enable = switch(&inline),
            "--graph-mem" => graph_mem = switch(&inline),
            "--graph-memset" => graph_memset = switch(&inline),
            "--graph-auto-free" => graph_auto_free = switch(&inline),
            "--graph-mem-trim" => graph_mem_trim = switch(&inline),
            "--timing-events" => timing_events = switch(&inline),
            "--event-blocking-sync" => event_blocking_sync = switch(&inline),
            "--compute-slots" => {
                compute_slots = parse_compute_slots(&value("compute-slots", inline, &mut it)?)?
            }
            "--decode-sms" => {
                decode_sm_permille = parse_decode_sms(&value("decode-sms", inline, &mut it)?)?
            }
            "--plan-window" => {
                plan_window = parse_usize("plan-window", &value("plan-window", inline, &mut it)?)?
            }
            "--plan-threshold" => {
                plan_threshold =
                    parse_u32("plan-threshold", &value("plan-threshold", inline, &mut it)?)?
            }
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
            "--decode-first" => decode_first = switch(&inline),
            "--slo-reject" => slo_reject = switch(&inline),
            "--prefix-cache" => prefix_cache = switch(&inline),
            "--place" => place = value("place", inline, &mut it)?,
            flag if flag.starts_with('-') => return Err(format!("unknown flag {flag}\n{USAGE}")),
            other => {
                if path.is_some() {
                    return Err(format!("unexpected argument {other}\n{USAGE}"));
                }
                path = Some(other.to_string());
            }
        }
    }
    if vmm_page > 0 || multicast {
        vmm = true;
    }
    if multicast && accessed_by {
        return Err("choose one of --multicast, --accessed-by".into());
    }
    if decode_priority {
        stream_priority = true;
    }
    if graph_mem && graph_auto_free {
        return Err("choose one of --graph-mem, --graph-auto-free".into());
    }
    if graph_memset && !graph_mem {
        return Err("--graph-memset needs --graph-mem".into());
    }
    if graph_update && graph_set_params {
        return Err("choose one of --graph-update, --graph-set-params".into());
    }
    if graph_update && device_launch {
        return Err("choose one of --graph-update, --device-launch".into());
    }
    if graph_update && device_updatable {
        return Err("choose one of --graph-update, --device-updatable".into());
    }
    if device_launch && (graph_mem || graph_auto_free) {
        return Err("device-launch cannot graph-mem".into());
    }
    if graph_build && graph_piecewise {
        return Err("choose one of --graph-build, --graph-piecewise".into());
    }
    if graph_build_deps && !graph_build {
        return Err("--graph-build-deps needs --graph-build".into());
    }
    if graph_host && !graph_build {
        return Err("--graph-host needs --graph-build".into());
    }
    if graph_capture_deps && !graph_piecewise {
        return Err("--graph-capture-deps needs --graph-piecewise".into());
    }
    if graph_enable && device_launch {
        return Err("graph-enable cannot device-launch".into());
    }
    if launch_completion && device_launch {
        return Err("launch-completion cannot device-launch".into());
    }
    if programmatic_event && device_launch {
        return Err("programmatic-event cannot device-launch".into());
    }
    if stream_attach || managed_host || prefetch_host {
        managed = true;
    }
    if stream_attach && seq_streams {
        return Err("stream-attach cannot seq-streams".into());
    }
    if pdl && cooperative {
        return Err("choose one of --pdl, --cooperative".into());
    }
    if event_blocking_sync {
        timing_events = true;
    }
    if shareable {
        mempool = true;
    }
    if mempool_trim {
        mempool = true;
    }
    if mempool_no_reuse {
        mempool = true;
    }
    if mempool_max != 0 {
        mempool = true;
    }
    if mempool_trim && sync_alloc {
        return Err("mempool-trim needs cudaMallocAsync".into());
    }
    if mempool_no_reuse && sync_alloc {
        return Err("mempool-no-reuse needs cudaMallocAsync".into());
    }
    if mempool_max != 0 && sync_alloc {
        return Err("mempool-max needs cudaMallocAsync".into());
    }
    if shareable && (sync_alloc || mapped || managed || vmm) {
        return Err("shareable needs cudaMallocAsync".into());
    }
    if host_register_mapped {
        mapped = true;
    }
    if host_register {
        pageable = true;
    }
    if host_register && host_register_mapped {
        return Err("choose one of --host-register, --host-register-mapped".into());
    }
    if host_register && (mapped || managed) {
        return Err("host-register needs pinned/vmm H2D".into());
    }
    if memcpy_batch
        && (pageable || sync_alloc || mapped || managed || sync_memops || device_sync_memops)
    {
        return Err("memcpy-batch needs async pinned/vmm H2D".into());
    }
    if memcpy_during && !memcpy_batch {
        return Err("--memcpy-during needs --memcpy-batch".into());
    }
    if memcpy_any && !memcpy_batch {
        return Err("--memcpy-any needs --memcpy-batch".into());
    }
    if memcpy_any && memcpy_during {
        return Err("choose one of --memcpy-any, --memcpy-during".into());
    }
    if sync_memops && mapped {
        return Err("sync-memops needs device memcpy".into());
    }
    if device_sync_memops && mapped {
        return Err("device-sync-memops needs device memcpy".into());
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
    if mem_sync_collapse && mem_sync_domain != MemSyncDomain::Remote {
        return Err("--mem-sync-map collapse needs --mem-sync-domain remote".into());
    }
    if mem_sync_launch && mem_sync_domain != MemSyncDomain::Remote {
        return Err("--mem-sync-launch needs --mem-sync-domain remote".into());
    }
    if mem_sync_launch_map && mem_sync_domain != MemSyncDomain::Remote {
        return Err("--mem-sync-launch-map needs --mem-sync-domain remote".into());
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
        path: path.ok_or("missing trace.jsonl or workload name")?,
        capacity,
        lookahead,
        expert_bytes,
        activation_bytes,
        profile,
        tokens,
        experts,
        hbm_bytes,
        prefetch,
        seq_streams,
        cuda_graphs,
        plan_window,
        plan_threshold,
        max_batch,
        sync_alloc,
        mempool,
        mempool_trim,
        mempool_no_reuse,
        mempool_max,
        shareable,
        mapped,
        managed,
        vmm,
        vmm_page,
        host_func,
        blocking_streams,
        pageable,
        host_register,
        host_register_mapped,
        sync_memops,
        device_sync_memops,
        memcpy_batch,
        memcpy_during,
        memcpy_any,
        accessed_by,
        legacy_null,
        stream_priority,
        graph_update,
        graph_set_params,
        graph_clone,
        graph_build,
        graph_build_deps,
        graph_host,
        graph_piecewise,
        graph_capture_deps,
        graph_enable,
        graph_mem,
        graph_memset,
        graph_auto_free,
        graph_mem_trim,
        timing_events,
        event_blocking_sync,
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
        mem_sync_domain,
        mem_sync_collapse,
        mem_sync_launch,
        mem_sync_launch_map,
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
        wait_value,
        multicast,
        interarrival_ns,
        ttft_slo_ns,
        itl_slo_ns,
        prefill_chunk,
        decode_first,
        slo_reject,
        prefix_cache,
        place,
    })
}

fn sim_cfg_from(cfg: &Cfg, prefetch: Prefetch, max_batch: usize) -> SimCfg {
    SimCfg {
        slots: cfg.capacity,
        policy: Policy::Lru,
        bytes_per_expert: cfg.expert_bytes,
        lookahead: cfg.lookahead,
        prefetch,
        seq_streams: cfg.seq_streams,
        cuda_graphs: cfg.cuda_graphs
            || cfg.graph_build
            || cfg.graph_piecewise
            || cfg.graph_enable
            || cfg.graph_mem
            || cfg.graph_auto_free
            || cfg.graph_set_params
            || cfg.device_launch
            || cfg.device_updatable,
        plan_window: cfg.plan_window,
        plan_threshold: cfg.plan_threshold,
        max_batch,
        sync_alloc: cfg.sync_alloc,
        mempool: cfg.mempool,
        mempool_trim: cfg.mempool_trim,
        mempool_no_reuse: cfg.mempool_no_reuse,
        mempool_max: cfg.mempool_max,
        shareable: cfg.shareable,
        mapped: cfg.mapped,
        managed: cfg.managed,
        vmm: cfg.vmm,
        vmm_page: cfg.vmm_page,
        host_func: cfg.host_func,
        blocking_streams: cfg.blocking_streams,
        pageable: cfg.pageable,
        host_register: cfg.host_register,
        host_register_mapped: cfg.host_register_mapped,
        sync_memops: cfg.sync_memops,
        device_sync_memops: cfg.device_sync_memops,
        memcpy_batch: cfg.memcpy_batch,
        memcpy_during: cfg.memcpy_during,
        memcpy_any: cfg.memcpy_any,
        accessed_by: cfg.accessed_by,
        legacy_null: cfg.legacy_null,
        stream_priority: cfg.stream_priority,
        graph_update: cfg.graph_update,
        graph_set_params: cfg.graph_set_params,
        graph_clone: cfg.graph_clone,
        graph_build: cfg.graph_build,
        graph_build_deps: cfg.graph_build_deps,
        graph_host: cfg.graph_host,
        graph_piecewise: cfg.graph_piecewise,
        graph_capture_deps: cfg.graph_capture_deps,
        graph_enable: cfg.graph_enable,
        graph_mem: cfg.graph_mem,
        graph_memset: cfg.graph_memset,
        graph_auto_free: cfg.graph_auto_free,
        graph_mem_trim: cfg.graph_mem_trim,
        compute_slots: cfg.compute_slots,
        decode_sm_permille: cfg.decode_sm_permille,
        decode_priority: cfg.decode_priority,
        cooperative: cfg.cooperative,
        pdl: cfg.pdl,
        l2_persist: cfg.l2_persist || cfg.l2_reset || cfg.l2_fetch != 0 || cfg.l2_ratio != 0,
        l2_reset: cfg.l2_reset,
        l2_fetch: cfg.l2_fetch,
        l2_ratio: cfg.l2_ratio,
        l2_streaming: cfg.l2_streaming,
        cluster: cfg.cluster,
        preferred_cluster: cfg.preferred_cluster,
        cluster_spread: cfg.cluster_spread,
        func_cluster_spread: cfg.func_cluster_spread,
        cluster_load_balance: cfg.cluster_load_balance,
        cluster_must_set: cfg.cluster_must_set,
        required_cluster: cfg.required_cluster,
        max_shared: cfg.max_shared,
        func_max_shared: cfg.func_max_shared,
        max_l1: cfg.max_l1,
        non_portable_cluster: cfg.non_portable_cluster,
        sync_policy: cfg.sync_policy,
        device_sync_policy: cfg.device_sync_policy,
        mem_sync_domain: cfg.mem_sync_domain,
        mem_sync_collapse: cfg.mem_sync_collapse,
        mem_sync_launch: cfg.mem_sync_launch,
        mem_sync_launch_map: cfg.mem_sync_launch_map,
        shared_mem: cfg.shared_mem,
        func_shared_mem: cfg.func_shared_mem,
        device_shared_mem: cfg.device_shared_mem,
        portable_cluster: cfg.portable_cluster,
        optin_shared: cfg.optin_shared,
        dynamic_shared: cfg.dynamic_shared,
        portable_shared: cfg.portable_shared,
        nvlink_util_centric: cfg.nvlink_util_centric,
        device_updatable: cfg.device_updatable,
        kernel_priority: cfg.kernel_priority,
        device_launch: cfg.device_launch,
        launch_completion: cfg.launch_completion,
        programmatic_event: cfg.programmatic_event,
        stream_attach: cfg.stream_attach,
        managed_host: cfg.managed_host,
        prefetch_host: cfg.prefetch_host,
        wait_value: cfg.wait_value,
        multicast: cfg.multicast,
    }
}

fn switch(inline: &Option<String>) -> bool {
    !matches!(inline.as_deref(), Some("0" | "false"))
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

fn parse_compute_slots(s: &str) -> Result<u8, String> {
    let n = s
        .parse::<u8>()
        .map_err(|_| format!("invalid compute-slots {s:?}"))?;
    if n == 0 {
        return Err("compute-slots must be > 0".into());
    }
    Ok(n)
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

fn parse_mempool_max(s: &str) -> Result<u64, String> {
    let n = parse_u64("mempool-max", s)?;
    if n == 0 {
        return Err("mempool-max must be > 0".into());
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

fn parse_mem_sync_domain(s: &str) -> Result<MemSyncDomain, String> {
    MemSyncDomain::parse(s).map_err(|_| format!("unknown mem-sync-domain {s}"))
}

fn parse_mem_sync_map(s: &str) -> Result<bool, String> {
    match s {
        "identity" => Ok(false),
        "collapse" => Ok(true),
        _ => Err(format!("unknown mem-sync-map {s}")),
    }
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

fn parse_decode_sms(s: &str) -> Result<u16, String> {
    let n = s
        .parse::<u16>()
        .map_err(|_| format!("invalid decode-sms {s:?}"))?;
    if n == 0 || n > 1000 {
        return Err("decode-sms must be 1..=1000".into());
    }
    Ok(n)
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

fn print_replay(trace: &Trace, capacity: usize, lookahead: usize) -> Result<(), String> {
    println!("{}", analyze(trace).report());
    print!("{}", format_table(&compare(trace, capacity, lookahead)));
    Ok(())
}

fn print_usage() -> Result<(), String> {
    let mut out = std::io::stdout();
    out.write_all(USAGE.as_bytes()).map_err(|e| e.to_string())
}

fn run_bench<I>(args: I) -> Result<(), String>
where
    I: IntoIterator<Item = String>,
{
    let cfg = parse_cfg(args)?;
    if cfg.path == "adversarial" {
        let profile = load_profile(&cfg.profile)?;
        let rows = adversarial_suite(cfg.tokens, cfg.experts, cfg.capacity, profile)
            .map_err(|e| e.to_string())?;
        for row in rows {
            print!("{}", row.render());
        }
        return Ok(());
    }
    let trace = load_trace(&cfg.path)?;
    let profile = load_profile(&cfg.profile)?;
    let row = report(
        &cfg.path,
        &trace,
        cfg.capacity,
        cfg.lookahead,
        Some(profile),
        cfg.expert_bytes,
    )
    .map_err(|e| e.to_string())?;
    print!("{}", row.render());
    Ok(())
}

fn run_workload<I>(args: I) -> Result<(), String>
where
    I: IntoIterator<Item = String>,
{
    let cfg = parse_cfg(args)?;
    let kind = parse_workload(&cfg.path)?;
    let trace = generate(kind, cfg.tokens, cfg.experts, 1, 1);
    let profile = load_profile(&cfg.profile)?;
    let row = report(
        kind.name(),
        &trace,
        cfg.capacity,
        cfg.lookahead,
        Some(profile),
        cfg.expert_bytes,
    )
    .map_err(|e| e.to_string())?;
    print!("{}", row.render());
    Ok(())
}

fn parse_workload(name: &str) -> Result<Workload, String> {
    Workload::from_name(name).ok_or_else(|| format!("unknown workload {name}\n{USAGE}"))
}

fn run_topology<I>(args: I) -> Result<(), String>
where
    I: IntoIterator<Item = String>,
{
    let mut bytes = 8u64 << 20;
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        let (key, inline) = match arg.split_once('=') {
            Some((k, v)) => (k.to_string(), Some(v.to_string())),
            None => (arg, None),
        };
        match key.as_str() {
            "--bytes" | "--expert-bytes" => {
                bytes = parse_u64("bytes", &value("bytes", inline, &mut it)?)?
            }
            flag => return Err(format!("unknown flag {flag}\n{USAGE}")),
        }
    }
    let rows = topology_suite(bytes).map_err(|e| e.to_string())?;
    for row in rows {
        println!("{}", row.line());
    }
    Ok(())
}

fn run_ep<I>(args: I) -> Result<(), String>
where
    I: IntoIterator<Item = String>,
{
    let cfg = parse_cfg(args)?;
    let trace = load_trace(&cfg.path)?;
    let mut profile = load_profile(&cfg.profile)?;
    if let Some(bytes) = cfg.hbm_bytes {
        profile = profile.restrict_hbm(bytes);
    }
    let row = compare_ep(
        &trace,
        profile,
        cfg.capacity,
        cfg.expert_bytes,
        cfg.lookahead,
    )
    .map_err(|e| e.to_string())?;
    println!("{}", row.line());
    Ok(())
}

fn run_place<I>(args: I) -> Result<(), String>
where
    I: IntoIterator<Item = String>,
{
    let mut path = None;
    let mut gpus = 8u16;
    let mut hot_pt = 200u32;
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        let (key, inline) = match arg.split_once('=') {
            Some((k, v)) => (k.to_string(), Some(v.to_string())),
            None => (arg, None),
        };
        match key.as_str() {
            "--gpus" => gpus = parse_u16("gpus", &value("gpus", inline, &mut it)?)?,
            "--hot-pt" => hot_pt = parse_u32("hot-pt", &value("hot-pt", inline, &mut it)?)?,
            flag if flag.starts_with('-') => return Err(format!("unknown flag {flag}\n{USAGE}")),
            other => {
                if path.is_some() {
                    return Err(format!("unexpected argument {other}\n{USAGE}"));
                }
                path = Some(other.to_string());
            }
        }
    }
    let path = path.ok_or("place <trace.jsonl>")?;
    let trace = load_trace(&path)?;
    let stripe = striped(&trace, gpus);
    let colo = colocated(&trace, gpus);
    let hot = with_hot_replicas(colo.clone(), &trace, gpus, hot_pt);
    println!("striped {}", stripe.line());
    println!("colocated {}", colo.line());
    println!("replicas {}", hot.line());
    Ok(())
}

fn run_remote<I>(args: I) -> Result<(), String>
where
    I: IntoIterator<Item = String>,
{
    let cfg = parse_cfg_profile(args, "2node-rdma")?;
    let trace = load_trace(&cfg.path)?;
    let profile = load_profile(&cfg.profile)?;
    let n = u16::try_from(profile.n_gpus()).unwrap_or(1).max(1);
    let map = striped(&trace, n);
    let local =
        sim_placed(&trace, profile.clone(), cfg.expert_bytes, &map).map_err(|e| e.to_string())?;
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

fn run_store<I>(args: I) -> Result<(), String>
where
    I: IntoIterator<Item = String>,
{
    let cfg = parse_cfg(args)?;
    let trace = load_trace(&cfg.path)?;
    let profile = load_profile(&cfg.profile)?;
    let fill = GpuFill::from_flags(cfg.mapped, cfg.managed, cfg.vmm).map_err(|e| e.to_string())?;
    let prefetch = Prefetch::parse(&cfg.prefetch).map_err(|e| e.to_string())?;
    let row = store_replay_cfg(
        &trace,
        profile,
        StoreReplayCfg {
            slots: cfg.capacity,
            bytes_per_expert: cfg.expert_bytes,
            fill,
            gpu: GpuStoreCfg {
                host_func: cfg.host_func,
                blocking_streams: cfg.blocking_streams,
                sync_alloc: cfg.sync_alloc,
                mempool: cfg.mempool,
                mempool_trim: cfg.mempool_trim,
                mempool_no_reuse: cfg.mempool_no_reuse,
                mempool_max: cfg.mempool_max,
                shareable: cfg.shareable,
                vmm_page: cfg.vmm_page,
                pageable: cfg.pageable,
                host_register: cfg.host_register,
                host_register_mapped: cfg.host_register_mapped,
                sync_memops: cfg.sync_memops,
                device_sync_memops: cfg.device_sync_memops,
                memcpy_batch: cfg.memcpy_batch,
                memcpy_during: cfg.memcpy_during,
                memcpy_any: cfg.memcpy_any,
                accessed_by: cfg.accessed_by,
                legacy_null: cfg.legacy_null,
                stream_priority: cfg.stream_priority,
                graph_update: cfg.graph_update,
                graph_set_params: cfg.graph_set_params,
                graph_clone: cfg.graph_clone,
                graph_build: cfg.graph_build,
                graph_build_deps: cfg.graph_build_deps,
                graph_host: cfg.graph_host,
                graph_piecewise: cfg.graph_piecewise,
                graph_capture_deps: cfg.graph_capture_deps,
                graph_enable: cfg.graph_enable,
                graph_mem: cfg.graph_mem,
                graph_memset: cfg.graph_memset,
                graph_auto_free: cfg.graph_auto_free,
                graph_mem_trim: cfg.graph_mem_trim,
                timing_events: cfg.timing_events || cfg.event_blocking_sync,
                event_blocking_sync: cfg.event_blocking_sync,
                seq_streams: cfg.seq_streams,
                kv_sim: false,
                decode_priority: cfg.decode_priority,
                cooperative: cfg.cooperative,
                pdl: cfg.pdl,
                l2_persist: cfg.l2_persist
                    || cfg.l2_reset
                    || cfg.l2_fetch != 0
                    || cfg.l2_ratio != 0,
                l2_reset: cfg.l2_reset,
                l2_fetch: cfg.l2_fetch,
                l2_ratio: cfg.l2_ratio,
                l2_streaming: cfg.l2_streaming,
                cluster: cfg.cluster,
                preferred_cluster: cfg.preferred_cluster,
                cluster_spread: cfg.cluster_spread,
                func_cluster_spread: cfg.func_cluster_spread,
                cluster_load_balance: cfg.cluster_load_balance,
                cluster_must_set: cfg.cluster_must_set,
                required_cluster: cfg.required_cluster,
                max_shared: cfg.max_shared,
                func_max_shared: cfg.func_max_shared,
                max_l1: cfg.max_l1,
                non_portable_cluster: cfg.non_portable_cluster,
                sync_policy: cfg.sync_policy,
                device_sync_policy: cfg.device_sync_policy,
                mem_sync_domain: cfg.mem_sync_domain,
                mem_sync_collapse: cfg.mem_sync_collapse,
                mem_sync_launch: cfg.mem_sync_launch,
                mem_sync_launch_map: cfg.mem_sync_launch_map,
                shared_mem: cfg.shared_mem,
                func_shared_mem: cfg.func_shared_mem,
                device_shared_mem: cfg.device_shared_mem,
                portable_cluster: cfg.portable_cluster,
                optin_shared: cfg.optin_shared,
                dynamic_shared: cfg.dynamic_shared,
                portable_shared: cfg.portable_shared,
                nvlink_util_centric: cfg.nvlink_util_centric,
                device_updatable: cfg.device_updatable,
                kernel_priority: cfg.kernel_priority,
                device_launch: cfg.device_launch,
                launch_completion: cfg.launch_completion,
                programmatic_event: cfg.programmatic_event,
                stream_attach: cfg.stream_attach,
                managed_host: cfg.managed_host,
                prefetch_host: cfg.prefetch_host,
                wait_value: cfg.wait_value,
                multicast: cfg.multicast,
                compute_slots: cfg.compute_slots,
                decode_sm_permille: cfg.decode_sm_permille,
            },
            prefetch,
            plan_window: cfg.plan_window,
            plan_threshold: cfg.plan_threshold,
        },
    )
    .map_err(|e| e.to_string())?;
    println!("{}", row.line());
    Ok(())
}

fn run_kv<I>(args: I) -> Result<(), String>
where
    I: IntoIterator<Item = String>,
{
    let mut pages = 8u32;
    let mut page_bytes = 4096u64;
    let mut capacity = 2usize;
    let mut tokens = 64u32;
    let mut profile = String::from("h100");
    let mut fill = KvFill::H2d;
    let mut sequences = 1u32;
    let mut row_width = 0u64;
    let mut pitch = 0u64;
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        let (key, inline) = match arg.split_once('=') {
            Some((k, v)) => (k.to_string(), Some(v.to_string())),
            None => (arg, None),
        };
        match key.as_str() {
            "--pages" => pages = parse_u32("pages", &value("pages", inline, &mut it)?)?,
            "--page-bytes" => {
                page_bytes = parse_u64("page-bytes", &value("page-bytes", inline, &mut it)?)?
            }
            "--capacity" | "-c" => {
                capacity = parse_usize("capacity", &value("capacity", inline, &mut it)?)?
            }
            "--tokens" => tokens = parse_u32("tokens", &value("tokens", inline, &mut it)?)?,
            "--profile" => profile = value("profile", inline, &mut it)?,
            "--fill" => {
                fill = KvFill::parse(&value("fill", inline, &mut it)?).map_err(|e| e.to_string())?
            }
            "--sequences" => {
                sequences = parse_u32("sequences", &value("sequences", inline, &mut it)?)?
            }
            "--row-width" => {
                row_width = parse_u64("row-width", &value("row-width", inline, &mut it)?)?
            }
            "--pitch" => pitch = parse_u64("pitch", &value("pitch", inline, &mut it)?)?,
            flag if flag.starts_with('-') => return Err(format!("unknown flag {flag}\n{USAGE}")),
            other => return Err(format!("unexpected argument {other}\n{USAGE}")),
        }
    }
    if pages == 0 {
        return Err("pages must be > 0".to_string());
    }
    let accesses = cycling_pages(pages, tokens);
    let hw = load_profile(&profile)?;
    let row = kv_paged(
        &accesses,
        hw,
        KvCfg {
            page_bytes,
            slots: capacity,
            fill,
            sequences,
            row_width,
            pitch,
        },
    )
    .map_err(|e| e.to_string())?;
    println!("{}", row.line());
    Ok(())
}

fn parse_u16(name: &str, s: &str) -> Result<u16, String> {
    s.parse::<u16>()
        .map_err(|_| format!("invalid {name} {s:?}"))
}

fn run_schedule<I>(args: I) -> Result<(), String>
where
    I: IntoIterator<Item = String>,
{
    let cfg = parse_cfg(args)?;
    let trace = load_trace(&cfg.path)?;
    println!("{}", analyze(&trace).report());
    let profile = load_profile(&cfg.profile)?;
    let prefetch = Prefetch::parse(&cfg.prefetch).map_err(|e| e.to_string())?;
    let n_gpus = u16::try_from(profile.n_gpus()).unwrap_or(1).max(1);
    let sim_cfg = sim_cfg_from(&cfg, prefetch, 0);
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
