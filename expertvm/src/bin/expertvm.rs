//! `expertvm analyze | replay | sim` — traces in, measured tables out.

use expertvm::{
    adversarial_suite, analyze, colocated, compare, compare_ep, format_table, generate, report,
    schedule_placed, schedule_remote, sim_placed, sim_remote_home_cfg, sim_replay_cfg, striped,
    topology_suite, with_hot_replicas, Policy, Prefetch, SchedCfg, SimCfg, Trace, Workload,
    DECODE_ACTIVATION_BYTES,
};
use gpu_sim::HardwareProfile;
use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::process::ExitCode;

const USAGE: &str = "\
usage: expertvm <command> [args]
  analyze  <trace.jsonl>
  replay   <trace.jsonl> [--capacity N] [--lookahead N]
  sim      <trace.jsonl> [--capacity N] [--lookahead N] [--expert-bytes N] [--profile NAME] [--prefetch none|copy-forward|markov|both] [--seq-streams] [--cuda-graphs] [--plan-window N] [--plan-threshold N] [--max-batch N]
  schedule <trace.jsonl> [--capacity N] [--lookahead N] [--expert-bytes N] [--profile NAME] [--prefetch none|copy-forward|markov|both] [--seq-streams] [--cuda-graphs] [--plan-window N] [--plan-threshold N] [--max-batch N] [--interarrival-ns N] [--ttft-slo-ns N] [--itl-slo-ns N] [--prefill-chunk N] [--decode-first] [--slo-reject] [--place none|striped|colocated|replicas|remote] [--activation-bytes N]
  bench    <trace.jsonl> [--capacity N] [--lookahead N] [--expert-bytes N] [--profile NAME]
  bench    adversarial [--tokens N] [--experts N] [--capacity N] [--profile NAME]
  workload <NAME> [--tokens N] [--experts N] [--capacity N] [--profile NAME]
  topology [--bytes N]
  ep       <trace.jsonl> [--capacity N] [--expert-bytes N] [--hbm-bytes N] [--profile NAME]
  place    <trace.jsonl> [--gpus N] [--hot-pt N]
  remote   <trace.jsonl> [--expert-bytes N] [--activation-bytes N] [--profile NAME]

NAME: uniform, hotset, shifting-hotset, thrash, coding, chat, long-context,
      prefill-heavy, decode-heavy, batch, prefill-batch
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
            let row = sim_replay_cfg(
                &trace,
                profile,
                SimCfg {
                    slots: cfg.capacity,
                    policy: Policy::Lru,
                    bytes_per_expert: cfg.expert_bytes,
                    lookahead: cfg.lookahead,
                    prefetch,
                    seq_streams: cfg.seq_streams,
                    cuda_graphs: cfg.cuda_graphs,
                    plan_window: cfg.plan_window,
                    plan_threshold: cfg.plan_threshold,
                    max_batch: cfg.max_batch,
                },
            )
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
    interarrival_ns: u64,
    ttft_slo_ns: Option<u64>,
    itl_slo_ns: Option<u64>,
    prefill_chunk: usize,
    decode_first: bool,
    slo_reject: bool,
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
    let mut plan_window = 0usize;
    let mut plan_threshold = 500u32;
    let mut max_batch = 0usize;
    let mut interarrival_ns = 0u64;
    let mut ttft_slo_ns = None;
    let mut itl_slo_ns = None;
    let mut prefill_chunk = 0usize;
    let mut decode_first = false;
    let mut slo_reject = false;
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
            "--seq-streams" => {
                seq_streams = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--cuda-graphs" => {
                cuda_graphs = !matches!(inline.as_deref(), Some("0" | "false"));
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
            "--decode-first" => {
                decode_first = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--slo-reject" => {
                slo_reject = !matches!(inline.as_deref(), Some("0" | "false"));
            }
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
        interarrival_ns,
        ttft_slo_ns,
        itl_slo_ns,
        prefill_chunk,
        decode_first,
        slo_reject,
        place,
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
    let sim_cfg = SimCfg {
        slots: cfg.capacity,
        policy: Policy::Lru,
        bytes_per_expert: cfg.expert_bytes,
        lookahead: cfg.lookahead,
        prefetch,
        seq_streams: cfg.seq_streams,
        cuda_graphs: cfg.cuda_graphs,
        plan_window: cfg.plan_window,
        plan_threshold: cfg.plan_threshold,
        max_batch: 0,
    };
    let sched = SchedCfg {
        max_batch: cfg.max_batch,
        interarrival_ns: cfg.interarrival_ns,
        ttft_slo_ns: cfg.ttft_slo_ns,
        itl_slo_ns: cfg.itl_slo_ns,
        prefill_chunk_layers: cfg.prefill_chunk,
        decode_first: cfg.decode_first,
        slo_reject: cfg.slo_reject,
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
