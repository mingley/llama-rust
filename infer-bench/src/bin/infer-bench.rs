//! `infer-bench adversarial | trace` — measured hit rates and sim scores.

use infer_bench::{
    adversarial_suite, colocated, report, schedule_placed, sim_placed, sim_remote_home_cfg,
    striped, topology_suite, with_hot_replicas, HardwareProfile, SchedCfg, SimCfg, Trace, Workload,
    DECODE_ACTIVATION_BYTES,
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
  schedule <trace.jsonl> [--capacity N] [--profile NAME] [--expert-bytes N] [--max-batch N] [--interarrival-ns N] [--ttft-slo-ns N] [--itl-slo-ns N] [--prefill-chunk N] [--decode-first] [--slo-reject] [--place none|striped|colocated|replicas]

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
                        "unknown --place {other} (none|striped|colocated|replicas)"
                    ))
                }
            };
            let row = schedule_placed(
                &trace,
                profile,
                SimCfg::lru(cfg.capacity, cfg.expert_bytes, 8),
                SchedCfg {
                    max_batch: cfg.max_batch,
                    interarrival_ns: cfg.interarrival_ns,
                    ttft_slo_ns: cfg.ttft_slo_ns,
                    itl_slo_ns: cfg.itl_slo_ns,
                    prefill_chunk_layers: cfg.prefill_chunk,
                    decode_first: cfg.decode_first,
                    slo_reject: cfg.slo_reject,
                },
                map.as_ref(),
            )
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
    place: String,
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
            "--slo-reject" => {
                slo_reject = !matches!(inline.as_deref(), Some("0" | "false"));
            }
            "--place" => place = value("place", inline, &mut it)?,
            "--bytes" => expert_bytes = parse_u64("bytes", &value("bytes", inline, &mut it)?)?,
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
