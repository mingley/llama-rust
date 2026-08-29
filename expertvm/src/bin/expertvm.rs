//! `expertvm analyze | replay | sim` — traces in, measured tables out.

use expertvm::{
    adversarial_suite, analyze, compare, format_table, generate, report, sim_replay, Policy, Trace,
    Workload,
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
  sim      <trace.jsonl> [--capacity N] [--lookahead N] [--expert-bytes N] [--profile NAME]
  bench    <trace.jsonl> [--capacity N] [--lookahead N] [--expert-bytes N] [--profile NAME]
  bench    adversarial [--tokens N] [--experts N] [--capacity N] [--profile NAME]
  workload <NAME> [--tokens N] [--experts N] [--capacity N] [--profile NAME]

NAME: uniform, hotset, shifting-hotset, thrash, coding, chat, long-context,
      prefill-heavy, decode-heavy, batch
profiles: h100 (default), h200, 8xh100, cheap, or a path to a .profile file
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
            let row = sim_replay(
                &trace,
                profile,
                cfg.capacity,
                Policy::Lru,
                cfg.expert_bytes,
                cfg.lookahead,
            )
            .map_err(|e| e.to_string())?;
            println!("{}", row.line());
            Ok(())
        }
        "bench" => run_bench(args),
        "workload" => run_workload(args),
        other => Err(format!("{USAGE}got {other}")),
    }
}

struct Cfg {
    path: String,
    capacity: usize,
    lookahead: usize,
    expert_bytes: u64,
    profile: String,
    tokens: u32,
    experts: u32,
}

fn parse_cfg<I>(args: I) -> Result<Cfg, String>
where
    I: IntoIterator<Item = String>,
{
    let mut path = None;
    let mut capacity = 8usize;
    let mut lookahead = 8usize;
    let mut expert_bytes = 4096u64;
    let mut profile = "h100".to_string();
    let mut tokens = 64u32;
    let mut experts = 16u32;
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
            "--profile" => profile = value("profile", inline, &mut it)?,
            "--tokens" => tokens = parse_u32("tokens", &value("tokens", inline, &mut it)?)?,
            "--experts" => experts = parse_u32("experts", &value("experts", inline, &mut it)?)?,
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
        profile,
        tokens,
        experts,
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
    match name {
        "h100" => Ok(HardwareProfile::example_h100_sxm()),
        "h200" => Ok(HardwareProfile::example_h200_sxm()),
        "8xh100" => Ok(HardwareProfile::example_8xh100_nvlink()),
        "cheap" => Ok(HardwareProfile::example_cheap_48gb()),
        path => {
            let mut f = File::open(path).map_err(|e| format!("{path}: {e}"))?;
            let mut buf = String::new();
            let _n = f
                .read_to_string(&mut buf)
                .map_err(|e| format!("{path}: {e}"))?;
            HardwareProfile::parse(&buf).map_err(|e| e.to_string())
        }
    }
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
