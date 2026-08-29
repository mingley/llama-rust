//! `gpu-profile names | example | parse | probe` — example hardware, not captures.
//!
//! A real capture needs a physical GPU. This binary prints named example
//! profiles, validates `key=value` files, and runs a topology probe on the
//! discrete-event simulator. It does not invent `$/M tokens`.

use gpu_sim::{probe_topology, HardwareProfile};
use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::process::ExitCode;

const USAGE: &str = "\
usage: gpu-profile <command> [args]
  names
  example <NAME>
  parse <file.profile>
  probe <NAME> [--bytes N]   pinned H2D per GPU + D2D per pair
  capture

NAME: h100, h200, 8xh100, cheap, 2xh100-pcie, bad-numa, 2node-rdma, asymmetric
capture: refused here (no GPU in this crate). Someone with silicon writes a
         key=value file; agents consume it with parse / HardwareProfile::parse.
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
        "help" | "--help" | "-h" => print_usage(),
        "names" => {
            for name in HardwareProfile::example_names() {
                println!("{name}");
            }
            Ok(())
        }
        "example" => {
            let name = args.next().ok_or("example <NAME>")?;
            let p = HardwareProfile::by_name(&name).map_err(|e| e.to_string())?;
            print!("{}", p.to_profile_text());
            Ok(())
        }
        "parse" => {
            let path = args.next().ok_or("parse <file.profile>")?;
            let p = parse_file(&path)?;
            println!("ok name={} gpus={}", p.name, p.n_gpus());
            Ok(())
        }
        "probe" => {
            let name = args.next().ok_or("probe <NAME>")?;
            let mut bytes = 8u64 << 20;
            if let Some(flag) = args.next() {
                if flag == "--bytes" {
                    let v = args.next().ok_or("missing --bytes value")?;
                    bytes = v.parse::<u64>().map_err(|_| format!("invalid bytes {v}"))?;
                } else {
                    return Err(format!("unknown flag {flag}"));
                }
            }
            let p = HardwareProfile::by_name(&name).map_err(|e| e.to_string())?;
            let probe = probe_topology(p, bytes).map_err(|e| e.to_string())?;
            println!("{}", probe.line());
            Ok(())
        }
        "capture" => Err(
            "gpu-profile capture needs a physical GPU; this crate only consumes profiles"
                .to_string(),
        ),
        other => Err(format!("{USAGE}got {other}")),
    }
}

fn print_usage() -> Result<(), String> {
    let mut out = std::io::stdout();
    out.write_all(USAGE.as_bytes()).map_err(|e| e.to_string())
}

fn parse_file(path: &str) -> Result<HardwareProfile, String> {
    let mut f = File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut buf = String::new();
    let _n = f
        .read_to_string(&mut buf)
        .map_err(|e| format!("{path}: {e}"))?;
    HardwareProfile::parse(&buf).map_err(|e| e.to_string())
}
