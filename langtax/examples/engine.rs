//! Continuous batching: intern hits, then recompute preemption on a tight pool.

use llama_rust::{Engine, EngineCfg, Model};
use std::io::{self, Write};
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let _w = writeln!(io::stderr(), "{e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let model = Model::from_bytes(llama_rust::tiny_qwen3moe_gguf()).map_err(|e| e.to_string())?;
    let mut cfg = EngineCfg::tiny();
    cfg.prefill_chunk = 2;
    let mut eng = Engine::new(model.llama(), cfg).map_err(|e| e.to_string())?;
    let a = eng.add(&[1, 2, 3, 4], 2).map_err(|e| e.to_string())?;
    let _s = eng.step().map_err(|e| e.to_string())?;
    let b = eng.add(&[1, 2, 0, 1], 2).map_err(|e| e.to_string())?;
    eng.run().map_err(|e| e.to_string())?;
    let out_a = eng.take(a).ok_or("missing a")?;
    let out_b = eng.take(b).ok_or("missing b")?;
    let hits = eng.pool().hits();
    let mut out = io::stdout();
    out.write_all(
        format!(
            "a_gen={} b_gen={} intern_hits={hits} gemm_peak={} active={}\n",
            out_a.generated.len(),
            out_b.generated.len(),
            eng.stats().gemm_peak,
            eng.active()
        )
        .as_bytes(),
    )
    .map_err(|e| e.to_string())?;

    let mut tight = EngineCfg::tiny();
    tight.pool_blocks = 3;
    tight.max_seqs = 2;
    let mut squeezed = Engine::new(model.llama(), tight).map_err(|e| e.to_string())?;
    let pa = squeezed.add(&[1, 2, 3, 4], 2).map_err(|e| e.to_string())?;
    let pb = squeezed.add(&[5, 0, 5, 0], 2).map_err(|e| e.to_string())?;
    squeezed.run().map_err(|e| e.to_string())?;
    let out_pa = squeezed.take(pa).ok_or("missing pa")?;
    let out_pb = squeezed.take(pb).ok_or("missing pb")?;
    out.write_all(
        format!(
            "preempt_a={} preempt_b={} preempts={}\n",
            out_pa.generated.len(),
            out_pb.generated.len(),
            squeezed.preempts()
        )
        .as_bytes(),
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}
