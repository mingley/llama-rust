//! Layered `Model` / `Session` API with an optional ExpertStore.

use llama_rust::{CachedStore, LiveStore, Model};
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
    let ids = model.encode("ab").map_err(|e| e.to_string())?;
    let mut sess = model.session(16).map_err(|e| e.to_string())?;
    let direct = model
        .llama()
        .expert_direct_store()
        .map_err(|e| e.to_string())?;
    let slots = direct.len().max(1);
    sess.attach_expert_store(LiveStore::Cached(
        CachedStore::new(direct, slots).map_err(|e| e.to_string())?,
    ));
    let n_logits = sess.prefill(&ids).map_err(|e| e.to_string())?.len();
    let n_past = sess.n_past();
    let hits = sess.expert_metrics().map_or(0, |m| m.hits);
    let mut out = io::stdout();
    out.write_all(format!("n_past={n_past} vocab_logits={n_logits} hits={hits}\n").as_bytes())
        .map_err(|e| e.to_string())?;
    Ok(())
}
