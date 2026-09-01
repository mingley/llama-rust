//! Load a GGUF checkpoint from disk and greedily generate a continuation.
//!
//! Demonstrates the shortest path from a file to text: `Model::from_path`,
//! `Model::session`, `Session::generate_streaming`. Also reports prefill and
//! decode throughput separately, because they are very different workloads —
//! prefill shares each weight row across the whole prompt (GEMM), decode does
//! not (GEMV).
//!
//! Needs a real model file. To fetch a small one:
//!
//!     curl -fL -o qwen2.5-0.5b-q4_k_m.gguf \
//!       'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf?download=true'
//!
//! Run:
//!
//!     cargo run --release --example generate -- qwen2.5-0.5b-q4_k_m.gguf "The capital of France is" 32
//!
//! For a version that needs no download, see the `stream_tokens` example.

use std::time::Instant;

use llama_rust::{GenerateOptions, Model, StepAction};

const USAGE: &str = "usage: generate <model.gguf> [prompt] [n_predict]";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        return Err(USAGE.into());
    };
    let prompt = args
        .next()
        .unwrap_or_else(|| "The capital of France is".into());
    let n_predict: usize = match args.next() {
        Some(raw) => raw.parse()?,
        None => 32,
    };

    let load_start = Instant::now();
    let model = Model::from_path(&path)?;
    println!(
        "loaded {path} in {:.2}s: vocab {}, embedding width {}, {} MiB of weights",
        load_start.elapsed().as_secs_f64(),
        model.n_vocab(),
        model.n_embd(),
        model.weights().blob_len() / (1024 * 1024),
    );

    let options = GenerateOptions::new(n_predict);
    let n_ctx = model
        .encode(&prompt)?
        .len()
        .saturating_add(options.n_predict)
        .saturating_add(1);
    let mut session = model.session(n_ctx)?;

    // Time the two phases apart without doing the work twice: the gap before
    // the first token is prefill, everything after it is decode.
    let start = Instant::now();
    let mut first_token_at = None;
    let done = session.generate_streaming(&prompt, &options, |_step| {
        if first_token_at.is_none() {
            first_token_at = Some(start.elapsed());
        }
        StepAction::Continue
    })?;
    let total = start.elapsed();

    println!("\nprompt: {prompt}");
    println!("output: {}", done.text);

    let prefill = first_token_at.unwrap_or(total).as_secs_f64();
    let decode = total.as_secs_f64() - prefill;
    let decoded = done.tokens.len().saturating_sub(1);
    println!(
        "\nprefill {} tokens in {prefill:.3}s ({:.1} tok/s)",
        done.prompt_tokens.len(),
        rate(done.prompt_tokens.len(), prefill),
    );
    println!(
        "decode  {decoded} tokens in {decode:.3}s ({:.1} tok/s), stopped on {:?}",
        rate(decoded, decode),
        done.stop,
    );
    Ok(())
}

fn rate(tokens: usize, seconds: f64) -> f64 {
    let n = f64::from(u32::try_from(tokens).unwrap_or(u32::MAX));
    if seconds > 0.0 {
        n / seconds
    } else {
        0.0
    }
}
