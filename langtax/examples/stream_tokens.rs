//! Stream generated tokens as they are produced, and stop early from the
//! callback.
//!
//! Demonstrates `Session::generate_streaming`: one callback per token carrying
//! the token id, its text, and the full logit vector it was drawn from. The
//! callback here prints each piece immediately, records how confident the model
//! was, and returns `StepAction::Stop` when a stop sequence appears — which is
//! how you implement stop strings, token budgets, or an abort signal without
//! any support from the library.
//!
//! Runs with no download by default, against an in-memory fixture:
//!
//!     cargo run --release --example stream_tokens
//!
//! Or point it at a real checkpoint:
//!
//!     cargo run --release --example stream_tokens -- model.gguf "Once upon a time"

use std::io::Write;

use llama_rust::{fixtures, GenerateOptions, Model, StepAction};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (model, prompt) = match args.next() {
        Some(path) => {
            let prompt = args.next().unwrap_or_else(|| "Once upon a time".into());
            println!("# streaming from {path}\n");
            (Model::from_path(&path)?, prompt)
        }
        None => {
            println!("# no model path given, streaming from the in-memory tiny_qwen2 fixture\n");
            (
                Model::from_bytes(fixtures::tiny_qwen2_gguf())?,
                "ab".to_string(),
            )
        }
    };

    // Stop as soon as the text so far ends with a blank line, the usual
    // "the model finished its paragraph" heuristic.
    const STOP_SEQUENCE: &str = "\n\n";

    let mut seen = String::new();
    let mut min_margin = f32::MAX;
    let mut io_error = None;

    print!("{prompt}");
    flush()?;

    let options = GenerateOptions::new(64);
    let done = model
        .session()
        .generate_streaming(&prompt, &options, |step| {
            print!("{}", step.piece);
            // Flushing per token is what makes the output appear live. The
            // callback cannot return an error, so if the pipe has gone away we
            // stash the failure and ask generation to stop.
            if let Err(err) = flush() {
                io_error = Some(err);
                return StepAction::Stop;
            }

            min_margin = min_margin.min(top_two_margin(step.logits));
            seen.push_str(&step.piece);
            if seen.ends_with(STOP_SEQUENCE) {
                StepAction::Stop
            } else {
                StepAction::Continue
            }
        })?;

    if let Some(err) = io_error {
        return Err(err.into());
    }

    println!("\n\n---");
    println!("{} tokens, stopped on {:?}", done.tokens.len(), done.stop);
    println!("token ids: {:?}", done.tokens);
    if min_margin.is_finite() {
        // A small margin means the top two candidates were nearly tied, which
        // is where a different sampler would have diverged from greedy.
        println!("narrowest top-1 vs top-2 logit gap: {min_margin:.4}");
    }
    Ok(())
}

fn flush() -> Result<(), std::io::Error> {
    std::io::stdout().flush()
}

/// Gap between the best and second-best logit: how decided the model was.
fn top_two_margin(logits: &[f32]) -> f32 {
    let mut best = f32::MIN;
    let mut second = f32::MIN;
    for value in logits {
        if *value > best {
            second = best;
            best = *value;
        } else if *value > second {
            second = *value;
        }
    }
    if second > f32::MIN {
        best - second
    } else {
        f32::MAX
    }
}
