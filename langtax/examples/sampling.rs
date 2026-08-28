//! Configure sampling: temperature, top-k, top-p, repeat penalty, and seeding.
//!
//! Demonstrates `SampleParams` through `GenerateOptions::with_sampling`, and
//! the two reproducibility rules worth internalising:
//!
//!   * `SampleParams::greedy` (temperature <= 0) draws no random numbers at
//!     all, so it repeats exactly and needs no seed.
//!   * Any temperature above zero *requires* a seed. Passing one makes the run
//!     reproducible; changing it changes the output.
//!
//! Filters are applied in llama.cpp's order: repeat penalty on unique previous
//! ids, then temperature, then softmax, then top-k, then top-p, then a
//! categorical draw from what survives.
//!
//! Runs with no download by default:
//!
//!     cargo run --release --example sampling
//!
//! Or against a real checkpoint, where the differences are legible as text:
//!
//!     cargo run --release --example sampling -- model.gguf "Once upon a time"

use llama_rust::{fixtures, GenerateOptions, Model, SampleParams};

/// Base options shared by every configuration below: 24 tokens, one pinned KV
/// allocation. Sampling knobs are layered on top per run.
fn base() -> GenerateOptions {
    GenerateOptions::new(24).with_n_ctx(128)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (model, prompt) = match args.next() {
        Some(path) => {
            let prompt = args.next().unwrap_or_else(|| "Once upon a time".into());
            (Model::from_path(&path)?, prompt)
        }
        None => {
            println!("# no model path given, using the in-memory tiny_qwen2 fixture");
            println!("# its weights are untrained, so compare token ids rather than prose\n");
            (
                Model::from_bytes(fixtures::tiny_qwen2_gguf())?,
                "ab".to_string(),
            )
        }
    };

    let configs: [(&str, GenerateOptions); 6] = [
        ("greedy (no RNG, no seed)", base()),
        (
            "temperature 0.7, seed 1",
            base().with_temperature(0.7).with_seed(1),
        ),
        (
            "temperature 0.7, seed 2",
            base().with_temperature(0.7).with_seed(2),
        ),
        (
            "temperature 1.2, top-k 3",
            base().with_temperature(1.2).with_top_k(3).with_seed(1),
        ),
        (
            "temperature 1.2, top-p 0.9",
            base().with_temperature(1.2).with_top_p(0.9).with_seed(1),
        ),
        (
            "temperature 0.7, repeat penalty 1.6",
            base()
                .with_temperature(0.7)
                .with_repeat_penalty(1.6)
                .with_seed(1),
        ),
    ];

    // One session, reused: a pinned n_ctx means the KV cache is allocated once
    // instead of once per configuration.
    let mut session = model.session();

    for (label, options) in configs {
        let done = session.generate_detailed(&prompt, &options)?;
        println!("{label}");
        println!("  ids  {:?} (stopped on {:?})", done.tokens, done.stop);
        println!("  text {:?}", done.text);

        // Same knobs and same seed must land on the same tokens.
        let again = session.generate_detailed(&prompt, &options)?;
        let reproducible = again.tokens == done.tokens;
        println!("  reproducible: {reproducible}\n");
        assert_reproducible(reproducible)?;
    }

    // A temperature above zero with no seed is refused rather than silently
    // seeded from the clock, so a run can never be accidentally unrepeatable.
    // Spelled as a struct literal here to show that the fields are public when
    // the `with_*` shortcuts are not enough.
    let unseeded = SampleParams {
        temperature: 0.7,
        seed: None,
        ..SampleParams::greedy()
    };
    let refused =
        session.generate_detailed(&prompt, &GenerateOptions::new(4).with_sampling(unseeded));
    println!("temperature > 0 with no seed => {:?}", refused.err());
    Ok(())
}

fn assert_reproducible(ok: bool) -> Result<(), Box<dyn std::error::Error>> {
    if ok {
        Ok(())
    } else {
        Err("same seed produced different tokens".into())
    }
}
