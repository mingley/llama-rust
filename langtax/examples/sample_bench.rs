//! Wall-clock sampler microbenchmark; run with `--release` on an idle host.
use llama_rust::{sample_next, SampleParams};
use std::hint::black_box;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let logits: Vec<f32> = (0..128_256_u32)
        .map(|id| f32::from(u16::try_from(id.wrapping_mul(7919) % 65536).unwrap_or(0)) / 4096.0)
        .collect();
    for (name, params) in [
        ("greedy", SampleParams::greedy()),
        (
            "top-k=40 top-p=0.95",
            SampleParams {
                temperature: 0.8,
                top_k: 40,
                top_p: 0.95,
                repeat_penalty: 1.0,
                seed: Some(42),
            },
        ),
    ] {
        let mut state = 42;
        for _ in 0..10 {
            let _token = black_box(sample_next(
                &params,
                Some(&mut state),
                black_box(&logits),
                &[],
            )?);
        }
        let start = Instant::now();
        for _ in 0..100 {
            let _token = black_box(sample_next(
                &params,
                Some(&mut state),
                black_box(&logits),
                &[],
            )?);
        }
        println!(
            "{name}: {} ns/draw (128256 logits, 100 draws)",
            start.elapsed().as_nanos() / 100
        );
    }
    Ok(())
}
