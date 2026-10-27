//! Native CUDA Rust GEMV validation runner. Build with `cargo oxide run`.
use llama_native_cuda::{CudaGemv, validate};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = CudaGemv::new(0)?;
    // Revisit shapes after resizing, and alternate vector/batch launches.
    // Every equal-sized reuse changes all inputs; every output is checked.
    let shapes = [1usize, 7, 9, 257]
        .into_iter()
        .flat_map(|rows| {
            [63usize, 64, 65, 129, 1025]
                .into_iter()
                .map(move |cols| (rows, cols))
        })
        .chain([(17, 33), (1, 1), (257, 129), (17, 33)]);
    for (rows, cols) in shapes {
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 23) as f32 - 11.0) / 16.0)
            .collect();
        let matrix = gpu.upload(rows, cols, &weights)?;
        for (round, tokens) in [1usize, 1, 3, 3, 1, 7, 1].into_iter().enumerate() {
            let input: Vec<f32> = (0..tokens * cols)
                .map(|i| (((i * 5 + round * 7) % 29) as f32 - 14.0) / 16.0)
                .collect();
            let expected: Vec<f64> = input
                .chunks_exact(cols)
                .flat_map(|token| {
                    weights.chunks_exact(cols).map(move |row| {
                        row.iter()
                            .zip(token)
                            .map(|(&w, &x)| f64::from(w) * f64::from(x))
                            .sum()
                    })
                })
                .collect();
            let mut actual = vec![f32::NAN; rows * tokens];
            if tokens == 1 {
                gpu.gemv_resident(&matrix, &input, &mut actual)?;
            } else {
                gpu.gemm_resident(&matrix, tokens, &input, &mut actual)?;
            }
            validate(&actual, &expected)?;
            println!(
                "native CUDA Rust matrix validated: {tokens} tokens, {rows}x{cols}, round {round}"
            );
        }
    }
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!(
                "native CUDA GEMV failed: {error}. Requires CUDA 13+, an NVIDIA GPU/driver, and cargo-oxide compilation; no CPU fallback was used."
            );
            std::process::ExitCode::FAILURE
        }
    }
}
