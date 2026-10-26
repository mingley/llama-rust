//! Native CUDA Rust GEMV validation runner. Build with `cargo oxide run`.
use llama_native_cuda::{CudaGemv, validate};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = CudaGemv::new(0)?;
    // Non-warp-multiple and multi-block row counts exercise tail handling.
    for (rows, cols) in [(1usize, 1usize), (17, 33), (257, 129)] {
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 23) as f32 - 11.0) / 16.0)
            .collect();
        let input: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32 - 6.0) / 8.0).collect();
        let expected: Vec<f64> = weights
            .chunks_exact(cols)
            .map(|row| {
                row.iter()
                    .zip(&input)
                    .map(|(&w, &x)| f64::from(w) * f64::from(x))
                    .sum()
            })
            .collect();
        let matrix = gpu.upload(rows, cols, &weights)?;
        let mut actual = vec![0.0; rows];
        gpu.gemv_resident(&matrix, &input, &mut actual)?;
        validate(&actual, &expected)?;
        println!("native CUDA Rust GEMV validated: {rows}x{cols}");
        let batch_input: Vec<f32> = [1.0, 0.5, -2.0]
            .into_iter()
            .flat_map(|scale| input.iter().map(move |value| value * scale))
            .collect();
        let batch_expected: Vec<f64> = [1.0, 0.5, -2.0]
            .into_iter()
            .flat_map(|scale| expected.iter().map(move |value| value * scale))
            .collect();
        let mut batch_actual = vec![0.0; rows * 3];
        gpu.gemm_resident(&matrix, 3, &batch_input, &mut batch_actual)?;
        validate(&batch_actual, &batch_expected)?;
        println!("native CUDA Rust GEMM validated: 3 tokens, {rows}x{cols}");
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
