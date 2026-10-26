//! Native Rust GPU GEMV smoke test. Build with `cargo oxide run --features cuda`.
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
use llama_native_cuda::{Shape, validate};

#[cuda_module]
mod kernels {
    use super::*;

    /// One thread per row. Deliberately simple baseline, not a tuned kernel.
    #[kernel]
    pub fn gemv(weights: &[f32], input: &[f32], mut output: DisjointSlice<f32>, cols: u32) {
        let row = thread::index_1d();
        let cols = cols as usize;
        if let Some(value) = output.get_mut(row) {
            let mut sum = 0.0f32;
            let mut col = 0usize;
            while col < cols {
                sum += weights[row.get() * cols + col] * input[col];
                col += 1;
            }
            *value = sum;
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = kernels::load(&context)?;
    // Non-warp-multiple and multi-block row counts exercise tail handling.
    for (rows, cols) in [(1usize, 1usize), (17, 33), (257, 129)] {
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 23) as f32 - 11.0) / 16.0)
            .collect();
        let input: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32 - 6.0) / 8.0).collect();
        let shape = Shape::checked(rows, cols, weights.len(), input.len())?;
        let expected: Vec<f64> = weights
            .chunks_exact(cols)
            .map(|row| {
                row.iter()
                    .zip(&input)
                    .map(|(&w, &x)| f64::from(w) * f64::from(x))
                    .sum()
            })
            .collect();
        let device_weights = DeviceBuffer::from_host(&stream, &weights)?;
        let device_input = DeviceBuffer::from_host(&stream, &input)?;
        let mut device_output = DeviceBuffer::<f32>::zeroed(&stream, rows)?;
        // SAFETY: Shape::checked establishes flattened u32 index bounds and
        // exact matrix/input lengths. The kernel only accesses a row when
        // DisjointSlice grants its unique output element. All device buffers
        // remain alive through the synchronizing to_host_vec below.
        unsafe {
            module.gemv(
                &stream,
                LaunchConfig::for_num_elems(shape.rows),
                &device_weights,
                &device_input,
                &mut device_output,
                shape.cols,
            )
        }?;
        let actual = device_output.to_host_vec(&stream)?;
        validate(&actual, &expected)?;
        println!("native CUDA Rust GEMV validated: {rows}x{cols}");
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
