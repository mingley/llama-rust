use crate::Shape;
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread, warp};
use std::sync::Arc;

#[expect(
    clippy::disallowed_types,
    reason = "NVIDIA's generated synchronous module registry uses std::sync::Mutex; no lock crosses an await"
)]
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn gemv(weights: &[f32], input: &[f32], mut output: DisjointSlice<f32>, cols: u32) {
        let row = thread::index_1d();
        let row_index = row.get();
        let cols = cols as usize;
        if let Some(value) = output.get_mut(row) {
            let mut sum = 0.0f32;
            let mut col = 0usize;
            while col < cols {
                sum += weights[row_index * cols + col] * input[col];
                col += 1;
            }
            *value = sum;
        }
    }

    #[kernel]
    pub fn gemm(
        weights: &[f32],
        input: &[f32],
        mut output: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
    ) {
        let index = thread::index_1d();
        let output_index = index.get();
        if let Some(value) = output.get_mut(index) {
            let row = output_index % rows as usize;
            let token = output_index / rows as usize;
            let mut sum = 0.0f32;
            let mut col = 0usize;
            while col < cols as usize {
                sum += weights[row * cols as usize + col] * input[token * cols as usize + col];
                col += 1;
            }
            *value = sum;
        }
    }

    // One full warp cooperates on each output. Adjacent lanes read adjacent
    // columns of the same row, avoiding the strided loads of scalar-per-row
    // execution. Padding warps still participate in every shuffle with zero
    // contributions; no individual lane returns before the reduction.
    #[kernel]
    pub fn gemv_warp(
        weights: &[f32],
        input: &[f32],
        mut output: DisjointSlice<f32, thread::WarpIndex>,
        rows: u32,
        cols: u32,
    ) {
        let index = thread::index_1d();
        let row = index.get() / 32;
        let mut sum = 0.0f32;
        if row < rows as usize {
            let mut col = warp::lane_id() as usize;
            while col < cols as usize {
                sum += weights[row * cols as usize + col] * input[col];
                col += 32;
            }
        }
        let total = warp::reduce_sum_f32(sum);
        if let Some(owner) = thread::warp_index()
            && let Some(value) = output.get_mut(owner)
        {
            *value = total;
        }
    }

    #[kernel]
    pub fn gemm_warp(
        weights: &[f32],
        input: &[f32],
        mut output: DisjointSlice<f32, thread::WarpIndex>,
        rows: u32,
        cols: u32,
        outputs: u32,
    ) {
        let index = thread::index_1d();
        let output_index = index.get() / 32;
        let mut sum = 0.0f32;
        if output_index < outputs as usize {
            let row = output_index % rows as usize;
            let token = output_index / rows as usize;
            let mut col = warp::lane_id() as usize;
            while col < cols as usize {
                sum += weights[row * cols as usize + col] * input[token * cols as usize + col];
                col += 32;
            }
        }
        let total = warp::reduce_sum_f32(sum);
        if let Some(owner) = thread::warp_index()
            && let Some(value) = output.get_mut(owner)
        {
            *value = total;
        }
    }
}

// Eight full warps per block, one output per warp. Construct geometry from
// outputs directly so outputs*32 cannot overflow the u32 launch helper.
fn warp_launch(outputs: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (outputs.div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Immutable row-major matrix uploaded once and retained on an NVIDIA GPU.
pub struct CudaMatrix {
    context: Arc<CudaContext>,
    weights: DeviceBuffer<f32>,
    shape: Shape,
}

/// Native NVIDIA CUDA Rust executor. Errors propagate without CPU fallback.
pub struct CudaGemv {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    input: Option<DeviceBuffer<f32>>,
    output: Option<DeviceBuffer<f32>>,
}

impl CudaGemv {
    /// Initialize the selected CUDA device, stream, and compiled Rust kernel.
    pub fn new(device_ordinal: usize) -> Result<Self, String> {
        let context = CudaContext::new(device_ordinal).map_err(|e| e.to_string())?;
        let stream = context.default_stream();
        let module = kernels::load(&context).map_err(|e| e.to_string())?;
        Ok(Self {
            context,
            stream,
            module,
            input: None,
            output: None,
        })
    }

    // Reuse same-sized scratch buffers across model layers and tokens. Every
    // launch overwrites the entire output, and copy_to_host synchronizes it
    // before the next operation can reuse either buffer.
    fn prepare_scratch(&mut self, input_len: usize, output_len: usize) -> Result<(), String> {
        if self
            .input
            .as_ref()
            .is_none_or(|buffer| buffer.len() != input_len)
        {
            // Drop the old allocation before growing; otherwise resize can
            // fail when the old and new buffers do not fit simultaneously.
            self.input = None;
            self.input =
                Some(DeviceBuffer::zeroed(&self.stream, input_len).map_err(|e| e.to_string())?);
        }
        if self
            .output
            .as_ref()
            .is_none_or(|buffer| buffer.len() != output_len)
        {
            self.output = None;
            self.output =
                Some(DeviceBuffer::zeroed(&self.stream, output_len).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    /// Upload an immutable F32 matrix for reuse across generated tokens.
    pub fn upload(
        &mut self,
        rows: usize,
        cols: usize,
        weights: &[f32],
    ) -> Result<CudaMatrix, String> {
        let shape = Shape::checked(rows, cols, weights.len(), cols)?;
        let weights = DeviceBuffer::from_host(&self.stream, weights).map_err(|e| e.to_string())?;
        Ok(CudaMatrix {
            context: self.context.clone(),
            weights,
            shape,
        })
    }

    /// Multiply a resident matrix by a host vector and synchronize the output.
    pub fn gemv_resident(
        &mut self,
        matrix: &CudaMatrix,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        if !Arc::ptr_eq(&self.context, &matrix.context) {
            return Err("CUDA matrix belongs to a different executor".into());
        }
        if input.len() != matrix.shape.cols as usize || output.len() != matrix.shape.rows as usize {
            return Err("CUDA GEMV vector lengths do not match resident matrix".into());
        }
        self.prepare_scratch(input.len(), output.len())?;
        let device_input = self
            .input
            .as_mut()
            .ok_or("CUDA input scratch unavailable")?;
        let device_output = self
            .output
            .as_mut()
            .ok_or("CUDA output scratch unavailable")?;
        device_input
            .copy_from_host(&self.stream, input)
            .map_err(|e| e.to_string())?;
        // SAFETY: upload validates flattened matrix indexing, this method
        // validates both vectors, and DisjointSlice assigns a unique output
        // element to each thread (or lane-zero WarpIndex). Warp launches use
        // 1D full-warp blocks; every lane participates in each reduction.
        // All buffers outlive the synchronized copy.
        unsafe {
            if matrix.shape.cols >= 64 {
                self.module.gemv_warp(
                    &self.stream,
                    warp_launch(matrix.shape.rows),
                    &matrix.weights,
                    device_input,
                    device_output,
                    matrix.shape.rows,
                    matrix.shape.cols,
                )
            } else {
                self.module.gemv(
                    &self.stream,
                    LaunchConfig::for_num_elems(matrix.shape.rows),
                    &matrix.weights,
                    device_input,
                    device_output,
                    matrix.shape.cols,
                )
            }
        }
        .map_err(|e| e.to_string())?;
        device_output
            .copy_to_host(&self.stream, output)
            .map_err(|e| e.to_string())
    }

    /// Upload and multiply a host matrix. Prefer `upload` and `gemv_resident`
    /// for model weights reused across tokens to avoid repeated weight DMA.
    pub fn gemv(
        &mut self,
        weights: &[f32],
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        let matrix = self.upload(output.len(), input.len(), weights)?;
        self.gemv_resident(&matrix, input, output)
    }

    /// Multiply token-major input vectors by one resident matrix in one launch.
    /// Output is token-major with one `rows`-element vector per input token.
    pub fn gemm_resident(
        &mut self,
        matrix: &CudaMatrix,
        tokens: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        if !Arc::ptr_eq(&self.context, &matrix.context) {
            return Err("CUDA matrix belongs to a different executor".into());
        }
        let inputs = tokens
            .checked_mul(matrix.shape.cols as usize)
            .ok_or("CUDA GEMM input shape overflow")?;
        let outputs = tokens
            .checked_mul(matrix.shape.rows as usize)
            .ok_or("CUDA GEMM output shape overflow")?;
        if tokens == 0 || input.len() != inputs || output.len() != outputs {
            return Err("CUDA GEMM buffer lengths do not match shape".into());
        }
        let launch_elements = u32::try_from(outputs).map_err(|_| "CUDA GEMM launch exceeds u32")?;
        let _input_bound = u32::try_from(inputs).map_err(|_| "CUDA GEMM input exceeds u32")?;
        self.prepare_scratch(input.len(), outputs)?;
        let device_input = self
            .input
            .as_mut()
            .ok_or("CUDA input scratch unavailable")?;
        let device_output = self
            .output
            .as_mut()
            .ok_or("CUDA output scratch unavailable")?;
        device_input
            .copy_from_host(&self.stream, input)
            .map_err(|e| e.to_string())?;
        // SAFETY: upload validated weights. Checked products and exact vector
        // lengths bound token/row indexing; each output is uniquely assigned.
        // Warp geometry is 1D with full 32-lane warps; all lanes converge at
        // each reduction and only lane zero owns an output WarpIndex.
        // All buffers remain alive through the synchronized host copy.
        unsafe {
            if matrix.shape.cols >= 64 {
                self.module.gemm_warp(
                    &self.stream,
                    warp_launch(launch_elements),
                    &matrix.weights,
                    device_input,
                    device_output,
                    matrix.shape.rows,
                    matrix.shape.cols,
                    launch_elements,
                )
            } else {
                self.module.gemm(
                    &self.stream,
                    LaunchConfig::for_num_elems(launch_elements),
                    &matrix.weights,
                    device_input,
                    device_output,
                    matrix.shape.rows,
                    matrix.shape.cols,
                )
            }
        }
        .map_err(|e| e.to_string())?;
        device_output
            .copy_to_host(&self.stream, output)
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::warp_launch;

    #[test]
    fn warp_geometry_covers_outputs_without_partial_warps_or_u32_overflow() {
        for outputs in [1, 7, 8, 9, 257, u32::MAX] {
            let launch = warp_launch(outputs);
            assert_eq!(launch.block_dim, (256, 1, 1));
            assert_eq!(launch.grid_dim.1, 1);
            assert_eq!(launch.grid_dim.2, 1);
            let warps = u64::from(launch.grid_dim.0) * 8;
            assert!(warps >= u64::from(outputs));
            assert!(warps - u64::from(outputs) < 8);
            assert_eq!(launch.shared_mem_bytes, 0);
        }
    }

    #[test]
    fn lane_striding_and_butterfly_match_f64_oracle_for_column_tails() {
        for cols in [1, 31, 32, 33, 63, 64, 65, 129, 1025, 4096] {
            let weights: Vec<f32> = (0..cols).map(|i| ((i % 23) as f32 - 11.0) / 16.0).collect();
            let input: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32 - 6.0) / 8.0).collect();
            let oracle: f64 = weights
                .iter()
                .zip(&input)
                .map(|(&weight, &value)| f64::from(weight) * f64::from(value))
                .sum();
            let mut lanes = [0.0f32; 32];
            for (lane, partial) in lanes.iter_mut().enumerate() {
                *partial = weights
                    .iter()
                    .zip(&input)
                    .skip(lane)
                    .step_by(32)
                    .map(|(&weight, &value)| weight * value)
                    .sum();
            }
            for distance in [16, 8, 4, 2, 1] {
                let previous = lanes;
                for (lane, partial) in lanes.iter_mut().enumerate() {
                    *partial += previous[lane ^ distance];
                }
            }
            // Binary-fraction fixtures have exact sums; this checks every
            // column is included once and every lane receives the same sum.
            for value in lanes {
                assert_eq!(f64::from(value), oracle, "cols={cols}");
            }
        }
    }
}
