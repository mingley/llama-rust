use crate::Shape;
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
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
        })
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
        let device_input =
            DeviceBuffer::from_host(&self.stream, input).map_err(|e| e.to_string())?;
        let mut device_output =
            DeviceBuffer::<f32>::zeroed(&self.stream, output.len()).map_err(|e| e.to_string())?;
        // SAFETY: upload validates flattened matrix indexing, this method
        // validates both vectors, and DisjointSlice assigns a unique output
        // element to each thread. All buffers outlive the synchronized copy.
        unsafe {
            self.module.gemv(
                &self.stream,
                LaunchConfig::for_num_elems(matrix.shape.rows),
                &matrix.weights,
                &device_input,
                &mut device_output,
                matrix.shape.cols,
            )
        }
        .map_err(|e| e.to_string())?;
        let actual = device_output
            .to_host_vec(&self.stream)
            .map_err(|e| e.to_string())?;
        output.copy_from_slice(&actual);
        Ok(())
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
        let device_input =
            DeviceBuffer::from_host(&self.stream, input).map_err(|e| e.to_string())?;
        let mut device_output =
            DeviceBuffer::<f32>::zeroed(&self.stream, outputs).map_err(|e| e.to_string())?;
        // SAFETY: upload validated weights. Checked products and exact vector
        // lengths bound token/row indexing; each output is uniquely assigned.
        // All buffers remain alive through the synchronized host copy.
        unsafe {
            self.module.gemm(
                &self.stream,
                LaunchConfig::for_num_elems(launch_elements),
                &matrix.weights,
                &device_input,
                &mut device_output,
                matrix.shape.rows,
                matrix.shape.cols,
            )
        }
        .map_err(|e| e.to_string())?;
        let actual = device_output
            .to_host_vec(&self.stream)
            .map_err(|e| e.to_string())?;
        output.copy_from_slice(&actual);
        Ok(())
    }
}
