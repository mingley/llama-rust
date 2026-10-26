//! Native NVIDIA matrix dispatch and bounded device-weight residency.
//!
//! Cache identity includes the complete packed bytes, dtype and dimensions;
//! pointer reuse or a hash collision cannot return another model's weights.

use crate::quant::QuantError;
use llama_native_cuda::{CudaGemv, CudaMatrix};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::hash::{DefaultHasher, Hash, Hasher};

const CACHE_BYTES: usize = 256 * 1024 * 1024;

struct Entry {
    dtype: u8,
    rows: usize,
    cols: usize,
    hash: u64,
    packed: Vec<u8>,
    matrix: CudaMatrix,
    bytes: usize,
}

struct State {
    gpu: CudaGemv,
    cache: VecDeque<Entry>,
    bytes: usize,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// Execute a token-major matrix product. Format conversion runs only on a
/// device-cache miss; CUDA errors propagate without CPU fallback.
#[expect(
    clippy::too_many_arguments,
    reason = "matrix shape and packed identity must remain explicit"
)]
pub(crate) fn matmul(
    dtype: u8,
    rows: usize,
    cols: usize,
    tokens: usize,
    packed: &[u8],
    input: &[f32],
    output: &mut [f32],
    expand: impl FnOnce() -> Result<Vec<f32>, QuantError>,
) -> Result<(), QuantError> {
    STATE.with(|slot| {
        let mut slot = slot
            .try_borrow_mut()
            .map_err(|_| QuantError::NativeCuda("recursive GPU execution is unsupported".into()))?;
        if slot.is_none() {
            *slot = Some(State {
                gpu: CudaGemv::new(0).map_err(QuantError::NativeCuda)?,
                cache: VecDeque::new(),
                bytes: 0,
            });
        }
        let state = slot.as_mut().ok_or_else(|| {
            QuantError::NativeCuda("GPU initialization did not create a context".into())
        })?;
        let mut hasher = DefaultHasher::new();
        packed.hash(&mut hasher);
        let hash = hasher.finish();
        let hit = state.cache.iter().position(|entry| {
            entry.hash == hash
                && entry.dtype == dtype
                && entry.rows == rows
                && entry.cols == cols
                && entry.packed == packed
        });
        if let Some(index) = hit {
            let entry = state
                .cache
                .remove(index)
                .ok_or_else(|| QuantError::NativeCuda("GPU cache entry disappeared".into()))?;
            let result = state
                .gpu
                .gemm_resident(&entry.matrix, tokens, input, output);
            state.cache.push_back(entry);
            return result.map_err(QuantError::NativeCuda);
        }

        let bytes = rows
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .and_then(|n| n.checked_add(packed.len()))
            .ok_or_else(|| QuantError::NativeCuda("resident matrix size overflow".into()))?;
        while !state.cache.is_empty() && state.bytes.saturating_add(bytes) > CACHE_BYTES {
            if let Some(entry) = state.cache.pop_front() {
                state.bytes = state.bytes.saturating_sub(entry.bytes);
            }
        }
        let weights = expand()?;
        let matrix = state
            .gpu
            .upload(rows, cols, &weights)
            .map_err(QuantError::NativeCuda)?;
        state
            .gpu
            .gemm_resident(&matrix, tokens, input, output)
            .map_err(QuantError::NativeCuda)?;
        // Oversized matrices are executed once and immediately released. The
        // cache budget counts resident F32 weights plus exact packed identity.
        if bytes <= CACHE_BYTES {
            state.bytes += bytes;
            state.cache.push_back(Entry {
                dtype,
                rows,
                cols,
                hash,
                packed: packed.to_vec(),
                matrix,
                bytes,
            });
        }
        Ok(())
    })
}
