//! Native NVIDIA matrix dispatch and bounded device-weight residency.
//!
//! Cache identity includes the complete packed bytes, dtype and dimensions;
//! pointer reuse or a hash collision cannot return another model's weights.
//! `LLAMA_CUDA_CACHE_BYTES` sets the per-thread residency budget once at context
//! initialization (default 256 MiB; zero disables caching). The budget counts
//! device F32 weights and retained packed identity, not transient launch memory.

use crate::quant::QuantError;
use llama_native_cuda::{CudaGemv, CudaMatrix};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::hash::{DefaultHasher, Hash, Hasher};

const DEFAULT_CACHE_BYTES: usize = 256 * 1024 * 1024;
const CACHE_BYTES_ENV: &str = "LLAMA_CUDA_CACHE_BYTES";

fn parse_cache_bytes(value: Option<&str>) -> Result<usize, QuantError> {
    let Some(value) = value else {
        return Ok(DEFAULT_CACHE_BYTES);
    };
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(QuantError::NativeCuda(format!(
            "{CACHE_BYTES_ENV} must be a nonnegative decimal byte count"
        )));
    }
    value.parse().map_err(|_| {
        QuantError::NativeCuda(format!("{CACHE_BYTES_ENV} exceeds addressable memory"))
    })
}

fn configured_cache_bytes() -> Result<usize, QuantError> {
    match std::env::var(CACHE_BYTES_ENV) {
        Ok(value) => parse_cache_bytes(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_cache_bytes(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(QuantError::NativeCuda(format!(
            "{CACHE_BYTES_ENV} must be a nonnegative decimal byte count"
        ))),
    }
}

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
    limit: usize,
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
            // Resolve once per inference thread before initializing CUDA. A
            // malformed configuration fails rather than silently changing its
            // memory policy. Zero opts out of residency, not GPU execution.
            let limit = configured_cache_bytes()?;
            *slot = Some(State {
                gpu: CudaGemv::new(0).map_err(QuantError::NativeCuda)?,
                cache: VecDeque::new(),
                bytes: 0,
                limit,
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
        while !state.cache.is_empty() && state.bytes.saturating_add(bytes) > state.limit {
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
        if bytes <= state.limit {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_budget_is_explicit_and_strictly_parsed() {
        assert_eq!(parse_cache_bytes(None), Ok(DEFAULT_CACHE_BYTES));
        assert_eq!(parse_cache_bytes(Some("0")), Ok(0));
        assert_eq!(
            parse_cache_bytes(Some("1073741824")),
            Ok(1024 * 1024 * 1024)
        );
        for value in [
            "",
            "-1",
            "+1",
            " 1",
            "1 ",
            "1GiB",
            "999999999999999999999999999999999999",
        ] {
            assert!(parse_cache_bytes(Some(value)).is_err(), "{value}");
        }
    }
}
