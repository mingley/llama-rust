//! Native NVIDIA matrix dispatch and bounded device-weight residency.
//!
//! Cache identity includes the complete packed bytes, dtype and dimensions;
//! pointer reuse or a shortlist-key collision cannot return another model's weights.
//! `LLAMA_CUDA_CACHE_BYTES` sets the per-thread residency budget once at context
//! initialization (automatic by default; zero disables caching). Automatic
//! budgets leave at least 25% of currently free VRAM or 512 MiB as headroom.
//! The budget counts device F32 weights and retained packed identity, not
//! transient launch memory.

use crate::quant::QuantError;
use llama_native_cuda::{CudaGemv, CudaMatrix};
use std::cell::RefCell;
use std::collections::VecDeque;

const MIN_HEADROOM_BYTES: usize = 512 * 1024 * 1024;
const CACHE_BYTES_ENV: &str = "LLAMA_CUDA_CACHE_BYTES";

// None denotes automatic sizing; Some(0) explicitly disables residency.
fn parse_cache_bytes(value: Option<&str>) -> Result<Option<usize>, QuantError> {
    let Some(value) = value.filter(|value| *value != "auto") else {
        return Ok(None);
    };
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(QuantError::NativeCuda(format!(
            "{CACHE_BYTES_ENV} must be auto or a nonnegative decimal byte count"
        )));
    }
    value.parse().map(Some).map_err(|_| {
        QuantError::NativeCuda(format!("{CACHE_BYTES_ENV} exceeds addressable memory"))
    })
}

fn automatic_cache_bytes(free: usize, total: usize) -> usize {
    // Clamp inconsistent reports conservatively and avoid multiplying byte
    // counts, including when testing address-space-limit values.
    let available = free.min(total);
    let reserve = (available / 4).max(MIN_HEADROOM_BYTES);
    available.saturating_sub(reserve)
}

fn admission_requires_eviction(used: usize, incoming: usize, limit: usize) -> bool {
    incoming > limit || used > limit - incoming
}

fn configured_cache_bytes() -> Result<Option<usize>, QuantError> {
    match std::env::var(CACHE_BYTES_ENV) {
        Ok(value) => parse_cache_bytes(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_cache_bytes(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(QuantError::NativeCuda(format!(
            "{CACHE_BYTES_ENV} must be auto or a nonnegative decimal byte count"
        ))),
    }
}

// A cheap shortlist only: these edge samples are never accepted as identity
// without verifying all packed bytes. This avoids hashing an entire matrix
// before scanning it a second time for exact equality on every cache hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WeightKey {
    dtype: u8,
    rows: usize,
    cols: usize,
    len: usize,
    first: [u8; 32],
    last: [u8; 32],
}

impl WeightKey {
    fn new(dtype: u8, rows: usize, cols: usize, packed: &[u8]) -> Self {
        let mut first = [0; 32];
        let mut last = [0; 32];
        for (slot, byte) in first.iter_mut().zip(packed.iter()) {
            *slot = *byte;
        }
        for (slot, byte) in last.iter_mut().zip(packed.iter().rev()) {
            *slot = *byte;
        }
        Self {
            dtype,
            rows,
            cols,
            len: packed.len(),
            first,
            last,
        }
    }
}

struct WeightIdentity {
    key: WeightKey,
    packed: Vec<u8>,
}

impl WeightIdentity {
    fn matches(&self, key: WeightKey, packed: &[u8]) -> bool {
        self.key == key && self.packed == packed
    }
}

struct Entry {
    identity: WeightIdentity,
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
            let configured = configured_cache_bytes()?;
            let gpu = CudaGemv::new(0).map_err(QuantError::NativeCuda)?;
            let limit = match configured {
                Some(bytes) => bytes,
                None => {
                    let (free, total) = gpu.memory_info().map_err(QuantError::NativeCuda)?;
                    automatic_cache_bytes(free, total)
                }
            };
            *slot = Some(State {
                gpu,
                cache: VecDeque::new(),
                bytes: 0,
                limit,
            });
        }
        let state = slot.as_mut().ok_or_else(|| {
            QuantError::NativeCuda("GPU initialization did not create a context".into())
        })?;
        let key = WeightKey::new(dtype, rows, cols, packed);
        let hit = state
            .cache
            .iter()
            .position(|entry| entry.identity.matches(key, packed));
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
        while !state.cache.is_empty()
            && admission_requires_eviction(state.bytes, bytes, state.limit)
        {
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
                identity: WeightIdentity {
                    key,
                    packed: packed.to_vec(),
                },
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
    fn shortlist_collision_never_accepts_different_middle_weights() {
        let packed = vec![7; 128];
        let key = WeightKey::new(1, 2, 64, &packed);
        let identity = WeightIdentity {
            key,
            packed: packed.clone(),
        };
        assert!(identity.matches(key, &packed));

        let mut changed = packed;
        *changed.get_mut(64).expect("middle element") = 8;
        let changed_key = WeightKey::new(1, 2, 64, &changed);
        assert_eq!(key, changed_key, "deliberate edge-sample collision");
        assert!(!identity.matches(changed_key, &changed));
        // Reusing the same allocation after mutation cannot return stale data.
        *changed.get_mut(64).expect("middle element") = 7;
        assert!(identity.matches(key, &changed));
    }

    #[test]
    fn shortlist_identity_checks_shape_dtype_length_and_short_buffers() {
        for packed in [
            vec![],
            vec![1],
            vec![1; 31],
            vec![1; 32],
            vec![1; 33],
            vec![1; 65],
        ] {
            let key = WeightKey::new(1, 2, 3, &packed);
            let identity = WeightIdentity {
                key,
                packed: packed.clone(),
            };
            assert!(identity.matches(key, &packed));
            assert!(!identity.matches(WeightKey::new(2, 2, 3, &packed), &packed));
            assert!(!identity.matches(WeightKey::new(1, 3, 2, &packed), &packed));
            let mut longer = packed.clone();
            longer.push(1);
            assert!(!identity.matches(WeightKey::new(1, 2, 3, &longer), &longer));
        }
    }

    #[test]
    #[ignore = "manual CPU cache-lookup microbenchmark; no GPU timing"]
    fn resident_weight_identity_benchmark() {
        use std::hash::{DefaultHasher, Hash, Hasher};
        use std::hint::black_box;
        use std::time::Instant;

        fn fingerprint(packed: &[u8]) -> u64 {
            let mut hasher = DefaultHasher::new();
            packed.hash(&mut hasher);
            hasher.finish()
        }

        let packed = vec![7; 8 * 1024 * 1024];
        let query = packed.clone();
        let identity = WeightIdentity {
            key: WeightKey::new(1, 1024, 2048, &packed),
            packed,
        };
        let expected_hash = fingerprint(&query);
        let repetitions = 64;
        let start = Instant::now();
        for _ in 0..repetitions {
            let query = black_box(query.as_slice());
            assert!(black_box(
                fingerprint(query) == expected_hash && identity.packed == query
            ));
        }
        let old = start.elapsed().as_nanos() / repetitions;
        let start = Instant::now();
        for _ in 0..repetitions {
            let query = black_box(query.as_slice());
            assert!(black_box(
                identity.matches(WeightKey::new(1, 1024, 2048, query), query)
            ));
        }
        let new = start.elapsed().as_nanos() / repetitions;
        println!("8 MiB resident cache hit: hash+verify={old} ns, edge-key+verify={new} ns");
    }

    #[test]
    fn residency_admission_accounting_cannot_overflow_the_byte_override() {
        assert!(!admission_requires_eviction(90, 10, 100));
        assert!(admission_requires_eviction(90, 11, 100));
        assert!(admission_requires_eviction(0, 101, 100));
        assert!(admission_requires_eviction(0, 1, 0));
        // A saturating used+incoming comparison incorrectly accepted this
        // admission when limit was usize::MAX, then overflowed its counter.
        assert!(admission_requires_eviction(usize::MAX - 10, 20, usize::MAX));
        assert!(!admission_requires_eviction(
            usize::MAX - 20,
            20,
            usize::MAX
        ));
    }

    #[test]
    fn automatic_residency_uses_available_vram_with_scratch_headroom() {
        const GIB: usize = 1024 * 1024 * 1024;
        assert_eq!(automatic_cache_bytes(8 * GIB, 16 * GIB), 6 * GIB);
        assert_eq!(automatic_cache_bytes(2 * GIB, 16 * GIB), GIB + GIB / 2);
        assert_eq!(automatic_cache_bytes(GIB, 16 * GIB), GIB / 2);
        // Below the minimum headroom, execution can still use transient
        // matrices but must not retain weights across operations.
        assert_eq!(automatic_cache_bytes(256 * 1024 * 1024, GIB), 0);
        assert_eq!(automatic_cache_bytes(0, GIB), 0);
        assert_eq!(automatic_cache_bytes(GIB, 0), 0);
        assert_eq!(automatic_cache_bytes(16 * GIB, 8 * GIB), 6 * GIB);
        assert_eq!(
            automatic_cache_bytes(usize::MAX, usize::MAX),
            usize::MAX - usize::MAX / 4
        );
    }

    #[test]
    fn cache_budget_is_explicit_and_strictly_parsed() {
        assert_eq!(parse_cache_bytes(None), Ok(None));
        assert_eq!(parse_cache_bytes(Some("auto")), Ok(None));
        assert_eq!(parse_cache_bytes(Some("0")), Ok(Some(0)));
        assert_eq!(
            parse_cache_bytes(Some("1073741824")),
            Ok(Some(1024 * 1024 * 1024))
        );
        for value in [
            "",
            "-1",
            "+1",
            " 1",
            "1 ",
            "1GiB",
            "AUTO",
            "999999999999999999999999999999999999",
        ] {
            assert!(parse_cache_bytes(Some(value)).is_err(), "{value}");
        }
    }
}
