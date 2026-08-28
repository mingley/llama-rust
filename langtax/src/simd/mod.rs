//! Opt-in SIMD fast path for the five dtypes a Q4_K_M-class GGUF touches in
//! bulk: F32, F16, Q4_K, Q6_K and Q8_0. The other 25 ggml dtypes keep running
//! on the scalar kernels in [`crate::quant`].
//!
//! # Shape of the fast path
//!
//! Every kernel here is a *row dot*: one weight row of raw GGUF block bytes
//! against one activation row. It has the same signature as the scalar kernel
//! it replaces, so [`crate::quant`] resolves a function pointer once per
//! GEMV/GEMM call and the row loop is unchanged.
//!
//! CPU feature detection runs at most once per process and is cached in
//! [`CAPS`]. The `*_row_dot` selectors read that cache and hand back `None`
//! when no vector kernel applies, which is the signal to use the scalar
//! kernel. Nothing probes the CPU per row.
//!
//! # Safety posture
//!
//! `unsafe` lives only in the `#[cfg(target_arch)]` submodules, which opt in
//! with a module-level `expect(unsafe_code)` and a written reason. Building
//! with `--no-default-features` drops this module entirely and restores a
//! crate-level `forbid(unsafe_code)`.
//!
//! Miri interprets the x86-64 intrinsics used here, so CI runs the differential
//! tests under `-Zmiri-strict-provenance` against the real AVX2 kernels rather
//! than merely compiling them. It has to be told the CPU features exist
//! (`-C target-feature=+avx2,+fma,+f16c`), because under the interpreter
//! `is_x86_feature_detected!` otherwise says no and every selector returns
//! `None`. Miri does not implement the NEON reductions, so the aarch64 kernels
//! rest on the differential tests alone, run natively and under qemu.
//!
//! # Numerics
//!
//! The Q8_0 kernels accumulate in `i32`, which is exact, so they are
//! bit-identical to the scalar kernel. The F32/F16/Q4_K/Q6_K kernels keep one
//! accumulator per vector lane and use fused multiply-add, so they reassociate
//! the summation and skip one intermediate rounding. Both kernels compute the
//! same exact-arithmetic quantity; see `tests` for the error bound that the
//! differential tests enforce.

use std::sync::atomic::{AtomicU8, Ordering};

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod aarch64;
#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
mod x86;

#[cfg(all(
    test,
    target_endian = "little",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
mod tests;

/// Row kernel over GGUF weight bytes and an `f32` activation row.
pub(crate) type RowDotF32 = fn(&[u8], &[f32]) -> f32;

/// Row kernel over Q8_0 weight bytes and a Q8_0 activation row.
pub(crate) type RowDotQ8 = fn(&[u8], &[u8]) -> f32;

/// Sentinel for "detection has not run yet". Detection always sets
/// [`CAP_DETECTED`], so a cached value is never zero.
const CAP_NONE: u8 = 0;
/// Set once detection has run, even when nothing was found.
const CAP_DETECTED: u8 = 1 << 0;
/// x86-64: `avx2` and `fma` are both present.
#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
const CAP_AVX2_FMA: u8 = 1 << 1;
/// x86-64: `f16c` is present (binary16 -> `f32` conversion).
#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
const CAP_F16C: u8 = 1 << 2;
/// aarch64: Advanced SIMD is present.
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
const CAP_NEON: u8 = 1 << 3;

/// Cached CPU capabilities. Written at most once with a value that always has
/// [`CAP_DETECTED`] set; racing writers agree because detection is pure.
static CAPS: AtomicU8 = AtomicU8::new(CAP_NONE);

/// CPU capabilities, detected on first call and cached for the process.
fn caps() -> u8 {
    let cached = CAPS.load(Ordering::Relaxed);
    if cached != CAP_NONE {
        return cached;
    }
    let found = detect() | CAP_DETECTED;
    CAPS.store(found, Ordering::Relaxed);
    found
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
fn detect() -> u8 {
    let mut found = CAP_NONE;
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        found |= CAP_AVX2_FMA;
    }
    if std::arch::is_x86_feature_detected!("f16c") {
        found |= CAP_F16C;
    }
    found
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
fn detect() -> u8 {
    if std::arch::is_aarch64_feature_detected!("neon") {
        CAP_NEON
    } else {
        CAP_NONE
    }
}

/// Targets with no kernels here: every selector returns `None` and callers stay
/// on the scalar path.
#[cfg(not(all(
    target_endian = "little",
    any(target_arch = "x86_64", target_arch = "aarch64")
)))]
fn detect() -> u8 {
    CAP_NONE
}

/// Vector kernel for `GGML_TYPE_F32` weights, or `None` for the scalar path.
pub(crate) fn f32_row_dot() -> Option<RowDotF32> {
    let found = caps();
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    if found & CAP_AVX2_FMA != 0 {
        return Some(x86::dot_f32_row);
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if found & CAP_NEON != 0 {
        return Some(aarch64::dot_f32_row);
    }
    let _ = found;
    None
}

/// Vector kernel for `GGML_TYPE_F16` weights, or `None` for the scalar path.
pub(crate) fn f16_row_dot() -> Option<RowDotF32> {
    let found = caps();
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    if found & CAP_AVX2_FMA != 0 && found & CAP_F16C != 0 {
        return Some(x86::dot_f16_row);
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if found & CAP_NEON != 0 {
        return Some(aarch64::dot_f16_row);
    }
    let _ = found;
    None
}

/// Vector kernel for `GGML_TYPE_Q4_K` weights, or `None` for the scalar path.
pub(crate) fn q4_k_f32_row_dot() -> Option<RowDotF32> {
    let found = caps();
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    if found & CAP_AVX2_FMA != 0 {
        return Some(x86::dot_q4_k_f32_row);
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if found & CAP_NEON != 0 {
        return Some(aarch64::dot_q4_k_f32_row);
    }
    let _ = found;
    None
}

/// Vector kernel for `GGML_TYPE_Q6_K` weights, or `None` for the scalar path.
pub(crate) fn q6_k_f32_row_dot() -> Option<RowDotF32> {
    let found = caps();
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    if found & CAP_AVX2_FMA != 0 {
        return Some(x86::dot_q6_k_f32_row);
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if found & CAP_NEON != 0 {
        return Some(aarch64::dot_q6_k_f32_row);
    }
    let _ = found;
    None
}

/// Vector kernel for Q8_0 weights against a Q8_0 activation row, or `None` for
/// the scalar path. Bit-identical to the scalar kernel.
pub(crate) fn q8_0_row_dot() -> Option<RowDotQ8> {
    let found = caps();
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    if found & CAP_AVX2_FMA != 0 && found & CAP_F16C != 0 {
        return Some(x86::dot_q8_0_row);
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if found & CAP_NEON != 0 {
        return Some(aarch64::dot_q8_0_row);
    }
    let _ = found;
    None
}
