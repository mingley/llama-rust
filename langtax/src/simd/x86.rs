//! AVX2 + FMA row kernels for F32, F16, Q4_K, Q6_K and Q8_0 on x86-64.
//!
//! # Where `unsafe` lives
//!
//! Only two kinds of `unsafe` appear below, and nothing else in the module
//! needs any: the arithmetic intrinsics are safe to call from a function that
//! already carries the matching `#[target_feature]`.
//!
//! 1. Four load helpers. Each takes a fixed-size array reference (`&[u8; 32]`,
//!    `&[u8; 16]`, `&[u8; 8]`, `&[f32; 8]`) and issues one unaligned load of
//!    exactly that many bytes. "The array owns exactly N bytes and the
//!    intrinsic reads exactly N bytes" is the entire bounds argument, and the
//!    borrow checker supplies it. The kernels obtain those arrays from
//!    `as_chunks` / `first_chunk` / `last_chunk`, so no raw offset arithmetic
//!    is involved anywhere.
//! 2. Five dispatch wrappers, which call a `#[target_feature]` kernel after
//!    [`super`] has confirmed the CPU supports it.
#![expect(
    unsafe_code,
    reason = "vendor SIMD loads are unsafe fns; this module is the only place in \
              the crate allowed to call them, and --no-default-features compiles \
              it out entirely under a crate-level forbid(unsafe_code)"
)]
#![deny(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::{
    __m128i, __m256, __m256i, _mm256_add_epi32, _mm256_add_ps, _mm256_and_si256,
    _mm256_castps256_ps128, _mm256_castsi256_ps, _mm256_castsi256_si128, _mm256_cvtepi32_ps,
    _mm256_cvtepi8_epi16, _mm256_cvtepu8_epi32, _mm256_cvtph_ps, _mm256_extractf128_ps,
    _mm256_extracti128_si256, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_loadu_si256,
    _mm256_madd_epi16, _mm256_mul_ps, _mm256_or_si256, _mm256_set1_epi32, _mm256_set1_ps,
    _mm256_setzero_ps, _mm256_slli_epi32, _mm256_srli_epi32, _mm256_sub_epi32, _mm256_sub_ps,
    _mm_add_epi32, _mm_add_ps, _mm_add_ss, _mm_and_si128, _mm_cvtsi128_si32, _mm_cvtss_f32,
    _mm_loadl_epi64, _mm_loadu_si128, _mm_movehl_ps, _mm_set1_epi8, _mm_shuffle_epi32,
    _mm_shuffle_ps, _mm_srli_epi16,
};

use crate::fp16::load_f16_le;
use crate::quant::scalar as sc;
use crate::quant::{i8_from_bits, Q4_K_BLOCK, Q6_K_BLOCK, Q8_0_BLOCK, QK_K};

/// True when this CPU has the features the kernels below are compiled for.
/// `std`'s detection macros cache their answer, so this is a load and a test.
fn have_avx2_fma() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

/// `GGML_TYPE_F32` weight row against an `f32` activation row.
pub(super) fn dot_f32_row(row: &[u8], x: &[f32]) -> f32 {
    debug_assert!(have_avx2_fma(), "AVX2+FMA kernel reached without AVX2+FMA");
    // SAFETY: `dot_f32_avx2` carries `#[target_feature(enable = "avx2", enable
    // = "fma")]`, so it may only be called on a CPU with both. This wrapper is
    // private to the module and its address escapes only through
    // `super::f32_row_dot`, which returns `Some` only when `super::caps()`
    // reported `CAP_AVX2_FMA` from `is_x86_feature_detected!`. CPU features
    // cannot be withdrawn mid-process, so a cached "yes" cannot go stale. The
    // `debug_assert` above re-derives the same fact in test and debug builds.
    unsafe { dot_f32_avx2(row, x) }
}

/// `GGML_TYPE_F16` weight row against an `f32` activation row.
pub(super) fn dot_f16_row(row: &[u8], x: &[f32]) -> f32 {
    debug_assert!(
        have_avx2_fma() && std::arch::is_x86_feature_detected!("f16c"),
        "F16C kernel reached without AVX2+FMA+F16C"
    );
    // SAFETY: as `dot_f32_row`, plus `f16c` for `_mm256_cvtph_ps`.
    // `super::f16_row_dot` requires both `CAP_AVX2_FMA` and `CAP_F16C` before
    // it will hand out this function.
    unsafe { dot_f16_avx2(row, x) }
}

/// Q4_K weight row against an `f32` activation row.
pub(super) fn dot_q4_k_f32_row(row: &[u8], x: &[f32]) -> f32 {
    debug_assert!(have_avx2_fma(), "AVX2+FMA kernel reached without AVX2+FMA");
    // SAFETY: as `dot_f32_row`; reachable only through
    // `super::q4_k_f32_row_dot`, which checks `CAP_AVX2_FMA`.
    unsafe { dot_q4_k_f32_avx2(row, x) }
}

/// Q6_K weight row against an `f32` activation row.
pub(super) fn dot_q6_k_f32_row(row: &[u8], x: &[f32]) -> f32 {
    debug_assert!(have_avx2_fma(), "AVX2+FMA kernel reached without AVX2+FMA");
    // SAFETY: as `dot_f32_row`; reachable only through
    // `super::q6_k_f32_row_dot`, which checks `CAP_AVX2_FMA`.
    unsafe { dot_q6_k_f32_avx2(row, x) }
}

/// Q8_0 weight row against a Q8_0 activation row.
pub(super) fn dot_q8_0_row(row: &[u8], x: &[u8]) -> f32 {
    debug_assert!(have_avx2_fma(), "AVX2+FMA kernel reached without AVX2+FMA");
    // SAFETY: as `dot_f32_row`; reachable only through `super::q8_0_row_dot`,
    // which checks `CAP_AVX2_FMA`.
    unsafe { dot_q8_0_avx2(row, x) }
}

/// Load the 32 bytes of `bytes` into one 256-bit register.
#[inline]
#[target_feature(enable = "avx2")]
fn load_u8x32(bytes: &[u8; 32]) -> __m256i {
    // SAFETY: `_mm256_loadu_si256` reads exactly 32 bytes and imposes no
    // alignment requirement. `bytes` is a live `&[u8; 32]`, so all 32 bytes are
    // readable and initialized for the duration of the borrow.
    unsafe { _mm256_loadu_si256(bytes.as_ptr().cast::<__m256i>()) }
}

/// Load the 16 bytes of `bytes` into one 128-bit register.
#[inline]
#[target_feature(enable = "avx2")]
fn load_u8x16(bytes: &[u8; 16]) -> __m128i {
    // SAFETY: `_mm_loadu_si128` reads exactly 16 bytes and imposes no alignment
    // requirement. `bytes` is a live `&[u8; 16]`.
    unsafe { _mm_loadu_si128(bytes.as_ptr().cast::<__m128i>()) }
}

/// Load the 8 bytes of `bytes` into the low half of a 128-bit register.
#[inline]
#[target_feature(enable = "avx2")]
fn load_u8x8(bytes: &[u8; 8]) -> __m128i {
    // SAFETY: `_mm_loadl_epi64` reads exactly 8 bytes and imposes no alignment
    // requirement. `bytes` is a live `&[u8; 8]`.
    unsafe { _mm_loadl_epi64(bytes.as_ptr().cast::<__m128i>()) }
}

/// Load eight contiguous `f32`.
#[inline]
#[target_feature(enable = "avx2")]
fn load_f32x8(vals: &[f32; 8]) -> __m256 {
    // SAFETY: `_mm256_loadu_ps` reads exactly 32 bytes with no alignment
    // requirement. `vals` is a live `&[f32; 8]`, which owns
    // `8 * size_of::<f32>() == 32` initialized bytes.
    unsafe { _mm256_loadu_ps(vals.as_ptr()) }
}

/// Sum the eight `f32` lanes of `v`.
#[inline]
#[target_feature(enable = "avx2")]
fn hsum_ps(v: __m256) -> f32 {
    let quad = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    let pair = _mm_add_ps(quad, _mm_movehl_ps(quad, quad));
    let one = _mm_add_ss(pair, _mm_shuffle_ps::<0x55>(pair, pair));
    _mm_cvtss_f32(one)
}

/// Sum the eight `i32` lanes of `v`. Exact: `i32` addition does not round.
#[inline]
#[target_feature(enable = "avx2")]
fn hsum_epi32(v: __m256i) -> i32 {
    let quad = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let pair = _mm_add_epi32(quad, _mm_shuffle_epi32::<0b0100_1110>(quad));
    let one = _mm_add_epi32(pair, _mm_shuffle_epi32::<0b1011_0001>(pair));
    _mm_cvtsi128_si32(one)
}

/// Elements addressable in both a weight row of `row_bytes` bytes at
/// `stride` bytes per element and an activation row of `x_len` floats.
fn common_len(row_bytes: usize, stride: usize, x_len: usize) -> usize {
    (row_bytes / stride).min(x_len)
}

/// # Safety
///
/// The caller must run on a CPU with `avx2` and `fma`.
#[target_feature(enable = "avx2", enable = "fma")]
fn dot_f32_avx2(row: &[u8], x: &[f32]) -> f32 {
    let n = common_len(row.len(), 4, x.len());
    let (Some(wr), Some(xr)) = (row.get(..n.saturating_mul(4)), x.get(..n)) else {
        return 0.0;
    };
    // `wr` holds `4 * n` bytes and `xr` holds `n` floats, so both chunk
    // iterators yield exactly `n / 8` items and both tails exactly `n % 8`
    // elements: the vector body and the scalar tail partition the row.
    let (w_chunks, w_tail) = wr.as_chunks::<32>();
    let (x_chunks, x_tail) = xr.as_chunks::<8>();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut pairs = w_chunks.iter().zip(x_chunks.iter());
    while let Some((wa, xa)) = pairs.next() {
        acc0 = _mm256_fmadd_ps(_mm256_castsi256_ps(load_u8x32(wa)), load_f32x8(xa), acc0);
        let Some((wb, xb)) = pairs.next() else { break };
        acc1 = _mm256_fmadd_ps(_mm256_castsi256_ps(load_u8x32(wb)), load_f32x8(xb), acc1);
    }
    let mut sum = hsum_ps(_mm256_add_ps(acc0, acc1));
    for (chunk, xv) in w_tail.as_chunks::<4>().0.iter().zip(x_tail.iter()) {
        sum += f32::from_bits(u32::from_le_bytes(*chunk)) * *xv;
    }
    sum
}

/// # Safety
///
/// The caller must run on a CPU with `avx2`, `fma` and `f16c`.
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
fn dot_f16_avx2(row: &[u8], x: &[f32]) -> f32 {
    let n = common_len(row.len(), 2, x.len());
    let (Some(wr), Some(xr)) = (row.get(..n.saturating_mul(2)), x.get(..n)) else {
        return 0.0;
    };
    // As in `dot_f32_avx2`: `n / 8` full vectors plus an `n % 8` scalar tail.
    //
    // `_mm256_cvtph_ps` is the IEEE binary16 -> `f32` conversion in hardware,
    // which is exact for every finite and infinite input and therefore agrees
    // with `crate::fp16::f16_to_f32` bit for bit. The two differ only in the
    // payload of a signaling NaN, which the hardware quiets; that cannot appear
    // in valid GGUF weight data, and either way the dot product is NaN.
    let (w_chunks, w_tail) = wr.as_chunks::<16>();
    let (x_chunks, x_tail) = xr.as_chunks::<8>();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut pairs = w_chunks.iter().zip(x_chunks.iter());
    while let Some((wa, xa)) = pairs.next() {
        acc0 = _mm256_fmadd_ps(_mm256_cvtph_ps(load_u8x16(wa)), load_f32x8(xa), acc0);
        let Some((wb, xb)) = pairs.next() else { break };
        acc1 = _mm256_fmadd_ps(_mm256_cvtph_ps(load_u8x16(wb)), load_f32x8(xb), acc1);
    }
    let mut sum = hsum_ps(_mm256_add_ps(acc0, acc1));
    for (chunk, xv) in w_tail.as_chunks::<2>().0.iter().zip(x_tail.iter()) {
        sum += load_f16_le(chunk).unwrap_or(0.0) * *xv;
    }
    sum
}

/// # Safety
///
/// The caller must run on a CPU with `avx2` and `fma`.
#[target_feature(enable = "avx2", enable = "fma")]
fn dot_q4_k_f32_avx2(row: &[u8], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let (w_blocks, _) = row.as_chunks::<Q4_K_BLOCK>();
    for (b, wb) in w_blocks.iter().enumerate() {
        let Some(d) = load_f16_le(wb) else { continue };
        let Some(dmin) = load_f16_le(wb.get(2..).unwrap_or(&[])) else {
            continue;
        };
        let Some(scales) = wb.get(4..16) else {
            continue;
        };
        let Some(qs) = wb.get(16..144) else { continue };
        let x_base = b.saturating_mul(QK_K);
        let Some(xr) = x.get(x_base..x_base.saturating_add(QK_K)) else {
            continue;
        };
        let mut acc = _mm256_setzero_ps();
        let mask = _mm_set1_epi8(0x0f);
        for group in 0..4usize {
            let Some((sc0, m0)) = sc::q4_k_scale_min(scales, group * 2) else {
                continue;
            };
            let Some((sc1, m1)) = sc::q4_k_scale_min(scales, group * 2 + 1) else {
                continue;
            };
            let Some(packed) = qs.get(group * 32..group * 32 + 32) else {
                continue;
            };
            let xb = group * 64;
            let Some(xlo) = xr.get(xb..xb + 32) else {
                continue;
            };
            let Some(xhi) = xr.get(xb + 32..xb + 64) else {
                continue;
            };
            // Broadcast the sub-block affine terms once per 32 nibble pairs.
            let a0 = _mm256_set1_ps(d * f32::from(sc0));
            let b0 = _mm256_set1_ps(dmin * f32::from(m0));
            let a1 = _mm256_set1_ps(d * f32::from(sc1));
            let b1 = _mm256_set1_ps(dmin * f32::from(m1));
            let quads = packed
                .as_chunks::<8>()
                .0
                .iter()
                .zip(xlo.as_chunks::<8>().0.iter())
                .zip(xhi.as_chunks::<8>().0.iter());
            for ((pk, xl), xh) in quads {
                let nib = load_u8x8(pk);
                let lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(nib, mask)));
                let hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(
                    _mm_srli_epi16::<4>(nib),
                    mask,
                )));
                // Separate multiply and subtract, not an FMA: each dequantized
                // weight is then bit-identical to the scalar kernel's, and only
                // the accumulation below reassociates.
                let wlo = _mm256_sub_ps(_mm256_mul_ps(a0, lo), b0);
                let whi = _mm256_sub_ps(_mm256_mul_ps(a1, hi), b1);
                acc = _mm256_fmadd_ps(wlo, load_f32x8(xl), acc);
                acc = _mm256_fmadd_ps(whi, load_f32x8(xh), acc);
            }
        }
        sum += hsum_ps(acc);
    }
    sum
}

/// # Safety
///
/// The caller must run on a CPU with `avx2` and `fma`.
#[target_feature(enable = "avx2", enable = "fma")]
fn dot_q6_k_f32_avx2(row: &[u8], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let (w_blocks, _) = row.as_chunks::<Q6_K_BLOCK>();
    for (b, wb) in w_blocks.iter().enumerate() {
        let Some(d) = load_f16_le(wb.get(208..).unwrap_or(&[])) else {
            continue;
        };
        let Some(ql) = wb.get(..128) else { continue };
        let Some(qh) = wb.get(128..192) else { continue };
        let Some(scb) = wb.get(192..208) else {
            continue;
        };
        let x_base = b.saturating_mul(QK_K);
        let Some(xr) = x.get(x_base..x_base.saturating_add(QK_K)) else {
            continue;
        };
        let mut acc = _mm256_setzero_ps();
        let nib = _mm256_set1_epi32(0x0f);
        let pair = _mm256_set1_epi32(3);
        let bias = _mm256_set1_epi32(32);
        for group in 0..2usize {
            let ql_off = group * 64;
            let qh_off = group * 32;
            let sc_off = group * 8;
            let y_off = group * 128;
            // Eight elements per step. `is = l / 16` is constant across a step
            // because 8 divides 16.
            for step in 0..4usize {
                let l = step * 8;
                let is = l / 16;
                let (Some(s0), Some(s2), Some(s4), Some(s6)) = (
                    scb.get(sc_off + is),
                    scb.get(sc_off + is + 2),
                    scb.get(sc_off + is + 4),
                    scb.get(sc_off + is + 6),
                ) else {
                    continue;
                };
                let (Some(lo8), Some(hi8), Some(h8)) = (
                    ql.get(ql_off + l..).and_then(<[u8]>::first_chunk::<8>),
                    ql.get(ql_off + l + 32..).and_then(<[u8]>::first_chunk::<8>),
                    qh.get(qh_off + l..).and_then(<[u8]>::first_chunk::<8>),
                ) else {
                    continue;
                };
                let (Some(x1), Some(x2), Some(x3), Some(x4)) = (
                    xr.get(y_off + l..).and_then(<[f32]>::first_chunk::<8>),
                    xr.get(y_off + l + 32..).and_then(<[f32]>::first_chunk::<8>),
                    xr.get(y_off + l + 64..).and_then(<[f32]>::first_chunk::<8>),
                    xr.get(y_off + l + 96..).and_then(<[f32]>::first_chunk::<8>),
                ) else {
                    continue;
                };
                let ds0 = _mm256_set1_ps(d * f32::from(i8_from_bits(*s0)));
                let ds2 = _mm256_set1_ps(d * f32::from(i8_from_bits(*s2)));
                let ds4 = _mm256_set1_ps(d * f32::from(i8_from_bits(*s4)));
                let ds6 = _mm256_set1_ps(d * f32::from(i8_from_bits(*s6)));
                let lo = _mm256_cvtepu8_epi32(load_u8x8(lo8));
                let hi = _mm256_cvtepu8_epi32(load_u8x8(hi8));
                let h = _mm256_cvtepu8_epi32(load_u8x8(h8));
                // `q6_from_bits` in lanes: the 6-bit code is at most 63, so the
                // `- 32` recentring cannot wrap and the `i32` lanes hold the
                // same value the scalar kernel derives.
                let q1 = q6_lane(_mm256_and_si256(lo, nib), h, 0, pair, bias);
                let q2 = q6_lane(_mm256_and_si256(hi, nib), h, 2, pair, bias);
                let q3 = q6_lane(_mm256_srli_epi32::<4>(lo), h, 4, pair, bias);
                let q4 = q6_lane(_mm256_srli_epi32::<4>(hi), h, 6, pair, bias);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(ds0, q1), load_f32x8(x1), acc);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(ds2, q2), load_f32x8(x2), acc);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(ds4, q3), load_f32x8(x3), acc);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(ds6, q4), load_f32x8(x4), acc);
            }
        }
        sum += hsum_ps(acc);
    }
    sum
}

/// `(low | ((qh >> shift) & 3) << 4) - 32` per lane, as `f32`.
///
/// `shift` is 0, 2, 4 or 6; the match spells the four cases out because
/// `_mm256_srli_epi32` takes its count as a const generic.
#[inline]
#[target_feature(enable = "avx2")]
fn q6_lane(low: __m256i, qh: __m256i, shift: u32, mask2: __m256i, bias: __m256i) -> __m256 {
    let hi = match shift {
        0 => qh,
        2 => _mm256_srli_epi32::<2>(qh),
        4 => _mm256_srli_epi32::<4>(qh),
        _ => _mm256_srli_epi32::<6>(qh),
    };
    let bits = _mm256_or_si256(low, _mm256_slli_epi32::<4>(_mm256_and_si256(hi, mask2)));
    _mm256_cvtepi32_ps(_mm256_sub_epi32(bits, bias))
}

/// # Safety
///
/// The caller must run on a CPU with `avx2` and `fma`.
#[target_feature(enable = "avx2", enable = "fma")]
fn dot_q8_0_avx2(row: &[u8], x: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    let (w_blocks, _) = row.as_chunks::<Q8_0_BLOCK>();
    let (x_blocks, _) = x.as_chunks::<Q8_0_BLOCK>();
    for (wb, xb) in w_blocks.iter().zip(x_blocks.iter()) {
        let Some(dw) = load_f16_le(wb) else { continue };
        let Some(dx) = load_f16_le(xb) else { continue };
        // A Q8_0 block is a binary16 scale followed by exactly 32 `i8`.
        let (Some(wqs), Some(xqs)) = (wb.last_chunk::<32>(), xb.last_chunk::<32>()) else {
            continue;
        };
        // Bit-identical to the scalar kernel. `_mm256_cvtepi8_epi16`
        // sign-extends, `_mm256_madd_epi16` forms products of magnitude at most
        // 128 * 128 = 16_384 and adds them pairwise, and the 32 products of one
        // block sum to at most 524_288 in magnitude. Nothing saturates, nothing
        // overflows `i32`, and `i32` addition is associative, so neither the
        // pairing nor the lane order can change the total.
        let wv = load_u8x32(wqs);
        let xv = load_u8x32(xqs);
        let prod = _mm256_add_epi32(
            _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(wv)),
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(xv)),
            ),
            _mm256_madd_epi16(
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(wv)),
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(xv)),
            ),
        );
        sum = sc::add_f32(sum, (hsum_epi32(prod) as f32) * (dw * dx));
    }
    sum
}
