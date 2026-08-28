//! NEON row kernels for F32, F16, Q4_K, Q5_0, Q5_1, Q6_K and Q8_0 on aarch64.
//!
//! # Where `unsafe` lives
//!
//! The same two places as the x86-64 module, and nowhere else:
//!
//! 1. Three load helpers, each taking a fixed-size array reference
//!    (`&[u8; 16]`, `&[u8; 8]`, `&[f32; 4]`) and issuing one load of exactly
//!    that many bytes. The array reference is the bounds proof. NEON `LD1` has
//!    no alignment requirement beyond the element type, which a live reference
//!    to the array already guarantees. All arrays come from `as_chunks` /
//!    `first_chunk` / `last_chunk`; no raw offset arithmetic appears anywhere.
//! 2. Eight dispatch wrappers, which call a `#[target_feature]` kernel after
//!    [`super`] has confirmed the CPU supports Advanced SIMD.
//!
//! Everything else is register-only and safe to call from a function that
//! already carries `#[target_feature(enable = "neon")]`.
#![expect(
    unsafe_code,
    reason = "vendor SIMD loads are unsafe fns; this module is the only place in \
              the crate allowed to call them, and --no-default-features compiles \
              it out entirely under a crate-level forbid(unsafe_code)"
)]
#![deny(unsafe_op_in_unsafe_fn)]

use core::arch::aarch64::{
    float32x4_t, int16x8_t, int32x4_t, int8x16_t, int8x8_t, uint32x4_t, uint8x16_t, uint8x8_t,
    vaddq_f32, vaddq_s32, vaddvq_f32, vaddvq_s32, vand_u8, vandq_s32, vandq_u32, vceqq_u32,
    vcvt_f32_f16, vcvtq_f32_s32, vcvtq_f32_u32, vdup_n_u8, vdupq_laneq_f32, vdupq_n_f32,
    vdupq_n_s32, vdupq_n_u32, vfmaq_f32, vget_high_s16, vget_high_s8, vget_high_u16, vget_low_s16,
    vget_low_s8, vget_low_u16, vgetq_lane_f32, vld1_u8, vld1q_f32, vld1q_u8, vmovl_s16, vmovl_s8,
    vmovl_u16, vmovl_u8, vmull_s8, vmulq_f32, vorrq_u32, vpadalq_s16, vreinterpret_f16_u16,
    vreinterpret_s8_u8, vreinterpret_u16_u8, vreinterpretq_f32_u8, vreinterpretq_s32_u32,
    vreinterpretq_s8_u8, vsetq_lane_s32, vshlq_n_u32, vshlq_u32, vshr_n_u8, vshrq_n_u32, vsubq_f32,
    vsubq_s32,
};

use crate::fp16::load_f16_le;
use crate::quant::scalar as sc;
use crate::quant::{
    i8_from_bits, Q4_K_BLOCK, Q5_0_BLOCK, Q5_1_BLOCK, Q6_K_BLOCK, Q8_0_BLOCK, QK5_0, QK5_1, QK8_0,
    QK_K,
};

/// True when this CPU reports Advanced SIMD. `std`'s detection macro caches its
/// answer, so this is a load and a test.
fn have_neon() -> bool {
    std::arch::is_aarch64_feature_detected!("neon")
}

/// `GGML_TYPE_F32` weight row against an `f32` activation row.
pub(super) fn dot_f32_row(row: &[u8], x: &[f32]) -> f32 {
    debug_assert!(have_neon(), "NEON kernel reached without NEON");
    // SAFETY: `dot_f32_neon` carries `#[target_feature(enable = "neon")]`, so
    // it may only be called on a CPU with Advanced SIMD. This wrapper is
    // private to the module and its address escapes only through
    // `super::f32_row_dot`, which returns `Some` only when `super::caps()`
    // reported `CAP_NEON` from `is_aarch64_feature_detected!`. CPU features
    // cannot be withdrawn mid-process, so a cached "yes" cannot go stale. The
    // `debug_assert` above re-derives the same fact in test and debug builds.
    unsafe { dot_f32_neon(row, x) }
}

/// `GGML_TYPE_F16` weight row against an `f32` activation row.
pub(super) fn dot_f16_row(row: &[u8], x: &[f32]) -> f32 {
    debug_assert!(have_neon(), "NEON kernel reached without NEON");
    // SAFETY: as `dot_f32_row`; reachable only through `super::f16_row_dot`.
    unsafe { dot_f16_neon(row, x) }
}

/// Q4_K weight row against an `f32` activation row.
pub(super) fn dot_q4_k_f32_row(row: &[u8], x: &[f32]) -> f32 {
    debug_assert!(have_neon(), "NEON kernel reached without NEON");
    // SAFETY: as `dot_f32_row`; reachable only through
    // `super::q4_k_f32_row_dot`.
    unsafe { dot_q4_k_f32_neon(row, x) }
}

/// Q6_K weight row against an `f32` activation row.
pub(super) fn dot_q6_k_f32_row(row: &[u8], x: &[f32]) -> f32 {
    debug_assert!(have_neon(), "NEON kernel reached without NEON");
    // SAFETY: as `dot_f32_row`; reachable only through
    // `super::q6_k_f32_row_dot`.
    unsafe { dot_q6_k_f32_neon(row, x) }
}

/// Q5_0 weight row against an `f32` activation row.
pub(super) fn dot_q5_0_f32_row(row: &[u8], x: &[f32]) -> f32 {
    debug_assert!(have_neon(), "NEON kernel reached without NEON");
    // SAFETY: as `dot_f32_row`. Reachable only through
    // `super::q5_0_f32_row_dot`, which checks `CAP_NEON`.
    unsafe { dot_q5_0_f32_neon(row, x) }
}

/// Q5_1 weight row against an `f32` activation row.
pub(super) fn dot_q5_1_f32_row(row: &[u8], x: &[f32]) -> f32 {
    debug_assert!(have_neon(), "NEON kernel reached without NEON");
    // SAFETY: as `dot_f32_row`. Reachable only through
    // `super::q5_1_f32_row_dot`, which checks `CAP_NEON`.
    unsafe { dot_q5_1_f32_neon(row, x) }
}

/// Q8_0 weight row against an `f32` activation row. This is the model path;
/// [`dot_q8_0_row`] is the Q8_0-activation kernel the GEMV benchmark uses.
pub(super) fn dot_q8_0_f32_row(row: &[u8], x: &[f32]) -> f32 {
    debug_assert!(have_neon(), "NEON kernel reached without NEON");
    // SAFETY: as `dot_f32_row`. Reachable only through
    // `super::q8_0_f32_row_dot`, which checks `CAP_NEON`.
    unsafe { dot_q8_0_f32_neon(row, x) }
}

/// Q8_0 weight row against a Q8_0 activation row.
pub(super) fn dot_q8_0_row(row: &[u8], x: &[u8]) -> f32 {
    debug_assert!(have_neon(), "NEON kernel reached without NEON");
    // SAFETY: as `dot_f32_row`; reachable only through `super::q8_0_row_dot`.
    unsafe { dot_q8_0_neon(row, x) }
}

/// Load the 16 bytes of `bytes`.
#[inline]
#[target_feature(enable = "neon")]
fn load_u8x16(bytes: &[u8; 16]) -> uint8x16_t {
    // SAFETY: `vld1q_u8` reads exactly 16 bytes and, being a `u8` load, needs
    // only byte alignment. `bytes` is a live `&[u8; 16]`, so all 16 bytes are
    // readable and initialized for the duration of the borrow.
    unsafe { vld1q_u8(bytes.as_ptr()) }
}

/// Load the 8 bytes of `bytes`.
#[inline]
#[target_feature(enable = "neon")]
fn load_u8x8(bytes: &[u8; 8]) -> uint8x8_t {
    // SAFETY: `vld1_u8` reads exactly 8 bytes and needs only byte alignment.
    // `bytes` is a live `&[u8; 8]`.
    unsafe { vld1_u8(bytes.as_ptr()) }
}

/// Load four contiguous `f32`.
#[inline]
#[target_feature(enable = "neon")]
fn load_f32x4(vals: &[f32; 4]) -> float32x4_t {
    // SAFETY: `vld1q_f32` reads exactly `4 * size_of::<f32>() == 16` bytes and
    // needs only `f32` alignment. `vals` is a live `&[f32; 4]`, which is both
    // long enough and correctly aligned by construction.
    unsafe { vld1q_f32(vals.as_ptr()) }
}

/// Widen eight `u8` lanes to two vectors of four `u32`.
#[inline]
#[target_feature(enable = "neon")]
fn widen_u8x8(v: uint8x8_t) -> (uint32x4_t, uint32x4_t) {
    let wide = vmovl_u8(v);
    (
        vmovl_u16(vget_low_u16(wide)),
        vmovl_u16(vget_high_u16(wide)),
    )
}

/// Elements addressable in both a weight row of `row_bytes` bytes at
/// `stride` bytes per element and an activation row of `x_len` floats.
fn common_len(row_bytes: usize, stride: usize, x_len: usize) -> usize {
    (row_bytes / stride).min(x_len)
}

/// # Safety
///
/// The caller must run on a CPU with Advanced SIMD.
#[target_feature(enable = "neon")]
fn dot_f32_neon(row: &[u8], x: &[f32]) -> f32 {
    let n = common_len(row.len(), 4, x.len());
    let (Some(wr), Some(xr)) = (row.get(..n.saturating_mul(4)), x.get(..n)) else {
        return 0.0;
    };
    // `wr` holds `4 * n` bytes and `xr` holds `n` floats, so both chunk
    // iterators yield exactly `n / 4` items and both tails exactly `n % 4`
    // elements: the vector body and the scalar tail partition the row.
    //
    // Little-endian is a module precondition (`super` gates this file on
    // `target_endian = "little"`), so reinterpreting the loaded bytes as `f32`
    // reproduces `f32::from_bits(u32::from_le_bytes(..))` lane for lane.
    let (w_chunks, w_tail) = wr.as_chunks::<16>();
    let (x_chunks, x_tail) = xr.as_chunks::<4>();
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let mut quads = w_chunks.iter().zip(x_chunks.iter());
    while let Some((wa, xa)) = quads.next() {
        acc0 = vfmaq_f32(acc0, vreinterpretq_f32_u8(load_u8x16(wa)), load_f32x4(xa));
        let Some((wb, xb)) = quads.next() else { break };
        acc1 = vfmaq_f32(acc1, vreinterpretq_f32_u8(load_u8x16(wb)), load_f32x4(xb));
        let Some((wc, xc)) = quads.next() else { break };
        acc2 = vfmaq_f32(acc2, vreinterpretq_f32_u8(load_u8x16(wc)), load_f32x4(xc));
        let Some((wd, xd)) = quads.next() else { break };
        acc3 = vfmaq_f32(acc3, vreinterpretq_f32_u8(load_u8x16(wd)), load_f32x4(xd));
    }
    let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for (chunk, xv) in w_tail.as_chunks::<4>().0.iter().zip(x_tail.iter()) {
        sum += f32::from_bits(u32::from_le_bytes(*chunk)) * *xv;
    }
    sum
}

/// # Safety
///
/// The caller must run on a CPU with Advanced SIMD.
#[target_feature(enable = "neon")]
fn dot_f16_neon(row: &[u8], x: &[f32]) -> f32 {
    let n = common_len(row.len(), 2, x.len());
    let (Some(wr), Some(xr)) = (row.get(..n.saturating_mul(2)), x.get(..n)) else {
        return 0.0;
    };
    // As in `dot_f32_neon`: `n / 4` full vectors plus an `n % 4` scalar tail.
    //
    // `FCVTL` is the IEEE binary16 -> `f32` widening in hardware. It is exact
    // for every finite and infinite input, including binary16 subnormals (which
    // are ordinary `f32` normals, so `FPCR.FZ16` cannot affect the result), and
    // therefore agrees with `crate::fp16::f16_to_f32` bit for bit. The two
    // differ only in the payload of a signaling NaN, which the hardware quiets;
    // that cannot appear in valid GGUF weight data, and either way the dot
    // product is NaN.
    let (w_chunks, w_tail) = wr.as_chunks::<8>();
    let (x_chunks, x_tail) = xr.as_chunks::<4>();
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let mut quads = w_chunks.iter().zip(x_chunks.iter());
    while let Some((wa, xa)) = quads.next() {
        acc0 = vfmaq_f32(acc0, widen_f16x4(wa), load_f32x4(xa));
        let Some((wb, xb)) = quads.next() else { break };
        acc1 = vfmaq_f32(acc1, widen_f16x4(wb), load_f32x4(xb));
        let Some((wc, xc)) = quads.next() else { break };
        acc2 = vfmaq_f32(acc2, widen_f16x4(wc), load_f32x4(xc));
        let Some((wd, xd)) = quads.next() else { break };
        acc3 = vfmaq_f32(acc3, widen_f16x4(wd), load_f32x4(xd));
    }
    let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for (chunk, xv) in w_tail.as_chunks::<2>().0.iter().zip(x_tail.iter()) {
        sum += load_f16_le(chunk).unwrap_or(0.0) * *xv;
    }
    sum
}

/// Widen four little-endian binary16 values to `f32`.
#[inline]
#[target_feature(enable = "neon")]
fn widen_f16x4(bytes: &[u8; 8]) -> float32x4_t {
    vcvt_f32_f16(vreinterpret_f16_u16(vreinterpret_u16_u8(load_u8x8(bytes))))
}

/// The product of two little-endian binary16 block scales.
///
/// `FCVTL` is exact for every finite and infinite binary16, including
/// subnormals, so each lane matches `crate::fp16::f16_to_f32` bit for bit; the
/// two differ only in the payload of a signaling NaN, which the hardware
/// quiets. Doing it this way rather than through the software conversion is
/// worth about 2x on this kernel, because Q8_0 decodes two scales for every 32
/// elements.
#[inline]
#[target_feature(enable = "neon")]
fn scale_product(w: [u8; 2], x: [u8; 2]) -> f32 {
    let [w0, w1] = w;
    let [x0, x1] = x;
    let both = widen_f16x4(&[w0, w1, x0, x1, 0, 0, 0, 0]);
    // Lane 0 is the weight scale and lane 1 the activation scale, so this is
    // `dw * dx`, the same product in the same order as the scalar kernel.
    vgetq_lane_f32::<0>(both) * vgetq_lane_f32::<1>(both)
}

/// One little-endian binary16 block scale, decoded in hardware and broadcast
/// across all four lanes.
///
/// Same exactness argument as [`scale_product`]. Q5_0, Q5_1 and Q8_0 all carry
/// one or two scales per 32 elements, dense enough that the software
/// conversion would otherwise dominate.
///
/// The result stays in a vector register throughout. Returning an `f32` and
/// letting the caller call `vdupq_n_f32` would round-trip it out to a general
/// register and back once per block, which on a 32-element block is two domain
/// crossings per 32 multiply-accumulates.
#[inline]
#[target_feature(enable = "neon")]
fn f16_scale(bits: [u8; 2]) -> float32x4_t {
    let [b0, b1] = bits;
    vdupq_laneq_f32::<0>(widen_f16x4(&[b0, b1, 0, 0, 0, 0, 0, 0]))
}

/// The lane index vector `[0, 1, 2, 3]`, built without a memory load.
#[inline]
#[target_feature(enable = "neon")]
fn lane_index_s32() -> int32x4_t {
    let v = vdupq_n_s32(0);
    vsetq_lane_s32::<3>(3, vsetq_lane_s32::<2>(2, vsetq_lane_s32::<1>(1, v)))
}

/// Four Q5_0 codes, already biased: `(nibble | fifth) - 16`.
///
/// `nibbles` holds the low four bits of each code, one per lane. The fifth bit
/// of lane `l` is bit `base + l` of the block's `qh` word, which the scalar
/// kernel reaches as `(qh >> sj) << 4 & 0x10` for the low half and
/// `(qh >> (sj + 12)) & 0x10` for the high half: the same bit, the same
/// element. Lanes whose bit index is 32 or more cannot arise, since `base` is
/// at most 28 and `l` at most 3.
///
/// Testing the bit where it lies, rather than shifting it down into position 4
/// and OR-ing it in, folds it into the bias subtraction the kernel has to do
/// anyway: a Q5_0 code with the bit set is exactly `nibble`, and one with it
/// clear is `nibble - 16`.
#[inline]
#[target_feature(enable = "neon")]
fn q5_0_codes(nibbles: uint32x4_t, qh: uint32x4_t, base: i32) -> int32x4_t {
    let shifts = vaddq_s32(vdupq_n_s32(base), lane_index_s32());
    let clear = vceqq_u32(
        vandq_u32(qh, vshlq_u32(vdupq_n_u32(1), shifts)),
        vdupq_n_u32(0),
    );
    vaddq_s32(
        vreinterpretq_s32_u32(nibbles),
        vandq_s32(vreinterpretq_s32_u32(clear), vdupq_n_s32(-16)),
    )
}

/// Four Q5_1 codes: `nibble | fifth`, unbiased.
///
/// Q5_1 has no bias to fold, so the shift-and-OR form is a wash on instruction
/// count and measures faster on x86-64 than the masked form [`q5_0_codes`]
/// uses. `VSHL` with a negative count is a right shift, so the shift vector is
/// negated.
#[inline]
#[target_feature(enable = "neon")]
fn q5_1_codes(nibbles: uint32x4_t, qh: uint32x4_t, base: i32) -> uint32x4_t {
    let shifts = vsubq_s32(vdupq_n_s32(base.saturating_neg()), lane_index_s32());
    let fifth = vshlq_n_u32::<4>(vandq_u32(vshlq_u32(qh, shifts), vdupq_n_u32(1)));
    vorrq_u32(nibbles, fifth)
}

/// The two halves of a Q5 block's `qs` byte pack, as four `u32` lanes each.
///
/// Element `j` of the block takes the low nibble of `qs[j]` and element
/// `j + 16` the high nibble, exactly as the scalar kernel's `*p & 0x0f` and
/// `*p >> 4`. NEON widens eight bytes at a time and the caller steps in fours,
/// so the upper half is padded and discarded.
#[inline]
#[target_feature(enable = "neon")]
fn q5_nibbles(pack: &[u8; 4]) -> (uint32x4_t, uint32x4_t) {
    let &[p0, p1, p2, p3] = pack;
    let bytes = widen_u8x8(load_u8x8(&[p0, p1, p2, p3, 0, 0, 0, 0])).0;
    (vandq_u32(bytes, vdupq_n_u32(0x0f)), vshrq_n_u32::<4>(bytes))
}

/// Widen eight `i8` lanes to two vectors of four `i32`.
#[inline]
#[target_feature(enable = "neon")]
fn widen_s8x8(v: int8x8_t) -> (int32x4_t, int32x4_t) {
    let wide = vmovl_s8(v);
    (
        vmovl_s16(vget_low_s16(wide)),
        vmovl_s16(vget_high_s16(wide)),
    )
}

/// One accumulator for the whole row, reduced once at the end.
///
/// The `QK_K` kernels reduce per super-block, which costs one horizontal sum
/// per 256 elements. A Q5 or Q8_0 block is 32 elements, so the same shape
/// would be eight times as many reductions and the reduction, not the
/// arithmetic, would set the rate: reducing once per row instead measured
/// 1.28x on Q5_0, 1.40x on Q5_1 and 1.53x on Q8_0. The scalar kernel runs one
/// `sum` across every block too, so this is the same reassociation the bound
/// already covers, not a new one.
///
/// # Safety
///
/// The caller must run on a CPU with Advanced SIMD.
#[target_feature(enable = "neon")]
fn dot_q5_0_f32_neon(row: &[u8], x: &[f32]) -> f32 {
    let mut acc = vdupq_n_f32(0.0);
    let (w_blocks, _) = row.as_chunks::<Q5_0_BLOCK>();
    for (b, wb) in w_blocks.iter().enumerate() {
        // A Q5_0 block is a binary16 scale, a `u32` of fifth bits, then 16
        // bytes holding two nibbles each.
        let Some(dbits) = wb.first_chunk::<2>() else {
            continue;
        };
        let Some(qhb) = wb.get(2..).and_then(<[u8]>::first_chunk::<4>) else {
            continue;
        };
        let Some(qs) = wb.get(6..) else { continue };
        let x_base = b.saturating_mul(QK5_0);
        let Some(xr) = x.get(x_base..x_base.saturating_add(QK5_0)) else {
            continue;
        };
        let dv = f16_scale(*dbits);
        let qh = vdupq_n_u32(u32::from_le_bytes(*qhb));
        for (c, pack) in qs.as_chunks::<4>().0.iter().enumerate() {
            let j = c.saturating_mul(4);
            let (Ok(base), Some(xlo), Some(xhi)) = (
                i32::try_from(j),
                xr.get(j..).and_then(<[f32]>::first_chunk::<4>),
                xr.get(j.saturating_add(16)..)
                    .and_then(<[f32]>::first_chunk::<4>),
            ) else {
                continue;
            };
            let (low, high) = q5_nibbles(pack);
            // Codes are 0..=31 and the bias makes them -16..=15, so the scalar
            // kernel's `i32::from(..) - 16` is reproduced exactly. Multiplying
            // by `d` separately rather than folding it into the FMA keeps each
            // dequantized weight bit-identical; only the accumulation
            // reassociates.
            let q_lo = q5_0_codes(low, qh, base);
            let q_hi = q5_0_codes(high, qh, base.saturating_add(16));
            acc = vfmaq_f32(acc, vmulq_f32(vcvtq_f32_s32(q_lo), dv), load_f32x4(xlo));
            acc = vfmaq_f32(acc, vmulq_f32(vcvtq_f32_s32(q_hi), dv), load_f32x4(xhi));
        }
    }
    vaddvq_f32(acc)
}

/// Reduced once per row, as [`dot_q5_0_f32_neon`].
///
/// # Safety
///
/// The caller must run on a CPU with Advanced SIMD.
#[target_feature(enable = "neon")]
fn dot_q5_1_f32_neon(row: &[u8], x: &[f32]) -> f32 {
    let mut acc = vdupq_n_f32(0.0);
    let (w_blocks, _) = row.as_chunks::<Q5_1_BLOCK>();
    for (b, wb) in w_blocks.iter().enumerate() {
        // Q5_1 is Q5_0 with a second binary16, the offset `m`, inserted after
        // the scale; everything after it shifts along by two bytes.
        let Some(dbits) = wb.first_chunk::<2>() else {
            continue;
        };
        let Some(mbits) = wb.get(2..).and_then(<[u8]>::first_chunk::<2>) else {
            continue;
        };
        let Some(qhb) = wb.get(4..).and_then(<[u8]>::first_chunk::<4>) else {
            continue;
        };
        let Some(qs) = wb.get(8..) else { continue };
        let x_base = b.saturating_mul(QK5_1);
        let Some(xr) = x.get(x_base..x_base.saturating_add(QK5_1)) else {
            continue;
        };
        let dv = f16_scale(*dbits);
        let mv = f16_scale(*mbits);
        let qh = vdupq_n_u32(u32::from_le_bytes(*qhb));
        for (c, pack) in qs.as_chunks::<4>().0.iter().enumerate() {
            let j = c.saturating_mul(4);
            let (Ok(base), Some(xlo), Some(xhi)) = (
                i32::try_from(j),
                xr.get(j..).and_then(<[f32]>::first_chunk::<4>),
                xr.get(j.saturating_add(16)..)
                    .and_then(<[f32]>::first_chunk::<4>),
            ) else {
                continue;
            };
            let (low, high) = q5_nibbles(pack);
            // The scalar kernel writes `q * d + m`, a multiply and then an add,
            // so this must not contract to an FMA: a separate `vaddq_f32` keeps
            // both roundings and the dequantized weight bit-identical.
            let w_lo = vaddq_f32(vmulq_f32(vcvtq_f32_u32(q5_1_codes(low, qh, base)), dv), mv);
            let w_hi = vaddq_f32(
                vmulq_f32(
                    vcvtq_f32_u32(q5_1_codes(high, qh, base.saturating_add(16))),
                    dv,
                ),
                mv,
            );
            acc = vfmaq_f32(acc, w_lo, load_f32x4(xlo));
            acc = vfmaq_f32(acc, w_hi, load_f32x4(xhi));
        }
    }
    vaddvq_f32(acc)
}

/// Reduced once per row, as [`dot_q5_0_f32_neon`].
///
/// # Safety
///
/// The caller must run on a CPU with Advanced SIMD.
#[target_feature(enable = "neon")]
fn dot_q8_0_f32_neon(row: &[u8], x: &[f32]) -> f32 {
    let mut acc = vdupq_n_f32(0.0);
    let (w_blocks, _) = row.as_chunks::<Q8_0_BLOCK>();
    for (b, wb) in w_blocks.iter().enumerate() {
        let Some(dbits) = wb.first_chunk::<2>() else {
            continue;
        };
        let Some(qs) = wb.last_chunk::<32>() else {
            continue;
        };
        let x_base = b.saturating_mul(QK8_0);
        let Some(xr) = x.get(x_base..x_base.saturating_add(QK8_0)) else {
            continue;
        };
        let dv = f16_scale(*dbits);
        for (c, pack) in qs.as_chunks::<8>().0.iter().enumerate() {
            let off = c.saturating_mul(8);
            let (Some(xlo), Some(xhi)) = (
                xr.get(off..).and_then(<[f32]>::first_chunk::<4>),
                xr.get(off.saturating_add(4)..)
                    .and_then(<[f32]>::first_chunk::<4>),
            ) else {
                continue;
            };
            // `vmovl_s8` sign-extends, matching the scalar kernel's
            // `i8_from_bits`. As in Q5_0, `d` is applied by a separate multiply
            // so the dequantized weight is bit-identical.
            let (q_lo, q_hi) = widen_s8x8(vreinterpret_s8_u8(load_u8x8(pack)));
            acc = vfmaq_f32(acc, vmulq_f32(vcvtq_f32_s32(q_lo), dv), load_f32x4(xlo));
            acc = vfmaq_f32(acc, vmulq_f32(vcvtq_f32_s32(q_hi), dv), load_f32x4(xhi));
        }
    }
    vaddvq_f32(acc)
}

/// # Safety
///
/// The caller must run on a CPU with Advanced SIMD.
#[target_feature(enable = "neon")]
fn dot_q4_k_f32_neon(row: &[u8], x: &[f32]) -> f32 {
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
        let mut acc = vdupq_n_f32(0.0);
        let mask = vdup_n_u8(0x0f);
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
            let a0 = vdupq_n_f32(d * f32::from(sc0));
            let b0 = vdupq_n_f32(dmin * f32::from(m0));
            let a1 = vdupq_n_f32(d * f32::from(sc1));
            let b1 = vdupq_n_f32(dmin * f32::from(m1));
            let octets = packed
                .as_chunks::<8>()
                .0
                .iter()
                .zip(xlo.as_chunks::<8>().0.iter())
                .zip(xhi.as_chunks::<8>().0.iter());
            for ((pk, xl), xh) in octets {
                let (Some(xl0), Some(xl1), Some(xh0), Some(xh1)) = (
                    xl.first_chunk::<4>(),
                    xl.last_chunk::<4>(),
                    xh.first_chunk::<4>(),
                    xh.last_chunk::<4>(),
                ) else {
                    continue;
                };
                let nib = load_u8x8(pk);
                let (lo0, lo1) = widen_u8x8(vand_u8(nib, mask));
                let (hi0, hi1) = widen_u8x8(vand_u8(vshr_n_u8::<4>(nib), mask));
                // Separate multiply and subtract, not an FMA: each dequantized
                // weight is then bit-identical to the scalar kernel's, and only
                // the accumulation below reassociates.
                let w0 = vsubq_f32(vmulq_f32(a0, vcvtq_f32_u32(lo0)), b0);
                let w1 = vsubq_f32(vmulq_f32(a0, vcvtq_f32_u32(lo1)), b0);
                let w2 = vsubq_f32(vmulq_f32(a1, vcvtq_f32_u32(hi0)), b1);
                let w3 = vsubq_f32(vmulq_f32(a1, vcvtq_f32_u32(hi1)), b1);
                acc = vfmaq_f32(acc, w0, load_f32x4(xl0));
                acc = vfmaq_f32(acc, w1, load_f32x4(xl1));
                acc = vfmaq_f32(acc, w2, load_f32x4(xh0));
                acc = vfmaq_f32(acc, w3, load_f32x4(xh1));
            }
        }
        sum += vaddvq_f32(acc);
    }
    sum
}

/// # Safety
///
/// The caller must run on a CPU with Advanced SIMD.
#[target_feature(enable = "neon")]
fn dot_q6_k_f32_neon(row: &[u8], x: &[f32]) -> f32 {
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
        let mut acc = vdupq_n_f32(0.0);
        for group in 0..2usize {
            let ql_off = group * 64;
            let qh_off = group * 32;
            let sc_off = group * 8;
            let y_off = group * 128;
            // Eight elements per step, split into two four-lane halves.
            // `is = l / 16` is constant across a step because 8 divides 16.
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
                let ds0 = vdupq_n_f32(d * f32::from(i8_from_bits(*s0)));
                let ds2 = vdupq_n_f32(d * f32::from(i8_from_bits(*s2)));
                let ds4 = vdupq_n_f32(d * f32::from(i8_from_bits(*s4)));
                let ds6 = vdupq_n_f32(d * f32::from(i8_from_bits(*s6)));
                let (l0, l1) = widen_u8x8(load_u8x8(lo8));
                let (h0, h1) = widen_u8x8(load_u8x8(hi8));
                let (q0, q1) = widen_u8x8(load_u8x8(h8));
                // `q6_from_bits` in lanes: the 6-bit code is at most 63, so the
                // `- 32` recentring cannot wrap and the lanes hold the same
                // value the scalar kernel derives.
                let nib = vdupq_n_u32(0x0f);
                acc = q6_step(acc, vandq_u32(l0, nib), q0, 0, ds0, x1, false);
                acc = q6_step(acc, vandq_u32(l1, nib), q1, 0, ds0, x1, true);
                acc = q6_step(acc, vandq_u32(h0, nib), q0, 2, ds2, x2, false);
                acc = q6_step(acc, vandq_u32(h1, nib), q1, 2, ds2, x2, true);
                acc = q6_step(acc, vshrq_n_u32::<4>(l0), q0, 4, ds4, x3, false);
                acc = q6_step(acc, vshrq_n_u32::<4>(l1), q1, 4, ds4, x3, true);
                acc = q6_step(acc, vshrq_n_u32::<4>(h0), q0, 6, ds6, x4, false);
                acc = q6_step(acc, vshrq_n_u32::<4>(h1), q1, 6, ds6, x4, true);
            }
        }
        sum += vaddvq_f32(acc);
    }
    sum
}

/// One four-lane Q6_K accumulation: assemble `(low | ((qh >> shift) & 3) << 4)
/// minus 32`, scale by `ds`, and fuse-multiply-add against the low half of `xs`
/// when `upper` is false, the high half when it is true.
///
/// `shift` is 0, 2, 4 or 6; the match spells the four cases out because
/// `vshrq_n_u32` takes its count as a const generic.
#[inline]
#[target_feature(enable = "neon")]
fn q6_step(
    acc: float32x4_t,
    low: uint32x4_t,
    qh: uint32x4_t,
    shift: u32,
    ds: float32x4_t,
    xs: &[f32; 8],
    upper: bool,
) -> float32x4_t {
    let hi = match shift {
        0 => qh,
        2 => vshrq_n_u32::<2>(qh),
        4 => vshrq_n_u32::<4>(qh),
        _ => vshrq_n_u32::<6>(qh),
    };
    let bits = vorrq_u32(low, vshlq_n_u32::<4>(vandq_u32(hi, vdupq_n_u32(3))));
    let q = vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(bits), vdupq_n_s32(32)));
    let (Some(front), Some(back)) = (xs.first_chunk::<4>(), xs.last_chunk::<4>()) else {
        return acc;
    };
    let xv = load_f32x4(if upper { back } else { front });
    vfmaq_f32(acc, vmulq_f32(ds, q), xv)
}

/// # Safety
///
/// The caller must run on a CPU with Advanced SIMD.
#[target_feature(enable = "neon")]
fn dot_q8_0_neon(row: &[u8], x: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    let (w_blocks, _) = row.as_chunks::<Q8_0_BLOCK>();
    let (x_blocks, _) = x.as_chunks::<Q8_0_BLOCK>();
    for (wb, xb) in w_blocks.iter().zip(x_blocks.iter()) {
        // A Q8_0 block is a binary16 scale followed by exactly 32 `i8`.
        let (Some(dw), Some(dx)) = (wb.first_chunk::<2>(), xb.first_chunk::<2>()) else {
            continue;
        };
        let (Some(wqs), Some(xqs)) = (wb.last_chunk::<32>(), xb.last_chunk::<32>()) else {
            continue;
        };
        let (Some(wlo), Some(whi), Some(xlo), Some(xhi)) = (
            wqs.first_chunk::<16>(),
            wqs.last_chunk::<16>(),
            xqs.first_chunk::<16>(),
            xqs.last_chunk::<16>(),
        ) else {
            continue;
        };
        // Bit-identical to the scalar kernel. `SMULL` forms products of
        // magnitude at most 128 * 128 = 16_384 in `i16` lanes, `SADALP`
        // pairwise-adds them into `i32` lanes without saturating, and the 32
        // products of one block sum to at most 524_288 in magnitude. Nothing
        // overflows `i32`, and `i32` addition is associative, so neither the
        // pairing nor the lane order can change the total.
        let acc = vdupq_n_s32(0);
        let acc = mull_acc(acc, load_s8x16(wlo), load_s8x16(xlo));
        let acc = mull_acc(acc, load_s8x16(whi), load_s8x16(xhi));
        sum = sc::add_f32(sum, (vaddvq_s32(acc) as f32) * scale_product(*dw, *dx));
    }
    sum
}

/// Load 16 bytes as signed lanes.
#[inline]
#[target_feature(enable = "neon")]
fn load_s8x16(bytes: &[u8; 16]) -> int8x16_t {
    vreinterpretq_s8_u8(load_u8x16(bytes))
}

/// Exact `sum(a[i] * b[i])` over 16 `i8` pairs, accumulated into `acc`.
#[inline]
#[target_feature(enable = "neon")]
fn mull_acc(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let lo: int16x8_t = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    let hi: int16x8_t = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    vpadalq_s16(vpadalq_s16(acc, lo), hi)
}
