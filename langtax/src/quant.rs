//! GGUF on-disk F16 / Q4_0 / Q8_0 / Q4_K / Q5_K / Q6_K / Q8_K / IQ4_XS blocks. GEMV reads those bytes; no f32-scale copy.

use std::fmt;

use crate::fp16::{load_f16_le, store_f16_le};
use crate::pool::{for_each_group, for_each_row};

/// Q8_0 / Q4_0 block width in elements (`ggml` `QK8_0` / `QK4_0`).
pub const QK8_0: usize = 32;
/// Q4_0 block width in elements. Same 32-wide super-block as Q8_0.
pub const QK4_0: usize = 32;
/// ggml `block_q8_0`: `ggml_half d` + `int8 qs[32]`.
pub const Q8_0_BLOCK: usize = 2 + QK8_0;
/// ggml `block_q4_0`: `ggml_half d` + `uint8 qs[16]`.
pub const Q4_0_BLOCK: usize = 2 + QK4_0 / 2;
/// ggml super-block width (`QK_K`).
pub const QK_K: usize = 256;
/// ggml `K_SCALE_SIZE`: packed 6-bit scales/mins for 8 Q4_K sub-blocks.
pub(crate) const K_SCALE_SIZE: usize = 12;
/// ggml `block_q4_K`: `d`/`dmin` binary16, 12 scale bytes, 128 nibble bytes.
pub const Q4_K_BLOCK: usize = 4 + K_SCALE_SIZE + QK_K / 2;
/// ggml `block_q5_K`: `d`/`dmin` binary16, 12 scale bytes, 32 `qh` bytes, 128 nibble bytes.
pub const Q5_K_BLOCK: usize = 4 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2;
/// ggml `block_q8_K`: f32 `d`, `int8 qs[256]`, `int16 bsums[16]`.
pub const Q8_K_BLOCK: usize = 4 + QK_K + QK_K / 16 * 2;
/// ggml `block_q6_K`: `ql[128]` + `qh[64]` + `scales[16]` + binary16 `d`.
pub const Q6_K_BLOCK: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
/// ggml `block_iq4_xs`: binary16 `d`, `uint16 scales_h`, `scales_l[4]`, `qs[128]`.
pub const IQ4_XS_BLOCK: usize = 2 + 2 + QK_K / 64 + QK_K / 2;
/// ggml `kvalues_iq4nl` (shared by IQ4_NL and IQ4_XS).
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];
/// ggml `GGML_TYPE_F32` element size.
pub const F32_SIZE: usize = 4;
/// ggml `GGML_TYPE_F16` element size (`ggml_fp16_t` / IEEE binary16).
pub const F16_SIZE: usize = 2;

/// Size / alignment failure for a quantized GEMV.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantError {
    /// `n_cols` is not a multiple of the block width.
    UnalignedCols {
        /// Requested column count.
        n_cols: usize,
        /// Required block multiple.
        block: usize,
    },
    /// Packed byte length did not match rows × row-stride.
    Size {
        /// Which buffer failed the check.
        what: &'static str,
        /// Byte length required by `n_cols` and `y.len()`.
        expected: usize,
        /// Byte length actually supplied.
        actual: usize,
    },
}

impl fmt::Display for QuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnalignedCols { n_cols, block } => {
                write!(f, "n_cols {n_cols} is not a multiple of {block}")
            }
            Self::Size {
                what,
                expected,
                actual,
            } => write!(f, "{what} size {actual} != {expected}"),
        }
    }
}

impl std::error::Error for QuantError {}

fn row_bytes(w: &[u8], rb: usize, r: usize) -> Option<&[u8]> {
    let start = r.checked_mul(rb)?;
    w.get(start..)?.get(..rb)
}

/// Bit-cast `u8` to `i8` without a wrapping `as`.
pub(crate) fn i8_from_bits(b: u8) -> i8 {
    i8::from_le_bytes([b])
}

/// Bit-cast `i8` to `u8` without a sign-loss `as`.
fn u8_from_bits(v: i8) -> u8 {
    u8::from_le_bytes(v.to_le_bytes())
}

/// Packed Q8_0 bytes for one matrix row of `n_cols` columns.
pub fn q8_0_row_bytes(n_cols: usize) -> Result<usize, QuantError> {
    if !n_cols.is_multiple_of(QK8_0) {
        return Err(QuantError::UnalignedCols {
            n_cols,
            block: QK8_0,
        });
    }
    Ok((n_cols / QK8_0) * Q8_0_BLOCK)
}

/// Packed Q4_0 bytes for one matrix row of `n_cols` columns.
pub fn q4_0_row_bytes(n_cols: usize) -> Result<usize, QuantError> {
    if !n_cols.is_multiple_of(QK4_0) {
        return Err(QuantError::UnalignedCols {
            n_cols,
            block: QK4_0,
        });
    }
    Ok((n_cols / QK4_0) * Q4_0_BLOCK)
}

/// Pack one Q8_0 block: binary16 scale + 32 signed `qs` bytes.
pub fn pack_q8_0_block(scale: f32, qs: &[i8; QK8_0]) -> [u8; Q8_0_BLOCK] {
    let mut out = [0u8; Q8_0_BLOCK];
    let d = store_f16_le(scale);
    out[0] = d[0];
    out[1] = d[1];
    for (dst, src) in out.iter_mut().skip(2).zip(qs.iter()) {
        *dst = u8_from_bits(*src);
    }
    out
}

/// Pack one Q4_0 block: binary16 scale + 16 already-nibbled `qs` bytes.
pub fn pack_q4_0_block(scale: f32, qs: &[u8; QK4_0 / 2]) -> [u8; Q4_0_BLOCK] {
    let mut out = [0u8; Q4_0_BLOCK];
    let d = store_f16_le(scale);
    out[0] = d[0];
    out[1] = d[1];
    for (dst, src) in out.iter_mut().skip(2).zip(qs.iter()) {
        *dst = *src;
    }
    out
}

/// Pack 32 signed 4-bit values in GGUF/ggml order: `v[j]` in low nibble of
/// `qs[j]`, `v[j+16]` in high nibble (`dequantize_row_q4_0`).
pub fn pack_q4_0_from_i4(scale: f32, v: &[i8; QK4_0]) -> [u8; Q4_0_BLOCK] {
    let mut qs = [0u8; QK4_0 / 2];
    let Some((lo_src, hi_src)) = v.split_at_checked(QK4_0 / 2) else {
        return pack_q4_0_block(scale, &qs);
    };
    for ((slot, lo), hi) in qs.iter_mut().zip(lo_src.iter()).zip(hi_src.iter()) {
        let lo_n = u8::try_from(i32::from(*lo) + 8).unwrap_or(0) & 0x0f;
        let hi_n = u8::try_from(i32::from(*hi) + 8).unwrap_or(0) & 0x0f;
        *slot = lo_n | (hi_n << 4);
    }
    pack_q4_0_block(scale, &qs)
}

fn load_f32_le(bytes: &[u8]) -> Option<f32> {
    let b0 = *bytes.first()?;
    let b1 = *bytes.get(1)?;
    let b2 = *bytes.get(2)?;
    let b3 = *bytes.get(3)?;
    Some(f32::from_bits(u32::from_le_bytes([b0, b1, b2, b3])))
}

fn store_f32_le(v: f32) -> [u8; 4] {
    v.to_bits().to_le_bytes()
}

fn pack_q4_k_scale_bytes(ls: &[u8; 8], lm: &[u8; 8]) -> [u8; K_SCALE_SIZE] {
    let mut s = [0u8; K_SCALE_SIZE];
    for j in 0..4 {
        let lsj = ls.get(j).copied().unwrap_or(0) & 63;
        let lmj = lm.get(j).copied().unwrap_or(0) & 63;
        if let Some(slot) = s.get_mut(j) {
            *slot = lsj;
        }
        if let Some(slot) = s.get_mut(j + 4) {
            *slot = lmj;
        }
    }
    for j in 4..8 {
        let lsj = ls.get(j).copied().unwrap_or(0) & 63;
        let lmj = lm.get(j).copied().unwrap_or(0) & 63;
        if let Some(slot) = s.get_mut(j + 4) {
            *slot = (lsj & 0x0f) | ((lmj & 0x0f) << 4);
        }
        if let Some(slot) = s.get_mut(j.saturating_sub(4)) {
            *slot |= (lsj >> 4) << 6;
        }
        if let Some(slot) = s.get_mut(j) {
            *slot |= (lmj >> 4) << 6;
        }
    }
    s
}

/// Pack one Q4_K block: binary16 `d`/`dmin`, 6-bit scales/mins, 256 4-bit `qs`.
///
/// `qs_i4[i]` is 0..=15 in element order (`dequantize_row_q4_K`: for each 64-wide
/// group, lo nibble is `y[j + l]`, hi nibble is `y[j + 32 + l]`).
pub fn pack_q4_k_block(
    d: f32,
    dmin: f32,
    scales: &[u8; 8],
    mins: &[u8; 8],
    qs_i4: &[u8; QK_K],
) -> [u8; Q4_K_BLOCK] {
    let mut out = [0u8; Q4_K_BLOCK];
    let db = store_f16_le(d);
    let dmb = store_f16_le(dmin);
    out[0] = db[0];
    out[1] = db[1];
    out[2] = dmb[0];
    out[3] = dmb[1];
    let sc = pack_q4_k_scale_bytes(scales, mins);
    for (dst, src) in out.iter_mut().skip(4).zip(sc.iter()) {
        *dst = *src;
    }
    for group in 0..4 {
        let base = group * 64;
        let Some(lo) = qs_i4.get(base..base + 32) else {
            continue;
        };
        let Some(hi) = qs_i4.get(base + 32..base + 64) else {
            continue;
        };
        let qs_off = 16 + group * 32;
        let Some(dst) = out.get_mut(qs_off..qs_off + 32) else {
            continue;
        };
        for ((slot, l), h) in dst.iter_mut().zip(lo.iter()).zip(hi.iter()) {
            *slot = (l & 0x0f) | ((h & 0x0f) << 4);
        }
    }
    out
}

/// Pack one Q5_K block: binary16 `d`/`dmin`, 6-bit scales/mins, 256 5-bit `qs`.
///
/// `qs_i5[i]` is 0..=31 in element order (`dequantize_row_q5_K`: for each 64-wide
/// group `g`, lo nibble + `qh[l]` bit `2g` is `y[j + l]`, hi nibble + bit `2g+1`
/// is `y[j + 32 + l]`).
pub fn pack_q5_k_block(
    d: f32,
    dmin: f32,
    scales: &[u8; 8],
    mins: &[u8; 8],
    qs_i5: &[u8; QK_K],
) -> [u8; Q5_K_BLOCK] {
    let mut out = [0u8; Q5_K_BLOCK];
    let db = store_f16_le(d);
    let dmb = store_f16_le(dmin);
    out[0] = db[0];
    out[1] = db[1];
    out[2] = dmb[0];
    out[3] = dmb[1];
    let sc = pack_q4_k_scale_bytes(scales, mins);
    for (dst, src) in out.iter_mut().skip(4).zip(sc.iter()) {
        *dst = *src;
    }
    let mut qh = [0u8; QK_K / 8];
    for group in 0..4 {
        let base = group * 64;
        let Some(lo) = qs_i5.get(base..base + 32) else {
            continue;
        };
        let Some(hi) = qs_i5.get(base + 32..base + 64) else {
            continue;
        };
        let qs_off = 48 + group * 32;
        let Some(dst) = out.get_mut(qs_off..qs_off + 32) else {
            continue;
        };
        let shift = u32::try_from(group.saturating_mul(2)).unwrap_or(0);
        let m1 = 1u8.wrapping_shl(shift);
        let m2 = 2u8.wrapping_shl(shift);
        for (l, ((slot, lv), hv)) in dst.iter_mut().zip(lo.iter()).zip(hi.iter()).enumerate() {
            let l5 = *lv & 31;
            let h5 = *hv & 31;
            *slot = (l5 & 0x0f) | ((h5 & 0x0f) << 4);
            if l5 > 15 {
                if let Some(bit) = qh.get_mut(l) {
                    *bit |= m1;
                }
            }
            if h5 > 15 {
                if let Some(bit) = qh.get_mut(l) {
                    *bit |= m2;
                }
            }
        }
    }
    for (dst, src) in out.iter_mut().skip(16).zip(qh.iter()) {
        *dst = *src;
    }
    out
}

/// Pack one IQ4_XS block: binary16 `d`, 6-bit scales (`scales_h` + `scales_l`), 256 nibbles.
///
/// `qs_i4[i]` is 0..=15 in element order (`dequantize_row_iq4_xs`: sub-block `ib` of 32,
/// lo nibble is `y[ib*32 + j]`, hi nibble is `y[ib*32 + 16 + j]`, `j < 16`).
pub fn pack_iq4_xs_block(d: f32, scales: &[u8; 8], qs_i4: &[u8; QK_K]) -> [u8; IQ4_XS_BLOCK] {
    let mut out = [0u8; IQ4_XS_BLOCK];
    let db = store_f16_le(d);
    out[0] = db[0];
    out[1] = db[1];
    let mut scales_h = 0u16;
    for ib in 0usize..8 {
        let ls = scales.get(ib).copied().unwrap_or(0) & 63;
        let nibble_shift = if ib.is_multiple_of(2) { 0u32 } else { 4u32 };
        if let Some(slot) = out.get_mut(4 + ib / 2) {
            *slot |= (ls & 0x0f).wrapping_shl(nibble_shift);
        }
        let hi_shift = u32::try_from(ib.saturating_mul(2)).unwrap_or(0);
        scales_h |= u16::from((ls >> 4) & 3).wrapping_shl(hi_shift);
    }
    let hb = scales_h.to_le_bytes();
    out[2] = hb[0];
    out[3] = hb[1];
    for ib in 0usize..8 {
        let base = ib.saturating_mul(32);
        let Some(lo) = qs_i4.get(base..base.saturating_add(16)) else {
            continue;
        };
        let Some(hi) = qs_i4.get(base.saturating_add(16)..base.saturating_add(32)) else {
            continue;
        };
        let qs_off = 8 + ib.saturating_mul(16);
        let Some(dst) = out.get_mut(qs_off..qs_off.saturating_add(16)) else {
            continue;
        };
        for ((slot, l), h) in dst.iter_mut().zip(lo.iter()).zip(hi.iter()) {
            *slot = (l & 0x0f) | ((h & 0x0f) << 4);
        }
    }
    out
}

/// Pack one Q8_K block: f32 `d`, 256 signed `qs`, and ggml `bsums` of 16.
pub fn pack_q8_k_block(d: f32, qs: &[i8; QK_K]) -> [u8; Q8_K_BLOCK] {
    let mut out = [0u8; Q8_K_BLOCK];
    let db = store_f32_le(d);
    for (dst, src) in out.iter_mut().zip(db.iter()) {
        *dst = *src;
    }
    for (dst, src) in out.iter_mut().skip(4).zip(qs.iter()) {
        *dst = u8_from_bits(*src);
    }
    for g in 0..16 {
        let start = g * 16;
        let Some(group) = qs.get(start..start + 16) else {
            continue;
        };
        let mut sum = 0i32;
        for q in group {
            sum += i32::from(*q);
        }
        let b = i16::try_from(sum).unwrap_or(0).to_le_bytes();
        let off = 4 + QK_K + g * 2;
        if let Some(slot) = out.get_mut(off) {
            *slot = b[0];
        }
        if let Some(slot) = out.get_mut(off + 1) {
            *slot = b[1];
        }
    }
    out
}

/// Packed Q4_K bytes for one matrix row of `n_cols` columns.
pub fn q4_k_row_bytes(n_cols: usize) -> Result<usize, QuantError> {
    if !n_cols.is_multiple_of(QK_K) {
        return Err(QuantError::UnalignedCols {
            n_cols,
            block: QK_K,
        });
    }
    Ok((n_cols / QK_K) * Q4_K_BLOCK)
}

/// Packed Q5_K bytes for one matrix row of `n_cols` columns.
pub fn q5_k_row_bytes(n_cols: usize) -> Result<usize, QuantError> {
    if !n_cols.is_multiple_of(QK_K) {
        return Err(QuantError::UnalignedCols {
            n_cols,
            block: QK_K,
        });
    }
    Ok((n_cols / QK_K) * Q5_K_BLOCK)
}

/// Packed IQ4_XS bytes for one matrix row of `n_cols` columns.
pub fn iq4_xs_row_bytes(n_cols: usize) -> Result<usize, QuantError> {
    if !n_cols.is_multiple_of(QK_K) {
        return Err(QuantError::UnalignedCols {
            n_cols,
            block: QK_K,
        });
    }
    Ok((n_cols / QK_K) * IQ4_XS_BLOCK)
}

/// Packed Q8_K bytes for one matrix row of `n_cols` columns.
pub fn q8_k_row_bytes(n_cols: usize) -> Result<usize, QuantError> {
    if !n_cols.is_multiple_of(QK_K) {
        return Err(QuantError::UnalignedCols {
            n_cols,
            block: QK_K,
        });
    }
    Ok((n_cols / QK_K) * Q8_K_BLOCK)
}

/// Packed F32 bytes for one matrix row of `n_cols` columns.
pub fn f32_row_bytes(n_cols: usize) -> Result<usize, QuantError> {
    n_cols.checked_mul(F32_SIZE).ok_or(QuantError::Size {
        what: "F32 row overflow",
        expected: n_cols,
        actual: F32_SIZE,
    })
}

/// Packed F16 bytes for one matrix row of `n_cols` columns.
pub fn f16_row_bytes(n_cols: usize) -> Result<usize, QuantError> {
    n_cols.checked_mul(F16_SIZE).ok_or(QuantError::Size {
        what: "F16 row overflow",
        expected: n_cols,
        actual: F16_SIZE,
    })
}

/// Packed Q6_K bytes for one matrix row of `n_cols` columns.
pub fn q6_k_row_bytes(n_cols: usize) -> Result<usize, QuantError> {
    if !n_cols.is_multiple_of(QK_K) {
        return Err(QuantError::UnalignedCols {
            n_cols,
            block: QK_K,
        });
    }
    Ok((n_cols / QK_K) * Q6_K_BLOCK)
}

/// Pack F32 values as little-endian on-disk bytes.
pub fn pack_f32(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len().saturating_mul(F32_SIZE));
    for v in values {
        out.extend_from_slice(&store_f32_le(*v));
    }
    out
}

/// Pack values as little-endian IEEE binary16 GGUF bytes (`GGML_TYPE_F16`).
pub fn pack_f16(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len().saturating_mul(F16_SIZE));
    for v in values {
        out.extend_from_slice(&store_f16_le(*v));
    }
    out
}

/// Pack one Q6_K block. `qs` is ggml `q` in `-32..=31` (`x = d * scale * q`).
pub fn pack_q6_k_block(d: f32, scales: &[i8; 16], qs: &[i8; QK_K]) -> [u8; Q6_K_BLOCK] {
    let mut out = [0u8; Q6_K_BLOCK];
    for group in 0..2 {
        let j = group * 128;
        let ql_off = group * 64;
        let qh_off = 128 + group * 32;
        for l in 0..32 {
            let q1 = six_bit(qs.get(j + l).copied().unwrap_or(0));
            let q2 = six_bit(qs.get(j + l + 32).copied().unwrap_or(0));
            let q3 = six_bit(qs.get(j + l + 64).copied().unwrap_or(0));
            let q4 = six_bit(qs.get(j + l + 96).copied().unwrap_or(0));
            if let Some(slot) = out.get_mut(ql_off + l) {
                *slot = (q1 & 0x0f) | ((q3 & 0x0f) << 4);
            }
            if let Some(slot) = out.get_mut(ql_off + l + 32) {
                *slot = (q2 & 0x0f) | ((q4 & 0x0f) << 4);
            }
            if let Some(slot) = out.get_mut(qh_off + l) {
                *slot = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
            }
        }
    }
    for (dst, src) in out.iter_mut().skip(192).zip(scales.iter()) {
        *dst = u8_from_bits(*src);
    }
    let db = store_f16_le(d);
    if let Some(slot) = out.get_mut(208) {
        *slot = db[0];
    }
    if let Some(slot) = out.get_mut(209) {
        *slot = db[1];
    }
    out
}

fn six_bit(q: i8) -> u8 {
    u8::try_from(i32::from(q) + 32).unwrap_or(0) & 63
}

fn q6_from_bits(bits: u8) -> i8 {
    i8::try_from(i32::from(bits & 63) - 32).unwrap_or(0)
}

/// ggml Q5_K 5-bit quant: low nibble plus optional high bit 16.
fn q5_from_nibble(nibble: u8, qh: u8, mask: u8) -> u8 {
    (nibble & 0x0f).saturating_add(if qh & mask == 0 { 0 } else { 16 })
}

/// ggml `kvalues_iq4nl[nibble]`.
fn kvalue_iq4nl(nibble: u8) -> i8 {
    KVALUES_IQ4NL
        .get(usize::from(nibble & 0x0f))
        .copied()
        .unwrap_or(0)
}

/// ggml IQ4_XS 6-bit scale `ls` for sub-block `ib` (`0..8`).
fn iq4_xs_ls(scales_l: &[u8], scales_h: u16, ib: usize) -> u8 {
    let lo_shift = if ib.is_multiple_of(2) { 0u32 } else { 4u32 };
    let lo = scales_l
        .get(ib / 2)
        .copied()
        .unwrap_or(0)
        .wrapping_shr(lo_shift)
        & 0x0f;
    let hi_shift = u32::try_from(ib.saturating_mul(2)).unwrap_or(0);
    let hi = u8::try_from((u32::from(scales_h) >> hi_shift) & 3).unwrap_or(0);
    lo | (hi << 4)
}

fn require_len(what: &'static str, actual: usize, expected: usize) -> Result<(), QuantError> {
    if actual == expected {
        Ok(())
    } else {
        Err(QuantError::Size {
            what,
            expected,
            actual,
        })
    }
}

/// `y[m] = W[m, n_cols] x[n_cols]`, W and x as GGUF Q8_0 block streams.
pub fn gemv_q8_0(n_cols: usize, w: &[u8], x: &[u8], y: &mut [f32]) -> Result<(), QuantError> {
    let rb = q8_0_row_bytes(n_cols)?;
    require_len("x Q8_0 bytes", x.len(), rb)?;
    let expected_w = rb.checked_mul(y.len()).ok_or(QuantError::Size {
        what: "W Q8_0 bytes overflow",
        expected: rb,
        actual: y.len(),
    })?;
    require_len("W Q8_0 bytes", w.len(), expected_w)?;
    if y.is_empty() {
        return Ok(());
    }
    for_each_row(y, |r, out| {
        *out = row_bytes(w, rb, r)
            .map(|row| vec_dot_q8_row(row, x))
            .unwrap_or(0.0);
    });
    Ok(())
}

/// `y[m] = W_q4[m, n_cols] x_q8[n_cols]`.
pub fn gemv_q4_0(n_cols: usize, w: &[u8], x: &[u8], y: &mut [f32]) -> Result<(), QuantError> {
    let w_rb = q4_0_row_bytes(n_cols)?;
    let x_rb = q8_0_row_bytes(n_cols)?;
    require_len("x Q8_0 bytes", x.len(), x_rb)?;
    let expected_w = w_rb.checked_mul(y.len()).ok_or(QuantError::Size {
        what: "W Q4_0 bytes overflow",
        expected: w_rb,
        actual: y.len(),
    })?;
    require_len("W Q4_0 bytes", w.len(), expected_w)?;
    if y.is_empty() {
        return Ok(());
    }
    for_each_row(y, |r, out| {
        *out = row_bytes(w, w_rb, r)
            .map(|row| vec_dot_q4_row(row, x))
            .unwrap_or(0.0);
    });
    Ok(())
}

/// `y[m] = W_q4k[m, n_cols] x_q8k[n_cols]`. Scales stay in the GGUF block bytes.
pub fn gemv_q4_k(n_cols: usize, w: &[u8], x: &[u8], y: &mut [f32]) -> Result<(), QuantError> {
    let w_rb = q4_k_row_bytes(n_cols)?;
    let x_rb = q8_k_row_bytes(n_cols)?;
    require_len("x Q8_K bytes", x.len(), x_rb)?;
    let expected_w = w_rb.checked_mul(y.len()).ok_or(QuantError::Size {
        what: "W Q4_K bytes overflow",
        expected: w_rb,
        actual: y.len(),
    })?;
    require_len("W Q4_K bytes", w.len(), expected_w)?;
    if y.is_empty() {
        return Ok(());
    }
    for_each_row(y, |r, out| {
        *out = row_bytes(w, w_rb, r)
            .map(|row| vec_dot_q4_k_row(row, x))
            .unwrap_or(0.0);
    });
    Ok(())
}

/// `y[m] = W_f16[m, n_cols] x[n_cols]`. On-disk bytes stay IEEE binary16.
pub fn gemv_f16(n_cols: usize, w: &[u8], x: &[f32], y: &mut [f32]) -> Result<(), QuantError> {
    let rb = f16_row_bytes(n_cols)?;
    require_len("x F32 elems", x.len(), n_cols)?;
    let expected_w = rb.checked_mul(y.len()).ok_or(QuantError::Size {
        what: "W F16 bytes overflow",
        expected: rb,
        actual: y.len(),
    })?;
    require_len("W F16 bytes", w.len(), expected_w)?;
    if y.is_empty() {
        return Ok(());
    }
    for_each_row(y, |r, out| {
        *out = row_bytes(w, rb, r)
            .map(|row| vec_dot_f16_row(row, x))
            .unwrap_or(0.0);
    });
    Ok(())
}

/// `y[m] = W_f32[m, n_cols] x[n_cols]`.
pub fn gemv_f32(n_cols: usize, w: &[u8], x: &[f32], y: &mut [f32]) -> Result<(), QuantError> {
    let rb = f32_row_bytes(n_cols)?;
    require_len("x F32 elems", x.len(), n_cols)?;
    let expected_w = rb.checked_mul(y.len()).ok_or(QuantError::Size {
        what: "W F32 bytes overflow",
        expected: rb,
        actual: y.len(),
    })?;
    require_len("W F32 bytes", w.len(), expected_w)?;
    if y.is_empty() {
        return Ok(());
    }
    for_each_row(y, |r, out| {
        *out = row_bytes(w, rb, r)
            .map(|row| vec_dot_f32_row(row, x))
            .unwrap_or(0.0);
    });
    Ok(())
}

/// `y[m] = W_q4k[m, n_cols] x_f32[n_cols]`.
pub fn gemv_q4_k_f32(n_cols: usize, w: &[u8], x: &[f32], y: &mut [f32]) -> Result<(), QuantError> {
    let w_rb = q4_k_row_bytes(n_cols)?;
    require_len("x F32 elems", x.len(), n_cols)?;
    let expected_w = w_rb.checked_mul(y.len()).ok_or(QuantError::Size {
        what: "W Q4_K bytes overflow",
        expected: w_rb,
        actual: y.len(),
    })?;
    require_len("W Q4_K bytes", w.len(), expected_w)?;
    if y.is_empty() {
        return Ok(());
    }
    for_each_row(y, |r, out| {
        *out = row_bytes(w, w_rb, r)
            .map(|row| vec_dot_q4_k_f32_row(row, x))
            .unwrap_or(0.0);
    });
    Ok(())
}

/// `Y[t, r] = W_f16[r, n_cols] · X[t, n_cols]`. Token-major `x` / `y`.
pub fn gemm_f16(
    n_cols: usize,
    n_tokens: usize,
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
) -> Result<(), QuantError> {
    if n_tokens == 1 {
        return gemv_f16(n_cols, w, x, y);
    }
    gemm_f32_x(GemmKind::F16, n_cols, n_tokens, w, x, y)
}

/// `Y[t, r] = W_f32[r, n_cols] · X[t, n_cols]`.
///
/// `x` is `n_tokens * n_cols` (token-major). `y` is `n_tokens * n_rows`
/// (token-major). Each weight row is read once and dotted with every token.
pub fn gemm_f32(
    n_cols: usize,
    n_tokens: usize,
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
) -> Result<(), QuantError> {
    if n_tokens == 1 {
        return gemv_f32(n_cols, w, x, y);
    }
    gemm_f32_x(GemmKind::F32, n_cols, n_tokens, w, x, y)
}

/// `Y[t, r] = W_q4k[r, n_cols] · X[t, n_cols]`. Token-major `x` / `y`.
pub fn gemm_q4_k_f32(
    n_cols: usize,
    n_tokens: usize,
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
) -> Result<(), QuantError> {
    if n_tokens == 1 {
        return gemv_q4_k_f32(n_cols, w, x, y);
    }
    gemm_f32_x(GemmKind::Q4K, n_cols, n_tokens, w, x, y)
}

/// `Y[t, r] = W_q6k[r, n_cols] · X[t, n_cols]`. Token-major `x` / `y`.
pub fn gemm_q6_k_f32(
    n_cols: usize,
    n_tokens: usize,
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
) -> Result<(), QuantError> {
    if n_tokens == 1 {
        return gemv_q6_k_f32(n_cols, w, x, y);
    }
    gemm_f32_x(GemmKind::Q6K, n_cols, n_tokens, w, x, y)
}

/// `Y[t, r] = W_q5k[r, n_cols] · X[t, n_cols]`. Token-major `x` / `y`.
pub fn gemm_q5_k_f32(
    n_cols: usize,
    n_tokens: usize,
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
) -> Result<(), QuantError> {
    if n_tokens == 1 {
        return gemv_q5_k_f32(n_cols, w, x, y);
    }
    gemm_f32_x(GemmKind::Q5K, n_cols, n_tokens, w, x, y)
}

/// `Y[t, r] = W_iq4xs[r, n_cols] · X[t, n_cols]`. Token-major `x` / `y`.
pub fn gemm_iq4_xs_f32(
    n_cols: usize,
    n_tokens: usize,
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
) -> Result<(), QuantError> {
    if n_tokens == 1 {
        return gemv_iq4_xs_f32(n_cols, w, x, y);
    }
    gemm_f32_x(GemmKind::IQ4XS, n_cols, n_tokens, w, x, y)
}

#[derive(Clone, Copy)]
enum GemmKind {
    F16,
    F32,
    Q4K,
    Q5K,
    Q6K,
    IQ4XS,
}

fn gemm_f32_x(
    kind: GemmKind,
    n_cols: usize,
    n_tokens: usize,
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
) -> Result<(), QuantError> {
    if n_tokens == 0 {
        return Err(QuantError::Size {
            what: "n_tokens",
            expected: 1,
            actual: 0,
        });
    }
    let n_x = n_tokens.checked_mul(n_cols).ok_or(QuantError::Size {
        what: "X F32 elems overflow",
        expected: n_cols,
        actual: n_tokens,
    })?;
    require_len("X F32 elems", x.len(), n_x)?;
    if !y.len().is_multiple_of(n_tokens) {
        return Err(QuantError::Size {
            what: "Y F32 elems",
            expected: n_tokens,
            actual: y.len(),
        });
    }
    let n_rows = y.len() / n_tokens;
    let (rb, w_what) = match kind {
        GemmKind::F16 => (f16_row_bytes(n_cols)?, "W F16 bytes"),
        GemmKind::F32 => (f32_row_bytes(n_cols)?, "W F32 bytes"),
        GemmKind::Q4K => (q4_k_row_bytes(n_cols)?, "W Q4_K bytes"),
        GemmKind::Q5K => (q5_k_row_bytes(n_cols)?, "W Q5_K bytes"),
        GemmKind::Q6K => (q6_k_row_bytes(n_cols)?, "W Q6_K bytes"),
        GemmKind::IQ4XS => (iq4_xs_row_bytes(n_cols)?, "W IQ4_XS bytes"),
    };
    let expected_w = rb.checked_mul(n_rows).ok_or(QuantError::Size {
        what: "W bytes overflow",
        expected: rb,
        actual: n_rows,
    })?;
    require_len(w_what, w.len(), expected_w)?;
    if n_rows == 0 {
        return Ok(());
    }
    let mut scratch = vec![0.0f32; y.len()];
    for_each_group(&mut scratch, n_tokens, |r, out| {
        let Some(wrow) = row_bytes(w, rb, r) else {
            return;
        };
        for (t, slot) in out.iter_mut().enumerate() {
            let start = t.saturating_mul(n_cols);
            let Some(xt) = x.get(start..start.saturating_add(n_cols)) else {
                continue;
            };
            *slot = match kind {
                GemmKind::F16 => vec_dot_f16_row(wrow, xt),
                GemmKind::F32 => vec_dot_f32_row(wrow, xt),
                GemmKind::Q4K => vec_dot_q4_k_f32_row(wrow, xt),
                GemmKind::Q5K => vec_dot_q5_k_f32_row(wrow, xt),
                GemmKind::Q6K => vec_dot_q6_k_f32_row(wrow, xt),
                GemmKind::IQ4XS => vec_dot_iq4_xs_f32_row(wrow, xt),
            };
        }
    });
    for t in 0..n_tokens {
        for r in 0..n_rows {
            let src = r.saturating_mul(n_tokens).saturating_add(t);
            let dst = t.saturating_mul(n_rows).saturating_add(r);
            if let (Some(&sv), Some(dv)) = (scratch.get(src), y.get_mut(dst)) {
                *dv = sv;
            }
        }
    }
    Ok(())
}

/// `y[m] = W_q6k[m, n_cols] x_f32[n_cols]`.
pub fn gemv_q6_k_f32(n_cols: usize, w: &[u8], x: &[f32], y: &mut [f32]) -> Result<(), QuantError> {
    let w_rb = q6_k_row_bytes(n_cols)?;
    require_len("x F32 elems", x.len(), n_cols)?;
    let expected_w = w_rb.checked_mul(y.len()).ok_or(QuantError::Size {
        what: "W Q6_K bytes overflow",
        expected: w_rb,
        actual: y.len(),
    })?;
    require_len("W Q6_K bytes", w.len(), expected_w)?;
    if y.is_empty() {
        return Ok(());
    }
    for_each_row(y, |r, out| {
        *out = row_bytes(w, w_rb, r)
            .map(|row| vec_dot_q6_k_f32_row(row, x))
            .unwrap_or(0.0);
    });
    Ok(())
}

/// `y[m] = W_q5k[m, n_cols] x_f32[n_cols]`.
pub fn gemv_q5_k_f32(n_cols: usize, w: &[u8], x: &[f32], y: &mut [f32]) -> Result<(), QuantError> {
    let w_rb = q5_k_row_bytes(n_cols)?;
    require_len("x F32 elems", x.len(), n_cols)?;
    let expected_w = w_rb.checked_mul(y.len()).ok_or(QuantError::Size {
        what: "W Q5_K bytes overflow",
        expected: w_rb,
        actual: y.len(),
    })?;
    require_len("W Q5_K bytes", w.len(), expected_w)?;
    if y.is_empty() {
        return Ok(());
    }
    for_each_row(y, |r, out| {
        *out = row_bytes(w, w_rb, r)
            .map(|row| vec_dot_q5_k_f32_row(row, x))
            .unwrap_or(0.0);
    });
    Ok(())
}

/// `y[m] = W_iq4xs[m, n_cols] x_f32[n_cols]`.
pub fn gemv_iq4_xs_f32(
    n_cols: usize,
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
) -> Result<(), QuantError> {
    let w_rb = iq4_xs_row_bytes(n_cols)?;
    require_len("x F32 elems", x.len(), n_cols)?;
    let expected_w = w_rb.checked_mul(y.len()).ok_or(QuantError::Size {
        what: "W IQ4_XS bytes overflow",
        expected: w_rb,
        actual: y.len(),
    })?;
    require_len("W IQ4_XS bytes", w.len(), expected_w)?;
    if y.is_empty() {
        return Ok(());
    }
    for_each_row(y, |r, out| {
        *out = row_bytes(w, w_rb, r)
            .map(|row| vec_dot_iq4_xs_f32_row(row, x))
            .unwrap_or(0.0);
    });
    Ok(())
}

/// Unpack one F16 GGUF row into `y[n_cols]` (`ggml_fp16_to_fp32` / IEEE binary16).
pub fn dequant_f16_row(n_cols: usize, row: &[u8], y: &mut [f32]) -> Result<(), QuantError> {
    let rb = f16_row_bytes(n_cols)?;
    require_len("F16 row bytes", row.len(), rb)?;
    require_len("F16 y elems", y.len(), n_cols)?;
    for (chunk, yv) in row.as_chunks::<2>().0.iter().zip(y.iter_mut()) {
        *yv = load_f16_le(chunk).unwrap_or(0.0);
    }
    Ok(())
}

/// Unpack one F32 GGUF row into `y[n_cols]`.
pub fn dequant_f32_row(n_cols: usize, row: &[u8], y: &mut [f32]) -> Result<(), QuantError> {
    let rb = f32_row_bytes(n_cols)?;
    require_len("F32 row bytes", row.len(), rb)?;
    require_len("F32 y elems", y.len(), n_cols)?;
    for (chunk, yv) in row.as_chunks::<4>().0.iter().zip(y.iter_mut()) {
        *yv = f32::from_bits(u32::from_le_bytes(*chunk));
    }
    Ok(())
}

/// Unpack one Q4_K GGUF row into `y[n_cols]` (`x = d*sc*q - dmin*m`).
pub fn dequant_q4_k_row(n_cols: usize, row: &[u8], y: &mut [f32]) -> Result<(), QuantError> {
    let rb = q4_k_row_bytes(n_cols)?;
    require_len("Q4_K row bytes", row.len(), rb)?;
    require_len("Q4_K y elems", y.len(), n_cols)?;
    for yv in y.iter_mut() {
        *yv = 0.0;
    }
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
        for group in 0..4 {
            let Some((sc0, m0)) = scale_min_k4(scales, group * 2) else {
                continue;
            };
            let Some((sc1, m1)) = scale_min_k4(scales, group * 2 + 1) else {
                continue;
            };
            let Some(packed) = qs.get(group * 32..group * 32 + 32) else {
                continue;
            };
            let a0 = d * f32::from(sc0);
            let b0 = dmin * f32::from(m0);
            let a1 = d * f32::from(sc1);
            let b1 = dmin * f32::from(m1);
            let lo_base = x_base.saturating_add(group * 64);
            let hi_base = lo_base.saturating_add(32);
            for (l, p) in packed.iter().enumerate() {
                let q0 = f32::from(p & 0x0f);
                let q1 = f32::from(p >> 4);
                if let Some(slot) = y.get_mut(lo_base.saturating_add(l)) {
                    *slot = a0 * q0 - b0;
                }
                if let Some(slot) = y.get_mut(hi_base.saturating_add(l)) {
                    *slot = a1 * q1 - b1;
                }
            }
        }
    }
    Ok(())
}

/// Unpack one Q5_K GGUF row into `y[n_cols]` (`x = d*sc*q5 - dmin*m`).
pub fn dequant_q5_k_row(n_cols: usize, row: &[u8], y: &mut [f32]) -> Result<(), QuantError> {
    let rb = q5_k_row_bytes(n_cols)?;
    require_len("Q5_K row bytes", row.len(), rb)?;
    require_len("Q5_K y elems", y.len(), n_cols)?;
    for yv in y.iter_mut() {
        *yv = 0.0;
    }
    let (w_blocks, _) = row.as_chunks::<Q5_K_BLOCK>();
    for (b, wb) in w_blocks.iter().enumerate() {
        let Some(d) = load_f16_le(wb) else { continue };
        let Some(dmin) = load_f16_le(wb.get(2..).unwrap_or(&[])) else {
            continue;
        };
        let Some(scales) = wb.get(4..16) else {
            continue;
        };
        let Some(qh) = wb.get(16..48) else { continue };
        let Some(qs) = wb.get(48..) else { continue };
        let x_base = b.saturating_mul(QK_K);
        for group in 0..4 {
            let Some((sc0, m0)) = scale_min_k4(scales, group * 2) else {
                continue;
            };
            let Some((sc1, m1)) = scale_min_k4(scales, group * 2 + 1) else {
                continue;
            };
            let Some(packed) = qs.get(group * 32..group * 32 + 32) else {
                continue;
            };
            let a0 = d * f32::from(sc0);
            let b0 = dmin * f32::from(m0);
            let a1 = d * f32::from(sc1);
            let b1 = dmin * f32::from(m1);
            let shift = u32::try_from(group.saturating_mul(2)).unwrap_or(0);
            let u1 = 1u8.wrapping_shl(shift);
            let u2 = 2u8.wrapping_shl(shift);
            let lo_base = x_base.saturating_add(group * 64);
            let hi_base = lo_base.saturating_add(32);
            for (l, p) in packed.iter().enumerate() {
                let Some(&qhv) = qh.get(l) else {
                    continue;
                };
                let q0 = f32::from(q5_from_nibble(*p & 0x0f, qhv, u1));
                let q1 = f32::from(q5_from_nibble(*p >> 4, qhv, u2));
                if let Some(slot) = y.get_mut(lo_base.saturating_add(l)) {
                    *slot = a0 * q0 - b0;
                }
                if let Some(slot) = y.get_mut(hi_base.saturating_add(l)) {
                    *slot = a1 * q1 - b1;
                }
            }
        }
    }
    Ok(())
}

/// Unpack one IQ4_XS GGUF row into `y[n_cols]` (`x = d * (ls - 32) * kvalues_iq4nl[q]`).
pub fn dequant_iq4_xs_row(n_cols: usize, row: &[u8], y: &mut [f32]) -> Result<(), QuantError> {
    let rb = iq4_xs_row_bytes(n_cols)?;
    require_len("IQ4_XS row bytes", row.len(), rb)?;
    require_len("IQ4_XS y elems", y.len(), n_cols)?;
    for yv in y.iter_mut() {
        *yv = 0.0;
    }
    let (w_blocks, _) = row.as_chunks::<IQ4_XS_BLOCK>();
    for (b, wb) in w_blocks.iter().enumerate() {
        let Some(d) = load_f16_le(wb) else { continue };
        let Some(&h0) = wb.get(2) else { continue };
        let Some(&h1) = wb.get(3) else { continue };
        let scales_h = u16::from_le_bytes([h0, h1]);
        let Some(scales_l) = wb.get(4..8) else {
            continue;
        };
        let Some(qs) = wb.get(8..) else { continue };
        let x_base = b.saturating_mul(QK_K);
        for ib in 0..8 {
            let ls = iq4_xs_ls(scales_l, scales_h, ib);
            let dl = d * (f32::from(ls) - 32.0);
            let qs_off = ib.saturating_mul(16);
            let Some(packed) = qs.get(qs_off..qs_off.saturating_add(16)) else {
                continue;
            };
            let lo_base = x_base.saturating_add(ib.saturating_mul(32));
            let hi_base = lo_base.saturating_add(16);
            for (j, p) in packed.iter().enumerate() {
                let q0 = f32::from(kvalue_iq4nl(*p & 0x0f));
                let q1 = f32::from(kvalue_iq4nl(*p >> 4));
                if let Some(slot) = y.get_mut(lo_base.saturating_add(j)) {
                    *slot = dl * q0;
                }
                if let Some(slot) = y.get_mut(hi_base.saturating_add(j)) {
                    *slot = dl * q1;
                }
            }
        }
    }
    Ok(())
}

/// Unpack one Q6_K GGUF row into `y[n_cols]` (`x = d * scale * q`).
pub fn dequant_q6_k_row(n_cols: usize, row: &[u8], y: &mut [f32]) -> Result<(), QuantError> {
    let rb = q6_k_row_bytes(n_cols)?;
    require_len("Q6_K row bytes", row.len(), rb)?;
    require_len("Q6_K y elems", y.len(), n_cols)?;
    for yv in y.iter_mut() {
        *yv = 0.0;
    }
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
        for group in 0..2 {
            let ql_off = group * 64;
            let qh_off = group * 32;
            let sc_off = group * 8;
            let y_off = x_base.saturating_add(group * 128);
            for l in 0..32 {
                let is = l / 16;
                let Some(&ql0) = ql.get(ql_off + l) else {
                    continue;
                };
                let Some(&ql1) = ql.get(ql_off + l + 32) else {
                    continue;
                };
                let Some(&qhv) = qh.get(qh_off + l) else {
                    continue;
                };
                let q1 = q6_from_bits((ql0 & 0x0f) | ((qhv & 3) << 4));
                let q2 = q6_from_bits((ql1 & 0x0f) | (((qhv >> 2) & 3) << 4));
                let q3 = q6_from_bits((ql0 >> 4) | (((qhv >> 4) & 3) << 4));
                let q4 = q6_from_bits((ql1 >> 4) | (((qhv >> 6) & 3) << 4));
                let Some(&s0) = scb.get(sc_off + is) else {
                    continue;
                };
                let Some(&s2) = scb.get(sc_off + is + 2) else {
                    continue;
                };
                let Some(&s4) = scb.get(sc_off + is + 4) else {
                    continue;
                };
                let Some(&s6) = scb.get(sc_off + is + 6) else {
                    continue;
                };
                let v1 = d * f32::from(i8_from_bits(s0)) * f32::from(q1);
                let v2 = d * f32::from(i8_from_bits(s2)) * f32::from(q2);
                let v3 = d * f32::from(i8_from_bits(s4)) * f32::from(q3);
                let v4 = d * f32::from(i8_from_bits(s6)) * f32::from(q4);
                if let Some(slot) = y.get_mut(y_off.saturating_add(l)) {
                    *slot = v1;
                }
                if let Some(slot) = y.get_mut(y_off.saturating_add(l + 32)) {
                    *slot = v2;
                }
                if let Some(slot) = y.get_mut(y_off.saturating_add(l + 64)) {
                    *slot = v3;
                }
                if let Some(slot) = y.get_mut(y_off.saturating_add(l + 96)) {
                    *slot = v4;
                }
            }
        }
    }
    Ok(())
}

#[inline(never)]
fn add_f32(a: f32, b: f32) -> f32 {
    a + b
}

fn vec_dot_q8_row(row: &[u8], x: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    let (w_blocks, _) = row.as_chunks::<Q8_0_BLOCK>();
    let (x_blocks, _) = x.as_chunks::<Q8_0_BLOCK>();
    for (wb, xb) in w_blocks.iter().zip(x_blocks.iter()) {
        let Some(dw) = load_f16_le(wb) else { continue };
        let Some(dx) = load_f16_le(xb) else { continue };
        let Some(wqs) = wb.get(2..) else { continue };
        let Some(xqs) = xb.get(2..) else { continue };
        let mut acc = 0i32;
        for (wq, xq) in wqs.iter().zip(xqs.iter()).take(QK8_0) {
            acc += i32::from(i8_from_bits(*wq)) * i32::from(i8_from_bits(*xq));
        }
        sum = add_f32(sum, (acc as f32) * (dw * dx));
    }
    sum
}

fn vec_dot_q4_row(row: &[u8], x: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    let (w_blocks, _) = row.as_chunks::<Q4_0_BLOCK>();
    let (x_blocks, _) = x.as_chunks::<Q8_0_BLOCK>();
    for (wb, xb) in w_blocks.iter().zip(x_blocks.iter()) {
        let Some(dw) = load_f16_le(wb) else { continue };
        let Some(dx) = load_f16_le(xb) else { continue };
        let Some(packed) = wb.get(2..) else { continue };
        let Some(xqs) = xb.get(2..) else { continue };
        let Some(xlo) = xqs.get(..16) else { continue };
        let Some(xhi) = xqs.get(16..32) else { continue };
        let mut acc = 0i32;
        for ((p, xl), xh) in packed.iter().zip(xlo.iter()).zip(xhi.iter()) {
            let lo = i32::from(*p & 0x0f) - 8;
            let hi = i32::from(*p >> 4) - 8;
            acc += lo * i32::from(i8_from_bits(*xl));
            acc += hi * i32::from(i8_from_bits(*xh));
        }
        sum += (acc as f32) * (dw * dx);
    }
    sum
}

fn scale_min_k4(scales: &[u8], j: usize) -> Option<(u8, u8)> {
    if j < 4 {
        let d = *scales.get(j)? & 63;
        let m = *scales.get(j + 4)? & 63;
        Some((d, m))
    } else {
        let hi = *scales.get(j + 4)?;
        let d = (hi & 0x0f) | ((scales.get(j.saturating_sub(4))? >> 6) << 4);
        let m = (hi >> 4) | ((scales.get(j)? >> 6) << 4);
        Some((d, m))
    }
}

fn vec_dot_q4_k_row(row: &[u8], x: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    let (w_blocks, _) = row.as_chunks::<Q4_K_BLOCK>();
    let (x_blocks, _) = x.as_chunks::<Q8_K_BLOCK>();
    for (wb, xb) in w_blocks.iter().zip(x_blocks.iter()) {
        let Some(d) = load_f16_le(wb) else { continue };
        let Some(dmin) = load_f16_le(wb.get(2..).unwrap_or(&[])) else {
            continue;
        };
        let Some(scales) = wb.get(4..16) else {
            continue;
        };
        let Some(qs) = wb.get(16..144) else { continue };
        let Some(dx) = load_f32_le(xb) else { continue };
        let Some(xqs) = xb.get(4..260) else { continue };
        for group in 0..4 {
            let Some((sc0, m0)) = scale_min_k4(scales, group * 2) else {
                continue;
            };
            let Some((sc1, m1)) = scale_min_k4(scales, group * 2 + 1) else {
                continue;
            };
            let Some(packed) = qs.get(group * 32..group * 32 + 32) else {
                continue;
            };
            let x_base = group * 64;
            let Some(xlo) = xqs.get(x_base..x_base + 32) else {
                continue;
            };
            let Some(xhi) = xqs.get(x_base + 32..x_base + 64) else {
                continue;
            };
            let a0 = d * f32::from(sc0);
            let b0 = dmin * f32::from(m0);
            let a1 = d * f32::from(sc1);
            let b1 = dmin * f32::from(m1);
            for ((p, xl), xh) in packed.iter().zip(xlo.iter()).zip(xhi.iter()) {
                let q0 = f32::from(p & 0x0f);
                let q1 = f32::from(p >> 4);
                let x0 = f32::from(i8_from_bits(*xl)) * dx;
                let x1 = f32::from(i8_from_bits(*xh)) * dx;
                sum += (a0 * q0 - b0) * x0;
                sum += (a1 * q1 - b1) * x1;
            }
        }
    }
    sum
}

fn vec_dot_f16_row(row: &[u8], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (chunk, xv) in row.as_chunks::<2>().0.iter().zip(x.iter()) {
        sum += load_f16_le(chunk).unwrap_or(0.0) * *xv;
    }
    sum
}

fn vec_dot_f32_row(row: &[u8], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (chunk, xv) in row.as_chunks::<4>().0.iter().zip(x.iter()) {
        sum += f32::from_bits(u32::from_le_bytes(*chunk)) * *xv;
    }
    sum
}

fn vec_dot_q4_k_f32_row(row: &[u8], x: &[f32]) -> f32 {
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
        let Some(xr) = x.get(x_base..x_base + QK_K) else {
            continue;
        };
        for group in 0..4 {
            let Some((sc0, m0)) = scale_min_k4(scales, group * 2) else {
                continue;
            };
            let Some((sc1, m1)) = scale_min_k4(scales, group * 2 + 1) else {
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
            let a0 = d * f32::from(sc0);
            let b0 = dmin * f32::from(m0);
            let a1 = d * f32::from(sc1);
            let b1 = dmin * f32::from(m1);
            for ((p, xl), xh) in packed.iter().zip(xlo.iter()).zip(xhi.iter()) {
                let q0 = f32::from(p & 0x0f);
                let q1 = f32::from(p >> 4);
                sum += (a0 * q0 - b0) * *xl;
                sum += (a1 * q1 - b1) * *xh;
            }
        }
    }
    sum
}

fn vec_dot_q5_k_f32_row(row: &[u8], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let (w_blocks, _) = row.as_chunks::<Q5_K_BLOCK>();
    for (b, wb) in w_blocks.iter().enumerate() {
        let Some(d) = load_f16_le(wb) else { continue };
        let Some(dmin) = load_f16_le(wb.get(2..).unwrap_or(&[])) else {
            continue;
        };
        let Some(scales) = wb.get(4..16) else {
            continue;
        };
        let Some(qh) = wb.get(16..48) else { continue };
        let Some(qs) = wb.get(48..) else { continue };
        let x_base = b.saturating_mul(QK_K);
        let Some(xr) = x.get(x_base..x_base + QK_K) else {
            continue;
        };
        for group in 0..4 {
            let Some((sc0, m0)) = scale_min_k4(scales, group * 2) else {
                continue;
            };
            let Some((sc1, m1)) = scale_min_k4(scales, group * 2 + 1) else {
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
            let a0 = d * f32::from(sc0);
            let b0 = dmin * f32::from(m0);
            let a1 = d * f32::from(sc1);
            let b1 = dmin * f32::from(m1);
            let shift = u32::try_from(group.saturating_mul(2)).unwrap_or(0);
            let u1 = 1u8.wrapping_shl(shift);
            let u2 = 2u8.wrapping_shl(shift);
            for (l, ((p, xl), xh)) in packed.iter().zip(xlo.iter()).zip(xhi.iter()).enumerate() {
                let Some(&qhv) = qh.get(l) else {
                    continue;
                };
                let q0 = f32::from(q5_from_nibble(*p & 0x0f, qhv, u1));
                let q1 = f32::from(q5_from_nibble(*p >> 4, qhv, u2));
                sum += (a0 * q0 - b0) * *xl;
                sum += (a1 * q1 - b1) * *xh;
            }
        }
    }
    sum
}

fn vec_dot_iq4_xs_f32_row(row: &[u8], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let (w_blocks, _) = row.as_chunks::<IQ4_XS_BLOCK>();
    for (b, wb) in w_blocks.iter().enumerate() {
        let Some(d) = load_f16_le(wb) else { continue };
        let Some(&h0) = wb.get(2) else { continue };
        let Some(&h1) = wb.get(3) else { continue };
        let scales_h = u16::from_le_bytes([h0, h1]);
        let Some(scales_l) = wb.get(4..8) else {
            continue;
        };
        let Some(qs) = wb.get(8..) else { continue };
        let x_base = b.saturating_mul(QK_K);
        let Some(xr) = x.get(x_base..x_base.saturating_add(QK_K)) else {
            continue;
        };
        for ib in 0..8 {
            let ls = iq4_xs_ls(scales_l, scales_h, ib);
            let dl = d * (f32::from(ls) - 32.0);
            let qs_off = ib.saturating_mul(16);
            let Some(packed) = qs.get(qs_off..qs_off.saturating_add(16)) else {
                continue;
            };
            let xb = ib.saturating_mul(32);
            let Some(xlo) = xr.get(xb..xb.saturating_add(16)) else {
                continue;
            };
            let Some(xhi) = xr.get(xb.saturating_add(16)..xb.saturating_add(32)) else {
                continue;
            };
            for ((p, xl), xh) in packed.iter().zip(xlo.iter()).zip(xhi.iter()) {
                let q0 = f32::from(kvalue_iq4nl(*p & 0x0f));
                let q1 = f32::from(kvalue_iq4nl(*p >> 4));
                sum += (dl * q0) * *xl;
                sum += (dl * q1) * *xh;
            }
        }
    }
    sum
}

fn vec_dot_q6_k_f32_row(row: &[u8], x: &[f32]) -> f32 {
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
        let Some(xr) = x.get(x_base..x_base + QK_K) else {
            continue;
        };
        for group in 0..2 {
            let ql_off = group * 64;
            let qh_off = group * 32;
            let sc_off = group * 8;
            let y_off = group * 128;
            for l in 0..32 {
                let is = l / 16;
                let Some(&ql0) = ql.get(ql_off + l) else {
                    continue;
                };
                let Some(&ql1) = ql.get(ql_off + l + 32) else {
                    continue;
                };
                let Some(&qhv) = qh.get(qh_off + l) else {
                    continue;
                };
                let q1 = q6_from_bits((ql0 & 0x0f) | ((qhv & 3) << 4));
                let q2 = q6_from_bits((ql1 & 0x0f) | (((qhv >> 2) & 3) << 4));
                let q3 = q6_from_bits((ql0 >> 4) | (((qhv >> 4) & 3) << 4));
                let q4 = q6_from_bits((ql1 >> 4) | (((qhv >> 6) & 3) << 4));
                let Some(&s0) = scb.get(sc_off + is) else {
                    continue;
                };
                let Some(&s2) = scb.get(sc_off + is + 2) else {
                    continue;
                };
                let Some(&s4) = scb.get(sc_off + is + 4) else {
                    continue;
                };
                let Some(&s6) = scb.get(sc_off + is + 6) else {
                    continue;
                };
                let Some(&x1) = xr.get(y_off + l) else {
                    continue;
                };
                let Some(&x2) = xr.get(y_off + l + 32) else {
                    continue;
                };
                let Some(&x3) = xr.get(y_off + l + 64) else {
                    continue;
                };
                let Some(&x4) = xr.get(y_off + l + 96) else {
                    continue;
                };
                sum += d * f32::from(i8_from_bits(s0)) * f32::from(q1) * x1;
                sum += d * f32::from(i8_from_bits(s2)) * f32::from(q2) * x2;
                sum += d * f32::from(i8_from_bits(s4)) * f32::from(q3) * x3;
                sum += d * f32::from(i8_from_bits(s6)) * f32::from(q4) * x4;
            }
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemv_q8_rejects_bad_x_len() {
        let mut y = [0.0f32];
        assert!(gemv_q8_0(32, &[0u8; 34], &[], &mut y).is_err());
    }

    #[test]
    fn gemv_q4_rejects_unaligned_cols() {
        let mut y = [0.0f32];
        assert!(gemv_q4_0(31, &[], &[], &mut y).is_err());
    }

    #[test]
    fn gemv_q4_k_rejects_unaligned_cols() {
        let mut y = [0.0f32];
        assert!(gemv_q4_k(255, &[], &[], &mut y).is_err());
    }

    #[test]
    fn gemv_q5_k_rejects_unaligned_cols() {
        let mut y = [0.0f32];
        assert!(gemv_q5_k_f32(255, &[], &[], &mut y).is_err());
    }

    fn dequant_q8_row(bytes: &[u8]) -> Vec<f32> {
        let mut out = Vec::new();
        let (blocks, _) = bytes.as_chunks::<Q8_0_BLOCK>();
        for block in blocks {
            let Some(d) = load_f16_le(block) else {
                continue;
            };
            let Some(qs) = block.get(2..) else {
                continue;
            };
            for q in qs.iter().take(QK8_0) {
                out.push(d * f32::from(i8_from_bits(*q)));
            }
        }
        out
    }

    fn oracle_q8_dot(w_row: &[u8], x: &[u8]) -> f32 {
        dequant_q8_row(w_row)
            .iter()
            .zip(dequant_q8_row(x).iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    #[test]
    fn gemv_q8_multi_row_parallel_matches_sequential_and_oracle() {
        let n_cols = 64usize;
        let n_rows = 8usize;
        let mut w = Vec::new();
        let mut x = Vec::new();
        for r in 0..n_rows {
            for _b in 0..(n_cols / QK8_0) {
                let mut qs = [0i8; QK8_0];
                for (i, q) in qs.iter_mut().enumerate() {
                    let base = i8::try_from(r).unwrap_or(0);
                    let off = i8::try_from(i).unwrap_or(0);
                    *q = base.wrapping_add(off).wrapping_sub(16);
                }
                w.extend_from_slice(&pack_q8_0_block(
                    5.0 / 100.0 + f32::from(u16::try_from(r).unwrap_or(0)) / 100.0,
                    &qs,
                ));
            }
        }
        for b in 0..(n_cols / QK8_0) {
            let mut qs = [0i8; QK8_0];
            for (i, q) in qs.iter_mut().enumerate() {
                let base = i8::try_from(b).unwrap_or(0);
                let off = i8::try_from(i).unwrap_or(0);
                *q = base.wrapping_add(off).wrapping_sub(8);
            }
            x.extend_from_slice(&pack_q8_0_block(2.0 / 100.0, &qs));
        }

        let mut y_par = vec![0.0f32; n_rows];
        gemv_q8_0(n_cols, &w, &x, &mut y_par).unwrap();
        let mut y_seq = vec![0.0f32; n_rows];
        crate::pool::with_sequential(|| {
            gemv_q8_0(n_cols, &w, &x, &mut y_seq).unwrap();
        });
        assert_eq!(y_par, y_seq);

        let rb = q8_0_row_bytes(n_cols).unwrap();
        for (r, yv) in y_par.iter().enumerate() {
            let start = r.saturating_mul(rb);
            let row = w.get(start..start.saturating_add(rb)).unwrap_or(&[]);
            let expected = oracle_q8_dot(row, &x);
            let rel = (yv - expected).abs() / (1.0 + expected.abs());
            assert!(rel * 100_000.0 < 1.0, "row {r}: {yv} vs {expected}");
        }
    }

    #[test]
    fn pack_q6_k_gemv_f32_matches_one_nonzero() {
        let mut qs = [0i8; QK_K];
        qs[0] = 1;
        qs[32] = 2;
        let mut sc = [0i8; 16];
        sc[0] = 1;
        sc[2] = 1;
        let w = pack_q6_k_block(1.0, &sc, &qs);
        let mut x = [0.0f32; QK_K];
        x[0] = 3.0;
        x[32] = 4.0;
        let mut y = [0.0f32];
        gemv_q6_k_f32(QK_K, &w, &x, &mut y).unwrap();
        // y[0]*d*sc0*x0 + y[32]*d*sc2*x32 = 1*1*3 + 2*1*4 = 11
        assert!((y[0] - 11.0).abs() * 100_000.0 < 1.0, "{}", y[0]);
        let mut row = [0.0f32; QK_K];
        dequant_q6_k_row(QK_K, &w, &mut row).unwrap();
        let via_dequant: f32 = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        assert!(
            (via_dequant - y[0]).abs() * 100_000.0 < 1.0,
            "{via_dequant} vs {}",
            y[0]
        );
    }

    #[test]
    fn dequant_q4_k_row_dot_matches_gemv() {
        let mut qs = [0u8; QK_K];
        qs[0] = 3;
        qs[32] = 5;
        let sc = [2u8, 1, 1, 1, 1, 1, 1, 1];
        let mn = [0u8; 8];
        let w = pack_q4_k_block(25.0 / 100.0, 0.0, &sc, &mn, &qs);
        let mut x = [0.0f32; QK_K];
        x[0] = 2.0;
        x[32] = 4.0;
        let mut y = [0.0f32];
        gemv_q4_k_f32(QK_K, &w, &x, &mut y).unwrap();
        let mut row = [0.0f32; QK_K];
        dequant_q4_k_row(QK_K, &w, &mut row).unwrap();
        let via_dequant: f32 = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        assert!(
            (via_dequant - y[0]).abs() * 100_000.0 < 1.0,
            "{via_dequant} vs {}",
            y[0]
        );
    }

    /// ggml `get_scale_min_k4` (oracle; not the GEMV loop).
    fn oracle_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
        if j < 4 {
            (q[j] & 63, q[j + 4] & 63)
        } else {
            (
                (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4),
                (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
            )
        }
    }

    /// ggml `dequantize_row_q5_K` (oracle). Independent of `dequant_q5_k_row` / `gemv_q5_k_f32`.
    fn oracle_q5_k_row(w: &[u8]) -> Vec<f32> {
        let nblocks = w.len() / Q5_K_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK_K];
        for b in 0..nblocks {
            let wb = &w[b * Q5_K_BLOCK..(b + 1) * Q5_K_BLOCK];
            let d = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let dmin = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[2], wb[3]]));
            let scales = &wb[4..16];
            let qh = &wb[16..48];
            let mut ql_off = 48usize;
            let mut yo = b * QK_K;
            let mut is = 0usize;
            let mut u1: u8 = 1;
            let mut u2: u8 = 2;
            for _ in 0..4 {
                let (sc, m) = oracle_scale_min_k4(is, scales);
                let d1 = d * f32::from(sc);
                let m1 = dmin * f32::from(m);
                let (sc, m) = oracle_scale_min_k4(is + 1, scales);
                let d2 = d * f32::from(sc);
                let m2 = dmin * f32::from(m);
                let q = &wb[ql_off..ql_off + 32];
                for l in 0..32 {
                    let q5 = (q[l] & 0x0f) + if qh[l] & u1 == 0 { 0 } else { 16 };
                    y[yo + l] = d1 * f32::from(q5) - m1;
                }
                yo += 32;
                for l in 0..32 {
                    let q5 = (q[l] >> 4) + if qh[l] & u2 == 0 { 0 } else { 16 };
                    y[yo + l] = d2 * f32::from(q5) - m2;
                }
                yo += 32;
                ql_off += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        y
    }

    fn oracle_q5_k_dot(row: &[u8], x: &[f32]) -> f32 {
        oracle_q5_k_row(row)
            .iter()
            .zip(x.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    #[test]
    fn pack_q5_k_gemv_and_dequant_match_independent_oracle() {
        let mut qs = [0u8; QK_K];
        qs[0] = 3;
        qs[32] = 17;
        qs[64] = 20;
        qs[128] = 21;
        let sc = [2u8, 1, 3, 1, 4, 1, 1, 1];
        let mn = [1u8, 0, 0, 0, 2, 0, 0, 0];
        let w = pack_q5_k_block(25.0 / 100.0, 5.0 / 100.0, &sc, &mn, &qs);
        assert_eq!(w.len(), Q5_K_BLOCK);
        // ggml qh[0]: bit 1 (q=17), bit 2 (q=20), bit 4 (q=21). Bit 0 clear (q=3).
        assert_eq!(w[16] & 0b0001_0111, 0b0001_0110);
        let mut x = [0.0f32; QK_K];
        x[0] = 2.0;
        x[32] = 4.0;
        x[64] = 1.0;
        x[128] = 3.0;
        let mut y = [0.0f32];
        gemv_q5_k_f32(QK_K, &w, &x, &mut y).unwrap();
        let expected = oracle_q5_k_dot(&w, &x);
        let rel = (y[0] - expected).abs() / (1.0 + expected.abs());
        assert!(rel * 100_000.0 < 1.0, "gemv {} vs {expected}", y[0]);
        let mut row = [0.0f32; QK_K];
        dequant_q5_k_row(QK_K, &w, &mut row).unwrap();
        let oracle = oracle_q5_k_row(&w);
        assert_close(&row, &oracle);
        let via_dequant: f32 = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        assert!(
            (via_dequant - y[0]).abs() * 100_000.0 < 1.0,
            "{via_dequant} vs {}",
            y[0]
        );
    }

    fn assert_close(got: &[f32], exp: &[f32]) {
        assert_eq!(got.len(), exp.len());
        for (i, (a, b)) in got.iter().zip(exp.iter()).enumerate() {
            let rel = (a - b).abs() / (1.0 + b.abs());
            assert!(rel * 100_000.0 < 1.0, "elem {i}: {a} vs {b}");
        }
    }

    /// Independent IEEE binary16 element (ggml `ggml_fp16_to_fp32`).
    fn oracle_f16_elem(bytes: &[u8]) -> f32 {
        crate::fp16::f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn oracle_f16_dot(row: &[u8], x: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for (i, xv) in x.iter().enumerate() {
            let off = i * 2;
            sum += oracle_f16_elem(&row[off..off + 2]) * *xv;
        }
        sum
    }

    #[test]
    fn pack_f16_gemv_and_dequant_match_independent_oracle() {
        let vals = [1.0f32, -0.5, 0.25, 2.0, -1.0, 0.125, 3.5, -2.0];
        let w = pack_f16(&vals);
        assert_eq!(w.len(), vals.len() * F16_SIZE);
        let mut row = vec![0.0f32; vals.len()];
        dequant_f16_row(vals.len(), &w, &mut row).unwrap();
        for (i, (got, src)) in row.iter().zip(vals.iter()).enumerate() {
            let exp = oracle_f16_elem(&w[i * 2..i * 2 + 2]);
            let rel = (got - exp).abs() / (1.0 + exp.abs());
            assert!(rel * 100_000.0 < 1.0, "dequant {i}: {got} vs {exp}");
            let back = crate::fp16::f16_to_f32(crate::fp16::f32_to_f16(*src));
            assert_eq!(*got, back);
        }
        let x = [1.0f32, 2.0, 0.5, -1.0, 0.25, 4.0, -0.5, 1.0];
        let mut y = [0.0f32];
        gemv_f16(vals.len(), &w, &x, &mut y).unwrap();
        let expected = oracle_f16_dot(&w, &x);
        let rel = (y[0] - expected).abs() / (1.0 + expected.abs());
        assert!(rel * 100_000.0 < 1.0, "gemv {} vs {expected}", y[0]);
        let via_dequant: f32 = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        assert!(
            (via_dequant - y[0]).abs() * 100_000.0 < 1.0,
            "{via_dequant} vs {}",
            y[0]
        );
    }

    #[test]
    fn gemm_f16_matches_repeated_gemv() {
        let n_cols = 4usize;
        let n_rows = 3usize;
        let n_tokens = 3usize;
        let mut w = Vec::new();
        for r in 0..n_rows {
            for c in 0..n_cols {
                let v = f32::from(u16::try_from(r * 10 + c + 1).unwrap_or(1)) / 10.0;
                w.extend_from_slice(&store_f16_le(v));
            }
        }
        let mut x = Vec::new();
        for t in 0..n_tokens {
            for c in 0..n_cols {
                x.push(f32::from(u16::try_from(t + c + 1).unwrap_or(1)) / 4.0);
            }
        }
        let mut y_gemm = vec![0.0f32; n_rows * n_tokens];
        gemm_f16(n_cols, n_tokens, &w, &x, &mut y_gemm).unwrap();
        let mut y_gemv = Vec::new();
        for t in 0..n_tokens {
            let start = t * n_cols;
            let xt = &x[start..start + n_cols];
            let mut yt = vec![0.0f32; n_rows];
            gemv_f16(n_cols, &w, xt, &mut yt).unwrap();
            y_gemv.extend_from_slice(&yt);
        }
        assert_close(&y_gemm, &y_gemv);
        let mut y_oracle = Vec::new();
        for t in 0..n_tokens {
            let xt = &x[t * n_cols..(t + 1) * n_cols];
            let rb = n_cols * F16_SIZE;
            for r in 0..n_rows {
                let row = &w[r * rb..(r + 1) * rb];
                y_oracle.push(oracle_f16_dot(row, xt));
            }
        }
        assert_close(&y_gemm, &y_oracle);
        crate::pool::with_sequential(|| {
            let mut y_seq = vec![0.0f32; n_rows * n_tokens];
            gemm_f16(n_cols, n_tokens, &w, &x, &mut y_seq).unwrap();
            assert_close(&y_seq, &y_oracle);
        });
    }

    #[test]
    fn gemm_f32_q4k_q6k_match_repeated_gemv() {
        let n_tokens = 3usize;

        let n_cols_f = 4usize;
        let n_rows_f = 3usize;
        let mut w_f = Vec::new();
        for r in 0..n_rows_f {
            for c in 0..n_cols_f {
                let v = f32::from(u16::try_from(r * 10 + c + 1).unwrap_or(1)) / 10.0;
                w_f.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        let mut x_f = Vec::new();
        for t in 0..n_tokens {
            for c in 0..n_cols_f {
                x_f.push(f32::from(u16::try_from(t + c + 1).unwrap_or(1)) / 4.0);
            }
        }
        let mut y_gemm = vec![0.0f32; n_rows_f * n_tokens];
        gemm_f32(n_cols_f, n_tokens, &w_f, &x_f, &mut y_gemm).unwrap();
        let mut y_gemv = Vec::new();
        for t in 0..n_tokens {
            let start = t * n_cols_f;
            let xt = &x_f[start..start + n_cols_f];
            let mut yt = vec![0.0f32; n_rows_f];
            gemv_f32(n_cols_f, &w_f, xt, &mut yt).unwrap();
            y_gemv.extend_from_slice(&yt);
        }
        assert_eq!(y_gemm, y_gemv);

        let n_cols = QK_K;
        let n_rows = 2usize;
        let mut w4 = Vec::new();
        let mut w6 = Vec::new();
        for r in 0..n_rows {
            let mut qs4 = [0u8; QK_K];
            let mut qs6 = [0i8; QK_K];
            qs4[0] = u8::try_from(2 + r).unwrap_or(2);
            qs4[32] = u8::try_from(3 + r).unwrap_or(3);
            qs6[0] = i8::try_from(1 + r).unwrap_or(1);
            qs6[32] = i8::try_from(2 + r).unwrap_or(2);
            let sc4 = [2u8, 1, 1, 1, 1, 1, 1, 1];
            let mut sc6 = [0i8; 16];
            sc6[0] = 1;
            sc6[2] = 1;
            w4.extend_from_slice(&pack_q4_k_block(25.0 / 100.0, 0.0, &sc4, &[0u8; 8], &qs4));
            w6.extend_from_slice(&pack_q6_k_block(1.0, &sc6, &qs6));
        }
        let mut xk = vec![0.0f32; n_cols * n_tokens];
        for t in 0..n_tokens {
            if let Some(slot) = xk.get_mut(t * n_cols) {
                *slot = f32::from(u16::try_from(t + 1).unwrap_or(1));
            }
            if let Some(slot) = xk.get_mut(t * n_cols + 32) {
                *slot = 2.0;
            }
        }
        let mut y4 = vec![0.0f32; n_rows * n_tokens];
        let mut y6 = vec![0.0f32; n_rows * n_tokens];
        gemm_q4_k_f32(n_cols, n_tokens, &w4, &xk, &mut y4).unwrap();
        gemm_q6_k_f32(n_cols, n_tokens, &w6, &xk, &mut y6).unwrap();
        let mut exp4 = Vec::new();
        let mut exp6 = Vec::new();
        for t in 0..n_tokens {
            let xt = &xk[t * n_cols..(t + 1) * n_cols];
            let mut a = vec![0.0f32; n_rows];
            let mut b = vec![0.0f32; n_rows];
            gemv_q4_k_f32(n_cols, &w4, xt, &mut a).unwrap();
            gemv_q6_k_f32(n_cols, &w6, xt, &mut b).unwrap();
            exp4.extend_from_slice(&a);
            exp6.extend_from_slice(&b);
        }
        assert_close(&y4, &exp4);
        assert_close(&y6, &exp6);

        crate::pool::with_sequential(|| {
            let mut y_seq = vec![0.0f32; n_rows * n_tokens];
            gemm_q4_k_f32(n_cols, n_tokens, &w4, &xk, &mut y_seq).unwrap();
            assert_close(&y_seq, &exp4);
        });

        let mut w5 = Vec::new();
        for r in 0..n_rows {
            let mut qs5 = [0u8; QK_K];
            qs5[0] = u8::try_from(3 + r).unwrap_or(3);
            qs5[32] = u8::try_from(17 + r).unwrap_or(17);
            qs5[64] = 20;
            let sc5 = [2u8, 1, 3, 1, 1, 1, 1, 1];
            w5.extend_from_slice(&pack_q5_k_block(
                25.0 / 100.0,
                5.0 / 100.0,
                &sc5,
                &[1u8, 0, 0, 0, 0, 0, 0, 0],
                &qs5,
            ));
        }
        let mut y5 = vec![0.0f32; n_rows * n_tokens];
        gemm_q5_k_f32(n_cols, n_tokens, &w5, &xk, &mut y5).unwrap();
        let mut exp5 = Vec::new();
        let mut exp5_oracle = Vec::new();
        let rb5 = Q5_K_BLOCK;
        for t in 0..n_tokens {
            let xt = &xk[t * n_cols..(t + 1) * n_cols];
            let mut a = vec![0.0f32; n_rows];
            gemv_q5_k_f32(n_cols, &w5, xt, &mut a).unwrap();
            exp5.extend_from_slice(&a);
            for r in 0..n_rows {
                let row = &w5[r * rb5..(r + 1) * rb5];
                exp5_oracle.push(oracle_q5_k_dot(row, xt));
            }
        }
        assert_close(&y5, &exp5);
        assert_close(&y5, &exp5_oracle);
        crate::pool::with_sequential(|| {
            let mut y_seq = vec![0.0f32; n_rows * n_tokens];
            gemm_q5_k_f32(n_cols, n_tokens, &w5, &xk, &mut y_seq).unwrap();
            assert_close(&y_seq, &exp5_oracle);
        });
    }

    /// ggml `dequantize_row_iq4_xs` (oracle). Independent of crate kernels.
    fn oracle_iq4_xs_row(w: &[u8]) -> Vec<f32> {
        const KVALUES: [i8; 16] = [
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        ];
        let nblocks = w.len() / IQ4_XS_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK_K];
        for b in 0..nblocks {
            let wb = &w[b * IQ4_XS_BLOCK..(b + 1) * IQ4_XS_BLOCK];
            let d = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let scales_h = u16::from_le_bytes([wb[2], wb[3]]);
            let scales_l = &wb[4..8];
            let qs = &wb[8..];
            let mut yo = b * QK_K;
            for ib in 0..8 {
                let sl = scales_l[ib / 2];
                let lo = (sl >> (4 * (ib % 2))) & 0x0f;
                let hi = u8::try_from((u32::from(scales_h) >> (2 * ib)) & 3).unwrap_or(0);
                let ls = lo | (hi << 4);
                let dl = d * (f32::from(ls) - 32.0);
                let packed = &qs[ib * 16..ib * 16 + 16];
                for j in 0..16 {
                    y[yo + j] = dl * f32::from(KVALUES[usize::from(packed[j] & 0x0f)]);
                    y[yo + j + 16] = dl * f32::from(KVALUES[usize::from(packed[j] >> 4)]);
                }
                yo += 32;
            }
        }
        y
    }

    fn oracle_iq4_xs_dot(row: &[u8], x: &[f32]) -> f32 {
        oracle_iq4_xs_row(row)
            .iter()
            .zip(x.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    #[test]
    fn pack_iq4_xs_gemv_and_dequant_match_independent_oracle() {
        let mut qs = [0u8; QK_K];
        qs[0] = 3;
        qs[16] = 12;
        qs[32] = 7;
        qs[48] = 15;
        let sc = [33u8, 34, 31, 32, 40, 24, 36, 28];
        let w = pack_iq4_xs_block(25.0 / 100.0, &sc, &qs);
        assert_eq!(w.len(), IQ4_XS_BLOCK);
        // scales_l[0] = (33 & 0xf) | ((34 & 0xf) << 4) = 1 | (2 << 4) = 0x21
        assert_eq!(w[4], 0x21);
        // scales_h low bits: ib0 hi=(33>>4)&3=2, ib1 hi=(34>>4)&3=2 → bits 0..3 = 0b1010
        assert_eq!(w[2] & 0x0f, 0b1010);
        let mut x = [0.0f32; QK_K];
        x[0] = 2.0;
        x[16] = 4.0;
        x[32] = 1.0;
        x[48] = 3.0;
        let mut y = [0.0f32];
        gemv_iq4_xs_f32(QK_K, &w, &x, &mut y).unwrap();
        let expected = oracle_iq4_xs_dot(&w, &x);
        let rel = (y[0] - expected).abs() / (1.0 + expected.abs());
        assert!(rel * 100_000.0 < 1.0, "gemv {} vs {expected}", y[0]);
        let mut row = [0.0f32; QK_K];
        dequant_iq4_xs_row(QK_K, &w, &mut row).unwrap();
        let oracle = oracle_iq4_xs_row(&w);
        assert_close(&row, &oracle);
        let via_dequant: f32 = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        assert!(
            (via_dequant - y[0]).abs() * 100_000.0 < 1.0,
            "{via_dequant} vs {}",
            y[0]
        );
    }

    #[test]
    fn gemm_iq4_xs_matches_repeated_gemv_and_oracle() {
        let n_cols = QK_K;
        let n_rows = 2usize;
        let n_tokens = 3usize;
        let mut w = Vec::new();
        for r in 0..n_rows {
            let mut qs = [0u8; QK_K];
            qs[0] = u8::try_from(3 + r).unwrap_or(3);
            qs[16] = u8::try_from(12 + r).unwrap_or(12);
            qs[32] = 7;
            let sc = [33u8, 34, 31, 32, 40, 24, 36, 28];
            w.extend_from_slice(&pack_iq4_xs_block(25.0 / 100.0, &sc, &qs));
        }
        let mut xk = vec![0.0f32; n_cols * n_tokens];
        for t in 0..n_tokens {
            if let Some(slot) = xk.get_mut(t * n_cols) {
                *slot = f32::from(u16::try_from(t + 1).unwrap_or(1));
            }
            if let Some(slot) = xk.get_mut(t * n_cols + 16) {
                *slot = 2.0;
            }
        }
        let mut y = vec![0.0f32; n_rows * n_tokens];
        gemm_iq4_xs_f32(n_cols, n_tokens, &w, &xk, &mut y).unwrap();
        let mut exp = Vec::new();
        let mut exp_oracle = Vec::new();
        for t in 0..n_tokens {
            let xt = &xk[t * n_cols..(t + 1) * n_cols];
            let mut a = vec![0.0f32; n_rows];
            gemv_iq4_xs_f32(n_cols, &w, xt, &mut a).unwrap();
            exp.extend_from_slice(&a);
            for r in 0..n_rows {
                let row = &w[r * IQ4_XS_BLOCK..(r + 1) * IQ4_XS_BLOCK];
                exp_oracle.push(oracle_iq4_xs_dot(row, xt));
            }
        }
        assert_close(&y, &exp);
        assert_close(&y, &exp_oracle);
        crate::pool::with_sequential(|| {
            let mut y_seq = vec![0.0f32; n_rows * n_tokens];
            gemm_iq4_xs_f32(n_cols, n_tokens, &w, &xk, &mut y_seq).unwrap();
            assert_close(&y_seq, &exp_oracle);
        });
    }

    #[test]
    fn gemm_rejects_zero_tokens_and_bad_x_len() {
        let mut y = [0.0f32; 2];
        assert!(gemm_f32(4, 0, &[0u8; 16], &[], &mut y).is_err());
        assert!(gemm_f32(4, 2, &[0u8; 16], &[1.0], &mut y).is_err());
    }
}
