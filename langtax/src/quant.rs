//! GGUF on-disk Q4_0 / Q8_0 / Q4_K / Q8_K blocks. GEMV reads those bytes; no f32-scale copy.

use std::fmt;

use rayon::prelude::*;

use crate::fp16::{load_f16_le, store_f16_le};
use crate::pool::install;

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
/// ggml `block_q8_K`: f32 `d`, `int8 qs[256]`, `int16 bsums[16]`.
pub const Q8_K_BLOCK: usize = 4 + QK_K + QK_K / 16 * 2;

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
    install(|| {
        y.par_iter_mut().enumerate().for_each(|(r, out)| {
            *out = match r.checked_mul(rb) {
                Some(start) => w
                    .get(start..)
                    .and_then(|s| s.get(..rb))
                    .map(|row| vec_dot_q8_row(row, x))
                    .unwrap_or(0.0),
                None => 0.0,
            };
        });
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
    install(|| {
        y.par_iter_mut().enumerate().for_each(|(r, out)| {
            *out = match r.checked_mul(w_rb) {
                Some(start) => w
                    .get(start..)
                    .and_then(|s| s.get(..w_rb))
                    .map(|row| vec_dot_q4_row(row, x))
                    .unwrap_or(0.0),
                None => 0.0,
            };
        });
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
    install(|| {
        y.par_iter_mut().enumerate().for_each(|(r, out)| {
            *out = match r.checked_mul(w_rb) {
                Some(start) => w
                    .get(start..)
                    .and_then(|s| s.get(..w_rb))
                    .map(|row| vec_dot_q4_k_row(row, x))
                    .unwrap_or(0.0),
                None => 0.0,
            };
        });
    });
    Ok(())
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
        sum += (acc as f32) * (dw * dx);
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
}
