//! GGUF on-disk Q4_0 / Q8_0 blocks. GEMV reads those bytes; no f32-scale copy.

use rayon::prelude::*;

use crate::fp16::{load_f16_le, store_f16_le};
use crate::pool::install;

pub const QK8_0: usize = 32;
pub const QK4_0: usize = 32;
/// ggml `block_q8_0`: `ggml_half d` + `int8 qs[32]`.
pub const Q8_0_BLOCK: usize = 2 + QK8_0;
/// ggml `block_q4_0`: `ggml_half d` + `uint8 qs[16]`.
pub const Q4_0_BLOCK: usize = 2 + QK4_0 / 2;

pub fn q8_0_row_bytes(n_cols: usize) -> usize {
    assert!(
        n_cols.is_multiple_of(QK8_0),
        "n_cols must be a multiple of {QK8_0}"
    );
    (n_cols / QK8_0) * Q8_0_BLOCK
}

pub fn q4_0_row_bytes(n_cols: usize) -> usize {
    assert!(
        n_cols.is_multiple_of(QK4_0),
        "n_cols must be a multiple of {QK4_0}"
    );
    (n_cols / QK4_0) * Q4_0_BLOCK
}

pub fn pack_q8_0_block(scale: f32, qs: &[i8; QK8_0]) -> [u8; Q8_0_BLOCK] {
    let mut out = [0u8; Q8_0_BLOCK];
    let d = store_f16_le(scale);
    out[0] = d[0];
    out[1] = d[1];
    for i in 0..QK8_0 {
        out[2 + i] = qs[i] as u8;
    }
    out
}

pub fn pack_q4_0_block(scale: f32, qs: &[u8; QK4_0 / 2]) -> [u8; Q4_0_BLOCK] {
    let mut out = [0u8; Q4_0_BLOCK];
    let d = store_f16_le(scale);
    out[0] = d[0];
    out[1] = d[1];
    out[2..].copy_from_slice(qs);
    out
}

/// y[m] = W[m, n_cols] x[n_cols], W and x as GGUF Q8_0 block streams.
pub fn gemv_q8_0(n_cols: usize, w: &[u8], x: &[u8], y: &mut [f32]) {
    let rb = q8_0_row_bytes(n_cols);
    assert_eq!(x.len(), rb, "x Q8_0 bytes");
    assert_eq!(w.len(), rb * y.len(), "W Q8_0 bytes");
    if y.is_empty() {
        return;
    }
    install(|| {
        y.par_iter_mut().enumerate().for_each(|(r, out)| {
            *out = vec_dot_q8_row(&w[r * rb..(r + 1) * rb], x);
        });
    });
}

/// y[m] = W_q4[m, n_cols] x_q8[n_cols].
pub fn gemv_q4_0(n_cols: usize, w: &[u8], x: &[u8], y: &mut [f32]) {
    let w_rb = q4_0_row_bytes(n_cols);
    let x_rb = q8_0_row_bytes(n_cols);
    assert_eq!(x.len(), x_rb, "x Q8_0 bytes");
    assert_eq!(w.len(), w_rb * y.len(), "W Q4_0 bytes");
    if y.is_empty() {
        return;
    }
    install(|| {
        y.par_iter_mut().enumerate().for_each(|(r, out)| {
            *out = vec_dot_q4_row(&w[r * w_rb..(r + 1) * w_rb], x);
        });
    });
}

fn vec_dot_q8_row(row: &[u8], x: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    let mut off = 0usize;
    while off + Q8_0_BLOCK <= row.len() {
        let wb = &row[off..off + Q8_0_BLOCK];
        let xb = &x[off..off + Q8_0_BLOCK];
        let dw = load_f16_le(wb);
        let dx = load_f16_le(xb);
        let mut acc = 0i32;
        for i in 0..QK8_0 {
            acc += (wb[2 + i] as i8) as i32 * (xb[2 + i] as i8) as i32;
        }
        sum += acc as f32 * (dw * dx);
        off += Q8_0_BLOCK;
    }
    debug_assert_eq!(off, row.len());
    sum
}

fn vec_dot_q4_row(row: &[u8], x: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    let nblocks = row.len() / Q4_0_BLOCK;
    for b in 0..nblocks {
        let wb = &row[b * Q4_0_BLOCK..(b + 1) * Q4_0_BLOCK];
        let xb = &x[b * Q8_0_BLOCK..(b + 1) * Q8_0_BLOCK];
        let dw = load_f16_le(wb);
        let dx = load_f16_le(xb);
        let mut acc = 0i32;
        for i in 0..(QK4_0 / 2) {
            let packed = wb[2 + i];
            let lo = i32::from(packed & 0x0f) - 8;
            let hi = i32::from(packed >> 4) - 8;
            acc += lo * (xb[2 + 2 * i] as i8) as i32;
            acc += hi * (xb[2 + 2 * i + 1] as i8) as i32;
        }
        sum += acc as f32 * (dw * dx);
    }
    sum
}
