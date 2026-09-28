//! Use the quantization kernels directly, without the decode loop.
//!
//! Demonstrates the workflow quantization research actually needs:
//!
//!   * locate a quantized tensor in a GGUF and get at its packed bytes
//!   * expand one row to f32 with `dequant_*_row`
//!   * run the fused `gemv_*_f32` straight over the packed bytes
//!   * cross-check the two against each other -- dequantize-then-dot must
//!     equal the fused kernel, which is exactly how this crate's kernel tests
//!     are written, and the first thing to check when adding a dtype
//!   * measure what a dtype costs, by re-quantizing a row to Q8_0 with
//!     `pack_q8_0_block` and comparing the round trip
//!
//! Runs with no download, against an in-memory fixture:
//!
//!     cargo run --release --example kernels
//!
//! Or against real weights, which is where the error numbers get interesting:
//!
//!     cargo run --release --example kernels -- model.gguf

use std::fs::File;
use std::io::Read;

use llama_rust::gguf::{load_gguf_owned, GgmlType, Gguf, Tensor};
use llama_rust::kernels::{
    dequant_bf16_row, dequant_f16_row, dequant_f32_row, dequant_q2_k_row, dequant_q3_k_row,
    dequant_q4_1_row, dequant_q4_k_row, dequant_q5_0_row, dequant_q5_1_row, dequant_q5_k_row,
    dequant_q6_k_row, dequant_q8_0_row, gemv_bf16, gemv_f16, gemv_f32, gemv_q2_k_f32,
    gemv_q3_k_f32, gemv_q4_1_f32, gemv_q4_k_f32, gemv_q5_0_f32, gemv_q5_1_f32, gemv_q5_k_f32,
    gemv_q6_k_f32, gemv_q8_0_f32, pack_q8_0_block, Q8_0_BLOCK, QK8_0,
};

/// Rows to run the GEMV over. Enough to be a real matrix-vector product,
/// small enough to print.
const ROWS: usize = 8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gguf = match std::env::args().nth(1) {
        Some(path) => {
            println!("# reading {path}\n");
            let mut file = File::open(&path)?;
            let mut bytes = Vec::new();
            let _len = file.read_to_end(&mut bytes)?;
            load_gguf_owned(bytes)?
        }
        None => {
            println!("# no model path given, using the in-memory tiny_llama fixture");
            println!("# (mixed F32 / Q4_K / Q6_K weights)\n");
            load_gguf_owned(llama_rust::fixtures::tiny_llama_gguf())?
        }
    };

    let Some(tensor) = widest_quantized_tensor(&gguf) else {
        return Err("no 2-D quantized tensor in this GGUF".into());
    };
    let n_cols = tensor.n_cols();
    let n_rows = tensor.n_rows().min(ROWS);

    let block_elems = tensor.ty.block_size();
    let block_bytes = tensor.ty.type_size();
    println!("tensor      {}", tensor.name);
    println!(
        "dtype       {:?} (ggml_type {})",
        tensor.ty,
        tensor.ty.to_i32()
    );
    println!("shape       {n_cols} cols x {} rows", tensor.n_rows());
    println!(
        "block       {block_elems} weights in {block_bytes} bytes ({:.3} bits/weight)",
        bits_per_weight(block_bytes, block_elems),
    );
    let row_bytes = packed_row_bytes(n_cols, block_elems, block_bytes)?;
    println!("row         {row_bytes} packed bytes for {n_cols} weights\n");

    // --- Dequantize row 0 ------------------------------------------------
    let first_row = tensor
        .data
        .get(..row_bytes)
        .ok_or("tensor payload shorter than one row")?;
    let mut weights = vec![0.0f32; n_cols];
    dequant_row(tensor.ty, n_cols, first_row, &mut weights)?;
    println!("row 0, first 8 weights: {:.5?}", head(&weights, 8));

    // --- Fused GEMV over the packed bytes vs dequantize-then-dot ---------
    let x: Vec<f32> = (0..n_cols)
        .map(|i| f32::from(u8::try_from(i % 11).unwrap_or(0)) / 10.0 - 0.5)
        .collect();

    let all_rows = tensor
        .data
        .get(..row_bytes.checked_mul(n_rows).ok_or("row bytes overflow")?)
        .ok_or("tensor payload shorter than the rows requested")?;
    let mut fused = vec![0.0f32; n_rows];
    gemv(tensor.ty, n_cols, all_rows, &x, &mut fused)?;

    let mut unfused = vec![0.0f32; n_rows];
    let mut scratch = vec![0.0f32; n_cols];
    for (r, out) in unfused.iter_mut().enumerate() {
        let start = r.saturating_mul(row_bytes);
        let row = all_rows
            .get(start..start.saturating_add(row_bytes))
            .ok_or("row slice out of range")?;
        dequant_row(tensor.ty, n_cols, row, &mut scratch)?;
        *out = scratch.iter().zip(&x).map(|(w, v)| w * v).sum();
    }

    println!("\ngemv over {n_rows} rows");
    println!("  fused (packed bytes) {:.5?}", head(&fused, 4));
    println!("  dequantize then dot  {:.5?}", head(&unfused, 4));
    let drift = max_abs_diff(&fused, &unfused);
    println!("  max absolute difference {drift:.3e}");

    // --- What does Q8_0 cost on this row? -------------------------------
    let (rms, worst) = q8_0_round_trip_error(&weights)?;
    println!("\nre-quantizing row 0 to Q8_0 (8.5 bits/weight)");
    println!("  rms error   {rms:.3e}");
    println!("  worst error {worst:.3e}");
    Ok(())
}

/// The 2-D tensor with the most weights, preferring a quantized dtype: the one
/// where kernel behaviour is most worth looking at.
fn widest_quantized_tensor(gguf: &Gguf) -> Option<Tensor<'_>> {
    gguf.tensors()
        .filter(|t| t.n_rows() > 1)
        .max_by_key(|t| (t.ty.block_size() > 1, t.n_cols().saturating_mul(t.n_rows())))
}

fn bits_per_weight(block_bytes: usize, block_elems: usize) -> f64 {
    let bytes = f64::from(u32::try_from(block_bytes).unwrap_or(0));
    let elems = f64::from(u32::try_from(block_elems).unwrap_or(1));
    if elems > 0.0 {
        bytes * 8.0 / elems
    } else {
        0.0
    }
}

fn packed_row_bytes(
    n_cols: usize,
    block_elems: usize,
    block_bytes: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    if block_elems == 0 || !n_cols.is_multiple_of(block_elems) {
        return Err(format!("{n_cols} columns is not a multiple of {block_elems}").into());
    }
    Ok((n_cols / block_elems).saturating_mul(block_bytes))
}

fn head(values: &[f32], n: usize) -> &[f32] {
    values.get(..n.min(values.len())).unwrap_or(values)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Quantize to Q8_0 and back, reporting rms and worst-case absolute error.
///
/// Q8_0 is one binary16 scale plus 32 signed bytes per block, and the scale is
/// the usual `max|x| / 127`.
fn q8_0_round_trip_error(weights: &[f32]) -> Result<(f64, f32), Box<dyn std::error::Error>> {
    let n = weights.len();
    if !n.is_multiple_of(QK8_0) {
        return Err(format!("{n} weights is not a multiple of {QK8_0}").into());
    }
    let mut packed = Vec::with_capacity((n / QK8_0).saturating_mul(Q8_0_BLOCK));
    for block in weights.chunks(QK8_0) {
        let peak = block.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let scale = if peak > 0.0 { peak / 127.0 } else { 0.0 };
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let mut qs = [0i8; QK8_0];
        for (q, v) in qs.iter_mut().zip(block) {
            *q = quantize_i8(v * inv);
        }
        packed.extend_from_slice(&pack_q8_0_block(scale, &qs));
    }
    let mut back = vec![0.0f32; n];
    dequant_q8_0_row(n, &packed, &mut back)?;

    let mut sum_sq = 0.0f64;
    let mut worst = 0.0f32;
    for (original, restored) in weights.iter().zip(&back) {
        let err = original - restored;
        sum_sq += f64::from(err) * f64::from(err);
        worst = worst.max(err.abs());
    }
    let count = f64::from(u32::try_from(n).unwrap_or(1)).max(1.0);
    Ok(((sum_sq / count).sqrt(), worst))
}

/// Round to the nearest `i8`, saturating rather than wrapping.
#[expect(
    clippy::cast_possible_truncation,
    reason = "float-to-int casts saturate since Rust 1.45: out-of-range clamps to -128/127 and NaN becomes 0, which is the wanted behaviour"
)]
fn quantize_i8(v: f32) -> i8 {
    v.round() as i8
}

/// Expand one packed row to f32.
///
/// The crate has no dtype-generic dequantizer, so callers dispatch. This covers
/// the dtypes a downloaded checkpoint usually contains; extend the match for
/// the I-quants, ternary, and micro-float types, which follow the same
/// `dequant_<t>_row(n_cols, row, y)` shape.
///
/// Q4_0 is absent because the crate has no `dequant_q4_0_row` -- see the
/// `gguf_inventory` example for why that is worth knowing.
fn dequant_row(
    ty: GgmlType,
    n_cols: usize,
    row: &[u8],
    out: &mut [f32],
) -> Result<(), Box<dyn std::error::Error>> {
    match ty {
        GgmlType::F32 => dequant_f32_row(n_cols, row, out)?,
        GgmlType::F16 => dequant_f16_row(n_cols, row, out)?,
        GgmlType::BF16 => dequant_bf16_row(n_cols, row, out)?,
        GgmlType::Q4_1 => dequant_q4_1_row(n_cols, row, out)?,
        GgmlType::Q5_0 => dequant_q5_0_row(n_cols, row, out)?,
        GgmlType::Q5_1 => dequant_q5_1_row(n_cols, row, out)?,
        GgmlType::Q8_0 => dequant_q8_0_row(n_cols, row, out)?,
        GgmlType::Q2_K => dequant_q2_k_row(n_cols, row, out)?,
        GgmlType::Q3_K => dequant_q3_k_row(n_cols, row, out)?,
        GgmlType::Q4_K => dequant_q4_k_row(n_cols, row, out)?,
        GgmlType::Q5_K => dequant_q5_k_row(n_cols, row, out)?,
        GgmlType::Q6_K => dequant_q6_k_row(n_cols, row, out)?,
        other => return Err(format!("extend this match for {other:?}").into()),
    }
    Ok(())
}

/// `y[m] = W[m, ..] . x` straight over the packed bytes. Same dispatch note as
/// [`dequant_row`].
fn gemv(
    ty: GgmlType,
    n_cols: usize,
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
) -> Result<(), Box<dyn std::error::Error>> {
    match ty {
        GgmlType::F32 => gemv_f32(n_cols, w, x, y)?,
        GgmlType::F16 => gemv_f16(n_cols, w, x, y)?,
        GgmlType::BF16 => gemv_bf16(n_cols, w, x, y)?,
        GgmlType::Q4_1 => gemv_q4_1_f32(n_cols, w, x, y)?,
        GgmlType::Q5_0 => gemv_q5_0_f32(n_cols, w, x, y)?,
        GgmlType::Q5_1 => gemv_q5_1_f32(n_cols, w, x, y)?,
        GgmlType::Q8_0 => gemv_q8_0_f32(n_cols, w, x, y)?,
        GgmlType::Q2_K => gemv_q2_k_f32(n_cols, w, x, y)?,
        GgmlType::Q3_K => gemv_q3_k_f32(n_cols, w, x, y)?,
        GgmlType::Q4_K => gemv_q4_k_f32(n_cols, w, x, y)?,
        GgmlType::Q5_K => gemv_q5_k_f32(n_cols, w, x, y)?,
        GgmlType::Q6_K => gemv_q6_k_f32(n_cols, w, x, y)?,
        other => return Err(format!("extend this match for {other:?}").into()),
    }
    Ok(())
}
