//! GGUF v3 little-endian write/load. Pure functions over bytes.

use std::collections::HashMap;

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";
pub const GGUF_VERSION: u32 = 3;
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;

/// ggml_type values used in tensor info.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum GgmlType {
    Q4_0 = 2,
    Q8_0 = 8,
}

impl GgmlType {
    fn from_i32(v: i32) -> Result<Self, GgufError> {
        match v {
            2 => Ok(Self::Q4_0),
            8 => Ok(Self::Q8_0),
            other => Err(GgufError::UnsupportedType(other)),
        }
    }
}

const GGUF_TYPE_UINT32: i32 = 4;
const GGUF_TYPE_STRING: i32 = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufError {
    Magic,
    Version(u32),
    Truncated,
    Utf8,
    UnsupportedType(i32),
    UnsupportedKv(i32),
    Shape,
}

#[derive(Clone, Debug)]
pub struct TensorWrite {
    pub name: String,
    pub ty: GgmlType,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub ty: GgmlType,
    pub shape: Vec<u64>,
    /// GGUF tensor payload, same bytes as on disk.
    pub data: Vec<u8>,
}

impl Tensor {
    pub fn n_cols(&self) -> usize {
        self.shape.first().copied().unwrap_or(0) as usize
    }

    pub fn n_rows(&self) -> usize {
        if self.shape.len() < 2 {
            1
        } else {
            self.shape[1] as usize
        }
    }
}

#[derive(Clone, Debug)]
pub struct Gguf {
    pub alignment: usize,
    pub kv: HashMap<String, Kv>,
    pub tensors: Vec<Tensor>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Kv {
    U32(u32),
    String(String),
}

impl Gguf {
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

pub fn write_gguf(tensors: &[TensorWrite]) -> Vec<u8> {
    let alignment = GGUF_DEFAULT_ALIGNMENT;
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut off = 0usize;
    for t in tensors {
        off = align_up(off, alignment);
        offsets.push(off as u64);
        off += t.data.len();
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(GGUF_MAGIC);
    put_u32(&mut buf, GGUF_VERSION);
    put_i64(&mut buf, tensors.len() as i64);
    put_i64(&mut buf, 2);

    put_string(&mut buf, "general.alignment");
    put_i32(&mut buf, GGUF_TYPE_UINT32);
    put_u32(&mut buf, alignment as u32);

    put_string(&mut buf, "general.name");
    put_i32(&mut buf, GGUF_TYPE_STRING);
    put_string(&mut buf, "llama-rust");

    for (t, offset) in tensors.iter().zip(offsets.iter()) {
        put_string(&mut buf, &t.name);
        put_u32(&mut buf, t.shape.len() as u32);
        for &d in &t.shape {
            put_i64(&mut buf, d as i64);
        }
        put_i32(&mut buf, t.ty as i32);
        put_u64(&mut buf, *offset);
    }

    let data_start = align_up(buf.len(), alignment);
    buf.resize(data_start, 0);
    let mut cursor = 0usize;
    for t in tensors {
        let aligned = align_up(cursor, alignment);
        buf.resize(data_start + aligned, 0);
        buf.extend_from_slice(&t.data);
        cursor = aligned + t.data.len();
    }
    buf
}

pub fn load_gguf(bytes: &[u8]) -> Result<Gguf, GgufError> {
    let mut pos = 0usize;
    let magic = read_exact(bytes, &mut pos, 4)?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::Magic);
    }
    let version = read_u32(bytes, &mut pos)?;
    if version != GGUF_VERSION {
        return Err(GgufError::Version(version));
    }
    let n_tensors = read_i64(bytes, &mut pos)?;
    let n_kv = read_i64(bytes, &mut pos)?;
    if n_tensors < 0 || n_kv < 0 {
        return Err(GgufError::Truncated);
    }

    let mut kv = HashMap::new();
    for _ in 0..n_kv {
        let key = read_string(bytes, &mut pos)?;
        let ty = read_i32(bytes, &mut pos)?;
        let val = read_kv_value(bytes, &mut pos, ty)?;
        kv.insert(key, val);
    }

    let alignment = match kv.get("general.alignment") {
        Some(Kv::U32(v)) if *v > 0 => *v as usize,
        _ => GGUF_DEFAULT_ALIGNMENT,
    };

    let mut infos = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = read_string(bytes, &mut pos)?;
        let n_dims = read_u32(bytes, &mut pos)? as usize;
        if n_dims == 0 || n_dims > 4 {
            return Err(GgufError::Shape);
        }
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            let d = read_i64(bytes, &mut pos)?;
            if d <= 0 {
                return Err(GgufError::Shape);
            }
            shape.push(d as u64);
        }
        let ty = GgmlType::from_i32(read_i32(bytes, &mut pos)?)?;
        let offset = read_u64(bytes, &mut pos)?;
        infos.push((name, shape, ty, offset));
    }

    let data_start = align_up(pos, alignment);
    let mut tensors = Vec::with_capacity(infos.len());
    for (name, shape, ty, offset) in infos {
        let n_el = shape.iter().try_fold(1u64, |a, &b| a.checked_mul(b));
        let n_el = n_el.ok_or(GgufError::Shape)? as usize;
        let block = match ty {
            GgmlType::Q8_0 => super::quant::Q8_0_BLOCK,
            GgmlType::Q4_0 => super::quant::Q4_0_BLOCK,
        };
        let k = super::quant::QK8_0;
        if !n_el.is_multiple_of(k) {
            return Err(GgufError::Shape);
        }
        let nbytes = (n_el / k) * block;
        let start = data_start
            .checked_add(offset as usize)
            .ok_or(GgufError::Truncated)?;
        let end = start.checked_add(nbytes).ok_or(GgufError::Truncated)?;
        if end > bytes.len() {
            return Err(GgufError::Truncated);
        }
        tensors.push(Tensor {
            name,
            ty,
            shape,
            data: bytes[start..end].to_vec(),
        });
    }

    Ok(Gguf {
        alignment,
        kv,
        tensors,
    })
}

fn align_up(n: usize, a: usize) -> usize {
    n.div_ceil(a) * a
}

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_i32(buf: &mut Vec<u8>, v: i32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_i64(buf: &mut Vec<u8>, v: i64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_string(buf: &mut Vec<u8>, s: &str) {
    put_u64(buf, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}

fn read_exact<'a>(bytes: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8], GgufError> {
    let end = pos.checked_add(n).ok_or(GgufError::Truncated)?;
    if end > bytes.len() {
        return Err(GgufError::Truncated);
    }
    let s = &bytes[*pos..end];
    *pos = end;
    Ok(s)
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> Result<u32, GgufError> {
    let s = read_exact(bytes, pos, 4)?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}
fn read_u64(bytes: &[u8], pos: &mut usize) -> Result<u64, GgufError> {
    let s = read_exact(bytes, pos, 8)?;
    Ok(u64::from_le_bytes([
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
    ]))
}
fn read_i32(bytes: &[u8], pos: &mut usize) -> Result<i32, GgufError> {
    Ok(read_u32(bytes, pos)? as i32)
}
fn read_i64(bytes: &[u8], pos: &mut usize) -> Result<i64, GgufError> {
    Ok(read_u64(bytes, pos)? as i64)
}

fn read_string(bytes: &[u8], pos: &mut usize) -> Result<String, GgufError> {
    let len = read_u64(bytes, pos)? as usize;
    let s = read_exact(bytes, pos, len)?;
    String::from_utf8(s.to_vec()).map_err(|_| GgufError::Utf8)
}

fn read_kv_value(bytes: &[u8], pos: &mut usize, ty: i32) -> Result<Kv, GgufError> {
    match ty {
        GGUF_TYPE_UINT32 => Ok(Kv::U32(read_u32(bytes, pos)?)),
        GGUF_TYPE_STRING => Ok(Kv::String(read_string(bytes, pos)?)),
        other => Err(GgufError::UnsupportedKv(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp16::load_f16_le;
    use crate::quant::{
        gemv_q4_0, gemv_q8_0, pack_q4_0_from_i4, pack_q8_0_block, Q4_0_BLOCK, Q8_0_BLOCK, QK4_0,
        QK8_0,
    };

    fn independent_q8_dot(w: &[u8], x: &[u8]) -> f32 {
        assert_eq!(w.len() % Q8_0_BLOCK, 0);
        assert_eq!(w.len(), x.len());
        let mut sum = 0.0f32;
        let mut off = 0;
        while off < w.len() {
            let wb = &w[off..off + Q8_0_BLOCK];
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
        sum
    }

    /// ggml `dequantize_row_q4_0`: lo -> y[j], hi -> y[j+16], then f32 dot.
    fn dequant_q4_0_row(w: &[u8]) -> Vec<f32> {
        let nblocks = w.len() / Q4_0_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK4_0];
        for b in 0..nblocks {
            let wb = &w[b * Q4_0_BLOCK..(b + 1) * Q4_0_BLOCK];
            let d = load_f16_le(wb);
            for j in 0..(QK4_0 / 2) {
                let packed = wb[2 + j];
                let lo = i32::from(packed & 0x0f) - 8;
                let hi = i32::from(packed >> 4) - 8;
                y[b * QK4_0 + j] = lo as f32 * d;
                y[b * QK4_0 + j + 16] = hi as f32 * d;
            }
        }
        y
    }

    fn dequant_q8_0_row(x: &[u8]) -> Vec<f32> {
        let nblocks = x.len() / Q8_0_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK8_0];
        for b in 0..nblocks {
            let xb = &x[b * Q8_0_BLOCK..(b + 1) * Q8_0_BLOCK];
            let d = load_f16_le(xb);
            for i in 0..QK8_0 {
                y[b * QK8_0 + i] = (xb[2 + i] as i8) as f32 * d;
            }
        }
        y
    }

    fn independent_q4_dot(w: &[u8], x: &[u8]) -> f32 {
        let wy = dequant_q4_0_row(w);
        let xy = dequant_q8_0_row(x);
        assert_eq!(wy.len(), xy.len());
        wy.iter().zip(xy.iter()).map(|(a, b)| a * b).sum()
    }

    #[test]
    fn write_load_gemv_q8_and_q4_match_file_bytes() {
        let n_cols = 64usize;
        let n_rows = 5usize;
        let mut w8 = Vec::new();
        let mut x8 = Vec::new();
        let mut w4 = Vec::new();
        for r in 0..n_rows {
            for _b in 0..(n_cols / QK8_0) {
                let mut qs = [0i8; QK8_0];
                for (i, q) in qs.iter_mut().enumerate() {
                    *q = (r as i8).wrapping_add(i as i8).wrapping_sub(16);
                }
                w8.extend_from_slice(&pack_q8_0_block(0.05 + r as f32 * 0.01, &qs));
            }
        }
        for b in 0..(n_cols / QK8_0) {
            let mut qs = [0i8; QK8_0];
            for (i, q) in qs.iter_mut().enumerate() {
                *q = (b as i8).wrapping_add(i as i8).wrapping_sub(8);
            }
            x8.extend_from_slice(&pack_q8_0_block(0.02, &qs));
        }
        for r in 0..n_rows {
            for b in 0..(n_cols / QK4_0) {
                let mut v = [0i8; QK4_0];
                for (i, q) in v.iter_mut().enumerate() {
                    *q = ((r * 3 + b + i) % 15) as i8 - 7;
                }
                w4.extend_from_slice(&pack_q4_0_from_i4(0.07 + r as f32 * 0.01, &v));
            }
        }

        let bytes = write_gguf(&[
            TensorWrite {
                name: "w_q8".into(),
                ty: GgmlType::Q8_0,
                shape: vec![n_cols as u64, n_rows as u64],
                data: w8.clone(),
            },
            TensorWrite {
                name: "x_q8".into(),
                ty: GgmlType::Q8_0,
                shape: vec![n_cols as u64],
                data: x8.clone(),
            },
            TensorWrite {
                name: "w_q4".into(),
                ty: GgmlType::Q4_0,
                shape: vec![n_cols as u64, n_rows as u64],
                data: w4.clone(),
            },
        ]);

        let g = load_gguf(&bytes).expect("load");
        let tw8 = g.tensor("w_q8").expect("w_q8");
        let tx = g.tensor("x_q8").expect("x_q8");
        let tw4 = g.tensor("w_q4").expect("w_q4");
        assert_eq!(tw8.data, w8);
        assert_eq!(tx.data, x8);
        assert_eq!(tw4.data, w4);
        assert_eq!(tw8.ty, GgmlType::Q8_0);
        assert_eq!(tw4.ty, GgmlType::Q4_0);

        let mut y8 = vec![0.0f32; n_rows];
        gemv_q8_0(n_cols, &tw8.data, &tx.data, &mut y8);
        let rb8 = (n_cols / QK8_0) * Q8_0_BLOCK;
        for (r, yv) in y8.iter().enumerate() {
            let expected = independent_q8_dot(&tw8.data[r * rb8..(r + 1) * rb8], &tx.data);
            let rel = (yv - expected).abs() / (1.0 + expected.abs());
            assert!(rel < 1e-5, "q8 row {r}: {yv} vs {expected}");
        }

        let mut y4 = vec![0.0f32; n_rows];
        gemv_q4_0(n_cols, &tw4.data, &tx.data, &mut y4);
        let rb4 = (n_cols / QK4_0) * Q4_0_BLOCK;
        for (r, yv) in y4.iter().enumerate() {
            let expected = independent_q4_dot(&tw4.data[r * rb4..(r + 1) * rb4], &tx.data);
            let rel = (yv - expected).abs() / (1.0 + expected.abs());
            assert!(rel < 1e-5, "q4 row {r}: {yv} vs {expected}");
        }
    }

    /// If GEMV interleaved nibbles with x[2j]/x[2j+1], this is 3*dw*dx not 11*dw*dx.
    #[test]
    fn q4_0_lo_hi_map_to_elem_j_and_j16() {
        let mut v = [0i8; QK4_0];
        v[0] = 1;
        v[16] = 2;
        let w = pack_q4_0_from_i4(1.0, &v);
        let mut xq = [0i8; QK8_0];
        xq[0] = 3;
        xq[16] = 4;
        let x = pack_q8_0_block(1.0, &xq);
        let bytes = write_gguf(&[
            TensorWrite {
                name: "w_q4".into(),
                ty: GgmlType::Q4_0,
                shape: vec![32, 1],
                data: w.to_vec(),
            },
            TensorWrite {
                name: "x_q8".into(),
                ty: GgmlType::Q8_0,
                shape: vec![32],
                data: x.to_vec(),
            },
        ]);
        let g = load_gguf(&bytes).expect("load");
        let mut y = [0.0f32];
        gemv_q4_0(
            32,
            &g.tensor("w_q4").unwrap().data,
            &g.tensor("x_q8").unwrap().data,
            &mut y,
        );
        let expected = independent_q4_dot(&w, &x);
        // 1*3 + 2*4 = 11 at scale 1 (fp16 1.0 is exact).
        assert!((expected - 11.0).abs() < 1e-5, "oracle {expected}");
        assert!((y[0] - 11.0).abs() < 1e-5, "gemv {}", y[0]);
    }
}
