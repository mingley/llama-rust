//! GGUF v3 little-endian write/load. Pure functions over bytes.

use std::collections::HashMap;
use std::fmt;

use crate::quant::{
    F16_SIZE, F32_SIZE, Q4_0_BLOCK, Q4_K_BLOCK, Q6_K_BLOCK, Q8_0_BLOCK, Q8_K_BLOCK, QK4_0, QK8_0,
    QK_K,
};

/// GGUF magic `GGUF`.
pub(crate) const GGUF_MAGIC: &[u8; 4] = b"GGUF";
/// GGUF container version written and accepted by this crate.
pub(crate) const GGUF_VERSION: u32 = 3;
/// Default tensor-data alignment when `general.alignment` is absent.
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;

/// ggml_type values used in tensor info.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum GgmlType {
    /// `GGML_TYPE_F32`.
    F32 = 0,
    /// `GGML_TYPE_F16`.
    F16 = 1,
    /// `GGML_TYPE_Q4_0`.
    Q4_0 = 2,
    /// `GGML_TYPE_Q8_0`.
    Q8_0 = 8,
    /// `GGML_TYPE_Q4_K`.
    #[expect(non_camel_case_types, reason = "matches ggml GGML_TYPE_Q4_K")]
    Q4_K = 12,
    /// `GGML_TYPE_Q6_K`.
    #[expect(non_camel_case_types, reason = "matches ggml GGML_TYPE_Q6_K")]
    Q6_K = 14,
    /// `GGML_TYPE_Q8_K`.
    #[expect(non_camel_case_types, reason = "matches ggml GGML_TYPE_Q8_K")]
    Q8_K = 15,
}

impl GgmlType {
    fn from_i32(v: i32) -> Result<Self, GgufError> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            8 => Ok(Self::Q8_0),
            12 => Ok(Self::Q4_K),
            14 => Ok(Self::Q6_K),
            15 => Ok(Self::Q8_K),
            other => Err(GgufError::UnsupportedType(other)),
        }
    }

    /// GGUF `ggml_type` integer.
    pub const fn to_i32(self) -> i32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q8_0 => 8,
            Self::Q4_K => 12,
            Self::Q6_K => 14,
            Self::Q8_K => 15,
        }
    }

    fn layout(self) -> (usize, usize) {
        match self {
            Self::F32 => (F32_SIZE, 1),
            Self::F16 => (F16_SIZE, 1),
            Self::Q4_0 => (Q4_0_BLOCK, QK4_0),
            Self::Q8_0 => (Q8_0_BLOCK, QK8_0),
            Self::Q4_K => (Q4_K_BLOCK, QK_K),
            Self::Q6_K => (Q6_K_BLOCK, QK_K),
            Self::Q8_K => (Q8_K_BLOCK, QK_K),
        }
    }
}

const GGUF_TYPE_UINT8: i32 = 0;
const GGUF_TYPE_INT8: i32 = 1;
const GGUF_TYPE_UINT16: i32 = 2;
const GGUF_TYPE_INT16: i32 = 3;
const GGUF_TYPE_UINT32: i32 = 4;
const GGUF_TYPE_INT32: i32 = 5;
const GGUF_TYPE_FLOAT32: i32 = 6;
const GGUF_TYPE_BOOL: i32 = 7;
const GGUF_TYPE_STRING: i32 = 8;
const GGUF_TYPE_ARRAY: i32 = 9;
const GGUF_TYPE_UINT64: i32 = 10;
const GGUF_TYPE_INT64: i32 = 11;
const GGUF_TYPE_FLOAT64: i32 = 12;

/// Failure while parsing a GGUF blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufError {
    /// Bytes do not start with `GGUF`.
    Magic,
    /// `version` is not 3.
    Version(u32),
    /// A read ran past the end of the buffer, or an offset overflowed.
    Truncated,
    /// A GGUF string was not valid UTF-8.
    Utf8,
    /// Tensor `ggml_type` is not F32, F16, Q4_0, Q8_0, Q4_K, Q6_K, or Q8_K.
    UnsupportedType(i32),
    /// KV type is not a GGUF v3 value type.
    UnsupportedKv(i32),
    /// Rank, extent, or element count is unusable.
    Shape,
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Magic => write!(f, "not a GGUF file"),
            Self::Version(v) => write!(f, "unsupported GGUF version {v}"),
            Self::Truncated => write!(f, "truncated GGUF"),
            Self::Utf8 => write!(f, "non-utf8 GGUF string"),
            Self::UnsupportedType(t) => write!(f, "unsupported ggml type {t}"),
            Self::UnsupportedKv(t) => write!(f, "unsupported GGUF kv type {t}"),
            Self::Shape => write!(f, "invalid GGUF tensor shape"),
        }
    }
}

impl std::error::Error for GgufError {}

/// Tensor to serialize into a GGUF.
#[derive(Clone, Debug)]
pub struct TensorWrite {
    /// Tensor name in the GGUF name table.
    pub name: String,
    /// ggml type tag.
    pub ty: GgmlType,
    /// Dimension sizes, GGUF order (innermost first).
    pub shape: Vec<u64>,
    /// Packed GGUF payload bytes.
    pub data: Vec<u8>,
}

/// Loaded tensor view. `data` is a subslice of the GGUF's single file blob.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tensor<'a> {
    /// Tensor name in the GGUF name table.
    pub name: &'a str,
    /// ggml type tag.
    pub ty: GgmlType,
    /// Dimension sizes, GGUF order (innermost first).
    pub shape: &'a [u64],
    /// GGUF tensor payload, same bytes as on disk (range of [`Gguf::blob`]).
    pub data: &'a [u8],
    start: usize,
    end: usize,
}

impl Tensor<'_> {
    /// Innermost dimension (columns for a 2-D weight).
    pub fn n_cols(&self) -> usize {
        self.shape
            .first()
            .copied()
            .and_then(|d| usize::try_from(d).ok())
            .unwrap_or(0)
    }

    /// Second dimension, or 1 for a vector.
    pub fn n_rows(&self) -> usize {
        match self.shape.get(1) {
            Some(&d) => usize::try_from(d).unwrap_or(1),
            None => 1,
        }
    }

    /// Inclusive-exclusive byte range of [`Self::data`] inside [`Gguf::blob`].
    pub fn blob_range(&self) -> (usize, usize) {
        (self.start, self.end)
    }
}

/// On-disk tensor location inside [`Gguf::blob`].
#[derive(Clone, Debug)]
struct TensorInfo {
    name: String,
    ty: GgmlType,
    shape: Vec<u64>,
    start: usize,
    end: usize,
}

/// Parsed GGUF v3 file: one owned file blob, KV, tensor ranges.
#[derive(Clone, Debug)]
pub struct Gguf {
    /// Entire file. Tensor payloads are ranges of this allocation.
    blob: Vec<u8>,
    /// Tensor data alignment in bytes.
    pub(crate) alignment: usize,
    /// Key-value metadata from the header.
    pub(crate) kv: HashMap<String, Kv>,
    /// Tensor metadata in file order.
    infos: Vec<TensorInfo>,
}

/// GGUF metadata value (`gguf_type`).
#[derive(Clone, Debug, PartialEq)]
pub enum Kv {
    /// `GGUF_TYPE_UINT8`.
    U8(u8),
    /// `GGUF_TYPE_INT8`.
    I8(i8),
    /// `GGUF_TYPE_UINT16`.
    U16(u16),
    /// `GGUF_TYPE_INT16`.
    I16(i16),
    /// `GGUF_TYPE_UINT32`.
    U32(u32),
    /// `GGUF_TYPE_INT32`.
    I32(i32),
    /// `GGUF_TYPE_FLOAT32`.
    F32(f32),
    /// `GGUF_TYPE_BOOL` (stored as `i8`).
    Bool(bool),
    /// `GGUF_TYPE_STRING`.
    String(String),
    /// `GGUF_TYPE_ARRAY`. `elem` is the inner `gguf_type`; items may themselves be arrays.
    Array {
        /// Element `gguf_type` (may be `ARRAY` for a nested header).
        elem: i32,
        /// Array elements, each matching `elem`.
        items: Vec<Kv>,
    },
    /// `GGUF_TYPE_UINT64`.
    U64(u64),
    /// `GGUF_TYPE_INT64`.
    I64(i64),
    /// `GGUF_TYPE_FLOAT64`.
    F64(f64),
}

impl Kv {
    fn tag(&self) -> i32 {
        match self {
            Self::U8(_) => GGUF_TYPE_UINT8,
            Self::I8(_) => GGUF_TYPE_INT8,
            Self::U16(_) => GGUF_TYPE_UINT16,
            Self::I16(_) => GGUF_TYPE_INT16,
            Self::U32(_) => GGUF_TYPE_UINT32,
            Self::I32(_) => GGUF_TYPE_INT32,
            Self::F32(_) => GGUF_TYPE_FLOAT32,
            Self::Bool(_) => GGUF_TYPE_BOOL,
            Self::String(_) => GGUF_TYPE_STRING,
            Self::Array { .. } => GGUF_TYPE_ARRAY,
            Self::U64(_) => GGUF_TYPE_UINT64,
            Self::I64(_) => GGUF_TYPE_INT64,
            Self::F64(_) => GGUF_TYPE_FLOAT64,
        }
    }
}

impl Gguf {
    fn view(&self, index: usize) -> Option<Tensor<'_>> {
        let info = self.infos.get(index)?;
        let data = self.blob.get(info.start..info.end)?;
        Some(Tensor {
            name: info.name.as_str(),
            ty: info.ty,
            shape: info.shape.as_slice(),
            data,
            start: info.start,
            end: info.end,
        })
    }

    /// First tensor named `name`, if present. Payload is a subslice of [`Self::blob`].
    pub fn tensor(&self, name: &str) -> Option<Tensor<'_>> {
        self.infos
            .iter()
            .position(|t| t.name == name)
            .and_then(|i| self.view(i))
    }

    /// Tensors in file order. Each payload is a range of [`Self::blob`].
    pub fn tensors(&self) -> impl Iterator<Item = Tensor<'_>> + '_ {
        (0..self.infos.len()).filter_map(|i| self.view(i))
    }

    /// Entire file bytes this GGUF was parsed from.
    pub fn blob(&self) -> &[u8] {
        &self.blob
    }

    /// Byte length of the single owned file blob.
    pub fn blob_len(&self) -> usize {
        self.blob.len()
    }

    /// Consume the GGUF and return the file blob. Weight loaders take this once.
    pub fn into_blob(self) -> Vec<u8> {
        self.blob
    }

    /// True when `t.data` is a subslice of this GGUF's blob (no private copy).
    pub fn payload_in_blob(&self, t: Tensor<'_>) -> bool {
        slice_in_slice(&self.blob, t.data)
    }

    /// Tensor-data alignment in bytes.
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// `uint32` metadata value, if present.
    pub fn kv_u32(&self, key: &str) -> Option<u32> {
        match self.kv.get(key) {
            Some(Kv::U32(v)) => Some(*v),
            _ => None,
        }
    }

    /// String metadata value, if present.
    pub fn kv_string(&self, key: &str) -> Option<&str> {
        match self.kv.get(key) {
            Some(Kv::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Metadata value for `key`, if present.
    pub fn kv(&self, key: &str) -> Option<&Kv> {
        self.kv.get(key)
    }

    /// `float32` metadata value, if present.
    pub fn kv_f32(&self, key: &str) -> Option<f32> {
        match self.kv.get(key) {
            Some(Kv::F32(v)) => Some(*v),
            _ => None,
        }
    }

    /// `bool` metadata value, if present.
    pub fn kv_bool(&self, key: &str) -> Option<bool> {
        match self.kv.get(key) {
            Some(Kv::Bool(v)) => Some(*v),
            _ => None,
        }
    }
}

/// Serialize `tensors` as a GGUF v3 blob with `general.alignment` and `general.name`.
pub fn write_gguf(tensors: &[TensorWrite]) -> Vec<u8> {
    write_gguf_with_kv(
        &[
            ("general.alignment".into(), Kv::U32(32)),
            ("general.name".into(), Kv::String("llama-rust".into())),
        ],
        tensors,
    )
}

/// Serialize `kv` then `tensors` as a GGUF v3 blob.
pub fn write_gguf_with_kv(kv: &[(String, Kv)], tensors: &[TensorWrite]) -> Vec<u8> {
    write_gguf_kv_tensors(kv, tensors, None)
}

/// Like [`write_gguf_with_kv`], but each tensor’s ggml_type integer is taken
/// from `type_ids` (same length as `tensors`). Used to emit unsupported types.
#[cfg(test)]
pub(crate) fn write_gguf_with_type_ids(
    kv: &[(String, Kv)],
    tensors: &[TensorWrite],
    type_ids: &[i32],
) -> Vec<u8> {
    write_gguf_kv_tensors(kv, tensors, Some(type_ids))
}

fn write_gguf_kv_tensors(
    kv: &[(String, Kv)],
    tensors: &[TensorWrite],
    type_ids: Option<&[i32]>,
) -> Vec<u8> {
    let alignment = GGUF_DEFAULT_ALIGNMENT;
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut off = 0usize;
    for t in tensors {
        off = align_up(off, alignment);
        offsets.push(u64::try_from(off).unwrap_or(0));
        off = off.saturating_add(t.data.len());
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(GGUF_MAGIC);
    put_u32(&mut buf, GGUF_VERSION);
    put_i64(&mut buf, i64::try_from(tensors.len()).unwrap_or(0));
    put_i64(&mut buf, i64::try_from(kv.len()).unwrap_or(0));

    for (key, val) in kv {
        put_string(&mut buf, key);
        put_i32(&mut buf, val.tag());
        put_kv_payload(&mut buf, val);
    }

    for (i, (t, offset)) in tensors.iter().zip(offsets.iter()).enumerate() {
        put_string(&mut buf, &t.name);
        put_u32(&mut buf, u32::try_from(t.shape.len()).unwrap_or(0));
        for &d in &t.shape {
            put_i64(&mut buf, i64::try_from(d).unwrap_or(0));
        }
        let type_id = type_ids
            .and_then(|ids| ids.get(i).copied())
            .unwrap_or_else(|| t.ty.to_i32());
        put_i32(&mut buf, type_id);
        put_u64(&mut buf, *offset);
    }

    let data_start = align_up(buf.len(), alignment);
    buf.resize(data_start, 0);
    let mut cursor = 0usize;
    for t in tensors {
        let aligned = align_up(cursor, alignment);
        buf.resize(data_start.saturating_add(aligned), 0);
        buf.extend_from_slice(&t.data);
        cursor = aligned.saturating_add(t.data.len());
    }
    buf
}

/// Parse a GGUF v3 slice. Copies the file once into a single blob; tensor
/// payloads are ranges of that blob, not per-tensor clones.
pub fn load_gguf(bytes: &[u8]) -> Result<Gguf, GgufError> {
    load_gguf_owned(bytes.to_vec())
}

/// Parse an owned GGUF file. The `Vec` becomes the blob; tensor bytes are not
/// copied again.
pub fn load_gguf_owned(bytes: Vec<u8>) -> Result<Gguf, GgufError> {
    let parsed = parse_gguf(&bytes)?;
    Ok(Gguf {
        blob: bytes,
        alignment: parsed.alignment,
        kv: parsed.kv,
        infos: parsed.infos,
    })
}

struct Parsed {
    alignment: usize,
    kv: HashMap<String, Kv>,
    infos: Vec<TensorInfo>,
}

fn parse_gguf(bytes: &[u8]) -> Result<Parsed, GgufError> {
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
        if let Some(_prev) = kv.insert(key, val) {
            // last write wins
        }
    }

    let alignment = match kv.get("general.alignment") {
        Some(Kv::U32(v)) if *v > 0 => usize::try_from(*v).unwrap_or(GGUF_DEFAULT_ALIGNMENT),
        _ => GGUF_DEFAULT_ALIGNMENT,
    };

    let n_tensors_usize = usize::try_from(n_tensors).map_err(|_| GgufError::Truncated)?;
    let mut raw = Vec::with_capacity(n_tensors_usize);
    for _ in 0..n_tensors {
        let name = read_string(bytes, &mut pos)?;
        let n_dims = usize::try_from(read_u32(bytes, &mut pos)?).map_err(|_| GgufError::Shape)?;
        if n_dims == 0 || n_dims > 4 {
            return Err(GgufError::Shape);
        }
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            let d = read_i64(bytes, &mut pos)?;
            if d <= 0 {
                return Err(GgufError::Shape);
            }
            shape.push(u64::try_from(d).map_err(|_| GgufError::Shape)?);
        }
        let ty = GgmlType::from_i32(read_i32(bytes, &mut pos)?)?;
        let offset = read_u64(bytes, &mut pos)?;
        raw.push((name, shape, ty, offset));
    }

    let data_start = align_up(pos, alignment);
    let mut infos = Vec::with_capacity(raw.len());
    for (name, shape, ty, offset) in raw {
        let n_el = shape.iter().try_fold(1u64, |a, &b| a.checked_mul(b));
        let n_el = usize::try_from(n_el.ok_or(GgufError::Shape)?).map_err(|_| GgufError::Shape)?;
        let (block, k) = ty.layout();
        if !n_el.is_multiple_of(k) {
            return Err(GgufError::Shape);
        }
        let nbytes = (n_el / k) * block;
        let start = data_start
            .checked_add(usize::try_from(offset).map_err(|_| GgufError::Truncated)?)
            .ok_or(GgufError::Truncated)?;
        let end = start.checked_add(nbytes).ok_or(GgufError::Truncated)?;
        if bytes.get(start..end).is_none() {
            return Err(GgufError::Truncated);
        }
        infos.push(TensorInfo {
            name,
            ty,
            shape,
            start,
            end,
        });
    }

    Ok(Parsed {
        alignment,
        kv,
        infos,
    })
}

fn align_up(n: usize, a: usize) -> usize {
    n.div_ceil(a) * a
}

fn slice_in_slice(hay: &[u8], needle: &[u8]) -> bool {
    let hay_addr = hay.as_ptr() as usize;
    let needle_addr = needle.as_ptr() as usize;
    let Some(hay_end) = hay_addr.checked_add(hay.len()) else {
        return false;
    };
    let Some(needle_end) = needle_addr.checked_add(needle.len()) else {
        return false;
    };
    needle_addr >= hay_addr && needle_end <= hay_end
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
    put_u64(buf, u64::try_from(s.len()).unwrap_or(0));
    buf.extend_from_slice(s.as_bytes());
}

fn put_u8(buf: &mut Vec<u8>, v: u8) {
    buf.push(v);
}

fn put_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_kv_payload(buf: &mut Vec<u8>, val: &Kv) {
    match val {
        Kv::U8(v) => put_u8(buf, *v),
        Kv::I8(v) => put_u8(buf, u8::from_le_bytes(v.to_le_bytes())),
        Kv::U16(v) => put_u16(buf, *v),
        Kv::I16(v) => buf.extend_from_slice(&v.to_le_bytes()),
        Kv::U32(v) => put_u32(buf, *v),
        Kv::I32(v) => put_i32(buf, *v),
        Kv::F32(v) => put_u32(buf, v.to_bits()),
        Kv::Bool(v) => put_u8(buf, u8::from(*v)),
        Kv::String(s) => put_string(buf, s),
        Kv::Array { elem, items } => {
            put_i32(buf, *elem);
            put_u64(buf, u64::try_from(items.len()).unwrap_or(0));
            for item in items {
                put_kv_payload(buf, item);
            }
        }
        Kv::U64(v) => put_u64(buf, *v),
        Kv::I64(v) => put_i64(buf, *v),
        Kv::F64(v) => put_u64(buf, v.to_bits()),
    }
}

fn read_exact<'a>(bytes: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8], GgufError> {
    let end = pos.checked_add(n).ok_or(GgufError::Truncated)?;
    let s = bytes.get(*pos..end).ok_or(GgufError::Truncated)?;
    *pos = end;
    Ok(s)
}

fn read_array<const N: usize>(bytes: &[u8], pos: &mut usize) -> Result<[u8; N], GgufError> {
    let s = read_exact(bytes, pos, N)?;
    <[u8; N]>::try_from(s).map_err(|_| GgufError::Truncated)
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> Result<u32, GgufError> {
    Ok(u32::from_le_bytes(read_array(bytes, pos)?))
}
fn read_u64(bytes: &[u8], pos: &mut usize) -> Result<u64, GgufError> {
    Ok(u64::from_le_bytes(read_array(bytes, pos)?))
}
fn read_i32(bytes: &[u8], pos: &mut usize) -> Result<i32, GgufError> {
    Ok(i32::from_le_bytes(read_array(bytes, pos)?))
}
fn read_i64(bytes: &[u8], pos: &mut usize) -> Result<i64, GgufError> {
    Ok(i64::from_le_bytes(read_array(bytes, pos)?))
}

fn read_string(bytes: &[u8], pos: &mut usize) -> Result<String, GgufError> {
    let len = usize::try_from(read_u64(bytes, pos)?).map_err(|_| GgufError::Truncated)?;
    let s = read_exact(bytes, pos, len)?;
    String::from_utf8(s.to_vec()).map_err(|_| GgufError::Utf8)
}

fn read_u8(bytes: &[u8], pos: &mut usize) -> Result<u8, GgufError> {
    let s = read_exact(bytes, pos, 1)?;
    Ok(*s.first().ok_or(GgufError::Truncated)?)
}

fn read_u16(bytes: &[u8], pos: &mut usize) -> Result<u16, GgufError> {
    Ok(u16::from_le_bytes(read_array(bytes, pos)?))
}

fn read_i16(bytes: &[u8], pos: &mut usize) -> Result<i16, GgufError> {
    Ok(i16::from_le_bytes(read_array(bytes, pos)?))
}

fn read_kv_value(bytes: &[u8], pos: &mut usize, ty: i32) -> Result<Kv, GgufError> {
    match ty {
        GGUF_TYPE_UINT8 => Ok(Kv::U8(read_u8(bytes, pos)?)),
        GGUF_TYPE_INT8 => Ok(Kv::I8(i8::from_le_bytes([read_u8(bytes, pos)?]))),
        GGUF_TYPE_UINT16 => Ok(Kv::U16(read_u16(bytes, pos)?)),
        GGUF_TYPE_INT16 => Ok(Kv::I16(read_i16(bytes, pos)?)),
        GGUF_TYPE_UINT32 => Ok(Kv::U32(read_u32(bytes, pos)?)),
        GGUF_TYPE_INT32 => Ok(Kv::I32(read_i32(bytes, pos)?)),
        GGUF_TYPE_FLOAT32 => Ok(Kv::F32(f32::from_bits(read_u32(bytes, pos)?))),
        GGUF_TYPE_BOOL => Ok(Kv::Bool(read_u8(bytes, pos)? != 0)),
        GGUF_TYPE_STRING => Ok(Kv::String(read_string(bytes, pos)?)),
        GGUF_TYPE_ARRAY => {
            let elem = read_i32(bytes, pos)?;
            let n = usize::try_from(read_u64(bytes, pos)?).map_err(|_| GgufError::Truncated)?;
            let mut items = Vec::new();
            for _ in 0..n {
                items.push(read_kv_value(bytes, pos, elem)?);
            }
            Ok(Kv::Array { elem, items })
        }
        GGUF_TYPE_UINT64 => Ok(Kv::U64(read_u64(bytes, pos)?)),
        GGUF_TYPE_INT64 => Ok(Kv::I64(read_i64(bytes, pos)?)),
        GGUF_TYPE_FLOAT64 => Ok(Kv::F64(f64::from_bits(read_u64(bytes, pos)?))),
        other => Err(GgufError::UnsupportedKv(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp16::load_f16_le;
    use crate::quant::{
        gemv_q4_0, gemv_q4_k, gemv_q8_0, i8_from_bits, pack_f16, pack_f32, pack_q4_0_from_i4,
        pack_q4_k_block, pack_q6_k_block, pack_q8_0_block, pack_q8_k_block, Q4_0_BLOCK, Q4_K_BLOCK,
        Q6_K_BLOCK, Q8_0_BLOCK, Q8_K_BLOCK, QK4_0, QK8_0, QK_K,
    };

    fn independent_q8_dot(w: &[u8], x: &[u8]) -> f32 {
        assert_eq!(w.len() % Q8_0_BLOCK, 0);
        assert_eq!(w.len(), x.len());
        let mut sum = 0.0f32;
        let mut off = 0;
        while off < w.len() {
            let wb = &w[off..off + Q8_0_BLOCK];
            let xb = &x[off..off + Q8_0_BLOCK];
            let dw = load_f16_le(wb).unwrap();
            let dx = load_f16_le(xb).unwrap();
            let mut acc = 0i32;
            for i in 0..QK8_0 {
                acc += i32::from(i8_from_bits(wb[2 + i])) * i32::from(i8_from_bits(xb[2 + i]));
            }
            sum += (acc as f32) * (dw * dx);
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
            let d = load_f16_le(wb).unwrap();
            for j in 0..(QK4_0 / 2) {
                let packed = wb[2 + j];
                let lo = i32::from(packed & 0x0f) - 8;
                let hi = i32::from(packed >> 4) - 8;
                y[b * QK4_0 + j] = (lo as f32) * d;
                y[b * QK4_0 + j + 16] = (hi as f32) * d;
            }
        }
        y
    }

    fn dequant_q8_0_row(x: &[u8]) -> Vec<f32> {
        let nblocks = x.len() / Q8_0_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK8_0];
        for b in 0..nblocks {
            let xb = &x[b * Q8_0_BLOCK..(b + 1) * Q8_0_BLOCK];
            let d = load_f16_le(xb).unwrap();
            for i in 0..QK8_0 {
                y[b * QK8_0 + i] = f32::from(i8_from_bits(xb[2 + i])) * d;
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
                    let base = i8::try_from(r).unwrap_or(0);
                    let off = i8::try_from(i).unwrap_or(0);
                    *q = base.wrapping_add(off).wrapping_sub(16);
                }
                w8.extend_from_slice(&pack_q8_0_block(
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
            x8.extend_from_slice(&pack_q8_0_block(2.0 / 100.0, &qs));
        }
        for r in 0..n_rows {
            for b in 0..(n_cols / QK4_0) {
                let mut v = [0i8; QK4_0];
                for (i, q) in v.iter_mut().enumerate() {
                    let n = (r * 3 + b + i) % 15;
                    let centered = i32::try_from(n).unwrap_or(0) - 7;
                    *q = i8::try_from(centered).unwrap_or(0);
                }
                w4.extend_from_slice(&pack_q4_0_from_i4(
                    7.0 / 100.0 + f32::from(u16::try_from(r).unwrap_or(0)) / 100.0,
                    &v,
                ));
            }
        }

        let bytes = write_gguf(&[
            TensorWrite {
                name: "w_q8".into(),
                ty: GgmlType::Q8_0,
                shape: vec![
                    u64::try_from(n_cols).unwrap(),
                    u64::try_from(n_rows).unwrap(),
                ],
                data: w8.clone(),
            },
            TensorWrite {
                name: "x_q8".into(),
                ty: GgmlType::Q8_0,
                shape: vec![u64::try_from(n_cols).unwrap()],
                data: x8.clone(),
            },
            TensorWrite {
                name: "w_q4".into(),
                ty: GgmlType::Q4_0,
                shape: vec![
                    u64::try_from(n_cols).unwrap(),
                    u64::try_from(n_rows).unwrap(),
                ],
                data: w4.clone(),
            },
        ]);

        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.alignment(), GGUF_DEFAULT_ALIGNMENT);
        assert_eq!(g.kv_u32("general.alignment"), Some(32));
        assert_eq!(g.kv_string("general.name"), Some("llama-rust"));
        let tw8 = g.tensor("w_q8").expect("w_q8");
        let tx = g.tensor("x_q8").expect("x_q8");
        let tw4 = g.tensor("w_q4").expect("w_q4");
        assert_eq!(tw8.data, w8);
        assert_eq!(tx.data, x8);
        assert_eq!(tw4.data, w4);
        assert_eq!(tw8.ty, GgmlType::Q8_0);
        assert_eq!(tw4.ty, GgmlType::Q4_0);

        let mut y8 = vec![0.0f32; n_rows];
        gemv_q8_0(n_cols, tw8.data, tx.data, &mut y8).unwrap();
        let rb8 = (n_cols / QK8_0) * Q8_0_BLOCK;
        for (r, yv) in y8.iter().enumerate() {
            let expected = independent_q8_dot(&tw8.data[r * rb8..(r + 1) * rb8], tx.data);
            let rel = (yv - expected).abs() / (1.0 + expected.abs());
            assert!(rel * 100_000.0 < 1.0, "q8 row {r}: {yv} vs {expected}");
        }

        let mut y4 = vec![0.0f32; n_rows];
        gemv_q4_0(n_cols, tw4.data, tx.data, &mut y4).unwrap();
        let rb4 = (n_cols / QK4_0) * Q4_0_BLOCK;
        for (r, yv) in y4.iter().enumerate() {
            let expected = independent_q4_dot(&tw4.data[r * rb4..(r + 1) * rb4], tx.data);
            let rel = (yv - expected).abs() / (1.0 + expected.abs());
            assert!(rel * 100_000.0 < 1.0, "q4 row {r}: {yv} vs {expected}");
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
            g.tensor("w_q4").unwrap().data,
            g.tensor("x_q8").unwrap().data,
            &mut y,
        )
        .unwrap();
        let expected = independent_q4_dot(&w, &x);
        // 1*3 + 2*4 = 11 at scale 1 (fp16 1.0 is exact).
        assert!(
            (expected - 11.0).abs() * 100_000.0 < 1.0,
            "oracle {expected}"
        );
        assert!((y[0] - 11.0).abs() * 100_000.0 < 1.0, "gemv {}", y[0]);
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

    /// ggml `dequantize_row_q4_K`: `y = d*sc*q - dmin*m`.
    fn dequant_q4_k_row(w: &[u8]) -> Vec<f32> {
        let nblocks = w.len() / Q4_K_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK_K];
        for b in 0..nblocks {
            let wb = &w[b * Q4_K_BLOCK..(b + 1) * Q4_K_BLOCK];
            let d = load_f16_le(wb).unwrap();
            let minv = load_f16_le(&wb[2..]).unwrap();
            let scales = &wb[4..16];
            let mut qoff = 16usize;
            let mut yo = b * QK_K;
            let mut is = 0usize;
            for _ in 0..4 {
                let (sc, m) = oracle_scale_min_k4(is, scales);
                let d1 = d * f32::from(sc);
                let m1 = minv * f32::from(m);
                let (sc, m) = oracle_scale_min_k4(is + 1, scales);
                let d2 = d * f32::from(sc);
                let m2 = minv * f32::from(m);
                let q = &wb[qoff..qoff + 32];
                for l in 0..32 {
                    y[yo + l] = d1 * f32::from(q[l] & 0x0f) - m1;
                }
                yo += 32;
                for l in 0..32 {
                    y[yo + l] = d2 * f32::from(q[l] >> 4) - m2;
                }
                yo += 32;
                qoff += 32;
                is += 2;
            }
        }
        y
    }

    /// ggml `dequantize_row_q8_K`: `y[j] = d * qs[j]`.
    fn dequant_q8_k_row(x: &[u8]) -> Vec<f32> {
        let nblocks = x.len() / Q8_K_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK_K];
        for b in 0..nblocks {
            let xb = &x[b * Q8_K_BLOCK..(b + 1) * Q8_K_BLOCK];
            let d = f32::from_bits(u32::from_le_bytes(xb[0..4].try_into().unwrap()));
            for i in 0..QK_K {
                y[b * QK_K + i] = d * f32::from(i8_from_bits(xb[4 + i]));
            }
        }
        y
    }

    fn independent_q4k_dot(w: &[u8], x: &[u8]) -> f32 {
        let wy = dequant_q4_k_row(w);
        let xy = dequant_q8_k_row(x);
        assert_eq!(wy.len(), xy.len());
        wy.iter().zip(xy.iter()).map(|(a, b)| a * b).sum()
    }

    #[test]
    fn write_load_kv_types_and_q4k_gemv_match_file_bytes() {
        let mut qs0 = [0u8; QK_K];
        qs0[0] = 1;
        qs0[32] = 2;
        let mut sc0 = [0u8; 8];
        sc0[0] = 1;
        sc0[1] = 1;
        let z8 = [0u8; 8];
        let w0 = pack_q4_k_block(1.0, 0.0, &sc0, &z8, &qs0);

        let qs1 = [0u8; QK_K];
        let mut mn1 = [0u8; 8];
        mn1[0] = 1;
        let w1 = pack_q4_k_block(1.0, 1.0, &z8, &mn1, &qs1);

        let mut w4k = Vec::new();
        w4k.extend_from_slice(&w0);
        w4k.extend_from_slice(&w1);

        let mut xq = [0i8; QK_K];
        xq[0] = 3;
        xq[32] = 4;
        let x8k = pack_q8_k_block(1.0, &xq);

        let w8 = pack_q8_0_block(1.0, &[0i8; QK8_0]);
        let x8 = pack_q8_0_block(1.0, &[0i8; QK8_0]);
        let w4 = pack_q4_0_from_i4(1.0, &[0i8; QK4_0]);

        let nested = Kv::Array {
            elem: GGUF_TYPE_ARRAY,
            items: vec![Kv::Array {
                elem: GGUF_TYPE_UINT8,
                items: vec![Kv::U8(1), Kv::U8(2)],
            }],
        };
        let kv = vec![
            ("general.alignment".into(), Kv::U32(32)),
            ("general.name".into(), Kv::String("llama-rust".into())),
            ("llama.ok".into(), Kv::Bool(true)),
            ("llama.scale".into(), Kv::F32(1.5)),
            (
                "llama.ids".into(),
                Kv::Array {
                    elem: GGUF_TYPE_UINT32,
                    items: vec![Kv::U32(7), Kv::U32(9)],
                },
            ),
            ("llama.u8".into(), Kv::U8(1)),
            ("llama.i8".into(), Kv::I8(-2)),
            ("llama.u16".into(), Kv::U16(3)),
            ("llama.i16".into(), Kv::I16(-4)),
            ("llama.i32".into(), Kv::I32(-5)),
            ("llama.u64".into(), Kv::U64(6)),
            ("llama.i64".into(), Kv::I64(-7)),
            ("llama.f64".into(), Kv::F64(2.0)),
            ("llama.nested".into(), nested.clone()),
        ];
        let bytes = write_gguf_with_kv(
            &kv,
            &[
                TensorWrite {
                    name: "w_q8".into(),
                    ty: GgmlType::Q8_0,
                    shape: vec![32, 1],
                    data: w8.to_vec(),
                },
                TensorWrite {
                    name: "x_q8".into(),
                    ty: GgmlType::Q8_0,
                    shape: vec![32],
                    data: x8.to_vec(),
                },
                TensorWrite {
                    name: "w_q4".into(),
                    ty: GgmlType::Q4_0,
                    shape: vec![32, 1],
                    data: w4.to_vec(),
                },
                TensorWrite {
                    name: "w_q4k".into(),
                    ty: GgmlType::Q4_K,
                    shape: vec![256, 2],
                    data: w4k.clone(),
                },
                TensorWrite {
                    name: "x_q8k".into(),
                    ty: GgmlType::Q8_K,
                    shape: vec![256],
                    data: x8k.to_vec(),
                },
            ],
        );

        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.kv("llama.ok"), Some(&Kv::Bool(true)));
        assert_eq!(g.kv("llama.scale"), Some(&Kv::F32(1.5)));
        assert_eq!(
            g.kv("llama.ids"),
            Some(&Kv::Array {
                elem: GGUF_TYPE_UINT32,
                items: vec![Kv::U32(7), Kv::U32(9)],
            })
        );
        assert_eq!(g.kv("llama.u8"), Some(&Kv::U8(1)));
        assert_eq!(g.kv("llama.i8"), Some(&Kv::I8(-2)));
        assert_eq!(g.kv("llama.u16"), Some(&Kv::U16(3)));
        assert_eq!(g.kv("llama.i16"), Some(&Kv::I16(-4)));
        assert_eq!(g.kv("llama.i32"), Some(&Kv::I32(-5)));
        assert_eq!(g.kv("llama.u64"), Some(&Kv::U64(6)));
        assert_eq!(g.kv("llama.i64"), Some(&Kv::I64(-7)));
        assert_eq!(g.kv("llama.f64"), Some(&Kv::F64(2.0)));
        assert_eq!(g.kv("llama.nested"), Some(&nested));
        assert_eq!(g.tensor("w_q8").unwrap().data, w8);
        assert_eq!(g.tensor("w_q4").unwrap().data, w4);
        assert_eq!(g.tensor("w_q4k").unwrap().data, w4k);
        assert_eq!(g.tensor("x_q8k").unwrap().data, x8k.to_vec());

        let mut y = [0.0f32, 0.0];
        gemv_q4_k(
            256,
            g.tensor("w_q4k").unwrap().data,
            g.tensor("x_q8k").unwrap().data,
            &mut y,
        )
        .unwrap();
        let e0 = independent_q4k_dot(&w0, &x8k);
        let e1 = independent_q4k_dot(&w1, &x8k);
        assert!((e0 - 11.0).abs() * 100_000.0 < 1.0, "oracle0 {e0}");
        assert!((e1 + 3.0).abs() * 100_000.0 < 1.0, "oracle1 {e1}");
        assert!((y[0] - e0).abs() * 100_000.0 < 1.0, "gemv0 {}", y[0]);
        assert!((y[1] - e1).abs() * 100_000.0 < 1.0, "gemv1 {}", y[1]);

        let mut y8 = [0.0f32];
        gemv_q8_0(
            32,
            g.tensor("w_q8").unwrap().data,
            g.tensor("x_q8").unwrap().data,
            &mut y8,
        )
        .unwrap();
        let mut y4 = [0.0f32];
        gemv_q4_0(
            32,
            g.tensor("w_q4").unwrap().data,
            g.tensor("x_q8").unwrap().data,
            &mut y4,
        )
        .unwrap();
        assert!(y8[0].abs() * 100_000.0 < 1.0);
        assert!(y4[0].abs() * 100_000.0 < 1.0);
    }

    #[test]
    fn write_load_f32_and_q6k_match_file_bytes() {
        let f32_data = pack_f32(&[1.0, 2.0, 3.0, 4.0]);
        let mut qs = [0i8; QK_K];
        qs[0] = 1;
        let mut sc = [0i8; 16];
        sc[0] = 1;
        let q6 = pack_q6_k_block(1.0, &sc, &qs);
        let q4 = pack_q4_k_block(1.0, 0.0, &[1u8; 8], &[0u8; 8], &[0u8; QK_K]);
        let bytes = write_gguf(&[
            TensorWrite {
                name: "norm".into(),
                ty: GgmlType::F32,
                shape: vec![4],
                data: f32_data.clone(),
            },
            TensorWrite {
                name: "w_q6k".into(),
                ty: GgmlType::Q6_K,
                shape: vec![256, 1],
                data: q6.to_vec(),
            },
            TensorWrite {
                name: "w_q4k".into(),
                ty: GgmlType::Q4_K,
                shape: vec![256, 1],
                data: q4.to_vec(),
            },
        ]);
        let g = load_gguf(&bytes).expect("load mixed");
        assert_eq!(g.tensor("norm").unwrap().ty, GgmlType::F32);
        assert_eq!(g.tensor("norm").unwrap().data, f32_data);
        assert_eq!(g.tensor("w_q6k").unwrap().ty, GgmlType::Q6_K);
        assert_eq!(g.tensor("w_q6k").unwrap().data.len(), Q6_K_BLOCK);
        assert_eq!(g.tensor("w_q6k").unwrap().data, q6.to_vec());
        assert_eq!(g.tensor("w_q4k").unwrap().data, q4.to_vec());
    }

    #[test]
    fn write_load_f16_matches_file_bytes() {
        let f16_data = pack_f16(&[1.0, -0.5, 0.25, 2.0]);
        let bytes = write_gguf(&[TensorWrite {
            name: "w_f16".into(),
            ty: GgmlType::F16,
            shape: vec![4],
            data: f16_data.clone(),
        }]);
        let g = load_gguf(&bytes).expect("load f16");
        let t = g.tensor("w_f16").expect("w_f16");
        assert_eq!(t.ty, GgmlType::F16);
        assert_eq!(t.data, f16_data.as_slice());
        assert_eq!(t.data.len(), 8);
    }

    #[test]
    fn load_owned_keeps_one_blob_and_tensor_ranges() {
        let f32_data = pack_f32(&[1.0, 2.0, 3.0, 4.0]);
        let bytes = write_gguf(&[TensorWrite {
            name: "norm".into(),
            ty: GgmlType::F32,
            shape: vec![4],
            data: f32_data.clone(),
        }]);
        let g = load_gguf_owned(bytes.clone()).expect("owned");
        assert_eq!(g.blob_len(), bytes.len());
        assert_eq!(g.blob(), bytes.as_slice());
        let t = g.tensor("norm").expect("norm");
        assert!(g.payload_in_blob(t));
        assert_eq!(t.data, f32_data.as_slice());
        let (start, end) = t.blob_range();
        assert_eq!(&g.blob()[start..end], f32_data.as_slice());
        let sliced = load_gguf(&bytes).expect("slice");
        assert_eq!(sliced.blob_len(), bytes.len());
        let st = sliced.tensor("norm").expect("norm2");
        assert!(sliced.payload_in_blob(st));
        assert_eq!(st.data, f32_data.as_slice());
    }

    #[test]
    fn load_unsupported_ggml_type_error_includes_type_id() {
        // ggml Q5_K is 13; still not a shipped decode dtype.
        const Q5_K: i32 = 13;
        let bytes = write_gguf_with_type_ids(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.name".into(), Kv::String("bad-type".into())),
            ],
            &[TensorWrite {
                name: "w".into(),
                ty: GgmlType::F32,
                shape: vec![1],
                data: vec![0, 0, 0, 0],
            }],
            &[Q5_K],
        );
        let err = load_gguf(&bytes).expect_err("unsupported type");
        let msg = err.to_string();
        assert!(
            msg.contains(&Q5_K.to_string()),
            "error should include type id {Q5_K}: {msg}"
        );
    }
}
