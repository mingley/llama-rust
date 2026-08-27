//! Llama-family decode on GGUF bytes: RMSNorm, RoPE, GQA+KV, SwiGLU, lm_head.

use crate::gguf::{GgmlType, Gguf, GgufError, Kv, Tensor, TensorWrite};
use crate::quant::{
    dequant_f32_row, dequant_q4_k_row, dequant_q6_k_row, f32_row_bytes, gemv_f32, gemv_q4_k_f32,
    gemv_q6_k_f32, pack_f32, pack_q4_k_block, pack_q6_k_block, q4_k_row_bytes, q6_k_row_bytes,
    QuantError, QK_K,
};
use crate::tok::{TokError, Tokenizer};
use crate::{write_gguf_with_kv, GGUF_DEFAULT_ALIGNMENT};

const TINY_N_EMBD: usize = 256;
const TINY_N_HEAD: usize = 4;
const TINY_N_HEAD_KV: usize = 2;
const TINY_N_FF: usize = 256;
const TINY_N_LAYER: usize = 1;
const TINY_N_VOCAB: usize = 6;
const TINY_N_ROT: usize = 64;

/// Decode / load failure.
#[derive(Debug)]
pub enum LlamaError {
    /// Missing tensor. Carries the GGUF tensor name.
    Tensor(String),
    /// Hyperparameter / shape mismatch. Carries the tensor, KV key, or check involved.
    Shape(String),
    /// Tensor ggml type cannot be used for this op. `ty` is the GGUF `ggml_type` integer.
    Type {
        /// Tensor name.
        tensor: String,
        /// `ggml_type` integer.
        ty: i32,
    },
    /// Required KV key missing.
    MissingKv(String),
    /// Quantized matmul failed.
    Quant(QuantError),
    /// GGUF parse failed.
    Gguf(GgufError),
    /// Tokenizer failed.
    Tok(TokError),
}

impl std::fmt::Display for LlamaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tensor(n) => write!(f, "missing tensor {n}"),
            Self::Shape(what) => write!(f, "llama shape mismatch: {what}"),
            Self::Type { tensor, ty } => {
                write!(f, "unsupported ggml type {ty} on tensor {tensor}")
            }
            Self::MissingKv(k) => write!(f, "missing kv {k}"),
            Self::Quant(e) => write!(f, "{e}"),
            Self::Gguf(e) => write!(f, "{e}"),
            Self::Tok(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for LlamaError {}

impl From<QuantError> for LlamaError {
    fn from(v: QuantError) -> Self {
        Self::Quant(v)
    }
}

impl From<GgufError> for LlamaError {
    fn from(v: GgufError) -> Self {
        Self::Gguf(v)
    }
}

impl From<TokError> for LlamaError {
    fn from(v: TokError) -> Self {
        Self::Tok(v)
    }
}

struct QuantMat {
    name: String,
    ty: GgmlType,
    n_cols: usize,
    data: Vec<u8>,
    n_rows: usize,
}

/// Per-layer weights.
struct Layer {
    attn_norm: Vec<f32>,
    wq: QuantMat,
    bq: Option<Vec<f32>>,
    wk: QuantMat,
    bk: Option<Vec<f32>>,
    wv: QuantMat,
    bv: Option<Vec<f32>>,
    wo: QuantMat,
    ffn_norm: Vec<f32>,
    gate: QuantMat,
    up: QuantMat,
    down: QuantMat,
}

/// Loaded Llama-family weights. Quantized matrices stay on-disk bytes.
pub struct Llama {
    /// Vocab size.
    pub n_vocab: usize,
    /// Embedding width.
    pub n_embd: usize,
    n_head: usize,
    n_head_kv: usize,
    n_rot: usize,
    rms_eps: f32,
    rope_base: f32,
    token_embd: QuantMat,
    output_norm: Vec<f32>,
    output: QuantMat,
    layers: Vec<Layer>,
}

/// KV cache for GQA decode.
pub struct KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    /// Tokens already in the cache.
    pub n_past: usize,
    max_seq: usize,
}

impl Llama {
    /// Build from a loaded GGUF using `{arch}.*` KV (`llama`, `qwen2`, `mistral`,
    /// or `phi3`) and `blk.{i}.*` tensor names.
    pub fn from_gguf(g: &Gguf) -> Result<Self, LlamaError> {
        let arch = architecture(g)?;
        let n_layer = require_usize(g, arch, "block_count")?;
        let n_embd = require_usize(g, arch, "embedding_length")?;
        let n_head = require_usize(g, arch, "attention.head_count")?;
        let n_head_kv = require_usize(g, arch, "attention.head_count_kv")?;
        if n_layer == 0 {
            return Err(LlamaError::Shape(arch_key(arch, "block_count")));
        }
        if n_embd == 0 {
            return Err(LlamaError::Shape(arch_key(arch, "embedding_length")));
        }
        if n_head == 0 {
            return Err(LlamaError::Shape(arch_key(arch, "attention.head_count")));
        }
        if n_head_kv == 0 {
            return Err(LlamaError::Shape(arch_key(arch, "attention.head_count_kv")));
        }
        if !n_head.is_multiple_of(n_head_kv) {
            return Err(LlamaError::Shape(arch_key(arch, "attention.head_count_kv")));
        }
        let n_rot = rope_dimension(g, arch, n_embd, n_head)?;
        let n_vocab = g
            .tensor("token_embd.weight")
            .map(Tensor::n_rows)
            .ok_or_else(|| LlamaError::Tensor("token_embd.weight".into()))?;
        let rms_eps = arch_f32(g, arch, "attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let rope_base = arch_f32(g, arch, "rope.freq_base").unwrap_or(10_000.0);
        let token_embd = quant_mat(need(g, "token_embd.weight")?)?;
        let output_norm = f32s(need(g, "output_norm.weight")?)?;
        let output = quant_mat(need(g, "output.weight")?)?;
        let mut layers = Vec::new();
        for i in 0..n_layer {
            layers.push(load_layer(g, i)?);
        }
        Ok(Self {
            n_vocab,
            n_embd,
            n_head,
            n_head_kv,
            n_rot,
            rms_eps,
            rope_base,
            token_embd,
            output_norm,
            output,
            layers,
        })
    }

    /// Allocate a KV cache for `max_seq` generated+prompt tokens.
    pub fn new_cache(&self, max_seq: usize) -> Result<KvCache, LlamaError> {
        let hd = self.head_dim()?;
        let n_k = self
            .layers
            .len()
            .checked_mul(self.n_head_kv)
            .and_then(|v| v.checked_mul(max_seq))
            .and_then(|v| v.checked_mul(hd))
            .ok_or_else(|| LlamaError::Shape("kv cache size".into()))?;
        Ok(KvCache {
            k: vec![0.0; n_k],
            v: vec![0.0; n_k],
            n_past: 0,
            max_seq,
        })
    }

    fn head_dim(&self) -> Result<usize, LlamaError> {
        if self.n_embd.is_multiple_of(self.n_head) {
            Ok(self.n_embd / self.n_head)
        } else {
            Err(LlamaError::Shape("embedding_length / head_count".into()))
        }
    }

    /// One decode step. Writes K/V at `cache.n_past` and increments it. Returns logits.
    pub fn forward(&self, cache: &mut KvCache, token: u32) -> Result<Vec<f32>, LlamaError> {
        let hd = self.head_dim()?;
        if cache.n_past >= cache.max_seq {
            return Err(LlamaError::Shape("kv cache full".into()));
        }
        let mut x = embed(&self.token_embd, token)?;
        for (li, layer) in self.layers.iter().enumerate() {
            let residual = x.clone();
            x = rmsnorm(&x, &layer.attn_norm, self.rms_eps)?;
            let q = add_bias(gemv_mat(&layer.wq, &x)?, layer.bq.as_deref())?;
            let k = add_bias(gemv_mat(&layer.wk, &x)?, layer.bk.as_deref())?;
            let v = add_bias(gemv_mat(&layer.wv, &x)?, layer.bv.as_deref())?;
            let mut qh = split_heads(&q, self.n_head, hd)?;
            let mut kh = split_heads(&k, self.n_head_kv, hd)?;
            let vh = split_heads(&v, self.n_head_kv, hd)?;
            for h in &mut qh {
                rope(h, cache.n_past, self.n_rot, self.rope_base)?;
            }
            for h in &mut kh {
                rope(h, cache.n_past, self.n_rot, self.rope_base)?;
            }
            store_kv(
                &mut cache.k,
                li,
                self.n_head_kv,
                cache.max_seq,
                cache.n_past,
                &kh,
            )?;
            store_kv(
                &mut cache.v,
                li,
                self.n_head_kv,
                cache.max_seq,
                cache.n_past,
                &vh,
            )?;
            let seq = cache.n_past + 1;
            let mut attn = vec![0.0f32; self.n_embd];
            let scale = (f32::from(u16::try_from(hd).unwrap_or(1))).sqrt();
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            let gqa = self.n_head / self.n_head_kv;
            for (hq, qvec) in qh.iter().enumerate() {
                let hkv = hq / gqa;
                let mut scores = vec![0.0f32; seq];
                for t in 0..seq {
                    let kv = kv_at(&cache.k, li, hkv, self.n_head_kv, cache.max_seq, t, hd)?;
                    let mut dot = 0.0f32;
                    for (a, b) in qvec.iter().zip(kv.iter()) {
                        dot += *a * *b;
                    }
                    if let Some(s) = scores.get_mut(t) {
                        *s = dot * inv;
                    }
                }
                softmax(&mut scores);
                let mut acc = vec![0.0f32; hd];
                for t in 0..seq {
                    let Some(&st) = scores.get(t) else { continue };
                    let vv = kv_at(&cache.v, li, hkv, self.n_head_kv, cache.max_seq, t, hd)?;
                    for (a, b) in acc.iter_mut().zip(vv.iter()) {
                        *a += st * *b;
                    }
                }
                let off = hq.saturating_mul(hd);
                if let Some(dst) = attn.get_mut(off..off + hd) {
                    for (d, s) in dst.iter_mut().zip(acc.iter()) {
                        *d = *s;
                    }
                }
            }
            let proj = gemv_mat(&layer.wo, &attn)?;
            x = add(&proj, &residual)?;
            let residual = x.clone();
            x = rmsnorm(&x, &layer.ffn_norm, self.rms_eps)?;
            let gate = gemv_mat(&layer.gate, &x)?;
            let up = gemv_mat(&layer.up, &x)?;
            let mut h = silu(&gate)?;
            for (hv, uv) in h.iter_mut().zip(up.iter()) {
                *hv *= *uv;
            }
            let down = gemv_mat(&layer.down, &h)?;
            x = add(&down, &residual)?;
        }
        x = rmsnorm(&x, &self.output_norm, self.rms_eps)?;
        let logits = gemv_mat(&self.output, &x)?;
        cache.n_past += 1;
        Ok(logits)
    }
}

/// Greedy generate: encode prompt, decode `n_predict` tokens, return decoded string.
pub fn greedy_generate(
    model: &Llama,
    tok: &Tokenizer,
    prompt: &str,
    n_predict: usize,
) -> Result<String, LlamaError> {
    let mut ids = prompt_ids(tok, prompt)?;
    if ids.is_empty() {
        return Err(LlamaError::Shape("empty prompt".into()));
    }
    if n_predict == 0 {
        return Ok(tok.decode(&ids));
    }
    let max_seq = ids.len().saturating_add(n_predict).saturating_add(1);
    let mut cache = model.new_cache(max_seq)?;
    let mut last = Vec::new();
    for id in &ids {
        last = model.forward(&mut cache, *id)?;
    }
    for _ in 0..n_predict {
        let next = argmax(&last);
        if tok.eos == Some(next) {
            break;
        }
        ids.push(next);
        last = model.forward(&mut cache, next)?;
    }
    Ok(tok.decode(&ids))
}

fn prompt_ids(tok: &Tokenizer, prompt: &str) -> Result<Vec<u32>, LlamaError> {
    let mut ids = tok.encode(prompt)?;
    if tok.add_bos {
        if let Some(bos) = tok.bos {
            if ids.first().copied() != Some(bos) {
                ids.insert(0, bos);
            }
        }
    }
    Ok(ids)
}

/// Writer-built Llama-shaped GGUF: mixed F32/Q4_K/Q6_K weights + tokenizer KV.
pub fn tiny_llama_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::F32,
        output: GgmlType::F32,
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Qwen2-shaped GGUF: `qwen2.*` KV, no `rope.dimension_count`, QKV bias,
/// `add_bos_token=false`.
pub fn tiny_qwen2_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "qwen2",
        token_embd: GgmlType::F32,
        output: GgmlType::F32,
        rope_dimension_count: false,
        qkv_bias: true,
        add_bos_token: Some(false),
    })
}

/// Writer-built Mistral-shaped GGUF: same tensors as [`tiny_llama_gguf`], `mistral.*` KV.
pub fn tiny_mistral_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "mistral",
        token_embd: GgmlType::F32,
        output: GgmlType::F32,
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Phi-3-shaped GGUF: same tensors as [`tiny_llama_gguf`], `phi3.*` KV.
pub fn tiny_phi3_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "phi3",
        token_embd: GgmlType::F32,
        output: GgmlType::F32,
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with Q4_K `token_embd.weight`.
pub fn tiny_q4k_embd_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::Q4_K,
        output: GgmlType::F32,
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with Q6_K `token_embd.weight` and Q6_K `output.weight`.
pub fn tiny_q6k_embd_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::Q6_K,
        output: GgmlType::Q6_K,
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

struct TinySpec {
    arch: &'static str,
    token_embd: GgmlType,
    output: GgmlType,
    rope_dimension_count: bool,
    qkv_bias: bool,
    add_bos_token: Option<bool>,
}

fn tiny_arch_gguf(spec: TinySpec) -> Vec<u8> {
    let n_embd = TINY_N_EMBD;
    let n_ff = TINY_N_FF;
    let n_vocab = TINY_N_VOCAB;
    let n_kv = TINY_N_HEAD_KV * TINY_N_ROT;
    let ones = vec![1.0f32; n_embd];
    let mut tensors = vec![
        tw(
            "token_embd.weight",
            spec.token_embd,
            vec![n_embd, n_vocab],
            pack_mat(spec.token_embd, n_embd, n_vocab, 1),
        ),
        tw(
            "output_norm.weight",
            GgmlType::F32,
            vec![n_embd],
            pack_f32(&ones),
        ),
        tw(
            "output.weight",
            spec.output,
            vec![n_embd, n_vocab],
            pack_mat(spec.output, n_embd, n_vocab, 2),
        ),
        tw(
            "blk.0.attn_norm.weight",
            GgmlType::F32,
            vec![n_embd],
            pack_f32(&ones),
        ),
        tw(
            "blk.0.ffn_norm.weight",
            GgmlType::F32,
            vec![n_embd],
            pack_f32(&ones),
        ),
        tw(
            "blk.0.attn_k.weight",
            GgmlType::F32,
            vec![n_embd, n_kv],
            pack_f32(&pat_f32(n_embd * n_kv, 3)),
        ),
        tw(
            "blk.0.attn_v.weight",
            GgmlType::F32,
            vec![n_embd, n_kv],
            pack_f32(&pat_f32(n_embd * n_kv, 4)),
        ),
    ];
    tensors.push(tw(
        "blk.0.attn_q.weight",
        GgmlType::Q4_K,
        vec![n_embd, n_embd],
        pack_q4k_mat(n_embd, n_embd, 5),
    ));
    tensors.push(tw(
        "blk.0.attn_output.weight",
        GgmlType::Q4_K,
        vec![n_embd, n_embd],
        pack_q4k_mat(n_embd, n_embd, 6),
    ));
    tensors.push(tw(
        "blk.0.ffn_up.weight",
        GgmlType::Q4_K,
        vec![n_embd, n_ff],
        pack_q4k_mat(n_embd, n_ff, 7),
    ));
    tensors.push(tw(
        "blk.0.ffn_down.weight",
        GgmlType::Q4_K,
        vec![n_ff, n_embd],
        pack_q4k_mat(n_ff, n_embd, 8),
    ));
    tensors.push(tw(
        "blk.0.ffn_gate.weight",
        GgmlType::Q6_K,
        vec![n_embd, n_ff],
        pack_q6k_mat(n_embd, n_ff, 9),
    ));
    if spec.qkv_bias {
        tensors.push(tw(
            "blk.0.attn_q.bias",
            GgmlType::F32,
            vec![n_embd],
            pack_f32(&pat_f32(n_embd, 11)),
        ));
        tensors.push(tw(
            "blk.0.attn_k.bias",
            GgmlType::F32,
            vec![n_kv],
            pack_f32(&pat_f32(n_kv, 12)),
        ));
        tensors.push(tw(
            "blk.0.attn_v.bias",
            GgmlType::F32,
            vec![n_kv],
            pack_f32(&pat_f32(n_kv, 13)),
        ));
    }
    write_gguf_with_kv(&tiny_kv(&spec), &tensors)
}

fn architecture(g: &Gguf) -> Result<&str, LlamaError> {
    match g.kv("general.architecture") {
        Some(Kv::String(s)) if supported_arch(s) => Ok(s.as_str()),
        Some(Kv::String(s)) => Err(LlamaError::Shape(format!("unknown architecture {s}"))),
        _ => Ok("llama"),
    }
}

fn supported_arch(s: &str) -> bool {
    s == "llama" || s == "qwen2" || s == "mistral" || s == "phi3"
}

fn arch_key(arch: &str, field: &str) -> String {
    let mut k = String::new();
    k.push_str(arch);
    k.push('.');
    k.push_str(field);
    k
}

#[cfg(test)]
fn arch_u32(g: &Gguf, arch: &str, field: &str) -> Option<u32> {
    g.kv_u32(&arch_key(arch, field))
}

fn arch_f32(g: &Gguf, arch: &str, field: &str) -> Option<f32> {
    g.kv_f32(&arch_key(arch, field))
}

fn require_usize(g: &Gguf, arch: &str, field: &str) -> Result<usize, LlamaError> {
    let key = arch_key(arch, field);
    let v = g
        .kv_u32(&key)
        .ok_or_else(|| LlamaError::MissingKv(key.clone()))?;
    usize::try_from(v).map_err(|_| LlamaError::Shape(key))
}

fn rope_dimension(g: &Gguf, arch: &str, n_embd: usize, n_head: usize) -> Result<usize, LlamaError> {
    let key = arch_key(arch, "rope.dimension_count");
    if let Some(v) = g.kv_u32(&key) {
        let n = usize::try_from(v).map_err(|_| LlamaError::Shape(key.clone()))?;
        if n == 0 {
            return Err(LlamaError::Shape(key));
        }
        return Ok(n);
    }
    if n_head == 0 || !n_embd.is_multiple_of(n_head) {
        return Err(LlamaError::Shape(key));
    }
    Ok(n_embd / n_head)
}

fn tiny_kv(spec: &TinySpec) -> Vec<(String, Kv)> {
    let arch = spec.arch;
    let mut kv = vec![
        (
            "general.alignment".into(),
            Kv::U32(u32::try_from(GGUF_DEFAULT_ALIGNMENT).unwrap_or(32)),
        ),
        ("general.name".into(), Kv::String("llama-rust-tiny".into())),
        ("general.architecture".into(), Kv::String(arch.into())),
        (
            arch_key(arch, "block_count"),
            Kv::U32(u32::try_from(TINY_N_LAYER).unwrap_or(0)),
        ),
        (
            arch_key(arch, "embedding_length"),
            Kv::U32(u32::try_from(TINY_N_EMBD).unwrap_or(0)),
        ),
        (
            arch_key(arch, "feed_forward_length"),
            Kv::U32(u32::try_from(TINY_N_FF).unwrap_or(0)),
        ),
        (
            arch_key(arch, "attention.head_count"),
            Kv::U32(u32::try_from(TINY_N_HEAD).unwrap_or(0)),
        ),
        (
            arch_key(arch, "attention.head_count_kv"),
            Kv::U32(u32::try_from(TINY_N_HEAD_KV).unwrap_or(0)),
        ),
        (arch_key(arch, "rope.freq_base"), Kv::F32(10_000.0)),
        (
            arch_key(arch, "attention.layer_norm_rms_epsilon"),
            Kv::F32(1.0 / 100_000.0),
        ),
        (
            "tokenizer.ggml.tokens".into(),
            Kv::Array {
                elem: 8,
                items: ["<unk>", "a", "b", "ab", "<s>", "</s>"]
                    .into_iter()
                    .map(|s| Kv::String(s.into()))
                    .collect(),
            },
        ),
        (
            "tokenizer.ggml.merges".into(),
            Kv::Array {
                elem: 8,
                items: vec![Kv::String("a b".into())],
            },
        ),
        ("tokenizer.ggml.bos_token_id".into(), Kv::U32(4)),
        ("tokenizer.ggml.eos_token_id".into(), Kv::U32(5)),
    ];
    if spec.rope_dimension_count {
        kv.push((
            arch_key(arch, "rope.dimension_count"),
            Kv::U32(u32::try_from(TINY_N_ROT).unwrap_or(0)),
        ));
    }
    if let Some(v) = spec.add_bos_token {
        kv.push(("tokenizer.ggml.add_bos_token".into(), Kv::Bool(v)));
    }
    kv
}

fn tw(name: &str, ty: GgmlType, shape: Vec<usize>, data: Vec<u8>) -> TensorWrite {
    TensorWrite {
        name: name.into(),
        ty,
        shape: shape
            .into_iter()
            .map(|d| u64::try_from(d).unwrap_or(0))
            .collect(),
        data,
    }
}

fn pat_f32(n: usize, seed: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed;
    for _ in 0..n {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let k = u16::try_from(s % 200).unwrap_or(0);
        out.push((f32::from(k) - 100.0) / 4000.0);
    }
    out
}

fn pack_mat(ty: GgmlType, n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    match ty {
        GgmlType::Q4_K => pack_q4k_mat(n_cols, n_rows, seed),
        GgmlType::Q6_K => pack_q6k_mat(n_cols, n_rows, seed),
        _ => pack_f32(&pat_f32(n_cols.saturating_mul(n_rows), seed)),
    }
}

fn pack_q4k_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    for _ in 0..n_rows {
        let mut qs = [0u8; QK_K];
        for q in &mut qs {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *q = u8::try_from(s % 8).unwrap_or(0);
        }
        let mut sc = [1u8; 8];
        for c in &mut sc {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *c = u8::try_from(1 + s % 4).unwrap_or(1);
        }
        out.extend_from_slice(&pack_q4_k_block(25.0 / 100.0, 0.0, &sc, &[0u8; 8], &qs));
        let _ = n_cols;
    }
    out
}

fn pack_q6k_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    for _ in 0..n_rows {
        let mut qs = [0i8; QK_K];
        for q in &mut qs {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            let v = i32::try_from(s % 5).unwrap_or(0) - 2;
            *q = i8::try_from(v).unwrap_or(0);
        }
        let mut sc = [1i8; 16];
        for c in &mut sc {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *c = i8::try_from(1 + s % 3).unwrap_or(1);
        }
        out.extend_from_slice(&pack_q6_k_block(25.0 / 100.0, &sc, &qs));
        let _ = n_cols;
    }
    out
}

fn need<'a>(g: &'a Gguf, name: &str) -> Result<&'a Tensor, LlamaError> {
    g.tensor(name)
        .ok_or_else(|| LlamaError::Tensor(name.to_string()))
}

fn load_layer(g: &Gguf, i: usize) -> Result<Layer, LlamaError> {
    Ok(Layer {
        attn_norm: f32s(need(g, &format!("blk.{i}.attn_norm.weight"))?)?,
        wq: quant_mat(need(g, &format!("blk.{i}.attn_q.weight"))?)?,
        bq: optional_f32(g, &format!("blk.{i}.attn_q.bias"))?,
        wk: quant_mat(need(g, &format!("blk.{i}.attn_k.weight"))?)?,
        bk: optional_f32(g, &format!("blk.{i}.attn_k.bias"))?,
        wv: quant_mat(need(g, &format!("blk.{i}.attn_v.weight"))?)?,
        bv: optional_f32(g, &format!("blk.{i}.attn_v.bias"))?,
        wo: quant_mat(need(g, &format!("blk.{i}.attn_output.weight"))?)?,
        ffn_norm: f32s(need(g, &format!("blk.{i}.ffn_norm.weight"))?)?,
        gate: quant_mat(need(g, &format!("blk.{i}.ffn_gate.weight"))?)?,
        up: quant_mat(need(g, &format!("blk.{i}.ffn_up.weight"))?)?,
        down: quant_mat(need(g, &format!("blk.{i}.ffn_down.weight"))?)?,
    })
}

fn optional_f32(g: &Gguf, name: &str) -> Result<Option<Vec<f32>>, LlamaError> {
    match g.tensor(name) {
        Some(t) => Ok(Some(f32s(t)?)),
        None => Ok(None),
    }
}

fn f32s(t: &Tensor) -> Result<Vec<f32>, LlamaError> {
    if t.ty != GgmlType::F32 {
        return Err(LlamaError::Type {
            tensor: t.name.clone(),
            ty: t.ty.to_i32(),
        });
    }
    let (chunks, rem) = t.data.as_chunks::<4>();
    if !rem.is_empty() {
        return Err(LlamaError::Shape(t.name.clone()));
    }
    Ok(chunks
        .iter()
        .map(|c| f32::from_bits(u32::from_le_bytes(*c)))
        .collect())
}

fn quant_mat(t: &Tensor) -> Result<QuantMat, LlamaError> {
    match t.ty {
        GgmlType::F32 | GgmlType::Q4_K | GgmlType::Q6_K => Ok(QuantMat {
            name: t.name.clone(),
            ty: t.ty,
            n_cols: t.n_cols(),
            n_rows: t.n_rows(),
            data: t.data.clone(),
        }),
        other => Err(LlamaError::Type {
            tensor: t.name.clone(),
            ty: other.to_i32(),
        }),
    }
}

fn gemv_mat(m: &QuantMat, x: &[f32]) -> Result<Vec<f32>, LlamaError> {
    let mut y = vec![0.0f32; m.n_rows];
    match m.ty {
        GgmlType::F32 => gemv_f32(m.n_cols, &m.data, x, &mut y)?,
        GgmlType::Q4_K => gemv_q4_k_f32(m.n_cols, &m.data, x, &mut y)?,
        GgmlType::Q6_K => gemv_q6_k_f32(m.n_cols, &m.data, x, &mut y)?,
        other => {
            return Err(LlamaError::Type {
                tensor: m.name.clone(),
                ty: other.to_i32(),
            })
        }
    }
    Ok(y)
}

fn embed(emb: &QuantMat, token: u32) -> Result<Vec<f32>, LlamaError> {
    let row = usize::try_from(token).map_err(|_| LlamaError::Shape(emb.name.clone()))?;
    let rb = match emb.ty {
        GgmlType::F32 => f32_row_bytes(emb.n_cols)?,
        GgmlType::Q4_K => q4_k_row_bytes(emb.n_cols)?,
        GgmlType::Q6_K => q6_k_row_bytes(emb.n_cols)?,
        other => {
            return Err(LlamaError::Type {
                tensor: emb.name.clone(),
                ty: other.to_i32(),
            })
        }
    };
    let start = row
        .checked_mul(rb)
        .ok_or_else(|| LlamaError::Shape(emb.name.clone()))?;
    let end = start
        .checked_add(rb)
        .ok_or_else(|| LlamaError::Shape(emb.name.clone()))?;
    let bytes = emb
        .data
        .get(start..end)
        .ok_or_else(|| LlamaError::Shape(emb.name.clone()))?;
    let mut y = vec![0.0f32; emb.n_cols];
    match emb.ty {
        GgmlType::F32 => dequant_f32_row(emb.n_cols, bytes, &mut y)?,
        GgmlType::Q4_K => dequant_q4_k_row(emb.n_cols, bytes, &mut y)?,
        GgmlType::Q6_K => dequant_q6_k_row(emb.n_cols, bytes, &mut y)?,
        other => {
            return Err(LlamaError::Type {
                tensor: emb.name.clone(),
                ty: other.to_i32(),
            })
        }
    }
    Ok(y)
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Result<Vec<f32>, LlamaError> {
    if x.len() != w.len() {
        return Err(LlamaError::Shape("rmsnorm".into()));
    }
    let mut ss = 0.0f32;
    for v in x {
        ss += *v * *v;
    }
    let n = f32::from(u16::try_from(x.len()).unwrap_or(1));
    let rms = (ss / n + eps).sqrt();
    let inv = if rms > 0.0 { 1.0 / rms } else { 0.0 };
    let mut out = vec![0.0f32; x.len()];
    for ((o, xv), wv) in out.iter_mut().zip(x.iter()).zip(w.iter()) {
        *o = *xv * inv * *wv;
    }
    Ok(out)
}

fn rope(vec: &mut [f32], pos: usize, n_rot: usize, base: f32) -> Result<(), LlamaError> {
    let n = n_rot.min(vec.len());
    if n < 2 {
        return Ok(());
    }
    let n_rot_f = f32::from(u16::try_from(n_rot).unwrap_or(2));
    let mut theta = f32::from(u16::try_from(pos).unwrap_or(u16::MAX));
    let theta_scale = base.powf(-2.0 / n_rot_f);
    let mut i = 0usize;
    while i + 1 < n {
        let cos = theta.cos();
        let sin = theta.sin();
        let x = *vec.get(i).unwrap_or(&0.0);
        let y = *vec.get(i + 1).unwrap_or(&0.0);
        if let Some(slot) = vec.get_mut(i) {
            *slot = x * cos - y * sin;
        }
        if let Some(slot) = vec.get_mut(i + 1) {
            *slot = x * sin + y * cos;
        }
        theta *= theta_scale;
        i += 2;
    }
    Ok(())
}

fn silu(x: &[f32]) -> Result<Vec<f32>, LlamaError> {
    let mut out = vec![0.0f32; x.len()];
    for (o, xv) in out.iter_mut().zip(x.iter()) {
        *o = *xv / (1.0 + (-*xv).exp());
    }
    Ok(out)
}

fn softmax(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    if s > 0.0 {
        for v in x.iter_mut() {
            *v /= s;
        }
    }
}

fn add(a: &[f32], b: &[f32]) -> Result<Vec<f32>, LlamaError> {
    if a.len() != b.len() {
        return Err(LlamaError::Shape("vector add".into()));
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
}

fn add_bias(x: Vec<f32>, bias: Option<&[f32]>) -> Result<Vec<f32>, LlamaError> {
    match bias {
        None => Ok(x),
        Some(b) => add(&x, b),
    }
}

fn split_heads(x: &[f32], n_head: usize, hd: usize) -> Result<Vec<Vec<f32>>, LlamaError> {
    if x.len() != n_head.saturating_mul(hd) {
        return Err(LlamaError::Shape("split_heads".into()));
    }
    let mut out = Vec::new();
    for h in 0..n_head {
        let off = h.saturating_mul(hd);
        let row = x
            .get(off..off + hd)
            .ok_or_else(|| LlamaError::Shape("split_heads".into()))?;
        out.push(row.to_vec());
    }
    Ok(out)
}

fn store_kv(
    cache: &mut [f32],
    layer: usize,
    n_head_kv: usize,
    max_seq: usize,
    t: usize,
    heads: &[Vec<f32>],
) -> Result<(), LlamaError> {
    let hd = heads.first().map(Vec::len).unwrap_or(0);
    for (h, vec) in heads.iter().enumerate() {
        let off = kv_offset(layer, h, n_head_kv, max_seq, t, hd)?;
        let dst = cache
            .get_mut(off..off + hd)
            .ok_or_else(|| LlamaError::Shape("kv store".into()))?;
        for (d, s) in dst.iter_mut().zip(vec.iter()) {
            *d = *s;
        }
    }
    Ok(())
}

fn kv_at(
    cache: &[f32],
    layer: usize,
    head: usize,
    n_head_kv: usize,
    max_seq: usize,
    t: usize,
    hd: usize,
) -> Result<&[f32], LlamaError> {
    let off = kv_offset(layer, head, n_head_kv, max_seq, t, hd)?;
    cache
        .get(off..off + hd)
        .ok_or_else(|| LlamaError::Shape("kv load".into()))
}

fn kv_offset(
    layer: usize,
    head: usize,
    n_head_kv: usize,
    max_seq: usize,
    t: usize,
    hd: usize,
) -> Result<usize, LlamaError> {
    layer
        .checked_mul(n_head_kv)
        .and_then(|v| v.checked_add(head))
        .and_then(|v| v.checked_mul(max_seq))
        .and_then(|v| v.checked_add(t))
        .and_then(|v| v.checked_mul(hd))
        .ok_or_else(|| LlamaError::Shape("kv offset".into()))
}

fn argmax(x: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, v) in x.iter().enumerate() {
        if *v > best {
            best = *v;
            best_i = i;
        }
    }
    u32::try_from(best_i).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_gguf;

    /// ggml `dequantize_row_q4_K` (oracle).
    fn dequant_q4_k_row(w: &[u8]) -> Vec<f32> {
        let nblocks = w.len() / crate::quant::Q4_K_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK_K];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::Q4_K_BLOCK..(b + 1) * crate::quant::Q4_K_BLOCK];
            let d = crate::fp16::load_f16_le(wb).unwrap();
            let minv = crate::fp16::load_f16_le(&wb[2..]).unwrap();
            let scales = &wb[4..16];
            let mut qoff = 16usize;
            let mut yo = b * QK_K;
            let mut is = 0usize;
            for _ in 0..4 {
                let (sc, m) = oracle_scale_min(is, scales);
                let d1 = d * f32::from(sc);
                let m1 = minv * f32::from(m);
                let (sc, m) = oracle_scale_min(is + 1, scales);
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

    fn oracle_scale_min(j: usize, q: &[u8]) -> (u8, u8) {
        if j < 4 {
            (q[j] & 63, q[j + 4] & 63)
        } else {
            (
                (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4),
                (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
            )
        }
    }

    /// ggml `dequantize_row_q6_K` (oracle).
    fn dequant_q6_k_row(w: &[u8]) -> Vec<f32> {
        let nblocks = w.len() / crate::quant::Q6_K_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK_K];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::Q6_K_BLOCK..(b + 1) * crate::quant::Q6_K_BLOCK];
            let d = crate::fp16::load_f16_le(&wb[208..]).unwrap();
            let ql = &wb[0..128];
            let qh = &wb[128..192];
            let sc = &wb[192..208];
            for group in 0..2 {
                let ql_off = group * 64;
                let qh_off = group * 32;
                let sc_off = group * 8;
                let yo = b * QK_K + group * 128;
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = i32::from((ql[ql_off + l] & 0x0f) | ((qh[qh_off + l] & 3) << 4)) - 32;
                    let q2 = i32::from(
                        (ql[ql_off + l + 32] & 0x0f) | (((qh[qh_off + l] >> 2) & 3) << 4),
                    ) - 32;
                    let q3 =
                        i32::from((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) - 32;
                    let q4 =
                        i32::from((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4))
                            - 32;
                    let s = |off: usize| i8::from_le_bytes([sc[sc_off + off]]);
                    y[yo + l] = d * f32::from(s(is)) * (q1 as f32);
                    y[yo + l + 32] = d * f32::from(s(is + 2)) * (q2 as f32);
                    y[yo + l + 64] = d * f32::from(s(is + 4)) * (q3 as f32);
                    y[yo + l + 96] = d * f32::from(s(is + 6)) * (q4 as f32);
                }
            }
        }
        y
    }

    fn oracle_gemv(t: &Tensor, x: &[f32]) -> Vec<f32> {
        let n_cols = t.n_cols();
        let n_rows = t.n_rows();
        let mut y = vec![0.0f32; n_rows];
        match t.ty {
            GgmlType::F32 => {
                for (r, yv) in y.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for (c, xv) in x.iter().enumerate() {
                        let off = (r * n_cols + c) * 4;
                        let w = f32::from_bits(u32::from_le_bytes(
                            t.data[off..off + 4].try_into().unwrap(),
                        ));
                        acc += w * *xv;
                    }
                    *yv = acc;
                }
            }
            GgmlType::Q4_K => {
                let rb = (n_cols / QK_K) * crate::quant::Q4_K_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_q4_k_row(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::Q6_K => {
                let rb = (n_cols / QK_K) * crate::quant::Q6_K_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_q6_k_row(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            _ => panic!("oracle ty"),
        }
        y
    }

    fn oracle_embed(t: &Tensor, token: u32) -> Vec<f32> {
        let n_cols = t.n_cols();
        let row = usize::try_from(token).unwrap();
        match t.ty {
            GgmlType::F32 => {
                let mut x = vec![0.0f32; n_cols];
                for (c, xv) in x.iter_mut().enumerate() {
                    let off = (row * n_cols + c) * 4;
                    *xv = f32::from_bits(u32::from_le_bytes(
                        t.data[off..off + 4].try_into().unwrap(),
                    ));
                }
                x
            }
            GgmlType::Q4_K => {
                let rb = (n_cols / QK_K) * crate::quant::Q4_K_BLOCK;
                dequant_q4_k_row(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::Q6_K => {
                let rb = (n_cols / QK_K) * crate::quant::Q6_K_BLOCK;
                dequant_q6_k_row(&t.data[row * rb..(row + 1) * rb])
            }
            _ => panic!("oracle embed ty"),
        }
    }

    fn oracle_add_bias(mut y: Vec<f32>, bias: Option<&Tensor>) -> Vec<f32> {
        let Some(b) = bias else {
            return y;
        };
        let bv = f32s(b).unwrap();
        for (yv, bb) in y.iter_mut().zip(bv.iter()) {
            *yv += *bb;
        }
        y
    }

    fn oracle_rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let ss: f32 = x.iter().map(|v| v * v).sum();
        let rms = (ss / x.len() as f32 + eps).sqrt();
        x.iter().zip(w.iter()).map(|(a, b)| a / rms * b).collect()
    }

    fn oracle_rope(mut v: Vec<f32>, pos: usize, n_rot: usize, base: f32) -> Vec<f32> {
        let mut theta = pos as f32;
        let theta_scale = base.powf(-2.0 / n_rot as f32);
        let mut i = 0usize;
        while i + 1 < n_rot.min(v.len()) {
            let (c, s) = (theta.cos(), theta.sin());
            let x = v[i];
            let y = v[i + 1];
            v[i] = x * c - y * s;
            v[i + 1] = x * s + y * c;
            theta *= theta_scale;
            i += 2;
        }
        v
    }

    fn oracle_n_rot(g: &Gguf, arch: &str, n_embd: usize, n_head: usize) -> usize {
        arch_u32(g, arch, "rope.dimension_count")
            .map(|v| v as usize)
            .unwrap_or(n_embd / n_head)
    }

    fn oracle_forward(g: &Gguf, token: u32) -> Vec<f32> {
        let arch = architecture(g).expect("arch");
        let n_embd = arch_u32(g, arch, "embedding_length").unwrap() as usize;
        let n_head = arch_u32(g, arch, "attention.head_count").unwrap() as usize;
        let n_kv = arch_u32(g, arch, "attention.head_count_kv").unwrap() as usize;
        let n_rot = oracle_n_rot(g, arch, n_embd, n_head);
        let eps = arch_f32(g, arch, "attention.layer_norm_rms_epsilon").unwrap();
        let base = arch_f32(g, arch, "rope.freq_base").unwrap();
        let hd = n_embd / n_head;
        let emb = g.tensor("token_embd.weight").unwrap();
        let residual = oracle_embed(emb, token);
        let an = f32s(g.tensor("blk.0.attn_norm.weight").unwrap()).unwrap();
        let x = oracle_rmsnorm(&residual, &an, eps);
        let q = oracle_add_bias(
            oracle_gemv(g.tensor("blk.0.attn_q.weight").unwrap(), &x),
            g.tensor("blk.0.attn_q.bias"),
        );
        let k = oracle_add_bias(
            oracle_gemv(g.tensor("blk.0.attn_k.weight").unwrap(), &x),
            g.tensor("blk.0.attn_k.bias"),
        );
        let v = oracle_add_bias(
            oracle_gemv(g.tensor("blk.0.attn_v.weight").unwrap(), &x),
            g.tensor("blk.0.attn_v.bias"),
        );
        let mut qh: Vec<Vec<f32>> = q.chunks(hd).map(<[f32]>::to_vec).collect();
        let mut kh: Vec<Vec<f32>> = k.chunks(hd).map(<[f32]>::to_vec).collect();
        let vh: Vec<Vec<f32>> = v.chunks(hd).map(<[f32]>::to_vec).collect();
        for h in &mut qh {
            *h = oracle_rope(h.clone(), 0, n_rot, base);
        }
        for h in &mut kh {
            *h = oracle_rope(h.clone(), 0, n_rot, base);
        }
        let gqa = n_head / n_kv;
        let mut attn = vec![0.0f32; n_embd];
        // seq=1: softmax of a single score is 1, so attn = v_head (GQA).
        for (hq, _) in qh.iter().enumerate() {
            let hkv = hq / gqa;
            let off = hq * hd;
            if let Some(dst) = attn.get_mut(off..off + hd) {
                for (d, s) in dst.iter_mut().zip(vh[hkv].iter()) {
                    *d = *s;
                }
            }
        }
        let mut x = oracle_gemv(g.tensor("blk.0.attn_output.weight").unwrap(), &attn);
        x = x.iter().zip(residual.iter()).map(|(a, b)| a + b).collect();
        let fnorm = f32s(g.tensor("blk.0.ffn_norm.weight").unwrap()).unwrap();
        let xn = oracle_rmsnorm(&x, &fnorm, eps);
        let gate = oracle_gemv(g.tensor("blk.0.ffn_gate.weight").unwrap(), &xn);
        let up = oracle_gemv(g.tensor("blk.0.ffn_up.weight").unwrap(), &xn);
        let h: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let down = oracle_gemv(g.tensor("blk.0.ffn_down.weight").unwrap(), &h);
        x = down.iter().zip(x.iter()).map(|(a, b)| a + b).collect();
        let on = f32s(g.tensor("output_norm.weight").unwrap()).unwrap();
        x = oracle_rmsnorm(&x, &on, eps);
        oracle_gemv(g.tensor("output.weight").unwrap(), &x)
    }

    fn assert_logits_match(got: &[f32], exp: &[f32]) {
        assert_eq!(got.len(), exp.len());
        for (i, (a, b)) in got.iter().zip(exp.iter()).enumerate() {
            let rel = (a - b).abs() / (1.0 + b.abs());
            assert!(rel * 1000.0 < 1.0, "logit {i}: {a} vs {b}");
        }
    }

    fn load_fwd_match(bytes: &[u8], token: u32) {
        let g = load_gguf(bytes).expect("load");
        let model = Llama::from_gguf(&g).expect("model");
        let mut cache = model.new_cache(4).expect("cache");
        let got = model.forward(&mut cache, token).expect("fwd");
        let exp = oracle_forward(&g, token);
        assert_logits_match(&got, &exp);
        assert_eq!(cache.n_past, 1);
    }

    #[test]
    fn tiny_llama_logits_match_independent_oracle() {
        let bytes = tiny_llama_gguf();
        let g = load_gguf(&bytes).expect("load tiny");
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::Q6_K
        );
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::Q4_K);
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert!(g.tensor("blk.0.attn_q.bias").is_none());
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_llama_encode_greedy_decode_uses_shipped_path() {
        let bytes = tiny_llama_gguf();
        let g = load_gguf(&bytes).expect("load");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        assert!(tok.add_bos);
        assert_eq!(tok.encode("ab").unwrap(), vec![3]);
        assert_eq!(tok.decode(&[1, 2]), "ab");
        assert_eq!(prompt_ids(&tok, "ab").unwrap().first().copied(), tok.bos);
        let model = Llama::from_gguf(&g).expect("model");
        let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
        assert!(out.contains("ab"), "{out}");
        let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
        assert_eq!(out, out2);
        assert_eq!(greedy_generate(&model, &tok, "ab", 0).expect("n=0"), "ab");
    }

    #[test]
    fn tiny_qwen2_logits_match_independent_oracle() {
        let bytes = tiny_qwen2_gguf();
        let g = load_gguf(&bytes).expect("load qwen2");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("qwen2".into()))
        );
        assert_eq!(g.kv_u32("qwen2.block_count"), Some(1));
        assert!(g.kv_u32("llama.block_count").is_none());
        assert!(g.kv_u32("qwen2.rope.dimension_count").is_none());
        assert!(g.tensor("blk.0.attn_q.bias").is_some());
        assert_eq!(g.kv_bool("tokenizer.ggml.add_bos_token"), Some(false));
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_qwen2_greedy_two_runs_match_without_hardcoded_text() {
        let bytes = tiny_qwen2_gguf();
        let g = load_gguf(&bytes).expect("load");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        assert!(!tok.add_bos);
        let ids = prompt_ids(&tok, "ab").expect("ids");
        assert_eq!(ids, tok.encode("ab").unwrap());
        assert_ne!(ids.first().copied(), tok.bos);
        let model = Llama::from_gguf(&g).expect("model");
        let prompt = tok.decode(&[3]);
        assert!(!prompt.is_empty());
        let out = greedy_generate(&model, &tok, &prompt, 2).expect("gen");
        assert!(!out.is_empty());
        let out2 = greedy_generate(&model, &tok, &prompt, 2).expect("gen2");
        assert_eq!(out, out2);
        let empty = greedy_generate(&model, &tok, "", 2);
        match empty {
            Ok(s) => panic!("empty qwen2 prompt should fail, got {s:?}"),
            Err(e) => assert!(e.to_string().contains("empty prompt"), "{e}"),
        }
    }

    #[test]
    fn tiny_q4k_embd_logits_match_independent_oracle() {
        let bytes = tiny_q4k_embd_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::Q4_K);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_q6k_embd_logits_match_independent_oracle() {
        let bytes = tiny_q6k_embd_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::Q6_K);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::Q6_K);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_mistral_and_phi3_load_and_greedy() {
        for bytes in [tiny_mistral_gguf(), tiny_phi3_gguf()] {
            let g = load_gguf(&bytes).expect("load");
            let tok = Tokenizer::from_gguf(&g).expect("tok");
            let model = Llama::from_gguf(&g).expect("model");
            let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
            let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
            assert_eq!(out, out2);
            assert!(!out.is_empty());
            load_fwd_match(&bytes, 3);
        }
        let m = load_gguf(&tiny_mistral_gguf()).unwrap();
        assert_eq!(m.kv_u32("mistral.block_count"), Some(1));
        let p = load_gguf(&tiny_phi3_gguf()).unwrap();
        assert_eq!(p.kv_u32("phi3.block_count"), Some(1));
    }

    #[test]
    fn decode_load_unsupported_ggml_type_error_includes_type_id() {
        const Q5_K: i32 = 10;
        let bytes = crate::gguf::write_gguf_with_type_ids(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("qwen2".into())),
            ],
            &[TensorWrite {
                name: "w".into(),
                ty: GgmlType::F32,
                shape: vec![1],
                data: vec![0, 0, 0, 0],
            }],
            &[Q5_K],
        );
        let err = match load_gguf(&bytes) {
            Err(e) => e.to_string(),
            Ok(g) => match Llama::from_gguf(&g) {
                Ok(_) => panic!("decode should reject unknown type"),
                Err(e) => e.to_string(),
            },
        };
        assert!(
            err.contains(&Q5_K.to_string()),
            "error should include type id {Q5_K}: {err}"
        );
    }

    #[test]
    fn missing_tensor_error_names_tensor() {
        let bytes = write_gguf_with_kv(
            &tiny_kv(&TinySpec {
                arch: "llama",
                token_embd: GgmlType::F32,
                output: GgmlType::F32,
                rope_dimension_count: true,
                qkv_bias: false,
                add_bos_token: None,
            }),
            &[tw(
                "token_embd.weight",
                GgmlType::F32,
                vec![TINY_N_EMBD, TINY_N_VOCAB],
                pack_f32(&pat_f32(TINY_N_EMBD * TINY_N_VOCAB, 1)),
            )],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(&g) {
            Ok(_) => panic!("expected missing tensor"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("output_norm.weight"),
            "error should name tensor: {err}"
        );
    }

    #[test]
    fn unknown_architecture_error_names_arch() {
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("falcon".into())),
            ],
            &[],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(&g) {
            Ok(_) => panic!("expected unknown arch"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("falcon"), "error should name arch: {err}");
    }

    #[test]
    fn missing_kv_error_names_key() {
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("llama".into())),
            ],
            &[],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(&g) {
            Ok(_) => panic!("expected missing kv"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("llama.block_count"),
            "error should name kv key: {err}"
        );
    }
}
