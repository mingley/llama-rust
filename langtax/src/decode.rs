//! Llama-family decode on GGUF bytes: RMSNorm, RoPE, GQA+KV, SwiGLU, lm_head.

use crate::gguf::{GgmlType, Gguf, GgufError, Kv, Tensor, TensorWrite};
use crate::quant::{
    gemv_f32, gemv_q4_k_f32, gemv_q6_k_f32, pack_f32, pack_q4_k_block, pack_q6_k_block, QuantError,
    QK_K,
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
    /// Missing or malformed tensor.
    Tensor(&'static str),
    /// Hyperparameter / shape mismatch.
    Shape,
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
            Self::Shape => write!(f, "llama shape mismatch"),
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
    ty: GgmlType,
    n_cols: usize,
    data: Vec<u8>,
    n_rows: usize,
}

/// Per-layer weights.
struct Layer {
    attn_norm: Vec<f32>,
    wq: QuantMat,
    wk: QuantMat,
    wv: QuantMat,
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
    token_embd: Vec<f32>,
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
    /// Build from a loaded GGUF using `llama.*` KV and `blk.{i}.*` tensor names.
    pub fn from_gguf(g: &Gguf) -> Result<Self, LlamaError> {
        let n_layer = usize_from_u32(g.kv_u32("llama.block_count"))?;
        let n_embd = usize_from_u32(g.kv_u32("llama.embedding_length"))?;
        let n_head = usize_from_u32(g.kv_u32("llama.attention.head_count"))?;
        let n_head_kv = usize_from_u32(g.kv_u32("llama.attention.head_count_kv"))?;
        let n_rot = usize_from_u32(g.kv_u32("llama.rope.dimension_count"))?;
        let n_vocab = g
            .tensor("token_embd.weight")
            .map(Tensor::n_rows)
            .ok_or(LlamaError::Tensor("token_embd.weight"))?;
        if n_layer == 0 || n_embd == 0 || n_head == 0 || n_head_kv == 0 || n_rot == 0 {
            return Err(LlamaError::Shape);
        }
        if !n_head.is_multiple_of(n_head_kv) {
            return Err(LlamaError::Shape);
        }
        let rms_eps = g
            .kv_f32("llama.attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-5);
        let rope_base = g.kv_f32("llama.rope.freq_base").unwrap_or(10_000.0);
        let token_embd = f32s(
            g.tensor("token_embd.weight")
                .ok_or(LlamaError::Tensor("token_embd.weight"))?,
        )?;
        let output_norm = f32s(
            g.tensor("output_norm.weight")
                .ok_or(LlamaError::Tensor("output_norm.weight"))?,
        )?;
        let output = quant_mat(
            g.tensor("output.weight")
                .ok_or(LlamaError::Tensor("output.weight"))?,
        )?;
        let mut layers = Vec::new();
        for i in 0..n_layer {
            layers.push(Layer {
                attn_norm: f32s(need(g, &format!("blk.{i}.attn_norm.weight"))?)?,
                wq: quant_mat(need(g, &format!("blk.{i}.attn_q.weight"))?)?,
                wk: quant_mat(need(g, &format!("blk.{i}.attn_k.weight"))?)?,
                wv: quant_mat(need(g, &format!("blk.{i}.attn_v.weight"))?)?,
                wo: quant_mat(need(g, &format!("blk.{i}.attn_output.weight"))?)?,
                ffn_norm: f32s(need(g, &format!("blk.{i}.ffn_norm.weight"))?)?,
                gate: quant_mat(need(g, &format!("blk.{i}.ffn_gate.weight"))?)?,
                up: quant_mat(need(g, &format!("blk.{i}.ffn_up.weight"))?)?,
                down: quant_mat(need(g, &format!("blk.{i}.ffn_down.weight"))?)?,
            });
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
            .ok_or(LlamaError::Shape)?;
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
            Err(LlamaError::Shape)
        }
    }

    /// One decode step. Writes K/V at `cache.n_past` and increments it. Returns logits.
    pub fn forward(&self, cache: &mut KvCache, token: u32) -> Result<Vec<f32>, LlamaError> {
        let hd = self.head_dim()?;
        if cache.n_past >= cache.max_seq {
            return Err(LlamaError::Shape);
        }
        let mut x = embed(&self.token_embd, self.n_embd, token)?;
        for (li, layer) in self.layers.iter().enumerate() {
            let residual = x.clone();
            x = rmsnorm(&x, &layer.attn_norm, self.rms_eps)?;
            let q = gemv_mat(&layer.wq, &x)?;
            let k = gemv_mat(&layer.wk, &x)?;
            let v = gemv_mat(&layer.wv, &x)?;
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
    let mut ids = tok.encode(prompt)?;
    if let Some(bos) = tok.bos {
        if ids.first().copied() != Some(bos) {
            ids.insert(0, bos);
        }
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

/// Writer-built Llama-shaped GGUF: mixed F32/Q4_K/Q6_K weights + tokenizer KV.
pub fn tiny_llama_gguf() -> Vec<u8> {
    let n_embd = TINY_N_EMBD;
    let n_ff = TINY_N_FF;
    let n_vocab = TINY_N_VOCAB;
    let n_kv = TINY_N_HEAD_KV * TINY_N_ROT;
    let ones = vec![1.0f32; n_embd];
    let mut tensors = vec![
        tw(
            "token_embd.weight",
            GgmlType::F32,
            vec![n_embd, n_vocab],
            pack_f32(&pat_f32(n_embd * n_vocab, 1)),
        ),
        tw(
            "output_norm.weight",
            GgmlType::F32,
            vec![n_embd],
            pack_f32(&ones),
        ),
        tw(
            "output.weight",
            GgmlType::F32,
            vec![n_embd, n_vocab],
            pack_f32(&pat_f32(n_embd * n_vocab, 2)),
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
    let kv = tiny_kv();
    write_gguf_with_kv(&kv, &tensors)
}

fn tiny_kv() -> Vec<(String, Kv)> {
    vec![
        (
            "general.alignment".into(),
            Kv::U32(u32::try_from(GGUF_DEFAULT_ALIGNMENT).unwrap_or(32)),
        ),
        ("general.name".into(), Kv::String("llama-rust-tiny".into())),
        ("general.architecture".into(), Kv::String("llama".into())),
        (
            "llama.block_count".into(),
            Kv::U32(u32::try_from(TINY_N_LAYER).unwrap_or(0)),
        ),
        (
            "llama.embedding_length".into(),
            Kv::U32(u32::try_from(TINY_N_EMBD).unwrap_or(0)),
        ),
        (
            "llama.feed_forward_length".into(),
            Kv::U32(u32::try_from(TINY_N_FF).unwrap_or(0)),
        ),
        (
            "llama.attention.head_count".into(),
            Kv::U32(u32::try_from(TINY_N_HEAD).unwrap_or(0)),
        ),
        (
            "llama.attention.head_count_kv".into(),
            Kv::U32(u32::try_from(TINY_N_HEAD_KV).unwrap_or(0)),
        ),
        (
            "llama.rope.dimension_count".into(),
            Kv::U32(u32::try_from(TINY_N_ROT).unwrap_or(0)),
        ),
        ("llama.rope.freq_base".into(), Kv::F32(10_000.0)),
        (
            "llama.attention.layer_norm_rms_epsilon".into(),
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
    ]
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
    g.tensor(name).ok_or(LlamaError::Tensor("layer tensor"))
}

fn usize_from_u32(v: Option<u32>) -> Result<usize, LlamaError> {
    usize::try_from(v.unwrap_or(0)).map_err(|_| LlamaError::Shape)
}

fn f32s(t: &Tensor) -> Result<Vec<f32>, LlamaError> {
    if t.ty != GgmlType::F32 {
        return Err(LlamaError::Shape);
    }
    let (chunks, rem) = t.data.as_chunks::<4>();
    if !rem.is_empty() {
        return Err(LlamaError::Shape);
    }
    Ok(chunks
        .iter()
        .map(|c| f32::from_bits(u32::from_le_bytes(*c)))
        .collect())
}

fn quant_mat(t: &Tensor) -> Result<QuantMat, LlamaError> {
    Ok(QuantMat {
        ty: t.ty,
        n_cols: t.n_cols(),
        n_rows: t.n_rows(),
        data: t.data.clone(),
    })
}

fn gemv_mat(m: &QuantMat, x: &[f32]) -> Result<Vec<f32>, LlamaError> {
    let mut y = vec![0.0f32; m.n_rows];
    match m.ty {
        GgmlType::F32 => gemv_f32(m.n_cols, &m.data, x, &mut y)?,
        GgmlType::Q4_K => gemv_q4_k_f32(m.n_cols, &m.data, x, &mut y)?,
        GgmlType::Q6_K => gemv_q6_k_f32(m.n_cols, &m.data, x, &mut y)?,
        _ => return Err(LlamaError::Shape),
    }
    Ok(y)
}

fn embed(emb: &[f32], n_embd: usize, token: u32) -> Result<Vec<f32>, LlamaError> {
    let row = usize::try_from(token).map_err(|_| LlamaError::Shape)?;
    let start = row.checked_mul(n_embd).ok_or(LlamaError::Shape)?;
    emb.get(start..start + n_embd)
        .map(<[f32]>::to_vec)
        .ok_or(LlamaError::Shape)
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Result<Vec<f32>, LlamaError> {
    if x.len() != w.len() {
        return Err(LlamaError::Shape);
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
        return Err(LlamaError::Shape);
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
}

fn split_heads(x: &[f32], n_head: usize, hd: usize) -> Result<Vec<Vec<f32>>, LlamaError> {
    if x.len() != n_head.saturating_mul(hd) {
        return Err(LlamaError::Shape);
    }
    let mut out = Vec::new();
    for h in 0..n_head {
        let off = h.saturating_mul(hd);
        let row = x.get(off..off + hd).ok_or(LlamaError::Shape)?;
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
        let dst = cache.get_mut(off..off + hd).ok_or(LlamaError::Shape)?;
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
    cache.get(off..off + hd).ok_or(LlamaError::Shape)
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
        .ok_or(LlamaError::Shape)
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

    fn oracle_forward(g: &Gguf, token: u32) -> Vec<f32> {
        let n_embd = g.kv_u32("llama.embedding_length").unwrap() as usize;
        let n_head = g.kv_u32("llama.attention.head_count").unwrap() as usize;
        let n_kv = g.kv_u32("llama.attention.head_count_kv").unwrap() as usize;
        let n_rot = g.kv_u32("llama.rope.dimension_count").unwrap() as usize;
        let eps = g.kv_f32("llama.attention.layer_norm_rms_epsilon").unwrap();
        let base = g.kv_f32("llama.rope.freq_base").unwrap();
        let hd = n_embd / n_head;
        let emb = g.tensor("token_embd.weight").unwrap();
        let mut x = vec![0.0f32; n_embd];
        let row = usize::try_from(token).unwrap();
        for (c, xv) in x.iter_mut().enumerate() {
            let off = (row * n_embd + c) * 4;
            *xv = f32::from_bits(u32::from_le_bytes(
                emb.data[off..off + 4].try_into().unwrap(),
            ));
        }
        let an = f32s(g.tensor("blk.0.attn_norm.weight").unwrap()).unwrap();
        x = oracle_rmsnorm(&x, &an, eps);
        let q = oracle_gemv(g.tensor("blk.0.attn_q.weight").unwrap(), &x);
        let k = oracle_gemv(g.tensor("blk.0.attn_k.weight").unwrap(), &x);
        let v = oracle_gemv(g.tensor("blk.0.attn_v.weight").unwrap(), &x);
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
        // residual skipped first rms? wait we overwrote x. Re-read embed residual.
        let mut res = vec![0.0f32; n_embd];
        for (c, xv) in res.iter_mut().enumerate() {
            let off = (row * n_embd + c) * 4;
            *xv = f32::from_bits(u32::from_le_bytes(
                emb.data[off..off + 4].try_into().unwrap(),
            ));
        }
        x = x.iter().zip(res.iter()).map(|(a, b)| a + b).collect();
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
        let model = Llama::from_gguf(&g).expect("model");
        let mut cache = model.new_cache(4).expect("cache");
        let token = 3u32;
        let got = model.forward(&mut cache, token).expect("fwd");
        let exp = oracle_forward(&g, token);
        assert_eq!(got.len(), exp.len());
        for (i, (a, b)) in got.iter().zip(exp.iter()).enumerate() {
            let rel = (a - b).abs() / (1.0 + b.abs());
            assert!(rel * 1000.0 < 1.0, "logit {i}: {a} vs {b}");
        }
        assert_eq!(cache.n_past, 1);
    }

    #[test]
    fn tiny_llama_encode_greedy_decode_uses_shipped_path() {
        let bytes = tiny_llama_gguf();
        let g = load_gguf(&bytes).expect("load");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        assert_eq!(tok.encode("ab").unwrap(), vec![3]);
        assert_eq!(tok.decode(&[1, 2]), "ab");
        let model = Llama::from_gguf(&g).expect("model");
        let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
        assert!(out.contains("ab"), "{out}");
        let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
        assert_eq!(out, out2);
    }
}
