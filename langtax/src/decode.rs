//! Llama-family decode on GGUF bytes: RMSNorm, RoPE, GQA+KV, SwiGLU or Gemma GeGLU, lm_head.

use crate::gguf::{GgmlType, Gguf, GgufError, Kv, Tensor, TensorWrite};
use crate::quant::{
    bf16_row_bytes, dequant_bf16_row, dequant_f16_row, dequant_f32_row, dequant_iq1_m_row,
    dequant_iq1_s_row, dequant_iq2_s_row, dequant_iq2_xs_row, dequant_iq2_xxs_row,
    dequant_iq3_s_row, dequant_iq3_xxs_row, dequant_iq4_nl_row, dequant_iq4_xs_row,
    dequant_mxfp4_row, dequant_nvfp4_row, dequant_q1_0_row, dequant_q2_0_row, dequant_q2_k_row,
    dequant_q3_k_row, dequant_q4_1_row, dequant_q4_k_row, dequant_q5_0_row, dequant_q5_1_row,
    dequant_q5_k_row, dequant_q6_k_row, dequant_q8_1_row, dequant_tq1_0_row, dequant_tq2_0_row,
    f16_row_bytes, f32_row_bytes, gemm_bf16, gemm_f16, gemm_f32, gemm_iq1_m_f32, gemm_iq1_s_f32,
    gemm_iq2_s_f32, gemm_iq2_xs_f32, gemm_iq2_xxs_f32, gemm_iq3_s_f32, gemm_iq3_xxs_f32,
    gemm_iq4_nl_f32, gemm_iq4_xs_f32, gemm_mxfp4_f32, gemm_nvfp4_f32, gemm_q1_0_f32, gemm_q2_0_f32,
    gemm_q2_k_f32, gemm_q3_k_f32, gemm_q4_1_f32, gemm_q4_k_f32, gemm_q5_0_f32, gemm_q5_1_f32,
    gemm_q5_k_f32, gemm_q6_k_f32, gemm_q8_1_f32, gemm_tq1_0_f32, gemm_tq2_0_f32, gemv_bf16,
    gemv_f16, gemv_f32, gemv_iq1_m_f32, gemv_iq1_s_f32, gemv_iq2_s_f32, gemv_iq2_xs_f32,
    gemv_iq2_xxs_f32, gemv_iq3_s_f32, gemv_iq3_xxs_f32, gemv_iq4_nl_f32, gemv_iq4_xs_f32,
    gemv_mxfp4_f32, gemv_nvfp4_f32, gemv_q1_0_f32, gemv_q2_0_f32, gemv_q2_k_f32, gemv_q3_k_f32,
    gemv_q4_1_f32, gemv_q4_k_f32, gemv_q5_0_f32, gemv_q5_1_f32, gemv_q5_k_f32, gemv_q6_k_f32,
    gemv_q8_1_f32, gemv_tq1_0_f32, gemv_tq2_0_f32, iq1_m_row_bytes, iq1_s_row_bytes,
    iq2_s_row_bytes, iq2_xs_row_bytes, iq2_xxs_row_bytes, iq3_s_row_bytes, iq3_xxs_row_bytes,
    iq4_nl_row_bytes, iq4_xs_row_bytes, mxfp4_row_bytes, nvfp4_row_bytes, pack_bf16, pack_f16,
    pack_f32, pack_iq1_m_block, pack_iq1_s_block, pack_iq2_s_block, pack_iq2_xs_block,
    pack_iq2_xxs_block, pack_iq3_s_block, pack_iq3_xxs_block, pack_iq4_nl_block, pack_iq4_xs_block,
    pack_mxfp4_block, pack_nvfp4_block, pack_q1_0_block, pack_q2_0_block, pack_q2_k_block,
    pack_q3_k_block, pack_q4_1_block, pack_q4_k_block, pack_q5_0_block, pack_q5_1_block,
    pack_q5_k_block, pack_q6_k_block, pack_q8_1_block, pack_tq1_0_block, pack_tq2_0_block,
    q1_0_row_bytes, q2_0_row_bytes, q2_k_row_bytes, q3_k_row_bytes, q4_1_row_bytes, q4_k_row_bytes,
    q5_0_row_bytes, q5_1_row_bytes, q5_k_row_bytes, q6_k_row_bytes, q8_1_row_bytes,
    tq1_0_row_bytes, tq2_0_row_bytes, QuantError, QK1_0, QK2_0, QK4_1, QK4_NL, QK5_0, QK5_1, QK8_1,
    QK_K, QK_MXFP4, QK_NVFP4,
};
use crate::sample::{SampleError, SampleParams, Sampler};
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
    /// `--prompt` was empty (no tokens to decode).
    EmptyPrompt,
    /// Sampling failed.
    Sample(SampleError),
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
            Self::EmptyPrompt => write!(f, "empty prompt"),
            Self::Sample(e) => write!(f, "{e}"),
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

impl From<SampleError> for LlamaError {
    fn from(v: SampleError) -> Self {
        Self::Sample(v)
    }
}

struct QuantMat {
    name: String,
    ty: GgmlType,
    n_cols: usize,
    n_rows: usize,
    start: usize,
    end: usize,
}

/// Reuse `token_embd.weight` as the lm_head when `output.weight` is absent.
/// Copies only the range metadata (same on-disk bytes). Does not clone the matrix.
fn reuse_token_embd_as_output(token_embd: &QuantMat) -> QuantMat {
    QuantMat {
        name: token_embd.name.clone(),
        ty: token_embd.ty,
        n_cols: token_embd.n_cols,
        n_rows: token_embd.n_rows,
        start: token_embd.start,
        end: token_embd.end,
    }
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

/// Loaded Llama-family weights. Quantized matrices are ranges of one file blob.
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
    /// `1` for llama/qwen2/mistral/phi3. Gemma official walk scales embeds by `sqrt(n_embd)`.
    embed_scale: f32,
    /// Official Gemma FFN is `LLM_FFN_GELU` (GeGLU). Other loaded arches stay SwiGLU.
    ffn_gelu: bool,
    blob: Vec<u8>,
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
    /// `phi3`, or `gemma`) and `blk.{i}.*` tensor names.
    ///
    /// Takes the GGUF's file blob once. Weight matrices keep offsets into that
    /// blob; they do not clone tensor bytes. When `output.weight` is absent,
    /// the lm_head reuses the `token_embd.weight` range (same on-disk bytes).
    pub fn from_gguf(g: Gguf) -> Result<Self, LlamaError> {
        let arch = architecture(&g)?;
        let n_layer = require_usize(&g, arch, "block_count")?;
        let n_embd = require_usize(&g, arch, "embedding_length")?;
        let n_head = require_usize(&g, arch, "attention.head_count")?;
        let n_head_kv = require_usize(&g, arch, "attention.head_count_kv")?;
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
        let n_rot = rope_dimension(&g, arch, n_embd, n_head)?;
        let n_vocab = g
            .tensor("token_embd.weight")
            .map(|t| t.n_rows())
            .ok_or_else(|| LlamaError::Tensor("token_embd.weight".into()))?;
        let rms_eps = arch_f32(&g, arch, "attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let rope_base = arch_f32(&g, arch, "rope.freq_base").unwrap_or(10_000.0);
        let gemma = arch == "gemma";
        let n_embd_f = f32::from(u16::try_from(n_embd).unwrap_or(1));
        let embed_scale = if gemma { n_embd_f.sqrt() } else { 1.0 };
        let token_embd = quant_mat(need(&g, "token_embd.weight")?)?;
        let output_norm = f32s(need(&g, "output_norm.weight")?)?;
        let output = match g.tensor("output.weight") {
            Some(t) => quant_mat(t)?,
            None => reuse_token_embd_as_output(&token_embd),
        };
        let mut layers = Vec::new();
        for i in 0..n_layer {
            layers.push(load_layer(&g, i)?);
        }
        Ok(Self {
            n_vocab,
            n_embd,
            n_head,
            n_head_kv,
            n_rot,
            rms_eps,
            rope_base,
            embed_scale,
            ffn_gelu: gemma,
            blob: g.into_blob(),
            token_embd,
            output_norm,
            output,
            layers,
        })
    }

    /// Byte length of the single owned GGUF blob. Weight payloads are ranges of it.
    pub fn blob_len(&self) -> usize {
        self.blob.len()
    }

    fn mat_bytes(&self, m: &QuantMat) -> Result<&[u8], LlamaError> {
        self.blob
            .get(m.start..m.end)
            .ok_or_else(|| LlamaError::Shape(m.name.clone()))
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
        self.prefill(cache, &[token])
    }

    /// Decode `tokens` in one causal pass. Prompt tokens share each weight
    /// row (GEMM; a single token stays GEMV). Writes K/V at
    /// `cache.n_past .. n_past+len` and returns logits of the last token.
    pub fn prefill(&self, cache: &mut KvCache, tokens: &[u32]) -> Result<Vec<f32>, LlamaError> {
        let n = tokens.len();
        if n == 0 {
            return Err(LlamaError::Shape("prefill tokens".into()));
        }
        let hd = self.head_dim()?;
        let n0 = cache.n_past;
        let end = n0
            .checked_add(n)
            .ok_or_else(|| LlamaError::Shape("kv cache full".into()))?;
        if end > cache.max_seq {
            return Err(LlamaError::Shape("kv cache full".into()));
        }
        let mut x = Vec::new();
        for tok in tokens {
            x.extend(self.embed(*tok)?);
        }
        if x.len() != n.saturating_mul(self.n_embd) {
            return Err(LlamaError::Shape("prefill embed".into()));
        }
        for v in &mut x {
            *v *= self.embed_scale;
        }
        for (li, layer) in self.layers.iter().enumerate() {
            let residual = x.clone();
            x = rmsnorm_rows(&x, self.n_embd, &layer.attn_norm, self.rms_eps)?;
            let q = add_bias_rows(
                self.gemm_mat(&layer.wq, n, &x)?,
                layer.wq.n_rows,
                layer.bq.as_deref(),
            )?;
            let k = add_bias_rows(
                self.gemm_mat(&layer.wk, n, &x)?,
                layer.wk.n_rows,
                layer.bk.as_deref(),
            )?;
            let v = add_bias_rows(
                self.gemm_mat(&layer.wv, n, &x)?,
                layer.wv.n_rows,
                layer.bv.as_deref(),
            )?;
            for t in 0..n {
                let pos = n0.saturating_add(t);
                let k_t = token_row(&k, t, layer.wk.n_rows, "prefill k")?;
                let v_t = token_row(&v, t, layer.wv.n_rows, "prefill v")?;
                let mut kh = split_heads(k_t, self.n_head_kv, hd)?;
                let vh = split_heads(v_t, self.n_head_kv, hd)?;
                for h in &mut kh {
                    rope(h, pos, self.n_rot, self.rope_base)?;
                }
                store_kv(&mut cache.k, li, self.n_head_kv, cache.max_seq, pos, &kh)?;
                store_kv(&mut cache.v, li, self.n_head_kv, cache.max_seq, pos, &vh)?;
            }
            let mut attn = vec![0.0f32; n.saturating_mul(self.n_embd)];
            for t in 0..n {
                let pos = n0.saturating_add(t);
                let q_t = token_row(&q, t, layer.wq.n_rows, "prefill q")?;
                let mut qh = split_heads(q_t, self.n_head, hd)?;
                for h in &mut qh {
                    rope(h, pos, self.n_rot, self.rope_base)?;
                }
                let one = attend_query(cache, li, &qh, self.n_head_kv, hd, pos.saturating_add(1))?;
                let dst_off = t.saturating_mul(self.n_embd);
                let dst = attn
                    .get_mut(dst_off..dst_off.saturating_add(self.n_embd))
                    .ok_or_else(|| LlamaError::Shape("prefill attn".into()))?;
                for (d, s) in dst.iter_mut().zip(one.iter()) {
                    *d = *s;
                }
            }
            let proj = self.gemm_mat(&layer.wo, n, &attn)?;
            x = add(&proj, &residual)?;
            let residual = x.clone();
            x = rmsnorm_rows(&x, self.n_embd, &layer.ffn_norm, self.rms_eps)?;
            let gate = self.gemm_mat(&layer.gate, n, &x)?;
            let up = self.gemm_mat(&layer.up, n, &x)?;
            let mut h = ffn_gate_act(&gate, self.ffn_gelu)?;
            for (hv, uv) in h.iter_mut().zip(up.iter()) {
                *hv *= *uv;
            }
            let down = self.gemm_mat(&layer.down, n, &h)?;
            x = add(&down, &residual)?;
        }
        let last_off = n.saturating_sub(1).saturating_mul(self.n_embd);
        let last = x
            .get(last_off..last_off.saturating_add(self.n_embd))
            .ok_or_else(|| LlamaError::Shape("prefill last".into()))?;
        let xn = rmsnorm(last, &self.output_norm, self.rms_eps)?;
        let logits = self.gemv_mat(&self.output, &xn)?;
        cache.n_past = end;
        Ok(logits)
    }
}

/// Greedy generate: encode prompt, decode `n_predict` tokens, return decoded string.
///
/// KV is sized to `prompt + n_predict + 1`. See [`greedy_generate_ctx`] to set `--n-ctx`.
pub fn greedy_generate(
    model: &Llama,
    tok: &Tokenizer,
    prompt: &str,
    n_predict: usize,
) -> Result<String, LlamaError> {
    greedy_generate_ctx(model, tok, prompt, n_predict, None)
}

/// Seedless greedy generate with optional KV capacity (`--n-ctx`).
///
/// `n_ctx` must be at least prompt tokens + `n_predict`. `None` uses prompt + `n_predict` + 1.
pub fn greedy_generate_ctx(
    model: &Llama,
    tok: &Tokenizer,
    prompt: &str,
    n_predict: usize,
    n_ctx: Option<usize>,
) -> Result<String, LlamaError> {
    if prompt.is_empty() {
        return Err(LlamaError::EmptyPrompt);
    }
    let mut ids = prompt_ids(tok, prompt)?;
    if ids.is_empty() {
        return Err(LlamaError::EmptyPrompt);
    }
    let needed = ids.len().saturating_add(n_predict);
    let max_seq = match n_ctx {
        Some(n) if n < needed => return Err(LlamaError::Shape("n_ctx".into())),
        Some(n) => n,
        None => needed.saturating_add(1),
    };
    let mut cache = model.new_cache(max_seq)?;
    let mut last = model.prefill(&mut cache, &ids)?;
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

/// Generate with [`SampleParams`]. [`SampleParams::greedy`] is the seedless path.
pub fn generate(
    model: &Llama,
    tok: &Tokenizer,
    prompt: &str,
    n_predict: usize,
    params: &SampleParams,
) -> Result<String, LlamaError> {
    generate_ctx(model, tok, prompt, n_predict, None, params)
}

/// [`generate`] with optional KV capacity (`n_ctx`).
pub fn generate_ctx(
    model: &Llama,
    tok: &Tokenizer,
    prompt: &str,
    n_predict: usize,
    n_ctx: Option<usize>,
    params: &SampleParams,
) -> Result<String, LlamaError> {
    if prompt.is_empty() {
        return Err(LlamaError::EmptyPrompt);
    }
    let mut ids = prompt_ids(tok, prompt)?;
    if ids.is_empty() {
        return Err(LlamaError::EmptyPrompt);
    }
    let needed = ids.len().saturating_add(n_predict);
    let max_seq = match n_ctx {
        Some(n) if n < needed => return Err(LlamaError::Shape("n_ctx".into())),
        Some(n) => n,
        None => needed.saturating_add(1),
    };
    let mut cache = model.new_cache(max_seq)?;
    let mut last = model.prefill(&mut cache, &ids)?;
    let mut sampler = Sampler::new(*params)?;
    for _ in 0..n_predict {
        let next = sampler.sample(&last, &ids)?;
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

fn tiny_llama_spec() -> TinySpec {
    TinySpec {
        arch: "llama",
        token_embd: GgmlType::F32,
        output: GgmlType::F32,
        layer: None,
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    }
}

/// Writer-built Llama-shaped GGUF: mixed F32/Q4_K/Q6_K weights + tokenizer KV.
pub fn tiny_llama_gguf() -> Vec<u8> {
    tiny_arch_gguf(tiny_llama_spec())
}

/// Writer-built Llama GGUF that omits `output.weight` (tied embeddings).
///
/// Load reuses the already-loaded `token_embd.weight` range. Same on-disk
/// bytes; no matrix clone.
pub fn tiny_tied_gguf() -> Vec<u8> {
    tiny_arch_gguf_lm_head(tiny_llama_spec(), TinyLmHead::Tied)
}

/// Writer-built Llama GGUF whose `output.weight` is an identical copy of
/// `token_embd.weight` (same type, same pack seed). Untied control for the
/// tied-reuse oracle.
pub fn tiny_tied_copy_gguf() -> Vec<u8> {
    tiny_arch_gguf_lm_head(tiny_llama_spec(), TinyLmHead::CopyTokenEmbd)
}

/// Writer-built Qwen2-shaped GGUF: `qwen2.*` KV, no `rope.dimension_count`, QKV bias,
/// `add_bos_token=false`.
pub fn tiny_qwen2_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "qwen2",
        token_embd: GgmlType::F32,
        output: GgmlType::F32,
        layer: None,
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
        layer: None,
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
        layer: None,
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Gemma-shaped GGUF: same tensors as [`tiny_llama_gguf`], `gemma.*` KV.
///
/// Official `general.architecture=gemma` (not `gemma2`). Decode uses the
/// measured Gemma walk: embed `* sqrt(n_embd)`, GeGLU (`ggml_gelu`), RMSNorm
/// on GGUF bytes as-is (convert-hf bakes `norm.weight + 1`; runtime does not
/// add 1 again).
pub fn tiny_gemma_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "gemma",
        token_embd: GgmlType::F32,
        output: GgmlType::F32,
        layer: None,
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
        layer: None,
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
        layer: None,
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with F16 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32, matching common OSS F16 GGUF files from convert-hf-to-gguf.
/// See [`tiny_f16_1d_gguf`] for F16 1-D norms (same 2-D bytes, F32-norm twin).
pub fn tiny_f16_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::F16,
        output: GgmlType::F16,
        layer: Some(GgmlType::F16),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with F16 2-D weights and F16 1-D norms.
///
/// On-disk IEEE binary16 (`GGML_TYPE_F16` = 1). Load walks those bytes with
/// the same `ggml_fp16_to_fp32` scalar used for 2-D F16, then applies the
/// existing F32 RMSNorm. F32-norm twin is [`tiny_f16_gguf`].
pub fn tiny_f16_1d_gguf() -> Vec<u8> {
    tiny_arch_gguf_lm_head_vec1d(
        TinySpec {
            arch: "llama",
            token_embd: GgmlType::F16,
            output: GgmlType::F16,
            layer: Some(GgmlType::F16),
            rope_dimension_count: true,
            qkv_bias: false,
            add_bos_token: None,
        },
        TinyLmHead::Distinct,
        GgmlType::F16,
    )
}

/// Writer-built Qwen2-shaped GGUF with F16 1-D norms and F16 QKV bias.
///
/// Same optional-bias writer path as [`tiny_qwen2_gguf`]; 1-D tensors are
/// on-disk IEEE binary16. 2-D weights stay the mixed Q4_K_M mix.
pub fn tiny_f16_1d_bias_gguf() -> Vec<u8> {
    tiny_arch_gguf_lm_head_vec1d(
        TinySpec {
            arch: "qwen2",
            token_embd: GgmlType::F32,
            output: GgmlType::F32,
            layer: None,
            rope_dimension_count: false,
            qkv_bias: true,
            add_bos_token: Some(false),
        },
        TinyLmHead::Distinct,
        GgmlType::F16,
    )
}

/// Writer-built Llama GGUF with BF16 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_BF16` = 30. Common OSS BF16 GGUF files from
/// convert-hf-to-gguf use this type for 2-D weights.
pub fn tiny_bf16_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::BF16,
        output: GgmlType::BF16,
        layer: Some(GgmlType::BF16),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with Q2_K 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_Q2_K` = 10.
pub fn tiny_q2k_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::Q2_K,
        output: GgmlType::Q2_K,
        layer: Some(GgmlType::Q2_K),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with Q3_K 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_Q3_K` = 11.
pub fn tiny_q3k_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::Q3_K,
        output: GgmlType::Q3_K,
        layer: Some(GgmlType::Q3_K),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with Q4_1 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_Q4_1` = 3.
pub fn tiny_q41_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::Q4_1,
        output: GgmlType::Q4_1,
        layer: Some(GgmlType::Q4_1),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with Q5_0 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_Q5_0` = 6.
pub fn tiny_q50_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::Q5_0,
        output: GgmlType::Q5_0,
        layer: Some(GgmlType::Q5_0),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with Q5_1 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_Q5_1` = 7.
pub fn tiny_q51_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::Q5_1,
        output: GgmlType::Q5_1,
        layer: Some(GgmlType::Q5_1),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with MXFP4 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_MXFP4` = 39.
pub fn tiny_mxfp4_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::MXFP4,
        output: GgmlType::MXFP4,
        layer: Some(GgmlType::MXFP4),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with NVFP4 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_NVFP4` = 40.
pub fn tiny_nvfp4_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::NVFP4,
        output: GgmlType::NVFP4,
        layer: Some(GgmlType::NVFP4),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with Q1_0 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_Q1_0` = 41.
pub fn tiny_q10_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::Q1_0,
        output: GgmlType::Q1_0,
        layer: Some(GgmlType::Q1_0),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with Q2_0 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_Q2_0` = 42.
pub fn tiny_q20_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::Q2_0,
        output: GgmlType::Q2_0,
        layer: Some(GgmlType::Q2_0),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with Q8_1 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_Q8_1` = 9.
pub fn tiny_q81_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::Q8_1,
        output: GgmlType::Q8_1,
        layer: Some(GgmlType::Q8_1),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with TQ1_0 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_TQ1_0` = 34.
pub fn tiny_tq10_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::TQ1_0,
        output: GgmlType::TQ1_0,
        layer: Some(GgmlType::TQ1_0),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with TQ2_0 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_TQ2_0` = 35.
pub fn tiny_tq20_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::TQ2_0,
        output: GgmlType::TQ2_0,
        layer: Some(GgmlType::TQ2_0),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with Q5_K 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_Q5_K` = 13.
pub fn tiny_q5k_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::Q5_K,
        output: GgmlType::Q5_K,
        layer: Some(GgmlType::Q5_K),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with IQ4_NL 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_IQ4_NL` = 20. Common OSS `*-IQ4_NL.gguf` files
/// use this type for 32-wide 2-D weights (standalone and mixed IQ*_M).
pub fn tiny_iq4nl_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::IQ4_NL,
        output: GgmlType::IQ4_NL,
        layer: Some(GgmlType::IQ4_NL),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with IQ2_XXS 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_IQ2_XXS` = 16. Common OSS `*-IQ2_XXS.gguf`
/// files use this type for 256-wide 2-D weights.
pub fn tiny_iq2xxs_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::IQ2_XXS,
        output: GgmlType::IQ2_XXS,
        layer: Some(GgmlType::IQ2_XXS),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with IQ2_XS 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_IQ2_XS` = 17. Common OSS `*-IQ2_XS.gguf`
/// files use this type for 256-wide 2-D weights.
pub fn tiny_iq2xs_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::IQ2_XS,
        output: GgmlType::IQ2_XS,
        layer: Some(GgmlType::IQ2_XS),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with IQ1_S 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_IQ1_S` = 19. Common OSS `*-IQ1_S.gguf`
/// files use this type for 256-wide 2-D weights.
pub fn tiny_iq1s_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::IQ1_S,
        output: GgmlType::IQ1_S,
        layer: Some(GgmlType::IQ1_S),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with IQ1_M 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_IQ1_M` = 29. Common OSS `*-IQ1_M.gguf`
/// files use this type for 256-wide 2-D weights.
pub fn tiny_iq1m_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::IQ1_M,
        output: GgmlType::IQ1_M,
        layer: Some(GgmlType::IQ1_M),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with IQ2_S 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_IQ2_S` = 22. Common OSS `*-IQ2_S.gguf` files
/// and mixed `*-IQ2_M.gguf` tensors use this type for 256-wide 2-D weights.
pub fn tiny_iq2s_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::IQ2_S,
        output: GgmlType::IQ2_S,
        layer: Some(GgmlType::IQ2_S),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with IQ3_XXS 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_IQ3_XXS` = 18. Common OSS `*-IQ3_XXS.gguf`
/// files and mixed `*-IQ3_XS.gguf` tensors use this type for 256-wide 2-D weights.
pub fn tiny_iq3xxs_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::IQ3_XXS,
        output: GgmlType::IQ3_XXS,
        layer: Some(GgmlType::IQ3_XXS),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with IQ3_S 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_IQ3_S` = 21. Common OSS `*-IQ3_S.gguf` files
/// and mixed `*-IQ3_M.gguf` tensors use this type for 256-wide 2-D weights.
pub fn tiny_iq3s_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::IQ3_S,
        output: GgmlType::IQ3_S,
        layer: Some(GgmlType::IQ3_S),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

/// Writer-built Llama GGUF with IQ4_XS 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_IQ4_XS` = 23. Common OSS `*-IQ4_XS.gguf` files
/// use this type for 256-wide 2-D weights.
pub fn tiny_iq4xs_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::IQ4_XS,
        output: GgmlType::IQ4_XS,
        layer: Some(GgmlType::IQ4_XS),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
    })
}

struct TinySpec {
    arch: &'static str,
    token_embd: GgmlType,
    output: GgmlType,
    /// When set, every 2-D layer weight uses this type. Otherwise the mixed
    /// Q4_K / Q6_K / F32 mix used by [`tiny_llama_gguf`]. Q2_K / Q3_K / Q4_1 / Q5_0 / Q5_1 / Q5_K / IQ1_M /
    /// IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ4_NL / IQ4_XS /
    /// BF16 / MXFP4 / NVFP4 / Q1_0 / Q2_0 / Q8_1 / TQ1_0 / TQ2_0 are used by [`tiny_q2k_gguf`] / [`tiny_q3k_gguf`] / [`tiny_q41_gguf`] /
    /// [`tiny_q50_gguf`] / [`tiny_q51_gguf`] / [`tiny_q5k_gguf`] /
    /// [`tiny_iq1m_gguf`] / [`tiny_iq1s_gguf`] / [`tiny_iq2xxs_gguf`] /
    /// [`tiny_iq2xs_gguf`] / [`tiny_iq2s_gguf`] / [`tiny_iq3xxs_gguf`] /
    /// [`tiny_iq3s_gguf`] / [`tiny_iq4nl_gguf`] / [`tiny_iq4xs_gguf`] /
    /// [`tiny_bf16_gguf`] / [`tiny_mxfp4_gguf`] / [`tiny_nvfp4_gguf`] / [`tiny_q10_gguf`] /
    /// [`tiny_q20_gguf`] / [`tiny_q81_gguf`] / [`tiny_tq10_gguf`] / [`tiny_tq20_gguf`].
    layer: Option<GgmlType>,
    rope_dimension_count: bool,
    qkv_bias: bool,
    add_bos_token: Option<bool>,
}

/// How the writer emits `output.weight` relative to `token_embd.weight`.
enum TinyLmHead {
    /// Distinct `output.weight` packed with seed 2 (existing tiny files).
    Distinct,
    /// Omit `output.weight` so load reuses `token_embd.weight`.
    Tied,
    /// Write `output.weight` as an identical copy of `token_embd.weight` (seed 1).
    CopyTokenEmbd,
}

fn tiny_arch_gguf(spec: TinySpec) -> Vec<u8> {
    tiny_arch_gguf_lm_head(spec, TinyLmHead::Distinct)
}

fn tiny_arch_gguf_lm_head(spec: TinySpec, lm_head: TinyLmHead) -> Vec<u8> {
    tiny_arch_gguf_lm_head_vec1d(spec, lm_head, GgmlType::F32)
}

fn tiny_arch_gguf_lm_head_vec1d(spec: TinySpec, lm_head: TinyLmHead, vec1d: GgmlType) -> Vec<u8> {
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
            vec1d,
            vec![n_embd],
            pack_vec1d(vec1d, &ones),
        ),
    ];
    match lm_head {
        TinyLmHead::Tied => {}
        TinyLmHead::Distinct => tensors.push(tw(
            "output.weight",
            spec.output,
            vec![n_embd, n_vocab],
            pack_mat(spec.output, n_embd, n_vocab, 2),
        )),
        TinyLmHead::CopyTokenEmbd => tensors.push(tw(
            "output.weight",
            spec.token_embd,
            vec![n_embd, n_vocab],
            pack_mat(spec.token_embd, n_embd, n_vocab, 1),
        )),
    }
    tensors.push(tw(
        "blk.0.attn_norm.weight",
        vec1d,
        vec![n_embd],
        pack_vec1d(vec1d, &ones),
    ));
    tensors.push(tw(
        "blk.0.ffn_norm.weight",
        vec1d,
        vec![n_embd],
        pack_vec1d(vec1d, &ones),
    ));
    tensors.push(layer_tw(
        &spec,
        "blk.0.attn_k.weight",
        n_embd,
        n_kv,
        3,
        GgmlType::F32,
    ));
    tensors.push(layer_tw(
        &spec,
        "blk.0.attn_v.weight",
        n_embd,
        n_kv,
        4,
        GgmlType::F32,
    ));
    tensors.push(layer_tw(
        &spec,
        "blk.0.attn_q.weight",
        n_embd,
        n_embd,
        5,
        GgmlType::Q4_K,
    ));
    tensors.push(layer_tw(
        &spec,
        "blk.0.attn_output.weight",
        n_embd,
        n_embd,
        6,
        GgmlType::Q4_K,
    ));
    tensors.push(layer_tw(
        &spec,
        "blk.0.ffn_up.weight",
        n_embd,
        n_ff,
        7,
        GgmlType::Q4_K,
    ));
    tensors.push(layer_tw(
        &spec,
        "blk.0.ffn_down.weight",
        n_ff,
        n_embd,
        8,
        GgmlType::Q4_K,
    ));
    tensors.push(layer_tw(
        &spec,
        "blk.0.ffn_gate.weight",
        n_embd,
        n_ff,
        9,
        GgmlType::Q6_K,
    ));
    if spec.qkv_bias {
        tensors.push(tw(
            "blk.0.attn_q.bias",
            vec1d,
            vec![n_embd],
            pack_vec1d(vec1d, &pat_f32(n_embd, 11)),
        ));
        tensors.push(tw(
            "blk.0.attn_k.bias",
            vec1d,
            vec![n_kv],
            pack_vec1d(vec1d, &pat_f32(n_kv, 12)),
        ));
        tensors.push(tw(
            "blk.0.attn_v.bias",
            vec1d,
            vec![n_kv],
            pack_vec1d(vec1d, &pat_f32(n_kv, 13)),
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
    s == "llama" || s == "qwen2" || s == "mistral" || s == "phi3" || s == "gemma"
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

fn layer_tw(
    spec: &TinySpec,
    name: &str,
    n_cols: usize,
    n_rows: usize,
    seed: u32,
    mixed: GgmlType,
) -> TensorWrite {
    let ty = spec.layer.unwrap_or(mixed);
    tw(
        name,
        ty,
        vec![n_cols, n_rows],
        pack_mat(ty, n_cols, n_rows, seed),
    )
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

fn pack_vec1d(ty: GgmlType, values: &[f32]) -> Vec<u8> {
    match ty {
        GgmlType::F16 => pack_f16(values),
        _ => pack_f32(values),
    }
}

fn pack_mat(ty: GgmlType, n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    match ty {
        GgmlType::Q2_K => pack_q2k_mat(n_cols, n_rows, seed),
        GgmlType::Q3_K => pack_q3k_mat(n_cols, n_rows, seed),
        GgmlType::Q4_1 => pack_q41_mat(n_cols, n_rows, seed),
        GgmlType::Q5_0 => pack_q50_mat(n_cols, n_rows, seed),
        GgmlType::Q5_1 => pack_q51_mat(n_cols, n_rows, seed),
        GgmlType::MXFP4 => pack_mxfp4_mat(n_cols, n_rows, seed),
        GgmlType::NVFP4 => pack_nvfp4_mat(n_cols, n_rows, seed),
        GgmlType::Q1_0 => pack_q10_mat(n_cols, n_rows, seed),
        GgmlType::Q2_0 => pack_q20_mat(n_cols, n_rows, seed),
        GgmlType::Q8_1 => pack_q81_mat(n_cols, n_rows, seed),
        GgmlType::TQ1_0 => pack_tq10_mat(n_cols, n_rows, seed),
        GgmlType::TQ2_0 => pack_tq20_mat(n_cols, n_rows, seed),
        GgmlType::Q4_K => pack_q4k_mat(n_cols, n_rows, seed),
        GgmlType::Q5_K => pack_q5k_mat(n_cols, n_rows, seed),
        GgmlType::Q6_K => pack_q6k_mat(n_cols, n_rows, seed),
        GgmlType::IQ1_S => pack_iq1s_mat(n_cols, n_rows, seed),
        GgmlType::IQ1_M => pack_iq1m_mat(n_cols, n_rows, seed),
        GgmlType::IQ2_XXS => pack_iq2xxs_mat(n_cols, n_rows, seed),
        GgmlType::IQ2_XS => pack_iq2xs_mat(n_cols, n_rows, seed),
        GgmlType::IQ2_S => pack_iq2s_mat(n_cols, n_rows, seed),
        GgmlType::IQ3_XXS => pack_iq3xxs_mat(n_cols, n_rows, seed),
        GgmlType::IQ3_S => pack_iq3s_mat(n_cols, n_rows, seed),
        GgmlType::IQ4_NL => pack_iq4nl_mat(n_cols, n_rows, seed),
        GgmlType::IQ4_XS => pack_iq4xs_mat(n_cols, n_rows, seed),
        GgmlType::F16 => pack_f16(&pat_f32(n_cols.saturating_mul(n_rows), seed)),
        GgmlType::BF16 => pack_bf16(&pat_f32(n_cols.saturating_mul(n_rows), seed)),
        _ => pack_f32(&pat_f32(n_cols.saturating_mul(n_rows), seed)),
    }
}

fn pack_iq2xxs_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    for _ in 0..n_rows {
        let mut qs = [0u8; 32];
        for q in &mut qs {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *q = u8::try_from(s % 256).unwrap_or(0);
        }
        let mut signs = [0u8; 32];
        for c in &mut signs {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *c = u8::try_from(s % 128).unwrap_or(0);
        }
        let mut sc = [1u8; 8];
        for c in &mut sc {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *c = u8::try_from(s % 8).unwrap_or(1);
        }
        out.extend_from_slice(&pack_iq2_xxs_block(25.0 / 100.0, &sc, &qs, &signs));
        let _ = n_cols;
    }
    out
}

fn pack_iq2xs_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    for _ in 0..n_rows {
        let mut qs = [0u16; 32];
        for q in &mut qs {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *q = u16::try_from(s % 512).unwrap_or(0);
        }
        let mut signs = [0u8; 32];
        for c in &mut signs {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *c = u8::try_from(s % 128).unwrap_or(0);
        }
        let mut sc = [1u8; 16];
        for c in &mut sc {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *c = u8::try_from(s % 8).unwrap_or(1);
        }
        out.extend_from_slice(&pack_iq2_xs_block(25.0 / 100.0, &sc, &qs, &signs));
        let _ = n_cols;
    }
    out
}

fn pack_iq1s_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    for _ in 0..n_rows {
        let mut qs = [0u16; 32];
        for q in &mut qs {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *q = u16::try_from(s % 2048).unwrap_or(0);
        }
        let mut sc = [1u8; 8];
        for c in &mut sc {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *c = u8::try_from(s % 8).unwrap_or(1);
        }
        let mut delta_neg = [0u8; 8];
        for c in &mut delta_neg {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *c = u8::try_from(s % 2).unwrap_or(0);
        }
        out.extend_from_slice(&pack_iq1_s_block(25.0 / 100.0, &sc, &qs, &delta_neg));
        let _ = n_cols;
    }
    out
}

fn pack_iq1m_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    for _ in 0..n_rows {
        let mut qs = [0u16; 32];
        for q in &mut qs {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *q = u16::try_from(s % 2048).unwrap_or(0);
        }
        let mut sc = [1u8; 16];
        for c in &mut sc {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *c = u8::try_from(s % 8).unwrap_or(1);
        }
        let mut delta_neg = [0u8; 32];
        for c in &mut delta_neg {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *c = u8::try_from(s % 2).unwrap_or(0);
        }
        out.extend_from_slice(&pack_iq1_m_block(25.0 / 100.0, &sc, &qs, &delta_neg));
        let _ = n_cols;
    }
    out
}

fn pack_iq2s_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    for _ in 0..n_rows {
        let mut qs = [0u16; 32];
        for q in &mut qs {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *q = u16::try_from(s % 1024).unwrap_or(0);
        }
        let mut signs = [0u8; 32];
        for c in &mut signs {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *c = u8::try_from(s % 256).unwrap_or(0);
        }
        let mut sc = [1u8; 16];
        for c in &mut sc {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *c = u8::try_from(s % 8).unwrap_or(1);
        }
        out.extend_from_slice(&pack_iq2_s_block(25.0 / 100.0, &sc, &qs, &signs));
        let _ = n_cols;
    }
    out
}

fn pack_iq3xxs_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    for _ in 0..n_rows {
        let mut qs = [0u8; 64];
        for q in &mut qs {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *q = u8::try_from(s % 256).unwrap_or(0);
        }
        let mut signs = [0u8; 32];
        for c in &mut signs {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *c = u8::try_from(s % 128).unwrap_or(0);
        }
        let mut sc = [1u8; 8];
        for c in &mut sc {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *c = u8::try_from(s % 8).unwrap_or(1);
        }
        out.extend_from_slice(&pack_iq3_xxs_block(25.0 / 100.0, &sc, &qs, &signs));
        let _ = n_cols;
    }
    out
}

fn pack_iq3s_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    for _ in 0..n_rows {
        let mut qs = [0u16; 64];
        for q in &mut qs {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *q = u16::try_from(s % 512).unwrap_or(0);
        }
        let mut signs = [0u8; 32];
        for c in &mut signs {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *c = u8::try_from(s % 256).unwrap_or(0);
        }
        let mut sc = [1u8; 8];
        for c in &mut sc {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *c = u8::try_from(s % 8).unwrap_or(1);
        }
        out.extend_from_slice(&pack_iq3_s_block(25.0 / 100.0, &sc, &qs, &signs));
        let _ = n_cols;
    }
    out
}

fn pack_iq4nl_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    let nblocks = n_cols / QK4_NL;
    for _ in 0..n_rows {
        for _ in 0..nblocks {
            let mut qs = [0u8; QK4_NL];
            for q in &mut qs {
                s = s.wrapping_mul(1_664_525).wrapping_add(1);
                *q = u8::try_from(s % 16).unwrap_or(0);
            }
            out.extend_from_slice(&pack_iq4_nl_block(25.0 / 100.0, &qs));
        }
    }
    out
}

fn pack_iq4xs_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    for _ in 0..n_rows {
        let mut qs = [0u8; QK_K];
        for q in &mut qs {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *q = u8::try_from(s % 16).unwrap_or(0);
        }
        let mut sc = [33u8; 8];
        for c in &mut sc {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *c = u8::try_from(28 + s % 9).unwrap_or(33);
        }
        out.extend_from_slice(&pack_iq4_xs_block(25.0 / 100.0, &sc, &qs));
        let _ = n_cols;
    }
    out
}

fn pack_q2k_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    for _ in 0..n_rows {
        let mut qs = [0u8; QK_K];
        for q in &mut qs {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *q = u8::try_from(s % 4).unwrap_or(0);
        }
        let mut sc = [1u8; 16];
        for c in &mut sc {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *c = u8::try_from(1 + s % 4).unwrap_or(1);
        }
        let mut mn = [0u8; 16];
        for c in &mut mn {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *c = u8::try_from(s % 3).unwrap_or(0);
        }
        out.extend_from_slice(&pack_q2_k_block(25.0 / 100.0, 5.0 / 100.0, &sc, &mn, &qs));
        let _ = n_cols;
    }
    out
}

fn pack_q3k_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    for _ in 0..n_rows {
        let mut qs = [0u8; QK_K];
        for q in &mut qs {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *q = u8::try_from(s % 8).unwrap_or(0);
        }
        let mut sc = [32u8; 16];
        for c in &mut sc {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *c = u8::try_from(28 + s % 9).unwrap_or(32);
        }
        out.extend_from_slice(&pack_q3_k_block(25.0 / 100.0, &sc, &qs));
        let _ = n_cols;
    }
    out
}

fn pack_q41_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    let nblocks = n_cols / QK4_1;
    for _ in 0..n_rows {
        for _ in 0..nblocks {
            let mut qs = [0u8; QK4_1];
            for q in &mut qs {
                s = s.wrapping_mul(1_664_525).wrapping_add(1);
                *q = u8::try_from(s % 16).unwrap_or(0);
            }
            out.extend_from_slice(&pack_q4_1_block(25.0 / 100.0, 5.0 / 100.0, &qs));
        }
    }
    out
}

fn pack_q50_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    let nblocks = n_cols / QK5_0;
    for _ in 0..n_rows {
        for _ in 0..nblocks {
            let mut qs = [0u8; QK5_0];
            for q in &mut qs {
                s = s.wrapping_mul(1_664_525).wrapping_add(1);
                *q = u8::try_from(s % 32).unwrap_or(0);
            }
            out.extend_from_slice(&pack_q5_0_block(25.0 / 100.0, &qs));
        }
    }
    out
}

fn pack_q51_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    let nblocks = n_cols / QK5_1;
    for _ in 0..n_rows {
        for _ in 0..nblocks {
            let mut qs = [0u8; QK5_1];
            for q in &mut qs {
                s = s.wrapping_mul(1_664_525).wrapping_add(1);
                *q = u8::try_from(s % 32).unwrap_or(0);
            }
            out.extend_from_slice(&pack_q5_1_block(25.0 / 100.0, 5.0 / 100.0, &qs));
        }
    }
    out
}

fn pack_mxfp4_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    let nblocks = n_cols / QK_MXFP4;
    for _ in 0..n_rows {
        for _ in 0..nblocks {
            let mut qs = [0u8; QK_MXFP4];
            for q in &mut qs {
                s = s.wrapping_mul(1_664_525).wrapping_add(1);
                *q = u8::try_from(s % 16).unwrap_or(0);
            }
            // e=127 → GGML_E8M0_TO_FP32_HALF = 0.5 (same bar as Q5_1's fixed d).
            out.extend_from_slice(&pack_mxfp4_block(127, &qs));
        }
    }
    out
}

fn pack_nvfp4_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    let nblocks = n_cols / QK_NVFP4;
    for _ in 0..n_rows {
        for _ in 0..nblocks {
            let mut qs = [0u8; QK_NVFP4];
            for q in &mut qs {
                s = s.wrapping_mul(1_664_525).wrapping_add(1);
                *q = u8::try_from(s % 16).unwrap_or(0);
            }
            // Four UE4M3 scales (one per 16-wide sub-block). 0x38 = exp 7 man 0 → d=0.5.
            let mut d = [0x38u8; 4];
            for slot in &mut d {
                s = s.wrapping_mul(1_664_525).wrapping_add(1);
                *slot = u8::try_from(0x28 + s % 24).unwrap_or(0x38);
            }
            out.extend_from_slice(&pack_nvfp4_block(d, &qs));
        }
    }
    out
}

fn pack_q10_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    let nblocks = n_cols / QK1_0;
    for _ in 0..n_rows {
        for _ in 0..nblocks {
            let mut qs = [0u8; QK1_0];
            for q in &mut qs {
                s = s.wrapping_mul(1_664_525).wrapping_add(1);
                *q = u8::try_from(s % 2).unwrap_or(0);
            }
            out.extend_from_slice(&pack_q1_0_block(25.0 / 100.0, &qs));
        }
    }
    out
}

fn pack_q20_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    let nblocks = n_cols / QK2_0;
    for _ in 0..n_rows {
        for _ in 0..nblocks {
            let mut qs = [0u8; QK2_0];
            for q in &mut qs {
                s = s.wrapping_mul(1_664_525).wrapping_add(1);
                *q = u8::try_from(s % 4).unwrap_or(0);
            }
            out.extend_from_slice(&pack_q2_0_block(25.0 / 100.0, &qs));
        }
    }
    out
}

fn pack_q81_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    let nblocks = n_cols / QK8_1;
    for _ in 0..n_rows {
        for _ in 0..nblocks {
            let mut qs = [0i8; QK8_1];
            for q in &mut qs {
                s = s.wrapping_mul(1_664_525).wrapping_add(1);
                let k = u8::try_from(s % 17).unwrap_or(0);
                *q = i8::try_from(i32::from(k).saturating_sub(8)).unwrap_or(0);
            }
            out.extend_from_slice(&pack_q8_1_block(25.0 / 100.0, &qs));
        }
    }
    out
}

fn pack_tq10_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    let nblocks = n_cols / QK_K;
    for _ in 0..n_rows {
        for _ in 0..nblocks {
            let mut qs = [0u8; QK_K];
            for q in &mut qs {
                s = s.wrapping_mul(1_664_525).wrapping_add(1);
                *q = u8::try_from(s % 3).unwrap_or(0);
            }
            out.extend_from_slice(&pack_tq1_0_block(25.0 / 100.0, &qs));
        }
    }
    out
}

fn pack_tq20_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    let nblocks = n_cols / QK_K;
    for _ in 0..n_rows {
        for _ in 0..nblocks {
            let mut qs = [0u8; QK_K];
            for q in &mut qs {
                s = s.wrapping_mul(1_664_525).wrapping_add(1);
                *q = u8::try_from(s % 3).unwrap_or(0);
            }
            out.extend_from_slice(&pack_tq2_0_block(25.0 / 100.0, &qs));
        }
    }
    out
}

fn pack_q5k_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    for _ in 0..n_rows {
        let mut qs = [0u8; QK_K];
        for q in &mut qs {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *q = u8::try_from(s % 24).unwrap_or(0);
        }
        let mut sc = [1u8; 8];
        for c in &mut sc {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *c = u8::try_from(1 + s % 4).unwrap_or(1);
        }
        let mut mn = [0u8; 8];
        for c in &mut mn {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            *c = u8::try_from(s % 3).unwrap_or(0);
        }
        out.extend_from_slice(&pack_q5_k_block(25.0 / 100.0, 5.0 / 100.0, &sc, &mn, &qs));
        let _ = n_cols;
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

fn need<'a>(g: &'a Gguf, name: &str) -> Result<Tensor<'a>, LlamaError> {
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

fn is_applied_norm_or_bias(name: &str) -> bool {
    name == "output_norm.weight"
        || name.ends_with(".attn_norm.weight")
        || name.ends_with(".ffn_norm.weight")
        || name.ends_with(".attn_q.bias")
        || name.ends_with(".attn_k.bias")
        || name.ends_with(".attn_v.bias")
}

fn f32s(t: Tensor<'_>) -> Result<Vec<f32>, LlamaError> {
    match t.ty {
        GgmlType::F32 => {
            let (chunks, rem) = t.data.as_chunks::<4>();
            if !rem.is_empty() {
                return Err(LlamaError::Shape(t.name.to_string()));
            }
            Ok(chunks
                .iter()
                .map(|c| f32::from_bits(u32::from_le_bytes(*c)))
                .collect())
        }
        GgmlType::F16 if is_applied_norm_or_bias(t.name) && t.n_rows() == 1 => {
            let n = t.n_cols();
            let mut y = vec![0.0f32; n];
            dequant_f16_row(n, t.data, &mut y)?;
            Ok(y)
        }
        other => Err(LlamaError::Type {
            tensor: t.name.to_string(),
            ty: other.to_i32(),
        }),
    }
}

fn quant_mat(t: Tensor<'_>) -> Result<QuantMat, LlamaError> {
    if t.ty == GgmlType::F16 && t.shape.len() < 2 {
        return Err(LlamaError::Type {
            tensor: t.name.to_string(),
            ty: t.ty.to_i32(),
        });
    }
    match t.ty {
        GgmlType::F32
        | GgmlType::F16
        | GgmlType::BF16
        | GgmlType::Q2_K
        | GgmlType::Q3_K
        | GgmlType::Q4_1
        | GgmlType::Q5_0
        | GgmlType::Q5_1
        | GgmlType::MXFP4
        | GgmlType::NVFP4
        | GgmlType::Q1_0
        | GgmlType::Q2_0
        | GgmlType::Q8_1
        | GgmlType::TQ1_0
        | GgmlType::TQ2_0
        | GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q6_K
        | GgmlType::IQ1_S
        | GgmlType::IQ1_M
        | GgmlType::IQ2_XXS
        | GgmlType::IQ2_XS
        | GgmlType::IQ2_S
        | GgmlType::IQ3_XXS
        | GgmlType::IQ3_S
        | GgmlType::IQ4_NL
        | GgmlType::IQ4_XS => {
            let (start, end) = t.blob_range();
            Ok(QuantMat {
                name: t.name.to_string(),
                ty: t.ty,
                n_cols: t.n_cols(),
                n_rows: t.n_rows(),
                start,
                end,
            })
        }
        other => Err(LlamaError::Type {
            tensor: t.name.to_string(),
            ty: other.to_i32(),
        }),
    }
}

impl Llama {
    fn gemm_mat(&self, m: &QuantMat, n_tokens: usize, x: &[f32]) -> Result<Vec<f32>, LlamaError> {
        if n_tokens == 1 {
            return self.gemv_mat(m, x);
        }
        let data = self.mat_bytes(m)?;
        let n_out = m
            .n_rows
            .checked_mul(n_tokens)
            .ok_or_else(|| LlamaError::Shape(m.name.clone()))?;
        let mut y = vec![0.0f32; n_out];
        match m.ty {
            GgmlType::F32 => gemm_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::F16 => gemm_f16(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::BF16 => gemm_bf16(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::Q2_K => gemm_q2_k_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::Q3_K => gemm_q3_k_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::Q4_1 => gemm_q4_1_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::Q5_0 => gemm_q5_0_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::Q5_1 => gemm_q5_1_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::MXFP4 => gemm_mxfp4_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::NVFP4 => gemm_nvfp4_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::Q1_0 => gemm_q1_0_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::Q2_0 => gemm_q2_0_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::Q8_1 => gemm_q8_1_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::TQ1_0 => gemm_tq1_0_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::TQ2_0 => gemm_tq2_0_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::Q4_K => gemm_q4_k_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::Q5_K => gemm_q5_k_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::Q6_K => gemm_q6_k_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::IQ1_S => gemm_iq1_s_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::IQ1_M => gemm_iq1_m_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::IQ2_XXS => gemm_iq2_xxs_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::IQ2_XS => gemm_iq2_xs_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::IQ2_S => gemm_iq2_s_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::IQ3_XXS => gemm_iq3_xxs_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::IQ3_S => gemm_iq3_s_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::IQ4_NL => gemm_iq4_nl_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::IQ4_XS => gemm_iq4_xs_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            other => {
                return Err(LlamaError::Type {
                    tensor: m.name.clone(),
                    ty: other.to_i32(),
                })
            }
        }
        Ok(y)
    }

    fn gemv_mat(&self, m: &QuantMat, x: &[f32]) -> Result<Vec<f32>, LlamaError> {
        let data = self.mat_bytes(m)?;
        let mut y = vec![0.0f32; m.n_rows];
        match m.ty {
            GgmlType::F32 => gemv_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::F16 => gemv_f16(m.n_cols, data, x, &mut y)?,
            GgmlType::BF16 => gemv_bf16(m.n_cols, data, x, &mut y)?,
            GgmlType::Q2_K => gemv_q2_k_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::Q3_K => gemv_q3_k_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::Q4_1 => gemv_q4_1_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::Q5_0 => gemv_q5_0_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::Q5_1 => gemv_q5_1_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::MXFP4 => gemv_mxfp4_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::NVFP4 => gemv_nvfp4_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::Q1_0 => gemv_q1_0_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::Q2_0 => gemv_q2_0_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::Q8_1 => gemv_q8_1_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::TQ1_0 => gemv_tq1_0_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::TQ2_0 => gemv_tq2_0_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::Q4_K => gemv_q4_k_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::Q5_K => gemv_q5_k_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::Q6_K => gemv_q6_k_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::IQ1_S => gemv_iq1_s_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::IQ1_M => gemv_iq1_m_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::IQ2_XXS => gemv_iq2_xxs_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::IQ2_XS => gemv_iq2_xs_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::IQ2_S => gemv_iq2_s_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::IQ3_XXS => gemv_iq3_xxs_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::IQ3_S => gemv_iq3_s_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::IQ4_NL => gemv_iq4_nl_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::IQ4_XS => gemv_iq4_xs_f32(m.n_cols, data, x, &mut y)?,
            other => {
                return Err(LlamaError::Type {
                    tensor: m.name.clone(),
                    ty: other.to_i32(),
                })
            }
        }
        Ok(y)
    }

    fn embed(&self, token: u32) -> Result<Vec<f32>, LlamaError> {
        let emb = &self.token_embd;
        let data = self.mat_bytes(emb)?;
        let row = usize::try_from(token).map_err(|_| LlamaError::Shape(emb.name.clone()))?;
        let rb = match emb.ty {
            GgmlType::F32 => f32_row_bytes(emb.n_cols)?,
            GgmlType::F16 => f16_row_bytes(emb.n_cols)?,
            GgmlType::BF16 => bf16_row_bytes(emb.n_cols)?,
            GgmlType::Q2_K => q2_k_row_bytes(emb.n_cols)?,
            GgmlType::Q3_K => q3_k_row_bytes(emb.n_cols)?,
            GgmlType::Q4_1 => q4_1_row_bytes(emb.n_cols)?,
            GgmlType::Q5_0 => q5_0_row_bytes(emb.n_cols)?,
            GgmlType::Q5_1 => q5_1_row_bytes(emb.n_cols)?,
            GgmlType::MXFP4 => mxfp4_row_bytes(emb.n_cols)?,
            GgmlType::NVFP4 => nvfp4_row_bytes(emb.n_cols)?,
            GgmlType::Q1_0 => q1_0_row_bytes(emb.n_cols)?,
            GgmlType::Q2_0 => q2_0_row_bytes(emb.n_cols)?,
            GgmlType::Q8_1 => q8_1_row_bytes(emb.n_cols)?,
            GgmlType::TQ1_0 => tq1_0_row_bytes(emb.n_cols)?,
            GgmlType::TQ2_0 => tq2_0_row_bytes(emb.n_cols)?,
            GgmlType::Q4_K => q4_k_row_bytes(emb.n_cols)?,
            GgmlType::Q5_K => q5_k_row_bytes(emb.n_cols)?,
            GgmlType::Q6_K => q6_k_row_bytes(emb.n_cols)?,
            GgmlType::IQ1_S => iq1_s_row_bytes(emb.n_cols)?,
            GgmlType::IQ1_M => iq1_m_row_bytes(emb.n_cols)?,
            GgmlType::IQ2_XXS => iq2_xxs_row_bytes(emb.n_cols)?,
            GgmlType::IQ2_XS => iq2_xs_row_bytes(emb.n_cols)?,
            GgmlType::IQ2_S => iq2_s_row_bytes(emb.n_cols)?,
            GgmlType::IQ3_XXS => iq3_xxs_row_bytes(emb.n_cols)?,
            GgmlType::IQ3_S => iq3_s_row_bytes(emb.n_cols)?,
            GgmlType::IQ4_NL => iq4_nl_row_bytes(emb.n_cols)?,
            GgmlType::IQ4_XS => iq4_xs_row_bytes(emb.n_cols)?,
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
        let bytes = data
            .get(start..end)
            .ok_or_else(|| LlamaError::Shape(emb.name.clone()))?;
        let mut y = vec![0.0f32; emb.n_cols];
        match emb.ty {
            GgmlType::F32 => dequant_f32_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::F16 => dequant_f16_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::BF16 => dequant_bf16_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::Q2_K => dequant_q2_k_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::Q3_K => dequant_q3_k_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::Q4_1 => dequant_q4_1_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::Q5_0 => dequant_q5_0_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::Q5_1 => dequant_q5_1_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::MXFP4 => dequant_mxfp4_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::NVFP4 => dequant_nvfp4_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::Q1_0 => dequant_q1_0_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::Q2_0 => dequant_q2_0_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::Q8_1 => dequant_q8_1_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::TQ1_0 => dequant_tq1_0_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::TQ2_0 => dequant_tq2_0_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::Q4_K => dequant_q4_k_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::Q5_K => dequant_q5_k_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::Q6_K => dequant_q6_k_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::IQ1_S => dequant_iq1_s_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::IQ1_M => dequant_iq1_m_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::IQ2_XXS => dequant_iq2_xxs_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::IQ2_XS => dequant_iq2_xs_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::IQ2_S => dequant_iq2_s_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::IQ3_XXS => dequant_iq3_xxs_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::IQ3_S => dequant_iq3_s_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::IQ4_NL => dequant_iq4_nl_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::IQ4_XS => dequant_iq4_xs_row(emb.n_cols, bytes, &mut y)?,
            other => {
                return Err(LlamaError::Type {
                    tensor: emb.name.clone(),
                    ty: other.to_i32(),
                })
            }
        }
        Ok(y)
    }

    #[cfg(test)]
    fn output_blob_range(&self) -> (usize, usize) {
        (self.output.start, self.output.end)
    }

    #[cfg(test)]
    fn token_embd_blob_range(&self) -> (usize, usize) {
        (self.token_embd.start, self.token_embd.end)
    }

    #[cfg(test)]
    fn gemv_output(&self, x: &[f32]) -> Result<Vec<f32>, LlamaError> {
        self.gemv_mat(&self.output, x)
    }

    #[cfg(test)]
    fn gemv_token_embd(&self, x: &[f32]) -> Result<Vec<f32>, LlamaError> {
        self.gemv_mat(&self.token_embd, x)
    }

    #[cfg(test)]
    fn gemm_output(&self, n_tokens: usize, x: &[f32]) -> Result<Vec<f32>, LlamaError> {
        self.gemm_mat(&self.output, n_tokens, x)
    }

    #[cfg(test)]
    fn embed_token(&self, token: u32) -> Result<Vec<f32>, LlamaError> {
        self.embed(token)
    }
}

fn token_row<'a>(
    x: &'a [f32],
    t: usize,
    width: usize,
    what: &'static str,
) -> Result<&'a [f32], LlamaError> {
    let start = t
        .checked_mul(width)
        .ok_or_else(|| LlamaError::Shape(what.into()))?;
    let end = start
        .checked_add(width)
        .ok_or_else(|| LlamaError::Shape(what.into()))?;
    x.get(start..end)
        .ok_or_else(|| LlamaError::Shape(what.into()))
}

fn rmsnorm_rows(x: &[f32], width: usize, w: &[f32], eps: f32) -> Result<Vec<f32>, LlamaError> {
    if width == 0 || !x.len().is_multiple_of(width) {
        return Err(LlamaError::Shape("rmsnorm".into()));
    }
    let mut out = Vec::new();
    for row in x.chunks(width) {
        out.extend(rmsnorm(row, w, eps)?);
    }
    Ok(out)
}

fn add_bias_rows(
    mut x: Vec<f32>,
    width: usize,
    bias: Option<&[f32]>,
) -> Result<Vec<f32>, LlamaError> {
    let Some(b) = bias else {
        return Ok(x);
    };
    if b.len() != width || width == 0 || !x.len().is_multiple_of(width) {
        return Err(LlamaError::Shape("vector add".into()));
    }
    for row in x.chunks_mut(width) {
        for (xv, bv) in row.iter_mut().zip(b.iter()) {
            *xv += *bv;
        }
    }
    Ok(x)
}

fn attend_query(
    cache: &KvCache,
    layer: usize,
    qh: &[Vec<f32>],
    n_head_kv: usize,
    hd: usize,
    seq: usize,
) -> Result<Vec<f32>, LlamaError> {
    if n_head_kv == 0 {
        return Err(LlamaError::Shape("gqa".into()));
    }
    let n_embd = qh.len().saturating_mul(hd);
    let mut attn = vec![0.0f32; n_embd];
    let scale = (f32::from(u16::try_from(hd).unwrap_or(1))).sqrt();
    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    let gqa = qh.len() / n_head_kv;
    if gqa == 0 {
        return Err(LlamaError::Shape("gqa".into()));
    }
    for (hq, qvec) in qh.iter().enumerate() {
        let hkv = hq / gqa;
        let mut scores = vec![0.0f32; seq];
        for t in 0..seq {
            let kv = kv_at(&cache.k, layer, hkv, n_head_kv, cache.max_seq, t, hd)?;
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
            let vv = kv_at(&cache.v, layer, hkv, n_head_kv, cache.max_seq, t, hd)?;
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
    Ok(attn)
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

/// Official llama.cpp Gemma FFN is `LLM_FFN_GELU` → `ggml_gelu` (tanh approx).
fn ggml_gelu_f32(x: f32) -> f32 {
    let coef_a = 44_715.0 / 1_000_000.0;
    let sqrt_2_over_pi = (2.0 / core::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (sqrt_2_over_pi * x * (1.0 + coef_a * x * x)).tanh())
}

fn gelu(x: &[f32]) -> Result<Vec<f32>, LlamaError> {
    let mut out = vec![0.0f32; x.len()];
    for (o, xv) in out.iter_mut().zip(x.iter()) {
        *o = ggml_gelu_f32(*xv);
    }
    Ok(out)
}

fn ffn_gate_act(x: &[f32], use_gelu: bool) -> Result<Vec<f32>, LlamaError> {
    if use_gelu {
        gelu(x)
    } else {
        silu(x)
    }
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
    use crate::{load_gguf, load_gguf_owned};

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

    /// ggml `ggml_fp16_to_fp32` (oracle). Independent of `dequant_f16_row` / `gemv_f16`.
    fn oracle_f16_elem(bytes: &[u8]) -> f32 {
        crate::fp16::f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    /// ggml `GGML_BF16_TO_FP32` (oracle). Independent of `dequant_bf16_row` / `gemv_bf16`.
    fn oracle_bf16_elem(bytes: &[u8]) -> f32 {
        f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16)
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

    /// ggml `ksigns_iq2xs` table. Independent of crate `ksigns_iq2xs`.
    const KSIGNS_IQ2XS: [u8; 128] = [
        0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20,
        149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40,
        169, 170, 43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60,
        189, 190, 63, 192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80,
        209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228,
        101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246,
        119, 120, 249, 250, 123, 252, 125, 126, 255,
    ];

    /// ggml `dequantize_row_iq2_xxs` (oracle). Independent of crate `dequant_iq2_xxs_row`.
    fn dequant_iq2_xxs_row_oracle(w: &[u8]) -> Vec<f32> {
        const GRID: [u64; 256] = crate::quant::IQ2XXS_GRID;
        let nblocks = w.len() / crate::quant::IQ2_XXS_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK_K];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::IQ2_XXS_BLOCK..(b + 1) * crate::quant::IQ2_XXS_BLOCK];
            let d = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let qs = &wb[2..66];
            let mut yo = b * QK_K;
            for ib32 in 0..QK_K / 32 {
                let off = ib32 * 8;
                let aux32 =
                    u32::from_le_bytes([qs[off + 4], qs[off + 5], qs[off + 6], qs[off + 7]]);
                let db = d * (0.5 + f32::from(u8::try_from(aux32 >> 28).unwrap())) * 0.25;
                for l in 0..4 {
                    let g = GRID[usize::from(qs[off + l])].to_le_bytes();
                    let signs = KSIGNS_IQ2XS[usize::try_from((aux32 >> (7 * l)) & 127).unwrap()];
                    for j in 0..8 {
                        let s = if signs & (1u8 << j) == 0 { 1.0 } else { -1.0 };
                        y[yo + j] = db * f32::from(g[j]) * s;
                    }
                    yo += 8;
                }
            }
        }
        y
    }

    /// ggml `dequantize_row_iq1_s` (oracle). Independent of crate `dequant_iq1_s_row`.
    fn dequant_iq1_s_row_oracle(w: &[u8]) -> Vec<f32> {
        const GRID: [u64; 2048] = crate::quant::IQ1S_GRID;
        const DELTA: f32 = 0.125;
        let nblocks = w.len() / crate::quant::IQ1_S_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK_K];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::IQ1_S_BLOCK..(b + 1) * crate::quant::IQ1_S_BLOCK];
            let d = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let qs = &wb[2..34];
            let qh = &wb[34..50];
            let mut yo = b * QK_K;
            for ib in 0..8 {
                let qhv = u16::from_le_bytes([qh[ib * 2], qh[ib * 2 + 1]]);
                let dl = d * (2.0 * f32::from((qhv >> 12) & 7) + 1.0);
                let delta = if qhv & 0x8000 != 0 { -DELTA } else { DELTA };
                for l in 0..4 {
                    let idx =
                        usize::from(qs[ib * 4 + l]) | (usize::from((qhv >> (3 * l)) & 7) << 8);
                    let g = GRID[idx].to_le_bytes();
                    for j in 0..8 {
                        y[yo + j] = dl * (f32::from(i8::from_le_bytes([g[j]])) + delta);
                    }
                    yo += 8;
                }
            }
        }
        y
    }

    /// ggml `dequantize_row_iq1_m` (oracle). Independent of crate `dequant_iq1_m_row`.
    fn dequant_iq1_m_row_oracle(w: &[u8]) -> Vec<f32> {
        const GRID: [u64; 2048] = crate::quant::IQ1S_GRID;
        const DELTA: f32 = 0.125;
        let nblocks = w.len() / crate::quant::IQ1_M_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK_K];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::IQ1_M_BLOCK..(b + 1) * crate::quant::IQ1_M_BLOCK];
            let qs = &wb[0..32];
            let qh = &wb[32..48];
            let scb = &wb[48..56];
            let sc = [
                u16::from_le_bytes([scb[0], scb[1]]),
                u16::from_le_bytes([scb[2], scb[3]]),
                u16::from_le_bytes([scb[4], scb[5]]),
                u16::from_le_bytes([scb[6], scb[7]]),
            ];
            let bits = (sc[0] >> 12)
                | ((sc[1] >> 8) & 0x00f0)
                | ((sc[2] >> 4) & 0x0f00)
                | (sc[3] & 0xf000);
            let d = crate::fp16::f16_to_f32(bits);
            let mut yo = b * QK_K;
            for ib in 0..8 {
                let scv = sc[ib / 2];
                let sh = 6 * (ib % 2);
                let dl1 = d * (2.0 * f32::from((scv >> sh) & 7) + 1.0);
                let dl2 = d * (2.0 * f32::from((scv >> (sh + 3)) & 7) + 1.0);
                for l in 0..4 {
                    let qhb = qh[ib * 2 + l / 2];
                    let idx = if l % 2 == 0 {
                        usize::from(qs[ib * 4 + l]) | (usize::from(qhb) << 8 & 0x700)
                    } else {
                        usize::from(qs[ib * 4 + l]) | (usize::from(qhb) << 4 & 0x700)
                    };
                    let delta_bit = if l % 2 == 0 { 0x08 } else { 0x80 };
                    let delta = if qhb & delta_bit != 0 { -DELTA } else { DELTA };
                    let dl = if l < 2 { dl1 } else { dl2 };
                    let g = GRID[idx].to_le_bytes();
                    for j in 0..8 {
                        y[yo + j] = dl * (f32::from(i8::from_le_bytes([g[j]])) + delta);
                    }
                    yo += 8;
                }
            }
        }
        y
    }

    /// ggml `dequantize_row_iq2_xs` (oracle). Independent of crate `dequant_iq2_xs_row`.
    fn dequant_iq2_xs_row_oracle(w: &[u8]) -> Vec<f32> {
        const GRID: [u64; 512] = crate::quant::IQ2XS_GRID;
        let nblocks = w.len() / crate::quant::IQ2_XS_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK_K];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::IQ2_XS_BLOCK..(b + 1) * crate::quant::IQ2_XS_BLOCK];
            let d = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let qs = &wb[2..66];
            let scales = &wb[66..74];
            let mut yo = b * QK_K;
            for (ib32, &sc) in scales.iter().enumerate() {
                let db0 = d * (0.5 + f32::from(sc & 0x0f)) * 0.25;
                let db1 = d * (0.5 + f32::from(sc >> 4)) * 0.25;
                for l in 0..4 {
                    let off = ib32 * 8 + l * 2;
                    let q16 = u16::from_le_bytes([qs[off], qs[off + 1]]);
                    let g = GRID[usize::from(q16 & 511)].to_le_bytes();
                    let signs = KSIGNS_IQ2XS[usize::from(q16 >> 9)];
                    let dl = if l < 2 { db0 } else { db1 };
                    for j in 0..8 {
                        let s = if signs & (1u8 << j) == 0 { 1.0 } else { -1.0 };
                        y[yo + j] = dl * f32::from(g[j]) * s;
                    }
                    yo += 8;
                }
            }
        }
        y
    }

    /// ggml `dequantize_row_iq2_s` (oracle). Independent of crate `dequant_iq2_s_row`.
    fn dequant_iq2_s_row_oracle(w: &[u8]) -> Vec<f32> {
        const GRID: [u64; 1024] = crate::quant::IQ2S_GRID;
        let nblocks = w.len() / crate::quant::IQ2_S_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK_K];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::IQ2_S_BLOCK..(b + 1) * crate::quant::IQ2_S_BLOCK];
            let d = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let qs = &wb[2..34];
            let signs = &wb[34..66];
            let qh = &wb[66..74];
            let scales = &wb[74..82];
            let mut yo = b * QK_K;
            for ib32 in 0..QK_K / 32 {
                let sc = scales[ib32];
                let db0 = d * (0.5 + f32::from(sc & 0x0f)) * 0.25;
                let db1 = d * (0.5 + f32::from(sc >> 4)) * 0.25;
                for l in 0..4 {
                    let sh = 8u32.saturating_sub(u32::try_from(l).unwrap_or(0).saturating_mul(2));
                    let idx = usize::from(qs[ib32 * 4 + l])
                        | (usize::from(qh[ib32]).wrapping_shl(sh) & 0x300);
                    let g = GRID[idx].to_le_bytes();
                    let sign = signs[ib32 * 4 + l];
                    let dl = if l < 2 { db0 } else { db1 };
                    for j in 0..8 {
                        let s = if sign & (1u8 << j) == 0 { 1.0 } else { -1.0 };
                        y[yo + j] = dl * f32::from(g[j]) * s;
                    }
                    yo += 8;
                }
            }
        }
        y
    }

    /// ggml `dequantize_row_iq3_xxs` (oracle). Independent of crate `dequant_iq3_xxs_row`.
    fn dequant_iq3_xxs_row_oracle(w: &[u8]) -> Vec<f32> {
        const GRID: [u32; 256] = crate::quant::IQ3XXS_GRID;
        let nblocks = w.len() / crate::quant::IQ3_XXS_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK_K];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::IQ3_XXS_BLOCK..(b + 1) * crate::quant::IQ3_XXS_BLOCK];
            let d = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let qs = &wb[2..66];
            let ss = &wb[66..98];
            let mut yo = b * QK_K;
            let mut qs_i = 0usize;
            for ib32 in 0..QK_K / 32 {
                let aux32 = u32::from_le_bytes([
                    ss[ib32 * 4],
                    ss[ib32 * 4 + 1],
                    ss[ib32 * 4 + 2],
                    ss[ib32 * 4 + 3],
                ]);
                let db = d * (0.5 + f32::from(u8::try_from(aux32 >> 28).unwrap())) * 0.5;
                for l in 0..4 {
                    let g1 = GRID[usize::from(qs[qs_i + 2 * l])].to_le_bytes();
                    let g2 = GRID[usize::from(qs[qs_i + 2 * l + 1])].to_le_bytes();
                    let signs = KSIGNS_IQ2XS[usize::try_from((aux32 >> (7 * l)) & 127).unwrap()];
                    for j in 0..4 {
                        let s0 = if signs & (1u8 << j) == 0 { 1.0 } else { -1.0 };
                        let s1 = if signs & (1u8 << (j + 4)) == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        y[yo + j] = db * f32::from(g1[j]) * s0;
                        y[yo + j + 4] = db * f32::from(g2[j]) * s1;
                    }
                    yo += 8;
                }
                qs_i += 8;
            }
        }
        y
    }

    /// ggml `dequantize_row_iq3_s` (oracle). Independent of crate `dequant_iq3_s_row`.
    fn dequant_iq3_s_row_oracle(w: &[u8]) -> Vec<f32> {
        const GRID: [u32; 512] = crate::quant::IQ3S_GRID;
        let nblocks = w.len() / crate::quant::IQ3_S_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK_K];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::IQ3_S_BLOCK..(b + 1) * crate::quant::IQ3_S_BLOCK];
            let d = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let qs = &wb[2..66];
            let qh = &wb[66..74];
            let signs = &wb[74..106];
            let scales = &wb[106..110];
            let mut yo = b * QK_K;
            let mut qs_i = 0usize;
            let mut sg_i = 0usize;
            let mut qh_i = 0usize;
            for ib32 in (0..QK_K / 32).step_by(2) {
                let sc = scales[ib32 / 2];
                let db1 = d * (1.0 + 2.0 * f32::from(sc & 0x0f));
                let db2 = d * (1.0 + 2.0 * f32::from(sc >> 4));
                for l in 0..4 {
                    let sh = u32::try_from(l).unwrap_or(0).saturating_mul(2);
                    let idx1 = usize::from(qs[qs_i + 2 * l])
                        | (usize::from(qh[qh_i]).wrapping_shl(8u32.saturating_sub(sh)) & 256);
                    let idx2 = usize::from(qs[qs_i + 2 * l + 1])
                        | (usize::from(qh[qh_i]).wrapping_shl(7u32.saturating_sub(sh)) & 256);
                    let g1 = GRID[idx1].to_le_bytes();
                    let g2 = GRID[idx2].to_le_bytes();
                    let sign = signs[sg_i + l];
                    for j in 0..4 {
                        let s0 = if sign & (1u8 << j) == 0 { 1.0 } else { -1.0 };
                        let s1 = if sign & (1u8 << (j + 4)) == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        y[yo + j] = db1 * f32::from(g1[j]) * s0;
                        y[yo + j + 4] = db1 * f32::from(g2[j]) * s1;
                    }
                    yo += 8;
                }
                qs_i += 8;
                sg_i += 4;
                for l in 0..4 {
                    let sh = u32::try_from(l).unwrap_or(0).saturating_mul(2);
                    let idx1 = usize::from(qs[qs_i + 2 * l])
                        | (usize::from(qh[qh_i + 1]).wrapping_shl(8u32.saturating_sub(sh)) & 256);
                    let idx2 = usize::from(qs[qs_i + 2 * l + 1])
                        | (usize::from(qh[qh_i + 1]).wrapping_shl(7u32.saturating_sub(sh)) & 256);
                    let g1 = GRID[idx1].to_le_bytes();
                    let g2 = GRID[idx2].to_le_bytes();
                    let sign = signs[sg_i + l];
                    for j in 0..4 {
                        let s0 = if sign & (1u8 << j) == 0 { 1.0 } else { -1.0 };
                        let s1 = if sign & (1u8 << (j + 4)) == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        y[yo + j] = db2 * f32::from(g1[j]) * s0;
                        y[yo + j + 4] = db2 * f32::from(g2[j]) * s1;
                    }
                    yo += 8;
                }
                qs_i += 8;
                sg_i += 4;
                qh_i += 2;
            }
        }
        y
    }

    /// ggml `dequantize_row_iq4_nl` (oracle). Independent of crate `dequant_iq4_nl_row`.
    fn dequant_iq4_nl_row_oracle(w: &[u8]) -> Vec<f32> {
        const KVALUES: [i8; 16] = [
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        ];
        let nblocks = w.len() / crate::quant::IQ4_NL_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK4_NL];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::IQ4_NL_BLOCK..(b + 1) * crate::quant::IQ4_NL_BLOCK];
            let d = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let qs = &wb[2..];
            let yo = b * QK4_NL;
            for j in 0..16 {
                y[yo + j] = d * f32::from(KVALUES[usize::from(qs[j] & 0x0f)]);
                y[yo + j + 16] = d * f32::from(KVALUES[usize::from(qs[j] >> 4)]);
            }
        }
        y
    }

    /// ggml `dequantize_row_iq4_xs` (oracle). Independent of crate `dequant_iq4_xs_row`.
    fn dequant_iq4_xs_row_oracle(w: &[u8]) -> Vec<f32> {
        const KVALUES: [i8; 16] = [
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        ];
        let nblocks = w.len() / crate::quant::IQ4_XS_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK_K];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::IQ4_XS_BLOCK..(b + 1) * crate::quant::IQ4_XS_BLOCK];
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

    /// ggml `dequantize_row_q2_K` (oracle). Independent of crate `dequant_q2_k_row`.
    fn dequant_q2_k_row_oracle(w: &[u8]) -> Vec<f32> {
        let nblocks = w.len() / crate::quant::Q2_K_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK_K];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::Q2_K_BLOCK..(b + 1) * crate::quant::Q2_K_BLOCK];
            let d = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[80], wb[81]]));
            let minv = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[82], wb[83]]));
            let scales = &wb[0..16];
            let mut yo = b * QK_K;
            let mut is = 0usize;
            for n in 0..2 {
                let q = &wb[16 + n * 32..16 + n * 32 + 32];
                let mut shift = 0u32;
                for _j in 0..4 {
                    let sc = scales[is];
                    is += 1;
                    let dl = d * f32::from(sc & 0x0f);
                    let ml = minv * f32::from(sc >> 4);
                    for l in 0..16 {
                        y[yo + l] = dl * f32::from((q[l] >> shift) & 3) - ml;
                    }
                    yo += 16;
                    let sc = scales[is];
                    is += 1;
                    let dl = d * f32::from(sc & 0x0f);
                    let ml = minv * f32::from(sc >> 4);
                    for l in 0..16 {
                        y[yo + l] = dl * f32::from((q[l + 16] >> shift) & 3) - ml;
                    }
                    yo += 16;
                    shift += 2;
                }
            }
        }
        y
    }

    /// ggml `dequantize_row_q3_K` (oracle). Independent of crate `dequant_q3_k_row`.
    fn dequant_q3_k_row_oracle(w: &[u8]) -> Vec<f32> {
        let nblocks = w.len() / crate::quant::Q3_K_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK_K];
        const KMASK1: u32 = 0x0303_0303;
        const KMASK2: u32 = 0x0f0f_0f0f;
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::Q3_K_BLOCK..(b + 1) * crate::quant::Q3_K_BLOCK];
            let d_all = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[108], wb[109]]));
            let hmask = &wb[0..32];
            let mut aux = [0u32; 4];
            aux[0] = u32::from_le_bytes([wb[96], wb[97], wb[98], wb[99]]);
            aux[1] = u32::from_le_bytes([wb[100], wb[101], wb[102], wb[103]]);
            aux[2] = u32::from_le_bytes([wb[104], wb[105], wb[106], wb[107]]);
            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
            aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
            aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
            let mut scb = [0u8; 16];
            scb[0..4].copy_from_slice(&aux[0].to_le_bytes());
            scb[4..8].copy_from_slice(&aux[1].to_le_bytes());
            scb[8..12].copy_from_slice(&aux[2].to_le_bytes());
            scb[12..16].copy_from_slice(&aux[3].to_le_bytes());
            let mut yo = b * QK_K;
            let mut is = 0usize;
            let mut m: u8 = 1;
            for n in 0..2 {
                let q = &wb[32 + n * 32..32 + n * 32 + 32];
                let mut shift = 0u32;
                for _j in 0..4 {
                    let dl = d_all * (f32::from(scb[is]) - 32.0);
                    is += 1;
                    for l in 0..16 {
                        let q2 = i32::from((q[l] >> shift) & 3);
                        let sub = if hmask[l] & m == 0 { 4 } else { 0 };
                        y[yo + l] = dl * (q2 - sub) as f32;
                    }
                    yo += 16;
                    let dl = d_all * (f32::from(scb[is]) - 32.0);
                    is += 1;
                    for l in 0..16 {
                        let q2 = i32::from((q[l + 16] >> shift) & 3);
                        let sub = if hmask[l + 16] & m == 0 { 4 } else { 0 };
                        y[yo + l] = dl * (q2 - sub) as f32;
                    }
                    yo += 16;
                    shift += 2;
                    m <<= 1;
                }
            }
        }
        y
    }

    /// ggml `dequantize_row_q4_1` (oracle). Independent of crate `dequant_q4_1_row`.
    fn dequant_q4_1_row_oracle(w: &[u8]) -> Vec<f32> {
        let nblocks = w.len() / crate::quant::Q4_1_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK4_1];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::Q4_1_BLOCK..(b + 1) * crate::quant::Q4_1_BLOCK];
            let d = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let m = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[2], wb[3]]));
            let qs = &wb[4..];
            let yo = b * QK4_1;
            for j in 0..16 {
                y[yo + j] = f32::from(qs[j] & 0x0f) * d + m;
                y[yo + j + 16] = f32::from(qs[j] >> 4) * d + m;
            }
        }
        y
    }

    /// ggml `dequantize_row_q5_0` (oracle). Independent of crate `dequant_q5_0_row`.
    fn dequant_q5_0_row_oracle(w: &[u8]) -> Vec<f32> {
        let nblocks = w.len() / crate::quant::Q5_0_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK5_0];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::Q5_0_BLOCK..(b + 1) * crate::quant::Q5_0_BLOCK];
            let d = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let qh = u32::from_le_bytes([wb[2], wb[3], wb[4], wb[5]]);
            let qs = &wb[6..];
            let yo = b * QK5_0;
            for j in 0..16 {
                let xh_0 = ((qh >> j) << 4) & 0x10;
                let xh_1 = (qh >> (j + 12)) & 0x10;
                let x0 = i32::from((qs[j] & 0x0f) | u8::try_from(xh_0).unwrap_or(0)) - 16;
                let x1 = i32::from((qs[j] >> 4) | u8::try_from(xh_1).unwrap_or(0)) - 16;
                y[yo + j] = (x0 as f32) * d;
                y[yo + j + 16] = (x1 as f32) * d;
            }
        }
        y
    }

    /// ggml `dequantize_row_q5_1` (oracle). Independent of crate `dequant_q5_1_row`.
    fn dequant_q5_1_row_oracle(w: &[u8]) -> Vec<f32> {
        let nblocks = w.len() / crate::quant::Q5_1_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK5_1];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::Q5_1_BLOCK..(b + 1) * crate::quant::Q5_1_BLOCK];
            let d = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let m = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[2], wb[3]]));
            let qh = u32::from_le_bytes([wb[4], wb[5], wb[6], wb[7]]);
            let qs = &wb[8..];
            let yo = b * QK5_1;
            for j in 0..16 {
                let xh_0 = ((qh >> j) << 4) & 0x10;
                let xh_1 = (qh >> (j + 12)) & 0x10;
                let x0 = f32::from((qs[j] & 0x0f) | u8::try_from(xh_0).unwrap_or(0));
                let x1 = f32::from((qs[j] >> 4) | u8::try_from(xh_1).unwrap_or(0));
                y[yo + j] = x0 * d + m;
                y[yo + j + 16] = x1 * d + m;
            }
        }
        y
    }

    /// ggml `dequantize_row_mxfp4` (oracle). Independent of crate `dequant_mxfp4_row`.
    fn dequant_mxfp4_row_oracle(w: &[u8]) -> Vec<f32> {
        const KVALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
        let nblocks = w.len() / crate::quant::MXFP4_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK_MXFP4];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::MXFP4_BLOCK..(b + 1) * crate::quant::MXFP4_BLOCK];
            let e = wb[0];
            let d = if e < 2 {
                f32::from_bits(0x0020_0000u32 << e)
            } else {
                f32::from_bits(u32::from(e - 1) << 23)
            };
            let qs = &wb[1..];
            let yo = b * QK_MXFP4;
            for j in 0..16 {
                let x0 = f32::from(KVALUES[usize::from(qs[j] & 0x0f)]);
                let x1 = f32::from(KVALUES[usize::from(qs[j] >> 4)]);
                y[yo + j] = x0 * d;
                y[yo + j + 16] = x1 * d;
            }
        }
        y
    }

    /// ggml `dequantize_row_nvfp4` (oracle). Independent of crate `dequant_nvfp4_row`.
    fn dequant_nvfp4_row_oracle(w: &[u8]) -> Vec<f32> {
        const KVALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
        let nblocks = w.len() / crate::quant::NVFP4_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK_NVFP4];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::NVFP4_BLOCK..(b + 1) * crate::quant::NVFP4_BLOCK];
            let d_bytes = &wb[..4];
            let qs = &wb[4..];
            for s in 0..4 {
                let ue = d_bytes[s];
                let d = if ue == 0 || ue == 0x7f {
                    0.0
                } else {
                    let exp = i32::from((ue >> 3) & 0x0f);
                    let man = f32::from(ue & 0x07);
                    let raw = if exp == 0 {
                        man * f32::from_bits(0x3b00_0000)
                    } else {
                        let biased = u32::try_from(120i32.saturating_add(exp)).unwrap_or(0);
                        (1.0 + man * 0.125) * f32::from_bits(biased.checked_shl(23).unwrap_or(0))
                    };
                    raw * 0.5
                };
                let yo = b * QK_NVFP4 + s * 16;
                for j in 0..8 {
                    let p = qs[s * 8 + j];
                    let x0 = f32::from(KVALUES[usize::from(p & 0x0f)]);
                    let x1 = f32::from(KVALUES[usize::from(p >> 4)]);
                    y[yo + j] = x0 * d;
                    y[yo + j + 8] = x1 * d;
                }
            }
        }
        y
    }

    /// ggml `dequantize_row_q1_0` (oracle). Independent of crate `dequant_q1_0_row`.
    fn dequant_q1_0_row_oracle(w: &[u8]) -> Vec<f32> {
        let nblocks = w.len() / crate::quant::Q1_0_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK1_0];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::Q1_0_BLOCK..(b + 1) * crate::quant::Q1_0_BLOCK];
            let d = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let qs = &wb[2..];
            let yo = b * QK1_0;
            for j in 0..QK1_0 {
                let bit = (qs[j / 8] >> (j % 8)) & 1;
                y[yo + j] = if bit != 0 { d } else { -d };
            }
        }
        y
    }

    /// ggml `dequantize_row_q2_0` (oracle). Independent of crate `dequant_q2_0_row`.
    fn dequant_q2_0_row_oracle(w: &[u8]) -> Vec<f32> {
        let nblocks = w.len() / crate::quant::Q2_0_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK2_0];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::Q2_0_BLOCK..(b + 1) * crate::quant::Q2_0_BLOCK];
            let d = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let qs = &wb[2..];
            let yo = b * QK2_0;
            for j in 0..QK2_0 {
                let q = (qs[j / 4] >> ((j % 4) * 2)) & 0x03;
                y[yo + j] = (f32::from(q) - 1.0) * d;
            }
        }
        y
    }

    /// ggml `dequantize_row_tq1_0` (oracle). Independent of crate `dequant_tq1_0_row`.
    /// 32-byte then 16-byte `qs` chunks, then `qh`. `y = (xi - 1) * d`.
    fn dequant_tq1_0_row_oracle(w: &[u8]) -> Vec<f32> {
        const POW3: [u8; 5] = [1, 3, 9, 27, 81];
        let nblocks = w.len() / crate::quant::TQ1_0_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK_K];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::TQ1_0_BLOCK..(b + 1) * crate::quant::TQ1_0_BLOCK];
            let qs = &wb[0..48];
            let qh = &wb[48..52];
            let d = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[52], wb[53]]));
            let mut yo = b * QK_K;
            for &p in &POW3 {
                for &byte in qs.iter().take(32) {
                    let q = byte.wrapping_mul(p);
                    let xi = (u16::from(q).wrapping_mul(3)) >> 8;
                    y[yo] = (f32::from(xi) - 1.0) * d;
                    yo += 1;
                }
            }
            for &p in &POW3 {
                for &byte in qs.iter().skip(32).take(16) {
                    let q = byte.wrapping_mul(p);
                    let xi = (u16::from(q).wrapping_mul(3)) >> 8;
                    y[yo] = (f32::from(xi) - 1.0) * d;
                    yo += 1;
                }
            }
            for &p in POW3.iter().take(4) {
                for &byte in qh.iter() {
                    let q = byte.wrapping_mul(p);
                    let xi = (u16::from(q).wrapping_mul(3)) >> 8;
                    y[yo] = (f32::from(xi) - 1.0) * d;
                    yo += 1;
                }
            }
        }
        y
    }

    /// ggml `dequantize_row_tq2_0` (oracle). Independent of crate `dequant_tq2_0_row`.
    /// Two 32-byte qs groups; for each, `l` then `m`. `y = (xi - 1) * d`.
    fn dequant_tq2_0_row_oracle(w: &[u8]) -> Vec<f32> {
        let nblocks = w.len() / crate::quant::TQ2_0_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK_K];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::TQ2_0_BLOCK..(b + 1) * crate::quant::TQ2_0_BLOCK];
            let qs = &wb[0..64];
            let d = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[64], wb[65]]));
            let mut yo = b * QK_K;
            for j in [0usize, 32] {
                for l in 0..4 {
                    for m in 0..32 {
                        let q = (qs[j + m] >> (l * 2)) & 3;
                        y[yo] = (f32::from(q) - 1.0) * d;
                        yo += 1;
                    }
                }
            }
        }
        y
    }

    /// ggml `dequantize_row_q8_1` (oracle). Independent of crate `dequant_q8_1_row`.
    /// `y[j] = q * d`. `s` at bytes 2..4 is unused by scalar dequant.
    fn dequant_q8_1_row_oracle(w: &[u8]) -> Vec<f32> {
        let nblocks = w.len() / crate::quant::Q8_1_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK8_1];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::Q8_1_BLOCK..(b + 1) * crate::quant::Q8_1_BLOCK];
            let d = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let qs = &wb[4..];
            let yo = b * QK8_1;
            for j in 0..QK8_1 {
                y[yo + j] = f32::from(i8::from_le_bytes([qs[j]])) * d;
            }
        }
        y
    }

    /// ggml `dequantize_row_q5_K` (oracle). Independent of crate `dequant_q5_k_row`.
    fn dequant_q5_k_row_oracle(w: &[u8]) -> Vec<f32> {
        let nblocks = w.len() / crate::quant::Q5_K_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK_K];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::Q5_K_BLOCK..(b + 1) * crate::quant::Q5_K_BLOCK];
            let d = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let minv = crate::fp16::f16_to_f32(u16::from_le_bytes([wb[2], wb[3]]));
            let scales = &wb[4..16];
            let qh = &wb[16..48];
            let mut qoff = 48usize;
            let mut yo = b * QK_K;
            let mut is = 0usize;
            let mut u1: u8 = 1;
            let mut u2: u8 = 2;
            for _ in 0..4 {
                let (sc, m) = oracle_scale_min(is, scales);
                let d1 = d * f32::from(sc);
                let m1 = minv * f32::from(m);
                let (sc, m) = oracle_scale_min(is + 1, scales);
                let d2 = d * f32::from(sc);
                let m2 = minv * f32::from(m);
                let q = &wb[qoff..qoff + 32];
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
                qoff += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        y
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

    fn oracle_gemv(t: Tensor<'_>, x: &[f32]) -> Vec<f32> {
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
            GgmlType::F16 => {
                for (r, yv) in y.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for (c, xv) in x.iter().enumerate() {
                        let off = (r * n_cols + c) * 2;
                        let w = oracle_f16_elem(&t.data[off..off + 2]);
                        acc += w * *xv;
                    }
                    *yv = acc;
                }
            }
            GgmlType::BF16 => {
                for (r, yv) in y.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for (c, xv) in x.iter().enumerate() {
                        let off = (r * n_cols + c) * 2;
                        let w = oracle_bf16_elem(&t.data[off..off + 2]);
                        acc += w * *xv;
                    }
                    *yv = acc;
                }
            }
            GgmlType::Q2_K => {
                let rb = (n_cols / QK_K) * crate::quant::Q2_K_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_q2_k_row_oracle(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::Q3_K => {
                let rb = (n_cols / QK_K) * crate::quant::Q3_K_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_q3_k_row_oracle(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::Q4_1 => {
                let rb = (n_cols / QK4_1) * crate::quant::Q4_1_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_q4_1_row_oracle(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::Q5_0 => {
                let rb = (n_cols / QK5_0) * crate::quant::Q5_0_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_q5_0_row_oracle(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::Q5_1 => {
                let rb = (n_cols / QK5_1) * crate::quant::Q5_1_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_q5_1_row_oracle(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::MXFP4 => {
                let rb = (n_cols / QK_MXFP4) * crate::quant::MXFP4_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_mxfp4_row_oracle(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::NVFP4 => {
                let rb = (n_cols / QK_NVFP4) * crate::quant::NVFP4_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_nvfp4_row_oracle(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::Q1_0 => {
                let rb = (n_cols / QK1_0) * crate::quant::Q1_0_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_q1_0_row_oracle(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::Q2_0 => {
                let rb = (n_cols / QK2_0) * crate::quant::Q2_0_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_q2_0_row_oracle(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::Q8_1 => {
                let rb = (n_cols / QK8_1) * crate::quant::Q8_1_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_q8_1_row_oracle(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::TQ1_0 => {
                let rb = (n_cols / QK_K) * crate::quant::TQ1_0_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_tq1_0_row_oracle(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::TQ2_0 => {
                let rb = (n_cols / QK_K) * crate::quant::TQ2_0_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_tq2_0_row_oracle(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::Q4_K => {
                let rb = (n_cols / QK_K) * crate::quant::Q4_K_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_q4_k_row(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::Q5_K => {
                let rb = (n_cols / QK_K) * crate::quant::Q5_K_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_q5_k_row_oracle(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::IQ1_S => {
                let rb = (n_cols / QK_K) * crate::quant::IQ1_S_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_iq1_s_row_oracle(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::IQ1_M => {
                let rb = (n_cols / QK_K) * crate::quant::IQ1_M_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_iq1_m_row_oracle(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::IQ2_XXS => {
                let rb = (n_cols / QK_K) * crate::quant::IQ2_XXS_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_iq2_xxs_row_oracle(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::IQ2_XS => {
                let rb = (n_cols / QK_K) * crate::quant::IQ2_XS_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_iq2_xs_row_oracle(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::IQ2_S => {
                let rb = (n_cols / QK_K) * crate::quant::IQ2_S_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_iq2_s_row_oracle(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::IQ3_XXS => {
                let rb = (n_cols / QK_K) * crate::quant::IQ3_XXS_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_iq3_xxs_row_oracle(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::IQ3_S => {
                let rb = (n_cols / QK_K) * crate::quant::IQ3_S_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_iq3_s_row_oracle(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::IQ4_NL => {
                let rb = (n_cols / QK4_NL) * crate::quant::IQ4_NL_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_iq4_nl_row_oracle(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::IQ4_XS => {
                let rb = (n_cols / QK_K) * crate::quant::IQ4_XS_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_iq4_xs_row_oracle(&t.data[r * rb..(r + 1) * rb]);
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

    fn oracle_embed(t: Tensor<'_>, token: u32) -> Vec<f32> {
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
            GgmlType::F16 => {
                let mut x = vec![0.0f32; n_cols];
                for (c, xv) in x.iter_mut().enumerate() {
                    let off = (row * n_cols + c) * 2;
                    *xv = oracle_f16_elem(&t.data[off..off + 2]);
                }
                x
            }
            GgmlType::BF16 => {
                let mut x = vec![0.0f32; n_cols];
                for (c, xv) in x.iter_mut().enumerate() {
                    let off = (row * n_cols + c) * 2;
                    *xv = oracle_bf16_elem(&t.data[off..off + 2]);
                }
                x
            }
            GgmlType::Q2_K => {
                let rb = (n_cols / QK_K) * crate::quant::Q2_K_BLOCK;
                dequant_q2_k_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::Q3_K => {
                let rb = (n_cols / QK_K) * crate::quant::Q3_K_BLOCK;
                dequant_q3_k_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::Q4_1 => {
                let rb = (n_cols / QK4_1) * crate::quant::Q4_1_BLOCK;
                dequant_q4_1_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::Q5_0 => {
                let rb = (n_cols / QK5_0) * crate::quant::Q5_0_BLOCK;
                dequant_q5_0_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::Q5_1 => {
                let rb = (n_cols / QK5_1) * crate::quant::Q5_1_BLOCK;
                dequant_q5_1_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::MXFP4 => {
                let rb = (n_cols / QK_MXFP4) * crate::quant::MXFP4_BLOCK;
                dequant_mxfp4_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::NVFP4 => {
                let rb = (n_cols / QK_NVFP4) * crate::quant::NVFP4_BLOCK;
                dequant_nvfp4_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::Q1_0 => {
                let rb = (n_cols / QK1_0) * crate::quant::Q1_0_BLOCK;
                dequant_q1_0_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::Q2_0 => {
                let rb = (n_cols / QK2_0) * crate::quant::Q2_0_BLOCK;
                dequant_q2_0_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::Q8_1 => {
                let rb = (n_cols / QK8_1) * crate::quant::Q8_1_BLOCK;
                dequant_q8_1_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::TQ1_0 => {
                let rb = (n_cols / QK_K) * crate::quant::TQ1_0_BLOCK;
                dequant_tq1_0_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::TQ2_0 => {
                let rb = (n_cols / QK_K) * crate::quant::TQ2_0_BLOCK;
                dequant_tq2_0_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::Q4_K => {
                let rb = (n_cols / QK_K) * crate::quant::Q4_K_BLOCK;
                dequant_q4_k_row(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::Q5_K => {
                let rb = (n_cols / QK_K) * crate::quant::Q5_K_BLOCK;
                dequant_q5_k_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::IQ1_S => {
                let rb = (n_cols / QK_K) * crate::quant::IQ1_S_BLOCK;
                dequant_iq1_s_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::IQ1_M => {
                let rb = (n_cols / QK_K) * crate::quant::IQ1_M_BLOCK;
                dequant_iq1_m_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::IQ2_XXS => {
                let rb = (n_cols / QK_K) * crate::quant::IQ2_XXS_BLOCK;
                dequant_iq2_xxs_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::IQ2_XS => {
                let rb = (n_cols / QK_K) * crate::quant::IQ2_XS_BLOCK;
                dequant_iq2_xs_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::IQ2_S => {
                let rb = (n_cols / QK_K) * crate::quant::IQ2_S_BLOCK;
                dequant_iq2_s_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::IQ3_XXS => {
                let rb = (n_cols / QK_K) * crate::quant::IQ3_XXS_BLOCK;
                dequant_iq3_xxs_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::IQ3_S => {
                let rb = (n_cols / QK_K) * crate::quant::IQ3_S_BLOCK;
                dequant_iq3_s_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::IQ4_NL => {
                let rb = (n_cols / QK4_NL) * crate::quant::IQ4_NL_BLOCK;
                dequant_iq4_nl_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::IQ4_XS => {
                let rb = (n_cols / QK_K) * crate::quant::IQ4_XS_BLOCK;
                dequant_iq4_xs_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::Q6_K => {
                let rb = (n_cols / QK_K) * crate::quant::Q6_K_BLOCK;
                dequant_q6_k_row(&t.data[row * rb..(row + 1) * rb])
            }
            _ => panic!("oracle embed ty"),
        }
    }

    fn oracle_add_bias(mut y: Vec<f32>, bias: Option<Tensor<'_>>) -> Vec<f32> {
        let Some(b) = bias else {
            return y;
        };
        let bv = f32s(b).unwrap();
        for (yv, bb) in y.iter_mut().zip(bv.iter()) {
            *yv += *bb;
        }
        y
    }

    fn oracle_gelu(x: f32) -> f32 {
        let coef_a = 44_715.0 / 1_000_000.0;
        let sqrt_2_over_pi = (2.0 / core::f32::consts::PI).sqrt();
        0.5 * x * (1.0 + (sqrt_2_over_pi * x * (1.0 + coef_a * x * x)).tanh())
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
        oracle_forward_seq(g, &[token])
    }

    /// Independent scalar Llama math for a token sequence (causal attn + GQA).
    fn oracle_forward_seq(g: &Gguf, tokens: &[u32]) -> Vec<f32> {
        let arch = architecture(g).expect("arch");
        let n_embd = arch_u32(g, arch, "embedding_length").unwrap() as usize;
        let n_head = arch_u32(g, arch, "attention.head_count").unwrap() as usize;
        let n_kv = arch_u32(g, arch, "attention.head_count_kv").unwrap() as usize;
        let n_rot = oracle_n_rot(g, arch, n_embd, n_head);
        let eps = arch_f32(g, arch, "attention.layer_norm_rms_epsilon").unwrap();
        let base = arch_f32(g, arch, "rope.freq_base").unwrap();
        let hd = n_embd / n_head;
        let gqa = n_head / n_kv;
        let emb = g.tensor("token_embd.weight").unwrap();
        let gemma = arch == "gemma";
        let embed_scale = if gemma {
            f32::from(u16::try_from(n_embd).unwrap_or(1)).sqrt()
        } else {
            1.0
        };
        let mut k_cache: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_kv];
        let mut v_cache: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_kv];
        let mut last = Vec::new();
        for (pos, &token) in tokens.iter().enumerate() {
            let mut residual = oracle_embed(emb, token);
            for v in &mut residual {
                *v *= embed_scale;
            }
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
                *h = oracle_rope(h.clone(), pos, n_rot, base);
            }
            for h in &mut kh {
                *h = oracle_rope(h.clone(), pos, n_rot, base);
            }
            for (hkv, khv) in kh.iter().enumerate() {
                k_cache[hkv].push(khv.clone());
                v_cache[hkv].push(vh[hkv].clone());
            }
            let seq = pos + 1;
            let inv = 1.0 / (hd as f32).sqrt();
            let mut attn = vec![0.0f32; n_embd];
            for (hq, qvec) in qh.iter().enumerate() {
                let hkv = hq / gqa;
                let mut scores = vec![0.0f32; seq];
                for t in 0..seq {
                    let kv = &k_cache[hkv][t];
                    scores[t] = qvec.iter().zip(kv.iter()).map(|(a, b)| a * b).sum::<f32>() * inv;
                }
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut s = 0.0f32;
                for v in &mut scores {
                    *v = (*v - m).exp();
                    s += *v;
                }
                if s > 0.0 {
                    for v in &mut scores {
                        *v /= s;
                    }
                }
                let mut acc = vec![0.0f32; hd];
                for t in 0..seq {
                    let st = scores[t];
                    for (a, b) in acc.iter_mut().zip(v_cache[hkv][t].iter()) {
                        *a += st * *b;
                    }
                }
                let off = hq * hd;
                attn[off..off + hd].copy_from_slice(&acc);
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
                .map(|(gv, u)| {
                    let act = if gemma {
                        oracle_gelu(*gv)
                    } else {
                        gv / (1.0 + (-gv).exp())
                    };
                    act * u
                })
                .collect();
            let down = oracle_gemv(g.tensor("blk.0.ffn_down.weight").unwrap(), &h);
            x = down.iter().zip(x.iter()).map(|(a, b)| a + b).collect();
            let on = f32s(g.tensor("output_norm.weight").unwrap()).unwrap();
            x = oracle_rmsnorm(&x, &on, eps);
            let lm_head = g
                .tensor("output.weight")
                .or_else(|| g.tensor("token_embd.weight"))
                .expect("lm_head");
            last = oracle_gemv(lm_head, &x);
        }
        last
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
        let model = Llama::from_gguf(g.clone()).expect("model");
        let mut cache = model.new_cache(4).expect("cache");
        let got = model.forward(&mut cache, token).expect("fwd");
        let exp = oracle_forward(&g, token);
        assert_logits_match(&got, &exp);
        assert_eq!(cache.n_past, 1);
    }

    fn load_prefill_match(bytes: &[u8], tokens: &[u32]) {
        let g = load_gguf(bytes).expect("load");
        let model = Llama::from_gguf(g.clone()).expect("model");
        let mut cache = model
            .new_cache(tokens.len().saturating_add(2))
            .expect("cache");
        let got = model.prefill(&mut cache, tokens).expect("prefill");
        let exp = oracle_forward_seq(&g, tokens);
        assert_logits_match(&got, &exp);
        assert_eq!(cache.n_past, tokens.len());
    }

    #[test]
    fn from_gguf_takes_file_blob_once_and_logits_match() {
        let bytes = tiny_llama_gguf();
        let g = load_gguf_owned(bytes.clone()).expect("owned");
        assert_eq!(g.blob_len(), bytes.len());
        let mut n_tensors = 0usize;
        for t in g.tensors() {
            assert!(g.payload_in_blob(t), "{}", t.name);
            n_tensors = n_tensors.saturating_add(1);
        }
        assert!(n_tensors > 0);
        let blob_len = g.blob_len();
        let model = Llama::from_gguf(g).expect("model");
        assert_eq!(model.blob_len(), blob_len);
        let g2 = load_gguf(&bytes).expect("reload");
        let mut cache = model.new_cache(4).expect("cache");
        let got = model.forward(&mut cache, 3).expect("fwd");
        let exp = oracle_forward(&g2, 3);
        assert_logits_match(&got, &exp);
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
        let model = Llama::from_gguf(g.clone()).expect("model");
        let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
        assert!(out.contains("ab"), "{out}");
        let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
        assert_eq!(out, out2);
    }

    #[test]
    fn greedy_prompt_and_n_predict_are_inputs_not_hardcoded() {
        let bytes = tiny_llama_gguf();
        let g = load_gguf(&bytes).expect("load");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g.clone()).expect("model");
        assert_eq!(greedy_generate(&model, &tok, "a", 0).expect("a"), "a");
        assert_eq!(greedy_generate(&model, &tok, "b", 0).expect("b"), "b");
        assert_eq!(greedy_generate(&model, &tok, "ab", 0).expect("ab"), "ab");
        let err = greedy_generate(&model, &tok, "", 2).expect_err("empty");
        assert!(err.to_string().contains("empty prompt"), "{err}");
        let q = load_gguf(&tiny_qwen2_gguf()).expect("qwen2");
        let qtok = Tokenizer::from_gguf(&q).expect("qtok");
        let qmodel = Llama::from_gguf(q.clone()).expect("qmodel");
        assert_eq!(
            greedy_generate_ctx(&qmodel, &qtok, "ab", 0, Some(1)).expect("n_ctx=1"),
            "ab"
        );
        let short = greedy_generate_ctx(&model, &tok, "ab", 2, Some(1)).expect_err("n_ctx");
        assert!(short.to_string().contains("n_ctx"), "{short}");
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
        let model = Llama::from_gguf(g.clone()).expect("model");
        let prompt = tok.decode(&[3]);
        assert!(!prompt.is_empty());
        let out = greedy_generate(&model, &tok, &prompt, 2).expect("gen");
        assert!(!out.is_empty());
        let out2 = greedy_generate(&model, &tok, &prompt, 2).expect("gen2");
        assert_eq!(out, out2);
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
    fn tiny_f16_logits_match_independent_oracle() {
        let bytes = tiny_f16_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::F16);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::F16);
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::F16);
        assert_eq!(g.tensor("blk.0.ffn_gate.weight").unwrap().ty, GgmlType::F16);
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_f16_1d_norms_load_and_match_f32_norm_twin() {
        let f16_1d = tiny_f16_1d_gguf();
        let f32_norm = tiny_f16_gguf();
        let g16 = load_gguf(&f16_1d).expect("load f16 1d");
        let g32 = load_gguf(&f32_norm).expect("load f32 norm twin");
        assert_eq!(g16.tensor("token_embd.weight").unwrap().ty, GgmlType::F16);
        assert_eq!(g16.tensor("output.weight").unwrap().ty, GgmlType::F16);
        assert_eq!(g16.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::F16);
        assert_eq!(g16.tensor("output_norm.weight").unwrap().ty, GgmlType::F16);
        assert_eq!(
            g16.tensor("blk.0.attn_norm.weight").unwrap().ty,
            GgmlType::F16
        );
        assert_eq!(
            g16.tensor("blk.0.ffn_norm.weight").unwrap().ty,
            GgmlType::F16
        );
        assert_eq!(g32.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(
            g16.tensor("token_embd.weight").unwrap().data,
            g32.tensor("token_embd.weight").unwrap().data
        );
        assert_eq!(
            g16.tensor("output.weight").unwrap().data,
            g32.tensor("output.weight").unwrap().data
        );

        let m16 = Llama::from_gguf(g16.clone()).expect("f16 1d model");
        let m32 = Llama::from_gguf(g32.clone()).expect("f32 twin model");
        let x = pat_f32(TINY_N_EMBD, 21);
        let gemv16 = m16.gemv_output(&x).expect("f16 1d gemv");
        let gemv32 = m32.gemv_output(&x).expect("f32 twin gemv");
        assert_logits_match(&gemv16, &gemv32);
        let exp_gemv = oracle_gemv(g16.tensor("output.weight").unwrap(), &x);
        assert_logits_match(&gemv16, &exp_gemv);

        let mut x2 = pat_f32(TINY_N_EMBD, 22);
        x2.extend(pat_f32(TINY_N_EMBD, 23));
        let gemm16 = m16.gemm_output(2, &x2).expect("f16 1d gemm");
        let gemm32 = m32.gemm_output(2, &x2).expect("f32 twin gemm");
        assert_logits_match(&gemm16, &gemm32);

        let emb16 = m16.embed_token(3).expect("f16 1d embed");
        let emb32 = m32.embed_token(3).expect("f32 twin embed");
        assert_logits_match(&emb16, &emb32);
        let exp_emb = oracle_embed(g16.tensor("token_embd.weight").unwrap(), 3);
        assert_logits_match(&emb16, &exp_emb);

        load_fwd_match(&f16_1d, 3);
        load_prefill_match(&f16_1d, &[1, 2, 3]);
        let mut c1 = m16.new_cache(4).expect("c1");
        let mut c2 = m32.new_cache(4).expect("c2");
        let got = m16.forward(&mut c1, 3).expect("f16 1d fwd");
        let twin = m32.forward(&mut c2, 3).expect("f32 twin fwd");
        let exp = oracle_forward(&g16, 3);
        assert_logits_match(&got, &exp);
        assert_logits_match(&got, &twin);
    }

    #[test]
    fn tiny_f16_1d_bias_loads_and_matches_oracle() {
        let bytes = tiny_f16_1d_bias_gguf();
        let g = load_gguf(&bytes).expect("load f16 1d bias");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("qwen2".into()))
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F16);
        assert_eq!(
            g.tensor("blk.0.attn_norm.weight").unwrap().ty,
            GgmlType::F16
        );
        assert_eq!(g.tensor("blk.0.attn_q.bias").unwrap().ty, GgmlType::F16);
        assert_eq!(g.tensor("blk.0.attn_k.bias").unwrap().ty, GgmlType::F16);
        assert_eq!(g.tensor("blk.0.attn_v.bias").unwrap().ty, GgmlType::F16);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::F32);
        load_fwd_match(&bytes, 3);
        load_prefill_match(&bytes, &[1, 2, 3]);
    }

    #[test]
    fn one_d_f16_that_is_not_applied_norm_or_bias_fails_named() {
        let n = TINY_N_EMBD;
        let bytes = write_gguf_with_kv(
            &tiny_kv(&tiny_llama_spec()),
            &[
                tw(
                    "token_embd.weight",
                    GgmlType::F32,
                    vec![n, TINY_N_VOCAB],
                    pack_f32(&pat_f32(n.saturating_mul(TINY_N_VOCAB), 1)),
                ),
                tw(
                    "output.bias",
                    GgmlType::F16,
                    vec![n],
                    pack_f16(&pat_f32(n, 2)),
                ),
                tw(
                    "rope_freqs.weight",
                    GgmlType::F16,
                    vec![n],
                    pack_f16(&pat_f32(n, 3)),
                ),
            ],
        );
        let g = load_gguf(&bytes).expect("load extra 1-D F16");
        for name in ["output.bias", "rope_freqs.weight"] {
            let t = g.tensor(name).expect("extra tensor");
            assert_eq!(t.ty, GgmlType::F16);
            assert_eq!(t.shape.len(), 1);
            let err = match f32s(t) {
                Ok(_) => panic!("f32s should reject 1-D F16 {name}"),
                Err(e) => e.to_string(),
            };
            assert!(err.contains(name), "error should name tensor {name}: {err}");
            assert!(
                err.contains("type 1"),
                "error should name ggml type 1: {err}"
            );
        }

        let ones = vec![1.0f32; n];
        let bad_embd = write_gguf_with_kv(
            &tiny_kv(&tiny_llama_spec()),
            &[
                tw(
                    "token_embd.weight",
                    GgmlType::F16,
                    vec![n],
                    pack_f16(&pat_f32(n, 1)),
                ),
                tw(
                    "output_norm.weight",
                    GgmlType::F32,
                    vec![n],
                    pack_f32(&ones),
                ),
            ],
        );
        let g_bad = load_gguf(&bad_embd).expect("load 1-D F16 token_embd");
        let err = match Llama::from_gguf(g_bad) {
            Ok(_) => panic!("1-D F16 token_embd.weight should fail"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("token_embd.weight"),
            "error should name tensor: {err}"
        );
        assert!(
            err.contains("type 1"),
            "error should name ggml type 1: {err}"
        );
    }

    #[test]
    fn tiny_bf16_logits_match_independent_oracle() {
        let bytes = tiny_bf16_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::BF16);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::BF16);
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::BF16);
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::BF16
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 30);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_q2k_logits_match_independent_oracle() {
        let bytes = tiny_q2k_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::Q2_K);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::Q2_K);
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::Q2_K);
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::Q2_K
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 10);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_q3k_logits_match_independent_oracle() {
        let bytes = tiny_q3k_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::Q3_K);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::Q3_K);
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::Q3_K);
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::Q3_K
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 11);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_q41_logits_match_independent_oracle() {
        let bytes = tiny_q41_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::Q4_1);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::Q4_1);
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::Q4_1);
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::Q4_1
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 3);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_q50_logits_match_independent_oracle() {
        let bytes = tiny_q50_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::Q5_0);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::Q5_0);
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::Q5_0);
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::Q5_0
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 6);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_q51_logits_match_independent_oracle() {
        let bytes = tiny_q51_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::Q5_1);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::Q5_1);
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::Q5_1);
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::Q5_1
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 7);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_mxfp4_logits_match_independent_oracle() {
        let bytes = tiny_mxfp4_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::MXFP4);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::MXFP4);
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::MXFP4);
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::MXFP4
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 39);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_nvfp4_logits_match_independent_oracle() {
        let bytes = tiny_nvfp4_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::NVFP4);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::NVFP4);
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::NVFP4);
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::NVFP4
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 40);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_q10_logits_match_independent_oracle() {
        let bytes = tiny_q10_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::Q1_0);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::Q1_0);
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::Q1_0);
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::Q1_0
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 41);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_q20_logits_match_independent_oracle() {
        let bytes = tiny_q20_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::Q2_0);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::Q2_0);
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::Q2_0);
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::Q2_0
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 42);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_q81_logits_match_independent_oracle() {
        let bytes = tiny_q81_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::Q8_1);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::Q8_1);
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::Q8_1);
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::Q8_1
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(
            g.tensor("blk.0.attn_norm.weight").unwrap().ty,
            GgmlType::F32
        );
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 9);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_tq10_logits_match_independent_oracle() {
        let bytes = tiny_tq10_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::TQ1_0);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::TQ1_0);
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::TQ1_0);
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::TQ1_0
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(
            g.tensor("blk.0.attn_norm.weight").unwrap().ty,
            GgmlType::F32
        );
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 34);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_tq20_logits_match_independent_oracle() {
        let bytes = tiny_tq20_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::TQ2_0);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::TQ2_0);
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::TQ2_0);
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::TQ2_0
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(
            g.tensor("blk.0.attn_norm.weight").unwrap().ty,
            GgmlType::F32
        );
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 35);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_q5k_logits_match_independent_oracle() {
        let bytes = tiny_q5k_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::Q5_K);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::Q5_K);
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::Q5_K);
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::Q5_K
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 13);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_iq4nl_logits_match_independent_oracle() {
        let bytes = tiny_iq4nl_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::IQ4_NL);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::IQ4_NL);
        assert_eq!(
            g.tensor("blk.0.attn_q.weight").unwrap().ty,
            GgmlType::IQ4_NL
        );
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::IQ4_NL
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 20);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_iq2xxs_logits_match_independent_oracle() {
        let bytes = tiny_iq2xxs_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::IQ2_XXS);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::IQ2_XXS);
        assert_eq!(
            g.tensor("blk.0.attn_q.weight").unwrap().ty,
            GgmlType::IQ2_XXS
        );
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::IQ2_XXS
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 16);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_iq2xs_logits_match_independent_oracle() {
        let bytes = tiny_iq2xs_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::IQ2_XS);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::IQ2_XS);
        assert_eq!(
            g.tensor("blk.0.attn_q.weight").unwrap().ty,
            GgmlType::IQ2_XS
        );
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::IQ2_XS
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 17);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_iq1s_logits_match_independent_oracle() {
        let bytes = tiny_iq1s_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::IQ1_S);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::IQ1_S);
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::IQ1_S);
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::IQ1_S
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 19);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_iq1m_logits_match_independent_oracle() {
        let bytes = tiny_iq1m_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::IQ1_M);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::IQ1_M);
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::IQ1_M);
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::IQ1_M
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 29);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_iq2s_logits_match_independent_oracle() {
        let bytes = tiny_iq2s_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::IQ2_S);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::IQ2_S);
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::IQ2_S);
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::IQ2_S
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 22);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_iq3xxs_logits_match_independent_oracle() {
        let bytes = tiny_iq3xxs_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::IQ3_XXS);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::IQ3_XXS);
        assert_eq!(
            g.tensor("blk.0.attn_q.weight").unwrap().ty,
            GgmlType::IQ3_XXS
        );
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::IQ3_XXS
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 18);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_iq3s_logits_match_independent_oracle() {
        let bytes = tiny_iq3s_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::IQ3_S);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::IQ3_S);
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::IQ3_S);
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::IQ3_S
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 21);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn tiny_iq4xs_logits_match_independent_oracle() {
        let bytes = tiny_iq4xs_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::IQ4_XS);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::IQ4_XS);
        assert_eq!(
            g.tensor("blk.0.attn_q.weight").unwrap().ty,
            GgmlType::IQ4_XS
        );
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::IQ4_XS
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 23);
        load_fwd_match(&bytes, 3);
    }

    #[test]
    fn prefill_prompt_logits_match_independent_oracle() {
        let tokens = [1u32, 2, 3];
        for bytes in [
            tiny_llama_gguf(),
            tiny_qwen2_gguf(),
            tiny_q4k_embd_gguf(),
            tiny_q6k_embd_gguf(),
            tiny_f16_gguf(),
            tiny_f16_1d_gguf(),
            tiny_f16_1d_bias_gguf(),
            tiny_bf16_gguf(),
            tiny_q2k_gguf(),
            tiny_q3k_gguf(),
            tiny_q41_gguf(),
            tiny_q50_gguf(),
            tiny_q51_gguf(),
            tiny_mxfp4_gguf(),
            tiny_nvfp4_gguf(),
            tiny_q10_gguf(),
            tiny_q20_gguf(),
            tiny_q81_gguf(),
            tiny_tq10_gguf(),
            tiny_tq20_gguf(),
            tiny_q5k_gguf(),
            tiny_iq4nl_gguf(),
            tiny_iq2xxs_gguf(),
            tiny_iq2xs_gguf(),
            tiny_iq1s_gguf(),
            tiny_iq1m_gguf(),
            tiny_iq2s_gguf(),
            tiny_iq3xxs_gguf(),
            tiny_iq3s_gguf(),
            tiny_iq4xs_gguf(),
            tiny_tied_gguf(),
            tiny_tied_copy_gguf(),
            tiny_gemma_gguf(),
        ] {
            load_prefill_match(&bytes, &tokens);
            load_prefill_match(&bytes, &[3]);
        }
    }

    #[test]
    fn prefill_logits_match_token_by_token_forward() {
        let tokens = [4u32, 3, 1];
        let bytes = tiny_llama_gguf();
        let g = load_gguf(&bytes).expect("load");
        let model = Llama::from_gguf(g.clone()).expect("model");
        let mut batched = model.new_cache(8).expect("c1");
        let pref = model.prefill(&mut batched, &tokens).expect("prefill");
        let model2 = Llama::from_gguf(load_gguf(&bytes).expect("reload")).expect("m2");
        let mut step = model2.new_cache(8).expect("c2");
        let mut last = Vec::new();
        for t in tokens {
            last = model2.forward(&mut step, t).expect("fwd");
        }
        assert_logits_match(&pref, &last);
        assert_eq!(batched.n_past, step.n_past);
        let exp = oracle_forward_seq(&g, &tokens);
        assert_logits_match(&pref, &exp);
    }

    #[test]
    fn tiny_gemma_load_gemv_gemm_embed_and_greedy() {
        let bytes = tiny_gemma_gguf();
        let g = load_gguf(&bytes).expect("load gemma");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("gemma".into()))
        );
        assert_eq!(g.kv_u32("gemma.block_count"), Some(1));
        assert_eq!(g.kv_u32("gemma.embedding_length"), Some(256));
        assert_eq!(g.kv_u32("gemma.feed_forward_length"), Some(256));
        assert_eq!(g.kv_u32("gemma.attention.head_count"), Some(4));
        assert_eq!(g.kv_u32("gemma.attention.head_count_kv"), Some(2));
        assert!(g.kv_u32("llama.block_count").is_none());
        assert!(g.tensor("blk.0.attn_post_norm.weight").is_none());
        assert!(g.tensor("blk.0.ffn_post_norm.weight").is_none());
        let model = Llama::from_gguf(g.clone()).expect("model");
        let x = pat_f32(TINY_N_EMBD, 21);
        let got_gemv = model.gemv_output(&x).expect("gemv");
        let exp_gemv = oracle_gemv(g.tensor("output.weight").unwrap(), &x);
        assert_logits_match(&got_gemv, &exp_gemv);
        let mut x2 = pat_f32(TINY_N_EMBD, 22);
        x2.extend(pat_f32(TINY_N_EMBD, 23));
        let got_gemm = model.gemm_output(2, &x2).expect("gemm");
        let llama_bytes = tiny_llama_gguf();
        let llama_g = load_gguf(&llama_bytes).expect("llama");
        let llama_m = Llama::from_gguf(llama_g).expect("llama m");
        let llama_gemm = llama_m.gemm_output(2, &x2).expect("llama gemm");
        assert_logits_match(&got_gemm, &llama_gemm);
        let emb = model.embed_token(3).expect("embed");
        let exp_emb = oracle_embed(g.tensor("token_embd.weight").unwrap(), 3);
        assert_logits_match(&emb, &exp_emb);
        load_fwd_match(&bytes, 3);
        load_prefill_match(&bytes, &[1, 2, 3]);
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
        let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
        assert_eq!(out, out2);
        assert!(!out.is_empty());
        let llama_fwd = {
            let lg = load_gguf(&llama_bytes).expect("llama reload");
            let lm = Llama::from_gguf(lg).expect("lm");
            let mut c = lm.new_cache(4).expect("c");
            lm.forward(&mut c, 3).expect("llama fwd")
        };
        let mut gc = model.new_cache(4).expect("gc");
        let gemma_fwd = model.forward(&mut gc, 3).expect("gemma fwd");
        assert_ne!(
            gemma_fwd, llama_fwd,
            "gemma embed-scale+GeGLU must change logits vs llama SwiGLU on the same tiny weights"
        );
    }

    #[test]
    fn gemma2_architecture_error_names_arch() {
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("gemma2".into())),
            ],
            &[],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected unknown arch"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("gemma2"), "error should name arch: {err}");
        assert!(
            err.contains("unknown architecture"),
            "error should name unknown architecture: {err}"
        );
    }

    #[test]
    fn tiny_mistral_and_phi3_load_and_greedy() {
        for bytes in [tiny_mistral_gguf(), tiny_phi3_gguf()] {
            let g = load_gguf(&bytes).expect("load");
            let tok = Tokenizer::from_gguf(&g).expect("tok");
            let model = Llama::from_gguf(g.clone()).expect("model");
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
    fn tiny_tied_omits_output_weight_and_reuses_token_embd_range() {
        let tied = tiny_tied_gguf();
        let g = load_gguf(&tied).expect("load tied");
        assert!(g.tensor("output.weight").is_none());
        assert!(g.tensor("token_embd.weight").is_some());
        let model = Llama::from_gguf(g).expect("tied model");
        assert_eq!(model.output_blob_range(), model.token_embd_blob_range());
        let x = vec![0.25f32; TINY_N_EMBD];
        let y_out = model.gemv_output(&x).expect("gemv output");
        let y_emb = model.gemv_token_embd(&x).expect("gemv embd");
        assert_eq!(y_out, y_emb);
    }

    #[test]
    fn tiny_tied_load_gemv_gemm_embed_match_untied_copy_oracle() {
        let tied = tiny_tied_gguf();
        let copy = tiny_tied_copy_gguf();
        let gt = load_gguf(&tied).expect("tied");
        let gc = load_gguf(&copy).expect("copy");
        assert!(gt.tensor("output.weight").is_none());
        let out = gc.tensor("output.weight").expect("copy output");
        let emb = gc.tensor("token_embd.weight").expect("copy embd");
        assert_eq!(out.ty, emb.ty);
        assert_eq!(out.data, emb.data);
        let tied_m = Llama::from_gguf(gt.clone()).expect("tied m");
        let copy_m = Llama::from_gguf(gc.clone()).expect("copy m");
        assert_eq!(tied_m.output_blob_range(), tied_m.token_embd_blob_range());
        assert_ne!(copy_m.output_blob_range(), copy_m.token_embd_blob_range());

        let x = pat_f32(TINY_N_EMBD, 21);
        let tied_gemv = tied_m.gemv_output(&x).expect("tied gemv");
        let copy_gemv = copy_m.gemv_output(&x).expect("copy gemv");
        assert_logits_match(&tied_gemv, &copy_gemv);
        let exp_gemv = oracle_gemv(out, &x);
        assert_logits_match(&tied_gemv, &exp_gemv);

        let mut x2 = pat_f32(TINY_N_EMBD, 22);
        x2.extend(pat_f32(TINY_N_EMBD, 23));
        let tied_gemm = tied_m.gemm_output(2, &x2).expect("tied gemm");
        let copy_gemm = copy_m.gemm_output(2, &x2).expect("copy gemm");
        assert_logits_match(&tied_gemm, &copy_gemm);

        let emb_t = tied_m.embed_token(3).expect("tied embed");
        let emb_c = copy_m.embed_token(3).expect("copy embed");
        assert_logits_match(&emb_t, &emb_c);
        let exp_emb = oracle_embed(gt.tensor("token_embd.weight").unwrap(), 3);
        assert_logits_match(&emb_t, &exp_emb);

        load_fwd_match(&tied, 3);
        load_prefill_match(&tied, &[1, 2, 3]);
        load_fwd_match(&copy, 3);
        load_prefill_match(&copy, &[1, 2, 3]);

        let mut c1 = tied_m.new_cache(4).expect("c1");
        let mut c2 = copy_m.new_cache(4).expect("c2");
        let got = tied_m.forward(&mut c1, 3).expect("tied fwd");
        let exp = oracle_forward(&gc, 3);
        assert_logits_match(&got, &exp);
        let copy_fwd = copy_m.forward(&mut c2, 3).expect("copy fwd");
        assert_logits_match(&got, &copy_fwd);
    }

    #[test]
    fn missing_token_embd_and_output_fails_named() {
        let bytes = write_gguf_with_kv(&tiny_kv(&tiny_llama_spec()), &[]);
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected missing tensor"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("token_embd.weight"),
            "error should name tensor: {err}"
        );
    }

    #[test]
    fn decode_load_type_36_fails_named_removed_and_walk_skips_it() {
        // ggml IQ4_NL_4_4/4_8/8_8 (36..=38) are removed slots, not the next live hole.
        const IQ4_NL_4_4: i32 = 36;
        assert_eq!(
            crate::gguf::classify_ggml_type_id(IQ4_NL_4_4),
            crate::gguf::GgmlTypeClass::Removed
        );
        assert_eq!(
            crate::gguf::next_remaining_live_rejected_ggml_type_id(),
            None,
            "no remaining live rejected ggml weight type after skipping 36..=38"
        );
        let bytes = crate::gguf::write_gguf_with_type_ids(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("qwen2".into())),
            ],
            &[TensorWrite {
                name: "w".into(),
                ty: GgmlType::F32,
                shape: vec![4, 2],
                data: vec![0u8; 32],
            }],
            &[IQ4_NL_4_4],
        );
        let err = match load_gguf(&bytes) {
            Err(e) => e.to_string(),
            Ok(g) => match Llama::from_gguf(g.clone()) {
                Ok(_) => panic!("decode should reject ggml-removed type 36"),
                Err(e) => e.to_string(),
            },
        };
        assert!(
            err.contains(&IQ4_NL_4_4.to_string()),
            "error should include type id {IQ4_NL_4_4}: {err}"
        );
        assert!(
            err.contains("removed"),
            "error should name type 36 as removed: {err}"
        );
    }

    #[test]
    fn missing_tensor_error_names_tensor() {
        let bytes = write_gguf_with_kv(
            &tiny_kv(&TinySpec {
                arch: "llama",
                token_embd: GgmlType::F32,
                output: GgmlType::F32,
                layer: None,
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
        let err = match Llama::from_gguf(g.clone()) {
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
        let err = match Llama::from_gguf(g.clone()) {
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
        let err = match Llama::from_gguf(g.clone()) {
            Ok(_) => panic!("expected missing kv"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("llama.block_count"),
            "error should name kv key: {err}"
        );
    }
}
