//! Llama-family decode on GGUF bytes: RMSNorm, RoPE, GQA+KV, SwiGLU, lm_head.

use crate::gguf::{GgmlType, Gguf, GgufError, Kv, Tensor, TensorWrite};
use crate::quant::{
    dequant_f16_row, dequant_f32_row, dequant_iq3_s_row, dequant_iq3_xxs_row, dequant_iq4_nl_row,
    dequant_iq4_xs_row, dequant_q4_k_row, dequant_q5_k_row, dequant_q6_k_row, f16_row_bytes,
    f32_row_bytes, gemm_f16, gemm_f32, gemm_iq3_s_f32, gemm_iq3_xxs_f32, gemm_iq4_nl_f32,
    gemm_iq4_xs_f32, gemm_q4_k_f32, gemm_q5_k_f32, gemm_q6_k_f32, gemv_f16, gemv_f32,
    gemv_iq3_s_f32, gemv_iq3_xxs_f32, gemv_iq4_nl_f32, gemv_iq4_xs_f32, gemv_q4_k_f32,
    gemv_q5_k_f32, gemv_q6_k_f32, iq3_s_row_bytes, iq3_xxs_row_bytes, iq4_nl_row_bytes,
    iq4_xs_row_bytes, pack_f16, pack_f32, pack_iq3_s_block, pack_iq3_xxs_block, pack_iq4_nl_block,
    pack_iq4_xs_block, pack_q4_k_block, pack_q5_k_block, pack_q6_k_block, q4_k_row_bytes,
    q5_k_row_bytes, q6_k_row_bytes, QuantError, QK4_NL, QK_K,
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
    /// or `phi3`) and `blk.{i}.*` tensor names.
    ///
    /// Takes the GGUF's file blob once. Weight matrices keep offsets into that
    /// blob; they do not clone tensor bytes.
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
        let token_embd = quant_mat(need(&g, "token_embd.weight")?)?;
        let output_norm = f32s(need(&g, "output_norm.weight")?)?;
        let output = quant_mat(need(&g, "output.weight")?)?;
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
            let mut h = silu(&gate)?;
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

/// Writer-built Llama-shaped GGUF: mixed F32/Q4_K/Q6_K weights + tokenizer KV.
pub fn tiny_llama_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::F32,
        output: GgmlType::F32,
        layer: None,
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
    /// Q4_K / Q6_K / F32 mix used by [`tiny_llama_gguf`]. Q5_K / IQ3_XXS / IQ3_S /
    /// IQ4_NL / IQ4_XS are used by [`tiny_q5k_gguf`] / [`tiny_iq3xxs_gguf`] /
    /// [`tiny_iq3s_gguf`] / [`tiny_iq4nl_gguf`] / [`tiny_iq4xs_gguf`].
    layer: Option<GgmlType>,
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
        layer_tw(&spec, "blk.0.attn_k.weight", n_embd, n_kv, 3, GgmlType::F32),
        layer_tw(&spec, "blk.0.attn_v.weight", n_embd, n_kv, 4, GgmlType::F32),
    ];
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

fn pack_mat(ty: GgmlType, n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    match ty {
        GgmlType::Q4_K => pack_q4k_mat(n_cols, n_rows, seed),
        GgmlType::Q5_K => pack_q5k_mat(n_cols, n_rows, seed),
        GgmlType::Q6_K => pack_q6k_mat(n_cols, n_rows, seed),
        GgmlType::IQ3_XXS => pack_iq3xxs_mat(n_cols, n_rows, seed),
        GgmlType::IQ3_S => pack_iq3s_mat(n_cols, n_rows, seed),
        GgmlType::IQ4_NL => pack_iq4nl_mat(n_cols, n_rows, seed),
        GgmlType::IQ4_XS => pack_iq4xs_mat(n_cols, n_rows, seed),
        GgmlType::F16 => pack_f16(&pat_f32(n_cols.saturating_mul(n_rows), seed)),
        _ => pack_f32(&pat_f32(n_cols.saturating_mul(n_rows), seed)),
    }
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

fn f32s(t: Tensor<'_>) -> Result<Vec<f32>, LlamaError> {
    if t.ty != GgmlType::F32 {
        return Err(LlamaError::Type {
            tensor: t.name.to_string(),
            ty: t.ty.to_i32(),
        });
    }
    let (chunks, rem) = t.data.as_chunks::<4>();
    if !rem.is_empty() {
        return Err(LlamaError::Shape(t.name.to_string()));
    }
    Ok(chunks
        .iter()
        .map(|c| f32::from_bits(u32::from_le_bytes(*c)))
        .collect())
}

fn quant_mat(t: Tensor<'_>) -> Result<QuantMat, LlamaError> {
    match t.ty {
        GgmlType::F32
        | GgmlType::F16
        | GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q6_K
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
            GgmlType::Q4_K => gemm_q4_k_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::Q5_K => gemm_q5_k_f32(m.n_cols, n_tokens, data, x, &mut y)?,
            GgmlType::Q6_K => gemm_q6_k_f32(m.n_cols, n_tokens, data, x, &mut y)?,
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
            GgmlType::Q4_K => gemv_q4_k_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::Q5_K => gemv_q5_k_f32(m.n_cols, data, x, &mut y)?,
            GgmlType::Q6_K => gemv_q6_k_f32(m.n_cols, data, x, &mut y)?,
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
            GgmlType::Q4_K => q4_k_row_bytes(emb.n_cols)?,
            GgmlType::Q5_K => q5_k_row_bytes(emb.n_cols)?,
            GgmlType::Q6_K => q6_k_row_bytes(emb.n_cols)?,
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
            GgmlType::Q4_K => dequant_q4_k_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::Q5_K => dequant_q5_k_row(emb.n_cols, bytes, &mut y)?,
            GgmlType::Q6_K => dequant_q6_k_row(emb.n_cols, bytes, &mut y)?,
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
            GgmlType::Q4_K => {
                let rb = (n_cols / QK_K) * crate::quant::Q4_K_BLOCK;
                dequant_q4_k_row(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::Q5_K => {
                let rb = (n_cols / QK_K) * crate::quant::Q5_K_BLOCK;
                dequant_q5_k_row_oracle(&t.data[row * rb..(row + 1) * rb])
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
        let mut k_cache: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_kv];
        let mut v_cache: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_kv];
        let mut last = Vec::new();
        for (pos, &token) in tokens.iter().enumerate() {
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
                .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
                .collect();
            let down = oracle_gemv(g.tensor("blk.0.ffn_down.weight").unwrap(), &h);
            x = down.iter().zip(x.iter()).map(|(a, b)| a + b).collect();
            let on = f32s(g.tensor("output_norm.weight").unwrap()).unwrap();
            x = oracle_rmsnorm(&x, &on, eps);
            last = oracle_gemv(g.tensor("output.weight").unwrap(), &x);
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
            tiny_q5k_gguf(),
            tiny_iq4nl_gguf(),
            tiny_iq3xxs_gguf(),
            tiny_iq3s_gguf(),
            tiny_iq4xs_gguf(),
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
    fn decode_load_unsupported_ggml_type_error_includes_type_id() {
        // ggml IQ2_XXS is 16; remaining IQ* stay rejected after IQ3_XXS shipped.
        const IQ2_XXS: i32 = 16;
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
            &[IQ2_XXS],
        );
        let err = match load_gguf(&bytes) {
            Err(e) => e.to_string(),
            Ok(g) => match Llama::from_gguf(g.clone()) {
                Ok(_) => panic!("decode should reject unknown type"),
                Err(e) => e.to_string(),
            },
        };
        assert!(
            err.contains(&IQ2_XXS.to_string()),
            "error should include type id {IQ2_XXS}: {err}"
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
