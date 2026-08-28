//! Llama-family decode on GGUF bytes: RMSNorm, RoPE, GQA+KV, SwiGLU or Gemma GeGLU, lm_head.
//! Official Qwen3 adds per-head QK-Norm (`attn_q_norm` / `attn_k_norm`) before RoPE.
//! Official Llama4 text (llama.cpp `src/models/llama4.cpp`) adds iRoPE/NoPE, unweighted
//! QK-Norm after RoPE, and interleaved expert FFN (sigmoid top-k + shared expert).
//! Official llama MoE (`architecture=llama` with `n_expert>0`) follows
//! `src/models/llama.cpp` `build_moe_ffn`: softmax, then top-k; SwiGLU weights
//! after the expert with `norm_w` clamp `2^-14`. No Mixtral architecture.
//! Official Qwen2MoE (`architecture=qwen2moe`) follows `src/models/qwen2moe.cpp`:
//! softmax then top-k without `norm_w`; SwiGLU experts; shared expert gated by
//! `silu(x)/x` (sigmoid) on `ffn_gate_inp_shexp`.
//! Official Qwen3MoE (`architecture=qwen3moe`) follows `src/models/qwen3moe.cpp`:
//! Qwen3 QK-Norm before RoPE; `build_moe_ffn` softmax then top-k with `norm_w`
//! clamp `2^-14`; no shared expert. Not Mixtral, not vision.
//! Official Qwen2VL (`architecture=qwen2vl`) follows `src/models/qwen2vl.cpp`:
//! Qwen2 language walk plus m-RoPE (`ggml_rope_multi` / `LLAMA_ROPE_TYPE_MROPE`,
//! `rope.dimension_sections`, text `n_pos_per_embd=4`). Vision / mmproj lives
//! in official `tools/mtmd/models/qwen2vl.cpp` (clip), not a second language arch.
//! Official Qwen3VL (`architecture=qwen3vl`) follows `src/models/qwen3vl.cpp`:
//! Qwen3 QK-Norm plus interleaved m-RoPE (`ggml_rope_multi` /
//! `LLAMA_ROPE_TYPE_IMROPE`, required `rope.dimension_sections`, text
//! `n_pos_per_embd=4`). Vision / mmproj lives in official
//! `tools/mtmd/models/qwen3vl.cpp` (clip), not a second language arch.
//! Official Qwen3Next (`architecture=qwen3next`) follows `src/models/qwen3next.cpp`:
//! gated full attention (joint Q+gate, QK-Norm before RoPE, sigmoid after attn),
//! `post_attention_norm` before MoE, `build_moe_ffn` softmax then top-k with
//! `norm_w` clamp `2^-14`, plus shared expert gated by sigmoid. Official load
//! rejects `n_expert==0`. Not `qwen3vlmoe`. Not Mixtral.
//! Official Qwen35 (`architecture=qwen35`) follows `src/models/qwen35.cpp`:
//! gated full attention (joint Q+gate, QK-Norm before RoPE, IMROPE, sigmoid
//! after attn), official `post_attention_norm`, dense SwiGLU. Linear-attn /
//! gated-delta layers are refused. Not `qwen3vlmoe`. Not Mixtral.
//! Official Phi2 (`architecture=phi2`) follows `src/models/phi2.cpp`:
//! `LLM_NORM` (LayerNorm + bias), `LLAMA_ROPE_TYPE_NEOX`, Q scaled by
//! `1/sqrt(n_embd_head)` then `build_attn` scale `1.0`, parallel residual
//! (`attn` and `LLM_FFN_GELU`/`LLM_FFN_SEQ` both from `attn_norm`), output
//! bias. Not Mixtral, not `qwen3vlmoe`, not linear-attn. Not a phi3 redo.

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
/// Writer-built tiny Llama4 expert count (`llama4.expert_count`). Official load rejects 0.
const TINY_N_EXPERT: usize = 2;
/// Writer-built tiny Llama4 `llama4.expert_used_count` (Scout uses 1).
const TINY_N_EXPERT_USED: usize = 1;
/// Writer-built tiny official llama MoE (`llama.expert_count`). Mixtral-shaped: k < n.
const TINY_LLAMA_N_EXPERT: usize = 4;
/// Writer-built tiny `llama.expert_used_count` (official Mixtral convert uses 2).
const TINY_LLAMA_N_EXPERT_USED: usize = 2;
/// Writer-built tiny Qwen2MoE `qwen2moe.expert_count` (official load rejects 0).
const TINY_QWEN2MOE_N_EXPERT: usize = 4;
/// Writer-built tiny `qwen2moe.expert_used_count` (Qwen1.5-MoE convert uses 4; tiny uses 2).
const TINY_QWEN2MOE_N_EXPERT_USED: usize = 2;
/// Writer-built tiny Qwen3MoE `qwen3moe.expert_count` (official load rejects 0).
const TINY_QWEN3MOE_N_EXPERT: usize = 4;
/// Writer-built tiny `qwen3moe.expert_used_count` (Qwen3-30B-A3B convert uses 8; tiny uses 2).
const TINY_QWEN3MOE_N_EXPERT_USED: usize = 2;
/// Writer-built tiny Qwen3Next `qwen3next.expert_count` (official load rejects 0).
const TINY_QWEN3NEXT_N_EXPERT: usize = 4;
/// Writer-built tiny `qwen3next.expert_used_count`.
const TINY_QWEN3NEXT_N_EXPERT_USED: usize = 2;
/// Official convert: `rope.dimension_count = head_dim * partial_rotary_factor` (default 0.25).
const TINY_QWEN3NEXT_N_ROT: usize = 16;
/// Official default `full_attention_interval` is 4. Writer-tiny uses 1 so layer 0
/// is the official full-attention path (`(il+1) % interval == 0`).
const TINY_QWEN3NEXT_FULL_ATTN_INTERVAL: u32 = 1;
/// Official `LLM_KV_SSM_*` keys are required on load even when no layer is recurrent.
const TINY_QWEN3NEXT_SSM_CONV: u32 = 4;
const TINY_QWEN3NEXT_SSM_STATE: u32 = 16;
const TINY_QWEN3NEXT_SSM_GROUP: u32 = 2;
const TINY_QWEN3NEXT_SSM_DT_RANK: u32 = 2;
const TINY_QWEN3NEXT_SSM_INNER: u32 = 32;
/// Official default `full_attention_interval` is 4. Writer-tiny uses 1 so layer 0
/// is the official full-attention path (`(il+1) % interval == 0`).
const TINY_QWEN35_FULL_ATTN_INTERVAL: u32 = 1;
/// Official `LLM_KV_SSM_*` keys are required on load even when no layer is recurrent.
const TINY_QWEN35_SSM_CONV: u32 = 4;
const TINY_QWEN35_SSM_STATE: u32 = 16;
const TINY_QWEN35_SSM_GROUP: u32 = 2;
const TINY_QWEN35_SSM_DT_RANK: u32 = 2;
const TINY_QWEN35_SSM_INNER: u32 = 32;
/// Official convert `_QWEN35_DEFAULT_MROPE_SECTION` and official GGUF
/// `qwen35.rope.dimension_sections` are `[11, 11, 10, 0]` at `n_rot=64`.
const TINY_QWEN35_ROPE_SECTIONS: [i32; 4] = [11, 11, 10, 0];
/// Official `build_moe_ffn` `norm_w` clamp: smallest f16 (`2^-14`).
const MOE_NORM_W_CLAMP: f32 = 1.0 / 16_384.0;
/// Official llama.cpp `hparams.n_no_rope_layer_step` default for Llama4 text.
const LLAMA4_NO_ROPE_LAYER_STEP: usize = 4;
/// Official Llama4 NoPE temperature floor (`n_attn_temp_floor_scale` = 8192).
const LLAMA4_ATTN_TEMP_FLOOR: f32 = 8192.0;
/// Official Llama4 NoPE temperature scale (`f_attn_temp_scale` = 0.1).
const LLAMA4_ATTN_TEMP_SCALE: f32 = 1.0 / 10.0;
/// Official Llama4 NoPE temperature offset (`f_attn_temp_offset` = 1.0).
const LLAMA4_ATTN_TEMP_OFFSET: f32 = 1.0;
/// Official GGUF `GGUF_TYPE_INT32` (array elem for `rope.dimension_sections`).
const GGUF_TYPE_INT32: i32 = 5;
/// Official Qwen2-VL-2B HF `mrope_section` `[16, 24, 24]` padded to 4 by convert
/// (`[16, 24, 24, 0]`) at `n_rot=128`. Writer-tiny `n_rot=64` uses the same ratio.
const TINY_QWEN2VL_ROPE_SECTIONS: [i32; 4] = [8, 12, 12, 0];
/// Official Qwen3-VL HF `mrope_section` `[24, 20, 20]` padded to 4 by convert
/// (`conversion/base.py`, `[24, 20, 20, 0]`) at `n_rot=128`. Writer-tiny
/// `n_rot=64` uses the same ratio. Official `src/llama-model.cpp` maps
/// `LLM_ARCH_QWEN3VL` to `LLAMA_ROPE_TYPE_IMROPE`.
const TINY_QWEN3VL_ROPE_SECTIONS: [i32; 4] = [12, 10, 10, 0];
/// Official convert `add_feed_forward_length(4 * n_embd)`.
const TINY_PHI2_N_FF: usize = 1024;
/// Official convert `add_head_count_kv(n_head)` (no GQA).
const TINY_PHI2_N_HEAD_KV: usize = TINY_N_HEAD;
/// Official convert `int(partial_rotary_factor * n_embd) // n_head`.
/// microsoft/phi-2 uses `0.4` → `32` of `80`. Writer-tiny uses `32` of `64`
/// (`int(0.5 * 256) // 4`): even `n_rot`, same official formula.
const TINY_PHI2_N_ROT: usize = 32;

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
    /// GGUF dim-2 length. `1` for 2-D weights; expert count for official Llama4 3-D `*_exps`.
    n_parts: usize,
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
        n_parts: token_embd.n_parts,
        start: token_embd.start,
        end: token_embd.end,
    }
}

/// Official llama MoE (`build_moe_ffn` SOFTMAX + `norm_w`, no shared expert).
struct LlamaMoe {
    gate_inp: QuantMat,
    gate_exps: QuantMat,
    up_exps: QuantMat,
    down_exps: QuantMat,
    n_expert: usize,
    n_expert_used: usize,
}

/// Official Llama4 expert FFN (`build_moe_ffn` SIGMOID + shared expert).
struct Llama4Moe {
    gate_inp: QuantMat,
    gate_exps: QuantMat,
    up_exps: QuantMat,
    down_exps: QuantMat,
    gate_shexp: QuantMat,
    up_shexp: QuantMat,
    down_shexp: QuantMat,
    n_expert: usize,
    n_expert_used: usize,
}

/// Official Qwen3MoE (`build_moe_ffn` SOFTMAX + `norm_w`, QK-Norm, no shared expert).
struct Qwen3Moe {
    gate_inp: QuantMat,
    gate_exps: QuantMat,
    up_exps: QuantMat,
    down_exps: QuantMat,
    n_expert: usize,
    n_expert_used: usize,
}

/// Official Qwen2MoE (`build_moe_ffn` SOFTMAX, `norm_w=false`, gated shared expert).
struct Qwen2Moe {
    gate_inp: QuantMat,
    gate_exps: QuantMat,
    up_exps: QuantMat,
    down_exps: QuantMat,
    gate_inp_shexp: QuantMat,
    gate_shexp: QuantMat,
    up_shexp: QuantMat,
    down_shexp: QuantMat,
    n_expert: usize,
    n_expert_used: usize,
}

/// Dense SwiGLU / GeGLU weights (`ffn_gate` / `ffn_up` / `ffn_down`).
struct DenseFfn {
    gate: QuantMat,
    up: QuantMat,
    down: QuantMat,
}

/// Official phi2 sequential GELU (`LLM_FFN_GELU` + `LLM_FFN_SEQ`): no gate.
struct Phi2Ffn {
    up: QuantMat,
    up_b: Vec<f32>,
    down: QuantMat,
    down_b: Vec<f32>,
}

/// Dense SwiGLU / Gemma GeGLU, official llama MoE, Llama4, Qwen2MoE, or Qwen3MoE.
enum LayerFfn {
    /// `ffn_gate` / `ffn_up` / `ffn_down` (dense llama / qwen2 / mistral / phi3 / gemma / qwen3).
    Dense(Box<DenseFfn>),
    /// Official `llama` MoE (`n_expert>0`): routed `*_exps`, softmax then top-k.
    LlamaMoe(Box<LlamaMoe>),
    /// Official `llama4` MoE layer: routed `*_exps` + shared `*_shexp`.
    Llama4Moe(Box<Llama4Moe>),
    /// Official `qwen2moe`: routed `*_exps` + gated shared `*_shexp`.
    Qwen2Moe(Box<Qwen2Moe>),
    /// Official `qwen3moe`: routed `*_exps`, softmax then top-k with `norm_w`.
    Qwen3Moe(Box<Qwen3Moe>),
    /// Official `qwen3next`: routed `*_exps` (`norm_w`) + gated shared `*_shexp`.
    Qwen3Next(Box<Qwen2Moe>),
    /// Official `phi2`: `ffn_up` / `ffn_down` GELU sequential (no `ffn_gate`).
    Phi2(Box<Phi2Ffn>),
}

/// Per-layer weights.
struct Layer {
    attn_norm: Vec<f32>,
    /// Official phi2 `blk.{i}.attn_norm.bias` (`LLM_NORM`).
    attn_norm_b: Option<Vec<f32>>,
    wq: QuantMat,
    bq: Option<Vec<f32>>,
    wk: QuantMat,
    bk: Option<Vec<f32>>,
    wv: QuantMat,
    bv: Option<Vec<f32>>,
    wo: QuantMat,
    /// Official phi2 `blk.{i}.attn_output.bias` (required, flag 0).
    wo_b: Option<Vec<f32>>,
    /// Official Qwen3 / Qwen3MoE / Qwen3VL / Qwen3Next / Qwen35 `blk.{i}.attn_q_norm` (RMSNorm on Q after projection, before RoPE).
    attn_q_norm: Option<Vec<f32>>,
    /// Official Qwen3 / Qwen3MoE / Qwen3VL / Qwen3Next / Qwen35 `blk.{i}.attn_k_norm` (RMSNorm on K after projection, before RoPE).
    attn_k_norm: Option<Vec<f32>>,
    /// Official Qwen3Next / Qwen35: `attn_q` is query+gate (`n_embd_head * n_head * 2`).
    attn_q_gate: bool,
    /// Official Llama4 iRoPE: skip RoPE when `(il+1) % n_no_rope_layer_step == 0`.
    use_rope: bool,
    /// Official Llama4 `Llama4TextL2Norm`: unweighted RMS after RoPE on RoPE layers.
    qk_l2: bool,
    ffn_norm: Vec<f32>,
    ffn: LayerFfn,
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
    /// Official Qwen2VL / Qwen3VL / Qwen35 `{arch}.rope.dimension_sections` for `ggml_rope_multi`.
    rope_sections: Option<[i32; 4]>,
    /// Official Qwen3VL / Qwen35 `LLAMA_ROPE_TYPE_IMROPE` (`ggml_mrope_cache_init` `is_imrope`).
    rope_imrope: bool,
    /// `1` for llama/qwen2/mistral/phi3. Gemma official walk scales embeds by `sqrt(n_embd)`.
    embed_scale: f32,
    /// Official Gemma FFN is `LLM_FFN_GELU` (GeGLU). Other loaded arches stay SwiGLU
    /// (`LLM_FFN_SILU`), including official Qwen3, Llama4, and Qwen3MoE.
    ffn_gelu: bool,
    blob: Vec<u8>,
    token_embd: QuantMat,
    output_norm: Vec<f32>,
    /// Official phi2 `output_norm.bias` (`LLM_NORM`).
    output_norm_b: Option<Vec<f32>>,
    output: QuantMat,
    /// Official phi2 `output.bias` (required, flag 0).
    output_b: Option<Vec<f32>>,
    /// Official `architecture=phi2` language walk (`src/models/phi2.cpp`).
    phi2: bool,
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
    /// `phi3`, `gemma`, `qwen3`, `llama4`, `qwen2moe`, `qwen3moe`, `qwen2vl`,
    /// `qwen3vl`, `qwen3next`, `qwen35`, or `phi2`) and `blk.{i}.*` tensor names. Official llama
    /// MoE is still `architecture=llama` with `n_expert>0`. Official Qwen2VL is
    /// Qwen2 plus m-RoPE (`LLAMA_ROPE_TYPE_MROPE`). Official Qwen3VL is Qwen3
    /// QK-Norm plus interleaved m-RoPE (`LLAMA_ROPE_TYPE_IMROPE`). Official
    /// Qwen3Next is gated full attention plus MoE (`norm_w`) and a sigmoid-gated
    /// shared expert. Official Qwen35 is gated full attention plus IMROPE and
    /// dense SwiGLU; linear-attn / gated-delta layers are refused. Official
    /// Phi2 is LayerNorm + NEOX RoPE + parallel GELU FFN.
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
        // Official qwen3next.cpp `load_arch_hparams` runs before tensors and
        // rejects `n_expert==0` / missing required `ssm.*` KV.
        let qwen3next_hparams = if arch == "qwen3next" {
            Some(load_qwen3next_hparams(&g, arch, n_layer)?)
        } else {
            None
        };
        // Official qwen35.cpp `load_arch_hparams` requires `ssm.*` KV and
        // `rope.dimension_sections` even when the tiny uses full-attn only.
        let qwen35_hparams = if arch == "qwen35" {
            Some(load_qwen35_hparams(&g, arch)?)
        } else {
            None
        };
        let n_rot = rope_dimension(&g, arch, n_embd, n_head)?;
        let rope_sections = if arch == "qwen2vl" || arch == "qwen3vl" || arch == "qwen35" {
            Some(load_rope_dimension_sections(&g, arch)?)
        } else {
            None
        };
        let rope_imrope = arch == "qwen3vl" || arch == "qwen35";
        let n_vocab = g
            .tensor("token_embd.weight")
            .map(|t| t.n_rows())
            .ok_or_else(|| LlamaError::Tensor("token_embd.weight".into()))?;
        let phi2 = arch == "phi2";
        let rms_eps = if phi2 {
            require_f32(&g, arch, "attention.layer_norm_epsilon")?
        } else {
            arch_f32(&g, arch, "attention.layer_norm_rms_epsilon").unwrap_or(1e-5)
        };
        let rope_base = arch_f32(&g, arch, "rope.freq_base").unwrap_or(10_000.0);
        let gemma = arch == "gemma";
        let n_embd_f = f32::from(u16::try_from(n_embd).unwrap_or(1));
        let embed_scale = if gemma { n_embd_f.sqrt() } else { 1.0 };
        let token_embd = quant_mat(need(&g, "token_embd.weight")?)?;
        let output_norm = f32s(need(&g, "output_norm.weight")?)?;
        let output_norm_b = if phi2 {
            Some(f32s(need(&g, "output_norm.bias")?)?)
        } else {
            None
        };
        let output = match g.tensor("output.weight") {
            Some(t) => quant_mat(t)?,
            None => reuse_token_embd_as_output(&token_embd),
        };
        let output_b = if phi2 {
            Some(f32s(need(&g, "output.bias")?)?)
        } else {
            None
        };
        let qk_norm = arch == "qwen3"
            || arch == "qwen3moe"
            || arch == "qwen3vl"
            || arch == "qwen3next"
            || arch == "qwen35";
        let llama4 = arch == "llama4";
        let llama4_hparams = if llama4 {
            Some(load_llama4_hparams(&g, arch, n_layer)?)
        } else {
            None
        };
        let llama_moe_hparams = if arch == "llama" {
            load_llama_moe_hparams(&g, arch)?
        } else {
            None
        };
        let qwen2moe_hparams = if arch == "qwen2moe" {
            Some(load_qwen2moe_hparams(&g, arch)?)
        } else {
            None
        };
        let qwen3moe_hparams = if arch == "qwen3moe" {
            Some(load_qwen3moe_hparams(&g, arch)?)
        } else {
            None
        };
        let layer_h = LayerHparams {
            qk_norm,
            phi2,
            llama4: llama4_hparams.as_ref(),
            llama_moe: llama_moe_hparams.as_ref(),
            qwen2moe: qwen2moe_hparams.as_ref(),
            qwen3moe: qwen3moe_hparams.as_ref(),
            qwen3next: qwen3next_hparams.as_ref(),
            qwen35: qwen35_hparams.as_ref(),
        };
        let mut layers = Vec::new();
        for i in 0..n_layer {
            layers.push(load_layer(&g, i, &layer_h)?);
        }
        Ok(Self {
            n_vocab,
            n_embd,
            n_head,
            n_head_kv,
            n_rot,
            rms_eps,
            rope_base,
            rope_sections,
            rope_imrope,
            embed_scale,
            ffn_gelu: gemma,
            blob: g.into_blob(),
            token_embd,
            output_norm,
            output_norm_b,
            output,
            output_b,
            phi2,
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
            x = if self.phi2 {
                layernorm_rows(
                    &x,
                    self.n_embd,
                    &layer.attn_norm,
                    layer.attn_norm_b.as_deref(),
                    self.rms_eps,
                )?
            } else {
                rmsnorm_rows(&x, self.n_embd, &layer.attn_norm, self.rms_eps)?
            };
            let attn_norm_output = x.clone();
            let q_full = add_bias_rows(
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
            // Official Qwen3Next: joint Q projection is query+gate
            // (`n_embd_head * n_head * 2`), split per head, gate applied after attn.
            let (q, attn_gate) = if layer.attn_q_gate {
                split_qwen3next_q_gate(&q_full, n, self.n_head, hd)?
            } else {
                (q_full, None)
            };
            // Official Qwen3 / Qwen3MoE / Qwen3VL / Qwen3Next: RMSNorm on Q and K
            // after projection, before RoPE (`attn_q_norm` / `attn_k_norm`,
            // per-head, `LLM_NORM_RMS`).
            let q = qk_norm_rows(q, hd, layer.attn_q_norm.as_deref(), self.rms_eps)?;
            let k = qk_norm_rows(k, hd, layer.attn_k_norm.as_deref(), self.rms_eps)?;
            for t in 0..n {
                let pos = n0.saturating_add(t);
                let k_t = token_row(&k, t, layer.wk.n_rows, "prefill k")?;
                let v_t = token_row(&v, t, layer.wv.n_rows, "prefill v")?;
                let mut kh = split_heads(k_t, self.n_head_kv, hd)?;
                let vh = split_heads(v_t, self.n_head_kv, hd)?;
                if layer.use_rope {
                    for h in &mut kh {
                        apply_rope(
                            h,
                            pos,
                            self.n_rot,
                            self.rope_base,
                            self.rope_sections,
                            self.rope_imrope,
                            self.phi2,
                        )?;
                    }
                    if layer.qk_l2 {
                        // Official Llama4: unweighted RMS after RoPE (`Llama4TextL2Norm`).
                        for h in &mut kh {
                            *h = rmsnorm_unweighted(h, self.rms_eps)?;
                        }
                    }
                }
                store_kv(&mut cache.k, li, self.n_head_kv, cache.max_seq, pos, &kh)?;
                store_kv(&mut cache.v, li, self.n_head_kv, cache.max_seq, pos, &vh)?;
            }
            let mut attn = vec![0.0f32; n.saturating_mul(self.n_embd)];
            for t in 0..n {
                let pos = n0.saturating_add(t);
                let q_width = if layer.attn_q_gate {
                    self.n_embd
                } else {
                    layer.wq.n_rows
                };
                let q_t = token_row(&q, t, q_width, "prefill q")?;
                let mut qh = split_heads(q_t, self.n_head, hd)?;
                if layer.use_rope {
                    for h in &mut qh {
                        apply_rope(
                            h,
                            pos,
                            self.n_rot,
                            self.rope_base,
                            self.rope_sections,
                            self.rope_imrope,
                            self.phi2,
                        )?;
                    }
                    if layer.qk_l2 {
                        for h in &mut qh {
                            *h = rmsnorm_unweighted(h, self.rms_eps)?;
                        }
                    }
                    if self.phi2 {
                        // Official phi2.cpp: scale Q after RoPE, then attn scale 1.0.
                        let hd_f = f32::from(u16::try_from(hd).unwrap_or(1));
                        let q_scale = if hd_f > 0.0 { 1.0 / hd_f.sqrt() } else { 0.0 };
                        for h in &mut qh {
                            for v in h.iter_mut() {
                                *v *= q_scale;
                            }
                        }
                    }
                } else {
                    // Official Llama4 NoPE: Q *= attn temperature scale.
                    let scale = llama4_attn_temp_scale(pos);
                    for h in &mut qh {
                        for v in h.iter_mut() {
                            *v *= scale;
                        }
                    }
                }
                let score_scale = if self.phi2 {
                    1.0
                } else {
                    let scale = f32::from(u16::try_from(hd).unwrap_or(1)).sqrt();
                    if scale > 0.0 {
                        1.0 / scale
                    } else {
                        0.0
                    }
                };
                let one = attend_query(
                    cache,
                    li,
                    &qh,
                    self.n_head_kv,
                    hd,
                    pos.saturating_add(1),
                    score_scale,
                )?;
                let dst_off = t.saturating_mul(self.n_embd);
                let dst = attn
                    .get_mut(dst_off..dst_off.saturating_add(self.n_embd))
                    .ok_or_else(|| LlamaError::Shape("prefill attn".into()))?;
                for (d, s) in dst.iter_mut().zip(one.iter()) {
                    *d = *s;
                }
            }
            if let Some(gate) = attn_gate.as_ref() {
                if gate.len() != attn.len() {
                    return Err(LlamaError::Shape("attn gate".into()));
                }
                for (a, g) in attn.iter_mut().zip(gate.iter()) {
                    *a *= sigmoid_f32(*g);
                }
            }
            let proj = add_bias_rows(
                self.gemm_mat(&layer.wo, n, &attn)?,
                layer.wo.n_rows,
                layer.wo_b.as_deref(),
            )?;
            if self.phi2 {
                // Official phi2.cpp: FFN on the same `attn_norm` as attention
                // (parallel residual), then `attn + ffn + inpL`.
                let down = match &layer.ffn {
                    LayerFfn::Phi2(ffn) => {
                        let up = add_bias_rows(
                            self.gemm_mat(&ffn.up, n, &attn_norm_output)?,
                            ffn.up.n_rows,
                            Some(ffn.up_b.as_slice()),
                        )?;
                        let h = gelu(&up)?;
                        add_bias_rows(
                            self.gemm_mat(&ffn.down, n, &h)?,
                            ffn.down.n_rows,
                            Some(ffn.down_b.as_slice()),
                        )?
                    }
                    _ => return Err(LlamaError::Shape("phi2 ffn".into())),
                };
                let summed = add(&proj, &down)?;
                x = add(&summed, &residual)?;
            } else {
                x = add(&proj, &residual)?;
                let residual = x.clone();
                x = rmsnorm_rows(&x, self.n_embd, &layer.ffn_norm, self.rms_eps)?;
                let down = match &layer.ffn {
                    LayerFfn::Dense(dense) => {
                        let gate = self.gemm_mat(&dense.gate, n, &x)?;
                        let up = self.gemm_mat(&dense.up, n, &x)?;
                        let mut h = ffn_gate_act(&gate, self.ffn_gelu)?;
                        for (hv, uv) in h.iter_mut().zip(up.iter()) {
                            *hv *= *uv;
                        }
                        self.gemm_mat(&dense.down, n, &h)?
                    }
                    LayerFfn::LlamaMoe(moe) => self.llama_moe_rows(moe.as_ref(), n, &x)?,
                    LayerFfn::Llama4Moe(moe) => self.llama4_moe_rows(moe.as_ref(), n, &x)?,
                    LayerFfn::Qwen2Moe(moe) => self.qwen2moe_rows(moe.as_ref(), n, &x)?,
                    LayerFfn::Qwen3Moe(moe) => self.qwen3moe_rows(moe.as_ref(), n, &x)?,
                    LayerFfn::Qwen3Next(moe) => self.qwen3next_rows(moe.as_ref(), n, &x)?,
                    LayerFfn::Phi2(_) => return Err(LlamaError::Shape("phi2 ffn".into())),
                };
                x = add(&down, &residual)?;
            }
        }
        let last_off = n.saturating_sub(1).saturating_mul(self.n_embd);
        let last = x
            .get(last_off..last_off.saturating_add(self.n_embd))
            .ok_or_else(|| LlamaError::Shape("prefill last".into()))?;
        let xn = if self.phi2 {
            layernorm(
                last,
                &self.output_norm,
                self.output_norm_b.as_deref(),
                self.rms_eps,
            )?
        } else {
            rmsnorm(last, &self.output_norm, self.rms_eps)?
        };
        let logits = add_bias_rows(
            self.gemv_mat(&self.output, &xn)?,
            self.n_vocab,
            self.output_b.as_deref(),
        )?;
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
    })
}

/// Writer-built Qwen3-shaped GGUF: `qwen3.*` KV plus official QK-Norm tensors.
///
/// Official `general.architecture=qwen3` (`MODEL_ARCH_NAMES[QWEN3] = "qwen3"`,
/// not `qwen3moe`). Decode follows llama.cpp `src/models/qwen3.cpp`: RMSNorm
/// on Q and K after projection / before RoPE (`blk.{i}.attn_q_norm` /
/// `attn_k_norm`, per-head). FFN stays SwiGLU. No embed-scale, GeGLU, or
/// softcap. RMSNorm on GGUF bytes as-is.
pub fn tiny_qwen3_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "qwen3",
        token_embd: GgmlType::F32,
        output: GgmlType::F32,
        layer: None,
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: Some(false),
        llama_moe: false,
    })
}

/// Writer-built Llama4-shaped GGUF: `llama4.*` KV plus official expert FFN tensors.
///
/// Official `general.architecture=llama4` (`MODEL_ARCH_NAMES[LLAMA4] = "llama4"`,
/// not `mixtral` / `qwen3moe`). Decode follows llama.cpp
/// `src/models/llama4.cpp` text walk: iRoPE/NoPE (`n_no_rope_layer_step = 4`),
/// unweighted QK-Norm after RoPE (`Llama4TextL2Norm`, no `attn_q_norm` tensors),
/// expert FFN (`ffn_gate_inp` / `*_exps` / `*_shexp`, sigmoid top-k on raw
/// logits, weight-before-FFN, shared expert). FFN stays SwiGLU. No embed-scale,
/// GeGLU, extra norms, softcap, or vision. RMSNorm on GGUF bytes as-is.
pub fn tiny_llama4_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama4",
        token_embd: GgmlType::F32,
        output: GgmlType::F32,
        layer: None,
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: Some(false),
        llama_moe: false,
    })
}

/// Writer-built official llama MoE GGUF: `architecture=llama` with `n_expert>0`.
///
/// Official convert writes MixtralForCausalLM as `general.architecture=llama`
/// (`MixtralForCausalLM` → `"llama"`). Official `LLM_ARCH_NAMES` has no
/// `"mixtral"`; `general.architecture=mixtral` is `LLM_ARCH_UNKNOWN`. Decode
/// follows llama.cpp `src/models/llama.cpp` `build_moe_ffn`: softmax, then
/// top-k; SwiGLU; weights after the expert with `norm_w` clamp `2^-14`. No
/// shared expert, iRoPE, QK-Norm, or Mixtral architecture. Dense `llama`
/// (`n_expert==0`) stays the existing SwiGLU path.
pub fn tiny_llama_moe_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::F32,
        output: GgmlType::F32,
        layer: None,
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
        llama_moe: true,
    })
}

/// Writer-built official Qwen2MoE GGUF: `architecture=qwen2moe` with `qwen2moe.*` KV.
///
/// Official `LLM_ARCH_NAMES` has `LLM_ARCH_QWEN2MOE = "qwen2moe"`; convert writes
/// `general.architecture=qwen2moe`. Decode follows llama.cpp
/// `src/models/qwen2moe.cpp`: `build_moe_ffn` softmax then top-k, `norm_w=false`;
/// SwiGLU experts; shared expert (`*_shexp` + `ffn_gate_inp_shexp`) gated by
/// `silu(x)/x` (sigmoid). Not `qwen3moe`, not Mixtral, not Llama4 sigmoid /
/// weight-before-FFN, not llama-MoE `norm_w`. No QK-Norm, embed-scale, or vision.
pub fn tiny_qwen2moe_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "qwen2moe",
        token_embd: GgmlType::F32,
        output: GgmlType::F32,
        layer: None,
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: Some(false),
        llama_moe: false,
    })
}

/// Writer-built official Qwen3MoE GGUF: `architecture=qwen3moe` with `qwen3moe.*` KV.
///
/// Official `LLM_ARCH_NAMES` has `LLM_ARCH_QWEN3MOE = "qwen3moe"`; convert writes
/// `general.architecture=qwen3moe`. Decode follows llama.cpp
/// `src/models/qwen3moe.cpp`: Qwen3 QK-Norm (`attn_q_norm` / `attn_k_norm` after
/// projection / before RoPE); `build_moe_ffn` softmax then top-k with
/// `norm_w=true` clamp `2^-14`; no shared expert. Official load rejects
/// `n_expert==0` / `n_expert_used==0`. Tied `output.weight` reuse is allowed.
/// Not Mixtral, not `qwen2moe` shexp, not Llama4 sigmoid / weight-before-FFN,
/// not vision.
pub fn tiny_qwen3moe_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "qwen3moe",
        token_embd: GgmlType::F32,
        output: GgmlType::F32,
        layer: None,
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: Some(false),
        llama_moe: false,
    })
}

/// Writer-built official Qwen2VL GGUF: `architecture=qwen2vl` with `qwen2vl.*` KV.
///
/// Official `LLM_ARCH_NAMES` has `LLM_ARCH_QWEN2VL = "qwen2vl"`; convert writes
/// `general.architecture=qwen2vl` (`Qwen2VLForConditionalGeneration` and
/// `Qwen2_5_VLForConditionalGeneration` → `MODEL_ARCH.QWEN2VL`). Decode follows
/// llama.cpp `src/models/qwen2vl.cpp`: Qwen2 plus `ggml_rope_multi`
/// (`LLAMA_ROPE_TYPE_MROPE`, `rope.dimension_sections`, text positions
/// `[t,h,w,e] = [p,p,p,0]`, `n_pos_per_embd=4`). Vision / mmproj lives in
/// official `tools/mtmd/models/qwen2vl.cpp` (clip), not a second language arch.
/// Not `qwen3vl` / `qwen3vlmoe` / a separate `qwen25vl` language arch. Not Mixtral.
pub fn tiny_qwen2vl_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "qwen2vl",
        token_embd: GgmlType::F32,
        output: GgmlType::F32,
        layer: None,
        rope_dimension_count: false,
        qkv_bias: true,
        add_bos_token: Some(false),
        llama_moe: false,
    })
}

/// Writer-built official Qwen3VL GGUF: `architecture=qwen3vl` with `qwen3vl.*` KV.
///
/// Official `LLM_ARCH_NAMES` has `LLM_ARCH_QWEN3VL = "qwen3vl"`; convert writes
/// `general.architecture=qwen3vl` (`Qwen3VLForConditionalGeneration` →
/// `MODEL_ARCH.QWEN3VL`). Decode follows llama.cpp `src/models/qwen3vl.cpp`:
/// Qwen3 QK-Norm (`attn_q_norm` / `attn_k_norm` after projection / before RoPE)
/// plus `ggml_rope_multi` (`LLAMA_ROPE_TYPE_IMROPE`, required
/// `rope.dimension_sections`, text positions `[t,h,w,e] = [p,p,p,0]`,
/// `n_pos_per_embd=4`). Dense SwiGLU. Tied `output.weight` reuse is allowed.
/// Official `n_deepstack_layers` is optional (`false`) and is vision-side;
/// language-only tiny omits it (default 0). Vision / mmproj lives in official
/// `tools/mtmd/models/qwen3vl.cpp` (clip), not a second language arch.
/// Not `qwen3vlmoe` / `qwen2vl` redo / a separate `qwen25vl` language arch.
/// Not Mixtral. No extra norms.
pub fn tiny_qwen3vl_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "qwen3vl",
        token_embd: GgmlType::F32,
        output: GgmlType::F32,
        layer: None,
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: Some(false),
        llama_moe: false,
    })
}

/// Writer-built official Qwen3Next GGUF: `architecture=qwen3next` with `qwen3next.*` KV.
///
/// Official `LLM_ARCH_NAMES` has `LLM_ARCH_QWEN3NEXT = "qwen3next"`; convert writes
/// `general.architecture=qwen3next` (`Qwen3NextForCausalLM` → `MODEL_ARCH.QWEN3NEXT`).
/// Decode follows llama.cpp `src/models/qwen3next.cpp` language-model walk:
/// gated full attention (joint Q+gate, QK-Norm before RoPE, sigmoid after attn),
/// official `post_attention_norm` (not `ffn_norm`), `build_moe_ffn` softmax then
/// top-k with `norm_w` clamp `2^-14`, plus shared expert gated by sigmoid.
/// Official load rejects `n_expert==0`. Official convert writes
/// `rope.dimension_count = head_dim * partial_rotary_factor` (default 0.25) and
/// required `ssm.*` KV. Writer-tiny uses `full_attention_interval=1` so the
/// single layer is the official full-attention path. Tied `output.weight` reuse
/// is allowed. Not Mixtral, not `qwen3vlmoe`, not a qwen3 / qwen3vl / qwen3moe
/// redo. No invented extra norms.
pub fn tiny_qwen3next_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "qwen3next",
        token_embd: GgmlType::F32,
        output: GgmlType::F32,
        layer: None,
        rope_dimension_count: false,
        qkv_bias: false,
        add_bos_token: Some(false),
        llama_moe: false,
    })
}

/// Writer-built official Qwen35 GGUF: `architecture=qwen35` with `qwen35.*` KV.
///
/// Official `LLM_ARCH_NAMES` has `LLM_ARCH_QWEN35 = "qwen35"`; convert writes
/// `general.architecture=qwen35` (`Qwen3_5ForConditionalGeneration` /
/// `Qwen3_5ForCausalLM` → `MODEL_ARCH.QWEN35`). Decode follows llama.cpp
/// `src/models/qwen35.cpp` language-model full-attention walk: gated Q+gate,
/// QK-Norm before RoPE, `ggml_rope_multi` / `LLAMA_ROPE_TYPE_IMROPE` (required
/// `rope.dimension_sections`, text `[t,h,w,e]=[p,p,p,0]`, `n_pos_per_embd=4`),
/// sigmoid after attn, official `post_attention_norm` (not `ffn_norm`), dense
/// SwiGLU (`LLM_FFN_SILU`; official assert: no `ffn_gate_inp`). Official load
/// requires `ssm.*` KV. Official convert default `mrope_section` is
/// `[11, 11, 10, 0]`. Writer-tiny uses `full_attention_interval=1` so the
/// single layer is the official full-attention path. Linear-attn / gated-delta
/// layers are refused. Tied `output.weight` reuse is allowed. Not Mixtral,
/// not `qwen3vlmoe`, not a qwen3next / qwen3vl / qwen3moe redo. No invented
/// extra norms.
pub fn tiny_qwen35_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "qwen35",
        token_embd: GgmlType::F32,
        output: GgmlType::F32,
        layer: None,
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: Some(false),
        llama_moe: false,
    })
}

/// Writer-built official Phi2 GGUF: `architecture=phi2` with `phi2.*` KV.
///
/// Official `LLM_ARCH_NAMES` has `LLM_ARCH_PHI2 = "phi2"`; convert writes
/// `general.architecture=phi2` (`PhiForCausalLM` → `MODEL_ARCH.PHI2`). Decode
/// follows llama.cpp `src/models/phi2.cpp`: `LLM_NORM` (LayerNorm + bias),
/// `LLAMA_ROPE_TYPE_NEOX`, Q scaled by `1/sqrt(n_embd_head)` then attn scale
/// `1.0`, parallel residual (`LLM_FFN_GELU` / `LLM_FFN_SEQ` from the same
/// `attn_norm`, no `ffn_gate` / no `ffn_norm`), `output.bias`. Official convert
/// writes `layer_norm_epsilon`, `feed_forward_length = 4 * n_embd`,
/// `head_count_kv = n_head`, `rope.dimension_count = int(partial_rotary_factor
/// * n_embd) // n_head`, and `add_bos_token=false`. Tied `output.weight` reuse
/// is allowed. Not Mixtral, not `qwen3vlmoe`, not linear-attn, not a phi3 redo.
pub fn tiny_phi2_gguf() -> Vec<u8> {
    let n_embd = TINY_N_EMBD;
    let n_ff = TINY_PHI2_N_FF;
    let n_vocab = TINY_N_VOCAB;
    let n_head = TINY_N_HEAD;
    let n_kv = TINY_PHI2_N_HEAD_KV.saturating_mul(n_embd / n_head);
    let ones = vec![1.0f32; n_embd];
    let kv = vec![
        (
            "general.alignment".into(),
            Kv::U32(u32::try_from(GGUF_DEFAULT_ALIGNMENT).unwrap_or(32)),
        ),
        ("general.name".into(), Kv::String("llama-rust-tiny".into())),
        ("general.architecture".into(), Kv::String("phi2".into())),
        (
            "phi2.block_count".into(),
            Kv::U32(u32::try_from(TINY_N_LAYER).unwrap_or(0)),
        ),
        (
            "phi2.embedding_length".into(),
            Kv::U32(u32::try_from(n_embd).unwrap_or(0)),
        ),
        (
            "phi2.feed_forward_length".into(),
            Kv::U32(u32::try_from(n_ff).unwrap_or(0)),
        ),
        (
            "phi2.attention.head_count".into(),
            Kv::U32(u32::try_from(n_head).unwrap_or(0)),
        ),
        (
            "phi2.attention.head_count_kv".into(),
            Kv::U32(u32::try_from(TINY_PHI2_N_HEAD_KV).unwrap_or(0)),
        ),
        (
            "phi2.rope.dimension_count".into(),
            Kv::U32(u32::try_from(TINY_PHI2_N_ROT).unwrap_or(0)),
        ),
        ("phi2.rope.freq_base".into(), Kv::F32(10_000.0)),
        (
            "phi2.attention.layer_norm_epsilon".into(),
            Kv::F32(1.0 / 100_000.0),
        ),
        ("tokenizer.ggml.add_bos_token".into(), Kv::Bool(false)),
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
    let tensors = vec![
        tw(
            "token_embd.weight",
            GgmlType::F32,
            vec![n_embd, n_vocab],
            pack_mat(GgmlType::F32, n_embd, n_vocab, 1),
        ),
        tw(
            "output_norm.weight",
            GgmlType::F32,
            vec![n_embd],
            pack_vec1d(GgmlType::F32, &ones),
        ),
        tw(
            "output_norm.bias",
            GgmlType::F32,
            vec![n_embd],
            pack_vec1d(GgmlType::F32, &pat_f32(n_embd, 19)),
        ),
        tw(
            "output.weight",
            GgmlType::F32,
            vec![n_embd, n_vocab],
            pack_mat(GgmlType::F32, n_embd, n_vocab, 2),
        ),
        tw(
            "output.bias",
            GgmlType::F32,
            vec![n_vocab],
            pack_vec1d(GgmlType::F32, &pat_f32(n_vocab, 17)),
        ),
        tw(
            "blk.0.attn_norm.weight",
            GgmlType::F32,
            vec![n_embd],
            pack_vec1d(GgmlType::F32, &ones),
        ),
        tw(
            "blk.0.attn_norm.bias",
            GgmlType::F32,
            vec![n_embd],
            pack_vec1d(GgmlType::F32, &pat_f32(n_embd, 18)),
        ),
        tw(
            "blk.0.attn_k.weight",
            GgmlType::F32,
            vec![n_embd, n_kv],
            pack_mat(GgmlType::F32, n_embd, n_kv, 3),
        ),
        tw(
            "blk.0.attn_v.weight",
            GgmlType::F32,
            vec![n_embd, n_kv],
            pack_mat(GgmlType::F32, n_embd, n_kv, 4),
        ),
        tw(
            "blk.0.attn_q.weight",
            GgmlType::Q4_K,
            vec![n_embd, n_embd],
            pack_mat(GgmlType::Q4_K, n_embd, n_embd, 5),
        ),
        tw(
            "blk.0.attn_output.weight",
            GgmlType::Q4_K,
            vec![n_embd, n_embd],
            pack_mat(GgmlType::Q4_K, n_embd, n_embd, 6),
        ),
        tw(
            "blk.0.ffn_up.weight",
            GgmlType::Q4_K,
            vec![n_embd, n_ff],
            pack_mat(GgmlType::Q4_K, n_embd, n_ff, 7),
        ),
        tw(
            "blk.0.ffn_down.weight",
            GgmlType::Q4_K,
            vec![n_ff, n_embd],
            pack_mat(GgmlType::Q4_K, n_ff, n_embd, 8),
        ),
        tw(
            "blk.0.attn_q.bias",
            GgmlType::F32,
            vec![n_embd],
            pack_vec1d(GgmlType::F32, &pat_f32(n_embd, 11)),
        ),
        tw(
            "blk.0.attn_k.bias",
            GgmlType::F32,
            vec![n_kv],
            pack_vec1d(GgmlType::F32, &pat_f32(n_kv, 12)),
        ),
        tw(
            "blk.0.attn_v.bias",
            GgmlType::F32,
            vec![n_kv],
            pack_vec1d(GgmlType::F32, &pat_f32(n_kv, 13)),
        ),
        tw(
            "blk.0.attn_output.bias",
            GgmlType::F32,
            vec![n_embd],
            pack_vec1d(GgmlType::F32, &pat_f32(n_embd, 14)),
        ),
        tw(
            "blk.0.ffn_up.bias",
            GgmlType::F32,
            vec![n_ff],
            pack_vec1d(GgmlType::F32, &pat_f32(n_ff, 15)),
        ),
        tw(
            "blk.0.ffn_down.bias",
            GgmlType::F32,
            vec![n_embd],
            pack_vec1d(GgmlType::F32, &pat_f32(n_embd, 16)),
        ),
    ];
    write_gguf_with_kv(&kv, &tensors)
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
            llama_moe: false,
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
            llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
        llama_moe: false,
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
    /// Official llama MoE: `architecture=llama` with `n_expert>0` (not a mixtral arch).
    llama_moe: bool,
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
    if spec.arch == "qwen3next" || spec.arch == "qwen35" {
        // Official qwen3next.cpp / qwen35.cpp load `ATTN_POST_NORM`
        // (`post_attention_norm`), not `ffn_norm`.
        tensors.push(tw(
            "blk.0.post_attention_norm.weight",
            vec1d,
            vec![n_embd],
            pack_vec1d(vec1d, &ones),
        ));
    } else {
        tensors.push(tw(
            "blk.0.ffn_norm.weight",
            vec1d,
            vec![n_embd],
            pack_vec1d(vec1d, &ones),
        ));
    }
    if spec.arch == "qwen3"
        || spec.arch == "qwen3moe"
        || spec.arch == "qwen3vl"
        || spec.arch == "qwen3next"
        || spec.arch == "qwen35"
    {
        // Official Qwen3Next / Qwen35 QK-Norm is `{n_embd_head_k}`, not `n_rot`.
        let qk_len = if spec.arch == "qwen3next" || spec.arch == "qwen35" {
            TINY_N_EMBD / TINY_N_HEAD
        } else {
            TINY_N_ROT
        };
        let qk_ones = vec![1.0f32; qk_len];
        tensors.push(tw(
            "blk.0.attn_q_norm.weight",
            vec1d,
            vec![qk_len],
            pack_vec1d(vec1d, &qk_ones),
        ));
        tensors.push(tw(
            "blk.0.attn_k_norm.weight",
            vec1d,
            vec![qk_len],
            pack_vec1d(vec1d, &qk_ones),
        ));
    }
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
    let q_rows = if spec.arch == "qwen3next" || spec.arch == "qwen35" {
        // Official qwen3next.cpp / qwen35.cpp: joint Q+gate is
        // `n_embd_head_k * n_head * 2`.
        n_embd.saturating_mul(2)
    } else {
        n_embd
    };
    tensors.push(layer_tw(
        &spec,
        "blk.0.attn_q.weight",
        n_embd,
        q_rows,
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
    if spec.arch == "llama4" {
        // Official llama4.cpp MoE layer: no dense ffn_gate/up/down.
        tensors.push(tw(
            "blk.0.ffn_gate_inp.weight",
            GgmlType::F32,
            vec![n_embd, TINY_N_EXPERT],
            pack_mat(GgmlType::F32, n_embd, TINY_N_EXPERT, 14),
        ));
        tensors.push(layer_tw_exps(
            &spec,
            "blk.0.ffn_gate_exps.weight",
            n_embd,
            n_ff,
            TINY_N_EXPERT,
            15,
            GgmlType::Q4_K,
        ));
        tensors.push(layer_tw_exps(
            &spec,
            "blk.0.ffn_up_exps.weight",
            n_embd,
            n_ff,
            TINY_N_EXPERT,
            17,
            GgmlType::Q4_K,
        ));
        tensors.push(layer_tw_exps(
            &spec,
            "blk.0.ffn_down_exps.weight",
            n_ff,
            n_embd,
            TINY_N_EXPERT,
            19,
            GgmlType::Q4_K,
        ));
        tensors.push(layer_tw(
            &spec,
            "blk.0.ffn_up_shexp.weight",
            n_embd,
            n_ff,
            7,
            GgmlType::Q4_K,
        ));
        tensors.push(layer_tw(
            &spec,
            "blk.0.ffn_down_shexp.weight",
            n_ff,
            n_embd,
            8,
            GgmlType::Q4_K,
        ));
        tensors.push(layer_tw(
            &spec,
            "blk.0.ffn_gate_shexp.weight",
            n_embd,
            n_ff,
            9,
            GgmlType::Q6_K,
        ));
    } else if spec.llama_moe {
        // Official llama.cpp MoE: architecture=llama, *_exps, no shexp, no mixtral arch.
        tensors.push(tw(
            "blk.0.ffn_gate_inp.weight",
            GgmlType::F32,
            vec![n_embd, TINY_LLAMA_N_EXPERT],
            pack_mat(GgmlType::F32, n_embd, TINY_LLAMA_N_EXPERT, 14),
        ));
        tensors.push(layer_tw_exps(
            &spec,
            "blk.0.ffn_gate_exps.weight",
            n_embd,
            n_ff,
            TINY_LLAMA_N_EXPERT,
            15,
            GgmlType::Q4_K,
        ));
        tensors.push(layer_tw_exps(
            &spec,
            "blk.0.ffn_up_exps.weight",
            n_embd,
            n_ff,
            TINY_LLAMA_N_EXPERT,
            17,
            GgmlType::Q4_K,
        ));
        tensors.push(layer_tw_exps(
            &spec,
            "blk.0.ffn_down_exps.weight",
            n_ff,
            n_embd,
            TINY_LLAMA_N_EXPERT,
            19,
            GgmlType::Q4_K,
        ));
    } else if spec.arch == "qwen2moe" {
        // Official qwen2moe.cpp: *_exps + *_shexp + ffn_gate_inp_shexp. No dense FFN.
        tensors.push(tw(
            "blk.0.ffn_gate_inp.weight",
            GgmlType::F32,
            vec![n_embd, TINY_QWEN2MOE_N_EXPERT],
            pack_mat(GgmlType::F32, n_embd, TINY_QWEN2MOE_N_EXPERT, 14),
        ));
        tensors.push(layer_tw_exps(
            &spec,
            "blk.0.ffn_gate_exps.weight",
            n_embd,
            n_ff,
            TINY_QWEN2MOE_N_EXPERT,
            15,
            GgmlType::Q4_K,
        ));
        tensors.push(layer_tw_exps(
            &spec,
            "blk.0.ffn_up_exps.weight",
            n_embd,
            n_ff,
            TINY_QWEN2MOE_N_EXPERT,
            17,
            GgmlType::Q4_K,
        ));
        tensors.push(layer_tw_exps(
            &spec,
            "blk.0.ffn_down_exps.weight",
            n_ff,
            n_embd,
            TINY_QWEN2MOE_N_EXPERT,
            19,
            GgmlType::Q4_K,
        ));
        tensors.push(tw(
            "blk.0.ffn_gate_inp_shexp.weight",
            GgmlType::F32,
            vec![n_embd],
            pack_mat(GgmlType::F32, n_embd, 1, 20),
        ));
        tensors.push(layer_tw(
            &spec,
            "blk.0.ffn_up_shexp.weight",
            n_embd,
            n_ff,
            7,
            GgmlType::Q4_K,
        ));
        tensors.push(layer_tw(
            &spec,
            "blk.0.ffn_down_shexp.weight",
            n_ff,
            n_embd,
            8,
            GgmlType::Q4_K,
        ));
        tensors.push(layer_tw(
            &spec,
            "blk.0.ffn_gate_shexp.weight",
            n_embd,
            n_ff,
            9,
            GgmlType::Q6_K,
        ));
    } else if spec.arch == "qwen3next" {
        // Official qwen3next.cpp: *_exps + *_shexp + ffn_gate_inp_shexp. No dense FFN.
        tensors.push(tw(
            "blk.0.ffn_gate_inp.weight",
            GgmlType::F32,
            vec![n_embd, TINY_QWEN3NEXT_N_EXPERT],
            pack_mat(GgmlType::F32, n_embd, TINY_QWEN3NEXT_N_EXPERT, 14),
        ));
        tensors.push(layer_tw_exps(
            &spec,
            "blk.0.ffn_gate_exps.weight",
            n_embd,
            n_ff,
            TINY_QWEN3NEXT_N_EXPERT,
            15,
            GgmlType::Q4_K,
        ));
        tensors.push(layer_tw_exps(
            &spec,
            "blk.0.ffn_up_exps.weight",
            n_embd,
            n_ff,
            TINY_QWEN3NEXT_N_EXPERT,
            17,
            GgmlType::Q4_K,
        ));
        tensors.push(layer_tw_exps(
            &spec,
            "blk.0.ffn_down_exps.weight",
            n_ff,
            n_embd,
            TINY_QWEN3NEXT_N_EXPERT,
            19,
            GgmlType::Q4_K,
        ));
        tensors.push(tw(
            "blk.0.ffn_gate_inp_shexp.weight",
            GgmlType::F32,
            vec![n_embd],
            pack_mat(GgmlType::F32, n_embd, 1, 20),
        ));
        tensors.push(layer_tw(
            &spec,
            "blk.0.ffn_up_shexp.weight",
            n_embd,
            n_ff,
            7,
            GgmlType::Q4_K,
        ));
        tensors.push(layer_tw(
            &spec,
            "blk.0.ffn_down_shexp.weight",
            n_ff,
            n_embd,
            8,
            GgmlType::Q4_K,
        ));
        tensors.push(layer_tw(
            &spec,
            "blk.0.ffn_gate_shexp.weight",
            n_embd,
            n_ff,
            9,
            GgmlType::Q6_K,
        ));
    } else if spec.arch == "qwen3moe" {
        // Official qwen3moe.cpp: QK-Norm + *_exps, no shexp, no dense FFN.
        tensors.push(tw(
            "blk.0.ffn_gate_inp.weight",
            GgmlType::F32,
            vec![n_embd, TINY_QWEN3MOE_N_EXPERT],
            pack_mat(GgmlType::F32, n_embd, TINY_QWEN3MOE_N_EXPERT, 14),
        ));
        tensors.push(layer_tw_exps(
            &spec,
            "blk.0.ffn_gate_exps.weight",
            n_embd,
            n_ff,
            TINY_QWEN3MOE_N_EXPERT,
            15,
            GgmlType::Q4_K,
        ));
        tensors.push(layer_tw_exps(
            &spec,
            "blk.0.ffn_up_exps.weight",
            n_embd,
            n_ff,
            TINY_QWEN3MOE_N_EXPERT,
            17,
            GgmlType::Q4_K,
        ));
        tensors.push(layer_tw_exps(
            &spec,
            "blk.0.ffn_down_exps.weight",
            n_ff,
            n_embd,
            TINY_QWEN3MOE_N_EXPERT,
            19,
            GgmlType::Q4_K,
        ));
    } else {
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
    }
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
    s == "llama"
        || s == "qwen2"
        || s == "mistral"
        || s == "phi3"
        || s == "gemma"
        || s == "qwen3"
        || s == "llama4"
        || s == "qwen2moe"
        || s == "qwen3moe"
        || s == "qwen2vl"
        || s == "qwen3vl"
        || s == "qwen3next"
        || s == "qwen35"
        || s == "phi2"
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

fn require_f32(g: &Gguf, arch: &str, field: &str) -> Result<f32, LlamaError> {
    let key = arch_key(arch, field);
    g.kv_f32(&key).ok_or(LlamaError::MissingKv(key))
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
    if arch == "llama4" {
        kv.push((
            arch_key(arch, "expert_count"),
            Kv::U32(u32::try_from(TINY_N_EXPERT).unwrap_or(0)),
        ));
        kv.push((
            arch_key(arch, "expert_used_count"),
            Kv::U32(u32::try_from(TINY_N_EXPERT_USED).unwrap_or(0)),
        ));
        kv.push((arch_key(arch, "interleave_moe_layer_step"), Kv::U32(1)));
        kv.push((
            arch_key(arch, "expert_feed_forward_length"),
            Kv::U32(u32::try_from(TINY_N_FF).unwrap_or(0)),
        ));
    }
    if spec.llama_moe {
        kv.push((
            arch_key(arch, "expert_count"),
            Kv::U32(u32::try_from(TINY_LLAMA_N_EXPERT).unwrap_or(0)),
        ));
        kv.push((
            arch_key(arch, "expert_used_count"),
            Kv::U32(u32::try_from(TINY_LLAMA_N_EXPERT_USED).unwrap_or(0)),
        ));
    }
    if arch == "qwen2moe" {
        kv.push((
            arch_key(arch, "expert_count"),
            Kv::U32(u32::try_from(TINY_QWEN2MOE_N_EXPERT).unwrap_or(0)),
        ));
        kv.push((
            arch_key(arch, "expert_used_count"),
            Kv::U32(u32::try_from(TINY_QWEN2MOE_N_EXPERT_USED).unwrap_or(0)),
        ));
        kv.push((
            arch_key(arch, "expert_feed_forward_length"),
            Kv::U32(u32::try_from(TINY_N_FF).unwrap_or(0)),
        ));
        kv.push((
            arch_key(arch, "expert_shared_feed_forward_length"),
            Kv::U32(u32::try_from(TINY_N_FF).unwrap_or(0)),
        ));
    }
    if arch == "qwen3moe" {
        kv.push((
            arch_key(arch, "expert_count"),
            Kv::U32(u32::try_from(TINY_QWEN3MOE_N_EXPERT).unwrap_or(0)),
        ));
        kv.push((
            arch_key(arch, "expert_used_count"),
            Kv::U32(u32::try_from(TINY_QWEN3MOE_N_EXPERT_USED).unwrap_or(0)),
        ));
        kv.push((
            arch_key(arch, "expert_feed_forward_length"),
            Kv::U32(u32::try_from(TINY_N_FF).unwrap_or(0)),
        ));
    }
    if arch == "qwen3next" {
        kv.push((
            arch_key(arch, "expert_count"),
            Kv::U32(u32::try_from(TINY_QWEN3NEXT_N_EXPERT).unwrap_or(0)),
        ));
        kv.push((
            arch_key(arch, "expert_used_count"),
            Kv::U32(u32::try_from(TINY_QWEN3NEXT_N_EXPERT_USED).unwrap_or(0)),
        ));
        kv.push((
            arch_key(arch, "expert_feed_forward_length"),
            Kv::U32(u32::try_from(TINY_N_FF).unwrap_or(0)),
        ));
        kv.push((
            arch_key(arch, "expert_shared_feed_forward_length"),
            Kv::U32(u32::try_from(TINY_N_FF).unwrap_or(0)),
        ));
        kv.push((
            arch_key(arch, "full_attention_interval"),
            Kv::U32(TINY_QWEN3NEXT_FULL_ATTN_INTERVAL),
        ));
        kv.push((
            arch_key(arch, "rope.dimension_count"),
            Kv::U32(u32::try_from(TINY_QWEN3NEXT_N_ROT).unwrap_or(0)),
        ));
        kv.push((
            arch_key(arch, "ssm.conv_kernel"),
            Kv::U32(TINY_QWEN3NEXT_SSM_CONV),
        ));
        kv.push((
            arch_key(arch, "ssm.inner_size"),
            Kv::U32(TINY_QWEN3NEXT_SSM_INNER),
        ));
        kv.push((
            arch_key(arch, "ssm.state_size"),
            Kv::U32(TINY_QWEN3NEXT_SSM_STATE),
        ));
        kv.push((
            arch_key(arch, "ssm.time_step_rank"),
            Kv::U32(TINY_QWEN3NEXT_SSM_DT_RANK),
        ));
        kv.push((
            arch_key(arch, "ssm.group_count"),
            Kv::U32(TINY_QWEN3NEXT_SSM_GROUP),
        ));
    }
    if arch == "qwen35" {
        kv.push((
            arch_key(arch, "full_attention_interval"),
            Kv::U32(TINY_QWEN35_FULL_ATTN_INTERVAL),
        ));
        kv.push((
            arch_key(arch, "ssm.conv_kernel"),
            Kv::U32(TINY_QWEN35_SSM_CONV),
        ));
        kv.push((
            arch_key(arch, "ssm.inner_size"),
            Kv::U32(TINY_QWEN35_SSM_INNER),
        ));
        kv.push((
            arch_key(arch, "ssm.state_size"),
            Kv::U32(TINY_QWEN35_SSM_STATE),
        ));
        kv.push((
            arch_key(arch, "ssm.time_step_rank"),
            Kv::U32(TINY_QWEN35_SSM_DT_RANK),
        ));
        kv.push((
            arch_key(arch, "ssm.group_count"),
            Kv::U32(TINY_QWEN35_SSM_GROUP),
        ));
    }
    if arch == "qwen2vl" || arch == "qwen3vl" || arch == "qwen35" {
        // Official convert: HF `mrope_section` padded to 4 (`conversion/base.py`).
        // Official qwen35 convert default is `[11, 11, 10, 0]`.
        let sections = if arch == "qwen35" {
            TINY_QWEN35_ROPE_SECTIONS
        } else if arch == "qwen3vl" {
            TINY_QWEN3VL_ROPE_SECTIONS
        } else {
            TINY_QWEN2VL_ROPE_SECTIONS
        };
        kv.push((
            arch_key(arch, "rope.dimension_sections"),
            Kv::Array {
                elem: GGUF_TYPE_INT32,
                items: sections.into_iter().map(Kv::I32).collect(),
            },
        ));
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

fn layer_tw_exps(
    spec: &TinySpec,
    name: &str,
    n_cols: usize,
    n_rows: usize,
    n_expert: usize,
    seed: u32,
    mixed: GgmlType,
) -> TensorWrite {
    let ty = spec.layer.unwrap_or(mixed);
    tw(
        name,
        ty,
        vec![n_cols, n_rows, n_expert],
        pack_mat_exps(ty, n_cols, n_rows, n_expert, seed),
    )
}

fn pack_mat_exps(
    ty: GgmlType,
    n_cols: usize,
    n_rows: usize,
    n_expert: usize,
    seed: u32,
) -> Vec<u8> {
    let mut out = Vec::new();
    for e in 0..n_expert {
        let e32 = u32::try_from(e).unwrap_or(0);
        out.extend(pack_mat(ty, n_cols, n_rows, seed.wrapping_add(e32)));
    }
    out
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

struct LlamaMoeHparams {
    n_expert: usize,
    n_expert_used: usize,
}

struct Llama4Hparams {
    n_expert: usize,
    n_expert_used: usize,
    n_ff_exp: usize,
    n_moe_layer_step: usize,
    n_no_rope_layer_step: usize,
    use_kq_norm: bool,
}

struct Qwen2MoeHparams {
    n_expert: usize,
    n_expert_used: usize,
    n_ff_exp: usize,
    n_ff_shexp: usize,
}

/// Official qwen3moe.cpp: `n_ff_exp` from `expert_feed_forward_length`, else `n_ff / n_expert_used`.
struct Qwen3MoeHparams {
    n_expert: usize,
    n_expert_used: usize,
    n_ff_exp: usize,
}

/// Official qwen3next.cpp: same expert lengths as Qwen2MoE plus hybrid interval.
struct Qwen3NextHparams {
    n_expert: usize,
    n_expert_used: usize,
    n_ff_exp: usize,
    n_ff_shexp: usize,
    full_attn_interval: u32,
}

/// Official qwen35.cpp: dense FFN, hybrid interval, required `ssm.*` KV.
struct Qwen35Hparams {
    full_attn_interval: u32,
}

/// Official llama.cpp: `n_expert==0` (or missing `llama.expert_count`) is dense.
fn load_llama_moe_hparams(g: &Gguf, arch: &str) -> Result<Option<LlamaMoeHparams>, LlamaError> {
    let key = arch_key(arch, "expert_count");
    let Some(v) = g.kv_u32(&key) else {
        return Ok(None);
    };
    let n_expert = usize::try_from(v).map_err(|_| LlamaError::Shape(key.clone()))?;
    if n_expert == 0 {
        return Ok(None);
    }
    let n_expert_used = require_usize(g, arch, "expert_used_count")?;
    if n_expert_used == 0 || n_expert_used > n_expert {
        return Err(LlamaError::Shape(arch_key(arch, "expert_used_count")));
    }
    Ok(Some(LlamaMoeHparams {
        n_expert,
        n_expert_used,
    }))
}

fn load_llama4_hparams(g: &Gguf, arch: &str, n_layer: usize) -> Result<Llama4Hparams, LlamaError> {
    let n_expert = require_usize(g, arch, "expert_count")?;
    if n_expert == 0 {
        return Err(LlamaError::Shape(format!(
            "{arch} model cannot have zero experts"
        )));
    }
    let n_expert_used = require_usize(g, arch, "expert_used_count")?;
    if n_expert_used == 0 || n_expert_used > n_expert {
        return Err(LlamaError::Shape(arch_key(arch, "expert_used_count")));
    }
    let n_ff_exp = require_usize(g, arch, "expert_feed_forward_length")?;
    if n_ff_exp == 0 {
        return Err(LlamaError::Shape(arch_key(
            arch,
            "expert_feed_forward_length",
        )));
    }
    let n_moe_layer_step = require_usize(g, arch, "interleave_moe_layer_step")?;
    // Official llama4.cpp: sliding_window==0 (MobileLLM) sets n_no_rope_layer_step = n_layer.
    // Otherwise the hparam default is 4. convert-hf does not write a NoPE KV.
    let n_no_rope_layer_step = match g.kv_u32(&arch_key(arch, "attention.sliding_window")) {
        Some(0) => n_layer,
        _ => LLAMA4_NO_ROPE_LAYER_STEP,
    };
    // Official: `hparams.use_kq_norm = type != LLM_TYPE_17B_128E` (n_expert == 128).
    let use_kq_norm = n_expert != 128;
    Ok(Llama4Hparams {
        n_expert,
        n_expert_used,
        n_ff_exp,
        n_moe_layer_step,
        n_no_rope_layer_step,
        use_kq_norm,
    })
}

fn llama4_is_moe_layer(i: usize, step: usize) -> bool {
    step > 0 && i.saturating_add(1).is_multiple_of(step)
}

fn llama4_use_rope(i: usize, step: usize) -> bool {
    step > 0 && !i.saturating_add(1).is_multiple_of(step)
}

/// Official qwen2moe.cpp: `n_expert` / `n_expert_used` must be > 0.
/// `n_ff_exp` defaults to `n_ff / n_expert_used`; `n_ff_shexp` defaults to `n_ff`.
fn load_qwen2moe_hparams(g: &Gguf, arch: &str) -> Result<Qwen2MoeHparams, LlamaError> {
    let n_expert = require_usize(g, arch, "expert_count")?;
    if n_expert == 0 {
        return Err(LlamaError::Shape(
            "n_expert must be > 0 for QWEN2MOE".into(),
        ));
    }
    let n_expert_used = require_usize(g, arch, "expert_used_count")?;
    if n_expert_used == 0 {
        return Err(LlamaError::Shape(
            "n_expert_used must be > 0 for QWEN2MOE".into(),
        ));
    }
    if n_expert_used > n_expert {
        return Err(LlamaError::Shape(arch_key(arch, "expert_used_count")));
    }
    let n_ff = require_usize(g, arch, "feed_forward_length")?;
    if n_ff == 0 {
        return Err(LlamaError::Shape(arch_key(arch, "feed_forward_length")));
    }
    let n_ff_exp = match g.kv_u32(&arch_key(arch, "expert_feed_forward_length")) {
        Some(v) if v > 0 => usize::try_from(v)
            .map_err(|_| LlamaError::Shape(arch_key(arch, "expert_feed_forward_length")))?,
        _ => {
            if !n_ff.is_multiple_of(n_expert_used) {
                return Err(LlamaError::Shape(arch_key(
                    arch,
                    "expert_feed_forward_length",
                )));
            }
            n_ff / n_expert_used
        }
    };
    let n_ff_shexp = match g.kv_u32(&arch_key(arch, "expert_shared_feed_forward_length")) {
        Some(v) if v > 0 => usize::try_from(v)
            .map_err(|_| LlamaError::Shape(arch_key(arch, "expert_shared_feed_forward_length")))?,
        _ => n_ff,
    };
    Ok(Qwen2MoeHparams {
        n_expert,
        n_expert_used,
        n_ff_exp,
        n_ff_shexp,
    })
}

/// Official qwen2vl.cpp / qwen3vl.cpp / qwen35.cpp: `LLM_KV_ROPE_DIMENSION_SECTIONS` is required (`true`).
fn load_rope_dimension_sections(g: &Gguf, arch: &str) -> Result<[i32; 4], LlamaError> {
    let key = arch_key(arch, "rope.dimension_sections");
    let items = g
        .kv_i32s(&key)
        .ok_or_else(|| LlamaError::MissingKv(key.clone()))?;
    if items.len() != 4 {
        return Err(LlamaError::Shape(key));
    }
    let mut out = [0i32; 4];
    for (dst, src) in out.iter_mut().zip(items.iter()) {
        if *src < 0 {
            return Err(LlamaError::Shape(key.clone()));
        }
        *dst = *src;
    }
    if out.iter().all(|s| *s == 0) {
        return Err(LlamaError::Shape(key));
    }
    Ok(out)
}

/// Official qwen3moe.cpp: `n_expert` / `n_expert_used` must be > 0.
/// `n_ff_exp` defaults to `n_ff / n_expert_used`. No shared expert.
fn load_qwen3moe_hparams(g: &Gguf, arch: &str) -> Result<Qwen3MoeHparams, LlamaError> {
    let n_expert = require_usize(g, arch, "expert_count")?;
    if n_expert == 0 {
        return Err(LlamaError::Shape(
            "n_expert must be > 0 for QWEN3MOE".into(),
        ));
    }
    let n_expert_used = require_usize(g, arch, "expert_used_count")?;
    if n_expert_used == 0 {
        return Err(LlamaError::Shape(
            "n_expert_used must be > 0 for QWEN3MOE".into(),
        ));
    }
    if n_expert_used > n_expert {
        return Err(LlamaError::Shape(arch_key(arch, "expert_used_count")));
    }
    let n_ff = require_usize(g, arch, "feed_forward_length")?;
    if n_ff == 0 {
        return Err(LlamaError::Shape(arch_key(arch, "feed_forward_length")));
    }
    let n_ff_exp = match g.kv_u32(&arch_key(arch, "expert_feed_forward_length")) {
        Some(v) if v > 0 => usize::try_from(v)
            .map_err(|_| LlamaError::Shape(arch_key(arch, "expert_feed_forward_length")))?,
        _ => {
            if !n_ff.is_multiple_of(n_expert_used) {
                return Err(LlamaError::Shape(arch_key(
                    arch,
                    "expert_feed_forward_length",
                )));
            }
            n_ff / n_expert_used
        }
    };
    Ok(Qwen3MoeHparams {
        n_expert,
        n_expert_used,
        n_ff_exp,
    })
}

/// Official qwen3next.cpp: `n_expert` must be > 0. `LLM_KV_SSM_*` is required.
/// `full_attention_interval` defaults to 4. Layer `i` is recurrent when
/// `(i+1) % interval != 0`.
fn load_qwen3next_hparams(
    g: &Gguf,
    arch: &str,
    _n_layer: usize,
) -> Result<Qwen3NextHparams, LlamaError> {
    let n_expert = require_usize(g, arch, "expert_count")?;
    if n_expert == 0 {
        return Err(LlamaError::Shape(format!(
            "{arch} model cannot have zero experts"
        )));
    }
    let n_expert_used = require_usize(g, arch, "expert_used_count")?;
    if n_expert_used == 0 {
        return Err(LlamaError::Shape(
            "n_expert_used must be > 0 for QWEN3NEXT".into(),
        ));
    }
    if n_expert_used > n_expert {
        return Err(LlamaError::Shape(arch_key(arch, "expert_used_count")));
    }
    let n_ff = require_usize(g, arch, "feed_forward_length")?;
    if n_ff == 0 {
        return Err(LlamaError::Shape(arch_key(arch, "feed_forward_length")));
    }
    let n_ff_exp = match g.kv_u32(&arch_key(arch, "expert_feed_forward_length")) {
        Some(v) if v > 0 => usize::try_from(v)
            .map_err(|_| LlamaError::Shape(arch_key(arch, "expert_feed_forward_length")))?,
        _ => {
            if !n_ff.is_multiple_of(n_expert_used) {
                return Err(LlamaError::Shape(arch_key(
                    arch,
                    "expert_feed_forward_length",
                )));
            }
            n_ff / n_expert_used
        }
    };
    let n_ff_shexp = match g.kv_u32(&arch_key(arch, "expert_shared_feed_forward_length")) {
        Some(v) if v > 0 => usize::try_from(v)
            .map_err(|_| LlamaError::Shape(arch_key(arch, "expert_shared_feed_forward_length")))?,
        _ => n_ff,
    };
    // Official load_arch_hparams requires these SSM keys (no optional flag).
    let _ = require_usize(g, arch, "ssm.conv_kernel")?;
    let _ = require_usize(g, arch, "ssm.inner_size")?;
    let _ = require_usize(g, arch, "ssm.state_size")?;
    let _ = require_usize(g, arch, "ssm.time_step_rank")?;
    let _ = require_usize(g, arch, "ssm.group_count")?;
    let full_attn_interval = g
        .kv_u32(&arch_key(arch, "full_attention_interval"))
        .unwrap_or(4);
    if full_attn_interval == 0 {
        return Err(LlamaError::Shape(arch_key(arch, "full_attention_interval")));
    }
    Ok(Qwen3NextHparams {
        n_expert,
        n_expert_used,
        n_ff_exp,
        n_ff_shexp,
        full_attn_interval,
    })
}

fn qwen3next_is_recr(i: usize, interval: u32) -> bool {
    let step = i.saturating_add(1);
    let iv = usize::try_from(interval).unwrap_or(0);
    iv == 0 || !step.is_multiple_of(iv)
}

/// Official qwen35.cpp `load_arch_hparams`: `LLM_KV_SSM_*` is required.
/// `full_attention_interval` defaults to 4. Layer `i` is recurrent when
/// `(i+1) % interval != 0` (same formula as official qwen3next.cpp).
/// Dense FFN; no `n_expert`. Linear-attn / gated-delta is refused at layer load.
fn load_qwen35_hparams(g: &Gguf, arch: &str) -> Result<Qwen35Hparams, LlamaError> {
    // Official load_arch_hparams requires these SSM keys (no optional flag).
    let _ = require_usize(g, arch, "ssm.conv_kernel")?;
    let _ = require_usize(g, arch, "ssm.inner_size")?;
    let _ = require_usize(g, arch, "ssm.state_size")?;
    let _ = require_usize(g, arch, "ssm.time_step_rank")?;
    let _ = require_usize(g, arch, "ssm.group_count")?;
    let full_attn_interval = g
        .kv_u32(&arch_key(arch, "full_attention_interval"))
        .unwrap_or(4);
    if full_attn_interval == 0 {
        return Err(LlamaError::Shape(arch_key(arch, "full_attention_interval")));
    }
    Ok(Qwen35Hparams { full_attn_interval })
}

struct LayerHparams<'a> {
    qk_norm: bool,
    phi2: bool,
    llama4: Option<&'a Llama4Hparams>,
    llama_moe: Option<&'a LlamaMoeHparams>,
    qwen2moe: Option<&'a Qwen2MoeHparams>,
    qwen3moe: Option<&'a Qwen3MoeHparams>,
    qwen3next: Option<&'a Qwen3NextHparams>,
    qwen35: Option<&'a Qwen35Hparams>,
}

fn load_layer(g: &Gguf, i: usize, h: &LayerHparams<'_>) -> Result<Layer, LlamaError> {
    let llama4 = h.llama4;
    let llama_moe = h.llama_moe;
    let qwen2moe = h.qwen2moe;
    let qwen3moe = h.qwen3moe;
    let qwen3next = h.qwen3next;
    let qwen35 = h.qwen35;
    let qk_norm = h.qk_norm;
    let use_rope = match llama4 {
        Some(h) => llama4_use_rope(i, h.n_no_rope_layer_step),
        None => true,
    };
    let qk_l2 = match llama4 {
        Some(h) => use_rope && h.use_kq_norm,
        None => false,
    };
    let ffn = if let Some(h) = llama4 {
        if llama4_is_moe_layer(i, h.n_moe_layer_step) {
            LayerFfn::Llama4Moe(Box::new(load_llama4_moe(g, i, h)?))
        } else {
            LayerFfn::Dense(Box::new(DenseFfn {
                gate: quant_mat(need(g, &format!("blk.{i}.ffn_gate.weight"))?)?,
                up: quant_mat(need(g, &format!("blk.{i}.ffn_up.weight"))?)?,
                down: quant_mat(need(g, &format!("blk.{i}.ffn_down.weight"))?)?,
            }))
        }
    } else if let Some(h) = llama_moe {
        LayerFfn::LlamaMoe(Box::new(load_llama_moe(g, i, h)?))
    } else if let Some(h) = qwen2moe {
        LayerFfn::Qwen2Moe(Box::new(load_qwen2moe(g, i, h)?))
    } else if let Some(h) = qwen3moe {
        LayerFfn::Qwen3Moe(Box::new(load_qwen3moe(g, i, h)?))
    } else if let Some(h) = qwen3next {
        if qwen3next_is_recr(i, h.full_attn_interval) {
            return Err(LlamaError::Shape(format!(
                "qwen3next linear attention layer {i}"
            )));
        }
        let q2 = Qwen2MoeHparams {
            n_expert: h.n_expert,
            n_expert_used: h.n_expert_used,
            n_ff_exp: h.n_ff_exp,
            n_ff_shexp: h.n_ff_shexp,
        };
        LayerFfn::Qwen3Next(Box::new(load_qwen2moe(g, i, &q2)?))
    } else if let Some(h) = qwen35 {
        if qwen3next_is_recr(i, h.full_attn_interval) {
            return Err(LlamaError::Shape(format!(
                "qwen35 linear attention layer {i}"
            )));
        }
        LayerFfn::Dense(Box::new(DenseFfn {
            gate: quant_mat(need(g, &format!("blk.{i}.ffn_gate.weight"))?)?,
            up: quant_mat(need(g, &format!("blk.{i}.ffn_up.weight"))?)?,
            down: quant_mat(need(g, &format!("blk.{i}.ffn_down.weight"))?)?,
        }))
    } else if h.phi2 {
        LayerFfn::Phi2(Box::new(Phi2Ffn {
            up: quant_mat(need(g, &format!("blk.{i}.ffn_up.weight"))?)?,
            up_b: f32s(need(g, &format!("blk.{i}.ffn_up.bias"))?)?,
            down: quant_mat(need(g, &format!("blk.{i}.ffn_down.weight"))?)?,
            down_b: f32s(need(g, &format!("blk.{i}.ffn_down.bias"))?)?,
        }))
    } else {
        LayerFfn::Dense(Box::new(DenseFfn {
            gate: quant_mat(need(g, &format!("blk.{i}.ffn_gate.weight"))?)?,
            up: quant_mat(need(g, &format!("blk.{i}.ffn_up.weight"))?)?,
            down: quant_mat(need(g, &format!("blk.{i}.ffn_down.weight"))?)?,
        }))
    };
    Ok(Layer {
        attn_norm: f32s(need(g, &format!("blk.{i}.attn_norm.weight"))?)?,
        attn_norm_b: if h.phi2 {
            Some(f32s(need(g, &format!("blk.{i}.attn_norm.bias"))?)?)
        } else {
            None
        },
        wq: quant_mat(need(g, &format!("blk.{i}.attn_q.weight"))?)?,
        bq: optional_f32(g, &format!("blk.{i}.attn_q.bias"))?,
        wk: quant_mat(need(g, &format!("blk.{i}.attn_k.weight"))?)?,
        bk: optional_f32(g, &format!("blk.{i}.attn_k.bias"))?,
        wv: quant_mat(need(g, &format!("blk.{i}.attn_v.weight"))?)?,
        bv: optional_f32(g, &format!("blk.{i}.attn_v.bias"))?,
        wo: quant_mat(need(g, &format!("blk.{i}.attn_output.weight"))?)?,
        wo_b: if h.phi2 {
            Some(f32s(need(g, &format!("blk.{i}.attn_output.bias"))?)?)
        } else {
            None
        },
        attn_q_norm: if qk_norm {
            Some(f32s(need(g, &format!("blk.{i}.attn_q_norm.weight"))?)?)
        } else {
            None
        },
        attn_k_norm: if qk_norm {
            Some(f32s(need(g, &format!("blk.{i}.attn_k_norm.weight"))?)?)
        } else {
            None
        },
        use_rope,
        qk_l2,
        attn_q_gate: qwen3next.is_some() || qwen35.is_some(),
        ffn_norm: if h.phi2 {
            Vec::new()
        } else if qwen3next.is_some() || qwen35.is_some() {
            f32s(need(g, &format!("blk.{i}.post_attention_norm.weight"))?)?
        } else {
            f32s(need(g, &format!("blk.{i}.ffn_norm.weight"))?)?
        },
        ffn,
    })
}

fn load_llama_moe(g: &Gguf, i: usize, h: &LlamaMoeHparams) -> Result<LlamaMoe, LlamaError> {
    let gate_inp = quant_mat(need(g, &format!("blk.{i}.ffn_gate_inp.weight"))?)?;
    let gate_exps = quant_mat(need(g, &format!("blk.{i}.ffn_gate_exps.weight"))?)?;
    let up_exps = quant_mat(need(g, &format!("blk.{i}.ffn_up_exps.weight"))?)?;
    let down_exps = quant_mat(need(g, &format!("blk.{i}.ffn_down_exps.weight"))?)?;
    if gate_inp.n_rows != h.n_expert || gate_inp.n_parts != 1 {
        return Err(LlamaError::Shape(format!("blk.{i}.ffn_gate_inp.weight")));
    }
    for (t, name) in [
        (&gate_exps, format!("blk.{i}.ffn_gate_exps.weight")),
        (&up_exps, format!("blk.{i}.ffn_up_exps.weight")),
        (&down_exps, format!("blk.{i}.ffn_down_exps.weight")),
    ] {
        if t.n_parts != h.n_expert {
            return Err(LlamaError::Shape(name));
        }
    }
    if gate_exps.n_rows != up_exps.n_rows || down_exps.n_cols != up_exps.n_rows {
        return Err(LlamaError::Shape(format!("blk.{i}.ffn_*_exps.weight")));
    }
    Ok(LlamaMoe {
        gate_inp,
        gate_exps,
        up_exps,
        down_exps,
        n_expert: h.n_expert,
        n_expert_used: h.n_expert_used,
    })
}

fn load_llama4_moe(g: &Gguf, i: usize, h: &Llama4Hparams) -> Result<Llama4Moe, LlamaError> {
    let gate_inp = quant_mat(need(g, &format!("blk.{i}.ffn_gate_inp.weight"))?)?;
    let gate_exps = quant_mat(need(g, &format!("blk.{i}.ffn_gate_exps.weight"))?)?;
    let up_exps = quant_mat(need(g, &format!("blk.{i}.ffn_up_exps.weight"))?)?;
    let down_exps = quant_mat(need(g, &format!("blk.{i}.ffn_down_exps.weight"))?)?;
    let gate_shexp = quant_mat(need(g, &format!("blk.{i}.ffn_gate_shexp.weight"))?)?;
    let up_shexp = quant_mat(need(g, &format!("blk.{i}.ffn_up_shexp.weight"))?)?;
    let down_shexp = quant_mat(need(g, &format!("blk.{i}.ffn_down_shexp.weight"))?)?;
    if gate_inp.n_rows != h.n_expert || gate_inp.n_parts != 1 {
        return Err(LlamaError::Shape(format!("blk.{i}.ffn_gate_inp.weight")));
    }
    for (t, name) in [
        (&gate_exps, format!("blk.{i}.ffn_gate_exps.weight")),
        (&up_exps, format!("blk.{i}.ffn_up_exps.weight")),
        (&down_exps, format!("blk.{i}.ffn_down_exps.weight")),
    ] {
        if t.n_parts != h.n_expert {
            return Err(LlamaError::Shape(name));
        }
    }
    if gate_exps.n_rows != h.n_ff_exp
        || up_exps.n_rows != h.n_ff_exp
        || down_exps.n_cols != h.n_ff_exp
    {
        return Err(LlamaError::Shape(format!("blk.{i}.ffn_*_exps.weight")));
    }
    Ok(Llama4Moe {
        gate_inp,
        gate_exps,
        up_exps,
        down_exps,
        gate_shexp,
        up_shexp,
        down_shexp,
        n_expert: h.n_expert,
        n_expert_used: h.n_expert_used,
    })
}

fn load_qwen2moe(g: &Gguf, i: usize, h: &Qwen2MoeHparams) -> Result<Qwen2Moe, LlamaError> {
    let gate_inp = quant_mat(need(g, &format!("blk.{i}.ffn_gate_inp.weight"))?)?;
    let gate_exps = quant_mat(need(g, &format!("blk.{i}.ffn_gate_exps.weight"))?)?;
    let up_exps = quant_mat(need(g, &format!("blk.{i}.ffn_up_exps.weight"))?)?;
    let down_exps = quant_mat(need(g, &format!("blk.{i}.ffn_down_exps.weight"))?)?;
    let gate_inp_shexp = quant_mat(need(g, &format!("blk.{i}.ffn_gate_inp_shexp.weight"))?)?;
    let gate_shexp = quant_mat(need(g, &format!("blk.{i}.ffn_gate_shexp.weight"))?)?;
    let up_shexp = quant_mat(need(g, &format!("blk.{i}.ffn_up_shexp.weight"))?)?;
    let down_shexp = quant_mat(need(g, &format!("blk.{i}.ffn_down_shexp.weight"))?)?;
    if gate_inp.n_rows != h.n_expert || gate_inp.n_parts != 1 {
        return Err(LlamaError::Shape(format!("blk.{i}.ffn_gate_inp.weight")));
    }
    for (t, name) in [
        (&gate_exps, format!("blk.{i}.ffn_gate_exps.weight")),
        (&up_exps, format!("blk.{i}.ffn_up_exps.weight")),
        (&down_exps, format!("blk.{i}.ffn_down_exps.weight")),
    ] {
        if t.n_parts != h.n_expert {
            return Err(LlamaError::Shape(name));
        }
    }
    if gate_exps.n_rows != h.n_ff_exp
        || up_exps.n_rows != h.n_ff_exp
        || down_exps.n_cols != h.n_ff_exp
    {
        return Err(LlamaError::Shape(format!("blk.{i}.ffn_*_exps.weight")));
    }
    if gate_inp_shexp.n_cols != down_shexp.n_rows
        || gate_inp_shexp.n_rows != 1
        || gate_inp_shexp.n_parts != 1
    {
        return Err(LlamaError::Shape(format!(
            "blk.{i}.ffn_gate_inp_shexp.weight"
        )));
    }
    if gate_shexp.n_rows != h.n_ff_shexp
        || up_shexp.n_rows != h.n_ff_shexp
        || down_shexp.n_cols != h.n_ff_shexp
    {
        return Err(LlamaError::Shape(format!("blk.{i}.ffn_*_shexp.weight")));
    }
    Ok(Qwen2Moe {
        gate_inp,
        gate_exps,
        up_exps,
        down_exps,
        gate_inp_shexp,
        gate_shexp,
        up_shexp,
        down_shexp,
        n_expert: h.n_expert,
        n_expert_used: h.n_expert_used,
    })
}

fn load_qwen3moe(g: &Gguf, i: usize, h: &Qwen3MoeHparams) -> Result<Qwen3Moe, LlamaError> {
    let gate_inp = quant_mat(need(g, &format!("blk.{i}.ffn_gate_inp.weight"))?)?;
    let gate_exps = quant_mat(need(g, &format!("blk.{i}.ffn_gate_exps.weight"))?)?;
    let up_exps = quant_mat(need(g, &format!("blk.{i}.ffn_up_exps.weight"))?)?;
    let down_exps = quant_mat(need(g, &format!("blk.{i}.ffn_down_exps.weight"))?)?;
    if gate_inp.n_rows != h.n_expert || gate_inp.n_parts != 1 {
        return Err(LlamaError::Shape(format!("blk.{i}.ffn_gate_inp.weight")));
    }
    for (t, name) in [
        (&gate_exps, format!("blk.{i}.ffn_gate_exps.weight")),
        (&up_exps, format!("blk.{i}.ffn_up_exps.weight")),
        (&down_exps, format!("blk.{i}.ffn_down_exps.weight")),
    ] {
        if t.n_parts != h.n_expert {
            return Err(LlamaError::Shape(name));
        }
    }
    if gate_exps.n_rows != h.n_ff_exp
        || up_exps.n_rows != h.n_ff_exp
        || down_exps.n_cols != h.n_ff_exp
    {
        return Err(LlamaError::Shape(format!("blk.{i}.ffn_*_exps.weight")));
    }
    Ok(Qwen3Moe {
        gate_inp,
        gate_exps,
        up_exps,
        down_exps,
        n_expert: h.n_expert,
        n_expert_used: h.n_expert_used,
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
        || name == "output_norm.bias"
        || name == "output.bias"
        || name.ends_with(".attn_norm.weight")
        || name.ends_with(".attn_norm.bias")
        || name.ends_with(".ffn_norm.weight")
        || name.ends_with(".attn_q_norm.weight")
        || name.ends_with(".attn_k_norm.weight")
        || name.ends_with(".attn_q.bias")
        || name.ends_with(".attn_k.bias")
        || name.ends_with(".attn_v.bias")
        || name.ends_with(".attn_output.bias")
        || name.ends_with(".ffn_up.bias")
        || name.ends_with(".ffn_down.bias")
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
            if t.shape.len() > 3 {
                return Err(LlamaError::Shape(t.name.to_string()));
            }
            let n_parts = match t.shape.get(2) {
                Some(&d) => {
                    usize::try_from(d).map_err(|_| LlamaError::Shape(t.name.to_string()))?
                }
                None => 1,
            };
            if n_parts == 0 {
                return Err(LlamaError::Shape(t.name.to_string()));
            }
            let (start, end) = t.blob_range();
            Ok(QuantMat {
                name: t.name.to_string(),
                ty: t.ty,
                n_cols: t.n_cols(),
                n_rows: t.n_rows(),
                n_parts,
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

    /// Official llama.cpp `build_moe_ffn` for `architecture=llama` + `n_expert>0`:
    /// softmax over all experts, then top-k; SwiGLU; weights after the expert
    /// with `norm_w` clamp `2^-14`. No shared expert (Mixtral-shaped).
    fn llama_moe_rows(
        &self,
        moe: &LlamaMoe,
        n_tokens: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, LlamaError> {
        let n_embd = moe.down_exps.n_rows;
        if n_embd == 0 || !x.len().is_multiple_of(n_embd) {
            return Err(LlamaError::Shape("llama moe".into()));
        }
        let n_out = n_tokens
            .checked_mul(n_embd)
            .ok_or_else(|| LlamaError::Shape("llama moe".into()))?;
        let mut out = vec![0.0f32; n_out];
        for t in 0..n_tokens {
            let xt = token_row(x, t, n_embd, "llama moe x")?;
            let logits = self.gemv_mat(&moe.gate_inp, xt)?;
            if logits.len() != moe.n_expert {
                return Err(LlamaError::Shape("llama ffn_gate_inp".into()));
            }
            let mut probs = logits;
            softmax(&mut probs);
            let selected = topk_logits(&probs, moe.n_expert_used)?;
            let mut wsum = 0.0f32;
            let mut weights = Vec::new();
            for &e in &selected {
                let w = probs.get(e).copied().unwrap_or(0.0);
                weights.push(w);
                wsum += w;
            }
            if wsum < MOE_NORM_W_CLAMP {
                wsum = MOE_NORM_W_CLAMP;
            }
            for w in &mut weights {
                *w /= wsum;
            }
            let off = t.saturating_mul(n_embd);
            let dst = out
                .get_mut(off..off.saturating_add(n_embd))
                .ok_or_else(|| LlamaError::Shape("llama moe out".into()))?;
            for (e, w) in selected.iter().zip(weights.iter()) {
                let ge = expert_view(&moe.gate_exps, *e)?;
                let ue = expert_view(&moe.up_exps, *e)?;
                let de = expert_view(&moe.down_exps, *e)?;
                let g = self.gemv_mat(&ge, xt)?;
                let u = self.gemv_mat(&ue, xt)?;
                let mut hh = silu(&g)?;
                for (a, b) in hh.iter_mut().zip(u.iter()) {
                    *a *= *b;
                }
                let y = self.gemv_mat(&de, &hh)?;
                for (d, v) in dst.iter_mut().zip(y.iter()) {
                    *d += *v * *w;
                }
            }
        }
        Ok(out)
    }

    /// Official llama4.cpp MoE: top-k on raw router logits, sigmoid weights applied
    /// to the FFN input (`weight_before_ffn`), SwiGLU experts, plus shared expert.
    fn llama4_moe_rows(
        &self,
        moe: &Llama4Moe,
        n_tokens: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, LlamaError> {
        let n_embd = moe.down_shexp.n_rows;
        if n_embd == 0 || !x.len().is_multiple_of(n_embd) {
            return Err(LlamaError::Shape("llama4 moe".into()));
        }
        let gate_s = self.gemm_mat(&moe.gate_shexp, n_tokens, x)?;
        let up_s = self.gemm_mat(&moe.up_shexp, n_tokens, x)?;
        let mut h_s = silu(&gate_s)?;
        for (hv, uv) in h_s.iter_mut().zip(up_s.iter()) {
            *hv *= *uv;
        }
        let mut out = self.gemm_mat(&moe.down_shexp, n_tokens, &h_s)?;
        for t in 0..n_tokens {
            let xt = token_row(x, t, n_embd, "llama4 moe x")?;
            let logits = self.gemv_mat(&moe.gate_inp, xt)?;
            if logits.len() != moe.n_expert {
                return Err(LlamaError::Shape("llama4 ffn_gate_inp".into()));
            }
            let selected = topk_logits(&logits, moe.n_expert_used)?;
            let mut routed = vec![0.0f32; n_embd];
            for e in selected {
                let w = sigmoid_f32(logits.get(e).copied().unwrap_or(0.0));
                let xw: Vec<f32> = xt.iter().map(|v| *v * w).collect();
                let ge = expert_view(&moe.gate_exps, e)?;
                let ue = expert_view(&moe.up_exps, e)?;
                let de = expert_view(&moe.down_exps, e)?;
                let g = self.gemv_mat(&ge, &xw)?;
                let u = self.gemv_mat(&ue, &xw)?;
                let mut hh = silu(&g)?;
                for (a, b) in hh.iter_mut().zip(u.iter()) {
                    *a *= *b;
                }
                let y = self.gemv_mat(&de, &hh)?;
                for (o, v) in routed.iter_mut().zip(y.iter()) {
                    *o += *v;
                }
            }
            let off = t.saturating_mul(n_embd);
            let dst = out
                .get_mut(off..off.saturating_add(n_embd))
                .ok_or_else(|| LlamaError::Shape("llama4 moe out".into()))?;
            for (d, s) in dst.iter_mut().zip(routed.iter()) {
                *d += *s;
            }
        }
        Ok(out)
    }

    /// Official qwen2moe.cpp: softmax then top-k, weights after SwiGLU (`norm_w=false`),
    /// plus shared expert gated by `silu(x)/x` on `ffn_gate_inp_shexp`.
    fn qwen2moe_rows(
        &self,
        moe: &Qwen2Moe,
        n_tokens: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, LlamaError> {
        let n_embd = moe.down_shexp.n_rows;
        if n_embd == 0 || !x.len().is_multiple_of(n_embd) {
            return Err(LlamaError::Shape("qwen2moe".into()));
        }
        let gate_s = self.gemm_mat(&moe.gate_shexp, n_tokens, x)?;
        let up_s = self.gemm_mat(&moe.up_shexp, n_tokens, x)?;
        let mut h_s = silu(&gate_s)?;
        for (hv, uv) in h_s.iter_mut().zip(up_s.iter()) {
            *hv *= *uv;
        }
        let mut shexp = self.gemm_mat(&moe.down_shexp, n_tokens, &h_s)?;
        let shexp_gate = self.gemm_mat(&moe.gate_inp_shexp, n_tokens, x)?;
        if shexp_gate.len() != n_tokens {
            return Err(LlamaError::Shape("qwen2moe ffn_gate_inp_shexp".into()));
        }
        for t in 0..n_tokens {
            let w = sigmoid_f32(shexp_gate.get(t).copied().unwrap_or(0.0));
            let off = t.saturating_mul(n_embd);
            let row = shexp
                .get_mut(off..off.saturating_add(n_embd))
                .ok_or_else(|| LlamaError::Shape("qwen2moe shexp".into()))?;
            for v in row.iter_mut() {
                *v *= w;
            }
        }
        let n_out = n_tokens
            .checked_mul(n_embd)
            .ok_or_else(|| LlamaError::Shape("qwen2moe".into()))?;
        let mut out = vec![0.0f32; n_out];
        for t in 0..n_tokens {
            let xt = token_row(x, t, n_embd, "qwen2moe x")?;
            let logits = self.gemv_mat(&moe.gate_inp, xt)?;
            if logits.len() != moe.n_expert {
                return Err(LlamaError::Shape("qwen2moe ffn_gate_inp".into()));
            }
            let mut probs = logits;
            softmax(&mut probs);
            let selected = topk_logits(&probs, moe.n_expert_used)?;
            let off = t.saturating_mul(n_embd);
            let dst = out
                .get_mut(off..off.saturating_add(n_embd))
                .ok_or_else(|| LlamaError::Shape("qwen2moe out".into()))?;
            for e in selected {
                let w = probs.get(e).copied().unwrap_or(0.0);
                let ge = expert_view(&moe.gate_exps, e)?;
                let ue = expert_view(&moe.up_exps, e)?;
                let de = expert_view(&moe.down_exps, e)?;
                let g = self.gemv_mat(&ge, xt)?;
                let u = self.gemv_mat(&ue, xt)?;
                let mut hh = silu(&g)?;
                for (a, b) in hh.iter_mut().zip(u.iter()) {
                    *a *= *b;
                }
                let y = self.gemv_mat(&de, &hh)?;
                for (d, v) in dst.iter_mut().zip(y.iter()) {
                    *d += *v * w;
                }
            }
            let srow = shexp
                .get(off..off.saturating_add(n_embd))
                .ok_or_else(|| LlamaError::Shape("qwen2moe shexp add".into()))?;
            for (d, s) in dst.iter_mut().zip(srow.iter()) {
                *d += *s;
            }
        }
        Ok(out)
    }

    /// Official qwen3moe.cpp: softmax then top-k, weights after SwiGLU (`norm_w`
    /// clamp `2^-14`). No shared expert. QK-Norm is applied on Q/K before RoPE.
    fn qwen3moe_rows(
        &self,
        moe: &Qwen3Moe,
        n_tokens: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, LlamaError> {
        let n_embd = moe.down_exps.n_rows;
        if n_embd == 0 || !x.len().is_multiple_of(n_embd) {
            return Err(LlamaError::Shape("qwen3moe".into()));
        }
        let n_out = n_tokens
            .checked_mul(n_embd)
            .ok_or_else(|| LlamaError::Shape("qwen3moe".into()))?;
        let mut out = vec![0.0f32; n_out];
        for t in 0..n_tokens {
            let xt = token_row(x, t, n_embd, "qwen3moe x")?;
            let logits = self.gemv_mat(&moe.gate_inp, xt)?;
            if logits.len() != moe.n_expert {
                return Err(LlamaError::Shape("qwen3moe ffn_gate_inp".into()));
            }
            let mut probs = logits;
            softmax(&mut probs);
            let selected = topk_logits(&probs, moe.n_expert_used)?;
            let mut wsum = 0.0f32;
            let mut weights = Vec::new();
            for &e in &selected {
                let w = probs.get(e).copied().unwrap_or(0.0);
                weights.push(w);
                wsum += w;
            }
            if wsum < MOE_NORM_W_CLAMP {
                wsum = MOE_NORM_W_CLAMP;
            }
            for w in &mut weights {
                *w /= wsum;
            }
            let off = t.saturating_mul(n_embd);
            let dst = out
                .get_mut(off..off.saturating_add(n_embd))
                .ok_or_else(|| LlamaError::Shape("qwen3moe out".into()))?;
            for (e, w) in selected.iter().zip(weights.iter()) {
                let ge = expert_view(&moe.gate_exps, *e)?;
                let ue = expert_view(&moe.up_exps, *e)?;
                let de = expert_view(&moe.down_exps, *e)?;
                let g = self.gemv_mat(&ge, xt)?;
                let u = self.gemv_mat(&ue, xt)?;
                let mut hh = silu(&g)?;
                for (a, b) in hh.iter_mut().zip(u.iter()) {
                    *a *= *b;
                }
                let y = self.gemv_mat(&de, &hh)?;
                for (d, v) in dst.iter_mut().zip(y.iter()) {
                    *d += *v * *w;
                }
            }
        }
        Ok(out)
    }

    /// Official qwen3next.cpp: softmax then top-k, weights after SwiGLU (`norm_w`
    /// clamp `2^-14`), plus shared expert * sigmoid(`ffn_gate_inp_shexp`).
    fn qwen3next_rows(
        &self,
        moe: &Qwen2Moe,
        n_tokens: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, LlamaError> {
        let n_embd = moe.down_shexp.n_rows;
        if n_embd == 0 || !x.len().is_multiple_of(n_embd) {
            return Err(LlamaError::Shape("qwen3next".into()));
        }
        let gate_s = self.gemm_mat(&moe.gate_shexp, n_tokens, x)?;
        let up_s = self.gemm_mat(&moe.up_shexp, n_tokens, x)?;
        let mut h_s = silu(&gate_s)?;
        for (hv, uv) in h_s.iter_mut().zip(up_s.iter()) {
            *hv *= *uv;
        }
        let mut shexp = self.gemm_mat(&moe.down_shexp, n_tokens, &h_s)?;
        let shexp_gate = self.gemm_mat(&moe.gate_inp_shexp, n_tokens, x)?;
        if shexp_gate.len() != n_tokens {
            return Err(LlamaError::Shape("qwen3next ffn_gate_inp_shexp".into()));
        }
        for t in 0..n_tokens {
            let w = sigmoid_f32(shexp_gate.get(t).copied().unwrap_or(0.0));
            let off = t.saturating_mul(n_embd);
            let row = shexp
                .get_mut(off..off.saturating_add(n_embd))
                .ok_or_else(|| LlamaError::Shape("qwen3next shexp".into()))?;
            for v in row.iter_mut() {
                *v *= w;
            }
        }
        let n_out = n_tokens
            .checked_mul(n_embd)
            .ok_or_else(|| LlamaError::Shape("qwen3next".into()))?;
        let mut out = vec![0.0f32; n_out];
        for t in 0..n_tokens {
            let xt = token_row(x, t, n_embd, "qwen3next x")?;
            let logits = self.gemv_mat(&moe.gate_inp, xt)?;
            if logits.len() != moe.n_expert {
                return Err(LlamaError::Shape("qwen3next ffn_gate_inp".into()));
            }
            let mut probs = logits;
            softmax(&mut probs);
            let selected = topk_logits(&probs, moe.n_expert_used)?;
            let mut wsum = 0.0f32;
            let mut weights = Vec::new();
            for &e in &selected {
                let w = probs.get(e).copied().unwrap_or(0.0);
                weights.push(w);
                wsum += w;
            }
            if wsum < MOE_NORM_W_CLAMP {
                wsum = MOE_NORM_W_CLAMP;
            }
            for w in &mut weights {
                *w /= wsum;
            }
            let off = t.saturating_mul(n_embd);
            let dst = out
                .get_mut(off..off.saturating_add(n_embd))
                .ok_or_else(|| LlamaError::Shape("qwen3next out".into()))?;
            for (e, w) in selected.iter().zip(weights.iter()) {
                let ge = expert_view(&moe.gate_exps, *e)?;
                let ue = expert_view(&moe.up_exps, *e)?;
                let de = expert_view(&moe.down_exps, *e)?;
                let g = self.gemv_mat(&ge, xt)?;
                let u = self.gemv_mat(&ue, xt)?;
                let mut hh = silu(&g)?;
                for (a, b) in hh.iter_mut().zip(u.iter()) {
                    *a *= *b;
                }
                let y = self.gemv_mat(&de, &hh)?;
                for (d, v) in dst.iter_mut().zip(y.iter()) {
                    *d += *v * *w;
                }
            }
            let srow = shexp
                .get(off..off.saturating_add(n_embd))
                .ok_or_else(|| LlamaError::Shape("qwen3next shexp add".into()))?;
            for (d, s) in dst.iter_mut().zip(srow.iter()) {
                *d += *s;
            }
        }
        Ok(out)
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

/// Official Qwen3Next: split joint Q+gate (`n_embd_head * 2` per head) into query and gate.
fn split_qwen3next_q_gate(
    q_full: &[f32],
    n_tokens: usize,
    n_head: usize,
    hd: usize,
) -> Result<(Vec<f32>, Option<Vec<f32>>), LlamaError> {
    let q_width = n_head
        .checked_mul(hd)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| LlamaError::Shape("qwen3next q gate".into()))?;
    let q_out_w = n_head
        .checked_mul(hd)
        .ok_or_else(|| LlamaError::Shape("qwen3next q gate".into()))?;
    if hd == 0
        || n_head == 0
        || !q_full.len().is_multiple_of(q_width)
        || q_full.len() / q_width != n_tokens
    {
        return Err(LlamaError::Shape("qwen3next q gate".into()));
    }
    let n_out = n_tokens
        .checked_mul(q_out_w)
        .ok_or_else(|| LlamaError::Shape("qwen3next q gate".into()))?;
    let mut q = vec![0.0f32; n_out];
    let mut gate = vec![0.0f32; n_out];
    for t in 0..n_tokens {
        for h in 0..n_head {
            let src = t
                .saturating_mul(q_width)
                .saturating_add(h.saturating_mul(hd.saturating_mul(2)));
            let dst = t
                .saturating_mul(q_out_w)
                .saturating_add(h.saturating_mul(hd));
            let q_src = q_full
                .get(src..src.saturating_add(hd))
                .ok_or_else(|| LlamaError::Shape("qwen3next q gate".into()))?;
            let g_src = q_full
                .get(src.saturating_add(hd)..src.saturating_add(hd.saturating_mul(2)))
                .ok_or_else(|| LlamaError::Shape("qwen3next q gate".into()))?;
            let q_dst = q
                .get_mut(dst..dst.saturating_add(hd))
                .ok_or_else(|| LlamaError::Shape("qwen3next q gate".into()))?;
            let g_dst = gate
                .get_mut(dst..dst.saturating_add(hd))
                .ok_or_else(|| LlamaError::Shape("qwen3next q gate".into()))?;
            for (d, s) in q_dst.iter_mut().zip(q_src.iter()) {
                *d = *s;
            }
            for (d, s) in g_dst.iter_mut().zip(g_src.iter()) {
                *d = *s;
            }
        }
    }
    Ok((q, Some(gate)))
}

/// Official Qwen3 / Qwen3MoE QK-Norm: per-head `LLM_NORM_RMS` on Q or K.
fn qk_norm_rows(
    x: Vec<f32>,
    head_dim: usize,
    w: Option<&[f32]>,
    eps: f32,
) -> Result<Vec<f32>, LlamaError> {
    match w {
        Some(w) => rmsnorm_rows(&x, head_dim, w, eps),
        None => Ok(x),
    }
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
    score_scale: f32,
) -> Result<Vec<f32>, LlamaError> {
    if n_head_kv == 0 {
        return Err(LlamaError::Shape("gqa".into()));
    }
    let n_embd = qh.len().saturating_mul(hd);
    let mut attn = vec![0.0f32; n_embd];
    let inv = score_scale;
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

/// Official Llama4 `ggml_rms_norm` without a weight (`Llama4TextL2Norm`).
fn rmsnorm_unweighted(x: &[f32], eps: f32) -> Result<Vec<f32>, LlamaError> {
    let mut ss = 0.0f32;
    for v in x {
        ss += *v * *v;
    }
    let n = f32::from(u16::try_from(x.len()).unwrap_or(1));
    let rms = (ss / n + eps).sqrt();
    let inv = if rms > 0.0 { 1.0 / rms } else { 0.0 };
    let mut out = vec![0.0f32; x.len()];
    for (o, xv) in out.iter_mut().zip(x.iter()) {
        *o = *xv * inv;
    }
    Ok(out)
}

/// Official llama.cpp `llm_graph_input_attn_temp` for Llama4 NoPE layers.
fn llama4_attn_temp_scale(pos: usize) -> f32 {
    let pos_f = f32::from(u16::try_from(pos).unwrap_or(u16::MAX));
    let floor = ((pos_f + LLAMA4_ATTN_TEMP_OFFSET) / LLAMA4_ATTN_TEMP_FLOOR).floor();
    (floor + 1.0).ln() * LLAMA4_ATTN_TEMP_SCALE + 1.0
}

fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// ggml `ggml_argsort_top_k` descending; ties keep the lower index.
fn topk_logits(logits: &[f32], k: usize) -> Result<Vec<usize>, LlamaError> {
    if k == 0 || k > logits.len() {
        return Err(LlamaError::Shape("expert_used_count".into()));
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| {
        let va = logits.get(a).copied().unwrap_or(f32::NEG_INFINITY);
        let vb = logits.get(b).copied().unwrap_or(f32::NEG_INFINITY);
        match vb.partial_cmp(&va) {
            Some(core::cmp::Ordering::Equal) | None => a.cmp(&b),
            Some(ord) => ord,
        }
    });
    idx.truncate(k);
    Ok(idx)
}

fn expert_view(m: &QuantMat, e: usize) -> Result<QuantMat, LlamaError> {
    if m.n_parts == 0 || e >= m.n_parts {
        return Err(LlamaError::Shape(m.name.clone()));
    }
    let total = m.end.saturating_sub(m.start);
    let part = total / m.n_parts;
    if part.saturating_mul(m.n_parts) != total {
        return Err(LlamaError::Shape(m.name.clone()));
    }
    let start = m.start.saturating_add(e.saturating_mul(part));
    Ok(QuantMat {
        name: m.name.clone(),
        ty: m.ty,
        n_cols: m.n_cols,
        n_rows: m.n_rows,
        n_parts: 1,
        start,
        end: start.saturating_add(part),
    })
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

/// Official `ggml_compute_forward_norm_f32` + `build_norm` `LLM_NORM` (weight, optional bias).
fn layernorm(
    x: &[f32],
    w: &[f32],
    b: Option<&[f32]>,
    eps: f32,
) -> Result<Vec<f32>, LlamaError> {
    if x.len() != w.len() {
        return Err(LlamaError::Shape("layernorm".into()));
    }
    if let Some(b) = b {
        if b.len() != x.len() {
            return Err(LlamaError::Shape("layernorm".into()));
        }
    }
    let n = f32::from(u16::try_from(x.len()).unwrap_or(1));
    let mut sum = 0.0f32;
    for v in x {
        sum += *v;
    }
    let mean = if n > 0.0 { sum / n } else { 0.0 };
    let mut var = 0.0f32;
    for v in x {
        let d = *v - mean;
        var += d * d;
    }
    if n > 0.0 {
        var /= n;
    }
    let scale = 1.0 / (var + eps).sqrt();
    let mut out = vec![0.0f32; x.len()];
    for (i, ((o, xv), wv)) in out.iter_mut().zip(x.iter()).zip(w.iter()).enumerate() {
        let mut y = (*xv - mean) * scale * *wv;
        if let Some(b) = b {
            if let Some(bv) = b.get(i) {
                y += *bv;
            }
        }
        *o = y;
    }
    Ok(out)
}

fn layernorm_rows(
    x: &[f32],
    width: usize,
    w: &[f32],
    b: Option<&[f32]>,
    eps: f32,
) -> Result<Vec<f32>, LlamaError> {
    if width == 0 || !x.len().is_multiple_of(width) {
        return Err(LlamaError::Shape("layernorm".into()));
    }
    let mut out = Vec::new();
    for row in x.chunks(width) {
        out.extend(layernorm(row, w, b, eps)?);
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

/// Official Qwen2VL / Qwen3VL text walk: `ggml_rope_multi` when sections are present.
/// Official phi2 uses `LLAMA_ROPE_TYPE_NEOX` (`llama_model_rope_type`).
fn apply_rope(
    vec: &mut [f32],
    pos: usize,
    n_rot: usize,
    base: f32,
    sections: Option<[i32; 4]>,
    is_imrope: bool,
    neox: bool,
) -> Result<(), LlamaError> {
    if let Some(sections) = sections {
        // Official `llama-graph.cpp` text tokens: `[t,h,w,e] = [p,p,p,0]`.
        // Official `n_pos_per_embd()` is 4 for both MROPE and IMROPE.
        rope_multi(vec, [pos, pos, pos, 0], n_rot, base, sections, is_imrope)
    } else if neox {
        rope_neox(vec, pos, n_rot, base)
    } else {
        rope(vec, pos, n_rot, base)
    }
}

/// Official `ggml_compute_forward_rope` `LLAMA_ROPE_TYPE_NEOX`: pairs offset by `n_rot/2`.
fn rope_neox(vec: &mut [f32], pos: usize, n_rot: usize, base: f32) -> Result<(), LlamaError> {
    let n = n_rot.min(vec.len());
    if n < 2 {
        return Ok(());
    }
    if !n.is_multiple_of(2) {
        return Err(LlamaError::Shape("rope.dimension_count".into()));
    }
    let n_offset = n / 2;
    let n_rot_f = f32::from(u16::try_from(n_rot).unwrap_or(2));
    let mut theta = f32::from(u16::try_from(pos).unwrap_or(u16::MAX));
    let theta_scale = base.powf(-2.0 / n_rot_f);
    let mut i = 0usize;
    while i < n_offset {
        let cos = theta.cos();
        let sin = theta.sin();
        let x0 = *vec.get(i).unwrap_or(&0.0);
        let x1 = *vec.get(i.saturating_add(n_offset)).unwrap_or(&0.0);
        if let Some(slot) = vec.get_mut(i) {
            *slot = x0 * cos - x1 * sin;
        }
        if let Some(slot) = vec.get_mut(i.saturating_add(n_offset)) {
            *slot = x0 * sin + x1 * cos;
        }
        theta *= theta_scale;
        i = i.saturating_add(1);
    }
    Ok(())
}

/// Official `ggml_compute_forward_rope_flt` `ggml_mrope_cache_init` + NEOX
/// `rotate_pairs`. `GGML_ROPE_TYPE_MROPE` when `is_imrope` is false (qwen2vl).
/// `GGML_ROPE_TYPE_IMROPE` when `is_imrope` is true (qwen3vl interleaved).
/// Not VISION (`indep_sects`). `pos` is `[t, h, w, e]` (`n_pos_per_embd=4`).
fn rope_multi(
    vec: &mut [f32],
    pos: [usize; 4],
    n_rot: usize,
    base: f32,
    sections: [i32; 4],
    is_imrope: bool,
) -> Result<(), LlamaError> {
    let n = n_rot.min(vec.len());
    if n < 2 {
        return Ok(());
    }
    if !n.is_multiple_of(2) {
        return Err(LlamaError::Shape("rope.dimension_count".into()));
    }
    let pos_t = *pos
        .first()
        .ok_or_else(|| LlamaError::Shape("n_pos_per_embd".into()))?;
    let pos_h = *pos
        .get(1)
        .ok_or_else(|| LlamaError::Shape("n_pos_per_embd".into()))?;
    let pos_w = *pos
        .get(2)
        .ok_or_else(|| LlamaError::Shape("n_pos_per_embd".into()))?;
    let pos_e = *pos
        .get(3)
        .ok_or_else(|| LlamaError::Shape("n_pos_per_embd".into()))?;
    let s0 = *sections
        .first()
        .ok_or_else(|| LlamaError::Shape("rope.dimension_sections".into()))?;
    let s1 = *sections
        .get(1)
        .ok_or_else(|| LlamaError::Shape("rope.dimension_sections".into()))?;
    let s2 = *sections
        .get(2)
        .ok_or_else(|| LlamaError::Shape("rope.dimension_sections".into()))?;
    let s3 = *sections
        .get(3)
        .ok_or_else(|| LlamaError::Shape("rope.dimension_sections".into()))?;
    let sect_dims = s0
        .checked_add(s1)
        .and_then(|v| v.checked_add(s2))
        .and_then(|v| v.checked_add(s3))
        .ok_or_else(|| LlamaError::Shape("rope.dimension_sections".into()))?;
    if sect_dims <= 0 {
        return Err(LlamaError::Shape("rope.dimension_sections".into()));
    }
    let sect_dims_us = usize::try_from(sect_dims)
        .map_err(|_| LlamaError::Shape("rope.dimension_sections".into()))?;
    if sect_dims_us > n {
        return Err(LlamaError::Shape("rope.dimension_sections".into()));
    }
    let sec_w = s0
        .checked_add(s1)
        .ok_or_else(|| LlamaError::Shape("rope.dimension_sections".into()))?;
    let n_offset = n / 2;
    let n_rot_f = f32::from(u16::try_from(n_rot).unwrap_or(2));
    let theta_scale = base.powf(-2.0 / n_rot_f);
    let mut theta_t = f32::from(u16::try_from(pos_t).unwrap_or(u16::MAX));
    let mut theta_h = f32::from(u16::try_from(pos_h).unwrap_or(u16::MAX));
    let mut theta_w = f32::from(u16::try_from(pos_w).unwrap_or(u16::MAX));
    let mut theta_e = f32::from(u16::try_from(pos_e).unwrap_or(u16::MAX));
    let mut i0 = 0usize;
    while i0 + 1 < n {
        let ic = i0 / 2;
        let sector = ic % sect_dims_us;
        let sector_i = i32::try_from(sector)
            .map_err(|_| LlamaError::Shape("rope.dimension_sections".into()))?;
        let theta = if is_imrope {
            // Official `ggml_mrope_cache_init` `is_imrope` (qwen3vl).
            let bound_h = s1
                .checked_mul(3)
                .ok_or_else(|| LlamaError::Shape("rope.dimension_sections".into()))?;
            let bound_w = s2
                .checked_mul(3)
                .ok_or_else(|| LlamaError::Shape("rope.dimension_sections".into()))?;
            let bound_t = s0
                .checked_mul(3)
                .ok_or_else(|| LlamaError::Shape("rope.dimension_sections".into()))?;
            if sector_i % 3 == 1 && sector_i < bound_h {
                theta_h
            } else if sector_i % 3 == 2 && sector_i < bound_w {
                theta_w
            } else if sector_i % 3 == 0 && sector_i < bound_t {
                theta_t
            } else {
                theta_e
            }
        } else if sector_i >= s0 && sector_i < sec_w {
            theta_h
        } else if sector_i >= sec_w && sector_i < sec_w.saturating_add(s2) {
            theta_w
        } else if sector_i >= sec_w.saturating_add(s2) {
            theta_e
        } else {
            theta_t
        };
        let cos = theta.cos();
        let sin = theta.sin();
        let x0 = *vec.get(ic).unwrap_or(&0.0);
        let x1 = *vec.get(ic.saturating_add(n_offset)).unwrap_or(&0.0);
        if let Some(slot) = vec.get_mut(ic) {
            *slot = x0 * cos - x1 * sin;
        }
        if let Some(slot) = vec.get_mut(ic.saturating_add(n_offset)) {
            *slot = x0 * sin + x1 * cos;
        }
        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
        theta_e *= theta_scale;
        i0 += 2;
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

    /// Independent scalar of official `ggml_compute_forward_norm_f32` + `LLM_NORM`.
    fn oracle_layernorm(x: &[f32], w: &[f32], b: Option<&[f32]>, eps: f32) -> Vec<f32> {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let scale = 1.0 / (var + eps).sqrt();
        x.iter()
            .zip(w.iter())
            .enumerate()
            .map(|(i, (xv, wv))| {
                let mut y = (xv - mean) * scale * wv;
                if let Some(b) = b {
                    y += b[i];
                }
                y
            })
            .collect()
    }

    /// Official Llama4 `ggml_rms_norm` without weight (`Llama4TextL2Norm`).
    fn oracle_rmsnorm_unweighted(x: &[f32], eps: f32) -> Vec<f32> {
        let ss: f32 = x.iter().map(|v| v * v).sum();
        let rms = (ss / x.len() as f32 + eps).sqrt();
        x.iter().map(|a| a / rms).collect()
    }

    fn oracle_gemv_expert(t: Tensor<'_>, expert: usize, x: &[f32]) -> Vec<f32> {
        let n_parts = t
            .shape
            .get(2)
            .copied()
            .and_then(|d| usize::try_from(d).ok())
            .unwrap_or(1);
        assert!(expert < n_parts, "expert {expert} >= {n_parts}");
        let part = t.data.len() / n_parts;
        let start = expert * part;
        let bytes = crate::write_gguf(&[TensorWrite {
            name: "e".into(),
            ty: t.ty,
            shape: vec![
                u64::try_from(t.n_cols()).unwrap_or(0),
                u64::try_from(t.n_rows()).unwrap_or(0),
            ],
            data: t.data[start..start + part].to_vec(),
        }]);
        let g = load_gguf(&bytes).expect("oracle expert slice");
        oracle_gemv(g.tensor("e").expect("e"), x)
    }

    /// Official `build_moe_ffn` on one token: softmax, then top-k; SwiGLU;
    /// weights after the expert with `norm_w` clamp `2^-14`. Used by official
    /// llama MoE and official qwen3moe.cpp (`norm_w=true`).
    fn oracle_softmax_norm_w_moe(g: &Gguf, arch: &str, xn: &[f32]) -> Vec<f32> {
        let n_expert = arch_u32(g, arch, "expert_count").unwrap() as usize;
        let n_used = arch_u32(g, arch, "expert_used_count").unwrap() as usize;
        let logits = oracle_gemv(g.tensor("blk.0.ffn_gate_inp.weight").unwrap(), xn);
        assert_eq!(logits.len(), n_expert);
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|v| (v - m).exp()).collect();
        let z: f32 = probs.iter().sum();
        if z > 0.0 {
            for p in &mut probs {
                *p /= z;
            }
        }
        let mut idx: Vec<usize> = (0..n_expert).collect();
        idx.sort_by(|&a, &b| match probs[b].partial_cmp(&probs[a]) {
            Some(core::cmp::Ordering::Equal) | None => a.cmp(&b),
            Some(ord) => ord,
        });
        idx.truncate(n_used);
        let mut wsum = 0.0f32;
        let mut weights = Vec::new();
        for &e in &idx {
            let w = probs[e];
            weights.push(w);
            wsum += w;
        }
        if wsum < MOE_NORM_W_CLAMP {
            wsum = MOE_NORM_W_CLAMP;
        }
        for w in &mut weights {
            *w /= wsum;
        }
        let gate_exps = g.tensor("blk.0.ffn_gate_exps.weight").unwrap();
        let up_exps = g.tensor("blk.0.ffn_up_exps.weight").unwrap();
        let down_exps = g.tensor("blk.0.ffn_down_exps.weight").unwrap();
        let mut routed = vec![0.0f32; xn.len()];
        for (e, w) in idx.iter().zip(weights.iter()) {
            let gate = oracle_gemv_expert(gate_exps, *e, xn);
            let up = oracle_gemv_expert(up_exps, *e, xn);
            let h: Vec<f32> = gate
                .iter()
                .zip(up.iter())
                .map(|(gv, u)| (gv / (1.0 + (-gv).exp())) * u)
                .collect();
            let y = oracle_gemv_expert(down_exps, *e, &h);
            for (o, v) in routed.iter_mut().zip(y.iter()) {
                *o += *v * *w;
            }
        }
        routed
    }

    /// Official llama.cpp `build_moe_ffn` on one token: softmax, then top-k;
    /// SwiGLU; weights after the expert with `norm_w` clamp `2^-14`.
    fn oracle_llama_moe(g: &Gguf, xn: &[f32]) -> Vec<f32> {
        oracle_softmax_norm_w_moe(g, "llama", xn)
    }

    /// Official qwen3moe.cpp on one token: same `build_moe_ffn` as llama MoE
    /// (`SOFTMAX` + `norm_w`). QK-Norm is applied on Q/K in `oracle_forward_seq`.
    fn oracle_qwen3moe(g: &Gguf, xn: &[f32]) -> Vec<f32> {
        oracle_softmax_norm_w_moe(g, "qwen3moe", xn)
    }

    /// Official qwen3next.cpp on one token: softmax then top-k with `norm_w`,
    /// plus shared expert * sigmoid(`ffn_gate_inp_shexp`).
    fn oracle_qwen3next(g: &Gguf, xn: &[f32]) -> Vec<f32> {
        let mut routed = oracle_softmax_norm_w_moe(g, "qwen3next", xn);
        let gate_s = oracle_gemv(g.tensor("blk.0.ffn_gate_shexp.weight").unwrap(), xn);
        let up_s = oracle_gemv(g.tensor("blk.0.ffn_up_shexp.weight").unwrap(), xn);
        let h_s: Vec<f32> = gate_s
            .iter()
            .zip(up_s.iter())
            .map(|(gv, u)| (gv / (1.0 + (-gv).exp())) * u)
            .collect();
        let mut shexp = oracle_gemv(g.tensor("blk.0.ffn_down_shexp.weight").unwrap(), &h_s);
        let gate_inp_s = oracle_gemv(g.tensor("blk.0.ffn_gate_inp_shexp.weight").unwrap(), xn);
        assert_eq!(gate_inp_s.len(), 1);
        let sw = 1.0 / (1.0 + (-gate_inp_s[0]).exp());
        for v in &mut shexp {
            *v *= sw;
        }
        for (o, s) in routed.iter_mut().zip(shexp.iter()) {
            *o += *s;
        }
        routed
    }

    /// Official llama4.cpp MoE on one token: top-k raw logits, sigmoid weight
    /// before SwiGLU, plus shared expert.
    fn oracle_llama4_moe(g: &Gguf, xn: &[f32]) -> Vec<f32> {
        let n_expert = arch_u32(g, "llama4", "expert_count").unwrap() as usize;
        let n_used = arch_u32(g, "llama4", "expert_used_count").unwrap() as usize;
        let logits = oracle_gemv(g.tensor("blk.0.ffn_gate_inp.weight").unwrap(), xn);
        assert_eq!(logits.len(), n_expert);
        let mut idx: Vec<usize> = (0..n_expert).collect();
        idx.sort_by(|&a, &b| match logits[b].partial_cmp(&logits[a]) {
            Some(core::cmp::Ordering::Equal) | None => a.cmp(&b),
            Some(ord) => ord,
        });
        idx.truncate(n_used);
        let gate_exps = g.tensor("blk.0.ffn_gate_exps.weight").unwrap();
        let up_exps = g.tensor("blk.0.ffn_up_exps.weight").unwrap();
        let down_exps = g.tensor("blk.0.ffn_down_exps.weight").unwrap();
        let mut routed = vec![0.0f32; xn.len()];
        for e in idx {
            let w = 1.0 / (1.0 + (-logits[e]).exp());
            let xw: Vec<f32> = xn.iter().map(|v| v * w).collect();
            let gate = oracle_gemv_expert(gate_exps, e, &xw);
            let up = oracle_gemv_expert(up_exps, e, &xw);
            let h: Vec<f32> = gate
                .iter()
                .zip(up.iter())
                .map(|(gv, u)| (gv / (1.0 + (-gv).exp())) * u)
                .collect();
            let y = oracle_gemv_expert(down_exps, e, &h);
            for (o, v) in routed.iter_mut().zip(y.iter()) {
                *o += *v;
            }
        }
        let gate_s = oracle_gemv(g.tensor("blk.0.ffn_gate_shexp.weight").unwrap(), xn);
        let up_s = oracle_gemv(g.tensor("blk.0.ffn_up_shexp.weight").unwrap(), xn);
        let h_s: Vec<f32> = gate_s
            .iter()
            .zip(up_s.iter())
            .map(|(gv, u)| (gv / (1.0 + (-gv).exp())) * u)
            .collect();
        let shexp = oracle_gemv(g.tensor("blk.0.ffn_down_shexp.weight").unwrap(), &h_s);
        routed
            .iter()
            .zip(shexp.iter())
            .map(|(a, b)| a + b)
            .collect()
    }

    /// Official qwen2moe.cpp on one token: softmax then top-k, weights after
    /// SwiGLU without `norm_w`, plus shared expert * sigmoid(`ffn_gate_inp_shexp`).
    fn oracle_qwen2moe(g: &Gguf, xn: &[f32]) -> Vec<f32> {
        let n_expert = arch_u32(g, "qwen2moe", "expert_count").unwrap() as usize;
        let n_used = arch_u32(g, "qwen2moe", "expert_used_count").unwrap() as usize;
        let logits = oracle_gemv(g.tensor("blk.0.ffn_gate_inp.weight").unwrap(), xn);
        assert_eq!(logits.len(), n_expert);
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|v| (v - m).exp()).collect();
        let z: f32 = probs.iter().sum();
        if z > 0.0 {
            for p in &mut probs {
                *p /= z;
            }
        }
        let mut idx: Vec<usize> = (0..n_expert).collect();
        idx.sort_by(|&a, &b| match probs[b].partial_cmp(&probs[a]) {
            Some(core::cmp::Ordering::Equal) | None => a.cmp(&b),
            Some(ord) => ord,
        });
        idx.truncate(n_used);
        let gate_exps = g.tensor("blk.0.ffn_gate_exps.weight").unwrap();
        let up_exps = g.tensor("blk.0.ffn_up_exps.weight").unwrap();
        let down_exps = g.tensor("blk.0.ffn_down_exps.weight").unwrap();
        let mut routed = vec![0.0f32; xn.len()];
        for e in idx {
            let w = probs[e];
            let gate = oracle_gemv_expert(gate_exps, e, xn);
            let up = oracle_gemv_expert(up_exps, e, xn);
            let h: Vec<f32> = gate
                .iter()
                .zip(up.iter())
                .map(|(gv, u)| (gv / (1.0 + (-gv).exp())) * u)
                .collect();
            let y = oracle_gemv_expert(down_exps, e, &h);
            for (o, v) in routed.iter_mut().zip(y.iter()) {
                *o += *v * w;
            }
        }
        let gate_s = oracle_gemv(g.tensor("blk.0.ffn_gate_shexp.weight").unwrap(), xn);
        let up_s = oracle_gemv(g.tensor("blk.0.ffn_up_shexp.weight").unwrap(), xn);
        let h_s: Vec<f32> = gate_s
            .iter()
            .zip(up_s.iter())
            .map(|(gv, u)| (gv / (1.0 + (-gv).exp())) * u)
            .collect();
        let mut shexp = oracle_gemv(g.tensor("blk.0.ffn_down_shexp.weight").unwrap(), &h_s);
        let gate_inp_s = oracle_gemv(g.tensor("blk.0.ffn_gate_inp_shexp.weight").unwrap(), xn);
        assert_eq!(gate_inp_s.len(), 1);
        let sw = 1.0 / (1.0 + (-gate_inp_s[0]).exp());
        for v in &mut shexp {
            *v *= sw;
        }
        routed
            .iter()
            .zip(shexp.iter())
            .map(|(a, b)| a + b)
            .collect()
    }

    fn oracle_rope_neox(mut v: Vec<f32>, pos: usize, n_rot: usize, base: f32) -> Vec<f32> {
        let n = n_rot.min(v.len());
        if n < 2 || !n.is_multiple_of(2) {
            return v;
        }
        let n_offset = n / 2;
        let mut theta = pos as f32;
        let theta_scale = base.powf(-2.0 / n_rot as f32);
        for i in 0..n_offset {
            let (c, s) = (theta.cos(), theta.sin());
            let x0 = v[i];
            let x1 = v[i + n_offset];
            v[i] = x0 * c - x1 * s;
            v[i + n_offset] = x0 * s + x1 * c;
            theta *= theta_scale;
        }
        v
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

    /// Independent scalar of official `ggml_rope_multi` /
    /// `GGML_ROPE_TYPE_MROPE` or `GGML_ROPE_TYPE_IMROPE`.
    fn oracle_rope_multi(
        mut v: Vec<f32>,
        pos: [usize; 4],
        n_rot: usize,
        base: f32,
        sections: [i32; 4],
        is_imrope: bool,
    ) -> Vec<f32> {
        let n = n_rot.min(v.len());
        if n < 2 || !n.is_multiple_of(2) {
            return v;
        }
        let s0 = sections[0];
        let s1 = sections[1];
        let s2 = sections[2];
        let s3 = sections[3];
        let sect_dims = s0 + s1 + s2 + s3;
        if sect_dims <= 0 {
            return v;
        }
        let Ok(sect_dims_us) = usize::try_from(sect_dims) else {
            return v;
        };
        let sec_w = s0 + s1;
        let n_offset = n / 2;
        let theta_scale = base.powf(-2.0 / n_rot as f32);
        let mut theta_t = pos[0] as f32;
        let mut theta_h = pos[1] as f32;
        let mut theta_w = pos[2] as f32;
        let mut theta_e = pos[3] as f32;
        let mut i0 = 0usize;
        while i0 + 1 < n {
            let ic = i0 / 2;
            let Ok(sector) = i32::try_from(ic % sect_dims_us) else {
                return v;
            };
            let theta = if is_imrope {
                if sector % 3 == 1 && sector < 3 * s1 {
                    theta_h
                } else if sector % 3 == 2 && sector < 3 * s2 {
                    theta_w
                } else if sector % 3 == 0 && sector < 3 * s0 {
                    theta_t
                } else {
                    theta_e
                }
            } else if sector >= s0 && sector < sec_w {
                theta_h
            } else if sector >= sec_w && sector < sec_w + s2 {
                theta_w
            } else if sector >= sec_w + s2 {
                theta_e
            } else {
                theta_t
            };
            let (c, s) = (theta.cos(), theta.sin());
            let x0 = v[ic];
            let x1 = v[ic + n_offset];
            v[ic] = x0 * c - x1 * s;
            v[ic + n_offset] = x0 * s + x1 * c;
            theta_t *= theta_scale;
            theta_h *= theta_scale;
            theta_w *= theta_scale;
            theta_e *= theta_scale;
            i0 += 2;
        }
        v
    }

    fn oracle_rope_sections(g: &Gguf, arch: &str) -> Option<[i32; 4]> {
        let items = g.kv_i32s(&arch_key(arch, "rope.dimension_sections"))?;
        if items.len() != 4 {
            return None;
        }
        Some([items[0], items[1], items[2], items[3]])
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
        let phi2 = arch == "phi2";
        let eps = if phi2 {
            arch_f32(g, arch, "attention.layer_norm_epsilon").unwrap()
        } else {
            arch_f32(g, arch, "attention.layer_norm_rms_epsilon").unwrap()
        };
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
            let an_b = g.tensor("blk.0.attn_norm.bias").and_then(|t| f32s(t).ok());
            let x = if phi2 {
                oracle_layernorm(&residual, &an, an_b.as_deref(), eps)
            } else {
                oracle_rmsnorm(&residual, &an, eps)
            };
            let attn_norm_output = x.clone();
            let q_full = oracle_add_bias(
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
            let qwen3next = arch == "qwen3next";
            let qwen35 = arch == "qwen35";
            let (q, attn_gate) = if qwen3next || qwen35 {
                let mut q = Vec::new();
                let mut gate = Vec::new();
                for chunk in q_full.chunks(hd * 2) {
                    q.extend_from_slice(&chunk[..hd]);
                    gate.extend_from_slice(&chunk[hd..]);
                }
                (q, Some(gate))
            } else {
                (q_full, None)
            };
            let mut qh: Vec<Vec<f32>> = q.chunks(hd).map(<[f32]>::to_vec).collect();
            let mut kh: Vec<Vec<f32>> = k.chunks(hd).map(<[f32]>::to_vec).collect();
            let vh: Vec<Vec<f32>> = v.chunks(hd).map(<[f32]>::to_vec).collect();
            // Official Qwen3 / Qwen3MoE / Qwen3VL / Qwen3Next / Qwen35 QK-Norm (`LLM_NORM_RMS` on each head) before RoPE.
            if let Some(qn) = g.tensor("blk.0.attn_q_norm.weight") {
                let w = f32s(qn).unwrap();
                for h in &mut qh {
                    *h = oracle_rmsnorm(h, &w, eps);
                }
            }
            if let Some(kn) = g.tensor("blk.0.attn_k_norm.weight") {
                let w = f32s(kn).unwrap();
                for h in &mut kh {
                    *h = oracle_rmsnorm(h, &w, eps);
                }
            }
            // Official llama4.cpp: iRoPE when `(il+1) % n_no_rope_layer_step != 0`.
            // Writer-built tiny is 1 layer with default step 4, so layer 0 uses RoPE.
            let llama4 = arch == "llama4";
            let n_no_rope = if llama4 { LLAMA4_NO_ROPE_LAYER_STEP } else { 0 };
            let use_rope = !llama4 || (n_no_rope > 0 && 1 % n_no_rope != 0);
            let n_expert = arch_u32(g, arch, "expert_count").unwrap_or(0) as usize;
            let qk_l2 = llama4 && use_rope && n_expert != 128;
            if use_rope {
                let sections = if arch == "qwen2vl" || arch == "qwen3vl" || arch == "qwen35" {
                    oracle_rope_sections(g, arch)
                } else {
                    None
                };
                let is_imrope = arch == "qwen3vl" || arch == "qwen35";
                for h in &mut qh {
                    *h = match sections {
                        Some(s) => oracle_rope_multi(
                            h.clone(),
                            [pos, pos, pos, 0],
                            n_rot,
                            base,
                            s,
                            is_imrope,
                        ),
                        None if phi2 => oracle_rope_neox(h.clone(), pos, n_rot, base),
                        None => oracle_rope(h.clone(), pos, n_rot, base),
                    };
                }
                for h in &mut kh {
                    *h = match sections {
                        Some(s) => oracle_rope_multi(
                            h.clone(),
                            [pos, pos, pos, 0],
                            n_rot,
                            base,
                            s,
                            is_imrope,
                        ),
                        None if phi2 => oracle_rope_neox(h.clone(), pos, n_rot, base),
                        None => oracle_rope(h.clone(), pos, n_rot, base),
                    };
                }
                if qk_l2 {
                    for h in &mut qh {
                        *h = oracle_rmsnorm_unweighted(h, eps);
                    }
                    for h in &mut kh {
                        *h = oracle_rmsnorm_unweighted(h, eps);
                    }
                }
            } else {
                let scale = llama4_attn_temp_scale(pos);
                for h in &mut qh {
                    for v in h.iter_mut() {
                        *v *= scale;
                    }
                }
            }
            for (hkv, khv) in kh.iter().enumerate() {
                k_cache[hkv].push(khv.clone());
                v_cache[hkv].push(vh[hkv].clone());
            }
            if phi2 && use_rope {
                let q_scale = 1.0 / (hd as f32).sqrt();
                for h in &mut qh {
                    for v in h.iter_mut() {
                        *v *= q_scale;
                    }
                }
            }
            let seq = pos + 1;
            let inv = if phi2 { 1.0 } else { 1.0 / (hd as f32).sqrt() };
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
            if let Some(gate) = attn_gate.as_ref() {
                for (a, gv) in attn.iter_mut().zip(gate.iter()) {
                    *a *= 1.0 / (1.0 + (-gv).exp());
                }
            }
            let mut x = oracle_add_bias(
                oracle_gemv(g.tensor("blk.0.attn_output.weight").unwrap(), &attn),
                g.tensor("blk.0.attn_output.bias"),
            );
            if phi2 {
                let up = oracle_add_bias(
                    oracle_gemv(g.tensor("blk.0.ffn_up.weight").unwrap(), &attn_norm_output),
                    g.tensor("blk.0.ffn_up.bias"),
                );
                let h: Vec<f32> = up.iter().map(|u| oracle_gelu(*u)).collect();
                let down = oracle_add_bias(
                    oracle_gemv(g.tensor("blk.0.ffn_down.weight").unwrap(), &h),
                    g.tensor("blk.0.ffn_down.bias"),
                );
                x = x
                    .iter()
                    .zip(down.iter())
                    .zip(residual.iter())
                    .map(|((a, d), r)| a + d + r)
                    .collect();
            } else {
                x = x.iter().zip(residual.iter()).map(|(a, b)| a + b).collect();
                let fnorm = if qwen3next || qwen35 {
                    f32s(g.tensor("blk.0.post_attention_norm.weight").unwrap()).unwrap()
                } else {
                    f32s(g.tensor("blk.0.ffn_norm.weight").unwrap()).unwrap()
                };
                let xn = oracle_rmsnorm(&x, &fnorm, eps);
                let llama_moe = arch == "llama" && arch_u32(g, arch, "expert_count").unwrap_or(0) > 0;
                let qwen2moe = arch == "qwen2moe";
                let qwen3moe = arch == "qwen3moe";
                let down = if llama4 {
                    oracle_llama4_moe(g, &xn)
                } else if llama_moe {
                    oracle_llama_moe(g, &xn)
                } else if qwen2moe {
                    oracle_qwen2moe(g, &xn)
                } else if qwen3moe {
                    oracle_qwen3moe(g, &xn)
                } else if qwen3next {
                    oracle_qwen3next(g, &xn)
                } else {
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
                    oracle_gemv(g.tensor("blk.0.ffn_down.weight").unwrap(), &h)
                };
                x = down.iter().zip(x.iter()).map(|(a, b)| a + b).collect();
            }
            let on = f32s(g.tensor("output_norm.weight").unwrap()).unwrap();
            let on_b = g.tensor("output_norm.bias").and_then(|t| f32s(t).ok());
            x = if phi2 {
                oracle_layernorm(&x, &on, on_b.as_deref(), eps)
            } else {
                oracle_rmsnorm(&x, &on, eps)
            };
            let lm_head = g
                .tensor("output.weight")
                .or_else(|| g.tensor("token_embd.weight"))
                .expect("lm_head");
            last = oracle_add_bias(oracle_gemv(lm_head, &x), g.tensor("output.bias"));
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
            tiny_qwen3_gguf(),
            tiny_llama4_gguf(),
            tiny_llama_moe_gguf(),
            tiny_qwen2moe_gguf(),
            tiny_qwen3moe_gguf(),
            tiny_qwen2vl_gguf(),
            tiny_qwen3vl_gguf(),
            tiny_qwen3next_gguf(),
            tiny_qwen35_gguf(),
            tiny_phi2_gguf(),
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
    fn tiny_qwen3_load_gemv_gemm_embed_and_greedy() {
        let bytes = tiny_qwen3_gguf();
        let g = load_gguf(&bytes).expect("load qwen3");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("qwen3".into()))
        );
        assert_eq!(g.kv_u32("qwen3.block_count"), Some(1));
        assert_eq!(g.kv_u32("qwen3.embedding_length"), Some(256));
        assert_eq!(g.kv_u32("qwen3.feed_forward_length"), Some(256));
        assert_eq!(g.kv_u32("qwen3.attention.head_count"), Some(4));
        assert_eq!(g.kv_u32("qwen3.attention.head_count_kv"), Some(2));
        assert!(g.kv_u32("qwen2.block_count").is_none());
        assert!(g.kv_u32("llama.block_count").is_none());
        assert!(g.tensor("blk.0.attn_q_norm.weight").is_some());
        assert!(g.tensor("blk.0.attn_k_norm.weight").is_some());
        assert_eq!(
            g.tensor("blk.0.attn_q_norm.weight").unwrap().n_cols(),
            TINY_N_ROT
        );
        assert_eq!(
            g.tensor("blk.0.attn_k_norm.weight").unwrap().n_cols(),
            TINY_N_ROT
        );
        assert!(g.tensor("blk.0.attn_q.bias").is_none());
        assert!(g.tensor("blk.0.attn_post_norm.weight").is_none());
        assert!(g.tensor("blk.0.ffn_post_norm.weight").is_none());
        assert_eq!(g.kv_bool("tokenizer.ggml.add_bos_token"), Some(false));
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
        assert!(!tok.add_bos);
        let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
        let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
        assert_eq!(out, out2);
        assert!(!out.is_empty());
        // Seq-1 GEMV: softmax of one score is 1, V is un-normed, so QK-Norm
        // cannot change that step. Prefill over several tokens does: per-key
        // K RMS reweights scores (official Qwen3 walk vs llama/qwen2).
        let tokens = [1u32, 2, 3];
        let llama_pref = {
            let lg = load_gguf(&llama_bytes).expect("llama reload");
            let lm = Llama::from_gguf(lg).expect("lm");
            let mut c = lm.new_cache(8).expect("c");
            lm.prefill(&mut c, &tokens).expect("llama pref")
        };
        let mut qc = model.new_cache(8).expect("qc");
        let qwen3_pref = model.prefill(&mut qc, &tokens).expect("qwen3 pref");
        assert_ne!(
            qwen3_pref, llama_pref,
            "qwen3 QK-Norm must change multi-token logits vs llama on the same tiny weights"
        );
    }

    #[test]
    fn tiny_llama4_load_gemv_gemm_embed_and_greedy() {
        let bytes = tiny_llama4_gguf();
        let g = load_gguf(&bytes).expect("load llama4");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("llama4".into()))
        );
        assert_eq!(g.kv_u32("llama4.block_count"), Some(1));
        assert_eq!(g.kv_u32("llama4.embedding_length"), Some(256));
        assert_eq!(g.kv_u32("llama4.feed_forward_length"), Some(256));
        assert_eq!(g.kv_u32("llama4.attention.head_count"), Some(4));
        assert_eq!(g.kv_u32("llama4.attention.head_count_kv"), Some(2));
        assert_eq!(g.kv_u32("llama4.expert_count"), Some(2));
        assert_eq!(g.kv_u32("llama4.expert_used_count"), Some(1));
        assert_eq!(g.kv_u32("llama4.interleave_moe_layer_step"), Some(1));
        assert_eq!(g.kv_u32("llama4.expert_feed_forward_length"), Some(256));
        assert!(g.kv_u32("llama.block_count").is_none());
        assert!(g.kv_u32("mixtral.block_count").is_none());
        assert!(g.tensor("blk.0.attn_q_norm.weight").is_none());
        assert!(g.tensor("blk.0.attn_k_norm.weight").is_none());
        assert!(g.tensor("blk.0.ffn_gate.weight").is_none());
        assert!(g.tensor("blk.0.ffn_up.weight").is_none());
        assert!(g.tensor("blk.0.ffn_down.weight").is_none());
        assert!(g.tensor("blk.0.ffn_gate_inp.weight").is_some());
        assert!(g.tensor("blk.0.ffn_gate_exps.weight").is_some());
        assert!(g.tensor("blk.0.ffn_up_exps.weight").is_some());
        assert!(g.tensor("blk.0.ffn_down_exps.weight").is_some());
        assert!(g.tensor("blk.0.ffn_gate_shexp.weight").is_some());
        assert!(g.tensor("blk.0.ffn_up_shexp.weight").is_some());
        assert!(g.tensor("blk.0.ffn_down_shexp.weight").is_some());
        assert_eq!(
            g.tensor("blk.0.ffn_gate_exps.weight").unwrap().shape,
            &[256, 256, 2]
        );
        assert!(g.tensor("v.blk.0.attn_q.weight").is_none());
        assert_eq!(g.kv_bool("tokenizer.ggml.add_bos_token"), Some(false));
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
        assert!(!tok.add_bos);
        let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
        let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
        assert_eq!(out, out2);
        assert!(!out.is_empty());
        let tokens = [1u32, 2, 3];
        let llama_pref = {
            let lg = load_gguf(&llama_bytes).expect("llama reload");
            let lm = Llama::from_gguf(lg).expect("lm");
            let mut c = lm.new_cache(8).expect("c");
            lm.prefill(&mut c, &tokens).expect("llama pref")
        };
        let mut lc = model.new_cache(8).expect("lc");
        let llama4_pref = model.prefill(&mut lc, &tokens).expect("llama4 pref");
        assert_ne!(
            llama4_pref, llama_pref,
            "llama4 iRoPE/QK-L2/expert FFN must change multi-token logits vs llama"
        );
    }

    #[test]
    fn tiny_llama_moe_load_gemv_gemm_embed_and_greedy() {
        let bytes = tiny_llama_moe_gguf();
        let g = load_gguf(&bytes).expect("load llama moe");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("llama".into()))
        );
        assert_eq!(g.kv_u32("llama.block_count"), Some(1));
        assert_eq!(g.kv_u32("llama.embedding_length"), Some(256));
        assert_eq!(g.kv_u32("llama.feed_forward_length"), Some(256));
        assert_eq!(g.kv_u32("llama.attention.head_count"), Some(4));
        assert_eq!(g.kv_u32("llama.attention.head_count_kv"), Some(2));
        assert_eq!(g.kv_u32("llama.expert_count"), Some(4));
        assert_eq!(g.kv_u32("llama.expert_used_count"), Some(2));
        assert!(g.kv_u32("llama.expert_feed_forward_length").is_none());
        assert!(g.kv_u32("llama.interleave_moe_layer_step").is_none());
        assert!(g.kv_u32("mixtral.block_count").is_none());
        assert!(g.kv_u32("mixtral.expert_count").is_none());
        assert!(g.tensor("blk.0.attn_q_norm.weight").is_none());
        assert!(g.tensor("blk.0.attn_k_norm.weight").is_none());
        assert!(g.tensor("blk.0.ffn_gate.weight").is_none());
        assert!(g.tensor("blk.0.ffn_up.weight").is_none());
        assert!(g.tensor("blk.0.ffn_down.weight").is_none());
        assert!(g.tensor("blk.0.ffn_gate_inp.weight").is_some());
        assert!(g.tensor("blk.0.ffn_gate_exps.weight").is_some());
        assert!(g.tensor("blk.0.ffn_up_exps.weight").is_some());
        assert!(g.tensor("blk.0.ffn_down_exps.weight").is_some());
        assert!(g.tensor("blk.0.ffn_gate_shexp.weight").is_none());
        assert!(g.tensor("blk.0.ffn_up_shexp.weight").is_none());
        assert!(g.tensor("blk.0.ffn_down_shexp.weight").is_none());
        assert_eq!(
            g.tensor("blk.0.ffn_gate_exps.weight").unwrap().shape,
            &[256, 256, 4]
        );
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
        let tokens = [1u32, 2, 3];
        let llama_pref = {
            let lg = load_gguf(&llama_bytes).expect("llama reload");
            let lm = Llama::from_gguf(lg).expect("lm");
            let mut c = lm.new_cache(8).expect("c");
            lm.prefill(&mut c, &tokens).expect("llama pref")
        };
        let mut mc = model.new_cache(8).expect("mc");
        let moe_pref = model.prefill(&mut mc, &tokens).expect("llama moe pref");
        assert_ne!(
            moe_pref, llama_pref,
            "llama MoE softmax/top-k/norm_w must change logits vs dense llama"
        );
        let llama4_pref = {
            let l4 = load_gguf(&tiny_llama4_gguf()).expect("llama4");
            let m4 = Llama::from_gguf(l4).expect("m4");
            let mut c = m4.new_cache(8).expect("c4");
            m4.prefill(&mut c, &tokens).expect("llama4 pref")
        };
        assert_ne!(
            moe_pref, llama4_pref,
            "official llama MoE must not copy Llama4 sigmoid/shared-expert"
        );
    }

    #[test]
    fn tiny_qwen3moe_load_gemv_gemm_embed_and_greedy() {
        let bytes = tiny_qwen3moe_gguf();
        let g = load_gguf(&bytes).expect("load qwen3moe");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("qwen3moe".into()))
        );
        assert_eq!(g.kv_u32("qwen3moe.block_count"), Some(1));
        assert_eq!(g.kv_u32("qwen3moe.embedding_length"), Some(256));
        assert_eq!(g.kv_u32("qwen3moe.feed_forward_length"), Some(256));
        assert_eq!(g.kv_u32("qwen3moe.attention.head_count"), Some(4));
        assert_eq!(g.kv_u32("qwen3moe.attention.head_count_kv"), Some(2));
        assert_eq!(g.kv_u32("qwen3moe.expert_count"), Some(4));
        assert_eq!(g.kv_u32("qwen3moe.expert_used_count"), Some(2));
        assert_eq!(g.kv_u32("qwen3moe.expert_feed_forward_length"), Some(256));
        assert!(g
            .kv_u32("qwen3moe.expert_shared_feed_forward_length")
            .is_none());
        assert!(g.kv_u32("llama.block_count").is_none());
        assert!(g.kv_u32("llama4.block_count").is_none());
        assert!(g.kv_u32("qwen2moe.block_count").is_none());
        assert!(g.kv_u32("mixtral.block_count").is_none());
        assert!(g.tensor("blk.0.attn_q_norm.weight").is_some());
        assert!(g.tensor("blk.0.attn_k_norm.weight").is_some());
        assert_eq!(
            g.tensor("blk.0.attn_q_norm.weight").unwrap().n_cols(),
            TINY_N_ROT
        );
        assert_eq!(
            g.tensor("blk.0.attn_k_norm.weight").unwrap().n_cols(),
            TINY_N_ROT
        );
        assert!(g.tensor("blk.0.ffn_gate.weight").is_none());
        assert!(g.tensor("blk.0.ffn_up.weight").is_none());
        assert!(g.tensor("blk.0.ffn_down.weight").is_none());
        assert!(g.tensor("blk.0.ffn_gate_inp.weight").is_some());
        assert!(g.tensor("blk.0.ffn_gate_exps.weight").is_some());
        assert!(g.tensor("blk.0.ffn_up_exps.weight").is_some());
        assert!(g.tensor("blk.0.ffn_down_exps.weight").is_some());
        assert!(g.tensor("blk.0.ffn_gate_inp_shexp.weight").is_none());
        assert!(g.tensor("blk.0.ffn_gate_shexp.weight").is_none());
        assert!(g.tensor("blk.0.ffn_up_shexp.weight").is_none());
        assert!(g.tensor("blk.0.ffn_down_shexp.weight").is_none());
        assert_eq!(
            g.tensor("blk.0.ffn_gate_exps.weight").unwrap().shape,
            &[256, 256, 4]
        );
        assert_eq!(g.kv_bool("tokenizer.ggml.add_bos_token"), Some(false));
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
        assert!(!tok.add_bos);
        let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
        let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
        assert_eq!(out, out2);
        assert!(!out.is_empty());
        let tokens = [1u32, 2, 3];
        let llama_pref = {
            let lg = load_gguf(&llama_bytes).expect("llama reload");
            let lm = Llama::from_gguf(lg).expect("lm");
            let mut c = lm.new_cache(8).expect("c");
            lm.prefill(&mut c, &tokens).expect("llama pref")
        };
        let mut qc = model.new_cache(8).expect("qc");
        let qwen3moe_pref = model.prefill(&mut qc, &tokens).expect("qwen3moe pref");
        assert_ne!(
            qwen3moe_pref, llama_pref,
            "qwen3moe QK-Norm+softmax/norm_w must change logits vs dense llama"
        );
        let qwen3_pref = {
            let q3 = load_gguf(&tiny_qwen3_gguf()).expect("qwen3");
            let m3 = Llama::from_gguf(q3).expect("m3");
            let mut c = m3.new_cache(8).expect("c3");
            m3.prefill(&mut c, &tokens).expect("qwen3 pref")
        };
        assert_ne!(
            qwen3moe_pref, qwen3_pref,
            "qwen3moe MoE must change logits vs dense qwen3 QK-Norm"
        );
        let llama_moe_pref = {
            let mg = load_gguf(&tiny_llama_moe_gguf()).expect("llama moe");
            let mm = Llama::from_gguf(mg).expect("mm");
            let mut c = mm.new_cache(8).expect("cm");
            mm.prefill(&mut c, &tokens).expect("llama moe pref")
        };
        assert_ne!(
            qwen3moe_pref, llama_moe_pref,
            "qwen3moe QK-Norm must change logits vs llama-MoE on the same expert seeds"
        );
        let qwen2moe_pref = {
            let q2 = load_gguf(&tiny_qwen2moe_gguf()).expect("qwen2moe");
            let m2 = Llama::from_gguf(q2).expect("m2");
            let mut c = m2.new_cache(8).expect("c2");
            m2.prefill(&mut c, &tokens).expect("qwen2moe pref")
        };
        assert_ne!(
            qwen3moe_pref, qwen2moe_pref,
            "qwen3moe must not copy qwen2moe shexp / norm_w=false"
        );
        let llama4_pref = {
            let l4 = load_gguf(&tiny_llama4_gguf()).expect("llama4");
            let m4 = Llama::from_gguf(l4).expect("m4");
            let mut c = m4.new_cache(8).expect("c4");
            m4.prefill(&mut c, &tokens).expect("llama4 pref")
        };
        assert_ne!(
            qwen3moe_pref, llama4_pref,
            "qwen3moe must not copy Llama4 sigmoid / weight-before-FFN"
        );
    }

    #[test]
    fn tiny_qwen2vl_load_gemv_gemm_embed_and_greedy() {
        let bytes = tiny_qwen2vl_gguf();
        let g = load_gguf(&bytes).expect("load qwen2vl");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("qwen2vl".into()))
        );
        assert_eq!(g.kv_u32("qwen2vl.block_count"), Some(1));
        assert_eq!(g.kv_u32("qwen2vl.embedding_length"), Some(256));
        assert_eq!(g.kv_u32("qwen2vl.feed_forward_length"), Some(256));
        assert_eq!(g.kv_u32("qwen2vl.attention.head_count"), Some(4));
        assert_eq!(g.kv_u32("qwen2vl.attention.head_count_kv"), Some(2));
        assert!(g.kv_u32("qwen2vl.rope.dimension_count").is_none());
        assert_eq!(
            g.kv_i32s("qwen2vl.rope.dimension_sections"),
            Some(TINY_QWEN2VL_ROPE_SECTIONS.to_vec())
        );
        assert!(g.kv_u32("llama.block_count").is_none());
        assert!(g.kv_u32("qwen2.block_count").is_none());
        assert!(g.kv_u32("qwen3.block_count").is_none());
        assert!(g.kv_u32("qwen3vl.block_count").is_none());
        assert!(g.kv_u32("qwen3vlmoe.block_count").is_none());
        assert!(g.kv_u32("qwen25vl.block_count").is_none());
        assert!(g.kv_u32("mixtral.block_count").is_none());
        assert!(g.tensor("blk.0.attn_q.bias").is_some());
        assert!(g.tensor("blk.0.attn_k.bias").is_some());
        assert!(g.tensor("blk.0.attn_v.bias").is_some());
        assert!(g.tensor("blk.0.attn_q_norm.weight").is_none());
        assert!(g.tensor("blk.0.attn_k_norm.weight").is_none());
        assert!(g.tensor("blk.0.ffn_gate.weight").is_some());
        assert_eq!(g.kv_bool("tokenizer.ggml.add_bos_token"), Some(false));
        let model = Llama::from_gguf(g.clone()).expect("model");
        let x = pat_f32(TINY_N_EMBD, 21);
        let got_gemv = model.gemv_output(&x).expect("gemv");
        let exp_gemv = oracle_gemv(g.tensor("output.weight").unwrap(), &x);
        assert_logits_match(&got_gemv, &exp_gemv);
        let mut x2 = pat_f32(TINY_N_EMBD, 22);
        x2.extend(pat_f32(TINY_N_EMBD, 23));
        let got_gemm = model.gemm_output(2, &x2).expect("gemm");
        let qwen2_bytes = tiny_qwen2_gguf();
        let qwen2_g = load_gguf(&qwen2_bytes).expect("qwen2");
        let qwen2_m = Llama::from_gguf(qwen2_g).expect("qwen2 m");
        let qwen2_gemm = qwen2_m.gemm_output(2, &x2).expect("qwen2 gemm");
        assert_logits_match(&got_gemm, &qwen2_gemm);
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
        let tokens = [1u32, 2, 3];
        let qwen2_pref = {
            let qg = load_gguf(&qwen2_bytes).expect("qwen2 reload");
            let qm = Llama::from_gguf(qg).expect("qm");
            let mut c = qm.new_cache(8).expect("c");
            qm.prefill(&mut c, &tokens).expect("qwen2 pref")
        };
        let qwen3_pref = {
            let q3 = load_gguf(&tiny_qwen3_gguf()).expect("qwen3");
            let m3 = Llama::from_gguf(q3).expect("m3");
            let mut c = m3.new_cache(8).expect("c3");
            m3.prefill(&mut c, &tokens).expect("qwen3 pref")
        };
        let mut qc = model.new_cache(8).expect("qc");
        let qwen2vl_pref = model.prefill(&mut qc, &tokens).expect("qwen2vl pref");
        assert_ne!(
            qwen2vl_pref, qwen2_pref,
            "qwen2vl m-RoPE must change multi-token logits vs qwen2 adjacent-pair RoPE"
        );
        assert_ne!(
            qwen2vl_pref, qwen3_pref,
            "qwen2vl must not copy qwen3 QK-Norm"
        );
    }

    #[test]
    fn tiny_qwen3vl_load_gemv_gemm_embed_and_greedy() {
        let bytes = tiny_qwen3vl_gguf();
        let g = load_gguf(&bytes).expect("load qwen3vl");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("qwen3vl".into()))
        );
        assert_eq!(g.kv_u32("qwen3vl.block_count"), Some(1));
        assert_eq!(g.kv_u32("qwen3vl.embedding_length"), Some(256));
        assert_eq!(g.kv_u32("qwen3vl.feed_forward_length"), Some(256));
        assert_eq!(g.kv_u32("qwen3vl.attention.head_count"), Some(4));
        assert_eq!(g.kv_u32("qwen3vl.attention.head_count_kv"), Some(2));
        assert_eq!(g.kv_u32("qwen3vl.rope.dimension_count"), Some(64));
        assert_eq!(
            g.kv_i32s("qwen3vl.rope.dimension_sections"),
            Some(TINY_QWEN3VL_ROPE_SECTIONS.to_vec())
        );
        assert!(g.kv_u32("llama.block_count").is_none());
        assert!(g.kv_u32("qwen2.block_count").is_none());
        assert!(g.kv_u32("qwen3.block_count").is_none());
        assert!(g.kv_u32("qwen2vl.block_count").is_none());
        assert!(g.kv_u32("qwen3vlmoe.block_count").is_none());
        assert!(g.kv_u32("qwen25vl.block_count").is_none());
        assert!(g.kv_u32("mixtral.block_count").is_none());
        assert!(g.kv_u32("qwen3vl.expert_count").is_none());
        assert!(g.tensor("blk.0.attn_q_norm.weight").is_some());
        assert!(g.tensor("blk.0.attn_k_norm.weight").is_some());
        assert_eq!(
            g.tensor("blk.0.attn_q_norm.weight").unwrap().n_cols(),
            TINY_N_ROT
        );
        assert_eq!(
            g.tensor("blk.0.attn_k_norm.weight").unwrap().n_cols(),
            TINY_N_ROT
        );
        assert!(g.tensor("blk.0.attn_q.bias").is_none());
        assert!(g.tensor("blk.0.attn_k.bias").is_none());
        assert!(g.tensor("blk.0.attn_v.bias").is_none());
        assert!(g.tensor("blk.0.attn_post_norm.weight").is_none());
        assert!(g.tensor("blk.0.ffn_post_norm.weight").is_none());
        assert!(g.tensor("blk.0.ffn_gate.weight").is_some());
        assert!(g.tensor("blk.0.ffn_gate_inp.weight").is_none());
        assert_eq!(g.kv_bool("tokenizer.ggml.add_bos_token"), Some(false));
        let model = Llama::from_gguf(g.clone()).expect("model");
        let x = pat_f32(TINY_N_EMBD, 21);
        let got_gemv = model.gemv_output(&x).expect("gemv");
        let exp_gemv = oracle_gemv(g.tensor("output.weight").unwrap(), &x);
        assert_logits_match(&got_gemv, &exp_gemv);
        let mut x2 = pat_f32(TINY_N_EMBD, 22);
        x2.extend(pat_f32(TINY_N_EMBD, 23));
        let got_gemm = model.gemm_output(2, &x2).expect("gemm");
        let qwen3_bytes = tiny_qwen3_gguf();
        let qwen3_g = load_gguf(&qwen3_bytes).expect("qwen3");
        let qwen3_m = Llama::from_gguf(qwen3_g).expect("qwen3 m");
        let qwen3_gemm = qwen3_m.gemm_output(2, &x2).expect("qwen3 gemm");
        assert_logits_match(&got_gemm, &qwen3_gemm);
        let emb = model.embed_token(3).expect("embed");
        let exp_emb = oracle_embed(g.tensor("token_embd.weight").unwrap(), 3);
        assert_logits_match(&emb, &exp_emb);
        load_fwd_match(&bytes, 3);
        load_prefill_match(&bytes, &[1, 2, 3]);
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        assert!(!tok.add_bos);
        let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
        let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
        assert_eq!(out, out2);
        assert!(!out.is_empty());
        let tokens = [1u32, 2, 3];
        let qwen3_pref = {
            let q3 = load_gguf(&qwen3_bytes).expect("qwen3 reload");
            let m3 = Llama::from_gguf(q3).expect("m3");
            let mut c = m3.new_cache(8).expect("c3");
            m3.prefill(&mut c, &tokens).expect("qwen3 pref")
        };
        let qwen2vl_pref = {
            let qv = load_gguf(&tiny_qwen2vl_gguf()).expect("qwen2vl");
            let mv = Llama::from_gguf(qv).expect("mv");
            let mut c = mv.new_cache(8).expect("cv");
            mv.prefill(&mut c, &tokens).expect("qwen2vl pref")
        };
        let mut qc = model.new_cache(8).expect("qc");
        let qwen3vl_pref = model.prefill(&mut qc, &tokens).expect("qwen3vl pref");
        assert_ne!(
            qwen3vl_pref, qwen3_pref,
            "qwen3vl IMROPE must change multi-token logits vs qwen3 adjacent-pair RoPE"
        );
        assert_ne!(
            qwen3vl_pref, qwen2vl_pref,
            "qwen3vl must not copy qwen2vl MROPE / QKV bias"
        );
    }

    #[test]
    fn tiny_qwen3next_load_gemv_gemm_embed_and_greedy() {
        let bytes = tiny_qwen3next_gguf();
        let g = load_gguf(&bytes).expect("load qwen3next");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("qwen3next".into()))
        );
        assert_eq!(g.kv_u32("qwen3next.block_count"), Some(1));
        assert_eq!(g.kv_u32("qwen3next.embedding_length"), Some(256));
        assert_eq!(g.kv_u32("qwen3next.feed_forward_length"), Some(256));
        assert_eq!(g.kv_u32("qwen3next.attention.head_count"), Some(4));
        assert_eq!(g.kv_u32("qwen3next.attention.head_count_kv"), Some(2));
        assert_eq!(g.kv_u32("qwen3next.expert_count"), Some(4));
        assert_eq!(g.kv_u32("qwen3next.expert_used_count"), Some(2));
        assert_eq!(g.kv_u32("qwen3next.expert_feed_forward_length"), Some(256));
        assert_eq!(
            g.kv_u32("qwen3next.expert_shared_feed_forward_length"),
            Some(256)
        );
        assert_eq!(g.kv_u32("qwen3next.full_attention_interval"), Some(1));
        assert_eq!(g.kv_u32("qwen3next.rope.dimension_count"), Some(16));
        assert_eq!(g.kv_u32("qwen3next.ssm.conv_kernel"), Some(4));
        assert_eq!(g.kv_u32("qwen3next.ssm.inner_size"), Some(32));
        assert_eq!(g.kv_u32("qwen3next.ssm.state_size"), Some(16));
        assert_eq!(g.kv_u32("qwen3next.ssm.time_step_rank"), Some(2));
        assert_eq!(g.kv_u32("qwen3next.ssm.group_count"), Some(2));
        assert!(g.kv_u32("llama.block_count").is_none());
        assert!(g.kv_u32("qwen3.block_count").is_none());
        assert!(g.kv_u32("qwen3moe.block_count").is_none());
        assert!(g.kv_u32("qwen2moe.block_count").is_none());
        assert!(g.kv_u32("qwen3vl.block_count").is_none());
        assert!(g.kv_u32("qwen3vlmoe.block_count").is_none());
        assert!(g.kv_u32("mixtral.block_count").is_none());
        assert!(g.tensor("blk.0.attn_q_norm.weight").is_some());
        assert!(g.tensor("blk.0.attn_k_norm.weight").is_some());
        assert_eq!(
            g.tensor("blk.0.attn_q_norm.weight").unwrap().n_cols(),
            TINY_N_EMBD / TINY_N_HEAD
        );
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().n_rows(), 512);
        assert!(g.tensor("blk.0.post_attention_norm.weight").is_some());
        assert!(g.tensor("blk.0.ffn_norm.weight").is_none());
        assert!(g.tensor("blk.0.ffn_gate_inp.weight").is_some());
        assert!(g.tensor("blk.0.ffn_gate_inp_shexp.weight").is_some());
        assert!(g.tensor("blk.0.attn_q.bias").is_none());
        assert_eq!(g.kv_bool("tokenizer.ggml.add_bos_token"), Some(false));
        let model = Llama::from_gguf(g.clone()).expect("model");
        let x = pat_f32(TINY_N_EMBD, 21);
        let got_gemv = model.gemv_output(&x).expect("gemv");
        let exp_gemv = oracle_gemv(g.tensor("output.weight").unwrap(), &x);
        assert_logits_match(&got_gemv, &exp_gemv);
        let mut x2 = pat_f32(TINY_N_EMBD, 22);
        x2.extend(pat_f32(TINY_N_EMBD, 23));
        let got_gemm = model.gemm_output(2, &x2).expect("gemm");
        let exp_gemm = {
            let mut y = oracle_gemv(g.tensor("output.weight").unwrap(), &x2[..TINY_N_EMBD]);
            y.extend(oracle_gemv(
                g.tensor("output.weight").unwrap(),
                &x2[TINY_N_EMBD..],
            ));
            y
        };
        assert_logits_match(&got_gemm, &exp_gemm);
        let emb = model.embed_token(3).expect("embed");
        let exp_emb = oracle_embed(g.tensor("token_embd.weight").unwrap(), 3);
        assert_logits_match(&emb, &exp_emb);
        load_fwd_match(&bytes, 3);
        load_prefill_match(&bytes, &[1, 2, 3]);
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        assert!(!tok.add_bos);
        let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
        let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
        assert_eq!(out, out2);
        assert!(!out.is_empty());
        let tokens = [1u32, 2, 3];
        let qwen3_pref = {
            let q3 = load_gguf(&tiny_qwen3_gguf()).expect("qwen3");
            let m3 = Llama::from_gguf(q3).expect("m3");
            let mut c = m3.new_cache(8).expect("c3");
            m3.prefill(&mut c, &tokens).expect("qwen3 pref")
        };
        let qwen3moe_pref = {
            let qm = load_gguf(&tiny_qwen3moe_gguf()).expect("qwen3moe");
            let mm = Llama::from_gguf(qm).expect("mm");
            let mut c = mm.new_cache(8).expect("cm");
            mm.prefill(&mut c, &tokens).expect("qwen3moe pref")
        };
        let qwen2moe_pref = {
            let q2 = load_gguf(&tiny_qwen2moe_gguf()).expect("qwen2moe");
            let m2 = Llama::from_gguf(q2).expect("m2");
            let mut c = m2.new_cache(8).expect("c2");
            m2.prefill(&mut c, &tokens).expect("qwen2moe pref")
        };
        let qwen3vl_pref = {
            let qv = load_gguf(&tiny_qwen3vl_gguf()).expect("qwen3vl");
            let mv = Llama::from_gguf(qv).expect("mv");
            let mut c = mv.new_cache(8).expect("cv");
            mv.prefill(&mut c, &tokens).expect("qwen3vl pref")
        };
        let mut qc = model.new_cache(8).expect("qc");
        let qwen3next_pref = model.prefill(&mut qc, &tokens).expect("qwen3next pref");
        assert_ne!(
            qwen3next_pref, qwen3_pref,
            "qwen3next gated-Q/MoE must change logits vs dense qwen3"
        );
        assert_ne!(
            qwen3next_pref, qwen3moe_pref,
            "qwen3next must not copy qwen3moe (no shexp / no gated Q)"
        );
        assert_ne!(
            qwen3next_pref, qwen2moe_pref,
            "qwen3next must not copy qwen2moe shexp / norm_w=false"
        );
        assert_ne!(
            qwen3next_pref, qwen3vl_pref,
            "qwen3next must not copy qwen3vl IMROPE"
        );
    }

    #[test]
    fn qwen3next_zero_experts_names_arch() {
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                (
                    "general.architecture".into(),
                    Kv::String("qwen3next".into()),
                ),
                ("qwen3next.block_count".into(), Kv::U32(1)),
                ("qwen3next.embedding_length".into(), Kv::U32(256)),
                ("qwen3next.feed_forward_length".into(), Kv::U32(256)),
                ("qwen3next.attention.head_count".into(), Kv::U32(4)),
                ("qwen3next.attention.head_count_kv".into(), Kv::U32(2)),
                ("qwen3next.expert_count".into(), Kv::U32(0)),
                ("qwen3next.expert_used_count".into(), Kv::U32(2)),
                ("qwen3next.ssm.conv_kernel".into(), Kv::U32(4)),
                ("qwen3next.ssm.inner_size".into(), Kv::U32(32)),
                ("qwen3next.ssm.state_size".into(), Kv::U32(16)),
                ("qwen3next.ssm.time_step_rank".into(), Kv::U32(2)),
                ("qwen3next.ssm.group_count".into(), Kv::U32(2)),
                ("qwen3next.rope.freq_base".into(), Kv::F32(10_000.0)),
                (
                    "qwen3next.attention.layer_norm_rms_epsilon".into(),
                    Kv::F32(1.0 / 100_000.0),
                ),
            ],
            &[],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected zero experts"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("qwen3next") && err.contains("zero experts"),
            "error should name zero experts: {err}"
        );
    }

    #[test]
    fn qwen3next_missing_ssm_names_key() {
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                (
                    "general.architecture".into(),
                    Kv::String("qwen3next".into()),
                ),
                ("qwen3next.block_count".into(), Kv::U32(1)),
                ("qwen3next.embedding_length".into(), Kv::U32(256)),
                ("qwen3next.feed_forward_length".into(), Kv::U32(256)),
                ("qwen3next.attention.head_count".into(), Kv::U32(4)),
                ("qwen3next.attention.head_count_kv".into(), Kv::U32(2)),
                ("qwen3next.expert_count".into(), Kv::U32(4)),
                ("qwen3next.expert_used_count".into(), Kv::U32(2)),
                ("qwen3next.full_attention_interval".into(), Kv::U32(1)),
                ("qwen3next.rope.freq_base".into(), Kv::F32(10_000.0)),
                (
                    "qwen3next.attention.layer_norm_rms_epsilon".into(),
                    Kv::F32(1.0 / 100_000.0),
                ),
            ],
            &[],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected missing ssm"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("qwen3next.ssm."),
            "error should name ssm kv key: {err}"
        );
    }

    #[test]
    fn tiny_qwen35_load_gemv_gemm_embed_and_greedy() {
        let bytes = tiny_qwen35_gguf();
        let g = load_gguf(&bytes).expect("load qwen35");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("qwen35".into()))
        );
        assert_eq!(g.kv_u32("qwen35.block_count"), Some(1));
        assert_eq!(g.kv_u32("qwen35.embedding_length"), Some(256));
        assert_eq!(g.kv_u32("qwen35.feed_forward_length"), Some(256));
        assert_eq!(g.kv_u32("qwen35.attention.head_count"), Some(4));
        assert_eq!(g.kv_u32("qwen35.attention.head_count_kv"), Some(2));
        assert_eq!(g.kv_u32("qwen35.full_attention_interval"), Some(1));
        assert_eq!(g.kv_u32("qwen35.rope.dimension_count"), Some(64));
        assert_eq!(
            g.kv_i32s("qwen35.rope.dimension_sections"),
            Some(TINY_QWEN35_ROPE_SECTIONS.to_vec())
        );
        assert_eq!(g.kv_u32("qwen35.ssm.conv_kernel"), Some(4));
        assert_eq!(g.kv_u32("qwen35.ssm.inner_size"), Some(32));
        assert_eq!(g.kv_u32("qwen35.ssm.state_size"), Some(16));
        assert_eq!(g.kv_u32("qwen35.ssm.time_step_rank"), Some(2));
        assert_eq!(g.kv_u32("qwen35.ssm.group_count"), Some(2));
        assert!(g.kv_u32("qwen35.expert_count").is_none());
        assert!(g.kv_u32("llama.block_count").is_none());
        assert!(g.kv_u32("qwen3.block_count").is_none());
        assert!(g.kv_u32("qwen3moe.block_count").is_none());
        assert!(g.kv_u32("qwen3vl.block_count").is_none());
        assert!(g.kv_u32("qwen3next.block_count").is_none());
        assert!(g.kv_u32("qwen3vlmoe.block_count").is_none());
        assert!(g.kv_u32("mixtral.block_count").is_none());
        assert!(g.tensor("blk.0.attn_q_norm.weight").is_some());
        assert!(g.tensor("blk.0.attn_k_norm.weight").is_some());
        assert_eq!(
            g.tensor("blk.0.attn_q_norm.weight").unwrap().n_cols(),
            TINY_N_EMBD / TINY_N_HEAD
        );
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().n_rows(), 512);
        assert!(g.tensor("blk.0.post_attention_norm.weight").is_some());
        assert!(g.tensor("blk.0.ffn_norm.weight").is_none());
        assert!(g.tensor("blk.0.ffn_gate.weight").is_some());
        assert!(g.tensor("blk.0.ffn_up.weight").is_some());
        assert!(g.tensor("blk.0.ffn_down.weight").is_some());
        assert!(g.tensor("blk.0.ffn_gate_inp.weight").is_none());
        assert!(g.tensor("blk.0.ffn_gate_exps.weight").is_none());
        assert!(g.tensor("blk.0.attn_q.bias").is_none());
        assert_eq!(g.kv_bool("tokenizer.ggml.add_bos_token"), Some(false));
        let model = Llama::from_gguf(g.clone()).expect("model");
        let x = pat_f32(TINY_N_EMBD, 21);
        let got_gemv = model.gemv_output(&x).expect("gemv");
        let exp_gemv = oracle_gemv(g.tensor("output.weight").unwrap(), &x);
        assert_logits_match(&got_gemv, &exp_gemv);
        let mut x2 = pat_f32(TINY_N_EMBD, 22);
        x2.extend(pat_f32(TINY_N_EMBD, 23));
        let got_gemm = model.gemm_output(2, &x2).expect("gemm");
        let exp_gemm = {
            let mut y = oracle_gemv(g.tensor("output.weight").unwrap(), &x2[..TINY_N_EMBD]);
            y.extend(oracle_gemv(
                g.tensor("output.weight").unwrap(),
                &x2[TINY_N_EMBD..],
            ));
            y
        };
        assert_logits_match(&got_gemm, &exp_gemm);
        let emb = model.embed_token(3).expect("embed");
        let exp_emb = oracle_embed(g.tensor("token_embd.weight").unwrap(), 3);
        assert_logits_match(&emb, &exp_emb);
        load_fwd_match(&bytes, 3);
        load_prefill_match(&bytes, &[1, 2, 3]);
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        assert!(!tok.add_bos);
        let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
        let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
        assert_eq!(out, out2);
        assert!(!out.is_empty());
        let tokens = [1u32, 2, 3];
        let qwen3_pref = {
            let q3 = load_gguf(&tiny_qwen3_gguf()).expect("qwen3");
            let m3 = Llama::from_gguf(q3).expect("m3");
            let mut c = m3.new_cache(8).expect("c3");
            m3.prefill(&mut c, &tokens).expect("qwen3 pref")
        };
        let qwen3vl_pref = {
            let qv = load_gguf(&tiny_qwen3vl_gguf()).expect("qwen3vl");
            let mv = Llama::from_gguf(qv).expect("mv");
            let mut c = mv.new_cache(8).expect("cv");
            mv.prefill(&mut c, &tokens).expect("qwen3vl pref")
        };
        let qwen3next_pref = {
            let qn = load_gguf(&tiny_qwen3next_gguf()).expect("qwen3next");
            let mn = Llama::from_gguf(qn).expect("mn");
            let mut c = mn.new_cache(8).expect("cn");
            mn.prefill(&mut c, &tokens).expect("qwen3next pref")
        };
        let mut qc = model.new_cache(8).expect("qc");
        let qwen35_pref = model.prefill(&mut qc, &tokens).expect("qwen35 pref");
        assert_ne!(
            qwen35_pref, qwen3_pref,
            "qwen35 gated-Q/IMROPE must change logits vs dense qwen3"
        );
        assert_ne!(
            qwen35_pref, qwen3vl_pref,
            "qwen35 must not copy qwen3vl (no gated Q / no post_attention_norm)"
        );
        assert_ne!(
            qwen35_pref, qwen3next_pref,
            "qwen35 must not copy qwen3next MoE / partial RoPE"
        );
    }

    #[test]
    fn tiny_phi2_load_gemv_gemm_embed_and_greedy() {
        let bytes = tiny_phi2_gguf();
        let g = load_gguf(&bytes).expect("load phi2");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("phi2".into()))
        );
        assert_eq!(g.kv_u32("phi2.block_count"), Some(1));
        assert_eq!(g.kv_u32("phi2.embedding_length"), Some(256));
        assert_eq!(g.kv_u32("phi2.feed_forward_length"), Some(1024));
        assert_eq!(g.kv_u32("phi2.attention.head_count"), Some(4));
        assert_eq!(g.kv_u32("phi2.attention.head_count_kv"), Some(4));
        assert_eq!(g.kv_u32("phi2.rope.dimension_count"), Some(32));
        assert_eq!(
            g.kv_f32("phi2.attention.layer_norm_epsilon"),
            Some(1.0 / 100_000.0)
        );
        assert!(g.kv_f32("phi2.attention.layer_norm_rms_epsilon").is_none());
        assert!(g.kv_u32("llama.block_count").is_none());
        assert!(g.kv_u32("phi3.block_count").is_none());
        assert!(g.kv_u32("qwen35.block_count").is_none());
        assert!(g.kv_u32("qwen3vlmoe.block_count").is_none());
        assert!(g.kv_u32("mixtral.block_count").is_none());
        assert!(g.tensor("blk.0.attn_norm.bias").is_some());
        assert!(g.tensor("output_norm.bias").is_some());
        assert!(g.tensor("output.bias").is_some());
        assert!(g.tensor("blk.0.attn_output.bias").is_some());
        assert!(g.tensor("blk.0.ffn_up.bias").is_some());
        assert!(g.tensor("blk.0.ffn_down.bias").is_some());
        assert!(g.tensor("blk.0.ffn_gate.weight").is_none());
        assert!(g.tensor("blk.0.ffn_norm.weight").is_none());
        assert!(g.tensor("blk.0.attn_q_norm.weight").is_none());
        assert!(g.tensor("blk.0.post_attention_norm.weight").is_none());
        assert_eq!(g.kv_bool("tokenizer.ggml.add_bos_token"), Some(false));
        let model = Llama::from_gguf(g.clone()).expect("model");
        let x = pat_f32(TINY_N_EMBD, 21);
        let got_gemv = model.gemv_output(&x).expect("gemv");
        let exp_gemv = oracle_gemv(g.tensor("output.weight").unwrap(), &x);
        assert_logits_match(&got_gemv, &exp_gemv);
        let mut x2 = pat_f32(TINY_N_EMBD, 22);
        x2.extend(pat_f32(TINY_N_EMBD, 23));
        let got_gemm = model.gemm_output(2, &x2).expect("gemm");
        let exp_gemm = {
            let mut y = oracle_gemv(g.tensor("output.weight").unwrap(), &x2[..TINY_N_EMBD]);
            y.extend(oracle_gemv(
                g.tensor("output.weight").unwrap(),
                &x2[TINY_N_EMBD..],
            ));
            y
        };
        assert_logits_match(&got_gemm, &exp_gemm);
        let emb = model.embed_token(3).expect("embed");
        let exp_emb = oracle_embed(g.tensor("token_embd.weight").unwrap(), 3);
        assert_logits_match(&emb, &exp_emb);
        load_fwd_match(&bytes, 3);
        load_prefill_match(&bytes, &[1, 2, 3]);
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        assert!(!tok.add_bos);
        let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
        let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
        assert_eq!(out, out2);
        assert!(!out.is_empty());
        let tokens = [1u32, 2, 3];
        let llama_pref = {
            let l = load_gguf(&tiny_llama_gguf()).expect("llama");
            let m = Llama::from_gguf(l).expect("ml");
            let mut c = m.new_cache(8).expect("cl");
            m.prefill(&mut c, &tokens).expect("llama pref")
        };
        let phi3_pref = {
            let p = load_gguf(&tiny_phi3_gguf()).expect("phi3");
            let m = Llama::from_gguf(p).expect("mp");
            let mut c = m.new_cache(8).expect("cp");
            m.prefill(&mut c, &tokens).expect("phi3 pref")
        };
        let gemma_pref = {
            let ge = load_gguf(&tiny_gemma_gguf()).expect("gemma");
            let m = Llama::from_gguf(ge).expect("mg");
            let mut c = m.new_cache(8).expect("cg");
            m.prefill(&mut c, &tokens).expect("gemma pref")
        };
        let mut qc = model.new_cache(8).expect("qc");
        let phi2_pref = model.prefill(&mut qc, &tokens).expect("phi2 pref");
        assert_ne!(
            phi2_pref, llama_pref,
            "phi2 LayerNorm/GELU-seq/NEOX must change logits vs llama"
        );
        assert_ne!(
            phi2_pref, phi3_pref,
            "phi2 must not copy phi3 (RMSNorm + SwiGLU)"
        );
        assert_ne!(
            phi2_pref, gemma_pref,
            "phi2 must not copy gemma (embed-scale + GeGLU)"
        );
    }

    #[test]
    fn phi2_missing_layer_norm_epsilon_names_key() {
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("phi2".into())),
                ("phi2.block_count".into(), Kv::U32(1)),
                ("phi2.embedding_length".into(), Kv::U32(256)),
                ("phi2.feed_forward_length".into(), Kv::U32(1024)),
                ("phi2.attention.head_count".into(), Kv::U32(4)),
                ("phi2.attention.head_count_kv".into(), Kv::U32(4)),
                ("phi2.rope.freq_base".into(), Kv::F32(10_000.0)),
            ],
            &[],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected missing layer_norm_epsilon"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("phi2.attention.layer_norm_epsilon"),
            "error should name kv key: {err}"
        );
    }

    #[test]
    fn qwen35_missing_ssm_names_key() {
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("qwen35".into())),
                ("qwen35.block_count".into(), Kv::U32(1)),
                ("qwen35.embedding_length".into(), Kv::U32(256)),
                ("qwen35.feed_forward_length".into(), Kv::U32(256)),
                ("qwen35.attention.head_count".into(), Kv::U32(4)),
                ("qwen35.attention.head_count_kv".into(), Kv::U32(2)),
                ("qwen35.full_attention_interval".into(), Kv::U32(1)),
                ("qwen35.rope.freq_base".into(), Kv::F32(10_000.0)),
                (
                    "qwen35.attention.layer_norm_rms_epsilon".into(),
                    Kv::F32(1.0 / 100_000.0),
                ),
            ],
            &[],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected missing ssm"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("qwen35.ssm."),
            "error should name ssm kv key: {err}"
        );
    }

    #[test]
    fn qwen35_missing_dimension_sections_names_key() {
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("qwen35".into())),
                ("qwen35.block_count".into(), Kv::U32(1)),
                ("qwen35.embedding_length".into(), Kv::U32(256)),
                ("qwen35.feed_forward_length".into(), Kv::U32(256)),
                ("qwen35.attention.head_count".into(), Kv::U32(4)),
                ("qwen35.attention.head_count_kv".into(), Kv::U32(2)),
                ("qwen35.ssm.conv_kernel".into(), Kv::U32(4)),
                ("qwen35.ssm.inner_size".into(), Kv::U32(32)),
                ("qwen35.ssm.state_size".into(), Kv::U32(16)),
                ("qwen35.ssm.time_step_rank".into(), Kv::U32(2)),
                ("qwen35.ssm.group_count".into(), Kv::U32(2)),
                ("qwen35.full_attention_interval".into(), Kv::U32(1)),
                ("qwen35.rope.freq_base".into(), Kv::F32(10_000.0)),
                (
                    "qwen35.attention.layer_norm_rms_epsilon".into(),
                    Kv::F32(1.0 / 100_000.0),
                ),
            ],
            &[],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected missing sections"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("qwen35.rope.dimension_sections"),
            "error should name kv key: {err}"
        );
    }

    #[test]
    fn qwen35_default_interval_refuses_linear_attn() {
        let n_embd = TINY_N_EMBD;
        let ones = vec![1.0f32; n_embd];
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("qwen35".into())),
                ("qwen35.block_count".into(), Kv::U32(1)),
                ("qwen35.embedding_length".into(), Kv::U32(256)),
                ("qwen35.feed_forward_length".into(), Kv::U32(256)),
                ("qwen35.attention.head_count".into(), Kv::U32(4)),
                ("qwen35.attention.head_count_kv".into(), Kv::U32(2)),
                ("qwen35.ssm.conv_kernel".into(), Kv::U32(4)),
                ("qwen35.ssm.inner_size".into(), Kv::U32(32)),
                ("qwen35.ssm.state_size".into(), Kv::U32(16)),
                ("qwen35.ssm.time_step_rank".into(), Kv::U32(2)),
                ("qwen35.ssm.group_count".into(), Kv::U32(2)),
                (
                    "qwen35.rope.dimension_sections".into(),
                    Kv::Array {
                        elem: GGUF_TYPE_INT32,
                        items: TINY_QWEN35_ROPE_SECTIONS.into_iter().map(Kv::I32).collect(),
                    },
                ),
                ("qwen35.rope.freq_base".into(), Kv::F32(10_000.0)),
                (
                    "qwen35.attention.layer_norm_rms_epsilon".into(),
                    Kv::F32(1.0 / 100_000.0),
                ),
            ],
            &[
                tw(
                    "token_embd.weight",
                    GgmlType::F32,
                    vec![n_embd, TINY_N_VOCAB],
                    pack_mat(GgmlType::F32, n_embd, TINY_N_VOCAB, 1),
                ),
                tw(
                    "output_norm.weight",
                    GgmlType::F32,
                    vec![n_embd],
                    pack_vec1d(GgmlType::F32, &ones),
                ),
            ],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected linear attention refuse"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("qwen35 linear attention layer 0"),
            "error should refuse gated-delta layer: {err}"
        );
    }

    #[test]
    fn qwen2vl_missing_dimension_sections_names_key() {
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("qwen2vl".into())),
                ("qwen2vl.block_count".into(), Kv::U32(1)),
                ("qwen2vl.embedding_length".into(), Kv::U32(256)),
                ("qwen2vl.feed_forward_length".into(), Kv::U32(256)),
                ("qwen2vl.attention.head_count".into(), Kv::U32(4)),
                ("qwen2vl.attention.head_count_kv".into(), Kv::U32(2)),
                ("qwen2vl.rope.freq_base".into(), Kv::F32(10_000.0)),
                (
                    "qwen2vl.attention.layer_norm_rms_epsilon".into(),
                    Kv::F32(1.0 / 100_000.0),
                ),
            ],
            &[],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected missing sections"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("qwen2vl.rope.dimension_sections"),
            "error should name kv key: {err}"
        );
    }

    #[test]
    fn qwen3vl_missing_dimension_sections_names_key() {
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("qwen3vl".into())),
                ("qwen3vl.block_count".into(), Kv::U32(1)),
                ("qwen3vl.embedding_length".into(), Kv::U32(256)),
                ("qwen3vl.feed_forward_length".into(), Kv::U32(256)),
                ("qwen3vl.attention.head_count".into(), Kv::U32(4)),
                ("qwen3vl.attention.head_count_kv".into(), Kv::U32(2)),
                ("qwen3vl.rope.freq_base".into(), Kv::F32(10_000.0)),
                (
                    "qwen3vl.attention.layer_norm_rms_epsilon".into(),
                    Kv::F32(1.0 / 100_000.0),
                ),
            ],
            &[],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected missing sections"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("qwen3vl.rope.dimension_sections"),
            "error should name kv key: {err}"
        );
    }

    #[test]
    fn qwen3vlmoe_qwen25vl_architecture_error_names_arch() {
        for arch in ["qwen3vlmoe", "qwen25vl"] {
            let bytes = write_gguf_with_kv(
                &[
                    ("general.alignment".into(), Kv::U32(32)),
                    ("general.architecture".into(), Kv::String(arch.into())),
                ],
                &[],
            );
            let g = load_gguf(&bytes).expect("load");
            let err = match Llama::from_gguf(g) {
                Ok(_) => panic!("expected unknown arch {arch}"),
                Err(e) => e.to_string(),
            };
            assert!(err.contains(arch), "error should name arch: {err}");
            assert!(
                err.contains("unknown architecture"),
                "error should name unknown architecture: {err}"
            );
        }
    }

    #[test]
    fn mixtral_architecture_error_names_arch() {
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("mixtral".into())),
            ],
            &[],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected unknown arch"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("mixtral"), "error should name arch: {err}");
        assert!(
            err.contains("unknown architecture"),
            "error should name unknown architecture: {err}"
        );
    }

    #[test]
    fn tiny_qwen2moe_load_gemv_gemm_embed_and_greedy() {
        let bytes = tiny_qwen2moe_gguf();
        let g = load_gguf(&bytes).expect("load qwen2moe");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("qwen2moe".into()))
        );
        assert_eq!(g.kv_u32("qwen2moe.block_count"), Some(1));
        assert_eq!(g.kv_u32("qwen2moe.embedding_length"), Some(256));
        assert_eq!(g.kv_u32("qwen2moe.feed_forward_length"), Some(256));
        assert_eq!(g.kv_u32("qwen2moe.attention.head_count"), Some(4));
        assert_eq!(g.kv_u32("qwen2moe.attention.head_count_kv"), Some(2));
        assert_eq!(g.kv_u32("qwen2moe.expert_count"), Some(4));
        assert_eq!(g.kv_u32("qwen2moe.expert_used_count"), Some(2));
        assert_eq!(g.kv_u32("qwen2moe.expert_feed_forward_length"), Some(256));
        assert_eq!(
            g.kv_u32("qwen2moe.expert_shared_feed_forward_length"),
            Some(256)
        );
        assert!(g.kv_u32("llama.block_count").is_none());
        assert!(g.kv_u32("llama4.block_count").is_none());
        assert!(g.kv_u32("mixtral.block_count").is_none());
        assert!(g.kv_u32("qwen3moe.block_count").is_none());
        assert!(g.tensor("blk.0.attn_q_norm.weight").is_none());
        assert!(g.tensor("blk.0.attn_k_norm.weight").is_none());
        assert!(g.tensor("blk.0.ffn_gate.weight").is_none());
        assert!(g.tensor("blk.0.ffn_up.weight").is_none());
        assert!(g.tensor("blk.0.ffn_down.weight").is_none());
        assert!(g.tensor("blk.0.ffn_gate_inp.weight").is_some());
        assert!(g.tensor("blk.0.ffn_gate_exps.weight").is_some());
        assert!(g.tensor("blk.0.ffn_up_exps.weight").is_some());
        assert!(g.tensor("blk.0.ffn_down_exps.weight").is_some());
        assert!(g.tensor("blk.0.ffn_gate_inp_shexp.weight").is_some());
        assert!(g.tensor("blk.0.ffn_gate_shexp.weight").is_some());
        assert!(g.tensor("blk.0.ffn_up_shexp.weight").is_some());
        assert!(g.tensor("blk.0.ffn_down_shexp.weight").is_some());
        assert_eq!(
            g.tensor("blk.0.ffn_gate_exps.weight").unwrap().shape,
            &[256, 256, 4]
        );
        assert_eq!(
            g.tensor("blk.0.ffn_gate_inp_shexp.weight").unwrap().shape,
            &[256]
        );
        assert_eq!(g.kv_bool("tokenizer.ggml.add_bos_token"), Some(false));
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
        assert!(!tok.add_bos);
        let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
        let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
        assert_eq!(out, out2);
        assert!(!out.is_empty());
        let tokens = [1u32, 2, 3];
        let llama_pref = {
            let lg = load_gguf(&llama_bytes).expect("llama reload");
            let lm = Llama::from_gguf(lg).expect("lm");
            let mut c = lm.new_cache(8).expect("c");
            lm.prefill(&mut c, &tokens).expect("llama pref")
        };
        let mut qc = model.new_cache(8).expect("qc");
        let qwen2moe_pref = model.prefill(&mut qc, &tokens).expect("qwen2moe pref");
        assert_ne!(
            qwen2moe_pref, llama_pref,
            "qwen2moe softmax/shared-expert must change logits vs dense llama"
        );
        let llama_moe_pref = {
            let mg = load_gguf(&tiny_llama_moe_gguf()).expect("llama moe");
            let mm = Llama::from_gguf(mg).expect("mm");
            let mut c = mm.new_cache(8).expect("cm");
            mm.prefill(&mut c, &tokens).expect("llama moe pref")
        };
        assert_ne!(
            qwen2moe_pref, llama_moe_pref,
            "qwen2moe must not copy llama-MoE norm_w / no-shexp"
        );
        let llama4_pref = {
            let l4 = load_gguf(&tiny_llama4_gguf()).expect("llama4");
            let m4 = Llama::from_gguf(l4).expect("m4");
            let mut c = m4.new_cache(8).expect("c4");
            m4.prefill(&mut c, &tokens).expect("llama4 pref")
        };
        assert_ne!(
            qwen2moe_pref, llama4_pref,
            "qwen2moe must not copy Llama4 sigmoid / weight-before-FFN"
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
                llama_moe: false,
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
        assert!(
            err.contains("unknown architecture"),
            "error should name unknown architecture: {err}"
        );
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
