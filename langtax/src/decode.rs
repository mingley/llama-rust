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
//! Official Bloom (`architecture=bloom`) follows `src/models/bloom.cpp`:
//! `token_embd_norm` LayerNorm, fused `attn_qkv` (convert restacks HF
//! interleaved QKV to concatenated Q/K/V), `LLM_NORM` on attn/ffn/output,
//! sequential residual, `LLM_FFN_GELU`/`LLM_FFN_SEQ` with biases, ALiBi
//! (`f_max_alibi_bias = 8`, no RoPE). Not Mixtral, not `qwen3vlmoe`.
//! Official Gemma2 (`architecture=gemma2`) follows `src/models/gemma2.cpp`:
//! Gemma embed-scale + GeGLU, `post_attention_norm` / `post_ffw_norm`,
//! sliding-window attention (`set_swa_pattern(2)`, `LLAMA_SWA_TYPE_STANDARD`),
//! attn/final tanh logit softcap.
//! Official Gemma3 (`architecture=gemma3`) follows `src/models/gemma3.cpp`:
//! Gemma2 post-norms + GeGLU, QK-Norm before RoPE, no attn softcap, optional
//! final tanh softcap, SWA default period 6.
//! Official Gemma3n (`architecture=gemma3n`) follows `src/models/gemma3n.cpp`:
//! Gemma3 QK-Norm + post-norms + GeGLU, attention scale `1.0`, unweighted
//! RMSNorm on V, SWA period 5, required sliding window, final tanh softcap
//! (default 30), AltUp (4 streams) + Laurel + per-layer inputs, gaussian_topk
//! on the first 10 layers.
//! Official Gemma4 (`architecture=gemma4`) follows `src/models/gemma4.cpp`:
//! Gemma embed-scale + GeGLU, QK-Norm before RoPE, unweighted RMSNorm on V,
//! attention scale `1.0`, RMSNorm `post_attention_norm` / `ffn_norm` /
//! `post_ffw_norm`, required SWA plus convert `attention.sliding_window_pattern`
//! as a per-layer bool array, required `attention.key_length_swa` /
//! `value_length_swa` and `embedding_length_per_layer_input`. Dense layers stay
//! GeGLU. MoE layers (`ffn_gate_inp`) add a shared dense MLP plus routed GELU
//! experts (`build_moe_ffn` softmax then top-k with `norm_w` clamp `2^-14`,
//! custom router on `attn_out`). Writer-tiny dense has no `ffn_gate_inp`.
//! Writer-tiny MoE keeps the same arch. Writer-tiny dense and MoE keep
//! `embedding_length_per_layer_input = 0`. Writer-tiny PLE
//! ([`tiny_gemma4_ple_gguf`]) sets that key `> 0` and loads
//! `per_layer_token_embd` / `per_layer_model_proj` / `per_layer_proj_norm`
//! plus per-layer `inp_gate` / `proj` / `post_norm` (no AltUp / Laurel).
//! Writer-tiny MoE+PLE ([`tiny_gemma4_moe_ple_gguf`]) is the production
//! E2B/E4B shape: both `ffn_gate_inp` and `n_embd_per_layer > 0`.
//! Writer-tiny fused experts ([`tiny_gemma4_moe_fused_gguf`]) pack
//! `ffn_gate_up_exps` instead of separate gate/up. Writer-tiny fused plus
//! PLE ([`tiny_gemma4_moe_fused_ple_gguf`]) is the production E2B/E4B packing:
//! fused experts and `n_embd_per_layer > 0`. Writer-tiny dense/MoE/PLE keep
//! `attention.shared_kv_layers` unset (every layer has KV). Writer-tiny
//! shared KV ([`tiny_gemma4_shared_kv_gguf`]) is three layers, all SWA,
//! `shared_kv_layers=1`, so layer 2 reuses layer 0's KV
//! (`n_layer_kv_from_start` minus 2). Optional final tanh logit softcap.
//! Mixed SWA/global head dims stay refused with named keys.

use crate::gguf::{load_gguf_owned, GgmlType, Gguf, GgufError, Kv, Tensor, TensorWrite};
pub use crate::kv_page::PagedKvPool;
use crate::kv_page::{KvGeom, KvPages};
use crate::pool::{Pool, RowKernel};
use crate::quant::{
    bf16_row_bytes, dequant_bf16_row, dequant_f16_row, dequant_f32_row, dequant_iq1_m_row,
    dequant_iq1_s_row, dequant_iq2_s_row, dequant_iq2_xs_row, dequant_iq2_xxs_row,
    dequant_iq3_s_row, dequant_iq3_xxs_row, dequant_iq4_nl_row, dequant_iq4_xs_row,
    dequant_mxfp4_row, dequant_nvfp4_row, dequant_q1_0_row, dequant_q2_0_row, dequant_q2_k_row,
    dequant_q3_k_row, dequant_q4_0_row, dequant_q4_1_row, dequant_q4_k_row, dequant_q5_0_row,
    dequant_q5_1_row, dequant_q5_k_row, dequant_q6_k_row, dequant_q8_0_row, dequant_q8_1_row,
    dequant_tq1_0_row, dequant_tq2_0_row, f16_row_bytes, f32_row_bytes, gemm_bf16, gemm_f16,
    gemm_f32, gemm_iq1_m_f32, gemm_iq1_s_f32, gemm_iq2_s_f32, gemm_iq2_xs_f32, gemm_iq2_xxs_f32,
    gemm_iq3_s_f32, gemm_iq3_xxs_f32, gemm_iq4_nl_f32, gemm_iq4_xs_f32, gemm_mxfp4_f32,
    gemm_nvfp4_f32, gemm_q1_0_f32, gemm_q2_0_f32, gemm_q2_k_f32, gemm_q3_k_f32, gemm_q4_0_f32,
    gemm_q4_1_f32, gemm_q4_k_f32, gemm_q5_0_f32, gemm_q5_1_f32, gemm_q5_k_f32, gemm_q6_k_f32,
    gemm_q8_0_f32, gemm_q8_1_f32, gemm_tq1_0_f32, gemm_tq2_0_f32, gemv_bf16, gemv_f16, gemv_f32,
    gemv_iq1_m_f32, gemv_iq1_s_f32, gemv_iq2_s_f32, gemv_iq2_xs_f32, gemv_iq2_xxs_f32,
    gemv_iq3_s_f32, gemv_iq3_xxs_f32, gemv_iq4_nl_f32, gemv_iq4_xs_f32, gemv_mxfp4_f32,
    gemv_nvfp4_f32, gemv_q1_0_f32, gemv_q2_0_f32, gemv_q2_k_f32, gemv_q3_k_f32, gemv_q4_0_f32,
    gemv_q4_1_f32, gemv_q4_k_f32, gemv_q5_0_f32, gemv_q5_1_f32, gemv_q5_k_f32, gemv_q6_k_f32,
    gemv_q8_0_f32, gemv_q8_1_f32, gemv_tq1_0_f32, gemv_tq2_0_f32, iq1_m_row_bytes, iq1_s_row_bytes,
    iq2_s_row_bytes, iq2_xs_row_bytes, iq2_xxs_row_bytes, iq3_s_row_bytes, iq3_xxs_row_bytes,
    iq4_nl_row_bytes, iq4_xs_row_bytes, mxfp4_row_bytes, nvfp4_row_bytes, pack_bf16, pack_f16,
    pack_f32, pack_iq1_m_block, pack_iq1_s_block, pack_iq2_s_block, pack_iq2_xs_block,
    pack_iq2_xxs_block, pack_iq3_s_block, pack_iq3_xxs_block, pack_iq4_nl_block, pack_iq4_xs_block,
    pack_mxfp4_block, pack_nvfp4_block, pack_q1_0_block, pack_q2_0_block, pack_q2_k_block,
    pack_q3_k_block, pack_q4_0_block, pack_q4_1_block, pack_q4_k_block, pack_q5_0_block,
    pack_q5_1_block, pack_q5_k_block, pack_q6_k_block, pack_q8_0_block, pack_q8_1_block,
    pack_tq1_0_block, pack_tq2_0_block, q1_0_row_bytes, q2_0_row_bytes, q2_k_row_bytes,
    q3_k_row_bytes, q4_0_row_bytes, q4_1_row_bytes, q4_k_row_bytes, q5_0_row_bytes, q5_1_row_bytes,
    q5_k_row_bytes, q6_k_row_bytes, q8_0_row_bytes, q8_1_row_bytes, tq1_0_row_bytes,
    tq2_0_row_bytes, QuantError, QK1_0, QK2_0, QK4_0, QK4_1, QK4_NL, QK5_0, QK5_1, QK8_0, QK8_1,
    QK_K, QK_MXFP4, QK_NVFP4,
};
use crate::sample::{SampleError, SampleParams, Sampler};
use crate::tok::{TokError, Tokenizer};
use crate::{write_gguf_with_kv, GGUF_DEFAULT_ALIGNMENT};
use expertvm::{
    plan_keys, predicted_keys, prefix_hash, weight_permille, ChainState, DirectStore, ExpertAccess,
    ExpertKey, ExpertParts, ExpertStore, LiveStore, Markov, Plan, Prefetch, Trace,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

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
/// Writer-built tiny official Gemma4 MoE (`gemma4.expert_count`).
const TINY_GEMMA4_N_EXPERT: usize = 4;
/// Writer-built tiny `gemma4.expert_used_count`.
const TINY_GEMMA4_N_EXPERT_USED: usize = 2;
/// Writer-built tiny Gemma4 PLE `embedding_length_per_layer_input`. Distinct
/// from [`TINY_N_EMBD`] so a layout mix-up cannot silently match dense gemma4.
const TINY_GEMMA4_N_EMBD_PER_LAYER: usize = 64;
/// Writer-built tiny Gemma4 shared KV (`attention.shared_kv_layers=1`).
/// llama.cpp `create_memory` asserts `n_layer_kv_from_start >= 2`, and the SWA
/// donor is `n_from_start` minus 2, so the smallest valid fixture is 3 layers.
const TINY_GEMMA4_SHARED_N_LAYER: usize = 3;
/// Official `n_layer_kv_from_start` is `n_layer` minus `shared_kv_layers`.
const TINY_GEMMA4_SHARED_KV_LAYERS: u32 = 1;
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
/// Official convert `add_feed_forward_length(4 * n_embed)`.
const TINY_BLOOM_N_FF: usize = 1024;
/// Official convert `add_head_count_kv(n_head)` (no GQA).
const TINY_BLOOM_N_HEAD_KV: usize = TINY_N_HEAD;
/// Official `src/models/bloom.cpp` `load_arch_hparams`: hardcoded
/// `hparams.f_max_alibi_bias = 8.0f` (not a GGUF KV).
const BLOOM_MAX_ALIBI_BIAS: f32 = 8.0;
/// Official Google Gemma2 `attn_logit_softcapping` (convert + llama.cpp default).
const GEMMA2_ATTN_LOGIT_SOFTCAPPING: f32 = 50.0;
/// Official Google Gemma2 `final_logit_softcapping` (convert + llama.cpp default).
const GEMMA2_FINAL_LOGIT_SOFTCAPPING: f32 = 30.0;
/// Writer-tiny `{arch}.attention.sliding_window` so short-seq tests clip
/// (`is_masked_swa`: `p1 - p0 >= n_swa`). Official default is 4096.
const GEMMA2_TINY_N_SWA: u32 = 2;
/// Official `set_swa_pattern` default period (`dense_first=false`).
const GEMMA2_SWA_PERIOD_DEFAULT: u32 = 2;
/// Official gemma3.cpp `set_swa_pattern` default period.
const GEMMA3_SWA_PERIOD_DEFAULT: u32 = 6;
/// Official gemma3n.cpp `get_key_or_arr` default `swa_period`.
const GEMMA3N_SWA_PERIOD_DEFAULT: u32 = 5;
/// Official `llama_hparams::n_altup` / convert `altup_num_inputs`.
const GEMMA3N_N_ALTUP: usize = 4;
/// Official `llama_hparams::i_altup_act` / convert `altup_active_idx`.
const GEMMA3N_I_ALTUP_ACT: usize = 0;
/// Official `llama_hparams::laurel_rank`.
const GEMMA3N_LAUREL_RANK: usize = 64;
/// Official `llama_hparams::n_embd_altup` (equals [`TINY_N_EMBD`] on the writer-tiny).
const GEMMA3N_N_EMBD_ALTUP: usize = 256;
/// Official `models.h` `n_layer_sparsity` (not a GGUF KV).
const GEMMA3N_N_LAYER_SPARSITY: usize = 10;
/// Official `models.h` `f_sparsity_std_mul` = `Normal(0,1).icdf(0.95)`.
/// Stored as the exact `f32` of the official double `1.6448533535003662`.
const GEMMA3N_SPARSITY_STD_MUL: f32 = 1.644_853_4;
/// Official gemma3n.cpp `n_layer_kv_from_start` (convert `attention.shared_kv_layers`).
const GEMMA3N_N_LAYER_KV_FROM_START: u32 = 20;
/// Official GGUF `GGUF_TYPE_BOOL` (array elem for `attention.sliding_window_pattern`).
const GGUF_TYPE_BOOL: i32 = 7;

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
    /// Expert store (lease, unknown key, gpu-sim).
    Store(String),
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
            Self::Store(e) => write!(f, "expert store: {e}"),
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

impl From<expertvm::Error> for LlamaError {
    fn from(v: expertvm::Error) -> Self {
        Self::Store(v.to_string())
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

/// One expert part plus optional store bytes for a grouped GEMM.
struct PartBytes<'a> {
    m: &'a QuantMat,
    part: usize,
    n_tokens: usize,
    bytes: Option<&'a [u8]>,
}

/// Softmax-then-top-k routed experts (llama / Qwen2MoE / Qwen3MoE / Qwen3Next).
struct SoftmaxMoE<'a> {
    gate_inp: &'a QuantMat,
    gate: &'a QuantMat,
    up: &'a QuantMat,
    down: &'a QuantMat,
    n_expert: usize,
    n_used: usize,
    n_embd: usize,
    norm_w: bool,
    /// Llama4 applies the router weight to `x` before SwiGLU (`weight_before_ffn`).
    scale_x: bool,
    /// Official gemma4.cpp experts are `LLM_FFN_GELU`; every other softmax MoE is SILU.
    gelu: bool,
    /// Official gemma4.cpp fused `ffn_gate_up_exps` (twice `n_ff` rows, gate then up).
    fused: bool,
    err: &'static str,
}

struct MoeExec<'a> {
    pool: &'a mut GemvPool,
    moe_trace: &'a mut MoeTraceBuf,
    store: &'a mut Option<LiveStore>,
}

struct GroupedRun<'a> {
    pool: &'a mut GemvPool,
    store: &'a mut Option<LiveStore>,
    layer: u32,
    row_seq: &'a [u64],
    sequence: u64,
}

struct ExpertJob<'a> {
    expert: usize,
    toks: &'a [usize],
    layer: u32,
    row_seq: &'a [u64],
    sequence: u64,
}

struct ExpertGemm<'a> {
    expert: usize,
    n: usize,
    parts: Option<&'a ExpertParts>,
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

/// Official Gemma4 MoE (`build_moe_ffn` SOFTMAX + `norm_w` + GELU, plus dense shared MLP).
struct Gemma4Moe {
    shared: DenseFfn,
    post_norm_1: Vec<f32>,
    pre_norm_2: Vec<f32>,
    post_norm_2: Vec<f32>,
    gate_inp: QuantMat,
    gate_inp_s: Vec<f32>,
    gate_exps: Option<QuantMat>,
    up_exps: Option<QuantMat>,
    /// Official fused `ffn_gate_up_exps` (`n_embd` by twice `n_ff` by `n_expert`).
    gate_up: Option<QuantMat>,
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
/// Official bloom uses the same FFN op (`bloom.cpp` `build_ffn` GELU/SEQ).
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
    /// Official `gemma4` MoE: dense shared GeGLU plus routed GELU `*_exps`.
    Gemma4Moe(Box<Gemma4Moe>),
    /// Official `qwen3moe`: routed `*_exps`, softmax then top-k with `norm_w`.
    Qwen3Moe(Box<Qwen3Moe>),
    /// Official `qwen3next`: routed `*_exps` (`norm_w`) + gated shared `*_shexp`.
    Qwen3Next(Box<Qwen2Moe>),
    /// Official `phi2` / `bloom`: `ffn_up` / `ffn_down` GELU sequential (no `ffn_gate`).
    Phi2(Box<Phi2Ffn>),
}

/// Per-layer weights.
struct Layer {
    attn_norm: Vec<f32>,
    /// Official phi2 / bloom `blk.{i}.attn_norm.bias` (`LLM_NORM`).
    attn_norm_b: Option<Vec<f32>>,
    wq: QuantMat,
    bq: Option<Vec<f32>>,
    /// `None` on official gemma4 shared-KV layers (`!has_kv(il)`).
    wk: Option<QuantMat>,
    bk: Option<Vec<f32>>,
    /// `None` on official gemma4 shared-KV layers (`!has_kv(il)`).
    wv: Option<QuantMat>,
    bv: Option<Vec<f32>>,
    /// KV cache layer index: `il` when `has_kv`, else the official reuse donor
    /// (`n_layer_kv_from_start` minus 2 on SWA, minus 1 on global).
    kv_slot: usize,
    wo: QuantMat,
    /// Official phi2 / bloom `blk.{i}.attn_output.bias` (required, flag 0).
    wo_b: Option<Vec<f32>>,
    /// Official Qwen3 / Qwen3MoE / Qwen3VL / Qwen3Next / Qwen35 / Gemma3 / Gemma3n / Gemma4 `blk.{i}.attn_q_norm` (RMSNorm on Q after projection, before RoPE).
    attn_q_norm: Option<Vec<f32>>,
    /// Official Qwen3 / Qwen3MoE / Qwen3VL / Qwen3Next / Qwen35 / Gemma3 / Gemma3n / Gemma4 `blk.{i}.attn_k_norm` (RMSNorm on K after projection, before RoPE).
    attn_k_norm: Option<Vec<f32>>,
    /// Official Qwen3Next / Qwen35: `attn_q` is query+gate (`n_embd_head * n_head * 2`).
    attn_q_gate: bool,
    /// Official Llama4 iRoPE: skip RoPE when `(il+1) % n_no_rope_layer_step == 0`.
    use_rope: bool,
    /// Official Llama4 `Llama4TextL2Norm`: unweighted RMS after RoPE on RoPE layers.
    qk_l2: bool,
    ffn_norm: Vec<f32>,
    /// Official bloom `blk.{i}.ffn_norm.bias` (`LLM_NORM`). Phi2 has no `ffn_norm`.
    ffn_norm_b: Option<Vec<f32>>,
    /// Official gemma2 / gemma3 / gemma3n / gemma4 `blk.{i}.post_attention_norm` (RMSNorm on attn out before residual).
    attn_post_norm: Option<Vec<f32>>,
    /// Official gemma2 / gemma3 / gemma3n / gemma4 `blk.{i}.post_ffw_norm` (RMSNorm on FFN out before residual).
    ffn_post_norm: Option<Vec<f32>>,
    ffn: LayerFfn,
    /// Official gemma3n AltUp / Laurel / per-layer input tensors.
    gemma3n: Option<Gemma3nLayer>,
    /// Official gemma4.cpp per-layer embedding inject (`n_embd_per_layer > 0`).
    gemma4_ple: Option<Gemma4PleLayer>,
}

/// Official gemma4.cpp per-layer input inject weights (`inp_gate` / `proj` / `post_norm`).
struct Gemma4PleLayer {
    inp_gate: QuantMat,
    proj: QuantMat,
    post_norm: Vec<f32>,
}

/// Official gemma4.cpp model-level per-layer embedding tensors.
struct Gemma4PleWeights {
    per_layer_token_embd: QuantMat,
    per_layer_model_proj: QuantMat,
    per_layer_proj_norm: Vec<f32>,
    n_embd_per_layer: usize,
}

/// Official gemma3n.cpp per-layer AltUp, Laurel, and per-layer input weights.
struct Gemma3nLayer {
    inp_gate: QuantMat,
    proj: QuantMat,
    post_norm: Vec<f32>,
    altup_correct_coef: QuantMat,
    altup_correct_scale: Vec<f32>,
    altup_predict_coef: QuantMat,
    altup_router: QuantMat,
    altup_router_norm: Vec<f32>,
    laurel_l: QuantMat,
    laurel_r: QuantMat,
    laurel_post_norm: Vec<f32>,
}

/// Official gemma3n.cpp model-level AltUp / per-layer embedding tensors.
struct Gemma3nWeights {
    altup_proj: QuantMat,
    altup_unembd_proj: QuantMat,
    per_layer_token_embd: QuantMat,
    per_layer_model_proj: QuantMat,
    per_layer_proj_norm: Vec<f32>,
    n_altup: usize,
    i_altup_act: usize,
    n_embd_altup: usize,
    n_layer_sparsity: usize,
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
    /// Official `llama_model_rope_type` = `LLAMA_ROPE_TYPE_NEOX` for this arch.
    /// Selects the `n_dims/2` offset pairing instead of adjacent-lane NORM.
    rope_neox: bool,
    /// `1` for llama/qwen2/mistral/phi3. Gemma official walk scales embeds by `sqrt(n_embd)`.
    embed_scale: f32,
    /// Official Gemma FFN is `LLM_FFN_GELU` (GeGLU). Other loaded arches stay SwiGLU
    /// (`LLM_FFN_SILU`), including official Qwen3, Llama4, and Qwen3MoE.
    ffn_gelu: bool,
    /// The GGUF file blob. `Arc` because the GEMV pool's workers read weight
    /// rows out of it and must not borrow from a decode frame.
    blob: Arc<Vec<u8>>,
    token_embd: QuantMat,
    output_norm: Vec<f32>,
    /// Official phi2 / bloom `output_norm.bias` (`LLM_NORM`).
    output_norm_b: Option<Vec<f32>>,
    output: QuantMat,
    /// Official phi2 `output.bias` (required, flag 0). Bloom has none.
    output_b: Option<Vec<f32>>,
    /// Official `architecture=phi2` language walk (`src/models/phi2.cpp`).
    phi2: bool,
    /// Official `architecture=bloom` language walk (`src/models/bloom.cpp`).
    bloom: bool,
    /// Official bloom `token_embd_norm.weight` (`LLM_TENSOR_TOKEN_EMBD_NORM`).
    token_embd_norm: Option<Vec<f32>>,
    /// Official bloom `token_embd_norm.bias`.
    token_embd_norm_b: Option<Vec<f32>>,
    /// Official gemma2 `attn_logit_softcapping` (`0` disables).
    attn_logit_softcapping: f32,
    /// Official gemma2 `final_logit_softcapping` (`0` disables).
    final_logit_softcapping: f32,
    /// Official gemma2 `{arch}.attention.sliding_window` (`0` is full causal).
    n_swa: usize,
    /// Official gemma2 `set_swa_pattern` period (default 2; `0` means every layer).
    swa_period: u32,
    /// Official gemma4 convert `attention.sliding_window_pattern` bool array.
    /// Empty means [`gemma2_is_swa`] with [`Self::swa_period`].
    is_swa: Vec<bool>,
    /// Official `architecture=gemma4` language walk (`src/models/gemma4.cpp`):
    /// attention scale `1.0` and unweighted RMSNorm on V.
    gemma4: bool,
    /// Official `architecture=gemma3n` AltUp / Laurel / per-layer embeddings.
    gemma3n: Option<Gemma3nWeights>,
    /// Official gemma4.cpp per-layer embeddings when
    /// `embedding_length_per_layer_input > 0`.
    gemma4_ple: Option<Gemma4PleWeights>,
    layers: Vec<Layer>,
}

/// One weight matrix as the GEMV pool sees it: plain numbers, no borrows.
///
/// `gemv_*` derives its row count from `y.len()` and requires exactly
/// `row_bytes * y.len()` weight bytes, so rows `first .. last` of a matrix are
/// just the byte range `[base + first * row_bytes, base + last * row_bytes)`
/// against `y[first..last]`. No kernel signature has to change for this.
#[derive(Clone, Copy)]
struct GemvJob {
    ty: GgmlType,
    n_cols: usize,
    /// Byte offset of row 0 in the model blob.
    base: usize,
    row_bytes: usize,
}

/// GEMV rows out of the model blob, run on the pool's worker threads.
struct GemvRows {
    blob: Arc<Vec<u8>>,
}

/// The pool slot threaded through the forward pass. `&mut` so a decode step can
/// spawn the workers on first use and reuse them afterwards.
type GemvPool = Option<Pool<GemvRows>>;

impl RowKernel for GemvRows {
    type Job = GemvJob;

    fn rows(&self, job: GemvJob, first: usize, x: &[f32], y: &mut [f32]) -> bool {
        let Some(start) = first
            .checked_mul(job.row_bytes)
            .and_then(|off| off.checked_add(job.base))
        else {
            return false;
        };
        let Some(end) = y
            .len()
            .checked_mul(job.row_bytes)
            .and_then(|len| start.checked_add(len))
        else {
            return false;
        };
        let Some(w) = self.blob.get(start..end) else {
            return false;
        };
        matches!(gemv_rows(job.ty, job.n_cols, w, x, y), Ok(true))
    }
}

/// Work below which a pooled GEMV loses to a sequential one. Chosen from
/// `bench_dispatch_overhead`: the pool costs a few microseconds per call, and
/// the sequential kernels run on the order of a MAC per nanosecond per core.
const PAR_MIN_WORK: usize = 1 << 16;

#[cfg(test)]
thread_local! {
    /// Test hook overriding the size threshold: `Some(true)` sends every GEMV
    /// to the pool (the tiny fixtures are all below it), `Some(false)` sends
    /// none, which is the pre-pool behaviour and what the benchmark compares
    /// against inside one process.
    static POOL_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn with_pool_override<R>(to: Option<bool>, f: impl FnOnce() -> R) -> R {
    POOL_OVERRIDE.with(|flag| {
        let prev = flag.replace(to);
        let out = f();
        flag.set(prev);
        out
    })
}

/// Run `f` with every GEMV routed through the pool regardless of size.
#[cfg(test)]
pub(crate) fn with_forced_pool<R>(f: impl FnOnce() -> R) -> R {
    with_pool_override(Some(true), f)
}

/// Run `f` with the pool disabled, i.e. every GEMV in the calling thread.
#[cfg(test)]
pub(crate) fn without_pool<R>(f: impl FnOnce() -> R) -> R {
    with_pool_override(Some(false), f)
}

#[cfg(test)]
fn pool_override() -> Option<bool> {
    POOL_OVERRIDE.with(std::cell::Cell::get)
}

#[cfg(not(test))]
fn pool_override() -> Option<bool> {
    None
}

/// Whether an `n_rows x n_cols` GEMV is big enough to hand to the pool.
fn pooled_gemv(n_rows: usize, n_cols: usize) -> bool {
    match pool_override() {
        Some(forced) => forced && n_rows > 1,
        None => n_rows > 1 && n_rows.saturating_mul(n_cols) >= PAR_MIN_WORK,
    }
}

/// KV cache for GQA decode.
pub struct KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    /// Tokens already in the cache.
    pub n_past: usize,
    max_seq: usize,
    scratch: Scratch,
    /// GEMV workers, spawned on the first decode step that can use them and
    /// joined when the cache drops. `None` for models too small to benefit and
    /// under `pool::with_sequential`.
    pool: Option<Pool<GemvRows>>,
    moe_trace: MoeTraceBuf,
    /// Opt-in expert residency. `None` keeps the default blob GEMV path.
    expert_store: Option<LiveStore>,
    /// Longest common prefix of the last [`Llama::prompt_logits`] vs cached ids.
    last_prefix_hit: usize,
    /// Opt-in paged KV. `None` keeps the dense `max_seq` layout.
    pages: Option<KvPages>,
}

impl KvCache {
    /// Record MoE router decisions into an [`expertvm::Trace`].
    ///
    /// Off by default: the events vec stays empty and decode does not allocate
    /// for tracing. Cached token ids are kept so [`KvCache::reuse_prefix`] still
    /// sees the sequence. Enable before the first prefill of a traced run.
    pub fn enable_moe_trace(&mut self, sequence: u64) {
        self.moe_trace.enabled = true;
        self.moe_trace.sequence = sequence;
        self.moe_trace.events.clear();
        self.moe_trace.batch.clear();
    }

    /// Sequence id for batched MoE rows (Engine seq-streams). Does not enable tracing.
    pub(crate) fn set_moe_sequence(&mut self, sequence: u64) {
        self.moe_trace.sequence = sequence;
    }

    /// Take recorded router events. Empty if tracing was never enabled.
    pub fn take_moe_trace(&mut self) -> Trace {
        Trace {
            events: core::mem::take(&mut self.moe_trace.events),
        }
    }

    /// Decode expert FFN from store copies instead of the GGUF blob.
    ///
    /// Off by default so allocation-free blob decode stays unchanged.
    pub fn attach_expert_store(&mut self, store: LiveStore) {
        self.expert_store = Some(store);
    }

    /// Remove the attached store.
    pub fn take_expert_store(&mut self) -> Option<LiveStore> {
        self.expert_store.take()
    }

    /// Mutable borrow of the attached store, if any.
    pub(crate) fn expert_store_mut(&mut self) -> Option<&mut LiveStore> {
        self.expert_store.as_mut()
    }

    /// Borrow the attached store, if any.
    #[must_use]
    pub fn attached_store(&self) -> Option<&LiveStore> {
        self.expert_store.as_ref()
    }

    /// Counters from the attached store, if any.
    #[must_use]
    pub fn expert_store_metrics(&self) -> Option<expertvm::StoreMetrics> {
        self.expert_store.as_ref().map(ExpertStore::metrics)
    }

    /// Token ids occupying KV slots `0 .. n_past`.
    #[must_use]
    pub fn cached_ids(&self) -> &[u32] {
        &self.moe_trace.ids
    }

    /// KV capacity in tokens (the `max_seq` stride).
    #[must_use]
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Longest common prefix reused by the last [`Llama::prompt_logits`].
    ///
    /// Zero until a prompt call. A full-prefix hit still recomputes the last
    /// prompt token so the returned logits belong to that token.
    #[must_use]
    pub fn last_prefix_hit(&self) -> usize {
        self.last_prefix_hit
    }

    /// Rewind `n_past` to the longest common prefix of `tokens` and the cached
    /// ids. Positions `0..lcp` stay valid (head-major KV). Markov lookback is
    /// cleared; the tables stay. Does **not** run inside [`Llama::prefill_logits`]
    /// (`forward` is that path and would otherwise LCP to 0).
    pub fn reuse_prefix(&mut self, tokens: &[u32]) -> usize {
        if self.moe_trace.ids.len() != self.n_past {
            self.rewind(0);
            return 0;
        }
        let lcp = self
            .moe_trace
            .ids
            .iter()
            .zip(tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();
        self.rewind(lcp);
        lcp
    }

    fn rewind(&mut self, n: usize) {
        let n = n.min(self.n_past).min(self.moe_trace.ids.len());
        self.n_past = n;
        self.moe_trace.ids.truncate(n);
        self.moe_trace.batch.clear();
        self.moe_trace.row_seq.clear();
        self.moe_trace.row_tok.clear();
        self.moe_trace.row_prefix.clear();
        let tok_u = u32::try_from(n).unwrap_or(u32::MAX);
        self.moe_trace.events.retain(|e| e.token < tok_u);
        self.moe_trace.chain.clear();
        if let Some(p) = self.pages.as_mut() {
            p.rewind_tokens(n);
        }
    }

    /// Drop this sequence's KV blocks so another sequence can allocate.
    ///
    /// Interned prefixes stay in the pool (`refs` may remain 1). The engine
    /// re-prefills and replays already sampled tokens (greedy recompute).
    pub fn preempt(&mut self) {
        self.rewind(0);
    }

    /// Bind interned prefixes / LCP, then return the suffix still to forward.
    ///
    /// A full-prefix hit rewinds one token so the caller recomputes the last
    /// prompt token. `max_suffix` caps newly forwarded tokens (`0` = the rest).
    /// Used by [`Llama::prompt_chunk`] and Engine batched prefill.
    pub(crate) fn prompt_suffix<'a>(
        &mut self,
        tokens: &'a [u32],
        max_suffix: usize,
    ) -> Result<&'a [u32], LlamaError> {
        if tokens.is_empty() {
            return Err(LlamaError::Shape("prefill tokens".into()));
        }
        let lcp = self.reuse_prefix(tokens);
        let mut hit = lcp;
        let bound = if let Some(p) = self.pages.as_mut() {
            p.bind_full_prefix(tokens, self.n_past)
        } else {
            lcp
        };
        if bound > self.n_past {
            if let Some(extra) = tokens.get(self.n_past..bound) {
                self.moe_trace.ids.extend_from_slice(extra);
            }
            self.n_past = bound;
            hit = bound;
        }
        self.last_prefix_hit = hit;
        if hit == tokens.len() {
            let keep = tokens.len().saturating_sub(1);
            self.rewind(keep);
            return tokens
                .get(keep..)
                .ok_or_else(|| LlamaError::Shape("prefill tokens".into()));
        }
        let suffix = tokens
            .get(hit..)
            .ok_or_else(|| LlamaError::Shape("prefill tokens".into()))?;
        let cap = if max_suffix == 0 {
            suffix.len()
        } else {
            suffix.len().min(max_suffix)
        };
        suffix
            .get(..cap)
            .ok_or_else(|| LlamaError::Shape("prefill tokens".into()))
    }

    /// Ensure `n_past + extra` KV positions exist so a later write cannot fail
    /// the page cap. Paged block allocation is idempotent: a second call with
    /// the same `extra` before advancing `n_past` does not take another block.
    /// Dense caches only check `n_past + extra <= max_seq`.
    pub fn prepare_append(&mut self, extra: usize) -> Result<(), LlamaError> {
        let end = self
            .n_past
            .checked_add(extra)
            .ok_or_else(|| LlamaError::Shape("kv cache full".into()))?;
        if extra == 0 {
            return Ok(());
        }
        if end > self.max_seq {
            return Err(LlamaError::Shape("kv cache full".into()));
        }
        if let Some(p) = self.pages.as_mut() {
            for pos in self.n_past..end {
                p.ensure_write(pos)
                    .map_err(|e| LlamaError::Shape(e.into()))?;
            }
        }
        Ok(())
    }

    /// Physical blocks on this sequence's page table (`0` when dense).
    #[must_use]
    pub fn page_table_len(&self) -> usize {
        self.pages
            .as_ref()
            .map(|p| p.table_ids().len())
            .unwrap_or(0)
    }

    /// Paged block size when this cache uses [`KvPages`].
    #[must_use]
    pub fn page_size(&self) -> Option<usize> {
        self.pages.as_ref().map(KvPages::block_size)
    }

    /// Interned-block hits on this cache's pool (`0` when dense).
    #[must_use]
    pub fn page_hits(&self) -> u64 {
        self.pages.as_ref().map_or(0, KvPages::hits)
    }

    /// Physical blocks with a positive refcount (`0` when dense).
    #[must_use]
    pub fn page_occupied(&self) -> usize {
        self.pages.as_ref().map_or(0, KvPages::occupied)
    }
}

/// Opt-in MoE access log. Events allocate only when enabled. Predictor prefetch
/// (`Prefetch::Both` by default) runs when a store is attached.
#[derive(Default)]
struct MoeTraceBuf {
    enabled: bool,
    sequence: u64,
    layer: u32,
    token0: u32,
    events: Vec<ExpertAccess>,
    /// Token ids from completed forwards (the prefix so far).
    ids: Vec<u32>,
    /// Token ids of the in-flight forward.
    batch: Vec<u32>,
    /// Per-row sequence id for a batched GEMM (`empty` = [`Self::sequence`]).
    row_seq: Vec<u64>,
    /// Per-row token index for a batched GEMM (`empty` = `token0 + off`).
    row_tok: Vec<u32>,
    /// Per-row prefix hash for a batched GEMM (`empty` = [`Self::prefix_at`]).
    row_prefix: Vec<u64>,
    markov: Markov,
    chain: ChainState,
    policy: PlanPolicy,
}

/// Engine/decode predictor knobs. Default is copy-forward ∪ lookback-2, ungated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PlanPolicy {
    prefetch: Prefetch,
    plan_window: usize,
    plan_threshold: u32,
}

impl Default for PlanPolicy {
    fn default() -> Self {
        Self {
            prefetch: Prefetch::Both,
            plan_window: 0,
            plan_threshold: 500,
        }
    }
}

/// Unique predicted keys, at most `n` (order preserved).
fn unique_prefix(keys: &[ExpertKey], n: usize) -> Vec<ExpertKey> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for k in keys {
        if !seen.insert(*k) {
            continue;
        }
        out.push(*k);
        if out.len() >= n {
            break;
        }
    }
    out
}

/// Online Markov + last two router events, parked on Engine across GEMMs.
#[derive(Default)]
pub(crate) struct PrefetchChain {
    markov: Markov,
    chain: ChainState,
    policy: PlanPolicy,
}

impl PrefetchChain {
    /// Engine serving policy. Default decode is [`Prefetch::Both`], ungated.
    pub(crate) fn with_policy(prefetch: Prefetch, plan_window: usize, plan_threshold: u32) -> Self {
        Self {
            policy: PlanPolicy {
                prefetch,
                plan_window,
                plan_threshold,
            },
            ..Self::default()
        }
    }

    /// Last selected experts union the configured predictor (no JSONL future leak).
    pub(crate) fn keep_hot_keys(&self) -> Vec<ExpertKey> {
        let Some(prev) = self.chain.last_event() else {
            return Vec::new();
        };
        let mut out = prev.keys();
        let extra = predicted_keys(
            self.policy.prefetch,
            &self.markov,
            self.chain.last_pred(),
            &out,
        );
        for k in extra {
            if !out.contains(&k) {
                out.push(k);
            }
        }
        out
    }

    /// Selected-expert count of the last parked router event (`1` if none).
    pub(crate) fn last_fan_in(&self) -> u64 {
        let n = self.chain.last_event().map_or(0, |e| e.experts.len());
        u64::try_from(n).unwrap_or(1).max(1)
    }
}

impl KvCache {
    /// Install Engine-level Markov prefetch state onto this cache.
    pub(crate) fn set_prefetch_chain(&mut self, chain: PrefetchChain) {
        self.moe_trace.markov = chain.markov;
        self.moe_trace.chain = chain.chain;
        self.moe_trace.policy = chain.policy;
    }

    /// Take Markov prefetch state so Engine can park it across GEMMs.
    pub(crate) fn take_prefetch_chain(&mut self) -> PrefetchChain {
        PrefetchChain {
            markov: core::mem::take(&mut self.moe_trace.markov),
            chain: core::mem::take(&mut self.moe_trace.chain),
            policy: self.moe_trace.policy,
        }
    }
}

impl MoeTraceBuf {
    fn seq_of(&self, token_off: usize) -> u64 {
        self.row_seq
            .get(token_off)
            .copied()
            .unwrap_or(self.sequence)
    }

    fn event(&self, token_off: usize, experts: &[usize], weights: &[f32]) -> ExpertAccess {
        let token = self.row_tok.get(token_off).copied().unwrap_or_else(|| {
            self.token0
                .saturating_add(u32::try_from(token_off).unwrap_or(u32::MAX))
        });
        let sequence = self.seq_of(token_off);
        let prefix = self
            .row_prefix
            .get(token_off)
            .copied()
            .unwrap_or_else(|| self.prefix_at(token_off));
        let mut ids = Vec::new();
        let mut weight_pt = Vec::new();
        for (i, e) in experts.iter().enumerate() {
            ids.push(u32::try_from(*e).unwrap_or(u32::MAX));
            if !weights.is_empty() {
                let r = weights.get(i).copied().unwrap_or(0.0);
                weight_pt.push(weight_permille(r));
            }
        }
        ExpertAccess {
            sequence,
            token,
            layer: self.layer,
            experts: ids,
            weight_pt,
            prefix: Some(prefix),
        }
    }

    fn prefix_at(&self, token_off: usize) -> u64 {
        let take = token_off.saturating_add(1);
        let mut buf = Vec::with_capacity(self.ids.len().saturating_add(take));
        buf.extend_from_slice(&self.ids);
        buf.extend(self.batch.iter().copied().take(take));
        prefix_hash(&buf)
    }

    fn record(&mut self, token_off: usize, experts: &[usize], weights: &[f32]) {
        if !self.enabled {
            return;
        }
        self.events.push(self.event(token_off, experts, weights));
    }

    fn prefetch_experts(
        &mut self,
        store: &mut Option<LiveStore>,
        token_off: usize,
        selected: &[usize],
    ) {
        let Some(store) = store.as_mut() else {
            return;
        };
        let mut keys = Vec::new();
        for e in selected {
            if let Ok(ex) = u32::try_from(*e) {
                keys.push(ExpertKey::new(self.layer, ex));
            }
        }
        let ev = self.event(token_off, selected, &[]);
        let planned = predicted_keys(
            self.policy.prefetch,
            &self.markov,
            self.chain.predecessor(&ev),
            &keys,
        );
        if self.want_prefetch(store, &planned) {
            match store.prefetch(&planned) {
                Ok(_n) => {}
                Err(_e) => {}
            }
        }
        self.chain.observe(&mut self.markov, &ev);
    }

    fn want_prefetch(&self, store: &LiveStore, predicted: &[ExpertKey]) -> bool {
        if self.policy.plan_window == 0 {
            return true;
        }
        let upcoming = unique_prefix(predicted, self.policy.plan_window);
        let resident: BTreeSet<ExpertKey> = upcoming
            .iter()
            .copied()
            .filter(|k| store.is_resident(*k))
            .collect();
        !matches!(
            plan_keys(&resident, &upcoming, self.policy.plan_threshold),
            Plan::Stay
        )
    }
}

/// Working buffers for one forward pass, reused across decode steps.
///
/// Every buffer starts empty and is grown by [`fit`] on first use, so the first
/// `prefill` / `forward` on a fresh cache allocates and every later step of the
/// same or smaller token count allocates nothing. Buffers do not shrink: a
/// 512-token prefill leaves 512-token capacity behind, which is what the
/// following single-token steps then reuse.
///
/// The dense and official MoE expert FFNs run entirely out of these buffers.
/// A one-token decode after warmup allocates nothing on that path.
#[derive(Default)]
struct Scratch {
    /// Layer activations, `n_tokens * n_embd`. Normed in place per sublayer.
    x: Vec<f32>,
    /// `x` as it was before the current sublayer, `n_tokens * n_embd`.
    residual: Vec<f32>,
    /// Q / K / V projections, `n_tokens * w{q,k,v}.n_rows`.
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    /// Official Qwen3Next attention gate split out of the joint Q projection,
    /// `n_tokens * n_head * n_embd_head`. Empty for every other architecture.
    q_gate: Vec<f32>,
    /// Concatenated per-head attention output, `n_tokens * n_embd`.
    attn: Vec<f32>,
    /// `attn_output` projection of `attn`, `n_tokens * n_embd`.
    attn_proj: Vec<f32>,
    /// FFN gate (holds the SwiGLU/GeGLU product after activation) and up,
    /// `n_tokens * n_ff`.
    gate: Vec<f32>,
    up: Vec<f32>,
    /// `ffn_down` output, or the MoE sum, `n_tokens * n_embd`.
    ffn_out: Vec<f32>,
    /// Softmaxed attention scores for one head, `n_past + 1`.
    scores: Vec<f32>,
    /// `output_norm` of the last token, `n_embd`.
    xn: Vec<f32>,
    /// Returned logits, `n_vocab`.
    logits: Vec<f32>,
    moe: MoeScratch,
    /// Official gemma3n AltUp / Laurel / per-layer working buffers.
    g3n: Gemma3nScratch,
}

/// Working buffers for official gemma3n.cpp AltUp, Laurel, and per-layer inputs.
#[derive(Default)]
struct Gemma3nScratch {
    /// Stacked residual streams, `n_altup * n_tokens * n_embd`.
    streams: Vec<f32>,
    /// AltUp predictions, same layout as `streams`.
    pred: Vec<f32>,
    /// Per-layer inputs, token-major `[token][layer][n_embd_altup]`.
    per_layer: Vec<f32>,
    /// Laurel residual, `n_tokens * n_embd`.
    laurel: Vec<f32>,
    /// Router modalities, `n_tokens * n_altup`.
    router: Vec<f32>,
    /// Predict mix coefficients, `n_tokens * n_altup * n_altup`.
    coefs: Vec<f32>,
    /// Attn+Laurel / extra-stream / first-prediction workspace, `n_tokens * n_embd`.
    tmp: Vec<f32>,
    /// One per-layer token embedding row, `n_embd_altup * n_layer`.
    per_tok: Vec<f32>,
}

/// Working buffers for MoE expert FFNs (Llama4 and the other official walks).
#[derive(Default)]
struct MoeScratch {
    /// Router logits, `n_tokens * n_expert` after a batched router GEMM.
    logits: Vec<f32>,
    /// Router top-k expert indices, `n_expert_used`.
    order: Vec<usize>,
    /// Selected router weights after optional `norm_w`, `n_expert_used`.
    weights: Vec<f32>,
    /// Shared-expert gate per token (`ffn_gate_inp_shexp`), `n_tokens`.
    shexp_gate: Vec<f32>,
    /// One routed expert's gate / up / down, `n_ff_exp` and `n_embd`.
    g: Vec<f32>,
    u: Vec<f32>,
    y: Vec<f32>,
    /// Flattened top-k expert ids, `n_tokens * n_used`.
    sel_e: Vec<usize>,
    /// Flattened top-k weights, `n_tokens * n_used`.
    sel_w: Vec<f32>,
    /// Token indices that selected each expert this layer.
    buckets: Vec<Vec<usize>>,
    /// Packed token rows for a grouped expert GEMM, `n_group * n_embd`.
    pack_x: Vec<f32>,
}

impl Scratch {
    /// Heap address and capacity of every buffer.
    ///
    /// A reallocation moves the address, changes the capacity, or both, so
    /// comparing this across decode steps proves the steady state reuses its
    /// buffers. `#![forbid(unsafe_code)]` rules out a `GlobalAlloc` counter
    /// (`GlobalAlloc` is an unsafe trait), so this is the check that stands in
    /// for one.
    #[cfg(test)]
    fn buffer_ids(&self) -> Vec<(usize, usize)> {
        let f32s = [
            &self.x,
            &self.residual,
            &self.q,
            &self.k,
            &self.v,
            &self.attn,
            &self.attn_proj,
            &self.gate,
            &self.up,
            &self.ffn_out,
            &self.scores,
            &self.xn,
            &self.logits,
            &self.g3n.streams,
            &self.g3n.pred,
            &self.g3n.per_layer,
            &self.g3n.laurel,
            &self.g3n.router,
            &self.g3n.coefs,
            &self.g3n.tmp,
            &self.g3n.per_tok,
            &self.moe.logits,
            &self.moe.weights,
            &self.moe.shexp_gate,
            &self.moe.g,
            &self.moe.u,
            &self.moe.y,
            &self.moe.sel_w,
            &self.moe.pack_x,
        ];
        let mut out: Vec<(usize, usize)> = f32s
            .iter()
            .map(|b| (b.as_ptr() as usize, b.capacity()))
            .collect();
        out.push((self.moe.order.as_ptr() as usize, self.moe.order.capacity()));
        out.push((self.moe.sel_e.as_ptr() as usize, self.moe.sel_e.capacity()));
        out
    }
}

/// Paged block table or dense `max_seq` stride for one sequence in a transformer walk.
struct KvAddr {
    table: Vec<u32>,
    block_size: usize,
    n_layers: usize,
    dense_max: usize,
}

impl KvAddr {
    fn geom(&self, n_head_kv: usize, hd: usize) -> KvGeom<'_> {
        if self.dense_max > 0 {
            KvGeom::dense(n_head_kv, hd, self.dense_max)
        } else {
            KvGeom {
                n_head_kv,
                hd,
                n_layers: self.n_layers,
                time_stride: self.block_size,
                table: Some(self.table.as_slice()),
            }
        }
    }
}

fn kv_addr(addrs: &[KvAddr], t: usize) -> Result<&KvAddr, LlamaError> {
    if addrs.len() == 1 {
        addrs
            .first()
            .ok_or_else(|| LlamaError::Shape("kv addr".into()))
    } else {
        addrs
            .get(t)
            .ok_or_else(|| LlamaError::Shape("kv addr".into()))
    }
}

fn token_pos(pos: &[usize], t: usize) -> Result<usize, LlamaError> {
    pos.get(t)
        .copied()
        .ok_or_else(|| LlamaError::Shape("kv pos".into()))
}

fn batch_kv_layout(
    caches: &[&mut KvCache],
    groups: &[&[u32]],
) -> Result<(Vec<KvAddr>, Vec<usize>, usize), LlamaError> {
    let mut addrs = Vec::new();
    let mut positions = Vec::new();
    let mut max_seq = 0usize;
    for (cache, group) in caches.iter().zip(groups.iter()) {
        max_seq = max_seq.max(cache.max_seq);
        let Some(p) = cache.pages.as_ref() else {
            return Err(LlamaError::Shape("kv page".into()));
        };
        let table = p.table_ids().to_vec();
        let block_size = p.block_size();
        let n_layers = p.n_layers();
        let n0 = cache.n_past;
        for t in 0..group.len() {
            positions.push(n0.saturating_add(t));
            addrs.push(KvAddr {
                table: table.clone(),
                block_size,
                n_layers,
                dense_max: 0,
            });
        }
    }
    Ok((addrs, positions, max_seq))
}

fn last_group_logits(
    logits: &[f32],
    n_vocab: usize,
    groups: &[&[u32]],
) -> Result<Vec<Vec<f32>>, LlamaError> {
    let mut out = Vec::new();
    let mut start = 0usize;
    for group in groups {
        if group.is_empty() {
            return Err(LlamaError::Shape("prefill tokens".into()));
        }
        let last = start.saturating_add(group.len().saturating_sub(1));
        let off = last.saturating_mul(n_vocab);
        let row = logits
            .get(off..off.saturating_add(n_vocab))
            .ok_or_else(|| LlamaError::Shape("forward batch logits".into()))?;
        out.push(row.to_vec());
        start = start.saturating_add(group.len());
    }
    Ok(out)
}

fn restore_batch_store(caches: &mut [&mut KvCache], idx: Option<usize>, store: Option<LiveStore>) {
    if let Some(i) = idx {
        if let Some(c) = caches.get_mut(i) {
            c.expert_store = store;
        }
    }
}

fn collect_batch_trace_rows(
    caches: &[&mut KvCache],
    groups: &[&[u32]],
) -> (Vec<u64>, Vec<u32>, Vec<u64>) {
    let mut row_seq = Vec::new();
    let mut row_tok = Vec::new();
    let mut row_prefix = Vec::new();
    for (cache, group) in caches.iter().zip(groups.iter()) {
        let sequence = cache.moe_trace.sequence;
        let t0 = u32::try_from(cache.n_past).unwrap_or(u32::MAX);
        for t in 0..group.len() {
            row_seq.push(sequence);
            row_tok.push(t0.saturating_add(u32::try_from(t).unwrap_or(u32::MAX)));
            let take = t.saturating_add(1);
            let mut buf = Vec::with_capacity(cache.moe_trace.ids.len().saturating_add(take));
            buf.extend_from_slice(&cache.moe_trace.ids);
            if let Some(part) = group.get(..take) {
                buf.extend_from_slice(part);
            }
            row_prefix.push(prefix_hash(&buf));
        }
    }
    (row_seq, row_tok, row_prefix)
}

fn apply_batch_trace_rows(
    caches: &mut [&mut KvCache],
    rows: (Vec<u64>, Vec<u32>, Vec<u64>),
    flat: &[u32],
    any_trace: bool,
) -> Result<(), LlamaError> {
    let first = caches
        .first_mut()
        .ok_or_else(|| LlamaError::Shape("empty batch".into()))?;
    first.moe_trace.token0 = u32::try_from(first.n_past).unwrap_or(u32::MAX);
    first.moe_trace.batch.clear();
    first.moe_trace.batch.extend(flat.iter().copied());
    first.moe_trace.row_seq = rows.0;
    first.moe_trace.row_tok = rows.1;
    first.moe_trace.row_prefix = rows.2;
    if any_trace {
        first.moe_trace.enabled = true;
    }
    Ok(())
}

fn finish_batch_trace(caches: &mut [&mut KvCache], enabled: &[bool], first_was: bool) {
    scatter_trace_events(caches, enabled);
    if let Some(first) = caches.first_mut() {
        first.moe_trace.enabled = first_was;
        first.moe_trace.batch.clear();
        first.moe_trace.row_seq.clear();
        first.moe_trace.row_tok.clear();
        first.moe_trace.row_prefix.clear();
    }
}

fn scatter_trace_events(caches: &mut [&mut KvCache], enabled: &[bool]) {
    let events = {
        let Some(first) = caches.first_mut() else {
            return;
        };
        core::mem::take(&mut first.moe_trace.events)
    };
    for e in events {
        let Some(i) = caches
            .iter()
            .position(|c| c.moe_trace.sequence == e.sequence)
        else {
            continue;
        };
        if !enabled.get(i).copied().unwrap_or(false) {
            continue;
        }
        if let Some(c) = caches.get_mut(i) {
            c.moe_trace.events.push(e);
        }
    }
}

enum LogitsKind {
    Last,
    All,
}

struct TransformerRun<'a> {
    s: &'a mut Scratch,
    pool: &'a mut GemvPool,
    moe_trace: &'a mut MoeTraceBuf,
    expert_store: &'a mut Option<LiveStore>,
    cache_k: &'a mut [f32],
    cache_v: &'a mut [f32],
    n: usize,
    width: usize,
    hd: usize,
}

/// Resize `buf` to `len` zeros. Does not allocate once capacity is reached,
/// which is what makes a steady-state decode step allocation-free.
fn fit(buf: &mut Vec<f32>, len: usize) {
    buf.clear();
    buf.resize(len, 0.0);
}

/// `fit` plus a copy, for the residual and the last-token norm input.
fn copy_buf(dst: &mut Vec<f32>, src: &[f32]) {
    dst.clear();
    dst.extend_from_slice(src);
}

/// Size official gemma4.cpp per-layer scratch for `n` tokens.
fn gemma4_ple_fit(s: &mut Scratch, n_pl: usize, n: usize, n_layer: usize, n_embd: usize) {
    let pl = n.saturating_mul(n_pl.saturating_mul(n_layer));
    let tmp_w = n_embd.max(n_pl.saturating_mul(n_layer));
    fit(&mut s.g3n.per_layer, pl);
    fit(&mut s.g3n.tmp, n.saturating_mul(tmp_w));
    fit(&mut s.g3n.per_tok, n_pl.saturating_mul(n_layer));
}

/// Size official gemma3n AltUp / Laurel / per-layer scratch for `n` tokens.
fn gemma3n_fit(s: &mut Scratch, w: &Gemma3nWeights, n: usize, n_embd: usize, n_layer: usize) {
    let width = n.saturating_mul(n_embd);
    let n_altup = w.n_altup;
    let n_ea = w.n_embd_altup;
    let stacked = n_altup.saturating_mul(width);
    let pl = n.saturating_mul(n_ea.saturating_mul(n_layer));
    let tmp_w = n_embd.max(n_ea.saturating_mul(n_layer));
    fit(&mut s.g3n.streams, stacked);
    fit(&mut s.g3n.pred, stacked);
    fit(&mut s.g3n.per_layer, pl);
    fit(&mut s.g3n.laurel, width);
    fit(&mut s.g3n.router, n.saturating_mul(n_altup));
    fit(
        &mut s.g3n.coefs,
        n.saturating_mul(n_altup.saturating_mul(n_altup)),
    );
    fit(&mut s.g3n.tmp, n.saturating_mul(tmp_w));
    fit(&mut s.g3n.per_tok, n_ea.saturating_mul(n_layer));
}

impl Llama {
    /// Build from a loaded GGUF using `{arch}.*` KV (`llama`, `qwen2`, `mistral`,
    /// `phi3`, `gemma`, `gemma2`, `gemma3`, `gemma3n`, `gemma4`, `qwen3`, `llama4`, `qwen2moe`, `qwen3moe`, `qwen2vl`,
    /// `qwen3vl`, `qwen3next`, `qwen35`, `phi2`, or `bloom`) and `blk.{i}.*` tensor names. Official llama
    /// MoE is still `architecture=llama` with `n_expert>0`. Official Qwen2VL is
    /// Qwen2 plus m-RoPE (`LLAMA_ROPE_TYPE_MROPE`). Official Qwen3VL is Qwen3
    /// QK-Norm plus interleaved m-RoPE (`LLAMA_ROPE_TYPE_IMROPE`). Official
    /// Qwen3Next is gated full attention plus MoE (`norm_w`) and a sigmoid-gated
    /// shared expert. Official Qwen35 is gated full attention plus IMROPE and
    /// dense SwiGLU; linear-attn / gated-delta layers are refused. Official
    /// Phi2 is LayerNorm + NEOX RoPE + parallel GELU FFN. Official Bloom is
    /// `token_embd_norm` + fused QKV + ALiBi + sequential GELU FFN. Official
    /// Gemma2 is Gemma embed-scale + GeGLU plus post-norms, SWA, and tanh
    /// softcap. Official Gemma3 is Gemma2 post-norms + GeGLU plus QK-Norm,
    /// no attn softcap, optional final softcap. Official Gemma3n is Gemma3
    /// plus attention scale `1.0`, unweighted V RMSNorm, AltUp, Laurel,
    /// per-layer inputs, gaussian_topk, SWA period 5, and final softcap 30.
    /// Official Gemma4 is Gemma3 QK-Norm plus post-norms plus GeGLU, attention
    /// scale `1.0`, unweighted V RMSNorm, convert bool-array SWA pattern, and
    /// no AltUp / Laurel / gaussian_topk. Dense layers stay GeGLU. MoE layers
    /// (`ffn_gate_inp`) add a shared dense MLP plus routed GELU experts.
    /// Writer-tiny dense has no `ffn_gate_inp`. Writer-tiny MoE is the same
    /// arch. Writer-tiny PLE is the same arch with
    /// `embedding_length_per_layer_input > 0`. Writer-tiny MoE+PLE is the
    /// production E2B/E4B shape on the same arch. Writer-tiny fused experts
    /// pack `ffn_gate_up_exps`. Writer-tiny fused plus PLE packs both.
    /// Shared KV, mixed SWA/global
    /// head dims stay refused.
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
        // Official phi2.cpp / bloom.cpp `load_arch_hparams` reads
        // `LLM_KV_ATTENTION_LAYERNORM_EPS` (`{arch}.attention.layer_norm_epsilon`)
        // with the other hparams, before tensor materialization.
        let phi2 = arch == "phi2";
        let bloom = arch == "bloom";
        let gemma2 = arch == "gemma2";
        let gemma3 = arch == "gemma3";
        let gemma3n = arch == "gemma3n";
        let gemma4 = arch == "gemma4";
        let rms_eps = if phi2 || bloom {
            require_f32(&g, arch, "attention.layer_norm_epsilon")?
        } else if gemma2 || gemma3 || gemma3n || gemma4 {
            // Official gemma2.cpp / gemma3.cpp / gemma3n.cpp / gemma4.cpp `get_key` without `false`.
            require_f32(&g, arch, "attention.layer_norm_rms_epsilon")?
        } else {
            arch_f32(&g, arch, "attention.layer_norm_rms_epsilon").unwrap_or(1e-5)
        };
        // Official gemma2.cpp: convert always writes softcap / SWA; require so a
        // writer-tiny cannot silently drop them. Official gemma3.cpp: attn
        // softcap is gone; SWA is optional (`n_swa==0` is `LLAMA_SWA_TYPE_NONE`);
        // final softcap defaults to 0. Official gemma3n.cpp: SWA is required;
        // final softcap defaults to 30; attn scale is 1.0 (not a KV).
        let (attn_logit_softcapping, final_logit_softcapping, n_swa, swa_period) = if gemma2 {
            (
                require_f32(&g, arch, "attn_logit_softcapping")?,
                require_f32(&g, arch, "final_logit_softcapping")?,
                require_usize(&g, arch, "attention.sliding_window")?,
                g.kv_u32(&arch_key(arch, "attention.sliding_window_pattern"))
                    .unwrap_or(GEMMA2_SWA_PERIOD_DEFAULT),
            )
        } else if gemma3 {
            (
                0.0,
                arch_f32(&g, arch, "final_logit_softcapping").unwrap_or(0.0),
                g.kv_u32(&arch_key(arch, "attention.sliding_window"))
                    .map_or(0, |v| usize::try_from(v).unwrap_or(0)),
                g.kv_u32(&arch_key(arch, "attention.sliding_window_pattern"))
                    .unwrap_or(GEMMA3_SWA_PERIOD_DEFAULT),
            )
        } else if gemma3n {
            (
                0.0,
                arch_f32(&g, arch, "final_logit_softcapping")
                    .unwrap_or(GEMMA2_FINAL_LOGIT_SOFTCAPPING),
                require_usize(&g, arch, "attention.sliding_window")?,
                g.kv_u32(&arch_key(arch, "attention.sliding_window_pattern"))
                    .unwrap_or(GEMMA3N_SWA_PERIOD_DEFAULT),
            )
        } else if gemma4 {
            // Official gemma4.cpp: SWA required; pattern is convert bool[] via
            // `get_key_or_arr`; final softcap optional (default 0); attn scale
            // 1.0 is not a KV. `swa_period` stays 0; [`Llama::is_swa`] holds
            // the per-layer flags.
            (
                0.0,
                arch_f32(&g, arch, "final_logit_softcapping").unwrap_or(0.0),
                require_usize(&g, arch, "attention.sliding_window")?,
                0,
            )
        } else {
            (0.0, 0.0, 0, 0)
        };
        let is_swa = if gemma4 {
            load_gemma4_is_swa(&g, arch, n_layer)?
        } else {
            Vec::new()
        };
        let gemma4_n_pl;
        let n_layer_kv_from_start;
        if gemma4 {
            let h = load_gemma4_hparams(&g, arch, n_embd, n_head, n_layer)?;
            gemma4_n_pl = h.0;
            n_layer_kv_from_start = h.1;
        } else {
            gemma4_n_pl = 0;
            n_layer_kv_from_start = i32::try_from(n_layer).unwrap_or(i32::MAX);
        }
        let n_rot = rope_dimension(&g, arch, n_embd, n_head)?;
        let rope_sections = if arch == "qwen2vl" || arch == "qwen3vl" || arch == "qwen35" {
            Some(load_rope_dimension_sections(&g, arch)?)
        } else {
            None
        };
        let rope_imrope = arch == "qwen3vl" || arch == "qwen35";
        let rope_neox = rope_is_neox(arch);
        let n_vocab = g
            .tensor("token_embd.weight")
            .map(|t| t.n_rows())
            .ok_or_else(|| LlamaError::Tensor("token_embd.weight".into()))?;
        let rope_base = arch_f32(&g, arch, "rope.freq_base").unwrap_or(10_000.0);
        let gemma = arch == "gemma" || gemma2 || gemma3 || gemma3n || gemma4;
        let n_embd_f = f32::from(u16::try_from(n_embd).unwrap_or(1));
        let embed_scale = if gemma { n_embd_f.sqrt() } else { 1.0 };
        let token_embd = quant_mat(need(&g, "token_embd.weight")?)?;
        let output_norm = f32s(need(&g, "output_norm.weight")?)?;
        let output_norm_b = if phi2 || bloom {
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
        let token_embd_norm = if bloom {
            Some(f32s(need(&g, "token_embd_norm.weight")?)?)
        } else {
            None
        };
        let token_embd_norm_b = if bloom {
            Some(f32s(need(&g, "token_embd_norm.bias")?)?)
        } else {
            None
        };
        let qk_norm = arch == "qwen3"
            || arch == "qwen3moe"
            || arch == "qwen3vl"
            || arch == "qwen3next"
            || arch == "qwen35"
            || gemma3
            || gemma3n
            || gemma4;
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
        let gemma4_moe_hparams = if gemma4 {
            load_gemma4_moe_hparams(&g, arch)?
        } else {
            None
        };
        let gemma4_ple = if gemma4_n_pl > 0 {
            Some(load_gemma4_ple_weights(
                &g,
                n_embd,
                n_layer,
                n_vocab,
                gemma4_n_pl,
            )?)
        } else {
            None
        };
        let layers = {
            let layer_h = LayerHparams {
                qk_norm,
                phi2,
                bloom,
                gemma2,
                gemma3,
                gemma3n,
                gemma4,
                gemma4_n_pl,
                n_layer_kv_from_start,
                is_swa: &is_swa,
                gemma4_moe: gemma4_moe_hparams.as_ref(),
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
            layers
        };
        let gemma3n = if gemma3n {
            Some(load_gemma3n_weights(&g, n_embd, n_layer, n_vocab)?)
        } else {
            None
        };
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
            rope_neox,
            embed_scale,
            ffn_gelu: gemma,
            blob: Arc::new(g.into_blob()),
            token_embd,
            output_norm,
            output_norm_b,
            output,
            output_b,
            phi2,
            bloom,
            token_embd_norm,
            token_embd_norm_b,
            attn_logit_softcapping,
            final_logit_softcapping,
            n_swa,
            swa_period,
            is_swa,
            gemma4,
            gemma3n,
            gemma4_ple,
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
            scratch: Scratch::default(),
            pool: None,
            moe_trace: MoeTraceBuf::default(),
            expert_store: None,
            last_prefix_hit: 0,
            pages: None,
        })
    }

    /// KV cache whose K/V live in paged blocks of `block_size` tokens.
    ///
    /// Dense [`Llama::new_cache`] stays the default. Prefill/decode logits must
    /// bit-match that path. Completed blocks are interned so a later prompt on
    /// this cache can hit them after a rewind to 0.
    pub fn new_paged_cache(
        &self,
        max_seq: usize,
        block_size: usize,
    ) -> Result<KvCache, LlamaError> {
        let hd = self.head_dim()?;
        let n_layers = self.layers.len();
        let bs = block_size.min(max_seq.max(1));
        if bs == 0 || max_seq == 0 {
            return Err(LlamaError::Shape("kv page".into()));
        }
        let cap = max_seq.div_ceil(bs).saturating_add(1);
        let pages = KvPages::new(n_layers, self.n_head_kv, hd, bs, cap.max(2))
            .map_err(|e| LlamaError::Shape(e.into()))?;
        Ok(KvCache {
            k: Vec::new(),
            v: Vec::new(),
            n_past: 0,
            max_seq,
            scratch: Scratch::default(),
            pool: None,
            moe_trace: MoeTraceBuf::default(),
            expert_store: None,
            last_prefix_hit: 0,
            pages: Some(pages),
        })
    }

    /// Shared interned-block arena (`cap` physical blocks).
    ///
    /// Clone the handle into [`Llama::new_paged_cache_on`] so two sequences
    /// intern-hit the same prefixes. Distinct from `expertvm kv`.
    pub fn new_paged_pool(&self, block_size: usize, cap: usize) -> Result<PagedKvPool, LlamaError> {
        let hd = self.head_dim()?;
        let n_layers = self.layers.len();
        let bs = block_size.max(1);
        PagedKvPool::create(n_layers, self.n_head_kv, hd, bs, cap.max(2))
            .map_err(|e| LlamaError::Shape(e.into()))
    }

    /// Paged KV cache on an existing [`PagedKvPool`].
    pub fn new_paged_cache_on(
        &self,
        pool: &PagedKvPool,
        max_seq: usize,
    ) -> Result<KvCache, LlamaError> {
        if max_seq == 0 || pool.block_size() == 0 {
            return Err(LlamaError::Shape("kv page".into()));
        }
        Ok(KvCache {
            k: Vec::new(),
            v: Vec::new(),
            n_past: 0,
            max_seq,
            scratch: Scratch::default(),
            pool: None,
            moe_trace: MoeTraceBuf::default(),
            expert_store: None,
            last_prefix_hit: 0,
            pages: Some(KvPages::on(pool.clone())),
        })
    }

    /// Size or reuse `slot` so `needed` tokens fit.
    ///
    /// `Some(n_ctx)` allocates once at that capacity (error if `needed > n`).
    /// `None` keeps a cache whose `max_seq >= needed`, else allocates
    /// `needed + 1` (drops the previous KV layout; the stride is `max_seq`).
    pub fn ensure_cache<'a>(
        &self,
        slot: &'a mut Option<KvCache>,
        needed: usize,
        n_ctx: Option<usize>,
    ) -> Result<&'a mut KvCache, LlamaError> {
        let max_seq = match n_ctx {
            Some(n) if n < needed => return Err(LlamaError::Shape("n_ctx".into())),
            Some(n) => n,
            None => needed.saturating_add(1),
        };
        let keep = match (slot.as_ref(), n_ctx) {
            (Some(c), Some(n)) => c.max_seq == n && c.page_size().is_none(),
            (Some(c), None) => c.max_seq >= needed && c.page_size().is_none(),
            (None, _) => false,
        };
        if !keep {
            *slot = Some(self.new_cache(max_seq)?);
        }
        slot.as_mut()
            .ok_or_else(|| LlamaError::Shape("kv cache".into()))
    }

    /// [`Llama::ensure_cache`] with optional paged KV (`block_size`).
    pub fn ensure_cache_page<'a>(
        &self,
        slot: &'a mut Option<KvCache>,
        needed: usize,
        n_ctx: Option<usize>,
        page: Option<usize>,
    ) -> Result<&'a mut KvCache, LlamaError> {
        if page.is_none() {
            return self.ensure_cache(slot, needed, n_ctx);
        }
        let max_seq = match n_ctx {
            Some(n) if n < needed => return Err(LlamaError::Shape("n_ctx".into())),
            Some(n) => n,
            None => needed.saturating_add(1),
        };
        let keep = match (slot.as_ref(), n_ctx) {
            (Some(c), Some(n)) => c.max_seq == n && c.page_size() == page,
            (Some(c), None) => c.max_seq >= needed && c.page_size() == page,
            (None, _) => false,
        };
        if !keep {
            let bs = page.ok_or_else(|| LlamaError::Shape("kv page".into()))?;
            *slot = Some(self.new_paged_cache(max_seq, bs)?);
        }
        slot.as_mut()
            .ok_or_else(|| LlamaError::Shape("kv cache".into()))
    }

    /// Catalog every routed expert's gate/up/down part bytes. Identity oracle
    /// for the ExpertStore seam: GEMV on these copies matches the blob path.
    pub fn expert_direct_store(&self) -> Result<DirectStore, LlamaError> {
        let mut blobs = BTreeMap::new();
        for (li, layer) in self.layers.iter().enumerate() {
            let layer_u = u32::try_from(li).unwrap_or(u32::MAX);
            match &layer.ffn {
                LayerFfn::LlamaMoe(m) => self.catalog_exps(
                    layer_u,
                    m.n_expert,
                    &m.gate_exps,
                    &m.up_exps,
                    &m.down_exps,
                    &mut blobs,
                )?,
                LayerFfn::Llama4Moe(m) => self.catalog_exps(
                    layer_u,
                    m.n_expert,
                    &m.gate_exps,
                    &m.up_exps,
                    &m.down_exps,
                    &mut blobs,
                )?,
                LayerFfn::Qwen2Moe(m) | LayerFfn::Qwen3Next(m) => self.catalog_exps(
                    layer_u,
                    m.n_expert,
                    &m.gate_exps,
                    &m.up_exps,
                    &m.down_exps,
                    &mut blobs,
                )?,
                LayerFfn::Qwen3Moe(m) => self.catalog_exps(
                    layer_u,
                    m.n_expert,
                    &m.gate_exps,
                    &m.up_exps,
                    &m.down_exps,
                    &mut blobs,
                )?,
                LayerFfn::Gemma4Moe(m) => {
                    if let Some(gu) = m.gate_up.as_ref() {
                        self.catalog_fused_exps(layer_u, m.n_expert, gu, &m.down_exps, &mut blobs)?;
                    } else {
                        let gate = m
                            .gate_exps
                            .as_ref()
                            .ok_or_else(|| LlamaError::Shape("gemma4 moe".into()))?;
                        let up = m
                            .up_exps
                            .as_ref()
                            .ok_or_else(|| LlamaError::Shape("gemma4 moe".into()))?;
                        self.catalog_exps(layer_u, m.n_expert, gate, up, &m.down_exps, &mut blobs)?;
                    }
                }
                LayerFfn::Dense(_) | LayerFfn::Phi2(_) => {}
            }
        }
        Ok(DirectStore::new(blobs))
    }

    fn catalog_exps(
        &self,
        layer: u32,
        n_expert: usize,
        gate: &QuantMat,
        up: &QuantMat,
        down: &QuantMat,
        into: &mut BTreeMap<ExpertKey, ExpertParts>,
    ) -> Result<(), LlamaError> {
        for e in 0..n_expert {
            let expert = u32::try_from(e).map_err(|_| LlamaError::Shape("expert id".into()))?;
            let parts = ExpertParts {
                gate: self.part_bytes(gate, e)?,
                up: self.part_bytes(up, e)?,
                down: self.part_bytes(down, e)?,
            };
            let _prev = into.insert(ExpertKey::new(layer, expert), parts);
        }
        Ok(())
    }

    fn catalog_fused_exps(
        &self,
        layer: u32,
        n_expert: usize,
        gate_up: &QuantMat,
        down: &QuantMat,
        into: &mut BTreeMap<ExpertKey, ExpertParts>,
    ) -> Result<(), LlamaError> {
        for e in 0..n_expert {
            let expert = u32::try_from(e).map_err(|_| LlamaError::Shape("expert id".into()))?;
            let parts = ExpertParts {
                gate: self.part_bytes(gate_up, e)?,
                up: Vec::new(),
                down: self.part_bytes(down, e)?,
            };
            let _prev = into.insert(ExpertKey::new(layer, expert), parts);
        }
        Ok(())
    }

    fn part_bytes(&self, m: &QuantMat, part: usize) -> Result<Vec<u8>, LlamaError> {
        let (base, len) = self.mat_part_range(m, part)?;
        let end = base.saturating_add(len);
        match self.blob.get(base..end) {
            Some(s) => Ok(s.to_vec()),
            None => Err(LlamaError::Shape(m.name.clone())),
        }
    }

    #[cfg(test)]
    fn cache_buffer_ids(cache: &KvCache) -> Vec<(usize, usize)> {
        let mut out = vec![
            (cache.k.as_ptr() as usize, cache.k.capacity()),
            (cache.v.as_ptr() as usize, cache.v.capacity()),
        ];
        out.extend(cache.scratch.buffer_ids());
        if let Some(pool) = cache.pool.as_ref() {
            out.extend(pool.buffer_ids());
        }
        out
    }

    #[cfg(test)]
    fn cache_pool_workers(cache: &KvCache) -> usize {
        cache.pool.as_ref().map_or(0, Pool::workers)
    }

    /// Whether any weight matrix is big enough for the GEMV pool to pay off.
    /// `attn_q` / `attn_output` are `n_embd x n_embd` and the lm_head is the
    /// widest matrix, so those two bound the rest.
    fn wants_pool(&self) -> bool {
        pooled_gemv(self.output.n_rows, self.output.n_cols) || pooled_gemv(self.n_embd, self.n_embd)
    }

    fn head_dim(&self) -> Result<usize, LlamaError> {
        if self.n_embd.is_multiple_of(self.n_head) {
            Ok(self.n_embd / self.n_head)
        } else {
            Err(LlamaError::Shape("embedding_length / head_count".into()))
        }
    }

    /// Official gemma4 convert bool-array SWA, else [`gemma2_is_swa`].
    fn layer_is_swa(&self, li: usize) -> bool {
        self.is_swa
            .get(li)
            .copied()
            .unwrap_or_else(|| gemma2_is_swa(li, self.swa_period))
    }

    /// One decode token per cache. Shared-pool paged caches GEMM together.
    ///
    /// Q/K/V, FFN, and lm_head are one GEMM of `caches.len()` rows. Attention
    /// stays per sequence (its own `n_past` and block table). Logits of each
    /// row bit-match a sequential [`Llama::forward`]. Mixed dense/paged,
    /// two attached stores, or MoE traces fall back to one-at-a-time
    /// forwards. A single [`KvCache::attach_expert_store`] is used for the
    /// whole GEMM (Engine parks one store on the first cache). Markov
    /// prefetch state lives on that first cache across GEMMs. Enabled MoE
    /// traces record per-row sequence and token (not a sequential fallback).
    pub fn forward_batch(
        &self,
        caches: &mut [&mut KvCache],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, LlamaError> {
        if caches.len() != tokens.len() {
            return Err(LlamaError::Shape("forward batch".into()));
        }
        if caches.is_empty() {
            return Ok(Vec::new());
        }
        if caches.len() == 1 || !self.can_batch_paged(caches) {
            let mut out = Vec::new();
            for (cache, tok) in caches.iter_mut().zip(tokens.iter()) {
                out.push(self.forward(cache, *tok)?);
            }
            return Ok(out);
        }
        let mut groups: Vec<&[u32]> = Vec::new();
        for tok in tokens {
            groups.push(std::slice::from_ref(tok));
        }
        self.run_paged_batch(caches, &groups)
    }

    /// Prefill each cache with its token group. Shared-pool paged caches GEMM
    /// together even when the groups have different lengths.
    ///
    /// Attention stays per sequence. Returns last-token logits of each group,
    /// bit-matching sequential [`Llama::prefill`]. Prefix reuse / intern bind
    /// is the caller's job ([`Llama::prompt_chunk`], Engine). Mixed
    /// dense/two-store falls back to one-at-a-time prefills. One attached
    /// store is used for the whole GEMM. Enabled MoE traces stay on this path.
    pub fn prefill_batch(
        &self,
        caches: &mut [&mut KvCache],
        groups: &[&[u32]],
    ) -> Result<Vec<Vec<f32>>, LlamaError> {
        if caches.len() != groups.len() {
            return Err(LlamaError::Shape("prefill batch".into()));
        }
        if caches.is_empty() {
            return Ok(Vec::new());
        }
        if groups.iter().any(|g| g.is_empty()) {
            return Err(LlamaError::Shape("prefill tokens".into()));
        }
        if caches.len() == 1 || !self.can_batch_paged(caches) {
            let mut out = Vec::new();
            for (cache, group) in caches.iter_mut().zip(groups.iter()) {
                out.push(self.prefill(cache, group)?);
            }
            return Ok(out);
        }
        self.run_paged_batch(caches, groups)
    }

    fn can_batch_paged(&self, caches: &[&mut KvCache]) -> bool {
        let Some(first) = caches.first() else {
            return false;
        };
        let Some(home) = first.pages.as_ref().map(KvPages::pool) else {
            return false;
        };
        let stores = caches.iter().filter(|c| c.expert_store.is_some()).count();
        stores <= 1
            && caches
                .iter()
                .all(|c| c.pages.as_ref().is_some_and(|p| p.pool().same_as(home)))
    }

    fn run_paged_batch(
        &self,
        caches: &mut [&mut KvCache],
        groups: &[&[u32]],
    ) -> Result<Vec<Vec<f32>>, LlamaError> {
        let hd = self.head_dim()?;
        for (cache, group) in caches.iter_mut().zip(groups.iter()) {
            cache.prepare_append(group.len())?;
        }
        let mut flat = Vec::new();
        for group in groups {
            flat.extend_from_slice(group);
        }
        let n = flat.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let width = n
            .checked_mul(self.n_embd)
            .ok_or_else(|| LlamaError::Shape("prefill embed".into()))?;
        let (addrs, positions, max_seq) = batch_kv_layout(caches, groups)?;
        let store_idx = caches.iter().position(|c| c.expert_store.is_some());
        let mut parked_store =
            store_idx.and_then(|i| caches.get_mut(i).and_then(|c| c.expert_store.take()));
        let any_trace = caches.iter().any(|c| c.moe_trace.enabled);
        let enabled: Vec<bool> = caches.iter().map(|c| c.moe_trace.enabled).collect();
        let first_was = enabled.first().copied().unwrap_or(false);
        let rows = collect_batch_trace_rows(caches, groups);
        let result = (|| {
            apply_batch_trace_rows(caches, rows, &flat, any_trace)?;
            let Some(first) = caches.first_mut() else {
                return Ok(Vec::new());
            };
            let pages = first
                .pages
                .as_ref()
                .ok_or_else(|| LlamaError::Shape("kv page".into()))?;
            let mut pool_guard = pages
                .try_pool_mut()
                .map_err(|e| LlamaError::Shape(e.into()))?;
            let (cache_k, cache_v) = pool_guard.kv_mut();
            let s = &mut first.scratch;
            let pool = &mut first.pool;
            let moe_trace = &mut first.moe_trace;
            if s.scores.capacity() < max_seq {
                fit(&mut s.scores, max_seq);
            }
            fit(&mut s.x, width);
            for (t, tok) in flat.iter().enumerate() {
                let row = token_row_mut(&mut s.x, t, self.n_embd, "prefill embed")?;
                self.embed_into(*tok, row)?;
                for v in row.iter_mut() {
                    *v *= self.embed_scale;
                }
            }
            self.apply_token_embd_norm(&mut s.x)?;
            self.transformer(
                &mut TransformerRun {
                    s,
                    pool,
                    moe_trace,
                    expert_store: &mut parked_store,
                    cache_k,
                    cache_v,
                    n,
                    width,
                    hd,
                },
                &addrs,
                &positions,
                LogitsKind::All,
            )?;
            drop(pool_guard);
            let out = {
                let Some(first) = caches.first_mut() else {
                    return Ok(Vec::new());
                };
                last_group_logits(&first.scratch.logits, self.n_vocab, groups)?
            };
            for (cache, group) in caches.iter_mut().zip(groups.iter()) {
                cache.n_past = cache.n_past.saturating_add(group.len());
                cache.moe_trace.ids.extend_from_slice(group);
                if let Some(p) = cache.pages.as_mut() {
                    p.intern_full(&cache.moe_trace.ids);
                }
            }
            Ok(out)
        })();
        finish_batch_trace(caches, &enabled, first_was);
        restore_batch_store(caches, store_idx, parked_store);
        result
    }

    fn transformer(
        &self,
        run: &mut TransformerRun<'_>,
        addrs: &[KvAddr],
        pos: &[usize],
        logits: LogitsKind,
    ) -> Result<(), LlamaError> {
        if self.gemma3n.is_some() {
            return self.transformer_gemma3n(run, addrs, pos, logits);
        }
        let TransformerRun {
            s,
            pool,
            moe_trace,
            expert_store,
            cache_k,
            cache_v,
            n,
            width,
            hd,
        } = run;
        let n = *n;
        let width = *width;
        let hd = *hd;
        if let Some(w) = self.gemma4_ple.as_ref() {
            gemma4_ple_fit(s, w.n_embd_per_layer, n, self.layers.len(), self.n_embd);
            self.gemma4_project_per_layer(w, n, s, pool, &moe_trace.batch)?;
        }
        for (li, layer) in self.layers.iter().enumerate() {
            moe_trace.layer = u32::try_from(li).unwrap_or(u32::MAX);
            let kv_slot = layer.kv_slot;
            copy_buf(&mut s.residual, &s.x);
            if self.phi2 || self.bloom {
                layernorm_rows_inplace(
                    &mut s.x,
                    self.n_embd,
                    &layer.attn_norm,
                    layer.attn_norm_b.as_deref(),
                    self.rms_eps,
                )?;
            } else {
                rmsnorm_rows_inplace(&mut s.x, self.n_embd, &layer.attn_norm, self.rms_eps)?;
            }
            self.gemm_into(&layer.wq, n, &s.x, &mut s.q, pool)?;
            add_bias_rows(&mut s.q, layer.wq.n_rows, layer.bq.as_deref())?;
            match (&layer.wk, &layer.wv) {
                (Some(wk), Some(wv)) => {
                    if wk.n_rows != self.n_head_kv.saturating_mul(hd)
                        || wv.n_rows != self.n_head_kv.saturating_mul(hd)
                    {
                        return Err(LlamaError::Shape("qkv head split".into()));
                    }
                    self.gemm_into(wk, n, &s.x, &mut s.k, pool)?;
                    add_bias_rows(&mut s.k, wk.n_rows, layer.bk.as_deref())?;
                    self.gemm_into(wv, n, &s.x, &mut s.v, pool)?;
                    add_bias_rows(&mut s.v, wv.n_rows, layer.bv.as_deref())?;
                }
                (None, None) => {}
                _ => return Err(LlamaError::Shape("qkv head split".into())),
            }
            let q_width = self.n_head.saturating_mul(hd);
            if layer.attn_q_gate {
                split_qwen3next_q_gate_into(&mut s.q, &mut s.q_gate, n, self.n_head, hd)?;
            } else if layer.wq.n_rows != q_width {
                return Err(LlamaError::Shape("qkv head split".into()));
            }
            if let Some(w) = layer.attn_q_norm.as_deref() {
                rmsnorm_rows_inplace(&mut s.q, hd, w, self.rms_eps)?;
            }
            if let Some(w) = layer.attn_k_norm.as_deref() {
                rmsnorm_rows_inplace(&mut s.k, hd, w, self.rms_eps)?;
            }
            if self.gemma4 && layer.wv.is_some() {
                rmsnorm_unweighted_rows_inplace(&mut s.v, hd, self.rms_eps)?;
            }
            if let (Some(wk), Some(wv)) = (&layer.wk, &layer.wv) {
                for t in 0..n {
                    let p = token_pos(pos, t)?;
                    let geom = kv_addr(addrs, t)?.geom(self.n_head_kv, hd);
                    let k_t = token_row_mut(&mut s.k, t, wk.n_rows, "prefill k")?;
                    if layer.use_rope {
                        for h in k_t.chunks_mut(hd) {
                            apply_rope(
                                h,
                                p,
                                self.n_rot,
                                self.rope_base,
                                self.rope_sections,
                                self.rope_imrope,
                                self.rope_neox,
                            )?;
                        }
                        if layer.qk_l2 {
                            for h in k_t.chunks_mut(hd) {
                                rmsnorm_unweighted_inplace(h, self.rms_eps);
                            }
                        }
                    }
                    store_kv(cache_k, kv_slot, &geom, p, k_t)?;
                    let v_t = token_row(&s.v, t, wv.n_rows, "prefill v")?;
                    store_kv(cache_v, kv_slot, &geom, p, v_t)?;
                }
            }
            fit(&mut s.attn, width);
            for t in 0..n {
                let p = token_pos(pos, t)?;
                let geom = kv_addr(addrs, t)?.geom(self.n_head_kv, hd);
                let q_t = token_row_mut(&mut s.q, t, q_width, "prefill q")?;
                if layer.use_rope {
                    for h in q_t.chunks_mut(hd) {
                        apply_rope(
                            h,
                            p,
                            self.n_rot,
                            self.rope_base,
                            self.rope_sections,
                            self.rope_imrope,
                            self.rope_neox,
                        )?;
                    }
                    if layer.qk_l2 {
                        for h in q_t.chunks_mut(hd) {
                            rmsnorm_unweighted_inplace(h, self.rms_eps);
                        }
                    }
                    if self.phi2 {
                        let hd_f = f32::from(u16::try_from(hd).unwrap_or(1));
                        let q_scale = if hd_f > 0.0 { 1.0 / hd_f.sqrt() } else { 0.0 };
                        for v in q_t.iter_mut() {
                            *v *= q_scale;
                        }
                    }
                } else if !self.bloom {
                    let scale = llama4_attn_temp_scale(p);
                    for v in q_t.iter_mut() {
                        *v *= scale;
                    }
                }
                let score_scale = if self.phi2 || self.gemma4 {
                    1.0
                } else {
                    let scale = f32::from(u16::try_from(hd).unwrap_or(1)).sqrt();
                    if scale > 0.0 {
                        1.0 / scale
                    } else {
                        0.0
                    }
                };
                let dst_off = t.saturating_mul(self.n_embd);
                let dst = s
                    .attn
                    .get_mut(dst_off..dst_off.saturating_add(self.n_embd))
                    .ok_or_else(|| LlamaError::Shape("prefill attn".into()))?;
                let layer_swa = if self.layer_is_swa(li) { self.n_swa } else { 0 };
                attend_query(
                    cache_k,
                    cache_v,
                    kv_slot,
                    q_t,
                    &geom,
                    p.saturating_add(1),
                    score_scale,
                    if self.bloom {
                        BLOOM_MAX_ALIBI_BIAS
                    } else {
                        0.0
                    },
                    self.attn_logit_softcapping,
                    layer_swa,
                    &mut s.scores,
                    dst,
                )?;
            }
            if layer.attn_q_gate {
                if s.q_gate.len() != s.attn.len() {
                    return Err(LlamaError::Shape("attn gate".into()));
                }
                for (a, g) in s.attn.iter_mut().zip(s.q_gate.iter()) {
                    *a *= sigmoid_f32(*g);
                }
            }
            self.gemm_into(&layer.wo, n, &s.attn, &mut s.attn_proj, pool)?;
            add_bias_rows(&mut s.attn_proj, layer.wo.n_rows, layer.wo_b.as_deref())?;
            if let Some(w) = layer.attn_post_norm.as_deref() {
                rmsnorm_rows_inplace(&mut s.attn_proj, self.n_embd, w, self.rms_eps)?;
            }
            if self.phi2 {
                match &layer.ffn {
                    LayerFfn::Phi2(ffn) => {
                        self.gemm_into(&ffn.up, n, &s.x, &mut s.up, pool)?;
                        add_bias_rows(&mut s.up, ffn.up.n_rows, Some(ffn.up_b.as_slice()))?;
                        gelu_inplace(&mut s.up);
                        self.gemm_into(&ffn.down, n, &s.up, &mut s.ffn_out, pool)?;
                        add_bias_rows(
                            &mut s.ffn_out,
                            ffn.down.n_rows,
                            Some(ffn.down_b.as_slice()),
                        )?;
                    }
                    _ => return Err(LlamaError::Shape("phi2 ffn".into())),
                }
                add_into(&mut s.x, &s.attn_proj, &s.ffn_out)?;
                add_assign(&mut s.x, &s.residual)?;
            } else {
                add_into(&mut s.x, &s.attn_proj, &s.residual)?;
                copy_buf(&mut s.residual, &s.x);
                if self.bloom {
                    layernorm_rows_inplace(
                        &mut s.x,
                        self.n_embd,
                        &layer.ffn_norm,
                        layer.ffn_norm_b.as_deref(),
                        self.rms_eps,
                    )?;
                } else {
                    rmsnorm_rows_inplace(&mut s.x, self.n_embd, &layer.ffn_norm, self.rms_eps)?;
                }
                match &layer.ffn {
                    LayerFfn::Dense(dense) => {
                        self.gemm_into(&dense.gate, n, &s.x, &mut s.gate, pool)?;
                        self.gemm_into(&dense.up, n, &s.x, &mut s.up, pool)?;
                        ffn_gate_act_inplace(&mut s.gate, self.ffn_gelu);
                        for (hv, uv) in s.gate.iter_mut().zip(s.up.iter()) {
                            *hv *= *uv;
                        }
                        self.gemm_into(&dense.down, n, &s.gate, &mut s.ffn_out, pool)?;
                    }
                    LayerFfn::Llama4Moe(moe) => {
                        self.llama4_moe_into(moe.as_ref(), n, s, pool, moe_trace, expert_store)?
                    }
                    LayerFfn::LlamaMoe(moe) => {
                        self.llama_moe_into(moe.as_ref(), n, s, pool, moe_trace, expert_store)?
                    }
                    LayerFfn::Qwen2Moe(moe) => {
                        self.qwen2moe_into(moe.as_ref(), n, s, pool, moe_trace, expert_store)?
                    }
                    LayerFfn::Qwen3Moe(moe) => {
                        self.qwen3moe_into(moe.as_ref(), n, s, pool, moe_trace, expert_store)?
                    }
                    LayerFfn::Gemma4Moe(moe) => {
                        self.gemma4_moe_into(moe.as_ref(), n, s, pool, moe_trace, expert_store)?
                    }
                    LayerFfn::Qwen3Next(moe) => {
                        self.qwen3next_into(moe.as_ref(), n, s, pool, moe_trace, expert_store)?
                    }
                    LayerFfn::Phi2(ffn) => {
                        if !self.bloom {
                            return Err(LlamaError::Shape("phi2 ffn".into()));
                        }
                        self.gemm_into(&ffn.up, n, &s.x, &mut s.up, pool)?;
                        add_bias_rows(&mut s.up, ffn.up.n_rows, Some(ffn.up_b.as_slice()))?;
                        gelu_inplace(&mut s.up);
                        self.gemm_into(&ffn.down, n, &s.up, &mut s.ffn_out, pool)?;
                        add_bias_rows(
                            &mut s.ffn_out,
                            ffn.down.n_rows,
                            Some(ffn.down_b.as_slice()),
                        )?;
                    }
                }
                if let Some(w) = layer.ffn_post_norm.as_deref() {
                    rmsnorm_rows_inplace(&mut s.ffn_out, self.n_embd, w, self.rms_eps)?;
                }
                add_into(&mut s.x, &s.ffn_out, &s.residual)?;
                if let Some(w) = self.gemma4_ple.as_ref() {
                    let gl = layer
                        .gemma4_ple
                        .as_ref()
                        .ok_or_else(|| LlamaError::Shape("gemma4 per_layer".into()))?;
                    self.gemma4_per_layer_inject(w, gl, li, n, s, pool)?;
                }
            }
        }
        match logits {
            LogitsKind::Last => {
                let last_off = n.saturating_sub(1).saturating_mul(self.n_embd);
                let last =
                    s.x.get(last_off..last_off.saturating_add(self.n_embd))
                        .ok_or_else(|| LlamaError::Shape("prefill last".into()))?;
                copy_buf(&mut s.xn, last);
                if self.phi2 || self.bloom {
                    layernorm_inplace(
                        &mut s.xn,
                        &self.output_norm,
                        self.output_norm_b.as_deref(),
                        self.rms_eps,
                    )?;
                } else {
                    rmsnorm_inplace(&mut s.xn, &self.output_norm, self.rms_eps)?;
                }
                self.gemv_into(&self.output, &s.xn, &mut s.logits, pool)?;
                add_bias_rows(&mut s.logits, self.n_vocab, self.output_b.as_deref())?;
                tanh_softcap_inplace(&mut s.logits, self.final_logit_softcapping);
            }
            LogitsKind::All => {
                if self.phi2 || self.bloom {
                    layernorm_rows_inplace(
                        &mut s.x,
                        self.n_embd,
                        &self.output_norm,
                        self.output_norm_b.as_deref(),
                        self.rms_eps,
                    )?;
                } else {
                    rmsnorm_rows_inplace(&mut s.x, self.n_embd, &self.output_norm, self.rms_eps)?;
                }
                self.gemm_into(&self.output, n, &s.x, &mut s.logits, pool)?;
                add_bias_rows(&mut s.logits, self.n_vocab, self.output_b.as_deref())?;
                tanh_softcap_inplace(&mut s.logits, self.final_logit_softcapping);
            }
        }
        Ok(())
    }

    /// Official gemma3n.cpp language walk: AltUp + Laurel + per-layer inputs.
    fn transformer_gemma3n(
        &self,
        run: &mut TransformerRun<'_>,
        addrs: &[KvAddr],
        pos: &[usize],
        logits: LogitsKind,
    ) -> Result<(), LlamaError> {
        let Some(w) = self.gemma3n.as_ref() else {
            return Err(LlamaError::Shape("gemma3n".into()));
        };
        let n = run.n;
        gemma3n_fit(run.s, w, n, self.n_embd, self.layers.len());
        self.gemma3n_init_streams(w, n, run.s, run.pool)?;
        self.gemma3n_project_per_layer(w, n, run)?;
        for (li, layer) in self.layers.iter().enumerate() {
            run.moe_trace.layer = u32::try_from(li).unwrap_or(u32::MAX);
            let Some(gl) = layer.gemma3n.as_ref() else {
                return Err(LlamaError::Shape("gemma3n layer".into()));
            };
            self.gemma3n_altup_predict(w, gl, n, run.s, run.pool)?;
            let active = altup_stream(&run.s.g3n.pred, w.i_altup_act, n, self.n_embd)?;
            copy_buf(&mut run.s.x, active);
            copy_buf(&mut run.s.residual, &run.s.x);
            rmsnorm_rows_inplace(&mut run.s.x, self.n_embd, &layer.attn_norm, self.rms_eps)?;
            self.gemma3n_laurel(gl, n, run.s, run.pool)?;
            self.gemma3n_layer_attn(layer, li, run, addrs, pos)?;
            if let Some(pn) = layer.attn_post_norm.as_deref() {
                rmsnorm_rows_inplace(&mut run.s.attn_proj, self.n_embd, pn, self.rms_eps)?;
            }
            add_into(&mut run.s.g3n.tmp, &run.s.attn_proj, &run.s.residual)?;
            add_into(&mut run.s.x, &run.s.g3n.tmp, &run.s.g3n.laurel)?;
            let inv_sqrt2 = 1.0 / 2.0f32.sqrt();
            for v in run.s.x.iter_mut() {
                *v *= inv_sqrt2;
            }
            copy_buf(&mut run.s.g3n.tmp, &run.s.x);
            rmsnorm_rows_inplace(&mut run.s.x, self.n_embd, &layer.ffn_norm, self.rms_eps)?;
            match &layer.ffn {
                LayerFfn::Dense(dense) => {
                    self.gemm_into(&dense.gate, n, &run.s.x, &mut run.s.gate, run.pool)?;
                    self.gemm_into(&dense.up, n, &run.s.x, &mut run.s.up, run.pool)?;
                    if li < w.n_layer_sparsity {
                        gaussian_topk_inplace(&mut run.s.gate, dense.gate.n_rows)?;
                    }
                    ffn_gate_act_inplace(&mut run.s.gate, self.ffn_gelu);
                    for (hv, uv) in run.s.gate.iter_mut().zip(run.s.up.iter()) {
                        *hv *= *uv;
                    }
                    self.gemm_into(&dense.down, n, &run.s.gate, &mut run.s.ffn_out, run.pool)?;
                }
                _ => return Err(LlamaError::Shape("gemma3n ffn".into())),
            }
            if let Some(pn) = layer.ffn_post_norm.as_deref() {
                rmsnorm_rows_inplace(&mut run.s.ffn_out, self.n_embd, pn, self.rms_eps)?;
            }
            add_into(&mut run.s.x, &run.s.ffn_out, &run.s.g3n.tmp)?;
            self.gemma3n_altup_correct(w, gl, n, run.s, run.pool)?;
            self.gemma3n_per_layer_inject(w, gl, li, n, run.s, run.pool)?;
            copy_buf(&mut run.s.g3n.streams, &run.s.g3n.pred);
        }
        self.gemma3n_unembed(w, n, run.s, run.pool)?;
        self.gemma3n_logits(n, run.s, run.pool, logits)
    }

    fn gemma3n_init_streams(
        &self,
        w: &Gemma3nWeights,
        n: usize,
        s: &mut Scratch,
        pool: &mut GemvPool,
    ) -> Result<(), LlamaError> {
        let width = n.saturating_mul(self.n_embd);
        let Some(first) = s.g3n.streams.get_mut(..width) else {
            return Err(LlamaError::Shape("gemma3n streams".into()));
        };
        first.copy_from_slice(&s.x);
        for p in 0..w.n_altup.saturating_sub(1) {
            self.gemm_part_tokens_into(&w.altup_proj, p, n, &s.x, &mut s.g3n.tmp, pool)?;
            scale_rows_to_match(&s.x, &mut s.g3n.tmp, self.n_embd)?;
            let dst = altup_stream_mut(&mut s.g3n.streams, p.saturating_add(1), n, self.n_embd)?;
            dst.copy_from_slice(&s.g3n.tmp);
        }
        Ok(())
    }

    fn gemma3n_project_per_layer(
        &self,
        w: &Gemma3nWeights,
        n: usize,
        run: &mut TransformerRun<'_>,
    ) -> Result<(), LlamaError> {
        let n_ea = w.n_embd_altup;
        let n_layer = self.layers.len();
        let pl_cols = n_ea
            .checked_mul(n_layer)
            .ok_or_else(|| LlamaError::Shape("gemma3n per_layer".into()))?;
        if run.moe_trace.batch.len() != n {
            return Err(LlamaError::Shape("gemma3n tokens".into()));
        }
        let tok_scale = f32::from(u16::try_from(n_ea).unwrap_or(1)).sqrt();
        let n_embd_f = f32::from(u16::try_from(self.n_embd).unwrap_or(1));
        let proj_scale = if n_embd_f > 0.0 {
            1.0 / n_embd_f.sqrt()
        } else {
            0.0
        };
        let mix_scale = 1.0 / 2.0f32.sqrt();
        for t in 0..n {
            let tok = *run
                .moe_trace
                .batch
                .get(t)
                .ok_or_else(|| LlamaError::Shape("gemma3n tokens".into()))?;
            self.embed_mat_into(&w.per_layer_token_embd, tok, &mut run.s.g3n.per_tok)?;
            for v in run.s.g3n.per_tok.iter_mut() {
                *v *= tok_scale;
            }
            let off = t.saturating_mul(pl_cols);
            let dst = run
                .s
                .g3n
                .per_layer
                .get_mut(off..off.saturating_add(pl_cols))
                .ok_or_else(|| LlamaError::Shape("gemma3n per_layer".into()))?;
            dst.copy_from_slice(&run.s.g3n.per_tok);
        }
        self.gemm_into(
            &w.per_layer_model_proj,
            n,
            &run.s.x,
            &mut run.s.g3n.tmp,
            run.pool,
        )?;
        for v in run.s.g3n.tmp.iter_mut() {
            *v *= proj_scale;
        }
        rmsnorm_rows_inplace(
            &mut run.s.g3n.tmp,
            n_ea,
            &w.per_layer_proj_norm,
            self.rms_eps,
        )?;
        if run.s.g3n.tmp.len() != run.s.g3n.per_layer.len() {
            return Err(LlamaError::Shape("gemma3n per_layer".into()));
        }
        for (pl, pv) in run.s.g3n.per_layer.iter_mut().zip(run.s.g3n.tmp.iter()) {
            *pl = (*pl + *pv) * mix_scale;
        }
        Ok(())
    }

    fn gemma3n_altup_predict(
        &self,
        w: &Gemma3nWeights,
        gl: &Gemma3nLayer,
        n: usize,
        s: &mut Scratch,
        pool: &mut GemvPool,
    ) -> Result<(), LlamaError> {
        let active = altup_stream(&s.g3n.streams, w.i_altup_act, n, self.n_embd)?;
        copy_buf(&mut s.g3n.tmp, active);
        self.gemma3n_router(gl, n, s, pool)?;
        self.gemm_into(
            &gl.altup_predict_coef,
            n,
            &s.g3n.router,
            &mut s.g3n.coefs,
            pool,
        )?;
        let n_altup = w.n_altup;
        let n_embd = self.n_embd;
        let nsq = n_altup.saturating_mul(n_altup);
        for t in 0..n {
            let coef_off = t.saturating_mul(nsq);
            for i in 0..n_altup {
                {
                    let residual = altup_token(&s.g3n.streams, i, t, n, n_embd)?;
                    let pred = altup_token_mut(&mut s.g3n.pred, i, t, n, n_embd)?;
                    pred.copy_from_slice(residual);
                }
                for j in 0..n_altup {
                    let c = *s
                        .g3n
                        .coefs
                        .get(coef_off.saturating_add(j.saturating_add(i.saturating_mul(n_altup))))
                        .ok_or_else(|| LlamaError::Shape("gemma3n predict coef".into()))?;
                    let src = altup_token(&s.g3n.streams, j, t, n, n_embd)?;
                    let pred = altup_token_mut(&mut s.g3n.pred, i, t, n, n_embd)?;
                    for (d, sv) in pred.iter_mut().zip(src.iter()) {
                        *d += c * *sv;
                    }
                }
            }
        }
        Ok(())
    }

    fn gemma3n_altup_correct(
        &self,
        w: &Gemma3nWeights,
        gl: &Gemma3nLayer,
        n: usize,
        s: &mut Scratch,
        pool: &mut GemvPool,
    ) -> Result<(), LlamaError> {
        copy_buf(&mut s.g3n.tmp, &s.x);
        self.gemma3n_router(gl, n, s, pool)?;
        self.gemm_into(
            &gl.altup_correct_coef,
            n,
            &s.g3n.router,
            &mut s.g3n.coefs,
            pool,
        )?;
        for c in s.g3n.coefs.iter_mut() {
            *c += 1.0;
        }
        let n_altup = w.n_altup;
        let n_embd = self.n_embd;
        fit(&mut s.attn, n.saturating_mul(n_embd));
        for t in 0..n {
            let activated = token_row(&s.g3n.tmp, t, n_embd, "gemma3n correct")?;
            let active_pred = altup_token(&s.g3n.pred, w.i_altup_act, t, n, n_embd)?;
            let inn = token_row_mut(&mut s.attn, t, n_embd, "gemma3n innov")?;
            for (d, (av, pv)) in inn.iter_mut().zip(activated.iter().zip(active_pred.iter())) {
                *d = *av - *pv;
            }
        }
        for t in 0..n {
            let coef_off = t.saturating_mul(n_altup);
            for i in 0..n_altup {
                let c = *s
                    .g3n
                    .coefs
                    .get(coef_off.saturating_add(i))
                    .ok_or_else(|| LlamaError::Shape("gemma3n correct coef".into()))?;
                let inn = token_row(&s.attn, t, n_embd, "gemma3n innov")?;
                let pred = altup_token_mut(&mut s.g3n.pred, i, t, n, n_embd)?;
                for (d, iv) in pred.iter_mut().zip(inn.iter()) {
                    *d += c * *iv;
                }
            }
        }
        Ok(())
    }

    fn gemma3n_router(
        &self,
        gl: &Gemma3nLayer,
        n: usize,
        s: &mut Scratch,
        pool: &mut GemvPool,
    ) -> Result<(), LlamaError> {
        copy_buf(&mut s.x, &s.g3n.tmp);
        rmsnorm_rows_inplace(&mut s.x, self.n_embd, &gl.altup_router_norm, self.rms_eps)?;
        let n_embd_f = f32::from(u16::try_from(self.n_embd).unwrap_or(1));
        let inv = if n_embd_f > 0.0 { 1.0 / n_embd_f } else { 0.0 };
        for v in s.x.iter_mut() {
            *v *= inv;
        }
        self.gemm_into(&gl.altup_router, n, &s.x, &mut s.g3n.router, pool)?;
        for v in s.g3n.router.iter_mut() {
            *v = v.tanh();
        }
        Ok(())
    }

    fn gemma3n_laurel(
        &self,
        gl: &Gemma3nLayer,
        n: usize,
        s: &mut Scratch,
        pool: &mut GemvPool,
    ) -> Result<(), LlamaError> {
        self.gemm_into(&gl.laurel_l, n, &s.x, &mut s.gate, pool)?;
        self.gemm_into(&gl.laurel_r, n, &s.gate, &mut s.g3n.laurel, pool)?;
        rmsnorm_rows_inplace(
            &mut s.g3n.laurel,
            self.n_embd,
            &gl.laurel_post_norm,
            self.rms_eps,
        )?;
        add_assign(&mut s.g3n.laurel, &s.x)?;
        Ok(())
    }

    fn gemma3n_layer_attn(
        &self,
        layer: &Layer,
        li: usize,
        run: &mut TransformerRun<'_>,
        addrs: &[KvAddr],
        pos: &[usize],
    ) -> Result<(), LlamaError> {
        let n = run.n;
        let hd = run.hd;
        let width = run.width;
        let s = &mut *run.s;
        let pool = &mut *run.pool;
        let cache_k = &mut *run.cache_k;
        let cache_v = &mut *run.cache_v;
        let wk = layer
            .wk
            .as_ref()
            .ok_or_else(|| LlamaError::Shape("qkv head split".into()))?;
        let wv = layer
            .wv
            .as_ref()
            .ok_or_else(|| LlamaError::Shape("qkv head split".into()))?;
        if wk.n_rows != self.n_head_kv.saturating_mul(hd)
            || wv.n_rows != self.n_head_kv.saturating_mul(hd)
        {
            return Err(LlamaError::Shape("qkv head split".into()));
        }
        self.gemm_into(&layer.wq, n, &s.x, &mut s.q, pool)?;
        self.gemm_into(wk, n, &s.x, &mut s.k, pool)?;
        self.gemm_into(wv, n, &s.x, &mut s.v, pool)?;
        let q_width = self.n_head.saturating_mul(hd);
        if layer.wq.n_rows != q_width {
            return Err(LlamaError::Shape("qkv head split".into()));
        }
        if let Some(w) = layer.attn_q_norm.as_deref() {
            rmsnorm_rows_inplace(&mut s.q, hd, w, self.rms_eps)?;
        }
        if let Some(w) = layer.attn_k_norm.as_deref() {
            rmsnorm_rows_inplace(&mut s.k, hd, w, self.rms_eps)?;
        }
        rmsnorm_unweighted_rows_inplace(&mut s.v, hd, self.rms_eps)?;
        for t in 0..n {
            let p = token_pos(pos, t)?;
            let geom = kv_addr(addrs, t)?.geom(self.n_head_kv, hd);
            let k_t = token_row_mut(&mut s.k, t, wk.n_rows, "gemma3n k")?;
            for h in k_t.chunks_mut(hd) {
                apply_rope(
                    h,
                    p,
                    self.n_rot,
                    self.rope_base,
                    self.rope_sections,
                    self.rope_imrope,
                    self.rope_neox,
                )?;
            }
            store_kv(cache_k, li, &geom, p, k_t)?;
            let v_t = token_row(&s.v, t, wv.n_rows, "gemma3n v")?;
            store_kv(cache_v, li, &geom, p, v_t)?;
        }
        fit(&mut s.attn, width);
        for t in 0..n {
            let p = token_pos(pos, t)?;
            let geom = kv_addr(addrs, t)?.geom(self.n_head_kv, hd);
            let q_t = token_row_mut(&mut s.q, t, q_width, "gemma3n q")?;
            for h in q_t.chunks_mut(hd) {
                apply_rope(
                    h,
                    p,
                    self.n_rot,
                    self.rope_base,
                    self.rope_sections,
                    self.rope_imrope,
                    self.rope_neox,
                )?;
            }
            let dst_off = t.saturating_mul(self.n_embd);
            let dst = s
                .attn
                .get_mut(dst_off..dst_off.saturating_add(self.n_embd))
                .ok_or_else(|| LlamaError::Shape("gemma3n attn".into()))?;
            let layer_swa = if gemma2_is_swa(li, self.swa_period) {
                self.n_swa
            } else {
                0
            };
            attend_query(
                cache_k,
                cache_v,
                li,
                q_t,
                &geom,
                p.saturating_add(1),
                1.0,
                0.0,
                0.0,
                layer_swa,
                &mut s.scores,
                dst,
            )?;
        }
        self.gemm_into(&layer.wo, n, &s.attn, &mut s.attn_proj, pool)
    }

    fn gemma3n_per_layer_inject(
        &self,
        w: &Gemma3nWeights,
        gl: &Gemma3nLayer,
        li: usize,
        n: usize,
        s: &mut Scratch,
        pool: &mut GemvPool,
    ) -> Result<(), LlamaError> {
        let active = altup_stream(&s.g3n.pred, w.i_altup_act, n, self.n_embd)?;
        copy_buf(&mut s.x, active);
        if s.x.len() != gl.altup_correct_scale.len().saturating_mul(n)
            && gl.altup_correct_scale.len() != self.n_embd
        {
            return Err(LlamaError::Shape("altup_correct_scale".into()));
        }
        for t in 0..n {
            let row = token_row_mut(&mut s.x, t, self.n_embd, "gemma3n scale")?;
            for (v, sc) in row.iter_mut().zip(gl.altup_correct_scale.iter()) {
                *v *= *sc;
            }
        }
        self.gemm_into(&gl.inp_gate, n, &s.x, &mut s.gate, pool)?;
        gelu_inplace(&mut s.gate);
        let n_ea = w.n_embd_altup;
        let n_layer = self.layers.len();
        for t in 0..n {
            let gate = token_row_mut(&mut s.gate, t, n_ea, "gemma3n inp_gate")?;
            let pl = per_layer_at(&s.g3n.per_layer, li, t, n_ea, n_layer)?;
            for (g, p) in gate.iter_mut().zip(pl.iter()) {
                *g *= *p;
            }
        }
        self.gemm_into(&gl.proj, n, &s.gate, &mut s.ffn_out, pool)?;
        rmsnorm_rows_inplace(&mut s.ffn_out, self.n_embd, &gl.post_norm, self.rms_eps)?;
        for i in 1..w.n_altup {
            for t in 0..n {
                let dst = altup_token_mut(&mut s.g3n.pred, i, t, n, self.n_embd)?;
                let src = token_row(&s.ffn_out, t, self.n_embd, "gemma3n inject")?;
                add_assign(dst, src)?;
            }
        }
        Ok(())
    }

    /// Official gemma4.cpp `build_inp_per_layer` plus `project_per_layer_inputs`.
    /// Token-major layout `[token][layer][n_embd_per_layer]` matches ggml before
    /// the inject permute (`n_layer=1` is identical after permute).
    fn gemma4_project_per_layer(
        &self,
        w: &Gemma4PleWeights,
        n: usize,
        s: &mut Scratch,
        pool: &mut GemvPool,
        tokens: &[u32],
    ) -> Result<(), LlamaError> {
        let n_pl = w.n_embd_per_layer;
        let n_layer = self.layers.len();
        let pl_cols = n_pl
            .checked_mul(n_layer)
            .ok_or_else(|| LlamaError::Shape("gemma4 per_layer".into()))?;
        if tokens.len() != n {
            return Err(LlamaError::Shape("gemma4 tokens".into()));
        }
        let tok_scale = f32::from(u16::try_from(n_pl).unwrap_or(1)).sqrt();
        let n_embd_f = f32::from(u16::try_from(self.n_embd).unwrap_or(1));
        let proj_scale = if n_embd_f > 0.0 {
            1.0 / n_embd_f.sqrt()
        } else {
            0.0
        };
        let mix_scale = 1.0 / 2.0f32.sqrt();
        for t in 0..n {
            let tok = *tokens
                .get(t)
                .ok_or_else(|| LlamaError::Shape("gemma4 tokens".into()))?;
            self.embed_mat_into(&w.per_layer_token_embd, tok, &mut s.g3n.per_tok)?;
            for v in s.g3n.per_tok.iter_mut() {
                *v *= tok_scale;
            }
            let off = t.saturating_mul(pl_cols);
            let dst = s
                .g3n
                .per_layer
                .get_mut(off..off.saturating_add(pl_cols))
                .ok_or_else(|| LlamaError::Shape("gemma4 per_layer".into()))?;
            dst.copy_from_slice(&s.g3n.per_tok);
        }
        self.gemm_into(&w.per_layer_model_proj, n, &s.x, &mut s.g3n.tmp, pool)?;
        for v in s.g3n.tmp.iter_mut() {
            *v *= proj_scale;
        }
        rmsnorm_rows_inplace(&mut s.g3n.tmp, n_pl, &w.per_layer_proj_norm, self.rms_eps)?;
        if s.g3n.tmp.len() != s.g3n.per_layer.len() {
            return Err(LlamaError::Shape("gemma4 per_layer".into()));
        }
        for (pl, pv) in s.g3n.per_layer.iter_mut().zip(s.g3n.tmp.iter()) {
            *pl = (*pl + *pv) * mix_scale;
        }
        Ok(())
    }

    /// Official gemma4.cpp per-layer inject after the FFN residual (`pe_in`).
    fn gemma4_per_layer_inject(
        &self,
        w: &Gemma4PleWeights,
        gl: &Gemma4PleLayer,
        li: usize,
        n: usize,
        s: &mut Scratch,
        pool: &mut GemvPool,
    ) -> Result<(), LlamaError> {
        self.gemm_into(&gl.inp_gate, n, &s.x, &mut s.gate, pool)?;
        gelu_inplace(&mut s.gate);
        let n_pl = w.n_embd_per_layer;
        let n_layer = self.layers.len();
        for t in 0..n {
            let gate = token_row_mut(&mut s.gate, t, n_pl, "gemma4 inp_gate")?;
            let pl = per_layer_at(&s.g3n.per_layer, li, t, n_pl, n_layer)?;
            for (g, p) in gate.iter_mut().zip(pl.iter()) {
                *g *= *p;
            }
        }
        self.gemm_into(&gl.proj, n, &s.gate, &mut s.ffn_out, pool)?;
        rmsnorm_rows_inplace(&mut s.ffn_out, self.n_embd, &gl.post_norm, self.rms_eps)?;
        add_assign(&mut s.x, &s.ffn_out)?;
        Ok(())
    }

    fn gemma3n_unembed(
        &self,
        w: &Gemma3nWeights,
        n: usize,
        s: &mut Scratch,
        pool: &mut GemvPool,
    ) -> Result<(), LlamaError> {
        let active = altup_stream(&s.g3n.streams, w.i_altup_act, n, self.n_embd)?;
        copy_buf(&mut s.x, active);
        copy_buf(&mut s.residual, &s.x);
        let inv = 1.0 / f32::from(u16::try_from(w.n_altup).unwrap_or(1));
        for p in 0..w.n_altup.saturating_sub(1) {
            let src = altup_stream(&s.g3n.streams, p.saturating_add(1), n, self.n_embd)?;
            copy_buf(&mut s.g3n.tmp, src);
            self.gemm_part_tokens_into(
                &w.altup_unembd_proj,
                p,
                n,
                &s.g3n.tmp,
                &mut s.ffn_out,
                pool,
            )?;
            scale_rows_to_match(&s.x, &mut s.ffn_out, self.n_embd)?;
            add_assign(&mut s.residual, &s.ffn_out)?;
        }
        for v in s.residual.iter_mut() {
            *v *= inv;
        }
        copy_buf(&mut s.x, &s.residual);
        Ok(())
    }

    fn gemma3n_logits(
        &self,
        n: usize,
        s: &mut Scratch,
        pool: &mut GemvPool,
        logits: LogitsKind,
    ) -> Result<(), LlamaError> {
        match logits {
            LogitsKind::Last => {
                let last_off = n.saturating_sub(1).saturating_mul(self.n_embd);
                let last =
                    s.x.get(last_off..last_off.saturating_add(self.n_embd))
                        .ok_or_else(|| LlamaError::Shape("gemma3n last".into()))?;
                copy_buf(&mut s.xn, last);
                rmsnorm_inplace(&mut s.xn, &self.output_norm, self.rms_eps)?;
                self.gemv_into(&self.output, &s.xn, &mut s.logits, pool)?;
            }
            LogitsKind::All => {
                rmsnorm_rows_inplace(&mut s.x, self.n_embd, &self.output_norm, self.rms_eps)?;
                self.gemm_into(&self.output, n, &s.x, &mut s.logits, pool)?;
            }
        }
        tanh_softcap_inplace(&mut s.logits, self.final_logit_softcapping);
        Ok(())
    }

    fn gemm_part_tokens_into(
        &self,
        m: &QuantMat,
        part: usize,
        n_tokens: usize,
        x: &[f32],
        y: &mut Vec<f32>,
        pool: &mut GemvPool,
    ) -> Result<(), LlamaError> {
        self.gemm_part_bytes_into(
            PartBytes {
                m,
                part,
                n_tokens,
                bytes: None,
            },
            x,
            y,
            pool,
        )
    }

    /// One decode step. Writes K/V at `cache.n_past` and increments it. Returns logits.
    ///
    /// Copies the logits out of the cache. [`Llama::forward_logits`] borrows
    /// them instead and is the allocation-free entry point.
    pub fn forward(&self, cache: &mut KvCache, token: u32) -> Result<Vec<f32>, LlamaError> {
        self.forward_logits(cache, token).map(<[f32]>::to_vec)
    }

    /// [`Llama::forward`] returning the cache's own logits buffer instead of a
    /// fresh `Vec`. `cache` stays borrowed for as long as the slice is held.
    ///
    /// Allocates nothing once the cache's buffers have been sized by a first
    /// call of the same or larger token count.
    pub fn forward_logits<'c>(
        &self,
        cache: &'c mut KvCache,
        token: u32,
    ) -> Result<&'c [f32], LlamaError> {
        self.prefill_logits(cache, &[token])
    }

    /// Decode `tokens` in one causal pass. Prompt tokens share each weight
    /// row (GEMM; a single token stays GEMV). Writes K/V at
    /// `cache.n_past .. n_past+len` and returns logits of the last token.
    pub fn prefill(&self, cache: &mut KvCache, tokens: &[u32]) -> Result<Vec<f32>, LlamaError> {
        self.prefill_logits(cache, tokens).map(<[f32]>::to_vec)
    }

    /// [`Llama::prefill`] returning the cache's own logits buffer.
    pub fn prefill_logits<'c>(
        &self,
        cache: &'c mut KvCache,
        tokens: &[u32],
    ) -> Result<&'c [f32], LlamaError> {
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
        let width = n
            .checked_mul(self.n_embd)
            .ok_or_else(|| LlamaError::Shape("prefill embed".into()))?;
        let KvCache {
            k: dense_k,
            v: dense_v,
            n_past,
            max_seq,
            scratch: s,
            pool,
            moe_trace,
            expert_store,
            last_prefix_hit: _,
            pages,
        } = cache;
        let max_seq = *max_seq;
        moe_trace.token0 = u32::try_from(n0).unwrap_or(u32::MAX);
        moe_trace.batch.clear();
        moe_trace.batch.extend(tokens.iter().copied());
        moe_trace.row_seq.clear();
        moe_trace.row_tok.clear();
        moe_trace.row_prefix.clear();
        // Single-token steps are the only ones the pool serves: GEMM lays `y`
        // out token-major, so a row range there is not a contiguous slice.
        if n == 1 && pool.is_none() && self.wants_pool() {
            *pool = Pool::new(
                Arc::new(GemvRows {
                    blob: Arc::clone(&self.blob),
                }),
                self.output.n_rows.max(self.n_embd),
            );
        }
        // Attention scores are `n_past + 1` long, so they would otherwise grow
        // on every decode step. Take the full-context capacity up front.
        if s.scores.capacity() < max_seq {
            fit(&mut s.scores, max_seq);
        }
        fit(&mut s.x, width);
        for (t, tok) in tokens.iter().enumerate() {
            let row = token_row_mut(&mut s.x, t, self.n_embd, "prefill embed")?;
            self.embed_into(*tok, row)?;
            for v in row.iter_mut() {
                *v *= self.embed_scale;
            }
        }
        self.apply_token_embd_norm(&mut s.x)?;
        let mut table_copy = Vec::new();
        let mut page_bs = max_seq;
        let mut page_nl = 0usize;
        if let Some(p) = pages.as_mut() {
            for pos in n0..end {
                p.ensure_write(pos)
                    .map_err(|e| LlamaError::Shape(e.into()))?;
            }
            table_copy.extend_from_slice(p.table_ids());
            page_bs = p.block_size();
            page_nl = p.n_layers();
        }
        let addr = if pages.is_some() {
            KvAddr {
                table: table_copy,
                block_size: page_bs,
                n_layers: page_nl,
                dense_max: 0,
            }
        } else {
            KvAddr {
                table: Vec::new(),
                block_size: max_seq,
                n_layers: 0,
                dense_max: max_seq,
            }
        };
        let mut positions = Vec::new();
        for t in 0..n {
            positions.push(n0.saturating_add(t));
        }
        let mut pool_guard = match pages.as_ref() {
            Some(p) => Some(p.try_pool_mut().map_err(|e| LlamaError::Shape(e.into()))?),
            None => None,
        };
        let (cache_k, cache_v): (&mut [f32], &mut [f32]) = match pool_guard.as_mut() {
            Some(g) => g.kv_mut(),
            None => (dense_k.as_mut_slice(), dense_v.as_mut_slice()),
        };
        self.transformer(
            &mut TransformerRun {
                s,
                pool,
                moe_trace,
                expert_store,
                cache_k,
                cache_v,
                n,
                width,
                hd,
            },
            &[addr],
            &positions,
            LogitsKind::Last,
        )?;
        drop(pool_guard);
        *n_past = end;
        moe_trace.ids.extend_from_slice(&moe_trace.batch);
        if let Some(p) = pages.as_mut() {
            p.intern_full(&moe_trace.ids);
        }
        Ok(&s.logits)
    }

    /// [`Llama::prompt_logits`] copying logits out of the cache.
    pub fn prompt(&self, cache: &mut KvCache, tokens: &[u32]) -> Result<Vec<f32>, LlamaError> {
        self.prompt_logits(cache, tokens).map(<[f32]>::to_vec)
    }

    /// Prefill `tokens` as a full prompt, reusing cached KV for the longest
    /// matching prefix (vLLM Automatic Prefix Caching on this engine).
    ///
    /// Not used by [`Llama::forward`] / append [`Llama::prefill_logits`]. A
    /// full-prefix hit rewinds one token and recomputes it so the logits are
    /// of the last prompt token (scratch otherwise still holds the last
    /// generated token). `max_suffix` caps newly forwarded tokens (`0` = the
    /// whole remaining suffix) for chunked prefill.
    fn prompt_with_suffix_cap<'c>(
        &self,
        cache: &'c mut KvCache,
        tokens: &[u32],
        max_suffix: usize,
    ) -> Result<&'c [f32], LlamaError> {
        let part = cache.prompt_suffix(tokens, max_suffix)?;
        self.prefill_logits(cache, part)
    }

    /// [`Llama::prompt_logits`]: reuse + intern bind, then at most `max_suffix`
    /// new tokens (`0` = the whole remaining suffix).
    ///
    /// A full-prefix intern/LCP hit still recomputes the last prompt token.
    /// Used by the continuous-batching engine for chunked prefill.
    pub fn prompt_chunk<'c>(
        &self,
        cache: &'c mut KvCache,
        tokens: &[u32],
        max_suffix: usize,
    ) -> Result<&'c [f32], LlamaError> {
        self.prompt_with_suffix_cap(cache, tokens, max_suffix)
    }

    /// [`Llama::prompt_chunk`] with `max_suffix = 0` (whole remaining suffix).
    pub fn prompt_logits<'c>(
        &self,
        cache: &'c mut KvCache,
        tokens: &[u32],
    ) -> Result<&'c [f32], LlamaError> {
        self.prompt_with_suffix_cap(cache, tokens, 0)
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
    let mut slot = None;
    greedy_generate_slot(model, tok, &mut slot, prompt, n_predict, n_ctx, None)
}

/// [`greedy_generate_ctx`] on a persistent KV slot (Automatic Prefix Caching).
pub fn greedy_generate_slot(
    model: &Llama,
    tok: &Tokenizer,
    slot: &mut Option<KvCache>,
    prompt: &str,
    n_predict: usize,
    n_ctx: Option<usize>,
    kv_page: Option<usize>,
) -> Result<String, LlamaError> {
    if prompt.is_empty() {
        return Err(LlamaError::EmptyPrompt);
    }
    let mut ids = prompt_ids(tok, prompt)?;
    if ids.is_empty() {
        return Err(LlamaError::EmptyPrompt);
    }
    let needed = ids.len().saturating_add(n_predict);
    let cache = model.ensure_cache_page(slot, needed, n_ctx, kv_page)?;
    greedy_generate_cache(model, tok, cache, &mut ids, n_predict)
}

/// Seedless greedy on an existing KV cache. Prefills with [`Llama::prompt_logits`]
/// so a shared prefix is not recomputed. `cache.max_seq` must fit prompt +
/// `n_predict`. `ids` is the encoded prompt (mutated to include generated tokens).
pub fn greedy_generate_cache(
    model: &Llama,
    tok: &Tokenizer,
    cache: &mut KvCache,
    ids: &mut Vec<u32>,
    n_predict: usize,
) -> Result<String, LlamaError> {
    if ids.is_empty() {
        return Err(LlamaError::EmptyPrompt);
    }
    let needed = ids.len().saturating_add(n_predict);
    if cache.max_seq < needed {
        return Err(LlamaError::Shape("n_ctx".into()));
    }
    let mut next = argmax(model.prompt_logits(cache, ids)?);
    for _ in 0..n_predict {
        if tok.eos == Some(next) {
            break;
        }
        ids.push(next);
        next = argmax(model.forward_logits(cache, next)?);
    }
    Ok(tok.decode(ids))
}

/// [`greedy_generate_ctx`] plus an [`expertvm::Trace`] of every MoE router decision.
///
/// Tracing is opt-in. Logits and greedy tokens must match an untraced run.
pub fn greedy_generate_traced(
    model: &Llama,
    tok: &Tokenizer,
    prompt: &str,
    n_predict: usize,
    n_ctx: Option<usize>,
    sequence: u64,
) -> Result<(String, Trace), LlamaError> {
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
    cache.enable_moe_trace(sequence);
    let text = greedy_generate_cache(model, tok, &mut cache, &mut ids, n_predict)?;
    let trace = cache.take_moe_trace();
    Ok((text, trace))
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
    // The sampler needs `ids` while it reads the logits, so keep one reusable
    // copy here rather than borrowing the cache across the call.
    let mut last = Vec::new();
    copy_buf(&mut last, model.prompt_logits(&mut cache, &ids)?);
    let mut sampler = Sampler::new(*params)?;
    for _ in 0..n_predict {
        let next = sampler.sample(&last, &ids)?;
        if tok.eos == Some(next) {
            break;
        }
        ids.push(next);
        copy_buf(&mut last, model.forward_logits(&mut cache, next)?);
    }
    Ok(tok.decode(&ids))
}

pub(crate) fn prompt_ids(tok: &Tokenizer, prompt: &str) -> Result<Vec<u32>, LlamaError> {
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
    })
}

/// Writer-built official Gemma2 GGUF: `architecture=gemma2` with `gemma2.*` KV.
///
/// Official `general.architecture=gemma2` (`MODEL_ARCH_NAMES[GEMMA2] = "gemma2"`,
/// `Gemma2ForCausalLM` → `MODEL_ARCH.GEMMA2`). Decode follows llama.cpp
/// `src/models/gemma2.cpp`: Gemma embed-scale + GeGLU, RMSNorm
/// `post_attention_norm` on attn out before residual, `ffn_norm` before FFN,
/// RMSNorm `post_ffw_norm` on FFN out before residual, sliding-window attention
/// (`set_swa_pattern(2)`, `LLAMA_SWA_TYPE_STANDARD`), attn/final tanh logit
/// softcap. Convert skips `lm_head.weight` (tied), omits `rope.dimension_count`
/// / `sliding_window_pattern`, writes `attn_logit_softcapping` /
/// `final_logit_softcapping` / `attention.sliding_window` / `attention.key_length`
/// / `attention.value_length` / `context_length`. Writer-tiny uses
/// `attention.sliding_window = 2` so short-seq tests clip. Convert-hf bakes
/// `norm.weight + 1`; runtime does not add 1 again.
pub fn tiny_gemma2_gguf() -> Vec<u8> {
    tiny_arch_gguf_lm_head(
        TinySpec {
            arch: "gemma2",
            token_embd: GgmlType::F32,
            output: GgmlType::F32,
            layer: None,
            rope_dimension_count: false,
            qkv_bias: false,
            add_bos_token: None,
            llama_moe: false,
            gemma4_moe: false,
            gemma4_ple: false,
            gemma4_fused: false,
        },
        TinyLmHead::Tied,
    )
}

/// Writer-built official Gemma3 GGUF: `architecture=gemma3` with `gemma3.*` KV.
///
/// Official `general.architecture=gemma3` (`MODEL_ARCH_NAMES[GEMMA3] = "gemma3"`,
/// `Gemma3ForCausalLM` → `MODEL_ARCH.GEMMA3`). Decode follows llama.cpp
/// `src/models/gemma3.cpp`: Gemma embed-scale + GeGLU, QK-Norm before RoPE
/// (`attn_q_norm` / `attn_k_norm`), RMSNorm `post_attention_norm` /
/// `post_ffw_norm`, SWA default period 6 (`set_swa_pattern`,
/// `LLAMA_SWA_TYPE_STANDARD` when `n_swa > 0`), no attn logit softcap,
/// optional final tanh softcap. Convert skips `lm_head.weight` when tied,
/// omits `attn_logit_softcapping`, writes `attention.sliding_window` when
/// `sliding_window_pattern != 1`. Writer-tiny uses `attention.sliding_window = 2`
/// so short-seq tests clip. Convert-hf bakes `norm.weight + 1`; runtime does
/// not add 1 again. `gemma3n` is a separate official family.
pub fn tiny_gemma3_gguf() -> Vec<u8> {
    tiny_arch_gguf_lm_head(
        TinySpec {
            arch: "gemma3",
            token_embd: GgmlType::F32,
            output: GgmlType::F32,
            layer: None,
            rope_dimension_count: false,
            qkv_bias: false,
            add_bos_token: None,
            llama_moe: false,
            gemma4_moe: false,
            gemma4_ple: false,
            gemma4_fused: false,
        },
        TinyLmHead::Tied,
    )
}

/// Writer-built official Gemma3n GGUF: `architecture=gemma3n` with `gemma3n.*` KV.
///
/// Official `general.architecture=gemma3n` (`MODEL_ARCH_NAMES[GEMMA3N] = "gemma3n"`,
/// `Gemma3nForCausalLM` → `MODEL_ARCH.GEMMA3N`). Decode follows llama.cpp
/// `src/models/gemma3n.cpp`: Gemma embed-scale + GeGLU, QK-Norm before RoPE,
/// unweighted RMSNorm on V, attention scale `1.0`, RMSNorm `post_attention_norm`
/// / `ffn_norm` / `post_ffw_norm`, SWA default period 5 (`set_swa_pattern`,
/// required `attention.sliding_window`), final tanh softcap (default 30),
/// AltUp (4 residual streams, convert asserts `altup_num_inputs == 4`), Laurel,
/// per-layer inputs, gaussian_topk on the first 10 layers (`n_layer_sparsity`
/// is hardcoded, not GGUF). Convert skips `lm_head.weight` when tied, omits
/// `attn_logit_softcapping` / `rope.dimension_count`. Writer-tiny uses
/// `attention.sliding_window = 2` so short-seq tests clip, omits
/// `sliding_window_pattern` (period 5), and writes convert-shaped
/// `gemma3n.altup.*` / `embedding_length_per_layer_input`. Convert
/// `norm_shift` is 0 (runtime does not add 1). `gemma4` is a separate official
/// family (`architecture=gemma4`).
pub fn tiny_gemma3n_gguf() -> Vec<u8> {
    tiny_arch_gguf_lm_head(
        TinySpec {
            arch: "gemma3n",
            token_embd: GgmlType::F32,
            output: GgmlType::F32,
            layer: None,
            rope_dimension_count: false,
            qkv_bias: false,
            add_bos_token: None,
            llama_moe: false,
            gemma4_moe: false,
            gemma4_ple: false,
            gemma4_fused: false,
        },
        TinyLmHead::Tied,
    )
}

/// Writer-built official Gemma4 GGUF: `architecture=gemma4` with `gemma4.*` KV.
///
/// Official `general.architecture=gemma4` (`MODEL_ARCH_NAMES[GEMMA4] = "gemma4"`,
/// `Gemma4ForCausalLM` → `MODEL_ARCH.GEMMA4`). Decode follows llama.cpp
/// `src/models/gemma4.cpp`: Gemma embed-scale + GeGLU, QK-Norm before RoPE,
/// unweighted RMSNorm on V, attention scale `1.0`, RMSNorm `post_attention_norm`
/// / `ffn_norm` / `post_ffw_norm`, required `attention.sliding_window`, convert
/// `attention.sliding_window_pattern` as a per-layer bool array, required
/// `attention.key_length_swa` / `value_length_swa` and
/// `embedding_length_per_layer_input`. Optional final tanh logit softcap
/// (default 0). Writer-tiny is dense (no `ffn_gate_inp`), writes
/// `embedding_length_per_layer_input = 0` (no per-layer embeddings), omits
/// `attention.shared_kv_layers` (every layer has KV), uses equal SWA/global
/// head dims, and `attention.sliding_window = 2` so short-seq tests clip.
/// Convert skips `lm_head.weight` when tied, omits `attn_logit_softcapping` /
/// `rope.dimension_count`. Convert `norm_shift` is 0 (runtime does not add 1).
/// Official Gemma4 MoE is the same `architecture=gemma4` ([`tiny_gemma4_moe_gguf`]).
/// Official Gemma4 PLE is the same arch ([`tiny_gemma4_ple_gguf`]).
/// Official Gemma4 MoE+PLE is the same arch ([`tiny_gemma4_moe_ple_gguf`]).
/// Official Gemma4 fused experts are the same arch ([`tiny_gemma4_moe_fused_gguf`]).
/// Official Gemma4 fused plus PLE is the same arch ([`tiny_gemma4_moe_fused_ple_gguf`]).
/// Official Gemma4 shared KV is the same arch ([`tiny_gemma4_shared_kv_gguf`]).
pub fn tiny_gemma4_gguf() -> Vec<u8> {
    tiny_arch_gguf_lm_head(
        TinySpec {
            arch: "gemma4",
            token_embd: GgmlType::F32,
            output: GgmlType::F32,
            layer: None,
            rope_dimension_count: false,
            qkv_bias: false,
            add_bos_token: None,
            llama_moe: false,
            gemma4_moe: false,
            gemma4_ple: false,
            gemma4_fused: false,
        },
        TinyLmHead::Tied,
    )
}

/// Writer-built official Gemma4 MoE GGUF: `architecture=gemma4` with `n_expert>0`.
///
/// Same `gemma4.*` KV as [`tiny_gemma4_gguf`], plus `expert_count` /
/// `expert_used_count` / `expert_feed_forward_length`. Decode follows llama.cpp
/// `src/models/gemma4.cpp` when `ffn_gate_inp` is present: shared dense GeGLU
/// (`ffn_gate` / `ffn_up` / `ffn_down`) with `ffn_post_norm_1`; expert input
/// `ffn_pre_norm_2`; custom router on `attn_out` (unweighted RMSNorm, scale
/// `1/sqrt(n_embd)`, `ffn_gate_inp.scale`, then `ffn_gate_inp`); `build_moe_ffn`
/// softmax then top-k with `norm_w` clamp `2^-14` and **GELU** experts; then
/// `ffn_post_norm_2` and add into the shared MLP. Not a second architecture.
/// This tiny writes separate `ffn_gate_exps` / `ffn_up_exps`; fused packing is
/// [`tiny_gemma4_moe_fused_gguf`]. Mixed SWA/global head dims stay refused.
/// Shared KV is [`tiny_gemma4_shared_kv_gguf`].
/// Writer-tiny uses `n_expert=4`, `n_expert_used=2`, `n_ff_exp=n_ff`.
/// PLE is the same arch ([`tiny_gemma4_ple_gguf`]). MoE+PLE is the same arch
/// ([`tiny_gemma4_moe_ple_gguf`]).
pub fn tiny_gemma4_moe_gguf() -> Vec<u8> {
    tiny_arch_gguf_lm_head(
        TinySpec {
            arch: "gemma4",
            token_embd: GgmlType::F32,
            output: GgmlType::F32,
            layer: None,
            rope_dimension_count: false,
            qkv_bias: false,
            add_bos_token: None,
            llama_moe: false,
            gemma4_moe: true,
            gemma4_ple: false,
            gemma4_fused: false,
        },
        TinyLmHead::Tied,
    )
}

/// Writer-built official Gemma4 PLE GGUF: `architecture=gemma4` with
/// `embedding_length_per_layer_input > 0`.
///
/// Same `gemma4.*` KV as [`tiny_gemma4_gguf`] except per-layer width is
/// [`TINY_GEMMA4_N_EMBD_PER_LAYER`]. Decode follows llama.cpp
/// `src/models/gemma4.cpp` when `n_embd_per_layer > 0`: `build_inp_per_layer`
/// (token table scaled by `sqrt(n_embd_per_layer)`), `project_per_layer_inputs`
/// (`mm(per_layer_model_proj, inpL)` scaled by `1/sqrt(n_embd)`, RMSNorm,
/// add, scale `1/sqrt(2)`), then after the FFN residual
/// `gelu(mm(inp_gate, cur)) * slice(il)`, `mm(proj)`, RMSNorm `post_norm`,
/// residual add. No AltUp / Laurel. Not a second architecture. Mixed
/// SWA/global head dims stay refused. Shared KV is
/// [`tiny_gemma4_shared_kv_gguf`]. Writer-tiny is dense (no `ffn_gate_inp`).
/// MoE+PLE is the same arch ([`tiny_gemma4_moe_ple_gguf`]).
pub fn tiny_gemma4_ple_gguf() -> Vec<u8> {
    tiny_arch_gguf_lm_head(
        TinySpec {
            arch: "gemma4",
            token_embd: GgmlType::F32,
            output: GgmlType::F32,
            layer: None,
            rope_dimension_count: false,
            qkv_bias: false,
            add_bos_token: None,
            llama_moe: false,
            gemma4_moe: false,
            gemma4_ple: true,
            gemma4_fused: false,
        },
        TinyLmHead::Tied,
    )
}

/// Writer-built official Gemma4 MoE+PLE GGUF: `architecture=gemma4` with
/// `ffn_gate_inp` and `embedding_length_per_layer_input > 0`.
///
/// Production E2B/E4B shape on the same arch as [`tiny_gemma4_gguf`],
/// [`tiny_gemma4_moe_gguf`], and [`tiny_gemma4_ple_gguf`]. Not a second
/// family. Decode follows llama.cpp `src/models/gemma4.cpp`: shared GeGLU plus
/// custom-router GELU experts, then after the FFN residual the PLE inject
/// (`gelu(mm(inp_gate)) * slice(il)`, `mm(proj)`, RMSNorm `post_norm`). No
/// AltUp / Laurel. This tiny writes separate gate/up experts. Mixed
/// SWA/global head dims stay refused. Shared KV is
/// [`tiny_gemma4_shared_kv_gguf`]. Writer-tiny uses `n_expert=4`,
/// `n_expert_used=2`, `n_embd_per_layer=64`.
pub fn tiny_gemma4_moe_ple_gguf() -> Vec<u8> {
    tiny_arch_gguf_lm_head(
        TinySpec {
            arch: "gemma4",
            token_embd: GgmlType::F32,
            output: GgmlType::F32,
            layer: None,
            rope_dimension_count: false,
            qkv_bias: false,
            add_bos_token: None,
            llama_moe: false,
            gemma4_moe: true,
            gemma4_ple: true,
            gemma4_fused: false,
        },
        TinyLmHead::Tied,
    )
}

/// Writer-built official Gemma4 fused-expert GGUF: `architecture=gemma4` with
/// `ffn_gate_up_exps` instead of separate `ffn_gate_exps` / `ffn_up_exps`.
///
/// Same MoE walk as [`tiny_gemma4_moe_gguf`]. Official `src/models/gemma4.cpp`
/// prefers fused `ffn_gate_up_exps` (twice `n_ff` rows, gate then up) when
/// present. Not a second architecture. Mixed SWA/global head dims stay
/// refused. Shared KV is [`tiny_gemma4_shared_kv_gguf`]. Fused plus PLE is
/// the same arch ([`tiny_gemma4_moe_fused_ple_gguf`]).
pub fn tiny_gemma4_moe_fused_gguf() -> Vec<u8> {
    tiny_arch_gguf_lm_head(
        TinySpec {
            arch: "gemma4",
            token_embd: GgmlType::F32,
            output: GgmlType::F32,
            layer: None,
            rope_dimension_count: false,
            qkv_bias: false,
            add_bos_token: None,
            llama_moe: false,
            gemma4_moe: true,
            gemma4_ple: false,
            gemma4_fused: true,
        },
        TinyLmHead::Tied,
    )
}

/// Writer-built official Gemma4 fused-expert plus PLE GGUF: `architecture=gemma4`
/// with `ffn_gate_up_exps` and `embedding_length_per_layer_input > 0`.
///
/// Production E2B/E4B packing on the same arch as [`tiny_gemma4_moe_fused_gguf`]
/// and [`tiny_gemma4_moe_ple_gguf`]. Decode is fused GELU experts then PLE
/// inject after the FFN residual. Not a second architecture.
/// Mixed SWA/global head dims stay refused. Writer-tiny uses `n_expert=4`,
/// `n_expert_used=2`, `n_embd_per_layer=64`.
pub fn tiny_gemma4_moe_fused_ple_gguf() -> Vec<u8> {
    tiny_arch_gguf_lm_head(
        TinySpec {
            arch: "gemma4",
            token_embd: GgmlType::F32,
            output: GgmlType::F32,
            layer: None,
            rope_dimension_count: false,
            qkv_bias: false,
            add_bos_token: None,
            llama_moe: false,
            gemma4_moe: true,
            gemma4_ple: true,
            gemma4_fused: true,
        },
        TinyLmHead::Tied,
    )
}

/// Writer-built official Gemma4 shared-KV GGUF: `architecture=gemma4` with
/// `attention.shared_kv_layers=1`.
///
/// Same dense `gemma4.*` KV as [`tiny_gemma4_gguf`] except three layers, all
/// SWA, `n_layer_kv_from_start = 2`. Decode follows llama.cpp
/// `src/models/gemma4.cpp` plus `llama_model::create_memory` reuse:
/// `has_kv(il) = il < n_layer_kv_from_start`; shared layers skip K/V project
/// and store, then `build_attn` against the donor
/// `n_layer_kv_from_start` minus 2 (SWA) or minus 1 (global). Layer 2 is SWA
/// so it reuses layer 0. `wk` / `wv` / `attn_k_norm` are omitted on the
/// shared layer (`TENSOR_NOT_REQUIRED`). Not a second architecture. Mixed
/// SWA/global head dims stay refused. 1-layer tinies stay full-KV.
pub fn tiny_gemma4_shared_kv_gguf() -> Vec<u8> {
    expand_tiny_gemma4_layers(
        tiny_gemma4_gguf(),
        TINY_GEMMA4_SHARED_N_LAYER,
        TINY_GEMMA4_SHARED_KV_LAYERS,
    )
    .unwrap_or_else(|_| Vec::new())
}

/// Clone `blk.0.*` onto `blk.1..n_layer-1`, set `block_count`, expand the SWA
/// pattern, and omit K/V tensors on shared layers when `shared_kv > 0`.
fn expand_tiny_gemma4_layers(
    bytes: Vec<u8>,
    n_layer: usize,
    shared_kv: u32,
) -> Result<Vec<u8>, GgufError> {
    let g = load_gguf_owned(bytes)?;
    let n_layer_u32 = u32::try_from(n_layer).unwrap_or(0);
    let n_from_start = n_layer.saturating_sub(usize::try_from(shared_kv).unwrap_or(0));
    let mut kv: Vec<(String, Kv)> = g.kv.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    for (k, v) in &mut kv {
        if k.ends_with(".block_count") {
            *v = Kv::U32(n_layer_u32);
        }
        if k.ends_with(".attention.sliding_window_pattern") {
            *v = Kv::Array {
                elem: GGUF_TYPE_BOOL,
                items: vec![Kv::Bool(true); n_layer],
            };
        }
    }
    if shared_kv != 0 {
        kv.push((
            "gemma4.attention.shared_kv_layers".into(),
            Kv::U32(shared_kv),
        ));
    }
    let mut tensors = Vec::new();
    let mut extra = Vec::new();
    for t in g.tensors() {
        let shape = t.shape.to_vec();
        let data = t.data.to_vec();
        if let Some(rest) = t.name.strip_prefix("blk.0.") {
            for li in 1..n_layer {
                if li >= n_from_start && gemma4_shared_kv_omitted(rest) {
                    continue;
                }
                extra.push(TensorWrite {
                    name: format!("blk.{li}.{rest}"),
                    ty: t.ty,
                    shape: shape.clone(),
                    data: data.clone(),
                });
            }
        }
        tensors.push(TensorWrite {
            name: t.name.to_string(),
            ty: t.ty,
            shape,
            data,
        });
    }
    tensors.extend(extra);
    Ok(write_gguf_with_kv(&kv, &tensors))
}

/// Writer-built Qwen3-shaped GGUF: `architecture=qwen3` with `qwen3.*` KV.
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
    })
}

/// Writer-built two-layer Qwen3MoE tiny (`qwen3moe.block_count=2`).
///
/// Layer 1 is a clone of layer 0 tensor bytes so copy-forward L+1 keys exist
/// in the ExpertStore catalog. 1-layer tinies skip those keys as unknown.
pub fn tiny_qwen3moe_2layer_gguf() -> Vec<u8> {
    clone_tiny_blk0_as_next_layer(tiny_qwen3moe_gguf()).unwrap_or_else(|_| Vec::new())
}

/// Clone every `blk.0.*` tensor as `blk.1.*` and set `*.block_count` to 2.
fn clone_tiny_blk0_as_next_layer(bytes: Vec<u8>) -> Result<Vec<u8>, GgufError> {
    let g = load_gguf_owned(bytes)?;
    let mut kv: Vec<(String, Kv)> = g.kv.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    for (k, v) in &mut kv {
        if k.ends_with(".block_count") {
            *v = Kv::U32(2);
        }
    }
    let mut tensors = Vec::new();
    let mut extra = Vec::new();
    for t in g.tensors() {
        let shape = t.shape.to_vec();
        let data = t.data.to_vec();
        if let Some(rest) = t.name.strip_prefix("blk.0.") {
            extra.push(TensorWrite {
                name: format!("blk.1.{rest}"),
                ty: t.ty,
                shape: shape.clone(),
                data: data.clone(),
            });
        }
        tensors.push(TensorWrite {
            name: t.name.to_string(),
            ty: t.ty,
            shape,
            data,
        });
    }
    tensors.extend(extra);
    Ok(write_gguf_with_kv(&kv, &tensors))
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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

/// Writer-built official Bloom GGUF: `architecture=bloom` with `bloom.*` KV.
///
/// Official `LLM_ARCH_NAMES` has `LLM_ARCH_BLOOM = "bloom"`; convert writes
/// `general.architecture=bloom` (`BloomForCausalLM` / `BloomModel` →
/// `MODEL_ARCH.BLOOM`). Decode follows llama.cpp `src/models/bloom.cpp`:
/// `token_embd_norm` LayerNorm, fused `attn_qkv` (convert restacks HF
/// interleaved QKV to concatenated Q/K/V), `LLM_NORM` on attn/ffn/output,
/// sequential residual, `LLM_FFN_GELU`/`LLM_FFN_SEQ` with biases, ALiBi
/// (`f_max_alibi_bias = 8`, hardcoded; no RoPE). Official convert writes
/// `layer_norm_epsilon`, `feed_forward_length = 4 * n_embed`,
/// `head_count_kv = n_head`, `context_length` (`seq_length` else `n_embed`),
/// and `add_bos_token=false`. No `rope.dimension_count` / `rope.freq_base`.
/// No `output.bias`. Tied `output.weight` reuse is allowed. Not Mixtral, not
/// `qwen3vlmoe`, not linear-attn, not a phi2 redo.
pub fn tiny_bloom_gguf() -> Vec<u8> {
    let n_embd = TINY_N_EMBD;
    let n_ff = TINY_BLOOM_N_FF;
    let n_vocab = TINY_N_VOCAB;
    let n_head = TINY_N_HEAD;
    let n_kv = TINY_BLOOM_N_HEAD_KV.saturating_mul(n_embd / n_head);
    let n_qkv = n_embd.saturating_add(n_kv.saturating_mul(2));
    let ones = vec![1.0f32; n_embd];
    let mut qkv_w = pack_mat(GgmlType::Q4_K, n_embd, n_embd, 5);
    qkv_w.extend(pack_mat(GgmlType::Q4_K, n_embd, n_kv, 3));
    qkv_w.extend(pack_mat(GgmlType::Q4_K, n_embd, n_kv, 4));
    let mut qkv_b = pat_f32(n_embd, 11);
    qkv_b.extend(pat_f32(n_kv, 12));
    qkv_b.extend(pat_f32(n_kv, 13));
    let kv = vec![
        (
            "general.alignment".into(),
            Kv::U32(u32::try_from(GGUF_DEFAULT_ALIGNMENT).unwrap_or(32)),
        ),
        ("general.name".into(), Kv::String("llama-rust-tiny".into())),
        ("general.architecture".into(), Kv::String("bloom".into())),
        (
            "bloom.block_count".into(),
            Kv::U32(u32::try_from(TINY_N_LAYER).unwrap_or(0)),
        ),
        (
            "bloom.embedding_length".into(),
            Kv::U32(u32::try_from(n_embd).unwrap_or(0)),
        ),
        (
            "bloom.feed_forward_length".into(),
            Kv::U32(u32::try_from(n_ff).unwrap_or(0)),
        ),
        (
            "bloom.attention.head_count".into(),
            Kv::U32(u32::try_from(n_head).unwrap_or(0)),
        ),
        (
            "bloom.attention.head_count_kv".into(),
            Kv::U32(u32::try_from(TINY_BLOOM_N_HEAD_KV).unwrap_or(0)),
        ),
        (
            "bloom.context_length".into(),
            Kv::U32(u32::try_from(TINY_N_EMBD).unwrap_or(0)),
        ),
        (
            "bloom.attention.layer_norm_epsilon".into(),
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
            "token_embd_norm.weight",
            GgmlType::F32,
            vec![n_embd],
            pack_vec1d(GgmlType::F32, &ones),
        ),
        tw(
            "token_embd_norm.bias",
            GgmlType::F32,
            vec![n_embd],
            pack_vec1d(GgmlType::F32, &pat_f32(n_embd, 20)),
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
            "blk.0.attn_qkv.weight",
            GgmlType::Q4_K,
            vec![n_embd, n_qkv],
            qkv_w,
        ),
        tw(
            "blk.0.attn_qkv.bias",
            GgmlType::F32,
            vec![n_qkv],
            pack_vec1d(GgmlType::F32, &qkv_b),
        ),
        tw(
            "blk.0.attn_output.weight",
            GgmlType::Q4_K,
            vec![n_embd, n_embd],
            pack_mat(GgmlType::Q4_K, n_embd, n_embd, 6),
        ),
        tw(
            "blk.0.attn_output.bias",
            GgmlType::F32,
            vec![n_embd],
            pack_vec1d(GgmlType::F32, &pat_f32(n_embd, 14)),
        ),
        tw(
            "blk.0.ffn_norm.weight",
            GgmlType::F32,
            vec![n_embd],
            pack_vec1d(GgmlType::F32, &ones),
        ),
        tw(
            "blk.0.ffn_norm.bias",
            GgmlType::F32,
            vec![n_embd],
            pack_vec1d(GgmlType::F32, &pat_f32(n_embd, 21)),
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
            gemma4_moe: false,
            gemma4_ple: false,
            gemma4_fused: false,
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
            gemma4_moe: false,
            gemma4_ple: false,
            gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
    })
}

/// Writer-built Llama GGUF with no grouping: `head_count_kv == head_count == 4`.
///
/// GQA ratio 1 (plain multi-head attention). Every other fixture uses ratio 2.
pub fn tiny_mha_gguf() -> Vec<u8> {
    tiny_arch_gguf_gqa(tiny_llama_spec(), TinyLmHead::Distinct, GgmlType::F32, 4)
}

/// Writer-built Llama GGUF with a single KV head: `head_count_kv = 1`.
///
/// GQA ratio 4 (multi-query attention), the opposite extreme from
/// [`tiny_mha_gguf`]. Real checkpoints sit between these: Qwen2.5-0.5B is 14/2.
pub fn tiny_mqa_gguf() -> Vec<u8> {
    tiny_arch_gguf_gqa(tiny_llama_spec(), TinyLmHead::Distinct, GgmlType::F32, 1)
}

/// Writer-built Llama GGUF with Q4_0 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_Q4_0` = 2 (18-byte `block_q4_0`).
pub fn tiny_q40_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::Q4_0,
        output: GgmlType::Q4_0,
        layer: Some(GgmlType::Q4_0),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
        llama_moe: false,
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
    })
}

/// Writer-built Llama GGUF with Q8_0 2-D weights (token_embd, output, attn/ffn).
///
/// 1-D norms stay F32. `GGML_TYPE_Q8_0` = 8 (34-byte `block_q8_0`).
pub fn tiny_q80_gguf() -> Vec<u8> {
    tiny_arch_gguf(TinySpec {
        arch: "llama",
        token_embd: GgmlType::Q8_0,
        output: GgmlType::Q8_0,
        layer: Some(GgmlType::Q8_0),
        rope_dimension_count: true,
        qkv_bias: false,
        add_bos_token: None,
        llama_moe: false,
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
        gemma4_moe: false,
        gemma4_ple: false,
        gemma4_fused: false,
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
    /// Official gemma4 MoE: `architecture=gemma4` with `ffn_gate_inp` (not a second arch).
    /// Combines with `gemma4_ple` for production E2B/E4B.
    gemma4_moe: bool,
    /// Official gemma4 PLE: `embedding_length_per_layer_input > 0` (not a second arch).
    /// Combines with `gemma4_moe` for production E2B/E4B.
    gemma4_ple: bool,
    /// Official gemma4 fused `ffn_gate_up_exps` instead of separate gate/up.
    gemma4_fused: bool,
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
    tiny_arch_gguf_gqa(spec, lm_head, vec1d, TINY_N_HEAD_KV)
}

/// Writer-built tiny with an explicit `attention.head_count_kv`.
///
/// Every other fixture pins `head_count_kv = 2` against `head_count = 4`, so the
/// suite only ever exercised a GQA ratio of 2. Real checkpoints use other ratios
/// (Qwen2.5-0.5B is 14/2 = 7), and a grouping bug at any other ratio would pass
/// the whole harness.
fn tiny_arch_gguf_gqa(
    spec: TinySpec,
    lm_head: TinyLmHead,
    vec1d: GgmlType,
    n_head_kv: usize,
) -> Vec<u8> {
    let n_embd = TINY_N_EMBD;
    let n_ff = TINY_N_FF;
    let n_vocab = TINY_N_VOCAB;
    let n_kv = n_head_kv * TINY_N_ROT;
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
        if spec.arch == "gemma2"
            || spec.arch == "gemma3"
            || spec.arch == "gemma3n"
            || spec.arch == "gemma4"
        {
            // Official gemma2.cpp: `post_attention_norm` AND `ffn_norm` AND
            // `post_ffw_norm`. Not the qwen3next reuse of post_attention_norm
            // as pre-FFN.
            tensors.push(tw(
                "blk.0.post_attention_norm.weight",
                vec1d,
                vec![n_embd],
                pack_vec1d(vec1d, &ones),
            ));
            tensors.push(tw(
                "blk.0.post_ffw_norm.weight",
                vec1d,
                vec![n_embd],
                pack_vec1d(vec1d, &ones),
            ));
        }
    }
    if spec.arch == "qwen3"
        || spec.arch == "qwen3moe"
        || spec.arch == "qwen3vl"
        || spec.arch == "qwen3next"
        || spec.arch == "qwen35"
        || spec.arch == "gemma3"
        || spec.arch == "gemma3n"
        || spec.arch == "gemma4"
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
        if spec.gemma4_moe {
            // Official gemma4.cpp MoE: dense shared MLP stays, plus router /
            // extra norms / GELU `*_exps`. Separate gate/up unless
            // `gemma4_fused` writes `ffn_gate_up_exps`.
            tensors.push(tw(
                "blk.0.ffn_gate_inp.weight",
                GgmlType::F32,
                vec![n_embd, TINY_GEMMA4_N_EXPERT],
                pack_mat(GgmlType::F32, n_embd, TINY_GEMMA4_N_EXPERT, 14),
            ));
            tensors.push(tw(
                "blk.0.ffn_gate_inp.scale",
                GgmlType::F32,
                vec![n_embd],
                pack_vec1d(GgmlType::F32, &ones),
            ));
            tensors.push(tw(
                "blk.0.ffn_post_norm_1.weight",
                vec1d,
                vec![n_embd],
                pack_vec1d(vec1d, &ones),
            ));
            tensors.push(tw(
                "blk.0.ffn_pre_norm_2.weight",
                vec1d,
                vec![n_embd],
                pack_vec1d(vec1d, &ones),
            ));
            tensors.push(tw(
                "blk.0.ffn_post_norm_2.weight",
                vec1d,
                vec![n_embd],
                pack_vec1d(vec1d, &ones),
            ));
            if spec.gemma4_fused {
                // F32 plus a distinct seed so fused logits cannot match split
                // Q4_K gate/up (Q4_K GELU(gate) on the tiny can be ~0 so the
                // up-half Q4_K difference would not move the 6 logits).
                tensors.push(layer_tw_exps(
                    &spec,
                    "blk.0.ffn_gate_up_exps.weight",
                    n_embd,
                    n_ff.saturating_mul(2),
                    TINY_GEMMA4_N_EXPERT,
                    23,
                    GgmlType::F32,
                ));
            } else {
                tensors.push(layer_tw_exps(
                    &spec,
                    "blk.0.ffn_gate_exps.weight",
                    n_embd,
                    n_ff,
                    TINY_GEMMA4_N_EXPERT,
                    15,
                    GgmlType::Q4_K,
                ));
                tensors.push(layer_tw_exps(
                    &spec,
                    "blk.0.ffn_up_exps.weight",
                    n_embd,
                    n_ff,
                    TINY_GEMMA4_N_EXPERT,
                    17,
                    GgmlType::Q4_K,
                ));
            }
            tensors.push(layer_tw_exps(
                &spec,
                "blk.0.ffn_down_exps.weight",
                n_ff,
                n_embd,
                TINY_GEMMA4_N_EXPERT,
                19,
                GgmlType::Q4_K,
            ));
        }
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
    if spec.arch == "gemma3n" {
        push_tiny_gemma3n_tensors(&spec, vec1d, n_embd, n_vocab, &ones, &mut tensors);
    }
    if spec.gemma4_ple {
        push_tiny_gemma4_ple_tensors(&spec, vec1d, n_embd, n_vocab, &ones, &mut tensors);
    }
    write_gguf_with_kv(&tiny_kv_gqa(&spec, n_head_kv), &tensors)
}

fn push_tiny_gemma3n_tensors(
    spec: &TinySpec,
    vec1d: GgmlType,
    n_embd: usize,
    n_vocab: usize,
    ones: &[f32],
    tensors: &mut Vec<TensorWrite>,
) {
    let n_altup = GEMMA3N_N_ALTUP;
    let extra = n_altup.saturating_sub(1);
    let n_embd_altup = GEMMA3N_N_EMBD_ALTUP;
    let rank = GEMMA3N_LAUREL_RANK;
    let pl_cols = n_embd_altup.saturating_mul(TINY_N_LAYER);
    let altup_ones = vec![1.0f32; n_embd_altup];
    tensors.push(layer_tw_exps(
        spec,
        "altup_proj.weight",
        n_embd,
        n_embd,
        extra,
        40,
        GgmlType::F32,
    ));
    tensors.push(layer_tw_exps(
        spec,
        "altup_unembd_proj.weight",
        n_embd,
        n_embd,
        extra,
        41,
        GgmlType::F32,
    ));
    tensors.push(tw(
        "per_layer_token_embd.weight",
        spec.token_embd,
        vec![pl_cols, n_vocab],
        pack_mat(spec.token_embd, pl_cols, n_vocab, 42),
    ));
    tensors.push(layer_tw(
        spec,
        "per_layer_model_proj.weight",
        n_embd,
        pl_cols,
        43,
        GgmlType::F32,
    ));
    tensors.push(tw(
        "per_layer_proj_norm.weight",
        vec1d,
        vec![n_embd_altup],
        pack_vec1d(vec1d, &altup_ones),
    ));
    tensors.push(layer_tw(
        spec,
        "blk.0.inp_gate.weight",
        n_embd,
        n_embd_altup,
        44,
        GgmlType::F32,
    ));
    tensors.push(layer_tw(
        spec,
        "blk.0.proj.weight",
        n_embd_altup,
        n_embd,
        45,
        GgmlType::F32,
    ));
    tensors.push(tw(
        "blk.0.post_norm.weight",
        vec1d,
        vec![n_embd],
        pack_vec1d(vec1d, ones),
    ));
    tensors.push(layer_tw(
        spec,
        "blk.0.altup_correct_coef.weight",
        n_altup,
        n_altup,
        46,
        GgmlType::F32,
    ));
    tensors.push(tw(
        "blk.0.altup_correct_scale.weight",
        vec1d,
        vec![n_embd],
        pack_vec1d(vec1d, ones),
    ));
    tensors.push(layer_tw(
        spec,
        "blk.0.altup_predict_coef.weight",
        n_altup,
        n_altup.saturating_mul(n_altup),
        47,
        GgmlType::F32,
    ));
    tensors.push(layer_tw(
        spec,
        "blk.0.altup_router.weight",
        n_embd,
        n_altup,
        48,
        GgmlType::F32,
    ));
    tensors.push(tw(
        "blk.0.altup_router_norm.weight",
        vec1d,
        vec![n_embd],
        pack_vec1d(vec1d, ones),
    ));
    tensors.push(layer_tw(
        spec,
        "blk.0.laurel_l.weight",
        n_embd,
        rank,
        49,
        GgmlType::F32,
    ));
    tensors.push(layer_tw(
        spec,
        "blk.0.laurel_r.weight",
        rank,
        n_embd,
        50,
        GgmlType::F32,
    ));
    tensors.push(tw(
        "blk.0.laurel_post_norm.weight",
        vec1d,
        vec![n_embd],
        pack_vec1d(vec1d, ones),
    ));
}

fn push_tiny_gemma4_ple_tensors(
    spec: &TinySpec,
    vec1d: GgmlType,
    n_embd: usize,
    n_vocab: usize,
    ones: &[f32],
    tensors: &mut Vec<TensorWrite>,
) {
    let n_pl = TINY_GEMMA4_N_EMBD_PER_LAYER;
    let pl_cols = n_pl.saturating_mul(TINY_N_LAYER);
    let pl_ones = vec![1.0f32; n_pl];
    tensors.push(tw(
        "per_layer_token_embd.weight",
        spec.token_embd,
        vec![pl_cols, n_vocab],
        pack_mat(spec.token_embd, pl_cols, n_vocab, 42),
    ));
    tensors.push(layer_tw(
        spec,
        "per_layer_model_proj.weight",
        n_embd,
        pl_cols,
        43,
        GgmlType::F32,
    ));
    tensors.push(tw(
        "per_layer_proj_norm.weight",
        vec1d,
        vec![n_pl],
        pack_vec1d(vec1d, &pl_ones),
    ));
    tensors.push(layer_tw(
        spec,
        "blk.0.inp_gate.weight",
        n_embd,
        n_pl,
        44,
        GgmlType::F32,
    ));
    tensors.push(layer_tw(
        spec,
        "blk.0.proj.weight",
        n_pl,
        n_embd,
        45,
        GgmlType::F32,
    ));
    tensors.push(tw(
        "blk.0.post_norm.weight",
        vec1d,
        vec![n_embd],
        pack_vec1d(vec1d, ones),
    ));
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
        || s == "gemma2"
        || s == "gemma3"
        || s == "gemma3n"
        || s == "gemma4"
        || s == "qwen3"
        || s == "llama4"
        || s == "qwen2moe"
        || s == "qwen3moe"
        || s == "qwen2vl"
        || s == "qwen3vl"
        || s == "qwen3next"
        || s == "qwen35"
        || s == "phi2"
        || s == "bloom"
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

/// Default-GQA convenience wrapper. Only tests build KV without going through
/// [`tiny_arch_gguf_gqa`].
#[cfg(test)]
fn tiny_kv(spec: &TinySpec) -> Vec<(String, Kv)> {
    tiny_kv_gqa(spec, TINY_N_HEAD_KV)
}

fn tiny_kv_gqa(spec: &TinySpec, n_head_kv: usize) -> Vec<(String, Kv)> {
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
            Kv::U32(u32::try_from(n_head_kv).unwrap_or(0)),
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
    if arch == "gemma2" || arch == "gemma3" || arch == "gemma3n" || arch == "gemma4" {
        // Official convert/gemma.py Gemma2Model / Gemma3Model / Gemma3NModel / Gemma4Model.
        if arch == "gemma2" {
            kv.push((
                arch_key(arch, "attn_logit_softcapping"),
                Kv::F32(GEMMA2_ATTN_LOGIT_SOFTCAPPING),
            ));
            kv.push((
                arch_key(arch, "final_logit_softcapping"),
                Kv::F32(GEMMA2_FINAL_LOGIT_SOFTCAPPING),
            ));
        }
        if arch == "gemma3n" {
            kv.push((
                arch_key(arch, "final_logit_softcapping"),
                Kv::F32(GEMMA2_FINAL_LOGIT_SOFTCAPPING),
            ));
            kv.push((
                arch_key(arch, "altup.active_idx"),
                Kv::U32(u32::try_from(GEMMA3N_I_ALTUP_ACT).unwrap_or(0)),
            ));
            kv.push((
                arch_key(arch, "altup.num_inputs"),
                Kv::U32(u32::try_from(GEMMA3N_N_ALTUP).unwrap_or(0)),
            ));
            kv.push((
                arch_key(arch, "embedding_length_per_layer_input"),
                Kv::U32(u32::try_from(GEMMA3N_N_EMBD_ALTUP).unwrap_or(0)),
            ));
            kv.push((
                arch_key(arch, "attention.shared_kv_layers"),
                Kv::U32(GEMMA3N_N_LAYER_KV_FROM_START),
            ));
        }
        if arch == "gemma4" {
            // Official gemma4.cpp `get_key` without `false` for SWA head dim and
            // per-layer embedding width. Writer-tiny dense/MoE keep PLE off;
            // [`tiny_gemma4_ple_gguf`], [`tiny_gemma4_moe_ple_gguf`], and
            // [`tiny_gemma4_moe_fused_ple_gguf`] write `n_embd_per_layer > 0`.
            let n_pl = if spec.gemma4_ple {
                TINY_GEMMA4_N_EMBD_PER_LAYER
            } else {
                0
            };
            kv.push((
                arch_key(arch, "embedding_length_per_layer_input"),
                Kv::U32(u32::try_from(n_pl).unwrap_or(0)),
            ));
            kv.push((
                arch_key(arch, "attention.key_length_swa"),
                Kv::U32(u32::try_from(TINY_N_ROT).unwrap_or(0)),
            ));
            kv.push((
                arch_key(arch, "attention.value_length_swa"),
                Kv::U32(u32::try_from(TINY_N_ROT).unwrap_or(0)),
            ));
            kv.push((
                arch_key(arch, "attention.sliding_window_pattern"),
                Kv::Array {
                    elem: GGUF_TYPE_BOOL,
                    items: vec![Kv::Bool(true); TINY_N_LAYER],
                },
            ));
            if spec.gemma4_moe {
                kv.push((
                    arch_key(arch, "expert_count"),
                    Kv::U32(u32::try_from(TINY_GEMMA4_N_EXPERT).unwrap_or(0)),
                ));
                kv.push((
                    arch_key(arch, "expert_used_count"),
                    Kv::U32(u32::try_from(TINY_GEMMA4_N_EXPERT_USED).unwrap_or(0)),
                ));
                kv.push((
                    arch_key(arch, "expert_feed_forward_length"),
                    Kv::U32(u32::try_from(TINY_N_FF).unwrap_or(0)),
                ));
            }
        }
        kv.push((
            arch_key(arch, "attention.sliding_window"),
            Kv::U32(GEMMA2_TINY_N_SWA),
        ));
        kv.push((
            arch_key(arch, "attention.key_length"),
            Kv::U32(u32::try_from(TINY_N_ROT).unwrap_or(0)),
        ));
        kv.push((
            arch_key(arch, "attention.value_length"),
            Kv::U32(u32::try_from(TINY_N_ROT).unwrap_or(0)),
        ));
        kv.push((
            arch_key(arch, "context_length"),
            Kv::U32(u32::try_from(TINY_N_EMBD).unwrap_or(0)),
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
        GgmlType::Q4_0 => pack_q40_mat(n_cols, n_rows, seed),
        GgmlType::Q8_0 => pack_q80_mat(n_cols, n_rows, seed),
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

fn pack_q40_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    let nblocks = n_cols / QK4_0;
    for _ in 0..n_rows {
        for _ in 0..nblocks {
            // `pack_q4_0_block` takes the packed nibbles: byte `j` holds element
            // `j` in the low nibble and element `j + 16` in the high nibble.
            let mut packed = [0u8; QK4_0 / 2];
            for byte in &mut packed {
                s = s.wrapping_mul(1_664_525).wrapping_add(1);
                let lo = u8::try_from(s % 16).unwrap_or(0);
                s = s.wrapping_mul(1_664_525).wrapping_add(1);
                let hi = u8::try_from(s % 16).unwrap_or(0);
                *byte = lo | (hi << 4);
            }
            out.extend_from_slice(&pack_q4_0_block(25.0 / 100.0, &packed));
        }
    }
    out
}

fn pack_q80_mat(n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut s = seed;
    let nblocks = n_cols / QK8_0;
    for _ in 0..n_rows {
        for _ in 0..nblocks {
            let mut qs = [0i8; QK8_0];
            for q in &mut qs {
                s = s.wrapping_mul(1_664_525).wrapping_add(1);
                let k = u8::try_from(s % 17).unwrap_or(0);
                *q = i8::try_from(i32::from(k).saturating_sub(8)).unwrap_or(0);
            }
            out.extend_from_slice(&pack_q8_0_block(25.0 / 100.0, &qs));
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
    let nblocks = n_cols / QK_K;
    for _ in 0..n_rows {
        for _ in 0..nblocks {
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
        }
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

/// Official gemma4.cpp MoE: `n_expert` / `n_expert_used` when `ffn_gate_inp` is present.
struct Gemma4MoeHparams {
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

/// Official gemma4.cpp: dense when `expert_count` is missing or 0. MoE layers
/// require `ffn_gate_inp` plus `expert_used_count`.
fn load_gemma4_moe_hparams(g: &Gguf, arch: &str) -> Result<Option<Gemma4MoeHparams>, LlamaError> {
    let key = arch_key(arch, "expert_count");
    let n_expert = match g.kv_u32(&key) {
        None => return Ok(None),
        Some(0) => return Ok(None),
        Some(v) => usize::try_from(v).map_err(|_| LlamaError::Shape(key))?,
    };
    let n_expert_used = require_usize(g, arch, "expert_used_count")?;
    if n_expert_used == 0 || n_expert_used > n_expert {
        return Err(LlamaError::Shape(arch_key(arch, "expert_used_count")));
    }
    let n_ff = require_usize(g, arch, "feed_forward_length")?;
    if n_ff == 0 {
        return Err(LlamaError::Shape(arch_key(arch, "feed_forward_length")));
    }
    let n_ff_exp = match g.kv_u32(&arch_key(arch, "expert_feed_forward_length")) {
        Some(v) if v > 0 => usize::try_from(v)
            .map_err(|_| LlamaError::Shape(arch_key(arch, "expert_feed_forward_length")))?,
        _ => n_ff,
    };
    Ok(Some(Gemma4MoeHparams {
        n_expert,
        n_expert_used,
        n_ff_exp,
    }))
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
    bloom: bool,
    gemma2: bool,
    gemma3: bool,
    gemma3n: bool,
    gemma4: bool,
    gemma4_n_pl: usize,
    n_layer_kv_from_start: i32,
    is_swa: &'a [bool],
    gemma4_moe: Option<&'a Gemma4MoeHparams>,
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
    let use_rope = if h.bloom {
        false
    } else {
        match llama4 {
            Some(h) => llama4_use_rope(i, h.n_no_rope_layer_step),
            None => true,
        }
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
    } else if h.gemma4 {
        if g.tensor(&format!("blk.{i}.ffn_gate_inp.weight")).is_some() {
            let hp = h
                .gemma4_moe
                .ok_or_else(|| LlamaError::Shape(arch_key("gemma4", "expert_count")))?;
            LayerFfn::Gemma4Moe(Box::new(load_gemma4_moe(g, i, hp)?))
        } else {
            LayerFfn::Dense(Box::new(DenseFfn {
                gate: quant_mat(need(g, &format!("blk.{i}.ffn_gate.weight"))?)?,
                up: quant_mat(need(g, &format!("blk.{i}.ffn_up.weight"))?)?,
                down: quant_mat(need(g, &format!("blk.{i}.ffn_down.weight"))?)?,
            }))
        }
    } else if h.phi2 || h.bloom {
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
    let has_kv = gemma4_has_kv(h.n_layer_kv_from_start, i);
    let layer_swa = h.is_swa.get(i).copied().unwrap_or(false);
    let kv_slot = if has_kv {
        i
    } else {
        gemma4_kv_slot(h.n_layer_kv_from_start, i, layer_swa)?
    };
    let (wq, bq, wk, bk, wv, bv) = if h.bloom {
        let qkv = load_bloom_qkv(g, i)?;
        (qkv.wq, qkv.bq, Some(qkv.wk), qkv.bk, Some(qkv.wv), qkv.bv)
    } else if has_kv {
        (
            quant_mat(need(g, &format!("blk.{i}.attn_q.weight"))?)?,
            optional_f32(g, &format!("blk.{i}.attn_q.bias"))?,
            Some(quant_mat(need(g, &format!("blk.{i}.attn_k.weight"))?)?),
            optional_f32(g, &format!("blk.{i}.attn_k.bias"))?,
            Some(quant_mat(need(g, &format!("blk.{i}.attn_v.weight"))?)?),
            optional_f32(g, &format!("blk.{i}.attn_v.bias"))?,
        )
    } else {
        (
            quant_mat(need(g, &format!("blk.{i}.attn_q.weight"))?)?,
            optional_f32(g, &format!("blk.{i}.attn_q.bias"))?,
            None,
            None,
            None,
            None,
        )
    };
    Ok(Layer {
        attn_norm: f32s(need(g, &format!("blk.{i}.attn_norm.weight"))?)?,
        attn_norm_b: if h.phi2 || h.bloom {
            Some(f32s(need(g, &format!("blk.{i}.attn_norm.bias"))?)?)
        } else {
            None
        },
        wq,
        bq,
        wk,
        bk,
        wv,
        bv,
        kv_slot,
        wo: quant_mat(need(g, &format!("blk.{i}.attn_output.weight"))?)?,
        wo_b: if h.phi2 || h.bloom {
            Some(f32s(need(g, &format!("blk.{i}.attn_output.bias"))?)?)
        } else {
            None
        },
        attn_q_norm: if qk_norm {
            Some(f32s(need(g, &format!("blk.{i}.attn_q_norm.weight"))?)?)
        } else {
            None
        },
        attn_k_norm: if qk_norm && has_kv {
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
        ffn_norm_b: if h.bloom {
            Some(f32s(need(g, &format!("blk.{i}.ffn_norm.bias"))?)?)
        } else {
            None
        },
        attn_post_norm: if h.gemma2 || h.gemma3 || h.gemma3n || h.gemma4 {
            Some(f32s(need(
                g,
                &format!("blk.{i}.post_attention_norm.weight"),
            )?)?)
        } else {
            None
        },
        ffn_post_norm: if h.gemma2 || h.gemma3 || h.gemma3n || h.gemma4 {
            Some(f32s(need(g, &format!("blk.{i}.post_ffw_norm.weight"))?)?)
        } else {
            None
        },
        ffn,
        gemma3n: if h.gemma3n {
            Some(load_gemma3n_layer(g, i)?)
        } else {
            None
        },
        gemma4_ple: if h.gemma4 && h.gemma4_n_pl > 0 {
            Some(load_gemma4_ple_layer(g, i)?)
        } else {
            None
        },
    })
}

fn load_gemma4_ple_layer(g: &Gguf, i: usize) -> Result<Gemma4PleLayer, LlamaError> {
    Ok(Gemma4PleLayer {
        inp_gate: quant_mat(need(g, &format!("blk.{i}.inp_gate.weight"))?)?,
        proj: quant_mat(need(g, &format!("blk.{i}.proj.weight"))?)?,
        post_norm: f32s(need(g, &format!("blk.{i}.post_norm.weight"))?)?,
    })
}

fn load_gemma4_ple_weights(
    g: &Gguf,
    n_embd: usize,
    n_layer: usize,
    n_vocab: usize,
    n_pl: usize,
) -> Result<Gemma4PleWeights, LlamaError> {
    if n_pl == 0 {
        return Err(LlamaError::Shape(arch_key(
            "gemma4",
            "embedding_length_per_layer_input",
        )));
    }
    let per_layer_token_embd = quant_mat(need(g, "per_layer_token_embd.weight")?)?;
    let per_layer_model_proj = quant_mat(need(g, "per_layer_model_proj.weight")?)?;
    let want_pl = n_pl
        .checked_mul(n_layer)
        .ok_or_else(|| LlamaError::Shape("per_layer_token_embd".into()))?;
    if per_layer_token_embd.n_cols != want_pl
        || per_layer_token_embd.n_rows != n_vocab
        || per_layer_model_proj.n_cols != n_embd
        || per_layer_model_proj.n_rows != want_pl
    {
        return Err(LlamaError::Shape("per_layer_token_embd".into()));
    }
    let per_layer_proj_norm = f32s(need(g, "per_layer_proj_norm.weight")?)?;
    if per_layer_proj_norm.len() != n_pl {
        return Err(LlamaError::Shape("per_layer_proj_norm".into()));
    }
    Ok(Gemma4PleWeights {
        per_layer_token_embd,
        per_layer_model_proj,
        per_layer_proj_norm,
        n_embd_per_layer: n_pl,
    })
}

fn load_gemma3n_layer(g: &Gguf, i: usize) -> Result<Gemma3nLayer, LlamaError> {
    Ok(Gemma3nLayer {
        inp_gate: quant_mat(need(g, &format!("blk.{i}.inp_gate.weight"))?)?,
        proj: quant_mat(need(g, &format!("blk.{i}.proj.weight"))?)?,
        post_norm: f32s(need(g, &format!("blk.{i}.post_norm.weight"))?)?,
        altup_correct_coef: quant_mat(need(g, &format!("blk.{i}.altup_correct_coef.weight"))?)?,
        altup_correct_scale: f32s(need(g, &format!("blk.{i}.altup_correct_scale.weight"))?)?,
        altup_predict_coef: quant_mat(need(g, &format!("blk.{i}.altup_predict_coef.weight"))?)?,
        altup_router: quant_mat(need(g, &format!("blk.{i}.altup_router.weight"))?)?,
        altup_router_norm: f32s(need(g, &format!("blk.{i}.altup_router_norm.weight"))?)?,
        laurel_l: quant_mat(need(g, &format!("blk.{i}.laurel_l.weight"))?)?,
        laurel_r: quant_mat(need(g, &format!("blk.{i}.laurel_r.weight"))?)?,
        laurel_post_norm: f32s(need(g, &format!("blk.{i}.laurel_post_norm.weight"))?)?,
    })
}

fn load_gemma3n_weights(
    g: &Gguf,
    n_embd: usize,
    n_layer: usize,
    n_vocab: usize,
) -> Result<Gemma3nWeights, LlamaError> {
    let n_altup = g
        .kv_u32(&arch_key("gemma3n", "altup.num_inputs"))
        .map_or(GEMMA3N_N_ALTUP, |v| usize::try_from(v).unwrap_or(0));
    let i_altup_act = g
        .kv_u32(&arch_key("gemma3n", "altup.active_idx"))
        .map_or(GEMMA3N_I_ALTUP_ACT, |v| usize::try_from(v).unwrap_or(0));
    let n_embd_altup = g
        .kv_u32(&arch_key("gemma3n", "embedding_length_per_layer_input"))
        .map_or(GEMMA3N_N_EMBD_ALTUP, |v| usize::try_from(v).unwrap_or(0));
    if n_altup < 2 || i_altup_act >= n_altup || n_embd_altup == 0 {
        return Err(LlamaError::Shape("gemma3n altup".into()));
    }
    let extra = n_altup.saturating_sub(1);
    let altup_proj = quant_mat(need(g, "altup_proj.weight")?)?;
    let altup_unembd_proj = quant_mat(need(g, "altup_unembd_proj.weight")?)?;
    if altup_proj.n_parts != extra
        || altup_unembd_proj.n_parts != extra
        || altup_proj.n_cols != n_embd
        || altup_proj.n_rows != n_embd
        || altup_unembd_proj.n_cols != n_embd
        || altup_unembd_proj.n_rows != n_embd
    {
        return Err(LlamaError::Shape("altup_proj".into()));
    }
    let per_layer_token_embd = quant_mat(need(g, "per_layer_token_embd.weight")?)?;
    let per_layer_model_proj = quant_mat(need(g, "per_layer_model_proj.weight")?)?;
    let want_pl = n_embd_altup
        .checked_mul(n_layer)
        .ok_or_else(|| LlamaError::Shape("per_layer_token_embd".into()))?;
    if per_layer_token_embd.n_cols != want_pl
        || per_layer_token_embd.n_rows != n_vocab
        || per_layer_model_proj.n_cols != n_embd
        || per_layer_model_proj.n_rows != want_pl
    {
        return Err(LlamaError::Shape("per_layer_token_embd".into()));
    }
    let per_layer_proj_norm = f32s(need(g, "per_layer_proj_norm.weight")?)?;
    if per_layer_proj_norm.len() != n_embd_altup {
        return Err(LlamaError::Shape("per_layer_proj_norm".into()));
    }
    Ok(Gemma3nWeights {
        altup_proj,
        altup_unembd_proj,
        per_layer_token_embd,
        per_layer_model_proj,
        per_layer_proj_norm,
        n_altup,
        i_altup_act,
        n_embd_altup,
        n_layer_sparsity: GEMMA3N_N_LAYER_SPARSITY,
    })
}

/// Official bloom fused `attn_qkv` is concatenated Q then K then V
/// (`bloom.cpp` view after `wqkv`; convert restacks HF interleaved heads).
struct BloomQkv {
    wq: QuantMat,
    bq: Option<Vec<f32>>,
    wk: QuantMat,
    bk: Option<Vec<f32>>,
    wv: QuantMat,
    bv: Option<Vec<f32>>,
}

fn load_bloom_qkv(g: &Gguf, i: usize) -> Result<BloomQkv, LlamaError> {
    let name = format!("blk.{i}.attn_qkv.weight");
    let qkv = quant_mat(need(g, &name)?)?;
    if qkv.n_parts != 1 || qkv.n_cols == 0 || qkv.n_rows <= qkv.n_cols {
        return Err(LlamaError::Shape(name));
    }
    let extra = qkv.n_rows.saturating_sub(qkv.n_cols);
    if extra == 0 || !extra.is_multiple_of(2) {
        return Err(LlamaError::Shape(name));
    }
    let n_kv = extra / 2;
    let n_q = qkv.n_cols;
    let wq = quant_mat_rows(&qkv, 0, n_q)?;
    let wk = quant_mat_rows(&qkv, n_q, n_kv)?;
    let wv = quant_mat_rows(&qkv, n_q.saturating_add(n_kv), n_kv)?;
    let bname = format!("blk.{i}.attn_qkv.bias");
    let bias = f32s(need(g, &bname)?)?;
    let want = n_q.saturating_add(n_kv.saturating_mul(2));
    if bias.len() != want {
        return Err(LlamaError::Shape(bname));
    }
    let bq = bias
        .get(..n_q)
        .ok_or_else(|| LlamaError::Shape(bname.clone()))?
        .to_vec();
    let bk = bias
        .get(n_q..n_q.saturating_add(n_kv))
        .ok_or_else(|| LlamaError::Shape(bname.clone()))?
        .to_vec();
    let bv = bias
        .get(n_q.saturating_add(n_kv)..)
        .ok_or(LlamaError::Shape(bname))?
        .to_vec();
    Ok(BloomQkv {
        wq,
        bq: Some(bq),
        wk,
        bk: Some(bk),
        wv,
        bv: Some(bv),
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

fn load_gemma4_moe(g: &Gguf, i: usize, h: &Gemma4MoeHparams) -> Result<Gemma4Moe, LlamaError> {
    let shared = DenseFfn {
        gate: quant_mat(need(g, &format!("blk.{i}.ffn_gate.weight"))?)?,
        up: quant_mat(need(g, &format!("blk.{i}.ffn_up.weight"))?)?,
        down: quant_mat(need(g, &format!("blk.{i}.ffn_down.weight"))?)?,
    };
    let post_norm_1 = f32s(need(g, &format!("blk.{i}.ffn_post_norm_1.weight"))?)?;
    let pre_norm_2 = f32s(need(g, &format!("blk.{i}.ffn_pre_norm_2.weight"))?)?;
    let post_norm_2 = f32s(need(g, &format!("blk.{i}.ffn_post_norm_2.weight"))?)?;
    let gate_inp = quant_mat(need(g, &format!("blk.{i}.ffn_gate_inp.weight"))?)?;
    let gate_inp_s = f32s(need(g, &format!("blk.{i}.ffn_gate_inp.scale"))?)?;
    let down_exps = quant_mat(need(g, &format!("blk.{i}.ffn_down_exps.weight"))?)?;
    if gate_inp.n_rows != h.n_expert || gate_inp.n_parts != 1 {
        return Err(LlamaError::Shape(format!("blk.{i}.ffn_gate_inp.weight")));
    }
    if gate_inp.n_cols != shared.down.n_rows {
        return Err(LlamaError::Shape(format!("blk.{i}.ffn_gate_inp.weight")));
    }
    if gate_inp_s.len() != gate_inp.n_cols {
        return Err(LlamaError::Shape(format!("blk.{i}.ffn_gate_inp.scale")));
    }
    if post_norm_1.len() != gate_inp.n_cols {
        return Err(LlamaError::Shape(format!("blk.{i}.ffn_post_norm_1.weight")));
    }
    if pre_norm_2.len() != gate_inp.n_cols {
        return Err(LlamaError::Shape(format!("blk.{i}.ffn_pre_norm_2.weight")));
    }
    if post_norm_2.len() != gate_inp.n_cols {
        return Err(LlamaError::Shape(format!("blk.{i}.ffn_post_norm_2.weight")));
    }
    if down_exps.n_parts != h.n_expert || down_exps.n_cols != h.n_ff_exp {
        return Err(LlamaError::Shape(format!("blk.{i}.ffn_down_exps.weight")));
    }
    let fused_n = h
        .n_ff_exp
        .checked_mul(2)
        .ok_or_else(|| LlamaError::Shape(format!("blk.{i}.ffn_gate_up_exps.weight")))?;
    let (gate_exps, up_exps, gate_up) = if g
        .tensor(&format!("blk.{i}.ffn_gate_up_exps.weight"))
        .is_some()
    {
        let gate_up = quant_mat(need(g, &format!("blk.{i}.ffn_gate_up_exps.weight"))?)?;
        if gate_up.n_parts != h.n_expert
            || gate_up.n_rows != fused_n
            || gate_up.n_cols != gate_inp.n_cols
        {
            return Err(LlamaError::Shape(format!(
                "blk.{i}.ffn_gate_up_exps.weight"
            )));
        }
        (None, None, Some(gate_up))
    } else {
        let gate_exps = quant_mat(need(g, &format!("blk.{i}.ffn_gate_exps.weight"))?)?;
        let up_exps = quant_mat(need(g, &format!("blk.{i}.ffn_up_exps.weight"))?)?;
        for (t, name) in [
            (&gate_exps, format!("blk.{i}.ffn_gate_exps.weight")),
            (&up_exps, format!("blk.{i}.ffn_up_exps.weight")),
        ] {
            if t.n_parts != h.n_expert {
                return Err(LlamaError::Shape(name));
            }
        }
        if gate_exps.n_rows != h.n_ff_exp || up_exps.n_rows != h.n_ff_exp {
            return Err(LlamaError::Shape(format!("blk.{i}.ffn_*_exps.weight")));
        }
        (Some(gate_exps), Some(up_exps), None)
    };
    Ok(Gemma4Moe {
        shared,
        post_norm_1,
        pre_norm_2,
        post_norm_2,
        gate_inp,
        gate_inp_s,
        gate_exps,
        up_exps,
        gate_up,
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
        || name == "token_embd_norm.weight"
        || name == "token_embd_norm.bias"
        || name.ends_with(".attn_norm.weight")
        || name.ends_with(".attn_norm.bias")
        || name.ends_with(".ffn_norm.weight")
        || name.ends_with(".ffn_norm.bias")
        || name.ends_with(".post_attention_norm.weight")
        || name.ends_with(".post_ffw_norm.weight")
        || name.ends_with(".ffn_post_norm_1.weight")
        || name.ends_with(".ffn_pre_norm_2.weight")
        || name.ends_with(".ffn_post_norm_2.weight")
        || name.ends_with(".ffn_gate_inp.scale")
        || name.ends_with(".attn_q_norm.weight")
        || name.ends_with(".attn_k_norm.weight")
        || name.ends_with(".post_norm.weight")
        || name.ends_with(".altup_router_norm.weight")
        || name.ends_with(".laurel_post_norm.weight")
        || name.ends_with(".altup_correct_scale.weight")
        || name == "per_layer_proj_norm.weight"
        || name.ends_with(".attn_q.bias")
        || name.ends_with(".attn_k.bias")
        || name.ends_with(".attn_v.bias")
        || name.ends_with(".attn_qkv.bias")
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
        | GgmlType::Q4_0
        | GgmlType::Q8_0
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

/// Row-contiguous slice of `m` (`first .. first + n_rows`) sharing the blob range.
fn quant_mat_rows(m: &QuantMat, first: usize, n_rows: usize) -> Result<QuantMat, LlamaError> {
    let last = first
        .checked_add(n_rows)
        .ok_or_else(|| LlamaError::Shape(m.name.clone()))?;
    if n_rows == 0 || last > m.n_rows {
        return Err(LlamaError::Shape(m.name.clone()));
    }
    let rb = row_bytes_for(m.ty, m.n_cols, &m.name)?;
    let off = first
        .checked_mul(rb)
        .ok_or_else(|| LlamaError::Shape(m.name.clone()))?;
    let bytes = n_rows
        .checked_mul(rb)
        .ok_or_else(|| LlamaError::Shape(m.name.clone()))?;
    let start = m
        .start
        .checked_add(off)
        .ok_or_else(|| LlamaError::Shape(m.name.clone()))?;
    let end = start
        .checked_add(bytes)
        .ok_or_else(|| LlamaError::Shape(m.name.clone()))?;
    if end > m.end {
        return Err(LlamaError::Shape(m.name.clone()));
    }
    Ok(QuantMat {
        name: m.name.clone(),
        ty: m.ty,
        n_cols: m.n_cols,
        n_rows,
        n_parts: m.n_parts,
        start,
        end,
    })
}

impl Llama {
    /// `y[t, r] = W[r] · x[t]`, into `y` (grown to `n_rows * n_tokens`).
    fn gemm_into(
        &self,
        m: &QuantMat,
        n_tokens: usize,
        x: &[f32],
        y: &mut Vec<f32>,
        pool: &mut GemvPool,
    ) -> Result<(), LlamaError> {
        if n_tokens == 1 {
            return self.gemv_into(m, x, y, pool);
        }
        let data = self.mat_bytes(m)?;
        Self::gemm_data_into(m, n_tokens, data, x, y)
    }

    /// `y[t, r] = W[r] · x[t]` against an already-sliced weight blob.
    fn gemm_data_into(
        m: &QuantMat,
        n_tokens: usize,
        data: &[u8],
        x: &[f32],
        y: &mut Vec<f32>,
    ) -> Result<(), LlamaError> {
        let n_out = m
            .n_rows
            .checked_mul(n_tokens)
            .ok_or_else(|| LlamaError::Shape(m.name.clone()))?;
        fit(y, n_out);
        let y = y.as_mut_slice();
        match m.ty {
            GgmlType::F32 => gemm_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::F16 => gemm_f16(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::BF16 => gemm_bf16(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::Q2_K => gemm_q2_k_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::Q3_K => gemm_q3_k_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::Q4_1 => gemm_q4_1_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::Q5_0 => gemm_q5_0_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::Q5_1 => gemm_q5_1_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::MXFP4 => gemm_mxfp4_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::NVFP4 => gemm_nvfp4_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::Q1_0 => gemm_q1_0_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::Q2_0 => gemm_q2_0_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::Q4_0 => gemm_q4_0_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::Q8_0 => gemm_q8_0_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::Q8_1 => gemm_q8_1_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::TQ1_0 => gemm_tq1_0_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::TQ2_0 => gemm_tq2_0_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::Q4_K => gemm_q4_k_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::Q5_K => gemm_q5_k_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::Q6_K => gemm_q6_k_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::IQ1_S => gemm_iq1_s_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::IQ1_M => gemm_iq1_m_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::IQ2_XXS => gemm_iq2_xxs_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::IQ2_XS => gemm_iq2_xs_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::IQ2_S => gemm_iq2_s_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::IQ3_XXS => gemm_iq3_xxs_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::IQ3_S => gemm_iq3_s_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::IQ4_NL => gemm_iq4_nl_f32(m.n_cols, n_tokens, data, x, y)?,
            GgmlType::IQ4_XS => gemm_iq4_xs_f32(m.n_cols, n_tokens, data, x, y)?,
            other => {
                return Err(LlamaError::Type {
                    tensor: m.name.clone(),
                    ty: other.to_i32(),
                })
            }
        }
        Ok(())
    }

    /// `y[r] = W[r] · x`, into `y` (grown to `n_rows`).
    fn gemv_into(
        &self,
        m: &QuantMat,
        x: &[f32],
        y: &mut Vec<f32>,
        pool: &mut GemvPool,
    ) -> Result<(), LlamaError> {
        self.gemv_range_into(m, m.start, m.end.saturating_sub(m.start), x, y, pool)
    }

    /// [`Llama::gemv_into`] against one part of a 3-D `*_exps` matrix.
    fn gemv_part_into(
        &self,
        m: &QuantMat,
        part: usize,
        x: &[f32],
        y: &mut Vec<f32>,
        pool: &mut GemvPool,
    ) -> Result<(), LlamaError> {
        let (base, len) = self.mat_part_range(m, part)?;
        self.gemv_range_into(m, base, len, x, y, pool)
    }

    /// GEMV against the blob range `[base, base + len)`, on the pool when the
    /// matrix is big enough and the pool accepts the job, else in this thread.
    fn gemv_range_into(
        &self,
        m: &QuantMat,
        base: usize,
        len: usize,
        x: &[f32],
        y: &mut Vec<f32>,
        pool: &mut GemvPool,
    ) -> Result<(), LlamaError> {
        fit(y, m.n_rows);
        let y = y.as_mut_slice();
        if pooled_gemv(m.n_rows, m.n_cols) {
            if let Some(p) = pool.as_mut() {
                let row_bytes = row_bytes_for(m.ty, m.n_cols, &m.name)?;
                let job = GemvJob {
                    ty: m.ty,
                    n_cols: m.n_cols,
                    base,
                    row_bytes,
                };
                if p.run(job, x, y) {
                    return Ok(());
                }
            }
        }
        let data = self
            .blob
            .get(base..base.saturating_add(len))
            .ok_or_else(|| LlamaError::Shape(m.name.clone()))?;
        if gemv_rows(m.ty, m.n_cols, data, x, y)? {
            Ok(())
        } else {
            Err(LlamaError::Type {
                tensor: m.name.clone(),
                ty: m.ty.to_i32(),
            })
        }
    }

    /// Byte offset and length of one part of a 3-D `*_exps` matrix.
    /// `n_parts == 1` matrices have a single part covering the whole range.
    fn mat_part_range(&self, m: &QuantMat, part: usize) -> Result<(usize, usize), LlamaError> {
        if m.n_parts == 0 || part >= m.n_parts {
            return Err(LlamaError::Shape(m.name.clone()));
        }
        let total = self.mat_bytes(m)?.len();
        let per = total / m.n_parts;
        if per.saturating_mul(m.n_parts) != total {
            return Err(LlamaError::Shape(m.name.clone()));
        }
        Ok((m.start.saturating_add(part.saturating_mul(per)), per))
    }

    fn gemv_part_bytes_into(
        &self,
        m: &QuantMat,
        part: usize,
        x: &[f32],
        y: &mut Vec<f32>,
        pool: &mut GemvPool,
        bytes: Option<&[u8]>,
    ) -> Result<(), LlamaError> {
        match bytes {
            Some(data) => {
                fit(y, m.n_rows);
                let y = y.as_mut_slice();
                if gemv_rows(m.ty, m.n_cols, data, x, y)? {
                    Ok(())
                } else {
                    Err(LlamaError::Type {
                        tensor: m.name.clone(),
                        ty: m.ty.to_i32(),
                    })
                }
            }
            None => self.gemv_part_into(m, part, x, y, pool),
        }
    }

    /// GEMM `n_tokens` rows against one expert part. One token stays on the GEMV pool.
    fn gemm_part_bytes_into(
        &self,
        spec: PartBytes<'_>,
        x: &[f32],
        y: &mut Vec<f32>,
        pool: &mut GemvPool,
    ) -> Result<(), LlamaError> {
        if spec.n_tokens == 1 {
            return self.gemv_part_bytes_into(spec.m, spec.part, x, y, pool, spec.bytes);
        }
        match spec.bytes {
            Some(data) => Self::gemm_data_into(spec.m, spec.n_tokens, data, x, y),
            None => {
                let (base, len) = self.mat_part_range(spec.m, spec.part)?;
                let data = self
                    .blob
                    .get(base..base.saturating_add(len))
                    .ok_or_else(|| LlamaError::Shape(spec.m.name.clone()))?;
                Self::gemm_data_into(spec.m, spec.n_tokens, data, x, y)
            }
        }
    }

    fn take_expert(
        store: &mut Option<LiveStore>,
        layer: u32,
        expert: usize,
    ) -> Result<(Option<ExpertParts>, Option<ExpertKey>), LlamaError> {
        let Some(store) = store.as_mut() else {
            return Ok((None, None));
        };
        let e = u32::try_from(expert).map_err(|_| LlamaError::Shape("expert id".into()))?;
        let key = ExpertKey::new(layer, e);
        let parts = store.acquire(key)?;
        store.lease(key)?;
        Ok((Some(parts), Some(key)))
    }

    fn put_expert(store: &mut Option<LiveStore>, held: Option<ExpertKey>) {
        if let (Some(store), Some(key)) = (store.as_mut(), held) {
            store.release(key);
        }
    }

    /// Route every token with one `ffn_gate_inp` GEMM, then one expert GEMM per
    /// selected expert.
    fn softmax_routed_layer(
        &self,
        spec: SoftmaxMoE<'_>,
        n_tokens: usize,
        s: &mut Scratch,
        run: MoeExec<'_>,
    ) -> Result<(), LlamaError> {
        let MoeExec {
            pool,
            moe_trace,
            store,
        } = run;
        self.route_softmax_tokens(&spec, n_tokens, s, pool, moe_trace)?;
        prefetch_routed(moe_trace, store, n_tokens, spec.n_used, &s.moe.sel_e);
        prefetch_selected(moe_trace, store, n_tokens, spec.n_used, &s.moe.sel_e);
        self.grouped_routed_ffn(
            &spec,
            n_tokens,
            s,
            GroupedRun {
                pool,
                store,
                layer: moe_trace.layer,
                row_seq: &moe_trace.row_seq,
                sequence: moe_trace.sequence,
            },
        )?;
        Ok(())
    }

    /// `ffn_gate_inp` GEMM then per-row softmax + top-k. Bit-equal to serial GEMV.
    fn route_softmax_tokens(
        &self,
        spec: &SoftmaxMoE<'_>,
        n_tokens: usize,
        s: &mut Scratch,
        pool: &mut GemvPool,
        moe_trace: &mut MoeTraceBuf,
    ) -> Result<(), LlamaError> {
        self.gemm_into(spec.gate_inp, n_tokens, &s.x, &mut s.moe.logits, pool)?;
        Self::softmax_select_tokens(spec, n_tokens, s, moe_trace)
    }

    /// Softmax then top-k on already-written `s.moe.logits`. Gemma4 MoE fills
    /// those logits from a custom router on `attn_out`, not `ffn_gate_inp · x`.
    fn softmax_select_tokens(
        spec: &SoftmaxMoE<'_>,
        n_tokens: usize,
        s: &mut Scratch,
        moe_trace: &mut MoeTraceBuf,
    ) -> Result<(), LlamaError> {
        s.moe.sel_e.clear();
        s.moe.sel_w.clear();
        for t in 0..n_tokens {
            {
                let row = token_row_mut(&mut s.moe.logits, t, spec.n_expert, spec.err)?;
                softmax(row);
            }
            let row = token_row(&s.moe.logits, t, spec.n_expert, spec.err)?;
            topk_into(row, spec.n_used, &mut s.moe.order)?;
            fill_router_weights(row, &s.moe.order, &mut s.moe.weights, spec.norm_w);
            moe_trace.record(t, &s.moe.order, &s.moe.weights);
            s.moe.sel_e.extend_from_slice(&s.moe.order);
            s.moe.sel_w.extend_from_slice(&s.moe.weights);
        }
        Ok(())
    }

    fn grouped_routed_ffn(
        &self,
        spec: &SoftmaxMoE<'_>,
        n_tokens: usize,
        s: &mut Scratch,
        run: GroupedRun<'_>,
    ) -> Result<(), LlamaError> {
        while s.moe.buckets.len() < spec.n_expert {
            s.moe.buckets.push(Vec::new());
        }
        for b in &mut s.moe.buckets {
            b.clear();
        }
        for t in 0..n_tokens {
            for j in 0..spec.n_used {
                let idx = t.saturating_mul(spec.n_used).saturating_add(j);
                let e = *s
                    .moe
                    .sel_e
                    .get(idx)
                    .ok_or_else(|| LlamaError::Shape(spec.err.into()))?;
                let bucket = s
                    .moe
                    .buckets
                    .get_mut(e)
                    .ok_or_else(|| LlamaError::Shape(spec.err.into()))?;
                bucket.push(t);
            }
        }
        for e in 0..spec.n_expert {
            let toks = s
                .moe
                .buckets
                .get_mut(e)
                .map(core::mem::take)
                .unwrap_or_default();
            if toks.is_empty() {
                continue;
            }
            self.grouped_one_expert(
                spec,
                ExpertJob {
                    expert: e,
                    toks: &toks,
                    layer: run.layer,
                    row_seq: run.row_seq,
                    sequence: run.sequence,
                },
                s,
                run.pool,
                run.store,
            )?;
        }
        Ok(())
    }

    fn grouped_one_expert(
        &self,
        spec: &SoftmaxMoE<'_>,
        job: ExpertJob<'_>,
        s: &mut Scratch,
        pool: &mut GemvPool,
        store: &mut Option<LiveStore>,
    ) -> Result<(), LlamaError> {
        let n = job.toks.len();
        let width = n
            .checked_mul(spec.n_embd)
            .ok_or_else(|| LlamaError::Shape(spec.err.into()))?;
        fit(&mut s.moe.pack_x, width);
        for (i, t) in job.toks.iter().copied().enumerate() {
            let w = if spec.scale_x {
                token_expert_weight(&s.moe, t, spec.n_used, job.expert).unwrap_or(0.0)
            } else {
                0.0
            };
            let xt = token_row(&s.x, t, spec.n_embd, spec.err)?;
            let off = i.saturating_mul(spec.n_embd);
            let dst = s
                .moe
                .pack_x
                .get_mut(off..off.saturating_add(spec.n_embd))
                .ok_or_else(|| LlamaError::Shape(spec.err.into()))?;
            if spec.scale_x {
                for (d, v) in dst.iter_mut().zip(xt.iter()) {
                    *d = *v * w;
                }
            } else {
                dst.copy_from_slice(xt);
            }
        }
        bind_row(
            store,
            job.row_seq,
            job.sequence,
            job.toks.first().copied().unwrap_or(0),
        );
        let (parts, held) = Self::take_expert(store, job.layer, job.expert)?;
        let r = self.swiglu_expert_gemm(
            spec,
            ExpertGemm {
                expert: job.expert,
                n,
                parts: parts.as_ref(),
            },
            s,
            pool,
        );
        Self::put_expert(store, held);
        r?;
        Self::scatter_expert_rows(spec, job.expert, job.toks, s)
    }

    fn route_llama4_tokens(
        &self,
        moe: &Llama4Moe,
        n_tokens: usize,
        s: &mut Scratch,
        pool: &mut GemvPool,
        moe_trace: &mut MoeTraceBuf,
    ) -> Result<(), LlamaError> {
        self.gemm_into(&moe.gate_inp, n_tokens, &s.x, &mut s.moe.logits, pool)?;
        s.moe.sel_e.clear();
        s.moe.sel_w.clear();
        for t in 0..n_tokens {
            let row = token_row(&s.moe.logits, t, moe.n_expert, "llama4 ffn_gate_inp")?;
            topk_into(row, moe.n_expert_used, &mut s.moe.order)?;
            s.moe.weights.clear();
            for i in 0..s.moe.order.len() {
                let Some(e) = s.moe.order.get(i).copied() else {
                    continue;
                };
                s.moe
                    .weights
                    .push(sigmoid_f32(row.get(e).copied().unwrap_or(0.0)));
            }
            moe_trace.record(t, &s.moe.order, &s.moe.weights);
            s.moe.sel_e.extend_from_slice(&s.moe.order);
            s.moe.sel_w.extend_from_slice(&s.moe.weights);
        }
        Ok(())
    }

    fn swiglu_expert_gemm(
        &self,
        spec: &SoftmaxMoE<'_>,
        g: ExpertGemm<'_>,
        s: &mut Scratch,
        pool: &mut GemvPool,
    ) -> Result<(), LlamaError> {
        let gate_b = g.parts.map(|p| p.gate.as_slice());
        self.gemm_part_bytes_into(
            PartBytes {
                m: spec.gate,
                part: g.expert,
                n_tokens: g.n,
                bytes: gate_b,
            },
            &s.moe.pack_x,
            &mut s.moe.g,
            pool,
        )?;
        if spec.fused {
            let n_ff = spec.down.n_cols;
            let fused_w = n_ff
                .checked_mul(2)
                .ok_or_else(|| LlamaError::Shape(spec.err.into()))?;
            if n_ff == 0 || spec.gate.n_rows != fused_w {
                return Err(LlamaError::Shape(spec.err.into()));
            }
            let n_out =
                g.n.checked_mul(n_ff)
                    .ok_or_else(|| LlamaError::Shape(spec.err.into()))?;
            fit(&mut s.moe.u, n_out);
            for t in 0..g.n {
                let src = t
                    .checked_mul(fused_w)
                    .ok_or_else(|| LlamaError::Shape(spec.err.into()))?;
                let dst = t
                    .checked_mul(n_ff)
                    .ok_or_else(|| LlamaError::Shape(spec.err.into()))?;
                let up = s
                    .moe
                    .g
                    .get(src.saturating_add(n_ff)..src.saturating_add(fused_w))
                    .ok_or_else(|| LlamaError::Shape(spec.err.into()))?;
                let ud = s
                    .moe
                    .u
                    .get_mut(dst..dst.saturating_add(n_ff))
                    .ok_or_else(|| LlamaError::Shape(spec.err.into()))?;
                ud.copy_from_slice(up);
            }
            for t in 0..g.n {
                let src = t
                    .checked_mul(fused_w)
                    .ok_or_else(|| LlamaError::Shape(spec.err.into()))?;
                let dst = t
                    .checked_mul(n_ff)
                    .ok_or_else(|| LlamaError::Shape(spec.err.into()))?;
                if src != dst {
                    for i in 0..n_ff {
                        let v = *s
                            .moe
                            .g
                            .get(src.saturating_add(i))
                            .ok_or_else(|| LlamaError::Shape(spec.err.into()))?;
                        if let Some(slot) = s.moe.g.get_mut(dst.saturating_add(i)) {
                            *slot = v;
                        } else {
                            return Err(LlamaError::Shape(spec.err.into()));
                        }
                    }
                }
            }
            s.moe.g.truncate(n_out);
        } else {
            let up_b = g.parts.map(|p| p.up.as_slice());
            self.gemm_part_bytes_into(
                PartBytes {
                    m: spec.up,
                    part: g.expert,
                    n_tokens: g.n,
                    bytes: up_b,
                },
                &s.moe.pack_x,
                &mut s.moe.u,
                pool,
            )?;
        }
        ffn_gate_act_inplace(&mut s.moe.g, spec.gelu);
        for (a, b) in s.moe.g.iter_mut().zip(s.moe.u.iter()) {
            *a *= *b;
        }
        let down_b = g.parts.map(|p| p.down.as_slice());
        self.gemm_part_bytes_into(
            PartBytes {
                m: spec.down,
                part: g.expert,
                n_tokens: g.n,
                bytes: down_b,
            },
            &s.moe.g,
            &mut s.moe.y,
            pool,
        )
    }

    fn scatter_expert_rows(
        spec: &SoftmaxMoE<'_>,
        expert: usize,
        toks: &[usize],
        s: &mut Scratch,
    ) -> Result<(), LlamaError> {
        for (i, t) in toks.iter().copied().enumerate() {
            let w = if spec.scale_x {
                1.0
            } else {
                token_expert_weight(&s.moe, t, spec.n_used, expert).unwrap_or(0.0)
            };
            let src_off = i.saturating_mul(spec.n_embd);
            let src = s
                .moe
                .y
                .get(src_off..src_off.saturating_add(spec.n_embd))
                .ok_or_else(|| LlamaError::Shape(spec.err.into()))?;
            let dst_off = t.saturating_mul(spec.n_embd);
            let dst = s
                .ffn_out
                .get_mut(dst_off..dst_off.saturating_add(spec.n_embd))
                .ok_or_else(|| LlamaError::Shape(spec.err.into()))?;
            for (d, v) in dst.iter_mut().zip(src.iter()) {
                *d += *v * w;
            }
        }
        Ok(())
    }

    fn shexp_silu_into(
        &self,
        moe: &Qwen2Moe,
        n_tokens: usize,
        s: &mut Scratch,
        pool: &mut GemvPool,
        err: &'static str,
    ) -> Result<(), LlamaError> {
        let n_embd = moe.down_shexp.n_rows;
        self.gemm_into(&moe.gate_shexp, n_tokens, &s.x, &mut s.gate, pool)?;
        self.gemm_into(&moe.up_shexp, n_tokens, &s.x, &mut s.up, pool)?;
        silu_inplace(&mut s.gate);
        for (hv, uv) in s.gate.iter_mut().zip(s.up.iter()) {
            *hv *= *uv;
        }
        self.gemm_into(&moe.down_shexp, n_tokens, &s.gate, &mut s.ffn_out, pool)?;
        self.gemm_into(
            &moe.gate_inp_shexp,
            n_tokens,
            &s.x,
            &mut s.moe.shexp_gate,
            pool,
        )?;
        if s.moe.shexp_gate.len() != n_tokens {
            return Err(LlamaError::Shape(format!("{err} ffn_gate_inp_shexp")));
        }
        for t in 0..n_tokens {
            let w = sigmoid_f32(s.moe.shexp_gate.get(t).copied().unwrap_or(0.0));
            let off = t.saturating_mul(n_embd);
            let row = s
                .ffn_out
                .get_mut(off..off.saturating_add(n_embd))
                .ok_or_else(|| LlamaError::Shape(format!("{err} shexp")))?;
            for v in row.iter_mut() {
                *v *= w;
            }
        }
        Ok(())
    }

    /// Official llama.cpp `build_moe_ffn` for `architecture=llama` + `n_expert>0`:
    /// softmax over all experts, then top-k; SwiGLU; weights after the expert
    /// with `norm_w` clamp `2^-14`. No shared expert (Mixtral-shaped).
    ///
    /// Reads `s.x` and writes `s.ffn_out`.
    fn llama_moe_into(
        &self,
        moe: &LlamaMoe,
        n_tokens: usize,
        s: &mut Scratch,
        pool: &mut GemvPool,
        moe_trace: &mut MoeTraceBuf,
        store: &mut Option<LiveStore>,
    ) -> Result<(), LlamaError> {
        let n_embd = moe.down_exps.n_rows;
        if n_embd == 0 || !s.x.len().is_multiple_of(n_embd) {
            return Err(LlamaError::Shape("llama moe".into()));
        }
        let n_out = n_tokens
            .checked_mul(n_embd)
            .ok_or_else(|| LlamaError::Shape("llama moe".into()))?;
        fit(&mut s.ffn_out, n_out);
        self.softmax_routed_layer(
            SoftmaxMoE {
                gate_inp: &moe.gate_inp,
                gate: &moe.gate_exps,
                up: &moe.up_exps,
                down: &moe.down_exps,
                n_expert: moe.n_expert,
                n_used: moe.n_expert_used,
                n_embd,
                norm_w: true,
                scale_x: false,
                gelu: false,
                fused: false,
                err: "llama moe",
            },
            n_tokens,
            s,
            MoeExec {
                pool,
                moe_trace,
                store,
            },
        )
    }

    /// Official llama4.cpp MoE: top-k on raw router logits, sigmoid weights applied
    /// to the FFN input (`weight_before_ffn`), SwiGLU experts, plus shared expert.
    ///
    /// Reads `s.x` and writes `s.ffn_out`.
    fn llama4_moe_into(
        &self,
        moe: &Llama4Moe,
        n_tokens: usize,
        s: &mut Scratch,
        pool: &mut GemvPool,
        moe_trace: &mut MoeTraceBuf,
        store: &mut Option<LiveStore>,
    ) -> Result<(), LlamaError> {
        let n_embd = moe.down_shexp.n_rows;
        if n_embd == 0 || !s.x.len().is_multiple_of(n_embd) {
            return Err(LlamaError::Shape("llama4 moe".into()));
        }
        self.gemm_into(&moe.gate_shexp, n_tokens, &s.x, &mut s.gate, pool)?;
        self.gemm_into(&moe.up_shexp, n_tokens, &s.x, &mut s.up, pool)?;
        silu_inplace(&mut s.gate);
        for (hv, uv) in s.gate.iter_mut().zip(s.up.iter()) {
            *hv *= *uv;
        }
        self.gemm_into(&moe.down_shexp, n_tokens, &s.gate, &mut s.ffn_out, pool)?;
        self.route_llama4_tokens(moe, n_tokens, s, pool, moe_trace)?;
        prefetch_routed(moe_trace, store, n_tokens, moe.n_expert_used, &s.moe.sel_e);
        prefetch_selected(moe_trace, store, n_tokens, moe.n_expert_used, &s.moe.sel_e);
        let spec = SoftmaxMoE {
            gate_inp: &moe.gate_inp,
            gate: &moe.gate_exps,
            up: &moe.up_exps,
            down: &moe.down_exps,
            n_expert: moe.n_expert,
            n_used: moe.n_expert_used,
            n_embd,
            norm_w: false,
            scale_x: true,
            gelu: false,
            fused: false,
            err: "llama4 moe",
        };
        self.grouped_routed_ffn(
            &spec,
            n_tokens,
            s,
            GroupedRun {
                pool,
                store,
                layer: moe_trace.layer,
                row_seq: &moe_trace.row_seq,
                sequence: moe_trace.sequence,
            },
        )?;
        Ok(())
    }

    /// Official qwen2moe.cpp: softmax then top-k, weights after SwiGLU (`norm_w=false`),
    /// plus shared expert gated by `silu(x)/x` on `ffn_gate_inp_shexp`.
    ///
    /// Reads `s.x` and writes `s.ffn_out`.
    fn qwen2moe_into(
        &self,
        moe: &Qwen2Moe,
        n_tokens: usize,
        s: &mut Scratch,
        pool: &mut GemvPool,
        moe_trace: &mut MoeTraceBuf,
        store: &mut Option<LiveStore>,
    ) -> Result<(), LlamaError> {
        let n_embd = moe.down_shexp.n_rows;
        if n_embd == 0 || !s.x.len().is_multiple_of(n_embd) {
            return Err(LlamaError::Shape("qwen2moe".into()));
        }
        self.shexp_silu_into(moe, n_tokens, s, pool, "qwen2moe")?;
        self.softmax_routed_layer(
            SoftmaxMoE {
                gate_inp: &moe.gate_inp,
                gate: &moe.gate_exps,
                up: &moe.up_exps,
                down: &moe.down_exps,
                n_expert: moe.n_expert,
                n_used: moe.n_expert_used,
                n_embd,
                norm_w: false,
                scale_x: false,
                gelu: false,
                fused: false,
                err: "qwen2moe",
            },
            n_tokens,
            s,
            MoeExec {
                pool,
                moe_trace,
                store,
            },
        )
    }

    /// Official qwen3moe.cpp: softmax then top-k, weights after SwiGLU (`norm_w`
    /// clamp `2^-14`). No shared expert. QK-Norm is applied on Q/K before RoPE.
    ///
    /// Reads `s.x` and writes `s.ffn_out`.
    fn qwen3moe_into(
        &self,
        moe: &Qwen3Moe,
        n_tokens: usize,
        s: &mut Scratch,
        pool: &mut GemvPool,
        moe_trace: &mut MoeTraceBuf,
        store: &mut Option<LiveStore>,
    ) -> Result<(), LlamaError> {
        let n_embd = moe.down_exps.n_rows;
        if n_embd == 0 || !s.x.len().is_multiple_of(n_embd) {
            return Err(LlamaError::Shape("qwen3moe".into()));
        }
        let n_out = n_tokens
            .checked_mul(n_embd)
            .ok_or_else(|| LlamaError::Shape("qwen3moe".into()))?;
        fit(&mut s.ffn_out, n_out);
        self.softmax_routed_layer(
            SoftmaxMoE {
                gate_inp: &moe.gate_inp,
                gate: &moe.gate_exps,
                up: &moe.up_exps,
                down: &moe.down_exps,
                n_expert: moe.n_expert,
                n_used: moe.n_expert_used,
                n_embd,
                norm_w: true,
                scale_x: false,
                gelu: false,
                fused: false,
                err: "qwen3moe",
            },
            n_tokens,
            s,
            MoeExec {
                pool,
                moe_trace,
                store,
            },
        )
    }

    /// Official qwen3next.cpp: softmax then top-k, weights after SwiGLU (`norm_w`
    /// clamp `2^-14`), plus shared expert * sigmoid(`ffn_gate_inp_shexp`).
    ///
    /// Reads `s.x` and writes `s.ffn_out`.
    fn qwen3next_into(
        &self,
        moe: &Qwen2Moe,
        n_tokens: usize,
        s: &mut Scratch,
        pool: &mut GemvPool,
        moe_trace: &mut MoeTraceBuf,
        store: &mut Option<LiveStore>,
    ) -> Result<(), LlamaError> {
        let n_embd = moe.down_shexp.n_rows;
        if n_embd == 0 || !s.x.len().is_multiple_of(n_embd) {
            return Err(LlamaError::Shape("qwen3next".into()));
        }
        self.shexp_silu_into(moe, n_tokens, s, pool, "qwen3next")?;
        self.softmax_routed_layer(
            SoftmaxMoE {
                gate_inp: &moe.gate_inp,
                gate: &moe.gate_exps,
                up: &moe.up_exps,
                down: &moe.down_exps,
                n_expert: moe.n_expert,
                n_used: moe.n_expert_used,
                n_embd,
                norm_w: true,
                scale_x: false,
                gelu: false,
                fused: false,
                err: "qwen3next",
            },
            n_tokens,
            s,
            MoeExec {
                pool,
                moe_trace,
                store,
            },
        )
    }

    /// Official gemma4.cpp MoE: shared dense GeGLU plus routed GELU experts.
    ///
    /// On entry `s.x` is `ffn_norm(attn_out)` and `s.residual` is `attn_out`.
    /// Writes `s.ffn_out` = `post_norm_1(shared)` plus `post_norm_2(experts)`.
    /// The caller still applies `post_ffw_norm` and the residual add.
    fn gemma4_moe_into(
        &self,
        moe: &Gemma4Moe,
        n_tokens: usize,
        s: &mut Scratch,
        pool: &mut GemvPool,
        moe_trace: &mut MoeTraceBuf,
        store: &mut Option<LiveStore>,
    ) -> Result<(), LlamaError> {
        let n_embd = moe.down_exps.n_rows;
        if n_embd == 0 || n_embd != self.n_embd || !s.x.len().is_multiple_of(n_embd) {
            return Err(LlamaError::Shape("gemma4 moe".into()));
        }
        let n_out = n_tokens
            .checked_mul(n_embd)
            .ok_or_else(|| LlamaError::Shape("gemma4 moe".into()))?;
        self.gemm_into(&moe.shared.gate, n_tokens, &s.x, &mut s.gate, pool)?;
        self.gemm_into(&moe.shared.up, n_tokens, &s.x, &mut s.up, pool)?;
        ffn_gate_act_inplace(&mut s.gate, true);
        for (hv, uv) in s.gate.iter_mut().zip(s.up.iter()) {
            *hv *= *uv;
        }
        self.gemm_into(&moe.shared.down, n_tokens, &s.gate, &mut s.ffn_out, pool)?;
        rmsnorm_rows_inplace(&mut s.ffn_out, n_embd, &moe.post_norm_1, self.rms_eps)?;
        copy_buf(&mut s.attn, &s.ffn_out);
        copy_buf(&mut s.x, &s.residual);
        rmsnorm_rows_inplace(&mut s.x, n_embd, &moe.pre_norm_2, self.rms_eps)?;
        copy_buf(&mut s.moe.pack_x, &s.residual);
        rmsnorm_unweighted_rows_inplace(&mut s.moe.pack_x, n_embd, self.rms_eps)?;
        let n_embd_f = f32::from(u16::try_from(n_embd).unwrap_or(1));
        let inv = if n_embd_f > 0.0 {
            1.0 / n_embd_f.sqrt()
        } else {
            0.0
        };
        if moe.gate_inp_s.len() != n_embd {
            return Err(LlamaError::Shape("gemma4 moe".into()));
        }
        for t in 0..n_tokens {
            let row = token_row_mut(&mut s.moe.pack_x, t, n_embd, "gemma4 moe")?;
            for (v, scale) in row.iter_mut().zip(moe.gate_inp_s.iter()) {
                *v *= inv * *scale;
            }
        }
        self.gemm_into(
            &moe.gate_inp,
            n_tokens,
            &s.moe.pack_x,
            &mut s.moe.logits,
            pool,
        )?;
        let fused = moe.gate_up.as_ref();
        let gate = fused
            .or(moe.gate_exps.as_ref())
            .ok_or_else(|| LlamaError::Shape("gemma4 moe".into()))?;
        let up = fused
            .or(moe.up_exps.as_ref())
            .ok_or_else(|| LlamaError::Shape("gemma4 moe".into()))?;
        let spec = SoftmaxMoE {
            gate_inp: &moe.gate_inp,
            gate,
            up,
            down: &moe.down_exps,
            n_expert: moe.n_expert,
            n_used: moe.n_expert_used,
            n_embd,
            norm_w: true,
            scale_x: false,
            gelu: true,
            fused: fused.is_some(),
            err: "gemma4 moe",
        };
        Self::softmax_select_tokens(&spec, n_tokens, s, moe_trace)?;
        prefetch_routed(moe_trace, store, n_tokens, spec.n_used, &s.moe.sel_e);
        prefetch_selected(moe_trace, store, n_tokens, spec.n_used, &s.moe.sel_e);
        fit(&mut s.ffn_out, n_out);
        self.grouped_routed_ffn(
            &spec,
            n_tokens,
            s,
            GroupedRun {
                pool,
                store,
                layer: moe_trace.layer,
                row_seq: &moe_trace.row_seq,
                sequence: moe_trace.sequence,
            },
        )?;
        rmsnorm_rows_inplace(&mut s.ffn_out, n_embd, &moe.post_norm_2, self.rms_eps)?;
        add_assign(&mut s.ffn_out, &s.attn)?;
        Ok(())
    }

    /// Dequantize `token`'s embedding row into `y` (`token_embd.n_cols` long).
    fn embed_into(&self, token: u32, y: &mut [f32]) -> Result<(), LlamaError> {
        self.embed_mat_into(&self.token_embd, token, y)
    }

    /// Dequantize `token`'s row of `emb` into `y` (`emb.n_cols` long).
    fn embed_mat_into(&self, emb: &QuantMat, token: u32, y: &mut [f32]) -> Result<(), LlamaError> {
        if y.len() != emb.n_cols {
            return Err(LlamaError::Shape(emb.name.clone()));
        }
        let data = self.mat_bytes(emb)?;
        let row = usize::try_from(token).map_err(|_| LlamaError::Shape(emb.name.clone()))?;
        let rb = row_bytes_for(emb.ty, emb.n_cols, &emb.name)?;
        let start = row
            .checked_mul(rb)
            .ok_or_else(|| LlamaError::Shape(emb.name.clone()))?;
        let end = start
            .checked_add(rb)
            .ok_or_else(|| LlamaError::Shape(emb.name.clone()))?;
        let bytes = data
            .get(start..end)
            .ok_or_else(|| LlamaError::Shape(emb.name.clone()))?;
        match emb.ty {
            GgmlType::F32 => dequant_f32_row(emb.n_cols, bytes, y)?,
            GgmlType::F16 => dequant_f16_row(emb.n_cols, bytes, y)?,
            GgmlType::BF16 => dequant_bf16_row(emb.n_cols, bytes, y)?,
            GgmlType::Q2_K => dequant_q2_k_row(emb.n_cols, bytes, y)?,
            GgmlType::Q3_K => dequant_q3_k_row(emb.n_cols, bytes, y)?,
            GgmlType::Q4_1 => dequant_q4_1_row(emb.n_cols, bytes, y)?,
            GgmlType::Q5_0 => dequant_q5_0_row(emb.n_cols, bytes, y)?,
            GgmlType::Q5_1 => dequant_q5_1_row(emb.n_cols, bytes, y)?,
            GgmlType::MXFP4 => dequant_mxfp4_row(emb.n_cols, bytes, y)?,
            GgmlType::NVFP4 => dequant_nvfp4_row(emb.n_cols, bytes, y)?,
            GgmlType::Q1_0 => dequant_q1_0_row(emb.n_cols, bytes, y)?,
            GgmlType::Q2_0 => dequant_q2_0_row(emb.n_cols, bytes, y)?,
            GgmlType::Q4_0 => dequant_q4_0_row(emb.n_cols, bytes, y)?,
            GgmlType::Q8_0 => dequant_q8_0_row(emb.n_cols, bytes, y)?,
            GgmlType::Q8_1 => dequant_q8_1_row(emb.n_cols, bytes, y)?,
            GgmlType::TQ1_0 => dequant_tq1_0_row(emb.n_cols, bytes, y)?,
            GgmlType::TQ2_0 => dequant_tq2_0_row(emb.n_cols, bytes, y)?,
            GgmlType::Q4_K => dequant_q4_k_row(emb.n_cols, bytes, y)?,
            GgmlType::Q5_K => dequant_q5_k_row(emb.n_cols, bytes, y)?,
            GgmlType::Q6_K => dequant_q6_k_row(emb.n_cols, bytes, y)?,
            GgmlType::IQ1_S => dequant_iq1_s_row(emb.n_cols, bytes, y)?,
            GgmlType::IQ1_M => dequant_iq1_m_row(emb.n_cols, bytes, y)?,
            GgmlType::IQ2_XXS => dequant_iq2_xxs_row(emb.n_cols, bytes, y)?,
            GgmlType::IQ2_XS => dequant_iq2_xs_row(emb.n_cols, bytes, y)?,
            GgmlType::IQ2_S => dequant_iq2_s_row(emb.n_cols, bytes, y)?,
            GgmlType::IQ3_XXS => dequant_iq3_xxs_row(emb.n_cols, bytes, y)?,
            GgmlType::IQ3_S => dequant_iq3_s_row(emb.n_cols, bytes, y)?,
            GgmlType::IQ4_NL => dequant_iq4_nl_row(emb.n_cols, bytes, y)?,
            GgmlType::IQ4_XS => dequant_iq4_xs_row(emb.n_cols, bytes, y)?,
            other => {
                return Err(LlamaError::Type {
                    tensor: emb.name.clone(),
                    ty: other.to_i32(),
                })
            }
        }
        Ok(())
    }

    /// Official bloom `token_embd_norm` (`LLM_NORM`) after the embedding lookup.
    fn apply_token_embd_norm(&self, x: &mut [f32]) -> Result<(), LlamaError> {
        match (
            self.token_embd_norm.as_deref(),
            self.token_embd_norm_b.as_deref(),
        ) {
            (Some(w), Some(b)) => layernorm_rows_inplace(x, self.n_embd, w, Some(b), self.rms_eps),
            (None, None) => Ok(()),
            _ => Err(LlamaError::Shape("token_embd_norm".into())),
        }
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
        let mut y = Vec::new();
        self.gemv_into(&self.output, x, &mut y, &mut None)?;
        Ok(y)
    }

    #[cfg(test)]
    fn gemv_token_embd(&self, x: &[f32]) -> Result<Vec<f32>, LlamaError> {
        let mut y = Vec::new();
        self.gemv_into(&self.token_embd, x, &mut y, &mut None)?;
        Ok(y)
    }

    #[cfg(test)]
    fn gemm_output(&self, n_tokens: usize, x: &[f32]) -> Result<Vec<f32>, LlamaError> {
        let mut y = Vec::new();
        self.gemm_into(&self.output, n_tokens, x, &mut y, &mut None)?;
        Ok(y)
    }

    #[cfg(test)]
    fn embed_token(&self, token: u32) -> Result<Vec<f32>, LlamaError> {
        let mut y = vec![0.0f32; self.token_embd.n_cols];
        self.embed_into(token, &mut y)?;
        Ok(y)
    }
}

/// `y[r] = W[r] · x` for exactly `y.len()` rows starting at `w`.
///
/// The dtype dispatch for both the in-thread path and the pool's workers.
/// Rows are independent, so a row range is bit-identical to the whole matrix.
/// `Ok(false)` means `ty` has no GEMV kernel.
fn gemv_rows(
    ty: GgmlType,
    n_cols: usize,
    w: &[u8],
    x: &[f32],
    y: &mut [f32],
) -> Result<bool, QuantError> {
    match ty {
        GgmlType::F32 => gemv_f32(n_cols, w, x, y)?,
        GgmlType::F16 => gemv_f16(n_cols, w, x, y)?,
        GgmlType::BF16 => gemv_bf16(n_cols, w, x, y)?,
        GgmlType::Q2_K => gemv_q2_k_f32(n_cols, w, x, y)?,
        GgmlType::Q3_K => gemv_q3_k_f32(n_cols, w, x, y)?,
        GgmlType::Q4_1 => gemv_q4_1_f32(n_cols, w, x, y)?,
        GgmlType::Q5_0 => gemv_q5_0_f32(n_cols, w, x, y)?,
        GgmlType::Q5_1 => gemv_q5_1_f32(n_cols, w, x, y)?,
        GgmlType::MXFP4 => gemv_mxfp4_f32(n_cols, w, x, y)?,
        GgmlType::NVFP4 => gemv_nvfp4_f32(n_cols, w, x, y)?,
        GgmlType::Q1_0 => gemv_q1_0_f32(n_cols, w, x, y)?,
        GgmlType::Q2_0 => gemv_q2_0_f32(n_cols, w, x, y)?,
        GgmlType::Q4_0 => gemv_q4_0_f32(n_cols, w, x, y)?,
        GgmlType::Q8_0 => gemv_q8_0_f32(n_cols, w, x, y)?,
        GgmlType::Q8_1 => gemv_q8_1_f32(n_cols, w, x, y)?,
        GgmlType::TQ1_0 => gemv_tq1_0_f32(n_cols, w, x, y)?,
        GgmlType::TQ2_0 => gemv_tq2_0_f32(n_cols, w, x, y)?,
        GgmlType::Q4_K => gemv_q4_k_f32(n_cols, w, x, y)?,
        GgmlType::Q5_K => gemv_q5_k_f32(n_cols, w, x, y)?,
        GgmlType::Q6_K => gemv_q6_k_f32(n_cols, w, x, y)?,
        GgmlType::IQ1_S => gemv_iq1_s_f32(n_cols, w, x, y)?,
        GgmlType::IQ1_M => gemv_iq1_m_f32(n_cols, w, x, y)?,
        GgmlType::IQ2_XXS => gemv_iq2_xxs_f32(n_cols, w, x, y)?,
        GgmlType::IQ2_XS => gemv_iq2_xs_f32(n_cols, w, x, y)?,
        GgmlType::IQ2_S => gemv_iq2_s_f32(n_cols, w, x, y)?,
        GgmlType::IQ3_XXS => gemv_iq3_xxs_f32(n_cols, w, x, y)?,
        GgmlType::IQ3_S => gemv_iq3_s_f32(n_cols, w, x, y)?,
        GgmlType::IQ4_NL => gemv_iq4_nl_f32(n_cols, w, x, y)?,
        GgmlType::IQ4_XS => gemv_iq4_xs_f32(n_cols, w, x, y)?,
        _ => return Ok(false),
    }
    Ok(true)
}

/// Packed bytes per matrix row for `ty`.
fn row_bytes_for(ty: GgmlType, n_cols: usize, name: &str) -> Result<usize, LlamaError> {
    let rb = match ty {
        GgmlType::F32 => f32_row_bytes(n_cols)?,
        GgmlType::F16 => f16_row_bytes(n_cols)?,
        GgmlType::BF16 => bf16_row_bytes(n_cols)?,
        GgmlType::Q2_K => q2_k_row_bytes(n_cols)?,
        GgmlType::Q3_K => q3_k_row_bytes(n_cols)?,
        GgmlType::Q4_1 => q4_1_row_bytes(n_cols)?,
        GgmlType::Q5_0 => q5_0_row_bytes(n_cols)?,
        GgmlType::Q5_1 => q5_1_row_bytes(n_cols)?,
        GgmlType::MXFP4 => mxfp4_row_bytes(n_cols)?,
        GgmlType::NVFP4 => nvfp4_row_bytes(n_cols)?,
        GgmlType::Q1_0 => q1_0_row_bytes(n_cols)?,
        GgmlType::Q2_0 => q2_0_row_bytes(n_cols)?,
        GgmlType::Q4_0 => q4_0_row_bytes(n_cols)?,
        GgmlType::Q8_0 => q8_0_row_bytes(n_cols)?,
        GgmlType::Q8_1 => q8_1_row_bytes(n_cols)?,
        GgmlType::TQ1_0 => tq1_0_row_bytes(n_cols)?,
        GgmlType::TQ2_0 => tq2_0_row_bytes(n_cols)?,
        GgmlType::Q4_K => q4_k_row_bytes(n_cols)?,
        GgmlType::Q5_K => q5_k_row_bytes(n_cols)?,
        GgmlType::Q6_K => q6_k_row_bytes(n_cols)?,
        GgmlType::IQ1_S => iq1_s_row_bytes(n_cols)?,
        GgmlType::IQ1_M => iq1_m_row_bytes(n_cols)?,
        GgmlType::IQ2_XXS => iq2_xxs_row_bytes(n_cols)?,
        GgmlType::IQ2_XS => iq2_xs_row_bytes(n_cols)?,
        GgmlType::IQ2_S => iq2_s_row_bytes(n_cols)?,
        GgmlType::IQ3_XXS => iq3_xxs_row_bytes(n_cols)?,
        GgmlType::IQ3_S => iq3_s_row_bytes(n_cols)?,
        GgmlType::IQ4_NL => iq4_nl_row_bytes(n_cols)?,
        GgmlType::IQ4_XS => iq4_xs_row_bytes(n_cols)?,
        other => {
            return Err(LlamaError::Type {
                tensor: name.into(),
                ty: other.to_i32(),
            })
        }
    };
    Ok(rb)
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

fn token_row_mut<'a>(
    x: &'a mut [f32],
    t: usize,
    width: usize,
    what: &'static str,
) -> Result<&'a mut [f32], LlamaError> {
    let start = t
        .checked_mul(width)
        .ok_or_else(|| LlamaError::Shape(what.into()))?;
    let end = start
        .checked_add(width)
        .ok_or_else(|| LlamaError::Shape(what.into()))?;
    x.get_mut(start..end)
        .ok_or_else(|| LlamaError::Shape(what.into()))
}

fn rmsnorm_rows_inplace(
    x: &mut [f32],
    width: usize,
    w: &[f32],
    eps: f32,
) -> Result<(), LlamaError> {
    if width == 0 || !x.len().is_multiple_of(width) {
        return Err(LlamaError::Shape("rmsnorm".into()));
    }
    for row in x.chunks_mut(width) {
        rmsnorm_inplace(row, w, eps)?;
    }
    Ok(())
}

/// Official Qwen3Next: split joint Q+gate (`n_embd_head * 2` per head) into
/// query and gate, compacting the query half of `q` in place.
///
/// Head `h` of token `t` starts at `t*2*n_head*hd + h*2*hd` and has to end up at
/// `t*n_head*hd + h*hd`, which is never later in the buffer, so a forward walk
/// only ever writes behind what it has already read.
fn split_qwen3next_q_gate_into(
    q: &mut Vec<f32>,
    gate: &mut Vec<f32>,
    n_tokens: usize,
    n_head: usize,
    hd: usize,
) -> Result<(), LlamaError> {
    let q_width = n_head
        .checked_mul(hd)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| LlamaError::Shape("qwen3next q gate".into()))?;
    let q_out_w = n_head
        .checked_mul(hd)
        .ok_or_else(|| LlamaError::Shape("qwen3next q gate".into()))?;
    if hd == 0 || n_head == 0 || !q.len().is_multiple_of(q_width) || q.len() / q_width != n_tokens {
        return Err(LlamaError::Shape("qwen3next q gate".into()));
    }
    let n_out = n_tokens
        .checked_mul(q_out_w)
        .ok_or_else(|| LlamaError::Shape("qwen3next q gate".into()))?;
    fit(gate, n_out);
    for t in 0..n_tokens {
        for h in 0..n_head {
            let src = t
                .saturating_mul(q_width)
                .saturating_add(h.saturating_mul(hd.saturating_mul(2)));
            let dst = t
                .saturating_mul(q_out_w)
                .saturating_add(h.saturating_mul(hd));
            let g_src = q
                .get(src.saturating_add(hd)..src.saturating_add(hd.saturating_mul(2)))
                .ok_or_else(|| LlamaError::Shape("qwen3next q gate".into()))?;
            let g_dst = gate
                .get_mut(dst..dst.saturating_add(hd))
                .ok_or_else(|| LlamaError::Shape("qwen3next q gate".into()))?;
            for (d, s) in g_dst.iter_mut().zip(g_src.iter()) {
                *d = *s;
            }
            if src.saturating_add(hd) > q.len() || dst.saturating_add(hd) > q.len() {
                return Err(LlamaError::Shape("qwen3next q gate".into()));
            }
            q.copy_within(src..src.saturating_add(hd), dst);
        }
    }
    q.truncate(n_out);
    Ok(())
}

fn add_bias_rows(x: &mut [f32], width: usize, bias: Option<&[f32]>) -> Result<(), LlamaError> {
    let Some(b) = bias else {
        return Ok(());
    };
    if b.len() != width || width == 0 || !x.len().is_multiple_of(width) {
        return Err(LlamaError::Shape("vector add".into()));
    }
    for row in x.chunks_mut(width) {
        for (xv, bv) in row.iter_mut().zip(b.iter()) {
            *xv += *bv;
        }
    }
    Ok(())
}

/// Official `ggml_soft_max_ext` ALiBi slope for head `head` of `n_head`.
///
/// `n_head_log2 = 1 << floor(log2(n_head))`, `m0 = 2 ** (-max_bias / n_head_log2)`,
/// `m1 = 2 ** (-(max_bias/2) / n_head_log2)`, slope is `m0**(h+1)` when
/// `h < n_head_log2` else `m1**(2*(h-n_head_log2)+1)`. `max_bias <= 0` is off.
fn alibi_slope(head: usize, n_head: usize, max_bias: f32) -> f32 {
    if max_bias <= 0.0 || n_head == 0 {
        return 0.0;
    }
    let n_head_log2 = 1usize << n_head.ilog2();
    let n_head_log2_f = f32::from(u16::try_from(n_head_log2).unwrap_or(1));
    let m0 = 2.0_f32.powf(-max_bias / n_head_log2_f);
    let m1 = 2.0_f32.powf(-(max_bias / 2.0) / n_head_log2_f);
    if head < n_head_log2 {
        let exp = i32::try_from(head.saturating_add(1)).unwrap_or(i32::MAX);
        m0.powi(exp)
    } else {
        let odd = head
            .saturating_sub(n_head_log2)
            .saturating_mul(2)
            .saturating_add(1);
        let exp = i32::try_from(odd).unwrap_or(i32::MAX);
        m1.powi(exp)
    }
}

/// Softmax(`score_scale` QK^T) V for one token's heads, accumulated into `out`.
///
/// `q` and `out` are `n_head * hd` long; `scores` is grown to `seq`. GQA maps
/// query head `hq` to KV head `hq / (n_head / n_head_kv)`.
///
/// `score_scale` is `1/sqrt(hd)` for every arch except official phi2, which
/// scales Q after RoPE instead and passes 1.0 here, and official gemma3n /
/// gemma4, which pass `hparams.f_attention_scale = 1.0`.
///
/// `alibi_bias` is official bloom `f_max_alibi_bias` (`8`); `0` disables ALiBi.
///
/// `attn_softcap` is official gemma2 `attn_logit_softcapping`; `0` disables.
/// Applied after scale/ALiBi and before the SWA mask (tanh of `-inf` is `-1`).
///
/// `n_swa` is official `LLAMA_SWA_TYPE_STANDARD` window (`p1 - p0 >= n_swa`);
/// `0` is full causal. Pass `0` on gemma2 dense layers (`set_swa_pattern`).
#[expect(
    clippy::too_many_arguments,
    reason = "cache halves, layout, scale, ALiBi, gemma2 softcap/SWA and both scratch buffers are all per-call"
)]
fn attend_query(
    cache_k: &[f32],
    cache_v: &[f32],
    layer: usize,
    q: &[f32],
    geom: &KvGeom<'_>,
    seq: usize,
    score_scale: f32,
    alibi_bias: f32,
    attn_softcap: f32,
    n_swa: usize,
    scores: &mut Vec<f32>,
    out: &mut [f32],
) -> Result<(), LlamaError> {
    let n_head_kv = geom.n_head_kv;
    let hd = geom.hd;
    if n_head_kv == 0 || hd == 0 || q.len() != out.len() || !q.len().is_multiple_of(hd) {
        return Err(LlamaError::Shape("gqa".into()));
    }
    let n_head = q.len() / hd;
    let gqa = n_head / n_head_kv;
    if gqa == 0 {
        return Err(LlamaError::Shape("gqa".into()));
    }
    fit(scores, seq);
    let qpos = seq.saturating_sub(1);
    let qpos_f = f32::from(u16::try_from(qpos).unwrap_or(u16::MAX));
    for (hq, (qvec, dst)) in q.chunks(hd).zip(out.chunks_mut(hd)).enumerate() {
        let hkv = hq / gqa;
        let slope = if alibi_bias > 0.0 {
            alibi_slope(hq, n_head, alibi_bias)
        } else {
            0.0
        };
        for (t, score) in scores.iter_mut().enumerate() {
            let kv = kv_at(cache_k, layer, hkv, geom, t)?;
            let mut dot = 0.0f32;
            for (a, b) in qvec.iter().zip(kv.iter()) {
                dot += *a * *b;
            }
            *score = dot * score_scale;
            if alibi_bias > 0.0 {
                let t_f = f32::from(u16::try_from(t).unwrap_or(u16::MAX));
                *score += slope * (t_f - qpos_f);
            }
            if attn_softcap > 0.0 {
                *score = attn_softcap * (*score / attn_softcap).tanh();
            }
            if n_swa > 0 && qpos.saturating_sub(t) >= n_swa {
                *score = f32::NEG_INFINITY;
            }
        }
        softmax(scores);
        for d in dst.iter_mut() {
            *d = 0.0;
        }
        for (t, st) in scores.iter().enumerate() {
            let vv = kv_at(cache_v, layer, hkv, geom, t)?;
            for (a, b) in dst.iter_mut().zip(vv.iter()) {
                *a += *st * *b;
            }
        }
    }
    Ok(())
}

/// Official Llama4 `ggml_rms_norm` without a weight (`Llama4TextL2Norm`).
/// Gemma3n applies the same op to V (`n_embd_head` rows) before KV store.
/// Official gemma4.cpp uses the same unweighted V RMSNorm.
fn rmsnorm_unweighted_inplace(x: &mut [f32], eps: f32) {
    let mut ss = 0.0f32;
    for v in x.iter() {
        ss += *v * *v;
    }
    let n = f32::from(u16::try_from(x.len()).unwrap_or(1));
    let rms = (ss / n + eps).sqrt();
    let inv = if rms > 0.0 { 1.0 / rms } else { 0.0 };
    for xv in x.iter_mut() {
        *xv *= inv;
    }
}

fn rmsnorm_unweighted_rows_inplace(
    x: &mut [f32],
    width: usize,
    eps: f32,
) -> Result<(), LlamaError> {
    if width == 0 || !x.len().is_multiple_of(width) {
        return Err(LlamaError::Shape("rmsnorm".into()));
    }
    for row in x.chunks_mut(width) {
        rmsnorm_unweighted_inplace(row, eps);
    }
    Ok(())
}

fn altup_stream_off(i: usize, n: usize, n_embd: usize) -> Result<usize, LlamaError> {
    i.checked_mul(n)
        .and_then(|v| v.checked_mul(n_embd))
        .ok_or_else(|| LlamaError::Shape("gemma3n stream".into()))
}

fn altup_stream(buf: &[f32], i: usize, n: usize, n_embd: usize) -> Result<&[f32], LlamaError> {
    let off = altup_stream_off(i, n, n_embd)?;
    let width = n.saturating_mul(n_embd);
    buf.get(off..off.saturating_add(width))
        .ok_or_else(|| LlamaError::Shape("gemma3n stream".into()))
}

fn altup_stream_mut(
    buf: &mut [f32],
    i: usize,
    n: usize,
    n_embd: usize,
) -> Result<&mut [f32], LlamaError> {
    let off = altup_stream_off(i, n, n_embd)?;
    let width = n.saturating_mul(n_embd);
    buf.get_mut(off..off.saturating_add(width))
        .ok_or_else(|| LlamaError::Shape("gemma3n stream".into()))
}

fn altup_token(
    buf: &[f32],
    i: usize,
    t: usize,
    n: usize,
    n_embd: usize,
) -> Result<&[f32], LlamaError> {
    let stream = altup_stream(buf, i, n, n_embd)?;
    token_row(stream, t, n_embd, "gemma3n stream")
}

fn altup_token_mut(
    buf: &mut [f32],
    i: usize,
    t: usize,
    n: usize,
    n_embd: usize,
) -> Result<&mut [f32], LlamaError> {
    let tok_off = t
        .checked_mul(n_embd)
        .ok_or_else(|| LlamaError::Shape("gemma3n stream".into()))?;
    let off = altup_stream_off(i, n, n_embd)?
        .checked_add(tok_off)
        .ok_or_else(|| LlamaError::Shape("gemma3n stream".into()))?;
    buf.get_mut(off..off.saturating_add(n_embd))
        .ok_or_else(|| LlamaError::Shape("gemma3n stream".into()))
}

fn per_layer_at(
    buf: &[f32],
    li: usize,
    t: usize,
    n_ea: usize,
    n_layer: usize,
) -> Result<&[f32], LlamaError> {
    let off = t
        .checked_mul(n_layer)
        .and_then(|v| v.checked_mul(n_ea))
        .and_then(|v| li.checked_mul(n_ea).and_then(|o| v.checked_add(o)))
        .ok_or_else(|| LlamaError::Shape("gemma3n per_layer".into()))?;
    buf.get(off..off.saturating_add(n_ea))
        .ok_or_else(|| LlamaError::Shape("gemma3n per_layer".into()))
}

/// Official gemma3n.cpp `calc_magnitude` then `x * target / new` (no epsilon).
fn scale_rows_to_match(src: &[f32], dst: &mut [f32], n_embd: usize) -> Result<(), LlamaError> {
    if src.len() != dst.len() || n_embd == 0 || !src.len().is_multiple_of(n_embd) {
        return Err(LlamaError::Shape("gemma3n magnitude".into()));
    }
    for (s, d) in src.chunks(n_embd).zip(dst.chunks_mut(n_embd)) {
        let mut ss = 0.0f32;
        let mut ds = 0.0f32;
        for (sv, dv) in s.iter().zip(d.iter()) {
            ss += *sv * *sv;
            ds += *dv * *dv;
        }
        let mag_s = ss.sqrt();
        let mag_d = ds.sqrt();
        let scale = if mag_d > 0.0 { mag_s / mag_d } else { 0.0 };
        for v in d.iter_mut() {
            *v *= scale;
        }
    }
    Ok(())
}

/// Official gemma3n.cpp `gaussian_topk`: Bessel std over `n_ff-1`, then ReLU.
fn gaussian_topk_inplace(x: &mut [f32], n_ff: usize) -> Result<(), LlamaError> {
    if n_ff < 2 || !x.len().is_multiple_of(n_ff) {
        return Err(LlamaError::Shape("gaussian_topk".into()));
    }
    let n_f = f32::from(u16::try_from(n_ff).unwrap_or(1));
    let denom = f32::from(u16::try_from(n_ff.saturating_sub(1)).unwrap_or(1));
    for row in x.chunks_mut(n_ff) {
        let mut sum = 0.0f32;
        for v in row.iter() {
            sum += *v;
        }
        let mean = sum / n_f;
        let mut var = 0.0f32;
        for v in row.iter() {
            let d = *v - mean;
            var += d * d;
        }
        let std = (var / denom).sqrt();
        let cutoff = mean + std * GEMMA3N_SPARSITY_STD_MUL;
        for v in row.iter_mut() {
            *v = (*v - cutoff).max(0.0);
        }
    }
    Ok(())
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

/// ggml `ggml_argsort_top_k` descending; ties keep the lower index. Writes the
/// selected indices into `order`.
///
/// `sort_unstable_by` rather than `sort_by`: the comparator is a total order
/// (ties break on index) so the result is the same, and it does not allocate.
fn topk_into(logits: &[f32], k: usize, order: &mut Vec<usize>) -> Result<(), LlamaError> {
    if k == 0 || k > logits.len() {
        return Err(LlamaError::Shape("expert_used_count".into()));
    }
    order.clear();
    order.extend(0..logits.len());
    order.sort_unstable_by(|&a, &b| {
        let va = logits.get(a).copied().unwrap_or(f32::NEG_INFINITY);
        let vb = logits.get(b).copied().unwrap_or(f32::NEG_INFINITY);
        match vb.partial_cmp(&va) {
            Some(core::cmp::Ordering::Equal) | None => a.cmp(&b),
            Some(ord) => ord,
        }
    });
    order.truncate(k);
    Ok(())
}

fn token_expert_weight(moe: &MoeScratch, t: usize, n_used: usize, expert: usize) -> Option<f32> {
    let start = t.saturating_mul(n_used);
    for j in 0..n_used {
        let idx = start.saturating_add(j);
        if moe.sel_e.get(idx).copied() == Some(expert) {
            return moe.sel_w.get(idx).copied();
        }
    }
    None
}

fn fill_router_weights(logits: &[f32], order: &[usize], weights: &mut Vec<f32>, norm_w: bool) {
    weights.clear();
    let mut wsum = 0.0f32;
    for &e in order {
        let w = logits.get(e).copied().unwrap_or(0.0);
        weights.push(w);
        wsum += w;
    }
    if !norm_w {
        return;
    }
    if wsum < MOE_NORM_W_CLAMP {
        wsum = MOE_NORM_W_CLAMP;
    }
    for w in weights.iter_mut() {
        *w /= wsum;
    }
}

/// Observe this layer's router events and prefetch predicted destinations
/// **before** grouped expert GEMM so H2D of L+1 can overlap this layer's
/// compute. Unknown catalog keys are skipped. [`Prefetch::None`] is demand
/// paging only.
fn prefetch_selected(
    moe_trace: &mut MoeTraceBuf,
    store: &mut Option<LiveStore>,
    n_tokens: usize,
    n_used: usize,
    sel_e: &[usize],
) {
    for t in 0..n_tokens {
        bind_row(store, &moe_trace.row_seq, moe_trace.sequence, t);
        let start = t.saturating_mul(n_used);
        let experts = sel_e
            .get(start..start.saturating_add(n_used))
            .unwrap_or(&[]);
        moe_trace.prefetch_experts(store, t, experts);
    }
}

/// Fault in this layer's unique routed experts before grouped GEMM.
///
/// Skipped when the unique set is larger than the cache (`slots`) so a tight
/// LRU still demand-pages one expert at a time. Distinct keys H2D on the
/// first using sequence's copy stream ([`LiveStore::bind_sequence`]).
fn prefetch_routed(
    moe_trace: &MoeTraceBuf,
    store: &mut Option<LiveStore>,
    n_tokens: usize,
    n_used: usize,
    sel_e: &[usize],
) {
    if matches!(moe_trace.policy.prefetch, Prefetch::None) {
        return;
    }
    let Some(s) = store.as_mut() else {
        return;
    };
    let mut seen = BTreeSet::new();
    let mut all = Vec::new();
    let mut owner = BTreeMap::new();
    for t in 0..n_tokens {
        let seq = moe_trace.seq_of(t);
        let start = t.saturating_mul(n_used);
        let experts = sel_e
            .get(start..start.saturating_add(n_used))
            .unwrap_or(&[]);
        for e in experts {
            let Ok(ex) = u32::try_from(*e) else {
                continue;
            };
            let k = ExpertKey::new(moe_trace.layer, ex);
            if seen.insert(k) {
                all.push(k);
            }
            let _owned = owner.entry(k).or_insert(seq);
        }
    }
    if all.is_empty() {
        return;
    }
    if let Some(slots) = s.slots() {
        if all.len() > slots {
            return;
        }
    }
    let mut by_seq: BTreeMap<u64, Vec<ExpertKey>> = BTreeMap::new();
    for (k, seq) in owner {
        by_seq.entry(seq).or_default().push(k);
    }
    for (seq, keys) in by_seq {
        s.bind_sequence(seq);
        match s.prefetch(&keys) {
            Ok(_n) => {}
            Err(_e) => {}
        }
    }
}

fn bind_row(store: &mut Option<LiveStore>, row_seq: &[u64], sequence: u64, token_off: usize) {
    let seq = row_seq.get(token_off).copied().unwrap_or(sequence);
    if let Some(s) = store.as_mut() {
        s.bind_sequence(seq);
    }
}

fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) -> Result<(), LlamaError> {
    if x.len() != w.len() {
        return Err(LlamaError::Shape("rmsnorm".into()));
    }
    let mut ss = 0.0f32;
    for v in x.iter() {
        ss += *v * *v;
    }
    let n = f32::from(u16::try_from(x.len()).unwrap_or(1));
    let rms = (ss / n + eps).sqrt();
    let inv = if rms > 0.0 { 1.0 / rms } else { 0.0 };
    for (xv, wv) in x.iter_mut().zip(w.iter()) {
        *xv = *xv * inv * *wv;
    }
    Ok(())
}

/// Official `ggml_compute_forward_norm_f32` + `build_norm` `LLM_NORM` (weight,
/// optional bias), writing over its input.
///
/// Mean and variance are both taken before any element is written, so this
/// produces the same bits as normalizing into a fresh buffer.
fn layernorm_inplace(
    x: &mut [f32],
    w: &[f32],
    b: Option<&[f32]>,
    eps: f32,
) -> Result<(), LlamaError> {
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
    for v in x.iter() {
        sum += *v;
    }
    let mean = if n > 0.0 { sum / n } else { 0.0 };
    let mut var = 0.0f32;
    for v in x.iter() {
        let d = *v - mean;
        var += d * d;
    }
    if n > 0.0 {
        var /= n;
    }
    let scale = 1.0 / (var + eps).sqrt();
    for (i, (xv, wv)) in x.iter_mut().zip(w.iter()).enumerate() {
        let mut y = (*xv - mean) * scale * *wv;
        if let Some(b) = b {
            if let Some(bv) = b.get(i) {
                y += *bv;
            }
        }
        *xv = y;
    }
    Ok(())
}

/// [`layernorm_rows`] writing over its input.
fn layernorm_rows_inplace(
    x: &mut [f32],
    width: usize,
    w: &[f32],
    b: Option<&[f32]>,
    eps: f32,
) -> Result<(), LlamaError> {
    if width == 0 || !x.len().is_multiple_of(width) {
        return Err(LlamaError::Shape("layernorm".into()));
    }
    for row in x.chunks_mut(width) {
        layernorm_inplace(row, w, b, eps)?;
    }
    Ok(())
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

/// Official `GGML_ROPE_TYPE_NEOX`: `rotate_pairs(n_dims, n_dims/2, .., scale = 2)`.
///
/// Same theta walk as [`rope`]; only the element pairing differs. NORM rotates
/// adjacent lanes `(i0, i0+1)`; NEOX rotates `(i0/2, i0/2 + n_dims/2)`.
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
    let mut i0 = 0usize;
    while i0 + 1 < n {
        let ic = i0 / 2;
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
        theta *= theta_scale;
        i0 += 2;
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

/// Official `llama_model_rope_type` (`src/llama-model.cpp`) for the architectures
/// this crate loads: `true` selects `LLAMA_ROPE_TYPE_NEOX`, `false` selects
/// `LLAMA_ROPE_TYPE_NORM`.
///
/// `qwen2vl` (MROPE) and `qwen3vl` / `qwen35` (IMROPE) are not listed here: they
/// carry `rope.dimension_sections` and take the [`rope_multi`] path, which already
/// rotates on the NEOX `n_dims/2` offset.
///
/// `mistral` is not an official `LLM_ARCH_NAMES` entry (official Mistral GGUF is
/// `architecture=llama`); it is Llama-family, so it stays NORM.
fn rope_is_neox(arch: &str) -> bool {
    match arch {
        // LLAMA / LLAMA4 => LLAMA_ROPE_TYPE_NORM.
        "llama" | "llama4" | "mistral" => false,
        // QWEN2 / QWEN2MOE / QWEN3 / QWEN3MOE / QWEN3NEXT / PHI2 / PHI3 / GEMMA /
        // GEMMA2 / GEMMA3 => LLAMA_ROPE_TYPE_NEOX.
        "qwen2" | "qwen2moe" | "qwen3" | "qwen3moe" | "qwen3next" | "phi2" | "phi3" | "gemma"
        | "gemma2" | "gemma3" | "gemma3n" | "gemma4" => true,
        // MROPE / IMROPE arches reach `rope_multi`; the flag is unused for them.
        _ => false,
    }
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

fn silu_inplace(x: &mut [f32]) {
    for xv in x.iter_mut() {
        *xv = *xv / (1.0 + (-*xv).exp());
    }
}

/// Official llama.cpp Gemma FFN is `LLM_FFN_GELU` → `ggml_gelu` (tanh approx).
fn ggml_gelu_f32(x: f32) -> f32 {
    let coef_a = 44_715.0 / 1_000_000.0;
    let sqrt_2_over_pi = (2.0 / core::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (sqrt_2_over_pi * x * (1.0 + coef_a * x * x)).tanh())
}

fn gelu_inplace(x: &mut [f32]) {
    for xv in x.iter_mut() {
        *xv = ggml_gelu_f32(*xv);
    }
}

fn ffn_gate_act_inplace(x: &mut [f32], use_gelu: bool) {
    if use_gelu {
        gelu_inplace(x);
    } else {
        silu_inplace(x);
    }
}

fn tanh_softcap_inplace(x: &mut [f32], cap: f32) {
    if cap <= 0.0 {
        return;
    }
    for v in x.iter_mut() {
        *v = cap * (*v / cap).tanh();
    }
}

/// Official `llama_hparams::set_swa_pattern(n_pattern, dense_first=false)`:
/// `is_swa(il) = n_pattern == 0 || (il % n_pattern < n_pattern - 1)`.
/// Period 2: even layers are SWA, odd layers are dense. Layer 0 is SWA.
fn gemma2_is_swa(il: usize, period: u32) -> bool {
    if period == 0 {
        true
    } else {
        let il = u32::try_from(il).unwrap_or(u32::MAX);
        il % period < period.saturating_sub(1)
    }
}

/// Official gemma4.cpp `get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN)`.
/// Convert writes a per-layer bool array; a scalar broadcasts (`!= 0` is SWA).
fn load_gemma4_is_swa(g: &Gguf, arch: &str, n_layer: usize) -> Result<Vec<bool>, LlamaError> {
    let key = arch_key(arch, "attention.sliding_window_pattern");
    match g.kv(&key) {
        Some(Kv::Array { items, .. }) => {
            if items.len() != n_layer {
                return Err(LlamaError::Shape(key));
            }
            let mut out = Vec::with_capacity(n_layer);
            for it in items {
                match it {
                    Kv::Bool(b) => out.push(*b),
                    Kv::U32(v) => out.push(*v != 0),
                    Kv::I32(v) => out.push(*v != 0),
                    _ => return Err(LlamaError::Shape(key)),
                }
            }
            Ok(out)
        }
        Some(Kv::U32(v)) => Ok(vec![*v != 0; n_layer]),
        Some(Kv::I32(v)) => Ok(vec![*v != 0; n_layer]),
        Some(Kv::Bool(b)) => Ok(vec![*b; n_layer]),
        None => Err(LlamaError::MissingKv(key)),
        _ => Err(LlamaError::Shape(key)),
    }
}

/// Official `llama_hparams::has_kv`: when `n_layer_kv_from_start >= 0`, the first
/// that many layers own KV. A negative value (unset) means every layer has KV.
fn gemma4_has_kv(n_layer_kv_from_start: i32, il: usize) -> bool {
    match usize::try_from(n_layer_kv_from_start) {
        Ok(n) => il < n,
        Err(_) => true,
    }
}

/// Official gemma4 / gemma3n `layer_reuse_cb` when `!has_kv(il)`:
/// `n_from_start` minus 2 on SWA, minus 1 on global.
/// llama.cpp `create_memory` asserts `n_layer_kv_from_start >= 2`.
fn gemma4_kv_slot(
    n_layer_kv_from_start: i32,
    il: usize,
    is_swa: bool,
) -> Result<usize, LlamaError> {
    if gemma4_has_kv(n_layer_kv_from_start, il) {
        return Ok(il);
    }
    if n_layer_kv_from_start < 2 {
        return Err(LlamaError::Shape(arch_key(
            "gemma4",
            "attention.shared_kv_layers",
        )));
    }
    let delta = if is_swa { 2 } else { 1 };
    let donor = n_layer_kv_from_start - delta;
    let donor = usize::try_from(donor)
        .map_err(|_| LlamaError::Shape(arch_key("gemma4", "attention.shared_kv_layers")))?;
    if !gemma4_has_kv(n_layer_kv_from_start, donor) {
        return Err(LlamaError::Shape(arch_key(
            "gemma4",
            "attention.shared_kv_layers",
        )));
    }
    Ok(donor)
}

fn gemma4_shared_kv_omitted(rest: &str) -> bool {
    matches!(
        rest,
        "attn_k.weight" | "attn_v.weight" | "attn_k_norm.weight"
    )
}

/// Official gemma4.cpp required KV that the writer-tiny honors, plus
/// refusal for mixed SWA head dim (not this item).
/// Returns `(embedding_length_per_layer_input, n_layer_kv_from_start)`.
/// `n_pl == 0` is dense/MoE without PLE. Unset `shared_kv_layers` is 0
/// (every layer has KV). Nonzero shared KV requires `n_from_start >= 2`.
fn load_gemma4_hparams(
    g: &Gguf,
    arch: &str,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
) -> Result<(usize, i32), LlamaError> {
    let n_pl = require_usize(g, arch, "embedding_length_per_layer_input")?;
    let n_shared = g
        .kv_u32(&arch_key(arch, "attention.shared_kv_layers"))
        .unwrap_or(0);
    let n_layer_i =
        i32::try_from(n_layer).map_err(|_| LlamaError::Shape(arch_key(arch, "block_count")))?;
    let n_shared_i = i32::try_from(n_shared)
        .map_err(|_| LlamaError::Shape(arch_key(arch, "attention.shared_kv_layers")))?;
    let n_from_start = n_layer_i - n_shared_i;
    if n_shared != 0 && n_from_start < 2 {
        return Err(LlamaError::Shape(arch_key(
            arch,
            "attention.shared_kv_layers",
        )));
    }
    if n_head == 0 || !n_embd.is_multiple_of(n_head) {
        return Err(LlamaError::Shape(arch_key(arch, "attention.head_count")));
    }
    let hd = n_embd / n_head;
    let k_swa = require_usize(g, arch, "attention.key_length_swa")?;
    let v_swa = require_usize(g, arch, "attention.value_length_swa")?;
    if k_swa != v_swa || k_swa != hd {
        return Err(LlamaError::Shape(arch_key(
            arch,
            "attention.key_length_swa",
        )));
    }
    let k = g.kv_u32(&arch_key(arch, "attention.key_length"));
    let v = g.kv_u32(&arch_key(arch, "attention.value_length"));
    if let (Some(k), Some(v)) = (k, v) {
        if k != v {
            return Err(LlamaError::Shape(arch_key(arch, "attention.key_length")));
        }
    }
    Ok((n_pl, n_from_start))
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

/// `dst = a + b`, growing `dst` to fit. `dst` may be neither `a` nor `b`.
fn add_into(dst: &mut Vec<f32>, a: &[f32], b: &[f32]) -> Result<(), LlamaError> {
    if a.len() != b.len() {
        return Err(LlamaError::Shape("vector add".into()));
    }
    fit(dst, a.len());
    for ((d, av), bv) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
        *d = *av + *bv;
    }
    Ok(())
}

/// `x += r`, elementwise.
fn add_assign(x: &mut [f32], r: &[f32]) -> Result<(), LlamaError> {
    if x.len() != r.len() {
        return Err(LlamaError::Shape("vector add".into()));
    }
    for (xv, rv) in x.iter_mut().zip(r.iter()) {
        *xv += *rv;
    }
    Ok(())
}

/// Scatter one token's `n_head_kv * hd` row into the cache.
fn store_kv(
    cache: &mut [f32],
    layer: usize,
    geom: &KvGeom<'_>,
    t: usize,
    row: &[f32],
) -> Result<(), LlamaError> {
    let n_head_kv = geom.n_head_kv;
    let hd = geom.hd;
    if hd == 0 || row.len() != n_head_kv.saturating_mul(hd) {
        return Err(LlamaError::Shape("kv store".into()));
    }
    for (h, head) in row.chunks(hd).enumerate() {
        let off = geom
            .offset(layer, h, t)
            .ok_or_else(|| LlamaError::Shape("kv offset".into()))?;
        let dst = cache
            .get_mut(off..off.saturating_add(hd))
            .ok_or_else(|| LlamaError::Shape("kv store".into()))?;
        for (d, s) in dst.iter_mut().zip(head.iter()) {
            *d = *s;
        }
    }
    Ok(())
}

fn kv_at<'a>(
    cache: &'a [f32],
    layer: usize,
    head: usize,
    geom: &KvGeom<'_>,
    t: usize,
) -> Result<&'a [f32], LlamaError> {
    let off = geom
        .offset(layer, head, t)
        .ok_or_else(|| LlamaError::Shape("kv offset".into()))?;
    cache
        .get(off..off.saturating_add(geom.hd))
        .ok_or_else(|| LlamaError::Shape("kv load".into()))
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
            let d = crate::fp16::oracle_load_f16_le(wb).unwrap();
            let minv = crate::fp16::oracle_load_f16_le(&wb[2..]).unwrap();
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
        crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]))
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
            let d = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
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
            let d = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
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
            let d = crate::fp16::oracle_f16_to_f32(bits);
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
            let d = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
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
            let d = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
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
            let d = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
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
            let d = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
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
            let d = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
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
            let d = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
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
            let d = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[80], wb[81]]));
            let minv = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[82], wb[83]]));
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
            let d_all = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[108], wb[109]]));
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
            let d = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let m = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[2], wb[3]]));
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
            let d = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
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
            let d = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let m = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[2], wb[3]]));
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
            let d = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
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
            let d = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
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
            let d = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[52], wb[53]]));
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
            let d = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[64], wb[65]]));
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
            let d = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let qs = &wb[4..];
            let yo = b * QK8_1;
            for j in 0..QK8_1 {
                y[yo + j] = f32::from(i8::from_le_bytes([qs[j]])) * d;
            }
        }
        y
    }

    /// ggml `dequantize_row_q4_0` (oracle). `block_q4_0` is fp16 `d` then 16
    /// packed nibble bytes, 18 total; `y[j] = ((qs[j] & 0xF) - 8) * d` and
    /// `y[j + 16] = ((qs[j] >> 4) - 8) * d`. Independent of `dequant_q4_0_row`.
    fn dequant_q4_0_row_oracle(w: &[u8]) -> Vec<f32> {
        let nblocks = w.len() / crate::quant::Q4_0_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK4_0];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::Q4_0_BLOCK..(b + 1) * crate::quant::Q4_0_BLOCK];
            let d = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let qs = &wb[2..];
            let yo = b * QK4_0;
            for j in 0..(QK4_0 / 2) {
                y[yo + j] = ((i32::from(qs[j] & 0x0f) - 8) as f32) * d;
                y[yo + j + 16] = ((i32::from(qs[j] >> 4) - 8) as f32) * d;
            }
        }
        y
    }

    /// ggml `dequantize_row_q8_0` (oracle). `block_q8_0` is fp16 `d` then `qs[32]`
    /// int8, 34 bytes; `y = q * d`. Independent of crate `dequant_q8_0_row`.
    fn dequant_q8_0_row_oracle(w: &[u8]) -> Vec<f32> {
        let nblocks = w.len() / crate::quant::Q8_0_BLOCK;
        let mut y = vec![0.0f32; nblocks * QK8_0];
        for b in 0..nblocks {
            let wb = &w[b * crate::quant::Q8_0_BLOCK..(b + 1) * crate::quant::Q8_0_BLOCK];
            let d = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let qs = &wb[2..];
            let yo = b * QK8_0;
            for j in 0..QK8_0 {
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
            let d = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let minv = crate::fp16::oracle_f16_to_f32(u16::from_le_bytes([wb[2], wb[3]]));
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
            let d = crate::fp16::oracle_load_f16_le(&wb[208..]).unwrap();
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
            GgmlType::Q4_0 => {
                let rb = (n_cols / QK4_0) * crate::quant::Q4_0_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_q4_0_row_oracle(&t.data[r * rb..(r + 1) * rb]);
                    *yv = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            }
            GgmlType::Q8_0 => {
                let rb = (n_cols / QK8_0) * crate::quant::Q8_0_BLOCK;
                for (r, yv) in y.iter_mut().enumerate() {
                    let row = dequant_q8_0_row_oracle(&t.data[r * rb..(r + 1) * rb]);
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
            GgmlType::Q4_0 => {
                let rb = (n_cols / QK4_0) * crate::quant::Q4_0_BLOCK;
                dequant_q4_0_row_oracle(&t.data[row * rb..(row + 1) * rb])
            }
            GgmlType::Q8_0 => {
                let rb = (n_cols / QK8_0) * crate::quant::Q8_0_BLOCK;
                dequant_q8_0_row_oracle(&t.data[row * rb..(row + 1) * rb])
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

    fn tname(layer: usize, suffix: &str) -> String {
        format!("blk.{layer}.{suffix}")
    }

    /// Official `build_moe_ffn` on one token: softmax, then top-k; SwiGLU;
    /// weights after the expert with `norm_w` clamp `2^-14`. Used by official
    /// llama MoE and official qwen3moe.cpp (`norm_w=true`).
    fn oracle_softmax_norm_w_moe(g: &Gguf, arch: &str, xn: &[f32], layer: usize) -> Vec<f32> {
        let logits = oracle_gemv(g.tensor(&tname(layer, "ffn_gate_inp.weight")).unwrap(), xn);
        oracle_softmax_norm_w_moe_from_logits(g, arch, xn, layer, &logits, false)
    }

    /// Same `build_moe_ffn` walk as [`oracle_softmax_norm_w_moe`] with precomputed
    /// router logits and SILU or GELU expert gates.
    fn oracle_softmax_norm_w_moe_from_logits(
        g: &Gguf,
        arch: &str,
        xn: &[f32],
        layer: usize,
        logits: &[f32],
        gelu: bool,
    ) -> Vec<f32> {
        let n_expert = arch_u32(g, arch, "expert_count").unwrap() as usize;
        let n_used = arch_u32(g, arch, "expert_used_count").unwrap() as usize;
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
        let gate_exps = g.tensor(&tname(layer, "ffn_gate_exps.weight"));
        let up_exps = g.tensor(&tname(layer, "ffn_up_exps.weight"));
        let gate_up = g.tensor(&tname(layer, "ffn_gate_up_exps.weight"));
        let down_exps = g.tensor(&tname(layer, "ffn_down_exps.weight")).unwrap();
        let mut routed = vec![0.0f32; xn.len()];
        for (e, w) in idx.iter().zip(weights.iter()) {
            let (gate, up) = if let Some(fused) = gate_up {
                let gu = oracle_gemv_expert(fused, *e, xn);
                assert_eq!(gu.len() % 2, 0, "fused gate_up rows");
                let mid = gu.len() / 2;
                (gu[..mid].to_vec(), gu[mid..].to_vec())
            } else {
                (
                    oracle_gemv_expert(gate_exps.unwrap(), *e, xn),
                    oracle_gemv_expert(up_exps.unwrap(), *e, xn),
                )
            };
            let h: Vec<f32> = gate
                .iter()
                .zip(up.iter())
                .map(|(gv, u)| {
                    let act = if gelu {
                        oracle_gelu(*gv)
                    } else {
                        gv / (1.0 + (-gv).exp())
                    };
                    act * u
                })
                .collect();
            let y = oracle_gemv_expert(down_exps, *e, &h);
            for (o, v) in routed.iter_mut().zip(y.iter()) {
                *o += *v * *w;
            }
        }
        routed
    }

    /// Official gemma4.cpp MoE on one token: shared GeGLU plus GELU experts.
    ///
    /// `attn_out` is the residual after `post_attention_norm`. `xn_shared` is
    /// `ffn_norm(attn_out)` (the transformer pre-FFN RMSNorm).
    fn oracle_gemma4_moe(
        g: &Gguf,
        attn_out: &[f32],
        xn_shared: &[f32],
        layer: usize,
        eps: f32,
    ) -> Vec<f32> {
        let gate = oracle_gemv(
            g.tensor(&tname(layer, "ffn_gate.weight")).unwrap(),
            xn_shared,
        );
        let up = oracle_gemv(g.tensor(&tname(layer, "ffn_up.weight")).unwrap(), xn_shared);
        let h: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(gv, u)| oracle_gelu(*gv) * *u)
            .collect();
        let mut shared = oracle_gemv(g.tensor(&tname(layer, "ffn_down.weight")).unwrap(), &h);
        let pn1 = f32s(g.tensor(&tname(layer, "ffn_post_norm_1.weight")).unwrap()).unwrap();
        shared = oracle_rmsnorm(&shared, &pn1, eps);
        let pre2 = f32s(g.tensor(&tname(layer, "ffn_pre_norm_2.weight")).unwrap()).unwrap();
        let xn_exp = oracle_rmsnorm(attn_out, &pre2, eps);
        let mut tmp = oracle_rmsnorm_unweighted(attn_out, eps);
        let n_embd_f = f32::from(u16::try_from(attn_out.len()).unwrap_or(1));
        let inv = if n_embd_f > 0.0 {
            1.0 / n_embd_f.sqrt()
        } else {
            0.0
        };
        let scale = f32s(g.tensor(&tname(layer, "ffn_gate_inp.scale")).unwrap()).unwrap();
        for (v, s) in tmp.iter_mut().zip(scale.iter()) {
            *v *= inv * *s;
        }
        let logits = oracle_gemv(
            g.tensor(&tname(layer, "ffn_gate_inp.weight")).unwrap(),
            &tmp,
        );
        let mut routed =
            oracle_softmax_norm_w_moe_from_logits(g, "gemma4", &xn_exp, layer, &logits, true);
        let pn2 = f32s(g.tensor(&tname(layer, "ffn_post_norm_2.weight")).unwrap()).unwrap();
        routed = oracle_rmsnorm(&routed, &pn2, eps);
        shared
            .iter()
            .zip(routed.iter())
            .map(|(a, b)| a + b)
            .collect()
    }

    /// Official llama.cpp `build_moe_ffn` on one token: softmax, then top-k;
    /// SwiGLU; weights after the expert with `norm_w` clamp `2^-14`.
    fn oracle_llama_moe(g: &Gguf, xn: &[f32], layer: usize) -> Vec<f32> {
        oracle_softmax_norm_w_moe(g, "llama", xn, layer)
    }

    /// Official qwen3moe.cpp on one token: same `build_moe_ffn` as llama MoE
    /// (`SOFTMAX` + `norm_w`). QK-Norm is applied on Q/K in `oracle_forward_seq`.
    fn oracle_qwen3moe(g: &Gguf, xn: &[f32], layer: usize) -> Vec<f32> {
        oracle_softmax_norm_w_moe(g, "qwen3moe", xn, layer)
    }

    /// Official qwen3next.cpp on one token: softmax then top-k with `norm_w`,
    /// plus shared expert * sigmoid(`ffn_gate_inp_shexp`).
    fn oracle_qwen3next(g: &Gguf, xn: &[f32], layer: usize) -> Vec<f32> {
        let mut routed = oracle_softmax_norm_w_moe(g, "qwen3next", xn, layer);
        let gate_s = oracle_gemv(
            g.tensor(&tname(layer, "ffn_gate_shexp.weight")).unwrap(),
            xn,
        );
        let up_s = oracle_gemv(g.tensor(&tname(layer, "ffn_up_shexp.weight")).unwrap(), xn);
        let h_s: Vec<f32> = gate_s
            .iter()
            .zip(up_s.iter())
            .map(|(gv, u)| (gv / (1.0 + (-gv).exp())) * u)
            .collect();
        let mut shexp = oracle_gemv(
            g.tensor(&tname(layer, "ffn_down_shexp.weight")).unwrap(),
            &h_s,
        );
        let gate_inp_s = oracle_gemv(
            g.tensor(&tname(layer, "ffn_gate_inp_shexp.weight"))
                .unwrap(),
            xn,
        );
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
    fn oracle_llama4_moe(g: &Gguf, xn: &[f32], layer: usize) -> Vec<f32> {
        let n_expert = arch_u32(g, "llama4", "expert_count").unwrap() as usize;
        let n_used = arch_u32(g, "llama4", "expert_used_count").unwrap() as usize;
        let logits = oracle_gemv(g.tensor(&tname(layer, "ffn_gate_inp.weight")).unwrap(), xn);
        assert_eq!(logits.len(), n_expert);
        let mut idx: Vec<usize> = (0..n_expert).collect();
        idx.sort_by(|&a, &b| match logits[b].partial_cmp(&logits[a]) {
            Some(core::cmp::Ordering::Equal) | None => a.cmp(&b),
            Some(ord) => ord,
        });
        idx.truncate(n_used);
        let gate_exps = g.tensor(&tname(layer, "ffn_gate_exps.weight")).unwrap();
        let up_exps = g.tensor(&tname(layer, "ffn_up_exps.weight")).unwrap();
        let down_exps = g.tensor(&tname(layer, "ffn_down_exps.weight")).unwrap();
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
        let gate_s = oracle_gemv(
            g.tensor(&tname(layer, "ffn_gate_shexp.weight")).unwrap(),
            xn,
        );
        let up_s = oracle_gemv(g.tensor(&tname(layer, "ffn_up_shexp.weight")).unwrap(), xn);
        let h_s: Vec<f32> = gate_s
            .iter()
            .zip(up_s.iter())
            .map(|(gv, u)| (gv / (1.0 + (-gv).exp())) * u)
            .collect();
        let shexp = oracle_gemv(
            g.tensor(&tname(layer, "ffn_down_shexp.weight")).unwrap(),
            &h_s,
        );
        routed
            .iter()
            .zip(shexp.iter())
            .map(|(a, b)| a + b)
            .collect()
    }

    /// Official qwen2moe.cpp on one token: softmax then top-k, weights after
    /// SwiGLU without `norm_w`, plus shared expert * sigmoid(`ffn_gate_inp_shexp`).
    fn oracle_qwen2moe(g: &Gguf, xn: &[f32], layer: usize) -> Vec<f32> {
        let n_expert = arch_u32(g, "qwen2moe", "expert_count").unwrap() as usize;
        let n_used = arch_u32(g, "qwen2moe", "expert_used_count").unwrap() as usize;
        let logits = oracle_gemv(g.tensor(&tname(layer, "ffn_gate_inp.weight")).unwrap(), xn);
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
        let gate_exps = g.tensor(&tname(layer, "ffn_gate_exps.weight")).unwrap();
        let up_exps = g.tensor(&tname(layer, "ffn_up_exps.weight")).unwrap();
        let down_exps = g.tensor(&tname(layer, "ffn_down_exps.weight")).unwrap();
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
        let gate_s = oracle_gemv(
            g.tensor(&tname(layer, "ffn_gate_shexp.weight")).unwrap(),
            xn,
        );
        let up_s = oracle_gemv(g.tensor(&tname(layer, "ffn_up_shexp.weight")).unwrap(), xn);
        let h_s: Vec<f32> = gate_s
            .iter()
            .zip(up_s.iter())
            .map(|(gv, u)| (gv / (1.0 + (-gv).exp())) * u)
            .collect();
        let mut shexp = oracle_gemv(
            g.tensor(&tname(layer, "ffn_down_shexp.weight")).unwrap(),
            &h_s,
        );
        let gate_inp_s = oracle_gemv(
            g.tensor(&tname(layer, "ffn_gate_inp_shexp.weight"))
                .unwrap(),
            xn,
        );
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

    /// Independent scalar of official `GGML_ROPE_TYPE_NEOX`, transcribed from
    /// `ggml/src/ggml-cpu/ops.cpp` `rotate_pairs(n_dims, n_dims/2, cache, .., scale = 2)`:
    /// `ic = i0/2`, `x0 = src[ic]`, `x1 = src[ic + n_dims/2]`. The theta walk is the
    /// same `i0 += 2` progression as NORM; only the rotated lane pair differs.
    fn oracle_rope_neox(mut v: Vec<f32>, pos: usize, n_rot: usize, base: f32) -> Vec<f32> {
        let n = n_rot.min(v.len());
        if n < 2 || !n.is_multiple_of(2) {
            return v;
        }
        let n_offset = n / 2;
        let mut theta = pos as f32;
        let theta_scale = base.powf(-2.0 / n_rot as f32);
        let mut i0 = 0usize;
        while i0 + 1 < n {
            let ic = i0 / 2;
            let (c, s) = (theta.cos(), theta.sin());
            let x0 = v[ic];
            let x1 = v[ic + n_offset];
            v[ic] = x0 * c - x1 * s;
            v[ic + n_offset] = x0 * s + x1 * c;
            theta *= theta_scale;
            i0 += 2;
        }
        v
    }

    /// Independent transcription of official `llama_model_rope_type`
    /// (`src/llama-model.cpp`) for the architectures this crate loads.
    /// `true` = `LLAMA_ROPE_TYPE_NEOX`, `false` = `LLAMA_ROPE_TYPE_NORM`.
    fn oracle_rope_is_neox(arch: &str) -> bool {
        matches!(
            arch,
            "qwen2"
                | "qwen2moe"
                | "qwen3"
                | "qwen3moe"
                | "qwen3next"
                | "phi2"
                | "phi3"
                | "gemma"
                | "gemma2"
                | "gemma3"
                | "gemma3n"
                | "gemma4"
        )
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

    fn oracle_scale_match(src: &[f32], dst: &mut [f32]) {
        let mag_s: f32 = src.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mag_d: f32 = dst.iter().map(|v| v * v).sum::<f32>().sqrt();
        let scale = if mag_d > 0.0 { mag_s / mag_d } else { 0.0 };
        for v in dst.iter_mut() {
            *v *= scale;
        }
    }

    fn oracle_gaussian_topk(gate: &mut [f32]) {
        let n_ff = gate.len();
        if n_ff < 2 {
            return;
        }
        let n_f = n_ff as f32;
        let mean = gate.iter().sum::<f32>() / n_f;
        let var = gate
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f32>()
            / (n_f - 1.0);
        let cutoff = mean + var.sqrt() * GEMMA3N_SPARSITY_STD_MUL;
        for v in gate.iter_mut() {
            *v = (*v - cutoff).max(0.0);
        }
    }

    fn oracle_gemma3n_router(g: &Gguf, x: &[f32], li: usize, eps: f32, n_embd: usize) -> Vec<f32> {
        let rn = f32s(g.tensor(&tname(li, "altup_router_norm.weight")).unwrap()).unwrap();
        let mut inp = oracle_rmsnorm(x, &rn, eps);
        let inv = 1.0 / n_embd as f32;
        for v in &mut inp {
            *v *= inv;
        }
        let mut y = oracle_gemv(g.tensor(&tname(li, "altup_router.weight")).unwrap(), &inp);
        for v in &mut y {
            *v = v.tanh();
        }
        y
    }

    fn oracle_gemma3n_predict(
        g: &Gguf,
        streams: &[Vec<f32>],
        li: usize,
        eps: f32,
        n_embd: usize,
        i_act: usize,
    ) -> Vec<Vec<f32>> {
        let n_altup = streams.len();
        let modalities = oracle_gemma3n_router(g, &streams[i_act], li, eps, n_embd);
        let coefs = oracle_gemv(
            g.tensor(&tname(li, "altup_predict_coef.weight")).unwrap(),
            &modalities,
        );
        let mut pred = streams.to_vec();
        for i in 0..n_altup {
            for j in 0..n_altup {
                let c = coefs[j + i * n_altup];
                for (d, s) in pred[i].iter_mut().zip(streams[j].iter()) {
                    *d += c * *s;
                }
            }
        }
        pred
    }

    fn oracle_gemma3n_correct(
        g: &Gguf,
        pred: &mut [Vec<f32>],
        activated: &[f32],
        li: usize,
        eps: f32,
        n_embd: usize,
        i_act: usize,
    ) {
        let n_altup = pred.len();
        let modalities = oracle_gemma3n_router(g, activated, li, eps, n_embd);
        let mut coefs = oracle_gemv(
            g.tensor(&tname(li, "altup_correct_coef.weight")).unwrap(),
            &modalities,
        );
        for c in &mut coefs {
            *c += 1.0;
        }
        let mut innov = vec![0.0f32; n_embd];
        for ((d, a), p) in innov
            .iter_mut()
            .zip(activated.iter())
            .zip(pred[i_act].iter())
        {
            *d = *a - *p;
        }
        for i in 0..n_altup {
            let c = coefs[i];
            for (d, iv) in pred[i].iter_mut().zip(innov.iter()) {
                *d += c * *iv;
            }
        }
    }

    /// Independent scalar of official `src/models/gemma3n.cpp` (AltUp, Laurel,
    /// per-layer inputs, V RMSNorm, attention scale 1.0, gaussian_topk, SWA
    /// period 5, final tanh softcap).
    fn oracle_gemma3n_forward_seq(g: &Gguf, tokens: &[u32]) -> Vec<f32> {
        let arch = "gemma3n";
        let n_embd = arch_u32(g, arch, "embedding_length").unwrap() as usize;
        let n_head = arch_u32(g, arch, "attention.head_count").unwrap() as usize;
        let n_kv = arch_u32(g, arch, "attention.head_count_kv").unwrap() as usize;
        let n_layer = arch_u32(g, arch, "block_count").unwrap() as usize;
        let n_rot = oracle_n_rot(g, arch, n_embd, n_head);
        let eps = arch_f32(g, arch, "attention.layer_norm_rms_epsilon").unwrap();
        let base = arch_f32(g, arch, "rope.freq_base").unwrap_or(10_000.0);
        let hd = n_embd / n_head;
        let gqa = n_head / n_kv;
        let n_altup = arch_u32(g, arch, "altup.num_inputs")
            .unwrap_or(u32::try_from(GEMMA3N_N_ALTUP).unwrap_or(0)) as usize;
        let i_act = arch_u32(g, arch, "altup.active_idx")
            .unwrap_or(u32::try_from(GEMMA3N_I_ALTUP_ACT).unwrap_or(0))
            as usize;
        let n_ea = arch_u32(g, arch, "embedding_length_per_layer_input")
            .unwrap_or(u32::try_from(GEMMA3N_N_EMBD_ALTUP).unwrap_or(0))
            as usize;
        let n_swa = arch_u32(g, arch, "attention.sliding_window").unwrap() as usize;
        let period = arch_u32(g, arch, "attention.sliding_window_pattern")
            .unwrap_or(GEMMA3N_SWA_PERIOD_DEFAULT);
        let cap =
            arch_f32(g, arch, "final_logit_softcapping").unwrap_or(GEMMA2_FINAL_LOGIT_SOFTCAPPING);
        let embed_scale = (n_embd as f32).sqrt();
        let tok_scale = (n_ea as f32).sqrt();
        let mix_scale = 1.0 / 2.0f32.sqrt();
        let proj_scale = 1.0 / (n_embd as f32).sqrt();
        let extra = n_altup - 1;
        let mut k_cache: Vec<Vec<Vec<Vec<f32>>>> = vec![vec![Vec::new(); n_kv]; n_layer];
        let mut v_cache: Vec<Vec<Vec<Vec<f32>>>> = vec![vec![Vec::new(); n_kv]; n_layer];
        let mut last = Vec::new();
        for (pos, &token) in tokens.iter().enumerate() {
            let mut residual = oracle_embed(g.tensor("token_embd.weight").unwrap(), token);
            for v in &mut residual {
                *v *= embed_scale;
            }
            let mut streams: Vec<Vec<f32>> = vec![residual.clone()];
            for p in 0..extra {
                let mut extra_s =
                    oracle_gemv_expert(g.tensor("altup_proj.weight").unwrap(), p, &residual);
                oracle_scale_match(&residual, &mut extra_s);
                streams.push(extra_s);
            }
            let mut per_tok = oracle_embed(g.tensor("per_layer_token_embd.weight").unwrap(), token);
            for v in &mut per_tok {
                *v *= tok_scale;
            }
            let mut proj = oracle_gemv(g.tensor("per_layer_model_proj.weight").unwrap(), &residual);
            for v in &mut proj {
                *v *= proj_scale;
            }
            let pn = f32s(g.tensor("per_layer_proj_norm.weight").unwrap()).unwrap();
            let mut per_layer: Vec<Vec<f32>> = Vec::new();
            for li in 0..n_layer {
                let row = &proj[li * n_ea..(li + 1) * n_ea];
                let mut nrm = oracle_rmsnorm(row, &pn, eps);
                let tok_row = &per_tok[li * n_ea..(li + 1) * n_ea];
                for (nv, tv) in nrm.iter_mut().zip(tok_row.iter()) {
                    *nv = (*nv + *tv) * mix_scale;
                }
                per_layer.push(nrm);
            }
            for li in 0..n_layer {
                let mut pred = oracle_gemma3n_predict(g, &streams, li, eps, n_embd, i_act);
                let active = pred[i_act].clone();
                let an = f32s(g.tensor(&tname(li, "attn_norm.weight")).unwrap()).unwrap();
                let xn = oracle_rmsnorm(&active, &an, eps);
                let l = oracle_gemv(g.tensor(&tname(li, "laurel_l.weight")).unwrap(), &xn);
                let mut laurel = oracle_gemv(g.tensor(&tname(li, "laurel_r.weight")).unwrap(), &l);
                let ln = f32s(g.tensor(&tname(li, "laurel_post_norm.weight")).unwrap()).unwrap();
                laurel = oracle_rmsnorm(&laurel, &ln, eps);
                for (a, b) in laurel.iter_mut().zip(xn.iter()) {
                    *a += *b;
                }
                let q = oracle_gemv(g.tensor(&tname(li, "attn_q.weight")).unwrap(), &xn);
                let k = oracle_gemv(g.tensor(&tname(li, "attn_k.weight")).unwrap(), &xn);
                let v = oracle_gemv(g.tensor(&tname(li, "attn_v.weight")).unwrap(), &xn);
                let qn = f32s(g.tensor(&tname(li, "attn_q_norm.weight")).unwrap()).unwrap();
                let kn = f32s(g.tensor(&tname(li, "attn_k_norm.weight")).unwrap()).unwrap();
                let mut qh: Vec<Vec<f32>> = q.chunks(hd).map(<[f32]>::to_vec).collect();
                let mut kh: Vec<Vec<f32>> = k.chunks(hd).map(<[f32]>::to_vec).collect();
                let mut vh: Vec<Vec<f32>> = v.chunks(hd).map(<[f32]>::to_vec).collect();
                for h in &mut qh {
                    *h = oracle_rmsnorm(h, &qn, eps);
                }
                for h in &mut kh {
                    *h = oracle_rmsnorm(h, &kn, eps);
                }
                for h in &mut vh {
                    *h = oracle_rmsnorm_unweighted(h, eps);
                }
                for h in &mut kh {
                    *h = oracle_rope_neox(h.clone(), pos, n_rot, base);
                }
                for h in &mut qh {
                    *h = oracle_rope_neox(h.clone(), pos, n_rot, base);
                }
                for (hkv, khv) in kh.iter().enumerate() {
                    k_cache[li][hkv].push(khv.clone());
                    v_cache[li][hkv].push(vh[hkv].clone());
                }
                let seq = pos + 1;
                let mut attn = vec![0.0f32; n_embd];
                for (hq, qvec) in qh.iter().enumerate() {
                    let hkv = hq / gqa;
                    let mut scores = vec![0.0f32; seq];
                    for t in 0..seq {
                        let kv = &k_cache[li][hkv][t];
                        let mut s = qvec.iter().zip(kv.iter()).map(|(a, b)| a * b).sum::<f32>();
                        if gemma2_is_swa(li, period) && n_swa > 0 && pos.saturating_sub(t) >= n_swa
                        {
                            s = f32::NEG_INFINITY;
                        }
                        scores[t] = s;
                    }
                    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut z = 0.0f32;
                    for v in &mut scores {
                        *v = (*v - m).exp();
                        z += *v;
                    }
                    if z > 0.0 {
                        for v in &mut scores {
                            *v /= z;
                        }
                    }
                    let mut acc = vec![0.0f32; hd];
                    for t in 0..seq {
                        let st = scores[t];
                        for (a, b) in acc.iter_mut().zip(v_cache[li][hkv][t].iter()) {
                            *a += st * *b;
                        }
                    }
                    let off = hq * hd;
                    attn[off..off + hd].copy_from_slice(&acc);
                }
                let mut attn_out =
                    oracle_gemv(g.tensor(&tname(li, "attn_output.weight")).unwrap(), &attn);
                let pn_attn =
                    f32s(g.tensor(&tname(li, "post_attention_norm.weight")).unwrap()).unwrap();
                attn_out = oracle_rmsnorm(&attn_out, &pn_attn, eps);
                let attn_gated: Vec<f32> = attn_out
                    .iter()
                    .zip(active.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                let attn_laurel: Vec<f32> = attn_gated
                    .iter()
                    .zip(laurel.iter())
                    .map(|(a, b)| (a + b) * mix_scale)
                    .collect();
                let fnorm = f32s(g.tensor(&tname(li, "ffn_norm.weight")).unwrap()).unwrap();
                let xn_ff = oracle_rmsnorm(&attn_laurel, &fnorm, eps);
                let mut gate =
                    oracle_gemv(g.tensor(&tname(li, "ffn_gate.weight")).unwrap(), &xn_ff);
                let up = oracle_gemv(g.tensor(&tname(li, "ffn_up.weight")).unwrap(), &xn_ff);
                if li < GEMMA3N_N_LAYER_SPARSITY {
                    oracle_gaussian_topk(&mut gate);
                }
                let h: Vec<f32> = gate
                    .iter()
                    .zip(up.iter())
                    .map(|(gv, u)| oracle_gelu(*gv) * *u)
                    .collect();
                let mut down = oracle_gemv(g.tensor(&tname(li, "ffn_down.weight")).unwrap(), &h);
                let pn_ff = f32s(g.tensor(&tname(li, "post_ffw_norm.weight")).unwrap()).unwrap();
                down = oracle_rmsnorm(&down, &pn_ff, eps);
                let activated: Vec<f32> = down
                    .iter()
                    .zip(attn_laurel.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                oracle_gemma3n_correct(g, &mut pred, &activated, li, eps, n_embd, i_act);
                let scale_w =
                    f32s(g.tensor(&tname(li, "altup_correct_scale.weight")).unwrap()).unwrap();
                let mut first: Vec<f32> = pred[i_act]
                    .iter()
                    .zip(scale_w.iter())
                    .map(|(a, s)| a * s)
                    .collect();
                first = oracle_gemv(g.tensor(&tname(li, "inp_gate.weight")).unwrap(), &first);
                for v in &mut first {
                    *v = oracle_gelu(*v);
                }
                for (f, p) in first.iter_mut().zip(per_layer[li].iter()) {
                    *f *= *p;
                }
                first = oracle_gemv(g.tensor(&tname(li, "proj.weight")).unwrap(), &first);
                let post = f32s(g.tensor(&tname(li, "post_norm.weight")).unwrap()).unwrap();
                first = oracle_rmsnorm(&first, &post, eps);
                for stream in pred.iter_mut().skip(1) {
                    for (d, s) in stream.iter_mut().zip(first.iter()) {
                        *d += *s;
                    }
                }
                streams = pred;
            }
            let mut merged = streams[i_act].clone();
            for p in 0..extra {
                let mut extra_s = oracle_gemv_expert(
                    g.tensor("altup_unembd_proj.weight").unwrap(),
                    p,
                    &streams[p + 1],
                );
                oracle_scale_match(&streams[i_act], &mut extra_s);
                for (d, s) in merged.iter_mut().zip(extra_s.iter()) {
                    *d += *s;
                }
            }
            let inv = 1.0 / n_altup as f32;
            for v in &mut merged {
                *v *= inv;
            }
            let on = f32s(g.tensor("output_norm.weight").unwrap()).unwrap();
            let x = oracle_rmsnorm(&merged, &on, eps);
            let lm_head = g
                .tensor("output.weight")
                .or_else(|| g.tensor("token_embd.weight"))
                .expect("lm_head");
            last = oracle_gemv(lm_head, &x);
            if cap > 0.0 {
                for v in &mut last {
                    *v = cap * (*v / cap).tanh();
                }
            }
        }
        last
    }

    fn oracle_gemma4_layer_is_swa(g: &Gguf, arch: &str, li: usize) -> bool {
        match g.kv(&arch_key(arch, "attention.sliding_window_pattern")) {
            Some(Kv::Array { items, .. }) => match items.get(li) {
                Some(Kv::Bool(b)) => *b,
                Some(Kv::U32(v)) => *v != 0,
                Some(Kv::I32(v)) => *v != 0,
                _ => false,
            },
            Some(Kv::U32(v)) => *v != 0,
            Some(Kv::I32(v)) => *v != 0,
            Some(Kv::Bool(b)) => *b,
            _ => false,
        }
    }

    fn oracle_gemma4_per_layer(
        g: &Gguf,
        residual: &[f32],
        token: u32,
        n_layer: usize,
        n_pl: usize,
        n_embd: usize,
        eps: f32,
    ) -> Vec<Vec<f32>> {
        let tok_scale = (n_pl as f32).sqrt();
        let proj_scale = 1.0 / (n_embd as f32).sqrt();
        let mix_scale = 1.0 / 2.0f32.sqrt();
        let mut per_tok = oracle_embed(g.tensor("per_layer_token_embd.weight").unwrap(), token);
        for v in &mut per_tok {
            *v *= tok_scale;
        }
        let mut proj = oracle_gemv(g.tensor("per_layer_model_proj.weight").unwrap(), residual);
        for v in &mut proj {
            *v *= proj_scale;
        }
        let pn = f32s(g.tensor("per_layer_proj_norm.weight").unwrap()).unwrap();
        let mut out = Vec::new();
        for li in 0..n_layer {
            let row = &proj[li * n_pl..(li + 1) * n_pl];
            let mut nrm = oracle_rmsnorm(row, &pn, eps);
            let tok_row = &per_tok[li * n_pl..(li + 1) * n_pl];
            for (nv, tv) in nrm.iter_mut().zip(tok_row.iter()) {
                *nv = (*nv + *tv) * mix_scale;
            }
            out.push(nrm);
        }
        out
    }

    fn oracle_gemma4_ple_inject(g: &Gguf, x: &[f32], pl: &[f32], li: usize, eps: f32) -> Vec<f32> {
        let mut gate = oracle_gemv(g.tensor(&tname(li, "inp_gate.weight")).unwrap(), x);
        for v in &mut gate {
            *v = oracle_gelu(*v);
        }
        for (f, p) in gate.iter_mut().zip(pl.iter()) {
            *f *= *p;
        }
        let mut inj = oracle_gemv(g.tensor(&tname(li, "proj.weight")).unwrap(), &gate);
        let post = f32s(g.tensor(&tname(li, "post_norm.weight")).unwrap()).unwrap();
        inj = oracle_rmsnorm(&inj, &post, eps);
        x.iter().zip(inj.iter()).map(|(a, b)| a + b).collect()
    }

    /// Independent scalar Llama math for a token sequence (causal attn + GQA).
    fn oracle_forward_seq(g: &Gguf, tokens: &[u32]) -> Vec<f32> {
        let arch = architecture(g).expect("arch");
        if arch == "gemma3n" {
            return oracle_gemma3n_forward_seq(g, tokens);
        }
        let n_embd = arch_u32(g, arch, "embedding_length").unwrap() as usize;
        let n_head = arch_u32(g, arch, "attention.head_count").unwrap() as usize;
        let n_kv = arch_u32(g, arch, "attention.head_count_kv").unwrap() as usize;
        let n_layer = arch_u32(g, arch, "block_count").unwrap() as usize;
        let n_rot = oracle_n_rot(g, arch, n_embd, n_head);
        let phi2 = arch == "phi2";
        let bloom = arch == "bloom";
        let eps = if phi2 || bloom {
            arch_f32(g, arch, "attention.layer_norm_epsilon").unwrap()
        } else {
            arch_f32(g, arch, "attention.layer_norm_rms_epsilon").unwrap()
        };
        let base = arch_f32(g, arch, "rope.freq_base").unwrap_or(10_000.0);
        let hd = n_embd / n_head;
        let gqa = n_head / n_kv;
        let emb = g.tensor("token_embd.weight").unwrap();
        let gemma2 = arch == "gemma2";
        let gemma3 = arch == "gemma3";
        let gemma4 = arch == "gemma4";
        let gemma = arch == "gemma" || gemma2 || gemma3 || gemma4;
        let gemma_post = gemma2 || gemma3 || gemma4;
        let gemma4_n_pl = if gemma4 {
            arch_u32(g, arch, "embedding_length_per_layer_input").unwrap_or(0) as usize
        } else {
            0
        };
        let n_from_start = if gemma4 {
            let n_shared = arch_u32(g, arch, "attention.shared_kv_layers").unwrap_or(0);
            (n_layer as i32) - (n_shared as i32)
        } else {
            n_layer as i32
        };
        let embed_scale = if gemma {
            f32::from(u16::try_from(n_embd).unwrap_or(1)).sqrt()
        } else {
            1.0
        };
        let mut k_cache: Vec<Vec<Vec<Vec<f32>>>> = vec![vec![Vec::new(); n_kv]; n_layer];
        let mut v_cache: Vec<Vec<Vec<Vec<f32>>>> = vec![vec![Vec::new(); n_kv]; n_layer];
        let mut last = Vec::new();
        for (pos, &token) in tokens.iter().enumerate() {
            let mut residual = oracle_embed(emb, token);
            for v in &mut residual {
                *v *= embed_scale;
            }
            if bloom {
                let tw = f32s(g.tensor("token_embd_norm.weight").unwrap()).unwrap();
                let tb = f32s(g.tensor("token_embd_norm.bias").unwrap()).unwrap();
                residual = oracle_layernorm(&residual, &tw, Some(&tb), eps);
            }
            let per_layer = if gemma4_n_pl > 0 {
                Some(oracle_gemma4_per_layer(
                    g,
                    &residual,
                    token,
                    n_layer,
                    gemma4_n_pl,
                    n_embd,
                    eps,
                ))
            } else {
                None
            };
            let mut x = residual.clone();
            for li in 0..n_layer {
                let an = f32s(g.tensor(&tname(li, "attn_norm.weight")).unwrap()).unwrap();
                let an_b = g
                    .tensor(&tname(li, "attn_norm.bias"))
                    .and_then(|t| f32s(t).ok());
                let xn_attn = if phi2 || bloom {
                    oracle_layernorm(&residual, &an, an_b.as_deref(), eps)
                } else {
                    oracle_rmsnorm(&residual, &an, eps)
                };
                let attn_norm_output = xn_attn.clone();
                let (q_full, k, v) = if bloom {
                    let qkv = oracle_add_bias(
                        oracle_gemv(g.tensor(&tname(li, "attn_qkv.weight")).unwrap(), &xn_attn),
                        g.tensor(&tname(li, "attn_qkv.bias")),
                    );
                    let n_q = n_head * hd;
                    let n_k = n_kv * hd;
                    let q = qkv[..n_q].to_vec();
                    let k = qkv[n_q..n_q + n_k].to_vec();
                    let v = qkv[n_q + n_k..].to_vec();
                    (q, k, v)
                } else {
                    let q = oracle_add_bias(
                        oracle_gemv(g.tensor(&tname(li, "attn_q.weight")).unwrap(), &xn_attn),
                        g.tensor(&tname(li, "attn_q.bias")),
                    );
                    let (k, v) = if gemma4 && !gemma4_has_kv(n_from_start, li) {
                        (Vec::new(), Vec::new())
                    } else {
                        (
                            oracle_add_bias(
                                oracle_gemv(
                                    g.tensor(&tname(li, "attn_k.weight")).unwrap(),
                                    &xn_attn,
                                ),
                                g.tensor(&tname(li, "attn_k.bias")),
                            ),
                            oracle_add_bias(
                                oracle_gemv(
                                    g.tensor(&tname(li, "attn_v.weight")).unwrap(),
                                    &xn_attn,
                                ),
                                g.tensor(&tname(li, "attn_v.bias")),
                            ),
                        )
                    };
                    (q, k, v)
                };
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
                let mut vh: Vec<Vec<f32>> = v.chunks(hd).map(<[f32]>::to_vec).collect();
                // Official Qwen3 / Qwen3MoE / Qwen3VL / Qwen3Next / Qwen35 QK-Norm (`LLM_NORM_RMS` on each head) before RoPE.
                if let Some(qn) = g.tensor(&tname(li, "attn_q_norm.weight")) {
                    let w = f32s(qn).unwrap();
                    for h in &mut qh {
                        *h = oracle_rmsnorm(h, &w, eps);
                    }
                }
                if let Some(kn) = g.tensor(&tname(li, "attn_k_norm.weight")) {
                    let w = f32s(kn).unwrap();
                    for h in &mut kh {
                        *h = oracle_rmsnorm(h, &w, eps);
                    }
                }
                if gemma4 && gemma4_has_kv(n_from_start, li) {
                    for h in &mut vh {
                        *h = oracle_rmsnorm_unweighted(h, eps);
                    }
                }
                // Official llama4.cpp: iRoPE when `(il+1) % n_no_rope_layer_step != 0`.
                let llama4 = arch == "llama4";
                let n_no_rope = if llama4 { LLAMA4_NO_ROPE_LAYER_STEP } else { 0 };
                let use_rope = !bloom && (!llama4 || (n_no_rope > 0 && (li + 1) % n_no_rope != 0));
                let n_expert = arch_u32(g, arch, "expert_count").unwrap_or(0) as usize;
                let qk_l2 = llama4 && use_rope && n_expert != 128;
                if use_rope {
                    let sections = if arch == "qwen2vl" || arch == "qwen3vl" || arch == "qwen35" {
                        oracle_rope_sections(g, arch)
                    } else {
                        None
                    };
                    let is_imrope = arch == "qwen3vl" || arch == "qwen35";
                    let neox = oracle_rope_is_neox(arch);
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
                            None if neox => oracle_rope_neox(h.clone(), pos, n_rot, base),
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
                            None if neox => oracle_rope_neox(h.clone(), pos, n_rot, base),
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
                } else if !bloom {
                    let scale = llama4_attn_temp_scale(pos);
                    for h in &mut qh {
                        for v in h.iter_mut() {
                            *v *= scale;
                        }
                    }
                }
                if gemma4_has_kv(n_from_start, li) {
                    for (hkv, khv) in kh.iter().enumerate() {
                        k_cache[li][hkv].push(khv.clone());
                        v_cache[li][hkv].push(vh[hkv].clone());
                    }
                }
                if phi2 && use_rope {
                    let q_scale = 1.0 / (hd as f32).sqrt();
                    for h in &mut qh {
                        for v in h.iter_mut() {
                            *v *= q_scale;
                        }
                    }
                }
                let kv_slot = gemma4_kv_slot(
                    n_from_start,
                    li,
                    gemma4 && oracle_gemma4_layer_is_swa(g, arch, li),
                )
                .expect("kv slot");
                let seq = pos + 1;
                let inv = if phi2 || gemma4 {
                    1.0
                } else {
                    1.0 / (hd as f32).sqrt()
                };
                let mut attn = vec![0.0f32; n_embd];
                for (hq, qvec) in qh.iter().enumerate() {
                    let hkv = hq / gqa;
                    let mut scores = vec![0.0f32; seq];
                    for t in 0..seq {
                        let kv = &k_cache[kv_slot][hkv][t];
                        let mut s =
                            qvec.iter().zip(kv.iter()).map(|(a, b)| a * b).sum::<f32>() * inv;
                        if bloom {
                            s += alibi_slope(hq, n_head, BLOOM_MAX_ALIBI_BIAS)
                                * (t as f32 - pos as f32);
                        }
                        if gemma2 {
                            let cap = arch_f32(g, arch, "attn_logit_softcapping").unwrap_or(0.0);
                            if cap > 0.0 {
                                s = cap * (s / cap).tanh();
                            }
                        }
                        if gemma2 || gemma3 || gemma4 {
                            let n_swa =
                                arch_u32(g, arch, "attention.sliding_window").unwrap_or(0) as usize;
                            let is_swa = if gemma4 {
                                oracle_gemma4_layer_is_swa(g, arch, li)
                            } else {
                                let period = arch_u32(g, arch, "attention.sliding_window_pattern")
                                    .unwrap_or(if gemma3 {
                                        GEMMA3_SWA_PERIOD_DEFAULT
                                    } else {
                                        GEMMA2_SWA_PERIOD_DEFAULT
                                    });
                                gemma2_is_swa(li, period)
                            };
                            if is_swa && n_swa > 0 && pos.saturating_sub(t) >= n_swa {
                                s = f32::NEG_INFINITY;
                            }
                        }
                        scores[t] = s;
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
                        for (a, b) in acc.iter_mut().zip(v_cache[kv_slot][hkv][t].iter()) {
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
                x = oracle_add_bias(
                    oracle_gemv(g.tensor(&tname(li, "attn_output.weight")).unwrap(), &attn),
                    g.tensor(&tname(li, "attn_output.bias")),
                );
                if gemma_post {
                    let pn =
                        f32s(g.tensor(&tname(li, "post_attention_norm.weight")).unwrap()).unwrap();
                    x = oracle_rmsnorm(&x, &pn, eps);
                }
                if phi2 {
                    let up = oracle_add_bias(
                        oracle_gemv(
                            g.tensor(&tname(li, "ffn_up.weight")).unwrap(),
                            &attn_norm_output,
                        ),
                        g.tensor(&tname(li, "ffn_up.bias")),
                    );
                    let h: Vec<f32> = up.iter().map(|u| oracle_gelu(*u)).collect();
                    let down = oracle_add_bias(
                        oracle_gemv(g.tensor(&tname(li, "ffn_down.weight")).unwrap(), &h),
                        g.tensor(&tname(li, "ffn_down.bias")),
                    );
                    x = x
                        .iter()
                        .zip(down.iter())
                        .zip(residual.iter())
                        .map(|((a, d), r)| a + d + r)
                        .collect();
                } else {
                    x = x.iter().zip(residual.iter()).map(|(a, b)| a + b).collect();
                    if bloom {
                        let fnorm = f32s(g.tensor(&tname(li, "ffn_norm.weight")).unwrap()).unwrap();
                        let fnorm_b = g
                            .tensor(&tname(li, "ffn_norm.bias"))
                            .and_then(|t| f32s(t).ok());
                        let xn = oracle_layernorm(&x, &fnorm, fnorm_b.as_deref(), eps);
                        let up = oracle_add_bias(
                            oracle_gemv(g.tensor(&tname(li, "ffn_up.weight")).unwrap(), &xn),
                            g.tensor(&tname(li, "ffn_up.bias")),
                        );
                        let h: Vec<f32> = up.iter().map(|u| oracle_gelu(*u)).collect();
                        let down = oracle_add_bias(
                            oracle_gemv(g.tensor(&tname(li, "ffn_down.weight")).unwrap(), &h),
                            g.tensor(&tname(li, "ffn_down.bias")),
                        );
                        x = down.iter().zip(x.iter()).map(|(a, b)| a + b).collect();
                    } else {
                        let fnorm = if qwen3next || qwen35 {
                            f32s(g.tensor(&tname(li, "post_attention_norm.weight")).unwrap())
                                .unwrap()
                        } else {
                            f32s(g.tensor(&tname(li, "ffn_norm.weight")).unwrap()).unwrap()
                        };
                        let xn = oracle_rmsnorm(&x, &fnorm, eps);
                        let llama_moe =
                            arch == "llama" && arch_u32(g, arch, "expert_count").unwrap_or(0) > 0;
                        let qwen2moe = arch == "qwen2moe";
                        let qwen3moe = arch == "qwen3moe";
                        let gemma4_moe =
                            gemma4 && g.tensor(&tname(li, "ffn_gate_inp.weight")).is_some();
                        let mut down = if llama4 {
                            oracle_llama4_moe(g, &xn, li)
                        } else if llama_moe {
                            oracle_llama_moe(g, &xn, li)
                        } else if qwen2moe {
                            oracle_qwen2moe(g, &xn, li)
                        } else if qwen3moe {
                            oracle_qwen3moe(g, &xn, li)
                        } else if qwen3next {
                            oracle_qwen3next(g, &xn, li)
                        } else if gemma4_moe {
                            oracle_gemma4_moe(g, &x, &xn, li, eps)
                        } else {
                            let gate =
                                oracle_gemv(g.tensor(&tname(li, "ffn_gate.weight")).unwrap(), &xn);
                            let up =
                                oracle_gemv(g.tensor(&tname(li, "ffn_up.weight")).unwrap(), &xn);
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
                            oracle_gemv(g.tensor(&tname(li, "ffn_down.weight")).unwrap(), &h)
                        };
                        if gemma_post {
                            let pn = f32s(g.tensor(&tname(li, "post_ffw_norm.weight")).unwrap())
                                .unwrap();
                            down = oracle_rmsnorm(&down, &pn, eps);
                        }
                        x = down.iter().zip(x.iter()).map(|(a, b)| a + b).collect();
                        if let Some(pl) = per_layer.as_ref() {
                            x = oracle_gemma4_ple_inject(g, &x, &pl[li], li, eps);
                        }
                    }
                }
                residual = x.clone();
            }
            let on = f32s(g.tensor("output_norm.weight").unwrap()).unwrap();
            let on_b = g.tensor("output_norm.bias").and_then(|t| f32s(t).ok());
            x = if phi2 || bloom {
                oracle_layernorm(&x, &on, on_b.as_deref(), eps)
            } else {
                oracle_rmsnorm(&x, &on, eps)
            };
            let lm_head = g
                .tensor("output.weight")
                .or_else(|| g.tensor("token_embd.weight"))
                .expect("lm_head");
            last = oracle_add_bias(oracle_gemv(lm_head, &x), g.tensor("output.bias"));
            if gemma2 || gemma3 || gemma4 {
                let cap = arch_f32(g, arch, "final_logit_softcapping").unwrap_or(0.0);
                if cap > 0.0 {
                    for v in &mut last {
                        *v = cap * (*v / cap).tanh();
                    }
                }
            }
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

    /// Official `ggml/src/ggml-cpu/ops.cpp` `rotate_pairs`:
    /// `GGML_ROPE_TYPE_NORMAL` passes `n_offset = 1, scale = 1` so `ic = i0` and the
    /// rotated pair is `(i0, i0 + 1)`; `GGML_ROPE_TYPE_NEOX` passes
    /// `n_offset = n_dims/2, scale = 2` so `ic = i0/2` and the pair is
    /// `(i0/2, i0/2 + n_dims/2)`. Both walk theta identically (`i0 += 2`).
    ///
    /// Pinned against values computed by hand from that definition, so this test does
    /// not depend on the model oracle and would fail if either convention regressed
    /// or the two were swapped.
    #[test]
    fn rope_norm_and_neox_pair_different_lanes() {
        let n_rot = 4usize;
        let base = 10_000.0f32;
        let pos = 1usize;
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let theta0 = 1.0f32;
        let theta1 = theta0 * base.powf(-2.0 / 4.0);
        let (c0, s0) = (theta0.cos(), theta0.sin());
        let (c1, s1) = (theta1.cos(), theta1.sin());

        // NORM rotates (0,1) at theta0 and (2,3) at theta1.
        let want_norm = [
            src[0] * c0 - src[1] * s0,
            src[0] * s0 + src[1] * c0,
            src[2] * c1 - src[3] * s1,
            src[2] * s1 + src[3] * c1,
        ];
        let mut got_norm = src;
        rope(&mut got_norm, pos, n_rot, base).expect("norm rope");
        for (got, want) in got_norm.iter().zip(want_norm.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "NORM {got_norm:?} vs {want_norm:?}"
            );
        }

        // NEOX rotates (0,2) at theta0 and (1,3) at theta1.
        let want_neox = [
            src[0] * c0 - src[2] * s0,
            src[1] * c1 - src[3] * s1,
            src[0] * s0 + src[2] * c0,
            src[1] * s1 + src[3] * c1,
        ];
        let mut got_neox = src;
        rope_neox(&mut got_neox, pos, n_rot, base).expect("neox rope");
        for (got, want) in got_neox.iter().zip(want_neox.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "NEOX {got_neox:?} vs {want_neox:?}"
            );
        }

        assert_ne!(
            got_norm, got_neox,
            "NORM and NEOX must not coincide on asymmetric input"
        );
    }

    /// Per-architecture rope type, transcribed from official `llama_model_rope_type`.
    /// Production and oracle must agree, and the Llama family must stay NORM while
    /// the Qwen dense/MoE line, Phi-3, and Gemma are NEOX.
    #[test]
    fn rope_type_matches_official_arch_table() {
        for arch in ["llama", "llama4", "mistral"] {
            assert!(!rope_is_neox(arch), "{arch} is LLAMA_ROPE_TYPE_NORM");
            assert!(!oracle_rope_is_neox(arch), "{arch} oracle NORM");
        }
        for arch in [
            "qwen2",
            "qwen2moe",
            "qwen3",
            "qwen3moe",
            "qwen3next",
            "phi2",
            "phi3",
            "gemma",
            "gemma2",
            "gemma3",
            "gemma3n",
            "gemma4",
        ] {
            assert!(rope_is_neox(arch), "{arch} is LLAMA_ROPE_TYPE_NEOX");
            assert!(oracle_rope_is_neox(arch), "{arch} oracle NEOX");
        }
        // MROPE / IMROPE arches carry `rope.dimension_sections` and take the
        // `rope_multi` path, which already rotates on the NEOX offset.
        for arch in ["qwen2vl", "qwen3vl", "qwen35"] {
            assert!(!rope_is_neox(arch), "{arch} routes through rope_multi");
        }
    }

    /// Text tokens feed `[t, h, w, e] = [p, p, p, 0]`, which collapses the m-RoPE
    /// sector walk onto plain NEOX lane math. Distinct per-axis positions (what an
    /// image token supplies) must diverge, otherwise m-RoPE is untested.
    #[test]
    fn rope_multi_reduces_to_neox_on_text_and_differs_on_distinct_axes() {
        let n_rot = 8usize;
        let base = 10_000.0f32;
        let sections = [2i32, 1, 1, 0];
        let src = [0.5f32, -1.5, 2.0, 3.5, -0.25, 1.25, -2.5, 0.75];

        let mut neox = src;
        rope_neox(&mut neox, 3, n_rot, base).expect("neox");

        let mut text = src;
        rope_multi(&mut text, [3, 3, 3, 0], n_rot, base, sections, false).expect("mrope text");
        for (got, want) in text.iter().zip(neox.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "text m-RoPE {text:?} must equal NEOX {neox:?}"
            );
        }

        let mut axes = src;
        rope_multi(&mut axes, [3, 7, 11, 0], n_rot, base, sections, false).expect("mrope axes");
        assert_ne!(
            axes, neox,
            "distinct t/h/w positions must exercise the m-RoPE sector walk"
        );
    }

    /// The writer-built gated-attention tinies saturate the attention gate, so
    /// `attn *= sigmoid(gate)` is a numerical no-op (or annihilates attention) on
    /// those fixtures and model-level tests cannot distinguish gated-Q from ungated
    /// attention. Recorded here so the limitation stays visible instead of being
    /// implied by a passing inequality assertion. A gated fixture whose
    /// pre-activation lands near zero would let the qwen35 test discriminate again.
    #[test]
    fn gated_attn_fixture_saturates_sigmoid() {
        // Observed qwen35 tiny gate pre-activation.
        assert_eq!(sigmoid_f32(19.617_193), 1.0);
        // Observed qwen3next tiny gate pre-activation.
        assert!(sigmoid_f32(-44.839_5) < 1e-19);
        // A non-degenerate gate would actually scale attention.
        assert!((sigmoid_f32(0.0) - 0.5).abs() < 1e-6);
    }

    /// Differential check against llama.cpp on a real quantized checkpoint.
    ///
    /// The writer-built tinies cannot substitute for this. Tiny fixtures use
    /// hand-picked scales that avoid the ranges where real weights live. A
    /// subnormal binary16 bug that halved real Q4_K / Q6_K weights survived
    /// 221 passing tests for that reason; per-dtype oracles now call
    /// `oracle_f16_to_f32` (IEEE arithmetic, not production bit-surgery)
    /// so that class of bug is visible without a real GGUF.
    /// This test is still the llama.cpp greedy identity on a real checkpoint.
    ///
    /// Skips unless `LLAMA_RUST_REAL_MODEL_DIR` names a directory holding the
    /// GGUF named in the sidecar JSON. Reference values live in
    /// `tests/reference/*.json` (NEOX Qwen + NORM Llama).
    #[test]
    fn real_model_matches_llama_cpp_reference() {
        check_real_fixture("qwen2.5-0.5b-instruct-q4_k_m.json", true);
    }

    /// NORM-RoPE Llama control (`llama-3.2-1b-instruct-q4_k_m.json`).
    /// A set `LLAMA_RUST_REAL_MODEL_DIR` must contain the GGUF (fail-loud,
    /// same as Qwen). Do not invent the capture.
    #[test]
    fn real_llama_norm_matches_llama_cpp_reference() {
        check_real_fixture("llama-3.2-1b-instruct-q4_k_m.json", true);
    }

    struct RealRef {
        file: String,
        architecture: String,
        n_vocab: u32,
        tokens: Vec<u32>,
        prompt: String,
        max_logit: f32,
        greedy_ids: Vec<u32>,
        greedy_text: String,
    }

    fn check_real_fixture(json_name: &str, required_when_env_set: bool) {
        let json_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("tests")
            .join("reference")
            .join(json_name);
        if !required_when_env_set && !json_path.is_file() {
            return;
        }
        let Ok(dir) = std::env::var("LLAMA_RUST_REAL_MODEL_DIR") else {
            return;
        };
        let spec = load_real_ref(&json_path);
        let path = std::path::Path::new(&dir).join(&spec.file);
        let opened = std::fs::File::open(&path);
        assert!(
            opened.is_ok(),
            "LLAMA_RUST_REAL_MODEL_DIR={dir} is set but {} could not be opened. \
             Tests run with the crate directory as CWD, so a relative path resolves \
             against langtax/, not the repo root -- pass an absolute path. \
             See tests/reference/README.md to fetch the weights.",
            path.display()
        );
        let mut file = opened.expect("checked by the assertion above");
        eprintln!("real-model differential test: {}", path.display());
        let mut blob = Vec::new();
        let _read = std::io::Read::read_to_end(&mut file, &mut blob).expect("read gguf");
        let g = load_gguf_owned(blob).expect("load real gguf");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String(spec.architecture.clone()))
        );

        let tok = Tokenizer::from_gguf(&g).expect("tokenizer");
        let ids = prompt_ids(&tok, &spec.prompt).expect("encode");
        assert_eq!(ids, spec.tokens, "tokenization diverged from llama.cpp");

        let model = Llama::from_gguf(g).expect("model");
        assert_eq!(
            model.n_vocab,
            usize::try_from(spec.n_vocab).expect("n_vocab")
        );

        let mut cache = model.new_cache(64).expect("cache");
        let logits = model.prefill(&mut cache, &ids).expect("prefill");
        let best = argmax(&logits);
        let want_first = *spec.greedy_ids.first().expect("greedy");
        assert_eq!(best, want_first, "argmax must be token {want_first}");

        let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (mx - spec.max_logit).abs() < 0.5,
            "max logit {mx} drifted from llama.cpp {}",
            spec.max_logit
        );

        let mut gen = Vec::new();
        let mut cur = best;
        for _ in 0..spec.greedy_ids.len() {
            gen.push(cur);
            let step = model.forward(&mut cache, cur).expect("forward");
            cur = argmax(&step);
        }
        assert_eq!(
            gen, spec.greedy_ids,
            "greedy token ids diverged from llama.cpp"
        );
        let text = tok.decode(&gen);
        assert_eq!(text, spec.greedy_text);
    }

    fn load_real_ref(path: &std::path::Path) -> RealRef {
        let mut f =
            std::fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
        let mut text = String::new();
        let _n = std::io::Read::read_to_string(&mut f, &mut text).expect("read json");
        RealRef {
            file: json_string(&text, "file"),
            architecture: json_string(&text, "architecture"),
            n_vocab: json_u32(&text, "n_vocab"),
            tokens: json_u32_list(&text, "tokens"),
            prompt: json_string(&text, "prompt"),
            max_logit: json_f32(&text, "max"),
            greedy_ids: json_u32_list(&text, "greedy_ids"),
            greedy_text: json_string(&text, "greedy_text"),
        }
    }

    fn after_json_key<'a>(text: &'a str, key: &str) -> &'a str {
        let needle = format!("\"{key}\":");
        text.split(&needle)
            .nth(1)
            .unwrap_or_else(|| panic!("missing {key}"))
            .trim()
    }

    fn json_string(text: &str, key: &str) -> String {
        let rest = after_json_key(text, key);
        let rest = rest
            .strip_prefix('"')
            .unwrap_or_else(|| panic!("{key} not a string"));
        let mut out = String::new();
        let mut chars = rest.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some(o) => out.push(o),
                    None => break,
                }
            } else if c == '"' {
                break;
            } else {
                out.push(c);
            }
        }
        out
    }

    fn json_u32(text: &str, key: &str) -> u32 {
        let tok = after_json_key(text, key)
            .split([',', '}', '\n'])
            .next()
            .unwrap_or("")
            .trim();
        tok.parse().unwrap_or_else(|_| panic!("bad u32 {key}"))
    }

    fn json_f32(text: &str, key: &str) -> f32 {
        let tok = after_json_key(text, key)
            .split([',', '}', '\n'])
            .next()
            .unwrap_or("")
            .trim();
        tok.parse().unwrap_or_else(|_| panic!("bad f32 {key}"))
    }

    fn json_u32_list(text: &str, key: &str) -> Vec<u32> {
        let rest = after_json_key(text, key);
        let start = rest.strip_prefix('[').expect("expected []");
        let inner = start.split(']').next().unwrap_or("");
        if inner.trim().is_empty() {
            return Vec::new();
        }
        inner
            .split(',')
            .map(|p| p.trim().parse().expect("bad u32 in list"))
            .collect()
    }

    #[test]
    fn real_qwen_sidecar_json_parses() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("tests")
            .join("reference")
            .join("qwen2.5-0.5b-instruct-q4_k_m.json");
        let spec = load_real_ref(&path);
        assert_eq!(spec.file, "qwen2.5-0.5b-instruct-q4_k_m.gguf");
        assert_eq!(spec.architecture, "qwen2");
        assert_eq!(spec.n_vocab, 151_936);
        assert_eq!(spec.tokens, vec![785, 6722, 315, 9625, 374]);
        assert_eq!(spec.greedy_ids.len(), 8);
        assert_eq!(spec.greedy_text, " Paris. It is the largest city in");
        assert!((spec.max_logit - 17.504_87).abs() < 1e-5);
        assert_eq!(spec.prompt, "The capital of France is");
    }

    #[test]
    fn real_llama_sidecar_json_parses() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("tests")
            .join("reference")
            .join("llama-3.2-1b-instruct-q4_k_m.json");
        let spec = load_real_ref(&path);
        assert_eq!(spec.file, "Llama-3.2-1B-Instruct-Q4_K_M.gguf");
        assert_eq!(spec.architecture, "llama");
        assert_eq!(spec.n_vocab, 128_256);
        assert_eq!(spec.tokens, vec![128000, 791, 6864, 315, 9822, 374]);
        assert_eq!(spec.greedy_ids.len(), 24);
        assert_eq!(
            spec.greedy_text,
            " Paris. The Eiffel Tower is a famous landmark in Paris. The Eiffel Tower is a symbol of France"
        );
        assert!((spec.max_logit - 18.816_57).abs() < 1e-5);
        assert_eq!(spec.prompt, "The capital of France is");
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

    /// GQA ratios other than 2 were entirely uncovered: every writer-built fixture
    /// pins `head_count = 4, head_count_kv = 2`. A grouping bug at any other ratio
    /// would pass the whole harness. Real checkpoints do use other ratios
    /// (Qwen2.5-0.5B is 14/2 = 7, covered only by the gated real-model test).
    #[test]
    fn gqa_ratios_other_than_two_match_independent_oracle() {
        for (bytes, n_head_kv, gqa) in [
            (tiny_mha_gguf(), 4u32, 1u32),
            (tiny_mqa_gguf(), 1, 4),
            (tiny_llama_gguf(), 2, 2),
        ] {
            let g = load_gguf(&bytes).expect("load");
            assert_eq!(g.kv_u32("llama.attention.head_count"), Some(4));
            assert_eq!(g.kv_u32("llama.attention.head_count_kv"), Some(n_head_kv));
            assert_eq!(4 / n_head_kv, gqa, "gqa ratio");
            // Single token and multi-token prefill both walk the KV grouping.
            load_fwd_match(&bytes, 3);
            load_prefill_match(&bytes, &[1, 2, 3]);
        }
    }

    /// The three ratios must not collapse onto the same logits, otherwise the
    /// test above would pass even if `head_count_kv` were ignored entirely.
    #[test]
    fn gqa_ratio_changes_logits() {
        let mut seen: Vec<Vec<f32>> = Vec::new();
        for bytes in [tiny_mha_gguf(), tiny_llama_gguf(), tiny_mqa_gguf()] {
            let g = load_gguf(&bytes).expect("load");
            let m = Llama::from_gguf(g).expect("model");
            let mut c = m.new_cache(8).expect("cache");
            seen.push(m.prefill(&mut c, &[1, 2, 3]).expect("prefill"));
        }
        assert_ne!(seen[0], seen[1], "gqa 1 vs 2 must differ");
        assert_ne!(seen[1], seen[2], "gqa 2 vs 4 must differ");
        assert_ne!(seen[0], seen[2], "gqa 1 vs 4 must differ");
    }

    #[test]
    fn tiny_q40_logits_match_independent_oracle() {
        let bytes = tiny_q40_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::Q4_0);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::Q4_0);
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::Q4_0);
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::Q4_0
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 2);
        load_fwd_match(&bytes, 3);
    }

    /// Pinned against ggml's own `to_float`, not against the in-tree oracle.
    ///
    /// These 18 bytes are one `block_q4_0` whose scale is `0x812B`, a binary16
    /// *subnormal*. That is deliberate: a subnormal super-block scale is exactly
    /// what exposed the `f16_to_f32` off-by-one that halved real Q4_K / Q6_K
    /// weights. The in-tree oracle now uses `oracle_f16_to_f32`
    /// (IEEE arithmetic) and would catch a repeat; this test still pins the
    /// dequant against ggml's captured `to_float` on these exact bytes. See
    /// `tests/reference/README.md` for the capture tool.
    #[test]
    fn q4_0_matches_ggml_reference_with_subnormal_scale() {
        let block: [u8; 18] = [
            0x2b, 0x81, 0xd9, 0x1e, 0x3f, 0x72, 0x1f, 0xcb, 0x19, 0x71, 0x17, 0x44, 0x94, 0xd6,
            0x49, 0x3c, 0x9d, 0x5c,
        ];
        // Full f32 precision, not rounded: at 1e-5 magnitudes a truncated decimal
        // dump keeps only ~3 significant digits and would pass against a halved
        // value, which is precisely the bug being guarded against.
        let want: [f32; 32] = [
            -1.7821789e-05,
            -1.0693073e-04,
            -1.2475252e-04,
            1.0693073e-04,
            -1.2475252e-04,
            -5.3465366e-05,
            -1.7821789e-05,
            1.2475252e-04,
            1.7821789e-05,
            7.1287155e-05,
            7.1287155e-05,
            3.5643578e-05,
            -1.7821789e-05,
            -7.1287155e-05,
            -8.9108944e-05,
            -7.1287155e-05,
            -8.9108944e-05,
            1.2475252e-04,
            8.9108944e-05,
            1.7821789e-05,
            1.2475252e-04,
            -7.1287155e-05,
            1.2475252e-04,
            1.7821789e-05,
            1.2475252e-04,
            7.1287155e-05,
            -1.7821789e-05,
            -8.9108944e-05,
            7.1287155e-05,
            8.9108944e-05,
            -1.7821789e-05,
            5.3465366e-05,
        ];
        let mut got = vec![0.0f32; 32];
        dequant_q4_0_row(32, &block, &mut got).expect("dequant q4_0");
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            assert!((a - b).abs() < 1e-9, "lane {i}: got {a:e}, ggml {b:e}");
        }
        // The oracle must agree too, but that is the weaker of the two checks.
        assert_eq!(got, dequant_q4_0_row_oracle(&block));
    }

    #[test]
    fn tiny_q80_logits_match_independent_oracle() {
        let bytes = tiny_q80_gguf();
        let g = load_gguf(&bytes).expect("load");
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty, GgmlType::Q8_0);
        assert_eq!(g.tensor("output.weight").unwrap().ty, GgmlType::Q8_0);
        assert_eq!(g.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::Q8_0);
        assert_eq!(
            g.tensor("blk.0.ffn_gate.weight").unwrap().ty,
            GgmlType::Q8_0
        );
        assert_eq!(g.tensor("output_norm.weight").unwrap().ty, GgmlType::F32);
        assert_eq!(
            g.tensor("blk.0.attn_norm.weight").unwrap().ty,
            GgmlType::F32
        );
        assert_eq!(g.tensor("token_embd.weight").unwrap().ty.to_i32(), 8);
        load_fwd_match(&bytes, 3);
    }

    /// `block_q8_0` is 34 bytes (fp16 `d` + `qs[32]`), distinct from Q8_1 = 9
    /// (36 B, extra fp16 `s`) and Q8_K = 15 (292 B). Reading Q8_0 with the Q8_1
    /// stride would silently misalign every block after the first.
    #[test]
    fn q8_0_block_layout_is_distinct_from_q8_1() {
        assert_eq!(crate::quant::Q8_0_BLOCK, 34);
        assert_eq!(crate::quant::Q8_1_BLOCK, 36);
        assert_eq!(QK8_0, 32);
        let qs: Vec<i8> = (0i8..32).map(|i| i.saturating_sub(16)).collect();
        let mut arr = [0i8; 32];
        arr.copy_from_slice(&qs);
        let block = pack_q8_0_block(0.5, &arr);
        assert_eq!(block.len(), crate::quant::Q8_0_BLOCK);
        let mut y = vec![0.0f32; 32];
        dequant_q8_0_row(32, &block, &mut y).expect("dequant q8_0");
        let want = dequant_q8_0_row_oracle(&block);
        assert_eq!(y, want);
        for (j, v) in y.iter().enumerate() {
            assert!((v - f32::from(qs[j]) * 0.5).abs() < 1e-6, "lane {j}");
        }
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
            tiny_q40_gguf(),
            tiny_q80_gguf(),
            tiny_mha_gguf(),
            tiny_mqa_gguf(),
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
            tiny_gemma2_gguf(),
            tiny_gemma3_gguf(),
            tiny_gemma3n_gguf(),
            tiny_gemma4_gguf(),
            tiny_gemma4_moe_gguf(),
            tiny_gemma4_ple_gguf(),
            tiny_gemma4_moe_ple_gguf(),
            tiny_gemma4_moe_fused_gguf(),
            tiny_gemma4_moe_fused_ple_gguf(),
            tiny_gemma4_shared_kv_gguf(),
            tiny_qwen3_gguf(),
            tiny_llama4_gguf(),
            tiny_llama_moe_gguf(),
            tiny_qwen2moe_gguf(),
            tiny_qwen3moe_gguf(),
            tiny_qwen3moe_2layer_gguf(),
            tiny_qwen2vl_gguf(),
            tiny_qwen3vl_gguf(),
            tiny_qwen3next_gguf(),
            tiny_qwen35_gguf(),
            tiny_phi2_gguf(),
            tiny_bloom_gguf(),
        ] {
            load_prefill_match(&bytes, &tokens);
            load_prefill_match(&bytes, &[3]);
        }
    }

    /// Every logit bit pattern from a decode step routed through the persistent
    /// GEMV pool, against the same step run single-threaded. A pooled GEMV
    /// computes rows `first..last` from the byte range `base + first*row_bytes`
    /// with the same kernel, so it must agree exactly, not just within the
    /// oracle's 1e-3.
    #[test]
    fn pooled_gemv_is_bit_identical_to_sequential() {
        fn steps(model: &Llama) -> (Vec<u32>, usize) {
            let mut cache = model.new_cache(16).expect("cache");
            let mut bits = Vec::new();
            let push = |l: &[f32], bits: &mut Vec<u32>| {
                bits.extend(l.iter().map(|v| v.to_bits()));
            };
            push(
                &model.prefill(&mut cache, &[1, 3, 2]).expect("prefill"),
                &mut bits,
            );
            for t in 0..4u32 {
                push(
                    &model.forward(&mut cache, t % 4).expect("forward"),
                    &mut bits,
                );
            }
            (bits, Llama::cache_pool_workers(&cache))
        }
        for (name, bytes) in [
            ("llama", tiny_llama_gguf()),
            ("qwen2", tiny_qwen2_gguf()),
            ("gemma", tiny_gemma_gguf()),
            ("gemma2", tiny_gemma2_gguf()),
            ("gemma3", tiny_gemma3_gguf()),
            ("gemma3n", tiny_gemma3n_gguf()),
            ("gemma4", tiny_gemma4_gguf()),
            ("gemma4-moe", tiny_gemma4_moe_gguf()),
            ("gemma4-ple", tiny_gemma4_ple_gguf()),
            ("gemma4-moe-ple", tiny_gemma4_moe_ple_gguf()),
            ("gemma4-moe-fused", tiny_gemma4_moe_fused_gguf()),
            ("gemma4-moe-fused-ple", tiny_gemma4_moe_fused_ple_gguf()),
            ("gemma4-shared-kv", tiny_gemma4_shared_kv_gguf()),
            ("qwen3", tiny_qwen3_gguf()),
            ("llama4", tiny_llama4_gguf()),
            ("f16", tiny_f16_gguf()),
            ("bf16", tiny_bf16_gguf()),
            ("q2k", tiny_q2k_gguf()),
            ("q5k", tiny_q5k_gguf()),
            ("iq2xxs", tiny_iq2xxs_gguf()),
            ("iq3s", tiny_iq3s_gguf()),
            ("tq2_0", tiny_tq20_gguf()),
        ] {
            let model = Llama::from_gguf(load_gguf_owned(bytes).expect("load")).expect("model");
            let (want, seq_workers) = crate::pool::with_sequential(|| steps(&model));
            assert_eq!(seq_workers, 0, "{name}: with_sequential built a pool");
            let (got, par_workers) = with_forced_pool(|| steps(&model));
            if crate::pool::sequential() || par_workers == 0 {
                // Single-core host: there is no pool to compare against.
                continue;
            }
            assert!(par_workers >= 2, "{name}: pool had {par_workers} workers");
            assert_eq!(got, want, "{name}: pooled GEMV changed a logit bit");
        }
    }

    /// A steady-state decode step must not touch the allocator. `unsafe impl
    /// GlobalAlloc` is not available under `#![forbid(unsafe_code)]`, so this
    /// checks the observable consequence instead: every reused buffer keeps its
    /// address and capacity, i.e. no `Vec` in the cache, its scratch or its
    /// pool grows or moves. A changed buffer count fails the same comparison.
    ///
    /// [`Llama::forward_logits`] is the entry point under test.
    /// [`Llama::forward`] copies the logits into a fresh `Vec` by definition,
    /// so it can never be allocation-free.
    ///
    /// The tiny fixtures cover the dense, GeGLU, QK-norm and MoE walks; the
    /// forced-pool pass covers the pool's input staging and spare buffers,
    /// which the fixtures are otherwise too small to reach.
    #[test]
    fn steady_state_decode_never_regrows_a_buffer() {
        /// One step through the borrowed-logits path, dropping the borrow so
        /// the caller can inspect the cache again.
        fn step(model: &Llama, cache: &mut KvCache, tok: u32) {
            let logits = model.forward_logits(cache, tok).expect("forward");
            assert!(!logits.is_empty(), "empty logits");
        }
        fn check(name: &str, model: &Llama) {
            let mut cache = model.new_cache(64).expect("cache");
            // Warm up: a 3-token prefill plus one decode step sizes every buffer.
            let _ = model.prefill(&mut cache, &[1, 3, 2]).expect("prefill");
            step(model, &mut cache, 1);
            let before = Llama::cache_buffer_ids(&cache);
            for s in 0..40u32 {
                step(model, &mut cache, s % 4);
                assert_eq!(
                    Llama::cache_buffer_ids(&cache),
                    before,
                    "{name}: decode step {s} reallocated a buffer"
                );
            }
            // A prefill wider than the warmup does grow the buffers once, and
            // then the following decode steps must be steady again.
            let mut cache = model.new_cache(64).expect("cache");
            step(model, &mut cache, 1);
            let narrow = Llama::cache_buffer_ids(&cache);
            let _ = model.prefill(&mut cache, &[1, 3, 2, 1, 3]).expect("wide");
            assert_ne!(
                Llama::cache_buffer_ids(&cache),
                narrow,
                "{name}: a 5-token prefill after a 1-token warmup should grow"
            );
            let grown = Llama::cache_buffer_ids(&cache);
            for s in 0..8u32 {
                step(model, &mut cache, s % 4);
                assert_eq!(
                    Llama::cache_buffer_ids(&cache),
                    grown,
                    "{name}: decode after prefill reallocated at step {s}"
                );
            }
        }
        for (name, bytes) in [
            ("llama", tiny_llama_gguf()),
            ("gemma", tiny_gemma_gguf()),
            ("gemma2", tiny_gemma2_gguf()),
            ("gemma3", tiny_gemma3_gguf()),
            ("gemma3n", tiny_gemma3n_gguf()),
            ("gemma4", tiny_gemma4_gguf()),
            ("gemma4-moe", tiny_gemma4_moe_gguf()),
            ("gemma4-ple", tiny_gemma4_ple_gguf()),
            ("gemma4-moe-ple", tiny_gemma4_moe_ple_gguf()),
            ("gemma4-moe-fused", tiny_gemma4_moe_fused_gguf()),
            ("gemma4-moe-fused-ple", tiny_gemma4_moe_fused_ple_gguf()),
            ("gemma4-shared-kv", tiny_gemma4_shared_kv_gguf()),
            ("qwen3", tiny_qwen3_gguf()),
            ("llama4", tiny_llama4_gguf()),
            ("llama-moe", tiny_llama_moe_gguf()),
            ("qwen2moe", tiny_qwen2moe_gguf()),
            ("qwen3moe", tiny_qwen3moe_gguf()),
            ("qwen3next", tiny_qwen3next_gguf()),
        ] {
            let model = Llama::from_gguf(load_gguf_owned(bytes).expect("load")).expect("model");
            check(name, &model);
            with_forced_pool(|| check(&format!("{name} pooled"), &model));
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
    fn reuse_prefix_rewinds_n_past_and_keeps_lcp_ids() {
        let model = Llama::from_gguf(load_gguf_owned(tiny_llama_gguf()).expect("load")).expect("m");
        let mut cache = model.new_cache(16).expect("c");
        let _l = model.prefill(&mut cache, &[1, 2, 3, 4]).expect("p");
        assert_eq!(cache.reuse_prefix(&[1, 2, 0]), 2);
        assert_eq!(cache.n_past, 2);
        assert_eq!(cache.cached_ids(), &[1, 2]);
        let _f = model.forward(&mut cache, 5).expect("append");
        assert_eq!(cache.n_past, 3);
        assert_eq!(cache.cached_ids(), &[1, 2, 5]);
    }

    #[test]
    fn prompt_reuse_logits_match_cold_prefill() {
        let tokens = [1u32, 2, 3, 4];
        for bytes in [tiny_llama_gguf(), tiny_qwen3moe_gguf(), tiny_llama4_gguf()] {
            let model = Llama::from_gguf(load_gguf_owned(bytes).expect("load")).expect("m");
            let mut cold = model.new_cache(16).expect("cold");
            let exp = model.prefill(&mut cold, &tokens).expect("cold");
            let mut hot = model.new_cache(16).expect("hot");
            let _p = model.prefill(&mut hot, &[1, 2, 3, 0]).expect("warm");
            let _d = model.forward(&mut hot, 5).expect("dec");
            let got = model.prompt(&mut hot, &tokens).expect("prompt");
            assert_logits_match(&got, &exp);
            assert_eq!(hot.n_past, tokens.len());
            assert_eq!(hot.cached_ids(), tokens.as_slice());
            assert_eq!(hot.last_prefix_hit(), 3);
            let _d2 = model.forward(&mut hot, 5).expect("dec2");
            let full = model.prompt(&mut hot, &tokens).expect("full");
            assert_logits_match(&full, &exp);
            assert_eq!(hot.last_prefix_hit(), tokens.len());
        }
    }

    #[test]
    fn prompt_reuse_greedy_matches_cold_and_store() {
        let bytes = tiny_qwen3moe_gguf();
        let g = load_gguf_owned(bytes).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("m");
        let cold = greedy_generate(&model, &tok, "ab", 2).expect("cold");
        let mut slot = None;
        let a = greedy_generate_slot(&model, &tok, &mut slot, "ab", 2, Some(16), None).expect("a");
        let hit = slot.as_ref().expect("slot").last_prefix_hit();
        let b = greedy_generate_slot(&model, &tok, &mut slot, "ab", 2, Some(16), None).expect("b");
        assert_eq!(a, cold);
        assert_eq!(b, cold);
        assert_eq!(hit, 0);
        let hit2 = slot.as_ref().expect("slot").last_prefix_hit();
        assert!(hit2 > 0, "second prompt must reuse the first prompt prefix");
        let mut held = model.new_cache(16).expect("store cache");
        held.attach_expert_store(LiveStore::Direct(
            model.expert_direct_store().expect("catalog"),
        ));
        let mut store_s = Some(held);
        let via =
            greedy_generate_slot(&model, &tok, &mut store_s, "ab", 2, Some(16), None).expect("st");
        assert_eq!(via, cold);
    }

    #[test]
    fn paged_kv_logits_match_dense_and_intern_after_divergent_prompt() {
        let tokens = [1u32, 2, 3, 4];
        for bytes in [tiny_llama_gguf(), tiny_qwen3moe_gguf(), tiny_llama4_gguf()] {
            let model = Llama::from_gguf(load_gguf_owned(bytes).expect("load")).expect("m");
            let mut dense = model.new_cache(16).expect("d");
            let exp = model.prefill(&mut dense, &tokens).expect("dense");
            let mut paged = model.new_paged_cache(16, 2).expect("p");
            let got = model.prefill(&mut paged, &tokens).expect("paged");
            assert_logits_match(&got, &exp);
            assert_eq!(paged.n_past, tokens.len());
            let cow_exp = {
                let mut d = model.new_cache(16).expect("cow dense");
                let _p = model.prefill(&mut d, &tokens).expect("cow pre");
                model
                    .prompt(&mut d, &[1, 0, 3, 4])
                    .expect("cow dense prompt")
            };
            let cow_got = model.prompt(&mut paged, &[1, 0, 3, 4]).expect("cow paged");
            assert_logits_match(&cow_got, &cow_exp);
            let _div = model.prompt(&mut paged, &[5, 0, 5, 0]).expect("div");
            let again = model.prompt(&mut paged, &tokens).expect("hit");
            assert_logits_match(&again, &exp);
            assert!(
                paged.page_hits() > 0,
                "interned blocks must be reusable after a rewind"
            );
        }
    }

    #[test]
    fn paged_kv_greedy_matches_dense() {
        for bytes in [tiny_llama_gguf(), tiny_qwen3moe_gguf(), tiny_llama4_gguf()] {
            let g = load_gguf_owned(bytes).expect("owned");
            let tok = Tokenizer::from_gguf(&g).expect("tok");
            let model = Llama::from_gguf(g).expect("m");
            let dense =
                greedy_generate_slot(&model, &tok, &mut None, "ab", 2, Some(16), None).expect("d");
            let paged = greedy_generate_slot(&model, &tok, &mut None, "ab", 2, Some(16), Some(2))
                .expect("p");
            assert_eq!(dense, paged, "paged greedy must match dense");
        }
    }

    #[test]
    fn prompt_chunk_two_steps_match_full_prompt_logits() {
        let tokens = [1u32, 2, 3, 4];
        let model = Llama::from_gguf(load_gguf_owned(tiny_llama_gguf()).expect("load")).expect("m");
        let mut full = model.new_cache(16).expect("f");
        let exp = model.prompt(&mut full, &tokens).expect("full");
        let mut chunked = model.new_paged_cache(16, 2).expect("c");
        let _a = model.prompt_chunk(&mut chunked, &tokens, 2).expect("c1");
        assert_eq!(chunked.n_past, 2);
        let got = model.prompt_chunk(&mut chunked, &tokens, 2).expect("c2");
        assert_logits_match(got, &exp);
        assert_eq!(chunked.n_past, tokens.len());
    }

    #[test]
    fn shared_paged_pool_interns_across_sequences() {
        let tokens = [1u32, 2, 3, 4];
        for bytes in [tiny_llama_gguf(), tiny_qwen3moe_gguf(), tiny_llama4_gguf()] {
            let model = Llama::from_gguf(load_gguf_owned(bytes).expect("load")).expect("m");
            let mut dense = model.new_cache(16).expect("d");
            let exp = model.prefill(&mut dense, &tokens).expect("dense");
            let pool = model.new_paged_pool(2, 16).expect("pool");
            let mut a = model.new_paged_cache_on(&pool, 16).expect("a");
            let mut b = model.new_paged_cache_on(&pool, 16).expect("b");
            let got = model.prefill(&mut a, &tokens).expect("a prefill");
            assert_logits_match(&got, &exp);
            let hit = model.prompt(&mut b, &tokens).expect("b intern");
            assert_logits_match(&hit, &exp);
            assert!(
                b.page_hits() > 0 && pool.hits() > 0,
                "second sequence must intern-hit the first sequence's full blocks"
            );
        }
    }

    #[test]
    fn forward_batch_two_paged_matches_sequential() {
        let prompt = [1u32, 2, 3, 4];
        for bytes in [tiny_llama_gguf(), tiny_qwen3moe_gguf(), tiny_llama4_gguf()] {
            let model = Llama::from_gguf(load_gguf_owned(bytes).expect("load")).expect("m");
            let pool = model.new_paged_pool(2, 16).expect("pool");
            let mut a = model.new_paged_cache_on(&pool, 16).expect("a");
            let mut b = model.new_paged_cache_on(&pool, 16).expect("b");
            let _pa = model.prefill(&mut a, &prompt).expect("pa");
            let _pb = model.prefill(&mut b, &[5, 0, 5, 0]).expect("pb");
            let mut a_seq = model.new_paged_cache_on(&pool, 16).expect("as");
            let mut b_seq = model.new_paged_cache_on(&pool, 16).expect("bs");
            let _pas = model.prefill(&mut a_seq, &prompt).expect("pas");
            let _pbs = model.prefill(&mut b_seq, &[5, 0, 5, 0]).expect("pbs");
            let exp_a = model.forward(&mut a_seq, 1).expect("fa");
            let exp_b = model.forward(&mut b_seq, 2).expect("fb");
            let mut pair = [&mut a, &mut b];
            let got = model.forward_batch(&mut pair, &[1, 2]).expect("batch");
            assert_eq!(got.len(), 2);
            assert_logits_match(&got[0], &exp_a);
            assert_logits_match(&got[1], &exp_b);
        }
    }

    #[test]
    fn forward_batch_three_paged_matches_sequential() {
        let prompts = [[1u32, 2, 3, 4], [5, 0, 5, 0], [1, 2, 0, 1]];
        let toks = [1u32, 2, 3];
        let model = Llama::from_gguf(load_gguf_owned(tiny_llama_gguf()).expect("load")).expect("m");
        let pool = model.new_paged_pool(2, 32).expect("pool");
        let mut a = model.new_paged_cache_on(&pool, 16).expect("a");
        let mut b = model.new_paged_cache_on(&pool, 16).expect("b");
        let mut c = model.new_paged_cache_on(&pool, 16).expect("c");
        let _pa = model.prefill(&mut a, &prompts[0]).expect("pa");
        let _pb = model.prefill(&mut b, &prompts[1]).expect("pb");
        let _pc = model.prefill(&mut c, &prompts[2]).expect("pc");
        let mut a_seq = model.new_paged_cache_on(&pool, 16).expect("as");
        let mut b_seq = model.new_paged_cache_on(&pool, 16).expect("bs");
        let mut c_seq = model.new_paged_cache_on(&pool, 16).expect("cs");
        let _pas = model.prefill(&mut a_seq, &prompts[0]).expect("pas");
        let _pbs = model.prefill(&mut b_seq, &prompts[1]).expect("pbs");
        let _pcs = model.prefill(&mut c_seq, &prompts[2]).expect("pcs");
        let exp_a = model.forward(&mut a_seq, toks[0]).expect("fa");
        let exp_b = model.forward(&mut b_seq, toks[1]).expect("fb");
        let exp_c = model.forward(&mut c_seq, toks[2]).expect("fc");
        let mut triple = [&mut a, &mut b, &mut c];
        let got = model.forward_batch(&mut triple, &toks).expect("batch");
        assert_eq!(got.len(), 3);
        assert_logits_match(&got[0], &exp_a);
        assert_logits_match(&got[1], &exp_b);
        assert_logits_match(&got[2], &exp_c);
    }

    #[test]
    fn forward_batch_mixed_dense_paged_matches_sequential() {
        let prompt = [1u32, 2, 3, 4];
        let model = Llama::from_gguf(load_gguf_owned(tiny_llama_gguf()).expect("load")).expect("m");
        let pool = model.new_paged_pool(2, 16).expect("pool");
        let mut dense = model.new_cache(16).expect("d");
        let mut paged = model.new_paged_cache_on(&pool, 16).expect("p");
        let _pd = model.prefill(&mut dense, &prompt).expect("pd");
        let _pp = model.prefill(&mut paged, &[5, 0, 5, 0]).expect("pp");
        let mut d_seq = model.new_cache(16).expect("ds");
        let mut p_seq = model.new_paged_cache_on(&pool, 16).expect("ps");
        let _pds = model.prefill(&mut d_seq, &prompt).expect("pds");
        let _pps = model.prefill(&mut p_seq, &[5, 0, 5, 0]).expect("pps");
        let exp_d = model.forward(&mut d_seq, 1).expect("fd");
        let exp_p = model.forward(&mut p_seq, 2).expect("fp");
        let mut pair = [&mut dense, &mut paged];
        let got = model.forward_batch(&mut pair, &[1, 2]).expect("batch");
        assert_eq!(got.len(), 2);
        assert_logits_match(&got[0], &exp_d);
        assert_logits_match(&got[1], &exp_p);
    }

    #[test]
    fn forward_batch_one_expert_store_matches_sequential_and_counts_hits() {
        let prompt = [1u32, 2, 3, 4];
        let model =
            Llama::from_gguf(load_gguf_owned(tiny_qwen3moe_gguf()).expect("load")).expect("m");
        let pool = model.new_paged_pool(2, 16).expect("pool");
        let mut a = model.new_paged_cache_on(&pool, 16).expect("a");
        let mut b = model.new_paged_cache_on(&pool, 16).expect("b");
        a.attach_expert_store(LiveStore::Direct(
            model.expert_direct_store().expect("catalog"),
        ));
        let _pa = model.prefill(&mut a, &prompt).expect("pa");
        let _pb = model.prefill(&mut b, &[5, 0, 5, 0]).expect("pb");
        let mut a_seq = model.new_paged_cache_on(&pool, 16).expect("as");
        let mut b_seq = model.new_paged_cache_on(&pool, 16).expect("bs");
        a_seq.attach_expert_store(LiveStore::Direct(
            model.expert_direct_store().expect("catalog2"),
        ));
        let _pas = model.prefill(&mut a_seq, &prompt).expect("pas");
        let _pbs = model.prefill(&mut b_seq, &[5, 0, 5, 0]).expect("pbs");
        let exp_a = model.forward(&mut a_seq, 1).expect("fa");
        let exp_b = model.forward(&mut b_seq, 2).expect("fb");
        let hits_before = a.expert_store_metrics().expect("mb").hits;
        let got = {
            let mut pair = [&mut a, &mut b];
            model.forward_batch(&mut pair, &[1, 2]).expect("batch")
        };
        assert_eq!(got.len(), 2);
        assert_logits_match(&got[0], &exp_a);
        assert_logits_match(&got[1], &exp_b);
        let hits_after = a.expert_store_metrics().expect("ma").hits;
        assert!(
            hits_after > hits_before,
            "batched GEMM must acquire from the parked store, hits {hits_before} -> {hits_after}"
        );
    }

    #[test]
    fn forward_batch_two_expert_stores_fallback_matches_sequential() {
        let prompt = [1u32, 2, 3, 4];
        let model =
            Llama::from_gguf(load_gguf_owned(tiny_qwen3moe_gguf()).expect("load")).expect("m");
        let pool = model.new_paged_pool(2, 16).expect("pool");
        let mut a = model.new_paged_cache_on(&pool, 16).expect("a");
        let mut b = model.new_paged_cache_on(&pool, 16).expect("b");
        a.attach_expert_store(LiveStore::Direct(
            model.expert_direct_store().expect("catalog"),
        ));
        b.attach_expert_store(LiveStore::Direct(
            model.expert_direct_store().expect("catalog-b"),
        ));
        let _pa = model.prefill(&mut a, &prompt).expect("pa");
        let _pb = model.prefill(&mut b, &[5, 0, 5, 0]).expect("pb");
        let mut a_seq = model.new_paged_cache_on(&pool, 16).expect("as");
        let mut b_seq = model.new_paged_cache_on(&pool, 16).expect("bs");
        a_seq.attach_expert_store(LiveStore::Direct(
            model.expert_direct_store().expect("catalog2"),
        ));
        b_seq.attach_expert_store(LiveStore::Direct(
            model.expert_direct_store().expect("catalog2b"),
        ));
        let _pas = model.prefill(&mut a_seq, &prompt).expect("pas");
        let _pbs = model.prefill(&mut b_seq, &[5, 0, 5, 0]).expect("pbs");
        let exp_a = model.forward(&mut a_seq, 1).expect("fa");
        let exp_b = model.forward(&mut b_seq, 2).expect("fb");
        let mut pair = [&mut a, &mut b];
        let got = model.forward_batch(&mut pair, &[1, 2]).expect("batch");
        assert_eq!(got.len(), 2);
        assert_logits_match(&got[0], &exp_a);
        assert_logits_match(&got[1], &exp_b);
    }

    #[test]
    fn prefill_batch_ragged_paged_matches_sequential() {
        let a_tok = [1u32, 2];
        let b_tok = [5u32, 0, 5, 0];
        for bytes in [tiny_llama_gguf(), tiny_qwen3moe_gguf(), tiny_llama4_gguf()] {
            let model = Llama::from_gguf(load_gguf_owned(bytes).expect("load")).expect("m");
            let pool = model.new_paged_pool(2, 16).expect("pool");
            let mut a = model.new_paged_cache_on(&pool, 16).expect("a");
            let mut b = model.new_paged_cache_on(&pool, 16).expect("b");
            let mut a_seq = model.new_paged_cache_on(&pool, 16).expect("as");
            let mut b_seq = model.new_paged_cache_on(&pool, 16).expect("bs");
            let exp_a = model.prefill(&mut a_seq, &a_tok).expect("fa");
            let exp_b = model.prefill(&mut b_seq, &b_tok).expect("fb");
            let mut pair = [&mut a, &mut b];
            let groups: [&[u32]; 2] = [&a_tok, &b_tok];
            let got = model.prefill_batch(&mut pair, &groups).expect("batch");
            assert_eq!(got.len(), 2);
            assert_eq!(a.n_past, a_tok.len());
            assert_eq!(b.n_past, b_tok.len());
            assert_logits_match(&got[0], &exp_a);
            assert_logits_match(&got[1], &exp_b);
        }
    }

    #[test]
    fn prefill_batch_one_expert_store_matches_sequential() {
        let a_tok = [1u32, 2];
        let b_tok = [5u32, 0, 5, 0];
        let model =
            Llama::from_gguf(load_gguf_owned(tiny_qwen3moe_gguf()).expect("load")).expect("m");
        let pool = model.new_paged_pool(2, 16).expect("pool");
        let mut a = model.new_paged_cache_on(&pool, 16).expect("a");
        let mut b = model.new_paged_cache_on(&pool, 16).expect("b");
        a.attach_expert_store(LiveStore::Direct(
            model.expert_direct_store().expect("catalog"),
        ));
        let mut a_seq = model.new_paged_cache_on(&pool, 16).expect("as");
        let mut b_seq = model.new_paged_cache_on(&pool, 16).expect("bs");
        a_seq.attach_expert_store(LiveStore::Direct(
            model.expert_direct_store().expect("catalog2"),
        ));
        let exp_a = model.prefill(&mut a_seq, &a_tok).expect("fa");
        let exp_b = model.prefill(&mut b_seq, &b_tok).expect("fb");
        let hits_before = a.expert_store_metrics().expect("mb").hits;
        let got = {
            let mut pair = [&mut a, &mut b];
            let groups: [&[u32]; 2] = [&a_tok, &b_tok];
            model.prefill_batch(&mut pair, &groups).expect("batch")
        };
        assert_eq!(got.len(), 2);
        assert_logits_match(&got[0], &exp_a);
        assert_logits_match(&got[1], &exp_b);
        let hits_after = a.expert_store_metrics().expect("ma").hits;
        assert!(
            hits_after > hits_before,
            "prefill GEMM must acquire from the parked store, hits {hits_before} -> {hits_after}"
        );
    }

    #[test]
    fn prefill_batch_equal_length_paged_matches_sequential() {
        let prompt = [1u32, 2, 3, 4];
        let other = [5u32, 0, 5, 0];
        let model = Llama::from_gguf(load_gguf_owned(tiny_llama_gguf()).expect("load")).expect("m");
        let pool = model.new_paged_pool(2, 16).expect("pool");
        let mut a = model.new_paged_cache_on(&pool, 16).expect("a");
        let mut b = model.new_paged_cache_on(&pool, 16).expect("b");
        let mut a_seq = model.new_paged_cache_on(&pool, 16).expect("as");
        let mut b_seq = model.new_paged_cache_on(&pool, 16).expect("bs");
        let exp_a = model.prefill(&mut a_seq, &prompt).expect("fa");
        let exp_b = model.prefill(&mut b_seq, &other).expect("fb");
        let mut pair = [&mut a, &mut b];
        let groups: [&[u32]; 2] = [&prompt, &other];
        let got = model.prefill_batch(&mut pair, &groups).expect("batch");
        assert_eq!(got.len(), 2);
        assert_logits_match(&got[0], &exp_a);
        assert_logits_match(&got[1], &exp_b);
    }

    #[test]
    fn prefill_batch_traced_qwen3moe_matches_sequential() {
        let a_tok = [1u32, 2, 3, 4];
        let b_tok = [5u32, 0, 5, 0];
        let model =
            Llama::from_gguf(load_gguf_owned(tiny_qwen3moe_gguf()).expect("load")).expect("m");
        let pool = model.new_paged_pool(2, 16).expect("pool");
        let mut a_seq = model.new_paged_cache_on(&pool, 16).expect("as");
        let mut b_seq = model.new_paged_cache_on(&pool, 16).expect("bs");
        a_seq.enable_moe_trace(1);
        b_seq.enable_moe_trace(2);
        let exp_a = model.prefill(&mut a_seq, &a_tok).expect("fa");
        let exp_b = model.prefill(&mut b_seq, &b_tok).expect("fb");
        let tr_a = a_seq.take_moe_trace();
        let tr_b = b_seq.take_moe_trace();
        assert!(!tr_a.events.is_empty(), "sequential A must emit MoE events");
        assert!(!tr_b.events.is_empty(), "sequential B must emit MoE events");
        let mut a = model.new_paged_cache_on(&pool, 16).expect("a");
        let mut b = model.new_paged_cache_on(&pool, 16).expect("b");
        a.enable_moe_trace(1);
        b.enable_moe_trace(2);
        let got = {
            let mut pair = [&mut a, &mut b];
            let groups: [&[u32]; 2] = [&a_tok, &b_tok];
            model.prefill_batch(&mut pair, &groups).expect("batch")
        };
        assert_eq!(got.len(), 2);
        assert_logits_match(&got[0], &exp_a);
        assert_logits_match(&got[1], &exp_b);
        assert_eq!(a.take_moe_trace(), tr_a);
        assert_eq!(b.take_moe_trace(), tr_b);
    }

    #[test]
    fn prefill_batch_trace_does_not_dump_on_untraced_first() {
        let a_tok = [1u32, 2, 3, 4];
        let b_tok = [5u32, 0, 5, 0];
        let model =
            Llama::from_gguf(load_gguf_owned(tiny_qwen3moe_gguf()).expect("load")).expect("m");
        let pool = model.new_paged_pool(2, 16).expect("pool");
        let mut b_seq = model.new_paged_cache_on(&pool, 16).expect("bs");
        b_seq.enable_moe_trace(2);
        let _exp_b = model.prefill(&mut b_seq, &b_tok).expect("fb");
        let tr_b = b_seq.take_moe_trace();
        let mut a = model.new_paged_cache_on(&pool, 16).expect("a");
        let mut b = model.new_paged_cache_on(&pool, 16).expect("b");
        b.enable_moe_trace(2);
        let _got = {
            let mut pair = [&mut a, &mut b];
            let groups: [&[u32]; 2] = [&a_tok, &b_tok];
            model.prefill_batch(&mut pair, &groups).expect("batch")
        };
        assert!(
            a.take_moe_trace().events.is_empty(),
            "untraced first cache must not receive the GEMM-local log"
        );
        assert_eq!(b.take_moe_trace(), tr_b);
    }

    #[test]
    fn forward_batch_traced_qwen3moe_matches_sequential() {
        let prompt_a = [1u32, 2, 3, 4];
        let prompt_b = [5u32, 0, 5, 0];
        let model =
            Llama::from_gguf(load_gguf_owned(tiny_qwen3moe_gguf()).expect("load")).expect("m");
        let pool = model.new_paged_pool(2, 16).expect("pool");
        let mut a_seq = model.new_paged_cache_on(&pool, 16).expect("as");
        let mut b_seq = model.new_paged_cache_on(&pool, 16).expect("bs");
        let _pa = model.prefill(&mut a_seq, &prompt_a).expect("pas");
        let _pb = model.prefill(&mut b_seq, &prompt_b).expect("pbs");
        a_seq.enable_moe_trace(1);
        b_seq.enable_moe_trace(2);
        let exp_a = model.forward(&mut a_seq, 1).expect("fa");
        let exp_b = model.forward(&mut b_seq, 2).expect("fb");
        let tr_a = a_seq.take_moe_trace();
        let tr_b = b_seq.take_moe_trace();
        assert!(!tr_a.events.is_empty());
        assert!(!tr_b.events.is_empty());
        let mut a = model.new_paged_cache_on(&pool, 16).expect("a");
        let mut b = model.new_paged_cache_on(&pool, 16).expect("b");
        let _pa = model.prefill(&mut a, &prompt_a).expect("pa");
        let _pb = model.prefill(&mut b, &prompt_b).expect("pb");
        a.enable_moe_trace(1);
        b.enable_moe_trace(2);
        let got = {
            let mut pair = [&mut a, &mut b];
            model.forward_batch(&mut pair, &[1, 2]).expect("batch")
        };
        assert_eq!(got.len(), 2);
        assert_logits_match(&got[0], &exp_a);
        assert_logits_match(&got[1], &exp_b);
        assert_eq!(a.take_moe_trace(), tr_a);
        assert_eq!(b.take_moe_trace(), tr_b);
    }

    #[test]
    fn prepare_append_twice_multi_block_is_idempotent() {
        let model = Llama::from_gguf(load_gguf_owned(tiny_llama_gguf()).expect("load")).expect("m");
        let mut cache = model.new_paged_cache(16, 2).expect("c");
        cache.prepare_append(4).expect("p1");
        assert_eq!(cache.page_table_len(), 2);
        cache.prepare_append(4).expect("p2");
        assert_eq!(cache.page_table_len(), 2);
        cache.prepare_append(5).expect("p3");
        assert_eq!(cache.page_table_len(), 3);
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
    fn tiny_gemma2_load_gemv_gemm_embed_and_greedy() {
        let bytes = tiny_gemma2_gguf();
        let g = load_gguf(&bytes).expect("load gemma2");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("gemma2".into()))
        );
        assert_eq!(g.kv_u32("gemma2.block_count"), Some(1));
        assert_eq!(g.kv_u32("gemma2.embedding_length"), Some(256));
        assert_eq!(g.kv_u32("gemma2.feed_forward_length"), Some(256));
        assert_eq!(g.kv_u32("gemma2.attention.head_count"), Some(4));
        assert_eq!(g.kv_u32("gemma2.attention.head_count_kv"), Some(2));
        assert_eq!(g.kv_u32("gemma2.context_length"), Some(256));
        assert_eq!(
            g.kv_f32("gemma2.attn_logit_softcapping"),
            Some(GEMMA2_ATTN_LOGIT_SOFTCAPPING)
        );
        assert_eq!(
            g.kv_f32("gemma2.final_logit_softcapping"),
            Some(GEMMA2_FINAL_LOGIT_SOFTCAPPING)
        );
        assert_eq!(
            g.kv_u32("gemma2.attention.sliding_window"),
            Some(GEMMA2_TINY_N_SWA)
        );
        assert_eq!(g.kv_u32("gemma2.attention.key_length"), Some(64));
        assert_eq!(g.kv_u32("gemma2.attention.value_length"), Some(64));
        assert_eq!(
            g.kv_f32("gemma2.attention.layer_norm_rms_epsilon"),
            Some(1.0 / 100_000.0)
        );
        assert!(g.kv_u32("gemma2.rope.dimension_count").is_none());
        assert!(g
            .kv_u32("gemma2.attention.sliding_window_pattern")
            .is_none());
        assert!(g.kv_u32("llama.block_count").is_none());
        assert!(g.kv_u32("gemma.block_count").is_none());
        assert!(g.kv_u32("mixtral.block_count").is_none());
        assert!(g.kv_u32("qwen3vlmoe.block_count").is_none());
        assert!(g.tensor("output.weight").is_none());
        assert!(g.tensor("blk.0.ffn_norm.weight").is_some());
        assert!(g.tensor("blk.0.post_attention_norm.weight").is_some());
        assert!(g.tensor("blk.0.post_ffw_norm.weight").is_some());
        assert!(g.tensor("blk.0.attn_post_norm.weight").is_none());
        assert!(g.tensor("blk.0.ffn_post_norm.weight").is_none());
        let model = Llama::from_gguf(g.clone()).expect("model");
        let x = pat_f32(TINY_N_EMBD, 21);
        let got_gemv = model.gemv_output(&x).expect("gemv");
        let exp_gemv = oracle_gemv(g.tensor("token_embd.weight").unwrap(), &x);
        assert_logits_match(&got_gemv, &exp_gemv);
        let mut x2 = pat_f32(TINY_N_EMBD, 22);
        x2.extend(pat_f32(TINY_N_EMBD, 23));
        let got_gemm = model.gemm_output(2, &x2).expect("gemm");
        let exp_gemm = {
            let mut y = oracle_gemv(g.tensor("token_embd.weight").unwrap(), &x2[..TINY_N_EMBD]);
            y.extend(oracle_gemv(
                g.tensor("token_embd.weight").unwrap(),
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
        let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
        let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
        assert_eq!(out, out2);
        assert!(!out.is_empty());
        let tokens = [1u32, 2, 3];
        let gemma_pref = {
            let ge = load_gguf(&tiny_gemma_gguf()).expect("gemma");
            let m = Llama::from_gguf(ge).expect("mg");
            let mut c = m.new_cache(8).expect("cg");
            m.prefill(&mut c, &tokens).expect("gemma pref")
        };
        let mut gc = model.new_cache(8).expect("gc");
        let gemma2_pref = model.prefill(&mut gc, &tokens).expect("gemma2 pref");
        assert_ne!(
            gemma2_pref, gemma_pref,
            "gemma2 post-norms+softcap+SWA must change logits vs gemma"
        );
        assert!(
            gemma2_is_swa(0, GEMMA2_SWA_PERIOD_DEFAULT),
            "layer 0 is SWA under set_swa_pattern(2)"
        );
        assert!(
            !gemma2_is_swa(1, GEMMA2_SWA_PERIOD_DEFAULT),
            "layer 1 is dense under set_swa_pattern(2)"
        );
    }

    #[test]
    fn tiny_gemma3_load_gemv_gemm_embed_and_greedy() {
        let bytes = tiny_gemma3_gguf();
        let g = load_gguf(&bytes).expect("load gemma3");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("gemma3".into()))
        );
        assert_eq!(g.kv_u32("gemma3.block_count"), Some(1));
        assert_eq!(g.kv_u32("gemma3.embedding_length"), Some(256));
        assert_eq!(g.kv_u32("gemma3.feed_forward_length"), Some(256));
        assert_eq!(g.kv_u32("gemma3.attention.head_count"), Some(4));
        assert_eq!(g.kv_u32("gemma3.attention.head_count_kv"), Some(2));
        assert_eq!(g.kv_u32("gemma3.context_length"), Some(256));
        assert_eq!(
            g.kv_u32("gemma3.attention.sliding_window"),
            Some(GEMMA2_TINY_N_SWA)
        );
        assert_eq!(g.kv_u32("gemma3.attention.key_length"), Some(64));
        assert_eq!(g.kv_u32("gemma3.attention.value_length"), Some(64));
        assert!(g.kv_f32("gemma3.attn_logit_softcapping").is_none());
        assert!(g.kv_f32("gemma3.final_logit_softcapping").is_none());
        assert!(g.kv_u32("gemma3.rope.dimension_count").is_none());
        assert!(g
            .kv_u32("gemma3.attention.sliding_window_pattern")
            .is_none());
        assert!(g.kv_u32("gemma2.block_count").is_none());
        assert!(g.kv_u32("llama.block_count").is_none());
        assert!(g.kv_u32("mixtral.block_count").is_none());
        assert!(g.tensor("output.weight").is_none());
        assert!(g.tensor("blk.0.ffn_norm.weight").is_some());
        assert!(g.tensor("blk.0.post_attention_norm.weight").is_some());
        assert!(g.tensor("blk.0.post_ffw_norm.weight").is_some());
        assert!(g.tensor("blk.0.attn_q_norm.weight").is_some());
        assert!(g.tensor("blk.0.attn_k_norm.weight").is_some());
        assert_eq!(
            g.tensor("blk.0.attn_q_norm.weight").unwrap().n_cols(),
            TINY_N_ROT
        );
        let model = Llama::from_gguf(g.clone()).expect("model");
        let x = pat_f32(TINY_N_EMBD, 21);
        let got_gemv = model.gemv_output(&x).expect("gemv");
        let exp_gemv = oracle_gemv(g.tensor("token_embd.weight").unwrap(), &x);
        assert_logits_match(&got_gemv, &exp_gemv);
        load_fwd_match(&bytes, 3);
        load_prefill_match(&bytes, &[1, 2, 3]);
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
        let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
        assert_eq!(out, out2);
        assert!(!out.is_empty());
        let tokens = [1u32, 2, 3];
        let gemma2_pref = {
            let g2 = load_gguf(&tiny_gemma2_gguf()).expect("gemma2");
            let m = Llama::from_gguf(g2).expect("m2");
            let mut c = m.new_cache(8).expect("c2");
            m.prefill(&mut c, &tokens).expect("gemma2 pref")
        };
        let mut gc = model.new_cache(8).expect("gc");
        let gemma3_pref = model.prefill(&mut gc, &tokens).expect("gemma3 pref");
        assert_ne!(
            gemma3_pref, gemma2_pref,
            "gemma3 QK-Norm and no attn softcap must change logits vs gemma2"
        );
        assert!(
            gemma2_is_swa(0, GEMMA3_SWA_PERIOD_DEFAULT),
            "layer 0 is SWA under set_swa_pattern(6)"
        );
        assert!(
            gemma2_is_swa(4, GEMMA3_SWA_PERIOD_DEFAULT),
            "layer 4 is SWA under set_swa_pattern(6)"
        );
        assert!(
            !gemma2_is_swa(5, GEMMA3_SWA_PERIOD_DEFAULT),
            "layer 5 is dense under set_swa_pattern(6)"
        );
    }

    #[test]
    fn tiny_gemma3n_load_gemv_gemm_embed_and_greedy() {
        let bytes = tiny_gemma3n_gguf();
        let g = load_gguf(&bytes).expect("load gemma3n");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("gemma3n".into()))
        );
        assert_eq!(g.kv_u32("gemma3n.block_count"), Some(1));
        assert_eq!(g.kv_u32("gemma3n.embedding_length"), Some(256));
        assert_eq!(g.kv_u32("gemma3n.feed_forward_length"), Some(256));
        assert_eq!(g.kv_u32("gemma3n.attention.head_count"), Some(4));
        assert_eq!(g.kv_u32("gemma3n.attention.head_count_kv"), Some(2));
        assert_eq!(g.kv_u32("gemma3n.context_length"), Some(256));
        assert_eq!(
            g.kv_u32("gemma3n.attention.sliding_window"),
            Some(GEMMA2_TINY_N_SWA)
        );
        assert_eq!(g.kv_u32("gemma3n.attention.key_length"), Some(64));
        assert_eq!(g.kv_u32("gemma3n.attention.value_length"), Some(64));
        assert_eq!(
            g.kv_f32("gemma3n.final_logit_softcapping"),
            Some(GEMMA2_FINAL_LOGIT_SOFTCAPPING)
        );
        assert_eq!(
            g.kv_u32("gemma3n.altup.num_inputs"),
            Some(u32::try_from(GEMMA3N_N_ALTUP).unwrap())
        );
        assert_eq!(
            g.kv_u32("gemma3n.altup.active_idx"),
            Some(u32::try_from(GEMMA3N_I_ALTUP_ACT).unwrap())
        );
        assert_eq!(
            g.kv_u32("gemma3n.embedding_length_per_layer_input"),
            Some(u32::try_from(GEMMA3N_N_EMBD_ALTUP).unwrap())
        );
        assert_eq!(
            g.kv_u32("gemma3n.attention.shared_kv_layers"),
            Some(GEMMA3N_N_LAYER_KV_FROM_START)
        );
        assert!(g.kv_f32("gemma3n.attn_logit_softcapping").is_none());
        assert!(g.kv_u32("gemma3n.rope.dimension_count").is_none());
        assert!(g
            .kv_u32("gemma3n.attention.sliding_window_pattern")
            .is_none());
        assert!(g.kv_u32("gemma3.block_count").is_none());
        assert!(g.kv_u32("llama.block_count").is_none());
        assert!(g.kv_u32("mixtral.block_count").is_none());
        assert!(g.tensor("output.weight").is_none());
        assert!(g.tensor("altup_proj.weight").is_some());
        assert!(g.tensor("altup_unembd_proj.weight").is_some());
        assert!(g.tensor("per_layer_token_embd.weight").is_some());
        assert!(g.tensor("per_layer_model_proj.weight").is_some());
        assert!(g.tensor("per_layer_proj_norm.weight").is_some());
        assert!(g.tensor("blk.0.ffn_norm.weight").is_some());
        assert!(g.tensor("blk.0.post_attention_norm.weight").is_some());
        assert!(g.tensor("blk.0.post_ffw_norm.weight").is_some());
        assert!(g.tensor("blk.0.attn_q_norm.weight").is_some());
        assert!(g.tensor("blk.0.attn_k_norm.weight").is_some());
        assert!(g.tensor("blk.0.inp_gate.weight").is_some());
        assert!(g.tensor("blk.0.laurel_l.weight").is_some());
        assert_eq!(
            g.tensor("blk.0.attn_q_norm.weight").unwrap().n_cols(),
            TINY_N_ROT
        );
        let model = Llama::from_gguf(g.clone()).expect("model");
        let x = pat_f32(TINY_N_EMBD, 21);
        let got_gemv = model.gemv_output(&x).expect("gemv");
        let exp_gemv = oracle_gemv(g.tensor("token_embd.weight").unwrap(), &x);
        assert_logits_match(&got_gemv, &exp_gemv);
        load_fwd_match(&bytes, 3);
        load_prefill_match(&bytes, &[1, 2, 3]);
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
        let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
        assert_eq!(out, out2);
        assert!(!out.is_empty());
        let tokens = [1u32, 2, 3];
        let gemma3_pref = {
            let g3 = load_gguf(&tiny_gemma3_gguf()).expect("gemma3");
            let m = Llama::from_gguf(g3).expect("m3");
            let mut c = m.new_cache(8).expect("c3");
            m.prefill(&mut c, &tokens).expect("gemma3 pref")
        };
        let mut gc = model.new_cache(8).expect("gc");
        let gemma3n_pref = model.prefill(&mut gc, &tokens).expect("gemma3n pref");
        assert_ne!(
            gemma3n_pref, gemma3_pref,
            "gemma3n AltUp/Laurel/V-norm/attn-scale must change logits vs gemma3"
        );
        assert!(
            gemma2_is_swa(0, GEMMA3N_SWA_PERIOD_DEFAULT),
            "layer 0 is SWA under set_swa_pattern(5)"
        );
        assert!(
            gemma2_is_swa(3, GEMMA3N_SWA_PERIOD_DEFAULT),
            "layer 3 is SWA under set_swa_pattern(5)"
        );
        assert!(
            !gemma2_is_swa(4, GEMMA3N_SWA_PERIOD_DEFAULT),
            "layer 4 is dense under set_swa_pattern(5)"
        );
    }

    #[test]
    fn tiny_gemma4_load_gemv_gemm_embed_and_greedy() {
        let bytes = tiny_gemma4_gguf();
        let g = load_gguf(&bytes).expect("load gemma4");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("gemma4".into()))
        );
        assert_eq!(g.kv_u32("gemma4.block_count"), Some(1));
        assert_eq!(g.kv_u32("gemma4.embedding_length"), Some(256));
        assert_eq!(g.kv_u32("gemma4.feed_forward_length"), Some(256));
        assert_eq!(g.kv_u32("gemma4.attention.head_count"), Some(4));
        assert_eq!(g.kv_u32("gemma4.attention.head_count_kv"), Some(2));
        assert_eq!(g.kv_u32("gemma4.context_length"), Some(256));
        assert_eq!(
            g.kv_u32("gemma4.attention.sliding_window"),
            Some(GEMMA2_TINY_N_SWA)
        );
        assert_eq!(g.kv_u32("gemma4.attention.key_length"), Some(64));
        assert_eq!(g.kv_u32("gemma4.attention.value_length"), Some(64));
        assert_eq!(g.kv_u32("gemma4.attention.key_length_swa"), Some(64));
        assert_eq!(g.kv_u32("gemma4.attention.value_length_swa"), Some(64));
        assert_eq!(g.kv_u32("gemma4.embedding_length_per_layer_input"), Some(0));
        assert_eq!(
            g.kv("gemma4.attention.sliding_window_pattern"),
            Some(&Kv::Array {
                elem: GGUF_TYPE_BOOL,
                items: vec![Kv::Bool(true)],
            })
        );
        assert!(g.kv_f32("gemma4.attn_logit_softcapping").is_none());
        assert!(g.kv_f32("gemma4.final_logit_softcapping").is_none());
        assert!(g.kv_u32("gemma4.rope.dimension_count").is_none());
        assert!(g.kv_u32("gemma4.attention.shared_kv_layers").is_none());
        assert!(g.kv_u32("gemma3.block_count").is_none());
        assert!(g.kv_u32("gemma3n.block_count").is_none());
        assert!(g.kv_u32("llama.block_count").is_none());
        assert!(g.kv_u32("mixtral.block_count").is_none());
        assert!(g.tensor("output.weight").is_none());
        assert!(g.tensor("altup_proj.weight").is_none());
        assert!(g.tensor("per_layer_token_embd.weight").is_none());
        assert!(g.tensor("blk.0.ffn_gate_inp.weight").is_none());
        assert!(g.tensor("blk.0.ffn_norm.weight").is_some());
        assert!(g.tensor("blk.0.post_attention_norm.weight").is_some());
        assert!(g.tensor("blk.0.post_ffw_norm.weight").is_some());
        assert!(g.tensor("blk.0.attn_q_norm.weight").is_some());
        assert!(g.tensor("blk.0.attn_k_norm.weight").is_some());
        assert_eq!(
            g.tensor("blk.0.attn_q_norm.weight").unwrap().n_cols(),
            TINY_N_ROT
        );
        let model = Llama::from_gguf(g.clone()).expect("model");
        assert!(model.gemma4);
        assert_eq!(model.is_swa, vec![true]);
        let x = pat_f32(TINY_N_EMBD, 21);
        let got_gemv = model.gemv_output(&x).expect("gemv");
        let exp_gemv = oracle_gemv(g.tensor("token_embd.weight").unwrap(), &x);
        assert_logits_match(&got_gemv, &exp_gemv);
        load_fwd_match(&bytes, 3);
        load_prefill_match(&bytes, &[1, 2, 3]);
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
        let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
        assert_eq!(out, out2);
        assert!(!out.is_empty());
        let tokens = [1u32, 2, 3];
        let gemma3_pref = {
            let g3 = load_gguf(&tiny_gemma3_gguf()).expect("gemma3");
            let m = Llama::from_gguf(g3).expect("m3");
            let mut c = m.new_cache(8).expect("c3");
            m.prefill(&mut c, &tokens).expect("gemma3 pref")
        };
        let mut gc = model.new_cache(8).expect("gc");
        let gemma4_pref = model.prefill(&mut gc, &tokens).expect("gemma4 pref");
        assert_ne!(
            gemma4_pref, gemma3_pref,
            "gemma4 V-norm and attn-scale 1.0 must change logits vs gemma3"
        );
        let gemma3n_pref = {
            let g3n = load_gguf(&tiny_gemma3n_gguf()).expect("gemma3n");
            let m = Llama::from_gguf(g3n).expect("m3n");
            let mut c = m.new_cache(8).expect("c3n");
            m.prefill(&mut c, &tokens).expect("gemma3n pref")
        };
        assert_ne!(
            gemma4_pref, gemma3n_pref,
            "gemma4 must not copy gemma3n AltUp/Laurel/per-layer/softcap"
        );
    }

    #[test]
    fn tiny_gemma4_moe_load_gemv_gemm_embed_and_greedy() {
        let bytes = tiny_gemma4_moe_gguf();
        let g = load_gguf(&bytes).expect("load gemma4 moe");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("gemma4".into()))
        );
        assert_eq!(g.kv_u32("gemma4.block_count"), Some(1));
        assert_eq!(g.kv_u32("gemma4.embedding_length"), Some(256));
        assert_eq!(g.kv_u32("gemma4.feed_forward_length"), Some(256));
        assert_eq!(g.kv_u32("gemma4.expert_count"), Some(4));
        assert_eq!(g.kv_u32("gemma4.expert_used_count"), Some(2));
        assert_eq!(g.kv_u32("gemma4.expert_feed_forward_length"), Some(256));
        assert_eq!(g.kv_u32("gemma4.attention.head_count"), Some(4));
        assert_eq!(g.kv_u32("gemma4.attention.head_count_kv"), Some(2));
        assert_eq!(
            g.kv_u32("gemma4.attention.sliding_window"),
            Some(GEMMA2_TINY_N_SWA)
        );
        assert_eq!(g.kv_u32("gemma4.attention.key_length_swa"), Some(64));
        assert_eq!(g.kv_u32("gemma4.attention.value_length_swa"), Some(64));
        assert_eq!(g.kv_u32("gemma4.embedding_length_per_layer_input"), Some(0));
        assert!(g.kv_u32("gemma4.attention.shared_kv_layers").is_none());
        assert!(g.kv_u32("qwen3moe.block_count").is_none());
        assert!(g.kv_u32("llama.block_count").is_none());
        assert!(g.kv_u32("mixtral.block_count").is_none());
        assert!(g.tensor("output.weight").is_none());
        assert!(g.tensor("blk.0.ffn_gate.weight").is_some());
        assert!(g.tensor("blk.0.ffn_up.weight").is_some());
        assert!(g.tensor("blk.0.ffn_down.weight").is_some());
        assert!(g.tensor("blk.0.ffn_gate_inp.weight").is_some());
        assert!(g.tensor("blk.0.ffn_gate_inp.scale").is_some());
        assert!(g.tensor("blk.0.ffn_post_norm_1.weight").is_some());
        assert!(g.tensor("blk.0.ffn_pre_norm_2.weight").is_some());
        assert!(g.tensor("blk.0.ffn_post_norm_2.weight").is_some());
        assert!(g.tensor("blk.0.ffn_gate_exps.weight").is_some());
        assert!(g.tensor("blk.0.ffn_up_exps.weight").is_some());
        assert!(g.tensor("blk.0.ffn_down_exps.weight").is_some());
        assert!(g.tensor("blk.0.ffn_gate_up_exps.weight").is_none());
        assert_eq!(
            g.tensor("blk.0.ffn_gate_exps.weight").unwrap().shape,
            &[256, 256, 4]
        );
        let model = Llama::from_gguf(g.clone()).expect("model");
        assert!(model.gemma4);
        assert_eq!(model.is_swa, vec![true]);
        let x = pat_f32(TINY_N_EMBD, 21);
        let got_gemv = model.gemv_output(&x).expect("gemv");
        let exp_gemv = oracle_gemv(g.tensor("token_embd.weight").unwrap(), &x);
        assert_logits_match(&got_gemv, &exp_gemv);
        load_fwd_match(&bytes, 3);
        load_prefill_match(&bytes, &[1, 2, 3]);
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
        let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
        assert_eq!(out, out2);
        assert!(!out.is_empty());
        let tokens = [1u32, 2, 3];
        let dense_pref = {
            let dg = load_gguf(&tiny_gemma4_gguf()).expect("dense gemma4");
            let dm = Llama::from_gguf(dg).expect("dm");
            let mut c = dm.new_cache(8).expect("cd");
            dm.prefill(&mut c, &tokens).expect("dense pref")
        };
        let mut mc = model.new_cache(8).expect("mc");
        let moe_pref = model.prefill(&mut mc, &tokens).expect("moe pref");
        assert_ne!(
            moe_pref, dense_pref,
            "gemma4 MoE must change logits vs dense gemma4"
        );
        let qwen3moe_pref = {
            let q3 = load_gguf(&tiny_qwen3moe_gguf()).expect("qwen3moe");
            let m3 = Llama::from_gguf(q3).expect("m3");
            let mut c = m3.new_cache(8).expect("c3");
            m3.prefill(&mut c, &tokens).expect("qwen3moe pref")
        };
        assert_ne!(
            moe_pref, qwen3moe_pref,
            "gemma4 MoE must not copy qwen3moe SILU / ffn_norm router"
        );
        let llama_moe_pref = {
            let lg = load_gguf(&tiny_llama_moe_gguf()).expect("llama moe");
            let lm = Llama::from_gguf(lg).expect("lm");
            let mut c = lm.new_cache(8).expect("cl");
            lm.prefill(&mut c, &tokens).expect("llama moe pref")
        };
        assert_ne!(
            moe_pref, llama_moe_pref,
            "gemma4 MoE must not copy llama-MoE SwiGLU / ffn_norm router"
        );
        let catalog = model.expert_direct_store().expect("catalog");
        let via_direct = store_prefill(&model, LiveStore::Direct(catalog), &tokens);
        assert_eq!(moe_pref, via_direct, "DirectStore GEMV must match the blob");
    }

    #[test]
    fn tiny_gemma4_moe_trace_is_identity_and_feeds_expertvm() {
        let bytes = tiny_gemma4_moe_gguf();
        let g = load_gguf(&bytes).expect("load");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("model");
        let plain = greedy_generate(&model, &tok, "ab", 4).expect("plain");
        let (traced, trace) =
            greedy_generate_traced(&model, &tok, "ab", 4, None, 0).expect("traced");
        assert_eq!(plain, traced, "tracing must not change greedy tokens");
        assert!(!trace.events.is_empty(), "MoE layers must emit accesses");
        let mut by_token: BTreeMap<u32, u64> = BTreeMap::new();
        for e in &trace.events {
            assert_eq!(e.layer, 0);
            assert_eq!(e.experts.len(), TINY_GEMMA4_N_EXPERT_USED);
            assert_eq!(e.weight_pt.len(), e.experts.len());
            for w in &e.weight_pt {
                assert!(*w <= 1000);
            }
            let p = e.prefix.expect("decode traces hash the token-id prefix");
            match by_token.get(&e.token) {
                None => {
                    let _prev = by_token.insert(e.token, p);
                }
                Some(old) => assert_eq!(*old, p, "same token must share prefix hash"),
            }
        }
        assert!(
            by_token.len() >= 2,
            "prefill + decode must emit more than one prefix hash"
        );
        let mut store = expertvm::DirectStore::from_trace(&trace);
        for k in trace.keys() {
            let blob = expertvm::ExpertStore::acquire(&mut store, k).expect("blob");
            assert_eq!(blob.gate, vec![1]);
        }
        assert_eq!(expertvm::ExpertStore::misses(&store), 0);
    }

    #[test]
    fn tiny_gemma4_ple_load_gemv_gemm_embed_and_greedy() {
        let bytes = tiny_gemma4_ple_gguf();
        let g = load_gguf(&bytes).expect("load gemma4 ple");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("gemma4".into()))
        );
        assert_eq!(g.kv_u32("gemma4.block_count"), Some(1));
        assert_eq!(g.kv_u32("gemma4.embedding_length"), Some(256));
        assert_eq!(
            g.kv_u32("gemma4.embedding_length_per_layer_input"),
            Some(u32::try_from(TINY_GEMMA4_N_EMBD_PER_LAYER).unwrap())
        );
        assert_eq!(
            g.kv_u32("gemma4.attention.sliding_window"),
            Some(GEMMA2_TINY_N_SWA)
        );
        assert!(g.kv_u32("gemma4.attention.shared_kv_layers").is_none());
        assert!(g.kv_u32("gemma4.expert_count").is_none());
        assert!(g.tensor("output.weight").is_none());
        assert!(g.tensor("altup_proj.weight").is_none());
        assert!(g.tensor("blk.0.ffn_gate_inp.weight").is_none());
        assert!(g.tensor("per_layer_token_embd.weight").is_some());
        assert!(g.tensor("per_layer_model_proj.weight").is_some());
        assert!(g.tensor("per_layer_proj_norm.weight").is_some());
        assert!(g.tensor("blk.0.inp_gate.weight").is_some());
        assert!(g.tensor("blk.0.proj.weight").is_some());
        assert!(g.tensor("blk.0.post_norm.weight").is_some());
        assert_eq!(
            g.tensor("per_layer_token_embd.weight").unwrap().shape,
            &[
                u64::try_from(TINY_GEMMA4_N_EMBD_PER_LAYER).unwrap(),
                u64::try_from(TINY_N_VOCAB).unwrap()
            ]
        );
        assert_eq!(
            g.tensor("per_layer_model_proj.weight").unwrap().shape,
            &[
                u64::try_from(TINY_N_EMBD).unwrap(),
                u64::try_from(TINY_GEMMA4_N_EMBD_PER_LAYER).unwrap()
            ]
        );
        let model = Llama::from_gguf(g.clone()).expect("model");
        assert!(model.gemma4);
        assert!(model.gemma4_ple.is_some());
        assert_eq!(model.is_swa, vec![true]);
        let x = pat_f32(TINY_N_EMBD, 21);
        let got_gemv = model.gemv_output(&x).expect("gemv");
        let exp_gemv = oracle_gemv(g.tensor("token_embd.weight").unwrap(), &x);
        assert_logits_match(&got_gemv, &exp_gemv);
        load_fwd_match(&bytes, 3);
        load_prefill_match(&bytes, &[1, 2, 3]);
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
        let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
        assert_eq!(out, out2);
        assert!(!out.is_empty());
        let tokens = [1u32, 2, 3];
        let dense_pref = {
            let dg = load_gguf(&tiny_gemma4_gguf()).expect("dense gemma4");
            let dm = Llama::from_gguf(dg).expect("dm");
            let mut c = dm.new_cache(8).expect("cd");
            dm.prefill(&mut c, &tokens).expect("dense pref")
        };
        let mut pc = model.new_cache(8).expect("pc");
        let ple_pref = model.prefill(&mut pc, &tokens).expect("ple pref");
        assert_ne!(
            ple_pref, dense_pref,
            "gemma4 PLE must change logits vs dense gemma4"
        );
        let gemma3n_pref = {
            let g3n = load_gguf(&tiny_gemma3n_gguf()).expect("gemma3n");
            let m = Llama::from_gguf(g3n).expect("m3n");
            let mut c = m.new_cache(8).expect("c3n");
            m.prefill(&mut c, &tokens).expect("gemma3n pref")
        };
        assert_ne!(
            ple_pref, gemma3n_pref,
            "gemma4 PLE must not copy gemma3n AltUp/Laurel/softcap"
        );
        let moe_pref = {
            let mg = load_gguf(&tiny_gemma4_moe_gguf()).expect("moe");
            let mm = Llama::from_gguf(mg).expect("mm");
            let mut c = mm.new_cache(8).expect("cm");
            mm.prefill(&mut c, &tokens).expect("moe pref")
        };
        assert_ne!(ple_pref, moe_pref, "gemma4 PLE must not copy gemma4 MoE");
    }

    #[test]
    fn tiny_gemma4_moe_ple_load_gemv_gemm_embed_and_greedy() {
        let bytes = tiny_gemma4_moe_ple_gguf();
        let g = load_gguf(&bytes).expect("load gemma4 moe ple");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("gemma4".into()))
        );
        assert_eq!(g.kv_u32("gemma4.block_count"), Some(1));
        assert_eq!(g.kv_u32("gemma4.embedding_length"), Some(256));
        assert_eq!(g.kv_u32("gemma4.expert_count"), Some(4));
        assert_eq!(g.kv_u32("gemma4.expert_used_count"), Some(2));
        assert_eq!(
            g.kv_u32("gemma4.embedding_length_per_layer_input"),
            Some(u32::try_from(TINY_GEMMA4_N_EMBD_PER_LAYER).unwrap())
        );
        assert!(g.kv_u32("gemma4.attention.shared_kv_layers").is_none());
        assert!(g.tensor("output.weight").is_none());
        assert!(g.tensor("altup_proj.weight").is_none());
        assert!(g.tensor("blk.0.ffn_gate_inp.weight").is_some());
        assert!(g.tensor("blk.0.ffn_gate_inp.scale").is_some());
        assert!(g.tensor("blk.0.ffn_post_norm_1.weight").is_some());
        assert!(g.tensor("blk.0.ffn_pre_norm_2.weight").is_some());
        assert!(g.tensor("blk.0.ffn_post_norm_2.weight").is_some());
        assert!(g.tensor("blk.0.ffn_gate_exps.weight").is_some());
        assert!(g.tensor("blk.0.ffn_up_exps.weight").is_some());
        assert!(g.tensor("blk.0.ffn_down_exps.weight").is_some());
        assert!(g.tensor("blk.0.ffn_gate_up_exps.weight").is_none());
        assert!(g.tensor("per_layer_token_embd.weight").is_some());
        assert!(g.tensor("per_layer_model_proj.weight").is_some());
        assert!(g.tensor("per_layer_proj_norm.weight").is_some());
        assert!(g.tensor("blk.0.inp_gate.weight").is_some());
        assert!(g.tensor("blk.0.proj.weight").is_some());
        assert!(g.tensor("blk.0.post_norm.weight").is_some());
        let model = Llama::from_gguf(g.clone()).expect("model");
        assert!(model.gemma4);
        assert!(model.gemma4_ple.is_some());
        assert_eq!(model.is_swa, vec![true]);
        let x = pat_f32(TINY_N_EMBD, 21);
        let got_gemv = model.gemv_output(&x).expect("gemv");
        let exp_gemv = oracle_gemv(g.tensor("token_embd.weight").unwrap(), &x);
        assert_logits_match(&got_gemv, &exp_gemv);
        load_fwd_match(&bytes, 3);
        load_prefill_match(&bytes, &[1, 2, 3]);
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
        let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
        assert_eq!(out, out2);
        assert!(!out.is_empty());
        let tokens = [1u32, 2, 3];
        let mut combo_c = model.new_cache(8).expect("cc");
        let combo_pref = model.prefill(&mut combo_c, &tokens).expect("combo pref");
        let dense_pref = {
            let dg = load_gguf(&tiny_gemma4_gguf()).expect("dense");
            let dm = Llama::from_gguf(dg).expect("dm");
            let mut c = dm.new_cache(8).expect("cd");
            dm.prefill(&mut c, &tokens).expect("dense pref")
        };
        assert_ne!(
            combo_pref, dense_pref,
            "gemma4 MoE+PLE must change logits vs dense gemma4"
        );
        let moe_pref = {
            let mg = load_gguf(&tiny_gemma4_moe_gguf()).expect("moe");
            let mm = Llama::from_gguf(mg).expect("mm");
            let mut c = mm.new_cache(8).expect("cm");
            mm.prefill(&mut c, &tokens).expect("moe pref")
        };
        assert_ne!(
            combo_pref, moe_pref,
            "gemma4 MoE+PLE must change logits vs MoE-only"
        );
        let ple_pref = {
            let pg = load_gguf(&tiny_gemma4_ple_gguf()).expect("ple");
            let pm = Llama::from_gguf(pg).expect("pm");
            let mut c = pm.new_cache(8).expect("cp");
            pm.prefill(&mut c, &tokens).expect("ple pref")
        };
        assert_ne!(
            combo_pref, ple_pref,
            "gemma4 MoE+PLE must change logits vs PLE-only"
        );
        let catalog = model.expert_direct_store().expect("catalog");
        let via_direct = store_prefill(&model, LiveStore::Direct(catalog), &tokens);
        assert_eq!(
            combo_pref, via_direct,
            "DirectStore GEMV must match the blob"
        );
    }

    #[test]
    fn tiny_gemma4_moe_ple_trace_is_identity_and_feeds_expertvm() {
        let bytes = tiny_gemma4_moe_ple_gguf();
        let g = load_gguf(&bytes).expect("load");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("model");
        let plain = greedy_generate(&model, &tok, "ab", 4).expect("plain");
        let (traced, trace) =
            greedy_generate_traced(&model, &tok, "ab", 4, None, 0).expect("traced");
        assert_eq!(plain, traced, "tracing must not change greedy tokens");
        assert!(!trace.events.is_empty(), "MoE layers must emit accesses");
        for e in &trace.events {
            assert_eq!(e.layer, 0);
            assert_eq!(e.experts.len(), TINY_GEMMA4_N_EXPERT_USED);
            assert_eq!(e.weight_pt.len(), e.experts.len());
        }
        let mut store = expertvm::DirectStore::from_trace(&trace);
        for k in trace.keys() {
            let blob = expertvm::ExpertStore::acquire(&mut store, k).expect("blob");
            assert_eq!(blob.gate, vec![1]);
        }
        assert_eq!(expertvm::ExpertStore::misses(&store), 0);
    }

    #[test]
    fn tiny_gemma4_moe_fused_load_gemv_gemm_embed_and_greedy() {
        let bytes = tiny_gemma4_moe_fused_gguf();
        let g = load_gguf(&bytes).expect("load gemma4 fused");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("gemma4".into()))
        );
        assert_eq!(g.kv_u32("gemma4.expert_count"), Some(4));
        assert_eq!(g.kv_u32("gemma4.expert_used_count"), Some(2));
        assert!(g.tensor("blk.0.ffn_gate_up_exps.weight").is_some());
        assert!(g.tensor("blk.0.ffn_gate_exps.weight").is_none());
        assert!(g.tensor("blk.0.ffn_up_exps.weight").is_none());
        assert!(g.tensor("blk.0.ffn_down_exps.weight").is_some());
        assert_eq!(
            g.tensor("blk.0.ffn_gate_up_exps.weight").unwrap().shape,
            &[256, 512, 4]
        );
        assert_eq!(
            g.tensor("blk.0.ffn_gate_up_exps.weight").unwrap().ty,
            GgmlType::F32
        );
        let model = Llama::from_gguf(g.clone()).expect("model");
        assert!(model.gemma4);
        load_fwd_match(&bytes, 3);
        load_prefill_match(&bytes, &[1, 2, 3]);
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
        let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
        assert_eq!(out, out2);
        let tokens = [1u32, 2, 3];
        let mut fc = model.new_cache(8).expect("fc");
        let fused_pref = model.prefill(&mut fc, &tokens).expect("fused pref");
        let moe_pref = {
            let mg = load_gguf(&tiny_gemma4_moe_gguf()).expect("moe");
            let mm = Llama::from_gguf(mg).expect("mm");
            let mut c = mm.new_cache(8).expect("cm");
            mm.prefill(&mut c, &tokens).expect("moe pref")
        };
        assert_ne!(
            fused_pref, moe_pref,
            "fused packing must not copy split-expert logits"
        );
        let catalog = model.expert_direct_store().expect("catalog");
        let via_direct = store_prefill(&model, LiveStore::Direct(catalog), &tokens);
        assert_eq!(
            fused_pref, via_direct,
            "DirectStore fused GEMV must match the blob"
        );
    }

    #[test]
    fn tiny_gemma4_moe_fused_trace_is_identity_and_feeds_expertvm() {
        let bytes = tiny_gemma4_moe_fused_gguf();
        let g = load_gguf(&bytes).expect("load");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("model");
        let plain = greedy_generate(&model, &tok, "ab", 4).expect("plain");
        let (traced, trace) =
            greedy_generate_traced(&model, &tok, "ab", 4, None, 0).expect("traced");
        assert_eq!(plain, traced, "tracing must not change greedy tokens");
        assert!(!trace.events.is_empty(), "MoE layers must emit accesses");
        let mut by_token: BTreeMap<u32, u64> = BTreeMap::new();
        for e in &trace.events {
            assert_eq!(e.layer, 0);
            assert_eq!(e.experts.len(), TINY_GEMMA4_N_EXPERT_USED);
            assert_eq!(e.weight_pt.len(), e.experts.len());
            for w in &e.weight_pt {
                assert!(*w <= 1000);
            }
            let p = e.prefix.expect("decode traces hash the token-id prefix");
            match by_token.get(&e.token) {
                None => {
                    let _prev = by_token.insert(e.token, p);
                }
                Some(old) => assert_eq!(*old, p, "same token must share prefix hash"),
            }
        }
        assert!(
            by_token.len() >= 2,
            "prefill + decode must emit more than one prefix hash"
        );
        let mut store = expertvm::DirectStore::from_trace(&trace);
        for k in trace.keys() {
            let blob = expertvm::ExpertStore::acquire(&mut store, k).expect("blob");
            assert_eq!(blob.gate, vec![1]);
        }
        assert_eq!(expertvm::ExpertStore::misses(&store), 0);
    }

    #[test]
    fn tiny_gemma4_moe_fused_ple_load_gemv_gemm_embed_and_greedy() {
        let bytes = tiny_gemma4_moe_fused_ple_gguf();
        let g = load_gguf(&bytes).expect("load gemma4 fused ple");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("gemma4".into()))
        );
        assert_eq!(g.kv_u32("gemma4.expert_count"), Some(4));
        assert_eq!(g.kv_u32("gemma4.expert_used_count"), Some(2));
        assert_eq!(
            g.kv_u32("gemma4.embedding_length_per_layer_input"),
            Some(u32::try_from(TINY_GEMMA4_N_EMBD_PER_LAYER).unwrap())
        );
        assert!(g.tensor("blk.0.ffn_gate_up_exps.weight").is_some());
        assert!(g.tensor("blk.0.ffn_gate_exps.weight").is_none());
        assert!(g.tensor("blk.0.ffn_up_exps.weight").is_none());
        assert!(g.tensor("blk.0.ffn_down_exps.weight").is_some());
        assert!(g.tensor("per_layer_token_embd.weight").is_some());
        assert!(g.tensor("blk.0.inp_gate.weight").is_some());
        assert!(g.tensor("blk.0.proj.weight").is_some());
        assert!(g.tensor("blk.0.post_norm.weight").is_some());
        assert_eq!(
            g.tensor("blk.0.ffn_gate_up_exps.weight").unwrap().shape,
            &[256, 512, 4]
        );
        let model = Llama::from_gguf(g.clone()).expect("model");
        assert!(model.gemma4);
        assert!(model.gemma4_ple.is_some());
        load_fwd_match(&bytes, 3);
        load_prefill_match(&bytes, &[1, 2, 3]);
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
        let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
        assert_eq!(out, out2);
        let tokens = [1u32, 2, 3];
        let mut fc = model.new_cache(8).expect("fc");
        let combo_pref = model.prefill(&mut fc, &tokens).expect("combo pref");
        let fused_pref = {
            let fg = load_gguf(&tiny_gemma4_moe_fused_gguf()).expect("fused");
            let fm = Llama::from_gguf(fg).expect("fm");
            let mut c = fm.new_cache(8).expect("cf");
            fm.prefill(&mut c, &tokens).expect("fused pref")
        };
        assert_ne!(
            combo_pref, fused_pref,
            "fused plus PLE must change logits vs fused-only"
        );
        let moe_ple_pref = {
            let mg = load_gguf(&tiny_gemma4_moe_ple_gguf()).expect("moe ple");
            let mm = Llama::from_gguf(mg).expect("mm");
            let mut c = mm.new_cache(8).expect("cm");
            mm.prefill(&mut c, &tokens).expect("moe ple pref")
        };
        assert_ne!(
            combo_pref, moe_ple_pref,
            "fused plus PLE must change logits vs split MoE plus PLE"
        );
        let ple_pref = {
            let pg = load_gguf(&tiny_gemma4_ple_gguf()).expect("ple");
            let pm = Llama::from_gguf(pg).expect("pm");
            let mut c = pm.new_cache(8).expect("cp");
            pm.prefill(&mut c, &tokens).expect("ple pref")
        };
        assert_ne!(
            combo_pref, ple_pref,
            "fused plus PLE must change logits vs PLE-only"
        );
        let catalog = model.expert_direct_store().expect("catalog");
        let via_direct = store_prefill(&model, LiveStore::Direct(catalog), &tokens);
        assert_eq!(
            combo_pref, via_direct,
            "DirectStore fused plus PLE GEMV must match the blob"
        );
    }

    #[test]
    fn tiny_gemma4_moe_fused_ple_trace_is_identity_and_feeds_expertvm() {
        let bytes = tiny_gemma4_moe_fused_ple_gguf();
        let g = load_gguf(&bytes).expect("load");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("model");
        let plain = greedy_generate(&model, &tok, "ab", 4).expect("plain");
        let (traced, trace) =
            greedy_generate_traced(&model, &tok, "ab", 4, None, 0).expect("traced");
        assert_eq!(plain, traced, "tracing must not change greedy tokens");
        assert!(!trace.events.is_empty(), "MoE layers must emit accesses");
        let mut by_token: BTreeMap<u32, u64> = BTreeMap::new();
        for e in &trace.events {
            assert_eq!(e.layer, 0);
            assert_eq!(e.experts.len(), TINY_GEMMA4_N_EXPERT_USED);
            assert_eq!(e.weight_pt.len(), e.experts.len());
            for w in &e.weight_pt {
                assert!(*w <= 1000);
            }
            let p = e.prefix.expect("decode traces hash the token-id prefix");
            match by_token.get(&e.token) {
                None => {
                    let _prev = by_token.insert(e.token, p);
                }
                Some(old) => assert_eq!(*old, p, "same token must share prefix hash"),
            }
        }
        assert!(
            by_token.len() >= 2,
            "prefill + decode must emit more than one prefix hash"
        );
        let mut store = expertvm::DirectStore::from_trace(&trace);
        for k in trace.keys() {
            let blob = expertvm::ExpertStore::acquire(&mut store, k).expect("blob");
            assert_eq!(blob.gate, vec![1]);
        }
        assert_eq!(expertvm::ExpertStore::misses(&store), 0);
    }

    #[test]
    fn gemma4_kv_slot_matches_llama_cpp_reuse() {
        assert!(gemma4_has_kv(2, 0));
        assert!(gemma4_has_kv(2, 1));
        assert!(!gemma4_has_kv(2, 2));
        assert_eq!(gemma4_kv_slot(2, 0, true).expect("l0"), 0);
        assert_eq!(gemma4_kv_slot(2, 1, false).expect("l1"), 1);
        assert_eq!(gemma4_kv_slot(2, 2, true).expect("swa share"), 0);
        assert_eq!(gemma4_kv_slot(2, 2, false).expect("global share"), 1);
        assert!(gemma4_kv_slot(1, 1, true).is_err());
        assert!(gemma4_has_kv(-19, 0));
    }

    #[test]
    fn tiny_gemma4_shared_kv_load_gemv_gemm_embed_and_greedy() {
        let bytes = tiny_gemma4_shared_kv_gguf();
        let g = load_gguf(&bytes).expect("load gemma4 shared kv");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("gemma4".into()))
        );
        assert_eq!(g.kv_u32("gemma4.block_count"), Some(3));
        assert_eq!(g.kv_u32("gemma4.attention.shared_kv_layers"), Some(1));
        assert_eq!(
            g.kv("gemma4.attention.sliding_window_pattern"),
            Some(&Kv::Array {
                elem: GGUF_TYPE_BOOL,
                items: vec![Kv::Bool(true), Kv::Bool(true), Kv::Bool(true)],
            })
        );
        assert!(g.tensor("blk.0.attn_k.weight").is_some());
        assert!(g.tensor("blk.1.attn_k.weight").is_some());
        assert!(g.tensor("blk.2.attn_k.weight").is_none());
        assert!(g.tensor("blk.2.attn_v.weight").is_none());
        assert!(g.tensor("blk.2.attn_k_norm.weight").is_none());
        assert!(g.tensor("blk.2.attn_q.weight").is_some());
        assert!(g.tensor("blk.2.attn_q_norm.weight").is_some());
        assert!(g.tensor("blk.2.attn_output.weight").is_some());
        let model = Llama::from_gguf(g.clone()).expect("model");
        assert!(model.gemma4);
        assert_eq!(model.layers.len(), 3);
        assert!(model.layers[0].wk.is_some());
        assert!(model.layers[1].wk.is_some());
        assert!(model.layers[2].wk.is_none());
        assert!(model.layers[2].wv.is_none());
        assert_eq!(model.layers[0].kv_slot, 0);
        assert_eq!(model.layers[1].kv_slot, 1);
        assert_eq!(model.layers[2].kv_slot, 0);
        assert_eq!(model.is_swa, vec![true, true, true]);
        load_fwd_match(&bytes, 3);
        load_prefill_match(&bytes, &[1, 2, 3]);
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let out = greedy_generate(&model, &tok, "ab", 2).expect("gen");
        let out2 = greedy_generate(&model, &tok, "ab", 2).expect("gen2");
        assert_eq!(out, out2);
        let tokens = [1u32, 2, 3];
        let mut sc = model.new_cache(8).expect("sc");
        let shared_pref = model.prefill(&mut sc, &tokens).expect("shared pref");
        let dense_pref = {
            let dg = load_gguf(&tiny_gemma4_gguf()).expect("dense");
            let dm = Llama::from_gguf(dg).expect("dm");
            let mut c = dm.new_cache(8).expect("dc");
            dm.prefill(&mut c, &tokens).expect("dense pref")
        };
        assert_ne!(
            shared_pref, dense_pref,
            "shared KV 3-layer must not copy 1-layer dense logits"
        );
        let full_pref = {
            let fb = expand_tiny_gemma4_layers(tiny_gemma4_gguf(), 3, 0).expect("full kv");
            let fg = load_gguf(&fb).expect("full g");
            assert!(fg.tensor("blk.2.attn_k.weight").is_some());
            assert!(fg.kv_u32("gemma4.attention.shared_kv_layers").is_none());
            let fm = Llama::from_gguf(fg).expect("fm");
            assert!(fm.layers[2].wk.is_some());
            assert_eq!(fm.layers[2].kv_slot, 2);
            let mut c = fm.new_cache(8).expect("fc");
            fm.prefill(&mut c, &tokens).expect("full pref")
        };
        assert_ne!(
            shared_pref, full_pref,
            "reusing layer 0 KV must not copy a 3-layer full-KV walk"
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
    fn tiny_qwen3moe_trace_is_identity_and_feeds_expertvm() {
        let bytes = tiny_qwen3moe_gguf();
        let g = load_gguf(&bytes).expect("load");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("model");
        let plain = greedy_generate(&model, &tok, "ab", 4).expect("plain");
        let (traced, trace) =
            greedy_generate_traced(&model, &tok, "ab", 4, None, 0).expect("traced");
        assert_eq!(plain, traced, "tracing must not change greedy tokens");
        assert!(!trace.events.is_empty(), "MoE layers must emit accesses");
        let mut by_token: BTreeMap<u32, u64> = BTreeMap::new();
        for e in &trace.events {
            assert_eq!(e.layer, 0);
            assert_eq!(e.experts.len(), TINY_QWEN3MOE_N_EXPERT_USED);
            assert_eq!(e.weight_pt.len(), e.experts.len());
            for w in &e.weight_pt {
                assert!(*w <= 1000);
            }
            let p = e.prefix.expect("decode traces hash the token-id prefix");
            match by_token.get(&e.token) {
                None => {
                    let _prev = by_token.insert(e.token, p);
                }
                Some(old) => assert_eq!(*old, p, "same token must share prefix hash"),
            }
        }
        assert!(
            by_token.len() >= 2,
            "prefill + decode must emit more than one prefix hash"
        );
        let hashes: Vec<u64> = by_token.values().copied().collect();
        assert_ne!(
            hashes[0], hashes[1],
            "a later token must change the prefix hash"
        );
        let mut store = expertvm::DirectStore::from_trace(&trace);
        for k in trace.keys() {
            let blob = expertvm::ExpertStore::acquire(&mut store, k).expect("blob");
            assert_eq!(blob.gate, vec![1]);
        }
        assert_eq!(expertvm::ExpertStore::misses(&store), 0);
        let table = expertvm::compare(&trace, 2, 8);
        let oracle = table
            .iter()
            .find(|r| r.policy == expertvm::Policy::Oracle)
            .expect("oracle");
        let lru = table
            .iter()
            .find(|r| r.policy == expertvm::Policy::Lru)
            .expect("lru");
        assert!(oracle.hits >= lru.hits);
    }

    fn moe_cached_prefetches(bytes: Vec<u8>, tokens: &[u32]) -> u64 {
        let g = load_gguf_owned(bytes).expect("owned");
        let model = Llama::from_gguf(g).expect("m");
        let catalog = model.expert_direct_store().expect("catalog");
        let n = catalog.len().max(1);
        let cached = expertvm::CachedStore::new(catalog, n).expect("cached");
        let mut c = model
            .new_cache(tokens.len().saturating_add(2))
            .expect("cache");
        c.attach_expert_store(LiveStore::Cached(cached));
        let _l = model.prefill(&mut c, tokens).expect("prefill");
        c.expert_store_metrics().expect("metrics").prefetches
    }

    #[test]
    fn tiny_qwen3moe_2layer_oracle_store_trace_and_copy_forward() {
        let bytes = tiny_qwen3moe_2layer_gguf();
        let g = load_gguf(&bytes).expect("load 2layer");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("qwen3moe".into()))
        );
        assert_eq!(g.kv_u32("qwen3moe.block_count"), Some(2));
        assert!(g.tensor("blk.1.ffn_gate_exps.weight").is_some());
        assert!(g.tensor("blk.1.ffn_up_exps.weight").is_some());
        assert!(g.tensor("blk.1.ffn_down_exps.weight").is_some());
        assert!(g.tensor("blk.1.attn_q.weight").is_some());
        load_fwd_match(&bytes, 3);
        load_prefill_match(&bytes, &[1, 2, 3]);
        let one = tiny_qwen3moe_gguf();
        let g1 = load_gguf(&one).expect("1layer");
        let m1 = Llama::from_gguf(g1).expect("m1");
        let m2 = Llama::from_gguf(g.clone()).expect("m2");
        let tokens = [1u32, 2, 3];
        let l1 = {
            let mut c = m1.new_cache(8).expect("c1");
            m1.prefill(&mut c, &tokens).expect("p1")
        };
        let l2 = {
            let mut c = m2.new_cache(8).expect("c2");
            m2.prefill(&mut c, &tokens).expect("p2")
        };
        assert_ne!(
            l1, l2,
            "cloned layer 1 must still change logits vs the 1-layer tiny"
        );
        let catalog = m2.expert_direct_store().expect("catalog");
        assert!(
            catalog.keys().any(|k| k.layer == 1),
            "ExpertStore catalog must include L+1 keys"
        );
        let blob = l2.clone();
        let via_direct = store_prefill(&m2, LiveStore::Direct(catalog.clone()), &tokens);
        assert_eq!(blob, via_direct, "2-layer DirectStore must match the blob");
        let n = catalog.len().max(1);
        let via_cached = store_prefill(
            &m2,
            LiveStore::Cached(expertvm::CachedStore::new(catalog, n).expect("cached")),
            &tokens,
        );
        assert_eq!(blob, via_cached, "2-layer CachedStore must match the blob");
        let n_gpu = m2.expert_direct_store().expect("c4").len().max(1);
        let gpu = expertvm::SimulatedGpuStore::new(
            m2.expert_direct_store().expect("c4"),
            n_gpu,
            expertvm::HardwareProfile::example_h100_sxm(),
            4096,
        )
        .expect("gpu");
        let via_gpu = store_prefill(&m2, LiveStore::simulated(gpu), &tokens);
        assert_eq!(
            blob, via_gpu,
            "2-layer SimulatedGpuStore must match the blob"
        );
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let (text, trace) = greedy_generate_traced(&m2, &tok, "ab", 4, None, 0).expect("traced");
        assert!(!text.is_empty());
        assert!(
            trace.events.iter().any(|e| e.layer == 0),
            "2-layer traces must include layer 0"
        );
        assert!(
            trace.events.iter().any(|e| e.layer == 1),
            "2-layer traces must include layer 1"
        );
        let p1 = moe_cached_prefetches(one, &tokens);
        let p2 = moe_cached_prefetches(bytes, &tokens);
        assert!(
            p2 > p1,
            "copy-forward L+1 must prefetch on 2-layer (1-layer skips unknown keys), 1={p1} 2={p2}"
        );
    }

    #[test]
    fn dense_llama_trace_is_empty() {
        let bytes = tiny_llama_gguf();
        let g = load_gguf(&bytes).expect("load");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("model");
        let (text, trace) = greedy_generate_traced(&model, &tok, "ab", 2, None, 0).expect("traced");
        assert!(!text.is_empty());
        assert!(trace.events.is_empty());
    }

    fn store_prefill(model: &Llama, store: LiveStore, tokens: &[u32]) -> Vec<f32> {
        let mut c = model.new_cache(8).expect("cache");
        c.attach_expert_store(store);
        model.prefill(&mut c, tokens).expect("store prefill")
    }

    #[test]
    fn tiny_qwen3moe_expert_store_logits_match_blob() {
        let bytes = tiny_qwen3moe_gguf();
        let g = load_gguf(&bytes).expect("load");
        let model = Llama::from_gguf(g).expect("model");
        let tokens = [3u32];
        let blob = {
            let mut c = model.new_cache(8).expect("c");
            model.prefill(&mut c, &tokens).expect("blob")
        };
        let direct = model.expert_direct_store().expect("catalog");
        assert!(!direct.is_empty());
        let via_direct = store_prefill(&model, LiveStore::Direct(direct.clone()), &tokens);
        assert_eq!(
            blob, via_direct,
            "DirectStore GEMV must match the blob path"
        );
        let n = model.expert_direct_store().expect("catalog2").len().max(1);
        let cached = expertvm::CachedStore::new(model.expert_direct_store().expect("c3"), n)
            .expect("cached");
        let via_cached = store_prefill(&model, LiveStore::Cached(cached), &tokens);
        assert_eq!(
            blob, via_cached,
            "CachedStore copies must match DirectStore"
        );
        let gpu = expertvm::SimulatedGpuStore::new(
            model.expert_direct_store().expect("c4"),
            n,
            expertvm::HardwareProfile::example_h100_sxm(),
            4096,
        )
        .expect("gpu");
        let mut c = model.new_cache(8).expect("cg");
        c.attach_expert_store(LiveStore::simulated(gpu));
        let via_gpu = model.prefill(&mut c, &tokens).expect("gpu prefill");
        assert_eq!(blob, via_gpu, "SimulatedGpuStore CPU copies must match");
        let gpu_um = expertvm::SimulatedGpuStore::with_managed(
            model.expert_direct_store().expect("c5"),
            n,
            expertvm::HardwareProfile::example_h100_sxm(),
            4096,
        )
        .expect("gpu um");
        let mut c_um = model.new_cache(8).expect("cgum");
        c_um.attach_expert_store(LiveStore::simulated(gpu_um));
        let via_um = model.prefill(&mut c_um, &tokens).expect("gpu um prefill");
        assert_eq!(
            blob, via_um,
            "managed SimulatedGpuStore CPU copies must match"
        );
        let via_mapped = store_prefill(
            &model,
            LiveStore::simulated(
                expertvm::SimulatedGpuStore::with_mapped(
                    model.expert_direct_store().expect("cmap"),
                    n,
                    expertvm::HardwareProfile::example_h100_sxm(),
                    4096,
                )
                .expect("gpu mapped"),
            ),
            &tokens,
        );
        assert_eq!(
            blob, via_mapped,
            "mapped SimulatedGpuStore CPU copies must match"
        );
        let via_vmm = store_prefill(
            &model,
            LiveStore::simulated(
                expertvm::SimulatedGpuStore::with_vmm(
                    model.expert_direct_store().expect("cvmm"),
                    n,
                    expertvm::HardwareProfile::example_h100_sxm(),
                    4096,
                )
                .expect("gpu vmm"),
            ),
            &tokens,
        );
        assert_eq!(blob, via_vmm, "VMM SimulatedGpuStore CPU copies must match");
        let tier = expertvm::TieredStore::memory(model.expert_direct_store().expect("t"), n)
            .expect("tier");
        let via_tier = store_prefill(&model, LiveStore::tiered(tier), &tokens);
        assert_eq!(blob, via_tier, "TieredStore copies must match DirectStore");
        let one = expertvm::CachedStore::new(model.expert_direct_store().expect("c1"), 1)
            .expect("one slot");
        let via_one = store_prefill(&model, LiveStore::Cached(one), &tokens);
        assert_eq!(
            blob, via_one,
            "slots=1 sequential lease/release must still match the blob"
        );
        let mut store = c.take_expert_store().expect("store");
        let score = store.score().expect("score");
        assert!(score.is_some());
        assert!(c.expert_store_metrics().is_none());
    }

    #[test]
    fn tiny_llama_moe_direct_store_is_identity() {
        let bytes = tiny_llama_moe_gguf();
        let g = load_gguf(&bytes).expect("load");
        let model = Llama::from_gguf(g).expect("model");
        let tokens = [3u32];
        let blob = {
            let mut c = model.new_cache(8).expect("c");
            model.prefill(&mut c, &tokens).expect("blob")
        };
        let via = store_prefill(
            &model,
            LiveStore::Direct(model.expert_direct_store().expect("d")),
            &tokens,
        );
        assert_eq!(blob, via);
    }

    #[test]
    fn tiny_qwen2moe_direct_store_is_identity() {
        let bytes = tiny_qwen2moe_gguf();
        let g = load_gguf(&bytes).expect("load");
        let model = Llama::from_gguf(g).expect("model");
        let tokens = [3u32];
        let blob = {
            let mut c = model.new_cache(8).expect("c");
            model.prefill(&mut c, &tokens).expect("blob")
        };
        let via = store_prefill(
            &model,
            LiveStore::Direct(model.expert_direct_store().expect("d")),
            &tokens,
        );
        assert_eq!(blob, via);
    }

    #[test]
    fn tiny_llama4_direct_store_is_identity() {
        let bytes = tiny_llama4_gguf();
        let g = load_gguf(&bytes).expect("load");
        let model = Llama::from_gguf(g).expect("model");
        let tokens = [3u32];
        let blob = {
            let mut c = model.new_cache(8).expect("c");
            model.prefill(&mut c, &tokens).expect("blob")
        };
        let via = store_prefill(
            &model,
            LiveStore::Direct(model.expert_direct_store().expect("d")),
            &tokens,
        );
        assert_eq!(blob, via);
    }

    #[test]
    fn tiny_qwen3next_direct_store_is_identity() {
        let bytes = tiny_qwen3next_gguf();
        let g = load_gguf(&bytes).expect("load");
        let model = Llama::from_gguf(g).expect("model");
        let tokens = [3u32];
        let blob = {
            let mut c = model.new_cache(8).expect("c");
            model.prefill(&mut c, &tokens).expect("blob")
        };
        let via = store_prefill(
            &model,
            LiveStore::Direct(model.expert_direct_store().expect("d")),
            &tokens,
        );
        assert_eq!(blob, via);
    }

    #[test]
    fn dense_llama_expert_store_is_empty_catalog() {
        let bytes = tiny_llama_gguf();
        let g = load_gguf(&bytes).expect("load");
        let model = Llama::from_gguf(g).expect("model");
        assert!(model.expert_direct_store().expect("d").is_empty());
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
        // Official `ggml_mrope_cache_init` on text positions `[p, p, p, 0]` assigns
        // theta_t / theta_h / theta_w to every rotated lane, and all three equal the
        // token position, so the m-RoPE cache is identical to the NEOX cache. A VL
        // checkpoint on pure text therefore must match its text-only sibling.
        //
        // This previously asserted inequality, which only held because `qwen2` was
        // rotating NORM adjacent lanes while m-RoPE rotated the NEOX `n_dims/2`
        // offset: the assertion was detecting a bug, not m-RoPE. Real m-RoPE lane
        // math is covered by `rope_multi_reduces_to_neox_on_text_and_differs_on_distinct_axes`.
        assert_logits_match(&qwen2vl_pref, &qwen2_pref);
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
        // IMROPE on text positions likewise reduces to NEOX (every sector maps to
        // theta_t / theta_h / theta_w, all equal to the position). The writer-built
        // tiny additionally saturates the attention gate, so gated-Q is a numerical
        // no-op here — see `gated_attn_fixture_saturates_sigmoid`. Both effects mean
        // qwen35 and dense qwen3 must agree on this fixture; the previous inequality
        // assertion only held because `qwen3` was rotating NORM adjacent lanes.
        assert_logits_match(&qwen35_pref, &qwen3_pref);
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
    fn tiny_bloom_load_gemv_gemm_embed_and_greedy() {
        let bytes = tiny_bloom_gguf();
        let g = load_gguf(&bytes).expect("load bloom");
        assert_eq!(
            g.kv("general.architecture"),
            Some(&Kv::String("bloom".into()))
        );
        assert_eq!(g.kv_u32("bloom.block_count"), Some(1));
        assert_eq!(g.kv_u32("bloom.embedding_length"), Some(256));
        assert_eq!(g.kv_u32("bloom.feed_forward_length"), Some(1024));
        assert_eq!(g.kv_u32("bloom.attention.head_count"), Some(4));
        assert_eq!(g.kv_u32("bloom.attention.head_count_kv"), Some(4));
        assert_eq!(g.kv_u32("bloom.context_length"), Some(256));
        assert_eq!(
            g.kv_f32("bloom.attention.layer_norm_epsilon"),
            Some(1.0 / 100_000.0)
        );
        assert!(g.kv_f32("bloom.attention.layer_norm_rms_epsilon").is_none());
        assert!(g.kv_u32("bloom.rope.dimension_count").is_none());
        assert!(g.kv_f32("bloom.rope.freq_base").is_none());
        assert!(g.kv_u32("llama.block_count").is_none());
        assert!(g.kv_u32("phi2.block_count").is_none());
        assert!(g.kv_u32("phi3.block_count").is_none());
        assert!(g.kv_u32("qwen3vlmoe.block_count").is_none());
        assert!(g.kv_u32("mixtral.block_count").is_none());
        assert!(g.tensor("token_embd_norm.weight").is_some());
        assert!(g.tensor("token_embd_norm.bias").is_some());
        assert!(g.tensor("blk.0.attn_norm.bias").is_some());
        assert!(g.tensor("output_norm.bias").is_some());
        assert!(g.tensor("output.bias").is_none());
        assert!(g.tensor("blk.0.attn_qkv.weight").is_some());
        assert!(g.tensor("blk.0.attn_qkv.bias").is_some());
        assert!(g.tensor("blk.0.attn_q.weight").is_none());
        assert!(g.tensor("blk.0.attn_k.weight").is_none());
        assert!(g.tensor("blk.0.attn_v.weight").is_none());
        assert!(g.tensor("blk.0.attn_output.bias").is_some());
        assert!(g.tensor("blk.0.ffn_norm.weight").is_some());
        assert!(g.tensor("blk.0.ffn_norm.bias").is_some());
        assert!(g.tensor("blk.0.ffn_up.bias").is_some());
        assert!(g.tensor("blk.0.ffn_down.bias").is_some());
        assert!(g.tensor("blk.0.ffn_gate.weight").is_none());
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
        let phi2_pref = {
            let p = load_gguf(&tiny_phi2_gguf()).expect("phi2");
            let m = Llama::from_gguf(p).expect("mp2");
            let mut c = m.new_cache(8).expect("cp2");
            m.prefill(&mut c, &tokens).expect("phi2 pref")
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
        let bloom_pref = model.prefill(&mut qc, &tokens).expect("bloom pref");
        assert_ne!(
            bloom_pref, llama_pref,
            "bloom ALiBi/tok_norm/GELU-seq must change logits vs llama"
        );
        assert_ne!(
            bloom_pref, phi2_pref,
            "bloom must not copy phi2 (parallel residual + RoPE)"
        );
        assert_ne!(
            bloom_pref, phi3_pref,
            "bloom must not copy phi3 (RMSNorm + SwiGLU)"
        );
        assert_ne!(
            bloom_pref, gemma_pref,
            "bloom must not copy gemma (embed-scale + GeGLU)"
        );
    }

    #[test]
    fn bloom_missing_layer_norm_epsilon_names_key() {
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("bloom".into())),
                ("bloom.block_count".into(), Kv::U32(1)),
                ("bloom.embedding_length".into(), Kv::U32(256)),
                ("bloom.feed_forward_length".into(), Kv::U32(1024)),
                ("bloom.attention.head_count".into(), Kv::U32(4)),
                ("bloom.attention.head_count_kv".into(), Kv::U32(4)),
            ],
            &[],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected missing layer_norm_epsilon"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("bloom.attention.layer_norm_epsilon"),
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
    fn gemma4_missing_layer_norm_rms_epsilon_names_key() {
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("gemma4".into())),
                ("gemma4.block_count".into(), Kv::U32(1)),
                ("gemma4.embedding_length".into(), Kv::U32(256)),
                ("gemma4.attention.head_count".into(), Kv::U32(4)),
                ("gemma4.attention.head_count_kv".into(), Kv::U32(2)),
            ],
            &[],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected missing layer_norm_rms_epsilon"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("gemma4.attention.layer_norm_rms_epsilon"),
            "error should name kv key: {err}"
        );
    }

    #[test]
    fn gemma4_missing_sliding_window_pattern_names_key() {
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("gemma4".into())),
                ("gemma4.block_count".into(), Kv::U32(1)),
                ("gemma4.embedding_length".into(), Kv::U32(256)),
                ("gemma4.attention.head_count".into(), Kv::U32(4)),
                ("gemma4.attention.head_count_kv".into(), Kv::U32(2)),
                (
                    "gemma4.attention.layer_norm_rms_epsilon".into(),
                    Kv::F32(1.0 / 100_000.0),
                ),
                ("gemma4.attention.sliding_window".into(), Kv::U32(2)),
            ],
            &[],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected missing sliding_window_pattern"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("gemma4.attention.sliding_window_pattern"),
            "error should name kv key: {err}"
        );
    }

    #[test]
    fn gemma4_ple_without_token_embd_names_tensor() {
        let n_embd = TINY_N_EMBD;
        let ones = vec![1.0f32; n_embd];
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("gemma4".into())),
                ("gemma4.block_count".into(), Kv::U32(1)),
                ("gemma4.embedding_length".into(), Kv::U32(256)),
                ("gemma4.attention.head_count".into(), Kv::U32(4)),
                ("gemma4.attention.head_count_kv".into(), Kv::U32(2)),
                (
                    "gemma4.attention.layer_norm_rms_epsilon".into(),
                    Kv::F32(1.0 / 100_000.0),
                ),
                ("gemma4.attention.sliding_window".into(), Kv::U32(2)),
                (
                    "gemma4.attention.sliding_window_pattern".into(),
                    Kv::Array {
                        elem: GGUF_TYPE_BOOL,
                        items: vec![Kv::Bool(true)],
                    },
                ),
                (
                    "gemma4.embedding_length_per_layer_input".into(),
                    Kv::U32(32),
                ),
                ("gemma4.attention.key_length_swa".into(), Kv::U32(64)),
                ("gemma4.attention.value_length_swa".into(), Kv::U32(64)),
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
            Ok(_) => panic!("expected missing per_layer_token_embd"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("per_layer_token_embd"),
            "error should name tensor: {err}"
        );
    }

    #[test]
    fn gemma4_shared_kv_too_small_is_refused() {
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("gemma4".into())),
                ("gemma4.block_count".into(), Kv::U32(1)),
                ("gemma4.embedding_length".into(), Kv::U32(256)),
                ("gemma4.attention.head_count".into(), Kv::U32(4)),
                ("gemma4.attention.head_count_kv".into(), Kv::U32(2)),
                (
                    "gemma4.attention.layer_norm_rms_epsilon".into(),
                    Kv::F32(1.0 / 100_000.0),
                ),
                ("gemma4.attention.sliding_window".into(), Kv::U32(2)),
                (
                    "gemma4.attention.sliding_window_pattern".into(),
                    Kv::Array {
                        elem: GGUF_TYPE_BOOL,
                        items: vec![Kv::Bool(true)],
                    },
                ),
                ("gemma4.embedding_length_per_layer_input".into(), Kv::U32(0)),
                ("gemma4.attention.shared_kv_layers".into(), Kv::U32(1)),
                ("gemma4.attention.key_length_swa".into(), Kv::U32(64)),
                ("gemma4.attention.value_length_swa".into(), Kv::U32(64)),
            ],
            &[],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected shared KV refuse"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("gemma4.attention.shared_kv_layers"),
            "error should name kv key: {err}"
        );
    }

    #[test]
    fn gemma4_mixed_swa_head_dim_is_refused() {
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("gemma4".into())),
                ("gemma4.block_count".into(), Kv::U32(1)),
                ("gemma4.embedding_length".into(), Kv::U32(256)),
                ("gemma4.attention.head_count".into(), Kv::U32(4)),
                ("gemma4.attention.head_count_kv".into(), Kv::U32(2)),
                (
                    "gemma4.attention.layer_norm_rms_epsilon".into(),
                    Kv::F32(1.0 / 100_000.0),
                ),
                ("gemma4.attention.sliding_window".into(), Kv::U32(2)),
                (
                    "gemma4.attention.sliding_window_pattern".into(),
                    Kv::Array {
                        elem: GGUF_TYPE_BOOL,
                        items: vec![Kv::Bool(true)],
                    },
                ),
                ("gemma4.embedding_length_per_layer_input".into(), Kv::U32(0)),
                ("gemma4.attention.key_length_swa".into(), Kv::U32(32)),
                ("gemma4.attention.value_length_swa".into(), Kv::U32(64)),
            ],
            &[],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected mixed SWA head dim refuse"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("gemma4.attention.key_length_swa"),
            "error should name kv key: {err}"
        );
    }

    #[test]
    fn gemma4_missing_sliding_window_names_key() {
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("gemma4".into())),
                ("gemma4.block_count".into(), Kv::U32(1)),
                ("gemma4.embedding_length".into(), Kv::U32(256)),
                ("gemma4.attention.head_count".into(), Kv::U32(4)),
                ("gemma4.attention.head_count_kv".into(), Kv::U32(2)),
                (
                    "gemma4.attention.layer_norm_rms_epsilon".into(),
                    Kv::F32(1.0 / 100_000.0),
                ),
            ],
            &[],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected missing sliding_window"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("gemma4.attention.sliding_window"),
            "error should name kv key: {err}"
        );
    }

    #[test]
    fn gemma4_missing_key_length_swa_names_key() {
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("gemma4".into())),
                ("gemma4.block_count".into(), Kv::U32(1)),
                ("gemma4.embedding_length".into(), Kv::U32(256)),
                ("gemma4.attention.head_count".into(), Kv::U32(4)),
                ("gemma4.attention.head_count_kv".into(), Kv::U32(2)),
                (
                    "gemma4.attention.layer_norm_rms_epsilon".into(),
                    Kv::F32(1.0 / 100_000.0),
                ),
                ("gemma4.attention.sliding_window".into(), Kv::U32(2)),
                (
                    "gemma4.attention.sliding_window_pattern".into(),
                    Kv::Array {
                        elem: GGUF_TYPE_BOOL,
                        items: vec![Kv::Bool(true)],
                    },
                ),
                ("gemma4.embedding_length_per_layer_input".into(), Kv::U32(0)),
            ],
            &[],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected missing key_length_swa"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("gemma4.attention.key_length_swa"),
            "error should name kv key: {err}"
        );
    }

    #[test]
    fn gemma4_moe_without_expert_count_names_key() {
        let n_embd = TINY_N_EMBD;
        let ones = vec![1.0f32; n_embd];
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("gemma4".into())),
                ("gemma4.block_count".into(), Kv::U32(1)),
                ("gemma4.embedding_length".into(), Kv::U32(256)),
                ("gemma4.attention.head_count".into(), Kv::U32(4)),
                ("gemma4.attention.head_count_kv".into(), Kv::U32(2)),
                (
                    "gemma4.attention.layer_norm_rms_epsilon".into(),
                    Kv::F32(1.0 / 100_000.0),
                ),
                ("gemma4.attention.sliding_window".into(), Kv::U32(2)),
                (
                    "gemma4.attention.sliding_window_pattern".into(),
                    Kv::Array {
                        elem: GGUF_TYPE_BOOL,
                        items: vec![Kv::Bool(true)],
                    },
                ),
                ("gemma4.embedding_length_per_layer_input".into(), Kv::U32(0)),
                ("gemma4.attention.key_length_swa".into(), Kv::U32(64)),
                ("gemma4.attention.value_length_swa".into(), Kv::U32(64)),
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
                tw(
                    "blk.0.ffn_gate_inp.weight",
                    GgmlType::F32,
                    vec![n_embd, 1],
                    pack_mat(GgmlType::F32, n_embd, 1, 7),
                ),
            ],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected missing expert_count"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("gemma4.expert_count"),
            "error should name kv key: {err}"
        );
    }

    #[test]
    fn gemma4_fused_gate_up_without_layer_still_fails_load() {
        let n_embd = TINY_N_EMBD;
        let ones = vec![1.0f32; n_embd];
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("gemma4".into())),
                ("gemma4.block_count".into(), Kv::U32(1)),
                ("gemma4.embedding_length".into(), Kv::U32(256)),
                ("gemma4.feed_forward_length".into(), Kv::U32(256)),
                ("gemma4.attention.head_count".into(), Kv::U32(4)),
                ("gemma4.attention.head_count_kv".into(), Kv::U32(2)),
                (
                    "gemma4.attention.layer_norm_rms_epsilon".into(),
                    Kv::F32(1.0 / 100_000.0),
                ),
                ("gemma4.attention.sliding_window".into(), Kv::U32(2)),
                (
                    "gemma4.attention.sliding_window_pattern".into(),
                    Kv::Array {
                        elem: GGUF_TYPE_BOOL,
                        items: vec![Kv::Bool(true)],
                    },
                ),
                ("gemma4.embedding_length_per_layer_input".into(), Kv::U32(0)),
                ("gemma4.attention.key_length_swa".into(), Kv::U32(64)),
                ("gemma4.attention.value_length_swa".into(), Kv::U32(64)),
                ("gemma4.expert_count".into(), Kv::U32(4)),
                ("gemma4.expert_used_count".into(), Kv::U32(2)),
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
                tw(
                    "blk.0.ffn_gate_inp.weight",
                    GgmlType::F32,
                    vec![n_embd, 4],
                    pack_mat(GgmlType::F32, n_embd, 4, 7),
                ),
                tw(
                    "blk.0.ffn_gate_up_exps.weight",
                    GgmlType::F32,
                    vec![n_embd, 512, 4],
                    pack_mat_exps(GgmlType::F32, n_embd, 512, 4, 8),
                ),
            ],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected missing layer tensors"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("blk.0.ffn_gate.weight"),
            "incomplete fused GGUF must still name the missing shared MLP: {err}"
        );
    }

    #[test]
    fn gemma2_missing_attn_logit_softcapping_names_key() {
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("gemma2".into())),
                ("gemma2.block_count".into(), Kv::U32(1)),
                ("gemma2.embedding_length".into(), Kv::U32(256)),
                ("gemma2.attention.head_count".into(), Kv::U32(4)),
                ("gemma2.attention.head_count_kv".into(), Kv::U32(2)),
                (
                    "gemma2.attention.layer_norm_rms_epsilon".into(),
                    Kv::F32(1.0 / 100_000.0),
                ),
            ],
            &[],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected missing attn_logit_softcapping"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("gemma2.attn_logit_softcapping"),
            "error should name kv key: {err}"
        );
    }

    #[test]
    fn gemma3_missing_layer_norm_rms_epsilon_names_key() {
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("gemma3".into())),
                ("gemma3.block_count".into(), Kv::U32(1)),
                ("gemma3.embedding_length".into(), Kv::U32(256)),
                ("gemma3.attention.head_count".into(), Kv::U32(4)),
                ("gemma3.attention.head_count_kv".into(), Kv::U32(2)),
            ],
            &[],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected missing layer_norm_rms_epsilon"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("gemma3.attention.layer_norm_rms_epsilon"),
            "error should name kv key: {err}"
        );
    }

    #[test]
    fn gemma3n_missing_layer_norm_rms_epsilon_names_key() {
        let bytes = write_gguf_with_kv(
            &[
                ("general.alignment".into(), Kv::U32(32)),
                ("general.architecture".into(), Kv::String("gemma3n".into())),
                ("gemma3n.block_count".into(), Kv::U32(1)),
                ("gemma3n.embedding_length".into(), Kv::U32(256)),
                ("gemma3n.attention.head_count".into(), Kv::U32(4)),
                ("gemma3n.attention.head_count_kv".into(), Kv::U32(2)),
            ],
            &[],
        );
        let g = load_gguf(&bytes).expect("load");
        let err = match Llama::from_gguf(g) {
            Ok(_) => panic!("expected missing layer_norm_rms_epsilon"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("gemma3n.attention.layer_norm_rms_epsilon"),
            "error should name kv key: {err}"
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
                gemma4_moe: false,
                gemma4_ple: false,
                gemma4_fused: false,
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

/// Self-contained timing harness. Every test here is `#[ignore]`d so the normal
/// suite stays fast; run it with
/// `cargo test --release --lib bench_ -- --ignored --nocapture`.
///
/// The tiny oracle fixtures are far too small to time (256-wide, 1 layer,
/// 6-token vocab), so this module writes a larger synthetic Llama-shaped GGUF
/// with the Q4_K/Q6_K mix a real `*-Q4_K_M.gguf` uses.
#[cfg(test)]
mod bench {
    use super::*;
    use crate::load_gguf_owned;
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    /// Dimensions of the synthetic benchmark model.
    struct BenchSpec {
        n_embd: usize,
        n_head: usize,
        n_head_kv: usize,
        n_ff: usize,
        n_layer: usize,
        n_vocab: usize,
    }

    impl BenchSpec {
        /// ~40 MiB of weights: wide enough that GEMV work dominates per-row
        /// bookkeeping, small enough to build and run inside a test.
        fn default() -> Self {
            Self {
                n_embd: 1024,
                n_head: 16,
                n_head_kv: 4,
                n_ff: 2816,
                n_layer: 4,
                n_vocab: 4096,
            }
        }

        fn head_dim(&self) -> usize {
            self.n_embd / self.n_head
        }

        fn n_kv(&self) -> usize {
            self.n_head_kv * self.head_dim()
        }
    }

    /// `pack_mat` emits exactly one block per row (the tiny fixtures are all
    /// `n_cols == QK_K`). The benchmark needs wide rows, so pack
    /// `n_cols / QK_K` blocks per row here.
    fn bench_pack_mat(ty: GgmlType, n_cols: usize, n_rows: usize, seed: u32) -> Vec<u8> {
        assert!(n_cols.is_multiple_of(QK_K));
        let mut out = Vec::new();
        let mut s = seed;
        let mut next = move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1);
            s
        };
        for _ in 0..n_rows.saturating_mul(n_cols / QK_K) {
            match ty {
                GgmlType::Q4_K => {
                    // `y = d*sc*q - dmin*m`. Pick `q` in 0..=8 and `m = 4*sc`
                    // so rows are zero-mean; all-positive weights saturate a
                    // 4-layer stack and every logit comes out equal.
                    let mut qs = [0u8; QK_K];
                    for q in &mut qs {
                        *q = u8::try_from(next() % 9).unwrap();
                    }
                    let mut sc = [1u8; 8];
                    let mut mins = [0u8; 8];
                    for (c, m) in sc.iter_mut().zip(mins.iter_mut()) {
                        *c = u8::try_from(1 + next() % 4).unwrap();
                        *m = c.saturating_mul(4);
                    }
                    out.extend_from_slice(&pack_q4_k_block(
                        25.0 / 100.0,
                        25.0 / 100.0,
                        &sc,
                        &mins,
                        &qs,
                    ));
                }
                GgmlType::Q6_K => {
                    let mut qs = [0i8; QK_K];
                    for q in &mut qs {
                        *q = i8::try_from(i32::try_from(next() % 5).unwrap() - 2).unwrap();
                    }
                    let mut sc = [1i8; 16];
                    for c in &mut sc {
                        *c = i8::try_from(1 + next() % 3).unwrap();
                    }
                    out.extend_from_slice(&pack_q6_k_block(25.0 / 100.0, &sc, &qs));
                }
                other => panic!("bench_pack_mat: unhandled type {other:?}"),
            }
        }
        out
    }

    fn bench_gguf(s: &BenchSpec) -> Vec<u8> {
        let arch = "llama";
        let u32v = |v: usize| Kv::U32(u32::try_from(v).unwrap());
        let kv = vec![
            ("general.alignment".into(), u32v(GGUF_DEFAULT_ALIGNMENT)),
            ("general.architecture".into(), Kv::String(arch.into())),
            (arch_key(arch, "block_count"), u32v(s.n_layer)),
            (arch_key(arch, "embedding_length"), u32v(s.n_embd)),
            (arch_key(arch, "feed_forward_length"), u32v(s.n_ff)),
            (arch_key(arch, "attention.head_count"), u32v(s.n_head)),
            (arch_key(arch, "attention.head_count_kv"), u32v(s.n_head_kv)),
            (arch_key(arch, "rope.dimension_count"), u32v(s.head_dim())),
            (arch_key(arch, "rope.freq_base"), Kv::F32(10_000.0)),
            (
                arch_key(arch, "attention.layer_norm_rms_epsilon"),
                Kv::F32(1.0 / 100_000.0),
            ),
        ];
        let ones = vec![1.0f32; s.n_embd];
        let mut tensors = vec![
            tw(
                "token_embd.weight",
                GgmlType::Q4_K,
                vec![s.n_embd, s.n_vocab],
                bench_pack_mat(GgmlType::Q4_K, s.n_embd, s.n_vocab, 1),
            ),
            tw(
                "output.weight",
                GgmlType::Q6_K,
                vec![s.n_embd, s.n_vocab],
                bench_pack_mat(GgmlType::Q6_K, s.n_embd, s.n_vocab, 2),
            ),
            tw(
                "output_norm.weight",
                GgmlType::F32,
                vec![s.n_embd],
                pack_f32(&ones),
            ),
        ];
        for li in 0..s.n_layer {
            let seed = u32::try_from(li).unwrap() * 16 + 3;
            let mat = |name: &str, ty: GgmlType, n_cols: usize, n_rows: usize, bump: u32| {
                tw(
                    &format!("blk.{li}.{name}.weight"),
                    ty,
                    vec![n_cols, n_rows],
                    bench_pack_mat(ty, n_cols, n_rows, seed + bump),
                )
            };
            tensors.push(tw(
                &format!("blk.{li}.attn_norm.weight"),
                GgmlType::F32,
                vec![s.n_embd],
                pack_f32(&ones),
            ));
            tensors.push(tw(
                &format!("blk.{li}.ffn_norm.weight"),
                GgmlType::F32,
                vec![s.n_embd],
                pack_f32(&ones),
            ));
            tensors.push(mat("attn_q", GgmlType::Q4_K, s.n_embd, s.n_embd, 0));
            tensors.push(mat("attn_k", GgmlType::Q4_K, s.n_embd, s.n_kv(), 1));
            tensors.push(mat("attn_v", GgmlType::Q4_K, s.n_embd, s.n_kv(), 2));
            tensors.push(mat("attn_output", GgmlType::Q4_K, s.n_embd, s.n_embd, 3));
            tensors.push(mat("ffn_gate", GgmlType::Q4_K, s.n_embd, s.n_ff, 4));
            tensors.push(mat("ffn_up", GgmlType::Q4_K, s.n_embd, s.n_ff, 5));
            tensors.push(mat("ffn_down", GgmlType::Q6_K, s.n_ff, s.n_embd, 6));
        }
        write_gguf_with_kv(&kv, &tensors)
    }

    fn bench_model(s: &BenchSpec) -> Llama {
        let g = load_gguf_owned(bench_gguf(s)).expect("load bench gguf");
        Llama::from_gguf(g).expect("bench model")
    }

    fn median(samples: &mut [Duration]) -> Duration {
        samples.sort_unstable();
        samples[samples.len() / 2]
    }

    fn report(label: &str, mut samples: Vec<Duration>) {
        let best = *samples.iter().min().unwrap();
        let med = median(&mut samples);
        println!(
            "{label:<44} min {:>10.3} ms   median {:>10.3} ms   n={}",
            best.as_secs_f64() * 1e3,
            med.as_secs_f64() * 1e3,
            samples.len()
        );
    }

    /// FNV-1a over the raw bit patterns of a logits vector.
    fn fingerprint(logits: &[f32]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for v in logits {
            for b in v.to_bits().to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        }
        h
    }

    /// Prints bit-exact fingerprints of production logits for every arch and a
    /// spread of dtypes. Not an assertion: it is the tool for checking that a
    /// refactor is bit-identical, by diffing this output across two checkouts.
    /// The oracle tests only bound the relative error at 1e-3.
    #[test]
    #[ignore = "diagnostic: diff across checkouts"]
    fn bench_logits_fingerprint() {
        // Every zero-argument fixture in the suite, so that each architecture
        // walk and each dtype kernel the decode path can reach is pinned.
        let cases: [(&str, Vec<u8>); 59] = [
            ("llama", tiny_llama_gguf()),
            ("tied", tiny_tied_gguf()),
            ("tied_copy", tiny_tied_copy_gguf()),
            ("mistral", tiny_mistral_gguf()),
            ("mha", tiny_mha_gguf()),
            ("mqa", tiny_mqa_gguf()),
            ("llama_moe", tiny_llama_moe_gguf()),
            ("llama4", tiny_llama4_gguf()),
            ("qwen2", tiny_qwen2_gguf()),
            ("qwen2moe", tiny_qwen2moe_gguf()),
            ("qwen2vl", tiny_qwen2vl_gguf()),
            ("qwen3", tiny_qwen3_gguf()),
            ("qwen3moe", tiny_qwen3moe_gguf()),
            ("qwen3next", tiny_qwen3next_gguf()),
            ("qwen3vl", tiny_qwen3vl_gguf()),
            ("qwen35", tiny_qwen35_gguf()),
            ("phi2", tiny_phi2_gguf()),
            ("bloom", tiny_bloom_gguf()),
            ("phi3", tiny_phi3_gguf()),
            ("gemma", tiny_gemma_gguf()),
            ("gemma2", tiny_gemma2_gguf()),
            ("gemma3", tiny_gemma3_gguf()),
            ("gemma3n", tiny_gemma3n_gguf()),
            ("gemma4", tiny_gemma4_gguf()),
            ("gemma4_moe", tiny_gemma4_moe_gguf()),
            ("gemma4_ple", tiny_gemma4_ple_gguf()),
            ("gemma4_moe_ple", tiny_gemma4_moe_ple_gguf()),
            ("gemma4_moe_fused", tiny_gemma4_moe_fused_gguf()),
            ("gemma4_moe_fused_ple", tiny_gemma4_moe_fused_ple_gguf()),
            ("gemma4_shared_kv", tiny_gemma4_shared_kv_gguf()),
            ("f16", tiny_f16_gguf()),
            ("f16_1d", tiny_f16_1d_gguf()),
            ("f16_1d_bias", tiny_f16_1d_bias_gguf()),
            ("bf16", tiny_bf16_gguf()),
            ("q10", tiny_q10_gguf()),
            ("q20", tiny_q20_gguf()),
            ("q41", tiny_q41_gguf()),
            ("q50", tiny_q50_gguf()),
            ("q51", tiny_q51_gguf()),
            ("q80", tiny_q80_gguf()),
            ("q81", tiny_q81_gguf()),
            ("q2k", tiny_q2k_gguf()),
            ("q3k", tiny_q3k_gguf()),
            ("q5k", tiny_q5k_gguf()),
            ("q4k_embd", tiny_q4k_embd_gguf()),
            ("q6k_embd", tiny_q6k_embd_gguf()),
            ("tq10", tiny_tq10_gguf()),
            ("tq20", tiny_tq20_gguf()),
            ("mxfp4", tiny_mxfp4_gguf()),
            ("nvfp4", tiny_nvfp4_gguf()),
            ("iq1s", tiny_iq1s_gguf()),
            ("iq1m", tiny_iq1m_gguf()),
            ("iq2xxs", tiny_iq2xxs_gguf()),
            ("iq2xs", tiny_iq2xs_gguf()),
            ("iq2s", tiny_iq2s_gguf()),
            ("iq3xxs", tiny_iq3xxs_gguf()),
            ("iq3s", tiny_iq3s_gguf()),
            ("iq4nl", tiny_iq4nl_gguf()),
            ("iq4xs", tiny_iq4xs_gguf()),
        ];
        for (name, bytes) in cases {
            let model = Llama::from_gguf(load_gguf_owned(bytes).expect("load")).expect("model");
            let mut cache = model.new_cache(8).expect("cache");
            let one = model.prefill(&mut cache, &[1]).expect("prefill 1");
            let two = model.forward(&mut cache, 3).expect("forward");
            let mut cache = model.new_cache(8).expect("cache");
            let seq = model.prefill(&mut cache, &[1, 3, 2]).expect("prefill 3");
            println!(
                "{name:<12} fwd1 {:016x} fwd2 {:016x} pre3 {:016x}",
                fingerprint(&one),
                fingerprint(&two),
                fingerprint(&seq)
            );
        }
        let spec = BenchSpec::default();
        let model = bench_model(&spec);
        let mut cache = model.new_cache(16).expect("cache");
        let one = model.prefill(&mut cache, &[1]).expect("prefill 1");
        let two = model.forward(&mut cache, 7).expect("forward");
        let mut cache = model.new_cache(16).expect("cache");
        let seq = model.prefill(&mut cache, &[1, 7, 2, 9]).expect("prefill 4");
        println!(
            "{:<12} fwd1 {:016x} fwd2 {:016x} pre4 {:016x}",
            "bench1024",
            fingerprint(&one),
            fingerprint(&two),
            fingerprint(&seq)
        );
        // A degenerate synthetic model would make every fingerprint match for
        // uninteresting reasons, so check the logits actually vary and are finite.
        for l in [&one, &two, &seq] {
            let lo = l.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!(lo.is_finite() && hi.is_finite() && hi > lo, "{lo} {hi}");
        }
        assert_ne!(fingerprint(&one), fingerprint(&two));
    }

    /// `steps` timed decode steps from one cache, after `warmup` untimed ones.
    ///
    /// The warmup matters more than it looks. The weights are 32 MiB and this
    /// class of host has hundreds of MiB of L3, so the first timed loop in a
    /// process streams the blob from RAM while every later loop finds it in
    /// cache. At four warmup steps that alone was worth ~70%, i.e. more than
    /// anything measured here, and it landed entirely on whichever variant ran
    /// first.
    ///
    /// The cache is sized so no run has to rebuild it mid-loop. `n_past` climbs
    /// as the loop goes, but attention over the past is under 1% of a step at
    /// these dimensions, so the samples stay comparable.
    fn decode_samples(model: &Llama, vocab: u32, warmup: usize, steps: usize) -> Vec<Duration> {
        let total = warmup.saturating_add(steps);
        let mut cache = model.new_cache(total.saturating_add(2)).expect("cache");
        let mut last = model.prefill(&mut cache, &[1]).expect("prefill");
        let mut samples = Vec::new();
        for i in 0..total {
            let tok = argmax(&last) % vocab;
            let t0 = Instant::now();
            last = model.forward(&mut cache, tok).expect("forward");
            let dt = t0.elapsed();
            if i >= warmup {
                samples.push(dt);
            }
            assert!(!black_box(&last).is_empty());
        }
        samples
    }

    /// [`decode_samples`] through the borrowed-logits entry point.
    fn decode_borrowed_samples(
        model: &Llama,
        vocab: u32,
        warmup: usize,
        steps: usize,
    ) -> Vec<Duration> {
        let total = warmup.saturating_add(steps);
        let mut cache = model.new_cache(total.saturating_add(2)).expect("cache");
        let mut tok = argmax(&model.prefill(&mut cache, &[1]).expect("prefill")) % vocab;
        let mut samples = Vec::new();
        for i in 0..total {
            let t0 = Instant::now();
            let logits = model.forward_logits(&mut cache, tok).expect("forward");
            let dt = t0.elapsed();
            if i >= warmup {
                samples.push(dt);
            }
            tok = argmax(logits) % vocab;
            assert!(!black_box(logits).is_empty());
        }
        samples
    }

    /// Single-token decode latency.
    ///
    /// `forward` is the entry point the pre-refactor checkout has too, so it is
    /// the number to compare across checkouts. `no pool` runs the same step
    /// with every GEMV back on `thread::scope`, which isolates the pool inside
    /// one process and is the only pool comparison not exposed to the ~25%
    /// run-to-run spread a shared host shows across two binaries.
    /// `forward_logits` drops the vocab-sized copy. The 1-thread run is immune
    /// to thread scheduling noise, so it is the honest view of what buffer
    /// reuse alone bought.
    #[test]
    #[ignore = "timing harness"]
    fn bench_decode_step() {
        let spec = BenchSpec::default();
        let model = bench_model(&spec);
        println!("blob {} MiB", model.blob_len() / (1024 * 1024));
        let vocab = u32::try_from(spec.n_vocab).unwrap();
        let (mut par, mut nopool) = (Vec::new(), Vec::new());
        let (mut borrowed, mut seq) = (Vec::new(), Vec::new());
        // Interleaved, so clock drift or a noisy neighbour hits every variant
        // instead of only the one that ran first.
        for _ in 0..3 {
            par.extend(decode_samples(&model, vocab, 8, 16));
            nopool.extend(without_pool(|| decode_samples(&model, vocab, 8, 16)));
            borrowed.extend(decode_borrowed_samples(&model, vocab, 8, 16));
            seq.extend(crate::pool::with_sequential(|| {
                decode_samples(&model, vocab, 4, 8)
            }));
        }
        report("decode 1 token (forward)", par);
        report("decode 1 token (forward, no pool)", nopool);
        report("decode 1 token (forward_logits)", borrowed);
        report("decode 1 token (forward, 1 thread)", seq);
    }

    /// `reps` timed prefills of `tokens`, after `warmup` untimed ones. Each rep
    /// gets a fresh cache, so this is the cost a real prompt pays.
    fn prefill_samples(model: &Llama, tokens: &[u32], warmup: usize, reps: usize) -> Vec<Duration> {
        let mut samples = Vec::new();
        for i in 0..warmup.saturating_add(reps) {
            let mut cache = model
                .new_cache(tokens.len().saturating_add(1))
                .expect("cache");
            let t0 = Instant::now();
            let out = model.prefill(&mut cache, tokens).expect("prefill");
            let dt = t0.elapsed();
            if i >= warmup {
                samples.push(dt);
            }
            assert!(!black_box(&out).is_empty());
        }
        samples
    }

    /// Multi-token prefill (`prefill`, GEMM path).
    #[test]
    #[ignore = "timing harness"]
    fn bench_prefill() {
        let spec = BenchSpec::default();
        let model = bench_model(&spec);
        let sizes = [8usize, 32, 64];
        let mut all: Vec<Vec<Duration>> = sizes.iter().map(|_| Vec::new()).collect();
        for _ in 0..3 {
            for (n, samples) in sizes.iter().zip(all.iter_mut()) {
                let tokens: Vec<u32> = (0..*n)
                    .map(|i| u32::try_from(i % spec.n_vocab).unwrap())
                    .collect();
                samples.extend(prefill_samples(&model, &tokens, 1, 3));
            }
        }
        for (n, samples) in sizes.iter().zip(all) {
            report(&format!("prefill {n} tokens"), samples);
        }
    }

    /// What is left to win, and where. Times one decode step's worth of GEMV
    /// against the whole step, both single-threaded, plus the two per-token
    /// non-matmul loops that looked worth suspecting.
    ///
    /// This is the measurement that says when to stop: if the GEMV kernels are
    /// the step, then no amount of work in `decode.rs` moves the number and the
    /// next win has to come from the kernels or from more cores.
    #[test]
    #[ignore = "timing harness"]
    fn bench_step_breakdown() {
        let spec = BenchSpec::default();
        let model = bench_model(&spec);
        let vocab = u32::try_from(spec.n_vocab).unwrap();
        let x = vec![0.01f32; spec.n_embd];
        let xff = vec![0.01f32; spec.n_ff];

        // Every GEMV a decode step performs, in the order the step does them.
        let mut mats: Vec<(&QuantMat, &[f32])> = Vec::new();
        for layer in &model.layers {
            mats.push((&layer.wq, &x));
            if let Some(wk) = layer.wk.as_ref() {
                mats.push((wk, &x));
            }
            if let Some(wv) = layer.wv.as_ref() {
                mats.push((wv, &x));
            }
            mats.push((&layer.wo, &x));
            if let LayerFfn::Dense(dense) = &layer.ffn {
                mats.push((&dense.gate, &x));
                mats.push((&dense.up, &x));
                mats.push((&dense.down, &xff));
            }
        }
        mats.push((&model.output, &x));
        let n_gemv = mats.len();

        let mut y = Vec::new();
        let mut samples = Vec::new();
        for _ in 0..9 {
            let t0 = Instant::now();
            crate::pool::with_sequential(|| {
                for (m, xv) in &mats {
                    model.gemv_into(m, xv, &mut y, &mut None).expect("gemv");
                }
            });
            samples.push(t0.elapsed());
            assert!(!black_box(&y).is_empty());
        }
        report(&format!("{n_gemv} GEMV only, 1 thread"), samples);

        let rope_calls = spec
            .n_layer
            .saturating_mul(spec.n_head.saturating_add(spec.n_head_kv));
        let mut v = vec![0.5f32; spec.head_dim()];
        let mut samples = Vec::new();
        for _ in 0..9 {
            let t0 = Instant::now();
            for i in 0..rope_calls {
                rope(&mut v, 17 + (i & 7), spec.head_dim(), 10_000.0).expect("rope");
            }
            samples.push(t0.elapsed());
        }
        assert!(black_box(v.iter().sum::<f32>()).is_finite());
        report(&format!("rope x{rope_calls}, 1 thread"), samples);

        let seq = crate::pool::with_sequential(|| decode_samples(&model, vocab, 4, 8));
        report("whole step, 1 thread", seq);
    }

    /// Writes `y[i] = first + i` and nothing else, so a [`Pool::run`] sample is
    /// all dispatch and no arithmetic — the same shape as the empty-body
    /// closure the `thread::scope` measurement uses.
    struct NoopRows;

    impl RowKernel for NoopRows {
        type Job = ();

        fn rows(&self, _job: (), first: usize, _x: &[f32], y: &mut [f32]) -> bool {
            for (i, out) in y.iter_mut().enumerate() {
                *out = black_box(first.saturating_add(i) as f32);
            }
            true
        }
    }

    /// Row-dispatch cost in isolation: the row body does nothing but a store,
    /// so the whole sample is fork/join (or hand-off, or loop) overhead.
    ///
    /// This is the per-matmul dispatch cost that a decode step pays 7+ times
    /// per layer. `x` is 1024 floats, the width the benchmark model projects
    /// from, so the pool's input staging copy is counted at its real size.
    #[test]
    #[ignore = "timing harness"]
    fn bench_dispatch_overhead() {
        let x = vec![1.0f32; 1024];
        for n_rows in [64usize, 1024, 4096] {
            let mut y = vec![0.0f32; n_rows];
            let reps = 200usize;
            let per_rep = u32::try_from(reps).unwrap();
            let mut samples = Vec::new();
            for _ in 0..9 {
                let t0 = Instant::now();
                for _ in 0..reps {
                    crate::pool::for_each_row(&mut y, |i, out| {
                        *out = black_box(i as f32);
                    });
                }
                samples.push(t0.elapsed() / per_rep);
            }
            report(&format!("dispatch {n_rows} rows, thread::scope"), samples);

            if let Some(mut pool) = Pool::new(Arc::new(NoopRows), n_rows) {
                let mut samples = Vec::new();
                for _ in 0..9 {
                    let t0 = Instant::now();
                    for _ in 0..reps {
                        assert!(pool.run((), &x, &mut y));
                    }
                    samples.push(t0.elapsed() / per_rep);
                }
                report(&format!("dispatch {n_rows} rows, persistent pool"), samples);
            }

            let mut samples = Vec::new();
            for _ in 0..9 {
                let t0 = Instant::now();
                for _ in 0..reps {
                    crate::pool::with_sequential(|| {
                        crate::pool::for_each_row(&mut y, |i, out| {
                            *out = black_box(i as f32);
                        });
                    });
                }
                samples.push(t0.elapsed() / per_rep);
            }
            report(&format!("dispatch {n_rows} rows, sequential"), samples);
            assert!(black_box(y.iter().sum::<f32>()) >= 0.0);
        }
    }
}
