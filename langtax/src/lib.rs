//! Pure-safe GGUF-native Llama decode + F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q1_0/Q2_0/TQ1_0/TQ2_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K/IQ1_M/IQ1_S/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_NL/IQ4_XS/MXFP4/NVFP4. No llama.cpp, no C GGML.

//! Without the `simd` feature the whole engine is compiler-checked unsafe-free.
//! With it, `src/simd/` is the only module allowed to use `unsafe`; the
//! workspace lint keeps `unsafe_code` denied everywhere else.
#![cfg_attr(not(feature = "simd"), forbid(unsafe_code))]

mod cli;
mod decode;
mod fp16;
mod gguf;
mod pool;
mod pretok;
mod quant;
mod sample;
mod serve;
#[cfg(feature = "simd")]
mod simd;
mod template;
mod tok;
mod ucd;

pub use cli::{
    parse_chat_args, parse_infer_args, parse_trace_args, run_chat, ChatArgs, ChatCmd, InferArgs,
    InferCmd, TraceArgs, TraceCmd, BIN_USAGE, CHAT_USAGE, INFER_USAGE, TRACE_USAGE,
};
pub use decode::{
    generate, generate_ctx, greedy_generate, greedy_generate_ctx, greedy_generate_traced,
    tiny_bf16_gguf, tiny_f16_1d_bias_gguf, tiny_f16_1d_gguf, tiny_f16_gguf, tiny_gemma_gguf,
    tiny_iq1m_gguf, tiny_iq1s_gguf, tiny_iq2s_gguf, tiny_iq2xs_gguf, tiny_iq2xxs_gguf,
    tiny_iq3s_gguf, tiny_iq3xxs_gguf, tiny_iq4nl_gguf, tiny_iq4xs_gguf, tiny_llama4_gguf,
    tiny_llama_gguf, tiny_llama_moe_gguf, tiny_mha_gguf, tiny_mistral_gguf, tiny_mqa_gguf,
    tiny_mxfp4_gguf, tiny_nvfp4_gguf, tiny_phi2_gguf, tiny_phi3_gguf, tiny_q10_gguf, tiny_q20_gguf,
    tiny_q2k_gguf, tiny_q3k_gguf, tiny_q40_gguf, tiny_q41_gguf, tiny_q4k_embd_gguf, tiny_q50_gguf,
    tiny_q51_gguf, tiny_q5k_gguf, tiny_q6k_embd_gguf, tiny_q80_gguf, tiny_q81_gguf,
    tiny_qwen2_gguf, tiny_qwen2moe_gguf, tiny_qwen2vl_gguf, tiny_qwen35_gguf, tiny_qwen3_gguf,
    tiny_qwen3moe_gguf, tiny_qwen3next_gguf, tiny_qwen3vl_gguf, tiny_tied_copy_gguf,
    tiny_tied_gguf, tiny_tq10_gguf, tiny_tq20_gguf, KvCache, Llama, LlamaError,
};
pub use expertvm::{ExpertAccess, ExpertKey, Trace};
pub use gguf::{
    load_gguf, load_gguf_owned, write_gguf, write_gguf_with_kv, GgmlType, Gguf, GgufError, Kv,
    Tensor, TensorWrite, GGUF_DEFAULT_ALIGNMENT,
};
pub use pretok::PreTokenizer;
pub use quant::{
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
    gemv_nvfp4_f32, gemv_q1_0_f32, gemv_q2_0_f32, gemv_q2_k_f32, gemv_q3_k_f32, gemv_q4_0,
    gemv_q4_0_f32, gemv_q4_1_f32, gemv_q4_k, gemv_q4_k_f32, gemv_q5_0_f32, gemv_q5_1_f32,
    gemv_q5_k_f32, gemv_q6_k_f32, gemv_q8_0, gemv_q8_0_f32, gemv_q8_1_f32, gemv_tq1_0_f32,
    gemv_tq2_0_f32, iq1_m_row_bytes, iq1_s_row_bytes, iq2_s_row_bytes, iq2_xs_row_bytes,
    iq2_xxs_row_bytes, iq3_s_row_bytes, iq3_xxs_row_bytes, iq4_nl_row_bytes, iq4_xs_row_bytes,
    mxfp4_row_bytes, nvfp4_row_bytes, pack_bf16, pack_f16, pack_f32, pack_iq1_m_block,
    pack_iq1_s_block, pack_iq2_s_block, pack_iq2_xs_block, pack_iq2_xxs_block, pack_iq3_s_block,
    pack_iq3_xxs_block, pack_iq4_nl_block, pack_iq4_xs_block, pack_mxfp4_block, pack_nvfp4_block,
    pack_q1_0_block, pack_q2_0_block, pack_q2_k_block, pack_q3_k_block, pack_q4_0_block,
    pack_q4_0_from_i4, pack_q4_1_block, pack_q4_k_block, pack_q5_0_block, pack_q5_1_block,
    pack_q5_k_block, pack_q6_k_block, pack_q8_0_block, pack_q8_1_block, pack_q8_k_block,
    pack_tq1_0_block, pack_tq2_0_block, q1_0_row_bytes, q2_0_row_bytes, q2_k_row_bytes,
    q3_k_row_bytes, q4_0_row_bytes, q4_1_row_bytes, q4_k_row_bytes, q5_0_row_bytes, q5_1_row_bytes,
    q5_k_row_bytes, q6_k_row_bytes, q8_0_row_bytes, q8_1_row_bytes, q8_k_row_bytes,
    tq1_0_row_bytes, tq2_0_row_bytes, QuantError, BF16_SIZE, F16_SIZE, F32_SIZE, IQ1_M_BLOCK,
    IQ1_S_BLOCK, IQ2_S_BLOCK, IQ2_XS_BLOCK, IQ2_XXS_BLOCK, IQ3_S_BLOCK, IQ3_XXS_BLOCK,
    IQ4_NL_BLOCK, IQ4_XS_BLOCK, MXFP4_BLOCK, NVFP4_BLOCK, Q1_0_BLOCK, Q2_0_BLOCK, Q2_K_BLOCK,
    Q3_K_BLOCK, Q4_0_BLOCK, Q4_1_BLOCK, Q4_K_BLOCK, Q5_0_BLOCK, Q5_1_BLOCK, Q5_K_BLOCK, Q6_K_BLOCK,
    Q8_0_BLOCK, Q8_1_BLOCK, Q8_K_BLOCK, QK1_0, QK2_0, QK4_0, QK4_1, QK4_NL, QK5_0, QK5_1, QK8_0,
    QK8_1, QK_K, QK_MXFP4, QK_NVFP4, TQ1_0_BLOCK, TQ2_0_BLOCK,
};
pub use sample::{argmax, sample_next, splitmix64, SampleError, SampleParams, Sampler};
pub use serve::{parse_serve_args, run_serve, ServeArgs, ServeCmd, ServeError, SERVE_USAGE};
pub use template::{
    render_chat_template, ChatMessage, ChatOptions, Template, TemplateError, Value,
};
pub use tok::{TokError, Tokenizer};
