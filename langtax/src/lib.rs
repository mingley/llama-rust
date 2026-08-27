//! Pure-safe GGUF-native Llama decode + F32/F16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/Q8_K/IQ3_S/IQ4_NL/IQ4_XS. No llama.cpp, no C GGML.

#![forbid(unsafe_code)]

mod cli;
mod decode;
mod fp16;
mod gguf;
mod pool;
mod quant;
mod sample;
mod serve;
mod tok;

pub use cli::{parse_infer_args, InferArgs, InferCmd, BIN_USAGE, INFER_USAGE};
pub use decode::{
    generate, generate_ctx, greedy_generate, greedy_generate_ctx, tiny_f16_gguf, tiny_iq3s_gguf,
    tiny_iq4nl_gguf, tiny_iq4xs_gguf, tiny_llama_gguf, tiny_mistral_gguf, tiny_phi3_gguf,
    tiny_q4k_embd_gguf, tiny_q5k_gguf, tiny_q6k_embd_gguf, tiny_qwen2_gguf, KvCache, Llama,
    LlamaError,
};
pub use gguf::{
    load_gguf, load_gguf_owned, write_gguf, write_gguf_with_kv, GgmlType, Gguf, GgufError, Kv,
    Tensor, TensorWrite, GGUF_DEFAULT_ALIGNMENT,
};
pub use quant::{
    dequant_f16_row, dequant_f32_row, dequant_iq3_s_row, dequant_iq4_nl_row, dequant_iq4_xs_row,
    dequant_q4_k_row, dequant_q5_k_row, dequant_q6_k_row, f16_row_bytes, f32_row_bytes, gemm_f16,
    gemm_f32, gemm_iq3_s_f32, gemm_iq4_nl_f32, gemm_iq4_xs_f32, gemm_q4_k_f32, gemm_q5_k_f32,
    gemm_q6_k_f32, gemv_f16, gemv_f32, gemv_iq3_s_f32, gemv_iq4_nl_f32, gemv_iq4_xs_f32, gemv_q4_0,
    gemv_q4_k, gemv_q4_k_f32, gemv_q5_k_f32, gemv_q6_k_f32, gemv_q8_0, iq3_s_row_bytes,
    iq4_nl_row_bytes, iq4_xs_row_bytes, pack_f16, pack_f32, pack_iq3_s_block, pack_iq4_nl_block,
    pack_iq4_xs_block, pack_q4_0_block, pack_q4_0_from_i4, pack_q4_k_block, pack_q5_k_block,
    pack_q6_k_block, pack_q8_0_block, pack_q8_k_block, q4_0_row_bytes, q4_k_row_bytes,
    q5_k_row_bytes, q6_k_row_bytes, q8_0_row_bytes, q8_k_row_bytes, QuantError, F16_SIZE, F32_SIZE,
    IQ3_S_BLOCK, IQ4_NL_BLOCK, IQ4_XS_BLOCK, Q4_0_BLOCK, Q4_K_BLOCK, Q5_K_BLOCK, Q6_K_BLOCK,
    Q8_0_BLOCK, Q8_K_BLOCK, QK4_0, QK4_NL, QK8_0, QK_K,
};
pub use sample::{argmax, sample_next, splitmix64, SampleError, SampleParams, Sampler};
pub use serve::{parse_serve_args, run_serve, ServeArgs, ServeCmd, ServeError, SERVE_USAGE};
pub use tok::{TokError, Tokenizer};
