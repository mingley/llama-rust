//! Pure-safe GGUF-native Llama decode + F32/F16/Q4_0/Q8_0/Q4_K/Q6_K/Q8_K. No llama.cpp, no C GGML.

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
    generate, generate_ctx, greedy_generate, greedy_generate_ctx, tiny_f16_gguf, tiny_llama_gguf,
    tiny_mistral_gguf, tiny_phi3_gguf, tiny_q4k_embd_gguf, tiny_q6k_embd_gguf, tiny_qwen2_gguf,
    KvCache, Llama, LlamaError,
};
pub use gguf::{
    load_gguf, load_gguf_owned, write_gguf, write_gguf_with_kv, GgmlType, Gguf, GgufError, Kv,
    Tensor, TensorWrite, GGUF_DEFAULT_ALIGNMENT,
};
pub use quant::{
    dequant_f16_row, dequant_f32_row, dequant_q4_k_row, dequant_q6_k_row, f16_row_bytes,
    f32_row_bytes, gemm_f16, gemm_f32, gemm_q4_k_f32, gemm_q6_k_f32, gemv_f16, gemv_f32, gemv_q4_0,
    gemv_q4_k, gemv_q4_k_f32, gemv_q6_k_f32, gemv_q8_0, pack_f16, pack_f32, pack_q4_0_block,
    pack_q4_0_from_i4, pack_q4_k_block, pack_q6_k_block, pack_q8_0_block, pack_q8_k_block,
    q4_0_row_bytes, q4_k_row_bytes, q6_k_row_bytes, q8_0_row_bytes, q8_k_row_bytes, QuantError,
    F16_SIZE, F32_SIZE, Q4_0_BLOCK, Q4_K_BLOCK, Q6_K_BLOCK, Q8_0_BLOCK, Q8_K_BLOCK, QK4_0, QK8_0,
    QK_K,
};
pub use sample::{argmax, sample_next, splitmix64, SampleError, SampleParams, Sampler};
pub use serve::{parse_serve_args, run_serve, ServeArgs, ServeCmd, ServeError, SERVE_USAGE};
pub use tok::{TokError, Tokenizer};
