//! Pure-safe GGUF-native Q4_0/Q8_0 load + GEMV. No llama.cpp, no C GGML.

#![forbid(unsafe_code)]

mod fp16;
mod gguf;
mod pool;
mod quant;

pub use gguf::{
    load_gguf, write_gguf, GgmlType, Gguf, GgufError, Tensor, TensorWrite, GGUF_DEFAULT_ALIGNMENT,
};
pub use quant::{
    gemv_q4_0, gemv_q8_0, pack_q4_0_block, pack_q4_0_from_i4, pack_q8_0_block, q4_0_row_bytes,
    q8_0_row_bytes, QuantError, Q4_0_BLOCK, Q8_0_BLOCK, QK4_0, QK8_0,
};
