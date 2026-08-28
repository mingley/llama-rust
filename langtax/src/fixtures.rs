//! In-memory GGUF checkpoints, built by this crate's own writer.
//!
//! Every builder returns a complete, loadable GGUF as bytes: header, metadata,
//! tokenizer vocab, and packed weights. They are a few kilobytes each, so a
//! test, doctest, example, or benchmark can load a working model and generate
//! text with no download and no fixture files on disk:
//!
//! ```
//! use llama_rust::{fixtures, GenerateOptions, Model};
//!
//! # fn main() -> Result<(), llama_rust::Error> {
//! let model = Model::from_bytes(fixtures::tiny_q4k_embd_gguf())?;
//! let opts = GenerateOptions::new(4).with_stop_at_eos(false);
//! let done = model.session().generate_detailed("ab", &opts)?;
//! assert_eq!(done.tokens.len(), 4);
//! # Ok(())
//! # }
//! ```
//!
//! The weights are arbitrary, not trained, so the text they produce is
//! meaningless. What they exercise is the *machinery*: one fixture per
//! supported architecture and one per supported ggml dtype, which is how this
//! crate's tests cover 13 architectures and 30 dtypes without shipping
//! gigabytes of checkpoints.
//!
//! Their vocabularies are six tokens wide, and untrained weights often make
//! `argmax` land on BOS or EOS. Since detokenizing drops those ids, a fixture
//! can return several token ids and still decode to an empty string — assert on
//! [`crate::Generated::tokens`] rather than on the text. [`tiny_qwen2_gguf`]
//! and [`tiny_qwen3_gguf`] are the ones that reliably emit visible characters.
//!
//! # Stability
//!
//! This module is test scaffolding that happens to be useful downstream. It is
//! deliberately namespaced rather than exported at the crate root, and it is
//! **excluded from the crate's semantic-versioning promise**: builders may be
//! renamed, added, or removed in a patch release as dtype and architecture
//! coverage changes. Do not build a public API on top of it.
//!
//! # Choosing a fixture
//!
//! - Architecture coverage: [`tiny_llama_gguf`], [`tiny_qwen2_gguf`],
//!   [`tiny_mistral_gguf`], [`tiny_phi3_gguf`], [`tiny_gemma_gguf`],
//!   [`tiny_qwen3_gguf`], [`tiny_llama4_gguf`], [`tiny_llama_moe_gguf`],
//!   [`tiny_qwen2moe_gguf`], [`tiny_qwen3moe_gguf`], [`tiny_qwen2vl_gguf`],
//!   [`tiny_qwen3vl_gguf`], [`tiny_qwen3next_gguf`], [`tiny_qwen35_gguf`].
//! - Tied embeddings (no `output.weight`): [`tiny_tied_gguf`],
//!   [`tiny_tied_copy_gguf`].
//! - Float dtypes: [`tiny_f16_gguf`], [`tiny_bf16_gguf`],
//!   [`tiny_f16_1d_gguf`], [`tiny_f16_1d_bias_gguf`].
//! - Round-number quants: [`tiny_q41_gguf`], [`tiny_q50_gguf`],
//!   [`tiny_q51_gguf`], [`tiny_q80_gguf`], [`tiny_q81_gguf`],
//!   [`tiny_q10_gguf`], [`tiny_q20_gguf`].
//! - K-quants: [`tiny_q2k_gguf`], [`tiny_q3k_gguf`], [`tiny_q4k_embd_gguf`],
//!   [`tiny_q5k_gguf`], [`tiny_q6k_embd_gguf`].
//! - I-quants: [`tiny_iq1s_gguf`], [`tiny_iq1m_gguf`], [`tiny_iq2xxs_gguf`],
//!   [`tiny_iq2xs_gguf`], [`tiny_iq2s_gguf`], [`tiny_iq3xxs_gguf`],
//!   [`tiny_iq3s_gguf`], [`tiny_iq4nl_gguf`], [`tiny_iq4xs_gguf`].
//! - Ternary and micro-float: [`tiny_tq10_gguf`], [`tiny_tq20_gguf`],
//!   [`tiny_mxfp4_gguf`], [`tiny_nvfp4_gguf`].
//!
//! To build a GGUF of your own shape instead, use [`crate::gguf::write_gguf`]
//! and [`crate::gguf::write_gguf_with_kv`] directly.

pub use crate::decode::{
    tiny_bf16_gguf, tiny_f16_1d_bias_gguf, tiny_f16_1d_gguf, tiny_f16_gguf, tiny_gemma_gguf,
    tiny_iq1m_gguf, tiny_iq1s_gguf, tiny_iq2s_gguf, tiny_iq2xs_gguf, tiny_iq2xxs_gguf,
    tiny_iq3s_gguf, tiny_iq3xxs_gguf, tiny_iq4nl_gguf, tiny_iq4xs_gguf, tiny_llama4_gguf,
    tiny_llama_gguf, tiny_llama_moe_gguf, tiny_mistral_gguf, tiny_mxfp4_gguf, tiny_nvfp4_gguf,
    tiny_phi3_gguf, tiny_q10_gguf, tiny_q20_gguf, tiny_q2k_gguf, tiny_q3k_gguf, tiny_q41_gguf,
    tiny_q4k_embd_gguf, tiny_q50_gguf, tiny_q51_gguf, tiny_q5k_gguf, tiny_q6k_embd_gguf,
    tiny_q80_gguf, tiny_q81_gguf, tiny_qwen2_gguf, tiny_qwen2moe_gguf, tiny_qwen2vl_gguf,
    tiny_qwen35_gguf, tiny_qwen3_gguf, tiny_qwen3moe_gguf, tiny_qwen3next_gguf, tiny_qwen3vl_gguf,
    tiny_tied_copy_gguf, tiny_tied_gguf, tiny_tq10_gguf, tiny_tq20_gguf,
};
