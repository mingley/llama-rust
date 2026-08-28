//! A from-scratch Rust engine for GGUF-native Llama-family inference: hand it a
//! `.gguf` checkpoint and a prompt, get text back.
//!
//! There are no bindings to `llama.cpp`, no GGML FFI, no `mmap`, no SIMD crate,
//! and no thread-pool crate. `Cargo.lock` names exactly one package: this one.
//! Every byte of the loader, the tokenizer, the decode graph, and all thirty
//! quantized dtype kernels is safe Rust under a crate-wide
//! `#![forbid(unsafe_code)]`.
//!
//! That is the trade this crate makes. It is not the fastest CPU inference
//! engine — `llama.cpp` is several times quicker at decode, and says so in
//! hand-written SIMD. What you get instead is an engine you can read in an
//! afternoon, step through in a debugger, and change without fear: no unsafe
//! blocks to audit, no C toolchain, no build script, and an in-tree oracle plus
//! a differential test against `llama.cpp` to tell you when you broke something.
//!
//! # Quick start
//!
//! ```no_run
//! use llama_rust::{GenerateOptions, Model};
//!
//! # fn main() -> Result<(), llama_rust::Error> {
//! let model = Model::from_path("qwen2.5-0.5b-instruct-q4_k_m.gguf")?;
//! let text = model.generate("The capital of France is", &GenerateOptions::new(24))?;
//! println!("{text}");
//! # Ok(())
//! # }
//! ```
//!
//! [`Model::generate`] is the one-shot form. When you generate more than once,
//! open a [`Session`] so the KV cache allocation is reused:
//!
//! ```no_run
//! use llama_rust::{GenerateOptions, Model};
//!
//! # fn main() -> Result<(), llama_rust::Error> {
//! let model = Model::from_path("model.gguf")?;
//! let mut session = model.session();
//! let opts = GenerateOptions::new(32).with_n_ctx(512);
//! for prompt in ["1 + 1 =", "2 + 2 ="] {
//!     println!("{}", session.generate(prompt, &opts)?);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Runnable without a download
//!
//! [`fixtures`] builds complete, loadable GGUF checkpoints in memory — a few
//! kilobytes each, one per supported architecture and dtype. Every doctest in
//! this crate that generates text uses them, so the examples you are reading
//! actually run in CI:
//!
//! ```
//! use llama_rust::{fixtures, GenerateOptions, Model};
//!
//! # fn main() -> Result<(), llama_rust::Error> {
//! let model = Model::from_bytes(fixtures::tiny_qwen2_gguf())?;
//! let mut session = model.session();
//! let opts = GenerateOptions::new(4);
//!
//! let text = session.generate("ab", &opts)?;
//! // Greedy decoding uses no RNG, so the same prompt gives the same text.
//! assert_eq!(text, session.generate("ab", &opts)?);
//! # Ok(())
//! # }
//! ```
//!
//! The weights are arbitrary rather than trained, so the text is meaningless.
//! What they exercise is the machinery, which is exactly what you want when
//! developing against the API.
//!
//! # Streaming, logits, and swapping the sampler
//!
//! [`Session::generate_streaming`] calls you once per token with the token id,
//! its text, and the full logit vector the token was drawn from. Return
//! [`StepAction::Stop`] to end generation early.
//!
//! ```
//! use llama_rust::{fixtures, GenerateOptions, Model, StepAction};
//!
//! # fn main() -> Result<(), llama_rust::Error> {
//! let model = Model::from_bytes(fixtures::tiny_qwen2_gguf())?;
//! let opts = GenerateOptions::new(8)
//!     .with_temperature(0.8)
//!     .with_top_k(40)
//!     .with_top_p(0.95)
//!     .with_repeat_penalty(1.1)
//!     .with_seed(1234);
//!
//! let mut peak_logit = f32::MIN;
//! let done = model.session().generate_streaming("ab", &opts, |step| {
//!     for logit in step.logits {
//!         peak_logit = peak_logit.max(*logit);
//!     }
//!     print!("{}", step.piece);
//!     StepAction::Continue
//! })?;
//! println!(
//!     "\n{} tokens, peak logit {peak_logit}, stopped on {:?}",
//!     done.tokens.len(),
//!     done.stop,
//! );
//! # Ok(())
//! # }
//! ```
//!
//! Want a sampler this crate does not have? Skip the built-in one: drive
//! [`Llama::prefill`] and [`Llama::forward`] yourself and do whatever you like
//! with the logits.
//!
//! ```
//! use llama_rust::{fixtures, Llama, Model};
//!
//! # fn main() -> Result<(), llama_rust::Error> {
//! let model = Model::from_bytes(fixtures::tiny_llama_gguf())?;
//! let weights: &Llama = model.weights();
//! let prompt = model.encode("ab")?;
//!
//! let mut cache = weights.new_cache(prompt.len() + 4)?;
//! let mut logits = weights.prefill(&mut cache, &prompt)?;
//! let mut ids = Vec::new();
//! for _ in 0..3 {
//!     // Your decision rule goes here; this one is plain argmax.
//!     let (best, _) = logits
//!         .iter()
//!         .enumerate()
//!         .fold((0usize, f32::MIN), |acc, (i, v)| {
//!             if *v > acc.1 { (i, *v) } else { acc }
//!         });
//!     let next = u32::try_from(best).unwrap_or(0);
//!     ids.push(next);
//!     logits = weights.forward(&mut cache, next)?;
//! }
//! assert_eq!(ids.len(), 3);
//! println!("{}", model.tokenizer().decode(&ids));
//! # Ok(())
//! # }
//! ```
//!
//! # Safe by default
//!
//! `#![forbid(unsafe_code)]` is a hard constraint on the whole crate, not a
//! default that kernels opt out of. Concretely, that shapes the code in ways
//! worth knowing about before you start hacking:
//!
//! - **No `mmap`.** A GGUF is read into one owned `Vec<u8>` and every weight
//!   matrix is a byte *range* of that buffer, never a copy. Peak resident
//!   memory is about the file size plus the KV cache.
//! - **No pointer casts into quantized blocks.** Kernels read packed bytes
//!   through `slice::as_chunks` and `u16::from_le_bytes`, so a malformed or
//!   truncated checkpoint returns [`kernels::QuantError`] or
//!   [`gguf::GgufError`] instead of reading out of bounds.
//! - **Errors, not panics.** `unwrap`, `expect`, `panic!`, and slice indexing
//!   are denied by lint outside tests. Every fallible operation returns
//!   `Result`, so a bad file cannot take the process down.
//! - **Checked arithmetic.** Shape and offset maths uses `checked_*` and
//!   `saturating_*`, because a bogus tensor extent in a header must not become
//!   an overflowing multiply.
//!
//! # Module map
//!
//! The crate root holds only what a first-time user needs. Everything else is
//! namespaced:
//!
//! | Module | What lives there |
//! |---|---|
//! | *(root)* | [`Model`], [`Session`], [`GenerateOptions`], [`Generated`], [`Step`], [`SampleParams`], [`Tokenizer`], [`Error`] |
//! | [`gguf`] | GGUF v3 reader and writer: [`gguf::Gguf`], [`gguf::load_gguf`], [`gguf::write_gguf`], [`gguf::GgmlType`], [`gguf::Kv`] |
//! | [`kernels`] | Dequant and matmul kernels: `dequant_*`, `gemv_*`, `gemm_*`, `pack_*`, `*_row_bytes`, block-size constants |
//! | [`sample`] | [`sample::SampleParams`], [`sample::Sampler`], [`sample::sample_next`], [`sample::argmax`] |
//! | [`tokenizer`] | [`tokenizer::Tokenizer`] and the GGUF-embedded vocab / BPE merges |
//! | [`fixtures`] | In-memory GGUF checkpoints for tests, doctests, and benchmarks |
//! | [`cli`], [`serve`] | Argument parsing and the tiny HTTP server behind the `gguf_gemv` binary |
//!
//! [`Llama`] and [`KvCache`] also sit at the root: they are the raw decode
//! handle and its cache, one layer below [`Session`], and you will want them as
//! soon as you care about logits or cache placement.
//!
//! # Example programs
//!
//! Five in `examples/`. All but the first default to an in-memory fixture, so
//! `cargo run --example <name>` works with nothing downloaded.
//!
//! | Example | What it shows |
//! |---|---|
//! | `generate` | File to text, with prefill and decode throughput timed apart. Needs a real `.gguf`. |
//! | `stream_tokens` | Per-token callback, stopping early on a stop sequence |
//! | `sampling` | Six sampler configurations side by side, and why a seed is mandatory |
//! | `kernels` | Dequantize a row, run the fused GEMV over packed bytes, cross-check the two |
//! | `gguf_inventory` | Metadata dump and a per-dtype tensor census of a checkpoint |
//!
//! # What it loads
//!
//! Architectures, keyed off `general.architecture`: `llama` (dense and MoE),
//! `qwen2`, `mistral`, `phi3`, `gemma`, `qwen3`, `llama4`, `qwen2moe`,
//! `qwen3moe`, `qwen2vl`, `qwen3vl`, `qwen3next`, and `qwen35`. Vision towers
//! are not run; the language model of a VL checkpoint is.
//!
//! Tensor dtypes: `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`,
//! `Q8_1`, `Q1_0`, `Q2_0`, `TQ1_0`, `TQ2_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`,
//! `Q6_K`, `Q8_K`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`,
//! `IQ3_S`, `IQ4_NL`, `IQ4_XS`, `MXFP4`, and `NVFP4`.
//!
//! # How correctness is checked
//!
//! Two independent layers, because each catches what the other cannot:
//!
//! 1. **An in-tree oracle.** Every dtype has a second, deliberately naive
//!    implementation, and tests assert the production kernel matches it. Tiny
//!    writer-built checkpoints ([`fixtures`]) cover every architecture and
//!    dtype end to end.
//! 2. **A differential test against `llama.cpp`.** Gated behind an environment
//!    variable because it downloads real weights. This layer is not optional:
//!    every per-dtype oracle shares the crate's `binary16` conversion, so a bug
//!    in that one primitive is invisible to the entire oracle suite. Exactly
//!    that happened — a subnormal `binary16` decode that silently halved real
//!    `Q4_K` and `Q6_K` weights survived 221 passing tests.
//!
//! If you add a dtype or an architecture, add both layers. Do not loosen a
//! tolerance to make a test pass.

#![forbid(unsafe_code)]

pub mod cli;
pub mod fixtures;
pub mod gguf;
pub mod sample;
pub mod serve;

// `quant.rs` and `tok.rs` keep their filenames while presenting under clearer
// public names, so the ~140 internal `crate::quant::` / `crate::tok::` paths
// and any in-flight work on those files stay valid.
#[path = "quant.rs"]
pub mod kernels;
#[path = "tok.rs"]
pub mod tokenizer;

pub(crate) use crate::kernels as quant;
pub(crate) use crate::tokenizer as tok;

mod decode;
mod fp16;
mod pool;
mod session;

pub use crate::decode::LlamaError as Error;
pub use crate::decode::{KvCache, Llama};
pub use crate::sample::SampleParams;
pub use crate::session::{
    GenerateOptions, Generated, Model, Session, Step, StepAction, StopReason,
};
pub use crate::tokenizer::Tokenizer;

/// Former name of [`Error`].
#[deprecated(since = "0.2.0", note = "renamed to `llama_rust::Error`")]
pub type LlamaError = Error;

/// Greedy generate, returning the prompt *and* its continuation as one string.
///
/// # Deprecated
///
/// Use [`Model::generate`], which loads the tokenizer alongside the weights so
/// there is nothing to thread through by hand, and returns the continuation on
/// its own:
///
/// ```
/// # use llama_rust::{fixtures, GenerateOptions, Model};
/// # fn main() -> Result<(), llama_rust::Error> {
/// let model = Model::from_bytes(fixtures::tiny_llama_gguf())?;
/// let text = model.generate("ab", &GenerateOptions::new(4))?;
/// # Ok(())
/// # }
/// ```
#[deprecated(since = "0.2.0", note = "use `Model::generate` or `Session::generate`")]
pub fn greedy_generate(
    model: &Llama,
    tok: &Tokenizer,
    prompt: &str,
    n_predict: usize,
) -> Result<String, Error> {
    decode::greedy_generate(model, tok, prompt, n_predict)
}

/// [`greedy_generate`] with an explicit KV capacity.
///
/// # Deprecated
///
/// Use [`GenerateOptions::with_n_ctx`] with [`Session::generate`].
#[deprecated(
    since = "0.2.0",
    note = "use `GenerateOptions::with_n_ctx` and `Session::generate`"
)]
pub fn greedy_generate_ctx(
    model: &Llama,
    tok: &Tokenizer,
    prompt: &str,
    n_predict: usize,
    n_ctx: Option<usize>,
) -> Result<String, Error> {
    decode::greedy_generate_ctx(model, tok, prompt, n_predict, n_ctx)
}

/// Generate with explicit [`SampleParams`], returning prompt and continuation.
///
/// # Deprecated
///
/// Use [`GenerateOptions::with_sampling`] with [`Session::generate`].
#[deprecated(
    since = "0.2.0",
    note = "use `GenerateOptions::with_sampling` and `Session::generate`"
)]
pub fn generate(
    model: &Llama,
    tok: &Tokenizer,
    prompt: &str,
    n_predict: usize,
    params: &SampleParams,
) -> Result<String, Error> {
    decode::generate(model, tok, prompt, n_predict, params)
}

/// [`generate`] with an explicit KV capacity.
///
/// # Deprecated
///
/// Use [`GenerateOptions::with_sampling`] and [`GenerateOptions::with_n_ctx`]
/// with [`Session::generate`].
#[deprecated(
    since = "0.2.0",
    note = "use `GenerateOptions` with `Session::generate`"
)]
pub fn generate_ctx(
    model: &Llama,
    tok: &Tokenizer,
    prompt: &str,
    n_predict: usize,
    n_ctx: Option<usize>,
    params: &SampleParams,
) -> Result<String, Error> {
    decode::generate_ctx(model, tok, prompt, n_predict, n_ctx, params)
}
