//! The front door: load a [`Model`], open a [`Session`], generate text.
//!
//! This module is the layer a first-time user should reach for. It bundles the
//! three things a GGUF checkpoint always ships together — weights, tokenizer,
//! and hyperparameters — into one handle, and hides the KV cache behind a
//! session object that sizes and reuses it.
//!
//! Everything here is a thin, allocation-light wrapper over the raw API in
//! [`Llama`]. Nothing is hidden: [`Model::weights`] hands back the loaded
//! [`Llama`], and [`Llama::new_cache`] / [`Llama::prefill`] / [`Llama::forward`]
//! remain available when you want to drive the decode loop yourself.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::decode::{prompt_ids, KvCache, Llama, LlamaError};
use crate::gguf::{load_gguf_owned, Gguf};
use crate::sample::{SampleParams, Sampler};
use crate::tok::Tokenizer;

/// Knobs for one generation call.
///
/// Construct with [`GenerateOptions::new`], which asks for the one value that
/// has no sensible default — how many tokens to generate — and leaves the rest
/// at seedless greedy decoding. The `with_*` methods are chainable; the fields
/// are public, so a struct literal works too.
///
/// ```
/// use llama_rust::{GenerateOptions, SampleParams};
///
/// let opts = GenerateOptions::new(64).with_sampling(SampleParams {
///     temperature: 0.8,
///     top_k: 40,
///     top_p: 0.95,
///     repeat_penalty: 1.1,
///     seed: Some(7),
/// });
/// assert_eq!(opts.n_predict, 64);
/// assert!(!opts.sampling.is_greedy());
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerateOptions {
    /// Upper bound on generated tokens. Generation may stop earlier on EOS or
    /// when a streaming callback asks it to.
    pub n_predict: usize,
    /// KV cache capacity in tokens, covering prompt *and* continuation.
    ///
    /// `None` sizes the cache to `prompt + n_predict + 1`, which always fits.
    /// `Some(n)` pins the capacity, which lets a [`Session`] reuse one
    /// allocation across calls; it is an error if `n` cannot hold the prompt
    /// plus `n_predict`.
    pub n_ctx: Option<usize>,
    /// Temperature, top-k, top-p, repeat penalty, and seed.
    ///
    /// The default is [`SampleParams::greedy`], which needs no seed and is
    /// bit-for-bit reproducible.
    pub sampling: SampleParams,
    /// Stop as soon as the tokenizer's EOS id is drawn.
    ///
    /// The EOS token is never included in the returned text or token ids. Set
    /// this to `false` to force exactly `n_predict` tokens, which is
    /// occasionally what you want when measuring throughput.
    pub stop_at_eos: bool,
}

impl GenerateOptions {
    /// Greedy decoding of at most `n_predict` tokens, cache sized automatically.
    pub const fn new(n_predict: usize) -> Self {
        Self {
            n_predict,
            n_ctx: None,
            sampling: SampleParams::greedy(),
            stop_at_eos: true,
        }
    }

    /// Set [`Self::n_predict`].
    #[must_use]
    pub const fn with_n_predict(mut self, n_predict: usize) -> Self {
        self.n_predict = n_predict;
        self
    }

    /// Pin the KV capacity. See [`Self::n_ctx`].
    #[must_use]
    pub const fn with_n_ctx(mut self, n_ctx: usize) -> Self {
        self.n_ctx = Some(n_ctx);
        self
    }

    /// Set the sampling knobs. See [`SampleParams`].
    #[must_use]
    pub const fn with_sampling(mut self, sampling: SampleParams) -> Self {
        self.sampling = sampling;
        self
    }

    /// Set [`Self::stop_at_eos`].
    #[must_use]
    pub const fn with_stop_at_eos(mut self, stop_at_eos: bool) -> Self {
        self.stop_at_eos = stop_at_eos;
        self
    }
}

/// Why a generation call returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// [`GenerateOptions::n_predict`] tokens were produced.
    Length,
    /// The tokenizer's EOS id was drawn. It is not part of the output.
    Eos,
    /// A streaming callback returned [`StepAction::Stop`].
    Callback,
}

/// What a streaming callback wants to happen next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepAction {
    /// Draw another token.
    Continue,
    /// Return now. The token just reported stays in the output.
    Stop,
}

/// One generated token, handed to the [`Session::generate_streaming`] callback.
#[derive(Debug)]
pub struct Step<'a> {
    /// Position of this token within the continuation, counting from zero.
    /// Prompt tokens are not counted.
    pub index: usize,
    /// The token id that was drawn.
    pub token: u32,
    /// [`Self::token`] detokenized on its own.
    ///
    /// Best-effort: a multi-byte character split across two tokens decodes to
    /// one replacement character in each piece. Concatenating pieces is fine
    /// for a terminal, but use [`Generated::text`] when you need the exact
    /// string, since that detokenizes the whole continuation in one pass.
    pub piece: String,
    /// The full `n_vocab`-wide logit vector this token was drawn from, before
    /// any sampling transform.
    ///
    /// This is the hook for logit-level research: inspect it, or ignore the
    /// crate's sampler entirely and drive [`Llama::forward`] yourself.
    pub logits: &'a [f32],
}

/// The result of a generation call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generated {
    /// Prompt token ids as fed to the model, including BOS when the GGUF asked
    /// for one.
    pub prompt_tokens: Vec<u32>,
    /// Generated token ids. Never includes EOS.
    pub tokens: Vec<u32>,
    /// [`Self::tokens`] detokenized: the continuation only, with no echo of the
    /// prompt.
    ///
    /// Detokenizing drops BOS and EOS ids, so this can be shorter than
    /// [`Self::tokens`] suggests — and empty even when tokens were produced, if
    /// the model kept choosing a special token. When you care about what the
    /// model actually emitted, look at [`Self::tokens`].
    pub text: String,
    /// Why generation stopped.
    pub stop: StopReason,
}

impl Generated {
    /// Prompt ids followed by continuation ids: every token the model saw or
    /// produced, in order.
    ///
    /// Detokenizing this is how you reproduce the prompt-echoing string that
    /// the deprecated free functions returned:
    ///
    /// ```
    /// use llama_rust::{fixtures, GenerateOptions, Model};
    ///
    /// # fn main() -> Result<(), llama_rust::Error> {
    /// let model = Model::from_bytes(fixtures::tiny_qwen2_gguf())?;
    /// let done = model.session().generate_detailed("ab", &GenerateOptions::new(3))?;
    /// let echoed = model.tokenizer().decode(&done.all_tokens());
    /// assert!(!done.text.is_empty());
    /// assert!(echoed.ends_with(&done.text));
    /// # Ok(())
    /// # }
    /// ```
    pub fn all_tokens(&self) -> Vec<u32> {
        let mut all =
            Vec::with_capacity(self.prompt_tokens.len().saturating_add(self.tokens.len()));
        all.extend_from_slice(&self.prompt_tokens);
        all.extend_from_slice(&self.tokens);
        all
    }
}

/// A loaded model: weights plus the tokenizer embedded in the same GGUF.
///
/// Loading is the expensive step, so a `Model` is meant to be built once and
/// shared. It is immutable — all mutable decode state lives in [`Session`] —
/// so `&Model` can be handed to as many sessions as you like.
///
/// ```no_run
/// use llama_rust::{GenerateOptions, Model};
///
/// # fn main() -> Result<(), llama_rust::Error> {
/// let model = Model::from_path("qwen2.5-0.5b-instruct-q4_k_m.gguf")?;
/// println!("vocab {} embd {}", model.n_vocab(), model.n_embd());
/// let text = model.generate("The capital of France is", &GenerateOptions::new(16))?;
/// println!("{text}");
/// # Ok(())
/// # }
/// ```
pub struct Model {
    weights: Llama,
    tokenizer: Tokenizer,
}

impl Model {
    /// Read a GGUF file from disk and load it.
    ///
    /// The whole file is read into one owned buffer; weight matrices are byte
    /// ranges of it, never copies. There is no memory mapping, so peak resident
    /// memory is roughly the file size plus the KV cache.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, LlamaError> {
        let mut file = File::open(path.as_ref())?;
        let mut bytes = Vec::new();
        let _len = file.read_to_end(&mut bytes)?;
        Self::from_bytes(bytes)
    }

    /// Load from GGUF bytes already in memory.
    ///
    /// Takes ownership so weight payloads stay borrowed from a single
    /// allocation. Useful for embedded checkpoints and for the in-memory
    /// fixtures in [`crate::fixtures`].
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, LlamaError> {
        Self::from_gguf(load_gguf_owned(bytes)?)
    }

    /// Build from an already-parsed [`Gguf`].
    ///
    /// Use this when you want to read metadata before committing to a load —
    /// for example to check `general.architecture` or take a dtype inventory.
    pub fn from_gguf(gguf: Gguf) -> Result<Self, LlamaError> {
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        let weights = Llama::from_gguf(gguf)?;
        Ok(Self { weights, tokenizer })
    }

    /// Vocabulary size, and so the width of every logit vector.
    pub fn n_vocab(&self) -> usize {
        self.weights.n_vocab
    }

    /// Embedding width (`{arch}.embedding_length`).
    pub fn n_embd(&self) -> usize {
        self.weights.n_embd
    }

    /// The loaded weights, for driving the decode loop directly.
    pub fn weights(&self) -> &Llama {
        &self.weights
    }

    /// The tokenizer that shipped inside the GGUF.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Encode a prompt exactly as generation would, BOS handling included.
    ///
    /// Use this when you want to inspect or edit the ids before feeding them to
    /// [`Llama::prefill`].
    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>, LlamaError> {
        prompt_ids(&self.tokenizer, prompt)
    }

    /// Open a session. Its KV cache is allocated on first use and sized to fit.
    pub fn session(&self) -> Session<'_> {
        Session {
            model: self,
            cache: None,
            last_logits: None,
        }
    }

    /// Generate text in one call, using a throwaway session.
    ///
    /// Equivalent to `self.session().generate(prompt, options)`. Reach for a
    /// [`Session`] instead when you generate more than once, so the KV cache
    /// allocation is reused.
    pub fn generate(&self, prompt: &str, options: &GenerateOptions) -> Result<String, LlamaError> {
        self.session().generate(prompt, options)
    }
}

/// One conversation's worth of mutable decode state: a KV cache and the logits
/// from the most recent forward pass.
///
/// A session borrows its [`Model`], so weights are never copied. The cache is
/// allocated lazily and reused across calls when it is already large enough,
/// which is why pinning [`GenerateOptions::n_ctx`] is worthwhile in a loop.
///
/// Each `generate*` call starts a fresh sequence: the cache is rewound to
/// position zero and the prompt is prefilled again.
pub struct Session<'a> {
    model: &'a Model,
    cache: Option<KvCache>,
    last_logits: Option<Vec<f32>>,
}

impl<'a> Session<'a> {
    /// The model this session decodes with.
    pub fn model(&self) -> &'a Model {
        self.model
    }

    /// Tokens currently in the KV cache, prompt included.
    pub fn n_past(&self) -> usize {
        self.cache.as_ref().map_or(0, |c| c.n_past)
    }

    /// Tokens the KV cache can hold. Zero until the first generation call.
    pub fn capacity(&self) -> usize {
        self.cache.as_ref().map_or(0, KvCache::capacity)
    }

    /// Logits from the last forward pass: the model's next-token distribution
    /// given everything generated so far.
    ///
    /// `None` before the first generation call. After [`Self::generate`] this
    /// is the continuation of the text you just got back, which is free to
    /// look at — the forward pass that produced it had already happened.
    pub fn last_logits(&self) -> Option<&[f32]> {
        self.last_logits.as_deref()
    }

    /// Drop cached history without freeing the allocation.
    ///
    /// Rarely needed, since every generation call rewinds the cache itself.
    pub fn reset(&mut self) {
        if let Some(cache) = self.cache.as_mut() {
            cache.n_past = 0;
        }
        self.last_logits = None;
    }

    /// Generate a continuation of `prompt`.
    ///
    /// The returned string is the continuation only — it does not echo the
    /// prompt. Use [`Self::generate_streaming`] when you also want token ids,
    /// the stop reason, or a per-token hook.
    ///
    /// ```
    /// use llama_rust::{fixtures, GenerateOptions, Model};
    ///
    /// # fn main() -> Result<(), llama_rust::Error> {
    /// let model = Model::from_bytes(fixtures::tiny_qwen2_gguf())?;
    /// let mut session = model.session();
    /// let opts = GenerateOptions::new(4);
    ///
    /// let text = session.generate("ab", &opts)?;
    /// assert!(!text.is_empty());
    /// // Greedy decoding is deterministic, so a second run agrees.
    /// assert_eq!(text, session.generate("ab", &opts)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn generate(
        &mut self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<String, LlamaError> {
        self.run(prompt, options, &mut |_step| StepAction::Continue)
            .map(|done| done.text)
    }

    /// Generate a continuation of `prompt` and report token ids and the stop
    /// reason alongside the text.
    ///
    /// Same work as [`Self::generate`]; use this when you need more than the
    /// string, and [`Self::generate_streaming`] when you need it token by token.
    ///
    /// ```
    /// use llama_rust::{fixtures, GenerateOptions, Model, StopReason};
    ///
    /// # fn main() -> Result<(), llama_rust::Error> {
    /// let model = Model::from_bytes(fixtures::tiny_llama_gguf())?;
    /// let opts = GenerateOptions::new(4).with_stop_at_eos(false);
    /// let done = model.session().generate_detailed("ab", &opts)?;
    /// assert_eq!(done.tokens.len(), 4);
    /// assert_eq!(done.stop, StopReason::Length);
    /// # Ok(())
    /// # }
    /// ```
    pub fn generate_detailed(
        &mut self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<Generated, LlamaError> {
        self.run(prompt, options, &mut |_step| StepAction::Continue)
    }

    /// Generate a continuation of `prompt`, calling `on_token` once per token.
    ///
    /// The callback sees the token id, its text, and the logit vector the token
    /// was drawn from, and can stop generation early by returning
    /// [`StepAction::Stop`].
    ///
    /// ```
    /// use llama_rust::{fixtures, GenerateOptions, Model, StepAction, StopReason};
    ///
    /// # fn main() -> Result<(), llama_rust::Error> {
    /// let model = Model::from_bytes(fixtures::tiny_llama_gguf())?;
    /// let mut session = model.session();
    /// let mut seen = 0usize;
    /// let done = session.generate_streaming("ab", &GenerateOptions::new(8), |step| {
    ///     assert_eq!(step.logits.len(), model.n_vocab());
    ///     seen += 1;
    ///     if step.index == 2 {
    ///         StepAction::Stop
    ///     } else {
    ///         StepAction::Continue
    ///     }
    /// })?;
    /// assert_eq!(seen, 3);
    /// assert_eq!(done.stop, StopReason::Callback);
    /// assert_eq!(done.tokens.len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn generate_streaming(
        &mut self,
        prompt: &str,
        options: &GenerateOptions,
        mut on_token: impl FnMut(Step<'_>) -> StepAction,
    ) -> Result<Generated, LlamaError> {
        self.run(prompt, options, &mut on_token)
    }

    /// Prefill `prompt` and return the next-token logits, generating nothing.
    ///
    /// The cheapest way to ask "what does the model predict here?", which is
    /// what perplexity and logit-comparison work needs.
    ///
    /// ```
    /// use llama_rust::{fixtures, Model};
    ///
    /// # fn main() -> Result<(), llama_rust::Error> {
    /// let model = Model::from_bytes(fixtures::tiny_llama_gguf())?;
    /// let logits = model.session().prompt_logits("ab")?;
    /// assert_eq!(logits.len(), model.n_vocab());
    /// # Ok(())
    /// # }
    /// ```
    pub fn prompt_logits(&mut self, prompt: &str) -> Result<Vec<f32>, LlamaError> {
        let ids = self.encode_non_empty(prompt)?;
        let n_ctx = ids.len().max(1);
        let logits = self.prefill_fresh(&ids, n_ctx)?;
        self.last_logits = Some(logits.clone());
        Ok(logits)
    }

    fn encode_non_empty(&self, prompt: &str) -> Result<Vec<u32>, LlamaError> {
        if prompt.is_empty() {
            return Err(LlamaError::EmptyPrompt);
        }
        let ids = prompt_ids(&self.model.tokenizer, prompt)?;
        if ids.is_empty() {
            return Err(LlamaError::EmptyPrompt);
        }
        Ok(ids)
    }

    /// Rewind to position zero, growing the cache first if `n_ctx` needs it,
    /// then prefill `ids`.
    fn prefill_fresh(&mut self, ids: &[u32], n_ctx: usize) -> Result<Vec<f32>, LlamaError> {
        let grow = self.cache.as_ref().is_none_or(|c| c.capacity() < n_ctx);
        if grow {
            self.cache = Some(self.model.weights.new_cache(n_ctx)?);
        }
        let Some(cache) = self.cache.as_mut() else {
            return Err(LlamaError::Shape("session kv cache".into()));
        };
        cache.n_past = 0;
        self.model.weights.prefill(cache, ids)
    }

    fn run(
        &mut self,
        prompt: &str,
        options: &GenerateOptions,
        on_token: &mut dyn FnMut(Step<'_>) -> StepAction,
    ) -> Result<Generated, LlamaError> {
        let prompt_tokens = self.encode_non_empty(prompt)?;
        let needed = prompt_tokens.len().saturating_add(options.n_predict);
        let n_ctx = match options.n_ctx {
            Some(n) if n < needed => return Err(LlamaError::Shape("n_ctx".into())),
            Some(n) => n,
            None => needed.saturating_add(1),
        };
        let mut logits = self.prefill_fresh(&prompt_tokens, n_ctx)?;

        // `self.model` is a shared reference field, so copying it out lets the
        // loop hold `&mut self.cache` and read the model at the same time.
        let model = self.model;
        let Some(cache) = self.cache.as_mut() else {
            return Err(LlamaError::Shape("session kv cache".into()));
        };
        let mut sampler = Sampler::new(options.sampling)?;
        let mut history = prompt_tokens.clone();
        let mut tokens = Vec::with_capacity(options.n_predict);
        let mut stop = StopReason::Length;
        for index in 0..options.n_predict {
            let next = sampler.sample(&logits, &history)?;
            if options.stop_at_eos && model.tokenizer.eos == Some(next) {
                stop = StopReason::Eos;
                break;
            }
            history.push(next);
            tokens.push(next);
            let action = on_token(Step {
                index,
                token: next,
                piece: model.tokenizer.decode(&[next]),
                logits: &logits,
            });
            // Advance the cache even when stopping, so `n_past` always counts
            // every token in `history` and `last_logits` stays meaningful.
            logits = model.weights.forward(cache, next)?;
            if action == StepAction::Stop {
                stop = StopReason::Callback;
                break;
            }
        }
        let text = model.tokenizer.decode(&tokens);
        self.last_logits = Some(logits);
        Ok(Generated {
            prompt_tokens,
            tokens,
            text,
            stop,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{
        generate_ctx, greedy_generate, greedy_generate_ctx, tiny_llama_gguf, tiny_q4k_embd_gguf,
        tiny_qwen2_gguf,
    };
    use crate::gguf::load_gguf;

    /// The same GGUF loaded twice: once through [`Model`], once through the raw
    /// `Llama` + `Tokenizer` pair the legacy free functions take.
    fn both_ways(bytes: Vec<u8>) -> (Model, Llama, Tokenizer) {
        let gguf = load_gguf(&bytes).expect("parse gguf");
        let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
        let raw = Llama::from_gguf(gguf).expect("weights");
        let model = Model::from_bytes(bytes).expect("model");
        (model, raw, tok)
    }

    fn model_of(bytes: Vec<u8>) -> Model {
        Model::from_bytes(bytes).expect("model")
    }

    fn run_to_end(model: &Model, prompt: &str, options: &GenerateOptions) -> Generated {
        model
            .session()
            .generate_detailed(prompt, options)
            .expect("generate")
    }

    /// The session loop and the original free-function loop are separate code
    /// on purpose, so that neither can be quietly broken by a change to the
    /// other. This pins them together token for token.
    #[test]
    fn session_generate_matches_legacy_free_functions() {
        for bytes in [tiny_llama_gguf(), tiny_qwen2_gguf(), tiny_q4k_embd_gguf()] {
            let (model, raw, tok) = both_ways(bytes);
            for n_predict in [0usize, 1, 3, 6] {
                let legacy = greedy_generate(&raw, &tok, "ab", n_predict).expect("legacy");
                let done = run_to_end(&model, "ab", &GenerateOptions::new(n_predict));
                assert_eq!(
                    tok.decode(&done.all_tokens()),
                    legacy,
                    "prompt + continuation must equal the legacy string at n_predict={n_predict}"
                );
            }
        }
    }

    /// A pinned `n_ctx` must reach the same tokens as the legacy `--n-ctx` path.
    #[test]
    fn pinned_capacity_matches_legacy_n_ctx_path() {
        let (model, raw, tok) = both_ways(tiny_llama_gguf());
        let legacy = greedy_generate_ctx(&raw, &tok, "ab", 4, Some(24)).expect("legacy");
        let done = run_to_end(&model, "ab", &GenerateOptions::new(4).with_n_ctx(24));
        assert_eq!(tok.decode(&done.all_tokens()), legacy);
    }

    /// Stochastic sampling must route the seed through [`GenerateOptions`] and
    /// land on the same tokens as the legacy sampling path.
    #[test]
    fn seeded_sampling_matches_legacy_and_repeats() {
        let params = SampleParams {
            temperature: 0.9,
            top_k: 3,
            top_p: 0.95,
            repeat_penalty: 1.2,
            seed: Some(42),
        };
        let (model, raw, tok) = both_ways(tiny_qwen2_gguf());
        let legacy = generate_ctx(&raw, &tok, "ab", 5, None, &params).expect("legacy");
        let opts = GenerateOptions::new(5).with_sampling(params);
        let mut session = model.session();
        let done = session
            .generate_streaming("ab", &opts, |_step| StepAction::Continue)
            .expect("session");
        assert_eq!(tok.decode(&done.all_tokens()), legacy);
        // A fresh draw from the same seed on a warm session repeats exactly.
        assert_eq!(session.generate("ab", &opts).expect("again"), done.text);
    }

    #[test]
    fn greedy_generate_is_deterministic_across_sessions() {
        let model = model_of(tiny_qwen2_gguf());
        let opts = GenerateOptions::new(4);
        let one = model.generate("ab", &opts).expect("model");
        let two = model.session().generate("ab", &opts).expect("session");
        assert_eq!(one, two);
    }

    /// `n_ctx = None` must size the cache exactly as the legacy path did:
    /// prompt + `n_predict` + 1.
    #[test]
    fn lazy_capacity_matches_legacy_sizing() {
        let model = model_of(tiny_llama_gguf());
        let prompt_len = model.encode("ab").expect("encode").len();
        let mut session = model.session();
        let _text = session
            .generate("ab", &GenerateOptions::new(5))
            .expect("generate");
        assert_eq!(session.capacity(), prompt_len + 5 + 1);
    }

    #[test]
    fn pinned_n_ctx_is_reused_and_bounds_checked() {
        let model = model_of(tiny_llama_gguf());
        let mut session = model.session();
        assert_eq!(session.capacity(), 0, "cache is allocated lazily");
        assert_eq!(session.n_past(), 0);

        let opts = GenerateOptions::new(3).with_n_ctx(32);
        let _first = session.generate("ab", &opts).expect("first");
        assert_eq!(session.capacity(), 32);
        let after_first = session.n_past();

        let _second = session.generate("ab", &opts).expect("second");
        assert_eq!(session.capacity(), 32, "allocation must be reused");
        assert_eq!(
            session.n_past(),
            after_first,
            "each call rewinds instead of appending"
        );

        assert!(
            matches!(
                session.generate("ab", &GenerateOptions::new(64).with_n_ctx(4)),
                Err(LlamaError::Shape(_))
            ),
            "n_ctx too small for prompt + n_predict must be refused"
        );
    }

    #[test]
    fn stop_at_eos_disabled_runs_to_length() {
        let model = model_of(tiny_llama_gguf());
        let done = run_to_end(
            &model,
            "ab",
            &GenerateOptions::new(4).with_stop_at_eos(false),
        );
        assert_eq!(done.stop, StopReason::Length);
        assert_eq!(done.tokens.len(), 4);
    }

    #[test]
    fn callback_stop_keeps_the_reported_token() {
        let model = model_of(tiny_llama_gguf());
        let mut pieces = Vec::new();
        let done = model
            .session()
            .generate_streaming("ab", &GenerateOptions::new(8), |step| {
                pieces.push(step.piece.clone());
                if step.index == 0 {
                    StepAction::Stop
                } else {
                    StepAction::Continue
                }
            })
            .expect("generate");
        assert_eq!(done.stop, StopReason::Callback);
        assert_eq!(done.tokens.len(), 1);
        assert_eq!(pieces.len(), 1);
    }

    #[test]
    fn streaming_reports_every_token_once_in_order() {
        let model = model_of(tiny_qwen2_gguf());
        let mut seen = Vec::new();
        let done = model
            .session()
            .generate_streaming(
                "ab",
                &GenerateOptions::new(6).with_stop_at_eos(false),
                |step| {
                    assert_eq!(step.index, seen.len(), "indices must be dense and ordered");
                    assert_eq!(step.logits.len(), model.n_vocab());
                    seen.push(step.token);
                    StepAction::Continue
                },
            )
            .expect("generate");
        assert_eq!(seen, done.tokens);
    }

    #[test]
    fn zero_n_predict_yields_an_empty_continuation() {
        let model = model_of(tiny_llama_gguf());
        let done = run_to_end(&model, "ab", &GenerateOptions::new(0));
        assert!(done.tokens.is_empty());
        assert_eq!(done.text, "");
        assert_eq!(done.stop, StopReason::Length);
        assert!(!done.prompt_tokens.is_empty(), "prompt is still encoded");
    }

    #[test]
    fn empty_prompt_is_refused() {
        let model = model_of(tiny_llama_gguf());
        assert!(matches!(
            model.generate("", &GenerateOptions::new(1)),
            Err(LlamaError::EmptyPrompt)
        ));
        assert!(matches!(
            model.session().prompt_logits(""),
            Err(LlamaError::EmptyPrompt)
        ));
    }

    /// `prompt_logits` must equal a raw `new_cache` + `prefill`, and must equal
    /// the logits the first generated token is drawn from.
    #[test]
    fn prompt_logits_match_raw_prefill_and_first_step() {
        let (model, raw, _tok) = both_ways(tiny_llama_gguf());
        let ids = model.encode("ab").expect("encode");
        let mut cache = raw.new_cache(ids.len()).expect("cache");
        let expected = raw.prefill(&mut cache, &ids).expect("prefill");

        let mut session = model.session();
        assert_eq!(session.prompt_logits("ab").expect("logits"), expected);
        assert_eq!(session.last_logits(), Some(expected.as_slice()));

        let mut first = None;
        let _done = model
            .session()
            .generate_streaming("ab", &GenerateOptions::new(1), |step| {
                first = Some(step.logits.to_vec());
                StepAction::Continue
            })
            .expect("generate");
        assert_eq!(first.as_deref(), Some(expected.as_slice()));
    }

    /// After generating, `last_logits` must be the next-token distribution for
    /// the whole sequence, so a raw prefill over prompt + continuation agrees.
    #[test]
    fn last_logits_continue_the_generated_sequence() {
        let (model, raw, _tok) = both_ways(tiny_llama_gguf());
        let mut session = model.session();
        let done = session
            .generate_streaming(
                "ab",
                &GenerateOptions::new(3).with_stop_at_eos(false),
                |_step| StepAction::Continue,
            )
            .expect("generate");
        let all = done.all_tokens();
        assert_eq!(session.n_past(), all.len(), "cache holds every token");
        let mut cache = raw.new_cache(all.len()).expect("cache");
        let expected = raw.prefill(&mut cache, &all).expect("prefill");
        assert_eq!(session.last_logits(), Some(expected.as_slice()));
    }

    #[test]
    fn reset_clears_history_but_keeps_the_allocation() {
        let model = model_of(tiny_llama_gguf());
        let mut session = model.session();
        let _text = session
            .generate("ab", &GenerateOptions::new(2))
            .expect("generate");
        let capacity = session.capacity();
        assert!(capacity > 0);
        assert!(session.n_past() > 0);
        session.reset();
        assert_eq!(session.n_past(), 0);
        assert_eq!(session.capacity(), capacity);
        assert!(session.last_logits().is_none());
    }

    #[test]
    fn model_exposes_weights_tokenizer_and_shape() {
        let model = model_of(tiny_llama_gguf());
        assert_eq!(model.n_vocab(), model.weights().n_vocab);
        assert_eq!(model.n_embd(), model.weights().n_embd);
        assert!(!model.tokenizer().tokens.is_empty());
        assert_eq!(model.session().model().n_vocab(), model.n_vocab());
    }

    #[test]
    fn from_path_reports_a_missing_file_as_io() {
        let loaded = Model::from_path("definitely-not-a-real-file.gguf");
        assert!(matches!(loaded, Err(LlamaError::Io(_))));
    }

    #[test]
    fn generate_options_builders_compose() {
        let opts = GenerateOptions::new(1)
            .with_n_predict(9)
            .with_n_ctx(77)
            .with_stop_at_eos(false)
            .with_sampling(SampleParams {
                temperature: 0.5,
                top_k: 2,
                top_p: 0.9,
                repeat_penalty: 1.05,
                seed: Some(3),
            });
        assert_eq!(opts.n_predict, 9);
        assert_eq!(opts.n_ctx, Some(77));
        assert!(!opts.stop_at_eos);
        assert!(!opts.sampling.is_greedy());
        assert_eq!(GenerateOptions::new(9), GenerateOptions::new(9));
    }
}
