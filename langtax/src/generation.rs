//! Text-generation conveniences layered over [`crate::Model`] and [`crate::Session`].

use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::decode::{prompt_ids, LlamaError};
use crate::sample::{SampleParams, Sampler};
use crate::{Model, Session};

/// Knobs for one generation call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerateOptions {
    /// Upper bound on generated tokens.
    pub n_predict: usize,
    /// Requested KV capacity, including prompt and continuation.
    pub n_ctx: Option<usize>,
    /// Sampling configuration.
    pub sampling: SampleParams,
    /// Stop when the tokenizer's EOS id is drawn.
    pub stop_at_eos: bool,
}

impl GenerateOptions {
    /// Greedy decoding of at most `n_predict` tokens.
    pub const fn new(n_predict: usize) -> Self {
        Self {
            n_predict,
            n_ctx: None,
            sampling: SampleParams::greedy(),
            stop_at_eos: true,
        }
    }

    /// Set the generated-token limit.
    #[must_use]
    pub const fn with_n_predict(mut self, n_predict: usize) -> Self {
        self.n_predict = n_predict;
        self
    }

    /// Set the requested KV capacity.
    #[must_use]
    pub const fn with_n_ctx(mut self, n_ctx: usize) -> Self {
        self.n_ctx = Some(n_ctx);
        self
    }

    /// Replace all sampling knobs.
    #[must_use]
    pub const fn with_sampling(mut self, sampling: SampleParams) -> Self {
        self.sampling = sampling;
        self
    }

    /// Set sampling temperature.
    #[must_use]
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.sampling.temperature = temperature;
        self
    }

    /// Set the top-k filter.
    #[must_use]
    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.sampling.top_k = top_k;
        self
    }

    /// Set the top-p filter.
    #[must_use]
    pub const fn with_top_p(mut self, top_p: f32) -> Self {
        self.sampling.top_p = top_p;
        self
    }

    /// Set the repeat penalty.
    #[must_use]
    pub const fn with_repeat_penalty(mut self, repeat_penalty: f32) -> Self {
        self.sampling.repeat_penalty = repeat_penalty;
        self
    }

    /// Set the deterministic sampling seed.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.sampling.seed = Some(seed);
        self
    }

    /// Enable or disable EOS stopping.
    #[must_use]
    pub const fn with_stop_at_eos(mut self, stop_at_eos: bool) -> Self {
        self.stop_at_eos = stop_at_eos;
        self
    }
}

/// Why generation returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// The token limit was reached.
    Length,
    /// EOS was drawn.
    Eos,
    /// The callback requested a stop.
    Callback,
}

/// A streaming callback's requested action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepAction {
    /// Generate another token.
    Continue,
    /// Return after the current token.
    Stop,
}

/// One generated token passed to a streaming callback.
#[derive(Debug)]
pub struct Step<'a> {
    /// Zero-based continuation position.
    pub index: usize,
    /// Generated token id.
    pub token: u32,
    /// This token decoded in isolation.
    pub piece: String,
    /// Logits from which the token was sampled.
    pub logits: &'a [f32],
}

/// Detailed generation result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generated {
    /// Encoded prompt ids, including an automatically inserted BOS.
    pub prompt_tokens: Vec<u32>,
    /// Generated ids, excluding EOS.
    pub tokens: Vec<u32>,
    /// Decoded continuation.
    pub text: String,
    /// Why generation stopped.
    pub stop: StopReason,
}

impl Generated {
    /// Prompt ids followed by generated ids.
    #[must_use]
    pub fn all_tokens(&self) -> Vec<u32> {
        let mut all =
            Vec::with_capacity(self.prompt_tokens.len().saturating_add(self.tokens.len()));
        all.extend_from_slice(&self.prompt_tokens);
        all.extend_from_slice(&self.tokens);
        all
    }
}

impl Model {
    /// Read and load a GGUF file.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, LlamaError> {
        let mut file = File::open(path.as_ref())?;
        let mut bytes = Vec::new();
        let _len = file.read_to_end(&mut bytes)?;
        Self::from_bytes(bytes)
    }

    /// Vocabulary size.
    #[must_use]
    pub fn n_vocab(&self) -> usize {
        self.llama.n_vocab
    }

    /// Embedding width.
    #[must_use]
    pub fn n_embd(&self) -> usize {
        self.llama.n_embd
    }

    /// Reference weights.
    #[must_use]
    pub fn weights(&self) -> &crate::Llama {
        &self.llama
    }

    /// Generate a continuation with a throwaway session.
    pub fn generate(&self, prompt: &str, options: &GenerateOptions) -> Result<String, LlamaError> {
        let ids = prompt_ids(&self.tok, prompt)?;
        if ids.is_empty() {
            return Err(LlamaError::EmptyPrompt);
        }
        let needed = ids.len().saturating_add(options.n_predict);
        let n_ctx = options.n_ctx.unwrap_or_else(|| needed.saturating_add(1));
        self.session(n_ctx)?.generate(prompt, options)
    }
}

impl Session<'_> {
    /// KV capacity in tokens.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.cache.capacity()
    }

    /// Generate and return only the continuation text.
    pub fn generate(
        &mut self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<String, LlamaError> {
        self.run(prompt, options, &mut |_step| StepAction::Continue)
            .map(|done| done.text)
    }

    /// Generate and return tokens, text, and stop reason.
    pub fn generate_detailed(
        &mut self,
        prompt: &str,
        options: &GenerateOptions,
    ) -> Result<Generated, LlamaError> {
        self.run(prompt, options, &mut |_step| StepAction::Continue)
    }

    /// Generate while calling `on_token` once per emitted token.
    pub fn generate_streaming(
        &mut self,
        prompt: &str,
        options: &GenerateOptions,
        mut on_token: impl FnMut(Step<'_>) -> StepAction,
    ) -> Result<Generated, LlamaError> {
        self.run(prompt, options, &mut on_token)
    }

    fn run(
        &mut self,
        prompt: &str,
        options: &GenerateOptions,
        on_token: &mut dyn FnMut(Step<'_>) -> StepAction,
    ) -> Result<Generated, LlamaError> {
        let prompt_tokens = prompt_ids(self.tok, prompt)?;
        if prompt_tokens.is_empty() {
            return Err(LlamaError::EmptyPrompt);
        }
        let needed = prompt_tokens.len().saturating_add(options.n_predict);
        if needed > self.cache.capacity()
            || options
                .n_ctx
                .is_some_and(|n_ctx| n_ctx < needed || n_ctx > self.cache.capacity())
        {
            return Err(LlamaError::Shape("n_ctx".into()));
        }

        let mut logits = self
            .llama
            .prompt_logits(&mut self.cache, &prompt_tokens)?
            .to_vec();
        let mut sampler = Sampler::new(options.sampling)?;
        let mut history = prompt_tokens.clone();
        let mut tokens = Vec::with_capacity(options.n_predict);
        let mut stop = StopReason::Length;

        for index in 0..options.n_predict {
            let next = sampler.sample(&logits, &history)?;
            if options.stop_at_eos && self.tok.eos == Some(next) {
                stop = StopReason::Eos;
                break;
            }
            history.push(next);
            tokens.push(next);
            let action = on_token(Step {
                index,
                token: next,
                piece: self.tok.decode(&[next]),
                logits: &logits,
            });
            logits = self.llama.forward(&mut self.cache, next)?;
            if action == StepAction::Stop {
                stop = StopReason::Callback;
                break;
            }
        }

        Ok(Generated {
            prompt_tokens,
            text: self.tok.decode(&tokens),
            tokens,
            stop,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{GenerateOptions, StepAction, StopReason};
    use crate::{fixtures, Model};

    #[test]
    fn detailed_generation_is_reproducible() {
        let model = Model::from_bytes(fixtures::tiny_qwen2_gguf()).expect("model");
        let options = GenerateOptions::new(4).with_stop_at_eos(false);
        let mut session = model.session(16).expect("session");
        let first = session
            .generate_detailed("ab", &options)
            .expect("first generation");
        let second = session
            .generate_detailed("ab", &options)
            .expect("second generation");
        assert_eq!(first, second);
        assert_eq!(first.tokens.len(), 4);
        assert_eq!(first.stop, StopReason::Length);
    }

    #[test]
    fn callback_stops_after_reported_token() {
        let model = Model::from_bytes(fixtures::tiny_qwen2_gguf()).expect("model");
        let options = GenerateOptions::new(8).with_stop_at_eos(false);
        let mut session = model.session(16).expect("session");
        let done = session
            .generate_streaming("ab", &options, |step| {
                if step.index == 2 {
                    StepAction::Stop
                } else {
                    StepAction::Continue
                }
            })
            .expect("generation");
        assert_eq!(done.tokens.len(), 3);
        assert_eq!(done.stop, StopReason::Callback);
    }
}
