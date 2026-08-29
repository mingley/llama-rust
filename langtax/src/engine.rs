//! Continuous batching over a shared [`crate::PagedKvPool`].
//!
//! Several sequences share interned KV blocks. Each [`Engine::step`] prefills
//! at most `prefill_chunk` tokens per waiting sequence, then greedily samples
//! one token for every sequence whose prompt is in KV. Sequences may be added
//! while others are already decoding. Greedy token ids must match an
//! independent [`crate::greedy_generate_cache`] run. Not an HTTP server.

use crate::decode::{KvCache, Llama, LlamaError, PagedKvPool};
use crate::sample::argmax;

/// Handle for one sequence on an [`Engine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SeqId(u32);

impl SeqId {
    /// Raw slot index.
    #[must_use]
    pub fn index(self) -> usize {
        usize::try_from(self.0).unwrap_or(usize::MAX)
    }
}

/// Knobs for [`Engine::new`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineCfg {
    /// KV capacity in tokens per sequence.
    pub n_ctx: usize,
    /// Paged block size in tokens.
    pub block_size: usize,
    /// Physical blocks in the shared intern pool.
    pub pool_blocks: usize,
    /// Maximum in-flight sequences.
    pub max_seqs: usize,
    /// Prefill tokens per sequence per step (`0` = the rest of the prompt).
    pub prefill_chunk: usize,
    /// Stop sampling when this id is drawn (`None` never stops early).
    pub eos: Option<u32>,
}

impl EngineCfg {
    /// Tiny-model defaults: 16-token context, 2-token pages, 4 sequences.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            n_ctx: 16,
            block_size: 2,
            pool_blocks: 32,
            max_seqs: 4,
            prefill_chunk: 0,
            eos: None,
        }
    }
}

/// Finished prompt + continuation for a removed sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeqOutput {
    /// Prompt token ids.
    pub prompt: Vec<u32>,
    /// Generated token ids (not including the prompt).
    pub generated: Vec<u32>,
}

struct Slot {
    cache: KvCache,
    prompt: Vec<u32>,
    n_predict: usize,
    generated: Vec<u32>,
    last: Vec<f32>,
    done: bool,
}

/// Multi-sequence greedy decode on one interned paged-KV pool.
pub struct Engine<'a> {
    llama: &'a Llama,
    pool: PagedKvPool,
    cfg: EngineCfg,
    slots: Vec<Option<Slot>>,
}

impl<'a> Engine<'a> {
    /// Shared pool plus empty slots.
    pub fn new(llama: &'a Llama, cfg: EngineCfg) -> Result<Self, LlamaError> {
        if cfg.n_ctx == 0 || cfg.block_size == 0 || cfg.pool_blocks == 0 || cfg.max_seqs == 0 {
            return Err(LlamaError::Shape("engine cfg".into()));
        }
        let pool = llama.new_paged_pool(cfg.block_size, cfg.pool_blocks)?;
        Ok(Self {
            llama,
            pool,
            cfg,
            slots: Vec::new(),
        })
    }

    /// Shared intern pool (hits are across every sequence).
    #[must_use]
    pub fn pool(&self) -> &PagedKvPool {
        &self.pool
    }

    /// Admit a prompt. Prefill starts on the next [`Engine::step`].
    pub fn add(&mut self, prompt: &[u32], n_predict: usize) -> Result<SeqId, LlamaError> {
        if prompt.is_empty() {
            return Err(LlamaError::EmptyPrompt);
        }
        let needed = prompt.len().saturating_add(n_predict);
        if needed > self.cfg.n_ctx {
            return Err(LlamaError::Shape("n_ctx".into()));
        }
        let cache = self.llama.new_paged_cache_on(&self.pool, self.cfg.n_ctx)?;
        let slot = Slot {
            cache,
            prompt: prompt.to_vec(),
            n_predict,
            generated: Vec::new(),
            last: Vec::new(),
            done: false,
        };
        for (i, cell) in self.slots.iter_mut().enumerate() {
            if cell.is_none() {
                *cell = Some(slot);
                let id = u32::try_from(i).map_err(|_| LlamaError::Shape("seq id".into()))?;
                return Ok(SeqId(id));
            }
        }
        if self.slots.len() >= self.cfg.max_seqs {
            return Err(LlamaError::Shape("engine full".into()));
        }
        let id = u32::try_from(self.slots.len()).map_err(|_| LlamaError::Shape("seq id".into()))?;
        self.slots.push(Some(slot));
        Ok(SeqId(id))
    }

    /// One scheduler iteration: chunked prefill, then one decode token per ready sequence.
    ///
    /// Returns how many sequences made progress.
    pub fn step(&mut self) -> Result<usize, LlamaError> {
        let mut n = 0usize;
        n = n.saturating_add(self.prefill_ready()?);
        n = n.saturating_add(self.decode_ready()?);
        Ok(n)
    }

    /// [`Engine::step`] until every admitted sequence is finished.
    pub fn run(&mut self) -> Result<(), LlamaError> {
        while !self.all_done() {
            if self.step()? == 0 {
                return Err(LlamaError::Shape("engine stall".into()));
            }
        }
        Ok(())
    }

    /// True when every occupied slot has finished sampling.
    #[must_use]
    pub fn all_done(&self) -> bool {
        self.slots.iter().flatten().all(|s| s.done)
    }

    /// Occupied slots.
    #[must_use]
    pub fn active(&self) -> usize {
        self.slots.iter().flatten().count()
    }

    /// Generated ids for `id` (empty until decode starts).
    #[must_use]
    pub fn generated(&self, id: SeqId) -> Option<&[u32]> {
        self.slots
            .get(id.index())
            .and_then(|c| c.as_ref().map(|s| s.generated.as_slice()))
    }

    /// Remove a finished (or in-flight) sequence and return its tokens.
    pub fn take(&mut self, id: SeqId) -> Option<SeqOutput> {
        let cell = self.slots.get_mut(id.index())?;
        let slot = cell.take()?;
        Some(SeqOutput {
            prompt: slot.prompt,
            generated: slot.generated,
        })
    }

    fn prefill_ready(&mut self) -> Result<usize, LlamaError> {
        let mut n = 0usize;
        let chunk = self.cfg.prefill_chunk;
        for cell in &mut self.slots {
            let Some(slot) = cell.as_mut() else {
                continue;
            };
            if slot.done || slot.cache.n_past >= slot.prompt.len() {
                continue;
            }
            let logits = self
                .llama
                .prompt_chunk(&mut slot.cache, &slot.prompt, chunk)?;
            slot.last = logits.to_vec();
            n = n.saturating_add(1);
        }
        Ok(n)
    }

    fn decode_ready(&mut self) -> Result<usize, LlamaError> {
        let mut n = 0usize;
        let eos = self.cfg.eos;
        for cell in &mut self.slots {
            let Some(slot) = cell.as_mut() else {
                continue;
            };
            if slot.done || slot.cache.n_past < slot.prompt.len() {
                continue;
            }
            if slot.generated.len() >= slot.n_predict {
                slot.done = true;
                continue;
            }
            if slot.last.is_empty() {
                return Err(LlamaError::Shape("engine logits".into()));
            }
            let next = argmax(&slot.last);
            if eos == Some(next) {
                slot.done = true;
                n = n.saturating_add(1);
                continue;
            }
            slot.generated.push(next);
            if slot.generated.len() >= slot.n_predict {
                slot.done = true;
            } else {
                let logits = self.llama.forward_logits(&mut slot.cache, next)?;
                slot.last = logits.to_vec();
            }
            n = n.saturating_add(1);
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::{Engine, EngineCfg};
    use crate::decode::{
        greedy_generate_cache, tiny_llama4_gguf, tiny_llama_gguf, tiny_qwen3moe_gguf, Llama,
        LlamaError,
    };
    use crate::gguf::load_gguf_owned;
    use crate::tok::Tokenizer;

    fn independent(model: &Llama, tok: &Tokenizer, prompt: &[u32], n: usize) -> Vec<u32> {
        let mut cache = model.new_cache(16).expect("d");
        let mut ids = prompt.to_vec();
        let _s = greedy_generate_cache(model, tok, &mut cache, &mut ids, n).expect("g");
        ids.split_off(prompt.len())
    }

    #[test]
    fn engine_two_sequences_match_independent_and_intern() {
        let tokens_a = [1u32, 2, 3, 4];
        let tokens_b = [1u32, 2, 0, 1];
        for bytes in [tiny_llama_gguf(), tiny_qwen3moe_gguf(), tiny_llama4_gguf()] {
            let g = load_gguf_owned(bytes).expect("owned");
            let tok = Tokenizer::from_gguf(&g).expect("tok");
            let model = Llama::from_gguf(g).expect("m");
            let exp_a = independent(&model, &tok, &tokens_a, 2);
            let exp_b = independent(&model, &tok, &tokens_b, 2);
            let mut cfg = EngineCfg::tiny();
            cfg.eos = tok.eos;
            let mut eng = Engine::new(&model, cfg).expect("eng");
            let a = eng.add(&tokens_a, 2).expect("a");
            let _n = eng.step().expect("prefill a");
            let b = eng.add(&tokens_b, 2).expect("b joins");
            eng.run().expect("run");
            assert_eq!(eng.take(a).expect("ta").generated, exp_a);
            assert_eq!(eng.take(b).expect("tb").generated, exp_b);
            assert!(
                eng.pool().hits() > 0,
                "B must intern-hit A's [1,2] full block"
            );
        }
    }

    #[test]
    fn engine_chunked_prefill_matches_full_and_interleaves() {
        let tokens = [1u32, 2, 3, 4];
        let g = load_gguf_owned(tiny_llama_gguf()).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("m");
        let exp = independent(&model, &tok, &tokens, 2);
        let mut cfg = EngineCfg::tiny();
        cfg.prefill_chunk = 2;
        cfg.eos = tok.eos;
        let mut eng = Engine::new(&model, cfg).expect("eng");
        let a = eng.add(&tokens, 2).expect("a");
        let n = eng.step().expect("chunk");
        assert_eq!(n, 1);
        assert_eq!(eng.generated(a).expect("g").len(), 0);
        eng.run().expect("run");
        assert_eq!(eng.take(a).expect("t").generated, exp);
    }

    #[test]
    fn engine_rejects_empty_prompt_and_full_slots() {
        let model = Llama::from_gguf(load_gguf_owned(tiny_llama_gguf()).expect("o")).expect("m");
        let mut cfg = EngineCfg::tiny();
        cfg.max_seqs = 1;
        let mut eng = Engine::new(&model, cfg).expect("eng");
        let err = eng.add(&[], 1).unwrap_err();
        assert!(matches!(err, LlamaError::EmptyPrompt));
        let _a = eng.add(&[1, 2], 1).expect("one");
        let err = eng.add(&[3, 4], 1).unwrap_err();
        match err {
            LlamaError::Shape(s) => assert!(s.contains("full"), "{s}"),
            other => panic!("{other}"),
        }
    }
}
