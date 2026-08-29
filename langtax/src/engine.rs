//! Continuous batching over a shared [`crate::PagedKvPool`].
//!
//! Several sequences share interned KV blocks. Each [`Engine::step`] prefills
//! at most `prefill_chunk` tokens per waiting sequence, then greedily samples
//! one token for every sequence whose prompt is in KV. Sequences may be added
//! while others are already decoding. Sequences that do not fit in
//! `max_seqs` wait and are admitted when a finished sequence is retired.
//! A full pool **preempts** another
//! sequence (unique blocks drop; intern pins remain) and later re-prefills
//! plus replays already sampled greedy tokens. Greedy ids must match
//! [`crate::greedy_generate_cache`]. Not an HTTP server.

use crate::decode::{KvCache, Llama, LlamaError, PagedKvPool};
use crate::sample::argmax;
use std::collections::{BTreeMap, VecDeque};

/// Handle for one sequence on an [`Engine`].
///
/// Stable for the lifetime of the sequence, including time spent in the
/// waiting queue. Not a physical slot index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SeqId(u32);

impl SeqId {
    /// Raw identifier (not a physical slot index).
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
    /// Physical blocks in the shared intern pool. A sequence may be preempted
    /// (recompute) when allocate would exceed this cap.
    pub pool_blocks: usize,
    /// Maximum in-flight sequences. Extra [`Engine::add`]s wait.
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
    id: SeqId,
    cache: KvCache,
    prompt: Vec<u32>,
    n_predict: usize,
    generated: Vec<u32>,
    last: Vec<f32>,
    replay: usize,
    done: bool,
}

struct Waiter {
    id: SeqId,
    prompt: Vec<u32>,
    n_predict: usize,
}

impl Slot {
    fn is_prefill(&self) -> bool {
        !self.done && self.cache.n_past < self.prompt.len()
    }

    /// True when the prompt is in KV and we can sample or replay generated ids.
    ///
    /// `last` is empty after a sample that has not yet been forwarded (for
    /// example a cap error between `push` and `forward`). Replay still has to
    /// run so the sequence does not stall with `n_past >= prompt` and no logits.
    fn is_decode(&self) -> bool {
        !self.done
            && self.cache.n_past >= self.prompt.len()
            && (!self.last.is_empty() || self.replay < self.generated.len())
    }
}

/// Multi-sequence greedy decode on one interned paged-KV pool.
pub struct Engine<'a> {
    llama: &'a Llama,
    pool: PagedKvPool,
    cfg: EngineCfg,
    slots: Vec<Option<Slot>>,
    wait: VecDeque<Waiter>,
    finished: BTreeMap<SeqId, SeqOutput>,
    next_id: u32,
    preempts: u64,
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
            wait: VecDeque::new(),
            finished: BTreeMap::new(),
            next_id: 0,
            preempts: 0,
        })
    }

    /// Shared intern pool (hits are across every sequence).
    #[must_use]
    pub fn pool(&self) -> &PagedKvPool {
        &self.pool
    }

    /// Admit a prompt. Prefill starts on the next [`Engine::step`].
    ///
    /// When [`EngineCfg::max_seqs`] in-flight slots are occupied the sequence
    /// waits and is installed after a finished slot is retired.
    pub fn add(&mut self, prompt: &[u32], n_predict: usize) -> Result<SeqId, LlamaError> {
        if prompt.is_empty() {
            return Err(LlamaError::EmptyPrompt);
        }
        let needed = prompt.len().saturating_add(n_predict);
        if needed > self.cfg.n_ctx {
            return Err(LlamaError::Shape("n_ctx".into()));
        }
        let id = self.alloc_id()?;
        if self.active() >= self.cfg.max_seqs {
            self.wait.push_back(Waiter {
                id,
                prompt: prompt.to_vec(),
                n_predict,
            });
            return Ok(id);
        }
        self.install(id, prompt.to_vec(), n_predict)?;
        Ok(id)
    }

    /// One scheduler iteration: retire finished slots, admit waiters, prefill, decode.
    ///
    /// Returns how many sequences made progress (including admits after retire).
    pub fn step(&mut self) -> Result<usize, LlamaError> {
        let mut n = self.retire_done();
        n = n.saturating_add(self.admit()?);
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

    /// True when every admitted sequence is finished (running and waiting).
    #[must_use]
    pub fn all_done(&self) -> bool {
        self.wait.is_empty() && self.slots.iter().flatten().all(|s| s.done)
    }

    /// Occupied in-flight slots (not the waiting queue).
    #[must_use]
    pub fn active(&self) -> usize {
        self.slots.iter().flatten().count()
    }

    /// Sequences admitted but not yet in a running slot.
    #[must_use]
    pub fn waiting(&self) -> usize {
        self.wait.len()
    }

    /// Generated ids for `id` (empty until decode starts, including waiters).
    #[must_use]
    pub fn generated(&self, id: SeqId) -> Option<&[u32]> {
        if let Some(s) = self.slot_by_id(id) {
            return Some(s.generated.as_slice());
        }
        if let Some(out) = self.finished.get(&id) {
            return Some(out.generated.as_slice());
        }
        self.wait.iter().find(|w| w.id == id).map(|_| &[][..])
    }

    /// How many times a sequence was preempted to free pool blocks.
    #[must_use]
    pub fn preempts(&self) -> u64 {
        self.preempts
    }

    /// Remove a finished (or in-flight / waiting) sequence and return its tokens.
    pub fn take(&mut self, id: SeqId) -> Option<SeqOutput> {
        if let Some(i) = self.slot_index(id) {
            let slot = self.slots.get_mut(i).and_then(Option::take)?;
            return Some(SeqOutput {
                prompt: slot.prompt,
                generated: slot.generated,
            });
        }
        if let Some(out) = self.finished.remove(&id) {
            return Some(out);
        }
        let pos = self.wait.iter().position(|w| w.id == id)?;
        let w = self.wait.remove(pos)?;
        Some(SeqOutput {
            prompt: w.prompt,
            generated: Vec::new(),
        })
    }

    fn alloc_id(&mut self) -> Result<SeqId, LlamaError> {
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| LlamaError::Shape("seq id".into()))?;
        Ok(SeqId(id))
    }

    fn install(&mut self, id: SeqId, prompt: Vec<u32>, n_predict: usize) -> Result<(), LlamaError> {
        let cache = self.llama.new_paged_cache_on(&self.pool, self.cfg.n_ctx)?;
        let slot = Slot {
            id,
            cache,
            prompt,
            n_predict,
            generated: Vec::new(),
            last: Vec::new(),
            replay: 0,
            done: false,
        };
        for cell in &mut self.slots {
            if cell.is_none() {
                *cell = Some(slot);
                return Ok(());
            }
        }
        if self.slots.len() >= self.cfg.max_seqs {
            return Err(LlamaError::Shape("engine full".into()));
        }
        self.slots.push(Some(slot));
        Ok(())
    }

    fn retire_done(&mut self) -> usize {
        let mut n = 0usize;
        for cell in &mut self.slots {
            let Some(slot) = cell.as_ref() else {
                continue;
            };
            if !slot.done {
                continue;
            }
            let Some(slot) = cell.take() else {
                continue;
            };
            let _prev = self.finished.insert(
                slot.id,
                SeqOutput {
                    prompt: slot.prompt,
                    generated: slot.generated,
                },
            );
            n = n.saturating_add(1);
        }
        n
    }

    fn admit(&mut self) -> Result<usize, LlamaError> {
        let mut n = 0usize;
        while !self.wait.is_empty() && self.active() < self.cfg.max_seqs {
            let Some(w) = self.wait.pop_front() else {
                break;
            };
            self.install(w.id, w.prompt, w.n_predict)?;
            n = n.saturating_add(1);
        }
        Ok(n)
    }

    fn slot_by_id(&self, id: SeqId) -> Option<&Slot> {
        self.slots.iter().flatten().find(|s| s.id == id)
    }

    fn slot_index(&self, id: SeqId) -> Option<usize> {
        self.slots
            .iter()
            .position(|c| c.as_ref().is_some_and(|s| s.id == id))
    }

    /// Prefill every sequence that still has prompt tokens outside KV.
    ///
    /// A `kv page cap` error drops another sequence's unique pages and retries
    /// the same cell. A sequence that cannot fit alone still fails.
    fn prefill_ready(&mut self) -> Result<usize, LlamaError> {
        let mut n = 0usize;
        for i in 0..self.slots.len() {
            loop {
                if !self.slot(i).is_some_and(Slot::is_prefill) {
                    break;
                }
                match self.prompt_chunk_at(i) {
                    Ok(()) => {
                        n = n.saturating_add(1);
                        break;
                    }
                    Err(e) => self.on_cap(i, e)?,
                }
            }
        }
        Ok(n)
    }

    fn prompt_chunk_at(&mut self, i: usize) -> Result<(), LlamaError> {
        let llama = self.llama;
        let chunk = self.cfg.prefill_chunk;
        let slot = self.slot_mut(i)?;
        slot.last = llama
            .prompt_chunk(&mut slot.cache, &slot.prompt, chunk)?
            .to_vec();
        Ok(())
    }

    /// Replay generated ids into KV, then sample one new greedy token.
    ///
    /// Decode forwards that are ready in the same step share one batched GEMM
    /// (`Llama::forward_batch`) when they sit on this engine's intern pool.
    fn decode_ready(&mut self) -> Result<usize, LlamaError> {
        let mut n = 0usize;
        loop {
            let mut replay = Vec::new();
            for i in 0..self.slots.len() {
                let Some(cell) = self.slot(i) else {
                    continue;
                };
                if !cell.is_decode() || cell.replay >= cell.generated.len() {
                    continue;
                }
                let tok = cell
                    .generated
                    .get(cell.replay)
                    .copied()
                    .ok_or_else(|| LlamaError::Shape("engine replay token".into()))?;
                replay.push((i, tok));
            }
            if replay.is_empty() {
                break;
            }
            n = n.saturating_add(self.forward_batch_items(&replay, true)?);
        }
        let mut batch = Vec::new();
        for i in 0..self.slots.len() {
            n = n.saturating_add(self.sample_into_batch(i, &mut batch)?);
        }
        n = n.saturating_add(self.forward_batch_items(&batch, false)?);
        Ok(n)
    }

    fn sample_into_batch(
        &mut self,
        i: usize,
        batch: &mut Vec<(usize, u32)>,
    ) -> Result<usize, LlamaError> {
        if !self.slot(i).is_some_and(Slot::is_decode) {
            return Ok(0);
        }
        let generated_len = self.slot(i).map_or(0, |s| s.generated.len());
        let n_predict = self.slot(i).map_or(0, |s| s.n_predict);
        if generated_len >= n_predict {
            if let Ok(cell) = self.slot_mut(i) {
                cell.done = true;
            }
            return Ok(1);
        }
        if self.slot(i).is_some_and(|s| s.last.is_empty()) {
            return Err(LlamaError::Shape("engine logits".into()));
        }
        let eos = self.cfg.eos;
        let next = {
            let slot = self.slot_mut(i)?;
            argmax(&slot.last)
        };
        let slot = self.slot_mut(i)?;
        if eos == Some(next) {
            slot.done = true;
            return Ok(1);
        }
        slot.generated.push(next);
        slot.last.clear();
        if slot.generated.len() >= slot.n_predict {
            slot.done = true;
            return Ok(1);
        }
        batch.push((i, next));
        Ok(0)
    }

    fn forward_batch_items(
        &mut self,
        items: &[(usize, u32)],
        replay: bool,
    ) -> Result<usize, LlamaError> {
        if items.is_empty() {
            return Ok(0);
        }
        let mut rest: Vec<(usize, u32)> = items.to_vec();
        loop {
            match self.try_forward_batch(&rest, replay) {
                Ok(n) => return Ok(n),
                Err(BatchFail::Other(e)) => return Err(e),
                Err(BatchFail::Cap(except, e)) => {
                    self.on_cap(except, e)?;
                    rest.retain(|(i, _)| self.slot_batchable(*i));
                    if rest.is_empty() {
                        return Ok(1);
                    }
                }
            }
        }
    }

    /// True when this slot still has decode KV and an unforwarded generated id.
    ///
    /// After a cap preempt the victim's `n_past` is 0, so it must leave the
    /// batch. Retrying the original items would write the sampled token at
    /// position 0 instead of re-prefill + replay.
    fn slot_batchable(&self, i: usize) -> bool {
        self.slot(i)
            .is_some_and(|s| s.is_decode() && s.replay < s.generated.len())
    }

    fn try_forward_batch(
        &mut self,
        items: &[(usize, u32)],
        replay: bool,
    ) -> Result<usize, BatchFail> {
        for &(i, _) in items {
            let slot = self.slot_mut(i).map_err(BatchFail::Other)?;
            match slot.cache.prepare_append(1) {
                Ok(()) => {}
                Err(e) if is_kv_cap(&e) => return Err(BatchFail::Cap(i, e)),
                Err(e) => return Err(BatchFail::Other(e)),
            }
        }
        let mut order: Vec<(usize, u32)> = items.to_vec();
        order.sort_by_key(|(i, _)| *i);
        let indices: Vec<usize> = order.iter().map(|(i, _)| *i).collect();
        let tokens: Vec<u32> = order.iter().map(|(_, t)| *t).collect();
        let mut slots = borrow_slots_mut(&mut self.slots, &indices).map_err(BatchFail::Other)?;
        let mut caches: Vec<&mut KvCache> = slots.iter_mut().map(|s| &mut s.cache).collect();
        let rows = match self.llama.forward_batch(&mut caches, &tokens) {
            Ok(r) => r,
            Err(e) if is_kv_cap(&e) => {
                let except = items.first().map_or(0, |(i, _)| *i);
                return Err(BatchFail::Cap(except, e));
            }
            Err(e) => return Err(BatchFail::Other(e)),
        };
        if rows.len() != slots.len() {
            return Err(BatchFail::Other(LlamaError::Shape("forward batch".into())));
        }
        for (slot, row) in slots.iter_mut().zip(rows) {
            slot.last = row;
            if replay {
                slot.replay = slot.replay.saturating_add(1);
            } else {
                slot.replay = slot.generated.len();
            }
        }
        Ok(indices.len())
    }

    fn slot(&self, i: usize) -> Option<&Slot> {
        self.slots.get(i).and_then(|c| c.as_ref())
    }

    fn slot_mut(&mut self, i: usize) -> Result<&mut Slot, LlamaError> {
        self.slots
            .get_mut(i)
            .and_then(|c| c.as_mut())
            .ok_or_else(|| LlamaError::Shape("engine slot".into()))
    }

    /// Retry after dropping a victim, or propagate a non-cap / unpreemptable error.
    fn on_cap(&mut self, except: usize, err: LlamaError) -> Result<(), LlamaError> {
        if !is_kv_cap(&err) {
            return Err(err);
        }
        if self.preempt_except(except) {
            Ok(())
        } else {
            Err(err)
        }
    }

    /// Drop unique KV for the occupant with the most tokens, not `except`.
    ///
    /// Finished sequences still hold pages until `take`, so they are valid
    /// victims. A failed `ensure_write` can leave `n_past == 0` with a
    /// non-empty table; those rows count too. Interned prefixes stay in the
    /// pool. Returns whether a victim was found.
    fn preempt_except(&mut self, except: usize) -> bool {
        let best = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(idx, cell)| {
                let slot = cell.as_ref()?;
                if idx == except {
                    return None;
                }
                let n = slot.cache.n_past;
                let pages = slot.cache.page_table_len();
                if n == 0 && pages == 0 {
                    return None;
                }
                Some((idx, n, pages))
            })
            .max_by_key(|(_, n, pages)| (*n, *pages));
        let Some((idx, _, _)) = best else {
            return false;
        };
        let Ok(cell) = self.slot_mut(idx) else {
            return false;
        };
        cell.cache.preempt();
        cell.last.clear();
        cell.replay = 0;
        self.preempts = self.preempts.saturating_add(1);
        true
    }
}

fn is_kv_cap(err: &LlamaError) -> bool {
    matches!(err, LlamaError::Shape(s) if s.as_str() == "kv page cap")
}

enum BatchFail {
    Cap(usize, LlamaError),
    Other(LlamaError),
}

fn borrow_slots_mut<'a>(
    slots: &'a mut [Option<Slot>],
    indices: &[usize],
) -> Result<Vec<&'a mut Slot>, LlamaError> {
    let mut out = Vec::new();
    let mut rest = slots;
    let mut base = 0usize;
    for &i in indices {
        if i < base {
            return Err(LlamaError::Shape("engine batch order".into()));
        }
        let skip = i.saturating_sub(base);
        if skip >= rest.len() {
            return Err(LlamaError::Shape("engine slot".into()));
        }
        let (_, tail) = rest.split_at_mut(skip);
        let (one, next) = tail.split_at_mut(1);
        let cell = one
            .first_mut()
            .ok_or_else(|| LlamaError::Shape("engine slot".into()))?;
        let slot = cell
            .as_mut()
            .ok_or_else(|| LlamaError::Shape("engine slot".into()))?;
        out.push(slot);
        rest = next;
        base = i.saturating_add(1);
    }
    Ok(out)
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
    fn engine_rejects_empty_prompt() {
        let model = Llama::from_gguf(load_gguf_owned(tiny_llama_gguf()).expect("o")).expect("m");
        let mut cfg = EngineCfg::tiny();
        cfg.max_seqs = 1;
        let mut eng = Engine::new(&model, cfg).expect("eng");
        let err = eng.add(&[], 1).unwrap_err();
        assert!(matches!(err, LlamaError::EmptyPrompt));
    }

    #[test]
    fn engine_waiting_queue_runs_overflow_and_ids_still_match() {
        let tokens_a = [1u32, 2, 3, 4];
        let tokens_b = [5u32, 0, 5, 0];
        let g = load_gguf_owned(tiny_llama_gguf()).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("m");
        let exp_a = independent(&model, &tok, &tokens_a, 2);
        let exp_b = independent(&model, &tok, &tokens_b, 2);
        let mut cfg = EngineCfg::tiny();
        cfg.max_seqs = 1;
        cfg.eos = tok.eos;
        let mut eng = Engine::new(&model, cfg).expect("eng");
        let a = eng.add(&tokens_a, 2).expect("a");
        let b = eng.add(&tokens_b, 2).expect("queued");
        assert_eq!(eng.waiting(), 1);
        assert_eq!(eng.active(), 1);
        eng.run().expect("run");
        assert_eq!(eng.take(a).expect("ta").generated, exp_a);
        assert_eq!(eng.take(b).expect("tb").generated, exp_b);
        assert_eq!(eng.waiting(), 0);
    }

    #[test]
    fn engine_preempts_when_pool_is_full_and_ids_still_match() {
        let tokens_a = [1u32, 2, 3, 4];
        let tokens_b = [5u32, 0, 5, 0];
        let g = load_gguf_owned(tiny_llama_gguf()).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("m");
        let exp_a = independent(&model, &tok, &tokens_a, 2);
        let exp_b = independent(&model, &tok, &tokens_b, 2);
        let mut cfg = EngineCfg::tiny();
        cfg.pool_blocks = 3;
        cfg.max_seqs = 2;
        cfg.eos = tok.eos;
        let mut eng = Engine::new(&model, cfg).expect("eng");
        let a = eng.add(&tokens_a, 2).expect("a");
        let b = eng.add(&tokens_b, 2).expect("b");
        eng.run().expect("run");
        assert_eq!(eng.take(a).expect("ta").generated, exp_a);
        assert_eq!(eng.take(b).expect("tb").generated, exp_b);
        assert!(
            eng.preempts() > 0,
            "disjoint 4-token prompts on 3 blocks must preempt"
        );
    }

    #[test]
    fn engine_cap_without_victim_is_hard_error() {
        let tokens = [1u32, 2, 3, 4, 5, 0, 1, 2];
        let model = Llama::from_gguf(load_gguf_owned(tiny_llama_gguf()).expect("o")).expect("m");
        let mut cfg = EngineCfg::tiny();
        cfg.pool_blocks = 2;
        cfg.max_seqs = 1;
        let mut eng = Engine::new(&model, cfg).expect("eng");
        let _a = eng.add(&tokens, 1).expect("a");
        let err = eng.run().unwrap_err();
        match err {
            LlamaError::Shape(s) => assert_eq!(s, "kv page cap", "{s}"),
            other => panic!("{other}"),
        }
    }

    #[test]
    fn engine_take_releases_blocks_for_the_next_sequence() {
        let tokens_a = [1u32, 2, 3, 4];
        let tokens_b = [5u32, 0, 5, 0];
        let g = load_gguf_owned(tiny_llama_gguf()).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("m");
        let exp_b = independent(&model, &tok, &tokens_b, 0);
        let mut cfg = EngineCfg::tiny();
        cfg.pool_blocks = 2;
        cfg.max_seqs = 1;
        cfg.eos = tok.eos;
        let mut eng = Engine::new(&model, cfg).expect("eng");
        let a = eng.add(&tokens_a, 0).expect("a");
        eng.run().expect("a run");
        let _out_a = eng.take(a).expect("ta");
        let b = eng.add(&tokens_b, 0).expect("b after take");
        eng.run().expect("b run");
        assert_eq!(eng.take(b).expect("tb").generated, exp_b);
        assert_eq!(eng.preempts(), 0);
    }
}
