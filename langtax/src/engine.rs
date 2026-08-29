//! Continuous batching over a shared [`crate::PagedKvPool`].
//!
//! Several sequences share interned KV blocks. Each [`Engine::step`] prefills
//! at most `prefill_chunk` tokens per waiting sequence, then greedily samples
//! one token for every sequence whose prompt is in KV. [`EngineCfg::decode_first`]
//! holds leftover prefill while any live sequence is already decoding (same
//! policy as `expertvm schedule --decode-first`). Sequences may be added
//! while others are already decoding. Sequences that do not fit in
//! `max_seqs` wait and are admitted when a finished sequence is retired.
//! Prefill chunks and replay tokens that are ready in the same step share one
//! batched GEMM (`Llama::prefill_batch`); new greedy tokens share
//! `Llama::forward_batch`. One attached [`expertvm::LiveStore`] is parked on
//! the first cache of each GEMM so MoE serving stays on the batched path.
//! Opt-in [`Engine::enable_moe_trace`] records per-sequence [`expertvm::Trace`]
//! events from those GEMMs (not a sequential fallback). Markov copy-forward ∪
//! lookback-2 prefetch runs **before** grouped expert GEMM so H2D of L+1 can
//! overlap this layer's compute. After each GEMM the
//! parked store sticky-pins last-used ∪ Markov experts (`slots.saturating_sub(1)`;
//! `slots == 1` pins nothing so demand paging can evict). Multi-GPU SimulatedGpuStore
//! then runs [`plan_placement`](expertvm::plan_placement) on those pins: D2D onto GPU0
//! when weight bytes beat activation volume, otherwise leave them on the striped home.
//! SimulatedGpuStore scores sample the virtual clock at each generated token
//! (`ttft_ns` / `itl_ns` / `ns_per_token`; not `$/M tokens`). Default GPU
//! stores capture per-page GEMM graphs (`Engine::graph_launches`).
//! `GpuStoreCfg` knobs (`host_func`, blocking streams, `sync_alloc`, mempool,
//! `vmm_page`, pageable H2D, `SetAccessedBy`, legacy NULL, stream priority,
//! graph update/clone, timing events) are the same mechanical CUDA surface as
//! `expertvm sim`. Default pinned async stays decode identity.
//! [`EngineCfg::slo_reject`] drops waiters whose gpu-sim queue wait already
//! meets [`EngineCfg::ttft_slo_ns`]. [`EngineCfg::itl_slo_ns`] counts later-token
//! gaps that miss the ITL budget (`Engine::itl_slo_miss`; does not drop).
//! A full pool **preempts** another sequence (unique blocks drop; intern pins remain)
//! and later re-prefills
//! plus replays already sampled greedy tokens. Greedy ids must match
//! [`crate::greedy_generate_cache`]. `gguf_gemv serve --engine` is the HTTP
//! loop around this scheduler.

use crate::decode::{KvCache, Llama, LlamaError, PagedKvPool, PrefetchChain};
use crate::sample::argmax;
use expertvm::{DeviceId, ExpertKey, ExpertStore, LiveStore, Score, StoreMetrics, StreamId, Trace};
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
    /// Hold leftover prefill while any live sequence is already decoding.
    ///
    /// New waiters still admit into a slot; their prompt stays unforwarded
    /// until every in-flight decode finishes. Default `false` interleaves
    /// chunked prefill with decode in the same step.
    pub decode_first: bool,
    /// Drop waiting sequences whose gpu-sim queue wait already meets
    /// [`EngineCfg::ttft_slo_ns`]. No-op without a SimulatedGpuStore clock.
    pub slo_reject: bool,
    /// Virtual-ns TTFT budget for [`EngineCfg::slo_reject`]. `None` never drops.
    pub ttft_slo_ns: Option<u64>,
    /// Later-token gap budget. Misses increment [`Engine::itl_slo_miss`].
    ///
    /// Does not drop sequences. No-op without a SimulatedGpuStore clock.
    pub itl_slo_ns: Option<u64>,
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
            decode_first: false,
            slo_reject: false,
            ttft_slo_ns: None,
            itl_slo_ns: None,
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
    arrival_ns: Option<u64>,
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

/// Counters for cross-sequence GEMMs. Not wall-clock tok/s and not `$/M tokens`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EngineStats {
    /// [`Engine::step`] calls that returned.
    pub steps: u64,
    /// Tokens in a shared-pool GEMM of two or more sequences.
    pub gemm_tokens: u64,
    /// Largest token count in one of those GEMMs.
    pub gemm_peak: usize,
    /// Tokens forwarded one sequence at a time.
    pub serial_tokens: u64,
}

/// Multi-sequence greedy decode on one interned paged-KV pool.
///
/// [`Engine::attach_expert_store`] parks one store on the first cache of each
/// batched GEMM so routed experts come from DirectStore / CachedStore /
/// TieredStore / SimulatedGpuStore instead of the GGUF blob. DirectStore ids match the blob
/// Engine. Two per-cache stores still fall back inside `forward_batch`.
/// [`Engine::enable_moe_trace`] records router events per [`SeqId`].
/// After each GEMM, [`LiveStore::pin_hot`] keeps last-used ∪ predicted experts
/// resident (`pin_budget` = `slots - 1`) without blocking demand paging.
/// A multi-GPU [`LiveStore::Simulated`] then calls [`LiveStore::place_hot`] on
/// those pins: [`expertvm::plan_placement`] D2Ds onto GPU0 when expert bytes beat
/// [`expertvm::DECODE_ACTIVATION_BYTES`] times fan-in times reuse, else activations
/// stay on the striped home (`StoreMetrics::dispatches`). [`LiveStore::migrate`]
/// stays unconditional. 1-GPU profiles skip. SimulatedGpuStore token samples
/// fill [`Engine::seq_ttft_ns`] / [`Engine::seq_itl_ns`] and the score line.
/// Default GPU GEMMs capture graphs ([`Engine::graph_launches`]).
/// [`EngineCfg::itl_slo_ns`] counts later-token misses ([`Engine::itl_slo_miss`]).
pub struct Engine<'a> {
    llama: &'a Llama,
    pool: PagedKvPool,
    cfg: EngineCfg,
    slots: Vec<Option<Slot>>,
    wait: VecDeque<Waiter>,
    finished: BTreeMap<SeqId, SeqOutput>,
    traces: BTreeMap<SeqId, Trace>,
    next_id: u32,
    preempts: u64,
    rejected: u64,
    itl_slo_miss: u64,
    stats: EngineStats,
    expert_store: Option<LiveStore>,
    prefetch: PrefetchChain,
    /// Online keep-hot counts per key (no JSONL future leak).
    hot_reuse: BTreeMap<ExpertKey, u64>,
    trace: bool,
    /// gpu-sim clock at each newly sampled greedy token, per sequence.
    seq_token_ns: BTreeMap<SeqId, Vec<u64>>,
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
            traces: BTreeMap::new(),
            next_id: 0,
            preempts: 0,
            rejected: 0,
            itl_slo_miss: 0,
            stats: EngineStats::default(),
            expert_store: None,
            prefetch: PrefetchChain::default(),
            hot_reuse: BTreeMap::new(),
            trace: false,
            seq_token_ns: BTreeMap::new(),
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
            let arrival_ns = self.sample_sim_clock()?;
            self.wait.push_back(Waiter {
                id,
                prompt: prompt.to_vec(),
                n_predict,
                arrival_ns,
            });
            return Ok(id);
        }
        self.install(id, prompt.to_vec(), n_predict)?;
        Ok(id)
    }

    /// One scheduler iteration: retire finished slots, admit waiters, prefill, decode.
    ///
    /// Prefill chunks and replay tokens that are ready together share one
    /// [`Llama::prefill_batch`] GEMM. New greedy tokens are sampled afterward
    /// and forwarded with [`Llama::forward_batch`].
    ///
    /// Returns how many sequences made progress (including admits after retire).
    pub fn step(&mut self) -> Result<usize, LlamaError> {
        let mut n = self.retire_done();
        n = n.saturating_add(self.admit()?);
        n = n.saturating_add(self.prefill_and_replay()?);
        n = n.saturating_add(self.decode_ready()?);
        self.stats.steps = self.stats.steps.saturating_add(1);
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

    /// KV tokens in a live slot. Waiters and retired sequences are `None`.
    #[must_use]
    pub fn n_past(&self, id: SeqId) -> Option<usize> {
        self.slot_by_id(id).map(|s| s.cache.n_past)
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

    /// Waiters dropped by [`EngineCfg::slo_reject`] (gpu-sim TTFT budget).
    #[must_use]
    pub fn rejected(&self) -> u64 {
        self.rejected
    }

    /// Later-token gaps that missed [`EngineCfg::itl_slo_ns`].
    #[must_use]
    pub fn itl_slo_miss(&self) -> u64 {
        self.itl_slo_miss
    }

    /// Captured GEMM graph launches on an attached SimulatedGpuStore.
    #[must_use]
    pub fn graph_launches(&self) -> u64 {
        self.live_store().map_or(0, LiveStore::graph_launches)
    }

    /// Parked-exec graph updates on an attached SimulatedGpuStore.
    #[must_use]
    pub fn graph_updates(&self) -> u64 {
        self.live_store().map_or(0, LiveStore::graph_updates)
    }

    /// Graph clones before instantiate on an attached SimulatedGpuStore.
    #[must_use]
    pub fn graph_clones(&self) -> u64 {
        self.live_store().map_or(0, LiveStore::graph_clones)
    }

    /// Timing-on copy elapsed ns on an attached SimulatedGpuStore.
    #[must_use]
    pub fn copy_elapsed_ns(&self) -> u64 {
        self.live_store().map_or(0, LiveStore::copy_elapsed_ns)
    }

    /// Stream priority on an attached SimulatedGpuStore (`None` unless GPU).
    #[must_use]
    pub fn gpu_stream_priority(&self, device: DeviceId, stream: StreamId) -> Option<i32> {
        self.live_store()
            .and_then(|s| s.stream_priority(device, stream))
    }

    /// True when any SimulatedGpuStore page has `SetAccessedBy` on `device`.
    #[must_use]
    pub fn gpu_any_accessed_by(&self, device: DeviceId) -> bool {
        self.live_store()
            .is_some_and(|s| s.any_page_accessed_by(device))
    }

    /// Unused bytes in GPU0's default mempool (`None` unless SimulatedGpuStore).
    pub fn gpu_pool_cached(&mut self) -> Result<Option<u64>, LlamaError> {
        self.reclaim_parked_stores();
        match self.expert_store.as_mut() {
            Some(s) => s.default_pool_cached().map_err(LlamaError::from),
            None => Ok(None),
        }
    }

    fn live_store(&self) -> Option<&LiveStore> {
        self.expert_store.as_ref().or_else(|| {
            self.slots
                .iter()
                .flatten()
                .find_map(|s| s.cache.attached_store())
        })
    }

    /// Cross-sequence GEMM counters for this engine.
    #[must_use]
    pub fn stats(&self) -> &EngineStats {
        &self.stats
    }

    /// True when `id` has finished sampling (slot still held or already retired).
    #[must_use]
    pub fn is_finished(&self, id: SeqId) -> bool {
        if self.finished.contains_key(&id) {
            return true;
        }
        self.slot_by_id(id).is_some_and(|s| s.done)
    }

    /// Waiting queue or a live slot that still needs prefill/decode.
    #[must_use]
    pub fn has_runnable(&self) -> bool {
        !self.wait.is_empty() || self.slots.iter().flatten().any(|s| !s.done)
    }

    /// Decode routed experts from `store` on every Engine GEMM.
    ///
    /// The store is parked on the first cache of each batched prefill/decode
    /// so [`Llama::prefill_batch`] / [`Llama::forward_batch`] stay on the
    /// shared-pool path. Markov prefetch (copy-forward ∪ lookback-2) is parked
    /// on the same cache so CachedStore can prefetch across GEMMs. Omit this
    /// to keep the GGUF blob FFN.
    pub fn attach_expert_store(&mut self, store: LiveStore) {
        self.reclaim_parked_stores();
        self.expert_store = Some(store);
        self.hot_reuse.clear();
    }

    /// Remove the attached store, including one parked on a live cache.
    pub fn take_expert_store(&mut self) -> Option<LiveStore> {
        self.reclaim_parked_stores();
        self.hot_reuse.clear();
        self.expert_store.take()
    }

    /// Counters from the attached store, if any.
    #[must_use]
    pub fn expert_store_metrics(&self) -> Option<StoreMetrics> {
        if let Some(s) = self.expert_store.as_ref() {
            return Some(ExpertStore::metrics(s));
        }
        self.slots
            .iter()
            .flatten()
            .find_map(|s| s.cache.expert_store_metrics())
    }

    /// gpu-sim score when the attached store is [`LiveStore::Simulated`].
    ///
    /// DirectStore / CachedStore return `Ok(None)`. Token-boundary samples
    /// fill `ttft_ns` / `itl_ns` / `ns_per_token` when any sequence produced
    /// a greedy token. Not `$/M tokens`.
    pub fn expert_store_score(&mut self) -> Result<Option<Score>, LlamaError> {
        self.reclaim_parked_stores();
        let Some(s) = self.expert_store.as_mut() else {
            return Ok(None);
        };
        let Some(mut score) = s.score().map_err(LlamaError::from)? else {
            return Ok(None);
        };
        score = attach_token_latencies(score, &self.seq_token_ns);
        Ok(Some(score))
    }

    /// gpu-sim clock of this sequence's first generated token (`None` if blob / CPU store).
    #[must_use]
    pub fn seq_ttft_ns(&self, id: SeqId) -> Option<u64> {
        self.seq_token_ns.get(&id).and_then(|v| v.first().copied())
    }

    /// Mean inter-token gap for `id` after TTFT (`None` unless two generated tokens).
    #[must_use]
    pub fn seq_itl_ns(&self, id: SeqId) -> Option<u64> {
        self.seq_token_ns.get(&id).and_then(|v| mean_itl(v))
    }

    /// Record MoE router events on every live and later-admitted sequence.
    ///
    /// [`expertvm::ExpertAccess::sequence`] is the [`SeqId`] integer. Call
    /// before the first [`Engine::step`]. [`Engine::take_moe_trace`] returns
    /// the events after run, including after [`Engine::take`]. Does not change
    /// [`SeqOutput`].
    pub fn enable_moe_trace(&mut self) {
        self.trace = true;
        for slot in self.slots.iter_mut().flatten() {
            slot.cache.enable_moe_trace(u64::from(slot.id.0));
        }
    }

    /// Take recorded router events for `id`.
    ///
    /// Live caches are drained first; retired / taken sequences use the bank
    /// filled by [`Engine::take`] and slot retirement.
    pub fn take_moe_trace(&mut self, id: SeqId) -> Option<Trace> {
        if let Some(i) = self.slot_index(id) {
            if let Ok(slot) = self.slot_mut(i) {
                let t = slot.cache.take_moe_trace();
                if !t.events.is_empty() {
                    return Some(t);
                }
            }
        }
        self.traces.remove(&id)
    }

    fn bank_trace(&mut self, id: SeqId, cache: &mut KvCache) {
        let t = cache.take_moe_trace();
        if t.events.is_empty() {
            return;
        }
        let _prev = self.traces.insert(id, t);
    }

    fn reclaim_parked_stores(&mut self) {
        for cell in &mut self.slots {
            if let Some(slot) = cell.as_mut() {
                if let Some(store) = slot.cache.take_expert_store() {
                    self.expert_store = Some(store);
                }
            }
        }
    }

    fn reclaim_from(&mut self, cache: &mut KvCache) {
        if let Some(store) = cache.take_expert_store() {
            self.expert_store = Some(store);
        }
    }

    fn park_store_on(&mut self, idx: usize) {
        let store = self.expert_store.take();
        let chain = core::mem::take(&mut self.prefetch);
        match self.slot_mut(idx) {
            Ok(slot) => {
                if let Some(store) = store {
                    slot.cache.attach_expert_store(store);
                }
                slot.cache.set_prefetch_chain(chain);
            }
            Err(_) => {
                self.expert_store = store;
                self.prefetch = chain;
            }
        }
    }

    fn unpark_store_from(&mut self, idx: usize) {
        let (store, chain) = match self.slot_mut(idx) {
            Ok(slot) => (
                slot.cache.take_expert_store(),
                slot.cache.take_prefetch_chain(),
            ),
            Err(_) => return,
        };
        if let Some(store) = store {
            self.expert_store = Some(store);
        }
        self.prefetch = chain;
    }

    fn with_store_parked<T>(&mut self, idx: usize, f: impl FnOnce(&mut Self) -> T) -> T {
        self.park_store_on(idx);
        let out = f(self);
        self.unpark_store_from(idx);
        self.pin_predicted();
        out
    }

    /// Sticky-pin last-used ∪ Markov keys, then place them (move vs dispatch).
    fn pin_predicted(&mut self) {
        let budget = match self.expert_store.as_ref() {
            Some(s) => s.pin_budget(),
            None => return,
        };
        if budget == 0 {
            return;
        }
        let fan_in = self.prefetch.last_fan_in();
        let keys = self.prefetch.keep_hot_keys();
        let pinned = {
            let Some(store) = self.expert_store.as_mut() else {
                return;
            };
            store.unpin_all();
            let mut pinned = Vec::new();
            for key in keys {
                if pinned.len() >= budget {
                    break;
                }
                if let Ok(()) = store.pin_hot(&[key]) {
                    if store.is_pinned(key) {
                        pinned.push(key);
                    }
                }
            }
            pinned
        };
        let mut jobs = Vec::new();
        for key in pinned {
            let slot = self.hot_reuse.entry(key).or_insert(0);
            *slot = slot.saturating_add(1);
            jobs.push((key, *slot));
        }
        let Some(store) = self.expert_store.as_mut() else {
            return;
        };
        for (key, reuse) in jobs {
            store.place_hot(key, reuse, fan_in);
        }
    }

    fn note_gemm(&mut self, n: usize) {
        self.stats.gemm_tokens = self
            .stats
            .gemm_tokens
            .saturating_add(u64::try_from(n).unwrap_or(u64::MAX));
        self.stats.gemm_peak = self.stats.gemm_peak.max(n);
    }

    fn note_serial(&mut self, n: usize) {
        self.stats.serial_tokens = self
            .stats
            .serial_tokens
            .saturating_add(u64::try_from(n).unwrap_or(u64::MAX));
    }

    /// Remove a finished (or in-flight / waiting) sequence and return its tokens.
    pub fn take(&mut self, id: SeqId) -> Option<SeqOutput> {
        if let Some(i) = self.slot_index(id) {
            let mut slot = self.slots.get_mut(i).and_then(Option::take)?;
            self.reclaim_from(&mut slot.cache);
            self.bank_trace(slot.id, &mut slot.cache);
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
        let mut cache = self.llama.new_paged_cache_on(&self.pool, self.cfg.n_ctx)?;
        if self.trace {
            cache.enable_moe_trace(u64::from(id.0));
        }
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
        let mut reclaimed = None;
        let mut banked = Vec::new();
        for cell in &mut self.slots {
            let Some(slot) = cell.as_ref() else {
                continue;
            };
            if !slot.done {
                continue;
            }
            let Some(mut slot) = cell.take() else {
                continue;
            };
            if let Some(store) = slot.cache.take_expert_store() {
                reclaimed = Some(store);
            }
            banked.push((slot.id, slot.cache.take_moe_trace()));
            let _prev = self.finished.insert(
                slot.id,
                SeqOutput {
                    prompt: slot.prompt,
                    generated: slot.generated,
                },
            );
            n = n.saturating_add(1);
        }
        if let Some(store) = reclaimed {
            self.expert_store = Some(store);
        }
        for (id, t) in banked {
            if t.events.is_empty() {
                continue;
            }
            let _prev = self.traces.insert(id, t);
        }
        n
    }

    fn admit(&mut self) -> Result<usize, LlamaError> {
        let mut n = 0usize;
        while !self.wait.is_empty() && self.active() < self.cfg.max_seqs {
            let Some(w) = self.wait.pop_front() else {
                break;
            };
            if self.slo_late(&w)? {
                let _prev = self.finished.insert(
                    w.id,
                    SeqOutput {
                        prompt: w.prompt,
                        generated: Vec::new(),
                    },
                );
                self.rejected = self.rejected.saturating_add(1);
                n = n.saturating_add(1);
                continue;
            }
            self.install(w.id, w.prompt, w.n_predict)?;
            n = n.saturating_add(1);
        }
        Ok(n)
    }

    fn slo_late(&mut self, w: &Waiter) -> Result<bool, LlamaError> {
        if !self.cfg.slo_reject {
            return Ok(false);
        }
        let Some(slo) = self.cfg.ttft_slo_ns else {
            return Ok(false);
        };
        let Some(arrival) = w.arrival_ns else {
            return Ok(false);
        };
        let Some(now) = self.sample_sim_clock()? else {
            return Ok(false);
        };
        Ok(now.saturating_sub(arrival) >= slo)
    }

    fn slot_by_id(&self, id: SeqId) -> Option<&Slot> {
        self.slots.iter().flatten().find(|s| s.id == id)
    }

    fn slot_index(&self, id: SeqId) -> Option<usize> {
        self.slots
            .iter()
            .position(|c| c.as_ref().is_some_and(|s| s.id == id))
    }

    /// Prefill suffixes and unforwarded replay ids that are ready this step.
    ///
    /// Two or more jobs share one [`Llama::prefill_batch`] GEMM, including a
    /// mix of prefill chunks and single-token replays. A `kv page cap` error
    /// drops another sequence's unique pages and retries the remaining set.
    fn prefill_and_replay(&mut self) -> Result<usize, LlamaError> {
        let jobs = self.collect_compute_jobs();
        if jobs.len() <= 1 {
            return self.compute_one_at_a_time();
        }
        self.compute_batch_items(&jobs)
    }

    fn collect_compute_jobs(&self) -> Vec<(usize, bool)> {
        let hold = self.hold_prefill();
        let mut jobs = Vec::new();
        for i in 0..self.slots.len() {
            let Some(s) = self.slot(i) else {
                continue;
            };
            if s.is_prefill() {
                if !hold {
                    jobs.push((i, false));
                }
            } else if s.is_decode() && s.replay < s.generated.len() {
                jobs.push((i, true));
            }
        }
        jobs
    }

    fn hold_prefill(&self) -> bool {
        self.cfg.decode_first && self.slots.iter().flatten().any(Slot::is_decode)
    }

    fn compute_one_at_a_time(&mut self) -> Result<usize, LlamaError> {
        let mut n = self.prefill_one_at_a_time()?;
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
        Ok(n)
    }

    fn compute_batch_items(&mut self, jobs: &[(usize, bool)]) -> Result<usize, LlamaError> {
        let mut rest = jobs.to_vec();
        loop {
            if rest.len() <= 1 {
                return self.compute_one_at_a_time();
            }
            match self.try_compute_batch(&rest) {
                Ok(n) => return Ok(n),
                Err(BatchFail::Other(e)) => return Err(e),
                Err(BatchFail::Cap(except, e)) => {
                    let victim = self.on_cap(except, e)?;
                    rest.retain(|(i, _)| *i != victim && self.slot_compute(*i));
                    if rest.is_empty() {
                        return Ok(1);
                    }
                }
            }
        }
    }

    fn slot_compute(&self, i: usize) -> bool {
        let hold = self.hold_prefill();
        self.slot(i).is_some_and(|s| {
            (s.is_prefill() && !hold) || (s.is_decode() && s.replay < s.generated.len())
        })
    }

    fn try_compute_batch(&mut self, jobs: &[(usize, bool)]) -> Result<usize, BatchFail> {
        let chunk = self.cfg.prefill_chunk;
        let mut owned: Vec<(usize, Vec<u32>, bool)> = Vec::new();
        for &(i, replay) in jobs {
            let slot = self.slot_mut(i).map_err(BatchFail::Other)?;
            let part = if replay {
                let tok = slot
                    .generated
                    .get(slot.replay)
                    .copied()
                    .ok_or_else(|| LlamaError::Shape("engine replay token".into()))
                    .map_err(BatchFail::Other)?;
                vec![tok]
            } else {
                slot.cache
                    .prompt_suffix(&slot.prompt, chunk)
                    .map_err(BatchFail::Other)?
                    .to_vec()
            };
            match slot.cache.prepare_append(part.len()) {
                Ok(()) => {}
                Err(e) if is_kv_cap(&e) => return Err(BatchFail::Cap(i, e)),
                Err(e) => return Err(BatchFail::Other(e)),
            }
            owned.push((i, part, replay));
        }
        owned.sort_by_key(|(i, _, _)| *i);
        let order: Vec<usize> = owned.iter().map(|(i, _, _)| *i).collect();
        let first = *order
            .first()
            .ok_or_else(|| BatchFail::Other(LlamaError::Shape("engine batch".into())))?;
        let rows = self.with_store_parked(first, |eng| {
            let mut slots = borrow_slots_mut(&mut eng.slots, &order).map_err(BatchFail::Other)?;
            let mut caches: Vec<&mut KvCache> = slots.iter_mut().map(|s| &mut s.cache).collect();
            let groups: Vec<&[u32]> = owned.iter().map(|(_, t, _)| t.as_slice()).collect();
            match eng.llama.prefill_batch(&mut caches, &groups) {
                Ok(r) => Ok(r),
                Err(e) if is_kv_cap(&e) => {
                    let except = jobs.first().map_or(0, |(i, _)| *i);
                    Err(BatchFail::Cap(except, e))
                }
                Err(e) => Err(BatchFail::Other(e)),
            }
        })?;
        if rows.len() != order.len() {
            return Err(BatchFail::Other(LlamaError::Shape("prefill batch".into())));
        }
        for ((i, toks, replay), row) in owned.iter().zip(rows) {
            let slot = self.slot_mut(*i).map_err(BatchFail::Other)?;
            slot.last = row;
            if *replay {
                slot.replay = slot.replay.saturating_add(toks.len());
            }
        }
        let ntok: usize = owned.iter().map(|(_, t, _)| t.len()).sum();
        self.note_gemm(ntok);
        Ok(order.len())
    }

    fn prefill_one_at_a_time(&mut self) -> Result<usize, LlamaError> {
        if self.hold_prefill() {
            return Ok(0);
        }
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
                    Err(e) => {
                        let _victim = self.on_cap(i, e)?;
                    }
                }
            }
        }
        Ok(n)
    }

    fn prompt_chunk_at(&mut self, i: usize) -> Result<(), LlamaError> {
        self.with_store_parked(i, |eng| {
            let llama = eng.llama;
            let chunk = eng.cfg.prefill_chunk;
            let n = {
                let slot = eng.slot_mut(i)?;
                let before = slot.cache.n_past;
                slot.last = llama
                    .prompt_chunk(&mut slot.cache, &slot.prompt, chunk)?
                    .to_vec();
                slot.cache.n_past.saturating_sub(before)
            };
            eng.note_serial(n);
            Ok(())
        })
    }

    /// Sample one new greedy token per decode-ready sequence, then forward.
    ///
    /// Replay of already sampled ids happens in [`Engine::prefill_and_replay`]
    /// so a prefill chunk and a replay token can share one GEMM.
    fn decode_ready(&mut self) -> Result<usize, LlamaError> {
        let mut n = 0usize;
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
        let (id, finished) = {
            let slot = self.slot_mut(i)?;
            if eos == Some(next) {
                slot.done = true;
                return Ok(1);
            }
            slot.generated.push(next);
            slot.last.clear();
            let finished = slot.generated.len() >= slot.n_predict;
            if finished {
                slot.done = true;
            }
            (slot.id, finished)
        };
        if !finished {
            batch.push((i, next));
        }
        self.record_seq_token(id)?;
        Ok(usize::from(finished))
    }

    fn record_seq_token(&mut self, id: SeqId) -> Result<(), LlamaError> {
        let Some(ns) = self.sample_sim_clock()? else {
            return Ok(());
        };
        self.note_itl_slo(id, ns);
        self.seq_token_ns.entry(id).or_default().push(ns);
        Ok(())
    }

    fn note_itl_slo(&mut self, id: SeqId, ns: u64) {
        let Some(slo) = self.cfg.itl_slo_ns else {
            return;
        };
        let Some(prev) = self.seq_token_ns.get(&id).and_then(|v| v.last()).copied() else {
            return;
        };
        if ns.saturating_sub(prev) > slo {
            self.itl_slo_miss = self.itl_slo_miss.saturating_add(1);
        }
    }

    fn sample_sim_clock(&mut self) -> Result<Option<u64>, LlamaError> {
        self.reclaim_parked_stores();
        match self.expert_store.as_mut() {
            Some(s) => s.clock_ns().map_err(LlamaError::from),
            None => Ok(None),
        }
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
                    let victim = self.on_cap(except, e)?;
                    rest.retain(|(i, _)| *i != victim && self.slot_batchable(*i));
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
        let first = *indices
            .first()
            .ok_or_else(|| BatchFail::Other(LlamaError::Shape("engine batch".into())))?;
        let rows = self.with_store_parked(first, |eng| {
            let mut slots = borrow_slots_mut(&mut eng.slots, &indices).map_err(BatchFail::Other)?;
            let mut caches: Vec<&mut KvCache> = slots.iter_mut().map(|s| &mut s.cache).collect();
            match eng.llama.forward_batch(&mut caches, &tokens) {
                Ok(r) => Ok(r),
                Err(e) if is_kv_cap(&e) => {
                    let except = items.first().map_or(0, |(i, _)| *i);
                    Err(BatchFail::Cap(except, e))
                }
                Err(e) => Err(BatchFail::Other(e)),
            }
        })?;
        if rows.len() != indices.len() {
            return Err(BatchFail::Other(LlamaError::Shape("forward batch".into())));
        }
        for (i, row) in indices.iter().zip(rows) {
            let slot = self.slot_mut(*i).map_err(BatchFail::Other)?;
            slot.last = row;
            if replay {
                slot.replay = slot.replay.saturating_add(1);
            } else {
                slot.replay = slot.generated.len();
            }
        }
        let n = indices.len();
        if n >= 2 {
            self.note_gemm(n);
        } else {
            self.note_serial(n);
        }
        Ok(n)
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
    ///
    /// Returns the preempted slot index so a batched retry can drop that
    /// sequence. Prefill victims stay `is_prefill` (n_past is 0) and would
    /// otherwise re-enter the same batch forever.
    fn on_cap(&mut self, except: usize, err: LlamaError) -> Result<usize, LlamaError> {
        if !is_kv_cap(&err) {
            return Err(err);
        }
        self.preempt_except(except).ok_or(err)
    }

    /// Drop unique KV for the occupant with the most tokens, not `except`.
    ///
    /// Finished sequences still hold pages until `take`, so they are valid
    /// victims. A failed `ensure_write` can leave `n_past == 0` with a
    /// non-empty table; those rows count too. Interned prefixes stay in the
    /// pool. Returns the victim index when one was found.
    fn preempt_except(&mut self, except: usize) -> Option<usize> {
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
        let (idx, _, _) = best?;
        let Ok(cell) = self.slot_mut(idx) else {
            return None;
        };
        cell.cache.preempt();
        cell.last.clear();
        cell.replay = 0;
        self.preempts = self.preempts.saturating_add(1);
        Some(idx)
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

fn mean_itl(ends: &[u64]) -> Option<u64> {
    if ends.len() < 2 {
        return None;
    }
    let first = *ends.first()?;
    let last = *ends.last()?;
    let n = u64::try_from(ends.len().saturating_sub(1)).ok()?;
    last.saturating_sub(first).checked_div(n.max(1))
}

fn attach_token_latencies(mut score: Score, by_seq: &BTreeMap<SeqId, Vec<u64>>) -> Score {
    let mut ttft = None;
    let mut gap_sum = 0u64;
    let mut gap_n = 0u64;
    let mut ntok = 0u64;
    for ends in by_seq.values() {
        ntok = ntok.saturating_add(u64::try_from(ends.len()).unwrap_or(0));
        if let Some(&t) = ends.first() {
            ttft = Some(ttft.map_or(t, |u: u64| u.min(t)));
        }
        for pair in ends.windows(2) {
            let Some(&a) = pair.first() else {
                continue;
            };
            let Some(&b) = pair.get(1) else {
                continue;
            };
            gap_sum = gap_sum.saturating_add(b.saturating_sub(a));
            gap_n = gap_n.saturating_add(1);
        }
    }
    if ntok > 0 {
        score = score.with_tokens(ntok);
    }
    if let Some(t) = ttft {
        score = score.with_latencies(t, gap_sum.checked_div(gap_n));
    }
    score
}

#[cfg(test)]
mod tests {
    use super::{Engine, EngineCfg};
    use crate::decode::{
        greedy_generate_cache, tiny_llama4_gguf, tiny_llama_gguf, tiny_qwen3moe_2layer_gguf,
        tiny_qwen3moe_gguf, Llama, LlamaError,
    };
    use crate::gguf::load_gguf_owned;
    use crate::tok::Tokenizer;
    use expertvm::{
        CachedStore, DeviceId, ExpertKey, GpuFill, GpuStoreCfg, HardwareProfile, LiveStore, Score,
        SimulatedGpuStore, StoreMetrics, StreamId, TieredStore, Trace,
    };

    struct GpuEngineOut {
        ids_a: Vec<u32>,
        ids_b: Vec<u32>,
        exp_a: Vec<u32>,
        exp_b: Vec<u32>,
        peak: usize,
        metrics: StoreMetrics,
        score: Score,
        prefetches: u64,
    }

    fn run_two_seq_gpu(
        bytes: Vec<u8>,
        profile: HardwareProfile,
        expert_bytes: u64,
        min_slots: usize,
    ) -> GpuEngineOut {
        let tokens_a = [1u32, 2, 3, 4];
        let tokens_b = [5u32, 0, 5, 0];
        let g = load_gguf_owned(bytes).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("m");
        let exp_a = independent(&model, &tok, &tokens_a, 2);
        let exp_b = independent(&model, &tok, &tokens_b, 2);
        let n = model.expert_direct_store().expect("n").len().max(min_slots);
        let mut cfg = EngineCfg::tiny();
        cfg.eos = tok.eos;
        let mut eng = Engine::new(&model, cfg).expect("eng");
        let gpu = SimulatedGpuStore::new(
            model.expert_direct_store().expect("c"),
            n,
            profile,
            expert_bytes,
        )
        .expect("gpu");
        eng.attach_expert_store(LiveStore::simulated(gpu));
        let a = eng.add(&tokens_a, 2).expect("a");
        let b = eng.add(&tokens_b, 2).expect("b");
        eng.run().expect("run");
        let ids_a = eng.take(a).expect("ta").generated;
        let ids_b = eng.take(b).expect("tb").generated;
        let metrics = eng.expert_store_metrics().expect("metrics");
        let score = eng.expert_store_score().expect("sc").expect("sim");
        GpuEngineOut {
            ids_a,
            ids_b,
            exp_a,
            exp_b,
            peak: eng.stats().gemm_peak,
            prefetches: metrics.prefetches,
            metrics,
            score,
        }
    }

    fn independent(model: &Llama, tok: &Tokenizer, prompt: &[u32], n: usize) -> Vec<u32> {
        let mut cache = model.new_cache(16).expect("d");
        let mut ids = prompt.to_vec();
        let _s = greedy_generate_cache(model, tok, &mut cache, &mut ids, n).expect("g");
        ids.split_off(prompt.len())
    }

    fn traced_independent(
        model: &Llama,
        tok: &Tokenizer,
        prompt: &[u32],
        n: usize,
        sequence: u64,
    ) -> (Vec<u32>, Trace) {
        let mut cache = model.new_cache(16).expect("d");
        cache.enable_moe_trace(sequence);
        let mut ids = prompt.to_vec();
        let _s = greedy_generate_cache(model, tok, &mut cache, &mut ids, n).expect("g");
        (ids.split_off(prompt.len()), cache.take_moe_trace())
    }

    /// Engine does not forward the last sampled token; greedy `forward`s it.
    fn assert_engine_trace_prefix(
        engine: &Trace,
        greedy: &Trace,
        generated: usize,
        n_predict: usize,
    ) {
        let extra = usize::from(n_predict > 0 && generated == n_predict);
        assert!(
            !engine.events.is_empty(),
            "Engine MoE trace must record router events"
        );
        assert_eq!(
            engine.events.len().saturating_add(extra),
            greedy.events.len(),
            "Engine should omit only the unused last greedy forward"
        );
        assert!(
            greedy.events.starts_with(&engine.events),
            "Engine events must bit-match greedy for every forwarded token"
        );
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
    fn engine_two_added_together_match_independent() {
        let tokens_a = [1u32, 2, 3, 4];
        let tokens_b = [5u32, 0, 5, 0];
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
            let b = eng.add(&tokens_b, 2).expect("b");
            eng.run().expect("run");
            assert_eq!(eng.take(a).expect("ta").generated, exp_a);
            assert_eq!(eng.take(b).expect("tb").generated, exp_b);
            assert!(
                eng.stats().gemm_peak >= 8,
                "two 4-token prefills must GEMM together, peak={}",
                eng.stats().gemm_peak
            );
            assert!(eng.stats().steps > 0);
            assert!(eng.stats().gemm_tokens >= 8);
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
        assert_eq!(eng.n_past(b), None);
        eng.run().expect("run");
        assert_eq!(eng.take(a).expect("ta").generated, exp_a);
        assert_eq!(eng.take(b).expect("tb").generated, exp_b);
        assert_eq!(eng.waiting(), 0);
    }

    #[test]
    fn engine_decode_first_holds_leftover_prefill() {
        let prompt_a = [1u32, 2];
        let prompt_b = [1u32, 2, 3, 4, 5, 0, 1, 2];
        let model = Llama::from_gguf(load_gguf_owned(tiny_llama_gguf()).expect("o")).expect("m");
        let held = n_past_b_when_a_done(&model, &prompt_a, &prompt_b, true);
        let open = n_past_b_when_a_done(&model, &prompt_a, &prompt_b, false);
        assert_eq!(held, 2, "decode_first must hold B after A finishes prompt");
        assert!(
            open >= 4,
            "without decode_first B must keep prefilling, n_past={open}"
        );
        assert!(open > held, "open n_past={open} held={held}");
    }

    fn n_past_b_when_a_done(
        model: &Llama,
        prompt_a: &[u32],
        prompt_b: &[u32],
        decode_first: bool,
    ) -> usize {
        let mut cfg = EngineCfg::tiny();
        cfg.max_seqs = 2;
        cfg.prefill_chunk = 1;
        cfg.decode_first = decode_first;
        let mut eng = Engine::new(model, cfg).expect("eng");
        let a = eng.add(prompt_a, 4).expect("a");
        let b = eng.add(prompt_b, 1).expect("b");
        assert_eq!(eng.n_past(a), Some(0));
        assert_eq!(eng.n_past(b), Some(0));
        for _ in 0..128 {
            if eng.is_finished(a) {
                break;
            }
            let n = eng.step().expect("step");
            assert!(n > 0, "engine stall before A finished");
        }
        assert!(eng.is_finished(a), "A must finish generating");
        eng.n_past(b).expect("B still live")
    }

    #[test]
    fn engine_decode_first_ids_still_match_independent() {
        let tokens_a = [1u32, 2, 3, 4];
        let tokens_b = [5u32, 0, 5, 0];
        for bytes in [tiny_llama_gguf(), tiny_qwen3moe_gguf()] {
            let g = load_gguf_owned(bytes).expect("owned");
            let tok = Tokenizer::from_gguf(&g).expect("tok");
            let model = Llama::from_gguf(g).expect("m");
            let exp_a = independent(&model, &tok, &tokens_a, 2);
            let exp_b = independent(&model, &tok, &tokens_b, 2);
            let mut cfg = EngineCfg::tiny();
            cfg.decode_first = true;
            cfg.prefill_chunk = 1;
            cfg.eos = tok.eos;
            let mut eng = Engine::new(&model, cfg).expect("eng");
            let a = eng.add(&tokens_a, 2).expect("a");
            let b = eng.add(&tokens_b, 2).expect("b");
            eng.run().expect("run");
            assert_eq!(eng.take(a).expect("ta").generated, exp_a);
            assert_eq!(eng.take(b).expect("tb").generated, exp_b);
        }
    }

    #[test]
    fn engine_slo_reject_drops_waiting_sequence() {
        let tokens_a = [1u32, 2, 3, 4];
        let tokens_b = [5u32, 0, 5, 0];
        let g = load_gguf_owned(tiny_qwen3moe_gguf()).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("m");
        let exp_a = independent(&model, &tok, &tokens_a, 2);
        let exp_b = independent(&model, &tok, &tokens_b, 2);
        let n = model.expert_direct_store().expect("n").len().max(1);
        let mut cfg = EngineCfg::tiny();
        cfg.max_seqs = 1;
        cfg.eos = tok.eos;
        cfg.slo_reject = true;
        cfg.ttft_slo_ns = Some(1);
        let mut eng = Engine::new(&model, cfg).expect("eng");
        let gpu = SimulatedGpuStore::new(
            model.expert_direct_store().expect("c"),
            n,
            HardwareProfile::example_h100_sxm(),
            4096,
        )
        .expect("gpu");
        eng.attach_expert_store(LiveStore::simulated(gpu));
        let a = eng.add(&tokens_a, 2).expect("a");
        let b = eng.add(&tokens_b, 2).expect("queued");
        assert_eq!(eng.waiting(), 1);
        eng.run().expect("run");
        assert_eq!(eng.take(a).expect("ta").generated, exp_a);
        assert!(eng.is_finished(b));
        assert!(eng.take(b).expect("tb").generated.is_empty());
        assert_eq!(eng.rejected(), 1);

        let mut cfg = EngineCfg::tiny();
        cfg.max_seqs = 1;
        cfg.eos = tok.eos;
        let mut eng = Engine::new(&model, cfg).expect("open");
        let gpu = SimulatedGpuStore::new(
            model.expert_direct_store().expect("c2"),
            n,
            HardwareProfile::example_h100_sxm(),
            4096,
        )
        .expect("gpu2");
        eng.attach_expert_store(LiveStore::simulated(gpu));
        let a = eng.add(&tokens_a, 2).expect("a");
        let b = eng.add(&tokens_b, 2).expect("queued");
        eng.run().expect("run");
        assert_eq!(eng.take(a).expect("ta").generated, exp_a);
        assert_eq!(eng.take(b).expect("tb").generated, exp_b);
        assert_eq!(eng.rejected(), 0);
    }

    fn mixed_gpu_decode_itl(
        bytes: Vec<u8>,
        decode_first: bool,
        itl_slo_ns: Option<u64>,
    ) -> (u64, Score, usize, u64) {
        let prompt_a = [1u32, 2];
        let prompt_b = [1u32, 2, 3, 4, 5, 0, 1, 2];
        let g = load_gguf_owned(bytes).expect("owned");
        let model = Llama::from_gguf(g).expect("m");
        let n = model.expert_direct_store().expect("n").len().max(1);
        let mut cfg = EngineCfg::tiny();
        cfg.max_seqs = 2;
        cfg.prefill_chunk = 1;
        cfg.decode_first = decode_first;
        cfg.itl_slo_ns = itl_slo_ns;
        let mut eng = Engine::new(&model, cfg).expect("eng");
        let gpu = SimulatedGpuStore::new(
            model.expert_direct_store().expect("c"),
            n,
            HardwareProfile::example_h100_sxm(),
            4096,
        )
        .expect("gpu");
        eng.attach_expert_store(LiveStore::simulated(gpu));
        let a = eng.add(&prompt_a, 4).expect("a");
        let _b = eng.add(&prompt_b, 1).expect("b");
        eng.run().expect("run");
        let n_gen = eng.generated(a).expect("ga").len();
        let itl = eng.seq_itl_ns(a).expect("A ITL");
        let miss = eng.itl_slo_miss();
        let score = eng.expert_store_score().expect("sc").expect("sim");
        (itl, score, n_gen, miss)
    }

    #[test]
    fn engine_decode_first_shortens_mixed_gpu_itl() {
        let mixed = mixed_gpu_decode_itl(tiny_qwen3moe_2layer_gguf(), false, None);
        let prefer = mixed_gpu_decode_itl(tiny_qwen3moe_2layer_gguf(), true, None);
        assert_eq!(mixed.2, 4);
        assert_eq!(prefer.2, 4);
        assert!(prefer.1.ttft_ns.is_some(), "{}", prefer.1.line());
        assert!(prefer.1.itl_ns.is_some(), "{}", prefer.1.line());
        assert!(prefer.1.ns_per_token.is_some(), "{}", prefer.1.line());
        assert!(
            prefer.0 < mixed.0,
            "decode_first must not wait A's ITL on leftover prefill; prefer={} mixed={} prefer_line={} mixed_line={}",
            prefer.0,
            mixed.0,
            prefer.1.line(),
            mixed.1.line()
        );
    }

    #[test]
    fn engine_itl_slo_miss_counts_mixed_prefill() {
        let mixed = mixed_gpu_decode_itl(tiny_qwen3moe_2layer_gguf(), false, None);
        let prefer = mixed_gpu_decode_itl(tiny_qwen3moe_2layer_gguf(), true, None);
        let slo = prefer
            .0
            .saturating_add(mixed.0)
            .checked_div(2)
            .unwrap_or(1)
            .max(1);
        let mixed_miss = mixed_gpu_decode_itl(tiny_qwen3moe_2layer_gguf(), false, Some(slo)).3;
        let prefer_miss = mixed_gpu_decode_itl(tiny_qwen3moe_2layer_gguf(), true, Some(slo)).3;
        assert!(
            mixed_miss > prefer_miss,
            "leftover prefill must miss a mid ITL SLO more than decode_first; mixed={mixed_miss} prefer={prefer_miss} slo={slo} mixed_itl={} prefer_itl={}",
            mixed.0,
            prefer.0
        );
    }

    fn batch_prompt(i: u32) -> [u32; 2] {
        [i % 6, (i / 6) % 6]
    }

    #[derive(Clone, Copy)]
    enum Batch128Store {
        Blob,
        Direct,
        Cached,
        Tiered,
        Gpu,
    }

    fn attach_batch_128(eng: &mut Engine<'_>, model: &Llama, kind: Batch128Store) {
        match kind {
            Batch128Store::Blob => {}
            Batch128Store::Direct => {
                eng.attach_expert_store(LiveStore::Direct(
                    model.expert_direct_store().expect("catalog"),
                ));
            }
            Batch128Store::Cached => {
                let catalog = model.expert_direct_store().expect("catalog");
                let n = catalog.len().max(1);
                eng.attach_expert_store(LiveStore::Cached(
                    CachedStore::new(catalog, n).expect("cached"),
                ));
            }
            Batch128Store::Tiered => {
                let catalog = model.expert_direct_store().expect("catalog");
                let n = catalog.len().max(1);
                eng.attach_expert_store(LiveStore::tiered(
                    TieredStore::memory(catalog, n).expect("tiered"),
                ));
            }
            Batch128Store::Gpu => {
                let catalog = model.expert_direct_store().expect("catalog");
                let n = catalog.len().max(1);
                let gpu =
                    SimulatedGpuStore::new(catalog, n, HardwareProfile::example_h100_sxm(), 4096)
                        .expect("gpu");
                eng.attach_expert_store(LiveStore::simulated(gpu));
            }
        }
    }

    fn run_batch_128(bytes: Vec<u8>, kind: Batch128Store) {
        let g = load_gguf_owned(bytes).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("m");
        let n = 128u32;
        let n_predict = 1usize;
        let mut expected = Vec::new();
        for i in 0..n {
            expected.push(independent(&model, &tok, &batch_prompt(i), n_predict));
        }
        let mut cfg = EngineCfg::tiny();
        cfg.max_seqs = 8;
        cfg.pool_blocks = 64;
        cfg.eos = tok.eos;
        let mut eng = Engine::new(&model, cfg).expect("eng");
        attach_batch_128(&mut eng, &model, kind);
        let mut ids = Vec::new();
        for i in 0..n {
            ids.push(eng.add(&batch_prompt(i), n_predict).expect("add"));
        }
        assert_eq!(eng.active(), 8);
        assert_eq!(eng.waiting(), 120);
        eng.run().expect("run");
        for (i, id) in ids.iter().copied().enumerate() {
            let got = eng.take(id).expect("take").generated;
            let exp = expected.get(i).expect("exp");
            assert_eq!(&got, exp, "seq {i}");
        }
        assert_eq!(eng.waiting(), 0);
        assert!(
            eng.stats().gemm_peak >= 8,
            "8 in-flight sequences must GEMM together, peak={}",
            eng.stats().gemm_peak
        );
        match kind {
            Batch128Store::Blob => {}
            Batch128Store::Direct | Batch128Store::Cached | Batch128Store::Tiered => {
                let hits = eng.expert_store_metrics().expect("metrics").hits;
                assert!(hits > 0, "MoE batch-128 must acquire from the store");
            }
            Batch128Store::Gpu => {
                let hits = eng.expert_store_metrics().expect("metrics").hits;
                assert!(
                    hits > 0,
                    "MoE batch-128 must acquire from SimulatedGpuStore"
                );
                let score = eng.expert_store_score().expect("sc").expect("sim");
                assert!(score.wall_ns > 0, "gpu-sim must bill wall_ns");
            }
        }
    }

    #[test]
    fn engine_batch_128_waiting_queue_matches_independent() {
        run_batch_128(tiny_llama_gguf(), Batch128Store::Blob);
        run_batch_128(tiny_qwen3moe_gguf(), Batch128Store::Direct);
        run_batch_128(tiny_qwen3moe_2layer_gguf(), Batch128Store::Direct);
    }

    #[test]
    fn engine_batch_128_cached_and_gpu_match_independent() {
        run_batch_128(tiny_qwen3moe_gguf(), Batch128Store::Cached);
        run_batch_128(tiny_qwen3moe_2layer_gguf(), Batch128Store::Cached);
        run_batch_128(tiny_qwen3moe_gguf(), Batch128Store::Gpu);
        run_batch_128(tiny_qwen3moe_2layer_gguf(), Batch128Store::Gpu);
    }

    #[test]
    fn engine_batch_128_tiered_matches_independent() {
        run_batch_128(tiny_qwen3moe_gguf(), Batch128Store::Tiered);
        run_batch_128(tiny_qwen3moe_2layer_gguf(), Batch128Store::Tiered);
    }

    fn shared_prefix(i: u32) -> Vec<u32> {
        vec![1, 2, i % 6]
    }

    fn disjoint_prefix(i: u32) -> Vec<u32> {
        vec![i % 6, (i / 6) % 6, (i / 36) % 6]
    }

    fn intern_hits_128(prompt: fn(u32) -> Vec<u32>) -> u64 {
        let g = load_gguf_owned(tiny_llama_gguf()).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("m");
        let n = 128u32;
        let n_predict = 1usize;
        let mut expected = Vec::new();
        for i in 0..n {
            expected.push(independent(&model, &tok, &prompt(i), n_predict));
        }
        let mut cfg = EngineCfg::tiny();
        cfg.max_seqs = 8;
        cfg.pool_blocks = 256;
        cfg.eos = tok.eos;
        let mut eng = Engine::new(&model, cfg).expect("eng");
        let mut ids = Vec::new();
        for i in 0..n {
            ids.push(eng.add(&prompt(i), n_predict).expect("add"));
        }
        eng.run().expect("run");
        for (i, id) in ids.iter().copied().enumerate() {
            let got = eng.take(id).expect("take").generated;
            let exp = expected.get(i).expect("exp");
            assert_eq!(&got, exp, "seq {i}");
        }
        eng.pool().hits()
    }

    #[test]
    fn engine_batch_128_shared_prefix_interns_more_than_disjoint() {
        let shared = intern_hits_128(shared_prefix);
        let disjoint = intern_hits_128(disjoint_prefix);
        assert!(
            shared > disjoint,
            "shared [1,2] first page must intern more than disjoint 3-grams; shared={shared} disjoint={disjoint}"
        );
        assert!(
            shared > 8,
            "128 sequences sharing a completed page must intern-hit, hits={shared}"
        );
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

    #[test]
    fn engine_direct_store_two_sequences_match_blob_and_gemm() {
        let tokens_a = [1u32, 2, 3, 4];
        let tokens_b = [5u32, 0, 5, 0];
        let g = load_gguf_owned(tiny_qwen3moe_gguf()).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("m");
        let exp_a = independent(&model, &tok, &tokens_a, 2);
        let exp_b = independent(&model, &tok, &tokens_b, 2);
        let mut cfg = EngineCfg::tiny();
        cfg.eos = tok.eos;
        let mut blob = Engine::new(&model, cfg.clone()).expect("blob");
        let ba = blob.add(&tokens_a, 2).expect("ba");
        let bb = blob.add(&tokens_b, 2).expect("bb");
        blob.run().expect("blob run");
        assert_eq!(blob.take(ba).expect("tba").generated, exp_a);
        assert_eq!(blob.take(bb).expect("tbb").generated, exp_b);
        let mut eng = Engine::new(&model, cfg).expect("eng");
        eng.attach_expert_store(LiveStore::Direct(
            model.expert_direct_store().expect("direct"),
        ));
        let a = eng.add(&tokens_a, 2).expect("a");
        let b = eng.add(&tokens_b, 2).expect("b");
        eng.run().expect("run");
        assert_eq!(eng.take(a).expect("ta").generated, exp_a);
        assert_eq!(eng.take(b).expect("tb").generated, exp_b);
        assert!(
            eng.stats().gemm_peak >= 8,
            "Engine store must GEMM together, peak={}",
            eng.stats().gemm_peak
        );
        let hits = eng.expert_store_metrics().expect("metrics").hits;
        assert!(hits > 0, "batched MoE must acquire from the Engine store");
        assert!(
            hits < 16,
            "grouped expert GEMM must acquire each expert once per layer, hits={hits}"
        );
        assert!(eng.take_expert_store().is_some());
        assert!(eng.expert_store_metrics().is_none());
    }

    #[test]
    fn engine_simulated_gpu_store_matches_blob_and_scores() {
        let tokens_a = [1u32, 2, 3, 4];
        let tokens_b = [5u32, 0, 5, 0];
        let g = load_gguf_owned(tiny_qwen3moe_gguf()).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("m");
        let exp_a = independent(&model, &tok, &tokens_a, 2);
        let exp_b = independent(&model, &tok, &tokens_b, 2);
        let n = model.expert_direct_store().expect("n").len().max(1);
        let mut cfg = EngineCfg::tiny();
        cfg.eos = tok.eos;
        let mut eng = Engine::new(&model, cfg).expect("eng");
        let gpu = SimulatedGpuStore::new(
            model.expert_direct_store().expect("c"),
            n,
            HardwareProfile::example_h100_sxm(),
            4096,
        )
        .expect("gpu");
        eng.attach_expert_store(LiveStore::simulated(gpu));
        let a = eng.add(&tokens_a, 2).expect("a");
        let b = eng.add(&tokens_b, 2).expect("b");
        eng.run().expect("run");
        assert_eq!(eng.take(a).expect("ta").generated, exp_a);
        assert_eq!(eng.take(b).expect("tb").generated, exp_b);
        assert!(
            eng.stats().gemm_peak >= 8,
            "SimulatedGpuStore must GEMM together, peak={}",
            eng.stats().gemm_peak
        );
        let score = eng.expert_store_score().expect("sc").expect("sim");
        assert!(score.wall_ns > 0, "virtual clock must advance");
        assert!(
            score.bytes_moved > 0 || score.hbm_peak > 0,
            "H2D / HBM must be billed, {}",
            score.line()
        );
        assert!(
            eng.graph_launches() >= 2,
            "default SimulatedGpuStore captures GEMM graphs, launches={}",
            eng.graph_launches()
        );
    }

    #[test]
    fn engine_simulated_gpu_store_dispatches_large_experts() {
        let tokens_a = [1u32, 2, 3, 4];
        let tokens_b = [5u32, 0, 5, 0];
        let g = load_gguf_owned(tiny_qwen3moe_gguf()).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("m");
        let exp_a = independent(&model, &tok, &tokens_a, 2);
        let exp_b = independent(&model, &tok, &tokens_b, 2);
        let n = model.expert_direct_store().expect("n").len().max(2);
        let mut cfg = EngineCfg::tiny();
        cfg.eos = tok.eos;
        let mut eng = Engine::new(&model, cfg).expect("eng");
        let gpu = SimulatedGpuStore::new(
            model.expert_direct_store().expect("c"),
            n,
            HardwareProfile::example_8xh100_nvlink(),
            4096,
        )
        .expect("gpu");
        eng.attach_expert_store(LiveStore::simulated(gpu));
        let a = eng.add(&tokens_a, 2).expect("a");
        let b = eng.add(&tokens_b, 2).expect("b");
        eng.run().expect("run");
        assert_eq!(eng.take(a).expect("ta").generated, exp_a);
        assert_eq!(eng.take(b).expect("tb").generated, exp_b);
        assert!(
            eng.stats().gemm_peak >= 8,
            "dispatch must not force serial GEMM, peak={}",
            eng.stats().gemm_peak
        );
        let m = eng.expert_store_metrics().expect("metrics");
        assert!(
            m.dispatches > 0,
            "4096-byte experts beat a short reuse window, {m:?}"
        );
        assert_eq!(
            m.migrates, 0,
            "short Engine run must not D2D 4096-byte experts, {m:?}"
        );
        let score = eng.expert_store_score().expect("sc").expect("sim");
        assert!(score.wall_ns > 0, "virtual clock must advance");
    }

    #[test]
    fn engine_simulated_gpu_store_moves_tiny_experts_to_gpu0() {
        let tokens_a = [1u32, 2, 3, 4];
        let tokens_b = [5u32, 0, 5, 0];
        let g = load_gguf_owned(tiny_qwen3moe_gguf()).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("m");
        let exp_a = independent(&model, &tok, &tokens_a, 2);
        let exp_b = independent(&model, &tok, &tokens_b, 2);
        let n = model.expert_direct_store().expect("n").len().max(2);
        let mut cfg = EngineCfg::tiny();
        cfg.eos = tok.eos;
        let mut eng = Engine::new(&model, cfg).expect("eng");
        let gpu = SimulatedGpuStore::new(
            model.expert_direct_store().expect("c"),
            n,
            HardwareProfile::example_8xh100_nvlink(),
            64,
        )
        .expect("gpu");
        eng.attach_expert_store(LiveStore::simulated(gpu));
        let a = eng.add(&tokens_a, 2).expect("a");
        let b = eng.add(&tokens_b, 2).expect("b");
        eng.run().expect("run");
        assert_eq!(eng.take(a).expect("ta").generated, exp_a);
        assert_eq!(eng.take(b).expect("tb").generated, exp_b);
        assert!(
            eng.stats().gemm_peak >= 8,
            "migrate must not force serial GEMM, peak={}",
            eng.stats().gemm_peak
        );
        let m = eng.expert_store_metrics().expect("metrics");
        assert!(
            m.migrates > 0,
            "tiny experts must D2D striped-home pins onto GPU0, {m:?}"
        );
        let score = eng.expert_store_score().expect("sc").expect("sim");
        assert!(score.wall_ns > 0, "virtual clock must advance");
    }

    #[test]
    fn engine_simulated_gpu_store_two_layer_matches_blob_and_scores() {
        let one = run_two_seq_gpu(
            tiny_qwen3moe_gguf(),
            HardwareProfile::example_h100_sxm(),
            4096,
            1,
        );
        let two = run_two_seq_gpu(
            tiny_qwen3moe_2layer_gguf(),
            HardwareProfile::example_h100_sxm(),
            4096,
            1,
        );
        assert_eq!(two.ids_a, two.exp_a);
        assert_eq!(two.ids_b, two.exp_b);
        assert!(
            two.peak >= 8,
            "2-layer SimulatedGpuStore must GEMM together, peak={}",
            two.peak
        );
        assert!(two.score.wall_ns > 0, "virtual clock must advance");
        assert!(
            two.score.bytes_moved > 0 || two.score.hbm_peak > 0,
            "H2D / HBM must be billed, {}",
            two.score.line()
        );
        assert!(
            two.prefetches > one.prefetches,
            "copy-forward L+1 must prefetch on 2-layer, 1={} 2={}",
            one.prefetches,
            two.prefetches
        );
    }

    #[test]
    fn engine_simulated_gpu_store_two_layer_dispatches_large_experts() {
        let out = run_two_seq_gpu(
            tiny_qwen3moe_2layer_gguf(),
            HardwareProfile::example_8xh100_nvlink(),
            4096,
            2,
        );
        assert_eq!(out.ids_a, out.exp_a);
        assert_eq!(out.ids_b, out.exp_b);
        assert!(
            out.peak >= 8,
            "2-layer dispatch must not force serial GEMM, peak={}",
            out.peak
        );
        assert!(
            out.metrics.dispatches > 0,
            "4096-byte L0 and L+1 experts beat a short reuse window, {:?}",
            out.metrics
        );
        assert_eq!(
            out.metrics.migrates, 0,
            "short Engine run must not D2D 4096-byte experts, {:?}",
            out.metrics
        );
        assert!(out.score.wall_ns > 0, "virtual clock must advance");
    }

    #[test]
    fn engine_simulated_gpu_store_two_layer_moves_tiny_experts_to_gpu0() {
        let out = run_two_seq_gpu(
            tiny_qwen3moe_2layer_gguf(),
            HardwareProfile::example_8xh100_nvlink(),
            64,
            2,
        );
        assert_eq!(out.ids_a, out.exp_a);
        assert_eq!(out.ids_b, out.exp_b);
        assert!(
            out.peak >= 8,
            "2-layer migrate must not force serial GEMM, peak={}",
            out.peak
        );
        assert!(
            out.metrics.migrates > 0,
            "tiny L0 and L+1 experts must D2D striped-home pins onto GPU0, {:?}",
            out.metrics
        );
        assert!(out.score.wall_ns > 0, "virtual clock must advance");
    }

    #[test]
    fn engine_cached_store_two_sequences_match_blob_and_gemm() {
        let tokens_a = [1u32, 2, 3, 4];
        let tokens_b = [5u32, 0, 5, 0];
        let g = load_gguf_owned(tiny_qwen3moe_gguf()).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("m");
        let exp_a = independent(&model, &tok, &tokens_a, 2);
        let exp_b = independent(&model, &tok, &tokens_b, 2);
        let n = model.expert_direct_store().expect("n").len().max(1);
        let mut cfg = EngineCfg::tiny();
        cfg.eos = tok.eos;
        let mut eng = Engine::new(&model, cfg).expect("eng");
        eng.attach_expert_store(LiveStore::Cached(
            CachedStore::new(model.expert_direct_store().expect("c"), n).expect("cached"),
        ));
        let a = eng.add(&tokens_a, 2).expect("a");
        let b = eng.add(&tokens_b, 2).expect("b");
        eng.run().expect("run");
        assert_eq!(eng.take(a).expect("ta").generated, exp_a);
        assert_eq!(eng.take(b).expect("tb").generated, exp_b);
        assert!(
            eng.stats().gemm_peak >= 8,
            "CachedStore Engine must GEMM together, peak={}",
            eng.stats().gemm_peak
        );
        let m = eng.expert_store_metrics().expect("metrics");
        assert!(m.hits > 0 || m.misses > 0, "CachedStore must be used");
        assert!(
            m.prefetches > 0,
            "routed experts that fit in slots must prefetch before grouped GEMM, {m:?}"
        );
    }

    #[test]
    fn engine_tiered_store_two_sequences_match_blob_and_gemm() {
        let tokens_a = [1u32, 2, 3, 4];
        let tokens_b = [5u32, 0, 5, 0];
        let g = load_gguf_owned(tiny_qwen3moe_gguf()).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("m");
        let exp_a = independent(&model, &tok, &tokens_a, 2);
        let exp_b = independent(&model, &tok, &tokens_b, 2);
        let n = model.expert_direct_store().expect("n").len().max(1);
        let mut cfg = EngineCfg::tiny();
        cfg.eos = tok.eos;
        let mut eng = Engine::new(&model, cfg).expect("eng");
        eng.attach_expert_store(LiveStore::tiered(
            TieredStore::memory(model.expert_direct_store().expect("c"), n).expect("tier"),
        ));
        let a = eng.add(&tokens_a, 2).expect("a");
        let b = eng.add(&tokens_b, 2).expect("b");
        eng.run().expect("run");
        assert_eq!(eng.take(a).expect("ta").generated, exp_a);
        assert_eq!(eng.take(b).expect("tb").generated, exp_b);
        assert!(
            eng.stats().gemm_peak >= 8,
            "TieredStore Engine must GEMM together, peak={}",
            eng.stats().gemm_peak
        );
        let m = eng.expert_store_metrics().expect("metrics");
        assert!(m.hits > 0 || m.misses > 0, "TieredStore must be used");
    }

    #[test]
    fn engine_cached_store_prefetches_across_gemm_steps() {
        let tokens_a = [1u32, 2, 3, 4];
        let tokens_b = [5u32, 0, 5, 0];
        let g = load_gguf_owned(tiny_qwen3moe_gguf()).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("m");
        let exp_a = independent(&model, &tok, &tokens_a, 2);
        let exp_b = independent(&model, &tok, &tokens_b, 2);
        let mut cfg = EngineCfg::tiny();
        cfg.eos = tok.eos;
        let mut eng = Engine::new(&model, cfg).expect("eng");
        eng.attach_expert_store(LiveStore::Cached(
            CachedStore::new(model.expert_direct_store().expect("c"), 1).expect("cached"),
        ));
        let a = eng.add(&tokens_a, 2).expect("a");
        let b = eng.add(&tokens_b, 2).expect("b");
        eng.run().expect("run");
        assert_eq!(eng.take(a).expect("ta").generated, exp_a);
        assert_eq!(eng.take(b).expect("tb").generated, exp_b);
        let m = eng.expert_store_metrics().expect("metrics");
        assert!(
            m.prefetches > 0,
            "Engine Markov must prefetch across GEMMs, {m:?}"
        );
        assert_eq!(m.pins, 0, "slots=1 must not sticky-pin (demand paging)");
        assert!(
            eng.stats().gemm_peak >= 8,
            "prefetch must not force serial GEMM, peak={}",
            eng.stats().gemm_peak
        );
    }

    #[test]
    fn engine_cached_store_copy_forward_prefetches_layer1() {
        let tokens_a = [1u32, 2, 3, 4];
        let tokens_b = [5u32, 0, 5, 0];
        fn run(bytes: Vec<u8>, tokens_a: &[u32], tokens_b: &[u32]) -> u64 {
            let g = load_gguf_owned(bytes).expect("owned");
            let tok = Tokenizer::from_gguf(&g).expect("tok");
            let model = Llama::from_gguf(g).expect("m");
            let exp_a = independent(&model, &tok, tokens_a, 2);
            let exp_b = independent(&model, &tok, tokens_b, 2);
            let mut cfg = EngineCfg::tiny();
            cfg.eos = tok.eos;
            let mut eng = Engine::new(&model, cfg).expect("eng");
            let catalog = model.expert_direct_store().expect("c");
            let n = catalog.len().max(1);
            eng.attach_expert_store(LiveStore::Cached(
                CachedStore::new(catalog, n).expect("cached"),
            ));
            let a = eng.add(tokens_a, 2).expect("a");
            let b = eng.add(tokens_b, 2).expect("b");
            eng.run().expect("run");
            assert_eq!(eng.take(a).expect("ta").generated, exp_a);
            assert_eq!(eng.take(b).expect("tb").generated, exp_b);
            eng.expert_store_metrics().expect("metrics").prefetches
        }
        let p1 = run(tiny_qwen3moe_gguf(), &tokens_a, &tokens_b);
        let p2 = run(tiny_qwen3moe_2layer_gguf(), &tokens_a, &tokens_b);
        assert!(
            p2 > p1,
            "Engine copy-forward L+1 must prefetch on 2-layer, 1={p1} 2={p2}"
        );
    }

    #[test]
    fn engine_moe_trace_two_layer_includes_layer1() {
        let tokens = [1u32, 2, 3, 4];
        let g = load_gguf_owned(tiny_qwen3moe_2layer_gguf()).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("m");
        let mut cfg = EngineCfg::tiny();
        cfg.eos = tok.eos;
        let mut eng = Engine::new(&model, cfg).expect("eng");
        eng.enable_moe_trace();
        let a = eng.add(&tokens, 2).expect("a");
        eng.run().expect("run");
        let got = eng.take_moe_trace(a).expect("trace");
        assert!(
            got.events.iter().any(|e| e.layer == 0),
            "Engine 2-layer traces must include layer 0"
        );
        assert!(
            got.events.iter().any(|e| e.layer == 1),
            "Engine 2-layer traces must include layer 1"
        );
    }

    #[test]
    fn engine_cached_store_pins_hot_experts_with_demand_slot() {
        let tokens_a = [1u32, 2, 3, 4];
        let tokens_b = [5u32, 0, 5, 0];
        let g = load_gguf_owned(tiny_qwen3moe_gguf()).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("m");
        let exp_a = independent(&model, &tok, &tokens_a, 2);
        let exp_b = independent(&model, &tok, &tokens_b, 2);
        let mut cfg = EngineCfg::tiny();
        cfg.eos = tok.eos;
        let mut eng = Engine::new(&model, cfg).expect("eng");
        eng.attach_expert_store(LiveStore::Cached(
            CachedStore::new(model.expert_direct_store().expect("c"), 2).expect("cached"),
        ));
        let a = eng.add(&tokens_a, 2).expect("a");
        let b = eng.add(&tokens_b, 2).expect("b");
        eng.run().expect("run");
        assert_eq!(eng.take(a).expect("ta").generated, exp_a);
        assert_eq!(eng.take(b).expect("tb").generated, exp_b);
        let m = eng.expert_store_metrics().expect("metrics");
        assert!(m.pins > 0, "Engine must pin_hot last-used experts, {m:?}");
        assert!(
            eng.stats().gemm_peak >= 8,
            "pin_hot must not force serial GEMM, peak={}",
            eng.stats().gemm_peak
        );
    }

    #[test]
    fn engine_moe_trace_two_sequences_match_dense_and_gemm() {
        let tokens_a = [1u32, 2, 3, 4];
        let tokens_b = [5u32, 0, 5, 0];
        let g = load_gguf_owned(tiny_qwen3moe_gguf()).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("m");
        let (exp_a, tr_a) = traced_independent(&model, &tok, &tokens_a, 2, 0);
        let (exp_b, tr_b) = traced_independent(&model, &tok, &tokens_b, 2, 1);
        let mut cfg = EngineCfg::tiny();
        cfg.eos = tok.eos;
        let mut eng = Engine::new(&model, cfg).expect("eng");
        eng.enable_moe_trace();
        let a = eng.add(&tokens_a, 2).expect("a");
        let b = eng.add(&tokens_b, 2).expect("b");
        eng.run().expect("run");
        let got_a = eng.take_moe_trace(a).expect("live a");
        assert_engine_trace_prefix(&got_a, &tr_a, exp_a.len(), 2);
        assert_eq!(Trace::parse(&got_a.to_jsonl()).expect("jsonl"), got_a);
        let out_a = eng.take(a).expect("ta");
        let out_b = eng.take(b).expect("tb");
        assert_eq!(out_a.generated, exp_a);
        assert_eq!(out_b.generated, exp_b);
        let got_b = eng.take_moe_trace(b).expect("banked b");
        assert_engine_trace_prefix(&got_b, &tr_b, exp_b.len(), 2);
        assert!(eng.take_moe_trace(a).is_none());
        assert!(
            eng.stats().gemm_peak >= 8,
            "traced Engine must GEMM together, peak={}",
            eng.stats().gemm_peak
        );
    }

    struct GpuKnobOut {
        launches: u64,
        updates: u64,
        clones: u64,
        copy_elapsed_ns: u64,
        wall_ns: u64,
        copy_pri: i32,
        compute_pri: i32,
        accessed_peer: bool,
    }

    fn two_seq_gpu_on(
        slots: usize,
        gpu_cfg: GpuStoreCfg,
        fill: GpuFill,
        profile: HardwareProfile,
    ) -> GpuKnobOut {
        two_seq_gpu_bytes(slots, gpu_cfg, fill, profile, 4096)
    }

    fn two_seq_gpu_bytes(
        slots: usize,
        gpu_cfg: GpuStoreCfg,
        fill: GpuFill,
        profile: HardwareProfile,
        expert_bytes: u64,
    ) -> GpuKnobOut {
        let tokens_a = [1u32, 2, 3, 4];
        let tokens_b = [5u32, 0, 5, 0];
        let g = load_gguf_owned(tiny_qwen3moe_2layer_gguf()).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("m");
        let exp_a = independent(&model, &tok, &tokens_a, 2);
        let exp_b = independent(&model, &tok, &tokens_b, 2);
        let mut cfg = EngineCfg::tiny();
        cfg.eos = tok.eos;
        let mut eng = Engine::new(&model, cfg).expect("eng");
        let gpu = SimulatedGpuStore::with_cfg(
            model.expert_direct_store().expect("c"),
            slots,
            profile,
            expert_bytes,
            fill,
            gpu_cfg,
        )
        .expect("gpu");
        eng.attach_expert_store(LiveStore::simulated(gpu));
        let a = eng.add(&tokens_a, 2).expect("a");
        let b = eng.add(&tokens_b, 2).expect("b");
        eng.run().expect("run");
        assert_eq!(eng.take(a).expect("ta").generated, exp_a);
        assert_eq!(eng.take(b).expect("tb").generated, exp_b);
        let wall_ns = eng.expert_store_score().expect("sc").expect("sim").wall_ns;
        GpuKnobOut {
            launches: eng.graph_launches(),
            updates: eng.graph_updates(),
            clones: eng.graph_clones(),
            copy_elapsed_ns: eng.copy_elapsed_ns(),
            wall_ns,
            copy_pri: eng
                .gpu_stream_priority(DeviceId(0), StreamId(0))
                .unwrap_or(0),
            compute_pri: eng
                .gpu_stream_priority(DeviceId(0), StreamId(1))
                .unwrap_or(0),
            accessed_peer: eng.gpu_any_accessed_by(DeviceId(1)),
        }
    }

    fn two_seq_gpu_store(slots: usize, gpu_cfg: GpuStoreCfg, fill: GpuFill) -> GpuKnobOut {
        two_seq_gpu_on(slots, gpu_cfg, fill, HardwareProfile::example_h100_sxm())
    }

    fn two_seq_gpu_knobs(slots: usize, gpu_cfg: GpuStoreCfg) -> GpuKnobOut {
        two_seq_gpu_store(slots, gpu_cfg, GpuFill::Pinned)
    }

    #[test]
    fn engine_gpu_graph_update_after_tight_slots() {
        let out = two_seq_gpu_knobs(
            2,
            GpuStoreCfg {
                graph_update: true,
                ..GpuStoreCfg::default()
            },
        );
        assert!(
            out.updates > 0,
            "tight slots must park+update GEMM graphs, updates={} launches={}",
            out.updates,
            out.launches
        );
        assert!(out.launches >= 2, "launches={}", out.launches);
    }

    #[test]
    fn engine_gpu_graph_clone_and_timing_events() {
        let out = two_seq_gpu_knobs(
            8,
            GpuStoreCfg {
                graph_clone: true,
                timing_events: true,
                ..GpuStoreCfg::default()
            },
        );
        assert!(
            out.clones > 0,
            "graph_clone must copy captures, clones={}",
            out.clones
        );
        assert!(
            out.copy_elapsed_ns > 0,
            "timing_events must record copy elapsed, ns={}",
            out.copy_elapsed_ns
        );
        assert!(out.launches >= 2, "launches={}", out.launches);
    }

    #[test]
    fn engine_gpu_fill_modes_match_independent() {
        for fill in [GpuFill::Managed, GpuFill::Mapped, GpuFill::Vmm] {
            let out = two_seq_gpu_store(8, GpuStoreCfg::default(), fill);
            assert!(out.launches >= 2, "fill={fill:?} launches={}", out.launches);
        }
    }

    #[test]
    fn engine_gpu_cfg_knobs_lengthen_wall() {
        let base = two_seq_gpu_knobs(8, GpuStoreCfg::default()).wall_ns;
        let knobs = [
            (
                "host_func",
                GpuStoreCfg {
                    host_func: true,
                    ..GpuStoreCfg::default()
                },
            ),
            (
                "pageable",
                GpuStoreCfg {
                    pageable: true,
                    ..GpuStoreCfg::default()
                },
            ),
            (
                "legacy_null",
                GpuStoreCfg {
                    legacy_null: true,
                    ..GpuStoreCfg::default()
                },
            ),
            (
                "blocking_streams",
                GpuStoreCfg {
                    blocking_streams: true,
                    ..GpuStoreCfg::default()
                },
            ),
            (
                "sync_alloc",
                GpuStoreCfg {
                    sync_alloc: true,
                    ..GpuStoreCfg::default()
                },
            ),
        ];
        for (name, cfg) in knobs {
            let out = two_seq_gpu_knobs(8, cfg);
            assert!(
                out.wall_ns > base,
                "{name} must lengthen wall; knob={} base={base}",
                out.wall_ns
            );
            assert!(out.launches >= 2, "{name} launches={}", out.launches);
        }
    }

    #[test]
    fn engine_gpu_stream_priority_marks_compute() {
        let off = two_seq_gpu_knobs(8, GpuStoreCfg::default());
        assert_eq!(off.copy_pri, 0);
        assert_eq!(off.compute_pri, 0);
        let on = two_seq_gpu_knobs(
            8,
            GpuStoreCfg {
                stream_priority: true,
                ..GpuStoreCfg::default()
            },
        );
        assert_eq!(on.copy_pri, 0, "copy stays NULL priority 0");
        assert_eq!(on.compute_pri, 1, "compute is stream 1 at priority 1");
        assert!(on.launches >= 2, "launches={}", on.launches);
    }

    #[test]
    fn engine_gpu_accessed_by_maps_peer_on_8gpu() {
        let off = two_seq_gpu_store(8, GpuStoreCfg::default(), GpuFill::Managed);
        assert!(!off.accessed_peer, "1-GPU managed must not map a peer");
        let on = two_seq_gpu_on(
            8,
            GpuStoreCfg {
                accessed_by: true,
                ..GpuStoreCfg::default()
            },
            GpuFill::Managed,
            HardwareProfile::example_8xh100_nvlink(),
        );
        assert!(on.accessed_peer, "accessed_by must map a peer GPU");
        assert!(on.launches >= 2, "launches={}", on.launches);
    }

    #[test]
    fn engine_gpu_mempool_holds_after_evict() {
        let cached = |mempool: bool| {
            let tokens_a = [1u32, 2, 3, 4];
            let tokens_b = [5u32, 0, 5, 0];
            let g = load_gguf_owned(tiny_qwen3moe_2layer_gguf()).expect("owned");
            let tok = Tokenizer::from_gguf(&g).expect("tok");
            let model = Llama::from_gguf(g).expect("m");
            let exp_a = independent(&model, &tok, &tokens_a, 2);
            let exp_b = independent(&model, &tok, &tokens_b, 2);
            let mut cfg = EngineCfg::tiny();
            cfg.eos = tok.eos;
            let mut eng = Engine::new(&model, cfg).expect("eng");
            let gpu = SimulatedGpuStore::with_cfg(
                model.expert_direct_store().expect("c"),
                2,
                HardwareProfile::example_h100_sxm(),
                4096,
                GpuFill::Pinned,
                GpuStoreCfg {
                    mempool,
                    ..GpuStoreCfg::default()
                },
            )
            .expect("gpu");
            eng.attach_expert_store(LiveStore::simulated(gpu));
            let a = eng.add(&tokens_a, 2).expect("a");
            let b = eng.add(&tokens_b, 2).expect("b");
            eng.run().expect("run");
            assert_eq!(eng.take(a).expect("ta").generated, exp_a);
            assert_eq!(eng.take(b).expect("tb").generated, exp_b);
            let mut store = eng.take_expert_store().expect("store");
            store.unpin_all();
            for layer in 0..2u32 {
                for expert in 0..4u32 {
                    let k = ExpertKey::new(layer, expert);
                    if store.is_resident(k) {
                        store.evict(k).expect("evict");
                    }
                }
            }
            store.default_pool_cached().expect("pool").unwrap_or(0)
        };
        assert_eq!(cached(false), 0, "default pool must release");
        let held = cached(true);
        assert!(
            held >= 4096,
            "mempool must hold an evicted expert page, cached={held}"
        );
    }

    #[test]
    fn engine_gpu_vmm_page_pays_map_overhead() {
        let bytes = 1u64 << 16;
        let page = bytes / 4;
        let profile = HardwareProfile::example_h100_sxm();
        let full = two_seq_gpu_bytes(
            8,
            GpuStoreCfg::default(),
            GpuFill::Vmm,
            profile.clone(),
            bytes,
        );
        let paged = two_seq_gpu_bytes(
            8,
            GpuStoreCfg {
                vmm_page: page,
                ..GpuStoreCfg::default()
            },
            GpuFill::Vmm,
            profile,
            bytes,
        );
        assert_eq!(full.launches, paged.launches);
        assert!(
            paged.wall_ns > full.wall_ns,
            "paged VMM must pay per-block map overhead; paged={} full={}",
            paged.wall_ns,
            full.wall_ns
        );
    }
}
