//! Layered `Model` / `Session` API over the reference engine.

use crate::decode::{KvCache, Llama, LlamaError, PagedKvPool};
use crate::gguf::{load_gguf_owned, Gguf};
use crate::tok::Tokenizer;
use expertvm::{LiveStore, StoreMetrics};

/// Loaded GGUF weights and tokenizer.
pub struct Model {
    llama: Llama,
    tok: Tokenizer,
}

impl Model {
    /// Parse an owned GGUF file (one blob; tensors stay ranges).
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, LlamaError> {
        let g = load_gguf_owned(bytes)?;
        Self::from_gguf(g)
    }

    /// Load from an already-parsed GGUF. Takes the file blob.
    pub fn from_gguf(g: Gguf) -> Result<Self, LlamaError> {
        let tok = Tokenizer::from_gguf(&g)?;
        let llama = Llama::from_gguf(g)?;
        Ok(Self { llama, tok })
    }

    /// Reference weights.
    #[must_use]
    pub fn llama(&self) -> &Llama {
        &self.llama
    }

    /// Tokenizer bound to this checkpoint.
    #[must_use]
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    /// Encode `text`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, LlamaError> {
        self.tok.encode(text).map_err(Into::into)
    }

    /// New KV session with capacity `n_ctx`.
    pub fn session(&self, n_ctx: usize) -> Result<Session<'_>, LlamaError> {
        Ok(Session {
            llama: &self.llama,
            cache: self.llama.new_cache(n_ctx)?,
        })
    }

    /// Session whose KV is paged (`block_size` tokens per block).
    pub fn session_paged(
        &self,
        n_ctx: usize,
        block_size: usize,
    ) -> Result<Session<'_>, LlamaError> {
        Ok(Session {
            llama: &self.llama,
            cache: self.llama.new_paged_cache(n_ctx, block_size)?,
        })
    }

    /// Shared interned-block arena for [`Model::session_on_pool`].
    pub fn paged_pool(&self, block_size: usize, cap: usize) -> Result<PagedKvPool, LlamaError> {
        self.llama.new_paged_pool(block_size, cap)
    }

    /// Session on a shared [`PagedKvPool`] so interned prefixes are visible
    /// to other sessions on the same pool.
    pub fn session_on_pool(
        &self,
        n_ctx: usize,
        pool: &PagedKvPool,
    ) -> Result<Session<'_>, LlamaError> {
        Ok(Session {
            llama: &self.llama,
            cache: self.llama.new_paged_cache_on(pool, n_ctx)?,
        })
    }
}

/// One sequence: KV cache plus optional expert store.
pub struct Session<'a> {
    llama: &'a Llama,
    cache: KvCache,
}

impl Session<'_> {
    /// Opt-in ExpertStore seam. Default decode stays on the GGUF blob.
    pub fn attach_expert_store(&mut self, store: LiveStore) {
        self.cache.attach_expert_store(store);
    }

    /// Prompt tokens in one causal pass. Logits of the last token.
    ///
    /// Appends to the cache. Use [`Session::prompt`] to treat `tokens` as a
    /// full prompt and reuse a matching KV prefix.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<&[f32], LlamaError> {
        self.llama.prefill_logits(&mut self.cache, tokens)
    }

    /// Full-prompt prefill with Automatic Prefix Caching.
    ///
    /// Longest common prefix of `tokens` and [`Session::cached_ids`] keeps its
    /// KV; only the suffix is forwarded. [`Session::prefill`] still appends.
    pub fn prompt(&mut self, tokens: &[u32]) -> Result<&[f32], LlamaError> {
        self.llama.prompt_logits(&mut self.cache, tokens)
    }

    /// One generated token.
    pub fn decode(&mut self, token: u32) -> Result<&[f32], LlamaError> {
        self.llama.forward_logits(&mut self.cache, token)
    }

    /// Tokens already in the KV cache.
    #[must_use]
    pub fn n_past(&self) -> usize {
        self.cache.n_past
    }

    /// Token ids occupying KV slots `0 .. n_past`.
    #[must_use]
    pub fn cached_ids(&self) -> &[u32] {
        self.cache.cached_ids()
    }

    /// Prefix length reused by the last [`Session::prompt`].
    #[must_use]
    pub fn last_prefix_hit(&self) -> usize {
        self.cache.last_prefix_hit()
    }

    /// Interned paged-KV block hits (`0` on a dense session).
    #[must_use]
    pub fn page_hits(&self) -> u64 {
        self.cache.page_hits()
    }

    /// Paged block size when this session was built with [`Model::session_paged`].
    #[must_use]
    pub fn page_size(&self) -> Option<usize> {
        self.cache.page_size()
    }

    /// Store counters, if a store is attached.
    #[must_use]
    pub fn expert_metrics(&self) -> Option<StoreMetrics> {
        self.cache.expert_store_metrics()
    }
}

#[cfg(test)]
mod tests {
    use super::Model;
    use crate::decode::tiny_llama_gguf;
    use expertvm::LiveStore;

    #[test]
    fn session_prefill_matches_llama() {
        let model = Model::from_bytes(tiny_llama_gguf()).expect("model");
        let ids = model.encode("ab").expect("enc");
        assert_eq!(ids, vec![3]);
        let mut sess = model.session(8).expect("sess");
        sess.attach_expert_store(LiveStore::Direct(
            model.llama().expert_direct_store().expect("empty moe"),
        ));
        let logits = sess.prefill(&ids).expect("pref").to_vec();
        assert_eq!(sess.n_past(), 1);
        assert_eq!(logits.len(), model.llama().n_vocab);
        assert!(sess.expert_metrics().is_some());
    }

    #[test]
    fn session_qwen3moe_direct_store_matches_blob() {
        let model = Model::from_bytes(crate::tiny_qwen3moe_gguf()).expect("model");
        let ids = model.encode("ab").expect("enc");
        let blob = {
            let mut s = model.session(8).expect("blob sess");
            s.prefill(&ids).expect("blob").to_vec()
        };
        let mut sess = model.session(8).expect("store sess");
        sess.attach_expert_store(LiveStore::Direct(
            model.llama().expert_direct_store().expect("catalog"),
        ));
        let via = sess.prefill(&ids).expect("store").to_vec();
        assert_eq!(blob, via, "Session DirectStore must match the blob path");
        assert!(sess.expert_metrics().is_some());
    }

    #[test]
    fn session_prompt_reuses_prefix_and_prefill_still_appends() {
        let model = Model::from_bytes(tiny_llama_gguf()).expect("model");
        let mut sess = model.session(16).expect("sess");
        let first = sess.prefill(&[1, 2, 3]).expect("append").to_vec();
        assert_eq!(sess.n_past(), 3);
        let _more = sess.prefill(&[4]).expect("still append");
        assert_eq!(sess.n_past(), 4);
        assert_eq!(sess.cached_ids(), &[1, 2, 3, 4]);
        let reused = sess.prompt(&[1, 2, 3]).expect("prompt").to_vec();
        assert_eq!(sess.n_past(), 3);
        assert_eq!(sess.cached_ids(), &[1, 2, 3]);
        assert_eq!(sess.last_prefix_hit(), 3);
        assert_eq!(first, reused);
        let mut cold = model.session(16).expect("cold");
        let exp = cold.prefill(&[1, 2, 0]).expect("cold").to_vec();
        let got = sess.prompt(&[1, 2, 0]).expect("partial").to_vec();
        assert_eq!(got, exp);
        assert_eq!(sess.last_prefix_hit(), 2);
    }

    #[test]
    fn session_paged_logits_match_dense_and_intern_hits() {
        let model = Model::from_bytes(tiny_llama_gguf()).expect("model");
        let tokens = [1u32, 2, 3, 4];
        let dense = {
            let mut s = model.session(16).expect("dense");
            s.prefill(&tokens).expect("d").to_vec()
        };
        let mut paged = model.session_paged(16, 2).expect("paged");
        let got = paged.prefill(&tokens).expect("p").to_vec();
        assert_eq!(got, dense);
        assert_eq!(paged.n_past(), 4);
        assert_eq!(paged.page_size(), Some(2));
        let _other = paged.prompt(&[5, 0, 5, 0]).expect("divergent");
        let again = paged.prompt(&tokens).expect("intern").to_vec();
        assert_eq!(again, dense);
        assert!(
            paged.page_hits() > 0,
            "prompt after a rewind must hit interned prefix blocks"
        );
    }

    #[test]
    fn session_on_pool_interns_across_sessions() {
        let model = Model::from_bytes(tiny_llama_gguf()).expect("model");
        let tokens = [1u32, 2, 3, 4];
        let dense = {
            let mut s = model.session(16).expect("dense");
            s.prefill(&tokens).expect("d").to_vec()
        };
        let pool = model.paged_pool(2, 16).expect("pool");
        let mut a = model.session_on_pool(16, &pool).expect("a");
        let mut b = model.session_on_pool(16, &pool).expect("b");
        let got = a.prefill(&tokens).expect("a").to_vec();
        assert_eq!(got, dense);
        let hit = b.prompt(&tokens).expect("b").to_vec();
        assert_eq!(hit, dense);
        assert!(b.page_hits() > 0);
        assert!(pool.hits() > 0);
    }
}
