//! GGUF-embedded tokenizer: vocab tokens plus optional BPE merges.

use crate::gguf::{Gguf, GgufError, Kv};

/// Failure while encoding, decoding, or reading tokenizer KV.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokError {
    /// `tokenizer.ggml.tokens` missing or not a string array.
    Vocab,
    /// A piece of the prompt is not in the vocab.
    Unknown(String),
    /// GGUF parse failure used as tokenizer source.
    Gguf(GgufError),
}

impl std::fmt::Display for TokError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vocab => write!(f, "missing tokenizer.ggml.tokens"),
            Self::Unknown(s) => write!(f, "unknown token piece {s:?}"),
            Self::Gguf(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for TokError {}

/// Vocab + optional BPE merges loaded from GGUF KV.
#[derive(Clone, Debug)]
pub struct Tokenizer {
    /// Token strings indexed by id.
    pub tokens: Vec<String>,
    /// Merge pairs in rank order (`"a b"`).
    pub merges: Vec<(String, String)>,
    /// BOS id, if present.
    pub bos: Option<u32>,
    /// EOS id, if present.
    pub eos: Option<u32>,
    /// Whether encode/generate should prepend `bos`. Default true when the KV is absent.
    pub add_bos: bool,
}

impl Tokenizer {
    /// Load tokenizer metadata from a parsed GGUF.
    pub fn from_gguf(g: &Gguf) -> Result<Self, TokError> {
        let Some(Kv::Array { items, .. }) = g.kv("tokenizer.ggml.tokens") else {
            return Err(TokError::Vocab);
        };
        let mut tokens = Vec::new();
        for it in items {
            let Kv::String(s) = it else {
                return Err(TokError::Vocab);
            };
            tokens.push(s.clone());
        }
        if tokens.is_empty() {
            return Err(TokError::Vocab);
        }
        let mut merges = Vec::new();
        if let Some(Kv::Array { items, .. }) = g.kv("tokenizer.ggml.merges") {
            for it in items {
                let Kv::String(s) = it else {
                    continue;
                };
                let mut sp = s.split(' ');
                if let (Some(a), Some(b)) = (sp.next(), sp.next()) {
                    merges.push((a.to_string(), b.to_string()));
                }
            }
        }
        Ok(Self {
            tokens,
            merges,
            bos: g.kv_u32("tokenizer.ggml.bos_token_id"),
            eos: g.kv_u32("tokenizer.ggml.eos_token_id"),
            add_bos: g.kv_bool("tokenizer.ggml.add_bos_token").unwrap_or(true),
        })
    }

    /// Encode `text` with longest-known characters then BPE merges.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokError> {
        let mut parts: Vec<String> = Vec::new();
        for ch in text.chars() {
            let s = ch.to_string();
            if self.id_of(&s).is_none() {
                return Err(TokError::Unknown(s));
            }
            parts.push(s);
        }
        loop {
            let mut best_rank = self.merges.len();
            let mut best_i = None;
            for i in 0..parts.len().saturating_sub(1) {
                let Some(a) = parts.get(i) else { continue };
                let Some(b) = parts.get(i + 1) else { continue };
                if let Some(rank) = self.merge_rank(a, b) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_i = Some(i);
                    }
                }
            }
            let Some(i) = best_i else { break };
            let a = parts.get(i).cloned().unwrap_or_default();
            let b = parts.get(i + 1).cloned().unwrap_or_default();
            if let Some(slot) = parts.get_mut(i) {
                slot.clear();
                slot.push_str(&a);
                slot.push_str(&b);
            }
            if i + 1 < parts.len() {
                let _removed = parts.remove(i + 1);
            }
        }
        let mut ids = Vec::new();
        for p in &parts {
            let id = self.id_of(p).ok_or_else(|| TokError::Unknown(p.clone()))?;
            ids.push(id);
        }
        Ok(ids)
    }

    /// Concatenate token strings for `ids`.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for id in ids {
            if Some(*id) == self.bos || Some(*id) == self.eos {
                continue;
            }
            let i = usize::try_from(*id).unwrap_or(0);
            if let Some(t) = self.tokens.get(i) {
                out.push_str(t);
            }
        }
        out
    }

    fn id_of(&self, s: &str) -> Option<u32> {
        self.tokens
            .iter()
            .position(|t| t == s)
            .and_then(|i| u32::try_from(i).ok())
    }

    fn merge_rank(&self, a: &str, b: &str) -> Option<usize> {
        self.merges.iter().position(|(x, y)| x == a && y == b)
    }
}
