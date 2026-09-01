//! GGUF-embedded tokenizer: vocab tokens plus optional BPE merges.
//!
//! GPT-2 / Qwen pieces use the bytes-to-unicode map (`Ċ` is newline, `Ġ` is space).
//! SentencePiece pieces use `▁` for space and `<0xHH>` byte-fallback tokens.
//! Token and merge lookup is by `HashMap`, not a linear scan of the vocab.
//!
//! Encoding follows llama.cpp's three stages:
//!
//! 1. Split the raw text on the vocabulary's special/added tokens and emit
//!    their ids directly, so `<|im_start|>` is one id and never byte-merged.
//! 2. For byte-level BPE, split each remaining span with the pre-tokenizer
//!    regex ([`crate::pretok`]) so merges cannot cross a pre-token boundary.
//! 3. Merge inside each piece by merge rank, then map to ids with `<0xHH>`
//!    byte fallback.

use std::collections::HashMap;

use crate::gguf::{Gguf, GgufError, Kv};
use crate::pretok::PreTokenizer;
use crate::template::{render_chat_template, ChatMessage, ChatOptions, TemplateError};

/// Failure while encoding, decoding, or reading tokenizer KV.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokError {
    /// `tokenizer.ggml.tokens` missing or not a string array.
    Vocab,
    /// A piece of the prompt is not in the vocab and has no byte-fallback token.
    Unknown(String),
    /// GGUF parse failure used as tokenizer source.
    Gguf(GgufError),
    /// The GGUF has no `tokenizer.chat_template`.
    NoChatTemplate,
    /// The chat template could not be parsed or rendered.
    Template(TemplateError),
}

impl std::fmt::Display for TokError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vocab => write!(f, "missing tokenizer.ggml.tokens"),
            Self::Unknown(s) => write!(f, "unknown token piece {s:?}"),
            Self::Gguf(e) => write!(f, "{e}"),
            Self::NoChatTemplate => write!(f, "model has no tokenizer.chat_template"),
            Self::Template(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for TokError {}

impl From<TemplateError> for TokError {
    fn from(e: TemplateError) -> Self {
        Self::Template(e)
    }
}

/// `tokenizer.ggml.token_type` values, from llama.cpp's `llama_token_type`.
mod token_type {
    /// Ordinary vocabulary entry.
    pub(super) const NORMAL: i32 = 1;
    /// The unknown token.
    pub(super) const UNKNOWN: i32 = 2;
    /// A control token such as `<|im_start|>` or `<s>`.
    pub(super) const CONTROL: i32 = 3;
    /// An added token from `added_tokens` / `tokenizer.json`.
    pub(super) const USER_DEFINED: i32 = 4;
    /// A `<0xHH>` byte-fallback entry. Never a special token.
    pub(super) const BYTE: i32 = 6;
}

/// GGUF keys naming a single distinguished token id.
const NAMED_TOKEN_ID_KEYS: [&str; 10] = [
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.eot_token_id",
    "tokenizer.ggml.eom_token_id",
    "tokenizer.ggml.eog_token_id",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.seperator_token_id",
    "tokenizer.ggml.cls_token_id",
    "tokenizer.ggml.mask_token_id",
];

/// One special token that pre-tokenization splits the raw text on.
#[derive(Clone, Debug)]
struct Special {
    text: String,
    id: u32,
    /// `USER_DEFINED` tokens split even when `parse_special` is off, matching
    /// llama.cpp's `tokenizer_st_partition`.
    always: bool,
}

/// A span of input: either raw text to BPE, or a special token id verbatim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Frag<'a> {
    Text(&'a str),
    Special(u32),
}

/// How prompt bytes are turned into BPE symbols.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Characters as stored; merge only. Writer-built tiny vocabs.
    Raw,
    /// GPT-2 bytes-to-unicode, then BPE. Qwen2 and `tokenizer.ggml.model=gpt2`.
    Gpt2,
    /// SentencePiece: space → `▁`, unknown UTF-8 → `<0xHH>`.
    Spiece,
}

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
    /// `tokenizer.ggml.model` (`gpt2`, `llama`, …), if present.
    pub model: Option<String>,
    /// `tokenizer.ggml.pre`, the pre-tokenizer regex family, if present.
    pub pre: Option<String>,
    /// `tokenizer.chat_template` Jinja, if present. Render it with
    /// [`Tokenizer::apply_chat_template`].
    pub chat_template: Option<String>,
    /// SentencePiece dummy-prefix (`▁` before the first symbol).
    pub add_space_prefix: bool,
    token_to_id: HashMap<String, u32>,
    merge_rank: HashMap<(String, String), usize>,
    /// Special tokens bucketed by first character, each bucket sorted longest
    /// first so a scan takes the leftmost-longest match.
    specials: HashMap<char, Vec<Special>>,
    pre_tokenizer: PreTokenizer,
    kind: Kind,
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
                if let Some((a, b)) = s.split_once(' ') {
                    if !a.is_empty() && !b.is_empty() {
                        merges.push((a.to_string(), b.to_string()));
                    }
                }
            }
        }
        let model = g.kv_string("tokenizer.ggml.model").map(str::to_string);
        let kind = detect_kind(model.as_deref(), &tokens);
        let add_space_prefix = g
            .kv_bool("tokenizer.ggml.add_space_prefix")
            .unwrap_or(kind == Kind::Spiece);
        let pre = g.kv_string("tokenizer.ggml.pre").map(str::to_string);
        let mut named = Vec::new();
        for key in NAMED_TOKEN_ID_KEYS {
            if let Some(id) = g.kv_u32(key) {
                named.push(id);
            }
        }
        let specials = collect_specials(&tokens, &token_types(g, tokens.len()), &named);
        Ok(Self::with_indexes(Self {
            token_to_id: HashMap::new(),
            merge_rank: HashMap::new(),
            specials,
            pre_tokenizer: PreTokenizer::from_pre_key(pre.as_deref()),
            kind,
            tokens,
            merges,
            bos: g.kv_u32("tokenizer.ggml.bos_token_id"),
            eos: g.kv_u32("tokenizer.ggml.eos_token_id"),
            add_bos: g.kv_bool("tokenizer.ggml.add_bos_token").unwrap_or(true),
            model,
            pre,
            chat_template: g.kv_string("tokenizer.chat_template").map(str::to_string),
            add_space_prefix,
        }))
    }

    fn with_indexes(mut self) -> Self {
        let mut token_to_id = HashMap::with_capacity(self.tokens.len());
        for (i, t) in self.tokens.iter().enumerate() {
            if let Ok(id) = u32::try_from(i) {
                if !token_to_id.contains_key(t) {
                    let _prev = token_to_id.insert(t.clone(), id);
                }
            }
        }
        let mut merge_rank = HashMap::with_capacity(self.merges.len());
        for (rank, pair) in self.merges.iter().enumerate() {
            if !merge_rank.contains_key(pair) {
                let _prev = merge_rank.insert(pair.clone(), rank);
            }
        }
        self.token_to_id = token_to_id;
        self.merge_rank = merge_rank;
        self
    }

    /// Special / added token pieces and their ids, longest first.
    ///
    /// These are the pieces [`Tokenizer::encode`] emits verbatim instead of
    /// running through BPE.
    pub fn special_tokens(&self) -> Vec<(&str, u32)> {
        let mut out: Vec<(&str, u32)> = self
            .specials
            .values()
            .flatten()
            .map(|s| (s.text.as_str(), s.id))
            .collect();
        out.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
        out
    }

    /// Whether `id` is a special / added token rather than ordinary text.
    pub fn is_special(&self, id: u32) -> bool {
        self.specials.values().flatten().any(|s| s.id == id)
    }

    /// Piece text for `id`.
    pub fn token_text(&self, id: u32) -> Option<&str> {
        self.tokens
            .get(usize::try_from(id).ok()?)
            .map(String::as_str)
    }

    /// Render this model's own `tokenizer.chat_template` for `messages`.
    ///
    /// `bos_token` and `eos_token` are bound from the vocabulary, so templates
    /// that emit them (Llama-3, Mistral, Gemma) produce the model's real
    /// markers. Feed the result to [`Tokenizer::encode`], which maps them back
    /// to single ids.
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String, TokError> {
        let Some(template) = self.chat_template.as_deref() else {
            return Err(TokError::NoChatTemplate);
        };
        let opts = ChatOptions {
            add_generation_prompt,
            bos_token: self
                .bos
                .and_then(|id| self.token_text(id))
                .map(str::to_string),
            eos_token: self
                .eos
                .and_then(|id| self.token_text(id))
                .map(str::to_string),
            ..ChatOptions::default()
        };
        Ok(render_chat_template(template, messages, &opts)?)
    }

    /// Id of an exact vocab piece. `O(1)` map lookup, not a scan of `tokens`.
    pub fn token_id(&self, piece: &str) -> Option<u32> {
        self.token_to_id.get(piece).copied()
    }

    /// Encode `text` with byte-level GPT-2 BPE, SentencePiece + `<0xHH>`, or raw chars.
    ///
    /// Special tokens written in `text` (`<|im_start|>`, `<|eot_id|>`,
    /// `<start_of_turn>`, …) become their own id and are not byte-merged, which
    /// is llama.cpp's `parse_special = true`. Use [`Tokenizer::encode_ordinary`]
    /// for untrusted text that should be tokenized literally.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokError> {
        self.encode_with(text, true)
    }

    /// Encode `text` treating control tokens as ordinary characters.
    ///
    /// llama.cpp's `parse_special = false`: added (`USER_DEFINED`) tokens are
    /// still split out, because those are vocabulary entries rather than
    /// control markers, but `<|im_start|>` typed in by a user is byte-merged
    /// instead of becoming the control token.
    pub fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>, TokError> {
        self.encode_with(text, false)
    }

    fn encode_with(&self, text: &str, parse_special: bool) -> Result<Vec<u32>, TokError> {
        let mut ids = Vec::new();
        // llama.cpp seeds this true so the first raw span also gets the
        // SentencePiece dummy prefix.
        let mut prev_special = true;
        for frag in self.split_specials(text, parse_special) {
            match frag {
                Frag::Special(id) => {
                    ids.push(id);
                    prev_special = true;
                }
                Frag::Text(span) => {
                    self.encode_fragment(span, prev_special, &mut ids)?;
                    prev_special = false;
                }
            }
        }
        Ok(ids)
    }

    /// Leftmost-longest scan for special tokens.
    ///
    /// llama.cpp splits by iterating its special-token list in descending
    /// length order; scanning each position for the longest match reaches the
    /// same partition for real vocabularies and is deterministic when two
    /// specials of equal length both match, which llama.cpp's unstable sort
    /// leaves open.
    fn split_specials<'a>(&self, text: &'a str, parse_special: bool) -> Vec<Frag<'a>> {
        if self.specials.is_empty() {
            return if text.is_empty() {
                Vec::new()
            } else {
                vec![Frag::Text(text)]
            };
        }
        let mut out = Vec::new();
        let mut pending = 0usize;
        let mut i = 0usize;
        while i < text.len() {
            let Some(rest) = text.get(i..) else { break };
            let Some(c) = rest.chars().next() else { break };
            let hit = self.specials.get(&c).and_then(|bucket| {
                bucket
                    .iter()
                    .find(|s| (parse_special || s.always) && rest.starts_with(&s.text))
            });
            let Some(s) = hit else {
                i = i.saturating_add(c.len_utf8());
                continue;
            };
            if let Some(before) = text.get(pending..i) {
                if !before.is_empty() {
                    out.push(Frag::Text(before));
                }
            }
            out.push(Frag::Special(s.id));
            i = i.saturating_add(s.text.len());
            pending = i;
        }
        if let Some(tail) = text.get(pending..) {
            if !tail.is_empty() {
                out.push(Frag::Text(tail));
            }
        }
        out
    }

    /// Encode one span of ordinary text between special tokens.
    fn encode_fragment(
        &self,
        text: &str,
        prev_special: bool,
        out: &mut Vec<u32>,
    ) -> Result<(), TokError> {
        match self.kind {
            Kind::Gpt2 => self.encode_gpt2(text, out),
            Kind::Spiece => self.encode_spiece(text, prev_special, out),
            Kind::Raw => self.encode_raw(text, out),
        }
    }

    /// Concatenate pieces for `ids`, mapping GPT-2 / SentencePiece symbols to UTF-8.
    ///
    /// `Ċ` becomes newline, `Ġ` becomes space, `▁` becomes space, `<0xHH>` becomes that byte.
    /// BOS and EOS ids are omitted.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            if Some(*id) == self.bos || Some(*id) == self.eos {
                continue;
            }
            let i = usize::try_from(*id).unwrap_or(0);
            if let Some(piece) = self.tokens.get(i) {
                bytes.extend(piece_bytes(piece));
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn encode_raw(&self, text: &str, out: &mut Vec<u32>) -> Result<(), TokError> {
        let mut parts: Vec<String> = Vec::new();
        for ch in text.chars() {
            let s = ch.to_string();
            if self.token_id(&s).is_none() {
                return Err(TokError::Unknown(s));
            }
            parts.push(s);
        }
        self.bpe_merge(&mut parts);
        out.extend(self.parts_to_ids(&parts)?);
        Ok(())
    }

    /// Byte-level BPE: pre-tokenize, then merge inside each piece only.
    fn encode_gpt2(&self, text: &str, out: &mut Vec<u32>) -> Result<(), TokError> {
        for piece in self.pre_tokenizer.split(text) {
            let mut parts: Vec<String> = piece
                .as_bytes()
                .iter()
                .map(|&b| gpt2_byte_to_char(b).to_string())
                .collect();
            self.bpe_merge(&mut parts);
            out.extend(self.parts_to_ids(&parts)?);
        }
        Ok(())
    }

    /// SentencePiece: `▁` for space, `<0xHH>` for anything not in the vocab.
    ///
    /// The dummy prefix goes on the first span and on any span that follows a
    /// special token, which is what llama.cpp does; a `<|user|>` marker
    /// therefore does not swallow the space in front of the next word.
    fn encode_spiece(
        &self,
        text: &str,
        prev_special: bool,
        out: &mut Vec<u32>,
    ) -> Result<(), TokError> {
        let prefixed = if self.add_space_prefix && prev_special {
            format!(" {text}")
        } else {
            text.to_string()
        };
        let normalized: String = prefixed
            .chars()
            .map(|c| if c == ' ' { '\u{2581}' } else { c })
            .collect();
        let mut parts: Vec<String> = Vec::new();
        for ch in normalized.chars() {
            let s = ch.to_string();
            if self.token_id(&s).is_some() {
                parts.push(s);
                continue;
            }
            let mut buf = [0u8; 4];
            for &b in ch.encode_utf8(&mut buf).as_bytes() {
                let hex = hex_byte_token(b);
                if self.token_id(&hex).is_none() {
                    return Err(TokError::Unknown(s));
                }
                parts.push(hex);
            }
        }
        self.bpe_merge(&mut parts);
        out.extend(self.parts_to_ids(&parts)?);
        Ok(())
    }

    fn bpe_merge(&self, parts: &mut Vec<String>) {
        loop {
            let mut best_rank = self.merges.len();
            let mut best_i = None;
            for i in 0..parts.len().saturating_sub(1) {
                let Some(a) = parts.get(i) else { continue };
                let Some(b) = parts.get(i + 1) else { continue };
                if let Some(rank) = self.merge_rank.get(&(a.clone(), b.clone())).copied() {
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
    }

    fn parts_to_ids(&self, parts: &[String]) -> Result<Vec<u32>, TokError> {
        let mut ids = Vec::new();
        for p in parts {
            ids.extend(self.piece_to_ids(p)?);
        }
        Ok(ids)
    }

    fn piece_to_ids(&self, piece: &str) -> Result<Vec<u32>, TokError> {
        if let Some(id) = self.token_id(piece) {
            return Ok(vec![id]);
        }
        let mut ids = Vec::new();
        for c in piece.chars() {
            let s = c.to_string();
            if let Some(id) = self.token_id(&s) {
                ids.push(id);
                continue;
            }
            if let Some(b) = gpt2_char_to_byte(c) {
                if let Some(id) = self.token_id(&hex_byte_token(b)) {
                    ids.push(id);
                    continue;
                }
            }
            let mut buf = [0u8; 4];
            let mut fb = Vec::new();
            let mut ok = true;
            for &b in c.encode_utf8(&mut buf).as_bytes() {
                if let Some(id) = self.token_id(&hex_byte_token(b)) {
                    fb.push(id);
                } else {
                    ok = false;
                    break;
                }
            }
            if ok && !fb.is_empty() {
                ids.extend(fb);
            } else {
                return Err(TokError::Unknown(piece.to_string()));
            }
        }
        if ids.is_empty() {
            return Err(TokError::Unknown(piece.to_string()));
        }
        Ok(ids)
    }
}

/// `tokenizer.ggml.token_type`, when it is present and sized like the vocab.
fn token_types(g: &Gguf, n_tokens: usize) -> Vec<i32> {
    let Some(Kv::Array { items, .. }) = g.kv("tokenizer.ggml.token_type") else {
        return Vec::new();
    };
    if items.len() != n_tokens {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(items.len());
    for it in items {
        out.push(match it {
            Kv::I32(v) => *v,
            Kv::U32(v) => i32::try_from(*v).unwrap_or(token_type::NORMAL),
            _ => return Vec::new(),
        });
    }
    out
}

/// Build the special-token buckets.
///
/// `token_type` is authoritative when the GGUF carries it, exactly as in
/// llama.cpp. Without it we fall back to the ids the KV keys name plus the
/// `<|...|>` added-token convention, which is the only form byte-level BPE
/// never produces by merging. `<0xHH>` byte-fallback entries are never
/// special, or every high byte in the prompt would be re-routed.
fn collect_specials(
    tokens: &[String],
    types: &[i32],
    named: &[u32],
) -> HashMap<char, Vec<Special>> {
    let mut by_first: HashMap<char, Vec<Special>> = HashMap::new();
    for (i, text) in tokens.iter().enumerate() {
        let Ok(id) = u32::try_from(i) else { continue };
        let Some(first) = text.chars().next() else {
            continue;
        };
        let ty = types.get(i).copied();
        if parse_byte_token(text).is_some() || ty == Some(token_type::BYTE) {
            continue;
        }
        let (special, always) = match ty {
            Some(token_type::USER_DEFINED) => (true, true),
            Some(token_type::CONTROL | token_type::UNKNOWN) => (true, false),
            Some(_) => (false, false),
            // No token_type array: named ids and `<|...|>` entries only.
            None => {
                let angle_pipe = text.starts_with("<|") && text.ends_with("|>") && text.len() > 4;
                let named_id = named.contains(&id);
                (angle_pipe || named_id, angle_pipe)
            }
        };
        if !special {
            continue;
        }
        by_first.entry(first).or_default().push(Special {
            text: text.clone(),
            id,
            always,
        });
    }
    for bucket in by_first.values_mut() {
        bucket.sort_by(|a, b| b.text.len().cmp(&a.text.len()).then(a.id.cmp(&b.id)));
    }
    by_first
}

fn detect_kind(model: Option<&str>, tokens: &[String]) -> Kind {
    match model {
        Some("gpt2") => Kind::Gpt2,
        Some("llama") => Kind::Spiece,
        _ => {
            let gpt2 = tokens
                .iter()
                .any(|t| t.contains('\u{0120}') || t.contains('\u{010A}'));
            let sp = tokens
                .iter()
                .any(|t| t.contains('\u{2581}') || parse_byte_token(t).is_some());
            if gpt2 {
                Kind::Gpt2
            } else if sp {
                Kind::Spiece
            } else {
                Kind::Raw
            }
        }
    }
}

fn is_gpt2_direct(n: u32) -> bool {
    (0x21..=0x7e).contains(&n) || (0xa1..=0xac).contains(&n) || (0xae..=0xff).contains(&n)
}

/// GPT-2 `bytes_to_unicode`: printable bytes stay; others map to `U+0100+`.
fn gpt2_byte_to_char(b: u8) -> char {
    let n = u32::from(b);
    if is_gpt2_direct(n) {
        return char::from_u32(n).unwrap_or('\u{fffd}');
    }
    let mut extra = 0u32;
    for x in 0u32..n {
        if !is_gpt2_direct(x) {
            extra = extra.saturating_add(1);
        }
    }
    char::from_u32(256u32.saturating_add(extra)).unwrap_or('\u{fffd}')
}

fn gpt2_char_to_byte(c: char) -> Option<u8> {
    let n = u32::from(c);
    if is_gpt2_direct(n) {
        return u8::try_from(n).ok();
    }
    if n < 256 {
        return None;
    }
    let want = n.saturating_sub(256);
    let mut seen = 0u32;
    for x in 0u32..256 {
        if !is_gpt2_direct(x) {
            if seen == want {
                return u8::try_from(x).ok();
            }
            seen = seen.saturating_add(1);
        }
    }
    None
}

fn hex_byte_token(b: u8) -> String {
    format!("<0x{b:02X}>")
}

fn parse_byte_token(s: &str) -> Option<u8> {
    let hex = s.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

fn piece_bytes(piece: &str) -> Vec<u8> {
    if let Some(b) = parse_byte_token(piece) {
        return vec![b];
    }
    let mut out = Vec::new();
    for c in piece.chars() {
        if c == '\u{2581}' {
            out.push(b' ');
            continue;
        }
        if let Some(b) = gpt2_char_to_byte(c) {
            out.push(b);
            continue;
        }
        let mut buf = [0u8; 4];
        out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{
        gpt2_byte_to_char, gpt2_char_to_byte, hex_byte_token, parse_byte_token, token_type, Kind,
        TokError, Tokenizer,
    };
    use crate::gguf::{write_gguf_with_kv, Kv};
    use crate::load_gguf;
    use crate::template::ChatMessage;

    fn strings(items: &[&str]) -> Kv {
        Kv::Array {
            elem: 8,
            items: items.iter().map(|s| Kv::String((*s).into())).collect(),
        }
    }

    fn load_tok(kv: Vec<(String, Kv)>) -> Tokenizer {
        let bytes = write_gguf_with_kv(&kv, &[]);
        let g = load_gguf(&bytes).expect("gguf");
        Tokenizer::from_gguf(&g).expect("tok")
    }

    fn gpt2_kv(extra_tokens: &[&str], merges: &[&str], chat: Option<&str>) -> Vec<(String, Kv)> {
        let mut tokens = vec!["<|endoftext|>".to_string()];
        for b in 0..=255u8 {
            tokens.push(gpt2_byte_to_char(b).to_string());
        }
        for t in extra_tokens {
            tokens.push((*t).to_string());
        }
        let token_items: Vec<Kv> = tokens.into_iter().map(Kv::String).collect();
        let merge_items: Vec<Kv> = merges.iter().map(|s| Kv::String((*s).into())).collect();
        let mut kv = vec![
            ("tokenizer.ggml.model".into(), Kv::String("gpt2".into())),
            (
                "tokenizer.ggml.tokens".into(),
                Kv::Array {
                    elem: 8,
                    items: token_items,
                },
            ),
            (
                "tokenizer.ggml.merges".into(),
                Kv::Array {
                    elem: 8,
                    items: merge_items,
                },
            ),
            ("tokenizer.ggml.bos_token_id".into(), Kv::U32(0)),
            ("tokenizer.ggml.eos_token_id".into(), Kv::U32(0)),
            ("tokenizer.ggml.add_bos_token".into(), Kv::Bool(false)),
        ];
        if let Some(t) = chat {
            kv.push(("tokenizer.chat_template".into(), Kv::String(t.into())));
        }
        kv
    }

    #[test]
    fn gpt2_bytes_to_unicode_space_and_newline() {
        assert_eq!(gpt2_byte_to_char(b' '), '\u{0120}');
        assert_eq!(gpt2_byte_to_char(b'\n'), '\u{010A}');
        assert_eq!(gpt2_byte_to_char(b'\t'), '\u{0109}');
        assert_eq!(gpt2_byte_to_char(b'a'), 'a');
        assert_eq!(gpt2_char_to_byte('\u{0120}'), Some(b' '));
        assert_eq!(gpt2_char_to_byte('\u{010A}'), Some(b'\n'));
        assert_eq!(gpt2_char_to_byte('a'), Some(b'a'));
        for b in 0..=255u8 {
            let c = gpt2_byte_to_char(b);
            assert_eq!(gpt2_char_to_byte(c), Some(b), "byte {b} char {c:?}");
        }
    }

    #[test]
    fn qwen_piece_decode_maps_c_with_dot_to_newlines() {
        let tok = load_tok(gpt2_kv(&[], &[], None));
        let nl = tok.token_id("Ċ").expect("Ċ in gpt2 vocab");
        let out = tok.decode(&[nl, nl]);
        assert_eq!(out, "\n\n");
        assert_ne!(out, "ĊĊ");
        assert!(!out.contains('Ċ'));
    }

    #[test]
    fn gpt2_encode_decode_space_newline_and_merge() {
        let tok = load_tok(gpt2_kv(&["ab", "ĊĊ"], &["a b", "Ċ Ċ"], None));
        assert_eq!(tok.kind, Kind::Gpt2);
        assert_eq!(tok.encode("ab").unwrap(), vec![tok.token_id("ab").unwrap()]);
        assert_eq!(
            tok.encode("\n\n").unwrap(),
            vec![tok.token_id("ĊĊ").unwrap()]
        );
        assert_eq!(tok.decode(&[tok.token_id("ab").unwrap()]), "ab");
        assert_eq!(tok.decode(&[tok.token_id("ĊĊ").unwrap()]), "\n\n");
        assert_eq!(tok.encode(" ").unwrap(), vec![tok.token_id("Ġ").unwrap()]);
        assert_eq!(tok.decode(&[tok.token_id("Ġ").unwrap()]), " ");
        let text = "ab \n";
        let ids = tok.encode(text).unwrap();
        assert_eq!(tok.decode(&ids), text);
        let utf8 = "café";
        assert_eq!(tok.decode(&tok.encode(utf8).unwrap()), utf8);
    }

    #[test]
    fn gpt2_byte_fallback_uses_hex_when_mapped_char_missing() {
        let mut tokens = vec!["<unk>".to_string(), "a".to_string(), "b".to_string()];
        tokens.push(hex_byte_token(b' '));
        tokens.push(hex_byte_token(10));
        let tok = load_tok(vec![
            ("tokenizer.ggml.model".into(), Kv::String("gpt2".into())),
            (
                "tokenizer.ggml.tokens".into(),
                Kv::Array {
                    elem: 8,
                    items: tokens.into_iter().map(Kv::String).collect(),
                },
            ),
        ]);
        let space = tok.token_id("<0x20>").unwrap();
        let nl = tok.token_id("<0x0A>").unwrap();
        assert_eq!(tok.encode(" ").unwrap(), vec![space]);
        assert_eq!(tok.encode("\n").unwrap(), vec![nl]);
        assert_eq!(tok.decode(&[space, nl]), " \n");
    }

    #[test]
    fn spiece_underline_and_hex_byte_fallback() {
        let mut tokens = vec![
            "<unk>".to_string(),
            "\u{2581}".to_string(),
            "a".to_string(),
            "b".to_string(),
            "ab".to_string(),
            "\u{2581}ab".to_string(),
        ];
        for b in 0..=255u8 {
            tokens.push(hex_byte_token(b));
        }
        let tok = load_tok(vec![
            ("tokenizer.ggml.model".into(), Kv::String("llama".into())),
            (
                "tokenizer.ggml.tokens".into(),
                Kv::Array {
                    elem: 8,
                    items: tokens.into_iter().map(Kv::String).collect(),
                },
            ),
            (
                "tokenizer.ggml.merges".into(),
                strings(&["a b", "\u{2581} ab"]),
            ),
            ("tokenizer.ggml.add_space_prefix".into(), Kv::Bool(true)),
        ]);
        assert_eq!(tok.kind, Kind::Spiece);
        assert!(tok.add_space_prefix);
        assert_eq!(tok.decode(&[tok.token_id("\u{2581}ab").unwrap()]), " ab");
        assert_eq!(tok.decode(&[tok.token_id("<0x0A>").unwrap()]), "\n");
        assert_eq!(
            tok.encode("ab").unwrap(),
            vec![tok.token_id("\u{2581}ab").unwrap()]
        );
        let nl = tok.encode("\n").unwrap();
        assert_eq!(tok.decode(&nl), " \n");
        let ids = tok.encode("é").unwrap();
        assert_eq!(tok.decode(&ids), " é");
    }

    #[test]
    fn token_id_is_map_lookup_not_first_of_vocab() {
        let mut tokens = Vec::new();
        for i in 0..2_000u32 {
            tokens.push(format!("pad{i}"));
        }
        tokens.push("a".into());
        tokens.push("b".into());
        tokens.push("ab".into());
        let tok = load_tok(vec![
            (
                "tokenizer.ggml.tokens".into(),
                Kv::Array {
                    elem: 8,
                    items: tokens.into_iter().map(Kv::String).collect(),
                },
            ),
            ("tokenizer.ggml.merges".into(), strings(&["a b"])),
        ]);
        assert_eq!(tok.kind, Kind::Raw);
        assert_eq!(tok.token_id("ab"), Some(2_002));
        assert_eq!(tok.token_id("pad0"), Some(0));
        assert_eq!(tok.token_id("missing"), None);
        assert_eq!(tok.encode("ab").unwrap(), vec![2_002]);
    }

    #[test]
    fn chat_template_kv_is_read() {
        let tmpl = "{% for m in messages %}{{ m.content }}{% endfor %}";
        let tok = load_tok(gpt2_kv(&[], &[], Some(tmpl)));
        assert_eq!(tok.chat_template.as_deref(), Some(tmpl));
        assert_eq!(tok.model.as_deref(), Some("gpt2"));
    }

    #[test]
    fn raw_tiny_vocab_still_merges_ab() {
        let tok = load_tok(vec![
            (
                "tokenizer.ggml.tokens".into(),
                strings(&["<unk>", "a", "b", "ab", "<s>", "</s>"]),
            ),
            ("tokenizer.ggml.merges".into(), strings(&["a b"])),
            ("tokenizer.ggml.bos_token_id".into(), Kv::U32(4)),
            ("tokenizer.ggml.eos_token_id".into(), Kv::U32(5)),
        ]);
        assert_eq!(tok.kind, Kind::Raw);
        assert_eq!(tok.encode("ab").unwrap(), vec![3]);
        assert_eq!(tok.decode(&[1, 2]), "ab");
        assert_eq!(tok.decode(&[4, 1, 2, 5]), "ab");
    }

    #[test]
    fn parse_byte_token_accepts_upper_and_lower_hex() {
        assert_eq!(parse_byte_token("<0x0A>"), Some(10));
        assert_eq!(parse_byte_token("<0x0a>"), Some(10));
        assert_eq!(parse_byte_token("<0x20>"), Some(32));
        assert_eq!(parse_byte_token("<0x>"), None);
        assert_eq!(parse_byte_token("Ċ"), None);
    }

    /// A GPT-2 byte vocab plus typed extra entries: the shape a real instruct
    /// GGUF has, with `tokenizer.ggml.token_type` parallel to the token array.
    fn typed_kv(pre: &str, extra: &[(&str, i32)], merges: &[&str]) -> Vec<(String, Kv)> {
        let mut tokens = Vec::new();
        let mut types = Vec::new();
        for b in 0..=255u8 {
            tokens.push(Kv::String(gpt2_byte_to_char(b).to_string()));
            types.push(Kv::I32(token_type::NORMAL));
        }
        for (t, ty) in extra {
            tokens.push(Kv::String((*t).to_string()));
            types.push(Kv::I32(*ty));
        }
        vec![
            ("tokenizer.ggml.model".into(), Kv::String("gpt2".into())),
            ("tokenizer.ggml.pre".into(), Kv::String(pre.into())),
            (
                "tokenizer.ggml.tokens".into(),
                Kv::Array {
                    elem: 8,
                    items: tokens,
                },
            ),
            (
                "tokenizer.ggml.token_type".into(),
                Kv::Array {
                    elem: 5,
                    items: types,
                },
            ),
            ("tokenizer.ggml.merges".into(), strings(merges)),
        ]
    }

    fn ids_of(tok: &Tokenizer, pieces: &[&str]) -> Vec<u32> {
        pieces
            .iter()
            .map(|p| tok.token_id(p).unwrap_or_else(|| panic!("{p:?} in vocab")))
            .collect()
    }

    #[test]
    fn control_tokens_are_one_id_each_and_never_byte_merged() {
        let tok = load_tok(typed_kv(
            "qwen2",
            &[
                ("<|im_start|>", token_type::CONTROL),
                ("<|im_end|>", token_type::CONTROL),
            ],
            &[],
        ));
        let ims = tok.token_id("<|im_start|>").expect("im_start");
        let ime = tok.token_id("<|im_end|>").expect("im_end");
        let ids = tok
            .encode("<|im_start|>user\nhi<|im_end|>\n")
            .expect("encode");
        assert_eq!(
            ids,
            [
                vec![ims],
                ids_of(&tok, &["u", "s", "e", "r", "Ċ", "h", "i"]),
                vec![ime],
                ids_of(&tok, &["Ċ"]),
            ]
            .concat()
        );
        assert_eq!(tok.decode(&ids), "<|im_start|>user\nhi<|im_end|>\n");

        // The same text as literal characters: llama.cpp's `parse_special=false`.
        let plain = tok
            .encode_ordinary("<|im_start|>user\nhi<|im_end|>\n")
            .expect("plain");
        assert!(
            !plain.contains(&ims),
            "control token leaked into plain text"
        );
        assert!(!plain.contains(&ime));
        assert!(plain.len() > ids.len());
        assert_eq!(tok.decode(&plain), "<|im_start|>user\nhi<|im_end|>\n");
    }

    #[test]
    fn specials_split_mid_word_and_the_longest_one_wins() {
        let tok = load_tok(typed_kv(
            "qwen2",
            &[
                ("<tool>", token_type::CONTROL),
                ("<tool_call>", token_type::CONTROL),
            ],
            &[],
        ));
        let short = tok.token_id("<tool>").expect("tool");
        let long = tok.token_id("<tool_call>").expect("tool_call");
        // Leftmost-longest: `<tool_call>` must not tokenize as `<tool>` + `_call>`.
        assert_eq!(tok.encode("<tool_call>").expect("long"), vec![long]);
        assert_eq!(tok.encode("<tool>").expect("short"), vec![short]);
        // A special glued to ordinary text still splits, with no dropped bytes.
        let ids = tok.encode("ab<tool>cd").expect("mid");
        assert_eq!(
            ids,
            [
                ids_of(&tok, &["a", "b"]),
                vec![short],
                ids_of(&tok, &["c", "d"]),
            ]
            .concat()
        );
        // A proper prefix of a special is ordinary text.
        let ids = tok.encode("<too").expect("prefix");
        assert!(!ids.contains(&short) && !ids.contains(&long));
        assert_eq!(tok.decode(&ids), "<too");
    }

    #[test]
    fn added_tokens_split_even_when_special_parsing_is_off() {
        let tok = load_tok(typed_kv(
            "qwen2",
            &[
                ("<|im_start|>", token_type::CONTROL),
                ("<pad>", token_type::USER_DEFINED),
            ],
            &[],
        ));
        let ims = tok.token_id("<|im_start|>").expect("im_start");
        let pad = tok.token_id("<pad>").expect("pad");
        // llama.cpp's `tokenizer_st_partition` always splits USER_DEFINED
        // entries; only CONTROL tokens are gated on `parse_special`.
        let ids = tok.encode_ordinary("<|im_start|><pad>").expect("encode");
        assert!(!ids.contains(&ims));
        assert!(ids.contains(&pad));
        assert_eq!(
            tok.encode("<|im_start|><pad>").expect("special"),
            vec![ims, pad]
        );
    }

    #[test]
    fn byte_fallback_entries_are_never_treated_as_special() {
        // `<0xHH>` carries token_type BYTE. Splitting on it would re-route every
        // high byte in the prompt through the special path.
        let mut extra: Vec<(String, i32)> = Vec::new();
        for b in 0..=255u8 {
            extra.push((hex_byte_token(b), token_type::BYTE));
        }
        let extra_refs: Vec<(&str, i32)> = extra.iter().map(|(s, t)| (s.as_str(), *t)).collect();
        let tok = load_tok(typed_kv("qwen2", &extra_refs, &[]));
        assert!(tok.special_tokens().is_empty());
        let hex = tok.token_id("<0x41>").expect("<0x41>");
        assert!(!tok.is_special(hex));
        // Typing the literal text `<0x41>` must not produce the byte token.
        let ids = tok.encode("<0x41>").expect("encode");
        assert!(!ids.contains(&hex));
        assert_eq!(tok.decode(&ids), "<0x41>");
    }

    #[test]
    fn without_token_type_the_angle_pipe_convention_and_named_ids_are_special() {
        let mut kv = gpt2_kv(&["<|im_start|>", "<s>", "not|special"], &[], None);
        // `<s>` is only special because a KV key names it.
        kv.push(("tokenizer.ggml.bos_token_id".into(), Kv::U32(258)));
        let tok = load_tok(kv);
        assert_eq!(tok.token_id("<s>"), Some(258));
        let ims = tok.token_id("<|im_start|>").expect("im_start");
        let bos = tok.token_id("<s>").expect("s");
        assert!(tok.is_special(ims));
        assert!(tok.is_special(bos));
        assert!(!tok.is_special(tok.token_id("not|special").expect("plain")));
        assert_eq!(tok.encode("<|im_start|>").expect("ims"), vec![ims]);
        assert_eq!(tok.encode("<s>").expect("bos"), vec![bos]);
    }

    #[test]
    fn special_tokens_are_listed_longest_first() {
        let tok = load_tok(typed_kv(
            "qwen2",
            &[
                ("<a>", token_type::CONTROL),
                ("<|longer_one|>", token_type::CONTROL),
                ("<pad>", token_type::USER_DEFINED),
            ],
            &[],
        ));
        let listed: Vec<&str> = tok.special_tokens().into_iter().map(|(t, _)| t).collect();
        assert_eq!(listed, vec!["<|longer_one|>", "<pad>", "<a>"]);
    }

    #[test]
    fn merges_cannot_cross_a_pre_token_boundary() {
        // `i Ġ` would glue the end of one word to the space starting the next.
        // The pre-tokenizer regex splits `hi there` into `hi` + ` there`, so
        // that pair is never adjacent and the merge cannot fire.
        let tok = load_tok(typed_kv(
            "qwen2",
            &[("iĠ", token_type::NORMAL), ("th", token_type::NORMAL)],
            &["i Ġ", "t h"],
        ));
        let glued = tok.token_id("iĠ").expect("iĠ");
        let ids = tok.encode("hi there").expect("encode");
        assert!(!ids.contains(&glued), "merge crossed a pre-token boundary");
        assert_eq!(ids, ids_of(&tok, &["h", "i", "Ġ", "th", "e", "r", "e"]));
        assert_eq!(tok.decode(&ids), "hi there");
    }

    #[test]
    fn spiece_dummy_prefix_restarts_after_a_special_token() {
        let mut tokens = vec![
            Kv::String("<unk>".into()),
            Kv::String("\u{2581}".into()),
            Kv::String("a".into()),
            Kv::String("b".into()),
            Kv::String("ab".into()),
            Kv::String("\u{2581}ab".into()),
            Kv::String("<|user|>".into()),
        ];
        let mut types = vec![
            Kv::I32(token_type::UNKNOWN),
            Kv::I32(token_type::NORMAL),
            Kv::I32(token_type::NORMAL),
            Kv::I32(token_type::NORMAL),
            Kv::I32(token_type::NORMAL),
            Kv::I32(token_type::NORMAL),
            Kv::I32(token_type::CONTROL),
        ];
        for b in 0..=255u8 {
            tokens.push(Kv::String(hex_byte_token(b)));
            types.push(Kv::I32(token_type::BYTE));
        }
        let tok = load_tok(vec![
            ("tokenizer.ggml.model".into(), Kv::String("llama".into())),
            (
                "tokenizer.ggml.tokens".into(),
                Kv::Array {
                    elem: 8,
                    items: tokens,
                },
            ),
            (
                "tokenizer.ggml.token_type".into(),
                Kv::Array {
                    elem: 5,
                    items: types,
                },
            ),
            (
                "tokenizer.ggml.merges".into(),
                strings(&["a b", "\u{2581} ab"]),
            ),
            ("tokenizer.ggml.add_space_prefix".into(), Kv::Bool(true)),
        ]);
        assert_eq!(tok.kind, Kind::Spiece);
        let user = tok.token_id("<|user|>").expect("user");
        let sp_ab = tok.token_id("\u{2581}ab").expect("▁ab");
        let ab = tok.token_id("ab").expect("ab");
        // First fragment gets the prefix, and so does the one after a special.
        assert_eq!(tok.encode("ab").expect("plain"), vec![sp_ab]);
        assert_eq!(tok.encode("<|user|>ab").expect("after"), vec![user, sp_ab]);
        // A fragment that merely follows text does not get a second prefix.
        assert_eq!(
            tok.encode("ab<|user|>ab").expect("both"),
            vec![sp_ab, user, sp_ab]
        );
        assert!(!tok.encode("ab").expect("plain").contains(&ab));
    }

    /// Verbatim `tokenizer.chat_template` from `Qwen/Qwen2.5-0.5B-Instruct`.
    const QWEN25_TEMPLATE: &str = include_str!("template/testdata/qwen2_5.jinja");

    #[test]
    fn apply_chat_template_renders_and_encodes_to_single_control_ids() {
        let tok = load_tok(typed_kv(
            "qwen2",
            &[
                ("<|im_start|>", token_type::CONTROL),
                ("<|im_end|>", token_type::CONTROL),
            ],
            &[],
        ));
        let mut kv = typed_kv(
            "qwen2",
            &[
                ("<|im_start|>", token_type::CONTROL),
                ("<|im_end|>", token_type::CONTROL),
            ],
            &[],
        );
        kv.push((
            "tokenizer.chat_template".into(),
            Kv::String(QWEN25_TEMPLATE.into()),
        ));
        kv.push((
            "tokenizer.ggml.eos_token_id".into(),
            Kv::U32(tok.token_id("<|im_end|>").expect("im_end")),
        ));
        let tok = load_tok(kv);
        let ims = tok.token_id("<|im_start|>").expect("im_start");
        let ime = tok.token_id("<|im_end|>").expect("im_end");
        let prompt = tok
            .apply_chat_template(
                &[ChatMessage::system("Be terse."), ChatMessage::user("Hi")],
                true,
            )
            .expect("render");
        assert_eq!(
            prompt,
            "<|im_start|>system\nBe terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
        let ids = tok.encode(&prompt).expect("encode");
        assert_eq!(ids.iter().filter(|i| **i == ims).count(), 3);
        assert_eq!(ids.iter().filter(|i| **i == ime).count(), 2);
        assert_eq!(ids.first().copied(), Some(ims));
        assert_eq!(ids.last().copied(), tok.token_id("Ċ"));
        assert_eq!(tok.decode(&ids), prompt.replace("<|im_end|>", ""));
    }

    #[test]
    fn apply_chat_template_without_one_is_an_error_not_a_guess() {
        let tok = load_tok(gpt2_kv(&[], &[], None));
        let err = tok
            .apply_chat_template(&[ChatMessage::user("Hi")], true)
            .expect_err("no template");
        assert_eq!(err, TokError::NoChatTemplate);
        assert!(err.to_string().contains("chat_template"));
    }

    #[test]
    fn a_template_that_rejects_the_conversation_surfaces_its_message() {
        let mut kv = gpt2_kv(&[], &[], None);
        kv.push((
            "tokenizer.chat_template".into(),
            Kv::String(
                "{% for m in messages %}{% if m.role == 'system' %}\
                 {{ raise_exception('System role not supported') }}{% endif %}{% endfor %}"
                    .into(),
            ),
        ));
        let tok = load_tok(kv);
        let err = tok
            .apply_chat_template(&[ChatMessage::system("no")], false)
            .expect_err("raised");
        assert!(
            err.to_string().contains("System role not supported"),
            "{err}"
        );
    }
}
