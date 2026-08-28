//! GGUF-embedded tokenizer: vocab tokens plus optional BPE merges.
//!
//! GPT-2 / Qwen pieces use the bytes-to-unicode map (`Ċ` is newline, `Ġ` is space).
//! SentencePiece pieces use `▁` for space and `<0xHH>` byte-fallback tokens.
//! Token and merge lookup is by `HashMap`, not a linear scan of the vocab.

use std::collections::HashMap;

use crate::gguf::{Gguf, GgufError, Kv};

/// Failure while encoding, decoding, or reading tokenizer KV.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokError {
    /// `tokenizer.ggml.tokens` missing or not a string array.
    Vocab,
    /// A piece of the prompt is not in the vocab and has no byte-fallback token.
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
    /// `tokenizer.chat_template` Jinja, if present. Stored, not rendered.
    pub chat_template: Option<String>,
    /// SentencePiece dummy-prefix (`▁` before the first symbol).
    pub add_space_prefix: bool,
    token_to_id: HashMap<String, u32>,
    merge_rank: HashMap<(String, String), usize>,
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
        Ok(Self::with_indexes(Self {
            token_to_id: HashMap::new(),
            merge_rank: HashMap::new(),
            kind,
            tokens,
            merges,
            bos: g.kv_u32("tokenizer.ggml.bos_token_id"),
            eos: g.kv_u32("tokenizer.ggml.eos_token_id"),
            add_bos: g.kv_bool("tokenizer.ggml.add_bos_token").unwrap_or(true),
            model,
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

    /// Id of an exact vocab piece. `O(1)` map lookup, not a scan of `tokens`.
    pub fn token_id(&self, piece: &str) -> Option<u32> {
        self.token_to_id.get(piece).copied()
    }

    /// Encode `text` with byte-level GPT-2 BPE, SentencePiece + `<0xHH>`, or raw chars.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokError> {
        match self.kind {
            Kind::Gpt2 => self.encode_gpt2(text),
            Kind::Spiece => self.encode_spiece(text),
            Kind::Raw => self.encode_raw(text),
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

    fn encode_raw(&self, text: &str) -> Result<Vec<u32>, TokError> {
        let mut parts: Vec<String> = Vec::new();
        for ch in text.chars() {
            let s = ch.to_string();
            if self.token_id(&s).is_none() {
                return Err(TokError::Unknown(s));
            }
            parts.push(s);
        }
        self.bpe_merge(&mut parts);
        self.parts_to_ids(&parts)
    }

    fn encode_gpt2(&self, text: &str) -> Result<Vec<u32>, TokError> {
        let mut parts: Vec<String> = Vec::new();
        for &b in text.as_bytes() {
            parts.push(gpt2_byte_to_char(b).to_string());
        }
        self.bpe_merge(&mut parts);
        self.parts_to_ids(&parts)
    }

    fn encode_spiece(&self, text: &str) -> Result<Vec<u32>, TokError> {
        let replaced: String = text
            .chars()
            .map(|c| if c == ' ' { '\u{2581}' } else { c })
            .collect();
        let normalized = if self.add_space_prefix && !replaced.starts_with('\u{2581}') {
            format!("\u{2581}{replaced}")
        } else {
            replaced
        };
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
        self.parts_to_ids(&parts)
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
        gpt2_byte_to_char, gpt2_char_to_byte, hex_byte_token, parse_byte_token, Kind, Tokenizer,
    };
    use crate::gguf::load_gguf;
    use crate::gguf::{write_gguf_with_kv, Kv};

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
}
