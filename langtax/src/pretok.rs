//! Hand-written GPT-2 / Qwen2 / Llama-3 BPE pre-tokenizer. No regex crate.
//!
//! Byte-level BPE never merges across a pre-token boundary, so the split has to
//! agree with the reference regex exactly or the ids diverge from llama.cpp and
//! HuggingFace on ordinary prose. The three patterns implemented here are the
//! ones llama.cpp keys off `tokenizer.ggml.pre`:
//!
//! ```text
//! gpt2   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//! qwen2  (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//! llama3 (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//! ```
//!
//! Alternation is leftmost-first with greedy quantifiers and backtracking, which
//! is what both Oniguruma (HuggingFace `tokenizers`) and the Rust `regex` crate
//! do, and what llama.cpp's `unicode_regex_split_custom_*` hand-codes. Every
//! code point is matched by some alternative, so the scan never skips input.
//!
//! `\p{L}` and `\p{N}` come from [`crate::ucd`], not [`char::is_alphabetic`].
//! `\s` is [`char::is_whitespace`] (the Unicode `White_Space` property), which
//! is llama.cpp's `unicode_set_whitespace`. Python's `regex` module additionally
//! treats `U+001C..U+001F` as `\s`; those four control characters are the only
//! known divergence and llama.cpp does not treat them as whitespace either.

use crate::ucd::{is_letter, is_number};

/// Which pre-tokenizer regex to emulate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreTokenizer {
    /// Original GPT-2 pattern. llama.cpp `LLAMA_VOCAB_PRE_TYPE_DEFAULT`/`GPT2`.
    Gpt2,
    /// Qwen2 / Qwen2.5 / Qwen3 pattern: single-digit numbers, newline runs.
    Qwen2,
    /// Llama-3 pattern: like Qwen2 but numbers group in runs of up to three.
    Llama3,
}

/// The seven English contractions the patterns special-case, in regex order.
const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

impl PreTokenizer {
    /// Map a GGUF `tokenizer.ggml.pre` value to a pattern.
    ///
    /// Unrecognised values fall back to [`PreTokenizer::Gpt2`], which is what
    /// llama.cpp's `default` pre-type uses.
    pub fn from_pre_key(pre: Option<&str>) -> Self {
        match pre {
            Some("llama3" | "llama-v3" | "llama-bpe") => Self::Llama3,
            Some("qwen2" | "deepseek-r1-qwen") => Self::Qwen2,
            _ => Self::Gpt2,
        }
    }

    /// Split `text` into pre-token spans. The spans concatenate back to `text`.
    pub fn split<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut out = Vec::new();
        let mut i = 0usize;
        while i < text.len() {
            let end = match *self {
                Self::Gpt2 => gpt2_match(text, i),
                Self::Qwen2 => qwen_match(text, i, 1),
                Self::Llama3 => qwen_match(text, i, 3),
            };
            // Every code point is covered by some alternative; `step` only
            // guards against a pathological zero-width match looping forever.
            let end = if end > i { end } else { step(text, i) };
            if let Some(span) = text.get(i..end) {
                out.push(span);
            }
            i = end;
        }
        out
    }
}

fn char_at(s: &str, i: usize) -> Option<char> {
    s.get(i..)?.chars().next()
}

fn step(s: &str, i: usize) -> usize {
    match char_at(s, i) {
        Some(c) => i.saturating_add(c.len_utf8()),
        None => s.len(),
    }
}

/// End offset of the longest run of `pred` starting at `i`; `i` when empty.
fn run(s: &str, i: usize, pred: impl Fn(char) -> bool) -> usize {
    let mut j = i;
    while let Some(c) = char_at(s, j) {
        if !pred(c) {
            break;
        }
        j = j.saturating_add(c.len_utf8());
    }
    j
}

/// End offset of a run of at most `max` chars matching `pred`; `i` when empty.
fn run_max(s: &str, i: usize, max: usize, pred: impl Fn(char) -> bool) -> usize {
    let mut j = i;
    let mut n = 0usize;
    while n < max {
        let Some(c) = char_at(s, j) else { break };
        if !pred(c) {
            break;
        }
        j = j.saturating_add(c.len_utf8());
        n = n.saturating_add(1);
    }
    j
}

/// `[^\s\p{L}\p{N}]`: the "everything else" class, punctuation and symbols.
fn is_other(c: char) -> bool {
    !c.is_whitespace() && !is_letter(c) && !is_number(c)
}

/// `[^\r\n\p{L}\p{N}]`: the optional single-character prefix of a letter run.
fn is_letter_prefix(c: char) -> bool {
    c != '\r' && c != '\n' && !is_letter(c) && !is_number(c)
}

fn is_newline(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// `'s|'t|'re|'ve|'m|'ll|'d`, optionally case-insensitive.
fn match_contraction(s: &str, i: usize, fold_case: bool) -> Option<usize> {
    let rest = s.get(i..)?;
    for lit in CONTRACTIONS {
        let Some(head) = rest.get(..lit.len()) else {
            continue;
        };
        let hit = if fold_case {
            head.eq_ignore_ascii_case(lit)
        } else {
            head == lit
        };
        if hit {
            return Some(i.saturating_add(lit.len()));
        }
    }
    None
}

/// ` ?X+` with the greedy `?` and its one backtrack step.
fn opt_space_run(s: &str, i: usize, pred: impl Fn(char) -> bool + Copy) -> Option<usize> {
    if char_at(s, i) == Some(' ') {
        let after = i.saturating_add(1);
        let end = run(s, after, pred);
        if end > after {
            return Some(end);
        }
    }
    let end = run(s, i, pred);
    if end > i {
        Some(end)
    } else {
        None
    }
}

/// `\s+(?!\S)` then `\s+`: the trailing pair both patterns end with.
///
/// Greedy `\s+` swallows the whole whitespace run, so the lookahead only holds
/// at end of input; otherwise it backtracks one code point, which leaves the
/// run minus its last character. When that would be empty the alternative fails
/// and the bare `\s+` takes the whole run.
fn whitespace_tail(s: &str, i: usize) -> Option<usize> {
    let end = run(s, i, char::is_whitespace);
    if end == i {
        return None;
    }
    if end >= s.len() {
        return Some(end);
    }
    let mut last = i;
    let mut j = i;
    while j < end {
        last = j;
        j = step(s, j);
    }
    if last > i {
        Some(last)
    } else {
        Some(end)
    }
}

fn gpt2_match(s: &str, i: usize) -> usize {
    if let Some(e) = match_contraction(s, i, false) {
        return e;
    }
    if let Some(e) = opt_space_run(s, i, is_letter) {
        return e;
    }
    if let Some(e) = opt_space_run(s, i, is_number) {
        return e;
    }
    if let Some(e) = opt_space_run(s, i, is_other) {
        return e;
    }
    whitespace_tail(s, i).unwrap_or(i)
}

/// Qwen2 (`num_max` = 1) and Llama-3 (`num_max` = 3) share everything else.
fn qwen_match(s: &str, i: usize, num_max: usize) -> usize {
    if let Some(e) = match_contraction(s, i, true) {
        return e;
    }
    // [^\r\n\p{L}\p{N}]?\p{L}+
    if let Some(c) = char_at(s, i) {
        if is_letter_prefix(c) {
            let after = i.saturating_add(c.len_utf8());
            let end = run(s, after, is_letter);
            if end > after {
                return end;
            }
        }
    }
    let end = run(s, i, is_letter);
    if end > i {
        return end;
    }
    // \p{N} or \p{N}{1,3}
    let end = run_max(s, i, num_max, is_number);
    if end > i {
        return end;
    }
    // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
    if let Some(end) = opt_space_run(s, i, is_other) {
        return run(s, end, is_newline);
    }
    // '\s*[\r\n]+': the whitespace run up to and including its last newline.
    let ws_end = run(s, i, char::is_whitespace);
    if ws_end > i {
        let mut last_nl = None;
        let mut j = i;
        while j < ws_end {
            if matches!(char_at(s, j), Some(c) if is_newline(c)) {
                last_nl = Some(step(s, j));
            }
            j = step(s, j);
        }
        if let Some(end) = last_nl {
            return end;
        }
    }
    whitespace_tail(s, i).unwrap_or(i)
}

#[cfg(test)]
mod tests {
    use super::PreTokenizer;

    fn gpt2(text: &str) -> Vec<&str> {
        PreTokenizer::Gpt2.split(text)
    }

    fn qwen2(text: &str) -> Vec<&str> {
        PreTokenizer::Qwen2.split(text)
    }

    fn llama3(text: &str) -> Vec<&str> {
        PreTokenizer::Llama3.split(text)
    }

    /// Every pattern must be a partition: the spans rebuild the input exactly.
    fn assert_partition(pre: PreTokenizer, text: &str) {
        let parts = pre.split(text);
        assert_eq!(parts.concat(), text, "{pre:?} lost bytes on {text:?}");
        for p in &parts {
            assert!(!p.is_empty(), "{pre:?} produced an empty span on {text:?}");
        }
    }

    #[test]
    fn pre_key_maps_to_llama_cpp_pre_types() {
        assert_eq!(PreTokenizer::from_pre_key(None), PreTokenizer::Gpt2);
        assert_eq!(
            PreTokenizer::from_pre_key(Some("default")),
            PreTokenizer::Gpt2
        );
        assert_eq!(
            PreTokenizer::from_pre_key(Some("gpt-2")),
            PreTokenizer::Gpt2
        );
        assert_eq!(
            PreTokenizer::from_pre_key(Some("qwen2")),
            PreTokenizer::Qwen2
        );
        assert_eq!(
            PreTokenizer::from_pre_key(Some("llama-bpe")),
            PreTokenizer::Llama3
        );
        assert_eq!(
            PreTokenizer::from_pre_key(Some("llama3")),
            PreTokenizer::Llama3
        );
        // Unknown pre-types fall back to GPT-2, like llama.cpp's `default`.
        assert_eq!(
            PreTokenizer::from_pre_key(Some("some-future-model")),
            PreTokenizer::Gpt2
        );
    }

    // Expected splits below were captured from Python
    // `regex.findall(pattern, text)` using the exact llama.cpp pattern strings.

    #[test]
    fn gpt2_contractions_and_leading_spaces() {
        assert_eq!(gpt2("Hello world"), vec!["Hello", " world"]);
        assert_eq!(gpt2("don't"), vec!["don", "'t"]);
        assert_eq!(
            gpt2("I'm sure it's fine"),
            vec!["I", "'m", " sure", " it", "'s", " fine"]
        );
        assert_eq!(
            gpt2("we've you're I'd we'll"),
            vec!["we", "'ve", " you", "'re", " I", "'d", " we", "'ll"]
        );
        // GPT-2 does not fold case in the contraction alternative, so the
        // apostrophe joins the punctuation class instead.
        assert_eq!(gpt2("DON'T"), vec!["DON", "'", "T"]);
    }

    #[test]
    fn gpt2_whitespace_runs_and_trailing_space() {
        assert_eq!(gpt2("a  b"), vec!["a", " ", " b"]);
        assert_eq!(gpt2("a   b"), vec!["a", "  ", " b"]);
        assert_eq!(gpt2("a "), vec!["a", " "]);
        assert_eq!(gpt2("a  "), vec!["a", "  "]);
        assert_eq!(gpt2(" a"), vec![" a"]);
        assert_eq!(gpt2("   "), vec!["   "]);
        assert_eq!(gpt2("\n\n"), vec!["\n\n"]);
        assert_eq!(gpt2("a\n\nb"), vec!["a", "\n", "\n", "b"]);
        assert_eq!(gpt2("hi\tthere"), vec!["hi", "\t", "there"]);
    }

    #[test]
    fn gpt2_digits_punctuation_and_mixed() {
        assert_eq!(gpt2("1234"), vec!["1234"]);
        assert_eq!(gpt2("a1"), vec!["a", "1"]);
        assert_eq!(gpt2("3.14"), vec!["3", ".", "14"]);
        assert_eq!(gpt2("!!!"), vec!["!!!"]);
        assert_eq!(gpt2("x = 1;"), vec!["x", " =", " 1", ";"]);
        assert_eq!(gpt2("(a, b)"), vec!["(", "a", ",", " b", ")"]);
        assert_eq!(gpt2("<|im_start|>"), vec!["<|", "im", "_", "start", "|>"]);
    }

    #[test]
    fn gpt2_cjk_emoji_and_mixed_scripts() {
        assert_eq!(gpt2("你好世界"), vec!["你好世界"]);
        assert_eq!(gpt2("hello 你好"), vec!["hello", " 你好"]);
        assert_eq!(gpt2("Привет мир"), vec!["Привет", " мир"]);
        assert_eq!(gpt2("日本語123"), vec!["日本語", "123"]);
        // Emoji are symbols, so they land in the punctuation class and glue to
        // a preceding space.
        assert_eq!(gpt2("hi 🙂🙂"), vec!["hi", " 🙂🙂"]);
        assert_eq!(gpt2("🙂a"), vec!["🙂", "a"]);
    }

    #[test]
    fn gpt2_marks_are_not_letters_which_is_where_is_alphabetic_would_differ() {
        // Devanagari: the virama and vowel signs are `M*`, not `\p{L}`, so the
        // reference regex breaks the cluster. `char::is_alphabetic` would not.
        assert_eq!(gpt2("नमस्ते"), vec!["नमस", "्", "त", "े"]);
        // Hebrew with points, same story.
        assert_eq!(gpt2("שָׁלוֹם"), vec!["ש", "ָׁ", "לו", "ֹ", "ם"]);
        // A roman numeral is `Nl`: alphabetic to `std`, a number to the regex.
        assert_eq!(gpt2("aⅧb"), vec!["a", "Ⅷ", "b"]);
    }

    #[test]
    fn qwen2_folds_case_and_splits_digits_one_at_a_time() {
        assert_eq!(qwen2("Hello world"), vec!["Hello", " world"]);
        assert_eq!(qwen2("DON'T"), vec!["DON", "'T"]);
        assert_eq!(qwen2("don't"), vec!["don", "'t"]);
        assert_eq!(qwen2("1234"), vec!["1", "2", "3", "4"]);
        assert_eq!(qwen2("a1"), vec!["a", "1"]);
        // `[^\r\n\p{L}\p{N}]?\p{L}+` lets punctuation lead a word.
        assert_eq!(qwen2("_start"), vec!["_start"]);
        assert_eq!(qwen2("<|im_start|>"), vec!["<|", "im", "_start", "|>"]);
        assert_eq!(qwen2("\tword"), vec!["\tword"]);
    }

    #[test]
    fn qwen2_newline_runs_hang_together() {
        assert_eq!(qwen2("a\n\nb"), vec!["a", "\n\n", "b"]);
        assert_eq!(qwen2("a  \n  b"), vec!["a", "  \n", " ", " b"]);
        assert_eq!(qwen2("a\r\nb"), vec!["a", "\r\n", "b"]);
        assert_eq!(qwen2("...\n\n"), vec!["...\n\n"]);
        assert_eq!(qwen2("a  b"), vec!["a", " ", " b"]);
        assert_eq!(qwen2("   "), vec!["   "]);
    }

    #[test]
    fn llama3_groups_digits_in_threes() {
        assert_eq!(llama3("1234567"), vec!["123", "456", "7"]);
        assert_eq!(llama3("12"), vec!["12"]);
        assert_eq!(llama3("DON'T"), vec!["DON", "'T"]);
        assert_eq!(llama3("a\n\nb"), vec!["a", "\n\n", "b"]);
        assert_eq!(llama3("Hello world"), vec!["Hello", " world"]);
        assert_eq!(
            llama3("<|begin_of_text|>"),
            vec!["<|", "begin", "_of", "_text", "|>"]
        );
    }

    #[test]
    fn every_pattern_partitions_a_stress_corpus() {
        let corpus = [
            "",
            " ",
            "\n",
            "\r\n\r\n",
            "a",
            "The quick brown fox jumps over the lazy dog.",
            "It's 3:45pm — don't be late!!",
            "def f(x):\n    return x ** 2  # square\n",
            "混合 mixed テキスト 123 ٣٤٥",
            "🙂🇬🇧👩‍💻 family",
            "नमस्ते दुनिया",
            "  leading and trailing  ",
            "tabs\tand\u{000B}vertical",
            "\u{00A0}nbsp\u{2003}emspace",
            "<|im_start|>user\nhi<|im_end|>\n",
        ];
        for pre in [
            PreTokenizer::Gpt2,
            PreTokenizer::Qwen2,
            PreTokenizer::Llama3,
        ] {
            for text in corpus {
                assert_partition(pre, text);
            }
        }
    }
}
