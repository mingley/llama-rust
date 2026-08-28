//! Template source to text / expression-token pieces.
//!
//! Whitespace control is the part that silently corrupts prompts when it is
//! wrong, so it follows the environment HuggingFace `transformers` builds in
//! `_compile_jinja_template`: `trim_blocks=True` and `lstrip_blocks=True`.
//!
//! * `{%- ` / `{{- ` / `{#- ` strip *all* whitespace before the tag.
//! * ` -%}` / ` -}}` / ` -#}` strip *all* whitespace after the tag.
//! * `lstrip_blocks` additionally drops a run of spaces and tabs that reaches
//!   back to the start of a line before a `{%` or `{#` tag. `{%+` opts out.
//! * `trim_blocks` additionally drops one `\n` directly after a `%}` or `#}`.
//!
//! Neither default applies to `{{ }}`, which is why templates indent their
//! `{{- ... }}` lines and rely on the explicit minus.

use super::TemplateError;

/// One expression token inside `{{ }}` or `{% %}`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Tok {
    /// Identifier or keyword.
    Name(String),
    /// String literal, already unescaped.
    Str(String),
    /// Integer literal.
    Int(i64),
    /// Operator or delimiter.
    Punct(&'static str),
}

impl Tok {
    /// Whether this is the given keyword or identifier.
    pub(crate) fn is_name(&self, want: &str) -> bool {
        matches!(self, Self::Name(n) if n == want)
    }

    /// Whether this is the given operator or delimiter.
    pub(crate) fn is_punct(&self, want: &str) -> bool {
        matches!(self, Self::Punct(p) if *p == want)
    }
}

/// A lexed template piece.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Piece {
    /// Literal output text.
    Text(String),
    /// `{{ ... }}`.
    Expr(Vec<Tok>),
    /// `{% ... %}`.
    Block(Vec<Tok>),
}

/// Operators, longest first so `==` wins over `=`.
const PUNCT: [&str; 24] = [
    "//", "==", "!=", ">=", "<=", "**", "(", ")", "[", "]", "{", "}", ",", ":", ".", "|", "=", "<",
    ">", "+", "-", "*", "/", "%",
];

/// Split template source into text and tag pieces.
pub(crate) fn lex(src: &str) -> Result<Vec<Piece>, TemplateError> {
    let mut out: Vec<Piece> = Vec::new();
    let mut text_start = 0usize;
    let mut i = 0usize;
    while i < src.len() {
        let Some(rest) = src.get(i..) else { break };
        let kind = if rest.starts_with("{{") {
            TagKind::Expr
        } else if rest.starts_with("{%") {
            TagKind::Block
        } else if rest.starts_with("{#") {
            TagKind::Comment
        } else {
            i = next_char(src, i);
            continue;
        };
        let open_end = i.saturating_add(2);
        let (strip_left, body_start) = match src.get(open_end..) {
            Some(r) if r.starts_with('-') => (true, open_end.saturating_add(1)),
            Some(r) if r.starts_with('+') => (false, open_end.saturating_add(1)),
            _ => (false, open_end),
        };
        let plus = matches!(src.get(open_end..), Some(r) if r.starts_with('+'));
        let close = kind.close();
        let Some(rel) = find(src, body_start, close) else {
            return Err(TemplateError::Syntax(format!(
                "unclosed {} tag",
                kind.name()
            )));
        };
        let strip_right =
            rel > body_start && matches!(src.get(rel.saturating_sub(1)..rel), Some("-"));
        let body_end = if strip_right {
            rel.saturating_sub(1)
        } else {
            rel
        };
        let after = rel.saturating_add(close.len());

        let mut text = src.get(text_start..i).unwrap_or("").to_string();
        if strip_left {
            text.truncate(text.trim_end().len());
        } else if kind.lstrip_default() && !plus {
            lstrip_block(&mut text, text_start == 0);
        }
        if !text.is_empty() {
            out.push(Piece::Text(text));
        }

        let body = src.get(body_start..body_end).unwrap_or("");
        match kind {
            TagKind::Expr => out.push(Piece::Expr(lex_tokens(body)?)),
            TagKind::Block => out.push(Piece::Block(lex_tokens(body)?)),
            TagKind::Comment => {}
        }

        text_start = after;
        if strip_right {
            while let Some(c) = src.get(text_start..).and_then(|r| r.chars().next()) {
                if !c.is_whitespace() {
                    break;
                }
                text_start = text_start.saturating_add(c.len_utf8());
            }
        } else if kind.trim_default()
            && matches!(src.get(text_start..), Some(r) if r.starts_with('\n'))
        {
            text_start = text_start.saturating_add(1);
        }
        i = text_start;
    }
    let tail = src.get(text_start..).unwrap_or("");
    if !tail.is_empty() {
        out.push(Piece::Text(tail.to_string()));
    }
    Ok(out)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TagKind {
    Expr,
    Block,
    Comment,
}

impl TagKind {
    fn close(self) -> &'static str {
        match self {
            Self::Expr => "}}",
            Self::Block => "%}",
            Self::Comment => "#}",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Expr => "{{",
            Self::Block => "{%",
            Self::Comment => "{#",
        }
    }

    /// `lstrip_blocks` and `trim_blocks` apply to block and comment tags only.
    fn lstrip_default(self) -> bool {
        self != Self::Expr
    }

    fn trim_default(self) -> bool {
        self != Self::Expr
    }
}

/// Drop a spaces-and-tabs run that reaches back to the start of a line.
fn lstrip_block(text: &mut String, at_source_start: bool) {
    let trimmed = text.trim_end_matches([' ', '\t']);
    if trimmed.len() == text.len() {
        return;
    }
    if trimmed.ends_with('\n') || (trimmed.is_empty() && at_source_start) {
        text.truncate(trimmed.len());
    }
}

fn next_char(src: &str, i: usize) -> usize {
    match src.get(i..).and_then(|r| r.chars().next()) {
        Some(c) => i.saturating_add(c.len_utf8()),
        None => src.len(),
    }
}

/// Byte offset of `needle` at or after `from`, skipping over string literals so
/// a `%}` inside `'...'` does not close the tag.
fn find(src: &str, from: usize, needle: &str) -> Option<usize> {
    let mut i = from;
    while i < src.len() {
        let rest = src.get(i..)?;
        if rest.starts_with(needle) {
            return Some(i);
        }
        let c = rest.chars().next()?;
        if c == '\'' || c == '"' {
            i = skip_string(src, i, c)?;
            continue;
        }
        i = i.saturating_add(c.len_utf8());
    }
    None
}

/// Offset just past a quoted literal starting at `i`.
fn skip_string(src: &str, i: usize, quote: char) -> Option<usize> {
    let mut j = i.saturating_add(quote.len_utf8());
    loop {
        let c = src.get(j..)?.chars().next()?;
        j = j.saturating_add(c.len_utf8());
        if c == '\\' {
            let esc = src.get(j..)?.chars().next()?;
            j = j.saturating_add(esc.len_utf8());
            continue;
        }
        if c == quote {
            return Some(j);
        }
    }
}

/// Tokenize the inside of one tag.
fn lex_tokens(body: &str) -> Result<Vec<Tok>, TemplateError> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < body.len() {
        let Some(rest) = body.get(i..) else { break };
        let Some(c) = rest.chars().next() else { break };
        if c.is_whitespace() {
            i = i.saturating_add(c.len_utf8());
            continue;
        }
        if c == '\'' || c == '"' {
            let (s, next) = lex_string(body, i, c)?;
            out.push(Tok::Str(s));
            i = next;
            continue;
        }
        if c.is_ascii_digit() {
            let (n, next) = lex_number(body, i)?;
            out.push(Tok::Int(n));
            i = next;
            continue;
        }
        if c == '_' || c.is_alphabetic() {
            let mut j = i;
            while let Some(ch) = body.get(j..).and_then(|r| r.chars().next()) {
                if ch == '_' || ch.is_alphanumeric() {
                    j = j.saturating_add(ch.len_utf8());
                } else {
                    break;
                }
            }
            out.push(Tok::Name(body.get(i..j).unwrap_or("").to_string()));
            i = j;
            continue;
        }
        let mut matched = None;
        for p in PUNCT {
            if rest.starts_with(p) {
                matched = Some(p);
                break;
            }
        }
        let Some(p) = matched else {
            return Err(TemplateError::Syntax(format!(
                "unexpected character {c:?} in expression {body:?}"
            )));
        };
        out.push(Tok::Punct(p));
        i = i.saturating_add(p.len());
    }
    Ok(out)
}

fn lex_number(body: &str, i: usize) -> Result<(i64, usize), TemplateError> {
    let mut j = i;
    while matches!(body.get(j..).and_then(|r| r.chars().next()), Some(c) if c.is_ascii_digit()) {
        j = j.saturating_add(1);
    }
    if matches!(body.get(j..), Some(r) if r.starts_with('.'))
        && matches!(body.get(j.saturating_add(1)..).and_then(|r| r.chars().next()), Some(c) if c.is_ascii_digit())
    {
        return Err(TemplateError::Unsupported(
            "float literals are not supported".into(),
        ));
    }
    let digits = body.get(i..j).unwrap_or("");
    let n = digits
        .parse::<i64>()
        .map_err(|_| TemplateError::Syntax(format!("integer literal out of range: {digits}")))?;
    Ok((n, j))
}

/// Read a quoted literal, applying Python string-escape rules.
fn lex_string(body: &str, i: usize, quote: char) -> Result<(String, usize), TemplateError> {
    let mut out = String::new();
    let mut j = i.saturating_add(quote.len_utf8());
    loop {
        let Some(c) = body.get(j..).and_then(|r| r.chars().next()) else {
            return Err(TemplateError::Syntax("unterminated string literal".into()));
        };
        j = j.saturating_add(c.len_utf8());
        if c == quote {
            // Jinja normalises newlines inside literals to `\n`.
            return Ok((out.replace("\r\n", "\n").replace('\r', "\n"), j));
        }
        if c != '\\' {
            out.push(c);
            continue;
        }
        let Some(esc) = body.get(j..).and_then(|r| r.chars().next()) else {
            return Err(TemplateError::Syntax("unterminated string escape".into()));
        };
        j = j.saturating_add(esc.len_utf8());
        match esc {
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            'b' => out.push('\u{8}'),
            'f' => out.push('\u{c}'),
            'a' => out.push('\u{7}'),
            'v' => out.push('\u{b}'),
            '0' => out.push('\0'),
            '\\' => out.push('\\'),
            '\'' => out.push('\''),
            '"' => out.push('"'),
            'x' | 'u' | 'U' => {
                let width = match esc {
                    'x' => 2,
                    'u' => 4,
                    _ => 8,
                };
                let end = j.saturating_add(width);
                let hex = body.get(j..end).ok_or_else(|| {
                    TemplateError::Syntax(format!("truncated \\{esc} escape in string literal"))
                })?;
                let n = u32::from_str_radix(hex, 16).map_err(|_| {
                    TemplateError::Syntax(format!("bad \\{esc} escape {hex:?} in string literal"))
                })?;
                out.push(char::from_u32(n).ok_or_else(|| {
                    TemplateError::Syntax(format!("\\{esc}{hex} is not a code point"))
                })?);
                j = end;
            }
            // Python leaves unknown escapes alone, backslash included.
            other => {
                out.push('\\');
                out.push(other);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{lex, Piece, Tok};

    fn text(pieces: &[Piece]) -> Vec<String> {
        pieces
            .iter()
            .filter_map(|p| match p {
                Piece::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn plain_text_passes_through() {
        assert_eq!(lex("hello").unwrap(), vec![Piece::Text("hello".into())]);
        assert_eq!(lex("").unwrap(), vec![]);
    }

    #[test]
    fn expression_tags_tokenize() {
        let p = lex("{{ message['role'] }}").unwrap();
        assert_eq!(
            p,
            vec![Piece::Expr(vec![
                Tok::Name("message".into()),
                Tok::Punct("["),
                Tok::Str("role".into()),
                Tok::Punct("]"),
            ])]
        );
    }

    #[test]
    fn minus_strips_whitespace_on_both_sides() {
        assert_eq!(text(&lex("a\n   {{- 'x' }}b").unwrap()), vec!["a", "b"]);
        assert_eq!(text(&lex("a{{ 'x' -}}   \n b").unwrap()), vec!["a", "b"]);
        assert_eq!(
            text(&lex("a\n  {%- if x %}b{% endif %}").unwrap()),
            vec!["a", "b"]
        );
    }

    #[test]
    fn trim_blocks_eats_one_newline_after_a_block_tag() {
        // A block tag on its own line leaves nothing behind.
        assert_eq!(
            text(&lex("{% if x %}\nbody\n{% endif %}\n").unwrap()),
            vec!["body\n"]
        );
        // A variable tag does not get the same treatment.
        assert_eq!(text(&lex("{{ x }}\nbody").unwrap()), vec!["\nbody"]);
    }

    #[test]
    fn lstrip_blocks_only_eats_a_whole_leading_indent() {
        assert_eq!(
            text(&lex("a\n    {% if x %}b{% endif %}").unwrap()),
            vec!["a\n", "b"]
        );
        // Not at a line start: the spaces stay.
        assert_eq!(
            text(&lex("a  {% if x %}b{% endif %}").unwrap()),
            vec!["a  ", "b"]
        );
        // `{%+` opts out of lstrip.
        assert_eq!(
            text(&lex("a\n    {%+ if x %}b{% endif %}").unwrap()),
            vec!["a\n    ", "b"]
        );
    }

    #[test]
    fn comments_disappear_with_the_same_whitespace_rules() {
        assert_eq!(text(&lex("a\n{# note #}\nb").unwrap()), vec!["a\n", "b"]);
        assert_eq!(
            text(&lex("a\n  {#- note -#}  \nb").unwrap()),
            vec!["a", "b"]
        );
        assert!(lex("a{# unclosed").is_err());
    }

    #[test]
    fn string_literals_keep_close_delimiters_and_escapes() {
        let p = lex(r#"{{ '%} and }} inside' }}"#).unwrap();
        assert_eq!(
            p,
            vec![Piece::Expr(vec![Tok::Str("%} and }} inside".into())])]
        );
        let p = lex(r"{{ 'a\nb\t\\c\'d' }}").unwrap();
        assert_eq!(p, vec![Piece::Expr(vec![Tok::Str("a\nb\t\\c'd".into())])]);
        let p = lex(r"{{ '\u00e9\x41' }}").unwrap();
        assert_eq!(p, vec![Piece::Expr(vec![Tok::Str("éA".into())])]);
    }

    #[test]
    fn numbers_operators_and_names() {
        let p = lex("{% set i = loop.index0 % 2 == 0 %}").unwrap();
        assert_eq!(
            p,
            vec![Piece::Block(vec![
                Tok::Name("set".into()),
                Tok::Name("i".into()),
                Tok::Punct("="),
                Tok::Name("loop".into()),
                Tok::Punct("."),
                Tok::Name("index0".into()),
                Tok::Punct("%"),
                Tok::Int(2),
                Tok::Punct("=="),
                Tok::Int(0),
            ])]
        );
        assert!(lex("{{ 1.5 }}").is_err());
        assert!(lex("{{ a @ b }}").is_err());
        assert!(lex("{{ 'x'").is_err());
    }
}
