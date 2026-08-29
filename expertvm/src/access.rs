//! One router decision: which experts a token touched at a layer.

use crate::error::Error;
use std::collections::BTreeSet;

/// Identifies one expert at one layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpertKey {
    /// Transformer block index.
    pub layer: u32,
    /// Expert index within that block.
    pub expert: u32,
}

impl ExpertKey {
    /// Constructor.
    #[must_use]
    pub fn new(layer: u32, expert: u32) -> Self {
        Self { layer, expert }
    }
}

impl core::fmt::Display for ExpertKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "L{}E{}", self.layer, self.expert)
    }
}

/// One MoE routing event from a real decoder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertAccess {
    /// Sequence / request id.
    pub sequence: u64,
    /// Token index within the sequence (prefill tokens included).
    pub token: u32,
    /// Layer index.
    pub layer: u32,
    /// Selected expert ids (top-k).
    pub experts: Vec<u32>,
    /// Router mass per selected expert, permille (`0..=1000`). Empty if unknown.
    pub weight_pt: Vec<u32>,
}

impl ExpertAccess {
    /// Keys for the selected experts.
    #[must_use]
    pub fn keys(&self) -> Vec<ExpertKey> {
        self.experts
            .iter()
            .map(|e| ExpertKey::new(self.layer, *e))
            .collect()
    }

    /// JSONL line without a serde dependency. Omits `w` when [`Self::weight_pt`] is empty.
    #[must_use]
    pub fn to_jsonl(&self) -> String {
        let experts = format_u32_list(&self.experts);
        let w = if self.weight_pt.is_empty() {
            String::new()
        } else {
            format!(",\"w\":{}", format_u32_list(&self.weight_pt))
        };
        format!(
            "{{\"sequence\":{},\"token\":{},\"layer\":{},\"experts\":{experts}{w}}}",
            self.sequence, self.token, self.layer
        )
    }

    /// Parse one JSONL object of the shape `to_jsonl` emits.
    pub fn from_jsonl(line: &str) -> Result<Self, Error> {
        parse_access(line.trim())
    }
}

/// Ordered list of routing events.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Trace {
    /// Events in decode order.
    pub events: Vec<ExpertAccess>,
}

impl Trace {
    /// Empty trace.
    #[must_use]
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Parse many JSONL lines; blank lines and `#` comments skipped.
    pub fn parse(text: &str) -> Result<Self, Error> {
        let mut events = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            events.push(ExpertAccess::from_jsonl(line)?);
        }
        Ok(Self { events })
    }

    /// Serialize.
    #[must_use]
    pub fn to_jsonl(&self) -> String {
        let mut out = String::new();
        for e in &self.events {
            out.push_str(&e.to_jsonl());
            out.push('\n');
        }
        out
    }

    /// Number of expert acquires (sum of top-k).
    #[must_use]
    pub fn n_acquires(&self) -> u64 {
        let mut n = 0u64;
        for e in &self.events {
            n = n.saturating_add(u64::try_from(e.experts.len()).unwrap_or(0));
        }
        n
    }

    /// Flattened expert keys in decode order (one per routed expert).
    #[must_use]
    pub fn keys(&self) -> Vec<ExpertKey> {
        let mut out = Vec::new();
        for e in &self.events {
            out.extend(e.keys());
        }
        out
    }

    /// Distinct `sequence` ids (batch width).
    #[must_use]
    pub fn n_sequences(&self) -> u64 {
        let mut seen = BTreeSet::new();
        for e in &self.events {
            let _ins = seen.insert(e.sequence);
        }
        u64::try_from(seen.len()).unwrap_or(0)
    }
}

fn parse_access(line: &str) -> Result<ExpertAccess, Error> {
    if !line.starts_with('{') || !line.ends_with('}') {
        return Err(Error::Trace("expected {..}"));
    }
    Ok(ExpertAccess {
        sequence: field_u64(line, "sequence")?,
        token: field_u32(line, "token")?,
        layer: field_u32(line, "layer")?,
        experts: field_u32_list(line, "experts")?,
        weight_pt: field_u32_list_opt(line, "w")?,
    })
}

/// Floor of `r` as permille without an `f32 as u32` cast (clippy `cast_possible_truncation`).
#[must_use]
pub fn weight_permille(r: f32) -> u32 {
    let mut best = 0u32;
    for i in 0..=1000u32 {
        let numer = f32::from(u16::try_from(i).unwrap_or(1000));
        if r >= numer / 1000.0 {
            best = i;
        }
    }
    best
}

fn after_key<'a>(line: &'a str, key: &str) -> Result<&'a str, Error> {
    let needle = format!("\"{key}\":");
    match line.split(&needle).nth(1) {
        Some(rest) => Ok(rest.trim()),
        None => Err(Error::Trace("missing field")),
    }
}

fn field_u64(line: &str, key: &str) -> Result<u64, Error> {
    let rest = after_key(line, key)?;
    let tok = rest.split([',', '}']).next().unwrap_or("");
    tok.trim().parse().map_err(|_| Error::Trace("bad u64"))
}

fn field_u32(line: &str, key: &str) -> Result<u32, Error> {
    let rest = after_key(line, key)?;
    let tok = rest.split([',', '}']).next().unwrap_or("");
    tok.trim().parse().map_err(|_| Error::Trace("bad u32"))
}

fn field_u32_list(line: &str, key: &str) -> Result<Vec<u32>, Error> {
    parse_u32_list(after_key(line, key)?.trim())
}

fn field_u32_list_opt(line: &str, key: &str) -> Result<Vec<u32>, Error> {
    let needle = format!("\"{key}\":");
    match line.split(&needle).nth(1) {
        None => Ok(Vec::new()),
        Some(rest) => parse_u32_list(rest.trim()),
    }
}

fn parse_u32_list(rest: &str) -> Result<Vec<u32>, Error> {
    let start = rest.strip_prefix('[').ok_or(Error::Trace("expected []"))?;
    let inner = start.split(']').next().unwrap_or("");
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for p in inner.split(',') {
        out.push(p.trim().parse().map_err(|_| Error::Trace("bad u32"))?);
    }
    Ok(out)
}

fn format_u32_list(xs: &[u32]) -> String {
    let mut s = String::from("[");
    for (i, e) in xs.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&e.to_string());
    }
    s.push(']');
    s
}
