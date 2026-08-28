//! Runtime values for the chat-template renderer.
//!
//! Python semantics where chat templates depend on them: `1 == true`, empty
//! containers are falsy, `str()` of a bool is `True`/`False`, and dictionaries
//! keep insertion order so `tojson` output matches `json.dumps`.

use super::TemplateError;

/// A value flowing through a chat template.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Value {
    /// A name or key that was never bound. Renders as the empty string and is
    /// falsy, but is an error to index or take an attribute of, like Jinja's
    /// default `Undefined`.
    #[default]
    Undefined,
    /// Python `None`.
    None,
    /// `true` / `false`.
    Bool(bool),
    /// Integer. Chat templates never use floats; float literals are rejected.
    Int(i64),
    /// String.
    Str(String),
    /// List.
    List(Vec<Value>),
    /// Mapping, in insertion order.
    Map(Vec<(String, Value)>),
}

impl Value {
    /// Convenience constructor for string values.
    pub fn str(s: impl Into<String>) -> Self {
        Self::Str(s.into())
    }

    /// Python truthiness: empty string, empty container, zero and `None` are false.
    pub fn truthy(&self) -> bool {
        match self {
            Self::Undefined | Self::None => false,
            Self::Bool(b) => *b,
            Self::Int(i) => *i != 0,
            Self::Str(s) => !s.is_empty(),
            Self::List(v) => !v.is_empty(),
            Self::Map(m) => !m.is_empty(),
        }
    }

    /// Name of the type, for error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Undefined => "undefined",
            Self::None => "none",
            Self::Bool(_) => "bool",
            Self::Int(_) => "int",
            Self::Str(_) => "string",
            Self::List(_) => "list",
            Self::Map(_) => "mapping",
        }
    }

    /// `str(value)` as Jinja renders it into the output stream.
    pub fn render(&self) -> String {
        match self {
            Self::Undefined => String::new(),
            Self::None => "None".to_string(),
            Self::Bool(true) => "True".to_string(),
            Self::Bool(false) => "False".to_string(),
            Self::Int(i) => i.to_string(),
            Self::Str(s) => s.clone(),
            Self::List(_) | Self::Map(_) => self.repr(),
        }
    }

    /// Python `repr`, used when a container is interpolated into text.
    fn repr(&self) -> String {
        match self {
            Self::Str(s) => py_repr_str(s),
            Self::List(items) => {
                let inner: Vec<String> = items.iter().map(Self::repr).collect();
                format!("[{}]", inner.join(", "))
            }
            Self::Map(entries) => {
                let inner: Vec<String> = entries
                    .iter()
                    .map(|(k, v)| format!("{}: {}", py_repr_str(k), v.repr()))
                    .collect();
                format!("{{{}}}", inner.join(", "))
            }
            other => other.render(),
        }
    }

    /// Look up a mapping key. Missing keys are [`Value::Undefined`].
    pub fn get_key(&self, key: &str) -> Option<Self> {
        match self {
            Self::Map(entries) => Some(
                entries
                    .iter()
                    .find(|(k, _)| k == key)
                    .map_or(Self::Undefined, |(_, v)| v.clone()),
            ),
            _ => None,
        }
    }

    /// Iterate a list, a string (by character), or a mapping (by key).
    pub fn iterate(&self) -> Result<Vec<Self>, TemplateError> {
        match self {
            Self::List(items) => Ok(items.clone()),
            Self::Str(s) => Ok(s.chars().map(|c| Self::Str(c.to_string())).collect()),
            Self::Map(entries) => Ok(entries.iter().map(|(k, _)| Self::str(k)).collect()),
            other => Err(TemplateError::Runtime(format!(
                "{} is not iterable",
                other.type_name()
            ))),
        }
    }

    /// `json.dumps(value, ensure_ascii=False)`, optionally pretty-printed.
    pub fn to_json(&self, indent: Option<usize>) -> Result<String, TemplateError> {
        let mut out = String::new();
        self.write_json(&mut out, indent, 0)?;
        Ok(out)
    }

    fn write_json(
        &self,
        out: &mut String,
        indent: Option<usize>,
        depth: usize,
    ) -> Result<(), TemplateError> {
        match self {
            Self::Undefined => {
                return Err(TemplateError::Runtime(
                    "cannot serialise an undefined value to json".into(),
                ))
            }
            Self::None => out.push_str("null"),
            Self::Bool(true) => out.push_str("true"),
            Self::Bool(false) => out.push_str("false"),
            Self::Int(i) => out.push_str(&i.to_string()),
            Self::Str(s) => json_string(out, s),
            Self::List(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return Ok(());
                }
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                        if indent.is_none() {
                            out.push(' ');
                        }
                    }
                    json_newline_indent(out, indent, depth.saturating_add(1));
                    item.write_json(out, indent, depth.saturating_add(1))?;
                }
                json_newline_indent(out, indent, depth);
                out.push(']');
            }
            Self::Map(entries) => {
                if entries.is_empty() {
                    out.push_str("{}");
                    return Ok(());
                }
                out.push('{');
                for (i, (k, v)) in entries.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                        if indent.is_none() {
                            out.push(' ');
                        }
                    }
                    json_newline_indent(out, indent, depth.saturating_add(1));
                    json_string(out, k);
                    out.push_str(": ");
                    v.write_json(out, indent, depth.saturating_add(1))?;
                }
                json_newline_indent(out, indent, depth);
                out.push('}');
            }
        }
        Ok(())
    }
}

fn json_newline_indent(out: &mut String, indent: Option<usize>, depth: usize) {
    let Some(width) = indent else { return };
    out.push('\n');
    for _ in 0..width.saturating_mul(depth) {
        out.push(' ');
    }
}

/// `json.dumps` string escaping with `ensure_ascii=False`.
fn json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if u32::from(c) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", u32::from(c)));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python `repr` of a string: single quotes unless the string contains one.
fn py_repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::new();
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python `==`: `True == 1`, and an undefined only equals another undefined.
pub(crate) fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Undefined, Value::Undefined) | (Value::None, Value::None) => true,
        (Value::Undefined, _) | (_, Value::Undefined) => false,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Bool(x), Value::Int(y)) | (Value::Int(y), Value::Bool(x)) => i64::from(*x) == *y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::List(x), Value::List(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| values_equal(p, q))
        }
        (Value::Map(x), Value::Map(y)) => {
            x.len() == y.len()
                && x.iter().all(|(k, v)| {
                    y.iter()
                        .find(|(k2, _)| k2 == k)
                        .is_some_and(|(_, v2)| values_equal(v, v2))
                })
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{values_equal, Value};

    #[test]
    fn truthiness_follows_python() {
        assert!(!Value::Undefined.truthy());
        assert!(!Value::None.truthy());
        assert!(!Value::Bool(false).truthy());
        assert!(Value::Bool(true).truthy());
        assert!(!Value::Int(0).truthy());
        assert!(Value::Int(-1).truthy());
        assert!(!Value::str("").truthy());
        assert!(Value::str("x").truthy());
        assert!(!Value::List(vec![]).truthy());
        assert!(Value::List(vec![Value::None]).truthy());
        assert!(!Value::Map(vec![]).truthy());
    }

    #[test]
    fn render_matches_python_str() {
        assert_eq!(Value::Undefined.render(), "");
        assert_eq!(Value::None.render(), "None");
        assert_eq!(Value::Bool(true).render(), "True");
        assert_eq!(Value::Bool(false).render(), "False");
        assert_eq!(Value::Int(-7).render(), "-7");
        assert_eq!(Value::str("hi").render(), "hi");
        assert_eq!(
            Value::List(vec![Value::str("a"), Value::Int(1)]).render(),
            "['a', 1]"
        );
        assert_eq!(
            Value::Map(vec![("k".into(), Value::str("v"))]).render(),
            "{'k': 'v'}"
        );
    }

    #[test]
    fn equality_treats_true_as_one_and_undefined_as_its_own() {
        assert!(values_equal(&Value::Bool(true), &Value::Int(1)));
        assert!(values_equal(&Value::Int(0), &Value::Bool(false)));
        assert!(!values_equal(&Value::str("1"), &Value::Int(1)));
        assert!(values_equal(&Value::Undefined, &Value::Undefined));
        assert!(!values_equal(&Value::Undefined, &Value::None));
        assert!(!values_equal(&Value::Undefined, &Value::str("system")));
        assert!(values_equal(
            &Value::List(vec![Value::Int(1)]),
            &Value::List(vec![Value::Int(1)])
        ));
    }

    #[test]
    fn json_matches_python_json_dumps() {
        let v = Value::Map(vec![
            ("name".into(), Value::str("get_weather")),
            (
                "args".into(),
                Value::List(vec![Value::Int(1), Value::Bool(true), Value::None]),
            ),
        ]);
        assert_eq!(
            v.to_json(None).unwrap(),
            r#"{"name": "get_weather", "args": [1, true, null]}"#
        );
        assert_eq!(
            v.to_json(Some(2)).unwrap(),
            "{\n  \"name\": \"get_weather\",\n  \"args\": [\n    1,\n    true,\n    null\n  ]\n}"
        );
        // ensure_ascii=False: non-ASCII is not escaped.
        assert_eq!(
            Value::str("héllo 漢").to_json(None).unwrap(),
            "\"héllo 漢\""
        );
        assert_eq!(Value::str("a\"b\n").to_json(None).unwrap(), "\"a\\\"b\\n\"");
        assert_eq!(Value::List(vec![]).to_json(Some(4)).unwrap(), "[]");
        assert!(Value::Undefined.to_json(None).is_err());
    }

    #[test]
    fn map_lookup_returns_undefined_for_missing_keys() {
        let m = Value::Map(vec![("role".into(), Value::str("user"))]);
        assert_eq!(m.get_key("role"), Some(Value::str("user")));
        assert_eq!(m.get_key("content"), Some(Value::Undefined));
        assert_eq!(Value::str("x").get_key("role"), None);
    }
}
