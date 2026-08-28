//! Filters, tests, methods and global functions.
//!
//! The set is deliberately closed: anything a template asks for that is not
//! here is a named error, never a silent empty string. `strftime_now` is
//! *intentionally* absent so templates that guard it with `is defined` (the
//! Llama-3 family) take their deterministic fallback date instead of embedding
//! the machine clock in a prompt.

use super::value::{values_equal, Value};
use super::TemplateError;

/// Jinja's `soft_str`: an undefined value stringifies to the empty string even
/// though most other operations on it raise.
fn soft_str(v: &Value) -> String {
    v.render()
}

fn arg(args: &[Value], i: usize) -> Option<&Value> {
    args.get(i)
}

fn kwarg<'a>(kwargs: &'a [(String, Value)], name: &str) -> Option<&'a Value> {
    kwargs.iter().find(|(k, _)| k == name).map(|(_, v)| v)
}

fn want_int(v: &Value, what: &str) -> Result<i64, TemplateError> {
    match v {
        Value::Int(i) => Ok(*i),
        Value::Bool(b) => Ok(i64::from(*b)),
        other => Err(TemplateError::Runtime(format!(
            "{what} must be an integer, got {}",
            other.type_name()
        ))),
    }
}

fn len_of(v: &Value) -> Result<usize, TemplateError> {
    match v {
        Value::Str(s) => Ok(s.chars().count()),
        Value::List(items) => Ok(items.len()),
        Value::Map(entries) => Ok(entries.len()),
        other => Err(TemplateError::Runtime(format!(
            "{} has no length",
            other.type_name()
        ))),
    }
}

/// Apply `value | name(args, kwargs)`.
pub(crate) fn filter(
    name: &str,
    value: &Value,
    args: &[Value],
    kwargs: &[(String, Value)],
) -> Result<Value, TemplateError> {
    match name {
        "trim" => Ok(Value::Str(trim_with(value, arg(args, 0)))),
        "lower" => Ok(Value::Str(soft_str(value).to_lowercase())),
        "upper" => Ok(Value::Str(soft_str(value).to_uppercase())),
        "capitalize" => {
            let s = soft_str(value);
            let mut chars = s.chars();
            Ok(Value::Str(match chars.next() {
                Some(c) => c.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
                None => String::new(),
            }))
        }
        "title" => Ok(Value::Str(title_case(&soft_str(value)))),
        "string" => Ok(Value::Str(soft_str(value))),
        "safe" | "forceescape" => Ok(value.clone()),
        "length" | "count" => Ok(Value::Int(
            i64::try_from(len_of(value)?).unwrap_or(i64::MAX),
        )),
        "first" => Ok(value
            .iterate()?
            .first()
            .cloned()
            .unwrap_or(Value::Undefined)),
        "last" => Ok(value.iterate()?.last().cloned().unwrap_or(Value::Undefined)),
        "list" => Ok(Value::List(value.iterate()?)),
        "reverse" => {
            let mut items = value.iterate()?;
            items.reverse();
            Ok(Value::List(items))
        }
        "sort" => {
            let mut items: Vec<String> = value.iterate()?.iter().map(soft_str).collect();
            items.sort();
            Ok(Value::List(items.into_iter().map(Value::Str).collect()))
        }
        "join" => {
            let sep = arg(args, 0)
                .or_else(|| kwarg(kwargs, "d"))
                .map_or(String::new(), soft_str);
            let parts: Vec<String> = value.iterate()?.iter().map(soft_str).collect();
            Ok(Value::Str(parts.join(&sep)))
        }
        "default" | "d" => {
            let fallback = arg(args, 0).cloned().unwrap_or(Value::str(""));
            let boolean = arg(args, 1)
                .or_else(|| kwarg(kwargs, "boolean"))
                .is_some_and(Value::truthy);
            let missing = matches!(value, Value::Undefined) || (boolean && !value.truthy());
            Ok(if missing { fallback } else { value.clone() })
        }
        "tojson" => {
            let indent = arg(args, 0)
                .or_else(|| kwarg(kwargs, "indent"))
                .map(|v| want_int(v, "tojson indent"))
                .transpose()?
                .map(|n| usize::try_from(n).unwrap_or(0));
            Ok(Value::Str(value.to_json(indent)?))
        }
        "int" => {
            let fallback = arg(args, 0)
                .or_else(|| kwarg(kwargs, "default"))
                .cloned()
                .unwrap_or(Value::Int(0));
            Ok(match value {
                Value::Int(_) => value.clone(),
                Value::Bool(b) => Value::Int(i64::from(*b)),
                Value::Str(s) => s.trim().parse::<i64>().map_or(fallback, Value::Int),
                _ => fallback,
            })
        }
        "abs" => Ok(Value::Int(want_int(value, "abs")?.saturating_abs())),
        "replace" => {
            let (Some(from), Some(to)) = (arg(args, 0), arg(args, 1)) else {
                return Err(TemplateError::Runtime("replace needs two arguments".into()));
            };
            Ok(Value::Str(
                soft_str(value).replace(&soft_str(from), &soft_str(to)),
            ))
        }
        other => Err(TemplateError::Unsupported(format!(
            "filter `{other}` is not supported"
        ))),
    }
}

/// Python `str.strip`, optionally over an explicit character set.
fn trim_with(value: &Value, chars: Option<&Value>) -> String {
    let s = soft_str(value);
    match chars {
        Some(c) => {
            let set: Vec<char> = soft_str(c).chars().collect();
            s.trim_matches(|ch| set.contains(&ch)).to_string()
        }
        None => s.trim().to_string(),
    }
}

fn title_case(s: &str) -> String {
    let mut out = String::new();
    let mut start_of_word = true;
    for c in s.chars() {
        if start_of_word {
            out.extend(c.to_uppercase());
        } else {
            out.extend(c.to_lowercase());
        }
        start_of_word = !c.is_alphanumeric();
    }
    out
}

/// Apply `value is name(args)`.
pub(crate) fn test(name: &str, value: &Value, args: &[Value]) -> Result<bool, TemplateError> {
    match name {
        "defined" => Ok(!matches!(value, Value::Undefined)),
        "undefined" => Ok(matches!(value, Value::Undefined)),
        "none" => Ok(matches!(value, Value::None)),
        "string" => Ok(matches!(value, Value::Str(_))),
        "number" | "integer" => Ok(matches!(value, Value::Int(_))),
        "boolean" => Ok(matches!(value, Value::Bool(_))),
        "mapping" => Ok(matches!(value, Value::Map(_))),
        "sequence" => Ok(matches!(
            value,
            Value::List(_) | Value::Str(_) | Value::Map(_)
        )),
        // Python calls a string iterable; templates that branch on
        // `content is iterable` after `content is string` rely on that order.
        "iterable" => Ok(matches!(
            value,
            Value::List(_) | Value::Str(_) | Value::Map(_)
        )),
        "true" => Ok(matches!(value, Value::Bool(true))),
        "false" => Ok(matches!(value, Value::Bool(false))),
        "callable" => Ok(false),
        "equalto" | "eq" => {
            let Some(other) = arg(args, 0) else {
                return Err(TemplateError::Runtime("equalto needs an argument".into()));
            };
            Ok(values_equal(value, other))
        }
        "in" => {
            let Some(other) = arg(args, 0) else {
                return Err(TemplateError::Runtime("in needs an argument".into()));
            };
            Ok(match other {
                Value::List(items) => items.iter().any(|i| values_equal(i, value)),
                Value::Map(entries) => {
                    matches!(value, Value::Str(k) if entries.iter().any(|(key, _)| key == k))
                }
                Value::Str(h) => matches!(value, Value::Str(n) if h.contains(n.as_str())),
                _ => false,
            })
        }
        "odd" => Ok(want_int(value, "odd")? % 2 != 0),
        "even" => Ok(want_int(value, "even")? % 2 == 0),
        "divisibleby" => {
            let Some(other) = arg(args, 0) else {
                return Err(TemplateError::Runtime(
                    "divisibleby needs an argument".into(),
                ));
            };
            let d = want_int(other, "divisibleby")?;
            if d == 0 {
                return Err(TemplateError::Runtime("divisibleby zero".into()));
            }
            Ok(want_int(value, "divisibleby")? % d == 0)
        }
        other => Err(TemplateError::Unsupported(format!(
            "test `{other}` is not supported"
        ))),
    }
}

/// Apply `receiver.name(args)`.
pub(crate) fn method(name: &str, receiver: &Value, args: &[Value]) -> Result<Value, TemplateError> {
    match (receiver, name) {
        (Value::Map(entries), "items") => Ok(Value::List(
            entries
                .iter()
                .map(|(k, v)| Value::List(vec![Value::str(k), v.clone()]))
                .collect(),
        )),
        (Value::Map(entries), "keys") => Ok(Value::List(
            entries.iter().map(|(k, _)| Value::str(k)).collect(),
        )),
        (Value::Map(entries), "values") => Ok(Value::List(
            entries.iter().map(|(_, v)| v.clone()).collect(),
        )),
        (Value::Map(_), "get") => {
            let Some(Value::Str(key)) = arg(args, 0) else {
                return Err(TemplateError::Runtime("get needs a string key".into()));
            };
            let fallback = arg(args, 1).cloned().unwrap_or(Value::None);
            Ok(match receiver.get_key(key) {
                Some(Value::Undefined) | None => fallback,
                Some(v) => v,
            })
        }
        (_, "strip") => Ok(Value::Str(trim_with(receiver, arg(args, 0)))),
        (_, "lstrip") => Ok(Value::Str(soft_str(receiver).trim_start().to_string())),
        (_, "rstrip") => Ok(Value::Str(soft_str(receiver).trim_end().to_string())),
        (_, "upper") => Ok(Value::Str(soft_str(receiver).to_uppercase())),
        (_, "lower") => Ok(Value::Str(soft_str(receiver).to_lowercase())),
        (_, "title") => Ok(Value::Str(title_case(&soft_str(receiver)))),
        (_, "startswith") => Ok(Value::Bool(
            soft_str(receiver).starts_with(&soft_str(arg(args, 0).unwrap_or(&Value::str("")))),
        )),
        (_, "endswith") => Ok(Value::Bool(
            soft_str(receiver).ends_with(&soft_str(arg(args, 0).unwrap_or(&Value::str("")))),
        )),
        (_, "replace") => {
            let (Some(from), Some(to)) = (arg(args, 0), arg(args, 1)) else {
                return Err(TemplateError::Runtime("replace needs two arguments".into()));
            };
            Ok(Value::Str(
                soft_str(receiver).replace(&soft_str(from), &soft_str(to)),
            ))
        }
        (Value::Str(s), "split") => {
            let parts: Vec<Value> = match arg(args, 0) {
                Some(sep) => s.split(soft_str(sep).as_str()).map(Value::str).collect(),
                None => s.split_whitespace().map(Value::str).collect(),
            };
            Ok(Value::List(parts))
        }
        (Value::Str(s), "splitlines") => Ok(Value::List(s.lines().map(Value::str).collect())),
        (Value::Str(sep), "join") => {
            let Some(items) = arg(args, 0) else {
                return Err(TemplateError::Runtime("join needs a sequence".into()));
            };
            let parts: Vec<String> = items.iterate()?.iter().map(soft_str).collect();
            Ok(Value::Str(parts.join(sep)))
        }
        (recv, other) => Err(TemplateError::Unsupported(format!(
            "method `{other}` on {} is not supported",
            recv.type_name()
        ))),
    }
}

/// Call a global function.
pub(crate) fn function(
    name: &str,
    args: &[Value],
    kwargs: &[(String, Value)],
) -> Result<Value, TemplateError> {
    match name {
        "raise_exception" => Err(TemplateError::Raised(
            arg(args, 0).map_or_else(String::new, soft_str),
        )),
        "namespace" | "dict" => Ok(Value::Map(kwargs.to_vec())),
        "range" => {
            let (start, stop, step) = match args {
                [stop] => (0, want_int(stop, "range")?, 1),
                [start, stop] => (want_int(start, "range")?, want_int(stop, "range")?, 1),
                [start, stop, step] => (
                    want_int(start, "range")?,
                    want_int(stop, "range")?,
                    want_int(step, "range")?,
                ),
                _ => {
                    return Err(TemplateError::Runtime(
                        "range takes 1 to 3 arguments".into(),
                    ))
                }
            };
            if step == 0 {
                return Err(TemplateError::Runtime("range step must not be zero".into()));
            }
            let mut out = Vec::new();
            let mut i = start;
            while (step > 0 && i < stop) || (step < 0 && i > stop) {
                out.push(Value::Int(i));
                i = i.saturating_add(step);
                if out.len() > 100_000 {
                    return Err(TemplateError::Runtime("range is too long".into()));
                }
            }
            Ok(Value::List(out))
        }
        other => Err(TemplateError::Unsupported(format!(
            "function `{other}` is not supported"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::{filter, function, method, test};
    use crate::template::value::Value;

    fn f(name: &str, v: Value, args: &[Value]) -> Value {
        filter(name, &v, args, &[]).expect("filter")
    }

    #[test]
    fn string_filters_treat_undefined_as_empty_like_soft_str() {
        assert_eq!(f("trim", Value::str("  hi \n"), &[]), Value::str("hi"));
        assert_eq!(f("trim", Value::Undefined, &[]), Value::str(""));
        assert_eq!(
            f("trim", Value::str("xxhixx"), &[Value::str("x")]),
            Value::str("hi")
        );
        assert_eq!(f("upper", Value::str("aB"), &[]), Value::str("AB"));
        assert_eq!(f("lower", Value::str("aB"), &[]), Value::str("ab"));
        assert_eq!(
            f(
                "replace",
                Value::str("a-b"),
                &[Value::str("-"), Value::str("_")]
            ),
            Value::str("a_b")
        );
    }

    #[test]
    fn default_only_substitutes_for_undefined_unless_boolean() {
        assert_eq!(
            f("default", Value::Undefined, &[Value::str("z")]),
            Value::str("z")
        );
        assert_eq!(
            f("default", Value::str(""), &[Value::str("z")]),
            Value::str("")
        );
        assert_eq!(
            f(
                "default",
                Value::str(""),
                &[Value::str("z"), Value::Bool(true)]
            ),
            Value::str("z")
        );
    }

    #[test]
    fn length_first_last_join_and_int() {
        let xs = Value::List(vec![Value::str("a"), Value::str("b")]);
        assert_eq!(f("length", xs.clone(), &[]), Value::Int(2));
        assert_eq!(f("first", xs.clone(), &[]), Value::str("a"));
        assert_eq!(f("last", xs.clone(), &[]), Value::str("b"));
        assert_eq!(f("join", xs, &[Value::str(", ")]), Value::str("a, b"));
        assert_eq!(f("length", Value::str("héllo"), &[]), Value::Int(5));
        assert_eq!(f("int", Value::str("42"), &[]), Value::Int(42));
        assert_eq!(f("int", Value::str("x"), &[]), Value::Int(0));
        assert!(filter("length", &Value::Undefined, &[], &[]).is_err());
    }

    #[test]
    fn tests_cover_what_real_templates_ask() {
        assert!(test("defined", &Value::str("x"), &[]).unwrap());
        assert!(!test("defined", &Value::Undefined, &[]).unwrap());
        assert!(test("none", &Value::None, &[]).unwrap());
        assert!(test("string", &Value::str("x"), &[]).unwrap());
        assert!(test("mapping", &Value::Map(vec![]), &[]).unwrap());
        assert!(test("iterable", &Value::str("x"), &[]).unwrap());
        assert!(!test("iterable", &Value::Int(1), &[]).unwrap());
        assert!(test("even", &Value::Int(4), &[]).unwrap());
    }

    #[test]
    fn methods_cover_items_get_split_and_startswith() {
        let m = Value::Map(vec![("a".into(), Value::Int(1))]);
        assert_eq!(
            method("items", &m, &[]).unwrap(),
            Value::List(vec![Value::List(vec![Value::str("a"), Value::Int(1)])])
        );
        assert_eq!(
            method("get", &m, &[Value::str("a")]).unwrap(),
            Value::Int(1)
        );
        assert_eq!(
            method("get", &m, &[Value::str("z"), Value::str("d")]).unwrap(),
            Value::str("d")
        );
        assert_eq!(
            method("split", &Value::str("a,b"), &[Value::str(",")]).unwrap(),
            Value::List(vec![Value::str("a"), Value::str("b")])
        );
        assert_eq!(
            method("startswith", &Value::str("abc"), &[Value::str("ab")]).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn globals_and_the_closed_set_of_names() {
        assert_eq!(
            function("range", &[Value::Int(3)], &[]).unwrap(),
            Value::List(vec![Value::Int(0), Value::Int(1), Value::Int(2)])
        );
        assert_eq!(
            function("namespace", &[], &[("v".into(), Value::Int(1))]).unwrap(),
            Value::Map(vec![("v".into(), Value::Int(1))])
        );
        let err = function("raise_exception", &[Value::str("bad roles")], &[])
            .expect_err("raised")
            .to_string();
        assert!(err.contains("bad roles"), "{err}");
        // `strftime_now` is intentionally not a function: templates guard it
        // with `is defined` and fall back to a fixed date.
        for (name, want) in [("strftime_now", "strftime_now"), ("lipsum", "lipsum")] {
            let err = function(name, &[], &[]).expect_err(name).to_string();
            assert!(err.contains(want) && err.contains("not supported"), "{err}");
        }
        let err = filter("selectattr", &Value::List(vec![]), &[], &[])
            .expect_err("filter")
            .to_string();
        assert!(
            err.contains("selectattr") && err.contains("not supported"),
            "{err}"
        );
        let err = test("sameas", &Value::None, &[])
            .expect_err("test")
            .to_string();
        assert!(err.contains("sameas"), "{err}");
    }
}
