//! AST + variables to rendered text.
//!
//! Undefined values follow Jinja's default `Undefined`: they render as the
//! empty string and are falsy, but taking an attribute, indexing, iterating,
//! or adding raises. Getting that wrong in the lenient direction is how a
//! template quietly renders a prompt with a hole in it.

use std::collections::HashMap;

use super::parse::{BinOp, Expr, Node, Target};
use super::value::{values_equal, Value};
use super::TemplateError;

/// Variable scopes, innermost last.
pub(crate) struct Env {
    scopes: Vec<HashMap<String, Value>>,
}

/// Whether a `{% break %}` or `{% continue %}` is unwinding.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Flow {
    Normal,
    Break,
    Continue,
}

impl Env {
    /// New environment with one global scope.
    pub(crate) fn new(globals: Vec<(String, Value)>) -> Self {
        Self {
            scopes: vec![globals.into_iter().collect()],
        }
    }

    fn lookup(&self, name: &str) -> Value {
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(name) {
                return v.clone();
            }
        }
        Value::Undefined
    }

    /// Bind in the innermost scope, like Jinja's per-frame `{% set %}`.
    fn bind(&mut self, name: &str, value: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            let _prev = scope.insert(name.to_string(), value);
        }
    }

    /// Mutate an existing binding wherever it lives, for `{% set ns.x = ... %}`.
    fn bind_attr(
        &mut self,
        name: &str,
        path: &[String],
        value: Value,
    ) -> Result<(), TemplateError> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(slot) = scope.get_mut(name) {
                return set_path(slot, path, value);
            }
        }
        Err(TemplateError::Runtime(format!(
            "{name} is undefined, cannot assign to one of its attributes"
        )))
    }
}

fn set_path(slot: &mut Value, path: &[String], value: Value) -> Result<(), TemplateError> {
    let Some((head, rest)) = path.split_first() else {
        *slot = value;
        return Ok(());
    };
    let Value::Map(entries) = slot else {
        return Err(TemplateError::Runtime(format!(
            "cannot assign attribute {head} on {}",
            slot.type_name()
        )));
    };
    if let Some(existing) = entries.iter_mut().find(|(k, _)| k == head) {
        return set_path(&mut existing.1, rest, value);
    }
    let mut fresh = Value::Map(Vec::new());
    set_path(&mut fresh, rest, value)?;
    entries.push((head.clone(), fresh));
    Ok(())
}

/// Render `body` and append to `out`.
pub(crate) fn render(body: &[Node], env: &mut Env) -> Result<String, TemplateError> {
    let mut out = String::new();
    let flow = exec(body, env, &mut out)?;
    if flow != Flow::Normal {
        return Err(TemplateError::Syntax(
            "{% break %} or {% continue %} outside a loop".into(),
        ));
    }
    Ok(out)
}

fn exec(body: &[Node], env: &mut Env, out: &mut String) -> Result<Flow, TemplateError> {
    for node in body {
        match node {
            Node::Text(t) => out.push_str(t),
            Node::Output(e) => out.push_str(&eval(e, env)?.render()),
            Node::Set { target, value } => {
                let v = eval(value, env)?;
                match target {
                    Target::Name(n) => env.bind(n, v),
                    Target::Attr(n, path) => env.bind_attr(n, path, v)?,
                }
            }
            Node::If {
                branches,
                otherwise,
            } => {
                let mut taken = false;
                for (cond, block) in branches {
                    if eval(cond, env)?.truthy() {
                        let flow = exec(block, env, out)?;
                        if flow != Flow::Normal {
                            return Ok(flow);
                        }
                        taken = true;
                        break;
                    }
                }
                if !taken {
                    let flow = exec(otherwise, env, out)?;
                    if flow != Flow::Normal {
                        return Ok(flow);
                    }
                }
            }
            Node::For {
                targets,
                iter,
                filter,
                body,
                otherwise,
            } => {
                let flow = exec_for(targets, iter, filter.as_ref(), body, otherwise, env, out)?;
                if flow != Flow::Normal {
                    return Ok(flow);
                }
            }
            Node::Break => return Ok(Flow::Break),
            Node::Continue => return Ok(Flow::Continue),
        }
    }
    Ok(Flow::Normal)
}

fn exec_for(
    targets: &[String],
    iter: &Expr,
    filter: Option<&Expr>,
    body: &[Node],
    otherwise: &[Node],
    env: &mut Env,
    out: &mut String,
) -> Result<Flow, TemplateError> {
    let seq = eval(iter, env)?.iterate()?;
    let mut items = Vec::new();
    for item in seq {
        let Some(pred) = filter else {
            items.push(item);
            continue;
        };
        env.scopes.push(HashMap::new());
        bind_targets(targets, &item, env)?;
        let keep = eval(pred, env)?.truthy();
        let _frame = env.scopes.pop();
        if keep {
            items.push(item);
        }
    }
    if items.is_empty() {
        return exec(otherwise, env, out);
    }
    let total = items.len();
    for (i, item) in items.iter().enumerate() {
        env.scopes.push(HashMap::new());
        bind_targets(targets, item, env)?;
        env.bind("loop", loop_value(i, total));
        let flow = exec(body, env, out);
        let _frame = env.scopes.pop();
        match flow? {
            Flow::Break => break,
            Flow::Normal | Flow::Continue => {}
        }
    }
    Ok(Flow::Normal)
}

fn bind_targets(targets: &[String], item: &Value, env: &mut Env) -> Result<(), TemplateError> {
    if let [only] = targets {
        env.bind(only, item.clone());
        return Ok(());
    }
    let parts = item.iterate()?;
    if parts.len() != targets.len() {
        return Err(TemplateError::Runtime(format!(
            "cannot unpack {} values into {} loop variables",
            parts.len(),
            targets.len()
        )));
    }
    for (name, value) in targets.iter().zip(parts) {
        env.bind(name, value);
    }
    Ok(())
}

fn loop_value(i: usize, total: usize) -> Value {
    let idx = i64::try_from(i).unwrap_or(i64::MAX);
    let len = i64::try_from(total).unwrap_or(i64::MAX);
    Value::Map(vec![
        ("index".into(), Value::Int(idx.saturating_add(1))),
        ("index0".into(), Value::Int(idx)),
        ("revindex".into(), Value::Int(len.saturating_sub(idx))),
        (
            "revindex0".into(),
            Value::Int(len.saturating_sub(idx).saturating_sub(1)),
        ),
        ("first".into(), Value::Bool(i == 0)),
        ("last".into(), Value::Bool(i.saturating_add(1) == total)),
        ("length".into(), Value::Int(len)),
    ])
}

fn eval(expr: &Expr, env: &mut Env) -> Result<Value, TemplateError> {
    match expr {
        Expr::Const(v) => Ok(v.clone()),
        Expr::Name(n) => Ok(env.lookup(n)),
        Expr::Attr(base, field) => {
            let b = eval(base, env)?;
            member(&b, field)
        }
        Expr::Item(base, index) => {
            let b = eval(base, env)?;
            let i = eval(index, env)?;
            subscript(&b, &i)
        }
        Expr::Slice(base, start, stop) => {
            let b = eval(base, env)?;
            let s = start.as_deref().map(|e| eval(e, env)).transpose()?;
            let e = stop.as_deref().map(|e| eval(e, env)).transpose()?;
            slice(&b, s.as_ref(), e.as_ref())
        }
        Expr::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(eval(item, env)?);
            }
            Ok(Value::List(out))
        }
        Expr::Dict(entries) => {
            let mut out = Vec::with_capacity(entries.len());
            for (k, v) in entries {
                let key = eval(k, env)?;
                let Value::Str(key) = key else {
                    return Err(TemplateError::Runtime(format!(
                        "dict keys must be strings, got {}",
                        key.type_name()
                    )));
                };
                out.push((key, eval(v, env)?));
            }
            Ok(Value::Map(out))
        }
        Expr::Neg(inner) => match eval(inner, env)? {
            Value::Int(i) => Ok(Value::Int(i.saturating_neg())),
            other => Err(TemplateError::Runtime(format!(
                "cannot negate {}",
                other.type_name()
            ))),
        },
        Expr::Not(inner) => Ok(Value::Bool(!eval(inner, env)?.truthy())),
        Expr::And(a, b) => {
            let lhs = eval(a, env)?;
            if lhs.truthy() {
                eval(b, env)
            } else {
                Ok(lhs)
            }
        }
        Expr::Or(a, b) => {
            let lhs = eval(a, env)?;
            if lhs.truthy() {
                Ok(lhs)
            } else {
                eval(b, env)
            }
        }
        Expr::Bin(op, a, b) => {
            let lhs = eval(a, env)?;
            let rhs = eval(b, env)?;
            binary(*op, &lhs, &rhs)
        }
        Expr::Cond { body, test, other } => {
            if eval(test, env)?.truthy() {
                eval(body, env)
            } else {
                match other {
                    Some(e) => eval(e, env),
                    None => Ok(Value::Undefined),
                }
            }
        }
        Expr::Filter {
            value,
            name,
            args,
            kwargs,
        } => {
            let v = eval(value, env)?;
            let args = eval_all(args, env)?;
            let kwargs = eval_kwargs(kwargs, env)?;
            super::builtins::filter(name, &v, &args, &kwargs)
        }
        Expr::Test {
            value,
            name,
            negated,
            args,
        } => {
            let v = eval(value, env)?;
            let args = eval_all(args, env)?;
            let got = super::builtins::test(name, &v, &args)?;
            Ok(Value::Bool(got != *negated))
        }
        Expr::Call { func, args, kwargs } => call(func, args, kwargs, env),
    }
}

fn eval_all(exprs: &[Expr], env: &mut Env) -> Result<Vec<Value>, TemplateError> {
    let mut out = Vec::with_capacity(exprs.len());
    for e in exprs {
        out.push(eval(e, env)?);
    }
    Ok(out)
}

fn eval_kwargs(
    kwargs: &[(String, Expr)],
    env: &mut Env,
) -> Result<Vec<(String, Value)>, TemplateError> {
    let mut out = Vec::with_capacity(kwargs.len());
    for (k, e) in kwargs {
        out.push((k.clone(), eval(e, env)?));
    }
    Ok(out)
}

fn call(
    func: &Expr,
    args: &[Expr],
    kwargs: &[(String, Expr)],
    env: &mut Env,
) -> Result<Value, TemplateError> {
    if let Expr::Attr(base, method) = func {
        let receiver = eval(base, env)?;
        let args = eval_all(args, env)?;
        return super::builtins::method(method, &receiver, &args);
    }
    let Expr::Name(name) = func else {
        return Err(TemplateError::Unsupported(
            "only named functions and methods can be called".into(),
        ));
    };
    let args = eval_all(args, env)?;
    let kwargs = eval_kwargs(kwargs, env)?;
    super::builtins::function(name, &args, &kwargs)
}

/// `a.b`: mapping key, then the string/list pseudo-attributes Jinja exposes.
fn member(base: &Value, field: &str) -> Result<Value, TemplateError> {
    if let Some(v) = base.get_key(field) {
        return Ok(v);
    }
    match base {
        Value::Undefined => Err(TemplateError::Runtime(format!(
            "{field} of an undefined value"
        ))),
        other => Err(TemplateError::Runtime(format!(
            "{} has no attribute {field}",
            other.type_name()
        ))),
    }
}

fn subscript(base: &Value, index: &Value) -> Result<Value, TemplateError> {
    match (base, index) {
        (Value::Undefined, _) => Err(TemplateError::Runtime(
            "cannot index an undefined value".into(),
        )),
        (Value::Map(_), Value::Str(k)) => Ok(base.get_key(k).unwrap_or(Value::Undefined)),
        (Value::List(items), Value::Int(i)) => Ok(nth(items.len(), *i)
            .and_then(|n| items.get(n).cloned())
            .unwrap_or(Value::Undefined)),
        (Value::Str(s), Value::Int(i)) => {
            let chars: Vec<char> = s.chars().collect();
            Ok(nth(chars.len(), *i)
                .and_then(|n| chars.get(n))
                .map_or(Value::Undefined, |c| Value::Str(c.to_string())))
        }
        (b, i) => Err(TemplateError::Runtime(format!(
            "cannot index {} with {}",
            b.type_name(),
            i.type_name()
        ))),
    }
}

/// Python index: negatives count from the end, out of range is `None`.
fn nth(len: usize, i: i64) -> Option<usize> {
    let len_i = i64::try_from(len).ok()?;
    let abs = if i < 0 { len_i.checked_add(i)? } else { i };
    if abs < 0 || abs >= len_i {
        return None;
    }
    usize::try_from(abs).ok()
}

/// Python slice bound: negatives count from the end, then clamp.
fn bound(len: usize, i: Option<&Value>, default: usize) -> Result<usize, TemplateError> {
    let Some(v) = i else { return Ok(default) };
    let Value::Int(n) = v else {
        return Err(TemplateError::Runtime(format!(
            "slice bounds must be integers, got {}",
            v.type_name()
        )));
    };
    let len_i = i64::try_from(len).unwrap_or(i64::MAX);
    let abs = if *n < 0 {
        len_i.saturating_add(*n).max(0)
    } else {
        (*n).min(len_i)
    };
    Ok(usize::try_from(abs).unwrap_or(0))
}

fn slice(
    base: &Value,
    start: Option<&Value>,
    stop: Option<&Value>,
) -> Result<Value, TemplateError> {
    match base {
        Value::List(items) => {
            let a = bound(items.len(), start, 0)?;
            let b = bound(items.len(), stop, items.len())?;
            Ok(Value::List(
                items
                    .get(a..b.max(a))
                    .map(<[Value]>::to_vec)
                    .unwrap_or_default(),
            ))
        }
        Value::Str(s) => {
            let chars: Vec<char> = s.chars().collect();
            let a = bound(chars.len(), start, 0)?;
            let b = bound(chars.len(), stop, chars.len())?;
            Ok(Value::Str(
                chars.get(a..b.max(a)).unwrap_or_default().iter().collect(),
            ))
        }
        other => Err(TemplateError::Runtime(format!(
            "cannot slice {}",
            other.type_name()
        ))),
    }
}

fn binary(op: BinOp, a: &Value, b: &Value) -> Result<Value, TemplateError> {
    match op {
        BinOp::Eq => return Ok(Value::Bool(values_equal(a, b))),
        BinOp::Ne => return Ok(Value::Bool(!values_equal(a, b))),
        BinOp::In => return contains(b, a).map(Value::Bool),
        BinOp::NotIn => return contains(b, a).map(|c| Value::Bool(!c)),
        BinOp::Concat => {
            return Ok(Value::Str(format!("{}{}", a.render(), b.render())));
        }
        _ => {}
    }
    if matches!(a, Value::Undefined) || matches!(b, Value::Undefined) {
        return Err(TemplateError::Runtime(format!(
            "{op:?} on an undefined value"
        )));
    }
    if let (BinOp::Add, Value::Str(x), Value::Str(y)) = (op, a, b) {
        return Ok(Value::Str(format!("{x}{y}")));
    }
    if let (BinOp::Add, Value::List(x), Value::List(y)) = (op, a, b) {
        let mut out = x.clone();
        out.extend(y.iter().cloned());
        return Ok(Value::List(out));
    }
    if let (BinOp::Mul, Value::Str(x), Value::Int(n)) = (op, a, b) {
        let times = usize::try_from(*n).unwrap_or(0);
        return Ok(Value::Str(x.repeat(times)));
    }
    let (Some(x), Some(y)) = (as_int(a), as_int(b)) else {
        if matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) {
            if let (Value::Str(x), Value::Str(y)) = (a, b) {
                return Ok(Value::Bool(compare_ord(op, x.cmp(y))));
            }
        }
        return Err(TemplateError::Runtime(format!(
            "{op:?} is not defined for {} and {}",
            a.type_name(),
            b.type_name()
        )));
    };
    Ok(match op {
        BinOp::Add => Value::Int(x.saturating_add(y)),
        BinOp::Sub => Value::Int(x.saturating_sub(y)),
        BinOp::Mul => Value::Int(x.saturating_mul(y)),
        BinOp::Div | BinOp::FloorDiv => Value::Int(div_floor(x, y)?),
        BinOp::Rem => Value::Int(rem_floor(x, y)?),
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => Value::Bool(compare_ord(op, x.cmp(&y))),
        BinOp::Eq | BinOp::Ne | BinOp::In | BinOp::NotIn | BinOp::Concat => Value::Undefined,
    })
}

fn compare_ord(op: BinOp, ord: std::cmp::Ordering) -> bool {
    match op {
        BinOp::Lt => ord.is_lt(),
        BinOp::Le => ord.is_le(),
        BinOp::Gt => ord.is_gt(),
        _ => ord.is_ge(),
    }
}

fn as_int(v: &Value) -> Option<i64> {
    match v {
        Value::Int(i) => Some(*i),
        Value::Bool(b) => Some(i64::from(*b)),
        _ => None,
    }
}

/// Python floor division, which differs from Rust's truncation for negatives.
fn div_floor(x: i64, y: i64) -> Result<i64, TemplateError> {
    if y == 0 {
        return Err(TemplateError::Runtime("division by zero".into()));
    }
    let q = x.checked_div(y).unwrap_or(0);
    let r = x.checked_rem(y).unwrap_or(0);
    Ok(if r != 0 && ((r < 0) != (y < 0)) {
        q.saturating_sub(1)
    } else {
        q
    })
}

/// Python `%`, whose result takes the sign of the divisor.
fn rem_floor(x: i64, y: i64) -> Result<i64, TemplateError> {
    if y == 0 {
        return Err(TemplateError::Runtime("modulo by zero".into()));
    }
    let r = x % y;
    Ok(if r != 0 && ((r < 0) != (y < 0)) {
        r.saturating_add(y)
    } else {
        r
    })
}

fn contains(haystack: &Value, needle: &Value) -> Result<bool, TemplateError> {
    match (haystack, needle) {
        (Value::Str(h), Value::Str(n)) => Ok(h.contains(n.as_str())),
        (Value::List(items), n) => Ok(items.iter().any(|item| values_equal(item, n))),
        (Value::Map(entries), Value::Str(k)) => Ok(entries.iter().any(|(key, _)| key == k)),
        (h, n) => Err(TemplateError::Runtime(format!(
            "`in` is not defined for {} in {}",
            n.type_name(),
            h.type_name()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::{div_floor, rem_floor, Env};
    use crate::template::parse::parse;
    use crate::template::value::Value;

    fn render_with(
        src: &str,
        vars: &[(&str, Value)],
    ) -> Result<String, crate::template::TemplateError> {
        let body = parse(src)?;
        let globals = vars
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect();
        super::render(&body, &mut Env::new(globals))
    }

    fn r(src: &str) -> String {
        render_with(src, &[]).expect("render")
    }

    #[test]
    fn python_modulo_and_division_semantics() {
        assert_eq!(rem_floor(7, 2).unwrap(), 1);
        assert_eq!(rem_floor(-7, 2).unwrap(), 1);
        assert_eq!(rem_floor(7, -2).unwrap(), -1);
        assert_eq!(div_floor(7, 2).unwrap(), 3);
        assert_eq!(div_floor(-7, 2).unwrap(), -4);
        assert!(rem_floor(1, 0).is_err());
        assert_eq!(r("{{ 5 % 2 }}{{ 4 % 2 }}"), "10");
    }

    #[test]
    fn loop_exposes_index_first_last_and_length() {
        let xs = Value::List(vec![Value::str("a"), Value::str("b"), Value::str("c")]);
        let src = "{% for x in xs %}{{ loop.index }}{{ x }}{% if loop.first %}F{% endif %}\
                   {% if loop.last %}L{% endif %}{{ loop.index0 }}{{ loop.revindex }}\
                   {{ loop.length }};{% endfor %}";
        assert_eq!(
            render_with(src, &[("xs", xs)]).unwrap(),
            "1aF033;2b123;3cL213;"
        );
    }

    #[test]
    fn for_else_break_and_continue() {
        let xs = Value::List(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
        assert_eq!(
            render_with(
                "{% for x in xs %}{{ x }}{% endfor %}",
                &[("xs", xs.clone())]
            )
            .unwrap(),
            "123"
        );
        assert_eq!(
            render_with(
                "{% for x in xs %}{% if x == 2 %}{% break %}{% endif %}{{ x }}{% endfor %}",
                &[("xs", xs.clone())]
            )
            .unwrap(),
            "1"
        );
        assert_eq!(
            render_with(
                "{% for x in xs %}{% if x == 2 %}{% continue %}{% endif %}{{ x }}{% endfor %}",
                &[("xs", xs.clone())]
            )
            .unwrap(),
            "13"
        );
        assert_eq!(
            render_with(
                "{% for x in xs %}{{ x }}{% else %}none{% endfor %}",
                &[("xs", Value::List(vec![]))]
            )
            .unwrap(),
            "none"
        );
        assert_eq!(
            render_with(
                "{% for x in xs if x != 2 %}{{ x }}{% endfor %}",
                &[("xs", xs)]
            )
            .unwrap(),
            "13"
        );
    }

    #[test]
    fn set_scoping_matches_jinja_frames() {
        // `{% if %}` does not open a frame, so the assignment survives.
        assert_eq!(r("{% if true %}{% set x = 'a' %}{% endif %}{{ x }}"), "a");
        // `{% for %}` does, so it does not.
        assert_eq!(
            r("{% set x = 'out' %}{% for i in [1] %}{% set x = 'in' %}{% endfor %}{{ x }}"),
            "out"
        );
        // ... which is exactly why `namespace()` exists.
        assert_eq!(
            r("{% set ns = namespace(v='out') %}{% for i in [1] %}{% set ns.v = 'in' %}{% endfor %}{{ ns.v }}"),
            "in"
        );
    }

    #[test]
    fn undefined_renders_empty_but_is_an_error_to_walk_into() {
        assert_eq!(r("[{{ nope }}]"), "[]");
        assert_eq!(r("{% if nope %}y{% else %}n{% endif %}"), "n");
        assert_eq!(r("{{ nope == 'x' }}"), "False");
        assert_eq!(r("{{ not nope }}"), "True");
        assert!(render_with("{{ nope.field }}", &[]).is_err());
        assert!(render_with("{{ nope['k'] }}", &[]).is_err());
        assert!(render_with("{{ 'a' + nope }}", &[]).is_err());
        assert!(render_with("{% for x in nope %}{% endfor %}", &[]).is_err());
    }

    #[test]
    fn indexing_slicing_and_attribute_access() {
        let msgs = Value::List(vec![
            Value::Map(vec![
                ("role".into(), Value::str("system")),
                ("content".into(), Value::str("S")),
            ]),
            Value::Map(vec![
                ("role".into(), Value::str("user")),
                ("content".into(), Value::str("U")),
            ]),
        ]);
        let v = [("messages", msgs)];
        assert_eq!(
            render_with("{{ messages[0]['role'] }}", &v).unwrap(),
            "system"
        );
        assert_eq!(render_with("{{ messages[0].content }}", &v).unwrap(), "S");
        assert_eq!(render_with("{{ messages[-1].role }}", &v).unwrap(), "user");
        assert_eq!(render_with("{{ messages[1:] | length }}", &v).unwrap(), "1");
        assert_eq!(render_with("{{ messages[5] }}", &v).unwrap(), "");
        assert_eq!(
            render_with("{{ 'role' in messages[0] }}", &v).unwrap(),
            "True"
        );
        assert_eq!(
            render_with("{{ 'tool_calls' in messages[0] }}", &v).unwrap(),
            "False"
        );
        assert_eq!(r("{{ 'abcdef'[1:3] }}"), "bc");
        assert_eq!(r("{{ 'abc' in 'xabcy' }}"), "True");
    }

    #[test]
    fn raise_exception_becomes_an_error_carrying_the_message() {
        let err =
            render_with("{{ raise_exception('roles must alternate') }}", &[]).expect_err("raised");
        assert!(err.to_string().contains("roles must alternate"), "{err}");
        // It only fires on the branch that reaches it.
        assert_eq!(
            r("{% if false %}{{ raise_exception('no') }}{% endif %}ok"),
            "ok"
        );
    }

    #[test]
    fn ternary_and_short_circuit() {
        assert_eq!(r("{{ 'y' if true else 'n' }}"), "y");
        assert_eq!(r("{{ 'y' if false else 'n' }}"), "n");
        assert_eq!(r("[{{ 'y' if false }}]"), "[]");
        // `and`/`or` must not evaluate the far side when they do not need it.
        assert_eq!(r("{{ false and raise_exception('boom') }}"), "False");
        assert_eq!(r("{{ 'a' or raise_exception('boom') }}"), "a");
    }
}
