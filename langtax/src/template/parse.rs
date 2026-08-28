//! Tag pieces to an AST.
//!
//! Precedence follows Jinja's recursive-descent grammar, which matters in real
//! templates: filters and tests bind tighter than everything else, so
//! `not tools is defined` is `not (tools is defined)` and
//! `message['content'] | trim + '<|eot_id|>'` trims before it concatenates.
//!
//! Anything outside the supported subset is rejected by name rather than
//! ignored, because a chat template that renders *almost* right produces a
//! prompt the model was never trained on.

use super::lex::{lex, Piece, Tok};
use super::value::Value;
use super::TemplateError;

/// Binary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BinOp {
    /// `+`. Numeric addition, string concatenation, or list concatenation.
    Add,
    /// `-`.
    Sub,
    /// `*`.
    Mul,
    /// `/`. Integer division, since floats are unsupported.
    Div,
    /// `//`.
    FloorDiv,
    /// `%`.
    Rem,
    /// `~`, Jinja's string concatenation.
    Concat,
    /// `==`.
    Eq,
    /// `!=`.
    Ne,
    /// `<`.
    Lt,
    /// `<=`.
    Le,
    /// `>`.
    Gt,
    /// `>=`.
    Ge,
    /// `in`.
    In,
    /// `not in`.
    NotIn,
}

/// An expression.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Expr {
    /// Literal.
    Const(Value),
    /// Variable lookup.
    Name(String),
    /// `a.b`.
    Attr(Box<Expr>, String),
    /// `a[b]`.
    Item(Box<Expr>, Box<Expr>),
    /// `a[start:stop]`.
    Slice(Box<Expr>, Option<Box<Expr>>, Option<Box<Expr>>),
    /// `[a, b]`.
    List(Vec<Expr>),
    /// `{'a': b}`.
    Dict(Vec<(Expr, Expr)>),
    /// `-a`.
    Neg(Box<Expr>),
    /// `not a`.
    Not(Box<Expr>),
    /// `a and b`, short-circuiting.
    And(Box<Expr>, Box<Expr>),
    /// `a or b`, short-circuiting.
    Or(Box<Expr>, Box<Expr>),
    /// Binary operator application.
    Bin(BinOp, Box<Expr>, Box<Expr>),
    /// `body if test else other`.
    Cond {
        /// Value when `test` is true.
        body: Box<Expr>,
        /// Condition.
        test: Box<Expr>,
        /// Value when `test` is false; undefined when the `else` is omitted.
        other: Option<Box<Expr>>,
    },
    /// `value | name(args)`.
    Filter {
        /// Filtered value.
        value: Box<Expr>,
        /// Filter name.
        name: String,
        /// Positional arguments.
        args: Vec<Expr>,
        /// Keyword arguments.
        kwargs: Vec<(String, Expr)>,
    },
    /// `value is name(args)`, optionally negated by `is not`.
    Test {
        /// Tested value.
        value: Box<Expr>,
        /// Test name.
        name: String,
        /// Whether the test was written `is not`.
        negated: bool,
        /// Positional arguments.
        args: Vec<Expr>,
    },
    /// `name(args)`.
    Call {
        /// Callee.
        func: Box<Expr>,
        /// Positional arguments.
        args: Vec<Expr>,
        /// Keyword arguments.
        kwargs: Vec<(String, Expr)>,
    },
}

/// Positional and keyword arguments of a call, filter or test.
type CallArgs = (Vec<Expr>, Vec<(String, Expr)>);

/// A `{% set %}` assignment target.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Target {
    /// `{% set name = ... %}`.
    Name(String),
    /// `{% set ns.attr = ... %}` into a `namespace()`.
    Attr(String, Vec<String>),
}

/// A statement in the rendered body.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Node {
    /// Literal output.
    Text(String),
    /// `{{ expr }}`.
    Output(Expr),
    /// `{% if %}` / `{% elif %}` / `{% else %}`.
    If {
        /// Condition/body pairs, `if` first then each `elif`.
        branches: Vec<(Expr, Vec<Node>)>,
        /// `{% else %}` body.
        otherwise: Vec<Node>,
    },
    /// `{% for %}`.
    For {
        /// Loop variable names; more than one unpacks each item.
        targets: Vec<String>,
        /// Sequence expression.
        iter: Expr,
        /// Optional `{% for x in xs if cond %}` filter.
        filter: Option<Expr>,
        /// Loop body.
        body: Vec<Node>,
        /// `{% else %}` body, rendered when nothing was iterated.
        otherwise: Vec<Node>,
    },
    /// `{% set %}`.
    Set {
        /// Assignment target.
        target: Target,
        /// Assigned expression.
        value: Expr,
    },
    /// `{% break %}` from `jinja2.ext.loopcontrols`.
    Break,
    /// `{% continue %}` from `jinja2.ext.loopcontrols`.
    Continue,
}

/// Parse template source into a body of statements.
pub(crate) fn parse(src: &str) -> Result<Vec<Node>, TemplateError> {
    let pieces = lex(src)?;
    let mut p = Parser { pieces, at: 0 };
    let (body, end) = p.parse_body(&[])?;
    if let Some(tag) = end {
        return Err(TemplateError::Syntax(format!("unexpected {{% {tag} %}}")));
    }
    Ok(body)
}

struct Parser {
    pieces: Vec<Piece>,
    at: usize,
}

impl Parser {
    /// Parse until one of `stop` (or end of input), returning the tag that stopped it.
    fn parse_body(&mut self, stop: &[&str]) -> Result<(Vec<Node>, Option<String>), TemplateError> {
        let mut body = Vec::new();
        while let Some(piece) = self.pieces.get(self.at).cloned() {
            match piece {
                Piece::Text(t) => {
                    self.at = self.at.saturating_add(1);
                    body.push(Node::Text(t));
                }
                Piece::Expr(toks) => {
                    self.at = self.at.saturating_add(1);
                    let mut e = ExprParser { toks, at: 0 };
                    let expr = e.parse_expr()?;
                    e.expect_end("{{ }}")?;
                    body.push(Node::Output(expr));
                }
                Piece::Block(toks) => {
                    let Some(Tok::Name(keyword)) = toks.first().cloned() else {
                        return Err(TemplateError::Syntax("empty {% %} tag".into()));
                    };
                    if stop.contains(&keyword.as_str()) {
                        return Ok((body, Some(keyword)));
                    }
                    self.at = self.at.saturating_add(1);
                    self.parse_statement(&keyword, &toks, &mut body)?;
                }
            }
        }
        Ok((body, None))
    }

    fn parse_statement(
        &mut self,
        keyword: &str,
        toks: &[Tok],
        body: &mut Vec<Node>,
    ) -> Result<(), TemplateError> {
        let rest = toks.get(1..).unwrap_or(&[]).to_vec();
        match keyword {
            "if" => body.push(self.parse_if(rest)?),
            "for" => body.push(self.parse_for(rest)?),
            "set" => body.push(parse_set(&rest)?),
            "break" => body.push(Node::Break),
            "continue" => body.push(Node::Continue),
            // The `{% generation %}` marker from the `transformers`
            // AssistantTracker extension only tags spans for assistant-token
            // masking; it contributes nothing to the rendered text.
            "generation" => {
                let (inner, end) = self.parse_body(&["endgeneration"])?;
                self.expect_close(end, "endgeneration")?;
                body.extend(inner);
            }
            other => {
                return Err(TemplateError::Unsupported(format!(
                    "{{% {other} %}} is not supported"
                )))
            }
        }
        Ok(())
    }

    fn parse_if(&mut self, first: Vec<Tok>) -> Result<Node, TemplateError> {
        let mut branches = Vec::new();
        let mut cond_toks = first;
        let mut otherwise = Vec::new();
        loop {
            let mut e = ExprParser {
                toks: cond_toks,
                at: 0,
            };
            let cond = e.parse_expr()?;
            e.expect_end("{% if %}")?;
            let (block, end) = self.parse_body(&["elif", "else", "endif"])?;
            branches.push((cond, block));
            let Some(tag) = end else {
                return Err(TemplateError::Syntax("missing {% endif %}".into()));
            };
            let toks = self.take_stop_tag();
            match tag.as_str() {
                "elif" => cond_toks = toks.get(1..).unwrap_or(&[]).to_vec(),
                "else" => {
                    let (block, end) = self.parse_body(&["endif"])?;
                    self.expect_close(end, "endif")?;
                    otherwise = block;
                    break;
                }
                _ => break,
            }
        }
        Ok(Node::If {
            branches,
            otherwise,
        })
    }

    fn parse_for(&mut self, header: Vec<Tok>) -> Result<Node, TemplateError> {
        let mut targets = Vec::new();
        let mut at = 0usize;
        loop {
            match header.get(at) {
                Some(Tok::Name(n)) if n != "in" => targets.push(n.clone()),
                _ => {
                    return Err(TemplateError::Syntax(
                        "{% for %} needs a loop variable".into(),
                    ))
                }
            }
            at = at.saturating_add(1);
            if matches!(header.get(at), Some(t) if t.is_punct(",")) {
                at = at.saturating_add(1);
                continue;
            }
            break;
        }
        if !matches!(header.get(at), Some(t) if t.is_name("in")) {
            return Err(TemplateError::Syntax("{% for %} needs `in`".into()));
        }
        at = at.saturating_add(1);
        let mut e = ExprParser {
            toks: header.get(at..).unwrap_or(&[]).to_vec(),
            at: 0,
        };
        // The iterable is parsed without the ternary, or `{% for x in xs if c %}`
        // would swallow the loop filter as `xs if c`. Jinja does the same.
        let iter = e.parse_or()?;
        let filter = if matches!(e.peek(), Some(t) if t.is_name("if")) {
            let _if = e.bump();
            Some(e.parse_or()?)
        } else {
            None
        };
        if matches!(e.peek(), Some(t) if t.is_name("recursive")) {
            return Err(TemplateError::Unsupported(
                "recursive {% for %} is not supported".into(),
            ));
        }
        e.expect_end("{% for %}")?;
        let (body, end) = self.parse_body(&["else", "endfor"])?;
        let Some(tag) = end else {
            return Err(TemplateError::Syntax("missing {% endfor %}".into()));
        };
        let _tag = self.take_stop_tag();
        let otherwise = if tag == "else" {
            let (block, end) = self.parse_body(&["endfor"])?;
            self.expect_close(end, "endfor")?;
            block
        } else {
            Vec::new()
        };
        Ok(Node::For {
            targets,
            iter,
            filter,
            body,
            otherwise,
        })
    }

    /// Consume the `{% ... %}` piece `parse_body` stopped on.
    fn take_stop_tag(&mut self) -> Vec<Tok> {
        let toks = match self.pieces.get(self.at) {
            Some(Piece::Block(t)) => t.clone(),
            _ => Vec::new(),
        };
        self.at = self.at.saturating_add(1);
        toks
    }

    fn expect_close(&mut self, end: Option<String>, want: &str) -> Result<(), TemplateError> {
        if end.as_deref() == Some(want) {
            let _toks = self.take_stop_tag();
            Ok(())
        } else {
            Err(TemplateError::Syntax(format!("missing {{% {want} %}}")))
        }
    }
}

fn parse_set(toks: &[Tok]) -> Result<Node, TemplateError> {
    let Some(Tok::Name(head)) = toks.first() else {
        return Err(TemplateError::Syntax("{% set %} needs a name".into()));
    };
    let mut path = Vec::new();
    let mut at = 1usize;
    while matches!(toks.get(at), Some(t) if t.is_punct(".")) {
        let Some(Tok::Name(field)) = toks.get(at.saturating_add(1)) else {
            return Err(TemplateError::Syntax(
                "{% set %} attribute target needs a name".into(),
            ));
        };
        path.push(field.clone());
        at = at.saturating_add(2);
    }
    if !matches!(toks.get(at), Some(t) if t.is_punct("=")) {
        return Err(TemplateError::Unsupported(
            "block {% set %} without `=` is not supported".into(),
        ));
    }
    let mut e = ExprParser {
        toks: toks.get(at.saturating_add(1)..).unwrap_or(&[]).to_vec(),
        at: 0,
    };
    let value = e.parse_expr()?;
    e.expect_end("{% set %}")?;
    let target = if path.is_empty() {
        Target::Name(head.clone())
    } else {
        Target::Attr(head.clone(), path)
    };
    Ok(Node::Set { target, value })
}

struct ExprParser {
    toks: Vec<Tok>,
    at: usize,
}

impl ExprParser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.at)
    }

    fn bump(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.at).cloned();
        if t.is_some() {
            self.at = self.at.saturating_add(1);
        }
        t
    }

    fn eat_punct(&mut self, want: &str) -> bool {
        if matches!(self.peek(), Some(t) if t.is_punct(want)) {
            self.at = self.at.saturating_add(1);
            true
        } else {
            false
        }
    }

    fn eat_name(&mut self, want: &str) -> bool {
        if matches!(self.peek(), Some(t) if t.is_name(want)) {
            self.at = self.at.saturating_add(1);
            true
        } else {
            false
        }
    }

    fn expect_punct(&mut self, want: &str) -> Result<(), TemplateError> {
        if self.eat_punct(want) {
            Ok(())
        } else {
            Err(TemplateError::Syntax(format!(
                "expected {want:?}, got {:?}",
                self.peek()
            )))
        }
    }

    fn expect_end(&mut self, what: &str) -> Result<(), TemplateError> {
        match self.peek() {
            None => Ok(()),
            Some(t) => Err(TemplateError::Syntax(format!(
                "trailing {t:?} in {what} expression"
            ))),
        }
    }

    /// `a if cond else b`.
    fn parse_expr(&mut self) -> Result<Expr, TemplateError> {
        let body = self.parse_or()?;
        if !self.eat_name("if") {
            return Ok(body);
        }
        let test = self.parse_or()?;
        let other = if self.eat_name("else") {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };
        Ok(Expr::Cond {
            body: Box::new(body),
            test: Box::new(test),
            other,
        })
    }

    fn parse_or(&mut self) -> Result<Expr, TemplateError> {
        let mut lhs = self.parse_and()?;
        while self.eat_name("or") {
            let rhs = self.parse_and()?;
            lhs = Expr::Or(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<Expr, TemplateError> {
        let mut lhs = self.parse_not()?;
        while self.eat_name("and") {
            let rhs = self.parse_not()?;
            lhs = Expr::And(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_not(&mut self) -> Result<Expr, TemplateError> {
        if self.eat_name("not") {
            return Ok(Expr::Not(Box::new(self.parse_not()?)));
        }
        self.parse_compare()
    }

    fn parse_compare(&mut self) -> Result<Expr, TemplateError> {
        let mut lhs = self.parse_math1()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Punct("==")) => Some(BinOp::Eq),
                Some(Tok::Punct("!=")) => Some(BinOp::Ne),
                Some(Tok::Punct("<")) => Some(BinOp::Lt),
                Some(Tok::Punct("<=")) => Some(BinOp::Le),
                Some(Tok::Punct(">")) => Some(BinOp::Gt),
                Some(Tok::Punct(">=")) => Some(BinOp::Ge),
                Some(Tok::Name(n)) if n == "in" => Some(BinOp::In),
                Some(Tok::Name(n)) if n == "not" => Some(BinOp::NotIn),
                _ => None,
            };
            let Some(op) = op else { break };
            if op == BinOp::NotIn {
                // Only `not in` continues a comparison; a bare `not` here is
                // the start of the next construct and belongs to the caller.
                if !matches!(self.toks.get(self.at.saturating_add(1)), Some(t) if t.is_name("in")) {
                    break;
                }
                self.at = self.at.saturating_add(2);
            } else {
                self.at = self.at.saturating_add(1);
            }
            let rhs = self.parse_math1()?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_math1(&mut self) -> Result<Expr, TemplateError> {
        let mut lhs = self.parse_concat()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Punct("+")) => BinOp::Add,
                Some(Tok::Punct("-")) => BinOp::Sub,
                _ => break,
            };
            self.at = self.at.saturating_add(1);
            let rhs = self.parse_concat()?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_concat(&mut self) -> Result<Expr, TemplateError> {
        let mut lhs = self.parse_math2()?;
        while self.eat_punct("~") {
            let rhs = self.parse_math2()?;
            lhs = Expr::Bin(BinOp::Concat, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_math2(&mut self) -> Result<Expr, TemplateError> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Punct("*")) => BinOp::Mul,
                Some(Tok::Punct("//")) => BinOp::FloorDiv,
                Some(Tok::Punct("/")) => BinOp::Div,
                Some(Tok::Punct("%")) => BinOp::Rem,
                Some(Tok::Punct("**")) => {
                    return Err(TemplateError::Unsupported("`**` is not supported".into()))
                }
                _ => break,
            };
            self.at = self.at.saturating_add(1);
            let rhs = self.parse_unary()?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr, TemplateError> {
        let expr = if self.eat_punct("-") {
            Expr::Neg(Box::new(self.parse_unary()?))
        } else if self.eat_punct("+") {
            self.parse_unary()?
        } else {
            let primary = self.parse_primary()?;
            self.parse_postfix(primary)?
        };
        self.parse_filters(expr)
    }

    /// `|filter` and `is test`, which bind tighter than any operator.
    fn parse_filters(&mut self, mut expr: Expr) -> Result<Expr, TemplateError> {
        loop {
            if self.eat_punct("|") {
                let Some(Tok::Name(name)) = self.bump() else {
                    return Err(TemplateError::Syntax("filter needs a name".into()));
                };
                let (args, kwargs) = self.parse_call_args()?;
                expr = Expr::Filter {
                    value: Box::new(expr),
                    name,
                    args,
                    kwargs,
                };
                continue;
            }
            if matches!(self.peek(), Some(t) if t.is_name("is")) {
                self.at = self.at.saturating_add(1);
                let negated = self.eat_name("not");
                let Some(Tok::Name(name)) = self.bump() else {
                    return Err(TemplateError::Syntax("test needs a name".into()));
                };
                let (mut args, kwargs) = self.parse_call_args()?;
                if !kwargs.is_empty() {
                    return Err(TemplateError::Unsupported(
                        "keyword arguments to tests are not supported".into(),
                    ));
                }
                // A test may take one unparenthesised operand, as in
                // `loop.index is divisibleby 3`.
                if args.is_empty() && self.starts_bare_test_arg() {
                    args.push(self.parse_or()?);
                }
                expr = Expr::Test {
                    value: Box::new(expr),
                    name,
                    negated,
                    args,
                };
                continue;
            }
            return Ok(expr);
        }
    }

    /// Whether the next token opens a bare test operand rather than continuing
    /// the surrounding expression.
    fn starts_bare_test_arg(&self) -> bool {
        match self.peek() {
            Some(Tok::Str(_) | Tok::Int(_)) => true,
            Some(Tok::Punct(p)) => matches!(*p, "(" | "[" | "{"),
            Some(Tok::Name(n)) => !matches!(
                n.as_str(),
                "else" | "or" | "and" | "if" | "in" | "is" | "not"
            ),
            None => false,
        }
    }

    fn parse_postfix(&mut self, mut expr: Expr) -> Result<Expr, TemplateError> {
        loop {
            if self.eat_punct(".") {
                let Some(Tok::Name(field)) = self.bump() else {
                    return Err(TemplateError::Syntax("attribute needs a name".into()));
                };
                expr = Expr::Attr(Box::new(expr), field);
                continue;
            }
            if self.eat_punct("[") {
                expr = self.parse_subscript(expr)?;
                continue;
            }
            if matches!(self.peek(), Some(t) if t.is_punct("(")) {
                let (args, kwargs) = self.parse_call_args()?;
                expr = Expr::Call {
                    func: Box::new(expr),
                    args,
                    kwargs,
                };
                continue;
            }
            return Ok(expr);
        }
    }

    fn parse_subscript(&mut self, base: Expr) -> Result<Expr, TemplateError> {
        if self.eat_punct(":") {
            let stop = if matches!(self.peek(), Some(t) if t.is_punct("]")) {
                None
            } else {
                Some(Box::new(self.parse_expr()?))
            };
            self.expect_punct("]")?;
            return Ok(Expr::Slice(Box::new(base), None, stop));
        }
        let first = self.parse_expr()?;
        if self.eat_punct(":") {
            let stop = if matches!(self.peek(), Some(t) if t.is_punct("]")) {
                None
            } else {
                Some(Box::new(self.parse_expr()?))
            };
            self.expect_punct("]")?;
            return Ok(Expr::Slice(Box::new(base), Some(Box::new(first)), stop));
        }
        self.expect_punct("]")?;
        Ok(Expr::Item(Box::new(base), Box::new(first)))
    }

    /// `(a, b, k=v)`, or nothing when the next token is not `(`.
    fn parse_call_args(&mut self) -> Result<CallArgs, TemplateError> {
        let mut args = Vec::new();
        let mut kwargs = Vec::new();
        if !self.eat_punct("(") {
            return Ok((args, kwargs));
        }
        if self.eat_punct(")") {
            return Ok((args, kwargs));
        }
        loop {
            let is_kwarg = matches!(self.peek(), Some(Tok::Name(_)))
                && matches!(self.toks.get(self.at.saturating_add(1)), Some(t) if t.is_punct("="));
            if is_kwarg {
                let Some(Tok::Name(key)) = self.bump() else {
                    return Err(TemplateError::Syntax("keyword argument name".into()));
                };
                self.expect_punct("=")?;
                kwargs.push((key, self.parse_expr()?));
            } else {
                args.push(self.parse_expr()?);
            }
            if self.eat_punct(",") {
                if matches!(self.peek(), Some(t) if t.is_punct(")")) {
                    break;
                }
                continue;
            }
            break;
        }
        self.expect_punct(")")?;
        Ok((args, kwargs))
    }

    fn parse_primary(&mut self) -> Result<Expr, TemplateError> {
        let Some(tok) = self.bump() else {
            return Err(TemplateError::Syntax("expected an expression".into()));
        };
        match tok {
            Tok::Str(s) => {
                // Adjacent string literals concatenate, as in Python.
                let mut joined = s;
                while let Some(Tok::Str(next)) = self.peek().cloned() {
                    self.at = self.at.saturating_add(1);
                    joined.push_str(&next);
                }
                Ok(Expr::Const(Value::Str(joined)))
            }
            Tok::Int(n) => Ok(Expr::Const(Value::Int(n))),
            Tok::Name(n) => Ok(match n.as_str() {
                "true" | "True" => Expr::Const(Value::Bool(true)),
                "false" | "False" => Expr::Const(Value::Bool(false)),
                "none" | "None" => Expr::Const(Value::None),
                _ => Expr::Name(n),
            }),
            Tok::Punct("(") => {
                let inner = self.parse_expr()?;
                if self.eat_punct(",") {
                    let mut items = vec![inner];
                    while !matches!(self.peek(), Some(t) if t.is_punct(")")) {
                        items.push(self.parse_expr()?);
                        if !self.eat_punct(",") {
                            break;
                        }
                    }
                    self.expect_punct(")")?;
                    return Ok(Expr::List(items));
                }
                self.expect_punct(")")?;
                Ok(inner)
            }
            Tok::Punct("[") => {
                let mut items = Vec::new();
                if !self.eat_punct("]") {
                    loop {
                        items.push(self.parse_expr()?);
                        if !self.eat_punct(",") {
                            break;
                        }
                        if matches!(self.peek(), Some(t) if t.is_punct("]")) {
                            break;
                        }
                    }
                    self.expect_punct("]")?;
                }
                Ok(Expr::List(items))
            }
            Tok::Punct("{") => {
                let mut entries = Vec::new();
                if !self.eat_punct("}") {
                    loop {
                        let key = self.parse_expr()?;
                        self.expect_punct(":")?;
                        entries.push((key, self.parse_expr()?));
                        if !self.eat_punct(",") {
                            break;
                        }
                        if matches!(self.peek(), Some(t) if t.is_punct("}")) {
                            break;
                        }
                    }
                    self.expect_punct("}")?;
                }
                Ok(Expr::Dict(entries))
            }
            other => Err(TemplateError::Syntax(format!(
                "unexpected {other:?} in expression"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{parse, BinOp, Expr, Node, Target};
    use crate::template::value::Value;

    fn one(src: &str) -> Node {
        let mut body = parse(src).expect("parse");
        assert_eq!(body.len(), 1, "{src:?} -> {body:?}");
        body.remove(0)
    }

    fn out(src: &str) -> Expr {
        match one(src) {
            Node::Output(e) => e,
            other => panic!("expected output, got {other:?}"),
        }
    }

    #[test]
    fn filters_and_tests_bind_tighter_than_operators() {
        // `not tools is defined` is `not (tools is defined)`, which decides
        // whether Llama-3 renders its tool preamble at all.
        assert_eq!(
            out("{{ not tools is defined }}"),
            Expr::Not(Box::new(Expr::Test {
                value: Box::new(Expr::Name("tools".into())),
                name: "defined".into(),
                negated: false,
                args: vec![],
            }))
        );
        // `a + b | trim` trims `b` before concatenating.
        let Expr::Bin(BinOp::Add, lhs, rhs) = out("{{ a + b | trim }}") else {
            panic!("expected add");
        };
        assert_eq!(*lhs, Expr::Name("a".into()));
        assert!(matches!(*rhs, Expr::Filter { .. }));
    }

    #[test]
    fn comparison_arithmetic_and_boolean_precedence() {
        // `(m.role == 'user') != (loop.index0 % 2 == 0)` from Mistral/Gemma.
        let e = out("{{ (m.role == 'user') != (loop.index0 % 2 == 0) }}");
        let Expr::Bin(BinOp::Ne, _, rhs) = e else {
            panic!("expected !=")
        };
        assert!(matches!(*rhs, Expr::Bin(BinOp::Eq, _, _)));
        // `and` binds tighter than `or`.
        let e = out("{{ a or b and c }}");
        let Expr::Or(_, rhs) = e else {
            panic!("expected or")
        };
        assert!(matches!(*rhs, Expr::And(_, _)));
    }

    #[test]
    fn not_in_is_a_comparison_but_a_bare_not_is_not() {
        assert_eq!(
            out("{{ 'x' not in m }}"),
            Expr::Bin(
                BinOp::NotIn,
                Box::new(Expr::Const(Value::str("x"))),
                Box::new(Expr::Name("m".into()))
            )
        );
        assert!(matches!(out("{{ a and not b }}"), Expr::And(_, _)));
    }

    #[test]
    fn subscripts_slices_attributes_and_calls() {
        assert!(matches!(out("{{ messages[0]['role'] }}"), Expr::Item(_, _)));
        assert!(matches!(out("{{ message.content }}"), Expr::Attr(_, _)));
        assert_eq!(
            out("{{ messages[1:] }}"),
            Expr::Slice(
                Box::new(Expr::Name("messages".into())),
                Some(Box::new(Expr::Const(Value::Int(1)))),
                None
            )
        );
        assert!(matches!(
            out("{{ messages[:2] }}"),
            Expr::Slice(_, None, Some(_))
        ));
        assert!(matches!(
            out("{{ raise_exception('no') }}"),
            Expr::Call { .. }
        ));
        let Expr::Filter { name, kwargs, .. } = out("{{ t | tojson(indent=4) }}") else {
            panic!("expected filter");
        };
        assert_eq!(name, "tojson");
        assert_eq!(kwargs.len(), 1);
    }

    #[test]
    fn ternary_literals_and_containers() {
        assert!(matches!(
            out("{{ a if b else c }}"),
            Expr::Cond { other: Some(_), .. }
        ));
        assert!(matches!(
            out("{{ a if b }}"),
            Expr::Cond { other: None, .. }
        ));
        assert_eq!(out("{{ true }}"), Expr::Const(Value::Bool(true)));
        assert_eq!(out("{{ None }}"), Expr::Const(Value::None));
        assert_eq!(
            out("{{ ['a', 1] }}"),
            Expr::List(vec![
                Expr::Const(Value::str("a")),
                Expr::Const(Value::Int(1))
            ])
        );
        assert!(matches!(out("{{ {'k': 'v'} }}"), Expr::Dict(_)));
        // Python-style adjacent literal concatenation.
        assert_eq!(out("{{ 'a' 'b' }}"), Expr::Const(Value::str("ab")));
    }

    #[test]
    fn statements_nest() {
        let Node::If {
            branches,
            otherwise,
        } = one("{% if a %}x{% elif b %}y{% else %}z{% endif %}")
        else {
            panic!("expected if");
        };
        assert_eq!(branches.len(), 2);
        assert_eq!(otherwise, vec![Node::Text("z".into())]);
        let Node::For { targets, body, .. } = one("{% for m in messages %}{{ m }}{% endfor %}")
        else {
            panic!("expected for");
        };
        assert_eq!(targets, vec!["m".to_string()]);
        assert_eq!(body.len(), 1);
        let Node::For { targets, .. } = one("{% for k, v in d.items() %}x{% endfor %}") else {
            panic!("expected for");
        };
        assert_eq!(targets, vec!["k".to_string(), "v".to_string()]);
        assert_eq!(
            one("{% set x = 1 %}"),
            Node::Set {
                target: Target::Name("x".into()),
                value: Expr::Const(Value::Int(1)),
            }
        );
        assert_eq!(
            one("{% set ns.found = true %}"),
            Node::Set {
                target: Target::Attr("ns".into(), vec!["found".into()]),
                value: Expr::Const(Value::Bool(true)),
            }
        );
        // `{% generation %}` is transparent.
        assert_eq!(
            one("{% generation %}hi{% endgeneration %}"),
            Node::Text("hi".into())
        );
    }

    #[test]
    fn unsupported_constructs_are_named_not_ignored() {
        for (src, want) in [
            ("{% macro m() %}{% endmacro %}", "macro"),
            ("{% include 'x' %}", "include"),
            ("{% extends 'x' %}", "extends"),
            ("{% block b %}{% endblock %}", "block"),
            ("{% filter upper %}{% endfilter %}", "filter"),
        ] {
            let err = parse(src).expect_err(src).to_string();
            assert!(err.contains(want), "{src}: {err}");
            assert!(err.contains("not supported"), "{src}: {err}");
        }
        assert!(parse("{% if a %}x")
            .expect_err("unterminated")
            .to_string()
            .contains("endif"));
        assert!(parse("{% for a in b %}x")
            .expect_err("unterminated")
            .to_string()
            .contains("endfor"));
        assert!(parse("{% endif %}").is_err());
        assert!(parse("{{ 2 ** 8 }}")
            .expect_err("pow")
            .to_string()
            .contains("**"));
    }
}
