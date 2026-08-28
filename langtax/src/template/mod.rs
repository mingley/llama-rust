//! A minimal Jinja2 subset, enough to render real GGUF `tokenizer.chat_template`
//! strings. Zero dependencies: no `minijinja`, no regex crate.
//!
//! # What is supported
//!
//! * `{{ expr }}` output and `{# comment #}`.
//! * Whitespace control: `{%- -%}`, `{{- -}}`, `{#- -#}`, plus the
//!   `trim_blocks` and `lstrip_blocks` defaults that HuggingFace
//!   `transformers` compiles chat templates with.
//! * `{% if %}` / `{% elif %}` / `{% else %}` / `{% endif %}`.
//! * `{% for x in xs %}` / `{% else %}` / `{% endfor %}`, tuple targets,
//!   `{% for x in xs if cond %}` filters, `{% break %}`, `{% continue %}`, and
//!   `loop.index`, `index0`, `revindex`, `revindex0`, `first`, `last`, `length`.
//! * `{% set x = expr %}` and `{% set ns.attr = expr %}` over `namespace()`.
//! * `{% generation %}` as a transparent wrapper.
//! * Literals (string, integer, `true`/`false`/`none` in both casings, lists,
//!   dicts), attribute and index access, Python slicing, `a if b else c`,
//!   `and`/`or`/`not`, `== != < <= > >=`, `in`/`not in`, `+ - * / // % ~`.
//! * Filters `trim`, `length`/`count`, `default`/`d`, `tojson`, `join`, `first`,
//!   `last`, `list`, `lower`, `upper`, `capitalize`, `title`, `string`, `int`,
//!   `abs`, `replace`, `reverse`, `sort`, `safe`.
//! * Tests `defined`, `undefined`, `none`, `string`, `number`, `integer`,
//!   `boolean`, `mapping`, `sequence`, `iterable`, `true`, `false`, `callable`,
//!   `equalto`/`eq`, `in`, `odd`, `even`, `divisibleby`.
//! * Methods `items`, `keys`, `values`, `get`, `strip`, `lstrip`, `rstrip`,
//!   `upper`, `lower`, `title`, `split`, `splitlines`, `join`, `startswith`,
//!   `endswith`, `replace`.
//! * Globals `raise_exception`, `namespace`, `dict`, `range`.
//!
//! # What is rejected
//!
//! Everything else, by name, as a [`TemplateError::Unsupported`]: `{% macro %}`,
//! `{% call %}`, `{% include %}`, `{% extends %}`, `{% block %}`, `{% filter %}`,
//! block `{% set %}`, `recursive` loops, `**`, float literals, `selectattr` and
//! friends, list `append`, and any other filter, test, method or function. A
//! template that renders *almost* right is a prompt the model was never trained
//! on, so silence is not an option.
//!
//! `strftime_now` is deliberately not defined. The Llama-3 templates probe it
//! with `{% if strftime_now is defined %}` and fall back to a fixed date, which
//! keeps rendering deterministic instead of baking the wall clock into a prompt.
//!
//! `raise_exception(msg)` maps to [`TemplateError::Raised`], so a conversation
//! a template refuses to format (Mistral's alternating-roles rule, Gemma's "no
//! system role") surfaces as an `Err` rather than a malformed prompt.

mod builtins;
mod chat;
mod eval;
mod lex;
mod parse;
mod value;

pub use chat::{render_chat_template, ChatMessage, ChatOptions};
pub use value::Value;

/// Why a chat template could not be parsed or rendered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TemplateError {
    /// Malformed template source.
    Syntax(String),
    /// A Jinja construct outside the supported subset, named explicitly.
    Unsupported(String),
    /// The template is valid but could not be evaluated against these variables.
    Runtime(String),
    /// The template called `raise_exception(...)`.
    Raised(String),
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Syntax(m) => write!(f, "chat template syntax error: {m}"),
            Self::Unsupported(m) => write!(f, "chat template uses an unsupported construct: {m}"),
            Self::Runtime(m) => write!(f, "chat template failed: {m}"),
            Self::Raised(m) => write!(f, "chat template rejected these messages: {m}"),
        }
    }
}

impl std::error::Error for TemplateError {}

/// A parsed chat template, reusable across renders.
#[derive(Clone, Debug)]
pub struct Template {
    body: Vec<parse::Node>,
}

impl Template {
    /// Parse template source.
    pub fn parse(src: &str) -> Result<Self, TemplateError> {
        Ok(Self {
            body: parse::parse(src)?,
        })
    }

    /// Render with `vars` bound in the global scope.
    pub fn render(&self, vars: Vec<(String, Value)>) -> Result<String, TemplateError> {
        let mut env = eval::Env::new(vars);
        eval::render(&self.body, &mut env)
    }
}

#[cfg(test)]
mod tests {
    use super::{Template, TemplateError, Value};

    fn render(src: &str) -> Result<String, TemplateError> {
        Template::parse(src)?.render(Vec::new())
    }

    #[test]
    fn errors_name_the_construct_they_reject() {
        let err = render("{% macro x() %}{% endmacro %}").expect_err("macro");
        assert!(matches!(err, TemplateError::Unsupported(_)));
        assert!(err.to_string().contains("macro"), "{err}");
        let err = render("{{ 'x' | selectattr('a') }}").expect_err("filter");
        assert!(err.to_string().contains("selectattr"), "{err}");
        let err = render("{{ 'x' is sameas 'y' }}").expect_err("test");
        assert!(err.to_string().contains("sameas"), "{err}");
        let err = render("{{ oops() }}").expect_err("function");
        assert!(err.to_string().contains("oops"), "{err}");
    }

    #[test]
    fn a_parsed_template_can_be_rendered_more_than_once() {
        let t = Template::parse("{{ name }}!").expect("parse");
        assert_eq!(
            t.render(vec![("name".into(), Value::str("a"))]).unwrap(),
            "a!"
        );
        assert_eq!(
            t.render(vec![("name".into(), Value::str("b"))]).unwrap(),
            "b!"
        );
    }
}
