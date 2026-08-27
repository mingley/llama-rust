//! argv for `gguf_gemv infer`. No clap.

use std::fmt;

/// Failure while parsing infer argv.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliError {
    /// Missing path, unknown flag, or unparsable `--n-predict`.
    Usage(String),
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for CliError {}

/// `infer` path, prompt, and `n_predict`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InferArgs {
    /// GGUF file path.
    pub path: String,
    /// Prompt to encode.
    pub prompt: String,
    /// Tokens to sample after the prompt.
    pub n_predict: usize,
}

impl InferArgs {
    /// Prompt used when `--prompt` / `-p` is omitted.
    pub const DEFAULT_PROMPT: &'static str = "ab";
    /// `n_predict` used when `--n-predict` / `-n` is omitted.
    pub const DEFAULT_N_PREDICT: usize = 2;
}

/// Usage line for `infer`.
pub const INFER_USAGE: &str = "infer <path> [--prompt TEXT] [--n-predict N]";

/// Parse argv after the `infer` verb.
pub fn parse_infer<I>(args: I) -> Result<InferArgs, CliError>
where
    I: IntoIterator,
    I::Item: AsRef<str>,
{
    let mut path = None;
    let mut prompt = None;
    let mut n_predict = None;
    let mut it = args.into_iter();
    while let Some(item) = it.next() {
        let a = item.as_ref();
        if a == "--help" || a == "-h" {
            return Err(CliError::Usage(INFER_USAGE.into()));
        }
        if let Some(v) = strip_eq(a, "--prompt") {
            set_once(&mut prompt, "--prompt", v.to_string())?;
            continue;
        }
        if a == "--prompt" || a == "-p" {
            let v = next_val(&mut it, a)?;
            set_once(&mut prompt, a, v)?;
            continue;
        }
        if let Some(v) = strip_eq(a, "-p") {
            set_once(&mut prompt, "-p", v.to_string())?;
            continue;
        }
        if let Some(v) = strip_eq(a, "--n-predict") {
            set_once(&mut n_predict, "--n-predict", parse_n_predict(v)?)?;
            continue;
        }
        if a == "--n-predict" || a == "-n" {
            let v = next_val(&mut it, a)?;
            set_once(&mut n_predict, a, parse_n_predict(&v)?)?;
            continue;
        }
        if let Some(v) = strip_eq(a, "-n") {
            set_once(&mut n_predict, "-n", parse_n_predict(v)?)?;
            continue;
        }
        if a.starts_with('-') {
            return Err(CliError::Usage(format!("unknown flag {a}")));
        }
        if path.is_some() {
            return Err(CliError::Usage(format!(
                "unexpected argument {a} ({INFER_USAGE})"
            )));
        }
        path = Some(a.to_string());
    }
    let path = path.ok_or_else(|| CliError::Usage(INFER_USAGE.into()))?;
    Ok(InferArgs {
        path,
        prompt: prompt.unwrap_or_else(|| InferArgs::DEFAULT_PROMPT.to_string()),
        n_predict: n_predict.unwrap_or(InferArgs::DEFAULT_N_PREDICT),
    })
}

fn strip_eq<'a>(arg: &'a str, flag: &str) -> Option<&'a str> {
    arg.strip_prefix(flag)?.strip_prefix('=')
}

fn next_val<I, S>(it: &mut I, flag: &str) -> Result<String, CliError>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    match it.next() {
        Some(v) => Ok(v.as_ref().to_string()),
        None => Err(CliError::Usage(format!("{flag} needs a value"))),
    }
}

fn set_once<T>(slot: &mut Option<T>, flag: &str, val: T) -> Result<(), CliError> {
    if slot.is_some() {
        return Err(CliError::Usage(format!("duplicate {flag}")));
    }
    *slot = Some(val);
    Ok(())
}

fn parse_n_predict(s: &str) -> Result<usize, CliError> {
    s.parse::<usize>().map_err(|_| {
        CliError::Usage(format!(
            "n_predict must be a non-negative integer (got {s})"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::greedy_generate;
    use crate::gguf::load_gguf;
    use crate::tok::Tokenizer;
    use crate::{tiny_llama_gguf, tiny_qwen2_gguf, Llama};

    fn must_err(args: &[&str]) -> String {
        match parse_infer(args) {
            Ok(v) => panic!("expected error, got {v:?}"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn infer_defaults_match_previous_harness() {
        let a = parse_infer(["model.gguf"]).expect("path only");
        assert_eq!(
            a,
            InferArgs {
                path: "model.gguf".into(),
                prompt: InferArgs::DEFAULT_PROMPT.into(),
                n_predict: InferArgs::DEFAULT_N_PREDICT,
            }
        );
        assert_eq!(a.prompt, "ab");
        assert_eq!(a.n_predict, 2);
    }

    #[test]
    fn infer_flags_before_or_after_path() {
        let a = parse_infer(["--prompt", "a", "--n-predict", "4", "w.gguf"]).expect("flags first");
        assert_eq!(a.path, "w.gguf");
        assert_eq!(a.prompt, "a");
        assert_eq!(a.n_predict, 4);
        let b = parse_infer(["w.gguf", "-p", "b", "-n", "0"]).expect("flags last");
        assert_eq!(b.prompt, "b");
        assert_eq!(b.n_predict, 0);
        let c = parse_infer(["w.gguf", "--prompt=ab", "--n-predict=8"]).expect("equals");
        assert_eq!(c.prompt, "ab");
        assert_eq!(c.n_predict, 8);
        let d = parse_infer(["w.gguf", "-p=a", "-n=1"]).expect("short equals");
        assert_eq!(d.prompt, "a");
        assert_eq!(d.n_predict, 1);
    }

    #[test]
    fn infer_parse_rejects_bad_argv() {
        assert!(must_err(&[]).contains("infer <path>"));
        assert!(must_err(&["--help"]).contains("infer <path>"));
        assert!(must_err(&["m.gguf", "--seed", "1"]).contains("unknown flag"));
        assert!(must_err(&["m.gguf", "extra"]).contains("unexpected"));
        assert!(must_err(&["m.gguf", "--n-predict", "-1"]).contains("n_predict"));
        assert!(must_err(&["m.gguf", "--n-predict"]).contains("needs a value"));
        assert!(must_err(&["m.gguf", "--prompt"]).contains("needs a value"));
        assert!(must_err(&["m.gguf", "-p", "a", "--prompt", "b"]).contains("duplicate"));
        assert!(must_err(&["m.gguf", "-n", "1", "--n-predict", "2"]).contains("duplicate"));
    }

    #[test]
    fn infer_args_drive_tiny_greedy() {
        let args = parse_infer(["tiny.gguf", "-p", "a", "-n", "1"]).expect("parse");
        let bytes = tiny_llama_gguf();
        let g = load_gguf(&bytes).expect("load");
        let model = Llama::from_gguf(&g).expect("model");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let just_prompt = greedy_generate(&model, &tok, &args.prompt, 0).expect("n=0");
        assert_eq!(just_prompt, "a");
        let out = greedy_generate(&model, &tok, &args.prompt, args.n_predict).expect("n=1");
        let out2 = greedy_generate(&model, &tok, &args.prompt, args.n_predict).expect("n=1 again");
        assert_eq!(out, out2);
        assert!(out.contains('a'), "{out}");
        let q = tiny_qwen2_gguf();
        let gq = load_gguf(&q).expect("qwen2");
        let mq = Llama::from_gguf(&gq).expect("qwen2 model");
        let tq = Tokenizer::from_gguf(&gq).expect("qwen2 tok");
        let q0 = greedy_generate(&mq, &tq, "ab", 0).expect("qwen2 n=0");
        assert_eq!(q0, "ab");
    }
}
