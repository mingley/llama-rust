//! `gguf_gemv infer` argument parsing. No crates.io CLI crate.

/// Usage for the `infer` verb.
pub const INFER_USAGE: &str = "\
usage: gguf_gemv infer <path> [--prompt TEXT] [--n-predict N] [--n-ctx N]
  -p, --prompt TEXT   prompt (default: ab)
  -n, --n-predict N   tokens to generate (default: 2)
      --n-ctx N       KV capacity (default: prompt + n_predict + 1)
";

/// Top-level binary usage.
pub const BIN_USAGE: &str = "\
usage: gguf_gemv <command> [args]
  infer <path> [--prompt TEXT] [--n-predict N] [--n-ctx N]
  serve <path> [--n-predict N] [--n-ctx N] [--bind HOST:PORT]
  write|gemv|write-q4k|gemv-q4k|write-tiny|write-tiny-qwen2|write-tiny-qwen3|write-tiny-gemma|write-tiny-llama4|write-tiny-llama-moe <path>
";

/// Parsed `infer` invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferCmd {
    /// `--help` / `-h`.
    Help,
    /// Run greedy decode with these arguments.
    Run(InferArgs),
}

/// Arguments for seedless greedy `infer`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferArgs {
    /// GGUF path.
    pub path: String,
    /// Prompt text.
    pub prompt: String,
    /// Tokens to generate after the prompt.
    pub n_predict: usize,
    /// Optional KV capacity. `None` sizes to prompt + `n_predict` + 1.
    pub n_ctx: Option<usize>,
}

impl InferArgs {
    /// Default prompt when `--prompt` is omitted.
    pub const DEFAULT_PROMPT: &'static str = "ab";
    /// Default `--n-predict` when omitted.
    pub const DEFAULT_N_PREDICT: usize = 2;
}

/// Parse operands after the `infer` verb.
///
/// `infer <path> [--prompt TEXT] [--n-predict N] [--n-ctx N]`
/// Path may appear before or after flags. `--flag=value` is accepted.
pub fn parse_infer_args<I, S>(args: I) -> Result<InferCmd, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut path = None;
    let mut prompt = InferArgs::DEFAULT_PROMPT.to_string();
    let mut n_predict = InferArgs::DEFAULT_N_PREDICT;
    let mut n_ctx = None;
    let mut it = args.into_iter();
    while let Some(raw) = it.next() {
        let arg = raw.as_ref();
        if arg == "--help" || arg == "-h" {
            return Ok(InferCmd::Help);
        }
        let (key, inline) = match arg.split_once('=') {
            Some((k, v)) => (k, Some(v)),
            None => (arg, None),
        };
        match key {
            "--prompt" | "-p" => {
                prompt = opt_value("prompt", inline, &mut it)?;
            }
            "--n-predict" | "-n" => {
                n_predict = parse_usize("n-predict", &opt_value("n-predict", inline, &mut it)?)?;
            }
            "--n-ctx" => {
                let n = parse_usize("n-ctx", &opt_value("n-ctx", inline, &mut it)?)?;
                if n == 0 {
                    return usage_err("n-ctx must be > 0");
                }
                n_ctx = Some(n);
            }
            flag if flag.starts_with('-') => {
                return usage_err(&format!("unknown flag {flag}"));
            }
            other => {
                if path.is_some() {
                    return usage_err(&format!("unexpected argument {other}"));
                }
                path = Some(other.to_string());
            }
        }
    }
    let Some(path) = path else {
        return usage_err("missing GGUF path");
    };
    Ok(InferCmd::Run(InferArgs {
        path,
        prompt,
        n_predict,
        n_ctx,
    }))
}

fn usage_err<T>(msg: &str) -> Result<T, String> {
    Err(format!("{msg}\n{INFER_USAGE}"))
}

fn opt_value<I, S>(name: &str, inline: Option<&str>, it: &mut I) -> Result<String, String>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    if let Some(v) = inline {
        return Ok(v.to_string());
    }
    match it.next() {
        Some(s) => Ok(s.as_ref().to_string()),
        None => usage_err(&format!("missing --{name} value")),
    }
}

fn parse_usize(name: &str, s: &str) -> Result<usize, String> {
    s.parse::<usize>()
        .map_err(|_| format!("invalid {name} {s:?}\n{INFER_USAGE}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(args: &[&str]) -> InferArgs {
        match parse_infer_args(args).expect("parse") {
            InferCmd::Run(a) => a,
            InferCmd::Help => panic!("expected Run"),
        }
    }

    #[test]
    fn omitted_flags_keep_shipped_ab_and_two() {
        let a = run(&["tiny.gguf"]);
        assert_eq!(a.path, "tiny.gguf");
        assert_eq!(a.prompt, InferArgs::DEFAULT_PROMPT);
        assert_eq!(a.n_predict, InferArgs::DEFAULT_N_PREDICT);
        assert_eq!(a.n_ctx, None);
        assert_eq!(a.prompt, "ab");
        assert_eq!(a.n_predict, 2);
    }

    #[test]
    fn long_flags_and_path_after() {
        let a = run(&[
            "--prompt",
            "a",
            "--n-predict",
            "4",
            "--n-ctx",
            "16",
            "m.gguf",
        ]);
        assert_eq!(
            a,
            InferArgs {
                path: "m.gguf".into(),
                prompt: "a".into(),
                n_predict: 4,
                n_ctx: Some(16),
            }
        );
    }

    #[test]
    fn short_flags_equals_and_path_first() {
        let a = run(&["model.gguf", "-p=hello", "-n=0", "--n-ctx=8"]);
        assert_eq!(a.path, "model.gguf");
        assert_eq!(a.prompt, "hello");
        assert_eq!(a.n_predict, 0);
        assert_eq!(a.n_ctx, Some(8));
    }

    #[test]
    fn help_is_not_a_run() {
        assert_eq!(parse_infer_args(["--help"]).unwrap(), InferCmd::Help);
        assert_eq!(parse_infer_args(["-h", "x.gguf"]).unwrap(), InferCmd::Help);
    }

    #[test]
    fn missing_path_and_bad_values_error() {
        let err = parse_infer_args(["--prompt", "hi"]).unwrap_err();
        assert!(err.contains("missing GGUF path"), "{err}");
        assert!(err.contains("gguf_gemv infer"), "{err}");
        let err = parse_infer_args(["m.gguf", "-n", "x"]).unwrap_err();
        assert!(err.contains("invalid n-predict"), "{err}");
        let err = parse_infer_args(["m.gguf", "--n-ctx", "0"]).unwrap_err();
        assert!(err.contains("n-ctx must be > 0"), "{err}");
        let err = parse_infer_args(["m.gguf", "--nope"]).unwrap_err();
        assert!(err.contains("unknown flag"), "{err}");
        let err = parse_infer_args(["a.gguf", "b.gguf"]).unwrap_err();
        assert!(err.contains("unexpected argument"), "{err}");
        let err = parse_infer_args(["--prompt"]).unwrap_err();
        assert!(err.contains("missing --prompt value"), "{err}");
    }
}
