//! Std-only argv for `gguf_gemv`. No clap. Defaults keep `infer <path>` on `ab` / 2.

/// Historical harness prompt. `infer <path>` with no flags still uses this.
pub const DEFAULT_INFER_PROMPT: &str = "ab";

/// Historical harness predict count. `infer <path>` with no flags still uses this.
pub const DEFAULT_N_PREDICT: usize = 2;

/// Parsed `infer` flags. Prompt text is owned so the binary can print it back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferArgs {
    /// GGUF path (positional).
    pub path: String,
    /// Prompt text (`--prompt` / `-p`). Default [`DEFAULT_INFER_PROMPT`].
    pub prompt: String,
    /// Tokens to generate (`--n-predict` / `-n`). Default [`DEFAULT_N_PREDICT`].
    pub n_predict: usize,
    /// Optional KV length override (`--max-seq`). `None` sizes to prompt + predict + 1.
    pub max_seq: Option<usize>,
}

/// Top-level command after `argv[0]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cmd {
    /// Print [`usage`] to stdout.
    Help,
    /// Write the Q8_0 GEMV demo GGUF.
    Write {
        /// Output path.
        path: String,
    },
    /// Q8_0 GEMV measurement on a GGUF written by `write`.
    Gemv {
        /// Input path.
        path: String,
    },
    /// Write the Q4_K GEMV demo GGUF.
    WriteQ4k {
        /// Output path.
        path: String,
    },
    /// Q4_K GEMV measurement on a GGUF written by `write-q4k`.
    GemvQ4k {
        /// Input path.
        path: String,
    },
    /// Write the tiny Llama-shaped GGUF used by decode tests.
    WriteTiny {
        /// Output path.
        path: String,
    },
    /// Write the tiny Qwen2-shaped GGUF used by decode tests.
    WriteTinyQwen2 {
        /// Output path.
        path: String,
    },
    /// Greedy decode.
    Infer(InferArgs),
}

/// Flag / operand failure. Missing command prints help instead of this.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliError {
    /// Bad flags or missing operands. Carries the reason.
    Usage(String),
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Usage(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for CliError {}

/// Usage text for stdout (`help`) and stderr (parse failure).
#[must_use]
pub fn usage() -> &'static str {
    "\
usage: gguf_gemv <command> [args]

commands:
  infer <path> [--prompt TEXT] [--n-predict N] [--max-seq N]
      greedy decode (seedless). defaults: --prompt ab --n-predict 2
      --max-seq overrides KV length; default is prompt tokens + N + 1
  write <path>              Q8_0 GEMV demo GGUF
  gemv <path>               Q8_0 GEMV measurement
  write-q4k <path>          Q4_K GEMV demo GGUF
  gemv-q4k <path>           Q4_K GEMV measurement
  write-tiny <path>         tiny Llama-shaped GGUF
  write-tiny-qwen2 <path>   tiny Qwen2-shaped GGUF
  help                      this text

infer flags: --prompt/-p, --n-predict/-n, --max-seq; --flag=value is ok
"
}

/// Parse argv after the binary name.
///
/// No operands → [`Cmd::Help`]. `help` / `--help` / `-h` are help at any
/// command position that is not a flag value.
pub fn parse_args<I, S>(args: I) -> Result<Cmd, CliError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut it = args.into_iter();
    let Some(cmd) = it.next() else {
        return Ok(Cmd::Help);
    };
    match cmd.as_ref() {
        "help" | "--help" | "-h" => Ok(Cmd::Help),
        "infer" => parse_infer(&mut it),
        "write" => one_path("write", &mut it).map(|path| Cmd::Write { path }),
        "gemv" => one_path("gemv", &mut it).map(|path| Cmd::Gemv { path }),
        "write-q4k" => one_path("write-q4k", &mut it).map(|path| Cmd::WriteQ4k { path }),
        "gemv-q4k" => one_path("gemv-q4k", &mut it).map(|path| Cmd::GemvQ4k { path }),
        "write-tiny" => one_path("write-tiny", &mut it).map(|path| Cmd::WriteTiny { path }),
        "write-tiny-qwen2" => {
            one_path("write-tiny-qwen2", &mut it).map(|path| Cmd::WriteTinyQwen2 { path })
        }
        other => Err(CliError::Usage(format!(
            "unknown command {other}. try gguf_gemv help"
        ))),
    }
}

fn parse_infer<I, S>(it: &mut I) -> Result<Cmd, CliError>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    let mut path = None;
    let mut prompt = None;
    let mut n_predict = None;
    let mut max_seq = None;
    while let Some(raw) = it.next() {
        let a = raw.as_ref();
        if a == "--help" || a == "-h" {
            return Ok(Cmd::Help);
        }
        if take_opt(a, it, "--prompt", Some("-p"), &mut prompt)? {
            continue;
        }
        if take_usize(a, it, "--n-predict", Some("-n"), &mut n_predict)? {
            continue;
        }
        if take_usize(a, it, "--max-seq", None, &mut max_seq)? {
            continue;
        }
        if a.starts_with('-') {
            return Err(CliError::Usage(format!("unknown infer flag {a}")));
        }
        set_once(&mut path, "path", a.to_string())?;
    }
    let path = path.ok_or_else(|| CliError::Usage("infer needs a GGUF path".into()))?;
    Ok(Cmd::Infer(InferArgs {
        path,
        prompt: prompt.unwrap_or_else(|| DEFAULT_INFER_PROMPT.into()),
        n_predict: n_predict.unwrap_or(DEFAULT_N_PREDICT),
        max_seq,
    }))
}

fn one_path<I, S>(cmd: &str, it: &mut I) -> Result<String, CliError>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    let Some(path) = it.next() else {
        return Err(CliError::Usage(format!("{cmd} <path>")));
    };
    if let Some(extra) = it.next() {
        return Err(CliError::Usage(format!(
            "{cmd} got extra argument {}",
            extra.as_ref()
        )));
    }
    Ok(path.as_ref().to_string())
}

fn take_opt<I, S>(
    a: &str,
    it: &mut I,
    long: &str,
    short: Option<&str>,
    slot: &mut Option<String>,
) -> Result<bool, CliError>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    if let Some(v) = strip_long_eq(a, long) {
        set_once(slot, long, v.to_string())?;
        return Ok(true);
    }
    if a == long || short == Some(a) {
        set_once(slot, long, take_val(it, long)?)?;
        return Ok(true);
    }
    Ok(false)
}

fn take_usize<I, S>(
    a: &str,
    it: &mut I,
    long: &str,
    short: Option<&str>,
    slot: &mut Option<usize>,
) -> Result<bool, CliError>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    if let Some(v) = strip_long_eq(a, long) {
        set_once(slot, long, parse_usize(long, v)?)?;
        return Ok(true);
    }
    if a == long || short == Some(a) {
        set_once(slot, long, parse_usize(long, &take_val(it, long)?)?)?;
        return Ok(true);
    }
    Ok(false)
}

fn strip_long_eq<'a>(a: &'a str, long: &str) -> Option<&'a str> {
    let rest = a.strip_prefix(long)?;
    rest.strip_prefix('=')
}

fn take_val<I, S>(it: &mut I, flag: &str) -> Result<String, CliError>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    let Some(v) = it.next() else {
        return Err(CliError::Usage(format!("{flag} needs a value")));
    };
    Ok(v.as_ref().to_string())
}

fn parse_usize(flag: &str, raw: &str) -> Result<usize, CliError> {
    raw.parse::<usize>()
        .map_err(|_| CliError::Usage(format!("{flag} needs a non-negative integer, got {raw:?}")))
}

fn set_once<T>(slot: &mut Option<T>, name: &str, value: T) -> Result<(), CliError> {
    if slot.is_some() {
        return Err(CliError::Usage(format!("duplicate {name}")));
    }
    *slot = Some(value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        parse_args, usage, CliError, Cmd, InferArgs, DEFAULT_INFER_PROMPT, DEFAULT_N_PREDICT,
    };

    fn p<const N: usize>(args: [&str; N]) -> Result<Cmd, CliError> {
        parse_args(args)
    }

    fn infer(args: &[&str]) -> InferArgs {
        match parse_args(args.iter().copied()).expect("parse") {
            Cmd::Infer(i) => i,
            other => panic!("expected infer, got {other:?}"),
        }
    }

    #[test]
    fn no_args_and_help_tokens_are_help() {
        assert_eq!(p([]).unwrap(), Cmd::Help);
        assert_eq!(p(["help"]).unwrap(), Cmd::Help);
        assert_eq!(p(["--help"]).unwrap(), Cmd::Help);
        assert_eq!(p(["-h"]).unwrap(), Cmd::Help);
        assert_eq!(p(["infer", "--help"]).unwrap(), Cmd::Help);
        assert_eq!(p(["infer", "-h", "x.gguf"]).unwrap(), Cmd::Help);
    }

    #[test]
    fn infer_path_only_keeps_harness_defaults() {
        assert_eq!(
            infer(&["infer", "m.gguf"]),
            InferArgs {
                path: "m.gguf".into(),
                prompt: DEFAULT_INFER_PROMPT.into(),
                n_predict: DEFAULT_N_PREDICT,
                max_seq: None,
            }
        );
    }

    #[test]
    fn infer_flags_any_order_and_eq_form() {
        let want = InferArgs {
            path: "m.gguf".into(),
            prompt: "hello".into(),
            n_predict: 8,
            max_seq: Some(64),
        };
        assert_eq!(
            infer(&[
                "infer",
                "m.gguf",
                "--prompt",
                "hello",
                "--n-predict",
                "8",
                "--max-seq",
                "64"
            ]),
            want
        );
        assert_eq!(
            infer(&[
                "infer",
                "--prompt=hello",
                "--n-predict=8",
                "--max-seq=64",
                "m.gguf"
            ]),
            want
        );
        assert_eq!(
            infer(&[
                "infer",
                "-p",
                "hello",
                "-n",
                "8",
                "--max-seq",
                "64",
                "m.gguf"
            ]),
            want
        );
    }

    #[test]
    fn infer_zero_predict_is_allowed() {
        let i = infer(&["infer", "m.gguf", "-n", "0"]);
        assert_eq!(i.n_predict, 0);
    }

    #[test]
    fn infer_rejects_missing_path_unknown_flag_and_bad_int() {
        assert!(matches!(p(["infer"]), Err(CliError::Usage(_))));
        assert!(matches!(
            p(["infer", "m.gguf", "--temp", "1"]),
            Err(CliError::Usage(s)) if s.contains("--temp")
        ));
        assert!(matches!(
            p(["infer", "m.gguf", "--n-predict", "x"]),
            Err(CliError::Usage(s)) if s.contains("--n-predict") && s.contains("x")
        ));
        assert!(matches!(
            p(["infer", "m.gguf", "--prompt"]),
            Err(CliError::Usage(s)) if s.contains("--prompt")
        ));
        assert!(matches!(
            p(["infer", "a.gguf", "b.gguf"]),
            Err(CliError::Usage(_))
        ));
        assert!(matches!(
            p(["infer", "m.gguf", "-p", "a", "--prompt", "b"]),
            Err(CliError::Usage(s)) if s.contains("duplicate")
        ));
    }

    #[test]
    fn path_commands_and_unknown() {
        assert_eq!(
            p(["write", "o.gguf"]).unwrap(),
            Cmd::Write {
                path: "o.gguf".into()
            }
        );
        assert_eq!(
            p(["write-tiny-qwen2", "q.gguf"]).unwrap(),
            Cmd::WriteTinyQwen2 {
                path: "q.gguf".into()
            }
        );
        assert!(matches!(p(["gemv"]), Err(CliError::Usage(s)) if s.contains("gemv")));
        assert!(matches!(
            p(["gemv", "a", "b"]),
            Err(CliError::Usage(s)) if s.contains("extra")
        ));
        assert!(matches!(
            p(["serve"]),
            Err(CliError::Usage(s)) if s.contains("serve")
        ));
    }

    #[test]
    fn usage_names_infer_flags_and_defaults() {
        let u = usage();
        assert!(u.contains("--prompt"), "{u}");
        assert!(u.contains("--n-predict"), "{u}");
        assert!(u.contains("--max-seq"), "{u}");
        assert!(u.contains("ab"), "{u}");
        assert!(u.contains('2'), "{u}");
        assert!(u.contains("greedy"), "{u}");
    }
}
