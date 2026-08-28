//! `gguf_gemv infer` / `chat` argument parsing, and the `chat` conversation
//! loop. No crates.io CLI crate.

use std::fs::File;
use std::io::{BufRead, Read, Write};
use std::path::Path;

use crate::decode::Llama;
use crate::gguf::load_gguf_owned;
use crate::sample::argmax;
use crate::template::ChatMessage;
use crate::tok::Tokenizer;

/// Usage for the `infer` verb.
pub const INFER_USAGE: &str = "\
usage: gguf_gemv infer <path> [--prompt TEXT] [--n-predict N] [--n-ctx N]
  -p, --prompt TEXT   prompt (default: ab)
  -n, --n-predict N   tokens to generate (default: 2)
      --n-ctx N       KV capacity (default: prompt + n_predict + 1)
";

/// Usage for the `chat` verb.
pub const CHAT_USAGE: &str = "\
usage: gguf_gemv chat <path> [--system TEXT] [--prompt TEXT] [--n-predict N] [--n-ctx N] [--show-prompt]
  -s, --system TEXT   system message placed before the conversation
  -p, --prompt TEXT   send one user turn and exit; omit to read turns from stdin
  -n, --n-predict N   tokens to generate per reply (default: 64)
      --n-ctx N       KV capacity (default: prompt + n_predict + 1)
      --show-prompt   print the rendered chat template before each reply
";

/// Top-level binary usage.
pub const BIN_USAGE: &str = "\
usage: gguf_gemv <command> [args]
  infer <path> [--prompt TEXT] [--n-predict N] [--n-ctx N]
  chat <path> [--system TEXT] [--prompt TEXT] [--n-predict N] [--n-ctx N] [--show-prompt]
  serve <path> [--n-predict N] [--n-ctx N] [--bind HOST:PORT]
  write|gemv|write-q4k|gemv-q4k|write-tiny|write-tiny-qwen2|write-tiny-qwen3|write-tiny-gemma|write-tiny-llama4|write-tiny-llama-moe|write-tiny-qwen2moe|write-tiny-qwen3moe|write-tiny-qwen2vl|write-tiny-qwen3vl|write-tiny-qwen3next|write-tiny-qwen35|write-tiny-phi2 <path>
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

/// Parsed `chat` invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatCmd {
    /// `--help` / `-h`.
    Help,
    /// Hold a conversation with these arguments.
    Run(ChatArgs),
}

/// Arguments for `chat`, which renders the model's own `tokenizer.chat_template`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatArgs {
    /// GGUF path.
    pub path: String,
    /// Optional `system` message placed before the conversation.
    pub system: Option<String>,
    /// One user turn. `None` reads turns from stdin until EOF.
    pub prompt: Option<String>,
    /// Tokens to generate per reply.
    pub n_predict: usize,
    /// Optional KV capacity. `None` sizes to prompt + `n_predict` + 1.
    pub n_ctx: Option<usize>,
    /// Print the rendered template before each reply.
    pub show_prompt: bool,
}

impl ChatArgs {
    /// Default `--n-predict` when omitted. Larger than `infer`'s, because a
    /// chat reply that stops after two tokens tells you nothing.
    pub const DEFAULT_N_PREDICT: usize = 64;
}

/// Parse operands after the `chat` verb.
///
/// `chat <path> [--system TEXT] [--prompt TEXT] [--n-predict N] [--n-ctx N] [--show-prompt]`
/// Path may appear before or after flags. `--flag=value` is accepted.
pub fn parse_chat_args<I, S>(args: I) -> Result<ChatCmd, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut path = None;
    let mut system = None;
    let mut prompt = None;
    let mut n_predict = ChatArgs::DEFAULT_N_PREDICT;
    let mut n_ctx = None;
    let mut show_prompt = false;
    let mut it = args.into_iter();
    while let Some(raw) = it.next() {
        let arg = raw.as_ref();
        if arg == "--help" || arg == "-h" {
            return Ok(ChatCmd::Help);
        }
        let (key, inline) = match arg.split_once('=') {
            Some((k, v)) => (k, Some(v)),
            None => (arg, None),
        };
        match key {
            "--system" | "-s" => system = Some(chat_value("system", inline, &mut it)?),
            "--prompt" | "-p" => prompt = Some(chat_value("prompt", inline, &mut it)?),
            "--n-predict" | "-n" => {
                let v = chat_value("n-predict", inline, &mut it)?;
                n_predict = v
                    .parse::<usize>()
                    .map_err(|_| format!("invalid n-predict {v:?}\n{CHAT_USAGE}"))?;
            }
            "--n-ctx" => {
                let v = chat_value("n-ctx", inline, &mut it)?;
                let n = v
                    .parse::<usize>()
                    .map_err(|_| format!("invalid n-ctx {v:?}\n{CHAT_USAGE}"))?;
                if n == 0 {
                    return chat_usage_err("n-ctx must be > 0");
                }
                n_ctx = Some(n);
            }
            "--show-prompt" => show_prompt = true,
            flag if flag.starts_with('-') => {
                return chat_usage_err(&format!("unknown flag {flag}"));
            }
            other => {
                if path.is_some() {
                    return chat_usage_err(&format!("unexpected argument {other}"));
                }
                path = Some(other.to_string());
            }
        }
    }
    let Some(path) = path else {
        return chat_usage_err("missing GGUF path");
    };
    Ok(ChatCmd::Run(ChatArgs {
        path,
        system,
        prompt,
        n_predict,
        n_ctx,
        show_prompt,
    }))
}

/// Hold a conversation using the model's own `tokenizer.chat_template`.
///
/// With `--prompt` this is one turn and exits. Otherwise each line of stdin is
/// a user turn, and replies accumulate so the model sees the whole history.
/// Every turn re-renders the template and re-prefills from scratch, which is
/// the honest thing to do while there is no prefix cache: templates are free to
/// rewrite earlier turns, so a KV cache keyed on turn count would be wrong.
pub fn run_chat(args: &ChatArgs) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = read_file(Path::new(&args.path))?;
    let g = load_gguf_owned(bytes)?;
    let tok = Tokenizer::from_gguf(&g)?;
    if tok.chat_template.is_none() {
        return Err(format!(
            "{} has no tokenizer.chat_template; use `gguf_gemv infer` with a raw prompt",
            args.path
        )
        .into());
    }
    let model = Llama::from_gguf(g)?;
    let mut history: Vec<ChatMessage> = Vec::new();
    if let Some(s) = &args.system {
        history.push(ChatMessage::system(s));
    }
    let mut out = std::io::stdout();
    if let Some(p) = &args.prompt {
        return chat_turn(&model, &tok, args, &mut history, p, &mut out);
    }
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        write!(out, "> ")?;
        out.flush()?;
        line.clear();
        if stdin.lock().read_line(&mut line)? == 0 {
            writeln!(out)?;
            return Ok(());
        }
        let turn = line.trim_end_matches(['\n', '\r']);
        if turn.is_empty() {
            continue;
        }
        chat_turn(&model, &tok, args, &mut history, turn, &mut out)?;
    }
}

fn chat_turn<W: Write>(
    model: &Llama,
    tok: &Tokenizer,
    args: &ChatArgs,
    history: &mut Vec<ChatMessage>,
    turn: &str,
    out: &mut W,
) -> Result<(), Box<dyn std::error::Error>> {
    history.push(ChatMessage::user(turn));
    let prompt = tok.apply_chat_template(history, true)?;
    if args.show_prompt {
        writeln!(out, "--- rendered prompt ---\n{prompt}\n--- end ---")?;
    }
    let reply = generate_reply(model, tok, &prompt, args)?;
    writeln!(out, "{}", reply.trim())?;
    history.push(ChatMessage::assistant(reply.trim()));
    Ok(())
}

/// Greedy-decode one assistant turn and return only the new text.
///
/// [`greedy_generate_ctx`] decodes the prompt and the continuation together,
/// which is what `infer` wants but not what a conversation wants: the reply has
/// to go into the history on its own, and the prompt is full of markers
/// [`Tokenizer::decode`] drops.
fn generate_reply(
    model: &Llama,
    tok: &Tokenizer,
    prompt: &str,
    args: &ChatArgs,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut ids = tok.encode(prompt)?;
    if tok.add_bos {
        if let Some(bos) = tok.bos {
            if ids.first().copied() != Some(bos) {
                ids.insert(0, bos);
            }
        }
    }
    if ids.is_empty() {
        return Err("the chat template rendered an empty prompt".into());
    }
    let needed = ids.len().saturating_add(args.n_predict);
    let max_seq = match args.n_ctx {
        Some(n) if n < needed => {
            return Err(format!("--n-ctx {n} is below the {needed} tokens this turn needs").into())
        }
        Some(n) => n,
        None => needed.saturating_add(1),
    };
    let mut cache = model.new_cache(max_seq)?;
    let mut logits = model.prefill(&mut cache, &ids)?;
    let mut reply = Vec::new();
    for _ in 0..args.n_predict {
        let next = argmax(&logits);
        if ends_turn(tok, next) {
            break;
        }
        reply.push(next);
        logits = model.forward(&mut cache, next)?;
    }
    Ok(tok.decode(&reply))
}

/// Whether `id` ends the assistant's turn.
///
/// `eos` alone is not enough. An instruct model closes its turn with whatever
/// marker its own template uses (`<|im_end|>`, `<|eot_id|>`, `<end_of_turn>`),
/// and in several families that is a different id from `eos`. llama.cpp keeps
/// an explicit end-of-generation set for this; GGUF does not always carry the
/// flags, so we use the rule that holds regardless: a control token in the
/// output stream means the reply is over, because control tokens never belong
/// inside one.
fn ends_turn(tok: &Tokenizer, id: u32) -> bool {
    tok.eos == Some(id) || tok.is_special(id)
}

/// Read a file with `File` + `Read`, not `std::fs::read`.
fn read_file(path: &Path) -> Result<Vec<u8>, std::io::Error> {
    let mut f = File::open(path)?;
    let mut buf = Vec::new();
    let _n = f.read_to_end(&mut buf)?;
    Ok(buf)
}

fn chat_usage_err<T>(msg: &str) -> Result<T, String> {
    Err(format!("{msg}\n{CHAT_USAGE}"))
}

fn chat_value<I, S>(name: &str, inline: Option<&str>, it: &mut I) -> Result<String, String>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    if let Some(v) = inline {
        return Ok(v.to_string());
    }
    match it.next() {
        Some(s) => Ok(s.as_ref().to_string()),
        None => chat_usage_err(&format!("missing --{name} value")),
    }
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

    fn chat(args: &[&str]) -> ChatArgs {
        match parse_chat_args(args).expect("parse") {
            ChatCmd::Run(a) => a,
            ChatCmd::Help => panic!("expected Run"),
        }
    }

    #[test]
    fn chat_defaults_to_stdin_turns_and_no_system_message() {
        let a = chat(&["model.gguf"]);
        assert_eq!(a.path, "model.gguf");
        assert_eq!(a.system, None);
        assert_eq!(a.prompt, None);
        assert_eq!(a.n_predict, ChatArgs::DEFAULT_N_PREDICT);
        assert_eq!(a.n_predict, 64);
        assert_eq!(a.n_ctx, None);
        assert!(!a.show_prompt);
    }

    #[test]
    fn chat_takes_system_prompt_and_flags_in_any_order() {
        let a = chat(&[
            "--system",
            "You are terse.",
            "-p=Hi",
            "m.gguf",
            "--n-predict",
            "8",
            "--n-ctx=32",
            "--show-prompt",
        ]);
        assert_eq!(
            a,
            ChatArgs {
                path: "m.gguf".into(),
                system: Some("You are terse.".into()),
                prompt: Some("Hi".into()),
                n_predict: 8,
                n_ctx: Some(32),
                show_prompt: true,
            }
        );
        let a = chat(&["m.gguf", "-s", "sys", "-n", "1"]);
        assert_eq!(a.system.as_deref(), Some("sys"));
        assert_eq!(a.n_predict, 1);
    }

    #[test]
    fn chat_help_and_bad_arguments_do_not_run() {
        assert_eq!(parse_chat_args(["--help"]).unwrap(), ChatCmd::Help);
        assert_eq!(parse_chat_args(["-h", "m.gguf"]).unwrap(), ChatCmd::Help);
        let err = parse_chat_args(["--system", "s"]).unwrap_err();
        assert!(err.contains("missing GGUF path"), "{err}");
        assert!(err.contains("gguf_gemv chat"), "{err}");
        let err = parse_chat_args(["m.gguf", "-n", "x"]).unwrap_err();
        assert!(err.contains("invalid n-predict"), "{err}");
        let err = parse_chat_args(["m.gguf", "--n-ctx", "0"]).unwrap_err();
        assert!(err.contains("n-ctx must be > 0"), "{err}");
        let err = parse_chat_args(["m.gguf", "--nope"]).unwrap_err();
        assert!(err.contains("unknown flag"), "{err}");
        let err = parse_chat_args(["a.gguf", "b.gguf"]).unwrap_err();
        assert!(err.contains("unexpected argument"), "{err}");
        let err = parse_chat_args(["m.gguf", "--system"]).unwrap_err();
        assert!(err.contains("missing --system value"), "{err}");
    }

    #[test]
    fn any_control_token_ends_the_turn_not_just_eos() {
        use crate::gguf::{write_gguf_with_kv, Kv};
        use crate::load_gguf;
        // Llama-3's shape: `eos` is `<|end_of_text|>` but an instruct turn ends
        // with `<|eot_id|>`, a different id.
        let kv = vec![
            (
                "tokenizer.ggml.tokens".to_string(),
                Kv::Array {
                    elem: 8,
                    items: ["a", "<|end_of_text|>", "<|eot_id|>", "b"]
                        .into_iter()
                        .map(|s| Kv::String(s.into()))
                        .collect(),
                },
            ),
            (
                "tokenizer.ggml.token_type".to_string(),
                Kv::Array {
                    elem: 5,
                    items: vec![Kv::I32(1), Kv::I32(3), Kv::I32(3), Kv::I32(1)],
                },
            ),
            ("tokenizer.ggml.eos_token_id".to_string(), Kv::U32(1)),
        ];
        let bytes = write_gguf_with_kv(&kv, &[]);
        let g = load_gguf(&bytes).expect("gguf");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        assert!(!ends_turn(&tok, 0), "ordinary text must not stop the turn");
        assert!(!ends_turn(&tok, 3));
        assert!(ends_turn(&tok, 1), "eos");
        assert!(
            ends_turn(&tok, 2),
            "eot_id is not eos but still ends the turn"
        );
    }
}
