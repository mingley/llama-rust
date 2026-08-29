//! `gguf_gemv infer` / `chat` argument parsing, and the `chat` conversation
//! loop. No crates.io CLI crate.

use std::fs::File;
use std::io::{BufRead, Read, Write};
use std::path::Path;

use crate::decode::{prompt_ids, KvCache, Llama};
use crate::engine::{Engine, EngineCfg, SeqId};
use crate::gguf::load_gguf_owned;
use crate::sample::argmax;
use crate::template::ChatMessage;
use crate::tok::Tokenizer;
use expertvm::{CachedStore, HardwareProfile, LiveStore, SimulatedGpuStore};

/// Usage for the `infer` verb.
pub const INFER_USAGE: &str = "\
usage: gguf_gemv infer <path> [--prompt TEXT] [--n-predict N] [--n-ctx N]
  -p, --prompt TEXT   prompt (default: ab)
  -n, --n-predict N   tokens to generate (default: 2)
      --n-ctx N       KV capacity (default: prompt + n_predict + 1)
";

/// Usage for the `trace` verb.
pub const TRACE_USAGE: &str = "\
usage: gguf_gemv trace <path> [--prompt TEXT] [--n-predict N] [--n-ctx N] --out FILE [--capacity N]
  -p, --prompt TEXT   prompt (default: ab)
  -n, --n-predict N   tokens to generate (default: 2)
      --n-ctx N       KV capacity (default: prompt + n_predict + 1)
      --out FILE      write ExpertAccess JSONL (required; optional per-event w permille)
      --capacity N    expertvm cache slots for the printed table (default: 8)
";

/// Usage for the `chat` verb.
pub const CHAT_USAGE: &str = "\
usage: gguf_gemv chat <path> [--system TEXT] [--prompt TEXT] [--n-predict N] [--n-ctx N] [--kv-page N] [--show-prompt]
  -s, --system TEXT   system message placed before the conversation
  -p, --prompt TEXT   send one user turn and exit; omit to read turns from stdin
  -n, --n-predict N   tokens to generate per reply (default: 64)
      --n-ctx N       persistent KV capacity (default: grow per turn)
      --kv-page N     paged KV block size in tokens (default: dense layout)
      --show-prompt   print the rendered chat template before each reply
";

/// Top-level binary usage.
pub const BIN_USAGE: &str = "\
usage: gguf_gemv <command> [args]
  infer <path> [--prompt TEXT] [--n-predict N] [--n-ctx N]
  trace <path> [--prompt TEXT] [--n-predict N] [--n-ctx N] --out FILE [--capacity N]
  chat <path> [--system TEXT] [--prompt TEXT] [--n-predict N] [--n-ctx N] [--kv-page N] [--show-prompt]
  serve <path> [--n-predict N] [--n-ctx N] [--kv-page N] [--bind HOST:PORT]
  engine <path> [-p TEXT]... [-n N] [--n-ctx N] [--kv-page N] [--pool-blocks N] [--max-seqs N] [--prefill-chunk N] [--expert-slots N] [--expert-sim] [--trace-out FILE]
  write|gemv|write-q4k|gemv-q4k|write-tiny|write-tiny-qwen2|write-tiny-qwen3|write-tiny-gemma|write-tiny-llama4|write-tiny-llama-moe|write-tiny-qwen2moe|write-tiny-qwen3moe|write-tiny-qwen2vl|write-tiny-qwen3vl|write-tiny-qwen3next|write-tiny-qwen35|write-tiny-phi2 <path>
";

/// Usage for the `engine` verb.
pub const ENGINE_USAGE: &str = "\
usage: gguf_gemv engine <path> [--prompt TEXT]... [--n-predict N] [--n-ctx N] [--kv-page N] [--pool-blocks N] [--max-seqs N] [--prefill-chunk N] [--expert-slots N] [--expert-sim] [--trace-out FILE]
  -p, --prompt TEXT     prompt (repeatable; default: one `ab`)
  -n, --n-predict N     tokens to generate per sequence (default: 2)
      --n-ctx N         KV capacity (default: longest prompt + n_predict + 1)
      --kv-page N       paged KV block size in tokens (default: 16)
      --pool-blocks N   physical intern blocks (default: max_seqs * pages)
      --max-seqs N      in-flight sequences (default: number of prompts; extras wait)
      --prefill-chunk N prefill tokens per step (`0` = the rest; default: 0)
      --expert-slots N  ExpertStore on the Engine: omit = blob FFN, `0` = DirectStore,
                        N>0 = CachedStore with N slots (`--expert-sim` uses N, default 8)
      --expert-sim      SimulatedGpuStore (example H100 profile, 4096-byte experts)
      --trace-out FILE  write batched MoE ExpertAccess JSONL (all sequences)

Runs Engine continuous batching on one interned pool. Several `--prompt`s
join the same scheduler. A tight `--pool-blocks` preempts (recompute +
replay). One ExpertStore is parked on each batched GEMM so MoE serving
stays on the shared-pool path. MoE traces stay on that GEMM (per-row
sequence / token / prefix). Prints each continuation (`n_gen` plus
decoded text), then intern_hits, preempts, GEMM stats, store metrics,
and a gpu-sim score line when `--expert-sim` is set. Not `$/M tokens`.
Not an HTTP server.
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

/// Parsed `trace` invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceCmd {
    /// `--help` / `-h`.
    Help,
    /// Emit MoE JSONL with these arguments.
    Run(TraceArgs),
}

/// Arguments for `gguf_gemv trace`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceArgs {
    /// Same greedy settings as [`InferArgs`].
    pub infer: InferArgs,
    /// Destination JSONL path.
    pub out: String,
    /// Expert cache slots for the printed `expertvm replay` table.
    pub capacity: usize,
}

/// Parse operands after the `trace` verb.
pub fn parse_trace_args<I, S>(args: I) -> Result<TraceCmd, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut path = None;
    let mut prompt = InferArgs::DEFAULT_PROMPT.to_string();
    let mut n_predict = InferArgs::DEFAULT_N_PREDICT;
    let mut n_ctx = None;
    let mut out = None;
    let mut capacity = 8usize;
    let mut it = args.into_iter();
    while let Some(raw) = it.next() {
        let arg = raw.as_ref();
        if arg == "--help" || arg == "-h" {
            return Ok(TraceCmd::Help);
        }
        let (key, inline) = match arg.split_once('=') {
            Some((k, v)) => (k, Some(v)),
            None => (arg, None),
        };
        match key {
            "--prompt" | "-p" => {
                prompt = trace_value("prompt", inline, &mut it)?;
            }
            "--n-predict" | "-n" => {
                n_predict = parse_usize("n-predict", &trace_value("n-predict", inline, &mut it)?)?;
            }
            "--n-ctx" => {
                let n = parse_usize("n-ctx", &trace_value("n-ctx", inline, &mut it)?)?;
                if n == 0 {
                    return trace_usage_err("n-ctx must be > 0");
                }
                n_ctx = Some(n);
            }
            "--out" => {
                out = Some(trace_value("out", inline, &mut it)?);
            }
            "--capacity" | "-c" => {
                capacity = parse_usize("capacity", &trace_value("capacity", inline, &mut it)?)?;
                if capacity == 0 {
                    return trace_usage_err("capacity must be > 0");
                }
            }
            flag if flag.starts_with('-') => {
                return trace_usage_err(&format!("unknown flag {flag}"));
            }
            other => {
                if path.is_some() {
                    return trace_usage_err(&format!("unexpected argument {other}"));
                }
                path = Some(other.to_string());
            }
        }
    }
    let Some(path) = path else {
        return trace_usage_err("missing GGUF path");
    };
    let Some(out) = out else {
        return trace_usage_err("missing --out");
    };
    Ok(TraceCmd::Run(TraceArgs {
        infer: InferArgs {
            path,
            prompt,
            n_predict,
            n_ctx,
        },
        out,
        capacity,
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
    /// Paged KV block size in tokens. `None` keeps the dense layout.
    pub kv_page: Option<usize>,
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
/// `chat <path> [--system TEXT] [--prompt TEXT] [--n-predict N] [--n-ctx N] [--kv-page N] [--show-prompt]`
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
    let mut kv_page = None;
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
            "--kv-page" => {
                let v = chat_value("kv-page", inline, &mut it)?;
                let n = v
                    .parse::<usize>()
                    .map_err(|_| format!("invalid kv-page {v:?}\n{CHAT_USAGE}"))?;
                if n == 0 {
                    return chat_usage_err("kv-page must be > 0");
                }
                kv_page = Some(n);
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
        kv_page,
        show_prompt,
    }))
}

/// Parsed `engine` invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineCmd {
    /// `--help` / `-h`.
    Help,
    /// Run continuous batching with these arguments.
    Run(EngineArgs),
}

/// Arguments for `gguf_gemv engine`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineArgs {
    /// GGUF path.
    pub path: String,
    /// Prompts, one sequence each. Empty is replaced with a single `ab`.
    pub prompts: Vec<String>,
    /// Tokens to generate after each prompt.
    pub n_predict: usize,
    /// Optional KV capacity. `None` sizes to the longest prompt + `n_predict` + 1.
    pub n_ctx: Option<usize>,
    /// Paged KV block size in tokens.
    pub block_size: usize,
    /// Physical intern blocks. `None` sizes to `max_seqs` times pages per sequence.
    pub pool_blocks: Option<usize>,
    /// In-flight sequences. `None` is the number of prompts.
    pub max_seqs: Option<usize>,
    /// Prefill tokens per sequence per step (`0` = the rest of the prompt).
    pub prefill_chunk: usize,
    /// ExpertStore slots. `None` keeps blob FFN. `Some(0)` is DirectStore.
    pub expert_slots: Option<usize>,
    /// Attach [`SimulatedGpuStore`] (example H100) instead of Direct/Cached.
    pub expert_sim: bool,
    /// Write Engine MoE traces as JSONL. `None` leaves tracing off.
    pub trace_out: Option<String>,
}

impl EngineArgs {
    /// Default `--kv-page` when omitted.
    pub const DEFAULT_BLOCK_SIZE: usize = 16;
}

/// Parse operands after the `engine` verb.
///
/// Path may appear before or after flags. `--flag=value` is accepted.
/// `--prompt` / `-p` may be repeated.
pub fn parse_engine_args<I, S>(args: I) -> Result<EngineCmd, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut path = None;
    let mut prompts = Vec::new();
    let mut n_predict = InferArgs::DEFAULT_N_PREDICT;
    let mut n_ctx = None;
    let mut block_size = EngineArgs::DEFAULT_BLOCK_SIZE;
    let mut pool_blocks = None;
    let mut max_seqs = None;
    let mut prefill_chunk = 0usize;
    let mut expert_slots = None;
    let mut expert_sim = false;
    let mut trace_out = None;
    let mut it = args.into_iter();
    while let Some(raw) = it.next() {
        let arg = raw.as_ref();
        if arg == "--help" || arg == "-h" {
            return Ok(EngineCmd::Help);
        }
        let (key, inline) = match arg.split_once('=') {
            Some((k, v)) => (k, Some(v)),
            None => (arg, None),
        };
        match key {
            "--prompt" | "-p" => prompts.push(engine_value("prompt", inline, &mut it)?),
            "--n-predict" | "-n" => {
                n_predict =
                    engine_usize("n-predict", &engine_value("n-predict", inline, &mut it)?)?;
            }
            "--n-ctx" => {
                let n = engine_usize("n-ctx", &engine_value("n-ctx", inline, &mut it)?)?;
                if n == 0 {
                    return engine_err("n-ctx must be > 0");
                }
                n_ctx = Some(n);
            }
            "--kv-page" => {
                let n = engine_usize("kv-page", &engine_value("kv-page", inline, &mut it)?)?;
                if n == 0 {
                    return engine_err("kv-page must be > 0");
                }
                block_size = n;
            }
            "--pool-blocks" => {
                let n = engine_usize(
                    "pool-blocks",
                    &engine_value("pool-blocks", inline, &mut it)?,
                )?;
                if n == 0 {
                    return engine_err("pool-blocks must be > 0");
                }
                pool_blocks = Some(n);
            }
            "--max-seqs" => {
                let n = engine_usize("max-seqs", &engine_value("max-seqs", inline, &mut it)?)?;
                if n == 0 {
                    return engine_err("max-seqs must be > 0");
                }
                max_seqs = Some(n);
            }
            "--prefill-chunk" => {
                prefill_chunk = engine_usize(
                    "prefill-chunk",
                    &engine_value("prefill-chunk", inline, &mut it)?,
                )?;
            }
            "--expert-slots" => {
                expert_slots = Some(engine_usize(
                    "expert-slots",
                    &engine_value("expert-slots", inline, &mut it)?,
                )?);
            }
            "--expert-sim" => {
                if inline.is_some() {
                    return engine_err("--expert-sim does not take a value");
                }
                expert_sim = true;
            }
            "--trace-out" => {
                trace_out = Some(engine_value("trace-out", inline, &mut it)?);
            }
            flag if flag.starts_with('-') => {
                return engine_err(&format!("unknown flag {flag}"));
            }
            other => {
                if path.is_some() {
                    return engine_err(&format!("unexpected argument {other}"));
                }
                path = Some(other.to_string());
            }
        }
    }
    let Some(path) = path else {
        return engine_err("missing GGUF path");
    };
    if prompts.is_empty() {
        prompts.push(InferArgs::DEFAULT_PROMPT.to_string());
    }
    Ok(EngineCmd::Run(EngineArgs {
        path,
        prompts,
        n_predict,
        n_ctx,
        block_size,
        pool_blocks,
        max_seqs,
        prefill_chunk,
        expert_slots,
        expert_sim,
        trace_out,
    }))
}

/// Continuous batching: several prompts on one interned [`Engine`] pool.
pub fn run_engine(args: &EngineArgs) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = read_file(Path::new(&args.path))?;
    let g = load_gguf_owned(bytes)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let model = Llama::from_gguf(g)?;
    let cfg = engine_cfg(&tok, args)?;
    let mut eng = Engine::new(&model, cfg)?;
    attach_engine_store(&mut eng, &model, args)?;
    if args.trace_out.is_some() {
        eng.enable_moe_trace();
    }
    let mut handles = Vec::new();
    for ids in engine_prompts(&tok, args)? {
        handles.push(eng.add(&ids, args.n_predict)?);
    }
    eng.run()?;
    if let Some(path) = args.trace_out.as_ref() {
        write_engine_traces(&mut eng, &handles, path)?;
    }
    let mut out = std::io::stdout();
    for (i, id) in handles.iter().enumerate() {
        let seq = eng.take(*id).ok_or("engine take")?;
        let text = tok.decode(&seq.generated);
        writeln!(
            out,
            "seq={i} n_gen={} generated={text}",
            seq.generated.len()
        )?;
    }
    writeln!(
        out,
        "intern_hits={} preempts={} steps={} gemm_tokens={} gemm_peak={} serial_tokens={}",
        eng.pool().hits(),
        eng.preempts(),
        eng.stats().steps,
        eng.stats().gemm_tokens,
        eng.stats().gemm_peak,
        eng.stats().serial_tokens
    )?;
    if let Some(m) = eng.expert_store_metrics() {
        writeln!(out, "{}", m.line())?;
    }
    if let Some(score) = eng.expert_store_score()? {
        writeln!(out, "{}", score.line())?;
    }
    Ok(())
}

fn write_engine_traces(
    eng: &mut Engine<'_>,
    ids: &[SeqId],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut buf = String::new();
    for id in ids {
        if let Some(t) = eng.take_moe_trace(*id) {
            buf.push_str(&t.to_jsonl());
        }
    }
    write_file(Path::new(path), buf.as_bytes())?;
    Ok(())
}

fn attach_engine_store(
    eng: &mut Engine<'_>,
    llama: &Llama,
    args: &EngineArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    if args.expert_sim {
        let slots = match args.expert_slots {
            Some(0) => return Err("--expert-sim needs --expert-slots > 0".into()),
            Some(n) => n,
            None => 8,
        };
        let gpu = SimulatedGpuStore::new(
            llama.expert_direct_store()?,
            slots,
            HardwareProfile::example_h100_sxm(),
            4096,
        )?;
        eng.attach_expert_store(LiveStore::simulated(gpu));
        return Ok(());
    }
    let Some(slots) = args.expert_slots else {
        return Ok(());
    };
    let direct = llama.expert_direct_store()?;
    let store = if slots == 0 {
        LiveStore::Direct(direct)
    } else {
        LiveStore::Cached(CachedStore::new(direct, slots)?)
    };
    eng.attach_expert_store(store);
    Ok(())
}

fn engine_prompts(
    tok: &Tokenizer,
    args: &EngineArgs,
) -> Result<Vec<Vec<u32>>, Box<dyn std::error::Error>> {
    let mut encoded = Vec::new();
    for p in &args.prompts {
        let ids = prompt_ids(tok, p)?;
        if ids.is_empty() {
            return Err("empty prompt tokens".into());
        }
        encoded.push(ids);
    }
    Ok(encoded)
}

fn engine_cfg(tok: &Tokenizer, args: &EngineArgs) -> Result<EngineCfg, Box<dyn std::error::Error>> {
    let encoded = engine_prompts(tok, args)?;
    let mut max_needed = 0usize;
    for ids in &encoded {
        max_needed = max_needed.max(ids.len().saturating_add(args.n_predict));
    }
    let n_ctx = match args.n_ctx {
        Some(n) if n < max_needed => {
            return Err(format!("--n-ctx {n} is below the {max_needed} tokens needed").into());
        }
        Some(n) => n,
        None => max_needed.saturating_add(1),
    };
    let max_seqs = args.max_seqs.unwrap_or(encoded.len());
    let pages = n_ctx.div_ceil(args.block_size).saturating_add(1);
    let pool_blocks = args
        .pool_blocks
        .unwrap_or(max_seqs.saturating_mul(pages).max(2));
    Ok(EngineCfg {
        n_ctx,
        block_size: args.block_size,
        pool_blocks,
        max_seqs,
        prefill_chunk: args.prefill_chunk,
        eos: tok.eos,
    })
}

/// Hold a conversation using the model's own `tokenizer.chat_template`.
///
/// With `--prompt` this is one turn and exits. Otherwise each line of stdin is
/// a user turn, and replies accumulate so the model sees the whole history.
/// Every turn re-renders the template. Prefill uses [`crate::decode::Llama::prompt`]
/// so a matching token prefix keeps its KV. Templates that rewrite earlier
/// turns simply get a shorter hit. `--n-ctx` sizes the persistent cache;
/// `--kv-page N` stores that cache in interned blocks. Omit `--n-ctx` and
/// the cache is reallocated when a turn no longer fits.
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
    let mut cache = None;
    if let Some(p) = &args.prompt {
        return chat_turn(&model, &tok, args, &mut history, p, &mut out, &mut cache);
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
        chat_turn(&model, &tok, args, &mut history, turn, &mut out, &mut cache)?;
    }
}

fn chat_turn<W: Write>(
    model: &Llama,
    tok: &Tokenizer,
    args: &ChatArgs,
    history: &mut Vec<ChatMessage>,
    turn: &str,
    out: &mut W,
    cache: &mut Option<KvCache>,
) -> Result<(), Box<dyn std::error::Error>> {
    history.push(ChatMessage::user(turn));
    let prompt = tok.apply_chat_template(history, true)?;
    if args.show_prompt {
        writeln!(out, "--- rendered prompt ---\n{prompt}\n--- end ---")?;
    }
    let reply = generate_reply(model, tok, &prompt, args, cache)?;
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
    cache: &mut Option<KvCache>,
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
    if let Some(n) = args.n_ctx {
        if n < needed {
            return Err(format!("--n-ctx {n} is below the {needed} tokens this turn needs").into());
        }
    }
    let kv = model.ensure_cache_page(cache, needed, args.n_ctx, args.kv_page)?;
    let mut logits = model.prompt(kv, &ids)?;
    let mut reply = Vec::new();
    for _ in 0..args.n_predict {
        let next = argmax(&logits);
        if ends_turn(tok, next) {
            break;
        }
        reply.push(next);
        logits = model.forward(kv, next)?;
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
fn write_file(path: &Path, bytes: &[u8]) -> Result<(), std::io::Error> {
    let mut f = File::create(path)?;
    f.write_all(bytes)?;
    Ok(())
}

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

fn engine_err<T>(msg: &str) -> Result<T, String> {
    Err(format!("{msg}\n{ENGINE_USAGE}"))
}

fn engine_value<I, S>(name: &str, inline: Option<&str>, it: &mut I) -> Result<String, String>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    if let Some(v) = inline {
        return Ok(v.to_string());
    }
    match it.next() {
        Some(s) => Ok(s.as_ref().to_string()),
        None => engine_err(&format!("missing --{name} value")),
    }
}

fn engine_usize(name: &str, s: &str) -> Result<usize, String> {
    s.parse::<usize>()
        .map_err(|_| format!("invalid {name} {s:?}\n{ENGINE_USAGE}"))
}

fn usage_err<T>(msg: &str) -> Result<T, String> {
    Err(format!("{msg}\n{INFER_USAGE}"))
}

fn trace_usage_err<T>(msg: &str) -> Result<T, String> {
    Err(format!("{msg}\n{TRACE_USAGE}"))
}

fn trace_value<I, S>(name: &str, inline: Option<&str>, it: &mut I) -> Result<String, String>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    if let Some(v) = inline {
        return Ok(v.to_string());
    }
    match it.next() {
        Some(s) => Ok(s.as_ref().to_string()),
        None => trace_usage_err(&format!("missing --{name} value")),
    }
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
        assert_eq!(a.kv_page, None);
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
            "--kv-page=2",
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
                kv_page: Some(2),
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
        let err = parse_chat_args(["m.gguf", "--kv-page", "0"]).unwrap_err();
        assert!(err.contains("kv-page must be > 0"), "{err}");
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

    #[test]
    fn trace_requires_out_and_path() {
        let err = parse_trace_args(["m.gguf"]).unwrap_err();
        assert!(err.contains("missing --out"), "{err}");
        assert!(err.contains("gguf_gemv trace"), "{err}");
        match parse_trace_args(["m.gguf", "--out", "t.jsonl"]).expect("ok") {
            TraceCmd::Run(a) => {
                assert_eq!(a.out, "t.jsonl");
                assert_eq!(a.infer.path, "m.gguf");
                assert_eq!(a.infer.prompt, InferArgs::DEFAULT_PROMPT);
                assert_eq!(a.capacity, 8);
            }
            TraceCmd::Help => panic!("expected Run"),
        }
        assert_eq!(parse_trace_args(["--help"]).unwrap(), TraceCmd::Help);
        match parse_trace_args(["m.gguf", "--out", "t.jsonl", "-c", "2"]).expect("cap") {
            TraceCmd::Run(a) => assert_eq!(a.capacity, 2),
            TraceCmd::Help => panic!("expected Run"),
        }
    }

    #[test]
    fn engine_repeatable_prompts_and_rejects_bad_flags() {
        match parse_engine_args(["m.gguf"]).expect("def") {
            EngineCmd::Run(a) => {
                assert_eq!(a.prompts, vec![String::from("ab")]);
                assert_eq!(a.n_predict, 2);
                assert_eq!(a.block_size, EngineArgs::DEFAULT_BLOCK_SIZE);
                assert_eq!(a.pool_blocks, None);
                assert_eq!(a.prefill_chunk, 0);
                assert_eq!(a.expert_slots, None);
                assert!(!a.expert_sim);
                assert_eq!(a.trace_out, None);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["-p", "a", "-p", "b", "m.gguf", "--kv-page", "2"]).expect("two") {
            EngineCmd::Run(a) => {
                assert_eq!(a.prompts, vec![String::from("a"), String::from("b")]);
                assert_eq!(a.block_size, 2);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-slots", "0"]).expect("direct") {
            EngineCmd::Run(a) => assert_eq!(a.expert_slots, Some(0)),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-slots=8"]).expect("cached") {
            EngineCmd::Run(a) => assert_eq!(a.expert_slots, Some(8)),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim"]).expect("sim") {
            EngineCmd::Run(a) => {
                assert!(a.expert_sim);
                assert_eq!(a.expert_slots, None);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--trace-out", "t.jsonl"]).expect("trace") {
            EngineCmd::Run(a) => assert_eq!(a.trace_out.as_deref(), Some("t.jsonl")),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--trace-out=out.jsonl"]).expect("eq") {
            EngineCmd::Run(a) => assert_eq!(a.trace_out.as_deref(), Some("out.jsonl")),
            EngineCmd::Help => panic!("expected Run"),
        }
        assert_eq!(parse_engine_args(["--help"]).unwrap(), EngineCmd::Help);
        let err = parse_engine_args(["--n-predict", "2"]).unwrap_err();
        assert!(err.contains("missing GGUF path"), "{err}");
        let err = parse_engine_args(["m.gguf", "--pool-blocks", "0"]).unwrap_err();
        assert!(err.contains("pool-blocks must be > 0"), "{err}");
        let err = parse_engine_args(["m.gguf", "--max-seqs", "0"]).unwrap_err();
        assert!(err.contains("max-seqs must be > 0"), "{err}");
        let err = parse_engine_args(["m.gguf", "--nope"]).unwrap_err();
        assert!(err.contains("unknown flag"), "{err}");
        let err = parse_engine_args(["m.gguf", "--trace-out"]).unwrap_err();
        assert!(err.contains("missing --trace-out value"), "{err}");
    }

    #[test]
    fn engine_trace_out_writes_parseable_jsonl() {
        let pid = std::process::id();
        let dir = std::env::temp_dir();
        let gguf = dir.join(format!("llama-rust-eng-trace-{pid}.gguf"));
        let jsonl = dir.join(format!("llama-rust-eng-trace-{pid}.jsonl"));
        write_file(&gguf, &crate::decode::tiny_qwen3moe_gguf()).expect("gguf");
        let args = match parse_engine_args([
            gguf.to_str().expect("utf8 gguf"),
            "-p",
            "a",
            "-p",
            "b",
            "--kv-page",
            "2",
            "--trace-out",
            jsonl.to_str().expect("utf8 jsonl"),
        ])
        .expect("parse")
        {
            EngineCmd::Run(a) => a,
            EngineCmd::Help => panic!("expected Run"),
        };
        run_engine(&args).expect("run");
        let bytes = read_file(&jsonl).expect("read");
        let text = String::from_utf8(bytes).expect("utf8");
        let t = expertvm::Trace::parse(&text).expect("jsonl");
        assert!(
            !t.events.is_empty(),
            "engine --trace-out must emit MoE events"
        );
        assert!(t.events.iter().any(|e| e.sequence == 0));
        assert!(t.events.iter().any(|e| e.sequence == 1));
        let _rm_g = std::fs::remove_file(&gguf);
        let _rm_j = std::fs::remove_file(&jsonl);
    }
}
