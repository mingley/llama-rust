//! Local HTTP/1.1 serve. Not a production inference server.
//!
//! Default `gguf_gemv serve` accepts one connection at a time with a persistent
//! KV cache. `--engine` admits concurrent `POST /generate` onto one [`Engine`]
//! so prefills and decodes GEMM together. No keep-alive, no OpenAI SDK surface,
//! no crates.io HTTP stack.

use std::fs::File;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, TcpListener};
use std::path::Path;
use std::time::Duration;

use crate::cli::InferArgs;
use crate::decode::{greedy_generate_slot, KvCache, Llama, LlamaError};
use crate::gguf::load_gguf_owned;
use crate::serve_engine;
use crate::store_attach::{gpu_knobs, PlannerCli};
use crate::template::ChatMessage;
use crate::tok::Tokenizer;
use expertvm::{GpuFill, GpuStoreCfg, Prefetch};

/// Usage for the `serve` verb.
pub const SERVE_USAGE: &str = "\
usage: gguf_gemv serve <path> [--n-predict N] [--n-ctx N] [--kv-page N] [--bind HOST:PORT] [--engine] [--max-seqs N] [--expert-slots N] [--expert-sim] [--expert-8gpu] [--expert-bytes N] [--prefill-chunk N] [--decode-first] [--slo-reject] [--ttft-slo-ns N] [--itl-slo-ns N] [--cuda-graphs] [--graph-update] [--graph-clone] [--timing-events] [--mapped] [--managed] [--vmm] [--vmm-page N] [--host-func] [--blocking-streams] [--sync-alloc] [--mempool] [--pageable] [--accessed-by] [--legacy-null] [--stream-priority] [--seq-streams] [--kv-sim] [--kv-bytes N] [--decode-priority] [--compute-slots N] [--decode-sms N] [--prefetch none|copy-forward|markov|both] [--plan-window N] [--plan-threshold N] [--trace-out FILE]
  -n, --n-predict N   tokens to generate (default: 2)
      --n-ctx N       KV capacity (default: grow per request; `--engine` default 64)
      --kv-page N     paged KV block size (default: dense; `--engine` default 16)
      --bind ADDR     loopback listen address (default: 127.0.0.1:8080)
      --engine        concurrent requests on one Engine (continuous-batch GEMM)
      --max-seqs N    in-flight Engine sequences (`--engine`; default: 4)
      --expert-slots N  ExpertStore on `--engine`: omit = blob FFN, `0` = DirectStore,
                        N>0 = CachedStore (`--expert-sim` uses N, default 8)
      --expert-sim      SimulatedGpuStore (`--engine`; example H100, 4096-byte experts)
      --expert-8gpu     8×H100 NVLink profile (`--expert-sim`; enables plan_placement)
      --expert-bytes N  simulated expert page bytes (`--expert-sim`; default: 4096)
      --prefill-chunk N prefill tokens per Engine step (`--engine`; `0` = the rest)
      --decode-first    hold leftover prefill while any live sequence is decoding (`--engine`)
      --slo-reject      drop waiters whose gpu-sim queue wait meets `--ttft-slo-ns` (`--engine`)
      --ttft-slo-ns N   virtual-ns TTFT budget (`--slo-reject`; needs `--expert-sim`)
      --itl-slo-ns N    count later-token gaps over this budget (`--engine`; `--expert-sim`)
      --cuda-graphs     document default GEMM graph capture (`--expert-sim`; always on)
      --graph-update    cudaGraphExecUpdate parked leaves (`--expert-sim`)
      --graph-clone     cudaGraphClone before instantiate (`--expert-sim`)
      --timing-events   cudaEventElapsedTime on copy start/end (`--expert-sim`)
      --mapped          cudaHostAllocMapped miss pages (`--expert-sim`)
      --managed         cudaMallocManaged miss pages (`--expert-sim`)
      --vmm             va_acquire miss pages (`--expert-sim`)
      --vmm-page N      va_acquire_paged span (`--expert-sim`; `N>0` implies `--vmm`)
      --host-func       cudaLaunchHostFunc after acquire GEMM (`--expert-sim`)
      --blocking-streams  blocking compute stream (`--expert-sim`)
      --sync-alloc      host-sync malloc/memcpy/free (`--expert-sim`)
      --mempool         hold unused cudaMallocAsync bytes (`--expert-sim`)
      --pageable        pageable H2D (`--expert-sim`)
      --accessed-by     SetAccessedBy on managed fills (`--expert-sim`)
      --legacy-null     NULL copy serializes with compute (`--expert-sim`)
      --stream-priority cudaStreamCreateWithPriority on compute (`--expert-sim`)
      --seq-streams     per-sequence copy streams (`--expert-sim`; grouped GEMM stays fused)
      --kv-sim          interned KV on the SimulatedGpuStore clock (`--expert-sim`; default off)
      --kv-bytes N      KV page bytes for `--kv-sim` (default: f32 K+V of one intern block)
      --decode-priority decode GEMMs on a higher-priority compute stream (`--expert-sim`; implies `--stream-priority`; ITL samples that stream)
      --compute-slots N Hyper-Q occupancy (`--expert-sim`; `1` exclusive, `>=2` overlaps leftover prefill with decode on two streams; default profile `1`)
      --decode-sms N    decode-stream SM permille (`--expert-sim`; `1..=1000`; leftover prefill gets the remainder; implies `--decode-priority`; default full chip)
      --prefetch MODE   none|copy-forward|markov|both (`--engine`; default: both)
      --plan-window N   Stay vs Fetch over N unique predicted keys (`--engine`; `0` ungated)
      --plan-threshold N  Stay permille of that window (`--engine`; default: 500)
      --trace-out FILE  append batched MoE ExpertAccess JSONL (`--engine`)

POST /generate takes {\"prompt\": TEXT} or, to render the model's own
tokenizer.chat_template, {\"messages\": [{\"role\": R, \"content\": C}, ...]}
with optional \"add_generation_prompt\" (default true), \"n_predict\", and
\"stream\" (NDJSON token lines then a final generated object; `--engine` only).
Default serve keeps one KV cache across requests (`prefix_hit` in the JSON).
`--engine` admits several connections onto one interned pool so they GEMM
together (`gguf_gemv engine` is the same scheduler). `--kv-page N` interned
completed blocks so a later prompt can hit them after a rewind (`page_hits`).
`--engine` JSON `page_hits` is the intern-hit delta for that sequence.
`--expert-slots` / `--expert-sim` park the same ExpertStore as `gguf_gemv engine`.
`--prefill-chunk` interleaves long prefills with decode. `--decode-first` holds
leftover prefill while any live sequence is already decoding. `--slo-reject` /
`--ttft-slo-ns` drop a waiter whose gpu-sim queue wait meets the TTFT budget
(`--expert-sim`). `--itl-slo-ns` counts later-token ITL misses (does not drop).
`--cuda-graphs` / `--graph-update` / `--graph-clone` / `--timing-events` are
the same SimulatedGpuStore knobs as `gguf_gemv engine`. `--host-func` /
`--blocking-streams` / `--sync-alloc` / `--mempool` / `--vmm-page` /
`--pageable` / `--accessed-by` / `--legacy-null` / `--stream-priority` / `--seq-streams` /
`--kv-sim` / `--kv-bytes` / `--decode-priority` / `--compute-slots` /
`--decode-sms` match
`GpuStoreCfg`. `--kv-sim` bills interned KV on the same clock as expert H2D
(distinct from `expertvm kv`; default off). `--decode-priority` ITL samples
the decode compute stream so leftover prefill does not inflate it.
`--compute-slots N` (`N>=2`, with `--decode-priority`) lets leftover prefill
and decode GEMMs overlap at full issue rate. Default profile occupancy is
exclusive (`1`). `--decode-sms N` (`1..=1000`) reserves that permille of
peak FLOP/s for decode GEMMs (green-context; leftover prefill gets the
remainder). Default unset is a full chip. `--mapped` / `--managed` / `--vmm` choose miss-page placement
(default pinned H2D). `--prefetch` / `--plan-window` / `--plan-threshold`
match `gguf_gemv engine` (predicted keys only; `--engine`). `--trace-out` writes
router JSONL as sequences finish. Not a production inference server.
";

pub(crate) const MAX_REQ: u64 = 65_536;
const IO_TIMEOUT_SECS: u64 = 30;

/// Parsed `serve` invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[expect(
    clippy::large_enum_variant,
    reason = "ServeArgs is the parsed CLI record; boxing would churn every match"
)]
pub enum ServeCmd {
    /// `--help` / `-h`.
    Help,
    /// Listen with these arguments.
    Run(ServeArgs),
}

/// Arguments for local one-request serve. Seedless greedy, same defaults as `infer`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeArgs {
    /// GGUF path.
    pub path: String,
    /// Tokens to generate when the JSON body omits `n_predict`.
    pub n_predict: usize,
    /// Optional persistent KV capacity. `None` grows to prompt + `n_predict` + 1
    /// on the first request that does not fit the current cache.
    pub n_ctx: Option<usize>,
    /// Paged KV block size in tokens. `None` keeps the dense `max_seq` layout.
    pub kv_page: Option<usize>,
    /// Loopback `HOST:PORT`. Host must be `127.0.0.1` or `localhost`.
    pub bind: String,
    /// Admit concurrent requests onto one [`crate::Engine`].
    pub engine: bool,
    /// In-flight Engine sequences. `None` is 4 when `--engine`.
    pub max_seqs: Option<usize>,
    /// ExpertStore slots. `None` keeps blob FFN. `Some(0)` is DirectStore.
    pub expert_slots: Option<usize>,
    /// Attach [`expertvm::SimulatedGpuStore`] (`--engine`).
    pub expert_sim: bool,
    /// Example 8×H100 NVLink profile (`--expert-sim`).
    pub expert_8gpu: bool,
    /// Simulated expert page bytes (`--expert-sim`). `None` is 4096.
    pub expert_bytes: Option<u64>,
    /// Prefill tokens per Engine step (`0` = the rest). `--engine` only.
    pub prefill_chunk: usize,
    /// Hold leftover prefill while any live sequence is decoding. `--engine` only.
    pub decode_first: bool,
    /// Drop waiters whose gpu-sim queue wait meets `ttft_slo_ns`. `--engine` only.
    pub slo_reject: bool,
    /// Virtual-ns TTFT budget with `--slo-reject`. `None` never drops.
    pub ttft_slo_ns: Option<u64>,
    /// Later-token gap budget. Misses increment `Engine::itl_slo_miss`.
    pub itl_slo_ns: Option<u64>,
    /// CUDA-like knobs for SimulatedGpuStore. Identity stays default.
    pub gpu_cfg: GpuStoreCfg,
    /// Simulated KV page bytes with `--kv-sim`. `None` uses intern geometry.
    pub kv_bytes: Option<u64>,
    /// Miss-page placement (`--expert-sim`). Default is pinned H2D.
    pub fill: GpuFill,
    /// Predictor prefetch (`--engine`). Default [`Prefetch::Both`].
    pub prefetch: Prefetch,
    /// Unique predicted-key Stay vs Fetch window (`--engine`). `0` is ungated.
    pub plan_window: usize,
    /// Stay permille of the predicted window already resident (`--engine`).
    pub plan_threshold: u32,
    /// Append Engine MoE traces as JSONL (`--engine`). `None` leaves tracing off.
    pub trace_out: Option<String>,
}

impl ServeArgs {
    /// Default `--bind` when omitted.
    pub const DEFAULT_BIND: &'static str = "127.0.0.1:8080";
}

/// Local serve failure.
#[derive(Debug)]
pub enum ServeError {
    /// `--bind` was not loopback or listen failed.
    Bind(String),
    /// HTTP/1.1 request could not be read or framed.
    Http(String),
    /// GGUF path could not be opened.
    MissingFile(String),
    /// I/O while reading or writing a connection.
    Io(std::io::Error),
    /// Model load or generate failed.
    Infer(String),
}

impl std::fmt::Display for ServeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bind(s) | Self::Http(s) | Self::MissingFile(s) | Self::Infer(s) => {
                write!(f, "{s}")
            }
            Self::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ServeError {}

impl From<std::io::Error> for ServeError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

struct ServeNeed {
    engine: bool,
    expert_sim: bool,
    expert_8gpu: bool,
    has_slots: bool,
    has_bytes: bool,
    has_max: bool,
    prefill_chunk: usize,
    decode_first: bool,
    slo_reject: bool,
    has_ttft: bool,
    has_itl: bool,
    has_trace: bool,
    plan: PlannerCli,
}

fn check_serve_need(n: &ServeNeed) -> Result<(), String> {
    if n.has_max && !n.engine {
        return usage_err("--max-seqs requires --engine");
    }
    if n.has_slots && !n.engine {
        return usage_err("--expert-slots requires --engine");
    }
    if n.expert_sim && !n.engine {
        return usage_err("--expert-sim requires --engine");
    }
    if n.expert_8gpu && !n.engine {
        return usage_err("--expert-8gpu requires --engine");
    }
    if n.has_bytes && !n.engine {
        return usage_err("--expert-bytes requires --engine");
    }
    if n.expert_8gpu && !n.expert_sim {
        return usage_err("--expert-8gpu requires --expert-sim");
    }
    if n.has_bytes && !n.expert_sim {
        return usage_err("--expert-bytes requires --expert-sim");
    }
    if n.prefill_chunk > 0 && !n.engine {
        return usage_err("--prefill-chunk requires --engine");
    }
    if n.decode_first && !n.engine {
        return usage_err("--decode-first requires --engine");
    }
    if n.slo_reject && !n.engine {
        return usage_err("--slo-reject requires --engine");
    }
    if n.has_ttft && !n.engine {
        return usage_err("--ttft-slo-ns requires --engine");
    }
    if n.slo_reject && !n.has_ttft {
        return usage_err("--slo-reject requires --ttft-slo-ns");
    }
    if n.has_ttft && !n.slo_reject {
        return usage_err("--ttft-slo-ns requires --slo-reject");
    }
    if (n.slo_reject || n.has_ttft) && !n.expert_sim {
        return usage_err("--slo-reject requires --expert-sim");
    }
    if n.has_itl && !n.engine {
        return usage_err("--itl-slo-ns requires --engine");
    }
    if n.has_itl && !n.expert_sim {
        return usage_err("--itl-slo-ns requires --expert-sim");
    }
    if n.plan.gpu.kv_bytes.is_some() && !n.plan.gpu.kv_sim {
        return usage_err("--kv-bytes requires --kv-sim");
    }
    if let Some(flag) = n.plan.gpu.sim_flag() {
        if !n.engine {
            return usage_err(&format!("{flag} requires --engine"));
        }
        if !n.expert_sim {
            return usage_err(&format!("{flag} requires --expert-sim"));
        }
    }
    if let Some(flag) = n.plan.serve_engine_flag() {
        if !n.engine {
            return usage_err(&format!("{flag} requires --engine"));
        }
    }
    if n.has_trace && !n.engine {
        return usage_err("--trace-out requires --engine");
    }
    Ok(())
}

/// Parse operands after the `serve` verb.
///
/// `serve <path> [--n-predict N] [--n-ctx N] [--kv-page N] [--bind HOST:PORT] [--engine] [--max-seqs N] [--expert-slots N] [--expert-sim] [--expert-8gpu] [--expert-bytes N] [--prefill-chunk N] [--decode-first] [--slo-reject] [--ttft-slo-ns N] [--itl-slo-ns N] [--cuda-graphs] [--graph-update] [--graph-clone] [--timing-events] [--mapped] [--managed] [--vmm] [--vmm-page N] [--host-func] [--blocking-streams] [--sync-alloc] [--mempool] [--pageable] [--accessed-by] [--legacy-null] [--stream-priority] [--seq-streams] [--kv-sim] [--kv-bytes N] [--decode-priority] [--compute-slots N] [--decode-sms N] [--prefetch none|copy-forward|markov|both] [--plan-window N] [--plan-threshold N] [--trace-out FILE]`
/// Path may appear before or after flags. `--flag=value` is accepted.
pub fn parse_serve_args<I, S>(args: I) -> Result<ServeCmd, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut path = None;
    let mut n_predict = InferArgs::DEFAULT_N_PREDICT;
    let mut n_ctx = None;
    let mut kv_page = None;
    let mut bind = ServeArgs::DEFAULT_BIND.to_string();
    let mut engine = false;
    let mut max_seqs = None;
    let mut expert_slots = None;
    let mut expert_sim = false;
    let mut expert_8gpu = false;
    let mut expert_bytes = None;
    let mut prefill_chunk = 0usize;
    let mut decode_first = false;
    let mut slo_reject = false;
    let mut ttft_slo_ns = None;
    let mut itl_slo_ns = None;
    let mut planner = PlannerCli::default();
    let mut trace_out = None;
    let mut it = args.into_iter();
    while let Some(raw) = it.next() {
        let arg = raw.as_ref();
        if arg == "--help" || arg == "-h" {
            return Ok(ServeCmd::Help);
        }
        let (key, inline) = match arg.split_once('=') {
            Some((k, v)) => (k, Some(v)),
            None => (arg, None),
        };
        match key {
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
            "--kv-page" => {
                let n = parse_usize("kv-page", &opt_value("kv-page", inline, &mut it)?)?;
                if n == 0 {
                    return usage_err("kv-page must be > 0");
                }
                kv_page = Some(n);
            }
            "--bind" => {
                bind = opt_value("bind", inline, &mut it)?;
                let _addr = parse_bind(&bind)?;
            }
            "--engine" => {
                if inline.is_some() {
                    return usage_err("--engine does not take a value");
                }
                engine = true;
            }
            "--max-seqs" => {
                let n = parse_usize("max-seqs", &opt_value("max-seqs", inline, &mut it)?)?;
                if n == 0 {
                    return usage_err("max-seqs must be > 0");
                }
                max_seqs = Some(n);
            }
            "--expert-slots" => {
                expert_slots = Some(parse_usize(
                    "expert-slots",
                    &opt_value("expert-slots", inline, &mut it)?,
                )?);
            }
            "--expert-sim" => {
                if inline.is_some() {
                    return usage_err("--expert-sim does not take a value");
                }
                expert_sim = true;
            }
            "--expert-8gpu" => {
                if inline.is_some() {
                    return usage_err("--expert-8gpu does not take a value");
                }
                expert_8gpu = true;
            }
            "--expert-bytes" => {
                let n = parse_u64("expert-bytes", &opt_value("expert-bytes", inline, &mut it)?)?;
                if n == 0 {
                    return usage_err("expert-bytes must be > 0");
                }
                expert_bytes = Some(n);
            }
            "--prefill-chunk" => {
                prefill_chunk = parse_usize(
                    "prefill-chunk",
                    &opt_value("prefill-chunk", inline, &mut it)?,
                )?;
            }
            "--decode-first" => {
                if inline.is_some() {
                    return usage_err("--decode-first does not take a value");
                }
                decode_first = true;
            }
            "--slo-reject" => {
                if inline.is_some() {
                    return usage_err("--slo-reject does not take a value");
                }
                slo_reject = true;
            }
            "--ttft-slo-ns" => {
                let n = parse_u64("ttft-slo-ns", &opt_value("ttft-slo-ns", inline, &mut it)?)?;
                if n == 0 {
                    return usage_err("ttft-slo-ns must be > 0");
                }
                ttft_slo_ns = Some(n);
            }
            "--itl-slo-ns" => {
                let n = parse_u64("itl-slo-ns", &opt_value("itl-slo-ns", inline, &mut it)?)?;
                if n == 0 {
                    return usage_err("itl-slo-ns must be > 0");
                }
                itl_slo_ns = Some(n);
            }
            "--trace-out" => {
                trace_out = Some(opt_value("trace-out", inline, &mut it)?);
            }
            flag if flag.starts_with('-') => match planner.take(key, inline, &mut it) {
                Ok(true) => {}
                Ok(false) => return usage_err(&format!("unknown flag {flag}")),
                Err(e) => return usage_err(&e),
            },
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
    check_serve_need(&ServeNeed {
        engine,
        expert_sim,
        expert_8gpu,
        has_slots: expert_slots.is_some(),
        has_bytes: expert_bytes.is_some(),
        has_max: max_seqs.is_some(),
        prefill_chunk,
        decode_first,
        slo_reject,
        has_ttft: ttft_slo_ns.is_some(),
        has_itl: itl_slo_ns.is_some(),
        has_trace: trace_out.is_some(),
        plan: planner,
    })?;
    planner.gpu.imply_vmm();
    planner.gpu.imply_decode_priority();
    let fill = planner.gpu.fill()?;
    let gpu_cfg = gpu_knobs(planner.gpu);
    Ok(ServeCmd::Run(ServeArgs {
        path,
        n_predict,
        n_ctx,
        kv_page,
        bind,
        engine,
        max_seqs,
        expert_slots,
        expert_sim,
        expert_8gpu,
        expert_bytes,
        prefill_chunk,
        decode_first,
        slo_reject,
        ttft_slo_ns,
        itl_slo_ns,
        gpu_cfg,
        kv_bytes: planner.gpu.kv_bytes,
        fill,
        prefetch: planner.prefetch,
        plan_window: planner.plan_window,
        plan_threshold: planner.plan_threshold,
        trace_out,
    }))
}

/// Bind IPv4 loopback. Host must be `127.0.0.1` or `localhost`.
fn parse_bind(spec: &str) -> Result<(Ipv4Addr, u16), String> {
    let (host, port_s) = spec
        .rsplit_once(':')
        .ok_or_else(|| format!("bind needs HOST:PORT, got {spec}\n{SERVE_USAGE}"))?;
    if host.is_empty() {
        return usage_err("bind host is empty");
    }
    if host == "0.0.0.0" || host == "*" || host == "::" || host == "[::]" {
        return usage_err("bind must be localhost (127.0.0.1), not a public interface");
    }
    if !(host.eq_ignore_ascii_case("127.0.0.1") || host.eq_ignore_ascii_case("localhost")) {
        return usage_err(&format!("bind must be localhost, got {host}"));
    }
    let port: u16 = port_s
        .parse()
        .map_err(|_| format!("invalid port {port_s:?}\n{SERVE_USAGE}"))?;
    Ok((Ipv4Addr::LOCALHOST, port))
}

/// Listen on loopback. `spec` is `HOST:PORT` (`localhost` → `127.0.0.1`).
pub(crate) fn bind_loopback(spec: &str) -> Result<TcpListener, ServeError> {
    let (ip, port) = parse_bind(spec).map_err(ServeError::Bind)?;
    TcpListener::bind((ip, port)).map_err(|e| ServeError::Bind(format!("bind {ip}:{port}: {e}")))
}

/// Open a GGUF from `path` with `File` + `Read` (not `std::fs::read`).
fn read_gguf_path(path: &str) -> Result<Vec<u8>, ServeError> {
    let p = Path::new(path);
    let mut f = File::open(p)
        .map_err(|e| ServeError::MissingFile(format!("missing GGUF file {path}: {e}")))?;
    let mut buf = Vec::new();
    let _n = f.read_to_end(&mut buf)?;
    Ok(buf)
}

/// Load the GGUF, bind loopback, serve one HTTP/1.1 request at a time.
pub fn run_serve(args: &ServeArgs) -> Result<(), ServeError> {
    let bytes = read_gguf_path(&args.path)?;
    let g = load_gguf_owned(bytes).map_err(|e| ServeError::Infer(e.to_string()))?;
    let tok = Tokenizer::from_gguf(&g).map_err(|e| ServeError::Infer(e.to_string()))?;
    let model = Llama::from_gguf(g).map_err(|e| ServeError::Infer(e.to_string()))?;
    let listener = bind_loopback(&args.bind)?;
    let addr = listener.local_addr()?;
    println!("listening {addr}");
    println!("model={} n_predict={}", args.path, args.n_predict);
    if args.engine {
        println!("engine continuous batch; loopback only");
        listener.set_nonblocking(true)?;
        return serve_engine::run_loop(&model, &tok, args, listener);
    }
    println!("one request at a time; loopback only; persistent KV prefix reuse");
    let mut cache = None;
    loop {
        let (mut stream, _peer) = listener.accept()?;
        stream.set_read_timeout(Some(Duration::from_secs(IO_TIMEOUT_SECS)))?;
        stream.set_write_timeout(Some(Duration::from_secs(IO_TIMEOUT_SECS)))?;
        if let Err(e) = handle_connection(&mut stream, &model, &tok, args, &mut cache) {
            eprintln!("{e}");
        }
    }
}

#[derive(Debug)]
pub(crate) struct HttpRequest {
    pub(crate) method: String,
    pub(crate) path: String,
    pub(crate) body: Vec<u8>,
}

#[derive(Debug)]
pub(crate) struct GenReq {
    prompt: Option<String>,
    messages: Option<Vec<ChatMessage>>,
    add_generation_prompt: bool,
    pub(crate) n_predict: Option<usize>,
    pub(crate) stream: bool,
}

impl GenReq {
    /// The text to prefill: either the raw `prompt`, or `messages` rendered
    /// through the model's own `tokenizer.chat_template`.
    pub(crate) fn resolve(&self, tok: &Tokenizer) -> Result<String, String> {
        match (&self.prompt, &self.messages) {
            (Some(_), Some(_)) => Err("send either prompt or messages, not both".into()),
            (Some(p), None) => Ok(p.clone()),
            (None, Some(m)) => {
                if m.is_empty() {
                    return Err("messages is empty".into());
                }
                tok.apply_chat_template(m, self.add_generation_prompt)
                    .map_err(|e| e.to_string())
            }
            (None, None) => Err("missing prompt".into()),
        }
    }
}

/// Read one HTTP/1.1 request, generate, write one response, close.
fn handle_connection<S: Read + Write>(
    stream: &mut S,
    model: &Llama,
    tok: &Tokenizer,
    args: &ServeArgs,
    cache: &mut Option<KvCache>,
) -> Result<(), ServeError> {
    match read_request(stream) {
        Ok(req) => {
            let (status, reason, body) = dispatch(&req, model, tok, args, cache);
            write_http_json(stream, status, reason, &body)
        }
        Err(ServeError::Io(_)) => Ok(()),
        Err(e) => {
            let (status, reason) = http_err_status(&e);
            let body = json_error(&e.to_string());
            match write_http_json(stream, status, reason, &body) {
                Ok(()) | Err(ServeError::Io(_)) => Ok(()),
                Err(werr) => Err(werr),
            }
        }
    }
}

pub(crate) fn http_err_status(e: &ServeError) -> (u16, &'static str) {
    match e {
        ServeError::Http(m) if m.contains("too large") => (413, "Payload Too Large"),
        _ => (400, "Bad Request"),
    }
}

fn dispatch(
    req: &HttpRequest,
    model: &Llama,
    tok: &Tokenizer,
    args: &ServeArgs,
    cache: &mut Option<KvCache>,
) -> (u16, &'static str, String) {
    if req.method != "POST" {
        return (405, "Method Not Allowed", json_error("method must be POST"));
    }
    if normalize_path(&req.path) != "/generate" {
        return (404, "Not Found", json_error("not found"));
    }
    let body = match std::str::from_utf8(&req.body) {
        Ok(s) => s,
        Err(_) => return (400, "Bad Request", json_error("body must be utf-8")),
    };
    let gen = match parse_gen_req(body) {
        Ok(g) => g,
        Err(e) => return (400, "Bad Request", json_error(&e)),
    };
    let prompt = match gen.resolve(tok) {
        Ok(p) => p,
        Err(e) => return (400, "Bad Request", json_error(&e)),
    };
    if prompt.is_empty() {
        return (400, "Bad Request", json_error("empty prompt"));
    }
    let n_predict = gen.n_predict.unwrap_or(args.n_predict);
    match greedy_generate_slot(
        model,
        tok,
        cache,
        &prompt,
        n_predict,
        args.n_ctx,
        args.kv_page,
    ) {
        Ok(text) => {
            let hit = cache.as_ref().map_or(0, KvCache::last_prefix_hit);
            let pages = cache.as_ref().map_or(0, KvCache::page_hits);
            (200, "OK", json_generated(&text, hit, pages))
        }
        Err(LlamaError::EmptyPrompt) => (400, "Bad Request", json_error("empty prompt")),
        Err(e) => (500, "Internal Server Error", json_error(&e.to_string())),
    }
}

pub(crate) fn normalize_path(path: &str) -> &str {
    path.strip_suffix('/')
        .filter(|p| !p.is_empty())
        .unwrap_or(path)
}

fn read_request<R: Read>(reader: &mut R) -> Result<HttpRequest, ServeError> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 512];
    loop {
        let n = reader.read(&mut tmp)?;
        if n == 0 {
            return match try_parse_http_request(&buf)? {
                Some(r) => Ok(r),
                None => Err(ServeError::Http("unexpected eof".into())),
            };
        }
        let chunk = tmp
            .get(..n)
            .ok_or_else(|| ServeError::Http("read slice".into()))?;
        buf.extend_from_slice(chunk);
        if u64::try_from(buf.len()).unwrap_or(u64::MAX) > MAX_REQ {
            return Err(ServeError::Http("request too large".into()));
        }
        if let Some(r) = try_parse_http_request(&buf)? {
            return Ok(r);
        }
    }
}

pub(crate) fn try_parse_http_request(buf: &[u8]) -> Result<Option<HttpRequest>, ServeError> {
    let Some((header_bytes, body_so_far)) = header_body_split(buf) else {
        return Ok(None);
    };
    let headers = std::str::from_utf8(header_bytes)
        .map_err(|_| ServeError::Http("headers must be utf-8".into()))?;
    if let Some(te) = header_value(headers, "transfer-encoding") {
        if !te.eq_ignore_ascii_case("identity") {
            return Err(ServeError::Http("chunked transfer unsupported".into()));
        }
    }
    let cl = match header_value(headers, "content-length") {
        Some(s) => s
            .parse::<u64>()
            .map_err(|_| ServeError::Http(format!("invalid content-length {s:?}")))?,
        None => 0,
    };
    if cl > MAX_REQ {
        return Err(ServeError::Http("request too large".into()));
    }
    let need = usize::try_from(cl).map_err(|_| ServeError::Http("content-length".into()))?;
    if body_so_far.len() < need {
        return Ok(None);
    }
    let body = body_so_far
        .get(..need)
        .ok_or_else(|| ServeError::Http("body slice".into()))?
        .to_vec();
    let mut lines = headers.split('\n');
    let req_line = lines.next().unwrap_or("").strip_suffix('\r').unwrap_or("");
    let mut parts = req_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| ServeError::Http("missing method".into()))?;
    let path = parts
        .next()
        .ok_or_else(|| ServeError::Http("missing path".into()))?;
    let version = parts
        .next()
        .ok_or_else(|| ServeError::Http("missing version".into()))?;
    if !version.starts_with("HTTP/") {
        return Err(ServeError::Http(format!("bad version {version}")));
    }
    Ok(Some(HttpRequest {
        method: method.to_string(),
        path: path.to_string(),
        body,
    }))
}

fn header_body_split(buf: &[u8]) -> Option<(&[u8], &[u8])> {
    if let Some(i) = find_subslice(buf, b"\r\n\r\n") {
        let headers = buf.get(..i)?;
        let body = buf.get(i.saturating_add(4)..)?;
        return Some((headers, body));
    }
    if let Some(i) = find_subslice(buf, b"\n\n") {
        let headers = buf.get(..i)?;
        let body = buf.get(i.saturating_add(2)..)?;
        return Some((headers, body));
    }
    None
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

fn header_value(headers: &str, name: &str) -> Option<String> {
    let mut lines = headers.split('\n');
    let _req = lines.next();
    for line in lines {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if let Some((k, v)) = line.split_once(':') {
            if k.eq_ignore_ascii_case(name) {
                return Some(v.trim().to_string());
            }
        }
    }
    None
}

fn write_http_json<W: Write>(
    w: &mut W,
    status: u16,
    reason: &str,
    json: &str,
) -> Result<(), ServeError> {
    w.write_all(&http_json_bytes(status, reason, json))?;
    w.flush()?;
    Ok(())
}

/// HTTP/1.1 JSON response bytes (`Connection: close`).
pub(crate) fn http_json_bytes(status: u16, reason: &str, json: &str) -> Vec<u8> {
    let len = json.len();
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n"
    );
    let mut out = head.into_bytes();
    out.extend_from_slice(json.as_bytes());
    out
}

pub(crate) fn json_generated(text: &str, prefix_hit: usize, page_hits: u64) -> String {
    let mut s = String::from("{\"generated\":");
    append_json_string(&mut s, text);
    s.push_str(",\"prefix_hit\":");
    s.push_str(&prefix_hit.to_string());
    s.push_str(",\"page_hits\":");
    s.push_str(&page_hits.to_string());
    s.push('}');
    s
}

/// One NDJSON token object plus a trailing newline.
pub(crate) fn json_token(token: &str) -> String {
    let mut s = String::from("{\"token\":");
    append_json_string(&mut s, token);
    s.push_str("}\n");
    s
}

/// HTTP/1.1 chunked NDJSON headers (`Connection: close`).
pub(crate) fn http_chunked_headers() -> Vec<u8> {
    b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        .to_vec()
}

/// One HTTP/1.1 chunk: hex size, payload, CRLF.
pub(crate) fn http_chunk(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(format!("{:x}\r\n", payload.len()).as_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(b"\r\n");
    out
}

/// Last HTTP/1.1 chunk (`0` + CRLF + trailer CRLF).
pub(crate) fn http_chunk_end() -> Vec<u8> {
    b"0\r\n\r\n".to_vec()
}

pub(crate) fn json_error(msg: &str) -> String {
    let mut s = String::from("{\"error\":");
    append_json_string(&mut s, msg);
    s.push('}');
    s
}

fn append_json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 32 => {
                let n = u32::from(c);
                out.push_str(&format!("\\u{n:04x}"));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Scan<'a> {
    s: &'a str,
    i: usize,
}

impl Scan<'_> {
    fn rest(&self) -> &str {
        self.s.get(self.i..).unwrap_or("")
    }

    fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.i = self.i.saturating_add(c.len_utf8());
        Some(c)
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_ascii_whitespace()) {
            let _c = self.bump();
        }
    }

    fn expect_char(&mut self, want: char) -> Result<(), String> {
        self.skip_ws();
        match self.bump() {
            Some(c) if c == want => Ok(()),
            Some(c) => Err(format!("expected {want:?}, got {c:?}")),
            None => Err(format!("expected {want:?}")),
        }
    }

    fn expect_lit(&mut self, lit: &str) -> Result<(), String> {
        for want in lit.chars() {
            match self.bump() {
                Some(c) if c == want => {}
                _ => return Err(format!("expected {lit:?}")),
            }
        }
        Ok(())
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.skip_ws();
        if self.bump() != Some('"') {
            return Err("expected string".into());
        }
        let mut out = String::new();
        loop {
            match self.bump() {
                None => return Err("unterminated string".into()),
                Some('"') => return Ok(out),
                Some('\\') => match self.bump() {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('/') => out.push('/'),
                    Some('n') => out.push('\n'),
                    Some('r') => out.push('\r'),
                    Some('t') => out.push('\t'),
                    Some(c) => return Err(format!("bad escape {c}")),
                    None => return Err("unterminated escape".into()),
                },
                Some(c) if u32::from(c) < 32 => return Err("raw control in string".into()),
                Some(c) => out.push(c),
            }
        }
    }

    fn parse_usize(&mut self) -> Result<usize, String> {
        self.skip_ws();
        if self.peek() == Some('-') {
            return Err("n_predict must be >= 0".into());
        }
        let start = self.i;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            let _c = self.bump();
        }
        let digits = self.s.get(start..self.i).unwrap_or("");
        if digits.is_empty() {
            return Err("expected number".into());
        }
        if self.peek() == Some('.') {
            return Err("n_predict must be an integer".into());
        }
        digits
            .parse::<usize>()
            .map_err(|_| format!("invalid n_predict {digits:?}"))
    }

    fn parse_bool(&mut self) -> Result<bool, String> {
        self.skip_ws();
        match self.peek() {
            Some('t') => self.expect_lit("true").map(|()| true),
            Some('f') => self.expect_lit("false").map(|()| false),
            _ => Err("expected true or false".into()),
        }
    }

    /// `[{"role": "...", "content": "..."}, ...]`, the OpenAI message shape.
    ///
    /// Unknown keys inside a message are skipped so a client can send the
    /// fields this engine has no use for. Nested objects and arrays inside a
    /// message are still rejected, same as at the top level.
    fn parse_messages(&mut self) -> Result<Vec<ChatMessage>, String> {
        self.expect_char('[')?;
        let mut out = Vec::new();
        let mut need_comma = false;
        loop {
            self.skip_ws();
            if self.peek() == Some(']') {
                let _c = self.bump();
                return Ok(out);
            }
            if need_comma {
                self.expect_char(',')?;
                self.skip_ws();
                if self.peek() == Some(']') {
                    return Err("trailing comma in messages".into());
                }
            }
            out.push(self.parse_message()?);
            need_comma = true;
        }
    }

    fn parse_message(&mut self) -> Result<ChatMessage, String> {
        self.expect_char('{')?;
        let mut role = None;
        let mut content = None;
        let mut need_comma = false;
        loop {
            self.skip_ws();
            if self.peek() == Some('}') {
                let _c = self.bump();
                break;
            }
            if need_comma {
                self.expect_char(',')?;
                self.skip_ws();
                if self.peek() == Some('}') {
                    return Err("trailing comma in message".into());
                }
            }
            let key = self.parse_string()?;
            self.expect_char(':')?;
            match key.as_str() {
                "role" => role = Some(self.parse_string()?),
                "content" => content = Some(self.parse_string()?),
                _ => self.skip_value()?,
            }
            need_comma = true;
        }
        let role = role.ok_or_else(|| "message missing role".to_string())?;
        let content = content.ok_or_else(|| "message missing content".to_string())?;
        Ok(ChatMessage::new(role, content))
    }

    fn skip_value(&mut self) -> Result<(), String> {
        self.skip_ws();
        match self.peek() {
            Some('"') => {
                let _s = self.parse_string()?;
                Ok(())
            }
            Some(c) if c == '-' || c.is_ascii_digit() => {
                if c == '-' {
                    let _c = self.bump();
                }
                if !matches!(self.peek(), Some(d) if d.is_ascii_digit()) {
                    return Err("expected number".into());
                }
                while matches!(self.peek(), Some(d) if d.is_ascii_digit()) {
                    let _c = self.bump();
                }
                if self.peek() == Some('.') {
                    return Err("floats unsupported".into());
                }
                Ok(())
            }
            Some('t') => self.expect_lit("true"),
            Some('f') => self.expect_lit("false"),
            Some('n') => self.expect_lit("null"),
            Some('{') | Some('[') => Err("nested json unsupported".into()),
            _ => Err("expected value".into()),
        }
    }
}

pub(crate) fn parse_gen_req(s: &str) -> Result<GenReq, String> {
    let mut scan = Scan { s, i: 0 };
    scan.expect_char('{')?;
    let mut prompt = None;
    let mut messages = None;
    let mut add_generation_prompt = None;
    let mut n_predict = None;
    let mut stream = None;
    let mut need_comma = false;
    loop {
        scan.skip_ws();
        if scan.peek() == Some('}') {
            let _c = scan.bump();
            break;
        }
        if need_comma {
            scan.expect_char(',')?;
            scan.skip_ws();
            if scan.peek() == Some('}') {
                return Err("trailing comma".into());
            }
        }
        let key = scan.parse_string()?;
        scan.expect_char(':')?;
        match key.as_str() {
            "prompt" => prompt = Some(scan.parse_string()?),
            "messages" => messages = Some(scan.parse_messages()?),
            "add_generation_prompt" => add_generation_prompt = Some(scan.parse_bool()?),
            "n_predict" => n_predict = Some(scan.parse_usize()?),
            "stream" => stream = Some(scan.parse_bool()?),
            _ => scan.skip_value()?,
        }
        need_comma = true;
    }
    scan.skip_ws();
    if scan.peek().is_some() {
        return Err("trailing junk after json".into());
    }
    Ok(GenReq {
        prompt,
        messages,
        // A chat request is asking the model to speak next, so the generation
        // prompt is on unless the caller says otherwise.
        add_generation_prompt: add_generation_prompt.unwrap_or(true),
        n_predict,
        stream: stream.unwrap_or(false),
    })
}

fn usage_err<T>(msg: &str) -> Result<T, String> {
    Err(format!("{msg}\n{SERVE_USAGE}"))
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
        .map_err(|_| format!("invalid {name} {s:?}\n{SERVE_USAGE}"))
}

fn parse_u64(name: &str, s: &str) -> Result<u64, String> {
    s.parse::<u64>()
        .map_err(|_| format!("invalid {name} {s:?}\n{SERVE_USAGE}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{greedy_generate_ctx, greedy_generate_slot, tiny_llama_gguf};
    use std::io::Cursor;
    use std::net::{IpAddr, Ipv4Addr};

    struct RwBuf {
        input: Cursor<Vec<u8>>,
        output: Vec<u8>,
    }

    impl Read for RwBuf {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.input.read(buf)
        }
    }

    impl Write for RwBuf {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.output.write(buf)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn run(args: &[&str]) -> ServeArgs {
        match parse_serve_args(args).expect("parse") {
            ServeCmd::Run(a) => a,
            ServeCmd::Help => panic!("expected Run"),
        }
    }

    fn tiny_model() -> (Llama, Tokenizer) {
        let g = load_gguf_owned(tiny_llama_gguf()).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("model");
        (model, tok)
    }

    fn defaults() -> ServeArgs {
        ServeArgs {
            path: "tiny.gguf".into(),
            n_predict: InferArgs::DEFAULT_N_PREDICT,
            n_ctx: None,
            kv_page: None,
            bind: ServeArgs::DEFAULT_BIND.into(),
            engine: false,
            max_seqs: None,
            expert_slots: None,
            expert_sim: false,
            expert_8gpu: false,
            expert_bytes: None,
            prefill_chunk: 0,
            decode_first: false,
            slo_reject: false,
            ttft_slo_ns: None,
            itl_slo_ns: None,
            gpu_cfg: GpuStoreCfg::default(),
            kv_bytes: None,
            fill: GpuFill::Pinned,
            prefetch: Prefetch::Both,
            plan_window: 0,
            plan_threshold: 500,
            trace_out: None,
        }
    }

    fn exchange(raw: &str, model: &Llama, tok: &Tokenizer, args: &ServeArgs) -> (String, String) {
        let mut cache = None;
        exchange_on(raw, model, tok, args, &mut cache)
    }

    fn exchange_on(
        raw: &str,
        model: &Llama,
        tok: &Tokenizer,
        args: &ServeArgs,
        cache: &mut Option<KvCache>,
    ) -> (String, String) {
        let mut sock = RwBuf {
            input: Cursor::new(raw.as_bytes().to_vec()),
            output: Vec::new(),
        };
        handle_connection(&mut sock, model, tok, args, cache).expect("handle");
        let (head, body) = header_body_split(&sock.output).expect("split");
        let head = std::str::from_utf8(head).expect("head utf8").to_string();
        let body = std::str::from_utf8(body).expect("body utf8").to_string();
        (head, body)
    }

    fn post_json(json: &str) -> String {
        format!(
            "POST /generate HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: {}\r\n\r\n{json}",
            json.len()
        )
    }

    fn json_field_string(s: &str, want: &str) -> Result<String, String> {
        let mut scan = Scan { s, i: 0 };
        scan.expect_char('{')?;
        let mut need_comma = false;
        loop {
            scan.skip_ws();
            if scan.peek() == Some('}') {
                break;
            }
            if need_comma {
                scan.expect_char(',')?;
            }
            let key = scan.parse_string()?;
            scan.expect_char(':')?;
            if key == want {
                return scan.parse_string();
            }
            scan.skip_value()?;
            need_comma = true;
        }
        Err(format!("missing {want}"))
    }

    #[test]
    fn omitted_flags_keep_infer_n_predict_and_loopback() {
        let a = run(&["tiny.gguf"]);
        assert_eq!(a.path, "tiny.gguf");
        assert_eq!(a.n_predict, InferArgs::DEFAULT_N_PREDICT);
        assert_eq!(a.n_predict, 2);
        assert_eq!(a.n_ctx, None);
        assert_eq!(a.kv_page, None);
        assert_eq!(a.bind, ServeArgs::DEFAULT_BIND);
        assert_eq!(a.bind, "127.0.0.1:8080");
        assert!(!a.engine);
        assert_eq!(a.max_seqs, None);
        assert_eq!(a.prefill_chunk, 0);
        assert!(!a.decode_first);
        assert!(!a.slo_reject);
        assert_eq!(a.ttft_slo_ns, None);
        assert_eq!(a.itl_slo_ns, None);
        assert_eq!(a.gpu_cfg, GpuStoreCfg::default());
        assert_eq!(a.prefetch, Prefetch::Both);
        assert_eq!(a.plan_window, 0);
        assert_eq!(a.plan_threshold, 500);
        assert_eq!(a.trace_out, None);
    }

    #[test]
    fn flags_and_path_after() {
        let a = run(&[
            "--n-predict",
            "4",
            "--n-ctx",
            "16",
            "--bind",
            "localhost:0",
            "m.gguf",
        ]);
        assert_eq!(
            a,
            ServeArgs {
                path: "m.gguf".into(),
                n_predict: 4,
                n_ctx: Some(16),
                kv_page: None,
                bind: "localhost:0".into(),
                engine: false,
                max_seqs: None,
                expert_slots: None,
                expert_sim: false,
                expert_8gpu: false,
                expert_bytes: None,
                prefill_chunk: 0,
                decode_first: false,
                slo_reject: false,
                ttft_slo_ns: None,
                itl_slo_ns: None,
                gpu_cfg: GpuStoreCfg::default(),
                kv_bytes: None,
                fill: GpuFill::Pinned,
                prefetch: Prefetch::Both,
                plan_window: 0,
                plan_threshold: 500,
                trace_out: None,
            }
        );
    }

    #[test]
    fn short_flags_equals_and_path_first() {
        let a = run(&[
            "model.gguf",
            "-n=0",
            "--n-ctx=8",
            "--kv-page=2",
            "--bind=127.0.0.1:9",
        ]);
        assert_eq!(a.n_predict, 0);
        assert_eq!(a.n_ctx, Some(8));
        assert_eq!(a.kv_page, Some(2));
        assert_eq!(a.bind, "127.0.0.1:9");
    }

    #[test]
    fn help_is_not_a_run() {
        assert_eq!(parse_serve_args(["--help"]).unwrap(), ServeCmd::Help);
        assert_eq!(parse_serve_args(["-h", "x.gguf"]).unwrap(), ServeCmd::Help);
    }

    #[test]
    fn missing_path_and_bad_values_error() {
        let err = parse_serve_args(["--n-predict", "2"]).unwrap_err();
        assert!(err.contains("missing GGUF path"), "{err}");
        assert!(err.contains("gguf_gemv serve"), "{err}");
        let err = parse_serve_args(["m.gguf", "-n", "x"]).unwrap_err();
        assert!(err.contains("invalid n-predict"), "{err}");
        let err = parse_serve_args(["m.gguf", "--n-ctx", "0"]).unwrap_err();
        assert!(err.contains("n-ctx must be > 0"), "{err}");
        let err = parse_serve_args(["m.gguf", "--kv-page", "0"]).unwrap_err();
        assert!(err.contains("kv-page must be > 0"), "{err}");
        let err = parse_serve_args(["m.gguf", "--nope"]).unwrap_err();
        assert!(err.contains("unknown flag"), "{err}");
        let err = parse_serve_args(["a.gguf", "b.gguf"]).unwrap_err();
        assert!(err.contains("unexpected argument"), "{err}");
        let err = parse_serve_args(["m.gguf", "--bind", "0.0.0.0:80"]).unwrap_err();
        assert!(err.contains("localhost"), "{err}");
        let err = parse_serve_args(["--bind"]).unwrap_err();
        assert!(err.contains("missing --bind value"), "{err}");
        let err = parse_serve_args(["m.gguf", "--max-seqs", "2"]).unwrap_err();
        assert!(err.contains("--max-seqs requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--max-seqs", "0"]).unwrap_err();
        assert!(err.contains("max-seqs must be > 0"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine=1"]).unwrap_err();
        assert!(err.contains("--engine does not take a value"), "{err}");
    }

    #[test]
    fn engine_flag_and_max_seqs() {
        let a = run(&["tiny.gguf", "--engine"]);
        assert!(a.engine);
        assert_eq!(a.max_seqs, None);
        let a = run(&["--engine", "--max-seqs", "8", "m.gguf"]);
        assert_eq!(
            a,
            ServeArgs {
                path: "m.gguf".into(),
                n_predict: InferArgs::DEFAULT_N_PREDICT,
                n_ctx: None,
                kv_page: None,
                bind: ServeArgs::DEFAULT_BIND.into(),
                engine: true,
                max_seqs: Some(8),
                expert_slots: None,
                expert_sim: false,
                expert_8gpu: false,
                expert_bytes: None,
                prefill_chunk: 0,
                decode_first: false,
                slo_reject: false,
                ttft_slo_ns: None,
                itl_slo_ns: None,
                gpu_cfg: GpuStoreCfg::default(),
                kv_bytes: None,
                fill: GpuFill::Pinned,
                prefetch: Prefetch::Both,
                plan_window: 0,
                plan_threshold: 500,
                trace_out: None,
            }
        );
        let a = run(&["m.gguf", "--engine", "--max-seqs=2"]);
        assert_eq!(a.max_seqs, Some(2));
        let a = run(&["m.gguf", "--engine", "--expert-slots", "0"]);
        assert_eq!(a.expert_slots, Some(0));
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--expert-8gpu",
            "--expert-bytes=8192",
        ]);
        assert!(a.expert_sim);
        assert!(a.expert_8gpu);
        assert_eq!(a.expert_bytes, Some(8192));
    }

    #[test]
    fn expert_store_flags_need_engine() {
        let err = parse_serve_args(["m.gguf", "--expert-slots", "0"]).unwrap_err();
        assert!(err.contains("--expert-slots requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--expert-sim"]).unwrap_err();
        assert!(err.contains("--expert-sim requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-8gpu"]).unwrap_err();
        assert!(err.contains("--expert-8gpu requires --expert-sim"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-bytes", "4"]).unwrap_err();
        assert!(
            err.contains("--expert-bytes requires --expert-sim"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-bytes", "0"]).unwrap_err();
        assert!(err.contains("expert-bytes must be > 0"), "{err}");
    }

    #[test]
    fn serve_prefetch_planner_needs_engine() {
        let err = parse_serve_args(["m.gguf", "--prefetch", "none"]).unwrap_err();
        assert!(err.contains("--prefetch requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--plan-window", "8"]).unwrap_err();
        assert!(err.contains("--plan-window requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--plan-threshold", "100"]).unwrap_err();
        assert!(err.contains("--plan-threshold requires --engine"), "{err}");
        let a = run(&["m.gguf", "--engine", "--prefetch=none"]);
        assert_eq!(a.prefetch, Prefetch::None);
        assert!(!a.expert_sim);
        let a = run(&[
            "m.gguf",
            "--engine",
            "--prefetch=copy-forward",
            "--plan-window=8",
            "--plan-threshold=100",
        ]);
        assert_eq!(a.prefetch, Prefetch::CopyForward);
        assert_eq!(a.plan_window, 8);
        assert_eq!(a.plan_threshold, 100);
        let err = parse_serve_args(["m.gguf", "--engine", "--prefetch", "nope"]).unwrap_err();
        assert!(err.contains("unknown prefetch"), "{err}");
    }

    #[test]
    fn prefill_chunk_and_trace_out_need_engine() {
        let err = parse_serve_args(["m.gguf", "--prefill-chunk", "1"]).unwrap_err();
        assert!(err.contains("--prefill-chunk requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--trace-out", "t.jsonl"]).unwrap_err();
        assert!(err.contains("--trace-out requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--decode-first"]).unwrap_err();
        assert!(err.contains("--decode-first requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--decode-first=1"]).unwrap_err();
        assert!(
            err.contains("--decode-first does not take a value"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--prefill-chunk=1",
            "--decode-first",
            "--trace-out=t.jsonl",
        ]);
        assert_eq!(a.prefill_chunk, 1);
        assert!(a.decode_first);
        assert_eq!(a.trace_out.as_deref(), Some("t.jsonl"));
        let err = parse_serve_args(["m.gguf", "--slo-reject"]).unwrap_err();
        assert!(err.contains("--slo-reject requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--slo-reject"]).unwrap_err();
        assert!(err.contains("--slo-reject requires --ttft-slo-ns"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--slo-reject", "--ttft-slo-ns=1"])
            .unwrap_err();
        assert!(err.contains("--slo-reject requires --expert-sim"), "{err}");
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--slo-reject",
            "--ttft-slo-ns=8",
        ]);
        assert!(a.slo_reject);
        assert_eq!(a.ttft_slo_ns, Some(8));
        let err = parse_serve_args(["m.gguf", "--itl-slo-ns", "1"]).unwrap_err();
        assert!(err.contains("--itl-slo-ns requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--itl-slo-ns", "1"]).unwrap_err();
        assert!(err.contains("--itl-slo-ns requires --expert-sim"), "{err}");
        let err = parse_serve_args(["m.gguf", "--graph-update"]).unwrap_err();
        assert!(err.contains("--graph-update requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--graph-update"]).unwrap_err();
        assert!(
            err.contains("--graph-update requires --expert-sim"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--cuda-graphs",
            "--graph-update",
            "--graph-clone",
            "--timing-events",
            "--itl-slo-ns=4",
        ]);
        assert!(a.gpu_cfg.graph_update);
        assert!(a.gpu_cfg.graph_clone);
        assert!(a.gpu_cfg.timing_events);
        assert_eq!(a.itl_slo_ns, Some(4));
        let err = parse_serve_args(["m.gguf", "--managed"]).unwrap_err();
        assert!(err.contains("--managed requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--mapped"]).unwrap_err();
        assert!(err.contains("--mapped requires --expert-sim"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--vmm"]);
        assert_eq!(a.fill, GpuFill::Vmm);
        let err = parse_serve_args(["m.gguf", "--host-func"]).unwrap_err();
        assert!(err.contains("--host-func requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--host-func"]).unwrap_err();
        assert!(err.contains("--host-func requires --expert-sim"), "{err}");
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--host-func",
            "--vmm-page=1024",
            "--stream-priority",
        ]);
        assert!(a.gpu_cfg.host_func);
        assert!(a.gpu_cfg.stream_priority);
        assert_eq!(a.gpu_cfg.vmm_page, 1024);
        assert_eq!(a.fill, GpuFill::Vmm);
        let err = parse_serve_args(["m.gguf", "--seq-streams"]).unwrap_err();
        assert!(err.contains("--seq-streams requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--seq-streams"]).unwrap_err();
        assert!(err.contains("--seq-streams requires --expert-sim"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--seq-streams"]);
        assert!(a.gpu_cfg.seq_streams);
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--seq-streams=1"])
            .unwrap_err();
        assert!(err.contains("--seq-streams does not take a value"), "{err}");
        let err = parse_serve_args(["m.gguf", "--kv-sim"]).unwrap_err();
        assert!(err.contains("--kv-sim requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--kv-sim"]).unwrap_err();
        assert!(err.contains("--kv-sim requires --expert-sim"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--kv-bytes", "1048576"]).unwrap_err();
        assert!(err.contains("--kv-bytes requires --kv-sim"), "{err}");
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--kv-sim",
            "--kv-bytes=1048576",
        ]);
        assert!(a.gpu_cfg.kv_sim);
        assert_eq!(a.kv_bytes, Some(1_048_576));
        let err =
            parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--kv-sim=1"]).unwrap_err();
        assert!(err.contains("--kv-sim does not take a value"), "{err}");
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--kv-sim",
            "--kv-bytes",
            "0",
        ])
        .unwrap_err();
        assert!(err.contains("kv-bytes must be > 0"), "{err}");
        let err = parse_serve_args(["m.gguf", "--decode-priority"]).unwrap_err();
        assert!(err.contains("--decode-priority requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--decode-priority"]).unwrap_err();
        assert!(
            err.contains("--decode-priority requires --expert-sim"),
            "{err}"
        );
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--decode-priority"]);
        assert!(a.gpu_cfg.decode_priority);
        assert!(a.gpu_cfg.stream_priority);
        let err = parse_serve_args(["m.gguf", "--compute-slots", "2"]).unwrap_err();
        assert!(err.contains("--compute-slots requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--compute-slots", "2"]).unwrap_err();
        assert!(
            err.contains("--compute-slots requires --expert-sim"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--compute-slots", "0"])
            .unwrap_err();
        assert!(err.contains("compute-slots must be > 0"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--compute-slots", "2"]);
        assert_eq!(a.gpu_cfg.compute_slots, 2);
        let err = parse_serve_args(["m.gguf", "--decode-sms", "250"]).unwrap_err();
        assert!(err.contains("--decode-sms requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--decode-sms", "250"]).unwrap_err();
        assert!(err.contains("--decode-sms requires --expert-sim"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--decode-sms", "0"])
            .unwrap_err();
        assert!(err.contains("decode-sms must be 1..=1000"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--decode-sms", "1001"])
            .unwrap_err();
        assert!(err.contains("decode-sms must be 1..=1000"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--decode-sms", "250"]);
        assert_eq!(a.gpu_cfg.decode_sm_permille, 250);
        assert!(a.gpu_cfg.decode_priority);
        assert!(a.gpu_cfg.stream_priority);
    }

    #[test]
    fn bind_parser_allows_loopback_only() {
        assert_eq!(
            parse_bind("127.0.0.1:8080").unwrap(),
            (Ipv4Addr::LOCALHOST, 8080)
        );
        assert_eq!(parse_bind("localhost:0").unwrap(), (Ipv4Addr::LOCALHOST, 0));
        assert_eq!(parse_bind("LOCALHOST:9").unwrap(), (Ipv4Addr::LOCALHOST, 9));
        for bad in [
            "0.0.0.0:8080",
            "1.2.3.4:80",
            "[::]:80",
            "::1:80",
            "*:80",
            "8080",
            ":8080",
        ] {
            let err = parse_bind(bad).unwrap_err();
            assert!(
                err.contains("localhost") || err.contains("HOST:PORT") || err.contains("host"),
                "{bad}: {err}"
            );
        }
    }

    #[test]
    fn bind_loopback_ephemeral_is_127_0_0_1() {
        let listener = bind_loopback("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        assert_eq!(addr.ip(), IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert_ne!(addr.port(), 0);
        let listener2 = bind_loopback("localhost:0").expect("localhost");
        assert_eq!(
            listener2.local_addr().expect("addr2").ip(),
            IpAddr::V4(Ipv4Addr::LOCALHOST)
        );
        let err = bind_loopback("0.0.0.0:8080").expect_err("public");
        assert!(err.to_string().contains("localhost"), "{err}");
    }

    #[test]
    fn missing_gguf_file_fails_before_listen() {
        let err = read_gguf_path("no-such-llama-rust-serve-test.gguf").expect_err("missing");
        assert!(err.to_string().contains("missing GGUF file"), "{err}");
        assert!(
            err.to_string()
                .contains("no-such-llama-rust-serve-test.gguf"),
            "{err}"
        );
    }

    #[test]
    fn http_request_shape_content_length() {
        let raw = b"POST /generate HTTP/1.1\r\nContent-Length: 15\r\n\r\n{\"prompt\":\"ab\"}";
        let req = try_parse_http_request(raw).unwrap().unwrap();
        assert_eq!(req.method, "POST");
        assert_eq!(req.path, "/generate");
        assert_eq!(req.body, br#"{"prompt":"ab"}"#);
        assert!(try_parse_http_request(b"POST /generate HTTP/1.1\r\n")
            .unwrap()
            .is_none());
        assert!(try_parse_http_request(
            b"POST /generate HTTP/1.1\r\nContent-Length: 10\r\n\r\n123"
        )
        .unwrap()
        .is_none());
        let extra = b"POST /generate HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}\x00leftover";
        let req = try_parse_http_request(extra).unwrap().unwrap();
        assert_eq!(req.body, b"{}");
        let err = try_parse_http_request(
            b"POST /generate HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n",
        )
        .unwrap_err();
        assert!(err.to_string().contains("chunked"), "{err}");
        let err =
            try_parse_http_request(b"POST /generate HTTP/1.1\r\nContent-Length: 999999\r\n\r\n")
                .unwrap_err();
        assert!(err.to_string().contains("too large"), "{err}");
    }

    #[test]
    fn json_request_shape() {
        let (_model, tok) = tiny_model();
        let a = parse_gen_req(r#"{"prompt":"ab"}"#).unwrap();
        assert_eq!(a.prompt.as_deref(), Some("ab"));
        assert_eq!(a.n_predict, None);
        assert!(!a.stream);
        assert_eq!(a.resolve(&tok).unwrap(), "ab");
        let streamed = parse_gen_req(r#"{"prompt":"ab","stream":true}"#).unwrap();
        assert!(streamed.stream);
        let unstreamed = parse_gen_req(r#"{"prompt":"ab","stream":false}"#).unwrap();
        assert!(!unstreamed.stream);
        let b = parse_gen_req(r#"{ "prompt" : "ab", "n_predict" : 4 }"#).unwrap();
        assert_eq!(b.prompt.as_deref(), Some("ab"));
        assert_eq!(b.n_predict, Some(4));
        let c = parse_gen_req(r#"{"prompt":"a\"b","extra":true}"#).unwrap();
        assert_eq!(c.prompt.as_deref(), Some("a\"b"));
        assert_eq!(
            parse_gen_req("{}").unwrap().resolve(&tok).unwrap_err(),
            "missing prompt"
        );
        assert_eq!(
            parse_gen_req(r#"{"prompt":""}"#)
                .unwrap()
                .resolve(&tok)
                .unwrap(),
            ""
        );
        assert!(parse_gen_req(r#"{"n_predict":2}"#)
            .unwrap()
            .resolve(&tok)
            .unwrap_err()
            .contains("missing prompt"));
        assert!(parse_gen_req(r#"{"prompt":"ab","n_predict":-1}"#)
            .unwrap_err()
            .contains("n_predict"));
        assert!(parse_gen_req(r#"{"prompt":"ab"} extra"#)
            .unwrap_err()
            .contains("trailing"));
    }

    /// The tiny fixture has no `chat_template` of its own; give it one whose
    /// output lands inside its three-token vocab.
    fn chat_model(template: &str) -> (Llama, Tokenizer) {
        let (model, mut tok) = tiny_model();
        tok.chat_template = Some(template.to_string());
        (model, tok)
    }

    const ECHO_TEMPLATE: &str = "{% for m in messages %}{{ m['content'] }}{% endfor %}";

    #[test]
    fn a_messages_array_renders_the_template_and_matches_the_same_prompt() {
        let (model, tok) = chat_model(ECHO_TEMPLATE);
        let args = defaults();
        let expect = greedy_generate_ctx(&model, &tok, "ab", 2, None).expect("greedy");
        let (head, body) = exchange(
            &post_json(r#"{"messages":[{"role":"user","content":"ab"}]}"#),
            &model,
            &tok,
            &args,
        );
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(json_field_string(&body, "generated").unwrap(), expect);
        // Multi-turn, with the keys in either order and an ignored extra field.
        let (_h, body) = exchange(
            &post_json(
                r#"{"messages":[{"content":"a","role":"user"},{"role":"assistant","name":"x","content":"b"}],"n_predict":2}"#,
            ),
            &model,
            &tok,
            &args,
        );
        assert_eq!(json_field_string(&body, "generated").unwrap(), expect);
    }

    #[test]
    fn add_generation_prompt_defaults_to_true_and_can_be_turned_off() {
        let (model, tok) = chat_model(
            "{% for m in messages %}{{ m.content }}{% endfor %}\
             {% if add_generation_prompt %}b{% endif %}",
        );
        let args = defaults();
        let with = greedy_generate_ctx(&model, &tok, "ab", 2, None).expect("greedy ab");
        let without = greedy_generate_ctx(&model, &tok, "a", 2, None).expect("greedy a");
        let (_h, body) = exchange(
            &post_json(r#"{"messages":[{"role":"user","content":"a"}]}"#),
            &model,
            &tok,
            &args,
        );
        assert_eq!(json_field_string(&body, "generated").unwrap(), with);
        let (_h, body) = exchange(
            &post_json(
                r#"{"messages":[{"role":"user","content":"a"}],"add_generation_prompt":false}"#,
            ),
            &model,
            &tok,
            &args,
        );
        assert_eq!(json_field_string(&body, "generated").unwrap(), without);
    }

    #[test]
    fn chat_requests_that_cannot_be_served_are_bad_requests_not_guesses() {
        let (model, tok) = chat_model(ECHO_TEMPLATE);
        let args = defaults();
        for (json, want) in [
            (
                r#"{"prompt":"ab","messages":[{"role":"user","content":"ab"}]}"#,
                "not both",
            ),
            (r#"{"messages":[]}"#, "messages is empty"),
            (r#"{"messages":[{"role":"user"}]}"#, "missing content"),
            (r#"{"messages":[{"content":"ab"}]}"#, "missing role"),
            (r#"{"messages":"ab"}"#, "expected '['"),
            (
                r#"{"messages":[{"role":"user","content":"ab"},]}"#,
                "trailing comma",
            ),
        ] {
            let (head, body) = exchange(&post_json(json), &model, &tok, &args);
            assert!(head.starts_with("HTTP/1.1 400"), "{json}: {head}");
            let msg = json_field_string(&body, "error").expect("error");
            assert!(msg.contains(want), "{json}: got {msg:?}, want {want:?}");
        }
        // A model with no template at all says so instead of inventing one.
        let (model, tok) = tiny_model();
        let (head, body) = exchange(
            &post_json(r#"{"messages":[{"role":"user","content":"ab"}]}"#),
            &model,
            &tok,
            &args,
        );
        assert!(head.starts_with("HTTP/1.1 400"), "{head}");
        assert!(json_field_string(&body, "error")
            .expect("error")
            .contains("chat_template"));
    }

    #[test]
    fn the_plain_prompt_field_is_untouched_by_the_chat_path() {
        let (model, tok) = chat_model(ECHO_TEMPLATE);
        let args = defaults();
        let expect = greedy_generate_ctx(&model, &tok, "ab", 2, None).expect("greedy");
        let (head, body) = exchange(&post_json(r#"{"prompt":"ab"}"#), &model, &tok, &args);
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(json_field_string(&body, "generated").unwrap(), expect);
    }

    #[test]
    fn json_response_shape_escapes() {
        assert_eq!(
            json_generated("ab", 0, 0),
            r#"{"generated":"ab","prefix_hit":0,"page_hits":0}"#
        );
        assert_eq!(
            json_generated("a\"b", 0, 0),
            r#"{"generated":"a\"b","prefix_hit":0,"page_hits":0}"#
        );
        assert_eq!(
            json_generated("a\nb", 2, 3),
            r#"{"generated":"a\nb","prefix_hit":2,"page_hits":3}"#
        );
        assert_eq!(json_error("empty prompt"), r#"{"error":"empty prompt"}"#);
        assert_eq!(
            json_field_string(&json_generated("a\"b\n", 0, 0), "generated").unwrap(),
            "a\"b\n"
        );
        assert_eq!(json_token("ab"), "{\"token\":\"ab\"}\n");
        assert_eq!(json_token("a\"b"), "{\"token\":\"a\\\"b\"}\n");
        assert_eq!(http_chunk(b"hi"), b"2\r\nhi\r\n");
        assert_eq!(http_chunk_end(), b"0\r\n\r\n");
    }

    #[test]
    fn handle_tiny_generate_matches_greedy_and_http_shape() {
        let (model, tok) = tiny_model();
        let args = defaults();
        let expect = greedy_generate_ctx(&model, &tok, "ab", 2, None).expect("greedy");
        let (head, body) = exchange(&post_json(r#"{"prompt":"ab"}"#), &model, &tok, &args);
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert!(head.contains("Content-Type: application/json"), "{head}");
        assert!(head.contains("Connection: close"), "{head}");
        let cl = header_value(&head, "content-length").expect("cl");
        assert_eq!(cl.parse::<usize>().unwrap(), body.len());
        assert_eq!(json_field_string(&body, "generated").unwrap(), expect);
        let (head_s, body_s) = exchange(
            &post_json(r#"{"prompt":"ab","stream":true}"#),
            &model,
            &tok,
            &args,
        );
        assert!(head_s.starts_with("HTTP/1.1 200 OK"), "{head_s}");
        assert!(head_s.contains("Content-Length"), "{head_s}");
        assert!(!head_s.to_ascii_lowercase().contains("chunked"), "{head_s}");
        assert_eq!(json_field_string(&body_s, "generated").unwrap(), expect);
        let zero = greedy_generate_ctx(&model, &tok, "ab", 0, None).expect("zero");
        let (_h, body0) = exchange(
            &post_json(r#"{"prompt":"ab","n_predict":0}"#),
            &model,
            &tok,
            &args,
        );
        assert_eq!(json_field_string(&body0, "generated").unwrap(), zero);
    }

    #[test]
    fn empty_prompt_and_bad_method_fail_cleanly() {
        let (model, tok) = tiny_model();
        let args = defaults();
        let (head, body) = exchange(&post_json(r#"{"prompt":""}"#), &model, &tok, &args);
        assert!(head.starts_with("HTTP/1.1 400 "), "{head}");
        assert_eq!(json_field_string(&body, "error").unwrap(), "empty prompt");
        let get = "GET /generate HTTP/1.1\r\nContent-Length: 0\r\n\r\n";
        let (head, body) = exchange(get, &model, &tok, &args);
        assert!(head.starts_with("HTTP/1.1 405 "), "{head}");
        assert!(json_field_string(&body, "error").unwrap().contains("POST"));
        let openai =
            "POST /v1/completions HTTP/1.1\r\nContent-Length: 15\r\n\r\n{\"prompt\":\"ab\"}";
        let (head, _body) = exchange(openai, &model, &tok, &args);
        assert!(head.starts_with("HTTP/1.1 404 "), "{head}");
    }

    #[test]
    fn serve_reuses_kv_prefix_across_requests() {
        let (model, tok) = tiny_model();
        let args = ServeArgs {
            n_ctx: Some(16),
            ..defaults()
        };
        let expect = greedy_generate_ctx(&model, &tok, "ab", 2, Some(16)).expect("greedy");
        let mut cache = None;
        let (head, body) = exchange_on(
            &post_json(r#"{"prompt":"ab"}"#),
            &model,
            &tok,
            &args,
            &mut cache,
        );
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(json_field_string(&body, "generated").unwrap(), expect);
        assert!(body.contains("\"prefix_hit\":0"), "{body}");
        let (head2, body2) = exchange_on(
            &post_json(r#"{"prompt":"ab"}"#),
            &model,
            &tok,
            &args,
            &mut cache,
        );
        assert!(head2.starts_with("HTTP/1.1 200 OK"), "{head2}");
        assert_eq!(json_field_string(&body2, "generated").unwrap(), expect);
        assert!(
            !body2.contains("\"prefix_hit\":0"),
            "second request must reuse the prompt prefix: {body2}"
        );
    }

    #[test]
    fn serve_paged_kv_matches_dense_greedy() {
        let (model, tok) = tiny_model();
        let args = ServeArgs {
            n_ctx: Some(16),
            kv_page: Some(2),
            ..defaults()
        };
        let expect =
            greedy_generate_slot(&model, &tok, &mut None, "ab", 2, Some(16), Some(2)).expect("p");
        let dense = greedy_generate_ctx(&model, &tok, "ab", 2, Some(16)).expect("d");
        assert_eq!(expect, dense);
        let mut cache = None;
        let (head, body) = exchange_on(
            &post_json(r#"{"prompt":"ab"}"#),
            &model,
            &tok,
            &args,
            &mut cache,
        );
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(json_field_string(&body, "generated").unwrap(), expect);
        assert!(body.contains("\"page_hits\":"), "{body}");
        let (head2, body2) = exchange_on(
            &post_json(r#"{"prompt":"ab"}"#),
            &model,
            &tok,
            &args,
            &mut cache,
        );
        assert!(head2.starts_with("HTTP/1.1 200 OK"), "{head2}");
        assert_eq!(json_field_string(&body2, "generated").unwrap(), expect);
        assert!(
            cache.as_ref().is_some_and(|c| c.page_size() == Some(2)),
            "persistent slot must stay paged"
        );
    }
}
