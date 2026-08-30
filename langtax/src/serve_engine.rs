//! Concurrent HTTP loop around one [`Engine`].
//!
//! Default `gguf_gemv serve` stays one connection at a time. `--engine` admits
//! several `POST /generate` bodies onto one interned pool so prefills and
//! decodes GEMM together. `"stream": true` is HTTP/1.1 chunked NDJSON token
//! lines then a final `generated` object. `/v1/completions` and
//! `/v1/chat/completions` use the same scheduler with an OpenAI `choices`
//! envelope; stream is chunked SSE. `GET /v1/models`, `GET /health`, and
//! `GET /metrics` (Engine counters) do not admit a sequence. `--prefill-chunk`, `--decode-first`,
//! `--slo-reject`, `--itl-slo-ns`, graph knobs, fill modes, `GpuStoreCfg`
//! CUDA knobs, `--prefetch` / `--plan-window` / `--plan-threshold`, `--seq-streams`, and
//! `--trace-out` match `gguf_gemv engine`. No Tokio.
//! HTTP/1.1 keep-alive is on unless the client sends `Connection: close`.

use std::fs::OpenOptions;
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};

use crate::decode::{prompt_ids, Llama, LlamaError};
use crate::engine::{Engine, EngineCfg, SeqId, SeqOutput};
use crate::serve::{
    dispatch_get, finish_reason, http_chunk, http_chunk_end, http_chunked_headers, http_err_status,
    http_json_bytes, http_sse_headers, json_error, json_generated, json_openai_chat,
    json_openai_completion, json_openai_error, json_token, parse_gen_req, path_only,
    sse_chat_delta, sse_completion_delta, sse_done, try_parse_http_request, HttpRequest, ServeApi,
    ServeArgs, ServeError, MAX_REQ,
};
use crate::store_attach::{attach_store, StoreAttach};
use crate::tok::Tokenizer;

/// Poll until the process exits. Listener must already be non-blocking.
pub(crate) fn run_loop(
    model: &Llama,
    tok: &Tokenizer,
    args: &ServeArgs,
    listener: TcpListener,
) -> Result<(), ServeError> {
    let mut http = EngineHttp::new(model, tok, args, listener)?;
    loop {
        http.poll()?;
    }
}

/// Non-blocking Engine HTTP server (loopback). Tests drive [`EngineHttp::poll`].
pub(crate) struct EngineHttp<'a> {
    tok: &'a Tokenizer,
    n_predict: usize,
    model_id: String,
    engine: Engine<'a>,
    listener: TcpListener,
    conns: Vec<Conn>,
    trace_out: Option<String>,
}

struct Conn {
    stream: TcpStream,
    read: Vec<u8>,
    req: Option<HttpRequest>,
    seq: Option<SeqId>,
    write: Vec<u8>,
    write_at: usize,
    done: bool,
    intern_hits_at_add: u64,
    ndjson: bool,
    streamed: usize,
    finalized: bool,
    keep_alive: bool,
    api: ServeApi,
    n_predict: usize,
}

enum AdmitOut {
    Seq {
        id: SeqId,
        ndjson: bool,
        api: ServeApi,
        n_predict: usize,
    },
    Bytes(Vec<u8>),
}

impl AdmitOut {
    fn http(
        status: u16,
        reason: &'static str,
        msg: &str,
        keep_alive: bool,
        api: Option<ServeApi>,
    ) -> Self {
        let body = if api.is_some_and(ServeApi::is_openai) {
            json_openai_error(msg)
        } else {
            json_error(msg)
        };
        Self::Bytes(http_json_bytes(status, reason, &body, keep_alive))
    }
}

fn io_again(e: &std::io::Error) -> bool {
    matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted)
}

fn json_engine_metrics(engine: &Engine<'_>) -> String {
    let mut s = String::from("{\"engine\":true,\"active\":");
    s.push_str(&engine.active().to_string());
    s.push_str(",\"waiting\":");
    s.push_str(&engine.waiting().to_string());
    s.push_str(",\"preempts\":");
    s.push_str(&engine.preempts().to_string());
    s.push_str(",\"rejected\":");
    s.push_str(&engine.rejected().to_string());
    s.push_str(",\"itl_slo_miss\":");
    s.push_str(&engine.itl_slo_miss().to_string());
    s.push_str(",\"graph_launches\":");
    s.push_str(&engine.graph_launches().to_string());
    s.push_str(",\"steps\":");
    s.push_str(&engine.stats().steps.to_string());
    s.push('}');
    s
}

fn engine_cfg(tok: &Tokenizer, args: &ServeArgs) -> EngineCfg {
    let n_ctx = args.n_ctx.unwrap_or(64);
    let block_size = args.kv_page.unwrap_or(16);
    let max_seqs = args.max_seqs.unwrap_or(4);
    let pages = n_ctx.div_ceil(block_size).saturating_add(1);
    EngineCfg {
        n_ctx,
        block_size,
        pool_blocks: max_seqs.saturating_mul(pages).max(2),
        max_seqs,
        prefill_chunk: args.prefill_chunk,
        eos: tok.eos,
        decode_first: args.decode_first,
        slo_reject: args.slo_reject,
        ttft_slo_ns: args.ttft_slo_ns,
        itl_slo_ns: args.itl_slo_ns,
        prefetch: args.prefetch,
        plan_window: args.plan_window,
        plan_threshold: args.plan_threshold,
    }
}

fn decode_seq(tok: &Tokenizer, out: &SeqOutput) -> String {
    let mut ids = out.prompt.clone();
    ids.extend_from_slice(&out.generated);
    tok.decode(&ids)
}

impl Conn {
    fn new(stream: TcpStream) -> Self {
        Self {
            stream,
            read: Vec::new(),
            req: None,
            seq: None,
            write: Vec::new(),
            write_at: 0,
            done: false,
            intern_hits_at_add: 0,
            ndjson: false,
            streamed: 0,
            finalized: false,
            keep_alive: false,
            api: ServeApi::Generate,
            n_predict: 0,
        }
    }

    fn can_admit(&self) -> bool {
        self.req.is_some() && self.seq.is_none() && self.write.is_empty() && !self.done
    }
}

fn try_take_request(c: &mut Conn) -> bool {
    match try_parse_http_request(&c.read) {
        Ok(Some(req)) => {
            let n = req.consumed.min(c.read.len());
            let rest = c.read.split_off(n);
            c.read = rest;
            c.keep_alive = req.keep_alive;
            c.req = Some(req);
            true
        }
        Ok(None) => false,
        Err(e) => {
            queue_err(c, &e);
            true
        }
    }
}

fn read_conn(c: &mut Conn) {
    if c.done || c.req.is_some() || !c.write.is_empty() {
        return;
    }
    if try_take_request(c) {
        return;
    }
    let mut tmp = [0u8; 512];
    loop {
        match c.stream.read(&mut tmp) {
            Ok(0) => {
                c.done = true;
                return;
            }
            Ok(n) => {
                let Some(chunk) = tmp.get(..n) else {
                    c.done = true;
                    return;
                };
                c.read.extend_from_slice(chunk);
                if u64::try_from(c.read.len()).unwrap_or(u64::MAX) > MAX_REQ {
                    let e = ServeError::Http("request too large".into());
                    queue_err(c, &e);
                    return;
                }
                if try_take_request(c) {
                    return;
                }
            }
            Err(e) if io_again(&e) => return,
            Err(_) => {
                c.done = true;
                return;
            }
        }
    }
}

fn queue_err(c: &mut Conn, e: &ServeError) {
    let (status, reason) = http_err_status(e);
    c.write = http_json_bytes(status, reason, &json_error(&e.to_string()), false);
    c.write_at = 0;
    c.keep_alive = false;
}

fn write_conn(c: &mut Conn) {
    if c.done || c.write.is_empty() {
        return;
    }
    loop {
        let Some(rest) = c.write.get(c.write_at..) else {
            c.done = true;
            return;
        };
        if rest.is_empty() {
            match c.stream.flush() {
                Ok(()) => {
                    c.write.clear();
                    c.write_at = 0;
                    if c.ndjson && !c.finalized {
                        return;
                    }
                    finish_write(c);
                    return;
                }
                Err(e) if io_again(&e) => return,
                Err(_) => {
                    c.done = true;
                    return;
                }
            }
        }
        match c.stream.write(rest) {
            Ok(0) => {
                c.done = true;
                return;
            }
            Ok(n) => c.write_at = c.write_at.saturating_add(n),
            Err(e) if io_again(&e) => return,
            Err(_) => {
                c.done = true;
                return;
            }
        }
    }
}

fn finish_write(c: &mut Conn) {
    if !c.keep_alive {
        c.done = true;
        return;
    }
    c.seq = None;
    c.ndjson = false;
    c.streamed = 0;
    c.finalized = false;
    c.intern_hits_at_add = 0;
    c.api = ServeApi::Generate;
    c.n_predict = 0;
    c.req = None;
    let _ = try_take_request(c);
}

fn queue_missing(c: Option<&mut Conn>, ndjson: bool, api: ServeApi) {
    let Some(c) = c else {
        return;
    };
    let err = if api.is_openai() {
        json_openai_error("missing sequence")
    } else {
        json_error("missing sequence")
    };
    if ndjson {
        c.write.extend(http_chunk(err.as_bytes()));
        c.write.extend(http_chunk_end());
        c.finalized = true;
    } else {
        c.write = http_json_bytes(500, "Internal Server Error", &err, c.keep_alive);
        c.write_at = 0;
    }
    c.seq = None;
}

fn append_trace(path: &str, jsonl: &str) -> Result<(), ServeError> {
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    f.write_all(jsonl.as_bytes())?;
    Ok(())
}

impl<'a> EngineHttp<'a> {
    /// Bind is already listening. `listener` must be non-blocking.
    pub(crate) fn new(
        model: &'a Llama,
        tok: &'a Tokenizer,
        args: &ServeArgs,
        listener: TcpListener,
    ) -> Result<Self, ServeError> {
        let mut engine = Engine::new(model, engine_cfg(tok, args))
            .map_err(|e| ServeError::Infer(e.to_string()))?;
        attach_store(
            &mut engine,
            model,
            &StoreAttach {
                expert_slots: args.expert_slots,
                expert_sim: args.expert_sim,
                expert_8gpu: args.expert_8gpu,
                expert_bytes: args.expert_bytes,
                gpu_cfg: args.gpu_cfg,
                fill: args.fill,
                kv_bytes: args.kv_bytes,
            },
        )
        .map_err(ServeError::Infer)?;
        if args.trace_out.is_some() {
            engine.enable_moe_trace();
        }
        Ok(Self {
            tok,
            n_predict: args.n_predict,
            model_id: args.model_id.clone(),
            engine,
            listener,
            conns: Vec::new(),
            trace_out: args.trace_out.clone(),
        })
    }

    /// Cross-sequence GEMM counters for the inner engine.
    #[cfg(test)]
    #[must_use]
    fn stats(&self) -> &crate::engine::EngineStats {
        self.engine.stats()
    }

    /// Hits from the attached ExpertStore, if any.
    #[cfg(test)]
    #[must_use]
    fn store_hits(&self) -> u64 {
        self.engine.expert_store_metrics().map_or(0, |m| m.hits)
    }

    /// Accept, admit, step, harvest, write. Drain new requests before `step`.
    pub(crate) fn poll(&mut self) -> Result<(), ServeError> {
        self.accept_ready()?;
        self.read_ready();
        self.admit_parsed();
        self.step_if_busy()?;
        self.stream_progress();
        self.harvest()?;
        self.write_ready();
        self.reap();
        Ok(())
    }

    fn accept_ready(&mut self) -> Result<(), ServeError> {
        loop {
            match self.listener.accept() {
                Ok((stream, _)) => {
                    stream.set_nonblocking(true)?;
                    stream.set_nodelay(true)?;
                    self.conns.push(Conn::new(stream));
                }
                Err(e) if io_again(&e) => return Ok(()),
                Err(e) => return Err(e.into()),
            }
        }
    }

    fn read_ready(&mut self) {
        let n = self.conns.len();
        for i in 0..n {
            if let Some(c) = self.conns.get_mut(i) {
                read_conn(c);
            }
        }
    }

    fn admit_parsed(&mut self) {
        let n = self.conns.len();
        for i in 0..n {
            if !self.conns.get(i).is_some_and(Conn::can_admit) {
                continue;
            }
            let Some(req) = self.conns.get_mut(i).and_then(|c| c.req.take()) else {
                continue;
            };
            match self.admit_req(&req) {
                AdmitOut::Seq {
                    id,
                    ndjson,
                    api,
                    n_predict,
                } => {
                    let hits = self.engine.pool().hits();
                    if let Some(c) = self.conns.get_mut(i) {
                        c.seq = Some(id);
                        c.intern_hits_at_add = hits;
                        c.ndjson = ndjson;
                        c.api = api;
                        c.n_predict = n_predict;
                        c.keep_alive = req.keep_alive;
                        if ndjson {
                            let head = if api.is_openai() {
                                http_sse_headers(req.keep_alive)
                            } else {
                                http_chunked_headers(req.keep_alive)
                            };
                            c.write.extend(head);
                        }
                    }
                }
                AdmitOut::Bytes(bytes) => {
                    if let Some(c) = self.conns.get_mut(i) {
                        c.write = bytes;
                        c.write_at = 0;
                    }
                }
            }
        }
    }

    fn admit_req(&mut self, req: &HttpRequest) -> AdmitOut {
        let ka = req.keep_alive;
        if req.method == "GET" {
            let path = crate::serve::normalize_path(path_only(&req.path));
            if path == "/metrics" {
                return AdmitOut::Bytes(http_json_bytes(
                    200,
                    "OK",
                    &json_engine_metrics(&self.engine),
                    ka,
                ));
            }
            if let Some((status, reason, body)) = dispatch_get(&req.path, &self.model_id, true) {
                return AdmitOut::Bytes(http_json_bytes(status, reason, &body, ka));
            }
        }
        let api = ServeApi::from_path(&req.path);
        if req.method != "POST" {
            return AdmitOut::http(405, "Method Not Allowed", "method must be POST", ka, api);
        }
        let Some(api) = api else {
            return AdmitOut::http(404, "Not Found", "not found", ka, None);
        };
        let body = match std::str::from_utf8(&req.body) {
            Ok(s) => s,
            Err(_) => {
                return AdmitOut::http(400, "Bad Request", "body must be utf-8", ka, Some(api))
            }
        };
        let gen = match parse_gen_req(body) {
            Ok(g) => g,
            Err(e) => return AdmitOut::http(400, "Bad Request", &e, ka, Some(api)),
        };
        let prompt = match gen.resolve(self.tok) {
            Ok(p) => p,
            Err(e) => return AdmitOut::http(400, "Bad Request", &e, ka, Some(api)),
        };
        if prompt.is_empty() {
            return AdmitOut::http(400, "Bad Request", "empty prompt", ka, Some(api));
        }
        let n_predict = gen.n_predict.unwrap_or(self.n_predict);
        let ids = match prompt_ids(self.tok, &prompt) {
            Ok(ids) if !ids.is_empty() => ids,
            Ok(_) => return AdmitOut::http(400, "Bad Request", "empty prompt", ka, Some(api)),
            Err(e) => {
                return AdmitOut::http(500, "Internal Server Error", &e.to_string(), ka, Some(api))
            }
        };
        match self.engine.add(&ids, n_predict) {
            Ok(id) => AdmitOut::Seq {
                id,
                ndjson: gen.stream,
                api,
                n_predict,
            },
            Err(LlamaError::EmptyPrompt) => {
                AdmitOut::http(400, "Bad Request", "empty prompt", ka, Some(api))
            }
            Err(e) => AdmitOut::http(500, "Internal Server Error", &e.to_string(), ka, Some(api)),
        }
    }

    fn step_if_busy(&mut self) -> Result<(), ServeError> {
        if !self.engine.has_runnable() {
            return Ok(());
        }
        let _n = self
            .engine
            .step()
            .map_err(|e| ServeError::Infer(e.to_string()))?;
        Ok(())
    }

    fn stream_progress(&mut self) {
        let n = self.conns.len();
        for i in 0..n {
            let (id, streamed) = match self.conns.get(i) {
                Some(c) if c.ndjson && !c.finalized => match c.seq {
                    Some(id) => (id, c.streamed),
                    None => continue,
                },
                _ => continue,
            };
            let Some(ids) = self.engine.generated(id) else {
                continue;
            };
            let Some(new_ids) = ids.get(streamed..) else {
                continue;
            };
            if new_ids.is_empty() {
                continue;
            }
            let next = ids.len();
            let piece = self.tok.decode(new_ids);
            let api = self.conns.get(i).map_or(ServeApi::Generate, |c| c.api);
            let model = self.model_id.clone();
            let payload = match api {
                ServeApi::Generate => json_token(&piece),
                ServeApi::Completions => sse_completion_delta(&piece, None, &model),
                ServeApi::Chat => sse_chat_delta(&piece, None, &model),
            };
            if let Some(c) = self.conns.get_mut(i) {
                c.write.extend(http_chunk(payload.as_bytes()));
                c.streamed = next;
            }
        }
    }

    fn harvest(&mut self) -> Result<(), ServeError> {
        let mut ready = Vec::new();
        for (i, c) in self.conns.iter().enumerate() {
            if c.done {
                continue;
            }
            let Some(id) = c.seq else {
                continue;
            };
            if !self.engine.is_finished(id) {
                continue;
            }
            if c.ndjson {
                if !c.finalized {
                    ready.push(i);
                }
            } else if c.write.is_empty() {
                ready.push(i);
            }
        }
        for i in ready {
            self.finish_seq(i)?;
        }
        Ok(())
    }

    fn finish_seq(&mut self, i: usize) -> Result<(), ServeError> {
        let (id, at, ndjson, api, n_predict) = match self.conns.get(i) {
            Some(c) => match c.seq {
                Some(id) => (id, c.intern_hits_at_add, c.ndjson, c.api, c.n_predict),
                None => return Ok(()),
            },
            None => return Ok(()),
        };
        if let Some(path) = self.trace_out.as_ref() {
            if let Some(trace) = self.engine.take_moe_trace(id) {
                append_trace(path, &trace.to_jsonl())?;
            }
        }
        let pages = self.engine.pool().hits().saturating_sub(at);
        let Some(out) = self.engine.take(id) else {
            queue_missing(self.conns.get_mut(i), ndjson, api);
            return Ok(());
        };
        let reason = finish_reason(out.generated.len(), n_predict);
        let model = self.model_id.clone();
        let json = match api {
            ServeApi::Generate => json_generated(&decode_seq(self.tok, &out), 0, pages),
            ServeApi::Completions => json_openai_completion(
                &self.tok.decode(&out.generated),
                out.prompt.len(),
                out.generated.len(),
                reason,
                &model,
            ),
            ServeApi::Chat => json_openai_chat(
                &self.tok.decode(&out.generated),
                out.prompt.len(),
                out.generated.len(),
                reason,
                &model,
            ),
        };
        if let Some(c) = self.conns.get_mut(i) {
            if ndjson {
                match api {
                    ServeApi::Generate => {
                        let mut line = json;
                        line.push('\n');
                        c.write.extend(http_chunk(line.as_bytes()));
                    }
                    ServeApi::Completions => {
                        c.write.extend(http_chunk(
                            sse_completion_delta("", Some(reason), &model).as_bytes(),
                        ));
                        c.write.extend(http_chunk(sse_done().as_bytes()));
                    }
                    ServeApi::Chat => {
                        c.write.extend(http_chunk(
                            sse_chat_delta("", Some(reason), &model).as_bytes(),
                        ));
                        c.write.extend(http_chunk(sse_done().as_bytes()));
                    }
                }
                c.write.extend(http_chunk_end());
                c.finalized = true;
            } else {
                c.write = http_json_bytes(200, "OK", &json, c.keep_alive);
                c.write_at = 0;
            }
            c.seq = None;
        }
        Ok(())
    }

    fn write_ready(&mut self) {
        let n = self.conns.len();
        for i in 0..n {
            if let Some(c) = self.conns.get_mut(i) {
                write_conn(c);
            }
        }
    }

    fn reap(&mut self) {
        self.conns.retain(|c| !c.done);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{greedy_generate_ctx, tiny_llama_gguf, tiny_qwen3moe_gguf};
    use crate::gguf::load_gguf_owned;
    use crate::serve::bind_loopback;
    use expertvm::{GpuFill, GpuStoreCfg, Prefetch};
    use std::fs::File;
    use std::net::SocketAddr;
    use std::time::Duration;

    fn tiny_model() -> (Llama, Tokenizer) {
        let g = load_gguf_owned(tiny_llama_gguf()).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("model");
        (model, tok)
    }

    fn engine_args() -> ServeArgs {
        ServeArgs {
            path: "tiny.gguf".into(),
            n_predict: 2,
            n_ctx: Some(64),
            kv_page: Some(16),
            bind: "127.0.0.1:0".into(),
            model_id: "tiny".into(),
            engine: true,
            max_seqs: Some(4),
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

    fn post_json(json: &str) -> Vec<u8> {
        post_path("/generate", json)
    }

    fn post_path(path: &str, json: &str) -> Vec<u8> {
        format!(
            "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: {}\r\n\r\n{json}",
            json.len()
        )
        .into_bytes()
    }

    fn connect_nb(addr: SocketAddr) -> TcpStream {
        let s = TcpStream::connect_timeout(&addr, Duration::from_secs(2)).expect("connect");
        s.set_nodelay(true).expect("nodelay");
        s.set_nonblocking(true).expect("nb");
        s
    }

    fn write_all_nb(s: &mut TcpStream, bytes: &[u8]) {
        let mut at = 0usize;
        for _ in 0..10_000 {
            if at >= bytes.len() {
                return;
            }
            let rest = bytes.get(at..).unwrap_or(&[]);
            match s.write(rest) {
                Ok(0) => panic!("write eof"),
                Ok(n) => at = at.saturating_add(n),
                Err(e) if io_again(&e) => {}
                Err(e) => panic!("{e}"),
            }
        }
        panic!("write stuck at {at}/{}", bytes.len());
    }

    fn read_nb(s: &mut TcpStream, buf: &mut Vec<u8>) {
        let mut tmp = [0u8; 512];
        loop {
            match s.read(&mut tmp) {
                Ok(0) => return,
                Ok(n) => buf.extend_from_slice(tmp.get(..n).unwrap_or(&[])),
                Err(e) if io_again(&e) => return,
                Err(e) => panic!("{e}"),
            }
        }
    }

    fn response_parts(buf: &[u8]) -> Option<(u16, String)> {
        let s = std::str::from_utf8(buf).ok()?;
        let (head, body) = s.split_once("\r\n\r\n")?;
        let status: u16 = head.split_whitespace().nth(1)?.parse().ok()?;
        let mut cl = None;
        for line in head.lines().skip(1) {
            if let Some((k, v)) = line.split_once(':') {
                if k.eq_ignore_ascii_case("content-length") {
                    cl = Some(v.trim().parse().ok()?);
                }
            }
        }
        let cl = cl?;
        if body.len() < cl {
            return None;
        }
        Some((status, body.get(..cl)?.to_string()))
    }

    fn chunked_parts(buf: &[u8]) -> Option<(u16, String)> {
        let s = std::str::from_utf8(buf).ok()?;
        let (head, rest) = s.split_once("\r\n\r\n")?;
        if !head
            .to_ascii_lowercase()
            .contains("transfer-encoding: chunked")
        {
            return None;
        }
        let status: u16 = head.split_whitespace().nth(1)?.parse().ok()?;
        let body = decode_chunk_body(rest)?;
        Some((status, body))
    }

    fn decode_chunk_body(rest: &str) -> Option<String> {
        let mut input = rest.as_bytes();
        let mut body = Vec::new();
        loop {
            let nl = input.windows(2).position(|w| w == b"\r\n")?;
            let hex = std::str::from_utf8(input.get(..nl)?).ok()?.trim();
            let size = usize::from_str_radix(hex, 16).ok()?;
            input = input.get(nl.saturating_add(2)..)?;
            if size == 0 {
                return String::from_utf8(body).ok();
            }
            if input.len() < size.saturating_add(2) {
                return None;
            }
            body.extend_from_slice(input.get(..size)?);
            input = input.get(size.saturating_add(2)..)?;
        }
    }

    fn response_done(buf: &[u8]) -> Option<(u16, String)> {
        response_parts(buf).or_else(|| chunked_parts(buf))
    }

    fn generated_field(body: &str) -> String {
        let key = "\"generated\":\"";
        let i = body.find(key).expect("generated");
        let rest = body.get(i.saturating_add(key.len())..).expect("rest");
        let end = rest.find('"').expect("end");
        rest.get(..end).expect("slice").to_string()
    }

    fn json_u64(body: &str, key: &str) -> u64 {
        let pat = format!("\"{key}\":");
        let i = body.find(&pat).expect("key");
        let rest = body.get(i.saturating_add(pat.len())..).expect("rest");
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        digits.parse().expect("u64")
    }

    fn drive(http: &mut EngineHttp<'_>, streams: &mut [TcpStream], bufs: &mut [Vec<u8>]) {
        assert_eq!(streams.len(), bufs.len());
        for _ in 0..50_000 {
            http.poll().expect("poll");
            let mut done = true;
            for (s, buf) in streams.iter_mut().zip(bufs.iter_mut()) {
                read_nb(s, buf);
                if response_done(buf).is_none() {
                    done = false;
                }
            }
            if done {
                return;
            }
        }
        panic!("engine http poll did not finish");
    }

    fn post_pair(
        http: &mut EngineHttp<'_>,
        addr: SocketAddr,
        ja: &str,
        jb: &str,
    ) -> (String, String) {
        let mut streams = [connect_nb(addr), connect_nb(addr)];
        write_all_nb(&mut streams[0], &post_json(ja));
        write_all_nb(&mut streams[1], &post_json(jb));
        let mut bufs = [Vec::new(), Vec::new()];
        drive(http, &mut streams, &mut bufs);
        let (sa, ba) = response_done(&bufs[0]).expect("a resp");
        let (sb, bb) = response_done(&bufs[1]).expect("b resp");
        assert_eq!(sa, 200, "{ba}");
        assert_eq!(sb, 200, "{bb}");
        (ba, bb)
    }

    #[test]
    fn two_posts_share_a_prefill_gemm() {
        let (model, tok) = tiny_model();
        let args = engine_args();
        let listener = bind_loopback("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nb");
        let addr = listener.local_addr().expect("addr");
        let mut http = EngineHttp::new(&model, &tok, &args, listener).expect("http");
        let mut streams = [connect_nb(addr), connect_nb(addr)];
        write_all_nb(&mut streams[0], &post_json(r#"{"prompt":"a"}"#));
        write_all_nb(&mut streams[1], &post_json(r#"{"prompt":"ab"}"#));
        let mut bufs = [Vec::new(), Vec::new()];
        drive(&mut http, &mut streams, &mut bufs);
        let (sa, ba) = response_parts(&bufs[0]).expect("a resp");
        let (sb, bb) = response_parts(&bufs[1]).expect("b resp");
        assert_eq!(sa, 200, "{ba}");
        assert_eq!(sb, 200, "{bb}");
        let expect_a = greedy_generate_ctx(&model, &tok, "a", 2, Some(64)).expect("ga");
        let expect_b = greedy_generate_ctx(&model, &tok, "ab", 2, Some(64)).expect("gb");
        assert_eq!(generated_field(&ba), expect_a);
        assert_eq!(generated_field(&bb), expect_b);
        assert!(
            http.stats().gemm_peak >= 2,
            "two concurrent posts must GEMM together, peak={}",
            http.stats().gemm_peak
        );
    }

    #[test]
    fn keep_alive_two_posts_one_engine_connection() {
        let (model, tok) = tiny_model();
        let args = engine_args();
        let listener = bind_loopback("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nb");
        let addr = listener.local_addr().expect("addr");
        let mut http = EngineHttp::new(&model, &tok, &args, listener).expect("http");
        let expect = greedy_generate_ctx(&model, &tok, "ab", 2, Some(64)).expect("g");
        let mut streams = [connect_nb(addr)];
        write_all_nb(&mut streams[0], &post_json(r#"{"prompt":"ab"}"#));
        let mut bufs = [Vec::new()];
        drive(&mut http, &mut streams, &mut bufs);
        let (s1, b1) = response_parts(&bufs[0]).expect("first");
        assert_eq!(s1, 200, "{b1}");
        assert_eq!(generated_field(&b1), expect);
        write_all_nb(&mut streams[0], &post_json(r#"{"prompt":"ab"}"#));
        bufs[0].clear();
        drive(&mut http, &mut streams, &mut bufs);
        let (s2, b2) = response_parts(&bufs[0]).expect("second");
        assert_eq!(s2, 200, "{b2}");
        assert_eq!(generated_field(&b2), expect);
    }

    #[test]
    fn engine_http_empty_prompt_is_400() {
        let (model, tok) = tiny_model();
        let args = engine_args();
        let listener = bind_loopback("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nb");
        let addr = listener.local_addr().expect("addr");
        let mut http = EngineHttp::new(&model, &tok, &args, listener).expect("http");
        let mut streams = [connect_nb(addr)];
        write_all_nb(&mut streams[0], &post_json(r#"{"prompt":""}"#));
        let mut bufs = [Vec::new()];
        drive(&mut http, &mut streams, &mut bufs);
        let (status, body) = response_parts(&bufs[0]).expect("resp");
        assert_eq!(status, 400, "{body}");
        assert!(body.contains("empty prompt"), "{body}");
        assert_eq!(http.stats().gemm_peak, 0);
    }

    #[test]
    fn two_moe_posts_acquire_from_direct_store() {
        let g = load_gguf_owned(tiny_qwen3moe_gguf()).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("model");
        let mut args = engine_args();
        args.expert_slots = Some(0);
        let listener = bind_loopback("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nb");
        let addr = listener.local_addr().expect("addr");
        let mut http = EngineHttp::new(&model, &tok, &args, listener).expect("http");
        let mut streams = [connect_nb(addr), connect_nb(addr)];
        write_all_nb(&mut streams[0], &post_json(r#"{"prompt":"a"}"#));
        write_all_nb(&mut streams[1], &post_json(r#"{"prompt":"ab"}"#));
        let mut bufs = [Vec::new(), Vec::new()];
        drive(&mut http, &mut streams, &mut bufs);
        let (sa, ba) = response_parts(&bufs[0]).expect("a resp");
        let (sb, bb) = response_parts(&bufs[1]).expect("b resp");
        assert_eq!(sa, 200, "{ba}");
        assert_eq!(sb, 200, "{bb}");
        let expect_a = greedy_generate_ctx(&model, &tok, "a", 2, Some(64)).expect("ga");
        let expect_b = greedy_generate_ctx(&model, &tok, "ab", 2, Some(64)).expect("gb");
        assert_eq!(generated_field(&ba), expect_a);
        assert_eq!(generated_field(&bb), expect_b);
        assert!(
            http.stats().gemm_peak >= 2,
            "two concurrent MoE posts must GEMM together, peak={}",
            http.stats().gemm_peak
        );
        assert!(
            http.store_hits() > 0,
            "HTTP Engine must acquire from DirectStore"
        );
    }

    #[test]
    fn second_engine_http_prompt_interns_completed_pages() {
        let (model, tok) = tiny_model();
        let mut args = engine_args();
        args.kv_page = Some(2);
        let listener = bind_loopback("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nb");
        let addr = listener.local_addr().expect("addr");
        let mut http = EngineHttp::new(&model, &tok, &args, listener).expect("http");
        let expect = greedy_generate_ctx(&model, &tok, "ab", 2, Some(64)).expect("g");
        let mut streams = [connect_nb(addr)];
        write_all_nb(&mut streams[0], &post_json(r#"{"prompt":"ab"}"#));
        let mut bufs = [Vec::new()];
        drive(&mut http, &mut streams, &mut bufs);
        let (s1, b1) = response_parts(&bufs[0]).expect("first");
        assert_eq!(s1, 200, "{b1}");
        assert_eq!(generated_field(&b1), expect);
        assert_eq!(json_u64(&b1, "page_hits"), 0, "{b1}");
        let mut streams = [connect_nb(addr)];
        write_all_nb(&mut streams[0], &post_json(r#"{"prompt":"ab"}"#));
        let mut bufs = [Vec::new()];
        drive(&mut http, &mut streams, &mut bufs);
        let (s2, b2) = response_parts(&bufs[0]).expect("second");
        assert_eq!(s2, 200, "{b2}");
        assert_eq!(generated_field(&b2), expect);
        assert!(
            json_u64(&b2, "page_hits") > 0,
            "second identical prompt must intern-hit completed pages: {b2}"
        );
    }

    #[test]
    fn engine_http_prefill_chunk_still_gemms() {
        let (model, tok) = tiny_model();
        assert!(
            prompt_ids(&tok, "aa").expect("aa").len() >= 2,
            "chunked prefill needs a multi-token prompt"
        );
        let mut full = engine_args();
        full.prefill_chunk = 0;
        let listener = bind_loopback("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nb");
        let addr = listener.local_addr().expect("addr");
        let mut http = EngineHttp::new(&model, &tok, &full, listener).expect("http");
        let (ba, bb) = post_pair(&mut http, addr, r#"{"prompt":"aa"}"#, r#"{"prompt":"aaa"}"#);
        let steps0 = http.stats().steps;
        let expect_a = greedy_generate_ctx(&model, &tok, "aa", 2, Some(64)).expect("ga");
        let expect_b = greedy_generate_ctx(&model, &tok, "aaa", 2, Some(64)).expect("gb");
        assert_eq!(generated_field(&ba), expect_a);
        assert_eq!(generated_field(&bb), expect_b);

        let mut chunked = engine_args();
        chunked.prefill_chunk = 1;
        let listener = bind_loopback("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nb");
        let addr = listener.local_addr().expect("addr");
        let mut http = EngineHttp::new(&model, &tok, &chunked, listener).expect("http");
        let (ba, bb) = post_pair(&mut http, addr, r#"{"prompt":"aa"}"#, r#"{"prompt":"aaa"}"#);
        assert_eq!(generated_field(&ba), expect_a);
        assert_eq!(generated_field(&bb), expect_b);
        assert!(
            http.stats().gemm_peak >= 2,
            "chunked prefill must still GEMM together, peak={}",
            http.stats().gemm_peak
        );
        assert!(
            http.stats().steps > steps0,
            "prefill_chunk=1 must take more Engine steps than a full prefill, chunk={} full={steps0}",
            http.stats().steps
        );
    }

    #[test]
    fn engine_http_trace_out_appends_moe_jsonl() {
        let g = load_gguf_owned(tiny_qwen3moe_gguf()).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("model");
        let path = std::env::temp_dir().join(format!(
            "llama-rust-engine-http-trace-{}.jsonl",
            std::process::id()
        ));
        let _f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .expect("trunc");
        drop(_f);
        let mut args = engine_args();
        args.expert_slots = Some(0);
        args.trace_out = Some(path.to_str().expect("utf8").to_string());
        let listener = bind_loopback("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nb");
        let addr = listener.local_addr().expect("addr");
        let mut http = EngineHttp::new(&model, &tok, &args, listener).expect("http");
        let (_ba, _bb) = post_pair(&mut http, addr, r#"{"prompt":"a"}"#, r#"{"prompt":"ab"}"#);
        let mut f = File::open(&path).expect("open");
        let mut buf = Vec::new();
        let _n = f.read_to_end(&mut buf).expect("read");
        let text = String::from_utf8(buf).expect("utf8");
        let tr = expertvm::Trace::parse(&text).expect("jsonl");
        assert!(
            !tr.events.is_empty(),
            "serve --trace-out must emit MoE events: {text:?}"
        );
    }

    #[test]
    fn engine_http_stream_ndjson_matches_greedy() {
        let (model, tok) = tiny_model();
        let args = engine_args();
        let listener = bind_loopback("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nb");
        let addr = listener.local_addr().expect("addr");
        let mut http = EngineHttp::new(&model, &tok, &args, listener).expect("http");
        let mut streams = [connect_nb(addr)];
        write_all_nb(
            &mut streams[0],
            &post_json(r#"{"prompt":"ab","stream":true}"#),
        );
        let mut bufs = [Vec::new()];
        drive(&mut http, &mut streams, &mut bufs);
        let raw = std::str::from_utf8(&bufs[0]).expect("utf8");
        assert!(
            raw.to_ascii_lowercase()
                .contains("transfer-encoding: chunked"),
            "{raw}"
        );
        assert!(raw.contains("application/x-ndjson"), "{raw}");
        let (status, body) = response_done(&bufs[0]).expect("resp");
        assert_eq!(status, 200, "{body}");
        let expect = greedy_generate_ctx(&model, &tok, "ab", 2, Some(64)).expect("g");
        assert_eq!(generated_field(&body), expect);
        assert!(
            body.contains("\"token\""),
            "stream must emit token lines: {body}"
        );
    }

    #[test]
    fn two_stream_posts_share_a_prefill_gemm() {
        let (model, tok) = tiny_model();
        let args = engine_args();
        let listener = bind_loopback("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nb");
        let addr = listener.local_addr().expect("addr");
        let mut http = EngineHttp::new(&model, &tok, &args, listener).expect("http");
        let (ba, bb) = post_pair(
            &mut http,
            addr,
            r#"{"prompt":"a","stream":true}"#,
            r#"{"prompt":"ab","stream":true}"#,
        );
        let expect_a = greedy_generate_ctx(&model, &tok, "a", 2, Some(64)).expect("ga");
        let expect_b = greedy_generate_ctx(&model, &tok, "ab", 2, Some(64)).expect("gb");
        assert_eq!(generated_field(&ba), expect_a);
        assert_eq!(generated_field(&bb), expect_b);
        assert!(ba.contains("\"token\""), "{ba}");
        assert!(bb.contains("\"token\""), "{bb}");
        assert!(
            http.stats().gemm_peak >= 2,
            "two concurrent streams must GEMM together, peak={}",
            http.stats().gemm_peak
        );
    }

    #[test]
    fn engine_http_v1_completions_matches_greedy_completion() {
        use crate::decode::{greedy_generate_cache, prompt_ids};
        let (model, tok) = tiny_model();
        let args = engine_args();
        let listener = bind_loopback("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nb");
        let addr = listener.local_addr().expect("addr");
        let mut http = EngineHttp::new(&model, &tok, &args, listener).expect("http");
        let mut streams = [connect_nb(addr)];
        write_all_nb(
            &mut streams[0],
            &post_path("/v1/completions", r#"{"prompt":"ab","max_tokens":2}"#),
        );
        let mut bufs = [Vec::new()];
        drive(&mut http, &mut streams, &mut bufs);
        let (status, body) = response_parts(&bufs[0]).expect("resp");
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("\"object\":\"text_completion\""), "{body}");
        let mut ids = prompt_ids(&tok, "ab").expect("ids");
        let prompt_n = ids.len();
        let mut slot = None;
        let cache = model
            .ensure_cache_page(&mut slot, ids.len().saturating_add(2), Some(64), Some(16))
            .expect("kv");
        let _full = greedy_generate_cache(&model, &tok, cache, &mut ids, 2).expect("g");
        let want = tok.decode(ids.get(prompt_n..).unwrap_or(&[]));
        let pat = "\"text\":\"";
        let i = body.find(pat).expect("text");
        let rest = body.get(i.saturating_add(pat.len())..).expect("rest");
        let end = rest.find('"').expect("end");
        let got = rest.get(..end).expect("slice");
        assert_eq!(got, want, "{body}");
    }

    #[test]
    fn engine_http_v1_stream_is_sse() {
        let (model, tok) = tiny_model();
        let args = engine_args();
        let listener = bind_loopback("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nb");
        let addr = listener.local_addr().expect("addr");
        let mut http = EngineHttp::new(&model, &tok, &args, listener).expect("http");
        let mut streams = [connect_nb(addr)];
        write_all_nb(
            &mut streams[0],
            &post_path("/v1/completions", r#"{"prompt":"ab","stream":true}"#),
        );
        let mut bufs = [Vec::new()];
        drive(&mut http, &mut streams, &mut bufs);
        let raw = std::str::from_utf8(&bufs[0]).expect("utf8");
        assert!(raw.contains("text/event-stream"), "{raw}");
        assert!(raw.contains("data: "), "{raw}");
        assert!(raw.contains("[DONE]"), "{raw}");
        let (status, body) = response_done(&bufs[0]).expect("resp");
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("text_completion"), "{body}");
    }

    fn get_path(path: &str) -> Vec<u8> {
        format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .into_bytes()
    }

    #[test]
    fn engine_http_get_health_models_and_metrics() {
        let (model, tok) = tiny_model();
        let args = engine_args();
        let listener = bind_loopback("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nb");
        let addr = listener.local_addr().expect("addr");
        let mut http = EngineHttp::new(&model, &tok, &args, listener).expect("http");
        let mut streams = [connect_nb(addr)];
        write_all_nb(&mut streams[0], &get_path("/health"));
        let mut bufs = [Vec::new()];
        drive(&mut http, &mut streams, &mut bufs);
        let (status, body) = response_done(&bufs[0]).expect("health");
        assert_eq!(status, 200, "{body}");
        assert_eq!(body, "{\"status\":\"ok\"}");
        let mut streams = [connect_nb(addr)];
        write_all_nb(&mut streams[0], &get_path("/v1/models"));
        let mut bufs = [Vec::new()];
        drive(&mut http, &mut streams, &mut bufs);
        let (status, body) = response_done(&bufs[0]).expect("models");
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("\"object\":\"list\""), "{body}");
        assert!(body.contains("\"id\":\"tiny\""), "{body}");
        let mut streams = [connect_nb(addr)];
        write_all_nb(&mut streams[0], &get_path("/metrics"));
        let mut bufs = [Vec::new()];
        drive(&mut http, &mut streams, &mut bufs);
        let (status, body) = response_done(&bufs[0]).expect("metrics");
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("\"engine\":true"), "{body}");
        assert!(body.contains("\"active\":0"), "{body}");
        assert!(body.contains("\"waiting\":0"), "{body}");
    }
}
