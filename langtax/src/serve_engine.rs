//! Concurrent HTTP loop around one [`Engine`].
//!
//! Default `gguf_gemv serve` stays one connection at a time. `--engine` admits
//! several `POST /generate` bodies onto one interned pool so prefills and
//! decodes GEMM together. No Tokio, no keep-alive, no OpenAI SDK surface.

use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};

use crate::decode::{prompt_ids, Llama, LlamaError};
use crate::engine::{Engine, EngineCfg, SeqId, SeqOutput};
use crate::serve::{
    http_err_status, http_json_bytes, json_error, json_generated, normalize_path, parse_gen_req,
    try_parse_http_request, HttpRequest, ServeArgs, ServeError, MAX_REQ,
};
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
    engine: Engine<'a>,
    listener: TcpListener,
    conns: Vec<Conn>,
}

struct Conn {
    stream: TcpStream,
    read: Vec<u8>,
    req: Option<HttpRequest>,
    seq: Option<SeqId>,
    write: Vec<u8>,
    write_at: usize,
    done: bool,
}

enum AdmitOut {
    Seq(SeqId),
    Bytes(Vec<u8>),
}

impl AdmitOut {
    fn http(status: u16, reason: &'static str, msg: &str) -> Self {
        Self::Bytes(http_json_bytes(status, reason, &json_error(msg)))
    }
}

fn io_again(e: &std::io::Error) -> bool {
    matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted)
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
        prefill_chunk: 0,
        eos: tok.eos,
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
        }
    }

    fn can_admit(&self) -> bool {
        self.req.is_some() && self.seq.is_none() && self.write.is_empty() && !self.done
    }
}

fn read_conn(c: &mut Conn) {
    if c.done || c.req.is_some() || !c.write.is_empty() {
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
                match try_parse_http_request(&c.read) {
                    Ok(Some(req)) => {
                        c.req = Some(req);
                        return;
                    }
                    Ok(None) => {}
                    Err(e) => {
                        queue_err(c, &e);
                        return;
                    }
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
    c.write = http_json_bytes(status, reason, &json_error(&e.to_string()));
    c.write_at = 0;
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
                    c.done = true;
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

impl<'a> EngineHttp<'a> {
    /// Bind is already listening. `listener` must be non-blocking.
    pub(crate) fn new(
        model: &'a Llama,
        tok: &'a Tokenizer,
        args: &ServeArgs,
        listener: TcpListener,
    ) -> Result<Self, ServeError> {
        let engine = Engine::new(model, engine_cfg(tok, args))
            .map_err(|e| ServeError::Infer(e.to_string()))?;
        Ok(Self {
            tok,
            n_predict: args.n_predict,
            engine,
            listener,
            conns: Vec::new(),
        })
    }

    /// Cross-sequence GEMM counters for the inner engine.
    #[cfg(test)]
    #[must_use]
    fn stats(&self) -> &crate::engine::EngineStats {
        self.engine.stats()
    }

    /// Accept, admit, step, harvest, write. Drain new requests before `step`.
    pub(crate) fn poll(&mut self) -> Result<(), ServeError> {
        self.accept_ready()?;
        self.read_ready();
        self.admit_parsed();
        self.step_if_busy()?;
        self.harvest();
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
                AdmitOut::Seq(id) => {
                    if let Some(c) = self.conns.get_mut(i) {
                        c.seq = Some(id);
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
        if req.method != "POST" {
            return AdmitOut::http(405, "Method Not Allowed", "method must be POST");
        }
        if normalize_path(&req.path) != "/generate" {
            return AdmitOut::http(404, "Not Found", "not found");
        }
        let body = match std::str::from_utf8(&req.body) {
            Ok(s) => s,
            Err(_) => return AdmitOut::http(400, "Bad Request", "body must be utf-8"),
        };
        let gen = match parse_gen_req(body) {
            Ok(g) => g,
            Err(e) => return AdmitOut::http(400, "Bad Request", &e),
        };
        let prompt = match gen.resolve(self.tok) {
            Ok(p) => p,
            Err(e) => return AdmitOut::http(400, "Bad Request", &e),
        };
        if prompt.is_empty() {
            return AdmitOut::http(400, "Bad Request", "empty prompt");
        }
        let n_predict = gen.n_predict.unwrap_or(self.n_predict);
        let ids = match prompt_ids(self.tok, &prompt) {
            Ok(ids) if !ids.is_empty() => ids,
            Ok(_) => return AdmitOut::http(400, "Bad Request", "empty prompt"),
            Err(e) => return AdmitOut::http(500, "Internal Server Error", &e.to_string()),
        };
        match self.engine.add(&ids, n_predict) {
            Ok(id) => AdmitOut::Seq(id),
            Err(LlamaError::EmptyPrompt) => AdmitOut::http(400, "Bad Request", "empty prompt"),
            Err(e) => AdmitOut::http(500, "Internal Server Error", &e.to_string()),
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

    fn harvest(&mut self) {
        let mut ready = Vec::new();
        for (i, c) in self.conns.iter().enumerate() {
            if c.done || !c.write.is_empty() {
                continue;
            }
            if let Some(id) = c.seq {
                if self.engine.is_finished(id) {
                    ready.push(i);
                }
            }
        }
        for i in ready {
            let Some(id) = self.conns.get(i).and_then(|c| c.seq) else {
                continue;
            };
            let bytes = match self.engine.take(id) {
                Some(out) => {
                    let text = decode_seq(self.tok, &out);
                    http_json_bytes(200, "OK", &json_generated(&text, 0, 0))
                }
                None => http_json_bytes(
                    500,
                    "Internal Server Error",
                    &json_error("missing sequence"),
                ),
            };
            if let Some(c) = self.conns.get_mut(i) {
                c.write = bytes;
                c.write_at = 0;
            }
        }
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
    use crate::decode::{greedy_generate_ctx, tiny_llama_gguf};
    use crate::gguf::load_gguf_owned;
    use crate::serve::bind_loopback;
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
            engine: true,
            max_seqs: Some(4),
        }
    }

    fn post_json(json: &str) -> Vec<u8> {
        format!(
            "POST /generate HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: {}\r\n\r\n{json}",
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
        let mut cl = 0usize;
        for line in head.lines().skip(1) {
            if let Some((k, v)) = line.split_once(':') {
                if k.eq_ignore_ascii_case("content-length") {
                    cl = v.trim().parse().ok()?;
                }
            }
        }
        if body.len() < cl {
            return None;
        }
        Some((status, body.get(..cl)?.to_string()))
    }

    fn generated_field(body: &str) -> String {
        let key = "\"generated\":\"";
        let i = body.find(key).expect("generated");
        let rest = body.get(i.saturating_add(key.len())..).expect("rest");
        let end = rest.find('"').expect("end");
        rest.get(..end).expect("slice").to_string()
    }

    fn drive(http: &mut EngineHttp<'_>, streams: &mut [TcpStream], bufs: &mut [Vec<u8>]) {
        assert_eq!(streams.len(), bufs.len());
        for _ in 0..50_000 {
            http.poll().expect("poll");
            let mut done = true;
            for (s, buf) in streams.iter_mut().zip(bufs.iter_mut()) {
                read_nb(s, buf);
                if response_parts(buf).is_none() {
                    done = false;
                }
            }
            if done {
                return;
            }
        }
        panic!("engine http poll did not finish");
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
}
