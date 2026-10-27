//! Measured HTTP/SSE benchmark. Chunk intervals are not token ITL.
use reqwest::blocking::Client;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{BufRead, BufReader};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc, Arc,
};
use std::time::{Duration, Instant};

pub fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn percentile(values: &[f64], fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * fraction).ceil() as usize).saturating_sub(1);
    sorted.get(index).copied()
}

fn parse_stream(reader: impl BufRead, started: Instant, result: &mut Value) -> Result<(), String> {
    let mut reader = reader;
    let mut data = Vec::new();
    let mut line = Vec::new();
    let mut output = String::new();
    let mut previous = None;
    loop {
        line.clear();
        // Limit per-line and event size so malformed streams cannot grow unbounded.
        loop {
            let available = reader.fill_buf().map_err(|e| e.to_string())?;
            if available.is_empty() {
                break;
            }
            let take = available
                .iter()
                .position(|b| *b == b'\n')
                .map_or(available.len(), |n| n + 1);
            if line.len() + take > 1_048_576 {
                return Err("SSE line exceeds 1 MiB".into());
            }
            line.extend_from_slice(&available[..take]);
            let done = line.last() == Some(&b'\n');
            reader.consume(take);
            if done {
                break;
            }
        }
        if line.is_empty() {
            return Err("SSE ended without [DONE]".into());
        }
        let text = std::str::from_utf8(&line)
            .map_err(|e| e.to_string())?
            .trim_end_matches(['\r', '\n']);
        if let Some(value) = text.strip_prefix("data:") {
            data.push(value.trim_start_matches(' ').to_owned());
            if data.iter().map(String::len).sum::<usize>() > 1_048_576 {
                return Err("SSE event exceeds 1 MiB".into());
            }
            continue;
        }
        if !text.is_empty() || data.is_empty() {
            continue;
        }
        let payload = data.join("\n");
        data.clear();
        let now = started.elapsed().as_secs_f64() * 1000.0;
        if payload == "[DONE]" {
            result["output_sha256"] = json!(hash(output.as_bytes()));
            return Ok(());
        }
        let event: Value = serde_json::from_str(&payload).map_err(|e| e.to_string())?;
        if !event["error"].is_null() {
            return Err("server returned an SSE error".into());
        }
        for name in ["completion_tokens", "prompt_tokens"] {
            if let Some(count) = event["usage"][name].as_u64() {
                result[name] = json!(count);
            }
        }
        if let Some(choices) = event["choices"].as_array() {
            for choice in choices {
                let value = &choice["text"];
                if value.is_null() {
                    continue;
                }
                let text = value.as_str().ok_or("completion text is not a string")?;
                if text.is_empty() {
                    continue;
                }
                output.push_str(text);
                if let Some(last) = previous {
                    result["inter_event_ms"]
                        .as_array_mut()
                        .ok_or("missing intervals")?
                        .push(json!(now - last));
                } else {
                    result["ttft_ms"] = json!(now);
                }
                previous = Some(now);
                result["content_events"] =
                    json!(result["content_events"].as_u64().unwrap_or(0) + 1);
            }
        }
    }
}

pub fn request_one(url: &str, payload: &Value, timeout: Duration, key: Option<&str>) -> Value {
    let bytes = serde_json::to_vec(payload).unwrap_or_default();
    let mut result = json!({"ok":false,"request_sha256":hash(&bytes),"ttft_ms":null,
        "duration_ms":null,"completion_tokens":null,"prompt_tokens":null,
        "content_events":0,"inter_event_ms":[]});
    let started = Instant::now();
    let outcome = (|| -> Result<(), String> {
        // Separate client per request avoids server-specific connection reuse effects.
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| e.to_string())?;
        let mut request = client
            .post(url)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .body(bytes);
        if let Some(key) = key {
            request = request.bearer_auth(key);
        }
        let response = request.send().map_err(|e| e.without_url().to_string())?;
        if !response.status().is_success() {
            return Err(format!("HTTP {}", response.status().as_u16()));
        }
        if !response
            .headers()
            .get("Content-Type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .contains("text/event-stream")
        {
            return Err("response is not text/event-stream".into());
        }
        parse_stream(BufReader::new(response), started, &mut result)
    })();
    match outcome {
        Ok(()) => result["ok"] = json!(true),
        Err(error) => result["error"] = json!(error),
    }
    result["duration_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
    result
}

pub fn summarize(results: &[Value], elapsed: f64) -> Value {
    let good: Vec<_> = results.iter().filter(|r| r["ok"] == true).collect();
    let ttfts: Vec<_> = good.iter().filter_map(|r| r["ttft_ms"].as_f64()).collect();
    let latencies: Vec<_> = good
        .iter()
        .filter_map(|r| r["duration_ms"].as_f64())
        .collect();
    let intervals: Vec<_> = good
        .iter()
        .flat_map(|r| {
            r["inter_event_ms"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_f64)
        })
        .collect();
    let counts: Option<Vec<u64>> = good
        .iter()
        .map(|r| r["completion_tokens"].as_u64())
        .collect();
    let tokens = counts
        .filter(|_| !good.is_empty())
        .and_then(|v| v.into_iter().try_fold(0u64, u64::checked_add));
    let tpots: Vec<f64> = good
        .iter()
        .filter_map(|r| {
            let count = r["completion_tokens"].as_u64()?;
            (count > 1).then(|| {
                Some((r["duration_ms"].as_f64()? - r["ttft_ms"].as_f64()?) / (count - 1) as f64)
            })?
        })
        .collect();
    json!({"requests":results.len(),"successful":good.len(),"errors":results.len()-good.len(),
        "elapsed_seconds":elapsed,"successful_requests_per_second":good.len() as f64 / elapsed,
        "completion_tokens":tokens,"completion_tokens_per_second":tokens.map(|n| n as f64 / elapsed),
        "ttft_ms_p50":percentile(&ttfts,0.5),"ttft_ms_p95":percentile(&ttfts,0.95),
        "request_latency_ms_p95":percentile(&latencies,0.95),
        "inter_content_event_ms_p95":percentile(&intervals,0.95),"tpot_ms_p95":percentile(&tpots,0.95),
        "itl_ms_p95":null,"token_count_source":if tokens.is_some(){"server_usage"}else{"unavailable"}})
}

pub fn run(
    url: &str,
    payloads: Vec<Value>,
    concurrency: usize,
    timeout: Duration,
    key: Option<String>,
) -> (Value, Vec<Value>) {
    let started = Instant::now();
    let payloads = Arc::new(payloads);
    let next = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = mpsc::channel();
    std::thread::scope(|scope| {
        for _ in 0..concurrency.min(payloads.len()) {
            let (payloads, next, sender, key) =
                (payloads.clone(), next.clone(), sender.clone(), key.clone());
            scope.spawn(move || loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(payload) = payloads.get(index) else {
                    break;
                };
                if sender
                    .send((index, request_one(url, payload, timeout, key.as_deref())))
                    .is_err()
                {
                    break;
                }
            });
        }
    });
    drop(sender);
    let mut results: Vec<_> = receiver.into_iter().collect();
    results.sort_by_key(|(index, _)| *index);
    let results: Vec<_> = results.into_iter().map(|(_, result)| result).collect();
    (
        summarize(&results, started.elapsed().as_secs_f64()),
        results,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    fn mock(
        count: usize,
        body: &'static str,
        status: u16,
    ) -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1/completions", listener.local_addr().unwrap());
        let thread = std::thread::spawn(move || {
            let mut workers = Vec::new();
            for _ in 0..count {
                let (mut stream, _) = listener.accept().unwrap();
                workers.push(std::thread::spawn(move || {
                    stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                    let mut head = Vec::new();
                    while !head.ends_with(b"\r\n\r\n") {
                        let mut byte = [0]; stream.read_exact(&mut byte).unwrap(); head.push(byte[0]);
                    }
                    let head = String::from_utf8(head).unwrap();
                    let length: usize = head.lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|n| n.trim().parse().unwrap())).unwrap();
                    stream.read_exact(&mut vec![0;length]).unwrap();
                    write!(stream,"HTTP/1.1 {status} OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
                }));
            }
            for worker in workers {
                worker.join().unwrap();
            }
        });
        (url, thread)
    }
    #[test]
    fn concurrent_http_streams_have_usage_counts_and_hashes() {
        let body = "data: {\"choices\":[{\"text\":\"hello\"}]}\n\ndata: {\"choices\":[{\"text\":\" world\"}]}\n\ndata: {\"usage\":{\"completion_tokens\":2}}\n\ndata: [DONE]\n\n";
        let (url, server) = mock(8, body, 200);
        let (summary, results) = run(
            &url,
            vec![json!({"prompt":"fixed"}); 8],
            4,
            Duration::from_secs(5),
            None,
        );
        server.join().unwrap();
        assert_eq!(summary["successful"], 8);
        assert_eq!(summary["completion_tokens"], 16);
        assert!(summary["itl_ms_p95"].is_null());
        for result in &results {
            assert_eq!(result["content_events"], 2);
            assert_eq!(result["inter_event_ms"].as_array().unwrap().len(), 1);
            assert_eq!(result["output_sha256"], json!(hash(b"hello world")));
        }
    }
    #[test]
    fn missing_usage_is_not_estimated() {
        let (url, server) = mock(
            1,
            "data: {\"choices\":[{\"text\":\"x\"}]}\n\ndata: [DONE]\n\n",
            200,
        );
        let (summary, _) = run(&url, vec![json!({})], 1, Duration::from_secs(5), None);
        server.join().unwrap();
        assert_eq!(summary["successful"], 1);
        assert!(summary["completion_tokens_per_second"].is_null());
    }
    #[test]
    fn http_errors_sse_errors_and_truncation_fail() {
        for (status, body) in [
            (503, ""),
            (200, "data: {\"error\":\"failed\"}\n\n"),
            (200, "data: {\"choices\":[]}\n\n"),
        ] {
            let (url, server) = mock(1, body, status);
            let result = request_one(&url, &json!({}), Duration::from_secs(5), None);
            server.join().unwrap();
            assert_eq!(result["ok"], false);
            assert!(result["error"].is_string());
        }
    }
    #[test]
    fn nearest_rank_percentile() {
        assert_eq!(
            percentile(&(1..=100).map(f64::from).collect::<Vec<_>>(), 0.95),
            Some(95.0)
        );
        assert_eq!(percentile(&[], 0.95), None);
    }
}
