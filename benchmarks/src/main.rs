use inference_serve_bench::{hash, request_one, run};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::time::Duration;

#[expect(
    clippy::disallowed_methods,
    reason = "standalone synchronous benchmark CLI intentionally uses blocking file IO"
)]
fn execute() -> Result<bool, Box<dyn std::error::Error>> {
    let mut args = BTreeMap::new();
    let mut usage = false;
    let mut iter = std::env::args().skip(1);
    while let Some(key) = iter.next() {
        if key == "--help" {
            println!("inference-serve-bench --url URL --model NAME --prompts FILE --output FILE [--requests 32] [--concurrency 4] [--max-tokens 128] [--warmup 2] [--timeout 120] [--usage]");
            return Ok(true);
        }
        if key == "--usage" {
            usage = true;
            continue;
        }
        if ![
            "--url",
            "--model",
            "--prompts",
            "--output",
            "--requests",
            "--concurrency",
            "--max-tokens",
            "--warmup",
            "--timeout",
        ]
        .contains(&key.as_str())
        {
            return Err(format!("unknown argument {key}").into());
        }
        let value = iter
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        if args.insert(key.clone(), value).is_some() {
            return Err(format!("duplicate argument {key}").into());
        }
    }
    let required = |name: &str| {
        args.get(name)
            .cloned()
            .ok_or_else(|| format!("missing {name}"))
    };
    let url = required("--url")?;
    let model = required("--model")?;
    let prompts_path = required("--prompts")?;
    let output = required("--output")?;
    let number = |key: &str, default: usize| -> Result<usize, Box<dyn std::error::Error>> {
        Ok(args
            .get(key)
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(default))
    };
    let requests = number("--requests", 32)?;
    let concurrency = number("--concurrency", 4)?;
    let max_tokens = number("--max-tokens", 128)?;
    let warmup = number("--warmup", 2)?;
    let timeout_seconds: f64 = args
        .get("--timeout")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(120.0);
    if requests == 0
        || concurrency == 0
        || max_tokens == 0
        || !timeout_seconds.is_finite()
        || timeout_seconds <= 0.0
    {
        return Err("counts and timeout must be positive".into());
    }
    let timeout = Duration::try_from_secs_f64(timeout_seconds)?;
    let mut prompts = Vec::new();
    for line in std::fs::read_to_string(&prompts_path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
    {
        let entry: Value = serde_json::from_str(line)?;
        let prompt = entry["prompt"]
            .as_str()
            .filter(|p| !p.is_empty())
            .ok_or("prompts must be nonempty strings")?;
        prompts.push(prompt.to_owned());
    }
    if prompts.is_empty() {
        return Err("prompt file is empty".into());
    }
    let payloads: Vec<_> = (0..requests).map(|index| {
        let mut payload = json!({"model":model,"prompt":prompts[index % prompts.len()],"max_tokens":max_tokens,"temperature":0,"stream":true,"n":1});
        if usage { payload["stream_options"] = json!({"include_usage":true}); }
        payload
    }).collect();
    let key = std::env::var("INFERENCE_API_KEY").ok();
    for index in 0..warmup {
        let result = request_one(
            &url,
            &payloads[index % payloads.len()],
            timeout,
            key.as_deref(),
        );
        if result["ok"] != true {
            return Err(format!("warmup failed: {}", result["error"]).into());
        }
    }
    let workload_hash = hash(&serde_json::to_vec(&payloads)?);
    let (summary, results) = run(&url, payloads, concurrency, timeout, key);
    let success = summary["errors"] == 0;
    let report = json!({"schema_version":1,"config":{"url":url,"model":model,"prompts":prompts_path,"requests":requests,"concurrency":concurrency,"max_tokens":max_tokens,"warmup":warmup,"timeout_seconds":timeout_seconds,"usage":usage,"workload_sha256":workload_hash},"summary":summary,"results":results});
    std::fs::write(output, serde_json::to_string_pretty(&report)? + "\n")?;
    println!("{}", serde_json::to_string_pretty(&report["summary"])?);
    Ok(success)
}

fn main() -> std::process::ExitCode {
    match execute() {
        Ok(true) => std::process::ExitCode::SUCCESS,
        Ok(false) => std::process::ExitCode::FAILURE,
        Err(error) => {
            eprintln!("benchmark failed: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
