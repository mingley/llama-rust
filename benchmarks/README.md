# Measured server comparison

The standalone Rust `inference-serve-bench` crate measures a running OpenAI-compatible `/v1/completions`
streaming endpoint with blocking HTTP/SSE requests. Use the same prompt file,
request count, concurrency, generation limit, and model alias for llama-rust,
vLLM, and llama.cpp runs. It cycles prompts in fixed order, warms up before
timing, and uses greedy generation. No model downloads or server launches
are performed by the harness.

```bash
cargo test --manifest-path benchmarks/Cargo.toml --locked
cargo run --release --manifest-path benchmarks/Cargo.toml --locked -- \
  --url http://127.0.0.1:8080/v1/completions --model comparison \
  --prompts benchmarks/prompts.jsonl --requests 128 --concurrency 8 \
  --max-tokens 128 --warmup 8 --usage --output /tmp/llama-rust-results.json
```

Run that command against each server separately, changing only the URL and
output path. Configure the same served model alias (`comparison`) where
supported. Sweep concurrency separately (e.g. 1, 4, 8, 16) and repeat each
configuration at least three times. Compare the emitted workload hash and
per-request hashes to verify identical request bodies. If aliases must differ,
record that difference and compare the other configuration fields explicitly.
For authentication, set `INFERENCE_API_KEY` in the environment. Its value is
not included in the report. Avoid credentials in the URL.

`--usage` requests final streamed usage. If an endpoint rejects
`stream_options`, omit `--usage` consistently for all runs. Token throughput
is **null**, rather than estimated from text length or chunk count, unless
every successful request reports `usage.completion_tokens`. Missing counts
are a measurement limitation, not zero tokens. The harness exits nonzero
when any measured request fails or a warmup fails; reports preserve each
HTTP/SSE error and incomplete streams count as failures.

Metrics:

- TTFT: request start through first nonempty completion text event. This
  includes client HTTP setup and transport, not merely GPU execution.
- Request latency p95 and successful requests/second: measured wall clock.
  This is a closed-loop concurrency test, not a fixed arrival-rate SLO test.
- Completion tokens/second: successful reported token counts divided by the
  whole measured run, including time spent on failed requests.
- Inter-content-event interval p95: observed SSE content spacing. Servers may
  combine multiple tokens in one chunk, so this is **not token ITL**.
- TPOT p95: `(request duration - TTFT) / (completion_tokens - 1)` for successful
  requests with known counts above one. It includes terminal protocol overhead.
- `itl_ms_p95` stays null: exact per-token intervals require server-side token
  timestamps or a verified one-token-per-event protocol. No token timings are
  invented from chunk boundaries.

Each report records configuration, workload hash, request/output hashes, all
per-request timings, success counts, and errors. Output hashes can reveal
generation differences but are not a numerical correctness oracle. P95s use
nearest rank; intervals pool all successful requests. Short runs have noisy
percentiles. `--timeout` bounds each complete HTTP request, including its streamed body; it is
not a deadline for the whole benchmark run.

For a fair published comparison, record GPU model/count, driver and CUDA
versions, CPU/RAM, server revision and exact launch arguments, checkpoint
revision, tokenizer/chat template, quantization, context length, prefix-cache
state, batching and KV settings, power limits, and other active workloads.
Use identical weights and precision when supported; if GGUF and another
format differ, state that the measurements are deployment comparisons, not
isolated engine comparisons. Repeated prompts deliberately exercise a warm
prefix cache; use a separate fixed set of unique prompts for cold-prefix runs.
Do not claim a speedup from these scripts until real matched-hardware runs
and semantic validation support it.

The Rust mock-server tests validate concurrent SSE parsing, usage-derived counts,
error accounting, incomplete streams, and percentile arithmetic. Mock results
provide no evidence about inference performance.

The crate has its own workspace and lockfile so benchmark HTTP dependencies do
not change inference runtime dependencies. Build with `cargo build --release
--manifest-path benchmarks/Cargo.toml --locked` before timing. Each measured
request uses a fresh HTTP client; this consistently includes client/connection
setup overhead rather than relying on server-specific keep-alive reuse.
