# Traces

Measured MoE access traces. Replay them; do not paste fictional hit rates.

| File | What |
| --- | --- |
| `cycling.jsonl` | Synthetic 3-expert cycle. LRU `--capacity 2` = 0 hits; oracle = 11/24. |
| `tiny-qwen3moe.jsonl` | Writer-built Qwen3MoE, `ab` + 8 tokens. Working set is 2 experts. |
| `tiny-llama-moe.jsonl` | Writer-built official llama MoE, same prompt. Working set is 2 experts. |
| `tiny-qwen2moe.jsonl` | Writer-built Qwen2MoE, same prompt. Working set is 2 experts. |

Writer-built tinies do **not** answer “is residency predictable on a 320B MoE?”.
They prove the decoder emits JSONL and that `expertvm replay` is honest.
A real Qwen3MoE / Qwen2MoE GGUF is the next input.

New traces include optional `w` (router mass in permille) and optional
`p` (content-addressed hash of the token-id prefix). Older lines without
`w` or `p` still parse; `analyze` then omits `mass‰`, and `--prefix-cache`
is a no-op when every event has `p` missing.

```
gguf_gemv write-tiny-qwen3moe tiny-qwen3moe.gguf
gguf_gemv trace tiny-qwen3moe.gguf -p ab -n 8 --out tests/traces/tiny-qwen3moe.jsonl
gguf_gemv write-tiny-qwen3moe-2layer tiny-qwen3moe-2layer.gguf
gguf_gemv trace tiny-qwen3moe-2layer.gguf -p ab -n 8 --out /tmp/tiny-qwen3moe-2layer.jsonl
gguf_gemv engine tiny-qwen3moe.gguf -p a -p b --kv-page 2 --trace-out /tmp/engine.jsonl
expertvm replay tests/traces/tiny-qwen3moe.jsonl --capacity 2
expertvm replay tests/traces/cycling.jsonl --capacity 2
expertvm bench adversarial --capacity 2 --profile cheap
expertvm remote tests/traces/cycling.jsonl --expert-bytes 1048576
infer-bench trace tests/traces/cycling.jsonl --capacity 2 --profile h100
infer-bench adversarial --capacity 2 --tokens 64 --experts 16
infer-bench remote tests/traces/cycling.jsonl --expert-bytes 1048576
```
