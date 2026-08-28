# Traces

Small MoE access traces checked in so `expertvm` CLI and docs stay honest.

| File | What |
| --- | --- |
| `cycling.jsonl` | 24 tokens, 3 experts cycling. LRU with `--capacity 2` thrashes (0 hits); oracle does not. |

Generate real traces from the decoder:

```
gguf_gemv trace <gguf> -p PROMPT -n N --out trace.jsonl
```

Do not paste fictional hit rates into PLAN.md or README. Replay the file.
