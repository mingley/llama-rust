# Real-model reference fixtures

Captured from llama.cpp so the crate can be checked against real quantized
weights instead of only against writer-built tiny GGUFs and the in-tree oracle.

This matters because the in-tree oracle is not fully independent. Every
per-dtype oracle calls `crate::fp16::f16_to_f32`, so a bug in that primitive is
invisible to the whole suite — that is exactly how a subnormal binary16 bug
that halved real Q4_K/Q6_K weights survived 221 passing tests. The writer-built
fixtures also use hand-picked scales that never land in the affected ranges.

## Files

`qwen2.5-0.5b-instruct-q4_k_m.json` — tokenization, prefill logit summary, the
top-20 logits, and the greedy continuation for one fixed prompt.

## Model identity

| field | value |
|---|---|
| HF repo | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` |
| file | `qwen2.5-0.5b-instruct-q4_k_m.gguf` |
| architecture | `qwen2` (NEOX rope) |
| tensor dtypes | Q5_0 x133, F32 x121, Q8_0 x13, Q6_K x12, Q4_K x12 |

This file is a useful regression target because it mixes five dtypes, uses the
NEOX rope convention, and has `add_bos_token=false`.

## Running the test

The model is not committed (it is ~470 MB). Point the test at a directory
containing it; the test skips cleanly when the variable is unset.

```
mkdir -p models
curl -L -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf?download=true"

LLAMA_RUST_REAL_MODEL_DIR=models cargo test --release --lib real_model
```

## Regenerating the reference

Build llama.cpp and dump the reference with `tests/reference/ref.cpp`, which
links against `libllama` so both sides read one GGUF on one host.

```
git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/lcpp
cd /tmp/lcpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
      -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build build --target llama-cli -j"$(nproc)"

g++ -O2 -std=c++17 -I/tmp/lcpp/include -I/tmp/lcpp/ggml/include \
    tests/reference/ref.cpp -o /tmp/ref \
    -L/tmp/lcpp/build/bin -lllama -lggml -lggml-base \
    -Wl,-rpath,/tmp/lcpp/build/bin

/tmp/ref models/qwen2.5-0.5b-instruct-q4_k_m.gguf "The capital of France is" 32
```

If `c++` is clang and fails with `cannot find -lstdc++`, pass gcc explicitly as
shown above.

## Interpreting a logit mismatch

Exact logit agreement is not expected and should not be asserted. llama.cpp's
K-quant dot products quantize the activation vector to Q8_K before
multiplying, while this crate dequantizes weights to f32 and multiplies against
f32 activations, so the two do genuinely different arithmetic. Measured
agreement on the prompt above is a max top-20 logit delta around 0.17 with an
identical argmax.

The meaningful acceptance criteria are, in order:

1. token ids match exactly,
2. argmax matches,
3. the greedy continuation matches over the full window,
4. logit deltas stay small (well under a logit).

A whole-logit-unit divergence, or a changed greedy continuation, indicates a
real bug rather than arithmetic noise.
