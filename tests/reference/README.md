# Real-model reference fixtures

Captured from llama.cpp so the crate can be checked against real quantized
weights instead of only against writer-built tiny GGUFs and the in-tree oracle.

This matters because a production `f16_to_f32` bug used to be invisible:
every per-dtype oracle called that same primitive, which is how a
subnormal binary16 bug that halved real Q4_K/Q6_K weights survived 221
passing tests. Oracles now use `oracle_f16_to_f32` (IEEE arithmetic, not
bit-surgery). The writer-built fixtures still use hand-picked scales that
never land in the affected ranges, so this llama.cpp capture remains the
greedy-identity check on a real checkpoint.

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

**Use an absolute path.** `cargo test` runs with the crate directory as the
working directory, so a relative `models` resolves against `langtax/`, not the
repo root. That is not a cosmetic detail: it originally made the CI job report
green in 0.00 s having loaded no weights. The test now distinguishes the two
cases — unset means "not requested" and skips, while set-but-unopenable fails
loudly — so a repeat of that mistake shows up as a red job rather than a false
green.

```
mkdir -p models
curl -L -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf?download=true"

LLAMA_RUST_REAL_MODEL_DIR="$PWD/models" cargo test --release --lib real_model -- --nocapture
```

When it really runs it prints the resolved model path and takes seconds, not
milliseconds. A 0.00 s pass means it skipped.

## Second fixture (Llama NORM control)

The Qwen2.5-0.5B capture is **NEOX** RoPE. PLAN.md also wants a **NORM**
Llama control so the two official pairing conventions cannot silently
swap. Intended file once someone has it on disk:

`Llama-3.2-1B-Instruct-Q4_K_M.gguf` (or another small official `llama`
GGUF whose `rope.scaling` / `rope.type` is the Llama NORM walk).

Capture with the same `ref.cpp` command below, write
`llama-3.2-1b-instruct-q4_k_m.json` next to this README, and point
`LLAMA_RUST_REAL_MODEL_DIR` at the directory. Do not download Hugging Face
checkpoints in CI.

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
