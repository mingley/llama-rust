# llama-rust-perf

Counterfactual: what tokens/s you get if you rewrite [llama.cpp](https://github.com/ggml-org/llama.cpp) in Rust.

Host language is not the limiter. Decode is weight-bandwidth bound. Same kernel in C vs Rust is **1.00x**. A naive kernel port that botches fp16 is **0.35x**.

## This machine

Apple M4 Pro 10P+4E+20GPU, 48 GiB, 273 GB/s unified. macOS 26.6.2. rustc 1.98.0. Apple clang 21.0.0.

llama.cpp `f280b26` (2026-08-24), Metal + Accelerate, flash-attn on.

Model: `Qwen/Qwen2.5-3B-Instruct-GGUF` Q4_K_M, 1.95 GiB, 3.40B. Not in this repo.

```
./llama.cpp/build/bin/llama-bench \
  -m models/qwen2.5-3b-instruct-q4_k_m.gguf \
  -ngl 99 -fa 1 -p 512,2048 -n 128 -r 8
```

| test | t/s |
|---|---:|
| pp512 Metal | **1090.81 ± 3.79** |
| pp2048 Metal | **1012.14 ± 29.56** |
| tg128 Metal | **88.27 ± 1.28** |
| pp512 `-ngl 0` | 222.99 ± 6.36 |
| tg128 `-ngl 0` | 32.41 ± 2.73 |

`88.27 tok/s * 1.95 GiB = 172 GiB/s` (~65% of 273 GB/s).

Same SKU on LocalScore: Llama 3.1 8B Q4_K 32.7 gen / 361 pp. Bandwidth scaling of the 3B run predicts ~38 t/s for 8B.

## Q8_0 GEMV pre/post (gating)

Same protocol as the C binary: M=4096, K=4096, 8 timed iterations, report `gemv/s`. C source is frozen. Fresh C pre (not the old README band):

```
clang -O3 -mcpu=native -o langtax/q8_gemv_c langtax/q8_gemv.c
./langtax/q8_gemv_c
RUSTFLAGS='-C target-cpu=native' cargo build --release --manifest-path langtax/Cargo.toml
./langtax/target/release/q8_gemv
```

| run | gemv/s | vs C pre |
|---|---:|---:|
| **C pre** `lang=C` clang `-O3 -mcpu=native` | **3430.59** | 1.00 |
| **Rust post 1** `lang=Rust` 8-thread NEON | **13234.08** | **3.86x** |
| **Rust post 2** (consecutive process) | **17457.72** | **5.09x** |

Both consecutive Rust launches are ≥2× the live C pre. Rust `gemv_q8_0` (what the CLI times) is an 8-worker persistent rayon pool over row chunks, 2-row NEON SDOT per worker. `cargo test --release --lib gemv_q8_0_matches_independent_scalar` drives that same function against an independent scalar pack-dot.

Single-thread identical-ISA kernels were ~1.00× (C 3000–3061 vs Rust 3065–3108). The ≥2× is extra P-cores, not codegen magic. Software-fp16 inner loop was 0.35× — kernel fidelity still matters.

## If you wrote llama.cpp in Rust

| rewrite | this Mac, 3B Q4_K_M pp / tg |
|---|---|
| FFI around ggml (`llama-cpp-2`) | 1091 / 88 |
| Rust host + same Metal shaders | 1091 / 88 |
| Mature Rust engine, c=1 (mistral.rs / ferrum) | ~800–1100 / 75–90 |
| Naive quantized (candle-class Metal) | ~680 / ~52 |
| Serving, c=16 (ferrum vs llama.cpp on M1 Max) | +36–44% aggregate tok/s |

CUDA is where Rust engines currently beat llama.cpp, via kernels not language: mistral.rs v0.8.2 is 1.8–2.2x prefill, 1.09–1.24x decode vs llama.cpp on GB10/B200.

llama.cpp HEAD is ~782k LOC. The hot path on this Mac is ~12k lines of Metal Shading Language. A Rust llama.cpp still writes those shaders.
