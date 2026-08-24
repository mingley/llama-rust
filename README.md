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

## Language tax: Q8_0 NEON SDOT GEMV

`langtax/` is llama.cpp's ARM NEON `nrc=1` Q8_0 vec-dot, lifted 1:1, C vs Rust. f32 scales (hardware-equivalent). M=K=4096, 8 iters.

```
clang -O3 -mcpu=native -o langtax/q8_gemv_c langtax/q8_gemv.c
RUSTFLAGS='-C target-cpu=native' cargo build --release --manifest-path langtax/Cargo.toml
./langtax/q8_gemv_c
./langtax/target/release/q8_gemv
```

| impl | gemv/s | weight GiB/s | vs C |
|---|---:|---:|---:|
| C `-O3 -mcpu=native` | 3000–3061 | 52.7–53.8 | 1.00 |
| Rust `--release -C target-cpu=native` | 3065–3108 | 53.9–54.6 | 1.01–1.03 |
| Rust with software fp16 in the inner loop | ~1064 | 17.7 | 0.35 |

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
