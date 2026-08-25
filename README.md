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
| pp512 Metal | **1096.77 ± 1.11** |
| pp2048 Metal | **1012.14 ± 29.56** |
| tg128 Metal | **90.92 ± 3.50** |
| pp512 `-ngl 0` | 222.99 ± 6.36 |
| tg128 `-ngl 0` | 32.41 ± 2.73 |

`90.92 tok/s * 1.95 GiB = 177 GiB/s` (~65% of 273 GB/s). pp2048 and `-ngl 0` rows are prior captures on this machine.

Same SKU on LocalScore: Llama 3.1 8B Q4_K 32.7 gen / 361 pp. Bandwidth scaling of the 3B run predicts ~38 t/s for 8B.

## Q8_0 GEMV pre/post (gating)

Same protocol as the C binary: M=4096, K=4096, 8 timed iterations, report `gemv/s`. `langtax/q8_gemv.c` is frozen (1-thread NEON). Fresh C pre, this capture:

```
clang -O3 -mcpu=native -o langtax/q8_gemv_c langtax/q8_gemv.c
./langtax/q8_gemv_c
RUSTFLAGS='-C target-cpu=native' cargo build --release --manifest-path langtax/Cargo.toml
./langtax/target/release/q8_gemv
```

| run | gemv/s | vs C pre |
|---|---:|---:|
| **C pre** `lang=C` clang `-O3 -mcpu=native` | **3267.08** | 1.00 |
| **Rust post 1** `lang=Rust kernel=q8_0_safe` | **7011.39** | **2.15x** |
| **Rust post 2** (consecutive process) | **6758.90** | **2.07x** |

Both consecutive Rust launches are ≥2×. `gemv_q8_0` is **fully safe** (`#![forbid(unsafe_code)]`): slice loops + a 10-worker persistent rayon pool. LLVM emits `sdot`. `cargo test --release --lib` drives `gemv_q8_0` and `gemv_q4_0` against independent scalar pack-dots.

## Extra cells (not the 4096×4096 Q8 gate)

1-thread C (`langtax/extra.c`) vs the same safe Rust functions (`langtax/src/extra.rs`). niter=8.

```
clang -O3 -mcpu=native -o langtax/extra_c langtax/extra.c
./langtax/extra_c
RUSTFLAGS='-C target-cpu=native' cargo run --release --manifest-path langtax/Cargo.toml --bin extra
```

**Q8_0 GEMV size sweep** (same kernel as the gate, different M=K):

| M=K | C gemv/s | Rust gemv/s | Rust/C |
|---:|---:|---:|---:|
| 1024 | 51962.10 | 19814.24 | **0.38x** |
| 2048 | 12109.74 | 21798.37 | **1.80x** |
| 8192 | 750.60 | 2689.83 | **3.58x** |

1024 is too small to amortize 10 workers; 8192 is where the pool pays.

**Q4_0 × Q8_0 GEMV** (llama.cpp decode-shaped: 4-bit weights, 8-bit activations, M=K=4096):

| run | gemv/s | vs C |
|---|---:|---:|
| C `kernel=q4_0` 1-thread | 388.55 | 1.00 |
| Rust `kernel=q4_0_safe` 10-worker | 1942.81 | **5.00x** |

Single-thread identical-ISA kernels were ~1.00×. The ≥2× on the 4096 Q8 gate is extra P-cores, not language. Software-fp16 inner loop was 0.35×.

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
