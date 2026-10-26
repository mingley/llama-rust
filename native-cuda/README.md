# NVIDIA CUDA Rust matrix execution

This crate supplies native Rust GPU GEMV and batched GEMM kernels for the default
NVIDIA execution path. Kernels use NVIDIA's `cuda-oxide` compiler and
`cuda-core` buffers/streams, pinned to upstream commit
`9df9e7cef561e03a066230b3bb433b3d8285da81`.

`CudaGemv` retains the device context, stream and compiled kernel module.
`upload` creates an immutable `CudaMatrix` in device memory; `gemv_resident`
and `gemm_resident` reuse that matrix without uploading its weights again.
GEMM accepts token-major input and output and launches the batch together.
The kernels are a simple correctness baseline, with one thread per output;
there is no CPU fallback after GPU selection and no simulated timing.

The `native-gemv` binary uploads three matrix shapes and compares device results
against an independent scalar f64 oracle. The host API checks dimensions,
vector lengths, indexing limits, and matrix/executor ownership before launch.

## Build and run

Native CUDA is enabled by default. Follow the pinned upstream
[installation instructions](https://github.com/NVIDIA/cuda-rust/blob/9df9e7cef561e03a066230b3bb433b3d8285da81/cuda-oxide/cuda-oxide-book/getting-started/installation.md).
Requirements include Rust **1.99.1**, its `rust-src`, `rustc-dev`, and
`llvm-tools` components, CUDA Toolkit **13 or newer**, libclang development
headers, an NVPTX-capable LLVM, and an NVIDIA GPU/compatible driver.
Configure `CUDA_TOOLKIT_PATH` for a nonstandard toolkit installation.

```bash
rustup toolchain install 1.99.1 --profile minimal
rustup component add --toolchain 1.99.1 rust-src rustc-dev llvm-tools
cargo +1.99.1 install --git https://github.com/NVIDIA/cuda-rust.git \
  --rev 9df9e7cef561e03a066230b3bb433b3d8285da81 cargo-oxide --locked
cargo +1.99.1 oxide doctor
cargo +1.99.1 oxide build -- --release -p llama-native-cuda --bin native-gemv
./target/release/native-gemv
```

Use `cargo oxide`, rather than ordinary `cargo run`, to embed device code.
The repository root `.cargo/config.toml` mirrors upstream's scaffold and sets
`RUSTC_BOOTSTRAP=1` because its macros use compiler internals on the pinned
stable toolchain. Compiler, macro, device, and host dependencies
must remain at the same revision.

## CPU-only validation

```bash
cargo test -p llama-native-cuda --no-default-features --lib
cargo clippy -p llama-native-cuda --no-default-features --lib -- -D warnings
cargo fmt --all --check
```

These checks exercise shape validation and oracle result comparison without
compiling GPU code. They do not validate GPU execution.

## Validation status

The development environment has no NVIDIA GPU/driver. Host validation and
device compilation status are reported separately from hardware execution;
GPU execution remains pending on a configured GPU runner. Default selection is a product policy, not evidence
of hardware validation or a speed advantage over vLLM or llama.cpp. Run
`native-gemv` and model parity tests under Compute Sanitizer on GPU hardware
before relying on numerical correctness or memory safety across real workloads.
