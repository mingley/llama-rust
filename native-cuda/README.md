# Native CUDA Rust GEMV experiment

This standalone crate is the first hardware execution seam for evaluating
[NVIDIA CUDA Rust](https://github.com/NVIDIA/cuda-rust). It uses NVIDIA's
`cuda-oxide` Rust kernel compiler and `cuda-core` buffers/streams, pinned to
upstream commit `9df9e7cef561e03a066230b3bb433b3d8285da81`.

`native-gemv` allocates device weights/input/output, launches a row-major F32
GEMV written in Rust, copies results back, and compares every output against
an independent scalar f64 oracle. Three shapes cover a single element,
non-warp-multiple dimensions, and multiple blocks. A failed initialization,
launch, synchronization or numerical check exits nonzero; there is no CPU
fallback and no simulated GPU timing.

This kernel is a correctness baseline with one thread per output row. It is
not tuned, is not wired into the serving engine, and does not establish a
speed advantage over vLLM or llama.cpp. The CPU engine and its dependency
policy remain unchanged.

## CPU-only checks

```bash
cargo test --manifest-path native-cuda/Cargo.toml --lib
cargo fmt --manifest-path native-cuda/Cargo.toml --check
```

The optional GPU dependencies are resolved into this crate's lockfile but
are not compiled for these checks. The pure Rust checks verify buffer/shape
validation and rejection of nonfinite or incorrect GPU results; they do not
validate device compilation or execution.

## Native GPU build and validation

Follow the pinned upstream
[installation instructions](https://github.com/NVIDIA/cuda-rust/blob/9df9e7cef561e03a066230b3bb433b3d8285da81/cuda-oxide/cuda-oxide-book/getting-started/installation.md).
Requirements include Rust **1.99.1**, its `rust-src`, `rustc-dev`, and
`llvm-tools` components, CUDA Toolkit **13 or newer**, libclang with development
headers, and an NVIDIA GPU/compatible driver. `cuda-oxide` also needs an
NVPTX-capable LLVM as described upstream. Configure `CUDA_TOOLKIT_PATH` if the
toolkit is outside the standard install location.

```bash
rustup toolchain install 1.99.1 --profile minimal
rustup component add --toolchain 1.99.1 rust-src rustc-dev llvm-tools
cargo +1.99.1 install --git https://github.com/NVIDIA/cuda-rust.git \
  --rev 9df9e7cef561e03a066230b3bb433b3d8285da81 cargo-oxide --locked
cd native-cuda
cargo +1.99.1 oxide doctor
cargo +1.99.1 oxide run --features cuda
```

The local `.cargo/config.toml` matches NVIDIA's standalone scaffold and enables
`RUSTC_BOOTSTRAP` for its compiler-internal proc-macro APIs. Run native commands
from this directory so Cargo reads that configuration. Use `cargo oxide`,
rather than ordinary `cargo run`, to embed compiled device code. CUDA Rust is an early-stage project: keeping compiler, macro, device,
and host dependencies at the same revision is intentional. On a GPU runner,
run this program under Compute Sanitizer as well before integrating the
kernel into a persistent device-weight store.

## Validation status

The development environment has no CUDA Toolkit, libclang, NVIDIA driver/GPU,
or `cargo-oxide`. CPU checks can run here. Native compilation and GPU execution
remain pending on a configured GPU runner; this experiment must not be enabled
as a production backend until those checks pass.
