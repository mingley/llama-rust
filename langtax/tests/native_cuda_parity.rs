//! Hardware parity for the actual quantized model dispatch, not just the kernel.
//!
//! Generate scalar logits with `LLAMA_CUDA_ORACLE_WRITE=/tmp/scalar-logits.txt
//! cargo test --release -p llama-rust --no-default-features --test native_cuda_parity`.
//! Then run `LLAMA_CUDA_ORACLE_READ=/tmp/scalar-logits.txt cargo oxide test --
//! --release -p llama-rust --test native_cuda_parity` on a working CUDA device.
//! Default builds fail without a driver; they never skip or use a CPU fallback.
use llama_rust::*;
use std::io::{Read, Write};

// Numerical checks deliberately use f64 row dots, outside the dispatched
// GEMV/GEMM path. Dequantization is shared because that is CPU-only in both
// backends; these tests validate conversion, residency, layout, and execution.
type Dequant = fn(usize, &[u8], &mut [f32]) -> Result<(), QuantError>;
type Gemv = fn(usize, &[u8], &[f32], &mut [f32]) -> Result<(), QuantError>;
type Gemm = fn(usize, usize, &[u8], &[f32], &mut [f32]) -> Result<(), QuantError>;

struct Format {
    name: &'static str,
    cols: usize,
    pack_row: fn(usize) -> Vec<u8>,
    unpack: Dequant,
    gemv: Gemv,
    gemm: Gemm,
}

fn values(row: usize, cols: usize) -> Vec<f32> {
    (0..cols)
        .map(|col| {
            let value = i16::try_from((row * 7 + col * 11) % 31).unwrap_or(0) - 15;
            f32::from(value) / 16.0
        })
        .collect()
}

fn formats() -> Vec<Format> {
    vec![
        Format {
            name: "F32",
            cols: 33,
            pack_row: |r| pack_f32(&values(r, 33)),
            unpack: dequant_f32_row,
            gemv: gemv_f32,
            gemm: gemm_f32,
        },
        Format {
            name: "F16",
            cols: 33,
            pack_row: |r| pack_f16(&values(r, 33)),
            unpack: dequant_f16_row,
            gemv: gemv_f16,
            gemm: gemm_f16,
        },
        Format {
            name: "BF16",
            cols: 33,
            pack_row: |r| pack_bf16(&values(r, 33)),
            unpack: dequant_bf16_row,
            gemv: gemv_bf16,
            gemm: gemm_bf16,
        },
        Format {
            name: "Q4_0",
            cols: QK4_0,
            pack_row: |r| {
                let q =
                    std::array::from_fn(|i| i8::try_from((r * 3 + i * 7) % 16).unwrap_or(0) - 8);
                pack_q4_0_from_i4(0.125, &q).to_vec()
            },
            unpack: dequant_q4_0_row,
            gemv: gemv_q4_0_f32,
            gemm: gemm_q4_0_f32,
        },
        Format {
            name: "Q8_0",
            cols: QK8_0,
            pack_row: |r| {
                let q =
                    std::array::from_fn(|i| i8::try_from((r * 3 + i * 7) % 127).unwrap_or(0) - 63);
                pack_q8_0_block(0.015625, &q).to_vec()
            },
            unpack: dequant_q8_0_row,
            gemv: gemv_q8_0_f32,
            gemm: gemm_q8_0_f32,
        },
        Format {
            name: "Q4_K",
            cols: QK_K,
            pack_row: |r| {
                let q = std::array::from_fn(|i| u8::try_from((r * 3 + i * 7) % 16).unwrap_or(0));
                pack_q4_k_block(
                    0.03125,
                    0.015625,
                    &[1, 2, 3, 4, 5, 6, 7, 8],
                    &[2, 1, 0, 3, 1, 4, 2, 5],
                    &q,
                )
                .to_vec()
            },
            unpack: dequant_q4_k_row,
            gemv: gemv_q4_k_f32,
            gemm: gemm_q4_k_f32,
        },
        Format {
            name: "Q6_K",
            cols: QK_K,
            pack_row: |r| {
                let q =
                    std::array::from_fn(|i| i8::try_from((r * 3 + i * 7) % 64).unwrap_or(0) - 32);
                pack_q6_k_block(
                    0.0078125,
                    &[1, 2, -3, 4, 5, -6, 7, 8, -1, 2, 3, -4, 5, 6, -7, 8],
                    &q,
                )
                .to_vec()
            },
            unpack: dequant_q6_k_row,
            gemv: gemv_q6_k_f32,
            gemm: gemm_q6_k_f32,
        },
    ]
}

fn oracle(
    format: &Format,
    rows: usize,
    packed: &[u8],
    input: &[f32],
) -> Result<Vec<f64>, QuantError> {
    let mut weights = vec![0.0; rows * format.cols];
    for (src, dst) in packed
        .chunks_exact(packed.len() / rows)
        .zip(weights.chunks_exact_mut(format.cols))
    {
        (format.unpack)(format.cols, src, dst)?;
    }
    Ok(input
        .chunks_exact(format.cols)
        .flat_map(|token| {
            weights.chunks_exact(format.cols).map(move |row| {
                row.iter()
                    .zip(token)
                    .map(|(&w, &x)| f64::from(w) * f64::from(x))
                    .sum()
            })
        })
        .collect())
}

fn check(actual: &[f32], expected: &[f64], name: &str) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && e.is_finite() && (f64::from(a) - e).abs() <= 1e-4 + 1e-4 * e.abs(),
            "{name} output {index}: expected {e}, got {a}"
        );
    }
}

#[test]
fn quantized_model_dispatch_matches_cpu_row_oracle() -> Result<(), QuantError> {
    // 257 rows crosses block boundaries, and each token has independent data.
    // Repeated token counts catch stale scratch contents; changes force resize.
    for format in formats() {
        let rows = 257;
        let mut packed: Vec<u8> = (0..rows).flat_map(format.pack_row).collect();
        for pass in 0..2 {
            for (round, tokens) in [1, 1, 3, 3, 7, 1].into_iter().enumerate() {
                let input: Vec<f32> = (0..tokens)
                    .flat_map(|token| values(token + round * 3 + pass, format.cols))
                    .collect();
                let expected = oracle(&format, rows, &packed, &input)?;
                let mut actual = vec![f32::NAN; rows * tokens];
                if tokens == 1 {
                    (format.gemv)(format.cols, &packed, &input, &mut actual)?;
                } else {
                    (format.gemm)(format.cols, tokens, &packed, &input, &mut actual)?;
                }
                check(&actual, &expected, format.name);
            }
            // Same backing pointer, same shape and dtype, different bytes:
            // residency must use content identity, never pointer identity.
            let row_bytes = packed.len() / rows;
            let replacement = (format.pack_row)(rows + 5);
            if let Some(first) = packed.get_mut(..row_bytes) {
                first.copy_from_slice(&replacement);
            }
        }
    }
    Ok(())
}

// End-to-end references are produced by a separate scalar build, so CUDA and
// CPU cannot accidentally exercise the same compiled matrix backend. The
// workflow preserves the snapshot across builds in RUNNER_TEMP.
type ModelSnapshots = Vec<(String, Vec<f32>)>;

fn model_logits() -> Result<ModelSnapshots, Box<dyn std::error::Error>> {
    let mut snapshots = Vec::new();
    for (name, bytes) in [
        ("llama-f32", tiny_llama_gguf()),
        ("llama-q4_0", tiny_q40_gguf()),
        ("llama-q4_k", tiny_q4k_embd_gguf()),
        ("qwen3-moe", tiny_qwen3moe_gguf()),
    ] {
        let model = Model::from_bytes(bytes)?;
        let mut dense = model.session(32)?;
        let mut paged = model.session_paged(32, 2)?;
        let dense_prefill = dense.prefill(&[1, 2, 3])?.to_vec();
        let paged_prefill = paged.prefill(&[1, 2, 3])?.to_vec();
        check(
            &paged_prefill,
            &dense_prefill
                .iter()
                .copied()
                .map(f64::from)
                .collect::<Vec<_>>(),
            name,
        );
        snapshots.push((format!("{name}/prefill"), dense_prefill));
        for (step, token) in [4, 5, 1].into_iter().enumerate() {
            let dense_logits = dense.decode(token)?.to_vec();
            let paged_logits = paged.decode(token)?.to_vec();
            check(
                &paged_logits,
                &dense_logits
                    .iter()
                    .copied()
                    .map(f64::from)
                    .collect::<Vec<_>>(),
                name,
            );
            snapshots.push((format!("{name}/decode-{step}"), dense_logits));
        }
        // Divergent full prompt exercises prefix reuse and replacement of KV.
        let dense_prompt = dense.prompt(&[1, 2, 4])?.to_vec();
        let paged_prompt = paged.prompt(&[1, 2, 4])?.to_vec();
        check(
            &paged_prompt,
            &dense_prompt
                .iter()
                .copied()
                .map(f64::from)
                .collect::<Vec<_>>(),
            name,
        );
        snapshots.push((format!("{name}/reused-prompt"), dense_prompt));
        // Scheduler jobs combine differently-sized prefill chunks and decode
        // tokens. Compare every continuation to its independent session.
        let prompts = [vec![1, 2, 3], vec![1, 2, 4, 5], vec![1, 2]];
        let mut cfg = EngineCfg::tiny();
        cfg.n_ctx = 32;
        cfg.pool_blocks = 64;
        cfg.max_seqs = 2;
        cfg.prefill_chunk = 2;
        let mut engine = Engine::new(model.llama(), cfg)?;
        let [first_prompt, second_prompt, third_prompt] = &prompts;
        let first = engine.add(first_prompt, 4)?;
        let second = engine.add(second_prompt, 4)?;
        let _progressed = engine.step()?;
        let third = engine.add(third_prompt, 4)?;
        engine.run()?;
        assert!(engine.stats().gemm_peak > 1, "{name}: batched path unused");
        for (index, (id, prompt)) in [first, second, third].into_iter().zip(&prompts).enumerate() {
            let mut session = model.session(32)?;
            let mut logits = session.prefill(prompt)?.to_vec();
            let mut expected = Vec::new();
            for _ in 0..4 {
                let token = argmax(&logits);
                expected.push(token);
                logits = session.decode(token)?.to_vec();
            }
            let output = engine.take(id).ok_or("missing finished engine sequence")?;
            assert_eq!(output.prompt, *prompt, "{name}: engine prompt changed");
            assert_eq!(
                output.generated, expected,
                "{name}: engine/session continuation mismatch"
            );
            let tokens = output
                .generated
                .into_iter()
                .map(|token| u16::try_from(token).map(f32::from))
                .collect::<Result<Vec<_>, _>>()?;
            snapshots.push((format!("{name}/batched-generated-{index}"), tokens));
        }
    }
    Ok(snapshots)
}

#[test]
fn end_to_end_models_match_scalar_snapshot() -> Result<(), Box<dyn std::error::Error>> {
    let snapshots = model_logits()?;
    if let Some(path) = std::env::var_os("LLAMA_CUDA_ORACLE_WRITE") {
        if cfg!(feature = "native-cuda") {
            return Err("CPU oracle must be generated with --no-default-features".into());
        }
        let mut text = String::new();
        for (name, logits) in &snapshots {
            text.push_str(name);
            for value in logits {
                if !value.is_finite() {
                    return Err("nonfinite scalar oracle".into());
                }
                text.push(' ');
                text.push_str(&value.to_string());
            }
            text.push('\n');
        }
        std::fs::File::create(path)?.write_all(text.as_bytes())?;
    }
    if let Some(path) = std::env::var_os("LLAMA_CUDA_ORACLE_READ") {
        let mut text = String::new();
        let _bytes = std::fs::File::open(path)?.read_to_string(&mut text)?;
        let lines: Vec<_> = text.lines().collect();
        assert_eq!(lines.len(), snapshots.len(), "oracle case count changed");
        for ((name, logits), line) in snapshots.iter().zip(lines) {
            let mut fields = line.split_whitespace();
            assert_eq!(
                fields.next(),
                Some(name.as_str()),
                "oracle case order changed"
            );
            let expected: Vec<f64> = fields.map(str::parse).collect::<Result<_, _>>()?;
            check(logits, &expected, name);
        }
    } else if cfg!(feature = "native-cuda") {
        return Err(
            "set LLAMA_CUDA_ORACLE_READ to a scalar-build snapshot (see native-cuda workflow)"
                .into(),
        );
    }
    Ok(())
}

#[test]
fn warp_boundary_shapes_match_independent_f64_dots() -> Result<(), QuantError> {
    for rows in [1, 7, 9, 257] {
        for cols in [63, 64, 65, 129, 1025] {
            let weights: Vec<f32> = (0..rows)
                .flat_map(|row| values(row, cols).into_iter().map(|value| value / 1.3))
                .collect();
            let packed = pack_f32(&weights);
            for round in 0..2 {
                for tokens in [1, 3] {
                    let input: Vec<f32> = (0..tokens)
                        .flat_map(|token| {
                            values(token + round * 17, cols)
                                .into_iter()
                                .map(|value| value / 1.7)
                        })
                        .collect();
                    let expected: Vec<f64> = input
                        .chunks_exact(cols)
                        .flat_map(|token| {
                            weights.chunks_exact(cols).map(move |row| {
                                row.iter()
                                    .zip(token)
                                    .map(|(&w, &x)| f64::from(w) * f64::from(x))
                                    .sum()
                            })
                        })
                        .collect();
                    let mut actual = vec![f32::NAN; rows * tokens];
                    if tokens == 1 {
                        gemv_f32(cols, &packed, &input, &mut actual)?;
                    } else {
                        gemm_f32(cols, tokens, &packed, &input, &mut actual)?;
                    }
                    check(&actual, &expected, "warp-boundary");
                }
            }
        }
    }
    Ok(())
}

#[test]
fn multi_block_quantized_batches_match_cpu_oracle() -> Result<(), QuantError> {
    for format in formats() {
        for blocks in [2, 3] {
            let expanded_format = Format {
                cols: format.cols * blocks,
                ..format
            };
            for rows in [1, 9] {
                // Blocks have different bytes, detecting incorrect packed-row
                // strides and accidental reuse of the first block in a row.
                let mut packed = Vec::new();
                for row in 0..rows {
                    for block in 0..blocks {
                        packed.extend((format.pack_row)(row + block * 5));
                    }
                }
                for tokens in [1, 2, 7, 8, 9, 31, 32, 33] {
                    let input: Vec<f32> = (0..tokens)
                        .flat_map(|token| values(token * 3 + 1, expanded_format.cols))
                        .collect();
                    let expected = oracle(&expanded_format, rows, &packed, &input)?;
                    let mut actual = vec![f32::NAN; rows * tokens];
                    (format.gemm)(expanded_format.cols, tokens, &packed, &input, &mut actual)?;
                    check(&actual, &expected, format.name);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn invalid_matrix_operations_preserve_output_and_empty_rows_are_noop() -> Result<(), QuantError> {
    for format in formats() {
        let packed = (format.pack_row)(0);
        let input = values(1, format.cols);
        let mut output = vec![123.5];
        assert!((format.gemv)(format.cols, &[], &input, &mut output).is_err());
        assert_eq!(output, [123.5], "invalid weights changed output");
        assert!((format.gemv)(format.cols, &packed, &[], &mut output).is_err());
        assert_eq!(output, [123.5], "invalid input changed output");
        assert!((format.gemm)(format.cols, 0, &packed, &input, &mut output).is_err());
        assert_eq!(output, [123.5], "zero tokens changed output");
        let doubled: Vec<f32> = input.iter().copied().chain(input.iter().copied()).collect();
        assert!((format.gemm)(format.cols, 2, &packed, &doubled, &mut output).is_err());
        assert_eq!(output, [123.5], "unaligned output changed contents");
        (format.gemv)(format.cols, &[], &input, &mut [])?;
        (format.gemm)(format.cols, 2, &[], &doubled, &mut [])?;
    }
    let mut output = [123.5];
    assert!(gemm_f32(usize::MAX, 2, &[], &[], &mut output).is_err());
    assert_eq!(output, [123.5], "overflow validation changed output");
    Ok(())
}
