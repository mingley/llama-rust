//! Print a GGUF's metadata and a per-dtype inventory of its tensors.
//!
//! Demonstrates reading a checkpoint without loading it as a model: the
//! key-value header, then every tensor grouped by ggml dtype with counts,
//! weight totals, byte totals, and effective bit width.
//!
//! This is not just a pretty-printer. An inventory like this is how you find
//! out that an engine is missing something: a dtype that appears in real
//! checkpoints but nowhere in the loader is invisible until you count. That is
//! exactly how the missing Q8_0 weight path in this crate was spotted -- Q8_0
//! had kernels and a writer, and real files use it for `token_embd` and
//! attention projections, but nothing wired it in as a loadable 2-D weight
//! dtype. Comparing "dtypes present in the file" against "dtypes the loader
//! accepts" makes that kind of gap obvious.
//!
//! The same comparison still finds one: `GgmlType::Q4_0` parses out of a header
//! and has a quantized-activation `gemv_q4_0`, but there is no
//! `dequant_q4_0_row`, no `gemv_q4_0_f32`, and no arm for it in the decode
//! dispatch -- so a legacy `*.Q4_0.gguf` loads its metadata here and then fails
//! with `Error::Type { ty: 2 }`. Run this example against one and the
//! inventory will show you a dtype the engine cannot actually multiply.
//!
//! Runs with no download, against an in-memory fixture:
//!
//!     cargo run --release --example gguf_inventory
//!
//! Or against a real checkpoint:
//!
//!     cargo run --release --example gguf_inventory -- model.gguf

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;

use llama_rust::gguf::{load_gguf_owned, GgmlType, Gguf, Kv};

/// Longest metadata array printed in full. Vocabularies are far longer than
/// anything worth putting on a terminal.
const ARRAY_PREVIEW: usize = 6;

#[derive(Default)]
struct DtypeStats {
    tensors: usize,
    weights: u64,
    bytes: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (label, gguf) = match std::env::args().nth(1) {
        Some(path) => {
            let mut file = File::open(&path)?;
            let mut bytes = Vec::new();
            let _len = file.read_to_end(&mut bytes)?;
            (path, load_gguf_owned(bytes)?)
        }
        None => (
            "tiny_llama fixture (in memory, no download)".to_string(),
            load_gguf_owned(llama_rust::fixtures::tiny_llama_gguf())?,
        ),
    };

    println!("== {label} ==");
    println!("file bytes     {}", gguf.blob_len());
    println!("data alignment {}", gguf.alignment());
    println!("tensors        {}", gguf.tensors().count());

    print_metadata(&gguf);
    print_dtype_inventory(&gguf);
    print_largest_tensors(&gguf);
    Ok(())
}

fn print_metadata(gguf: &Gguf) {
    println!("\n-- metadata --");
    // Read the keys that matter for identifying a checkpoint. There is no
    // iterator over every key, and that is usually a mercy: a vocabulary array
    // has tens of thousands of entries.
    const KEYS: [&str; 12] = [
        "general.architecture",
        "general.name",
        "general.file_type",
        "general.quantization_version",
        "tokenizer.ggml.model",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.merges",
        "tokenizer.chat_template",
        "general.alignment",
    ];
    for key in KEYS {
        if let Some(value) = gguf.kv(key) {
            println!("{key:38} {}", describe(value));
        }
    }

    // Per-architecture hyperparameters live under the architecture's own
    // prefix, so they have to be looked up once the architecture is known.
    if let Some(arch) = gguf.kv_string("general.architecture") {
        println!("\n-- {arch}.* hyperparameters --");
        for suffix in [
            "block_count",
            "context_length",
            "embedding_length",
            "feed_forward_length",
            "attention.head_count",
            "attention.head_count_kv",
            "attention.layer_norm_rms_epsilon",
            "rope.freq_base",
            "rope.dimension_count",
            "expert_count",
            "expert_used_count",
        ] {
            let key = format!("{arch}.{suffix}");
            if let Some(value) = gguf.kv(&key) {
                println!("{key:38} {}", describe(value));
            }
        }
    }
}

fn print_dtype_inventory(gguf: &Gguf) {
    let mut by_dtype: BTreeMap<i32, (GgmlType, DtypeStats)> = BTreeMap::new();
    for tensor in gguf.tensors() {
        let entry = by_dtype
            .entry(tensor.ty.to_i32())
            .or_insert_with(|| (tensor.ty, DtypeStats::default()));
        entry.1.tensors = entry.1.tensors.saturating_add(1);
        entry.1.weights = entry.1.weights.saturating_add(elements(tensor.shape));
        entry.1.bytes = entry.1.bytes.saturating_add(tensor.data.len());
    }

    println!("\n-- dtype inventory --");
    println!(
        "{:<10} {:>4}  {:>6}  {:>10}  {:>14}  {:>9}  {:>7}",
        "dtype", "id", "count", "weights", "bytes", "bits/wt", "% bytes"
    );
    let total_bytes = by_dtype.values().map(|(_, s)| s.bytes).sum::<usize>();
    for (id, (ty, stats)) in &by_dtype {
        println!(
            "{:<10} {id:>4}  {:>6}  {:>10}  {:>14}  {:>9.3}  {:>6.1}%",
            format!("{ty:?}"),
            stats.tensors,
            stats.weights,
            stats.bytes,
            bits_per_weight(stats.bytes, stats.weights),
            percent(stats.bytes, total_bytes),
        );
    }
    println!(
        "\n{} distinct dtypes; block layout is {} weights per block for the widest",
        by_dtype.len(),
        by_dtype
            .values()
            .map(|(ty, _)| ty.block_size())
            .max()
            .unwrap_or(0),
    );
}

fn print_largest_tensors(gguf: &Gguf) {
    let mut tensors: Vec<_> = gguf
        .tensors()
        .map(|t| (t.data.len(), t.name.to_string(), t.ty, t.shape.to_vec()))
        .collect();
    tensors.sort_by_key(|t| std::cmp::Reverse(t.0));
    println!("\n-- largest tensors --");
    for (bytes, name, ty, shape) in tensors.iter().take(8) {
        println!("{bytes:>14}  {:<8} {shape:?}  {name}", format!("{ty:?}"));
    }
}

fn describe(value: &Kv) -> String {
    match value {
        Kv::String(s) => {
            if s.len() > 60 {
                let head: String = s.chars().take(57).collect();
                format!("{head:?}... ({} chars)", s.len())
            } else {
                format!("{s:?}")
            }
        }
        Kv::Array { elem, items } => {
            let preview: Vec<String> = items.iter().take(ARRAY_PREVIEW).map(describe).collect();
            let ellipsis = if items.len() > ARRAY_PREVIEW {
                ", ..."
            } else {
                ""
            };
            format!(
                "array<{elem}> len {} [{}{ellipsis}]",
                items.len(),
                preview.join(", "),
            )
        }
        other => format!("{other:?}"),
    }
}

fn elements(shape: &[u64]) -> u64 {
    shape
        .iter()
        .try_fold(1u64, |acc, d| acc.checked_mul(*d))
        .unwrap_or(u64::MAX)
}

fn bits_per_weight(bytes: usize, weights: u64) -> f64 {
    let b = u32::try_from(bytes).map(f64::from).unwrap_or(f64::MAX);
    let w = u32::try_from(weights).map(f64::from).unwrap_or(f64::MAX);
    if w > 0.0 {
        b * 8.0 / w
    } else {
        0.0
    }
}

fn percent(part: usize, whole: usize) -> f64 {
    let p = u32::try_from(part).map(f64::from).unwrap_or(f64::MAX);
    let w = u32::try_from(whole).map(f64::from).unwrap_or(f64::MAX);
    if w > 0.0 {
        p * 100.0 / w
    } else {
        0.0
    }
}
