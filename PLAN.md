# Plan: cheap mid-tier serving, `llama-rust` as the lab

Derived from the ChatGPT share
[https://chatgpt.com/share/6a920fe1-48ac-83ea-9404-7f4c1062c17e](https://chatgpt.com/share/6a920fe1-48ac-83ea-9404-7f4c1062c17e).

The share **was reachable**. Visible five-turn HTML:
[`docs/chatgpt-share-6a920fe1.md`](docs/chatgpt-share-6a920fe1.md).
Complete share-API extract (searches, thought summaries, sources, raw JSON):
[`docs/chatgpt-share-6a920fe1/`](docs/chatgpt-share-6a920fe1/).
Plugin result bodies are redacted in the share; custom instructions were not included.

Work lands on `main`. No open PRs.

---

## Thesis

Open-weight models are getting good enough that the economic advantage
shifts from “who owns the weights?” to “who can turn those weights into the
cheapest reliable token stream?” GLM-5.3-Flash is the motivating example:
~320B stored parameters, ~18B active per token, hybrid sparse/linear
attention, claimed ~10× cheaper serving than GLM-5.2.

The interesting question is not “how do I execute a 320B model?” It is:

> **How little of this 320B model do I actually need to touch for each token?**

That is a systems problem: expert residency, state movement, scheduling,
quantization, prefix reuse, speculation. Rust is the language because the
hard part is ownership of pinned buffers, DMA, cache leases, cancellation,
and illegal GPU states — not another tensor library.

`llama-rust` is the **correctness laboratory**. It answers “what does this
model mean, and is the math right?” The new layers answer “where are the
weights, and how do we move them?” Do not turn this repo into a 500k-line
vLLM competitor.

---

## Split of concerns

```
llama-rust answers:     what does this model mean, and is the math right?
expertvm answers:      where are the weights needed for that math right now?
gpu-sim-rs answers:    what would that cost on a GPU we do not have?
```

Current `llama-rust` architecture (literal):

```
GGUF file → owned Vec<u8> → tensor metadata + byte ranges
         → model walk (attn / MoE routing / FFN)
         → quantized GEMV/GEMM → CPU
```

Weights are already “one blob + ranges,” not per-matrix clones. The
conceptual upgrade is:

```
today:   &blob[tensor.range]
future:  store.acquire(tensor.id(), target_device) → ExpertLease
```

That one seam opens residency, migration, prefetch, replication, and
later GPU/RDMA. The public API of that seam stays **safe Rust**. Unsafe
hardware interaction is confined below an audited boundary. Zero-unsafe
everywhere would handicap `expertvm`; zero-unsafe-by-default remains the
rule for the reference engine (`--no-default-features` still
`forbid(unsafe_code)`).

---

## What not to build

Do not spend the next slice of work on:

- a generic tensor / autograd library (Burn/CubeCL occupy that layer)
- a tokenizer product, Safetensors parser, or generic CUDA wrapper
- an OpenAI-compatible HTTP server as the *wedge*
- a Llama-only engine or llama.cpp clone
- a generic async runtime
- competing with DeepEP on all-to-all
- competing with FlashInfer on fused attention/MoE kernels
- competing with Mooncake on distributed KV head-on
- `model-ir` / “LLVM for LLM inference” (huge, later)
- matching llama.cpp tok/s as the headline

Use existing Rust where it is already the right tool (Tokio, `cudarc`,
CubeCL, safetensors, tokenizers, serde, axum/hyper) *when a later crate
needs them*. The reference engine stays lockfile-only until a named
layer actually requires a dep.

---

## Target stack (later crates, not this week)

| Pri | Crate | Job | When |
| --- | --- | --- | --- |
| 1 | `expertvm` | engine-agnostic virtual memory + dynamic residency for sparse weights | **the wedge** |
| — | `gpu-sim-rs` | deterministic GPU-systems VM for agents without GPUs | **the CI environment**; first killer app is expertvm |
| 2 | `kv-fabric` | distributed KV/state memory fabric | after expertvm has a result; do not beat Mooncake first |
| 3 | `moe-runtime` | sparse MoE execution (grouped GEMM, router+dispatch) | *not* the first repo; expertvm is narrower |
| 4 | `infer-scheduler` | continuous batching + SLO scheduling, engine-independent | hard to get vLLM/SGLang to outsource; **trace-level slice** is `expertvm schedule` |
| 5 | `quant-rs` | hardware-aware mixed quantization under quality/$ constraints | later |
| 6 | `attention-rs` | hybrid attention vocabulary (MHA/GQA/MLA/DSA/KDA/GDN, prefill ≠ decode) | later |
| 7 | `speculate-rs` | reusable speculative decoding | later |
| 8 | `tensor-loader` | fast load/reshard/lazy expert load | later |
| 9 | `infer-bench` | serving benchmarks people will actually run | quietly from day one of expertvm |
| later | `model-ir` | inference IR/compiler | postpone |

Eventual shape (crates stay independently useful; the commercial system
combines them):

```
                    API
                     │
               infer-router
                     │
              infer-scheduler
                /         \
          PREFILL         DECODE
             │               │
             └──── kv-fabric ┘
                     │
        attention-rs  moe-runtime  speculate-rs
                     │
                  quant-rs
                     │
                  model-ir
                     │
            CubeCL / cudarc
```

`llama-rust` sits beside this as **model semantics + reference
correctness**, not as the production server.

---

## The wedge: `expertvm`

Not “another MoE runtime.” Not “another KV cache.”

> Make a 300B–1T sparse MoE behave as though its experts live in a giant
> virtual memory space, while automatically keeping the right experts in
> GPU HBM.

The inference system has two different resources:

```
compute requirement  ≈ active parameters
memory requirement   ≈ total parameters
```

Sparse MoE reduced the first without proportionally reducing the second.
That gap *is* $/token.

DeepEP moves **tokens to resident experts**. The inverse problem is
unsettled:

```
should the expert move?
where should it live?
how many copies?
when?
can we know before it is needed?
is moving tokens cheaper than moving weights?
```

Signals that the primitive is forming but unowned: DeepEP’s elastic
GPU/CPU virtual address space (listed as ongoing); SGLang DWDP prefetch
of peer expert weights over NVLink (early, reported up to 1.92× in one
config). Basic RAM↔GPU expert LRU is already becoming commodity
(llama.cpp oversized-MoE residency, router-based prefetch, `moe-l2`).
Do **not** spend six months independently building “LRU experts between
RAM and GPU.” Use this repo to find the **next** abstraction:

- predictive residency from router behavior
- dynamic choice: move weights vs dispatch activations, based on fan-in,
  expert size, topology, predicted reuse

Eyebrow-raising demo (advertise the economic result, not “Rust is
memory safe”):

- run a 320B MoE efficiently with 160 GB HBM, or
- same GLM-5.x throughput on 4 GPUs instead of 8, or
- serve a 1T MoE with ~30% of expert weights resident in HBM

Exact numbers need experiment. The form of the claim is `$/M tokens` at
fixed latency and quality.

### `expertvm` versions

**V0** (CPU / simulated GPU, this repo as host):

- single node, 8 GPUs in the *model* (not required physically)
- HBM ↔ HBM and CPU pinned DRAM ↔ HBM as *state machine*
- expert registry, leases, async migration, LRU/LFU, hot replication,
  layer-ahead prefetch, metrics
- one real model family already in-tree (Qwen3MoE / Qwen2MoE / llama MoE)
- benchmark static EP vs expertvm under intentionally restricted HBM
  (`compare_ep`, `HardwareProfile::restrict_hbm`, `expertvm ep`)

**V1:** traffic-aware placement, co-activation placement, adaptive
replication (`colocated`, `with_hot_replicas`, `expertvm place`).

**V2:** RDMA, multi-node, remote expert residency (`sim_remote_home` +
`plan_placement` on the home↔compute hop, `schedule_remote` /
`expertvm schedule --place remote --prefetch copy-forward`,
`SimulatedGpuStore::migrate`,
`expertvm remote`).

**V3:** predictive residency:

```
P(expert_j at layer L+1 |
   experts at L, L-1, …, prompt class, session)
```

Online [`Markov`] table (no future leak, no invented prompt class) is
`P(to|from)` plus lookback-2 `P(to|from, from_prev)` with order-1 backoff.
`Prefetch::{None, CopyForward, Markov, Both}` / `copy-forward` in
`sim_replay_cfg`. `Both` is copy-forward ∪ lookback-2 Markov (decode's store
path). Prefetch before the next router event. `SimCfg::seq_streams` maps
`sequence % n_streams` onto CUDA streams so a batched token can overlap
H2D/GEMM (`expertvm sim --seq-streams`). Token-boundary TTFT samples when
`token` changes, not when `sequence` changes.

Design so vLLM / SGLang / mistral.rs can consume the crate. Nobody should
have to adopt this inference server.

---

## Why `llama-rust` is the right lab

Already present:

- GGUF v3, mixed dtypes, on-disk bytes, no per-matrix clone
- real MoE walks: llama MoE, Qwen2MoE, Qwen3MoE, Llama4, Qwen3Next
- NEOX vs NORM RoPE per official `llama_model_rope_type`
- IEEE binary16 including subnormals (ggml `ggml_fp16_to_fp32`)
- Q4_0 / Q8_0 as loadable 2-D weights
- SIMD row kernels behind `simd` (AVX2+FMA, NEON); scalar fallback
- persistent GEMV pool + Scratch for dense decode
- llama.cpp greedy differential on a real Qwen2.5-0.5B Q4_K_M file

Deliberately *not* yet: production serving, mmap, CUDA. Engine continuous
batching is on `main`. That is correct. The next high-signal layer is
**not** OpenAI HTTP, Tokio, or a tok/s race.

`llama-rust` stays the place that:

1. executes the model correctly (oracle + llama.cpp greedy identity)
2. emits **real** MoE access traces
3. hosts the first `ExpertStore` seam
4. feeds `gpu-sim-rs`

Unsafe/GPU stays out of the default engine. A later `expertvm` / CUDA
backend crate may use `unsafe` under a tiny audited boundary.

---

## Phase 0 — keep the engine world-class (this repo, `main`)

The reference engine has to be something researchers will actually load.

- [x] Official NEOX RoPE per architecture (not NORM pairing on Qwen/Phi)
- [x] binary16 subnormals (was 2× too small; corrupts Q4_K/Q6_K scales)
- [x] Q8_0 and Q4_0 2-D weights
- [x] Real-model llama.cpp differential (fail-loud if env is set)
- [x] SIMD fast path + persistent pool landed on `main`
- [x] Chat templates + special-token split + Unicode BPE pre-tokenizer
- [x] Layered public API (`Model`/`Session`), examples, crates.io metadata
- [x] README/STATUS rewritten for “verifiable reference,” not “no tok/s curiosity”
- [x] More than one real-model fixture (NEOX Qwen + NORM Llama control)
- [x] Oracle-owned f16 conversion (oracle must not call production `fp16`)
- [x] Q4_0 SIMD (AVX2+FMA+F16C / NEON row kernels on GEMV and GEMM)
- [x] Expert FFN on Scratch (llama / qwen2moe / qwen3moe / qwen3next / Llama4)

Push each of these to `main` as they land. Do not open PRs.

---

## Phase 1 — real MoE access traces (no GPU)

Change nothing about *where* experts live yet. Instrument the routers
already in `decode.rs`.

For every generated token record:

```
ExpertAccess {
    sequence, token, layer,
    experts: Vec<ExpertId>,   // selected
    weight_pt: Vec<u32>,      // optional router mass ‰; empty / omit `w` if unknown
    prefix: Option<u64>,      // JSONL `"p"`; omit when None. Hash of token ids, not a class
}
```

Replay and report (measured by `expertvm analyze`, integer ‰):

- `P(E_t,l | E_{t-1},l)` → `seq_persist‰`
- `P(E_t,l+1 | E_t,l)` → `layer_persist‰`
- `P(E_t | E_{t-1}, E_{t-2})` causal lookback-2 → `order2_persist‰`
- popularity / working set → `top20‰`, `ws90`
- reuse distance → `reuse8‰`
- `P(E_a | E_b)` mass as pair count → `coact_pairs`

Hit rates still come from `expertvm replay`, not from these locality
numbers. Prompt/domain class is not in the JSONL (no fake labels).
Optional `"p"` is a content-addressed hash of the token ids in the
prefix (`prefix_hash`), not a prompt class. `KvCache` emits `"p"` when
tracing; `--prefix-cache` on `expertvm schedule` skips GPU work for a
token whose hash already completed on another sequence. That is a
**trace-level** skip of expert GEMMs, not KV. The reference engine also
does Automatic Prefix Caching on real KV: `KvCache::reuse_prefix` plus
`Llama::prompt` (Session / `serve` / `chat`). `prefill` / `forward` still
append so decode identity is unchanged. A full-prefix hit recomputes the
last prompt token for logits. Opt-in paged KV (`Llama::new_paged_cache`,
`Model::session_paged`, `--kv-page`) stores K/V in interned blocks of
`N` tokens; logits bit-match dense. Distinct from `expertvm kv` (simulated
VMM pages, not decode blocks). Prompt/domain class is not in the JSONL.

The research question, before any CUDA:

> Are expert activation patterns predictable/local enough for dynamic
> residency to work?

Measured tables (regenerate; do not hand-edit the numbers):

Synthetic stress case `tests/traces/cycling.jsonl` (24 acquires, 3 keys,
`--capacity 2`). This is where policies **differ**:

```
policy        hits  misses  evicts  hits‰
random           7      17      15    291
lru              0      24      22      0
lfu              0      24      22      0
layer-ahead      3      21      19    125
predictor        7      17      15    291
oracle          11      13      11    458
```

Writer-built tiny Qwen3MoE `tests/traces/tiny-qwen3moe.jsonl` (`ab`, 8
predicted tokens, 1 layer, 4 experts, top-2). Working set is **2 experts**,
so `--capacity 2` is ~888‰ for every policy (2 compulsory misses) and
`--capacity 1` is 0‰ for every policy including oracle. That is a toy
router, not a 320B result. Checked-in `tests/traces/tiny-qwen3moe-2layer.jsonl`
is the writer-built two-layer tiny (`ab` + 8 tokens) so `layer_persist‰` and
same-layer `seq_persist‰` are measured. A real Qwen3MoE GGUF is required
before the kill-switch (“best non-oracle ≈ random ≈ 18% → stop”).

If the best realistic policy on a **real** trace is ~18% hit, stop. If a
1-layer predictor is ~80% with oracle ~94%, there is a paper and a crate.

Use writer-built tinies first, then a real Qwen3MoE / Qwen2MoE GGUF
when one is on disk. Traces are the product of this phase; check them in
under `tests/traces/` (small) or document how to regenerate (large).

---

## Phase 2 — `ExpertStore` seam inside `llama-rust`

Replace the expert-weight assumption “already locally addressable.”

```
trait ExpertStore {
    fn acquire(&mut self, expert: ExpertId) -> Result<ExpertLease>;
}
```

Backends:

| Store | Meaning |
| --- | --- |
| `DirectStore` | today’s blob range; bit-identical to current decode |
| `CachedStore` | bounded “fast memory” of N experts; rest fault in |
| `TieredStore` | fast RAM / slow RAM / disk |
| `SimulatedGpuStore` | fake HBM capacity, PCIe/NVLink bandwidth, DMA concurrency; `with_managed` is UM prefetch, `with_mapped` is zero-copy host (and `host_pin_bytes` occupancy), `with_vmm` is `va_acquire`; `with_cfg` is `host_func` / blocking streams / `sync_alloc` / mempool / `vmm_page` / pageable H2D / `SetAccessedBy` / legacy NULL / stream priority / `graph_update` / `graph_clone` / `timing_events`; `expertvm store` / `store_replay_cfg` is the CLI (Markov prefetch) |

`DirectStore` must keep every existing oracle / real-model test green.
The dense/common weights stay resident. Only expert tensors go through
the store. Decode wiring is on `main`: `KvCache::attach_expert_store`
takes a `LiveStore::{Direct, Cached, Tiered, Simulated}`. Default `None` keeps
the blob GEMV path (allocation-free tests unchanged). Identity tests
cover Qwen3MoE (Direct + Cached + SimulatedGpuStore + TieredStore), llama MoE,
Qwen2MoE, Llama4, and Qwen3Next. `WeightStorage::{InMemory, File, Synthetic}`
page experts into a bounded fast tier. mmap stays parked
(`WeightStorage::mmap` returns an error); File seek+read is the disk path.

Artificially constrain the cache (e.g. 64 experts total, 8 resident)
and execute a real trace.

Storage abstraction underneath (needed because today’s `Vec<u8>` of
the whole file cannot demonstrate constrained residency):

```
WeightStorage: InMemory | Mmap | Synthetic → TensorView
```

Mmap is currently a parked non-goal for the *default* engine. The
store/mmap path is opt-in, documented, and not the `--no-default-features`
path. llama.cpp already showed a 48.4 GB Qwen3-Next-80B GGUF on a 16 GB
M1 via bounded expert pages, and 31–52% decode from prefetching routed
expert ranges rather than scattered page faults. A laptop is a
legitimate laboratory.

---

## Phase 3 — `gpu-sim-rs`: mechanical GPU VM for agents without GPUs

Not a fake “GPU backend.” Not transistor-accurate (Accel-Sim/GPGPU-Sim
already do that, and it is too expensive).

**Semantics exact. Timing parameterized and calibratable.**

Exact (mechanical invariants agents may rely on):

- memory capacity, alloc/free, lifetimes
- host-synchronous `cudaMalloc` / `cudaFree` / `cudaMemcpy` / `cudaMemset`
  (`Sim::malloc` / `free_sync` / `memcpy_sync` / `memset_sync` /
  `memset_op_sync`) vs stream-ordered
  `cudaMallocAsync` / `cudaFreeAsync` / `cudaMemcpyAsync` / `cudaMemsetAsync`
  (`alloc` / `free` / `memcpy` / `memset`); `malloc` OOM is at the call;
  `cudaDeviceSynchronize` (`synchronize_device`) waits one GPU
- memory pools: `create_pool` / `create_pool_with_props` /
  `create_shareable_pool` / `alloc_from_pool` /
  `set_pool_release_threshold` / `pool_trim_to` / `set_device_mempool` /
  `pool_get_attribute` / `pool_set_attribute` / `pool_get_access` /
  `destroy_pool`
  (`cudaMemPoolCreate` / `cudaMemPoolCreate`+`MemPoolProps` /
  `cudaMallocFromPoolAsync` /
  `cudaMemPoolAttrReleaseThreshold` / `cudaMemPoolTrimTo` / `cudaDeviceSetMemPool` /
  `cudaMemPoolGetAttribute` / `SetAttribute` / `GetAccess` / `cudaMemPoolDestroy`); default
  threshold `0` returns unused bytes on free; `u64::MAX` holds them so
  `malloc` can OOM until trim; `cudaMalloc` cannot consume pool cache;
  `MemPoolProps::max_size` caps reserved (`live + cached`) or is unlimited (`0`);
  GetAttribute Used/Reserved wrap live+cached (no invented pool high-water);
  GetAccess is ReadWrite on the owner and after SetAccess on peers;
  Destroy returns unused cache to the OS, keeps outstanding allocs, cannot
  destroy the default pool, and rebinds the current pool to the default
- `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` / `cudaIpcCloseMemHandle`
  (`ipc_get` / `ipc_open` / `ipc_close`): import aliases source physicals
  (no extra HBM); free of the source while imports are live is Invalid;
  `ipc_get` of a mempool alloc is Invalid; capture is refused
- `cudaMemPoolExportToShareableHandle` / `ImportFromShareableHandle` /
  `ExportPointer` / `ImportPointer` (`pool_export` / `pool_import` /
  `pool_export_ptr` / `pool_import_ptr`): imported pool shares live/cached;
  pointer import aliases physicals; default/`create_pool` cannot export;
  capture is refused
- `cudaHostRegister` / mapped host (`alloc_host`, `host_register`,
  `host_register_mapped`, `alloc_host_mapped`): pin existing pageable
  memory; mapped pointers are kernel-readable over PCIe with no H2D and
  no HBM charge; `host_pin_bytes` is the `mlock` cap (`PinOom`)
- `cudaMallocManaged` / `cudaMemPrefetchAsync` (`alloc_managed`,
  `prefetch`, `prefetch_host`, `prefetch_with_flags`): no HBM until
  migrate; prefetch **moves** (does not replicate) unless
  `cudaMemAdviseSetReadMostly` ([`Sim::mem_advise`]
  [`MemAdvise::SetReadMostly`]); `prefetch_with_flags` requires flags 0
  (`PrefetchFlags::DEFAULT`) and a [`Place`] dest; typed helpers stay;
  a kernel first-touch prefetches when that kernel *starts* (after stream
  deps) unless
  [`MemAdvise::SetAccessedBy`] maps that
  GPU or [`MemAdvise::SetPreferredLocation`] already holds the page at
  another GPU (remote read, interconnect billing; writes still migrate;
  host preferred does not skip first-touch)
- `cudaStreamAttachMemAsync` (`stream_attach`, `stream_attach_with_flags`,
  `alloc_managed_host`): Global / Host / Single visibility; Host and
  other-stream Single fail device kernels / memset / device prefetch
  (`not attached`); Single cannot use the NULL stream; capture is refused;
  `stream_attach_with_flags` maps `MemAttachFlags::{GLOBAL,HOST,SINGLE}`
  then typed `stream_attach` (other bits Invalid `"stream attach flags"`)
- CUDA VMM (`va_reserve` / `va_map` / `va_unmap` / `va_free`,
  `va_create` / `va_map_handle` / `va_retain_handle` / `va_release_handle`,
  `va_acquire` / `va_release`, `va_map_range` / `va_unmap_range`):
  `cuMemAddressReserve` / `cuMemMap` / `cuMemUnmap` / `cuMemAddressFree`;
  `cuMemCreate` / `cuMemRetainAllocationHandle` / `cuMemRelease`;
  HBM charged while a handle has refs or maps; combined `va_map` still
  refunds on unmap until retain promotes the span;
  the pointer survives unmap;
  `va_release` parks the VA so `va_acquire` remaps without another reserve;
  sparse sub-range maps (vLLM KV-block analog) charge only the mapped span;
  two VAs may `va_map_handle` the same physical;
  `va_set_access` is PROT_READ; `va_set_access_write` is PROT_READWRITE
  (peer writes without dest HBM);
  [`Sim::kernel`] needs the whole VA covered; [`Sim::kernel_bufs`] /
  [`Sim::memset_buf`] / [`MemcpyOp::offset`] touch a mapped span so a paged
  KV working set need not cover the pointer; `va_acquire_paged` maps a VA in
  page-sized physicals (each pays map overhead)
- `cudaLaunchHostFunc` (`host_func`): stream-ordered host work; does not
  occupy compute or copy engines; graphs may record it
- `cudaStreamCreate` vs `cudaStreamNonBlocking` (`set_stream_blocking`):
  blocking streams serialize with the default/null stream; created
  streams default to non-blocking (vLLM-style). The legacy default
  stream (`set_legacy_null_stream`) still serializes with every stream
- `cudaEventCreateWithFlags(..., cudaEventDisableTiming)`
  (`create_event_disable_timing`): wait/query work; `event_elapsed_ns` fails
- `cudaEventDestroy` (`destroy_event`): waits a recorded incomplete event;
  never-recorded returns immediately; capture refused; the id may be created
  again
- `cudaEventRecordWithFlags(..., cudaEventRecordExternal)` /
  `cudaStreamWaitEvent(..., cudaEventWaitExternal)`
  (`record_event_external` / `wait_event_external`): captured without
  forked-capture join; graph WaitExternal waits for a live record, not the
  same graph's record of that event
- stream ordering, events, barriers
- kernel enqueue, async copies
- copy-engine availability, Hyper-Q `compute_slots` occupancy (default
  exclusive; `>=2` concurrent kernels at full issue rate), green-context
  `set_stream_sm_permille` (compute-bound kernels scale; memory-bound keep
  full HBM; default unset is a full chip),
  peer accessibility
- HBM vs host-pinned residency (`Place::{Host, HostPinned, Device}`,
  `alloc_host_pinned`, `memcpy_pinned_to_device`; pageable
  `cudaMemcpyAsync` is host-synchronous — `memcpy_host_to_device` /
  `memcpy_device_to_host` wait the stream; timing still uses
  `pageable_permille`)
- P2P / NVLink / PCIe topology
- concurrent transfer limits
- OOM, data dependencies, collective / CUDA-graph dependencies

Not exact (belongs in a profile, not the simulator core):

- “this grouped FP8 GEMM takes 42.7 μs”

Architecture:

```
llama-rust / expertvm
        → GPU operation API
        → CUDA-like semantics
        → resource simulator
        → topology simulator
        → timing model (roofline + empirical curves)
        → discrete-event engine
        → virtual clock
```

Primitive: `gpu_sim::GpuOp` (Kernel / Memcpy / Collective=`AllReduce` /
Event record+wait / Alloc / Free / Memset) compiled into a
`gpu_sim::Operation` dependency DAG (`Sim::operations`).
Contention is real: three transfers to GPU0 cannot each get the full
PCIe x16. `Sim::synchronize_stream` is `cudaStreamSynchronize`: the
virtual clock advances until that stream is idle; other streams keep
running. `synchronize_event` waits only until the record completes.

Kernels are structural (`Matmul {m,n,k,dtypes}`, `GroupedMoeGemm {…}`),
costed by a hardware profile (roofline first; `gemm_util_permille` and
`grouped_moe_permille` are the empirical-curve knobs; capture still
refused in this crate).

**Virtual time only.** No `sleep(40us)`. Discrete events. Deterministic:

```
commit A + hardware H + workload W + seed S  →  performance P
```

Profiles live in `profiles/` (`h100-sxm.profile`, `h200-sxm.profile`,
`8xh100-nvlink.profile`, `cheap-48gb.profile`, …)
and are **empirical**, not hard-coded folklore:

```
gpu-profile capture --output h100.json
```

Someone with a GPU captures; agents without GPUs consume. This is the
most important architectural choice in the simulator.

### Invariants (encode in types where possible)

- lease_count > 0 ⇒ allocation cannot be freed
- kernel reads are resident on that device **or** mapped host
  (`alloc_host_mapped` / `host_register_mapped`)
- completed transfer ⇒ destination contains the object
- stream[i+1].start ≥ stream[i].finish
- hbm.used_bytes ≤ hbm.capacity
- an expert cannot be Evicted and in-use

Preferred state machine (encoded as `expertvm::ExpertPhase`, fatal on
illegal lease/evict):

```
Cold → Transferring → Resident → Leased → Resident → Evicting → Cold
```

CPU `CachedStore` / `TieredStore` fault-in is instantaneous (Resident
or Leased). `evict` of a leased key is fatal. `SimulatedGpuStore` is
Transferring until the copy-stream event completes; `evict` leaves the
key `Evicting` until the stream-ordered free completes. Lease of
Transferring/Cold/Evicting is refused. `Operation` timestamps make
`stream[i+1].start ≥ stream[i].finish` inspectable. `set_stream_priority`
is CUDA stream priority (higher starts first when compute contends).
`sim_replay` `--max-batch N` is a trace-level admission cap (N sequences
per engine iteration at a token; `0` admits the whole token).
`expertvm schedule` is open-loop continuous batching: sequences arrive at
`sequence * interarrival_ns`, FCFS into a running set of size `max_batch`,
one next **chunk** (layer-major) per engine step, finished sequences leave,
`Sim::idle_until` jumps the virtual clock when the GPU would wait.
A chunk is the whole next token unless `--prefill-chunk N` limits a
sequence's first token to N layer-events so a short decode is not stuck
behind a long prefill. `--decode-first` holds leftover prefill while any
running sequence is already in decode. `--slo-reject` drops a waiter whose
queue wait already meets `--ttft-slo-ns`. TTFT is first-token end minus arrival. Optional
`--ttft-slo-ns` / `--itl-slo-ns` count misses. Cache order is demand paging (no JSONL future
leak). `--place striped|remote` keeps one walker per home GPU so `--capacity`
is that device's slots, not a cluster-wide LRU. Hot replicas occupy dest slots
and dest eviction frees replica HBM. `restrict_hbm` still evicts when fewer
pages fit than `--capacity`. `--prefix-cache` skips GPU work for a token
whose content-addressed `"p"` hash already completed on another sequence
(insert after the computing token finishes; a hit consumes the whole
remaining token, not one prefill chunk). This is not a 500k-line vLLM engine.

Agents may aggressively optimize policies. Illegal GPU states are
impossible or immediately fatal.

### Two scores

1. **Semantic (binary):** CUDA ordering, ownership, capacity, events,
   residency, sync. Fail ⇒ reject the change.
2. **Performance (continuous):** $/M tokens, ITL, TTFT, throughput,
   HBM needed, bytes moved, energy estimate.

   Shipped in `gpu-sim::Score`: `wall_ns`, `hbm_peak`,
   `bytes_moved`, `energy_uj` (`node_tdp_mw * wall_ns / 1e6` µJ), optional
   `ns_per_token`, `ttft_ns`, `itl_ns`, optional `usd_micros_per_m_tokens`
   (`rent_usd_micros_per_hour * wall_ns * 1e6 / (hour_ns * n_tokens)`).
   Example profiles leave rent at `0` so dollars stay omitted. Not a capture.
   `sim_replay` samples the virtual clock after each token.

Agent loop: modify expertvm → `cargo test` (semantics) → simulator
(score) → keep/revert. Thousands of experiments, no H100 required.

### Do not Goodhart the simulator

- adversarial workloads: uniform, hotset, shifting hotset, cache
  thrash, coding/chat/long-context traces, batch 1 vs 128
  (`batch-1` / `batch` / `batch-128`),
  prefill-heavy / decode-heavy / prefill-batch (4-seq mixed prefill+decode)
  / shared-prefix (identical token-0 prefix hash, diverging decode)
- topologies: 1 GPU, 2 GPU PCIe, 8 GPU NVLink, bad NUMA, 2-node RDMA,
  asymmetric links — named example profiles (`h100`, `2xh100-pcie`,
  `8xh100`, `bad-numa`, `2node-rdma`, `asymmetric`) plus `gpu-profile probe`
  / `expertvm topology` / `infer-bench topology`. Asymmetric is a 3-GPU
  NVLink *chain* (0–1–2); 0↔2 is `NoPeer`.
- faults: transfer delay (`Sim::set_extra_transfer_ns`), GPU unavailable
  (`SimError::Unavailable`), memory pressure (OOM), cancel
  (`Sim::cancel_stream` → `SimError::Cancelled`), expert load failure
  (`Sim::fail_next_memcpy` → `SimError::TransferFailed`).
- ring `allreduce` as a collective with a real mesh: missing wrap-around
  links fail `NoPeer`.
  - CUDA-graph capture: `begin_capture` / `end_capture` / `instantiate_graph`
  / `update_graph` / `clone_graph` / `destroy_graph` / `launch_graph`.
  `create_graph` / `graph_add_kernel` / `graph_add_memcpy` / `graph_add_memset`
  / `graph_add_host_func` / `graph_add_event_record` / `graph_add_event_wait`
  / `graph_add_child` are `cudaGraphCreate` / `cudaGraphAdd*` (empty graph,
  then nodes; illegal on an instantiated exec and during capture; legal on
  the definition after instantiate).
  `graph_add_dependencies` is `cudaGraphAddDependencies` (independent nodes
  may Hyper-Q overlap at launch; capture records same-stream edges).
  Recorded kernels and copies do not run until launch; `cudaMallocAsync` /
  `cudaFreeAsync` (`alloc` / `free`) **can** be captured as graph mem nodes.
  Host-sync `malloc` / `free_sync` / `memcpy_sync` / `synchronize_device` /
  VMM / mempool create/trim/set-attribute cannot be captured;
  instantiate/update/upload/clone/destroy cannot be captured; capture requires an idle stream.
  Instantiate is host-sync (`graph_instantiate_ns`); the first launch
  instantiates if needed (default flags reuse graph mem allocs on relaunch).
  `instantiate_graph_auto_free` is `cudaGraphInstantiateFlagAutoFreeOnLaunch`
  (stream-ordered free before a later launch's alloc nodes; illegal with mem
  free nodes). `cudaGraphUpload` (`Sim::upload_graph`,
  `graph_upload_ns`) is a separate host-sync after instantiate; the first
  launch uploads if needed.   `update_graph` replaces an instantiated exec's
  steps when device, stream, op kinds, and dependency edges match (`graph_update_ns`).
  Graphs with mem alloc/free nodes cannot be updated.
  `graph_exec_kernel_set_params` is `cudaGraphExecKernelNodeSetParams`
  (`graph_set_params_ns`): patch one kernel node's pointers / kind; mem nodes
  are legal. Capture cannot include it.
  `graph_exec_memcpy_set_params` is `cudaGraphExecMemcpyNodeSetParams`
  (same cost; pageable still illegal; mem nodes legal).
  `graph_exec_memset_set_params` is `cudaGraphExecMemsetNodeSetParams`
  (same cost; zero-byte still illegal; mem nodes legal).
  `graph_node_set_enabled` is `cudaGraphNodeSetEnabled` (skip launch;
  mem nodes cannot be disabled).
  `clone_graph` is an independent uninstantiated copy (`graph_clone_ns`);
  child-graph nodes are cloned recursively (shared children cloned once);
  graph mem alloc nodes get new ids (independent HBM).
  `destroy_graph` is `cudaGraphDestroy` (later launch is unknown; remaining
  graph mem is refunded).
  Independent streams stay live during capture. A `wait_event` on an
  event recorded in this capture **joins** (CUDA forked capture) so
  copy and compute can overlap in one `launch_graph`. Launch remaps
  origin-stream nodes onto the launch stream; forked streams keep their
  ids. Query/sync of a capturing stream, and node `synchronize`, are
  Invalid. `launch_graph` during capture records a child-graph node
  (`GpuOp::ChildGraph`) when the child is already instantiated; parent
  launch expands the child. Independent streams still launch live.
  Graph launch pays `graph_launch_ns` once; recorded kernels skip
  per-launch overhead.
  `sim_replay` / `SimulatedGpuStore` capture repeated expert GEMMs
  (`expertvm sim --cuda-graphs`). Grouped launches capture a leaf graph
  per alloc, instantiate it, then a parent of `GpuOp::ChildGraph` nodes
  so combos reuse leaves after one expert is evicted. `--graph-update`
  parks a leaf exec on evict and `update_graph`s the next miss on that
  `(device, stream)` (`graph_update_ns` instead of instantiate; parent
  combos still destroy). `--graph-clone` clones a leaf capture before
  instantiate (`graph_clone_ns`; the src is destroyed).   `--graph-build`
  is `cudaGraphCreate` / `cudaGraphAdd*` instead of stream capture (no idle
  stream; implies `--cuda-graphs` on the walker; combo children have no
  `graph_add_dependencies` edge so they may Hyper-Q overlap). Capture after a miss waits with
  `synchronize_stream` so the compute stream is idle (CUDA). `--max-batch N`
  admits N sequences per engine iteration. `expertvm schedule` is the
  open-loop running set (arrivals, retire, SLO misses, `idle_until`,
  `--prefill-chunk N`, `--decode-first`, `--slo-reject`, `--prefix-cache`, `--place striped|replicas|remote`). `query_event` is `cudaEventQuery`.
  `query_stream` is `cudaStreamQuery`. `mem_info` is `cudaMemGetInfo`.
  `plan_window` Stay vs Fetch gates prefetch
  in the GPU loop (`--plan-window N`). `prefetch_hits` / `prefetch_waste`
  measure whether those fills were used. `--sync-alloc` is host-sync
  `cudaMalloc`/`cudaMemcpy`/`cudaFree` on miss (`Sim::malloc`); default
  `sim`/`schedule` / `SimulatedGpuStore::new` stay on `cudaMallocAsync`.
  `SimulatedGpuStore::with_cfg` opts into `--sync-alloc`, `--mempool`,
  `--shareable`,
  `--host-func`, blocking compute, `--pageable`, `--accessed-by`,
  `--legacy-null`, `--stream-priority`, `--graph-update`, `--graph-set-params`, `--graph-clone`, `--graph-build`, `--graph-piecewise`, `--graph-mem`, `--graph-auto-free`, `--timing-events`, `--cooperative`, `--pdl`, `--l2-persist`, `--cluster`, `--preferred-cluster`, `--cluster-spread`, `--max-shared`, `--non-portable-cluster`, `--sync-policy`, `--shared-mem`, and `--multicast`. `--mempool` sets the default
  pool release threshold to `u64::MAX` (vLLM-style hold); reuse of a
  cached page pays `pool_reuse_ns`. `--shareable` is POSIX-FD mempool IPC
  (implies `--mempool`; illegal with `--sync-alloc` / mapped / managed / vmm). `--mapped` is `cudaHostAllocMapped`
  (no H2D, PCIe kernels, HBM unused; walker slots also cap at
  `host_pin_bytes / expert_bytes`). `--managed` is `cudaMallocManaged`
  plus `cudaMemAdviseSetReadMostly` plus `SetPreferredLocation` plus
  `cudaMemPrefetchAsync` on miss (HBM charged on migrate; a second GPU
  prefetch keeps the copy; a remote read can keep the page on the preferred GPU).
  `--accessed-by` is `cudaMemAdviseSetAccessedBy` on every GPU at fill
  (dest GEMM reads without migrating; migrate/pin skip dest prefetch).
  `--place replicas` then `prefetch`s hot keys onto dest GPUs unless
  `--accessed-by`; dest eviction is `drop_managed_copy` (one GPU's copy, allocation stays). `--vmm` is
  `va_acquire` (remap idle VA or reserve+map) then H2D; evict `va_release`s
  the pointer. `--vmm-page N` is `va_acquire_paged` (KV-block physicals;
  implies `--vmm`). `expertvm kv` demand-pages per-sequence VAs (`cuMemCreate`
  + `cuMemMap`; `--sequences N` aliases interned physicals; `kernel_bufs`
  + H2D or `memset_buf` at a mapped span; peak HBM is unique pages). `--host-func` is `cudaLaunchHostFunc` after each event's
  GEMMs (`host_func_ns`; no GPU occupancy). `--blocking-streams` is
  `cudaStreamCreate` on seq-streams (serialize with NULL); default is
  `cudaStreamNonBlocking`. `--legacy-null` is `set_legacy_null_stream`
  (NULL serializes with every stream). `--pageable` is host-sync
  `memcpy_host_to_device` (`pageable_permille`). `--stream-priority` is
  `cudaStreamCreateWithPriority` on seq-streams (priority = stream id). `--graph-update`
  is `cudaGraphExecUpdate` of a parked leaf (store and `--cuda-graphs`
  walker). `--graph-set-params` is `cudaGraphExecKernelNodeSetParams` of a
  parked leaf (no second capture; legal with mem nodes). `--graph-clone` is `cudaGraphClone` of a leaf capture before
  instantiate (graph vs exec). `--graph-build` is `cudaGraphCreate` /
  `cudaGraphAddKernelNode` (and child add for combo parents; independent
  children may Hyper-Q overlap). `--graph-piecewise` is
  `cudaStreamBeginCaptureToGraph` combo parents (independent child roots;
  not with `--graph-build`).   `--graph-mem`
  is in-graph scratch (`cudaGraphAddMemAllocNode` / capture `cudaMallocAsync`);
  `--graph-update` is skipped because CUDA cannot update mem nodes.
  `--graph-auto-free` is AutoFreeOnLaunch scratch without a matching free
  (illegal with `--graph-mem`).
  `--timing-events` is timing-on copy events
  plus `event_elapsed_ns` (`cudaEventElapsedTime`); default wait events stay
  `cudaEventDisableTiming`. `memset` / `memset_buf` of a mapped span, directed peer enable, and
  the legacy null stream are mechanical CUDA invariants.
  `synchronize_stream` / `synchronize_event` / `synchronize_device` are
  `cudaStreamSynchronize` / `cudaEventSynchronize` / `cudaDeviceSynchronize`. `event_elapsed_ns` is `cudaEventElapsedTime` in
  nanoseconds (`create_event_disable_timing` forbids it). `query_event` is `cudaEventQuery`. `query_stream` is `cudaStreamQuery`.
  `mem_info` is `cudaMemGetInfo` `(free, total)`. Public `GpuOp` / `Operation` is the compiled DAG
  (`Sim::operations`).   `expertvm bench` on a multi-sequence trace prints
  `schedule-all` vs `schedule-1`, `schedule-chunk1` when a first token
  has more than one layer, and `schedule-decode-first` when a later token
  exists too. Multi-GPU profiles also print `schedule-gpu0` vs
  `schedule-striped` (`schedule_placed`) vs `schedule-remote`
  (`plan_placement` on the home hop, compute pinned on GPU0). `--managed`
  `--place remote` prefetches the expert onto home with PreferredLocation
  and GEMMs on GPU0 as a remote read (no second HBM copy, no weight D2D).
- performance model must include fixed overhead, size-dependent
  throughput, queueing, concurrency limits, alignment, startup latency
  (`LinkProfile::align_bytes` rounds the billed payload up; a 1-byte DMA
  cannot beat a 128-byte beat)
- never let an agent “win” by issuing 8,000 tiny copies that the model
  treats as free

Stop the model above warp schedulers / register banks / Tensor Core
circuitry. Those effects enter only through empirical kernel curves.
That is ~six orders of magnitude cheaper than cycle-level sim while
preserving the knobs inference-system agents actually turn.

---

## Phase 4 — physical validation (rent, do not buy)

Only after simulator/replay shows a hypothesis worth silicon.

Rent H100/H200/B200 time. Compare:

```
static EP  vs  expertvm
same model, same latency/quality envelope, $/output-token
```

If it does not transfer, the profile or the policy is wrong; fix the
model, do not celebrate the sim.

---

## Immediate next commits on `main`

1. [x] This document + the visible share extract.
2. [x] Complete share-API extract (transcript, searches, sources, raw JSON).
3. [x] Layered `Model` / `Session` API and crates.io metadata on `main`.
4. [x] README/STATUS rewritten to the verifiable-reference positioning, with
   this plan linked.
5. [x] MoE `ExpertAccess` trace emission behind a test/bin flag
   (`gguf_gemv trace`, `KvCache::enable_moe_trace`, identity vs untraced greedy).
6. [x] `ExpertStore` trait + `DirectStore` (identity) + `CachedStore` (leases)
   + `SimulatedGpuStore` in `expertvm`. Decode's expert inner products
   `acquire` then GEMV `ExpertParts`, then `release` (one expert at a time so
   `slots < top-k` still works). Direct/Cached/Simulated bit-match
   the blob path on writer tinies. Shared experts stay on the blob.
   Prefetch is copy-forward ∪ lookback-2 [`Markov`] (`MoeTraceBuf` when a store is
   attached; `sim_replay_cfg` `Prefetch::{None, CopyForward, Markov, Both}`).
7. [x] `gpu-sim` workspace crate: streams, events, HBM vs host-pinned,
   memcpy, leases, virtual clock. No CUDA. `expertvm sim` is the first app.
8. [x] `infer-bench` crate: adversarial workloads + trace replay scores.
   Dual score is semantic (`Ok` vs illegal GPU state) and performance
   (`wall_ns`, `hbm_peak`, `bytes_moved`, `energy_uj`, optional `ns_per_token`,
   `ttft_ns`, `itl_ns`). No invented `$/M tokens`. Energy is profile TDP ×
   virtual wall. `sim_replay` reports TTFT / mean ITL at token boundaries.
9. [x] `Model` / `Session` layered API with `attach_expert_store`.
10. [x] Engine-level prefix KV reuse (`KvCache::reuse_prefix`, `Llama::prompt`,
    persistent `serve` / `chat` cache). Greedy tokens and logits bit-match a
    cold prefill. Distinct from JSONL `"p"` / `--prefix-cache`.
11. [x] Paged KV on the reference engine (`KvPages`, intern by `prefix_hash`,
    COW on `refs > 1`). `Llama::new_paged_cache` / `Model::session_paged` /
    `serve --kv-page` / `chat --kv-page`. Logits and greedy bit-match dense.
    Distinct from `expertvm kv` (sim VMM pages).
12. [x] Shared `PagedKvPool`: two sequences intern-hit the same full-block
    prefixes (`Llama::new_paged_pool` / `new_paged_cache_on` /
    `Model::session_on_pool`). Default `new_paged_cache` still owns a private
    pool.
13. [x] Engine-level continuous batching (`Engine` / `EngineCfg`) on a shared
    `PagedKvPool`. Sequences may join mid-flight. Chunked prefill
    (`prefill_chunk`) interleaves with decode. Greedy ids match an independent
    `greedy_generate_cache` run. Intern hits across sequences. Distinct from
    `expertvm schedule` (trace-level, not real KV). Not an HTTP server.
14. [x] Engine recompute preemption: when `PagedKvPool` alloc returns
    `kv page cap`, drop unique KV for the live sequence with the most
    tokens (intern pins remain). Re-prefill and replay already sampled
    greedy ids so output still matches `greedy_generate_cache`. Distinct
    from intern eviction (`refs==1` in `by_hash`). Not an HTTP server.
15. [x] `KvPages` Drop releases unique table refs so `Engine::take` (and
    cache drop) can reuse a tight intern pool. `gguf_gemv engine` runs
    several `--prompt`s through `Engine` and prints intern_hits / preempts.
16. [x] Engine waiting queue: `add` beyond `max_seqs` parks the prompt.
    Finished slots retire (KV Drop) and waiters install. Greedy ids still
    match `greedy_generate_cache`. Distinct from `expertvm schedule` arrivals.
17. [x] Batched decode GEMM: `Llama::forward_batch` GEMMs Q/K/V, FFN, and
    lm_head across sequences that share one `PagedKvPool`. Attention stays
    per sequence. Engine decode (replay and sample) uses it. Logits
    bit-match sequential `forward`. Mixed dense/two-store falls back.
    MoE traces stay on the GEMM (per-row sequence / token / prefix).
    One attached store is used for the whole GEMM.
18. [x] Batched prefill GEMM: `Llama::prefill_batch` GEMMs the same matrices
    across sequences with equal or ragged chunk lengths. Engine prefills
    that are ready in the same step use it. Logits bit-match sequential
    `prefill`. Prefix intern bind stays per sequence before the GEMM.
19. [x] Mixed Engine prefill+replay GEMM: ready prefill chunks and unforwarded
    replay tokens share one `Llama::prefill_batch` in the same step.
20. [x] Engine GEMM stats: `EngineStats` counts steps, tokens in a
    cross-sequence GEMM, peak GEMM width, and serial fallback tokens.
    `gguf_gemv engine` prints them. Not wall-clock tok/s and not `$/M tokens`.
21. [x] Engine-level ExpertStore: one `LiveStore` on `Engine` is parked on the
    first cache of each batched GEMM so MoE serving stays on
    `prefill_batch` / `forward_batch`. DirectStore and CachedStore greedy ids
    match the blob Engine. `gguf_gemv engine --expert-slots N` (`0` = DirectStore).
    Two per-cache stores still fall back sequential. Markov prefetch
    (copy-forward ∪ lookback-2) is parked on the same cache across GEMMs
    so a tight CachedStore can prefetch.
22. [x] Engine SimulatedGpuStore: `Engine::expert_store_score` is the gpu-sim
    dual performance vector (`wall_ns`, HBM, bytes, `energy_uj`) after a
    batched MoE run. `gguf_gemv engine --expert-sim` attaches an example-H100
    store. Greedy ids still match the blob Engine. Not `$/M tokens`.
23. [x] Engine batched MoE traces: `Llama::prefill_batch` / `forward_batch`
    record per-row sequence / token / prefix so Engine serving traces
    without falling back sequential. `Engine::enable_moe_trace` /
    `take_moe_trace` banks events on retire and `take`.
    `gguf_gemv engine --trace-out FILE` writes JSONL. Events bit-match
    sequential traced prefill/forward and dense `greedy_generate_cache`.
    GEMM peak stays on the batched path. Dual score still has no `$/M tokens`.
24. [x] Grouped expert GEMM: tokens that select the same expert share one
    gate/up/down GEMM (acquire once per expert per layer). Softmax MoE
    walks (llama / Qwen2MoE / Qwen3MoE / Qwen3Next) and Llama4
    `weight_before_ffn` stay bit-equal to the serial GEMV path. Engine
    DirectStore hits drop below one-acquire-per-token-expert. Dual score
    still has no `$/M tokens`.
25. [x] Batched router GEMM: `ffn_gate_inp` is one GEMM of all tokens in the
    layer, then per-row softmax (or Llama4 sigmoid top-k). Bit-equal to the
    serial GEMV router. Dual score still has no `$/M tokens`.
26. [x] Engine `pin_hot`: sticky pins are distinct from in-flight leases so a
    decode `release` cannot drop keep-hot. After each GEMM, Engine pins
    last-used ∪ Markov keys up to `slots.saturating_sub(1)` (`slots == 1`
    pins nothing). SimulatedGpuStore `pin_hot` still NVLink-replicates.
    Dual score still has no `$/M tokens`.
27. [x] Prefetch this layer's unique routed experts after the router GEMM and
    before grouped expert GEMM, when that set fits in `slots`. Tight caches
    still demand-page. Dual score still has no `$/M tokens`.
28. [x] Engine `migrate`: after `pin_hot`, a multi-GPU SimulatedGpuStore D2Ds
    pinned experts onto GPU0 (striped-home odd experts move; already-home is
    a no-op; 1-GPU profiles skip). `StoreMetrics::migrates` counts src≠dst
    moves. Greedy ids still match the blob Engine. Dual score still has no
    `$/M tokens`.
29. [x] Engine `plan_placement`: after `pin_hot`, a multi-GPU SimulatedGpuStore
    D2Ds pinned experts onto GPU0 only when expert bytes beat
    `DECODE_ACTIVATION_BYTES * fan_in * reuse` on the GPU0↔GPU1 hop
    (online reuse, no future leak). Otherwise weights stay on the striped
    home (`StoreMetrics::dispatches`). `migrate` itself stays unconditional.
    Dual score still has no `$/M tokens`.
30. [x] `gguf_gemv engine --expert-8gpu` / `--expert-bytes N`: SimulatedGpuStore
    can use the example 8×H100 NVLink profile and a chosen page size so
    CLI metrics show `plan_placement` `migrates` / `dispatches` (default
    `--expert-sim` stays 1×H100 / 4096 bytes). Dual score still has no
    `$/M tokens`.
31. [x] SimulatedGpuStore `pin_hot` NVLink-replicates onto `(home + 1) % n_gpus`
    (not a fixed GPU1). `StoreMetrics::replicates` counts peer copies.
    Dual score still has no `$/M tokens`.
32. [x] Engine-backed `gguf_gemv serve --engine`: concurrent `POST /generate`
    admits onto one `Engine` so prefills GEMM together (`gemm_peak >= 2`).
    Default serve stays one connection at a time. Not OpenAI-compat. Dual
    score still has no `$/M tokens`.
33. [x] `serve --engine --expert-slots` / `--expert-sim` / `--expert-8gpu` /
    `--expert-bytes`: the HTTP Engine parks the same DirectStore /
    CachedStore / SimulatedGpuStore as `gguf_gemv engine`. Writer-tiny
    Qwen3MoE concurrent posts acquire from DirectStore and GEMM together.
    Dual score still has no `$/M tokens`.
34. [x] `serve --engine` JSON `page_hits` is the intern-hit delta on the
    shared `PagedKvPool` for that sequence. A later identical prompt
    intern-hits completed pages after the first `take`. Dual score still
    has no `$/M tokens`.
35. [x] Prefetch copy-forward ∪ Markov destinations **before** grouped
    expert GEMM so H2D of L+1 can overlap this layer's compute.
    `CachedStore::prefetch` skips unknown catalog keys. Dual score still
    has no `$/M tokens`.
36. [x] `serve --engine --prefill-chunk N` / `--trace-out FILE`: the HTTP
    Engine uses the same chunked prefill as `gguf_gemv engine` and appends
    batched MoE JSONL as sequences finish. Dual score still has no
    `$/M tokens`.
37. [x] `serve --engine` `"stream": true` is HTTP/1.1 chunked NDJSON token
    lines then a final `generated` object. Concurrent streams still GEMM
    together. Default serve ignores `stream`. Dual score still has no
    `$/M tokens`.
38. [x] Writer-built two-layer Qwen3MoE tiny (`qwen3moe.block_count=2`,
    `blk.1.*` cloned from `blk.0.*`) so copy-forward L+1 catalog keys exist.
    The independent oracle walks every `block_count` layer. Dual score still
    has no `$/M tokens`.
39. [x] Multi-layer JSONL pairs `seq_persist‰` / lookback-2 Markov on the
    **same layer**, not adjacent lines. `tests/traces/tiny-qwen3moe-2layer.jsonl`
    measures `layer_persist‰` and `seq_persist‰`. Dual score still has no
    `$/M tokens`.
40. [x] Adversarial `batch-1` vs `batch-128` (PLAN anti-Goodhart list).
    Engine SimulatedGpuStore identity, `wall_ns`, and `plan_placement` on
    the two-layer tiny (copy-forward L+1 prefetches more than 1-layer).
    `infer-bench` / `expertvm schedule` replay the checked-in two-layer
    JSONL; copy-forward hits L+1. Dual score still has no `$/M tokens`.
41. [x] Engine waiting queue at batch-128: 128 prompts with `max_seqs=8`
    finish with greedy ids matching independent decode. GEMM peak stays
    batched. Dual score still has no `$/M tokens`.
42. [x] Engine batch-128 on Qwen3MoE DirectStore (1-layer and 2-layer
    tinies): 128 waiters, `max_seqs=8`, store acquires, GEMM together.
    Dual score still has no `$/M tokens`.
43. [x] Engine `decode_first`: hold leftover prefill while any live
    sequence is already decoding (same policy as
    `expertvm schedule --decode-first`). Greedy ids still match
    independent decode. `gguf_gemv engine --decode-first` /
    `serve --engine --decode-first`. Dual score still has no `$/M tokens`.
44. [x] Engine batch-128 on Qwen3MoE CachedStore and SimulatedGpuStore
    (1-layer and 2-layer tinies): 128 waiters, `max_seqs=8`, greedy
    identity, store hits, gpu-sim `wall_ns`. Dual score still has no
    `$/M tokens`.
45. [x] Engine SimulatedGpuStore token-boundary scores: each newly
    sampled greedy token records the gpu-sim clock. `expert_store_score`
    fills `ttft_ns` / `itl_ns` / `ns_per_token`. Mixed leftover prefill +
    decode: `decode_first` shortens the decode sequence's ITL (same
    policy as `expertvm schedule --decode-first`). Dual score still has
    no `$/M tokens`.
46. [x] Engine `slo_reject`: drop waiters whose gpu-sim queue wait
    already meets `ttft_slo_ns` (same policy as
    `expertvm schedule --slo-reject`). Needs SimulatedGpuStore.
    `gguf_gemv engine --slo-reject --ttft-slo-ns N --expert-sim` /
    `serve --engine --expert-sim --slo-reject`. Dual score still has no
    `$/M tokens`.
47. [x] Engine `TieredStore`: two-sequence and batch-128 identity on
    writer-tiny Qwen3MoE (1-layer and 2-layer) with in-memory paging
    (`WeightStorage::mmap` still parked). Store hits, GEMM together.
    Dual score still has no `$/M tokens`.
48. [x] Engine SimulatedGpuStore CUDA graphs: default `--expert-sim` captures
    per-page GEMM graphs (`Engine::graph_launches`). `--graph-update` /
    `--graph-clone` / `--graph-build` / `--graph-piecewise` / `--graph-mem` / `--graph-auto-free` / `--timing-events` / `--cuda-graphs` match
    `GpuStoreCfg` / `expertvm sim`. Tight slots park+update. Identity stays.
    Dual score still has no `$/M tokens`.
49. [x] Engine `itl_slo_ns`: count later-token gaps over a virtual-ns budget
    (`Engine::itl_slo_miss`; does not drop). Mixed leftover prefill misses
    a mid SLO more than `decode_first`. `gguf_gemv engine --itl-slo-ns N
    --expert-sim` / `serve --engine --expert-sim --itl-slo-ns`. Dual score
    still has no `$/M tokens`.
50. [x] Engine batch-128 shared-prefix intern: 128 sequences with a shared
    `[1, 2]` first page intern-hit more completed KV pages than disjoint
    3-grams (`max_seqs=8`). Greedy ids still match independent decode.
    Distinct from JSONL `"p"` / `--prefix-cache`. Dual score still has no
    `$/M tokens`.
    51. [x] Engine SimulatedGpuStore fill modes: `--mapped` / `--managed` /
    `--vmm` select `GpuFill` on the serving path (`cudaHostAllocMapped`,
    `cudaMallocManaged`, `va_acquire`). Default `--expert-sim` stays pinned
    H2D identity. Two-sequence greedy ids still match independent decode.
    Dual score still has no `$/M tokens`.
52. [x] Engine SimulatedGpuStore CUDA knobs: `--host-func` /
    `--blocking-streams` / `--sync-alloc` / `--mempool` / `--vmm-page` /
    `--pageable` / `--accessed-by` / `--legacy-null` / `--stream-priority`
    match `GpuStoreCfg` / `expertvm sim` on `gguf_gemv engine` and
    `serve --engine --expert-sim`. Default pinned async stays decode
    identity. Dual score still has no `$/M tokens`.
53. [x] Engine predictor planner: `--prefetch none|copy-forward|markov|both`
    / `--plan-window N` / `--plan-threshold N` on `gguf_gemv engine` and
    `serve --engine`. Default `both` / window `0` (ungated) / threshold
    `500` matches copy-forward ∪ lookback-2. Stay vs Fetch uses
    `plan_keys` over unique predicted keys (no JSONL future leak).
    `--prefetch none` is demand paging (`prefetches=0`). Two-sequence
    greedy ids still match independent decode. Dual score still has no
    `$/M tokens`.

54. [x] Engine `--seq-streams`: per-sequence copy streams on SimulatedGpuStore
    (`sequence % copy_engines.max(2)`) so concurrent H2D can overlap, matching
    `expertvm sim --seq-streams`. Grouped expert GEMM stays on one compute
    stream (`StreamId(n_copy)`). Default `--expert-sim` stays copy NULL +
    compute stream 1. Two-sequence greedy ids still match independent decode.
    Dual score still has no `$/M tokens`.

55. [x] Engine `place_hot` fail-loud: SimulatedGpuStore drains that page's
    GEMM/copy streams before managed replica prefetch, `drop_managed_copy`,
    and VMM `va_unmap` (no whole-device `synchronize`, so other experts'
    overlapping H2D stay concurrent). `LiveStore::place_hot` returns `Result`;
    Engine `pin_predicted` propagates pin/migrate errors instead of swallowing
    `allocation N is still leased`. Two-sequence greedy ids still match
    independent decode on managed and pinned 8×H100 with 64-byte pages
    (`migrates >= 1`). Dual score still has no `$/M tokens`.

56. [x] Engine `--kv-sim`: interned paged KV on the same SimulatedGpuStore
    clock as expert H2D (`va_reserve` + memset on fault, kernel on intern
    hit, unmap on Drop). Distinct from `expertvm kv`. Default `--expert-sim`
    stays expert-only identity. `--kv-bytes N` overrides intern geometry so
    TTFT/ITL include KV traffic (`kv_misses >= 1`, shared-prefix intern
    `kv_hits > 0`, large pages strictly lengthen `wall_ns`). Two-sequence
    greedy ids still match independent decode. Dual score still has no
    `$/M tokens`.

57. [x] Engine `--decode-priority`: decode `forward_batch` GEMMs on a second
    SimulatedGpuStore compute stream at higher CUDA priority than leftover
    prefill (implies `--stream-priority`). Prefill/replay stays on the existing
    compute stream. Default `--expert-sim` keeps one compute stream. Two-sequence
    greedy ids still match independent decode. Dual score still has no
    `$/M tokens`.

58. [x] Engine `--decode-priority` ITL: token-boundary samples
    `synchronize_stream` the decode compute stream so leftover prefill on the
    lower-priority stream does not inflate ITL (implies `--stream-priority`).
    `cudaStreamSynchronize` of an already-idle stream does not start leftover
    kernels on other streams. 1-GPU `pin_hot` skips the GEMM-lease wait (no
    replica). Default `--expert-sim` keeps one compute stream and a full-device
    clock. Mixed leftover-prefill ITL is strictly shorter than without the
    knob; greedy ids still match. Dual score still has no `$/M tokens`.

59. [x] Engine `--compute-slots N`: Hyper-Q occupancy on SimulatedGpuStore
    so leftover prefill and decode GEMMs on different streams overlap at full
    issue rate when `N>=2` (not an SM-partition / green-context model). Needs
    `--decode-priority` for two compute streams. Default profile occupancy is
    exclusive (`1`), which keeps decode identity and stream-priority
    contention. `cudaStreamSynchronize` of an idle stream still does not start
    leftover kernels. Mixed leftover-prefill `wall_ns` is strictly shorter
    with two slots than with one; greedy ids still match. Dual score still has
    no `$/M tokens`.

60. [x] Engine `--decode-sms N`: green-context SM fraction on the decode
    compute stream (`Sim::set_stream_sm_permille`, `1..=1000` ‰ of peak
    FLOP/s). Compute-bound kernels scale; memory-bound keep full HBM. Leftover
    prefill gets the remainder. Implies `--decode-priority` (two compute
    streams). Default unset is a full chip, which keeps decode identity.
    Mixed leftover-prefill ITL is strictly longer at `250` than at full chip
    on a GEMM-bound profile; greedy ids still match. Dual score still has no
    `$/M tokens`.

61. [x] Trace-walker Hyper-Q + green-context: `expertvm sim` / `schedule` /
    `store` and `infer-bench schedule` take `--compute-slots N` and
    `--decode-sms N`. `SimCfg::compute_slots` overrides profile occupancy so
    independent `--seq-streams` GEMMs overlap when `N>=2`. `decode_sm_permille`
    caps every replay stream (compute-bound kernels scale; memory-bound keep
    full HBM). Default unset keeps exclusive compute and a full chip. Dual
    score still has no `$/M tokens`.

62. [x] Trace-walker `--decode-priority`: `expertvm sim` / `schedule` / `store`
    and `infer-bench schedule` run decode GEMMs on a second compute stream
    (`StreamId(n_copy + 1)`) and sample token-boundary ITL from that stream so
    leftover prefill does not inflate it. Implies `--stream-priority` at the
    CLI (library `SimCfg` does not). Walker `--decode-sms` does not imply
    this (token 0 is prefill). Mixed leftover-prefill ITL is strictly shorter
    than a full-device sample on a GEMM-bound profile. Dual score still has
    no `$/M tokens`.

63. [x] VMM `cuMemSetAccess` peer maps: `Sim::va_set_access` lets a kernel
    on GPU1 read a VA whose physicals live on GPU0 without dest HBM
    (NVLink billed; writes still need a local map). `GpuStoreCfg::accessed_by`
    / `SimCfg::accessed_by` apply at VMM fills, pin (skip dest map+D2D), and
    migrate (retarget GEMM, keep home physicals). Dual score still has no
    `$/M tokens`.

64. [x] VMM `cuMemGetAllocationGranularity`: `HardwareProfile::va_granularity_bytes`
    (`0`/`1` = any size, example default `1` so 4096-byte expert pages stay
    legal). `va_reserve` / `va_map_range` reject unaligned sizes
    (`unaligned VA`). A 2 MiB profile is opt-in (`with_va_granularity` /
    `va_granularity_bytes=` in a `.profile` file). Dual score still has no
    `$/M tokens`.

65. [x] Trace-walker VMM `--place replicas`: `expertvm schedule --vmm` maps
    dest then D2D like SimulatedGpuStore pin (dest eviction `va_unmap_range`).
    `--accessed-by` skips dest HBM (`va_set_access` at fill). Dual score still
    has no `$/M tokens`.

66. [x] Mempool `cudaMemPoolSetAccess`: `Sim::pool_set_access` lets a kernel
    on GPU1 read **and write** a `cudaMallocAsync` pointer whose physicals
    live on GPU0 without dest HBM (NVLink billed). `--accessed-by` on pinned
    async applies it to every default pool (pin/migrate/replicas skip dest
    D2D). `cudaMalloc` / `--sync-alloc` still D2Ds. Dual score still has no
    `$/M tokens`.

67. [x] VMM `cuMemCreate` / `cuMemMap` split: `Sim::va_create` charges HBM with
    no VA; `va_map_handle` maps that handle into a reserved VA without a
    second charge (two VAs may share one physical). `va_release_handle` is
    `cuMemRelease` when maps are 0. `va_map` still Create+Maps in one call.
    Dual score still has no `$/M tokens`.
68. [x] VMM `cuMemRetainAllocationHandle`: `Sim::va_retain_handle` increments
    handle refs (combined `va_map` spans are promoted). `va_release_handle`
    is `cuMemRelease` while mapped; HBM refunds when refs and maps are both
    0. `expertvm kv --sequences N` and Engine `--kv-sim` interned blocks
    are `cuMemCreate` + `cuMemMap` so sequence VAs share one physical.
    Dual score still has no `$/M tokens`.
69. [x] CUDA-graph mem alloc/free nodes: `cudaMallocAsync` / `cudaFreeAsync`
    (`Sim::alloc` / `free`) during stream capture record graph mem nodes.
    Host-sync `malloc` / `free_sync` / VMM / mempool create still cannot be
    captured. Relaunch without a matching free reuses the pointer (no second
    HBM charge). `clone_graph` forks those ids. `destroy_graph` refunds
    remaining graph mem. `update_graph` of mem nodes is Invalid. `expertvm
    --cuda-graphs` stays kernel-only (alloc on miss, then capture GEMM).
    Dual score still has no `$/M tokens`.
70. [x] CUDA `cudaGraphInstantiateFlagAutoFreeOnLaunch`:
    `Sim::instantiate_graph_auto_free` stream-ordered-frees graph mem allocs
    before a later launch so relaunch recharges HBM instead of reusing the
    pointer. Illegal with mem free nodes and after a default instantiate.
    Dual score still has no `$/M tokens`.
71. [x] VMM `cuMemSetAccess` PROT_READWRITE: `Sim::va_set_access_write` lets
    a kernel on a peer read **and write** mapped VMM physicals without dest
    HBM (interconnect billed), same class as `pool_set_access`. Default
    `va_set_access` stays PROT_READ (writes still need a local map). Dual
    score still has no `$/M tokens`.
72. [x] `cudaStreamAttachMemAsync`: `Sim::stream_attach` is stream-ordered
    visibility (`MemAttach::{Global,Host,Single}`). Default `alloc_managed`
    is Global. `alloc_managed_host` is `cudaMallocManaged(..., cudaMemAttachHost)`.
    Device kernels / memset / device prefetch fail `not attached` when Host
    or a different Single stream. Single cannot use the NULL stream. Capture
    is refused (`cudaErrorStreamCaptureUnsupported`). Dual score still has no
    `$/M tokens`.
73. [x] `cudaEventRecordExternal` / `cudaEventWaitExternal`:
    `Sim::record_event_external` / `wait_event_external` capture as event
    nodes without forked-capture join. A default `wait_event` on a captured
    default record still joins. Graph WaitExternal waits for a live record,
    not the same graph's record of that event. Dual score still has no
    `$/M tokens`.
74. [x] CUDA IPC: `Sim::ipc_get` / `ipc_open` / `ipc_close` are
    `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` / `cudaIpcCloseMemHandle`.
    The import aliases source physicals (no extra HBM). Free of the source
    while imports are live is Invalid. Capture is refused. Dual score still
    has no `$/M tokens`.
75. [x] `cudaGraphCreate` / `cudaGraphAdd*`: `Sim::create_graph` is an empty
    uninstantiated graph. `graph_add_kernel` / `graph_add_memcpy` /
    `graph_add_memset` / `graph_add_host_func` / `graph_add_event_record` /
    `graph_add_event_wait` / `graph_add_child` append nodes (illegal after
    instantiate and during capture). `--graph-build` uses this path in
    `SimulatedGpuStore` and the `--cuda-graphs` walker (combo parents are
    `graph_add_child` of instantiated leaves; no idle-stream wait). Dual
    score still has no `$/M tokens`.
76. [x] `cudaGraphAddMemAllocNode` / `cudaGraphAddMemFreeNode`:
    `Sim::graph_add_alloc` / `graph_add_free` append mem nodes (illegal after
    instantiate and during capture). `--graph-mem` captures/builds leaf GEMM
    graphs with in-graph scratch (`GRAPH_SCRATCH_BYTES`); hits/misses stay the
    same and HBM peak includes the workspace. CUDA cannot
    `cudaGraphExecUpdate` mem nodes, so `--graph-update` is skipped.
    Implies `--cuda-graphs` on the walker. Decode identity stays kernel-only
    graphs. Dual score still has no `$/M tokens`.
77. [x] `cudaGraphInstantiateFlagAutoFreeOnLaunch` in expertvm: `--graph-auto-free`
    records leaf GEMM scratch without a matching free and instantiates with
    `Sim::instantiate_graph_auto_free` so relaunch recharges HBM. Illegal with
    `--graph-mem` (CUDA cannot AutoFree a graph that has mem free nodes).
    `--graph-update` is skipped. Implies `--cuda-graphs` on the walker. Decode
    identity stays kernel-only graphs. Dual score still has no `$/M tokens`.
78. [x] `cudaLaunchCooperativeKernel`: `Sim::cooperative_kernel` /
    `graph_add_cooperative_kernel` occupy every Hyper-Q
    [`GpuProfile::compute_slots`] so leftover kernels cannot overlap (CUDA
    `cudaDevAttrCooperativeLaunch`; example H100 is true). Capture is allowed
    (CUDA 11+). `--cooperative` on `expertvm sim` / `schedule` / `store`,
    `gguf_gemv engine` / `serve --engine --expert-sim`, and `infer-bench
    schedule` launches grouped GEMMs that way. Decode identity stays
    `cudaLaunchKernel`. Dual score still has no `$/M tokens`.
79. [x] `cudaGraphAddDependencies`: `Sim::graph_add_dependencies` records
    predecessor edges on `create_graph` nodes (illegal after instantiate and
    during capture). Independent nodes (empty deps) launch on internal streams
    so Hyper-Q can overlap them; the launch stream still waits for the graph.
    Stream capture records same-stream edges. `--graph-build` combo parents
    leave sibling `graph_add_child` nodes independent. `update_graph` treats
    edges as topology. Leaf `--graph-mem` / `--graph-auto-free` chains
    alloc→kernel→free. Decode identity stays stream capture. Dual score still
    has no `$/M tokens`.

80. [x] Hopper `cuMulticastCreate` NVLS replica fanout: `Sim::multicast_create` /
    `multicast_add_device` / `multicast_bind_mem` / `va_map_multicast` are
    `cuMulticastCreate` / `cuMulticastAddDevice` / `cuMulticastBindMem` /
    `cuMemMap` of a multicast handle. The team must be an NVLink clique
    (PCIe P2P and RDMA refuse). Bind uses existing VMM physicals (dest HBM is
    already charged). A kernel write to the multicast VA is one NVLS hop on
    compute, not N sequential copy-engine D2Ds. Capture cannot include
    create/add/bind/map. `Sim::multicast_store` binds whole-VA maps and
    enqueues that kernel. `--multicast` on `expertvm sim` / `schedule` /
    `store`, `gguf_gemv engine` / `serve --engine --expert-sim`, and
    `infer-bench schedule` implies `--vmm` and uses NVLS for `--place replicas`
    / `pin_hot`. Illegal with `--accessed-by` or `--vmm-page`. Decode identity
    stays copy-engine D2D. Dual score still has no `$/M tokens`.

81. [x] `cudaGraphExecKernelNodeSetParams`: `Sim::graph_exec_kernel_set_params`
    patches one instantiated kernel node's pointers / `KernelKind` without a
    second graph (`graph_set_params_ns`, cheaper than `cudaGraphExecUpdate`).
    Cooperative flag and edges stay (topology). Works on graphs with mem
    alloc/free nodes (CUDA cannot `cudaGraphExecUpdate` those). Capture cannot
    include it. `--graph-set-params` on `expertvm sim` / `schedule` / `store`,
    `gguf_gemv engine` / `serve --engine --expert-sim` parks leaf execs on
    evict and retargets the unique kernel (implies `--cuda-graphs` on the
    walker). Illegal with `--graph-update`. Decode identity stays
    destroy+instantiate.     Dual score still has no `$/M tokens`.

82. [x] Mempool shareable-handle IPC: `Sim::create_shareable_pool` is
    `cudaMemPoolCreate` with a POSIX-FD handle type. `Sim::pool_export` /
    `pool_import` are `cudaMemPoolExportToShareableHandle` /
    `ImportFromShareableHandle` (new `PoolId` shares live/cached/threshold;
    no extra HBM). `Sim::pool_export_ptr` / `pool_import_ptr` are
    `cudaMemPoolExportPointer` / `ImportPointer` (alias, no extra HBM).
    `Sim::set_device_mempool` is `cudaDeviceSetMemPool`. Default and
    `create_pool` pools cannot be exported. `ipc_get` of a mempool alloc is
    Invalid. Capture cannot include export/import. `--shareable` on
    `expertvm sim` / `schedule` / `store`, `gguf_gemv engine` / `serve
    --engine --expert-sim` implies `--mempool`, rebinds `cudaMallocAsync` to
    the shareable pool, and keeps an imported sibling so a later
    `alloc_from_pool` hits cache without extra HBM. Illegal with
    `--sync-alloc` / mapped / managed / vmm. Decode identity stays the
    device default pool. Dual score still has no `$/M tokens`.

83. [x] `cudaGraphExecMemcpyNodeSetParams`: `Sim::graph_exec_memcpy_set_params`
    patches one instantiated memcpy node's `MemcpyOp` without a second graph
    (`graph_set_params_ns`, cheaper than `cudaGraphExecUpdate`). Pageable
    copies stay illegal. Works on graphs with mem alloc/free nodes (CUDA
    cannot `cudaGraphExecUpdate` those). Capture cannot include it.
    `--graph-set-params` retargets a unique memcpy on a parked leaf if
    present (copy engine stays off the compute GEMM graph). Decode identity
    stays destroy+instantiate. Dual score still has no `$/M tokens`.

84. [x] `cudaGraphExecMemsetNodeSetParams`: `Sim::graph_exec_memset_set_params`
    patches one instantiated memset node's `KernelBuf` without a second
    graph (`graph_set_params_ns`, cheaper than `cudaGraphExecUpdate`).
    Zero-byte fills stay illegal. Works on graphs with mem alloc/free nodes.
    Capture cannot include it. `--graph-set-params` retargets a unique
    memset on a parked leaf if present. Decode identity stays
    destroy+instantiate. Dual score still has no `$/M tokens`.

85. [x] `cudaGraphNodeSetEnabled`: `Sim::graph_node_set_enabled` /
    `graph_node_get_enabled` skip an instantiated node at launch without a
    second graph (`graph_set_params_ns`). Dependents treat a disabled node as
    already complete. Memory alloc/free nodes cannot be disabled. Capture
    cannot include it. Decode identity stays every node enabled. Dual score
    still has no `$/M tokens`.

86. [x] `gguf_gemv engine --bench` records batched MoE traces and prints
    `expertvm::report` (policy table; with `--expert-sim` the same sim A/B
    lines as `infer-bench trace`). `--capacity N` is the replay cache size
    (default `--expert-slots`, or 8). Does not add llama-rust as an
    infer-bench dep. Dual score still has no `$/M tokens`.

87. [x] `cudaGraphExecChildGraphNodeSetParams`:
    `Sim::graph_exec_child_set_params` patches one instantiated child-graph
    node's nested graph without a second parent (`graph_set_params_ns`).
    Nested topology (device, stream, op kinds, cooperative flag, deps) must
    match; nested child ids are parameters (`cudaGraphExecUpdate` treats
    those ids as topology). The new child must already be instantiated on
    the same GPU. Capture cannot include it. Graphs with mem alloc/free
    nodes are legal. `--graph-set-params` parks combo parents and retargets
    nested leaves. Decode identity stays destroy+instantiate. Dual score
    still has no `$/M tokens`.

88. [x] `cudaGraphExecEventRecordNodeSetEvent` /
    `cudaGraphExecEventWaitNodeSetEvent`:
    `Sim::graph_exec_event_record_set_event` /
    `graph_exec_event_wait_set_event` patch the event id on an instantiated
    record or wait node (`graph_set_params_ns`). The External flag stays
    (topology). Capture cannot include it. Graphs with mem alloc/free nodes
    are legal. Decode identity stays kernel-only graphs. Dual score still
    has no `$/M tokens`.

89. [x] `cudaGraphRemoveDependencies`: `Sim::graph_remove_dependencies`
    drops a `from` → `to` edge on an uninstantiated graph (illegal after
    instantiate and during capture). Missing edges are a no-op. Independent
    nodes may Hyper-Q overlap at launch. `cudaGraphExecUpdate` treats remaining
    edges as topology. Decode identity stays stream-capture edges. Dual score
    still has no `$/M tokens`.

90. [x] `cudaStreamBeginCaptureToGraph`: `Sim::begin_capture_to_graph`
    records later submits into an existing uninstantiated graph. Capture
    roots additionally depend on the given node indices; empty `deps` makes
    those nodes extra roots so they may Hyper-Q overlap prior work.
    `end_capture` returns that graph. Nested capture, instantiated graphs,
    and GPU mismatch are Invalid. `graph_root_nodes` / `graph_edges` /
    `graph_node_dependents` are `cudaGraphGetRootNodes` / `GetEdges` /
    `NodeGetDependentNodes`. `--graph-piecewise` on `expertvm sim` /
    `schedule` / `store`, `gguf_gemv engine` / `serve --engine --expert-sim`
    captures combo parents as independent child roots (implies
    `--cuda-graphs` on the walker). Illegal with `--graph-build`. Decode
    identity stays a single `begin_capture` of child launches. Dual score
    still has no `$/M tokens`.

91. [x] `cudaGraphAddEmptyNode`: `Sim::graph_add_empty` is a join/fork with
    no work (1 ns, no compute or copy occupancy). Capture cannot include it.
    Illegal after instantiate. May be disabled (`cudaGraphNodeSetEnabled`).
    `cudaStreamBeginCaptureToGraph` can name an empty node as a dependency
    anchor. Decode identity stays kernel-only graphs. Dual score still has
    no `$/M tokens`.

92. [x] `cudaStreamUpdateCaptureDependencies`:
    `Sim::stream_update_capture_dependencies` extra deps for the next
    captured node **in addition to** stream-order (`Set` replaces, `Add`
    unions). Indices are existing graph nodes, then this-session nodes at
    `graph_len + i` (`graph_len` during capture excludes the session
    buffer). `stream_is_capturing` / `stream_capture_info` are
    `cudaStreamIsCapturing` / `GetCaptureInfo`. `graph_node_kind` is
    `cudaGraphNodeGetType`. Decode identity stays stream-capture edges (no
    extra pending). Dual score still has no `$/M tokens`.

93. [x] Conditional IF graphs: `Sim::graph_conditional_create` is
    `cudaGraphConditionalHandleCreate` (create-time default applied on each
    `launch_graph`). `graph_add_if` adds an IF node and returns the body
    graph. Body ops skip at start when the handle is `0` (no compute/copy
    occupancy, no alloc side effects). `set_conditional` is device
    `cudaGraphSetConditional` (capture allowed). Decode identity stays
    kernel-only graphs with no IF nodes. Dual score still has no `$/M
    tokens`.

94. [x] Conditional WHILE and SWITCH: `Sim::graph_add_while` is
    `cudaGraphCondTypeWhile` (body repeats while the handle is non-zero;
    Invalid after 64 iterations). `graph_add_switch` is
    `cudaGraphCondTypeSwitch` (`n` bodies, `1..=64`; branch `i` runs when
    the handle equals `i`; out of range skips every body). Decode identity
    stays kernel-only graphs. Dual score still has no `$/M tokens`.

95. [x] `cudaGraphNodeFindInClone`: `Sim::graph_node_find_in_clone` maps a
    node index on the original onto the same index of a graph produced by
    `clone_graph` of that original (nested graphs cloned in that call
    count). A second clone of the clone does not map the first original.
    Capture is allowed. Decode identity stays kernel-only graphs. Dual
    score still has no `$/M tokens`.

96. [x] `cudaGraphInstantiateWithFlags` / `cudaGraphExecGetFlags`:
    `Sim::instantiate_graph_with_flags` accepts
    `GraphInstantiateFlags::AUTO_FREE_ON_LAUNCH` and `UPLOAD` (upload during
    instantiate so the first launch skips `upload_graph`). Device-launch and
    node-priority bits are Invalid. `graph_exec_get_flags` returns those bits
    on an instantiated exec. Decode identity stays default flags. Dual score
    still has no `$/M tokens`.

97. [x] `cudaDeviceGetGraphMemAttribute`: `Sim::graph_mem_get` counts live
    graph-mem allocs (`graph_add_alloc` / captured `cudaMallocAsync`), not
    `malloc` / live `alloc`. Reserved equals used (those allocs charge
    device HBM directly). `graph_mem_set` resets a High attr when `value` is
    `0`. `graph_mem_trim` is host-sync and does not change `mem_info`.
    Capture cannot include set/trim. Decode identity stays kernel-only
    graphs. Dual score still has no `$/M tokens`.

98. [x] `cudaGraphInstantiateFlagUseNodePriority`:
    `instantiate_graph_with_flags(USE_NODE_PRIORITY)` schedules recorded
    kernels with the priority snapshotted at add/capture
    (`cudaKernelNodeAttributePriority`) instead of the launch stream.
    `graph_kernel_node_get_priority` / `set_priority` /
    `copy_attributes` are Get/Set/CopyAttributes. `stream_copy_attributes`
    is `cudaStreamCopyAttributes` (priority and SM permille). Device-launch
    is still Invalid. Decode identity stays default flags. Dual score still
    has no `$/M tokens`.

99. [x] Graph vs exec snapshot: `instantiate_graph` clones steps into an
    exec snapshot (`cudaGraph_t` vs `cudaGraphExec_t` on the same GraphId).
    `launch_graph` / `cudaGraphExec*SetParams` / `cudaGraphNodeSetEnabled`
    use the snapshot. `graph_kernel_set_params` is
    `cudaGraphKernelNodeSetParams` on the graph and does not retarget an
    already-instantiated exec. `graph_exec_kernel_node_set_priority` is
    the exec attribute; `graph_kernel_node_set_priority` stays graph-side.
    `update_graph` replaces the snapshot, not the graph. Decode identity
    stays destroy+instantiate / ExecSetParams. Dual score still has no
    `$/M tokens`.

100. [x] Dual-score `$/M tokens`: `HardwareProfile::rent_usd_micros_per_hour`
    is an example list-price knob (`0` omits dollars; example profiles stay
    `0`). `Score::with_tokens` fills `usd_micros_per_m_tokens =
    rent * wall_ns * 1e6 / (hour_ns * n_tokens)` microdollars per million
    tokens. Not a capture. `gpu-profile capture` is still refused.

101. [x] Graph-side memcpy/memset SetParams: `graph_memcpy_set_params` /
    `graph_memset_set_params` are `cudaGraphMemcpyNodeSetParams` /
    `cudaGraphMemsetNodeSetParams` on the graph definition. After
    instantiate they do not retarget the exec snapshot. Decode identity
    stays ExecSetParams. `gpu-profile capture` is still refused.

102. [x] Stream wait/write-value: `Sim::wait_value64` / `write_value64` /
    `wait_value32` / `write_value32` are `cuStreamWaitValue*` /
    `cuStreamWriteValue*` (`WaitValueCmp` Eq/Geq/And/Nor). A mailbox
    updates on write **complete**; unwritten locations read as 0.
    Kernel/memset/memcpy stores are not modeled. Wait stays pending until
    the compare matches (no compute/copy occupancy). Cross-stream write
    unblocks wait without an event; unsatisfied wait + `synchronize` is
    deadlock. `graph_add_wait_value64` / `graph_add_write_value64` /
    `graph_add_batch_mem_op` are `cudaGraphAddBatchMemOpNode` (a multi-item
    batch is sequential nodes with deps). Graph vs exec SetParams match
    memcpy. Decode identity stays kernel-only graphs (not wired into
    expertvm). `gpu-profile capture` is still refused.

103. [x] `cudaGraphInstantiateFlagDeviceLaunch`:
    `instantiate_graph_with_flags(DEVICE_LAUNCH)` on kernel/memcpy/memset
    graphs. `device_launch_graph` is device-side `cudaGraphLaunch` after
    upload (host `launch_graph` still auto-uploads). Launcher occupies one
    compute slot for `graph_launch_ns`; the body enqueues when it completes.
    Mem alloc/free, events, child graphs, conditionals, and host nodes are
    Invalid. `update_graph` of a device-launch exec is Invalid. Overlapping
    device launches of the same exec are Invalid. Decode identity stays
    default flags (no device launch). `gpu-profile capture` is still
    refused.

104. [x] True `cudaGraphAddBatchMemOpNode` / `cuStreamBatchMemOp`: a multi-item
    batch is **one** `GpuOp::BatchMem` node (not sequential wait/write nodes
    with deps). `batch_mem_op` is the live API; a single item still submits
    `WriteValue` / `WaitValue`. A wait sees earlier writes in that vector
    (overlay) and does not see later ones. Writes commit on complete.
    `graph_node_set_enabled` skips the whole batch. Item lists are parameters
    for `update_graph` and `graph_exec_batch_mem_ops_set_params`. Decode
    identity stays kernel-only graphs. `gpu-profile capture` is still refused.

105. [x] Separate `cudaGraph_t` / `cudaGraphExec_t` handles: `instantiate_graph`
    returns a new exec id. The source graph stays a definition and may be
    instantiated again (kernel/memcpy graphs; mem alloc/free graphs are
    Invalid — execs would need independent pointers). `launch_graph` of a
    definition uses the primary exec. `graph_add_*` stays legal on the
    definition after instantiate and is Invalid on the exec. Destroying the
    definition refunds graph mem and leaves execs launchable. Decode identity
    still launches the definition (primary exec). Dual score still has no
    `$/M tokens`. `gpu-profile capture` is still refused.

106. [x] Host-node SetParams: `HostNodeParams` (`fn_id` / `user_data`) on
    `GpuOp::HostFunc`. `host_func` / `graph_add_host_func` stay the unnamed
    default (`0`, `0`). `host_func_params` / `graph_add_host_func_params`
    record a payload. `graph_host_set_params` is `cudaGraphHostNodeSetParams`
    on the definition (does not retarget an already-instantiated exec).
    `graph_exec_host_set_params` is `cudaGraphExecHostNodeSetParams`
    (`graph_set_params_ns`). `fn_id` / `user_data` are parameters for
    `update_graph` (not topology). Capture records the params.
    Device-launch graphs still refuse host nodes. Decode identity stays
    kernel-only graphs (expertvm `--host-func` still uses the unnamed live
    callback). `gpu-profile capture` is still refused.

107. [x] `cudaStreamCaptureMode`: `begin_capture` / `begin_capture_to_graph`
    default to Relaxed (independent streams stay live; `wait_event` of a
    captured record still joins an idle stream). `begin_capture_with_mode` /
    `begin_capture_to_graph_with_mode` accept Global / ThreadLocal / Relaxed.
    Global and ThreadLocal are the same in this single-threaded VM: submits
    on a stream not in the capture set are Invalid (`stream not capturing`),
    except a joining wait. `thread_exchange_stream_capture_mode` is
    `cudaThreadExchangeStreamCaptureMode` (next `begin_capture`; in-flight
    capture keeps its mode). `stream_capture_info` reports the mode.
    Decode identity stays Relaxed. `gpu-profile capture` is still refused.

108. [x] `cudaGraphInstantiateWithParams`: `instantiate_graph_with_params`
    fills `GraphInstantiateParams` result and err node even on `Err`.
    Success is `GraphInstantiateResult::Success` with `err_node = None`.
    Device-launch of a host/event/mem node is `NodeOperationNotSupported`
    with that node index. Auto-free plus a mem-free node, and a second
    instantiate of a mem graph, are `InvalidStructure`. Unknown flags and
    capture are `Error`. `instantiate_graph_with_flags` uses this path.
    Decode identity stays `instantiate_graph`. `gpu-profile capture` is
    still refused.

109. [x] Indexed graph/exec GetParams: `graph_kernel_get_params` /
    `graph_memcpy_get_params` / `graph_memset_get_params` /
    `graph_host_get_params` / `graph_batch_mem_ops_get_params` are
    `cudaGraph*NodeGetParams` on the definition. `graph_exec_*_get_params`
    reads the exec snapshot (uninstantiated is Invalid). Unique helpers
    still use `resolved_graph` (launched snapshot). Decode identity stays
    kernel-only graphs. `gpu-profile capture` is still refused.

110. [x] `cudaGraphExecUpdateResultInfo`: `update_graph_with_info` fills
    result, `error_node` (src), and `error_from_node` (exec / from-edge)
    even on `Err`. Classifies TopologyChanged (count, device/stream, child
    id), NodeTypeChanged, DependenciesChanged, ParametersChanged
    (cooperative, wait/write), AttributesChanged (event External),
    NotSupported (mem nodes, device-launch), and Error (capture, same id,
    uninstantiated). `update_graph` uses this path and keeps the same
    `why` strings. Decode identity stays `update_graph`. `gpu-profile
    capture` is still refused.

111. [x] `cudaUserObjectCreate` / `cudaGraphRetainUserObject`:
    `user_object_create` requires `UserObjectFlags::NO_DESTRUCTOR_SYNC`
    and a non-zero initial refcount. `user_object_retain` /
    `user_object_release` adjust caller refs. `graph_retain_user_object`
    / `graph_release_user_object` are definition-only (`MOVE` transfers
    one caller ref; extra flags and exec ids are Invalid). Destroy of
    a graph releases its refs. The last remaining ref records
    `user_object_destructors` (`fn_id`; no Rust callback). Clone does
    not copy retains. Capture cannot include it. Decode identity does
    not create user objects. `gpu-profile capture` is still refused.

112. [x] Programmatic dependent launch: `kernel_pdl` /
    `graph_kernel_node_set_pdl` are `cudaLaunchAttributeProgrammaticStreamSerialization`
    plus an implicit `cudaTriggerProgrammaticLaunchCompletion` at
    `GpuProfile::pdl_trigger_permille` of the primary. A wait kernel's
    same-stream dep is that trigger, not completion. Overlap needs
    `compute_slots >= 2` (example H100 stays exclusive). CopyAttributes
    copies PDL. Decode identity stays `kernel` (both flags false).
    `gpu-profile capture` is still refused.

113. [x] expertvm / Engine `--pdl`: grouped expert GEMMs launch with
    `kernel_pdl` wait+trigger (graph leaves `graph_kernel_node_set_pdl`)
    so consecutive same-stream kernels may overlap after the previous
    trigger when `compute_slots >= 2`. Illegal with `--cooperative`.
    `cudaFreeAsync` still waits for the overlapped primary. Decode identity
    stays `kernel`. `gpu-profile capture` is still refused.

114. [x] `cudaLaunchAttributeProgrammaticEvent`: `kernel_pdl_event` /
    `graph_kernel_node_set_programmatic_event` record an event at the PDL
    trigger (or kernel completion when trigger is false). Other streams may
    `wait_event` it and start before the primary finishes. Same-stream later
    work still waits for completion. `query_event` / `synchronize_event` /
    `event_elapsed_ns` use the trigger. Device-launch graphs refuse it
    (event). CopyAttributes copies it. Decode identity stays `kernel`.
    `gpu-profile capture` is still refused.

115. [x] `cudaLaunchAttributeLaunchCompletionEvent`:
    `kernel_launch_completion` / `graph_kernel_node_set_launch_completion`
    record an event when the kernel *starts* (grid launched), not when it
    finishes. Other streams may `wait_event` it and overlap leftover compute
    with copies. Device-launch graphs refuse it. CopyAttributes copies it.
    Decode identity stays `kernel`. `gpu-profile capture` is still refused.

116. [x] `cudaLaunchAttributeAccessPolicyWindow`: `kernel_access_policy` /
    `graph_kernel_node_set_access_policy` apply a persisting (or streaming)
    L2 window. `set_persisting_l2_cache_size` is
    `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize)` (CUDA default 0).
    The first kernel fills persisting lines; a later overlapping kernel
    bills HBM at `1000 - l2_persist_hit_permille`. `reset_persisting_l2_cache`
    colds the next fill. `miss` cannot be Persisting. Device-launch graphs
    allow it (not an event). CopyAttributes copies it. `expertvm sim
    --l2-persist` / Engine `--expert-sim --l2-persist` enable the limit and
    attach a window to expert GEMMs. Decode identity stays `kernel` with
    persist limit 0. `gpu-profile capture` is still refused.

117. [x] `cudaLaunchAttributeMemSyncDomain` / `MemSyncDomainMap`:
    `kernel_with` and stream/graph SetAttribute select a logical domain
    (Default / Remote) and map it onto `GpuProfile::mem_sync_domain_count`
    physical ids (example H100 is 4). A completing kernel's implicit fence
    waits `same_domain_fence_permille` of leftover same-physical-domain
    kernel/allreduce traffic (default 0 = identity). Allreduce tags Remote
    like NCCL. Graph replay uses the node's map, not the launch stream.
    Device-launch graphs allow it. CopyAttributes copies it. Decode identity
    stays `kernel` (Default, identity map, tax 0). `gpu-profile capture` is
    still refused.

118. [x] `cudaLaunchAttributeClusterDimension`: `kernel_with` /
    `graph_kernel_node_set_cluster` launch a Hopper thread-block cluster.
    Product of `{x,y,z}` occupies `min(blocks, compute_slots)` Hyper-Q slots
    so leftover kernels cannot overlap a cluster that fills the cap. Zero
    dims and product `> GpuProfile::max_blocks_per_cluster` (example H100
    portable 8) are Invalid. Device-launch graphs allow it. CopyAttributes
    copies it. Decode identity stays `kernel` (no cluster). `gpu-profile
    capture` is still refused.

119. [x] expertvm / Engine `--cluster N`: grouped expert GEMMs launch with
    `cudaLaunchAttributeClusterDimension` `{N,1,1}` (graph leaves
    `graph_kernel_node_set_cluster`) so a cluster that fills
    `compute_slots` cannot Hyper-Q overlap leftover kernels. Occupies
    `min(N, compute_slots)`. `N==0` is refused at parse; `N > max_blocks_per_cluster`
    (Hopper portable 8) is Invalid at launch. Legal with `--pdl` and
    `--cooperative`. Decode identity stays `kernel` (no cluster).
    `gpu-profile capture` is still refused.

120. [x] `cudaFuncAttributeNonPortableClusterSizeAllowed`:
    `set_non_portable_cluster_size_allowed` lets a cluster exceed
    `GpuProfile::portable_cluster_size` up to `max_blocks_per_cluster`.
    Default is disallowed (Hopper portable 8). Example H100 keeps both at 8.
    Decode identity stays disallowed. `gpu-profile capture` is still refused.

121. [x] `cudaLaunchAttributeClusterSchedulingPolicyPreference`:
    Default / LoadBalancing occupy `min(blocks, compute_slots)`. Spread
    occupies every Hyper-Q slot so leftover kernels cannot overlap.
    Graph SetAttribute / CopyAttributes carry it. Device-launch graphs
    allow it. Decode identity stays Default. `gpu-profile capture` is
    still refused.

122. [x] `cudaLaunchAttributePreferredClusterDimension`: a preferred
    cluster must be an integer multiple of the required
    `ClusterDim` (CUDA: minimum must also be specified). Occupancy uses
    the preferred size when it fits in `compute_slots`, else the required
    size. Graph SetAttribute / CopyAttributes carry it. Device-launch
    graphs allow it. Decode identity stays no preferred dim.
    `gpu-profile capture` is still refused.

123. [x] expertvm / Engine `--cluster-spread`: grouped expert GEMMs launch
    with `cudaLaunchAttributeClusterSchedulingPolicyPreference` Spread
    (`graph_kernel_node_set_cluster_policy`). Occupies every Hyper-Q slot
    so leftover kernels cannot overlap even when `--cluster N` is smaller
    than `compute_slots`. A no-op without `--cluster` of at least 2.
    Legal with `--pdl` and `--cooperative`. Decode identity stays Default.
    `gpu-profile capture` is still refused.

124. [x] `cudaLaunchAttributePreferredSharedMemoryCarveout`: Default and
    MaxL1 keep current Hyper-Q occupancy. MaxShared occupies every slot
    so leftover kernels cannot overlap. Graph SetAttribute / CopyAttributes
    carry it. Device-launch graphs allow it. Decode identity stays Default.
    `gpu-profile capture` is still refused.

125. [x] expertvm / Engine `--max-shared`: grouped expert GEMMs launch with
    `cudaLaunchAttributePreferredSharedMemoryCarveout` MaxShared
    (`graph_kernel_node_set_carveout`). Occupies every Hyper-Q slot so
    leftover kernels cannot overlap. Legal with `--pdl` and `--cooperative`.
    Decode identity stays Default. `gpu-profile capture` is still refused.

126. [x] expertvm / Engine `--preferred-cluster N`: grouped expert GEMMs
    launch with `cudaLaunchAttributePreferredClusterDimension` `{N,1,1}`
    (`graph_kernel_node_set_preferred_cluster`) after the required
    `--cluster`. Occupancy uses the preferred size when it fits in
    `compute_slots`, else the required dim. `N==0` is refused at parse;
    preferred without `--cluster` is refused; `N` must be a multiple of
    `--cluster`. Legal with `--pdl` and `--cooperative`. Decode identity
    stays no preferred dim. `gpu-profile capture` is still refused.

127. [x] expertvm / Engine `--non-portable-cluster`:
    `cudaFuncSetAttribute` NonPortableClusterSizeAllowed on every GPU so
    `--cluster N` may exceed `portable_cluster_size` up to
    `max_blocks_per_cluster`. Default is disallowed (Hopper portable 8;
    example H100 keeps both at 8 so the flag is a no-op unless the profile
    opens max). `N > max_blocks_per_cluster` stays Invalid. Legal with
    `--pdl` and `--cooperative`. Decode identity stays disallowed.
    `gpu-profile capture` is still refused.

128. [x] `cudaLaunchAttributeSynchronizationPolicy`: stream-only
    (`cudaStreamSetAttribute` / CopyAttributes). Auto (default) / Spin /
    Yield / BlockingSync. Host-wait tax on `synchronize_stream` and
    `synchronize_event` (recording stream) after the GPU drain; profile
    `host_sync_spin_ns` / `yield` / `blocking` default 0 so decode identity
    and existing timing tests stay green. Auto tax is always 0.
    `synchronize` / `synchronize_device` do not take the tax (`cudaDeviceSynchronize`).
    Not a kernel launch attribute. Decode identity stays Auto.
    `gpu-profile capture` is still refused.

129. [x] expertvm / Engine `--sync-policy auto|spin|yield|blocking`:
    `cudaStreamSetAttribute` SynchronizationPolicy on copy/prefill/decode
    streams. Host-wait tax on `synchronize_stream` (decode-stream ITL when
    `--decode-priority`) after the GPU drain. Auto (default) tax is 0.
    Profile `host_sync_*_ns` default 0 so decode identity stays green.
    Legal with `--pdl` and `--cooperative`. Decode identity stays Auto.
    `gpu-profile capture` is still refused.

130. [x] `cudaLaunchAttributeDeviceUpdatableKernelNode`: kernel-node flag
    (graph SetAttribute / CopyAttributes / `KernelAttrs`). Default false.
    `graph_exec_kernel_set_params` skips clearing the upload flag so a
    later `device_launch_graph` needs no host `upload_graph`. Control
    without the flag still requires re-upload. Device-launch graphs allow
    it (unlike programmatic/launch-completion events). Decode identity
    stays not device-updatable. `gpu-profile capture` is still refused.

131. [x] `cudaLaunchAttributeSharedMemoryMode`: kernel-node bank width
    Default / FourByte / EightByte (graph SetAttribute / CopyAttributes /
    `KernelAttrs`). Default never scales duration. FourByte / EightByte
    scale kernel time by `1000 / GpuProfile::shared_mem_*_permille`
    (profile default 1000 is identity so decode stays green). Device-launch
    graphs allow it. Decode identity stays Default.
    `gpu-profile capture` is still refused.

132. [x] expertvm / Engine `--shared-mem default|four|eight`:
    `cudaLaunchAttributeSharedMemoryMode` on grouped expert GEMMs.
    Default (default) never scales duration. FourByte / EightByte scale
    kernel time by `1000 / GpuProfile::shared_mem_*_permille` (profile
    default 1000 is identity). Legal with `--pdl` and `--cooperative`.
    Decode identity stays Default. `gpu-profile capture` is still refused.

133. [x] `cudaLaunchAttributePortableClusterSizeMode`: launch-time
    override of `cudaFuncAttributeNonPortableClusterSizeAllowed`
    (graph SetAttribute / CopyAttributes / `KernelAttrs`). Default uses
    the current function attribute. RequirePortable refuses a cluster
    larger than `portable_cluster_size` even when the function attribute
    allows it. AllowNonPortable allows up to `max_blocks_per_cluster`
    even when the function attribute is off. Default is resolved at
    launch / graph replay. Device-launch graphs allow it. Decode identity
    stays Default (function attr disallowed). `gpu-profile capture` is
    still refused.

134. [x] expertvm / Engine `--portable-cluster default|portable|non-portable`:
    `cudaLaunchAttributePortableClusterSizeMode` on grouped expert GEMMs.
    Default uses the current function attribute. RequirePortable always
    refuses a cluster larger than `portable_cluster_size`. AllowNonPortable
    allows up to `max_blocks_per_cluster` even when `--non-portable-cluster`
    is off. Legal with `--pdl` and `--cooperative`. Decode identity stays
    Default. `gpu-profile capture` is still refused.

135. [x] CUDA 13 `cudaLaunchAttributeSharedMemoryMode` (`cudaSharedMemoryMode`)
    as `PortableSharedMode`, plus `cudaLaunchKernel` `sharedMemBytes`
    (`KernelAttrs::dynamic_shared`) and
    `cudaFuncAttributeMaxDynamicSharedMemorySize`. Distinct from bank-width
    `SharedMemoryMode`. Default uses the function attribute (`0` = portable
    `max_shared_mem_per_block`). RequirePortable always refuses oversize.
    AllowNonPortable allows up to `max_shared_mem_per_block_optin`.
    Example H100 keeps portable == optin (48 KiB) so decode identity stays
    0 bytes / Default. Default is resolved at launch / graph replay.
    Device-launch graphs allow it. `gpu-profile capture` is still refused.

136. [x] expertvm / Engine `--dynamic-shared N` / `--optin-shared` /
    `--portable-shared default|portable|non-portable`:
    `cudaLaunchKernel` `sharedMemBytes`,
    `cudaFuncAttributeMaxDynamicSharedMemorySize` to the SKU opt-in max, and
    CUDA 13 `cudaLaunchAttributeSharedMemoryMode` (`PortableSharedMode`) on
    grouped expert GEMMs. `N == 0` is refused. Default uses the function
    attribute (`0` = portable only). RequirePortable always refuses oversize.
    AllowNonPortable allows up to `max_shared_mem_per_block_optin` even when
    `--optin-shared` is off. Legal with `--pdl` and `--cooperative`. Decode
    identity stays 0 bytes / Default / opt-in off. `gpu-profile capture` is
    still refused.

137. [x] `cudaLaunchAttributeNvlinkUtilCentricScheduling`: `KernelAttrs::nvlink_util_centric`
    plus stream SetAttribute / graph Get/Set/CopyAttributes. Valid `0`/`1`.
    CUDA treats it as a hint; this VM occupies every Hyper-Q slot when the
    profile has NVLink (even NVLink traffic per block). Without NVLink the
    flag is stored and occupancy is unchanged. Stream attr is inherited by
    `kernel` / `kernel_bufs`. `kernel_with` and graph replay use the launch /
    node value. Device-launch graphs allow it. Decode identity stays disabled.
    `gpu-profile capture` is still refused.

138. [x] expertvm / Engine `--nvlink-util`:
    `cudaLaunchAttributeNvlinkUtilCentricScheduling` on grouped expert GEMMs.
    Occupies every Hyper-Q slot when the profile has NVLink so leftover
    prefill cannot overlap decode even with `--compute-slots 2`. Without NVLink
    the flag is stored and occupancy is unchanged. Legal with `--pdl` and
    `--cooperative`. Decode identity stays disabled. `gpu-profile capture` is
    still refused.

139. [x] expertvm / Engine `--device-launch` / `--device-updatable`:
    grouped expert GEMM graphs instantiate with
    `cudaGraphInstantiateFlagDeviceLaunch` and launch via
    `device_launch_graph` after upload. `--device-updatable` is
    `cudaLaunchAttributeDeviceUpdatableKernelNode` so
    `--graph-set-params` keeps the exec uploaded (no host re-upload).
    Illegal with `--graph-update` and with `--graph-mem` /
    `--graph-auto-free` (mem nodes). Combo parents stay per-leaf launches.
    A non-capturing `kernel_with` with the attr is Invalid (graphs-only).
    Legal with `--pdl` and `--cooperative`. Decode identity stays host
    `launch_graph` / not device-updatable. `gpu-profile capture` is still
    refused.

140. [x] `cudaLaunchAttributePriority`: launch-time override of
    `cudaStreamCreateWithPriority` (`KernelAttrs::priority`). `None` inherits
    the stream. `Some` schedules that kernel at the given priority when
    compute contends (higher first). Capture snapshots the effective value.
    Default instantiate still uses the launch stream unless
    `cudaGraphInstantiateFlagUseNodePriority`. Device-launch graphs allow it.
    Decode identity stays inherit-stream (`None`). Distinct from
    `--stream-priority`. `gpu-profile capture` is still refused.

141. [x] expertvm / Engine `--kernel-priority N`:
    `cudaLaunchAttributePriority` on grouped expert GEMMs. `None` inherits
    `cudaStreamCreateWithPriority`. `N` (including `0`) overrides that kernel
    when compute contends (higher first). Graph instantiate uses
    `cudaGraphInstantiateFlagUseNodePriority` so captured node values are
    used at replay. Distinct from `--stream-priority`. Legal with `--pdl`
    and `--cooperative`. Decode identity stays inherit-stream.
    `gpu-profile capture` is still refused.

142. [x] Dedicated graph-memory pool: captured `cudaMallocAsync` and
    `cudaGraphAddMemAllocNode` draw from a per-device pool with release
    threshold `u64::MAX`, not the default mempool. UsedMemCurrent is live
    graph allocs. ReservedMemCurrent is live plus unused cached bytes.
    `cudaDeviceGraphMemTrim` (`graph_mem_trim`) returns unused reserved
    bytes so `cudaMemGetInfo` free grows. `destroy_graph` of a definition
    parks remaining graph mem in that pool until trim. User
    `alloc_from_pool` / `set_device_mempool` / `set_pool_release_threshold`
    / `pool_set_access` refuse the graph pool. Decode identity stays
    kernel-only graphs. Dual score still has no `$/M tokens`.
    `gpu-profile capture` is still refused.

143. [x] HTTP/1.1 keep-alive on `gguf_gemv serve` (default and `--engine`).
    Persistent connections reuse the TCP socket and the Engine / KV cache
    without a reconnect. `Connection: close` and HTTP/1.0 still close.
    Dual score still has no `$/M tokens`.

144. [x] OpenAI-shaped `POST /v1/completions` and `POST /v1/chat/completions`
    on `gguf_gemv serve` (default and `--engine`). Same greedy path as
    `/generate`. `max_tokens` aliases `n_predict`. `choices[].text` /
    `choices[].message.content` is the completion, not the prompt+decode
    string. `--engine` `"stream": true` is chunked SSE (`data:` then
    `data: [DONE]`). Native `/generate` stays `{generated, prefix_hit,
    page_hits}` / NDJSON. Dual score still has no `$/M tokens`.

145. [x] expertvm / Engine `--graph-mem-trim`: `cudaDeviceGraphMemTrim` after
    the walk / `SimulatedGpuStore::score` so unused reserved graph-mem
    returns to the OS. `hbm_peak` is unchanged (scratch still counted).
    Live graph allocs are not trimmed. Decode identity stays off.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

146. [x] `cudaPointerGetAttributes`: `Sim::pointer_get_attributes` classifies
    Unregistered / Host / Device / Managed. A freed id is Unregistered
    (CUDA 11+). A never-created id is `UnknownAlloc`. Mapped host reports a
    device pointer. Decode identity stays kernel-only graphs.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

147. [x] `cudaDeviceSetLimit` / `GetLimit`: `Sim::set_limit` / `get_limit`
    wrap `DeviceLimit`. Persisting L2 is the existing
    `set_persisting_l2_cache_size`. `MaxL2FetchGranularity` is 32, 64, or
    128 (SM 8.0+ default 128); access-policy windows must align to it.
    Stack / printf / heap / CDP limits are stored (heap does not charge
    HBM). Capture cannot include SetLimit. Decode identity stays persist
    limit 0. `gpu-profile capture` is still refused. Dual score still has
    no `$/M tokens`.

148. [x] `cudaMallocPitch` / `cudaMemcpy2DAsync`: `Sim::malloc_pitch` returns
    `(ptr, pitch)` with pitch `align_up(width, 512)`. `MemcpyOp` `height` /
    pitches copy `width * height` payload (padding is not billed). Width
    above pitch is Invalid. Decode identity stays packed 1D copies.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

149. [x] `expertvm kv --row-width W --pitch P`: miss fill is
    `cudaMemcpy2DAsync` so a padded KV page bills `W * height`, not pitch
    padding. Packed H2D (`pitch = 0`) is unchanged. `memset` fill refuses
    pitch. Decode identity stays packed 1D. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

150. [x] `cudaHostGetDevicePointer`: `Sim::host_get_device_pointer` of mapped
    host returns the same id. Unmapped host and device allocs are Invalid
    `not mapped`. Query; legal during capture. Decode identity stays
    unmapped. `gpu-profile capture` is still refused. Dual score still has
    no `$/M tokens`.

151. [x] `cudaDeviceGetAttribute`: `Sim::device_get_attribute` exposes
    cooperative launch, concurrent kernels (`compute_slots > 1`), shared
    memory, L2 / persisting L2 max, max blocks per cluster, mem-sync domain
    count, and memory-pool support. Only attributes this VM already models.
    Example H100 `compute_slots` is 1 so concurrent kernels is 0. Query;
    legal during capture. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

152. [x] `cudaMemset2DAsync`: `MemsetOp` `height` / `pitch` fill `width *
    height` payload (padding is not written). Width above pitch is Invalid.
    The mapped span is the 2D extent. Packed 1D (`height` 0/1) is unchanged.
    Decode identity stays packed 1D. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

153. [x] `expertvm kv --row-width W --pitch P` with `--fill memset` is
    `cudaMemset2DAsync` so a padded KV page bills `W * height` of HBM write,
    not pitch padding. Packed memset (`pitch = 0`) is unchanged. Decode
    identity stays packed 1D. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

154. [x] `cudaMalloc3D` / `cudaMemcpy3DAsync`: `Sim::malloc_3d` returns
    `(ptr, pitch)` with pitch `align_up(width, 512)` and charges
    `pitch * height * depth`. `MemcpyOp` `depth` / slice heights copy
    `width * height * depth` payload (row and slice padding are not billed).
    Width above pitch or height above `ysize` is Invalid. Packed 1D / 2D
    (`depth` 0/1) is unchanged. Decode identity stays packed 1D.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

155. [x] `cudaMemset3DAsync`: `MemsetOp` `depth` / `ysize` fill
    `width * height * depth` payload (row and slice padding are not written).
    Width above pitch or height above `ysize` is Invalid. Packed 1D / 2D
    (`depth` 0/1) is unchanged. Decode identity stays packed 1D.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

156. [x] `cudaGetDeviceCount` / `cudaDeviceCanAccessPeer` /
    `cudaDeviceGetP2PAttribute`: `Sim::device_count` is the profile GPU
    count. `device_can_access_peer` / `device_get_p2p_attribute` expose
    `DeviceP2pAttr::AccessSupported` (a profile device–device link). Same
    device is 0. Missing links are 0, not NoPeer. Independent of
    `enable_peer` (D2D still needs that). Query; legal during capture.
    Decode identity stays single-device. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

157. [x] `cudaGetDeviceProperties` / extra `DeviceAttr`:
    `Sim::device_get_properties` wraps existing `GpuProfile` /
    `HardwareProfile.name` fields only (HBM, shared mem, L2, copy
    engines, Hyper-Q concurrent kernels, cooperative launch, cluster
    sizes, mem-sync domains, mempools). No SM count, clock, or warp size.
    `DeviceAttr::TotalGlobalMem` / `AsyncEngineCount` are `hbm_bytes` /
    `copy_engines`. Query; legal during capture. Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

158. [x] `cudaStreamGetFlags` / `cudaStreamGetPriority`:
    `Sim::stream_get_flags` is `0` (`cudaStreamDefault` / blocking) or
    `1` (`cudaStreamNonBlocking`). `StreamId::NULL` follows
    `set_legacy_null_stream` (off → NonBlocking). `stream_get_priority`
    is the create priority (unset `0`). This VM does not cap the range.
    Query; legal during capture. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

159. [x] Serve `GET /v1/models` / `GET /health` / `GET /metrics`:
    OpenAI list/retrieve of `--model-id` (default GGUF path stem).
    Completions/chat envelopes include `"model"`. `/health` is
    `{"status":"ok"}`. Default `/metrics` is `{"engine":false}`;
    `--engine` reports live Engine counters. POST generate/completions
    stay 405 on GET. Decode identity unchanged. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

160. [x] `cudaDeviceGetMemPool` vs `cudaDeviceGetDefaultMemPool`:
    `Sim::default_pool` is the seeded default (SetMemPool does not
    replace it). `device_mempool` is the current `cudaMallocAsync` pool.
    `alloc` / mempool SetAccess / cached-byte helpers use GetMemPool.
    Query; legal during capture. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

161. [x] `cudaMemPoolGetAttribute` / `SetAttribute`:
    `Sim::pool_get_attribute` / `pool_set_attribute` wrap existing pool
    live, cached, and release threshold (`MemPoolAttr`). Used/Reserved
    are read-only. No invented ordinary-pool high-water (graph mem stays
    `GraphMemAttr`). Graph-memory pool is Invalid. Imported pools report
    the exporter. Get is a query (capture-legal); Set cannot capture.
    Decode identity unchanged. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

162. [x] `cudaFuncGetAttributes` / extra `DeviceAttr`:
    `Sim::func_get_attributes` wraps per-device
    `set_max_dynamic_shared_memory` and
    `set_non_portable_cluster_size_allowed` (`FuncAttributes`). This VM
    has one function-attr set per device, not per kernel. No
    `maxThreadsPerBlock` or register count. `DeviceAttr::CanMapHostMemory`
    / `ManagedMemory` are always 1 (mapped host and UM already exist).
    Query; legal during capture. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

163. [x] `cudaMemPoolGetAccess`:
    `Sim::pool_get_access` is ProtReadWrite (`3`) on the owning device
    (default accessibility) and on peers after `pool_set_access`.
    Otherwise ProtNone (`0`). This VM does not model ProtRead. Graph
    pool is Invalid. Query; legal during capture. Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

164. [x] `cudaMemPoolDestroy`:
    `Sim::destroy_pool` returns immediately. Unused cached bytes return
    to the OS. Outstanding allocations stay valid until freed (later
    frees do not re-cache). Destroying the current device mempool
    rebinds `device_mempool` to `default_pool`. The default and
    graph-memory pools cannot be destroyed. A destroyed handle is
    Invalid for alloc/export/get/set. Capture cannot include it. Decode
    identity unchanged. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

165. [x] `cudaEventDestroy`:
    `Sim::destroy_event` waits a recorded incomplete event like
    `synchronize_event`. A never-recorded event returns immediately.
    Unknown ids are `UnknownEvent`. Capture cannot include it. The id
    may be created again. Decode identity unchanged. `gpu-profile
    capture` is still refused. Dual score still has no `$/M tokens`.

166. [x] `cudaGraphGetNodes`:
    `Sim::graph_nodes` is node indices `0 .. graph_len` in creation
    order. Query; legal during capture (destination graph only, same as
    `graph_len`). `graph_root_nodes` / `graph_edges` /
    `graph_node_dependents` stay GetRootNodes / GetEdges /
    NodeGetDependentNodes. Decode identity unchanged. `gpu-profile
    capture` is still refused. Dual score still has no `$/M tokens`.

167. [x] `cudaStreamGetAttribute` / `cudaStreamSetAttribute`:
    `Sim::stream_get_attribute` / `stream_set_attribute` wrap existing
    stream state (`StreamAttr`: Priority, SynchronizationPolicy,
    MemSyncDomain, MemSyncDomainMap, NvlinkUtilCentric). Green-context
    SM permille is not a CUDA stream attribute. Attr/value type
    mismatch is Invalid `"stream attr"`. Get is a query (capture-legal);
    Set is host-side like the dedicated setters. Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

168. [x] Serve `POST /tokenize` / `POST /detokenize`:
    Tokenize takes the same `prompt` / `messages` body as generate and
    returns `{"tokens":[...],"count":N}` of the ids generate would
    prefill. Detokenize takes `{"tokens":[...]}` and returns
    `{"text":"..."}`. GET is 405. Empty prompt / empty tokens are 400.
    `--engine` returns bytes without admitting a sequence. Decode
    identity unchanged. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

169. [x] `cudaGraphChildGraphNodeGetGraph`:
    `Sim::graph_child_get_graph` is the nested graph id of one child-graph
    node. Instantiated ids use the exec snapshot (same as
    `graph_child_nodes`). Not-a-child is Invalid. Query; legal during
    capture. Decode identity unchanged. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

170. [x] `cudaGraphEventRecordNodeGetEvent` / `WaitNodeGetEvent`:
    `Sim::graph_event_record_get_event` / `graph_event_wait_get_event`
    wrap the stored event id. Wrong node kind is Invalid. Query; legal
    during capture. Decode identity unchanged. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

171. [x] `cudaGraphMemAllocNodeGetParams`:
    `Sim::graph_alloc_get_params` returns stored `(AllocId, bytes)`. Pool
    identity stays the graph-memory pool. Wrong node kind is Invalid.
    Query; legal during capture. Decode identity unchanged. `gpu-profile
    capture` is still refused. Dual score still has no `$/M tokens`.

172. [x] `cudaGraphMemFreeNodeGetParams`:
    `Sim::graph_free_get_params` returns the stored `AllocId`. Instantiated
    ids use the exec snapshot (same as `graph_alloc_get_params`). Wrong
    node kind is Invalid `"not a mem free node"`. Query; legal during
    capture. `graph_allocs` stays alloc-node ids. Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

173. [x] `cudaGraphMemFreeNodeSetParams` / `cudaGraphExecMemFreeNodeSetParams`:
    `Sim::graph_free_set_params` mutates the definition (`g.steps`, 1 ns
    host-sync) and does not retarget an already-instantiated exec.
    `graph_exec_free_set_params` mutates the exec snapshot
    (`graph_set_params_ns`, clears upload). Node must already be
    `Kind::Free`. Unknown ids are `UnknownAlloc`. Capture refused.
    `update_graph` of mem nodes stays Invalid. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no `$/M
    tokens`.

174. [x] Serve OpenAI `echo` on `POST /v1/completions`:
    `GenReq.echo` (default false) selects `GreedyParts.full` vs
    `completion` for `choices[].text`. Chat ignores `echo`. `--engine`
    non-stream Completions decode prompt+generated when echo is set;
    streaming Completions emit the prompt as the first SSE delta.
    Decode identity unchanged. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

175. [x] Serve tokenize `add_special_tokens`:
    Default true is `prompt_ids` (BOS when `add_bos`). False is
    `Tokenizer::encode` without extra BOS. Do not invent `max_model_len`
    or `/v1/tokenize`. Decode identity unchanged. `gpu-profile capture`
    is still refused. Dual score still has no `$/M tokens`.

176. [x] `cudaGraphDebugDotPrint`:
    `Sim::graph_debug_dot` prints stored node kinds and edges as DOT.
    Query; legal during capture (destination graph only, same as
    `graph_len`). No verbose kernel-param flags. Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

177. [x] Serve OpenAI `n` must be 1:
    `parse_gen_req` accepts `"n":1` and rejects any other `n` (`n must be
    1`). Omitted `n` stays one greedy completion. Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

178. [x] `cudaGraphEventRecordNodeSetEvent` / `cudaGraphEventWaitNodeSetEvent`
    on a stored graph definition (`Sim::graph_event_record_set_event` /
    `graph_event_wait_set_event`). Mutates `g.steps`; **does not retarget
    instantiated exec**. Capture refused (`"cannot capture event record set
    event"` / `"cannot capture event wait set event"`). `UnknownEvent` /
    `"not an event record node"` / `"not an event wait node"`. The node's
    External flag is **not** rewritten — that is topology
    (`graph_add_event_record` / `graph_add_event_wait`). Host-sync 1 ns.
    Exec-side `graph_exec_event_*_set_event` already exists. Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score still has
    no `$/M tokens`.

179. [x] `cudaGraphChildGraphNodeSetParams` on a stored graph definition
    (`Sim::graph_child_set_params`). Nested topology **may** change (unlike
    exec-side `graph_exec_child_set_params`, which requires
    `child_param_topology_eq`). Capture refused `"cannot capture child graph
    node set params"`. Child must be instantiated, same GPU, not self, not
    cyclic. Stores the child **id as passed** (same as `graph_add_child`).
    Host-sync 1 ns; does not retarget exec. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

180. [x] `cudaGraphAddNode`: `Sim::graph_add_node` takes [`GraphNodeParams`]
    plus dependency indices in the same call (typed `graph_add_*` stay and
    still start with no deps). IF/WHILE/SWITCH stay `graph_add_if` /
    `graph_add_while` / `graph_add_switch` (those return body graphs).
    [`GraphNodeParams::Alloc`] fills [`GraphAddNode::alloc`]. Capture refused;
    illegal on an instantiated exec. Unknown dep index is `"graph
    dependency"`. Decode identity unchanged. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

181. [x] `cudaGraphNodeSetParams` / `cudaGraphExecNodeSetParams`:
    `Sim::graph_node_set_params` / `graph_exec_node_set_params` dispatch
    [`GraphNodeParams`] onto the typed SetParams. [`GraphNodeParams::Alloc`]
    is Invalid (`cannot set mem alloc node params`; would resize HBM).
    [`GraphNodeParams::Empty`] is Invalid (`empty node has no params`).
    Definition does not retarget exec. Capture refused. Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score still has
    no `$/M tokens`.

182. [x] Bulk `cudaGraphAddDependencies` / `cudaGraphRemoveDependencies`:
    `Sim::graph_add_dependencies_n` / `graph_remove_dependencies_n` take
    `numDependencies` from/to pairs. All-or-nothing (a cycle or out-of-range
    index changes nothing). Pairwise helpers call the bulk APIs. Duplicate
    add is a no-op; missing remove is a no-op. Empty slice is success.
    Capture refused; illegal on an instantiated exec. Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score still has
    no `$/M tokens`.

183. [x] `cudaGraphNodeGetParams`: `Sim::graph_node_get_params` /
    `graph_exec_node_get_params` return [`GraphNodeParams`] from the
    definition / exec snapshot. Query; legal during capture on the
    definition. Empty returns [`GraphNodeParams::Empty`]. Alloc is bytes
    only (pointer stays `graph_alloc_get_params`). IF/WHILE/SWITCH are
    Invalid `"not a graph node params kind"`. Uninstantiated exec GetParams
    is Invalid. Decode identity unchanged. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

184. [x] `cudaStreamGetCaptureInfo_v2` dependencies:
    [`StreamCaptureInfo::dependencies`] is the last same-stream captured
    node (destination-graph index) union extra `pending_deps`. Empty until
    a node is captured or `stream_update_capture_dependencies`. Query;
    `pending_deps` stays extras-only. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

185. [x] `cudaStreamGetCaptureInfo` `id_out`:
    [`StreamCaptureInfo::id`] is unique per `begin_capture` /
    `begin_capture_to_graph` sequence (starts at 1). Forked streams in
    the same session share it. Stable after capturing nodes. Query; not a
    graph id. Decode identity unchanged. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

186. [x] `cudaStreamGetId`: `Sim::stream_get_id` is unique per
    `(device, stream)` (not the caller-chosen [`StreamId`], not a capture
    sequence id). Query; legal during capture. Unknown devices are
    Invalid. This VM does not invent `cudaStreamDestroy`. Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

187. [x] `cudaGraphKernelNodeGetAttribute` / `SetAttribute`:
    `Sim::graph_kernel_node_get_attribute` /
    `graph_exec_kernel_node_get_attribute` /
    `graph_kernel_node_set_attribute` /
    `graph_exec_kernel_node_set_attribute` dispatch [`KernelNodeAttr`] onto
    the typed kernel-node getters/setters. Definition Set does not
    retarget exec. Get is a query (capture-legal); Set cannot include
    capture. Attr/value mismatch is Invalid `"kernel node attr"`. Typed
    helpers stay. Decode identity unchanged. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

188. [x] `cudaEventRecordWithFlags` / `cudaStreamWaitEvent` flags /
    `cudaEventCreateWithFlags`: `Sim::record_event_with_flags` /
    `wait_event_with_flags` / `create_event_with_flags` take the CUDA flag
    words. Known bits are [`EventRecordFlags::EXTERNAL`],
    [`EventWaitFlags::EXTERNAL`], [`EventCreateFlags::DISABLE_TIMING`].
    Unknown bits (including BlockingSync / Interprocess) are Invalid.
    Typed helpers stay. Decode identity unchanged. `gpu-profile capture`
    is still refused. Dual score still has no `$/M tokens`.

189. [x] `cudaGraphDebugDotPrint` flags: `Sim::graph_debug_dot_with_flags`
    takes [`GraphDebugDotFlags`] (CUDA bit values). Flags `0` stays kinds
    and edges. `VERBOSE` dumps modeled params (kernel coop/buffers,
    memcpy/memset/host/event/alloc/free/batch/conditional, kernel
    priority, graph handle). External-semaphore and extra-conditional-edge
    bits are Invalid. Query; legal during capture. Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

190. [x] `cudaGraphAddMemcpyNode1D` / `MemcpyNodeSetParams1D` /
    `ExecMemcpyNodeSetParams1D`: `Sim::graph_add_memcpy_1d` /
    `graph_memcpy_set_params_1d` / `graph_exec_memcpy_set_params_1d`
    pack [`MemcpyOp::packed_1d`] (height/depth/pitches `0`). SetParams1D
    may convert a 2D/3D node to 1D. Pageable copies stay illegal.
    Definition Set does not retarget exec. Capture cannot include Add or
    Set. Typed `graph_add_memcpy` / SetParams stay. Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

191. [x] `cudaGraphDestroyNode`: `Sim::graph_destroy_node` drops a definition
    node and incident edges. Remaining indices stay valid. Capture cannot
    include it. Illegal on an instantiated exec. Definition destroy does
    not retarget exec. Destroying a mem alloc node unlinks
    [`graph_mem_allocs`]. Child-graph objects are not destroyed. Decode
    identity unchanged. `gpu-profile capture` is still refused. Dual score
    still has no `$/M tokens`.

192. [x] `cudaHostGetFlags`: `Sim::host_get_flags` returns
    [`HostAllocFlags::MAPPED`] for `cudaHostAllocMapped` /
    `cudaHostRegisterMapped`, else `0` for pinned or registered host.
    Device, managed, VMM, and unregistered pageable pointers are Invalid
    `"not host alloc"`. Query; legal during capture. Portable /
    WriteCombined are not modeled. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

193. [x] `cudaHostAlloc` / `cudaHostRegister` flags: `Sim::alloc_host_with_flags`
    / `host_register_with_flags` take [`HostAllocFlags`]. Known bit is
    [`HostAllocFlags::MAPPED`]. Portable / WriteCombined / IoMemory /
    ReadOnly are Invalid. Typed helpers stay. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

194. [x] `cudaIpcGetEventHandle` / `cudaIpcOpenEventHandle`:
    `Sim::ipc_get_event` / `ipc_open_event` export an
    [`EventCreateFlags::INTERPROCESS`] event and import an alias that
    shares the source record. Interprocess requires DisableTiming
    (Invalid `"interprocess timing"` otherwise). `cudaEventBlockingSync`
    stays Invalid. Same event returns the same handle. An import cannot
    export. Destroy of the source while imports are live is Invalid
    `"ipc mapped"`. Destroy of an import does not destroy the source.
    Capture cannot include event IPC. Typed helper:
    `create_event_interprocess`. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

195. [x] `cudaStreamCreateWithFlags` / `cudaStreamCreateWithPriority`:
    `Sim::stream_create_with_flags` / `stream_create_with_priority` take
    [`StreamCreateFlags`]. Known bit is [`StreamCreateFlags::NON_BLOCKING`].
    Unknown bits are Invalid `"stream create flags"`. NULL is Invalid
    (use `set_legacy_null_stream`). Capture cannot include them. Typed
    `set_stream_blocking` / `set_stream_priority` stay. This VM does not
    cap the priority range. Decode identity unchanged. `gpu-profile
    capture` is still refused. Dual score still has no `$/M tokens`.

196. [x] `cudaFuncSetAttribute` / `cudaFuncGetAttribute`:
    `Sim::func_set_attribute` / `func_get_attribute` dispatch [`FuncAttr`]
    onto the typed per-device setters/getters. Negative max-dynamic-shared
    or a non-0/1 non-portable-cluster value is Invalid `"func attr"`. Get
    is a query (capture-legal); Set is host-side like the typed helpers.
    Typed helpers stay. Decode identity stays `0` / disallowed.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

197. [x] `cudaMemRangeGetAttribute`: `Sim::mem_range_get_attribute` returns
    modeled per-alloc managed advice ([`MemRangeAttr`] ReadMostly /
    PreferredLocation / AccessedBy). This VM does not track per-byte ranges
    or last-prefetch location. Non-managed pointers are Invalid
    `"not managed"`. Query; legal during capture. Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

198. [x] `cudaDeviceGetAttribute` / `cudaGetDeviceProperties` modeled
    caps: [`DeviceAttr::ClusterLaunch`] (`max_blocks_per_cluster > 0`),
    [`DeviceAttr::HostRegisterSupported`],
    [`DeviceAttr::IpcEventSupport`],
    [`DeviceAttr::CanUseHostPointerForRegisteredMem`] (always 1), and
    [`DeviceAttr::MemoryPoolSupportedHandleTypes`]
    ([`MemHandleType::POSIX_FILE_DESCRIPTOR`]). Query; legal during
    capture. No SM count or clock. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

199. [x] `cudaDevP2PAttrPerformanceRank`: `Sim::device_get_p2p_attribute`
    returns unique GPU↔GPU link `bps` descending (lower is better). Same
    device or no link is 0. Derived from existing profile links; do not
    invent `NativeAtomic` or CUDA array P2P. Query; legal during capture.
    Decode identity unchanged. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

200. [x] `cudaDeviceEnablePeerAccess` flags: `Sim::enable_peer_with_flags`
    requires [`PeerAccessFlags::DEFAULT`] (`0`). Unknown bits are Invalid
    `"peer access flags"`. Typed [`enable_peer`] stays. Capture is legal
    (same as the typed helper). Decode identity unchanged. `gpu-profile
    capture` is still refused. Dual score still has no `$/M tokens`.

201. [x] `cudaMemcpyPeer` / `cudaMemcpyPeerAsync`: `Sim::memcpy_peer` /
    `memcpy_peer_async` are the host-synchronous / stream-ordered peer
    replica copies. Typed [`memcpy_device_to_device`] stays. Peer capture
    is refused (`"cannot capture host-sync memcpy"`); Async records a
    memcpy node. Decode identity unchanged. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

202. [x] `cudaMemRangeGetAttributes`: `Sim::mem_range_get_attributes`
    batches [`MemRangeAttr`] queries (all-or-nothing; empty is `[]`). Same
    per-alloc rules as GetAttribute. Last-prefetch is not modeled. Query;
    legal during capture. Decode identity unchanged. `gpu-profile capture`
    is still refused. Dual score still has no `$/M tokens`.

203. [x] `cudaDevAttrGPUDirectRDMASupported`: [`DeviceAttr::GpuDirectRdmaSupported`]
    is 1 iff the profile has a GPU↔GPU [`LinkKind::Rdma`] incident on that
    device. Flush and write-ordering attrs are not modeled. Query; legal
    during capture. Decode identity unchanged. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

204. [x] `cudaMemcpy3DPeer` / `cudaMemcpy3DPeerAsync`: `Sim::memcpy_peer_3d`
    / `memcpy_peer_3d_async` are the host-synchronous / stream-ordered 3D
    peer replica copies ([`MemcpyOp`] height/depth; payload not padding).
    Places are forced to `src`/`dst`. Typed [`memcpy`] stays. Peer capture
    is refused; Async records a memcpy node. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

205. [x] `cudaMemcpy2DPeer` / `cudaMemcpy2DPeerAsync`: `Sim::memcpy_peer_2d`
    / `memcpy_peer_2d_async` are the host-synchronous / stream-ordered 2D
    peer replica copies ([`MemcpyOp`] height/pitches; payload not padding).
    Places are forced to `src`/`dst`. Typed [`memcpy`] stays. Peer capture
    is refused; Async records a memcpy node. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

206. [x] `cudaDevAttrHostRegisterReadOnlySupported` /
    `cudaDevAttrPageableMemoryAccess`: both 0. ReadOnly host register is
    Invalid; pageable H2D/D2H is bounce-buffer (not coherent). Do not
    report `PageableMemoryAccess=1`. Query; legal during capture. Decode
    identity unchanged. `gpu-profile capture` is still refused. Dual score
    still has no `$/M tokens`.

207. [x] `cudaHostGetDevicePointer` flags: `Sim::host_get_device_pointer_with_flags`
    requires [`HostGetDevicePointerFlags::DEFAULT`] (`0`). Unknown bits are
    Invalid `"host get device pointer flags"`. Typed helper stays. Query;
    legal during capture. Decode identity unchanged. `gpu-profile capture`
    is still refused. Dual score still has no `$/M tokens`.

208. [x] `cudaMallocManaged` flags: `Sim::alloc_managed_with_flags` takes
    [`MemAttachFlags::GLOBAL`] / `HOST`. Single and other bits are Invalid
    `"managed flags"`. Typed [`alloc_managed`] / [`alloc_managed_host`] stay.
    Capture refused. Decode identity unchanged. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

209. [x] `cudaIpcOpenMemHandle` flags: `Sim::ipc_open_with_flags` accepts
    [`IpcMemFlags::LAZY_ENABLE_PEER_ACCESS`] as a no-op (dest must already
    hold the source). Cross-GPU lazy peer is not modeled. Other bits are
    Invalid `"ipc open flags"`. Typed [`ipc_open`] stays. Capture refused.
    Decode identity unchanged. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

210. [x] `cudaStreamAttachMemAsync` flags: `Sim::stream_attach_with_flags`
    maps [`MemAttachFlags::GLOBAL`] / `HOST` / `SINGLE` onto
    [`MemAttach`] then typed [`stream_attach`] (capture refuse, Single+NULL,
    and stream-match stay). Other bits are Invalid `"stream attach flags"`.
    Typed [`stream_attach`] stays. Query; capture is refused by the typed
    helper. Decode identity unchanged. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

211. [x] `cudaMemset` / `cudaMemset2D` / `cudaMemset3D`: `Sim::memset_sync`
    / `memset_op_sync` enqueue then wait the stream (host-synchronous).
    Capture is refused (`"cannot capture host-sync memset"`). Typed
    [`memset`] / [`memset_op`] stay `cudaMemsetAsync` / `2DAsync` / `3DAsync`.
    Decode identity stays Async. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

212. [x] `cudaDevAttrStreamPrioritiesSupported` /
    `cudaDevAttrGpuOverlap` / `cudaDevAttrUnifiedAddressing`:
    StreamPrioritiesSupported and UnifiedAddressing are always 1 (this VM
    has [`set_stream_priority`] and one pointer space). GpuOverlap is
    [`GpuProfile::copy_engines`] `> 0`. Query; legal during capture.
    Decode identity unchanged. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

213. [x] `cudaMemPoolCreate` props: `Sim::create_pool_with_props` takes
    [`MemPoolProps`]. [`MemAllocationType::PINNED`] only. Handle types
    [`MemHandleType::NONE`] / `POSIX_FILE_DESCRIPTOR` map to typed
    [`create_pool`] / [`create_shareable_pool`]. Other bits Invalid
    `"pool handle types"`. Location must be [`Place::Device`].
    [`MemPoolProps::max_size`] `0` is unlimited; otherwise
    [`alloc_from_pool`] OOMs when reserved would exceed it. Typed helpers
    stay. Capture refused. Decode identity unchanged. `gpu-profile capture`
    is still refused. Dual score still has no `$/M tokens`.

214. [x] `cudaMemPrefetchAsync` flags: `Sim::prefetch_with_flags` requires
    [`PrefetchFlags::DEFAULT`] (`0`). Other bits are Invalid `"prefetch flags"`.
    [`Place::Device`] is typed [`prefetch`]; host places are
    [`prefetch_host`]. Typed helpers stay. Capture may record the memcpy.
    Decode identity unchanged. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

215. [x] `cudaDevP2PAttrNativeAtomicSupported` /
    `cudaDevP2PAttrCudaArrayAccessFromDevice`: always 0. Native atomics and
    CUDA-array P2P are not modeled. Query; legal during capture. Decode
    identity unchanged. `gpu-profile capture` is still refused. Dual score
    still has no `$/M tokens`.

216. [x] `cudaMemPoolAttr` reuse flags:
    [`ReuseFollowEventDependencies`] / [`ReuseAllowOpportunistic`] /
    [`ReuseAllowInternalDependencies`]. Default 1. Get/Set via
    [`pool_get_attribute`] / [`pool_set_attribute`] (0 or 1; other values
    Invalid `"pool reuse attr"`). Graph pool Invalid. Imported pools use
    the exporter. Set capture-refused; Get is legal during capture. Only
    `ReuseAllowOpportunistic=0` is mechanical: [`pool_acquire`] skips cache
    reuse (OS alloc; unused cached bytes stay reserved). FollowEvent and
    Internal are stored but do not insert event waits or extra sync.
    Decode identity unchanged. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

217. [x] `cudaDevAttrConcurrentManagedAccess` /
    `cudaDevAttrDirectManagedMemAccessFromHost` /
    `cudaDevAttrPageableMemoryAccessUsesHostPageTables`: always 0.
    `cudaDevAttrCanFlushRemoteWrites` is
    [`gpu_direct_rdma_supported`] (same as GpuDirectRdmaSupported). Query;
    legal during capture. Decode identity unchanged. `gpu-profile capture`
    is still refused. Dual score still has no `$/M tokens`.

218. [x] `cudaDeviceFlushGPUDirectRDMAWrites`:
    `Sim::flush_gpu_direct_rdma_writes`. Target
    [`FlushGpuDirectRdmaTarget::CURRENT_DEVICE`]; scope
    [`FlushGpuDirectRdmaScope::TO_OWNER`] / `TO_ALL_DEVICES`. Unsupported
    devices Invalid `"gpu direct rdma"`. Bad target/scope Invalid
    `"flush rdma target"` / `"flush rdma scope"`. Capture refused
    `"cannot capture flush rdma"`. 1 ns host-sync barrier; no
    write-visibility model. Write-ordering options are not modeled.
    Decode identity unchanged. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

219. [x] `cudaMemPoolSetAccess` flags: `Sim::pool_set_access_with_flags`
    maps [`MemAccessFlags::PROT_READ_WRITE`] onto typed
    [`pool_set_access`] and [`PROT_NONE`](MemAccessFlags::PROT_NONE) onto
    [`pool_unset_access`]. [`PROT_READ`](MemAccessFlags::PROT_READ) is
    Invalid `"pool prot read"` (pool ProtRead is not modeled). Other bits
    Invalid `"pool access flags"`. Typed helpers stay. Capture refused by
    those helpers. Decode identity unchanged. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

220. [x] `cudaDevAttrHostNativeAtomicSupported` /
    `cudaDevAttrCooperativeMultiDeviceLaunch` /
    `cudaDevAttrIntegrated`: always 0. Host-mapped atomics and
    multi-device cooperative are not modeled; example SKUs are discrete.
    Query; legal during capture. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

221. [x] `cuMemSetAccess` flags: `Sim::va_set_access_with_flags` maps
    [`MemAccessFlags::PROT_READ`] onto typed [`va_set_access`],
    [`PROT_READ_WRITE`](MemAccessFlags::PROT_READ_WRITE) onto
    [`va_set_access_write`], and [`PROT_NONE`](MemAccessFlags::PROT_NONE)
    onto [`va_unset_access`]. Other bits Invalid `"va access flags"`.
    Typed helpers stay. Capture refused by those helpers. Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

222. [x] `cudaMemGetAddressRange`: `Sim::mem_get_address_range` returns
    `(alloc, bytes)` for a live id. Interior offsets are not modeled.
    Never-created is [`UnknownAlloc`]. Freed is Invalid `"address range"`.
    Query; legal during capture. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

223. [x] `cudaMemAdvise_v2` location: `Sim::mem_advise_with_location`
    takes a [`Place`]. [`SetReadMostly`] / [`UnsetReadMostly`] /
    [`UnsetPreferredLocation`] / [`SetPreferredLocationHost`] ignore
    location. [`SetPreferredLocation`] with [`Place::Device`] is typed
    [`mem_advise`]; host places are [`SetPreferredLocationHost`].
    [`SetAccessedBy`] / [`UnsetAccessedBy`] require [`Place::Device`]
    (host is Invalid `"advise location"`). Typed [`mem_advise`] stays.
    Capture refused by that helper. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

224. [x] `cuPointerSetAttribute` SyncMemops: `Sim::pointer_set_attribute`
    / [`pointer_get_attribute`] for [`PointerAttr::SyncMemops`]. Value
    0 or 1 (other Invalid `"pointer attr"`). Default false. Runtime
    [`memcpy`] / [`memset_op`] wait the stream like pageable; capture
    of those copies is `"cannot capture sync memops memcpy"` /
    `"cannot capture sync memops memset"` (pageable still wins if
    both). Explicit graph-add / graph launch is not refused. Set is
    capture-refused `"cannot capture pointer attr"`; Get is a query.
    Unknown ids are [`UnknownAlloc`]; freed is Invalid `"pointer attr"`.
    Decode identity unchanged (default false; D2D stays async).
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

225. [x] `cudaDevAttrSparseCudaArraySupported` /
    `cudaDevAttrDeferredMappingCudaArraySupported` /
    `cudaDevAttrDmaBufSupported`: always 0. CUDA arrays and dma-buf
    are not modeled. Query; legal during capture. Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score
    still has no `$/M tokens`.

226. [x] `cudaDeviceSetSharedMemConfig` / `GetSharedMemConfig`:
    `Sim::set_shared_mem_config` / [`get_shared_mem_config`].
    [`SharedMemoryMode::Default`] kernels inherit the device config at
    duration time. Unset device config is unscaled (decode identity).
    Launch FourByte / EightByte still override. Set is host-sync
    (1 ns); capture refused. Get is a query. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

227. [x] `cudaEventBlockingSync`: [`EventCreateFlags::BLOCKING_SYNC`] (`1`).
    [`create_event_with_flags`] accepts it. [`synchronize_event`] pays
    [`host_sync_blocking_ns`] instead of the recording stream's policy.
    [`create_event_blocking_sync`] is the typed helper. Interprocess still
    requires DisableTiming. Other unknown bits stay Invalid
    `"event create flags"`. Decode identity unchanged (example profiles
    tax 0). `gpu-profile capture` is still refused. Dual score still has
    no `$/M tokens`.

228. [x] `cudaSetDeviceFlags` / `GetDeviceFlags`: `Sim::set_device_flags` /
    [`get_device_flags`]. Schedule mask only
    ([`DeviceFlags::SCHEDULE_AUTO`] / `SPIN` / `YIELD` /
    `BLOCKING_SYNC`). Combined schedule bits Invalid `"device schedule"`.
    MapHost / LmemResizeToMax / SyncMemops Invalid `"device flags"`.
    Auto streams inherit the schedule as host-wait tax; explicit stream
    policy wins. Default `0` is identity. SetOnActiveProcess is not
    modeled. Set is host-sync (1 ns); capture refused. Get is a query.
    Decode identity unchanged. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

229. [x] `cudaHostAllocPortable` / `cudaHostAllocWriteCombined`:
    [`HostAllocFlags::PORTABLE`] / [`WRITE_COMBINED`](HostAllocFlags::WRITE_COMBINED)
    are stored on [`alloc_host_with_flags`] (no DMA/pin change).
    [`host_register_with_flags`] accepts Portable (not WriteCombined;
    register IoMemory stays Invalid). [`host_get_flags`] returns the
    stored word. IoMemory / ReadOnly stay Invalid. Typed helpers stay.
    Decode identity unchanged. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

230. [x] `cudaFuncSetSharedMemConfig` / `GetSharedMemConfig`:
    [`set_func_shared_mem_config`] / [`get_func_shared_mem_config`].
    Per device (this VM is not per kernel-function object). Launch Default
    inherits the function config, then the device
    [`set_shared_mem_config`]. Launch FourByte / EightByte still override.
    Set is host-sync (1 ns); capture refused. Get is a query. Decode
    identity unchanged. `gpu-profile capture` is still refused. Dual score
    still has no `$/M tokens`.

231. [x] Remaining `cudaSetDeviceFlags` stored bits:
    [`DeviceFlags::MAP_HOST`] (`cudaDeviceMapHost`) /
    [`DeviceFlags::LMEM_RESIZE_TO_MAX`] (`cudaDeviceLmemResizeToMax`).
    Schedule bits stay exclusive via `flags & SCHEDULE_MASK`. MapHost /
    Lmem are stored (CanMapHostMemory is already 1; local-memory resize
    is not modeled). Unknown bits Invalid `"device flags"`. Decode
    identity unchanged (default `0`). `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

232. [x] `cudaDeviceSyncMemops` ([`DeviceFlags::SYNC_MEMOPS`] `0x80`):
    runtime memcpy/memset wait the stream like pointer
    [`PointerAttr::SyncMemops`]. Capture of those copies is refused
    (pageable still wins if both). Graph-add / graph launch are not
    refused. Decode identity unchanged (default flags `0`).
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

233. [x] `cudaFuncAttributePreferredSharedMemoryCarveout`:
    [`set_func_carveout`] / [`get_func_carveout`] and
    [`FuncAttr::PreferredSharedMemoryCarveout`]. Launch Default inherits
    the function carveout occupancy. Launch MaxL1 / MaxShared still
    override. CUDA ints `-1`/`0`/`100` only (other percentages Invalid
    `"func attr"`). Capture-legal like other function attributes.
    [`func_get_attributes`] reports `preferredShmemCarveout`. Decode
    identity stays Default. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

234. [x] Remaining `cuPointerGetAttribute` queries:
    [`PointerAttr::MemoryType`] / [`DevicePointer`](PointerAttr::DevicePointer) /
    [`HostPointer`](PointerAttr::HostPointer) / [`IsManaged`](PointerAttr::IsManaged) /
    [`RangeSize`](PointerAttr::RangeSize) / [`Mapped`](PointerAttr::Mapped) /
    [`MemPoolHandle`](PointerAttr::MemPoolHandle) wrap existing
    [`pointer_get_attributes`], range size, mapped host, and the backing
    pool. Set stays [`PointerAttr::SyncMemops`] only (other attrs Invalid
    `"pointer attr"`). Get is a query; capture-legal. CONTEXT / P2P tokens /
    DeviceOrdinal / RANGE_START_ADDR stay unmodeled. Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

235. [x] `cudaFuncAttributeClusterDimMustBeSet` /
    `RequiredClusterWidth` / `Height` / `Depth`:
    [`set_cluster_dim_must_be_set`] / [`set_required_cluster_width`]
    (height/depth). A kernel without cluster is Invalid `"cluster dim must
    be set"` or `"required cluster"`. A nonzero required axis must match
    the launch. `0` clears a required axis. Over-SKU sizes Invalid
    `"cluster size"` at Set. Capture-legal. Decode identity stays unset.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

236. [x] `cudaFuncAttributeClusterSchedulingPolicyPreference`:
    [`set_func_cluster_policy`] / [`get_func_cluster_policy`]. Per device.
    Launch [`ClusterSchedulingPolicy::Default`] inherits this occupancy;
    launch Spread / LoadBalancing still override. CUDA ints `0`/`1`/`2`
    (other values Invalid `"func attr"`). Capture-legal like other
    function attributes. [`func_get_attributes`] reports
    `clusterSchedulingPolicyPreference`. Decode identity stays Default.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

237. [x] Remaining `cuPointerGetAttribute` identity queries:
    [`PointerAttr::DeviceOrdinal`] / [`RangeStartAddr`](PointerAttr::RangeStartAddr) /
    [`BufferId`](PointerAttr::BufferId) wrap the owning GPU, [`mem_get_address_range`]
    base (alloc id; interior offsets are not modeled), and [`AllocId`].
    Unmapped host DeviceOrdinal is Invalid `"pointer attr"`. Set stays
    [`PointerAttr::SyncMemops`] only. Get is a query; capture-legal.
    CONTEXT / P2P tokens stay unmodeled. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

238. [x] Remaining `cuPointerGetAttribute` capability queries:
    [`PointerAttr::IsLegacyCudaIpcCapable`] / [`IsGpuDirectRdmaCapable`](PointerAttr::IsGpuDirectRdmaCapable) /
    [`AllowedHandleTypes`](PointerAttr::AllowedHandleTypes) wrap [`ipc_get`]
    eligibility, [`gpu_direct_rdma_supported`] on that GPU, and POSIX-FD
    on shareable-pool allocs. Set stays [`PointerAttr::SyncMemops`] only.
    Get is a query; capture-legal. ACCESS_FLAGS / CONTEXT / P2P tokens
    stay unmodeled. Decode identity unchanged. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

239. [x] `cudaMemRangeAttributeLastPrefetchLocation`:
    [`MemRangeAttr::LastPrefetchLocation`] is the dest of [`prefetch`] /
    [`prefetch_host`] (`None` never prefetched / `cudaInvalidDeviceId`;
    host is `cudaCpuDeviceId`). Recorded at submit, including already-local
    no-ops. Query; capture-legal. Decode identity stays unset. `gpu-profile
    capture` is still refused. Dual score still has no `$/M tokens`.

240. [x] `cudaMemRangeAttributePreferredLocationType` / `Id` /
    `LastPrefetchLocationType` / `Id`:
    [`MemLocationType`] wraps the stored [`Place`] (`0` Invalid / `1`
    Device / `2` Host). Id is the device ordinal when the type is Device,
    else `0` (ignored). Host NUMA is not modeled. Query; capture-legal.
    Decode identity stays Invalid / `0`. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

241. [x] `CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR` / `MAPPING_SIZE`:
    [`PointerAttr::MappingBaseAddr`] / [`MappingSize`](PointerAttr::MappingSize)
    wrap the mapping that covers VA offset 0. Non-VMM is the same as
    RangeStartAddr / RangeSize. VMM is the `cuMemMap` span at offset 0,
    not the reserved VA. Unmapped `va_reserve` and maps that skip offset
    0 are Invalid `"pointer attr"`. Interior offsets are not modeled.
    Set stays SyncMemops only. Get is a query; capture-legal.
    ACCESS_FLAGS / CONTEXT / P2P tokens stay unmodeled. Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

242. [x] `cuMemGetAccess`:
    [`Sim::va_get_access`] is the flags-word twin of [`pool_get_access`]
    for VMM. Local map and [`va_set_access_write`] are ProtReadWrite
    (`3`); [`va_set_access`] is ProtRead (`1`); else ProtNone (`0`).
    Unmapped `va_reserve` is Invalid `"not mapped"`. Non-VMM is Invalid
    `"not a VA"`. Query; capture-legal. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

243. [x] `cuMemGetAllocationPropertiesFromHandle`:
    [`Sim::va_get_allocation_properties`] returns [`MemAllocationProp`]
    (pinned device location of the handle; handle types always none;
    `gpuDirectRDMACapable` wraps the SKU). Compression / usage flags are
    not modeled. Query; capture-legal. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

244. [x] `cuMemGetAllocationGranularity`:
    [`Sim::va_get_allocation_granularity`] returns the profile
    [`va_granularity_bytes`](HardwareProfile::va_granularity_bytes)
    (`0`/`1` → `1`). Minimum and recommended flags are the same value
    (this VM has one granularity). Other flags Invalid
    `"granularity flags"`. Prop must be pinned device in the profile.
    Query; capture-legal. Decode identity unchanged. `gpu-profile
    capture` is still refused. Dual score still has no `$/M tokens`.

245. [x] `cuMulticastGetGranularity`:
    [`Sim::multicast_get_granularity`] returns the profile
    [`multicast_granularity_bytes`](HardwareProfile::multicast_granularity_bytes)
    (`0`/`1` → `1`). Minimum and recommended flags are the same value.
    Other flags Invalid `"multicast granularity flags"`. Query;
    capture-legal. Decode identity unchanged. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

246. [x] `cudaDevAttrMulticastSupported`:
    [`DeviceAttr::MulticastSupported`] is a GPU↔GPU NVLink on that
    device ([`HardwareProfile::multicast_supported`]). PCIe P2P and RDMA
    are not NVLS. Also on [`DeviceProperties`]. Query; capture-legal.
    Decode identity unchanged. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

247. [x] `cudaDevAttrVirtualMemoryManagementSupported`:
    [`DeviceAttr::VirtualMemoryManagementSupported`] is always 1 (this
    VM has [`va_reserve`](Sim::va_reserve)). Also on [`DeviceProperties`].
    Query; capture-legal. Decode identity unchanged. `gpu-profile
    capture` is still refused. Dual score still has no `$/M tokens`.

248. [x] `cudaDevAttrHandleTypePosixFileDescriptorSupported`:
    [`DeviceAttr::HandleTypePosixFileDescriptorSupported`] is always 1
    (this VM has [`create_shareable_pool`](Sim::create_shareable_pool)).
    Also on [`DeviceProperties`]. Query; capture-legal. Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

249. [x] `cudaDevAttrGPUDirectRDMAFlushWritesOptions` /
    `GPUDirectRDMAWithCudaVMMSupported` /
    `GenericCompressionSupported`: FlushWritesOptions is
    [`FlushGpuDirectRdmaWritesOptions::HOST`] (`1`) on an RDMA SKU
    (`flush_gpu_direct_rdma_writes` is a host-sync barrier; MemOps is
    never reported). WithCudaVMM is the same RDMA SKU bit (VMM is always
    on). GenericCompression is always 0 (compression is not modeled).
    Write-ordering options stay unmodeled. Also on [`DeviceProperties`].
    Query; capture-legal. Decode identity unchanged. `gpu-profile
    capture` is still refused. Dual score still has no `$/M tokens`.

250. [x] `cudaDevAttrHandleTypeWin32HandleSupported` /
    `HandleTypeWin32KmtHandleSupported` /
    `HandleTypeFabricSupported`: always 0 (this VM has POSIX-FD
    shareable pools; fabric handles are not modeled). Also on
    [`DeviceProperties`]. Query; capture-legal. Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

251. [x] `CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE`:
    [`PointerAttr::IsHwDecompressCapable`] is always 0 (compression is
    not modeled). Query-only; Set is Invalid `"pointer attr"`. Query;
    capture-legal. Decode identity unchanged. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

252. [x] `cuMulticastUnbind`: [`multicast_unbind`](Sim::multicast_unbind)
    drops a whole-handle bind. Live multicast VA maps are Invalid
    `"still mapped"` (`va_unmap` decrements the map count). Not currently
    bound is Invalid `"not bound"`. Partial offset/size unbind is not
    modeled. Host-synchronous; capture refused. Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

253. [x] `cuMemRelease` of a multicast object:
    [`multicast_destroy`](Sim::multicast_destroy) releases a
    `cuMulticastCreate` handle. Live multicast VA maps are Invalid
    `"still mapped"`. Remaining binds are dropped (handles stay live).
    Host-synchronous; capture refused. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

254. [x] `cuMulticastBindAddr`: [`multicast_bind_addr`](Sim::multicast_bind_addr)
    binds a mapped VMM VA on a team device (retain the map at offset 0,
    then BindMem). Partial offset/size bind is not modeled. Flags must be
    0 ([`MulticastBindFlags::DEFAULT`]; unknown bits Invalid
    `"multicast bind flags"`). `multicast_store` uses BindAddr.
    Host-synchronous; capture refused. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

255. [x] `cuMemCreate` prop + flags: [`va_create_with_prop`](Sim::va_create_with_prop)
    takes [`MemAllocationProp`] and flags. Flags must be 0
    ([`MemCreateFlags::DEFAULT`]). Location must be [`Place::Device`];
    handle types other than none Invalid `"vmm handle types"` (POSIX-FD
    VMM export is not modeled). RDMA capable on the prop is ignored
    (Get wraps the SKU). Typed [`va_create`](Sim::va_create) stays.
    Host-synchronous; capture refused. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

256. [x] `cuMemAddressReserve` alignment / addr / flags:
    [`va_reserve_with_flags`](Sim::va_reserve_with_flags). Flags must be 0
    ([`MemReserveFlags::DEFAULT`]). Nonzero addr Invalid `"reserve addr"`
    (fixed VA is not modeled). Nonzero alignment must be a power of two
    that divides size (Invalid `"reserve alignment"`). Typed
    [`va_reserve`](Sim::va_reserve) stays. Host-synchronous; capture
    refused. Decode identity unchanged. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

257. [x] `cuMemAddressFree` size: [`va_free_with_size`](Sim::va_free_with_size)
    requires `size` equal to the reserved bytes. Other sizes Invalid
    `"free size"`. Partial free is not modeled. Still mapped stays
    Invalid `"VA still mapped"`. Typed [`va_free`](Sim::va_free) stays.
    Host-synchronous; capture refused. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

258. [ ] Next numbered PLAN item after 257 is the next `gpu-sim` / Engine /
    serve / expertvm mechanical API that is still missing. Do not invent
    `cudaGraphMemAllocNodeSetParams` (it would resize HBM),
    `cudaDeviceGetStreamPriorityRange`, occupancy SM counts, CUDA version
    numbers, example `$/M tokens` rents, or `cudaStreamDestroy`.
    **`best_of` / `use_beam_search` stay out of `parse_gen_req` until a
    real beam Engine exists.** Prefer remaining CUDA-shaped twins over more
    OpenAI HTTP veneer. Do not default `--engine`. Do not invent
    `max_model_len` or `/v1/tokenize`.

Stop if Phase 1 traces say residency cannot work. Do not invent an
architecture or a dtype. Do not list `mixtral` or `qwen3vlmoe` as
accepted.

---

## Success

Researchers can:

- load a common GGUF, inspect logits, swap samplers/quants in safe Rust
- trust greedy decode against llama.cpp on real checkpoints
- emit MoE access traces from the same engine
- optimize expert residency against a deterministic GPU VM without
  owning a GPU
- eventually prove lower HBM / $ per token on rented silicon

That is the company-shaped outcome. `llama-rust` remains the
verifiable, hackable reference engine. `expertvm` is the OSS primitive.
`gpu-sim-rs` is how fleets of agents work on it anyway.

---

## Source conversation turns

| Turn | User | Conclusion kept in this plan |
| --- | --- | --- |
| 1 | What Rust OSS libs for cheap mid-tier / GLM-5.3-Flash serving? | Nine-crate stack. Do not build another ML framework. Independent primitives. |
| 2 | Which one actually fills a void? | Not a broad `moe-runtime`. **`expertvm`**: dynamic expert residency / weight virtualization. |
| 3 | How does that relate to `mingley/llama-rust`? | This repo = semantics + correctness lab. `expertvm` = physical location of weights. Do not merge into a serving monster. |
| 4 | Simulate without an expensive GPU? | Yes. ~70–80% of the policy question is CPU traces + a store/simulator. GPU is for proving economics. |
| 5 | Synthetic high-demand GPU with mechanical invariants for GPU-less agents | **`gpu-sim-rs`**: exact systems semantics, calibrated timing, discrete event. `expertvm` is the first app. |
