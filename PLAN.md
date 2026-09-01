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
| `SimulatedGpuStore` | fake HBM capacity, PCIe/NVLink bandwidth, DMA concurrency; `with_managed` is UM prefetch, `with_mapped` is zero-copy host (and `host_pin_bytes` occupancy), `with_vmm` is `va_acquire`; `with_cfg` is `host_func` / blocking streams / `sync_alloc` / mempool / `vmm_page` / pageable H2D / `SetAccessedBy` / legacy NULL / stream priority / `graph_update` / `graph_clone` / `timing_events` / `event_blocking_sync`; `expertvm store` / `store_replay_cfg` is the CLI (Markov prefetch) |

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
  GetAttribute Used/Reserved wrap live+cached; UsedMemHigh /
  ReservedMemHigh are CUDA high-water (Set `0` resets); graph mem stays
  `GraphMemAttr`;
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
  full HBM; default unset is a full chip), `cuGreenCtxRecordEvent` /
  `cuGreenCtxWaitEvent` (`green_ctx_record_event` / `green_ctx_wait_event`),
  `cudaExecutionCtxSynchronize` (`green_ctx_synchronize`),
  `cuStreamGetDevResource` (`stream_get_dev_resource`),
    `cuGreenCtxGetId` (`green_ctx_get_id`),
  `cudaExecutionCtxGetDevice` (`green_ctx_get_device`),
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
  instantiate (`graph_clone_ns`; the src is destroyed). `--graph-clone-parent`
  clones combo parents (recursive children; store GEMM stays per-leaf). `--graph-build`
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
  `--host-func`, `--copy-host`, blocking compute, `--pageable`, `--memset-fill`, `--accessed-by`,
  `--legacy-null`, `--stream-priority`, `--graph-update`, `--graph-set-params`, `--graph-clone`, `--graph-clone-parent`, `--graph-build`, `--graph-build-deps`, `--graph-host`, `--graph-piecewise`, `--graph-capture-deps`, `--graph-capture-host`, `--graph-mem`, `--graph-memset`, `--graph-memcpy`, `--graph-leaf-host`, `--graph-auto-free`, `--timing-events`, `--event-blocking-sync`, `--cooperative`, `--pdl`, `--l2-persist`, `--cluster`, `--preferred-cluster`, `--cluster-spread`, `--max-shared`, `--non-portable-cluster`, `--sync-policy`, `--device-sync-policy`, `--shared-mem`, and `--multicast`. `--mempool` sets the default
  pool release threshold to `u64::MAX` (vLLM-style hold); reuse of a
  cached page pays `pool_reuse_ns`. `--shareable` is POSIX-FD mempool IPC
  (implies `--mempool`; illegal with `--sync-alloc` / mapped / managed / vmm). `--mapped` is `cudaHostAllocMapped`
  (no H2D, PCIe kernels, HBM unused; walker slots also cap at
  `host_pin_bytes / expert_bytes`). `--managed` is `cudaMallocManaged`
  plus `cudaMemAdviseSetReadMostly` plus `SetPreferredLocation` plus
  `cudaMemPrefetchAsync` on miss (HBM charged on migrate; a second GPU
  prefetch keeps the copy; a remote read can keep the page on the preferred GPU).
  `--no-read-mostly` is `cudaMemAdviseUnsetReadMostly` at that fill so dest
  prefetch moves instead of replicating.   `--no-preferred` is
  `cudaMemAdviseUnsetPreferredLocation` at that fill so a remote GEMM
  first-touches instead of staying on home. `--no-mem-prefetch` skips
  `cudaMemPrefetchAsync` at that fill so the kernel first-touches on compute
  instead of copy-engine prefetch. `--accessed-by` is `cudaMemAdviseSetAccessedBy` on every GPU at fill
  (dest GEMM reads without migrating; migrate/pin skip dest prefetch).
  `--place replicas` then `prefetch`s hot keys onto dest GPUs unless
  `--accessed-by`; dest eviction is `drop_managed_copy` (one GPU's copy, allocation stays). `--vmm` is
  `va_acquire` (remap idle VA or reserve+map) then H2D; evict `va_release`s
  the pointer. `--vmm-page N` is `va_acquire_paged` (KV-block physicals;
  implies `--vmm`). `expertvm kv` demand-pages per-sequence VAs (`cuMemCreate`
  + `cuMemMap`; `--sequences N` aliases interned physicals; `kernel_bufs`
  + H2D or `memset_buf` at a mapped span; peak HBM is unique pages). `--host-func` is `cudaLaunchHostFunc` after each event's
  GEMMs (`host_func_ns`; no GPU occupancy). `--copy-host` is
  `cudaLaunchHostFunc` after miss DMA / prefetch (`host_func_ns` on the DMA
  stream before copy-ready; does not imply `--host-func`; mapped misses are
  a no-op). `--blocking-streams` is
  `cudaStreamCreate` on seq-streams (serialize with NULL); default is
  `cudaStreamNonBlocking`. `--legacy-null` is `set_legacy_null_stream`
  (NULL serializes with every stream). `--pageable` is host-sync
  `memcpy_host_to_device` (`pageable_permille`). `--memset-fill` is
  `cudaMemsetAsync` of pinned/VMM miss pages (HBM write, compute occupancy;
  not mapped/managed/pageable/memcpy-batch; distinct from `--graph-memset`
  scratch). `--stream-priority` is
  `cudaStreamCreateWithPriority` on seq-streams (priority = stream id). `--graph-update`
  is `cudaGraphExecUpdate` of a parked leaf (store and `--cuda-graphs`
  walker). `--graph-set-params` is `cudaGraphExecKernelNodeSetParams` of a
  parked leaf (no second capture; legal with mem nodes). `--graph-clone` is `cudaGraphClone` of a leaf capture before
  instantiate (graph vs exec). `--graph-clone-parent` is `cudaGraphClone` of combo
  parents (recursive children; store GEMM stays per-leaf). `--graph-build` is `cudaGraphCreate` /
  `cudaGraphAddKernelNode` (and child add for combo parents; independent
  children may Hyper-Q overlap). `--graph-piecewise` is
  `cudaStreamBeginCaptureToGraph` combo parents (independent child roots;
  not with `--graph-build`). `--graph-capture-deps` is non-empty
  `cudaStreamBeginCaptureToGraph` deps on those combo parents (needs
  `--graph-piecewise`; sibling GEMMs serialize). `--graph-capture-host` is
  captured `cudaLaunchHostFunc` BETWEEN those fragments (needs
  `--graph-piecewise`; sibling GEMMs serialize through `host_func_ns`; not a
  JOIN after overlap; does not imply `--host-func`). `--graph-build-deps` is
  `cudaGraphAddDependencies` on `--graph-build` combo parents (needs
  `--graph-build`; sibling GEMMs serialize). `--graph-host` is
  `cudaGraphAddHostNode` BETWEEN those children (needs `--graph-build`;
  sibling GEMMs serialize through `host_func_ns`; not a JOIN after overlap). `--graph-mem`
  is in-graph scratch (`cudaGraphAddMemAllocNode` / capture `cudaMallocAsync`);
  `--graph-memset` is `cudaGraphAddMemsetNode` / capture `cudaMemsetAsync` of
  that scratch BETWEEN alloc and GEMM (needs `--graph-mem`; extra HBM-write tax).
  `--graph-memcpy` is `cudaGraphAddMemcpyNode` / capture `cudaMemcpyAsync` H2D of
  that scratch BETWEEN alloc and GEMM (needs `--graph-mem`; copy-engine PCIe tax;
  legal with `--graph-memset`).
  `--graph-leaf-host` is `cudaGraphAddHostNode` / captured `cudaLaunchHostFunc`
  BEFORE the leaf GEMM (implies `--cuda-graphs`; each leaf bills `host_func_ns`;
  not with `--device-launch`; does not imply `--host-func`).
  `--graph-update` is skipped because CUDA cannot update mem nodes.
  `--graph-auto-free` is AutoFreeOnLaunch scratch without a matching free
  (illegal with `--graph-mem`).
  `--timing-events` is timing-on copy events
  plus `event_elapsed_ns` (`cudaEventElapsedTime`); default wait events stay
  `cudaEventDisableTiming`. `--event-blocking-sync` is `cudaEventBlockingSync`
  on those copy events (implies `--timing-events`; `synchronize_event` pays
  `host_sync_blocking_ns`; distinct from `--sync-policy blocking`). `memset` / `memset_buf` of a mapped span, directed peer enable, and
  the legacy null stream are mechanical CUDA invariants.
  `synchronize_stream` / `synchronize_event` / `synchronize_device` are
  `cudaStreamSynchronize` / `cudaEventSynchronize` / `cudaDeviceSynchronize`. `event_elapsed_ns` is `cudaEventElapsedTime` in
  nanoseconds (`create_event_disable_timing` forbids it). `query_event` is `cudaEventQuery`. `query_stream` is `cudaStreamQuery`.
  `event_get_flags` is `cudaEventGetFlags`.
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
    `--graph-clone` / `--graph-clone-parent` / `--graph-build` / `--graph-build-deps` / `--graph-host` / `--graph-piecewise` / `--graph-capture-deps` / `--graph-capture-host` / `--graph-mem` / `--graph-memset` / `--graph-memcpy` / `--graph-leaf-host` / `--graph-auto-free` / `--timing-events` / `--event-blocking-sync` / `--cuda-graphs` match
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
    `--pageable` / `--memset-fill` / `--memcpy-attr` / `--d2h-evict` / `--d2h-pageable` / `--copy-host` / `--accessed-by` / `--no-read-mostly` / `--no-preferred` / `--no-mem-prefetch` / `--legacy-null` / `--stream-priority`
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
    leave sibling `graph_add_child` nodes independent. `--graph-build-deps`
    (PLAN 359) chains those children. `--graph-host` (PLAN 360) inserts
    `graph_add_host_func` BETWEEN those children (`host_func_ns`). `update_graph` treats
    edges as topology. Leaf `--graph-mem` / `--graph-auto-free` chains
    alloc→kernel→free. `--graph-memset` (PLAN 361) inserts memset BETWEEN
    alloc and kernel (`alloc→memset→kernel→free`). `--graph-memcpy` (PLAN 362)
    inserts H2D memcpy BETWEEN alloc and kernel (`alloc→memcpy→kernel→free`;
    copy-engine PCIe). Decode identity stays stream capture. Dual score still
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
    `--cuda-graphs` on the walker). Illegal with `--graph-build`.
    `--graph-capture-deps` (PLAN 358) chains those fragments.
    `--graph-capture-host` (PLAN 363) inserts captured `host_func` BETWEEN
    those fragments (`host_func_ns`). Decode
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

258. [x] `cuMemMap` flags: [`va_map_handle_with_flags`](Sim::va_map_handle_with_flags)
    requires flags 0 ([`MemMapFlags::DEFAULT`]). Unknown bits Invalid
    `"mem map flags"`. Typed [`va_map_handle`](Sim::va_map_handle) stays.
    Host-synchronous; capture refused. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

259. [x] `cudaDevAttrHostMemoryPoolsSupported`: always 0 (this VM's pools
    are device-only; [`create_pool_with_props`](Sim::create_pool_with_props)
    refuses host location). Also on [`DeviceProperties`]. Query;
    capture-legal. Decode identity unchanged. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

260. [x] `cuMulticastBindMem` flags:
    [`multicast_bind_mem_with_flags`](Sim::multicast_bind_mem_with_flags)
    requires flags 0 ([`MulticastBindFlags::DEFAULT`]; unknown bits Invalid
    `"multicast bind flags"`). Typed [`multicast_bind_mem`](Sim::multicast_bind_mem)
    stays. Capture then flags. Partial offset/size bind is not modeled.
    Host-synchronous; capture refused. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

261. [x] `cudaDevAttrIsMultiGpuBoard` / `MultiGpuBoardGroupID`: always 0
    (example SKUs are discrete single-GPU packages). Also on
    [`DeviceProperties`]. Query; capture-legal. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

262. [x] `CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID`:
    [`PointerAttr::MemoryBlockId`] is the [`MemHandleId`] of the `cuMemMap`
    covering offset 0 (`va_map_handle` or [`va_retain_handle`](Sim::va_retain_handle)).
    `cudaMalloc`, combined [`va_map`](Sim::va_map) without retain, unmapped
    VMM, and maps that skip offset 0 are Invalid `"pointer attr"`. Does not
    duplicate [`PointerAttr::BufferId`]. Query-only; Set is Invalid
    `"pointer attr"`. Query; capture-legal. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

263. [x] `cuMemUnmap` size: [`va_unmap_with_size`](Sim::va_unmap_with_size)
    requires `size` equal to the reserved bytes. Other sizes Invalid
    `"unmap size"`. Partial unmap stays [`va_unmap_range`](Sim::va_unmap_range).
    Typed [`va_unmap`](Sim::va_unmap) stays. Host-synchronous; capture
    refused. Decode identity unchanged. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

264. [x] `cuMulticastUnbind` size:
    [`multicast_unbind_with_size`](Sim::multicast_unbind_with_size) requires
    `size` equal to the multicast object bytes. Other sizes Invalid
    `"unbind size"`. CUDA `mcOffset` is 0 (partial unbind is not modeled).
    Typed [`multicast_unbind`](Sim::multicast_unbind) stays.
    Host-synchronous; capture refused. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

265. [x] `cuMulticastBindAddr` size:
    [`multicast_bind_addr_with_size`](Sim::multicast_bind_addr_with_size)
    requires `size` equal to the reserved VA. Other sizes Invalid
    `"bind size"`. CUDA `mcOffset` is 0 (partial bind is not modeled).
    Flags still [`MulticastBindFlags::DEFAULT`]. Typed
    [`multicast_bind_addr`](Sim::multicast_bind_addr) /
    [`multicast_bind_addr_with_flags`](Sim::multicast_bind_addr_with_flags)
    stay. Host-synchronous; capture refused. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

266. [x] `cuMulticastBindMem` size:
    [`multicast_bind_mem_with_size`](Sim::multicast_bind_mem_with_size)
    requires `size` equal to the handle bytes. Other sizes Invalid
    `"bind size"`. CUDA `mcOffset` / `memOffset` are 0 (partial bind is
    not modeled). Handle vs object mismatch stays `"handle size mismatch"`.
    Flags still [`MulticastBindFlags::DEFAULT`]. Typed
    [`multicast_bind_mem`](Sim::multicast_bind_mem) /
    [`multicast_bind_mem_with_flags`](Sim::multicast_bind_mem_with_flags)
    stay. Host-synchronous; capture refused. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

267. [x] `cuMemMap` of a multicast handle flags:
    [`va_map_multicast_with_flags`](Sim::va_map_multicast_with_flags)
    requires flags 0 ([`MemMapFlags::DEFAULT`]). Unknown bits Invalid
    `"mem map flags"`. Typed [`va_map_multicast`](Sim::va_map_multicast)
    stays. Host-synchronous; capture refused. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

268. [x] `cuMemMap` of a multicast handle size:
    [`va_map_multicast_with_size`](Sim::va_map_multicast_with_size)
    requires `size` equal to the multicast object bytes. Other sizes
    Invalid `"mem map size"`. Flags still [`MemMapFlags::DEFAULT`]. Typed
    [`va_map_multicast`](Sim::va_map_multicast) /
    [`va_map_multicast_with_flags`](Sim::va_map_multicast_with_flags) stay.
    Host-synchronous; capture refused. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

269. [x] `cuMemMap` size: [`va_map_handle_with_size`](Sim::va_map_handle_with_size)
    requires `size` equal to the handle bytes. Other sizes Invalid
    `"mem map size"`. Flags still [`MemMapFlags::DEFAULT`]. Typed
    [`va_map_handle`](Sim::va_map_handle) /
    [`va_map_handle_with_flags`](Sim::va_map_handle_with_flags) stay.
    Host-synchronous; capture refused. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

270. [x] `cudaDeviceGetName`: [`device_get_name`](Sim::device_get_name) is
    the profile name (same as [`DeviceProperties::name`]). Query;
    capture-legal. Decode identity unchanged. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

271. [x] `cuDeviceTotalMem`: [`device_total_mem`](Sim::device_total_mem) is
    HBM bytes (same as [`DeviceAttr::TotalGlobalMem`] /
    [`DeviceProperties::total_global_mem`]). Query; capture-legal. Decode
    identity unchanged. `gpu-profile capture` is still refused. Dual score
    still has no `$/M tokens`.

272. [x] `cuMemSetAccess` size:
    [`va_set_access_with_size`](Sim::va_set_access_with_size) requires `size`
    equal to the reserved bytes. Other sizes Invalid `"access size"`.
    Partial SetAccess is not modeled. Flags stay [`MemAccessFlags`]. Typed
    [`va_set_access`](Sim::va_set_access) /
    [`va_set_access_with_flags`](Sim::va_set_access_with_flags) stay.
    Host-synchronous; capture refused. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

273. [x] `cuMulticastCreate` prop:
    [`multicast_create_with_prop`](Sim::multicast_create_with_prop) takes
    [`MulticastObjectProp`]. Handle types other than none Invalid
    `"multicast handle types"` (POSIX-FD multicast export is not modeled).
    Flags must be 0 ([`MulticastCreateFlags::DEFAULT`]). Typed
    [`multicast_create`](Sim::multicast_create) stays. Host-synchronous;
    capture refused. Decode identity unchanged. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

274. [x] `cuDeviceGet`: [`device_get`](Sim::device_get) is the ordinal in
    `0 .. device_count`. Other ordinals Invalid `"device not in profile"`.
    Query; capture-legal. Decode identity unchanged. `gpu-profile capture`
    is still refused. Dual score still has no `$/M tokens`.

275. [x] `cuMulticastGetGranularity` prop:
    [`multicast_get_granularity_with_prop`](Sim::multicast_get_granularity_with_prop)
    takes [`MulticastObjectProp`]. Handle types other than none Invalid
    `"multicast handle types"` (POSIX-FD multicast export is not modeled).
    Flags must be 0 ([`MulticastCreateFlags::DEFAULT`]). Size and team
    size are not validated (CUDA queries granularity before create).
    Granularity flags stay MINIMUM / RECOMMENDED. Typed
    [`multicast_get_granularity`](Sim::multicast_get_granularity) stays.
    Query; capture-legal. Decode identity unchanged. `gpu-profile capture`
    is still refused. Dual score still has no `$/M tokens`.

276. [x] `cudaDevAttrComputeMode`: [`DeviceAttr::ComputeMode`] is always
    [`ComputeMode::DEFAULT`] (`0`). Exclusive process / prohibited are
    not modeled. [`DeviceProperties::compute_mode`] matches. Query;
    capture-legal. Decode identity unchanged. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

277. [x] `cudaDevAttrTccDriver`: [`DeviceAttr::TccDriver`] is always 0
    (example SKUs are not Windows TCC). [`DeviceProperties::tcc_driver`]
    matches. Query; capture-legal. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

278. [x] `cudaDevAttrKernelExecTimeout`:
    [`DeviceAttr::KernelExecTimeout`] is always 0 (example SKUs have no
    display watchdog). [`DeviceProperties::kernel_exec_timeout`] matches.
    Query; capture-legal. Decode identity unchanged. `gpu-profile capture`
    is still refused. Dual score still has no `$/M tokens`.

279. [x] `cudaDevAttrCanUse64BitStreamMemOps`:
    [`DeviceAttr::CanUse64BitStreamMemOps`] is always 1 (this VM has
    [`wait_value64`](Sim::wait_value64) / [`write_value64`](Sim::write_value64)).
    [`DeviceProperties::can_use_64_bit_stream_mem_ops`] matches. Query;
    capture-legal. Decode identity unchanged. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

280. [x] `cudaDevAttrCanUseStreamWaitValueNor`:
    [`DeviceAttr::CanUseStreamWaitValueNor`] is always 1 (this VM has
    [`WaitValueCmp::Nor`]). [`DeviceProperties::can_use_stream_wait_value_nor`]
    matches. Query; capture-legal. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

281. [x] `cudaDevAttrTensorMapAccessSupported`:
    [`DeviceAttr::TensorMapAccessSupported`] is always 0 (`CUtensorMap` /
    TMA is not modeled). [`DeviceProperties::tensor_map_access_supported`]
    matches. Query; capture-legal. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

282. [x] `cudaDevAttrUnifiedFunctionPointers`:
    [`DeviceAttr::UnifiedFunctionPointers`] is always 0 (device-side
    function pointers are not modeled).
    [`DeviceProperties::unified_function_pointers`] matches. Query;
    capture-legal. Decode identity unchanged. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

283. [x] `cudaDevAttrTimelineSemaphoreInteropSupported`:
    [`DeviceAttr::TimelineSemaphoreInteropSupported`] is always 0 (NVSci
    / timeline semaphore interop is not modeled).
    [`DeviceProperties::timeline_semaphore_interop_supported`] matches.
    Query; capture-legal. Decode identity unchanged. `gpu-profile capture`
    is still refused. Dual score still has no `$/M tokens`.

284. [x] `cudaDevAttrMemDecompressAlgorithmMask`:
    [`DeviceAttr::MemDecompressAlgorithmMask`] is always 0 (hardware
    decompress is not modeled; same as
    [`PointerAttr::IsHwDecompressCapable`]).
    [`DeviceProperties::mem_decompress_algorithm_mask`] matches. Query;
    capture-legal. Decode identity unchanged. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

285. [x] `cudaDevAttrMemDecompressMaximumLength`:
    [`DeviceAttr::MemDecompressMaximumLength`] is always 0 (hardware
    decompress is not modeled). [`DeviceProperties::mem_decompress_maximum_length`]
    matches. Query; capture-legal. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

286. [x] `cuMemSetAccess` descriptor array:
    [`va_set_access_n`](Sim::va_set_access_n) takes [`MemAccessDesc`]
    (`desc`, `count`). `size` must equal the reserved bytes. Host location
    Invalid `"access location"`. Flags stay [`MemAccessFlags`]. All-or-nothing.
    Empty `descs` is a no-op after the size check. Typed
    [`va_set_access`](Sim::va_set_access) /
    [`va_set_access_with_flags`](Sim::va_set_access_with_flags) /
    [`va_set_access_with_size`](Sim::va_set_access_with_size) stay.
    Host-synchronous; capture refused. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

287. [x] `cudaMemPoolSetAccess` descriptor array:
    [`pool_set_access_n`](Sim::pool_set_access_n) takes [`MemAccessDesc`]
    (`descList`, `count`). Host location Invalid `"access location"`.
    Flags match [`pool_set_access_with_flags`](Sim::pool_set_access_with_flags).
    All-or-nothing. Empty `descs` is a no-op after pool checks. Typed
    [`pool_set_access`](Sim::pool_set_access) /
    [`pool_set_access_with_flags`](Sim::pool_set_access_with_flags) stay.
    Host-synchronous; capture refused. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

288. [x] `cudaDevAttrHostNumaVirtualMemoryManagementSupported`: always 0
    (host NUMA VMM is not modeled; [`va_create_with_prop`](Sim::va_create_with_prop)
    refuses host location). Distinct from
    [`DeviceAttr::VirtualMemoryManagementSupported`] (device VMM is 1)
    and [`DeviceAttr::HostMemoryPoolsSupported`] (host pools are already
    0). Do not invent HostNuma IDs. Query; capture-legal. Decode
    identity unchanged. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

289. [x] `cudaDevAttrCanUseStreamMemOps`: always 1 (this VM has
    [`wait_value32`](Sim::wait_value32) / [`write_value32`](Sim::write_value32)).
    CUDA deprecated this in favor of
    [`DeviceAttr::CanUse64BitStreamMemOps`]. Query; capture-legal. Decode
    identity unchanged. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

290. [x] `cudaDevAttrHostNumaMemoryPoolsSupported`: always 0 (host NUMA
    pools are not modeled; [`create_pool_with_props`](Sim::create_pool_with_props)
    refuses host location). Distinct from
    [`DeviceAttr::HostMemoryPoolsSupported`] (already 0) and
    [`DeviceAttr::HostNumaVirtualMemoryManagementSupported`] (VMM, not
    pools). Do not invent HostNuma IDs. Query; capture-legal. Decode
    identity unchanged. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

291. [x] `cudaDevAttrHostNumaMultinodeIpcSupported`: always 0 (this VM's
    IPC is same-node; [`ipc_open`](Sim::ipc_open) requires the dest GPU
    already in the allocation). Distinct from
    [`DeviceAttr::IpcEventSupport`] (same-process event IPC is 1). Do
    not invent HostNuma IDs. Query; capture-legal. Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

292. [x] `cudaDevAttrNumaConfig`: always [`DeviceNumaConfig::NONE`] (GPU
    memory NUMA nodes are not modeled). Do not invent `cudaDevAttrNumaId`
    or HostNuma IDs. Query; capture-legal. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

293. [x] `cudaDevAttrOnlyPartialHostNativeAtomicSupported`: always 0
    (host-mapped atomics are not modeled). Distinct from
    [`DeviceAttr::HostNativeAtomicSupported`] (already 0). Query;
    capture-legal. Decode identity unchanged. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

294. [x] `cuStreamWaitValue32` / `WaitValue64` flags:
    [`wait_value32_with_flags`](Sim::wait_value32_with_flags) /
    [`wait_value64_with_flags`](Sim::wait_value64_with_flags) take
    [`WaitValueFlags`] (GEQ/EQ/AND/NOR). [`WaitValueFlags::FLUSH`] and
    unknown bits Invalid `"wait value flags"`. Graph twins
    [`graph_add_wait_value32_with_flags`](Sim::graph_add_wait_value32_with_flags)
    /
    [`graph_add_wait_value64_with_flags`](Sim::graph_add_wait_value64_with_flags).
    Typed [`wait_value32`](Sim::wait_value32) / [`wait_value64`](Sim::wait_value64)
    stay. Decode identity unchanged. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

295. [x] `cuStreamWriteValue32` / `WriteValue64` flags:
    [`write_value32_with_flags`](Sim::write_value32_with_flags) /
    [`write_value64_with_flags`](Sim::write_value64_with_flags) take
    [`WriteValueFlags`]. Flags must be [`WriteValueFlags::DEFAULT`].
    [`WriteValueFlags::NO_MEMORY_BARRIER`] Invalid `"write value flags"`.
    Graph twins
    [`graph_add_write_value32_with_flags`](Sim::graph_add_write_value32_with_flags)
    /
    [`graph_add_write_value64_with_flags`](Sim::graph_add_write_value64_with_flags).
    Typed [`write_value32`](Sim::write_value32) / [`write_value64`](Sim::write_value64)
    stay. Decode identity unchanged. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

296. [x] `cuStreamBatchMemOp` flags:
    [`batch_mem_op_with_flags`](Sim::batch_mem_op_with_flags) takes
    [`BatchMemOpFlags`]. Flags must be [`BatchMemOpFlags::DEFAULT`].
    Unknown bits Invalid `"batch mem op flags"`. Graph twin
    [`graph_add_batch_mem_op_with_flags`](Sim::graph_add_batch_mem_op_with_flags).
    Typed [`batch_mem_op`](Sim::batch_mem_op) /
    [`graph_add_batch_mem_op`](Sim::graph_add_batch_mem_op) stay. Decode
    identity unchanged. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

297. [x] `cudaMemPoolExportToShareableHandle` /
    `ImportFromShareableHandle` handle type and flags:
    [`pool_export_with_type`](Sim::pool_export_with_type) /
    [`pool_import_with_type`](Sim::pool_import_with_type) take
    [`MemHandleType::POSIX_FILE_DESCRIPTOR`] and
    [`MemPoolExportFlags::DEFAULT`]. Other handle types Invalid
    `"pool handle types"`. Unknown flags Invalid `"pool export flags"` /
    `"pool import flags"`. Typed [`pool_export`](Sim::pool_export) /
    [`pool_import`](Sim::pool_import) stay. Host-synchronous; capture
    refused. Decode identity unchanged. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

298. [x] `cudaMemcpy2D` / `cudaMemcpy2DAsync`:
    [`memcpy_2d`](Sim::memcpy_2d) / [`memcpy_2d_async`](Sim::memcpy_2d_async)
    require [`MemcpyOp::is_2d`] (`height > 1`, not 3D). Other extents Invalid
    `"memcpy2d height"`. Typed [`memcpy`](Sim::memcpy) /
    [`memcpy_sync`](Sim::memcpy_sync) stay. Host-sync twin capture refused.
    Decode identity unchanged. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

299. [x] `cudaMemcpy3D` / `cudaMemcpy3DAsync`:
    [`memcpy_3d`](Sim::memcpy_3d) / [`memcpy_3d_async`](Sim::memcpy_3d_async)
    require [`MemcpyOp::is_3d`] (`depth > 1`). Other extents Invalid
    `"memcpy3d depth"`. Typed [`memcpy`](Sim::memcpy) /
    [`memcpy_sync`](Sim::memcpy_sync) stay. Host-sync twin capture refused.
    Decode identity unchanged. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

300. [x] `cudaMemset2D` / `cudaMemset2DAsync`:
    [`memset_2d`](Sim::memset_2d) / [`memset_2d_async`](Sim::memset_2d_async)
    require [`MemsetOp::is_2d`] (`height > 1`, not 3D). Other extents Invalid
    `"memset2d height"`. Typed [`memset_op`](Sim::memset_op) /
    [`memset_op_sync`](Sim::memset_op_sync) stay. Host-sync twin capture
    refused. Decode identity unchanged. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

301. [x] `cudaMemset3D` / `cudaMemset3DAsync`:
    [`memset_3d`](Sim::memset_3d) / [`memset_3d_async`](Sim::memset_3d_async)
    require [`MemsetOp::is_3d`] (`depth > 1`). Other extents Invalid
    `"memset3d depth"`. Typed [`memset_op`](Sim::memset_op) /
    [`memset_op_sync`](Sim::memset_op_sync) stay. Host-sync twin capture
    refused. Decode identity unchanged. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

302. [x] `cudaMemcpy2DPeer` / `cudaMemcpy2DPeerAsync` require 2D:
    [`memcpy_peer_2d`](Sim::memcpy_peer_2d) /
    [`memcpy_peer_2d_async`](Sim::memcpy_peer_2d_async) require
    [`MemcpyOp::is_2d`] (`height > 1`, not 3D). Other extents Invalid
    `"memcpy2d height"`. Host-sync twin capture refused. Typed
    [`memcpy_peer`](Sim::memcpy_peer) stays. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no `$/M
    tokens`.

303. [x] `cudaMemcpy3DPeer` / `cudaMemcpy3DPeerAsync` require 3D:
    [`memcpy_peer_3d`](Sim::memcpy_peer_3d) /
    [`memcpy_peer_3d_async`](Sim::memcpy_peer_3d_async) require
    [`MemcpyOp::is_3d`] (`depth > 1`). Other extents Invalid
    `"memcpy3d depth"`. Host-sync twin capture refused. Typed
    [`memcpy_peer`](Sim::memcpy_peer) stays. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no `$/M
    tokens`.

304. [x] expertvm pitched KV miss fill uses named 2D copies:
    [`kv_paged`](kv_paged) with [`KvCfg::with_pitch`] calls
    [`memcpy_2d_async`](Sim::memcpy_2d_async) / [`memset_2d_async`](Sim::memset_2d_async)
    when the op is 2D (`height > 1`, not 3D). Packed 1D stays
    [`memcpy`](Sim::memcpy) / [`memset_op`](Sim::memset_op). Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

305. [x] `cudaGraphAddMemcpyNode` 2D helper:
    [`graph_add_memcpy_2d`](Sim::graph_add_memcpy_2d) requires
    [`MemcpyOp::is_2d`] (`height > 1`, not 3D). Other extents Invalid
    `"memcpy2d height"`. Typed [`graph_add_memcpy`](Sim::graph_add_memcpy) /
    [`graph_add_memcpy_1d`](Sim::graph_add_memcpy_1d) stay. Decode identity
    unchanged. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

306. [x] `cudaGraphAddMemcpyNode` 3D helper:
    [`graph_add_memcpy_3d`](Sim::graph_add_memcpy_3d) requires
    [`MemcpyOp::is_3d`] (`depth > 1`). Other extents Invalid
    `"memcpy3d depth"`. Typed [`graph_add_memcpy`](Sim::graph_add_memcpy)
    stays. Decode identity unchanged. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

307. [x] `cudaGraphAddMemsetNode` 2D helper:
    [`graph_add_memset_2d`](Sim::graph_add_memset_2d) requires
    [`MemsetOp::is_2d`] (`height > 1`, not 3D). Other extents Invalid
    `"memset2d height"`. Typed [`graph_add_memset_op`](Sim::graph_add_memset_op)
    stays. Decode identity unchanged. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

308. [x] `cudaGraphAddMemsetNode` 3D helper:
    [`graph_add_memset_3d`](Sim::graph_add_memset_3d) requires
    [`MemsetOp::is_3d`] (`depth > 1`). Other extents Invalid
    `"memset3d depth"`. Typed [`graph_add_memset_op`](Sim::graph_add_memset_op)
    stays. Decode identity unchanged. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

309. [x] `cudaGraphMemcpyNodeSetParams` 2D helper:
    [`graph_memcpy_set_params_2d`](Sim::graph_memcpy_set_params_2d) /
    [`graph_exec_memcpy_set_params_2d`](Sim::graph_exec_memcpy_set_params_2d)
    require [`MemcpyOp::is_2d`] (`height > 1`, not 3D). Other extents
    Invalid `"memcpy2d height"`. Typed [`graph_memcpy_set_params`](Sim::graph_memcpy_set_params)
    / [`graph_exec_memcpy_set_params`](Sim::graph_exec_memcpy_set_params) /
    1D twins stay. Capture refused. Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no `$/M
    tokens`.

310. [x] `cudaGraphMemcpyNodeSetParams` 3D helper:
    [`graph_memcpy_set_params_3d`](Sim::graph_memcpy_set_params_3d) /
    [`graph_exec_memcpy_set_params_3d`](Sim::graph_exec_memcpy_set_params_3d)
    require [`MemcpyOp::is_3d`] (`depth > 1`). Other extents Invalid
    `"memcpy3d depth"`. Typed SetParams / 1D / 2D twins stay. Capture
    refused. Decode identity unchanged. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

311. [x] `cudaGraphMemsetNodeSetParams` 2D helper:
    [`graph_memset_set_params_2d`](Sim::graph_memset_set_params_2d) /
    [`graph_exec_memset_set_params_2d`](Sim::graph_exec_memset_set_params_2d)
    require [`MemsetOp::is_2d`] (`height > 1`, not 3D). Other extents
    Invalid `"memset2d height"`. Typed SetParams stays. Capture refused.
    Decode identity unchanged. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

312. [x] `cudaGraphMemsetNodeSetParams` 3D helper:
    [`graph_memset_set_params_3d`](Sim::graph_memset_set_params_3d) /
    [`graph_exec_memset_set_params_3d`](Sim::graph_exec_memset_set_params_3d)
    require [`MemsetOp::is_3d`] (`depth > 1`). Other extents Invalid
    `"memset3d depth"`. Typed SetParams / 2D twins stay. Capture refused.
    Decode identity unchanged. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

313. [x] `cudaMemPoolAttrUsedMemHigh` / `ReservedMemHigh`:
    [`MemPoolAttr::UsedMemHigh`] / [`MemPoolAttr::ReservedMemHigh`] track
    ordinary-pool high-water (live / live+cached). Set `0` resets to
    current; other values Invalid `"pool high attr"`. Current used/reserved
    stay read-only. Graph-memory pool still Invalid (use
    [`GraphMemAttr`]). Get is capture-legal; Set capture refused. Decode
    identity unchanged. `gpu-profile capture` is still refused. Dual score
    still has no `$/M tokens`.

314. [x] Official Bloom (`architecture=bloom`): writer-built `tiny-bloom`
    plus decode of llama.cpp `src/models/bloom.cpp` — `token_embd_norm`
    LayerNorm, fused `attn_qkv` (concatenated Q/K/V), `LLM_NORM` on
    attn/ffn/output, sequential residual, `LLM_FFN_GELU`/`LLM_FFN_SEQ`
    with biases, ALiBi (`f_max_alibi_bias = 8`, no RoPE). Convert-shaped
    `bloom.*` KV (`layer_norm_epsilon`, `feed_forward_length = 4 * n_embed`,
    `head_count_kv = n_head`, `context_length`). `{arch}.context_length`
    is still unused for KV sizing. `gemma2` stays rejected. Decode
    identity vs oracle. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

315. [x] `cudaStreamAttributeAccessPolicyWindow`:
    [`set_stream_access_policy`](Sim::set_stream_access_policy) /
    [`stream_access_policy`](Sim::stream_access_policy) plus
    [`StreamAttr::AccessPolicy`]. Inherited by [`kernel`](Sim::kernel) /
    [`kernel_bufs`](Sim::kernel_bufs) on that stream. [`kernel_with`](Sim::kernel_with)
    and graph replay use the launch / node window. Set `None` clears.
    [`stream_copy_attributes`](Sim::stream_copy_attributes) copies or
    clears. Validate on Some. Decode identity stays [`kernel`](Sim::kernel)
    with no stream window. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

316. [x] Official Gemma2 (`architecture=gemma2`): writer-built `tiny-gemma2`
    plus decode of llama.cpp `src/models/gemma2.cpp` — Gemma embed-scale +
    GeGLU, `post_attention_norm` / `ffn_norm` / `post_ffw_norm`, sliding-window
    attention (`set_swa_pattern(2)`, `LLAMA_SWA_TYPE_STANDARD`), attn/final
    tanh logit softcap. Convert-shaped `gemma2.*` KV (`attn_logit_softcapping`,
    `final_logit_softcapping`, `attention.sliding_window`, `attention.key_length`
    / `value_length`, `context_length`). Tied `output.weight`. Writer-tiny uses
    `attention.sliding_window = 2` so short-seq tests clip. `{arch}.context_length`
    is still unused for KV sizing. `gemma3` stays rejected. Decode identity vs
    oracle. `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

317. [x] Official Gemma3 (`architecture=gemma3`): writer-built `tiny-gemma3`
    plus decode of llama.cpp `src/models/gemma3.cpp` — Gemma embed-scale +
    GeGLU, QK-Norm before RoPE (`attn_q_norm` / `attn_k_norm`),
    `post_attention_norm` / `ffn_norm` / `post_ffw_norm`, SWA default period 6
    (`set_swa_pattern`, `LLAMA_SWA_TYPE_STANDARD` when `n_swa > 0`), no attn
    logit softcap, optional final tanh logit softcap (default 0). Convert-shaped
    `gemma3.*` KV (`attention.sliding_window` when present, `attention.key_length`
    / `value_length`, `context_length`; omit `attn_logit_softcapping`). Tied
    `output.weight`. Writer-tiny uses `attention.sliding_window = 2` so short-seq
    tests clip.     `{arch}.context_length` is still unused for KV sizing. `gemma3n`
    stays rejected. Decode identity vs oracle. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

318. [x] Official Gemma3n (`architecture=gemma3n`): writer-built
    `tiny-gemma3n` plus decode of llama.cpp `src/models/gemma3n.cpp` —
    Gemma embed-scale + GeGLU, QK-Norm before RoPE, unweighted RMSNorm on
    V, attention scale `1.0`, RMSNorm `post_attention_norm` / `ffn_norm` /
    `post_ffw_norm`, SWA default period 5 (required
    `attention.sliding_window`), final tanh softcap (default 30), AltUp
    (4 residual streams) + Laurel + per-layer inputs, gaussian_topk on
    the first 10 layers (`n_layer_sparsity` hardcoded, not GGUF). Convert
    skips `lm_head.weight` when tied, omits `attn_logit_softcapping` /
    `rope.dimension_count`. Writer-tiny uses `attention.sliding_window = 2`
    so short-seq tests clip, omits `sliding_window_pattern` (period 5),
    and writes convert-shaped `gemma3n.altup.*` /
    `embedding_length_per_layer_input`. Convert `norm_shift` is 0.
    `{arch}.context_length` is still unused for KV sizing. `gemma4` stays
    rejected. Decode identity vs oracle. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

319. [x] `cudaMemcpyBatchAsync` / `cudaMemcpyWithAttributesAsync`:
    [`memcpy_batch_async`](gpu-sim/src/sim.rs) is 1D pointer-to-pointer
    only (2D/3D Invalid `"memcpy batch 1d"`; `cudaMemcpy3DBatchAsync` is
    not this API). Copies in one batch share a snapshotted stream-order
    predecessor list (`MemcpySrcAccessOrder::Stream`) or empty deps
    (`DuringApiCall` / `Any`) so they do not wait for each other; later
    submits still wait for the whole batch. `attrs_idxs[0] == 0`, strictly increasing, last `< count`,
    `numAttrs <= count`; empty batch requires empty attrs. DuringApiCall
    waits those copies before return (not the stream). Any does not wait.
    [`memcpy_with_attributes`](gpu-sim/src/sim.rs) Stream is [`memcpy`](gpu-sim/src/sim.rs).
    [`MemcpyFlags::PREFER_OVERLAP_WITH_COMPUTE`] is ignored (discrete).
    Location hints omitted (`ConcurrentManagedAccess` /
    `PageableMemoryAccess` are 0). Capture cannot include a batch
    (`"cannot capture memcpy batch"`). Unknown flags Invalid
    `"memcpy flags"`. Decode identity unchanged. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

320. [x] `cudaMemcpy3DBatchAsync` / `cudaMemcpy3DWithAttributesAsync`:
    [`memcpy_3d_batch_async`](gpu-sim/src/sim.rs) is pointer-to-pointer
    [`MemcpyOp::is_3d`] only (other extents Invalid `"memcpy3d batch depth"`;
    CUDA arrays are not modeled). `ops.len() == attrs.len()`. `flags` must
    be `0` (reserved; Invalid `"memcpy3d batch flags"`). Intra-batch copies
    share a snapshotted stream-order predecessor list or empty
    DuringApiCall/Any deps like PLAN 319. Later submits wait for the whole
    batch. DuringApiCall waits those copies before return. Any does not.
    [`memcpy_3d_with_attributes`](gpu-sim/src/sim.rs) Stream is
    [`memcpy_3d_async`](gpu-sim/src/sim.rs). Capture cannot include a 3D
    batch (`"cannot capture memcpy3d batch"`). Decode identity unchanged.
    `gpu-profile capture` is still refused. Dual score still has no `$/M
    tokens`.

321. [x] `cudaMemPrefetchBatchAsync` / `cudaMemDiscardBatchAsync` /
    `cudaMemDiscardAndPrefetchBatchAsync`:
    [`prefetch_batch_async`](gpu-sim/src/sim.rs) /
    [`discard_batch_async`](gpu-sim/src/sim.rs) /
    [`discard_and_prefetch_batch_async`](gpu-sim/src/sim.rs) require
    `cudaDevAttrConcurrentManagedAccess` on every GPU. This VM reports `0`,
    so they are Invalid `"concurrent managed access"`. Typed
    [`prefetch`](gpu-sim/src/sim.rs) / [`prefetch_with_flags`](gpu-sim/src/sim.rs)
    stay (no CMA gate). Discard contents / Host NUMA prefetch locations are
    not modeled. Decode identity unchanged. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

322. [x] `cudaLaunchAttributeCooperative`:
    [`KernelNodeAttr::Cooperative`](gpu-sim/src/ops.rs) is
    `cudaGraphKernelNodeGetAttribute` / `SetAttribute` / `CopyAttributes`
    for cooperative launch. Typed
    [`graph_kernel_node_get_cooperative`](gpu-sim/src/sim.rs) /
    [`graph_kernel_node_set_cooperative`](gpu-sim/src/sim.rs) stay.
    Setting `true` requires [`GpuProfile::cooperative_launch`] and occupies
    every Hyper-Q slot at launch like
    [`cooperative_kernel`](gpu-sim/src/sim.rs). Definition Set does not
    retarget exec.
    [`graph_exec_kernel_set_params`](gpu-sim/src/sim.rs) still refuses a
    cooperative mismatch (`"cooperative is topology"`). Capture cannot
    include Set. Decode identity unchanged. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

323. [x] `cudaMemcpyBatchAsync` expert prefetch:
    [`GpuStoreCfg::memcpy_batch`](expertvm/src/gpu_store.rs) /
    [`SimCfg::memcpy_batch`](expertvm/src/sim_replay.rs) fill a
    multi-expert pinned/VMM prefetch window with
    [`memcpy_batch_async`](gpu-sim/src/sim.rs) so sibling H2D copies share
    one stream-order snapshot (copy engines together). Demand acquire
    stays sequential. A same-stream [`alloc`](gpu-sim/src/sim.rs)
    (`cudaMallocAsync`) pointer does not need a host sync. Illegal with
    pageable, host-sync, mapped, or managed fills. `--memcpy-batch` on
    `expertvm sim` / `schedule` / `store` and `gguf_gemv engine --expert-sim`.
    Decode identity stays sequential `memcpy_pinned_to_device`.
    `gpu-profile capture` is still refused. Dual score still has no `$/M
    tokens`.

324. [x] `cudaGraphNodeSetEnabled` combo reuse:
    [`GpuStoreCfg::graph_enable`](expertvm/src/gpu_store.rs) /
    [`SimCfg::graph_enable`](expertvm/src/sim_replay.rs) capture a wide
    walker combo parent of resident expert leaves, then
    [`graph_node_set_enabled`](gpu-sim/src/sim.rs) skips extra children on
    a later subset instead of instantiating a new parent. Child index is
    not cover order. Implies `--cuda-graphs`. Illegal with
    `--device-launch` (that path already splits combos to singles). Store
    GEMM stays per-leaf. `--graph-enable` on `expertvm sim` / `schedule` /
    `store` and `gguf_gemv engine --expert-sim`. Decode identity stays
    exact combo recapture. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

325. [x] `cudaLaunchAttributeMemSyncDomain` decode-stream isolation:
    [`GpuStoreCfg::mem_sync_domain`](expertvm/src/gpu_store.rs) /
    [`SimCfg::mem_sync_domain`](expertvm/src/sim_replay.rs) set
    [`set_stream_mem_sync_domain`](gpu-sim/src/sim.rs) on the decode
    compute stream (prefill stays Default).
    [`MemSyncDomain::Remote`](gpu-sim/src/ops.rs) isolates leftover
    prefill [`same_domain_fence_permille`](gpu-sim/src/profile.rs) when
    `--decode-priority` puts decode GEMMs on a second stream. Default is
    Default (decode identity). Engine `--mem-sync-domain remote` implies
    `--decode-priority`. Walker does not (same as `--decode-sms`). Store
    leaf graph replay SetAttributes the exec node to the launch stream
    when they disagree (CUDA graphs bake capture-time domain).
    `--mem-sync-domain` on `expertvm sim` / `schedule` / `store` and
    `gguf_gemv engine --expert-sim`. Profile tax default is 0.
    `gpu-profile capture` is still refused. Dual score still has no `$/M
    tokens`.

326. [x] `cudaLaunchAttributeLaunchCompletionEvent` replica overlap:
    [`GpuStoreCfg::launch_completion`](expertvm/src/gpu_store.rs) /
    [`SimCfg::launch_completion`](expertvm/src/sim_replay.rs) attach
    [`kernel_with`](gpu-sim/src/sim.rs) /
    [`graph_kernel_node_set_launch_completion`](gpu-sim/src/sim.rs) on
    grouped expert GEMMs so other streams may `wait_event` at kernel
    *start*. Store [`pin_hot`](expertvm/src/gpu_store.rs) replica D2D on
    `n_gpus >= 2` waits that event on the copy stream instead of draining
    the GEMM, so leftover compute overlaps the replica. Illegal with
    `--device-launch`. `--launch-completion` on `expertvm sim` /
    `schedule` / `store` and `gguf_gemv engine --expert-sim`. Decode
    identity stays no launch-completion event. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

327. [x] `cuStreamWaitValue64` / `cuStreamWriteValue64` copy-ready:
    [`GpuStoreCfg::wait_value`](expertvm/src/gpu_store.rs) /
    [`SimCfg::wait_value`](expertvm/src/sim_replay.rs) allocate an 8-byte
    device mailbox per expert page (not the weight alloc) and
    [`write_value64`](gpu-sim/src/sim.rs) generation 1 after H2D / prefetch
    instead of `record_event`. The mailbox is `cudaMallocAsync` on the copy
    stream; that stream is waited *before* H2D so the pointer is resident
    for a compute [`wait_value64`](gpu-sim/src/sim.rs) during DMA
    (`cudaMalloc` would `synchronize_device` and drain leftover prefill).
    Store GEMM [`wait_value64`](gpu-sim/src/sim.rs)
    Eq on the compute stream; replica D2D waits that mailbox on the copy
    stream. Walker wait/write are live stream ops (GEMM graphs stay
    kernel-only). `--wait-value` on `expertvm sim` / `schedule` / `store`
    and `gguf_gemv engine --expert-sim`. Decode identity stays CUDA events.
    `gpu-profile capture` is still refused. Dual score still has no `$/M
    tokens`.

328. [x] `cudaMemPoolTrimTo` idle return:
    [`GpuStoreCfg::mempool_trim`](expertvm/src/gpu_store.rs) /
    [`SimCfg::mempool_trim`](expertvm/src/sim_replay.rs) hold unused
    `cudaMallocAsync` bytes (`u64::MAX` release threshold) then
    [`pool_trim_to`](gpu-sim/src/sim.rs) `0` on every GPU's current
    device mempool after [`SimulatedGpuStore::score`](expertvm/src/gpu_store.rs)
    / walker / schedule finish. Implies `--mempool`. Illegal with
    `--sync-alloc` (`cudaMalloc` is not a mempool). Hits/misses and
    [`hbm_peak`](gpu-sim/src/lib.rs) stay the same; token ITL (`clock_ns`) does not
    trim. `--mempool-trim` on `expertvm sim` / `schedule` / `store` and
    `gguf_gemv engine --expert-sim`. Decode identity stays CUDA's default
    threshold 0 (free already returns). `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

329. [x] `cudaMemPoolReuseAllowOpportunistic=0` skip-reuse:
    [`GpuStoreCfg::mempool_no_reuse`](expertvm/src/gpu_store.rs) /
    [`SimCfg::mempool_no_reuse`](expertvm/src/sim_replay.rs) set
    [`MemPoolAttr::ReuseAllowOpportunistic`](gpu-sim/src/ops.rs) `0` on every
    GPU's current device mempool after the `u64::MAX` hold. Leftover cache
    stays reserved; the next miss is an OS alloc (`alloc_overhead_ns`) and
    charges extra HBM. Implies `--mempool`. Illegal with `--sync-alloc`.
    Hits/misses stay the same. `--mempool-no-reuse` on `expertvm sim` /
    `schedule` / `store` and `gguf_gemv engine --expert-sim`. Decode identity
    stays opportunistic reuse (CUDA default 1). `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

330. [x] `cudaLaunchAttributeProgrammaticEvent` replica overlap:
    [`GpuStoreCfg::programmatic_event`](expertvm/src/gpu_store.rs) /
    [`SimCfg::programmatic_event`](expertvm/src/sim_replay.rs) attach
    [`kernel_with`](gpu-sim/src/sim.rs)
    [`KernelAttrs::programmatic_event`](gpu-sim/src/ops.rs) /
    [`graph_kernel_node_set_programmatic_event`](gpu-sim/src/sim.rs) on
    grouped expert GEMMs so other streams may `wait_event` at the PDL
    trigger (`pdl_trigger_permille`) instead of kernel completion. Store
    [`pin_hot`](expertvm/src/gpu_store.rs) replica D2D on `n_gpus >= 2`
    waits that event on the copy stream instead of draining the GEMM, so
    leftover compute overlaps the replica. Implies a PDL trigger on those
    GEMMs (same-stream PDL wait stays `--pdl`). Illegal with
    `--device-launch`. `--programmatic-event` on `expertvm sim` /
    `schedule` / `store` and `gguf_gemv engine --expert-sim`. Decode
    identity stays no programmatic event. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

331. [x] `cudaStreamAttachMemAsync` Single on managed experts:
    [`GpuStoreCfg::stream_attach`](expertvm/src/gpu_store.rs) /
    [`SimCfg::stream_attach`](expertvm/src/sim_replay.rs) call
    [`stream_attach`](gpu-sim/src/sim.rs) `MemAttach::Single` on the
    compute stream after managed alloc+advise, then prefetch on that
    stream so GEMM stays legal under Single. Identity managed prefetch
    stays on the copy stream (overlaps leftover compute). Implies
    `--managed`. Illegal with `--seq-streams` (Single is one stream;
    seq-streams put walker GEMMs on per-sequence streams including NULL).
    `--stream-attach` on `expertvm sim` / `schedule` / `store` and
    `gguf_gemv engine --expert-sim`. Decode identity stays Global attach
    + copy-stream prefetch. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

332. [x] `cudaMallocManaged(..., cudaMemAttachHost)` then Global attach:
    [`GpuStoreCfg::managed_host`](expertvm/src/gpu_store.rs) /
    [`SimCfg::managed_host`](expertvm/src/sim_replay.rs) call
    [`alloc_managed_host`](gpu-sim/src/sim.rs) then
    [`stream_attach`](gpu-sim/src/sim.rs) `MemAttach::Global` on the copy
    stream so device prefetch is legal (Host attach fails device prefetch /
    kernels with `not attached`). Identity managed is Global at alloc (no
    Attach op). Implies `--managed`. Prefetch stays on the copy stream and
    still overlaps leftover compute unless `--stream-attach` (Host alloc then
    Single on compute). `--managed-host` on `expertvm sim` / `schedule` /
    `store` and `gguf_gemv engine --expert-sim`. Decode identity stays Global
    alloc + copy-stream prefetch. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

333. [x] `cudaHostRegister` pageable staging then pinned DMA:
    [`GpuStoreCfg::host_register`](expertvm/src/gpu_store.rs) /
    [`SimCfg::host_register`](expertvm/src/sim_replay.rs) `alloc_host` then
    [`host_register`](gpu-sim/src/sim.rs) so miss H2D is
    `memcpy_pinned_to_device` (not host-sync pageable). Implies `--pageable`.
    Illegal with `--mapped` / `--managed` (and therefore `--memcpy-batch`).
    Hits/misses stay the same. `--host-register` on `expertvm sim` /
    `schedule` / `store` and `gguf_gemv engine --expert-sim`. Decode identity
    stays `cudaMallocHost` staging. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

334. [x] `cudaMemPrefetchAsync(..., cudaCpuDeviceId)` on managed LRU evict:
    [`GpuStoreCfg::prefetch_host`](expertvm/src/gpu_store.rs) /
    [`SimCfg::prefetch_host`](expertvm/src/sim_replay.rs) call
    [`prefetch_host`](gpu-sim/src/sim.rs) instead of `free_sync` so the
    `cudaMallocManaged` allocation stays live on the host. The next miss
    prefetches the same pointer back (no second alloc). Implies `--managed`.
    Hits/misses stay the same; extra host↔device bytes move on thrash.
    `--prefetch-host` on `expertvm sim` / `schedule` / `store` and
    `gguf_gemv engine --expert-sim`. Decode identity stays `free_sync` on
    evict. `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

335. [x] `cudaHostRegisterMapped` expert pages:
    [`GpuStoreCfg::host_register_mapped`](expertvm/src/gpu_store.rs) /
    [`SimCfg::host_register_mapped`](expertvm/src/sim_replay.rs)
    `alloc_host` then [`host_register_mapped`](gpu-sim/src/sim.rs) so miss
    pages are pin-and-map (`cudaHostRegisterMapped`) instead of
    `cudaHostAllocMapped`. Implies `--mapped`. Illegal with
    `--host-register` (unmapped staging). Hits/misses and `hbm_peak` 0 stay
    the same; evict is `host_unregister` then `free_host`. `--host-register-mapped`
    on `expertvm sim` / `schedule` / `store` and
    `gguf_gemv engine --expert-sim`. Decode identity stays
    `cudaHostAllocMapped`. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

336. [x] `cuPointerSetAttribute` SyncMemops on miss pages:
    [`GpuStoreCfg::sync_memops`](expertvm/src/gpu_store.rs) /
    [`SimCfg::sync_memops`](expertvm/src/sim_replay.rs)
    [`pointer_set_attribute`](gpu-sim/src/sim.rs) [`PointerAttr::SyncMemops`]
    after device alloc, before H2D / managed prefetch, so memcpy of that
    pointer is host-synchronous (`synchronize_stream` after submit). Hits
    stay the same; leftover compute cannot overlap that copy. Illegal with
    `--mapped` (no device memcpy) and `--memcpy-batch` (needs async H2D).
    Does not imply pageable or `--sync-alloc`. `--sync-memops` on
    `expertvm sim` / `schedule` / `store` and
    `gguf_gemv engine --expert-sim`. Decode identity stays async
    `memcpy_pinned_to_device`. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

337. [x] `cudaSetDeviceFlags(cudaDeviceSyncMemops)` on every GPU:
    [`GpuStoreCfg::device_sync_memops`](expertvm/src/gpu_store.rs) /
    [`SimCfg::device_sync_memops`](expertvm/src/sim_replay.rs)
    [`set_device_flags`](gpu-sim/src/sim.rs) [`DeviceFlags::SYNC_MEMOPS`]
    after `Sim::new`, so every runtime memcpy/memset on that device is
    host-synchronous (including unmarked pointers). Distinct from per-page
    `--sync-memops`. Hits stay the same; leftover compute cannot overlap
    those copies. Illegal with `--mapped` and `--memcpy-batch`. Does not
    imply pageable or `--sync-alloc`. `--device-sync-memops` on
    `expertvm sim` / `schedule` / `store` and
    `gguf_gemv engine --expert-sim`. Decode identity stays async
    `memcpy_pinned_to_device`. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

338. [x] `cudaFuncSetAttribute` PreferredSharedMemoryCarveout MaxShared:
    [`GpuStoreCfg::func_max_shared`](expertvm/src/gpu_store.rs) /
    [`SimCfg::func_max_shared`](expertvm/src/sim_replay.rs)
    [`set_func_carveout`](gpu-sim/src/sim.rs) [`SharedMemCarveout::MaxShared`]
    after `Sim::new`. Launch Default inherits that occupancy so leftover
    kernels cannot Hyper-Q overlap. Distinct from launch-attribute
    `--max-shared`. Hits stay the same. Legal with `--pdl` and
    `--cooperative`. `--func-max-shared` on `expertvm sim` / `schedule` /
    `store` and `gguf_gemv engine --expert-sim`. Decode identity stays
    Default function carveout. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

339. [x] `cudaCtxResetPersistingL2Cache` after each grouped GEMM:
    [`GpuStoreCfg::l2_reset`](expertvm/src/gpu_store.rs) /
    [`SimCfg::l2_reset`](expertvm/src/sim_replay.rs)
    [`reset_persisting_l2_cache`](gpu-sim/src/sim.rs) after live GEMM (not
    inside capture). Implies `--l2-persist`. Hits stay the same; a reused
    expert does not keep persisting L2 lines. `--l2-reset` on `expertvm sim`
    / `schedule` / `store` and `gguf_gemv engine --expert-sim`. Decode
    identity stays no reset. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

340. [x] `cudaFuncSetSharedMemConfig` function bank width:
    [`GpuStoreCfg::func_shared_mem`](expertvm/src/gpu_store.rs) /
    [`SimCfg::func_shared_mem`](expertvm/src/sim_replay.rs)
    [`set_func_shared_mem_config`](gpu-sim/src/sim.rs) after `Sim::new`.
    Launch Default inherits that duration scale. Distinct from
    launch-attribute `--shared-mem` (launch FourByte / EightByte still
    override). Hits stay the same. Legal with `--pdl` and `--cooperative`.
    `--func-shared-mem` on `expertvm sim` / `schedule` / `store` and
    `gguf_gemv engine --expert-sim`. Decode identity stays Default function
    config. `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

341. [x] `cudaFuncSetAttribute` ClusterSchedulingPolicyPreference Spread:
    [`GpuStoreCfg::func_cluster_spread`](expertvm/src/gpu_store.rs) /
    [`SimCfg::func_cluster_spread`](expertvm/src/sim_replay.rs)
    [`set_func_cluster_policy`](gpu-sim/src/sim.rs)
    [`ClusterSchedulingPolicy::Spread`] after `Sim::new`. Launch Default
    inherits that occupancy so leftover kernels cannot Hyper-Q overlap when
    `--cluster` is at least 2. Distinct from launch-attribute
    `--cluster-spread`. Hits stay the same. Legal with `--pdl` and
    `--cooperative`. `--func-cluster-spread` on `expertvm sim` /
    `schedule` / `store` and `gguf_gemv engine --expert-sim`. Decode
    identity stays Default function policy. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

342. [x] `cudaDeviceSetSharedMemConfig` device bank width:
    [`GpuStoreCfg::device_shared_mem`](expertvm/src/gpu_store.rs) /
    [`SimCfg::device_shared_mem`](expertvm/src/sim_replay.rs)
    [`set_shared_mem_config`](gpu-sim/src/sim.rs) after `Sim::new`. Launch
    Default inherits that duration scale when function config is also Default.
    Distinct from `--func-shared-mem` and launch-attribute `--shared-mem`
    (launch FourByte / EightByte still override; function FourByte / EightByte
    still override). Hits stay the same. Legal with `--pdl` and
    `--cooperative`. `--device-shared-mem` on `expertvm sim` / `schedule` /
    `store` and `gguf_gemv engine --expert-sim`. Decode identity stays Default
    device config. `gpu-profile capture` is still refused. Dual score still has
    no `$/M tokens`.

343. [x] `cudaEventBlockingSync` on copy start/end events:
    [`GpuStoreCfg::event_blocking_sync`](expertvm/src/gpu_store.rs)
    [`create_event_blocking_sync`](gpu-sim/src/sim.rs) when timing copy events.
    Implies `--timing-events`. [`synchronize_event`] pays
    [`host_sync_blocking_ns`] instead of the recording stream's policy.
    Distinct from `--sync-policy blocking` (that taxes `synchronize_stream`).
    Hits stay the same. `--event-blocking-sync` on `expertvm store` and
    `gguf_gemv engine --expert-sim`. Walker `sim` / `schedule` do not create
    timing events. Decode identity stays `cudaEventDisableTiming`.
    `gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

344. [x] `cudaFuncAttributeClusterDimMustBeSet`:
    [`GpuStoreCfg::cluster_must_set`](expertvm/src/gpu_store.rs) /
    [`SimCfg::cluster_must_set`](expertvm/src/sim_replay.rs)
    [`set_cluster_dim_must_be_set`](gpu-sim/src/sim.rs) after `Sim::new`.
    Needs `--cluster` (a grouped GEMM without cluster is Invalid
    `"cluster dim must be set"`). Occupancy matches `--cluster`
    (SetAttribute is +1 ns). Hits stay the same. Legal with `--pdl` and
    `--cooperative`. `--cluster-must-set` on `expertvm sim` / `schedule` /
    `store` and `gguf_gemv engine --expert-sim`. Decode identity stays unset.
    `gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

345. [x] `cudaLimitMaxL2FetchGranularity`:
    [`GpuStoreCfg::l2_fetch`](expertvm/src/gpu_store.rs) /
    [`SimCfg::l2_fetch`](expertvm/src/sim_replay.rs)
    [`set_limit`](gpu-sim/src/sim.rs) `MaxL2FetchGranularity` after `Sim::new`.
    Implies `--l2-persist`. `32` / `64` / `128` only (default unset is 128).
    Access-policy windows must align; 64-byte persist is Invalid until
    `--l2-fetch 32`. Hits stay the same for 4096-byte experts. Legal with
    `--pdl` and `--cooperative`. `--l2-fetch N` on `expertvm sim` /
    `schedule` / `store` and `gguf_gemv engine --expert-sim`. Decode
    identity stays 128. `gpu-profile capture` is still refused. Dual score
    still has no `$/M tokens`.

346. [x] `cudaSetDeviceFlags` schedule (`cudaDeviceScheduleSpin` /
    `Yield` / `BlockingSync`):
    [`GpuStoreCfg::device_sync_policy`](expertvm/src/gpu_store.rs) /
    [`SimCfg::device_sync_policy`](expertvm/src/sim_replay.rs)
    [`set_device_flags`](gpu-sim/src/sim.rs) after `Sim::new`. Auto streams
    inherit the schedule as host-wait tax. Explicit `--sync-policy` wins.
    ORs with `--device-sync-memops` (`SYNC_MEMOPS` stays). Auto skips
    `set_device_flags` (decode identity). Hits stay the same. Legal with
    `--pdl` and `--cooperative`. `--device-sync-policy auto|spin|yield|blocking`
    on `expertvm sim` / `schedule` / `store` and
    `gguf_gemv engine --expert-sim`. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

347. [x] `cudaFuncAttributeRequiredClusterWidth`:
    [`GpuStoreCfg::required_cluster`](expertvm/src/gpu_store.rs) /
    [`SimCfg::required_cluster`](expertvm/src/sim_replay.rs)
    [`set_required_cluster_width`](gpu-sim/src/sim.rs) after `Sim::new`.
    Needs `--cluster` and must equal `--cluster` X. Occupancy matches
    `--cluster` (SetAttribute is +1 ns). Distinct from `--cluster-must-set`
    (bool: some cluster dim must be present) and `--preferred-cluster`.
    Hits stay the same. Legal with `--pdl` and `--cooperative`. Legal
    together with `--cluster-must-set` when both match `--cluster`.
    `--required-cluster N` on `expertvm sim` / `schedule` / `store` and
    `gguf_gemv engine --expert-sim`. Decode identity stays unset (`0`).
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

348. [x] `cudaLaunchAttributeMemSyncDomainMap` collapse (remote→0):
    [`GpuStoreCfg::mem_sync_collapse`](expertvm/src/gpu_store.rs) /
    [`SimCfg::mem_sync_collapse`](expertvm/src/sim_replay.rs)
    [`set_stream_mem_sync_domain_map`](gpu-sim/src/sim.rs) on the decode
    stream. Needs `--mem-sync-domain remote`. Collapse `{default: 0,
    remote: 0}` restores leftover prefill `same_domain_fence_permille`
    (Hopper identity is remote→1; SetAttribute does not tick the clock).
    Distinct from `--mem-sync-domain` (logical domain vs physical mapping).
    Hits stay the same. Legal with `--pdl` and `--cooperative`.
    `--mem-sync-map identity|collapse` on `expertvm sim` / `schedule` /
    `store` and `gguf_gemv engine --expert-sim`. infer-bench has no
    `--mem-sync-domain`, so it does not get `--mem-sync-map`. Decode
    identity stays Hopper identity. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

349. [x] `cudaLaunchAttributePreferredSharedMemoryCarveout` MaxL1:
    [`GpuStoreCfg::max_l1`](expertvm/src/gpu_store.rs) /
    [`SimCfg::max_l1`](expertvm/src/sim_replay.rs) launch carveout.
    Needs `--func-max-shared`. Overrides function MaxShared so leftover
    kernels can Hyper-Q overlap again. Exclusive with `--max-shared`.
    Hits stay the same. Legal with `--pdl` and `--cooperative`.
    `--max-l1` on `expertvm sim` / `schedule` / `store` and
    `gguf_gemv engine --expert-sim`. infer-bench has `--func-max-shared`,
    so it gets `--max-l1`. Decode identity stays Default. `gpu-profile
    capture` is still refused.     Dual score still has no `$/M tokens`.

350. [x] `cudaLaunchAttributeClusterSchedulingPolicyPreference` LoadBalancing:
    [`GpuStoreCfg::cluster_load_balance`](expertvm/src/gpu_store.rs) /
    [`SimCfg::cluster_load_balance`](expertvm/src/sim_replay.rs) launch
    policy. Needs `--func-cluster-spread`. Overrides function Spread so
    leftover kernels can Hyper-Q overlap again. Exclusive with
    `--cluster-spread`. Hits stay the same. Legal with `--pdl` and
    `--cooperative`. `--cluster-load-balance` on `expertvm sim` /
    `schedule` / `store` and `gguf_gemv engine --expert-sim`. infer-bench
    has `--func-cluster-spread`, so it gets `--cluster-load-balance`.
    Decode identity stays Default. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

351. [x] `cudaAccessPolicyWindow.hitRatio`:
    [`GpuStoreCfg::l2_ratio`](expertvm/src/gpu_store.rs) /
    [`SimCfg::l2_ratio`](expertvm/src/sim_replay.rs) launch window ratio.
    Implies `--l2-persist`. `1..=1000` (unset is 1000). A partial ratio bills
    more HBM than full persist on a reused expert (no extra construction tick).
    Hits stay the same. Legal with `--pdl` and `--cooperative`.
    `--l2-ratio N` on `expertvm sim` / `schedule` / `store` and
    `gguf_gemv engine --expert-sim`. infer-bench has `--l2-persist`, so it
    gets `--l2-ratio`. Decode identity stays 1000. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

352. [x] `cudaLaunchAttributeMemSyncDomain` launch Remote:
    [`GpuStoreCfg::mem_sync_launch`](expertvm/src/gpu_store.rs) /
    [`SimCfg::mem_sync_launch`](expertvm/src/sim_replay.rs) launch attr on
    grouped GEMMs. Needs `--mem-sync-domain remote`. Overrides prefill
    inherit-Default so leftover prefill shares the decode Remote domain and
    `same_domain_fence_permille` returns (Hopper identity tax is 0;
    SetAttribute does not tick the clock). Distinct from `--mem-sync-map`
    (logical launch domain vs physical mapping). Hits stay the same. Legal
    with `--pdl` and `--cooperative`. `--mem-sync-launch` on `expertvm sim`
    / `schedule` / `store` and `gguf_gemv engine --expert-sim`. infer-bench
    has no `--mem-sync-domain`, so it does not get `--mem-sync-launch`.
    Decode identity stays inherit-stream. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

353. [x] `cudaLaunchAttributeMemSyncDomainMap` launch collapse:
    [`GpuStoreCfg::mem_sync_launch_map`](expertvm/src/gpu_store.rs) /
    [`SimCfg::mem_sync_launch_map`](expertvm/src/sim_replay.rs) launch attr on
    grouped GEMMs. Needs `--mem-sync-domain remote`. Overrides prefill
    inherit-identity so leftover prefill maps Default→0 with decode Remote→0
    and `same_domain_fence_permille` returns (Hopper identity tax is 0;
    SetAttribute does not tick the clock). Distinct from `--mem-sync-launch`
    (physical mapping vs logical domain) and `--mem-sync-map` (launch vs
    stream). Hits stay the same. Legal with `--pdl` and `--cooperative`.
    `--mem-sync-launch-map` on `expertvm sim` / `schedule` / `store` and
    `gguf_gemv engine --expert-sim`. infer-bench has no `--mem-sync-domain`,
    so it does not get `--mem-sync-launch-map`. Decode identity stays
    inherit-stream. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

354. [x] `cudaMemPoolProps::maxSize`:
    [`GpuStoreCfg::mempool_max`](expertvm/src/gpu_store.rs) /
    [`SimCfg::mempool_max`](expertvm/src/sim_replay.rs) `cudaMemPoolCreate` +
    `maxSize` then `cudaDeviceSetMemPool`. Implies `--mempool`. Illegal with
    `--sync-alloc`. Hits stay the same when `N` fits; reserved `live+cached`
    cannot grow past `N` (leftover cache plus `--mempool-no-reuse` OS alloc
    OOMs). Distinct from `--mempool` (hold), `--mempool-trim` (idle
    `TrimTo(0)`), and `--mempool-no-reuse` (skip opportunistic reuse).
    `--mempool-max N` on `expertvm sim` / `schedule` / `store` and
    `gguf_gemv engine --expert-sim`. infer-bench has no `--mempool`, so it
    does not get `--mempool-max`. Decode identity stays the device default
    pool (`0` unset). `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

355. [x] `cudaMemcpySrcAccessOrderDuringApiCall`:
    [`GpuStoreCfg::memcpy_during`](expertvm/src/gpu_store.rs) /
    [`SimCfg::memcpy_during`](expertvm/src/sim_replay.rs) on `--memcpy-batch`
    H2D. The batch API waits those copies before return (not the whole
    stream). Needs `--memcpy-batch`. Hits stay the same. Distinct from
    `--memcpy-batch` (batch vs sequential H2D vs During wait). Legal with
    `--pdl` and `--cooperative`. Illegal with the same combos as
    `--memcpy-batch` once both are set. `--memcpy-during` on `expertvm sim`
    / `schedule` / `store` and `gguf_gemv engine --expert-sim`. infer-bench
    has no `--memcpy-batch`, so it does not get `--memcpy-during`. Decode
    identity stays Stream order (or sequential H2D). `gpu-profile capture`
    is still refused. Dual score still has no `$/M tokens`.

356. [x] `cudaMemcpySrcAccessOrderAny`:
    [`GpuStoreCfg::memcpy_any`](expertvm/src/gpu_store.rs) /
    [`SimCfg::memcpy_any`](expertvm/src/sim_replay.rs) on `--memcpy-batch`
    H2D. Empty intra-batch deps and no API wait (copies stay in flight).
    Needs `--memcpy-batch`. Exclusive with `--memcpy-during`. Hits stay the
    same. Distinct from `--memcpy-during` (wait vs no wait) and
    `--memcpy-batch` Stream (empty deps vs stream-order snapshot). Legal
    with `--pdl` and `--cooperative`. `--memcpy-any` on `expertvm sim` /
    `schedule` / `store` and `gguf_gemv engine --expert-sim`. infer-bench
    has no `--memcpy-batch`, so it does not get `--memcpy-any`. Decode
    identity stays Stream order (or sequential H2D). `gpu-profile capture`
    is still refused. Dual score still has no `$/M tokens`.

357. [x] `cudaAccessPropertyStreaming` persist-window hits:
    [`GpuStoreCfg::l2_streaming`](expertvm/src/gpu_store.rs) /
    [`SimCfg::l2_streaming`](expertvm/src/sim_replay.rs) on `--l2-persist`
    GEMM windows. Needs persist (`--l2-persist` / reset / fetch / ratio).
    Does not imply persist. Hits stay the same; a reused expert bills full
    HBM (no persist fill). Distinct from `--l2-ratio` (ratio of Persisting
    window vs hit property) and `--l2-reset` (fill then clear). Legal with
    `--pdl` and `--cooperative`. `--l2-streaming` on `expertvm sim` /
    `schedule` / `store` and `gguf_gemv engine --expert-sim`. infer-bench
    has `--l2-persist`, so it gets `--l2-streaming`. Decode identity stays
    Persisting hits. `gpu-profile capture` is still refused. Dual score
    still has no `$/M tokens`.

358. [x] `cudaStreamBeginCaptureToGraph` dependency array:
    [`GpuStoreCfg::graph_capture_deps`](expertvm/src/gpu_store.rs) /
    [`SimCfg::graph_capture_deps`](expertvm/src/sim_replay.rs) on
    `--graph-piecewise` combo parents. Later capture fragments depend on
    the previous last node (`numDependencies > 0`) so sibling expert GEMMs
    serialize. Needs `--graph-piecewise`. Does not imply piecewise. Hits
    stay the same. Distinct from `--graph-piecewise` (empty deps / extra
    roots vs chained deps) and `--graph-build` (no `graph_add_dependencies`
    edges). Legal with `--pdl` and `--cooperative`. `--graph-capture-deps`
    on `expertvm sim` / `schedule` / `store` and
    `gguf_gemv engine --expert-sim`. infer-bench has no `--graph-piecewise`,
    so it does not get `--graph-capture-deps`. Decode identity stays empty
    deps (Hyper-Q overlap when `compute_slots >= 2`). Store GEMM stays
    per-leaf. `gpu-profile capture` is still refused. Dual score still has
    no `$/M tokens`.

359. [x] `cudaGraphAddDependencies` combo edges:
    [`GpuStoreCfg::graph_build_deps`](expertvm/src/gpu_store.rs) /
    [`SimCfg::graph_build_deps`](expertvm/src/sim_replay.rs) on
    `--graph-build` combo parents. Later `graph_add_child` nodes depend on
    the previous child so sibling expert GEMMs serialize. Needs
    `--graph-build`. Does not imply graph-build. Hits stay the same.
    Distinct from `--graph-build` (no edges vs chained) and
    `--graph-capture-deps` (`cudaStreamBeginCaptureToGraph` deps). Legal
    with `--pdl` and `--cooperative`. `--graph-build-deps` on `expertvm sim`
    / `schedule` / `store` and `gguf_gemv engine --expert-sim`. infer-bench
    has no `--graph-build`, so it does not get `--graph-build-deps`. Decode
    identity stays no edges (Hyper-Q overlap when `compute_slots >= 2`).
    Store GEMM stays per-leaf. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

360. [x] `cudaGraphAddHostNode` between combo children:
    [`GpuStoreCfg::graph_host`](expertvm/src/gpu_store.rs) /
    [`SimCfg::graph_host`](expertvm/src/sim_replay.rs) on
    `--graph-build` combo parents. A host node sits BETWEEN later
    `graph_add_child` nodes (`child → host → child`) so sibling expert
    GEMMs serialize through `host_func_ns`. Needs `--graph-build`. Does
    not imply graph-build or `--host-func`. Hits stay the same. Distinct
    from `--graph-build` (overlap vs serial+tax), `--graph-build-deps`
    (serialize without host tax), and live `--host-func` (after the token's
    GEMMs, not between combo children). Legal with `--pdl` and
    `--cooperative`. `--graph-host` on `expertvm sim` / `schedule` /
    `store` and `gguf_gemv engine --expert-sim`. infer-bench has no
    `--graph-build`, so it does not get `--graph-host`. Decode identity
    stays no host nodes in combo parents. Store GEMM stays per-leaf.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

361. [x] `cudaGraphAddMemsetNode` / `cudaMemsetAsync` of graph-mem scratch:
    [`GpuStoreCfg::graph_memset`](expertvm/src/gpu_store.rs) /
    [`SimCfg::graph_memset`](expertvm/src/sim_replay.rs) on `--graph-mem`
    leaf GEMM graphs. A memset sits BETWEEN scratch alloc and the GEMM
    (`alloc → memset → kernel → free`) so launch bills an extra HBM write.
    Needs `--graph-mem`. Does not imply graph-mem. Hits stay the same.
    Distinct from `--graph-mem` (scratch alloc+free vs extra memset of that
    scratch). Does not work with `--graph-auto-free` alone. Legal with
    `--pdl` and `--cooperative`. `--graph-memset` on `expertvm sim` /
    `schedule` / `store` and `gguf_gemv engine --expert-sim`. infer-bench
    has no `--graph-mem`, so it does not get `--graph-memset`. Decode
    identity stays kernel-only graphs (no scratch, no memset). Store leaf
    GEMMs memset too (not walker-only). `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

362. [x] `cudaGraphAddMemcpyNode` / `cudaMemcpyAsync` H2D of graph-mem scratch:
    [`GpuStoreCfg::graph_memcpy`](expertvm/src/gpu_store.rs) /
    [`SimCfg::graph_memcpy`](expertvm/src/sim_replay.rs) on `--graph-mem`
    leaf GEMM graphs. A memcpy sits BETWEEN scratch alloc and the GEMM
    (`alloc → memcpy → kernel → free`, or `alloc → memset → memcpy →
    kernel → free` with `--graph-memset`) so launch bills copy-engine
    PCIe. Needs `--graph-mem`. Does not imply graph-mem or `--graph-memset`.
    Hits stay the same. Distinct from `--graph-mem` (scratch alloc+free vs
    extra H2D of that scratch) and `--graph-memset` (compute HBM write vs
    copy-engine). Does not work with `--graph-auto-free` alone. Legal with
    `--pdl` and `--cooperative`. `--graph-memcpy` on `expertvm sim` /
    `schedule` / `store` and `gguf_gemv engine --expert-sim`. infer-bench
    has no `--graph-mem`, so it does not get `--graph-memcpy`. Decode
    identity stays kernel-only graphs (no scratch, no memcpy). Store leaf
    GEMMs memcpy too (not walker-only). `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

363. [x] Captured `cudaLaunchHostFunc` BETWEEN piecewise combo fragments:
    [`GpuStoreCfg::graph_capture_host`](expertvm/src/gpu_store.rs) /
    [`SimCfg::graph_capture_host`](expertvm/src/sim_replay.rs) on
    `--graph-piecewise` combo parents. A host callback sits BETWEEN later
    `begin_capture_to_graph` fragments (`child → host → child`) so sibling
    expert GEMMs serialize through `host_func_ns`. Needs `--graph-piecewise`.
    Does not imply piecewise or `--host-func`. Hits stay the same. Distinct
    from `--graph-piecewise` (overlap vs serial+tax), `--graph-capture-deps`
    (serialize without host tax), and `--graph-host` (`cudaGraphAddHostNode`
    on graph-build). Legal with `--pdl` and `--cooperative`.
    `--graph-capture-host` on `expertvm sim` / `schedule` / `store` and
    `gguf_gemv engine --expert-sim`. infer-bench has no `--graph-piecewise`,
    so it does not get `--graph-capture-host`. Decode identity stays no host
    nodes in combo parents. Store GEMM stays per-leaf. `gpu-profile capture`
    is still refused. Dual score still has no `$/M tokens`.

364. [x] `cudaGraphAddHostNode` / captured `cudaLaunchHostFunc` BEFORE the leaf GEMM:
    [`GpuStoreCfg::graph_leaf_host`](expertvm/src/gpu_store.rs) /
    [`SimCfg::graph_leaf_host`](expertvm/src/sim_replay.rs) on leaf GEMM graphs.
    A host callback sits BEFORE the kernel (`host → kernel`, or
    `alloc → [memset] → [memcpy] → host → kernel → [free]` with `--graph-mem`)
    so each leaf launch bills `host_func_ns`. Implies `--cuda-graphs`. Does not
    imply `--host-func`, `--graph-host`, or `--graph-capture-host`. Hits stay
    the same. Distinct from `--host-func` (live after the token GEMM),
    `--graph-host` (`cudaGraphAddHostNode` BETWEEN graph-build combo children),
    and `--graph-capture-host` (captured host BETWEEN piecewise fragments).
    Illegal with `--device-launch` (CUDA cannot device-launch host nodes).
    Legal with `--pdl` and `--cooperative`. `--graph-leaf-host` on
    `expertvm sim` / `schedule` / `store` and `gguf_gemv engine --expert-sim`.
    infer-bench has no CUDA-graph leaf construction, so it does not get
    `--graph-leaf-host`. Decode identity stays kernel-only graphs. Store leaf
    GEMMs host too (not walker-only). `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

365. [x] `cudaGraphClone` of combo parents (recursive children):
    [`GpuStoreCfg::graph_clone_parent`](expertvm/src/gpu_store.rs) /
    [`SimCfg::graph_clone_parent`](expertvm/src/sim_replay.rs) on walker combo
    parents (`ids.len() >= 2`). Clone, destroy the src, then instantiate the
    copy so the parent tree is independent of GraphBank leaves (`graph_clone_ns`
    per graph in that tree, including nested children). Does not imply
    `--graph-clone` or `--cuda-graphs`. Hits stay the same. Distinct from
    `--graph-clone` (leaves vs combo parents). Legal with `--pdl` and
    `--cooperative`. `--graph-clone-parent` on `expertvm sim` / `schedule` /
    `store` and `gguf_gemv engine --expert-sim`. infer-bench has no combo
    graph construction, so it does not get `--graph-clone-parent`. Decode
    identity stays instantiate-in-place. Store GEMM stays per-leaf.
    `gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

366. [x] `cudaMemsetAsync` miss fill instead of pinned H2D:
    [`GpuStoreCfg::memset_fill`](expertvm/src/gpu_store.rs) /
    [`SimCfg::memset_fill`](expertvm/src/sim_replay.rs) on pinned/VMM miss
    pages. Fill is `cudaMemsetAsync` (`memset_buf` / `memset_sync` with
    `--sync-alloc`): HBM write, compute occupancy (`Share::Solo`), not copy-
    engine PCIe. Hits stay the same. Distinct from `--graph-memset` (in-graph
    scratch vs miss page) and default pinned H2D. Illegal with `--mapped`,
    `--managed`, `--pageable` (covers `--host-register`), and `--memcpy-batch`.
    Legal with `--pdl`, `--cooperative`, `--vmm`, `--sync-alloc`, and
    `--graph-build`. Does not imply `--cuda-graphs`. Replica `pin_hot` D2D
    stays memcpy. `--memset-fill` on `expertvm sim` / `schedule` / `store`
    and `gguf_gemv engine --expert-sim`. infer-bench has no expert H2D fill,
    so it does not get `--memset-fill`. Decode identity stays pinned H2D
    (inner ExpertStore still holds real weights). Store and walker (not
    walker-only). `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

367. [x] Live `cudaLaunchHostFunc` after miss DMA / prefetch:
    [`GpuStoreCfg::copy_host`](expertvm/src/gpu_store.rs) /
    [`SimCfg::copy_host`](expertvm/src/sim_replay.rs) enqueue
    [`gpu_sim::Sim::host_func`](gpu-sim/src/sim.rs) on the DMA stream after
    pinned/VMM H2D, `cudaMemsetAsync` miss fill, or managed prefetch, before
    copy-ready (event or wait/write-value). Bills `host_func_ns` so GEMM waits
    include the host callback. Does not imply `--host-func`. Hits stay the
    same. Distinct from `--host-func` (after every token GEMM, including
    hits). Mapped misses have no device fill (no-op). Replica `pin_hot` D2D
    stays memcpy-only. Legal with `--pdl`, `--cooperative`, `--wait-value`,
    and `--memset-fill`. Does not imply `--cuda-graphs`. Graphs stay
    kernel-only. `--copy-host` on `expertvm sim` / `schedule` / `store` and
    `gguf_gemv engine --expert-sim`. infer-bench has no expert miss DMA, so
    it does not get `--copy-host`. Decode identity stays no host after copy.
    Store and walker (not walker-only). `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

368. [x] Graph-build `cudaGraphSetConditional` node + exec SetParams:
    [`graph_add_set_conditional`](gpu-sim/src/sim.rs) is the graph-build analog
    of captured [`set_conditional`](gpu-sim/src/sim.rs) (`cudaGraphSetConditional`).
    Handle is topology; `value` is a parameter
    ([`graph_exec_set_conditional_params`](gpu-sim/src/sim.rs) /
    [`GraphNodeParams::SetConditional`](gpu-sim/src/ops.rs)). Device-launch
    instantiate refuses the node (conditionals). Hits/misses unchanged.
    Decode identity does not add set-conditional nodes. `--graph-if` stays
    for a later Engine twin (launch still resets handles to create-time
    default first). `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

369. [x] `cudaGraphAddIf` combo reuse:
    [`GpuStoreCfg::graph_if`](expertvm/src/gpu_store.rs) /
    [`SimCfg::graph_if`](expertvm/src/sim_replay.rs) wrap walker combo
    children in [`graph_add_if`](gpu-sim/src/sim.rs) +
    [`graph_add_set_conditional`](gpu-sim/src/sim.rs) (create-time handle
    default 0; node value 1) so a later subset retargets extras with
    [`graph_exec_set_conditional_params`](gpu-sim/src/sim.rs) instead of
    instantiating a new parent. Pays `graph_set_params_ns` and clears upload
    (next launch re-uploads). Needs `--graph-build`. Does not imply
    graph-build. Illegal with `--device-launch` (conditionals) and
    `--graph-enable` (SetEnabled does not clear upload). Store GEMM stays
    per-leaf. Hits stay the same. `--graph-if` on `expertvm sim` /
    `schedule` / `store` and `gguf_gemv engine --expert-sim`. infer-bench
    has no combo graph construction, so it does not get `--graph-if`.
    Decode identity stays exact combo recapture (no IF nodes).
    `gpu-profile capture` is still refused. Dual score still has no `$/M
    tokens`.

370. [x] Skip `cudaMemAdviseSetReadMostly` (`UnsetReadMostly`):
    [`GpuStoreCfg::no_read_mostly`](expertvm/src/gpu_store.rs) /
    [`SimCfg::no_read_mostly`](expertvm/src/sim_replay.rs) call
    [`MemAdvise::UnsetReadMostly`](gpu-sim/src/ops.rs) at managed fill
    instead of SetReadMostly so dest prefetch **moves** (one GPU copy;
    lower `hbm_peak`) instead of replicating. Implies `--managed`. Hits
    stay the same. Distinct from `--accessed-by` (dest GEMM without a
    second copy; pin skips dest prefetch). Keep SetPreferredLocation.
    `--no-read-mostly` on `expertvm sim` / `schedule` / `store` and
    `gguf_gemv engine --expert-sim`. infer-bench has managed fill via
    `--managed-host`, so it gets `--no-read-mostly`. Decode identity stays
    SetReadMostly. Store and walker (not walker-only). Dest replica drop is
    a no-op when the page is not ReadMostly (prefetch already moved).
    `gpu-profile capture` is still refused. Dual score still has no `$/M
    tokens`.

371. [x] Skip `cudaMemAdviseSetPreferredLocation` (`UnsetPreferredLocation`):
    [`GpuStoreCfg::no_preferred`](expertvm/src/gpu_store.rs) /
    [`SimCfg::no_preferred`](expertvm/src/sim_replay.rs) call
    [`MemAdvise::UnsetPreferredLocation`](gpu-sim/src/ops.rs) at managed
    fill instead of SetPreferredLocation so a remote GEMM first-touches
    (weight migrate onto the compute GPU) instead of staying on home.
    Implies `--managed`. Hits stay the same. Distinct from `--accessed-by`
    (dest GEMM without migrating) and `--no-read-mostly` (prefetch move vs
    replicate). Keep SetReadMostly unless `--no-read-mostly` is also set.
    `--no-preferred` on `expertvm sim` / `schedule` / `store` and
    `gguf_gemv engine --expert-sim`. infer-bench has managed fill via
    `--managed-host` and `--place remote`, so it gets `--no-preferred`.
    Decode identity stays SetPreferredLocation. Store and walker (not
    walker-only). Do not invent `SetPreferredLocationHost` as a decode-path
    skip of kernel first-touch. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

372. [x] Skip fill `cudaMemPrefetchAsync` (`--no-mem-prefetch`):
    [`GpuStoreCfg::no_mem_prefetch`](expertvm/src/gpu_store.rs) /
    [`SimCfg::no_mem_prefetch`](expertvm/src/sim_replay.rs) skip
    [`gpu_sim::Sim::prefetch`](gpu-sim/src/sim.rs) at managed miss fill so
    the first GEMM first-touches on compute instead of copy-engine prefetch
    overlapping leftover GEMM. Implies `--managed`. Hits stay the same.
    Distinct from `--prefetch none` (predictor) and `--prefetch-host` (evict
    to host). Keep replica dest prefetch and host-restore prefetch. Keep
    SetReadMostly / SetPreferredLocation unless those flags are also set.
    `--no-mem-prefetch` on `expertvm sim` / `schedule` / `store` and
    `gguf_gemv engine --expert-sim`. infer-bench has managed fill via
    `--managed-host`, so it gets `--no-mem-prefetch`. Decode identity stays
    fill prefetch. Store and walker (not walker-only). Do not invent a
    second handshake skip: copy-ready may still record. `gpu-profile
    capture` is still refused. Dual score still has no `$/M tokens`.

373. [x] `cudaMemcpyWithAttributesAsync` demand miss fill (`--memcpy-attr`):
    [`GpuStoreCfg::memcpy_attr`](expertvm/src/gpu_store.rs) /
    [`SimCfg::memcpy_attr`](expertvm/src/sim_replay.rs) use
    [`gpu_sim::Sim::memcpy_with_attributes`](gpu-sim/src/sim.rs)
    [`MemcpySrcAccessOrder::DuringApiCall`](gpu-sim/src/ops.rs) for demand
    pinned/VMM miss H2D so the API waits that copy (synchronize the DMA stream
    first so empty deps do not race `cudaMallocAsync`). Hits stay the same.
    Does not imply `--memcpy-batch`. Distinct from `--memcpy-during` (batch
    prefetch DuringApiCall) and `--memcpy-any` (empty deps, no API wait).
    Replica `pin_hot` D2D stays memcpy. Illegal with `--mapped` / `--managed` /
    `--pageable` / `--memset-fill` / `--sync-alloc` / `--sync-memops` /
    `--device-sync-memops`. Legal with `--pdl` and `--cooperative`.
    `--memcpy-attr` on `expertvm sim` / `schedule` / `store` and
    `gguf_gemv engine --expert-sim`. infer-bench has no expert H2D fill, so
    it does not get `--memcpy-attr`. Decode identity stays
    `memcpy_pinned_to_device`. Store and walker (not walker-only).
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

374. [x] `cudaMemcpyAsync` Device→HostPinned before pinned/VMM LRU free
    (`--d2h-evict`): [`GpuStoreCfg::d2h_evict`](expertvm/src/gpu_store.rs) /
    [`SimCfg::d2h_evict`](expertvm/src/sim_replay.rs) use
    [`gpu_sim::Sim::memcpy_device_to_pinned`](gpu-sim/src/sim.rs) on the
    copy stream before `cudaFreeAsync` / `va_release` / `free_sync` so evict
    pays extra PCIe. Hits stay the same; the next miss still fills from
    catalog staging. Distinct from `--prefetch-host` (managed keep-alloc).
    Illegal with `--mapped` / `--managed`. Legal with `--vmm`, `--pageable`,
    `--memset-fill`, `--sync-alloc`, `--pdl`, and `--cooperative`.
    `--d2h-evict` on `expertvm sim` / `schedule` / `store` and
    `gguf_gemv engine --expert-sim`. infer-bench has no pinned/VMM expert
    evict of this form, so it does not get `--d2h-evict`. Decode identity
    stays free with no D2H. Store and walker (not walker-only).
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

375. [x] `cudaMemcpyAsync` Device→Host (pageable) before pinned/VMM LRU free
    (`--d2h-pageable`): [`GpuStoreCfg::d2h_pageable`](expertvm/src/gpu_store.rs) /
    [`SimCfg::d2h_pageable`](expertvm/src/sim_replay.rs) use
    [`gpu_sim::Sim::memcpy_device_to_host`](gpu-sim/src/sim.rs) on the copy
    stream before `cudaFreeAsync` / `va_release` / `free_sync` so evict pays
    extra host-synchronous bounce-buffer PCIe. Hits stay the same; the next
    miss still fills from catalog staging. Distinct from `--d2h-evict`
    (Device→HostPinned overlapping DMA) and `--prefetch-host` (managed
    keep-alloc). Implies `--pageable`. Illegal with `--mapped` / `--managed`
    / `--host-register` / `--d2h-evict`. Legal with `--vmm`, `--sync-alloc`,
    `--pdl`, and `--cooperative`. `--d2h-pageable` on `expertvm sim` /
    `schedule` / `store` and `gguf_gemv engine --expert-sim`. infer-bench
    has no pinned/VMM expert evict of this form, so it does not get
    `--d2h-pageable`. Decode identity stays free with no D2H. Store and
    walker (not walker-only). `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

376. [x] `cudaHostUnregister` after miss DMA (`--host-unregister`):
    [`GpuStoreCfg::host_unregister`](expertvm/src/gpu_store.rs) /
    [`SimCfg::host_unregister`](expertvm/src/sim_replay.rs) call
    [`gpu_sim::Sim::host_unregister`](gpu-sim/src/sim.rs) after each miss
    DMA so staging is pageable between misses and the next miss re-registers
    (`synchronize` plus `first_alloc_ns` pin tax). Hits stay the same.
    Distinct from `--host-register` (keep registered for lifetime). Implies
    `--host-register` (which implies `--pageable`). Illegal with `--mapped`
    / `--managed` / `--host-register-mapped` / `--d2h-pageable`. Legal with
    `--vmm`, `--sync-alloc`, `--pdl`, and `--cooperative`. `--host-unregister`
    on `expertvm sim` / `schedule` / `store` and `gguf_gemv engine --expert-sim`.
    infer-bench has no pageable staging of this form, so it does not get
    `--host-unregister`. Decode identity keeps staging registered. Store and
    walker (not walker-only). `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

377. [x] `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` of each miss
    `cudaMalloc` (`--ipc`): [`GpuStoreCfg::ipc`](expertvm/src/gpu_store.rs) /
    [`SimCfg::ipc`](expertvm/src/sim_replay.rs) call
    [`gpu_sim::Sim::ipc_get`](gpu-sim/src/sim.rs) then
    [`ipc_open`](gpu-sim/src/sim.rs) after each pinned miss fill so the page
    holds a live alias (no extra HBM) and [`ipc_close`](gpu-sim/src/sim.rs)
    before `cudaFree` (`alloc_overhead_ns` on first get and every open).
    Hits stay the same. Distinct from `--shareable` (POSIX-FD mempool IPC).
    Implies `--sync-alloc`. Illegal with `--mapped` / `--managed` / `--vmm`
    / `--shareable`. Legal with `--pdl`, `--cooperative`, `--pageable`,
    `--host-register`, `--host-unregister`, `--d2h-evict`, `--d2h-pageable`,
    `--memset-fill`, and `--wait-value`. `--ipc` on `expertvm sim` /
    `schedule` / `store` and `gguf_gemv engine --expert-sim`. infer-bench
    has no `cudaMalloc` expert fill of this form, so it does not get
    `--ipc`. Decode identity GEMMs on the `cudaMalloc` pointer with no IPC
    handshake. Store and walker (not walker-only). `gpu-profile capture`
    is still refused. Dual score still has no `$/M tokens`.

378. [x] `cudaMemPoolExportPointer` / `cudaMemPoolImportPointer` of each miss
    `cudaMallocAsync` (`--share-ptr`): [`GpuStoreCfg::share_ptr`](expertvm/src/gpu_store.rs) /
    [`SimCfg::share_ptr`](expertvm/src/sim_replay.rs) call
    [`gpu_sim::Sim::pool_export_ptr`](gpu-sim/src/sim.rs) then
    [`pool_import_ptr`](gpu-sim/src/sim.rs) after each pinned miss fill so
    the page holds a live alias (no extra HBM) and `cudaFreeAsync` of the
    import before the source (`alloc_overhead_ns` on first export and every
    import). Hits stay the same. Distinct from `--shareable` (POSIX-FD
    pool-level IPC) and `--ipc` (`cudaIpcGetMemHandle` of `cudaMalloc`).
    Implies `--shareable` (which implies `--mempool`). Illegal with `--ipc`
    / `--mapped` / `--managed` / `--vmm` / `--sync-alloc`. Legal with
    `--pdl`, `--cooperative`, `--pageable`, `--memcpy-batch`, and
    `--wait-value`. `--share-ptr` on `expertvm sim` / `schedule` / `store`
    and `gguf_gemv engine --expert-sim`. infer-bench has no shareable pool
    expert fill of this form, so it does not get `--share-ptr`. Decode
    identity is `--shareable` without a per-page pointer handshake. Store
    and walker (not walker-only). `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

379. [x] `cuMemRetainAllocationHandle` after each VMM miss map (`--vmm-retain`):
    [`GpuStoreCfg::vmm_retain`](expertvm/src/gpu_store.rs) /
    [`SimCfg::vmm_retain`](expertvm/src/sim_replay.rs) call
    [`gpu_sim::Sim::va_retain_handle`](gpu-sim/src/sim.rs) at offset 0 after
    each VMM miss `va_acquire` / `va_acquire_paged` so the page holds a live
    handle and `va_release_handle` before `va_release` (`alloc_overhead_ns`
    on every retain). Hits stay the same. Distinct from `--vmm` (combined
    `va_map` without a handle) and `--vmm-page` (split maps; retain still
    one handle at offset 0). Implies `--vmm`. Illegal with `--mapped` /
    `--managed`. Legal with `--pdl`, `--cooperative`, `--pageable`,
    `--memset-fill`, `--wait-value`, `--d2h-evict`, `--d2h-pageable`,
    `--vmm-page`, `--multicast`, and `--accessed-by`. `--vmm-retain` on
    `expertvm sim` / `schedule` / `store`, `gguf_gemv engine --expert-sim`,
    and `infer-bench schedule` (implies `--vmm`; infer-bench has no
    standalone `--vmm`). Decode identity is `--vmm` without
    `cuMemRetainAllocationHandle`. Store and walker (not walker-only).
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

380. [x] `cuMemCreate` plus `cuMemMap` instead of combined `va_acquire` (`--vmm-handle`):
    [`GpuStoreCfg::vmm_handle`](expertvm/src/gpu_store.rs) /
    [`SimCfg::vmm_handle`](expertvm/src/sim_replay.rs) idle-or-reserve a VA
    ([`gpu_sim::Sim::va_reserve_idle`](gpu-sim/src/sim.rs)) then
    [`gpu_sim::Sim::va_create`](gpu-sim/src/sim.rs) plus
    [`gpu_sim::Sim::va_map_handle`](gpu-sim/src/sim.rs) at offset 0 so the
    page holds a live handle and `va_release_handle` before `va_release`
    (`alloc_overhead_ns` on create and map). Hits stay the same. Distinct
    from `--vmm` (combined `va_map` without a handle) and `--vmm-retain`
    (promote after acquire). Implies `--vmm`. Illegal with `--mapped` /
    `--managed` / `--vmm-retain`. Legal with `--pdl`, `--cooperative`,
    `--pageable`, `--memset-fill`, `--wait-value`, `--d2h-evict`,
    `--d2h-pageable`, `--vmm-page`, `--multicast`, and `--accessed-by`.
    `--vmm-handle` on `expertvm sim` / `schedule` / `store`,
    `gguf_gemv engine --expert-sim`, and `infer-bench schedule` (implies
    `--vmm`; infer-bench has no standalone `--vmm`). Decode identity is
    `--vmm` without `cuMemCreate` plus `cuMemMap`. Store and walker (not
    walker-only). `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

381. [x] Official Gemma4 (`architecture=gemma4`): writer-built
    `tiny-gemma4` plus decode of llama.cpp `src/models/gemma4.cpp` —
    Gemma embed-scale + GeGLU, QK-Norm before RoPE, unweighted RMSNorm on
    V, attention scale `1.0`, RMSNorm `post_attention_norm` / `ffn_norm` /
    `post_ffw_norm`, required `attention.sliding_window`, convert
    `attention.sliding_window_pattern` as a per-layer bool array, required
    `attention.key_length_swa` / `value_length_swa` and
    `embedding_length_per_layer_input`. Optional final tanh logit softcap
    (default 0). Writer-tiny is dense (no `ffn_gate_inp`), writes
    `embedding_length_per_layer_input = 0` (no per-layer embeddings), omits
    `attention.shared_kv_layers` (every layer has KV), uses equal SWA/global
    head dims, and `attention.sliding_window = 2` so short-seq tests clip.
    Convert skips `lm_head.weight` when tied, omits `attn_logit_softcapping`
    / `rope.dimension_count`. Convert `norm_shift` is 0. `{arch}.context_length`
    is still unused for KV sizing. Gemma4 MoE, per-layer embeddings, shared
    KV, and mixed SWA/global head dims stay refused with named keys.
    Decode identity vs oracle. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

382. [x] Official Gemma4 MoE (`architecture=gemma4` with `ffn_gate_inp`):
    writer-built `tiny-gemma4-moe` plus decode of llama.cpp
    `src/models/gemma4.cpp` when `ffn_gate_inp` is present. Same
    `architecture=gemma4` as 381 (not a second arch). Shared dense GeGLU
    (`ffn_gate` / `ffn_up` / `ffn_down`) after `ffn_norm(attn_out)`, then
    RMSNorm `ffn_post_norm_1`. Expert input is RMSNorm `ffn_pre_norm_2` of
    `attn_out`. Custom router operates on `attn_out` (unweighted RMSNorm,
    scale `1/sqrt(n_embd)`, `ffn_gate_inp.scale`, then `ffn_gate_inp` GEMM),
    not `ffn_norm` / expert input. `build_moe_ffn` softmax then top-k with
    `norm_w` clamp `2^-14` and GELU experts (not SILU). Then RMSNorm
    `ffn_post_norm_2` and add into the shared MLP. Caller still applies
    `post_ffw_norm` and the residual onto `attn_out`. Writer-tiny keeps
    dense `tiny-gemma4` unchanged; MoE writes `expert_count=4`,
    `expert_used_count=2`, `expert_feed_forward_length=n_ff`, separate
    `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps` (fused
    `ffn_gate_up_exps` is refused with that tensor name). PLE / shared-KV /
    mixed SWA/global head dims stay refused with named keys. Decode identity
    vs oracle. ExpertAccess traces plus DirectStore identity.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

383. [x] Official Gemma4 PLE (`architecture=gemma4` with
    `embedding_length_per_layer_input > 0`): same arch as dense/MoE gemma4,
    not a second family. Decode follows llama.cpp `src/models/gemma4.cpp`
    when `n_embd_per_layer > 0`: `build_inp_per_layer` (token table scaled
    by `sqrt(n_embd_per_layer)`), `project_per_layer_inputs`
    (`mm(per_layer_model_proj, inpL)` scaled by `1/sqrt(n_embd)`, RMSNorm
    `per_layer_proj_norm`, add, scale `1/sqrt(2)`), then after the FFN
    residual `gelu(mm(inp_gate, cur)) * slice(il)`, `mm(proj)`, RMSNorm
    `post_norm`, residual add. No AltUp / Laurel. Writer-tiny keeps dense
    `tiny-gemma4` and MoE `tiny-gemma4-moe` at `n_embd_per_layer=0`; PLE
    writes `embedding_length_per_layer_input=64` plus
    `per_layer_token_embd` / `per_layer_model_proj` / `per_layer_proj_norm`
    and per-layer `inp_gate` / `proj` / `post_norm`. Shared-KV / mixed
    SWA/global head dims stay refused with named keys. Decode identity vs
    oracle. `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

384. [x] Official Gemma4 MoE+PLE (`architecture=gemma4` with `ffn_gate_inp`
    and `embedding_length_per_layer_input > 0`): production E2B/E4B shape on
    the same arch as dense/MoE/PLE-only gemma4, not a second family. Decode
    is PLAN 382 MoE then PLAN 383 PLE inject after the FFN residual.
    Writer-tiny `tiny-gemma4-moe-ple` (`n_expert=4`, `n_expert_used=2`,
    `n_embd_per_layer=64`). Dense `tiny-gemma4`, MoE `tiny-gemma4-moe`, and
    PLE `tiny-gemma4-ple` stay. Fused `ffn_gate_up_exps`, shared-KV, and
    mixed SWA/global head dims stay refused with named keys. Decode identity
    vs oracle. ExpertAccess traces plus DirectStore identity.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

385. [x] Official Gemma4 fused `ffn_gate_up_exps` (`architecture=gemma4` with
    fused experts instead of separate `ffn_gate_exps` / `ffn_up_exps`): same
    arch as dense/MoE/PLE gemma4, not a second family. Decode follows
    llama.cpp `src/models/gemma4.cpp` `build_moe_ffn`: one GEMV of twice
    `n_ff` rows, then split gate then up, then GELU. DirectStore catalogs
    fused bytes in `ExpertParts.gate` with empty `up`. Writer-tiny
    `tiny-gemma4-moe-fused` (`n_expert=4`, `n_expert_used=2`). Split
    `tiny-gemma4-moe`, PLE, and MoE plus PLE stay. Shared-KV / mixed
    SWA/global head dims stay refused with named keys. Decode identity vs
    oracle. ExpertAccess traces plus DirectStore identity.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

386. [x] Official Gemma4 fused plus PLE (`architecture=gemma4` with
    `ffn_gate_up_exps` and `embedding_length_per_layer_input > 0`): production
    E2B/E4B packing on the same arch as fused-only and split MoE plus PLE,
    not a second family. Decode is PLAN 385 fused experts then PLAN 383 PLE
    inject after the FFN residual. Writer-tiny `tiny-gemma4-moe-fused-ple`
    (`n_expert=4`, `n_expert_used=2`, `n_embd_per_layer=64`). Split
    `tiny-gemma4-moe-ple` and fused-only `tiny-gemma4-moe-fused` stay.
    Shared-KV / mixed SWA/global head dims stay refused with named keys.
    Decode identity vs oracle. ExpertAccess traces plus DirectStore identity.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

387. [x] Official Gemma4 shared KV (`architecture=gemma4` with
    `attention.shared_kv_layers > 0`): same arch as dense/MoE/PLE/fused
    gemma4, not a second family. Decode follows llama.cpp
    `src/models/gemma4.cpp` `has_kv(il)` plus `llama_model::create_memory`
    reuse: `n_layer_kv_from_start = n_layer - shared_kv_layers`; layers
    `il >= n_from_start` skip K/V project/store (`wk` / `wv` /
    `attn_k_norm` are `TENSOR_NOT_REQUIRED`) and `build_attn` against the
    donor `n_from_start` minus 2 (SWA) or minus 1 (global). Nonzero shared
    KV requires `n_from_start >= 2` (llama.cpp `GGML_ASSERT`). Writer-tiny
    `tiny-gemma4-shared-kv` is three dense layers, all SWA,
    `shared_kv_layers=1`, so layer 2 reuses layer 0. 1-layer tinies still
    omit the key (every layer has KV). Mixed SWA/global head dims stay
    refused with named keys. Decode identity vs oracle.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

388. [x] Official Gemma4 mixed SWA/global head dims (`architecture=gemma4`
    with `key_length_swa != key_length`): same arch as dense/MoE/PLE/fused/
    shared-KV gemma4, not a second family. Decode follows llama.cpp
    `hparams.n_embd_head_k(il)` / `n_rot(il)`: SWA uses
    `attention.key_length_swa` / `rope.dimension_count_swa`, global uses
    `attention.key_length` (default `n_embd / n_head`) and the full
    `rope.dimension_count`. KV cache stride is the max head dim; SWA
    stores into the prefix of each slot. Writer-tiny
    `tiny-gemma4-mixed-hd` is two dense layers, SWA then global,
    `key_length_swa` half of `key_length`. 1-layer tinies stay equal-hd
    all-SWA. Shared KV stays `tiny-gemma4-shared-kv`. `k_swa != v_swa`
    still refused with named keys. Optional `wv` (K as V), `out_scale`,
    `rope_freqs`, and `rope.freq_base_swa` stay omitted. Decode identity
    vs oracle. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

389. [x] Official Gemma4 SWA RoPE base (`architecture=gemma4` with
    `rope.freq_base_swa`): same arch as dense/MoE/PLE/fused/shared-KV/
    mixed-hd gemma4, not a second family. Decode follows llama.cpp
    `get_rope_freq_base(il)`: SWA uses `rope.freq_base_swa` (default
    `10000` when omitted), global uses `rope.freq_base`. Writer-tiny
    `tiny-gemma4-swa-base` is the mixed-hd tensors with E2B/E4B convert
    values (`1000000` vs `10000`). Mixed-hd global QK-Norm is `0.1` so
    Gemma4 attn-scale `1.0` does not saturate softmax onto the unrotated
    self key. 1-layer tinies omit the SWA key so both layers keep
    `10000`. Optional `wv` (K as V), `out_scale`, and `rope_freqs` stay
    omitted. Decode identity vs oracle. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

390. [x] Official Gemma4 omitted `wv` (K as V): same arch as
    dense/MoE/PLE/fused/shared-KV/mixed-hd/SWA-base gemma4, not a second
    family. Decode follows llama.cpp `src/models/gemma4.cpp`: `wv` is
    `TENSOR_NOT_REQUIRED`; when missing, `Vcur = Kcur` before the
    unweighted V RMSNorm (K still gets `attn_k_norm` and RoPE). Writer-tiny
    `tiny-gemma4-no-wv` is dense `tiny-gemma4` without `attn_v`. Mixed-hd /
    SWA-base keep `wv`. Optional `out_scale` and `rope_freqs` stay omitted.
    Decode identity vs oracle. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

391. [x] Official Gemma4 `layer_output_scale`: same arch as
    dense/MoE/PLE/fused/shared-KV/mixed-hd/SWA-base/no-wv gemma4, not a
    second family. Decode follows llama.cpp `src/models/gemma4.cpp`:
    `out_scale` is `TENSOR_NOT_REQUIRED`; when present, `ggml_mul`
    broadcasts the length-1 tensor onto the residual after FFN (and PLE
    inject). Writer-tiny `tiny-gemma4-out-scale` is dense `tiny-gemma4`
    plus `blk.0.layer_output_scale` at `0.5`. Optional `rope_freqs` stay
    omitted. Decode identity vs oracle. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

392. [x] Official Gemma4 `rope_freqs` (proportional RoPE): same arch as
    dense/MoE/PLE/fused/shared-KV/mixed-hd/SWA-base/no-wv/out-scale gemma4,
    not a second family. Decode follows llama.cpp `src/models/gemma4.cpp`:
    non-SWA layers pass `rope_freqs.weight` (`{n_embd_head/2}`) into
    `ggml_rope_ext`; ggml `ggml_rope_cache_init` uses `theta/ff`. SWA keeps
    `freq_factors` null. Writer-tiny `tiny-gemma4-rope-freqs` is mixed-hd
    plus convert `generate_extra_tensors` packing (`1` then `1e30` for the
    unrotated NeoX pairs, `partial_rotary_factor=0.25`). Mixed-hd /
    SWA-base / no-wv / out-scale omit the tensor. Decode identity vs
    oracle. `gpu-profile capture` is still refused. Dual score still has
    no `$/M tokens`.

393. [x] Official Gemma4 `ffn_down_exps.scale` (per-expert down scale): same
    arch as dense/MoE/PLE/fused/shared-KV/mixed-hd/SWA-base/no-wv/out-scale/
    rope-freqs gemma4, not a second family. Decode follows llama.cpp
    `build_lora_mm_id`: after the down expert GEMM, `ggml_mul` broadcasts
    `ffn_down_exps.scale[e]` onto that expert's output
    (`TENSOR_NOT_REQUIRED`; missing is `1.0`). Writer-tiny
    `tiny-gemma4-moe-down-s` is split `tiny-gemma4-moe` plus `{n_expert}` F32.
    Writer-tiny zeros `ffn_gate_inp` so softmax is uniform and packs split
    experts as F32 (Q4_K GELU(gate) is ~0 on the tiny, and `ffn_post_norm_2`
    RMSNorm would cancel a one-hot mix). Fused/PLE/shared-KV/mixed-hd omit
    the tensor. Decode identity vs oracle. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

394. [x] CUDA green contexts (`cuDeviceGetDevResource` /
    `cuDevSmResourceSplitByCount` / `cuDevResourceGenerateDesc` /
    `cuGreenCtxCreate` / `cuGreenCtxStreamCreate` / `cuGreenCtxSetStream` /
    `cuStreamGetGreenCtx`): SM resources are ‰ of the chip, not occupancy
    SM counts. `set_stream_sm_permille` stays duration-only (compute-bound
    kernels scale; memory-bound keep full HBM; does not partition Hyper-Q).
    Complementary green contexts may overlap kernels even when
    `compute_slots` is 1. Same-span contexts still share exclusive compute.
    Capture cannot include create / desc / bind / destroy. Flags 0 only
    (no `CU_GREEN_CTX_DEFAULT_STREAM`, no split coscheduling bits).
    `expertvm sim` / `schedule` / `store` / Engine `--green-ctx` binds decode
    vs leftover prefill (implies `--decode-priority`; default 500/500;
    `--decode-sms N` with leftover SMs; `--decode-sms 1000` refused). Distinct
    from `--decode-sms` alone (duration scale, exclusive occupancy still
    serializes leftover prefill). Off by default keeps full-chip exclusive
    compute (decode identity). `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

395. [x] Official Gemma4 `ffn_gate_exps.scale` / `ffn_up_exps.scale` (per-expert
    gate/up scale): same arch as dense/MoE/PLE/fused/shared-KV/mixed-hd/
    SWA-base/no-wv/out-scale/rope-freqs/down-s gemma4, not a second family.
    Decode follows llama.cpp `build_lora_mm_id`: after the gate/up expert
    GEMM, `ggml_mul` broadcasts `ffn_{gate,up}_exps.scale[e]` onto that
    expert's output (`TENSOR_NOT_REQUIRED`; missing is `1.0`). Fused
    `ffn_gate_up_exps` applies `up_exps_s` to both halves and ignores
    `gate_exps_s`. Writer-tiny `tiny-gemma4-moe-gate-up-s` is split
    `tiny-gemma4-moe` plus `{n_expert}` F32. Writer-tiny zeros `ffn_gate_inp`
    so softmax is uniform and packs split experts as F32 (Q4_K GELU(gate) is
    ~0 on the tiny, and `ffn_post_norm_2` RMSNorm would cancel a one-hot
    mix). Omits `ffn_down_exps.scale`. Fused/PLE/shared-KV/mixed-hd omit the
    tensors. Decode identity vs oracle. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

396. [x] Official dense `{1}` `ffn_down.scale`: llama.cpp generic post-load
    (`TENSOR_NOT_REQUIRED`; missing is `1.0`). Decode follows `build_ffn`:
    `ggml_mul` after the down GEMM on every `DenseFfn` (llama / gemma / qwen
    dense and gemma4 shared MLP). Writer-tiny `tiny-llama-ffn-down-s` is
    `tiny-llama` plus scale `0.5`. Gemma2/3/4 `post_ffw_norm` RMSNorm would
    cancel a positive scalar, so the observable fixture is llama, not a
    second gemma4 family. Phi2/Bloom stay on `Phi2Ffn`. Decode identity vs
    oracle. `gpu-profile capture` is still refused. Dual score still has
    no `$/M tokens`.

397. [x] Official dense `{1}` `attn_q.scale`: llama.cpp generic post-load
    (`TENSOR_NOT_REQUIRED`; missing is `1.0`). Decode follows `build_lora_mm`:
    `ggml_mul` after the Q GEMM, before QK-Norm. Writer-tiny
    `tiny-llama-attn-q-s` is `tiny-llama` plus scale `2.0`. Gemma3/4 QK-Norm
    would cancel a positive scalar, so the observable fixture is llama, not
    a second gemma4 family. Decode identity vs oracle. `gpu-profile capture`
    is still refused. Dual score still has no `$/M tokens`.

398. [x] Official dense `{1}` `attn_output.scale`: llama.cpp generic post-load
    (`TENSOR_NOT_REQUIRED`; missing is `1.0`). Decode follows `build_lora_mm`:
    `ggml_mul` after the output GEMM, before `post_attention_norm`. Writer-tiny
    `tiny-llama-attn-out-s` is `tiny-llama` plus scale `1.5`. Gemma2/3/4
    `post_attention_norm` RMSNorm would cancel a positive scalar, so the
    observable fixture is llama, not a second gemma4 family. Decode identity
    vs oracle. `gpu-profile capture` is still refused. Dual score still has
    no `$/M tokens`.

399. [x] Official dense `{1}` `attn_k.scale`: llama.cpp generic post-load
    (`TENSOR_NOT_REQUIRED`; missing is `1.0`). Decode follows `build_lora_mm`:
    `ggml_mul` after the K GEMM, before K-norm. Writer-tiny
    `tiny-llama-attn-k-s` is `tiny-llama` plus scale `0.75`. Gemma3/4
    K-norm would cancel a positive scalar, so the observable fixture is
    llama, not a second gemma4 family. Decode identity vs oracle.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

400. [x] Official dense `{1}` `attn_v.scale`: llama.cpp generic post-load
    (`TENSOR_NOT_REQUIRED`; missing is `1.0`). Decode follows `build_lora_mm`:
    `ggml_mul` after the V GEMM, before V RMSNorm. Writer-tiny
    `tiny-llama-attn-v-s` is `tiny-llama` plus scale `2.5`. Gemma3/4
    unweighted V RMSNorm would cancel a positive scalar, so the observable
    fixture is llama, not a second gemma4 family. Decode identity vs oracle.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

401. [x] Official dense `{1}` `ffn_gate.scale`: llama.cpp generic post-load
    (`TENSOR_NOT_REQUIRED`; missing is `1.0`). Decode follows `build_ffn`:
    `ggml_mul` after the gate GEMM, before SiLU / GELU. Writer-tiny
    `tiny-llama-ffn-gate-s` is `tiny-llama` plus scale `1.25`. SiLU is
    nonlinear, so the walk stays observable. Decode identity vs oracle.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

402. [x] Official dense `{1}` `ffn_up.scale`: llama.cpp generic post-load
    (`TENSOR_NOT_REQUIRED`; missing is `1.0`). Decode follows `build_ffn`:
    `ggml_mul` after the up GEMM, before the gated product. Writer-tiny
    `tiny-llama-ffn-up-s` is `tiny-llama` plus scale `3.5`. Decode identity
    vs oracle. `gpu-profile capture` is still refused. Dual score still has
    no `$/M tokens`.

403. [x] Official NVFP4 `{1}` `output.scale`: llama.cpp creates this tensor
    only when `output.weight` is `GGML_TYPE_NVFP4` (`TENSOR_NOT_REQUIRED`;
    missing is `1.0`). Decode follows `build_lora_mm`: `ggml_mul` after the
    LM-head GEMM, before bias / tanh softcap. Writer-tiny
    `tiny-nvfp4-output-s` is `tiny-nvfp4` plus scale `4.0`. F32 files that
    contain `output.scale` are ignored (not loaded). Tied embeddings
    (absent `output.weight`) do not load the scale. Decode identity vs
    oracle. `gpu-profile capture` is still refused. Dual score still has
    no `$/M tokens`.

404. [x] CUDA green-context events (`cuGreenCtxRecordEvent` /
    `cuGreenCtxWaitEvent`): `green_ctx_record_event` captures every activity
    already submitted on streams bound to that ctx (not one stream;
    later work does not join). `green_ctx_wait_event` holds later work on
    that ctx, including streams bound after the wait. Capture is refused
    when a bound stream is capturing. No second `--green-ctx`.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

405. [x] CUDA green-context CPU sync (`cudaExecutionCtxSynchronize`):
    `green_ctx_synchronize` waits every stream bound to that ctx (and that
    ctx's wait-event ops). Other green contexts on the same GPU keep
    running. Distinct from `synchronize_stream` and `synchronize_device`.
    Capture is refused when a bound stream is capturing. No second
    `--green-ctx`. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

406. [x] CUDA `cuStreamGetDevResource` (`stream_get_dev_resource`): query the
    SM span available to a stream. Bound streams return that green ctx's
    span; unbound is a full chip. Legal during capture. Distinct from
    `stream_get_green_ctx` (ctx id) and `device_get_dev_resource` (always
    full chip). No second `--green-ctx`. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

407. [x] Official fused `{1}` `attn_qkv.scale`: llama.cpp `build_qkv` applies
    `wqkv_s` after the fused QKV GEMM, before QKV bias
    (`TENSOR_NOT_REQUIRED`; missing is `1.0`). Loaded only when fused
    `attn_qkv.weight` is present (bloom). Writer-tiny `tiny-bloom-attn-qkv-s`
    is `tiny-bloom` plus scale `3.0`. Llama files that contain
    `attn_qkv.scale` are ignored (not loaded). Split `attn_q.scale` /
    `attn_k.scale` / `attn_v.scale` stay the llama tensors. Do not invent
    `output.input_scale` this slice. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

408. [x] Official shared-expert `{1}` `ffn_down_shexp.scale`: llama.cpp
    `build_ffn` applies `ggml_mul` after the shared-expert down GEMM
    (`TENSOR_NOT_REQUIRED`; missing is `1.0`) before `ffn_gate_inp_shexp`
    sigmoid. Loaded only when `ffn_down_shexp.weight` is present
    (qwen2moe / qwen3next / llama4). Writer-tiny
    `tiny-qwen2moe-ffn-down-shexp-s` is `tiny-qwen2moe` plus scale `1.75`.
    Llama / qwen3moe files that contain `ffn_down_shexp.scale` are ignored
    (not loaded). Dense `ffn_down.scale` and per-expert `ffn_down_exps.scale`
    stay those tensors. Do not invent `output.input_scale` this slice.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

409. [x] Official shared-expert `{1}` `ffn_gate_shexp.scale`: llama.cpp
    `build_ffn` applies `ggml_mul` after the shared-expert gate GEMM
    (`TENSOR_NOT_REQUIRED`; missing is `1.0`) before SiLU. Loaded only when
    `ffn_gate_shexp.weight` is present (qwen2moe / qwen3next / llama4).
    Writer-tiny `tiny-qwen2moe-ffn-gate-shexp-s` is `tiny-qwen2moe` plus
    scale `2.25`. Llama / qwen3moe files that contain `ffn_gate_shexp.scale`
    are ignored (not loaded). Dense `ffn_gate.scale` and shared-expert
    `ffn_down_shexp.scale` stay those tensors. Do not invent
    `output.input_scale` this slice. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

410. [x] Official shared-expert `{1}` `ffn_up_shexp.scale`: llama.cpp
    `build_ffn` applies `ggml_mul` after the shared-expert up GEMM
    (`TENSOR_NOT_REQUIRED`; missing is `1.0`) before the gated product.
    Loaded only when `ffn_up_shexp.weight` is present (qwen2moe / qwen3next
    / llama4). Writer-tiny `tiny-qwen2moe-ffn-up-shexp-s` is `tiny-qwen2moe`
    plus scale `2.75`. Llama / qwen3moe files that contain
    `ffn_up_shexp.scale` are ignored (not loaded). Dense `ffn_up.scale` and
    shared-expert `ffn_gate_shexp.scale` / `ffn_down_shexp.scale` stay those
    tensors. Do not invent `output.input_scale` this slice.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

411. [x] CUDA `cuGreenCtxGetId` (`green_ctx_get_id`): unique id for a live
    green context (`cudaExecutionCtxGetId`). Distinct from `GreenCtxId`
    (the VM handle), `stream_get_id`, and `stream_get_green_ctx`. Query;
    legal during capture. Unknown or destroyed is Invalid. No
    NULL-means-current. No `cuCtxFromGreenCtx`. No second `--green-ctx`.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

412. [x] CUDA `cudaExecutionCtxGetDevice` (`green_ctx_get_device`): device of
    a live green context. Distinct from `green_ctx_get_id`,
    `stream_get_green_ctx`, and `device_get_dev_resource`. Query; legal
    during capture. Unknown or destroyed is Invalid. No
    `cuCtxFromGreenCtx`. No second `--green-ctx`. `gpu-profile capture`
    is still refused. Dual score still has no `$/M tokens`.

413. [x] CUDA `cudaEventGetFlags` (`event_get_flags`): create flags word for
    a live event. Distinct from `event_timing` / `event_blocking_sync`.
    Query; legal during capture. Unknown or destroyed is UnknownEvent.
    Implicit first-record events are Default. IPC imports report
    Interprocess plus DisableTiming. No second `--green-ctx`.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

414. [x] CUDA `cudaGraphGetId` (`graph_get_id`): unique id for a live graph
    or exec (`cudaGraphExecGetId`). Matches debug-dot HANDLES. Distinct
    from node indices. A definition, instantiate exec, and clone each
    have their own id. Query; legal during capture. Unknown or destroyed
    is Invalid. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

415. [x] CUDA `cuGraphNodeGetLocalId` (`graph_node_get_local_id`): live node
    id matching debug-dot `n0`. Together with `graph_get_id` uniquely
    identifies a node. Query; legal during capture. Destroyed or unknown
    is Invalid. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

416. [x] CUDA `cuGraphNodeGetToolsId` (`graph_node_get_tools_id`): unique
    tools id for a live graph node. Distinct from `graph_node_get_local_id`
    (debug-dot `n0`) and `graph_get_id`. A definition, instantiate exec, and
    clone each assign different tools ids. Query; legal during capture.
    Destroyed or unknown is Invalid. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

417. [x] CUDA `cuDeviceGetUuid` (`device_get_uuid`): synthetic 16-octet UUID
    for a live device (`cudaDeviceGetUuid`). Distinct from `device_get_name`
    and `DeviceId`. Two `Sim`s with the same profile agree; GPUs on an 8×H100
    profile differ. Also `DeviceProperties.uuid`. Query; legal during
    capture. Unknown is Invalid. Not a real NVIDIA UUID. No
    `cuDeviceGetLuid`. `gpu-profile capture` is still refused. Dual score
    still has no `$/M tokens`.

418. [x] CUDA `cuDeviceGetByUuid` (`device_get_by_uuid`): inverse of
    `device_get_uuid`. A matching synthetic UUID returns that device.
    Unknown UUID is Invalid. Query; legal during capture. Distinct from
    `device_get` (ordinal). No `cuDeviceGetLuid`. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

419. [x] CUDA `cudaDeviceGetPciBusId` (`device_get_pci_bus_id`): synthetic
    `domain:bus:device.function` string for a live device
    (`cuDeviceGetPCIBusId`). Distinct from `device_get_uuid` and
    `device_get_name`. Same ordinal on two profiles share a bus id. Also
    `DeviceProperties` PCI ids and `DeviceAttr::PciDomainId` /
    `PciBusId` / `PciDeviceId`. Query; legal during capture. Unknown is
    Invalid. Not a real PCI topology. No `cudaDeviceGetByPCIBusId` this
    slice. No `cuDeviceGetLuid`. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

420. [x] CUDA `cudaDeviceGetByPCIBusId` (`device_get_by_pci_bus_id`): inverse
    of `device_get_pci_bus_id`. A matching synthetic bus id returns that
    device. Unknown or malformed is Invalid. Query; legal during capture.
    Distinct from `device_get_by_uuid`. No `cuDeviceGetLuid`.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

421. [x] CUDA `cuDevicePrimaryCtxGetState` (`device_primary_ctx_get_state`):
    flags match `get_device_flags`; active is always true (this VM seeds a
    primary context at construct). Query; legal during capture. Unknown is
    Invalid. No `cuDevicePrimaryCtxRetain` / `Release` / `Reset` (no
    `CUcontext` object). `gpu-profile capture` is still refused. Dual score
    still has no `$/M tokens`.

422. [x] CUDA `cudaDevAttrMaxAccessPolicyWindowSize`
    (`DeviceAttr::MaxAccessPolicyWindowSize`): L2 bytes, same as
    `MaxPersistingL2CacheSize` / `DeviceProperties.l2_cache_size`. Also
    `DeviceProperties.access_policy_max_window_size`. Query; legal during
    capture. `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

423. [x] CUDA `cudaDevAttrGPUDirectRDMAWritesOrdering`
    (`DeviceAttr::GpuDirectRdmaWritesOrdering`): always None. Native write
    visibility is not modeled, so `flush_gpu_direct_rdma_writes` is never a
    no-op. Distinct from `GpuDirectRdmaFlushWritesOptions` (Host on an RDMA
    SKU). Also `DeviceProperties.gpu_direct_rdma_writes_ordering`. Query;
    legal during capture. Do not invent Owner / AllDevices as reported
    values. `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

424. [x] CUDA `cudaDevAttrGlobalL1CacheSupported`
    (`DeviceAttr::GlobalL1CacheSupported`): always 0. This VM does not
    model L1 caches. Distinct from `L2CacheSize`. Also
    `DeviceProperties.global_l1_cache_supported`. Query; legal during
    capture. Do not invent `cudaDevAttrLocalL1CacheSupported` this slice.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

425. [x] CUDA `cudaDevAttrLocalL1CacheSupported`
    (`DeviceAttr::LocalL1CacheSupported`): always 0. This VM does not
    model L1 caches. Distinct from `GlobalL1CacheSupported`. Also
    `DeviceProperties.local_l1_cache_supported`. Query; legal during
    capture. `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

426. [x] CUDA `cudaDevAttrComputePreemptionSupported`
    (`DeviceAttr::ComputePreemptionSupported`): always 0. Kernel
    preemption is not modeled. Distinct from `KernelExecTimeout`. Also
    `DeviceProperties.compute_preemption_supported`. Query; legal during
    capture. `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

427. [x] CUDA `cudaDevAttrEccEnabled` (`DeviceAttr::EccEnabled`): always 0.
    ECC is not modeled. Distinct from `TccDriver`. Also
    `DeviceProperties.ecc_enabled`. Query; legal during capture.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

428. [x] CUDA `cudaDeviceSetCacheConfig` / `GetCacheConfig` /
    `cudaFuncSetCacheConfig`: `set_cache_config` / `get_cache_config` /
    `set_func_cache_config` / `get_func_cache_config`. Default PreferNone.
    PreferShared / PreferL1 / PreferEqual are stored; L1 is not modeled so
    kernel duration does not change. Host-sync 1 ns on Set. Capture cannot
    include Set. Get is a query. Per device (this VM is not per
    kernel-function object). Decode identity stays PreferNone. No Engine
    `--cache-config`. Distinct from SharedMemCarveout / SharedMemoryMode.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

429. [x] CUDA `cudaDevAttrReservedSharedMemoryPerBlock`
    (`DeviceAttr::ReservedSharedMemoryPerBlock`): always 0. Driver-reserved
    shared memory is not modeled. Distinct from `MaxSharedMemoryPerBlock`.
    Also `DeviceProperties.reserved_shared_mem_per_block`. Query; legal
    during capture. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

430. [x] CUDA `cudaDevAttrTotalConstantMemory`
    (`DeviceAttr::TotalConstantMemory`): always 0. Constant memory is not
    modeled. Distinct from `TotalGlobalMem`. Also
    `DeviceProperties.total_constant_memory`. Query; legal during capture.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

431. [x] CUDA `cudaDevAttrTextureAlignment` (`DeviceAttr::TextureAlignment`):
    always 0. CUDA arrays / textures are not modeled. Distinct from
    `SparseCudaArraySupported`. Also `DeviceProperties.texture_alignment`.
    Query; legal during capture. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

432. [x] CUDA `cudaDevAttrSurfaceAlignment` (`DeviceAttr::SurfaceAlignment`):
    always 0. CUDA surfaces are not modeled. Distinct from
    `TextureAlignment`. Also `DeviceProperties.surface_alignment`. Query;
    legal during capture. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

433. [x] CUDA `cudaDevAttrTexturePitchAlignment`
    (`DeviceAttr::TexturePitchAlignment`): always 0. CUDA textures are not
    modeled. Distinct from `TextureAlignment` and from `MemcpyOp` 2D
    pitches. Also `DeviceProperties.texture_pitch_alignment`. Query; legal
    during capture. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

434. [x] CUDA `cudaDevAttrMaxTexture1DWidth` (`DeviceAttr::MaxTexture1DWidth`):
    always 0. CUDA arrays / textures are not modeled. Distinct from
    `TextureAlignment`. Also `DeviceProperties.max_texture_1d_width`. Query;
    legal during capture. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

435. [x] `gpu-sim` `stream_add_callback` / `stream_add_callback_params` /
    `stream_add_callback_with_flags` are `cudaStreamAddCallback`. Same
    stream-ordered host enqueue as `host_func` (`cudaLaunchHostFunc`):
    bills `host_func_ns`, does not occupy compute or copy engines,
    records `HostNodeParams`. **Cannot be captured**
    (`cudaErrorStreamCaptureUnsupported`; Invalid
    `"cannot capture stream callback"`). CUDA flags must be 0
    (`StreamCallbackFlags::DEFAULT`; nonzero Invalid
    `"stream callback flags"`). Distinct from capturable `host_func`.
    No Engine `--stream-callback`. Decode identity stays LaunchHostFunc /
    events. `gpu-profile capture` is still refused. Dual score still has
    no `$/M tokens`.

436. [x] `gpu-sim` `DeviceP2pAttr::OnlyPartialNativeAtomicSupported` is
    always 0. P2P native atomics are not modeled. Distinct from
    `NativeAtomicSupported` (full P2P atomics) and from
    `OnlyPartialHostNativeAtomicSupported` (host-mapped). Query; legal
    during capture. Do not invent `cudaDeviceGetP2PAtomicCapabilities`.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

437. [x] `gpu-sim` `init_device` / `init_device_with_flags` are
    `cudaInitDevice`. Primary ctx is already seeded; this does not make a
    thread-current device. `InitDeviceFlags::FLAGS_ARE_VALID` applies
    `deviceFlags` like `set_device_flags`; without that bit they are
    ignored. Unknown flags Invalid `"init device flags"`. Host-sync 1 ns.
    Capture cannot include it. Distinct from `set_device_flags` (always
    applies) and from parked `cudaSetDevice`. No Engine `--init-device`.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

438. [x] `gpu-sim` `DeviceAttr::MaxPitch` is `DeviceAttr::MAX_PITCH`
    (`i32::MAX`). This VM does not cap 2D memcpy / `cudaMallocPitch` pitch.
    Distinct from `TexturePitchAlignment` (always 0; textures are not
    modeled). Also `DeviceProperties.mem_pitch`. Query; legal during
    capture. `gpu-profile capture` is still refused. Dual score still has
    no `$/M tokens`.

439. [x] `gpu-sim` `va_get_handle_for_address_range` is
    `cuMemGetHandleForAddressRange`. Always Invalid `"dma-buf not modeled"`
    (`DeviceAttr::DmaBufSupported` is 0). `MemRangeHandleType::DMA_BUF_FD`
    only; flags 0 or `DMA_BUF_MAPPING_TYPE_PCIE`. Distinct from `ipc_get`
    and `create_shareable_pool`. Query; legal during capture. No Engine
    `--dma-buf`. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

440. [x] `gpu-sim` `va_export_to_shareable_handle` is
    `cuMemExportToShareableHandle`. Always Invalid `"not shareable"`
    (`MemAllocationProp::handle_types` is none; POSIX-FD VMM export is not
    modeled). `MemHandleType::POSIX_FILE_DESCRIPTOR` only; flags 0. Distinct
    from `ipc_get`, `pool_export`, and `va_get_handle_for_address_range`.
    Capture cannot include it. No Engine `--vmm-export`. `gpu-profile capture`
    is still refused. Dual score still has no `$/M tokens`.

441. [x] `gpu-sim` `graph_add_dependencies_n_with_data` is
    `cudaGraphAddDependencies` v2 (`GraphEdgeData`). Default type with ports
    0 is identity with `graph_add_dependencies_n`. Programmatic type Invalid
    `"graph dependency type"`. `graph_edges_with_data` is `cudaGraphGetEdges`
    v2 (existing edges are Default, ports 0). Query; legal during capture.
    Capture cannot include Add. Distinct from kernel-node PDL. No Engine
    `--graph-edge-data`. `gpu-profile capture` is still refused. Dual score
    still has no `$/M tokens`.

442. [x] `gpu-sim` `device_get_nvscisync_attributes` is
    `cudaDeviceGetNvSciSyncAttributes`. Always Invalid `"nvscisync not modeled"`
    (`DeviceAttr::TimelineSemaphoreInteropSupported` is 0). Flags SIGNAL /
    WAIT (or both). Distinct from `ipc_get_event` and `wait_value32`. Query;
    legal during capture. No Engine `--nvscisync`. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

443. [x] `gpu-sim` `GraphDebugDotFlags::RUNTIME_TYPES` is
    `cudaGraphDebugDotFlagsRuntimeTypes` (`1 << 1`). Debug-dot labels use
    CUDA runtime `cudaGraphNodeType*` names. Flags `0` stays
    `GraphNodeKind` Debug names. VM-only kinds keep Debug names.
    `VERBOSE` includes RuntimeTypes. Query; legal during capture. Distinct
    from param-class dumps and HANDLES. No Engine `--graph-debug-dot`.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

444. [x] `gpu-sim` `graph_node_deps_with_data` /
    `graph_node_dependents_with_data` are `cudaGraphNodeGetDependencies` /
    `GetDependentNodes` v2 (`GraphEdgeData`). Existing edges are Default,
    ports 0 (identity with `graph_node_deps` / `graph_node_dependents`).
    Query; legal during capture. Distinct from `graph_edges_with_data`.
    No Engine `--graph-edge-data`. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

445. [x] `gpu-sim` `StreamCaptureInfo::edge_data` is
    `cudaStreamGetCaptureInfo_v3` (`GraphEdgeData` parallel to
    `dependencies`). Existing capture deps are Default, ports 0.
    Query during capture. Distinct from `graph_edges_with_data` and
    `graph_node_deps_with_data`. No Engine `--graph-edge-data`. Do not
    invent `cudaStreamUpdateCaptureDependencies` v2 edgeData.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

446. [x] `gpu-sim` `DeviceProperties::persisting_l2_cache_max_size` is
    `cudaDeviceProp::persistingL2CacheMaxSize` /
    `DeviceAttr::MaxPersistingL2CacheSize` (L2 bytes). Distinct from
    `access_policy_max_window_size` and from the current
    `persisting_l2_cache_size` limit (default 0). Query; legal during
    capture. `gpu-profile capture` is still refused. Dual score still has
    no `$/M tokens`.

447. [x] `gpu-sim` `create_graph_with_flags` is `cudaGraphCreate` with the
    CUDA flags word. Flags 0 is identity with `create_graph`. Unknown bits
    Invalid `"graph create flags"`. Capture cannot include it. Distinct from
    `GraphInstantiateFlags`. No Engine `--graph-create-flags`.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

448. [x] `gpu-sim` `event_get_id` is `cuEventGetId` / `cudaEventGetId`.
    Unique per `EventId` handle (`EventId + 1`). Distinct from the
    caller-chosen handle, `stream_get_id`, and `event_get_flags`. Query;
    legal during capture. Recreate after destroy returns the same id.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

449. [x] `gpu-sim` `graph_conditional_create_with_flags` is
    `cudaGraphConditionalHandleCreate` with the CUDA flags word.
    `GraphCondFlags::ASSIGN_DEFAULT` (`cudaGraphCondAssignDefault`) is
    identity with `graph_conditional_create` (each launch resets to the
    create-time default). Flags 0 keeps the handle across launches.
    Unknown bits Invalid `"graph cond flags"`. Capture cannot include it.
    No Engine `--graph-cond-flags`. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

450. [x] `gpu-sim` `pool_get_id` is `cuMemPoolGetId`. Unique per `PoolId`
    handle (`PoolId + 1`). Distinct from the handle, `event_get_id`, and
    `stream_get_id`. Query; legal during capture. Graph-memory pools are
    legal (distinct from `pool_get_attribute`). Destroyed is Invalid.
    Imported shareable pools differ from the exporter. Recreate after
    destroy returns a new id. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

451. [x] `gpu-sim` `device_get_exec_affinity_support` is
    `cuDeviceGetExecAffinitySupport`. `SM_COUNT` is 0 (this VM uses
    permille green-context spans, not occupancy SM counts). Other type
    ids Invalid `"exec affinity type"`. Query; legal during capture. Do
    not invent `cuCtxSetExecAffinity`. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

452. [x] `gpu-sim` `pool_set_access_read` is `cudaMemPoolSetAccess`
    `cudaMemAccessFlagsProtRead`. Peer kernels may read without dest
    HBM; writes and memset stay `NotResident` until ReadWrite.
    `pool_set_access_with_flags` / `pool_set_access_n` accept
    `PROT_READ`. `pool_get_access` returns `PROT_READ` on those peers.
    Owner stays ReadWrite. Capture cannot include it. Graph-memory
    pools stay Invalid. No Engine `--pool-prot-read`. `gpu-profile
    capture` is still refused. Dual score still has no `$/M tokens`.

453. [x] `gpu-sim` `mem_advise_with_size` is `cudaMemAdvise` with the CUDA
    `count` argument. `size` must equal the allocation bytes. Other
    sizes Invalid `"advise size"`. Partial advise is not modeled. Typed
    `mem_advise` stays. Capture cannot include it. No Engine
    `--advise-size`. `gpu-profile capture` is still refused. Dual score
    still has no `$/M tokens`.

454. [x] `gpu-sim` `prefetch_with_size` / `prefetch_host_with_size` are
    `cudaMemPrefetchAsync` with the CUDA `count` argument. `size` must
    equal the allocation bytes. Other sizes Invalid `"prefetch size"`.
    Partial prefetch is not modeled. Typed `prefetch` / `prefetch_host`
    stay. Capture may record the memcpy. No Engine `--prefetch-size`.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

455. [x] `gpu-sim` `stream_attach_with_size` is `cudaStreamAttachMemAsync`
    with the CUDA `length` argument. `size` `0` is the entire
    allocation. A nonzero `size` must equal the allocation bytes. Other
    sizes Invalid `"attach size"`. Partial attach is not modeled. Typed
    `stream_attach` stays (`length` 0). Capture cannot include it. No
    Engine `--attach-size`. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

456. [x] `gpu-sim` `mem_range_get_attribute_with_size` /
    `mem_range_get_attributes_with_size` are `cudaMemRangeGetAttribute` /
    `GetAttributes` with the CUDA `count` argument. `size` must equal
    the allocation bytes. Other sizes Invalid `"range size"`. Partial
    range queries are not modeled. Typed helpers stay. Query; legal
    during capture. No Engine `--range-size`. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

457. [x] `gpu-sim` `host_register_with_size` is `cudaHostRegister` with
    the CUDA `size` argument. `size` must equal the allocation bytes.
    Other sizes Invalid `"register size"`. Partial register is not
    modeled. Typed `host_register` / `host_register_mapped` /
    `host_register_with_flags` stay (full size). Capture cannot include
    it. No Engine `--register-size`. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

458. [x] `gpu-sim` `mem_range_get_attribute_with_data_size` /
    `mem_range_get_attributes_with_data_sizes` are
    `cudaMemRangeGetAttribute` / `GetAttributes` `dataSize` /
    `dataSizes`. Scalar attrs need 4 bytes (`sizeof(int)`). AccessedBy
    needs 4 bytes per device plus a `cudaInvalidDeviceId` terminator.
    Smaller is Invalid `"range data size"`. `attrs` / `data_sizes`
    length mismatch is Invalid `"range data sizes"`. Typed helpers stay
    (implicit sufficient `dataSize`). Count is still
    `mem_range_get_attribute_with_size`. Query; legal during capture.
    No Engine `--range-data-size`. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

459. [x] `gpu-sim` `upload_graph_async` is `cudaGraphUpload` on a stream.
    Stream-ordered Solo `graph_upload_ns`; the exec is uploaded when the
    op **completes**. Already uploaded at start is 1 ns. Capture cannot
    include it. Typed `upload_graph` stays host-synchronous.
    `launch_graph` waits an in-flight upload instead of a second
    host-sync upload. No Engine `--graph-upload-stream`. `gpu-profile
    capture` is still refused. Dual score still has no `$/M tokens`.

460. [x] `gpu-sim` `GraphInstantiateParams::upload_stream` is
    `cudaGraphInstantiateWithParams` `hUploadStream`. When
    `GraphInstantiateFlags::UPLOAD` is set, `Some` enqueues
    `upload_graph_async` (uploaded when the op completes). `None`
    stays host-sync `upload_graph`. Ignored when UPLOAD is unset.
    `instantiate_graph_with_flags` stays host-sync. No Engine
    `--graph-upload-stream`. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

461. [x] `gpu-sim` `graph_add_if_else` is `cudaGraphCondTypeIf` size 2.
    Then-body skips when the handle is `0`; the else-body runs instead.
    `graph_add_if` stays size 1 (`else_body = None`). Capture cannot
    include it. Illegal on an instantiated exec. No Engine
    `--graph-if-else`. `gpu-profile capture` is still refused. Dual score
    still has no `$/M tokens`.

462. [x] `gpu-sim` `GraphNodeParams::If` / `IfElse` / `While` are
    `cudaGraphAddNode` conditional params. `GraphAddNode::body` /
    `else_body` are `phGraph_out`. Typed `graph_add_if` /
    `graph_add_if_else` / `graph_add_while` stay. SWITCH stays
    `graph_add_switch`. GetParams is handle-only. SetParams is Invalid
    `"conditional node params"`. No Engine `--graph-if-else`.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

463. [x] `gpu-sim` `GraphNodeParams::Switch` is `cudaGraphAddNode`
    SWITCH. `GraphAddNode::switch_bodies` is Copy `phGraph_out` (`n` is
    `1..=64`). Typed `graph_add_switch` stays. GetParams is handle plus
    branch count. SetParams is Invalid `"conditional node params"`.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

464. [x] `gpu-sim` `graph_exec_child_get_graph` is
    `cudaGraphExecChildGraphNodeGetParams`. Uninstantiated graphs are
    Invalid. After instantiate this is the launched child.
    `graph_child_get_graph` stays a view. Query; legal during capture.
    No Engine `--graph-exec-child`. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

465. [x] `gpu-sim` `graph_exec_event_record_get_event` /
    `graph_exec_event_wait_get_event` are exec-snapshot
    `cudaGraphEventRecordNodeGetEvent` / `WaitNodeGetEvent`.
    Uninstantiated graphs are Invalid. After instantiate this is the
    launched event. View GetEvent stays. Query; legal during capture.
    No Engine `--graph-exec-event`. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

466. [x] `gpu-sim` `graph_exec_alloc_get_params` /
    `graph_exec_free_get_params` are exec-snapshot
    `cudaGraphMemAllocNodeGetParams` / `MemFreeNodeGetParams`.
    Uninstantiated graphs are Invalid. After instantiate this is the
    launched node. View GetParams stays. Query; legal during capture.
    No Engine `--graph-exec-mem`. Alloc SetParams stays parked.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

467. [x] `gpu-sim` `graph_exec_kernel_node_copy_attributes` is
    exec-snapshot `cudaGraphKernelNodeCopyAttributes`. Uninstantiated
    graphs are Invalid. After instantiate this copies the launched
    attributes. Definition CopyAttributes does not retarget the exec.
    `sharedMemBytes` stays params, not CopyAttributes. Capture cannot
    include it. No Engine `--graph-exec-copy-attrs`. `gpu-profile
    capture` is still refused. Dual score still has no `$/M tokens`.

468. [x] `gpu-sim` `graph_if_else_nodes` lists size-2 IF nodes as
    `(index, handle, then, else)`. `graph_if_nodes` stays then-body.
    Query; legal during capture. Instantiated ids use the exec snapshot.
    No Engine `--graph-if-else`. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

469. [x] `gpu-sim` `cudaGraphConditionalNodeSetParams` /
    `cudaGraphExecConditionalNodeSetParams` retarget the IF / IfElse /
    WHILE / SWITCH handle. Type, size, and bodies stay topology.
    Definition SetParams does not retarget the exec. Capture cannot
    include it. No Engine `--graph-cond-set`. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

470. [x] `gpu-sim` `cudaGraphExecUpdate` copies IF / IfElse / WHILE /
    SWITCH handles when bodies match. Handle is a parameter (same as
    SetParams). Type, size, and bodies stay topology. No Engine
    `--graph-cond-update`. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

471. [x] `gpu-sim` `CU_POINTER_ATTRIBUTE_ACCESS_FLAGS`:
    [`pointer_get_access_flags`](Sim::pointer_get_access_flags) reports
    [`MemAccessFlags`] for an explicit [`DeviceId`] (no TLS current
    device). Flags are this VM's kernel residency, not
    `cudaDeviceEnablePeerAccess` (D2D memcpy only). Mapped host is
    ReadWrite; pool ProtRead / VMM `va_set_access` / managed
    SetAccessedBy are Read. Not a [`PointerAttr`]. Query; legal during
    capture. No Engine `--pointer-access`. CONTEXT / P2P tokens stay
    unmodeled. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

472. [x] `gpu-sim` `cudaMemsetNodeParams::elementSize`:
    [`MemsetOp::element_size`] is `1` / `2` / `4` (`cuMemsetD8` / `D16` /
    `D32`). Typed `memset` stays `1`. Offset, width, and nonzero pitch
    must divide that size. The fill value is not modeled. No Engine
    `--memset-element`. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

473. [x] `gpu-sim` `cuPointerGetAttributes`:
    [`pointer_get_attribute_n`](Sim::pointer_get_attribute_n) is a batch
    [`pointer_get_attribute`](Sim::pointer_get_attribute). Distinct from
    [`pointer_get_attributes`](Sim::pointer_get_attributes)
    (`cudaPointerGetAttributes` struct). Empty is `Ok([])` after a live
    alloc. All-or-nothing. ACCESS_FLAGS stays
    [`pointer_get_access_flags`](Sim::pointer_get_access_flags). Query;
    legal during capture. No Engine `--pointer-attrs`. `gpu-profile
    capture` is still refused. Dual score still has no `$/M tokens`.

474. [x] `gpu-sim` `CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES`:
    [`BatchMemOp::FlushRemoteWrites`] is stream-ordered 1 ns Solo on an
    RDMA SKU (`cuStreamBatchMemOp`). Distinct from host-sync
    [`flush_gpu_direct_rdma_writes`](Sim::flush_gpu_direct_rdma_writes)
    (capture refused). Write visibility is not modeled; flush is never
    a no-op. Non-RDMA is Invalid `"gpu direct rdma"`. Capture records a
    batch-mem-op node. No Engine `--flush-remote`. `gpu-profile capture`
    is still refused. Dual score still has no `$/M tokens`.

475. [x] `gpu-sim` `CU_STREAM_WAIT_VALUE_FLUSH`:
    [`WaitValueFlags::FLUSH`] follows a wait with a stream-ordered remote
    write flush on an RDMA SKU (same rule as
    [`BatchMemOp::FlushRemoteWrites`]). Stored on [`BatchMemOp::Wait`] /
    [`GpuOp::WaitValue`]; parameter, not topology. Unknown bits stay
    Invalid `"wait value flags"`. Non-RDMA is Invalid `"gpu direct rdma"`.
    Capture legal. No Engine `--wait-flush`. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

476. [x] `gpu-sim` `cuCtxGetId`:
    [`ctx_get_id`](Sim::ctx_get_id) is `cuCtxGetId` for the seeded primary
    context of an explicit [`DeviceId`] (no TLS current device / no
    `CUcontext` object). Distinct from [`green_ctx_get_id`](Sim::green_ctx_get_id)
    / [`stream_get_id`](Sim::stream_get_id) / [`event_get_id`](Sim::event_get_id).
    Unknown devices are Invalid `"device not in profile"`. Query; legal
    during capture. No Engine `--ctx-id`. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

477. [x] `gpu-sim` `cuGraphAddNode_v2`:
    [`graph_add_node_with_data`](Sim::graph_add_node_with_data) is
    `cudaGraphAddNode` with `dependencyData`. `deps` and `data` must match
    (`"graph add node data"`). [`GraphDependencyType::DEFAULT`] with ports
    0 is [`graph_add_node`](Sim::graph_add_node). Programmatic type stays
    Invalid. Edge checks run before the node is created. Capture cannot
    include it. No Engine `--graph-add-node-data`. `gpu-profile capture`
    is still refused. Dual score still has no `$/M tokens`.

478. [x] `gpu-sim` `cudaGraphAddMemAllocNode` accessDescs:
    [`graph_add_alloc_with_access`](Sim::graph_add_alloc_with_access) is
    `cudaMemAllocNodeParams::accessDescs`. Empty is
    [`graph_add_alloc`](Sim::graph_add_alloc). Peer
    [`MemAccessFlags::PROT_READ_WRITE`] / [`PROT_READ`](MemAccessFlags::PROT_READ)
    lets a live kernel on that GPU use the graph alloc without dest HBM
    (graph-memory pool stays Invalid for [`pool_set_access`](Sim::pool_set_access)).
    Host location `"access location"`; unknown flags `"alloc access flags"`.
    Applied when the alloc is live (launch), not at add. Last descriptor for a
    device wins. All-or-nothing before the node is created. Capture cannot
    include it. [`graph_alloc_get_access`](Sim::graph_alloc_get_access) /
    [`graph_exec_alloc_get_access`](Sim::graph_exec_alloc_get_access) are the
    GetParams accessDescs. SetParams of Alloc stays Invalid. No Engine
    `--graph-alloc-access`. `gpu-profile capture` is still refused. Dual score
    still has no `$/M tokens`.

479. [x] `gpu-sim` `cuMemsetD16` / `cuMemsetD32`:
    [`memset_d16_async`](Sim::memset_d16_async) / [`memset_d16`](Sim::memset_d16)
    are `cuMemsetD16Async` / `cuMemsetD16`. [`memset_d32_async`](Sim::memset_d32_async)
    / [`memset_d32`](Sim::memset_d32) are `cuMemsetD32Async` / `cuMemsetD32`.
    `count` is CUDA `N` (element count); payload is `count * 2` / `count * 4`.
    Overflow is Invalid `"memset count"`. Typed [`memset`](Sim::memset) stays
    byte-counted `element_size` 1. Capture of Async is legal; host-sync is
    refused. Fill value is not modeled. No Engine `--memset-d16`.
    `gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

480. [x] `gpu-sim` `cuMemsetD2D16` / `cuMemsetD2D32`:
    [`memset_d2d16_async`](Sim::memset_d2d16_async) / [`memset_d2d16`](Sim::memset_d2d16)
    are `cuMemsetD2D16Async` / `cuMemsetD2D16`. [`memset_d2d32_async`](Sim::memset_d2d32_async)
    / [`memset_d2d32`](Sim::memset_d2d32) are `cuMemsetD2D32Async` /
    `cuMemsetD2D32`. `width` is CUDA `Width` (element count); row payload is
    `width * 2` / `width * 4`. `pitch` is bytes. `height` `0` is Invalid
    `"memset2d height"`. Overflow is `"memset count"`.
    [`memset_2d_async`](Sim::memset_2d_async) stays byte-width. Capture of
    Async is legal; host-sync is refused. Fill value is not modeled. No Engine
    `--memset-d2d`. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

481. [x] `gpu-sim` `cudaGraphAddNode` mem-alloc accessDescs:
    [`GraphNodeParams::Alloc`] carries `accessDescs` (`bytes` plus
    [`MemAccessDesc`] list). [`graph_add_node`](Sim::graph_add_node) with
    empty access is [`graph_add_alloc`](Sim::graph_add_alloc); peer
    [`MemAccessFlags::PROT_READ_WRITE`] / [`PROT_READ`](MemAccessFlags::PROT_READ)
    matches [`graph_add_alloc_with_access`](Sim::graph_add_alloc_with_access)
    (kernel without dest HBM). GetParams
    [`graph_node_get_params`](Sim::graph_node_get_params) /
    [`graph_exec_node_get_params`](Sim::graph_exec_node_get_params) return
    stored accessDescs. Typed
    [`graph_add_alloc_with_access`](Sim::graph_add_alloc_with_access) stays.
    SetParams of Alloc stays Invalid. No Engine `--graph-alloc-access`.
    `gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

482. [x] `gpu-sim` `cuMemsetD2D8`:
    [`memset_d2d8_async`](Sim::memset_d2d8_async) / [`memset_d2d8`](Sim::memset_d2d8)
    are `cuMemsetD2D8Async` / `cuMemsetD2D8`. `width` is CUDA `Width`
    (8-bit element count); row payload is `width` bytes. `pitch` is bytes.
    `height` `0` is Invalid `"memset2d height"`.
    [`memset_2d_async`](Sim::memset_2d_async) stays byte-width [`MemsetOp`].
    Capture of Async is legal; host-sync is refused. Fill value is not
    modeled. No Engine `--memset-d2d`. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

483. [x] `gpu-sim` `cuMemcpy2DUnaligned`:
    [`memcpy_2d_unaligned`](Sim::memcpy_2d_unaligned) is
    `cuMemcpy2DUnaligned`. Identity with [`memcpy_2d`](Sim::memcpy_2d): this
    VM does not require CUDA 2D pitch/offset alignment. Host-synchronous;
    capture cannot include it. CUDA has no Async Unaligned;
    [`memcpy_2d_async`](Sim::memcpy_2d_async) stays. No Engine
    `--memcpy-unaligned`. `gpu-profile capture` is still refused. Dual score
    still has no `$/M tokens`.

484. [x] `gpu-sim` `CUmemPoolProps::usage`:
    [`MemPoolProps::usage`] is `cudaMemPoolProps::usage`. Must be
    [`MemHandleUsage::NONE`]. [`MemHandleUsage::HW_DECOMPRESS`] is Invalid
    `"pool usage"` (hardware decompress is not modeled;
    [`DeviceAttr::MemDecompressAlgorithmMask`] stays 0). Typed
    [`create_pool`](Sim::create_pool) stays. No Engine `--pool-usage`.
    `gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

485. [x] `gpu-sim` `CUmemAllocationProp` usage / compression:
    [`MemAllocationProp::usage`] must be [`MemHandleUsage::NONE`];
    [`MemAllocationProp::compression`] must be 0.
    [`va_create_with_prop`](Sim::va_create_with_prop) rejects HW decompress
    `"mem usage"` and nonzero compression `"mem compression"`. Get always
    reports none. Distinct from pool [`MemPoolProps::usage`] (`"pool usage"`).
    Typed [`va_create`](Sim::va_create) stays. No Engine `--vmm-usage`.
    `gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

486. [x] `gpu-sim` `cuMemcpyHtoD` / `cuMemcpyDtoH`:
    [`memcpy_htod`](Sim::memcpy_htod) is `cuMemcpyHtoD` (host-synchronous
    pinned H2D). [`memcpy_dtoh`](Sim::memcpy_dtoh) is `cuMemcpyDtoH`
    (host-synchronous Device→HostPinned). Capture cannot include them.
    [`memcpy_pinned_to_device`](Sim::memcpy_pinned_to_device) /
    [`memcpy_device_to_pinned`](Sim::memcpy_device_to_pinned) stay
    `cuMemcpyHtoDAsync` / `cuMemcpyDtoHAsync`. Pageable
    [`memcpy_host_to_device`](Sim::memcpy_host_to_device) /
    [`memcpy_device_to_host`](Sim::memcpy_device_to_host) stay.
    [`memcpy_sync`](Sim::memcpy_sync) stays generic `cudaMemcpy`. No Engine
    `--memcpy-htod`. `gpu-profile capture` is still refused. Dual score
    still has no `$/M tokens`.

487. [x] `gpu-sim` `cuMemAllocPitch`:
    [`malloc_pitch_with_element_size`](Sim::malloc_pitch_with_element_size)
    is `cuMemAllocPitch`. `element_size` is CUDA `ElementSizeBytes` and must
    be 4, 8, or 16 (`"malloc pitch element"`). Pitch stays
    `align_up(width, 512)` (this VM does not vary pitch by element size).
    [`malloc_pitch`](Sim::malloc_pitch) stays `cudaMallocPitch`. Host-sync;
    capture cannot include it. No Engine `--malloc-pitch-element`.
    `gpu-profile capture` is still refused. Dual score still has no `$/M tokens`.

488. [x] `gpu-sim` CUDA_MEMCPY2D / CUDA_MEMCPY3D srcPos / dstPos:
    [`MemcpyOp`] `src_x` / `src_y` / `src_z` / `dst_x` / `dst_y` /
    `dst_z` are `srcXInBytes` / `srcY` / `srcZ` / `dstXInBytes` /
    `dstY` / `dstZ`. Default 0 is origin `(0,0[,0])`. 1D with any
    origin, or 2D with a z origin, is Invalid `"memcpy origin"`. 2D/3D
    `x + width` vs pitch stays `"memcpy2d pitch"` / `"memcpy3d pitch"`.
    3D `y + height` vs ysize stays `"memcpy3d height"`. Oversized 2D
    `srcY` is `"memcpy range past alloc"`. Capture GetParams preserves
    origin. No Engine `--memcpy-origin`. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

489. [x] `gpu-sim` `cuMemcpy3DUnaligned`:
    [`memcpy_3d_unaligned`](Sim::memcpy_3d_unaligned) is
    `cuMemcpy3DUnaligned`. Identity with [`memcpy_3d`](Sim::memcpy_3d): this
    VM does not require CUDA 3D pitch/offset alignment. Host-sync; capture
    cannot include it. CUDA has no Async Unaligned;
    [`memcpy_3d_async`](Sim::memcpy_3d_async) stays. No Engine
    `--memcpy-3d-unaligned`. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

490. [x] `gpu-sim` CUDA_MEMCPY3D `srcLOD` / `dstLOD`:
    [`MemcpyOp`] `src_lod` / `dst_lod` must be 0. CUDA arrays are not
    modeled. Nonzero is Invalid `"memcpy lod"`. Default 0 is identical.
    No Engine `--memcpy-lod`. `gpu-profile capture` is still refused.
    Dual score still has no `$/M tokens`.

491. [x] `gpu-sim` `cuDevSmResourceSplit`:
    [`dev_sm_resource_split`](Sim::dev_sm_resource_split) is
    `cuDevSmResourceSplit`. [`DevSmResourceGroupParams`] `smCount` is ‰ of
    the chip (same unit as [`dev_sm_resource_split_by_count`]). `0` is
    discovery (remaining ‰). `coscheduledSmCount` /
    `preferredCoscheduledSmCount` must be 0 (occupancy SM counts are not
    modeled). Group flags must be 0 (`BACKFILL` is Invalid). API flags
    must be 0. Unequal groups; leftover is remaining. Query; legal during
    capture. Typed [`dev_sm_resource_split_by_count`] stays. No Engine
    `--sm-split`. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

492. [x] `gpu-sim` CUDA 13 `CUDA_KERNEL_NODE_PARAMS.ctx`:
    [`KernelNodeParams::ctx`] is `CUDA_KERNEL_NODE_PARAMS.ctx` /
    `cudaKernelNodeParamsV2.ctx`. Stored on the graph step, not
    `Kind::Kernel`. [`None`] inherits the launch stream. [`Some`] pins
    duration and SM occupancy to that live green context. Unknown or
    destroyed is Invalid `"unknown green ctx"`. Device mismatch is
    Invalid `"green ctx device"`. Parameter, not topology. Capture from
    a green-ctx stream snapshots that ctx. Typed [`graph_add_kernel`]
    stays [`None`]. No Engine `--kernel-ctx`. This VM does not invent
    `cuCtxFromGreenCtx`. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

493. [x] `gpu-sim` CUDA `CUDA_KERNEL_NODE_PARAMS.sharedMemBytes`:
    [`KernelNodeParams::shared_mem_bytes`] is
    `CUDA_KERNEL_NODE_PARAMS.sharedMemBytes`. Stored on the graph step,
    not `Kind::Kernel`. GetParams / SetParams / AddNode carry the field.
    Typed [`graph_add_kernel`] stays `0`. [`KernelNodeAttr::DynamicShared`]
    Get/SetAttribute stays. CopyAttributes does not copy it. Oversize
    without func attr / AllowNonPortable is Invalid `"dynamic shared"` /
    `"non-portable shared"`. Parameter, not topology. Duration is bank
    width, not byte count. No Engine `--kernel-shared`. `gpu-profile
    capture` is still refused. Dual score still has no `$/M tokens`.

494. [x] `gpu-sim` CUDA `CUDA_BATCH_MEM_OP_NODE_PARAMS.ctx`:
    [`BatchMemOpNodeParams::ctx`] is `CUDA_BATCH_MEM_OP_NODE_PARAMS.ctx`.
    Stored on the graph step, not `Kind::BatchMem`. [`None`] inherits the
    launch stream. Wait/write/flush do not occupy SMs, so duration is
    unchanged. Unknown or destroyed is Invalid `"unknown green ctx"`.
    Device mismatch is Invalid `"green ctx device"`. Parameter, not
    topology. Capture from a green-ctx stream snapshots that ctx. Typed
    [`graph_add_batch_mem_op`] stays [`None`]. Item-list SetParams does
    not clear ctx. No Engine `--batch-ctx`. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

495. [x] `gpu-sim` CUDA `CUDA_CONDITIONAL_NODE_PARAMS.ctx`:
    [`GraphNodeParams::If::ctx`] / `IfElse` / `While` / `Switch` is
    `CUDA_CONDITIONAL_NODE_PARAMS.ctx` (`cudaConditionalNodeParams.ctx`).
    [`graph_conditional_create_with_ctx`] stores the handle ctx
    (`cuGraphConditionalHandleCreate`). Node ctx must match the handle
    (Invalid `"conditional ctx"`). Stored on the graph step, not
    `Kind::If`. Unknown or destroyed is Invalid `"unknown green ctx"`.
    Device mismatch is Invalid `"green ctx device"`. Parameter, not
    topology. Conditionals do not occupy SMs, so duration is unchanged.
    Typed [`graph_add_if`] copies the handle ctx. Typed create stays
    [`None`]. This VM does not invent an Engine flag for conditional ctx.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

496. [x] `gpu-sim` CUDA `cuGraphAddMemcpyNode` ctx:
    [`MemcpyNodeParams::ctx`] is the extra driver `CUcontext` argument on
    `cuGraphAddMemcpyNode` / `cuGraphExecMemcpyNodeSetParams`. Stored on
    the graph step, not `Kind::Memcpy` or [`MemcpyOp`]. GetParams /
    SetParams / AddNode carry the field. Typed [`graph_add_memcpy`] stays
    [`None`]. Typed [`graph_memcpy_set_params`] does not clear ctx.
    Copies use copy engines, so duration is unchanged. Unknown or
    destroyed is Invalid `"unknown green ctx"`. Device mismatch is
    Invalid `"green ctx device"`. Parameter, not topology. Capture from a
    green-ctx stream snapshots that ctx. This VM does not invent an Engine
    flag for memcpy ctx. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

497. [x] `gpu-sim` CUDA `cuGraphAddMemsetNode` ctx:
    [`MemsetNodeParams::ctx`] is the extra driver `CUcontext` argument on
    `cuGraphAddMemsetNode` / `cuGraphExecMemsetNodeSetParams`. Stored on
    the graph step, not `Kind::Memset` or [`MemsetOp`]. GetParams /
    SetParams / AddNode carry the field. Typed [`graph_add_memset`] stays
    [`None`]. Typed [`graph_memset_set_params`] does not clear ctx.
    Fills use copy engines, so duration is unchanged. Unknown or
    destroyed is Invalid `"unknown green ctx"`. Device mismatch is
    Invalid `"green ctx device"`. Parameter, not topology. Capture from a
    green-ctx stream snapshots that ctx. This VM does not invent an Engine
    flag for memset ctx. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

498. [x] `gpu-sim` CUDA `CUgraphChildGraphNodeOwnership`:
    [`ChildGraphNodeParams::ownership`] is clone versus move. Typed
    [`graph_add_child`] stays clone of an instantiated child without mem
    alloc/free or conditional nodes. Move lets a parent own an
    uninstantiated child that may contain mem nodes. GetParams of a
    moved node reports [`GraphChildGraphOwnership::INVALID`]. After the
    move the child cannot be independently instantiated, launched,
    destroyed, cloned, updated, or added as a child of another parent.
    This VM does not invent an Engine flag for child-graph ownership.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

499. [x] `gpu-sim` CUDA `cudaLaunchAttributeSynchronizationPolicy` on graph
    kernel nodes:
    [`KernelNodeAttr::SynchronizationPolicy`] is CUDA 13
    `CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY` on graph kernel nodes.
    Valid for streams (already) and graph nodes, not host launches
    ([`KernelAttrs`] stays without it). Typed [`graph_add_kernel`] and
    capture snapshot the stream policy. [`SynchronizationPolicy::Auto`]
    inherits the recording stream at `cudaEventSynchronize` of that node's
    launch-completion or programmatic event. An explicit Spin / Yield /
    BlockingSync taxes that event wait even when the launch stream is Auto.
    Kernel duration is unchanged. Get/Set/CopyAttributes. Capture cannot
    include Set. This VM does not invent an Engine flag for kernel-node
    sync policy. `gpu-profile capture` is still refused. Dual score still
    has no `$/M tokens`.

500. [x] `gpu-sim` CUDA `cudaLaunchAttributeProgrammaticEvent.triggerAtBlockStart`:
    [`ProgrammaticEvent::trigger_at_block_start`] is CUDA
    `triggerAtBlockStart`. Default `false` records at the PDL trigger when
    [`ProgrammaticLaunch::trigger`], else at kernel completion (existing
    identity). Non-zero records when the kernel starts, same stamp as
    launch-completion (this VM does not model per-block begins). Other
    streams may [`wait_event`] earlier than the PDL trigger. Does not
    require `pdl.trigger`. [`PdlLaunch::trigger_event`] and
    `expertvm sim --programmatic-event` stay `false`. Get/Set/CopyAttributes
    and capture carry the field. Debug-dot KERNEL_NODE_ATTRIBUTES dumps
    `pde-block-start` when set. Kernel duration is unchanged. This VM does
    not invent an Engine flag for programmatic-event block-start.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

501. [x] `gpu-sim` CUDA device-updatable kernel-node restrictions:
    Once [`KernelAttrs::device_updatable`] is true, SetAttribute to false
    is Invalid `"device-updatable"`. The node cannot be destroyed.
    CopyAttributes cannot involve it. A definition with such a node cannot
    be instantiated twice. `cudaGraphExecUpdate` of an exec or source that
    has one is Invalid. Opt-in stays; decode identity stays false. This VM
    does not invent `CUgraphDeviceNode` / device-side kernel updates or an
    Engine flag for opt-out. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

502. [x] `gpu-sim` CUDA launch-attribute event flags and interprocess:
    [`ProgrammaticEvent::external`] / [`LaunchCompletionEvent::external`]
    (`cudaEventRecordExternal` on `programmaticEvent.flags` /
    `launchCompletionEvent.flags`) is Invalid. An interprocess or
    IPC-imported event is Invalid. The `external` field stays the CUDA
    flags word (always false). Decode identity stays no launch-attribute
    event. This VM does not invent a `flags: u32` field, disable-timing as
    a requirement, NvSciSync / interop event kinds, or an Engine flag for
    External. `gpu-profile capture` is still refused. Dual score still has
    no `$/M tokens`.

503. [x] `gpu-sim` CUDA graph launch-completion edge ports:
    [`GraphKernelNodePort::LAUNCH_COMPLETION`] (`cudaGraphKernelNodePortLaunchCompletion`)
    on [`GraphEdgeData::from_port`] waits for the source kernel to start,
    not finish. Typed [`graph_add_dependencies`] stays Default ports 0
    (completion wait). Programmatic type and port 1 stay Invalid. Source
    must be a kernel. GetEdges / GetDependencies v2 report the stored
    ports. This VM does not invent an Engine flag for edge ports or
    nonzero `to_port`. `gpu-profile capture` is still refused. Dual score
    still has no `$/M tokens`.

504. [x] `gpu-sim` `GraphDebugDotFlags::EXTRA_TOPO_INFO` is
    `cudaGraphDebugDotFlagsExtraTopoInfo` (`1 << 14`). Debug-dot numbers
    existing edges (`label="0"`). Launch-completion edges also dump
    `from_port=2`. `VERBOSE` includes ExtraTopoInfo. Flags `0` stays
    unlabeled edges. Query; legal during capture. Distinct from
    param-class dumps and HANDLES. This VM does not invent extra
    conditional edges or Engine `--graph-debug-dot`. `gpu-profile
    capture` is still refused. Dual score still has no `$/M tokens`.

505. [x] `gpu-sim` CUDA graph ExecUpdate treats launch-completion edge ports
    as topology: `cudaGraphExecUpdate` of a Default completion edge vs
    [`GraphKernelNodePort::LAUNCH_COMPLETION`] is
    [`GraphExecUpdateResult::DependenciesChanged`]. Same `(from, to)` with
    matching ports still updates. Child-graph SetParams nested topology
    includes ports. Typed `graph_add_dependencies` stays Default ports 0.
    This VM does not invent Engine `--graph-edge-port`, programmatic edges,
    or `cudaGraphRemoveDependencies` edgeData. `gpu-profile capture` is
    still refused. Dual score still has no `$/M tokens`.

506. [x] `gpu-sim` CUDA `cudaGraphExecUpdate` forbids kernel-node priority
    changes when the exec was instantiated with
    [`GraphInstantiateFlags::USE_NODE_PRIORITY`]: mismatch is
    [`GraphExecUpdateResult::AttributesChanged`]. Matching priorities
    still update. Default instantiate copies priority as a parameter.
    [`graph_exec_kernel_node_set_priority`] stays legal. This VM does not
    invent Engine `--graph-node-priority-update`, child SetParams nested
    priority, or `cudaDeviceGetStreamPriorityRange` clamping.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

507. [x] `gpu-sim` CUDA `cudaGraphExecUpdate` of 2D and 3D memset nodes
    may change address only (id / offset; fill value is not modeled).
    Width, height, pitch, depth, ysize, and elementSize are
    [`GraphExecUpdateResult::ParametersChanged`]. 1D memset may change
    dimensions. [`graph_exec_memset_set_params_2d`] stays legal. This VM
    does not invent Engine `--memset-update`, a memset fill value, or
    1D work-resource mapping failures. `gpu-profile capture` is still
    refused. Dual score still has no `$/M tokens`.

508. [x] `gpu-sim` CUDA `cudaGraphExecUpdate` memcpy source and destination
    memory types (`CU_MEMORYTYPE_*` / [`Place`]) cannot change
    ([`GraphExecUpdateResult::ParametersChanged`]). Alloc and size may.
    [`graph_exec_memcpy_set_params`] stays legal. CUDA arrays stay
    uninvented. This VM does not invent Engine `--memcpy-update` or
    CUDA-array memcpy. `gpu-profile capture` is still refused. Dual score
    still has no `$/M tokens`.

509. [x] `gpu-sim` CUDA `cudaMemPoolAttrMaxPoolSize` is
    [`MemPoolAttr::MaxPoolSize`] (`pool_get_attribute` /
    `pool_set_attribute`). Typed [`set_pool_max_size`] stays. Get reports
    stored [`MemPoolProps::max_size`] (`0` unlimited). Set updates the cap
    for later `alloc_from_pool` (reserved live plus cached). Does not free
    live allocs if the new cap is below current reserved. Graph-memory pool
    stays Invalid. Capture: Get legal, Set Invalid. Imported pools Get/Set
    the exporter. `expertvm sim --mempool-max` stays
    `create_pool_with_props` plus `set_device_mempool`. This VM does not
    invent a second MaxPoolSize, Engine `--mempool-max` as SetAttribute-only,
    alignment rounding, `cudaMemPoolAttrHwDecompressEnabled`, or pool
    LocationType / LocationId. `gpu-profile capture` is still refused. Dual
    score still has no `$/M tokens`.

510. [x] `gpu-sim` CUDA `cudaMemPoolAttrAllocationType` /
    `cudaMemPoolAttrExportHandleTypes` are [`MemPoolAttr::AllocationType`]
    / [`MemPoolAttr::ExportHandleTypes`] (`pool_get_attribute`). Get-only
    (`pool_set_attribute` is `"read-only pool attr"`). AllocationType is
    always [`MemAllocationType::PINNED`]. ExportHandleTypes is
    [`MemHandleType::POSIX_FILE_DESCRIPTOR`] on shareable exporters and
    [`MemHandleType::NONE`] on default, `create_pool`, and imported handles
    (imported cannot be re-exported). Graph-memory pool stays Invalid.
    Capture: Get legal, Set Invalid. This VM does not invent NT/fabric
    handle types, Engine `--pool-alloc-type`, or Set of these attrs.
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

511. [x] `gpu-sim` CUDA `cudaGraphExecUpdate` of a different kernel-node
    function ([`KernelKind`] variant: Other, Matmul, or GroupedMoeGemm) is
    [`GraphExecUpdateResult::FunctionChanged`]. Matching variants still
    update (work sizes may change). [`graph_exec_kernel_set_params`] stays
    legal. Child SetParams nested function stays a parameter. This VM does
    not invent Engine `--graph-function-update` or
    `UnsupportedFunctionChange` (no `CUfunction` signature or destroy).
    `gpu-profile capture` is still refused. Dual score still has no
    `$/M tokens`.

512. [x] `gpu-sim` CUDA `cudaGraphExecMemcpyNodeSetParams` is
    1-dimensional only ([`graph_exec_memcpy_set_params`]): both the
    instantiated node and new [`MemcpyOp`] must be [`MemcpyOp::is_1d`]
    (Invalid `"memcpy 1d"`). [`graph_exec_node_set_params`] of
    [`GraphNodeParams::Memcpy`] uses the same check.
    [`graph_exec_memcpy_set_params_1d`] may still convert a 2D/3D node
    (PLAN 190). Extra [`graph_exec_memcpy_set_params_2d`] plus
    [`graph_exec_memcpy_set_params_3d`] stay (not CUDA names). Definition
    [`graph_memcpy_set_params`] still accepts 2D/3D. This VM does not
    invent Engine `--graph-memcpy-1d` or ExecUpdate 2D memcpy geometry as
    ParametersChanged. `gpu-profile capture` is still refused. Dual score
    still has no `$/M tokens`.

513. [ ] Next numbered PLAN item after 512 is the next `gpu-sim` / Engine /
    serve / expertvm mechanical API that is still missing, or the next official
    decode family. Prefer remaining CUDA-shaped twins over more
    OpenAI HTTP veneer. Do not invent F32 `output.scale`. Do not invent a
    second NVFP4 `output.scale` as a new family. Do not invent
    `output.input_scale` / `*.input_scale` this slice. Do not invent a second
    fused `attn_qkv.scale` as a new family. Do not invent a second
    `ffn_down_shexp.scale` as a new family. Do not invent a second
    `ffn_gate_shexp.scale` as a new family. Do not invent a second
    `ffn_up_shexp.scale` as a new family. Do not invent a second
    `cuGreenCtxRecordEvent` / `WaitEvent`. Do not invent a second
    `cudaExecutionCtxSynchronize`. Do not invent a second
    `cuStreamGetDevResource`. Do not invent a second `cuGreenCtxGetId`.
    Do not invent a second `cuCtxGetId` / `ctx_get_id`. Do not invent Engine
    `--ctx-id`.
    Do not invent a second `cudaExecutionCtxGetDevice`. Do not invent a
    second `cudaEventGetFlags`. Do not invent a second `cudaGraphGetId` /
    `cudaGraphExecGetId`. Do not invent a second `cuGraphNodeGetLocalId`.
    Do not invent a second `cuGraphNodeGetToolsId`. Do not invent a second
    `cuDeviceGetUuid` / `cudaDeviceGetUuid`. Do not invent a second
    `cuDeviceGetByUuid`. Do not invent a second `cudaDeviceGetPciBusId` /
    `cuDeviceGetPCIBusId`. Do not invent a second `cudaDeviceGetByPCIBusId`.
    Do not invent a second `cuDevicePrimaryCtxGetState`. Do not invent
    `cuDevicePrimaryCtxRetain` / `Release` / `Reset` (no `CUcontext` object).
    Do not invent a second `cudaDevAttrMaxAccessPolicyWindowSize`.
    Do not invent a second `cudaDevAttrGPUDirectRDMAWritesOrdering`.
    Do not invent Owner / AllDevices GPUDirect RDMA write ordering (flush
    would become a no-op).
    Do not invent a second `cudaDevAttrGlobalL1CacheSupported`.
    Do not invent a second `cudaDevAttrLocalL1CacheSupported`.
    Do not invent a second `cudaDevAttrComputePreemptionSupported`.
    Do not invent a second `cudaDevAttrEccEnabled`.
    Do not invent a second `cudaDeviceSetCacheConfig` / `GetCacheConfig` /
    `cudaFuncSetCacheConfig`. Do not invent Engine `--cache-config`
    (stored-only; L1 vs Shared is not mechanically distinct).
    Do not invent a second `cudaDevAttrReservedSharedMemoryPerBlock`.
    Do not invent a second `cudaDevAttrTotalConstantMemory`.
    Do not invent a second `cudaDevAttrTextureAlignment`.
    Do not invent a second `cudaDevAttrSurfaceAlignment`.
    Do not invent a second `cudaDevAttrTexturePitchAlignment`.
    Do not invent a second `cudaDevAttrMaxTexture1DWidth`.
    Do not invent a second `cudaStreamAddCallback`.
    Do not invent Engine `--stream-callback` (same wall as second live
    `cudaLaunchHostFunc` after miss DMA).
    Do not invent a second `cudaDevP2PAttrOnlyPartialNativeAtomicSupported`.
    Do not invent `cudaDeviceGetP2PAtomicCapabilities` (atomics are not
    modeled; partial is 0). Do not invent
    `cudaDeviceGetHostAtomicCapabilities`.
    Do not invent a second `cudaInitDevice`. Do not invent Engine
    `--init-device` (same wall as `set_device_flags`). Do not invent
    `cudaSetDevice` (no thread-current device).
    Do not invent a second `cudaDevAttrMaxPitch`.
    Do not invent a second `cuMemGetHandleForAddressRange`. Do not invent
    a dma-buf file descriptor. Do not invent Engine `--dma-buf`.
    Do not invent a second `cuMemExportToShareableHandle`. Do not invent a
    VMM POSIX-FD file descriptor. Do not invent
    `cuMemImportFromShareableHandle` for VMM. Do not invent Engine
    `--vmm-export`.
    Do not invent a second `cudaDeviceGetNvSciSyncAttributes`. Do not invent
    an NvSciSync attribute list. Do not invent Engine `--nvscisync`. Do not
    invent `cudaImportExternalSemaphore` / graph external-semaphore nodes.
    Do not invent a second `cudaGraphDebugDotFlagsRuntimeTypes`. Do not
    invent Engine `--graph-debug-dot`. Do not invent
    `cudaGraphDebugDotFlagsExtSemasSignalNodeParams` / `WaitNodeParams`
    dumps (bits 7-8 stay Invalid). Do not invent
    `cudaGraphDebugDotFlagsExtraTopoInfo` extra edges. Do not invent a
    second `cudaGraphDebugDotFlagsExtraTopoInfo` edge numbering.
    Do not invent a second `cudaGraphNodeGetDependencies` /
    `GetDependentNodes` v2.
    Do not invent a second `cudaStreamGetCaptureInfo_v3`. Do not invent
    `cudaStreamUpdateCaptureDependencies` v2 edgeData.
    Do not invent a second `cudaDeviceProp::persistingL2CacheMaxSize`.
    Do not invent a second `cudaGraphCreate` flags word. Do not invent
    non-zero `cudaGraphCreate` flags. Do not invent Engine
    `--graph-create-flags`.
    Do not invent a second `cuEventGetId` / `cudaEventGetId`.
    Do not invent a second `cudaGraphConditionalHandleCreate` flags word.
    Do not invent Engine `--graph-cond-flags`. Do not invent other
    `cudaGraphCond*` create flags bits (only `ASSIGN_DEFAULT`).
    Do not invent a second `cuMemPoolGetId`.
    Do not invent a second `cuDeviceGetExecAffinitySupport`. Do not invent
    `cuCtxSetExecAffinity` (occupancy SM counts).
    Do not invent a second pool `PROT_READ`. Do not invent Engine
    `--pool-prot-read`.
    Do not invent a second `cudaMemAdvise` count /
    `mem_advise_with_size`. Do not invent Engine `--advise-size`. Do not
    invent partial-range `cudaMemAdvise`.
    Do not invent a second `cudaMemPrefetchAsync` count /
    `prefetch_with_size` / `prefetch_host_with_size`. Do not invent
    Engine `--prefetch-size`. Do not invent partial-range
    `cudaMemPrefetchAsync`.
    Do not invent a second `cudaStreamAttachMemAsync` length /
    `stream_attach_with_size`. Do not invent Engine `--attach-size`. Do
    not invent partial-range `cudaStreamAttachMemAsync`.
    Do not invent a second `cudaMemRangeGetAttribute` count /
    `mem_range_get_attribute_with_size`. Do not invent Engine
    `--range-size`. Do not invent partial-range
    `cudaMemRangeGetAttribute`.
    Do not invent a second `cudaHostRegister` size /
    `host_register_with_size`. Do not invent Engine `--register-size`.
    Do not invent partial-range `cudaHostRegister`.
    Do not invent a second `cudaMemRangeGetAttribute` `dataSize` /
    `mem_range_get_attribute_with_data_size` /
    `mem_range_get_attributes_with_data_sizes`. Do not invent Engine
    `--range-data-size`.
    Do not invent a second `cudaGraphUpload` stream /
    `upload_graph_async`. Do not invent Engine `--graph-upload-stream`.
    Do not invent a second `cudaGraphInstantiateWithParams`
    `hUploadStream` / `GraphInstantiateParams::upload_stream`.
    Stream-ordered `cuGraphUpload` / `hUploadStream` as stored-only
    (ignore the stream and still host-sync) stays weak.
    Do not invent a second `cudaGraphCondTypeIf` else /
    `graph_add_if_else`. Do not invent Engine `--graph-if-else`.
    Do not invent a second `graph_if_else_nodes`.
    Do not invent a second `cudaGraphConditionalNodeSetParams` /
    `cudaGraphExecConditionalNodeSetParams`. Do not invent Engine
    `--graph-cond-set`. Do not invent IF / WHILE / SWITCH SetParams that
    changes type, size, or bodies.
    Do not invent a second `cudaGraphExecUpdate` IF / WHILE / SWITCH
    handle copy. Do not invent Engine `--graph-cond-update`.
    Do not invent a second `CU_POINTER_ATTRIBUTE_ACCESS_FLAGS` /
    `pointer_get_access_flags`. Do not invent Engine `--pointer-access`.
    Do not invent `PointerAttr::AccessFlags` (no TLS current device;
    the typed helper takes an explicit `DeviceId`). Do not invent
    `CU_POINTER_ATTRIBUTE_CONTEXT` / P2P tokens.
    Do not invent a second `cudaMemsetNodeParams::elementSize` /
    `MemsetOp::element_size`. Do not invent Engine `--memset-element`.
    Do not invent a second `cuMemsetD16Async` / `cuMemsetD32Async` /
    `memset_d16_async` / `memset_d32_async`. Do not invent Engine
    `--memset-d16`. Do not invent a second `cuMemsetD2D16` / `D2D32` /
    `memset_d2d16_async` / `memset_d2d32_async`. Do not invent a second
    `cuMemsetD2D8` / `memset_d2d8_async`. Do not invent Engine
    `--memset-d2d`. Do not invent a memset fill value (this VM does not model memset
    stores into wait-value mailboxes).
    Do not invent a second `cuPointerGetAttributes` /
    `pointer_get_attribute_n`. Do not invent Engine `--pointer-attrs`.
    Do not invent a second `CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES` /
    `BatchMemOp::FlushRemoteWrites`. Do not invent Engine
    `--flush-remote`. Do not invent `CU_STREAM_MEM_OP_BARRIER` /
    `CU_STREAM_MEM_OP_MEMORY_BARRIER` (stream order already serializes
    wait/write in a batch).
    Do not invent a second `CU_STREAM_WAIT_VALUE_FLUSH` /
    `WaitValueFlags::FLUSH`. Do not invent Engine `--wait-flush`.
    Do not invent a second `cudaGraphAddNode` If / IfElse / While /
    `GraphNodeParams::If`. Do not invent a second
    `GraphNodeParams::Switch` / `switch_bodies`.
    Do not invent a second `graph_exec_child_get_graph` /
    `cudaGraphExecChildGraphNodeGetParams`. Do not invent Engine
    `--graph-exec-child`.
    Do not invent a second `graph_exec_event_record_get_event` /
    `graph_exec_event_wait_get_event`. Do not invent Engine
    `--graph-exec-event`.
    Do not invent a second `graph_exec_alloc_get_params` /
    `graph_exec_free_get_params`. Do not invent Engine
    `--graph-exec-mem`.
    Do not invent a second `graph_exec_kernel_node_copy_attributes`. Do
    not invent Engine `--graph-exec-copy-attrs`.
    Do not invent `cuDeviceGetLuid`
    (Windows).
    Do not invent gemma4 `attn_q.scale` / `attn_output.scale` / `attn_k.scale` /
    `attn_v.scale` as the writer-tiny. Do not invent a second dense
    `ffn_down.scale` / second `ffn_gate.scale` / second `ffn_up.scale` /
    second `attn_q.scale` / second `attn_output.scale` /
    second `attn_k.scale` / second `attn_v.scale` as a new family. Do not invent a second `--decode-sms` flag.
    Do not invent a second `--green-ctx`. Do not invent a second
    `ffn_down_exps.scale` / second gate/up scale as a new family.
    Do not invent `CU_GREEN_CTX_DEFAULT_STREAM` as a second NULL stream.
    Do not invent split `IGNORE_SM_COSCHEDULING` / max-cluster bits.
    Do not invent `cuCtxFromGreenCtx` (no `CUcontext` object).
    Do not invent occupancy SM counts (`cudaDevAttrMultiProcessorCount`,
    occupancy APIs). Do not invent
    `cudaGraphMemAllocNodeSetParams` (it would resize HBM),
    a second graph-alloc accessDescs / Engine `--graph-alloc-access`,
    `cudaDeviceGetStreamPriorityRange`, CUDA version
    numbers, example `$/M tokens` rents, or `cudaStreamDestroy`.
    **`best_of` / `use_beam_search` stay out of `parse_gen_req` until a
    real beam Engine exists.** Do not default `--engine`. Do not invent
    `max_model_len` or `/v1/tokenize`. Do not invent a second `gemma4`.
    Do not invent CUDA arrays on `memcpy_3d_batch_async`. Do not invent
    `ConcurrentManagedAccess` / discard contents to make batch prefetch
    succeed.
    Do not invent `ReuseFollowEventDependencies` /
    `ReuseAllowInternalDependencies` as Engine flags (stored-only; 0 does
    not insert waits). Do not invent `SetPreferredLocationHost` as a
    decode-path skip of kernel first-touch. Do not invent finite
    `--mempool-release N` (weaker than max-size OOM). Do not invent
    stream inherit (`set_stream_access_policy` /
    `set_stream_nvlink_util_centric`) — `kernel_with` / graph replay use
    launch/node, not stream. Do not invent
    `GraphInstantiateFlags::UPLOAD`, height/depth required cluster, or
    function LoadBalancing / MaxL1 occupancy standalone. Do not invent a
    second `MemcpySrcAccessOrder::Any` flag. Do not invent a second
    `AccessProperty::Streaming` flag. Do not invent `--l2-normal` (Normal
    vs Streaming is not mechanically distinct in gpu-sim billing). Do not
    invent a second `cudaStreamBeginCaptureToGraph` deps flag. Do not invent
    a second `graph_add_dependencies` flag. Do not invent a second
    `cudaGraphAddDependencies` v2 edgeData. Do not invent Programmatic
    graph dependency edges.     Do not invent a second `cuGraphAddNode_v2` /
    `graph_add_node_with_data`. Do not invent Engine `--graph-add-node-data`.
    Do not invent a second `cudaGraphAddMemAllocNode` accessDescs /
    `graph_add_alloc_with_access`. Do not invent Engine `--graph-alloc-access`.
    Do not invent a second `cudaGraphAddNode` mem-alloc accessDescs /
    `GraphNodeParams::Alloc` access field.
    Do not invent graph-alloc SetParams that only retargets accessDescs
    (SetParams of Alloc stays Invalid). Do not invent `poolProps` / a caller
    pool on graph mem-alloc nodes (always the device graph-memory pool).
    Do not invent `cudaGraphRemoveDependencies`
    edgeData. Do not invent Engine `--graph-edge-data`. Do not invent a second
    `cudaGraphAddHostNode` BETWEEN flag. Do not invent a second captured
    `cudaLaunchHostFunc` BETWEEN piecewise flag. Do not invent a second leaf
    `cudaGraphAddHostNode` / captured `cudaLaunchHostFunc` BEFORE GEMM flag.
    Do not invent a second `cudaGraphClone` of combo parents flag.
    Do not invent a second `cudaMemsetAsync` miss-fill flag.
    Do not invent a second live `cudaLaunchHostFunc` after miss DMA flag.
    Do not invent JOIN-style host
    after overlapping combo children (same wall as live `--host-func`
    after `launch_graph`). Do not invent `stream_update_capture_dependencies`
    as an Engine flag (same topology as begin-capture deps). Do not invent a
    second graph-build `cudaGraphSetConditional` flag. Live `set_conditional`
    before `launch_graph` is still wiped when `ASSIGN_DEFAULT` (unflagged
    `graph_conditional_create`). Do not invent a
    second `--graph-if` / IF wrap flag (SetParams upload tax is the wall vs
    `--graph-enable`). Do not invent a second `--no-read-mostly` /
    UnsetReadMostly flag (prefetch-move vs replicate is the wall vs default
    SetReadMostly). Do not invent a second `--no-preferred` /
    UnsetPreferredLocation flag (remote first-touch vs stay-on-home is the
    wall vs default SetPreferredLocation). Do not invent a second
    `--no-mem-prefetch` / skip fill prefetch (kernel first-touch vs
    copy-engine overlap is the wall vs default fill prefetch). Do not invent
    a second `--memcpy-attr` / demand `cudaMemcpyWithAttributesAsync`
    (API wait vs in-flight `memcpy_pinned_to_device` is the wall vs default
    demand H2D). Do not invent a second `--d2h-evict` / evict
    `cudaMemcpyAsync` Device→HostPinned (extra PCIe vs free-only is the wall
    vs default pinned/VMM evict). Do not invent a second `--d2h-pageable` /
    evict `cudaMemcpyAsync` Device→Host (host-sync bounce-buffer PCIe vs
    free-only is the wall vs default pageable pinned/VMM evict). Do not invent
    a second `--host-unregister` / `cudaHostUnregister` after miss DMA
    (re-register plus `synchronize` tax vs keep-registered is the wall vs
    `--host-register`). Do not invent a second `--ipc` /
    `cudaIpcGetMemHandle` (handshake tax vs no-handshake is the wall vs
    default `cudaMalloc`). Do not invent a second `--share-ptr` /
    `cudaMemPoolExportPointer` (per-page pointer handshake vs pool-level
    `--shareable` is the wall). Do not invent a second `--vmm-retain` /
    `cuMemRetainAllocationHandle`. Do not invent a second `--vmm-handle` /
    `va_create` plus `va_map_handle`. Do not invent
    `--memcpy-peer` host-sync pin_hot (alias of D2D; wall matches after
    `score()`). Do not invent `graph_add_empty` as a decode-path flag
    (1 ns join/fork). Do not invent a second `cudaGraphAddMemsetNode` of
    graph-mem scratch flag. Do not invent a second `cudaGraphAddMemcpyNode`
    of graph-mem scratch flag. Do not invent D2D memcpy between two
    different AllocIds (`MemcpyOp` names one alloc; PLAN 362 is H2D into
    scratch). Do not invent event record/wait BETWEEN combo
    children (1 ns). Do not invent graph wait/write_value BETWEEN combo
    children (1 ns Solo). Do not invent `graph_add_while` / `graph_add_switch`
    default-1 wrap (same wall as graph-build). Do not invent KernelAttrs
    SynchronizationPolicy as an Engine flag (graph kernel nodes shipped;
    still not a host-launch field). Do not invent a second
    `KernelNodeAttr::SynchronizationPolicy`. Do not invent
    `KernelAttrs::sync_policy`. Do not invent Engine `--kernel-sync-policy`.
    Do not invent PDL wait-only. Do not invent in-graph wait_value / event wait
    BEFORE a leaf (same GPU timeline as live wait). Do not invent host AFTER
    kernel on a leaf (exclusive-compute wall matches BEFORE). Do not invent
    `cuStreamBatchMemOp` packing of wait/write (1 ns Solo, not
    `launch_overhead_ns`). Do not invent a second `cuMemcpy2DUnaligned` /
    `memcpy_2d_unaligned`. Do not invent `cuMemcpy2DUnalignedAsync` (CUDA has
    no Async Unaligned). Do not invent Engine `--memcpy-unaligned`. Do not
    add 2D alignment checks to `memcpy_2d`. Do not invent a second
    `CUmemPoolProps::usage` / `MemPoolProps::usage` / `MemHandleUsage`.
    Do not invent Engine `--pool-usage`.     Do not invent hardware decompress
    succeeding (`MemDecompressAlgorithmMask` stays 0). Do not invent a second
    `CUmemAllocationProp` `allocFlags.usage` / `compressionType` /
    `MemAllocationProp::usage` / `compression`. Do not invent Engine
    `--vmm-usage`. Do not invent a second `cuMemcpyHtoD` / `memcpy_htod`.
    Do not invent a second `cuMemcpyDtoH` / `memcpy_dtoh`. Do not invent
    Engine `--memcpy-htod`. Do not invent `cuMemcpyDtoD` as a named alias
    of [`memcpy_peer`](Sim::memcpy_peer) (`cuMemcpyDtoDAsync` stays
    [`memcpy_device_to_device`](Sim::memcpy_device_to_device)). Do not invent
    a second `cuMemAllocPitch` / `malloc_pitch_with_element_size`. Do not
    invent Engine `--malloc-pitch-element`. Do not invent pitch that varies
    by `ElementSizeBytes` (this VM keeps 512-align). Do not invent a second
    CUDA_MEMCPY2D / CUDA_MEMCPY3D srcPos / dstPos / `MemcpyOp` origin
    fields. Do not invent Engine `--memcpy-origin`. Do not invent a second
    `cuMemcpy3DUnaligned` / `memcpy_3d_unaligned`. Do not invent
    `cuMemcpy3DUnalignedAsync` (CUDA has no Async Unaligned). Do not invent
    Engine `--memcpy-3d-unaligned`. Do not add 3D alignment checks to
    `memcpy_3d`. Do not invent a second CUDA_MEMCPY3D `srcLOD` / `dstLOD` /
    `MemcpyOp` `src_lod` / `dst_lod`. Do not invent Engine `--memcpy-lod`.
    Do not invent CUDA-array memcpy. Do not invent a second
    `cuDevSmResourceSplit` / `dev_sm_resource_split`. Do not invent Engine
    `--sm-split`. Do not invent occupancy SM counts on Split (`smCount`
    stays ‰). Do not invent `CU_DEV_SM_RESOURCE_GROUP_BACKFILL`. Do not
    invent `coscheduledSmCount` / `preferredCoscheduledSmCount` / workqueue
    resources. Do not invent a second CUDA_KERNEL_NODE_PARAMS.ctx /
    `KernelNodeParams::ctx`. Do not invent Engine `--kernel-ctx`. Do
    not put `ctx` on `Kind::Kernel`. Do not invent `cuCtxFromGreenCtx`.
    Do not invent a second CUDA_KERNEL_NODE_PARAMS.sharedMemBytes /
    `KernelNodeParams::shared_mem_bytes`. Do not invent Engine `--kernel-shared`.
    Do not put `sharedMemBytes` on `Kind::Kernel`. Do not invent a second
    CUDA_BATCH_MEM_OP_NODE_PARAMS.ctx / `BatchMemOpNodeParams::ctx`. Do not
    invent Engine `--batch-ctx`. Do not put `ctx` on `Kind::BatchMem`.
    Do not invent a second CUDA_CONDITIONAL_NODE_PARAMS.ctx /
    `GraphNodeParams::If` ctx. Do not invent an Engine flag for
    conditional ctx. Do not put `ctx` on `Kind::If` / `While` / `Switch`.
    Do not invent a second `graph_conditional_create_with_ctx`. Do not
    invent body-kernel ctx matching or rewrite. Do not invent
    `CUDA_CONDITIONAL_NODE_PARAMS` as a unified struct (If / IfElse /
    While / Switch stay split).
    Do not invent a second `cuGraphAddMemcpyNode` ctx /
    `MemcpyNodeParams::ctx`. Do not invent an Engine flag for memcpy ctx.
    Do not put `ctx` on `Kind::Memcpy` or [`MemcpyOp`].
    Do not invent a second `cuGraphAddMemsetNode` ctx /
    `MemsetNodeParams::ctx`. Do not invent an Engine flag for memset ctx.
    Do not put `ctx` on `Kind::Memset` or [`MemsetOp`].
    Do not invent a second `CUgraphChildGraphNodeOwnership` /
    `ChildGraphNodeParams::ownership`. Do not invent an Engine flag for
    child-graph ownership. Do not put `ownership` on `GraphNodeKind`.
    Do not invent a second graph kernel-node SynchronizationPolicy
    Get/Set/CopyAttributes. Do not invent KernelAttrs SynchronizationPolicy.
    Do not invent an Engine flag for kernel-node sync policy.
    Do not invent a second
    `cudaLaunchAttributeProgrammaticEvent.triggerAtBlockStart`.
    Do not invent an Engine flag for programmatic-event block-start.
    Do not invent `LaunchCompletionEvent::trigger_at_block_start`.
    Do not invent per-block programmatic event records.
    Do not invent a must-be-0 flags word on ProgrammaticEvent (external stays
    `cudaEventRecordExternal` and must be false).
    Do not invent a `flags: u32` field on ProgrammaticEvent /
    LaunchCompletionEvent. Do not invent a second External / interprocess
    rejection on those attributes. Do not invent disable-timing as a
    requirement for launch-attribute events. Do not invent Engine
    `--programmatic-event-external` / `--launch-completion-external`.
    Do not invent NvSciSync / interop event kinds as a second interprocess
    check.
    Do not invent a second device-updatable opt-out, destroy, CopyAttributes,
    second instantiate, or ExecUpdate. Do not invent `CUgraphDeviceNode` or
    device-side kernel-node updates. Do not invent an Engine flag for
    device-updatable opt-out.
    Do not invent a second `cudaGraphKernelNodePortLaunchCompletion` /
    [`GraphEdgeData::launch_completion`]. Do not invent Engine
    `--graph-edge-port`. Do not invent nonzero `to_port`. Do not invent
    [`GraphKernelNodePort::PROGRAMMATIC`] edges. Do not invent a second
    ExecUpdate launch-completion port topology check.
    Do not invent a second UseNodePriority ExecUpdate priority check.
    Do not invent Engine `--graph-node-priority-update`. Do not invent
    child-graph SetParams nested priority as a second check.
    `graph_exec_kernel_node_set_priority` stays legal. Do not invent
    `cudaDeviceGetStreamPriorityRange` clamping for this comparison.
    Do not invent a second 2D/3D memset ExecUpdate geometry check.
    Do not invent Engine `--memset-update`. Do not invent a memset fill
    value. Do not invent 1D memset work-resource mapping failures.
    `graph_exec_memset_set_params_2d` stays legal.
    Do not invent a second memcpy memory-type ExecUpdate check.
    Do not invent Engine `--memcpy-update`. Do not invent CUDA-array
    memcpy as a memory-type variant. `graph_exec_memcpy_set_params` stays
    legal.
    Do not invent a second `cudaMemPoolAttrMaxPoolSize` /
    [`MemPoolAttr::MaxPoolSize`] / `set_pool_max_size`. Do not invent Engine
    `--mempool-max` as SetAttribute-only (CLI stays `create_pool_with_props`
    plus `set_device_mempool`). Do not invent alignment rounding of
    MaxPoolSize. Do not invent `cudaMemPoolAttrHwDecompressEnabled`. Do not
    invent pool `LocationType` / `LocationId` as MemPoolAttr (HostNuma /
    Invisible stay unmodeled). Do not invent shrinking MaxPoolSize freeing
    live allocs.
    Do not invent a second `cudaMemPoolAttrAllocationType` /
    [`MemPoolAttr::AllocationType`]. Do not invent a second
    `cudaMemPoolAttrExportHandleTypes` / [`MemPoolAttr::ExportHandleTypes`].
    Do not invent Engine `--pool-alloc-type` / `--pool-export-handles`. Do
    not invent Set of AllocationType / ExportHandleTypes. Do not invent NT
    or fabric mempool handle types.
    Do not invent a second kernel-function ExecUpdate check.
    Do not invent Engine `--graph-function-update`. Do not invent
    `UnsupportedFunctionChange` (no `CUfunction` signature or destroy).
    Do not invent child SetParams nested kernel function as a second check.
    `graph_exec_kernel_set_params` stays legal. Do not treat Other FLOPs or
    GEMM shape as function identity.
    Do not invent a second CUDA-named `graph_exec_memcpy_set_params` that
    accepts 2D/3D (CUDA ExecMemcpy SetParams is 1D-only). Do not reverse
    PLAN 190 (`graph_exec_memcpy_set_params_1d` converting 2D to 1D). Do
    not reverse extra `graph_exec_memcpy_set_params_2d` plus
    `graph_exec_memcpy_set_params_3d` helpers. Do not invent Engine
    `--graph-memcpy-1d`. Do not invent CUDA-named
    `cudaGraphExecMemcpyNodeSetParams2D`. Do not invent ExecUpdate 2D
    memcpy geometry as ParametersChanged.
    Do not
    spend the next item on an OpenAI-compatible HTTP veneer.

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
