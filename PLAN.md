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
- [ ] More than one real-model fixture (NEOX Qwen + NORM Llama control)
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
- host-synchronous `cudaMalloc` / `cudaFree` / `cudaMemcpy`
  (`Sim::malloc` / `free_sync` / `memcpy_sync`) vs stream-ordered
  `cudaMallocAsync` / `cudaFreeAsync` / `cudaMemcpyAsync`
  (`alloc` / `free` / `memcpy`); `malloc` OOM is at the call;
  `cudaDeviceSynchronize` (`synchronize_device`) waits one GPU
- memory pools: `create_pool` / `alloc_from_pool` /
  `set_pool_release_threshold` / `pool_trim_to`
  (`cudaMemPoolCreate` / `cudaMallocFromPoolAsync` /
  `cudaMemPoolAttrReleaseThreshold` / `cudaMemPoolTrimTo`); default
  threshold `0` returns unused bytes on free; `u64::MAX` holds them so
  `malloc` can OOM until trim; `cudaMalloc` cannot consume pool cache
- `cudaHostRegister` / mapped host (`alloc_host`, `host_register`,
  `host_register_mapped`, `alloc_host_mapped`): pin existing pageable
  memory; mapped pointers are kernel-readable over PCIe with no H2D and
  no HBM charge; `host_pin_bytes` is the `mlock` cap (`PinOom`)
- `cudaMallocManaged` / `cudaMemPrefetchAsync` (`alloc_managed`,
  `prefetch`, `prefetch_host`): no HBM until migrate; prefetch **moves**
  (does not replicate) unless `cudaMemAdviseSetReadMostly`
  ([`Sim::mem_advise`] [`MemAdvise::SetReadMostly`]); a kernel first-touch
  prefetches when that kernel *starts* (after stream deps) unless
  [`MemAdvise::SetAccessedBy`] maps that
  GPU or [`MemAdvise::SetPreferredLocation`] already holds the page at
  another GPU (remote read, interconnect billing; writes still migrate;
  host preferred does not skip first-touch)
- `cudaStreamAttachMemAsync` (`stream_attach`, `alloc_managed_host`):
  Global / Host / Single visibility; Host and other-stream Single fail
  device kernels / memset / device prefetch (`not attached`); Single
  cannot use the NULL stream; capture is refused
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

   Shipped in `gpu-sim::Score` (not a rental price): `wall_ns`, `hbm_peak`,
   `bytes_moved`, `energy_uj` (`node_tdp_mw * wall_ns / 1e6` µJ), optional
   `ns_per_token`, `ttft_ns`, `itl_ns`. `sim_replay` samples the virtual clock
   after each token. There is no `$/M tokens` field.

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
  launch uploads if needed. `update_graph` replaces an instantiated exec's
  steps when device, stream, and op kinds match (`graph_update_ns`).
  Graphs with mem alloc/free nodes cannot be updated.
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
  instantiate (`graph_clone_ns`; the src is destroyed). Capture after a miss waits with
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
  `--host-func`, blocking compute, `--pageable`, `--accessed-by`,
  `--legacy-null`, `--stream-priority`, `--graph-update`, `--graph-clone`, and `--timing-events`. `--mempool` sets the default
  pool release threshold to `u64::MAX` (vLLM-style hold); reuse of a
  cached page pays `pool_reuse_ns`. `--mapped` is `cudaHostAllocMapped`
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
  walker). `--graph-clone` is `cudaGraphClone` of a leaf capture before
  instantiate (graph vs exec). `--timing-events` is timing-on copy events
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
    `--graph-clone` / `--timing-events` / `--cuda-graphs` match
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
