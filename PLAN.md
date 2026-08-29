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

Deliberately *not* yet: production serving, mmap, CUDA, batching. That is
correct. The next high-signal layer is **not** OpenAI HTTP, Tokio, or a
tok/s race.

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
token whose hash already completed on another sequence.

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
router, not a 320B result. A real Qwen3MoE GGUF is required before the
kill-switch (“best non-oracle ≈ random ≈ 18% → stop”).

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
| `SimulatedGpuStore` | fake HBM capacity, PCIe/NVLink bandwidth, DMA concurrency; `with_managed` is UM prefetch, `with_mapped` is zero-copy host (and `host_pin_bytes` occupancy), `with_vmm` is `va_acquire`; `with_cfg` is `host_func` / blocking streams / `sync_alloc` / mempool / `vmm_page` / pageable H2D / `SetAccessedBy` / legacy NULL / stream priority / `graph_update` / `graph_clone`; `expertvm store` / `store_replay_cfg` is the CLI (Markov prefetch) |

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
- CUDA VMM (`va_reserve` / `va_map` / `va_unmap` / `va_free`,
  `va_acquire` / `va_release`, `va_map_range` / `va_unmap_range`):
  `cuMemAddressReserve` / `cuMemMap` / `cuMemUnmap` / `cuMemAddressFree`;
  HBM charged only while mapped; the pointer survives unmap;
  `va_release` parks the VA so `va_acquire` remaps without another reserve;
  sparse sub-range maps (vLLM KV-block analog) charge only the mapped span;
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
- stream ordering, events, barriers
- kernel enqueue, async copies
- copy-engine availability, peer accessibility
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
  thrash, coding/chat/long-context traces, batch 1 vs 128,
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
  Recorded kernels and copies do not run until launch; alloc/free cannot
  be captured (`malloc` / `free_sync` / `memcpy_sync` / `synchronize_device`
  included); mempool create/trim/set-attribute cannot be captured;
  instantiate/update/upload/clone/destroy cannot be captured; capture requires an idle stream.
  Instantiate is host-sync (`graph_instantiate_ns`); the first launch
  instantiates if needed. `cudaGraphUpload` (`Sim::upload_graph`,
  `graph_upload_ns`) is a separate host-sync after instantiate; the first
  launch uploads if needed. `update_graph` replaces an instantiated exec's
  steps when device, stream, and op kinds match (`graph_update_ns`).
  `clone_graph` is an independent uninstantiated copy (`graph_clone_ns`);
  child-graph nodes are cloned recursively (shared children cloned once).
  `destroy_graph` is `cudaGraphDestroy` (later launch is unknown).
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
  `--legacy-null`, `--stream-priority`, `--graph-update`, and `--graph-clone`. `--mempool` sets the default
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
  implies `--vmm`). `expertvm kv` demand-pages a reserved VA (`kernel_bufs`
  + H2D or `memset_buf` at a mapped span; peak HBM is the working set). `--host-func` is `cudaLaunchHostFunc` after each event's
  GEMMs (`host_func_ns`; no GPU occupancy). `--blocking-streams` is
  `cudaStreamCreate` on seq-streams (serialize with NULL); default is
  `cudaStreamNonBlocking`. `--legacy-null` is `set_legacy_null_stream`
  (NULL serializes with every stream). `--pageable` is host-sync
  `memcpy_host_to_device` (`pageable_permille`). `--stream-priority` is
  `cudaStreamCreateWithPriority` on seq-streams (priority = stream id). `--graph-update`
  is `cudaGraphExecUpdate` of a parked leaf (store and `--cuda-graphs`
  walker). `--graph-clone` is `cudaGraphClone` of a leaf capture before
  instantiate (graph vs exec). `memset` / `memset_buf` of a mapped span, directed peer enable, and
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
