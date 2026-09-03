# ChatGPT share extract (visible HTML)

Source: https://chatgpt.com/share/6a920fe1-48ac-83ea-9404-7f4c1062c17e  
Fetched 2026-08-28. The share **was reachable**. This file is the five user
turns and five assistant replies as shown on the public HTML page.

The complete share-API extract (search batches, thought summaries, cited
URLs, raw JSON) is in [`chatgpt-share-6a920fe1/`](chatgpt-share-6a920fe1/).
Plugin/tool result bodies are redacted in ChatGPT’s share payload.

The historical research plan derived from this transcript is [`PLAN.md`](../PLAN.md). Live agent execution is [`CIVILIZATION.md`](../CIVILIZATION.md).

---

## User

I think there could be large market for making really good mid tier LLM type models really nice to use and extremely cheap to serve at scale. like GLM-5.3-Flash concept at scale. If I wanted to either start my own shop that researches this and does inference, or only does inference but just get's insanely good at doing it. What OSS libs should I make? Focus on Rust only(we will only build in Rust).


## ChatGPT

Yes. There is a plausible company thesis here: as open-weight models become increasingly capable, part of the economic advantage shifts from “who owns the weights?” toward “who can turn those weights into the cheapest reliable token stream?” GLM-5.3-Flash is a good example: 320B total parameters, only 18B active per token, hybrid sparse/linear attention, and Z.ai claims roughly one-tenth the serving price of GLM-5.2\. Z.ai+1

I would not begin by making another generic Rust ML framework. Burn/CubeCL already occupy that layer, `cudarc` gives Rust good access to CUDA/cuBLASLt/NCCL/etc., and `mistral.rs` already has continuous batching, paged attention, prefix caching, quantization, speculative decoding, multi-GPU, and model serving. GitHub+2GitHub+2

Instead, build the missing inference-systems primitives as independent crates.

### The stack I would build

| Priority | Library         | What it does                         | Company value     |
| -------- | --------------- | ------------------------------------ | ----------------- |
| 1        | kv-fabric       | distributed KV/state memory          | Extremely high    |
| 2        | moe-runtime     | optimized sparse MoE execution       | Extremely high    |
| 3        | infer-scheduler | continuous batching + SLO scheduling | Very high         |
| 4        | quant-rs        | hardware-aware mixed quantization    | Very high         |
| 5        | attention-rs    | modern attention/state kernels       | Very high         |
| 6        | speculate-rs    | generalized speculative decoding     | High              |
| 7        | tensor-loader   | ultra-fast weight loading/resharding | High              |
| 8        | infer-bench     | standardized inference benchmarking  | High OSS leverage |
| 9        | model-ir        | inference-oriented model IR/compiler | Huge, but later   |

The first five are where I think the real opportunity lies.

---

## 1\. `kv-fabric`: Rust-native LLM memory fabric

This would probably be my #1 company-level bet.

Think:

```
GPU HBM
   ↕
local DRAM
   ↕
remote DRAM
   ↕
NVMe
   ↕
other GPU nodes

```

But the API understands that what it is moving is specifically LLM state:

```
KvHandle
SequenceId
PrefixId
LayerRange
TokenRange
Device
MemoryTier

```

Features:

```
paged KV allocation
prefix deduplication
radix prefix cache

GPU ↔ GPU
GPU ↔ host
host ↔ remote host
GPU ↔ remote GPU

RDMA
GPUDirect RDMA
NVLink
TCP fallback

async transfers
transfer/compute overlap

KV eviction
KV compression
KV prefetch

distributed cache directory

prefill → decode state transfer

multimodal encoder-state caching

linear-attention recurrent-state transfer

```

This is increasingly becoming a fundamental layer of inference infrastructure.

vLLM and SGLang now support disaggregated prefill/decode, and systems like Mooncake and NVIDIA NIXL exist specifically because state movement has become a first-class inference problem. Mooncake reports 75% more requests under real Kimi workloads from its KV-centric architecture. NVIDIA Developer+3vLLM+3SGLang Documentation+3

A good Rust version could effectively become:

> **Tokio for LLM memory movement.**

Not an inference engine. A primitive every inference engine can consume.

And Rust is unusually appropriate here because you have:

```
lifetimes
ownership
async
zero-copy abstractions
memory safety
explicit resource ownership
strong concurrency

```

exactly around an area filled with pinned buffers, registered RDMA regions, GPU pointers, asynchronous DMA, cache ownership and cancellation.

That is a serious OSS project.

---

# 2\. `moe-runtime`: a dedicated Rust MoE execution engine

This is probably the most directly relevant project to your GLM-5.3-Flash thesis.

GLM-5.3-Flash illustrates why:

```
Total model:       320B
Active/token:       18B

```

Hugging Face+1

That means the model is computationally closer to a much smaller model than its total parameter count suggests.

But you still have **320B parameters worth of experts to store and make available**.

That turns MoE serving into a systems problem:

```
Where should each expert live?

Which experts are hot?

Can experts be replicated?

Can cold experts live in host RAM?

Can they be prefetched?

Can neighboring requests be batched by expert?

Can expert routing inform scheduling?

Can hot experts use FP8 while cold ones use FP4?

Can frequently co-activated experts be colocated?

Can we rebalance expert placement dynamically?

```

I would make the abstraction something like:

```
trait ExpertStore {
    async fn acquire(
        &self,
        expert: ExpertId,
        device: DeviceId,
    ) -> ExpertLease;
}

```

Then separately:

```
Router
ExpertPlacement
ExpertCache
ExpertExecutor
ExpertPrefetcher
ExpertProfiler

```

The executor should support:

```
grouped GEMM
fused router + dispatch
FP8
NVFP4
MXFP4
INT4
expert parallelism
tensor parallelism
dynamic expert replication
CPU/DRAM expert offloading

```

This can eventually become almost a specialized database:

```
tokens = queries
experts = partitions
HBM = cache
DRAM = secondary storage
routing = query plan

```

And the runtime continuously optimizes placement from observed traffic.

That could become a genuine competitive advantage.

---

# 3\. `infer-scheduler`: the scheduler separated from the inference engine

Current inference systems have increasingly sophisticated schedulers because batching directly determines economics.

Build a standalone Rust scheduling library.

Input:

```
Request {
    prompt_tokens,
    expected_output_tokens,
    deadline,
    priority,
    tenant,
    cached_prefix,
    model,
    adapter,
}

```

Hardware:

```
Worker {
    model,
    kv_available,
    compute_available,
    cached_prefixes,
    cached_experts,
}

```

Output:

```
Schedule {
    batch,
    worker,
    prefill_chunks,
    decode_slots,
}

```

Then support algorithms for:

```
continuous batching
chunked prefill
prefill/decode separation
prefix-aware routing
expert-aware routing
deadline scheduling
priority scheduling
fairness
tenant quotas
backpressure
early rejection
admission control
speculative-decoding-aware scheduling
latency vs throughput optimization

```

Mooncake's scheduler even performs SLO-aware early rejection rather than wasting GPU cycles on work it predicts cannot meet the requested latency objective. KVCACHE AI

Rust is a particularly good language for this because the hot scheduling path really should not involve:

```
Python GC
Python object graphs
Python interpreter overhead

```

SGLang itself explicitly advertises a "zero-overhead CPU scheduler", which tells you how important this layer has become. GitHub

---

# 4\. `quant-rs`: hardware-aware quantization rather than "convert to INT4"

I think quantization tooling has substantial room for improvement.

Don't make:

```
model -> AWQ

```

Make:

```
model + workload + hardware + quality constraint
                ↓
          optimized model

```

For example:

```
Layer 0       BF16
Layer 1       FP8
Layer 2       FP8
Expert 1      NVFP4
Expert 2      NVFP4
Expert 19     FP8
attention KV  FP8
embedding     BF16
lm_head       FP8

```

The optimizer searches for:

```
minimum $/M tokens

subject to:

MMLU drop < x
coding drop < y
perplexity < z
TTFT < target
ITL < target
VRAM < target

```

Something like:

```
let result = QuantOptimizer::new(model)
    .hardware(Hardware::B200)
    .objective(Minimize::CostPerToken)
    .max_quality_loss(0.005)
    .optimize();

```

`mistral.rs` already has an interesting version of this idea through per-layer quantization topology and hardware-aware quant-method selection. GitHub

I would push the idea much further.

For MoEs in particular, quantization doesn't necessarily need to be uniform across experts.

That's an interesting research area in itself.

---

# 5\. `attention-rs`: modern attention primitives

Don't build merely another FlashAttention implementation.

The architecture frontier is moving beyond homogeneous transformer attention.

GLM-5.3-Flash contains a hybrid of approximately:

```
34 Kimi Delta Attention layers
11 MLA / sparse-attention layers

```

plus MoE and mHC residual structure. Sebastian Raschka, PhD

SGLang already has specialized support for GDN linear attention, DeepSeek sparse attention, MLA, FlashAttention and different prefill/decode attention backends. GitHub

Rust needs an equivalent low-level vocabulary.

I'd want:

```
trait AttentionState { ... }

trait AttentionKernel {
    fn prefill(...);
    fn decode(...);
}

```

Implementations:

```
MHA
GQA
MLA
DSA
KDA
GDN
sliding window
block sparse
linear/recurrent attention

```

And importantly:

```
prefill implementation != decode implementation

```

because they are fundamentally different workloads.

Using CubeCL for portable kernels where it performs well would keep the source Rust-native while allowing CUDA/HIP/Metal backends. CubeCL now targets CUDA, ROCm, Metal, Vulkan, WebGPU and CPU from Rust-defined kernels. GitHub

For absolute NVIDIA performance, you can also wrap vendor primitives using `cudarc`.

---

# 6\. `speculate-rs`: speculative execution as a reusable library

Today engines tend to implement this internally.

I'd extract:

```
draft models
MTP
EAGLE
n-gram
suffix matching
tree speculation
adaptive speculation
heterogeneous draft hardware

```

into a generalized runtime.

Something interesting happens at hyperscale:

A speculation strategy shouldn't be static.

Imagine:

```
request begins
      ↓
measure acceptance
      ↓
acceptance 85%
      ↓
increase speculative depth

later acceptance 35%
      ↓
decrease depth

```

Or even:

```
coding request → draft model A
creative text → draft model B
JSON → grammar-aware speculation

```

Both SGLang and vLLM now have fairly elaborate speculative-decoding implementations, showing that the problem has become important enough to deserve its own systems layer. SGLang Documentation+1

---

# 7\. `tensor-loader`: make loading 300B parameters boring

Another underappreciated problem.

A server shouldn't spend forever doing:

```
download
deserialize
allocate CPU tensors
copy
transform
reshard
copy GPU

```

Build:

```
Safetensors
      ↓
memory map
      ↓
direct tensor views
      ↓
parallel decompression/transform
      ↓
GPU placement

```

Support:

```
zero-copy mmap
async file IO
GPUDirect Storage
parallel shard loading
tensor parallel resharding
expert parallel resharding
quantization during load
streaming model startup
lazy expert loading
remote object storage

```

Then you can do fun things like:

> Start accepting requests before every cold MoE expert has been loaded.

That becomes valuable for autoscaling.

---

# 8\. `infer-bench`: MLPerf-like serving benchmarks that people actually enjoy using

This sounds less glamorous, but I would build it **first**.

Every optimization you do subsequently needs measurement.

Something like:

```
infer-bench \
  --server http://localhost:8000 \
  --trace coding.jsonl \
  --concurrency 256

```

Outputs:

```
TTFT p50/p95/p99
ITL p50/p95/p99

input tok/s
output tok/s
total tok/s

GPU utilization
HBM bandwidth
KV utilization
prefix hit rate
expert cache hit rate
speculation acceptance

joules/token

$/1M input tokens
$/1M output tokens

SLO attainment %

```

And workload replay:

```
ChatGPT-like
coding agent
long context
RAG
short chat
batch extraction
tool-calling agents

```

This becomes incredibly useful for your own research.

More importantly, it gives you a way to publish claims like:

```
moe-runtime-rs
vs vLLM
vs SGLang

B200 × 8

+31% output tokens/sec
-19% p99 ITL
-24% $/M tokens

```

That attracts exactly the engineering community you'd eventually hire from.

---

# Then there is the very large bet: `model-ir`

This one I'd postpone.

Imagine an inference-specific IR:

```
HuggingFace model
      ↓
Model IR
      ↓
architecture lowering
      ↓
hardware lowering
      ↓
optimized execution graph

```

Rather than hand-implementing every:

```
Qwen
GLM
DeepSeek
Llama
Kimi
Mistral

```

architecture.

The IR understands concepts such as:

```
Attention
LinearAttention
MoE
Router
RMSNorm
RoPE
Residual
Conv
VisionEncoder

```

Then compiler passes perform:

```
operator fusion
quantization
parallelism partitioning
expert placement
memory planning
kernel selection
CUDA graph formation
prefill/decode specialization

```

That eventually becomes something resembling:

> LLVM for LLM inference.

Enormous upside.

Also enormous scope.

Don't start there.

---

## What I specifically would NOT build

I would avoid spending meaningful time building another:

```
generic tensor library
generic autograd engine
tokenizer
Safetensors parser
OpenAI-compatible HTTP server
basic CUDA wrapper
Llama-only inference engine
llama.cpp clone
generic async runtime

```

Those aren't where the differentiation is.

Use existing Rust:

```
Tokio
cudarc
CubeCL
safetensors
tokenizers
serde
axum/hyper

```

and contribute upstream when necessary. `cudarc`, for example, already exposes CUDA Driver, NVRTC, cuBLAS/cuBLASLt, NCCL, cuDNN, cuFILE, CUPTI and other primitives through Rust. GitHub

---

# The architecture I'd ultimately aim for

```
                    API
                     │
               infer-router
                     │
              infer-scheduler
                /         \
               /           \
          PREFILL         DECODE
             │               │
             └──── kv-fabric ┘
                     │
        ┌────────────┼────────────┐
        │            │            │
   attention-rs  moe-runtime  speculate-rs
        │            │            │
        └───────── quant-rs ──────┘
                     │
                 model-ir
                     │
            CubeCL / cudarc
                     │
      CUDA / ROCm / Metal / etc.

```

And:

```
                 infer-bench
                      ↑
              measures everything

```

The crates remain useful independently.

Your commercial system combines them.

That is a very good OSS/business boundary.

---

## If the target is specifically "GLM-5.3-Flash economics"

I'd prioritize differently:

```
1. moe-runtime
2. kv-fabric
3. attention-rs
4. quant-rs
5. infer-scheduler
6. infer-bench
7. speculate-rs

```

Why?

A model like GLM-5.3-Flash is basically a giant systems-engineering invitation.

It has:

```
320B stored parameters
18B active parameters

sparse MoE
linear attention
sparse attention
1M context
multimodality

```

Hugging Face+1

The naive inference problem is:

> "How do I execute a 320B model?"

The much more interesting problem is:

> **"How little of this 320B model do I actually need to touch for each token?"**

That question leads directly to:

```
expert placement
expert caching
expert prefetch
quantization
state caching
prefix reuse
sparse execution
speculative execution
request scheduling

```

And that's where I think your proposed company could get genuinely interesting.

My first OSS repo would therefore probably be **`moe-runtime-rs`**, but I would quietly build `infer-bench` alongside it from day one. If `moe-runtime-rs` can eventually take an open MoE and demonstrate materially lower **$/output-token at fixed latency and quality** than vLLM/SGLang on identical hardware, you have the beginnings of something much bigger than a Rust ML hobby project. GitHub+1


## User

from first principles. which one actually fills gaps to raise eyebrows and gain traction for filling a real void


## ChatGPT

The one I would actually build is narrower than any of my earlier suggestions:

**A Rust-native dynamic expert-residency / MoE memory virtualization library.**

Not “another MoE runtime.” Not “another KV cache.” Something more specific:

> Make a 300B–1T sparse MoE behave as though its experts live in a giant virtual memory space, while automatically keeping the right experts in GPU HBM.

Call it something like `expertvm`.

From first principles, this attacks the strange economic property of modern MoEs:

```
GLM-style sparse MoE

320B parameters stored
 ~18B parameters used per token

```

You pay for enough HBM to hold an enormous model even though each token touches only a fraction of it.

That is waste.

The inference system therefore has two fundamentally different resources:

```
compute requirement  ≈ active parameters
memory requirement   ≈ total parameters

```

Sparse MoE has dramatically reduced the first without proportionally reducing the second.

That gap is the opportunity.

If `expertvm` could make:

```
HBM
  ↓ overflow
other GPU HBM
  ↓
host DRAM
  ↓
NVMe

```

look like one expert address space, you could potentially serve large sparse models with less premium GPU memory.

And **that directly attacks $/token**.

What makes this more interesting is that the ecosystem does not have a clean, mature, engine-independent answer to it yet.

DeepEP is excellent, but its core abstraction is primarily **moving tokens to resident experts**:

```
token
  ↓
dispatch
  ↓
remote expert
  ↓
combine

```

It is now extremely sophisticated—GPU-initiated communication, scale-out to EP2048, computation overlap, etc. GitHub

That means trying to compete with DeepEP on all-to-all is a poor wedge.

FlashInfer similarly already covers fused MoE, grouped GEMM, FP8/FP4, routing, attention and distributed communication. GitHub

Again: don't compete there.

The interesting inverse problem is:

```
Instead of:

move token → expert

ask:

should expert → move?
where should expert live?
how many copies should exist?
when should it move?
can I know that before it is needed?

```

That is a much less settled problem.

And there are strong signals that the frontier projects themselves are moving toward it. DeepEP explicitly lists an ongoing feature for elastic GPU/CPU buffers that map a contiguous virtual address space over hybrid GPU and CPU physical memory. GitHub SGLang just added DWDP for MoE prefill, which prefetches peer expert weights over NVLink and computes them locally; they explicitly describe it as early development, while already reporting up to 1.92× improvement in one configuration. What's New

That is exactly the kind of signal I look for:

> very smart teams are independently reaching toward the primitive, but nobody owns the abstraction yet.

### What `expertvm` actually does

Imagine the inference engine asks:

```
let expert = experts.acquire(
    layer,
    expert_id,
    device,
).await?;

```

The engine doesn't care where it currently exists.

Internally:

```
                    Expert 193

                         │
                  residency table
                         │
       ┌─────────────────┼─────────────────┐
       │                 │                 │
    GPU 3 HBM        CPU DRAM          NVMe/object
     hot copy         warm copy          cold copy

```

And the runtime is continuously answering:

```
Which experts are hot?

Which experts tend to activate together?

Which GPU will need this expert next?

Should I replicate it?

Should I evict something else?

Should I transfer the expert,
or transfer the tokens?

Can I hide the transfer underneath
the previous layer's computation?

```

That last question is especially important.

Suppose you're computing layer 17.

The router for layer 18 determines:

```
batch needs experts:
4, 17, 23, 91, 106

```

If 106 isn't local:

```
compute layer 17
       │
       ├──────── DMA expert 106 ────────┐
       │                                │
       ▼                                ▼
layer 17 finishes                 expert ready
                                       │
                                       ▼
                                  layer 18

```

Now the expensive movement can disappear behind useful GPU work.

Eventually you get much more sophisticated policies.

For example:

```
expert 17:
    activation frequency = 14%
    replicate everywhere

expert 82:
    activation frequency = 0.02%
    keep in CPU RAM

expert 193:
    heavily used by coding prompts
    replicate on coding-serving workers

expert 221:
    correlated with experts 37 and 91
    colocate them

```

Now you're applying cache/locality theory to neural network weights.

That is the part I think could raise eyebrows.

### Why this is more compelling than the other ideas

`kv-fabric` has enormous value, but Mooncake has already become formidable. It supports DRAM/NVMe hierarchies, distributed KV storage, weight management, replication, RDMA transfer and integrations across SGLang, vLLM and TensorRT-LLM. PyPI

I would not try to beat Mooncake head-on.

`attention-rs` also isn't enough of a void. FlashInfer already exposes paged/ragged attention, MLA, sparse attention, fused prefill/decode and a large collection of highly tuned kernels. GitHub

`quant-rs` is useful, but there's an enormous amount of quantization work happening.

`infer-scheduler` is valuable, but schedulers are deeply coupled to execution engines. You'd have difficulty getting vLLM/SGLang to outsource their core scheduler to a Rust crate.

`model-ir` is fascinating but far too broad.

`infer-bench` would get users, but probably doesn't create the “holy shit, this changes serving economics” reaction.

Expert memory virtualization can.

### The eyebrow-raising demonstration

Don't initially advertise the Rust.

Advertise this:

> **Run a 320B MoE efficiently with 160 GB of HBM.**

Or:

> **Same GLM-5.x throughput using 4 GPUs instead of 8.**

Or:

> **Serve a 1T-parameter MoE while keeping only 30% of expert weights resident in HBM.**

Obviously the exact achievable numbers need experimentation.

But that's the type of result that matters.

Nobody cares that:

```
expertvm is memory safe!

```

People care that:

```
vLLM + expertvm
H200 x4

$0.19 / M tokens

```

beats:

```
vLLM
H200 x8

$0.31 / M tokens

```

at equivalent latency.

Then people ask:

> What the hell is `expertvm`?

That's how an infrastructure library gets traction.

### And Rust actually contributes something here

This also happens to be unusually well matched to Rust.

The runtime will contain complicated states like:

```
enum Residency {
    Loading {
        from: Tier,
        to: Device,
        transfer: TransferHandle,
    },

    Resident {
        device: Device,
        allocation: DeviceAllocation,
    },

    Replicated {
        copies: SmallVec<[DeviceAllocation; 4]>,
    },

    Evicting {
        allocation: DeviceAllocation,
        destination: Tier,
    },

    Cold {
        location: StorageLocation,
    },
}

```

You really don't want:

```
expert evicted
but CUDA kernel still references it

buffer freed
while RDMA operation is in flight

two workers independently loading same expert

GPU disappears
but residency directory thinks its copy exists

cancellation occurs halfway through migration

```

These are ownership/state-transition problems.

Rust gives you a genuine architectural advantage there rather than merely being the language you happen to prefer.

### Keep the first version extremely narrow

I wouldn't start with NVMe, RDMA, prediction, cluster scheduling and everything else.

V0:

```
single node
8 GPUs

HBM ↔ HBM
CPU pinned DRAM ↔ HBM

expert registry
expert leases
async migration
LRU/LFU residency
hot expert replication
layer-ahead prefetch
metrics

```

And integrate one model:

```
Qwen/GLM/DeepSeek sparse MoE

```

Then benchmark:

```
static EP
vs
expertvm

```

while intentionally restricting available HBM.

V1:

```
traffic-aware placement
co-activation placement
adaptive replication

```

V2:

```
RDMA
multi-node
remote expert residency

```

V3 gets interesting:

```
predictive residency

```

The runtime learns:

```
P(expert_j at layer L+1 |
  experts selected at layers L,L-1,...,
  prompt class,
  session)

```

and prefetches experts before the router even asks for them.

Now you're crossing from pure systems work into inference research.

That aligns almost perfectly with the company you described: **research into making good mid-tier/open models absurdly cheap to run.**

So if you asked me to put your next year of OSS effort behind exactly one bet:

**Don't build `moe-runtime-rs`.**

Build:

> **`expertvm`: engine-agnostic virtual memory and dynamic residency for sparse model weights.**

Design it so vLLM/SGLang/mistral.rs/etc. can eventually use it rather than requiring anyone to adopt your inference server.

That is the first idea from the original list where I see the combination of **real emerging bottleneck + incomplete incumbent solution + measurable economic win + clean OSS primitive + unusually strong fit for Rust.**


## User

how does this relate to that mingley/llama-rust


## ChatGPT

It relates very directly, but I would treat `llama-rust` as the execution/reference engine and `expertvm` as a new systems layer underneath it—not turn `llama-rust` itself into a giant production serving stack.

Today, `llama-rust` essentially has this architecture:

```
GGUF file
   ↓
owned Vec<u8>
   ↓
tensor metadata + byte ranges
   ↓
model implementation
   ↓
attention / MoE routing / FFN
   ↓
quantized GEMV/GEMM
   ↓
CPU

```

That description is literal: the repo currently keeps one owned file blob and refers to tensor payloads as ranges inside it. It already implements the actual MoE math for Llama-style MoE, Qwen2MoE, Qwen3MoE, Llama4, and Qwen3Next. GitHub+1

So you have already built a surprising amount of the machinery needed to experiment with `expertvm`.

The key distinction is:

```
llama-rust answers:

"What does this model mean,
and how do I execute its math correctly?"


expertvm answers:

"Where are the weights needed
for that math physically located right now?"

```

Those are separate problems.

### The exact seam

Right now, conceptually, an expert weight looks something like:

```
Tensor {
    dtype: Q4_K,
    shape: ...,
    range: 18372910..19483721,
}

```

and ultimately:

```
tensor.range
    ↓
&model_blob[range]
    ↓
GEMV

```

That means the assumption baked into the system is:

> Every weight is already locally addressable.

`expertvm` breaks that assumption.

Instead:

```
GGUF metadata
      ↓
 TensorHandle
      ↓
┌─────────────────────┐
│     ExpertStore     │
│                     │
│ HBM                 │
│ other GPU HBM       │
│ pinned CPU DRAM     │
│ NVMe eventually     │
└─────────────────────┘
      ↓
 ExpertLease
      ↓
 kernel

```

So the model execution code says:

```
let gate = experts.acquire(layer, expert, Weight::Gate).await?;
let up   = experts.acquire(layer, expert, Weight::Up).await?;
let down = experts.acquire(layer, expert, Weight::Down).await?;

execute_expert(x, gate, up, down);

```

It no longer needs to know whether expert 83 was sitting:

```
GPU 0
GPU 5
CPU RAM
NVMe

```

three milliseconds ago.

That is the relationship.

And your existing MoE work makes `llama-rust` unusually useful as the first integration target. It already knows the semantic differences between Qwen2MoE, Qwen3MoE, Llama MoE, Llama4, etc., rather than having only a toy generic MoE abstraction. GitHub

### In fact, one thing in `llama-rust` is almost the primitive ancestor of this idea

Your current weight representation is already:

> one backing storage object + tensor views into ranges

rather than:

> deserialize every matrix into independent allocations.

The README explicitly says quantized tensors remain in the original file blob and are addressed by range, with no per-matrix clone. GitHub

That's conceptually very close to what we'd want:

```
Current:

Tensor
  ↓
Range
  ↓
FileBlob


Future:

Tensor
  ↓
WeightId
  ↓
ResidencyManager
  ↓
PhysicalBacking

```

The big conceptual change is replacing:

```
&blob[tensor.range]

```

with something resembling:

```
let lease = store.acquire(tensor.id(), target_device).await?;

```

That tiny-looking abstraction change opens an enormous systems space.

---

There is an important fork, though.

Your current `llama-rust` is deliberately a very different project from what `expertvm` eventually needs to be.

It currently has:

```
#![forbid(unsafe_code)]
no FFI
essentially dependency-free
CPU decode
Metal only as external measurement code
no mmap
no batching
no concurrent serving

```

and the repo explicitly parks mmap and production serving as non-goals/current gaps. GitHub+1

That philosophy is excellent for what `llama-rust` currently demonstrates:

> **Can we correctly execute modern quantized LLM architectures from GGUF in pure safe Rust?**

But a serious `expertvm` implementation eventually wants:

```
CUDA
HIP
pinned memory
DMA
CUDA events
streams
peer-to-peer copies
NVLink
RDMA
registered memory
GPUDirect
possibly io_uring / mmap

```

At the bottom of that stack, someone necessarily interacts with unsafe foreign APIs.

Trying to preserve:

```
zero unsafe anywhere
zero dependencies

```

would actively handicap the project.

Instead I'd preserve the more valuable invariant:

> **All public memory/residency semantics are safe Rust. Unsafe hardware interaction is confined below a tiny audited boundary.**

For example:

```
                   safe Rust

              ExpertResidency
                    │
                 Lease
                    │
             MigrationPlan
                    │
            PlacementPolicy
                    │
──────────────────────────────────
             unsafe boundary
──────────────────────────────────
                    │
          CUDA / RDMA / mmap

```

That's actually a stronger demonstration of Rust's value than avoiding `unsafe` entirely.

---

### So I would evolve the repository family like this

Not:

```
llama-rust
   ↓
add everything
   ↓
500,000-line vLLM competitor

```

Instead:

```
                  llama-rust
                /            \
               /              \
     model semantics        reference
     GGUF parsing           correctness
     quant formats             tests
     MoE routing
             │
             │
             ▼
         expertvm
             │
             ├── expert registry
             ├── residency
             ├── leases
             ├── migration
             ├── caching
             ├── prefetch
             └── replication

             │
             ▼
       hardware backends

```

Eventually there might be a broader workspace:

```
llm-systems-rs/

    crates/
        gguf/
        quant/
        kernels/
        model/
        expertvm/
        cuda/
        metal/

    engines/
        llama-rust/

```

But I wouldn't reorganize yet.

### Your next step could actually use `llama-rust` as the research vehicle

You don't need B200s and RDMA to prove whether the basic idea has merit.

Your repo already understands Qwen3MoE.

Start with CPU memory residency.

Right now:

```
entire GGUF
    ↓
Vec<u8>
    ↓
all expert bytes resident

```

Change the expert path experimentally to:

```
dense/common weights
        ↓
resident

expert weights
        ↓
ExpertStore
        ↓
small resident cache

```

Artificially constrain the cache:

```
64 experts total
8 may be "resident"

```

Then execute a real trace and measure:

```
expert access frequency
cache hit rate

reuse distance
expert co-activation
layer-to-layer predictability

LRU hit rate
LFU hit rate
oracle hit rate

prefetch hit rate

```

This is extremely valuable because before building some elaborate CUDA transfer system, you can answer the fundamental research question:

> **Are expert activation patterns predictable/local enough for dynamic residency to work?**

Suppose Qwen3MoE produces:

```
random policy       12% hit
LRU                 37%
frequency-aware     58%
1-layer predictor   81%
oracle              94%

```

Now there is something here.

Conversely, if realistic workloads give:

```
best realistic policy    18%

```

you just learned that this particular direction is economically weak before spending six months writing CUDA infrastructure.

And `llama-rust` gives you an unusually clean environment for doing that because you own the entire execution path.

### There's an even more interesting experiment

Instrument the router you already have.

For every generated token:

```
token
layer
selected experts
routing probabilities

```

Record:

```
T0:
L0:   4, 17
L1:  31, 52
L2:   8, 91
...

T1:
L0:   4, 19
L1:  31, 52
L2:   8, 73
...

```

Then analyze:

```
P(E_t,l | E_t-1,l)

P(E_t,l+1 | E_t,l)

P(E | prompt/domain)

P(E_a | E_b)

expert popularity distribution

working-set size

reuse distance

```

This tells us something deeper than simply implementing inference.

You're studying the **memory-access behavior of sparse neural networks**.

That can directly inform the design of `expertvm`.

And if you find something like:

```
20% of experts account for 75% of activations

```

or:

```
given previous-layer routing,
next-layer expert selection is 85% predictable

```

then that itself is potentially a very interesting public research result.

---

So I would revise the trajectory of `llama-rust` slightly.

The work you've already done isn't wasted relative to the serving-company idea. It's almost the ideal **correctness laboratory**.

You've established:

```
GGUF understanding
quant formats
kernel implementation
real model architectures
KV
MoE routing
expert execution

```

while deliberately not yet going deep into serving infrastructure. GitHub

That means the natural next interesting layer isn't:

```
add an OpenAI HTTP API

```

or:

```
add Tokio

```

or even:

```
match llama.cpp tokens/sec

```

Those would make it a more complete inference engine, but they don't create much novelty.

The potentially high-signal pivot is:

```
              llama-rust

      "we can execute the model"
                 │
                 ▼
       instrument sparse MoE
                 │
                 ▼
     characterize expert locality
                 │
                 ▼
          build expertvm
                 │
                 ▼
     virtualize expert weights
                 │
                 ▼
        GPU implementation
                 │
                 ▼
 prove lower HBM / $ per token

```

That turns `llama-rust` from **“a very clean Rust implementation of LLM inference”** into the foundational testbed for a specific research thesis that the mainstream inference stack has not clearly solved.

That is a much more differentiated trajectory.


## User

so if I did not have access to a real expensive GPU, could I use the code from mingley/llama-rust to simulate this while building the expertvm at all? 


## ChatGPT

Yes. In fact, `llama-rust` is enough to build roughly the first 70–80% of the intellectually important part of `expertvm` without an expensive GPU.

The key is to separate two things:

```
A. Does expert virtualization work as a systems policy?
B. Does our implementation actually win on H100/B200 hardware?

```

You can answer A almost completely on CPU. You need expensive hardware mainly for B.

`llama-rust` is particularly useful because it already implements real MoE routing for Qwen2MoE, Qwen3MoE, Llama4, Qwen3Next, etc. GitHub

The first thing I would build is this seam:

```
// llama-rust currently conceptually does:
expert_weight.bytes()

// change to:
expert_store.acquire(expert_id, layer_id, weight_id)

```

Then define:

```
trait ExpertStore {
    fn acquire(
        &mut self,
        expert: ExpertId,
    ) -> Result<ExpertLease>;
}

```

Now the actual model math doesn't care where the expert lives.

You can implement multiple backends:

```
ExpertStore

├── DirectStore
│     current llama-rust behavior
│
├── CachedStore
│     bounded "fast memory" cache
│
├── TieredStore
│     fast RAM / slow RAM / disk
│
└── SimulatedGpuStore
      fake HBM capacity
      fake PCIe bandwidth
      fake NVLink bandwidth
      fake DMA concurrency

```

That last one is where this gets powerful.

Suppose you tell the simulator:

```
HBM capacity:          80 GiB

CPU → HBM:
    bandwidth:         32 GB/s
    latency:           10 us

GPU1 → GPU0 NVLink:
    bandwidth:         450 GB/s
    latency:           2 us

concurrent DMA:        4

```

Those buffers don't actually have to be HBM.

The simulator maintains the exact state transitions:

```
Expert 57

CPU_RESIDENT
      │
      │ acquire()
      ▼
TRANSFERRING_TO_GPU
      │
      │ simulated transfer completion
      ▼
GPU_RESIDENT
      │
      │ lease count = 1
      ▼
IN_USE

```

And you can test all of the difficult logic:

```
allocation
eviction
prefetch
replication
leases
concurrent requests
transfer scheduling
cache replacement
capacity constraints
expert pinning
hot/cold promotion
co-activation
admission
cancellation

```

without touching CUDA.

The execution timeline can even be simulated. If expert 32 is 180 MB and your modeled PCIe bandwidth is 32 GB/s:

```
transfer ≈ 180 MB / 32 GB/s
         ≈ 5.6 ms

```

You don't `sleep(5.6ms)` necessarily. A discrete-event simulator advances a logical clock:

```
t=0.000   layer 15 starts

t=0.400   router discovers experts
          [12, 32, 91]

t=0.400   12: HBM hit
          91: HBM hit
          32: miss

t=0.400   start CPU→HBM(32)

t=3.100   layer computation finishes

t=6.000   expert 32 arrives

          STALL = 2.900ms

```

Now you have an actual research environment.

### More importantly, `llama-rust` gives you real access traces

Don't generate synthetic expert choices.

Run an actual Qwen3MoE through `llama-rust`.

Its router produces something like:

```
token=137 layer=0  experts=[16, 31, 82]
token=137 layer=1  experts=[4, 19, 22]
token=137 layer=2  experts=[16, 73, 91]

token=138 layer=0  experts=[16, 31, 47]
...

```

Record that as:

```
struct ExpertAccess {
    sequence: SequenceId,
    token: usize,
    layer: usize,
    experts: Vec<ExpertId>,
}

```

Now replay the same trace through different `expertvm` policies.

For example:

```
Policy                   HBM hit    bytes moved    stalls
──────────────────────────────────────────────────────────
Random                     22%         8.7 TB       42.1s
LRU                        51%         5.3 TB       25.4s
LFU                        63%         3.9 TB       18.7s
Layer-aware LRU            68%         3.1 TB       14.2s
Coactivation-aware         74%         2.4 TB        9.8s
Predictive prefetch        89%         2.7 TB        2.1s
Oracle                     96%         2.0 TB        0.4s

```

Those are made-up numbers, but that's exactly the experiment you want.

And now you can ask much more important questions than “is my CUDA kernel fast?”

You can determine:

```
What is the minimum working set?

How skewed is expert popularity?

Does locality persist between tokens?

Does locality persist between sequences?

Does prompt domain affect expert selection?

Does conversation history?

Are expert activations correlated?

How far ahead can selection be predicted?

How much HBM is necessary for 95% hit rate?

At what PCIe bandwidth does virtualization become viable?

How much does NVLink change the optimal policy?

When is moving tokens cheaper than moving weights?

```

That's real inference research.

### You can also make the CPU implementation real, not just simulated

There is one architectural limitation in `llama-rust` today that you would need to change.

It currently reads the quantized model into a single owned file blob and tensors reference ranges within that blob. There is no mmap. GitHub

So:

```
today

GGUF
 ↓ read
Vec<u8> containing entire model
 ↓
TensorRange

```

doesn't demonstrate actual constrained residency because the whole model is still occupying RAM.

I'd change the storage abstraction first:

```
                  WeightStorage
                       │
          ┌────────────┼─────────────┐
          │            │             │
      InMemory       Mmap        Synthetic
          │            │             │
          └────────────┼─────────────┘
                       │
                  TensorView

```

Then an expert can genuinely be:

```
disk
 ↓
page/read
 ↓
bounded RAM expert cache
 ↓
GEMV

```

You could run a model substantially larger than the amount of RAM you're willing to dedicate to expert residency.

And this isn't hypothetical. A very recent llama.cpp experiment ran a 48.4 GB Qwen3-Next-80B GGUF on a 16 GB M1 Mac CPU-only by maintaining a bounded residency set for expert pages. GitHub Another recent llama.cpp experiment reports 31–52% decode improvements from explicitly prefetching routed expert weight ranges rather than relying on scattered page faults. GitHub

That is strong evidence that your laptop can be a legitimate laboratory for the problem.

### There is also an important update to my previous answer

After searching specifically for this, I would no longer claim that **basic expert virtualization itself is an untouched void**.

The field is moving extremely quickly.

In just the last few weeks there are projects doing:

* bounded CPU/disk residency for oversized MoEs, GitHub
* explicit expert prefetch based on router outputs, GitHub
* VRAM expert caches + CPU-pinned experts + PCIe access in `moe-l2`. GitHub

That actually makes the `llama-rust` research route more important. Don't spend six months independently building “LRU experts between RAM and GPU.” That is rapidly becoming commodity.

Use `llama-rust` to figure out what the **next abstraction** should be.

For example, a much more interesting result would be:

> Expert residency shouldn't be reactive caching at all. Router behavior is sufficiently predictable that a predictive policy reduces cold-expert stalls by 85%.

Or:

> The optimal decision is dynamically choosing between moving expert weights and dispatching activations based on fan-in, expert size, topology, and predicted reuse.

That starts looking like:

```
                   router
                      │
                      ▼
                Expert Planner
                      │
       ┌──────────────┼──────────────┐
       │              │              │
 move weights    move tokens      replicate
       │              │              │
 CPU → GPU       GPU → GPU       GPU → GPU

```

That's substantially more novel than an expert LRU.

So yes: **I would absolutely start this without buying a GPU.**

I'd make `llama-rust` produce real MoE access traces, build `expertvm` as a policy-independent residency/lease abstraction, add a deterministic topology simulator, and use actual model routing to discover what policies are worth pursuing.

Only once the simulator/replay results show a compelling hypothesis would I rent H100/H200/B200 time to validate it physically. You don't need to own that hardware to discover the algorithm.


## User

i want to build a synthetic high demand GPU environment with mechanical invarants perfectly replicating how GPU ops would work so that we can make each of these things as fast as possible with agents that don't have GPUs to actually use.


## ChatGPT

Yes. I think this is a substantially better idea than merely making a fake “GPU backend.”

The right target is:

> A deterministic Rust GPU-systems emulator for inference infrastructure, where the semantics are exact and the performance model is parameterized/calibratable.

Then `llama-rust` becomes your source of real model behavior, while `expertvm`, schedulers, caches, prefetchers, allocators, etc. execute against the simulated GPU environment.

The critical distinction is that you should not attempt to perfectly simulate an H100 transistor-by-transistor or instruction-by-instruction. Projects such as Accel-Sim/GPGPU-Sim already pursue detailed GPU architectural simulation, including SM cores, caches and memory systems, and those simulations can become extremely expensive. GitHub+1

You want a different abstraction level.

### 1\. Make semantics exact; make timing a model

There are two categories of GPU behavior.

These can be reproduced essentially exactly:

```
memory capacity
allocation/free
allocation lifetimes

stream ordering
event dependencies
barriers
synchronization

kernel enqueue semantics
async copies

copy engine availability
peer accessibility

HBM residency
host-pinned residency

P2P topology
NVLink topology
PCIe topology

concurrent transfer limits

resource ownership

OOM behavior

data dependencies

collective dependencies

CUDA graph dependencies

```

CUDA's stream/event model gives you concrete rules here. Operations within a stream are ordered; cross-stream dependencies are established explicitly with mechanisms such as events, and async GPU work can continue after the host-side call returns. NVIDIA Docs+1

Those are mechanical invariants.

Agents can safely optimize against them.

But this:

```
this grouped FP8 GEMM takes exactly 42.7 μs

```

cannot be known perfectly without the actual GPU.

That belongs in the parameterized performance model.

So architect the system as:

```
             EXACT
        mechanical model
               │
               ▼
     dependency/resource DAG
               │
               ▼
       PARAMETERIZED
        timing model
               │
               ▼
        virtual clock

```

That separation is extremely important.

---

The simulator I would build would have roughly six layers:

```
llama-rust / expertvm
        │
        ▼
GPU operation API
        │
        ▼
CUDA-like semantics
        │
        ▼
Resource simulator
        │
        ▼
Topology simulator
        │
        ▼
Timing model
        │
        ▼
Discrete-event engine

```

And all of it can be Rust.

## The fundamental primitive should be an operation

Something like:

```
enum GpuOp {
    Kernel(KernelOp),
    Memcpy(MemcpyOp),
    Collective(CollectiveOp),
    EventRecord(EventId),
    EventWait(EventId),
    Alloc(AllocOp),
    Free(AllocationId),
}

```

Every operation has dependencies:

```
struct Operation {
    id: OpId,
    stream: StreamId,

    reads: Vec<AllocationId>,
    writes: Vec<AllocationId>,

    kind: GpuOp,
}

```

The simulator turns that into a DAG.

For example:

```
Stream 0:

 GEMM A
   │
   ▼
 Event X


Stream 1:

 Wait X
   │
   ▼
 H2D expert 83
   │
   ▼
 Expert GEMM

```

The actual simulator therefore understands causality rather than just doing `sleep()`.

That matters because CUDA itself is fundamentally asynchronous. NVIDIA explicitly documents cases where operations in independent streams have undefined relative ordering, while event dependencies establish ordering between them. NVIDIA Docs+1

---

# Model physical resources explicitly

For an H100 profile you might have:

```
Gpu {
    hbm: MemoryPool {
        capacity: GiB(80),
        bandwidth: ...,
    },

    compute: ComputeResources {
        sms: 120,
        tensor_cores: ...,
    },

    copy_engines: [...],

    pcie: Link { ... },

    nvlinks: [...],
}

```

And for eight GPUs:

```
CPU NUMA 0
    │
  PCIe
    │
  switch
 ┌──┼────┬────┐
GPU0 GPU1 GPU2 GPU3
 │  ╲ ...
 │ NVLink
 │
GPU4 ...

```

Now contention becomes real.

If:

```
Expert A → GPU0
Expert B → GPU0
KV block → GPU0

```

all require the same PCIe path simultaneously, you cannot give all three:

```
32 GB/s

```

The link itself is a constrained resource.

You simulate:

```
PCIe x16 capacity = X

transfer A ──┐
transfer B ──┼── shared link
transfer C ──┘

```

Likewise NVLink.

Likewise HBM bandwidth.

Likewise copy engines.

This is where an `expertvm` agent starts having something meaningful to optimize.

---

# Then represent kernels structurally

Do not simulate CUDA instructions.

Have something like:

```
KernelOp::Matmul {
    m,
    n,
    k,
    lhs_dtype,
    rhs_dtype,
    output_dtype,
}

```

or:

```
KernelOp::GroupedMoeGemm {
    experts,
    tokens_per_expert,
    hidden_dim,
    intermediate_dim,
    dtype,
}

```

Then the hardware profile estimates:

```
compute requirement
memory bytes
HBM bandwidth requirement
launch overhead
occupancy

```

A basic first model is essentially roofline:

```
compute_time =
    FLOPs / effective_compute_rate

memory_time =
    bytes / effective_memory_bandwidth

kernel_time ≈
    launch_overhead +
    max(compute_time, memory_time)

```

Later you make it much better.

For example:

```
small GEMM efficiency curve
large GEMM efficiency curve
FP8 Tensor Core efficiency
grouped GEMM penalty
shape-dependent efficiency
attention kernel curves
expert count overhead

```

These become entries in the GPU profile rather than simulator logic.

---

# This gives you something very powerful: virtual GPUs

You could have:

```
profiles/
    a100-80gb.toml
    h100-sxm.toml
    h200-sxm.toml
    b200.toml

    8xh100-nvlink.toml
    8xb200-nvlink.toml

    hypothetical/
        cheap-48gb.toml

```

Then:

```
cargo run --release -- \
    --hardware profiles/8xh100-nvlink.toml \
    --workload traces/qwen3-moe-coding.trace

```

And receive:

```
Simulated workload

GPU utilization        78.3%
HBM utilization        94.1%

TTFT p50               41.8 ms
TTFT p99               92.4 ms

ITL p50                 7.2 ms

PCIe utilization       63.8%
NVLink utilization     28.1%

Expert cache hit       81.7%

GPU stalls:
  expert transfer      13.1%
  compute               2.3%
  synchronization       1.8%

```

Now an agent can modify:

```
expertvm/src/prefetch.rs

```

run:

```
cargo bench

```

and get:

```
before: 7.2ms ITL
after:  5.9ms ITL

expert transfer stalls:
13.1% → 4.8%

```

without having an H100.

That's exactly the environment you're describing.

---

# `llama-rust` makes this much better

You have two things to simulate:

```
1. model semantics
2. GPU execution

```

Do not simulate the first.

Actually execute it on CPU.

That's the clever part.

You can make:

```
                  llama-rust

             actual CPU execution
                /            \
               /              \
        real tensors       GPU trace
        real router           │
        real tokens           │
               \              /
                \            /
                    ▼
              virtual GPU

```

Suppose Qwen's router really selects:

```
layer 31:
experts = [7, 19, 82, 113]

```

`llama-rust` computes those selections for real.

`expertvm` then asks:

```
Where are 7,19,82,113?

```

The virtual GPU says:

```
7    GPU0
19   GPU0
82   CPU DRAM
113  GPU3

```

And the system has to decide:

```
dispatch token to GPU3?

copy expert 113?

reuse remote expert?

prefetch 82?

evict something?

```

All the decision making is real.

Only the physical time taken by the hardware is simulated.

That is a much stronger environment than synthetic/random MoE traces.

---

# Make time completely virtual

This is critical for agents.

Do not make:

```
tokio::time::sleep(Duration::from_micros(40)).await;

```

Instead implement a discrete-event simulator:

```
virtual clock = 120.000 μs

event queue:

120.400 kernel A done
121.300 memcpy B done
124.100 collective C done

```

Pop:

```
120.400

```

advance the clock instantly:

```
120.400

```

process consequences.

Then proceed.

A simulated 30-minute production workload could potentially execute in seconds or minutes depending on how much model computation you retain.

It is also deterministic:

```
commit A
hardware H
workload W
seed S

always produces

performance P

```

Agents love that.

---

# The other major thing: invariants

I would go extremely hard on this.

For example:

```
invariant!(
    allocation.lease_count > 0
        => !allocation.can_be_freed()
);

```

```
invariant!(
    kernel.reads.iter().all(
        |a| memory.is_resident(a, kernel.device)
    )
);

```

```
invariant!(
    transfer.completed()
        => destination.contains(transfer.object)
);

```

```
invariant!(
    stream[i + 1].start >= stream[i].finish
);

```

```
invariant!(
    hbm.used_bytes <= hbm.capacity
);

```

And:

```
an expert cannot simultaneously be:

Evicted
and
actively used by a kernel

```

You can encode many of these directly into the Rust types.

For example:

```
Expert<Cold>

Expert<Transferring>

Expert<Resident>

Expert<Leased>

```

with only legal transitions:

```
Cold
 ↓
Transferring
 ↓
Resident
 ↓
Leased
 ↓
Resident
 ↓
Evicting
 ↓
Cold

```

That gives you an unusually good agent-development environment:

> Agents are allowed to aggressively optimize policies, but the simulator makes illegal GPU states impossible or immediately detectable.

This may actually be one of the strongest arguments for Rust here.

---

# Build an adversarial workload generator too

This matters if agents are doing the optimization.

Otherwise they will Goodhart the benchmark.

Don't just give them:

```
one GLM trace
one H100 topology

```

Have:

```
workloads/

uniform_experts
hotset_experts
rapidly_shifting_hotset
adversarial_cache_thrash
coding_trace
chat_trace
long_context_trace
batch_1
batch_128
prefill_heavy
decode_heavy

topologies/

1_gpu
2_gpu_pcie
8_gpu_nvlink
8_gpu_bad_numa
2_node_rdma
asymmetric_links

faults/

transfer_delay
GPU unavailable
memory pressure
request cancellation
expert loading failure

```

Then optimization is evaluated across all of them.

An agent can't win by hard-coding:

```
keep expert 31 resident

```

because hidden scenarios punish it.

---

# You should have two separate test scores

This becomes important.

### Semantic score

Binary.

```
Correctness: PASS

```

Did the software obey:

```
CUDA ordering
memory ownership
capacity
event dependencies
residency
synchronization

```

If not, optimization rejected.

### Performance score

Continuous:

```
$/M tokens
ITL
TTFT
throughput
HBM needed
bytes transferred
energy estimate

```

Agents minimize that.

So your agent loop becomes:

```
       Agent
         │
         ▼
   modify expertvm
         │
         ▼
   cargo test
         │
   semantics pass?
      /       \
    no         yes
    │           │
 reject         ▼
             simulator
                │
                ▼
             score
                │
                ▼
        keep / revert

```

You could run thousands of optimization experiments.

---

# There is one serious danger

You cannot allow agents to optimize the simulator instead of the real GPU.

This is the classic simulation-to-reality problem.

Suppose the model says:

```
PCIe transfer = bytes / 32 GB/s

```

Agent discovers that breaking one transfer into 8,000 tiny copies somehow yields the same bandwidth.

Real hardware:

```
8,000 tiny transfers = terrible

```

The agent found a hole in your simulator.

So your performance model needs:

```
fixed overhead
size-dependent throughput
queueing
concurrency limits
alignment
startup latency
contention

```

And eventually calibration.

The same applies to kernels.

---

# Therefore I'd make calibration part of the architecture from day one

You won't need GPUs to develop against it.

Someone with GPUs can later run:

```
gpu-profile capture --output h100.json

```

It benchmarks:

```
memcpy size sweep
P2P size sweep

GEMM shapes
MoE GEMM shapes

FP16/BF16/FP8/FP4

attention shapes

concurrent copy+compute

multiple stream contention

```

And produces:

```
HardwareProfile

```

That profile goes into the simulator.

Now the open-source community could contribute:

```
profiles/
  h100-sxm-nvidia-driver-X.json
  h200-sxm.json
  b200.json
  mi355x.json

```

You personally don't need to own any of them.

This is probably the most important architectural choice:

> **The simulator should consume empirical hardware profiles, not contain hard-coded claims about hardware performance.**

---

There is precedent for detailed GPU simulation: Accel-Sim/GPGPU-Sim models SMs, caches and memory systems and is actively maintained. GitHub+1 But I think copying that scope would be a mistake for your purpose.

Your abstraction should stop around here:

```
                 inference system
                        │
               GPU systems semantics
                        │
              resource contention
                        │
              calibrated op costs
                        │
                       STOP
─────────────────────────────────────────
warp scheduler
register bank conflicts
instruction pipeline
L1 cache implementation
Tensor Core circuitry
─────────────────────────────────────────
             don't model initially

```

Those deeper effects appear indirectly through empirical kernel cost curves.

That lets the simulator be perhaps **six orders of magnitude cheaper to execute** than cycle-level simulation while preserving the properties your inference-system agents actually manipulate.

And this gives the whole project a much larger purpose than `expertvm`.

You'd effectively be building:

> **a GPU CI environment for systems software that doesn't require GPUs.**

`expertvm` would just be its first killer application.

The stack I'd now pursue is:

```
           llama-rust
               │
       real model behavior
               │
               ▼
         gpu-sim-rs
  deterministic GPU systems VM
               │
      ┌────────┼─────────┐
      ▼        ▼         ▼
 expertvm  scheduler  kv-fabric
      │        │         │
      └────────┼─────────┘
               ▼
        agents optimize
               │
               ▼
    eventually validate on
       physical hardware

```

That is more interesting to me than building `expertvm` alone. It creates the infrastructure that lets you use fleets of coding/research agents to work on high-end inference optimization despite none of those agents having access to a physical GPU. NVIDIA Docs+1
