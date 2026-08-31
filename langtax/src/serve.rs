//! Local HTTP/1.1 serve. Not a production inference server.
//!
//! Default `gguf_gemv serve` accepts one connection at a time with a persistent
//! KV cache. `--engine` admits concurrent `POST /generate` onto one [`Engine`]
//! so prefills and decodes GEMM together. `POST /v1/completions` and
//! `POST /v1/chat/completions` map onto the same greedy path (`max_tokens`
//! aliases `n_predict`; `n` must be 1). HTTP/1.1 keep-alive is on unless the client sends
//! `Connection: close`. `/v1/*` responses use the OpenAI `choices` envelope.
//! `GET /v1/models` and `GET /v1/models/{id}` are the OpenAI list/retrieve
//! objects (`--model-id`, default GGUF stem). `GET /health` is `{"status":"ok"}`.
//! `GET /metrics` is `{"engine":false}` here, or Engine counters with `--engine`.
//! `POST /tokenize` is `{"tokens","count"}` of the same prompt path as generate
//! (`add_special_tokens`, default true, is BOS via `prompt_ids`; false is
//! `Tokenizer::encode` without extra BOS). `POST /detokenize` is `{"text"}`
//! from a token-id array. `POST /v1/completions` `echo` (default false) returns
//! prompt plus completion in `choices[].text`. Chat ignores `echo`. No crates.io
//! HTTP stack.

use std::fs::File;
use std::io::{ErrorKind, Read, Write};
use std::net::{Ipv4Addr, TcpListener};
use std::path::Path;
use std::time::Duration;

use crate::cli::InferArgs;
use crate::decode::{greedy_generate_cache, prompt_ids, KvCache, Llama, LlamaError};
use crate::gguf::load_gguf_owned;
use crate::serve_engine;
use crate::store_attach::{gpu_knobs, PlannerCli};
use crate::template::ChatMessage;
use crate::tok::Tokenizer;
use expertvm::{GpuFill, GpuStoreCfg, Prefetch};

/// Usage for the `serve` verb.
pub const SERVE_USAGE: &str = "\
usage: gguf_gemv serve <path> [--n-predict N] [--n-ctx N] [--kv-page N] [--bind HOST:PORT] [--model-id ID] [--engine] [--max-seqs N] [--expert-slots N] [--expert-sim] [--expert-8gpu] [--expert-bytes N] [--prefill-chunk N] [--decode-first] [--slo-reject] [--ttft-slo-ns N] [--itl-slo-ns N] [--cuda-graphs] [--graph-update] [--graph-set-params] [--graph-clone] [--graph-build] [--graph-build-deps] [--graph-piecewise] [--graph-capture-deps] [--graph-enable] [--graph-mem] [--graph-auto-free] [--graph-mem-trim] [--timing-events] [--event-blocking-sync] [--mapped] [--managed] [--vmm] [--vmm-page N] [--host-func] [--blocking-streams] [--sync-alloc] [--mempool] [--mempool-trim] [--mempool-no-reuse] [--mempool-max N] [--shareable] [--pageable] [--host-register] [--host-register-mapped] [--sync-memops] [--device-sync-memops] [--memcpy-batch] [--memcpy-during] [--memcpy-any] [--accessed-by] [--legacy-null] [--stream-priority] [--seq-streams] [--kv-sim] [--kv-bytes N] [--decode-priority] [--cooperative] [--pdl] [--l2-persist] [--l2-reset] [--l2-fetch N] [--l2-ratio N] [--l2-streaming] [--cluster N] [--preferred-cluster N] [--cluster-spread] [--func-cluster-spread] [--cluster-load-balance] [--cluster-must-set] [--required-cluster N] [--max-shared] [--func-max-shared] [--max-l1] [--non-portable-cluster] [--sync-policy auto|spin|yield|blocking] [--device-sync-policy auto|spin|yield|blocking] [--mem-sync-domain default|remote] [--mem-sync-map identity|collapse] [--mem-sync-launch] [--mem-sync-launch-map] [--shared-mem] default|four|eight] [--func-shared-mem default|four|eight] [--device-shared-mem default|four|eight] [--portable-cluster default|portable|non-portable] [--optin-shared] [--dynamic-shared N] [--portable-shared default|portable|non-portable] [--nvlink-util] [--device-launch] [--device-updatable] [--kernel-priority N] [--launch-completion] [--programmatic-event] [--stream-attach] [--managed-host] [--prefetch-host] [--wait-value] [--multicast] [--compute-slots N] [--decode-sms N] [--prefetch none|copy-forward|markov|both] [--plan-window N] [--plan-threshold N] [--trace-out FILE]
  -n, --n-predict N   tokens to generate (default: 2)
      --n-ctx N       KV capacity (default: grow per request; `--engine` default 64)
      --kv-page N     paged KV block size (default: dense; `--engine` default 16)
      --bind ADDR     loopback listen address (default: 127.0.0.1:8080)
      --model-id ID   OpenAI `model` / GET /v1/models id (default: GGUF path stem)
      --engine        concurrent requests on one Engine (continuous-batch GEMM)
      --max-seqs N    in-flight Engine sequences (`--engine`; default: 4)
      --expert-slots N  ExpertStore on `--engine`: omit = blob FFN, `0` = DirectStore,
                        N>0 = CachedStore (`--expert-sim` uses N, default 8)
      --expert-sim      SimulatedGpuStore (`--engine`; example H100, 4096-byte experts)
      --expert-8gpu     8×H100 NVLink profile (`--expert-sim`; enables plan_placement)
      --expert-bytes N  simulated expert page bytes (`--expert-sim`; default: 4096)
      --prefill-chunk N prefill tokens per Engine step (`--engine`; `0` = the rest)
      --decode-first    hold leftover prefill while any live sequence is decoding (`--engine`)
      --slo-reject      drop waiters whose gpu-sim queue wait meets `--ttft-slo-ns` (`--engine`)
      --ttft-slo-ns N   virtual-ns TTFT budget (`--slo-reject`; needs `--expert-sim`)
      --itl-slo-ns N    count later-token gaps over this budget (`--engine`; `--expert-sim`)
      --cuda-graphs     document default GEMM graph capture (`--expert-sim`; always on)
      --graph-update    cudaGraphExecUpdate parked leaves (`--expert-sim`)
      --graph-set-params  cudaGraphExecKernelNodeSetParams parked leaves (`--expert-sim`; not with `--graph-update`)
      --graph-clone     cudaGraphClone before instantiate (`--expert-sim`)
      --graph-build     cudaGraphCreate / cudaGraphAdd* instead of capture (`--expert-sim`; independent children may Hyper-Q overlap; not with `--graph-piecewise`)
      --graph-build-deps  cudaGraphAddDependencies on `--graph-build` combo parents (`--expert-sim`; needs `--graph-build`; sibling GEMMs serialize; legal with `--pdl` and `--cooperative`)
      --graph-piecewise cudaStreamBeginCaptureToGraph combo parents (`--expert-sim`; independent child roots may Hyper-Q overlap; not with `--graph-build`)
      --graph-capture-deps  cudaStreamBeginCaptureToGraph deps on `--graph-piecewise` combo parents (`--expert-sim`; needs `--graph-piecewise`; sibling GEMMs serialize; legal with `--pdl` and `--cooperative`)
      --graph-enable    cudaGraphNodeSetEnabled skip extra combo children (`--expert-sim`; walker; not with `--device-launch`)
      --graph-mem       in-graph scratch cudaMallocAsync (`--expert-sim`; skips `--graph-update`)
      --graph-auto-free AutoFreeOnLaunch scratch without in-graph free (`--expert-sim`; not with `--graph-mem`)
      --graph-mem-trim  cudaDeviceGraphMemTrim unused reserved after score (`--expert-sim`)
      --timing-events   cudaEventElapsedTime on copy start/end (`--expert-sim`)
      --event-blocking-sync  cudaEventBlockingSync copy events (`--expert-sim`; implies `--timing-events`; distinct from `--sync-policy blocking`)
      --mapped          cudaHostAllocMapped miss pages (`--expert-sim`)
      --managed         cudaMallocManaged miss pages (`--expert-sim`)
      --vmm             va_acquire miss pages (`--expert-sim`)
      --vmm-page N      va_acquire_paged span (`--expert-sim`; `N>0` implies `--vmm`)
      --host-func       cudaLaunchHostFunc after acquire GEMM (`--expert-sim`)
      --blocking-streams  blocking compute stream (`--expert-sim`)
      --sync-alloc      host-sync malloc/memcpy/free (`--expert-sim`)
      --mempool         hold unused cudaMallocAsync bytes (`--expert-sim`)
      --mempool-trim    cudaMemPoolTrimTo unused cached after score (`--expert-sim`; implies `--mempool`)
      --mempool-no-reuse  cudaMemPoolReuseAllowOpportunistic=0 (`--expert-sim`; implies `--mempool`)
      --mempool-max N   cudaMemPoolProps::maxSize (`--expert-sim`; implies `--mempool`; `N>0`; needs cudaMallocAsync)
      --shareable       POSIX-FD mempool IPC (`--expert-sim`; implies `--mempool`; needs cudaMallocAsync)
      --pageable        pageable H2D (`--expert-sim`)
      --host-register   cudaHostRegister pageable staging then pinned DMA (`--expert-sim`; implies `--pageable`; not with `--mapped`/`--managed`)
      --host-register-mapped  cudaHostRegisterMapped expert pages (`--expert-sim`; implies `--mapped`; not with `--host-register`)
      --sync-memops     cuPointerSetAttribute SyncMemops on miss pages (`--expert-sim`; host-sync H2D; not with `--mapped` / `--memcpy-batch`)
      --device-sync-memops  cudaSetDeviceFlags SyncMemops (`--expert-sim`; host-sync memcpy on that GPU; not with `--mapped` / `--memcpy-batch`)
      --memcpy-batch    cudaMemcpyBatchAsync for multi-expert pinned/VMM prefetch (`--expert-sim`)
      --memcpy-during   cudaMemcpySrcAccessOrderDuringApiCall on `--memcpy-batch` (`--expert-sim`; needs `--memcpy-batch`; API waits those copies)
      --memcpy-any      cudaMemcpySrcAccessOrderAny on `--memcpy-batch` (`--expert-sim`; needs `--memcpy-batch`; empty deps; no API wait; not with `--memcpy-during`)
      --accessed-by     SetAccessedBy / VMM SetAccess / mempool SetAccess (`--expert-sim`; no dest HBM)
      --legacy-null     NULL copy serializes with compute (`--expert-sim`)
      --stream-priority cudaStreamCreateWithPriority on compute (`--expert-sim`)
      --seq-streams     per-sequence copy streams (`--expert-sim`; grouped GEMM stays fused)
      --kv-sim          interned KV on the SimulatedGpuStore clock (`--expert-sim`; default off)
      --kv-bytes N      KV page bytes for `--kv-sim` (default: f32 K+V of one intern block)
      --decode-priority decode GEMMs on a higher-priority compute stream (`--expert-sim`; implies `--stream-priority`; ITL samples that stream)
      --cooperative     cudaLaunchCooperativeKernel for grouped GEMMs (`--expert-sim`; exclusive compute; no Hyper-Q overlap)
      --pdl             programmatic dependent launch for consecutive same-stream GEMMs (`--expert-sim`; overlap needs `--compute-slots` >= 2; illegal with `--cooperative`)
      --l2-persist      cudaLaunchAttributeAccessPolicyWindow over expert pages (`--expert-sim`; persisting L2 after first fill)
      --l2-reset        cudaCtxResetPersistingL2Cache after each GEMM (`--expert-sim`; implies `--l2-persist`; live; cannot capture; reused expert is cold)
      --l2-fetch N      cudaLimitMaxL2FetchGranularity (`--expert-sim`; implies `--l2-persist`; `32`/`64`/`128`; windows must align; legal with `--pdl` and `--cooperative`)
      --l2-ratio N      cudaAccessPolicyWindow.hitRatio as ‰ (`--expert-sim`; implies `--l2-persist`; `1..=1000`; unset is 1000; partial bills more HBM than full persist; legal with `--pdl` and `--cooperative`)
      --l2-streaming    cudaAccessPropertyStreaming on persist GEMM windows (`--expert-sim`; needs `--l2-persist`; reused expert bills full HBM; legal with `--pdl` and `--cooperative`)
      --cluster N       Hopper thread-block cluster X (`--expert-sim`; occupies `min(N, compute_slots)` Hyper-Q slots; legal with `--pdl` and `--cooperative`)
      --preferred-cluster N  Hopper preferred cluster X (`--expert-sim`; needs `--cluster`; occupies preferred size when it fits in `compute_slots`; legal with `--pdl` and `--cooperative`)
      --cluster-spread  Hopper cluster scheduling Spread (`--expert-sim`; occupies every Hyper-Q slot; no-op without `--cluster` >= 2; legal with `--pdl` and `--cooperative`)
      --func-cluster-spread  function Spread cluster scheduling (`--expert-sim`; `cudaFuncSetAttribute` ClusterSchedulingPolicyPreference; launch Default inherits; no-op without `--cluster` >= 2; legal with `--pdl` and `--cooperative`)
      --cluster-load-balance  launch LoadBalancing cluster scheduling (`--expert-sim`; `cudaLaunchAttributeClusterSchedulingPolicyPreference`; needs `--func-cluster-spread`; restores Hyper-Q overlap; not with `--cluster-spread`; legal with `--pdl` and `--cooperative`)
      --cluster-must-set  function ClusterDimMustBeSet (`--expert-sim`; `cudaFuncSetAttribute`; needs `--cluster`; occupancy matches `--cluster`; legal with `--pdl` and `--cooperative`)
      --required-cluster N  function RequiredClusterWidth (`--expert-sim`; `cudaFuncSetAttribute`; needs `--cluster`; must match `--cluster`; occupancy matches `--cluster`; legal with `--pdl` and `--cooperative`)
      --max-shared      MaxShared L1/shared carveout (`--expert-sim`; occupies every Hyper-Q slot; legal with `--pdl` and `--cooperative`)
      --func-max-shared function MaxShared carveout (`--expert-sim`; `cudaFuncSetAttribute` PreferredSharedMemoryCarveout; launch Default inherits; occupies every Hyper-Q slot; legal with `--pdl` and `--cooperative`)
      --max-l1          launch MaxL1 carveout (`--expert-sim`; `cudaLaunchAttributePreferredSharedMemoryCarveout`; needs `--func-max-shared`; restores Hyper-Q overlap; not with `--max-shared`; legal with `--pdl` and `--cooperative`)
      --non-portable-cluster  allow cluster larger than portable size (`--expert-sim`; `cudaFuncAttributeNonPortableClusterSizeAllowed`; legal with `--pdl` and `--cooperative`)
      --sync-policy MODE  stream host-wait policy auto|spin|yield|blocking (`--expert-sim`; `cudaLaunchAttributeSynchronizationPolicy`; legal with `--pdl` and `--cooperative`)
      --device-sync-policy MODE  device host-wait schedule auto|spin|yield|blocking (`--expert-sim`; `cudaSetDeviceFlags` SCHEDULE_*; Auto streams inherit; explicit `--sync-policy` wins; legal with `--pdl` and `--cooperative`)
      --mem-sync-domain MODE  decode-stream mem-sync domain default|remote (`--expert-sim`; `cudaLaunchAttributeMemSyncDomain`; Remote isolates leftover prefill fence tax; engine implies `--decode-priority`)
      --mem-sync-map MODE  decode-stream mem-sync map identity|collapse (`--expert-sim`; `cudaLaunchAttributeMemSyncDomainMap`; collapse needs `--mem-sync-domain remote`; restores leftover prefill fence tax; legal with `--pdl` and `--cooperative`)
      --mem-sync-launch  launch-attribute Remote on grouped GEMMs (`--expert-sim`; `cudaLaunchAttributeMemSyncDomain`; needs `--mem-sync-domain remote`; restores leftover prefill fence tax; legal with `--pdl` and `--cooperative`)
      --mem-sync-launch-map  launch-attribute collapse map on grouped GEMMs (`--expert-sim`; `cudaLaunchAttributeMemSyncDomainMap`; needs `--mem-sync-domain remote`; restores leftover prefill fence tax; legal with `--pdl` and `--cooperative`)
      --shared-mem MODE   kernel-node bank width default|four|eight (`--expert-sim`; `cudaLaunchAttributeSharedMemoryMode`; Default never scales; legal with `--pdl` and `--cooperative`)
      --func-shared-mem MODE  function bank width default|four|eight (`--expert-sim`; `cudaFuncSetSharedMemConfig`; launch Default inherits; distinct from `--shared-mem`; legal with `--pdl` and `--cooperative`)
      --device-shared-mem MODE  device bank width default|four|eight (`--expert-sim`; `cudaDeviceSetSharedMemConfig`; launch Default inherits when function is Default; distinct from `--func-shared-mem` / `--shared-mem`; legal with `--pdl` and `--cooperative`)
      --portable-cluster MODE  launch-time portable cluster default|portable|non-portable (`--expert-sim`; `cudaLaunchAttributePortableClusterSizeMode`; Default uses the function attr; legal with `--pdl` and `--cooperative`)
      --optin-shared    cudaFuncAttributeMaxDynamicSharedMemorySize to SKU opt-in (`--expert-sim`; legal with `--pdl` and `--cooperative`)
      --dynamic-shared N  cudaLaunchKernel sharedMemBytes (`--expert-sim`; must be > 0; legal with `--pdl` and `--cooperative`)
      --portable-shared MODE  CUDA 13 portable-shared default|portable|non-portable (`--expert-sim`; `cudaLaunchAttributeSharedMemoryMode`; Default uses the function attr; legal with `--pdl` and `--cooperative`)
      --nvlink-util     NvlinkUtilCentricScheduling occupancy (`--expert-sim`; occupies every Hyper-Q slot when the profile has NVLink; legal with `--pdl` and `--cooperative`)
      --device-launch   cudaGraphInstantiateFlagDeviceLaunch + device_launch_graph (`--expert-sim`; no mem nodes / graph-update; legal with `--pdl` and `--cooperative`)
      --device-updatable  cudaLaunchAttributeDeviceUpdatableKernelNode (`--expert-sim`; set-params keeps the exec uploaded; not with `--graph-update`; legal with `--pdl` and `--cooperative`)
      --kernel-priority N  cudaLaunchAttributePriority on grouped expert GEMMs (`--expert-sim`; overrides stream create priority; `0` is a valid override; legal with `--pdl` and `--cooperative`)
      --launch-completion  cudaLaunchAttributeLaunchCompletionEvent on grouped expert GEMMs (`--expert-sim`; replica D2D waits kernel start; not with `--device-launch`)
      --programmatic-event  cudaLaunchAttributeProgrammaticEvent on grouped expert GEMMs (`--expert-sim`; replica D2D waits PDL trigger; not with `--device-launch`)
      --stream-attach   cudaStreamAttachMemAsync Single on managed experts (`--expert-sim`; prefetch on compute; implies `--managed`; not with `--seq-streams`)
      --managed-host    cudaMallocManaged Host attach then Global on the copy stream (`--expert-sim`; implies `--managed`; prefetch still overlaps leftover GEMM)
      --prefetch-host   cudaMemPrefetchAsync to host on managed evict (`--expert-sim`; implies `--managed`; next miss prefetches the same alloc back)
      --wait-value      cuStreamWaitValue64 / WriteValue64 copy-ready handshake (`--expert-sim`; 8-byte device mailbox; decode identity stays events)
      --multicast       cuMulticastCreate NVLS replica fanout (`--expert-sim`; implies `--vmm`; needs NVLink)
      --compute-slots N Hyper-Q occupancy (`--expert-sim`; `1` exclusive, `>=2` overlaps leftover prefill with decode on two streams; default profile `1`)
      --decode-sms N    decode-stream SM permille (`--expert-sim`; `1..=1000`; leftover prefill gets the remainder; implies `--decode-priority`; default full chip)
      --prefetch MODE   none|copy-forward|markov|both (`--engine`; default: both)
      --plan-window N   Stay vs Fetch over N unique predicted keys (`--engine`; `0` ungated)
      --plan-threshold N  Stay permille of that window (`--engine`; default: 500)
      --trace-out FILE  append batched MoE ExpertAccess JSONL (`--engine`)

POST /generate takes {\"prompt\": TEXT} or, to render the model's own
tokenizer.chat_template, {\"messages\": [{\"role\": R, \"content\": C}, ...]}
with optional \"add_generation_prompt\" (default true), \"n_predict\" or
\"max_tokens\", and \"stream\" (NDJSON token lines then a final generated
object; `--engine` only). POST /v1/completions and POST /v1/chat/completions
are the same greedy path with an OpenAI `choices` envelope (`text` /
`message.content` is the completion, not the prompt). GET /v1/models and
GET /v1/models/{id} list/retrieve that id (`--model-id`, default GGUF stem).
GET /health is {\"status\":\"ok\"}. GET /metrics is {\"engine\":false} or
Engine counters with `--engine`. POST /tokenize takes the same prompt /
messages body and returns {\"tokens\":[...],\"count\":N} (the ids generate
would prefill). POST /detokenize takes {\"tokens\":[...]} and returns
{\"text\":\"...\"}. GET on those routes is 405. `--engine` stream on
those generate routes is chunked SSE (`data:` lines, then `data: [DONE]`).
Default serve keeps one KV cache across requests (`prefix_hit` in the JSON).
`--engine` admits several connections onto one interned pool so they GEMM
together (`gguf_gemv engine` is the same scheduler). `--kv-page N` interned
completed blocks so a later prompt can hit them after a rewind (`page_hits`).
`--engine` JSON `page_hits` is the intern-hit delta for that sequence.
`--expert-slots` / `--expert-sim` park the same ExpertStore as `gguf_gemv engine`.
`--prefill-chunk` interleaves long prefills with decode. `--decode-first` holds
leftover prefill while any live sequence is already decoding. `--slo-reject` /
`--ttft-slo-ns` drop a waiter whose gpu-sim queue wait meets the TTFT budget
(`--expert-sim`). `--itl-slo-ns` counts later-token ITL misses (does not drop).
`--cuda-graphs` / `--graph-update` / `--graph-set-params` / `--graph-clone` / `--graph-build` / `--graph-build-deps` / `--graph-piecewise` / `--graph-capture-deps` / `--graph-enable` / `--graph-mem` / `--graph-auto-free` / `--graph-mem-trim` / `--timing-events` / `--event-blocking-sync` are
the same SimulatedGpuStore knobs as `gguf_gemv engine`. `--host-func` /
`--blocking-streams` / `--sync-alloc` / `--mempool` / `--mempool-trim` / `--mempool-no-reuse` / `--mempool-max` / `--shareable` / `--vmm-page` /
`--pageable` / `--host-register` / `--host-register-mapped` / `--sync-memops` / `--device-sync-memops` / `--memcpy-batch` / `--memcpy-during` / `--memcpy-any` / `--accessed-by` / `--legacy-null` / `--stream-priority` / `--seq-streams` /
`--kv-sim` / `--kv-bytes` / `--decode-priority` / `--cooperative` / `--pdl` / `--l2-persist` / `--l2-reset` / `--l2-fetch` / `--l2-ratio` / `--l2-streaming` / `--cluster` / `--preferred-cluster` / `--cluster-spread` / `--func-cluster-spread` / `--cluster-load-balance` / `--cluster-must-set` / `--required-cluster` / `--max-shared` / `--func-max-shared` / `--max-l1` / `--non-portable-cluster` / `--sync-policy` / `--device-sync-policy` / `--event-blocking-sync` / `--mem-sync-domain` / `--mem-sync-map` / `--mem-sync-launch` / `--mem-sync-launch-map` / `--shared-mem` / `--func-shared-mem` / `--device-shared-mem` / `--portable-cluster` / `--optin-shared` / `--dynamic-shared` / `--portable-shared` / `--nvlink-util` / `--device-launch` / `--device-updatable` / `--kernel-priority` / `--launch-completion` / `--programmatic-event` / `--stream-attach` / `--managed-host` / `--prefetch-host` / `--wait-value` / `--multicast` /
`--compute-slots` / `--decode-sms` match
`GpuStoreCfg`. `--mempool-max N` is `cudaMemPoolProps::maxSize` (implies `--mempool`;
illegal with `--sync-alloc`; `N>0`; legal with `--pdl` and `--cooperative`). `--memcpy-during` is `cudaMemcpySrcAccessOrderDuringApiCall` on `--memcpy-batch` prefetch (needs `--memcpy-batch`; the batch API waits those copies; identity stays Stream order; legal with `--pdl` and `--cooperative`). `--memcpy-any` is `cudaMemcpySrcAccessOrderAny` on `--memcpy-batch` prefetch (needs `--memcpy-batch`; empty deps; no API wait; not with `--memcpy-during`; legal with `--pdl` and `--cooperative`). `--graph-capture-deps` is `cudaStreamBeginCaptureToGraph` deps on `--graph-piecewise` combo parents (needs `--graph-piecewise`; later fragments chain so sibling GEMMs serialize; empty deps stay extra roots; store GEMM stays per-leaf; legal with `--pdl` and `--cooperative`). `--graph-build-deps` is `cudaGraphAddDependencies` on `--graph-build` combo parents (needs `--graph-build`; later children chain so sibling GEMMs serialize; empty edges stay independent; store GEMM stays per-leaf; legal with `--pdl` and `--cooperative`). `--kv-sim` bills interned KV on the same clock as expert H2D
(distinct from `expertvm kv`; default off). `--decode-priority` ITL samples
the decode compute stream so leftover prefill does not inflate it.
`--compute-slots N` (`N>=2`, with `--decode-priority`) lets leftover prefill
and decode GEMMs overlap at full issue rate. `--pdl` lets consecutive
same-stream expert GEMMs overlap after the previous kernel's programmatic
trigger (needs `--compute-slots` >= 2; illegal with `--cooperative`). `--l2-persist`
is `cudaLaunchAttributeAccessPolicyWindow` over expert pages (persisting L2
after the first fill). `--l2-reset` is `cudaCtxResetPersistingL2Cache` after
each GEMM (implies `--l2-persist`; live; cannot capture; a reused expert does
not keep persisting L2). `--l2-fetch N` is `cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity)`
(`32`/`64`/`128`; implies `--l2-persist`; access-policy windows must align; legal
with `--pdl` and `--cooperative`). `--l2-ratio N` is CUDA `hitRatio` as ‰
(`1..=1000`; implies `--l2-persist`; unset is 1000; a partial ratio bills more HBM
than full persist; legal with `--pdl` and `--cooperative`). `--l2-streaming` is
`cudaAccessPropertyStreaming` for persist GEMM window hits (needs `--l2-persist`;
a reused expert bills full HBM; legal with `--pdl` and `--cooperative`). `--cluster N` is `cudaLaunchAttributeClusterDimension`
on grouped expert GEMMs: the launch occupies `min(N, compute_slots)` Hyper-Q
slots (Hopper portable max 8; legal with `--pdl` and `--cooperative`). `--preferred-cluster N`
is `cudaLaunchAttributePreferredClusterDimension`: occupancy uses that size
when it fits in `compute_slots`, else the required `--cluster` (needs
`--cluster`; must be a multiple of it; legal with `--pdl` and `--cooperative`). `--cluster-spread`
is `cudaLaunchAttributeClusterSchedulingPolicyPreference` Spread: occupies
every Hyper-Q slot even when `N` is smaller than `compute_slots` (no-op
without `--cluster` of at least 2; legal with `--pdl` and `--cooperative`). `--func-cluster-spread`
is `cudaFuncSetAttribute` ClusterSchedulingPolicyPreference Spread: launch Default
inherits that occupancy (no-op without `--cluster` of at least 2; distinct from
launch-attribute `--cluster-spread`; legal with `--pdl` and `--cooperative`). `--cluster-load-balance`
is `cudaLaunchAttributeClusterSchedulingPolicyPreference` LoadBalancing: needs
`--func-cluster-spread`, restores leftover Hyper-Q overlap, exclusive with
`--cluster-spread` (legal with `--pdl` and `--cooperative`). `--cluster-must-set`
is `cudaFuncSetAttribute` ClusterDimMustBeSet (needs `--cluster`; occupancy matches
`--cluster`; SetAttribute is +1 ns; legal with `--pdl` and `--cooperative`). `--required-cluster N`
is `cudaFuncSetAttribute` RequiredClusterWidth (needs `--cluster`; must match `--cluster`;
occupancy matches `--cluster`; SetAttribute is +1 ns; legal with `--pdl` and `--cooperative`). `--max-shared`
is `cudaLaunchAttributePreferredSharedMemoryCarveout` MaxShared: occupies
every Hyper-Q slot (legal with `--pdl` and `--cooperative`). `--func-max-shared`
is `cudaFuncSetAttribute` PreferredSharedMemoryCarveout MaxShared: launch Default
inherits that occupancy (legal with `--pdl` and `--cooperative`). `--max-l1`
is `cudaLaunchAttributePreferredSharedMemoryCarveout` MaxL1: needs
`--func-max-shared`, restores leftover Hyper-Q overlap, exclusive with
`--max-shared` (legal with `--pdl` and `--cooperative`). `--non-portable-cluster`
is `cudaFuncAttributeNonPortableClusterSizeAllowed` so `--cluster N` may
exceed portable size up to the SKU max (Hopper portable 8; legal with
`--pdl` and `--cooperative`). `--sync-policy auto|spin|yield|blocking` is
`cudaLaunchAttributeSynchronizationPolicy` on created streams (host-wait tax
on decode-stream ITL when `--decode-priority`; Auto tax 0). `--device-sync-policy auto|spin|yield|blocking`
is `cudaSetDeviceFlags` SCHEDULE_* (Auto streams inherit host-wait tax; explicit
`--sync-policy` wins; ORs with `--device-sync-memops`; SetDeviceFlags is +1 ns). `--event-blocking-sync`
is `cudaEventBlockingSync` on copy start/end events (implies `--timing-events`;
`synchronize_event` pays `host_sync_blocking_ns`; distinct from `--sync-policy blocking`). `--mem-sync-domain
default|remote` is `cudaLaunchAttributeMemSyncDomain` on the decode compute
stream (prefill stays Default; Remote isolates leftover prefill
`same_domain_fence_permille`; engine implies `--decode-priority`; walker does
not). `--mem-sync-map identity|collapse` is `cudaLaunchAttributeMemSyncDomainMap`
on that decode stream (collapse maps remote→0 and restores leftover prefill
fence tax; needs `--mem-sync-domain remote`; legal with `--pdl` and `--cooperative`). `--mem-sync-launch`
is launch-attribute Remote on grouped expert GEMMs (overrides prefill
inherit-Default so leftover prefill shares the decode Remote domain and fence
tax returns; needs `--mem-sync-domain remote`; legal with `--pdl` and `--cooperative`). `--mem-sync-launch-map`
is launch-attribute collapse map on grouped expert GEMMs (keeps logical
domains different but maps both to physical 0 so leftover prefill fence tax
returns; needs `--mem-sync-domain remote`; legal with `--pdl` and `--cooperative`). `--shared-mem
default|four|eight` is `cudaLaunchAttributeSharedMemoryMode` on grouped expert
GEMMs (Default never scales duration; FourByte / EightByte scale by
`1000 / shared_mem_*_permille`). `--func-shared-mem default|four|eight` is
`cudaFuncSetSharedMemConfig`: launch Default inherits that duration scale
(distinct from launch-attribute `--shared-mem`; legal with `--pdl` and
`--cooperative`). `--device-shared-mem default|four|eight` is
`cudaDeviceSetSharedMemConfig`: launch Default inherits when function config
is also Default (distinct from `--func-shared-mem` / `--shared-mem`; legal
with `--pdl` and `--cooperative`). `--portable-cluster default|portable|non-portable`
is `cudaLaunchAttributePortableClusterSizeMode` on grouped expert GEMMs
(Default uses the function attribute; `portable` always refuses oversize;
`non-portable` allows up to the SKU max even when `--non-portable-cluster` is
off). `--optin-shared` is `cudaFuncAttributeMaxDynamicSharedMemorySize` to the
SKU opt-in max. `--dynamic-shared N` is `cudaLaunchKernel` `sharedMemBytes`
(`N` must be > 0). `--portable-shared default|portable|non-portable` is CUDA 13
`cudaLaunchAttributeSharedMemoryMode` (`cudaSharedMemoryMode`; Default uses the
function attribute; `portable` always refuses oversize; `non-portable` allows
up to the SKU opt-in even when `--optin-shared` is off). `--nvlink-util` is
`cudaLaunchAttributeNvlinkUtilCentricScheduling`: occupies every Hyper-Q slot
when the profile has NVLink (`--expert-8gpu`); without NVLink occupancy is
unchanged (legal with `--pdl` and `--cooperative`). `--device-launch` is
`cudaGraphInstantiateFlagDeviceLaunch` plus `device_launch_graph` after upload
(illegal with `--graph-mem` / `--graph-auto-free` / `--graph-update`; combo
parents stay leaf launches). `--device-updatable` is
`cudaLaunchAttributeDeviceUpdatableKernelNode` so `--graph-set-params` keeps
the exec uploaded (illegal with `--graph-update`; legal with `--pdl` and
`--cooperative`). `--kernel-priority N` is `cudaLaunchAttributePriority` on
grouped expert GEMMs (`None` inherits stream create priority; `0` is a valid
override; legal with `--pdl` and `--cooperative`). `--launch-completion` is
`cudaLaunchAttributeLaunchCompletionEvent` on grouped expert GEMMs so
replica D2D can `wait_event` at kernel start (illegal with `--device-launch`).
`--programmatic-event` is `cudaLaunchAttributeProgrammaticEvent` on those
GEMMs so replica D2D can `wait_event` at the PDL trigger (illegal with
`--device-launch`).
`--stream-attach` is `cudaStreamAttachMemAsync(..., cudaMemAttachSingle)`
on managed experts so miss prefetch and GEMM share the compute stream
(implies `--managed`; illegal with `--seq-streams`).
`--managed-host` is `cudaMallocManaged(..., cudaMemAttachHost)` then
`cudaStreamAttachMemAsync` Global on the copy stream so device prefetch is
legal (implies `--managed`; identity stays Global at alloc; leftover GEMM
still overlaps that prefetch unless `--stream-attach`).
`--prefetch-host` is `cudaMemPrefetchAsync(..., cudaCpuDeviceId)` on managed
LRU evict so the allocation stays live on the host; the next miss prefetches
the same pointer back (implies `--managed`; identity stays `cudaFree`).
`--host-register-mapped` is `cudaHostRegisterMapped` on expert pages
(`alloc_host` then pin+map; implies `--mapped`; identity stays
`cudaHostAllocMapped`; evict is `host_unregister` then `free_host`).
`--sync-memops` is `cuPointerSetAttribute(CU_POINTER_ATTRIBUTE_SYNC_MEMOPS)`
on miss device pages so H2D / managed prefetch is host-synchronous
(illegal with `--mapped` / `--memcpy-batch`; identity stays async pinned H2D).
`--device-sync-memops` is `cudaSetDeviceFlags(cudaDeviceSyncMemops)` so every
memcpy/memset on that GPU is host-synchronous (illegal with `--mapped` /
`--memcpy-batch`; identity stays async pinned H2D; distinct from per-page
`--sync-memops`).
`--wait-value` is `cuStreamWaitValue64` / `cuStreamWriteValue64` after H2D
so compute waits a device mailbox instead of a copy event (decode identity
stays events; GEMM graphs stay kernel-only).
`--cooperative` is
`cudaLaunchCooperativeKernel`: those GEMMs occupy every Hyper-Q slot, so
leftover prefill cannot overlap even with `--compute-slots 2`. `--multicast`
is Hopper NVLS replica fanout (`cuMulticastCreate`; implies `--vmm`; needs
NVLink / `--expert-8gpu`). Decode identity stays copy-engine D2D. Default
profile occupancy is exclusive (`1`). `--decode-sms N` (`1..=1000`) reserves that permille of
peak FLOP/s for decode GEMMs (green-context; leftover prefill gets the
remainder). Default unset is a full chip. `--mapped` / `--managed` / `--vmm` choose miss-page placement
(default pinned H2D). `--prefetch` / `--plan-window` / `--plan-threshold`
match `gguf_gemv engine` (predicted keys only; `--engine`). `--trace-out` writes
router JSONL as sequences finish. Not a production inference server.
";

pub(crate) const MAX_REQ: u64 = 65_536;
const IO_TIMEOUT_SECS: u64 = 30;

/// Parsed `serve` invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[expect(
    clippy::large_enum_variant,
    reason = "ServeArgs is the parsed CLI record; boxing would churn every match"
)]
pub enum ServeCmd {
    /// `--help` / `-h`.
    Help,
    /// Listen with these arguments.
    Run(ServeArgs),
}

/// Arguments for local one-request serve. Seedless greedy, same defaults as `infer`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeArgs {
    /// GGUF path.
    pub path: String,
    /// Tokens to generate when the JSON body omits `n_predict`.
    pub n_predict: usize,
    /// Optional persistent KV capacity. `None` grows to prompt + `n_predict` + 1
    /// on the first request that does not fit the current cache.
    pub n_ctx: Option<usize>,
    /// Paged KV block size in tokens. `None` keeps the dense `max_seq` layout.
    pub kv_page: Option<usize>,
    /// Loopback `HOST:PORT`. Host must be `127.0.0.1` or `localhost`.
    pub bind: String,
    /// OpenAI `model` id and `GET /v1/models` id. Default is the GGUF stem.
    pub model_id: String,
    /// Admit concurrent requests onto one [`crate::Engine`].
    pub engine: bool,
    /// In-flight Engine sequences. `None` is 4 when `--engine`.
    pub max_seqs: Option<usize>,
    /// ExpertStore slots. `None` keeps blob FFN. `Some(0)` is DirectStore.
    pub expert_slots: Option<usize>,
    /// Attach [`expertvm::SimulatedGpuStore`] (`--engine`).
    pub expert_sim: bool,
    /// Example 8×H100 NVLink profile (`--expert-sim`).
    pub expert_8gpu: bool,
    /// Simulated expert page bytes (`--expert-sim`). `None` is 4096.
    pub expert_bytes: Option<u64>,
    /// Prefill tokens per Engine step (`0` = the rest). `--engine` only.
    pub prefill_chunk: usize,
    /// Hold leftover prefill while any live sequence is decoding. `--engine` only.
    pub decode_first: bool,
    /// Drop waiters whose gpu-sim queue wait meets `ttft_slo_ns`. `--engine` only.
    pub slo_reject: bool,
    /// Virtual-ns TTFT budget with `--slo-reject`. `None` never drops.
    pub ttft_slo_ns: Option<u64>,
    /// Later-token gap budget. Misses increment `Engine::itl_slo_miss`.
    pub itl_slo_ns: Option<u64>,
    /// CUDA-like knobs for SimulatedGpuStore. Identity stays default.
    pub gpu_cfg: GpuStoreCfg,
    /// Simulated KV page bytes with `--kv-sim`. `None` uses intern geometry.
    pub kv_bytes: Option<u64>,
    /// Miss-page placement (`--expert-sim`). Default is pinned H2D.
    pub fill: GpuFill,
    /// Predictor prefetch (`--engine`). Default [`Prefetch::Both`].
    pub prefetch: Prefetch,
    /// Unique predicted-key Stay vs Fetch window (`--engine`). `0` is ungated.
    pub plan_window: usize,
    /// Stay permille of the predicted window already resident (`--engine`).
    pub plan_threshold: u32,
    /// Append Engine MoE traces as JSONL (`--engine`). `None` leaves tracing off.
    pub trace_out: Option<String>,
}

impl ServeArgs {
    /// Default `--bind` when omitted.
    pub const DEFAULT_BIND: &'static str = "127.0.0.1:8080";
}

/// Local serve failure.
#[derive(Debug)]
pub enum ServeError {
    /// `--bind` was not loopback or listen failed.
    Bind(String),
    /// HTTP/1.1 request could not be read or framed.
    Http(String),
    /// GGUF path could not be opened.
    MissingFile(String),
    /// I/O while reading or writing a connection.
    Io(std::io::Error),
    /// Model load or generate failed.
    Infer(String),
}

impl std::fmt::Display for ServeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bind(s) | Self::Http(s) | Self::MissingFile(s) | Self::Infer(s) => {
                write!(f, "{s}")
            }
            Self::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ServeError {}

impl From<std::io::Error> for ServeError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

struct ServeNeed {
    engine: bool,
    expert_sim: bool,
    expert_8gpu: bool,
    has_slots: bool,
    has_bytes: bool,
    has_max: bool,
    prefill_chunk: usize,
    decode_first: bool,
    slo_reject: bool,
    has_ttft: bool,
    has_itl: bool,
    has_trace: bool,
    plan: PlannerCli,
}

fn check_serve_need(n: &ServeNeed) -> Result<(), String> {
    if n.has_max && !n.engine {
        return usage_err("--max-seqs requires --engine");
    }
    if n.has_slots && !n.engine {
        return usage_err("--expert-slots requires --engine");
    }
    if n.expert_sim && !n.engine {
        return usage_err("--expert-sim requires --engine");
    }
    if n.expert_8gpu && !n.engine {
        return usage_err("--expert-8gpu requires --engine");
    }
    if n.has_bytes && !n.engine {
        return usage_err("--expert-bytes requires --engine");
    }
    if n.expert_8gpu && !n.expert_sim {
        return usage_err("--expert-8gpu requires --expert-sim");
    }
    if n.has_bytes && !n.expert_sim {
        return usage_err("--expert-bytes requires --expert-sim");
    }
    if n.prefill_chunk > 0 && !n.engine {
        return usage_err("--prefill-chunk requires --engine");
    }
    if n.decode_first && !n.engine {
        return usage_err("--decode-first requires --engine");
    }
    if n.slo_reject && !n.engine {
        return usage_err("--slo-reject requires --engine");
    }
    if n.has_ttft && !n.engine {
        return usage_err("--ttft-slo-ns requires --engine");
    }
    if n.slo_reject && !n.has_ttft {
        return usage_err("--slo-reject requires --ttft-slo-ns");
    }
    if n.has_ttft && !n.slo_reject {
        return usage_err("--ttft-slo-ns requires --slo-reject");
    }
    if (n.slo_reject || n.has_ttft) && !n.expert_sim {
        return usage_err("--slo-reject requires --expert-sim");
    }
    if n.has_itl && !n.engine {
        return usage_err("--itl-slo-ns requires --engine");
    }
    if n.has_itl && !n.expert_sim {
        return usage_err("--itl-slo-ns requires --expert-sim");
    }
    if n.plan.gpu.kv_bytes.is_some() && !n.plan.gpu.kv_sim {
        return usage_err("--kv-bytes requires --kv-sim");
    }
    if let Some(flag) = n.plan.gpu.sim_flag() {
        if !n.engine {
            return usage_err(&format!("{flag} requires --engine"));
        }
        if !n.expert_sim {
            return usage_err(&format!("{flag} requires --expert-sim"));
        }
    }
    if n.plan.gpu.graph_mem && n.plan.gpu.graph_auto_free {
        return usage_err("choose one of --graph-mem, --graph-auto-free");
    }
    if n.plan.gpu.graph_update && n.plan.gpu.graph_set_params {
        return usage_err("choose one of --graph-update, --graph-set-params");
    }
    if n.plan.gpu.graph_update && n.plan.gpu.device_launch {
        return usage_err("choose one of --graph-update, --device-launch");
    }
    if n.plan.gpu.graph_update && n.plan.gpu.device_updatable {
        return usage_err("choose one of --graph-update, --device-updatable");
    }
    if n.plan.gpu.device_launch && (n.plan.gpu.graph_mem || n.plan.gpu.graph_auto_free) {
        return usage_err("device-launch cannot graph-mem");
    }
    if n.plan.gpu.graph_build && n.plan.gpu.graph_piecewise {
        return usage_err("choose one of --graph-build, --graph-piecewise");
    }
    if let Err(e) = n.plan.gpu.check_graph_build_deps() {
        return usage_err(&e);
    }
    if let Err(e) = n.plan.gpu.check_graph_capture_deps() {
        return usage_err(&e);
    }
    if n.plan.gpu.graph_enable && n.plan.gpu.device_launch {
        return usage_err("graph-enable cannot device-launch");
    }
    if n.plan.gpu.launch_completion && n.plan.gpu.device_launch {
        return usage_err("launch-completion cannot device-launch");
    }
    if n.plan.gpu.programmatic_event && n.plan.gpu.device_launch {
        return usage_err("programmatic-event cannot device-launch");
    }
    if n.plan.gpu.stream_attach && n.plan.gpu.seq_streams {
        return usage_err("stream-attach cannot seq-streams");
    }
    if n.plan.gpu.pdl && n.plan.gpu.cooperative {
        return usage_err("choose one of --pdl, --cooperative");
    }
    if n.plan.gpu.shareable
        && (n.plan.gpu.sync_alloc || n.plan.gpu.mapped || n.plan.gpu.managed || n.plan.gpu.vmm)
    {
        return usage_err("shareable needs cudaMallocAsync");
    }
    if n.plan.gpu.mempool_trim && n.plan.gpu.sync_alloc {
        return usage_err("mempool-trim needs cudaMallocAsync");
    }
    if n.plan.gpu.mempool_no_reuse && n.plan.gpu.sync_alloc {
        return usage_err("mempool-no-reuse needs cudaMallocAsync");
    }
    if n.plan.gpu.mempool_max > 0 && n.plan.gpu.sync_alloc {
        return usage_err("mempool-max needs cudaMallocAsync");
    }
    if n.plan.gpu.memcpy_batch
        && (n.plan.gpu.pageable
            || n.plan.gpu.host_register
            || n.plan.gpu.host_register_mapped
            || n.plan.gpu.sync_alloc
            || n.plan.gpu.sync_memops
            || n.plan.gpu.device_sync_memops
            || n.plan.gpu.mapped
            || n.plan.gpu.managed)
    {
        return usage_err("memcpy-batch needs async pinned/vmm H2D");
    }
    if n.plan.gpu.sync_memops && (n.plan.gpu.mapped || n.plan.gpu.host_register_mapped) {
        return usage_err("sync-memops needs device memcpy");
    }
    if n.plan.gpu.device_sync_memops && (n.plan.gpu.mapped || n.plan.gpu.host_register_mapped) {
        return usage_err("device-sync-memops needs device memcpy");
    }
    if n.plan.gpu.host_register && n.plan.gpu.host_register_mapped {
        return usage_err("choose one of --host-register, --host-register-mapped");
    }
    if n.plan.gpu.host_register && (n.plan.gpu.mapped || n.plan.gpu.managed) {
        return usage_err("host-register needs pinned/vmm H2D");
    }
    if let Err(e) = n.plan.gpu.check_preferred_cluster() {
        return usage_err(&e);
    }
    if let Err(e) = n.plan.gpu.check_cluster_must_set() {
        return usage_err(&e);
    }
    if let Err(e) = n.plan.gpu.check_required_cluster() {
        return usage_err(&e);
    }
    if let Err(e) = n.plan.gpu.check_mem_sync_map() {
        return usage_err(&e);
    }
    if let Err(e) = n.plan.gpu.check_mem_sync_launch() {
        return usage_err(&e);
    }
    if let Err(e) = n.plan.gpu.check_mem_sync_launch_map() {
        return usage_err(&e);
    }
    if let Err(e) = n.plan.gpu.check_l2_streaming() {
        return usage_err(&e);
    }
    if let Err(e) = n.plan.gpu.check_memcpy_during() {
        return usage_err(&e);
    }
    if let Err(e) = n.plan.gpu.check_memcpy_any() {
        return usage_err(&e);
    }
    if let Err(e) = n.plan.gpu.check_cluster_load_balance() {
        return usage_err(&e);
    }
    if let Err(e) = n.plan.gpu.check_max_l1() {
        return usage_err(&e);
    }
    if let Some(flag) = n.plan.serve_engine_flag() {
        if !n.engine {
            return usage_err(&format!("{flag} requires --engine"));
        }
    }
    if n.has_trace && !n.engine {
        return usage_err("--trace-out requires --engine");
    }
    Ok(())
}

/// Parse operands after the `serve` verb.
///
/// `serve <path> [--n-predict N] [--n-ctx N] [--kv-page N] [--bind HOST:PORT] [--model-id ID] [--engine] [--max-seqs N] [--expert-slots N] [--expert-sim] [--expert-8gpu] [--expert-bytes N] [--prefill-chunk N] [--decode-first] [--slo-reject] [--ttft-slo-ns N] [--itl-slo-ns N] [--cuda-graphs] [--graph-update] [--graph-set-params] [--graph-clone] [--graph-build] [--graph-build-deps] [--graph-piecewise] [--graph-capture-deps] [--graph-enable] [--graph-mem] [--graph-auto-free] [--graph-mem-trim] [--timing-events] [--event-blocking-sync] [--mapped] [--managed] [--vmm] [--vmm-page N] [--host-func] [--blocking-streams] [--sync-alloc] [--mempool] [--mempool-trim] [--mempool-no-reuse] [--mempool-max N] [--shareable] [--pageable] [--host-register] [--host-register-mapped] [--sync-memops] [--device-sync-memops] [--memcpy-batch] [--memcpy-during] [--memcpy-any] [--accessed-by] [--legacy-null] [--stream-priority] [--seq-streams] [--kv-sim] [--kv-bytes N] [--decode-priority] [--cooperative] [--pdl] [--l2-persist] [--l2-reset] [--l2-fetch N] [--l2-ratio N] [--l2-streaming] [--cluster N] [--preferred-cluster N] [--cluster-spread] [--func-cluster-spread] [--cluster-load-balance] [--cluster-must-set] [--required-cluster N] [--max-shared] [--func-max-shared] [--max-l1] [--non-portable-cluster] [--sync-policy auto|spin|yield|blocking] [--device-sync-policy auto|spin|yield|blocking] [--mem-sync-domain default|remote] [--mem-sync-map identity|collapse] [--mem-sync-launch] [--mem-sync-launch-map] [--shared-mem] default|four|eight] [--func-shared-mem default|four|eight] [--device-shared-mem default|four|eight] [--portable-cluster default|portable|non-portable] [--optin-shared] [--dynamic-shared N] [--portable-shared default|portable|non-portable] [--nvlink-util] [--device-launch] [--device-updatable] [--kernel-priority N] [--launch-completion] [--programmatic-event] [--stream-attach] [--managed-host] [--prefetch-host] [--wait-value] [--multicast] [--compute-slots N] [--decode-sms N] [--prefetch none|copy-forward|markov|both] [--plan-window N] [--plan-threshold N] [--trace-out FILE]`
/// Path may appear before or after flags. `--flag=value` is accepted.
pub fn parse_serve_args<I, S>(args: I) -> Result<ServeCmd, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut path = None;
    let mut n_predict = InferArgs::DEFAULT_N_PREDICT;
    let mut n_ctx = None;
    let mut kv_page = None;
    let mut bind = ServeArgs::DEFAULT_BIND.to_string();
    let mut model_id = None;
    let mut engine = false;
    let mut max_seqs = None;
    let mut expert_slots = None;
    let mut expert_sim = false;
    let mut expert_8gpu = false;
    let mut expert_bytes = None;
    let mut prefill_chunk = 0usize;
    let mut decode_first = false;
    let mut slo_reject = false;
    let mut ttft_slo_ns = None;
    let mut itl_slo_ns = None;
    let mut planner = PlannerCli::default();
    let mut trace_out = None;
    let mut it = args.into_iter();
    while let Some(raw) = it.next() {
        let arg = raw.as_ref();
        if arg == "--help" || arg == "-h" {
            return Ok(ServeCmd::Help);
        }
        let (key, inline) = match arg.split_once('=') {
            Some((k, v)) => (k, Some(v)),
            None => (arg, None),
        };
        match key {
            "--n-predict" | "-n" => {
                n_predict = parse_usize("n-predict", &opt_value("n-predict", inline, &mut it)?)?;
            }
            "--n-ctx" => {
                let n = parse_usize("n-ctx", &opt_value("n-ctx", inline, &mut it)?)?;
                if n == 0 {
                    return usage_err("n-ctx must be > 0");
                }
                n_ctx = Some(n);
            }
            "--kv-page" => {
                let n = parse_usize("kv-page", &opt_value("kv-page", inline, &mut it)?)?;
                if n == 0 {
                    return usage_err("kv-page must be > 0");
                }
                kv_page = Some(n);
            }
            "--bind" => {
                bind = opt_value("bind", inline, &mut it)?;
                let _addr = parse_bind(&bind)?;
            }
            "--model-id" => {
                let id = opt_value("model-id", inline, &mut it)?;
                if id.is_empty() {
                    return usage_err("model-id must be non-empty");
                }
                model_id = Some(id);
            }
            "--engine" => {
                if inline.is_some() {
                    return usage_err("--engine does not take a value");
                }
                engine = true;
            }
            "--max-seqs" => {
                let n = parse_usize("max-seqs", &opt_value("max-seqs", inline, &mut it)?)?;
                if n == 0 {
                    return usage_err("max-seqs must be > 0");
                }
                max_seqs = Some(n);
            }
            "--expert-slots" => {
                expert_slots = Some(parse_usize(
                    "expert-slots",
                    &opt_value("expert-slots", inline, &mut it)?,
                )?);
            }
            "--expert-sim" => {
                if inline.is_some() {
                    return usage_err("--expert-sim does not take a value");
                }
                expert_sim = true;
            }
            "--expert-8gpu" => {
                if inline.is_some() {
                    return usage_err("--expert-8gpu does not take a value");
                }
                expert_8gpu = true;
            }
            "--expert-bytes" => {
                let n = parse_u64("expert-bytes", &opt_value("expert-bytes", inline, &mut it)?)?;
                if n == 0 {
                    return usage_err("expert-bytes must be > 0");
                }
                expert_bytes = Some(n);
            }
            "--prefill-chunk" => {
                prefill_chunk = parse_usize(
                    "prefill-chunk",
                    &opt_value("prefill-chunk", inline, &mut it)?,
                )?;
            }
            "--decode-first" => {
                if inline.is_some() {
                    return usage_err("--decode-first does not take a value");
                }
                decode_first = true;
            }
            "--slo-reject" => {
                if inline.is_some() {
                    return usage_err("--slo-reject does not take a value");
                }
                slo_reject = true;
            }
            "--ttft-slo-ns" => {
                let n = parse_u64("ttft-slo-ns", &opt_value("ttft-slo-ns", inline, &mut it)?)?;
                if n == 0 {
                    return usage_err("ttft-slo-ns must be > 0");
                }
                ttft_slo_ns = Some(n);
            }
            "--itl-slo-ns" => {
                let n = parse_u64("itl-slo-ns", &opt_value("itl-slo-ns", inline, &mut it)?)?;
                if n == 0 {
                    return usage_err("itl-slo-ns must be > 0");
                }
                itl_slo_ns = Some(n);
            }
            "--trace-out" => {
                trace_out = Some(opt_value("trace-out", inline, &mut it)?);
            }
            flag if flag.starts_with('-') => match planner.take(key, inline, &mut it) {
                Ok(true) => {}
                Ok(false) => return usage_err(&format!("unknown flag {flag}")),
                Err(e) => return usage_err(&e),
            },
            other => {
                if path.is_some() {
                    return usage_err(&format!("unexpected argument {other}"));
                }
                path = Some(other.to_string());
            }
        }
    }
    let Some(path) = path else {
        return usage_err("missing GGUF path");
    };
    check_serve_need(&ServeNeed {
        engine,
        expert_sim,
        expert_8gpu,
        has_slots: expert_slots.is_some(),
        has_bytes: expert_bytes.is_some(),
        has_max: max_seqs.is_some(),
        prefill_chunk,
        decode_first,
        slo_reject,
        has_ttft: ttft_slo_ns.is_some(),
        has_itl: itl_slo_ns.is_some(),
        has_trace: trace_out.is_some(),
        plan: planner,
    })?;
    planner.gpu.imply_vmm();
    planner.gpu.imply_shareable();
    planner.gpu.imply_pageable();
    planner.gpu.imply_mapped();
    planner.gpu.imply_managed();
    planner.gpu.imply_decode_priority();
    planner.gpu.imply_l2_persist();
    planner.gpu.imply_timing_events();
    let fill = planner.gpu.fill()?;
    let gpu_cfg = gpu_knobs(planner.gpu);
    let model_id = model_id.unwrap_or_else(|| model_id_from_path(&path));
    Ok(ServeCmd::Run(ServeArgs {
        path,
        n_predict,
        n_ctx,
        kv_page,
        bind,
        model_id,
        engine,
        max_seqs,
        expert_slots,
        expert_sim,
        expert_8gpu,
        expert_bytes,
        prefill_chunk,
        decode_first,
        slo_reject,
        ttft_slo_ns,
        itl_slo_ns,
        gpu_cfg,
        kv_bytes: planner.gpu.kv_bytes,
        fill,
        prefetch: planner.prefetch,
        plan_window: planner.plan_window,
        plan_threshold: planner.plan_threshold,
        trace_out,
    }))
}

/// Bind IPv4 loopback. Host must be `127.0.0.1` or `localhost`.
fn parse_bind(spec: &str) -> Result<(Ipv4Addr, u16), String> {
    let (host, port_s) = spec
        .rsplit_once(':')
        .ok_or_else(|| format!("bind needs HOST:PORT, got {spec}\n{SERVE_USAGE}"))?;
    if host.is_empty() {
        return usage_err("bind host is empty");
    }
    if host == "0.0.0.0" || host == "*" || host == "::" || host == "[::]" {
        return usage_err("bind must be localhost (127.0.0.1), not a public interface");
    }
    if !(host.eq_ignore_ascii_case("127.0.0.1") || host.eq_ignore_ascii_case("localhost")) {
        return usage_err(&format!("bind must be localhost, got {host}"));
    }
    let port: u16 = port_s
        .parse()
        .map_err(|_| format!("invalid port {port_s:?}\n{SERVE_USAGE}"))?;
    Ok((Ipv4Addr::LOCALHOST, port))
}

/// Listen on loopback. `spec` is `HOST:PORT` (`localhost` → `127.0.0.1`).
pub(crate) fn bind_loopback(spec: &str) -> Result<TcpListener, ServeError> {
    let (ip, port) = parse_bind(spec).map_err(ServeError::Bind)?;
    TcpListener::bind((ip, port)).map_err(|e| ServeError::Bind(format!("bind {ip}:{port}: {e}")))
}

/// Open a GGUF from `path` with `File` + `Read` (not `std::fs::read`).
fn read_gguf_path(path: &str) -> Result<Vec<u8>, ServeError> {
    let p = Path::new(path);
    let mut f = File::open(p)
        .map_err(|e| ServeError::MissingFile(format!("missing GGUF file {path}: {e}")))?;
    let mut buf = Vec::new();
    let _n = f.read_to_end(&mut buf)?;
    Ok(buf)
}

/// Load the GGUF, bind loopback, serve one HTTP/1.1 request at a time.
pub fn run_serve(args: &ServeArgs) -> Result<(), ServeError> {
    let bytes = read_gguf_path(&args.path)?;
    let g = load_gguf_owned(bytes).map_err(|e| ServeError::Infer(e.to_string()))?;
    let tok = Tokenizer::from_gguf(&g).map_err(|e| ServeError::Infer(e.to_string()))?;
    let model = Llama::from_gguf(g).map_err(|e| ServeError::Infer(e.to_string()))?;
    let listener = bind_loopback(&args.bind)?;
    let addr = listener.local_addr()?;
    println!("listening {addr}");
    println!("model={} n_predict={}", args.path, args.n_predict);
    if args.engine {
        println!("engine continuous batch; loopback only");
        listener.set_nonblocking(true)?;
        return serve_engine::run_loop(&model, &tok, args, listener);
    }
    println!("one request at a time; loopback only; persistent KV prefix reuse");
    let mut cache = None;
    loop {
        let (mut stream, _peer) = listener.accept()?;
        stream.set_read_timeout(Some(Duration::from_secs(IO_TIMEOUT_SECS)))?;
        stream.set_write_timeout(Some(Duration::from_secs(IO_TIMEOUT_SECS)))?;
        if let Err(e) = handle_connection(&mut stream, &model, &tok, args, &mut cache) {
            eprintln!("{e}");
        }
    }
}

#[derive(Debug)]
pub(crate) struct HttpRequest {
    pub(crate) method: String,
    pub(crate) path: String,
    pub(crate) body: Vec<u8>,
    pub(crate) keep_alive: bool,
    pub(crate) consumed: usize,
}

#[derive(Debug)]
pub(crate) struct GenReq {
    prompt: Option<String>,
    messages: Option<Vec<ChatMessage>>,
    add_generation_prompt: bool,
    pub(crate) n_predict: Option<usize>,
    pub(crate) stream: bool,
    /// OpenAI `/v1/completions` only: include the prompt in `choices[].text`.
    pub(crate) echo: bool,
    /// Tokenize: insert BOS when the tokenizer's `add_bos` is set (default true).
    pub(crate) add_special_tokens: bool,
}

impl GenReq {
    /// The text to prefill: either the raw `prompt`, or `messages` rendered
    /// through the model's own `tokenizer.chat_template`.
    pub(crate) fn resolve(&self, tok: &Tokenizer) -> Result<String, String> {
        match (&self.prompt, &self.messages) {
            (Some(_), Some(_)) => Err("send either prompt or messages, not both".into()),
            (Some(p), None) => Ok(p.clone()),
            (None, Some(m)) => {
                if m.is_empty() {
                    return Err("messages is empty".into());
                }
                tok.apply_chat_template(m, self.add_generation_prompt)
                    .map_err(|e| e.to_string())
            }
            (None, None) => Err("missing prompt".into()),
        }
    }
}

/// Read HTTP/1.1 requests until the client closes or sends `Connection: close`.
fn handle_connection<S: Read + Write>(
    stream: &mut S,
    model: &Llama,
    tok: &Tokenizer,
    args: &ServeArgs,
    cache: &mut Option<KvCache>,
) -> Result<(), ServeError> {
    let mut buf = Vec::new();
    loop {
        match read_request_into(stream, &mut buf) {
            Ok(req) => {
                let keep = req.keep_alive;
                let (status, reason, body) = dispatch(&req, model, tok, args, cache);
                write_http_json(stream, status, reason, &body, keep)?;
                if !keep {
                    return Ok(());
                }
            }
            Err(ServeError::Io(_)) => return Ok(()),
            Err(e) => {
                let (status, reason) = http_err_status(&e);
                let body = json_error(&e.to_string());
                match write_http_json(stream, status, reason, &body, false) {
                    Ok(()) | Err(ServeError::Io(_)) => return Ok(()),
                    Err(werr) => return Err(werr),
                }
            }
        }
    }
}

pub(crate) fn http_err_status(e: &ServeError) -> (u16, &'static str) {
    match e {
        ServeError::Http(m) if m.contains("too large") => (413, "Payload Too Large"),
        _ => (400, "Bad Request"),
    }
}

fn dispatch(
    req: &HttpRequest,
    model: &Llama,
    tok: &Tokenizer,
    args: &ServeArgs,
    cache: &mut Option<KvCache>,
) -> (u16, &'static str, String) {
    if req.method == "GET" {
        if let Some(out) = dispatch_get(&req.path, &args.model_id, false) {
            return out;
        }
    }
    if let Some(out) = dispatch_tok(req, tok) {
        return out;
    }
    let api = ServeApi::from_path(&req.path);
    if req.method != "POST" {
        return (
            405,
            "Method Not Allowed",
            json_err(api, "method must be POST"),
        );
    }
    let Some(api) = api else {
        return (404, "Not Found", json_error("not found"));
    };
    let body = match std::str::from_utf8(&req.body) {
        Ok(s) => s,
        Err(_) => {
            return (
                400,
                "Bad Request",
                json_err(Some(api), "body must be utf-8"),
            )
        }
    };
    let gen = match parse_gen_req(body) {
        Ok(g) => g,
        Err(e) => return (400, "Bad Request", json_err(Some(api), &e)),
    };
    let prompt = match gen.resolve(tok) {
        Ok(p) => p,
        Err(e) => return (400, "Bad Request", json_err(Some(api), &e)),
    };
    if prompt.is_empty() {
        return (400, "Bad Request", json_err(Some(api), "empty prompt"));
    }
    let n_predict = gen.n_predict.unwrap_or(args.n_predict);
    match greedy_parts(
        model,
        tok,
        cache,
        &prompt,
        n_predict,
        args.n_ctx,
        args.kv_page,
    ) {
        Ok(parts) => {
            let hit = cache.as_ref().map_or(0, KvCache::last_prefix_hit);
            let pages = cache.as_ref().map_or(0, KvCache::page_hits);
            (
                200,
                "OK",
                json_ok(api, &parts, hit, pages, &args.model_id, gen.echo),
            )
        }
        Err(LlamaError::EmptyPrompt) => (400, "Bad Request", json_err(Some(api), "empty prompt")),
        Err(e) => (
            500,
            "Internal Server Error",
            json_err(Some(api), &e.to_string()),
        ),
    }
}

/// Route identity for `/generate` vs the OpenAI `/v1/*` aliases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ServeApi {
    /// Native `{generated, prefix_hit, page_hits}`.
    Generate,
    /// `POST /v1/completions` (`choices[].text`).
    Completions,
    /// `POST /v1/chat/completions` (`choices[].message.content`).
    Chat,
}

impl ServeApi {
    pub(crate) fn from_path(path: &str) -> Option<Self> {
        match normalize_path(path_only(path)) {
            "/generate" => Some(Self::Generate),
            "/v1/completions" => Some(Self::Completions),
            "/v1/chat/completions" => Some(Self::Chat),
            _ => None,
        }
    }

    #[must_use]
    pub(crate) fn is_openai(self) -> bool {
        !matches!(self, Self::Generate)
    }
}

/// GGUF path stem, or `"default"` if the path has no file name.
#[must_use]
pub(crate) fn model_id_from_path(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("default")
        .to_string()
}

/// `GET /health`, `GET /v1/models`, `GET /v1/models/{id}`, and idle `GET /metrics`.
///
/// `engine_metrics` is true on `--engine` so `/metrics` is not the idle object.
pub(crate) fn dispatch_get(
    path: &str,
    model_id: &str,
    engine_metrics: bool,
) -> Option<(u16, &'static str, String)> {
    let path = normalize_path(path_only(path));
    match path {
        "/health" => Some((200, "OK", json_health())),
        "/v1/models" => Some((200, "OK", json_models_list(model_id))),
        "/metrics" if !engine_metrics => Some((200, "OK", json_idle_metrics())),
        p => p.strip_prefix("/v1/models/").and_then(|id| {
            if id.is_empty() {
                None
            } else if id == model_id {
                Some((200, "OK", json_model_object(id)))
            } else {
                Some((404, "Not Found", json_openai_error("model not found")))
            }
        }),
    }
}

/// `POST /tokenize` / `POST /detokenize`. GET is 405. Not an Engine sequence.
pub(crate) fn dispatch_tok(
    req: &HttpRequest,
    tok: &Tokenizer,
) -> Option<(u16, &'static str, String)> {
    let path = normalize_path(path_only(&req.path));
    let detok = match path {
        "/tokenize" => false,
        "/detokenize" => true,
        _ => return None,
    };
    if req.method != "POST" {
        return Some((405, "Method Not Allowed", json_error("method must be POST")));
    }
    let body = match std::str::from_utf8(&req.body) {
        Ok(s) => s,
        Err(_) => return Some((400, "Bad Request", json_error("body must be utf-8"))),
    };
    if detok {
        Some(detokenize_body(tok, body))
    } else {
        Some(tokenize_body(tok, body))
    }
}

fn tokenize_body(tok: &Tokenizer, body: &str) -> (u16, &'static str, String) {
    let gen = match parse_gen_req(body) {
        Ok(g) => g,
        Err(e) => return (400, "Bad Request", json_error(&e)),
    };
    let prompt = match gen.resolve(tok) {
        Ok(p) => p,
        Err(e) => return (400, "Bad Request", json_error(&e)),
    };
    if prompt.is_empty() {
        return (400, "Bad Request", json_error("empty prompt"));
    }
    match tokenize_ids(tok, &prompt, gen.add_special_tokens) {
        Ok(ids) if !ids.is_empty() => (200, "OK", json_tokens(&ids)),
        Ok(_) => (400, "Bad Request", json_error("empty prompt")),
        Err(LlamaError::EmptyPrompt) => (400, "Bad Request", json_error("empty prompt")),
        Err(e) => (500, "Internal Server Error", json_error(&e.to_string())),
    }
}

fn tokenize_ids(tok: &Tokenizer, prompt: &str, add_special: bool) -> Result<Vec<u32>, LlamaError> {
    if add_special {
        prompt_ids(tok, prompt)
    } else {
        Ok(tok.encode(prompt)?)
    }
}

fn detokenize_body(tok: &Tokenizer, body: &str) -> (u16, &'static str, String) {
    match parse_detok_req(body) {
        Ok(ids) if !ids.is_empty() => (200, "OK", json_detok(&tok.decode(&ids))),
        Ok(_) => (400, "Bad Request", json_error("empty tokens")),
        Err(e) => (400, "Bad Request", json_error(&e)),
    }
}

pub(crate) fn path_only(path: &str) -> &str {
    path.split(['?', '#']).next().unwrap_or(path)
}

pub(crate) fn normalize_path(path: &str) -> &str {
    path.strip_suffix('/')
        .filter(|p| !p.is_empty())
        .unwrap_or(path)
}

struct GreedyParts {
    full: String,
    completion: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish: &'static str,
}

fn greedy_parts(
    model: &Llama,
    tok: &Tokenizer,
    cache: &mut Option<KvCache>,
    prompt: &str,
    n_predict: usize,
    n_ctx: Option<usize>,
    kv_page: Option<usize>,
) -> Result<GreedyParts, LlamaError> {
    if prompt.is_empty() {
        return Err(LlamaError::EmptyPrompt);
    }
    let mut ids = prompt_ids(tok, prompt)?;
    if ids.is_empty() {
        return Err(LlamaError::EmptyPrompt);
    }
    let prompt_n = ids.len();
    let needed = ids.len().saturating_add(n_predict);
    let kv = model.ensure_cache_page(cache, needed, n_ctx, kv_page)?;
    let full = greedy_generate_cache(model, tok, kv, &mut ids, n_predict)?;
    let completion = tok.decode(ids.get(prompt_n..).unwrap_or(&[]));
    let completion_n = ids.len().saturating_sub(prompt_n);
    Ok(GreedyParts {
        full,
        completion,
        prompt_tokens: prompt_n,
        completion_tokens: completion_n,
        finish: finish_reason(completion_n, n_predict),
    })
}

pub(crate) fn finish_reason(generated: usize, n_predict: usize) -> &'static str {
    if generated < n_predict {
        "stop"
    } else {
        "length"
    }
}

fn json_ok(
    api: ServeApi,
    parts: &GreedyParts,
    hit: usize,
    pages: u64,
    model: &str,
    echo: bool,
) -> String {
    match api {
        ServeApi::Generate => json_generated(&parts.full, hit, pages),
        ServeApi::Completions => json_openai_completion(
            if echo { &parts.full } else { &parts.completion },
            parts.prompt_tokens,
            parts.completion_tokens,
            parts.finish,
            model,
        ),
        ServeApi::Chat => json_openai_chat(
            &parts.completion,
            parts.prompt_tokens,
            parts.completion_tokens,
            parts.finish,
            model,
        ),
    }
}

fn json_err(api: Option<ServeApi>, msg: &str) -> String {
    if api.is_some_and(ServeApi::is_openai) {
        json_openai_error(msg)
    } else {
        json_error(msg)
    }
}

fn read_request_into<R: Read>(
    reader: &mut R,
    buf: &mut Vec<u8>,
) -> Result<HttpRequest, ServeError> {
    loop {
        if let Some(r) = try_parse_http_request(buf)? {
            let n = r.consumed.min(buf.len());
            let rest = buf.split_off(n);
            *buf = rest;
            return Ok(r);
        }
        let mut tmp = [0u8; 512];
        let n = reader.read(&mut tmp)?;
        if n == 0 {
            if buf.is_empty() {
                return Err(std::io::Error::from(ErrorKind::UnexpectedEof).into());
            }
            return match try_parse_http_request(buf)? {
                Some(r) => {
                    let n = r.consumed.min(buf.len());
                    let rest = buf.split_off(n);
                    *buf = rest;
                    Ok(r)
                }
                None => Err(ServeError::Http("unexpected eof".into())),
            };
        }
        let chunk = tmp
            .get(..n)
            .ok_or_else(|| ServeError::Http("read slice".into()))?;
        buf.extend_from_slice(chunk);
        if u64::try_from(buf.len()).unwrap_or(u64::MAX) > MAX_REQ {
            return Err(ServeError::Http("request too large".into()));
        }
    }
}

pub(crate) fn try_parse_http_request(buf: &[u8]) -> Result<Option<HttpRequest>, ServeError> {
    let Some((header_bytes, body_so_far)) = header_body_split(buf) else {
        return Ok(None);
    };
    let headers = std::str::from_utf8(header_bytes)
        .map_err(|_| ServeError::Http("headers must be utf-8".into()))?;
    if let Some(te) = header_value(headers, "transfer-encoding") {
        if !te.eq_ignore_ascii_case("identity") {
            return Err(ServeError::Http("chunked transfer unsupported".into()));
        }
    }
    let cl = match header_value(headers, "content-length") {
        Some(s) => s
            .parse::<u64>()
            .map_err(|_| ServeError::Http(format!("invalid content-length {s:?}")))?,
        None => 0,
    };
    if cl > MAX_REQ {
        return Err(ServeError::Http("request too large".into()));
    }
    let need = usize::try_from(cl).map_err(|_| ServeError::Http("content-length".into()))?;
    if body_so_far.len() < need {
        return Ok(None);
    }
    let body = body_so_far
        .get(..need)
        .ok_or_else(|| ServeError::Http("body slice".into()))?
        .to_vec();
    let mut lines = headers.split('\n');
    let req_line = lines.next().unwrap_or("").strip_suffix('\r').unwrap_or("");
    let mut parts = req_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| ServeError::Http("missing method".into()))?;
    let path = parts
        .next()
        .ok_or_else(|| ServeError::Http("missing path".into()))?;
    let version = parts
        .next()
        .ok_or_else(|| ServeError::Http("missing version".into()))?;
    if !version.starts_with("HTTP/") {
        return Err(ServeError::Http(format!("bad version {version}")));
    }
    let prefix = buf.len().saturating_sub(body_so_far.len());
    let consumed = prefix.saturating_add(need);
    let connection = header_value(headers, "connection");
    Ok(Some(HttpRequest {
        method: method.to_string(),
        path: path.to_string(),
        body,
        keep_alive: http_keep_alive(version, connection.as_deref()),
        consumed,
    }))
}

fn http_keep_alive(version: &str, connection: Option<&str>) -> bool {
    if connection.is_some_and(|c| c.eq_ignore_ascii_case("close")) {
        return false;
    }
    if version == "HTTP/1.0" {
        return connection.is_some_and(|c| c.eq_ignore_ascii_case("keep-alive"));
    }
    true
}

fn connection_value(keep_alive: bool) -> &'static str {
    if keep_alive {
        "keep-alive"
    } else {
        "close"
    }
}

fn header_body_split(buf: &[u8]) -> Option<(&[u8], &[u8])> {
    if let Some(i) = find_subslice(buf, b"\r\n\r\n") {
        let headers = buf.get(..i)?;
        let body = buf.get(i.saturating_add(4)..)?;
        return Some((headers, body));
    }
    if let Some(i) = find_subslice(buf, b"\n\n") {
        let headers = buf.get(..i)?;
        let body = buf.get(i.saturating_add(2)..)?;
        return Some((headers, body));
    }
    None
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

fn header_value(headers: &str, name: &str) -> Option<String> {
    let mut lines = headers.split('\n');
    let _req = lines.next();
    for line in lines {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if let Some((k, v)) = line.split_once(':') {
            if k.eq_ignore_ascii_case(name) {
                return Some(v.trim().to_string());
            }
        }
    }
    None
}

fn write_http_json<W: Write>(
    w: &mut W,
    status: u16,
    reason: &str,
    json: &str,
    keep_alive: bool,
) -> Result<(), ServeError> {
    w.write_all(&http_json_bytes(status, reason, json, keep_alive))?;
    w.flush()?;
    Ok(())
}

/// HTTP/1.1 JSON response bytes.
pub(crate) fn http_json_bytes(status: u16, reason: &str, json: &str, keep_alive: bool) -> Vec<u8> {
    let len = json.len();
    let conn = connection_value(keep_alive);
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {len}\r\nConnection: {conn}\r\n\r\n"
    );
    let mut out = head.into_bytes();
    out.extend_from_slice(json.as_bytes());
    out
}

pub(crate) fn json_generated(text: &str, prefix_hit: usize, page_hits: u64) -> String {
    let mut s = String::from("{\"generated\":");
    append_json_string(&mut s, text);
    s.push_str(",\"prefix_hit\":");
    s.push_str(&prefix_hit.to_string());
    s.push_str(",\"page_hits\":");
    s.push_str(&page_hits.to_string());
    s.push('}');
    s
}

/// One NDJSON token object plus a trailing newline.
pub(crate) fn json_token(token: &str) -> String {
    let mut s = String::from("{\"token\":");
    append_json_string(&mut s, token);
    s.push_str("}\n");
    s
}

/// HTTP/1.1 chunked NDJSON headers.
pub(crate) fn http_chunked_headers(keep_alive: bool) -> Vec<u8> {
    http_chunked_typed(keep_alive, "application/x-ndjson")
}

/// HTTP/1.1 chunked SSE headers (`text/event-stream`).
pub(crate) fn http_sse_headers(keep_alive: bool) -> Vec<u8> {
    http_chunked_typed(keep_alive, "text/event-stream")
}

fn http_chunked_typed(keep_alive: bool, content_type: &str) -> Vec<u8> {
    let conn = connection_value(keep_alive);
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nTransfer-Encoding: chunked\r\nConnection: {conn}\r\n\r\n"
    )
    .into_bytes()
}

/// One HTTP/1.1 chunk: hex size, payload, CRLF.
pub(crate) fn http_chunk(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(format!("{:x}\r\n", payload.len()).as_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(b"\r\n");
    out
}

/// Last HTTP/1.1 chunk (`0` + CRLF + trailer CRLF).
pub(crate) fn http_chunk_end() -> Vec<u8> {
    b"0\r\n\r\n".to_vec()
}

pub(crate) fn json_error(msg: &str) -> String {
    let mut s = String::from("{\"error\":");
    append_json_string(&mut s, msg);
    s.push('}');
    s
}

pub(crate) fn json_openai_error(msg: &str) -> String {
    let mut s = String::from("{\"error\":{\"message\":");
    append_json_string(&mut s, msg);
    s.push_str(",\"type\":\"invalid_request_error\"}}");
    s
}

pub(crate) fn json_openai_completion(
    text: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish: &str,
    model: &str,
) -> String {
    let mut choice = String::from("\"text\":");
    append_json_string(&mut choice, text);
    choice.push_str(",\"index\":0,\"finish_reason\":");
    append_json_string(&mut choice, finish);
    openai_envelope(
        "cmpl-0",
        "text_completion",
        model,
        &choice,
        prompt_tokens,
        completion_tokens,
    )
}

pub(crate) fn json_openai_chat(
    text: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish: &str,
    model: &str,
) -> String {
    let mut choice = String::from("\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    append_json_string(&mut choice, text);
    choice.push_str("},\"finish_reason\":");
    append_json_string(&mut choice, finish);
    openai_envelope(
        "chatcmpl-0",
        "chat.completion",
        model,
        &choice,
        prompt_tokens,
        completion_tokens,
    )
}

fn openai_envelope(
    id: &str,
    object: &str,
    model: &str,
    choice: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> String {
    let total = prompt_tokens.saturating_add(completion_tokens);
    let mut s = String::from("{\"id\":");
    append_json_string(&mut s, id);
    s.push_str(",\"object\":");
    append_json_string(&mut s, object);
    s.push_str(",\"model\":");
    append_json_string(&mut s, model);
    s.push_str(",\"choices\":[{");
    s.push_str(choice);
    s.push_str("}],\"usage\":{\"prompt_tokens\":");
    s.push_str(&prompt_tokens.to_string());
    s.push_str(",\"completion_tokens\":");
    s.push_str(&completion_tokens.to_string());
    s.push_str(",\"total_tokens\":");
    s.push_str(&total.to_string());
    s.push_str("}}");
    s
}

pub(crate) fn json_health() -> String {
    String::from("{\"status\":\"ok\"}")
}

pub(crate) fn json_idle_metrics() -> String {
    String::from("{\"engine\":false}")
}

pub(crate) fn json_model_object(id: &str) -> String {
    let mut s = String::from("{\"id\":");
    append_json_string(&mut s, id);
    s.push_str(",\"object\":\"model\",\"created\":0,\"owned_by\":\"llama-rust\"}");
    s
}

pub(crate) fn json_models_list(id: &str) -> String {
    let mut s = String::from("{\"object\":\"list\",\"data\":[");
    s.push_str(&json_model_object(id));
    s.push_str("]}");
    s
}

fn json_tokens(ids: &[u32]) -> String {
    let mut s = String::from("{\"tokens\":[");
    for (i, id) in ids.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&id.to_string());
    }
    s.push_str("],\"count\":");
    s.push_str(&ids.len().to_string());
    s.push('}');
    s
}

fn json_detok(text: &str) -> String {
    let mut s = String::from("{\"text\":");
    append_json_string(&mut s, text);
    s.push('}');
    s
}

/// One SSE `data:` line for `/v1/completions` streaming.
pub(crate) fn sse_completion_delta(text: &str, finish: Option<&str>, model: &str) -> String {
    let mut json = String::from("{\"id\":\"cmpl-0\",\"object\":\"text_completion\",\"model\":");
    append_json_string(&mut json, model);
    json.push_str(",\"choices\":[{\"text\":");
    append_json_string(&mut json, text);
    json.push_str(",\"index\":0,\"finish_reason\":");
    match finish {
        Some(f) => append_json_string(&mut json, f),
        None => json.push_str("null"),
    }
    json.push_str("}]}");
    sse_data(&json)
}

/// One SSE `data:` line for `/v1/chat/completions` streaming.
pub(crate) fn sse_chat_delta(text: &str, finish: Option<&str>, model: &str) -> String {
    let mut json =
        String::from("{\"id\":\"chatcmpl-0\",\"object\":\"chat.completion.chunk\",\"model\":");
    append_json_string(&mut json, model);
    json.push_str(",\"choices\":[{\"index\":0,\"delta\":{");
    if !text.is_empty() {
        json.push_str("\"content\":");
        append_json_string(&mut json, text);
    }
    json.push_str("},\"finish_reason\":");
    match finish {
        Some(f) => append_json_string(&mut json, f),
        None => json.push_str("null"),
    }
    json.push_str("}]}");
    sse_data(&json)
}

/// OpenAI stream terminator.
pub(crate) fn sse_done() -> String {
    String::from("data: [DONE]\n\n")
}

fn sse_data(json: &str) -> String {
    let mut s = String::from("data: ");
    s.push_str(json);
    s.push_str("\n\n");
    s
}

fn append_json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 32 => {
                let n = u32::from(c);
                out.push_str(&format!("\\u{n:04x}"));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Scan<'a> {
    s: &'a str,
    i: usize,
}

impl Scan<'_> {
    fn rest(&self) -> &str {
        self.s.get(self.i..).unwrap_or("")
    }

    fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.i = self.i.saturating_add(c.len_utf8());
        Some(c)
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_ascii_whitespace()) {
            let _c = self.bump();
        }
    }

    fn expect_char(&mut self, want: char) -> Result<(), String> {
        self.skip_ws();
        match self.bump() {
            Some(c) if c == want => Ok(()),
            Some(c) => Err(format!("expected {want:?}, got {c:?}")),
            None => Err(format!("expected {want:?}")),
        }
    }

    fn expect_lit(&mut self, lit: &str) -> Result<(), String> {
        for want in lit.chars() {
            match self.bump() {
                Some(c) if c == want => {}
                _ => return Err(format!("expected {lit:?}")),
            }
        }
        Ok(())
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.skip_ws();
        if self.bump() != Some('"') {
            return Err("expected string".into());
        }
        let mut out = String::new();
        loop {
            match self.bump() {
                None => return Err("unterminated string".into()),
                Some('"') => return Ok(out),
                Some('\\') => match self.bump() {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('/') => out.push('/'),
                    Some('n') => out.push('\n'),
                    Some('r') => out.push('\r'),
                    Some('t') => out.push('\t'),
                    Some(c) => return Err(format!("bad escape {c}")),
                    None => return Err("unterminated escape".into()),
                },
                Some(c) if u32::from(c) < 32 => return Err("raw control in string".into()),
                Some(c) => out.push(c),
            }
        }
    }

    fn parse_usize(&mut self) -> Result<usize, String> {
        self.skip_ws();
        if self.peek() == Some('-') {
            return Err("n_predict must be >= 0".into());
        }
        let start = self.i;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            let _c = self.bump();
        }
        let digits = self.s.get(start..self.i).unwrap_or("");
        if digits.is_empty() {
            return Err("expected number".into());
        }
        if self.peek() == Some('.') {
            return Err("n_predict must be an integer".into());
        }
        digits
            .parse::<usize>()
            .map_err(|_| format!("invalid n_predict {digits:?}"))
    }

    fn parse_bool(&mut self) -> Result<bool, String> {
        self.skip_ws();
        match self.peek() {
            Some('t') => self.expect_lit("true").map(|()| true),
            Some('f') => self.expect_lit("false").map(|()| false),
            _ => Err("expected true or false".into()),
        }
    }

    /// `[{"role": "...", "content": "..."}, ...]`, the OpenAI message shape.
    ///
    /// Unknown keys inside a message are skipped so a client can send the
    /// fields this engine has no use for. Nested objects and arrays inside a
    /// message are still rejected, same as at the top level.
    fn parse_messages(&mut self) -> Result<Vec<ChatMessage>, String> {
        self.expect_char('[')?;
        let mut out = Vec::new();
        let mut need_comma = false;
        loop {
            self.skip_ws();
            if self.peek() == Some(']') {
                let _c = self.bump();
                return Ok(out);
            }
            if need_comma {
                self.expect_char(',')?;
                self.skip_ws();
                if self.peek() == Some(']') {
                    return Err("trailing comma in messages".into());
                }
            }
            out.push(self.parse_message()?);
            need_comma = true;
        }
    }

    fn parse_message(&mut self) -> Result<ChatMessage, String> {
        self.expect_char('{')?;
        let mut role = None;
        let mut content = None;
        let mut need_comma = false;
        loop {
            self.skip_ws();
            if self.peek() == Some('}') {
                let _c = self.bump();
                break;
            }
            if need_comma {
                self.expect_char(',')?;
                self.skip_ws();
                if self.peek() == Some('}') {
                    return Err("trailing comma in message".into());
                }
            }
            let key = self.parse_string()?;
            self.expect_char(':')?;
            match key.as_str() {
                "role" => role = Some(self.parse_string()?),
                "content" => content = Some(self.parse_string()?),
                _ => self.skip_value()?,
            }
            need_comma = true;
        }
        let role = role.ok_or_else(|| "message missing role".to_string())?;
        let content = content.ok_or_else(|| "message missing content".to_string())?;
        Ok(ChatMessage::new(role, content))
    }

    fn skip_value(&mut self) -> Result<(), String> {
        self.skip_ws();
        match self.peek() {
            Some('"') => {
                let _s = self.parse_string()?;
                Ok(())
            }
            Some(c) if c == '-' || c.is_ascii_digit() => {
                if c == '-' {
                    let _c = self.bump();
                }
                if !matches!(self.peek(), Some(d) if d.is_ascii_digit()) {
                    return Err("expected number".into());
                }
                while matches!(self.peek(), Some(d) if d.is_ascii_digit()) {
                    let _c = self.bump();
                }
                if self.peek() == Some('.') {
                    return Err("floats unsupported".into());
                }
                Ok(())
            }
            Some('t') => self.expect_lit("true"),
            Some('f') => self.expect_lit("false"),
            Some('n') => self.expect_lit("null"),
            Some('{') | Some('[') => Err("nested json unsupported".into()),
            _ => Err("expected value".into()),
        }
    }

    fn parse_u32(&mut self) -> Result<u32, String> {
        self.skip_ws();
        if self.peek() == Some('-') {
            return Err("tokens must be >= 0".into());
        }
        let start = self.i;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            let _c = self.bump();
        }
        let digits = self.s.get(start..self.i).unwrap_or("");
        if digits.is_empty() {
            return Err("expected number".into());
        }
        if self.peek() == Some('.') {
            return Err("tokens must be an integer".into());
        }
        digits
            .parse::<u32>()
            .map_err(|_| format!("invalid token {digits:?}"))
    }

    /// `[1, 2, 3]` token ids for `POST /detokenize`.
    fn parse_u32_array(&mut self) -> Result<Vec<u32>, String> {
        self.expect_char('[')?;
        let mut out = Vec::new();
        let mut need_comma = false;
        loop {
            self.skip_ws();
            if self.peek() == Some(']') {
                let _c = self.bump();
                return Ok(out);
            }
            if need_comma {
                self.expect_char(',')?;
                self.skip_ws();
                if self.peek() == Some(']') {
                    return Err("trailing comma in tokens".into());
                }
            }
            out.push(self.parse_u32()?);
            need_comma = true;
        }
    }
}

fn parse_detok_req(s: &str) -> Result<Vec<u32>, String> {
    let mut scan = Scan { s, i: 0 };
    scan.expect_char('{')?;
    let mut tokens = None;
    let mut need_comma = false;
    loop {
        scan.skip_ws();
        if scan.peek() == Some('}') {
            let _c = scan.bump();
            break;
        }
        if need_comma {
            scan.expect_char(',')?;
            scan.skip_ws();
            if scan.peek() == Some('}') {
                return Err("trailing comma".into());
            }
        }
        let key = scan.parse_string()?;
        scan.expect_char(':')?;
        match key.as_str() {
            "tokens" => tokens = Some(scan.parse_u32_array()?),
            _ => scan.skip_value()?,
        }
        need_comma = true;
    }
    scan.skip_ws();
    if scan.peek().is_some() {
        return Err("trailing junk after json".into());
    }
    tokens.ok_or_else(|| "missing tokens".to_string())
}

pub(crate) fn parse_gen_req(s: &str) -> Result<GenReq, String> {
    let mut scan = Scan { s, i: 0 };
    scan.expect_char('{')?;
    let mut prompt = None;
    let mut messages = None;
    let mut add_generation_prompt = None;
    let mut n_predict = None;
    let mut stream = None;
    let mut echo = None;
    let mut add_special_tokens = None;
    let mut need_comma = false;
    loop {
        scan.skip_ws();
        if scan.peek() == Some('}') {
            let _c = scan.bump();
            break;
        }
        if need_comma {
            scan.expect_char(',')?;
            scan.skip_ws();
            if scan.peek() == Some('}') {
                return Err("trailing comma".into());
            }
        }
        let key = scan.parse_string()?;
        scan.expect_char(':')?;
        match key.as_str() {
            "prompt" => prompt = Some(scan.parse_string()?),
            "messages" => messages = Some(scan.parse_messages()?),
            "add_generation_prompt" => add_generation_prompt = Some(scan.parse_bool()?),
            "n_predict" => set_n_predict(&mut n_predict, scan.parse_usize()?, "n_predict")?,
            "max_tokens" => set_n_predict(&mut n_predict, scan.parse_usize()?, "max_tokens")?,
            "stream" => stream = Some(scan.parse_bool()?),
            "echo" => echo = Some(scan.parse_bool()?),
            "add_special_tokens" => add_special_tokens = Some(scan.parse_bool()?),
            "n" => {
                let v = scan.parse_usize()?;
                if v != 1 {
                    return Err("n must be 1".into());
                }
            }
            _ => scan.skip_value()?,
        }
        need_comma = true;
    }
    scan.skip_ws();
    if scan.peek().is_some() {
        return Err("trailing junk after json".into());
    }
    Ok(GenReq {
        prompt,
        messages,
        // A chat request is asking the model to speak next, so the generation
        // prompt is on unless the caller says otherwise.
        add_generation_prompt: add_generation_prompt.unwrap_or(true),
        n_predict,
        stream: stream.unwrap_or(false),
        echo: echo.unwrap_or(false),
        add_special_tokens: add_special_tokens.unwrap_or(true),
    })
}

fn set_n_predict(slot: &mut Option<usize>, v: usize, name: &str) -> Result<(), String> {
    if let Some(old) = *slot {
        if old != v {
            return Err(format!("{name} disagrees with n_predict/max_tokens"));
        }
    }
    *slot = Some(v);
    Ok(())
}

fn usage_err<T>(msg: &str) -> Result<T, String> {
    Err(format!("{msg}\n{SERVE_USAGE}"))
}

fn opt_value<I, S>(name: &str, inline: Option<&str>, it: &mut I) -> Result<String, String>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    if let Some(v) = inline {
        return Ok(v.to_string());
    }
    match it.next() {
        Some(s) => Ok(s.as_ref().to_string()),
        None => usage_err(&format!("missing --{name} value")),
    }
}

fn parse_usize(name: &str, s: &str) -> Result<usize, String> {
    s.parse::<usize>()
        .map_err(|_| format!("invalid {name} {s:?}\n{SERVE_USAGE}"))
}

fn parse_u64(name: &str, s: &str) -> Result<u64, String> {
    s.parse::<u64>()
        .map_err(|_| format!("invalid {name} {s:?}\n{SERVE_USAGE}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{
        greedy_generate_cache, greedy_generate_ctx, greedy_generate_slot, prompt_ids,
        tiny_llama_gguf,
    };
    use expertvm::{
        MemSyncDomain, PortableClusterMode, PortableSharedMode, SharedMemoryMode,
        SynchronizationPolicy,
    };
    use std::io::Cursor;
    use std::net::{IpAddr, Ipv4Addr};

    struct RwBuf {
        input: Cursor<Vec<u8>>,
        output: Vec<u8>,
    }

    impl Read for RwBuf {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.input.read(buf)
        }
    }

    impl Write for RwBuf {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.output.write(buf)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn run(args: &[&str]) -> ServeArgs {
        match parse_serve_args(args).expect("parse") {
            ServeCmd::Run(a) => a,
            ServeCmd::Help => panic!("expected Run"),
        }
    }

    fn tiny_model() -> (Llama, Tokenizer) {
        let g = load_gguf_owned(tiny_llama_gguf()).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("model");
        (model, tok)
    }

    fn defaults() -> ServeArgs {
        ServeArgs {
            path: "tiny.gguf".into(),
            n_predict: InferArgs::DEFAULT_N_PREDICT,
            n_ctx: None,
            kv_page: None,
            bind: ServeArgs::DEFAULT_BIND.into(),
            model_id: "tiny".into(),
            engine: false,
            max_seqs: None,
            expert_slots: None,
            expert_sim: false,
            expert_8gpu: false,
            expert_bytes: None,
            prefill_chunk: 0,
            decode_first: false,
            slo_reject: false,
            ttft_slo_ns: None,
            itl_slo_ns: None,
            gpu_cfg: GpuStoreCfg::default(),
            kv_bytes: None,
            fill: GpuFill::Pinned,
            prefetch: Prefetch::Both,
            plan_window: 0,
            plan_threshold: 500,
            trace_out: None,
        }
    }

    fn exchange(raw: &str, model: &Llama, tok: &Tokenizer, args: &ServeArgs) -> (String, String) {
        let mut cache = None;
        exchange_on(raw, model, tok, args, &mut cache)
    }

    fn exchange_on(
        raw: &str,
        model: &Llama,
        tok: &Tokenizer,
        args: &ServeArgs,
        cache: &mut Option<KvCache>,
    ) -> (String, String) {
        let mut sock = RwBuf {
            input: Cursor::new(raw.as_bytes().to_vec()),
            output: Vec::new(),
        };
        handle_connection(&mut sock, model, tok, args, cache).expect("handle");
        let (head, body) = header_body_split(&sock.output).expect("split");
        let head = std::str::from_utf8(head).expect("head utf8").to_string();
        let body = std::str::from_utf8(body).expect("body utf8").to_string();
        (head, body)
    }

    fn post_json(json: &str) -> String {
        post_path("/generate", json)
    }

    fn post_path(path: &str, json: &str) -> String {
        format!(
            "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: {}\r\n\r\n{json}",
            json.len()
        )
    }

    fn json_field_string(s: &str, want: &str) -> Result<String, String> {
        json_string_at(s, want)
    }

    fn json_string_at(s: &str, want: &str) -> Result<String, String> {
        let pat = format!("\"{want}\":");
        let i = s.find(&pat).ok_or_else(|| format!("missing {want}"))?;
        let rest = s.get(i.saturating_add(pat.len())..).ok_or("rest")?;
        let mut scan = Scan { s: rest, i: 0 };
        scan.parse_string()
    }

    fn post_v1(path: &str, json: &str) -> String {
        format!(
            "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: {}\r\n\r\n{json}",
            json.len()
        )
    }

    #[test]
    fn omitted_flags_keep_infer_n_predict_and_loopback() {
        let a = run(&["tiny.gguf"]);
        assert_eq!(a.path, "tiny.gguf");
        assert_eq!(a.n_predict, InferArgs::DEFAULT_N_PREDICT);
        assert_eq!(a.n_predict, 2);
        assert_eq!(a.n_ctx, None);
        assert_eq!(a.kv_page, None);
        assert_eq!(a.bind, ServeArgs::DEFAULT_BIND);
        assert_eq!(a.bind, "127.0.0.1:8080");
        assert_eq!(a.model_id, "tiny");
        assert!(!a.engine);
        assert_eq!(a.max_seqs, None);
        assert_eq!(a.prefill_chunk, 0);
        assert!(!a.decode_first);
        assert!(!a.slo_reject);
        assert_eq!(a.ttft_slo_ns, None);
        assert_eq!(a.itl_slo_ns, None);
        assert_eq!(a.gpu_cfg, GpuStoreCfg::default());
        assert_eq!(a.prefetch, Prefetch::Both);
        assert_eq!(a.plan_window, 0);
        assert_eq!(a.plan_threshold, 500);
        assert_eq!(a.trace_out, None);
    }

    #[test]
    fn flags_and_path_after() {
        let a = run(&[
            "--n-predict",
            "4",
            "--n-ctx",
            "16",
            "--bind",
            "localhost:0",
            "m.gguf",
        ]);
        assert_eq!(
            a,
            ServeArgs {
                path: "m.gguf".into(),
                n_predict: 4,
                n_ctx: Some(16),
                kv_page: None,
                bind: "localhost:0".into(),
                model_id: "m".into(),
                engine: false,
                max_seqs: None,
                expert_slots: None,
                expert_sim: false,
                expert_8gpu: false,
                expert_bytes: None,
                prefill_chunk: 0,
                decode_first: false,
                slo_reject: false,
                ttft_slo_ns: None,
                itl_slo_ns: None,
                gpu_cfg: GpuStoreCfg::default(),
                kv_bytes: None,
                fill: GpuFill::Pinned,
                prefetch: Prefetch::Both,
                plan_window: 0,
                plan_threshold: 500,
                trace_out: None,
            }
        );
    }

    #[test]
    fn short_flags_equals_and_path_first() {
        let a = run(&[
            "model.gguf",
            "-n=0",
            "--n-ctx=8",
            "--kv-page=2",
            "--bind=127.0.0.1:9",
        ]);
        assert_eq!(a.n_predict, 0);
        assert_eq!(a.n_ctx, Some(8));
        assert_eq!(a.kv_page, Some(2));
        assert_eq!(a.bind, "127.0.0.1:9");
    }

    #[test]
    fn help_is_not_a_run() {
        assert_eq!(parse_serve_args(["--help"]).unwrap(), ServeCmd::Help);
        assert_eq!(parse_serve_args(["-h", "x.gguf"]).unwrap(), ServeCmd::Help);
    }

    #[test]
    fn missing_path_and_bad_values_error() {
        let err = parse_serve_args(["--n-predict", "2"]).unwrap_err();
        assert!(err.contains("missing GGUF path"), "{err}");
        assert!(err.contains("gguf_gemv serve"), "{err}");
        let err = parse_serve_args(["m.gguf", "-n", "x"]).unwrap_err();
        assert!(err.contains("invalid n-predict"), "{err}");
        let err = parse_serve_args(["m.gguf", "--n-ctx", "0"]).unwrap_err();
        assert!(err.contains("n-ctx must be > 0"), "{err}");
        let err = parse_serve_args(["m.gguf", "--kv-page", "0"]).unwrap_err();
        assert!(err.contains("kv-page must be > 0"), "{err}");
        let err = parse_serve_args(["m.gguf", "--nope"]).unwrap_err();
        assert!(err.contains("unknown flag"), "{err}");
        let err = parse_serve_args(["a.gguf", "b.gguf"]).unwrap_err();
        assert!(err.contains("unexpected argument"), "{err}");
        let err = parse_serve_args(["m.gguf", "--bind", "0.0.0.0:80"]).unwrap_err();
        assert!(err.contains("localhost"), "{err}");
        let err = parse_serve_args(["--bind"]).unwrap_err();
        assert!(err.contains("missing --bind value"), "{err}");
        let err = parse_serve_args(["m.gguf", "--max-seqs", "2"]).unwrap_err();
        assert!(err.contains("--max-seqs requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--max-seqs", "0"]).unwrap_err();
        assert!(err.contains("max-seqs must be > 0"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine=1"]).unwrap_err();
        assert!(err.contains("--engine does not take a value"), "{err}");
    }

    #[test]
    fn engine_flag_and_max_seqs() {
        let a = run(&["tiny.gguf", "--engine"]);
        assert!(a.engine);
        assert_eq!(a.max_seqs, None);
        let a = run(&["--engine", "--max-seqs", "8", "m.gguf"]);
        assert_eq!(
            a,
            ServeArgs {
                path: "m.gguf".into(),
                n_predict: InferArgs::DEFAULT_N_PREDICT,
                n_ctx: None,
                kv_page: None,
                bind: ServeArgs::DEFAULT_BIND.into(),
                model_id: "m".into(),
                engine: true,
                max_seqs: Some(8),
                expert_slots: None,
                expert_sim: false,
                expert_8gpu: false,
                expert_bytes: None,
                prefill_chunk: 0,
                decode_first: false,
                slo_reject: false,
                ttft_slo_ns: None,
                itl_slo_ns: None,
                gpu_cfg: GpuStoreCfg::default(),
                kv_bytes: None,
                fill: GpuFill::Pinned,
                prefetch: Prefetch::Both,
                plan_window: 0,
                plan_threshold: 500,
                trace_out: None,
            }
        );
        let a = run(&["m.gguf", "--engine", "--max-seqs=2"]);
        assert_eq!(a.max_seqs, Some(2));
        let a = run(&["m.gguf", "--engine", "--expert-slots", "0"]);
        assert_eq!(a.expert_slots, Some(0));
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--expert-8gpu",
            "--expert-bytes=8192",
        ]);
        assert!(a.expert_sim);
        assert!(a.expert_8gpu);
        assert_eq!(a.expert_bytes, Some(8192));
    }

    #[test]
    fn expert_store_flags_need_engine() {
        let err = parse_serve_args(["m.gguf", "--expert-slots", "0"]).unwrap_err();
        assert!(err.contains("--expert-slots requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--expert-sim"]).unwrap_err();
        assert!(err.contains("--expert-sim requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-8gpu"]).unwrap_err();
        assert!(err.contains("--expert-8gpu requires --expert-sim"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-bytes", "4"]).unwrap_err();
        assert!(
            err.contains("--expert-bytes requires --expert-sim"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-bytes", "0"]).unwrap_err();
        assert!(err.contains("expert-bytes must be > 0"), "{err}");
    }

    #[test]
    fn serve_prefetch_planner_needs_engine() {
        let err = parse_serve_args(["m.gguf", "--prefetch", "none"]).unwrap_err();
        assert!(err.contains("--prefetch requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--plan-window", "8"]).unwrap_err();
        assert!(err.contains("--plan-window requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--plan-threshold", "100"]).unwrap_err();
        assert!(err.contains("--plan-threshold requires --engine"), "{err}");
        let a = run(&["m.gguf", "--engine", "--prefetch=none"]);
        assert_eq!(a.prefetch, Prefetch::None);
        assert!(!a.expert_sim);
        let a = run(&[
            "m.gguf",
            "--engine",
            "--prefetch=copy-forward",
            "--plan-window=8",
            "--plan-threshold=100",
        ]);
        assert_eq!(a.prefetch, Prefetch::CopyForward);
        assert_eq!(a.plan_window, 8);
        assert_eq!(a.plan_threshold, 100);
        let err = parse_serve_args(["m.gguf", "--engine", "--prefetch", "nope"]).unwrap_err();
        assert!(err.contains("unknown prefetch"), "{err}");
    }

    #[test]
    fn prefill_chunk_and_trace_out_need_engine() {
        let err = parse_serve_args(["m.gguf", "--prefill-chunk", "1"]).unwrap_err();
        assert!(err.contains("--prefill-chunk requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--trace-out", "t.jsonl"]).unwrap_err();
        assert!(err.contains("--trace-out requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--decode-first"]).unwrap_err();
        assert!(err.contains("--decode-first requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--decode-first=1"]).unwrap_err();
        assert!(
            err.contains("--decode-first does not take a value"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--prefill-chunk=1",
            "--decode-first",
            "--trace-out=t.jsonl",
        ]);
        assert_eq!(a.prefill_chunk, 1);
        assert!(a.decode_first);
        assert_eq!(a.trace_out.as_deref(), Some("t.jsonl"));
        let err = parse_serve_args(["m.gguf", "--slo-reject"]).unwrap_err();
        assert!(err.contains("--slo-reject requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--slo-reject"]).unwrap_err();
        assert!(err.contains("--slo-reject requires --ttft-slo-ns"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--slo-reject", "--ttft-slo-ns=1"])
            .unwrap_err();
        assert!(err.contains("--slo-reject requires --expert-sim"), "{err}");
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--slo-reject",
            "--ttft-slo-ns=8",
        ]);
        assert!(a.slo_reject);
        assert_eq!(a.ttft_slo_ns, Some(8));
        let err = parse_serve_args(["m.gguf", "--itl-slo-ns", "1"]).unwrap_err();
        assert!(err.contains("--itl-slo-ns requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--itl-slo-ns", "1"]).unwrap_err();
        assert!(err.contains("--itl-slo-ns requires --expert-sim"), "{err}");
        let err = parse_serve_args(["m.gguf", "--graph-update"]).unwrap_err();
        assert!(err.contains("--graph-update requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--graph-set-params"]).unwrap_err();
        assert!(
            err.contains("--graph-set-params requires --engine"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--graph-update"]).unwrap_err();
        assert!(
            err.contains("--graph-update requires --expert-sim"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--graph-build"]).unwrap_err();
        assert!(err.contains("--graph-build requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--graph-build"]).unwrap_err();
        assert!(err.contains("--graph-build requires --expert-sim"), "{err}");
        let err = parse_serve_args(["m.gguf", "--graph-build-deps"]).unwrap_err();
        assert!(
            err.contains("--graph-build-deps requires --engine"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--graph-build-deps"]).unwrap_err();
        assert!(
            err.contains("--graph-build-deps requires --expert-sim"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--graph-build-deps"])
            .unwrap_err();
        assert!(
            err.contains("--graph-build-deps needs --graph-build"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--graph-build-deps=1"])
            .unwrap_err();
        assert!(
            err.contains("--graph-build-deps does not take a value"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--graph-build",
            "--graph-build-deps",
        ]);
        assert!(a.gpu_cfg.graph_build);
        assert!(a.gpu_cfg.graph_build_deps);
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--graph-build",
            "--graph-build-deps",
            "--pdl",
        ]);
        assert!(a.gpu_cfg.graph_build_deps);
        assert!(a.gpu_cfg.pdl);
        let err = parse_serve_args(["m.gguf", "--graph-piecewise"]).unwrap_err();
        assert!(err.contains("--graph-piecewise requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--graph-piecewise"]).unwrap_err();
        assert!(
            err.contains("--graph-piecewise requires --expert-sim"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--graph-mem-trim"]).unwrap_err();
        assert!(err.contains("--graph-mem-trim requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--graph-mem-trim"]).unwrap_err();
        assert!(
            err.contains("--graph-mem-trim requires --expert-sim"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--graph-mem"]).unwrap_err();
        assert!(err.contains("--graph-mem requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--graph-mem"]).unwrap_err();
        assert!(err.contains("--graph-mem requires --expert-sim"), "{err}");
        let err = parse_serve_args(["m.gguf", "--graph-auto-free"]).unwrap_err();
        assert!(err.contains("--graph-auto-free requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--graph-auto-free"]).unwrap_err();
        assert!(
            err.contains("--graph-auto-free requires --expert-sim"),
            "{err}"
        );
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--graph-auto-free"]);
        assert!(a.gpu_cfg.graph_auto_free);
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--graph-mem",
            "--graph-auto-free",
        ])
        .unwrap_err();
        assert!(
            err.contains("choose one of --graph-mem, --graph-auto-free"),
            "{err}"
        );
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--graph-update",
            "--graph-set-params",
        ])
        .unwrap_err();
        assert!(
            err.contains("choose one of --graph-update, --graph-set-params"),
            "{err}"
        );
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--graph-build",
            "--graph-piecewise",
        ])
        .unwrap_err();
        assert!(
            err.contains("choose one of --graph-build, --graph-piecewise"),
            "{err}"
        );
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--graph-piecewise"]);
        assert!(a.gpu_cfg.graph_piecewise);
        let err = parse_serve_args(["m.gguf", "--graph-capture-deps"]).unwrap_err();
        assert!(
            err.contains("--graph-capture-deps requires --engine"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--graph-capture-deps"]).unwrap_err();
        assert!(
            err.contains("--graph-capture-deps requires --expert-sim"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--graph-capture-deps"])
            .unwrap_err();
        assert!(
            err.contains("--graph-capture-deps needs --graph-piecewise"),
            "{err}"
        );
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--graph-capture-deps=1",
        ])
        .unwrap_err();
        assert!(
            err.contains("--graph-capture-deps does not take a value"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--graph-piecewise",
            "--graph-capture-deps",
        ]);
        assert!(a.gpu_cfg.graph_piecewise);
        assert!(a.gpu_cfg.graph_capture_deps);
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--graph-piecewise",
            "--graph-capture-deps",
            "--pdl",
        ]);
        assert!(a.gpu_cfg.graph_capture_deps);
        assert!(a.gpu_cfg.pdl);
        let err = parse_serve_args(["m.gguf", "--graph-enable"]).unwrap_err();
        assert!(err.contains("--graph-enable requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--graph-enable"]).unwrap_err();
        assert!(
            err.contains("--graph-enable requires --expert-sim"),
            "{err}"
        );
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--graph-enable"]);
        assert!(a.gpu_cfg.graph_enable);
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--graph-enable",
            "--device-launch",
        ])
        .unwrap_err();
        assert!(err.contains("graph-enable cannot device-launch"), "{err}");
        let err = parse_serve_args(["m.gguf", "--launch-completion"]).unwrap_err();
        assert!(
            err.contains("--launch-completion requires --engine"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--launch-completion"]).unwrap_err();
        assert!(
            err.contains("--launch-completion requires --expert-sim"),
            "{err}"
        );
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--launch-completion"]);
        assert!(a.gpu_cfg.launch_completion);
        assert!(!a.gpu_cfg.decode_priority);
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--launch-completion=1",
        ])
        .unwrap_err();
        assert!(
            err.contains("--launch-completion does not take a value"),
            "{err}"
        );
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--launch-completion",
            "--device-launch",
        ])
        .unwrap_err();
        assert!(
            err.contains("launch-completion cannot device-launch"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--programmatic-event"]).unwrap_err();
        assert!(
            err.contains("--programmatic-event requires --engine"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--programmatic-event"]).unwrap_err();
        assert!(
            err.contains("--programmatic-event requires --expert-sim"),
            "{err}"
        );
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--programmatic-event"]);
        assert!(a.gpu_cfg.programmatic_event);
        assert!(!a.gpu_cfg.decode_priority);
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--programmatic-event=1",
        ])
        .unwrap_err();
        assert!(
            err.contains("--programmatic-event does not take a value"),
            "{err}"
        );
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--programmatic-event",
            "--device-launch",
        ])
        .unwrap_err();
        assert!(
            err.contains("programmatic-event cannot device-launch"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--stream-attach"]).unwrap_err();
        assert!(err.contains("--stream-attach requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--stream-attach"]).unwrap_err();
        assert!(
            err.contains("--stream-attach requires --expert-sim"),
            "{err}"
        );
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--stream-attach"]);
        assert!(a.gpu_cfg.stream_attach);
        assert_eq!(a.fill, GpuFill::Managed);
        assert!(!a.gpu_cfg.decode_priority);
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--stream-attach=1"])
            .unwrap_err();
        assert!(
            err.contains("--stream-attach does not take a value"),
            "{err}"
        );
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--stream-attach",
            "--seq-streams",
        ])
        .unwrap_err();
        assert!(err.contains("stream-attach cannot seq-streams"), "{err}");
        let err = parse_serve_args(["m.gguf", "--managed-host"]).unwrap_err();
        assert!(err.contains("--managed-host requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--managed-host"]).unwrap_err();
        assert!(
            err.contains("--managed-host requires --expert-sim"),
            "{err}"
        );
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--managed-host"]);
        assert!(a.gpu_cfg.managed_host);
        assert_eq!(a.fill, GpuFill::Managed);
        assert!(!a.gpu_cfg.stream_attach);
        assert!(!a.gpu_cfg.decode_priority);
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--managed-host=1"])
            .unwrap_err();
        assert!(
            err.contains("--managed-host does not take a value"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--prefetch-host"]).unwrap_err();
        assert!(err.contains("--prefetch-host requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--prefetch-host"]).unwrap_err();
        assert!(
            err.contains("--prefetch-host requires --expert-sim"),
            "{err}"
        );
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--prefetch-host"]);
        assert!(a.gpu_cfg.prefetch_host);
        assert_eq!(a.fill, GpuFill::Managed);
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--prefetch-host=1"])
            .unwrap_err();
        assert!(
            err.contains("--prefetch-host does not take a value"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--host-register-mapped"]).unwrap_err();
        assert!(
            err.contains("--host-register-mapped requires --engine"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--host-register-mapped"]).unwrap_err();
        assert!(
            err.contains("--host-register-mapped requires --expert-sim"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--host-register-mapped",
        ]);
        assert!(a.gpu_cfg.host_register_mapped);
        assert_eq!(a.fill, GpuFill::Mapped);
        assert!(!a.gpu_cfg.host_register);
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--host-register-mapped=1",
        ])
        .unwrap_err();
        assert!(
            err.contains("--host-register-mapped does not take a value"),
            "{err}"
        );
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--host-register",
            "--host-register-mapped",
        ])
        .unwrap_err();
        assert!(
            err.contains("choose one of --host-register, --host-register-mapped"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--sync-memops"]).unwrap_err();
        assert!(err.contains("--sync-memops requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--sync-memops"]).unwrap_err();
        assert!(err.contains("--sync-memops requires --expert-sim"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--sync-memops"]);
        assert!(a.gpu_cfg.sync_memops);
        assert_eq!(a.fill, GpuFill::Pinned);
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--sync-memops=1"])
            .unwrap_err();
        assert!(err.contains("--sync-memops does not take a value"), "{err}");
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--sync-memops",
            "--mapped",
        ])
        .unwrap_err();
        assert!(err.contains("sync-memops needs device memcpy"), "{err}");
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--sync-memops",
            "--memcpy-batch",
        ])
        .unwrap_err();
        assert!(
            err.contains("memcpy-batch needs async pinned/vmm H2D"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--device-sync-memops"]).unwrap_err();
        assert!(
            err.contains("--device-sync-memops requires --engine"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--device-sync-memops"]).unwrap_err();
        assert!(
            err.contains("--device-sync-memops requires --expert-sim"),
            "{err}"
        );
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--device-sync-memops"]);
        assert!(a.gpu_cfg.device_sync_memops);
        assert!(!a.gpu_cfg.sync_memops);
        assert_eq!(a.fill, GpuFill::Pinned);
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--device-sync-memops=1",
        ])
        .unwrap_err();
        assert!(
            err.contains("--device-sync-memops does not take a value"),
            "{err}"
        );
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--device-sync-memops",
            "--mapped",
        ])
        .unwrap_err();
        assert!(
            err.contains("device-sync-memops needs device memcpy"),
            "{err}"
        );
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--device-sync-memops",
            "--memcpy-batch",
        ])
        .unwrap_err();
        assert!(
            err.contains("memcpy-batch needs async pinned/vmm H2D"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--wait-value"]).unwrap_err();
        assert!(err.contains("--wait-value requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--wait-value"]).unwrap_err();
        assert!(err.contains("--wait-value requires --expert-sim"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--wait-value"]);
        assert!(a.gpu_cfg.wait_value);
        assert!(!a.gpu_cfg.decode_priority);
        let err =
            parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--wait-value=1"]).unwrap_err();
        assert!(err.contains("--wait-value does not take a value"), "{err}");
        let err = parse_serve_args(["m.gguf", "--mempool-trim"]).unwrap_err();
        assert!(err.contains("--mempool-trim requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--mempool-trim"]).unwrap_err();
        assert!(
            err.contains("--mempool-trim requires --expert-sim"),
            "{err}"
        );
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--mempool-trim"]);
        assert!(a.gpu_cfg.mempool_trim);
        assert!(a.gpu_cfg.mempool);
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--mempool-trim=1"])
            .unwrap_err();
        assert!(
            err.contains("--mempool-trim does not take a value"),
            "{err}"
        );
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--mempool-trim",
            "--sync-alloc",
        ])
        .unwrap_err();
        assert!(err.contains("mempool-trim needs cudaMallocAsync"), "{err}");
        let err = parse_serve_args(["m.gguf", "--mempool-no-reuse"]).unwrap_err();
        assert!(
            err.contains("--mempool-no-reuse requires --engine"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--mempool-no-reuse"]).unwrap_err();
        assert!(
            err.contains("--mempool-no-reuse requires --expert-sim"),
            "{err}"
        );
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--mempool-no-reuse"]);
        assert!(a.gpu_cfg.mempool_no_reuse);
        assert!(a.gpu_cfg.mempool);
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--mempool-no-reuse=1"])
            .unwrap_err();
        assert!(
            err.contains("--mempool-no-reuse does not take a value"),
            "{err}"
        );
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--mempool-no-reuse",
            "--sync-alloc",
        ])
        .unwrap_err();
        assert!(
            err.contains("mempool-no-reuse needs cudaMallocAsync"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--mempool-max", "4096"]).unwrap_err();
        assert!(err.contains("--mempool-max requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--mempool-max", "4096"]).unwrap_err();
        assert!(err.contains("--mempool-max requires --expert-sim"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--mempool-max", "0"])
            .unwrap_err();
        assert!(err.contains("mempool-max must be > 0"), "{err}");
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--mempool-max",
            "4096",
        ]);
        assert_eq!(a.gpu_cfg.mempool_max, 4096);
        assert!(a.gpu_cfg.mempool);
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--mempool-max",
            "4096",
            "--pdl",
        ]);
        assert_eq!(a.gpu_cfg.mempool_max, 4096);
        assert!(a.gpu_cfg.pdl);
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--mempool-max",
            "4096",
            "--sync-alloc",
        ])
        .unwrap_err();
        assert!(err.contains("mempool-max needs cudaMallocAsync"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--graph-set-params"]);
        assert!(a.gpu_cfg.graph_set_params);
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--cuda-graphs",
            "--graph-update",
            "--graph-clone",
            "--graph-build",
            "--graph-mem",
            "--graph-mem-trim",
            "--timing-events",
            "--itl-slo-ns=4",
        ]);
        assert!(a.gpu_cfg.graph_update);
        assert!(a.gpu_cfg.graph_clone);
        assert!(a.gpu_cfg.graph_build);
        assert!(a.gpu_cfg.graph_mem);
        assert!(a.gpu_cfg.graph_mem_trim);
        assert!(a.gpu_cfg.timing_events);
        assert_eq!(a.itl_slo_ns, Some(4));
        let err = parse_serve_args(["m.gguf", "--managed"]).unwrap_err();
        assert!(err.contains("--managed requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--mapped"]).unwrap_err();
        assert!(err.contains("--mapped requires --expert-sim"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--vmm"]);
        assert_eq!(a.fill, GpuFill::Vmm);
        let err = parse_serve_args(["m.gguf", "--host-func"]).unwrap_err();
        assert!(err.contains("--host-func requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--host-func"]).unwrap_err();
        assert!(err.contains("--host-func requires --expert-sim"), "{err}");
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--host-func",
            "--vmm-page=1024",
            "--stream-priority",
        ]);
        assert!(a.gpu_cfg.host_func);
        assert!(a.gpu_cfg.stream_priority);
        assert_eq!(a.gpu_cfg.vmm_page, 1024);
        assert_eq!(a.fill, GpuFill::Vmm);
        let err = parse_serve_args(["m.gguf", "--seq-streams"]).unwrap_err();
        assert!(err.contains("--seq-streams requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--seq-streams"]).unwrap_err();
        assert!(err.contains("--seq-streams requires --expert-sim"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--seq-streams"]);
        assert!(a.gpu_cfg.seq_streams);
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--seq-streams=1"])
            .unwrap_err();
        assert!(err.contains("--seq-streams does not take a value"), "{err}");
        let err = parse_serve_args(["m.gguf", "--kv-sim"]).unwrap_err();
        assert!(err.contains("--kv-sim requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--kv-sim"]).unwrap_err();
        assert!(err.contains("--kv-sim requires --expert-sim"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--kv-bytes", "1048576"]).unwrap_err();
        assert!(err.contains("--kv-bytes requires --kv-sim"), "{err}");
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--kv-sim",
            "--kv-bytes=1048576",
        ]);
        assert!(a.gpu_cfg.kv_sim);
        assert_eq!(a.kv_bytes, Some(1_048_576));
        let err =
            parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--kv-sim=1"]).unwrap_err();
        assert!(err.contains("--kv-sim does not take a value"), "{err}");
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--kv-sim",
            "--kv-bytes",
            "0",
        ])
        .unwrap_err();
        assert!(err.contains("kv-bytes must be > 0"), "{err}");
        let err = parse_serve_args(["m.gguf", "--decode-priority"]).unwrap_err();
        assert!(err.contains("--decode-priority requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--decode-priority"]).unwrap_err();
        assert!(
            err.contains("--decode-priority requires --expert-sim"),
            "{err}"
        );
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--decode-priority"]);
        assert!(a.gpu_cfg.decode_priority);
        assert!(a.gpu_cfg.stream_priority);
        let err = parse_serve_args(["m.gguf", "--cooperative"]).unwrap_err();
        assert!(err.contains("--cooperative requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--cooperative"]).unwrap_err();
        assert!(err.contains("--cooperative requires --expert-sim"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--cooperative"]);
        assert!(a.gpu_cfg.cooperative);
        let err = parse_serve_args(["m.gguf", "--pdl"]).unwrap_err();
        assert!(err.contains("--pdl requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--pdl"]).unwrap_err();
        assert!(err.contains("--pdl requires --expert-sim"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--pdl"]);
        assert!(a.gpu_cfg.pdl);
        let err = parse_serve_args(["m.gguf", "--l2-persist"]).unwrap_err();
        assert!(err.contains("--l2-persist requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--l2-persist"]).unwrap_err();
        assert!(err.contains("--l2-persist requires --expert-sim"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--l2-persist"]);
        assert!(a.gpu_cfg.l2_persist);
        let err = parse_serve_args(["m.gguf", "--l2-reset"]).unwrap_err();
        assert!(err.contains("--l2-reset requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--l2-reset"]).unwrap_err();
        assert!(err.contains("--l2-reset requires --expert-sim"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--l2-reset"]);
        assert!(a.gpu_cfg.l2_reset);
        assert!(a.gpu_cfg.l2_persist);
        let err =
            parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--l2-reset=1"]).unwrap_err();
        assert!(err.contains("--l2-reset does not take a value"), "{err}");
        let err = parse_serve_args(["m.gguf", "--l2-fetch", "32"]).unwrap_err();
        assert!(err.contains("--l2-fetch requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--l2-fetch", "32"]).unwrap_err();
        assert!(err.contains("--l2-fetch requires --expert-sim"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--l2-fetch", "96"])
            .unwrap_err();
        assert!(err.contains("l2-fetch must be 32, 64, or 128"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--l2-fetch", "32"]);
        assert_eq!(a.gpu_cfg.l2_fetch, 32);
        assert!(a.gpu_cfg.l2_persist);
        let err = parse_serve_args(["m.gguf", "--l2-ratio", "500"]).unwrap_err();
        assert!(err.contains("--l2-ratio requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--l2-ratio", "500"]).unwrap_err();
        assert!(err.contains("--l2-ratio requires --expert-sim"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--l2-ratio", "1001"])
            .unwrap_err();
        assert!(err.contains("l2-ratio must be 1..=1000"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--l2-ratio", "500"]);
        assert_eq!(a.gpu_cfg.l2_ratio, 500);
        assert!(a.gpu_cfg.l2_persist);
        let err = parse_serve_args(["m.gguf", "--l2-streaming"]).unwrap_err();
        assert!(err.contains("--l2-streaming requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--l2-streaming"]).unwrap_err();
        assert!(
            err.contains("--l2-streaming requires --expert-sim"),
            "{err}"
        );
        let err =
            parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--l2-streaming"]).unwrap_err();
        assert!(err.contains("--l2-streaming needs --l2-persist"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--l2-streaming=1"])
            .unwrap_err();
        assert!(
            err.contains("--l2-streaming does not take a value"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--l2-persist",
            "--l2-streaming",
        ]);
        assert!(a.gpu_cfg.l2_persist);
        assert!(a.gpu_cfg.l2_streaming);
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--l2-persist",
            "--l2-streaming",
            "--pdl",
        ]);
        assert!(a.gpu_cfg.l2_streaming);
        assert!(a.gpu_cfg.pdl);
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--l2-ratio",
            "500",
            "--l2-streaming",
        ]);
        assert_eq!(a.gpu_cfg.l2_ratio, 500);
        assert!(a.gpu_cfg.l2_streaming);
        assert!(a.gpu_cfg.l2_persist);
        let err = parse_serve_args(["m.gguf", "--cluster", "2"]).unwrap_err();
        assert!(err.contains("--cluster requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--cluster", "2"]).unwrap_err();
        assert!(err.contains("--cluster requires --expert-sim"), "{err}");
        let err =
            parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--cluster", "0"]).unwrap_err();
        assert!(err.contains("cluster must be > 0"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--cluster", "2"]);
        assert_eq!(a.gpu_cfg.cluster, 2);
        let err = parse_serve_args(["m.gguf", "--preferred-cluster", "4"]).unwrap_err();
        assert!(
            err.contains("--preferred-cluster requires --engine"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--preferred-cluster", "4"]).unwrap_err();
        assert!(
            err.contains("--preferred-cluster requires --expert-sim"),
            "{err}"
        );
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--preferred-cluster",
            "0",
        ])
        .unwrap_err();
        assert!(err.contains("preferred-cluster must be > 0"), "{err}");
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--preferred-cluster",
            "4",
        ])
        .unwrap_err();
        assert!(err.contains("--preferred-cluster needs --cluster"), "{err}");
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--cluster",
            "2",
            "--preferred-cluster",
            "3",
        ])
        .unwrap_err();
        assert!(
            err.contains("preferred-cluster must be a multiple of cluster"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--cluster",
            "2",
            "--preferred-cluster",
            "4",
        ]);
        assert_eq!(a.gpu_cfg.cluster, 2);
        assert_eq!(a.gpu_cfg.preferred_cluster, 4);
        let err = parse_serve_args(["m.gguf", "--cluster-spread"]).unwrap_err();
        assert!(err.contains("--cluster-spread requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--cluster-spread"]).unwrap_err();
        assert!(
            err.contains("--cluster-spread requires --expert-sim"),
            "{err}"
        );
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--cluster-spread"]);
        assert!(a.gpu_cfg.cluster_spread);
        let err = parse_serve_args(["m.gguf", "--func-cluster-spread"]).unwrap_err();
        assert!(
            err.contains("--func-cluster-spread requires --engine"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--func-cluster-spread"]).unwrap_err();
        assert!(
            err.contains("--func-cluster-spread requires --expert-sim"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--func-cluster-spread",
        ]);
        assert!(a.gpu_cfg.func_cluster_spread);
        let err = parse_serve_args(["m.gguf", "--cluster-load-balance"]).unwrap_err();
        assert!(
            err.contains("--cluster-load-balance requires --engine"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--cluster-load-balance"]).unwrap_err();
        assert!(
            err.contains("--cluster-load-balance requires --expert-sim"),
            "{err}"
        );
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--cluster-load-balance",
        ])
        .unwrap_err();
        assert!(
            err.contains("--cluster-load-balance needs --func-cluster-spread"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--func-cluster-spread",
            "--cluster-load-balance",
        ]);
        assert!(a.gpu_cfg.cluster_load_balance);
        let err = parse_serve_args(["m.gguf", "--cluster-must-set"]).unwrap_err();
        assert!(
            err.contains("--cluster-must-set requires --engine"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--cluster-must-set"]).unwrap_err();
        assert!(
            err.contains("--cluster-must-set requires --expert-sim"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--cluster-must-set"])
            .unwrap_err();
        assert!(err.contains("--cluster-must-set needs --cluster"), "{err}");
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--cluster",
            "2",
            "--cluster-must-set",
        ]);
        assert_eq!(a.gpu_cfg.cluster, 2);
        assert!(a.gpu_cfg.cluster_must_set);
        let err = parse_serve_args(["m.gguf", "--required-cluster", "2"]).unwrap_err();
        assert!(
            err.contains("--required-cluster requires --engine"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--required-cluster", "2"]).unwrap_err();
        assert!(
            err.contains("--required-cluster requires --expert-sim"),
            "{err}"
        );
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--required-cluster",
            "0",
        ])
        .unwrap_err();
        assert!(err.contains("required-cluster must be > 0"), "{err}");
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--required-cluster",
            "2",
        ])
        .unwrap_err();
        assert!(err.contains("--required-cluster needs --cluster"), "{err}");
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--cluster",
            "2",
            "--required-cluster",
            "4",
        ])
        .unwrap_err();
        assert!(
            err.contains("required-cluster must match --cluster"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--cluster",
            "2",
            "--required-cluster",
            "2",
        ]);
        assert_eq!(a.gpu_cfg.cluster, 2);
        assert_eq!(a.gpu_cfg.required_cluster, 2);
        let err = parse_serve_args(["m.gguf", "--event-blocking-sync"]).unwrap_err();
        assert!(
            err.contains("--event-blocking-sync requires --engine"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--event-blocking-sync"]).unwrap_err();
        assert!(
            err.contains("--event-blocking-sync requires --expert-sim"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--event-blocking-sync",
        ]);
        assert!(a.gpu_cfg.event_blocking_sync);
        assert!(a.gpu_cfg.timing_events);
        let err = parse_serve_args(["m.gguf", "--max-shared"]).unwrap_err();
        assert!(err.contains("--max-shared requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--max-shared"]).unwrap_err();
        assert!(err.contains("--max-shared requires --expert-sim"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--max-shared"]);
        assert!(a.gpu_cfg.max_shared);
        let err = parse_serve_args(["m.gguf", "--func-max-shared"]).unwrap_err();
        assert!(err.contains("--func-max-shared requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--func-max-shared"]).unwrap_err();
        assert!(
            err.contains("--func-max-shared requires --expert-sim"),
            "{err}"
        );
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--func-max-shared"]);
        assert!(a.gpu_cfg.func_max_shared);
        let err = parse_serve_args(["m.gguf", "--max-l1"]).unwrap_err();
        assert!(err.contains("--max-l1 requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--max-l1"]).unwrap_err();
        assert!(err.contains("--max-l1 requires --expert-sim"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--max-l1"]).unwrap_err();
        assert!(err.contains("--max-l1 needs --func-max-shared"), "{err}");
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--func-max-shared",
            "--max-l1",
        ]);
        assert!(a.gpu_cfg.max_l1);
        let err = parse_serve_args(["m.gguf", "--non-portable-cluster"]).unwrap_err();
        assert!(
            err.contains("--non-portable-cluster requires --engine"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--non-portable-cluster"]).unwrap_err();
        assert!(
            err.contains("--non-portable-cluster requires --expert-sim"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--non-portable-cluster",
        ]);
        assert!(a.gpu_cfg.non_portable_cluster);
        let err = parse_serve_args(["m.gguf", "--sync-policy", "blocking"]).unwrap_err();
        assert!(err.contains("--sync-policy requires --engine"), "{err}");
        let err =
            parse_serve_args(["m.gguf", "--engine", "--sync-policy", "blocking"]).unwrap_err();
        assert!(err.contains("--sync-policy requires --expert-sim"), "{err}");
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--sync-policy",
            "blocking",
        ]);
        assert_eq!(a.gpu_cfg.sync_policy, SynchronizationPolicy::BlockingSync);
        let err = parse_serve_args(["m.gguf", "--device-sync-policy", "blocking"]).unwrap_err();
        assert!(
            err.contains("--device-sync-policy requires --engine"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--device-sync-policy", "blocking"])
            .unwrap_err();
        assert!(
            err.contains("--device-sync-policy requires --expert-sim"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--device-sync-policy",
            "blocking",
        ]);
        assert_eq!(
            a.gpu_cfg.device_sync_policy,
            SynchronizationPolicy::BlockingSync
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--device-sync-policy",
            "spin",
            "--pdl",
        ]);
        assert_eq!(a.gpu_cfg.device_sync_policy, SynchronizationPolicy::Spin);
        assert!(a.gpu_cfg.pdl);
        let err = parse_serve_args(["m.gguf", "--mem-sync-domain", "remote"]).unwrap_err();
        assert!(err.contains("--mem-sync-domain requires --engine"), "{err}");
        let err =
            parse_serve_args(["m.gguf", "--engine", "--mem-sync-domain", "remote"]).unwrap_err();
        assert!(
            err.contains("--mem-sync-domain requires --expert-sim"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--mem-sync-domain",
            "remote",
        ]);
        assert_eq!(a.gpu_cfg.mem_sync_domain, MemSyncDomain::Remote);
        assert!(a.gpu_cfg.decode_priority);
        assert!(a.gpu_cfg.stream_priority);
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--mem-sync-domain",
            "bogus",
        ])
        .unwrap_err();
        assert!(err.contains("unknown mem-sync-domain"), "{err}");
        let err = parse_serve_args(["m.gguf", "--mem-sync-map", "collapse"]).unwrap_err();
        assert!(err.contains("--mem-sync-map requires --engine"), "{err}");
        let err =
            parse_serve_args(["m.gguf", "--engine", "--mem-sync-map", "collapse"]).unwrap_err();
        assert!(
            err.contains("--mem-sync-map requires --expert-sim"),
            "{err}"
        );
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--mem-sync-map",
            "collapse",
        ])
        .unwrap_err();
        assert!(
            err.contains("--mem-sync-map collapse needs --mem-sync-domain remote"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--mem-sync-domain",
            "remote",
            "--mem-sync-map",
            "collapse",
        ]);
        assert!(a.gpu_cfg.mem_sync_collapse);
        assert_eq!(a.gpu_cfg.mem_sync_domain, MemSyncDomain::Remote);
        let err = parse_serve_args(["m.gguf", "--mem-sync-launch"]).unwrap_err();
        assert!(err.contains("--mem-sync-launch requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--mem-sync-launch"]).unwrap_err();
        assert!(
            err.contains("--mem-sync-launch requires --expert-sim"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--mem-sync-launch"])
            .unwrap_err();
        assert!(
            err.contains("--mem-sync-launch needs --mem-sync-domain remote"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--mem-sync-domain",
            "remote",
            "--mem-sync-launch",
        ]);
        assert!(a.gpu_cfg.mem_sync_launch);
        assert_eq!(a.gpu_cfg.mem_sync_domain, MemSyncDomain::Remote);
        let err = parse_serve_args(["m.gguf", "--mem-sync-launch-map"]).unwrap_err();
        assert!(
            err.contains("--mem-sync-launch-map requires --engine"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--mem-sync-launch-map"]).unwrap_err();
        assert!(
            err.contains("--mem-sync-launch-map requires --expert-sim"),
            "{err}"
        );
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--mem-sync-launch-map",
        ])
        .unwrap_err();
        assert!(
            err.contains("--mem-sync-launch-map needs --mem-sync-domain remote"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--mem-sync-domain",
            "remote",
            "--mem-sync-launch-map",
        ]);
        assert!(a.gpu_cfg.mem_sync_launch_map);
        assert_eq!(a.gpu_cfg.mem_sync_domain, MemSyncDomain::Remote);
        let err = parse_serve_args(["m.gguf", "--shared-mem", "eight"]).unwrap_err();
        assert!(err.contains("--shared-mem requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--shared-mem", "eight"]).unwrap_err();
        assert!(err.contains("--shared-mem requires --expert-sim"), "{err}");
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--shared-mem",
            "eight",
        ]);
        assert_eq!(a.gpu_cfg.shared_mem, SharedMemoryMode::EightByte);
        let err = parse_serve_args(["m.gguf", "--func-shared-mem", "eight"]).unwrap_err();
        assert!(err.contains("--func-shared-mem requires --engine"), "{err}");
        let err =
            parse_serve_args(["m.gguf", "--engine", "--func-shared-mem", "eight"]).unwrap_err();
        assert!(
            err.contains("--func-shared-mem requires --expert-sim"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--func-shared-mem",
            "eight",
        ]);
        assert_eq!(a.gpu_cfg.func_shared_mem, SharedMemoryMode::EightByte);
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--func-shared-mem",
            "bogus",
        ])
        .unwrap_err();
        assert!(err.contains("unknown func-shared-mem"), "{err}");
        let err = parse_serve_args(["m.gguf", "--device-shared-mem", "eight"]).unwrap_err();
        assert!(
            err.contains("--device-shared-mem requires --engine"),
            "{err}"
        );
        let err =
            parse_serve_args(["m.gguf", "--engine", "--device-shared-mem", "eight"]).unwrap_err();
        assert!(
            err.contains("--device-shared-mem requires --expert-sim"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--device-shared-mem",
            "eight",
        ]);
        assert_eq!(a.gpu_cfg.device_shared_mem, SharedMemoryMode::EightByte);
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--device-shared-mem",
            "bogus",
        ])
        .unwrap_err();
        assert!(err.contains("unknown device-shared-mem"), "{err}");
        let err = parse_serve_args(["m.gguf", "--portable-cluster", "non-portable"]).unwrap_err();
        assert!(
            err.contains("--portable-cluster requires --engine"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--portable-cluster", "non-portable"])
            .unwrap_err();
        assert!(
            err.contains("--portable-cluster requires --expert-sim"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--portable-cluster",
            "non-portable",
        ]);
        assert_eq!(
            a.gpu_cfg.portable_cluster,
            PortableClusterMode::AllowNonPortable
        );
        let err = parse_serve_args(["m.gguf", "--optin-shared"]).unwrap_err();
        assert!(err.contains("--optin-shared requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--optin-shared"]).unwrap_err();
        assert!(
            err.contains("--optin-shared requires --expert-sim"),
            "{err}"
        );
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--optin-shared"]);
        assert!(a.gpu_cfg.optin_shared);
        let err = parse_serve_args(["m.gguf", "--dynamic-shared", "65536"]).unwrap_err();
        assert!(err.contains("--dynamic-shared requires --engine"), "{err}");
        let err =
            parse_serve_args(["m.gguf", "--engine", "--dynamic-shared", "65536"]).unwrap_err();
        assert!(
            err.contains("--dynamic-shared requires --expert-sim"),
            "{err}"
        );
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--dynamic-shared",
            "0",
        ])
        .unwrap_err();
        assert!(err.contains("dynamic-shared must be > 0"), "{err}");
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--dynamic-shared",
            "65536",
        ]);
        assert_eq!(a.gpu_cfg.dynamic_shared, 65_536);
        let err = parse_serve_args(["m.gguf", "--portable-shared", "non-portable"]).unwrap_err();
        assert!(err.contains("--portable-shared requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--portable-shared", "non-portable"])
            .unwrap_err();
        assert!(
            err.contains("--portable-shared requires --expert-sim"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--portable-shared",
            "non-portable",
        ]);
        assert_eq!(
            a.gpu_cfg.portable_shared,
            PortableSharedMode::AllowNonPortable
        );
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--portable-shared",
            "bogus",
        ])
        .unwrap_err();
        assert!(err.contains("unknown portable-shared"), "{err}");
        let err = parse_serve_args(["m.gguf", "--nvlink-util"]).unwrap_err();
        assert!(err.contains("--nvlink-util requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--nvlink-util"]).unwrap_err();
        assert!(err.contains("--nvlink-util requires --expert-sim"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--nvlink-util"]);
        assert!(a.gpu_cfg.nvlink_util_centric);
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--nvlink-util",
            "--pdl",
        ]);
        assert!(a.gpu_cfg.nvlink_util_centric);
        assert!(a.gpu_cfg.pdl);
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--nvlink-util=1"])
            .unwrap_err();
        assert!(err.contains("--nvlink-util does not take a value"), "{err}");
        let err = parse_serve_args(["m.gguf", "--device-launch"]).unwrap_err();
        assert!(err.contains("--device-launch requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--device-launch"]).unwrap_err();
        assert!(
            err.contains("--device-launch requires --expert-sim"),
            "{err}"
        );
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--device-launch"]);
        assert!(a.gpu_cfg.device_launch);
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--device-updatable",
            "--pdl",
        ]);
        assert!(a.gpu_cfg.device_updatable);
        assert!(a.gpu_cfg.pdl);
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--device-launch=1"])
            .unwrap_err();
        assert!(
            err.contains("--device-launch does not take a value"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--kernel-priority", "5"]).unwrap_err();
        assert!(err.contains("--kernel-priority requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--kernel-priority", "5"]).unwrap_err();
        assert!(
            err.contains("--kernel-priority requires --expert-sim"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--kernel-priority",
            "0",
        ]);
        assert_eq!(a.gpu_cfg.kernel_priority, Some(0));
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--kernel-priority",
            "-5",
            "--pdl",
        ]);
        assert_eq!(a.gpu_cfg.kernel_priority, Some(-5));
        assert!(a.gpu_cfg.pdl);
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--pdl",
            "--cooperative",
        ])
        .unwrap_err();
        assert!(err.contains("choose one of --pdl, --cooperative"), "{err}");
        let err = parse_serve_args(["m.gguf", "--multicast"]).unwrap_err();
        assert!(err.contains("--multicast requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--multicast"]).unwrap_err();
        assert!(err.contains("--multicast requires --expert-sim"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--multicast"]);
        assert!(a.gpu_cfg.multicast);
        assert!(a.fill == GpuFill::Vmm);
        let err = parse_serve_args(["m.gguf", "--shareable"]).unwrap_err();
        assert!(err.contains("--shareable requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--shareable"]).unwrap_err();
        assert!(err.contains("--shareable requires --expert-sim"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--shareable"]);
        assert!(a.gpu_cfg.shareable);
        assert!(a.gpu_cfg.mempool);
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--shareable",
            "--sync-alloc",
        ])
        .unwrap_err();
        assert!(err.contains("shareable needs cudaMallocAsync"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--memcpy-batch"]);
        assert!(a.gpu_cfg.memcpy_batch);
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--memcpy-batch",
            "--pageable",
        ])
        .unwrap_err();
        assert!(
            err.contains("memcpy-batch needs async pinned/vmm H2D"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--memcpy-during"]).unwrap_err();
        assert!(err.contains("--memcpy-during requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--memcpy-during"]).unwrap_err();
        assert!(
            err.contains("--memcpy-during requires --expert-sim"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--memcpy-during"])
            .unwrap_err();
        assert!(
            err.contains("--memcpy-during needs --memcpy-batch"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--memcpy-during=1"])
            .unwrap_err();
        assert!(
            err.contains("--memcpy-during does not take a value"),
            "{err}"
        );
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--memcpy-batch",
            "--memcpy-during",
        ]);
        assert!(a.gpu_cfg.memcpy_batch);
        assert!(a.gpu_cfg.memcpy_during);
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--memcpy-batch",
            "--memcpy-during",
            "--pdl",
        ]);
        assert!(a.gpu_cfg.memcpy_during);
        assert!(a.gpu_cfg.pdl);
        let err = parse_serve_args(["m.gguf", "--memcpy-any"]).unwrap_err();
        assert!(err.contains("--memcpy-any requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--memcpy-any"]).unwrap_err();
        assert!(err.contains("--memcpy-any requires --expert-sim"), "{err}");
        let err =
            parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--memcpy-any"]).unwrap_err();
        assert!(err.contains("--memcpy-any needs --memcpy-batch"), "{err}");
        let err =
            parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--memcpy-any=1"]).unwrap_err();
        assert!(err.contains("--memcpy-any does not take a value"), "{err}");
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--memcpy-batch",
            "--memcpy-any",
        ]);
        assert!(a.gpu_cfg.memcpy_batch);
        assert!(a.gpu_cfg.memcpy_any);
        let a = run(&[
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--memcpy-batch",
            "--memcpy-any",
            "--pdl",
        ]);
        assert!(a.gpu_cfg.memcpy_any);
        assert!(a.gpu_cfg.pdl);
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--memcpy-batch",
            "--memcpy-during",
            "--memcpy-any",
        ])
        .unwrap_err();
        assert!(
            err.contains("choose one of --memcpy-any, --memcpy-during"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--host-register"]).unwrap_err();
        assert!(err.contains("--host-register requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--host-register"]).unwrap_err();
        assert!(
            err.contains("--host-register requires --expert-sim"),
            "{err}"
        );
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--host-register"]);
        assert!(a.gpu_cfg.host_register);
        assert!(a.gpu_cfg.pageable);
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--host-register=1"])
            .unwrap_err();
        assert!(
            err.contains("--host-register does not take a value"),
            "{err}"
        );
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--memcpy-batch",
            "--host-register",
        ])
        .unwrap_err();
        assert!(
            err.contains("memcpy-batch needs async pinned/vmm H2D"),
            "{err}"
        );
        let err = parse_serve_args([
            "m.gguf",
            "--engine",
            "--expert-sim",
            "--host-register",
            "--managed",
        ])
        .unwrap_err();
        assert!(err.contains("host-register needs pinned/vmm H2D"), "{err}");
        let err = parse_serve_args(["m.gguf", "--compute-slots", "2"]).unwrap_err();
        assert!(err.contains("--compute-slots requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--compute-slots", "2"]).unwrap_err();
        assert!(
            err.contains("--compute-slots requires --expert-sim"),
            "{err}"
        );
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--compute-slots", "0"])
            .unwrap_err();
        assert!(err.contains("compute-slots must be > 0"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--compute-slots", "2"]);
        assert_eq!(a.gpu_cfg.compute_slots, 2);
        let err = parse_serve_args(["m.gguf", "--decode-sms", "250"]).unwrap_err();
        assert!(err.contains("--decode-sms requires --engine"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--decode-sms", "250"]).unwrap_err();
        assert!(err.contains("--decode-sms requires --expert-sim"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--decode-sms", "0"])
            .unwrap_err();
        assert!(err.contains("decode-sms must be 1..=1000"), "{err}");
        let err = parse_serve_args(["m.gguf", "--engine", "--expert-sim", "--decode-sms", "1001"])
            .unwrap_err();
        assert!(err.contains("decode-sms must be 1..=1000"), "{err}");
        let a = run(&["m.gguf", "--engine", "--expert-sim", "--decode-sms", "250"]);
        assert_eq!(a.gpu_cfg.decode_sm_permille, 250);
        assert!(a.gpu_cfg.decode_priority);
        assert!(a.gpu_cfg.stream_priority);
    }

    #[test]
    fn bind_parser_allows_loopback_only() {
        assert_eq!(
            parse_bind("127.0.0.1:8080").unwrap(),
            (Ipv4Addr::LOCALHOST, 8080)
        );
        assert_eq!(parse_bind("localhost:0").unwrap(), (Ipv4Addr::LOCALHOST, 0));
        assert_eq!(parse_bind("LOCALHOST:9").unwrap(), (Ipv4Addr::LOCALHOST, 9));
        for bad in [
            "0.0.0.0:8080",
            "1.2.3.4:80",
            "[::]:80",
            "::1:80",
            "*:80",
            "8080",
            ":8080",
        ] {
            let err = parse_bind(bad).unwrap_err();
            assert!(
                err.contains("localhost") || err.contains("HOST:PORT") || err.contains("host"),
                "{bad}: {err}"
            );
        }
    }

    #[test]
    fn bind_loopback_ephemeral_is_127_0_0_1() {
        let listener = bind_loopback("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        assert_eq!(addr.ip(), IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert_ne!(addr.port(), 0);
        let listener2 = bind_loopback("localhost:0").expect("localhost");
        assert_eq!(
            listener2.local_addr().expect("addr2").ip(),
            IpAddr::V4(Ipv4Addr::LOCALHOST)
        );
        let err = bind_loopback("0.0.0.0:8080").expect_err("public");
        assert!(err.to_string().contains("localhost"), "{err}");
    }

    #[test]
    fn missing_gguf_file_fails_before_listen() {
        let err = read_gguf_path("no-such-llama-rust-serve-test.gguf").expect_err("missing");
        assert!(err.to_string().contains("missing GGUF file"), "{err}");
        assert!(
            err.to_string()
                .contains("no-such-llama-rust-serve-test.gguf"),
            "{err}"
        );
    }

    #[test]
    fn http_request_shape_content_length() {
        let raw = b"POST /generate HTTP/1.1\r\nContent-Length: 15\r\n\r\n{\"prompt\":\"ab\"}";
        let req = try_parse_http_request(raw).unwrap().unwrap();
        assert_eq!(req.method, "POST");
        assert_eq!(req.path, "/generate");
        assert_eq!(req.body, br#"{"prompt":"ab"}"#);
        assert!(req.keep_alive);
        assert_eq!(req.consumed, raw.len());
        let closed = b"POST /generate HTTP/1.0\r\nContent-Length: 2\r\n\r\n{}";
        let req0 = try_parse_http_request(closed).unwrap().unwrap();
        assert!(!req0.keep_alive);
        let close_h =
            b"POST /generate HTTP/1.1\r\nConnection: close\r\nContent-Length: 2\r\n\r\n{}";
        let reqc = try_parse_http_request(close_h).unwrap().unwrap();
        assert!(!reqc.keep_alive);
        assert!(try_parse_http_request(b"POST /generate HTTP/1.1\r\n")
            .unwrap()
            .is_none());
        assert!(try_parse_http_request(
            b"POST /generate HTTP/1.1\r\nContent-Length: 10\r\n\r\n123"
        )
        .unwrap()
        .is_none());
        let extra = b"POST /generate HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}\x00leftover";
        let req = try_parse_http_request(extra).unwrap().unwrap();
        assert_eq!(req.body, b"{}");
        assert_eq!(req.consumed, extra.len().saturating_sub(9));
        let err = try_parse_http_request(
            b"POST /generate HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n",
        )
        .unwrap_err();
        assert!(err.to_string().contains("chunked"), "{err}");
        let err =
            try_parse_http_request(b"POST /generate HTTP/1.1\r\nContent-Length: 999999\r\n\r\n")
                .unwrap_err();
        assert!(err.to_string().contains("too large"), "{err}");
    }

    #[test]
    fn json_request_shape() {
        let (_model, tok) = tiny_model();
        let a = parse_gen_req(r#"{"prompt":"ab"}"#).unwrap();
        assert_eq!(a.prompt.as_deref(), Some("ab"));
        assert_eq!(a.n_predict, None);
        assert!(!a.stream);
        assert!(!a.echo);
        assert!(a.add_special_tokens);
        assert_eq!(a.resolve(&tok).unwrap(), "ab");
        let streamed = parse_gen_req(r#"{"prompt":"ab","stream":true}"#).unwrap();
        assert!(streamed.stream);
        let echoed = parse_gen_req(r#"{"prompt":"ab","echo":true}"#).unwrap();
        assert!(echoed.echo);
        let no_special = parse_gen_req(r#"{"prompt":"ab","add_special_tokens":false}"#).unwrap();
        assert!(!no_special.add_special_tokens);
        let unstreamed = parse_gen_req(r#"{"prompt":"ab","stream":false}"#).unwrap();
        assert!(!unstreamed.stream);
        let b = parse_gen_req(r#"{ "prompt" : "ab", "n_predict" : 4 }"#).unwrap();
        assert_eq!(b.prompt.as_deref(), Some("ab"));
        assert_eq!(b.n_predict, Some(4));
        let mt = parse_gen_req(r#"{"prompt":"ab","max_tokens":4}"#).unwrap();
        assert_eq!(mt.n_predict, Some(4));
        let both = parse_gen_req(r#"{"prompt":"ab","n_predict":4,"max_tokens":4}"#).unwrap();
        assert_eq!(both.n_predict, Some(4));
        assert!(
            parse_gen_req(r#"{"prompt":"ab","n_predict":2,"max_tokens":4}"#)
                .unwrap_err()
                .contains("disagree")
        );
        assert!(parse_gen_req(r#"{"prompt":"ab","n":1}"#).is_ok());
        assert!(parse_gen_req(r#"{"prompt":"ab","n":2}"#)
            .unwrap_err()
            .contains("n must be 1"));
        let c = parse_gen_req(r#"{"prompt":"a\"b","extra":true}"#).unwrap();
        assert_eq!(c.prompt.as_deref(), Some("a\"b"));
        assert_eq!(
            parse_gen_req("{}").unwrap().resolve(&tok).unwrap_err(),
            "missing prompt"
        );
        assert_eq!(
            parse_gen_req(r#"{"prompt":""}"#)
                .unwrap()
                .resolve(&tok)
                .unwrap(),
            ""
        );
        assert!(parse_gen_req(r#"{"n_predict":2}"#)
            .unwrap()
            .resolve(&tok)
            .unwrap_err()
            .contains("missing prompt"));
        assert!(parse_gen_req(r#"{"prompt":"ab","n_predict":-1}"#)
            .unwrap_err()
            .contains("n_predict"));
        assert!(parse_gen_req(r#"{"prompt":"ab"} extra"#)
            .unwrap_err()
            .contains("trailing"));
        assert_eq!(parse_detok_req(r#"{"tokens":[1,2]}"#).unwrap(), vec![1, 2]);
        assert_eq!(parse_detok_req(r#"{ "tokens" : [ 0 ] }"#).unwrap(), vec![0]);
        assert!(parse_detok_req("{}")
            .unwrap_err()
            .contains("missing tokens"));
        assert!(parse_detok_req(r#"{"tokens":[]}"#).unwrap().is_empty());
        assert!(parse_detok_req(r#"{"tokens":[-1]}"#)
            .unwrap_err()
            .contains("tokens"));
        assert!(parse_detok_req(r#"{"tokens":[1,]}"#)
            .unwrap_err()
            .contains("trailing comma"));
    }

    /// The tiny fixture has no `chat_template` of its own; give it one whose
    /// output lands inside its three-token vocab.
    fn chat_model(template: &str) -> (Llama, Tokenizer) {
        let (model, mut tok) = tiny_model();
        tok.chat_template = Some(template.to_string());
        (model, tok)
    }

    const ECHO_TEMPLATE: &str = "{% for m in messages %}{{ m['content'] }}{% endfor %}";

    #[test]
    fn a_messages_array_renders_the_template_and_matches_the_same_prompt() {
        let (model, tok) = chat_model(ECHO_TEMPLATE);
        let args = defaults();
        let expect = greedy_generate_ctx(&model, &tok, "ab", 2, None).expect("greedy");
        let (head, body) = exchange(
            &post_json(r#"{"messages":[{"role":"user","content":"ab"}]}"#),
            &model,
            &tok,
            &args,
        );
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(json_field_string(&body, "generated").unwrap(), expect);
        // Multi-turn, with the keys in either order and an ignored extra field.
        let (_h, body) = exchange(
            &post_json(
                r#"{"messages":[{"content":"a","role":"user"},{"role":"assistant","name":"x","content":"b"}],"n_predict":2}"#,
            ),
            &model,
            &tok,
            &args,
        );
        assert_eq!(json_field_string(&body, "generated").unwrap(), expect);
    }

    #[test]
    fn add_generation_prompt_defaults_to_true_and_can_be_turned_off() {
        let (model, tok) = chat_model(
            "{% for m in messages %}{{ m.content }}{% endfor %}\
             {% if add_generation_prompt %}b{% endif %}",
        );
        let args = defaults();
        let with = greedy_generate_ctx(&model, &tok, "ab", 2, None).expect("greedy ab");
        let without = greedy_generate_ctx(&model, &tok, "a", 2, None).expect("greedy a");
        let (_h, body) = exchange(
            &post_json(r#"{"messages":[{"role":"user","content":"a"}]}"#),
            &model,
            &tok,
            &args,
        );
        assert_eq!(json_field_string(&body, "generated").unwrap(), with);
        let (_h, body) = exchange(
            &post_json(
                r#"{"messages":[{"role":"user","content":"a"}],"add_generation_prompt":false}"#,
            ),
            &model,
            &tok,
            &args,
        );
        assert_eq!(json_field_string(&body, "generated").unwrap(), without);
    }

    #[test]
    fn chat_requests_that_cannot_be_served_are_bad_requests_not_guesses() {
        let (model, tok) = chat_model(ECHO_TEMPLATE);
        let args = defaults();
        for (json, want) in [
            (
                r#"{"prompt":"ab","messages":[{"role":"user","content":"ab"}]}"#,
                "not both",
            ),
            (r#"{"messages":[]}"#, "messages is empty"),
            (r#"{"messages":[{"role":"user"}]}"#, "missing content"),
            (r#"{"messages":[{"content":"ab"}]}"#, "missing role"),
            (r#"{"messages":"ab"}"#, "expected '['"),
            (
                r#"{"messages":[{"role":"user","content":"ab"},]}"#,
                "trailing comma",
            ),
        ] {
            let (head, body) = exchange(&post_json(json), &model, &tok, &args);
            assert!(head.starts_with("HTTP/1.1 400"), "{json}: {head}");
            let msg = json_field_string(&body, "error").expect("error");
            assert!(msg.contains(want), "{json}: got {msg:?}, want {want:?}");
        }
        // A model with no template at all says so instead of inventing one.
        let (model, tok) = tiny_model();
        let (head, body) = exchange(
            &post_json(r#"{"messages":[{"role":"user","content":"ab"}]}"#),
            &model,
            &tok,
            &args,
        );
        assert!(head.starts_with("HTTP/1.1 400"), "{head}");
        assert!(json_field_string(&body, "error")
            .expect("error")
            .contains("chat_template"));
    }

    #[test]
    fn the_plain_prompt_field_is_untouched_by_the_chat_path() {
        let (model, tok) = chat_model(ECHO_TEMPLATE);
        let args = defaults();
        let expect = greedy_generate_ctx(&model, &tok, "ab", 2, None).expect("greedy");
        let (head, body) = exchange(&post_json(r#"{"prompt":"ab"}"#), &model, &tok, &args);
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(json_field_string(&body, "generated").unwrap(), expect);
    }

    #[test]
    fn json_response_shape_escapes() {
        assert_eq!(
            json_generated("ab", 0, 0),
            r#"{"generated":"ab","prefix_hit":0,"page_hits":0}"#
        );
        assert_eq!(
            json_generated("a\"b", 0, 0),
            r#"{"generated":"a\"b","prefix_hit":0,"page_hits":0}"#
        );
        assert_eq!(
            json_generated("a\nb", 2, 3),
            r#"{"generated":"a\nb","prefix_hit":2,"page_hits":3}"#
        );
        assert_eq!(json_error("empty prompt"), r#"{"error":"empty prompt"}"#);
        assert_eq!(
            json_openai_error("empty prompt"),
            r#"{"error":{"message":"empty prompt","type":"invalid_request_error"}}"#
        );
        assert_eq!(
            json_openai_completion("hi", 1, 2, "length", "tiny"),
            r#"{"id":"cmpl-0","object":"text_completion","model":"tiny","choices":[{"text":"hi","index":0,"finish_reason":"length"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}"#
        );
        assert_eq!(
            json_openai_chat("hi", 1, 2, "stop", "tiny"),
            r#"{"id":"chatcmpl-0","object":"chat.completion","model":"tiny","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}"#
        );
        assert_eq!(
            json_field_string(&json_generated("a\"b\n", 0, 0), "generated").unwrap(),
            "a\"b\n"
        );
        assert_eq!(json_token("ab"), "{\"token\":\"ab\"}\n");
        assert_eq!(json_token("a\"b"), "{\"token\":\"a\\\"b\"}\n");
        assert_eq!(json_tokens(&[3, 4]), "{\"tokens\":[3,4],\"count\":2}");
        assert_eq!(json_detok("a\"b"), "{\"text\":\"a\\\"b\"}");
        assert_eq!(http_chunk(b"hi"), b"2\r\nhi\r\n");
        assert_eq!(http_chunk_end(), b"0\r\n\r\n");
    }

    #[test]
    fn handle_tiny_generate_matches_greedy_and_http_shape() {
        let (model, tok) = tiny_model();
        let args = defaults();
        let expect = greedy_generate_ctx(&model, &tok, "ab", 2, None).expect("greedy");
        let (head, body) = exchange(&post_json(r#"{"prompt":"ab"}"#), &model, &tok, &args);
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert!(head.contains("Content-Type: application/json"), "{head}");
        assert!(head.contains("Connection: keep-alive"), "{head}");
        let cl = header_value(&head, "content-length").expect("cl");
        assert_eq!(cl.parse::<usize>().unwrap(), body.len());
        assert_eq!(json_field_string(&body, "generated").unwrap(), expect);
        let (head_s, body_s) = exchange(
            &post_json(r#"{"prompt":"ab","stream":true}"#),
            &model,
            &tok,
            &args,
        );
        assert!(head_s.starts_with("HTTP/1.1 200 OK"), "{head_s}");
        assert!(head_s.contains("Content-Length"), "{head_s}");
        assert!(!head_s.to_ascii_lowercase().contains("chunked"), "{head_s}");
        assert_eq!(json_field_string(&body_s, "generated").unwrap(), expect);
        let zero = greedy_generate_ctx(&model, &tok, "ab", 0, None).expect("zero");
        let (_h, body0) = exchange(
            &post_json(r#"{"prompt":"ab","n_predict":0}"#),
            &model,
            &tok,
            &args,
        );
        assert_eq!(json_field_string(&body0, "generated").unwrap(), zero);
        let close = format!(
            "POST /generate HTTP/1.1\r\nConnection: close\r\nHost: 127.0.0.1\r\nContent-Length: {}\r\n\r\n{{\"prompt\":\"ab\"}}",
            r#"{"prompt":"ab"}"#.len()
        );
        let (head_c, body_c) = exchange(&close, &model, &tok, &args);
        assert!(head_c.contains("Connection: close"), "{head_c}");
        assert_eq!(json_field_string(&body_c, "generated").unwrap(), expect);
    }

    #[test]
    fn keep_alive_two_requests_one_connection() {
        let (model, tok) = tiny_model();
        let args = defaults();
        let expect = greedy_generate_ctx(&model, &tok, "ab", 2, None).expect("greedy");
        let a = post_json(r#"{"prompt":"ab"}"#);
        let b = post_json(r#"{"prompt":"ab"}"#);
        let mut sock = RwBuf {
            input: Cursor::new(format!("{a}{b}").into_bytes()),
            output: Vec::new(),
        };
        let mut cache = None;
        handle_connection(&mut sock, &model, &tok, &args, &mut cache).expect("handle");
        let out = std::str::from_utf8(&sock.output).expect("utf8");
        assert_eq!(out.matches("HTTP/1.1 200 OK").count(), 2, "{out}");
        assert!(out.contains("Connection: keep-alive"), "{out}");
        assert!(out.contains(&expect), "{out} expect={expect}");
    }

    #[test]
    fn empty_prompt_and_bad_method_fail_cleanly() {
        let (model, tok) = tiny_model();
        let args = defaults();
        let (head, body) = exchange(&post_json(r#"{"prompt":""}"#), &model, &tok, &args);
        assert!(head.starts_with("HTTP/1.1 400 "), "{head}");
        assert_eq!(json_field_string(&body, "error").unwrap(), "empty prompt");
        let get = "GET /generate HTTP/1.1\r\nContent-Length: 0\r\n\r\n";
        let (head, body) = exchange(get, &model, &tok, &args);
        assert!(head.starts_with("HTTP/1.1 405 "), "{head}");
        assert!(json_field_string(&body, "error").unwrap().contains("POST"));
        let missing = "POST /v1/models HTTP/1.1\r\nContent-Length: 0\r\n\r\n";
        let (head, _body) = exchange(missing, &model, &tok, &args);
        assert!(head.starts_with("HTTP/1.1 404 "), "{head}");
        let get_v1 = "GET /v1/completions HTTP/1.1\r\nContent-Length: 0\r\n\r\n";
        let (head, body) = exchange(get_v1, &model, &tok, &args);
        assert!(head.starts_with("HTTP/1.1 405 "), "{head}");
        assert!(body.contains("invalid_request_error"), "{body}");
        assert!(body.contains("method must be POST"), "{body}");
    }

    #[test]
    fn model_id_flag_overrides_path_stem() {
        let a = run(&["tiny.gguf", "--model-id", "qwen"]);
        assert_eq!(a.model_id, "qwen");
        let a = run(&["--model-id=qwen3", "m.gguf"]);
        assert_eq!(a.model_id, "qwen3");
        let err = parse_serve_args(["m.gguf", "--model-id="]).unwrap_err();
        assert!(err.contains("model-id must be non-empty"), "{err}");
    }

    #[test]
    fn get_health_models_and_idle_metrics() {
        let (model, tok) = tiny_model();
        let args = defaults();
        let (head, body) = exchange(
            "GET /health HTTP/1.1\r\nContent-Length: 0\r\n\r\n",
            &model,
            &tok,
            &args,
        );
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(body, json_health());
        let (head, body) = exchange(
            "GET /v1/models HTTP/1.1\r\nContent-Length: 0\r\n\r\n",
            &model,
            &tok,
            &args,
        );
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(body, json_models_list("tiny"));
        assert!(body.contains("\"object\":\"list\""), "{body}");
        let (head, body) = exchange(
            "GET /v1/models/tiny HTTP/1.1\r\nContent-Length: 0\r\n\r\n",
            &model,
            &tok,
            &args,
        );
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(body, json_model_object("tiny"));
        let (head, body) = exchange(
            "GET /v1/models/nope HTTP/1.1\r\nContent-Length: 0\r\n\r\n",
            &model,
            &tok,
            &args,
        );
        assert!(head.starts_with("HTTP/1.1 404 "), "{head}");
        assert!(body.contains("invalid_request_error"), "{body}");
        assert!(body.contains("model not found"), "{body}");
        let (head, body) = exchange(
            "GET /metrics HTTP/1.1\r\nContent-Length: 0\r\n\r\n",
            &model,
            &tok,
            &args,
        );
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(body, json_idle_metrics());
        let named = ServeArgs {
            model_id: "qwen".into(),
            ..defaults()
        };
        let (head, body) = exchange(
            "GET /v1/models HTTP/1.1\r\nContent-Length: 0\r\n\r\n",
            &model,
            &tok,
            &named,
        );
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(body, json_models_list("qwen"));
    }

    #[test]
    fn tokenize_and_detokenize_match_prompt_ids() {
        let (model, tok) = tiny_model();
        let args = defaults();
        let ids = prompt_ids(&tok, "ab").expect("ids");
        let (head, body) = exchange(
            &post_path("/tokenize", r#"{"prompt":"ab"}"#),
            &model,
            &tok,
            &args,
        );
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(body, json_tokens(&ids));
        let (head, body) = exchange(
            &post_path(
                "/detokenize",
                &format!("{{\"tokens\":[{}]}}", ids_csv(&ids)),
            ),
            &model,
            &tok,
            &args,
        );
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(body, json_detok(&tok.decode(&ids)));
        let (head, body) = exchange(
            "GET /tokenize HTTP/1.1\r\nContent-Length: 0\r\n\r\n",
            &model,
            &tok,
            &args,
        );
        assert!(head.starts_with("HTTP/1.1 405 "), "{head}");
        assert!(json_field_string(&body, "error").unwrap().contains("POST"));
        let (head, body) = exchange(
            &post_path("/tokenize", r#"{"prompt":""}"#),
            &model,
            &tok,
            &args,
        );
        assert!(head.starts_with("HTTP/1.1 400 "), "{head}");
        assert_eq!(json_field_string(&body, "error").unwrap(), "empty prompt");
        let (head, body) = exchange(
            &post_path("/detokenize", r#"{"tokens":[]}"#),
            &model,
            &tok,
            &args,
        );
        assert!(head.starts_with("HTTP/1.1 400 "), "{head}");
        assert_eq!(json_field_string(&body, "error").unwrap(), "empty tokens");
        let (model, tok) = chat_model(ECHO_TEMPLATE);
        let rendered = tok
            .apply_chat_template(&[ChatMessage::new("user", "ab")], true)
            .expect("template");
        let chat_ids = prompt_ids(&tok, &rendered).expect("chat ids");
        let (head, body) = exchange(
            &post_path(
                "/tokenize",
                r#"{"messages":[{"role":"user","content":"ab"}]}"#,
            ),
            &model,
            &tok,
            &args,
        );
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(body, json_tokens(&chat_ids));
        let ordinary = tok.encode("ab").expect("encode");
        let special = prompt_ids(&tok, "ab").expect("special");
        let (head, body) = exchange(
            &post_path("/tokenize", r#"{"prompt":"ab","add_special_tokens":false}"#),
            &model,
            &tok,
            &args,
        );
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(body, json_tokens(&ordinary));
        let (head, body) = exchange(
            &post_path("/tokenize", r#"{"prompt":"ab","add_special_tokens":true}"#),
            &model,
            &tok,
            &args,
        );
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(body, json_tokens(&special));
    }

    fn ids_csv(ids: &[u32]) -> String {
        let mut s = String::new();
        for (i, id) in ids.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&id.to_string());
        }
        s
    }

    fn expect_completion(model: &Llama, tok: &Tokenizer, prompt: &str, n: usize) -> String {
        let mut ids = prompt_ids(tok, prompt).expect("ids");
        let prompt_n = ids.len();
        let mut slot = None;
        let needed = ids.len().saturating_add(n);
        let cache = model
            .ensure_cache_page(&mut slot, needed, None, None)
            .expect("kv");
        let _full = greedy_generate_cache(model, tok, cache, &mut ids, n).expect("g");
        tok.decode(ids.get(prompt_n..).unwrap_or(&[]))
    }

    #[test]
    fn v1_completions_and_chat_map_onto_greedy() {
        let (model, tok) = tiny_model();
        let args = defaults();
        let expect = greedy_generate_ctx(&model, &tok, "ab", 2, None).expect("greedy");
        let completion = expect_completion(&model, &tok, "ab", 2);
        assert_ne!(completion, expect);
        let (head, body) = exchange(
            &post_v1("/v1/completions", r#"{"prompt":"ab","max_tokens":2}"#),
            &model,
            &tok,
            &args,
        );
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert!(body.contains("\"object\":\"text_completion\""), "{body}");
        assert!(body.contains("\"model\":\"tiny\""), "{body}");
        assert_eq!(json_string_at(&body, "text").unwrap(), completion);
        let (head, body) = exchange(
            &post_v1(
                "/v1/completions",
                r#"{"prompt":"ab","max_tokens":2,"echo":true}"#,
            ),
            &model,
            &tok,
            &args,
        );
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(json_string_at(&body, "text").unwrap(), expect);
        let (head, native) = exchange(&post_json(r#"{"prompt":"ab"}"#), &model, &tok, &args);
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(json_field_string(&native, "generated").unwrap(), expect);
        let (head, body) = exchange(
            &post_v1("/v1/completions/", r#"{"prompt":"ab"}"#),
            &model,
            &tok,
            &args,
        );
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(json_string_at(&body, "text").unwrap(), completion);
        let (model, tok) = chat_model(ECHO_TEMPLATE);
        let expect = greedy_generate_ctx(&model, &tok, "ab", 2, None).expect("greedy");
        let completion = expect_completion(&model, &tok, "ab", 2);
        let (head, body) = exchange(
            &post_v1(
                "/v1/chat/completions",
                r#"{"messages":[{"role":"user","content":"ab"}],"max_tokens":2}"#,
            ),
            &model,
            &tok,
            &args,
        );
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert!(body.contains("\"object\":\"chat.completion\""), "{body}");
        assert_eq!(json_string_at(&body, "content").unwrap(), completion);
        let (head, body) = exchange(
            &post_v1(
                "/v1/chat/completions",
                r#"{"messages":[{"role":"user","content":"ab"}],"max_tokens":2,"echo":true}"#,
            ),
            &model,
            &tok,
            &args,
        );
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(json_string_at(&body, "content").unwrap(), completion);
        let (head, native) = exchange(&post_json(r#"{"prompt":"ab"}"#), &model, &tok, &args);
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(json_field_string(&native, "generated").unwrap(), expect);
        let empty = exchange(
            &post_v1("/v1/completions", r#"{"prompt":""}"#),
            &model,
            &tok,
            &args,
        );
        assert!(empty.0.starts_with("HTTP/1.1 400 "), "{}", empty.0);
        assert!(empty.1.contains("invalid_request_error"), "{}", empty.1);
        assert!(empty.1.contains("empty prompt"), "{}", empty.1);
        let many = exchange(
            &post_v1("/v1/completions", r#"{"prompt":"ab","n":2}"#),
            &model,
            &tok,
            &args,
        );
        assert!(many.0.starts_with("HTTP/1.1 400 "), "{}", many.0);
        assert!(many.1.contains("n must be 1"), "{}", many.1);
    }

    #[test]
    fn serve_reuses_kv_prefix_across_requests() {
        let (model, tok) = tiny_model();
        let args = ServeArgs {
            n_ctx: Some(16),
            ..defaults()
        };
        let expect = greedy_generate_ctx(&model, &tok, "ab", 2, Some(16)).expect("greedy");
        let mut cache = None;
        let (head, body) = exchange_on(
            &post_json(r#"{"prompt":"ab"}"#),
            &model,
            &tok,
            &args,
            &mut cache,
        );
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(json_field_string(&body, "generated").unwrap(), expect);
        assert!(body.contains("\"prefix_hit\":0"), "{body}");
        let (head2, body2) = exchange_on(
            &post_json(r#"{"prompt":"ab"}"#),
            &model,
            &tok,
            &args,
            &mut cache,
        );
        assert!(head2.starts_with("HTTP/1.1 200 OK"), "{head2}");
        assert_eq!(json_field_string(&body2, "generated").unwrap(), expect);
        assert!(
            !body2.contains("\"prefix_hit\":0"),
            "second request must reuse the prompt prefix: {body2}"
        );
    }

    #[test]
    fn serve_paged_kv_matches_dense_greedy() {
        let (model, tok) = tiny_model();
        let args = ServeArgs {
            n_ctx: Some(16),
            kv_page: Some(2),
            ..defaults()
        };
        let expect =
            greedy_generate_slot(&model, &tok, &mut None, "ab", 2, Some(16), Some(2)).expect("p");
        let dense = greedy_generate_ctx(&model, &tok, "ab", 2, Some(16)).expect("d");
        assert_eq!(expect, dense);
        let mut cache = None;
        let (head, body) = exchange_on(
            &post_json(r#"{"prompt":"ab"}"#),
            &model,
            &tok,
            &args,
            &mut cache,
        );
        assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
        assert_eq!(json_field_string(&body, "generated").unwrap(), expect);
        assert!(body.contains("\"page_hits\":"), "{body}");
        let (head2, body2) = exchange_on(
            &post_json(r#"{"prompt":"ab"}"#),
            &model,
            &tok,
            &args,
            &mut cache,
        );
        assert!(head2.starts_with("HTTP/1.1 200 OK"), "{head2}");
        assert_eq!(json_field_string(&body2, "generated").unwrap(), expect);
        assert!(
            cache.as_ref().is_some_and(|c| c.page_size() == Some(2)),
            "persistent slot must stay paged"
        );
    }
}
