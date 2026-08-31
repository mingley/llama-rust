//! `gguf_gemv infer` / `chat` argument parsing, and the `chat` conversation
//! loop. No crates.io CLI crate.

use std::fs::File;
use std::io::{BufRead, Read, Write};
use std::path::Path;

use crate::decode::{prompt_ids, KvCache, Llama};
use crate::engine::{Engine, EngineCfg, SeqId};
use crate::gguf::load_gguf_owned;
use crate::sample::argmax;
use crate::store_attach::{attach_store, gpu_knobs, GpuCli, PlannerCli, StoreAttach};
use crate::template::ChatMessage;
use crate::tok::Tokenizer;
use expertvm::{report, GpuFill, GpuStoreCfg, HardwareProfile, Prefetch, Trace};

/// Usage for the `infer` verb.
pub const INFER_USAGE: &str = "\
usage: gguf_gemv infer <path> [--prompt TEXT] [--n-predict N] [--n-ctx N]
  -p, --prompt TEXT   prompt (default: ab)
  -n, --n-predict N   tokens to generate (default: 2)
      --n-ctx N       KV capacity (default: prompt + n_predict + 1)
";

/// Usage for the `trace` verb.
pub const TRACE_USAGE: &str = "\
usage: gguf_gemv trace <path> [--prompt TEXT] [--n-predict N] [--n-ctx N] --out FILE [--capacity N]
  -p, --prompt TEXT   prompt (default: ab)
  -n, --n-predict N   tokens to generate (default: 2)
      --n-ctx N       KV capacity (default: prompt + n_predict + 1)
      --out FILE      write ExpertAccess JSONL (required; optional per-event w permille)
      --capacity N    expertvm cache slots for the printed table (default: 8)
";

/// Usage for the `chat` verb.
pub const CHAT_USAGE: &str = "\
usage: gguf_gemv chat <path> [--system TEXT] [--prompt TEXT] [--n-predict N] [--n-ctx N] [--kv-page N] [--show-prompt]
  -s, --system TEXT   system message placed before the conversation
  -p, --prompt TEXT   send one user turn and exit; omit to read turns from stdin
  -n, --n-predict N   tokens to generate per reply (default: 64)
      --n-ctx N       persistent KV capacity (default: grow per turn)
      --kv-page N     paged KV block size in tokens (default: dense layout)
      --show-prompt   print the rendered chat template before each reply
";

/// Top-level binary usage.
pub const BIN_USAGE: &str = "\
usage: gguf_gemv <command> [args]
  infer <path> [--prompt TEXT] [--n-predict N] [--n-ctx N]
  trace <path> [--prompt TEXT] [--n-predict N] [--n-ctx N] --out FILE [--capacity N]
  chat <path> [--system TEXT] [--prompt TEXT] [--n-predict N] [--n-ctx N] [--kv-page N] [--show-prompt]
  serve <path> [--n-predict N] [--n-ctx N] [--kv-page N] [--bind HOST:PORT] [--model-id ID] [--engine] [--max-seqs N] [--expert-slots N] [--expert-sim] [--expert-8gpu] [--expert-bytes N] [--prefill-chunk N] [--decode-first] [--slo-reject] [--ttft-slo-ns N] [--itl-slo-ns N] [--cuda-graphs] [--graph-update] [--graph-set-params] [--graph-clone] [--graph-clone-parent] [--graph-build] [--graph-build-deps] [--graph-host] [--graph-piecewise] [--graph-capture-deps] [--graph-capture-host] [--graph-enable] [--graph-if] [--graph-mem] [--graph-memset] [--graph-memcpy] [--graph-leaf-host] [--graph-auto-free] [--graph-mem-trim] [--timing-events] [--event-blocking-sync] [--mapped] [--managed] [--vmm] [--vmm-retain] [--vmm-handle] [--vmm-page N] [--host-func] [--copy-host] [--blocking-streams] [--sync-alloc] [--ipc] [--mempool] [--mempool-trim] [--mempool-no-reuse] [--mempool-max N] [--shareable] [--share-ptr] [--pageable] [--host-register] [--host-unregister] [--host-register-mapped] [--sync-memops] [--device-sync-memops] [--memcpy-batch] [--memcpy-during] [--memcpy-any] [--memcpy-attr] [--memset-fill] [--accessed-by] [--no-read-mostly] [--no-preferred] [--no-mem-prefetch] [--legacy-null] [--stream-priority] [--seq-streams] [--kv-sim] [--kv-bytes N] [--decode-priority] [--cooperative] [--pdl] [--l2-persist] [--l2-reset] [--l2-fetch N] [--l2-ratio N] [--l2-streaming] [--cluster N] [--preferred-cluster N] [--cluster-spread] [--func-cluster-spread] [--cluster-load-balance] [--cluster-must-set] [--required-cluster N] [--max-shared] [--func-max-shared] [--max-l1] [--non-portable-cluster] [--sync-policy auto|spin|yield|blocking] [--device-sync-policy auto|spin|yield|blocking] [--mem-sync-domain default|remote] [--mem-sync-map identity|collapse] [--mem-sync-launch] [--mem-sync-launch-map] [--shared-mem] default|four|eight] [--func-shared-mem default|four|eight] [--device-shared-mem default|four|eight] [--portable-cluster default|portable|non-portable] [--optin-shared] [--dynamic-shared N] [--portable-shared default|portable|non-portable] [--nvlink-util] [--device-launch] [--device-updatable] [--kernel-priority N] [--launch-completion] [--programmatic-event] [--stream-attach] [--managed-host] [--prefetch-host] [--d2h-evict] [--d2h-pageable] [--wait-value] [--multicast] [--compute-slots N] [--decode-sms N] [--green-ctx] [--trace-out FILE]
  engine <path> [-p TEXT]... [-n N] [--n-ctx N] [--kv-page N] [--pool-blocks N] [--max-seqs N] [--prefill-chunk N] [--decode-first] [--slo-reject] [--ttft-slo-ns N] [--itl-slo-ns N] [--expert-slots N] [--expert-sim] [--expert-8gpu] [--expert-bytes N] [--cuda-graphs] [--graph-update] [--graph-set-params] [--graph-clone] [--graph-clone-parent] [--graph-build] [--graph-build-deps] [--graph-host] [--graph-piecewise] [--graph-capture-deps] [--graph-capture-host] [--graph-enable] [--graph-if] [--graph-mem] [--graph-memset] [--graph-memcpy] [--graph-leaf-host] [--graph-auto-free] [--graph-mem-trim] [--timing-events] [--event-blocking-sync] [--mapped] [--managed] [--vmm] [--vmm-retain] [--vmm-handle] [--vmm-page N] [--host-func] [--copy-host] [--blocking-streams] [--sync-alloc] [--ipc] [--mempool] [--mempool-trim] [--mempool-no-reuse] [--mempool-max N] [--shareable] [--share-ptr] [--pageable] [--host-register] [--host-unregister] [--host-register-mapped] [--sync-memops] [--device-sync-memops] [--memcpy-batch] [--memcpy-during] [--memcpy-any] [--memcpy-attr] [--memset-fill] [--accessed-by] [--no-read-mostly] [--no-preferred] [--no-mem-prefetch] [--legacy-null] [--stream-priority] [--seq-streams] [--kv-sim] [--kv-bytes N] [--decode-priority] [--cooperative] [--pdl] [--l2-persist] [--l2-reset] [--l2-fetch N] [--l2-ratio N] [--l2-streaming] [--cluster N] [--preferred-cluster N] [--cluster-spread] [--func-cluster-spread] [--cluster-load-balance] [--cluster-must-set] [--required-cluster N] [--max-shared] [--func-max-shared] [--max-l1] [--non-portable-cluster] [--sync-policy auto|spin|yield|blocking] [--device-sync-policy auto|spin|yield|blocking] [--mem-sync-domain default|remote] [--mem-sync-map identity|collapse] [--mem-sync-launch] [--mem-sync-launch-map] [--shared-mem] default|four|eight] [--func-shared-mem default|four|eight] [--device-shared-mem default|four|eight] [--portable-cluster default|portable|non-portable] [--optin-shared] [--dynamic-shared N] [--portable-shared default|portable|non-portable] [--nvlink-util] [--device-launch] [--device-updatable] [--kernel-priority N] [--launch-completion] [--programmatic-event] [--stream-attach] [--managed-host] [--prefetch-host] [--d2h-evict] [--d2h-pageable] [--wait-value] [--multicast] [--compute-slots N] [--decode-sms N] [--green-ctx] [--trace-out FILE] [--bench] [--capacity N]
  write|gemv|write-q4k|gemv-q4k|write-tiny|write-tiny-qwen2|write-tiny-qwen3|write-tiny-gemma|write-tiny-gemma2|write-tiny-gemma3|write-tiny-gemma3n|write-tiny-gemma4|write-tiny-gemma4-moe|write-tiny-gemma4-ple|write-tiny-gemma4-moe-ple|write-tiny-gemma4-moe-fused|write-tiny-gemma4-moe-fused-ple|write-tiny-gemma4-shared-kv|write-tiny-gemma4-mixed-hd|write-tiny-gemma4-swa-base|write-tiny-gemma4-no-wv|write-tiny-gemma4-out-scale|write-tiny-gemma4-rope-freqs|write-tiny-gemma4-moe-down-s|write-tiny-gemma4-moe-gate-up-s|write-tiny-llama-ffn-down-s|write-tiny-llama-attn-q-s|write-tiny-llama4|write-tiny-llama-moe|write-tiny-qwen2moe|write-tiny-qwen3moe|write-tiny-qwen3moe-2layer|write-tiny-qwen2vl|write-tiny-qwen3vl|write-tiny-qwen3next|write-tiny-qwen35|write-tiny-phi2|write-tiny-bloom <path>
";

/// Usage for the `engine` verb.
pub const ENGINE_USAGE: &str = "\
usage: gguf_gemv engine <path> [--prompt TEXT]... [--n-predict N] [--n-ctx N] [--kv-page N] [--pool-blocks N] [--max-seqs N] [--prefill-chunk N] [--decode-first] [--slo-reject] [--ttft-slo-ns N] [--itl-slo-ns N] [--expert-slots N] [--expert-sim] [--expert-8gpu] [--expert-bytes N] [--cuda-graphs] [--graph-update] [--graph-set-params] [--graph-clone] [--graph-clone-parent] [--graph-build] [--graph-build-deps] [--graph-host] [--graph-piecewise] [--graph-capture-deps] [--graph-capture-host] [--graph-enable] [--graph-if] [--graph-mem] [--graph-memset] [--graph-memcpy] [--graph-leaf-host] [--graph-auto-free] [--graph-mem-trim] [--timing-events] [--event-blocking-sync] [--mapped] [--managed] [--vmm] [--vmm-retain] [--vmm-handle] [--vmm-page N] [--host-func] [--copy-host] [--blocking-streams] [--sync-alloc] [--ipc] [--mempool] [--mempool-trim] [--mempool-no-reuse] [--mempool-max N] [--shareable] [--share-ptr] [--pageable] [--host-register] [--host-unregister] [--host-register-mapped] [--sync-memops] [--device-sync-memops] [--memcpy-batch] [--memcpy-during] [--memcpy-any] [--memcpy-attr] [--memset-fill] [--accessed-by] [--no-read-mostly] [--no-preferred] [--no-mem-prefetch] [--legacy-null] [--stream-priority] [--seq-streams] [--kv-sim] [--kv-bytes N] [--decode-priority] [--cooperative] [--pdl] [--l2-persist] [--l2-reset] [--l2-fetch N] [--l2-ratio N] [--l2-streaming] [--cluster N] [--preferred-cluster N] [--cluster-spread] [--func-cluster-spread] [--cluster-load-balance] [--cluster-must-set] [--required-cluster N] [--max-shared] [--func-max-shared] [--max-l1] [--non-portable-cluster] [--sync-policy auto|spin|yield|blocking] [--device-sync-policy auto|spin|yield|blocking] [--mem-sync-domain default|remote] [--mem-sync-map identity|collapse] [--mem-sync-launch] [--mem-sync-launch-map] [--shared-mem] default|four|eight] [--func-shared-mem default|four|eight] [--device-shared-mem default|four|eight] [--portable-cluster default|portable|non-portable] [--optin-shared] [--dynamic-shared N] [--portable-shared default|portable|non-portable] [--nvlink-util] [--device-launch] [--device-updatable] [--kernel-priority N] [--launch-completion] [--programmatic-event] [--stream-attach] [--managed-host] [--prefetch-host] [--d2h-evict] [--d2h-pageable] [--wait-value] [--multicast] [--compute-slots N] [--decode-sms N] [--green-ctx] [--prefetch none|copy-forward|markov|both] [--plan-window N] [--plan-threshold N] [--trace-out FILE] [--bench] [--capacity N]
  -p, --prompt TEXT     prompt (repeatable; default: one `ab`)
  -n, --n-predict N     tokens to generate per sequence (default: 2)
      --n-ctx N         KV capacity (default: longest prompt + n_predict + 1)
      --kv-page N       paged KV block size in tokens (default: 16)
      --pool-blocks N   physical intern blocks (default: max_seqs * pages)
      --max-seqs N      in-flight sequences (default: number of prompts; extras wait).
                        Engine tests cover 128 waiters at max_seqs=8 (batch-128).
      --prefill-chunk N prefill tokens per step (`0` = the rest; default: 0)
      --decode-first    hold leftover prefill while any live sequence is decoding
      --slo-reject      drop waiters whose gpu-sim queue wait meets `--ttft-slo-ns`
      --ttft-slo-ns N   virtual-ns TTFT budget (`--slo-reject`; needs `--expert-sim`)
      --itl-slo-ns N    count later-token gaps over this budget (needs `--expert-sim`)
      --expert-slots N  ExpertStore on the Engine: omit = blob FFN, `0` = DirectStore,
                        N>0 = CachedStore with N slots (`--expert-sim` uses N, default 8)
      --expert-sim      SimulatedGpuStore (example H100, 4096-byte experts)
      --expert-8gpu     8×H100 NVLink profile (`--expert-sim`; enables plan_placement)
      --expert-bytes N  simulated expert page bytes (`--expert-sim`; default: 4096)
      --cuda-graphs     document default GEMM graph capture (`--expert-sim`; always on)
      --graph-update    cudaGraphExecUpdate parked leaves (`--expert-sim`)
      --graph-set-params  cudaGraphExecKernelNodeSetParams parked leaves (`--expert-sim`; not with `--graph-update`)
      --graph-clone     cudaGraphClone leaf capture before instantiate (`--expert-sim`)
      --graph-clone-parent  cudaGraphClone combo parents before instantiate (`--expert-sim`; recursive children; does not imply `--graph-clone`; store GEMM stays per-leaf; legal with `--pdl` and `--cooperative`)
      --graph-build     cudaGraphCreate / cudaGraphAdd* instead of capture (`--expert-sim`; independent children may Hyper-Q overlap; not with `--graph-piecewise`)
      --graph-build-deps  cudaGraphAddDependencies on `--graph-build` combo parents (`--expert-sim`; needs `--graph-build`; sibling GEMMs serialize; legal with `--pdl` and `--cooperative`)
      --graph-host      cudaGraphAddHostNode BETWEEN `--graph-build` combo children (`--expert-sim`; needs `--graph-build`; sibling GEMMs serialize through `host_func_ns`; does not imply `--host-func`; legal with `--pdl` and `--cooperative`)
      --graph-piecewise cudaStreamBeginCaptureToGraph combo parents (`--expert-sim`; independent child roots may Hyper-Q overlap; not with `--graph-build`)
      --graph-capture-deps  cudaStreamBeginCaptureToGraph deps on `--graph-piecewise` combo parents (`--expert-sim`; needs `--graph-piecewise`; sibling GEMMs serialize; legal with `--pdl` and `--cooperative`)
      --graph-capture-host  cudaLaunchHostFunc BETWEEN `--graph-piecewise` combo fragments (`--expert-sim`; needs `--graph-piecewise`; sibling GEMMs serialize through `host_func_ns`; does not imply `--host-func`; legal with `--pdl` and `--cooperative`)
      --graph-enable    cudaGraphNodeSetEnabled skip extra combo children (`--expert-sim`; walker; not with `--device-launch`)
      --graph-if        cudaGraphAddIf + set-conditional skip extra combo children (`--expert-sim`; walker; needs `--graph-build`; not with `--device-launch` or `--graph-enable`; exec SetParams re-uploads)
      --graph-mem       in-graph scratch cudaMallocAsync (`--expert-sim`; skips `--graph-update`)
      --graph-memset    cudaGraphAddMemsetNode / cudaMemsetAsync of `--graph-mem` scratch BETWEEN alloc and GEMM (`--expert-sim`; needs `--graph-mem`; legal with `--pdl` and `--cooperative`)
      --graph-memcpy    cudaGraphAddMemcpyNode / cudaMemcpyAsync H2D of `--graph-mem` scratch BETWEEN alloc and GEMM (`--expert-sim`; needs `--graph-mem`; copy-engine PCIe; legal with `--pdl` and `--cooperative`)
      --graph-leaf-host cudaGraphAddHostNode / cudaLaunchHostFunc BEFORE the leaf GEMM (`--expert-sim`; implies `--cuda-graphs`; each leaf bills `host_func_ns`; does not imply `--host-func`; not with `--device-launch`; legal with `--pdl` and `--cooperative`)
      --graph-auto-free AutoFreeOnLaunch scratch without in-graph free (`--expert-sim`; not with `--graph-mem`)
      --graph-mem-trim  cudaDeviceGraphMemTrim unused reserved after score (`--expert-sim`)
      --timing-events   cudaEventElapsedTime on copy start/end (`--expert-sim`)
      --event-blocking-sync  cudaEventBlockingSync copy events (`--expert-sim`; implies `--timing-events`; distinct from `--sync-policy blocking`)
      --mapped          cudaHostAllocMapped miss pages (`--expert-sim`; not pinned identity)
      --managed         cudaMallocManaged miss pages (`--expert-sim`; not pinned identity)
      --vmm             va_acquire miss pages (`--expert-sim`; not pinned identity)
      --vmm-page N      va_acquire_paged span (`--expert-sim`; `N>0` implies `--vmm`)
      --vmm-retain      cuMemRetainAllocationHandle after VMM miss map (`--expert-sim`; implies `--vmm`; `alloc_overhead_ns`; not with `--mapped`/`--managed`)
      --vmm-handle      cuMemCreate plus cuMemMap miss pages (`--expert-sim`; implies `--vmm`; `alloc_overhead_ns`; not with `--mapped`/`--managed`/`--vmm-retain`)
      --host-func       cudaLaunchHostFunc after acquire GEMM (`--expert-sim`)
      --copy-host       cudaLaunchHostFunc after miss DMA / prefetch (`--expert-sim`; bills `host_func_ns` before copy-ready; does not imply `--host-func`; legal with `--pdl` and `--cooperative`)
      --blocking-streams  blocking compute stream (`--expert-sim`)
      --sync-alloc      host-sync malloc/memcpy/free (`--expert-sim`)
      --mempool         hold unused cudaMallocAsync bytes (`--expert-sim`)
      --mempool-trim    cudaMemPoolTrimTo unused cached after score (`--expert-sim`; implies `--mempool`)
      --mempool-no-reuse  cudaMemPoolReuseAllowOpportunistic=0 (`--expert-sim`; implies `--mempool`)
      --mempool-max N   cudaMemPoolProps::maxSize (`--expert-sim`; implies `--mempool`; `N>0`; needs cudaMallocAsync)
      --shareable       POSIX-FD mempool IPC (`--expert-sim`; implies `--mempool`; needs cudaMallocAsync)
      --ipc             cudaIpcGetMemHandle / OpenMemHandle (`--expert-sim`; implies `--sync-alloc`; needs cudaMalloc; not with `--shareable`)
      --share-ptr       cudaMemPoolExportPointer / ImportPointer (`--expert-sim`; implies `--shareable`; needs cudaMallocAsync; not with `--ipc`)
      --pageable        pageable H2D (`--expert-sim`; slower than pinned)
      --host-register   cudaHostRegister pageable staging then pinned DMA (`--expert-sim`; implies `--pageable`; not with `--mapped`/`--managed`)
      --host-unregister cudaHostUnregister staging after miss DMA (`--expert-sim`; implies `--host-register`; pin refunded between misses; `synchronize`; not with `--mapped`/`--managed`/`--d2h-pageable`)
      --host-register-mapped  cudaHostRegisterMapped expert pages (`--expert-sim`; implies `--mapped`; not with `--host-register`)
      --sync-memops     cuPointerSetAttribute SyncMemops on miss pages (`--expert-sim`; host-sync H2D; not with `--mapped` / `--memcpy-batch`)
      --device-sync-memops  cudaSetDeviceFlags SyncMemops (`--expert-sim`; host-sync memcpy on that GPU; not with `--mapped` / `--memcpy-batch`)
      --memcpy-batch    cudaMemcpyBatchAsync for multi-expert pinned/VMM prefetch (`--expert-sim`)
      --memcpy-during   cudaMemcpySrcAccessOrderDuringApiCall on `--memcpy-batch` (`--expert-sim`; needs `--memcpy-batch`; API waits those copies)
      --memcpy-any      cudaMemcpySrcAccessOrderAny on `--memcpy-batch` (`--expert-sim`; needs `--memcpy-batch`; empty deps; no API wait; not with `--memcpy-during`)
      --memcpy-attr     cudaMemcpyWithAttributesAsync DuringApiCall on demand pinned/VMM H2D (`--expert-sim`; API waits that copy; not with `--mapped`/`--managed`/`--pageable`/`--memset-fill`/`--sync-alloc`/`--sync-memops`/`--device-sync-memops`)
      --memset-fill     cudaMemsetAsync miss fill instead of pinned H2D (`--expert-sim`; HBM write; not with `--mapped`/`--managed`/`--pageable`/`--memcpy-batch`; legal with `--pdl` and `--cooperative`)
      --accessed-by     SetAccessedBy / VMM SetAccess / mempool SetAccess (`--expert-sim`; no dest HBM)
      --no-read-mostly  skip cudaMemAdviseSetReadMostly (`--expert-sim`; implies `--managed`; prefetch moves instead of replicating)
      --no-preferred    skip cudaMemAdviseSetPreferredLocation (`--expert-sim`; implies `--managed`; remote GEMM first-touches instead of staying on home)
      --no-mem-prefetch skip cudaMemPrefetchAsync at managed fill (`--expert-sim`; implies `--managed`; kernel first-touches instead of copy-engine prefetch)
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
      --d2h-evict       cudaMemcpyAsync Device→HostPinned before pinned/VMM LRU free (`--expert-sim`; extra PCIe; next miss still fills from staging; not with `--mapped`/`--managed`)
      --d2h-pageable    cudaMemcpyAsync Device→Host (pageable) before pinned/VMM LRU free (`--expert-sim`; host-sync bounce-buffer; implies `--pageable`; not with `--mapped`/`--managed`/`--host-register`/`--d2h-evict`)
      --wait-value      cuStreamWaitValue64 / WriteValue64 copy-ready handshake (`--expert-sim`; 8-byte device mailbox; decode identity stays events)
      --multicast       cuMulticastCreate NVLS replica fanout (`--expert-sim`; implies `--vmm`; needs NVLink)
      --compute-slots N Hyper-Q occupancy (`--expert-sim`; `1` exclusive, `>=2` overlaps leftover prefill with decode on two streams; default profile `1`)
      --decode-sms N    decode-stream SM permille (`--expert-sim`; `1..=1000`; leftover prefill gets the remainder; implies `--decode-priority`; default full chip)
      --green-ctx       complementary CUDA green contexts on decode vs leftover prefill (`--expert-sim`; occupancy partition even when `compute_slots` is 1; implies `--decode-priority`; default 500/500; `--decode-sms N` with leftover SMs; wall vs `--decode-sms` duration scale; legal with `--pdl`/`--cooperative`)
      --prefetch MODE   none|copy-forward|markov|both (default: both; CachedStore or sim)
      --plan-window N   Stay vs Fetch over N unique predicted keys (`0` = ungated)
      --plan-threshold N  Stay permille of that window already resident (default: 500)
      --trace-out FILE  write batched MoE ExpertAccess JSONL (all sequences)
      --bench           infer-bench report on this run's MoE traces (policy table;
                        with `--expert-sim`, the same sim scorecard as `infer-bench trace`)
      --capacity N      expertvm cache slots for `--bench` (default: `--expert-slots`, or 8)

Runs Engine continuous batching on one interned pool. Several `--prompt`s
join the same scheduler. `--decode-first` holds leftover prefill while any
live sequence is already decoding (ITL over leftover prefill; same policy
as `expertvm schedule --decode-first`). `--slo-reject` / `--ttft-slo-ns`
drops a waiter whose gpu-sim queue wait already meets the TTFT budget
(same policy as `expertvm schedule --slo-reject`; needs `--expert-sim`).
`--itl-slo-ns` counts later-token gaps over the budget (`itl_slo_miss=`;
does not drop). `--expert-sim` captures per-page GEMM graphs (same
mechanical path as `expertvm sim --cuda-graphs`). `--graph-update` /
`--graph-set-params` / `--graph-clone` / `--graph-clone-parent` / `--graph-build` / `--graph-build-deps` / `--graph-host` / `--graph-piecewise` / `--graph-capture-deps` / `--graph-capture-host` / `--graph-enable` / `--graph-if` / `--graph-mem` / `--graph-memset` / `--graph-memcpy` / `--graph-leaf-host` / `--graph-auto-free` / `--graph-mem-trim` / `--timing-events` / `--event-blocking-sync` / `--host-func` / `--copy-host` / `--blocking-streams` /
`--sync-alloc` / `--ipc` / `--mempool` / `--mempool-trim` / `--mempool-no-reuse` / `--mempool-max` / `--shareable` / `--share-ptr` / `--vmm-retain` / `--vmm-handle` / `--vmm-page` / `--pageable` / `--host-register` / `--host-unregister` / `--host-register-mapped` / `--sync-memops` / `--device-sync-memops` / `--memcpy-batch` / `--memcpy-during` / `--memcpy-any` / `--memcpy-attr` / `--d2h-evict` / `--d2h-pageable` / `--memset-fill` / `--accessed-by` / `--no-read-mostly` /
`--no-preferred` / `--no-mem-prefetch` / `--legacy-null` / `--stream-priority` / `--seq-streams` / `--kv-sim` /
`--kv-bytes` / `--decode-priority` / `--cooperative` / `--pdl` / `--l2-persist` / `--l2-reset` / `--l2-fetch` / `--l2-ratio` / `--l2-streaming` / `--cluster` / `--preferred-cluster` / `--cluster-spread` / `--func-cluster-spread` / `--cluster-load-balance` / `--cluster-must-set` / `--required-cluster` / `--max-shared` / `--func-max-shared` / `--max-l1` / `--non-portable-cluster` / `--sync-policy` / `--device-sync-policy` / `--event-blocking-sync` / `--mem-sync-domain` / `--mem-sync-map` / `--mem-sync-launch` / `--mem-sync-launch-map` / `--shared-mem` / `--func-shared-mem` / `--device-shared-mem` / `--portable-cluster` / `--optin-shared` / `--dynamic-shared` / `--portable-shared` / `--nvlink-util` / `--device-launch` / `--device-updatable` / `--kernel-priority` / `--launch-completion` / `--programmatic-event` / `--stream-attach` / `--managed-host` / `--prefetch-host` / `--d2h-evict` / `--d2h-pageable` / `--host-unregister` / `--ipc` / `--share-ptr` / `--vmm-retain` / `--vmm-handle` / `--wait-value` / `--multicast` / `--compute-slots` / `--decode-sms` / `--green-ctx` match `GpuStoreCfg` / `expertvm sim`. `--mempool-max N` is `cudaMemPoolProps::maxSize` (implies `--mempool`; illegal with `--sync-alloc`; `N>0`; legal with `--pdl` and `--cooperative`). `--memcpy-during` is `cudaMemcpySrcAccessOrderDuringApiCall` on `--memcpy-batch` prefetch (needs `--memcpy-batch`; the batch API waits those copies; identity stays Stream order; legal with `--pdl` and `--cooperative`). `--memcpy-any` is `cudaMemcpySrcAccessOrderAny` on `--memcpy-batch` prefetch (needs `--memcpy-batch`; empty deps; no API wait; not with `--memcpy-during`; legal with `--pdl` and `--cooperative`). `--memcpy-attr` is `cudaMemcpyWithAttributesAsync` DuringApiCall on demand pinned/VMM miss H2D (the API waits that copy; identity stays `memcpy_pinned_to_device`; does not imply `--memcpy-batch`; not with `--mapped` / `--managed` / `--pageable` / `--memset-fill` / `--sync-alloc` / `--sync-memops` / `--device-sync-memops`; legal with `--pdl` and `--cooperative`). `--d2h-evict` is `cudaMemcpyAsync` Device→HostPinned before pinned/VMM LRU free (extra PCIe; next miss still fills from staging; not with `--mapped` / `--managed`; distinct from `--prefetch-host`; legal with `--pdl` and `--cooperative`). `--memset-fill` is `cudaMemsetAsync` of pinned/VMM miss pages (HBM write, compute occupancy; not with `--mapped` / `--managed` / `--pageable` / `--memcpy-batch`; legal with `--pdl` and `--cooperative`). `--copy-host` is `cudaLaunchHostFunc` after miss DMA / prefetch (bills `host_func_ns` before copy-ready; does not imply `--host-func`; mapped misses are a no-op; legal with `--pdl` and `--cooperative`). `--graph-capture-deps` is `cudaStreamBeginCaptureToGraph` deps on `--graph-piecewise` combo parents (needs `--graph-piecewise`; later fragments chain so sibling GEMMs serialize; empty deps stay extra roots; store GEMM stays per-leaf; legal with `--pdl` and `--cooperative`). `--graph-capture-host` is captured `cudaLaunchHostFunc` BETWEEN those fragments (needs `--graph-piecewise`; later fragments chain through `host_func_ns`; does not imply `--host-func`; store GEMM stays per-leaf; legal with `--pdl` and `--cooperative`). `--graph-clone-parent` is `cudaGraphClone` of combo parents (recursive children; does not imply `--graph-clone`; store GEMM stays per-leaf; legal with `--pdl` and `--cooperative`). `--graph-build-deps` is `cudaGraphAddDependencies` on `--graph-build` combo parents (needs `--graph-build`; later children chain so sibling GEMMs serialize; empty edges stay independent; store GEMM stays per-leaf; legal with `--pdl` and `--cooperative`). `--graph-host` is `cudaGraphAddHostNode` BETWEEN `--graph-build` combo children (needs `--graph-build`; later children serialize through `host_func_ns`; does not imply `--host-func`; store GEMM stays per-leaf; legal with `--pdl` and `--cooperative`). `--graph-if` wraps `--graph-build` combo children in `cudaGraphAddIf` + `cudaGraphSetConditional` (needs `--graph-build`; exec SetParams skips extras and re-uploads; not with `--device-launch` or `--graph-enable`; store GEMM stays per-leaf; legal with `--pdl` and `--cooperative`). `--graph-memset` is `cudaGraphAddMemsetNode` / `cudaMemsetAsync` of `--graph-mem` scratch BETWEEN alloc and GEMM (needs `--graph-mem`; extra HBM-write tax; does not imply graph-mem; store and walker leaf GEMMs; legal with `--pdl` and `--cooperative`). `--graph-memcpy` is `cudaGraphAddMemcpyNode` / `cudaMemcpyAsync` H2D of `--graph-mem` scratch BETWEEN alloc and GEMM (needs `--graph-mem`; copy-engine PCIe tax; does not imply graph-mem or `--graph-memset`; store and walker leaf GEMMs; legal with `--pdl` and `--cooperative`). `--kv-sim` bills interned
KV map/memset/hits on the same virtual clock as expert H2D (distinct from
`expertvm kv`; default off keeps decode identity). `--decode-priority` ITL
samples the decode compute stream so leftover prefill does not inflate it.
`--compute-slots N` (`N>=2`, with `--decode-priority`) lets leftover prefill
and decode GEMMs overlap at full issue rate. `--green-ctx` partitions decode
vs leftover prefill into complementary CUDA green contexts so they may overlap
even when occupancy is exclusive. `--pdl` lets consecutive
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
`--d2h-evict` is `cudaMemcpyAsync` Device→HostPinned before pinned/VMM LRU
free (extra PCIe; the next miss still fills from catalog staging; not with
`--mapped` / `--managed`; identity stays free with no D2H). Distinct from
`--prefetch-host` (managed keep-alloc).
`--d2h-pageable` is `cudaMemcpyAsync` Device→Host (pageable bounce-buffer)
before that free (host-synchronous; implies `--pageable`; not with
`--mapped` / `--managed` / `--host-register` / `--d2h-evict`; identity stays
free with no D2H).
`--host-unregister` is `cudaHostUnregister` after each miss DMA so the next
miss re-registers (`synchronize` plus `first_alloc_ns`; pin refunded between
misses; implies `--host-register` which implies `--pageable`; not with
`--mapped` / `--managed` / `--host-register-mapped` / `--d2h-pageable`;
identity keeps staging registered). Distinct from `--host-register` (keep
pinned for lifetime).
`--ipc` is `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` of each miss
`cudaMalloc` (alias shares source HBM; close before `cudaFree`; implies
`--sync-alloc`; not with `--mapped` / `--managed` / `--vmm` / `--shareable`;
identity GEMMs on the `cudaMalloc` pointer with no IPC handshake). Distinct
from `--shareable` (POSIX-FD mempool IPC).
`--share-ptr` is `cudaMemPoolExportPointer` / `cudaMemPoolImportPointer` of
each miss `cudaMallocAsync` (alias shares source HBM; `cudaFreeAsync` the
import before the source; implies `--shareable` which implies `--mempool`;
not with `--ipc` / `--mapped` / `--managed` / `--vmm` / `--sync-alloc`;
identity is pool-level `--shareable` without a per-page pointer handshake).
Distinct from `--shareable` (POSIX-FD pool-level IPC) and `--ipc`
(`cudaIpcGetMemHandle` of `cudaMalloc`).
`--vmm-retain` is `cuMemRetainAllocationHandle` after each VMM miss map
(`alloc_overhead_ns` on every retain; implies `--vmm`; paged VMM retains
offset 0 only; `va_release_handle` before `va_release` so hits stay the
same; not with `--mapped` / `--managed`; identity is `--vmm` without a
handle handshake). Distinct from `--vmm` (combined `va_map` without a
handle) and `--vmm-page` (split maps; retain still one handle at offset 0).
`--vmm-handle` is `cuMemCreate` plus `cuMemMap` of that handle instead of
combined `va_map` (`alloc_overhead_ns` on create and map; implies `--vmm`;
paged VMM creates offset 0 only; `va_release_handle` before `va_release` so
hits stay the same; not with `--mapped` / `--managed` / `--vmm-retain`;
identity is `--vmm` without create plus map). Distinct from `--vmm`
(combined `va_map`) and `--vmm-retain` (promote after acquire).
`--no-read-mostly` skips `cudaMemAdviseSetReadMostly` (`UnsetReadMostly` so dest
prefetch moves instead of replicating; implies `--managed`; identity stays
SetReadMostly).
`--no-preferred` skips `cudaMemAdviseSetPreferredLocation`
(`UnsetPreferredLocation` so a remote GEMM first-touches instead of staying
on home; implies `--managed`; identity stays SetPreferredLocation).
`--no-mem-prefetch` skips `cudaMemPrefetchAsync` at managed fill (kernel
first-touches on compute instead of copy-engine prefetch; implies
`--managed`; identity stays fill prefetch). Distinct from `--prefetch none`
(predictor) and `--prefetch-host` (evict to host).
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
so compute waits a device mailbox instead of a copy event (illegal with
nothing; decode identity stays events; GEMM graphs stay kernel-only).
`--cooperative` is
`cudaLaunchCooperativeKernel`: those GEMMs occupy every Hyper-Q slot, so
leftover prefill cannot overlap even with `--compute-slots 2`. `--multicast`
is Hopper NVLS replica fanout (`cuMulticastCreate`; implies `--vmm`; needs
NVLink / `--expert-8gpu`). Decode identity stays copy-engine D2D. Default
profile occupancy is exclusive (`1`). `--decode-sms N` (`1..=1000`) reserves that permille of
peak FLOP/s for decode GEMMs (duration-only; leftover prefill gets the
remainder). Default unset is a full chip. `--green-ctx` binds complementary
CUDA green contexts so leftover prefill may overlap decode even when occupancy
is exclusive (`compute_slots` 1; implies `--decode-priority`; default 500/500
split; `--decode-sms 1000` refused). Distinct from `--decode-sms` alone
(exclusive compute still serializes leftover prefill).
`--bench` records the Engine's batched MoE traces and prints
`expertvm::report` (same policy table and, with `--expert-sim`, the same
sim A/B lines as `infer-bench trace`). Does not add llama-rust as an
infer-bench dependency. `--capacity N` is the replay cache size for that
table (default `--expert-slots`, or 8 when slots are omitted / DirectStore).
`--prefetch` / `--plan-window` / `--plan-threshold` match `expertvm sim`
Stay vs Fetch on the serving path (predicted keys only; no JSONL future
leak). Default `both` / window `0` / threshold `500` is today's decode
policy. A tight `--pool-blocks` preempts
(recompute + replay). One ExpertStore is parked on each batched GEMM so
MoE serving stays on the shared-pool path. After each GEMM the store
sticky-pins last-used ∪ predicted experts (`slots - 1`; `slots == 1` pins
nothing). A multi-GPU SimulatedGpuStore then `plan_placement`s those pins
(D2D onto GPU0 vs leave on the striped home). Managed/VMM moves drain that
page's GEMM lease first; `place_hot` errors are fail-loud.
MoE traces stay on that GEMM (per-row
sequence / token / prefix). Prints each continuation (`n_gen` plus
decoded text), then intern_hits, preempts, GEMM stats, store metrics,
graph counters, `kv_miss` / `kv_hit` when `--kv-sim`, ITL SLO misses, and a gpu-sim score line when
`--expert-sim` is set (`ttft_ns` / `itl_ns` when sequences generate).
Not `$/M tokens`.
Not an HTTP server.
";

/// Parsed `infer` invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferCmd {
    /// `--help` / `-h`.
    Help,
    /// Run greedy decode with these arguments.
    Run(InferArgs),
}

/// Arguments for seedless greedy `infer`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferArgs {
    /// GGUF path.
    pub path: String,
    /// Prompt text.
    pub prompt: String,
    /// Tokens to generate after the prompt.
    pub n_predict: usize,
    /// Optional KV capacity. `None` sizes to prompt + `n_predict` + 1.
    pub n_ctx: Option<usize>,
}

impl InferArgs {
    /// Default prompt when `--prompt` is omitted.
    pub const DEFAULT_PROMPT: &'static str = "ab";
    /// Default `--n-predict` when omitted.
    pub const DEFAULT_N_PREDICT: usize = 2;
}

/// Parse operands after the `infer` verb.
///
/// `infer <path> [--prompt TEXT] [--n-predict N] [--n-ctx N]`
/// Path may appear before or after flags. `--flag=value` is accepted.
pub fn parse_infer_args<I, S>(args: I) -> Result<InferCmd, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut path = None;
    let mut prompt = InferArgs::DEFAULT_PROMPT.to_string();
    let mut n_predict = InferArgs::DEFAULT_N_PREDICT;
    let mut n_ctx = None;
    let mut it = args.into_iter();
    while let Some(raw) = it.next() {
        let arg = raw.as_ref();
        if arg == "--help" || arg == "-h" {
            return Ok(InferCmd::Help);
        }
        let (key, inline) = match arg.split_once('=') {
            Some((k, v)) => (k, Some(v)),
            None => (arg, None),
        };
        match key {
            "--prompt" | "-p" => {
                prompt = opt_value("prompt", inline, &mut it)?;
            }
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
            flag if flag.starts_with('-') => {
                return usage_err(&format!("unknown flag {flag}"));
            }
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
    Ok(InferCmd::Run(InferArgs {
        path,
        prompt,
        n_predict,
        n_ctx,
    }))
}

/// Parsed `trace` invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceCmd {
    /// `--help` / `-h`.
    Help,
    /// Emit MoE JSONL with these arguments.
    Run(TraceArgs),
}

/// Arguments for `gguf_gemv trace`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceArgs {
    /// Same greedy settings as [`InferArgs`].
    pub infer: InferArgs,
    /// Destination JSONL path.
    pub out: String,
    /// Expert cache slots for the printed `expertvm replay` table.
    pub capacity: usize,
}

/// Parse operands after the `trace` verb.
pub fn parse_trace_args<I, S>(args: I) -> Result<TraceCmd, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut path = None;
    let mut prompt = InferArgs::DEFAULT_PROMPT.to_string();
    let mut n_predict = InferArgs::DEFAULT_N_PREDICT;
    let mut n_ctx = None;
    let mut out = None;
    let mut capacity = 8usize;
    let mut it = args.into_iter();
    while let Some(raw) = it.next() {
        let arg = raw.as_ref();
        if arg == "--help" || arg == "-h" {
            return Ok(TraceCmd::Help);
        }
        let (key, inline) = match arg.split_once('=') {
            Some((k, v)) => (k, Some(v)),
            None => (arg, None),
        };
        match key {
            "--prompt" | "-p" => {
                prompt = trace_value("prompt", inline, &mut it)?;
            }
            "--n-predict" | "-n" => {
                n_predict = parse_usize("n-predict", &trace_value("n-predict", inline, &mut it)?)?;
            }
            "--n-ctx" => {
                let n = parse_usize("n-ctx", &trace_value("n-ctx", inline, &mut it)?)?;
                if n == 0 {
                    return trace_usage_err("n-ctx must be > 0");
                }
                n_ctx = Some(n);
            }
            "--out" => {
                out = Some(trace_value("out", inline, &mut it)?);
            }
            "--capacity" | "-c" => {
                capacity = parse_usize("capacity", &trace_value("capacity", inline, &mut it)?)?;
                if capacity == 0 {
                    return trace_usage_err("capacity must be > 0");
                }
            }
            flag if flag.starts_with('-') => {
                return trace_usage_err(&format!("unknown flag {flag}"));
            }
            other => {
                if path.is_some() {
                    return trace_usage_err(&format!("unexpected argument {other}"));
                }
                path = Some(other.to_string());
            }
        }
    }
    let Some(path) = path else {
        return trace_usage_err("missing GGUF path");
    };
    let Some(out) = out else {
        return trace_usage_err("missing --out");
    };
    Ok(TraceCmd::Run(TraceArgs {
        infer: InferArgs {
            path,
            prompt,
            n_predict,
            n_ctx,
        },
        out,
        capacity,
    }))
}

/// Parsed `chat` invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatCmd {
    /// `--help` / `-h`.
    Help,
    /// Hold a conversation with these arguments.
    Run(ChatArgs),
}

/// Arguments for `chat`, which renders the model's own `tokenizer.chat_template`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatArgs {
    /// GGUF path.
    pub path: String,
    /// Optional `system` message placed before the conversation.
    pub system: Option<String>,
    /// One user turn. `None` reads turns from stdin until EOF.
    pub prompt: Option<String>,
    /// Tokens to generate per reply.
    pub n_predict: usize,
    /// Optional KV capacity. `None` sizes to prompt + `n_predict` + 1.
    pub n_ctx: Option<usize>,
    /// Paged KV block size in tokens. `None` keeps the dense layout.
    pub kv_page: Option<usize>,
    /// Print the rendered template before each reply.
    pub show_prompt: bool,
}

impl ChatArgs {
    /// Default `--n-predict` when omitted. Larger than `infer`'s, because a
    /// chat reply that stops after two tokens tells you nothing.
    pub const DEFAULT_N_PREDICT: usize = 64;
}

/// Parse operands after the `chat` verb.
///
/// `chat <path> [--system TEXT] [--prompt TEXT] [--n-predict N] [--n-ctx N] [--kv-page N] [--show-prompt]`
/// Path may appear before or after flags. `--flag=value` is accepted.
pub fn parse_chat_args<I, S>(args: I) -> Result<ChatCmd, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut path = None;
    let mut system = None;
    let mut prompt = None;
    let mut n_predict = ChatArgs::DEFAULT_N_PREDICT;
    let mut n_ctx = None;
    let mut kv_page = None;
    let mut show_prompt = false;
    let mut it = args.into_iter();
    while let Some(raw) = it.next() {
        let arg = raw.as_ref();
        if arg == "--help" || arg == "-h" {
            return Ok(ChatCmd::Help);
        }
        let (key, inline) = match arg.split_once('=') {
            Some((k, v)) => (k, Some(v)),
            None => (arg, None),
        };
        match key {
            "--system" | "-s" => system = Some(chat_value("system", inline, &mut it)?),
            "--prompt" | "-p" => prompt = Some(chat_value("prompt", inline, &mut it)?),
            "--n-predict" | "-n" => {
                let v = chat_value("n-predict", inline, &mut it)?;
                n_predict = v
                    .parse::<usize>()
                    .map_err(|_| format!("invalid n-predict {v:?}\n{CHAT_USAGE}"))?;
            }
            "--n-ctx" => {
                let v = chat_value("n-ctx", inline, &mut it)?;
                let n = v
                    .parse::<usize>()
                    .map_err(|_| format!("invalid n-ctx {v:?}\n{CHAT_USAGE}"))?;
                if n == 0 {
                    return chat_usage_err("n-ctx must be > 0");
                }
                n_ctx = Some(n);
            }
            "--kv-page" => {
                let v = chat_value("kv-page", inline, &mut it)?;
                let n = v
                    .parse::<usize>()
                    .map_err(|_| format!("invalid kv-page {v:?}\n{CHAT_USAGE}"))?;
                if n == 0 {
                    return chat_usage_err("kv-page must be > 0");
                }
                kv_page = Some(n);
            }
            "--show-prompt" => show_prompt = true,
            flag if flag.starts_with('-') => {
                return chat_usage_err(&format!("unknown flag {flag}"));
            }
            other => {
                if path.is_some() {
                    return chat_usage_err(&format!("unexpected argument {other}"));
                }
                path = Some(other.to_string());
            }
        }
    }
    let Some(path) = path else {
        return chat_usage_err("missing GGUF path");
    };
    Ok(ChatCmd::Run(ChatArgs {
        path,
        system,
        prompt,
        n_predict,
        n_ctx,
        kv_page,
        show_prompt,
    }))
}

/// Parsed `engine` invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[expect(
    clippy::large_enum_variant,
    reason = "EngineArgs is the parsed CLI record; boxing would churn every match"
)]
pub enum EngineCmd {
    /// `--help` / `-h`.
    Help,
    /// Run continuous batching with these arguments.
    Run(EngineArgs),
}

/// Arguments for `gguf_gemv engine`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineArgs {
    /// GGUF path.
    pub path: String,
    /// Prompts, one sequence each. Empty is replaced with a single `ab`.
    pub prompts: Vec<String>,
    /// Tokens to generate after each prompt.
    pub n_predict: usize,
    /// Optional KV capacity. `None` sizes to the longest prompt + `n_predict` + 1.
    pub n_ctx: Option<usize>,
    /// Paged KV block size in tokens.
    pub block_size: usize,
    /// Physical intern blocks. `None` sizes to `max_seqs` times pages per sequence.
    pub pool_blocks: Option<usize>,
    /// In-flight sequences. `None` is the number of prompts.
    pub max_seqs: Option<usize>,
    /// Prefill tokens per sequence per step (`0` = the rest of the prompt).
    pub prefill_chunk: usize,
    /// Hold leftover prefill while any live sequence is already decoding.
    pub decode_first: bool,
    /// Drop waiters whose gpu-sim queue wait meets `ttft_slo_ns`.
    pub slo_reject: bool,
    /// Virtual-ns TTFT budget with `--slo-reject`. `None` never drops.
    pub ttft_slo_ns: Option<u64>,
    /// Later-token gap budget. Misses print `itl_slo_miss`. `None` does not count.
    pub itl_slo_ns: Option<u64>,
    /// ExpertStore slots. `None` keeps blob FFN. `Some(0)` is DirectStore.
    pub expert_slots: Option<usize>,
    /// Attach [`SimulatedGpuStore`] (example H100) instead of Direct/Cached.
    pub expert_sim: bool,
    /// Use [`HardwareProfile::example_8xh100_nvlink`] with `--expert-sim`.
    pub expert_8gpu: bool,
    /// Simulated expert page bytes with `--expert-sim`. `None` is 4096.
    pub expert_bytes: Option<u64>,
    /// CUDA-like knobs for SimulatedGpuStore. Identity stays default.
    pub gpu_cfg: GpuStoreCfg,
    /// Simulated KV page bytes with `--kv-sim`. `None` uses intern geometry.
    pub kv_bytes: Option<u64>,
    /// Miss-page placement (`--expert-sim`). Default is pinned H2D.
    pub fill: GpuFill,
    /// Predictor prefetch. Default [`Prefetch::Both`].
    pub prefetch: Prefetch,
    /// Unique predicted-key Stay vs Fetch window. `0` is ungated.
    pub plan_window: usize,
    /// Stay permille of the predicted window already resident.
    pub plan_threshold: u32,
    /// Write Engine MoE traces as JSONL. `None` leaves tracing off unless [`Self::bench`].
    pub trace_out: Option<String>,
    /// Print `expertvm::report` on this run's MoE traces (`infer-bench` shape).
    pub bench: bool,
    /// Replay cache slots for [`Self::bench`]. `None` uses [`Self::expert_slots`] or 8.
    pub bench_capacity: Option<usize>,
}

impl EngineArgs {
    /// Default `--kv-page` when omitted.
    pub const DEFAULT_BLOCK_SIZE: usize = 16;
}

struct EngineSimNeed {
    expert_sim: bool,
    expert_8gpu: bool,
    has_bytes: bool,
    slo_reject: bool,
    has_ttft: bool,
    has_itl: bool,
    gpu: GpuCli,
}

fn check_engine_sim_opts(n: &EngineSimNeed) -> Result<(), String> {
    if n.expert_8gpu && !n.expert_sim {
        return engine_err("--expert-8gpu requires --expert-sim");
    }
    if n.has_bytes && !n.expert_sim {
        return engine_err("--expert-bytes requires --expert-sim");
    }
    if n.slo_reject && !n.has_ttft {
        return engine_err("--slo-reject requires --ttft-slo-ns");
    }
    if n.has_ttft && !n.slo_reject {
        return engine_err("--ttft-slo-ns requires --slo-reject");
    }
    if (n.slo_reject || n.has_ttft) && !n.expert_sim {
        return engine_err("--slo-reject requires --expert-sim");
    }
    if n.has_itl && !n.expert_sim {
        return engine_err("--itl-slo-ns requires --expert-sim");
    }
    if n.gpu.kv_bytes.is_some() && !n.gpu.kv_sim {
        return engine_err("--kv-bytes requires --kv-sim");
    }
    if let Some(flag) = n.gpu.sim_flag() {
        if !n.expert_sim {
            return engine_err(&format!("{flag} requires --expert-sim"));
        }
    }
    if n.gpu.graph_mem && n.gpu.graph_auto_free {
        return engine_err("choose one of --graph-mem, --graph-auto-free");
    }
    if n.gpu.graph_update && n.gpu.graph_set_params {
        return engine_err("choose one of --graph-update, --graph-set-params");
    }
    if n.gpu.graph_update && n.gpu.device_launch {
        return engine_err("choose one of --graph-update, --device-launch");
    }
    if n.gpu.graph_update && n.gpu.device_updatable {
        return engine_err("choose one of --graph-update, --device-updatable");
    }
    if n.gpu.device_launch && (n.gpu.graph_mem || n.gpu.graph_auto_free) {
        return engine_err("device-launch cannot graph-mem");
    }
    if n.gpu.graph_build && n.gpu.graph_piecewise {
        return engine_err("choose one of --graph-build, --graph-piecewise");
    }
    if let Err(e) = n.gpu.check_graph_build_deps() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_graph_host() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_graph_if() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_graph_memset() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_graph_memcpy() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_graph_leaf_host() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_graph_capture_deps() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_graph_capture_host() {
        return engine_err(&e);
    }
    if n.gpu.graph_enable && n.gpu.device_launch {
        return engine_err("graph-enable cannot device-launch");
    }
    if n.gpu.graph_if && n.gpu.device_launch {
        return engine_err("graph-if cannot device-launch");
    }
    if n.gpu.launch_completion && n.gpu.device_launch {
        return engine_err("launch-completion cannot device-launch");
    }
    if n.gpu.programmatic_event && n.gpu.device_launch {
        return engine_err("programmatic-event cannot device-launch");
    }
    if n.gpu.stream_attach && n.gpu.seq_streams {
        return engine_err("stream-attach cannot seq-streams");
    }
    if n.gpu.pdl && n.gpu.cooperative {
        return engine_err("choose one of --pdl, --cooperative");
    }
    if (n.gpu.shareable || n.gpu.share_ptr)
        && (n.gpu.sync_alloc
            || n.gpu.ipc
            || n.gpu.mapped
            || n.gpu.managed
            || n.gpu.vmm
            || n.gpu.vmm_retain
            || n.gpu.vmm_handle)
    {
        return engine_err("shareable needs cudaMallocAsync");
    }
    if n.gpu.mempool_trim && (n.gpu.sync_alloc || n.gpu.ipc) {
        return engine_err("mempool-trim needs cudaMallocAsync");
    }
    if n.gpu.mempool_no_reuse && (n.gpu.sync_alloc || n.gpu.ipc) {
        return engine_err("mempool-no-reuse needs cudaMallocAsync");
    }
    if n.gpu.mempool_max > 0 && (n.gpu.sync_alloc || n.gpu.ipc) {
        return engine_err("mempool-max needs cudaMallocAsync");
    }
    if n.gpu.memcpy_batch
        && (n.gpu.pageable
            || n.gpu.host_register
            || n.gpu.host_unregister
            || n.gpu.host_register_mapped
            || n.gpu.sync_alloc
            || n.gpu.ipc
            || n.gpu.sync_memops
            || n.gpu.device_sync_memops
            || n.gpu.mapped
            || n.gpu.managed)
    {
        return engine_err("memcpy-batch needs async pinned/vmm H2D");
    }
    if n.gpu.sync_memops && (n.gpu.mapped || n.gpu.host_register_mapped) {
        return engine_err("sync-memops needs device memcpy");
    }
    if n.gpu.device_sync_memops && (n.gpu.mapped || n.gpu.host_register_mapped) {
        return engine_err("device-sync-memops needs device memcpy");
    }
    if (n.gpu.host_register || n.gpu.host_unregister) && n.gpu.host_register_mapped {
        return engine_err("choose one of --host-register, --host-register-mapped");
    }
    if (n.gpu.host_register || n.gpu.host_unregister) && (n.gpu.mapped || n.gpu.managed) {
        return engine_err("host-register needs pinned/vmm H2D");
    }
    if let Err(e) = n.gpu.check_ipc() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_vmm_handle() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_preferred_cluster() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_cluster_must_set() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_required_cluster() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_mem_sync_map() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_mem_sync_launch() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_mem_sync_launch_map() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_l2_streaming() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_memcpy_during() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_memcpy_any() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_memcpy_attr() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_d2h_evict() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_d2h_pageable() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_memset_fill() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_cluster_load_balance() {
        return engine_err(&e);
    }
    if let Err(e) = n.gpu.check_max_l1() {
        return engine_err(&e);
    }
    Ok(())
}

/// Parse operands after the `engine` verb.
///
/// Path may appear before or after flags. `--flag=value` is accepted.
/// `--prompt` / `-p` may be repeated.
pub fn parse_engine_args<I, S>(args: I) -> Result<EngineCmd, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut path = None;
    let mut prompts = Vec::new();
    let mut n_predict = InferArgs::DEFAULT_N_PREDICT;
    let mut n_ctx = None;
    let mut block_size = EngineArgs::DEFAULT_BLOCK_SIZE;
    let mut pool_blocks = None;
    let mut max_seqs = None;
    let mut prefill_chunk = 0usize;
    let mut decode_first = false;
    let mut slo_reject = false;
    let mut ttft_slo_ns = None;
    let mut itl_slo_ns = None;
    let mut expert_slots = None;
    let mut expert_sim = false;
    let mut expert_8gpu = false;
    let mut expert_bytes = None;
    let mut planner = PlannerCli::default();
    let mut trace_out = None;
    let mut bench = false;
    let mut bench_capacity = None;
    let mut it = args.into_iter();
    while let Some(raw) = it.next() {
        let arg = raw.as_ref();
        if arg == "--help" || arg == "-h" {
            return Ok(EngineCmd::Help);
        }
        let (key, inline) = match arg.split_once('=') {
            Some((k, v)) => (k, Some(v)),
            None => (arg, None),
        };
        match key {
            "--prompt" | "-p" => prompts.push(engine_value("prompt", inline, &mut it)?),
            "--n-predict" | "-n" => {
                n_predict =
                    engine_usize("n-predict", &engine_value("n-predict", inline, &mut it)?)?;
            }
            "--n-ctx" => {
                let n = engine_usize("n-ctx", &engine_value("n-ctx", inline, &mut it)?)?;
                if n == 0 {
                    return engine_err("n-ctx must be > 0");
                }
                n_ctx = Some(n);
            }
            "--kv-page" => {
                let n = engine_usize("kv-page", &engine_value("kv-page", inline, &mut it)?)?;
                if n == 0 {
                    return engine_err("kv-page must be > 0");
                }
                block_size = n;
            }
            "--pool-blocks" => {
                let n = engine_usize(
                    "pool-blocks",
                    &engine_value("pool-blocks", inline, &mut it)?,
                )?;
                if n == 0 {
                    return engine_err("pool-blocks must be > 0");
                }
                pool_blocks = Some(n);
            }
            "--max-seqs" => {
                let n = engine_usize("max-seqs", &engine_value("max-seqs", inline, &mut it)?)?;
                if n == 0 {
                    return engine_err("max-seqs must be > 0");
                }
                max_seqs = Some(n);
            }
            "--prefill-chunk" => {
                prefill_chunk = engine_usize(
                    "prefill-chunk",
                    &engine_value("prefill-chunk", inline, &mut it)?,
                )?;
            }
            "--decode-first" => {
                if inline.is_some() {
                    return engine_err("--decode-first does not take a value");
                }
                decode_first = true;
            }
            "--slo-reject" => {
                if inline.is_some() {
                    return engine_err("--slo-reject does not take a value");
                }
                slo_reject = true;
            }
            "--ttft-slo-ns" => {
                let n = engine_u64(
                    "ttft-slo-ns",
                    &engine_value("ttft-slo-ns", inline, &mut it)?,
                )?;
                if n == 0 {
                    return engine_err("ttft-slo-ns must be > 0");
                }
                ttft_slo_ns = Some(n);
            }
            "--itl-slo-ns" => {
                let n = engine_u64("itl-slo-ns", &engine_value("itl-slo-ns", inline, &mut it)?)?;
                if n == 0 {
                    return engine_err("itl-slo-ns must be > 0");
                }
                itl_slo_ns = Some(n);
            }
            "--expert-slots" => {
                expert_slots = Some(engine_usize(
                    "expert-slots",
                    &engine_value("expert-slots", inline, &mut it)?,
                )?);
            }
            "--expert-sim" => {
                if inline.is_some() {
                    return engine_err("--expert-sim does not take a value");
                }
                expert_sim = true;
            }
            "--expert-8gpu" => {
                if inline.is_some() {
                    return engine_err("--expert-8gpu does not take a value");
                }
                expert_8gpu = true;
            }
            "--expert-bytes" => {
                let n = engine_u64(
                    "expert-bytes",
                    &engine_value("expert-bytes", inline, &mut it)?,
                )?;
                if n == 0 {
                    return engine_err("expert-bytes must be > 0");
                }
                expert_bytes = Some(n);
            }
            "--trace-out" => {
                trace_out = Some(engine_value("trace-out", inline, &mut it)?);
            }
            "--bench" => {
                if inline.is_some() {
                    return engine_err("--bench does not take a value");
                }
                bench = true;
            }
            "--capacity" => {
                let n = engine_usize("capacity", &engine_value("capacity", inline, &mut it)?)?;
                if n == 0 {
                    return engine_err("capacity must be > 0");
                }
                bench_capacity = Some(n);
            }
            flag if flag.starts_with('-') => match planner.take(key, inline, &mut it) {
                Ok(true) => {}
                Ok(false) => return engine_err(&format!("unknown flag {flag}")),
                Err(e) => return engine_err(&e),
            },
            other => {
                if path.is_some() {
                    return engine_err(&format!("unexpected argument {other}"));
                }
                path = Some(other.to_string());
            }
        }
    }
    let Some(path) = path else {
        return engine_err("missing GGUF path");
    };
    if prompts.is_empty() {
        prompts.push(InferArgs::DEFAULT_PROMPT.to_string());
    }
    if bench_capacity.is_some() && !bench {
        return engine_err("--capacity requires --bench");
    }
    check_engine_sim_opts(&EngineSimNeed {
        expert_sim,
        expert_8gpu,
        has_bytes: expert_bytes.is_some(),
        slo_reject,
        has_ttft: ttft_slo_ns.is_some(),
        has_itl: itl_slo_ns.is_some(),
        gpu: planner.gpu,
    })?;
    planner.gpu.imply_vmm();
    planner.gpu.imply_shareable();
    planner.gpu.imply_sync_alloc();
    planner.gpu.imply_pageable();
    planner.gpu.imply_mapped();
    planner.gpu.imply_managed();
    planner.gpu.imply_decode_priority();
    planner.gpu.check_green_ctx()?;
    planner.gpu.imply_l2_persist();
    planner.gpu.imply_timing_events();
    let fill = planner.gpu.fill()?;
    let gpu_cfg = gpu_knobs(planner.gpu);
    Ok(EngineCmd::Run(EngineArgs {
        path,
        prompts,
        n_predict,
        n_ctx,
        block_size,
        pool_blocks,
        max_seqs,
        prefill_chunk,
        decode_first,
        slo_reject,
        ttft_slo_ns,
        itl_slo_ns,
        expert_slots,
        expert_sim,
        expert_8gpu,
        expert_bytes,
        gpu_cfg,
        kv_bytes: planner.gpu.kv_bytes,
        fill,
        prefetch: planner.prefetch,
        plan_window: planner.plan_window,
        plan_threshold: planner.plan_threshold,
        trace_out,
        bench,
        bench_capacity,
    }))
}

/// Continuous batching: several prompts on one interned [`Engine`] pool.
pub fn run_engine(args: &EngineArgs) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = read_file(Path::new(&args.path))?;
    let g = load_gguf_owned(bytes)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let model = Llama::from_gguf(g)?;
    let cfg = engine_cfg(&tok, args)?;
    let mut eng = Engine::new(&model, cfg)?;
    attach_engine_store(&mut eng, &model, args)?;
    if args.trace_out.is_some() || args.bench {
        eng.enable_moe_trace();
    }
    let mut handles = Vec::new();
    for ids in engine_prompts(&tok, args)? {
        handles.push(eng.add(&ids, args.n_predict)?);
    }
    eng.run()?;
    let traces = if args.trace_out.is_some() || args.bench {
        take_engine_traces(&mut eng, &handles)
    } else {
        Trace::new()
    };
    if let Some(path) = args.trace_out.as_ref() {
        write_file(Path::new(path), traces.to_jsonl().as_bytes())?;
    }
    let mut out = std::io::stdout();
    for (i, id) in handles.iter().enumerate() {
        let seq = eng.take(*id).ok_or("engine take")?;
        let text = tok.decode(&seq.generated);
        writeln!(
            out,
            "seq={i} n_gen={} generated={text}",
            seq.generated.len()
        )?;
    }
    writeln!(
        out,
        "intern_hits={} preempts={} rejected={} itl_slo_miss={} graph_launches={} graph_updates={} graph_clones={} graph_set_params={} kv_miss={} kv_hit={} steps={} gemm_tokens={} gemm_peak={} serial_tokens={}",
        eng.pool().hits(),
        eng.preempts(),
        eng.rejected(),
        eng.itl_slo_miss(),
        eng.graph_launches(),
        eng.graph_updates(),
        eng.graph_clones(),
        eng.graph_set_params(),
        eng.kv_misses(),
        eng.kv_hits(),
        eng.stats().steps,
        eng.stats().gemm_tokens,
        eng.stats().gemm_peak,
        eng.stats().serial_tokens
    )?;
    if let Some(m) = eng.expert_store_metrics() {
        writeln!(out, "{}", m.line())?;
    }
    if let Some(score) = eng.expert_store_score()? {
        writeln!(out, "{}", score.line())?;
    }
    if args.bench {
        write!(out, "{}", engine_bench_report(&traces, args)?)?;
    }
    Ok(())
}

fn take_engine_traces(eng: &mut Engine<'_>, ids: &[SeqId]) -> Trace {
    let mut events = Vec::new();
    for id in ids {
        if let Some(t) = eng.take_moe_trace(*id) {
            events.extend(t.events);
        }
    }
    Trace { events }
}

fn engine_bench_report(
    trace: &Trace,
    args: &EngineArgs,
) -> Result<String, Box<dyn std::error::Error>> {
    let capacity = bench_capacity(args);
    let profile = if args.expert_sim {
        Some(if args.expert_8gpu {
            HardwareProfile::example_8xh100_nvlink()
        } else {
            HardwareProfile::example_h100_sxm()
        })
    } else {
        None
    };
    let row = report(
        "engine",
        trace,
        capacity,
        8,
        profile,
        args.expert_bytes.unwrap_or(4096),
    )?;
    Ok(row.render())
}

fn bench_capacity(args: &EngineArgs) -> usize {
    if let Some(n) = args.bench_capacity {
        return n;
    }
    match args.expert_slots {
        Some(0) | None => 8,
        Some(n) => n,
    }
}

fn attach_engine_store(
    eng: &mut Engine<'_>,
    llama: &Llama,
    args: &EngineArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    attach_store(
        eng,
        llama,
        &StoreAttach {
            expert_slots: args.expert_slots,
            expert_sim: args.expert_sim,
            expert_8gpu: args.expert_8gpu,
            expert_bytes: args.expert_bytes,
            gpu_cfg: args.gpu_cfg,
            fill: args.fill,
            kv_bytes: args.kv_bytes,
        },
    )?;
    Ok(())
}

fn engine_prompts(
    tok: &Tokenizer,
    args: &EngineArgs,
) -> Result<Vec<Vec<u32>>, Box<dyn std::error::Error>> {
    let mut encoded = Vec::new();
    for p in &args.prompts {
        let ids = prompt_ids(tok, p)?;
        if ids.is_empty() {
            return Err("empty prompt tokens".into());
        }
        encoded.push(ids);
    }
    Ok(encoded)
}

fn engine_cfg(tok: &Tokenizer, args: &EngineArgs) -> Result<EngineCfg, Box<dyn std::error::Error>> {
    let encoded = engine_prompts(tok, args)?;
    let mut max_needed = 0usize;
    for ids in &encoded {
        max_needed = max_needed.max(ids.len().saturating_add(args.n_predict));
    }
    let n_ctx = match args.n_ctx {
        Some(n) if n < max_needed => {
            return Err(format!("--n-ctx {n} is below the {max_needed} tokens needed").into());
        }
        Some(n) => n,
        None => max_needed.saturating_add(1),
    };
    let max_seqs = args.max_seqs.unwrap_or(encoded.len());
    let pages = n_ctx.div_ceil(args.block_size).saturating_add(1);
    let pool_blocks = args
        .pool_blocks
        .unwrap_or(max_seqs.saturating_mul(pages).max(2));
    Ok(EngineCfg {
        n_ctx,
        block_size: args.block_size,
        pool_blocks,
        max_seqs,
        prefill_chunk: args.prefill_chunk,
        eos: tok.eos,
        decode_first: args.decode_first,
        slo_reject: args.slo_reject,
        ttft_slo_ns: args.ttft_slo_ns,
        itl_slo_ns: args.itl_slo_ns,
        prefetch: args.prefetch,
        plan_window: args.plan_window,
        plan_threshold: args.plan_threshold,
    })
}

/// Hold a conversation using the model's own `tokenizer.chat_template`.
///
/// With `--prompt` this is one turn and exits. Otherwise each line of stdin is
/// a user turn, and replies accumulate so the model sees the whole history.
/// Every turn re-renders the template. Prefill uses [`crate::decode::Llama::prompt`]
/// so a matching token prefix keeps its KV. Templates that rewrite earlier
/// turns simply get a shorter hit. `--n-ctx` sizes the persistent cache;
/// `--kv-page N` stores that cache in interned blocks. Omit `--n-ctx` and
/// the cache is reallocated when a turn no longer fits.
pub fn run_chat(args: &ChatArgs) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = read_file(Path::new(&args.path))?;
    let g = load_gguf_owned(bytes)?;
    let tok = Tokenizer::from_gguf(&g)?;
    if tok.chat_template.is_none() {
        return Err(format!(
            "{} has no tokenizer.chat_template; use `gguf_gemv infer` with a raw prompt",
            args.path
        )
        .into());
    }
    let model = Llama::from_gguf(g)?;
    let mut history: Vec<ChatMessage> = Vec::new();
    if let Some(s) = &args.system {
        history.push(ChatMessage::system(s));
    }
    let mut out = std::io::stdout();
    let mut cache = None;
    if let Some(p) = &args.prompt {
        return chat_turn(&model, &tok, args, &mut history, p, &mut out, &mut cache);
    }
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        write!(out, "> ")?;
        out.flush()?;
        line.clear();
        if stdin.lock().read_line(&mut line)? == 0 {
            writeln!(out)?;
            return Ok(());
        }
        let turn = line.trim_end_matches(['\n', '\r']);
        if turn.is_empty() {
            continue;
        }
        chat_turn(&model, &tok, args, &mut history, turn, &mut out, &mut cache)?;
    }
}

fn chat_turn<W: Write>(
    model: &Llama,
    tok: &Tokenizer,
    args: &ChatArgs,
    history: &mut Vec<ChatMessage>,
    turn: &str,
    out: &mut W,
    cache: &mut Option<KvCache>,
) -> Result<(), Box<dyn std::error::Error>> {
    history.push(ChatMessage::user(turn));
    let prompt = tok.apply_chat_template(history, true)?;
    if args.show_prompt {
        writeln!(out, "--- rendered prompt ---\n{prompt}\n--- end ---")?;
    }
    let reply = generate_reply(model, tok, &prompt, args, cache)?;
    writeln!(out, "{}", reply.trim())?;
    history.push(ChatMessage::assistant(reply.trim()));
    Ok(())
}

/// Greedy-decode one assistant turn and return only the new text.
///
/// [`greedy_generate_ctx`] decodes the prompt and the continuation together,
/// which is what `infer` wants but not what a conversation wants: the reply has
/// to go into the history on its own, and the prompt is full of markers
/// [`Tokenizer::decode`] drops.
fn generate_reply(
    model: &Llama,
    tok: &Tokenizer,
    prompt: &str,
    args: &ChatArgs,
    cache: &mut Option<KvCache>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut ids = tok.encode(prompt)?;
    if tok.add_bos {
        if let Some(bos) = tok.bos {
            if ids.first().copied() != Some(bos) {
                ids.insert(0, bos);
            }
        }
    }
    if ids.is_empty() {
        return Err("the chat template rendered an empty prompt".into());
    }
    let needed = ids.len().saturating_add(args.n_predict);
    if let Some(n) = args.n_ctx {
        if n < needed {
            return Err(format!("--n-ctx {n} is below the {needed} tokens this turn needs").into());
        }
    }
    let kv = model.ensure_cache_page(cache, needed, args.n_ctx, args.kv_page)?;
    let mut logits = model.prompt(kv, &ids)?;
    let mut reply = Vec::new();
    for _ in 0..args.n_predict {
        let next = argmax(&logits);
        if ends_turn(tok, next) {
            break;
        }
        reply.push(next);
        logits = model.forward(kv, next)?;
    }
    Ok(tok.decode(&reply))
}

/// Whether `id` ends the assistant's turn.
///
/// `eos` alone is not enough. An instruct model closes its turn with whatever
/// marker its own template uses (`<|im_end|>`, `<|eot_id|>`, `<end_of_turn>`),
/// and in several families that is a different id from `eos`. llama.cpp keeps
/// an explicit end-of-generation set for this; GGUF does not always carry the
/// flags, so we use the rule that holds regardless: a control token in the
/// output stream means the reply is over, because control tokens never belong
/// inside one.
fn ends_turn(tok: &Tokenizer, id: u32) -> bool {
    tok.eos == Some(id) || tok.is_special(id)
}

/// Read a file with `File` + `Read`, not `std::fs::read`.
fn write_file(path: &Path, bytes: &[u8]) -> Result<(), std::io::Error> {
    let mut f = File::create(path)?;
    f.write_all(bytes)?;
    Ok(())
}

fn read_file(path: &Path) -> Result<Vec<u8>, std::io::Error> {
    let mut f = File::open(path)?;
    let mut buf = Vec::new();
    let _n = f.read_to_end(&mut buf)?;
    Ok(buf)
}

fn chat_usage_err<T>(msg: &str) -> Result<T, String> {
    Err(format!("{msg}\n{CHAT_USAGE}"))
}

fn chat_value<I, S>(name: &str, inline: Option<&str>, it: &mut I) -> Result<String, String>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    if let Some(v) = inline {
        return Ok(v.to_string());
    }
    match it.next() {
        Some(s) => Ok(s.as_ref().to_string()),
        None => chat_usage_err(&format!("missing --{name} value")),
    }
}

fn engine_err<T>(msg: &str) -> Result<T, String> {
    Err(format!("{msg}\n{ENGINE_USAGE}"))
}

fn engine_value<I, S>(name: &str, inline: Option<&str>, it: &mut I) -> Result<String, String>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    if let Some(v) = inline {
        return Ok(v.to_string());
    }
    match it.next() {
        Some(s) => Ok(s.as_ref().to_string()),
        None => engine_err(&format!("missing --{name} value")),
    }
}

fn engine_usize(name: &str, s: &str) -> Result<usize, String> {
    s.parse::<usize>()
        .map_err(|_| format!("invalid {name} {s:?}\n{ENGINE_USAGE}"))
}

fn engine_u64(name: &str, s: &str) -> Result<u64, String> {
    s.parse::<u64>()
        .map_err(|_| format!("invalid {name} {s:?}\n{ENGINE_USAGE}"))
}

fn usage_err<T>(msg: &str) -> Result<T, String> {
    Err(format!("{msg}\n{INFER_USAGE}"))
}

fn trace_usage_err<T>(msg: &str) -> Result<T, String> {
    Err(format!("{msg}\n{TRACE_USAGE}"))
}

fn trace_value<I, S>(name: &str, inline: Option<&str>, it: &mut I) -> Result<String, String>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    if let Some(v) = inline {
        return Ok(v.to_string());
    }
    match it.next() {
        Some(s) => Ok(s.as_ref().to_string()),
        None => trace_usage_err(&format!("missing --{name} value")),
    }
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
        .map_err(|_| format!("invalid {name} {s:?}\n{INFER_USAGE}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use expertvm::{
        MemSyncDomain, PortableClusterMode, PortableSharedMode, SharedMemoryMode,
        SynchronizationPolicy,
    };

    fn run(args: &[&str]) -> InferArgs {
        match parse_infer_args(args).expect("parse") {
            InferCmd::Run(a) => a,
            InferCmd::Help => panic!("expected Run"),
        }
    }

    #[test]
    fn omitted_flags_keep_shipped_ab_and_two() {
        let a = run(&["tiny.gguf"]);
        assert_eq!(a.path, "tiny.gguf");
        assert_eq!(a.prompt, InferArgs::DEFAULT_PROMPT);
        assert_eq!(a.n_predict, InferArgs::DEFAULT_N_PREDICT);
        assert_eq!(a.n_ctx, None);
        assert_eq!(a.prompt, "ab");
        assert_eq!(a.n_predict, 2);
    }

    #[test]
    fn long_flags_and_path_after() {
        let a = run(&[
            "--prompt",
            "a",
            "--n-predict",
            "4",
            "--n-ctx",
            "16",
            "m.gguf",
        ]);
        assert_eq!(
            a,
            InferArgs {
                path: "m.gguf".into(),
                prompt: "a".into(),
                n_predict: 4,
                n_ctx: Some(16),
            }
        );
    }

    #[test]
    fn short_flags_equals_and_path_first() {
        let a = run(&["model.gguf", "-p=hello", "-n=0", "--n-ctx=8"]);
        assert_eq!(a.path, "model.gguf");
        assert_eq!(a.prompt, "hello");
        assert_eq!(a.n_predict, 0);
        assert_eq!(a.n_ctx, Some(8));
    }

    #[test]
    fn help_is_not_a_run() {
        assert_eq!(parse_infer_args(["--help"]).unwrap(), InferCmd::Help);
        assert_eq!(parse_infer_args(["-h", "x.gguf"]).unwrap(), InferCmd::Help);
    }

    #[test]
    fn missing_path_and_bad_values_error() {
        let err = parse_infer_args(["--prompt", "hi"]).unwrap_err();
        assert!(err.contains("missing GGUF path"), "{err}");
        assert!(err.contains("gguf_gemv infer"), "{err}");
        let err = parse_infer_args(["m.gguf", "-n", "x"]).unwrap_err();
        assert!(err.contains("invalid n-predict"), "{err}");
        let err = parse_infer_args(["m.gguf", "--n-ctx", "0"]).unwrap_err();
        assert!(err.contains("n-ctx must be > 0"), "{err}");
        let err = parse_infer_args(["m.gguf", "--nope"]).unwrap_err();
        assert!(err.contains("unknown flag"), "{err}");
        let err = parse_infer_args(["a.gguf", "b.gguf"]).unwrap_err();
        assert!(err.contains("unexpected argument"), "{err}");
        let err = parse_infer_args(["--prompt"]).unwrap_err();
        assert!(err.contains("missing --prompt value"), "{err}");
    }

    fn chat(args: &[&str]) -> ChatArgs {
        match parse_chat_args(args).expect("parse") {
            ChatCmd::Run(a) => a,
            ChatCmd::Help => panic!("expected Run"),
        }
    }

    #[test]
    fn chat_defaults_to_stdin_turns_and_no_system_message() {
        let a = chat(&["model.gguf"]);
        assert_eq!(a.path, "model.gguf");
        assert_eq!(a.system, None);
        assert_eq!(a.prompt, None);
        assert_eq!(a.n_predict, ChatArgs::DEFAULT_N_PREDICT);
        assert_eq!(a.n_predict, 64);
        assert_eq!(a.n_ctx, None);
        assert_eq!(a.kv_page, None);
        assert!(!a.show_prompt);
    }

    #[test]
    fn chat_takes_system_prompt_and_flags_in_any_order() {
        let a = chat(&[
            "--system",
            "You are terse.",
            "-p=Hi",
            "m.gguf",
            "--n-predict",
            "8",
            "--n-ctx=32",
            "--kv-page=2",
            "--show-prompt",
        ]);
        assert_eq!(
            a,
            ChatArgs {
                path: "m.gguf".into(),
                system: Some("You are terse.".into()),
                prompt: Some("Hi".into()),
                n_predict: 8,
                n_ctx: Some(32),
                kv_page: Some(2),
                show_prompt: true,
            }
        );
        let a = chat(&["m.gguf", "-s", "sys", "-n", "1"]);
        assert_eq!(a.system.as_deref(), Some("sys"));
        assert_eq!(a.n_predict, 1);
    }

    #[test]
    fn chat_help_and_bad_arguments_do_not_run() {
        assert_eq!(parse_chat_args(["--help"]).unwrap(), ChatCmd::Help);
        assert_eq!(parse_chat_args(["-h", "m.gguf"]).unwrap(), ChatCmd::Help);
        let err = parse_chat_args(["--system", "s"]).unwrap_err();
        assert!(err.contains("missing GGUF path"), "{err}");
        assert!(err.contains("gguf_gemv chat"), "{err}");
        let err = parse_chat_args(["m.gguf", "-n", "x"]).unwrap_err();
        assert!(err.contains("invalid n-predict"), "{err}");
        let err = parse_chat_args(["m.gguf", "--n-ctx", "0"]).unwrap_err();
        assert!(err.contains("n-ctx must be > 0"), "{err}");
        let err = parse_chat_args(["m.gguf", "--kv-page", "0"]).unwrap_err();
        assert!(err.contains("kv-page must be > 0"), "{err}");
        let err = parse_chat_args(["m.gguf", "--nope"]).unwrap_err();
        assert!(err.contains("unknown flag"), "{err}");
        let err = parse_chat_args(["a.gguf", "b.gguf"]).unwrap_err();
        assert!(err.contains("unexpected argument"), "{err}");
        let err = parse_chat_args(["m.gguf", "--system"]).unwrap_err();
        assert!(err.contains("missing --system value"), "{err}");
    }

    #[test]
    fn any_control_token_ends_the_turn_not_just_eos() {
        use crate::gguf::{write_gguf_with_kv, Kv};
        use crate::load_gguf;
        // Llama-3's shape: `eos` is `<|end_of_text|>` but an instruct turn ends
        // with `<|eot_id|>`, a different id.
        let kv = vec![
            (
                "tokenizer.ggml.tokens".to_string(),
                Kv::Array {
                    elem: 8,
                    items: ["a", "<|end_of_text|>", "<|eot_id|>", "b"]
                        .into_iter()
                        .map(|s| Kv::String(s.into()))
                        .collect(),
                },
            ),
            (
                "tokenizer.ggml.token_type".to_string(),
                Kv::Array {
                    elem: 5,
                    items: vec![Kv::I32(1), Kv::I32(3), Kv::I32(3), Kv::I32(1)],
                },
            ),
            ("tokenizer.ggml.eos_token_id".to_string(), Kv::U32(1)),
        ];
        let bytes = write_gguf_with_kv(&kv, &[]);
        let g = load_gguf(&bytes).expect("gguf");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        assert!(!ends_turn(&tok, 0), "ordinary text must not stop the turn");
        assert!(!ends_turn(&tok, 3));
        assert!(ends_turn(&tok, 1), "eos");
        assert!(
            ends_turn(&tok, 2),
            "eot_id is not eos but still ends the turn"
        );
    }

    #[test]
    fn trace_requires_out_and_path() {
        let err = parse_trace_args(["m.gguf"]).unwrap_err();
        assert!(err.contains("missing --out"), "{err}");
        assert!(err.contains("gguf_gemv trace"), "{err}");
        match parse_trace_args(["m.gguf", "--out", "t.jsonl"]).expect("ok") {
            TraceCmd::Run(a) => {
                assert_eq!(a.out, "t.jsonl");
                assert_eq!(a.infer.path, "m.gguf");
                assert_eq!(a.infer.prompt, InferArgs::DEFAULT_PROMPT);
                assert_eq!(a.capacity, 8);
            }
            TraceCmd::Help => panic!("expected Run"),
        }
        assert_eq!(parse_trace_args(["--help"]).unwrap(), TraceCmd::Help);
        match parse_trace_args(["m.gguf", "--out", "t.jsonl", "-c", "2"]).expect("cap") {
            TraceCmd::Run(a) => assert_eq!(a.capacity, 2),
            TraceCmd::Help => panic!("expected Run"),
        }
    }

    #[test]
    fn engine_repeatable_prompts_and_rejects_bad_flags() {
        match parse_engine_args(["m.gguf"]).expect("def") {
            EngineCmd::Run(a) => {
                assert_eq!(a.prompts, vec![String::from("ab")]);
                assert_eq!(a.n_predict, 2);
                assert_eq!(a.block_size, EngineArgs::DEFAULT_BLOCK_SIZE);
                assert_eq!(a.pool_blocks, None);
                assert_eq!(a.prefill_chunk, 0);
                assert!(!a.decode_first);
                assert!(!a.slo_reject);
                assert_eq!(a.ttft_slo_ns, None);
                assert_eq!(a.itl_slo_ns, None);
                assert_eq!(a.expert_slots, None);
                assert!(!a.expert_sim);
                assert!(!a.expert_8gpu);
                assert_eq!(a.expert_bytes, None);
                assert_eq!(a.kv_bytes, None);
                assert!(!a.gpu_cfg.kv_sim);
                assert_eq!(a.gpu_cfg, GpuStoreCfg::default());
                assert_eq!(a.fill, GpuFill::Pinned);
                assert_eq!(a.prefetch, Prefetch::Both);
                assert_eq!(a.plan_window, 0);
                assert_eq!(a.plan_threshold, 500);
                assert_eq!(a.trace_out, None);
                assert!(!a.bench);
                assert_eq!(a.bench_capacity, None);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["-p", "a", "-p", "b", "m.gguf", "--kv-page", "2"]).expect("two") {
            EngineCmd::Run(a) => {
                assert_eq!(a.prompts, vec![String::from("a"), String::from("b")]);
                assert_eq!(a.block_size, 2);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-slots", "0"]).expect("direct") {
            EngineCmd::Run(a) => assert_eq!(a.expert_slots, Some(0)),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-slots=8"]).expect("cached") {
            EngineCmd::Run(a) => assert_eq!(a.expert_slots, Some(8)),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim"]).expect("sim") {
            EngineCmd::Run(a) => {
                assert!(a.expert_sim);
                assert!(!a.expert_8gpu);
                assert_eq!(a.expert_bytes, None);
                assert_eq!(a.expert_slots, None);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--expert-8gpu",
            "--expert-bytes",
            "64",
        ])
        .expect("8gpu")
        {
            EngineCmd::Run(a) => {
                assert!(a.expert_sim);
                assert!(a.expert_8gpu);
                assert_eq!(a.expert_bytes, Some(64));
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--expert-bytes=128"]).expect("eq") {
            EngineCmd::Run(a) => assert_eq!(a.expert_bytes, Some(128)),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--trace-out", "t.jsonl"]).expect("trace") {
            EngineCmd::Run(a) => assert_eq!(a.trace_out.as_deref(), Some("t.jsonl")),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--trace-out=out.jsonl"]).expect("eq") {
            EngineCmd::Run(a) => assert_eq!(a.trace_out.as_deref(), Some("out.jsonl")),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--bench"]).expect("bench") {
            EngineCmd::Run(a) => {
                assert!(a.bench);
                assert_eq!(a.bench_capacity, None);
                assert_eq!(bench_capacity(&a), 8);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--bench", "--capacity", "2"]).expect("cap") {
            EngineCmd::Run(a) => {
                assert!(a.bench);
                assert_eq!(a.bench_capacity, Some(2));
                assert_eq!(bench_capacity(&a), 2);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--bench", "--capacity=4"]).expect("eqcap") {
            EngineCmd::Run(a) => assert_eq!(a.bench_capacity, Some(4)),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--bench", "--expert-slots", "3"]).expect("slots") {
            EngineCmd::Run(a) => assert_eq!(bench_capacity(&a), 3),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--bench", "--expert-slots", "0"]).expect("direct") {
            EngineCmd::Run(a) => assert_eq!(bench_capacity(&a), 8),
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--capacity", "2"]).unwrap_err();
        assert!(err.contains("--capacity requires --bench"), "{err}");
        let err = parse_engine_args(["m.gguf", "--bench", "--capacity", "0"]).unwrap_err();
        assert!(err.contains("capacity must be > 0"), "{err}");
        match parse_engine_args(["m.gguf", "--decode-first"]).expect("df") {
            EngineCmd::Run(a) => assert!(a.decode_first),
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--decode-first=1"]).unwrap_err();
        assert!(
            err.contains("--decode-first does not take a value"),
            "{err}"
        );
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--slo-reject",
            "--ttft-slo-ns",
            "1",
        ])
        .expect("slo")
        {
            EngineCmd::Run(a) => {
                assert!(a.slo_reject);
                assert_eq!(a.ttft_slo_ns, Some(1));
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--slo-reject"]).unwrap_err();
        assert!(err.contains("--slo-reject requires --ttft-slo-ns"), "{err}");
        let err = parse_engine_args(["m.gguf", "--ttft-slo-ns", "1"]).unwrap_err();
        assert!(err.contains("--ttft-slo-ns requires --slo-reject"), "{err}");
        let err = parse_engine_args(["m.gguf", "--slo-reject", "--ttft-slo-ns=1"]).unwrap_err();
        assert!(err.contains("--slo-reject requires --expert-sim"), "{err}");
        let err = parse_engine_args(["m.gguf", "--slo-reject=1"]).unwrap_err();
        assert!(err.contains("--slo-reject does not take a value"), "{err}");
        assert_eq!(parse_engine_args(["--help"]).unwrap(), EngineCmd::Help);
        let err = parse_engine_args(["--n-predict", "2"]).unwrap_err();
        assert!(err.contains("missing GGUF path"), "{err}");
        let err = parse_engine_args(["m.gguf", "--pool-blocks", "0"]).unwrap_err();
        assert!(err.contains("pool-blocks must be > 0"), "{err}");
        let err = parse_engine_args(["m.gguf", "--max-seqs", "0"]).unwrap_err();
        assert!(err.contains("max-seqs must be > 0"), "{err}");
        let err = parse_engine_args(["m.gguf", "--nope"]).unwrap_err();
        assert!(err.contains("unknown flag"), "{err}");
        let err = parse_engine_args(["m.gguf", "--trace-out"]).unwrap_err();
        assert!(err.contains("missing --trace-out value"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-8gpu"]).unwrap_err();
        assert!(err.contains("--expert-8gpu requires --expert-sim"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-bytes", "64"]).unwrap_err();
        assert!(
            err.contains("--expert-bytes requires --expert-sim"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--expert-bytes", "0"]).unwrap_err();
        assert!(err.contains("expert-bytes must be > 0"), "{err}");
    }

    #[test]
    fn engine_graph_and_itl_flags_need_expert_sim() {
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--cuda-graphs",
            "--graph-update",
            "--graph-clone",
            "--graph-build",
            "--graph-mem",
            "--graph-mem-trim",
            "--timing-events",
            "--itl-slo-ns",
            "8",
        ])
        .expect("graphs")
        {
            EngineCmd::Run(a) => {
                assert!(a.expert_sim);
                assert!(a.gpu_cfg.graph_update);
                assert!(a.gpu_cfg.graph_clone);
                assert!(a.gpu_cfg.graph_build);
                assert!(a.gpu_cfg.graph_mem);
                assert!(a.gpu_cfg.graph_mem_trim);
                assert!(a.gpu_cfg.timing_events);
                assert_eq!(a.itl_slo_ns, Some(8));
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--itl-slo-ns", "1"]).unwrap_err();
        assert!(err.contains("--itl-slo-ns requires --expert-sim"), "{err}");
        let err = parse_engine_args(["m.gguf", "--cuda-graphs"]).unwrap_err();
        assert!(err.contains("--cuda-graphs requires --expert-sim"), "{err}");
        let err = parse_engine_args(["m.gguf", "--graph-update"]).unwrap_err();
        assert!(
            err.contains("--graph-update requires --expert-sim"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--graph-set-params"]).unwrap_err();
        assert!(
            err.contains("--graph-set-params requires --expert-sim"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--graph-clone"]).unwrap_err();
        assert!(err.contains("--graph-clone requires --expert-sim"), "{err}");
        let err = parse_engine_args(["m.gguf", "--graph-clone-parent"]).unwrap_err();
        assert!(
            err.contains("--graph-clone-parent requires --expert-sim"),
            "{err}"
        );
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--graph-clone-parent=1"]).unwrap_err();
        assert!(
            err.contains("--graph-clone-parent does not take a value"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--graph-clone-parent"]).expect("gcp") {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_clone_parent);
                assert!(!a.gpu_cfg.graph_clone);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--graph-clone-parent", "--pdl"])
            .expect("gcp+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_clone_parent);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-clone-parent",
            "--cooperative",
        ])
        .expect("gcp+coop")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_clone_parent);
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        };
        let err = parse_engine_args(["m.gguf", "--graph-build"]).unwrap_err();
        assert!(err.contains("--graph-build requires --expert-sim"), "{err}");
        let err = parse_engine_args(["m.gguf", "--graph-build-deps"]).unwrap_err();
        assert!(
            err.contains("--graph-build-deps requires --expert-sim"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--graph-build-deps"]).unwrap_err();
        assert!(
            err.contains("--graph-build-deps needs --graph-build"),
            "{err}"
        );
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--graph-build-deps=1"]).unwrap_err();
        assert!(
            err.contains("--graph-build-deps does not take a value"),
            "{err}"
        );
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-build",
            "--graph-build-deps",
        ])
        .expect("gbd")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_build);
                assert!(a.gpu_cfg.graph_build_deps);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-build",
            "--graph-build-deps",
            "--pdl",
        ])
        .expect("gbd+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_build_deps);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-build",
            "--graph-build-deps",
            "--cooperative",
        ])
        .expect("gbd+coop")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_build_deps);
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--graph-host"]).unwrap_err();
        assert!(err.contains("--graph-host requires --expert-sim"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--graph-host"]).unwrap_err();
        assert!(err.contains("--graph-host needs --graph-build"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--graph-host=1"]).unwrap_err();
        assert!(err.contains("--graph-host does not take a value"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--graph-build", "--graph-host"])
            .expect("gh")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_build);
                assert!(a.gpu_cfg.graph_host);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-build",
            "--graph-host",
            "--pdl",
        ])
        .expect("gh+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_host);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-build",
            "--graph-host",
            "--cooperative",
        ])
        .expect("gh+coop")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_host);
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--graph-memset"]).unwrap_err();
        assert!(
            err.contains("--graph-memset requires --expert-sim"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--graph-memset"]).unwrap_err();
        assert!(err.contains("--graph-memset needs --graph-mem"), "{err}");
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-auto-free",
            "--graph-memset",
        ])
        .unwrap_err();
        assert!(err.contains("--graph-memset needs --graph-mem"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--graph-memset=1"]).unwrap_err();
        assert!(
            err.contains("--graph-memset does not take a value"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--graph-mem", "--graph-memset"])
            .expect("gmz")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_mem);
                assert!(a.gpu_cfg.graph_memset);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-mem",
            "--graph-memset",
            "--pdl",
        ])
        .expect("gmz+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_memset);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-mem",
            "--graph-memset",
            "--cooperative",
        ])
        .expect("gmz+coop")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_memset);
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--graph-memcpy"]).unwrap_err();
        assert!(
            err.contains("--graph-memcpy requires --expert-sim"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--graph-memcpy"]).unwrap_err();
        assert!(err.contains("--graph-memcpy needs --graph-mem"), "{err}");
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-auto-free",
            "--graph-memcpy",
        ])
        .unwrap_err();
        assert!(err.contains("--graph-memcpy needs --graph-mem"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--graph-memcpy=1"]).unwrap_err();
        assert!(
            err.contains("--graph-memcpy does not take a value"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--graph-mem", "--graph-memcpy"])
            .expect("gmc")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_mem);
                assert!(a.gpu_cfg.graph_memcpy);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-mem",
            "--graph-memcpy",
            "--pdl",
        ])
        .expect("gmc+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_memcpy);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-mem",
            "--graph-memcpy",
            "--cooperative",
        ])
        .expect("gmc+coop")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_memcpy);
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--graph-leaf-host"]).unwrap_err();
        assert!(
            err.contains("--graph-leaf-host requires --expert-sim"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--graph-leaf-host=1"]).unwrap_err();
        assert!(
            err.contains("--graph-leaf-host does not take a value"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--graph-leaf-host"]).expect("glh") {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_leaf_host);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--graph-leaf-host", "--pdl"])
            .expect("glh+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_leaf_host);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-leaf-host",
            "--cooperative",
        ])
        .expect("glh+coop")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_leaf_host);
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-leaf-host",
            "--device-launch",
        ])
        .unwrap_err();
        assert!(
            err.contains("graph-leaf-host cannot device-launch"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--graph-piecewise"]).unwrap_err();
        assert!(
            err.contains("--graph-piecewise requires --expert-sim"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--graph-mem-trim"]).unwrap_err();
        assert!(
            err.contains("--graph-mem-trim requires --expert-sim"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--graph-auto-free"]).unwrap_err();
        assert!(
            err.contains("--graph-auto-free requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--graph-auto-free"]).expect("af") {
            EngineCmd::Run(a) => assert!(a.gpu_cfg.graph_auto_free),
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--graph-mem", "--graph-auto-free"])
            .unwrap_err();
        assert!(
            err.contains("choose one of --graph-mem, --graph-auto-free"),
            "{err}"
        );
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-update",
            "--graph-set-params",
        ])
        .unwrap_err();
        assert!(
            err.contains("choose one of --graph-update, --graph-set-params"),
            "{err}"
        );
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-build",
            "--graph-piecewise",
        ])
        .unwrap_err();
        assert!(
            err.contains("choose one of --graph-build, --graph-piecewise"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--graph-piecewise"]).expect("pw") {
            EngineCmd::Run(a) => assert!(a.gpu_cfg.graph_piecewise),
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--graph-capture-deps"]).unwrap_err();
        assert!(
            err.contains("--graph-capture-deps requires --expert-sim"),
            "{err}"
        );
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--graph-capture-deps"]).unwrap_err();
        assert!(
            err.contains("--graph-capture-deps needs --graph-piecewise"),
            "{err}"
        );
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--cuda-graphs",
            "--graph-capture-deps",
        ])
        .unwrap_err();
        assert!(
            err.contains("--graph-capture-deps needs --graph-piecewise"),
            "{err}"
        );
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-build",
            "--graph-capture-deps",
        ])
        .unwrap_err();
        assert!(
            err.contains("--graph-capture-deps needs --graph-piecewise"),
            "{err}"
        );
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--graph-capture-deps=1"]).unwrap_err();
        assert!(
            err.contains("--graph-capture-deps does not take a value"),
            "{err}"
        );
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-piecewise",
            "--graph-capture-deps",
        ])
        .expect("gcd")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_piecewise);
                assert!(a.gpu_cfg.graph_capture_deps);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-piecewise",
            "--graph-capture-deps",
            "--pdl",
        ])
        .expect("gcd+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_capture_deps);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-piecewise",
            "--graph-capture-deps",
            "--cooperative",
        ])
        .expect("gcd+coop")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_capture_deps);
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--graph-capture-host"]).unwrap_err();
        assert!(
            err.contains("--graph-capture-host requires --expert-sim"),
            "{err}"
        );
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--graph-capture-host"]).unwrap_err();
        assert!(
            err.contains("--graph-capture-host needs --graph-piecewise"),
            "{err}"
        );
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--cuda-graphs",
            "--graph-capture-host",
        ])
        .unwrap_err();
        assert!(
            err.contains("--graph-capture-host needs --graph-piecewise"),
            "{err}"
        );
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-build",
            "--graph-capture-host",
        ])
        .unwrap_err();
        assert!(
            err.contains("--graph-capture-host needs --graph-piecewise"),
            "{err}"
        );
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--graph-capture-host=1"]).unwrap_err();
        assert!(
            err.contains("--graph-capture-host does not take a value"),
            "{err}"
        );
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-piecewise",
            "--graph-capture-host",
        ])
        .expect("gch")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_piecewise);
                assert!(a.gpu_cfg.graph_capture_host);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-piecewise",
            "--graph-capture-host",
            "--pdl",
        ])
        .expect("gch+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_capture_host);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-piecewise",
            "--graph-capture-host",
            "--cooperative",
        ])
        .expect("gch+coop")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_capture_host);
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-piecewise",
            "--graph-capture-deps",
            "--graph-capture-host",
        ])
        .expect("gcd+gch")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_capture_deps);
                assert!(a.gpu_cfg.graph_capture_host);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--graph-enable"]).unwrap_err();
        assert!(
            err.contains("--graph-enable requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--graph-enable"]).expect("ge") {
            EngineCmd::Run(a) => assert!(a.gpu_cfg.graph_enable),
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--graph-if"]).unwrap_err();
        assert!(err.contains("--graph-if requires --expert-sim"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--graph-if"]).unwrap_err();
        assert!(err.contains("--graph-if needs --graph-build"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--graph-build", "--graph-if"])
            .expect("gif")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.graph_if);
                assert!(a.gpu_cfg.graph_build);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-enable",
            "--graph-if",
            "--graph-build",
        ])
        .unwrap_err();
        assert!(
            err.contains("choose one of --graph-if, --graph-enable"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--graph-set-params"]).expect("sp") {
            EngineCmd::Run(a) => assert!(a.gpu_cfg.graph_set_params),
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--timing-events"]).unwrap_err();
        assert!(
            err.contains("--timing-events requires --expert-sim"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--graph-update=1"]).unwrap_err();
        assert!(
            err.contains("--graph-update does not take a value"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--managed"]).expect("um") {
            EngineCmd::Run(a) => assert_eq!(a.fill, GpuFill::Managed),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--mapped"]).expect("map") {
            EngineCmd::Run(a) => assert_eq!(a.fill, GpuFill::Mapped),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--vmm"]).expect("vmm") {
            EngineCmd::Run(a) => assert_eq!(a.fill, GpuFill::Vmm),
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--managed"]).unwrap_err();
        assert!(err.contains("--managed requires --expert-sim"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--mapped", "--vmm"]).unwrap_err();
        assert!(err.contains("choose one of mapped, managed, vmm"), "{err}");
    }

    #[test]
    fn engine_gpu_cfg_flags_need_expert_sim() {
        for flag in [
            "--host-func",
            "--blocking-streams",
            "--sync-alloc",
            "--mempool",
            "--mempool-trim",
            "--mempool-no-reuse",
            "--shareable",
            "--ipc",
            "--share-ptr",
            "--vmm-retain",
            "--vmm-handle",
            "--pageable",
            "--host-register",
            "--host-unregister",
            "--memcpy-batch",
            "--memcpy-during",
            "--memcpy-any",
            "--memcpy-attr",
            "--memset-fill",
            "--copy-host",
            "--accessed-by",
            "--no-read-mostly",
            "--no-preferred",
            "--no-mem-prefetch",
            "--legacy-null",
            "--stream-priority",
            "--seq-streams",
            "--kv-sim",
            "--decode-priority",
            "--cooperative",
            "--pdl",
            "--l2-streaming",
            "--graph-capture-deps",
            "--graph-capture-host",
            "--graph-build-deps",
            "--graph-host",
            "--graph-memset",
            "--graph-memcpy",
            "--graph-leaf-host",
            "--launch-completion",
            "--programmatic-event",
            "--stream-attach",
            "--managed-host",
            "--prefetch-host",
            "--d2h-evict",
            "--d2h-pageable",
            "--host-register-mapped",
            "--sync-memops",
            "--device-sync-memops",
            "--wait-value",
            "--multicast",
            "--mem-sync-launch",
            "--mem-sync-launch-map",
        ] {
            let err = parse_engine_args(["m.gguf", flag]).unwrap_err();
            assert!(
                err.contains(&format!("{flag} requires --expert-sim")),
                "{flag}: {err}"
            );
        }
        let err = parse_engine_args(["m.gguf", "--vmm-page", "1024"]).unwrap_err();
        assert!(err.contains("--vmm-page requires --expert-sim"), "{err}");
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--host-func",
            "--blocking-streams",
            "--sync-alloc",
            "--mempool",
            "--pageable",
            "--legacy-null",
            "--stream-priority",
            "--seq-streams",
            "--vmm-page",
            "1024",
        ])
        .expect("cfg")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.host_func);
                assert!(a.gpu_cfg.blocking_streams);
                assert!(a.gpu_cfg.sync_alloc);
                assert!(a.gpu_cfg.mempool);
                assert!(a.gpu_cfg.pageable);
                assert!(a.gpu_cfg.legacy_null);
                assert!(a.gpu_cfg.stream_priority);
                assert!(a.gpu_cfg.seq_streams);
                assert_eq!(a.gpu_cfg.vmm_page, 1024);
                assert_eq!(a.fill, GpuFill::Vmm);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--managed", "--accessed-by"])
            .expect("um")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.accessed_by);
                assert_eq!(a.fill, GpuFill::Managed);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--mapped", "--vmm-page", "1024"])
            .unwrap_err();
        assert!(err.contains("choose one of mapped, managed, vmm"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--host-func=1"]).unwrap_err();
        assert!(err.contains("--host-func does not take a value"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--seq-streams=1"]).unwrap_err();
        assert!(err.contains("--seq-streams does not take a value"), "{err}");
    }

    #[test]
    fn engine_kv_sim_flags() {
        let err = parse_engine_args(["m.gguf", "--kv-bytes", "1048576"]).unwrap_err();
        assert!(err.contains("--kv-bytes requires --kv-sim"), "{err}");
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--kv-bytes", "1048576"]).unwrap_err();
        assert!(err.contains("--kv-bytes requires --kv-sim"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--kv-sim=1"]).unwrap_err();
        assert!(err.contains("--kv-sim does not take a value"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--kv-sim", "--kv-bytes", "0"])
            .unwrap_err();
        assert!(err.contains("kv-bytes must be > 0"), "{err}");
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--kv-sim",
            "--kv-bytes",
            "1048576",
        ])
        .expect("kv")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.kv_sim);
                assert_eq!(a.kv_bytes, Some(1_048_576));
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--kv-sim"]).expect("on") {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.kv_sim);
                assert_eq!(a.kv_bytes, None);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--decode-priority"]).unwrap_err();
        assert!(
            err.contains("--decode-priority requires --expert-sim"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--decode-priority=1"]).unwrap_err();
        assert!(
            err.contains("--decode-priority does not take a value"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--decode-priority"]).expect("pri") {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.decode_priority);
                assert!(a.gpu_cfg.stream_priority);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--cooperative"]).unwrap_err();
        assert!(err.contains("--cooperative requires --expert-sim"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--cooperative"]).expect("coop") {
            EngineCmd::Run(a) => assert!(a.gpu_cfg.cooperative),
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--pdl"]).unwrap_err();
        assert!(err.contains("--pdl requires --expert-sim"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--pdl"]).expect("pdl") {
            EngineCmd::Run(a) => assert!(a.gpu_cfg.pdl),
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--l2-persist"]).unwrap_err();
        assert!(err.contains("--l2-persist requires --expert-sim"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--l2-persist"]).expect("l2") {
            EngineCmd::Run(a) => assert!(a.gpu_cfg.l2_persist),
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--l2-reset"]).unwrap_err();
        assert!(err.contains("--l2-reset requires --expert-sim"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--l2-reset"]).expect("l2-reset") {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.l2_reset);
                assert!(a.gpu_cfg.l2_persist);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--l2-reset=1"]).unwrap_err();
        assert!(err.contains("--l2-reset does not take a value"), "{err}");
        let err = parse_engine_args(["m.gguf", "--l2-fetch", "32"]).unwrap_err();
        assert!(err.contains("--l2-fetch requires --expert-sim"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--l2-fetch", "0"]).unwrap_err();
        assert!(err.contains("l2-fetch must be 32, 64, or 128"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--l2-fetch", "96"]).unwrap_err();
        assert!(err.contains("l2-fetch must be 32, 64, or 128"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--l2-fetch", "32"]).expect("l2-fetch") {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.l2_fetch, 32);
                assert!(a.gpu_cfg.l2_persist);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--l2-fetch", "32", "--pdl"])
            .expect("l2-fetch+pdl")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.l2_fetch, 32);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--l2-fetch",
            "32",
            "--cooperative",
        ])
        .expect("l2-fetch+coop")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.l2_fetch, 32);
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--l2-ratio", "500"]).unwrap_err();
        assert!(err.contains("--l2-ratio requires --expert-sim"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--l2-ratio", "0"]).unwrap_err();
        assert!(err.contains("l2-ratio must be 1..=1000"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--l2-ratio", "1001"]).unwrap_err();
        assert!(err.contains("l2-ratio must be 1..=1000"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--l2-ratio", "500"]).expect("l2-ratio")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.l2_ratio, 500);
                assert!(a.gpu_cfg.l2_persist);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--l2-ratio", "500", "--pdl"])
            .expect("l2-ratio+pdl")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.l2_ratio, 500);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--l2-ratio",
            "500",
            "--cooperative",
        ])
        .expect("l2-ratio+coop")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.l2_ratio, 500);
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--l2-streaming"]).unwrap_err();
        assert!(
            err.contains("--l2-streaming requires --expert-sim"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--l2-streaming"]).unwrap_err();
        assert!(err.contains("--l2-streaming needs --l2-persist"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--l2-streaming=1"]).unwrap_err();
        assert!(
            err.contains("--l2-streaming does not take a value"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--l2-persist", "--l2-streaming"])
            .expect("l2-streaming")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.l2_persist);
                assert!(a.gpu_cfg.l2_streaming);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--l2-persist",
            "--l2-streaming",
            "--pdl",
        ])
        .expect("l2-streaming+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.l2_streaming);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--l2-ratio",
            "500",
            "--l2-streaming",
        ])
        .expect("l2-streaming+ratio")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.l2_ratio, 500);
                assert!(a.gpu_cfg.l2_streaming);
                assert!(a.gpu_cfg.l2_persist);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--cluster", "2"]).unwrap_err();
        assert!(err.contains("--cluster requires --expert-sim"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--cluster", "0"]).unwrap_err();
        assert!(err.contains("cluster must be > 0"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--cluster", "2"]).expect("cluster") {
            EngineCmd::Run(a) => assert_eq!(a.gpu_cfg.cluster, 2),
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--preferred-cluster", "4"]).unwrap_err();
        assert!(
            err.contains("--preferred-cluster requires --expert-sim"),
            "{err}"
        );
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--preferred-cluster", "0"]).unwrap_err();
        assert!(err.contains("preferred-cluster must be > 0"), "{err}");
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--preferred-cluster", "4"]).unwrap_err();
        assert!(err.contains("--preferred-cluster needs --cluster"), "{err}");
        let err = parse_engine_args([
            "m.gguf",
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
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--cluster",
            "2",
            "--preferred-cluster",
            "4",
        ])
        .expect("preferred")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.cluster, 2);
                assert_eq!(a.gpu_cfg.preferred_cluster, 4);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--cluster",
            "2",
            "--preferred-cluster",
            "4",
            "--pdl",
        ])
        .expect("preferred+pdl")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.preferred_cluster, 4);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--cluster-spread"]).unwrap_err();
        assert!(
            err.contains("--cluster-spread requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--cluster-spread"]).expect("spread") {
            EngineCmd::Run(a) => assert!(a.gpu_cfg.cluster_spread),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--cluster",
            "2",
            "--cluster-spread",
            "--pdl",
        ])
        .expect("spread+pdl")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.cluster, 2);
                assert!(a.gpu_cfg.cluster_spread);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--cluster-spread",
            "--cooperative",
        ])
        .expect("spread+coop")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.cluster_spread);
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--event-blocking-sync"]).unwrap_err();
        assert!(
            err.contains("--event-blocking-sync requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--event-blocking-sync"])
            .expect("event-blocking-sync")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.event_blocking_sync);
                assert!(a.gpu_cfg.timing_events);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--event-blocking-sync=1"]).unwrap_err();
        assert!(
            err.contains("--event-blocking-sync does not take a value"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--event-blocking-sync", "--pdl"])
            .expect("event-blocking-sync+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.event_blocking_sync);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--event-blocking-sync",
            "--cooperative",
        ])
        .expect("event-blocking-sync+coop")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.event_blocking_sync);
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--func-cluster-spread"]).unwrap_err();
        assert!(
            err.contains("--func-cluster-spread requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--func-cluster-spread"])
            .expect("func-cluster-spread")
        {
            EngineCmd::Run(a) => assert!(a.gpu_cfg.func_cluster_spread),
            EngineCmd::Help => panic!("expected Run"),
        }
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--func-cluster-spread=1"]).unwrap_err();
        assert!(
            err.contains("--func-cluster-spread does not take a value"),
            "{err}"
        );
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--cluster",
            "2",
            "--func-cluster-spread",
            "--pdl",
        ])
        .expect("func-cluster-spread+pdl")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.cluster, 2);
                assert!(a.gpu_cfg.func_cluster_spread);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--func-cluster-spread",
            "--cooperative",
        ])
        .expect("func-cluster-spread+coop")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.func_cluster_spread);
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--cluster-load-balance"]).unwrap_err();
        assert!(
            err.contains("--cluster-load-balance requires --expert-sim"),
            "{err}"
        );
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--cluster-load-balance"]).unwrap_err();
        assert!(
            err.contains("--cluster-load-balance needs --func-cluster-spread"),
            "{err}"
        );
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--func-cluster-spread",
            "--cluster-load-balance",
        ])
        .expect("cluster-load-balance")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.func_cluster_spread);
                assert!(a.gpu_cfg.cluster_load_balance);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--func-cluster-spread",
            "--cluster-load-balance",
            "--pdl",
        ])
        .expect("cluster-load-balance+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.cluster_load_balance);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--func-cluster-spread",
            "--cluster-load-balance",
            "--cooperative",
        ])
        .expect("cluster-load-balance+coop")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.cluster_load_balance);
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--func-cluster-spread",
            "--cluster-spread",
            "--cluster-load-balance",
        ])
        .unwrap_err();
        assert!(
            err.contains("choose one of --cluster-load-balance, --cluster-spread"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--cluster-must-set"]).unwrap_err();
        assert!(
            err.contains("--cluster-must-set requires --expert-sim"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--cluster-must-set"]).unwrap_err();
        assert!(err.contains("--cluster-must-set needs --cluster"), "{err}");
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--cluster",
            "2",
            "--cluster-must-set",
        ])
        .expect("cluster-must-set")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.cluster, 2);
                assert!(a.gpu_cfg.cluster_must_set);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--cluster-must-set=1"]).unwrap_err();
        assert!(
            err.contains("--cluster-must-set does not take a value"),
            "{err}"
        );
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--cluster",
            "2",
            "--cluster-must-set",
            "--pdl",
        ])
        .expect("cluster-must-set+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.cluster_must_set);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--cluster",
            "2",
            "--cluster-must-set",
            "--cooperative",
        ])
        .expect("cluster-must-set+coop")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.cluster_must_set);
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--required-cluster", "2"]).unwrap_err();
        assert!(
            err.contains("--required-cluster requires --expert-sim"),
            "{err}"
        );
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--required-cluster", "0"]).unwrap_err();
        assert!(err.contains("required-cluster must be > 0"), "{err}");
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--required-cluster", "2"]).unwrap_err();
        assert!(err.contains("--required-cluster needs --cluster"), "{err}");
        let err = parse_engine_args([
            "m.gguf",
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
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--cluster",
            "2",
            "--required-cluster",
            "2",
        ])
        .expect("required-cluster")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.cluster, 2);
                assert_eq!(a.gpu_cfg.required_cluster, 2);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--cluster",
            "2",
            "--required-cluster",
            "2",
            "--pdl",
        ])
        .expect("required-cluster+pdl")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.required_cluster, 2);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--cluster",
            "2",
            "--required-cluster",
            "2",
            "--cooperative",
        ])
        .expect("required-cluster+coop")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.required_cluster, 2);
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--cluster",
            "2",
            "--cluster-must-set",
            "--required-cluster",
            "2",
        ])
        .expect("required-cluster+must-set")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.cluster_must_set);
                assert_eq!(a.gpu_cfg.required_cluster, 2);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--max-shared"]).unwrap_err();
        assert!(err.contains("--max-shared requires --expert-sim"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--max-shared"]).expect("max-shared") {
            EngineCmd::Run(a) => assert!(a.gpu_cfg.max_shared),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--max-shared", "--pdl"])
            .expect("max-shared+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.max_shared);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--func-max-shared"]).unwrap_err();
        assert!(
            err.contains("--func-max-shared requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--func-max-shared"])
            .expect("func-max-shared")
        {
            EngineCmd::Run(a) => assert!(a.gpu_cfg.func_max_shared),
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--func-max-shared=1"]).unwrap_err();
        assert!(
            err.contains("--func-max-shared does not take a value"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--func-max-shared", "--pdl"])
            .expect("func-max-shared+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.func_max_shared);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--max-l1"]).unwrap_err();
        assert!(err.contains("--max-l1 requires --expert-sim"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--max-l1"]).unwrap_err();
        assert!(err.contains("--max-l1 needs --func-max-shared"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--func-max-shared", "--max-l1"])
            .expect("max-l1")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.func_max_shared);
                assert!(a.gpu_cfg.max_l1);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--func-max-shared",
            "--max-l1",
            "--pdl",
        ])
        .expect("max-l1+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.max_l1);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--func-max-shared",
            "--max-l1",
            "--cooperative",
        ])
        .expect("max-l1+coop")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.max_l1);
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--func-max-shared",
            "--max-shared",
            "--max-l1",
        ])
        .unwrap_err();
        assert!(
            err.contains("choose one of --max-l1, --max-shared"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--non-portable-cluster"]).unwrap_err();
        assert!(
            err.contains("--non-portable-cluster requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--non-portable-cluster"])
            .expect("non-portable")
        {
            EngineCmd::Run(a) => assert!(a.gpu_cfg.non_portable_cluster),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--cluster",
            "2",
            "--non-portable-cluster",
            "--pdl",
        ])
        .expect("non-portable+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.non_portable_cluster);
                assert_eq!(a.gpu_cfg.cluster, 2);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--sync-policy", "blocking"]).unwrap_err();
        assert!(err.contains("--sync-policy requires --expert-sim"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--sync-policy", "blocking"])
            .expect("sync-policy")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.sync_policy, SynchronizationPolicy::BlockingSync)
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--sync-policy", "spin", "--pdl"])
            .expect("sync-policy+pdl")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.sync_policy, SynchronizationPolicy::Spin);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--sync-policy", "bogus"]).unwrap_err();
        assert!(err.contains("unknown sync-policy"), "{err}");
        let err = parse_engine_args(["m.gguf", "--device-sync-policy", "blocking"]).unwrap_err();
        assert!(
            err.contains("--device-sync-policy requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--device-sync-policy", "blocking"])
            .expect("device-sync-policy")
        {
            EngineCmd::Run(a) => assert_eq!(
                a.gpu_cfg.device_sync_policy,
                SynchronizationPolicy::BlockingSync
            ),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--device-sync-policy",
            "spin",
            "--pdl",
        ])
        .expect("device-sync-policy+pdl")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.device_sync_policy, SynchronizationPolicy::Spin);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--device-sync-policy",
            "blocking",
            "--cooperative",
        ])
        .expect("device-sync-policy+coop")
        {
            EngineCmd::Run(a) => {
                assert_eq!(
                    a.gpu_cfg.device_sync_policy,
                    SynchronizationPolicy::BlockingSync
                );
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--device-sync-policy", "bogus"])
            .unwrap_err();
        assert!(err.contains("unknown device-sync-policy"), "{err}");
        let err = parse_engine_args(["m.gguf", "--mem-sync-domain", "remote"]).unwrap_err();
        assert!(
            err.contains("--mem-sync-domain requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--mem-sync-domain", "remote"])
            .expect("mem-sync-domain")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.mem_sync_domain, MemSyncDomain::Remote);
                assert!(a.gpu_cfg.decode_priority);
                assert!(a.gpu_cfg.stream_priority);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--mem-sync-domain", "default"])
            .expect("mem-sync-default")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.mem_sync_domain, MemSyncDomain::Default);
                assert!(!a.gpu_cfg.decode_priority);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--mem-sync-domain", "bogus"])
            .unwrap_err();
        assert!(err.contains("unknown mem-sync-domain"), "{err}");
        let err = parse_engine_args(["m.gguf", "--mem-sync-map", "collapse"]).unwrap_err();
        assert!(
            err.contains("--mem-sync-map requires --expert-sim"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--mem-sync-map", "collapse"])
            .unwrap_err();
        assert!(
            err.contains("--mem-sync-map collapse needs --mem-sync-domain remote"),
            "{err}"
        );
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--mem-sync-domain",
            "remote",
            "--mem-sync-map",
            "collapse",
        ])
        .expect("mem-sync-map")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.mem_sync_collapse);
                assert_eq!(a.gpu_cfg.mem_sync_domain, MemSyncDomain::Remote);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--mem-sync-map", "identity"])
            .expect("mem-sync-map identity")
        {
            EngineCmd::Run(a) => assert!(!a.gpu_cfg.mem_sync_collapse),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--mem-sync-domain",
            "remote",
            "--mem-sync-map",
            "collapse",
            "--pdl",
        ])
        .expect("mem-sync-map+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.mem_sync_collapse);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--mem-sync-domain",
            "remote",
            "--mem-sync-map",
            "collapse",
            "--cooperative",
        ])
        .expect("mem-sync-map+coop")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.mem_sync_collapse);
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--mem-sync-map", "bogus"]).unwrap_err();
        assert!(err.contains("unknown mem-sync-map"), "{err}");
        let err = parse_engine_args(["m.gguf", "--mem-sync-launch"]).unwrap_err();
        assert!(
            err.contains("--mem-sync-launch requires --expert-sim"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--mem-sync-launch"]).unwrap_err();
        assert!(
            err.contains("--mem-sync-launch needs --mem-sync-domain remote"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--mem-sync-launch=1"]).unwrap_err();
        assert!(
            err.contains("--mem-sync-launch does not take a value"),
            "{err}"
        );
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--mem-sync-domain",
            "remote",
            "--mem-sync-launch",
        ])
        .expect("mem-sync-launch")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.mem_sync_launch);
                assert_eq!(a.gpu_cfg.mem_sync_domain, MemSyncDomain::Remote);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--mem-sync-domain",
            "remote",
            "--mem-sync-launch",
            "--pdl",
        ])
        .expect("mem-sync-launch+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.mem_sync_launch);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--mem-sync-domain",
            "remote",
            "--mem-sync-launch",
            "--cooperative",
        ])
        .expect("mem-sync-launch+coop")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.mem_sync_launch);
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--mem-sync-launch-map"]).unwrap_err();
        assert!(
            err.contains("--mem-sync-launch-map requires --expert-sim"),
            "{err}"
        );
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--mem-sync-launch-map"]).unwrap_err();
        assert!(
            err.contains("--mem-sync-launch-map needs --mem-sync-domain remote"),
            "{err}"
        );
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--mem-sync-launch-map=1"]).unwrap_err();
        assert!(
            err.contains("--mem-sync-launch-map does not take a value"),
            "{err}"
        );
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--mem-sync-domain",
            "remote",
            "--mem-sync-launch-map",
        ])
        .expect("mem-sync-launch-map")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.mem_sync_launch_map);
                assert_eq!(a.gpu_cfg.mem_sync_domain, MemSyncDomain::Remote);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--mem-sync-domain",
            "remote",
            "--mem-sync-launch-map",
            "--pdl",
        ])
        .expect("mem-sync-launch-map+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.mem_sync_launch_map);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--mem-sync-domain",
            "remote",
            "--mem-sync-launch-map",
            "--cooperative",
        ])
        .expect("mem-sync-launch-map+coop")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.mem_sync_launch_map);
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--shared-mem", "eight"]).unwrap_err();
        assert!(err.contains("--shared-mem requires --expert-sim"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--shared-mem", "eight"])
            .expect("shared-mem")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.shared_mem, SharedMemoryMode::EightByte)
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--shared-mem", "four", "--pdl"])
            .expect("shared-mem+pdl")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.shared_mem, SharedMemoryMode::FourByte);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--shared-mem", "bogus"]).unwrap_err();
        assert!(err.contains("unknown shared-mem"), "{err}");
        let err = parse_engine_args(["m.gguf", "--func-shared-mem", "eight"]).unwrap_err();
        assert!(
            err.contains("--func-shared-mem requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--func-shared-mem", "eight"])
            .expect("func-shared-mem")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.func_shared_mem, SharedMemoryMode::EightByte)
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--func-shared-mem",
            "four",
            "--pdl",
        ])
        .expect("func-shared-mem+pdl")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.func_shared_mem, SharedMemoryMode::FourByte);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--func-shared-mem", "bogus"])
            .unwrap_err();
        assert!(err.contains("unknown func-shared-mem"), "{err}");
        let err = parse_engine_args(["m.gguf", "--device-shared-mem", "eight"]).unwrap_err();
        assert!(
            err.contains("--device-shared-mem requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--device-shared-mem", "eight"])
            .expect("device-shared-mem")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.device_shared_mem, SharedMemoryMode::EightByte)
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--device-shared-mem",
            "four",
            "--pdl",
        ])
        .expect("device-shared-mem+pdl")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.device_shared_mem, SharedMemoryMode::FourByte);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--device-shared-mem", "bogus"])
            .unwrap_err();
        assert!(err.contains("unknown device-shared-mem"), "{err}");
        let err = parse_engine_args(["m.gguf", "--portable-cluster", "non-portable"]).unwrap_err();
        assert!(
            err.contains("--portable-cluster requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--portable-cluster",
            "non-portable",
        ])
        .expect("portable-cluster")
        {
            EngineCmd::Run(a) => {
                assert_eq!(
                    a.gpu_cfg.portable_cluster,
                    PortableClusterMode::AllowNonPortable
                )
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--portable-cluster",
            "portable",
            "--pdl",
        ])
        .expect("portable-cluster+pdl")
        {
            EngineCmd::Run(a) => {
                assert_eq!(
                    a.gpu_cfg.portable_cluster,
                    PortableClusterMode::RequirePortable
                );
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--portable-cluster", "bogus"])
            .unwrap_err();
        assert!(err.contains("unknown portable-cluster"), "{err}");
        let err = parse_engine_args(["m.gguf", "--optin-shared"]).unwrap_err();
        assert!(
            err.contains("--optin-shared requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--optin-shared"]).expect("optin-shared")
        {
            EngineCmd::Run(a) => assert!(a.gpu_cfg.optin_shared),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--optin-shared", "--pdl"])
            .expect("optin-shared+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.optin_shared);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--dynamic-shared", "65536"]).unwrap_err();
        assert!(
            err.contains("--dynamic-shared requires --expert-sim"),
            "{err}"
        );
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--dynamic-shared", "0"]).unwrap_err();
        assert!(err.contains("dynamic-shared must be > 0"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--dynamic-shared", "65536"])
            .expect("dynamic-shared")
        {
            EngineCmd::Run(a) => assert_eq!(a.gpu_cfg.dynamic_shared, 65_536),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--dynamic-shared",
            "65536",
            "--pdl",
        ])
        .expect("dynamic-shared+pdl")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.dynamic_shared, 65_536);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--portable-shared", "non-portable"]).unwrap_err();
        assert!(
            err.contains("--portable-shared requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--portable-shared",
            "non-portable",
        ])
        .expect("portable-shared")
        {
            EngineCmd::Run(a) => {
                assert_eq!(
                    a.gpu_cfg.portable_shared,
                    PortableSharedMode::AllowNonPortable
                )
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--portable-shared",
            "portable",
            "--pdl",
        ])
        .expect("portable-shared+pdl")
        {
            EngineCmd::Run(a) => {
                assert_eq!(
                    a.gpu_cfg.portable_shared,
                    PortableSharedMode::RequirePortable
                );
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--portable-shared", "bogus"])
            .unwrap_err();
        assert!(err.contains("unknown portable-shared"), "{err}");
        let err = parse_engine_args(["m.gguf", "--nvlink-util"]).unwrap_err();
        assert!(err.contains("--nvlink-util requires --expert-sim"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--nvlink-util"]).expect("nvlink-util") {
            EngineCmd::Run(a) => assert!(a.gpu_cfg.nvlink_util_centric),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--nvlink-util", "--pdl"])
            .expect("nvlink-util+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.nvlink_util_centric);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--nvlink-util=1"]).unwrap_err();
        assert!(err.contains("--nvlink-util does not take a value"), "{err}");
        let err = parse_engine_args(["m.gguf", "--device-launch"]).unwrap_err();
        assert!(
            err.contains("--device-launch requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--device-launch"])
            .expect("device-launch")
        {
            EngineCmd::Run(a) => assert!(a.gpu_cfg.device_launch),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--device-updatable", "--pdl"])
            .expect("device-updatable+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.device_updatable);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--device-launch=1"]).unwrap_err();
        assert!(
            err.contains("--device-launch does not take a value"),
            "{err}"
        );
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--device-launch",
            "--graph-update",
        ])
        .unwrap_err();
        assert!(
            err.contains("choose one of --graph-update, --device-launch"),
            "{err}"
        );
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-enable",
            "--device-launch",
        ])
        .unwrap_err();
        assert!(err.contains("graph-enable cannot device-launch"), "{err}");
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--graph-build",
            "--graph-if",
            "--device-launch",
        ])
        .unwrap_err();
        assert!(err.contains("graph-if cannot device-launch"), "{err}");
        let err = parse_engine_args(["m.gguf", "--launch-completion"]).unwrap_err();
        assert!(
            err.contains("--launch-completion requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--launch-completion"])
            .expect("launch-completion")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.launch_completion);
                assert!(!a.gpu_cfg.decode_priority);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--launch-completion=1"]).unwrap_err();
        assert!(
            err.contains("--launch-completion does not take a value"),
            "{err}"
        );
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--launch-completion",
            "--device-launch",
        ])
        .unwrap_err();
        assert!(
            err.contains("launch-completion cannot device-launch"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--programmatic-event"]).unwrap_err();
        assert!(
            err.contains("--programmatic-event requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--programmatic-event"])
            .expect("programmatic-event")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.programmatic_event);
                assert!(!a.gpu_cfg.decode_priority);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--programmatic-event=1"]).unwrap_err();
        assert!(
            err.contains("--programmatic-event does not take a value"),
            "{err}"
        );
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--programmatic-event",
            "--device-launch",
        ])
        .unwrap_err();
        assert!(
            err.contains("programmatic-event cannot device-launch"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--stream-attach"]).unwrap_err();
        assert!(
            err.contains("--stream-attach requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--stream-attach"])
            .expect("stream-attach")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.stream_attach);
                assert_eq!(a.fill, GpuFill::Managed);
                assert!(!a.gpu_cfg.decode_priority);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--stream-attach=1"]).unwrap_err();
        assert!(
            err.contains("--stream-attach does not take a value"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--stream-attach", "--seq-streams"])
            .unwrap_err();
        assert!(err.contains("stream-attach cannot seq-streams"), "{err}");
        let err = parse_engine_args(["m.gguf", "--managed-host"]).unwrap_err();
        assert!(
            err.contains("--managed-host requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--managed-host"]).expect("managed-host")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.managed_host);
                assert_eq!(a.fill, GpuFill::Managed);
                assert!(!a.gpu_cfg.stream_attach);
                assert!(!a.gpu_cfg.decode_priority);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--managed-host=1"]).unwrap_err();
        assert!(
            err.contains("--managed-host does not take a value"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--prefetch-host"]).unwrap_err();
        assert!(
            err.contains("--prefetch-host requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--prefetch-host"])
            .expect("prefetch-host")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.prefetch_host);
                assert_eq!(a.fill, GpuFill::Managed);
            }
            EngineCmd::Help => panic!("help"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--prefetch-host=1"]).unwrap_err();
        assert!(
            err.contains("--prefetch-host does not take a value"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--d2h-evict"]).unwrap_err();
        assert!(err.contains("--d2h-evict requires --expert-sim"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--d2h-evict"]).expect("d2h-evict") {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.d2h_evict);
                assert_eq!(a.fill, GpuFill::Pinned);
            }
            EngineCmd::Help => panic!("help"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--d2h-evict", "--pdl"])
            .expect("d2h+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.d2h_evict);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("help"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--d2h-evict=1"]).unwrap_err();
        assert!(err.contains("--d2h-evict does not take a value"), "{err}");
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--d2h-evict", "--managed"]).unwrap_err();
        assert!(err.contains("d2h-evict needs pinned/vmm"), "{err}");
        let err = parse_engine_args(["m.gguf", "--d2h-pageable"]).unwrap_err();
        assert!(
            err.contains("--d2h-pageable requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--d2h-pageable"]).expect("d2h-pageable")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.d2h_pageable);
                assert!(a.gpu_cfg.pageable);
                assert_eq!(a.fill, GpuFill::Pinned);
            }
            EngineCmd::Help => panic!("help"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--d2h-pageable", "--pdl"])
            .expect("d2h-pageable+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.d2h_pageable);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("help"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--d2h-pageable=1"]).unwrap_err();
        assert!(
            err.contains("--d2h-pageable does not take a value"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--d2h-pageable", "--managed"])
            .unwrap_err();
        assert!(err.contains("d2h-pageable needs pageable"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--d2h-pageable", "--d2h-evict"])
            .unwrap_err();
        assert!(
            err.contains("choose one of --d2h-pageable, --d2h-evict"),
            "{err}"
        );
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--d2h-pageable",
            "--host-register",
        ])
        .unwrap_err();
        assert!(
            err.contains("--d2h-pageable cannot --host-register"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--no-read-mostly"]).unwrap_err();
        assert!(
            err.contains("--no-read-mostly requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--no-read-mostly"])
            .expect("no-read-mostly")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.no_read_mostly);
                assert_eq!(a.fill, GpuFill::Managed);
            }
            EngineCmd::Help => panic!("help"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--no-read-mostly=1"]).unwrap_err();
        assert!(
            err.contains("--no-read-mostly does not take a value"),
            "{err}"
        );
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--no-read-mostly", "--vmm"]).unwrap_err();
        assert!(err.contains("choose one of mapped, managed, vmm"), "{err}");
        let err = parse_engine_args(["m.gguf", "--no-preferred"]).unwrap_err();
        assert!(
            err.contains("--no-preferred requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--no-preferred"]).expect("no-preferred")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.no_preferred);
                assert_eq!(a.fill, GpuFill::Managed);
            }
            EngineCmd::Help => panic!("help"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--no-preferred=1"]).unwrap_err();
        assert!(
            err.contains("--no-preferred does not take a value"),
            "{err}"
        );
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--no-preferred", "--vmm"]).unwrap_err();
        assert!(err.contains("choose one of mapped, managed, vmm"), "{err}");
        let err = parse_engine_args(["m.gguf", "--no-mem-prefetch"]).unwrap_err();
        assert!(
            err.contains("--no-mem-prefetch requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--no-mem-prefetch"])
            .expect("no-mem-prefetch")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.no_mem_prefetch);
                assert_eq!(a.fill, GpuFill::Managed);
            }
            EngineCmd::Help => panic!("help"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--no-mem-prefetch=1"]).unwrap_err();
        assert!(
            err.contains("--no-mem-prefetch does not take a value"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--no-mem-prefetch", "--vmm"])
            .unwrap_err();
        assert!(err.contains("choose one of mapped, managed, vmm"), "{err}");
        let err = parse_engine_args(["m.gguf", "--host-register-mapped"]).unwrap_err();
        assert!(
            err.contains("--host-register-mapped requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--host-register-mapped"])
            .expect("host-register-mapped")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.host_register_mapped);
                assert_eq!(a.fill, GpuFill::Mapped);
                assert!(!a.gpu_cfg.host_register);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--host-register-mapped=1"]).unwrap_err();
        assert!(
            err.contains("--host-register-mapped does not take a value"),
            "{err}"
        );
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--host-register",
            "--host-register-mapped",
        ])
        .unwrap_err();
        assert!(
            err.contains("choose one of --host-register, --host-register-mapped"),
            "{err}"
        );
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--memcpy-batch",
            "--host-register-mapped",
        ])
        .unwrap_err();
        assert!(
            err.contains("memcpy-batch needs async pinned/vmm H2D"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--sync-memops"]).unwrap_err();
        assert!(err.contains("--sync-memops requires --expert-sim"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--sync-memops"]).expect("sync-memops") {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.sync_memops);
                assert_eq!(a.fill, GpuFill::Pinned);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--sync-memops=1"]).unwrap_err();
        assert!(err.contains("--sync-memops does not take a value"), "{err}");
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--sync-memops", "--mapped"]).unwrap_err();
        assert!(err.contains("sync-memops needs device memcpy"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--sync-memops", "--memcpy-batch"])
            .unwrap_err();
        assert!(
            err.contains("memcpy-batch needs async pinned/vmm H2D"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--device-sync-memops"]).unwrap_err();
        assert!(
            err.contains("--device-sync-memops requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--device-sync-memops"])
            .expect("device-sync-memops")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.device_sync_memops);
                assert!(!a.gpu_cfg.sync_memops);
                assert_eq!(a.fill, GpuFill::Pinned);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--device-sync-memops=1"]).unwrap_err();
        assert!(
            err.contains("--device-sync-memops does not take a value"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--device-sync-memops", "--mapped"])
            .unwrap_err();
        assert!(
            err.contains("device-sync-memops needs device memcpy"),
            "{err}"
        );
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--device-sync-memops",
            "--memcpy-batch",
        ])
        .unwrap_err();
        assert!(
            err.contains("memcpy-batch needs async pinned/vmm H2D"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--wait-value"]).unwrap_err();
        assert!(err.contains("--wait-value requires --expert-sim"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--wait-value"]).expect("wait-value") {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.wait_value);
                assert!(!a.gpu_cfg.decode_priority);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--wait-value=1"]).unwrap_err();
        assert!(err.contains("--wait-value does not take a value"), "{err}");
        let err = parse_engine_args(["m.gguf", "--mempool-trim"]).unwrap_err();
        assert!(
            err.contains("--mempool-trim requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--mempool-trim"]).expect("mempool-trim")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.mempool_trim);
                assert!(a.gpu_cfg.mempool);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--mempool-trim=1"]).unwrap_err();
        assert!(
            err.contains("--mempool-trim does not take a value"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--mempool-trim", "--sync-alloc"])
            .unwrap_err();
        assert!(err.contains("mempool-trim needs cudaMallocAsync"), "{err}");
        let err = parse_engine_args(["m.gguf", "--mempool-no-reuse"]).unwrap_err();
        assert!(
            err.contains("--mempool-no-reuse requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--mempool-no-reuse"])
            .expect("mempool-no-reuse")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.mempool_no_reuse);
                assert!(a.gpu_cfg.mempool);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--mempool-no-reuse=1"]).unwrap_err();
        assert!(
            err.contains("--mempool-no-reuse does not take a value"),
            "{err}"
        );
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--mempool-no-reuse",
            "--sync-alloc",
        ])
        .unwrap_err();
        assert!(
            err.contains("mempool-no-reuse needs cudaMallocAsync"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--mempool-max", "4096"]).unwrap_err();
        assert!(err.contains("--mempool-max requires --expert-sim"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--mempool-max", "0"]).unwrap_err();
        assert!(err.contains("mempool-max must be > 0"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--mempool-max", "4096"])
            .expect("mempool-max")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.mempool_max, 4096);
                assert!(a.gpu_cfg.mempool);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--mempool-max", "4096", "--pdl"])
            .expect("mempool-max+pdl")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.mempool_max, 4096);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--mempool-max",
            "4096",
            "--sync-alloc",
        ])
        .unwrap_err();
        assert!(err.contains("mempool-max needs cudaMallocAsync"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--wait-value", "--device-launch"])
            .expect("wait-value+device-launch")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.wait_value);
                assert!(a.gpu_cfg.device_launch);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--pdl", "--cooperative"]).unwrap_err();
        assert!(err.contains("choose one of --pdl, --cooperative"), "{err}");
        let err = parse_engine_args(["m.gguf", "--kernel-priority", "5"]).unwrap_err();
        assert!(
            err.contains("--kernel-priority requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--kernel-priority", "0"])
            .expect("kernel-priority 0")
        {
            EngineCmd::Run(a) => assert_eq!(a.gpu_cfg.kernel_priority, Some(0)),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--kernel-priority", "-5", "--pdl"])
            .expect("kernel-priority+pdl")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.kernel_priority, Some(-5));
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--kernel-priority", "bogus"])
            .unwrap_err();
        assert!(err.contains("invalid kernel-priority"), "{err}");
        let err = parse_engine_args(["m.gguf", "--multicast"]).unwrap_err();
        assert!(err.contains("--multicast requires --expert-sim"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--multicast"]).expect("mc") {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.multicast);
                assert!(a.fill == GpuFill::Vmm);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--shareable"]).unwrap_err();
        assert!(err.contains("--shareable requires --expert-sim"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--shareable"]).expect("sh") {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.shareable);
                assert!(a.gpu_cfg.mempool);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--shareable", "--sync-alloc"])
            .unwrap_err();
        assert!(err.contains("shareable needs cudaMallocAsync"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--memcpy-batch"]).expect("mb") {
            EngineCmd::Run(a) => assert!(a.gpu_cfg.memcpy_batch),
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--memcpy-batch", "--pageable"])
            .unwrap_err();
        assert!(
            err.contains("memcpy-batch needs async pinned/vmm H2D"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--memcpy-batch", "--managed"])
            .unwrap_err();
        assert!(
            err.contains("memcpy-batch needs async pinned/vmm H2D"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--memcpy-during"]).unwrap_err();
        assert!(
            err.contains("--memcpy-during requires --expert-sim"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--memcpy-during"]).unwrap_err();
        assert!(
            err.contains("--memcpy-during needs --memcpy-batch"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--memcpy-during=1"]).unwrap_err();
        assert!(
            err.contains("--memcpy-during does not take a value"),
            "{err}"
        );
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--memcpy-batch",
            "--memcpy-during",
        ])
        .expect("md")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.memcpy_batch);
                assert!(a.gpu_cfg.memcpy_during);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--memcpy-batch",
            "--memcpy-during",
            "--pdl",
        ])
        .expect("md+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.memcpy_during);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--memcpy-any"]).unwrap_err();
        assert!(err.contains("--memcpy-any requires --expert-sim"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--memcpy-any"]).unwrap_err();
        assert!(err.contains("--memcpy-any needs --memcpy-batch"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--memcpy-any=1"]).unwrap_err();
        assert!(err.contains("--memcpy-any does not take a value"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--memcpy-batch", "--memcpy-any"])
            .expect("ma")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.memcpy_batch);
                assert!(a.gpu_cfg.memcpy_any);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--memcpy-batch",
            "--memcpy-any",
            "--pdl",
        ])
        .expect("ma+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.memcpy_any);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args([
            "m.gguf",
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
        let err = parse_engine_args(["m.gguf", "--memcpy-attr"]).unwrap_err();
        assert!(err.contains("--memcpy-attr requires --expert-sim"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--memcpy-attr=1"]).unwrap_err();
        assert!(err.contains("--memcpy-attr does not take a value"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--memcpy-attr"]).expect("mattr") {
            EngineCmd::Run(a) => assert!(a.gpu_cfg.memcpy_attr),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--memcpy-attr", "--pdl"])
            .expect("ma+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.memcpy_attr);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--memcpy-attr", "--managed"])
            .unwrap_err();
        assert!(
            err.contains("memcpy-attr needs async pinned/vmm H2D"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--memset-fill"]).unwrap_err();
        assert!(err.contains("--memset-fill requires --expert-sim"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--memset-fill=1"]).unwrap_err();
        assert!(err.contains("--memset-fill does not take a value"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--memset-fill"]).expect("mf") {
            EngineCmd::Run(a) => assert!(a.gpu_cfg.memset_fill),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--memset-fill", "--pdl"])
            .expect("mf+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.memset_fill);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--memset-fill", "--cooperative"])
            .expect("mf+coop")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.memset_fill);
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--memset-fill", "--mapped"]).unwrap_err();
        assert!(err.contains("--memset-fill cannot --mapped"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--memset-fill", "--managed"])
            .unwrap_err();
        assert!(err.contains("--memset-fill cannot --managed"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--memset-fill", "--pageable"])
            .unwrap_err();
        assert!(err.contains("--memset-fill cannot --pageable"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--memset-fill", "--memcpy-batch"])
            .unwrap_err();
        assert!(err.contains("--memset-fill cannot --memcpy-batch"), "{err}");
        let err = parse_engine_args(["m.gguf", "--copy-host"]).unwrap_err();
        assert!(err.contains("--copy-host requires --expert-sim"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--copy-host=1"]).unwrap_err();
        assert!(err.contains("--copy-host does not take a value"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--copy-host"]).expect("ch") {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.copy_host);
                assert!(!a.gpu_cfg.host_func);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--copy-host", "--pdl"]).expect("ch+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.copy_host);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--expert-sim", "--copy-host", "--cooperative"])
            .expect("ch+coop")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.copy_host);
                assert!(a.gpu_cfg.cooperative);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--host-register"]).unwrap_err();
        assert!(
            err.contains("--host-register requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--host-register"]).expect("hr") {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.host_register);
                assert!(a.gpu_cfg.pageable);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--host-register=1"]).unwrap_err();
        assert!(
            err.contains("--host-register does not take a value"),
            "{err}"
        );
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--memcpy-batch",
            "--host-register",
        ])
        .unwrap_err();
        assert!(
            err.contains("memcpy-batch needs async pinned/vmm H2D"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--host-register", "--managed"])
            .unwrap_err();
        assert!(err.contains("host-register needs pinned/vmm H2D"), "{err}");
        let err = parse_engine_args(["m.gguf", "--host-unregister"]).unwrap_err();
        assert!(
            err.contains("--host-unregister requires --expert-sim"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--host-unregister"]).expect("hunreg") {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.host_unregister);
                assert!(a.gpu_cfg.host_register);
                assert!(a.gpu_cfg.pageable);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--host-unregister=1"]).unwrap_err();
        assert!(
            err.contains("--host-unregister does not take a value"),
            "{err}"
        );
        match parse_engine_args(["m.gguf", "--expert-sim", "--host-unregister", "--pdl"])
            .expect("hunreg+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.host_unregister);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--host-unregister", "--managed"])
            .unwrap_err();
        assert!(err.contains("host-register needs pinned/vmm H2D"), "{err}");
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--d2h-pageable",
            "--host-unregister",
        ])
        .unwrap_err();
        assert!(
            err.contains("--d2h-pageable cannot --host-register"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--ipc"]).unwrap_err();
        assert!(err.contains("--ipc requires --expert-sim"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--ipc"]).expect("ipc") {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.ipc);
                assert!(a.gpu_cfg.sync_alloc);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--ipc=1"]).unwrap_err();
        assert!(err.contains("--ipc does not take a value"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--ipc", "--pdl"]).expect("ipc+pdl") {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.ipc);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--ipc", "--managed"]).unwrap_err();
        assert!(err.contains("--ipc needs cudaMalloc"), "{err}");
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--ipc", "--shareable"]).unwrap_err();
        assert!(
            err.contains("choose one of --ipc, --shareable")
                || err.contains("shareable needs cudaMallocAsync"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--share-ptr"]).unwrap_err();
        assert!(err.contains("--share-ptr requires --expert-sim"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--share-ptr"]).expect("sptr") {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.share_ptr);
                assert!(a.gpu_cfg.shareable);
                assert!(a.gpu_cfg.mempool);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--share-ptr=1"]).unwrap_err();
        assert!(err.contains("--share-ptr does not take a value"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--share-ptr", "--pdl"])
            .expect("sptr+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.share_ptr);
                assert!(a.gpu_cfg.pdl);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--share-ptr", "--managed"]).unwrap_err();
        assert!(err.contains("shareable needs cudaMallocAsync"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--share-ptr", "--sync-alloc"])
            .unwrap_err();
        assert!(err.contains("shareable needs cudaMallocAsync"), "{err}");
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--share-ptr", "--ipc"]).unwrap_err();
        assert!(
            err.contains("choose one of --ipc, --shareable")
                || err.contains("shareable needs cudaMallocAsync"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--vmm-retain"]).unwrap_err();
        assert!(err.contains("--vmm-retain requires --expert-sim"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--vmm-retain"]).expect("retain") {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.vmm_retain);
                assert_eq!(a.fill, GpuFill::Vmm);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--vmm-retain=1"]).unwrap_err();
        assert!(err.contains("--vmm-retain does not take a value"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--vmm-retain", "--pdl"])
            .expect("retain+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.vmm_retain);
                assert!(a.gpu_cfg.pdl);
                assert_eq!(a.fill, GpuFill::Vmm);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--vmm-retain", "--mapped"]).unwrap_err();
        assert!(err.contains("choose one of mapped, managed, vmm"), "{err}");
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--vmm-retain", "--managed"]).unwrap_err();
        assert!(err.contains("choose one of mapped, managed, vmm"), "{err}");
        let err = parse_engine_args(["m.gguf", "--vmm-handle"]).unwrap_err();
        assert!(err.contains("--vmm-handle requires --expert-sim"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--vmm-handle"]).expect("handle") {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.vmm_handle);
                assert_eq!(a.fill, GpuFill::Vmm);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--vmm-handle=1"]).unwrap_err();
        assert!(err.contains("--vmm-handle does not take a value"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--vmm-handle", "--pdl"])
            .expect("handle+pdl")
        {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.vmm_handle);
                assert!(a.gpu_cfg.pdl);
                assert_eq!(a.fill, GpuFill::Vmm);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--vmm-handle", "--mapped"]).unwrap_err();
        assert!(err.contains("choose one of mapped, managed, vmm"), "{err}");
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--vmm-handle", "--managed"]).unwrap_err();
        assert!(err.contains("choose one of mapped, managed, vmm"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--vmm-handle", "--vmm-retain"])
            .unwrap_err();
        assert!(
            err.contains("choose one of --vmm-handle, --vmm-retain"),
            "{err}"
        );
        let err = parse_engine_args(["m.gguf", "--compute-slots", "2"]).unwrap_err();
        assert!(
            err.contains("--compute-slots requires --expert-sim"),
            "{err}"
        );
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--compute-slots", "0"]).unwrap_err();
        assert!(err.contains("compute-slots must be > 0"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--compute-slots", "2"]).expect("slots")
        {
            EngineCmd::Run(a) => assert_eq!(a.gpu_cfg.compute_slots, 2),
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--decode-sms", "250"]).unwrap_err();
        assert!(err.contains("--decode-sms requires --expert-sim"), "{err}");
        let err = parse_engine_args(["m.gguf", "--expert-sim", "--decode-sms", "0"]).unwrap_err();
        assert!(err.contains("decode-sms must be 1..=1000"), "{err}");
        let err =
            parse_engine_args(["m.gguf", "--expert-sim", "--decode-sms", "1001"]).unwrap_err();
        assert!(err.contains("decode-sms must be 1..=1000"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--decode-sms", "250"]).expect("sms") {
            EngineCmd::Run(a) => {
                assert_eq!(a.gpu_cfg.decode_sm_permille, 250);
                assert!(a.gpu_cfg.decode_priority);
                assert!(a.gpu_cfg.stream_priority);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--green-ctx"]).unwrap_err();
        assert!(err.contains("--green-ctx requires --expert-sim"), "{err}");
        match parse_engine_args(["m.gguf", "--expert-sim", "--green-ctx"]).expect("green") {
            EngineCmd::Run(a) => {
                assert!(a.gpu_cfg.green_ctx);
                assert!(a.gpu_cfg.decode_priority);
                assert!(a.gpu_cfg.stream_priority);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args([
            "m.gguf",
            "--expert-sim",
            "--decode-sms",
            "1000",
            "--green-ctx",
        ])
        .unwrap_err();
        assert!(
            err.contains("green-ctx decode-sms must leave leftover SMs"),
            "{err}"
        );
    }

    #[test]
    fn engine_prefetch_planner_flags() {
        match parse_engine_args(["m.gguf", "--prefetch", "none"]).expect("none") {
            EngineCmd::Run(a) => {
                assert_eq!(a.prefetch, Prefetch::None);
                assert!(!a.expert_sim);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args([
            "m.gguf",
            "--prefetch=copy-forward",
            "--plan-window",
            "8",
            "--plan-threshold=100",
        ])
        .expect("plan")
        {
            EngineCmd::Run(a) => {
                assert_eq!(a.prefetch, Prefetch::CopyForward);
                assert_eq!(a.plan_window, 8);
                assert_eq!(a.plan_threshold, 100);
            }
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--prefetch=markov"]).expect("mk") {
            EngineCmd::Run(a) => assert_eq!(a.prefetch, Prefetch::Markov),
            EngineCmd::Help => panic!("expected Run"),
        }
        match parse_engine_args(["m.gguf", "--prefetch=both"]).expect("both") {
            EngineCmd::Run(a) => assert_eq!(a.prefetch, Prefetch::Both),
            EngineCmd::Help => panic!("expected Run"),
        }
        let err = parse_engine_args(["m.gguf", "--prefetch", "nope"]).unwrap_err();
        assert!(err.contains("unknown prefetch"), "{err}");
        let err = parse_engine_args(["m.gguf", "--prefetch"]).unwrap_err();
        assert!(err.contains("missing --prefetch value"), "{err}");
    }

    #[test]
    fn engine_trace_out_writes_parseable_jsonl() {
        let pid = std::process::id();
        let dir = std::env::temp_dir();
        let gguf = dir.join(format!("llama-rust-eng-trace-{pid}.gguf"));
        let jsonl = dir.join(format!("llama-rust-eng-trace-{pid}.jsonl"));
        write_file(&gguf, &crate::decode::tiny_qwen3moe_gguf()).expect("gguf");
        let args = match parse_engine_args([
            gguf.to_str().expect("utf8 gguf"),
            "-p",
            "a",
            "-p",
            "b",
            "--kv-page",
            "2",
            "--trace-out",
            jsonl.to_str().expect("utf8 jsonl"),
        ])
        .expect("parse")
        {
            EngineCmd::Run(a) => a,
            EngineCmd::Help => panic!("expected Run"),
        };
        run_engine(&args).expect("run");
        let bytes = read_file(&jsonl).expect("read");
        let text = String::from_utf8(bytes).expect("utf8");
        let t = expertvm::Trace::parse(&text).expect("jsonl");
        assert!(
            !t.events.is_empty(),
            "engine --trace-out must emit MoE events"
        );
        assert!(t.events.iter().any(|e| e.sequence == 0));
        assert!(t.events.iter().any(|e| e.sequence == 1));
        let _rm_g = std::fs::remove_file(&gguf);
        let _rm_j = std::fs::remove_file(&jsonl);
    }

    #[test]
    fn engine_bench_report_names_policies() {
        let pid = std::process::id();
        let dir = std::env::temp_dir();
        let gguf = dir.join(format!("llama-rust-eng-bench-{pid}.gguf"));
        write_file(&gguf, &crate::decode::tiny_qwen3moe_gguf()).expect("gguf");
        let args = match parse_engine_args([
            gguf.to_str().expect("utf8 gguf"),
            "-p",
            "a",
            "-p",
            "b",
            "--kv-page",
            "2",
            "--bench",
            "--capacity",
            "2",
        ])
        .expect("parse")
        {
            EngineCmd::Run(a) => a,
            EngineCmd::Help => panic!("expected Run"),
        };
        let g = load_gguf_owned(read_file(&gguf).expect("bytes")).expect("owned");
        let tok = Tokenizer::from_gguf(&g).expect("tok");
        let model = Llama::from_gguf(g).expect("m");
        let cfg = engine_cfg(&tok, &args).expect("cfg");
        let mut eng = Engine::new(&model, cfg).expect("eng");
        attach_engine_store(&mut eng, &model, &args).expect("store");
        eng.enable_moe_trace();
        let mut handles = Vec::new();
        for ids in engine_prompts(&tok, &args).expect("prompts") {
            handles.push(eng.add(&ids, args.n_predict).expect("add"));
        }
        eng.run().expect("run");
        let traces = take_engine_traces(&mut eng, &handles);
        let text = engine_bench_report(&traces, &args).expect("bench");
        assert!(text.contains("# engine capacity=2"), "{text}");
        assert!(text.contains("lru"), "{text}");
        assert!(text.contains("oracle"), "{text}");
        assert!(!traces.events.is_empty(), "bench must record MoE events");
        run_engine(&args).expect("cli");
        let _rm = std::fs::remove_file(&gguf);
    }

    #[test]
    fn engine_sim_8gpu_tiny_bytes_runs() {
        let pid = std::process::id();
        let dir = std::env::temp_dir();
        let gguf = dir.join(format!("llama-rust-eng-8gpu-{pid}.gguf"));
        write_file(&gguf, &crate::decode::tiny_qwen3moe_gguf()).expect("gguf");
        let args = match parse_engine_args([
            gguf.to_str().expect("utf8 gguf"),
            "-p",
            "a",
            "-p",
            "b",
            "--kv-page",
            "2",
            "--expert-sim",
            "--expert-8gpu",
            "--expert-bytes",
            "64",
            "--expert-slots",
            "4",
        ])
        .expect("parse")
        {
            EngineCmd::Run(a) => a,
            EngineCmd::Help => panic!("expected Run"),
        };
        run_engine(&args).expect("run");
        let _rm = std::fs::remove_file(&gguf);
    }
}
