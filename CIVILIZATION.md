# Civilization plan

Subsequent agents execute **this file**, not the next CUDA `mem_*` identity
wrap in [PLAN.md](PLAN.md).

PLAN.md remains the historical research plan and CUDA catalog. Its thesis
is still right. Its default “next numbered item” became a treadmill of
identity wrappers around helpers that already existed. That treadmill does
not make open-weight models cheaper, more correct, or easier to run.

Work that does not move an item below toward its **Done when** gate is
misaligned, even if it compiles, even if it adds a CUDA name.

---

## Why this repo can matter

Open weights are a public good only if people can **run them**: correctly,
on hardware they can afford, with numbers they can reproduce.

Closed stacks (vLLM, TensorRT-LLM, SGLang) already serve tokens. The
unsettled public problem is the **memory side of sparse MoE**:

```
compute ≈ active parameters per token
memory  ≈ total stored parameters
```

A 320B-class sparse model that touches ~18B weights per token still needs
somewhere to put the other ~300B. That gap *is* energy, HBM, and $/token.
The wedge in this repo is not another HTTP server. It is **expert virtual
memory**: leases, residency, prefetch, replication, and a GPU-systems VM
so researchers without an 8×H100 can still fail illegal states and compare
policies.

`llama-rust` (`langtax/`) is the correctness laboratory: GGUF-native
decode, independent oracle, llama.cpp greedy identity on real checkpoints.
`expertvm` is the OSS primitive other engines should be able to consume
without adopting this server. `gpu-sim` is how fleets of agents work on
that primitive anyway. `infer-bench` is serving-shaped measurement over
the same traces — semantic illegal-state vs performance wall/HBM/bytes/
energy. Example hardware profiles still leave rent at `0` (no invented
`$ / M tokens`).

If serving here becomes faster *and* easier to use than the status quo, it
may grow into a high-performance engine. That is allowed. Cloning vLLM in
this tree is not the wedge.

---

## Inventory (do not rebuild)

| Piece | Job today | Civilization gap |
| --- | --- | --- |
| `llama-rust` decode | Mixed-dtype GGUF walk, oracle, llama.cpp sidecars | Real MoE checkpoints in CI are opt-in (`LLAMA_RUST_REAL_MODEL_DIR`); writer-tinies prove emission, not economics |
| ExpertStore seam | DirectStore identity, CachedStore, TieredStore, SimulatedGpuStore | Foreign engines cannot attach without taking this Engine |
| ExpertAccess traces | JSONL from decode / Engine | No published residency table on a real 80B+ MoE |
| `expertvm` CLI | analyze, place, replay, sim, schedule, bench, ep, remote, kv | Flag surface is large; researcher path is not one afternoon |
| `gpu-sim` | Exact CUDA-like mechanical invariants + calibrated `HardwareProfile` | Catalog of unused `mem_*` twins grew faster than decode/expertvm demand; occupancy SM counts still walled; `gpu-profile capture` refused |
| `infer-bench` | Dual semantic/performance scores on traces | Same tinies and synthetic workloads; no cited rent on example profiles (keep it that way unless sourced) |
| Engine | Continuous batching, paged KV, intern, pin_hot / place_hot | 80+ `--flag`s; DX is not easier than vLLM |

Standing gates that **stay green** while this plan runs:

- Independent oracle + existing decode tests
- llama.cpp greedy sidecars (`tests/reference/`); fail-loud when
  `LLAMA_RUST_REAL_MODEL_DIR` is set
- DirectStore bit-equal logits vs the GGUF blob path
- Dual score: no invented `$ / M tokens` on example profiles
- `gpu-profile capture` remains refused until item 6
- Occupancy SM counts / clock rates / warp size on `DeviceProperties`
  remain uninvented until a named policy needs them

---

## What not to do

Do not spend the next item on:

- another CUDA identity wrap (`Sim::mem_*` calling an existing helper)
  unless a **decode / expertvm / sim_replay path actually hits a missing
  mechanical invariant**
- an OpenAI-compatible HTTP veneer as the wedge (`serve` already exists;
  it is not the product)
- a 500k-line vLLM competitor, generic tensor/autograd library, tokenizer
  product, or Safetensors parser
- filling occupancy SM counts, clock rates, or warp size “because CUDA
  has the struct”
- inventing `rent_usd_micros_per_hour` on example profiles
- wrapping `array_create` as `mem_array_create`, wrapping occupancy SM
  resource APIs a second time, or reversing PLAN.md historical walls
- claiming writer-tiny traces answer “is residency predictable on 320B?”

PLAN.md “What not to build” still applies.

---

## Agent operating rules

1. Read this file and the **Done when** gate for the lowest unchecked
   numbered item. Execute that item. Do not redefine it into a CUDA wrap.
2. Land on `main` with frequent commits unless the session forbids it.
   Keep decode identity green. Update this file’s checkboxes and
   [STATUS.md](STATUS.md) when an item’s gate is met with evidence
   (command output, test names, measured tables — not intent).
3. Prefer tests + a working CLI over prose. If you change UI or serving
   behavior, verify the path a researcher would run.
4. If the kill-switch in item 3 fires, stop claiming MoE residency is
   predictable. Record the negative result. That is still civilization-
   useful (it saves the next lab a year).
5. After item 1, a stranger with Rust and this repo should reach a
   measured expertvm table without reading PLAN.md’s CUDA catalog.

---

## Execution items

Do these **in order**. An item is not started until the previous **Done
when** is evidenced in-tree.

### 1. [ ] Researcher one-afternoon path

**Why.** A repo that only agents can operate is not useful. Civilization
needs a documented loop a person can finish in one sitting.

**Do.**

- Add a short “First hour” section to the root README (and keep it true):
  write a tiny Qwen3MoE GGUF, `infer`, emit a trace, `expertvm analyze` +
  `replay` + `sim`, `infer-bench schedule` on that trace.
- Name **three** Engine invocations a researcher needs (blob identity,
  CachedStore, SimulatedGpuStore). Put the rest of `GpuStoreCfg` behind
  documented profiles or `--help` groups — do not delete flags, do not
  make them the front door.
- Add a crate example or `cargo test` that runs the trace → replay
  loop on an in-tree JSONL so the path cannot rot.

**Done when.** README “First hour” commands work from a clean clone
(no GPU). A test covers the loop. Engine `--help` leads with the three
invocations, not a wall of CUDA knobs.

**Do not.** OpenAI veneer. New CUDA twins. Rewriting PLAN.md history.

### 2. [ ] Standing correctness gate

**Why.** A subnormal f16 bug once hid behind 221 passing tests because
oracles shared production conversion. Sidecars exist so that cannot
recur.

**Do.**

- Keep `cargo test --release --lib` and `cargo clippy --all-targets
  --all-features -- -D warnings` as the default gate for every change.
- Document in README that `LLAMA_RUST_REAL_MODEL_DIR` must be **absolute**
  and contain both GGUFs named in `tests/reference/` (fail-loud if set
  but unopenable). Do not download Hugging Face weights in CI.
- Any ExpertStore or Engine change that touches logits must keep
  DirectStore identity tests.

**Done when.** README states the gate in one place. Real-model skip vs
fail-loud behavior is documented next to the commands. No change in this
plan lands with a red `--lib` on writer-tinies.

**Do not.** Skip oracle tests. Invent a second llama.cpp capture format.

### 3. [ ] Real-checkpoint MoE residency result

**Why.** Writer-tinies prove JSONL emission. They do **not** answer
whether prefetch beats LRU on a model people actually serve. PLAN.md
already named the kill-switch: if the best non-oracle policy ≈ random,
stop.

**Do.**

- Capture ExpertAccess JSONL from a real Qwen3MoE or Qwen2MoE GGUF
  (absolute `LLAMA_RUST_REAL_MODEL_DIR` or a documented local path). Do
  not commit the GGUF. Commit the trace **or** a script + hash that
  regenerates it.
- Run `expertvm analyze`, `place`, `replay` (LRU, LFU, Markov lookback-2,
  oracle), `sim` under `HardwareProfile::restrict_hbm`, and `ep`
  (static expert-parallel vs GPU0 LRU).
- Publish the table in `tests/traces/README.md` (or a sibling) with
  capacity, hit rates, bytes moved, `energy_uj` / ns per token from
  gpu-sim. No fictional hit rates.

**Done when.** In-tree table from a **real** checkpoint, plus the
commands that produced it. If best non-oracle ≈ random at the capacities
that matter, the table says so and item 10 records a negative result
instead of a paper claim.

**Do not.** Paste cycling.jsonl numbers as if they were 320B. Do not
invent prompt-class features the Markov table does not have.

### 4. [ ] `expertvm` as a foreign-engine primitive

**Why.** PLAN.md: nobody should have to adopt this inference server.
vLLM / SGLang / mistral.rs should be able to consume residency.

**Do.**

- Document `ExpertStore` / leases / `ExpertPhase` / prefetch as the
  stable surface in `expertvm/README.md` with a minimal attach example
  that does not go through `Engine`.
- Keep the trait object-safe enough (or provide a small C ABI later —
  not this item) that a second decoder could acquire/release experts.
- Trace schema (`ExpertAccess` JSONL) stays the interchange with
  infer-bench.

**Done when.** A reader can implement a dummy store against the trait
docs without reading `gpu-sim`. Decode still bit-matches DirectStore.

**Do not.** Merge expertvm into the Engine crate. Require HTTP.

### 5. [x] Freeze the CUDA catalog; add invariants only on demand

**Why.** gpu-sim’s value is **exact illegal states** and a discrete-event
clock, not a second copy of the CUDA driver API. PLAN 1000–1333 already
catalogued twins. Occupancy SM counts are still walled. That is correct
until a policy needs them.

**Do.**

- Treat [PLAN.md](PLAN.md) numbered CUDA wraps as **frozen default
  work**. Next gpu-sim change must cite a decode, expertvm, or sim_replay
  path that currently cannot fail a real illegal state.
- Keep occupancy SM counts, clock rates, and warp size uninvented.
- Keep Engine `--mem-*` flags uninvented for catalog twins.
- Dual score still has no `$ / M tokens` on example profiles.
- `gpu-profile capture` still refused (item 6).

**Done when.** STATUS or this file records the freeze. A gpu-sim PR/commit
message that cannot name the consumer path is out of scope.

Recorded 2026-09-03 in STATUS.md and PLAN.md item 1334. Standing keep-rules
above still apply; do not resume catalog twins.

**Do not.** Resume `mem_surf_object_create` as the next item “because it
is next in sim.rs.”

### 6. [ ] Calibrated profiles when silicon exists

**Why.** Timing must come from a `HardwareProfile`, not folklore in
policy code. Capture is refused today because a fake capture would be
worse than example H100 numbers.

**Do.**

- Implement `gpu-profile capture` **only** against a real device, writing
  kernel/copy/sync curves the VM already consumes.
- Until a machine is available, leave capture refused. Do not synthesize
  a capture file from the example profile and call it measured.
- When capture exists, A/B `expertvm sim` on the item 3 trace: example
  H100 vs captured H100. Record deltas.

**Done when.** Either (a) capture still refused and this item stays open
with “no silicon this run”, or (b) a captured profile is in-tree or
documented, with the command that produced it, and example profiles still
have rent `0`.

**Do not.** Invent occupancy SM counts as a substitute for capture.

### 7. [ ] Serving that earns its keep

**Why.** Objective allows a vLLM-class engine **if it is faster and
easier**. PLAN.md still forbids an HTTP veneer as the wedge.

**Do.**

- Measure Engine continuous batching + paged KV + intern + `--decode-first`
  vs naive sequential decode on a shared-prefix workload, using gpu-sim
  TTFT / ITL / `energy_uj` (SimulatedGpuStore). Publish the comparison.
- If Engine is **not** easier, fix DX (item 1) before adding routes.
- If Engine **is** faster and easier on the measured path, then — and
  only then — deepen serving (keep-alive, multi-request `--engine` already
  exists; production hardening is later).

**Done when.** In-tree comparison (test or documented CLI) with TTFT/ITL/
energy. No claim of “vLLM-class” without those numbers.

**Do not.** OpenAI-compatible surface as this item. Do not match llama.cpp
tok/s as the headline.

### 8. [ ] Energy as a default score

**Why.** Civilization’s scarce resource is joules as much as dollars.
`energy_uj` (profile TDP × wall) already exists. It is not the front of
the report.

**Do.**

- Print joules/token (or `energy_uj` + `ns_per_token`) by default in
  `expertvm bench` and `infer-bench` summary lines.
- Keep rent unset on example profiles. Optional USD only with a **cited**
  list price on a named non-example profile.

**Done when.** Default bench output includes energy. Example profiles
still have `rent_usd_micros_per_hour == 0`. Tests lock that.

**Do not.** Invent `$ / M tokens` for `example_h100_sxm`.

### 9. [ ] Prefix and KV working set

**Why.** Prefill is often the expensive part. Interned prefixes and paged
KV are already in Engine / `expertvm kv`. They need a published working-
set result, not only unit tests.

**Do.**

- `infer-bench workload shared-prefix` (or Engine `--trace-out`) plus
  `expertvm kv` on a restricted KV byte budget.
- Show intern hits vs recompute, and gpu-sim bytes moved.

**Done when.** A table for shared-prefix vs two independent sequences
at the same KV budget.

**Do not.** Build Mooncake. Distributed KV is after expertvm has a
residency result (item 3).

### 10. [ ] Stop condition (paper or negative result)

**Why.** PLAN.md: if lookback-2 persist is real on a large MoE, there is
a paper and a crate. If not, the honest outcome is “residency is not
predictable enough; use static EP / more HBM.”

**Do.**

- Using item 3’s real trace: compare Markov lookback-2 vs LRU vs oracle
  at capacities that fit a mid-tier GPU (restricted HBM).
- If Markov is far above LRU and close to oracle, write the result
  (README + table + commands). That is the company-shaped outcome:
  lower HBM per token on rented silicon.
- If not, write the negative result in STATUS and stop iterating
  predictors on tinies.

**Done when.** A dated STATUS section with the table and a one-sentence
claim that matches the numbers.

**Do not.** Tune on `cycling.jsonl` and generalize.

---

## Mapping from PLAN.md

| PLAN.md intent | This file |
| --- | --- |
| `llama-rust` as correctness lab | Items 2, standing gates |
| `expertvm` as the wedge | Items 3, 4, 10 |
| `gpu-sim` for GPU-less agents | Items 5, 6 |
| `infer-bench` dual score | Items 7, 8, 9 |
| CUDA mechanical twins | Frozen unless a consumer path needs an invariant (item 5) |
| OpenAI HTTP veneer | Still forbidden as the next item |
| Occupancy SM counts | Still walled (item 5) |

PLAN.md Success (load GGUF, trust greedy, emit traces, optimize against
a deterministic GPU VM, eventually prove lower HBM per token) is the
same Success. This file is how agents get there without another thousand
identity wraps.

---

## First item to execute

Item **1**: researcher one-afternoon path. Start there. Do not open
`gpu-sim/src/sim.rs` to add `mem_surf_object_create` unless item 5’s
consumer-path rule is satisfied (it is not, today).
