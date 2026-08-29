# gpu-sim

Deterministic **GPU-systems** virtual machine for inference software.

Agents and researchers can optimize expert residency, prefetch, and
scheduling against this crate **without a physical GPU**. Semantics are
exact. Timing comes from a [`HardwareProfile`](crate::HardwareProfile),
not from folklore baked into policy code.

This is **not** Accel-Sim / GPGPU-Sim. It does not model warps, register
banks, or Tensor Core circuitry. Those effects enter only as empirical
kernel curves in a profile.

```text
inference system
        │
GPU systems semantics   ← exact (streams, events, residency, OOM, topology)
        │
resource contention     ← copy engines, links, exclusive compute
        │
calibrated op costs     ← HardwareProfile
        │
       STOP
─────────────────────────
warp scheduler, L1, …   ← do not model
```

## Exact vs timed

| Exact (tests fail if violated) | Timed (profile data) |
| --- | --- |
| HBM capacity, alloc/free, OOM | kernel microseconds |
| stream order, event dependencies | memcpy microseconds |
| residency: a kernel may only read local allocations | PCIe / NVLink bandwidth |
| copy-engine occupancy | launch overhead |
| peer accessibility | size-dependent efficiency |
| graph capture does not execute; launch replays | — |

## Anti-Goodhart timing

Memcpy cost is

```text
T = T_fixed + (bytes + ramp) / peak_bandwidth
```

Eight thousand tiny copies cannot harvest full PCIe bandwidth. Concurrent
copies on the same link share bandwidth. Kernels on one GPU are exclusive
in v0 (copy engines still overlap compute).

## Example profiles

`profiles/*.profile` are **examples for tests and development**. They are
order-of-magnitude public-spec numbers, not a capture from a real node.
Replace them with `HardwareProfile::parse` of a `gpu-profile capture`
once someone with a GPU runs one.

## Library

```rust
use gpu_sim::{DeviceId, HardwareProfile, KernelKind, Sim, StreamId};

let mut sim = Sim::new(HardwareProfile::example_h100_sxm());
let d0 = DeviceId(0);
let s0 = StreamId(0);
let a = sim.alloc(d0, 8 << 20, s0).expect("hbm");
sim.memcpy_host_to_device(d0, a, 8 << 20, s0).expect("h2d");
sim.kernel(d0, KernelKind::other(1 << 30, 8 << 20), &[a], &[a], s0)
    .expect("k");
sim.synchronize().expect("sync");
assert!(sim.clock_ns() > 0);
```

## Scores

[`Score`](crate::Score) splits the two numbers agents must not mix:

- **semantic**: `Ok` or a [`SimError`](crate::SimError) (binary)
- **performance**: `wall_ns`, `hbm_peak`, `bytes_moved`, `energy_uj`, optional
  `ns_per_token`, `ttft_ns`, `itl_ns`

`Score::with_tokens(n)` fills `ns_per_token = wall_ns / n`.
`Score::with_latencies` attaches TTFT / mean ITL when a caller samples the
virtual clock at token boundaries (`expertvm::sim_replay` does this).
`energy_uj` is profile board TDP × virtual wall (`mW × ns / 1e6`). There is
no invented `$/M tokens` field. Device-to-device replica copies charge the
destination HBM at memcpy start and OOM if that GPU is full. `free` of a
replicated allocation only drops `live` when no device still holds it.

Adversarial memcpy: many tiny copies cannot beat one large copy of the
same payload (fixed overhead + size-dependent bandwidth). Two concurrent
H2D copies on separate streams share PCIe and cannot finish in one-copy
time.

## Topologies

Named example profiles (order-of-magnitude, **not captures**):

| Name | Mesh |
| --- | --- |
| `h100` / `h200` / `cheap` | 1 GPU, host PCIe |
| `2xh100-pcie` | 2 GPU, PCIe P2P (no NVLink) |
| `8xh100` | 8 GPU NVLink clique + per-GPU PCIe |
| `bad-numa` | 2 GPU, GPU1 on a slow far PCIe root, no P2P |
| `2node-rdma` | 2 GPU, GPU-direct RDMA, no NVLink |
| `asymmetric` | 3 GPU NVLink chain 0–1–2; 0↔2 is `NoPeer` |

`HardwareProfile::by_name`, `probe_topology`, and `gpu-profile probe NAME`
measure H2D per GPU and D2D per pair. Missing links print `p2p=0->2:none`.
`restrict_hbm(bytes)` caps every GPU for the static-EP vs cache experiment.

## Faults

| Inject | Semantic error |
| --- | --- |
| `Sim::set_unavailable` | `SimError::Unavailable` |
| `Sim::cancel_stream` (queued ops) | `SimError::Cancelled` |
| `Sim::fail_next_memcpy` | `SimError::TransferFailed` (expert load) |
| `Sim::set_extra_transfer_ns` | longer memcpy / allreduce, still `Ok` |
| over-capacity alloc | `SimError::Oom` |

CUDA graphs: `begin_capture` / `end_capture` / `launch_graph`. Capture does
not advance the virtual clock. Alloc/free cannot be captured.

In-flight ops are not cancelled. `gpu-profile capture` is refused in this
crate: someone with a GPU writes a `key=value` file; agents `parse` it.
