//! Replay a trace through [`gpu_sim`] with H2D fills on miss.

use crate::access::{ExpertKey, Trace};
use crate::error::Error;
use crate::policy::Policy;
use crate::replay::{replay_keys, Touch, Walker};
use gpu_sim::{AllocId, DType, DeviceId, HardwareProfile, KernelKind, Sim, StreamId};
use std::collections::BTreeMap;

/// Simulated residency result: semantic score plus cache stats.
#[derive(Clone, Copy, Debug)]
pub struct SimReplay {
    /// Simulated nanoseconds.
    pub sim_ns: u64,
    /// Host↔device bytes moved.
    pub bytes_moved: u64,
    /// Peak HBM live bytes.
    pub hbm_peak: u64,
    /// Cache hits.
    pub hits: u64,
    /// Cache misses.
    pub misses: u64,
}

impl SimReplay {
    /// Single-line agent / CLI log.
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "sim_ns={} bytes_moved={} hbm_peak={} hits={} misses={}",
            self.sim_ns, self.bytes_moved, self.hbm_peak, self.hits, self.misses
        )
    }
}

/// Replay `trace` on `profile` with a `slots`-entry expert cache.
///
/// Each miss copies `bytes_per_expert` host→device, then a grouped GEMM runs
/// on the same stream (stream-ordered; no invented overlap). Hits skip the copy.
pub fn sim_replay(
    trace: &Trace,
    profile: HardwareProfile,
    slots: usize,
    policy: Policy,
    bytes_per_expert: u64,
    lookahead: usize,
) -> Result<SimReplay, Error> {
    let keys = trace.keys();
    let table = replay_keys(&keys, slots, policy, lookahead);
    let mut sim = Sim::new(profile);
    let d = DeviceId(0);
    let s = StreamId(0);
    let mut handles: BTreeMap<ExpertKey, AllocId> = BTreeMap::new();
    let mut w = Walker::new(&keys, slots, policy, lookahead);
    let bytes = bytes_per_expert.max(1);
    while let Some((key, touch)) = w.next_touch() {
        match touch {
            Touch::Hit => {
                let id = *handles.get(&key).ok_or(Error::Store("missing handle"))?;
                kernel(&mut sim, d, s, id)?;
            }
            Touch::Miss { evicted } => {
                if let Some(v) = evicted {
                    if let Some(id) = handles.remove(&v) {
                        sim.free(d, id, s)?;
                    }
                }
                if slots == 0 {
                    continue;
                }
                let id = sim.alloc(d, bytes, s)?;
                let _c = sim.memcpy_host_to_device(d, id, bytes, s)?;
                kernel(&mut sim, d, s, id)?;
                let _prev = handles.insert(key, id);
            }
        }
    }
    sim.synchronize()?;
    Ok(SimReplay {
        sim_ns: sim.clock_ns(),
        bytes_moved: sim.bytes_moved(),
        hbm_peak: sim.hbm_peak(),
        hits: table.hits,
        misses: table.misses,
    })
}

fn kernel(sim: &mut Sim, d: DeviceId, s: StreamId, id: AllocId) -> Result<(), Error> {
    let _k = sim.kernel(
        d,
        KernelKind::GroupedMoeGemm {
            experts: 1,
            tokens_per_expert: 1,
            hidden: 64,
            ff: 64,
            dtype: DType::Fp16,
        },
        &[id],
        &[id],
        s,
    )?;
    Ok(())
}
