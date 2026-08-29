//! Shared Engine ExpertStore attach for `engine` and `serve --engine`.

use crate::decode::Llama;
use crate::engine::Engine;
use expertvm::{CachedStore, GpuFill, GpuStoreCfg, HardwareProfile, LiveStore, SimulatedGpuStore};

/// CLI knobs that build a [`LiveStore`] for an [`Engine`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StoreAttach {
    /// `None` keeps blob FFN. `Some(0)` is DirectStore. `Some(n)` is CachedStore.
    pub expert_slots: Option<usize>,
    /// SimulatedGpuStore instead of Direct/Cached.
    pub expert_sim: bool,
    /// Example 8×H100 NVLink profile (`expert_sim`).
    pub expert_8gpu: bool,
    /// Simulated expert page bytes (`expert_sim`). `None` is 4096.
    pub expert_bytes: Option<u64>,
    /// CUDA-like knobs for [`SimulatedGpuStore::with_cfg`]. Identity stays default.
    pub gpu_cfg: GpuStoreCfg,
    /// Miss-page placement. Default is pinned H2D.
    pub fill: GpuFill,
}

/// CLI graph/timing switches for [`gpu_knobs`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GpuCli {
    pub cuda_graphs: bool,
    pub graph_update: bool,
    pub graph_clone: bool,
    pub timing_events: bool,
    pub mapped: bool,
    pub managed: bool,
    pub vmm: bool,
}

impl GpuCli {
    /// True when `key` is a graph/timing switch. Inline values are refused.
    pub(crate) fn apply(&mut self, key: &str, inline: Option<&str>) -> Result<bool, String> {
        let slot = match key {
            "--cuda-graphs" => &mut self.cuda_graphs,
            "--graph-update" => &mut self.graph_update,
            "--graph-clone" => &mut self.graph_clone,
            "--timing-events" => &mut self.timing_events,
            "--mapped" => &mut self.mapped,
            "--managed" => &mut self.managed,
            "--vmm" => &mut self.vmm,
            _ => return Ok(false),
        };
        if inline.is_some() {
            return Err(format!("{key} does not take a value"));
        }
        *slot = true;
        Ok(true)
    }

    /// Pinned when every fill flag is off; otherwise exactly one of mapped/managed/vmm.
    pub(crate) fn fill(self) -> Result<GpuFill, String> {
        GpuFill::from_flags(self.mapped, self.managed, self.vmm).map_err(|e| e.to_string())
    }
}

/// Build graph/timing knobs for [`StoreAttach::gpu_cfg`].
#[must_use]
pub(crate) fn gpu_knobs(graph_update: bool, graph_clone: bool, timing_events: bool) -> GpuStoreCfg {
    GpuStoreCfg {
        graph_update,
        graph_clone,
        timing_events,
        ..GpuStoreCfg::default()
    }
}

/// Park DirectStore / CachedStore / SimulatedGpuStore on `eng`.
pub(crate) fn attach_store(
    eng: &mut Engine<'_>,
    llama: &Llama,
    spec: &StoreAttach,
) -> Result<(), String> {
    if spec.expert_sim {
        let slots = match spec.expert_slots {
            Some(0) => return Err("--expert-sim needs --expert-slots > 0".into()),
            Some(n) => n,
            None => 8,
        };
        let profile = if spec.expert_8gpu {
            HardwareProfile::example_8xh100_nvlink()
        } else {
            HardwareProfile::example_h100_sxm()
        };
        let gpu = SimulatedGpuStore::with_cfg(
            llama.expert_direct_store().map_err(|e| e.to_string())?,
            slots,
            profile,
            spec.expert_bytes.unwrap_or(4096),
            spec.fill,
            spec.gpu_cfg,
        )
        .map_err(|e| e.to_string())?;
        eng.attach_expert_store(LiveStore::simulated(gpu));
        return Ok(());
    }
    let Some(slots) = spec.expert_slots else {
        return Ok(());
    };
    let direct = llama.expert_direct_store().map_err(|e| e.to_string())?;
    let store = if slots == 0 {
        LiveStore::Direct(direct)
    } else {
        LiveStore::Cached(CachedStore::new(direct, slots).map_err(|e| e.to_string())?)
    };
    eng.attach_expert_store(store);
    Ok(())
}
