//! Shared Engine ExpertStore attach for `engine` and `serve --engine`.

use crate::decode::Llama;
use crate::engine::Engine;
use expertvm::{
    CachedStore, GpuFill, GpuStoreCfg, HardwareProfile, LiveStore, Prefetch, SimulatedGpuStore,
};

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
    /// Simulated KV page bytes when [`GpuStoreCfg::kv_sim`]. `None` uses intern geometry.
    pub kv_bytes: Option<u64>,
}

/// CLI switches for [`gpu_knobs`] / [`GpuFill`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GpuCli {
    pub cuda_graphs: bool,
    pub graph_update: bool,
    pub graph_set_params: bool,
    pub graph_clone: bool,
    pub graph_build: bool,
    pub graph_piecewise: bool,
    pub graph_mem: bool,
    pub graph_auto_free: bool,
    pub timing_events: bool,
    pub mapped: bool,
    pub managed: bool,
    pub vmm: bool,
    pub host_func: bool,
    pub blocking_streams: bool,
    pub sync_alloc: bool,
    pub mempool: bool,
    /// POSIX-FD shareable mempool IPC (`GpuStoreCfg::shareable`). Implies mempool.
    pub shareable: bool,
    pub pageable: bool,
    pub accessed_by: bool,
    pub legacy_null: bool,
    pub stream_priority: bool,
    /// Per-sequence copy streams (`GpuStoreCfg::seq_streams`).
    pub seq_streams: bool,
    /// Engine interned KV on the SimulatedGpuStore clock (`GpuStoreCfg::kv_sim`).
    pub kv_sim: bool,
    /// Decode GEMMs on a higher-priority compute stream (`GpuStoreCfg::decode_priority`).
    pub decode_priority: bool,
    /// `cudaLaunchCooperativeKernel` (`GpuStoreCfg::cooperative`).
    pub cooperative: bool,
    /// Same-stream PDL wait+trigger (`GpuStoreCfg::pdl`).
    pub pdl: bool,
    /// Hopper NVLS replica fanout (`GpuStoreCfg::multicast`). Implies vmm.
    pub multicast: bool,
    /// Hyper-Q occupancy (`GpuStoreCfg::compute_slots`). `0` keeps the profile.
    pub compute_slots: u8,
    /// True when `--compute-slots` appeared.
    pub compute_slots_set: bool,
    /// Decode-stream SM permille (`GpuStoreCfg::decode_sm_permille`).
    pub decode_sm_permille: u16,
    /// True when `--decode-sms` appeared.
    pub decode_sm_set: bool,
    /// `--kv-bytes` override. `None` uses intern K+V geometry.
    pub kv_bytes: Option<u64>,
    /// Physical span for [`GpuFill::Vmm`]. `0` maps the whole expert.
    pub vmm_page: u64,
    /// True when `--vmm-page` appeared (even if the value is `0`).
    pub vmm_page_set: bool,
}

impl GpuCli {
    /// True when `key` is a GPU switch. Inline values are refused.
    pub(crate) fn apply(&mut self, key: &str, inline: Option<&str>) -> Result<bool, String> {
        let slot = match key {
            "--cuda-graphs" => &mut self.cuda_graphs,
            "--graph-update" => &mut self.graph_update,
            "--graph-set-params" => &mut self.graph_set_params,
            "--graph-clone" => &mut self.graph_clone,
            "--graph-build" => &mut self.graph_build,
            "--graph-piecewise" => &mut self.graph_piecewise,
            "--graph-mem" => &mut self.graph_mem,
            "--graph-auto-free" => &mut self.graph_auto_free,
            "--timing-events" => &mut self.timing_events,
            "--mapped" => &mut self.mapped,
            "--managed" => &mut self.managed,
            "--vmm" => &mut self.vmm,
            "--host-func" => &mut self.host_func,
            "--blocking-streams" => &mut self.blocking_streams,
            "--sync-alloc" => &mut self.sync_alloc,
            "--mempool" => &mut self.mempool,
            "--shareable" => &mut self.shareable,
            "--pageable" => &mut self.pageable,
            "--accessed-by" => &mut self.accessed_by,
            "--legacy-null" => &mut self.legacy_null,
            "--stream-priority" => &mut self.stream_priority,
            "--seq-streams" => &mut self.seq_streams,
            "--kv-sim" => &mut self.kv_sim,
            "--decode-priority" => &mut self.decode_priority,
            "--cooperative" => &mut self.cooperative,
            "--pdl" => &mut self.pdl,
            "--multicast" => &mut self.multicast,
            _ => return Ok(false),
        };
        if inline.is_some() {
            return Err(format!("{key} does not take a value"));
        }
        *slot = true;
        Ok(true)
    }

    /// `N>0` maps experts in `N`-byte physicals (`va_acquire_paged`).
    pub(crate) fn set_vmm_page(&mut self, n: u64) {
        self.vmm_page = n;
        self.vmm_page_set = true;
    }

    /// `--vmm-page N` with `N>0` implies [`Self::vmm`]. Call after sim-flag checks.
    pub(crate) fn imply_vmm(&mut self) {
        if self.vmm_page > 0 || self.multicast {
            self.vmm = true;
        }
    }

    /// `--shareable` implies [`Self::mempool`]. Call after sim-flag checks.
    pub(crate) fn imply_shareable(&mut self) {
        if self.shareable {
            self.mempool = true;
        }
    }

    /// `--decode-priority` implies [`Self::stream_priority`]. `--decode-sms`
    /// implies both. Call after sim-flag checks.
    pub(crate) fn imply_decode_priority(&mut self) {
        if self.decode_sm_permille > 0 {
            self.decode_priority = true;
        }
        if self.decode_priority {
            self.stream_priority = true;
        }
    }

    /// Decode-stream SM permille (`--decode-sms`). `0` and `>1000` are refused.
    pub(crate) fn set_decode_sms(&mut self, n: u16) -> Result<(), String> {
        if n == 0 || n > 1000 {
            return Err("decode-sms must be 1..=1000".into());
        }
        self.decode_sm_permille = n;
        self.decode_sm_set = true;
        Ok(())
    }

    /// KV page bytes (`--kv-bytes`). `0` is refused.
    pub(crate) fn set_kv_bytes(&mut self, n: u64) -> Result<(), String> {
        if n == 0 {
            return Err("kv-bytes must be > 0".into());
        }
        self.kv_bytes = Some(n);
        Ok(())
    }

    /// Hyper-Q occupancy (`--compute-slots`). `0` is refused.
    pub(crate) fn set_compute_slots(&mut self, n: u8) -> Result<(), String> {
        if n == 0 {
            return Err("compute-slots must be > 0".into());
        }
        self.compute_slots = n;
        self.compute_slots_set = true;
        Ok(())
    }

    /// First CUDA knob that needs `--expert-sim`, if any.
    #[must_use]
    pub(crate) fn sim_flag(self) -> Option<&'static str> {
        [
            (self.cuda_graphs, "--cuda-graphs"),
            (self.graph_update, "--graph-update"),
            (self.graph_set_params, "--graph-set-params"),
            (self.graph_clone, "--graph-clone"),
            (self.graph_build, "--graph-build"),
            (self.graph_piecewise, "--graph-piecewise"),
            (self.graph_mem, "--graph-mem"),
            (self.graph_auto_free, "--graph-auto-free"),
            (self.timing_events, "--timing-events"),
            (self.mapped, "--mapped"),
            (self.managed, "--managed"),
            (self.vmm, "--vmm"),
            (self.host_func, "--host-func"),
            (self.blocking_streams, "--blocking-streams"),
            (self.sync_alloc, "--sync-alloc"),
            (self.mempool, "--mempool"),
            (self.shareable, "--shareable"),
            (self.pageable, "--pageable"),
            (self.accessed_by, "--accessed-by"),
            (self.legacy_null, "--legacy-null"),
            (self.stream_priority, "--stream-priority"),
            (self.seq_streams, "--seq-streams"),
            (self.kv_sim, "--kv-sim"),
            (self.decode_priority, "--decode-priority"),
            (self.cooperative, "--cooperative"),
            (self.pdl, "--pdl"),
            (self.multicast, "--multicast"),
            (self.vmm_page_set, "--vmm-page"),
            (self.compute_slots_set, "--compute-slots"),
            (self.decode_sm_set, "--decode-sms"),
        ]
        .into_iter()
        .find_map(|(on, name)| on.then_some(name))
    }

    /// Pinned when every fill flag is off; otherwise exactly one of mapped/managed/vmm.
    pub(crate) fn fill(self) -> Result<GpuFill, String> {
        GpuFill::from_flags(self.mapped, self.managed, self.vmm).map_err(|e| e.to_string())
    }
}

/// GPU knobs plus predictor planner (`--prefetch` / Stay vs Fetch).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlannerCli {
    /// SimulatedGpuStore CUDA / fill flags.
    pub gpu: GpuCli,
    /// Predictor mode. Default [`Prefetch::Both`].
    pub prefetch: Prefetch,
    /// Unique predicted-key Stay vs Fetch window. `0` is ungated.
    pub plan_window: usize,
    /// Stay permille of that window already resident.
    pub plan_threshold: u32,
    prefetch_set: bool,
    window_set: bool,
    threshold_set: bool,
}

impl Default for PlannerCli {
    fn default() -> Self {
        Self {
            gpu: GpuCli::default(),
            prefetch: Prefetch::Both,
            plan_window: 0,
            plan_threshold: 500,
            prefetch_set: false,
            window_set: false,
            threshold_set: false,
        }
    }
}

/// Result of [`PlannerCli::take`] classifying one dashed operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dash {
    Taken,
    Need(PlanSlot),
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlanSlot {
    VmmPage,
    KvBytes,
    ComputeSlots,
    DecodeSms,
    Prefetch,
    PlanWindow,
    PlanThreshold,
}

impl PlanSlot {
    fn name(self) -> &'static str {
        match self {
            Self::VmmPage => "vmm-page",
            Self::KvBytes => "kv-bytes",
            Self::ComputeSlots => "compute-slots",
            Self::DecodeSms => "decode-sms",
            Self::Prefetch => "prefetch",
            Self::PlanWindow => "plan-window",
            Self::PlanThreshold => "plan-threshold",
        }
    }
}

impl PlannerCli {
    fn dash(&mut self, key: &str, inline: Option<&str>) -> Result<Dash, String> {
        if self.gpu.apply(key, inline)? {
            return Ok(Dash::Taken);
        }
        Ok(match key {
            "--vmm-page" => Dash::Need(PlanSlot::VmmPage),
            "--kv-bytes" => Dash::Need(PlanSlot::KvBytes),
            "--compute-slots" => Dash::Need(PlanSlot::ComputeSlots),
            "--decode-sms" => Dash::Need(PlanSlot::DecodeSms),
            "--prefetch" => Dash::Need(PlanSlot::Prefetch),
            "--plan-window" => Dash::Need(PlanSlot::PlanWindow),
            "--plan-threshold" => Dash::Need(PlanSlot::PlanThreshold),
            _ => Dash::Unknown,
        })
    }

    fn set(&mut self, slot: PlanSlot, raw: &str) -> Result<(), String> {
        match slot {
            PlanSlot::VmmPage => {
                let n = raw
                    .parse::<u64>()
                    .map_err(|_| format!("invalid vmm-page {raw:?}"))?;
                self.gpu.set_vmm_page(n);
            }
            PlanSlot::KvBytes => {
                let n = raw
                    .parse::<u64>()
                    .map_err(|_| format!("invalid kv-bytes {raw:?}"))?;
                self.gpu.set_kv_bytes(n)?;
            }
            PlanSlot::ComputeSlots => {
                let n = raw
                    .parse::<u8>()
                    .map_err(|_| format!("invalid compute-slots {raw:?}"))?;
                self.gpu.set_compute_slots(n)?;
            }
            PlanSlot::DecodeSms => {
                let n = raw
                    .parse::<u16>()
                    .map_err(|_| format!("invalid decode-sms {raw:?}"))?;
                self.gpu.set_decode_sms(n)?;
            }
            PlanSlot::Prefetch => {
                self.prefetch =
                    Prefetch::parse(raw).map_err(|_| format!("unknown prefetch {raw}"))?;
                self.prefetch_set = true;
            }
            PlanSlot::PlanWindow => {
                self.plan_window = raw
                    .parse::<usize>()
                    .map_err(|_| format!("invalid plan-window {raw:?}"))?;
                self.window_set = true;
            }
            PlanSlot::PlanThreshold => {
                self.plan_threshold = raw
                    .parse::<u32>()
                    .map_err(|_| format!("invalid plan-threshold {raw:?}"))?;
                self.threshold_set = true;
            }
        }
        Ok(())
    }

    /// Consume a dashed Engine / serve flag. `false` means the key is unknown.
    pub(crate) fn take<I, S>(
        &mut self,
        key: &str,
        inline: Option<&str>,
        it: &mut I,
    ) -> Result<bool, String>
    where
        I: Iterator<Item = S>,
        S: AsRef<str>,
    {
        match self.dash(key, inline)? {
            Dash::Taken => Ok(true),
            Dash::Need(slot) => {
                let v = match inline {
                    Some(v) => v.to_string(),
                    None => match it.next() {
                        Some(s) => s.as_ref().to_string(),
                        None => return Err(format!("missing --{} value", slot.name())),
                    },
                };
                self.set(slot, &v)?;
                Ok(true)
            }
            Dash::Unknown => Ok(false),
        }
    }

    /// First planner flag that needs `--engine` on `serve`.
    #[must_use]
    pub(crate) fn serve_engine_flag(self) -> Option<&'static str> {
        [
            (self.prefetch_set, "--prefetch"),
            (self.window_set, "--plan-window"),
            (self.threshold_set, "--plan-threshold"),
        ]
        .into_iter()
        .find_map(|(on, name)| on.then_some(name))
    }
}

/// Build [`GpuStoreCfg`] from parsed Engine / serve GPU flags.
#[must_use]
pub(crate) fn gpu_knobs(gpu: GpuCli) -> GpuStoreCfg {
    GpuStoreCfg {
        host_func: gpu.host_func,
        blocking_streams: gpu.blocking_streams,
        sync_alloc: gpu.sync_alloc,
        mempool: gpu.mempool,
        shareable: gpu.shareable,
        vmm_page: gpu.vmm_page,
        pageable: gpu.pageable,
        accessed_by: gpu.accessed_by,
        legacy_null: gpu.legacy_null,
        stream_priority: gpu.stream_priority,
        graph_update: gpu.graph_update,
        graph_set_params: gpu.graph_set_params,
        graph_clone: gpu.graph_clone,
        graph_build: gpu.graph_build,
        graph_piecewise: gpu.graph_piecewise,
        graph_mem: gpu.graph_mem,
        graph_auto_free: gpu.graph_auto_free,
        timing_events: gpu.timing_events,
        seq_streams: gpu.seq_streams,
        kv_sim: gpu.kv_sim,
        decode_priority: gpu.decode_priority,
        cooperative: gpu.cooperative,
        pdl: gpu.pdl,
        multicast: gpu.multicast,
        compute_slots: gpu.compute_slots,
        decode_sm_permille: gpu.decode_sm_permille,
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
        if spec.gpu_cfg.kv_sim {
            eng.enable_kv_sim(spec.kv_bytes)
                .map_err(|e| e.to_string())?;
        }
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
