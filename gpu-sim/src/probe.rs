//! Topology measurement. Same payload, different meshes, no invented dollars.

use crate::error::SimError;
use crate::ids::{DeviceId, StreamId};
use crate::profile::HardwareProfile;
use crate::sim::Sim;

/// One undirected GPU↔GPU attempt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct P2pProbe {
    /// Source GPU.
    pub src: DeviceId,
    /// Destination GPU.
    pub dst: DeviceId,
    /// Virtual nanoseconds for the D2D copy, or `None` when the mesh has no link.
    pub ns: Option<u64>,
}

/// H2D and P2P costs for every GPU in a profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TopologyProbe {
    /// Profile name.
    pub name: String,
    /// GPU count.
    pub n_gpus: usize,
    /// Host→device copy time per GPU, in `gpus` order.
    pub h2d_ns: Vec<u64>,
    /// Pairwise device-to-device samples (`i < j`).
    pub p2p: Vec<P2pProbe>,
}

impl TopologyProbe {
    /// Single-line agent / CLI log. Not a dollar figure.
    #[must_use]
    pub fn line(&self) -> String {
        let h2d = join_u64(&self.h2d_ns);
        let mut p2p = String::new();
        for (i, hop) in self.p2p.iter().enumerate() {
            if i > 0 {
                p2p.push(',');
            }
            p2p.push_str(&format!("{}->{}", hop.src.0, hop.dst.0));
            match hop.ns {
                Some(ns) => {
                    p2p.push(':');
                    p2p.push_str(&ns.to_string());
                }
                None => p2p.push_str(":none"),
            }
        }
        if p2p.is_empty() {
            p2p.push_str("none");
        }
        format!(
            "topology={} gpus={} h2d_ns={} p2p={}",
            self.name, self.n_gpus, h2d, p2p
        )
    }
}

/// Measure H2D to every GPU and D2D between every pair. Fresh [`Sim`] per hop.
pub fn probe_topology(profile: HardwareProfile, bytes: u64) -> Result<TopologyProbe, SimError> {
    let bytes = bytes.max(1);
    let n_gpus = profile.n_gpus();
    let mut h2d_ns = Vec::new();
    for gpu in &profile.gpus {
        h2d_ns.push(measure_h2d(profile.clone(), gpu.id, bytes)?);
    }
    let mut p2p = Vec::new();
    for (i, src) in profile.gpus.iter().enumerate() {
        for dst in profile.gpus.iter().skip(i.saturating_add(1)) {
            p2p.push(measure_p2p(profile.clone(), src.id, dst.id, bytes)?);
        }
    }
    Ok(TopologyProbe {
        name: profile.name.clone(),
        n_gpus,
        h2d_ns,
        p2p,
    })
}

fn measure_h2d(profile: HardwareProfile, device: DeviceId, bytes: u64) -> Result<u64, SimError> {
    let mut sim = Sim::new(profile);
    let s = StreamId(0);
    let a = sim.alloc(device, bytes, s)?;
    let _c = sim.memcpy_host_to_device(device, a, bytes, s)?;
    sim.synchronize()?;
    Ok(sim.clock_ns())
}

fn measure_p2p(
    profile: HardwareProfile,
    src: DeviceId,
    dst: DeviceId,
    bytes: u64,
) -> Result<P2pProbe, SimError> {
    let mut sim = Sim::new(profile);
    let s = StreamId(0);
    let a = sim.alloc(src, bytes, s)?;
    let _h = sim.memcpy_host_to_device(src, a, bytes, s)?;
    sim.synchronize()?;
    let t0 = sim.clock_ns();
    match sim.memcpy_device_to_device(src, dst, a, bytes, s) {
        Ok(_) => {}
        Err(SimError::NoPeer { .. }) => {
            return Ok(P2pProbe { src, dst, ns: None });
        }
        Err(e) => return Err(e),
    }
    match sim.synchronize() {
        Ok(()) => Ok(P2pProbe {
            src,
            dst,
            ns: Some(sim.clock_ns().saturating_sub(t0)),
        }),
        Err(SimError::NoPeer { .. }) => Ok(P2pProbe { src, dst, ns: None }),
        Err(e) => Err(e),
    }
}

fn join_u64(xs: &[u64]) -> String {
    let mut out = String::new();
    for (i, n) in xs.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&n.to_string());
    }
    out
}
