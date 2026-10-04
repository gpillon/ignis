//! Expert residency's metric families (ADR 0017 naming, spec flash-next/03
//! "Metrics") and their CPU-side accounting.
//!
//! Every family is fixed-cardinality: `class` takes the eight
//! [`KClass::as_str`] spellings, `phase` takes `decode|prefill`, `state`
//! takes `capacity|in_use`. Counters end in `_total`; occupancy is a gauge
//! in slots, never a ratio (ADR 0030 §Observability: a ratio hides which
//! term moved). The Monitor wiring and the exposition come with the serving
//! work (spec flash-next/04); this module fixes the contract and keeps the
//! counts.

use super::class::KClass;
use super::policy::Phase;

/// One metric family of the contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetricFamily {
    pub name: &'static str,
    /// `counter` or `gauge`.
    pub kind: &'static str,
    pub labels: &'static [&'static str],
    pub help: &'static str,
}

/// The residency families, in exposition order.
pub const FAMILIES: [MetricFamily; 7] = [
    MetricFamily {
        name: "ignis_expert_cache_hits_total",
        kind: "counter",
        labels: &["class"],
        help: "Selected expert projections already resident in the VRAM expert cache (or staged for their layer), by K class.",
    },
    MetricFamily {
        name: "ignis_expert_cache_misses_total",
        kind: "counter",
        labels: &["class", "phase"],
        help: "Selected expert projections copied in from the pinned host pool, by K class and by decode or prefill.",
    },
    MetricFamily {
        name: "ignis_expert_prefetches_issued_total",
        kind: "counter",
        labels: &[],
        help: "Expert projections copied ahead for the next layer by the router lookahead.",
    },
    MetricFamily {
        name: "ignis_expert_prefetches_used_total",
        kind: "counter",
        labels: &[],
        help: "Prefetched expert projections that the next layer then selected.",
    },
    MetricFamily {
        name: "ignis_expert_bytes_moved_total",
        kind: "counter",
        labels: &["phase"],
        help: "Bytes of expert projections copied host-to-device, misses and prefetches, by decode or prefill.",
    },
    MetricFamily {
        name: "ignis_expert_residency_stall_seconds_total",
        kind: "counter",
        labels: &[],
        help: "Time the expert kernels waited on residency to bring their projections in.",
    },
    MetricFamily {
        name: "ignis_expert_cache_slots",
        kind: "gauge",
        labels: &["class", "state"],
        help: "VRAM expert cache slots per K class: capacity reserved at load, and in use.",
    },
];

/// Everything residency counts, from the first step on.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResidencyCounters {
    /// Indexed like [`KClass::ALL`].
    pub hits: [u64; KClass::COUNT],
    /// `[class][phase]`, indexed like [`KClass::ALL`] and [`Phase::ALL`].
    pub misses: [[u64; 2]; KClass::COUNT],
    pub prefetch_issued: u64,
    pub prefetch_used: u64,
    /// Indexed like [`Phase::ALL`].
    pub bytes_moved: [u64; 2],
    /// Time the expert kernels waited on residency. The GPU side measures
    /// it; the CPU policy model has no time and leaves it at zero.
    pub stall_nanos: u64,
}

/// One exposition series: a family, its label values, its value.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub name: &'static str,
    pub labels: Vec<(&'static str, &'static str)>,
    pub value: f64,
}

impl ResidencyCounters {
    /// Every series of [`FAMILIES`], zeros included, with the pools' slot
    /// capacity and occupancy (indexed like [`KClass::ALL`]).
    pub fn samples(
        &self,
        capacity: &[u32; KClass::COUNT],
        occupancy: &[u32; KClass::COUNT],
    ) -> Vec<Sample> {
        let mut out = Vec::new();
        for class in KClass::ALL {
            out.push(Sample {
                name: FAMILIES[0].name,
                labels: vec![("class", class.as_str())],
                value: self.hits[class.index()] as f64,
            });
        }
        for class in KClass::ALL {
            for phase in Phase::ALL {
                out.push(Sample {
                    name: FAMILIES[1].name,
                    labels: vec![("class", class.as_str()), ("phase", phase.as_str())],
                    value: self.misses[class.index()][phase.index()] as f64,
                });
            }
        }
        out.push(Sample {
            name: FAMILIES[2].name,
            labels: vec![],
            value: self.prefetch_issued as f64,
        });
        out.push(Sample {
            name: FAMILIES[3].name,
            labels: vec![],
            value: self.prefetch_used as f64,
        });
        for phase in Phase::ALL {
            out.push(Sample {
                name: FAMILIES[4].name,
                labels: vec![("phase", phase.as_str())],
                value: self.bytes_moved[phase.index()] as f64,
            });
        }
        out.push(Sample {
            name: FAMILIES[5].name,
            labels: vec![],
            value: self.stall_nanos as f64 * 1e-9,
        });
        for class in KClass::ALL {
            for (state, slots) in [("capacity", capacity), ("in_use", occupancy)] {
                out.push(Sample {
                    name: FAMILIES[6].name,
                    labels: vec![("class", class.as_str()), ("state", state)],
                    value: f64::from(slots[class.index()]),
                });
            }
        }
        out
    }
}
