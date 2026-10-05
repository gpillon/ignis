//! What expert residency counts (spec flash-next/03): hits and misses per
//! class and phase, prefetches issued and used, bytes moved, the time the
//! expert kernels waited, from the first step on. Facts only: the server's
//! exposition names and renders them (ADR 0017), and nothing here knows it
//! exists.

use super::class::KClass;

/// Everything residency counts, from the first step on. `[class][phase]`
/// arrays are indexed like [`KClass::ALL`] and
/// [`Phase::ALL`](super::policy::Phase::ALL).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResidencyCounters {
    /// Selected projections already resident, or staged for their layer.
    pub hits: [[u64; 2]; KClass::COUNT],
    /// Selected projections copied in by their own step.
    pub misses: [[u64; 2]; KClass::COUNT],
    /// Projections copied ahead for the next layer.
    pub prefetch_issued: u64,
    /// Prefetched projections at their first use, whenever it comes.
    pub prefetch_used: u64,
    /// Bytes copied host-to-device, misses and prefetches, by phase.
    pub bytes_moved: [u64; 2],
    /// Time the expert kernels waited on residency. The GPU side measures
    /// it; the CPU policy model has no time and leaves it at zero.
    pub stall_nanos: u64,
}
