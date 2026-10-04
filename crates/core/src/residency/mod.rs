//! **Expert residency** for Flash-Next (spec flash-next/03, GitHub #301):
//! where each expert projection lives at each moment — all of them in one
//! pinned host pool for the life of the load, a fixed-size cache of them in
//! VRAM replaced by recency, and the ones the next layer will probably need
//! copied ahead by the next layer's router.
//!
//! This module is the CPU side, with no device code:
//! - [`class`]: the unit residency moves (one expert projection) and its
//!   eight K classes;
//! - [`policy`]: the replacement policy as a deterministic, trace-driven
//!   model — the contract the GPU implementation is tested against;
//! - [`metrics`]: the metric families and their accounting.

pub mod class;
pub mod metrics;
pub mod policy;

pub use class::{CatalogMismatch, ExpertCatalog, KBits, KClass, Projection, ProjectionId};
pub use metrics::{FAMILIES, MetricFamily, ResidencyCounters, Sample};
pub use policy::{
    Admission, DEFAULT_PREFETCH_WIDTH, LayerStep, Phase, PolicyConfig, ResidencyModel, StepError,
    StepOutcome,
};
