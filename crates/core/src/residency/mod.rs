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
//! - [`plan`]: the host plan and residency's VRAM plan lines (ADR 0030),
//!   and [`host_memory`], the host plan's one measured input;
//! - [`counters`]: what residency counts, for the server to export.
//!
//! # The device side (designed, not built yet)
//!
//! The miss path was measured first (spec acceptance 3,
//! `docs/findings/2026-10-05-expert-miss-path-sm-copy-matches-the-copy-engine.md`):
//! an SM-driven copy from mapped pinned memory reaches 88-112% of the copy
//! engine, so residency is **device-resident** and decode stays one CUDA
//! graph. Every structure below belongs to the loaded Flash-Next model and
//! is freed in its drop path; none is process-wide.
//!
//! - **Host pool:** one `cudaHostAlloc(Mapped | Portable)` of
//!   [`ExpertCatalog::total_bytes`], filled at load layer by layer from the
//!   artifact's expert index (offset per (layer, expert, projection)).
//! - **Device, reserved at load as [`ExpertCachePlan`]'s lines:** the eight
//!   class pools; the prefill staging ring; the tables — per projection its
//!   slot-table entry (device address and K, what spec 02's expert kernels
//!   read) and its slot (or none), per slot its owner key and stamp, and the
//!   clock.
//! - **Per layer step, on the compute stream:** a *resolve* kernel takes the
//!   router's selection (and the next router's lookahead) and does exactly
//!   [`ResidencyModel::step`] — hits stamped, victims the first unpinned
//!   slots by `(protected, stamp, key)` with key `layer · 1024 + expert · 2
//!   + projection` (the canonical order), misses and prefetches written as
//!   copy jobs and into the slot table; a *copy* kernel of about 8-16 blocks
//!   (the finding's best grid) moves the jobs from the mapped pool; the
//!   expert kernel follows. Prefetch jobs run on a second captured stream
//!   beside the expert kernel and join before the next resolve, held to a
//!   per-step byte budget: unbudgeted, W = 16 asks the link for more than a
//!   decode step lasts (7.2 ms of transfers per 6 ms step at one lane, 29
//!   at three, on the study's routing). Launch shapes are fixed; the job
//!   count lives in device memory.
//! - **Tested against this module:** the GPU replay drives resolve and copy
//!   with recorded traces and reads back each step's hits, misses and
//!   evictions, which must equal [`StepOutcome`]'s.
//! - **Counts:** the resolve kernel keeps [`ResidencyCounters`]' counts in a
//!   small device block the telemetry interval reads off the critical path.

pub mod class;
pub mod counters;
pub mod host_memory;
pub mod plan;
pub mod policy;

pub use class::{CatalogMismatch, ExpertCatalog, KBits, KClass, Projection, ProjectionId};
pub use host_memory::available_physical_bytes;
pub use counters::ResidencyCounters;
pub use plan::{
    ClassPool, EXPERT_CACHE_FLOOR_BYTES, ExpertCachePlan, ExpertCachePlanError,
    ExpertCacheRequest, ExpertTraffic, HOST_MARGIN_BYTES, HostPlan, HostPlanError,
    HostPlanRequest, MEASURED_LINK_BYTES_PER_SECOND, TrafficMismatch,
    default_prefetch_budget_bytes, estimated_decode_round_seconds, min_slots_per_class,
    plan_expert_cache, plan_host, prefill_staging_ring_bytes, residency_table_bytes,
    warm_start_order,
};
pub use policy::{
    Admission, DEFAULT_PREFETCH_WIDTH, LayerStep, Phase, PolicyConfig, ResidencyModel, StepError,
    StepOutcome,
};
