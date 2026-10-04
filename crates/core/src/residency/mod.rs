//! **Expert residency** for Flash-Next (spec flash-next/03, GitHub #301):
//! where each expert projection lives at each moment — all of them in one
//! pinned host pool for the life of the load, a fixed-size cache of them in
//! VRAM replaced by recency, and the ones the next layer will probably need
//! copied ahead by the next layer's router.
//!
//! This module is the CPU side, with no device code: the unit and its eight
//! classes ([`class`]). The GPU implementation (slot pools, the miss path,
//! trace replay through the real expert kernels) is tested against it.

pub mod class;

pub use class::{CatalogMismatch, ExpertCatalog, KBits, KClass, Projection, ProjectionId};
