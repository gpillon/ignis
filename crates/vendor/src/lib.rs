//! `ignis-vendor` — the kernel leaf's vendoring tool (ADR 0010).
//!
//! The kernel leaf hosts reference ops *verbatim*: kernels, launchers,
//! wrappers, headers and the reference's own op tests, copied unchanged from a
//! pinned commit of the reference fork. This crate is the machinery that makes
//! that claim checkable — a manifest pinning the commit and every file's
//! content hash, and the copy / verify / patch-record operations over it.
//!
//! The binary is `vendor-ninfer`; `scripts/vendor-ninfer.ps1` is the wrapper
//! the runbook and `kernel/vendor/VENDOR.md` refer to.

pub mod manifest;
pub mod sha256;
pub mod vendor;

pub use manifest::{Manifest, ManifestError, Patch, Reference, VendoredFile};
pub use sha256::sha256_hex;
pub use vendor::{
    Finding, Problem, Report, SyncError, SyncReport, refresh_reference_hashes, sync, unified_diff,
    verify_reference, verify_vendor_tree,
};
