//! Paged KV page-budget query (ADR 0001, GitHub #55, P1-19).
//!
//! Safe wrapper over the kernel leaf's `ignis_paged_kv_page_budget`
//! (`kernel/include/ignis_paged_kv_budget.h`): host-only arithmetic, no CUDA
//! device required — used at model-load time to size the paged KV pool's
//! `page_group_count` from the VRAM left after weights, and the value the
//! scheduler's [`crate::KvPool`]-style accounting (`ignis_core::kv::KvPool`)
//! sizes from instead of a constant.

#![cfg(feature = "cuda")]

use crate::{fail, Result};

/// One storage plane's per-page geometry — 1:1 with `struct
/// ignis_paged_kv_plane`.
#[derive(Debug, Clone, Copy)]
pub struct PagedKvPlane {
    /// Mirrors `ninfer::DType`'s ordinal: 0 BF16, 1 FP32, 2 I32, 3 U8, 4
    /// I64, 5 I8, 6 FP16, 7 FP8_E4M3FN.
    pub dtype: i32,
    pub leading_extent: i32,
    pub head_extent: i32,
}

impl PagedKvPlane {
    /// A BF16 plane (the paged KV cache's K / V planes, G1: no quantized KV
    /// yet).
    pub fn bf16(leading_extent: i32, head_extent: i32) -> Self {
        Self {
            dtype: 0,
            leading_extent,
            head_extent,
        }
    }
}

/// The page budget the leaf reports for a given VRAM allowance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedKvBudget {
    /// The largest page count fitting the budget.
    pub page_count: u32,
    /// The exact bytes that many pages consume (`<= vram_budget_bytes`).
    pub page_bytes: u64,
}

/// Query the leaf for the page budget `planes` (K, V, ...) fit within
/// `vram_budget_bytes`. `planes` must be non-empty.
pub fn paged_kv_page_budget(planes: &[PagedKvPlane], vram_budget_bytes: u64) -> Result<PagedKvBudget> {
    if planes.is_empty() {
        return Err(fail("paged_kv_page_budget: planes must be non-empty"));
    }
    let raw: Vec<crate::ffi::IgnisPagedKvPlane> = planes
        .iter()
        .map(|p| crate::ffi::IgnisPagedKvPlane {
            dtype: p.dtype,
            leading_extent: p.leading_extent,
            head_extent: p.head_extent,
        })
        .collect();

    let mut page_count = 0u32;
    let mut page_bytes = 0u64;
    let rc = unsafe {
        crate::ffi::ignis_paged_kv_page_budget(
            raw.as_ptr(),
            raw.len() as i32,
            vram_budget_bytes,
            &mut page_count,
            &mut page_bytes,
        )
    };
    if rc != 0 {
        return Err(fail("ignis_paged_kv_page_budget failed (bad argument)"));
    }
    Ok(PagedKvBudget { page_count, page_bytes })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_matches_the_leaf_formula() {
        // One BF16 plane, head_dim=128, num_kv_heads=8: 128*64*8*2 bytes/page
        // (kPagedKVPageSize=64, fixed by the vendored header).
        let plane = PagedKvPlane::bf16(128, 8);
        let unit_bytes: u64 = 128 * 64 * 8 * 2;
        let budget = paged_kv_page_budget(&[plane], unit_bytes * 10).expect("budget");
        assert_eq!(budget.page_count, 10);
        assert_eq!(budget.page_bytes, unit_bytes * 10);
    }

    #[test]
    fn empty_planes_is_rejected() {
        assert!(paged_kv_page_budget(&[], 1024).is_err());
    }
}
