//! The KV format's per-page byte arithmetic agrees with the leaf's own
//! (P4-04, GitHub #122).
//!
//! `ignis_core::kv_format` states what a format's storage *planes* are and
//! derives bytes from them with the vendored planner's formula
//! (`dtype_size * leading_extent * kPagedKVPageSize * head_extent`, summed).
//! That lets the whole byte-budget-to-token-capacity story be covered by fast
//! CPU unit tests — but it is a second copy of arithmetic the reference owns.
//! This test closes that gap: it hands the same plane descriptors to the
//! leaf's `ignis_paged_kv_page_budget`, which routes through the reference's
//! own `plan_paged_kv_pool` (ADR 0010), and requires the two to agree exactly.
//! A change to the reference's layout math therefore turns this red instead
//! of quietly drifting the capacity the engine reports.
//!
//! Host-only: `ignis_paged_kv_page_budget` does no CUDA call, so no GPU is
//! needed. It is `#[ignore]`d anyway because the `cuda` feature it needs is
//! only built for the GPU profile's run (`scripts/gpu-profile.ps1`).

#![cfg(feature = "cuda")]

use ignis_artifact::{PagedKvPlane, paged_kv_page_budget};
use ignis_core::kv_format::{KvFormat, KvGeometry, KvPlaneSpec, plan_kv_pool};

fn leaf_planes(specs: &[KvPlaneSpec]) -> Vec<PagedKvPlane> {
    specs
        .iter()
        .map(|p| PagedKvPlane {
            dtype: p.dtype.abi_code(),
            leading_extent: p.leading_extent as i32,
            head_extent: p.head_extent as i32,
        })
        .collect()
}

#[test]
#[ignore]
fn every_formats_page_geometry_matches_the_leafs_own_budget_query() {
    let geometry = KvGeometry::qwen38_27b();
    // A budget that is an exact multiple of neither format's page size, so a
    // rounding disagreement shows up as a page-count difference rather than
    // cancelling out.
    let budget: u64 = 3_000_000_000;

    for format in [KvFormat::Bf16, KvFormat::HqE8_2b] {
        let planes = leaf_planes(&format.pool_planes(geometry));
        let leaf = paged_kv_page_budget(&planes, budget)
            .unwrap_or_else(|e| panic!("{format}: ignis_paged_kv_page_budget: {e}"));
        let plan = plan_kv_pool(format, geometry, budget);

        assert_eq!(
            plan.page_count, leaf.page_count,
            "{format}: page count disagrees with the leaf's budget query"
        );
        assert_eq!(
            plan.pool_bytes, leaf.page_bytes,
            "{format}: pool bytes disagree with the leaf's budget query"
        );
        assert_eq!(
            plan.page_bytes * u64::from(plan.page_count),
            leaf.page_bytes,
            "{format}: per-page bytes disagree with the leaf's budget query"
        );
    }
}

#[test]
#[ignore]
fn one_gqa_layers_planes_cost_what_the_leaf_says_they_do() {
    // The same agreement one layer at a time, so a mismatch names the layer's
    // planes rather than only the pool total.
    let geometry = KvGeometry::qwen38_27b();
    for format in [KvFormat::Bf16, KvFormat::HqE8_2b] {
        let specs = format.planes_per_gqa_layer(geometry);
        let planes = leaf_planes(&specs);
        // One page's worth of budget: the query then reports exactly one
        // page and its byte cost.
        let ours: u64 = specs.iter().map(KvPlaneSpec::page_bytes).sum();
        let leaf = paged_kv_page_budget(&planes, ours)
            .unwrap_or_else(|e| panic!("{format}: ignis_paged_kv_page_budget: {e}"));
        assert_eq!(leaf.page_count, 1, "{format}: one page's budget buys one page");
        assert_eq!(leaf.page_bytes, ours, "{format}: one layer's per-page bytes");
    }
}
