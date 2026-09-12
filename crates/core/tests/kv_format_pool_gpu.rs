//! GPU integration test for the KV format as a model-load option (P4-04,
//! GitHub #122, ADR 0022): the pool is sized from a byte budget, and the
//! token capacity it reports is derived from the format in force.
//!
//! What this test is for is the word **reported**. The arithmetic that turns
//! a byte budget into a page count lives in `ignis_core::kv_format` and is
//! covered by fast CPU unit tests there; what those cannot show is that the
//! leaf actually builds the pool they describe and reports back the capacity
//! it holds. Every assertion below therefore reads
//! `ignis_seq_pool_stats` — the leaf's own account of what it built — rather
//! than the plan that asked for it. The standard target profile's
//! 8 x 40,960 = 327,680 resident tokens is checked the same way, which is
//! what the acceptance means by "verified from the reported capacity, not
//! from a constant in the allocator".
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing GPU is a **skip**; under the profile it is a **hard failure**.
//! Run via `scripts/gpu-profile.ps1` (stops the reference `ninfer-serve`
//! first — the RTX 5090 is exclusive, ADR 0006).

#![cfg(feature = "cuda")]

use ignis_artifact::CudaDevice;
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::{KvFormat, KvGeometry, N_DECODE_LANES, auto_kv_pool_bytes, plan_kv_pool};

/// The per-sequence context the phase-4 target profile is stated in: a 32K
/// prompt plus an 8K generation budget, the engine's own default.
const TARGET_CONTEXT: u32 = 40_960;

fn cuda_device_or_skip() -> Option<CudaDevice> {
    match CudaDevice::create(0) {
        Ok(d) => Some(d),
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA device unavailable: {e}")) {
                None
            } else {
                unreachable!("skip_or_fail panics under the profile");
            }
        }
    }
}

/// Build a pool of `format` from `budget_bytes`, at the real 27B geometry.
fn pool_from_budget(format: KvFormat, budget_bytes: u64, slot_count: u32) -> SeqPool {
    let plan = plan_kv_pool(format, KvGeometry::qwen38_27b(), budget_bytes);
    let cfg = ModelConfig::qwen38_27b();
    SeqPool::create(
        &cfg,
        &SeqPoolBudget {
            kv_format: format,
            kv_page_group_count: plan.page_count,
            max_context_tokens: TARGET_CONTEXT,
            slot_count,
        },
    )
    .unwrap_or_else(|e| panic!("ignis_seq_pool_create ({format}): {e}"))
}

#[test]
#[ignore]
fn the_leaf_reports_the_token_capacity_the_budget_bought_in_each_format() {
    let Some(_device) = cuda_device_or_skip() else {
        return;
    };

    // One modest budget, deliberately the same for both formats: nothing but
    // the format differs between these two pools.
    let budget = 1024 * 1024 * 1024u64;
    let geometry = KvGeometry::qwen38_27b();

    for format in [KvFormat::Bf16, KvFormat::HqE8_2b] {
        let plan = plan_kv_pool(format, geometry, budget);
        let pool = pool_from_budget(format, budget, 2);
        let stats = pool.stats();

        assert_eq!(
            stats.kv_format,
            format.abi_code(),
            "{format}: the leaf reports the format it was built with"
        );
        assert_eq!(
            stats.kv_page_group_count, plan.page_count,
            "{format}: the leaf built the pages the budget bought"
        );
        assert_eq!(
            stats.kv_page_bytes, plan.page_bytes,
            "{format}: the leaf's own page bytes match the planned planes"
        );
        assert_eq!(
            stats.kv_bytes_per_token,
            plan.bytes_per_token,
            "{format}: the leaf's per-token cost matches the planned planes"
        );
        assert_eq!(
            stats.kv_token_capacity, plan.token_capacity,
            "{format}: the reported capacity is the derived one"
        );
        assert!(
            u64::from(stats.kv_page_group_count) * stats.kv_page_bytes <= budget,
            "{format}: the pool fits inside its budget"
        );
    }
}

#[test]
#[ignore]
fn the_same_budget_reports_seven_times_the_tokens_under_hq() {
    let Some(_device) = cuda_device_or_skip() else {
        return;
    };

    let budget = 1024 * 1024 * 1024u64;
    let bf16 = pool_from_budget(KvFormat::Bf16, budget, 2).stats();
    let hq = pool_from_budget(KvFormat::HqE8_2b, budget, 2).stats();

    // The capacity finding's 7.11x, read off two pools the leaf actually
    // built rather than off the arithmetic that planned them.
    assert_eq!(bf16.kv_bytes_per_token, 65_536);
    assert_eq!(hq.kv_bytes_per_token, 9_216);
    assert!(
        hq.kv_token_capacity > 7 * bf16.kv_token_capacity,
        "hq bought {} tokens against BF16's {}",
        hq.kv_token_capacity,
        bf16.kv_token_capacity
    );
}

#[test]
#[ignore]
fn the_standard_target_profile_holds_eight_full_contexts_under_hq() {
    let Some(_device) = cuda_device_or_skip() else {
        return;
    };

    // The profile as the engine would actually run it: the auto default
    // budget, the default per-sequence context, all eight decode lanes.
    let slots = N_DECODE_LANES as u32;
    let required = u64::from(slots) * u64::from(TARGET_CONTEXT);
    let budget = auto_kv_pool_bytes(KvFormat::HqE8_2b, KvGeometry::qwen38_27b(), TARGET_CONTEXT);

    let pool = pool_from_budget(KvFormat::HqE8_2b, budget, slots);
    let stats = pool.stats();
    assert_eq!(stats.slot_count, slots);
    assert!(
        stats.kv_token_capacity >= required,
        "the hq pool reports {} resident tokens, short of {slots} x {TARGET_CONTEXT} = {required}",
        stats.kv_token_capacity
    );

    // And the capacity is spendable, not just reported: eight full-context
    // sequences are allocated at once out of the same pool.
    let mut sequences = Vec::new();
    for lane in 0..slots {
        sequences.push(
            pool.alloc(TARGET_CONTEXT)
                .unwrap_or_else(|e| panic!("lane {lane} of {slots} at {TARGET_CONTEXT} tokens: {e}")),
        );
    }
    for seq in &sequences {
        assert_eq!(seq.stats().token_capacity, u64::from(TARGET_CONTEXT));
    }
    assert_eq!(pool.stats().free_slot_count, 0);
    drop(sequences);
    assert_eq!(pool.stats().free_slot_count, slots);
}

#[test]
#[ignore]
fn the_same_profile_does_not_fit_under_bf16() {
    let Some(_device) = cuda_device_or_skip() else {
        return;
    };

    // The inequality every gate so far recorded, stated as a test: at the
    // auto default budget, BF16 cannot hold the target profile. This is what
    // makes the hq capacity above a result rather than a tautology.
    let budget = auto_kv_pool_bytes(KvFormat::Bf16, KvGeometry::qwen38_27b(), TARGET_CONTEXT);
    let stats = pool_from_budget(KvFormat::Bf16, budget, N_DECODE_LANES as u32).stats();
    assert!(
        stats.kv_token_capacity < u64::from(N_DECODE_LANES as u32) * u64::from(TARGET_CONTEXT),
        "BF16 unexpectedly reported {} resident tokens at the default budget",
        stats.kv_token_capacity
    );
}

#[test]
#[ignore]
fn an_unknown_kv_format_never_reaches_a_pool() {
    let Some(_device) = cuda_device_or_skip() else {
        return;
    };

    // `KvFormat` makes an unknown format unrepresentable from Rust, so this
    // covers the leaf's own defence: `ignis_seq_pool_create` validates the
    // code it is handed rather than treating anything unrecognized as BF16.
    // (The leaf-side check itself is `kernel/tests/test_seq_alloc.cpp`; this
    // asserts only that the two enums still agree, which is what would break
    // silently if either side ever renumbered.)
    assert_eq!(KvFormat::Bf16.abi_code(), 0);
    assert_eq!(KvFormat::HqE8_2b.abi_code(), 1);

    let stats = pool_from_budget(KvFormat::HqE8_2b, 1024 * 1024 * 1024, 1).stats();
    assert_eq!(stats.kv_format, KvFormat::HqE8_2b.abi_code());
}
