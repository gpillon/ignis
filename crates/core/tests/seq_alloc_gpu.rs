//! GPU integration test for the sequence handle step ABI (P1-19, GitHub
//! #55, ADR 0009): alloc / release / re-alloc cycles leave the device
//! pools balanced, exhaustion is a clean error, and `context_tokens == 0`
//! is rejected.
//!
//! (The stronger claim -- that a re-allocated sequence's KV pages and GDN
//! state actually read back as zero, not the previous occupant's bytes --
//! is proven at the kernel leaf's own level,
//! `kernel/tests/test_seq_alloc.cpp`, which can reach the vendored pools'
//! device memory directly; the flat ABI this test drives deliberately
//! never exposes a pointer to it, ADR 0009.)
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing GPU is a **skip**; under the profile it is a **hard failure**
//! (`ignis_core::gpu_profile::skip_or_fail`). Run via `scripts/gpu-profile.ps1`
//! (stops the reference `ninfer-serve` first — the RTX 5090 is exclusive,
//! ADR 0006).

#![cfg(feature = "cuda")]

use ignis_artifact::CudaDevice;
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::seq::{SeqPool, SeqPoolBudget};

/// A CUDA device must exist before the leaf's `cudaMalloc`-backed pools can
/// be built; `skip_or_fail` routes a missing/busy GPU the same way
/// `model_load_gpu.rs` does. Returns `None` when the caller should skip
/// (outside the profile).
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

#[test]
#[ignore]
fn alloc_release_realloc_cycles_balance_the_pools() {
    let Some(_device) = cuda_device_or_skip() else { return };

    let cfg = ModelConfig::synthetic();
    let budget = SeqPoolBudget {
        kv_page_group_count: 4,
        max_context_tokens: 128,
        slot_count: 2,
    };
    let pool =
        SeqPool::create(&cfg, &budget).unwrap_or_else(|e| panic!("ignis_seq_pool_create: {e}"));

    let stats = pool.stats();
    assert_eq!(stats.kv_page_group_count, 4);
    assert_eq!(stats.kv_free_pages, 4);
    assert_eq!(stats.slot_count, 2);
    assert_eq!(stats.free_slot_count, 2);

    let seq_a = pool.alloc(128).unwrap_or_else(|e| panic!("alloc A: {e}"));
    let a_stats = seq_a.stats();
    assert_eq!(a_stats.page_entitlement, 2, "128 tokens over 64-token pages");
    assert_eq!(a_stats.mapped_pages, 2);
    assert_eq!(a_stats.token_capacity, 128);

    let seq_b = pool.alloc(128).unwrap_or_else(|e| panic!("alloc B: {e}"));

    // Both slots and every page are now in use: the next alloc must fail
    // cleanly, and leave the pool exactly as it was.
    let Err(err) = pool.alloc(64) else {
        panic!("expected the pool to be exhausted");
    };
    assert!(!err.is_empty());
    let stats = pool.stats();
    assert_eq!(stats.kv_entitled_pages, 4, "a failed alloc does not change entitlement");
    assert_eq!(stats.free_slot_count, 0, "a failed alloc does not change the slot count");

    drop(seq_a);
    let stats = pool.stats();
    assert_eq!(stats.kv_entitled_pages, 2, "releasing A returns its 2 pages");
    assert_eq!(stats.kv_free_pages, 2);
    assert_eq!(stats.free_slot_count, 1, "releasing A returns its slot");

    let seq_c = pool.alloc(64).unwrap_or_else(|e| panic!("re-alloc C: {e}"));
    assert_eq!(seq_c.stats().mapped_pages, 1);

    drop(seq_c);
    drop(seq_b);
    let stats = pool.stats();
    assert_eq!(stats.kv_entitled_pages, 0, "every sequence released: pool is back to empty");
    assert_eq!(stats.kv_free_pages, 4);
    assert_eq!(stats.free_slot_count, 2);
}

#[test]
#[ignore]
fn zero_context_tokens_is_rejected() {
    let Some(_device) = cuda_device_or_skip() else { return };

    let cfg = ModelConfig::synthetic();
    let budget = SeqPoolBudget {
        kv_page_group_count: 4,
        max_context_tokens: 128,
        slot_count: 1,
    };
    let pool =
        SeqPool::create(&cfg, &budget).unwrap_or_else(|e| panic!("ignis_seq_pool_create: {e}"));

    assert!(pool.alloc(0).is_err(), "context_tokens == 0 is a clean error, not a 0-page sequence");
}

#[test]
#[ignore]
fn snapshot_and_restore_report_not_implemented() {
    let Some(_device) = cuda_device_or_skip() else { return };

    let cfg = ModelConfig::synthetic();
    let budget = SeqPoolBudget {
        kv_page_group_count: 4,
        max_context_tokens: 128,
        slot_count: 1,
    };
    let pool =
        SeqPool::create(&cfg, &budget).unwrap_or_else(|e| panic!("ignis_seq_pool_create: {e}"));
    let mut seq = pool.alloc(64).unwrap_or_else(|e| panic!("alloc: {e}"));

    let mut buf = [0u8; 16];
    assert_eq!(seq.snapshot(&mut buf), Err(ignis_core::seq::NOT_IMPLEMENTED));
    assert_eq!(seq.restore(&buf), Err(ignis_core::seq::NOT_IMPLEMENTED));
}
