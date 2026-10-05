//! A model's mutable state image, derived from its topology
//! (`ModelConfig::state_image`, spec flash-next/05), agrees with what the
//! leaf's sequence pool lays out for one slot (`ignis_seq_pool_plan`).
//!
//! The Rust derivation is what a load prints and what a host plan charges;
//! the leaf's planner is what a slot really holds. A section either one
//! sized with the other model's number -- the 27B's 16 attention layers in
//! Flash-Next's residual window, say -- turns this red.
//!
//! Host-only: planning a pool makes no CUDA call. `#[ignore]`d because the
//! `cuda` feature it needs is only built for the GPU profile's run, like
//! `kv_format_leaf_agreement.rs`.

#![cfg(feature = "cuda")]

use ignis_core::compute::ModelConfig;
use ignis_core::kv_format::KvFormat;
use ignis_core::seq::{SeqPool, SeqPoolBudget};

const PAGES: u32 = 8;
const SLOTS: u32 = 3;

fn budget(kv_format: KvFormat, retained_host_slot_count: u32) -> SeqPoolBudget {
    SeqPoolBudget {
        kv_format,
        kv_page_group_count: PAGES,
        max_context_tokens: 512,
        slot_count: SLOTS,
        retained_slot_count: 0,
        retained_host_slot_count,
    }
}

/// The 27B's image is exactly one host retained slot as the leaf lays it
/// out: every section from a 256-byte boundary.
#[test]
#[ignore]
fn the_27b_image_is_the_slot_the_leaf_lays_out() {
    let cfg = ModelConfig::qwen38_27b();
    for format in [KvFormat::Bf16, KvFormat::HqE8_2b] {
        let plan = SeqPool::plan(&cfg, &budget(format, 1), None).unwrap_or_else(|e| panic!("27B {format}: {e}"));
        assert_eq!(cfg.state_image(format).slot_bytes(), plan.retained_host_bytes, "27B {format}");
    }
}

/// Flash-Next's image read off the pool's own lines, one slot's share of
/// each: the GDN and penalty state, the residual window, the indexer's tails
/// (its block keys are per page, not per slot) and the n-gram conv state.
#[test]
#[ignore]
fn flash_next_image_is_one_slots_share_of_the_pools_state() {
    let cfg = ModelConfig::qwen38_flash_next();
    let indexer = cfg.indexer.expect("Flash-Next has an indexer");
    let block_keys = cfg.attention_layer_count() as u64
        * u64::from(PAGES)
        * (64 / indexer.compress_ratio)
        * indexer.kv_heads
        * indexer.head_dim
        * 2;
    for format in [KvFormat::Bf16, KvFormat::HqE8_2b] {
        let plan = SeqPool::plan(&cfg, &budget(format, 0), None).unwrap_or_else(|e| panic!("Flash-Next {format}: {e}"));
        let slots = u64::from(SLOTS);
        let per_slot = plan.slot_state_bytes
            + plan.hq_residual_bytes / slots
            + (plan.indexer_bytes - block_keys) / slots
            + plan.ngram_conv_bytes / slots;
        assert_eq!(cfg.state_image(format).total_bytes(), per_slot, "Flash-Next {format}");
    }
}

/// And a host retained slot holds exactly that image, every section from a
/// 256-byte boundary: its indexer tails and n-gram conv state are cloned
/// with the rest (spec flash-next/05), so a claim restores all of it.
#[test]
#[ignore]
fn the_flash_next_image_is_the_slot_the_leaf_lays_out() {
    let cfg = ModelConfig::qwen38_flash_next();
    for format in [KvFormat::Bf16, KvFormat::HqE8_2b] {
        let plan = SeqPool::plan(&cfg, &budget(format, 1), None).unwrap_or_else(|e| panic!("Flash-Next {format}: {e}"));
        assert_eq!(cfg.state_image(format).slot_bytes(), plan.retained_host_bytes, "Flash-Next {format}");
    }
}

/// Spec flash-next/05's default reuse load: 8 host retained slots in hq-e8-2b,
/// 130,014,464 bytes each (the spec's figure), about 1 GiB on the host plan's
/// retained-slots line -- and the pool keeps a page per slot beside every
/// lane's whole context.
#[test]
#[ignore]
fn the_default_flash_next_reuse_load_pins_eight_images_on_the_host() {
    use ignis_core::flash_next::EngineOptions;
    let cfg = ModelConfig::qwen38_flash_next();
    let options = EngineOptions { retained_host_slots: 8, ..EngineOptions::default() };
    let plan = SeqPool::plan(&cfg, &options.pool_budget(), None).unwrap_or_else(|e| panic!("Flash-Next: {e}"));
    assert_eq!(plan.retained_host_bytes, 8 * 130_014_464);
    assert_eq!(plan.retained_state_bytes, 0, "no device slot by default");
    let lanes = options.decode_lanes;
    assert_eq!(options.pool_budget().kv_page_group_count, lanes * options.max_context_tokens.div_ceil(64) + 8);
}
