//! A model's mutable state image, derived from its topology
//! (`ModelConfig::state_image`, spec flash-next/05), agrees with the slot the
//! leaf's sequence pool lays out for it (`ignis_seq_pool_plan`'s
//! `retained_host_bytes` at one host retained slot: exactly one image).
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

use ignis_core::compute::{ModelConfig, STATE_SECTION_ALIGN};
use ignis_core::kv_format::KvFormat;
use ignis_core::seq::{SeqPool, SeqPoolBudget};

/// One image, as the leaf lays a host retained slot out.
fn leaf_slot_bytes(cfg: &ModelConfig, kv_format: KvFormat) -> u64 {
    let budget = SeqPoolBudget {
        kv_format,
        kv_page_group_count: 8,
        max_context_tokens: 512,
        slot_count: 1,
        retained_slot_count: 0,
        retained_host_slot_count: 1,
    };
    SeqPool::plan(cfg, &budget, None)
        .unwrap_or_else(|e| panic!("{:?} {kv_format}: {e}", cfg.family))
        .retained_host_bytes
}

fn aligned(bytes: u64) -> u64 {
    bytes.div_ceil(STATE_SECTION_ALIGN) * STATE_SECTION_ALIGN
}

#[test]
#[ignore]
fn the_27b_image_is_the_slot_the_leaf_lays_out() {
    let cfg = ModelConfig::qwen38_27b();
    for format in [KvFormat::Bf16, KvFormat::HqE8_2b] {
        assert_eq!(
            cfg.state_image(format).slot_bytes(),
            leaf_slot_bytes(&cfg, format),
            "27B {format}"
        );
    }
}

/// The leaf's slot has every section but the two only Flash-Next has, the
/// n-gram conv state and the indexer tail: the forward that writes them
/// (spec flash-next/04, S1) adds them to the leaf's section table. When it
/// does, this goes red until the subtraction below is removed.
#[test]
#[ignore]
fn flash_next_image_is_the_slot_the_leaf_lays_out_less_the_sections_its_forward_adds() {
    let cfg = ModelConfig::qwen38_flash_next();
    for format in [KvFormat::Bf16, KvFormat::HqE8_2b] {
        let image = cfg.state_image(format);
        let not_yet_in_the_leaf = aligned(image.ngram_conv_state) + aligned(image.indexer_tail);
        assert_eq!(
            image.slot_bytes() - not_yet_in_the_leaf,
            leaf_slot_bytes(&cfg, format),
            "Flash-Next {format}"
        );
    }
}
