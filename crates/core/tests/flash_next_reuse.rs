//! Spec flash-next/05 (GitHub #303): prompt reuse on Flash-Next, on a CPU.
//!
//! ADR 0029's policy does not change for Flash-Next; its state does. So the
//! reuse-policy suites -- match, lineage, first victim, KV-RAM order, budget
//! exhaustion -- are mounted here unchanged, with one difference from their
//! own runs: every KV-RAM blob the mock writes is Flash-Next's size (its
//! state image as a slot holds it, plus 4,224 paged bytes per token in
//! hq-e8-2b) instead of one nominal byte, and every KV-RAM capacity the
//! suites state in blobs is that many of those. A policy that leaned on the
//! 27B's sizes, or on every blob being the same size, fails here.
//!
//! Below them, blob identity across the two models.

use std::sync::Arc;

use ignis_core::compute::ModelConfig;
use ignis_core::types::{DecodeParams, RequestClass, RequestInput};
use ignis_core::{
    ArtifactHash, BlobIdentity, ConcreteScheduler, IdentityField, KvFormat, MockCompute,
    MockSections, Scheduler, SchedulerConfig, Speculation, SpeculativeBackend,
};

/// Flash-Next's blob sizes under hq-e8-2b, the serving format: what every
/// mounted suite's mock charges.
fn sections() -> MockSections {
    MockSections::of(&ModelConfig::qwen38_flash_next(), KvFormat::HqE8_2b)
}

#[path = "prompt_checkpoint.rs"]
mod prompt_checkpoint;

#[path = "kv_ram_spill.rs"]
mod kv_ram_spill;

#[path = "prefix_reuse.rs"]
mod prefix_reuse;

#[path = "retained_prefix.rs"]
mod retained_prefix;

#[path = "retained_slots.rs"]
mod retained_slots;

#[path = "reuse_boundaries.rs"]
mod reuse_boundaries;

/// The mounted suites really run at Flash-Next's sizes: a blob is ~124 MiB
/// of image plus its tokens, not a byte.
#[test]
fn the_mounted_suites_charge_flash_next_blobs() {
    let sections = sections();
    assert_eq!(sections.image_bytes, 130_014_464);
    assert_eq!(sections.bytes_per_token, 4_224);
    assert_eq!(sections.blob_bytes(30_000), 130_014_464 + 126_720_000);
}

// ── Blob identity (spec flash-next/05 acceptance 5) ─────────────────────

/// Two artifacts' content hashes. Content hashes of two models' containers
/// differ by construction (`Reader::content_hash` covers every object's
/// name, format and offset), so any two distinct values stand in for them.
const QWEN38_27B: [u8; 32] = [27; 32];
const FLASH_NEXT: [u8; 32] = [38; 32];

const MODEL: &str = "qwen3.8-flash-next";

/// The blob layout version: one leaf, one number today. It is the same on
/// both sides here on purpose, so that only the artifact can refuse.
const LAYOUT: u32 = 5;

fn flash_next_load() -> BlobIdentity {
    // No drafter: Flash-Next has no speculative decoding (spec 04).
    BlobIdentity::of_load(ArtifactHash::from_bytes(FLASH_NEXT), KvFormat::HqE8_2b, None, LAYOUT)
}

/// The 27B as close to Flash-Next as a load of it can be: same KV format,
/// same layout, no drafter.
fn plain_27b_load() -> BlobIdentity {
    BlobIdentity::of_load(ArtifactHash::from_bytes(QWEN38_27B), KvFormat::HqE8_2b, None, LAYOUT)
}

/// The 27B as it serves by default, with DFlash2.
fn speculative_27b_load() -> BlobIdentity {
    BlobIdentity::of_load(
        ArtifactHash::from_bytes(QWEN38_27B),
        KvFormat::HqE8_2b,
        Some(Speculation::new(SpeculativeBackend::Dflash2, 4).unwrap()),
        LAYOUT,
    )
}

/// A conversation's turn whose opener leaves a prompt checkpoint.
fn turn() -> RequestInput {
    RequestInput {
        decision: None,
        model: MODEL.into(),
        tokens: (1..41).collect(),
        params: DecodeParams {
            max_tokens: Some(4),
            ..DecodeParams::default()
        },
        multimodal: None,
        opener_tokens: Some(37),
        user_turn_tokens: None,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        constrained: None,
        warm_up: false,
    }
}

/// A scheduler on a load of `identity`, after one turn: its pool's identity
/// and the header of the checkpoint the turn left.
fn after_one_turn(identity: BlobIdentity) -> ConcreteScheduler {
    let compute = Arc::new(MockCompute::with_blob_identity(identity));
    let config = SchedulerConfig {
        model: MODEL.into(),
        ..SchedulerConfig::default()
    };
    let mut sched = ConcreteScheduler::with_config(config, compute);
    sched.submit(turn(), RequestClass::Interactive).unwrap();
    while !sched.is_idle() {
        sched.advance();
    }
    sched
}

#[test]
fn a_27b_blob_is_refused_under_flash_next_by_its_artifact() {
    let flash_next = after_one_turn(flash_next_load());
    for dense in [plain_27b_load(), speculative_27b_load()] {
        let header = after_one_turn(dense).checkpoint_pool().entries()[0].header();
        assert_eq!(header.identity, dense, "the entry names the load that made it");
        let refusal = flash_next
            .checkpoint_pool()
            .accepts(&header)
            .expect_err("a 27B blob under Flash-Next");
        assert_eq!(refusal.field, IdentityField::Artifact, "{refusal}");
        assert!(refusal.to_string().contains("artifact"), "{refusal}");
    }
}

#[test]
fn a_flash_next_blob_is_refused_under_the_27b_by_its_artifact() {
    let header = after_one_turn(flash_next_load()).checkpoint_pool().entries()[0].header();
    for dense in [plain_27b_load(), speculative_27b_load()] {
        let refusal = after_one_turn(dense)
            .checkpoint_pool()
            .accepts(&header)
            .expect_err("a Flash-Next blob under the 27B");
        assert_eq!(refusal.field, IdentityField::Artifact, "{refusal}");
    }
    // And Flash-Next takes back what it produced.
    assert_eq!(after_one_turn(flash_next_load()).checkpoint_pool().accepts(&header), Ok(()));
}
