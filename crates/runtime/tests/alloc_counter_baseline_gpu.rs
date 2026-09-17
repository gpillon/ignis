//! Where serving allocates (GitHub #211, #215, ADR 0030): the leaf's
//! allocation counter read around a fixed request mix with prompt reuse on,
//! on a DFlash2 load.
//!
//! The mix is GitHub #190's three KV-RAM legs (`kv_ram_gpu_common`): a
//! conversation checkpointed, pushed to KV-RAM and resumed; a retained-prefix
//! claimant snapshotted mid-decode; a burst's system block spilled and brought
//! back. Between them they publish prefixes, capture checkpoints and write
//! KV-RAM blobs, and each still asserts its own tokens.
//!
//! #211 recorded 9 prefix images, 1 checkpoint image with its tail page and 4
//! KV-RAM blobs over this mix. Since #215 a publish and a capture put their
//! images in retained slots reserved at load and a checkpoint's tail page in a
//! KV page of the pool, so those counts are zero; the KV-RAM blobs are still
//! `cudaHostAlloc`ed one by one until #213's arena. Nothing is allocated from
//! `ignis_device_alloc` while serving.
//!
//! Its own test binary: it materializes the artifact, and the card fits one
//! at a time.

#![cfg(feature = "cuda")]

mod kv_ram_gpu_common;

use ignis_core::seq::{AllocCount, AllocKind, alloc_count};
use ignis_core::{Speculation, SpeculativeBackend};

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn serving_allocates_no_retained_state_image_and_a_blob_per_kv_ram_spill() {
    let speculation = Speculation::new(SpeculativeBackend::Dflash2, 7).unwrap();
    let Some(loaded) = kv_ram_gpu_common::load(Some(speculation)) else {
        return;
    };
    let before = AllocKind::ALL.map(alloc_count);

    kv_ram_gpu_common::an_idle_conversation_resumes_from_kv_ram_exactly(&loaded);
    kv_ram_gpu_common::a_retained_prefix_claimant_evicted_mid_decode_continues_exactly(&loaded);
    kv_ram_gpu_common::a_burst_block_brought_back_from_kv_ram_serves_exactly(&loaded);

    let counts: Vec<(AllocKind, AllocCount)> = AllocKind::ALL
        .iter()
        .zip(&before)
        .map(|(&kind, earlier)| (kind, alloc_count(kind).since(earlier)))
        .collect();
    eprintln!("allocation counts over the mix: {counts:#?}");
    let of = |kind: AllocKind| counts.iter().find(|(k, _)| *k == kind).unwrap().1;

    // Every scheduler of the mix has been dropped: whatever it retained went
    // with it, so every allocation was freed.
    for (kind, count) in &counts {
        assert_eq!(count.allocs, count.frees, "{kind:?} leaked: {count:?}");
    }
    assert_eq!(of(AllocKind::Device), AllocCount::default(), "no device arena while serving");

    // GitHub #215: the same mix publishes and captures, and allocates for
    // neither.
    for kind in [AllocKind::PrefixImage, AllocKind::CheckpointImage, AllocKind::CheckpointTailPage] {
        assert_eq!(of(kind), AllocCount::default(), "{kind:?} allocated while serving");
    }
    // What is left is #213's: the mix still spills to KV-RAM, blob by blob --
    // which is also what shows it is the mix #211 recorded.
    assert_eq!(of(AllocKind::KvRamBlob).allocs, EXPECTED_KV_RAM_BLOBS, "{:?}", of(AllocKind::KvRamBlob));
}

/// Recorded on the RTX 5090, 2026-09-17 (#211): 4 KV-RAM blobs (981,839,872 B
/// together). #211's 9 prefix images and 1 checkpoint image with its tail page
/// are 0 since #215.
const EXPECTED_KV_RAM_BLOBS: u64 = 4;
