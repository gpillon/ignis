//! Where serving allocates today (GitHub #211, ADR 0030): the leaf's
//! allocation counter read around a fixed request mix with prompt reuse on,
//! on a DFlash2 load.
//!
//! The mix is GitHub #190's three KV-RAM legs (`kv_ram_gpu_common`): a
//! conversation checkpointed, pushed to KV-RAM and resumed; a retained-prefix
//! claimant snapshotted mid-decode; a burst's system block spilled and brought
//! back. Between them they publish prefixes, capture checkpoints and write
//! KV-RAM blobs, and each still asserts its own tokens.
//!
//! The counts asserted are today's, and they are not zero: every prefix
//! publish and checkpoint capture `cudaMalloc`s an image, and every KV-RAM
//! blob `cudaHostAlloc`s its region. #215 (retained slots), #212 and #213
//! (the KV-RAM arena) drive the matching counts to zero and change this test
//! when they do. Nothing is allocated from `ignis_device_alloc` while serving.
//!
//! Its own test binary: it materializes the artifact, and the card fits one
//! at a time.

#![cfg(feature = "cuda")]

mod kv_ram_gpu_common;

use ignis_core::seq::{AllocCount, AllocKind, alloc_count};
use ignis_core::{Speculation, SpeculativeBackend};

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn serving_allocates_per_prefix_publish_checkpoint_capture_and_kv_ram_blob_today() {
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

    let expected = [
        (AllocKind::PrefixImage, EXPECTED_PREFIX_IMAGES),
        (AllocKind::CheckpointImage, EXPECTED_CHECKPOINT_IMAGES),
        (AllocKind::CheckpointTailPage, EXPECTED_CHECKPOINT_IMAGES),
        (AllocKind::KvRamBlob, EXPECTED_KV_RAM_BLOBS),
    ];
    for (kind, allocs) in expected {
        assert!(allocs > 0, "the baseline of {kind:?} is today's non-zero count");
        assert_eq!(of(kind).allocs, allocs, "{kind:?}: {:?}", of(kind));
    }
}

/// Recorded on the RTX 5090, 2026-09-17: 9 prefix images (238,823,424 B
/// each on this DFlash2 load), 1 checkpoint image with its 589,824 B tail
/// page, 4 KV-RAM blobs (981,839,872 B together).
const EXPECTED_PREFIX_IMAGES: u64 = 9;
const EXPECTED_CHECKPOINT_IMAGES: u64 = 1;
const EXPECTED_KV_RAM_BLOBS: u64 = 4;
