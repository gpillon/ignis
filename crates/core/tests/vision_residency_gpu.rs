//! GPU tests for the vision residency (GitHub #414, ADR 0075): where a vision
//! load keeps the 27B's vision tower, as the leaf itself accounts for it.
//!
//! - Under `host` the binder places the tower's 333 tensors in the mapped
//!   pinned allocation beside the token embedding, and the leaf reports their
//!   295,711,648 bytes as host-mapped bytes, not VRAM bytes; under `device` it
//!   is today's resident load. The embedding pool stays on the device under
//!   both (#415 moves it). The plan the server logs and the load it gets agree
//!   to the byte, the staging ring included, and a load whose residency
//!   disagrees with where the binder put the tower is refused.
//! - On a 16 GB budget with the `rtx50-16gb` profile (the trellis artifact,
//!   hq-e8-2b, 262,144 tokens, 1,024-token chunks, DFlash2 at 7, two lanes),
//!   the ring fits inside the prefill scratch the text load already reserves
//!   at a 1,024- and a 4,096-token envelope, so `host` grows nothing there;
//!   where the encoder's share sets the scratch's size (a 16,384-token item),
//!   it grows by the ring.
//!
//! The identity of the two residencies' encodes and logits is
//! `crates/server/tests/vision_residency_gpu.rs`, which has the vision canary.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a skip; under the profile, a hard failure.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::vision::{Grid, PATCH_FEATURES};
use ignis_artifact::{bind_model_scope_27b_with, materialize, CudaDevice, Reader};
use ignis_core::gpu_profile;
use ignis_core::model_load::{
    load_qwen38_27b_with_options, model_scope, plan_qwen38_27b_reservations, IgnisModelReservations,
};
use ignis_core::step::encode_media;
use ignis_core::vision::vision_item_control;
use ignis_core::{
    ExceptionWeights, KvFormat, PrefillAttention, RopeScaling, Speculation, SpeculativeBackend, Vision,
    VisionResidency, N_DECODE_LANES,
};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const TRELLIS: &str = "F:/ai/models/Qwen3.8-27B-trellis-3bpw-ignis/qwen3_8_27b_trellis_sc30-v2.ninfer";

/// The tower's 333 payloads (`real_artifact.rs` pins the same figure on the
/// binder's plan).
const TOWER_BYTES: u64 = 295_711_648;
/// Three slots of the largest stage, the merger's fc2 and its bias with the
/// padding their host spans may add: 25,094,656 bytes each (#409's
/// prototype, and ADR 0075's 75.3 MB).
const RING_BYTES: u64 = 3 * 25_094_656;
/// The processor's item bound for this artifact (a 4096 x 4096 image), which
/// the server hands every vision load.
const ITEM_BOUND: u32 = 16_384;

/// What a load held, and what its plan said it would.
struct Held {
    vram_bytes: u64,
    host_mapped_bytes: u64,
    vision_reserved_bytes: u64,
    reserved: IgnisModelReservations,
    planned: IgnisModelReservations,
}

/// Bind, plan, materialize and load the served artifact with vision at
/// `residency`; encode one small item from the tower; release everything.
/// Along the way a load whose residency is the other one is refused.
fn held_at(reader: &Reader, device: &mut CudaDevice, residency: VisionResidency) -> Held {
    // A 128-token chunk: the encoder's share of the scratch is the larger
    // (`vision_load_gpu.rs` pins it), so the ring shows in the workspace.
    const CONTEXT: u32 = 1024;
    const CHUNK: u32 = 128;
    let vision = Vision::new(1024).unwrap().with_item_max_tokens(ITEM_BOUND).with_residency(residency);
    let (plan, handles) = bind_model_scope_27b_with(reader, model_scope(None, Some(vision), ExceptionWeights::Bf16))
        .unwrap_or_else(|e| panic!("bind ({residency}): {e}"));
    let planned = plan_qwen38_27b_reservations(
        reader,
        &plan,
        &handles,
        CHUNK,
        CONTEXT,
        KvFormat::Bf16,
        None,
        Some(vision),
        RopeScaling::NONE,
        PrefillAttention::default(),
        N_DECODE_LANES as u32,
    )
    .unwrap_or_else(|e| panic!("plan ({residency}): {e}"));
    let mut artifact = materialize(reader, &plan, device, None).unwrap_or_else(|e| panic!("materialize ({residency}): {e}"));
    let load = |vision: Vision| {
        load_qwen38_27b_with_options(reader, &artifact, &handles, CHUNK, CONTEXT, KvFormat::Bf16, None, Some(vision), RopeScaling::NONE)
    };

    // The plan sized the scratch by the option, which the binder's placement
    // must match: never a tower read in place from host memory, never a ring
    // for a tower that is not there.
    let other = match residency {
        VisionResidency::Host => VisionResidency::Device,
        VisionResidency::Device => VisionResidency::Host,
    };
    let Err(refused) = load(vision.with_residency(other)) else {
        panic!("a {other} load of a tower bound for {residency} was accepted");
    };
    assert!(refused.contains("vision residency is") && refused.contains("bind the tower where the residency puts it"), "{refused}");

    let model = load(vision).unwrap_or_else(|e| panic!("ignis_model_load ({residency}): {e}"));
    // The tower works from where it lives: an item encodes.
    let grid = Grid { t: 1, h: 16, w: 16 };
    let patches = vec![0u16; grid.raw_patches() as usize * PATCH_FEATURES];
    let embedding = encode_media(&model, grid, &patches, &vision_item_control(grid))
        .unwrap_or_else(|(rc, e)| panic!("encode ({residency}): {rc} {e}"));
    assert_eq!(embedding.columns(), 64);
    drop(embedding);
    let stats = model.stats();
    drop(model);
    let _ = artifact.release_arena(device);
    Held {
        vram_bytes: stats.vram_bytes,
        host_mapped_bytes: stats.host_mapped_bytes,
        vision_reserved_bytes: stats.vision_reserved_bytes,
        reserved: stats.reserved,
        planned,
    }
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn under_host_the_leaf_holds_the_tower_in_host_mapped_bytes_and_not_in_vram_bytes() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("the real artifact is absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(d) => d,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA device unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let device_load = held_at(&reader, &mut device, VisionResidency::Device);
    let host_load = held_at(&reader, &mut device, VisionResidency::Host);
    eprintln!(
        "device: vram {} host-mapped {} workspace {}; host: vram {} host-mapped {} workspace {}",
        device_load.vram_bytes,
        device_load.host_mapped_bytes,
        device_load.reserved.workspace_bytes,
        host_load.vram_bytes,
        host_load.host_mapped_bytes,
        host_load.reserved.workspace_bytes
    );

    assert_eq!(device_load.vram_bytes - host_load.vram_bytes, TOWER_BYTES, "the tower leaves the VRAM bytes");
    assert_eq!(host_load.host_mapped_bytes - device_load.host_mapped_bytes, TOWER_BYTES, "and enters the host-mapped bytes");
    // The pool stays on the device in this build, whatever the residency.
    assert_eq!(host_load.reserved.media_embedding_bytes, device_load.reserved.media_embedding_bytes);
    // Each load holds what its plan said, the host one's ring included.
    for (name, held) in [("device", &device_load), ("host", &host_load)] {
        assert_eq!(held.reserved, held.planned, "{name}: the load holds its plan");
    }
    // At a 1,024-token item and a 128-token chunk the encoder's share is the
    // larger, so the ring is what host adds to the scratch, and to vision's
    // reservation.
    let grown = host_load.reserved.workspace_bytes - device_load.reserved.workspace_bytes;
    assert!((RING_BYTES..RING_BYTES + 256).contains(&grown), "the ring after the aligned workspace: {grown}");
    assert_eq!(host_load.vision_reserved_bytes - device_load.vision_reserved_bytes, grown);
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn on_the_16_gb_profile_the_ring_fits_the_prefill_scratch_at_1024_and_4096() {
    let path = Path::new(TRELLIS);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("the 16 GB artifact is absent: {TRELLIS}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {TRELLIS}: {e}"));
    // `rtx50-16gb`'s 27B: 262,144 tokens of hq-e8-2b at 1,024-token chunks,
    // DFlash2 at 7 with the full head, two decode lanes.
    let speculation = Speculation::new(SpeculativeBackend::Dflash2, 7).expect("dflash2-7");
    let workspace = |vision: Option<Vision>| {
        let (plan, handles) =
            bind_model_scope_27b_with(&reader, model_scope(Some(speculation), vision, ExceptionWeights::Bf16))
                .unwrap_or_else(|e| panic!("bind: {e}"));
        plan_qwen38_27b_reservations(
            &reader,
            &plan,
            &handles,
            1024,
            262_144,
            KvFormat::HqE8_2b,
            Some(speculation),
            vision,
            RopeScaling::NONE,
            PrefillAttention::default(),
            2,
        )
        .unwrap_or_else(|e| panic!("plan {vision:?}: {e}"))
        .workspace_bytes
    };
    let text = workspace(None);
    eprintln!("16 GB profile: text-only workspace {text}");
    for envelope in [1024, 4096] {
        let at = |residency| Vision::new(envelope).unwrap().with_item_max_tokens(ITEM_BOUND).with_residency(residency);
        let device = workspace(Some(at(VisionResidency::Device)));
        let host = workspace(Some(at(VisionResidency::Host)));
        eprintln!("16 GB profile at a {envelope}-token envelope: workspace device {device}, host {host}");
        assert_eq!(host, device, "the encode of a {envelope}-token item and the ring fit the scratch");
    }
    // Where the encoder's share sets the scratch's size, the ring is what it grows by.
    let at = |residency| Vision::new(ITEM_BOUND).unwrap().with_item_max_tokens(ITEM_BOUND).with_residency(residency);
    let device = workspace(Some(at(VisionResidency::Device)));
    let host = workspace(Some(at(VisionResidency::Host)));
    eprintln!("16 GB profile at a {ITEM_BOUND}-token item: workspace device {device}, host {host}");
    assert!(device > text, "the encoder's share sets the scratch here");
    assert!((RING_BYTES..RING_BYTES + 256).contains(&(host - device)), "{}", host - device);
}
