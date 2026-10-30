//! GPU coverage for the load's VRAM plan (GitHub #210, ADR 0030): what the
//! leaf says a load will reserve, asked before the weights are on the device,
//! is to the byte what the loaded model and pool then hold -- every workspace,
//! every lane's state, the retained slots and the KV arena.
//!
//! Run at the Makefile's serving shape (262K hq-e8-2b, DFlash2 with a
//! 7-token window, vision), where every line is nonzero. Its own test binary:
//! it materializes the artifact, and the card fits one at a time.

#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::Arc;

use ignis_artifact::{CudaDevice, Reader, bind_model_scope_27b_with, materialize};
use ignis_core::gpu_profile;
use ignis_core::lane_lines::lane_lines;
use ignis_core::{ExceptionWeights, KvFormat, KvGeometry, N_DECODE_LANES, Speculation, SpeculativeBackend, Vision, VisionResidency, model_load};
use ignis_runtime::{CudaLeaf, CudaLeafConfig, Model, ReservedBytes};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 262_144;

/// One retained image at the serving shape (hq-e8-2b, DFlash2), as the host
/// block packs it: 196,880,384 bytes of slot state and the hq residual
/// window's 35,651,648 rounded up to the packed image's 256-byte section
/// alignment (35,651,840).
const HOST_IMAGE_BYTES: u64 = 232_532_224;
/// One slot's hq residual window.
const HQ_WINDOW_BYTES: u64 = 35_651_648;

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_planned_reservations_are_what_the_load_holds() {
    planned_reservations_are_held(ExceptionWeights::Bf16);
}

/// GitHub #341: the same with the BF16 exception tensors held in FP8 -- the
/// plan reads their conversions' format and bytes off the binder's plan, the
/// load off the materialized views, and the two lay out the same arenas.
#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_planned_reservations_are_what_an_fp8_exceptions_load_holds() {
    planned_reservations_are_held(ExceptionWeights::Fp8);
}

fn planned_reservations_are_held(exceptions: ExceptionWeights) {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let speculation = Speculation::new(SpeculativeBackend::Dflash2, 7).expect("dflash2-7");
    // The tower and the pool on the device (ADR 0075's `device`), where the
    // media embedding is a VRAM line of its own.
    let vision = Vision::new(ignis_core::DEFAULT_VISION_MAX_TOKENS)
        .expect("default envelope")
        .with_residency(VisionResidency::Device);
    let scope = model_load::model_scope(Some(speculation), Some(vision), exceptions);
    let (plan, handles) =
        bind_model_scope_27b_with(&reader, scope).unwrap_or_else(|e| panic!("bind: {e}"));

    // One full context of pages: every line is exercised, and the pool stays
    // small enough to sit beside a desktop.
    let pages = MAX_CONTEXT.div_ceil(64);
    let page_bytes = KvFormat::HqE8_2b.page_bytes(KvGeometry::qwen38_27b());
    let config = CudaLeafConfig {
        max_context_tokens: MAX_CONTEXT,
        kv_format: KvFormat::HqE8_2b,
        kv_pool_bytes: u64::from(pages) * page_bytes,
        speculation: Some(speculation),
        vision: Some(vision),
        // GitHub #211: device retained slots are a line of their own; the
        // host ones (GitHub #281) are host memory beside the plan.
        retained_device_slots: 2,
        retained_host_slots: 3,
        ..CudaLeafConfig::default()
    };
    // Planned before any device memory exists, as the server does.
    let planned = config
        .plan_reservations(&reader, &plan, &handles)
        .unwrap_or_else(|e| panic!("plan the reservations: {e}"));
    let kv_arena = config.kv_pool_arena_bytes(pages).unwrap_or_else(|e| panic!("plan the pool: {e}"));
    for (line, bytes) in [
        ("workspace", planned.reserved.workspace),
        ("media_embedding", planned.reserved.media_embedding),
        ("sampling", planned.reserved.sampling),
        ("decode_graph", planned.reserved.decode_graph),
        ("verify_round", planned.reserved.verify_round),
        ("drafter_round", planned.reserved.drafter_round),
        ("lane_state", planned.reserved.lane_state),
        ("retained_slots", planned.reserved.retained_slots),
        // GitHub #257: an hq-e8-2b load keeps the residual window.
        ("hq_residual_window", planned.reserved.hq_residual_window),
    ] {
        assert!(bytes > 0, "{line} is planned at 0 bytes on a load that reserves it");
    }
    assert!(
        kv_arena > u64::from(pages) * page_bytes,
        "the KV arena carries its block tables beside the pages"
    );
    assert_eq!(planned.retained_host_bytes, 3 * HOST_IMAGE_BYTES, "one packed image per host slot");
    assert_eq!(
        planned.reserved.hq_residual_window,
        (N_DECODE_LANES as u64 + 2) * HQ_WINDOW_BYTES,
        "the residual window is the lanes' and the device slots', never a host slot's"
    );

    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!();
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize: {e}")) {
                return;
            }
            unreachable!();
        }
    };
    // GitHub #213: KV-RAM is one arena pinned at the load, so a leaf without
    // one refuses every spill. A gibibyte costs this load nothing it notices
    // and keeps a leg that starts spilling from failing for the wrong reason.
    let leaf = CudaLeaf::new(device, reader, artifact, handles, config)
        .with_kv_ram_arena(1024 * 1024 * 1024)
        .unwrap_or_else(|e| panic!("pin KV-RAM: {e}"));
    let model = Arc::new(Model::load(Arc::new(leaf)).unwrap_or_else(|e| panic!("model load: {e:?}")));
    let stats = model.stats().unwrap_or_else(|e| panic!("stats: {e:?}"));
    assert_eq!(stats.kv_page_count, pages);
    assert_eq!(
        stats.reserved,
        ReservedBytes {
            kv_pool: kv_arena,
            ..planned.reserved
        },
        "the load holds what its plan laid out"
    );
}

/// GitHub #281: a load at the server's defaults keeps no retained slot in
/// VRAM -- the `retained_slots` line is 0 and the residual window is the
/// lanes' alone -- and sixteen in the pinned host block. Asked of the leaf's
/// plan only, so nothing here touches the device.
#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_default_retained_slots_are_on_the_host() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let speculation = Speculation::new(SpeculativeBackend::Dflash2, 7).expect("dflash2-7");
    let scope = model_load::model_scope(Some(speculation), None, ExceptionWeights::Bf16);
    let (plan, handles) = bind_model_scope_27b_with(&reader, scope).unwrap_or_else(|e| panic!("bind: {e}"));
    let config = CudaLeafConfig {
        max_context_tokens: MAX_CONTEXT,
        kv_format: KvFormat::HqE8_2b,
        speculation: Some(speculation),
        ..CudaLeafConfig::default()
    };
    let planned = config
        .plan_reservations(&reader, &plan, &handles)
        .unwrap_or_else(|e| panic!("plan the reservations: {e}"));
    assert_eq!(planned.reserved.retained_slots, 0, "no retained slot in VRAM by default");
    assert_eq!(
        planned.reserved.hq_residual_window,
        N_DECODE_LANES as u64 * HQ_WINDOW_BYTES,
        "the lanes' window alone"
    );
    assert_eq!(
        planned.retained_host_bytes,
        u64::from(ignis_runtime::DEFAULT_RETAINED_HOST_SLOTS) * HOST_IMAGE_BYTES,
        "the default host slots, one packed image each"
    );
}

/// GitHub #212: the prefill scratch and the vision encoder's workspace are one
/// arena sized for the larger of the two, not two arenas added up. Asked of
/// the leaf's plan only, so nothing here touches the device.
#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_vision_encoder_workspace_is_the_prefill_scratch_not_beside_it() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let speculation = Speculation::new(SpeculativeBackend::Dflash2, 7).expect("dflash2-7");
    // The device residency (ADR 0075's `device`): its workspace has no staging
    // ring, and its media embedding is a VRAM line.
    let vision = Vision::new(ignis_core::DEFAULT_VISION_MAX_TOKENS)
        .expect("default envelope")
        .with_residency(VisionResidency::Device);
    let reserved_as = |prefill_chunk_tokens: u32, max_context_tokens: u32, vision: Option<Vision>, exceptions| {
        let scope = model_load::model_scope(Some(speculation), vision, exceptions);
        let (plan, handles) = bind_model_scope_27b_with(&reader, scope).unwrap_or_else(|e| panic!("bind: {e}"));
        let config = CudaLeafConfig {
            max_context_tokens,
            kv_format: KvFormat::HqE8_2b,
            prefill_chunk_tokens,
            speculation: Some(speculation),
            vision,
            ..CudaLeafConfig::default()
        };
        config
            .plan_reservations(&reader, &plan, &handles)
            .unwrap_or_else(|e| panic!("plan the reservations: {e}"))
            .reserved
    };
    let reserved = |prefill_chunk_tokens: u32, max_context_tokens: u32, vision: Option<Vision>| {
        reserved_as(prefill_chunk_tokens, max_context_tokens, vision, ExceptionWeights::Bf16)
    };

    // The serving context, where the encoder's 32,768-token workspace is the
    // larger: the chunk width moves a text load's scratch and not a vision
    // load's. Added up, both would move by the same amount.
    let text = reserved(1024, MAX_CONTEXT, None);
    let narrow_text = reserved(128, MAX_CONTEXT, None);
    let with_vision = reserved(1024, MAX_CONTEXT, Some(vision));
    let narrow_with_vision = reserved(128, MAX_CONTEXT, Some(vision));
    eprintln!(
        "workspace: text {} (128-token chunk {}), vision {} (128-token chunk {})",
        text.workspace, narrow_text.workspace, with_vision.workspace, narrow_with_vision.workspace
    );
    assert!(narrow_text.workspace < text.workspace, "the chunk width sizes a text load's scratch");
    assert!(with_vision.workspace > text.workspace, "the encoder's workspace is the larger here");
    assert_eq!(with_vision.workspace, narrow_with_vision.workspace, "one arena, sized by the encoder alone");
    // For each of the 384 heads a set may name, its packed result (8 bytes,
    // GitHub #263) and its four neighbour scores (16 bytes, GitHub #264).
    const HEAD_SET_BYTES: u64 = 16 * 24 * (8 + 16);
    // GitHub #275 (ADR 0041), #278 (ADR 0042): the served artifact is
    // calibrated for `locate`, so every load of it -- text or vision --
    // reserves a text readout's room over `LOCATE_WINDOW_KEYS`, the widest
    // span one reading reads: one f32 per key (rounded to the arena's
    // 256-byte alignment), the head set's buffers, and 32 whole rows.
    let text_keys = u64::from(ignis_core::locate::calibration(ignis_core::ArtifactHash::from_bytes(reader.content_hash()))
        .expect("the served artifact is calibrated for locate")
        .window_keys);
    let aligned = |bytes: u64| bytes.div_ceil(256) * 256;
    let text_scores = aligned(text_keys * 4);
    let text_room = text_scores + HEAD_SET_BYTES + aligned(32 * text_keys * 4);
    assert_eq!(text_room, 26_409_216, "200,000 keys: 800,000 B of scores, 9,216 of set results, 25,600,000 of rows");

    // A text load reserves its layers' prefill scratch and the text
    // readout's room beside it. GitHub #335 (ADR 0061): without the prompt
    // planes that scratch is 461,427,968 B, where the plan logged
    // 1,481,902,336 at these options on 2026-09-17.
    assert_eq!(text.workspace, 461_427_968 + text_room);
    assert_eq!(text.media_embedding, 0);
    assert!(with_vision.media_embedding > 0, "the media embedding keeps its own reservation");

    // Where the prefill scratch is the larger, vision grows the arena by the
    // attention readout alone (GitHub #260) -- one f32 per token of an item,
    // less the text scores it shares the room with: one readout reads at a
    // time, and the head set's buffers are the text room's already.

    // The serving load's item bound: this artifact's processor lets no item
    // through past 16,384 merged tokens, and the encoder holds one at a
    // time. Sized for that, the encoder fell below the prefill scratch while
    // the prompt planes were in it; without them (GitHub #335) it is the
    // larger again, and the arena is the encoder's alone -- still far below
    // the unbounded envelope's.
    const ITEM_BOUND: u32 = 16_384;
    let bounded = reserved(1024, MAX_CONTEXT, Some(vision.with_item_max_tokens(ITEM_BOUND)));
    let narrow_bounded = reserved(128, MAX_CONTEXT, Some(vision.with_item_max_tokens(ITEM_BOUND)));
    eprintln!("workspace: vision with a {ITEM_BOUND}-token item bound {}", bounded.workspace);
    assert!(bounded.workspace > text.workspace, "the item-bounded encoder is the larger without the planes");
    assert!(bounded.workspace < with_vision.workspace, "the item bound shrinks the encoder's workspace");
    assert_eq!(bounded.workspace, narrow_bounded.workspace, "one arena, sized by the bounded encoder alone");
    assert_eq!(bounded.media_embedding, with_vision.media_embedding, "the pool keeps the envelope's floor");

    // GitHub #341: the rtx5090 profile prefills 2,048 tokens a chunk. Its text scratch is still
    // below the bounded encoder, so with vision on the wider chunk costs no byte; at 4,096 the
    // scratch passes the encoder and the arena grows.
    let wide_text = reserved(2048, MAX_CONTEXT, None);
    let wide_bounded = reserved(2048, MAX_CONTEXT, Some(vision.with_item_max_tokens(ITEM_BOUND)));
    let widest_bounded = reserved(4096, MAX_CONTEXT, Some(vision.with_item_max_tokens(ITEM_BOUND)));
    eprintln!(
        "workspace: text at a 2,048-token chunk {}, bounded vision at 2,048 {} and at 4,096 {}",
        wide_text.workspace, wide_bounded.workspace, widest_bounded.workspace
    );
    // ~399 KB per chunk token: 12,128,000 B short of the bounded encoder at 2,048, 669,716,736 B
    // past it at 4,096 (the same arena the rtx5090 profile plans at 917,504 tokens).
    assert_eq!(wide_text.workspace, 870_055_680 + text_room);
    assert!(wide_text.workspace < bounded.workspace, "at 2,048 the bounded encoder is still the larger");
    assert_eq!(wide_bounded.workspace, bounded.workspace, "with vision on, the 2,048 chunk is free");
    assert_eq!(widest_bounded.workspace, 1_578_309_632, "at 4,096 the text scratch is the larger");

    // GitHub #341: held in FP8, the exception tensors' prefill quantizes its
    // activations into the call's workspace (a code per input and a scale per
    // token). Printed with the bounded encoder beside it: whether the 2,048
    // chunk stays free with vision on.
    let fp8_text = reserved_as(2048, MAX_CONTEXT, None, ExceptionWeights::Fp8);
    let fp8_bounded =
        reserved_as(2048, MAX_CONTEXT, Some(vision.with_item_max_tokens(ITEM_BOUND)), ExceptionWeights::Fp8);
    eprintln!(
        "workspace with FP8 exception tensors at 2,048: text {} ({:+} B), bounded vision {}",
        fp8_text.workspace,
        fp8_text.workspace as i64 - wide_text.workspace as i64,
        fp8_bounded.workspace
    );
    // 5,513,216 B more: a 2,048-token exception call's E4M3 codes and scales (12,591,104 B at
    // K = 6,144), less the room the layer's scratch already had at that step.
    assert_eq!(fp8_text.workspace, 875_568_896 + text_room);
    assert_eq!(fp8_bounded.workspace, bounded.workspace, "with vision on, still the encoder's arena");

    // A short context caps the envelope, and with it the item, below the
    // prefill scratch the same way -- and caps the text span too, to the
    // same 2,048 keys, so a head point's readout is the text room already.
    const SHORT_CONTEXT: u32 = 2048;
    assert_eq!(
        reserved(1024, SHORT_CONTEXT, Some(vision)).workspace,
        reserved(1024, SHORT_CONTEXT, None).workspace,
        "the prefill scratch already fits the encoder, and the text room a head point's readout"
    );
}

/// GitHub #336 (spec 27b-16gb/04 acceptance 1): every line that scales with
/// the round follows the load's decode lanes, and at eight lanes the plan is
/// main 75210360's to the byte. Two shapes: the `rtx5090` profile's 27B
/// (917,504 tokens of hq-e8-2b, DFlash2 at 7, vision with the processor's
/// 16,384-token item bound, 2,048-token chunks) and a 262,144-token text
/// load at 1,024; two device and sixteen host retained slots on both, so
/// the lines the lanes do not move are there to stay put. Asked of the
/// leaf's plan only, so nothing here touches the device.
///
/// At 1, 2 and 8 lanes the leaf plans, to the byte, what
/// `ignis_core::lane_lines` computes on the CPU -- sampling, the verify and
/// the drafter's rounds, the lanes' state and the hq residual window -- which
/// `cargo test` holds to main's eight-lane bytes and PIANO-VRAM's per-lane
/// costs. The decode round is the kernel's alone (that module's doc says
/// why): pinned here at one and two lanes, and exactly affine in them --
/// ~10.27 MB a lane beside 1,024 bytes, and vision's rope positions -- so the
/// four round lines are ~35.4 MB a lane beside ~33.6 MB that does not move
/// (the drafter's 32 MiB allowance and its fixed buffers): PIANO-VRAM's
/// "~35 MB a lane + 34 MB fixed" at DFlash2 7, fitted on 39 measured plans.
#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_round_lines_follow_the_loads_lanes() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let speculation = Speculation::new(SpeculativeBackend::Dflash2, 7).expect("dflash2-7");
    // What main planned is the device residency's (ADR 0075's `device`).
    let vision = Vision::new(ignis_core::DEFAULT_VISION_MAX_TOKENS)
        .expect("default envelope")
        .with_item_max_tokens(16_384)
        .with_residency(VisionResidency::Device);
    let topology = ignis_core::ModelConfig::qwen38_27b();

    // What main planned at eight lanes, and the decode round at one and two.
    struct Shape {
        label: &'static str,
        max_context_tokens: u32,
        prefill_chunk_tokens: u32,
        vision: Option<Vision>,
        main: ReservedBytes,
        decode_graph: [u64; 2],
    }
    let main = |workspace, media_embedding, decode_graph, verify_round| ReservedBytes {
        workspace,
        // GitHub #339: the served artifact's body is NVFP4.
        trellis_workspace: 0,
        media_embedding,
        sampling: 4_623_344,
        decode_graph,
        verify_round,
        drafter_round: 78_027_552,
        lane_state: 1_575_043_072,
        retained_slots: 393_760_768,
        hq_residual_window: 356_516_480,
        kv_pool: 0,
    };
    let shapes = [
        Shape {
            label: "rtx5090",
            max_context_tokens: 917_504,
            prefill_chunk_tokens: 2048,
            vision: Some(vision),
            main: main(908_592_896, 335_544_320, 82_125_920, 152_139_488),
            decode_graph: [10_266_636, 20_532_248],
        },
        Shape {
            label: "text262k",
            max_context_tokens: 262_144,
            prefill_chunk_tokens: 1024,
            vision: None,
            main: main(487_837_184, 0, 82_125_888, 152_139_232),
            decode_graph: [10_266_632, 20_532_240],
        },
    ];
    for shape in &shapes {
        let scope = model_load::model_scope(Some(speculation), shape.vision, ExceptionWeights::Bf16);
        let (plan, handles) = bind_model_scope_27b_with(&reader, scope).unwrap_or_else(|e| panic!("bind: {e}"));
        let planned_at = |lanes: u32| {
            let config = CudaLeafConfig {
                max_context_tokens: shape.max_context_tokens,
                kv_format: KvFormat::HqE8_2b,
                slot_count: lanes,
                prefill_chunk_tokens: shape.prefill_chunk_tokens,
                speculation: Some(speculation),
                vision: shape.vision,
                retained_device_slots: 2,
                retained_host_slots: 16,
                ..CudaLeafConfig::default()
            };
            config.plan_reservations(&reader, &plan, &handles).unwrap_or_else(|e| panic!("plan at {lanes} lanes: {e}"))
        };
        let eight = planned_at(N_DECODE_LANES as u32);
        assert_eq!(eight.reserved, shape.main, "{}: eight lanes are main's plan to the byte", shape.label);
        assert_eq!(eight.retained_host_bytes, 16 * HOST_IMAGE_BYTES, "{}", shape.label);

        let [decode_one, decode_two] = shape.decode_graph;
        let mut rounds = Vec::new();
        for (lanes, decode_graph) in [(1, decode_one), (2, decode_two), (8, shape.main.decode_graph)] {
            let at = planned_at(lanes).reserved;
            let lines = lane_lines(&topology, KvFormat::HqE8_2b, Some(speculation), shape.vision.is_some(), lanes, 2);
            assert_eq!(
                at,
                ReservedBytes {
                    sampling: lines.sampling,
                    decode_graph,
                    verify_round: lines.verify_round,
                    drafter_round: lines.drafter_round,
                    lane_state: lines.lane_state,
                    hq_residual_window: lines.hq_residual_window,
                    ..shape.main
                },
                "{} at {lanes} lanes: the CPU's lines, and the workspace, the media pool and the retained slots stay put",
                shape.label
            );
            rounds.push(at.sampling + at.decode_graph + at.verify_round + at.drafter_round);
        }
        assert_eq!(
            shape.main.decode_graph,
            decode_one + 7 * (decode_two - decode_one),
            "{}: the decode round is one lane and seven more of the step from one to two",
            shape.label
        );
        let (per_lane, fixed) = (rounds[1] - rounds[0], 2 * rounds[0] - rounds[1]);
        assert!(per_lane.abs_diff(35_000_000) <= 500_000, "{}: the round lines cost {per_lane} B a lane", shape.label);
        assert!(fixed.abs_diff(34_000_000) <= 500_000, "{}: {fixed} B of round lines do not move", shape.label);
    }
}
