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
use ignis_core::{KvFormat, KvGeometry, Speculation, SpeculativeBackend, Vision, model_load};
use ignis_runtime::{CudaLeaf, CudaLeafConfig, Model, ReservedBytes};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 262_144;

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_planned_reservations_are_what_the_load_holds() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let speculation = Speculation::new(SpeculativeBackend::Dflash2, 7).expect("dflash2-7");
    let vision = Vision::new(ignis_core::DEFAULT_VISION_MAX_TOKENS).expect("default envelope");
    let scope = model_load::model_scope(Some(speculation), Some(vision));
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
        // GitHub #211: retained slots are a line of their own.
        retained_slots: 2,
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
    let vision = Vision::new(ignis_core::DEFAULT_VISION_MAX_TOKENS).expect("default envelope");
    let reserved = |prefill_chunk_tokens: u32, max_context_tokens: u32, vision: Option<Vision>| {
        let scope = model_load::model_scope(Some(speculation), vision);
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
    // A text load reserves what it did before #212: the `prefill_scratch`
    // line the plan logged at these options on 2026-09-17.
    assert_eq!(text.workspace, 1_481_902_336);
    assert_eq!(text.media_embedding, 0);
    assert!(with_vision.media_embedding > 0, "the media embedding keeps its own reservation");

    // A short context caps the envelope below the prefill scratch: vision
    // then grows the arena by the attention readout alone (GitHub #260) —
    // one f32 per envelope token, the envelope capped by the context — and
    // the head set's results beside it (GitHub #263): 8 bytes for each of
    // the 384 heads a set may name.
    const SHORT_CONTEXT: u32 = 2048;
    const HEAD_SET_RESULTS: u64 = 16 * 24 * 8;
    assert_eq!(
        reserved(1024, SHORT_CONTEXT, Some(vision)).workspace,
        reserved(1024, SHORT_CONTEXT, None).workspace + u64::from(SHORT_CONTEXT) * 4 + HEAD_SET_RESULTS,
        "the prefill scratch already fits the encoder, and a head point's readout beside it"
    );
}
