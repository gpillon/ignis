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
    let leaf = CudaLeaf::new(device, reader, artifact, handles, config);
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
