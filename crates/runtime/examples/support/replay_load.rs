//! The 27B load the DFlash2 replay examples run on (GitHub #340), at a
//! [`LoadShape`] (`replay_shape.rs`). Planned with the server's own pieces
//! (`ignis_server::runtime::qwen27b_load`): the leaf's lines at the load's
//! options, laid out by `ignis_runtime::vram_lines` -- the CUDA context and
//! the measured residual among them -- and a KV pool of the rest of the VRAM
//! budget from the same `plan_vram`; then loaded the way
//! `CudaLeaf::load_model` loads, with every plan line logged, the leaf's
//! buffers read back against the plan (`ReservedBytes::check_planned`) and
//! the pool's pages against its plan (`verified_kv_pool`), both as the
//! server refuses a start.
//!
//! Where it is not the server's load:
//! - It does not adapt (ADR 0063 §3): a plan that does not fit is refused,
//!   naming its shortfall, and the shape is never moved to make it fit.
//! - It runs no host plan (ADR 0062): the server measures available RAM
//!   against the pinned token embedding (the trellis body's), KV-disk
//!   staging, the host retained slots and the KV-RAM arena before the first
//!   pinned allocation. This load holds the pinned embedding only -- no
//!   KV-RAM arena, no KV disk, no retained slots -- and a short host fails at
//!   materialization instead of naming the line.
//! - The leaf config is built here, mirroring the server's
//!   `leaf_config_for_shape`, which reads the server's `EngineShape`.
//!
//! No retained slots: the replay never publishes or captures a prefix. The
//! server's host ones (`--reuse-retained-host`, 16) are pinned host memory,
//! no line of the VRAM plan, and its device ones default to none.

use ignis_artifact::{bind_model_scope_27b_with, body_27b, materialize, CudaDevice, MaterializedArtifact, Reader};
use ignis_core::kv::{verified_kv_pool, LeafPoolGeometry};
use ignis_core::model_load::{self, Model};
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::capture_decode_graphs;
use ignis_core::{
    plan_vram, ExceptionWeights, KvFormat, KvGeometry, ModelConfig, PrefillAttention, Residency, VramRequest,
};
use ignis_runtime::CudaLeafConfig;

use crate::replay_shape::LoadShape;

/// The leaf config the server's `leaf_config_for_shape` builds for `shape`,
/// with a KV pool of `kv_pool_bytes`.
fn leaf_config(shape: &LoadShape, kv_pool_bytes: u64) -> CudaLeafConfig {
    CudaLeafConfig {
        max_context_tokens: shape.max_context,
        kv_format: KvFormat::HqE8_2b,
        kv_pool_bytes,
        slot_count: shape.lanes,
        retained_device_slots: 0,
        retained_host_slots: 0,
        prefill_chunk_tokens: shape.prefill_chunk,
        speculation: shape.speculation(),
        vision: None,
        rope_scaling: shape.rope,
        prefill_attention: PrefillAttention::default(),
        prefill_linear: ignis_core::PrefillLinear::default(),
    }
}

/// A loaded model, its pool and what they stand on.
pub struct Loaded {
    pub model: Model,
    pub pool: SeqPool,
    artifact: MaterializedArtifact,
    device: CudaDevice,
}

impl Loaded {
    /// Give the model, the pool and the weights back, in that order (the
    /// arena has no `Drop` of its own), so the next load starts from the
    /// memory this one took.
    pub fn release(self) {
        let Loaded { model, pool, mut artifact, mut device } = self;
        drop(model);
        drop(pool);
        let _ = artifact.release_arena(&mut device);
    }
}

/// Plan `reader`'s load at `shape`, log the plan, and load it -- refusing a
/// plan its budget cannot hold, or a load whose buffers are not its plan's.
pub fn load(reader: &Reader, shape: LoadShape) -> Result<Loaded, String> {
    let body = body_27b(reader);
    // What is free before this process holds anything, read before the
    // context exists, as the server reads it: the context is a plan line, the
    // figure it took when the first load made it (GitHub #604). A later load
    // in the process adds back what it still holds (GitHub #324).
    let (start, mut device) = ignis_runtime::plan_start(ignis_core::compute::ModelFamily::Qwen38_27b)?;
    let free_at_start = start.free_at_start_bytes;
    let nvml_free = free_at_start - start.held.bytes;
    let speculation = shape.speculation();
    let scope = model_load::model_scope(speculation, None, ExceptionWeights::default());
    let (bound, handles) = bind_model_scope_27b_with(reader, scope).map_err(|e| format!("bind: {e}"))?;

    let planning = leaf_config(&shape, 0);
    let reservations = planning.plan_reservations(reader, &bound, &handles)?;
    planning.kv_pool_arena_bytes(1)?;
    let kv_arena_bytes = |pages: u32| planning.kv_pool_arena_bytes(pages).unwrap_or(u64::MAX);
    let geometry = KvGeometry::qwen38_27b();
    let format = planning.kv_format;
    let plan = plan_vram(&VramRequest {
        mode: shape.vram,
        free_at_start_bytes: free_at_start,
        lines: ignis_runtime::vram_lines(bound.device_capacity_bytes, start.context.bytes, reservations.reserved),
        kv_page_bytes: format.page_bytes(geometry),
        max_context_tokens: shape.max_context,
        retained_slots: 0,
        kv_pool: None,
        residency: Residency::Resident,
        embedding_pool_named: false,
        kv_arena_bytes: &kv_arena_bytes,
        can_page: cfg!(windows),
    })
    .map_err(|e| format!("VRAM plan: {e}"))?;
    println!("load shape: {body:?} body, {shape:?}, kv {}, lines (bytes):", format.as_str());
    for (name, bytes) in plan.lines.entries() {
        println!("  {name:<20} {bytes:>14}");
    }
    println!(
        "  {:<20} {:>14} ({} pages, {} tokens)\n  total {} of a {} budget ({}; {} free at start)",
        "kv_pool",
        plan.kv_pool_bytes,
        plan.kv_page_count,
        plan.kv_token_capacity(),
        plan.total_bytes,
        plan.budget_bytes,
        plan.mode.as_str(),
        plan.free_at_start_bytes
    );
    for warning in &plan.warnings {
        println!("  warning: {warning}");
    }
    if plan.total_bytes > plan.budget_bytes {
        return Err(format!("the plan holds {} bytes, past its {}-byte budget", plan.total_bytes, plan.budget_bytes));
    }

    let artifact = materialize(reader, &bound, &mut device, None).map_err(|e| format!("materialize: {e}"))?;
    let config = leaf_config(&shape, plan.kv_budget_bytes(format, geometry));
    let pages = config.kv_pool_plan().map_err(|e| e.to_string())?.page_count;
    let model = model_load::load_qwen38_27b_with_decode_lanes(
        reader,
        &artifact,
        &handles,
        config.prefill_chunk_tokens,
        config.max_context_tokens,
        config.kv_format,
        config.speculation,
        config.vision,
        config.rope_scaling,
        config.prefill_attention,
        config.prefill_linear,
        config.slot_count,
    )
    .map_err(|e| format!("model load: {e}"))?;
    ignis_runtime::note_qwen27b_loaded();
    let pool = SeqPool::create_with_speculation(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: config.kv_format,
            kv_page_group_count: pages,
            max_context_tokens: config.max_context_tokens,
            slot_count: config.slot_count,
            retained_slot_count: config.retained_device_slots,
            retained_host_slot_count: config.retained_host_slots,
        },
        config.speculation.map(|s| s.backend()),
    )
    .map_err(|e| format!("seq pool: {e}"))?;
    let capture = capture_decode_graphs(&model, &pool).map_err(|e| format!("capture: {e}"))?;
    println!("decode graphs: {} of {} widths ready in {} us", capture.ready_count(), shape.lanes, capture.capture_micros);
    // The replay's rounds are one lane wide: without that graph they would
    // run eagerly, which is not the round serving runs.
    assert!(capture.is_ready(1), "no width-1 graph");

    // What the leaf built against what the plan laid out, the pool's pages
    // and every line beside the weights: the server refuses a start where
    // they differ, and so does this.
    let pool_stats = pool.stats();
    verified_kv_pool(
        pages,
        LeafPoolGeometry { page_count: pool_stats.kv_page_group_count, page_bytes: pool_stats.kv_page_bytes },
    )
    .map_err(|e| e.to_string())?;
    ignis_runtime::loaded_reserved_bytes(&model, &pool).check_planned(reservations.reserved, plan.kv_pool_bytes)?;
    if let Ok((free_after, _)) = CudaDevice::nvml_memory(0) {
        println!(
            "loaded: {} bytes planned, NVML free {nvml_free} -> {free_after} ({} taken)",
            plan.total_bytes,
            nvml_free.saturating_sub(free_after)
        );
    }
    Ok(Loaded { model, pool, artifact, device })
}
