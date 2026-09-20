//! Server construction over the safe step-ABI compute adapter.
//!
//! [`scheduler`] is the leaf-agnostic construction point: a loader hands it
//! a model handle and EOS token without leaking either into HTTP or
//! scheduler code. [`cuda_scheduler`] (GitHub #61 / P1-25, feature `cuda`)
//! is the production path built on it: materialize an artifact's weights
//! on the device and wrap the loaded model behind
//! [`ignis_runtime::CudaLeaf`] — the binary falls back to CPU `MockCompute`
//! (`main.rs`) only without this feature or without an artifact.

use std::sync::Arc;

use ignis_core::{ConcreteScheduler, SchedulerConfig, TokenId};
use ignis_runtime::{Model, RuntimeCompute, StepLeaf};

/// Build the server's concrete scheduler over a loaded step-ABI model.
pub fn scheduler<L: StepLeaf>(
    config: SchedulerConfig,
    model: Arc<Model<L>>,
    eos: TokenId,
) -> ConcreteScheduler {
    ConcreteScheduler::with_config(config, Arc::new(RuntimeCompute::new(model, eos)))
}

/// The engine shape the operator configured (`--prefill-chunk`,
/// `--max-context` — GitHub #87, resolved and validated by [`crate::config`]
/// before any loader work starts). Carried as one value so the flags
/// travel together from `main` to the leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineShape {
    /// The prefill chunk width, in tokens (a nonzero multiple of 128).
    pub prefill_chunk: u32,
    /// The maximum per-sequence context, in tokens.
    pub max_context: u32,
    /// The KV storage format the load runs on (`--kv-format`, GitHub #122).
    pub kv_format: ignis_core::KvFormat,
    /// The paged-KV pool budget, in bytes (`--kv-pool-bytes`). `None` gives
    /// the pool the rest of the VRAM budget (GitHub #210).
    pub kv_pool_bytes: Option<u64>,
    /// How the load's VRAM budget is chosen (GitHub #210, ADR 0030).
    pub vram: ignis_core::VramMode,
    /// The KV-RAM host tier's budget, in bytes (`--kv-host-pool-bytes`,
    /// P4-07 GitHub #125): pinned host memory for evicted (suspended)
    /// request snapshots, independent of the GPU-resident pool above.
    pub host_pool_bytes: u64,
    /// Cross-request state reuse (`--prompt-reuse`, GitHub #186, ADR 0029).
    pub prompt_reuse: bool,
    /// The retained slots the load reserves (`--retained-slots`, GitHub #215,
    /// ADR 0030): 0 with prompt reuse off.
    pub retained_slots: u32,
    /// How long a retained Interactive checkpoint in KV-RAM keeps its class's
    /// priority (`--retained-interactive-ttl`, GitHub #190).
    pub retained_interactive_ttl: std::time::Duration,
    /// Speculative decoding (`--spec`/`--draft-tokens`, P5-02 GitHub #150):
    /// `None` binds nothing of the drafter.
    pub speculation: Option<ignis_core::Speculation>,
    /// Vision (`--vision`/`--vision-max-tokens`, GitHub #177): `None` binds
    /// and reserves nothing of the vision tower.
    pub vision: Option<ignis_core::Vision>,
    /// The text rotary table (`--rope-scaling`, GitHub #227):
    /// [`ignis_core::RopeScaling::NONE`] is the linear one, a YaRN factor
    /// rescales the checkpoint's trained position envelope.
    pub rope_scaling: ignis_core::RopeScaling,
}

impl Default for EngineShape {
    /// The configured defaults, taken from [`ignis_runtime`] rather than
    /// restated here — one source of truth for what `ignis-server` runs
    /// with when the operator passes no flags.
    fn default() -> Self {
        Self {
            prefill_chunk: ignis_runtime::DEFAULT_PREFILL_CHUNK,
            max_context: ignis_runtime::DEFAULT_MAX_CONTEXT,
            kv_format: ignis_core::KvFormat::default(),
            kv_pool_bytes: None,
            vram: ignis_core::VramMode::Derived {
                headroom_bytes: crate::config::DEFAULT_VRAM_HEADROOM_BYTES,
            },
            host_pool_bytes: crate::config::DEFAULT_HOST_POOL_BYTES,
            prompt_reuse: crate::config::DEFAULT_PROMPT_REUSE,
            retained_slots: crate::config::DEFAULT_RETAINED_SLOTS,
            retained_interactive_ttl: ignis_core::host::DEFAULT_RETAINED_INTERACTIVE_TTL,
            speculation: None,
            vision: None,
            rope_scaling: ignis_core::RopeScaling::NONE,
        }
    }
}

impl From<&crate::config::Config> for EngineShape {
    fn from(config: &crate::config::Config) -> Self {
        Self {
            prefill_chunk: config.prefill_chunk,
            max_context: config.max_context,
            kv_format: config.kv_format,
            kv_pool_bytes: config.kv_pool_bytes,
            vram: config.vram,
            host_pool_bytes: config.host_pool_bytes,
            prompt_reuse: config.prompt_reuse,
            retained_slots: config.retained_slots,
            retained_interactive_ttl: std::time::Duration::from_secs(u64::from(
                config.retained_interactive_ttl_secs,
            )),
            speculation: config.speculation,
            vision: config.vision,
            rope_scaling: config.rope_scaling,
        }
    }
}

#[cfg(any(feature = "cuda", test))]
fn scheduler_config_for_shape(
    model: String,
    shape: EngineShape,
    kv_page_tokens: u32,
    capacity_pages: u32,
) -> SchedulerConfig {
    SchedulerConfig {
        model,
        kv_page_tokens,
        max_sequence_tokens: shape.max_context,
        kv_capacity_pages: capacity_pages,
        host_capacity_bytes: shape.host_pool_bytes,
        serving_chunk_tokens: shape.prefill_chunk,
        prompt_reuse: shape.prompt_reuse,
        // GitHub #215: the same count the leaf's pool reserved, so every slot
        // the scheduler hands out is one the pool holds.
        retained_slots: shape.retained_slots,
        retained_interactive_ttl: shape.retained_interactive_ttl,
        ..SchedulerConfig::default()
    }
}

/// The VRAM plan's lines for a load (GitHub #210): the weights' arena, the
/// CUDA context, what the leaf planned beside them -- its retained slots
/// included (GitHub #215) -- and the measured residual, in plan order.
pub fn vram_lines(
    weights_bytes: u64,
    reserved: ignis_runtime::ReservedBytes,
) -> ignis_core::VramLines {
    ignis_core::VramLines {
        weights: weights_bytes,
        cuda_context: ignis_runtime::CUDA_CONTEXT_BYTES,
        workspace: reserved.workspace,
        media_embedding: reserved.media_embedding,
        sampling: reserved.sampling,
        decode_graph: reserved.decode_graph,
        verify_round: reserved.verify_round,
        drafter_round: reserved.drafter_round,
        lane_state: reserved.lane_state,
        retained_slots: reserved.retained_slots,
        residual: ignis_runtime::LOAD_RESIDUAL_BYTES,
    }
}

/// The startup report of the VRAM plan (GitHub #210): one
/// `ignis.runtime.vram_plan` event with every line in bytes, the mode, what
/// was free, the headroom or the budget, and whether the plan oversubscribes
/// -- then one `ignis.runtime.vram_oversubscribed` warning per reason the
/// start proceeds anyway.
pub fn log_vram_plan(plan: &ignis_core::VramPlan) {
    let lines = plan.lines;
    macro_rules! vram_plan_event {
        ($($mode_field:ident = $mode_value:expr),*) => {
            // hotpath-lint-allow: one line per model load.
            tracing::info!(
                name: "ignis.runtime.vram_plan",
                mode = plan.mode.as_str(),
                free_at_start_bytes = plan.free_at_start_bytes,
                $($mode_field = $mode_value,)*
                budget_bytes = plan.budget_bytes,
                weights_bytes = lines.weights,
                cuda_context_bytes = lines.cuda_context,
                workspace_bytes = lines.workspace,
                media_embedding_bytes = lines.media_embedding,
                sampling_bytes = lines.sampling,
                decode_graph_bytes = lines.decode_graph,
                verify_round_bytes = lines.verify_round,
                drafter_round_bytes = lines.drafter_round,
                lane_state_bytes = lines.lane_state,
                retained_slots_bytes = lines.retained_slots,
                residual_bytes = lines.residual,
                kv_pool_bytes = plan.kv_pool_bytes,
                kv_page_count = plan.kv_page_count,
                total_bytes = plan.total_bytes,
                allocated_at_load_bytes = plan.total_bytes,
                oversubscribed = plan.oversubscribed,
                "vram plan"
            )
        };
    }
    match plan.mode {
        ignis_core::VramMode::Derived { headroom_bytes } => {
            vram_plan_event!(headroom_bytes = headroom_bytes)
        }
        ignis_core::VramMode::Explicit {
            allow_oversubscription,
            ..
        } => vram_plan_event!(allow_oversubscription = allow_oversubscription),
    }
    for warning in &plan.warnings {
        // hotpath-lint-allow: at most two lines per model load.
        tracing::warn!(name: "ignis.runtime.vram_oversubscribed", warning = %warning, "vram oversubscribed");
    }
}

/// The leaf the operator's [`EngineShape`] is loaded as: the last step
/// before a device exists, and the whole of what `ignis-runtime` is told
/// about the operator's flags.
///
/// Separate from [`cuda_scheduler`] because it is the only link in the chain
/// from a flag to a blob's identity that needs no card — `Config` reaches
/// `EngineShape` by [`From`], `EngineShape` reaches the leaf here, and
/// [`ignis_runtime::CudaLeafConfig::blob_identity`] decides which of *these*
/// fields the identity is made of (GitHub #189). Leaving this inline in
/// `cuda_scheduler` would put a field-by-field copy behind a GPU.
///
/// `kv_pool_bytes` is the pool's payload budget as the VRAM plan resolved it
/// (GitHub #210): the operator's `--kv-pool-bytes`, or the rest of the
/// budget, which only a device can say.
#[cfg(feature = "cuda")]
fn leaf_config_for_shape(shape: EngineShape, kv_pool_bytes: u64) -> ignis_runtime::CudaLeafConfig {
    ignis_runtime::CudaLeafConfig {
        max_context_tokens: shape.max_context,
        kv_format: shape.kv_format,
        kv_pool_bytes,
        prefill_chunk_tokens: shape.prefill_chunk,
        speculation: shape.speculation,
        vision: shape.vision,
        rope_scaling: shape.rope_scaling,
        retained_slots: shape.retained_slots,
        ..ignis_runtime::CudaLeafConfig::default()
    }
}

/// Build the server's real GPU-backed scheduler (GitHub #61 / P1-25):
/// open a second [`ignis_artifact::Reader`] over `artifact_path` (the
/// caller already verified the container through the loader path — this
/// read is for the device-materialization + model-load path, which needs
/// the reader kept alive, not just the frontend set it returned),
/// materialize the text-scope weights on the device, and wrap the loaded
/// model behind the production [`ignis_runtime::CudaLeaf`].
///
/// GitHub #210 (ADR 0030): before the first large allocation, the load lays
/// every reservation out inside the VRAM budget, logs the plan, and refuses
/// the start when it does not fit; after the load it checks the leaf holds
/// what the plan laid out.
#[cfg(feature = "cuda")]
pub fn cuda_scheduler(
    artifact_path: &std::path::Path,
    model_id: String,
    eos: TokenId,
    shape: EngineShape,
) -> Result<(ConcreteScheduler, crate::metrics::LoadReservations), String> {
    use ignis_artifact::{CudaDevice, Reader, bind_model_scope_27b_with, materialize};
    use ignis_runtime::{CudaLeaf, KV_PAGE_TOKENS};

    let reader = Reader::open(artifact_path).map_err(|e| format!("open artifact: {e}"))?;
    // P5-02 (GitHub #150) / GitHub #177: the drafter's and the vision tower's
    // objects are bound and uploaded only when the operator asked for them;
    // otherwise the plan is the text scope's alone, as before.
    let scope = ignis_core::model_load::model_scope(shape.speculation, shape.vision);
    let (plan, handles) =
        bind_model_scope_27b_with(&reader, scope).map_err(|e| format!("bind model scope: {e}"))?;
    // GitHub #210: what is free before this process holds anything, as NVML
    // (and so nvidia-smi and Task Manager) reports it, read before the context
    // exists; the context is a plan line. Not `cudaMemGetInfo`, which on
    // Windows ignores the caller's context and reads ~400 MiB optimistic.
    let (free_at_start_bytes, _) = CudaDevice::nvml_memory(0)
        .map_err(|e| format!("VRAM budget: free memory is unreadable: {e}"))?;
    let mut device = CudaDevice::create(0).map_err(|e| format!("CUDA device: {e}"))?;
    let geometry = ignis_core::KvGeometry::qwen38_27b();
    let planning = leaf_config_for_shape(shape, 0);
    let reservations = planning.plan_reservations(&reader, &plan, &handles)?;
    // Asked once where its error can surface: the plan below reads a pool the
    // leaf refuses to plan as one that never fits.
    planning.kv_pool_arena_bytes(1)?;
    let kv_arena_bytes = |pages: u32| planning.kv_pool_arena_bytes(pages).unwrap_or(u64::MAX);
    let vram = ignis_core::plan_vram(&ignis_core::VramRequest {
        mode: shape.vram,
        free_at_start_bytes,
        lines: vram_lines(plan.device_capacity_bytes, reservations.reserved),
        kv_format: shape.kv_format,
        kv_geometry: geometry,
        max_context_tokens: shape.max_context,
        retained_slots: shape.retained_slots,
        kv_pool_bytes: shape.kv_pool_bytes,
        // GitHub #243: only so a refusal names the knob the operator set.
        embedding_pool_named: shape
            .vision
            .is_some_and(|v| v.requested_pool_bytes() != ignis_core::vision::DEFAULT_EMBEDDING_POOL_BYTES),
        kv_arena_bytes: &kv_arena_bytes,
        // Windows WDDM pages an oversubscribed device allocation to system
        // RAM; elsewhere it fails.
        can_page: cfg!(windows),
    })
    .map_err(|e| e.to_string())?;
    log_vram_plan(&vram);

    let artifact = materialize(&reader, &plan, &mut device, None)
        .map_err(|e| format!("materialize weights: {e}"))?;

    let leaf_config = leaf_config_for_shape(shape, vram.kv_budget_bytes(shape.kv_format, geometry));

    // Match the scheduler's KV admission accounting to the pool the leaf
    // actually built. Both sides read the page count from the *same*
    // `kv_pool_plan` call — one byte budget, one format, one derived page
    // count — so growing the configured context (or the pool, or changing
    // the format) can never let admission promise capacity the GPU does not
    // have. The slot count is `CudaLeafConfig::default()`'s
    // `N_DECODE_LANES` (8), so the scheduler's own lane count
    // (`SchedulerConfig::default()`'s `max_in_flight`) still matches it.
    let expected_pages = leaf_config
        .kv_pool_plan()
        .map_err(|e| e.to_string())?
        .page_count;

    // GitHub #213: the whole KV-RAM tier, pinned once here and held for the
    // life of the load — the same figure the scheduler's host tier budgets in
    // bytes, so the ledger and the arena describe one region. A size the host
    // cannot page-lock refuses the start rather than surfacing as a spill
    // that quietly never happens.
    let leaf = CudaLeaf::new(device, reader, artifact, handles, leaf_config)
        .with_kv_ram_arena(shape.host_pool_bytes)?;
    let model = Arc::new(Model::load(Arc::new(leaf)).map_err(|e| format!("model load: {e:?}"))?);

    // GitHub #98 (P3-02): do not just trust that formula — ask the leaf
    // what it actually built (`ignis_seq_pool_stats`, surfaced through
    // `RuntimeStats::kv_page_count`/`kv_page_bytes`), refuse to start on a
    // disagreement, and hand the admission machine the real, leaf-verified
    // pool's own page count rather than the formula's guess.
    let stats = model.stats().map_err(|e| format!("runtime stats: {e:?}"))?;
    let kv_pool = ignis_core::kv::verified_kv_pool(
        expected_pages,
        ignis_core::kv::LeafPoolGeometry {
            page_count: stats.kv_page_count,
            page_bytes: stats.kv_page_bytes,
        },
    )
    .map_err(|e| e.to_string())?;
    let capacity_pages = kv_pool.block_count() as u32;

    // GitHub #210: the same for every reservation the plan laid out -- the
    // leaf's buffers, read back, must be the plan's lines, or the plan no
    // longer describes what this process holds.
    let planned = ignis_runtime::ReservedBytes {
        kv_pool: vram.kv_pool_bytes,
        ..reservations.reserved
    };
    if stats.reserved != planned {
        return Err(format!(
            "the load holds {:?} beside its weights, not the {:?} its VRAM plan laid out",
            stats.reserved, planned
        ));
    }
    // What NVML says the load took, beside what the plan said it would
    // (GitHub #210): every line, the retained slots included -- since GitHub
    // #215 serving allocates nothing more. The delta also moves with anything
    // else on the card.
    if let Ok((free_after_load_bytes, _)) = CudaDevice::nvml_memory(0) {
        // hotpath-lint-allow: one line per model load.
        tracing::info!(
            name: "ignis.runtime.vram_loaded",
            allocated_at_load_bytes = vram.total_bytes,
            free_memory_delta_bytes = free_at_start_bytes.saturating_sub(free_after_load_bytes),
            free_after_load_bytes,
            "vram loaded"
        );
    }

    let sched = scheduler(
        scheduler_config_for_shape(model_id, shape, KV_PAGE_TOKENS, capacity_pages),
        model,
        eos,
    );
    // GitHub #216 (ADR 0030 §Observability): the plan was a value this
    // function built, read once and dropped. What it reserved outlives it
    // now, so an operator can read it off `/metrics` instead of off the one
    // log line the load wrote. The retained slot count is the scheduler's
    // own, not the flag's: prompt reuse off hands out none whatever
    // `--retained-slots` said.
    let reserved = crate::metrics::LoadReservations {
        lines: vram.lines,
        budget_bytes: vram.budget_bytes,
        kv_pool_pages: capacity_pages,
        kv_page_bytes: stats.kv_page_bytes,
        kv_ram_arena_bytes: shape.host_pool_bytes,
        retained_slots: sched.retained_slot_count(),
    };
    Ok((sched, reserved))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ignis_core::{DecodeParams, RequestClass, RequestInput, Scheduler};

    #[test]
    fn the_default_engine_shape_is_what_the_config_resolves_with_no_flags() {
        // Two ways to say "the defaults" must not drift: the one `main`
        // uses (`config::resolve`) and the one the GPU tests use
        // (`EngineShape::default`).
        let crate::config::ConfigOutcome::Config(config) =
            crate::config::resolve(&[], |_| None).expect("resolve")
        else {
            panic!("expected a runnable config");
        };
        assert_eq!(EngineShape::from(&config), EngineShape::default());
    }

    /// The whole chain from an operator's flags to a blob's identity, with no
    /// card in it: `Config` -> `EngineShape` -> `CudaLeafConfig` ->
    /// `BlobIdentity` (GitHub #189). Only the first hop is device-free on its
    /// own, which is why `leaf_config_for_shape` exists.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_operator_flags_reach_the_identity_through_the_load_and_nothing_else() {
        let artifact = ignis_core::ArtifactHash::from_bytes([7; 32]);
        const LAYOUT: u32 = 2;
        let identity_of = |config: &crate::config::Config| {
            let kv_pool_bytes = config.kv_pool_bytes.unwrap_or(1 << 30);
            leaf_config_for_shape(EngineShape::from(config), kv_pool_bytes).blob_identity(artifact, LAYOUT)
        };
        let crate::config::ConfigOutcome::Config(base) =
            crate::config::resolve(&[], |_| None).expect("resolve")
        else {
            panic!("expected a runnable config");
        };

        // Every serving knob at once: the bind address, the timeout, the UI,
        // the metrics listener, the API key, the prefill chunk, the context,
        // both pool budgets, the VRAM budget, `--prompt-reuse` and
        // `--retained-slots`. The KV pool is the one that makes this a rule
        // rather than a nicety: unset, it is the rest of a budget derived from
        // the VRAM free at start, so it differs from one start of the same
        // server to the next, and an identity that moved with it would refuse
        // every blob after a reboot.
        let elsewhere = crate::config::Config {
            bind: "0.0.0.0:9999".into(),
            request_timeout_secs: base.request_timeout_secs + 7,
            ui: !base.ui,
            metrics: Some("127.0.0.1:9101".into()),
            api_key: Some(crate::config::ApiKeySetting::Generate),
            prefill_chunk: 512,
            max_context: base.max_context / 2,
            kv_pool_bytes: Some(8 * 1024 * 1024 * 1024),
            vram: ignis_core::VramMode::Explicit {
                budget_bytes: 20 * 1024 * 1024 * 1024,
                allow_oversubscription: true,
            },
            host_pool_bytes: base.host_pool_bytes * 2,
            prompt_reuse: !base.prompt_reuse,
            retained_slots: 3,
            ..base.clone()
        };
        assert_ne!(base, elsewhere, "the two configs really do differ");
        assert_eq!(
            identity_of(&base),
            identity_of(&elsewhere),
            "no serving flag reaches what the engine is loaded as"
        );

        // And the two load options that do, end to end from the flag.
        let bf16 = crate::config::Config {
            kv_format: ignis_core::KvFormat::Bf16,
            ..base.clone()
        };
        let hq = crate::config::Config {
            kv_format: ignis_core::KvFormat::HqE8_2b,
            ..base.clone()
        };
        assert_eq!(
            identity_of(&bf16)
                .accepts(&identity_of(&hq))
                .expect_err("another KV format")
                .field,
            ignis_core::IdentityField::KvFormat
        );
        let drafted = crate::config::Config {
            speculation: Some(
                ignis_core::Speculation::new(ignis_core::SpeculativeBackend::Dflash2, 4).unwrap(),
            ),
            ..base.clone()
        };
        assert_eq!(
            identity_of(&base)
                .accepts(&identity_of(&drafted))
                .expect_err("a drafter this load has not bound")
                .field,
            ignis_core::IdentityField::Drafter
        );
    }

    #[test]
    fn the_serving_flags_never_reach_the_load_the_identity_names() {
        // GitHub #189, ADR 0029, upstream half. A blob's compatibility
        // identity is assembled by `CudaLeafConfig::blob_identity` (which
        // tests which of *its* fields count), and `cuda_scheduler` builds
        // that struct out of nothing but an `EngineShape`. So the operator
        // flags below cannot change a blob's identity for a reason stronger
        // than a rule someone remembers: they do not survive the step into
        // the only value the load is configured from.
        let crate::config::ConfigOutcome::Config(base) =
            crate::config::resolve(&[], |_| None).expect("resolve")
        else {
            panic!("expected a runnable config");
        };
        let elsewhere = crate::config::Config {
            bind: "0.0.0.0:9999".into(),
            request_timeout_secs: base.request_timeout_secs + 7,
            ui: !base.ui,
            metrics: Some("127.0.0.1:9101".into()),
            api_key: Some(crate::config::ApiKeySetting::Generate),
            ..base.clone()
        };
        assert_ne!(base, elsewhere, "the two configs really do differ");
        assert_eq!(
            EngineShape::from(&base),
            EngineShape::from(&elsewhere),
            "the bind address, the timeout, the UI, the metrics listener and \
             the API key are not part of what the engine is loaded as"
        );
    }

    #[test]
    fn the_operator_prefill_chunk_reaches_the_scheduler_config() {
        let shape = EngineShape {
            prefill_chunk: 512,
            max_context: 65_536,
            kv_format: ignis_core::KvFormat::Bf16,
            kv_pool_bytes: Some(8 * 1024 * 1024 * 1024),
            vram: EngineShape::default().vram,
            host_pool_bytes: crate::config::DEFAULT_HOST_POOL_BYTES,
            prompt_reuse: true,
            retained_slots: 5,
            retained_interactive_ttl: std::time::Duration::from_secs(60),
            speculation: None,
            vision: None,
            rope_scaling: ignis_core::RopeScaling::NONE,
        };

        let config = scheduler_config_for_shape("test-model".into(), shape, 64, 32_768);

        assert_eq!(config.serving_chunk_tokens, 512);
        // GitHub #186: the reuse knobs reach the scheduler too -- and the
        // retained slots as the count the leaf's pool reserves (GitHub #215).
        assert!(config.prompt_reuse);
        assert_eq!(config.retained_slots, 5);
        // GitHub #190: and so does the Interactive TTL.
        assert_eq!(config.retained_interactive_ttl, std::time::Duration::from_secs(60));
    }

    #[test]
    fn prompt_reuse_off_reaches_the_scheduler_config() {
        let crate::config::ConfigOutcome::Config(off) =
            crate::config::resolve(&["--prompt-reuse".to_owned(), "off".to_owned()], |_| None)
                .expect("resolve")
        else {
            panic!("expected a runnable config");
        };
        let shape = EngineShape::from(&off);
        let config = scheduler_config_for_shape("test-model".into(), shape, 64, 32_768);
        assert!(!config.prompt_reuse);
        assert_eq!(config.retained_slots, 0, "reuse off reserves no retained slot");
    }

    // ── the VRAM plan (GitHub #210) ──────────────────────────────────────

    #[cfg(feature = "cuda")]
    #[test]
    fn the_leaf_reserves_the_retained_slots_the_scheduler_hands_out() {
        // GitHub #215: one count, from one flag, reaches both -- a slot the
        // scheduler names is one the pool holds.
        let shape = EngineShape {
            retained_slots: 5,
            ..EngineShape::default()
        };
        assert_eq!(leaf_config_for_shape(shape, 1 << 30).retained_slots, 5);
        assert_eq!(scheduler_config_for_shape("m".into(), shape, 64, 1024).retained_slots, 5);
    }

    #[test]
    fn the_plan_lines_carry_every_reservation_the_leaf_planned() {
        let reserved = ignis_runtime::ReservedBytes {
            workspace: 1,
            media_embedding: 3,
            sampling: 4,
            decode_graph: 5,
            verify_round: 6,
            drafter_round: 7,
            lane_state: 8,
            retained_slots: 10,
            kv_pool: 1 << 40,
        };
        let lines = vram_lines(100, reserved);
        assert_eq!(
            lines.entries().map(|(_, bytes)| bytes),
            [
                100,
                ignis_runtime::CUDA_CONTEXT_BYTES,
                1,
                3,
                4,
                5,
                6,
                7,
                8,
                10,
                ignis_runtime::LOAD_RESIDUAL_BYTES,
            ],
            "every line in plan order, and never the KV pool the plan sizes itself"
        );
    }

    fn captured_vram_plan(plan: &ignis_core::VramPlan) -> Vec<serde_json::Value> {
        use tracing_subscriber::layer::SubscriberExt;
        let sink = std::sync::Arc::new(ignis_logging::MemorySink::new());
        let subscriber =
            tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone()));
        tracing::subscriber::with_default(subscriber, || log_vram_plan(plan));
        sink.lines()
            .iter()
            .map(|line| serde_json::from_str(line).expect("valid json"))
            .collect()
    }

    fn plan_for(mode: ignis_core::VramMode, free: u64) -> ignis_core::VramPlan {
        let reserved = ignis_runtime::ReservedBytes {
            workspace: 1 << 30,
            lane_state: 1 << 30,
            retained_slots: u64::from(crate::config::DEFAULT_RETAINED_SLOTS) * 238_823_424,
            ..Default::default()
        };
        let page_bytes = ignis_core::KvFormat::HqE8_2b.page_bytes(ignis_core::KvGeometry::qwen38_27b());
        let arena = move |pages: u32| 4096 + u64::from(pages) * page_bytes;
        ignis_core::plan_vram(&ignis_core::VramRequest {
            mode,
            free_at_start_bytes: free,
            lines: vram_lines(17 << 30, reserved),
            kv_format: ignis_core::KvFormat::HqE8_2b,
            kv_geometry: ignis_core::KvGeometry::qwen38_27b(),
            max_context_tokens: 262_144,
            retained_slots: crate::config::DEFAULT_RETAINED_SLOTS,
            kv_pool_bytes: None,
            embedding_pool_named: false,
            kv_arena_bytes: &arena,
            can_page: true,
        })
        .expect("fits")
    }

    #[test]
    fn one_vram_plan_event_carries_every_line_the_mode_and_the_headroom() {
        let plan = plan_for(ignis_core::VramMode::Derived { headroom_bytes: 1 << 30 }, 30 << 30);
        let records = captured_vram_plan(&plan);
        assert_eq!(records.len(), 1, "{records:?}");
        let event = &records[0];
        assert_eq!(event["event_name"], "ignis.runtime.vram_plan", "{event}");
        let field = |name: &str| {
            event
                .get(name)
                .or_else(|| event["attributes"].get(name))
                .cloned()
                .unwrap_or_else(|| panic!("no `{name}` in {event}"))
        };
        assert_eq!(field("mode"), "derived");
        assert_eq!(field("free_at_start_bytes"), 30u64 << 30);
        assert_eq!(field("headroom_bytes"), 1u64 << 30);
        assert_eq!(field("budget_bytes"), plan.budget_bytes);
        for (name, bytes) in plan.lines.entries() {
            assert_eq!(field(&format!("{name}_bytes")), bytes, "{name}");
        }
        assert_eq!(field("kv_pool_bytes"), plan.kv_pool_bytes);
        assert_eq!(field("total_bytes"), plan.total_bytes);
        assert_eq!(
            field("allocated_at_load_bytes"),
            plan.total_bytes,
            "what Task Manager shows right after load: every line, the retained slots included"
        );
        assert_eq!(field("oversubscribed"), false);
    }

    #[test]
    fn an_oversubscribed_explicit_plan_logs_the_budget_and_a_warning() {
        let mode = ignis_core::VramMode::Explicit {
            budget_bytes: 31 << 30,
            allow_oversubscription: true,
        };
        let plan = plan_for(mode, 30 << 30);
        let records = captured_vram_plan(&plan);
        assert_eq!(records.len(), 2, "{records:?}");
        let event = &records[0];
        let field = |name: &str| event.get(name).or_else(|| event["attributes"].get(name)).cloned();
        assert_eq!(field("mode").expect("mode"), "explicit");
        assert_eq!(field("budget_bytes").expect("budget"), 31u64 << 30);
        assert_eq!(field("oversubscribed").expect("oversubscribed"), true);
        assert!(field("headroom_bytes").is_none(), "{event}");
        assert_eq!(records[1]["event_name"], "ignis.runtime.vram_oversubscribed");
        assert_eq!(records[1]["severity_text"], "WARN");
    }

    struct StubLeaf;

    impl StepLeaf for StubLeaf {
        type Model = ();
        type Sequence = ();
        type Prefix = ();
        type SnapshotBuf = Vec<u8>;
        type Media = ();
        type Checkpoint = ();

        fn load_model(&self) -> Result<Self::Model, i32> {
            Ok(())
        }
        fn release_model(&self, _model: Self::Model) {}
        fn stats(&self, _model: &Self::Model) -> Result<ignis_runtime::RuntimeStats, i32> {
            Ok(ignis_runtime::RuntimeStats::default())
        }
        fn allocate_sequence(
            &self,
            _model: &Self::Model,
            _context_tokens: u32,
        ) -> Result<Self::Sequence, i32> {
            Ok(())
        }
        fn release_sequence(&self, _model: &Self::Model, _sequence: Self::Sequence) {}
        fn allocate_sequence_shared(
            &self,
            _model: &Self::Model,
            _context_tokens: u32,
            _prefix: &Self::Prefix,
        ) -> Result<Self::Sequence, i32> {
            Ok(())
        }
        fn publish_prefix(
            &self,
            _model: &Self::Model,
            _sequence: &mut Self::Sequence,
            _prefix_tokens: u32,
            _retained_slot: u32,
        ) -> Result<Self::Prefix, i32> {
            Ok(())
        }
        fn release_prefix(&self, _model: &Self::Model, _prefix: Self::Prefix) {}
        fn vocab(&self, _model: &Self::Model) -> u32 {
            8
        }
        fn prefill(
            &self,
            _model: &Self::Model,
            _sequence: &mut Self::Sequence,
            _tokens: &[TokenId],
            _start_position: u32,
            _params: DecodeParams,
            _permitted: &[TokenId],
            _out_logits: Option<&mut [f32]>,
        ) -> Result<f32, i32> {
            Ok(0.0)
        }
        fn decode(
            &self,
            _model: &Self::Model,
            sequences: &mut [&mut Self::Sequence],
            _lanes: &[ignis_runtime::DecodeLane<'_>],
        ) -> Result<Vec<ignis_runtime::LaneRun>, i32> {
            Ok(vec![ignis_runtime::LaneRun::token(7); sequences.len()])
        }
        fn alloc_snapshot_buf(&self, bytes: u64) -> Result<Self::SnapshotBuf, i32> {
            Ok(vec![0u8; bytes as usize])
        }
        fn snapshot_bytes(&self, _model: &Self::Model, _sequence: &Self::Sequence) -> Result<u64, i32> {
            Ok(0)
        }
        fn snapshot_into(
            &self,
            _model: &Self::Model,
            _sequence: &Self::Sequence,
            _dst: &mut [u8],
        ) -> Result<(), i32> {
            Ok(())
        }
        fn restore_sequence(
            &self,
            _model: &Self::Model,
            _sequence: &mut Self::Sequence,
            _src: &[u8],
        ) -> Result<(), i32> {
            Ok(())
        }
    }

    #[test]
    fn scheduler_construction_uses_the_runtime_adapter() {
        let model = Arc::new(Model::load(Arc::new(StubLeaf)).unwrap());
        let mut scheduler = scheduler(
            SchedulerConfig {
                model: "stub".into(),
                ..SchedulerConfig::default()
            },
            model,
            99,
        );
        scheduler
            .submit(
                RequestInput {
                    decision: None,
                    constrained: None,
                    multimodal: None,
                    opener_tokens: None,
                    user_turn_tokens: None,
                    system_block_tokens: None,
                    model: "stub".into(),
                    tokens: vec![1],
                    params: DecodeParams {
                        max_tokens: Some(1),
                        ..DecodeParams::default()
                    },
                },
                RequestClass::Interactive,
            )
            .unwrap();

        assert!(
            scheduler
                .advance()
                .iter()
                .any(|event| matches!(event, ignis_core::SchedEvent::Done { tokens: 1, .. }))
        );
    }
}
