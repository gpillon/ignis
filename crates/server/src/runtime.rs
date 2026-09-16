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
    /// The paged-KV pool budget, in bytes (`--kv-pool-bytes`, or
    /// [`ignis_runtime::auto_kv_pool_bytes`] for the resolved format and
    /// context when the operator names none).
    pub kv_pool_bytes: u64,
    /// The KV-RAM host tier's budget, in bytes (`--kv-host-pool-bytes`,
    /// P4-07 GitHub #125): pinned host memory for evicted (suspended)
    /// request snapshots, independent of the GPU-resident pool above.
    pub host_pool_bytes: u64,
    /// Cross-request state reuse (`--prompt-reuse`, GitHub #186, ADR 0029).
    pub prompt_reuse: bool,
    /// The retained checkpoint pool's device budget, in bytes
    /// (`--retained-pool-bytes`, GitHub #186). `None` derives it from the
    /// VRAM left once the model and its pools have landed, which only the
    /// loaded leaf can report.
    pub retained_pool_bytes: Option<u64>,
    /// Speculative decoding (`--spec`/`--draft-tokens`, P5-02 GitHub #150):
    /// `None` binds nothing of the drafter.
    pub speculation: Option<ignis_core::Speculation>,
    /// Vision (`--vision`/`--vision-max-tokens`, GitHub #177): `None` binds
    /// and reserves nothing of the vision tower.
    pub vision: Option<ignis_core::Vision>,
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
            kv_pool_bytes: ignis_runtime::auto_kv_pool_bytes(
                ignis_core::KvFormat::default(),
                ignis_runtime::DEFAULT_MAX_CONTEXT,
            ),
            host_pool_bytes: crate::config::DEFAULT_HOST_POOL_BYTES,
            prompt_reuse: crate::config::DEFAULT_PROMPT_REUSE,
            retained_pool_bytes: None,
            speculation: None,
            vision: None,
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
            host_pool_bytes: config.host_pool_bytes,
            prompt_reuse: config.prompt_reuse,
            retained_pool_bytes: config.retained_pool_bytes,
            speculation: config.speculation,
            vision: config.vision,
        }
    }
}

#[cfg(any(feature = "cuda", test))]
fn scheduler_config_for_shape(
    model: String,
    shape: EngineShape,
    kv_page_tokens: u32,
    capacity_pages: u32,
    retained_pool_bytes: u64,
) -> SchedulerConfig {
    SchedulerConfig {
        model,
        kv_page_tokens,
        max_sequence_tokens: shape.max_context,
        kv_capacity_pages: capacity_pages,
        host_capacity_bytes: shape.host_pool_bytes,
        serving_chunk_tokens: shape.prefill_chunk,
        prompt_reuse: shape.prompt_reuse,
        retained_pool_bytes,
        ..SchedulerConfig::default()
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
#[cfg(feature = "cuda")]
fn leaf_config_for_shape(shape: EngineShape) -> ignis_runtime::CudaLeafConfig {
    ignis_runtime::CudaLeafConfig {
        max_context_tokens: shape.max_context,
        kv_format: shape.kv_format,
        kv_pool_bytes: shape.kv_pool_bytes,
        prefill_chunk_tokens: shape.prefill_chunk,
        speculation: shape.speculation,
        vision: shape.vision,
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
#[cfg(feature = "cuda")]
pub fn cuda_scheduler(
    artifact_path: &std::path::Path,
    model_id: String,
    eos: TokenId,
    shape: EngineShape,
) -> Result<ConcreteScheduler, String> {
    use ignis_artifact::{CudaDevice, Reader, bind_model_scope_27b_with, materialize};
    use ignis_runtime::{CudaLeaf, KV_PAGE_TOKENS};

    let reader = Reader::open(artifact_path).map_err(|e| format!("open artifact: {e}"))?;
    // P5-02 (GitHub #150) / GitHub #177: the drafter's and the vision tower's
    // objects are bound and uploaded only when the operator asked for them;
    // otherwise the plan is the text scope's alone, as before.
    let scope = ignis_core::model_load::model_scope(shape.speculation, shape.vision);
    let (plan, handles) =
        bind_model_scope_27b_with(&reader, scope).map_err(|e| format!("bind model scope: {e}"))?;
    let mut device = CudaDevice::create(0).map_err(|e| format!("CUDA device: {e}"))?;
    let artifact = materialize(&reader, &plan, &mut device, None)
        .map_err(|e| format!("materialize weights: {e}"))?;

    let leaf_config = leaf_config_for_shape(shape);

    // Match the scheduler's KV admission accounting to the pool the leaf
    // actually built. Both sides read the page count from the *same*
    // `kv_pool_plan` call — one byte budget, one format, one derived page
    // count — so growing the configured context (or the pool, or changing
    // the format) can never let admission promise capacity the GPU does not
    // have. The slot count is `CudaLeafConfig::default()`'s
    // `N_DECODE_LANES` (8), so the scheduler's own lane count
    // (`SchedulerConfig::default()`'s `max_in_flight`) still matches it.
    //
    // Planned before the load so a budget too small for the configured
    // context is refused here, naming the budget, the format and the
    // capacity it bought, rather than after a ~19 GB weight upload.
    let expected_pages = leaf_config
        .kv_pool_plan()
        .map_err(|e| e.to_string())?
        .page_count;

    let leaf = CudaLeaf::new(device, reader, artifact, handles, leaf_config);
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

    // GitHub #186 (ADR 0029): the retained pool's budget. The operator's own
    // number wins; absent one it is derived from what the device says is
    // *actually* free now that the weights, the KV pool and every other
    // reservation are down — measured, not guessed from `total - weights`,
    // which is the guess that OOM'd when the KV pool was sized that way
    // (`cuda_leaf.rs`'s module doc).
    let retained_pool_bytes = match (shape.prompt_reuse, shape.retained_pool_bytes) {
        (false, _) => 0,
        (true, Some(bytes)) => bytes,
        (true, None) => ignis_core::auto_retained_pool_bytes(stats.free_vram_bytes),
    };
    // The startup capacity report for the retained tier, beside
    // `ignis.runtime.kv_pool`: what was free, what the budget is, and
    // whether the operator chose it. Read the budget off this line rather
    // than computing it from a flag.
    // hotpath-lint-allow: one line per model load.
    tracing::info!(
        name: "ignis.runtime.retained_pool",
        prompt_reuse = shape.prompt_reuse,
        budget_bytes = retained_pool_bytes,
        derived = shape.retained_pool_bytes.is_none(),
        free_vram_bytes = stats.free_vram_bytes,
        "retained pool"
    );

    Ok(scheduler(
        scheduler_config_for_shape(
            model_id,
            shape,
            KV_PAGE_TOKENS,
            capacity_pages,
            retained_pool_bytes,
        ),
        model,
        eos,
    ))
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
            leaf_config_for_shape(EngineShape::from(config)).blob_identity(artifact, LAYOUT)
        };
        let crate::config::ConfigOutcome::Config(base) =
            crate::config::resolve(&[], |_| None).expect("resolve")
        else {
            panic!("expected a runnable config");
        };

        // Every serving knob at once: the bind address, the timeout, the UI,
        // the metrics listener, the API key, the prefill chunk, the context,
        // both pool budgets, `--prompt-reuse` and `--retained-pool-bytes`.
        // `retained_pool_bytes` is the one that makes this a rule rather than
        // a nicety: unset, its value is derived from the VRAM left after load,
        // so it differs from one start of the same server to the next, and an
        // identity that moved with it would refuse every blob after a reboot.
        let elsewhere = crate::config::Config {
            bind: "0.0.0.0:9999".into(),
            request_timeout_secs: base.request_timeout_secs + 7,
            ui: !base.ui,
            metrics: Some("127.0.0.1:9101".into()),
            api_key: Some(crate::config::ApiKeySetting::Generate),
            prefill_chunk: 512,
            max_context: base.max_context / 2,
            kv_pool_bytes: base.kv_pool_bytes * 2,
            host_pool_bytes: base.host_pool_bytes * 2,
            prompt_reuse: !base.prompt_reuse,
            retained_pool_bytes: Some(3 * 1024 * 1024 * 1024),
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
            kv_pool_bytes: 8 * 1024 * 1024 * 1024,
            host_pool_bytes: crate::config::DEFAULT_HOST_POOL_BYTES,
            prompt_reuse: true,
            retained_pool_bytes: None,
            speculation: None,
            vision: None,
        };

        let config = scheduler_config_for_shape("test-model".into(), shape, 64, 32_768, 4_096);

        assert_eq!(config.serving_chunk_tokens, 512);
        // GitHub #186: the reuse knobs reach the scheduler too — the budget
        // as the number the caller resolved (the operator's, or the one
        // derived from free VRAM), never re-derived here.
        assert!(config.prompt_reuse);
        assert_eq!(config.retained_pool_bytes, 4_096);
    }

    #[test]
    fn prompt_reuse_off_reaches_the_scheduler_config() {
        let shape = EngineShape {
            prompt_reuse: false,
            ..EngineShape::default()
        };
        let config = scheduler_config_for_shape("test-model".into(), shape, 64, 32_768, 0);
        assert!(!config.prompt_reuse);
        assert_eq!(config.retained_pool_bytes, 0);
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
        ) -> Result<Self::Prefix, i32> {
            Ok(())
        }
        fn release_prefix(&self, _model: &Self::Model, _prefix: Self::Prefix) {}
        fn prefill(
            &self,
            _model: &Self::Model,
            _sequence: &mut Self::Sequence,
            _tokens: &[TokenId],
            _start_position: u32,
            _params: DecodeParams,
        ) -> Result<(), i32> {
            Ok(())
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
                    multimodal: None,
                    opener_tokens: None,
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
