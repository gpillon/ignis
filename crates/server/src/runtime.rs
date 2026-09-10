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
    /// The paged-KV pool budget, in sequence-tokens: derived from
    /// `max_context` ([`ignis_runtime::kv_pool_tokens_for`]), not an
    /// independent flag.
    pub kv_pool_tokens: u32,
}

impl Default for EngineShape {
    /// The configured defaults, taken from [`ignis_runtime`] rather than
    /// restated here — one source of truth for what `ignis-server` runs
    /// with when the operator passes no flags.
    fn default() -> Self {
        Self {
            prefill_chunk: ignis_runtime::DEFAULT_PREFILL_CHUNK,
            max_context: ignis_runtime::DEFAULT_MAX_CONTEXT,
            kv_pool_tokens: ignis_runtime::kv_pool_tokens_for(ignis_runtime::DEFAULT_MAX_CONTEXT),
        }
    }
}

impl From<&crate::config::Config> for EngineShape {
    fn from(config: &crate::config::Config) -> Self {
        Self {
            prefill_chunk: config.prefill_chunk,
            max_context: config.max_context,
            kv_pool_tokens: config.kv_pool_tokens,
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
        host_capacity_pages: capacity_pages,
        serving_chunk_tokens: shape.prefill_chunk,
        ..SchedulerConfig::default()
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
    use ignis_artifact::{CudaDevice, Reader, bind_text_scope_27b, materialize};
    use ignis_runtime::{CudaLeaf, CudaLeafConfig, KV_PAGE_TOKENS, kv_pool_pages};

    let reader = Reader::open(artifact_path).map_err(|e| format!("open artifact: {e}"))?;
    let (plan, handles) =
        bind_text_scope_27b(&reader).map_err(|e| format!("bind text scope: {e}"))?;
    let mut device = CudaDevice::create(0).map_err(|e| format!("CUDA device: {e}"))?;
    let artifact = materialize(&reader, &plan, &mut device, None)
        .map_err(|e| format!("materialize weights: {e}"))?;

    let leaf_config = CudaLeafConfig {
        max_context_tokens: shape.max_context,
        kv_pool_tokens: shape.kv_pool_tokens,
        prefill_chunk_tokens: shape.prefill_chunk,
        ..CudaLeafConfig::default()
    };
    let kv_pool_tokens = leaf_config.kv_pool_tokens;
    let leaf = CudaLeaf::new(device, reader, artifact, handles, leaf_config);
    let model = Arc::new(Model::load(Arc::new(leaf)).map_err(|e| format!("model load: {e:?}"))?);

    // Match the scheduler's KV admission accounting to the pool the leaf
    // actually built. Both sides derive the page count from
    // `kv_pool_tokens` through the same `kv_pool_pages`, so growing the
    // configured context (or the pool) can never let admission promise
    // capacity the GPU does not have. The slot count is
    // `CudaLeafConfig::default()`'s `N_DECODE_LANES` (8), so the
    // scheduler's own lane count (`SchedulerConfig::default()`'s
    // `max_in_flight`) still matches it.
    let expected_pages = kv_pool_pages(kv_pool_tokens);

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

    Ok(scheduler(
        scheduler_config_for_shape(model_id, shape, KV_PAGE_TOKENS, capacity_pages),
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

    #[test]
    fn the_operator_prefill_chunk_reaches_the_scheduler_config() {
        let shape = EngineShape {
            prefill_chunk: 512,
            max_context: 65_536,
            kv_pool_tokens: 524_288,
        };

        let config = scheduler_config_for_shape("test-model".into(), shape, 64, 32_768);

        assert_eq!(config.serving_chunk_tokens, 512);
    }

    struct StubLeaf;

    impl StepLeaf for StubLeaf {
        type Model = ();
        type Sequence = ();

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
            _params: &[DecodeParams],
        ) -> Result<Vec<TokenId>, i32> {
            Ok(vec![7; sequences.len()])
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
