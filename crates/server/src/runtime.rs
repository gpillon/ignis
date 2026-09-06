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

/// The leaf's fixed paged-KV page size, in tokens
/// (`kernel/vendor/src/core/paged_kv_cache.h`'s `kPagedKVPageSize`) — the
/// step ABI does not report it, so the scheduler's own KV-page accounting
/// (`SchedulerConfig::kv_page_tokens`) must be kept in sync with it by hand
/// for the real backend (the mock backend is page-size-agnostic and keeps
/// the smaller default).
#[cfg(feature = "cuda")]
const LEAF_KV_PAGE_TOKENS: u32 = 64;

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
) -> Result<ConcreteScheduler, String> {
    use ignis_artifact::{bind_text_scope_27b, materialize, CudaDevice, Reader};
    use ignis_runtime::{CudaLeaf, CudaLeafConfig};

    let reader = Reader::open(artifact_path).map_err(|e| format!("open artifact: {e}"))?;
    let (plan, handles) =
        bind_text_scope_27b(&reader).map_err(|e| format!("bind text scope: {e}"))?;
    let mut device = CudaDevice::create(0).map_err(|e| format!("CUDA device: {e}"))?;
    let artifact = materialize(&reader, &plan, &mut device, None)
        .map_err(|e| format!("materialize weights: {e}"))?;

    let leaf_config = CudaLeafConfig::default();
    let max_sequence_tokens = leaf_config.max_context_tokens;
    let slot_count = leaf_config.slot_count;
    let leaf = CudaLeaf::new(device, reader, artifact, handles, leaf_config);
    let model = Arc::new(
        Model::load(Arc::new(leaf)).map_err(|e| format!("model load: {e:?}"))?,
    );

    // Match the scheduler's KV admission accounting to the pool the leaf
    // actually built (`slot_count` sequences, each up to
    // `max_sequence_tokens`) so admission never promises more than the
    // pool's own budget. `slot_count` is `CudaLeafConfig::default()`'s
    // `N_DECODE_LANES` (8), so the scheduler's own lane count
    // (`SchedulerConfig::default()`'s `max_in_flight`) still matches it.
    let pages_per_sequence = max_sequence_tokens / LEAF_KV_PAGE_TOKENS;
    Ok(scheduler(
        SchedulerConfig {
            model: model_id,
            kv_page_tokens: LEAF_KV_PAGE_TOKENS,
            max_sequence_tokens,
            kv_capacity_pages: slot_count * pages_per_sequence,
            host_capacity_pages: slot_count * pages_per_sequence,
            ..SchedulerConfig::default()
        },
        model,
        eos,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ignis_core::{DecodeParams, RequestClass, RequestInput, Scheduler};

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
        ) -> Result<(), i32> {
            Ok(())
        }
        fn decode(
            &self,
            _model: &Self::Model,
            sequences: &mut [&mut Self::Sequence],
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
