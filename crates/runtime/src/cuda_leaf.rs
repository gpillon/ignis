//! The production step-ABI leaf (GitHub #61 / P1-25): a real GPU-backed
//! [`StepLeaf`] over the vendored full program (P1-23, `ignis_core::step`)
//! — the first backend `ignis-server` drives beyond `MockCompute`.
//!
//! [`CudaLeaf`] owns everything `ignis_model_load` needs to (re)build a
//! model handle on demand — the artifact reader, its device-materialized
//! weights, the bound-tensor handles — plus the CUDA device context they
//! were materialized on (kept alive: the device memory is not valid once
//! the context is destroyed). [`StepLeaf::load_model`] sizes the
//! sequence-state pool deterministically from [`CudaLeafConfig`]
//! (`slot_count` sequences of up to `max_context_tokens` each) rather than
//! from the VRAM left over after the weights land: an earlier attempt at
//! the latter (`Device::free_bytes` minus a fixed reserve, handed straight
//! to `ignis_paged_kv_page_budget`) OOM'd on the real 27B artifact — the
//! model's real post-load footprint (weights + the program's own
//! workspace) leaves far less headroom than a naive `total - weights`
//! guess, and a wrong guess fails as a hard `cudaMalloc` error, not a
//! graceful degradation. Auto-sizing from real free VRAM (the target
//! envelope README describes) needs the leaf to report its own workspace
//! footprint first — later work, not needed for a correct G1 server.

#![cfg(feature = "cuda")]

use ignis_artifact::{CudaDevice, MaterializedArtifact, ObjectHandle, Reader};
use ignis_core::model_load::{self, Model as CoreModel};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step;
use ignis_core::{DecodeParams, ModelConfig, N_DECODE_LANES, TokenId};

use crate::{RuntimeStats, StepLeaf};

/// Sizing knobs for the leaf's sequence-state pool and program scratch.
#[derive(Debug, Clone, Copy)]
pub struct CudaLeafConfig {
    /// The largest single sequence's KV reservation, in tokens (mirrors
    /// `ignis_core::SchedulerConfig::max_sequence_tokens`). Also the bound
    /// `ignis_model_load` sizes the GQA attention workspace reservation
    /// for (P2-01, GitHub #83) — must not be raised without also rebuilding
    /// the model handle.
    pub max_context_tokens: u32,
    /// The paged-KV pool budget, in sequence-tokens: the pool every live
    /// sequence draws its pages from. Sized independently of
    /// `slot_count * max_context_tokens` — see [`CudaLeafConfig::default`].
    pub kv_pool_tokens: u32,
    /// Max concurrent sequences (mirrors [`N_DECODE_LANES`]).
    pub slot_count: u32,
    /// The prefill chunk width, in tokens: how wide a span the program's
    /// prefill scratch must serve (`--prefill-chunk`, GitHub #87). A
    /// nonzero multiple of 128, validated by the server's config module
    /// before any loader work starts. `ignis_model_load` reserves the
    /// program scratch for a chunk of this width at load time (P2-01,
    /// GitHub #83), and the chunk loop that spends it landed with GitHub
    /// #84: prefill traverses a span one chunk at a time, synchronizing
    /// once per chunk, not once per token.
    pub prefill_chunk_tokens: u32,
}

impl Default for CudaLeafConfig {
    fn default() -> Self {
        Self {
            // The same defaults `ignis_server::config` falls back to
            // (`crate::{DEFAULT_MAX_CONTEXT, kv_pool_tokens_for,
            // DEFAULT_PREFILL_CHUNK}`, defined once alongside this module
            // rather than restated here): `cuda_scheduler` always
            // overrides these three fields from the operator's resolved
            // `EngineShape`, so this default only matters to a caller that
            // builds a leaf directly (the GPU layer/program tests) rather
            // than through the server.
            max_context_tokens: crate::DEFAULT_MAX_CONTEXT,
            kv_pool_tokens: crate::kv_pool_tokens_for(crate::DEFAULT_MAX_CONTEXT),
            slot_count: N_DECODE_LANES as u32,
            prefill_chunk_tokens: crate::DEFAULT_PREFILL_CHUNK,
        }
    }
}

/// The production leaf: owns the CUDA device context, the device-
/// materialized artifact, and the bound-tensor handles `ignis_model_load`
/// reads on every (re)load.
pub struct CudaLeaf {
    // Never read directly (no more live free-VRAM query — see the module
    // doc): held purely so the device context outlives `artifact` and
    // every loaded model, since dropping it would invalidate their device
    // memory.
    #[allow(dead_code)]
    device: CudaDevice,
    reader: Reader,
    artifact: MaterializedArtifact,
    handles: Vec<ObjectHandle>,
    config: CudaLeafConfig,
}

impl CudaLeaf {
    /// Build a leaf over an already device-materialized artifact
    /// (`ignis_artifact::materialize` against `bind_text_scope_27b`'s
    /// plan). The caller does the one-time device setup + weight upload;
    /// the leaf owns all of it from here so it can (re)load the model and
    /// size the sequence pool.
    pub fn new(
        device: CudaDevice,
        reader: Reader,
        artifact: MaterializedArtifact,
        handles: Vec<ObjectHandle>,
        config: CudaLeafConfig,
    ) -> Self {
        Self {
            device,
            reader,
            artifact,
            handles,
            config,
        }
    }
}

impl Drop for CudaLeaf {
    fn drop(&mut self) {
        // `artifact`'s weight arena has no `Drop` of its own — releasing it
        // needs the `Device` that produced it, so it must happen explicitly
        // here, before field auto-drop runs `device`'s own `Drop` (which
        // only tears down its load stream/event, never this allocation).
        // Without this, every load leaks its ~19 GB arena for the rest of
        // the process (the fault this fixes: multiple GPU tests in one
        // process accumulate one arena per `harness()` call).
        let _ = self.artifact.release_arena(&mut self.device);
    }
}

/// The leaf's fixed paged-KV page size, in tokens
/// (`kernel/vendor/src/core/paged_kv_cache.h`'s `kPagedKVPageSize`).
pub const KV_PAGE_TOKENS: u32 = 64;

/// The page count a `kv_pool_tokens` budget buys, at the leaf's fixed page
/// size. The scheduler's admission accounting
/// (`ignis_server::runtime::cuda_scheduler`) is derived from this same
/// function, so the two can never drift.
pub fn kv_pool_pages(kv_pool_tokens: u32) -> u32 {
    kv_pool_tokens.div_ceil(KV_PAGE_TOKENS)
}

/// The leaf's model handle: the loaded weights plus the sequence-state
/// pool sized for them. Both travel together — the step ABI takes the
/// pool and the model as separate parameters on every prefill/decode call.
pub struct CudaModel {
    model: CoreModel,
    pool: SeqPool,
}

// These wrap raw FFI pointers with no synchronization of their own. That is
// sound here because every `Compute` call the scheduler makes runs under
// `ignis-server::engine::Engine`'s single `Mutex` — never concurrently —
// matching `ignis_core::seq::SeqPool`'s own documented single-thread-driver
// contract.
unsafe impl Send for CudaModel {}
unsafe impl Sync for CudaModel {}
unsafe impl Send for CudaLeaf {}
unsafe impl Sync for CudaLeaf {}

/// Log `message` (with `context`) and collapse it to the generic leaf
/// error code `StepLeaf` expects — `ignis_core`'s step/seq/model_load
/// wrappers already discarded the raw C return code in favor of a
/// descriptive string (`ignis_*_last_error`), so there is no real code
/// left to preserve here.
///
/// Goes through `tracing::error!` (GitHub #80), not a raw `eprintln!`: this
/// fires from `prefill`/`decode` on the leaf's own error path, and a raw
/// `eprintln!` is exactly the uncontrolled blocking console I/O spec §27
/// forbids on that path — `ignis-logging`'s bounded priority queue decouples
/// it from physical I/O the same as every other ERROR site. This is a
/// failure path only (never once per token/layer/kernel in the success
/// case, i.e. not the frequency spec §26 constrains), so it stays an ERROR
/// rather than needing further demotion.
fn leaf_error(context: &str, message: String) -> i32 {
    // hotpath-lint-allow: failure-only path (prefill/decode error return, see the doc comment above), reviewed exception (GitHub #80).
    tracing::error!(name: "ignis.runtime.leaf_error", context, error = %message, "step leaf error");
    -1
}

impl StepLeaf for CudaLeaf {
    type Model = CudaModel;
    type Sequence = Seq<'static>;

    fn load_model(&self) -> Result<Self::Model, i32> {
        let model = model_load::load_qwen38_27b(
            &self.reader,
            &self.artifact,
            &self.handles,
            self.config.prefill_chunk_tokens,
            self.config.max_context_tokens,
        )
        .map_err(|e| leaf_error("model load", e))?;
        let cfg = ModelConfig::qwen38_27b();
        // The leaf's fixed paged-KV page size (`kPagedKVPageSize`, 64
        // tokens). The pool holds `kv_pool_tokens` worth of pages, shared
        // across the slots; `max_context_tokens` is the per-sequence cap
        // drawn against it. `ignis_server::runtime::cuda_scheduler` derives
        // the scheduler's admission accounting from the same two numbers
        // with the same arithmetic, so admission can never promise more
        // pages than this pool holds.
        let pool = SeqPool::create(
            &cfg,
            &SeqPoolBudget {
                kv_page_group_count: kv_pool_pages(self.config.kv_pool_tokens),
                max_context_tokens: self.config.max_context_tokens,
                slot_count: self.config.slot_count,
            },
        )
        .map_err(|e| leaf_error("seq pool create", e))?;
        // P3-05 (GitHub #102, ADR 0019): capture the decode graphs once,
        // right after the pool exists and before any sequence is ever
        // allocated -- a per-width capture failure degrades that width to
        // the eager per-lane loop, never model load. `capture_decode_graphs`
        // itself never returns Err for that reason; only a null model/pool
        // (impossible here) would.
        let capture = step::capture_decode_graphs(&model, &pool)
            .map_err(|e| leaf_error("decode graph capture", e))?;
        // Reported unconditionally (GitHub #102's acceptance: "startup cost
        // of capturing eight graphs is measured and reported, not assumed")
        // -- not only when a width failed, so the common all-8-ready case
        // still surfaces the number rather than computing and discarding it.
        // GitHub #80: structured, not a raw `eprintln!` -- this still fires
        // exactly once per model load (never per-token/decode-round), same
        // frequency class as `ignis.process.started`.
        // hotpath-lint-allow: model-load-time only (`load_model`, runs once per process start), not per-token/decode-round (GitHub #80/#102).
        tracing::info!(
            name: "ignis.runtime.decode_graph_capture",
            ready = capture.ready_count(),
            capture_micros = capture.capture_micros,
            "decode graph capture"
        );
        if capture.ready_count() < 8 {
            // hotpath-lint-allow: same model-load-time call as above, one line down.
            tracing::warn!(
                name: "ignis.runtime.decode_graph_capture_incomplete",
                error = %step::last_decode_graph_error(),
                "decode graph capture: not all widths ready"
            );
        }
        Ok(CudaModel { model, pool })
    }

    fn release_model(&self, _model: Self::Model) {
        // `CudaModel`'s fields release themselves on drop (`ignis_model_free`,
        // `ignis_seq_pool_free`).
    }

    fn stats(&self, model: &Self::Model) -> Result<RuntimeStats, i32> {
        let program = step::program_stats(&model.model, &model.pool)
            .map_err(|e| leaf_error("program stats", e))?;
        let pool_stats = model.pool.stats();
        Ok(RuntimeStats {
            vram_bytes: program.vram_bytes,
            // The leaf's paged KV page size is fixed at 64 tokens
            // (`kernel/vendor/src/core/paged_kv_cache.h`'s
            // `kPagedKVPageSize`) — the ABI does not report it per call.
            kv_page_tokens: 64,
            kv_page_bytes: pool_stats.kv_page_bytes,
            kv_page_count: pool_stats.kv_page_group_count,
            last_step_micros: program.last_step_micros,
            kernel_count: program.kernel_count,
            // P3-05 (GitHub #102, ADR 0019): 1 when the most recent decode
            // round replayed a captured graph, 0 for every prefill step and
            // for a decode round whose exact width has no ready graph.
            graph_launches: program.graph_launches,
        })
    }

    fn allocate_sequence(
        &self,
        model: &Self::Model,
        context_tokens: u32,
    ) -> Result<Self::Sequence, i32> {
        let seq = model
            .pool
            .alloc(context_tokens)
            .map_err(|e| leaf_error("sequence alloc", e))?;
        // Safety: `model.pool` outlives every sequence drawn from it —
        // `RuntimeCompute::drop` (`ignis-runtime/src/lib.rs`) releases all
        // live sequences before its `Arc<Model<L>>` (and so this pool) can
        // drop.
        Ok(unsafe { seq.into_static() })
    }

    fn release_sequence(&self, _model: &Self::Model, _sequence: Self::Sequence) {
        // Drops here: `Seq::drop` calls `ignis_seq_release`.
    }

    fn prefill(
        &self,
        model: &Self::Model,
        sequence: &mut Self::Sequence,
        tokens: &[TokenId],
        start_position: u32,
        params: DecodeParams,
    ) -> Result<(), i32> {
        let token_ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        step::prefill_program_sampled(
            &model.model,
            &model.pool,
            sequence,
            &token_ids,
            u64::from(start_position),
            sampling_params(params),
            None,
        )
        .map_err(|e| leaf_error("prefill", e))
    }

    fn decode(
        &self,
        model: &Self::Model,
        sequences: &mut [&mut Self::Sequence],
        params: &[DecodeParams],
    ) -> Result<Vec<TokenId>, i32> {
        if params.len() > N_DECODE_LANES {
            return Err(leaf_error(
                "decode",
                format!(
                    "batch has {} sampling parameter sets; maximum is {N_DECODE_LANES}",
                    params.len()
                ),
            ));
        }
        let mut sampling = [step::SamplingParams::greedy(); N_DECODE_LANES];
        for (target, source) in sampling.iter_mut().zip(params.iter().copied()) {
            *target = sampling_params(source);
        }
        let ids = step::decode_program_batch_sampled(
            &model.model,
            &model.pool,
            sequences,
            &sampling[..params.len()],
        )
        .map_err(|e| leaf_error("decode", e))?;
        Ok(ids.into_iter().map(|id| id as TokenId).collect())
    }
}

fn sampling_params(params: DecodeParams) -> step::SamplingParams {
    step::SamplingParams {
        temperature: params.temperature,
        top_k: params.top_k,
        top_p: params.top_p,
        presence_penalty: params.presence_penalty,
        frequency_penalty: params.frequency_penalty,
        seed: params.seed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pure arithmetic, no device — compiled (and run) whenever the `cuda`
    // feature is built, without needing the GPU profile (ADR 0006).

    #[test]
    fn kv_pool_pages_rounds_up_to_a_whole_page() {
        assert_eq!(kv_pool_pages(0), 0);
        assert_eq!(
            kv_pool_pages(1),
            1,
            "a partial page still reserves one whole page"
        );
        assert_eq!(kv_pool_pages(KV_PAGE_TOKENS), 1);
        assert_eq!(kv_pool_pages(KV_PAGE_TOKENS + 1), 2);
        assert_eq!(kv_pool_pages(65_536), 65_536 / KV_PAGE_TOKENS);
    }

    #[test]
    fn the_default_leaf_config_pool_can_hold_the_default_context() {
        // `cuda_scheduler` always overrides these fields from the operator's
        // `EngineShape` in production; this default is what a GPU test gets
        // when it builds a leaf directly, and it must not promise a context
        // the pool it also defaults to cannot serve.
        let config = CudaLeafConfig::default();
        assert!(config.kv_pool_tokens >= config.max_context_tokens);
        assert_eq!(config.max_context_tokens, crate::DEFAULT_MAX_CONTEXT);
        assert_eq!(config.prefill_chunk_tokens, crate::DEFAULT_PREFILL_CHUNK);
    }

    #[test]
    fn leaf_error_emits_a_structured_event_and_returns_the_generic_code() {
        use std::sync::Arc;
        use tracing_subscriber::layer::SubscriberExt;

        let sink = Arc::new(ignis_logging::MemorySink::new());
        let subscriber = tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone()));
        let code = tracing::subscriber::with_default(subscriber, || {
            leaf_error("prefill", "device out of memory".to_owned())
        });

        assert_eq!(code, -1, "the generic leaf error code, regardless of context");
        let lines = sink.lines();
        assert_eq!(lines.len(), 1);
        let record: serde_json::Value = serde_json::from_str(&lines[0]).expect("valid json");
        assert_eq!(record["event_name"], "ignis.runtime.leaf_error");
        assert_eq!(record["severity_text"], "ERROR");
        assert_eq!(record["attributes"]["context"], "prefill");
        assert_eq!(record["attributes"]["error"], "device out of memory");
    }

    #[test]
    fn decode_params_map_every_sampling_field_without_changing_seed_bits() {
        let mapped = sampling_params(DecodeParams {
            max_tokens: Some(17),
            temperature: 1.25,
            top_p: 0.75,
            top_k: 13,
            presence_penalty: -0.5,
            frequency_penalty: 0.625,
            seed: u64::MAX,
        });

        assert_eq!(
            mapped,
            step::SamplingParams {
                temperature: 1.25,
                top_p: 0.75,
                top_k: 13,
                presence_penalty: -0.5,
                frequency_penalty: 0.625,
                seed: u64::MAX,
            }
        );
    }
}
