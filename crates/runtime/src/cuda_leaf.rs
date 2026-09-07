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

use ignis_core::model_load::{self, Model as CoreModel};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step;
use ignis_core::{ModelConfig, N_DECODE_LANES, TokenId};
use ignis_artifact::{CudaDevice, MaterializedArtifact, ObjectHandle, Reader};

use crate::{RuntimeStats, StepLeaf};

/// Sizing knobs for the leaf's sequence-state pool.
#[derive(Debug, Clone, Copy)]
pub struct CudaLeafConfig {
    /// The largest single sequence's KV reservation, in tokens (mirrors
    /// `ignis_core::SchedulerConfig::max_sequence_tokens`).
    pub max_context_tokens: u32,
    /// Max concurrent sequences (mirrors [`N_DECODE_LANES`]).
    pub slot_count: u32,
}

impl Default for CudaLeafConfig {
    fn default() -> Self {
        Self {
            // A modest, deterministic default: `slot_count *
            // max_context_tokens` sequence-tokens of BF16 paged KV is a
            // few hundred MiB at this model's KV geometry (head_dim 256,
            // 4 KV heads) — negligible next to the ~19 GB of weights, so
            // it fits regardless of exactly how much VRAM the weights and
            // the program's own workspace left behind.
            max_context_tokens: 4096,
            slot_count: N_DECODE_LANES as u32,
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
fn leaf_error(context: &str, message: String) -> i32 {
    eprintln!("ignis-runtime: {context}: {message}");
    -1
}

impl StepLeaf for CudaLeaf {
    type Model = CudaModel;
    type Sequence = Seq<'static>;

    fn load_model(&self) -> Result<Self::Model, i32> {
        let model = model_load::load_qwen38_27b(&self.reader, &self.artifact, &self.handles)
            .map_err(|e| leaf_error("model load", e))?;
        let cfg = ModelConfig::qwen38_27b();
        // The leaf's fixed paged-KV page size (`kPagedKVPageSize`, 64
        // tokens) — every slot reserves enough pages for its full context.
        let pages_per_sequence = self.config.max_context_tokens.div_ceil(64);
        let pool = SeqPool::create(
            &cfg,
            &SeqPoolBudget {
                kv_page_group_count: self.config.slot_count * pages_per_sequence,
                max_context_tokens: self.config.max_context_tokens,
                slot_count: self.config.slot_count,
            },
        )
        .map_err(|e| leaf_error("seq pool create", e))?;
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
            last_step_micros: program.last_step_micros,
            kernel_count: program.kernel_count,
            // CUDA graph capture lands at G3; the program always runs
            // eager today.
            graph_launches: 0,
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
    ) -> Result<(), i32> {
        let token_ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        step::prefill_program(
            &model.model,
            &model.pool,
            sequence,
            &token_ids,
            u64::from(start_position),
        )
        .map_err(|e| leaf_error("prefill", e))
    }

    fn decode(
        &self,
        model: &Self::Model,
        sequences: &mut [&mut Self::Sequence],
    ) -> Result<Vec<TokenId>, i32> {
        let ids = step::decode_program_batch(&model.model, &model.pool, sequences)
            .map_err(|e| leaf_error("decode", e))?;
        Ok(ids.into_iter().map(|id| id as TokenId).collect())
    }
}
