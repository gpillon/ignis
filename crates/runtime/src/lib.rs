//! Safe ownership of the step ABI and its [`ignis_core::Compute`] adapter.
//!
//! The runtime owns a loaded model, one opaque sequence per scheduler
//! request, integer error-code mapping, and the sequence-release lifecycle.
//! The C ABI adapter lands with P1-23; the small [`StepLeaf`] seam lets this
//! ownership logic be tested today against a CPU stub.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ignis_core::{
    Compute, ComputeError, DecodeJob, DecodeOutcome, DecodeParams, FinishReason, N_DECODE_LANES,
    PrefillJob, RequestId, TokenId,
};

#[cfg(feature = "cuda")]
mod cuda_leaf;
#[cfg(feature = "cuda")]
pub use cuda_leaf::{CudaLeaf, CudaLeafConfig, CudaModel};

/// Tokens held by one physical KV page, in either format
/// (`kPagedKVPageSize`). Re-exported from `ignis-core` so the server's
/// scheduler accounting and the leaf name the same constant.
pub use ignis_core::KV_PAGE_TOKENS;

/// The default prefill chunk width, in tokens (spec
/// `.scratch/runtime/specs/02-real-prefill.md`): the reference's own
/// default, left alone — the chunk width is a knob this phase exposes,
/// not a number it tunes. Unconditional on the `cuda` feature: it is a
/// plain number, and both `ignis_server::config` (always compiled) and
/// [`CudaLeafConfig::default`] (`cuda` only) fall back to it, so it has to
/// live somewhere both can reach without one depending on the other.
pub const DEFAULT_PREFILL_CHUNK: u32 = 1024;

/// The prefill chunk width's alignment rule, in tokens: the reference's
/// own alignment, and a multiple of the 64-token chunk the vendored GDN
/// chunked kernels work in.
pub const PREFILL_CHUNK_ALIGNMENT: u32 = 128;

/// The default maximum per-sequence context, in tokens: a 32,768-token
/// prompt plus an 8,192-token generation budget. G2's largest cell is a
/// 32K prompt, so the default must admit one without editing code (spec
/// `02-real-prefill.md`, user story 22).
pub const DEFAULT_MAX_CONTEXT: u32 = 32_768 + 8_192;

/// The paged-KV pool's auto byte budget for a configured `max_context`
/// under `format` (P4-04, GitHub #122): [`ignis_core::DEFAULT_KV_POOL_BYTES`]
/// (4 GiB), raised if one configured context would not fit inside it.
///
/// Deliberately not `slot_count * max_context`: reserving a full
/// 40,960-token context for each of the eight decode lanes is ~20 GiB of
/// BF16 paged KV at this model's geometry, which does not fit next to
/// ~19 GB of weights. The pool is sized so one sequence can take the whole
/// 32K cell and the other lanes still have a working budget; a request the
/// free pool cannot cover is a scheduler admission decision, not a load
/// failure.
///
/// The budget is in bytes, and what it *buys* is derived from the format:
/// 4 GiB is 65,536 resident BF16 tokens and 465,984 hq-e8-2b ones. That is
/// why this replaced the former `kv_pool_tokens_for` — a token target is
/// exactly the thing that cannot be format-independent.
pub fn auto_kv_pool_bytes(format: ignis_core::KvFormat, max_context: u32) -> u64 {
    ignis_core::auto_kv_pool_bytes(format, ignis_core::KvGeometry::qwen38_27b(), max_context)
}

/// A failure returned by the step ABI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    /// The leaf's integer return code.
    Leaf(i32),
}

/// Counters and geometry reported by the step runtime.
///
/// The scheduler consumes the page geometry for admission accounting; the
/// timing and dispatch counters feed the server/bench telemetry once P1-23's
/// FFI leaf exposes them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RuntimeStats {
    /// Bytes retained by the leaf for the loaded model and live sequences.
    pub vram_bytes: u64,
    /// Tokens held by one physical KV page.
    pub kv_page_tokens: u32,
    /// Bytes in one physical KV page.
    pub kv_page_bytes: u64,
    /// Physical KV pages the pool actually holds (the leaf's own build, not
    /// a requested budget; `ignis_core::kv::verified_kv_pool` cross-checks
    /// this against the scheduler's capacity).
    pub kv_page_count: u32,
    /// Duration of the most recent leaf step.
    pub last_step_micros: u64,
    /// Kernels dispatched by the most recent leaf step.
    pub kernel_count: u64,
    /// CUDA graph launches by the most recent leaf step.
    pub graph_launches: u64,
}

impl From<RuntimeError> for ComputeError {
    fn from(value: RuntimeError) -> Self {
        match value {
            RuntimeError::Leaf(code) => Self::Kernel(code),
        }
    }
}

/// The replaceable step-ABI leaf seam.
///
/// The FFI implementation will map these calls to ADR 0009. Its opaque
/// handles stay inside the runtime; callers can only use the safe model and
/// compute adapter.
pub trait StepLeaf: Send + Sync + 'static {
    /// Opaque loaded-model handle.
    type Model: Send + Sync + 'static;
    /// Opaque device-resident sequence handle.
    type Sequence: Send + 'static;

    /// Load a model handle.
    fn load_model(&self) -> Result<Self::Model, i32>;
    /// Release a model handle.
    fn release_model(&self, model: Self::Model);
    /// Read the leaf's current geometry and step counters.
    fn stats(&self, model: &Self::Model) -> Result<RuntimeStats, i32>;
    /// Allocate one sequence with its full context reservation.
    fn allocate_sequence(
        &self,
        model: &Self::Model,
        context_tokens: u32,
    ) -> Result<Self::Sequence, i32>;
    /// Release a sequence allocation.
    fn release_sequence(&self, model: &Self::Model, sequence: Self::Sequence);
    /// Warm one sequence with a prefill span.
    fn prefill(
        &self,
        model: &Self::Model,
        sequence: &mut Self::Sequence,
        tokens: &[TokenId],
        start_position: u32,
        params: DecodeParams,
    ) -> Result<(), i32>;
    /// Decode one round over a batch of warmed sequences with one parameter
    /// set per sequence. `params` is parallel to `sequences`. On an error,
    /// the leaf must leave every input sequence unchanged so the scheduler
    /// can retry the round without corrupting token order.
    fn decode(
        &self,
        model: &Self::Model,
        sequences: &mut [&mut Self::Sequence],
        params: &[DecodeParams],
    ) -> Result<Vec<TokenId>, i32>;
}

/// A loaded model whose leaf handle is released exactly once on drop.
pub struct Model<L: StepLeaf> {
    leaf: Arc<L>,
    handle: Option<L::Model>,
}

impl<L: StepLeaf> Model<L> {
    /// Load the leaf model behind a safe, owning handle.
    pub fn load(leaf: Arc<L>) -> Result<Self, RuntimeError> {
        let handle = leaf.load_model().map_err(RuntimeError::Leaf)?;
        Ok(Self {
            leaf,
            handle: Some(handle),
        })
    }

    fn handle(&self) -> &L::Model {
        self.handle
            .as_ref()
            .expect("a live model always owns its leaf handle")
    }

    /// Read the loaded model's leaf statistics through the safe wrapper.
    pub fn stats(&self) -> Result<RuntimeStats, RuntimeError> {
        self.leaf.stats(self.handle()).map_err(RuntimeError::Leaf)
    }
}

impl<L: StepLeaf> Drop for Model<L> {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            self.leaf.release_model(handle);
        }
    }
}

struct LiveSequence<S> {
    handle: S,
    generated: u32,
}

/// Scheduler adapter over a loaded step-ABI model.
///
/// A request gains a sequence on its first prefill. The map is private, so a
/// caller cannot decode without a sequence or forget its leaf release.
pub struct RuntimeCompute<L: StepLeaf> {
    model: Arc<Model<L>>,
    eos: TokenId,
    sequences: Mutex<HashMap<RequestId, LiveSequence<L::Sequence>>>,
}

impl<L: StepLeaf> RuntimeCompute<L> {
    /// Build an adapter for `model`; the server obtains `eos` from artifact
    /// generation defaults when it wires the real leaf.
    pub fn new(model: Arc<Model<L>>, eos: TokenId) -> Self {
        Self {
            model,
            eos,
            sequences: Mutex::new(HashMap::new()),
        }
    }

    /// Number of live leaf sequences (the CPU-stub observation point).
    pub fn live_sequences(&self) -> usize {
        self.sequences.lock().unwrap().len()
    }

    fn release_sequence(&self, sequence: L::Sequence) {
        self.model
            .leaf
            .release_sequence(self.model.handle(), sequence);
    }
}

impl<L: StepLeaf> Compute for RuntimeCompute<L> {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<(), ComputeError> {
        let mut sequences = self.sequences.lock().unwrap();
        for job in jobs {
            if !sequences.contains_key(&job.request) {
                let handle = match self
                    .model
                    .leaf
                    .allocate_sequence(self.model.handle(), job.context_tokens)
                {
                    Ok(handle) => handle,
                    Err(code) => {
                        let released: Vec<_> = jobs
                            .iter()
                            .filter_map(|job| sequences.remove(&job.request))
                            .collect();
                        drop(sequences);
                        for sequence in released {
                            self.release_sequence(sequence.handle);
                        }
                        return Err(RuntimeError::Leaf(code).into());
                    }
                };
                sequences.insert(
                    job.request,
                    LiveSequence {
                        handle,
                        generated: 0,
                    },
                );
            }
            let sequence = sequences
                .get_mut(&job.request)
                .expect("sequence was inserted or already existed");
            if let Err(code) = self.model.leaf.prefill(
                self.model.handle(),
                &mut sequence.handle,
                &job.tokens,
                job.start_position,
                job.params,
            ) {
                // Core retries a failed *batch*, not only this job. Return
                // every batch sequence to zero state so that retry does not
                // prefill a successful earlier span twice.
                let released: Vec<_> = jobs
                    .iter()
                    .filter_map(|job| sequences.remove(&job.request))
                    .collect();
                drop(sequences);
                for sequence in released {
                    self.release_sequence(sequence.handle);
                }
                return Err(RuntimeError::Leaf(code).into());
            }
        }
        Ok(())
    }

    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        if jobs.len() > N_DECODE_LANES {
            return Err(ComputeError::Kernel(-1));
        }
        let mut sequences = self.sequences.lock().unwrap();
        let mut batch: Vec<(DecodeJob, LiveSequence<L::Sequence>)> = Vec::with_capacity(jobs.len());
        for job in jobs {
            let Some(sequence) = sequences.remove(&job.request) else {
                for (job, sequence) in batch {
                    sequences.insert(job.request, sequence);
                }
                return Err(ComputeError::Kernel(-1));
            };
            batch.push((job.clone(), sequence));
        }

        // `None` until filled below; every index is either already-finished
        // (a prior round's `max_tokens`, `Length`), gets a fresh token, or
        // finishes this round (EOS, `Stop`) — so every slot is set once.
        let mut outcomes: Vec<Option<DecodeOutcome>> = vec![None; jobs.len()];
        let mut active = vec![false; jobs.len()];
        for (index, (job, sequence)) in batch.iter().enumerate() {
            if job
                .params
                .max_tokens
                .is_some_and(|max| sequence.generated >= max)
            {
                outcomes[index] = Some(DecodeOutcome::Finished(FinishReason::Length));
            } else {
                active[index] = true;
            }
        }
        let decoded = {
            let mut params = [DecodeParams::default(); N_DECODE_LANES];
            let mut params_len = 0;
            for (index, (job, _)) in batch.iter().enumerate() {
                if active[index] {
                    params[params_len] = job.params;
                    params_len += 1;
                }
            }
            let mut handles: Vec<&mut L::Sequence> = batch
                .iter_mut()
                .enumerate()
                .filter(|(index, _)| active[*index])
                .map(|(_, (_, sequence))| &mut sequence.handle)
                .collect();
            self.model
                .leaf
                .decode(self.model.handle(), &mut handles, &params[..params_len])
        };
        let decoded = match decoded {
            Ok(tokens) if tokens.len() == active.iter().filter(|&&active| active).count() => tokens,
            Ok(_) => {
                for (job, sequence) in batch {
                    sequences.insert(job.request, sequence);
                }
                return Err(ComputeError::Kernel(-1));
            }
            Err(code) => {
                for (job, sequence) in batch {
                    sequences.insert(job.request, sequence);
                }
                return Err(RuntimeError::Leaf(code).into());
            }
        };

        let mut decoded = decoded.into_iter();
        let mut released = Vec::new();
        for (index, (job, mut sequence)) in batch.into_iter().enumerate() {
            if !active[index] {
                released.push(sequence);
                continue;
            }
            let token = decoded.next().expect("decoded result length was checked");
            if token == self.eos && !job.params.ignore_eos {
                outcomes[index] = Some(DecodeOutcome::Finished(FinishReason::Stop));
                released.push(sequence);
            } else {
                sequence.generated += 1;
                outcomes[index] = Some(DecodeOutcome::Token(token));
                sequences.insert(job.request, sequence);
            }
        }
        drop(sequences);
        for sequence in released {
            self.release_sequence(sequence.handle);
        }
        Ok(outcomes
            .into_iter()
            .map(|o| o.expect("every job index is filled by one of the branches above"))
            .collect())
    }

    fn release(&self, request: RequestId) {
        let sequence = self.sequences.lock().unwrap().remove(&request);
        if let Some(sequence) = sequence {
            self.release_sequence(sequence.handle);
        }
    }
}

impl<L: StepLeaf> Drop for RuntimeCompute<L> {
    fn drop(&mut self) {
        let sequences = std::mem::take(
            self.sequences
                .get_mut()
                .expect("RuntimeCompute is not dropped while its sequence lock is held"),
        );
        for (_, sequence) in sequences {
            self.release_sequence(sequence.handle);
        }
    }
}
