//! Safe ownership of the step ABI and its [`ignis_core::Compute`] adapter.
//!
//! The runtime owns a loaded model, one opaque sequence per scheduler
//! request, integer error-code mapping, and the sequence-release lifecycle.
//! The C ABI adapter lands with P1-23; the small [`StepLeaf`] seam lets this
//! ownership logic be tested today against a CPU stub.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ignis_core::{Compute, ComputeError, DecodeJob, PrefillJob, RequestId, TokenId};

/// A failure returned by the step ABI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    /// The leaf's integer return code.
    Leaf(i32),
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
    ) -> Result<(), i32>;
    /// Decode one token from a warmed sequence.
    fn decode(&self, model: &Self::Model, sequence: &mut Self::Sequence) -> Result<TokenId, i32>;
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
                let handle = self
                    .model
                    .leaf
                    .allocate_sequence(self.model.handle(), job.context_tokens)
                    .map_err(RuntimeError::Leaf)
                    .map_err(ComputeError::from)?;
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
            ) {
                let failed = sequences
                    .remove(&job.request)
                    .expect("failed prefill still owns its sequence");
                drop(sequences);
                self.release_sequence(failed.handle);
                return Err(RuntimeError::Leaf(code).into());
            }
        }
        Ok(())
    }

    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<Option<TokenId>>, ComputeError> {
        let mut sequences = self.sequences.lock().unwrap();
        let mut finished = Vec::new();
        let mut tokens = Vec::with_capacity(jobs.len());
        for job in jobs {
            let sequence = sequences
                .get_mut(&job.request)
                .ok_or(ComputeError::Kernel(-1))?;
            if job
                .params
                .max_tokens
                .is_some_and(|max| sequence.generated >= max)
            {
                finished.push(job.request);
                tokens.push(None);
                continue;
            }
            let token = self
                .model
                .leaf
                .decode(self.model.handle(), &mut sequence.handle)
                .map_err(RuntimeError::Leaf)
                .map_err(ComputeError::from)?;
            if token == self.eos {
                finished.push(job.request);
                tokens.push(None);
            } else {
                sequence.generated += 1;
                tokens.push(Some(token));
            }
        }
        let released: Vec<_> = finished
            .into_iter()
            .filter_map(|request| sequences.remove(&request))
            .collect();
        drop(sequences);
        for sequence in released {
            self.release_sequence(sequence.handle);
        }
        Ok(tokens)
    }

    fn release(&self, request: RequestId) {
        let sequence = self.sequences.lock().unwrap().remove(&request);
        if let Some(sequence) = sequence {
            self.release_sequence(sequence.handle);
        }
    }
}
