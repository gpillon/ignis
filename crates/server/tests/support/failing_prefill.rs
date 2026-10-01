//! A backend whose every prefill fails, shared by the HTTP tests of each
//! surface that reports an engine error (GitHub #166, #296). Included via
//! `#[path]`, like the rest of `support/`.

use ignis_core::{Compute, ComputeError, DecodeJob, DecodeOutcome, MockCompute, PrefillJob, PrefillOutcome, RequestId};

/// Every `prefill_step` fails the way a leaf error does, so the scheduler
/// ends the request with `FinishReason::Error` after its retries.
pub struct FailingPrefill(pub MockCompute);

impl Compute for FailingPrefill {
    fn prefill_step(&self, _jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
        Err(ComputeError::Kernel(-1))
    }
    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        self.0.decode_step(jobs)
    }
    fn release(&self, request: RequestId) {
        self.0.release(request);
    }
}
