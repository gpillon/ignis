//! The engine's scheduling contract.
//!
//! Two traits define the public surface:
//!
//! - [`Scheduler`] — what the *server* drives: submit requests, advance the
//!   engine, stream events. This is the stable contract the `ignis-server`
//!   crate codes against.
//! - [`Compute`] — the *compute seam*: the only GPU-coupled step (prefill /
//!   decode). Production is the kernel leaf (C ABI, see `ffi.rs`); tests use
//!   a deterministic mock. Keeping this seam narrow is what makes the whole
//!   scheduler (admission, lanes, batched prefill, eviction) CPU-testable
//!   without a GPU (ADR 0006).

use crate::types::{
    ComputeError, DecodeParams, LaneId, RequestClass, RequestId, RequestInput, SchedEvent,
    SubmitError, TokenId,
};

/// One prefill job handed to the compute backend (batched prefill groups
/// several of these into one GPU batch to saturate the GPU and cut burst TTFT).
#[derive(Debug, Clone)]
pub struct PrefillJob {
    /// The request being prefilled.
    pub request: RequestId,
    /// The prompt tokens to warm the KV for.
    pub tokens: Vec<TokenId>,
    /// The request's generation parameters (carried so the backend can set
    /// up the decode state; prefill only warms the KV).
    pub params: DecodeParams,
}

/// One decode job: a single lane step for a running request.
#[derive(Debug, Clone)]
pub struct DecodeJob {
    /// The request decoding.
    pub request: RequestId,
    /// The resident lane it holds (used for KV block mapping).
    pub lane: LaneId,
    /// The request's generation parameters (sampler setup, `max_tokens` /
    /// EOS handling, fixed seed — ADR 0007).
    pub params: DecodeParams,
}

/// The compute seam the scheduler drives for actual token generation.
///
/// This is the *only* GPU-coupled step in the engine. The scheduler's logic —
/// admission, lane assignment, batched prefill grouping, eviction, state
/// machine — never touches the GPU directly; it only calls this trait. That
/// is the seam that keeps the engine testable on a CPU-only machine (ADR
/// 0006: GPU testing is exclusive; a mock stands in for the kernel leaf).
pub trait Compute: Send + Sync {
    /// Prefill a batch of prompts, warming their KV (fills the block tables).
    /// No tokens are emitted; this only sets the request up for decode.
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<(), ComputeError>;

    /// Generate the next token for each running lane (one decode step).
    /// Returns, per job in order, the token generated this step, or `None` if
    /// that request finished this step (reached `max_tokens` / EOS).
    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<Option<TokenId>>, ComputeError>;
}

/// The engine's scheduling interface — what the server drives.
///
/// The server submits requests and calls [`Scheduler::advance`] in a loop,
/// streaming the emitted [`SchedEvent`]s back to clients (SSE) and into the
/// telemetry writer. Production is the real engine (kernel leaf via FFI);
/// tests use a mock.
pub trait Scheduler: Send {
    /// Enqueue a new request. `class` is the admission / backfill class the
    /// server assigns at submission time (a foreground interactive request vs
    /// a background agent subtask); it drives admission priority + eviction
    /// order (ADR 0004). Returns the request's id. Fails with
    /// [`SubmitError::Full`] when the engine cannot admit it right now — the
    /// caller should retry or queue.
    fn submit(
        &mut self,
        input: RequestInput,
        class: RequestClass,
    ) -> Result<RequestId, SubmitError>;

    /// Advance the engine by one scheduling step:
    /// - run **batched prefill** for queued requests (grouped into one GPU
    ///   batch to saturate the GPU and cut burst TTFT),
    /// - decode one token on each resident decode lane.
    ///
    /// Returns the events emitted this step (new tokens, completions,
    /// evictions, admissions).
    fn advance(&mut self) -> Vec<SchedEvent>;

    /// True when no request is in flight (nothing to schedule).
    fn is_idle(&self) -> bool;

    /// The loaded model id (for `GET /v1/models`).
    fn model_id(&self) -> &str;

    /// The operating mode (reported by telemetry interval lines).
    fn mode(&self) -> crate::types::EngineMode;
}