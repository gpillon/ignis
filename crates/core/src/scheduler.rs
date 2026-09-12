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
    ComputeError, DecodeParams, FinishReason, LaneId, RequestClass, RequestId, RequestInput,
    SchedEvent, SubmitError, TokenId,
};

/// The shared prefix a prefill job claims (P4-10, GitHub #126, ADR 0024):
/// which request published it, and how many leading prompt tokens it covers.
///
/// The backend needs the publisher's identity rather than the scheduler's own
/// entry id because the leaf's prefix is what actually owns the pages, and
/// the publisher is the request whose prefill produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedPrefixClaim {
    /// The request whose prefill published the prefix.
    pub publisher: RequestId,
    /// The leading prompt tokens the prefix covers — always a whole number
    /// of KV pages, and always equal to this job's `start_position`.
    pub tokens: u32,
}

/// One prefill job handed to the compute backend (batched prefill groups
/// several of these into one GPU batch to saturate the GPU and cut burst TTFT).
#[derive(Debug, Clone)]
pub struct PrefillJob {
    /// The request being prefilled.
    pub request: RequestId,
    /// The prompt tokens to warm the KV for. A request that reuses a
    /// cached sibling prefix (core-07) carries only its *tail* — the
    /// leading shared tokens are already warm in the pool (the shared
    /// prefix's blocks are bound read-only by the kernel leaf). An empty
    /// tail (a full-prompt match) warms nothing: the job only sets up the
    /// decode state.
    pub tokens: Vec<TokenId>,
    /// Total sequence reservation, including the prompt and effective
    /// generation cap. The runtime uses it when it first allocates a leaf
    /// sequence for this request.
    pub context_tokens: u32,
    /// Position of `tokens[0]` in the sequence (nonzero after shared-prefix
    /// reuse).
    pub start_position: u32,
    /// The request's generation parameters (carried so the backend can set
    /// up the decode state; prefill only warms the KV).
    pub params: DecodeParams,
    /// The shared prefix this request claims (P4-10, GitHub #126), if any.
    /// Set on the request's **first** job: the backend allocates its
    /// sequence against the prefix, which shares the leading KV pages in
    /// place and clones the mutable state device-to-device, so the sequence
    /// begins at `start_position` with the publisher's state.
    pub shared_prefix: Option<SharedPrefixClaim>,
    /// Publish this request's first N tokens as a shared prefix once this
    /// chunk lands (P4-10, GitHub #126). Set only on the chunk that ends
    /// exactly at N: the mutable state a claimant clones is the state at the
    /// prefix's end, so the publish happens at that boundary and nowhere
    /// else.
    pub publish_prefix_tokens: Option<u32>,
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

/// One job's result from a decode step (GitHub #61 / P1-25): either the
/// token generated this step, or the reason the request finished instead
/// of generating one — the scheduler forwards the reason straight into the
/// [`SchedEvent::Done`] it emits, which the server maps to `finish_reason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeOutcome {
    /// The job produced a token and the request keeps running.
    Token(TokenId),
    /// The request finished this step instead of producing a token.
    Finished(FinishReason),
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
    /// Returns, per job in order, [`DecodeOutcome::Token`] when a token was
    /// generated, or [`DecodeOutcome::Finished`] with why (EOS or
    /// `max_tokens`) when that request finished this step instead.
    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError>;

    /// Release leaf-owned state for a request that completed or was evicted.
    /// CPU-only compute implementations need no lifecycle bookkeeping.
    fn release(&self, _request: RequestId) {}

    /// Release the backend's own handle on the shared prefix `publisher`
    /// published (P4-10, GitHub #126), once the scheduler's last claimant has
    /// gone. The leaf's pages return to the pool when every sequence holding
    /// them has been released too, so this is a handle drop and not a free.
    /// CPU-only compute implementations hold no such handle.
    fn release_prefix(&self, _publisher: RequestId) {}
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

    /// Abort an in-flight request. The scheduler must stop dealing it work
    /// and release its resources no later than the next [`Scheduler::advance`].
    /// Returns `false` when the request is unknown or already complete.
    fn cancel(&mut self, request: RequestId) -> bool;

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
