//! Shared types for the ignis engine's scheduling & state layer.
//!
//! This is the **stable public contract** between the engine (`ignis-core`)
//! and its consumers (the server crate, tests, the kernel leaf). The types
//! here are deliberately small and `Send + Sync` friendly. See
//! `docs/design/ignis-v1.md` §2 and `CONTEXT.md` for the domain vocabulary.

/// Opaque id for a request admitted by the scheduler.
pub type RequestId = u64;

/// Index of a resident decode lane (0..`N_DECODE_LANES`).
pub type LaneId = usize;

/// A single model token, in the loaded model's tokenizer id-space.
pub type TokenId = u32;

/// Fixed number of resident decode lanes (v1: 8, sized for a ~10-subagent
/// concurrent coding workload; overflow goes to the host KV-RAM tier).
pub const N_DECODE_LANES: usize = 8;

/// The engine's operating mode (what the scheduler + telemetry report).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineMode {
    /// Normal operation: decode lanes + admission active.
    Serving,
    /// No requests in flight.
    Idle,
}

/// A request as submitted to the scheduler: the prompt is **already
/// tokenized and chat-templated** (the server does that from the artifact's
/// frontend objects). The scheduler works on tokens, not raw messages.
#[derive(Debug, Clone)]
pub struct RequestInput {
    /// Model to route to. v1 loads a single model; validated against
    /// `Scheduler::model_id`.
    pub model: String,
    /// Tokenized + templated prompt.
    pub tokens: Vec<TokenId>,
    /// Generation parameters.
    pub params: DecodeParams,
}

/// Sampling / decoding parameters for a request.
///
/// v1 correctness floor is **greedy + fixed seed** (ADR 0007); `temperature`
/// is carried for the future but the acceptance gate is greedy.
#[derive(Debug, Clone, Copy)]
pub struct DecodeParams {
    /// Cap on generated tokens (`None` = until EOS / model max).
    pub max_tokens: Option<u32>,
    /// Sampling temperature (0 = greedy). v1 gate is greedy.
    pub temperature: f32,
    /// Sampling seed (fixed for reproducibility / the self-check).
    pub seed: u64,
}

impl Default for DecodeParams {
    fn default() -> Self {
        Self {
            max_tokens: None,
            temperature: 0.0,
            seed: 0,
        }
    }
}

/// The lifecycle of a request inside the engine (`CONTEXT.md`: "request state
/// machine (admit → prefill → decode → done / evict)").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    /// Accepted, queued, no lane yet.
    Admitted,
    /// Prefill in progress (global prefill lane / batched prefill).
    Prefilling,
    /// Holds a resident decode lane, generating.
    Running,
    /// Finished (reached `max_tokens` / EOS).
    Done,
}

/// Admission / backfill class for the admission state machine (ADR 0004).
/// Drives protection, backfill priority, and eviction ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestClass {
    /// Foreground interactive request — highest priority, protected from
    /// eviction while active.
    Interactive,
    /// Background agent subtask — the backfill class that fills lanes left
    /// free by interactive traffic.
    Agent,
}

/// An event emitted by a scheduler step. This is what the server streams to
/// clients and what the telemetry writer logs (ADR 0007: the telemetry
/// counters are derived from these events).
#[derive(Debug, Clone)]
pub enum SchedEvent {
    /// A new token was generated for a request.
    Token { request: RequestId, token: TokenId },
    /// A request completed (`tokens` = total generated this request).
    Done { request: RequestId, tokens: u32 },
    /// A request was admitted onto a decode lane.
    Admitted { request: RequestId, lane: LaneId },
    /// A request was evicted from a decode lane to the host KV-RAM tier
    /// (sibling prefix reuse will restore it instead of re-prefilling).
    Evicted { request: RequestId },
}

/// Errors from submitting a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmitError {
    /// The engine cannot admit the request right now (all lanes + admission
    /// capacity in use). The caller should retry or queue.
    Full,
    /// The request named a model the engine does not load.
    UnknownModel(String),
}

impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubmitError::Full => write!(f, "engine cannot admit the request (full)"),
            SubmitError::UnknownModel(m) => write!(f, "unknown model: {m}"),
        }
    }
}

impl std::error::Error for SubmitError {}

/// Errors from a compute step (prefill / decode) driven by the scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeError {
    /// The kernel leaf reported a CUDA / argument error (return code).
    Kernel(i32),
    /// A request was asked to generate beyond its `max_tokens` / EOS while
    /// still scheduled (a soft stop, not a fault).
    Stopped,
}

impl std::fmt::Display for ComputeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputeError::Kernel(rc) => write!(f, "kernel error (rc = {rc})"),
            ComputeError::Stopped => write!(f, "request stopped (max_tokens / EOS)"),
        }
    }
}

impl std::error::Error for ComputeError {}