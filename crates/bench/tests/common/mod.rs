//! Shared test infrastructure: an in-process mock of the engine's
//! OpenAI-compatible HTTP surface (the wire shape mimics
//! `crates/server/src/api.rs` — the SSE `chat.completion.chunk` framing, the
//! non-streaming `chat.completion` JSON, the `list` models envelope, and
//! OpenAI's `{"error": {message, type, code}}` bodies).
//!
//! The mock is a small axum router served on a random `127.0.0.1` port by a
//! background tokio runtime, so the *real* HTTP transport
//! (`HttpEndpoint` / reqwest) is exercised end to end — in-process, CPU-only
//! (ADR 0006: no GPU, no model, no external engine).
//!
//! Deterministic pacing: a fixed prefill before the first SSE chunk (the
//! mock's "prefill" — what the client measures as ttft); the remaining
//! chunks arrive at a fixed decode cadence, and a non-streaming
//! request's "generation" time is the prefill + a per-token decode interval.
//! Token counts are capped (`MAX_TOKENS`) to keep the tests fast. The mock
//! tracks how many requests are in flight (and the peak), so the tests can
//! prove the load was actually concurrent.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_core::Stream;
use serde::Deserialize;

/// The model id the mock reports (the "loaded model").
pub const MODEL: &str = "mock-model";
/// The prefill before the first SSE chunk (the mock's "ttft"), in ms.
const PREFILL: Duration = Duration::from_millis(20);
/// The decode interval between SSE chunks (the mock's "tok-s" pacing), in ms.
const TOKEN: Duration = Duration::from_millis(2);
/// Cap on the tokens the mock generates per request (keep the tests fast).
const MAX_TOKENS: u32 = 32;

/// The text of mock token `i` (deterministic).
pub fn token_text(i: u32) -> String {
    format!("tok-{i}")
}

// ── shared state ──────────────────────────────────────────────────────────

/// The shared mock state (behind `axum::extract::State`). `Clone` (cheap —
/// the inner state is `Arc`-shared across handlers) + `Send + Sync` (axum's
/// `State` bound).
#[derive(Clone, Debug)]
pub struct MockState {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    /// The loaded model id (reported at `/v1/models`).
    model: String,
    /// When set, completions return a 503 (the engine's `engine_full` shape).
    fail_503: AtomicBool,
    /// Requests currently in flight (requests admitted, not yet finished).
    in_flight: AtomicUsize,
    /// The max in-flight observed so far (the peak of the load profile).
    peak_in_flight: AtomicUsize,
    /// A per-request id counter (the mock's completion ids).
    next_id: std::sync::atomic::AtomicU64,
    /// The `prompt_tokens_details.cached_tokens` the mock reports, or -1
    /// for "the engine does not report the field" (the TTFT instrument's
    /// cold-prefix evidence — `ttft.rs`).
    cached_prompt_tokens: std::sync::atomic::AtomicI64,
    /// When set, the mock generates on the **thinking** channel: every
    /// token goes out as `delta.reasoning_content` (and
    /// `message.reasoning_content` when non-streaming), a `tool_calls`
    /// chunk follows, and the stream finishes on `"tool_calls"` — the shape
    /// a real agentic turn takes with `enable_thinking` on, and the one
    /// GitHub #137 found the client could not measure.
    thinking: AtomicBool,
    /// Every prompt the mock has been sent, in arrival order (so a test
    /// can prove the samples were distinct on the wire).
    prompts: Mutex<Vec<String>>,
    /// The token count of every completed request (for test assertions).
    completed: Mutex<Vec<u32>>,
}

#[allow(dead_code)] // each test binary uses a subset of the accessors
impl MockState {
    /// Build the state for `model` (the "loaded model" the mock reports).
    pub fn new(model: &str) -> Self {
        Self {
            inner: Arc::new(Inner {
                model: model.into(),
                fail_503: AtomicBool::new(false),
                thinking: AtomicBool::new(false),
                in_flight: AtomicUsize::new(0),
                peak_in_flight: AtomicUsize::new(0),
                next_id: std::sync::atomic::AtomicU64::new(0),
                cached_prompt_tokens: std::sync::atomic::AtomicI64::new(-1),
                completed: Mutex::new(Vec::new()),
                prompts: Mutex::new(Vec::new()),
            }),
        }
    }

    /// The "loaded model" id the mock reports.
    pub fn model(&self) -> &str {
        &self.inner.model
    }

    /// Make the next completions fail with a 503 `engine_full` (the
    /// engine's "cannot admit right now" shape).
    pub fn set_fail_503(&self, flag: bool) {
        self.inner.fail_503.store(flag, Ordering::Relaxed);
    }

    /// Generate on the thinking channel from now on: every token goes out
    /// as `reasoning_content`, followed by a tool call — no `delta.content`
    /// chunk at all (GitHub #137).
    pub fn set_thinking(&self, flag: bool) {
        self.inner.thinking.store(flag, Ordering::Relaxed);
    }

    /// The number of requests currently in flight.
    pub fn in_flight(&self) -> usize {
        self.inner.in_flight.load(Ordering::Relaxed)
    }

    /// The max in-flight observed so far (the peak of the load profile).
    pub fn peak_in_flight(&self) -> usize {
        self.inner.peak_in_flight.load(Ordering::Relaxed)
    }

    /// How many requests completed so far.
    pub fn n_completed(&self) -> usize {
        self.inner.completed.lock().unwrap().len()
    }

    /// The token counts of every completed request (in completion order).
    pub fn completed_tokens(&self) -> Vec<u32> {
        self.inner.completed.lock().unwrap().clone()
    }

    /// Serve the next requests as if `cached` of their prompt tokens came
    /// from a prefix cache (`usage.prompt_tokens_details.cached_tokens`) —
    /// the contamination the TTFT instrument must catch. `None` restores
    /// the default: the field is not reported at all.
    pub fn set_cached_prompt_tokens(&self, cached: Option<u32>) {
        self.inner
            .cached_prompt_tokens
            .store(cached.map(i64::from).unwrap_or(-1), Ordering::Relaxed);
    }

    /// Every prompt the mock received, in arrival order.
    pub fn prompts(&self) -> Vec<String> {
        self.inner.prompts.lock().unwrap().clone()
    }
}

/// An in-process mock engine: the axum router (above) served on a random
/// `127.0.0.1` port by a background multi-threaded tokio runtime.
pub struct MockEngine {
    /// The mock's shared state (the accessors above + `set_fail_503`).
    pub state: MockState,
    /// The base URL to point an `HttpEndpoint` at.
    url: String,
    /// The serve runtime (keeps the axum server alive; dropped on cleanup).
    _runtime: tokio::runtime::Runtime,
}

impl MockEngine {
    /// Start the mock on a random local port.
    pub fn start() -> Self {
        let state = MockState::new(MODEL);
        let state_for_router = state.clone();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("mock runtime");
        let url = runtime.block_on(async move {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind a local port");
            let port = listener.local_addr().expect("local addr").port();
            let app = Router::new()
                .route("/v1/models", get(list_models))
                .route("/v1/chat/completions", post(completions))
                .with_state(state_for_router);
            // The serve task runs on the runtime's worker threads (the
            // runtime is dropped on cleanup, cancelling the task).
            tokio::spawn(async move {
                let _ = axum::serve(listener, app).await;
            });
            format!("http://127.0.0.1:{port}")
        });
        Self {
            state,
            url,
            _runtime: runtime,
        }
    }

    /// The base URL of the running mock.
    pub fn url(&self) -> &str {
        &self.url
    }
}

// ── handlers ──────────────────────────────────────────────────────────────

/// `GET /v1/models` — the loaded model (v1: a single model), the server's
/// `list` envelope.
async fn list_models(State(state): State<MockState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{ "id": state.model(), "object": "model", "owned_by": "ignis" }],
    }))
}

/// A chat-completions request (the server's wire shape — unknown fields are
/// ignored, serde's default). `temperature` / `seed` / the message `role`
/// are part of the wire shape the client sends (the mock is deterministic,
/// so they are not read here — `#[allow(dead_code)]`).
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct CompletionsRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    seed: Option<u64>,
    /// OpenAI's `stream_options` — the TTFT instrument sets
    /// `include_usage` to read the engine's own prompt-token figures back
    /// off a streaming response.
    #[serde(default)]
    stream_options: Option<StreamOptions>,
}

/// `stream_options` (the trailing usage chunk opt-in).
#[derive(Debug, Default, Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

/// One conversation message (the server's `role` + `content` shape).
#[allow(dead_code)] // the `role` is part of the wire shape, not read here
#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

/// `POST /v1/chat/completions` — streaming (SSE) or non-streaming, mimicking
/// the server's shapes and error bodies.
async fn completions(
    State(state): State<MockState>,
    Json(req): Json<CompletionsRequest>,
) -> Response {
    if req.messages.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "messages must not be empty",
        );
    }
    if state.inner.fail_503.load(Ordering::Relaxed) {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "engine_full",
            "the engine cannot admit the request right now (all lanes in use); retry",
        );
    }
    // The mock's "prompt tokens": the last message's whitespace word count
    // (the server's real figure is the templated token count).
    let prompt_tokens = prompt_token_count(&req);
    state
        .inner
        .prompts
        .lock()
        .unwrap()
        .push(req.messages.last().map(|m| m.content.clone()).unwrap_or_default());
    let cached = match state.inner.cached_prompt_tokens.load(Ordering::Relaxed) {
        c if c < 0 => None,
        c => Some(c as u32),
    };
    let n = req.max_tokens.unwrap_or(MAX_TOKENS).min(MAX_TOKENS);
    let thinking = state.inner.thinking.load(Ordering::Relaxed);
    let model = req
        .model
        .clone()
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| state.model().to_string());
    let id = format!("mock-cmpl-{}", state.inner.next_id.fetch_add(1, Ordering::Relaxed));
    let created = now_secs();
    if req.stream {
        // The SSE response: hold the request for the prefill (the client's
        // "prefill" phase — its ttft), then emit the
        // `chat.completion.chunk` events (a content chunk per token, a
        // `finish_reason` chunk, then `[DONE]`) at a fixed cadence.
        let guard = InFlightGuard::enter(state.inner.clone());
        tokio::time::sleep(PREFILL).await;
        let usage = req
            .stream_options
            .as_ref()
            .is_some_and(|o| o.include_usage)
            .then_some((prompt_tokens, cached));
        let events = paced_chunks(&id, &model, created, n, usage, thinking);
        Sse::new(PacedSse::new(state.inner.clone(), n, guard, events)).into_response()
    } else {
        // Non-streaming: hold the request in flight for the "generation"
        // time (prefill + per-token decode), then the single JSON body.
        let guard = InFlightGuard::enter(state.inner.clone());
        tokio::time::sleep(generation_time(n)).await;
        state.inner.completed.lock().unwrap().push(n);
        drop(guard);
        Json(completion_json(&id, &model, created, &req, n, thinking)).into_response()
    }
}

// ── the wire shapes (mimicking the server's serializations) ──────────────

/// OpenAI's error body (`{"error": {message, type, code}}`), the server's
/// `submit_error` / `bad_request` shape.
fn error_response(status: StatusCode, code: &str, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({
            "error": { "message": message, "type": code, "code": code },
        })),
    )
        .into_response()
}

/// The mock's prompt-token figure: the last message's whitespace word
/// count (the server's real figure is the templated token count).
fn prompt_token_count(req: &CompletionsRequest) -> u32 {
    req.messages
        .last()
        .map(|m| m.content.split_whitespace().count() as u32)
        .unwrap_or(0)
}

/// The non-streaming completion response (the server's `chat.completion`
/// shape, the `usage` figures, the `message` content).
fn completion_json(
    id: &str,
    model: &str,
    created: u64,
    req: &CompletionsRequest,
    n: u32,
    thinking: bool,
) -> serde_json::Value {
    let prompt_tokens = prompt_token_count(req);
    let text = (0..n).map(token_text).collect::<Vec<_>>().join(" ");
    // A thinking turn whose budget ran out mid-thought: the whole generation
    // is on `reasoning_content` and `content` is the empty string, exactly
    // what the server sends (GitHub #137).
    let message = if thinking {
        serde_json::json!({ "role": "assistant", "content": "", "reasoning_content": text })
    } else {
        serde_json::json!({ "role": "assistant", "content": text })
    };
    serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": n,
            "total_tokens": prompt_tokens + n,
        },
    })
}

/// One SSE content chunk (the server's `Chunk` shape: a token's delta).
fn content_chunk(id: &str, model: &str, created: u64, content: &str) -> Event {
    Event::default().data(
        serde_json::to_string(&serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{ "index": 0, "delta": { "content": content }, "finish_reason": null }],
        }))
        .expect("chunk serializes"),
    )
}

/// One SSE thinking chunk (the server's `delta.reasoning_content` shape —
/// the reasoning and content channels are never sent on the same chunk,
/// GitHub #68).
fn reasoning_chunk(id: &str, model: &str, created: u64, reasoning: &str) -> Event {
    Event::default().data(
        serde_json::to_string(&serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": { "content": "", "reasoning_content": reasoning },
                "finish_reason": null,
            }],
        }))
        .expect("chunk serializes"),
    )
}

/// One complete tool call as its own chunk (the server's `tool_call_chunk`
/// shape): a delta carrying `tool_calls` and no text on either channel.
fn tool_call_chunk(id: &str, model: &str, created: u64) -> Event {
    Event::default().data(
        serde_json::to_string(&serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": { "content": "", "tool_calls": [{
                    "index": 0,
                    "id": "call_0",
                    "type": "function",
                    "function": { "name": "read_file", "arguments": "{\"path\":\"src/lib.rs\"}" },
                }] },
                "finish_reason": null,
            }],
        }))
        .expect("chunk serializes"),
    )
}

/// The SSE final chunk (the server's finish chunk: an empty `delta`
/// (serialized as `{}`) + the engine's `finish_reason`).
fn finish_chunk(id: &str, model: &str, created: u64, reason: &str) -> Event {
    Event::default().data(
        serde_json::to_string(&serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{ "index": 0, "delta": {}, "finish_reason": reason }],
        }))
        .expect("chunk serializes"),
    )
}

/// The SSE terminal marker (`data: [DONE]`).
fn done_event() -> Event {
    Event::default().data("[DONE]")
}

/// The trailing usage chunk (`stream_options.include_usage`): empty
/// `choices`, populated `usage`. `cached` rides along as OpenAI's
/// `prompt_tokens_details.cached_tokens` when the mock is set to serve a
/// prefix cache; a real engine without one simply omits the field.
fn usage_chunk(
    id: &str,
    model: &str,
    created: u64,
    prompt_tokens: u32,
    cached: Option<u32>,
    n: u32,
) -> Event {
    let mut usage = serde_json::json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": n,
        "total_tokens": prompt_tokens + n,
    });
    if let Some(cached) = cached {
        usage["prompt_tokens_details"] = serde_json::json!({ "cached_tokens": cached });
    }
    Event::default().data(
        serde_json::to_string(&serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": usage,
        }))
        .expect("chunk serializes"),
    )
}

/// The SSE events for a request: a chunk per generated token, the finish
/// chunk (empty delta + `finish_reason`), then the `[DONE]` marker. The
/// stream (`PacedSse`) emits them at the mock decode cadence; the prefill wait
/// happens in the handler, before the stream is built.
///
/// With `thinking` set, every token goes out on `reasoning_content` and a
/// tool call closes the turn — a stream with no `delta.content` chunk in it
/// at all (GitHub #137).
fn paced_chunks(
    id: &str,
    model: &str,
    created: u64,
    n: u32,
    usage: Option<(u32, Option<u32>)>,
    thinking: bool,
) -> VecDeque<Event> {
    let mut events = VecDeque::new();
    for i in 0..n {
        events.push_back(if thinking {
            reasoning_chunk(id, model, created, &token_text(i))
        } else {
            content_chunk(id, model, created, &token_text(i))
        });
    }
    if thinking {
        events.push_back(tool_call_chunk(id, model, created));
    }
    events.push_back(finish_chunk(
        id,
        model,
        created,
        if thinking { "tool_calls" } else { "stop" },
    ));
    // The trailing usage chunk goes right before `[DONE]`, the way the
    // server sends it.
    if let Some((prompt_tokens, cached)) = usage {
        events.push_back(usage_chunk(id, model, created, prompt_tokens, cached, n));
    }
    events.push_back(done_event());
    events
}

/// The "generation" time of a non-streaming request (prefill + per-token
/// decode), matching the paced SSE schedule.
fn generation_time(n: u32) -> Duration {
    PREFILL + TOKEN * n
}

/// The Unix epoch seconds (the server's `created` / `created_at` fields).
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ── in-flight tracking (the concurrency proof) ───────────────────────────

/// RAII in-flight counter: enters on construction (incrementing the count
/// and updating the peak), leaves on drop (decrementing). Held for the
/// request's whole in-flight window (the handler's "generation" time for
/// non-streaming; the SSE stream's life for streaming).
struct InFlightGuard(Arc<Inner>);

impl InFlightGuard {
    fn enter(inner: Arc<Inner>) -> Self {
        let current = inner.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        let mut peak = inner.peak_in_flight.load(Ordering::SeqCst);
        while current > peak {
            match inner.peak_in_flight.compare_exchange_weak(
                peak,
                current,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
        Self(inner)
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

// ── the paced SSE stream ──────────────────────────────────────────────────

/// The paced SSE stream: emits the pre-built events at [`TOKEN`] cadence
/// (the prefill wait happens in the handler, before the stream is built).
/// The in-flight guard is released when the stream ends (a normal end or a
/// drop).
struct PacedSse {
    inner: Arc<Inner>,
    n: u32,
    events: VecDeque<Event>,
    delay: Option<Pin<Box<tokio::time::Sleep>>>,
    guard: Option<InFlightGuard>,
}

impl PacedSse {
    fn new(
        inner: Arc<Inner>,
        n: u32,
        guard: InFlightGuard,
        events: VecDeque<Event>,
    ) -> Self {
        Self {
            inner,
            n,
            events,
            delay: None,
            guard: Some(guard),
        }
    }
}

impl Stream for PacedSse {
    type Item = Result<Event, Infallible>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let me = self.get_mut();
        if let Some(delay) = &mut me.delay {
            if delay.as_mut().poll(cx).is_pending() {
                return Poll::Pending;
            }
            me.delay = None;
        }
        if me.events.is_empty() {
            // The stream is done: record the completion and release the
            // in-flight guard (its drop decrements the count).
            me.inner.completed.lock().unwrap().push(me.n);
            me.guard.take();
            return Poll::Ready(None);
        }
        let event = me.events.pop_front().expect("non-empty");
        if !me.events.is_empty() {
            me.delay = Some(Box::pin(tokio::time::sleep(TOKEN)));
        }
        Poll::Ready(Some(Ok(event)))
    }
}

impl Drop for PacedSse {
    fn drop(&mut self) {
        // If the stream is aborted mid-way (the client drops the
        // connection), release the in-flight guard (a normal end already
        // released it — `take()` is a no-op then).
        self.guard.take();
    }
}
