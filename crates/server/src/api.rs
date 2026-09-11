//! The OpenAI-compatible HTTP surface (server-01): routes, request /
//! response schemas, handlers.
//!
//! Endpoints (localhost, no auth — `docs/design/ignis-v1.md` §2):
//! - `GET /v1/models` — the loaded model.
//! - `POST /v1/chat/completions` — chat completions, streaming (SSE) and
//!   non-streaming; routes into the core scheduler and streams tokens back
//!   as they are generated.
//! - `POST /v1/responses` — the OpenAI responses API (non-streaming in v1;
//!   a `stream: true` request is rejected with a 400).
//!
//! Error shape: OpenAI's `{"error": {message, type, code}}` body with the
//! matching status (400 bad request, 404 unknown model, 413 oversized
//! request, 503 engine full, 504 the engine did not finish the request in
//! the timeout).

use std::collections::VecDeque;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::extract::{Request, State};
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_core::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use tower_http::trace::{MakeSpan, TraceLayer};

use ignis_core::{
    DecodeParams, FinishReason, RequestClass, RequestId, RequestInput, SchedEvent, SubmitError,
};

use crate::Server;
use crate::decoder::{Channel, OutputDecoder};
use crate::engine::{Engine, EventStream, collect_tokens};
use crate::template::{ChatMessage, TemplateProvider};
use crate::thinking::{
    self, ThinkingDefaults, ThinkingError, ThinkingOptions, ThinkingRequestFields,
};
use crate::toolcall::{ToolCall as ScannedToolCall, ToolCallScanner, ToolEvent};

/// Build the OpenAI router for `server` (the axum state it serves behind).
///
/// `TraceLayer` (GitHub #81, ADR 0012) is the HTTP-ingress root span for
/// every request: `tower-http`'s well-tested span-per-request middleware,
/// not a hand-rolled equivalent. [`RootSpanMaker`] declares its
/// `request_id` field `Empty` at creation — the scheduler has not assigned
/// one yet at ingress — and the handler records it once
/// [`ignis_core::Scheduler::submit`] returns one, so every log record
/// emitted from inside the span (including this handler's own tail) gets
/// the request's real `trace_id` (`ignis_logging::trace_context`).
pub fn router(state: Arc<Server>) -> Router {
    Router::new()
        .route("/v1/models", get(list_models).options(cors_preflight))
        .route(
            "/v1/chat/completions",
            post(chat_completions).options(cors_preflight),
        )
        .route("/v1/responses", post(responses_api).options(cors_preflight))
        .layer(middleware::from_fn(cors_headers))
        .layer(TraceLayer::new_for_http().make_span_with(RootSpanMaker))
        .with_state(state)
}

/// The HTTP-ingress root span's shape: `request_id` is declared `Empty` —
/// unknown at ingress — and recorded once the request is admitted into the
/// scheduler (see `chat_completions`/`responses_api`). A request that never
/// reaches submission (a 400 before it, a CORS preflight, `GET
/// /v1/models`) simply never records it, so its span (and anything logged
/// under it) carries no `trace_id` — there is no request to correlate yet,
/// and this module never fabricates one (spec §19).
#[derive(Clone, Copy)]
struct RootSpanMaker;

impl<B> MakeSpan<B> for RootSpanMaker {
    fn make_span(&mut self, request: &axum::http::Request<B>) -> tracing::Span {
        tracing::info_span!(
            "ignis.http.request",
            method = %request.method(),
            path = %request.uri().path(),
            request_id = tracing::field::Empty,
        )
    }
}

/// Answers a CORS preflight request with no body; `cors_headers` attaches
/// the `Access-Control-Allow-*` headers below.
async fn cors_preflight() -> StatusCode {
    StatusCode::OK
}

/// Adds permissive CORS headers to every response so the API is reachable
/// from browser-based clients on other origins.
async fn cors_headers(req: Request, next: Next) -> Response {
    let mut res = next.run(req).await;
    let headers = res.headers_mut();
    headers.insert("Access-Control-Allow-Origin", HeaderValue::from_static("*"));
    headers.insert(
        "Access-Control-Allow-Methods",
        HeaderValue::from_static("GET, POST, OPTIONS"),
    );
    headers.insert(
        "Access-Control-Allow-Headers",
        HeaderValue::from_static("Content-Type, Authorization"),
    );
    res
}

/// The request's model, the templated prompt tokens, and the prompt-token
/// count (the usage figures) — one shared build path for both completion
/// endpoints.
fn build_request(
    server: &Server,
    model: Option<String>,
    messages: &[ChatMessage],
    params: DecodeParams,
    thinking: &ThinkingOptions,
) -> (RequestInput, String, u32) {
    // `model` is the model the request names; `None` (or a blank) falls
    // back to the loaded model. A model the engine does not load is
    // rejected at submit with a 404 (OpenAI's `model_not_found`).
    let model = model
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| server.engine.model_id());
    // The template seam: the artifact's frontend object set (artifact-02)
    // replaces this built-in provider through the same constructor
    // injection (v1 placeholder: deterministic word-hash tokens).
    let tokens = server.template.apply_chat_template(messages, thinking);
    let prompt_tokens = tokens.len() as u32;
    let input = RequestInput {
        model: model.clone(),
        tokens,
        params,
    };
    (input, model, prompt_tokens)
}

/// The sampling fields accepted by chat completions. Values stay as JSON at
/// the wire boundary so type, integer-width, and narrowing failures all use
/// the same OpenAI-shaped sampling error instead of Axum's generic rejection.
#[derive(Clone, Default, Deserialize)]
struct SamplingRequestFields {
    temperature: Option<JsonValue>,
    top_p: Option<JsonValue>,
    /// An ignis extension, not an OpenAI Chat Completions parameter.
    top_k: Option<JsonValue>,
    presence_penalty: Option<JsonValue>,
    frequency_penalty: Option<JsonValue>,
    seed: Option<JsonValue>,
}

impl SamplingRequestFields {
    /// `ignore_eos` is an ignis extension for bounded measurement streams.
    /// It is refused without a `max_tokens`, because the two together are
    /// what keeps such a request bounded: with neither an EOS nor a cap,
    /// a non-streaming request has nothing left to stop it, and only the
    /// streaming path cancels on client disconnect.
    fn resolve(self, max_tokens: Option<u32>, ignore_eos: bool) -> Result<DecodeParams, String> {
        let temperature = bounded_f32(
            "temperature",
            number("temperature", self.temperature, 0.0)?,
            0.0,
            2.0,
        )?;
        let top_p = bounded_f32("top_p", number("top_p", self.top_p, 1.0)?, 0.0, 1.0)?;
        let presence_penalty = bounded_f32(
            "presence_penalty",
            number("presence_penalty", self.presence_penalty, 0.0)?,
            -2.0,
            2.0,
        )?;
        let frequency_penalty = bounded_f32(
            "frequency_penalty",
            number("frequency_penalty", self.frequency_penalty, 0.0)?,
            -2.0,
            2.0,
        )?;
        let top_k = signed_integer(
            "top_k is an ignis extension and must be an integer",
            self.top_k,
            0,
        )?;
        if !(0..=20).contains(&top_k) {
            return Err(format!(
                "top_k is an ignis extension and must be between 0 and 20 inclusive; 0 selects ignis's 20-candidate sampler cap (got {top_k})"
            ));
        }

        if ignore_eos && max_tokens.is_none() {
            return Err(
                "ignore_eos is an ignis extension for bounded measurement streams and requires max_tokens"
                    .into(),
            );
        }

        if temperature == 0.0
            && (top_p != 1.0
                || (top_k != 0 && top_k != 20)
                || presence_penalty != 0.0
                || frequency_penalty != 0.0)
        {
            return Err(
                "temperature must be greater than 0 when top_p, the ignis top_k extension, presence_penalty, or frequency_penalty would otherwise be ignored by greedy sampling"
                    .into(),
            );
        }

        Ok(DecodeParams {
            max_tokens,
            temperature,
            top_p,
            top_k: top_k as i32,
            presence_penalty,
            frequency_penalty,
            // The leaf keys its counter-based RNG with all 64 bits. Casting
            // preserves the complete signed OpenAI seed domain bit-for-bit.
            seed: signed_integer("seed must be a signed 64-bit integer", self.seed, 0)? as u64,
            ignore_eos,
        })
    }
}

fn number(name: &str, value: Option<JsonValue>, default: f64) -> Result<f64, String> {
    match value {
        None => Ok(default),
        Some(JsonValue::Number(value)) => value
            .as_f64()
            .ok_or_else(|| format!("{name} must be a finite number")),
        Some(value) => Err(format!("{name} must be a number (got {value})")),
    }
}

fn signed_integer(message: &str, value: Option<JsonValue>, default: i64) -> Result<i64, String> {
    match value {
        None => Ok(default),
        Some(JsonValue::Number(value)) => value.as_i64().ok_or_else(|| message.to_owned()),
        Some(_) => Err(message.to_owned()),
    }
}

fn bounded_f32(name: &str, value: f64, min: f64, max: f64) -> Result<f32, String> {
    if value.is_finite() && (min..=max).contains(&value) {
        let narrowed = value as f32;
        if value != 0.0 && narrowed == 0.0 {
            Err(format!(
                "{name} magnitude is too small to be represented by the engine (got {value})"
            ))
        } else {
            Ok(narrowed)
        }
    } else {
        Err(format!(
            "{name} must be between {min} and {max} inclusive (got {value})"
        ))
    }
}

fn invalid_sampling_parameter(message: impl Into<String>) -> Response {
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        "invalid_sampling_parameter",
        message,
    )
}

/// Resolve one request's thinking controls, or the OpenAI-shaped 400 to
/// return instead. Validation errors and capability errors carry different
/// `code`s (the client's mistake vs. the loaded model's limitation).
fn resolve_thinking(
    server: &Server,
    fields: ThinkingRequestFields<'_>,
) -> Result<ThinkingOptions, Response> {
    let defaults = ThinkingDefaults {
        enable_thinking: server.default_enable_thinking,
        reasoning_effort: server.default_reasoning_effort,
    };
    let capabilities = server.template.thinking_capabilities();
    thinking::resolve(fields, &defaults, &capabilities).map_err(|err| match err {
        ThinkingError::Validation(message) => bad_request(&message),
        ThinkingError::Capability(message) => error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "reasoning_effort_unsupported",
            message,
        ),
    })
}

/// Split a finished generation's tokens into `reasoning_content` /
/// `content` (GitHub #68): feeds the whole token list through the same
/// [`OutputDecoder`] the streaming path uses one token at a time, so the
/// two modes agree by construction (spec story 24).
fn split_reasoning(
    template: &dyn TemplateProvider,
    tokens: &[ignis_core::TokenId],
    thinking: &ThinkingOptions,
) -> (Option<String>, String) {
    let starts_in_reasoning = template.decoder_starts_in_reasoning(thinking);
    let mut decoder = OutputDecoder::new(template.token_decoder(), starts_in_reasoning);
    let mut deltas = decoder.push(tokens);
    deltas.extend(decoder.finish());
    let mut reasoning = String::new();
    let mut content = String::new();
    for delta in deltas {
        match delta.channel {
            Channel::Reasoning => reasoning.push_str(&delta.text),
            Channel::Content => content.push_str(&delta.text),
        }
    }
    (
        if reasoning.is_empty() {
            None
        } else {
            Some(reasoning)
        },
        content,
    )
}

/// [`split_reasoning`] plus tool-call extraction (GitHub #121): the content
/// channel's text is fed through a fresh [`ToolCallScanner`] so the
/// non-streaming path parses tool calls with exactly the same rules as the
/// streaming path (a call left open by a truncated generation is dropped,
/// never returned half-written — acceptance criterion 3).
fn split_reasoning_and_tools(
    template: &dyn TemplateProvider,
    tokens: &[ignis_core::TokenId],
    thinking: &ThinkingOptions,
) -> (Option<String>, String, Vec<ScannedToolCall>) {
    let (reasoning, content_text) = split_reasoning(template, tokens, thinking);
    let mut scanner = ToolCallScanner::new();
    let mut events = scanner.feed(&content_text);
    events.extend(scanner.finish());
    let mut content = String::new();
    let mut calls = Vec::new();
    for event in events {
        match event {
            ToolEvent::Content(text) => content.push_str(&text),
            ToolEvent::Call(call) => calls.push(call),
        }
    }
    (reasoning, content, calls)
}

/// Map a [`SubmitError`] from the engine's submit to the OpenAI-shaped
/// error response (404 unknown model, 413 oversized, 503 engine full).
fn submit_error(server: &Server, err: SubmitError) -> Response {
    match err {
        SubmitError::UnknownModel(m) => error_response(
            StatusCode::NOT_FOUND,
            "model_not_found",
            "model_not_found",
            format!("unknown model: {m} (loaded: {})", server.engine.model_id()),
        ),
        SubmitError::Oversized => error_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            "request_too_large",
            "request_too_large",
            "request exceeds the engine's KV pool (prompt + max_tokens exceeds the pool)",
        ),
        SubmitError::Full => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "engine_full",
            "engine_full",
            "the engine cannot admit the request right now (all lanes in use); retry",
        ),
    }
}

/// A 400 with the OpenAI error body.
fn bad_request(message: &str) -> Response {
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        "invalid_request_error",
        message,
    )
}

/// The OpenAI error body (`{"error": {message, type, code}}`).
fn error_response(
    status: StatusCode,
    type_: &str,
    code: &str,
    message: impl Into<String>,
) -> Response {
    (
        status,
        Json(ApiError {
            error: ErrorBody {
                message: message.into(),
                r#type: type_.into(),
                code: code.into(),
            },
        }),
    )
        .into_response()
}

/// The `504` body for a request `collect_tokens` gave up on: names the
/// timeout that fired (GitHub #95) so an operator reading the error knows
/// what to raise with `--request-timeout`/`IGNIS_REQUEST_TIMEOUT`, rather
/// than suspecting a wedged engine when a healthy one just needed longer.
fn request_timeout_message(timeout: std::time::Duration) -> String {
    format!(
        "the request did not complete within the server's {}s timeout (the engine may be wedged) — raise it with --request-timeout/IGNIS_REQUEST_TIMEOUT",
        timeout.as_secs()
    )
}

/// Map the engine's [`FinishReason`] to the OpenAI `finish_reason` string
/// (GitHub #61 / P1-25): `stop` on the model's own EOS token, `length` on
/// `max_tokens` or the engine's reservation cap. `pub(crate)` since P3-06:
/// `telemetry.rs`'s `ignis.request.done` event reuses this exact mapping so
/// the request log and the HTTP response never disagree on the string.
pub(crate) fn finish_reason_str(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
    }
}

/// The OpenAI `finish_reason`, tool-calls aware (GitHub #121). A generation
/// that stopped cleanly (its own EOS) after emitting at least one complete
/// tool call reports `"tool_calls"`, matching what a real agent client
/// branches on to decide whether to execute a call rather than just render
/// text. A call left open when generation stopped (`mid_call`) never earns
/// that label — it was dropped whole by [`ToolCallScanner`], so reporting
/// `"tool_calls"` here would tell the client to run a call that was never
/// actually delivered (acceptance criterion 3); the plain `stop`/`length`
/// reason is reported instead, same as if no call had been attempted.
pub(crate) fn resolve_finish_reason(
    reason: FinishReason,
    any_calls: bool,
    mid_call: bool,
) -> &'static str {
    if !mid_call && any_calls && reason == FinishReason::Stop {
        "tool_calls"
    } else {
        finish_reason_str(reason)
    }
}

/// GitHub #70 via #121 acceptance criterion 5: a generation that reasoned
/// but never produced content or a tool call must not pass silently —
/// report it explicitly rather than let it look, on the wire, like an
/// ordinary short (or empty) answer. Shared by both response paths so the
/// message can't drift between them (each path computes `all_reasoning`
/// its own way — a whole-string check non-streaming, incremental flags
/// while streaming — since neither has the other's representation of the
/// generation to reuse).
fn report_if_all_reasoning_no_content(id: &str, all_reasoning: bool, finish_reason: &'static str) {
    if all_reasoning {
        tracing::warn!(
            id,
            finish_reason,
            "generation produced reasoning but no content or tool call \
             (token budget exhausted before an answer began)"
        );
    }
}

/// The Unix epoch seconds (OpenAI's `created` / `created_at` fields).
fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ── GET /v1/models ────────────────────────────────────────────────────────

/// `GET /v1/models` — the loaded model (v1: a single model).
async fn list_models(State(server): State<Arc<Server>>) -> Json<ModelList> {
    let id = server.engine.model_id();
    Json(ModelList {
        object: "list",
        data: vec![ModelInfo {
            id,
            object: "model",
            owned_by: "ignis",
        }],
    })
}

/// The models list envelope.
#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelInfo>,
}

/// One model entry.
#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

// ── POST /v1/chat/completions ─────────────────────────────────────────────

/// A chat-completions request (OpenAI wire shape; unknown fields are
/// ignored — serde's default).
#[derive(Deserialize)]
struct ChatCompletionsRequest {
    model: Option<String>,
    /// The conversation (role + content messages). Must be non-empty.
    messages: Vec<ChatMessage>,
    /// `true` → stream the completion as SSE chunks.
    #[serde(default)]
    stream: bool,
    /// Streaming-only options. `include_usage: true` appends the trailing
    /// usage chunk (empty `choices`, populated `usage`) before `[DONE]`.
    stream_options: Option<StreamOptions>,
    max_tokens: Option<u32>,
    /// An ignis extension: keep decoding past the model's EOS token. Used
    /// by the G3 inter-token-latency lanes, which are ended by the
    /// measurement window rather than by the model. Requires `max_tokens`.
    #[serde(default)]
    ignore_eos: bool,
    #[serde(flatten)]
    sampling: SamplingRequestFields,
    /// The thinking controls (GitHub #68) — kept as raw JSON so the wire
    /// contract's tri-state semantics (absent / `null` / a bad type) and
    /// its own error messages are decided by `thinking::resolve`, not by
    /// serde's generic type-mismatch error.
    enable_thinking: Option<JsonValue>,
    reasoning_effort: Option<JsonValue>,
    preserve_thinking: Option<JsonValue>,
    chat_template_kwargs: Option<JsonValue>,
}

#[derive(Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

/// `POST /v1/chat/completions` — non-streaming and streaming (SSE).
///
/// The request routes into the core scheduler (the request's event stream
/// carries its generated tokens) and the handler either collects them into
/// the single-response JSON (non-streaming) or wraps the stream in the
/// `chat.completion.chunk` SSE shape (streaming).
async fn chat_completions(
    State(server): State<Arc<Server>>,
    Json(req): Json<ChatCompletionsRequest>,
) -> Response {
    if req.messages.is_empty() {
        return bad_request("messages must not be empty");
    }
    let params = match req.sampling.resolve(req.max_tokens, req.ignore_eos) {
        Ok(params) => params,
        Err(message) => return invalid_sampling_parameter(message),
    };
    let thinking = match resolve_thinking(
        &server,
        ThinkingRequestFields {
            enable_thinking: req.enable_thinking.as_ref(),
            reasoning_effort: req.reasoning_effort.as_ref(),
            preserve_thinking: req.preserve_thinking.as_ref(),
            chat_template_kwargs: req.chat_template_kwargs.as_ref(),
        },
    ) {
        Ok(t) => t,
        Err(response) => return response,
    };
    let (input, model, prompt_tokens) =
        build_request(&server, req.model, &req.messages, params, &thinking);
    let (request_id, mut stream) = match server.engine.submit(input, RequestClass::Interactive).await
    {
        Ok(x) => x,
        Err(err) => return submit_error(&server, err),
    };
    // GitHub #81 / ADR 0012: the HTTP root span declared `request_id`
    // `Empty` at ingress (`RootSpanMaker`) since the scheduler had not
    // assigned one yet; record it now so every log record for the rest of
    // this span's life carries the request's real `trace_id`.
    tracing::Span::current().record("request_id", request_id);
    let id = format!("chatcmpl-{request_id}");
    let created = now();

    if req.stream {
        // The SSE response: the request's event stream wrapped in the
        // `chat.completion.chunk` shape (a `[DONE]` marker terminates).
        let include_usage = req.stream_options.is_some_and(|o| o.include_usage);
        let starts_in_reasoning = server.template.decoder_starts_in_reasoning(&thinking);
        return Sse::new(ChunkStream::new(
            stream,
            CancelOnDrop::new(server.engine.clone(), request_id),
            id,
            created,
            model,
            OutputDecoder::new(server.template.token_decoder(), starts_in_reasoning),
            prompt_tokens,
            include_usage,
        ))
        .into_response();
    }
    // Non-streaming: collect the request's tokens to its completion (a
    // timeout guards a wedged engine from hanging the client).
    match collect_tokens(&mut stream, server.request_timeout).await {
        Ok((tokens, reason)) => {
            let (reasoning_content, content, tool_calls) =
                split_reasoning_and_tools(server.template.as_ref(), &tokens, &thinking);
            let completion_tokens = tokens.len() as u32;
            let finish_reason = resolve_finish_reason(reason, !tool_calls.is_empty(), false);
            report_if_all_reasoning_no_content(
                &id,
                reasoning_content.as_deref().is_some_and(|r| !r.is_empty())
                    && content.is_empty()
                    && tool_calls.is_empty(),
                finish_reason,
            );
            let tool_calls = if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls.into_iter().map(ToolCallOut::from).collect())
            };
            Json(ChatCompletion {
                id,
                object: "chat.completion",
                created,
                model,
                choices: vec![CompletionChoice {
                    index: 0,
                    message: AssistantMessage {
                        role: "assistant",
                        reasoning_content,
                        content,
                        tool_calls,
                    },
                    finish_reason,
                }],
                usage: Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens.saturating_add(completion_tokens),
                },
            })
            .into_response()
        }
        Err(_) => error_response(
            StatusCode::GATEWAY_TIMEOUT,
            "request_timeout",
            "request_timeout",
            request_timeout_message(server.request_timeout),
        ),
    }
}

/// The non-streaming completion response (OpenAI wire shape).
#[derive(Serialize)]
struct ChatCompletion {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct CompletionChoice {
    index: u8,
    message: AssistantMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct AssistantMessage {
    role: &'static str,
    /// The model's thinking trace (GitHub #68) — omitted entirely (not
    /// serialized as an empty string) when thinking produced no reasoning,
    /// so a thinking-disabled response is structurally distinct from a
    /// thinking-enabled one that happened to reason not at all.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    content: String,
    /// Tool calls parsed out of the content channel (GitHub #121) — absent
    /// when the generation made none.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallOut>>,
}

/// One tool call, non-streaming wire shape (OpenAI: `message.tool_calls[]`).
#[derive(Serialize)]
struct ToolCallOut {
    id: String,
    r#type: &'static str,
    function: FunctionOut,
}

#[derive(Serialize)]
struct FunctionOut {
    name: String,
    /// A JSON-encoded object, e.g. `{"path":"a.txt"}` (never a
    /// half-written fragment — GitHub #121 acceptance criterion 3: an
    /// interrupted call is dropped entirely by `ToolCallScanner`, never
    /// surfaced here truncated).
    arguments: String,
}

impl From<ScannedToolCall> for ToolCallOut {
    fn from(call: ScannedToolCall) -> Self {
        ToolCallOut {
            id: call.id,
            r#type: "function",
            function: FunctionOut {
                name: call.name,
                arguments: call.arguments,
            },
        }
    }
}

/// The usage figures (the prompt's templated-token count + the generated
/// tokens).
#[derive(Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// One SSE chunk (the `chat.completion.chunk` shape): a token delta or the
/// final `finish_reason` chunk (an empty `delta`).
#[derive(Serialize)]
struct Chunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}

#[derive(Serialize)]
struct ChunkChoice {
    index: u8,
    delta: Delta,
    finish_reason: Option<&'static str>,
}

/// The token delta. An empty `content`, absent `reasoning_content` and
/// absent `tool_calls` serialize to `{}` (OpenAI's final chunk shape).
#[derive(Serialize, Default)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "String::is_empty")]
    content: String,
    /// A tool call's id + name + full arguments (GitHub #121) — delivered
    /// as one complete delta per call rather than character-by-character
    /// (the source is a closed XML block, not a token-streamed JSON
    /// fragment, so there is nothing partial left to stream by the time a
    /// call is known at all).
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

/// One streamed tool call delta (OpenAI: `delta.tool_calls[]`), keyed by
/// `index` — the field a real agent client's reassembly buffer groups on,
/// stable across chunks and never reused by two different calls (GitHub
/// #121 acceptance criterion 2).
#[derive(Serialize)]
struct ToolCallDelta {
    index: usize,
    id: String,
    r#type: &'static str,
    function: FunctionOut,
}

impl From<ScannedToolCall> for ToolCallDelta {
    fn from(call: ScannedToolCall) -> Self {
        ToolCallDelta {
            index: call.index,
            id: call.id,
            r#type: "function",
            function: FunctionOut {
                name: call.name,
                arguments: call.arguments,
            },
        }
    }
}

/// The streaming half of a chat completion: the request's event stream
/// (`EventStream`) wrapped in the `chat.completion.chunk` SSE shape.
///
/// The chunk shape (OpenAI): each generated token is one
/// `data: {"choices":[{"delta":{"content":...}}]}` line; the final chunk
/// carries `finish_reason: "stop"` and an empty `delta`; a terminal
/// `data: [DONE]` line ends the stream.
struct ChunkStream {
    /// The request's event stream (closes on the request's completion).
    stream: EventStream,
    /// Aborts the request if this body is dropped before it completed.
    cancel: CancelOnDrop,
    id: String,
    created: u64,
    model: String,
    /// The incremental reasoning/content decoder (GitHub #68) — owns the
    /// byte- and marker-splitting state across this request's whole token
    /// stream (`crate::decoder::OutputDecoder`).
    decoder: OutputDecoder,
    /// Scans the decoder's `Content` channel for `<tool_call>` blocks
    /// (GitHub #121) — reasoning text never passes through it.
    tool_scanner: ToolCallScanner,
    prompt_tokens: u32,
    /// `stream_options.include_usage` — gates the trailing usage chunk
    /// (OpenAI only sends it when the client opts in).
    include_usage: bool,
    /// Fully-formed events queued ahead of the next stream poll: a single
    /// generated token can yield zero, one, or two decoder deltas (a
    /// reasoning tail plus a content head, right at the `</think>`
    /// marker), and the `Done` event queues its flush deltas, the
    /// `finish_reason` chunk, and (opt-in) the usage chunk together, in
    /// that order, ahead of `[DONE]`.
    pending: VecDeque<Event>,
    /// The `[DONE]` marker has been emitted (exactly once, at the end).
    done_sent: bool,
    /// At least one non-empty `Reasoning` delta was ever queued (GitHub
    /// #70 via #121 acceptance criterion 5).
    emitted_reasoning: bool,
    /// At least one `Content` delta or tool call was ever queued.
    emitted_content_or_call: bool,
}

/// Cancels its request when dropped before the request has completed. An SSE
/// client that hangs up mid-generation would otherwise leave its lane and KV
/// reservation generating to `max_tokens` for nobody.
struct CancelOnDrop {
    engine: Engine,
    request: RequestId,
    completed: bool,
}

impl CancelOnDrop {
    fn new(engine: Engine, request: RequestId) -> Self {
        Self { engine, request, completed: false }
    }

    /// The request reached its own terminal event: dropping is now a
    /// clean end of stream, not a disconnect.
    fn completed(&mut self) {
        self.completed = true;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if !self.completed {
            self.engine.cancel(self.request);
        }
    }
}

impl ChunkStream {
    // The chunk envelope's fields are all per-request and all needed to
    // shape a `chat.completion.chunk`; grouping them would only rename the
    // argument list.
    #[allow(clippy::too_many_arguments)]
    fn new(
        stream: EventStream,
        cancel: CancelOnDrop,
        id: String,
        created: u64,
        model: String,
        decoder: OutputDecoder,
        prompt_tokens: u32,
        include_usage: bool,
    ) -> Self {
        Self {
            stream,
            cancel,
            id,
            created,
            model,
            decoder,
            tool_scanner: ToolCallScanner::new(),
            prompt_tokens,
            include_usage,
            pending: VecDeque::new(),
            done_sent: false,
            emitted_reasoning: false,
            emitted_content_or_call: false,
        }
    }

    /// Routes one decoder delta to its SSE chunk(s): a `Reasoning` delta
    /// streams straight through; a `Content` delta is fed to the tool-call
    /// scanner first, since it may resolve to plain content, a complete
    /// tool call, or nothing yet (GitHub #121).
    fn queue_decoder_delta(&mut self, delta: crate::decoder::Delta) {
        match delta.channel {
            Channel::Reasoning => {
                if !delta.text.is_empty() {
                    self.emitted_reasoning = true;
                }
                let event = self.delta_chunk(delta);
                self.pending.push_back(event);
            }
            Channel::Content => {
                for tool_event in self.tool_scanner.feed(&delta.text) {
                    self.queue_tool_event(tool_event);
                }
            }
        }
    }

    /// Queues one tool-call-scanner event as its SSE chunk, tracking
    /// `emitted_content_or_call` along the way — both `ToolEvent`
    /// variants only ever carry non-empty payloads (`ToolCallScanner`
    /// never emits an empty one), so every call here is real content.
    fn queue_tool_event(&mut self, event: ToolEvent) {
        self.emitted_content_or_call = true;
        let chunk = match event {
            ToolEvent::Content(text) => self.delta_chunk(crate::decoder::Delta {
                channel: Channel::Content,
                text,
            }),
            ToolEvent::Call(call) => self.tool_call_chunk(call),
        };
        self.pending.push_back(chunk);
    }

    /// One chunk (a decoded delta or the final `finish_reason` chunk).
    fn chunk(&self, delta: Delta, finish_reason: Option<&'static str>) -> Event {
        let chunk = Chunk {
            id: self.id.clone(),
            object: "chat.completion.chunk",
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta,
                finish_reason,
            }],
            usage: None,
        };
        // `serde_json` cannot fail on this (all serializable fields).
        Event::default().data(serde_json::to_string(&chunk).expect("chunk serializes"))
    }

    /// One decoder delta as its SSE chunk (`delta.reasoning_content` or
    /// `delta.content`, never both — GitHub #68).
    fn delta_chunk(&self, delta: crate::decoder::Delta) -> Event {
        let delta = match delta.channel {
            Channel::Reasoning => Delta {
                reasoning_content: Some(delta.text),
                content: String::new(),
                tool_calls: None,
            },
            Channel::Content => Delta {
                reasoning_content: None,
                content: delta.text,
                tool_calls: None,
            },
        };
        self.chunk(delta, None)
    }

    /// One complete tool call as its SSE chunk (GitHub #121): a `delta`
    /// carrying only `tool_calls`, mirroring `delta_chunk`'s shape for the
    /// reasoning/content channels.
    fn tool_call_chunk(&self, call: ScannedToolCall) -> Event {
        self.chunk(
            Delta {
                reasoning_content: None,
                content: String::new(),
                tool_calls: Some(vec![call.into()]),
            },
            None,
        )
    }

    /// The final summary chunk: empty `choices`, populated `usage` (OpenAI's
    /// trailing usage chunk, sent right before `[DONE]`).
    fn usage_chunk(&self, completion_tokens: u32) -> Event {
        let chunk = Chunk {
            id: self.id.clone(),
            object: "chat.completion.chunk",
            created: self.created,
            model: self.model.clone(),
            choices: vec![],
            usage: Some(Usage {
                prompt_tokens: self.prompt_tokens,
                completion_tokens,
                total_tokens: self.prompt_tokens.saturating_add(completion_tokens),
            }),
        };
        Event::default().data(serde_json::to_string(&chunk).expect("chunk serializes"))
    }
}

/// The stream adapter: the request's event stream into the SSE shape.
///
/// Pulls one event per poll (the driver routes events into the stream as
/// they are generated — tokens arrive in generation order); an `Evicted` /
/// `Restored` event for the request is skipped (it does not change the
/// generated-token sequence). A single scheduler event can queue more than
/// one SSE chunk (or none at all, while the decoder is holding back a
/// multi-byte character or a possible `</think>` prefix) — queued chunks
/// drain before the stream is polled again.
impl Stream for ChunkStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(self: Pin<&mut ChunkStream>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // `ChunkStream` is `Unpin` (all fields are `Unpin`), so we can get a
        // plain `&mut` back out of the pinned self.
        let this = self.get_mut();
        loop {
            if let Some(event) = this.pending.pop_front() {
                return Poll::Ready(Some(Ok(event)));
            }
            match this.stream.poll_recv(cx) {
                Poll::Ready(None) => {
                    // Stream closed (the request completed and the engine
                    // removed its route — or the engine stopped): the
                    // `[DONE]` marker ends the SSE stream (exactly once).
                    if !this.done_sent {
                        this.done_sent = true;
                        return Poll::Ready(Some(Ok(Event::default().data("[DONE]"))));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(event)) => match event {
                    SchedEvent::Token { token, .. } => {
                        // The incremental decoder: zero, one, or two deltas
                        // for this token (GitHub #68). Zero means the
                        // decoder is holding back — loop to poll the next
                        // scheduler event rather than returning nothing.
                        for delta in this.decoder.push(&[token]) {
                            this.queue_decoder_delta(delta);
                        }
                    }
                    // The request completed: flush whatever the decoder
                    // and the tool-call scanner held back, then the
                    // finish-reason chunk, then (opt-in) the usage chunk —
                    // all queued ahead of `[DONE]`.
                    SchedEvent::Done { reason, tokens, .. } => {
                        this.cancel.completed();
                        for delta in this.decoder.finish() {
                            this.queue_decoder_delta(delta);
                        }
                        // Read before `finish()` clears it (GitHub #121
                        // acceptance criterion 3): a stop/cancel landing
                        // mid-call must never be reported as a clean
                        // `tool_calls` finish.
                        let mid_call = this.tool_scanner.is_mid_call();
                        for tool_event in this.tool_scanner.finish() {
                            this.queue_tool_event(tool_event);
                        }
                        let any_calls = this.tool_scanner.any_calls();
                        let finish_reason = resolve_finish_reason(reason, any_calls, mid_call);
                        report_if_all_reasoning_no_content(
                            &this.id,
                            this.emitted_reasoning && !this.emitted_content_or_call,
                            finish_reason,
                        );
                        this.pending
                            .push_back(this.chunk(Delta::default(), Some(finish_reason)));
                        if this.include_usage {
                            this.pending.push_back(this.usage_chunk(tokens));
                        }
                    }
                    // Other events for this request (admissions,
                    // evictions, restorations, requeues) do not change the
                    // generated-token sequence — keep draining.
                    _ => {}
                },
            }
        }
    }
}

// ── POST /v1/responses ────────────────────────────────────────────────────

/// A responses-API request (OpenAI wire shape). `input` is a plain string
/// (a single user turn) or a list of messages.
#[derive(Deserialize)]
struct ResponsesRequest {
    input: ResponsesInput,
    model: Option<String>,
    /// The responses API's name for the completion's token cap.
    max_output_tokens: Option<u32>,
    temperature: Option<f32>,
    seed: Option<u64>,
    /// `true` is rejected in v1 (the responses API is non-streaming; a
    // streaming response is a later ticket).
    #[serde(default)]
    stream: bool,
    /// The thinking controls (GitHub #68) — same wire contract as chat
    /// completions, resolved through the same `thinking::resolve` path.
    enable_thinking: Option<JsonValue>,
    reasoning_effort: Option<JsonValue>,
    preserve_thinking: Option<JsonValue>,
    chat_template_kwargs: Option<JsonValue>,
}

/// The responses API's `input` (a string or a message list).
#[derive(Deserialize)]
#[serde(untagged)]
enum ResponsesInput {
    /// A plain string — treated as a single user turn.
    Text(String),
    /// A list of messages (role + content).
    Messages(Vec<ChatMessage>),
}

/// `POST /v1/responses` — the OpenAI responses API (non-streaming in v1).
///
/// The request routes into the core scheduler exactly like a chat
/// completion (the same submit / event-stream path); the response is the
/// responses API's `output`-message shape with the generated text in an
/// `output_text` content part.
async fn responses_api(
    State(server): State<Arc<Server>>,
    Json(req): Json<ResponsesRequest>,
) -> Response {
    if req.stream {
        return bad_request("streaming responses are not supported in v1 (non-streaming only)");
    }
    // `input` → messages: a string is a single user turn; a message list is
    // used as-is (an empty list is a 400).
    let messages = match req.input {
        ResponsesInput::Text(text) => vec![ChatMessage::text("user", text)],
        ResponsesInput::Messages(m) => m,
    };
    if messages.is_empty() {
        return bad_request("input must not be empty");
    }
    let thinking = match resolve_thinking(
        &server,
        ThinkingRequestFields {
            enable_thinking: req.enable_thinking.as_ref(),
            reasoning_effort: req.reasoning_effort.as_ref(),
            preserve_thinking: req.preserve_thinking.as_ref(),
            chat_template_kwargs: req.chat_template_kwargs.as_ref(),
        },
    ) {
        Ok(t) => t,
        Err(response) => return response,
    };
    let (input, model, prompt_tokens) = build_request(
        &server,
        req.model,
        &messages,
        DecodeParams {
            max_tokens: req.max_output_tokens,
            temperature: req.temperature.unwrap_or(0.0),
            seed: req.seed.unwrap_or(0),
            ..DecodeParams::default()
        },
        &thinking,
    );
    let (id, mut stream) = match server.engine.submit(input, RequestClass::Interactive).await {
        Ok(x) => x,
        Err(err) => return submit_error(&server, err),
    };
    // GitHub #81 / ADR 0012: see the matching comment in `chat_completions`.
    tracing::Span::current().record("request_id", id);
    match collect_tokens(&mut stream, server.request_timeout).await {
        Ok((tokens, _reason)) => {
            // The responses API's v1 shape carries no `finish_reason`
            // field (only `status: "completed"`); the stop reason is not
            // surfaced here.
            // GitHub #68: `text` is the content channel only — the
            // reasoning trace is discarded (this endpoint has no field to
            // carry it, and leaking it into `text` is the bug being fixed).
            let (_reasoning_content, text) =
                split_reasoning(server.template.as_ref(), &tokens, &thinking);
            let output_tokens = tokens.len() as u32;
            Json(Responses {
                id: format!("resp_{id}"),
                object: "response",
                created_at: now(),
                model,
                output: vec![ResponseMessage {
                    r#type: "message",
                    id: format!("msg_{id}"),
                    role: "assistant",
                    status: "completed",
                    content: vec![ResponseContent {
                        r#type: "output_text",
                        text,
                        annotations: Vec::new(),
                    }],
                }],
                status: "completed",
                usage: ResponsesUsage {
                    input_tokens: prompt_tokens,
                    output_tokens,
                    total_tokens: prompt_tokens.saturating_add(output_tokens),
                },
            })
            .into_response()
        }
        Err(_) => error_response(
            StatusCode::GATEWAY_TIMEOUT,
            "request_timeout",
            "request_timeout",
            request_timeout_message(server.request_timeout),
        ),
    }
}

/// The non-streaming responses response (the OpenAI responses API shape).
#[derive(Serialize)]
struct Responses {
    id: String,
    object: &'static str,
    created_at: u64,
    model: String,
    output: Vec<ResponseMessage>,
    status: &'static str,
    usage: ResponsesUsage,
}

#[derive(Serialize)]
struct ResponseMessage {
    r#type: &'static str,
    id: String,
    role: &'static str,
    status: &'static str,
    content: Vec<ResponseContent>,
}

#[derive(Serialize)]
struct ResponseContent {
    r#type: &'static str,
    text: String,
    annotations: Vec<()>,
}

#[derive(Serialize)]
struct ResponsesUsage {
    input_tokens: u32,
    output_tokens: u32,
    total_tokens: u32,
}

// ── the error envelope ───────────────────────────────────────────────────

/// The OpenAI error body (`{"error": {...}}`).
#[derive(Serialize)]
struct ApiError {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    message: String,
    r#type: String,
    code: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chunk_serializes_in_the_openai_shape() {
        // Pin the chunk shape (a token delta, no finish_reason): a client
        // that parses the SSE stream must see these exact fields.
        let c = Chunk {
            id: "chatcmpl-7".into(),
            object: "chat.completion.chunk",
            created: 123,
            model: "test-model".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    reasoning_content: None,
                    content: "hello".into(),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };
        let json = serde_json::to_value(&c).expect("chunk serializes");
        assert_eq!(json["id"], "chatcmpl-7");
        assert_eq!(json["object"], "chat.completion.chunk");
        assert_eq!(json["choices"][0]["delta"]["content"], "hello");
        // OpenAI's token-delta chunks carry `finish_reason: null` (the key
        // is present, the value null) — match that shape.
        assert!(json["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn the_final_chunk_has_a_finish_reason_and_an_empty_delta() {
        // The terminal chunk: finish_reason set, the delta empty (an empty
        // `content` is omitted — OpenAI's final-chunk shape).
        let c = Chunk {
            id: "chatcmpl-7".into(),
            object: "chat.completion.chunk",
            created: 123,
            model: "test-model".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta::default(),
                finish_reason: Some("stop"),
            }],
            usage: None,
        };
        let json = serde_json::to_value(&c).expect("chunk serializes");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        // An empty delta serializes to `{}` (the content key is omitted).
        assert_eq!(json["choices"][0]["delta"], serde_json::json!({}));
    }

    #[test]
    fn the_usage_chunk_has_empty_choices_and_populated_usage() {
        // The trailing summary chunk (sent right before `[DONE]`): empty
        // `choices`, `usage` populated with the token counts.
        let c = Chunk {
            id: "chatcmpl-7".into(),
            object: "chat.completion.chunk",
            created: 123,
            model: "test-model".into(),
            choices: vec![],
            usage: Some(Usage {
                prompt_tokens: 58,
                completion_tokens: 1500,
                total_tokens: 1558,
            }),
        };
        let json = serde_json::to_value(&c).expect("chunk serializes");
        assert_eq!(json["choices"], serde_json::json!([]));
        assert_eq!(json["usage"]["prompt_tokens"], 58);
        assert_eq!(json["usage"]["completion_tokens"], 1500);
        assert_eq!(json["usage"]["total_tokens"], 1558);
    }

    #[test]
    fn the_error_body_is_the_openai_shape() {
        let body = ApiError {
            error: ErrorBody {
                message: "nope".into(),
                r#type: "invalid_request_error".into(),
                code: "invalid_request_error".into(),
            },
        };
        let json = serde_json::to_value(&body).expect("error serializes");
        assert_eq!(json["error"]["message"], "nope");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["code"], "invalid_request_error");
    }

    #[test]
    fn a_responses_input_parses_as_a_string_or_a_message_list() {
        // `input: "hi"` → a single user turn.
        let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
            "input": "hi"
        }))
        .expect("string input parses");
        match req.input {
            ResponsesInput::Text(s) => assert_eq!(s, "hi"),
            _ => panic!("a string input must parse as Text"),
        }
        // `input: [...]` → a message list.
        let req: ResponsesRequest = serde_json::from_value(serde_json::json!({ "input": [
                { "role": "user", "content": "hi" }
            ] }))
        .expect("message input parses");
        match req.input {
            ResponsesInput::Messages(m) => {
                assert_eq!(m.len(), 1);
                assert_eq!(m[0].role, "user");
            }
            _ => panic!("a message-list input must parse as Messages"),
        }
    }

    // ── GitHub #121: tool-call and finish-reason hardening ──────────────

    #[test]
    fn a_clean_stop_with_a_complete_call_reports_tool_calls() {
        assert_eq!(
            resolve_finish_reason(FinishReason::Stop, true, false),
            "tool_calls"
        );
    }

    #[test]
    fn a_clean_stop_with_no_call_reports_stop() {
        assert_eq!(resolve_finish_reason(FinishReason::Stop, false, false), "stop");
    }

    #[test]
    fn a_mid_call_stop_never_reports_tool_calls_even_though_one_was_seen() {
        // Acceptance criterion 3: a call left open when generation stopped
        // was dropped whole by `ToolCallScanner` — reporting `tool_calls`
        // here would tell the client to run a call it never received.
        assert_eq!(resolve_finish_reason(FinishReason::Stop, true, true), "stop");
    }

    #[test]
    fn length_is_reported_regardless_of_calls_seen() {
        assert_eq!(resolve_finish_reason(FinishReason::Length, true, false), "length");
        assert_eq!(resolve_finish_reason(FinishReason::Length, false, false), "length");
    }

    #[test]
    fn a_streamed_tool_call_delta_serializes_in_the_openai_shape() {
        let delta = Delta {
            reasoning_content: None,
            content: String::new(),
            tool_calls: Some(vec![ToolCallDelta {
                index: 0,
                id: "call_0".into(),
                r#type: "function",
                function: FunctionOut {
                    name: "read_file".into(),
                    arguments: r#"{"path":"a.txt"}"#.into(),
                },
            }]),
        };
        let json = serde_json::to_value(&delta).expect("delta serializes");
        assert_eq!(json["tool_calls"][0]["index"], 0);
        assert_eq!(json["tool_calls"][0]["id"], "call_0");
        assert_eq!(json["tool_calls"][0]["type"], "function");
        assert_eq!(json["tool_calls"][0]["function"]["name"], "read_file");
        // A delta with tool_calls but no text carries no `content` key
        // (same "empty things are omitted" rule as `reasoning_content`).
        assert!(json.get("content").is_none());
    }

    #[test]
    fn an_assistant_message_with_no_tool_calls_omits_the_field_entirely() {
        let message = AssistantMessage {
            role: "assistant",
            reasoning_content: None,
            content: "hi".into(),
            tool_calls: None,
        };
        let json = serde_json::to_value(&message).expect("message serializes");
        assert!(json.as_object().unwrap().get("tool_calls").is_none());
    }
}
