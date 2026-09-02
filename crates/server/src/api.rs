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

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_core::Stream;
use serde::{Deserialize, Serialize};

use ignis_core::{DecodeParams, RequestClass, RequestInput, SchedEvent, SubmitError};

use crate::engine::{collect_tokens, EventStream};
use crate::template::{ChatMessage, TemplateProvider};
use crate::Server;

/// Build the OpenAI router for `server` (the axum state it serves behind).
pub fn router(state: Arc<Server>) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(responses_api))
        .with_state(state)
}

/// The request's model, the templated prompt tokens, and the prompt-token
/// count (the usage figures) — one shared build path for both completion
/// endpoints.
fn build_request(
    server: &Server,
    model: Option<String>,
    messages: &[ChatMessage],
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    seed: Option<u64>,
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
    let tokens = server.template.apply_chat_template(messages);
    let prompt_tokens = tokens.len() as u32;
    let input = RequestInput {
        model: model.clone(),
        tokens,
        params: DecodeParams {
            max_tokens,
            temperature: temperature.unwrap_or(0.0), // 0.0 = greedy (the v1 gate)
            seed: seed.unwrap_or(0),
        },
    };
    (input, model, prompt_tokens)
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
    error_response(StatusCode::BAD_REQUEST, "invalid_request_error", "invalid_request_error", message)
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
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    seed: Option<u64>,
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
    let (input, model, prompt_tokens) = build_request(
        &server,
        req.model,
        &req.messages,
        req.temperature,
        req.max_tokens,
        req.seed,
    );
    let (id, mut stream) = match server.engine.submit(input, RequestClass::Interactive) {
        Ok(x) => x,
        Err(err) => return submit_error(&server, err),
    };
    let id = format!("chatcmpl-{id}");
    let created = now();

    if req.stream {
        // The SSE response: the request's event stream wrapped in the
        // `chat.completion.chunk` shape (a `[DONE]` marker terminates).
        return Sse::new(ChunkStream::new(
            stream,
            id,
            created,
            model,
            server.template.clone(),
        ))
        .into_response();
    }
    // Non-streaming: collect the request's tokens to its completion (a
    // timeout guards a wedged engine from hanging the client).
    match collect_tokens(&mut stream, server.request_timeout).await {
        Ok(tokens) => {
            let content = server.template.render_tokens(&tokens);
            let completion_tokens = tokens.len() as u32;
            Json(ChatCompletion {
                id,
                object: "chat.completion",
                created,
                model,
                choices: vec![CompletionChoice {
                    index: 0,
                    message: AssistantMessage {
                        role: "assistant",
                        content,
                    },
                    finish_reason: "stop",
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
            "the request did not complete within the server's timeout (the engine may be wedged)",
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
    content: String,
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
}

#[derive(Serialize)]
struct ChunkChoice {
    index: u8,
    delta: Delta,
    finish_reason: Option<&'static str>,
}

/// The token delta. An empty `content` serializes to `{}` (OpenAI's final
/// chunk shape).
#[derive(Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "String::is_empty")]
    content: String,
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
    id: String,
    created: u64,
    model: String,
    template: Arc<dyn TemplateProvider>,
    /// The `[DONE]` marker has been emitted (exactly once, at the end).
    done_sent: bool,
}

impl ChunkStream {
    fn new(
        stream: EventStream,
        id: String,
        created: u64,
        model: String,
        template: Arc<dyn TemplateProvider>,
    ) -> Self {
        Self {
            stream,
            id,
            created,
            model,
            template,
            done_sent: false,
        }
    }

    /// One chunk (a token delta or the final `finish_reason` chunk).
    fn chunk(&self, content: &str, finish_reason: Option<&'static str>) -> Event {
        let chunk = Chunk {
            id: self.id.clone(),
            object: "chat.completion.chunk",
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    content: content.to_string(),
                },
                finish_reason,
            }],
        };
        // `serde_json` cannot fail on this (all serializable fields).
        Event::default().data(serde_json::to_string(&chunk).expect("chunk serializes"))
    }
}

/// The stream adapter: the request's event stream into the SSE shape.
///
/// Pulls one event per poll (the driver routes events into the stream as
/// they are generated — tokens arrive in generation order); an `Evicted` /
/// `Restored` event for the request is skipped (it does not change the
/// generated-token sequence).
impl Stream for ChunkStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(self: Pin<&mut ChunkStream>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // `ChunkStream` is `Unpin` (all fields are `Unpin`), so we can get a
        // plain `&mut` back out of the pinned self.
        let this = self.get_mut();
        loop {
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
                        // The template seam: token id → response text
                        // (artifact-02's real tokenizer replaces this
                        // built-in rendering).
                        let content = this.template.render_tokens(std::slice::from_ref(&token));
                        return Poll::Ready(Some(Ok(this.chunk(&content, None))));
                    }
                    // The request completed: the final chunk (finish
                    // reason), then the `[DONE]` marker on the next poll
                    // (the stream closes).
                    SchedEvent::Done { .. } => {
                        return Poll::Ready(Some(Ok(this.chunk("", Some("stop")))));
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
        ResponsesInput::Text(text) => vec![ChatMessage {
            role: "user".into(),
            content: text,
        }],
        ResponsesInput::Messages(m) => m,
    };
    if messages.is_empty() {
        return bad_request("input must not be empty");
    }
    let (input, model, prompt_tokens) = build_request(
        &server,
        req.model,
        &messages,
        req.temperature,
        req.max_output_tokens,
        req.seed,
    );
    let (id, mut stream) = match server.engine.submit(input, RequestClass::Interactive) {
        Ok(x) => x,
        Err(err) => return submit_error(&server, err),
    };
    match collect_tokens(&mut stream, server.request_timeout).await {
        Ok(tokens) => {
            // The generated text (the template seam: artifact-02's
            // tokenizer renders real text here).
            let text = server.template.render_tokens(&tokens);
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
            "the request did not complete within the server's timeout (the engine may be wedged)",
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
                    content: "hello".into(),
                },
                finish_reason: None,
            }],
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
                delta: Delta {
                    content: String::new(),
                },
                finish_reason: Some("stop"),
            }],
        };
        let json = serde_json::to_value(&c).expect("chunk serializes");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        // An empty delta serializes to `{}` (the content key is omitted).
        assert_eq!(json["choices"][0]["delta"], serde_json::json!({}));
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
        let req: ResponsesRequest =
            serde_json::from_value(serde_json::json!({ "input": [
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
}