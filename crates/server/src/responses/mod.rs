//! The OpenAI Responses API (GitHub #282, spec responses-api/01), over both
//! of its transports: `POST /v1/responses`, answered as one JSON body or as
//! server-sent events, and `GET /v1/responses`, its WebSocket mode
//! ([`socket`]).
//!
//! Both run on the engine path chat completions uses: a Responses
//! conversation is mapped onto chat messages ([`input`]) and rendered by the
//! same template, and the response's events come from the one producer in
//! [`events`], so the transports cannot disagree on a payload.

pub(crate) mod events;
pub(crate) mod input;
pub(crate) mod queue;
pub(crate) mod socket;

pub use queue::Hub;

use std::collections::VecDeque;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_core::Stream;
use serde::Serialize;
use serde_json::Value as JsonValue;
use utoipa::ToSchema;

use crate::api::{self, ApiError, CancelOnDrop};
use crate::engine::{drain_tokens, EventStream};
use ignis_core::SchedEvent;
use crate::Server;
use events::{ResponseEvents, ResponseObject};
use input::CreateResponse;

/// How [`drive`] left a response.
pub(crate) enum Driven {
    /// It reached its terminal event.
    Ended,
    /// The request's deadline passed first; the response has not ended.
    TimedOut,
}

/// Drive `events` through the request's scheduler stream until the response
/// ends or `deadline` passes, handing each event to `emit` as it is made.
/// A stream that closes without a finish (the engine dropped the request)
/// ends the response `failed`.
///
/// The deadline is checked before every event, not only while the stream
/// is idle: a request that keeps generating always has a token ready, and
/// would otherwise never time out.
pub(crate) async fn drive(
    events: &mut ResponseEvents,
    stream: &mut EventStream,
    deadline: tokio::time::Instant,
    mut emit: impl FnMut(JsonValue),
) -> Driven {
    let expiry = tokio::time::sleep_until(deadline);
    tokio::pin!(expiry);
    let mut held = None;
    loop {
        let event = match held.take() {
            Some(event) => Some(event),
            None => tokio::select! {
                biased;
                () = &mut expiry => return Driven::TimedOut,
                event = stream.recv() => event,
            },
        };
        match event {
            // A token and every one already waiting behind it, as one run.
            Some(SchedEvent::Token { token, .. }) => {
                let mut run = vec![token];
                held = drain_tokens(stream, &mut run);
                events.on_tokens(&run).into_iter().for_each(&mut emit);
            }
            Some(event) => {
                events.on_event(event).into_iter().for_each(&mut emit);
                if events.ending().is_some() {
                    return Driven::Ended;
                }
            }
            None => {
                events.failed("server_error", dropped_message()).into_iter().for_each(&mut emit);
                return Driven::Ended;
            }
        }
    }
}

/// The `failed` message of a request whose stream closed without a finish.
fn dropped_message() -> String {
    "the engine dropped the request before it finished; see the server log".to_owned()
}

/// The unix seconds a response is created at, off the server's wall clock.
pub(crate) fn created_at(server: &Server) -> u64 {
    server.wall_clock.now_ms() / 1000
}

/// One event of a streamed response (`text/event-stream`: an `event:` line
/// naming `type`, then a `data:` line carrying the whole event). Documents
/// the events rather than building them; a test holds it to the golden
/// transcripts (ADR 0036).
#[allow(dead_code)]
#[derive(Serialize, ToSchema)]
struct ResponseStreamEvent {
    /// `response.created`, `response.in_progress`,
    /// `response.output_item.added`, `response.output_text.delta`, ...,
    /// `response.completed` / `response.incomplete` / `response.failed`.
    r#type: String,
    /// Counts from 0 within the response.
    sequence_number: u64,
    /// On the lifecycle events: the response as it stands.
    response: Option<ResponseObject>,
}

/// `POST /v1/responses` — a response, streamed as server-sent events or not.
#[utoipa::path(
    post,
    path = "/v1/responses",
    tag = "responses",
    operation_id = "responses",
    summary = "A response, the Responses API",
    description = "The OpenAI Responses API on the engine path chat completions uses: `instructions` as the system prompt, `input` as a string or a list of items (`message`, `function_call`, `function_call_output`, `reasoning`), function `tools` in the Responses shape, and the answer as `output` items: a `reasoning` item for the thinking channel, a `message` item for the text, one `function_call` item per tool call. `tool_choice` is served as on chat completions: `\"required\"` and `{type: \"function\", name}` force the call's opening.

`stream: true` answers `text/event-stream`, one event per `event:`/`data:` pair: `response.created`, `response.in_progress`, then per output item `response.output_item.added`, its content events (`response.reasoning_text.delta`, `response.output_text.delta`, `response.function_call_arguments.delta`, ...), `response.output_item.done`, and one terminal event (`response.completed`, `response.incomplete` on `max_output_tokens`, `response.failed`). Every event carries a `sequence_number` counting from 0. Without it the body is the terminal event's `response`.

Not served, and refused with a 400 rather than ignored: hosted tools (anything but `type: \"function\"`), `text.format` other than `text`, `background: true`, and `previous_response_id`, since this server stores no responses over HTTP (the WebSocket mode, `GET /v1/responses`, continues from its connection). `usage.input_tokens_details.cached_tokens` is the prompt this request resumed from retained state instead of prefilling. The ignis extensions of chat completions are accepted with the same names and meaning, and `model` is read as there: another model the server lists switches it to that model before this request is served on it.",
    request_body = CreateResponse,
    responses(
        (status = 200, description = "The response. `application/json` when `stream` is false or absent; `text/event-stream` when it is true.", content(
            (ResponseObject = "application/json"),
            (ResponseStreamEvent = "text/event-stream"),
        )),
        (status = 400, description = "The request is malformed, or asks for what this server does not serve: a hosted tool, structured output, a background response, a `previous_response_id`.", body = ApiError),
        (status = 401, description = "The server was started with `--api-key` and the request carried no matching bearer token.", body = ApiError),
        (status = 404, description = "The request named a model this server neither loads nor may switch to.", body = ApiError),
        (status = 413, description = "The prompt is longer than this server's `--max-context`.", body = ApiError),
        (status = 503, description = "The engine is at capacity and the request was not admitted (`engine_full`); or a model switch is under way (`model_switching`, with `Retry-After`); or the switch this request's `model` began did not land (`model_switch_failed`).", body = ApiError),
        (status = 504, description = "A non-streaming request the engine did not finish within `--request-timeout`.", body = ApiError),
    ),
)]
pub(crate) async fn create_response(
    State(server): State<Arc<Server>>,
    Json(req): Json<CreateResponse>,
) -> Response {
    // A `model` naming another known model switches to it first (spec
    // model-switch/01 §Implicit switch), so the pin below takes that model.
    if let Err(refused) = input::switch_to_named_model(&server, &req).await {
        return refused;
    }
    // One model for the whole request (spec model-switch/01).
    let server = Arc::new(server.pinned());
    if let Some(refusal) = input::http_refusal(&req) {
        return refusal;
    }
    let stream = match api::optional_bool(req.stream.as_ref(), "stream") {
        Ok(stream) => stream.unwrap_or(false),
        Err(message) => return api::bad_request_param(&message, "stream"),
    };
    let prepared = match input::prepare(&server, req, Vec::new(), false).await {
        Ok(prepared) => prepared,
        Err(response) => return response,
    };
    // The engine the request is submitted to is the one its cancel guard
    // must reach, whatever a model switch does meanwhile.
    let engine = server.active().engine.clone();
    let (request_id, mut scheduled) =
        match engine.submit_with_notes(prepared.input, prepared.class, prepared.notes).await {
            Ok(submitted) => submitted,
            Err(err) => return api::submit_error(&server, err),
        };
    // GitHub #81 / ADR 0012: see the matching comment in `chat_completions`.
    tracing::Span::current().record("request_id", request_id);
    let mut events = prepared.start.events(request_id.to_string(), created_at(&server));
    let mut cancel = CancelOnDrop::new(engine, request_id);
    let deadline = tokio::time::Instant::now() + server.live().request_timeout;

    if stream {
        let mut pending: VecDeque<JsonValue> = events.created(false).into();
        pending.push_back(events.in_progress());
        return Sse::new(EventSse {
            stream: scheduled,
            cancel,
            events,
            pending,
            deadline: Box::pin(tokio::time::sleep_until(deadline)),
            timeout: server.live().request_timeout,
            ended: false,
            held: None,
        })
        .into_response();
    }
    match drive(&mut events, &mut scheduled, deadline, drop).await {
        Driven::Ended => {
            cancel.completed();
            Json(events.response().clone()).into_response()
        }
        // `cancel` drops unfinished here: the engine stops the request.
        Driven::TimedOut => api::error_response(
            StatusCode::GATEWAY_TIMEOUT,
            "request_timeout",
            "request_timeout",
            api::request_timeout_message(server.live().request_timeout),
        ),
    }
}

/// A streamed response: the request's scheduler stream through its
/// [`ResponseEvents`], one server-sent event per Responses event, ended by
/// the terminal one. The request's deadline ends it `failed` with
/// `request_timeout`; a client that hangs up cancels the request.
struct EventSse {
    stream: EventStream,
    cancel: CancelOnDrop,
    events: ResponseEvents,
    pending: VecDeque<JsonValue>,
    deadline: Pin<Box<tokio::time::Sleep>>,
    timeout: std::time::Duration,
    ended: bool,
    /// The event that ended the last run of tokens, handled next.
    held: Option<SchedEvent>,
}

impl Stream for EventSse {
    type Item = Result<Event, Infallible>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Every field is `Unpin` (the sleep is boxed).
        let this = self.get_mut();
        loop {
            if let Some(event) = this.pending.pop_front() {
                let kind = event["type"].as_str().unwrap_or_default().to_owned();
                return Poll::Ready(Some(Ok(Event::default().event(kind).data(event.to_string()))));
            }
            if this.ended {
                return Poll::Ready(None);
            }
            // First, for the reason `drive` checks it first.
            if this.deadline.as_mut().poll(cx).is_ready() {
                // Left unfinished: dropping the body cancels the request.
                let message = api::request_timeout_message(this.timeout);
                this.pending.extend(this.events.failed("request_timeout", message));
                this.ended = true;
                continue;
            }
            let polled = match this.held.take() {
                Some(event) => Poll::Ready(Some(event)),
                None => this.stream.poll_recv(cx),
            };
            match polled {
                // A token and every one already waiting behind it, as one run.
                Poll::Ready(Some(SchedEvent::Token { token, .. })) => {
                    let mut run = vec![token];
                    this.held = drain_tokens(&mut this.stream, &mut run);
                    this.pending.extend(this.events.on_tokens(&run));
                    continue;
                }
                Poll::Ready(Some(event)) => {
                    this.pending.extend(this.events.on_event(event));
                    if this.events.ending().is_some() {
                        this.cancel.completed();
                        this.ended = true;
                    }
                    continue;
                }
                Poll::Ready(None) => {
                    this.cancel.completed();
                    this.pending.extend(this.events.failed("server_error", dropped_message()));
                    this.ended = true;
                    continue;
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use utoipa::PartialSchema;

    /// A schema's property names and its required ones.
    fn fields<T: PartialSchema>() -> (Vec<String>, Vec<String>) {
        let schema = serde_json::to_value(T::schema()).expect("a schema serializes");
        let names = |key: &str| -> Vec<String> {
            match &schema[key] {
                JsonValue::Object(map) => map.keys().cloned().collect(),
                JsonValue::Array(list) => list.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect(),
                _ => Vec::new(),
            }
        };
        (names("properties"), names("required"))
    }

    /// ADR 0036: the documented event shape is the one the socket sends. Every
    /// event of the golden transcripts carries what [`ResponseStreamEvent`]
    /// requires, and every `response` in them is exactly a
    /// [`ResponseObject`] — each documented field, and no other. (Output
    /// items need no check: they are serialized from `OutputItem` itself.)
    #[test]
    fn the_golden_transcripts_match_the_documented_event_shapes() {
        let (_, required) = fields::<ResponseStreamEvent>();
        let (object_fields, object_required) = fields::<ResponseObject>();
        assert!(required.contains(&"sequence_number".to_owned()), "{required:?}");
        assert!(object_required.contains(&"output".to_owned()), "{object_required:?}");
        let transcripts = [
            include_str!("../../tests/fixtures/responses/named_stream_function_call.jsonl"),
            include_str!("../../tests/fixtures/responses/queued_then_cancelled.jsonl"),
        ];
        let mut responses = 0;
        for line in transcripts.iter().flat_map(|t| t.lines()) {
            let event: JsonValue = serde_json::from_str(line).expect("one JSON event per line");
            for field in &required {
                assert!(event.get(field).is_some(), "no `{field}` in {line}");
            }
            if let Some(response) = event.get("response") {
                responses += 1;
                let keys: Vec<&String> = response.as_object().expect("an object").keys().collect();
                for key in &keys {
                    assert!(object_fields.contains(key), "`{key}` is not a documented response field: {line}");
                }
                for field in &object_required {
                    assert!(response.get(field).is_some(), "no `{field}` in {line}");
                }
            }
        }
        assert!(responses >= 5, "the transcripts carry lifecycle events");
    }
}
