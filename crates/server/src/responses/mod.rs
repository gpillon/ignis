//! The OpenAI Responses API (GitHub #282, spec responses-api/01), over both
//! of its transports: `POST /v1/responses`, answered as one JSON body or as
//! server-sent events, and `GET /v1/responses`, its WebSocket mode
//! ([`socket`]).
//!
//! Both run on the engine path chat completions uses: a Responses
//! conversation is mapped onto chat messages ([`input`]) and rendered by the
//! same template, and the response's events come from the one producer in
//! [`events`], so the transports cannot disagree on a payload. The loop
//! guard (ADR 0052, [`crate::loop_guard`]) watches both as it watches chat
//! completions ([`crate::loop_guard::Guarding`]).

pub(crate) mod events;
pub(crate) mod ids;
pub(crate) mod input;
pub(crate) mod queue;
pub(crate) mod socket;

pub use ids::ResponseIds;
pub use queue::Hub;

use std::collections::VecDeque;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_core::Stream;
use serde::Serialize;
use serde_json::Value as JsonValue;
use utoipa::ToSchema;

use crate::api::{self, ApiError, CancelOnDrop};
use crate::deadline::{self, Next, StreamDeadline, Timeout, TimeoutReply};
use crate::engine::{drain_tokens, EventStream};
use crate::loop_guard::Guarding;
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

`stream: true` answers `text/event-stream`, one event per `event:`/`data:` pair: `response.created`, `response.in_progress`, then per output item `response.output_item.added`, its content events (`response.reasoning_text.delta`, `response.output_text.delta`, `response.function_call_arguments.delta`, ...), `response.output_item.done`, and one terminal event (`response.completed`, `response.incomplete` on `max_output_tokens` or the deadline, `response.failed`). Every event carries a `sequence_number` counting from 0. Without it the body is the terminal event's `response`.

Every request has a deadline from its admission (ADR 0058): `--server-request-timeout` answered whole, `--server-stream-timeout` streamed, `0` for none. Past it the request is cancelled and, under `--server-timeout-reply partial` (the default), ends as a length limit ends one, with what it generated and what the loop guard held back: `status: \"incomplete\"`, `incomplete_details.reason: \"max_output_tokens\"`, streamed as the `response.incomplete` event, answered whole as a `200` with the header `ignis-stopped-by: request_timeout`. Under `error` a stream ends `response.failed` with code `request_timeout`, and an answer awaited whole is a `504 request_timeout`.

`text.format` holds the answer as `response_format` does on chat completions: `{\"type\": \"json_object\"}` to a valid JSON object, `{\"type\": \"json_schema\", \"name\", \"schema\", \"strict\"}` to the schema, its members flat beside the type, with the same guarantees, refusals and limits, and the schema described to the model unless `ignis_describe_schema` is `false`. A cut by `max_output_tokens` is `incomplete`, the content as generated. Over the WebSocket mode too.

Not served, and refused with a 400 rather than ignored: hosted tools (anything but `type: \"function\"`), `background: true`, and `previous_response_id`, since this server stores no responses over HTTP (the WebSocket mode, `GET /v1/responses`, continues from its connection). `usage.input_tokens_details.cached_tokens` is the prompt this request resumed from retained state instead of prefilling. The ignis extensions of chat completions are accepted with the same names and meaning, and `model` is read as there: another model the server lists switches it to that model before this request is served on it.

The reasoning-loop guard (`--model-loop-guard`, ADR 0052) watches a request that reasons, as on chat completions: a loop is taken back, the model nudged and the generation continued, within this one response. The reasoning then carries a tag, `[LOOP DETECTED, REWINDING n TOKENS / m CHARS · h]`, after the `m` characters taken back, and `usage.output_tokens_details.rewound_tokens` (an ignis extension, absent when nothing was rewound) counts the tokens taken back, which `output_tokens` counts too. `include: [\"reasoning.encrypted_content\"]` gives every reasoning output item an `encrypted_content`: `ignis:v1:` and the base64 of the reasoning as the model saw it, without the tags and the text they took back. A `reasoning` input item whose `encrypted_content` is one of these is rendered from it in place of its text; any other `encrypted_content` is ignored, and the item's text is rendered with every verified rewind taken out. Any other `include` value is accepted and changes nothing.",
    request_body = CreateResponse,
    responses(
        (status = 200, description = "The response. `application/json` when `stream` is false or absent; `text/event-stream` when it is true.", content(
            (ResponseObject = "application/json"),
            (ResponseStreamEvent = "text/event-stream"),
        ), headers(("ignis-stopped-by" = String, description = "`request_timeout` on an answer its deadline cut, answered whole under `--server-timeout-reply partial` (ADR 0058); absent otherwise.")),),
        (status = 400, description = "The request is malformed, or asks for what this server does not serve: a hosted tool, a background response, a `previous_response_id`. Or `text.format` asks for what this model does not serve (`param: text.format`), as `response_format` on chat completions: a schema the grammar refuses, with the refusal's own code and a message naming the keyword or the limit and where; any type but `text`, `json_object` and `json_schema`; and on Qwen3.8-Flash-Next any structured output (`model_unsupported`, naming the model).", body = ApiError),
        (status = 401, description = "The server was started with `--server-api-key` and the request carried no matching bearer token.", body = ApiError),
        (status = 404, description = "The request named a model this server neither loads nor may switch to.", body = ApiError),
        (status = 409, description = "The model that would serve the request is the image model: the one loaded, with no `model` naming a text model to switch to, or the one `model` names (`model_kind_mismatch`, naming both kinds and the loaded id). Or `model` names a catalog entry this card cannot load, and no entry that answers to it can: by its class (`model_above_card_class`, naming the entry's class and the card's, ADR 0064) or by its GPU generation, alone or with its class (`model_above_gpu_generation`, naming the generation the entry needs and the card's name, compute capability and generation, ADR 0079): refused before anything loads or switches, the loaded model serving on.", body = ApiError),
        (status = 413, description = "The prompt is longer than this server's `--model-max-context`.", body = ApiError),
        (status = 503, description = "The engine is at capacity and the request was not admitted (`engine_full`, with `Retry-After`: when a lane is expected to free, 1 to 60 s); or a model switch is under way (`model_switching`, with `Retry-After`); or the switch this request's `model` began did not land (`model_switch_failed`); or the model was unloaded while idle and loading it again failed (`model_load_failed`); or the server began to stop while the request waited for a switch or a load (`server_stopping`, with `Retry-After`).", body = ApiError),
        (status = 504, description = "A non-streaming request the engine did not finish within `--server-request-timeout`, under `--server-timeout-reply error` (`request_timeout`). Under `partial`, the default, such a request is answered `200` with what it generated: `status: \"incomplete\"`, `incomplete_details.reason: \"max_output_tokens\"`, and the header `ignis-stopped-by: request_timeout`.", body = ApiError),
    ),
)]
pub(crate) async fn create_response(
    State(server): State<Arc<Server>>,
    Json(mut req): Json<CreateResponse>,
) -> Response {
    // A `model` naming another known model switches to it first (spec
    // model-switch/01 §Implicit switch), so the pin below takes that model.
    if let Err(refused) = input::switch_to_named_model(&server, &mut req).await {
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
    let engine = server.active().text_engine().clone();
    let guarding = Guarding::new(server.live().loop_guard.clone(), &prepared.input, prepared.starts_in_reasoning, server.metrics.clone());
    let rewound = guarding.rewound();
    let (request_id, scheduled) =
        match engine.submit_with_notes(prepared.input, prepared.class, prepared.notes).await {
            Ok(submitted) => submitted,
            Err(err) => return api::submit_error(&server, err),
        };
    // A continuation is submitted as this request was, straight to the
    // engine (its `Full` retried, as chat completions does).
    let ops = crate::loop_guard::EngineOps::new(engine.clone(), prepared.class, server.active().template.clone());
    let (mut scheduled, running) = guarding.guard(ops, request_id, scheduled);
    // GitHub #81 / ADR 0012: see the matching comment in `chat_completions`.
    tracing::Span::current().record("request_id", request_id);
    let mut events = prepared.start.events(server.response_ids.next_suffix(), created_at(&server)).with_rewound(rewound);
    ids::log_admitted(events.id(), request_id);
    let mut cancel = CancelOnDrop::following(engine, running.clone());
    // ADR 0058: a stream is held to `--server-stream-timeout`, a response
    // answered whole to `--server-request-timeout`.
    let live = server.live();
    let timeout = Timeout::of(&live, stream);

    if stream {
        let mut pending: VecDeque<JsonValue> = events.created(false).into();
        pending.push_back(events.in_progress());
        return Sse::new(EventSse {
            stream: scheduled,
            cancel,
            events,
            pending,
            deadline: StreamDeadline::after(timeout, running, live.timeout_reply),
            ended: false,
            held: None,
        })
        .into_response();
    }
    match drive(&mut events, &mut scheduled, timeout.from_now(), drop).await {
        Driven::Ended => {
            cancel.completed();
            Json(events.response().clone()).into_response()
        }
        Driven::TimedOut => {
            cancel.timed_out();
            if live.timeout_reply == TimeoutReply::Error {
                return timeout.response();
            }
            // What it generated, and what the loop guard held back of it.
            events.on_tokens(&api::owed_at_deadline(&running, &mut scheduled).await);
            events.cut();
            crate::deadline::stopped_by_timeout(Json(events.response().clone()).into_response())
        }
    }
}

/// A streamed response: the request's scheduler stream through its
/// [`ResponseEvents`], one server-sent event per Responses event, ended by
/// the terminal one. The request's deadline (`--server-stream-timeout`)
/// ends it as `reply` says, `incomplete` or `failed` with `request_timeout`
/// (ADR 0058), after what the loop guard held back (ADR 0052); a client that
/// hangs up cancels the request.
struct EventSse {
    stream: EventStream,
    cancel: CancelOnDrop,
    events: ResponseEvents,
    pending: VecDeque<JsonValue>,
    /// `server.stream_timeout` from admission; none at `0`.
    deadline: Option<StreamDeadline>,
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
            match deadline::poll_next_event(this.deadline.as_mut(), cx, &mut this.cancel, &mut this.stream, &mut this.held) {
                // A token and every one already waiting behind it, as one run.
                Poll::Ready(Next::Event(SchedEvent::Token { token, .. })) => {
                    let mut run = vec![token];
                    this.held = drain_tokens(&mut this.stream, &mut run);
                    this.pending.extend(this.events.on_tokens(&run));
                    continue;
                }
                Poll::Ready(Next::Event(event)) => {
                    this.pending.extend(this.events.on_event(event));
                    if this.events.ending().is_some() {
                        this.cancel.completed();
                        this.ended = true;
                    }
                    continue;
                }
                Poll::Ready(Next::Passed { owed, reply, timeout }) => {
                    this.pending.extend(this.events.on_tokens(&owed));
                    this.pending.extend(this.events.timed_out(reply, timeout));
                    this.ended = true;
                    continue;
                }
                Poll::Ready(Next::Closed) => {
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
