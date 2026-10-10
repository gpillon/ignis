//! The Responses API's WebSocket mode (GitHub #282, spec responses-api/01):
//! `GET /v1/responses` upgraded to a socket that carries `response.create`
//! events whose body is the HTTP body, answered with the same events, byte
//! for byte in payload, as the HTTP stream — plus `stream_id` on the events
//! of a named stream.
//!
//! One connection task owns everything the connection has: its streams, the
//! requests waiting behind them, the connection-local cache of each stream's
//! latest finished response, and the responses running. Each running
//! response is its own task, which sends its events back through the
//! connection task, so one socket writer puts every frame on the wire.
//!
//! - **Streams.** A `response.create` without `stream_id` runs on the
//!   default stream, one with it on that named stream (32 distinct names per
//!   connection). Requests on one stream run in order and never overlap;
//!   requests on different streams run concurrently, at most 16 at a time
//!   per connection. A request that waits behind its stream or that limit
//!   gets no event until it starts.
//! - **Continuation.** `previous_response_id` names a stream's latest
//!   finished response on this connection and is resolved when the event is
//!   received, so a request that then waits cannot lose its parent to the
//!   source stream advancing. The request renders that response's whole
//!   conversation plus its own items: the same prompt a client re-sending
//!   everything produces, so reuse stays ADR 0029's content match.
//! - **Admission.** A request the engine finds full waits in the server-wide
//!   queue ([`super::queue`]) with `response.queued`, instead of a 503.
//! - **Close.** Closing the socket, by either side, cancels every response it
//!   has running and drops the ones it has queued.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::ws::rejection::WebSocketUpgradeRejection;
use axum::extract::State;
use axum::http::header::{SEC_WEBSOCKET_PROTOCOL, UPGRADE};
use axum::http::HeaderMap;
use axum::response::Response;
use serde_json::{json, Map, Value as JsonValue};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;
use tokio::task::AbortHandle;
use tracing::Instrument;

use crate::api::{self, ApiError, CancelOnDrop};
use crate::responses::events::{Ending, ResponseEvents};
use crate::responses::input::{self, CreateResponse, Prepared};
use crate::responses::queue::{Admission, Hub};
use crate::responses::{created_at, drive, Driven};
use crate::Server;

/// The subprotocol prefix a browser authenticates with (OpenAI's Realtime
/// convention): `openai-insecure-api-key.<key>`.
pub(crate) const CREDENTIAL_PREFIX: &str = "openai-insecure-api-key.";

/// Distinct named streams one connection may use; the default stream does
/// not count.
const NAMED_STREAMS: usize = 32;

/// Responses one connection may have handed to the engine at once — running,
/// or waiting in the admission queue.
const ACTIVE_RESPONSES: usize = 16;

/// The key a WebSocket upgrade presents as a subprotocol entry, for the key
/// check (`api::require_api_key`). Read only on an upgrade request.
pub(crate) fn subprotocol_key(headers: &HeaderMap) -> Option<&str> {
    let upgrade = headers.get(UPGRADE).and_then(|v| v.to_str().ok());
    if !upgrade.is_some_and(|u| u.eq_ignore_ascii_case("websocket")) {
        return None;
    }
    offered(headers).find_map(|protocol| protocol.strip_prefix(CREDENTIAL_PREFIX))
}

/// Every subprotocol the client offered, in order, across however many
/// `Sec-WebSocket-Protocol` headers it split them over.
fn offered(headers: &HeaderMap) -> impl Iterator<Item = &str> {
    headers
        .get_all(SEC_WEBSOCKET_PROTOCOL)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .filter(|protocol| !protocol.is_empty())
}

/// `GET /v1/responses` — the WebSocket mode.
#[utoipa::path(
    get,
    path = "/v1/responses",
    tag = "responses",
    operation_id = "responses_websocket",
    summary = "The Responses API's WebSocket mode",
    description = "Upgrades to a WebSocket (OpenAI's Responses WebSocket mode; the events and their sources are in `docs/findings/2026-09-29-openai-websocket-inference-interfaces.md`). Authenticate with `Authorization: Bearer <key>` on the upgrade, or, from a browser, with a `Sec-WebSocket-Protocol` entry `openai-insecure-api-key.<key>`; the server selects the first offered entry that is not a credential.

The client sends text frames carrying one JSON event each. `response.create` takes the `POST /v1/responses` body's fields beside its `type`, plus `stream_id` (a named stream, 1-256 characters of `[A-Za-z0-9_.-]`; 32 per connection) and `generate: false` (a warm-up: the prompt is prefilled and its state kept, nothing is generated). The server answers with the same events as the HTTP stream, with `stream_id` on the events of a named stream. Requests on one stream run in order; different streams run concurrently, up to 16 responses per connection. `previous_response_id` continues a stream's latest finished response on this connection with only the new items. A request the engine cannot admit waits in a server-wide queue (`response.created` with `status: \"queued\"`, then `response.queued`, then `response.in_progress`).

`response.cancel` (an ignis extension borrowed from the Realtime API, `{type, response_id}`) ends one response with `response.incomplete`, `status: \"cancelled\"`. `response.steer` is refused (`steering_not_supported`). A failure is an `error` event `{type: \"error\", status, error: {type, code, message, param}, stream_id?}`, and never closes the socket. Closing it cancels everything the connection had running or queued.",
    responses(
        (status = 101, description = "Switching protocols: the socket is open."),
        (status = 400, description = "Not a WebSocket upgrade.", body = ApiError),
        (status = 401, description = "The server was started with `--server-api-key` and the upgrade carried no matching bearer token or `openai-insecure-api-key.<key>` subprotocol.", body = ApiError),
    ),
)]
pub(crate) async fn connect(
    State(server): State<Arc<Server>>,
    headers: HeaderMap,
    upgrade: Result<WebSocketUpgrade, WebSocketUpgradeRejection>,
) -> Response {
    let Ok(upgrade) = upgrade else {
        return api::bad_request(
            "GET /v1/responses is the Responses API's WebSocket mode and takes a WebSocket upgrade; POST /v1/responses answers over HTTP",
        );
    };
    // The client's own names, never the credential: selecting from this list
    // picks the first offered entry that is not one.
    let protocols: Vec<String> = offered(&headers)
        .filter(|protocol| !protocol.starts_with(CREDENTIAL_PREFIX))
        .map(str::to_owned)
        .collect();
    // A frame may carry what a request body may.
    let limit = if server.active().media.is_some() { api::MEDIA_REQUEST_BODY_LIMIT } else { api::TEXT_REQUEST_BODY_LIMIT };
    upgrade
        .protocols(protocols)
        .max_message_size(limit)
        .max_frame_size(limit)
        .on_upgrade(move |socket| Connection::new(server, socket).run())
}

/// A stream's name: `None` for the default stream.
type StreamKey = Option<String>;

/// One stream of a connection.
#[derive(Default)]
struct StreamState {
    /// A response of this stream is running.
    busy: bool,
    /// Its latest finished response, what `previous_response_id` may name.
    latest: Option<Cached>,
}

/// A finished response as a continuation starts from it: the whole
/// conversation it saw, then its output items.
#[derive(Clone)]
struct Cached {
    id: String,
    items: Vec<JsonValue>,
}

/// A `response.create` waiting for its stream and a free place.
struct Job {
    stream: StreamKey,
    body: CreateResponse,
    history: Vec<JsonValue>,
    parent: Option<Parent>,
    warm_up: bool,
}

/// The response a request continues, and the stream it was the latest of.
#[derive(Clone)]
struct Parent {
    id: String,
    stream: StreamKey,
}

/// A response handed to its own task.
struct Active {
    stream: StreamKey,
    parent: Option<Parent>,
    /// Known once the response is created.
    response_id: Option<String>,
    cancel: Option<oneshot::Sender<()>>,
    task: AbortHandle,
}

/// What a response task tells the connection.
enum Outgoing {
    /// A frame to send.
    Event(JsonValue),
    /// The response now has its id (sent before its `response.created`).
    Named { key: u64, id: String },
    /// The response is over.
    Finished { key: u64, outcome: Outcome },
}

/// How a response ended, for the connection's cache.
enum Outcome {
    /// Completed or incomplete (cancelled included): it is its stream's
    /// latest, with the conversation a continuation starts from.
    Ended(Cached),
    /// It failed, or never became a response: not cached, and a same-stream
    /// continuation evicts the parent it named.
    Failed,
}

struct Connection {
    server: Arc<Server>,
    socket: WebSocket,
    outbox: UnboundedSender<Outgoing>,
    inbox: UnboundedReceiver<Outgoing>,
    streams: HashMap<StreamKey, StreamState>,
    /// Requests waiting for their stream or a free place, in arrival order.
    waiting: VecDeque<Job>,
    active: HashMap<u64, Active>,
    next_key: u64,
}

impl Connection {
    fn new(server: Arc<Server>, socket: WebSocket) -> Self {
        let (outbox, inbox) = unbounded_channel();
        Self {
            server,
            socket,
            outbox,
            inbox,
            streams: HashMap::new(),
            waiting: VecDeque::new(),
            active: HashMap::new(),
            next_key: 0,
        }
    }

    async fn run(mut self) {
        // Counted until this future is dropped, however it ends: a closed
        // socket, or a task aborted or panicking.
        let _open = OpenSocket::new(Arc::clone(&self.server.responses));
        loop {
            tokio::select! {
                frame = self.socket.recv() => match frame {
                    Some(Ok(Message::Text(text))) => self.receive(text.as_str()),
                    Some(Ok(Message::Binary(_))) => self.send(error_event(
                        "invalid_request_error",
                        "a frame must be a text frame carrying one JSON event",
                        None,
                        &None,
                    )),
                    // Pings are answered by the socket itself.
                    Some(Ok(Message::Ping(_) | Message::Pong(_))) => {}
                    Some(Ok(Message::Close(_))) => {
                        // RFC 6455 §5.5.1: the socket has queued the echo of
                        // the client's Close; reading on flushes it and ends
                        // the connection cleanly.
                        while let Some(Ok(_)) = self.socket.recv().await {}
                        break;
                    }
                    Some(Err(_)) | None => break,
                },
                Some(outgoing) = self.inbox.recv() => match outgoing {
                    Outgoing::Event(event) => {
                        if self.socket.send(Message::Text(event.to_string().into())).await.is_err() {
                            break;
                        }
                    }
                    Outgoing::Named { key, id } => {
                        if let Some(active) = self.active.get_mut(&key) {
                            active.response_id = Some(id);
                        }
                    }
                    Outgoing::Finished { key, outcome } => self.finished(key, outcome),
                },
            }
        }
        // Each task's drop cancels its engine request or leaves the queue.
        for active in self.active.values() {
            active.task.abort();
        }
    }

    /// Queue a frame for the writer, behind the frames already queued.
    fn send(&self, event: JsonValue) {
        let _ = self.outbox.send(Outgoing::Event(event));
    }

    /// One client frame.
    fn receive(&mut self, text: &str) {
        let Ok(JsonValue::Object(event)) = serde_json::from_str::<JsonValue>(text) else {
            return self.send(error_event("invalid_request_error", "a frame must carry one JSON object event", None, &None));
        };
        match event.get("type").and_then(JsonValue::as_str) {
            Some("response.create") => self.create(event),
            Some("response.cancel") => self.cancel(&event),
            Some("response.steer") => self.send(coded_error(
                400,
                "steering_not_supported",
                "steering is not supported: this server serves no response.steer",
                None,
                &None,
            )),
            Some(other) => self.send(error_event(
                "invalid_request_error",
                &format!("unknown event type '{other}' (response.create and response.cancel are served)"),
                Some("type"),
                &None,
            )),
            None => self.send(error_event("invalid_request_error", "an event needs a string type", Some("type"), &None)),
        }
    }

    /// `response.create`: checked and resolved now, started when its stream
    /// and a place are free.
    fn create(&mut self, event: Map<String, JsonValue>) {
        let stream: StreamKey = match api::optional_str(event.get("stream_id"), "stream_id") {
            Ok(None) => None,
            Ok(Some(id)) if valid_stream_id(id) => Some(id.to_owned()),
            _ => {
                return self.send(coded_error(
                    400,
                    "invalid_stream_id",
                    "The 'stream_id' field must be a non-empty string with at most 256 characters and may only contain letters, numbers, underscores, hyphens, and periods.",
                    Some("stream_id"),
                    &None,
                ))
            }
        };
        let named = self.streams.keys().filter(|key| key.is_some()).count();
        if stream.is_some() && !self.streams.contains_key(&stream) && named >= NAMED_STREAMS {
            return self.send(coded_error(
                400,
                "websocket_stream_limit_reached",
                "This WebSocket connection has reached its maximum number of distinct stream IDs (32). Reuse an existing stream_id or open a new WebSocket connection.",
                Some("stream_id"),
                &stream,
            ));
        }
        // The parent this request names, read even when the rest of the body
        // is not: a refused same-stream continuation evicts it.
        let named_parent = event.get("previous_response_id").and_then(JsonValue::as_str).map(str::to_owned);
        match self.accept(event, &stream) {
            Ok(job) => {
                // Only an accepted request takes a name of the 32.
                self.streams.entry(stream).or_default();
                self.waiting.push_back(job);
                self.pump();
            }
            Err(refusal) => {
                if let Some(parent) = named_parent {
                    self.evict(&stream, &parent);
                }
                self.send(refusal);
            }
        }
    }

    /// A `response.create`'s body, checked, with the conversation it
    /// continues — or the `error` event refusing it.
    fn accept(&self, event: Map<String, JsonValue>, stream: &StreamKey) -> Result<Job, JsonValue> {
        let body: CreateResponse = serde_json::from_value(JsonValue::Object(event))
            .map_err(|e| error_event("invalid_request_error", &format!("response.create: {e}"), None, stream))?;
        let warm_up = api::optional_bool(body.generate.as_ref(), "generate")
            .map_err(|message| error_event("invalid_request_error", &message, Some("generate"), stream))?
            .is_some_and(|generate| !generate);
        let previous = api::optional_str(body.previous_response_id.as_ref(), "previous_response_id")
            .map_err(|message| error_event("invalid_request_error", &message, Some("previous_response_id"), stream))?;
        // Resolved at receipt, and the history copied: a parent superseded
        // while this request waits is still the one it named.
        let (history, parent) = match previous {
            None => (Vec::new(), None),
            Some(id) => {
                let found = self
                    .streams
                    .iter()
                    .find_map(|(key, state)| state.latest.as_ref().filter(|c| c.id == id).map(|c| (key, c)))
                    .ok_or_else(|| {
                        coded_error(
                            400,
                            "previous_response_not_found",
                            &format!("Previous response with id '{id}' not found."),
                            Some("previous_response_id"),
                            stream,
                        )
                    })?;
                (found.1.items.clone(), Some(Parent { id: id.to_owned(), stream: found.0.clone() }))
            }
        };
        Ok(Job { stream: stream.clone(), body, history, parent, warm_up })
    }

    /// A same-stream continuation of `parent` failed: `stream` forgets it.
    fn evict(&mut self, stream: &StreamKey, parent: &str) {
        if let Some(state) = self.streams.get_mut(stream) {
            if state.latest.as_ref().is_some_and(|c| c.id == parent) {
                state.latest = None;
            }
        }
    }

    /// Start every waiting request whose stream is free, in arrival order,
    /// while the connection has a place.
    fn pump(&mut self) {
        let mut at = 0;
        while at < self.waiting.len() && self.active.len() < ACTIVE_RESPONSES {
            if self.streams.get(&self.waiting[at].stream).is_some_and(|s| s.busy) {
                at += 1;
                continue;
            }
            let job = self.waiting.remove(at).expect("in range");
            self.start(job);
        }
    }

    fn start(&mut self, job: Job) {
        let key = self.next_key;
        self.next_key += 1;
        self.streams.entry(job.stream.clone()).or_default().busy = true;
        let (cancel, cancelled) = oneshot::channel();
        let (stream, parent) = (job.stream.clone(), job.parent.clone());
        // One request, one root span (ADR 0012): its request id is recorded
        // once the scheduler gives it one.
        let span = tracing::info_span!(
            "ignis.ws.request",
            path = "/v1/responses",
            request_id = tracing::field::Empty,
        );
        let task = tokio::spawn(
            respond(Arc::clone(&self.server), job, key, self.outbox.clone(), cancelled).instrument(span),
        );
        self.active.insert(key, Active { stream, parent, response_id: None, cancel: Some(cancel), task: task.abort_handle() });
    }

    fn finished(&mut self, key: u64, outcome: Outcome) {
        let Some(active) = self.active.remove(&key) else {
            return;
        };
        let state = self.streams.entry(active.stream.clone()).or_default();
        state.busy = false;
        match outcome {
            Outcome::Ended(cached) => state.latest = Some(cached),
            // A same-stream continuation that failed evicts the parent it
            // named; a fork that failed leaves its source alone.
            Outcome::Failed => {
                if let Some(parent) = active.parent.filter(|p| p.stream == active.stream) {
                    self.evict(&active.stream, &parent.id);
                }
            }
        }
        self.pump();
    }

    /// `response.cancel`: the named response, or without a name the default
    /// stream's, running or queued.
    fn cancel(&mut self, event: &Map<String, JsonValue>) {
        let target = match api::optional_str(event.get("response_id"), "response_id") {
            Ok(Some(id)) => self.active.values_mut().find(|a| a.response_id.as_deref() == Some(id)),
            Ok(None) => self.active.values_mut().find(|a| a.stream.is_none() && a.response_id.is_some()),
            Err(message) => return self.send(error_event("invalid_request_error", &message, Some("response_id"), &None)),
        };
        match target.and_then(|active| active.cancel.take()) {
            Some(cancel) => {
                let _ = cancel.send(());
            }
            None => {
                let named = event.get("response_id").and_then(JsonValue::as_str);
                let message = match named {
                    Some(id) => format!("Response with id '{id}' is not running or queued on this connection."),
                    None => "The default stream has no response running or queued.".to_owned(),
                };
                self.send(coded_error(400, "response_not_found", &message, Some("response_id"), &None))
            }
        }
    }
}

/// One open socket in the `ignis_responses_sockets` gauge, for as long as it
/// lives.
struct OpenSocket(Arc<Hub>);

impl OpenSocket {
    fn new(hub: Arc<Hub>) -> Self {
        hub.socket(true);
        Self(hub)
    }
}

impl Drop for OpenSocket {
    fn drop(&mut self) {
        self.0.socket(false);
    }
}

/// Whether `id` is a valid `stream_id`: 1-256 of `[A-Za-z0-9_.-]`.
fn valid_stream_id(id: &str) -> bool {
    (1..=256).contains(&id.len())
        && id.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'))
}

/// An `error` event with OpenAI's documented `code`.
fn coded_error(status: u16, code: &str, message: &str, param: Option<&str>, stream: &StreamKey) -> JsonValue {
    let mut event = json!({
        "type": "error",
        "status": status,
        "error": { "type": "invalid_request_error", "code": code, "message": message, "param": param },
    });
    if let Some(stream) = stream {
        event["stream_id"] = json!(stream);
    }
    event
}

/// A 400 `error` event of the generic `invalid_request_error` kind.
fn error_event(code: &str, message: &str, param: Option<&str>, stream: &StreamKey) -> JsonValue {
    coded_error(400, code, message, param, stream)
}

/// The `status` and `error` object of a rendered refusal — the response the
/// same failure is on HTTP.
async fn refusal_parts(response: Response) -> (u16, JsonValue) {
    let status = response.status().as_u16();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap_or_default();
    let mut error = serde_json::from_slice::<JsonValue>(&body)
        .ok()
        .and_then(|mut body| body.get_mut("error").map(JsonValue::take))
        .unwrap_or_else(|| json!({ "type": "server_error", "code": "server_error", "message": "unreadable refusal" }));
    if error.get("param").is_none() {
        error["param"] = JsonValue::Null;
    }
    (status, error)
}

/// A refusal as its `error` event.
async fn refusal_event(response: Response) -> JsonValue {
    let (status, error) = refusal_parts(response).await;
    json!({ "type": "error", "status": status, "error": error })
}

/// A response task: prepare, submit (or queue), stream, and report the
/// outcome to the connection.
async fn respond(
    server: Arc<Server>,
    job: Job,
    key: u64,
    outbox: UnboundedSender<Outgoing>,
    cancelled: oneshot::Receiver<()>,
) {
    let stream = job.stream.clone();
    let emit = |mut event: JsonValue| {
        if let Some(stream) = &stream {
            event["stream_id"] = json!(stream);
        }
        let _ = outbox.send(Outgoing::Event(event));
    };
    // A dropped sender (the connection is closing, and aborts this task) is
    // not a cancel.
    let cancel = async {
        if cancelled.await.is_err() {
            std::future::pending::<()>().await;
        }
    };
    let outcome = serve(&server, job, key, &outbox, &emit, cancel).await;
    let _ = outbox.send(Outgoing::Finished { key, outcome });
}

async fn serve(
    server: &Server,
    job: Job,
    key: u64,
    outbox: &UnboundedSender<Outgoing>,
    emit: &impl Fn(JsonValue),
    cancel: impl std::future::Future<Output = ()>,
) -> Outcome {
    // A `model` naming another known model switches to it first, as over
    // HTTP (spec model-switch/01 §Implicit switch).
    if let Err(refusal) = input::switch_to_named_model(server, &job.body).await {
        emit(refusal_event(refusal).await);
        return Outcome::Failed;
    }
    // One model per request, not per socket (spec model-switch/01): a socket
    // outlives a switch, each of its requests is served by the model loaded
    // when it began.
    let server = &server.pinned();
    let mut cancel = std::pin::pin!(cancel);
    let Prepared { input, class, notes, items, start } =
        match input::prepare(server, job.body, job.history, job.warm_up).await {
            Ok(prepared) => prepared,
            Err(refusal) => {
                emit(refusal_event(refusal).await);
                return Outcome::Failed;
            }
        };
    let queue = &server.responses.queue;
    // The engine the request is submitted to is the one its cancel guard
    // must reach, whatever a model switch does meanwhile.
    let engine = server.active().engine.clone();
    let (mut events, request, mut scheduled) = match queue.submit(&engine, &input, class, notes).await {
        Admission::Refused(refused) => {
            emit(refusal_event(api::submit_error(server, refused)).await);
            return Outcome::Failed;
        }
        Admission::Admitted(request, scheduled) => {
            tracing::Span::current().record("request_id", request);
            let mut events = start.events(request.to_string(), created_at(server));
            let _ = outbox.send(Outgoing::Named { key, id: events.id().to_owned() });
            events.created(false).into_iter().for_each(emit);
            emit(events.in_progress());
            (events, request, scheduled)
        }
        Admission::Queued(ticket) => {
            // No request id yet: the ticket names the response.
            let mut events = start.events(format!("q{}", ticket.number()), created_at(server));
            let _ = outbox.send(Outgoing::Named { key, id: events.id().to_owned() });
            events.created(true).into_iter().for_each(emit);
            let admitted = tokio::select! {
                admitted = queue.wait(&ticket, &engine, &input, class, notes) => admitted,
                () = &mut cancel => {
                    events.cancelled().into_iter().for_each(emit);
                    return ended(&events, items);
                }
            };
            match admitted {
                Ok((request, scheduled)) => {
                    tracing::Span::current().record("request_id", request);
                    tracing::info!(response_id = events.id(), request_id = request, "queued response admitted");
                    emit(events.in_progress());
                    (events, request, scheduled)
                }
                // It already exists, so it ends `failed` rather than as an
                // `error` event.
                Err(refused) => {
                    let (_, error) = refusal_parts(api::submit_error(server, refused)).await;
                    let code = error["code"].as_str().unwrap_or("server_error").to_owned();
                    let message = error["message"].as_str().unwrap_or_default().to_owned();
                    events.failed(&code, message).into_iter().for_each(emit);
                    return Outcome::Failed;
                }
            }
        }
    };
    let mut guard = CancelOnDrop::new(engine, request);
    // `--server-request-timeout` counts from admission, never from queueing.
    let deadline = tokio::time::Instant::now() + server.live().request_timeout;
    let driven = tokio::select! {
        driven = drive(&mut events, &mut scheduled, deadline, emit) => Some(driven),
        () = &mut cancel => None,
    };
    match driven {
        Some(Driven::Ended) => guard.completed(),
        Some(Driven::TimedOut) => {
            let message = api::request_timeout_message(server.live().request_timeout);
            events.failed("request_timeout", message).into_iter().for_each(emit);
        }
        None => {
            // Dropping the guard unfinished cancels the engine request.
            drop(guard);
            events.cancelled().into_iter().for_each(emit);
        }
    }
    ended(&events, items)
}

/// The outcome of a response that has its terminal event.
fn ended(events: &ResponseEvents, mut items: Vec<JsonValue>) -> Outcome {
    match events.ending() {
        Some(Ending::Completed | Ending::Incomplete | Ending::Cancelled) => {
            items.extend(events.response().output.iter().cloned());
            Outcome::Ended(Cached { id: events.id().to_owned(), items })
        }
        Some(Ending::Failed) | None => Outcome::Failed,
    }
}
