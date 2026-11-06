//! Request deadlines on every route (ADR 0058): `server.request_timeout`
//! for a request answered whole, `server.stream_timeout` for one that
//! streams, `0` for none, and `server.timeout_reply` for what a passed
//! deadline answers — `partial`, the answer so far ended as a length limit
//! ends one, or `error`.
//!
//! Every test runs on a one-lane engine whose request generates the tokens
//! the test gave it as soon as it is admitted, none of its steps waiting on
//! the test, and then waits at a gate until the test lets it go: the
//! deadline is what ends the request, never its own end, and the second it
//! is given is all margin, as the tokens are out microseconds after the
//! admission it counts from. The lane coming back — the next request
//! admitted on the only one — proves the request its deadline cut was
//! cancelled in the engine.

#[path = "support/lanes.rs"]
mod lanes;
#[path = "support/responses.rs"]
mod responses;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use futures_util::FutureExt;
use ignis_core::mock::{GateController, GatedCompute};
use ignis_core::{
    Compute, ComputeError, DecodeJob, DecodeOutcome, MockCompute, PrefillJob, PrefillOutcome, RequestId, SchedulerConfig,
};
use ignis_server::deadline::TimeoutReply;
use ignis_server::loop_guard::{LoopGuardMode, LoopGuardParams};
use ignis_server::Server;
use serde_json::{json, Value};
use tower::ServiceExt;

use lanes::{the_only_lane_admits_a_new_request, ReleaseObservedCompute};
use responses::{server, Script, MODEL};

/// The gate's compute, its decode steps counted: the first `free` run as
/// the engine schedules them, the one after them waits at the gate.
struct HoldsAfter {
    gated: Arc<GatedCompute>,
    free: usize,
    steps: AtomicUsize,
}

impl Compute for HoldsAfter {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
        self.gated.prefill_step(jobs)
    }

    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        if self.steps.fetch_add(1, Ordering::SeqCst) == self.free {
            self.gated.arm();
        }
        self.gated.decode_step(jobs)
    }

    fn release(&self, request: RequestId) {
        self.gated.release(request);
    }
}

/// A one-lane engine whose first request waits at the gate once it has
/// generated what the rig lets it.
struct Rig {
    server: Server,
    app: axum::Router,
    gated: Arc<GatedCompute>,
    gate: GateController,
    released: mpsc::Receiver<RequestId>,
}

/// A [`Rig`] whose template reads request 0's tokens as `texts`, in order,
/// with its server's knobs as `knobs` leaves them: the request generates
/// `free` tokens at once, then its next decode step waits at the gate.
fn rig(texts: &[&'static str], free: usize, knobs: impl FnOnce(Server) -> Server) -> Rig {
    let mock = MockCompute::new();
    let decode: HashMap<_, _> = texts.iter().enumerate().map(|(i, text)| (mock.token_for(0, i as u32), *text)).collect();
    let (released_tx, released) = mpsc::sync_channel(4);
    let observed = Arc::new(ReleaseObservedCompute { inner: MockCompute::new(), released: released_tx });
    let (gated, gate) = GatedCompute::new(observed as Arc<dyn Compute>);
    let holds = Arc::new(HoldsAfter { gated: Arc::clone(&gated), free, steps: AtomicUsize::new(0) });
    let config = SchedulerConfig { max_in_flight: 1, ..SchedulerConfig::default() };
    let server = knobs(server(&Script::new(decode), config, holds as Arc<dyn Compute>));
    Rig { app: server.app(), server, gated, gate, released }
}

/// A [`rig`] whose request generates every one of `texts`, its deadlines a
/// second, answered as `reply` says.
fn one_second(texts: &[&'static str], reply: TimeoutReply) -> Rig {
    rig(texts, texts.len(), |server| {
        server
            .with_request_timeout(Duration::from_secs(1))
            .with_stream_timeout(Duration::from_secs(1))
            .with_timeout_reply(reply)
    })
}

impl Rig {
    /// The request generated what the rig lets it and its next step waits
    /// at the gate; the gate is armed again for the next request.
    fn holds(&self) {
        self.gate.wait_entered();
        self.gated.arm();
    }

    /// The held step goes on: the request it belonged to, cut by its
    /// deadline, leaves the engine, and the only lane admits a new request.
    async fn the_lane_comes_back(&self) {
        self.gate.release();
        self.released.recv_timeout(Duration::from_secs(5)).expect("the request its deadline cut leaves the engine");
        the_only_lane_admits_a_new_request(&self.app, MODEL, &self.gate).await;
    }
}

fn post(app: &axum::Router, path: &str, body: Value) -> impl std::future::Future<Output = axum::response::Response> + use<> {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    app.clone().oneshot(request).map(Result::unwrap)
}

/// A response's status, its `ignis-stopped-by` header, and its body.
async fn read(response: axum::response::Response) -> (u16, Option<String>, String) {
    let status = response.status().as_u16();
    let stopped_by = response.headers().get("ignis-stopped-by").map(|v| v.to_str().unwrap().to_owned());
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, stopped_by, String::from_utf8(body.to_vec()).unwrap())
}

/// A chat request with thinking off, capped far past what the test lets it
/// generate.
fn chat(stream: bool) -> Value {
    json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "go on" }],
        "max_tokens": 4096,
        "enable_thinking": false,
        "stream": stream,
    })
}

/// Each `data:` payload of a chat SSE body, `[DONE]` as the string.
fn chunks(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str(data).unwrap_or_else(|_| Value::String(data.to_owned())))
        .collect()
}

// ── chat completions, answered whole: `server.request_timeout` ──────────

/// Under `partial`, a completion its deadline cut is the answer so far, as a
/// completion a length limit cut: `finish_reason: "length"`, the usage
/// counting what was generated, and the header naming the deadline. With
/// and without a stop sequence, which collect the answer two ways.
async fn check_a_cut_completion_answers_what_it_generated(stop: Option<&str>) {
    let rig = one_second(&["Hello", " world"], TimeoutReply::Partial);
    let mut body = chat(false);
    if let Some(stop) = stop {
        body["stop"] = json!([stop]);
    }
    let reply = tokio::spawn(post(&rig.app, "/v1/chat/completions", body).then(read));
    rig.holds();
    let (status, stopped_by, body) = reply.await.unwrap();
    assert_eq!((status, stopped_by.as_deref()), (200, Some("request_timeout")), "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "Hello world", "{body}");
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{body}");
    assert_eq!(v["usage"]["completion_tokens"], 2, "{body}");
    rig.the_lane_comes_back().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_completion_its_deadline_cut_answers_what_it_generated() {
    check_a_cut_completion_answers_what_it_generated(None).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_completion_with_a_stop_sequence_its_deadline_cut_answers_what_it_generated() {
    check_a_cut_completion_answers_what_it_generated(Some("zzz")).await;
}

/// A browser on another origin reads a response header outside the CORS
/// safelist only when the server exposes it: the one naming the deadline
/// that cut an answer, `Retry-After` on a full engine or a load, and the
/// model's status on `/v1/config`.
#[tokio::test]
async fn a_browser_on_another_origin_can_read_why_an_answer_stopped() {
    let rig = rig(&[], 0, |server| server);
    let response = rig.app.clone().oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap()).await.unwrap();
    let exposed = response.headers()["access-control-expose-headers"].to_str().unwrap().to_owned();
    let exposed: Vec<String> = exposed.split(',').map(|header| header.trim().to_ascii_lowercase()).collect();
    for header in ["ignis-stopped-by", "retry-after", "ignis-model-status"] {
        assert!(exposed.iter().any(|exposed| exposed == header), "{header} in {exposed:?}");
    }
}

/// A tool call the deadline cut mid-way is dropped whole: the content
/// before it stands, and no half-written call reaches the client. Answered
/// whole, with and without a stop sequence, and streamed: three ways of
/// assembling the answer.
async fn check_a_cut_tool_call_is_dropped(stream: bool, stop: Option<&str>) {
    let rig = one_second(&["Reading it. ", "<tool_call>\n<function=read_file>\n<parameter=path>\na."], TimeoutReply::Partial);
    let mut body = chat(stream);
    body["tools"] = json!([{ "type": "function", "function": { "name": "read_file", "parameters": {
        "type": "object", "properties": { "path": { "type": "string" } }
    } } }]);
    if let Some(stop) = stop {
        body["stop"] = json!([stop]);
    }
    let reply = tokio::spawn(post(&rig.app, "/v1/chat/completions", body).then(read));
    rig.holds();
    let (status, _, body) = reply.await.unwrap();
    assert_eq!(status, 200, "{body}");
    if stream {
        let chunks = chunks(&body);
        assert_eq!(streamed_text(&chunks, "content"), "Reading it. ", "{body}");
        assert!(chunks.iter().all(|c| c["choices"][0]["delta"].get("tool_calls").is_none()), "{body}");
        assert_eq!(finish_reasons(&chunks), ["length"], "{body}");
    } else {
        let v: Value = serde_json::from_str(&body).unwrap();
        let message = &v["choices"][0]["message"];
        assert_eq!(message["content"], "Reading it. ", "{body}");
        assert!(message.get("tool_calls").is_none(), "{body}");
        assert_eq!(v["choices"][0]["finish_reason"], "length", "{body}");
    }
    rig.the_lane_comes_back().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_tool_call_the_deadline_cut_is_dropped() {
    check_a_cut_tool_call_is_dropped(false, None).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_tool_call_the_deadline_cut_is_dropped_with_a_stop_sequence() {
    check_a_cut_tool_call_is_dropped(false, Some("zzz")).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_streamed_tool_call_the_deadline_cut_is_dropped() {
    check_a_cut_tool_call_is_dropped(true, None).await;
}

/// Under `error`, a completion its deadline cut is today's `504
/// request_timeout`, and its lane comes back all the same.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_completion_its_deadline_cut_is_a_504_under_the_error_reply() {
    let rig = one_second(&["Hello", " world"], TimeoutReply::Error);
    let reply = tokio::spawn(post(&rig.app, "/v1/chat/completions", chat(false)).then(read));
    rig.holds();
    let (status, stopped_by, body) = reply.await.unwrap();
    assert_eq!((status, stopped_by), (504, None), "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["code"], "request_timeout", "{body}");
    rig.the_lane_comes_back().await;
}

/// The answer so far includes what the loop guard held back: a reasoning
/// the client had not been sent a word of is flushed into the partial
/// answer before it ends (ADR 0052).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cut_completion_carries_what_the_loop_guard_held_back() {
    let guard = LoopGuardParams { mode: LoopGuardMode::Rewind, holdback_chars: 100_000, ..LoopGuardParams::default() };
    let rig = rig(&["Let me", " think"], 2, |server| {
        server.with_request_timeout(Duration::from_secs(1)).with_loop_guard(guard)
    });
    let mut body = chat(false);
    body["enable_thinking"] = json!(true);
    let reply = tokio::spawn(post(&rig.app, "/v1/chat/completions", body).then(read));
    rig.holds();
    let (status, _, body) = reply.await.unwrap();
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["message"]["reasoning_content"], "Let me think", "{body}");
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{body}");
    rig.the_lane_comes_back().await;
}

/// A completion its engine let go of — its stream closed without a finish —
/// is an engine error, not a completion its deadline cut: `500
/// engine_error`, never a `504 request_timeout`. With and without a stop
/// sequence, which collect the answer two ways.
async fn check_a_dropped_completion_is_an_engine_error(stop: Option<&str>) {
    let rig = rig(&["Hello"], 1, |server| server.with_request_timeout(Duration::from_secs(600)));
    let mut body = chat(false);
    if let Some(stop) = stop {
        body["stop"] = json!([stop]);
    }
    let reply = tokio::spawn(post(&rig.app, "/v1/chat/completions", body).then(read));
    rig.holds();
    // The scheduler releases a cancelled request without a `Done`.
    rig.server.active().text_engine().cancel(0);
    rig.gate.release();
    let (status, stopped_by, body) = reply.await.unwrap();
    assert_eq!((status, stopped_by), (500, None), "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["code"], "engine_error", "{body}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_completion_its_engine_dropped_is_an_engine_error() {
    check_a_dropped_completion_is_an_engine_error(None).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_completion_with_a_stop_sequence_its_engine_dropped_is_an_engine_error() {
    check_a_dropped_completion_is_an_engine_error(Some("zzz")).await;
}

/// `0` is no deadline: a request held well past what a zero-length deadline
/// would allow still ends on its own, at its cap, with no header.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_zero_request_timeout_is_no_deadline() {
    let rig = rig(&["Hello", " world"], 0, |server| server.with_request_timeout(Duration::ZERO));
    let mut body = chat(false);
    // Two tokens: the first leaves with the prefill (#518), the gate holds
    // the round the second needs.
    body["max_tokens"] = json!(2);
    let reply = tokio::spawn(post(&rig.app, "/v1/chat/completions", body).then(read));
    rig.gate.wait_entered();
    // Time passing is the point here, not a synchronization.
    tokio::time::sleep(Duration::from_millis(100)).await;
    rig.gate.release();
    let (status, stopped_by, body) = reply.await.unwrap();
    assert_eq!((status, stopped_by), (200, None), "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "Hello world", "{body}");
    assert_eq!(v["usage"]["completion_tokens"], 2, "{body}");
}

// ── chat completions, streamed: `server.stream_timeout` ─────────────────

/// The text deltas of a chat stream's chunks, joined per channel.
fn streamed_text(chunks: &[Value], channel: &str) -> String {
    chunks.iter().filter_map(|chunk| chunk["choices"][0]["delta"][channel].as_str()).collect()
}

/// Under `partial`, a stream its deadline cut ends as a stream a length
/// limit cut: its deltas, the `length` finish chunk, the usage chunk when
/// asked for, and `[DONE]`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stream_its_deadline_cut_ends_with_a_length_finish() {
    let rig = one_second(&["Hello", " world"], TimeoutReply::Partial);
    let mut body = chat(true);
    body["stream_options"] = json!({ "include_usage": true });
    let response = post(&rig.app, "/v1/chat/completions", body).await;
    assert_eq!(response.status(), 200);
    let reply = tokio::spawn(read(response));
    rig.holds();
    let (_, _, body) = reply.await.unwrap();
    let chunks = chunks(&body);
    assert_eq!(streamed_text(&chunks, "content"), "Hello world", "{body}");
    let finishes: Vec<&Value> = chunks.iter().filter(|c| !c["choices"][0]["finish_reason"].is_null()).collect();
    assert_eq!(finishes.len(), 1, "{body}");
    assert_eq!(finishes[0]["choices"][0]["finish_reason"], "length", "{body}");
    let n = chunks.len();
    assert_eq!(chunks[n - 2]["usage"]["completion_tokens"], 2, "{body}");
    assert_eq!(chunks[n - 1], "[DONE]", "{body}");
    assert!(!body.contains("\"error\""), "{body}");
    rig.the_lane_comes_back().await;
}

/// Under `error`, a stream its deadline cut ends with the `504`'s body as
/// its last chunk, as an engine error ends one (GitHub #296): no finish
/// chunk, then `[DONE]`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stream_its_deadline_cut_ends_with_a_request_timeout_error_under_the_error_reply() {
    let rig = one_second(&["Hello", " world"], TimeoutReply::Error);
    let response = post(&rig.app, "/v1/chat/completions", chat(true)).await;
    assert_eq!(response.status(), 200);
    let reply = tokio::spawn(read(response));
    rig.holds();
    let (_, _, body) = reply.await.unwrap();
    let chunks = chunks(&body);
    assert_eq!(streamed_text(&chunks, "content"), "Hello world", "{body}");
    assert!(chunks.iter().all(|c| c["choices"][0]["finish_reason"].is_null()), "no finish chunk: {body}");
    let n = chunks.len();
    assert_eq!(chunks[n - 2]["error"]["code"], "request_timeout", "{body}");
    assert!(chunks[n - 2]["error"]["message"].as_str().unwrap().contains("--server-stream-timeout"), "{body}");
    assert_eq!(chunks[n - 1], "[DONE]", "{body}");
    rig.the_lane_comes_back().await;
}

/// The loop guard's held-back reasoning reaches a stream its deadline cut
/// before the stream ends.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cut_stream_carries_what_the_loop_guard_held_back() {
    let guard = LoopGuardParams { mode: LoopGuardMode::Rewind, holdback_chars: 100_000, ..LoopGuardParams::default() };
    let rig = rig(&["Let me", " think"], 2, |server| server.with_stream_timeout(Duration::from_secs(1)).with_loop_guard(guard));
    let mut body = chat(true);
    body["enable_thinking"] = json!(true);
    let response = post(&rig.app, "/v1/chat/completions", body).await;
    let reply = tokio::spawn(read(response));
    rig.holds();
    let (_, _, body) = reply.await.unwrap();
    let chunks = chunks(&body);
    assert_eq!(streamed_text(&chunks, "reasoning_content"), "Let me think", "{body}");
    assert_eq!(finish_reasons(&chunks), ["length"], "{body}");
    assert_eq!(chunks.last().unwrap(), "[DONE]", "{body}");
    rig.the_lane_comes_back().await;
}

/// The `finish_reason` of each chunk that carries one.
fn finish_reasons(chunks: &[Value]) -> Vec<&str> {
    chunks.iter().filter_map(|chunk| chunk["choices"][0]["finish_reason"].as_str()).collect()
}

/// Past the deadline only the tokens still count: a request that finished
/// on its own while its stream went unread, its `Done` already queued
/// behind what the loop guard held back, ends with one finish chunk and one
/// usage chunk, the deadline's, then `[DONE]` — never its own finish too.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stream_its_deadline_cut_with_its_own_end_queued_ends_once() {
    let guard = LoopGuardParams { mode: LoopGuardMode::Rewind, holdback_chars: 100_000, ..LoopGuardParams::default() };
    let timeout = Duration::from_millis(200);
    let rig = rig(&["Let me", " think"], 2, |server| server.with_stream_timeout(timeout).with_loop_guard(guard));
    let mut body = chat(true);
    body["enable_thinking"] = json!(true);
    body["max_tokens"] = json!(2);
    body["stream_options"] = json!({ "include_usage": true });
    let response = post(&rig.app, "/v1/chat/completions", body).await;
    // Both steps run, and the request ends at its cap, before a byte of the
    // stream is read.
    rig.released.recv_timeout(Duration::from_secs(5)).expect("the request ends on its own");
    // Time passing is the point here: the deadline passes before the first poll.
    tokio::time::sleep(timeout * 2).await;
    let (_, _, body) = read(response).await;
    let chunks = chunks(&body);
    assert_eq!(streamed_text(&chunks, "reasoning_content"), "Let me think", "{body}");
    assert_eq!(finish_reasons(&chunks), ["length"], "one finish chunk: {body}");
    assert_eq!(chunks.iter().filter(|c| c.get("usage").is_some_and(|u| !u.is_null())).count(), 1, "one usage chunk: {body}");
    assert_eq!(chunks.last().unwrap(), "[DONE]", "{body}");
}

/// A stream is held to `server.stream_timeout` alone, and `0` is none: one
/// held past the non-streaming deadline ends on its own.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_zero_stream_timeout_is_no_deadline() {
    let rig = rig(&["Hello", " world"], 0, |server| server.with_request_timeout(Duration::from_millis(20)).with_stream_timeout(Duration::ZERO));
    let mut body = chat(true);
    body["max_tokens"] = json!(2);
    let response = post(&rig.app, "/v1/chat/completions", body).await;
    let reply = tokio::spawn(read(response));
    rig.gate.wait_entered();
    // Time passing is the point here, not a synchronization.
    tokio::time::sleep(Duration::from_millis(100)).await;
    rig.gate.release();
    let (_, _, body) = reply.await.unwrap();
    let chunks = chunks(&body);
    assert_eq!(streamed_text(&chunks, "content"), "Hello world", "{body}");
    assert!(chunks.iter().any(|c| c["choices"][0]["finish_reason"] == "length"), "its cap: {body}");
    assert!(!body.contains("\"error\""), "{body}");
}

// ── the Responses API: HTTP answered whole, streamed, and its WebSocket ─

/// A response request with thinking off, capped far past what the test lets
/// it generate.
fn response_request(stream: bool) -> Value {
    json!({ "model": MODEL, "input": "go on", "max_output_tokens": 4096, "enable_thinking": false, "stream": stream })
}

/// The message text a response's output holds.
fn output_text(response: &Value) -> String {
    response["output"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "message")
        .flat_map(|item| item["content"].as_array().unwrap().iter().map(|part| part["text"].as_str().unwrap().to_owned()))
        .collect()
}

/// Under `partial`, a response its deadline cut is `incomplete` with
/// `max_output_tokens`, as a length limit leaves one, its output what was
/// generated; over HTTP answered whole, with the header naming the
/// deadline.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_response_its_deadline_cut_is_incomplete_with_what_it_generated() {
    let rig = one_second(&["Hello", " world"], TimeoutReply::Partial);
    let reply = tokio::spawn(post(&rig.app, "/v1/responses", response_request(false)).then(read));
    rig.holds();
    let (status, stopped_by, body) = reply.await.unwrap();
    assert_eq!((status, stopped_by.as_deref()), (200, Some("request_timeout")), "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["status"], "incomplete", "{body}");
    assert_eq!(v["incomplete_details"]["reason"], "max_output_tokens", "{body}");
    assert_eq!(output_text(&v), "Hello world", "{body}");
    assert_eq!(v["usage"]["output_tokens"], 2, "{body}");
    rig.the_lane_comes_back().await;
}

/// Under `error`, a response answered whole is today's `504
/// request_timeout`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_response_its_deadline_cut_is_a_504_under_the_error_reply() {
    let rig = one_second(&["Hello", " world"], TimeoutReply::Error);
    let reply = tokio::spawn(post(&rig.app, "/v1/responses", response_request(false)).then(read));
    rig.holds();
    let (status, stopped_by, body) = reply.await.unwrap();
    assert_eq!((status, stopped_by), (504, None), "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["code"], "request_timeout", "{body}");
    rig.the_lane_comes_back().await;
}

/// A streamed response its deadline cut ends with `response.incomplete`
/// (`partial`) or `response.failed` with `request_timeout` (`error`), its
/// text deltas sent first.
async fn check_a_streamed_response_its_deadline_cut(reply: TimeoutReply) {
    let rig = one_second(&["Hello", " world"], reply);
    let response = post(&rig.app, "/v1/responses", response_request(true)).await;
    assert_eq!(response.status(), 200);
    let body = tokio::spawn(read(response));
    rig.holds();
    let (_, _, body) = body.await.unwrap();
    let events = responses::sse_events(&body);
    let deltas: String = events
        .iter()
        .filter(|(name, _)| name == "response.output_text.delta")
        .map(|(_, event)| event["delta"].as_str().unwrap())
        .collect();
    assert_eq!(deltas, "Hello world", "{body}");
    let (last, terminal) = events.last().unwrap();
    match reply {
        TimeoutReply::Partial => {
            assert_eq!(last, "response.incomplete", "{body}");
            assert_eq!(terminal["response"]["incomplete_details"]["reason"], "max_output_tokens", "{body}");
            assert_eq!(output_text(&terminal["response"]), "Hello world", "{body}");
        }
        TimeoutReply::Error => {
            assert_eq!(last, "response.failed", "{body}");
            assert_eq!(terminal["response"]["error"]["code"], "request_timeout", "{body}");
            assert!(terminal["response"]["error"]["message"].as_str().unwrap().contains("--server-stream-timeout"), "{body}");
        }
    }
    rig.the_lane_comes_back().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_streamed_response_its_deadline_cut_ends_incomplete() {
    check_a_streamed_response_its_deadline_cut(TimeoutReply::Partial).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_streamed_response_its_deadline_cut_ends_failed_under_the_error_reply() {
    check_a_streamed_response_its_deadline_cut(TimeoutReply::Error).await;
}

/// A response is held to the deadline of how it is answered alone, and
/// `0` is none: one held past the other deadline ends on its own, at its
/// cap, with no header — answered whole under a zero `request_timeout`,
/// streamed under a zero `stream_timeout`.
async fn check_a_zero_deadline_leaves_a_response_to_its_end(stream: bool) {
    let (request, streamed) = match stream {
        true => (Duration::from_millis(20), Duration::ZERO),
        false => (Duration::ZERO, Duration::from_millis(20)),
    };
    let rig = rig(&["Hello", " world"], 0, |server| server.with_request_timeout(request).with_stream_timeout(streamed));
    let mut body = response_request(stream);
    // Two tokens: the first leaves with the prefill (#518), the gate holds
    // the round the second needs.
    body["max_output_tokens"] = json!(2);
    let reply = tokio::spawn(post(&rig.app, "/v1/responses", body).then(read));
    rig.gate.wait_entered();
    // Time passing is the point here, not a synchronization.
    tokio::time::sleep(Duration::from_millis(100)).await;
    rig.gate.release();
    let (status, stopped_by, body) = reply.await.unwrap();
    assert_eq!((status, stopped_by), (200, None), "{body}");
    let response = match stream {
        true => {
            let (last, terminal) = responses::sse_events(&body).pop().unwrap();
            assert_eq!(last, "response.incomplete", "{body}");
            terminal["response"].clone()
        }
        false => serde_json::from_str(&body).unwrap(),
    };
    assert_eq!(response["incomplete_details"]["reason"], "max_output_tokens", "its cap: {body}");
    assert_eq!(output_text(&response), "Hello world", "{body}");
    assert_eq!(response["usage"]["output_tokens"], 2, "{body}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_zero_request_timeout_leaves_a_response_to_its_end() {
    check_a_zero_deadline_leaves_a_response_to_its_end(false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_zero_stream_timeout_leaves_a_streamed_response_to_its_end() {
    check_a_zero_deadline_leaves_a_response_to_its_end(true).await;
}

/// A `response.create` of a response request.
fn create(max_output_tokens: u32) -> Value {
    let mut create = response_request(false);
    create["type"] = json!("response.create");
    create["max_output_tokens"] = json!(max_output_tokens);
    create.as_object_mut().unwrap().remove("stream");
    create
}

/// The same over the WebSocket, held to `server.stream_timeout` however
/// long `server.request_timeout` is.
async fn check_a_socket_response_its_deadline_cut(reply: TimeoutReply) {
    let rig = rig(&["Hello", " world"], 2, |server| {
        server
            .with_request_timeout(Duration::from_secs(600))
            .with_stream_timeout(Duration::from_secs(1))
            .with_timeout_reply(reply)
    });
    let live = responses::live(rig.server.clone()).await;
    let mut socket = responses::socket(&live).await;
    responses::send(&mut socket, create(4096)).await;
    rig.holds();
    let events = responses::events_until(&mut socket, responses::terminal).await;
    let terminal = events.last().unwrap();
    match reply {
        TimeoutReply::Partial => {
            assert_eq!(terminal["type"], "response.incomplete", "{terminal}");
            assert_eq!(terminal["response"]["incomplete_details"]["reason"], "max_output_tokens", "{terminal}");
            assert_eq!(output_text(&terminal["response"]), "Hello world", "{terminal}");
        }
        TimeoutReply::Error => {
            assert_eq!(terminal["type"], "response.failed", "{terminal}");
            assert_eq!(terminal["response"]["error"]["code"], "request_timeout", "{terminal}");
        }
    }
    rig.the_lane_comes_back().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_socket_response_its_deadline_cut_ends_incomplete() {
    check_a_socket_response_its_deadline_cut(TimeoutReply::Partial).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_socket_response_its_deadline_cut_ends_failed_under_the_error_reply() {
    check_a_socket_response_its_deadline_cut(TimeoutReply::Error).await;
}

/// A socket's response is held to `server.stream_timeout` alone, and `0`
/// is none.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_zero_stream_timeout_leaves_a_socket_response_to_its_end() {
    let rig = rig(&["Hello", " world"], 0, |server| server.with_request_timeout(Duration::from_millis(20)).with_stream_timeout(Duration::ZERO));
    let live = responses::live(rig.server.clone()).await;
    let mut socket = responses::socket(&live).await;
    responses::send(&mut socket, create(2)).await;
    rig.gate.wait_entered();
    // Time passing is the point here, not a synchronization.
    tokio::time::sleep(Duration::from_millis(100)).await;
    rig.gate.release();
    let events = responses::events_until(&mut socket, responses::terminal).await;
    let terminal = events.last().unwrap();
    assert_eq!(terminal["type"], "response.incomplete", "its cap: {terminal}");
    assert_eq!(output_text(&terminal["response"]), "Hello world", "{terminal}");
}
