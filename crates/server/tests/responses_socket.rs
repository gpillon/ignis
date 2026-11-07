//! The Responses API's WebSocket mode (GitHub #282, spec responses-api/01,
//! seam 2): a real socket on a live listener, a `tokio-tungstenite` client,
//! `MockCompute` / `GatedCompute` underneath. Only what is socket-specific is
//! here — the event model itself is `responses_http.rs`'s.
//!
//! Three tests also pin the golden transcripts under
//! `tests/fixtures/responses/` (JSON Lines, one event per line exactly as the
//! socket sent it), which the Playground's own tests read. Run with
//! `IGNIS_WRITE_FIXTURES=1` to rewrite them after a deliberate change.

#[path = "support/responses.rs"]
mod responses;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use futures_util::SinkExt;
use ignis_core::mock::GatedCompute;
use ignis_core::{Compute, MockCompute, SchedulerConfig};
use ignis_server::deadline::TimeoutReply;
use ignis_server::images::job::base64;
use ignis_server::loop_guard::{repair, LoopGuardMode, LoopGuardParams, TAG_PREFIX};
use serde_json::{json, Value as JsonValue};
use tokio_tungstenite::tungstenite::Message;

use responses::{
    live, metric_reaches, next_event, next_frame, send, server, socket, sse_events, terminal, Script, Socket, MODEL,
};

fn plain() -> Arc<MockCompute> {
    Arc::new(MockCompute::new())
}

/// Request 0's scripted turn: a thought, a sentence, a tool call.
fn agent_turn() -> HashMap<u32, &'static str> {
    let mock = MockCompute::new();
    HashMap::from([
        (mock.token_for(0, 0), "planning"),
        (mock.token_for(0, 1), "</think>Reading it. "),
        (
            mock.token_for(0, 2),
            "<tool_call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>",
        ),
    ])
}

fn agent_body() -> JsonValue {
    json!({
        "model": MODEL,
        "input": "read a.txt",
        "tools": [{ "type": "function", "name": "read_file", "parameters": {
            "type": "object", "properties": { "path": { "type": "string" } }
        } }],
        "max_output_tokens": 16,
    })
}

/// A `response.create` of `body`, on `stream` when named.
fn create(body: &JsonValue, stream: Option<&str>) -> JsonValue {
    let mut event = body.clone();
    event["type"] = json!("response.create");
    if let Some(stream) = stream {
        event["stream_id"] = json!(stream);
    }
    event
}

/// A short request: one token, thinking off.
fn short(input: &str) -> JsonValue {
    json!({ "model": MODEL, "input": input, "max_output_tokens": 1, "enable_thinking": false })
}

/// A request that holds its lane until it is cancelled or times out: a
/// million tokens on a load sized for them. Its server's template is
/// [`Script::silent`], so it holds the lane without flooding the socket.
fn hog() -> JsonValue {
    json!({ "model": MODEL, "input": "hog", "max_output_tokens": 1_000_000, "enable_thinking": false })
}

/// A load with room for [`hog`]s and `lanes` requests in flight.
fn hog_config(lanes: usize) -> SchedulerConfig {
    SchedulerConfig {
        max_in_flight: lanes,
        max_sequence_tokens: 2 << 20,
        // Room for eight hogs at once: a request never waits on pages, so
        // none is ever evicted.
        kv_capacity_pages: (2 << 20) / 16 * 8,
        ..SchedulerConfig::default()
    }
}

/// Events on `socket` until the terminal event of the response on `stream`
/// (the default stream when `None`), keeping only that stream's.
async fn stream_until_terminal(socket: &mut Socket, stream: Option<&str>) -> Vec<JsonValue> {
    let mut kept = Vec::new();
    loop {
        let event = next_event(socket).await;
        if event.get("stream_id").and_then(JsonValue::as_str) != stream {
            continue;
        }
        let done = terminal(&event);
        kept.push(event);
        if done {
            return kept;
        }
    }
}

/// The next event on `stream` of type `kind`, skipping the rest.
async fn next_of(socket: &mut Socket, stream: Option<&str>, kind: &str) -> JsonValue {
    loop {
        let event = next_event(socket).await;
        if event.get("stream_id").and_then(JsonValue::as_str) == stream && event["type"] == kind {
            return event;
        }
    }
}

// ── golden transcripts ──────────────────────────────────────────────────

/// Compare `frames` with the golden transcript `name`, or rewrite it.
fn golden(name: &str, frames: &[String]) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/responses").join(name);
    let text: String = frames.iter().map(|frame| format!("{frame}\n")).collect();
    if std::env::var_os("IGNIS_WRITE_FIXTURES").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &text).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(text, expected.replace("\r\n", "\n"), "{} drifted", path.display());
}

/// Raw frames on `stream` until its terminal event.
async fn frames_until_terminal(socket: &mut Socket, stream: &str) -> Vec<String> {
    let mut frames = Vec::new();
    loop {
        let frame = next_frame(socket).await;
        let event: JsonValue = serde_json::from_str(&frame).unwrap();
        if event["stream_id"] != stream {
            continue;
        }
        let done = terminal(&event);
        frames.push(frame);
        if done {
            return frames;
        }
    }
}

/// Golden (a): reasoning, text and one function call on a named stream.
#[tokio::test]
async fn golden_a_function_call_on_a_named_stream() {
    let mock = plain();
    mock.eos_after(0, 3);
    let live = live(server(&Script::new(agent_turn()), SchedulerConfig::default(), mock)).await;
    let mut socket = socket(&live).await;
    send(&mut socket, create(&agent_body(), Some("main"))).await;
    let frames = frames_until_terminal(&mut socket, "main").await;
    let last: JsonValue = serde_json::from_str(frames.last().unwrap()).unwrap();
    assert_eq!(last["type"], "response.completed");
    let types: Vec<&str> = last["response"]["output"].as_array().unwrap().iter().map(|i| i["type"].as_str().unwrap()).collect();
    assert_eq!(types, ["reasoning", "message", "function_call"]);
    golden("named_stream_function_call.jsonl", &frames);
}

/// Golden (b): a request queued behind a full engine, then cancelled.
#[tokio::test]
async fn golden_b_a_queued_request_cancelled() {
    let live = live(server(&Script::silent(), hog_config(1), plain())).await;
    let mut socket = socket(&live).await;
    send(&mut socket, create(&hog(), Some("hog"))).await;
    next_of(&mut socket, Some("hog"), "response.in_progress").await;
    send(&mut socket, create(&short("wait"), Some("main"))).await;
    let mut frames = Vec::new();
    loop {
        let frame = next_frame(&mut socket).await;
        let event: JsonValue = serde_json::from_str(&frame).unwrap();
        if event["stream_id"] != "main" {
            continue;
        }
        let queued = event["type"] == "response.queued";
        let id = event["response"]["id"].as_str().unwrap().to_owned();
        frames.push(frame);
        if queued {
            send(&mut socket, json!({ "type": "response.cancel", "response_id": id })).await;
            break;
        }
    }
    frames.extend(frames_until_terminal(&mut socket, "main").await);
    let last: JsonValue = serde_json::from_str(frames.last().unwrap()).unwrap();
    assert_eq!(last["type"], "response.incomplete");
    assert_eq!(last["response"]["status"], "cancelled");
    golden("queued_then_cancelled.jsonl", &frames);
}

/// Golden (c): a request-scoped `error` event on a named stream.
#[tokio::test]
async fn golden_c_an_error_on_a_named_stream() {
    let live = live(server(&Script::new(HashMap::new()), SchedulerConfig::default(), plain())).await;
    let mut socket = socket(&live).await;
    let mut body = short("hi");
    body["previous_response_id"] = json!("resp_404");
    send(&mut socket, create(&body, Some("main"))).await;
    let frame = next_frame(&mut socket).await;
    let event: JsonValue = serde_json::from_str(&frame).unwrap();
    assert_eq!(event["type"], "error");
    assert_eq!(event["stream_id"], "main");
    assert_eq!(event["error"]["code"], "previous_response_not_found");
    golden("named_stream_error.jsonl", &[frame]);
}

// ── acceptance 6: upgrade (the key: `responses_socket_key.rs`) ───────────

#[tokio::test]
async fn a_get_that_is_not_an_upgrade_is_a_400() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let live = live(server(&Script::new(HashMap::new()), SchedulerConfig::default(), plain())).await;
    let mut stream = tokio::net::TcpStream::connect(live.addr).await.unwrap();
    stream
        .write_all(b"GET /v1/responses HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .unwrap();
    let mut answer = String::new();
    stream.read_to_string(&mut answer).await.unwrap();
    assert!(answer.starts_with("HTTP/1.1 400"), "{answer}");
    assert!(answer.contains("WebSocket"), "{answer}");
}

// ── acceptance 7: the same events as HTTP; client events ────────────────

#[tokio::test]
async fn a_socket_response_is_the_http_streams_events() {
    let http = {
        let mock = plain();
        mock.eos_after(0, 3);
        server(&Script::new(agent_turn()), SchedulerConfig::default(), mock).app()
    };
    let mut body = agent_body();
    body["stream"] = json!(true);
    let request = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body.to_string()))
        .unwrap();
    let answer = tower::ServiceExt::oneshot(http, request).await.unwrap();
    let text = axum::body::to_bytes(answer.into_body(), usize::MAX).await.unwrap();
    let over_http: Vec<JsonValue> = sse_events(std::str::from_utf8(&text).unwrap()).into_iter().map(|(_, e)| e).collect();

    let mock = plain();
    mock.eos_after(0, 3);
    let live = live(server(&Script::new(agent_turn()), SchedulerConfig::default(), mock)).await;
    let mut socket = socket(&live).await;
    // `stream` is accepted and ignored, as Codex always sends it.
    send(&mut socket, create(&body, None)).await;
    let over_socket = stream_until_terminal(&mut socket, None).await;
    assert_eq!(over_socket, over_http);
}

#[tokio::test]
async fn refused_events_and_bad_frames_are_error_events_and_the_socket_stays_open() {
    let live = live(server(&Script::new(HashMap::new()), SchedulerConfig::default(), plain())).await;
    let mut socket = socket(&live).await;

    send(&mut socket, json!({ "type": "response.steer", "previous_response_id": "resp_1", "input": [] })).await;
    let steer = next_event(&mut socket).await;
    assert_eq!(steer["type"], "error");
    assert_eq!(steer["error"]["code"], "steering_not_supported");
    assert_eq!(steer["status"], 400);

    socket.send(Message::Binary(vec![1, 2, 3].into())).await.unwrap();
    let binary = next_event(&mut socket).await;
    assert_eq!(binary["error"]["type"], "invalid_request_error");
    assert!(binary["error"]["param"].is_null());

    socket.send(Message::Text("not json".into())).await.unwrap();
    assert_eq!(next_event(&mut socket).await["error"]["param"], JsonValue::Null);

    send(&mut socket, json!({ "type": "session.update" })).await;
    let unknown = next_event(&mut socket).await;
    assert_eq!(unknown["error"]["type"], "invalid_request_error");
    assert_eq!(unknown["error"]["param"], "type");

    // A request the server refuses: its HTTP status and body, as an event.
    let mut hosted = short("hi");
    hosted["tools"] = json!([{ "type": "web_search" }]);
    send(&mut socket, create(&hosted, None)).await;
    let refused = next_event(&mut socket).await;
    assert_eq!(refused["type"], "error");
    assert_eq!(refused["status"], 400);
    assert_eq!(refused["error"]["param"], "tools[0].type");
    assert!(refused.get("stream_id").is_none(), "the default stream's events carry no stream_id");

    // Spec server/09: the field table is the socket's too.
    let mut stored = short("hi");
    stored["conversation"] = json!("conv_1");
    send(&mut socket, create(&stored, None)).await;
    let refused = next_event(&mut socket).await;
    assert_eq!(refused["status"], 400);
    assert_eq!(refused["error"]["param"], "conversation");
    assert_eq!(refused["error"]["code"], "unsupported_value");

    send(&mut socket, create(&short("still here"), None)).await;
    assert_eq!(stream_until_terminal(&mut socket, None).await.last().unwrap()["type"], "response.incomplete");
}

/// Spec image-model/05: the socket is a text route like the HTTP one. With
/// the image model loaded a `response.create` naming no model is refused as
/// over HTTP — `409 model_kind_mismatch`, as an `error` event — and the
/// socket stays open.
#[tokio::test]
async fn on_the_image_model_a_socket_response_is_a_kind_mismatch_and_the_socket_stays_open() {
    let runner = ignis_core::image::FakeImageRunner::default();
    let image = ignis_server::images::ImageModel::new("ignis-image-2.1-turbo", Box::new(runner));
    let live = live(ignis_server::Server::from_active(ignis_server::ActiveModel::image(image))).await;
    let mut socket = socket(&live).await;
    let mut body = short("hi");
    body.as_object_mut().unwrap().remove("model");
    for _ in 0..2 {
        send(&mut socket, create(&body, None)).await;
        let refused = next_event(&mut socket).await;
        assert_eq!(refused["type"], "error", "{refused}");
        assert_eq!(refused["status"], 409, "{refused}");
        assert_eq!(refused["error"]["code"], "model_kind_mismatch", "{refused}");
    }
}

// ── acceptance 8: response.cancel ────────────────────────────────────────

#[tokio::test]
async fn a_cancel_ends_one_running_response_and_leaves_the_others_running() {
    let (gated, gate) = GatedCompute::new(plain());
    let server = server(&Script::new(HashMap::new()), SchedulerConfig::default(), gated.clone() as Arc<dyn Compute>)
        .with_metrics();
    let metrics = Arc::clone(server.metrics.as_ref().unwrap());
    let live = live(server).await;
    let mut socket = socket(&live).await;

    gated.arm();
    send(&mut socket, create(&json!({ "model": MODEL, "input": "held", "max_output_tokens": 50, "enable_thinking": false }), None)).await;
    let created = next_of(&mut socket, None, "response.created").await;
    let id = created["response"]["id"].as_str().unwrap().to_owned();
    let gate = tokio::task::spawn_blocking(move || {
        gate.wait_entered();
        gate
    })
    .await
    .unwrap();
    // Another stream's request waits behind the held engine meanwhile.
    send(&mut socket, create(&short("other"), Some("b"))).await;
    send(&mut socket, json!({ "type": "response.cancel", "response_id": id })).await;
    let cancelled = next_of(&mut socket, None, "response.incomplete").await;
    assert_eq!(cancelled["response"]["status"], "cancelled");
    assert_eq!(cancelled["response"]["id"], id.as_str());
    tokio::task::spawn_blocking(move || gate.release()).await.unwrap();

    let other = stream_until_terminal(&mut socket, Some("b")).await;
    assert_eq!(other.last().unwrap()["type"], "response.incomplete");
    assert_eq!(other.last().unwrap()["response"]["incomplete_details"]["reason"], "max_output_tokens");
    metric_reaches(&metrics, "ignis_requests_cancelled_total 1").await;

    send(&mut socket, json!({ "type": "response.cancel", "response_id": id })).await;
    let unknown = next_event(&mut socket).await;
    assert_eq!(unknown["type"], "error");
    assert_eq!(unknown["error"]["code"], "response_not_found");
}

// ── acceptance 9: streams ─────────────────────────────────────────────────

#[tokio::test]
async fn a_stream_runs_in_order_while_streams_run_side_by_side() {
    let live = live(server(&Script::silent(), hog_config(8), plain())).await;
    let mut socket = socket(&live).await;
    // Stream a's first response is a hog: it ends only when the test cancels
    // it, so no timing can finish it before stream b's response starts.
    send(&mut socket, create(&hog(), Some("a"))).await;
    send(&mut socket, create(&short("second on a"), Some("a"))).await;
    send(&mut socket, create(&short("first on b"), Some("b"))).await;

    // (stream, type) of every lifecycle event, in arrival order.
    let mut seen: Vec<(String, String)> = Vec::new();
    let mut a_first_id = None;
    let mut terminals = 0;
    let mut cancelled = false;
    while terminals < 3 {
        let event = next_event(&mut socket).await;
        let kind = event["type"].as_str().unwrap().to_owned();
        let stream = event["stream_id"].as_str().expect("named-stream events carry stream_id").to_owned();
        if terminal(&event) {
            terminals += 1;
        }
        if kind == "response.created" || terminal(&event) {
            if a_first_id.is_none() && stream == "a" {
                a_first_id = Some(event["response"]["id"].clone());
            }
            seen.push((stream, kind));
        }
        // b's response ended while a's hog still holds a's first place:
        // release it, and a's second response may start.
        if !cancelled && seen.iter().any(|(s, k)| s == "b" && k.starts_with("response.incomplete")) {
            cancelled = true;
            send(&mut socket, json!({ "type": "response.cancel", "response_id": a_first_id.clone().unwrap() })).await;
        }
    }
    let at = |stream: &str, kind: &str, nth: usize| {
        seen.iter()
            .enumerate()
            .filter(|(_, (s, k))| s == stream && k.starts_with(kind))
            .nth(nth)
            .unwrap_or_else(|| panic!("no {stream} {kind} #{nth} in {seen:?}"))
            .0
    };
    // FIFO within a stream: a's second response starts after its first ends.
    assert!(at("a", "response.created", 1) > at("a", "response.incomplete", 0), "{seen:?}");
    // Concurrent across streams: b's response ends while a's first still runs.
    assert!(at("b", "response.incomplete", 0) < at("a", "response.incomplete", 0), "{seen:?}");
}

#[tokio::test]
async fn stream_ids_are_checked_and_limited_to_32_names() {
    let live = live(server(&Script::new(HashMap::new()), SchedulerConfig::default(), plain())).await;
    let mut socket = socket(&live).await;
    for bad in [json!(""), json!("a b"), json!("x".repeat(257)), json!(7)] {
        let mut event = create(&short("hi"), None);
        event["stream_id"] = bad.clone();
        send(&mut socket, event).await;
        let refused = next_event(&mut socket).await;
        assert_eq!(refused["error"]["code"], "invalid_stream_id", "{bad}");
        assert_eq!(refused["error"]["param"], "stream_id");
    }
    // A create whose body is refused on receipt takes no name.
    for n in 0..3 {
        let mut malformed = create(&short("hi"), Some(&format!("m{n}")));
        malformed["max_output_tokens"] = json!("many");
        send(&mut socket, malformed).await;
        let refused = next_event(&mut socket).await;
        assert_eq!(refused["type"], "error");
        assert_eq!(refused["stream_id"], format!("m{n}"));
    }
    // Each name is taken by a request accepted and then refused before it
    // runs, which still counts: 32 names, and the 33rd is refused.
    let mut hosted = short("hi");
    hosted["tools"] = json!([{ "type": "file_search" }]);
    for n in 0..32 {
        send(&mut socket, create(&hosted, Some(&format!("s{n}")))).await;
        let refused = next_event(&mut socket).await;
        assert_eq!(refused["error"]["param"], "tools[0].type");
        assert_eq!(refused["stream_id"], format!("s{n}"));
    }
    send(&mut socket, create(&short("hi"), Some("s32"))).await;
    let limited = next_event(&mut socket).await;
    assert_eq!(limited["error"]["code"], "websocket_stream_limit_reached");
    assert_eq!(limited["stream_id"], "s32");
    // A name already in use, and the default stream, still run.
    send(&mut socket, create(&short("hi"), Some("s0"))).await;
    assert_eq!(stream_until_terminal(&mut socket, Some("s0")).await.last().unwrap()["type"], "response.incomplete");
    send(&mut socket, create(&short("hi"), None)).await;
    assert_eq!(stream_until_terminal(&mut socket, None).await.last().unwrap()["type"], "response.incomplete");
}

/// Sixteen hogs hold the connection's sixteen places (eight running, eight
/// in the admission queue); the seventeenth and eighteenth start only as
/// places free, in the order they arrived.
#[tokio::test]
async fn a_connection_has_at_most_16_responses_active_and_the_rest_wait_in_order() {
    let live = live(server(&Script::silent(), hog_config(8), plain())).await;
    let mut socket = socket(&live).await;
    for n in 0..18 {
        send(&mut socket, create(&hog(), Some(&format!("s{n:02}")))).await;
    }
    let mut ids = HashMap::new();
    while ids.len() < 16 {
        let event = next_event(&mut socket).await;
        if event["type"] == "response.created" {
            ids.insert(event["stream_id"].as_str().unwrap().to_owned(), event["response"]["id"].clone());
        }
    }
    let mut first_sixteen: Vec<&String> = ids.keys().collect();
    first_sixteen.sort();
    assert_eq!(first_sixteen.last().unwrap().as_str(), "s15", "the first sixteen to arrive start");
    for (freed, waiting) in [("s00", "s16"), ("s01", "s17")] {
        send(&mut socket, json!({ "type": "response.cancel", "response_id": ids[freed] })).await;
        let mut freed_ended = false;
        loop {
            let event = next_event(&mut socket).await;
            let stream = event["stream_id"].as_str().unwrap_or_default();
            if stream == freed && terminal(&event) {
                freed_ended = true;
            }
            if event["type"] == "response.created" {
                assert_eq!(stream, waiting, "the next to arrive starts next");
                assert!(freed_ended, "{waiting} started only once a place was free");
                break;
            }
        }
    }
}

// ── acceptance 10: continuation ──────────────────────────────────────────

/// The conversation the template received for the latest render.
fn last_render(script: &Script) -> Vec<(String, String)> {
    script.seen().last().unwrap().messages.iter().map(|m| (m.role.clone(), m.content.text())).collect()
}

#[tokio::test]
async fn a_continuation_renders_the_parents_whole_conversation_without_its_instructions() {
    let mock = MockCompute::new();
    let script = Script::new(HashMap::from([(mock.token_for(0, 0), "hello")]));
    let live = live(server(&script, SchedulerConfig::default(), plain())).await;
    let mut socket = socket(&live).await;
    let mut first = short("hi");
    first["instructions"] = json!("be brief");
    send(&mut socket, create(&first, None)).await;
    let parent = stream_until_terminal(&mut socket, None).await.pop().unwrap();
    let parent_id = parent["response"]["id"].as_str().unwrap().to_owned();

    let mut second = short("and then?");
    second["previous_response_id"] = json!(parent_id);
    send(&mut socket, create(&second, None)).await;
    let child = stream_until_terminal(&mut socket, None).await.pop().unwrap();
    assert_eq!(child["response"]["previous_response_id"], parent_id.as_str());
    assert_eq!(
        last_render(&script),
        [
            ("user".to_owned(), "hi".to_owned()),
            ("assistant".to_owned(), "hello".to_owned()),
            ("user".to_owned(), "and then?".to_owned()),
        ],
        "the parent's input and output, then the new items; instructions are not inherited"
    );
}

#[tokio::test]
async fn unknown_superseded_and_foreign_ids_are_not_found() {
    let live = live(server(&Script::new(HashMap::new()), SchedulerConfig::default(), plain())).await;
    let mut socket = socket(&live).await;
    let mut other = responses::socket(&live).await;
    send(&mut socket, create(&short("one"), None)).await;
    let first = stream_until_terminal(&mut socket, None).await.pop().unwrap()["response"]["id"].clone();
    send(&mut other, create(&short("foreign"), None)).await;
    let foreign = stream_until_terminal(&mut other, None).await.pop().unwrap()["response"]["id"].clone();
    send(&mut socket, create(&short("two"), None)).await;
    stream_until_terminal(&mut socket, None).await;

    for id in [json!("resp_unknown"), first, foreign] {
        let mut body = short("next");
        body["previous_response_id"] = id.clone();
        send(&mut socket, create(&body, Some("x"))).await;
        let refused = next_event(&mut socket).await;
        assert_eq!(refused["error"]["code"], "previous_response_not_found", "{id}");
        assert_eq!(refused["error"]["param"], "previous_response_id");
        assert_eq!(refused["stream_id"], "x");
    }
}

#[tokio::test]
async fn a_failed_same_stream_continuation_evicts_its_parent_and_a_failed_fork_does_not() {
    let live = live(server(&Script::new(HashMap::new()), SchedulerConfig::default(), plain())).await;
    let mut socket = socket(&live).await;
    let hosted = |id: &JsonValue| {
        let mut body = short("next");
        body["previous_response_id"] = id.clone();
        body["tools"] = json!([{ "type": "web_search" }]);
        body
    };
    let continuing = |id: &JsonValue| {
        let mut body = short("next");
        body["previous_response_id"] = id.clone();
        body
    };

    // A fork from s1 onto s2 fails; s1's latest survives and can be continued.
    send(&mut socket, create(&short("one"), Some("s1"))).await;
    let parent = stream_until_terminal(&mut socket, Some("s1")).await.pop().unwrap()["response"]["id"].clone();
    send(&mut socket, create(&hosted(&parent), Some("s2"))).await;
    assert_eq!(next_event(&mut socket).await["type"], "error");
    send(&mut socket, create(&continuing(&parent), Some("s3"))).await;
    let fork = stream_until_terminal(&mut socket, Some("s3")).await.pop().unwrap();
    assert_eq!(fork["type"], "response.incomplete", "a fork onto a new stream: {fork}");

    // A same-stream continuation refused on receipt, before its parent was
    // even resolved, evicts it too.
    send(&mut socket, create(&short("one"), Some("s5"))).await;
    let parent5 = stream_until_terminal(&mut socket, Some("s5")).await.pop().unwrap()["response"]["id"].clone();
    let mut malformed = continuing(&parent5);
    malformed["generate"] = json!("yes");
    send(&mut socket, create(&malformed, Some("s5"))).await;
    assert_eq!(next_event(&mut socket).await["error"]["param"], "generate");
    send(&mut socket, create(&continuing(&parent5), Some("s5"))).await;
    assert_eq!(next_event(&mut socket).await["error"]["code"], "previous_response_not_found");

    // A same-stream continuation fails: its parent is evicted.
    send(&mut socket, create(&short("one"), Some("s4"))).await;
    let parent = stream_until_terminal(&mut socket, Some("s4")).await.pop().unwrap()["response"]["id"].clone();
    send(&mut socket, create(&hosted(&parent), Some("s4"))).await;
    assert_eq!(next_event(&mut socket).await["type"], "error");
    send(&mut socket, create(&continuing(&parent), Some("s4"))).await;
    assert_eq!(next_event(&mut socket).await["error"]["code"], "previous_response_not_found");
}

// ── acceptance 12: the admission queue, close, gauges ────────────────────

#[tokio::test]
async fn a_full_engine_queues_socket_requests_in_arrival_order_across_connections() {
    let server = server(&Script::silent(), hog_config(1), plain()).with_metrics();
    let metrics = Arc::clone(server.metrics.as_ref().unwrap());
    let live = live(server).await;
    let mut first = socket(&live).await;
    let mut second = socket(&live).await;
    metric_reaches(&metrics, "ignis_responses_sockets 2").await;

    send(&mut first, create(&hog(), Some("h"))).await;
    let hog_id = next_of(&mut first, Some("h"), "response.in_progress").await["response"]["id"].clone();
    // Behind it, first a hog on the first connection, then a request on the
    // second: both find the engine full.
    send(&mut first, create(&hog(), Some("b"))).await;
    let queued = next_of(&mut first, Some("b"), "response.created").await;
    assert_eq!(queued["response"]["status"], "queued");
    let b_id = queued["response"]["id"].clone();
    next_of(&mut first, Some("b"), "response.queued").await;
    send(&mut second, create(&short("c"), None)).await;
    assert_eq!(next_of(&mut second, None, "response.created").await["response"]["status"], "queued");
    next_of(&mut second, None, "response.queued").await;
    metric_reaches(&metrics, "ignis_responses_queued_requests 2").await;

    // The first hog goes: the head of the queue, the first connection's, is
    // admitted; the second connection's still waits behind it.
    send(&mut first, json!({ "type": "response.cancel", "response_id": hog_id })).await;
    next_of(&mut first, Some("b"), "response.in_progress").await;
    metric_reaches(&metrics, "ignis_responses_queued_requests 1").await;
    send(&mut first, json!({ "type": "response.cancel", "response_id": b_id })).await;
    let c = stream_until_terminal(&mut second, None).await;
    assert!(c.iter().any(|e| e["type"] == "response.in_progress"));
    assert_eq!(c.last().unwrap()["response"]["status"], "incomplete");
    metric_reaches(&metrics, "ignis_responses_queued_requests 0").await;
}

/// A socket's responses are held to `server.stream_timeout` (ADR 0058),
/// counted from admission; one cut by it ends `incomplete`, as its default
/// partial reply says.
#[tokio::test]
async fn the_stream_timeout_counts_from_admission_not_from_queueing() {
    let server = server(&Script::silent(), hog_config(1), plain())
        .with_stream_timeout(Duration::from_millis(400));
    let live = live(server).await;
    let mut socket = socket(&live).await;
    send(&mut socket, create(&hog(), Some("h"))).await;
    next_of(&mut socket, Some("h"), "response.in_progress").await;
    send(&mut socket, create(&short("after"), Some("b"))).await;
    next_of(&mut socket, Some("b"), "response.queued").await;
    // The hog times out mid-stream, a whole timeout after the queued request
    // arrived; that request still has its own timeout from admission.
    let hog = next_of(&mut socket, Some("h"), "response.incomplete").await;
    assert_eq!(hog["response"]["incomplete_details"]["reason"], "max_output_tokens");
    let after = stream_until_terminal(&mut socket, Some("b")).await;
    assert_eq!(after.last().unwrap()["type"], "response.incomplete", "{:?}", after.last());
}

#[tokio::test]
async fn closing_the_socket_cancels_what_it_had_running_and_queued() {
    let server = server(&Script::silent(), hog_config(1), plain()).with_metrics();
    let metrics = Arc::clone(server.metrics.as_ref().unwrap());
    let live = live(server).await;
    let mut socket = socket(&live).await;
    send(&mut socket, create(&hog(), Some("h"))).await;
    next_of(&mut socket, Some("h"), "response.in_progress").await;
    send(&mut socket, create(&hog(), Some("q"))).await;
    next_of(&mut socket, Some("q"), "response.queued").await;
    metric_reaches(&metrics, "ignis_responses_queued_requests 1").await;
    metric_reaches(&metrics, "ignis_responses_sockets 1").await;

    socket.close(None).await.unwrap();
    drop(socket);
    metric_reaches(&metrics, "ignis_requests_cancelled_total 1").await;
    metric_reaches(&metrics, "ignis_responses_queued_requests 0").await;
    metric_reaches(&metrics, "ignis_responses_sockets 0").await;
}

// ── acceptance 11: the warm-up (`generate: false`) ───────────────────────

/// Twenty words: past one whole 16-token page, where retained state can be
/// kept.
fn words(prefix: &str) -> String {
    (0..20).map(|n| format!("{prefix}{n}")).collect::<Vec<_>>().join(" ")
}

/// Warm up with `warm_up` (as `generate: false`), then continue from it with
/// `turn` on the same socket; the warm-up's terminal event and the turn's
/// `cached_tokens`.
async fn warm_then_continue(config: SchedulerConfig, warm_up: JsonValue, turn: JsonValue) -> (JsonValue, JsonValue) {
    let live = live(server(&Script::with_boundaries(HashMap::new()), config, plain())).await;
    let mut socket = socket(&live).await;
    let mut event = create(&warm_up, None);
    event["generate"] = json!(false);
    send(&mut socket, event).await;
    let warmed = stream_until_terminal(&mut socket, None).await;
    let done = warmed.last().unwrap().clone();
    let mut turn = create(&turn, None);
    turn["previous_response_id"] = done["response"]["id"].clone();
    send(&mut socket, turn).await;
    let answered = stream_until_terminal(&mut socket, None).await.pop().unwrap();
    assert_ne!(answered["type"], "error", "{answered}");
    (done, answered["response"]["usage"]["input_tokens_details"]["cached_tokens"].clone())
}

#[tokio::test]
async fn a_warm_up_prefills_without_generating_and_the_next_turn_resumes_from_it() {
    let warm_up = json!({ "model": MODEL, "input": words("w"), "max_output_tokens": 64, "enable_thinking": false });
    let turn = json!({ "model": MODEL, "input": "and now answer", "max_output_tokens": 1, "enable_thinking": false });
    let (warmed, cached) = warm_then_continue(SchedulerConfig::default(), warm_up, turn).await;
    assert_eq!(warmed["type"], "response.completed");
    assert_eq!(warmed["response"]["output"], json!([]), "a warm-up has no output");
    assert_eq!(warmed["response"]["usage"]["output_tokens"], 0);
    assert_eq!(cached, 20, "the turn resumed from the checkpoint the warm-up took at its opener");
}

#[tokio::test]
async fn a_warm_up_of_instructions_alone_serves_the_next_turn_its_system_block() {
    // Codex's prewarm: instructions and a tool, no input — which a template
    // that needs a user query cannot render as it stands.
    let tool = json!([{ "type": "function", "name": "shell", "parameters": { "type": "object", "properties": {} } }]);
    let warm_up = json!({ "model": MODEL, "instructions": words("sys"), "tools": tool, "input": [], "enable_thinking": false });
    let turn = json!({ "model": MODEL, "instructions": words("sys"), "tools": tool, "input": "hi", "max_output_tokens": 1, "enable_thinking": false });
    let (warmed, cached) = warm_then_continue(SchedulerConfig::default(), warm_up, turn).await;
    assert_eq!(warmed["type"], "response.completed", "{warmed}");
    assert_eq!(warmed["response"]["usage"]["input_tokens"], 20, "exactly the system block was prefilled");
    assert_eq!(cached, 16, "the turn stood on the retained prefix: the system block's whole page");
}

#[tokio::test]
async fn a_warm_up_with_no_retained_slot_free_still_completes() {
    let config = SchedulerConfig { retained_slots: 0, ..SchedulerConfig::default() };
    let warm_up = json!({ "model": MODEL, "input": words("w"), "enable_thinking": false });
    let turn = json!({ "model": MODEL, "input": "and now answer", "max_output_tokens": 1, "enable_thinking": false });
    let (warmed, cached) = warm_then_continue(config, warm_up, turn).await;
    assert_eq!(warmed["type"], "response.completed");
    assert_eq!(cached, 0, "nothing was kept, and the turn still ran");
}

/// A queued response keeps the id `response.created` gave it for its whole
/// life — queued, admitted, streamed, ended — and that id is the one a
/// continuation names.
#[tokio::test]
async fn a_queued_response_keeps_one_id_from_created_to_its_end_and_continues_by_it() {
    // The hog is request 0 and generates silently (a million `?` deltas
    // would bury the frames this test waits for); the queued one, request 1,
    // is scripted to stream text.
    let mock = MockCompute::new();
    let script = Script::quiet_except(HashMap::from([
        (mock.token_for(1, 0), "x"),
        (mock.token_for(1, 1), "y"),
        (mock.token_for(1, 2), "z"),
    ]));
    let live = live(server(&script, hog_config(1), plain())).await;
    let mut socket = socket(&live).await;
    send(&mut socket, create(&hog(), Some("hog"))).await;
    let hog_id = next_of(&mut socket, Some("hog"), "response.in_progress").await["response"]["id"].clone();
    let mut queued = short("wait");
    queued["max_output_tokens"] = json!(3);
    send(&mut socket, create(&queued, Some("main"))).await;
    let created = next_of(&mut socket, Some("main"), "response.created").await;
    assert_eq!(created["response"]["status"], "queued");
    let id = created["response"]["id"].as_str().unwrap().to_owned();
    next_of(&mut socket, Some("main"), "response.queued").await;
    send(&mut socket, json!({ "type": "response.cancel", "response_id": hog_id })).await;

    let events = stream_until_terminal(&mut socket, Some("main")).await;
    assert!(events.iter().any(|e| e["type"] == "response.in_progress"), "it was admitted: {events:?}");
    assert!(events.iter().any(|e| e["type"] == "response.output_text.delta"), "and streamed: {events:?}");
    let suffix = id.strip_prefix("resp_").unwrap();
    for event in &events {
        if let Some(response) = event.get("response") {
            assert_eq!(response["id"], id.as_str(), "{event}");
        }
        if let Some(item_id) = event.get("item_id").and_then(JsonValue::as_str) {
            assert!(item_id.ends_with(suffix), "item ids share the response's suffix: {event}");
        }
    }

    let mut next = short("and then?");
    next["previous_response_id"] = json!(id);
    send(&mut socket, create(&next, Some("main"))).await;
    let continued = stream_until_terminal(&mut socket, Some("main")).await.pop().unwrap();
    assert_eq!(continued["type"], "response.incomplete", "{continued}");
    assert_eq!(continued["response"]["previous_response_id"], id.as_str());
}

/// Only "full" queues: a request the engine can never admit is an `error`
/// event at once, whether the engine is merely full or others already wait.
#[tokio::test]
async fn a_request_the_engine_can_never_admit_is_refused_at_once_not_queued() {
    let live = live(server(&Script::silent(), hog_config(1), plain())).await;
    let mut socket = socket(&live).await;
    send(&mut socket, create(&hog(), Some("h"))).await;
    next_of(&mut socket, Some("h"), "response.in_progress").await;
    let mut too_long = short("long");
    too_long["max_output_tokens"] = json!(8 << 20);
    let mut elsewhere = short("model");
    elsewhere["model"] = json!("another-model");

    // The engine is full and nobody waits yet.
    send(&mut socket, create(&too_long, Some("a"))).await;
    let refused = next_of(&mut socket, Some("a"), "error").await;
    assert_eq!(refused["error"]["code"], "context_length_exceeded", "{refused}");

    // Someone waits: the refusals still come at once.
    send(&mut socket, create(&short("wait"), Some("b"))).await;
    next_of(&mut socket, Some("b"), "response.queued").await;
    send(&mut socket, create(&too_long, Some("c"))).await;
    let refused = next_event(&mut socket).await;
    assert_eq!((refused["type"].as_str(), refused["stream_id"].as_str()), (Some("error"), Some("c")), "{refused}");
    assert_eq!(refused["error"]["code"], "context_length_exceeded");
    send(&mut socket, create(&elsewhere, Some("d"))).await;
    let refused = next_event(&mut socket).await;
    assert_eq!((refused["type"].as_str(), refused["stream_id"].as_str()), (Some("error"), Some("d")), "{refused}");
    assert_eq!(refused["status"], 404);
    assert_eq!(refused["error"]["code"], "model_not_found");
}

/// RFC 6455 §5.5.1: a client's Close is answered with the server's own, so
/// the connection ends cleanly rather than as a dropped socket.
#[tokio::test]
async fn a_client_close_is_answered_with_a_close_frame() {
    use futures_util::StreamExt;
    use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
    use tokio_tungstenite::tungstenite::protocol::CloseFrame;

    let live = live(server(&Script::new(HashMap::new()), SchedulerConfig::default(), plain())).await;
    let mut socket = socket(&live).await;
    socket.close(Some(CloseFrame { code: CloseCode::Normal, reason: "bye".into() })).await.unwrap();
    let answer = tokio::time::timeout(Duration::from_secs(10), socket.next()).await.expect("an answer within 10 s");
    match answer {
        Some(Ok(Message::Close(frame))) => {
            assert_eq!(frame.map(|f| f.code), Some(CloseCode::Normal), "the close code is echoed");
        }
        other => panic!("expected the server's Close frame, got {other:?}"),
    }
}

// ── the loop guard (ADR 0052) ─────────────────────────────────────────────

/// A server with the loop guard on (`holdback` characters held back), over
/// `mock`, whose request 0 reasons prose and then a loop the guard rewinds.
fn looping_server(mock: &Arc<MockCompute>, config: SchedulerConfig, holdback: u32) -> (ignis_server::Server, Arc<Script>) {
    let reasoning = responses::prose(3000) + &responses::looping(4000);
    let script = Script::encoding(responses::scripted_text(mock, 0, &reasoning));
    let guard = LoopGuardParams { mode: LoopGuardMode::Rewind, holdback_chars: holdback, ..LoopGuardParams::default() };
    (server(&script, config, mock.clone() as Arc<dyn Compute>).with_loop_guard(guard), script)
}

/// Read the default stream's events until its reasoning has carried a
/// rewind tag.
async fn until_rewound(socket: &mut Socket) {
    let mut reasoning = String::new();
    while !reasoning.contains(TAG_PREFIX) {
        let event = next_event(socket).await;
        assert!(!terminal(&event), "it ended before a rewind: {event}");
        if event["type"] == "response.reasoning_text.delta" {
            reasoning.push_str(event["delta"].as_str().unwrap());
        }
    }
}

/// Wait until the scheduler has prefilled `request`.
async fn prefilled(mock: &MockCompute, request: u64) {
    let found = tokio::time::timeout(Duration::from_secs(10), async {
        while !mock.prefill_calls().iter().flatten().any(|job| job.request == request) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
    assert!(found.is_ok(), "request {request} was never prefilled");
}

/// Wait until the scheduler has released `request`.
async fn released(mock: &MockCompute, request: u64) {
    let found = tokio::time::timeout(Duration::from_secs(10), async {
        while !mock.released_requests().contains(&request) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
    assert!(found.is_ok(), "request {request} was never released: {:?}", mock.released_requests());
}

/// After a rewind the engine request generating for the response is the
/// continuation: `response.cancel`, and closing the socket, cancel it. The
/// looping request it replaced was already cancelled by the rewind, so the
/// count of cancelled requests reaches two: a continuation left to run to
/// its cap would end done, not cancelled. Also when the holdback hid the
/// rewind and holds every token back, so no failed send could tell the
/// guard the client left.
#[tokio::test]
async fn a_cancel_or_a_close_after_a_rewind_stops_the_continuation() {
    for (close, holdback) in [(false, 0), (true, 0), (false, 6000)] {
        let mock = Arc::new(MockCompute::new());
        let (server, _) = looping_server(&mock, hog_config(2), holdback);
        let server = server.with_metrics();
        let metrics = Arc::clone(server.metrics.as_ref().unwrap());
        let live = live(server).await;
        let mut socket = socket(&live).await;
        let body = json!({ "model": MODEL, "input": "the whole history of Rome", "max_output_tokens": 1_000_000 });
        send(&mut socket, create(&body, None)).await;
        let created = next_of(&mut socket, None, "response.created").await;
        let id = created["response"]["id"].as_str().unwrap().to_owned();
        match holdback {
            0 => until_rewound(&mut socket).await,
            _ => prefilled(&mock, 1).await,
        }
        released(&mock, 0).await;
        metric_reaches(&metrics, "ignis_requests_cancelled_total 1").await;
        assert!(!mock.released_requests().contains(&1), "the continuation runs until it is cancelled");
        if close {
            socket.close(None).await.unwrap();
            drop(socket);
        } else {
            send(&mut socket, json!({ "type": "response.cancel", "response_id": id })).await;
            let cancelled = next_of(&mut socket, None, "response.incomplete").await;
            assert_eq!(cancelled["response"]["status"], "cancelled");
        }
        metric_reaches(&metrics, "ignis_requests_cancelled_total 2").await;
        released(&mock, 1).await;
    }
}

/// A rewound response's stream continues from the model's view of it, the
/// struck text and the tag out; with the include its reasoning item's
/// `done` carries that view, and its usage the tokens taken back.
#[tokio::test]
async fn a_rewound_response_is_continued_from_the_models_view() {
    let mock = Arc::new(MockCompute::new());
    mock.eos_after(1, 20);
    let (server, script) = looping_server(&mock, SchedulerConfig::default(), 0);
    let live = live(server).await;
    let mut socket = socket(&live).await;
    let first = json!({
        "model": MODEL,
        "input": "the whole history of Rome",
        "max_output_tokens": 4000,
        "include": ["reasoning.encrypted_content"],
    });
    send(&mut socket, create(&first, None)).await;
    let events = stream_until_terminal(&mut socket, None).await;
    let done = events
        .iter()
        .find(|e| e["type"] == "response.output_item.done" && e["item"]["type"] == "reasoning")
        .expect("the reasoning item");
    let shown = done["item"]["content"][0]["text"].as_str().unwrap();
    let model_view = repair(shown).into_owned();
    assert!(shown.contains(TAG_PREFIX) && !model_view.contains(TAG_PREFIX), "{shown}");
    assert_eq!(done["item"]["encrypted_content"], format!("ignis:v1:{}", base64(model_view.as_bytes())));
    let finished = events.last().unwrap();
    assert_eq!(finished["type"], "response.completed", "{finished}");
    let tag_tokens: u64 = shown.split_once(TAG_PREFIX).unwrap().1.split_once(" TOKENS").unwrap().0.parse().unwrap();
    assert_eq!(finished["response"]["usage"]["output_tokens_details"]["rewound_tokens"], tag_tokens);

    let id = finished["response"]["id"].as_str().unwrap();
    let next = json!({ "model": MODEL, "input": "go on", "max_output_tokens": 1, "previous_response_id": id });
    send(&mut socket, create(&next, None)).await;
    stream_until_terminal(&mut socket, None).await;
    let rendered = script.seen().last().unwrap().messages.iter().find(|m| m.role == "assistant").unwrap().reasoning_content.clone();
    assert_eq!(rendered.as_deref(), Some(model_view.as_str()));
}

/// A response cut short, by `response.cancel` or by its timeout — answered
/// either way `server.timeout_reply` says (ADR 0058) — still gets what the
/// holdback held back (ADR 0052): the reasoning item carries it, its deltas
/// came first, `output_tokens` counts it, and a cancelled response
/// continues from it. Here the holdback hid the rewind, so what it held is
/// the kept reasoning and the nudge, the model's view, of which the client
/// had received nothing.
#[tokio::test]
async fn a_cancel_or_a_timeout_lets_what_the_holdback_held_reach_the_response() {
    let nudge = ignis_server::loop_guard::DEFAULT_NUDGE;
    for reply in [None, Some(TimeoutReply::Partial), Some(TimeoutReply::Error)] {
        let timeout = reply.is_some();
        let mock = Arc::new(MockCompute::new());
        let (server, script) = looping_server(&mock, hog_config(2), 6000);
        let server = match reply {
            Some(reply) => server.with_stream_timeout(Duration::from_millis(500)).with_timeout_reply(reply),
            None => server,
        };
        let live = live(server).await;
        let mut socket = socket(&live).await;
        let body = json!({ "model": MODEL, "input": "the whole history of Rome", "max_output_tokens": 1_000_000 });
        send(&mut socket, create(&body, None)).await;
        let created = next_of(&mut socket, None, "response.created").await;
        let id = created["response"]["id"].as_str().unwrap().to_owned();
        // The continuation runs: the rewind is behind it, held back.
        prefilled(&mock, 1).await;
        if !timeout {
            send(&mut socket, json!({ "type": "response.cancel", "response_id": id })).await;
        }
        let events = stream_until_terminal(&mut socket, None).await;
        let ended = events.last().unwrap();
        match reply {
            None => assert_eq!((&ended["type"], &ended["response"]["status"]), (&json!("response.incomplete"), &json!("cancelled"))),
            Some(TimeoutReply::Partial) => assert_eq!(
                (&ended["type"], &ended["response"]["incomplete_details"]["reason"]),
                (&json!("response.incomplete"), &json!("max_output_tokens"))
            ),
            Some(TimeoutReply::Error) => {
                assert_eq!((&ended["type"], &ended["response"]["error"]["code"]), (&json!("response.failed"), &json!("request_timeout")))
            }
        }
        let deltas: String =
            events.iter().filter(|e| e["type"] == "response.reasoning_text.delta").map(|e| e["delta"].as_str().unwrap()).collect();
        let reasoning = ended["response"]["output"][0]["content"][0]["text"].as_str().unwrap_or_default().to_owned();
        assert_eq!(deltas, reasoning, "the held text was streamed before the end");
        let at = reasoning.find(nudge).unwrap_or_else(|| panic!("the kept reasoning and the nudge: {reasoning:?}"));
        let kept = reasoning[..at].trim_end_matches('\n');
        assert!(kept.chars().count() > 1000 && (responses::prose(3000) + &responses::looping(4000)).starts_with(kept), "{kept:?}");
        assert!(!reasoning.contains(TAG_PREFIX), "the rewind stays hidden");
        let output_tokens = ended["response"]["usage"]["output_tokens"].as_u64().unwrap() as usize;
        assert!(output_tokens >= kept.chars().count() / 8 + nudge.chars().count(), "{}", ended["response"]["usage"]);
        if timeout {
            continue;
        }
        let next = json!({ "model": MODEL, "input": "go on", "max_output_tokens": 1, "previous_response_id": id });
        send(&mut socket, create(&next, None)).await;
        stream_until_terminal(&mut socket, None).await;
        let rendered = script.seen().last().unwrap().messages.iter().find(|m| m.role == "assistant").unwrap().reasoning_content.clone();
        assert_eq!(rendered.as_deref(), Some(reasoning.as_str()), "the cancelled response continues from what it held");
    }
}

/// A WebSocket request's continuation is admitted through the server-wide
/// queue, behind what already waits there (ADR 0052), never past it. One
/// lane: a hog holds it while the looping request and a second hog queue
/// in that order; the first hog goes, the looping request is admitted and
/// rewound, and its continuation joins the queue behind the second hog,
/// which takes the lane the looping request leaves. Straight to the engine,
/// the continuation would take the lane itself, or wait outside the queue.
#[tokio::test]
async fn a_socket_continuation_waits_in_the_queue_behind_what_already_waits() {
    let mock = Arc::new(MockCompute::new());
    // The looping request is admitted second: request 1.
    let script = Script::encoding(responses::scripted_text(&mock, 1, &(responses::prose(3000) + &responses::looping(4000))));
    let guard = LoopGuardParams { mode: LoopGuardMode::Rewind, ..LoopGuardParams::default() };
    let server = server(&script, hog_config(1), mock.clone() as Arc<dyn Compute>).with_loop_guard(guard).with_metrics();
    let metrics = Arc::clone(server.metrics.as_ref().unwrap());
    let live = live(server).await;
    let mut socket = socket(&live).await;
    send(&mut socket, create(&hog(), Some("first"))).await;
    let first = next_of(&mut socket, Some("first"), "response.in_progress").await;
    let looping = json!({ "model": MODEL, "input": "the whole history of Rome", "max_output_tokens": 4000 });
    send(&mut socket, create(&looping, Some("looping"))).await;
    next_of(&mut socket, Some("looping"), "response.queued").await;
    send(&mut socket, create(&hog(), Some("second"))).await;
    let second = next_of(&mut socket, Some("second"), "response.queued").await;
    metric_reaches(&metrics, "ignis_responses_queued_requests 2").await;

    send(&mut socket, json!({ "type": "response.cancel", "response_id": first["response"]["id"] })).await;
    next_of(&mut socket, Some("second"), "response.in_progress").await;
    released(&mock, 1).await;
    metric_reaches(&metrics, "ignis_responses_queued_requests 1").await;
    assert!(!mock.prefill_calls().iter().flatten().any(|job| job.request == 3), "the continuation waits behind the second hog");

    send(&mut socket, json!({ "type": "response.cancel", "response_id": second["response"]["id"] })).await;
    let ended = stream_until_terminal(&mut socket, Some("looping")).await;
    let ended = ended.last().unwrap();
    assert_eq!(ended["type"], "response.incomplete", "{ended}");
    assert!(ended["response"]["output"][0]["content"][0]["text"].as_str().unwrap().contains(TAG_PREFIX), "rewound");
    metric_reaches(&metrics, "ignis_responses_queued_requests 0").await;
    // GitHub #330: a socket's rewind counts on `/metrics` like any other.
    metric_reaches(&metrics, "ignis_loop_guard_loops_total{outcome=\"rewound\"} 1").await;
    let rewound = ended["response"]["usage"]["output_tokens_details"]["rewound_tokens"].as_u64().expect("rewound tokens");
    metric_reaches(&metrics, &format!("ignis_loop_guard_rewound_tokens_total {rewound}")).await;
}

// ── the idle close (GitHub #329) ──────────────────────────────────────────

/// The idle close these tests configure.
const IDLE: Duration = Duration::from_millis(300);

/// The next frame that is not a ping or a pong — a text frame or the close —
/// or `None` when nothing came within `within`.
async fn next_message(socket: &mut Socket, within: Duration) -> Option<Message> {
    use futures_util::StreamExt;
    let deadline = tokio::time::Instant::now() + within;
    loop {
        match tokio::time::timeout_at(deadline, socket.next()).await {
            Err(_) => return None,
            Ok(Some(Ok(Message::Ping(_) | Message::Pong(_)))) => continue,
            Ok(Some(Ok(message))) => return Some(message),
            Ok(other) => panic!("the socket ended without a close frame: {other:?}"),
        }
    }
}

/// The server's close frame, within a generous bound: its code and reason.
async fn closed(socket: &mut Socket) -> (tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode, String) {
    match next_message(socket, Duration::from_secs(10)).await {
        Some(Message::Close(Some(frame))) => (frame.code, frame.reason.to_string()),
        other => panic!("expected the server's close frame, got {other:?}"),
    }
}

/// A socket nobody uses for `server.socket_idle_close` is closed by the
/// server, with a normal close frame whose reason says why — so a tab left
/// open no longer holds a Knative pod, whose drain waits on every open
/// socket — and it leaves the open-socket gauge.
#[tokio::test]
async fn an_idle_socket_is_closed_with_a_normal_close_frame_saying_why() {
    use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;

    let server = server(&Script::new(HashMap::new()), SchedulerConfig::default(), plain()).with_metrics().with_socket_idle_close(IDLE);
    let metrics = Arc::clone(server.metrics.as_ref().unwrap());
    let live = live(server).await;
    let opened = std::time::Instant::now();
    let mut socket = socket(&live).await;
    let (code, reason) = closed(&mut socket).await;
    assert!(opened.elapsed() >= IDLE, "closed after {:?}", opened.elapsed());
    assert_eq!(code, CloseCode::Normal);
    assert!(reason.contains("idle") && reason.contains("server.socket_idle_close"), "{reason}");
    metric_reaches(&metrics, "ignis_responses_sockets 0").await;
}

/// The server waits for the client's Close after its own, but only so long:
/// a peer that read the frame and went silent (here, never read again, so
/// its echo is never flushed) does not hold the connection, or the gauge,
/// open.
#[tokio::test]
async fn a_peer_silent_after_the_idle_close_does_not_hold_the_connection() {
    let server = server(&Script::new(HashMap::new()), SchedulerConfig::default(), plain()).with_metrics().with_socket_idle_close(IDLE);
    let metrics = Arc::clone(server.metrics.as_ref().unwrap());
    let live = live(server).await;
    let mut socket = socket(&live).await;
    closed(&mut socket).await;
    let closed_at = std::time::Instant::now();
    metric_reaches(&metrics, "ignis_responses_sockets 0").await;
    assert!(closed_at.elapsed() < Duration::from_secs(5), "held {:?} by a silent peer", closed_at.elapsed());
    drop(socket);
}

/// A response running or queued keeps its socket open however long it
/// takes, and the idle time counts from its end: here a hog runs out a
/// stream timeout of three idle times while a second socket's request
/// waits for its lane, and each socket closes only after its own response
/// has ended.
#[tokio::test]
async fn a_response_running_or_queued_keeps_its_socket_open_and_the_idle_time_counts_from_its_end() {
    let timeout = IDLE * 3;
    let server = server(&Script::silent(), hog_config(1), plain()).with_stream_timeout(timeout).with_socket_idle_close(IDLE);
    let live = live(server).await;
    let mut running = socket(&live).await;
    let mut queued = socket(&live).await;
    let sent = std::time::Instant::now();
    send(&mut running, create(&hog(), None)).await;
    next_of(&mut running, None, "response.in_progress").await;
    send(&mut queued, create(&short("after"), None)).await;
    next_of(&mut queued, None, "response.queued").await;
    // A close frame here panics `next_event`: neither socket may close
    // before its response ends.
    let hog = stream_until_terminal(&mut running, None).await;
    assert_eq!(hog.last().unwrap()["response"]["incomplete_details"]["reason"], "max_output_tokens", "{:?}", hog.last());
    let after = stream_until_terminal(&mut queued, None).await;
    assert_eq!(after.last().unwrap()["type"], "response.incomplete", "{:?}", after.last());
    closed(&mut running).await;
    closed(&mut queued).await;
    // The hog ended a whole timeout after it was sent, and the clock started
    // there, not at the socket's last frame.
    assert!(sent.elapsed() >= timeout + IDLE, "closed {:?} after the hog was sent", sent.elapsed());
}

/// A keep-alive ping proves the connection, not a user: a client that
/// pings (the Python `websockets` library does, every 20 s by default) is
/// still closed once nothing else comes.
#[tokio::test]
async fn pings_do_not_keep_an_idle_socket_open() {
    let live = live(server(&Script::new(HashMap::new()), SchedulerConfig::default(), plain()).with_socket_idle_close(IDLE)).await;
    // GitHub #492: before the upgrade. The server's idle time starts when it
    // builds the socket's handler, during the upgrade and so before
    // `socket` returns: a clock taken after it starts late, and read the
    // close as early by the handshake's tail.
    let opened = std::time::Instant::now();
    let mut socket = socket(&live).await;
    let frame = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            socket.send(Message::Ping(Default::default())).await.expect("ping");
            if let Some(message) = next_message(&mut socket, IDLE / 3).await {
                return message;
            }
        }
    })
    .await
    .expect("closed within 10 s, pings or not");
    assert!(matches!(frame, Message::Close(Some(_))), "{frame:?}");
    assert!(opened.elapsed() >= IDLE);
}

/// A client's text frame restarts the idle time, whatever it carries.
#[tokio::test]
async fn a_client_frame_restarts_the_idle_time() {
    let live = live(server(&Script::new(HashMap::new()), SchedulerConfig::default(), plain()).with_socket_idle_close(IDLE)).await;
    let mut socket = socket(&live).await;
    // A third of the idle time: the frame lands well before the close is
    // due, and without the restart the close would come two thirds of one
    // after it.
    tokio::time::sleep(IDLE / 3).await;
    let spoke = std::time::Instant::now();
    send(&mut socket, json!({ "type": "response.unknown" })).await;
    assert_eq!(next_event(&mut socket).await["type"], "error");
    closed(&mut socket).await;
    assert!(spoke.elapsed() >= IDLE, "closed {:?} after the client's frame", spoke.elapsed());
}

/// Zero never closes, and a live change (`PATCH /v1/config`) reaches the
/// sockets already open: the knob is read at least once a second.
#[tokio::test]
async fn zero_never_closes_and_a_live_change_reaches_an_open_socket() {
    let server = server(&Script::new(HashMap::new()), SchedulerConfig::default(), plain());
    assert_eq!(server.live().socket_idle_close, Duration::ZERO, "a server built without a config never closes");
    let handle = server.clone();
    let live = live(server).await;
    let mut socket = socket(&live).await;
    assert!(next_message(&mut socket, IDLE * 4).await.is_none(), "zero never closes");
    handle.update_live(|live| live.socket_idle_close = IDLE);
    closed(&mut socket).await;
}
