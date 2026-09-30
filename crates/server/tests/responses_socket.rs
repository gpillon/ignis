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
use serde_json::{json, Value as JsonValue};
use tokio_tungstenite::tungstenite::Message;

use responses::{
    live, next_event, next_frame, send, server, socket, sse_events, terminal, Script, Socket, MODEL,
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

/// Poll `server`'s exposition until `line` is in it.
async fn metric_reaches(metrics: &ignis_server::metrics::Metrics, line: &str) {
    let found = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if metrics.render().contains(&format!("\n{line}\n")) {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await;
    assert!(found.is_ok(), "no `{line}` in:\n{}", metrics.render());
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
    let long = json!({ "model": MODEL, "input": "long", "max_output_tokens": 2000, "enable_thinking": false });
    send(&mut socket, create(&long, Some("a"))).await;
    send(&mut socket, create(&short("second on a"), Some("a"))).await;
    send(&mut socket, create(&short("first on b"), Some("b"))).await;

    // (stream, type) of every lifecycle event, in arrival order.
    let mut seen: Vec<(String, String)> = Vec::new();
    let mut terminals = 0;
    while terminals < 3 {
        let event = next_event(&mut socket).await;
        let kind = event["type"].as_str().unwrap().to_owned();
        let stream = event["stream_id"].as_str().expect("named-stream events carry stream_id").to_owned();
        if terminal(&event) {
            terminals += 1;
        }
        if kind == "response.created" || terminal(&event) {
            seen.push((stream, kind));
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
    // Concurrent across streams: b's short response ends while a's long one runs.
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

#[tokio::test]
async fn the_request_timeout_counts_from_admission_not_from_queueing() {
    let server = server(&Script::silent(), hog_config(1), plain())
        .with_request_timeout(Duration::from_millis(400));
    let live = live(server).await;
    let mut socket = socket(&live).await;
    send(&mut socket, create(&hog(), Some("h"))).await;
    next_of(&mut socket, Some("h"), "response.in_progress").await;
    send(&mut socket, create(&short("after"), Some("b"))).await;
    next_of(&mut socket, Some("b"), "response.queued").await;
    // The hog times out mid-stream, a whole timeout after the queued request
    // arrived; that request still has its own timeout from admission.
    let hog = next_of(&mut socket, Some("h"), "response.failed").await;
    assert_eq!(hog["response"]["error"]["code"], "request_timeout");
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
    let live = live(server(&Script::new(HashMap::new()), hog_config(1), plain())).await;
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
