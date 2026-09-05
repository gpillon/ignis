//! The capture proxy (`ignis-bench record`, spec 03) against a mock
//! target: a live session is recorded as a bench trace while the proxy
//! forwards every request to the in-process mock engine
//! (`common::MockEngine` — CPU-only, ADR 0006: no GPU, no model, no
//! external engine). The proxy's own HTTP surface is driven through the
//! real `HttpEndpoint` transport (the same one `replay` uses), so the
//! record + forward + pipe path is exercised end to end.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use common::MockEngine;
use ignis_bench::client::{replay, Endpoint, HttpEndpoint, MockEndpoint, ReplayConfig, Request};
use ignis_bench::record::{ClassPolicy, RecordConfig, RecordServer, SessionSummary};
use ignis_bench::trace::{RequestClass, Trace};
use tokio::runtime::Runtime;

/// A unique temp JSONL path (the in-process tests share one pid).
static NEXT: AtomicU64 = AtomicU64::new(0);

fn unique_out() -> std::path::PathBuf {
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir()
        .join(format!("ignis-bench-record-it-{pid}-{n}.jsonl", pid = std::process::id()))
}

/// A client request (the OpenAI chat-completions body the proxy records).
fn chat_request(id: &str, prompt: &str, max_tokens: u32, stream: bool) -> Request {
    Request {
        id: id.into(),
        class: RequestClass::Main, // the proxy decides the recorded class;
        // the client's `id` is only the replay's identifier
        prompt: prompt.into(),
        max_tokens,
        stream,
    }
}

#[test]
fn the_capture_proxy_records_a_live_session_against_a_mock_target() {
    let engine = MockEngine::start();
    let out = unique_out();
    let server = RecordServer::new(RecordConfig {
        listen: "127.0.0.1:0".into(), // ephemeral port (resolved by `bind`)
        target: engine.url().into(),
        out: out.clone(),
        class_policy: ClassPolicy::FirstIsMain,
    })
    .expect("the proxy builds");
    let rt = Runtime::new().expect("the runtime");

    // The proxy listens on an ephemeral port and serves until the session
    // ends (`POST /v1/session/end`).
    let (proxy_url, listener) = rt.block_on(server.bind()).expect("the proxy binds");
    let serve = rt.spawn(server.serve(listener));

    // Drive the proxy through the real transport: a streaming "main"
    // request, then two non-streaming sub requests. The proxy records each
    // request and pipes the mock engine's response back (SSE for the
    // streaming one).
    let ep = HttpEndpoint::new(&proxy_url);
    let o1 = ep
        .complete(&chat_request("main", "Hello from the main agent", 32, true))
        .expect("the proxy forwards the streaming request");
    assert!(
        !o1.output.is_empty(),
        "the SSE response is piped back through the proxy"
    );
    std::thread::sleep(Duration::from_millis(50)); // a visible arrival offset
    let o2 = ep
        .complete(&chat_request("sub-1", "Sub one", 16, false))
        .expect("the proxy forwards the first sub request");
    assert!(!o2.output.is_empty(), "the JSON response is piped back");
    ep.complete(&chat_request("sub-2", "Sub two", 16, false))
        .expect("the proxy forwards the second sub request");

    // End the session: the proxy finalizes the trace and stops.
    let end = reqwest::blocking::Client::new()
        .post(format!("{proxy_url}/v1/session/end"))
        .send()
        .expect("session end is accepted");
    assert!(end.status().is_success(), "session end is a success");
    let summary: SessionSummary = rt
        .block_on(serve)
        .expect("the proxy stops")
        .expect("the session ends");
    assert_eq!(summary.requests, 3);
    assert_eq!(summary.main, 1);
    assert_eq!(summary.sub, 2);
    assert_eq!(summary.class_policy, "first-is-main");
    assert_eq!(summary.file, Some(out.clone()));

    // The mock target saw all three requests (the proxy forwarded them,
    // byte-for-byte).
    assert_eq!(engine.state.n_completed(), 3);

    // The recorded trace: the `TraceLine` shape (the same shape `replay`
    // consumes) — the line shape + the arrival offsets (the spec's
    // acceptance: a unit test against a mock target checks both).
    let text = std::fs::read_to_string(&out).expect("the trace file is written");
    let trace = Trace::from_jsonl(&text).expect("the recorded trace parses");
    let lines = trace.lines();
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[0].class, RequestClass::Main, "the first request is main");
    assert_eq!(lines[1].class, RequestClass::Sub);
    assert_eq!(lines[2].class, RequestClass::Sub);
    assert_eq!(lines[0].t_arrive_ms, 0, "the session's first request is t0");
    assert!(
        lines[1].t_arrive_ms > 0,
        "the sleep between arrivals is visible in the offset"
    );
    assert!(
        lines[2].t_arrive_ms >= lines[1].t_arrive_ms,
        "the offsets are non-decreasing"
    );
    // The actual request content (the engine sees the same input).
    assert!(lines[0].prompt.contains("Hello from the main agent"));
    assert!(lines[1].prompt.contains("Sub one"));
    assert_eq!(lines[0].max_tokens, 32);
    assert!(lines[0].stream);
    assert_eq!(lines[1].max_tokens, 16);
    assert!(!lines[1].stream);

    // The recorded trace loads and replays (the `replay` seam the CLI's
    // `replay` subcommand uses — the trace is valid for the harness).
    let ep = Arc::new(MockEndpoint::deterministic());
    let metrics = replay(ep, &trace, &ReplayConfig { max_concurrency: 4, time_scale: 0.0 });
    assert_eq!(metrics.len(), 3);
    assert!(metrics.iter().all(|m| m.ok), "every recorded request replays");

    let _ = std::fs::remove_file(&out);
}

#[test]
fn the_capture_proxy_survives_a_target_outage() {
    // The target is unreachable: the proxy records the request (the load
    // still happened) and returns a 502 (the client sees the failure, not
    // a hang). The trace stays valid.
    let out = unique_out();
    // A port nothing listens on: the target is dead from the start.
    let server = RecordServer::new(RecordConfig {
        listen: "127.0.0.1:0".into(),
        target: "http://127.0.0.1:9".into(),
        out: out.clone(),
        class_policy: ClassPolicy::FirstIsMain,
    })
    .expect("the proxy builds");
    let rt = Runtime::new().expect("the runtime");
    let (proxy_url, listener) = rt.block_on(server.bind()).expect("the proxy binds");
    let serve = rt.spawn(server.serve(listener));

    let ep = HttpEndpoint::new(&proxy_url);
    let err = ep
        .complete(&chat_request("main", "Hello", 32, true))
        .expect_err("the proxy reports the target failure");
    assert!(
        err.contains("502"),
        "the failure surfaces as a 502 (got: {err})"
    );

    // End the session: the recorded line (the request was made, even
    // though the target refused it) is in the trace.
    reqwest::blocking::Client::new()
        .post(format!("{proxy_url}/v1/session/end"))
        .send()
        .expect("session end is accepted");
    let summary: SessionSummary = rt
        .block_on(serve)
        .expect("the proxy stops")
        .expect("the session ends");
    assert_eq!(summary.requests, 1);
    assert_eq!(summary.main, 1);
    let text = std::fs::read_to_string(&out).expect("the trace file is written");
    let trace = Trace::from_jsonl(&text).expect("the trace parses");
    assert_eq!(trace.len(), 1);
    let _ = std::fs::remove_file(&out);
}