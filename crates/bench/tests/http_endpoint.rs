//! The `HttpEndpoint` transport: the real HTTP client (reqwest) against the
//! in-process mock engine (`common::MockEngine` — the wire shape mimics the
//! server crate, so the transport is exercised end to end, in-process,
//! CPU-only: ADR 0006, no GPU, no model, no external engine).
//!
//! Covers the wire contract (the `POST /v1/chat/completions` streaming SSE
//! framing + the non-streaming JSON, `GET /v1/models`, OpenAI's error
//! bodies) and the per-request timing the harness measures (ttft / tok-s).

mod common;

use common::MockEngine;
use ignis_bench::client::{Endpoint, HttpEndpoint, Outcome, Request};
use ignis_bench::metrics::RequestMetrics;
use ignis_bench::trace::RequestClass;

fn request(id: &str, prompt: &str, max_tokens: u32, stream: bool) -> Request {
    Request {
        id: id.into(),
        class: RequestClass::Sub,
        prompt: prompt.into(),
        max_tokens,
        stream,
    }
}

fn metrics(out: &Outcome) -> RequestMetrics {
    RequestMetrics {
        id: "m".into(),
        class: RequestClass::Sub,
        ttft_ms: out.ttft_ms,
        n_tokens: out.n_tokens,
        total_ms: out.total_ms,
        ok: true,
    }
}

#[test]
fn http_endpoint_lists_the_loaded_model() {
    let engine = MockEngine::start();
    let ep = HttpEndpoint::new(engine.url());
    let models = ep.list_models().expect("the models list");
    assert_eq!(models, vec![engine.state.model().to_string()]);
}

#[test]
fn http_endpoint_measures_streaming_ttft_and_tokens() {
    let engine = MockEngine::start();
    let ep = HttpEndpoint::new(engine.url());
    let out = ep
        .complete(&request("s1", "hello world", 8, true))
        .expect("the streaming completion");
    // The mock generated 8 content chunks (the finish chunk's empty delta is
    // not a token; the `[DONE]` marker is not a chunk at all).
    assert_eq!(out.n_tokens, 8, "the mock generated 8 tokens");
    // The first content chunk arrives after the mock's prefill (20 ms) —
    // ttft is at least the prefill (plus the connection overhead).
    assert!(
        out.ttft_ms >= 15.0,
        "ttft {} ms should be at least the mock's prefill",
        out.ttft_ms
    );
    // The 8 content chunks are paced ~2 ms apart (the mock's decode) — a
    // measurable decode phase exists (loose bound: Windows timer
    // granularity).
    let decode_ms = out.total_ms - out.ttft_ms;
    assert!(
        decode_ms > 0.0 && decode_ms < 500.0,
        "decode {} ms should be bounded",
        decode_ms
    );
    // The per-token text flows through (concatenated deltas).
    assert!(out.output.contains("tok-0"), "the first token text");
    assert!(out.output.ends_with("tok-7"), "the last token text");
    // The per-request decode speed (tok-s) is computed from the timings.
    assert!(metrics(&out).tok_s() > 0.0, "a decode speed must be measured");
    assert_eq!(engine.state.n_completed(), 1, "the mock saw exactly one request");
}

#[test]
fn http_endpoint_non_streaming_has_no_decode_phase() {
    let engine = MockEngine::start();
    let ep = HttpEndpoint::new(engine.url());
    let out = ep
        .complete(&request("n1", "compute", 6, false))
        .expect("the non-streaming completion");
    // The token count is the usage figure (completion tokens).
    assert_eq!(out.n_tokens, 6, "the mock generated 6 tokens");
    assert!(out.output.contains("tok-5"), "the full content");
    // Non-streaming: ttft == total (no measurable decode phase) -> tok_s 0
    // (the metrics model — see `RequestMetrics::tok_s`).
    let m = metrics(&out);
    assert_eq!(m.tok_s(), 0.0, "a non-streaming request has no decode phase");
    assert_eq!(m.ttft_ms, m.total_ms, "ttft equals total when non-streaming");
}

#[test]
fn http_endpoint_surfaces_engine_errors() {
    let engine = MockEngine::start();
    engine.state.set_fail_503(true); // the server's `engine_full` 503 shape
    let ep = HttpEndpoint::new(engine.url());
    let err = ep
        .complete(&request("f", "hi", 4, false))
        .expect_err("a 503 must surface as an Err");
    let lower = err.to_lowercase();
    assert!(
        lower.contains("503") || lower.contains("engine_full"),
        "the error should carry the status or the engine's message: {err}"
    );
}

#[test]
fn http_endpoint_reports_a_connection_error() {
    // Bind + drop a listener to find a free port, then point the endpoint at
    // it: the connection must fail with a clean `Err` (the driver records a
    // failed request, not a panic).
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind a port");
    let port = listener.local_addr().expect("local addr").port();
    drop(listener); // the port is now free (nothing is listening)
    let ep = HttpEndpoint::new(format!("http://127.0.0.1:{port}"));
    assert!(
        ep.complete(&request("d", "p", 4, false)).is_err(),
        "a dead endpoint must return Err"
    );
}