//! The transport's own per-request deadline (GitHub #138).
//!
//! `ignis-bench g4`'s 131,072-token needle cell failed identically against
//! *both* engines, every attempt, while the 65,536-token cell passed every
//! time — the client dropped the connection ~30 s into a prefill the engine
//! was still computing (`read SSE: error decoding response body`, and
//! `finish=cancelled ... error client disconnected` on the server's side).
//! The reference's own request log puts the two cells either side of the
//! line: 65,536 tokens prefilled in 14.5 s and answered at 15.0 s, 131,072
//! tokens cancelled at 30.85 s with `gen=0`.
//!
//! The ~30 s was `reqwest::blocking::Client::new()`'s **default** total
//! request timeout — a deadline that covers reading the response body, so a
//! streamed measurement request lives entirely inside it. Nothing in this
//! crate asked for it, which is the whole problem: a measurement harness's
//! transport must not be what decides how long a request may take.
//!
//! These tests run against the in-process mock engine (CPU-only, ADR 0006).
//! The first one costs ~35 s of wall clock in the default `cargo test`, and
//! that cost is the test: a deadline at 30 s can only be shown to be gone by
//! a request that outlives 30 s. It is the longest test in the default
//! suite, deliberately.
//!
//! Every endpoint here is built with [`HttpEndpoint::with_timeout`] rather
//! than `new`, which reads `IGNIS_BENCH_REQUEST_TIMEOUT` — a test whose
//! verdict depends on the caller's environment is not a regression test.
//! The env-var semantics have their own unit tests, in `client.rs`.

mod common;

use std::time::{Duration, Instant};

use common::MockEngine;
use ignis_bench::client::{Endpoint, HttpEndpoint, Request, DEFAULT_REQUEST_TIMEOUT};
use ignis_bench::trace::RequestClass;

/// A prefill longer than the 30 s deadline `Client::new()` used to impose,
/// with margin for a loaded machine. The 128K needle cell's real prefill is
/// of this order (the reference cancelled at 30.85 s).
const LONG_PREFILL: Duration = Duration::from_secs(35);

fn needle_request(stream: bool) -> Request {
    Request {
        // The G4 needle cell's own request shape: a handful of answer
        // tokens after a very long prefill.
        id: "needle-131072".into(),
        class: RequestClass::Main,
        prompt: "…haystack… what is the secret code?".into(),
        max_tokens: 16,
        stream,
        include_usage: false,
        enable_thinking: Some(false),
    }
}

#[test]
fn a_stream_whose_first_token_takes_longer_than_thirty_seconds_is_read_to_completion() {
    let engine = MockEngine::start();
    engine.state.set_prefill(LONG_PREFILL);
    // The default deadline, named explicitly: what is under test is that
    // this value lets the request through, not what the environment says.
    let ep = HttpEndpoint::with_timeout(engine.url(), Some(DEFAULT_REQUEST_TIMEOUT));
    let start = Instant::now();
    let out = ep
        .complete(&needle_request(true))
        .expect("a slow prefill is not a transport failure");
    // The stream was read to its end: every token, and the engine's own
    // finish reason rather than a truncated read.
    assert_eq!(out.n_tokens, 16, "every generated token arrived");
    assert!(out.output.ends_with("tok-15"), "the answer is complete: {}", out.output);
    // And it really did outlive the old deadline — a mock that returned
    // early would pass the assertions above without reproducing anything.
    assert!(
        out.ttft_ms >= LONG_PREFILL.as_secs_f64() * 1000.0,
        "ttft {} ms must be the whole prefill",
        out.ttft_ms
    );
    assert!(
        start.elapsed() > Duration::from_secs(30),
        "the request must outlive reqwest's old 30 s default to prove anything"
    );
}

#[test]
fn an_explicit_deadline_still_cuts_a_request_that_outlives_it() {
    // The other half of the contract: the deadline is configured, not
    // absent. A client given a short one must still give up on a request
    // that outruns it — and say so, rather than report a bad measurement.
    let engine = MockEngine::start();
    engine.state.set_prefill(Duration::from_secs(5));
    let ep = HttpEndpoint::with_timeout(engine.url(), Some(Duration::from_millis(300)));
    let err = ep
        .complete(&needle_request(true))
        .expect_err("a request past its deadline is an error, not an outcome");
    // And it must say *whose* deadline ended the request. #138's own
    // symptom was `read SSE: error decoding response body` — a message that
    // reads like an engine fault and sent four launches chasing two engines.
    assert!(
        err.contains("client's own request deadline"),
        "the error must name the client's deadline as the cause: {err}"
    );
    assert!(
        err.contains("IGNIS_BENCH_REQUEST_TIMEOUT"),
        "and how to move it: {err}"
    );
}

#[test]
fn a_request_with_no_deadline_at_all_is_allowed() {
    // `IGNIS_BENCH_REQUEST_TIMEOUT=0` removes the deadline for a run that
    // needs to wait as long as the engine takes.
    let engine = MockEngine::start();
    let ep = HttpEndpoint::with_timeout(engine.url(), None);
    assert_eq!(ep.request_timeout(), None);
    let out = ep.complete(&needle_request(true)).expect("the completion");
    assert_eq!(out.n_tokens, 16);
}
