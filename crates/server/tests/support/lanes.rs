//! A one-lane engine whose releases a test can watch: what it needs to prove
//! that a request nobody reads any more — dropped, abandoned, cut by its
//! deadline — gave its lane back (GitHub #327, ADR 0058). Included via
//! `#[path]`, like the rest of `support/`.

use axum::body::Body;
use axum::http::Request;
use ignis_core::mock::GateController;
use ignis_core::{Compute, ComputeError, DecodeJob, DecodeOutcome, MockCompute, PrefillJob, PrefillOutcome, RequestId};
use tower::ServiceExt;

/// A compute decorator that exposes the public lifecycle release callback so
/// an HTTP test can wait deterministically for cancellation to free a
/// request before submitting the next one.
pub struct ReleaseObservedCompute {
    pub inner: MockCompute,
    pub released: std::sync::mpsc::SyncSender<RequestId>,
}

impl Compute for ReleaseObservedCompute {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
        self.inner.prefill_step(jobs)
    }

    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        self.inner.decode_step(jobs)
    }

    fn release(&self, request: RequestId) {
        self.inner.release(request);
        let _ = self.released.send(request);
    }
}

/// On a one-lane engine serving `model` behind `controller`'s gate, armed,
/// whose earlier request was released: a new submission is admitted, so the
/// earlier one no longer occupies the only in-flight slot. Leaves the gate
/// passed.
pub async fn the_only_lane_admits_a_new_request(app: &axum::Router, model: &str, controller: &GateController) {
    let second = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::json!({
                "model": model,
                "messages": [{ "role": "user", "content": "replacement" }],
                // Two tokens: the first leaves with the prefill (#518), the
                // second takes the decode step the gate holds.
                "max_tokens": 2,
                "stream": true
            })
            .to_string(),
        ))
        .unwrap();
    let second_response = app.clone().oneshot(second).await.unwrap();
    assert_eq!(second_response.status(), 200, "the only lane admits the next request");
    // The body lives until the gate is passed: dropping it cancels the
    // request, and a cancel the model thread reads before the request's
    // first advance releases it with no decode step at all — the gate is
    // never entered and `wait_entered` waits forever. A Windows CI run hung
    // four hours on exactly that.
    let second_body = second_response.into_body();
    controller.wait_entered();
    controller.release();
    drop(second_body);
}
