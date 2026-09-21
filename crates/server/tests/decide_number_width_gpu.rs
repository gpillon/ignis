//! Spec 09 acceptance 3: every width that can hold the value reads it
//! exactly (GitHub #254).
//!
//! The fix for #254 is a **clause in a prompt** and nothing else — there is
//! no padding code, only an instruction that was missing. `numbers.rs` pins
//! the clause's presence at every width, which catches a reword; only the
//! card catches the thing that actually matters, which is whether the model
//! obeys it.
//!
//! What this walks is the failure as it was reported: `digits: 3` over
//! evidence saying "failing for 3 days" answering **300**, with the first
//! digit at p = 0.998 so that nothing in the trace flagged it. Before the
//! clause, 11 of these 21 cells read their truth; after it, all 21 do
//! (`docs/findings/2026-09-21-the-number-prompt-declares-an-alignment.md`).
//!
//! Widths **narrower** than the truth are asked and printed but not
//! asserted: a three-digit value in a two-digit field can only be
//! truncated, and that is not a fault to fix.
//!
//! Every truth is stated literally in its own evidence — none of these asks
//! the model to compute — because what is under test is the *rendering* of a
//! number and not the model's arithmetic, which this change does not touch.
//!
//! Four questions at six widths is twenty-four prefills of a couple of
//! hundred tokens with one to six decoded tokens each: a few seconds of card
//! time.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or an unavailable GPU is a **skip**; under the profile
//! the same condition is a **hard failure**.

#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, to_bytes};
use axum::http::Request;
use serde_json::Value as JsonValue;
use tower::ServiceExt;

use ignis_artifact::{FrontendSet, Reader};
use ignis_core::gpu_profile;
use ignis_server::Server;
use ignis_server::engine::Engine;
use ignis_server::numbers::DIGITS;
use ignis_server::runtime::{EngineShape, cuda_scheduler};
use ignis_server::telemetry::SystemClock;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";

/// A live server over the real GPU-backed scheduler, text only.
///
/// The model thread owns the GPU-resident state and frees it only when it
/// exits, so the router is dropped before the thread is joined — the shape
/// `decide_point_gpu.rs` uses, and for the same reason.
struct Harness {
    app: Option<axum::Router>,
    driver: Option<std::thread::JoinHandle<()>>,
}

impl Harness {
    fn app(&self) -> &axum::Router {
        self.app.as_ref().expect("the app is only taken by Drop")
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        drop(self.app.take());
        if let Some(driver) = self.driver.take() {
            let _ = driver.join();
        }
    }
}

fn harness() -> Option<Harness> {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return None;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend
        .eos_token_id()
        .unwrap_or_else(|| panic!("qwen3.8-27b generation config must carry eos_token_id"));
    let scheduler = match cuda_scheduler(path, MODEL.into(), eos, EngineShape::default()) {
        Ok((scheduler, _reservations)) => scheduler,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("cuda_scheduler: {e}")) {
                return None;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let (engine, driver) = Engine::with_clock_and_driver(Box::new(scheduler), Arc::new(SystemClock));
    let provider = ignis_server::artifact_template::ArtifactTemplateProvider::new(frontend);
    let server =
        Server::new(engine, Box::new(provider)).with_request_timeout(Duration::from_secs(180));
    Some(Harness { app: Some(server.app()), driver: Some(driver) })
}

async fn decide(app: &axum::Router, body: String) -> (u16, JsonValue) {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/decide")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .expect("a well-formed request");
    let response = app.clone().oneshot(request).await.expect("the router answers");
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.expect("a body");
    let json = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        panic!("the response is JSON: {e}\n{}", String::from_utf8_lossy(&bytes))
    });
    (status, json)
}

/// The evidence that produced the report, and one whose truths are two and
/// three digits wide so that "the width that fits" is not a constant.
const COMPLAINT: &str = "Help! My payouts have been failing for 3 days. I've emailed twice and \
                         nobody has replied. If this isn't fixed by Friday I'm moving to another \
                         provider.";
const BATCH: &str = "Our batch job processed 47 invoices today and rejected 128 of them. It has \
                     been running for 6 days.";

const CASES: &[(&str, &str, &str, u64)] = &[
    (COMPLAINT, "days", "For how many days have the payouts been failing?", 3),
    (COMPLAINT, "emails", "How many emails has the customer sent?", 2),
    (BATCH, "invoices", "How many invoices were processed?", 47),
    (BATCH, "rejected", "How many invoices were rejected?", 128),
];

#[tokio::test]
#[ignore = "GPU"]
async fn every_width_that_holds_the_value_reads_it() {
    let Some(harness) = harness() else { return };
    let mut wrong = Vec::new();
    for (state, name, criterion, truth) in CASES {
        let own_width = truth.to_string().len() as u32;
        for digits in DIGITS {
            let body = format!(
                r#"{{"state":"{state}","model":"{MODEL}","questions":{{"q":{{"type":"number","instructions":"{criterion}","digits":{digits}}}}}}}"#
            );
            let (status, response) = decide(harness.app(), body).await;
            assert_eq!(status, 200, "{name}@{digits}: {response}");
            let answer = &response["answers"]["q"];
            assert_eq!(answer["type"], "number", "{name}@{digits}: {response}");
            let number = answer["number"].as_u64().expect("a whole number");
            let first = answer["digits"][0]["probability"].as_f64().expect("a probability");
            println!(
                "{name:9} truth={truth:4} width={digits} -> {number:7}{} first={first:.3}",
                if number == *truth { " *" } else { "  " }
            );
            // A field narrower than the value can only truncate, which is
            // the one thing this change does not claim to fix.
            if digits >= own_width && number != *truth {
                wrong.push(format!("{name} read {number} for {truth} at {digits} digits"));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "a field that can hold the value did not read it: {wrong:?}\n\
         The clause in `number_system` is the whole fix for this — there is no padding code — so \
         a failure here means the model stopped obeying it, which is \
         docs/findings/2026-09-21-the-number-prompt-declares-an-alignment.md moving."
    );
}
