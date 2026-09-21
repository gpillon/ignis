//! Spec 10 acceptance 8: a `scalar` reads its evidence **without the caller
//! choosing a width**, and closes its own object (GitHub #255).
//!
//! Every truth here is stated literally in its own evidence — a count, a
//! duration, a rate — because what is under test is whether the model picks
//! the right *shape* for a number it already knows, not its arithmetic.
//! Four of them are the whole-number truths GitHub #254 walked at every
//! width; the other two are a decimal and a negative, which is the half of
//! this primitive `number` cannot express at all.
//!
//! **No question names `digits`.** That is the acceptance: the caller does
//! not know the magnitude, does not say, and gets the number anyway. A
//! fixture that passed a width would be testing `number` with extra steps.
//!
//! The system text is new and therefore unmeasured (ADR 0034), which is why
//! this file exists before the primitive ships rather than after.
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
use ignis_server::runtime::{EngineShape, cuda_scheduler};
use ignis_server::telemetry::SystemClock;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";

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

const COMPLAINT: &str = "Help! My payouts have been failing for 3 days. I've emailed twice and \
                         nobody has replied.";
const BATCH: &str = "Our batch job processed 47 invoices today and rejected 128 of them. The \
                     average run took 2.5 hours and the account balance moved by -0.75 EUR.";

/// `(evidence, name, criterion, truth)`. No width anywhere.
const CASES: &[(&str, &str, &str, f64)] = &[
    (COMPLAINT, "days", "For how many days have the payouts been failing?", 3.0),
    (COMPLAINT, "emails", "How many emails has the customer sent?", 2.0),
    (BATCH, "invoices", "How many invoices were processed?", 47.0),
    (BATCH, "rejected", "How many invoices were rejected?", 128.0),
    (BATCH, "hours", "How many hours did the average run take?", 2.5),
    (BATCH, "balance", "By how much did the account balance move, in EUR?", -0.75),
];

#[tokio::test]
#[ignore = "GPU"]
async fn a_scalar_reads_its_evidence_without_being_told_a_width() {
    let Some(harness) = harness() else { return };
    let mut wrong = Vec::new();
    for (state, name, criterion, truth) in CASES {
        let body = format!(
            r#"{{"state":"{state}","model":"{MODEL}","questions":{{"q":{{"type":"scalar","instructions":"{criterion}"}}}}}}"#
        );
        let (status, response) = decide(harness.app(), body).await;
        assert_eq!(status, 200, "{name}: {response}");
        let answer = &response["answers"]["q"];
        assert_eq!(answer["type"], "scalar", "{name}: {response}");
        let value = answer["value"].as_f64().expect("a number");
        let text = answer["text"].as_str().expect("the spelling").to_owned();
        let billed = response["usage"]["output_tokens"].as_u64().expect("a count");
        println!(
            "{name:9} truth={truth:8} -> {value:8} text={text:8} tokens={billed} unc={:.4}",
            answer["uncertainty"].as_f64().unwrap_or(f64::NAN)
        );
        if (value - truth).abs() > 1e-9 {
            wrong.push(format!("{name} read {value} ({text}) for {truth}"));
        }
        // The run is what it wrote plus the brace that closed it — the
        // saving this primitive exists for. `number` at six digits spends
        // six rounds on every one of these.
        assert_eq!(
            billed,
            text.chars().count() as u64 + 1,
            "{name}: billed {billed} for {text:?}, so it did not close its own object"
        );
    }
    assert!(
        wrong.is_empty(),
        "a scalar misread its evidence: {wrong:?}\n\
         The system text is this primitive's own and is measured by this test alone \
         (spec 10), so a failure here is the prompt and not a flake."
    );
}
