//! A contributed regression set for `noul` on "does the cited evidence
//! support the claim?" (`fixtures/entailment/`, see its `NOTICE.md`):
//! eighteen synthetic account-planning cases, one fixed question whose
//! `criteria` define "support" as entailment rather than plausibility.
//!
//! **Only `gate` cases are asserted.** A `watch` case was right when first
//! measured but within one logit of the 0.5 line, where a quantisation or
//! artifact change flips it without anything being broken; a
//! `known_failure` (case 13, date arithmetic) has the right label and the
//! model gets it wrong. Both are printed, never asserted. Every line prints
//! the logit too: `noul` probabilities move in steps of 0.125 logit (the
//! logits are bf16), so a drift reads in steps rather than in decimals.
//!
//! The tiers come from one measurement of the fixture on this file's own
//! stock artifact (2026-09-25, through a live server with `--spec-backend dflash2`
//! and `--vision-enabled`): 17/18, only 13 missed.
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

#[path = "support/entailment.rs"]
mod entailment;

use entailment::Tier;

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

async fn decide(app: &axum::Router, body: &JsonValue) -> (u16, JsonValue) {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/decide")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("a well-formed request");
    let response = app.clone().oneshot(request).await.expect("the router answers");
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.expect("a body");
    let json = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        panic!("the response is JSON: {e}\n{}", String::from_utf8_lossy(&bytes))
    });
    (status, json)
}

#[tokio::test]
#[ignore = "GPU"]
async fn the_gate_cases_keep_their_entailment_labels() {
    let Some(harness) = harness() else { return };
    let fixture = entailment::load();
    let mut wrong = Vec::new();
    let mut right = 0;
    for case in &fixture.cases {
        let (status, response) = decide(harness.app(), &fixture.request(case, Some(MODEL))).await;
        assert_eq!(status, 200, "case {}: {response}", case.id);
        let answer = &response["answers"][entailment::QUESTION_ID];
        assert_eq!(answer["type"], "noul", "case {}: {response}", case.id);
        let p = answer["noul"].as_f64().expect("a probability");
        let logit = (p / (1.0 - p)).ln();
        let ok = (p > 0.5) == case.expected;
        right += usize::from(ok);
        let verdict = match (case.tier, ok) {
            (Tier::Gate, true) => "ok",
            (Tier::Gate, false) => "WRONG",
            (Tier::Watch, true) => "ok (watch)",
            (Tier::Watch, false) => "flipped (watch)",
            (Tier::KnownFailure, false) => "known failure",
            (Tier::KnownFailure, true) => "passes now: promote to gate",
        };
        println!(
            "{:2}{} {:5} p={p:.4} logit={logit:+.3}  {verdict:28} {}",
            case.id,
            if case.starred { "*" } else { " " },
            case.expected,
            case.name
        );
        if case.tier == Tier::Gate && !ok {
            wrong.push(format!("{} ({}): p(yes)={p:.4}", case.id, case.name));
        }
    }
    println!("label agreement {right}/{}", fixture.cases.len());
    assert!(
        wrong.is_empty(),
        "a gate case lost its entailment label: {wrong:?}\n\
         The question and its criteria are fixed by the fixture, so a failure here is the \
         model's reading through this build, not the prompt."
    );
}
