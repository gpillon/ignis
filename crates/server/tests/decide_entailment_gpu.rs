//! A regression set for `noul` on "does the cited evidence support the
//! claim?" (`fixtures/entailment/`, see its `NOTICE.md`): twenty synthetic
//! account-planning cases, eighteen contributed and two ignis wrote, one
//! fixed question whose `criteria` define "support" as entailment rather
//! than plausibility.
//!
//! **Only `gate` cases are asserted**, on every arm the run loads; `watch`
//! and `known_failure` cases are printed, never asserted. Which case is which
//! tier follows from its logits on every arm measured, by the fixture's one
//! rule (`measurement.rule`, applied by `entailment::tier_by_rule`), so a
//! watch case flipping is the numerics moving, not a fault. Every line prints
//! the logit too: `noul` probabilities move in steps of 0.125 logit (the
//! logits are bf16), so a drift reads in steps rather than in decimals.
//!
//! The tiers come from one measurement of the set's arms, recorded in the
//! fixture's `measurement` with its date, build, card, the command that
//! redoes each arm and every case's logit in every arm (`NOTICE.md` tells
//! the set's provenance). An arm names its exception weights, context, rope,
//! GDN prefill route, prompt attention and prompt reuse, and reads no config
//! file (`support/decide_arms.rs`, GitHub #675). `IGNIS_DECIDE_ARMS` picks
//! the arms, one load each: a comma list of names, or `all`; unset, the
//! harness's arm under each exception weights (`h-bf16`, `h-fp8`). Each arm
//! ends with a `logits <arm> {...}` line, its cases' logits as the fixture
//! records them; with `IGNIS_DECIDE_DRAFTS=1` the fixture's drafts are
//! posted after the cases and end with a `drafts <arm> {...}` line. That is
//! how an arm is measured and its record redone. GitHub #628: the same gates
//! hold on a load that drafts with the 27B's MTP head, whose prefill also
//! writes the head's state; the speculation-off load is the set's own.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or an unavailable GPU is a **skip**; under the profile
//! the same condition is a **hard failure**.

#![cfg(feature = "cuda")]

use axum::body::{Body, to_bytes};
use axum::http::Request;
use serde_json::Value as JsonValue;
use tower::ServiceExt;

#[path = "support/decide_arms.rs"]
mod decide_arms;
#[path = "support/entailment.rs"]
mod entailment;

use decide_arms::gpu::Loaded;
use decide_arms::{Arm, MODEL};
use entailment::{Fixture, Tier};

/// `IGNIS_DECIDE_DRAFTS=1`: an arm also reads the fixture's drafts.
const DRAFTS_ENV: &str = "IGNIS_DECIDE_DRAFTS";

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

/// p(yes) of `state`, and its logit to the fixture's three decimals.
async fn p_yes(app: &axum::Router, fixture: &Fixture, state: &str, what: &str) -> (f64, f64) {
    let (status, response) = decide(app, &fixture.request_for(state, Some(MODEL))).await;
    assert_eq!(status, 200, "{what}: {response}");
    let answer = &response["answers"][entailment::QUESTION_ID];
    assert_eq!(answer["type"], "noul", "{what}: {response}");
    let p = answer["noul"].as_f64().expect("a probability");
    let logit = (p / (1.0 - p)).ln();
    (p, (logit * 1000.0).round() / 1000.0)
}

#[tokio::test]
#[ignore = "GPU"]
async fn the_gate_cases_keep_their_entailment_labels() {
    gate_cases_keep_their_labels(None).await;
}

/// GitHub #628: MTP on against off, the regression set's quality leg.
#[tokio::test]
#[ignore = "GPU"]
async fn the_gate_cases_keep_their_entailment_labels_on_an_mtp_load() {
    let speculation = ignis_core::Speculation::new(ignis_core::SpeculativeBackend::Mtp, 7).expect("window 7");
    gate_cases_keep_their_labels(Some(speculation)).await;
}

async fn gate_cases_keep_their_labels(speculation: Option<ignis_core::Speculation>) {
    let fixture = entailment::load_recorded();
    let drafts = std::env::var(DRAFTS_ENV).is_ok_and(|value| value == "1");
    let mut wrong = Vec::new();
    for arm in decide_arms::selected(&fixture.measurement.arms, decide_arms::DEFAULT_ARMS) {
        let Some(loaded) = arm.load(speculation) else { return };
        // GitHub #628: a load that drafts binds a verify round, and the
        // head no drafter round (DFlash2's); the plan's lines say which
        // load the cases ran on.
        if speculation.is_some() {
            let lines = &loaded.reservations.as_ref().expect("the load's plan").lines;
            assert!(
                lines.verify_round > 0 && lines.drafter_round == 0,
                "an MTP load plans a verify round and no drafter round: {lines:?}"
            );
        }
        wrong.extend(arm_keeps_its_gates(&fixture, arm, &loaded, drafts).await);
    }
    assert!(
        wrong.is_empty(),
        "a gate case lost its entailment label: {wrong:?}\n\
         The question and its criteria are fixed by the fixture, so a failure here is the \
         model's reading through this build, not the prompt."
    );
}

/// Every case read on `arm`'s load, in the file's order, then its drafts if
/// asked: the gate cases that lost their label.
async fn arm_keeps_its_gates(fixture: &Fixture, arm: &Arm, loaded: &Loaded, drafts: bool) -> Vec<String> {
    let mut wrong = Vec::new();
    let mut right = 0;
    let mut logits = serde_json::Map::new();
    for case in &fixture.cases {
        let (p, logit) = p_yes(loaded.app(), fixture, &case.state, &format!("case {}", case.id)).await;
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
        logits.insert(case.id.to_string(), logit.into());
        if case.tier == Tier::Gate && !ok {
            wrong.push(format!("{} on {} ({}): p(yes)={p:.4}", case.id, arm.name, case.name));
        }
    }
    println!("label agreement {right}/{} on {}", fixture.cases.len(), arm.name);
    println!("logits {} {}", arm.name, JsonValue::Object(logits));
    if drafts {
        let mut read = serde_json::Map::new();
        for draft in &fixture.drafts {
            let (p, logit) = p_yes(loaded.app(), fixture, &draft.state, &format!("draft {}", draft.name)).await;
            println!("draft {:24} p={p:.4} logit={logit:+.3}", draft.name);
            read.insert(draft.name.clone(), logit.into());
        }
        println!("drafts {} {}", arm.name, JsonValue::Object(read));
    }
    wrong
}
