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
//! **The negative is a watch line**, printed and never asserted (GitHub
//! #675): the sign of the balance's −0.75 is a coin on the 27B in both
//! exception weights (GitHub #667: BF16 read it right on 2 of 9 routes and
//! wrong at its served shape), so asserting it reported the route, not a
//! fault. The minus
//! `scalar` drops on literal negatives is GitHub #671, and the sign's absence
//! from `uncertainty` GitHub #670. Every other case is asserted, and every
//! case's run must close its own object.
//!
//! **No question names `digits`.** That is the acceptance: the caller does
//! not know the magnitude, does not say, and gets the number anyway. A
//! fixture that passed a width would be testing `number` with extra steps.
//!
//! The system text is new and therefore unmeasured (ADR 0034), which is why
//! this file exists before the primitive ships rather than after.
//!
//! It runs on the arms the entailment set registers (`support/decide_arms.rs`,
//! GitHub #675), each naming its exception weights, context and rope and
//! reading no config file: `IGNIS_DECIDE_ARMS` picks them, one load each;
//! unset, the harness's arm under each exception weights (`h-bf16`, `h-fp8`).
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

use decide_arms::MODEL;

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

/// Whether a case's read is asserted or printed.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tier {
    Gate,
    Watch,
}

/// `(evidence, name, criterion, truth, tier)`. No width anywhere.
const CASES: &[(&str, &str, &str, f64, Tier)] = &[
    (COMPLAINT, "days", "For how many days have the payouts been failing?", 3.0, Tier::Gate),
    (COMPLAINT, "emails", "How many emails has the customer sent?", 2.0, Tier::Gate),
    (BATCH, "invoices", "How many invoices were processed?", 47.0, Tier::Gate),
    (BATCH, "rejected", "How many invoices were rejected?", 128.0, Tier::Gate),
    (BATCH, "hours", "How many hours did the average run take?", 2.5, Tier::Gate),
    (BATCH, "balance", "By how much did the account balance move, in EUR?", -0.75, Tier::Watch),
];

#[tokio::test]
#[ignore = "GPU"]
async fn a_scalar_reads_its_evidence_without_being_told_a_width() {
    let arms = decide_arms::registered();
    let mut wrong = Vec::new();
    for arm in decide_arms::selected(&arms, decide_arms::DEFAULT_ARMS) {
        let Some(loaded) = arm.load(None) else { return };
        for (state, name, criterion, truth, tier) in CASES {
            let body = format!(
                r#"{{"state":"{state}","model":"{MODEL}","questions":{{"q":{{"type":"scalar","instructions":"{criterion}"}}}}}}"#
            );
            let (status, response) = decide(loaded.app(), body).await;
            assert_eq!(status, 200, "{name}: {response}");
            let answer = &response["answers"]["q"];
            assert_eq!(answer["type"], "scalar", "{name}: {response}");
            let value = answer["value"].as_f64().expect("a number");
            let text = answer["text"].as_str().expect("the spelling").to_owned();
            let billed = response["usage"]["output_tokens"].as_u64().expect("a count");
            let read_right = (value - truth).abs() <= 1e-9;
            let verdict = match (tier, read_right) {
                (_, true) => "ok",
                (Tier::Gate, false) => "WRONG",
                (Tier::Watch, false) => "misread (watch)",
            };
            println!(
                "{name:9} truth={truth:8} -> {value:8} text={text:8} tokens={billed} unc={:.4}  {verdict} on {}",
                answer["uncertainty"].as_f64().unwrap_or(f64::NAN),
                arm.name
            );
            if *tier == Tier::Gate && !read_right {
                wrong.push(format!("{name} read {value} ({text}) for {truth} on {}", arm.name));
            }
            // The run is what it wrote plus the brace that closed it — the
            // saving this primitive exists for. `number` at six digits spends
            // six rounds on every one of these.
            assert_eq!(
                billed,
                text.chars().count() as u64 + 1,
                "{name} on {}: billed {billed} for {text:?}, so it did not close its own object",
                arm.name
            );
        }
    }
    assert!(
        wrong.is_empty(),
        "a scalar misread its evidence: {wrong:?}\n\
         The system text is this primitive's own and is measured by this test alone \
         (spec 10), so a failure here is the prompt and not a flake."
    );
}
