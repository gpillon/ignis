//! Spec 22 acceptance 5 (GitHub #278, ADR 0042): **end to end on the card,
//! the defaults name every present target and flag every absent question**,
//! through `/v1/decide` on the served artifact — a log of near-duplicate
//! lines (folded), a record array and titled prose paragraphs, one absent
//! question each — and a record array **past one window** is answered with
//! an index into the array as sent.
//!
//! The states are synthetic: `fixtures/locate/shortlist.json` holds the
//! prose and the questions, and the log and the record arrays are generated
//! here, deterministically. They are easy on purpose — this holds the route
//! on the card (the heads' rows over real windows, the fold's two levels,
//! the `choice`s and `found`), not its accuracy, which is the acceptance
//! run's (spec 22 § The acceptance run).
//!
//! The long array is ~6,000 records, past the 200,000-key window: two
//! windows, each prefilled once with its baseline (about a minute each), the
//! question claiming what each baseline kept.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38):
//! `cargo test -p ignis-server --features cuda --test locate_shortlist_gpu -- --ignored --nocapture`.

#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::{Body, to_bytes};
use axum::http::Request;
use serde_json::{Value as JsonValue, json};
use tower::ServiceExt;

use ignis_artifact::{FrontendSet, Reader};
use ignis_core::gpu_profile;
use ignis_server::Server;
use ignis_server::engine::Engine;
use ignis_server::runtime::{EngineShape, cuda_scheduler};
use ignis_server::telemetry::SystemClock;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";

/// A live server over the real GPU-backed scheduler, text only, at the
/// context `make start` serves; the router is dropped before the model
/// thread is joined, so the next load never overlaps this one.
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
    let eos = frontend.eos_token_id().expect("qwen3.8-27b generation config must carry eos_token_id");
    let shape = EngineShape { max_context: 262_144, ..EngineShape::default() };
    let scheduler = match cuda_scheduler(path, MODEL.into(), eos, shape) {
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
    let server = Server::new(engine, Box::new(provider)).with_request_timeout(Duration::from_secs(900));
    assert!(server.locate.is_some(), "the served artifact is calibrated for locate");
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
    (status, serde_json::from_slice(&bytes).unwrap_or(JsonValue::Null))
}

/// A payments service's log: one kind of line many times over with its
/// users and amounts, two other kinds, and one declined payment of user
/// u42 among u42's accepted ones — its near-duplicates.
fn log() -> Vec<String> {
    let mut lines = Vec::new();
    for i in 0..150u32 {
        let (m, s) = (i / 60, i % 60);
        let user = [7, 42, 13, 58, 42, 21, 90, 3][(i % 8) as usize];
        lines.push(match i {
            97 => format!(
                "2026-09-28T09:{m:02}:{s:02}Z WARN payments request=r{} user=u42 amount=18.90 status=402 reason=card_declined",
                4000 + i
            ),
            _ if i % 11 == 0 => format!("2026-09-28T09:{m:02}:{s:02}Z INFO auth login ok user=u{user} session=s{}", 900 + i),
            _ if i % 7 == 0 => format!("2026-09-28T09:{m:02}:{s:02}Z INFO ledger posted entry=e{} account=a{user}", 300 + i),
            _ => format!(
                "2026-09-28T09:{m:02}:{s:02}Z INFO payments request=r{} user=u{user} amount={}.{:02} status=200",
                4000 + i,
                10 + (i * 7) % 90,
                (i * 13) % 100
            ),
        });
    }
    lines
}

/// `n` employees, one of them in Reykjavik at `target`; no one in
/// Ulaanbaatar.
fn records(n: usize, target: usize) -> JsonValue {
    let cities = ["Lisbon", "Porto", "Madrid", "Valencia", "Lyon", "Turin", "Genoa", "Ghent", "Leeds", "Bergen", "Krakow", "Brno"];
    let teams = ["platform", "billing", "search", "mobile", "data", "support"];
    let names = ["Ada", "Bruno", "Chiara", "Dmitri", "Elena", "Farid", "Greta", "Hugo", "Ines", "Jonas"];
    JsonValue::Array(
        (0..n)
            .map(|i| {
                let city = if i == target { "Reykjavik" } else { cities[(i * 5 + i / 12) % cities.len()] };
                json!({
                    "id": 1000 + i,
                    "name": format!("{} {}", names[i % names.len()], ["Moreau", "Silva", "Rossi", "Novak", "Berg"][(i / 10) % 5]),
                    "team": teams[(i * 3) % teams.len()],
                    "city": city,
                    "since": 2010 + (i % 14),
                })
            })
            .collect(),
    )
}

/// The index of the one segment `needle` appears in.
fn target_of(segments: &[String], needle: &str) -> usize {
    let found: Vec<usize> = segments.iter().enumerate().filter(|(_, s)| s.contains(needle)).map(|(i, _)| i).collect();
    assert_eq!(found.len(), 1, "{needle:?} names exactly one segment: {found:?}");
    found[0]
}

/// A present question is right when its answer names the target and, where
/// the route carries `found`, keeps it at 0.5 or more; an absent one is
/// flagged when `found` is below 0.5 (spec 22 § Scoring).
fn judge(id: &str, answer: &JsonValue, target: Option<usize>) {
    assert_eq!(answer["type"], "locate", "{id}: {answer}");
    assert_eq!(answer["method"], "shortlist", "{id}: {answer}");
    let found = answer["found"].as_f64().unwrap_or_else(|| panic!("{id}: a default route carries found: {answer}"));
    match target {
        Some(target) => {
            assert!(found >= 0.5, "{id}: a present answer kept: {answer}");
            assert_eq!(answer["segment"], target, "{id}: {answer}");
        }
        None => {
            assert!(found < 0.5, "{id}: an absent question flagged: {answer}");
            assert_eq!(answer["segment"], JsonValue::Null, "{id}: {answer}");
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "GPU profile: loads the served artifact"]
async fn the_defaults_name_every_target_and_flag_every_absent_question() {
    let Some(harness) = harness() else { return };
    let fixture: JsonValue =
        serde_json::from_str(include_str!("fixtures/locate/shortlist.json")).expect("the fixture parses");
    let prose: Vec<String> =
        fixture["states"]["prose"].as_array().expect("prose").iter().map(|l| l.as_str().unwrap().to_owned()).collect();
    let log = log();
    let staff = records(80, 53);
    let staff_texts: Vec<String> = staff.as_array().unwrap().iter().map(|r| r.to_string()).collect();
    let states = [
        ("log", json!(log.join("\n")), log.clone(), "log", "template_fold"),
        ("records", staff.clone(), staff_texts, "records", "none"),
        ("prose", json!(prose.join("\n")), prose.clone(), "prose", "none"),
    ];
    for (name, state, segments, kind, compression) in states {
        let questions: Vec<&JsonValue> =
            fixture["questions"].as_array().unwrap().iter().filter(|q| q["state"] == name).collect();
        let asked: serde_json::Map<String, JsonValue> = questions
            .iter()
            .map(|q| (q["id"].as_str().unwrap().to_owned(), json!({"type": "locate", "instructions": q["instruction"]})))
            .collect();
        let started = Instant::now();
        let (status, response) = decide(harness.app(), json!({"state": state, "questions": asked}).to_string()).await;
        assert_eq!(status, 200, "{name}: {response}");
        println!("{name}: {:.1} s, {}", started.elapsed().as_secs_f64(), response["usage"]);
        for question in questions {
            let id = question["id"].as_str().unwrap();
            let answer = &response["answers"][id];
            println!("  {id}: {answer}");
            assert_eq!((answer["kind"].as_str(), answer["compression"].as_str()), (Some(kind), Some(compression)), "{id}: {answer}");
            let target = question["target"].as_str().map(|needle| target_of(&segments, needle));
            judge(id, answer, target);
        }
        assert_eq!(response["usage"]["output_tokens"], 0, "{name}: nothing generated");
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "GPU profile: loads the served artifact, reads ~250K tokens in two windows"]
async fn a_record_array_past_one_window_is_answered_in_the_original() {
    let Some(harness) = harness() else { return };
    // The target in the second window, and a second question over the same
    // array: its windows are read once, the question claiming each.
    let staff = records(6_000, 4_321);
    let body = json!({
        "state": staff,
        "questions": {
            "present": {"type": "locate", "instructions": "Which employee works from Reykjavik?"},
            "absent": {"type": "locate", "instructions": "Which employee works from Ulaanbaatar?"}
        }
    });
    let started = Instant::now();
    let (status, response) = decide(harness.app(), body.to_string()).await;
    assert_eq!(status, 200, "{response}");
    println!("6,000 records: {:.1} s, {}", started.elapsed().as_secs_f64(), response["usage"]);
    println!("  present: {}", response["answers"]["present"]);
    println!("  absent: {}", response["answers"]["absent"]);
    let input = response["usage"]["input_tokens"].as_u64().expect("usage");
    assert!(input > 400_000, "two windows and their baselines, each past 100K tokens: {input}");
    judge("present", &response["answers"]["present"], Some(4_321));
    assert_eq!(response["answers"]["present"]["value"]["city"], "Reykjavik");
    judge("absent", &response["answers"]["absent"], None);
}
