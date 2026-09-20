//! Spec 06 acceptance 3: a `point` over the committed pointing fixture
//! lands inside the target button on all three scenes — **end to end
//! through the server** (GitHub #242, ADR 0034).
//!
//! `classify_pointing_gpu.rs` established the mechanism by driving the
//! kernel directly: it prefilled a prompt, picked each digit host-side from
//! a full logits row, and forced the winner back in. That is not what
//! ships. This asks the same question of the whole stack — HTTP, the chat
//! template, the media path, the scheduler's schedule of permitted sets,
//! and the leaf masking its own logits — and it is the only test that
//! exercises the permitted set on the card *with a decode lane and a
//! sequence pool under it*.
//!
//! The oracle is the fixture's own manifest: each scene declares the blue
//! button's box in pixels of a 4096x4096 image, and the answer is in pixels
//! of the submitted image because the server rescales it. The measured
//! error was 84 px worst-case on a button 1100 px wide
//! (`docs/findings/2026-09-19-constrained-digit-readout-points.md`), so
//! "inside the button" is a real assertion here and not a formality.
//!
//! **The speculative case is deliberately covered too.** A drafter load runs
//! *every* round as a verify round, and the leaf refuses a constrained lane
//! there — so `CudaLeaf::decode` routes a batch with any constrained lane
//! through the plain path. That routing is a claim about the card that only
//! a drafter load can test, and a run without one would leave it an
//! untested comment.
//!
//! At the default vision budget a 4096x4096 image is not downscaled, so each
//! question is a ~16K-token prefill of roughly two seconds. Three scenes at
//! two shapes is six of them.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38).

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, to_bytes};
use axum::http::Request;
use serde::Deserialize;
use serde_json::Value as JsonValue;
use tower::ServiceExt;

use ignis_artifact::{FrontendSet, Reader};
use ignis_core::gpu_profile;
use ignis_server::Server;
use ignis_server::engine::Engine;
use ignis_server::media::{MediaAcquirer, MediaPolicy};
use ignis_server::runtime::{EngineShape, cuda_scheduler};
use ignis_server::telemetry::SystemClock;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";
/// The instruction the finding measured, unchanged.
const INSTRUCTION: &str = "click the blue button";

#[derive(Deserialize)]
struct Manifest {
    side: u32,
    scenes: Vec<Scene>,
}

#[derive(Deserialize)]
struct Scene {
    id: String,
    image: String,
    blue_box: [i64; 4],
    blue_centre: [i64; 2],
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("pointing")
}

/// A live server over the real GPU-backed scheduler with vision bound and
/// the media path wired, exactly as `main.rs` wires one under `--vision`.
///
/// The model thread owns the GPU-resident state and frees it only when it
/// exits, so the router is dropped before the thread is joined — the same
/// shape `openai_http_gpu.rs` uses, and for the same reason: the next load
/// must not allocate before this one has let go.
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

fn harness(shape: EngineShape) -> Option<Harness> {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return None;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend
        .eos_token_id()
        .unwrap_or_else(|| panic!("qwen3.8-27b generation config must carry eos_token_id"));
    let processor = frontend
        .vision_processor()
        .unwrap_or_else(|e| panic!("vision processor: {e}"));
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
    let acquirer = MediaAcquirer::new(
        Arc::new(processor.clone()),
        processor.options().clone(),
        MediaPolicy::new(false, 1 << 26),
    );
    let provider =
        ignis_server::artifact_template::ArtifactTemplateProvider::new(frontend).with_vision(processor);
    let server = Server::new(engine, Box::new(provider))
        .with_media(Arc::new(acquirer))
        .with_request_timeout(Duration::from_secs(180));
    Some(Harness { app: Some(server.app()), driver: Some(driver) })
}

fn vision_shape() -> EngineShape {
    EngineShape {
        vision: Some(ignis_core::Vision::default()),
        ..EngineShape::default()
    }
}

/// Base64 without a dependency: the fixture is a few hundred KB and this is
/// the same encoder the media tests use.
fn data_uri(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::from("data:image/png;base64,");
    for chunk in bytes.chunks(3) {
        let n = chunk
            .iter()
            .enumerate()
            .fold(0u32, |n, (i, &b)| n | (b as u32) << (16 - 8 * i));
        for i in 0..4 {
            out.push(match i <= chunk.len() {
                true => ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char,
                false => '=',
            });
        }
    }
    out
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
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("a body");
    let json = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        panic!("the response is JSON: {e}\n{}", String::from_utf8_lossy(&bytes))
    });
    (status, json)
}

/// Put one `point` question to the server over one scene, and report where
/// it landed.
async fn point_at(app: &axum::Router, image: &[u8]) -> JsonValue {
    let body = format!(
        r#"{{"state":[{{"type":"image_url","image_url":{{"url":"{}"}}}}],"model":"{MODEL}",
            "questions":{{"where":{{"type":"point","instructions":"{INSTRUCTION}"}}}}}}"#,
        data_uri(image)
    );
    let (status, response) = decide(app, body).await;
    assert_eq!(status, 200, "the decide request was served: {response}");
    let answer = response["answers"]["where"].clone();
    assert_eq!(answer["type"], "point", "{response}");
    // A constrained decode generates, and the usage says so: three digits, the forced
    // separator, three more digits.
    assert!(
        response["usage"]["output_tokens"].as_u64().expect("a count") >= 6,
        "a point generates its digits: {response}"
    );
    answer
}

/// Run every scene through `shape` and assert each answer lands inside its
/// own button.
async fn every_scene_lands_inside_its_button(shape: EngineShape, label: &str) {
    let dir = fixture_dir();
    let Ok(manifest_text) = std::fs::read_to_string(dir.join("manifest.json")) else {
        if gpu_profile::skip_or_fail(&format!("the pointing fixture is absent: {}", dir.display())) {
            return;
        }
        unreachable!("skip_or_fail panics under the profile");
    };
    let manifest: Manifest =
        serde_json::from_str(&manifest_text).unwrap_or_else(|e| panic!("parse the manifest: {e}"));
    let Some(h) = harness(shape) else { return };

    for scene in &manifest.scenes {
        let bytes = std::fs::read(dir.join(&scene.image))
            .unwrap_or_else(|e| panic!("{}: read image: {e}", scene.id));
        let answer = point_at(h.app(), &bytes).await;

        let (x, y) = (
            answer["pixels"]["x"].as_i64().expect("an x in pixels"),
            answer["pixels"]["y"].as_i64().expect("a y in pixels"),
        );
        let [bx0, by0, bx1, by1] = scene.blue_box;
        let [cx, cy] = scene.blue_centre;
        let error = (((x - cx) as f64).powi(2) + ((y - cy) as f64).powi(2)).sqrt();
        eprintln!(
            "ignis decide point [{label}] {}: ({x},{y}) px, target ({cx},{cy}), error {error:.0} px \
             ({:.1}% of side); normalized {} {} uncertainty {} {}",
            scene.id,
            100.0 * error / f64::from(manifest.side),
            answer["normalized"]["x"],
            answer["normalized"]["y"],
            answer["uncertainty"]["x"],
            answer["uncertainty"]["y"],
        );

        assert!(
            x >= bx0 && x <= bx1 && y >= by0 && y <= by1,
            "[{label}] {}: ({x},{y}) is outside the blue button {:?}",
            scene.id,
            scene.blue_box
        );

        // Acceptance 4, on the card: the reported uncertainty is the
        // trace's own place-weighted sum, rescaled onto this axis.
        for axis in ["x", "y"] {
            let digits = answer["digits"][axis].as_array().expect("a per-digit trace");
            assert_eq!(digits.len(), 3, "three digits per axis");
            let width = digits.len();
            let native: f64 = digits
                .iter()
                .enumerate()
                .map(|(place, digit)| {
                    let p = digit["probability"].as_f64().expect("a probability");
                    assert!(p > 0.0 && p <= 1.0, "a real probability: {p}");
                    (1.0 - p) * 10f64.powi((width - 1 - place) as i32)
                })
                .sum();
            let reported = answer["uncertainty"][axis].as_f64().expect("an uncertainty");
            let expected = native * f64::from(manifest.side) / 999.0;
            assert!(
                (reported - expected).abs() < 1.0,
                "[{label}] {}: {axis}'s uncertainty {reported} px is its trace's sum \
                 {expected} px",
                scene.id
            );
            // And the digits really are the number, read left to right.
            let spelled: u64 = digits
                .iter()
                .fold(0, |value, d| value * 10 + d["digit"].as_u64().expect("a digit"));
            assert_eq!(
                answer["normalized"][axis].as_u64().expect("a reading"),
                spelled,
                "[{label}] {}: {axis}'s reading is its own digits",
                scene.id
            );
        }
    }
}

/// Acceptance 3, in the shape an operator gets with `--vision` and no other
/// flags.
#[tokio::test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn a_point_lands_inside_the_blue_button_on_every_scene() {
    every_scene_lands_inside_its_button(vision_shape(), "default").await;
}

/// The same, on a **drafter** load — the only shape that exercises
/// `CudaLeaf::decode`'s claim that a batch with a constrained lane in it
/// takes the plain path.
///
/// A speculative load runs every round as a verify round, and the leaf
/// refuses a constrained lane there rather than accepting one it cannot
/// honour: a draft is proposed by a second model that knows nothing of a
/// permitted set. Without this test that routing is a comment, and a
/// `point` on an operator's drafter load would fail every round with no
/// diagnostic worth reading.
#[tokio::test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn a_point_is_served_on_a_drafter_load_too() {
    let shape = EngineShape {
        speculation: Some(
            ignis_core::Speculation::new(ignis_core::SpeculativeBackend::Dflash2, 7)
                .expect("dflash2-7"),
        ),
        ..vision_shape()
    };
    every_scene_lands_inside_its_button(shape, "dflash2-7").await;
}
