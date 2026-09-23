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
//! **GitHub #260 moved the default.** A `point` with no `method` is now read
//! off the served artifact's calibrated pointing head in one pass, so the
//! chain tests here ask for `"method": "chain"` by name — they are the
//! chain's regression guard and stay one — and a third test puts the same
//! scenes to the default, the head, with zero decode rounds. The head is not
//! held to "every scene": under hq-e8-2b it reads `large` at (1520, 3312),
//! off the button, with a region share of 0.025 (under BF16 it lands inside),
//! which is the head's measured error rate showing on three scenes — spec 13's
//! acceptance is 237 of 240 at this size, not 240. What it is held to is the
//! failure signature a caller can see: at most one miss in three, and never a
//! miss the head was sure of.
//!
//! **GitHub #263 moved it again.** On the served artifact the default is now
//! the head set anchored on the pointing head: the point is the centre of the
//! object's `extent`, which the answer carries. The same bar holds, and the
//! extent is checked to hold the point it is the centre of. A fourth test
//! asks each scene for a `box` with `"method": "head"` — the same extent, in
//! one pass — and holds its shape; its IoU with the button is printed, not
//! asserted (spec 14 reports buttons and floors large objects).
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
/// it landed. `method` is sent as given, or left out for the default.
async fn point_at(app: &axum::Router, image: &[u8], method: Option<&str>) -> JsonValue {
    let method = method.map(|m| format!(r#","method":"{m}""#)).unwrap_or_default();
    let body = format!(
        r#"{{"state":[{{"type":"image_url","image_url":{{"url":"{}"}}}}],"model":"{MODEL}",
            "questions":{{"where":{{"type":"point","instructions":"{INSTRUCTION}"{method}}}}}}}"#,
        data_uri(image)
    );
    let (status, response) = decide(app, body).await;
    assert_eq!(status, 200, "the decide request was served: {response}");
    let answer = response["answers"]["where"].clone();
    assert_eq!(answer["type"], "point", "{response}");
    let generated = response["usage"]["output_tokens"].as_u64().expect("a count");
    match answer["method"].as_str() {
        // A constrained decode generates, and the usage says so: three
        // digits, the forced separator, three more digits.
        Some("chain") => assert!(generated >= 6, "a chain point generates its digits: {response}"),
        // GitHub #260: one prefill and no decode round.
        Some("head") => assert_eq!(generated, 0, "a head point generates nothing: {response}"),
        _ => panic!("every point answer names its method: {response}"),
    }
    answer
}

/// Run every scene through `shape` and assert each answer lands inside its
/// own button.
async fn every_scene_lands_inside_its_button(shape: EngineShape, label: &str, method: Option<&str>) {
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

    // GitHub #260: a head point's misses, as (scene, share).
    let mut head_misses: Vec<(String, f64)> = Vec::new();
    for scene in &manifest.scenes {
        let bytes = std::fs::read(dir.join(&scene.image))
            .unwrap_or_else(|e| panic!("{}: read image: {e}", scene.id));
        let answer = point_at(h.app(), &bytes, method).await;

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

        let inside = x >= bx0 && x <= bx1 && y >= by0 && y <= by1;

        // GitHub #260: a head point's uncertainty is one image token per
        // axis and it carries no trace; the rest of this loop is the chain's.
        if answer["method"] == "head" {
            let cell = answer["uncertainty"]["x"].as_f64().expect("an uncertainty");
            assert!(cell > 0.0, "[{label}] {}: one token cell, in pixels: {answer}", scene.id);
            // GitHub #263: the served artifact has a head set, so the point
            // is its extent's centre.
            let corner = |c: &str| answer["extent"][c].as_i64().unwrap_or_else(|| panic!("{c}: {answer}"));
            let (x0, y0, x1, y1) = (corner("x0"), corner("y0"), corner("x1"), corner("y1"));
            assert!(
                x0 <= x && x <= x1 && y0 <= y && y <= y1 && ((x0 + x1) / 2 - x).abs() <= 1,
                "[{label}] {}: ({x},{y}) is the centre of its extent {answer}",
                scene.id
            );
            let share = answer["region"]["share"].as_f64().expect("a region share");
            if !inside {
                head_misses.push((scene.id.clone(), share));
            }
            continue;
        }
        assert!(
            inside,
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
    assert!(
        head_misses.len() <= 1,
        "[{label}] the head missed {} of {} scenes: {head_misses:?}",
        head_misses.len(),
        manifest.scenes.len()
    );
    assert!(
        head_misses.iter().all(|&(_, share)| share < CONFIDENT_SHARE),
        "[{label}] the head missed with a region share of {CONFIDENT_SHARE} or more — a miss its \
         own answer vouched for: {head_misses:?}"
    );
}

/// The region share a head miss must stay under: the median share of the
/// head's *hits* on set D4096 was 0.152 and of its misses 0.052
/// (`docs/findings/2026-09-22-the-head-points-through-decide.md`), so a miss
/// at or above this is one the answer itself vouched for.
const CONFIDENT_SHARE: f64 = 0.10;

/// Acceptance 3, in the shape an operator gets with `--vision` and no other
/// flags.
#[tokio::test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn a_point_lands_inside_the_blue_button_on_every_scene() {
    every_scene_lands_inside_its_button(vision_shape(), "chain", Some("chain")).await;
}

/// GitHub #260: the default — the served artifact's calibrated pointing head,
/// read in one pass — held to the same bar on the same scenes.
#[tokio::test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn a_point_with_no_method_is_read_off_the_head_inside_the_button() {
    every_scene_lands_inside_its_button(vision_shape(), "head", None).await;
}

/// GitHub #263: a `box` asking for `head` on every scene — the head set's
/// extent through the whole stack in one pass: no decode round, no digits,
/// four corners in the submitted image's pixels in order, one token cell of
/// uncertainty per edge. Its IoU with the button is printed for the record.
#[tokio::test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn a_box_asking_for_the_head_frames_each_scene_in_one_pass() {
    let dir = fixture_dir();
    let Ok(manifest_text) = std::fs::read_to_string(dir.join("manifest.json")) else {
        if gpu_profile::skip_or_fail(&format!("the pointing fixture is absent: {}", dir.display())) {
            return;
        }
        unreachable!("skip_or_fail panics under the profile");
    };
    let manifest: Manifest =
        serde_json::from_str(&manifest_text).unwrap_or_else(|e| panic!("parse the manifest: {e}"));
    let Some(h) = harness(vision_shape()) else { return };
    for scene in &manifest.scenes {
        let bytes = std::fs::read(dir.join(&scene.image)).unwrap_or_else(|e| panic!("{}: {e}", scene.id));
        let body = format!(
            r#"{{"state":[{{"type":"image_url","image_url":{{"url":"{}"}}}}],"model":"{MODEL}",
                "questions":{{"frame":{{"type":"box","instructions":"{INSTRUCTION}","method":"head"}}}}}}"#,
            data_uri(&bytes)
        );
        let (status, response) = decide(h.app(), body).await;
        assert_eq!(status, 200, "{response}");
        let answer = &response["answers"]["frame"];
        assert_eq!((answer["type"].as_str(), answer["method"].as_str()), (Some("box"), Some("head")), "{response}");
        assert_eq!(response["usage"]["output_tokens"], 0, "a head box generates nothing: {response}");
        assert!(answer.get("digits").is_none(), "{answer}");
        let corner = |c: &str| answer["pixels"][c].as_i64().unwrap_or_else(|| panic!("{c}: {answer}"));
        let got = [corner("x0"), corner("y0"), corner("x1"), corner("y1")];
        let side = i64::from(manifest.side);
        assert!(
            0 <= got[0] && got[0] < got[2] && got[2] <= side && 0 <= got[1] && got[1] < got[3] && got[3] <= side,
            "{}: the corners are in order and on the image: {answer}",
            scene.id
        );
        for edge in ["x0", "y0", "x1", "y1"] {
            assert!(answer["uncertainty"][edge].as_f64().is_some_and(|u| u > 0.0), "{edge}: {answer}");
        }
        let b = scene.blue_box;
        let inter = (got[2].min(b[2]) - got[0].max(b[0])).max(0) * (got[3].min(b[3]) - got[1].max(b[1])).max(0);
        let area = |r: [i64; 4]| (r[2] - r[0]) * (r[3] - r[1]);
        let iou = inter as f64 / (area(got) + area(b) - inter) as f64;
        eprintln!("ignis decide head box {}: {got:?} vs button {b:?}, IoU {iou:.2}", scene.id);
    }
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
    every_scene_lands_inside_its_button(shape, "dflash2-7", Some("chain")).await;
}
