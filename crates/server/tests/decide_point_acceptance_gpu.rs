//! The pre-registered acceptance of the one-pass `point` — spec 13 acceptance
//! 5 (GitHub #260) and spec 14 acceptance 6 (GitHub #263) — measured once,
//! **through `/v1/decide`**, the served artifact, hq-e8-2b with its residual
//! window.
//!
//! Every scene is asked five questions, each its own request:
//!
//! - **`point`, no `method`** — what a caller gets by default: on the served
//!   artifact the head set anchored on the pointing head (spec 14), the
//!   object's extent's centre. This is the floored number.
//! - **`box`, `"method": "head"`** — the same extent as a box; its IoU with the
//!   target is floored on the rectangle sets.
//! - **the pointing head alone** — the same `point`, no `method`, through a
//!   second router over the same engine whose server has no head set: spec
//!   13's answer, measured beside the set it is the anchor of.
//! - **`point`, `"method": "chain"`** and **`box`, `"method": "chain"`** — the
//!   digit chain beside both.
//!
//! Reported beside each set: every method's inside count and median distance
//! from the target's centre (in units of the target's diagonal, spec 14's
//! measure), both boxes' IoU >= 0.5 rate, and the wall time of the default
//! point and the chain point on the same prompt. Those two run in
//! alternating order — even scenes default first, odd scenes chain first —
//! because the second question over one image finds its embedding already
//! encoded (GitHub #243), and a timing that always gave one method the warm
//! cache would be measuring the cache.
//!
//! **Floors.** Asserted only on a whole pre-registered set, recognized by its
//! side, seed and size: spec 13's D (1024 px, seed 20260925, 240 scenes,
//! >= 221) and D4096 (4096 px, 20260926, 240, >= 230), and spec 14's
//!
//! | set | command | `point` inside | head `box` IoU >= 0.5 |
//! |---|---|---|---|
//! | E1 | `scenes.py --varied --seed 20260940` (240) | >= 223 | reported |
//! | E2 | `scenes.py --varied --side 4096 --seed 20260941` (240) | >= 230 | reported |
//! | E3 | `rectangles.py --seed 20260942 --n 120` | >= 108 | >= 84 |
//! | E4 | `rectangles.py --side 4096 --seed 20260943 --n 60` | >= 54 | >= 42 |
//!
//! **`box`'s default** is decided across E1-E4 by the rule spec 14 wrote
//! first: `head` if and only if the head box's IoU >= 0.5 rate is at least
//! the chain box's on every one of them. Each run prints its set's verdict
//! line; the four together are the finding.
//!
//! `IGNIS_POINT_SCENES=<dir with manifest.json>` names the set (generate it
//! into `.scratch/`, which git does not track; see
//! `tools/pointing-scenes/README.md`). Without it the test runs the
//! committed five 1024 px scenes (`fixtures/pointing/1024`, the first of set
//! C), which is what the GPU profile exercises: the whole path, every
//! method, no floor. `IGNIS_POINT_OUT=<dir>` is where the per-scene JSON goes
//! (default the OS temp dir); `IGNIS_POINT_LIMIT=<n>` is a smoke run. The
//! load is the one `make start` serves by default: vision on, hq-e8-2b, a
//! 1024-token prefill chunk, prompt reuse on, and the dflash2 drafter at 7.
//!
//! **A fresh load every few scenes.** The scheduler keeps a finished request
//! — its whole `RequestInput`, a 4096 px image's ~200 MB of patch rows
//! included — for the life of the process (GitHub #262), so one load cannot
//! carry a set's worth of image questions: the first D4096 run died of host
//! memory at scene 86. Five questions a scene makes it [`scenes_per_load`].
//! Reloading changes nothing a head reads.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38).

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::{Body, to_bytes};
use axum::http::Request;
use serde::Deserialize;
use serde_json::{Value as JsonValue, json};
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

#[derive(Deserialize)]
struct Manifest {
    side: u32,
    /// `null` on a set nobody generated from a seed — the owner's
    /// screenshots — which is never a pre-registered one.
    seed: Option<u64>,
    #[serde(default)]
    source: Option<String>,
    scenes: Vec<Scene>,
}

#[derive(Deserialize)]
struct Scene {
    id: String,
    image: String,
    blue_box: [i64; 4],
    blue_centre: [i64; 2],
    instruction: String,
    #[serde(default)]
    kind: Option<String>,
}

/// A pre-registered set's floors: `point` inside, and the head box's
/// IoU >= 0.5 count where the spec floors it.
struct Floors {
    name: &'static str,
    point: usize,
    head_box: Option<usize>,
}

/// The pre-registered floors of a set, if it is one of spec 13's or spec
/// 14's, whole. Any other set — C, a fixture, a partial run — is reported
/// and never judged: a floor was written for scenes never used to choose
/// anything.
fn floors_of(manifest: &Manifest, scenes: usize) -> Option<Floors> {
    let rectangles = manifest.source.as_deref().is_some_and(|s| s.starts_with("rectangles.py"));
    let floors = |name, point, head_box| Some(Floors { name, point, head_box });
    match (rectangles, manifest.side, manifest.seed?, scenes) {
        (false, 1024, 20_260_925, 240) => floors("D", 221, None),
        (false, 4096, 20_260_926, 240) => floors("D4096", 230, None),
        (false, 1024, 20_260_940, 240) => floors("E1", 223, None),
        (false, 4096, 20_260_941, 240) => floors("E2", 230, None),
        (true, 1024, 20_260_942, 120) => floors("E3", 108, Some(84)),
        (true, 4096, 20_260_943, 60) => floors("E4", 54, Some(42)),
        _ => None,
    }
}

/// Scenes per model load (the module docs): five image questions a scene,
/// ~200 MB of retained patch rows each at 4096 px.
fn scenes_per_load(side: u32) -> usize {
    if side > 1024 { 8 } else { 40 }
}

fn data_uri(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::from("data:image/png;base64,");
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().enumerate().fold(0u32, |n, (i, &b)| n | (b as u32) << (16 - 8 * i));
        for i in 0..4 {
            out.push(match i <= chunk.len() {
                true => ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char,
                false => '=',
            });
        }
    }
    out
}

/// One question over one scene; the answer, the usage and the wall time.
async fn ask(
    app: &axum::Router,
    uri: &str,
    kind: &str,
    instruction: &str,
    method: Option<&str>,
) -> (JsonValue, JsonValue, f64) {
    let method = method.map(|m| format!(r#","method":"{m}""#)).unwrap_or_default();
    let instruction = serde_json::to_string(instruction).expect("a string");
    let body = format!(
        r#"{{"state":[{{"type":"image_url","image_url":{{"url":"{uri}"}}}}],"model":"{MODEL}",
            "questions":{{"q":{{"type":"{kind}","instructions":{instruction}{method}}}}}}}"#
    );
    let request = Request::builder()
        .method("POST")
        .uri("/v1/decide")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .expect("a well-formed request");
    let started = Instant::now();
    let response = app.clone().oneshot(request).await.expect("the router answers");
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.expect("a body");
    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
    let json: JsonValue = serde_json::from_slice(&bytes).expect("a JSON body");
    assert_eq!(status, 200, "{json}");
    (json["answers"]["q"].clone(), json["usage"].clone(), elapsed)
}

/// A live server over the real scheduler, wired as `main.rs` wires one under
/// `--vision` — and a second router over the same engine whose server has no
/// head set: the pointing head alone, spec 13's `point`. `None` when the
/// profile says to skip.
struct Loaded {
    app: axum::Router,
    pointing_only: axum::Router,
    server: Server,
    driver: std::thread::JoinHandle<()>,
}

impl Loaded {
    fn close(self) {
        drop(self.app);
        drop(self.pointing_only);
        drop(self.server);
        let _ = self.driver.join();
    }
}

fn load(path: &Path, shape: EngineShape) -> Option<Loaded> {
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend.eos_token_id().expect("an eos token");
    let processor = frontend.vision_processor().unwrap_or_else(|e| panic!("vision processor: {e}"));
    let scheduler = match cuda_scheduler(path, MODEL.into(), eos, shape) {
        Ok((scheduler, _)) => scheduler,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("cuda_scheduler: {e}"));
            return None;
        }
    };
    let (engine, driver) = Engine::with_clock_and_driver(Box::new(scheduler), Arc::new(SystemClock));
    let acquirer =
        MediaAcquirer::new(Arc::new(processor.clone()), processor.options().clone(), MediaPolicy::new(false, 1 << 26));
    let provider = ignis_server::artifact_template::ArtifactTemplateProvider::new(frontend).with_vision(processor);
    let server = Server::new(engine, Box::new(provider))
        .with_media(Arc::new(acquirer))
        .with_request_timeout(Duration::from_secs(600));
    let calibration = server.calibration.expect("the served artifact is calibrated");
    assert!(calibration.set.is_some(), "with a head set beside its pointing head");
    let mut pointing_only = server.clone();
    pointing_only.calibration = Some(ignis_core::pointing::Calibration { set: None, ..calibration });
    Some(Loaded {
        app: server.app(),
        pointing_only: pointing_only.app(),
        server,
        driver,
    })
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some(sorted[sorted.len() / 2])
}

fn iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    let ix = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let iy = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = ix * iy;
    let area = |z: [f64; 4]| (z[2] - z[0]).max(0.0) * (z[3] - z[1]).max(0.0);
    if inter > 0.0 { inter / (area(a) + area(b) - inter) } else { 0.0 }
}

/// Per method: inside count, distances, and (for boxes) IoUs.
#[derive(Default)]
struct Tally {
    inside: usize,
    distance: Vec<f64>,
    iou: Vec<f64>,
}

impl Tally {
    fn iou_hits(&self) -> usize {
        self.iou.iter().filter(|&&v| v >= 0.5).count()
    }

    fn summary(&self, n: usize) -> JsonValue {
        json!({
            "inside": self.inside,
            "of": n,
            "median_distance_per_diagonal": median(&self.distance),
            "iou_ge_half": (!self.iou.is_empty()).then(|| self.iou_hits()),
            "median_iou": median(&self.iou),
        })
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1 (and IGNIS_POINT_SCENES)"]
async fn a_point_with_no_method_meets_the_preregistered_floor() {
    let dir = std::env::var("IGNIS_POINT_SCENES").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("pointing")
            .join("1024")
    });
    let manifest: Manifest = serde_json::from_str(
        &std::fs::read_to_string(dir.join("manifest.json")).unwrap_or_else(|e| panic!("{}: {e}", dir.display())),
    )
    .unwrap_or_else(|e| panic!("parse the manifest: {e}"));
    let limit = std::env::var("IGNIS_POINT_LIMIT").ok().map(|v| v.parse::<usize>().expect("a count"));
    let scenes: Vec<&Scene> = manifest.scenes.iter().take(limit.unwrap_or(usize::MAX)).collect();

    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let shape = EngineShape {
        vision: Some(ignis_core::Vision::default()),
        speculation: Some(
            ignis_core::Speculation::new(ignis_core::SpeculativeBackend::Dflash2, 7).expect("dflash2-7"),
        ),
        ..EngineShape::default()
    };
    let kv_format = shape.kv_format;

    let mut rows = Vec::new();
    let (mut set, mut set_box, mut anchor, mut chain, mut chain_box) =
        (Tally::default(), Tally::default(), Tally::default(), Tally::default(), Tally::default());
    let (mut default_first_ms, mut chain_first_ms, mut default_second_ms, mut chain_second_ms) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let per_load = scenes_per_load(manifest.side);
    let mut loaded: Option<Loaded> = None;
    for (i, scene) in scenes.iter().enumerate() {
        if i % per_load == 0 {
            if let Some(previous) = loaded.take() {
                previous.close();
            }
            let Some(fresh) = load(path, shape) else { return };
            loaded = Some(fresh);
        }
        let l = loaded.as_ref().expect("loaded above");
        let bytes = std::fs::read(dir.join(&scene.image)).unwrap_or_else(|e| panic!("{}: {e}", scene.id));
        let uri = data_uri(&bytes);
        let instruction = scene.instruction.as_str();
        let default_first = i % 2 == 0;
        let (default_point, chain_point) = if default_first {
            let d = ask(&l.app, &uri, "point", instruction, None).await;
            let c = ask(&l.app, &uri, "point", instruction, Some("chain")).await;
            (d, c)
        } else {
            let c = ask(&l.app, &uri, "point", instruction, Some("chain")).await;
            let d = ask(&l.app, &uri, "point", instruction, None).await;
            (d, c)
        };
        let head_box = ask(&l.app, &uri, "box", instruction, Some("head")).await;
        let anchor_point = ask(&l.pointing_only, &uri, "point", instruction, None).await;
        let chain_boxed = ask(&l.app, &uri, "box", instruction, Some("chain")).await;

        let b = scene.blue_box;
        let target = [b[0] as f64, b[1] as f64, b[2] as f64, b[3] as f64];
        let [cx, cy] = scene.blue_centre;
        let diagonal = ((b[2] - b[0]) as f64).hypot((b[3] - b[1]) as f64);
        let judge_point = |answer: &JsonValue, method: &str, extent: bool, tally: &mut Tally| {
            assert_eq!(answer["type"], "point", "{}: {answer}", scene.id);
            assert_eq!(answer["method"], method, "{}: {answer}", scene.id);
            assert_eq!(answer.get("extent").is_some(), extent, "{}: {answer}", scene.id);
            let x = answer["pixels"]["x"].as_i64().expect("an x");
            let y = answer["pixels"]["y"].as_i64().expect("a y");
            let inside = x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3];
            let distance = ((x - cx) as f64).hypot((y - cy) as f64) / diagonal;
            tally.inside += usize::from(inside);
            tally.distance.push(distance);
            (x, y, inside, distance)
        };
        let judge_box = |answer: &JsonValue, method: &str, tally: &mut Tally| {
            assert_eq!(answer["type"], "box", "{}: {answer}", scene.id);
            assert_eq!(answer["method"], method, "{}: {answer}", scene.id);
            let c = |k: &str| answer["pixels"][k].as_i64().expect("a corner") as f64;
            let got = [c("x0"), c("y0"), c("x1"), c("y1")];
            let value = iou(got, target);
            tally.iou.push(value);
            (got, value)
        };
        let (sx, sy, s_in, s_d) = judge_point(&default_point.0, "head", true, &mut set);
        let (ax, ay, a_in, a_d) = judge_point(&anchor_point.0, "head", false, &mut anchor);
        let (qx, qy, q_in, q_d) = judge_point(&chain_point.0, "chain", false, &mut chain);
        let (_, hb_iou) = judge_box(&head_box.0, "head", &mut set_box);
        let (_, cb_iou) = judge_box(&chain_boxed.0, "chain", &mut chain_box);
        // The point's extent is the head box: the same pass, the same rule.
        let extent = &default_point.0["extent"];
        let e = |k: &str| extent[k].as_i64().expect("an extent corner") as f64;
        set.iou.push(iou([e("x0"), e("y0"), e("x1"), e("y1")], target));
        assert_eq!(default_point.1["output_tokens"], 0, "{}: a head point generates nothing", scene.id);
        assert_eq!(head_box.1["output_tokens"], 0, "{}: a head box generates nothing", scene.id);
        match default_first {
            true => {
                default_first_ms.push(default_point.2);
                chain_second_ms.push(chain_point.2);
            }
            false => {
                chain_first_ms.push(chain_point.2);
                default_second_ms.push(default_point.2);
            }
        }
        let mark = |inside: bool| if inside { "in " } else { "OUT" };
        eprintln!(
            "[{}/{}] {}: set ({sx},{sy}) {} {:.0} ms share {:.3} box IoU {hb_iou:.2} | head ({ax},{ay}) {} | chain ({qx},{qy}) {} {:.0} ms box IoU {cb_iou:.2} | target {:?}",
            i + 1,
            scenes.len(),
            scene.id,
            mark(s_in),
            default_point.2,
            default_point.0["region"]["share"].as_f64().unwrap_or(f64::NAN),
            mark(a_in),
            mark(q_in),
            chain_point.2,
            scene.blue_box
        );
        rows.push(json!({
            "id": scene.id, "kind": scene.kind, "instruction": scene.instruction,
            "box": scene.blue_box, "centre": scene.blue_centre,
            "set": {"answer": default_point.0, "inside": s_in, "distance": s_d, "ms": default_point.2, "first": default_first},
            "head_box": {"answer": head_box.0, "iou": hb_iou, "ms": head_box.2},
            "pointing_head": {"answer": anchor_point.0, "inside": a_in, "distance": a_d},
            "chain": {"answer": chain_point.0, "inside": q_in, "distance": q_d, "ms": chain_point.2, "first": !default_first},
            "chain_box": {"answer": chain_boxed.0, "iou": cb_iou, "ms": chain_boxed.2},
        }));
    }
    if let Some(last) = loaded.take() {
        last.close();
    }

    let n = scenes.len();
    let floors = floors_of(&manifest, n);
    let rate = |hits: usize| hits as f64 / n.max(1) as f64;
    let head_box_wins = rate(set_box.iou_hits()) >= rate(chain_box.iou_hits());
    let summary = json!({
        "set": dir.file_name().and_then(|s| s.to_str()),
        "preregistered": floors.as_ref().map(|f| f.name),
        "side": manifest.side,
        "seed": manifest.seed,
        "scenes": n,
        "kv_format": format!("{kv_format:?}"),
        "point_default_head_set": set.summary(n),
        "box_head": set_box.summary(n),
        "point_pointing_head_alone": anchor.summary(n),
        "point_chain": chain.summary(n),
        "box_chain": chain_box.summary(n),
        "box_default_rule": {
            "head_box_iou_rate": rate(set_box.iou_hits()),
            "chain_box_iou_rate": rate(chain_box.iou_hits()),
            "head_at_least_chain_on_this_set": head_box_wins,
        },
        "default_ms_first_median": median(&default_first_ms),
        "chain_ms_first_median": median(&chain_first_ms),
        "default_ms_second_median": median(&default_second_ms),
        "chain_ms_second_median": median(&chain_second_ms),
    });
    eprintln!("summary: {summary}");
    let out = std::env::var("IGNIS_POINT_OUT").map(PathBuf::from).unwrap_or_else(|_| std::env::temp_dir());
    let file = out.join(format!(
        "point-acceptance-{}.json",
        dir.file_name().and_then(|s| s.to_str()).unwrap_or("scenes")
    ));
    std::fs::create_dir_all(&out).ok();
    std::fs::write(&file, serde_json::to_vec_pretty(&json!({"summary": summary, "rows": rows})).unwrap())
        .unwrap_or_else(|e| panic!("write {}: {e}", file.display()));
    eprintln!("wrote {}", file.display());

    if let Some(floors) = floors {
        assert!(
            set.inside >= floors.point,
            "{}: the pre-registered floor is {} `point`s inside of {n}; the default landed inside on {}",
            floors.name,
            floors.point,
            set.inside
        );
        if let Some(floor) = floors.head_box {
            assert!(
                set_box.iou_hits() >= floor,
                "{}: the pre-registered floor is {floor} head boxes at IoU >= 0.5 of {n}; they reached {}",
                floors.name,
                set_box.iou_hits()
            );
        }
    }
}
