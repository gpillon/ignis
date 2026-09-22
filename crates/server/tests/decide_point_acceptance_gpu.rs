//! Spec 13 acceptance 5 (GitHub #260): the pre-registered acceptance of the
//! one-pass `point`, measured once — **through `/v1/decide` with no
//! `method`**, the served artifact, hq-e8-2b with its residual window.
//!
//! Pre-registered in `docs/specs/decide/13-point-by-attention-head.md`
//! § Testing Decisions before any run: two sets from the scene generator
//! (`tools/pointing-scenes/scenes.py --varied`) never used to choose the head
//! or the rule — **D**, 240 scenes at 1024 px (seed 20260925), and **D4096**,
//! 240 at 4096 px (seed 20260926) — inside the target on **at least 221 of
//! 240 at 1024 px and at least 230 of 240 at 4096 px**. Asserted only on a
//! full 240-scene set of one of those two sizes.
//!
//! Reported beside it, not asserted: the chain (`"method": "chain"`) on the
//! same scenes, both methods' distance from the target's centre, and the
//! wall time of a head point and a chain point on the same prompt. The two
//! questions over one scene run in alternating order — even scenes head
//! first, odd scenes chain first — because the second question over one
//! image finds its embedding already encoded (GitHub #243), and a timing
//! that always gave one method the warm cache would be measuring the cache.
//!
//! `IGNIS_POINT_SCENES=<dir with manifest.json>` names the set (generate it
//! with the command above into `.scratch/`, which git does not track);
//! `IGNIS_POINT_OUT=<dir>` is where the per-scene JSON goes (default the OS
//! temp dir); `IGNIS_POINT_LIMIT=<n>` is a smoke run. The load is the one
//! `make start` serves by default: vision on, hq-e8-2b, a 1024-token prefill
//! chunk, prompt reuse on, and the dflash2 drafter at 7.
//!
//! **A fresh load every [`SCENES_PER_LOAD`] scenes.** The scheduler keeps a
//! finished request — its whole `RequestInput`, a 4096 px image's ~200 MB of
//! patch rows included — for the life of the process, so one load cannot
//! carry 480 image questions: the first D4096 run died of host memory at
//! scene 86 (GitHub #262). Reloading between batches changes nothing a head
//! point reads.
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
/// The pre-registered floors (spec 13): the engine's measured 227 and 236 on
/// set C and C4096, minus the slack of six the earlier criteria used.
const FLOOR_1024: usize = 221;
const FLOOR_4096: usize = 230;
const FULL_SET: usize = 240;
/// Scenes per model load (see the module docs): 40 scenes are 80 image
/// questions, ~16 GB of retained patch rows at 4096 px.
const SCENES_PER_LOAD: usize = 40;

#[derive(Deserialize)]
struct Manifest {
    side: u32,
    seed: u64,
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

/// One point question over one scene; the answer and the wall time.
async fn ask(app: &axum::Router, uri: &str, instruction: &str, method: Option<&str>) -> (JsonValue, JsonValue, f64) {
    let method = method.map(|m| format!(r#","method":"{m}""#)).unwrap_or_default();
    let instruction = serde_json::to_string(instruction).expect("a string");
    let body = format!(
        r#"{{"state":[{{"type":"image_url","image_url":{{"url":"{uri}"}}}}],"model":"{MODEL}",
            "questions":{{"where":{{"type":"point","instructions":{instruction}{method}}}}}}}"#
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
    (json["answers"]["where"].clone(), json["usage"].clone(), elapsed)
}

/// A live server over the real scheduler, wired as `main.rs` wires one under
/// `--vision`; `None` when the profile says to skip.
fn load(path: &Path, shape: EngineShape) -> Option<(axum::Router, Server, std::thread::JoinHandle<()>)> {
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
    assert!(server.pointing_head.is_some(), "the served artifact has a calibrated pointing head");
    Some((server.app(), server, driver))
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[values.len() / 2])
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1 (and IGNIS_POINT_SCENES)"]
async fn a_point_with_no_method_meets_the_preregistered_floor() {
    let Ok(dir) = std::env::var("IGNIS_POINT_SCENES") else {
        eprintln!("SKIP: IGNIS_POINT_SCENES names no scene set (generate D or D4096 first)");
        return;
    };
    let dir = PathBuf::from(dir);
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

    let side = f64::from(manifest.side);
    let unit = |px: i64| px as f64 / side * 999.0;
    let mut rows = Vec::new();
    let (mut head_inside, mut chain_inside) = (0usize, 0usize);
    let (mut head_dist, mut chain_dist) = (Vec::new(), Vec::new());
    let (mut head_first_ms, mut chain_first_ms, mut head_second_ms, mut chain_second_ms) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut loaded: Option<(axum::Router, Server, std::thread::JoinHandle<()>)> = None;
    for (i, scene) in scenes.iter().enumerate() {
        if i % SCENES_PER_LOAD == 0 {
            if let Some((app, server, driver)) = loaded.take() {
                drop(app);
                drop(server);
                let _ = driver.join();
            }
            let Some(fresh) = load(path, shape) else { return };
            loaded = Some(fresh);
        }
        let app = &loaded.as_ref().expect("loaded above").0;
        let bytes = std::fs::read(dir.join(&scene.image)).unwrap_or_else(|e| panic!("{}: {e}", scene.id));
        let uri = data_uri(&bytes);
        let head_first = i % 2 == 0;
        let (head, chain) = if head_first {
            let head = ask(app, &uri, &scene.instruction, None).await;
            let chain = ask(app, &uri, &scene.instruction, Some("chain")).await;
            (head, chain)
        } else {
            let chain = ask(app, &uri, &scene.instruction, Some("chain")).await;
            let head = ask(app, &uri, &scene.instruction, None).await;
            (head, chain)
        };
        let [bx0, by0, bx1, by1] = scene.blue_box;
        let [cx, cy] = scene.blue_centre;
        let judged = |answer: &JsonValue, expected: &str| {
            assert_eq!(answer["type"], "point", "{}: {answer}", scene.id);
            assert_eq!(answer["method"], expected, "{}: {answer}", scene.id);
            let x = answer["pixels"]["x"].as_i64().expect("an x");
            let y = answer["pixels"]["y"].as_i64().expect("a y");
            let inside = x >= bx0 && x <= bx1 && y >= by0 && y <= by1;
            let distance = (unit(x - cx)).abs().max((unit(y - cy)).abs());
            (x, y, inside, distance)
        };
        let (hx, hy, h_in, h_d) = judged(&head.0, "head");
        let (qx, qy, q_in, q_d) = judged(&chain.0, "chain");
        assert_eq!(head.1["output_tokens"], 0, "{}: a head point generates nothing", scene.id);
        head_inside += usize::from(h_in);
        chain_inside += usize::from(q_in);
        head_dist.push(h_d);
        chain_dist.push(q_d);
        match head_first {
            true => {
                head_first_ms.push(head.2);
                chain_second_ms.push(chain.2);
            }
            false => {
                chain_first_ms.push(chain.2);
                head_second_ms.push(head.2);
            }
        }
        eprintln!(
            "[{}/{}] {}: head ({hx},{hy}) {} {:.0} ms share {:.3} | chain ({qx},{qy}) {} {:.0} ms | target {:?}",
            i + 1,
            scenes.len(),
            scene.id,
            if h_in { "in " } else { "OUT" },
            head.2,
            head.0["region"]["share"].as_f64().unwrap_or(f64::NAN),
            if q_in { "in " } else { "OUT" },
            chain.2,
            scene.blue_box
        );
        rows.push(json!({
            "id": scene.id, "kind": scene.kind, "instruction": scene.instruction,
            "box": scene.blue_box, "centre": scene.blue_centre,
            "head": {"answer": head.0, "inside": h_in, "distance": h_d, "ms": head.2, "first": head_first},
            "chain": {"answer": chain.0, "inside": q_in, "distance": q_d, "ms": chain.2, "first": !head_first},
        }));
    }

    let n = scenes.len();
    let summary = json!({
        "set": dir.file_name().and_then(|s| s.to_str()),
        "side": manifest.side,
        "seed": manifest.seed,
        "scenes": n,
        "kv_format": format!("{kv_format:?}"),
        "head_inside": head_inside,
        "chain_inside": chain_inside,
        "head_median_distance": median(&mut head_dist.clone()),
        "chain_median_distance": median(&mut chain_dist.clone()),
        "head_ms_first_median": median(&mut head_first_ms),
        "chain_ms_first_median": median(&mut chain_first_ms),
        "head_ms_second_median": median(&mut head_second_ms),
        "chain_ms_second_median": median(&mut chain_second_ms),
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

    if let Some((app, server, driver)) = loaded.take() {
        drop(app);
        drop(server);
        let _ = driver.join();
    }

    if n == FULL_SET {
        let floor = match manifest.side {
            1024 => Some(FLOOR_1024),
            4096 => Some(FLOOR_4096),
            _ => None,
        };
        if let Some(floor) = floor {
            assert!(
                head_inside >= floor,
                "the pre-registered floor at {} px is {floor} of {FULL_SET}; the head landed inside on {head_inside}",
                manifest.side
            );
        }
    }
}
