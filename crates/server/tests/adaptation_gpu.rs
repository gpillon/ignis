//! ADR 0063 §3, spec 27b-16gb/04 AC 4 and 5 (GitHub #336): a 27B load under
//! `rtx5090` whose explicit VRAM budget refuses the profile's 917,504 tokens
//! adapts the values the profile gave, logs each step in
//! `ignis.runtime.vram_plan`'s `adapted`, and serves at the values it kept --
//! `GET /v1/models` says the context it runs, and a request decodes. AC 1's
//! leg part A left: the 27B at two and at one decode lane, named, inside an
//! explicit budget, loads and decodes. And under `rtx50-16gb`, through the
//! server's own load path, the 27B runs two lanes with its drafter's head
//! left to `auto` and says `draft_head auto -> full`.
//!
//! Every budget comes from the load's own plan. Refused at a 1 GiB budget,
//! the plan's refusal carries what it needs at the values it was given
//! (`VramPlanError::BelowMinimum`'s `needed_bytes`), before anything is
//! materialized. The spec's 17,179,869,184-byte budget cannot be used: it is
//! below today's NVFP4 27B's weights line alone, 17,234,723,840 bytes (the
//! 16 GB body is #337/#338).
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or an unavailable GPU is a **skip**; under the profile
//! the same condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, to_bytes};
use axum::http::Request;
use tower::ServiceExt;

use ignis_artifact::{CudaDevice, FrontendSet, Reader};
use ignis_core::compute::ModelFamily;
use ignis_core::gpu_profile;
use ignis_core::{KvFormat, KvGeometry, VramMode, VramPlanError};
use ignis_server::Server;
use ignis_server::adaptation::{Adaptable, Attempt, StartRefusal};
use ignis_server::engine::Engine;
use ignis_server::runtime::{EngineShape, Qwen27bLoadError, qwen27b_load};
use ignis_server::telemetry::SystemClock;
use tracing_subscriber::layer::SubscriberExt;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";

/// The options of a start with `flags`, fitted to the 27B.
fn the_27b(flags: &[&str]) -> ignis_server::config::Config {
    let args: Vec<String> = flags.iter().map(|flag| flag.to_string()).collect();
    let ignis_server::config::ConfigOutcome::Config(config) =
        ignis_server::config::resolve(&args, |_| None).expect("resolve")
    else {
        panic!("expected a runnable config");
    };
    config.for_family(ModelFamily::Qwen38_27b).expect("the 27B")
}

/// The 27B's shape under `--profile rtx5090`, as a start resolves it: every
/// value the profile's or the default's, so every one adapts.
fn rtx5090() -> EngineShape {
    EngineShape::from(&the_27b(&["--profile", "rtx5090"]))
}

fn explicit(budget_bytes: u64) -> VramMode {
    VramMode::Explicit { budget_bytes, allow_oversubscription: false }
}

fn page_bytes() -> u64 {
    KvFormat::HqE8_2b.page_bytes(KvGeometry::qwen38_27b())
}

/// The artifact's EOS and frontend, or `None` where the profile allows a
/// skip.
fn artifact() -> Option<(u32, FrontendSet)> {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return None;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend.eos_token_id().expect("qwen3.8-27b generation config must carry eos_token_id");
    Some((eos, frontend))
}

/// What the 27B's plan needs at `shape`'s own values: its refusal at a
/// 1 GiB budget, with nothing allowed to adapt, carries it. `None` where the
/// profile allows a skip (no GPU, or a card too busy for even that plan).
fn needed_bytes(shape: EngineShape, eos: u32) -> Option<u64> {
    let refused = EngineShape { vram: explicit(1 << 30), adaptable: Adaptable::NONE, ..shape };
    match qwen27b_load(Path::new(ARTIFACT), MODEL.into(), eos, refused, None, &Default::default()) {
        Ok(_) => panic!("a 1 GiB budget holds no 27B"),
        Err(Qwen27bLoadError::VramPlan(StartRefusal {
            error: VramPlanError::BelowMinimum { needed_bytes, .. },
            tried,
        })) => {
            assert!(tried.is_empty(), "nothing adapts: {tried:?}");
            Some(needed_bytes)
        }
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("plan the 27B at {:?}: {e}", shape.adaptation_start()));
            None
        }
    }
}

/// The card's free memory: an explicit budget above it is refused before
/// any plan, so a budget this test picks must fit under it. `None` where the
/// profile allows a skip.
fn free_bytes() -> Option<u64> {
    match CudaDevice::nvml_memory(0) {
        Ok((free, _)) => Some(free),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("NVML reads no card: {e}"));
            None
        }
    }
}

/// Every event `run` logs on this thread, as the JSON lines the server
/// writes, and what it returned.
fn captured<T>(run: impl FnOnce() -> T) -> (T, Vec<serde_json::Value>) {
    let sink = Arc::new(ignis_logging::MemorySink::new());
    let subscriber = tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone()));
    let value = tracing::subscriber::with_default(subscriber, run);
    let events = sink.lines().iter().map(|line| serde_json::from_str(line).expect("json")).collect();
    (value, events)
}

/// The events of `events` called `name`.
fn named<'a>(events: &'a [serde_json::Value], name: &str) -> Vec<&'a serde_json::Value> {
    events.iter().filter(|event| event["event_name"] == name).collect()
}

/// The GPU-backed router plus the model thread that owns its VRAM, the
/// `ignis.runtime.vram_plan` event its load wrote, and the shape the load
/// kept. Dropping it disconnects the engine and joins the model thread, which
/// releases the weights before the next load.
struct Harness {
    app: Option<axum::Router>,
    driver: Option<std::thread::JoinHandle<()>>,
    vram_plan: serde_json::Value,
    kept: EngineShape,
}

impl Harness {
    fn app(&self) -> &axum::Router {
        self.app.as_ref().expect("app is only taken by Drop")
    }

    fn plan_field(&self, name: &str) -> &serde_json::Value {
        &self.vram_plan["attributes"][name]
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

/// Load `shape`, capturing the plan event its load logs. `None` where the
/// profile allows a skip.
fn harness(shape: EngineShape, eos: u32, frontend: FrontendSet) -> Option<Harness> {
    let (loaded, events) = captured(|| qwen27b_load(Path::new(ARTIFACT), MODEL.into(), eos, shape, None, &Default::default()));
    let (scheduler, _reservations, kept, steps) = match loaded {
        Ok(loaded) => loaded,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the 27B at {:?}: {e}", shape.adaptation_start()));
            return None;
        }
    };
    let vram_plan = named(&events, "ignis.runtime.vram_plan").first().copied().cloned().expect("the load logs its plan");
    println!("vram_plan: {vram_plan}");
    // #340: the steps the load hands back are the ones its plan event logs.
    assert_eq!(vram_plan["attributes"]["adapted"], ignis_server::adaptation::steps_text(&steps).as_str());
    let (engine, driver) = Engine::with_clock_and_driver(Box::new(scheduler), Arc::new(SystemClock));
    let server = Server::with_artifact_template(engine, frontend).with_request_timeout(Duration::from_secs(300));
    Some(Harness { app: Some(server.app()), driver: Some(driver), vram_plan, kept })
}

async fn body_of(app: &axum::Router, request: Request<Body>) -> (u16, serde_json::Value) {
    let response = app.clone().oneshot(request).await.expect("route the request");
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.expect("read the body");
    let text = String::from_utf8(bytes.to_vec()).expect("the body is UTF-8");
    (status, serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}: {text}")))
}

/// The context `GET /v1/models` says the loaded model serves.
async fn max_model_len(app: &axum::Router) -> u64 {
    let request = Request::builder().uri("/v1/models").body(Body::empty()).expect("build the request");
    let (status, models) = body_of(app, request).await;
    assert_eq!(status, 200, "{models}");
    models["data"][0]["max_model_len"].as_u64().unwrap_or_else(|| panic!("{models}"))
}

/// One greedy completion of a short prompt: the load decodes.
async fn decodes(app: &axum::Router) -> String {
    let body = serde_json::json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "Name the capital of France in one word." }],
        "max_tokens": 24,
        "temperature": 0,
        "stream": false,
        "enable_thinking": false
    });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string().into_bytes()))
        .expect("build the request");
    let (status, value) = body_of(app, request).await;
    assert_eq!(status, 200, "{value}");
    let content = value["choices"][0]["message"]["content"].as_str().unwrap_or_else(|| panic!("{value}")).to_owned();
    println!("completion: {content:?}");
    assert!(content.contains("Paris"), "{value}");
    content
}

/// A budget at least 100 MB short of 917,504 tokens -- less when the card
/// has less free, an explicit budget above free memory being refused before
/// any plan: the load keeps the largest context whose pool holds a spare
/// 65,536-token reservation beside it (786,432 for a deficit up to ~613 MB:
/// 851,968 fits, without that room), says so, and serves it.
#[tokio::test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn a_budget_that_refuses_917504_loads_at_the_adapted_context_and_decodes() {
    let Some((eos, frontend)) = artifact() else { return };
    let shape = rtx5090();
    assert_eq!((shape.max_context, shape.adaptable), (917_504, Adaptable::ALL));
    let Some(needed) = needed_bytes(shape, eos) else { return };
    let Some(free) = free_bytes() else { return };
    let budget = (needed - 100_000_000).min(free - (256 << 20));
    println!("needed at 917,504: {needed} bytes; budget {budget}; free {free}");

    let Some(h) = harness(EngineShape { vram: explicit(budget), ..shape }, eos, frontend) else { return };
    let context = u64::from(h.kept.max_context);
    assert!(context < 917_504, "{context}");
    assert_eq!(
        h.plan_field("adapted"),
        format!("max_context 917504 -> {context} (adapt_context_step 65536, adapt_spare_tokens 65536)").as_str()
    );
    // Above 262,144 tokens no line follows the context, so the pool is the
    // same at the next step up, where it must have fallen short.
    let pool = h.plan_field("kv_pool_tokens").as_u64().unwrap();
    assert!(pool >= context + 65_536, "a spare reservation beside {context}: {pool}");
    assert!(context < 262_144 || pool < context + 2 * 65_536, "the largest such context: {context}, {pool}");
    assert_eq!(max_model_len(h.app()).await, context, "the context it runs, where clients read it");
    decodes(h.app()).await;
}

/// A budget where no context at eight lanes keeps a spare 65,536-token
/// reservation: the lanes drop to one -- their host retained slots to two --
/// and the ladder runs again from 917,504.
#[tokio::test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn a_budget_no_eight_lane_context_fits_drops_the_lanes_to_one_and_decodes() {
    let Some((eos, frontend)) = artifact() else { return };
    let shape = rtx5090();
    let start = shape.adaptation_start();
    let floor = shape.at(Attempt { max_context: 65_536, ..start });
    let floor_pages = 65_536 / 64 + u64::from(floor.retained_slots());
    // One page short of 65,536 + 65,536 tokens at eight lanes.
    let Some(needed) = needed_bytes(floor, eos) else { return };
    let budget = needed + (2 * 1024 - floor_pages - 1) * page_bytes();
    println!("budget {budget}; free {:?}", free_bytes());

    let Some(h) = harness(EngineShape { vram: explicit(budget), ..shape }, eos, frontend) else { return };
    let context = h.kept.max_context;
    assert_eq!(h.kept.decode_lanes_for(ModelFamily::Qwen38_27b), 1);
    assert!((65_536..917_504).contains(&context), "{context}");
    assert_eq!(
        h.plan_field("adapted"),
        format!("decode_lanes 8 -> 1, max_context 917504 -> {context} (adapt_context_step 65536, one lane)").as_str()
    );
    assert_eq!(h.plan_field("retained_host_slots"), 2, "two host retained slots a lane");
    assert_eq!(max_model_len(h.app()).await, u64::from(context));
    decodes(h.app()).await;
}

/// AC 1's leg part A left: the 27B with two lanes named, then one, inside
/// an explicit budget that holds 262,144 tokens with 256 MB to spare, loads
/// at the values named -- nothing adapts -- and decodes.
#[tokio::test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn the_27b_at_two_and_at_one_named_lane_loads_and_decodes_in_an_explicit_budget() {
    for lanes in [2, 1] {
        let Some((eos, frontend)) = artifact() else { return };
        let base = rtx5090();
        let shape = EngineShape {
            adaptable: Adaptable { decode_lanes: false, max_context: false, ..Adaptable::ALL },
            ..base.at(Attempt { max_context: 262_144, decode_lanes: lanes, ..base.adaptation_start() })
        };
        let Some(needed) = needed_bytes(shape, eos) else { return };
        let Some(h) = harness(EngineShape { vram: explicit(needed + (256 << 20)), ..shape }, eos, frontend) else { return };
        assert_eq!(h.plan_field("adapted"), "", "{lanes} lanes");
        assert_eq!(h.kept, EngineShape { vram: explicit(needed + (256 << 20)), ..shape }, "{lanes} lanes: as named");
        assert_eq!(h.plan_field("retained_host_slots"), 2 * lanes, "{lanes} lanes");
        assert_eq!(max_model_len(h.app()).await, 262_144);
        decodes(h.app()).await;
    }
}

/// AC 5 and the 16 GB profile's lanes, through the server's own load path
/// (`load::prepare_model`, `load::load_model`): the 27B under `rtx50-16gb`,
/// in an explicit budget 256 MB above its plan at the profile's values,
/// loads at 262,144 tokens on two decode lanes with nothing adapted, says
/// its drafter's head `auto` is the full head, and decodes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn the_16gb_profile_runs_two_lanes_and_says_its_draft_head_auto_is_full() {
    let Some((eos, _frontend)) = artifact() else { return };
    let profile = ["--profile", "rtx50-16gb", "--model-idle-unload", "0"];
    // The shape the load derives (no item bound: the text-only shape), built here
    // without `load::load_shape` so no draft-head line is logged outside the
    // capture below.
    let shape = EngineShape::from(&the_27b(&profile)).with_family_decode_share(ModelFamily::Qwen38_27b);
    assert_eq!((shape.max_context, shape.decode_lanes_for(ModelFamily::Qwen38_27b)), (262_144, 2));
    let Some(needed) = needed_bytes(shape, eos) else { return };
    let budget = (needed + (256 << 20)).to_string();
    println!("needed under rtx50-16gb: {needed} bytes; budget {budget}; free {:?}", free_bytes());

    let mut flags = profile.to_vec();
    flags.extend(["--vram-budget-bytes", budget.as_str()]);
    let options = ignis_server::config::Config { model: MODEL.to_owned(), model_named: true, ..the_27b(&flags) };
    let (loaded, events) = captured(|| ignis_server::load::prepare_model(&options, Path::new(ARTIFACT)).and_then(ignis_server::load::load_model));
    let loaded = match loaded {
        Ok(loaded) => loaded,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the 27B under rtx50-16gb: {e}"));
            return;
        }
    };
    let model = loaded.model;

    let heads = named(&events, "ignis.runtime.draft_head");
    assert_eq!(heads.len(), 1, "{events:?}");
    println!("draft_head: {}", heads[0]);
    assert_eq!(heads[0]["body"], "draft_head auto -> full");
    assert_eq!(heads[0]["attributes"]["draft_head"], "full");
    let plan = named(&events, "ignis.runtime.vram_plan");
    println!("vram_plan: {}", plan[0]);
    assert_eq!(plan[0]["attributes"]["adapted"], "", "the profile's values fit");
    assert_eq!(plan[0]["attributes"]["retained_host_slots"], 4, "two a lane");
    let load = named(&events, "ignis.model.loaded");
    println!("model.loaded: {}", load[0]);
    assert_eq!(load[0]["attributes"]["decode_lanes"], 2);
    assert_eq!(load[0]["attributes"]["max_context"], 262_144);
    let speculation = load[0]["attributes"]["speculation"].as_str().unwrap_or_default();
    assert!(speculation.starts_with("dflash2 draft_tokens=7 draft_head=full"), "{speculation}");

    let app = Server::from_active(model.clone()).with_request_timeout(Duration::from_secs(300)).app();
    assert_eq!(max_model_len(&app).await, 262_144);
    decodes(&app).await;
    drop(app);
    tokio::task::spawn_blocking(move || model.shut_down()).await.expect("the model shuts down");
}
