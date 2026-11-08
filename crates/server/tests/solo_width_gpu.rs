//! GitHub #489 (ADR 0018 and ADR 0063 §3 as amended): the solo width on the
//! two built-in profiles of the RTX 30 and RTX 40 classes, as a card of theirs
//! loads them.
//!
//! The simulation #508 ran the 24 GB class with: the 27B's 16 GB entry on the
//! RTX 5090 inside the class's budget, on the routes an 8.x card runs unset
//! (the dense prefill and the integer band, named so the 5090 runs them, and
//! our GDN prefill, every card's since GitHub #536), with the card the start
//! reads injected as an 8.x card so the
//! load takes the 8.x solo ladder (8,192, 4,096, 2,048 above the chunk).
//! - `rtx-24gb` (an RTX 3090, 24 GiB) and `rtx-16gb` (an RTX 4080, 16 GiB less
//!   1 GiB) pin the 27B's solo width to their chunk: one width, the KV pool
//!   their profile notes measured.
//! - `--model-prefill-chunk-solo auto` on the same loads runs the ladder: a
//!   width is kept only with nothing adapted and the spare reservation left
//!   beside the context. `rtx-24gb` keeps 4,096 and serves a prompt alone at
//!   it; `rtx-16gb` steps down to 2,048.
//! The load line, `GET /v1/models` and the request log say the width
//! (`GET /v1/config` keeping the setting is `config_http.rs`'s, on the CPU).
//!
//! Its own test binary: the request log is written on the engine's telemetry
//! task, so the capture is the process's global subscriber, installed once
//! for every test, each reading what its own load logged.
//!
//! Explicit GPU profile (ADR 0006): outside `IGNIS_GPU_PROFILE=1` a missing
//! artifact or GPU is a skip; under it, a failure.

#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, to_bytes};
use axum::http::Request;
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use ignis_core::compute::ModelFamily;
use ignis_core::gpu_profile;
use ignis_server::Server;
use ignis_server::config::ConfigOutcome;
use ignis_server::config::file::NoFiles;
use ignis_server::config::profile::Card;

#[path = "support/card.rs"]
mod card;

const TRELLIS: &str = "F:/ai/models/Qwen3.8-27B-trellis-3bpw-ignis/qwen3_8_27b_trellis_sc30-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";
/// The routes an 8.x card runs unset, named so the 5090 runs them.
const ROUTES_8X: [&str; 4] = ["--model-prefill-linear", "int8", "--model-prefill-attention", "int8"];
/// #508's pool on these routes at the 2,048-token chunk: 17,809 pages,
/// 91,200 tokens past the context (`config::profile`'s `rtx-24gb` note).
const POOL_AT_THE_CHUNK: u64 = 1_139_776;
/// `rtx-16gb`'s pool on these routes at its 1,024-token chunk, 16 GiB less
/// 1 GiB (`config::profile`'s `rtx-16gb` note).
const POOL_16GB_AT_THE_CHUNK: u64 = 386_048;

/// The process's one subscriber, installed by the first test that asks.
fn sink() -> &'static Arc<ignis_logging::MemorySink> {
    static SINK: std::sync::OnceLock<Arc<ignis_logging::MemorySink>> = std::sync::OnceLock::new();
    SINK.get_or_init(|| {
        let sink = Arc::new(ignis_logging::MemorySink::new());
        tracing_subscriber::registry()
            .with(ignis_logging::JsonLayer::new(sink.clone()))
            .try_init()
            .expect("this test binary installs the only subscriber");
        sink
    })
}

/// Every event logged since the sink held `start` lines.
fn events_since(start: usize) -> Vec<serde_json::Value> {
    sink().lines()[start..].iter().map(|line| serde_json::from_str(line).expect("json")).collect()
}

fn named<'a>(events: &'a [serde_json::Value], name: &str) -> Vec<&'a serde_json::Value> {
    events.iter().filter(|event| event["event_name"] == name).collect()
}

/// The start's config for `flags` on `card`, fitted to the 27B, served as the
/// default id.
fn on_card(card: Card, flags: &[&str]) -> ignis_server::config::Config {
    let args: Vec<String> = flags.iter().map(|flag| flag.to_string()).collect();
    let ConfigOutcome::Config(config) =
        ignis_server::config::resolve_on(&args, |_| None, &NoFiles, &|| card.clone()).expect("resolve")
    else {
        panic!("expected a runnable config");
    };
    let config = config.for_family(ModelFamily::Qwen38_27b).expect("the 27B");
    ignis_server::config::Config { model: MODEL.to_owned(), model_named: true, ..config }
}

/// The 16 GB entry loaded through the server's own path under `options`, and
/// the sink's length before it; `None` where the profile allows a skip.
fn load(options: &ignis_server::config::Config, what: &str) -> Option<(ignis_server::ActiveModel, usize)> {
    if !Path::new(TRELLIS).exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {TRELLIS}")) {
        return None;
    }
    let start = sink().lines().len();
    match ignis_server::load::prepare_model(options, Path::new(TRELLIS)).and_then(ignis_server::load::load_model) {
        Ok(loaded) => Some((loaded.model, start)),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the 16 GB artifact {what}: {e}"));
            None
        }
    }
}

async fn get(app: &axum::Router, uri: &str) -> serde_json::Value {
    let request = Request::builder().method("GET").uri(uri).body(Body::empty()).expect("build the request");
    let response = app.clone().oneshot(request).await.expect("route the request");
    assert_eq!(response.status().as_u16(), 200);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.expect("read the body");
    serde_json::from_slice(&bytes).expect("json")
}

async fn complete(app: &axum::Router, prompt: &str, max_tokens: u32) -> String {
    let body = serde_json::json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": prompt }],
        "max_tokens": max_tokens,
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
    let response = app.clone().oneshot(request).await.expect("route the request");
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.expect("read the body");
    let value: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    assert_eq!(status, 200, "{value}");
    value["choices"][0]["message"]["content"].as_str().unwrap_or_else(|| panic!("{value}")).to_owned()
}

/// The start flags of `profile` inside `budget` bytes on the 8.x routes, and
/// `extra`.
fn flags_of<'a>(profile: &'a str, budget: &'a str, extra: &[&'a str]) -> Vec<&'a str> {
    [
        &["--profile", profile, "--vram-budget-bytes", budget, "--model-idle-unload", "0", "--model-loop-guard", "off"][..],
        &ROUTES_8X[..],
        extra,
    ]
    .concat()
}

/// One load's facts: the plan's `adapted`, its KV pool, the width the solo
/// line says and its text, and the `ignis.model.loaded` line.
struct Loaded {
    adapted: String,
    pool: u64,
    width: u64,
    text: String,
    loaded: serde_json::Value,
    retained_device_slots: serde_json::Value,
}

fn facts(logged: &[serde_json::Value]) -> Loaded {
    let plan = named(logged, "ignis.runtime.vram_plan");
    assert_eq!(plan.len(), 1, "{logged:?}");
    println!("vram_plan: {}", plan[0]);
    let lines = &plan[0]["attributes"];
    let solo = named(logged, "ignis.runtime.prefill_chunk_solo");
    assert_eq!(solo.len(), 1, "{logged:?}");
    println!("prefill_chunk_solo: {}", solo[0]);
    let load = named(logged, "ignis.model.loaded");
    assert_eq!(load.len(), 1, "{logged:?}");
    Loaded {
        adapted: lines["adapted"].as_str().unwrap().to_owned(),
        pool: lines["kv_pool_tokens"].as_u64().unwrap(),
        width: solo[0]["attributes"]["width"].as_u64().unwrap(),
        text: solo[0]["body"].as_str().unwrap().to_owned(),
        loaded: load[0]["attributes"].clone(),
        retained_device_slots: lines["retained_device_slots"].clone(),
    }
}

const BUDGET_24GB: &str = "25769803776";

fn budget_16gb() -> String {
    (17_179_869_184u64 - (1 << 30)).to_string()
}

/// `rtx-24gb` on an RTX 3090 pins the 27B's solo width to its 2,048-token
/// chunk: the plan is #508's, its pool 91,200 tokens past the context.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn rtx_24gb_pins_its_solo_width_to_its_chunk() {
    let options = on_card(Card::Read(card::rtx_3090()), &flags_of("rtx-24gb", BUDGET_24GB, &[]));
    assert_eq!(options.prefill_chunk_solo, ignis_server::config::SoloWidth::Tokens(2048), "the profile's");
    let Some((model, start)) = load(&options, "under rtx-24gb in 24 GiB") else { return };
    let loaded = facts(&events_since(start));
    assert_eq!(loaded.adapted, "");
    assert_eq!((loaded.width, loaded.text.as_str()), (2048, "prefill_chunk_solo 2048"));
    assert_eq!(loaded.pool, POOL_AT_THE_CHUNK, "#508's pool");
    assert_eq!(loaded.loaded["max_context"], 1_048_576);
    tokio::task::spawn_blocking(move || model.shut_down()).await.expect("the model shuts down");
}

/// `auto` on the same load keeps 4,096 (8,192 does not fit, 4,096 leaves
/// 77,888 tokens past the context, more than the 65,536-token spare
/// reservation), and a prompt alone runs every chunk at it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn rtx_24gb_on_an_rtx_30_card_widens_its_solo_chunk_under_auto() {
    let options = on_card(Card::Read(card::rtx_3090()), &flags_of("rtx-24gb", BUDGET_24GB, &["--model-prefill-chunk-solo", "auto"]));
    assert_eq!(options.prefill_chunk, 2048);
    assert_eq!(options.prefill_chunk_solo, ignis_server::config::SoloWidth::Auto);
    let slots = ignis_server::config::profile::RTX_24GB_RETAINED_DEVICE_SLOTS;
    let Some((model, start)) = load(&options, "under rtx-24gb in 24 GiB, solo auto") else { return };
    let events = || events_since(start);
    let loaded = facts(&events());
    println!(
        "solo width {}: kv pool {} tokens, {} past the context, {} fewer than at the chunk",
        loaded.width,
        loaded.pool,
        loaded.pool.saturating_sub(1_048_576),
        POOL_AT_THE_CHUNK.saturating_sub(loaded.pool)
    );
    assert_eq!(loaded.adapted, "", "nothing adapted for the solo width");
    assert_eq!(loaded.retained_device_slots, slots, "beside the device retained slots");
    // Measured 2026-11-06: 8,192 does not fit, 4,096 keeps a pool of 17,601
    // pages (1,126,464 tokens), 77,888 past the context.
    assert_eq!(loaded.width, 4096, "the widest of the 8.x ladder that keeps the spare reservation");
    assert_eq!(loaded.text, "prefill_chunk_solo auto -> 4096");
    assert!(loaded.pool >= 1_048_576 + 65_536, "the context and the spare reservation: {}", loaded.pool);
    assert_eq!(loaded.loaded["max_context"], 1_048_576, "#474's context");
    assert_eq!(loaded.loaded["decode_lanes"], 8);
    assert_eq!(loaded.loaded["prefill_chunk_solo"], 4096);

    let app = Server::from_active(model.clone()).with_request_timeout(Duration::from_secs(600)).app();
    let models = get(&app, "/v1/models").await;
    assert_eq!(models["data"][0]["prefill_chunk_solo"], 4096, "{models}");

    // A prompt alone, past one solo chunk: every chunk at the solo width.
    let line = "The harbour master records each ship, its cargo and the hour it docked. ";
    let prompt = format!("{}\nHow many times does the word harbour appear above? Answer with a number.", line.repeat(700));
    let answer = complete(&app, &prompt, 8).await;
    println!("answer: {answer:?}");
    assert!(!answer.trim().is_empty());
    // The request log is written on the engine's telemetry task: poll until
    // the `admitted` line carries both counts this test reads.
    let mut admitted = serde_json::Value::Null;
    for _ in 0..100 {
        if let Some(line) = named(&events(), "ignis.request.admitted").last().map(|event| event["attributes"].clone())
            && line["prefill_chunks_consumed"].is_u64()
            && line["prefill_chunks_solo"].is_u64()
        {
            admitted = line;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    println!("admitted: {admitted}");
    let consumed = admitted["prefill_chunks_consumed"].as_u64().unwrap();
    assert!(consumed >= 2, "past one solo chunk: {admitted}");
    assert_eq!(admitted["prefill_chunks_solo"].as_u64(), Some(consumed), "nothing decoded beside it: {admitted}");
    tokio::task::spawn_blocking(move || model.shut_down()).await.expect("the model shuts down");
}

/// `rtx-16gb` on an RTX 4080 pins the 27B's solo width to its 1,024-token
/// chunk: the pool its 8.x note measured.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn rtx_16gb_pins_its_solo_width_to_its_chunk() {
    let budget = budget_16gb();
    let options = on_card(Card::Read(card::card(16, "NVIDIA GeForce RTX 4080", 8, 9)), &flags_of("rtx-16gb", &budget, &[]));
    assert_eq!(options.prefill_chunk_solo, ignis_server::config::SoloWidth::Tokens(1024), "the profile's");
    let Some((model, start)) = load(&options, "under rtx-16gb in 16 GiB less 1 GiB") else { return };
    let loaded = facts(&events_since(start));
    assert_eq!(loaded.adapted, "");
    assert_eq!((loaded.width, loaded.text.as_str()), (1024, "prefill_chunk_solo 1024"));
    assert_eq!(loaded.pool, POOL_16GB_AT_THE_CHUNK, "the profile note's pool");
    assert_eq!((&loaded.loaded["max_context"], &loaded.loaded["decode_lanes"]), (&262_144.into(), &2.into()));
    tokio::task::spawn_blocking(move || model.shut_down()).await.expect("the model shuts down");
}

/// `auto` on the same load: 4,096 would leave less than the spare
/// reservation past the context (40,768 tokens), so the ladder steps down to
/// 2,048.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn rtx_16gb_on_an_rtx_40_card_steps_down_to_2048_under_auto() {
    let budget = budget_16gb();
    let options =
        on_card(Card::Read(card::card(16, "NVIDIA GeForce RTX 4080", 8, 9)), &flags_of("rtx-16gb", &budget, &["--model-prefill-chunk-solo", "auto"]));
    assert_eq!(options.prefill_chunk, 1024);
    let Some((model, start)) = load(&options, "under rtx-16gb in 16 GiB less 1 GiB, solo auto") else { return };
    let loaded = facts(&events_since(start));
    println!(
        "solo width {}: kv pool {} tokens, {} past the context, {} fewer than at the chunk",
        loaded.width,
        loaded.pool,
        loaded.pool.saturating_sub(262_144),
        POOL_16GB_AT_THE_CHUNK.saturating_sub(loaded.pool)
    );
    assert_eq!(loaded.adapted, "", "nothing adapted for the solo width");
    assert_eq!(loaded.width, 2048, "4,096 short of the spare reservation, 2,048 not");
    assert!(loaded.pool >= 262_144 + 65_536, "the context and the spare reservation: {}", loaded.pool);
    assert_eq!((&loaded.loaded["max_context"], &loaded.loaded["decode_lanes"]), (&262_144.into(), &2.into()));
    tokio::task::spawn_blocking(move || model.shut_down()).await.expect("the model shuts down");
}
