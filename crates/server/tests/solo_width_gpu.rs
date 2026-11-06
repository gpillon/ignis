//! GitHub #489 (ADR 0018 and ADR 0063 §3 as amended): `rtx-24gb` on an RTX 30
//! card resolves its `auto` solo width with nothing else adapted, and serves
//! a prompt alone at it.
//!
//! The simulation #508 ran the 24 GB class with: the 27B's 16 GB entry on the
//! RTX 5090 inside the 24 GiB budget, on the routes an 8.x card runs unset
//! (the dense prefill, the integer band and our GDN prefill, named so the 5090
//! runs them), with the card the start reads injected as an RTX 3090 so the
//! load takes the 8.x solo ladder (8,192, then 4,096): it keeps 4,096. The VRAM plan keeps the
//! profile's 1,048,576 tokens on 8 lanes with the two device retained slots,
//! nothing adapted, and its KV pool still holds the whole context; the load
//! line, `GET /v1/models` and the request log say the width (`GET /v1/config`
//! keeping `auto` is `config_http.rs`'s, on the CPU).
//!
//! Its own test binary: the request log is written on the engine's telemetry
//! task, so the capture is the process's global subscriber.
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

/// The GDN prefill's measurement switch set to `ours`, the route an 8.x card
/// runs (the leaf reads it at each load). Removed on drop.
struct OurGdnPrefill;

impl OurGdnPrefill {
    fn set() -> Self {
        // SAFETY: this binary's one test sets it before its load and nothing
        // else reads or writes the environment meanwhile.
        unsafe { std::env::set_var("IGNIS_GDN_PREFILL", "ours") };
        OurGdnPrefill
    }
}

impl Drop for OurGdnPrefill {
    fn drop(&mut self) {
        // SAFETY: as in `set`.
        unsafe { std::env::remove_var("IGNIS_GDN_PREFILL") };
    }
}

fn named<'a>(events: &'a [serde_json::Value], name: &str) -> Vec<&'a serde_json::Value> {
    events.iter().filter(|event| event["event_name"] == name).collect()
}

/// The start's config for `flags` on an RTX 3090, fitted to the 27B.
fn on_an_rtx_3090(flags: &[&str]) -> ignis_server::config::Config {
    let args: Vec<String> = flags.iter().map(|flag| flag.to_string()).collect();
    let rtx3090 = Card::Read(card::rtx_3090());
    let ConfigOutcome::Config(config) =
        ignis_server::config::resolve_on(&args, |_| None, &NoFiles, &|| rtx3090.clone()).expect("resolve")
    else {
        panic!("expected a runnable config");
    };
    config.for_family(ModelFamily::Qwen38_27b).expect("the 27B")
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn rtx_24gb_on_an_rtx_30_card_widens_its_solo_chunk_and_keeps_its_context() {
    if !Path::new(TRELLIS).exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {TRELLIS}")) {
        return;
    }
    let _switch = OurGdnPrefill::set();
    let sink = Arc::new(ignis_logging::MemorySink::new());
    tracing_subscriber::registry()
        .with(ignis_logging::JsonLayer::new(sink.clone()))
        .try_init()
        .expect("this test binary installs the only subscriber");
    let events = || -> Vec<serde_json::Value> {
        sink.lines().iter().map(|line| serde_json::from_str(line).expect("json")).collect()
    };

    let flags = [
        &["--profile", "rtx-24gb", "--vram-budget-bytes", "25769803776", "--model-idle-unload", "0", "--model-loop-guard", "off"][..],
        &ROUTES_8X[..],
    ]
    .concat();
    let options = ignis_server::config::Config { model: MODEL.to_owned(), model_named: true, ..on_an_rtx_3090(&flags) };
    assert_eq!(options.prefill_chunk, 2048);
    assert_eq!(options.prefill_chunk_solo, ignis_server::config::SoloWidth::Auto, "no profile names one");
    let slots = ignis_server::config::profile::RTX_24GB_RETAINED_DEVICE_SLOTS;
    let loaded = match ignis_server::load::prepare_model(&options, Path::new(TRELLIS)).and_then(ignis_server::load::load_model)
    {
        Ok(loaded) => loaded,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the 16 GB artifact under rtx-24gb in 24 GiB: {e}"));
            return;
        }
    };
    let model = loaded.model;
    let logged = events();

    let plan = named(&logged, "ignis.runtime.vram_plan");
    assert_eq!(plan.len(), 1, "{logged:?}");
    println!("vram_plan: {}", plan[0]);
    let lines = &plan[0]["attributes"];
    assert_eq!(lines["budget_bytes"], 25_769_803_776u64);
    assert_eq!(lines["adapted"], "", "nothing adapted for the solo width");
    assert_eq!(lines["retained_device_slots"], slots, "beside the device retained slots");
    let pool = lines["kv_pool_tokens"].as_u64().unwrap();
    let solo = named(&logged, "ignis.runtime.prefill_chunk_solo");
    assert_eq!(solo.len(), 1, "{logged:?}");
    println!("prefill_chunk_solo: {}", solo[0]);
    let width = solo[0]["attributes"]["width"].as_u64().unwrap();
    println!(
        "solo width {width}: kv pool {pool} tokens, {} past the context, {} fewer than at the chunk; workspace {}",
        pool.saturating_sub(1_048_576),
        POOL_AT_THE_CHUNK.saturating_sub(pool),
        lines["workspace_bytes"]
    );
    // Measured 2026-11-06: 8,192 does not fit with nothing adapted, 4,096 does,
    // and keeps a pool of 17,601 pages (1,126,464 tokens), 77,888 past the
    // context, 13,312 fewer than at the chunk.
    assert_eq!(width, 4096, "the widest of the 8.x ladder the plan holds as the profile asks");
    assert_eq!(solo[0]["body"], format!("prefill_chunk_solo auto -> {width}"));
    assert!(pool >= 1_048_576 + u64::from(slots + 16) * 64, "the whole context and the slots' tail pages: {pool}");
    let load = named(&logged, "ignis.model.loaded");
    assert_eq!(load[0]["attributes"]["max_context"], 1_048_576, "#474's context");
    assert_eq!(load[0]["attributes"]["decode_lanes"], 8);
    assert_eq!(load[0]["attributes"]["prefill_chunk"], 2048);
    assert_eq!(load[0]["attributes"]["prefill_chunk_solo"], width);

    let app = Server::from_active(model.clone()).with_request_timeout(Duration::from_secs(600)).app();
    let models = get(&app, "/v1/models").await;
    assert_eq!(models["data"][0]["prefill_chunk_solo"], width, "{models}");

    // A prompt alone, past one solo chunk: every chunk at the solo width.
    let line = "The harbour master records each ship, its cargo and the hour it docked. ";
    let prompt = format!("{}\nHow many times does the word harbour appear above? Answer with a number.", line.repeat(700));
    let answer = complete(&app, &prompt, 8).await;
    println!("answer: {answer:?}");
    assert!(!answer.trim().is_empty());
    tokio::time::sleep(Duration::from_millis(200)).await;
    let admitted = named(&events(), "ignis.request.admitted").last().map(|event| event["attributes"].clone()).expect("admitted");
    println!("admitted: {admitted}");
    let consumed = admitted["prefill_chunks_consumed"].as_u64().unwrap();
    assert!(consumed >= 2, "past one solo chunk: {admitted}");
    assert_eq!(admitted["prefill_chunks_solo"].as_u64(), Some(consumed), "nothing decoded beside it: {admitted}");
    tokio::task::spawn_blocking(move || model.shut_down()).await.expect("the model shuts down");
}
