//! Spec #630 (GitHub #637): the 12 GB artifact served by ignis through the
//! server's own load path (`load::prepare_model`, `load::load_model`), on the
//! RTX 5090 inside the 12 GiB budget a 12 GB card has:
//! `--profile rtx-12gb --vram-budget-bytes 12884901888`.
//!
//! In 12 GiB the plan is the load's of #635 (2026-11-16, on `ig28a`, whose
//! objects the admitted artifact carries), to the byte: every line but the
//! pool and the card's own is pinned, the pool is the whole pages they leave
//! beside the card's lines, the CUDA context the process measured and the
//! trellis workspace, pinned on the RTX 5090 too (GitHub #607,
//! `support/kv_pool_pins.rs`), and 131,072 tokens load on one lane drafting
//! with MTP at 7, nothing adapted. The same load serves greedy coding
//! completions that are coherent (they hold what the prompt asks for) and
//! deterministic (the same prompt twice gives the same text), its rounds
//! drafting.
//!
//! Less 1 GiB of simulated desktop, 131,072 tokens miss the plan's own floor
//! (one sequence, the lane's two retained slots' pages and the three the MTP
//! head's holders keep back) and the ladder takes the profile's 32,768-token
//! step to 98,304, which the plan's `adapted` says; every line but the pool's
//! block tables is the 12 GiB load's.
//!
//! On the routes an RTX 30 or 40 card runs unset (the dense prefill and the
//! integer band, named so the RTX 5090 runs them, and our GDN prefill, every
//! card's), the dense prefill's 89 MB buffer is the one line more, as the CPU
//! pins derive it: 131,072 tokens still load in 12 GiB, and less 1 GiB the
//! ladder goes to 65,536.
//!
//! The artifact is the admitted one, by its path in the models folder.
//!
//! Its own test binary: the request log is written on the engine's telemetry
//! task, so the serving test's capture is the process's global subscriber.
//! The budget cases read only the load's own lines, logged on the loading
//! thread, so they capture with a scoped subscriber instead.
//!
//! ```text
//! cargo test -p ignis-server --features cuda --test trellis_12g_gpu -- --ignored --nocapture --test-threads=1
//! ```
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
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use ignis_artifact::{FrontendSet, Reader};
use ignis_core::compute::ModelFamily;
use ignis_core::gpu_profile;
use ignis_core::{VramMode, VramPlanError};
use ignis_server::Server;
use ignis_server::adaptation::{Adaptable, StartRefusal};
use ignis_server::runtime::{EngineShape, Qwen27bLoadError, qwen27b_load};

#[path = "support/kv_pool_pins.rs"]
mod kv_pool_pins;

use kv_pool_pins::{
    CardLines, RTX_12GB, RTX_12GB_8X, RTX_12GB_8X_LESS_1_GIB, RTX_12GB_8X_LESS_1_GIB_ADAPTED, RTX_12GB_BUDGET,
    RTX_12GB_LESS_1_GIB, RTX_12GB_LESS_1_GIB_131072, RTX_12GB_LESS_1_GIB_ADAPTED, TRELLIS_DENSE_BYTES, assert_pinned,
};

/// The admitted 12 GB artifact, in the models folder.
const ARTIFACT: &str = "F:/ai/q38/ninfer-models/qwen3_8_27b_trellis_ig28-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";
/// A 12 GB card less 1 GiB of simulated desktop.
const BUDGET_LESS_1_GIB: u64 = RTX_12GB_BUDGET - (1 << 30);
/// The prefill routes an RTX 30 or 40 card runs unset (spec #503, GitHub
/// #515), named so the RTX 5090 runs them: the dense prefill and the integer
/// band.
const ROUTES_8X: [&str; 4] = ["--model-prefill-linear", "int8", "--model-prefill-attention", "int8"];

/// The lines of the 12 GiB load's plan that are neither the pool nor the
/// card's, as #635's load logged them: they are the same in every budget and
/// at every context here.
const LINES: [(&str, u64); 8] = [
    ("weights_bytes", 9_651_141_632),
    ("workspace_bytes", 302_433_536),
    ("sampling_bytes", 579_048),
    ("decode_graph_bytes", 12_802_060),
    ("verify_round_bytes", 19_017_436),
    ("lane_state_bytes", 154_947_584),
    ("hq_residual_window_bytes", 37_879_936),
    ("residual_bytes", 124_474_800),
];

/// A 12 GB card's start inside `budget` bytes, as #635 loaded it.
fn flags_12g(budget: &str) -> [&str; 8] {
    ["--profile", "rtx-12gb", "--vram-budget-bytes", budget, "--model-idle-unload", "0", "--model-loop-guard", "off"]
}

fn the_27b(flags: &[&str]) -> ignis_server::config::Config {
    let args: Vec<String> = flags.iter().map(|flag| flag.to_string()).collect();
    let ignis_server::config::ConfigOutcome::Config(config) =
        ignis_server::config::resolve(&args, |_| None).expect("resolve")
    else {
        panic!("expected a runnable config");
    };
    config.for_family(ModelFamily::Qwen38_27b).expect("the 27B")
}

fn named<'a>(events: &'a [serde_json::Value], name: &str) -> Vec<&'a serde_json::Value> {
    events.iter().filter(|event| event["event_name"] == name).collect()
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

/// Whether the artifact is there, or the profile allows a skip.
fn artifact_present() -> bool {
    Path::new(ARTIFACT).exists() || !gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}"))
}

/// The 12 GB artifact through the server's own load path inside `budget`
/// bytes with `extra` flags after the profile's, and the events the load
/// logged; `None` where the profile allows a skip.
fn load_12g(budget: u64, extra: &[&str]) -> Option<(ignis_server::ActiveModel, Vec<serde_json::Value>)> {
    if !artifact_present() {
        return None;
    }
    let budget = budget.to_string();
    let flags = [&flags_12g(&budget)[..], extra].concat();
    let options = ignis_server::config::Config { model: MODEL.to_owned(), model_named: true, ..the_27b(&flags) };
    let (loaded, events) = captured(|| {
        ignis_server::load::prepare_model(&options, Path::new(ARTIFACT)).and_then(ignis_server::load::load_model)
    });
    match loaded {
        Ok(loaded) => Some((loaded.model, events)),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the 12 GB artifact in {budget} bytes: {e}"));
            None
        }
    }
}

/// The plan's lines of a load's `events`, its one `ignis.runtime.vram_plan`.
fn plan_lines(events: &[serde_json::Value]) -> serde_json::Value {
    let plan = named(events, "ignis.runtime.vram_plan");
    assert_eq!(plan.len(), 1, "{events:?}");
    println!("vram_plan: {}", plan[0]);
    plan[0]["attributes"].clone()
}

/// The load's `ignis.model.loaded` line.
fn loaded_line(events: &[serde_json::Value]) -> serde_json::Value {
    let load = named(events, "ignis.model.loaded");
    assert_eq!(load.len(), 1, "{events:?}");
    println!("model.loaded: {}", load[0]);
    load[0]["attributes"].clone()
}

/// The plan `lines` holds every pinned line beside the pool and the card's,
/// `trellis_dense` the dense buffer's, and the MTP head's holders' pages.
fn assert_the_lines(lines: &serde_json::Value, trellis_dense: u64) {
    for (name, bytes) in LINES {
        assert_eq!(lines[name].as_u64(), Some(bytes), "the plan's {name}");
    }
    assert_eq!(lines["trellis_dense_bytes"].as_u64(), Some(trellis_dense), "the dense buffer's line");
    assert_eq!((lines["drafter_round_bytes"].as_u64(), lines["media_embedding_bytes"].as_u64()), (Some(0), Some(0)));
    assert_eq!(lines["kv_holder_pages"].as_u64(), Some(3), "a page for the lane and one for each of its two retained slots");
    assert_eq!(lines["vision_residency"], "host");
}

/// What the 12 GB artifact's plan needs at `shape`'s own values, nothing
/// adapting -- its lines and a KV pool of the floor: its refusal at a 1 GiB
/// budget carries it (`VramPlanError::BelowMinimum`'s `needed_bytes`), before
/// anything is materialized. `None` where the profile allows a skip.
fn needed_bytes(shape: EngineShape, eos: u32) -> Option<u64> {
    let refused = EngineShape {
        vram: VramMode::Explicit { budget_bytes: 1 << 30, allow_oversubscription: false },
        adaptable: Adaptable::NONE,
        ..shape
    };
    match qwen27b_load(Path::new(ARTIFACT), MODEL.into(), eos, refused, None, &Default::default()) {
        Ok(_) => panic!("a 1 GiB budget holds no 27B"),
        Err(Qwen27bLoadError::VramPlan(StartRefusal { error: VramPlanError::BelowMinimum { needed_bytes, .. }, tried })) => {
            assert!(tried.is_empty(), "nothing adapts: {tried:?}");
            Some(needed_bytes)
        }
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("plan the 12 GB artifact at {:?}: {e}", shape.adaptation_start()));
            None
        }
    }
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
    let text = String::from_utf8(bytes.to_vec()).expect("the body is UTF-8");
    let value: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}: {text}"));
    assert_eq!(status, 200, "{value}");
    value["choices"][0]["message"]["content"].as_str().unwrap_or_else(|| panic!("{value}")).to_owned()
}

/// In 12 GiB: the plan's lines to the byte, 131,072 tokens on one lane
/// drafting with MTP, nothing adapted; then greedy code, each prompt twice.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn the_12gb_artifact_plans_its_lines_to_the_byte_in_12_gib_and_serves_greedy_code_under_mtp() {
    if !artifact_present() {
        return;
    }
    let sink = Arc::new(ignis_logging::MemorySink::new());
    tracing_subscriber::registry()
        .with(ignis_logging::JsonLayer::new(sink.clone()))
        .try_init()
        .expect("this test binary installs the only subscriber");
    let events = || -> Vec<serde_json::Value> {
        sink.lines().iter().map(|line| serde_json::from_str(line).expect("json")).collect()
    };

    let budget = RTX_12GB_BUDGET.to_string();
    let options = ignis_server::config::Config { model: MODEL.to_owned(), model_named: true, ..the_27b(&flags_12g(&budget)) };
    let loaded = match ignis_server::load::prepare_model(&options, Path::new(ARTIFACT)).and_then(ignis_server::load::load_model)
    {
        Ok(loaded) => loaded,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the 12 GB artifact under rtx-12gb: {e}"));
            return;
        }
    };
    let model = loaded.model;
    let logged = events();
    let lines = plan_lines(&logged);
    assert_eq!(lines["budget_bytes"], RTX_12GB_BUDGET);
    assert_eq!(lines["mode"], "explicit");
    assert_eq!(lines["adapted"], "", "the 12 GiB load adapted its shape");
    assert_the_lines(&lines, 0);
    assert_pinned(&lines, RTX_12GB, "#635's load in 12 GiB");
    let load = loaded_line(&logged);
    assert_eq!((load["decode_lanes"].as_u64(), load["max_context"].as_u64()), (Some(1), Some(131_072)));
    assert_eq!((load["prefill_chunk"].as_u64(), load["prefill_chunk_solo"].as_u64()), (Some(512), Some(512)));
    assert_eq!(load["speculation"], "mtp draft_tokens=7");

    let app = Server::from_active(model.clone()).with_request_timeout(Duration::from_secs(600)).app();
    let prompts = [
        (
            "Write a Python function `is_prime(n)` that returns True when n is a prime number. Reply with the code only.",
            "def is_prime",
        ),
        ("Write a Rust function `fn gcd(a: u64, b: u64) -> u64` using Euclid's algorithm. Reply with the code only.", "fn gcd"),
    ];
    for (prompt, needle) in prompts {
        let first = complete(&app, prompt, 160).await;
        let again = complete(&app, prompt, 160).await;
        println!("completion: {first:?}");
        assert!(first.contains(needle), "the completion holds `{needle}`: {first:?}");
        assert_eq!(first, again, "greedy decoding is deterministic");
    }
    // The request log is written on the telemetry task after each response.
    let mut done = Vec::new();
    for _ in 0..100 {
        done = events().into_iter().filter(|e| e["event_name"] == "ignis.request.done").collect::<Vec<_>>();
        if done.len() >= 2 * prompts.len() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(done.len(), 2 * prompts.len(), "every request logs its done line");
    let (mut rounds, mut accepted) = (0u64, 0u64);
    for line in &done {
        let a = &line["attributes"];
        println!(
            "request {}: tokens {}, {:.1} tok/s, spec.rounds {}, drafted {}, accepted {}",
            a["request_id"], a["tokens"], a["tok_s"].as_f64().unwrap_or(0.0), a["spec.rounds"], a["spec.drafted"], a["spec.accepted"]
        );
        rounds += a["spec.rounds"].as_u64().expect("an MTP load's request runs verify rounds");
        accepted += a["spec.accepted"].as_u64().unwrap_or(0);
    }
    // A round commits its accepted drafts and the target's own token.
    let per_round = (accepted + rounds) as f64 / rounds as f64;
    println!("tokens per round: {per_round:.2} over {rounds} rounds");
    assert!(per_round > 1.0, "the head's drafts are accepted at all: {per_round}");

    drop(app);
    tokio::task::spawn_blocking(move || model.shut_down()).await.expect("the model shuts down");
}

/// Less 1 GiB of simulated desktop: 131,072 tokens miss the plan's floor by
/// 420 pages at the context the RTX 5090 measured, and the load adapts to
/// 98,304 on one lane, its plan saying so, on a pool of 1,633 pages.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn less_1_gib_of_desktop_adapts_the_context_to_98304_and_the_plan_says_so() {
    if !artifact_present() {
        return;
    }
    let reader = Reader::open(Path::new(ARTIFACT)).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend.eos_token_id().expect("qwen3.8-27b generation config must carry eos_token_id");
    let budget = BUDGET_LESS_1_GIB.to_string();
    // The shape the load derives, built here without `load::load_shape` so
    // the plan below logs no draft-head line.
    let shape = EngineShape::from(&the_27b(&flags_12g(&budget))).with_family_decode_share(ModelFamily::Qwen38_27b);
    let start = shape.adaptation_start();
    assert_eq!((start.max_context, start.decode_lanes), (131_072, 1));
    let Some(floor_131072) = needed_bytes(shape, eos) else { return };
    // The refusal charged the context this process made.
    let reading = ignis_artifact::CudaDevice::context_reading().expect("the refusal made the context");
    let refused_context = ignis_runtime::ContextCharge::of(reading).bytes;

    let Some((model, events)) = load_12g(BUDGET_LESS_1_GIB, &[]) else { return };
    let lines = plan_lines(&events);
    assert_eq!(lines["budget_bytes"], BUDGET_LESS_1_GIB);
    let card = CardLines::of(&lines);
    assert_eq!(card.cuda_context, refused_context, "the load charges the context the refusal charged");
    println!("131,072 tokens need {floor_131072} bytes in a budget of {BUDGET_LESS_1_GIB}");
    assert_eq!(
        floor_131072 - card.cuda_context - card.trellis_workspace,
        RTX_12GB_LESS_1_GIB_131072.needed_bytes,
        "131,072 at its floor: the bytes beside the card's lines"
    );
    assert_eq!(lines["adapted"], RTX_12GB_LESS_1_GIB_ADAPTED);
    assert_the_lines(&lines, 0);
    assert_pinned(&lines, RTX_12GB_LESS_1_GIB, "the 12 GiB load's lines at 98,304 less 1 GiB");
    let load = loaded_line(&events);
    assert_eq!((load["decode_lanes"].as_u64(), load["max_context"].as_u64()), (Some(1), Some(98_304)));
    tokio::task::spawn_blocking(move || model.shut_down()).await.expect("the model shuts down");
}

/// The routes an RTX 30 or 40 card runs, named on the RTX 5090: the dense
/// buffer's line beside the 12 GiB load's others, the workspace unchanged, and
/// the pool the CPU pins derive -- 131,072 tokens in 12 GiB with nothing
/// adapted, 65,536 less 1 GiB. Each pool is printed beside Blackwell's.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn on_the_8x_routes_the_12gb_load_keeps_131072_in_12_gib_and_adapts_to_65536_less_1_gib() {
    for (budget, pinned, blackwell, adapted, context) in [
        (RTX_12GB_BUDGET, RTX_12GB_8X, RTX_12GB, "", 131_072),
        (BUDGET_LESS_1_GIB, RTX_12GB_8X_LESS_1_GIB, RTX_12GB_LESS_1_GIB, RTX_12GB_8X_LESS_1_GIB_ADAPTED, 65_536),
    ] {
        let Some((model, events)) = load_12g(budget, &ROUTES_8X) else { return };
        let routes = named(&events, "ignis.runtime.prefill_routes");
        assert_eq!(routes.len(), 1, "{events:?}");
        let routes = &routes[0]["attributes"];
        assert_eq!(
            (&routes["prefill_linear"], &routes["prefill_attention"], &routes["gdn_prefill"]),
            (&"int8".into(), &"int8".into(), &"ours".into()),
            "{routes}"
        );
        let lines = plan_lines(&events);
        assert_eq!(lines["budget_bytes"], budget);
        assert_eq!(lines["adapted"], adapted);
        assert_the_lines(&lines, TRELLIS_DENSE_BYTES);
        let card = CardLines::of(&lines);
        println!(
            "kv pool in {budget} bytes: {} tokens on the 8.x routes, {} on Blackwell's",
            lines["kv_pool_tokens"],
            blackwell.pool_tokens(card)
        );
        assert_pinned(&lines, pinned, "the 12 GB load on the 8.x routes");
        let load = loaded_line(&events);
        assert_eq!((load["decode_lanes"].as_u64(), load["max_context"].as_u64()), (Some(1), Some(context)));
        tokio::task::spawn_blocking(move || model.shut_down()).await.expect("the model shuts down");
    }
}
