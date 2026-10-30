//! Spec 27b-16gb/05a (GitHub #339): the 16 GB artifact served end to end by
//! ignis, through the server's own load path (`load::prepare_model`,
//! `load::load_model`), on the 5090 inside the 16 GiB budget a 16 GB card
//! has: `--profile rtx50-16gb --vram-budget-bytes 17179869184`.
//!
//! The load binds the trellis body by its head (no flag names it), plans the
//! trellis workspace as its own VRAM line, runs the profile's two decode
//! lanes with DFlash2, and serves greedy coding completions that are coherent
//! (they hold what the prompt asks for) and deterministic (the same prompt
//! twice gives the same text) -- one at a time, and the two prompts at once,
//! each then its own sequential text: the two requests in flight together
//! (both decoding before either is done, read off the request log) are what
//! runs the verify rounds two lanes wide. Each request's speculative rounds
//! are read off its `ignis.request.done` line, and the tokens per round
//! printed beside the plan's lines -- the figures Acceptance A1 and A2 report.
//!
//! The plan's three lines the trellis body changes are pinned to the bytes
//! spec 05a's As built derives them from (its "The plan's trellis lines"):
//! `workspace`, `trellis_workspace` and `decode_graph`.
//!
//! Spec 05b's capacity simulation (GitHub #340, AC 5) adds the card less 1 GiB
//! and less 3 GiB of simulated desktop, load and plan only. No line but the
//! pool follows the budget or, on this body, the context (the hq prompt
//! planes are gone, ADR 0061), so each pool is the 16 GiB load's less the
//! whole pages the desktop takes. Less 1 GiB the profile's shape still fits,
//! nothing adapted. Less 3 GiB the context adapts to 65,536 at two lanes:
//! 196,608 is refused by the plan's own floor, and 131,072 holds one
//! sequence but not the spare reservation beside it
//! (`vram.adapt_spare_tokens`), short by 577 pages.
//!
//! GitHub #415 (ADR 0075) loads the profile in the same 16 GiB with vision on
//! under the `host` residency, at a 1,024- and a 4,096-token envelope, and
//! finds the text-only load's KV pool, to the page; GitHub #418 makes that the
//! profile's own default, so the tests above load the profile as it ships,
//! vision on, and the text-only load is asked for with `--vision-enabled false`.
//!
//! Its own test binary: the request log is written on the engine's telemetry
//! task, so the capture is the process's global subscriber. The two budget
//! cases read only the load's own lines, logged on the loading thread, so
//! they capture with a scoped subscriber instead.
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
use ignis_core::{KvFormat, KvGeometry, KvPoolSize, VramMode, VramPlanError};
use ignis_server::Server;
use ignis_server::adaptation::{Adaptable, Attempt, StartRefusal};
use ignis_server::runtime::{EngineShape, Qwen27bLoadError, qwen27b_load};

const TRELLIS: &str = "F:/ai/models/Qwen3.8-27B-trellis-3bpw-ignis/qwen3_8_27b_trellis_sc30-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";
/// A 16 GB card's whole memory: spec 05a Acceptance A1's budget.
const BUDGET: &str = "17179869184";
/// The 16 GiB load's trellis lines on the 5090 (spec 05a's As built, "The
/// plan's trellis lines"): the prefill chunk's scratch without the vendored
/// projection workspaces and with the q | gate buffer, the head-width-bounded
/// trellis workspace at its widest call (`mlp/down` at the 1,024-token
/// chunk), and the round arena at 2 lanes x 8 columns.
///
/// The profile ships with vision on (#418: 4,096 tokens, `host` residency), so
/// the workspace line is the text-only load's 395,898,880 bytes plus 25,600:
/// the vision attention readout's share of the prefill scratch at a 4,096-token
/// envelope, under a KV page, so the pool and the context are the text-only
/// load's. The decode graph line grows by 8 bytes too (vision's rope staging;
/// 20,621,328 text-only) and the verify round's by 64; the trellis workspace is
/// unchanged.
const WORKSPACE_BYTES: u64 = 395_924_480;
/// The same line under `--vision-enabled false`.
const TEXT_ONLY_WORKSPACE_BYTES: u64 = 395_898_880;
const TRELLIS_WORKSPACE_BYTES: u64 = 36_225_280;
const DECODE_GRAPH_BYTES: u64 = 20_621_336;
/// Spec 05b's capacity simulation: the 16 GB card less 1 GiB and less 3 GiB
/// of simulated desktop.
const BUDGET_LESS_1_GIB: u64 = 17_179_869_184 - (1 << 30);
const BUDGET_LESS_3_GIB: u64 = 17_179_869_184 - (3 << 30);
/// The 16 GiB load's KV pool (#339 A1): 7,956 pages.
const POOL_TOKENS: u64 = 509_184;
/// The spare reservation the ladder keeps beside one full sequence at two
/// lanes: `vram.adapt_spare_tokens`' default (ADR 0063 §3).
const SPARE_TOKENS: u64 = 65_536;

/// The 16 GB card's start inside `budget` bytes (#339 A1's flags).
fn flags_16g(budget: &str) -> [&str; 8] {
    ["--profile", "rtx50-16gb", "--vram-budget-bytes", budget, "--model-idle-unload", "0", "--model-loop-guard", "off"]
}

fn page_bytes() -> u64 {
    KvFormat::HqE8_2b.page_bytes(KvGeometry::qwen38_27b())
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

/// The 16 GB artifact through the server's own load path inside `budget`
/// bytes, and the events the load logged; `None` where the profile allows a
/// skip.
fn load_16g(budget: u64) -> Option<(ignis_server::ActiveModel, Vec<serde_json::Value>)> {
    load_16g_with(budget, &[])
}

/// [`load_16g`] with `extra` flags after the profile's.
fn load_16g_with(budget: u64, extra: &[&str]) -> Option<(ignis_server::ActiveModel, Vec<serde_json::Value>)> {
    if !Path::new(TRELLIS).exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {TRELLIS}")) {
        return None;
    }
    let budget = budget.to_string();
    let flags = [&flags_16g(&budget)[..], extra].concat();
    let options = ignis_server::config::Config { model: MODEL.to_owned(), model_named: true, ..the_27b(&flags) };
    let (loaded, events) = captured(|| {
        ignis_server::load::prepare_model(&options, Path::new(TRELLIS)).and_then(ignis_server::load::load_model)
    });
    match loaded {
        Ok(loaded) => Some((loaded.model, events)),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the 16 GB artifact in {budget} bytes: {e}"));
            None
        }
    }
}

/// What the 16 GB artifact's plan needs at `shape`'s own values, nothing
/// adapting -- its lines and a KV pool of the floor, or of the size `shape`
/// names: its refusal at a 1 GiB budget carries it
/// (`VramPlanError::BelowMinimum`'s `needed_bytes`), before anything is
/// materialized. `None` where the profile allows a skip.
fn needed_bytes(shape: EngineShape, eos: u32) -> Option<u64> {
    let refused = EngineShape {
        vram: VramMode::Explicit { budget_bytes: 1 << 30, allow_oversubscription: false },
        adaptable: Adaptable::NONE,
        ..shape
    };
    match qwen27b_load(Path::new(TRELLIS), MODEL.into(), eos, refused, None, &Default::default()) {
        Ok(_) => panic!("a 1 GiB budget holds no 27B"),
        Err(Qwen27bLoadError::VramPlan(StartRefusal { error: VramPlanError::BelowMinimum { needed_bytes, .. }, tried })) => {
            assert!(tried.is_empty(), "nothing adapts: {tried:?}");
            Some(needed_bytes)
        }
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("plan the 16 GB artifact at {:?}: {e}", shape.adaptation_start()));
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn the_16gb_artifact_serves_greedy_code_on_two_dflash2_lanes_in_a_16_gib_budget() {
    if !Path::new(TRELLIS).exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {TRELLIS}")) {
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

    let flags = flags_16g(BUDGET);
    let options = ignis_server::config::Config { model: MODEL.to_owned(), model_named: true, ..the_27b(&flags) };
    let loaded = match ignis_server::load::prepare_model(&options, Path::new(TRELLIS)).and_then(ignis_server::load::load_model)
    {
        Ok(loaded) => loaded,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the 16 GB artifact under rtx50-16gb: {e}"));
            return;
        }
    };
    let model = loaded.model;
    let logged = events();
    let plan = named(&logged, "ignis.runtime.vram_plan");
    assert_eq!(plan.len(), 1, "{logged:?}");
    println!("vram_plan: {}", plan[0]);
    let lines = &plan[0]["attributes"];
    assert_eq!(lines["budget_bytes"], 17_179_869_184u64);
    assert_eq!(lines["mode"], "explicit");
    let trellis_workspace = lines["trellis_workspace_bytes"].as_u64().expect("the trellis line");
    let weights = lines["weights_bytes"].as_u64().unwrap();
    // Spec 01's As built: the text scope is 9,825,087,488 bytes of device
    // arena; the drafter adds its 66 objects (the full head: no shortlist).
    assert!((9_825_087_488..11_264_998_400).contains(&weights), "{weights}");
    assert!(lines["total_bytes"].as_u64().unwrap() <= 17_179_869_184, "the plan fits the budget");
    println!(
        "plan: weights {weights}, trellis workspace {trellis_workspace}, workspace {}, decode graph {}, kv pool {} \
         tokens, adapted '{}'",
        lines["workspace_bytes"], lines["decode_graph_bytes"], lines["kv_pool_tokens"], lines["adapted"]
    );
    // The profile's shape fits a 16 GiB card as asked: nothing adapted.
    assert_eq!(lines["adapted"], "", "the 16 GiB load adapted its shape");
    assert_eq!(trellis_workspace, TRELLIS_WORKSPACE_BYTES, "the trellis workspace line");
    assert_eq!(lines["workspace_bytes"].as_u64(), Some(WORKSPACE_BYTES), "the workspace line");
    assert_eq!(lines["decode_graph_bytes"].as_u64(), Some(DECODE_GRAPH_BYTES), "the decode graph line");
    let load = named(&logged, "ignis.model.loaded");
    println!("model.loaded: {}", load[0]);
    assert_eq!(load[0]["attributes"]["decode_lanes"], 2);
    let speculation = load[0]["attributes"]["speculation"].as_str().unwrap_or_default();
    assert!(speculation.starts_with("dflash2 draft_tokens=7 draft_head=full"), "{speculation}");

    let app = Server::from_active(model.clone()).with_request_timeout(Duration::from_secs(600)).app();
    let prompts = [
        (
            "Write a Python function `is_prime(n)` that returns True when n is a prime number. Reply with the code only.",
            "def is_prime",
        ),
        ("Write a Rust function `fn gcd(a: u64, b: u64) -> u64` using Euclid's algorithm. Reply with the code only.", "fn gcd"),
    ];
    let mut rounds = 0u64;
    let mut accepted = 0u64;
    let mut sequential = Vec::new();
    for (prompt, needle) in prompts {
        let first = complete(&app, prompt, 160).await;
        let again = complete(&app, prompt, 160).await;
        println!("completion: {first:?}");
        assert!(first.contains(needle), "the completion holds `{needle}`: {first:?}");
        assert_eq!(first, again, "greedy decoding is deterministic");
        sequential.push(first);
    }
    // The request log is written on the telemetry task after each response.
    let settled = |count: usize| async move {
        let mut done = Vec::new();
        for _ in 0..100 {
            done = events().into_iter().filter(|e| e["event_name"] == "ignis.request.done").collect::<Vec<_>>();
            if done.len() >= count {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        done
    };
    assert_eq!(settled(2 * prompts.len()).await.len(), 2 * prompts.len(), "every request logs its done line");

    // The two prompts at once: two lanes, so both are admitted and decode
    // together, and each lane's text is the one it had alone.
    let before = events().len();
    let (left, right) = tokio::join!(complete(&app, prompts[0].0, 160), complete(&app, prompts[1].0, 160));
    assert_eq!([left, right], [sequential[0].clone(), sequential[1].clone()], "a lane beside another keeps its text");
    let done = settled(3 * prompts.len()).await;
    assert_eq!(done.len(), 3 * prompts.len(), "every request logs its done line");
    // In flight together: both requests' first tokens are logged before
    // either request's done line.
    let pair: Vec<serde_json::Value> = events().split_off(before);
    let first_done = pair.iter().position(|e| e["event_name"] == "ignis.request.done").expect("a done line");
    let ttft_before_done = pair[..first_done].iter().filter(|e| e["event_name"] == "ignis.request.ttft").count();
    assert_eq!(ttft_before_done, 2, "the two requests did not decode together: {pair:?}");
    let (mut pair_rounds, mut pair_accepted) = (0u64, 0u64);
    for (index, line) in done.iter().enumerate() {
        let a = &line["attributes"];
        let (r, d, c) = (a["spec.rounds"].as_u64(), a["spec.drafted"].as_u64(), a["spec.accepted"].as_u64());
        println!(
            "request {}: tokens {}, {:.1} tok/s, lane {}, spec.rounds {r:?}, drafted {d:?}, accepted {c:?}, pos {}",
            a["request_id"], a["tokens"], a["tok_s"].as_f64().unwrap_or(0.0), a["lane"], a["spec.pos"]
        );
        let r = r.expect("a DFlash2 load's request runs verify rounds");
        if index < 2 * prompts.len() {
            rounds += r;
            accepted += c.unwrap_or(0);
        } else {
            pair_rounds += r;
            pair_accepted += c.unwrap_or(0);
        }
    }
    // A round commits its accepted drafts and the target's own token.
    let per_round = (accepted + rounds) as f64 / rounds as f64;
    println!("tokens per round: {per_round:.2} over {rounds} rounds (one request at a time)");
    println!(
        "tokens per round: {:.2} over {pair_rounds} rounds (the pair, two lanes)",
        (pair_accepted + pair_rounds) as f64 / pair_rounds.max(1) as f64
    );
    assert!(per_round > 1.0, "the drafter's proposals are accepted at all: {per_round}");

    drop(app);
    tokio::task::spawn_blocking(move || model.shut_down()).await.expect("the model shuts down");
}

/// The card less 1 GiB of simulated desktop: the profile's 262,144 tokens on
/// two lanes still fit with the spare reservation beside them, so nothing
/// adapts. Every line but the pool is the 16 GiB load's, and the pool, which
/// takes the rest, is its 7,956 pages less the 1,820 whole pages 1 GiB is
/// (the 16 GiB load left 493,468 bytes under a page; 1 GiB is 1,820 pages and
/// 262,144 bytes).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn less_1_gib_of_desktop_keeps_262144_tokens_on_two_lanes_and_adapts_nothing() {
    let Some((model, events)) = load_16g(BUDGET_LESS_1_GIB) else { return };
    let plan = named(&events, "ignis.runtime.vram_plan");
    assert_eq!(plan.len(), 1, "{events:?}");
    println!("vram_plan: {}", plan[0]);
    let lines = &plan[0]["attributes"];
    assert_eq!(lines["budget_bytes"], BUDGET_LESS_1_GIB);
    assert_eq!(lines["adapted"], "", "the profile's shape fits less 1 GiB");
    assert_eq!(lines["workspace_bytes"].as_u64(), Some(WORKSPACE_BYTES), "the workspace line");
    assert_eq!(lines["trellis_workspace_bytes"].as_u64(), Some(TRELLIS_WORKSPACE_BYTES), "the trellis workspace line");
    assert_eq!(lines["decode_graph_bytes"].as_u64(), Some(DECODE_GRAPH_BYTES), "the decode graph line");
    let pool = lines["kv_pool_tokens"].as_u64().unwrap();
    assert_eq!(pool, POOL_TOKENS - 1_820 * 64, "the 16 GiB load's pool less 1 GiB's whole pages");
    assert!(pool >= 262_144 + SPARE_TOKENS, "one sequence and the spare reservation: {pool}");
    let total = lines["total_bytes"].as_u64().unwrap();
    assert!(BUDGET_LESS_1_GIB - total < page_bytes(), "the pool takes the rest: {total}");
    let load = named(&events, "ignis.model.loaded");
    println!("model.loaded: {}", load[0]);
    assert_eq!(load[0]["attributes"]["decode_lanes"], 2);
    assert_eq!(load[0]["attributes"]["max_context"], 262_144);
    tokio::task::spawn_blocking(move || model.shut_down()).await.expect("the model shuts down");
}

/// The card less 3 GiB of simulated desktop: 262,144 tokens no longer fit,
/// and the ladder (ADR 0063 §3) keeps 65,536 at two lanes. 196,608 is refused
/// by the plan's own floor (one sequence and a page per retained slot);
/// 131,072 holds one sequence but not the spare reservation of
/// `vram.adapt_spare_tokens` beside it; 65,536 holds both. Each is the plan
/// at that context, the lines planned again for it, refused or not before
/// anything is materialized. The lines are the 16 GiB load's at every
/// context, so the pool is its 7,956 pages less the 5,461 whole pages 3 GiB
/// is, 2,495, and 131,072 misses its 3,072 by 577 pages (36,928 tokens).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn less_3_gib_of_desktop_adapts_the_context_to_65536_on_two_lanes() {
    if !Path::new(TRELLIS).exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {TRELLIS}")) {
        return;
    }
    let reader = Reader::open(Path::new(TRELLIS)).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend.eos_token_id().expect("qwen3.8-27b generation config must carry eos_token_id");
    let budget = BUDGET_LESS_3_GIB.to_string();
    // The shape the load derives, built here without `load::load_shape` so
    // the plans below log no draft-head line.
    let shape = EngineShape::from(&the_27b(&flags_16g(&budget))).with_family_decode_share(ModelFamily::Qwen38_27b);
    let start = shape.adaptation_start();
    assert_eq!((start.max_context, start.decode_lanes), (262_144, 2));
    let at = |max_context: u32, pool_tokens: Option<u64>| EngineShape {
        kv_pool: pool_tokens.map(KvPoolSize::Tokens),
        ..shape.at(Attempt { max_context, ..start })
    };
    let Some(floor_196608) = needed_bytes(at(196_608, None), eos) else { return };
    let Some(floor_131072) = needed_bytes(at(131_072, None), eos) else { return };
    let Some(spare_131072) = needed_bytes(at(131_072, Some(131_072 + SPARE_TOKENS)), eos) else { return };
    let Some(spare_65536) = needed_bytes(at(65_536, Some(65_536 + SPARE_TOKENS)), eos) else { return };
    println!(
        "needed: 196,608 with its floor {floor_196608}; 131,072 with its floor {floor_131072}, with the spare \
         reservation {spare_131072}; 65,536 with the spare reservation {spare_65536}; budget {BUDGET_LESS_3_GIB}"
    );
    assert!(floor_196608 > BUDGET_LESS_3_GIB, "196,608 tokens: the plan's floor alone does not fit");
    assert!(floor_131072 <= BUDGET_LESS_3_GIB, "131,072 tokens: one sequence fits");
    assert!(spare_131072 > BUDGET_LESS_3_GIB, "131,072 tokens: the spare reservation beside it does not");
    let short = spare_131072 - BUDGET_LESS_3_GIB;
    println!(
        "131,072 tokens miss the spare reservation by {short} bytes: {} KV pages, {} tokens",
        short.div_ceil(page_bytes()),
        short.div_ceil(page_bytes()) * 64
    );
    assert_eq!(short.div_ceil(page_bytes()), 577, "131,072 misses its spare reservation by 577 pages");
    assert!(spare_65536 <= BUDGET_LESS_3_GIB, "65,536 tokens and the spare reservation fit");

    let Some((model, events)) = load_16g(BUDGET_LESS_3_GIB) else { return };
    let plan = named(&events, "ignis.runtime.vram_plan");
    assert_eq!(plan.len(), 1, "{events:?}");
    println!("vram_plan: {}", plan[0]);
    let lines = &plan[0]["attributes"];
    assert_eq!(lines["budget_bytes"], BUDGET_LESS_3_GIB);
    assert_eq!(lines["adapted"], "max_context 262144 -> 65536 (adapt_context_step 65536, adapt_spare_tokens 65536)");
    assert_eq!(lines["workspace_bytes"].as_u64(), Some(WORKSPACE_BYTES), "the workspace line, as at 262,144");
    assert_eq!(lines["trellis_workspace_bytes"].as_u64(), Some(TRELLIS_WORKSPACE_BYTES), "the trellis workspace line");
    assert_eq!(lines["decode_graph_bytes"].as_u64(), Some(DECODE_GRAPH_BYTES), "the decode graph line, as at 262,144");
    let pool = lines["kv_pool_tokens"].as_u64().unwrap();
    assert_eq!(pool, POOL_TOKENS - 5_461 * 64, "the 16 GiB load's pool less 3 GiB's whole pages");
    assert!((65_536 + SPARE_TOKENS..131_072 + SPARE_TOKENS).contains(&pool), "{pool}");
    let total = lines["total_bytes"].as_u64().unwrap();
    assert!(BUDGET_LESS_3_GIB - total < page_bytes(), "the pool takes the rest: {total}");
    let load = named(&events, "ignis.model.loaded");
    println!("model.loaded: {}", load[0]);
    assert_eq!(load[0]["attributes"]["decode_lanes"], 2);
    assert_eq!(load[0]["attributes"]["max_context"], 65_536);
    tokio::task::spawn_blocking(move || model.shut_down()).await.expect("the model shuts down");
}

/// GitHub #415 (ADR 0075, the build's acceptance 1): under the `host` vision
/// residency a vision load inside the 16 GiB budget, no image in flight, keeps
/// exactly the text-only load's KV pool at a 1,024- and a 4,096-token envelope
/// (7,956 pages in #409's prototype, against 7,437 and 7,384 under `device`):
/// the tower leaves the weights line, staged through slots that fit the
/// prefill scratch, and the embedding pool is host-mapped, so its
/// `media_embedding` line is 0. The workspace may still grow by vision's
/// attention readout (13,312 bytes at 1,024 in the prototype), under a page.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn under_the_host_residency_vision_keeps_the_text_only_kv_pool_at_1024_and_4096() {
    async fn plan_of(extra: &[&str]) -> Option<serde_json::Value> {
        let (model, events) = load_16g_with(17_179_869_184, extra)?;
        let plan = named(&events, "ignis.runtime.vram_plan");
        assert_eq!(plan.len(), 1, "{events:?}");
        println!("vram_plan {extra:?}: {}", plan[0]);
        let lines = plan[0]["attributes"].clone();
        tokio::task::spawn_blocking(move || model.shut_down()).await.expect("the model shuts down");
        Some(lines)
    }
    let Some(text) = plan_of(&["--vision-enabled", "false"]).await else { return };
    assert_eq!(text["adapted"], "", "the text-only load fits as asked");
    assert!(text.get("vision_residency").is_none(), "{text}");
    assert_eq!(text["workspace_bytes"].as_u64(), Some(TEXT_ONLY_WORKSPACE_BYTES), "the text-only workspace line");
    for envelope in ["1024", "4096"] {
        let flags = ["--vision-enabled", "--vision-max-tokens", envelope, "--vision-residency", "host"];
        let Some(host) = plan_of(&flags).await else { return };
        assert_eq!(host["vision_residency"], "host");
        assert_eq!(host["adapted"], "", "{envelope}: the vision load fits as asked");
        assert_eq!(host["media_embedding_bytes"], 0, "{envelope}: the pool is host-mapped, no VRAM line");
        println!(
            "{envelope}: weights {} (text {}), workspace {} (text {}), kv pool {} pages (text {})",
            host["weights_bytes"], text["weights_bytes"], host["workspace_bytes"], text["workspace_bytes"],
            host["kv_pool_pages"], text["kv_pool_pages"]
        );
        assert_eq!(host["kv_pool_pages"], text["kv_pool_pages"], "{envelope}: the text-only load's KV pool");
    }
}

/// GitHub #418 (ADR 0075, the profile's default): `rtx50-16gb` with no vision
/// flag at all loads with vision on under the `host` residency at its
/// 4,096-token envelope, and, no image in flight, costs no context: the same KV
/// pool as the text-only load, a `media_embedding` line of 0, and a workspace
/// that grew by the readout's 25,600 bytes, under a page.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn the_profiles_own_default_load_has_host_vision_and_the_text_only_kv_pool() {
    async fn plan_of(extra: &[&str]) -> Option<serde_json::Value> {
        let (model, events) = load_16g_with(17_179_869_184, extra)?;
        let plan = named(&events, "ignis.runtime.vram_plan");
        assert_eq!(plan.len(), 1, "{events:?}");
        println!("vram_plan {extra:?}: {}", plan[0]);
        let lines = plan[0]["attributes"].clone();
        tokio::task::spawn_blocking(move || model.shut_down()).await.expect("the model shuts down");
        Some(lines)
    }
    let Some(text) = plan_of(&["--vision-enabled", "false"]).await else { return };
    let Some(shipped) = plan_of(&[]).await else { return };
    assert_eq!(shipped["vision_residency"], "host", "{shipped}");
    assert_eq!(shipped["adapted"], "", "the profile as shipped fits as asked");
    assert_eq!(shipped["media_embedding_bytes"], 0, "the pool is host-mapped, no VRAM line");
    assert_eq!(shipped["weights_bytes"], text["weights_bytes"], "the tower is staged, not resident");
    assert_eq!(shipped["workspace_bytes"].as_u64(), Some(WORKSPACE_BYTES), "the workspace line, readout included");
    assert_eq!(shipped["kv_pool_pages"], text["kv_pool_pages"], "vision on costs no context with no image in flight");
}
