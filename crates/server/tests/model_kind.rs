//! The model kind (spec image-model/05, ADR 0050, GitHub #322) over the
//! mock-backed router: with the image model loaded the text routes refuse a
//! request no text model would serve `409 model_kind_mismatch`, and one naming
//! a known text model switches to it, or is refused by name with implicit
//! switching off (AC 1); `/metrics`, the tokenize routes and `GET /v1/models`
//! answer without an engine (AC 2); an image job keeps the image model
//! loaded and an unused one unloads like a text model (AC 8); and the
//! production loader verifies the image artifact and loads it on the fake
//! runner. The image model is `support/switch.rs`'s mock loader's, over
//! `FakeImageRunner`; the text models run on `MockCompute`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_core::compute::ModelKind;
use ignis_server::model_switch::{ArtifactLoader, ModelLoader, Switcher};
use ignis_server::{ModelSource, Server};
use serde_json::{json, Value};
use tower::ServiceExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::{source, MockLoader};

/// The image model's served id: its family's own, so its kind is known by
/// the id alone.
const IMAGE: &str = "ignis-image-2.1-turbo";

const PATIENT: Duration = Duration::from_secs(30);

/// The tests' idle clock and `model.idle_unload`, as `idle_unload.rs` runs
/// them.
const TICK: Duration = Duration::from_millis(50);
const IDLE: Duration = Duration::from_secs(1);

/// The known-models table naming each of `ids` at the artifact [`source`]
/// gives it.
fn known(ids: &[&str]) -> BTreeMap<String, PathBuf> {
    ids.iter().map(|id| (id.to_string(), source(id).artifact)).collect()
}

/// A server on `start` whose requests may switch to any of `ids` by naming
/// it (`switch.allow_implicit` on).
fn implicit_on(loader: &Arc<MockLoader>, start: &str, ids: &[&str]) -> Server {
    Server::from_active(loader.model(start))
        .with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT).with_known_models(known(ids)))
}

/// The same server with `switch.allow_implicit` off: `ids` may be switched
/// to by id, never by a request naming them — what `main` builds.
fn implicit_off(loader: &Arc<MockLoader>, start: &str, ids: &[&str]) -> Server {
    Server::from_active(loader.model(start))
        .with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT).with_listed_models(known(ids)))
}

async fn send(app: &axum::Router, method: Method, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder().method(method).uri(uri).header(header::CONTENT_TYPE, "application/json");
    let request = request.body(body.map_or_else(Body::empty, |body| Body::from(body.to_string()))).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body = serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, body)
}

/// `body` with `model` set, when one is named.
fn naming(mut body: Value, model: Option<&str>) -> Value {
    if let Some(model) = model {
        body["model"] = json!(model);
    }
    body
}

/// `POST /v1/models/switch`'s body: the mock artifact [`source`] names.
fn switch_to(model: &str) -> Value {
    json!({ "artifact": source(model).artifact, "model": model })
}

fn chat(model: Option<&str>) -> Value {
    naming(json!({ "messages": [{ "role": "user", "content": "hello there" }], "max_tokens": 3 }), model)
}

fn response_body(model: Option<&str>) -> Value {
    naming(json!({ "input": "hello there", "max_output_tokens": 3, "enable_thinking": false }), model)
}

fn decision(model: Option<&str>) -> Value {
    naming(json!({ "state": "s", "questions": { "q": { "type": "noul", "instructions": "Urgent?" } } }), model)
}

/// The text routes, each with the body it takes.
fn text_routes(model: Option<&str>) -> [(&'static str, Value); 4] {
    [
        ("/v1/chat/completions", chat(model)),
        ("/v1/responses", response_body(model)),
        ("/v1/decide", decision(model)),
        ("/v1/systemone", decision(model)),
    ]
}

/// Wait (bounded) until the status reads `wanted`.
async fn until(server: &Server, wanted: &str, within: Duration) {
    let deadline = Instant::now() + within;
    while server.status().as_str() != wanted {
        assert!(Instant::now() < deadline, "the status never read {wanted}: {:?}", server.status());
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Wait (bounded) until the idle clock has unloaded the model and its
/// teardown has let go of the switch's slot.
async fn unloaded(server: &Server) {
    until(server, "unloaded", IDLE * 5).await;
    let switcher = server.switcher.as_ref().expect("a switcher");
    for _ in 0..1_000 {
        if switcher.running().is_none() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("the unload never let go of the switch's slot");
}

/// AC 1: with the image model loaded, a text request naming no model is
/// refused `409 model_kind_mismatch`, naming both kinds and the loaded id —
/// and nothing is loaded for it.
#[tokio::test]
async fn with_the_image_model_loaded_a_text_request_naming_no_model_is_refused_naming_both_kinds() {
    let loader = MockLoader::new();
    let server = implicit_on(&loader, IMAGE, &[IMAGE, "mock-a"]);
    let app = server.app();
    for (uri, body) in text_routes(None) {
        let (status, refused) = send(&app, Method::POST, uri, Some(body)).await;
        assert_eq!(status, StatusCode::CONFLICT, "{uri}: {refused}");
        assert_eq!(refused["error"]["code"], "model_kind_mismatch", "{uri}: {refused}");
        assert_eq!(refused["error"]["param"], "model", "{uri}: {refused}");
        let message = refused["error"]["message"].as_str().unwrap();
        assert!(message.contains("serves text models") && message.contains(&format!("{IMAGE} is an image model")), "{uri}: {message}");
    }
    // Naming the loaded image model itself is the same request.
    let (status, refused) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(Some(IMAGE)))).await;
    assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::CONFLICT, Some("model_kind_mismatch")), "{refused}");
    assert!(loader.loads().is_empty());
    assert_eq!(server.active().kind(), ModelKind::Image);
}

/// AC 1: naming a known text model switches from the image model to it, on
/// every text route, and the request is served by it; the image model's
/// runner is gone before the text model loads.
#[tokio::test]
async fn naming_a_known_text_model_switches_from_the_image_model_and_is_served_by_it() {
    let loader = MockLoader::new();
    let server = implicit_on(&loader, IMAGE, &[IMAGE, "mock-a", "mock-b"]);
    let app = server.app();
    let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(Some("mock-a@agent")))).await;
    assert_eq!(status, StatusCode::OK, "{completion}");
    assert_eq!(completion["model"], "mock-a");
    assert_eq!((server.active().kind(), server.active().model_id()), (ModelKind::Text, "mock-a".to_owned()));

    let back = send(&app, Method::POST, "/v1/models/switch", Some(switch_to(IMAGE))).await;
    assert_eq!(back.0, StatusCode::ACCEPTED, "{}", back.1);
    until(&server, "serving", PATIENT).await;
    let (status, response) = send(&app, Method::POST, "/v1/responses", Some(response_body(Some("mock-b")))).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["model"], "mock-b");

    let back = send(&app, Method::POST, "/v1/models/switch", Some(switch_to(IMAGE))).await;
    assert_eq!(back.0, StatusCode::ACCEPTED, "{}", back.1);
    until(&server, "serving", PATIENT).await;
    let (status, answered) = send(&app, Method::POST, "/v1/decide", Some(decision(Some("mock-a")))).await;
    // The mock template has no answer labels: past the model checks, the
    // question itself is refused.
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{answered}");
    assert!(!matches!(answered["error"]["code"].as_str(), Some("model_kind_mismatch" | "model_not_found")), "{answered}");
    assert_eq!(server.active().model_id(), "mock-a");

    assert_eq!(loader.loads(), ["mock-a", IMAGE, "mock-b", IMAGE, "mock-a"]);
    assert_eq!(loader.resident_at_load(), [0; 5], "each model left the card before the next loaded");
}

/// AC 1: with `switch.allow_implicit` off, a text request naming a text model
/// while the image model is loaded is refused by name, as a model the
/// server does not serve always was — `404`, and `/v1/decide`'s own `422`.
#[tokio::test]
async fn with_implicit_switching_off_a_named_text_model_is_refused_by_name() {
    let loader = MockLoader::new();
    let server = implicit_off(&loader, IMAGE, &[IMAGE, "mock-a"]);
    let app = server.app();
    for (uri, body) in text_routes(Some("mock-a")) {
        let (status, refused) = send(&app, Method::POST, uri, Some(body)).await;
        let expected = if uri.starts_with("/v1/chat") || uri == "/v1/responses" { StatusCode::NOT_FOUND } else { StatusCode::UNPROCESSABLE_ENTITY };
        assert_eq!(status, expected, "{uri}: {refused}");
        assert_eq!(refused["error"]["code"], "model_not_found", "{uri}: {refused}");
        let message = refused["error"]["message"].as_str().unwrap();
        assert!(message.contains("mock-a") && message.contains(IMAGE), "{uri}: {message}");
    }
    assert!(loader.loads().is_empty());
    assert_eq!(server.active().model_id(), IMAGE);
}

/// AC 1, the other way round: a text route naming the image model is
/// refused before any switch — a request never loads the model that would
/// refuse it — while an explicit switch still goes to it.
#[tokio::test]
async fn a_text_route_naming_the_image_model_is_refused_without_switching_to_it() {
    let loader = MockLoader::new();
    let server = implicit_on(&loader, "mock-a", &["mock-a", IMAGE]);
    let app = server.app();
    for (uri, body) in text_routes(Some(IMAGE)) {
        let (status, refused) = send(&app, Method::POST, uri, Some(body)).await;
        assert_eq!(status, StatusCode::CONFLICT, "{uri}: {refused}");
        assert_eq!(refused["error"]["code"], "model_kind_mismatch", "{uri}: {refused}");
        let message = refused["error"]["message"].as_str().unwrap();
        assert!(message.contains(&format!("`model` names {IMAGE}, an image model")) && message.contains("loaded: mock-a, a text model"), "{uri}: {message}");
    }
    assert!(loader.loads().is_empty());

    let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to(IMAGE))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    until(&server, "serving", PATIENT).await;
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!(models["data"][0]["id"], IMAGE);
    assert_eq!(models["data"][0]["kind"], "image");
    assert!(models["data"][0].get("max_model_len").is_none(), "an image model has no context: {models}");
    assert_eq!(loader.loads(), [IMAGE]);
}

/// A series' value in a `/metrics` scrape, if the scrape has it.
fn gauge(scrape: &Value, series: &str) -> Option<u64> {
    scrape.as_str()?.lines().find_map(|line| line.strip_prefix(series)?.strip_prefix(' ')?.parse().ok())
}

/// AC 2: `/metrics`, the tokenize routes and `GET /v1/models` answer with the
/// image model loaded — the counting routes refuse it, the others report —
/// and a switch to it and back with metrics on touches no engine that is not
/// there. What the text model's load reserved reads zero once the image
/// model holds the card.
#[tokio::test]
async fn metrics_the_tokenize_routes_and_the_models_list_answer_with_the_image_model_loaded() {
    let loader = MockLoader::new();
    let reserved = ignis_server::metrics::LoadReservations {
        lines: ignis_core::VramLines { weights: 7 << 30, ..Default::default() },
        budget_bytes: 30 << 30,
        kv_pool_pages: 1024,
        kv_page_bytes: 1 << 20,
        residency_bytes: 0,
        expert_cache_bytes: 0,
        condition_budget_bytes: 0,
        kv_ram_arena_bytes: 0,
        retained_slots: 0,
        retained_host_slots: 0,
        retained_host_bytes: 0,
        kv_disk_bytes: None,
        flash_next: None,
        draft_positions: 0,
    };
    let server = implicit_on(&loader, "mock-a", &["mock-a", IMAGE]).with_metrics().with_load_reservations(reserved);
    let (app, metrics) = (server.app(), server.metrics_app().expect("metrics are on"));
    let (_, scraped) = send(&metrics, Method::GET, "/metrics", None).await;
    assert_eq!(gauge(&scraped, "ignis_kv_pool_pages"), Some(1024), "the text model's load");
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!((models["data"][0]["kind"].as_str(), models["data"][0]["max_model_len"].is_u64()), (Some("text"), true), "{models}");

    let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to(IMAGE))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    until(&server, "serving", PATIENT).await;
    assert_eq!(server.active().kind(), ModelKind::Image);

    for (uri, body) in [
        ("/v1/tokenize", json!({ "messages": [{ "role": "user", "content": "count me" }] })),
        ("/v1/tokenize", json!({ "prompt": "count me" })),
        ("/v1/detokenize", json!({ "token_ids": [1, 2] })),
    ] {
        let (status, refused) = send(&app, Method::POST, uri, Some(body)).await;
        assert_eq!(status, StatusCode::CONFLICT, "{uri}: {refused}");
        assert_eq!(refused["error"]["code"], "model_kind_mismatch", "{uri}: {refused}");
    }
    let (status, scraped) = send(&metrics, Method::GET, "/metrics", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(scraped.as_str().is_some_and(|text| text.contains("# TYPE")), "{scraped}");
    for series in ["ignis_kv_pool_pages", "ignis_vram_budget_bytes", "ignis_vram_reserved_bytes{line=\"weights\"}"] {
        assert_eq!(gauge(&scraped, series), Some(0), "{series} with the image model loaded");
    }
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!(models["data"][0]["kind"], "image");

    // Back on a text model, its engine feeds the projection again.
    let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(Some("mock-a")))).await;
    assert_eq!(status, StatusCode::OK, "{completion}");
    let counted = json!({ "messages": [{ "role": "user", "content": "count me" }] });
    let (status, counted) = send(&app, Method::POST, "/v1/tokenize", Some(counted)).await;
    assert_eq!(status, StatusCode::OK, "{counted}");
    assert_eq!(send(&metrics, Method::GET, "/metrics", None).await.0, StatusCode::OK);
}

/// What a text model's load reserved, for the metrics tests: 21 GiB of
/// weights and a 1,024-page pool in a 30 GiB budget.
fn text_reservations() -> ignis_server::metrics::LoadReservations {
    ignis_server::metrics::LoadReservations {
        lines: ignis_core::VramLines { weights: 21 << 30, ..Default::default() },
        budget_bytes: 30 << 30,
        kv_pool_pages: 1024,
        kv_page_bytes: 1 << 20,
        residency_bytes: 0,
        expert_cache_bytes: 0,
        condition_budget_bytes: 0,
        kv_ram_arena_bytes: 0,
        retained_slots: 0,
        retained_host_slots: 0,
        retained_host_bytes: 0,
        kv_disk_bytes: None,
        flash_next: None,
        draft_positions: 0,
    }
}

/// GitHub #324: the image model's VRAM on `/metrics`. Its load's plan
/// lines land in the text models' family, at the start and after a switch
/// to it; the condition prefix a running job holds reads against the
/// condition budget, and is gone once the job ends; a switch to a text model
/// replaces every image figure with the text model's, and back again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_image_models_vram_plan_and_the_prefixes_its_jobs_hold_show_on_metrics_and_go_with_it() {
    use ignis_core::image::{fake_plan, FAKE_PREFIX_TOKEN_BYTES};
    let budget_tokens = fake_plan().condition_budget_tokens;
    let plan = ignis_core::image_leaf::ImagePlan {
        mode: ignis_core::VramMode::Derived { headroom_bytes: 4 << 30 },
        free_at_start_bytes: 33_400_000_000,
        budget_bytes: 29_100_000_000,
        weights_bytes: 15_460_000_000,
        arena_bytes: 1_320_000_000,
        condition_budget_tokens: budget_tokens,
        condition_budget_bytes: budget_tokens * FAKE_PREFIX_TOKEN_BYTES,
    };
    let image = ignis_server::metrics::LoadReservations::image(&plan);
    let loader = MockLoader::new();
    loader.reserve(IMAGE, image.clone());
    loader.reserve("mock-a", text_reservations());
    let (model, gate) = loader.gated_image_model(IMAGE);
    let server = Server::from_active(model)
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT).with_known_models(known(&["mock-a", IMAGE])))
        .with_metrics()
        .with_load_reservations(image);
    let (app, metrics) = (server.app(), server.metrics_app().expect("metrics are on"));
    let scrape = || async { send(&metrics, Method::GET, "/metrics", None).await.1 };
    let image_figures = |scraped: &Value| {
        ["weights", "workspace", "condition_budget"]
            .map(|line| gauge(scraped, &format!("ignis_vram_reserved_bytes{{line=\"{line}\"}}")))
    };
    let used = |scraped: &Value| gauge(scraped, "ignis_condition_budget_used_bytes");
    let image_plan = [Some(15_460_000_000), Some(1_320_000_000), Some(budget_tokens * FAKE_PREFIX_TOKEN_BYTES)];

    let scraped = scrape().await;
    assert_eq!(image_figures(&scraped), image_plan, "the start's load");
    assert_eq!(gauge(&scraped, "ignis_vram_budget_bytes"), Some(29_100_000_000));
    assert_eq!(gauge(&scraped, "ignis_kv_pool_pages"), Some(0), "no KV pool");
    assert_eq!(used(&scraped), Some(0));

    // A job of four prompt words holds a four-token prefix while it runs.
    let job = {
        let app = app.clone();
        tokio::spawn(async move {
            send(&app, Method::POST, "/v1/images/generations", Some(json!({ "prompt": "a quiet red fox", "size": "1024x1024" }))).await
        })
    };
    let deadline = Instant::now() + PATIENT;
    while used(&scrape().await) != Some(4 * FAKE_PREFIX_TOKEN_BYTES) {
        assert!(Instant::now() < deadline, "the running job's prefix never showed: {:?}", used(&scrape().await));
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    gate.open();
    let (status, answer) = job.await.unwrap();
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert_eq!(used(&scrape().await), Some(0), "the ended job released its prefix");

    let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to("mock-a"))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    until(&server, "serving", PATIENT).await;
    let scraped = scrape().await;
    assert_eq!(image_figures(&scraped), [Some(21 << 30), Some(0), Some(0)], "the text model's lines, none of the image's");
    assert_eq!((gauge(&scraped, "ignis_kv_pool_pages"), gauge(&scraped, "ignis_vram_budget_bytes")), (Some(1024), Some(30 << 30)));
    assert_eq!(used(&scraped), Some(0));

    let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to(IMAGE))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    until(&server, "serving", PATIENT).await;
    let scraped = scrape().await;
    assert_eq!(image_figures(&scraped), image_plan, "the switch's load");
    assert_eq!((gauge(&scraped, "ignis_kv_pool_pages"), gauge(&scraped, "ignis_vram_budget_bytes")), (Some(0), Some(29_100_000_000)));
    assert_eq!(used(&scraped), Some(0));
    assert_eq!(loader.loads(), ["mock-a", IMAGE]);
}

/// AC 8: a queued image job, then a running one, is use — the idle clock
/// leaves the image model loaded past `model.idle_unload` — and once it ends
/// the model unloads after the idle time, its runner dropped.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_queued_or_running_image_job_keeps_the_image_model_loaded() {
    let loader = MockLoader::new();
    let server = implicit_on(&loader, IMAGE, &[IMAGE]).with_idle_unload(IDLE);
    let _clock = server.spawn_idle_clock(TICK).expect("a switcher, so a clock");
    let jobs = Arc::clone(&server.active().image_model().expect("the image model").jobs);

    let waiting = jobs.wait();
    tokio::time::sleep(IDLE + IDLE / 2).await;
    assert_eq!(server.status().as_str(), "serving", "a queued job is use");
    let running = waiting.turn().await.expect("the only job's turn at once");
    tokio::time::sleep(IDLE + IDLE / 2).await;
    assert_eq!(server.status().as_str(), "serving", "a running job is use");
    assert_eq!(loader.resident(), 1);

    let ended = Instant::now();
    drop(running);
    unloaded(&server).await;
    // A job driven here holds no request's `InUse`, so its last use is the
    // clock's last tick that saw it running: up to one TICK before its end
    // (985 ms on a Windows runner, whose timer moves in ~15 ms steps).
    assert!(ended.elapsed() >= IDLE - TICK, "the idle time counts from the job's end: {:?}", ended.elapsed());
    assert_eq!(loader.resident(), 0, "the unloaded image model's runner was dropped");
    assert!(jobs.is_closed(), "the unloaded model's queue admits nothing more");
    assert!(loader.loads().is_empty());
}

/// AC 8: an unused image model unloads after `model.idle_unload` like a text
/// model, keeping its id and kind; a text request naming no model leaves it
/// unloaded, since loading it would only refuse the request; a switch loads
/// it again, and a request naming a text model loads that one alone.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_unused_image_model_unloads_and_the_next_request_that_needs_a_model_loads_one() {
    let loader = MockLoader::new();
    let server = implicit_on(&loader, IMAGE, &[IMAGE, "mock-a"]).with_idle_unload(IDLE);
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    let app = server.app();
    unloaded(&server).await;
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!((models["status"].as_str(), models["data"][0]["id"].as_str(), models["data"][0]["kind"].as_str()), (Some("unloaded"), Some(IMAGE), Some("image")));

    let (status, refused) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(None))).await;
    assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::CONFLICT, Some("model_kind_mismatch")), "{refused}");
    assert_eq!(server.status().as_str(), "unloaded", "nothing was loaded for a request it would refuse");

    let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to(IMAGE))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    until(&server, "serving", PATIENT).await;
    assert_eq!(server.active().kind(), ModelKind::Image);
    unloaded(&server).await;

    let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(Some("mock-a")))).await;
    assert_eq!(status, StatusCode::OK, "{completion}");
    assert_eq!(completion["model"], "mock-a");
    assert_eq!(loader.loads(), [IMAGE, "mock-a"], "one load each, never the image model before the text one");
}

/// A switch away from the image model drains its jobs as it drains a text
/// model's lanes: a running job holds the teardown back for the drain
/// window — the runner stays, nothing loads — and past a short window the
/// queue is closed and the switch lands anyway, the runner dropped before
/// the next model loads.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_switch_drains_the_image_jobs_and_past_its_window_closes_the_queue() {
    for (drain, released) in [(PATIENT, true), (Duration::from_millis(100), false)] {
        let loader = MockLoader::new();
        let server = Server::from_active(loader.model(IMAGE))
            .with_switcher(Switcher::new(Arc::clone(&loader) as _, drain).with_known_models(known(&[IMAGE, "mock-a"])));
        let app = server.app();
        let jobs = Arc::clone(&server.active().image_model().expect("the image model").jobs);
        let running = jobs.wait().turn().await.expect("the only job's turn at once");
        let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to("mock-a"))).await;
        assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
        if released {
            tokio::time::sleep(Duration::from_millis(300)).await;
            assert_eq!(server.status().as_str(), "switching", "a running job holds the switch in its drain");
            assert!(loader.loads().is_empty() && !jobs.is_closed());
            assert_eq!(loader.resident(), 1, "the image runner is still there");
            drop(running);
            until(&server, "serving", PATIENT).await;
        } else {
            until(&server, "serving", PATIENT).await;
            assert!(jobs.is_closed(), "past the window the queue is closed: the job stops at its next step");
            drop(running);
        }
        assert_eq!(server.active().model_id(), "mock-a");
        assert_eq!(loader.loads(), ["mock-a"]);
        assert_eq!(loader.resident_at_load(), [0], "the image runner left the card before the text model loaded");
    }
}

/// 322g's note: an image runner that panics mid-job fails its model. The
/// request is answered `500 engine_error`; the model is unloaded as the
/// idle clock unloads one — its runner dropped, `GET /v1/models` saying
/// `unloaded` — and the next request loads it again and is served.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_image_runner_that_panics_unloads_its_model_and_the_next_request_loads_it_again() {
    let loader = MockLoader::new();
    let server = implicit_on(&loader, IMAGE, &[IMAGE]);
    let app = server.app();
    loader.panic_next_step();
    let body = json!({ "prompt": "a fault", "size": "1024x1024", "seed": 4 });
    let (status, failed) = send(&app, Method::POST, "/v1/images/generations", Some(body.clone())).await;
    assert_eq!((status, failed["error"]["code"].as_str()), (StatusCode::INTERNAL_SERVER_ERROR, Some("engine_error")), "{failed}");
    unloaded(&server).await;
    assert_eq!(loader.resident(), 0, "the failed runner was dropped");
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!((models["status"].as_str(), models["data"][0]["id"].as_str()), (Some("unloaded"), Some(IMAGE)), "{models}");

    let (status, made) = send(&app, Method::POST, "/v1/images/generations", Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{made}");
    assert_eq!(made["data"].as_array().map(Vec::len), Some(1));
    assert_eq!(loader.loads(), [IMAGE], "loaded again, once");
    assert_eq!(server.status().as_str(), "serving");
}

/// The same as [`implicit_off`] or [`implicit_on`], with the idle clock's
/// `model.idle_unload` at [`IDLE`].
fn idle(server: Server) -> Server {
    server.with_idle_unload(IDLE)
}

/// AC 1 with the image model unloaded: a text request naming a model the
/// server cannot switch to — implicit switching off, or a model it does not
/// know — is refused by name at once, as the loaded image model refuses it,
/// and never reloads the image model only for that.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn with_the_image_model_unloaded_a_text_request_naming_a_model_it_cannot_switch_to_loads_nothing() {
    for (server, named) in [
        (idle(implicit_off(&MockLoader::new(), IMAGE, &[IMAGE, "mock-a"])), "mock-a"),
        (idle(implicit_on(&MockLoader::new(), IMAGE, &[IMAGE, "mock-a"])), "mock-unknown"),
    ] {
        let _clock = server.spawn_idle_clock(TICK).expect("a clock");
        let app = server.app();
        unloaded(&server).await;
        for (uri, body) in text_routes(Some(named)) {
            let (status, refused) = send(&app, Method::POST, uri, Some(body)).await;
            let expected = if uri.starts_with("/v1/chat") || uri == "/v1/responses" { StatusCode::NOT_FOUND } else { StatusCode::UNPROCESSABLE_ENTITY };
            assert_eq!(status, expected, "{named} on {uri}: {refused}");
            assert_eq!(refused["error"]["code"], "model_not_found", "{named} on {uri}: {refused}");
            let message = refused["error"]["message"].as_str().unwrap();
            assert!(message.contains(named) && message.contains(IMAGE), "{named} on {uri}: {message}");
        }
        assert_eq!(server.status().as_str(), "unloaded", "{named}: nothing was loaded for a request it would refuse");
    }
}

// ── the production loader on the image artifact ───────────────────────────

/// A scratch directory removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("ignis-model-kind-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The image model's container packed from the converter's synthetic work
/// tree (the artifact crate's committed fixture), its sidecar beside it.
fn image_artifact(dir: &Path) -> PathBuf {
    use ignis_artifact::packer::{pack, PackOptions, PackOutcome};
    let work = Path::new(env!("CARGO_MANIFEST_DIR")).join("../artifact/tests/fixtures/qwen_image_work");
    let artifact = dir.join("synthetic-v2.ninfer");
    let mut options = PackOptions::qwen_image(work, artifact.clone());
    options.keep_work = true;
    options.accept_status = vec!["fixture".into()];
    options.header_bytes = 1 << 20;
    let outcome = pack(&options, &mut |_| {}).unwrap();
    assert!(matches!(outcome, PackOutcome::Finished { .. }), "{outcome:?}");
    artifact
}

fn start_options() -> ignis_server::config::Config {
    match ignis_server::config::resolve(&[], |_| None).expect("the defaults resolve") {
        ignis_server::config::ConfigOutcome::Config(config) => config,
        other => panic!("{other:?}"),
    }
}

/// The production loader reads the image artifact's kind off its header,
/// verifies it against its sidecar, fits the options to the image family
/// and loads it — on the fake runner in a build without `cuda`; with it, the
/// leaf's runner refuses the synthetic artifact, whose front end is a stub
/// (its tokenizer is `{}`), naming the model. A sidecar that does not match is
/// refused before anything loads.
#[tokio::test]
async fn the_artifact_loader_verifies_the_image_artifact_and_loads_its_runner() {
    let scratch = Scratch::new("loader");
    let artifact = image_artifact(&scratch.0);
    let loader = ArtifactLoader::new(start_options());
    let target = ModelSource { artifact: artifact.clone(), model: IMAGE.to_owned() };
    assert_eq!(loader.kind_of(&target), Some(ModelKind::Image));
    assert_eq!(loader.kind_of(&ModelSource { artifact: scratch.0.join("absent.ninfer"), model: "x".into() }), None);

    let prepared = loader.prepare(&target, None).expect("a verified image artifact prepares");
    assert_eq!(prepared.options().expect("the fitted options").model, IMAGE);
    let loaded = prepared.load();
    #[cfg(not(feature = "cuda"))]
    {
        let model = loaded.expect("the fake runner loads").model;
        assert_eq!((model.kind(), model.model_id()), (ModelKind::Image, IMAGE.to_owned()));
        assert_eq!(model.source, Some(target.clone()));
        assert!(model.engine().is_none());
        let image = model.image_model().expect("the image model");
        assert_eq!(image.plan().sample_sigmas.len(), 8);
        model.shut_down();
        assert!(image.with_runner(|runner| runner.plan().clone()).is_none(), "shut down, the runner is gone");
    }
    #[cfg(feature = "cuda")]
    assert!(loaded.err().expect("the synthetic front end does not load").contains("cannot load"));

    let sidecar = ignis_artifact::packer::sidecar_path(&artifact);
    let mut record: Value = serde_json::from_slice(&std::fs::read(&sidecar).unwrap()).unwrap();
    record["artifact"]["bytes"] = json!(record["artifact"]["bytes"].as_u64().unwrap() + 1);
    std::fs::write(&sidecar, record.to_string()).unwrap();
    let refused = loader.prepare(&target, None).err().expect("a sidecar that does not match is refused");
    assert!(refused.contains("checksum verification failed"), "{refused}");
}
