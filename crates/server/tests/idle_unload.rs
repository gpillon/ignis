//! Idle unload (spec model-switch/02, ADR 0049, GitHub #317) over the
//! mock-backed router: a model nobody uses is torn down after
//! `model.idle_unload` seconds, the server stays up reporting `unloaded` under
//! the same id, and the next request that needs the model loads it again and
//! is served. Loads are `support/switch.rs`'s mock loader, held inside a load
//! when a test needs one in flight; a request is held in flight by spec
//! server/05's gate or by a template held mid-render, never by a timer. The
//! clock ticks every 50 ms and the idle values are 1-2 s, so no test waits
//! minutes (AC 17). The log lines are `idle_unload_log.rs`'s, a binary of
//! their own.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_server::config::file::RealFiles;
use ignis_server::config::{resolve_with, Config, ConfigOutcome};
use ignis_server::config_http::ConfigState;
use ignis_server::decoder::TokenDecoder;
use ignis_server::model_switch::Switcher;
use ignis_server::template::{ChatMessage, RenderedPrompt, SimpleTemplateProvider, TemplateProvider, TemplateRejection};
use ignis_server::thinking::{ThinkingCapabilities, ThinkingOptions};
use ignis_server::{ActiveModel, ModelStatus, Server};
use serde_json::{json, Value};
use tower::ServiceExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::{source, MockLoader};

#[path = "support/responses.rs"]
mod responses;

/// Long enough that no test here reaches it unless it means to.
const PATIENT: Duration = Duration::from_secs(30);

/// The tests' clock: a twentieth of the one a server runs.
const TICK: Duration = Duration::from_millis(50);

/// The tests' `model.idle_unload`.
const IDLE: Duration = Duration::from_secs(1);

/// The known-models table naming each of `ids` at the artifact [`source`]
/// gives it.
fn known(ids: &[&str]) -> BTreeMap<String, PathBuf> {
    ids.iter().map(|id| (id.to_string(), source(id).artifact)).collect()
}

/// A server on `model` that unloads it after `idle`, and may switch to any of
/// `ids` by a request naming it.
fn idle_server(loader: &Arc<MockLoader>, model: ActiveModel, idle: Duration, ids: &[&str]) -> Server {
    Server::from_active(model)
        .with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT).with_known_models(known(ids)))
        .with_idle_unload(idle)
}

/// A config file of its own for one test, under the system temp directory.
fn config_file(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ignis-idle-unload-{test}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("ignis.config.yaml");
    std::fs::write(&path, "server:\n  request_timeout: 30\n").unwrap();
    path
}

/// The configuration `argv` starts with, over a config file of its own.
fn started(test: &str, argv: &[&str]) -> Config {
    let path = config_file(test);
    let mut args = vec!["--config".to_owned(), path.to_string_lossy().into_owned()];
    args.extend(argv.iter().map(|arg| arg.to_string()));
    match resolve_with(&args, |_| None, &RealFiles).expect("the start options resolve") {
        ConfigOutcome::Config(config) => config,
        other => panic!("{other:?}"),
    }
}

/// [`idle_server`] with a running configuration (`GET`/`PATCH /v1/config`),
/// its live knobs — `model.idle_unload` among them — taken from `config`.
fn configured_server(loader: &Arc<MockLoader>, config: Config) -> Server {
    let state = Arc::new(ConfigState::new(config.clone(), Arc::new(RealFiles)));
    let server = Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT))
        .with_config(state);
    server.apply_config(&config);
    server
}

async fn send(app: &axum::Router, method: Method, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    send_keyed(app, method, uri, body, None).await
}

async fn send_keyed(app: &axum::Router, method: Method, uri: &str, body: Option<Value>, key: Option<&str>) -> (StatusCode, Value) {
    let mut request = Request::builder().method(method).uri(uri).header(header::CONTENT_TYPE, "application/json");
    if let Some(key) = key {
        request = request.header(header::AUTHORIZATION, format!("Bearer {key}"));
    }
    let request = request.body(body.map_or_else(Body::empty, |body| Body::from(body.to_string()))).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body = serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, body)
}

fn chat(model: Option<&str>) -> Value {
    let mut body = json!({ "messages": [{ "role": "user", "content": "hello there" }], "max_tokens": 3 });
    if let Some(model) = model {
        body["model"] = json!(model);
    }
    body
}

/// A chat completion on its own task: what a client sends and waits on.
fn chat_task(app: &axum::Router, model: Option<&str>) -> tokio::task::JoinHandle<(StatusCode, Value)> {
    let app = app.clone();
    let body = chat(model);
    tokio::spawn(async move { send(&app, Method::POST, "/v1/chat/completions", Some(body)).await })
}

/// Wait until `server` reports `wanted` — read off the server, never through
/// a route — and say when it first did. Panics after `within`.
async fn until(server: &Server, wanted: &str, within: Duration) -> Instant {
    let deadline = Instant::now() + within;
    loop {
        if server.status().as_str() == wanted {
            return Instant::now();
        }
        assert!(Instant::now() < deadline, "the status never read {wanted}: {:?}", server.status());
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Wait (bounded) until `check` holds.
async fn until_true(what: &str, check: impl Fn() -> bool) {
    for _ in 0..2_000 {
        if check() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("never: {what}");
}

/// [`until`] `unloaded`, from a server's start, and the teardown after it —
/// through the release of the switch's slot, a moment after the scheduler
/// is dropped: a switch asked for before it is `409 switch_in_progress`.
async fn unloaded(server: &Server, loader: &MockLoader) {
    until(server, "unloaded", IDLE * 5).await;
    until_true("the unloaded model's scheduler was dropped", || loader.resident() == 0).await;
    until_true("the unload freed the switch's slot", || server.switcher.as_ref().unwrap().running().is_none()).await;
}

/// AC 2: with nobody using it, the model is torn down after `idle_unload`
/// seconds, within a tick; `GET /v1/models` names it as before, `unloaded`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_unused_model_is_unloaded_after_its_seconds_and_keeps_its_id() {
    let loader = MockLoader::new();
    let server = idle_server(&loader, loader.model("mock-a"), IDLE, &[]);
    let app = server.app();
    let began = Instant::now();
    server.usage.touch();
    let _clock = server.spawn_idle_clock(TICK).expect("a server with a switcher has a clock");

    let at = until(&server, "unloaded", IDLE * 5).await - began;
    assert!(at >= IDLE, "unloaded after {at:?}, before its {IDLE:?} were up");
    // Within a tick of its due time; a second tick and a fifth of a second
    // are the margin a loaded test machine needs to schedule the clock.
    let latest = IDLE + TICK * 2 + Duration::from_millis(200);
    assert!(at < latest, "unloaded after {at:?}: the clock ticks every {TICK:?}");
    let (status, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK, "the API stays up: {models}");
    assert_eq!(models["status"], "unloaded", "{models}");
    assert_eq!(models["data"][0]["id"], "mock-a", "the model is named as it was");
    until_true("the unloaded model's scheduler was dropped", || loader.resident() == 0).await;
    assert!(loader.loads().is_empty());
}

/// AC 1 over the wire: `GET /v1/config/fields` lists `model.idle_unload`,
/// patchable, with no reload, under its flag and environment variable.
#[tokio::test]
async fn the_fields_route_lists_idle_unload_patchable_with_no_reload() {
    let loader = MockLoader::new();
    let app = idle_server(&loader, loader.model("mock-a"), IDLE, &[]).app();
    let (status, fields) = send(&app, Method::GET, "/v1/config/fields", None).await;
    assert_eq!(status, StatusCode::OK, "{fields}");
    let field = fields["data"].as_array().unwrap().iter().find(|field| field["file_key"] == "model.idle_unload").expect("listed");
    assert_eq!((field["patchable"].clone(), field["reload_required"].clone()), (json!(true), json!(false)), "{field}");
    assert_eq!((field["flag"].clone(), field["env"].clone()), (json!("--model-idle-unload"), json!("IGNIS_MODEL_IDLE_UNLOAD")));
    assert_eq!(field["default"], 300, "{field}");
}

/// AC 3: zero never unloads, however long the clock runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn with_zero_the_model_is_never_unloaded() {
    let loader = MockLoader::new();
    let server = idle_server(&loader, loader.model("mock-a"), Duration::ZERO, &[]);
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    tokio::time::sleep(IDLE * 2).await;
    assert_eq!(*server.status(), ModelStatus::Serving);
    assert_eq!(loader.resident(), 1);
    // And a server built without a configuration runs with zero.
    let bare = Server::from_active(loader.model("mock-b"));
    assert_eq!(bare.live().idle_unload, Duration::ZERO);
}

/// AC 4: a stream held mid-decode past `idle_unload` keeps the model, and the
/// clock starts when it has ended.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_request_in_flight_keeps_the_model_and_the_clock_starts_when_it_ends() {
    let loader = MockLoader::new();
    let (model, gated, gate) = loader.gated_model("mock-a");
    gated.arm();
    drop(gated);
    let server = idle_server(&loader, model, IDLE, &[]);
    let app = server.app();
    let mut body = chat(None);
    body["stream"] = json!(true);
    let streaming = tokio::spawn({
        let app = app.clone();
        async move { send(&app, Method::POST, "/v1/chat/completions", Some(body)).await }
    });
    let gate = tokio::task::spawn_blocking(move || {
        gate.wait_entered();
        gate
    })
    .await
    .unwrap();
    // The clock starts with the stream already held: no race to admit it.
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");

    tokio::time::sleep(IDLE * 2).await;
    assert_eq!(*server.status(), ModelStatus::Serving, "a stream held mid-decode is use");
    let released = Instant::now();
    tokio::task::spawn_blocking(move || gate.release()).await.unwrap();
    let (status, streamed) = streaming.await.unwrap();
    assert_eq!(status, StatusCode::OK);
    assert!(streamed.as_str().unwrap().contains("[DONE]"), "{streamed}");
    let at = until(&server, "unloaded", IDLE * 5).await - released;
    assert!(at >= IDLE, "unloaded {at:?} after the stream was let go: the clock starts when it ends");
}

/// AC 4's other end: a client hanging up mid-stream ends its use as the
/// stream's end does — the count is released, and the model is unloaded once
/// it has gone unused.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_client_hanging_up_mid_stream_releases_its_use() {
    let loader = MockLoader::new();
    let (model, gated, gate) = loader.gated_model("mock-a");
    gated.arm();
    drop(gated);
    let server = idle_server(&loader, model, IDLE, &[]);
    let app = server.app();
    let mut body = chat(None);
    body["stream"] = json!(true);
    // The response's head is handed over before its body is read, so the
    // hang-up below provably drops a body mid-stream — the counted body's
    // own way of ending the use — not a handler still running.
    let (began, head) = tokio::sync::oneshot::channel();
    let streaming = tokio::spawn({
        let app = app.clone();
        async move {
            let request = Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap();
            let response = app.oneshot(request).await.unwrap();
            let _ = began.send(response.status());
            response.into_body().collect().await.map(|_| ())
        }
    });
    let gate = tokio::task::spawn_blocking(move || {
        gate.wait_entered();
        gate
    })
    .await
    .unwrap();
    let status = tokio::time::timeout(PATIENT, head).await.expect("the stream began").unwrap();
    assert_eq!(status, StatusCode::OK);
    assert_eq!(server.usage.in_flight(), 1, "the stream is in use");

    // The client goes away mid-stream: its response body is dropped unread.
    streaming.abort();
    assert!(streaming.await.unwrap_err().is_cancelled());
    until_true("the hang-up released the count", || server.usage.in_flight() == 0).await;
    tokio::task::spawn_blocking(move || gate.release()).await.unwrap();
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    until(&server, "unloaded", IDLE * 5).await;
}

/// AC 5: polling every route that reports state — the model, the catalog,
/// the configuration, the metrics — neither keeps the model loaded nor
/// loads it again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn polling_the_state_routes_neither_keeps_the_model_nor_loads_it_again() {
    let loader = MockLoader::new();
    let server = idle_server(&loader, loader.model("mock-a"), IDLE, &[]).with_metrics();
    let app = server.app();
    let metrics = server.metrics_app().expect("metrics are on");
    let began = Instant::now();
    server.usage.touch();
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    let poll = || async {
        for uri in ["/v1/models", "/v1/models/catalog", "/v1/config", "/v1/config/fields", "/v1/config/origins"] {
            send(&app, Method::GET, uri, None).await;
        }
        send(&metrics, Method::GET, "/metrics", None).await;
        tokio::time::sleep(TICK).await;
    };
    while !matches!(*server.status(), ModelStatus::Unloaded) {
        assert!(began.elapsed() < IDLE * 5, "never unloaded while polled");
        poll().await;
    }
    assert!(began.elapsed() >= IDLE);
    let polled = Instant::now();
    while polled.elapsed() < IDLE * 2 {
        poll().await;
    }
    assert_eq!(*server.status(), ModelStatus::Unloaded, "polling loads nothing");
    assert!(loader.loads().is_empty(), "{:?}", loader.loads());
}

/// A series' value in a `/metrics` scrape, if the scrape has it.
fn gauge(scrape: &str, series: &str) -> Option<u64> {
    scrape.lines().find_map(|line| line.strip_prefix(series)?.strip_prefix(' ')?.parse().ok())
}

/// The spec's engine audit: the card is free while unloaded, and `/metrics`
/// says so — what the unloaded model's load reserved, and what was occupied
/// of it, read zero (the disk tier's series are gone, as on a load without
/// it) until the next load writes its own.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn while_unloaded_the_metrics_show_nothing_reserved() {
    let loader = MockLoader::new();
    let reserved = ignis_server::metrics::LoadReservations {
        lines: ignis_core::VramLines { weights: 7 << 30, workspace: 1 << 30, ..Default::default() },
        budget_bytes: 30 << 30,
        kv_pool_pages: 1024,
        kv_page_bytes: 1 << 20,
        residency_bytes: 0,
        expert_cache_bytes: 0,
        condition_budget_bytes: 0,
        kv_ram_arena_bytes: 4 << 30,
        retained_slots: 4,
        retained_host_slots: 2,
        retained_host_bytes: 1 << 30,
        kv_disk_bytes: Some(8 << 30),
        flash_next: None,
        draft_positions: 0,
    };
    let server = idle_server(&loader, loader.model("mock-a"), IDLE, &[]).with_metrics().with_load_reservations(reserved);
    let metrics = server.metrics_app().expect("metrics are on");
    let scrape = || async { send(&metrics, Method::GET, "/metrics", None).await.1.as_str().expect("text").to_owned() };
    let series = [
        ("ignis_vram_reserved_bytes{line=\"weights\"}", 7 << 30),
        ("ignis_vram_reserved_bytes{line=\"workspace\"}", 1 << 30),
        ("ignis_vram_budget_bytes", 30 << 30),
        ("ignis_kv_pool_pages", 1024),
        ("ignis_kv_page_bytes", 1 << 20),
        ("ignis_kv_ram_arena_bytes{state=\"capacity\"}", 4 << 30),
        ("ignis_retained_slots{state=\"capacity\"}", 4),
        ("ignis_retained_host_slots", 2),
        ("ignis_retained_host_bytes", 1 << 30),
    ];
    let loaded = scrape().await;
    for (name, value) in series {
        assert_eq!(gauge(&loaded, name), Some(value), "{name} while loaded");
    }
    assert_eq!(gauge(&loaded, "ignis_kv_disk_bytes{state=\"capacity\"}"), Some(8 << 30));

    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    unloaded(&server, &loader).await;
    let unloaded = scrape().await;
    for (name, _) in series {
        assert_eq!(gauge(&unloaded, name), Some(0), "{name} while unloaded");
    }
    for name in ["ignis_kv_pool_used_pages", "ignis_kv_ram_arena_bytes{state=\"used\"}", "ignis_retained_slots{state=\"in_use\"}"] {
        assert_eq!(gauge(&unloaded, name), Some(0), "{name} while unloaded");
    }
    assert!(!unloaded.contains("ignis_kv_disk_bytes"), "no disk tier while nothing is loaded");
}

/// AC 6: a chat request while unloaded loads the model — `loading`
/// meanwhile — and is served; two arriving together share the one load.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_chat_request_while_unloaded_loads_the_model_and_two_together_share_one_load() {
    let loader = MockLoader::new();
    let server = idle_server(&loader, loader.model("mock-a"), IDLE, &[]);
    let app = server.app();
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    unloaded(&server, &loader).await;
    server.update_live(|live| live.idle_unload = Duration::ZERO);

    let release = loader.hold_next_load();
    let (first, second) = (chat_task(&app, None), chat_task(&app, Some("mock-a")));
    until(&server, "loading", PATIENT).await;
    until_true("both requests are admitted", || server.usage.in_flight() == 2).await;
    let (status, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(models["status"], "loading", "{models}");
    assert_eq!(models["switching"], json!({ "from": "mock-a", "to": "mock-a" }), "{models}");
    assert!(!first.is_finished() && !second.is_finished(), "both wait for the load");

    release.send(()).expect("the held load is waiting");
    for task in [first, second] {
        let (status, completion) = task.await.unwrap();
        assert_eq!(status, StatusCode::OK, "{completion}");
        assert_eq!(completion["model"], "mock-a");
        assert!(completion["choices"][0]["message"].is_object(), "{completion}");
    }
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!((models["status"].as_str(), models["data"][0]["id"].as_str()), (Some("serving"), Some("mock-a")));
    assert_eq!(loader.loads(), ["mock-a"], "one load for both");
    assert_eq!(loader.resident_at_load(), [0], "nothing was resident when it loaded");
}

/// AC 6 on Responses sockets opened before the unload: an open socket does
/// not keep the model, and a job on it loads the model again — `loading`
/// meanwhile — and is served; jobs on two sockets share the one load.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_job_on_a_socket_opened_before_the_unload_loads_the_model() {
    let loader = MockLoader::new();
    let server = idle_server(&loader, loader.model("mock-a"), IDLE, &[]);
    let app = server.app();
    // The serve loop runs the server's own clock.
    let live = responses::live(server.clone()).await;
    let (mut first, mut second) = (responses::socket(&live).await, responses::socket(&live).await);
    until(&server, "unloaded", IDLE * 6).await;
    server.update_live(|live| live.idle_unload = Duration::ZERO);

    let release = loader.hold_next_load();
    let job = json!({ "type": "response.create", "input": "hello there", "max_output_tokens": 3, "enable_thinking": false });
    responses::send(&mut first, job.clone()).await;
    responses::send(&mut second, job).await;
    until(&server, "loading", PATIENT).await;
    until_true("both jobs are admitted", || server.usage.in_flight() == 2).await;
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!(models["status"], "loading", "{models}");
    release.send(()).expect("the held load is waiting");

    for socket in [&mut first, &mut second] {
        let events = responses::events_until(socket, responses::terminal).await;
        let last = events.last().expect("a terminal event");
        assert!(matches!(last["type"].as_str(), Some("response.completed" | "response.incomplete")), "{events:?}");
        assert_eq!(last["response"]["model"], "mock-a");
    }
    assert_eq!(*server.status(), ModelStatus::Serving);
    assert_eq!(loader.loads(), ["mock-a"], "one load for both jobs");
}

/// AC 7: a load out of `unloaded` that fails leaves `failed`, answers the
/// requests that waited on it `503` with its reason — one attempt for both —
/// and a later request tries again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failed_load_answers_its_waiters_503_and_a_later_request_tries_again() {
    let loader = MockLoader::new();
    let server = idle_server(&loader, loader.model("mock-a"), IDLE, &[]);
    let app = server.app();
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    unloaded(&server, &loader).await;
    server.update_live(|live| live.idle_unload = Duration::ZERO);
    loader.break_load("mock-a");

    let release = loader.hold_next_load();
    let (first, second) = (chat_task(&app, None), chat_task(&app, None));
    until(&server, "loading", PATIENT).await;
    until_true("both requests are admitted", || server.usage.in_flight() == 2).await;
    release.send(()).expect("the held load is waiting");
    for task in [first, second] {
        let (status, refused) = task.await.unwrap();
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{refused}");
        assert_eq!(refused["error"]["code"], "model_load_failed", "{refused}");
        assert!(refused["error"]["message"].as_str().unwrap().contains("simulated kernel load error"), "{refused}");
    }
    let (status, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(models["status"], "failed", "{models}");
    assert!(models["reason"].as_str().unwrap().contains("simulated kernel load error"), "{models}");
    assert_eq!(loader.loads(), ["mock-a"], "one attempt, shared");

    loader.mend("mock-a");
    let (status, completion) = chat_task(&app, None).await.unwrap();
    assert_eq!(status, StatusCode::OK, "a later request loads again: {completion}");
    assert_eq!(*server.status(), ModelStatus::Serving);
    assert_eq!(loader.loads(), ["mock-a", "mock-a"]);
}

/// AC 7 at prepare: a load out of `unloaded` refused before it reaches the
/// GPU leaves `failed` with the reason, the request waiting on it gets `503
/// model_load_failed`, nothing was loaded, and no switch is recorded.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_load_refused_at_prepare_answers_503_and_leaves_failed() {
    let loader = MockLoader::new();
    let server = idle_server(&loader, loader.model("mock-a"), IDLE, &[]);
    let app = server.app();
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    unloaded(&server, &loader).await;
    server.update_live(|live| live.idle_unload = Duration::ZERO);
    loader.refuse("mock-a");

    let (status, refused) = chat_task(&app, None).await.unwrap();
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{refused}");
    assert_eq!(refused["error"]["code"], "model_load_failed", "{refused}");
    assert!(refused["error"]["message"].as_str().unwrap().contains("refused at prepare"), "{refused}");
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!(models["status"], "failed", "{models}");
    assert!(models["reason"].as_str().unwrap().contains("refused at prepare"), "{models}");
    assert!(models["last_switch"].is_null(), "a request's load is not a switch: {models}");
    assert_eq!(loader.prepared_artifacts(), [source("mock-a").artifact]);
    assert!(loader.loads().is_empty(), "{:?}", loader.loads());
}

/// AC 11 refused at prepare: a switch out of `unloaded` whose target is
/// refused before anything loads leaves the server `unloaded` as it was —
/// the unloaded model still the one a request loads — and `last_switch` says
/// why.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_switch_out_of_unloaded_refused_at_prepare_leaves_it_unloaded() {
    let loader = MockLoader::new();
    let server = idle_server(&loader, loader.model("mock-a"), IDLE, &[]);
    let app = server.app();
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    unloaded(&server, &loader).await;
    server.update_live(|live| live.idle_unload = Duration::ZERO);
    loader.refuse("mock-b");

    let switch = json!({ "artifact": "mock-b.ninfer", "model": "mock-b" });
    let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(switch)).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    until(&server, "unloaded", PATIENT).await;
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!((models["status"].as_str(), models["data"][0]["id"].as_str()), (Some("unloaded"), Some("mock-a")), "{models}");
    let last = &models["last_switch"];
    assert_eq!((last["from"].as_str(), last["to"].as_str(), last["outcome"].as_str()), (Some("mock-a"), Some("mock-b"), Some("refused")), "{models}");
    assert!(last["reason"].as_str().unwrap().contains("refused at prepare"), "{models}");
    assert!(loader.loads().is_empty(), "{:?}", loader.loads());

    let (status, completion) = chat_task(&app, None).await.unwrap();
    assert_eq!(status, StatusCode::OK, "{completion}");
    assert_eq!(completion["model"], "mock-a");
    assert_eq!(loader.loads(), ["mock-a"]);
}

/// AC 11 failing at load: a switch out of `unloaded` whose target fails on
/// the GPU leaves `failed` — there is nothing to reload — and is recorded as
/// one; the next request loads the unloaded model again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_switch_out_of_unloaded_that_fails_to_load_leaves_failed_and_a_request_loads_again() {
    let loader = MockLoader::new();
    let server = idle_server(&loader, loader.model("mock-a"), IDLE, &[]);
    let app = server.app();
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    unloaded(&server, &loader).await;
    server.update_live(|live| live.idle_unload = Duration::ZERO);
    loader.break_load("mock-b");

    let switch = json!({ "artifact": "mock-b.ninfer", "model": "mock-b" });
    let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(switch)).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    until(&server, "failed", PATIENT).await;
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert!(models["reason"].as_str().unwrap().contains("simulated kernel load error"), "{models}");
    let last = &models["last_switch"];
    assert_eq!((last["from"].as_str(), last["to"].as_str(), last["outcome"].as_str()), (Some("mock-a"), Some("mock-b"), Some("failed")), "{models}");
    assert_eq!(loader.loads(), ["mock-b"], "no reload of a model that was not resident");

    let (status, completion) = chat_task(&app, None).await.unwrap();
    assert_eq!(status, StatusCode::OK, "{completion}");
    assert_eq!(completion["model"], "mock-a");
    assert_eq!(loader.loads(), ["mock-b", "mock-a"]);
}

/// AC 7: after a failed load out of `unloaded`, a switch is a way back too.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn after_a_failed_load_a_switch_loads_a_model() {
    let loader = MockLoader::new();
    let server = idle_server(&loader, loader.model("mock-a"), IDLE, &[]);
    let app = server.app();
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    unloaded(&server, &loader).await;
    server.update_live(|live| live.idle_unload = Duration::ZERO);
    loader.break_load("mock-a");
    let (status, _) = chat_task(&app, None).await.unwrap();
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(matches!(*server.status(), ModelStatus::Failed { .. }));

    let switch = json!({ "artifact": "mock-b.ninfer", "model": "mock-b" });
    let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(switch)).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    until(&server, "serving", PATIENT).await;
    assert_eq!(server.active().model_id(), "mock-b");
    assert_eq!(loader.loads(), ["mock-a", "mock-b"]);
}

/// AC 8: a request without the key, while unloaded, is refused `401` and
/// loads nothing; with it, the model loads.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_unauthenticated_request_while_unloaded_gets_401_and_loads_nothing() {
    let loader = MockLoader::new();
    let server = idle_server(&loader, loader.model("mock-a"), IDLE, &[])
        .with_api_key(ignis_server::config::ApiKey::new("sk-test"));
    let app = server.app();
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    unloaded(&server, &loader).await;
    server.update_live(|live| live.idle_unload = Duration::ZERO);

    for key in [None, Some("sk-wrong")] {
        let (status, refused) = send_keyed(&app, Method::POST, "/v1/chat/completions", Some(chat(None)), key).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{key:?}: {refused}");
    }
    tokio::time::sleep(TICK * 4).await;
    assert_eq!(*server.status(), ModelStatus::Unloaded);
    assert!(loader.loads().is_empty(), "{:?}", loader.loads());

    let (status, completion) = send_keyed(&app, Method::POST, "/v1/chat/completions", Some(chat(None)), Some("sk-test")).await;
    assert_eq!(status, StatusCode::OK, "{completion}");
    assert_eq!(loader.loads(), ["mock-a"]);
}

/// A decision whose body does not parse, while unloaded, is refused `422` as
/// before and loads nothing: only a request that can be served may cost a
/// load. `/v1/systemone` is the same handler under its other name.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_malformed_decision_while_unloaded_gets_422_and_loads_nothing() {
    let loader = MockLoader::new();
    let server = idle_server(&loader, loader.model("mock-a"), IDLE, &[]);
    let app = server.app();
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    unloaded(&server, &loader).await;
    server.update_live(|live| live.idle_unload = Duration::ZERO);

    for uri in ["/v1/decide", "/v1/systemone"] {
        let (status, refused) = send(&app, Method::POST, uri, Some(json!({ "questions": 7 }))).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{uri}: {refused}");
    }
    tokio::time::sleep(TICK * 4).await;
    assert_eq!(*server.status(), ModelStatus::Unloaded);
    assert!(loader.loads().is_empty(), "{:?}", loader.loads());
}

/// AC 9: a request naming another known model while unloaded loads that
/// model, and only it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_request_naming_another_known_model_while_unloaded_loads_that_model_only() {
    let loader = MockLoader::new();
    let server = idle_server(&loader, loader.model("mock-a"), IDLE, &["mock-a", "mock-b"]);
    let app = server.app();
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    unloaded(&server, &loader).await;
    server.update_live(|live| live.idle_unload = Duration::ZERO);

    let (status, completion) = chat_task(&app, Some("mock-b@agent")).await.unwrap();
    assert_eq!(status, StatusCode::OK, "{completion}");
    assert_eq!(completion["model"], "mock-b");
    assert_eq!(loader.loads(), ["mock-b"], "one load, of the model named");
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!((models["status"].as_str(), models["data"][0]["id"].as_str()), (Some("serving"), Some("mock-b")));
}

/// A template that holds its first render until released: a request
/// admitted and not yet submitted, as a slow render or media decode keeps
/// one (AC 10).
struct HeldTemplate {
    entered: Mutex<Option<Sender<()>>>,
    release: Mutex<Option<Receiver<()>>>,
}

impl HeldTemplate {
    /// The template, the signal its render began, and its release.
    fn new() -> (Self, Receiver<()>, Sender<()>) {
        let (entered_tx, entered) = channel();
        let (release, release_rx) = channel();
        (Self { entered: Mutex::new(Some(entered_tx)), release: Mutex::new(Some(release_rx)) }, entered, release)
    }
}

impl TemplateProvider for HeldTemplate {
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[Value],
    ) -> Result<RenderedPrompt, TemplateRejection> {
        if let Some(entered) = self.entered.lock().unwrap().take() {
            let _ = entered.send(());
        }
        let release = self.release.lock().unwrap().take();
        if let Some(release) = release {
            let _ = release.recv_timeout(PATIENT);
        }
        SimpleTemplateProvider.apply_chat_template(messages, options, tools)
    }

    fn render_tokens(&self, tokens: &[ignis_core::TokenId]) -> String {
        SimpleTemplateProvider.render_tokens(tokens)
    }

    fn thinking_capabilities(&self) -> ThinkingCapabilities {
        SimpleTemplateProvider.thinking_capabilities()
    }

    fn token_decoder(&self) -> Box<dyn TokenDecoder> {
        SimpleTemplateProvider.token_decoder()
    }

    fn decoder_starts_in_reasoning(&self, thinking: &ThinkingOptions) -> bool {
        SimpleTemplateProvider.decoder_starts_in_reasoning(thinking)
    }
}

/// AC 10: a request admitted but not yet submitted to the engine — held in
/// its template render, the engine empty meanwhile — is not unloaded under,
/// and the clock starts when its response has ended.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_request_admitted_but_not_yet_submitted_is_not_unloaded_under() {
    let loader = MockLoader::new();
    let (template, entered, release) = HeldTemplate::new();
    let mut model = loader.model("mock-a");
    model.template = Arc::new(template);
    let server = idle_server(&loader, model, IDLE, &[]);
    let app = server.app();
    let pending = chat_task(&app, None);
    tokio::task::spawn_blocking(move || entered.recv_timeout(PATIENT).expect("the render began")).await.unwrap();
    // The clock starts with the request already admitted: no race to admit it.
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");

    tokio::time::sleep(IDLE * 2).await;
    let counters = server.active().text_engine().interval_counters();
    assert_eq!(counters.waiting + counters.running, 0, "the engine has nothing yet: only the count holds the model");
    assert_eq!(*server.status(), ModelStatus::Serving, "an admitted request is use");
    let released = Instant::now();
    release.send(()).expect("the render is waiting");
    let (status, completion) = pending.await.unwrap();
    assert_eq!(status, StatusCode::OK, "{completion}");
    let at = until(&server, "unloaded", IDLE * 5).await - released;
    assert!(at >= IDLE, "unloaded {at:?} after the render was let go");
}

/// AC 11: `POST /v1/models/switch` while unloaded loads the named model —
/// the unloaded one's own id included — with nothing to drain or tear down.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_switch_while_unloaded_loads_the_named_model_the_same_id_included() {
    for target in ["mock-a", "mock-b"] {
        let loader = MockLoader::new();
        let server = idle_server(&loader, loader.model("mock-a"), IDLE, &[]);
        let app = server.app();
        let _clock = server.spawn_idle_clock(TICK).expect("a clock");
        unloaded(&server, &loader).await;
        server.update_live(|live| live.idle_unload = Duration::ZERO);

        let release = loader.hold_next_load();
        let switch = json!({ "artifact": format!("{target}.ninfer"), "model": target });
        let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(switch)).await;
        assert_eq!(status, StatusCode::ACCEPTED, "{target}: {accepted}");
        let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
        assert_eq!(models["status"], "loading", "{target}: {models}");
        release.send(()).expect("the held load is waiting");
        until(&server, "serving", PATIENT).await;
        assert_eq!(server.active().model_id(), target);
        assert_eq!(loader.loads(), [target], "{target}");
        assert_eq!(loader.resident_at_load(), [0], "{target}");
        let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
        assert_eq!(models["last_switch"], json!({ "from": "mock-a", "to": target, "outcome": "switched" }), "an explicit switch is one");
    }
}

/// AC 11: a `PATCH /v1/config` that needs a reload, while unloaded, loads
/// the model with the patched configuration.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_reloading_patch_while_unloaded_loads_the_model_with_the_new_configuration() {
    let loader = MockLoader::new();
    let server = configured_server(&loader, started("reload", &["--model-idle-unload", "1"]));
    let app = server.app();
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    unloaded(&server, &loader).await;

    let body = json!({ "reuse": { "kv_host_pool_bytes": "3G" }, "model": { "idle_unload": 0 } });
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(body)).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{applied}");
    assert_eq!(applied["applied"], "reload");
    until(&server, "serving", PATIENT).await;
    let handed = loader.prepared_options();
    let options = handed.last().unwrap().as_ref().expect("the reload hands its patched config to the loader");
    assert_eq!(options.host_pool_bytes, 3 << 30);
    assert_eq!(loader.loads(), ["mock-a"]);
    assert_eq!(server.live().idle_unload, Duration::ZERO, "the patch's live field landed with the reload");
}

/// AC 12: an unload due while a switch runs waits for it — the two never
/// overlap — and the clock starts again once the switch's model serves.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_unload_due_while_a_switch_runs_waits_for_it() {
    let loader = MockLoader::new();
    let server = idle_server(&loader, loader.model("mock-a"), IDLE, &[]);
    let app = server.app();
    let release = loader.hold_next_load();
    let switch = json!({ "artifact": "mock-b.ninfer", "model": "mock-b" });
    let (status, _) = send(&app, Method::POST, "/v1/models/switch", Some(switch)).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    // The clock starts with the switch already holding the slot.
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");

    tokio::time::sleep(IDLE * 2).await;
    assert!(matches!(*server.status(), ModelStatus::Switching { .. }), "no unload under a switch: {:?}", server.status());
    let released = Instant::now();
    release.send(()).expect("the held load is waiting");
    until(&server, "serving", PATIENT).await;
    let at = until(&server, "unloaded", IDLE * 5).await - released;
    assert!(at >= IDLE, "unloaded {at:?} after the switch's load was let go");
    assert_eq!(server.active().model_id(), "mock-b", "the switch's model was the one unloaded");
    assert_eq!(loader.loads(), ["mock-b"]);
}

/// AC 12: an unload due while a reload (`PATCH /v1/config`) runs waits for
/// it, and the clock starts again once the reloaded model serves.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_unload_due_while_a_reload_runs_waits_for_it() {
    let loader = MockLoader::new();
    let server = configured_server(&loader, started("reload-runs", &["--model-idle-unload", "1"]));
    let app = server.app();
    let release = loader.hold_next_load();
    let body = json!({ "reuse": { "kv_host_pool_bytes": "3G" } });
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(body)).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{applied}");
    // The clock starts with the reload already holding the slot.
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");

    tokio::time::sleep(IDLE * 2).await;
    assert!(matches!(*server.status(), ModelStatus::Switching { .. }), "no unload under a reload: {:?}", server.status());
    let released = Instant::now();
    release.send(()).expect("the held load is waiting");
    until(&server, "serving", PATIENT).await;
    let at = until(&server, "unloaded", IDLE * 5).await - released;
    assert!(at >= IDLE, "unloaded {at:?} after the reload's load was let go");
    assert_eq!(loader.loads(), ["mock-a"]);
}

/// AC 12: nothing unloads under the load a request began out of `unloaded`,
/// however long it takes, and the clock starts again once the request it
/// served has ended.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_unload_overlaps_the_load_a_request_began() {
    let loader = MockLoader::new();
    let server = idle_server(&loader, loader.model("mock-a"), IDLE, &[]);
    let app = server.app();
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    unloaded(&server, &loader).await;

    let release = loader.hold_next_load();
    let pending = chat_task(&app, None);
    until(&server, "loading", PATIENT).await;
    tokio::time::sleep(IDLE * 2).await;
    assert_eq!(*server.status(), ModelStatus::Loading, "no unload under a load");
    release.send(()).expect("the held load is waiting");
    let (status, completion) = pending.await.unwrap();
    assert_eq!(status, StatusCode::OK, "{completion}");
    let served = Instant::now();
    let at = until(&server, "unloaded", IDLE * 5).await - served;
    assert!(at >= IDLE - TICK, "unloaded {at:?} after the request it loaded for was served");
    assert_eq!(loader.loads(), ["mock-a"]);
}

/// A compute whose drop waits for the test: the model thread drops it with
/// its scheduler when the model is torn down, so a teardown is held open, as
/// a GPU one lasting seconds is, with nothing in flight.
struct HeldTeardown {
    inner: ignis_core::MockCompute,
    entered: Mutex<Option<Sender<()>>>,
    release: Mutex<Option<Receiver<()>>>,
}

impl HeldTeardown {
    /// The compute, the signal its teardown began, and its release. Hand the
    /// compute over whole: a copy kept here would keep it from being dropped.
    fn new() -> (Arc<Self>, Receiver<()>, Sender<()>) {
        let (entered_tx, entered) = channel();
        let (release, release_rx) = channel();
        let compute = Self {
            inner: ignis_core::MockCompute::new(),
            entered: Mutex::new(Some(entered_tx)),
            release: Mutex::new(Some(release_rx)),
        };
        (Arc::new(compute), entered, release)
    }
}

impl ignis_core::Compute for HeldTeardown {
    fn prefill_step(&self, jobs: &[ignis_core::PrefillJob]) -> Result<Vec<ignis_core::PrefillOutcome>, ignis_core::ComputeError> {
        self.inner.prefill_step(jobs)
    }

    fn decode_step(&self, jobs: &[ignis_core::DecodeJob]) -> Result<Vec<ignis_core::DecodeOutcome>, ignis_core::ComputeError> {
        self.inner.decode_step(jobs)
    }
}

impl Drop for HeldTeardown {
    fn drop(&mut self) {
        if let Some(entered) = self.entered.get_mut().unwrap().take() {
            let _ = entered.send(());
        }
        if let Some(release) = self.release.get_mut().unwrap().take() {
            let _ = release.recv_timeout(PATIENT);
        }
    }
}

/// AC 12 with the clock stopped mid-teardown, as the serve loop ending under
/// it stops it: the unload's teardown and the slot it holds end together, so
/// a request arriving meanwhile loads the model only once the old one is
/// gone — never a load beside a teardown — and a switch meanwhile is refused
/// `409` naming the unload.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_unload_keeps_its_slot_until_its_teardown_ends_even_when_the_clock_stops() {
    let loader = MockLoader::new();
    let (compute, entered, release) = HeldTeardown::new();
    let server = idle_server(&loader, loader.model_over("mock-a", compute), IDLE, &[]);
    let app = server.app();
    let clock = server.spawn_idle_clock(TICK).expect("a clock");
    tokio::task::spawn_blocking(move || entered.recv_timeout(PATIENT).expect("the teardown began")).await.unwrap();
    drop(clock);
    server.update_live(|live| live.idle_unload = Duration::ZERO);

    let pending = chat_task(&app, None);
    tokio::time::sleep(TICK * 4).await;
    assert!(loader.loads().is_empty(), "a load began beside the teardown: {:?}", loader.loads());
    assert_eq!(*server.status(), ModelStatus::Unloaded);
    assert_eq!(server.usage.in_flight(), 1, "the request is admitted");
    assert!(!pending.is_finished(), "the request waits for the teardown, then for its load");
    let switch = json!({ "artifact": "mock-b.ninfer", "model": "mock-b" });
    let (status, refused) = send(&app, Method::POST, "/v1/models/switch", Some(switch)).await;
    assert_eq!(status, StatusCode::CONFLICT, "{refused}");
    assert_eq!(refused["error"]["code"], "switch_in_progress", "{refused}");
    assert!(refused["error"]["message"].as_str().unwrap().contains("mock-a is being unloaded"), "{refused}");

    release.send(()).expect("the teardown is waiting");
    let (status, completion) = pending.await.unwrap();
    assert_eq!(status, StatusCode::OK, "{completion}");
    assert_eq!(loader.loads(), ["mock-a"]);
}

/// GitHub #329: what the serve exit leaves to the process end is a load,
/// never an unload. An unload still tearing down when serving ends is waited
/// for — a bounded join, the one the exit itself runs — and the serve ends
/// drained, as with nothing loaded.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_unload_tearing_down_when_serving_ends_is_waited_for() {
    let loader = MockLoader::new();
    let (compute, entered, release) = HeldTeardown::new();
    let server = idle_server(&loader, loader.model_over("mock-a", compute), IDLE, &[]);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    // The serve's own clock unloads the model nobody uses.
    let mut serving = tokio::spawn(server.clone().serve_on_until(listener, None, async {
        let _ = stopped.await;
    }));
    tokio::task::spawn_blocking(move || entered.recv_timeout(PATIENT).expect("the teardown began")).await.unwrap();

    stop.send(()).unwrap();
    assert!(tokio::time::timeout(TICK * 4, &mut serving).await.is_err(), "the serve returned mid-teardown");
    release.send(()).expect("the teardown is waiting");
    let served = tokio::time::timeout(PATIENT, serving).await.expect("the serve returned").unwrap().expect("serve");
    assert_eq!(served, ignis_server::Served::Drained);
    assert_eq!(*server.status(), ModelStatus::Unloaded);
}

/// AC 13: a live `PATCH` of `model.idle_unload` takes effect with no reload:
/// 0 → 1 unloads about a second later, the clock starting at the change.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_live_patch_turning_idle_unload_on_unloads_after_the_new_seconds() {
    let loader = MockLoader::new();
    let server = configured_server(&loader, started("on", &["--model-idle-unload", "0"]));
    let app = server.app();
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    tokio::time::sleep(IDLE * 2).await;
    assert_eq!(*server.status(), ModelStatus::Serving);

    let patched = Instant::now();
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "model": { "idle_unload": 1 } }))).await;
    assert_eq!(status, StatusCode::OK, "{applied}");
    assert_eq!(applied["applied"], "live");
    assert!(loader.prepared_options().is_empty(), "no reload");
    let at = until(&server, "unloaded", IDLE * 5).await - patched;
    assert!(at >= IDLE - TICK, "unloaded {at:?} after the change: the clock starts then, not at a stale last use");
}

/// AC 13: N → 0 stops an unload that was due.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_live_patch_turning_idle_unload_off_stops_a_pending_unload() {
    let loader = MockLoader::new();
    let server = configured_server(&loader, started("off", &["--model-idle-unload", "3"]));
    let app = server.app();
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    tokio::time::sleep(IDLE).await;
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "model": { "idle_unload": 0 } }))).await;
    assert_eq!(status, StatusCode::OK, "{applied}");
    assert_eq!(*server.status(), ModelStatus::Serving, "changed before it was due");
    tokio::time::sleep(IDLE * 3).await;
    assert_eq!(*server.status(), ModelStatus::Serving);
    assert_eq!(loader.resident(), 1);
}

/// AC 14: `last_switch` tells what the last switch did — neither an unload
/// nor the load a request begins after it changes it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn neither_the_unload_nor_the_load_after_it_changes_last_switch() {
    let loader = MockLoader::new();
    let server = idle_server(&loader, loader.model("mock-a"), IDLE, &[]);
    let app = server.app();
    let switch = json!({ "artifact": "mock-b.ninfer", "model": "mock-b" });
    let (status, _) = send(&app, Method::POST, "/v1/models/switch", Some(switch)).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    until(&server, "serving", PATIENT).await;
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    let switched = json!({ "from": "mock-a", "to": "mock-b", "outcome": "switched" });
    assert_eq!(models["last_switch"], switched);

    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    unloaded(&server, &loader).await;
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!(models["last_switch"], switched, "after the unload: {models}");
    server.update_live(|live| live.idle_unload = Duration::ZERO);
    let (status, _) = chat_task(&app, None).await.unwrap();
    assert_eq!(status, StatusCode::OK);
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!(models["last_switch"], switched, "after the load: {models}");
    assert_eq!(loader.loads(), ["mock-b", "mock-b"]);
}

/// The Server view's "Load" (spec playground/05) on the unloaded model: the
/// catalog lists no model as loaded, and the unloaded one as switchable.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_catalog_offers_the_unloaded_model_to_load() {
    let loader = MockLoader::new();
    // A switch by id resolves to a file: the start model's, on disk.
    let artifact = config_file("catalog").with_file_name("mock-a.ninfer");
    std::fs::write(&artifact, b"mock").unwrap();
    let switcher = Switcher::new(Arc::clone(&loader) as _, PATIENT)
        .with_known_models(BTreeMap::from([("mock-a".to_owned(), artifact)]));
    let server = Server::from_active(loader.model("mock-a")).with_switcher(switcher).with_idle_unload(IDLE);
    let app = server.app();
    let (_, catalog) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    let row = |catalog: &Value| catalog["data"].as_array().unwrap().iter().find(|row| row["id"] == "mock-a").cloned().unwrap();
    assert_eq!((row(&catalog)["loaded"].clone(), row(&catalog)["switchable"].clone()), (json!(true), json!(false)), "{catalog}");
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    unloaded(&server, &loader).await;
    let (_, catalog) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!((row(&catalog)["loaded"].clone(), row(&catalog)["switchable"].clone()), (json!(false), json!(true)), "{catalog}");
}


/// Manual lifecycle calls reuse the idle teardown and reload machinery.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn explicit_unload_is_idempotent_and_load_warms_without_generation() {
    let loader = MockLoader::new();
    let server = idle_server(&loader, loader.model("mock-a"), Duration::ZERO, &[]);
    let app = server.app();
    for _ in 0..2 {
        let (status, body) = send(&app, Method::POST, "/v1/ignis/models/unload", Some(json!({"model":"mock-a"}))).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["status"], "unloaded");
        assert_eq!(loader.resident(), 0);
    }
    let (status, entry) = send(&app, Method::GET, "/v1/models/mock-a", None).await;
    assert_eq!(status, StatusCode::OK, "{entry}");
    assert_eq!(entry["id"], "mock-a");
    assert!(loader.loads().is_empty(), "metadata must not reload");
    for _ in 0..2 {
        let (status, body) = send(&app, Method::POST, "/v1/ignis/models/load", Some(json!({"model":"mock-a"}))).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["status"], "serving");
        assert_eq!(*server.status(), ModelStatus::Serving);
    }
    assert_eq!(loader.loads(), vec!["mock-a"]);
    let (status, body) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(None))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn explicit_unload_refuses_an_admitted_request_and_unknown_ids() {
    let loader = MockLoader::new();
    let server = idle_server(&loader, loader.model("mock-a"), Duration::ZERO, &[]);
    let app = server.app();
    let in_use = server.usage.admit();
    let (status, body) = send(&app, Method::POST, "/v1/ignis/models/unload", Some(json!({"model":"mock-a"}))).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(*server.status(), ModelStatus::Serving);
    drop(in_use);
    for path in ["/v1/ignis/models/load", "/v1/ignis/models/unload"] {
        let (status, body) = send(&app, Method::POST, path, Some(json!({"model":"wrong"}))).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    }
    let (status, body) = send(&app, Method::GET, "/v1/models/wrong", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(loader.resident(), 1);
}

#[tokio::test]
async fn lifecycle_requires_authentication_and_rejects_unknown_body_fields() {
    let loader = MockLoader::new();
    let server = idle_server(&loader, loader.model("mock-a"), Duration::ZERO, &[])
        .with_api_key(ignis_server::config::ApiKey::new("sk-test"));
    let app = server.app();
    for path in ["/v1/ignis/models/load", "/v1/ignis/models/unload"] {
        let (status, body) = send(&app, Method::POST, path, Some(json!({"model":"mock-a"}))).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{body}");
        let (status, body) = send_keyed(&app, Method::POST, path, Some(json!({"model":"mock-a","force":true})), Some("sk-test")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    }
    let (status, body) = send(&app, Method::GET, "/v1/models/mock-a", None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "{body}");
    assert_eq!(loader.resident(), 1);
}


#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn explicit_unload_never_cancels_a_running_stream() {
    let loader = MockLoader::new();
    let (model, gated, gate) = loader.gated_model("mock-a");
    gated.arm();
    drop(gated);
    let server = idle_server(&loader, model, Duration::ZERO, &[]);
    let app = server.app();
    let mut body = chat(None);
    body["stream"] = json!(true);
    let streaming = tokio::spawn({
        let app = app.clone();
        async move { send(&app, Method::POST, "/v1/chat/completions", Some(body)).await }
    });
    let gate = tokio::task::spawn_blocking(move || { gate.wait_entered(); gate }).await.unwrap();
    let (status, body) = send(&app, Method::POST, "/v1/ignis/models/unload", Some(json!({"model":"mock-a"}))).await;
    // Release the held work before assertions, so a failed check cannot strand the test.
    tokio::task::spawn_blocking(move || gate.release()).await.unwrap();
    let (stream_status, stream_body) = streaming.await.unwrap();
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(stream_status, StatusCode::OK);
    assert!(stream_body.as_str().unwrap().contains("[DONE]"));
    assert_eq!(*server.status(), ModelStatus::Serving);
}
