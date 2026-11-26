//! What an image request waits on before its job (spec image-model/05
//! §Streaming, §Prompt enhancement, GitHub #322): the load of an unloaded
//! model and the switch to the image model, streamed as `loading` and
//! `switching` phases ahead of the job's own; the visit those switches
//! make for prompt enhancement; and prompt enhancement itself (AC 9). The
//! models are
//! `support/switch.rs`'s mock loader's: the text models on `MockCompute`,
//! the image model on `FakeImageRunner`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_core::compute::ModelKind;
use ignis_core::MockCompute;
use ignis_server::images::enhance::Task;
use ignis_server::model_switch::{visit, LastSwitch, SwitchOutcome, SwitchRefusal, SwitchStarted, SwitchStep, Switcher};
use ignis_server::template::{SimpleTemplateProvider, TemplateProvider};
use ignis_server::thinking::ThinkingOptions;
use ignis_server::{ModelStatus, Server};
use serde_json::{json, Value};
use tower::ServiceExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::{source, MockLoader};

#[path = "support/media.rs"]
mod media_support;

/// The image model's served id: its family's own.
const IMAGE: &str = "ignis-image-2.1-turbo";

const PATIENT: Duration = Duration::from_secs(30);

/// The idle clock's tick and `model.idle_unload`, as `idle_unload.rs` runs
/// them.
const TICK: Duration = Duration::from_millis(50);
const IDLE: Duration = Duration::from_secs(1);

/// The known-models table naming each of `ids` at the artifact [`source`]
/// gives it.
fn known(ids: &[&str]) -> BTreeMap<String, PathBuf> {
    ids.iter().map(|id| (id.to_string(), source(id).artifact)).collect()
}

/// A server on `start` whose requests may switch to any of `ids`.
fn implicit_on(loader: &Arc<MockLoader>, start: &str, ids: &[&str]) -> Server {
    Server::from_active(loader.model(start))
        .with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT).with_known_models(known(ids)))
}

fn generation(body: Value) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri("/v1/images/generations")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// A streamed answer's events, `(event, data)` in order.
async fn events_of(app: &axum::Router, body: Value) -> Vec<(String, Value)> {
    let response = app.clone().oneshot(generation(body)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CONTENT_TYPE], "text/event-stream");
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    text.split("\n\n")
        .filter(|block| !block.trim().is_empty())
        .map(|block| {
            let field = |name: &str| block.lines().find_map(|line| line.strip_prefix(name)).map(str::trim).unwrap_or_default().to_owned();
            (field("event:"), serde_json::from_str(&field("data:")).expect("a JSON event"))
        })
        .collect()
}

/// Each event as `phase step/total`, or its type when it has no phase.
fn phases(events: &[(String, Value)]) -> Vec<String> {
    events
        .iter()
        .map(|(event, data)| match data["phase"].as_str() {
            Some(phase) => format!("{phase} {}/{}", data["step"], data["total"]),
            None => event.clone(),
        })
        .collect()
}

/// The phases of a job of one image after what comes before it.
fn then_the_job(before: &[&str]) -> Vec<String> {
    let mut expected: Vec<String> = before.iter().map(|phase| phase.to_string()).collect();
    expected.push("encoding 1/1".into());
    expected.extend((1..=8).map(|step| format!("denoising {step}/8")));
    expected.push("decoding 1/1".into());
    expected.push("image_generation.completed".into());
    expected
}

/// Wait (bounded) until the status reads `wanted`.
async fn until(server: &Server, wanted: &str, within: Duration) {
    let deadline = Instant::now() + within;
    while server.status().as_str() != wanted {
        assert!(Instant::now() < deadline, "the status never read {wanted}: {:?}", server.status());
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Spec image-model/05 §Streaming: a streamed request that switches from a
/// text model to the image model it names hears `switching` and `loading`
/// before its job's phases; one on an image model the idle clock unloaded
/// hears `loading`. A refusal that needs the image model loaded comes after
/// the stream began: it is the stream's `error` event.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stream_hears_the_switch_and_the_load_before_its_job() {
    let loader = MockLoader::new();
    let server = implicit_on(&loader, "mock-a", &["mock-a", IMAGE]).with_idle_unload(IDLE);
    let app = server.app();
    let body = json!({ "prompt": "a lighthouse", "model": IMAGE, "size": "1024x1024", "seed": 1, "stream": true });
    let events = events_of(&app, body.clone()).await;
    assert_eq!(phases(&events), then_the_job(&["switching 1/1", "loading 1/1"]));
    assert_eq!(events[0].0, "image_generation.progress");
    assert_eq!((server.active().kind(), loader.loads()), (ModelKind::Image, vec![IMAGE.to_owned()]));

    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    until(&server, "unloaded", IDLE * 5).await;
    let mut unnamed = body.clone();
    unnamed.as_object_mut().unwrap().remove("model");
    let events = events_of(&app, unnamed).await;
    assert_eq!(phases(&events), then_the_job(&["loading 1/1"]));
    assert_eq!(loader.loads(), [IMAGE, IMAGE]);

    // Back on the text model, a prompt over the condition budget is known
    // only once the image model has loaded: an error event after the switch.
    let (status, _) = send_chat(&app).await;
    assert_eq!(status, StatusCode::OK);
    server.update_live(|live| live.image.condition_budget_tokens = 1);
    let events = events_of(&app, json!({ "prompt": "far too long a prompt", "model": IMAGE, "stream": true })).await;
    assert_eq!(phases(&events), ["switching 1/1", "loading 1/1", "error"]);
    assert_eq!(events[2].1["error"]["code"], "condition_budget_exceeded", "{:?}", events[2]);
}

/// A chat request naming `mock-a`, which switches back to it.
async fn send_chat(app: &axum::Router) -> (StatusCode, Value) {
    let body = json!({ "model": "mock-a", "messages": [{ "role": "user", "content": "hello" }], "max_tokens": 2 });
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// Review-322cf L3: a request to be rewritten on the enhancer the idle clock
/// unloaded loads the enhancer, not the image model it names — the rewrite on
/// it, then one switch: two loads, not three. A streamed request that comes
/// while another's load runs waits on it, and hears `loading` too.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_enhanced_request_on_the_unloaded_enhancer_loads_it_and_switches_once() {
    let loader = MockLoader::new();
    let server = implicit_on(&loader, "mock-a", &["mock-a", IMAGE]).with_idle_unload(IDLE);
    enhancing_with(&server, "mock-a");
    let app = server.app();
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    until(&server, "unloaded", IDLE * 5).await;
    let body = json!({ "prompt": "a red fox", "model": IMAGE, "size": "1024x1024", "seed": 4, "prompt_enhancement": true, "stream": true });
    let events = events_of(&app, body).await;
    assert_eq!(phases(&events), then_the_job(&["loading 1/1", "enhancing 1/1", "switching 1/1", "loading 1/1"]));
    assert!(events.last().unwrap().1["revised_prompt"].is_string(), "{:?}", events.last());
    assert_eq!(loader.loads(), ["mock-a", IMAGE], "the enhancer, then the image model");

    until(&server, "unloaded", IDLE * 5).await;
    let hold = loader.hold_next_load();
    let first = tokio::spawn({
        let app = app.clone();
        async move { send_generation(&app, json!({ "prompt": "first", "size": "1024x1024" })).await }
    });
    until(&server, "loading", PATIENT).await;
    let response = app.clone().oneshot(generation(json!({ "prompt": "second", "size": "1024x1024", "stream": true }))).await.unwrap();
    let mut stream = response.into_body();
    let heard = tokio::time::timeout(Duration::from_secs(5), stream.frame()).await.expect("an event while the load runs");
    let heard = String::from_utf8(heard.unwrap().unwrap().into_data().unwrap().to_vec()).unwrap();
    assert!(heard.contains("\"phase\":\"loading\""), "{heard}");
    hold.send(()).unwrap();
    assert_eq!(first.await.unwrap().0, StatusCode::OK);
    while stream.frame().await.is_some() {}
    assert_eq!(loader.loads(), ["mock-a", IMAGE, IMAGE]);
}

/// Review-324serving F2: the idle clock unloaded a text model that is not the
/// enhancer; an enhanced request naming the image model loads the enhancer
/// first, the rewrite on it, then one switch: two loads, not the image
/// model, a visit to the enhancer and back.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_enhanced_request_on_another_unloaded_text_model_loads_the_enhancer_first() {
    let loader = MockLoader::new();
    let server = implicit_on(&loader, "mock-b", &["mock-a", "mock-b", IMAGE]).with_idle_unload(IDLE);
    enhancing_with(&server, "mock-a");
    let app = server.app();
    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    until(&server, "unloaded", IDLE * 5).await;
    let body = json!({ "prompt": "a red fox", "model": IMAGE, "size": "1024x1024", "seed": 4, "prompt_enhancement": true, "stream": true });
    let events = events_of(&app, body).await;
    assert_eq!(phases(&events), then_the_job(&["loading 1/1", "enhancing 1/1", "switching 1/1", "loading 1/1"]));
    assert!(events.last().unwrap().1["revised_prompt"].is_string(), "{:?}", events.last());
    assert_eq!(loader.loads(), ["mock-a", IMAGE], "the enhancer, then the image model");
}

// ── the visit (model_switch::visit) ───────────────────────────────────────

/// What a visit's switches told, in order.
fn recording_steps() -> (ignis_server::model_switch::Steps, Arc<std::sync::Mutex<Vec<SwitchStep>>>) {
    let told = Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = Arc::clone(&told);
    (Arc::new(move |step| sink.lock().unwrap().push(step)), told)
}

/// A server on the image model, its `last_switch` set by an explicit switch
/// to it from `mock-b`, so that "left as it was" is a real assertion.
async fn on_the_image_model(loader: &Arc<MockLoader>) -> Server {
    let server = implicit_on(loader, "mock-b", &["mock-a", "mock-b", IMAGE]);
    let (_, task) = ignis_server::model_switch::begin(&server, source(IMAGE)).expect("a switch begins");
    assert_eq!(task.await.unwrap(), SwitchOutcome::Switched);
    assert_eq!(server.active().model_id(), IMAGE);
    server
}

fn the_switch_to_the_image_model() -> Option<LastSwitch> {
    Some(LastSwitch { from: "mock-b".into(), to: IMAGE.into(), outcome: SwitchOutcome::Switched, host_plan: None })
}

/// Spec image-model/05 §Prompt enhancement: a visit is two switches in one
/// hold — to the visited model, handed over with the gate still closed
/// (every other request `503 model_switching`, a switch `409`), then back —
/// each model off the card before the next loads, and `last_switch` left as
/// it was.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_visit_goes_to_the_visited_model_and_back_with_the_gate_closed_throughout() {
    let loader = MockLoader::new();
    let server = on_the_image_model(&loader).await;
    let app = server.app();
    let (steps, told) = recording_steps();
    let mut visit = visit(&server, source("mock-a"), source(IMAGE), Some(steps)).expect("a visit begins");
    let visited = visit.arrived().await.expect("the visited model serves");
    assert_eq!((visited.active().model_id(), visited.active().kind()), ("mock-a".to_owned(), ModelKind::Text));
    assert_eq!(*server.status(), ModelStatus::Switching { from: "mock-a".into(), to: IMAGE.into() });
    let (status, refused) = send_chat(&app).await;
    assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::SERVICE_UNAVAILABLE, Some("model_switching")), "{refused}");
    let response = app.clone().oneshot(generation(json!({ "prompt": "x", "size": "1024x1024" }))).await.unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(response.headers()[header::RETRY_AFTER], "5");
    let explicit = ignis_server::model_switch::begin(&server, source("mock-b")).map(|(started, _)| started);
    assert_eq!(explicit, Err(SwitchRefusal::InProgress(SwitchStarted { from: "mock-a".into(), to: IMAGE.into() })));

    assert_eq!(visit.on().await, SwitchOutcome::Switched);
    assert_eq!((server.status().as_str(), server.active().model_id()), ("serving", IMAGE.to_owned()));
    assert_eq!(loader.loads(), [IMAGE, "mock-a", IMAGE]);
    assert_eq!(loader.resident_at_load(), [0, 0, 0], "each model left the card before the next loaded");
    assert_eq!(*told.lock().unwrap(), [SwitchStep::Switching, SwitchStep::Loading, SwitchStep::Switching, SwitchStep::Loading]);
    assert_eq!(server.switcher.as_ref().unwrap().last_switch(), the_switch_to_the_image_model());
}

/// A visit whose way there fails is over: refused before anything stopped,
/// the image model never stopped serving; failed on the card, the image
/// model is loaded again. A visit whose way back fails leaves the visited
/// model serving. A visit its beginner leaves goes back on its own. None of
/// them writes `last_switch`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_visit_that_fails_either_way_ends_with_a_model_serving_and_writes_no_last_switch() {
    let loader = MockLoader::new();
    let server = on_the_image_model(&loader).await;
    let last = || server.switcher.as_ref().unwrap().last_switch();

    loader.refuse("mock-a");
    let mut refused = visit(&server, source("mock-a"), source(IMAGE), None).unwrap();
    assert!(refused.arrived().await.is_none());
    assert!(matches!(refused.on().await, SwitchOutcome::Refused { .. }));
    assert_eq!((server.status().as_str(), server.active().model_id(), loader.loads().len()), ("serving", IMAGE.to_owned(), 1));
    assert_eq!(last(), the_switch_to_the_image_model());

    loader.break_load("mock-b");
    let mut broken = visit(&server, source("mock-b"), source(IMAGE), None).unwrap();
    assert!(broken.arrived().await.is_none());
    assert!(matches!(broken.on().await, SwitchOutcome::Restored { .. }));
    assert_eq!((server.status().as_str(), server.active().model_id()), ("serving", IMAGE.to_owned()));
    assert_eq!(loader.loads(), [IMAGE, "mock-b", IMAGE]);
    assert_eq!(last(), the_switch_to_the_image_model());

    loader.mend("mock-b");
    loader.break_load(IMAGE);
    let mut back_broken = visit(&server, source("mock-b"), source(IMAGE), None).unwrap();
    assert!(back_broken.arrived().await.is_some());
    assert!(matches!(back_broken.on().await, SwitchOutcome::Restored { .. }));
    assert_eq!((server.status().as_str(), server.active().model_id()), ("serving", "mock-b".to_owned()));
    assert_eq!(last(), the_switch_to_the_image_model());

    // From mock-b again: an explicit switch, which writes `last_switch`.
    loader.mend(IMAGE);
    let (_, task) = ignis_server::model_switch::begin(&server, source(IMAGE)).unwrap();
    assert_eq!(task.await.unwrap(), SwitchOutcome::Switched);
    let loads = loader.loads().len();
    drop(visit(&server, source("mock-b"), source(IMAGE), None).unwrap());
    until(&server, "serving", PATIENT).await;
    assert_eq!(server.active().model_id(), IMAGE, "a visit left by its beginner goes back");
    assert_eq!(loader.loads()[loads..], ["mock-b", IMAGE]);
    assert_eq!(last(), the_switch_to_the_image_model());
}

// ── prompt enhancement (AC 9) ─────────────────────────────────────────────

/// `image.prompt_enhancer` set to `enhancer`, its rewrites short.
fn enhancing_with(server: &Server, enhancer: &str) {
    server.update_live(|live| {
        live.image.prompt_enhancer = enhancer.to_owned();
        live.image.prompt_enhancer_max_tokens = 16;
    });
}

async fn send_generation(app: &axum::Router, body: Value) -> (StatusCode, Value) {
    let response = app.clone().oneshot(generation(body)).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned())))
}

/// The prompt tokens each request prefilled on `compute`, with its decode
/// parameters, by request in order.
fn prefilled(compute: &MockCompute) -> Vec<(u64, Vec<u32>, ignis_core::DecodeParams)> {
    let mut requests: Vec<(u64, Vec<u32>, ignis_core::DecodeParams)> = Vec::new();
    for job in compute.prefill_calls().into_iter().flatten() {
        match requests.iter_mut().find(|(request, ..)| *request == job.request as u64) {
            // Only the last chunk carries the request's own parameters.
            Some((_, tokens, params)) => {
                tokens.extend(job.tokens.iter().map(|&token| token as u32));
                *params = job.params.clone();
            }
            None => requests.push((job.request as u64, job.tokens.iter().map(|&token| token as u32).collect(), job.params.clone())),
        }
    }
    requests
}

/// AC 9: with the enhancer loaded, an image request naming the image model
/// is rewritten on it first — a CJK prompt with Qwen's Chinese system
/// prompt, reasoning off, the job's seed — and then switches once, to the
/// image model, which runs on the rewrite; `revised_prompt` is returned.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn with_the_enhancer_loaded_the_rewrite_comes_first_and_the_request_switches_once() {
    let loader = MockLoader::new();
    let compute = Arc::new(MockCompute::new());
    let server = Server::from_active(loader.model_over("mock-a", Arc::clone(&compute) as _))
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT).with_known_models(known(&["mock-a", IMAGE])));
    enhancing_with(&server, "mock-a");
    let app = server.app();
    let prompt = "一只雪地里的红狐狸";
    let body = json!({ "prompt": prompt, "model": IMAGE, "size": "1024x1024", "seed": 5, "prompt_enhancement": true });
    let (status, answer) = send_generation(&app, body).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    let revised = answer["data"][0]["revised_prompt"].as_str().expect("revised_prompt").to_owned();
    assert!(!revised.is_empty() && revised != prompt, "{revised}");
    assert!(answer.get("warnings").is_none(), "{answer}");
    assert_eq!(loader.encoded_prompts(), [revised], "the image model ran on the rewrite");
    assert_eq!(loader.loads(), [IMAGE], "one switch: the rewrite needed none");
    let last = server.switcher.as_ref().unwrap().last_switch();
    assert_eq!(last, Some(LastSwitch { from: "mock-a".into(), to: IMAGE.into(), outcome: SwitchOutcome::Switched, host_plan: None }));

    let thinking_off = ThinkingOptions { enable_thinking: false, ..ThinkingOptions::default() };
    let expected = SimpleTemplateProvider.apply_chat_template(&Task::GenerationZh.messages(prompt, &[], ignis_server::images::job::PngCompression::Fast), &thinking_off, &[]).unwrap();
    let rewrites = prefilled(&compute);
    assert_eq!(rewrites.len(), 1, "one rewrite");
    let (_, tokens, params) = &rewrites[0];
    assert_eq!(*tokens, expected.tokens.iter().map(|&token| token as u32).collect::<Vec<_>>(), "the Chinese system prompt and upstream's user turn");
    assert_eq!((params.seed, params.max_tokens, params.starts_in_reasoning), (5, Some(16), false));
}

/// A server on `mock-a`, the enhancer, over `compute`, as the Playground
/// meets the 27B (GitHub #324); its requests may switch to the image model.
fn on_the_loaded_enhancer(loader: &Arc<MockLoader>) -> (Server, Arc<MockCompute>) {
    let compute = Arc::new(MockCompute::new());
    let server = Server::from_active(loader.model_over("mock-a", Arc::clone(&compute) as _))
        .with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT).with_known_models(known(&["mock-a", IMAGE])));
    enhancing_with(&server, "mock-a");
    (server, compute)
}

/// GitHub #324: the Playground's Generate with the enhancer loaded and
/// enhancement on, streamed. The rewrite runs on the enhancer first, then
/// the request switches once, to the image model: `enhancing`, `switching`,
/// `loading`, then the job's phases; one load, `last_switch` the enhancer
/// to the image model, and the rewrite on the completed event, the prompt
/// the image model ran on.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_streamed_request_on_the_loaded_enhancer_hears_the_rewrite_then_one_switch() {
    let loader = MockLoader::new();
    let (server, compute) = on_the_loaded_enhancer(&loader);
    let app = server.app();
    let body = json!({ "prompt": "a red fox in snow", "model": IMAGE, "size": "1024x1024", "seed": 6, "prompt_enhancement": true, "stream": true });
    let events = events_of(&app, body).await;
    assert_eq!(phases(&events), then_the_job(&["enhancing 1/1", "switching 1/1", "loading 1/1"]));
    let revised = events.last().unwrap().1["revised_prompt"].as_str().expect("revised_prompt").to_owned();
    assert_eq!(loader.encoded_prompts(), [revised], "the image model ran on the rewrite");
    assert_eq!(prefilled(&compute).len(), 1, "one rewrite, on mock-a");
    assert_eq!(loader.loads(), [IMAGE], "one switch: the rewrite needed none");
    let last = server.switcher.as_ref().unwrap().last_switch();
    assert_eq!(last, Some(LastSwitch { from: "mock-a".into(), to: IMAGE.into(), outcome: SwitchOutcome::Switched, host_plan: None }));
    assert_eq!((server.status().as_str(), server.active().model_id()), ("serving", IMAGE.to_owned()));
}

/// GitHub #324: the same request with enhancement off — the request's
/// `false` over `image.prompt_enhancement` on — is one switch and no
/// rewrite: `switching` and `loading` before the job, nothing prefilled on
/// the text model, the prompt as sent, no `revised_prompt`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn with_enhancement_off_a_request_on_the_loaded_enhancer_switches_once_and_rewrites_nothing() {
    let loader = MockLoader::new();
    let (server, compute) = on_the_loaded_enhancer(&loader);
    server.update_live(|live| live.image.prompt_enhancement = true);
    let app = server.app();
    let body = json!({ "prompt": "as sent", "model": IMAGE, "size": "1024x1024", "seed": 6, "prompt_enhancement": false, "stream": true });
    let events = events_of(&app, body).await;
    assert_eq!(phases(&events), then_the_job(&["switching 1/1", "loading 1/1"]));
    assert!(events.last().unwrap().1.get("revised_prompt").is_none(), "{:?}", events.last());
    assert!(prefilled(&compute).is_empty(), "no rewrite");
    assert_eq!((loader.loads(), loader.encoded_prompts()), (vec![IMAGE.to_owned()], vec!["as sent".to_owned()]));
}

/// GitHub #324: with `switch.allow_implicit` off, the Playground's Generate
/// naming the image model on the loaded enhancer is refused by name —
/// `404 model_not_found`, a status even when streamed — before any rewrite
/// or load; the text model serves on, and an explicit switch is the way.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn with_implicit_switching_off_a_request_naming_the_image_model_is_refused_before_any_rewrite() {
    let loader = MockLoader::new();
    let compute = Arc::new(MockCompute::new());
    let server = Server::from_active(loader.model_over("mock-a", Arc::clone(&compute) as _))
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT).with_listed_models(known(&["mock-a", IMAGE])));
    enhancing_with(&server, "mock-a");
    let body = json!({ "prompt": "a red fox", "model": IMAGE, "size": "1024x1024", "prompt_enhancement": true, "stream": true });
    let (status, refused) = send_generation(&server.app(), body).await;
    assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::NOT_FOUND, Some("model_not_found")), "{refused}");
    let message = refused["error"]["message"].as_str().unwrap();
    assert!(message.contains(IMAGE) && message.contains("mock-a"), "{message}");
    assert!(prefilled(&compute).is_empty() && loader.loads().is_empty(), "nothing rewritten, nothing loaded");
    assert_eq!((server.status().as_str(), server.active().model_id()), ("serving", "mock-a".to_owned()));
}

/// AC 9: with the image model loaded, an enhanced request visits the
/// enhancer — `switching`, `loading`, `enhancing`, `switching`, `loading`
/// streamed before the job — two switches that leave `last_switch` as it
/// was; the image model runs on the rewrite, returned as `revised_prompt`.
/// `image.prompt_enhancement` turns it on for a request that does not say,
/// and `prompt_enhancement: false` off.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn with_the_image_model_loaded_an_enhanced_request_visits_the_enhancer_and_comes_back() {
    let loader = MockLoader::new();
    let server = on_the_image_model(&loader).await;
    enhancing_with(&server, "mock-a");
    let app = server.app();
    let body = json!({ "prompt": "a red fox in snow", "size": "1024x1024", "seed": 3, "prompt_enhancement": true, "stream": true });
    let events = events_of(&app, body).await;
    assert_eq!(
        phases(&events),
        then_the_job(&["switching 1/2", "loading 1/2", "enhancing 1/1", "switching 2/2", "loading 2/2"])
    );
    let completed = &events.last().unwrap().1;
    let revised = completed["revised_prompt"].as_str().expect("revised_prompt").to_owned();
    assert_eq!(loader.encoded_prompts(), [revised.clone()]);
    assert_eq!(loader.loads(), [IMAGE, "mock-a", IMAGE]);
    assert_eq!(loader.resident_at_load(), [0, 0, 0]);
    assert_eq!(server.switcher.as_ref().unwrap().last_switch(), the_switch_to_the_image_model(), "a visit is not a switch");
    assert_eq!((server.status().as_str(), server.active().model_id()), ("serving", IMAGE.to_owned()));

    server.update_live(|live| live.image.prompt_enhancement = true);
    let (status, answer) = send_generation(&app, json!({ "prompt": "a red fox in snow", "size": "1024x1024", "seed": 3 })).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert_eq!(answer["data"][0]["revised_prompt"], revised, "the same seed, the same rewrite");
    assert_eq!(loader.loads().len(), 5, "the config's default visited too");
    let (status, answer) = send_generation(&app, json!({ "prompt": "as sent", "size": "1024x1024", "prompt_enhancement": false })).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert!(answer["data"][0].get("revised_prompt").is_none(), "{answer}");
    assert_eq!((loader.loads().len(), loader.encoded_prompts().last().map(String::as_str)), (5, Some("as sent")));
}

/// Review-322cf L2 (ADR 0049 §5): a visit from another text model ends on
/// the image model the request names — the request's switch, streamed as a
/// visit's two — so each of its switches is written to `last_switch` as a
/// switch is: the last, from the enhancer to the image model, once it lands;
/// when that one fails, how, the enhancer loaded again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_visit_from_another_text_model_is_written_to_last_switch() {
    let loader = MockLoader::new();
    let server = implicit_on(&loader, "mock-b", &["mock-a", "mock-b", IMAGE]);
    enhancing_with(&server, "mock-a");
    let app = server.app();
    let last = || server.switcher.as_ref().unwrap().last_switch();
    let body = json!({ "prompt": "a red fox", "model": IMAGE, "size": "1024x1024", "seed": 2, "prompt_enhancement": true, "stream": true });
    let events = events_of(&app, body.clone()).await;
    assert_eq!(
        phases(&events),
        then_the_job(&["switching 1/2", "loading 1/2", "enhancing 1/1", "switching 2/2", "loading 2/2"])
    );
    assert!(events.last().unwrap().1["revised_prompt"].is_string(), "{:?}", events.last());
    assert_eq!(loader.loads(), ["mock-a", IMAGE]);
    assert_eq!(last(), Some(LastSwitch { from: "mock-a".into(), to: IMAGE.into(), outcome: SwitchOutcome::Switched, host_plan: None }));

    let (_, task) = ignis_server::model_switch::begin(&server, source("mock-b")).unwrap();
    assert_eq!(task.await.unwrap(), SwitchOutcome::Switched);
    loader.break_load(IMAGE);
    let mut whole = body;
    whole["stream"] = json!(false);
    let (status, refused) = send_generation(&app, whole).await;
    assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::SERVICE_UNAVAILABLE, Some("model_switch_failed")), "{refused}");
    assert_eq!((server.status().as_str(), server.active().model_id()), ("serving", "mock-a".to_owned()));
    let restored = last().expect("a record");
    assert_eq!((restored.from.as_str(), restored.to.as_str()), ("mock-a", IMAGE));
    assert!(matches!(restored.outcome, SwitchOutcome::Restored { .. }), "{restored:?}");
}

/// AC 9: while a visit runs — here with the rewrite done and the image
/// model's reload held — every other request is answered `503
/// model_switching` with `Retry-After`; the enhanced request is answered
/// once the image model is back.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn during_a_visit_other_requests_are_answered_model_switching() {
    let loader = MockLoader::new();
    let server = on_the_image_model(&loader).await;
    enhancing_with(&server, "mock-a");
    let app = server.app();
    let there = loader.hold_next_load();
    let back = loader.hold_next_load();
    let body = json!({ "prompt": "a red fox", "size": "1024x1024", "prompt_enhancement": true });
    let enhanced = tokio::spawn({
        let app = app.clone();
        async move { send_generation(&app, body).await }
    });
    until(&server, "switching", PATIENT).await;
    there.send(()).unwrap();
    let visiting = ModelStatus::Switching { from: "mock-a".into(), to: IMAGE.into() };
    let deadline = Instant::now() + PATIENT;
    while *server.status() != visiting || loader.resident() != 0 {
        assert!(Instant::now() < deadline, "the visit never got to its way back: {:?}", server.status());
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    // The enhancer rewrote and was torn down; the image model's load waits.
    let (status, refused) = send_chat(&app).await;
    assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::SERVICE_UNAVAILABLE, Some("model_switching")), "{refused}");
    let response = app.clone().oneshot(generation(json!({ "prompt": "another" }))).await.unwrap();
    assert_eq!((response.status(), response.headers()[header::RETRY_AFTER].to_str().unwrap()), (StatusCode::SERVICE_UNAVAILABLE, "5"));
    assert!(!enhanced.is_finished());
    back.send(()).unwrap();
    let (status, answer) = enhanced.await.unwrap();
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert!(answer["data"][0]["revised_prompt"].is_string(), "{answer}");
    assert_eq!(server.status().as_str(), "serving");
}

/// GitHub #329: an enhanced request waits for its visit's loads inside its
/// handler. Once the serve stops it is answered `503 server_stopping`
/// instead, on the way there and on the way back alike, and the visit still
/// running is returned to be abandoned, not waited for.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn once_the_serve_stops_a_visit_is_not_waited_for() {
    for leg in ["there", "back"] {
        let loader = MockLoader::new();
        let server = on_the_image_model(&loader).await;
        enhancing_with(&server, "mock-a");
        let app = server.app();
        let there = loader.hold_next_load();
        let back = loader.hold_next_load();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let serving = tokio::spawn(server.clone().serve_on_until(listener, None, async {
            let _ = stopped.await;
        }));
        let body = json!({ "prompt": "a red fox", "size": "1024x1024", "prompt_enhancement": true });
        let enhanced = tokio::spawn({
            let app = app.clone();
            async move { send_generation(&app, body).await }
        });
        until(&server, "switching", PATIENT).await;
        if leg == "back" {
            there.send(()).unwrap();
            let visiting = ModelStatus::Switching { from: "mock-a".into(), to: IMAGE.into() };
            let deadline = Instant::now() + PATIENT;
            while *server.status() != visiting || loader.resident() != 0 {
                assert!(Instant::now() < deadline, "the visit never got to its way back: {:?}", server.status());
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }

        stop.send(()).unwrap();
        let bound = Duration::from_secs(5);
        let (status, refused) = tokio::time::timeout(bound, enhanced).await.expect("answered at the stop").unwrap();
        assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::SERVICE_UNAVAILABLE, Some("server_stopping")), "{leg}: {refused}");
        let served = tokio::time::timeout(bound, serving).await.expect("the serve returned").unwrap().expect("serve");
        assert!(matches!(served, ignis_server::Served::LoadAbandoned { .. }), "{leg}: {served:?}");
        // Released only so the runtime is not left joining a held thread.
        let _ = there.send(());
        let _ = back.send(());
    }
}

/// GitHub #329: an enhanced request whose turn in the queue comes after the
/// stop — the job admitted before it run to its end by the drain — begins
/// no visit: it is answered `503 server_stopping`, nothing is loaded, and
/// the serve ends drained, with no load left to abandon.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_enhanced_request_whose_turn_comes_after_the_stop_begins_no_visit() {
    let loader = MockLoader::new();
    let (model, gate) = loader.gated_image_model(IMAGE);
    let server = Server::from_active(model)
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT).with_known_models(known(&["mock-a", IMAGE])));
    enhancing_with(&server, "mock-a");
    let jobs = Arc::clone(&server.active().image_model().expect("the image model").jobs);
    let in_flight = |waiting: u32, running: u32| {
        let jobs = Arc::clone(&jobs);
        async move {
            let deadline = Instant::now() + PATIENT;
            while (jobs.in_flight().waiting, jobs.in_flight().running) != (waiting, running) {
                assert!(Instant::now() < deadline, "never {waiting} waiting and {running} running: {:?}", jobs.in_flight());
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let serving = tokio::spawn(server.clone().serve_on_until(listener, None, async {
        let _ = stopped.await;
    }));
    // Over the listener, so the serve's drain waits for both requests.
    let send = |body: Value| {
        tokio::spawn(async move {
            let client = reqwest::Client::builder().pool_max_idle_per_host(0).build().unwrap();
            let response = client
                .post(format!("http://{addr}/v1/images/generations"))
                .header(header::CONTENT_TYPE.as_str(), "application/json")
                .body(body.to_string())
                .send()
                .await
                .expect("answered");
            let status = response.status().as_u16();
            let retry_after = response.headers().get("retry-after").map(|value| value.to_str().unwrap().to_owned());
            (status, retry_after, serde_json::from_str::<Value>(&response.text().await.unwrap()).unwrap())
        })
    };
    let first = send(json!({ "prompt": "first", "size": "1024x1024" }));
    in_flight(0, 1).await;
    let enhanced = send(json!({ "prompt": "second", "size": "1024x1024", "prompt_enhancement": true }));
    in_flight(1, 1).await;

    stop.send(()).unwrap();
    let switcher = Arc::clone(server.switcher.as_ref().unwrap());
    let deadline = Instant::now() + PATIENT;
    while !switcher.stopping() {
        assert!(Instant::now() < deadline, "the serve never began to stop");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    gate.open();
    let (status, _, answer) = first.await.unwrap();
    assert_eq!(status, 200, "the job before it runs to its end: {answer}");
    let (status, retry_after, refused) = enhanced.await.unwrap();
    assert_eq!((status, refused["error"]["code"].as_str(), retry_after.as_deref()), (503, Some("server_stopping"), Some("5")), "{refused}");
    let served = tokio::time::timeout(Duration::from_secs(5), serving).await.expect("the serve returned").unwrap().expect("serve");
    assert_eq!(served, ignis_server::Served::Drained, "no visit began, so none was abandoned");
    assert!(loader.prepared_artifacts().is_empty() && loader.loads().is_empty(), "no visit began: {:?}", loader.loads());
}

/// Review-322cf M1: an enhanced request on the loaded image model takes its
/// turn in the queue before it visits the enhancer. The job running and the
/// one waiting before it run to their end on the image model — past the
/// drain window, which would have cut them had the visit begun — and the
/// visit comes after them; the enhanced job then runs on the image model
/// loaded again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_enhanced_request_visits_only_once_the_jobs_admitted_before_it_have_run() {
    let loader = MockLoader::new();
    let (model, gate) = loader.gated_image_model(IMAGE);
    let drain = Duration::from_millis(200);
    let server = Server::from_active(model)
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, drain).with_known_models(known(&["mock-a", IMAGE])));
    enhancing_with(&server, "mock-a");
    let app = server.app();
    let jobs = Arc::clone(&server.active().image_model().expect("the image model").jobs);
    let send = |body: Value| {
        let app = app.clone();
        tokio::spawn(async move { send_generation(&app, body).await })
    };
    let in_flight = |waiting: u32, running: u32| {
        let jobs = Arc::clone(&jobs);
        async move {
            let deadline = Instant::now() + PATIENT;
            while (jobs.in_flight().waiting, jobs.in_flight().running) != (waiting, running) {
                assert!(Instant::now() < deadline, "never {waiting} waiting and {running} running: {:?}", jobs.in_flight());
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }
    };
    let first = send(json!({ "prompt": "first", "size": "1024x1024" }));
    in_flight(0, 1).await;
    let second = send(json!({ "prompt": "second", "size": "1024x1024" }));
    in_flight(1, 1).await;
    let enhanced = send(json!({ "prompt": "third", "size": "1024x1024", "prompt_enhancement": true }));
    in_flight(2, 1).await;
    tokio::time::sleep(drain + drain / 2).await;
    assert_eq!((server.status().as_str(), loader.loads().len()), ("serving", 0), "no visit while jobs before it remain");

    gate.open();
    for (name, answer) in [("first", first), ("second", second)] {
        let (status, answer) = answer.await.unwrap();
        assert_eq!(status, StatusCode::OK, "{name}: {answer}");
    }
    let (status, answer) = enhanced.await.unwrap();
    assert_eq!(status, StatusCode::OK, "{answer}");
    let revised = answer["data"][0]["revised_prompt"].as_str().expect("revised_prompt").to_owned();
    assert_eq!(loader.encoded_prompts(), ["first".to_owned(), "second".to_owned(), revised], "the visit after both jobs");
    assert_eq!(loader.loads(), ["mock-a", IMAGE]);
    assert_eq!(server.status().as_str(), "serving");
}

/// AC 9: a failing enhancer falls back to the prompt as sent, with a
/// warning — refused before anything stopped, failed to load (the image
/// model loaded again), an id that names no model, an engine that refuses
/// the rewrite — and the request is served all the same.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failing_enhancer_falls_back_to_the_prompt_as_sent_with_a_warning() {
    let loader = MockLoader::new();
    let server = on_the_image_model(&loader).await;
    enhancing_with(&server, "mock-a");
    let app = server.app();
    let enhanced = |prompt: &str| json!({ "prompt": prompt, "size": "1024x1024", "prompt_enhancement": true });
    let fell_back = |answer: &Value, words: &str| {
        assert!(answer["data"][0].get("revised_prompt").is_none(), "{answer}");
        let warning = answer["warnings"][0].as_str().unwrap_or_default();
        assert!(warning.contains("did not rewrite the prompt") && warning.contains("used as sent") && warning.contains(words), "{answer}");
    };

    loader.refuse("mock-a");
    let (status, answer) = send_generation(&app, enhanced("refused")).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    fell_back(&answer, "refused at prepare");
    assert_eq!(loader.loads().len(), 1, "nothing stopped");

    enhancing_with(&server, "mock-b");
    loader.break_load("mock-b");
    let (status, answer) = send_generation(&app, enhanced("broken")).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    fell_back(&answer, "failed to load");
    assert_eq!(loader.loads()[1..], ["mock-b", IMAGE], "the image model was loaded again");

    enhancing_with(&server, "nobody");
    let (status, answer) = send_generation(&app, enhanced("unknown")).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    fell_back(&answer, "names no text model");
    assert_eq!(loader.encoded_prompts(), ["refused", "broken", "unknown"], "each job ran on the prompt as sent");
    assert_eq!(server.switcher.as_ref().unwrap().last_switch(), the_switch_to_the_image_model());

    // On a loaded enhancer whose engine refuses the rewrite: the system
    // prompt and 8,192 tokens of answer are past the mock's 8,192 context.
    let loader = MockLoader::new();
    let server = implicit_on(&loader, "mock-a", &["mock-a", IMAGE]);
    enhancing_with(&server, "mock-a");
    server.update_live(|live| live.image.prompt_enhancer_max_tokens = 8192);
    let app = server.app();
    let body = json!({ "prompt": "engine fault", "model": IMAGE, "size": "1024x1024", "prompt_enhancement": true });
    let (status, answer) = send_generation(&app, body).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    fell_back(&answer, "mock-a did not take the rewrite");
    assert_eq!((loader.loads(), loader.encoded_prompts()), (vec![IMAGE.to_owned()], vec!["engine fault".to_owned()]));
}

/// An enhancer whose rewrite waits at the gate, behind a request that names
/// the image model: `release` runs once the rewrite is held, and the answer
/// is the request's, which fell back to its prompt as sent.
async fn a_held_rewrite_falls_back(request_timeout: Option<Duration>, release: impl FnOnce(&Server) -> Duration) -> Value {
    let loader = MockLoader::new();
    let (model, gated, gate) = loader.gated_model("mock-a");
    gated.arm();
    let mut server = Server::from_active(model)
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT).with_known_models(known(&["mock-a", IMAGE])));
    if let Some(timeout) = request_timeout {
        server = server.with_request_timeout(timeout);
    }
    enhancing_with(&server, "mock-a");
    let app = server.app();
    let body = json!({ "prompt": "held", "model": IMAGE, "size": "1024x1024", "prompt_enhancement": true });
    let reply = tokio::spawn(async move { send_generation(&app, body).await });
    gate.wait_entered();
    tokio::time::sleep(release(&server)).await;
    gate.release();
    let (status, answer) = reply.await.unwrap();
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert!(answer["data"][0].get("revised_prompt").is_none(), "{answer}");
    assert_eq!(loader.encoded_prompts(), ["held"], "the job ran on the prompt as sent");
    answer
}

/// A rewrite its deadline cut falls back to the prompt as sent, the warning
/// saying the deadline passed (ADR 0058).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rewrite_past_its_deadline_falls_back_saying_so() {
    let answer = a_held_rewrite_falls_back(Some(Duration::from_millis(200)), |_| Duration::from_millis(600)).await;
    let warning = answer["warnings"][0].as_str().unwrap_or_default();
    assert!(warning.contains("did not finish the rewrite within server.request_timeout"), "{answer}");
}

/// A rewrite its engine dropped falls back the same way, the warning saying
/// the engine dropped it and not that time ran out.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rewrite_its_engine_dropped_falls_back_saying_so() {
    let answer = a_held_rewrite_falls_back(None, |server| {
        // The scheduler releases a cancelled request without a `Done`.
        server.active().text_engine().cancel(0);
        Duration::ZERO
    })
    .await;
    let warning = answer["warnings"][0].as_str().unwrap_or_default();
    assert!(warning.contains("engine dropped the rewrite"), "{answer}");
    assert!(!warning.contains("request_timeout"), "{answer}");
}

/// An enhanced edit naming the image model, with one 64x64 condition image.
fn enhanced_edit() -> Request<Body> {
    const BOUNDARY: &str = "ignis-enhancement-boundary";
    let image = ignis_core::image::RgbaImage { size: ignis_core::image::ImageSize::new(64, 64), pixels: vec![90; 64 * 64 * 4] };
    let png = ignis_server::images::job::encode_png(&image, ignis_server::images::job::PngCompression::Fast);
    let mut body = Vec::new();
    for (name, value) in [("prompt", "make it blue"), ("model", IMAGE), ("size", "1024x1024"), ("prompt_enhancement", "true")] {
        body.extend(format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").bytes());
    }
    body.extend(format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"a.png\"\r\nContent-Type: image/png\r\n\r\n").bytes());
    body.extend_from_slice(&png);
    body.extend(format!("\r\n--{BOUNDARY}--\r\n").bytes());
    Request::builder()
        .method(Method::POST)
        .uri("/v1/images/edits")
        .header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={BOUNDARY}"))
        .body(Body::from(body))
        .unwrap()
}

/// An edit's enhancer with no vision tower rewrites the instruction from its
/// text alone, and the answer says so beside the rewrite.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_edit_rewritten_without_a_vision_tower_says_so() {
    let loader = MockLoader::new();
    let server = implicit_on(&loader, "mock-a", &["mock-a", IMAGE]);
    enhancing_with(&server, "mock-a");
    let response = server.app().oneshot(enhanced_edit()).await.unwrap();
    let status = response.status();
    let answer: Value = serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert!(answer["data"][0]["revised_prompt"].is_string(), "{answer}");
    let warning = answer["warnings"][0].as_str().unwrap_or_default();
    assert!(warning.contains("no vision tower") && warning.contains("without seeing its condition images"), "{answer}");
    assert_eq!(loader.loads(), [IMAGE]);
}

/// An edit's enhancer with a vision tower is shown the condition images —
/// PNG-encoded off the async workers (review-322cf L6) — before the
/// instruction, and the answer has nothing to say of them.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_edit_rewritten_with_a_vision_tower_shows_it_the_images() {
    let loader = MockLoader::new();
    let compute = Arc::new(MockCompute::new());
    let limits = ignis_artifact::vision::ProcessorOptions {
        min_pixels: 32 * 32,
        max_pixels: 128 * 128,
        max_encoded_media_bytes: 256 << 20,
        max_decoded_pixels: 1 << 24,
        max_raw_patches: 1 << 17,
        max_vision_tokens: 1 << 15,
    };
    let acquirer = ignis_server::media::MediaAcquirer::new(
        Arc::new(media_support::processor(limits.clone())),
        limits,
        ignis_server::media::MediaPolicy::new(false, 0),
    );
    let server = Server::from_active(loader.model_over("mock-a", Arc::clone(&compute) as _))
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT).with_known_models(known(&["mock-a", IMAGE])))
        .with_media(Arc::new(acquirer));
    enhancing_with(&server, "mock-a");
    let response = server.app().oneshot(enhanced_edit()).await.unwrap();
    let status = response.status();
    let answer: Value = serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert!(answer["data"][0]["revised_prompt"].is_string(), "{answer}");
    assert!(answer.get("warnings").is_none(), "{answer}");
    let rewrites = prefilled(&compute);
    assert_eq!(rewrites.len(), 1, "one rewrite");
    assert!(rewrites[0].1.contains(&ignis_artifact::vision::IMAGE_PAD_ID), "the rewrite saw the condition image");
    assert_eq!(loader.loads(), [IMAGE]);
}
