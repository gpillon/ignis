//! `POST /v1/models/switch` and what the API says while a switch runs (spec
//! model-switch/01, GitHub #305), over the mock-backed router: `202` with
//! the envelope, `GET /v1/models` reporting `switching` and then `serving`
//! the new id, a request sent mid-switch refused `503 model_switching` with
//! `Retry-After`, a second switch `409` — and the implicit switch a request's
//! own `model` begins on chat completions, Responses (HTTP and socket) and
//! `/v1/decide` (§Implicit switch). The switch is held inside its load by
//! the mock loader (`support/switch.rs`), never by a timer (ADR 0006).

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_server::model_switch::Switcher;
use ignis_server::Server;
use serde_json::{json, Value};
use tower::ServiceExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::MockLoader;

#[path = "support/responses.rs"]
mod responses;

const PATIENT: Duration = Duration::from_secs(30);

fn server_on(loader: &Arc<MockLoader>) -> Server {
    Server::from_active(loader.model("mock-a")).with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT))
}

async fn send(app: &axum::Router, method: Method, uri: &str, body: Option<Value>) -> (StatusCode, axum::http::HeaderMap, Value) {
    let request = Request::builder().method(method).uri(uri).header(header::CONTENT_TYPE, "application/json");
    let request = request.body(body.map_or_else(Body::empty, |body| Body::from(body.to_string()))).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body = serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, headers, body)
}

fn switch_to(model: &str) -> Value {
    json!({ "artifact": format!("{model}.ninfer"), "model": model })
}

fn chat(model: &str) -> Value {
    json!({ "model": model, "messages": [{ "role": "user", "content": "hello there" }], "max_tokens": 3 })
}

/// `GET /v1/models` until its `status` is `serving` (scheduling turns,
/// bounded): the switch finishes on its own task.
async fn until_serving(app: &axum::Router) -> Value {
    until_status(app, "serving").await
}

/// `GET /v1/models` until its `status` is `wanted` (scheduling turns,
/// bounded).
async fn until_status(app: &axum::Router, wanted: &str) -> Value {
    for _ in 0..10_000 {
        let (status, _, body) = send(app, Method::GET, "/v1/models", None).await;
        if status == StatusCode::OK && body["status"] == wanted {
            return body;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("`GET /v1/models` never reported {wanted}");
}

#[tokio::test]
async fn a_switch_is_accepted_reported_while_it_runs_and_serves_the_new_model_after() {
    let loader = MockLoader::new();
    let server = server_on(&loader);
    let app = server.app();
    let (status, _, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(models["status"], "serving");
    assert_eq!(models["data"][0]["id"], "mock-a");
    assert!(models.get("switching").is_none() && models.get("reason").is_none(), "{models}");

    let release = loader.hold_next_load();
    let (status, _, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to("mock-b"))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    assert_eq!(accepted, json!({ "status": "switching", "from": "mock-a", "to": "mock-b" }));

    // Mid-switch: the models route reports it, everything else waits.
    let (status, _, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(models["status"], "switching");
    assert_eq!(models["switching"], json!({ "from": "mock-a", "to": "mock-b" }));
    assert_eq!(models["data"][0]["id"], "mock-a", "the old model is named until the new one serves");

    let (status, headers, refused) = send(&app, Method::POST, "/v1/chat/completions", Some(chat("mock-a"))).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(refused["error"]["code"], "model_switching", "{refused}");
    assert!(refused["error"]["message"].as_str().unwrap().contains("mock-b"), "{refused}");
    // GitHub #327: a switch is a load, which takes seconds.
    assert_eq!(headers[header::RETRY_AFTER], "5");
    assert_eq!(headers["access-control-allow-origin"], "*");

    let (status, _, conflict) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to("mock-c"))).await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(conflict["error"]["code"], "switch_in_progress");
    assert!(conflict["error"]["message"].as_str().unwrap().contains("mock-a to mock-b"), "{conflict}");

    release.send(()).expect("the held load is waiting");
    let models = until_serving(&app).await;
    assert_eq!(models["data"][0]["id"], "mock-b");
    let (status, _, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat("mock-b"))).await;
    assert_eq!(status, StatusCode::OK, "{completion}");
    assert_eq!(completion["model"], "mock-b");
    let (status, _, stale) = send(&app, Method::POST, "/v1/chat/completions", Some(chat("mock-a"))).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "the old id is no longer served: {stale}");
}

/// A target Flash-Next's host plan refused: `last_switch` says why for a
/// person (`reason`, in GiB, with the shortfall) and in exact bytes for an
/// agent (`host_plan`), and the old model serves again.
#[tokio::test]
async fn a_switch_short_of_ram_reports_the_host_plan_in_bytes_beside_its_reason() {
    let loader = MockLoader::new();
    loader.starve("mock-flash");
    let app = server_on(&loader).app();

    let (status, _, _) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to("mock-flash"))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let mut models = Value::Null;
    for _ in 0..10_000 {
        models = send(&app, Method::GET, "/v1/models", None).await.2;
        if models["last_switch"]["outcome"] == "restored" && models["status"] == "serving" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let last = &models["last_switch"];
    assert_eq!(last["outcome"], "restored", "{models}");
    assert_eq!(models["data"][0]["id"], "mock-a", "the old model serves again: {models}");
    let reason = last["reason"].as_str().unwrap();
    assert!(reason.contains("needs 43.2 GiB available, but only 38.8 GiB is (4.5 GiB short)"), "{reason}");
    let plan = &last["host_plan"];
    assert_eq!(plan["available_bytes"], 41_632_530_432u64, "{plan}");
    assert_eq!(plan["margin_bytes"], 4u64 << 30);
    assert_eq!(plan["planned_bytes"], 42_123_896_832u64);
    assert_eq!(plan["needed_bytes"], 46_418_864_128u64);
    assert_eq!(plan["short_bytes"], 4_786_333_696u64);
    assert_eq!(plan["crossing_line"], "expert_pool");
    assert_eq!(
        plan["lines"],
        json!([
            { "name": "token_embedding", "bytes": 0u64 },
            { "name": "vision_tower", "bytes": 0u64 },
            { "name": "expert_pool", "bytes": 37_795_446_784u64 },
            { "name": "load_staging", "bytes": 0u64 },
            { "name": "ngram_hot_rows", "bytes": 1u64 << 30 },
            { "name": "staging", "bytes": 64u64 << 20 },
            { "name": "retained_host_slots", "bytes": 1_040_115_712u64 },
            { "name": "kv_ram_arena", "bytes": 2u64 << 30 },
        ])
    );

    // A switch that fails any other way names no host plan.
    loader.break_load("mock-broken");
    let (status, _, _) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to("mock-broken"))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    for _ in 0..10_000 {
        models = send(&app, Method::GET, "/v1/models", None).await.2;
        if models["last_switch"]["to"] == "mock-broken" && models["status"] == "serving" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(models["last_switch"]["outcome"], "restored", "{models}");
    assert!(models["last_switch"].get("host_plan").is_none(), "{models}");
}

#[tokio::test]
async fn a_failed_switch_with_nothing_to_reload_says_why_and_takes_the_next_switch() {
    let loader = MockLoader::new();
    loader.break_load("mock-broken");
    let mut first = loader.model("mock-a");
    first.source = None;
    let server = Server::from_active(first).with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT));
    let app = server.app();

    let (status, _, _) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to("mock-broken"))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let mut models = Value::Null;
    for _ in 0..10_000 {
        models = send(&app, Method::GET, "/v1/models", None).await.2;
        if models["status"] == "failed" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(models["status"], "failed", "{models}");
    assert!(models["reason"].as_str().unwrap().contains("simulated kernel load error"), "{models}");
    let (status, _, refused) = send(&app, Method::POST, "/v1/chat/completions", Some(chat("mock-a"))).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(refused["error"]["code"], "server_not_ready", "{refused}");

    let (status, _, _) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to("mock-b"))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "a failed server is recovered by switching");
    assert_eq!(until_serving(&app).await["data"][0]["id"], "mock-b");
}

#[tokio::test]
async fn a_switch_names_both_fields_or_is_refused() {
    let loader = MockLoader::new();
    let app = server_on(&loader).app();
    for (body, param) in [
        (json!({ "artifact": "", "model": "mock-b" }), "artifact"),
        (json!({ "artifact": "mock-b.ninfer", "model": " " }), "model"),
    ] {
        let (status, _, refused) = send(&app, Method::POST, "/v1/models/switch", Some(body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(refused["error"]["param"], param, "{refused}");
    }
    let (status, _, _) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": "mock-b" }))).await;
    assert!(status.is_client_error(), "a missing artifact is refused: {status}");
    assert!(loader.loads().is_empty());
}

#[tokio::test]
async fn a_server_without_a_loader_answers_501_and_a_warming_one_503() {
    let loader = MockLoader::new();
    let bare = Server::from_active(loader.model("mock-a")).app();
    let (status, _, refused) = send(&bare, Method::POST, "/v1/models/switch", Some(switch_to("mock-b"))).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert_eq!(refused["error"]["code"], "switch_unavailable");

    let warming = server_on(&loader).with_warm_up().app();
    let (status, headers, refused) = send(&warming, Method::POST, "/v1/models/switch", Some(switch_to("mock-b"))).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(refused["error"]["code"], "server_not_ready");
    assert_eq!(headers[header::RETRY_AFTER], "5", "GitHub #327: before the first warm-up, a few seconds");
    let (status, _, _) = send(&warming, Method::GET, "/v1/models", None).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "the warm-up still holds every route, as it did");
    let (status, _, _) = send(&warming, Method::OPTIONS, "/v1/models/switch", None).await;
    assert_eq!(status, StatusCode::OK, "a preflight passes");
}

#[tokio::test]
async fn the_switch_route_is_behind_the_api_key() {
    let loader = MockLoader::new();
    let app = server_on(&loader).with_api_key(ignis_server::config::ApiKey::new("sk-test")).app();
    let (status, _, _) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to("mock-b"))).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert!(loader.loads().is_empty());
}

async fn sample(metrics: &axum::Router, name: &str) -> String {
    let (_, _, text) = send(metrics, Method::GET, "/metrics", None).await;
    let text = text.as_str().expect("the exposition is text").to_owned();
    let line = text.lines().find(|l| l.starts_with(&format!("{name} "))).unwrap_or_else(|| panic!("no {name}"));
    line.rsplit_once(' ').unwrap().1.to_owned()
}

/// `--server-metrics` follows the switch: the new model's requests are counted,
/// in the same counters the old model's were.
#[tokio::test]
async fn metrics_keep_counting_on_the_model_a_switch_loaded() {
    let loader = MockLoader::new();
    let server = server_on(&loader).with_metrics();
    let app = server.app();
    let metrics = server.metrics_app().expect("metrics are on");
    let (status, _, _) = send(&app, Method::POST, "/v1/chat/completions", Some(chat("mock-a"))).await;
    assert_eq!(status, StatusCode::OK);

    let (status, _, _) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to("mock-b"))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    until_serving(&app).await;
    let (status, _, _) = send(&app, Method::POST, "/v1/chat/completions", Some(chat("mock-b"))).await;
    assert_eq!(status, StatusCode::OK);
    for _ in 0..10_000 {
        if sample(&metrics, "ignis_requests_completed_total").await == "2" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(sample(&metrics, "ignis_requests_completed_total").await, "2", "one request on each model");
}

// ── the implicit switch: a request's own `model` (spec §Implicit switch) ──

/// A server on `mock-a` whose requests may switch to any of `ids` by naming
/// it — what `main` builds from `--switch-known-models` with `--switch-allow-implicit`
/// on.
fn implicit_on(loader: &Arc<MockLoader>, ids: &[&str]) -> Server {
    let known = ids.iter().map(|id| (id.to_string(), switch_support::source(id).artifact)).collect();
    Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT).with_known_models(known))
}

fn response_body(model: &str) -> Value {
    json!({ "model": model, "input": "hello there", "max_output_tokens": 3, "enable_thinking": false })
}

fn decision(model: &str) -> Value {
    json!({ "state": "s", "model": model, "questions": { "q": { "type": "noul", "instructions": "Urgent?" } } })
}

/// AC 17, 21: a chat completion naming another known model — lane tag and
/// all — switches the server to it and is answered by it, in one call.
#[tokio::test]
async fn a_chat_completion_naming_a_known_model_is_answered_by_it_after_the_switch() {
    let loader = MockLoader::new();
    let app = implicit_on(&loader, &["mock-a", "mock-b"]).app();

    let (status, _, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat("mock-b@agent"))).await;
    assert_eq!(status, StatusCode::OK, "{completion}");
    assert_eq!(completion["model"], "mock-b");
    let (_, _, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!((models["status"].as_str(), models["data"][0]["id"].as_str()), (Some("serving"), Some("mock-b")));

    let (status, _, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat("mock-a"))).await;
    assert_eq!(status, StatusCode::OK, "the start model is known without a flag of its own: {completion}");
    assert_eq!(completion["model"], "mock-a");
    assert_eq!(loader.loads(), ["mock-b", "mock-a"]);
    assert_eq!(loader.resident_at_load(), [0, 0], "teardown before load, as an explicit switch");
}

/// AC 20, 21 over the wire: while a request's switch runs, every other
/// request — on the old model or the new — is refused as the gate refuses
/// during any switch, and the triggering request is held, then answered.
#[tokio::test]
async fn while_a_requests_switch_runs_everything_else_is_refused_and_it_is_held_then_answered() {
    let loader = MockLoader::new();
    let app = implicit_on(&loader, &["mock-a", "mock-b"]).app();
    let release = loader.hold_next_load();
    let trigger = tokio::spawn({
        let app = app.clone();
        async move { send(&app, Method::POST, "/v1/chat/completions", Some(chat("mock-b"))).await }
    });
    let models = until_status(&app, "switching").await;
    assert_eq!(models["switching"], json!({ "from": "mock-a", "to": "mock-b" }));

    for (uri, body) in [
        ("/v1/chat/completions", chat("mock-a")),
        ("/v1/chat/completions", chat("mock-b")),
        ("/v1/responses", response_body("mock-b")),
        ("/v1/decide", decision("mock-a")),
    ] {
        let (status, headers, refused) = send(&app, Method::POST, uri, Some(body)).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{uri}: {refused}");
        assert_eq!(refused["error"]["code"], "model_switching", "{uri}: {refused}");
        assert_eq!(headers[header::RETRY_AFTER], "5");
    }
    assert!(!trigger.is_finished(), "the triggering request is held until the switch ends");

    release.send(()).expect("the held load is waiting");
    let (status, _, completion) = trigger.await.expect("the triggering request");
    assert_eq!(status, StatusCode::OK, "{completion}");
    assert_eq!(completion["model"], "mock-b");
    assert_eq!(loader.loads(), ["mock-b"], "one switch");
}

/// AC 17 on `/v1/responses`, over HTTP and over its WebSocket mode.
#[tokio::test]
async fn a_response_naming_a_known_model_is_answered_by_it_after_the_switch_over_http_and_socket() {
    let loader = MockLoader::new();
    let server = implicit_on(&loader, &["mock-a", "mock-b"]);
    let app = server.app();
    let (status, _, response) = send(&app, Method::POST, "/v1/responses", Some(response_body("mock-b"))).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["model"], "mock-b");
    assert_eq!(server.active().model_id(), "mock-b");

    let live = responses::live(server.clone()).await;
    let mut socket = responses::socket(&live).await;
    let mut create = response_body("mock-a");
    create["type"] = json!("response.create");
    responses::send(&mut socket, create).await;
    let events = responses::events_until(&mut socket, responses::terminal).await;
    let last = events.last().expect("a terminal event");
    assert!(matches!(last["type"].as_str(), Some("response.completed" | "response.incomplete")), "{events:?}");
    assert_eq!(last["response"]["model"], "mock-a");
    assert_eq!(server.active().model_id(), "mock-a");
    assert_eq!(loader.loads(), ["mock-b", "mock-a"]);
}

/// A socket outlives the loads under it, and the mock's loads count their
/// requests from 0 each, as the server's did until 2026-10-20: a response's
/// id is not its request's (spec responses-api/01, Output items), so the
/// latest responses of two streams on one socket — the Playground's
/// sessions — never share one, and a continuation on a stream continues that
/// stream's latest. Seven other
/// streams, each answered on a load of its own, on three sockets: a lookup
/// that could pick another stream's history would pick it 511 times in 512.
#[tokio::test]
async fn a_continuation_continues_its_own_stream_after_responses_on_other_streams_and_loads() {
    let loader = MockLoader::new();
    let server = implicit_on(&loader, &["mock-a", "mock-b"]);
    let live = responses::live(server.clone()).await;
    let long = vec!["limoncello"; 40].join(" ");
    let mut model = "mock-a";
    for round in 0..3 {
        let mut socket = responses::socket(&live).await;
        let mut ids = Vec::new();
        let mut other_input_tokens = 0;
        for n in 0..8 {
            // Every response on a load of its own: the request names the model not loaded.
            model = if model == "mock-a" { "mock-b" } else { "mock-a" };
            let (stream, input) = if n == 7 { ("own".to_owned(), "hello there") } else { (format!("other-{n}"), long.as_str()) };
            let create = json!({
                "type": "response.create", "stream_id": stream, "model": model, "input": input,
                "max_output_tokens": 3, "enable_thinking": false,
            });
            responses::send(&mut socket, create).await;
            let events = responses::events_until(&mut socket, responses::terminal).await;
            let last = events.last().expect("a terminal event");
            assert!(matches!(last["type"].as_str(), Some("response.completed" | "response.incomplete")), "{events:?}");
            ids.push(last["response"]["id"].as_str().unwrap().to_owned());
            if n < 7 {
                other_input_tokens = last["response"]["usage"]["input_tokens"].as_u64().unwrap();
            }
        }
        for id in &ids {
            let hex = id.strip_prefix("resp_").unwrap_or_default();
            assert!(hex.len() == 32 && hex.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()), "resp_<32 hex>: {id}");
        }
        let distinct: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(distinct.len(), ids.len(), "round {round}: one id per response, whatever load answered it: {ids:?}");

        let create = json!({
            "type": "response.create", "stream_id": "own", "model": model, "input": "and then?",
            "previous_response_id": ids[7], "max_output_tokens": 3, "enable_thinking": false,
        });
        responses::send(&mut socket, create).await;
        let events = responses::events_until(&mut socket, responses::terminal).await;
        let last = events.last().expect("a terminal event");
        assert!(matches!(last["type"].as_str(), Some("response.completed" | "response.incomplete")), "{events:?}");
        let input_tokens = last["response"]["usage"]["input_tokens"].as_u64().unwrap();
        assert!(
            input_tokens < other_input_tokens,
            "round {round}: the continuation read {input_tokens} tokens, at least another stream's whole first prompt \
             ({other_input_tokens}): it continued that stream's history, not its own"
        );
    }
}

/// `/v1/decide` pulls a switch too, ahead of the refusal a Flash-Next load
/// gives every decision: naming the 27B there moves the server back to the
/// model that serves it.
#[tokio::test]
async fn a_decision_naming_a_known_model_switches_first_even_from_a_load_that_serves_none() {
    let loader = MockLoader::new();
    let start = loader.model("mock-fn").with_family(ignis_core::compute::ModelFamily::FlashNext);
    let known = ["mock-fn", "mock-a"].iter().map(|id| (id.to_string(), switch_support::source(id).artifact)).collect();
    let server = Server::from_active(start)
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT).with_known_models(known));
    let app = server.app();
    let (status, _, refused) = send(&app, Method::POST, "/v1/decide", Some(decision("mock-fn"))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "the loaded Flash-Next serves no decision: {refused}");
    assert_eq!(refused["error"]["code"], "model_unsupported");

    let (status, _, answered) = send(&app, Method::POST, "/v1/decide", Some(decision("mock-a@interactive"))).await;
    assert_eq!(server.active().model_id(), "mock-a", "{answered}");
    assert_eq!(loader.loads(), ["mock-a"]);
    // The mock template has no answer labels, so the 27B-family mock refuses
    // the question itself — past the model checks, which is what this shows.
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{answered}");
    assert_ne!(answered["error"]["code"], "model_not_found", "{answered}");
    assert_ne!(answered["error"]["code"], "model_unsupported", "{answered}");
}

/// AC 19: a model nobody listed, or any other model with implicit switching
/// off, is refused by name exactly as before — and nothing loads.
#[tokio::test]
async fn an_unlisted_model_or_any_with_switching_off_is_refused_by_name_as_before() {
    let loader = MockLoader::new();
    let listed = implicit_on(&loader, &["mock-a", "mock-b"]).app();
    let off = server_on(&loader).app();
    for (app, model) in [(&listed, "mock-z"), (&off, "mock-b")] {
        let (status, _, refused) = send(app, Method::POST, "/v1/chat/completions", Some(chat(model))).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{model}: {refused}");
        assert_eq!(refused["error"]["code"], "model_not_found");
        let (status, _, refused) = send(app, Method::POST, "/v1/responses", Some(response_body(model))).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{model}: {refused}");
        assert_eq!(refused["error"]["code"], "model_not_found");
        let (_, _, models) = send(app, Method::GET, "/v1/models", None).await;
        assert_eq!((models["status"].as_str(), models["data"][0]["id"].as_str()), (Some("serving"), Some("mock-a")));
    }
    assert!(loader.loads().is_empty(), "{:?}", loader.loads());
}

/// A request whose switch does not land is refused with the switch's own
/// reason — never answered by the model reloaded in its place — and told it
/// is not the transient refusal a switch in progress is (no `Retry-After`).
#[tokio::test]
async fn a_request_whose_switch_does_not_land_is_refused_with_its_reason() {
    let loader = MockLoader::new();
    loader.break_load("mock-broken");
    let app = implicit_on(&loader, &["mock-a", "mock-broken"]).app();

    let (status, headers, refused) = send(&app, Method::POST, "/v1/chat/completions", Some(chat("mock-broken"))).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{refused}");
    assert_eq!(refused["error"]["code"], "model_switch_failed");
    let message = refused["error"]["message"].as_str().unwrap();
    assert!(message.contains("mock-broken") && message.contains("simulated kernel load error"), "{message}");
    assert!(headers.get(header::RETRY_AFTER).is_none(), "a retry is not known to fare better");
    let (_, _, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!((models["status"].as_str(), models["data"][0]["id"].as_str()), (Some("serving"), Some("mock-a")));

    let (status, _, refused) = send(&app, Method::POST, "/v1/decide", Some(decision("mock-broken"))).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "decide refuses in its own shape: {refused}");
    assert_eq!(refused["error"]["code"], "model_switch_failed");
    assert!(refused["error"]["message"].as_str().unwrap().contains("simulated kernel load error"), "{refused}");
}

// ── GET /ready across a failed switch (GitHub #327) ──────────────────────

/// A switch whose target fails is reloading the old model: nothing serves
/// meanwhile, but a model is on its way back, so readiness stays green.
#[tokio::test]
async fn the_readiness_probe_stays_200_while_a_failed_switch_reloads_the_old_model() {
    let loader = MockLoader::new();
    loader.break_load("mock-broken");
    drop(loader.hold_next_load()); // the target's load runs, and fails
    let reload = loader.hold_next_load(); // the old model's reload waits
    let app = server_on(&loader).app();

    let (status, _, _) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to("mock-broken"))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    until_status(&app, "failed").await;
    let (status, _, body) = send(&app, Method::GET, "/ready", None).await;
    assert_eq!((status, &body), (StatusCode::OK, &json!({ "status": "ok" })), "the old model is reloading");

    reload.send(()).expect("the reload is held");
    until_serving(&app).await;
    assert_eq!(send(&app, Method::GET, "/ready", None).await.0, StatusCode::OK);
}

/// A switch that failed with nothing it could reload leaves the process
/// with no model and none loading, until a switch succeeds: the one state
/// after the first warm-up readiness answers 503, without `Retry-After`
/// (asking again does not help; a switch does). The process is alive all
/// along: `/health` stays 200.
#[tokio::test]
async fn the_readiness_probe_answers_503_when_nothing_serves_and_nothing_loads() {
    let loader = MockLoader::new();
    loader.break_load("mock-broken");
    let mut first = loader.model("mock-a");
    first.source = None;
    let server = Server::from_active(first).with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT));
    let app = server.app();

    let (status, _, _) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to("mock-broken"))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    until_status(&app, "failed").await;
    // The switch's run lets go of its slot just after the status says so.
    let mut probe = send(&app, Method::GET, "/ready", None).await;
    for _ in 0..10_000 {
        if probe.0 == StatusCode::SERVICE_UNAVAILABLE {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
        probe = send(&app, Method::GET, "/ready", None).await;
    }
    let (status, headers, body) = probe;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["code"], "server_not_ready", "{body}");
    assert!(headers.get(header::RETRY_AFTER).is_none(), "{headers:?}");
    assert_eq!(send(&app, Method::GET, "/health", None).await.0, StatusCode::OK, "alive, though not ready");
    // `/v1` says the same: no model is coming, so no `Retry-After`, and the
    // way back is a switch.
    let (status, headers, refused) = send(&app, Method::POST, "/v1/chat/completions", Some(chat("mock-a"))).await;
    assert_eq!((status, &refused["error"]["code"]), (StatusCode::SERVICE_UNAVAILABLE, &json!("server_not_ready")), "{refused}");
    assert!(headers.get(header::RETRY_AFTER).is_none(), "{headers:?}");
    assert!(refused["error"]["message"].as_str().unwrap().contains("switch to a model that loads"), "{refused}");

    let (status, _, _) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to("mock-b"))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    until_serving(&app).await;
    assert_eq!(send(&app, Method::GET, "/ready", None).await.0, StatusCode::OK);
}
