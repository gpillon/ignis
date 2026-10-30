//! The card class (ADR 0064, spec 27b-16gb/04 §6, GitHub #336) over the
//! mock-backed router: on a 16 GB card a switch to a catalog entry above
//! the card's class — explicit, or a request's implicit one — is refused
//! `409 model_above_card_class` before anything is prepared, drained or torn
//! down, and the loaded model keeps serving; a request naming the 27B's
//! default id is served by the 16 GB entry that answers to it as an alias,
//! and only by an entry whose artifact is of the 27B's family;
//! `GET /v1/models` and `GET /v1/models/catalog` list such entries as
//! unavailable, with the reason. On a 32 GB card, or one of no known class,
//! nothing changes. The catalog is the built-in one, which holds the 16 GB
//! 27B (GitHub #338); every artifact is a file under `download.path`, and the
//! models are `support/switch.rs`'s mocks, loaded by id. An alias may also be
//! another entry's id of the same family (ADR 0074): the uncensored variant's
//! 16 GB build, from the 16 GB fixture catalog over the built-in one, answers
//! to `qwen3.8-27b-abliterated` on a 16 GB card.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_core::compute::ModelFamily;
use ignis_server::config::file::Format;
use ignis_server::config::profile::CardClass;
use ignis_server::download::catalog::{parse_operator, Catalog};
use ignis_server::model_switch::{CatalogModels, Switcher};
use ignis_server::{ModelStatus, Server};
use serde_json::{json, Value};
use tower::ServiceExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::MockLoader;

#[path = "support/responses.rs"]
mod responses;

const PATIENT: Duration = Duration::from_secs(30);

/// The 16 GB 27B's served id and artifact, and the id it answers to.
const SIXTEEN: &str = "qwen3.8-27b-16g";
const SIXTEEN_ARTIFACT: &str = "qwen3_8_27b_trellis_sc30-v2.ninfer";
const DEFAULT_27B: &str = "qwen3.8-27b";
const FLASH_NEXT: &str = "qwen3.8-flash-next";
const IMAGE: &str = "ignis-image-2.1-turbo";
/// The uncensored variant's served id, its 32 GB entry's, and the id of its
/// 16 GB build (ADR 0074), which the 16 GB fixture catalog holds and the
/// built-in catalog does not yet (GitHub #423).
const ABLITERATED: &str = "qwen3.8-27b-abliterated";
const ABLITERATED_16G: &str = "qwen3.8-27b-abliterated-16g";
/// The 16 GB fixture catalog, the one the catalog's own tests read.
const SIXTEEN_CATALOG: &str = include_str!("fixtures/catalog/sixteen.yaml");
/// The catalog's CPU models: the three speech entries and the diarizer.
const CPU_MODELS: [&str; 4] = ["qwen3-asr-1.7b", "qwen3-asr-1.7b-q8", "qwen3-asr-0.6b", "nemotron-3-diarization"];

/// A condition image an edit decodes.
const PNG: &[u8] = include_bytes!("../../artifact/tests/fixtures/vision/images/on_grid.png");

/// The built-in catalog and the operator's `yaml` (`models: []` for none),
/// every entry's artifact a file under a fresh `download.path`.
fn catalog_on_disk(tag: &str, yaml: &str) -> CatalogModels {
    let dir = std::env::temp_dir().join(format!("ignis-card-class-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let catalog = parse_operator(yaml, Format::Yaml, "test").expect("a valid catalog");
    for entry in catalog.entries() {
        std::fs::write(entry.artifact_path(&dir), b"mock").unwrap();
    }
    CatalogModels { catalog: Arc::new(catalog), dir }
}

/// A server on `start`, on a card of class `card`, whose requests may switch
/// to any built-in catalog entry on disk, the 16 GB 27B's artifact's header
/// naming the 27B's family.
fn server_on(loader: &Arc<MockLoader>, start: &str, card: Option<CardClass>, tag: &str) -> Server {
    loader.header(SIXTEEN, ModelFamily::Qwen38_27b);
    server_over(loader, start, card, catalog_on_disk(tag, "models: []"))
}

fn server_over(loader: &Arc<MockLoader>, start: &str, card: Option<CardClass>, catalog: CatalogModels) -> Server {
    Server::from_active(loader.model(start))
        .with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT).with_catalog(catalog, true).with_card_class(card))
}

/// The 16 GB fixture catalog's entries the built-in catalog does not hold —
/// the uncensored variant's 16 GB build — as an operator's catalog over it.
fn fixture_over_built_in() -> String {
    let mut fixture = Format::Yaml.read(SIXTEEN_CATALOG).expect("the fixture is YAML");
    fixture["models"].as_array_mut().unwrap().retain(|entry| Catalog::built_in().entry(entry["id"].as_str().unwrap()).is_none());
    fixture.to_string()
}

/// [`server_on`] over the built-in catalog and the uncensored variant's
/// 16 GB build, every artifact on disk, the build's header naming `header`.
fn abliterated_server(loader: &Arc<MockLoader>, start: &str, card: Option<CardClass>, header: ModelFamily, tag: &str) -> Server {
    loader.header(SIXTEEN, ModelFamily::Qwen38_27b);
    loader.header(ABLITERATED_16G, header);
    server_over(loader, start, card, catalog_on_disk(tag, &fixture_over_built_in()))
}

/// Where the catalog entry `id`'s artifact is on `server`'s disk.
fn artifact_of(server: &Server, id: &str) -> std::path::PathBuf {
    let catalog = server.switcher.as_ref().unwrap().catalog().unwrap();
    catalog.catalog.entries().iter().find(|entry| entry.id == id).unwrap().artifact_path(&catalog.dir)
}

async fn send(app: &axum::Router, method: Method, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder().method(method).uri(uri).header(header::CONTENT_TYPE, "application/json");
    let request = request.body(body.map_or_else(Body::empty, |body| Body::from(body.to_string()))).unwrap();
    respond(app, request).await
}

async fn respond(app: &axum::Router, request: Request<Body>) -> (StatusCode, Value) {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body = serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, body)
}

/// `POST /v1/images/edits` naming `model`, with a prompt and one condition
/// image.
async fn edit(app: &axum::Router, model: &str) -> (StatusCode, Value) {
    const BOUNDARY: &str = "ignis-card-class-boundary";
    let mut body = Vec::new();
    for (name, text) in [("model", model), ("prompt", "a fox")] {
        body.extend(format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{text}\r\n").bytes());
    }
    body.extend(
        format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"image\"\r\nContent-Type: application/octet-stream\r\n\r\n")
            .bytes(),
    );
    body.extend_from_slice(PNG);
    body.extend(format!("\r\n--{BOUNDARY}--\r\n").bytes());
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/images/edits")
        .header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={BOUNDARY}"))
        .body(Body::from(body))
        .unwrap();
    respond(app, request).await
}

/// One `response.create` over the Responses WebSocket naming `model`, and
/// the events up to its end or its `error` — with the listener, which the
/// caller keeps while it reads the server: once dropped, its serve stops,
/// and the model it shares with the router with it.
async fn over_the_socket(server: &Server, model: &str) -> (responses::Live, Vec<Value>) {
    let live = responses::live(server.clone()).await;
    let mut socket = responses::socket(&live).await;
    let mut create = response_body(model);
    create["type"] = json!("response.create");
    responses::send(&mut socket, create).await;
    let events = responses::events_until(&mut socket, responses::terminal).await;
    (live, events)
}

fn chat(model: &str) -> Value {
    json!({ "model": model, "messages": [{ "role": "user", "content": "hello there" }], "max_tokens": 3 })
}

fn response_body(model: &str) -> Value {
    json!({ "model": model, "input": "hello there", "max_output_tokens": 3, "enable_thinking": false })
}

fn decision(model: &str) -> Value {
    json!({ "state": "s", "model": model, "questions": { "q": { "type": "noul", "instructions": "Urgent?" } } })
}

/// `GET /v1/models` until its `status` is `serving` (scheduling turns,
/// bounded).
async fn until_serving(app: &axum::Router) -> Value {
    for _ in 0..10_000 {
        let (status, body) = send(app, Method::GET, "/v1/models", None).await;
        if status == StatusCode::OK && body["status"] == "serving" {
            return body;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("`GET /v1/models` never reported serving");
}

/// A refusal naming the entry's class and the card's.
fn refused_by_class(status: StatusCode, body: &Value, model: &str) {
    assert_eq!(status, StatusCode::CONFLICT, "{model}: {body}");
    assert_eq!(body["error"]["code"], "model_above_card_class", "{body}");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains(model) && message.contains("32 GB") && message.contains("16 GB"), "{message}");
}

/// The server still serves `loaded`, nothing was prepared or loaded, and a
/// request naming no model is answered by it.
async fn still_serving(app: &axum::Router, server: &Server, loader: &MockLoader, loaded: &str) {
    assert!(loader.prepared_artifacts().is_empty(), "nothing was prepared: {:?}", loader.prepared_artifacts());
    assert!(loader.loads().is_empty(), "nothing was loaded: {:?}", loader.loads());
    assert_eq!(loader.resident(), 1, "the loaded model was never torn down");
    assert_eq!(*server.status(), ModelStatus::Serving);
    let (_, models) = send(app, Method::GET, "/v1/models", None).await;
    assert_eq!((models["status"].as_str(), models["data"][0]["id"].as_str()), (Some("serving"), Some(loaded)), "{models}");
    assert!(models.get("last_switch").is_none(), "no switch began, so none is recorded: {models}");
    let (status, completion) = send(app, Method::POST, "/v1/chat/completions", Some(json!({ "messages": [{ "role": "user", "content": "hi" }], "max_tokens": 3 }))).await;
    assert_eq!(status, StatusCode::OK, "{completion}");
    assert_eq!(completion["model"], loaded);
}

/// ADR 0064 §3: `POST /v1/models/switch` by id to an entry above the card's
/// class is refused before anything stops — not even prepared, so the
/// status never leaves `serving` — and the loaded model serves on. A switch
/// naming its own artifact is the operator's word (ADR 0047), which only the
/// VRAM plan guards.
#[tokio::test]
async fn an_explicit_switch_to_an_entry_above_the_card_class_is_refused_before_anything_stops() {
    let loader = MockLoader::new();
    let server = server_on(&loader, SIXTEEN, Some(CardClass::Gb16), "explicit");
    let app = server.app();
    for model in [FLASH_NEXT, IMAGE, "qwen3.8-27b-abliterated"] {
        let (status, refused) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": model }))).await;
        refused_by_class(status, &refused, model);
    }
    still_serving(&app, &server, &loader, SIXTEEN).await;

    let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": FLASH_NEXT, "artifact": "flash.ninfer" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    assert_eq!(until_serving(&app).await["data"][0]["id"], FLASH_NEXT);
}

/// ADR 0064 §3: a request naming an entry above the card's class is refused
/// before its implicit switch begins, on every route that switches — chat,
/// Responses over HTTP and over its WebSocket, image generations and edits,
/// decisions (`422`, its own refusal shape) — and on the counting route; the
/// loaded model serves on.
#[tokio::test]
async fn a_request_naming_an_entry_above_the_card_class_is_refused_before_anything_stops() {
    let loader = MockLoader::new();
    let server = server_on(&loader, SIXTEEN, Some(CardClass::Gb16), "implicit");
    let app = server.app();
    for model in [FLASH_NEXT, &format!("{FLASH_NEXT}@agent")] {
        let (status, refused) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(model))).await;
        refused_by_class(status, &refused, FLASH_NEXT);
    }
    let (status, refused) = send(&app, Method::POST, "/v1/responses", Some(response_body(FLASH_NEXT))).await;
    refused_by_class(status, &refused, FLASH_NEXT);
    let (_live, events) = over_the_socket(&server, FLASH_NEXT).await;
    let refused = events.last().expect("an event");
    assert_eq!((refused["type"].as_str(), refused["status"].as_u64()), (Some("error"), Some(409)), "{events:?}");
    refused_by_class(StatusCode::CONFLICT, refused, FLASH_NEXT);
    let (status, refused) = send(&app, Method::POST, "/v1/tokenize", Some(json!({ "model": FLASH_NEXT, "messages": [{ "role": "user", "content": "count me" }] }))).await;
    refused_by_class(status, &refused, FLASH_NEXT);
    let (status, refused) = send(&app, Method::POST, "/v1/images/generations", Some(json!({ "model": IMAGE, "prompt": "a fox", "size": "1024x1024" }))).await;
    refused_by_class(status, &refused, IMAGE);
    let (status, refused) = edit(&app, IMAGE).await;
    refused_by_class(status, &refused, IMAGE);
    let (status, refused) = send(&app, Method::POST, "/v1/decide", Some(decision(FLASH_NEXT))).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{refused}");
    assert_eq!(refused["error"]["code"], "model_above_card_class", "{refused}");
    assert!(refused["error"]["message"].as_str().unwrap().contains("16 GB"), "{refused}");
    still_serving(&app, &server, &loader, SIXTEEN).await;
}

/// ADR 0064 §2: on a 16 GB card with the 16 GB 27B loaded, a request naming
/// the 27B's default id is served by it — no switch, no refusal — on every
/// route, lane tag and all; the answer names the model that served it. On
/// the image routes the id the kind check reads is the loaded 16 GB 27B's,
/// a text model, never the default id.
#[tokio::test]
async fn on_a_16gb_card_the_default_id_is_served_by_the_entry_that_answers_to_it() {
    let loader = MockLoader::new();
    let server = server_on(&loader, SIXTEEN, Some(CardClass::Gb16), "alias");
    let app = server.app();
    // The lane tag is kept, an empty one included.
    for model in [DEFAULT_27B, &format!("{DEFAULT_27B}@agent"), &format!("{DEFAULT_27B}@")] {
        let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(model))).await;
        assert_eq!(status, StatusCode::OK, "{model}: {completion}");
        assert_eq!(completion["model"], SIXTEEN);
    }
    let (status, answered) = send(&app, Method::POST, "/v1/responses", Some(response_body(DEFAULT_27B))).await;
    assert_eq!(status, StatusCode::OK, "{answered}");
    assert_eq!(answered["model"], SIXTEEN);
    let (_live, events) = over_the_socket(&server, DEFAULT_27B).await;
    let last = events.last().expect("a terminal event");
    assert!(matches!(last["type"].as_str(), Some("response.completed" | "response.incomplete")), "{events:?}");
    assert_eq!(last["response"]["model"], SIXTEEN, "{last}");
    let (status, counted) = send(&app, Method::POST, "/v1/tokenize", Some(json!({ "model": DEFAULT_27B, "messages": [{ "role": "user", "content": "count me" }] }))).await;
    assert_eq!(status, StatusCode::OK, "{counted}");
    let (status, decided) = send(&app, Method::POST, "/v1/decide", Some(decision(DEFAULT_27B))).await;
    // The mock template has no answer labels: refused past the model checks.
    assert_ne!(decided["error"]["code"], "model_not_found", "{status} {decided}");
    assert_ne!(decided["error"]["code"], "model_above_card_class", "{status} {decided}");
    let generation = send(&app, Method::POST, "/v1/images/generations", Some(json!({ "model": DEFAULT_27B, "prompt": "a fox", "size": "1024x1024" }))).await;
    for (status, refused) in [generation, edit(&app, DEFAULT_27B).await] {
        assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::CONFLICT, Some("model_kind_mismatch")), "{refused}");
        let message = refused["error"]["message"].as_str().unwrap();
        assert!(message.contains(&format!("the loaded model {SIXTEEN} is a text model")) && !message.contains(&format!("names {DEFAULT_27B},")), "{message}");
    }
    assert!(loader.loads().is_empty(), "served by the loaded model, no switch: {:?}", loader.loads());
    assert_eq!(server.active().model_id(), SIXTEEN);
}

/// ADR 0064 §2: from another model, the default id switches to the entry
/// that answers to it on this card, never to its own 32 GB one — and the
/// model loaded stays served by its own id whatever its class.
#[tokio::test]
async fn on_a_16gb_card_the_default_id_switches_to_the_entry_that_answers_to_it() {
    let loader = MockLoader::new();
    let server = server_on(&loader, FLASH_NEXT, Some(CardClass::Gb16), "alias-switch");
    let app = server.app();
    let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(FLASH_NEXT))).await;
    assert_eq!(status, StatusCode::OK, "the loaded model is served, whatever its class: {completion}");
    assert!(loader.loads().is_empty());

    let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(DEFAULT_27B))).await;
    assert_eq!(status, StatusCode::OK, "{completion}");
    assert_eq!(completion["model"], SIXTEEN);
    assert_eq!(loader.loads(), [SIXTEEN]);
    let dir = server.switcher.as_ref().unwrap().catalog().unwrap().dir.clone();
    assert_eq!(loader.prepared_artifacts(), [dir.join(SIXTEEN_ARTIFACT)], "the 16 GB entry's artifact, not the 32 GB one's");

    // And by id: the explicit switch resolves the alias the same way.
    let (status, _) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": FLASH_NEXT, "artifact": "flash.ninfer" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    until_serving(&app).await;
    let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": DEFAULT_27B }))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    assert_eq!(accepted["to"], SIXTEEN, "{accepted}");
    assert_eq!(until_serving(&app).await["data"][0]["id"], SIXTEEN);
}

/// ADR 0064 §2: an alias only ever resolves within one model family, and the
/// family is the artifact's header's (ADR 0047), not the alias's. The entry
/// that answers to the 27B's default id, holding a Flash-Next artifact — on
/// disk, so its header is read before anything stops — does not answer to
/// it: a request or a switch naming the default id is refused naming both
/// families, nothing is prepared, and the lists say why. The same entry with
/// a 27B header answers to it.
#[tokio::test]
async fn an_alias_never_resolves_to_an_artifact_of_another_family() {
    for (header, tag) in [(ModelFamily::FlashNext, "family-other"), (ModelFamily::Qwen38_27b, "family-same")] {
        let loader = MockLoader::new();
        loader.header(SIXTEEN, header);
        let server = server_over(&loader, "mock-a", Some(CardClass::Gb16), catalog_on_disk(tag, "models: []"));
        let app = server.app();
        if header == ModelFamily::Qwen38_27b {
            let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(DEFAULT_27B))).await;
            assert_eq!((status, completion["model"].as_str()), (StatusCode::OK, Some(SIXTEEN)), "{completion}");
            continue;
        }
        let (status, refused) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(DEFAULT_27B))).await;
        refused_by_class(status, &refused, DEFAULT_27B);
        let message = refused["error"]["message"].as_str().unwrap();
        for named in ["qwen3.8-27b-16g lists it as an alias", "Qwen3.8-Flash-Next", "Qwen3.8-27B", "one model family"] {
            assert!(message.contains(named), "{named}: {message}");
        }
        let (status, refused) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": DEFAULT_27B }))).await;
        refused_by_class(status, &refused, DEFAULT_27B);
        still_serving(&app, &server, &loader, "mock-a").await;
        let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
        let default = models["data"].as_array().unwrap().iter().find(|entry| entry["id"] == DEFAULT_27B).cloned().unwrap();
        assert_eq!(default["available"], false, "{default}");
        assert!(default["reason"].as_str().unwrap().contains("one model family"), "{default}");
        // Named by its own id, the entry is the request's word: served.
        let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(SIXTEEN))).await;
        assert_eq!((status, completion["model"].as_str()), (StatusCode::OK, Some(SIXTEEN)), "{completion}");
    }
}

/// Spec 27b-16gb/04 §6: on a 32 GB card — or one of no known class — nothing
/// changes: the default id is its own entry's, and a request naming
/// Flash-Next switches to it.
#[tokio::test]
async fn on_a_32gb_card_or_an_unknown_one_nothing_changes() {
    for (card, tag) in [(Some(CardClass::Gb32), "32"), (None, "unknown")] {
        let loader = MockLoader::new();
        let server = server_on(&loader, DEFAULT_27B, card, tag);
        let app = server.app();
        let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(DEFAULT_27B))).await;
        assert_eq!((status, completion["model"].as_str()), (StatusCode::OK, Some(DEFAULT_27B)), "{completion}");
        assert!(loader.loads().is_empty());
        let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(FLASH_NEXT))).await;
        assert_eq!((status, completion["model"].as_str()), (StatusCode::OK, Some(FLASH_NEXT)), "{tag}: {completion}");
        assert_eq!(loader.loads(), [FLASH_NEXT], "{tag}");
        let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
        let data = models["data"].as_array().unwrap();
        assert!(data.iter().all(|entry| entry["available"] == true && entry.get("reason").is_none()), "{tag}: {models}");
    }
}

/// ADR 0064 §3: `GET /v1/models` lists an entry on disk the card cannot
/// load as unavailable, with the reason — the default id's own entry saying
/// which entry serves a request naming it — and `GET /v1/models/catalog`
/// says the same of every row, none of them switchable.
#[tokio::test]
async fn the_model_lists_mark_entries_above_the_card_class_unavailable_with_the_reason() {
    let loader = MockLoader::new();
    let server = server_on(&loader, SIXTEEN, Some(CardClass::Gb16), "lists");
    let app = server.app();
    let (status, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK);
    let listed: Vec<(&str, bool)> =
        models["data"].as_array().unwrap().iter().map(|entry| (entry["id"].as_str().unwrap(), entry["available"].as_bool().unwrap())).collect();
    assert_eq!(
        listed,
        [(SIXTEEN, true), (DEFAULT_27B, false), ("qwen3.8-27b-abliterated", false), (FLASH_NEXT, false), (IMAGE, false)],
        "the entries it lists today, each marked: {models}"
    );
    let reason = |id: &str| -> String {
        let entry = models["data"].as_array().unwrap().iter().find(|entry| entry["id"] == id).unwrap();
        entry["reason"].as_str().unwrap_or_default().to_owned()
    };
    for id in [DEFAULT_27B, FLASH_NEXT, IMAGE] {
        let reason = reason(id);
        assert!(reason.contains(id) && reason.contains("32 GB") && reason.contains("16 GB"), "{id}: {reason}");
    }
    assert!(reason(DEFAULT_27B).contains(&format!("served by {SIXTEEN}")), "{}", reason(DEFAULT_27B));
    assert!(!reason(FLASH_NEXT).contains("served by"), "{}", reason(FLASH_NEXT));
    assert!(models["data"][0].get("reason").is_none(), "{models}");

    let (status, catalog) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!(status, StatusCode::OK);
    for row in catalog["data"].as_array().unwrap() {
        let id = row["id"].as_str().unwrap();
        // A speech model and the diarizer run on the CPU: no class, so no card refuses them.
        let fits = id == SIXTEEN || CPU_MODELS.contains(&id);
        assert_eq!(row["available"], fits, "{row}");
        assert_eq!(row.get("reason").is_some(), !fits, "{row}");
        if !fits {
            assert_eq!(row["switchable"], false, "{row}");
        }
    }

    let thirty_two = server_on(&MockLoader::new(), SIXTEEN, Some(CardClass::Gb32), "lists-32").app();
    let (_, catalog) = send(&thirty_two, Method::GET, "/v1/models/catalog", None).await;
    for row in catalog["data"].as_array().unwrap() {
        assert_eq!((row["available"].as_bool(), row.get("reason")), (Some(true), None), "{row}");
        assert_eq!(row["switchable"], row["id"] != SIXTEEN && !CPU_MODELS.contains(&row["id"].as_str().unwrap()), "a CPU model is never switched to: {row}");
    }
}

/// ADR 0064 §3: "A switch, explicit or implicit, and a download check the
/// card class first" — the class is the catalog entry's, so a catalog id the
/// operator also listed by path (`switch.known_models`) is fitted all the
/// same: unavailable on both lists, never switchable, and a request or a
/// switch naming it is refused before anything stops. An id that is no
/// catalog entry, the operator's own model, has no class and is never
/// refused by it.
#[tokio::test]
async fn a_catalog_id_the_operator_listed_is_fitted_to_the_card_all_the_same() {
    let loader = MockLoader::new();
    let server = server_on(&loader, SIXTEEN, Some(CardClass::Gb16), "listed");
    let switcher = server.switcher.as_ref().unwrap();
    let listed = [(FLASH_NEXT, switch_support::source(FLASH_NEXT).artifact), ("acme-own", switch_support::source("acme-own").artifact)];
    let listed: std::collections::BTreeMap<String, std::path::PathBuf> = listed.into_iter().map(|(id, artifact)| (id.to_owned(), artifact)).collect();
    switcher.set_knobs(PATIENT, listed.clone(), true);
    switcher.set_listed_models(listed);
    let app = server.app();
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    let entry = |id: &str| models["data"].as_array().unwrap().iter().find(|entry| entry["id"] == id).cloned().unwrap();
    assert_eq!(entry(FLASH_NEXT)["available"], false, "{models}");
    assert!(entry(FLASH_NEXT)["reason"].as_str().unwrap().contains("16 GB"), "{models}");
    assert_eq!((entry("acme-own")["available"].as_bool(), entry("acme-own").get("reason")), (Some(true), None), "{models}");
    let (_, catalog) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    let row = catalog["data"].as_array().unwrap().iter().find(|row| row["id"] == FLASH_NEXT).cloned().unwrap();
    assert_eq!((row["available"].as_bool(), row["switchable"].as_bool()), (Some(false), Some(false)), "{row}");
    let (status, refused) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": FLASH_NEXT }))).await;
    refused_by_class(status, &refused, FLASH_NEXT);
    let (status, refused) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(FLASH_NEXT))).await;
    refused_by_class(status, &refused, FLASH_NEXT);
    still_serving(&app, &server, &loader, SIXTEEN).await;

    let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat("acme-own"))).await;
    assert_eq!((status, completion["model"].as_str()), (StatusCode::OK, Some("acme-own")), "the operator's own model: {completion}");
}

/// ADR 0064 §2 for `image.prompt_enhancer`, `qwen3.8-27b` by default: it
/// takes the step a request's `model` takes, so on a 16 GB card the rewrite
/// runs on the entry that answers to the default id — the loaded one in
/// place, never a visit to itself, or the one a visit loads. With no such
/// entry it is an enhancer no visit reaches, which the start and a `PATCH`
/// say; on a 32 GB card it is its own entry's.
#[tokio::test]
async fn the_prompt_enhancer_is_served_by_the_entry_that_answers_to_it() {
    use ignis_server::images::enhance::{enhancer_on_this_card, unknown_enhancer};
    let none = std::collections::BTreeMap::new();
    let loaded = server_on(&MockLoader::new(), SIXTEEN, Some(CardClass::Gb16), "enhancer-loaded");
    assert_eq!(enhancer_on_this_card(&loaded, DEFAULT_27B), SIXTEEN, "the loaded model rewrites in place");
    assert_eq!(unknown_enhancer(&loaded, DEFAULT_27B, &none), None);

    let visited = server_on(&MockLoader::new(), FLASH_NEXT, Some(CardClass::Gb16), "enhancer-visit");
    assert_eq!(enhancer_on_this_card(&visited, DEFAULT_27B), SIXTEEN);
    let via = visited.switcher.as_ref().unwrap().text_model(SIXTEEN).expect("a visit loads the 16 GB 27B");
    assert_eq!(via.artifact.file_name().and_then(|name| name.to_str()), Some(SIXTEEN_ARTIFACT));
    assert_eq!(unknown_enhancer(&visited, DEFAULT_27B, &none), None);

    // An enhancer no entry answers to on this card: the abliterated 27B's id.
    const ABLITERATED: &str = "qwen3.8-27b-abliterated";
    let alone = server_on(&MockLoader::new(), FLASH_NEXT, Some(CardClass::Gb16), "enhancer-alone");
    assert_eq!(enhancer_on_this_card(&alone, ABLITERATED), ABLITERATED);
    assert!(unknown_enhancer(&alone, ABLITERATED, &none).is_some_and(|reason| reason.contains(ABLITERATED)));

    let thirty_two = server_on(&MockLoader::new(), FLASH_NEXT, Some(CardClass::Gb32), "enhancer-32");
    assert_eq!(enhancer_on_this_card(&thirty_two, DEFAULT_27B), DEFAULT_27B, "its own entry fits");
    assert_eq!(unknown_enhancer(&thirty_two, DEFAULT_27B, &none), None);
}

/// ADR 0074: on a 16 GB card a request naming the uncensored variant's id
/// switches to its 16 GB build, whose artifact is on disk — never to the
/// 32 GB entry's — and is served under the build's own id; once it is
/// loaded, a request naming the id is served by it, lane tag and all, with
/// no switch. An explicit switch by the id resolves it the same way.
#[tokio::test]
async fn on_a_16gb_card_the_uncensored_variants_id_is_served_by_its_16gb_build() {
    let loader = MockLoader::new();
    let server = abliterated_server(&loader, SIXTEEN, Some(CardClass::Gb16), ModelFamily::Qwen38_27b, "abliterated");
    let app = server.app();
    let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(ABLITERATED))).await;
    assert_eq!((status, completion["model"].as_str()), (StatusCode::OK, Some(ABLITERATED_16G)), "{completion}");
    assert_eq!(loader.loads(), [ABLITERATED_16G]);
    assert_eq!(loader.prepared_artifacts(), [artifact_of(&server, ABLITERATED_16G)], "the 16 GB build's artifact, not the 32 GB entry's");
    let (status, answered) = send(&app, Method::POST, "/v1/responses", Some(response_body(&format!("{ABLITERATED}@agent")))).await;
    assert_eq!((status, answered["model"].as_str()), (StatusCode::OK, Some(ABLITERATED_16G)), "{answered}");
    assert_eq!(loader.loads(), [ABLITERATED_16G], "served by the loaded build, no switch");

    let (status, _) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": SIXTEEN }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    assert_eq!(until_serving(&app).await["data"][0]["id"], SIXTEEN);
    let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": ABLITERATED }))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    assert_eq!(accepted["to"], ABLITERATED_16G, "{accepted}");
    assert_eq!(until_serving(&app).await["data"][0]["id"], ABLITERATED_16G);
    assert_eq!(loader.loads(), [ABLITERATED_16G, SIXTEEN, ABLITERATED_16G]);
}

/// ADR 0074: the implicit switch to the 16 GB build happens only when its
/// artifact is on disk. Without it, a request or a switch naming the
/// uncensored variant's id is refused before anything stops, naming the
/// build — never served by the 32 GB entry, which this card cannot load.
#[tokio::test]
async fn the_switch_to_the_uncensored_variants_16gb_build_needs_its_artifact_on_disk() {
    let loader = MockLoader::new();
    let server = abliterated_server(&loader, SIXTEEN, Some(CardClass::Gb16), ModelFamily::Qwen38_27b, "abliterated-absent");
    std::fs::remove_file(artifact_of(&server, ABLITERATED_16G)).unwrap();
    let app = server.app();
    let (status, refused) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(ABLITERATED))).await;
    assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::NOT_FOUND, Some("model_not_found")), "{refused}");
    assert!(refused["error"]["message"].as_str().unwrap().contains(ABLITERATED_16G), "{refused}");
    let (status, refused) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": ABLITERATED }))).await;
    assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::NOT_FOUND, Some("model_not_found")), "{refused}");
    still_serving(&app, &server, &loader, SIXTEEN).await;
}

/// ADR 0074: the header stays the authority. The 16 GB build states the
/// 27B's family; holding an artifact whose header names Flash-Next, it does
/// not answer to the uncensored variant's id: a request and a switch naming
/// it are refused before anything stops, naming both families, and the
/// lists say why. Named by its own id, the build is the request's word.
#[tokio::test]
async fn an_uncensored_variants_16gb_build_whose_header_names_another_family_does_not_answer() {
    let loader = MockLoader::new();
    let server = abliterated_server(&loader, SIXTEEN, Some(CardClass::Gb16), ModelFamily::FlashNext, "abliterated-header");
    let app = server.app();
    let (status, refused) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(ABLITERATED))).await;
    refused_by_class(status, &refused, ABLITERATED);
    let message = refused["error"]["message"].as_str().unwrap();
    for named in [&format!("{ABLITERATED_16G} lists it as an alias"), "its artifact is Qwen3.8-Flash-Next", &format!("{ABLITERATED} is Qwen3.8-27B"), "one model family"] {
        assert!(message.contains(named), "{named}: {message}");
    }
    let (status, refused) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": ABLITERATED }))).await;
    refused_by_class(status, &refused, ABLITERATED);
    still_serving(&app, &server, &loader, SIXTEEN).await;
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    let listed = models["data"].as_array().unwrap().iter().find(|entry| entry["id"] == ABLITERATED).cloned().unwrap();
    assert_eq!(listed["available"], false, "{listed}");
    assert!(listed["reason"].as_str().unwrap().contains("one model family"), "{listed}");

    let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(ABLITERATED_16G))).await;
    assert_eq!((status, completion["model"].as_str()), (StatusCode::OK, Some(ABLITERATED_16G)), "{completion}");
}

/// ADR 0074: on a 32 GB card, a 96 GB one or one of no known class nothing
/// changes for the uncensored variant: its id is its own entry's, for a
/// request's switch and an explicit one, whatever the 16 GB build's header.
#[tokio::test]
async fn on_other_cards_the_uncensored_variants_id_is_its_own_entrys() {
    for (card, tag) in [(Some(CardClass::Gb32), "abliterated-32"), (Some(CardClass::Gb96), "abliterated-96"), (None, "abliterated-unknown")] {
        let loader = MockLoader::new();
        let server = abliterated_server(&loader, DEFAULT_27B, card, ModelFamily::Qwen38_27b, tag);
        let app = server.app();
        let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(ABLITERATED))).await;
        assert_eq!((status, completion["model"].as_str()), (StatusCode::OK, Some(ABLITERATED)), "{tag}: {completion}");
        assert_eq!(loader.prepared_artifacts(), [artifact_of(&server, ABLITERATED)], "{tag}: the 32 GB entry's own artifact");
        let (status, _) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": DEFAULT_27B }))).await;
        assert_eq!(status, StatusCode::ACCEPTED, "{tag}");
        until_serving(&app).await;
        let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": ABLITERATED }))).await;
        assert_eq!((status, accepted["to"].as_str()), (StatusCode::ACCEPTED, Some(ABLITERATED)), "{tag}: {accepted}");
        assert_eq!(until_serving(&app).await["data"][0]["id"], ABLITERATED, "{tag}");
        assert_eq!(loader.loads(), [ABLITERATED, DEFAULT_27B, ABLITERATED], "{tag}");
        let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
        let data = models["data"].as_array().unwrap();
        assert!(data.iter().all(|entry| entry["available"] == true && entry.get("reason").is_none()), "{tag}: {models}");
    }
}

/// ADR 0074, ADR 0064 §3: on a 16 GB card `GET /v1/models` lists the
/// uncensored variant's 16 GB build, and marks its 32 GB entry unavailable
/// with the reason, naming the build that serves a request for its id here;
/// `GET /v1/models/catalog` says the same of their rows.
#[tokio::test]
async fn on_a_16gb_card_the_model_lists_name_the_build_that_serves_the_uncensored_variant() {
    let loader = MockLoader::new();
    let server = abliterated_server(&loader, SIXTEEN, Some(CardClass::Gb16), ModelFamily::Qwen38_27b, "abliterated-lists");
    let app = server.app();
    let (status, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK);
    let entry = |id: &str| models["data"].as_array().unwrap().iter().find(|entry| entry["id"] == id).cloned().unwrap_or_else(|| panic!("{id} listed: {models}"));
    assert_eq!((entry(ABLITERATED_16G)["available"].as_bool(), entry(ABLITERATED_16G).get("reason")), (Some(true), None), "{models}");
    assert_eq!(entry(ABLITERATED)["available"], false, "{models}");
    let reason = entry(ABLITERATED)["reason"].as_str().unwrap().to_owned();
    for said in [ABLITERATED, "32 GB", "16 GB", &format!("served by {ABLITERATED_16G}")] {
        assert!(reason.contains(said), "{said}: {reason}");
    }

    let (status, catalog) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!(status, StatusCode::OK);
    let row = |id: &str| catalog["data"].as_array().unwrap().iter().find(|row| row["id"] == id).cloned().unwrap();
    assert_eq!((row(ABLITERATED_16G)["available"].as_bool(), row(ABLITERATED_16G)["switchable"].as_bool()), (Some(true), Some(true)), "{catalog}");
    assert_eq!((row(ABLITERATED)["available"].as_bool(), row(ABLITERATED)["switchable"].as_bool()), (Some(false), Some(false)), "{catalog}");
    assert!(row(ABLITERATED)["reason"].as_str().unwrap().contains(&format!("served by {ABLITERATED_16G}")), "{catalog}");
}
