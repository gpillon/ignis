//! A model switch keeps vision (spec #555, GitHub #563), over the mock-backed
//! router (`support/switch.rs`): the request size cap follows the model
//! serving after a switch, whichever way it goes, and a switch to a model
//! whose vision companion is missing is refused `409` naming the file while
//! the active model serves on. A switch carries the vision of the model it
//! replaces (ADR 0087): a 27B on its built-in profile's vision switches to
//! Flash-Next with vision on, and so asks for its companion, unless vision
//! is named off for Flash-Next.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_artifact::vision::ProcessorOptions;
use ignis_core::compute::ModelFamily;
use ignis_server::config::file::NoFiles;
use ignis_server::config::{resolve, ConfigOutcome};
use ignis_server::config_http::ConfigState;
use ignis_server::media::{MediaAcquirer, MediaPolicy};
use ignis_server::model_switch::Switcher;
use ignis_server::Server;
use serde_json::{json, Value};
use tower::ServiceExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::MockLoader;

#[path = "support/media.rs"]
mod media;

const PATIENT: Duration = Duration::from_secs(30);

fn acquirer() -> Arc<MediaAcquirer> {
    let limits = ProcessorOptions {
        min_pixels: 32 * 32,
        max_pixels: 1 << 20,
        max_encoded_media_bytes: 256 << 20,
        max_decoded_pixels: 1 << 24,
        max_raw_patches: 1 << 17,
        max_vision_tokens: 1 << 15,
    };
    Arc::new(MediaAcquirer::new(Arc::new(media::processor(limits.clone())), limits, MediaPolicy::new(false, 0)))
}

async fn send(app: &axum::Router, method: Method, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder().method(method).uri(uri).header(header::CONTENT_TYPE, "application/json");
    let request = request.body(body.map_or_else(Body::empty, |body| Body::from(body.to_string()))).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// Switch to `model` and wait until it serves: the whole end state the tests
/// read after it. The request size cap is the serving model's `media`, a
/// field of the model a switch publishes before it stores `serving`, and
/// `GET /v1/models` reads the status before the model, so a `serving` read
/// beside the new id is never paired with the old cap. The status leaves
/// `serving` before the `202`, so the wait never ends on the model replaced.
async fn switch_and_settle(app: &axum::Router, model: &str) {
    let (status, accepted) = send(app, Method::POST, "/v1/models/switch", Some(json!({ "artifact": format!("{model}.ninfer"), "model": model }))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    for _ in 0..10_000 {
        let (_, models) = send(app, Method::GET, "/v1/models", None).await;
        if models["status"] == "serving" && models["data"][0]["id"] == model {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("the switch to {model} never settled");
}

/// A chat whose one text part is `text_bytes` long, with one image part: a
/// body of that size, as a request with an image inline has.
async fn post_sized(app: &axum::Router, model: &str, text_bytes: usize) -> StatusCode {
    let image = media::data_uri(&media::png(64, 64));
    let body = json!({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "x".repeat(text_bytes)},
            {"type": "image_url", "image_url": {"url": image}},
        ]}],
        "max_tokens": 1,
    });
    send(app, Method::POST, "/v1/chat/completions", Some(body)).await.0
}

/// Past the text-only cap, inside the media one.
const BETWEEN: usize = ignis_server::api::TEXT_REQUEST_BODY_LIMIT + (1 << 20);

#[tokio::test]
async fn the_request_size_cap_follows_the_model_serving_across_a_switch_both_ways() {
    let loader = MockLoader::new();
    // The 27B takes images and Flash-Next does too: two vision loads.
    loader.vision("mock-27b", acquirer());
    loader.vision("mock-fn", acquirer());
    let server = Server::from_active(loader.model("mock-27b")).with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT));
    let app = server.app();
    assert_ne!(post_sized(&app, "mock-27b", BETWEEN).await, StatusCode::PAYLOAD_TOO_LARGE, "vision on at start");
    switch_and_settle(&app, "mock-fn").await;
    assert_ne!(post_sized(&app, "mock-fn", BETWEEN).await, StatusCode::PAYLOAD_TOO_LARGE, "an image within the cap after the switch to Flash-Next");
    switch_and_settle(&app, "mock-27b").await;
    assert_ne!(post_sized(&app, "mock-27b", BETWEEN).await, StatusCode::PAYLOAD_TOO_LARGE, "and back");
}

#[tokio::test]
async fn a_text_start_that_switches_to_a_vision_model_takes_its_cap_and_a_switch_back_returns_the_text_one() {
    let loader = MockLoader::new();
    loader.vision("mock-vision", acquirer());
    let server = Server::from_active(loader.model("mock-text")).with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT));
    let app = server.app();
    assert_eq!(post_sized(&app, "mock-text", BETWEEN).await, StatusCode::PAYLOAD_TOO_LARGE, "a text model's own cap");
    switch_and_settle(&app, "mock-vision").await;
    assert_ne!(post_sized(&app, "mock-vision", BETWEEN).await, StatusCode::PAYLOAD_TOO_LARGE, "the cap the vision model takes");
    switch_and_settle(&app, "mock-text").await;
    assert_eq!(post_sized(&app, "mock-text", BETWEEN).await, StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn a_switch_to_a_model_with_its_companion_missing_is_refused_by_name_and_the_old_model_serves() {
    let loader = MockLoader::new();
    loader.vision("mock-27b", acquirer());
    loader.vision("mock-fn", acquirer());
    loader.without_companion("mock-fn");
    let server = Server::from_active(loader.model("mock-27b")).with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT));
    let app = server.app();
    let (status, body) =
        send(&app, Method::POST, "/v1/models/switch", Some(json!({ "artifact": "mock-fn.ninfer", "model": "mock-fn" }))).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["code"], "vision_companion_missing", "{body}");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("mock-fn.ninfer.vision"), "names the file: {message}");
    // Nothing was prepared, loaded or stopped: the 27B serves on, an image included.
    assert!(loader.loads().is_empty() && loader.prepared_artifacts().is_empty(), "{:?}", loader.loads());
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!((models["status"].as_str(), models["data"][0]["id"].as_str()), (Some("serving"), Some("mock-27b")), "{models}");
    assert_eq!(post_sized(&app, "mock-27b", 0).await, StatusCode::OK);
    // A switch to a model that lacks nothing still begins.
    switch_and_settle(&app, "mock-27b").await;
}

/// A 27B serving `argv`'s options, fitted to its family, on a server whose
/// loader fits every load from the running configuration as the production
/// loader does: `mock-27b` a 27B, `mock-fn` Flash-Next, both built taking
/// images, as a load with vision on is.
fn fitted_27b(loader: &Arc<MockLoader>, argv: &[&str]) -> Server {
    let argv: Vec<String> = argv.iter().map(|arg| arg.to_string()).collect();
    let ConfigOutcome::Config(started) = resolve(&argv, |_| None).expect("the start options resolve") else { panic!("a config") };
    let config = started.for_family(ModelFamily::Qwen38_27b).expect("a 27B start");
    let state = Arc::new(ConfigState::new(config.clone(), Arc::new(NoFiles)));
    loader.fit_from(Arc::clone(&state), &[("mock-27b", ModelFamily::Qwen38_27b), ("mock-fn", ModelFamily::FlashNext)]);
    loader.vision("mock-27b", acquirer());
    loader.vision("mock-fn", acquirer());
    let server = Server::from_active(loader.model("mock-27b"))
        .with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT))
        .with_config(state);
    server.update_active(|model| model.family = ModelFamily::Qwen38_27b);
    server.apply_config(&config);
    server
}

/// ADR 0087: a 27B on its built-in profile's vision, which the profile
/// turns on for the 27B alone, carries it to Flash-Next: without Flash-Next's
/// companion the switch is refused `409` by the file's name, nothing
/// prepared, the 27B serving on; with it, Flash-Next loads with vision on.
#[tokio::test]
async fn a_switch_carries_the_27bs_profile_vision_to_flash_next_and_asks_for_its_companion() {
    let loader = MockLoader::new();
    loader.without_companion("mock-fn");
    let app = fitted_27b(&loader, &["--profile", "rtx5090"]).app();
    let (status, body) =
        send(&app, Method::POST, "/v1/models/switch", Some(json!({ "artifact": "mock-fn.ninfer", "model": "mock-fn" }))).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["code"], "vision_companion_missing", "{body}");
    assert!(loader.loads().is_empty() && loader.prepared_artifacts().is_empty(), "{:?}", loader.loads());
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!((models["status"].as_str(), models["data"][0]["id"].as_str()), (Some("serving"), Some("mock-27b")), "{models}");

    let loader = MockLoader::new();
    let app = fitted_27b(&loader, &["--profile", "rtx5090"]).app();
    switch_and_settle(&app, "mock-fn").await;
    let fitted = loader.fitted_options();
    let flash_next = fitted.last().expect("the Flash-Next load, fitted");
    assert_eq!(flash_next.basis.family(), Some(ModelFamily::FlashNext));
    assert!(flash_next.vision.is_some(), "the 27B's vision, carried");
    // Shown as Flash-Next's own value, set while the server runs.
    let (_, origins) = send(&app, Method::GET, "/v1/config/origins", None).await;
    let row = origins["data"].as_array().unwrap().iter().find(|row| row["key"] == "vision.enabled" && row["scope"] == "qwen38flashnext");
    assert_eq!(row.map(|row| (row["source"].clone(), row["scoped"].clone())), Some((json!("live"), json!(true))), "{origins}");
    switch_and_settle(&app, "mock-27b").await;
    assert!(loader.fitted_options().last().unwrap().vision.is_some(), "and back on the 27B");
}

/// ADR 0087: vision named off for Flash-Next itself wins over the vision a
/// switch carries: the switch asks for no companion and Flash-Next loads
/// without vision.
#[tokio::test]
async fn vision_named_off_for_flash_next_wins_over_the_carried_vision() {
    let loader = MockLoader::new();
    loader.without_companion("mock-fn");
    let app = fitted_27b(&loader, &["--profile", "rtx5090", "--qwen38flashnext-vision-enabled", "false"]).app();
    switch_and_settle(&app, "mock-fn").await;
    assert!(loader.fitted_options().last().expect("the Flash-Next load, fitted").vision.is_none());
}

/// ADR 0087: the carried vision is Flash-Next's own live value while it
/// serves, so a `null` in its section takes it away: a reload, after which
/// Flash-Next serves without vision and the origin is no longer `live`.
#[tokio::test]
async fn a_null_in_flash_nexts_section_takes_the_carried_vision_away() {
    let loader = MockLoader::new();
    let app = fitted_27b(&loader, &["--profile", "rtx5090"]).app();
    switch_and_settle(&app, "mock-fn").await;
    let loads = loader.fitted_options().len();
    assert!(loader.fitted_options()[loads - 1].vision.is_some(), "carried");
    let body = json!({ "vision": { "qwen38flashnext": { "enabled": null } } });
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(body)).await;
    assert_eq!((status, applied["applied"].clone()), (StatusCode::ACCEPTED, json!("reload")), "{applied}");
    for _ in 0..10_000 {
        let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
        if loader.fitted_options().len() > loads && models["status"] == "serving" && models["data"][0]["id"] == "mock-fn" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert!(loader.fitted_options().last().unwrap().vision.is_none(), "the reload runs without the carried vision");
    let (_, origins) = send(&app, Method::GET, "/v1/config/origins", None).await;
    let row = origins["data"].as_array().unwrap().iter().find(|row| row["key"] == "vision.enabled" && row["scope"] == "qwen38flashnext");
    assert_ne!(row.map(|row| row["source"].clone()), Some(json!("live")), "{origins}");
}
