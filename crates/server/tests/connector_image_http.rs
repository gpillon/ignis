//! The connector's `generate_image` (spec #656, tickets #661 and #663),
//! driven as a harness drives it: `tools/call` over MCP's Streamable HTTP,
//! `POST /v1/mcp`, on the mock-backed router (`support/switch.rs`'s loader,
//! its image model on `FakeImageRunner`).
//!
//! What a harness sees: a preview of at most 512 px and one line of text, the
//! catalog's image model named so the server switches to it and the next chat
//! turn switches back, the errors when no image model is on disk or implicit
//! switching is off, the answer with vision off, several calls in one turn at
//! one switch, a full image queue as the tool's error, and progress.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use ignis_artifact::vision::ProcessorOptions;
use ignis_core::image::ImageSize;
use ignis_server::config::ApiKey;
use ignis_server::download::Catalog;
use ignis_server::media::{MediaAcquirer, MediaPolicy};
use ignis_server::model_switch::{CatalogModels, Switcher, TextModelLeft};
use ignis_server::Server;
use serde_json::{json, Value};

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::MockLoader;

#[path = "support/media.rs"]
mod media;

#[path = "support/harness.rs"]
mod harness;
use harness::{config_file, exchange, mcp, rpc, started};

#[path = "support/repo.rs"]
mod repo;

#[path = "support/model_routes.rs"]
mod routes;
use routes::{send, until_status, PATIENT};

const KEY: &str = "sk-connector-image-test";

/// The one text a result carries.
fn text_of(result: &Value) -> String {
    let content = result["content"].as_array().expect("content");
    assert_eq!(content.len(), 1, "{result}");
    content[0]["text"].as_str().unwrap().to_owned()
}

/// The image model's served id: its family's own, the built-in catalog's
/// image entry.
const IMAGE: &str = "ignis-image-2.1-turbo";

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

/// A server on `mock-a` with vision on, its catalog the built-in one over an
/// empty `download.path`: the ids in `on_disk` are on disk at the mock's
/// artifact, so the image entry is `complete` when it is one of them. With
/// `implicit`, a request naming one of them switches to it
/// (`switch.allow_implicit`); without, only a switch by id does. It makes
/// two image sizes, one past 512 px and one within.
fn image_server(test: &str, on_disk: &[&str], implicit: bool) -> (Arc<MockLoader>, Server) {
    image_server_seeing(test, on_disk, implicit, true)
}

/// [`image_server`], `mock-a`'s vision on or off as `vision` says.
fn image_server_seeing(test: &str, on_disk: &[&str], implicit: bool, vision: bool) -> (Arc<MockLoader>, Server) {
    let loader = MockLoader::new();
    if vision {
        loader.vision("mock-a", acquirer());
    }
    let table: BTreeMap<String, PathBuf> = on_disk.iter().map(|id| (id.to_string(), switch_support::source(id).artifact)).collect();
    let dir = std::env::temp_dir().join(format!("ignis-connector-http-{test}-models-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let catalog = CatalogModels { catalog: Arc::new(Catalog::built_in().clone()), dir };
    let switcher = Switcher::new(Arc::clone(&loader) as _, PATIENT).with_catalog(catalog, implicit);
    let switcher = if implicit { switcher.with_known_models(table) } else { switcher.with_listed_models(table) };
    let server = Server::from_active(loader.model("mock-a")).with_switcher(switcher);
    server.update_live(|live| live.image.sizes = vec![ImageSize::new(1024, 640), ImageSize::new(256, 128)]);
    (loader, server)
}

/// `generate_image`'s `result` for `arguments`.
async fn generate(app: &axum::Router, arguments: Value, key: Option<&str>) -> Value {
    rpc(app, "tools/call", json!({ "name": "generate_image", "arguments": arguments }), key).await
}

fn unbase64(text: &str) -> Vec<u8> {
    let value = |c: u8| match c {
        b'A'..=b'Z' => c - b'A',
        b'a'..=b'z' => c - b'a' + 26,
        b'0'..=b'9' => c - b'0' + 52,
        b'+' => 62,
        b'/' => 63,
        _ => panic!("not base64: {c}"),
    };
    let mut out = Vec::new();
    for chunk in text.as_bytes().chunks(4) {
        let digits: Vec<u8> = chunk.iter().filter(|&&c| c != b'=').map(|&c| value(c)).collect();
        let n = digits.iter().enumerate().fold(0u32, |n, (i, &d)| n | u32::from(d) << (18 - 6 * i));
        out.extend((0..digits.len() - 1).map(|i| (n >> (16 - 8 * i)) as u8));
    }
    out
}

/// The width and height of a result's image block, a PNG.
fn image_size(block: &Value) -> (u32, u32) {
    assert_eq!(block["type"], "image", "{block}");
    assert_eq!(block["mimeType"], "image/png", "{block}");
    let reader = png::Decoder::new(std::io::Cursor::new(unbase64(block["data"].as_str().unwrap()))).read_info().expect("a PNG");
    (reader.info().width, reader.info().height)
}

/// Spec #656: with vision on, `generate_image` answers the image first, a
/// PNG preview whose longest edge is at most 512 px and never enlarged, then
/// one line of text: the size made, that this is a preview, and the skill's
/// `generate-image` script for the full size. The prompt reaches the image
/// model as sent.
#[tokio::test]
async fn generate_image_answers_a_preview_of_at_most_512_px_and_one_line_of_text() {
    let (loader, server) = image_server("preview", &[IMAGE, "mock-a"], true);
    let app = server.app();
    let result = generate(&app, json!({ "prompt": "a red fox in the snow", "size": "1024x640" }), None).await;
    assert_ne!(result["isError"], json!(true), "{result}");
    let content = result["content"].as_array().expect("content");
    assert_eq!(content.len(), 2, "{result}");
    assert_eq!(image_size(&content[0]), (512, 320));
    assert_eq!(content[1]["type"], "text", "{result}");
    let line = content[1]["text"].as_str().unwrap();
    assert_eq!(line.lines().count(), 1, "{line}");
    for said in ["1024x640", "512x320", "preview", "generate-image"] {
        assert!(line.contains(said), "{said}: {line}");
    }
    assert_eq!(loader.encoded_prompts(), ["a red fox in the snow"]);

    // An image within 512 px is shown at its own size.
    let result = generate(&app, json!({ "prompt": "a small fox", "size": "256x128" }), None).await;
    let content = result["content"].as_array().expect("content");
    assert_eq!(image_size(&content[0]), (256, 128), "{result}");
    assert!(content[1]["text"].as_str().unwrap().contains("256x128"), "{result}");
}

/// A chat turn naming `model`, sent with `key`, as a harness's next turn.
async fn chat_turn(app: &axum::Router, model: &str, key: &str) -> (StatusCode, Value) {
    let body = json!({ "model": model, "messages": [{ "role": "user", "content": "what did you draw?" }], "max_tokens": 3 });
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {key}"));
    let (status, _, bytes) = exchange(app, request.body(Body::from(body.to_string())).unwrap()).await;
    (status, serde_json::from_slice(&bytes).unwrap())
}

/// Spec #656: `generate_image` reads the catalog and names the image model
/// it has on disk, with the caller's key, so the server switches to it (a
/// request naming no model is refused by a text model). No tool switches
/// back: the agent's next chat turn names its own model, and the implicit
/// switch returns to it.
#[tokio::test]
async fn generate_image_names_the_catalogs_image_model_and_the_next_chat_turn_switches_back() {
    let (loader, server) = image_server("switch-back", &[IMAGE, "mock-a"], true);
    let app = server.clone().with_api_key(ApiKey::new(KEY)).app();
    let result = generate(&app, json!({ "prompt": "a lighthouse at dusk", "size": "1024x640" }), Some(KEY)).await;
    assert_ne!(result["isError"], json!(true), "{result}");
    assert_eq!(loader.loads(), [IMAGE]);
    assert_eq!(server.active().model_id(), IMAGE);

    let (status, answer) = chat_turn(&app, "mock-a", KEY).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert_eq!(answer["model"], "mock-a");
    assert_eq!(loader.loads(), [IMAGE, "mock-a"]);
    assert_eq!(server.active().model_id(), "mock-a");
}

/// Spec #656: with no image model on disk, `generate_image` is the tool's
/// error, the server's `model_not_found`, naming the command that fetches
/// the catalog's image model (and not `download_model`, which is not
/// listed); nothing loads.
#[tokio::test]
async fn without_the_image_model_on_disk_generate_image_names_its_download() {
    let (loader, server) = image_server("absent", &["mock-a"], true);
    let app = server.app();
    let result = generate(&app, json!({ "prompt": "a lighthouse at dusk" }), None).await;
    assert_eq!(result["isError"], true, "{result}");
    let content = result["content"].as_array().expect("content");
    assert_eq!(content.len(), 1, "{result}");
    let error: Value = serde_json::from_str(content[0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(error["error"]["code"], "model_not_found", "{error}");
    let message = error["error"]["message"].as_str().unwrap();
    assert!(message.contains(&format!("`ignis-server model download {IMAGE}`")), "{message}");
    assert!(!message.contains("download_model"), "{message}");
    assert!(loader.loads().is_empty());
    assert_eq!(server.active().model_id(), "mock-a");
}

/// Spec #656, tickets #660 and #661: with `connector.allow_changes` on, so
/// `download_model` is listed, the same error names that tool beside the
/// command; nothing loads.
#[tokio::test]
async fn with_download_model_listed_generate_image_names_it_beside_the_command() {
    let (loader, server) = image_server("absent-listed", &["mock-a"], true);
    let path = config_file("absent-listed");
    let config = started(&["--config", path.to_str().unwrap(), "--profile", "none", "--connector-allow-changes", "true"]);
    let app = server.with_connector(config.settings().connector.clone()).app();
    let result = generate(&app, json!({ "prompt": "a lighthouse at dusk" }), None).await;
    assert_eq!(result["isError"], true, "{result}");
    let error: Value = serde_json::from_str(&text_of(&result)).unwrap();
    assert_eq!(error["error"]["code"], "model_not_found", "{error}");
    let message = error["error"]["message"].as_str().unwrap();
    assert!(message.contains(&format!("`ignis-server model download {IMAGE}`")), "{message}");
    assert!(message.contains(&format!("download_model with `model` {IMAGE}")), "{message}");
    assert!(loader.loads().is_empty());
}

/// Spec #656: with `switch.allow_implicit` off, the image route refuses the
/// image model by name, `404`, and `generate_image` passes that answer on
/// as its error, word for word; nothing loads.
#[tokio::test]
async fn with_implicit_switching_off_generate_image_passes_the_servers_404_on() {
    let (loader, server) = image_server("implicit-off", &[IMAGE, "mock-a"], false);
    let app = server.app();
    let result = generate(&app, json!({ "prompt": "a lighthouse at dusk" }), None).await;
    assert_eq!(result["isError"], true, "{result}");
    let error: Value = serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap();
    let (status, direct) = send(&app, Method::POST, "/v1/images/generations", Some(json!({ "model": IMAGE, "prompt": "a lighthouse at dusk" }))).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{direct}");
    assert_eq!(error, direct);
    assert!(loader.loads().is_empty());
}

// ── generate_image: vision, several calls, the queue, progress (#663) ────

/// The text of a result whose one block is text.
fn only_text(result: &Value) -> &str {
    let content = result["content"].as_array().expect("content");
    assert_eq!(content.len(), 1, "{result}");
    assert_eq!(content[0]["type"], "text", "{result}");
    content[0]["text"].as_str().unwrap()
}

/// The answer a call gets when the agent's text model has vision off: one
/// text, not an error, naming the skill's `generate-image` script and its
/// install in a POSIX shell form and a PowerShell form.
fn assert_points_at_the_skill(result: &Value) {
    assert_ne!(result["isError"], json!(true), "{result}");
    let text = only_text(result);
    for said in ["vision", "generate-image", "/v1/connector/skill", "tar xz -C", "curl.exe", "$env:IGNIS_API_KEY"] {
        assert!(text.contains(said), "{said}: {text}");
    }
}

/// Spec #656: with vision off in the agent's text model, `generate_image`
/// answers at once, naming the skill's script, and switches nothing: no
/// load, no switch recorded, no image made.
#[tokio::test]
async fn with_vision_off_generate_image_answers_at_once_naming_the_skill_and_switches_nothing() {
    let (loader, server) = image_server_seeing("vision-off", &[IMAGE, "mock-a"], true, false);
    let app = server.app();
    let result = generate(&app, json!({ "prompt": "a lighthouse at dusk" }), None).await;
    assert_points_at_the_skill(&result);
    assert!(loader.loads().is_empty(), "{:?}", loader.loads());
    assert_eq!(server.switcher.as_ref().unwrap().last_switch(), None);
    assert!(loader.encoded_prompts().is_empty());
    assert_eq!(server.active().model_id(), "mock-a");
}

/// Spec #656: the vision check reads the text model the server switched
/// away from, which the server remembers while the image model is loaded:
/// with the image model loaded by the first call, a second call is judged
/// by `mock-a`, which has vision, and makes its image with no further
/// switch. A server that remembers none takes the agent to see, so the
/// memory is asserted itself.
#[tokio::test]
async fn a_second_call_is_judged_by_the_text_model_switched_away_from_and_switches_no_further() {
    let (loader, server) = image_server("remembered", &[IMAGE, "mock-a"], true);
    let app = server.app();
    for prompt in ["a fox", "a second fox"] {
        let result = generate(&app, json!({ "prompt": prompt, "size": "256x128" }), None).await;
        assert_eq!(result["content"][0]["type"], "image", "{prompt}: {result}");
        assert_eq!(server.active().model_id(), IMAGE);
        let remembered = server.switcher.as_ref().unwrap().text_model_left();
        assert_eq!(remembered, Some(TextModelLeft { id: "mock-a".to_owned(), vision: true }), "{prompt}");
    }
    assert_eq!(loader.loads(), [IMAGE]);
    assert_eq!(loader.encoded_prompts(), ["a fox", "a second fox"]);
}

/// Spec #656: the image model loaded by another client's request, from a
/// text model with vision off: `generate_image` is judged by that text
/// model, not by the image model loaded, and answers at once naming the
/// skill; nothing more loads, no image is made for it.
#[tokio::test]
async fn the_image_model_loaded_from_a_text_model_without_vision_gets_the_skill_answer() {
    let (loader, server) = image_server_seeing("remembered-off", &[IMAGE, "mock-a"], true, false);
    let app = server.app();
    let (status, made) = send(&app, Method::POST, "/v1/images/generations", Some(json!({ "model": IMAGE, "prompt": "a client's own", "size": "256x128" }))).await;
    assert_eq!(status, StatusCode::OK, "{made}");
    assert_eq!(server.active().model_id(), IMAGE);
    let result = generate(&app, json!({ "prompt": "a lighthouse at dusk" }), None).await;
    assert_points_at_the_skill(&result);
    assert_eq!(loader.loads(), [IMAGE]);
    assert_eq!(loader.encoded_prompts(), ["a client's own"]);
}

/// Spec #656: behind the connector the image route's own queue serves
/// every client, and a full one is an ordinary tool error: the server's
/// `engine_full`, then its `Retry-After`; the job queued ahead is
/// another client's.
#[tokio::test]
async fn a_full_image_queue_is_the_tools_error_with_its_retry_after() {
    let (_, server) = image_server("queue-full", &[IMAGE, "mock-a"], true);
    let app = server.app();
    let first = generate(&app, json!({ "prompt": "a fox", "size": "256x128" }), None).await;
    assert_eq!(first["content"][0]["type"], "image", "{first}");
    server.update_live(|live| live.image.queue_depth = 0);
    let jobs = Arc::clone(&server.active().image_model().expect("the image model is loaded").jobs);
    let ahead = jobs.wait();
    let result = generate(&app, json!({ "prompt": "a second fox", "size": "256x128" }), None).await;
    assert_eq!(result["isError"], true, "{result}");
    let content = result["content"].as_array().expect("content");
    assert_eq!(content.len(), 2, "{result}");
    let error: Value = serde_json::from_str(content[0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(error["error"]["code"], "engine_full", "{error}");
    assert_eq!(content[1]["text"], "Retry-After: 1");
    drop(ahead);
    let again = generate(&app, json!({ "prompt": "a third fox", "size": "256x128" }), None).await;
    assert_eq!(again["content"][0]["type"], "image", "{again}");
}

/// `server`'s connector until `n` image calls wait for their turn
/// (scheduling turns, bounded).
async fn until_waiting(server: &Server, n: usize) {
    for _ in 0..10_000 {
        if server.image_calls.waiting() == n {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("{n} image calls never waited: {} do", server.image_calls.waiting());
}

/// Spec #656: the connector's image calls run one at a time. Sent while the
/// first call's switch is held in its load, a second and a third call wait
/// for it to end, in order, rather than meet that switch's `503
/// model_switching`, which a client's own request meets meanwhile; each then
/// finds the image model loaded. Three calls in one turn: one switch out,
/// three images. The third carries a progress token: it hears that it
/// waits, then its job's steps, and no switch.
#[tokio::test]
async fn three_calls_in_one_turn_make_one_switch_and_the_ones_sent_mid_switch_wait_instead_of_503() {
    let (loader, server) = image_server("three", &[IMAGE, "mock-a"], true);
    let app = server.app();
    let release = loader.hold_next_load();
    let call = |prompt: &str| {
        let (app, arguments) = (app.clone(), json!({ "prompt": prompt, "size": "256x128" }));
        tokio::spawn(async move { generate(&app, arguments, None).await })
    };
    let first = call("first");
    until_status(&app, "switching").await;
    let second = call("second");
    until_waiting(&server, 1).await;
    let third = {
        let app = app.clone();
        tokio::spawn(async move { generate_hearing(&app, json!({ "prompt": "third", "size": "256x128" }), "third").await })
    };
    until_waiting(&server, 2).await;
    let (status, refused) = send(&app, Method::POST, "/v1/images/generations", Some(json!({ "model": IMAGE, "prompt": "a client's own" }))).await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::SERVICE_UNAVAILABLE, json!("model_switching")));

    release.send(()).expect("the held load is waiting");
    let (notes, third) = third.await.unwrap();
    for (result, prompt) in [(first.await.unwrap(), "first"), (second.await.unwrap(), "second"), (third, "third")] {
        assert_ne!(result["isError"], json!(true), "{prompt}: {result}");
        assert_eq!(result["content"][0]["type"], "image", "{prompt}: {result}");
    }
    let phases = phases_of(&notes);
    assert_eq!(phases.first().map(String::as_str), Some("waiting"), "{notes:?}");
    assert!(phases.iter().any(|phase| phase == "denoising"), "{notes:?}");
    assert!(!phases.iter().any(|phase| phase == "switching" || phase == "loading"), "no switch for it: {notes:?}");
    assert_eq!(loader.loads(), [IMAGE]);
    assert_eq!(loader.encoded_prompts(), ["first", "second", "third"]);
    assert_eq!(server.image_calls.waiting(), 0);
}

/// `server` until no request uses the model (scheduling turns, bounded).
async fn until_unused(server: &Server) {
    for _ in 0..10_000 {
        if server.usage.in_flight() == 0 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("{} requests still use the model", server.usage.in_flight());
}

/// Spec #656: a call its harness gives up while the switch it began runs (a
/// timeout closing its request) keeps the turn until that switch ends. The
/// switch runs on without it, so the call sent next waits for it, as it would
/// for a call that had not been given up, rather than meet its `503
/// model_switching`, which a client's own request meets meanwhile; it then
/// finds the image model loaded. The call given up makes no image.
#[tokio::test]
async fn a_call_given_up_mid_switch_holds_the_next_one_until_the_switch_ends() {
    let (loader, server) = image_server("given-up", &[IMAGE, "mock-a"], true);
    let app = server.app();
    let release = loader.hold_next_load();
    let given_up = {
        let (app, arguments) = (app.clone(), json!({ "prompt": "given up", "size": "256x128" }));
        tokio::spawn(async move { generate(&app, arguments, None).await })
    };
    until_status(&app, "switching").await;
    // The harness closes the request: the call is given up, and its request
    // of the image route, held on the switch, is dropped with it.
    given_up.abort();
    until_unused(&server).await;

    let next = {
        let (app, arguments) = (app.clone(), json!({ "prompt": "next", "size": "256x128" }));
        tokio::spawn(async move { generate(&app, arguments, None).await })
    };
    until_waiting(&server, 1).await;
    let (status, refused) = send(&app, Method::POST, "/v1/images/generations", Some(json!({ "model": IMAGE, "prompt": "a client's own" }))).await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::SERVICE_UNAVAILABLE, json!("model_switching")));

    release.send(()).expect("the held load is waiting");
    let next = next.await.unwrap();
    assert_ne!(next["isError"], json!(true), "{next}");
    assert_eq!(next["content"][0]["type"], "image", "{next}");
    assert_eq!(loader.loads(), [IMAGE]);
    assert_eq!(loader.encoded_prompts(), ["next"]);
    assert_eq!(server.image_calls.waiting(), 0);
}

/// `server` until `id` serves, its status aside (scheduling turns, bounded).
async fn until_active(server: &Server, id: &str) {
    for _ in 0..10_000 {
        if server.active().model_id() == id {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("{id} never served: {} does", server.active().model_id());
}

/// Spec #656: a call is judged by the agent's text model once it has its
/// turn. With `image.prompt_enhancement` on, the call ahead visits the
/// enhancer on its way to the image model, and the enhancer serves
/// meanwhile: a call sent then is judged by `mock-a`, the agent's model, which
/// has vision, not by the enhancer, `mock-b`, which has none, and makes its
/// image.
#[tokio::test]
async fn a_call_sent_while_the_enhancer_serves_is_judged_by_the_agents_model() {
    let (loader, server) = image_server("enhancer-serves", &[IMAGE, "mock-a", "mock-b"], true);
    server.update_live(|live| {
        live.image.prompt_enhancement = true;
        live.image.prompt_enhancer = "mock-b".to_owned();
        live.image.prompt_enhancer_max_tokens = 16;
    });
    let app = server.app();
    let to_the_enhancer = loader.hold_next_load();
    let on_to_the_image_model = loader.hold_next_load();
    let call = |prompt: &str| {
        let (app, arguments) = (app.clone(), json!({ "prompt": prompt, "size": "256x128" }));
        tokio::spawn(async move { generate(&app, arguments, None).await })
    };
    let first = call("first");
    to_the_enhancer.send(()).expect("the load of the enhancer is waiting");
    until_active(&server, "mock-b").await;
    let second = call("second");
    until_waiting(&server, 1).await;

    on_to_the_image_model.send(()).expect("the load of the image model is waiting");
    for (result, prompt) in [(first.await.unwrap(), "first"), (second.await.unwrap(), "second")] {
        assert_eq!(result["content"][0]["type"], "image", "{prompt}: {result}");
    }
    assert_eq!(loader.loads(), ["mock-b", IMAGE, "mock-b", IMAGE]);
}

/// `generate_image`'s call with `arguments` and the progress token `token`,
/// answered with an event stream: the `params` of the progress
/// notifications it was sent, in order, then its `result`.
async fn generate_hearing(app: &axum::Router, arguments: Value, token: &str) -> (Vec<Value>, Value) {
    let params = json!({ "name": "generate_image", "arguments": arguments, "_meta": { "progressToken": token } });
    let body = json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params });
    let (status, headers, bytes) = exchange(app, mcp(&body, None)).await;
    let text = String::from_utf8_lossy(&bytes).into_owned();
    assert_eq!(status, StatusCode::OK, "{text}");
    assert!(headers[header::CONTENT_TYPE].to_str().unwrap().starts_with("text/event-stream"), "{headers:?}: {text}");
    let mut messages: Vec<Value> = text
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .filter(|data| !data.is_empty())
        .map(|data| serde_json::from_str(data).unwrap_or_else(|err| panic!("{err}: {data}")))
        .collect();
    let answer = messages.pop().expect("the answer");
    assert_eq!(answer["id"], 1, "{answer}");
    for note in &messages {
        assert_eq!(note["method"], "notifications/progress", "{note}");
        assert_eq!(note["params"]["progressToken"], token, "{note}");
    }
    let result = answer.get("result").cloned().unwrap_or_else(|| panic!("no result: {answer}"));
    (messages.into_iter().map(|note| note["params"].clone()).collect(), result)
}

/// The phase each progress notification names: the first word of its
/// message.
fn phases_of(notes: &[Value]) -> Vec<String> {
    notes.iter().map(|note| note["message"].as_str().unwrap_or_default().split(' ').next().unwrap().to_owned()).collect()
}

/// Spec #656: a call that carries a progress token hears the image job's
/// phases as MCP progress notifications, before its result: the switch to
/// the image model, its load, then the job's steps, each notification's
/// `progress` past the last. A call without one is answered JSON, as every
/// other test's is.
#[tokio::test]
async fn a_call_with_a_progress_token_hears_the_image_jobs_phases_as_progress_notifications() {
    let (loader, server) = image_server("progress", &[IMAGE, "mock-a"], true);
    let app = server.app();
    let (notes, result) = generate_hearing(&app, json!({ "prompt": "a fox", "size": "256x128" }), "image-1").await;
    assert_eq!(result["content"][0]["type"], "image", "{result}");
    assert_eq!(loader.loads(), [IMAGE]);
    let phases = phases_of(&notes);
    assert_eq!(&phases[..3], ["switching", "loading", "encoding"], "{notes:?}");
    assert!(phases[3..phases.len() - 1].iter().all(|phase| phase == "denoising"), "{notes:?}");
    assert!(phases.len() > 4, "{notes:?}");
    assert_eq!(phases.last().map(String::as_str), Some("decoding"), "{notes:?}");
    let progress: Vec<f64> = notes.iter().map(|note| note["progress"].as_f64().expect("progress")).collect();
    assert!(progress.windows(2).all(|pair| pair[0] < pair[1]), "{progress:?}");
}
