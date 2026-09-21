//! GitHub #251: the `/v1` surface documents itself. The document is
//! generated from the handler annotations by the same call that registers
//! the routes, so these tests are what stops the two drifting — a route
//! added with `.route()` alone fails [`the_document_lists_exactly_the_v1_surface`],
//! and a monitoring path that leaks in fails
//! [`the_document_carries_no_monitoring_path`].
//!
//! The page, the document and their assets answer without a key even on a
//! keyed server (the API's shape is not its load); the handler routes still
//! do not.

use std::collections::BTreeSet;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use http_body_util::BodyExt;
use ignis_core::{ConcreteScheduler, MockCompute, SchedulerConfig};
use ignis_server::Server;
use ignis_server::config::ApiKey;
use ignis_server::engine::Engine;
use ignis_server::template::SimpleTemplateProvider;
use tower::ServiceExt;

const MODEL: &str = "test-model";
const KEY: &str = "sk-test-key";

fn server() -> Server {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider))
}

async fn get(app: &axum::Router, uri: &str) -> axum::response::Response {
    let request = Request::builder().uri(uri).body(Body::empty()).unwrap();
    app.clone().oneshot(request).await.unwrap()
}

async fn body_bytes(response: axum::response::Response) -> Vec<u8> {
    response.into_body().collect().await.unwrap().to_bytes().to_vec()
}

fn header_value(response: &axum::response::Response, name: header::HeaderName) -> String {
    response
        .headers()
        .get(name)
        .map(|v| v.to_str().unwrap().to_owned())
        .unwrap_or_default()
}

/// The document itself, as the JSON a client generator reads. Asserting
/// over the serialized form rather than over `utoipa`'s types keeps these
/// tests about the contract, not about the generator's internals.
fn document() -> serde_json::Value {
    serde_json::to_value(ignis_server::api::openapi()).unwrap()
}

/// The document's paths and the methods each one serves.
fn documented() -> BTreeSet<String> {
    document()["paths"]
        .as_object()
        .expect("paths")
        .iter()
        .flat_map(|(path, item)| {
            item.as_object()
                .expect("a path item")
                .keys()
                .map(move |method| format!("{method} {path}"))
                .collect::<Vec<_>>()
        })
        .collect()
}

// ── the document ─────────────────────────────────────────────────────────

#[test]
fn the_document_lists_exactly_the_v1_surface() {
    // Set equality, both ways: a new `/v1` route that ships without its
    // `#[utoipa::path]` fails here, and so does a path entry whose route
    // was removed.
    let expected: BTreeSet<String> = [
        "get /v1/models",
        "post /v1/chat/completions",
        "post /v1/responses",
        "post /v1/decide",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    assert_eq!(documented(), expected);
}

#[test]
fn the_document_carries_no_monitoring_path() {
    // "Not for the monitoring" (GitHub #251), made checkable: the
    // Prometheus exposition is ADR 0017's contract, served on its own
    // listener, and the Playground's pages are assets rather than API.
    for path in document()["paths"].as_object().expect("paths").keys() {
        assert!(
            !path.contains("metrics") && !path.starts_with("/ui"),
            "the document must not carry a monitoring or Playground path: {path}",
        );
    }
}

#[test]
fn the_alias_is_named_but_not_listed() {
    // `/v1/systemone` is the same handler under Jev's name. OpenAPI has no
    // alias, so it is described in prose rather than duplicated as a path
    // with every schema reference under it repeated.
    let document = document();
    assert!(document["paths"]["/v1/systemone"].is_null());
    let description = document["info"]["description"].as_str().unwrap_or_default();
    assert!(
        description.contains("/v1/systemone"),
        "the alias must be named in the document: {description}",
    );
}

#[test]
fn the_error_envelope_is_one_component() {
    // Shared, not inlined per operation: a client handles failures on this
    // surface with one type.
    let document = document();
    assert!(document["components"]["schemas"]["ApiError"].is_object());
    let four_hundred = document["paths"]["/v1/chat/completions"]["post"]["responses"]["400"]
        ["content"]["application/json"]["schema"]["$ref"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    assert_eq!(four_hundred, "#/components/schemas/ApiError");
}

#[test]
fn the_streaming_half_is_documented_as_its_own_content_type() {
    // `stream: true` answers SSE chunks, not the single JSON body: a
    // client generator that sees only one of the two writes the wrong
    // reader.
    let document = document();
    let ok = &document["paths"]["/v1/chat/completions"]["post"]["responses"]["200"]["content"];
    assert!(ok["application/json"].is_object(), "{ok}");
    assert!(ok["text/event-stream"].is_object(), "{ok}");
}

#[test]
fn the_decision_endpoint_carries_its_question_kinds() {
    // `/v1/decide` is the one endpoint no external documentation covers,
    // so a path entry alone would not be worth much.
    let document = document();
    let text = document["components"]["schemas"]["QuestionKind"].to_string();
    for kind in ["noul", "choice", "score", "scalar", "number", "point", "box"] {
        assert!(text.contains(kind), "QuestionKind must name {kind}: {text}");
    }
    assert!(document["components"]["schemas"]["DecideResponse"].is_object());
}

// ── the routes ───────────────────────────────────────────────────────────

#[tokio::test]
async fn v1_redirects_to_the_page() {
    let app = server().app();
    for uri in ["/v1", "/v1/", "/v1/docs"] {
        let response = get(&app, uri).await;
        assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT, "{uri}");
        assert_eq!(header_value(&response, header::LOCATION), "/v1/docs/", "{uri}");
    }
}

#[tokio::test]
async fn the_page_and_its_assets_come_from_this_binary() {
    // The vendored distribution, not a CDN: a server with no route to the
    // internet serves the same page.
    let app = server().app();
    let page = get(&app, "/v1/docs/").await;
    assert_eq!(page.status(), StatusCode::OK);
    assert_eq!(
        header_value(&page, header::CONTENT_TYPE),
        "text/html; charset=utf-8",
    );
    let html = String::from_utf8(body_bytes(page).await).unwrap();
    assert!(html.contains("/v1/docs/swagger-ui-bundle.js"), "{html}");
    assert!(html.contains("/v1/openapi.json"), "{html}");
    assert!(!html.contains("//cdn."), "the page must name no CDN: {html}");

    for (file, content_type) in [
        ("swagger-ui-bundle.js", "text/javascript; charset=utf-8"),
        ("swagger-ui.css", "text/css; charset=utf-8"),
    ] {
        let response = get(&app, &format!("/v1/docs/{file}")).await;
        assert_eq!(response.status(), StatusCode::OK, "{file}");
        assert_eq!(header_value(&response, header::CONTENT_TYPE), content_type, "{file}");
        assert!(!body_bytes(response).await.is_empty(), "{file}");
    }

    // The bundle's own third-party notices, served beside it as its
    // licence asks.
    assert_eq!(
        get(&app, "/v1/docs/swagger-ui-bundle.js.LICENSE.txt").await.status(),
        StatusCode::OK,
    );
    // Exact table lookup: nothing else is reachable under the page.
    assert_eq!(get(&app, "/v1/docs/swagger-ui.js").await.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn the_document_is_served_as_openapi_3_1() {
    let app = server().app();
    let response = get(&app, "/v1/openapi.json").await;
    assert_eq!(response.status(), StatusCode::OK);
    assert!(header_value(&response, header::CONTENT_TYPE).starts_with("application/json"));
    let document: serde_json::Value = serde_json::from_slice(&body_bytes(response).await).unwrap();
    assert!(
        document["openapi"].as_str().unwrap_or_default().starts_with("3.1"),
        "{}",
        document["openapi"],
    );
    assert_eq!(document["info"]["version"], env!("CARGO_PKG_VERSION"));
    assert!(document["paths"]["/v1/chat/completions"]["post"].is_object());
    assert!(
        document["components"]["securitySchemes"]["bearerAuth"]["scheme"] == "bearer",
        "{}",
        document["components"]["securitySchemes"],
    );
}

#[tokio::test]
async fn the_reference_is_open_on_a_keyed_server() {
    // A browser pointed at `/v1` has no way to set an `Authorization`
    // header, so a keyed server that answered 401 here would publish a
    // reference nobody could read. The handler routes stay gated.
    let app = server().with_api_key(ApiKey::new(KEY.to_owned())).app();
    for uri in ["/v1", "/v1/docs/", "/v1/docs/swagger-ui.css", "/v1/openapi.json"] {
        let status = get(&app, uri).await.status();
        assert!(
            status == StatusCode::OK || status == StatusCode::TEMPORARY_REDIRECT,
            "{uri} answered {status}",
        );
    }
    assert_eq!(get(&app, "/v1/models").await.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn the_preflight_still_answers_on_every_v1_route() {
    // The `OPTIONS` handlers are attached beside the documented ones
    // rather than through `routes!`, so this is what says they survived
    // the move to `OpenApiRouter`.
    let app = server().app();
    for uri in [
        "/v1/models",
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/decide",
        "/v1/systemone",
    ] {
        let request = Request::builder()
            .method(Method::OPTIONS)
            .uri(uri)
            .body(Body::empty())
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{uri}");
    }
}
