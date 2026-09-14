//! GitHub #163 / ADR 0026: the Playground is served under `/ui/` only when
//! the server was built with it (`--ui`), from whichever asset table the
//! binary carries — the real `web/dist` build, or the fallback page when the
//! frontend was never built. Both tables are injected here, so the fallback
//! is covered whatever state this checkout's `web/dist` is in.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use ignis_core::{ConcreteScheduler, MockCompute, SchedulerConfig};
use ignis_server::Server;
use ignis_server::engine::Engine;
use ignis_server::playground::Assets;
use ignis_server::template::SimpleTemplateProvider;
use tower::ServiceExt;

const MODEL: &str = "test-model";

const BUILT: Assets = &[
    ("index.html", b"<!doctype html><title>Playground</title>"),
    ("assets/index-abc123.js", b"console.log('playground')"),
    ("assets/index-abc123.css", b"body{}"),
    ("favicon.svg", b"<svg/>"),
];

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

async fn body_text(response: axum::response::Response) -> String {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).unwrap()
}

fn header_value(response: &axum::response::Response, name: header::HeaderName) -> String {
    response
        .headers()
        .get(name)
        .map(|v| v.to_str().unwrap().to_owned())
        .unwrap_or_default()
}

#[tokio::test]
async fn without_ui_the_playground_route_is_absent() {
    let app = server().app();
    for uri in ["/ui", "/ui/", "/ui/index.html", "/ui/assets/index-abc123.js"] {
        assert_eq!(get(&app, uri).await.status(), StatusCode::NOT_FOUND, "{uri}");
    }
}

#[tokio::test]
async fn with_ui_the_built_page_is_served_under_ui() {
    let app = server().with_playground(BUILT).app();

    let page = get(&app, "/ui/").await;
    assert_eq!(page.status(), StatusCode::OK);
    assert!(header_value(&page, header::CONTENT_TYPE).starts_with("text/html"));
    assert_eq!(header_value(&page, header::CACHE_CONTROL), "no-cache");
    assert_eq!(body_text(page).await, "<!doctype html><title>Playground</title>");

    let redirect = get(&app, "/ui").await;
    assert!(redirect.status().is_redirection(), "{}", redirect.status());
    assert_eq!(header_value(&redirect, header::LOCATION), "/ui/");
}

#[tokio::test]
async fn built_assets_carry_their_content_type_and_hashed_assets_are_immutable() {
    let app = server().with_playground(BUILT).app();

    let script = get(&app, "/ui/assets/index-abc123.js").await;
    assert_eq!(script.status(), StatusCode::OK);
    assert!(header_value(&script, header::CONTENT_TYPE).starts_with("text/javascript"));
    assert!(header_value(&script, header::CACHE_CONTROL).contains("immutable"));
    assert_eq!(body_text(script).await, "console.log('playground')");

    let style = get(&app, "/ui/assets/index-abc123.css").await;
    assert!(header_value(&style, header::CONTENT_TYPE).starts_with("text/css"));

    // Outside `assets/` the name is not content-hashed, so it must not be
    // cached forever.
    let icon = get(&app, "/ui/favicon.svg").await;
    assert_eq!(header_value(&icon, header::CONTENT_TYPE), "image/svg+xml");
    assert_eq!(header_value(&icon, header::CACHE_CONTROL), "no-cache");
}

#[tokio::test]
async fn an_unknown_playground_path_is_a_404_not_the_page() {
    let app = server().with_playground(BUILT).app();
    for uri in ["/ui/assets/missing.js", "/ui/nope", "/ui/../v1/models"] {
        assert_eq!(get(&app, uri).await.status(), StatusCode::NOT_FOUND, "{uri}");
    }
}

#[tokio::test]
async fn the_playground_leaves_the_rest_of_the_surface_alone() {
    let app = server().with_playground(BUILT).app();
    assert_eq!(get(&app, "/v1/models").await.status(), StatusCode::OK);
    assert_eq!(get(&app, "/").await.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn a_binary_built_without_the_frontend_serves_the_fallback_page() {
    let app = server().with_playground(&[]).app();

    let page = get(&app, "/ui/").await;
    assert_eq!(page.status(), StatusCode::OK);
    assert!(header_value(&page, header::CONTENT_TYPE).starts_with("text/html"));
    let text = body_text(page).await;
    assert!(text.contains("npm --prefix web run build"), "the fallback must say how to build:\n{text}");

    assert_eq!(get(&app, "/ui/assets/index-abc123.js").await.status(), StatusCode::NOT_FOUND);
}
