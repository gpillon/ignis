//! The served API reference (GitHub #251): the OpenAPI 3.1 document of the
//! `/v1` inference surface, and the Swagger UI page over it.
//!
//! Routes:
//! - `GET /v1/openapi.json` — the document.
//! - `GET /v1/docs/` — Swagger UI over it, driven by the Swagger UI
//!   distribution vendored under `crates/server/assets/swagger-ui` and
//!   compiled into this binary: the page makes no outbound request, and
//!   the build makes none either (`IGNIS-VENDOR.md` says why the
//!   `utoipa-swagger-ui` crate is not used).
//! - `GET /v1` and `GET /v1/` — a 307 to `/v1/docs/`, the shape `/ui`
//!   already has towards `/ui/`, so navigating to `/v1` lands on the
//!   reference.
//!
//! All three are served **outside** `api::require_api_key`: they publish
//! the API's shape, never its load, and a browser pointed at `/v1` on a
//! keyed server has no way to set an `Authorization` header. What they
//! publish instead is that a key exists — the document declares the
//! `bearerAuth` scheme on every documented operation, so Swagger's
//! *Authorize* button drives the key-gated routes from the page.
//!
//! The document itself is not built here: [`crate::api::router`] builds the
//! `/v1` routes through `utoipa_axum::router::OpenApiRouter`, which
//! registers a handler and its path entry in the same call, and hands the
//! finished document to [`router`]. A path exists in the document because a
//! route exists — the two cannot drift.
//!
//! The Prometheus exposition (`/metrics`, `/ui/metrics` — ADR 0017) and the
//! Playground's assets are not part of this document: they are not the
//! inference API, and `crates/server/tests/openapi_http.rs` fails if one
//! ever appears in it.

use std::sync::Arc;

use axum::Router;
use axum::extract::Path;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::get;
use axum::Json;
use utoipa::openapi::security::{HttpAuthScheme, HttpBuilder, SecurityScheme};
use utoipa::{Modify, OpenApi};

/// Where the document is served, and where the page lives.

pub const DOCUMENT_PATH: &str = "/v1/openapi.json";
/// The page's mount. Deliberately *not* `/v1` itself: serving the page
/// there would put its assets on a catch-all right beside the real
/// `/v1/...` handler routes. The redirect below gives the same "navigate
/// to `/v1`" behaviour with no catch-all next to them.
pub const DOCS_PATH: &str = "/v1/docs";

/// The document's root: everything about the surface that is not one
/// operation. The paths and the schemas are added by `OpenApiRouter` as the
/// routes are registered (`crate::api::router`).
#[derive(OpenApi)]
#[openapi(
    info(
        title = "ignis",
        description = "\
The inference surface of an `ignis` server: OpenAI-compatible chat \
completions and responses, plus `/v1/decide` (ADR 0034), which reads an \
answer out of the model's own readout instead of generating one.\n\n\
Every operation below is served by the listener this page was loaded \
from. When the server runs with `--api-key`, they answer `401` without a \
bearer token — this page and the document do not, so *Authorize* is how \
you drive them from here.\n\n\
`POST /v1/systemone` is an alias of `POST /v1/decide`: the same handler \
under Jev's name, so an unmodified Jev client reaches this server by \
changing the URL alone. OpenAPI has no notion of an alias, so it is named \
here rather than listed as a second path.\n\n\
The Prometheus exposition is not part of this document (ADR 0017): it is \
served on its own listener with `--metrics`, and it is monitoring rather \
than API.",
        version = env!("CARGO_PKG_VERSION"),
        license(name = "Apache-2.0"),
    ),
    modifiers(&BearerKey),
    // The key gates every documented operation or none of them
    // (`api::require_api_key` is one layer over the lot), so the
    // requirement is stated once, on the document.
    security(("bearerAuth" = [])),
    tags(
        (name = "models", description = "What this server loaded."),
        (name = "chat", description = "Chat completions, streaming and not."),
        (name = "responses", description = "The OpenAI responses API."),
        (name = "decide", description = "Typed decisions read from the readout (ADR 0034)."),
    ),
)]
pub struct ApiDoc;

/// Declares `--api-key` as what it is on the wire: an HTTP bearer token.
///
/// Applied as a *security requirement on the whole document*, not per
/// operation: the key gates every `/v1` handler route or none of them
/// (`api::require_api_key` is one layer over the lot), so stating it once
/// is stating the truth once.
struct BearerKey;

impl Modify for BearerKey {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        let components = openapi.components.get_or_insert_with(Default::default);
        components.add_security_scheme(
            "bearerAuth",
            SecurityScheme::Http(
                HttpBuilder::new()
                    .scheme(HttpAuthScheme::Bearer)
                    .description(Some(
                        "The server's `--api-key`. Absent from a server started without one, \
                         in which case every operation answers unauthenticated.",
                    ))
                    .build(),
            ),
        );
    }
}

/// The reference's routes over `document`, to be merged into the server's
/// router outside the key layer.
///
/// Four of them: the document, the page, the page's two assets, and the
/// redirects a human's `/v1` lands on. The assets are looked up by exact
/// name in [`ASSETS`], so no request path can reach outside the table.
pub fn router<S>(document: utoipa::openapi::OpenApi) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    let document = Arc::new(document);
    Router::new()
        // `/v1` and `/v1/` are the addresses a human types. Temporary, not
        // permanent: a permanent redirect is cached by the browser, and
        // `/v1` is a path this server may well serve differently later.
        .route("/v1", get(|| async { Redirect::temporary("/v1/docs/") }))
        .route("/v1/", get(|| async { Redirect::temporary("/v1/docs/") }))
        .route(DOCS_PATH, get(|| async { Redirect::temporary("/v1/docs/") }))
        .route("/v1/docs/", get(|| async { page() }))
        .route(
            "/v1/docs/{file}",
            get(|Path(file): Path<String>| async move { asset(&file) }),
        )
        .route(
            DOCUMENT_PATH,
            get(move || {
                let document = Arc::clone(&document);
                async move { Json((*document).clone()) }
            }),
        )
}

/// The page itself: Swagger UI over [`DOCUMENT_PATH`], driven by the
/// vendored bundle beside it. No CDN, no outbound request — a server on a
/// machine with no route to the internet serves the same page.
fn page() -> Response {
    (
        [
            (header::CONTENT_TYPE, "text/html; charset=utf-8"),
            // The page names the two assets by fixed path, and a rebuilt
            // server may carry a different Swagger UI under them, so it is
            // never cached.
            (header::CACHE_CONTROL, "no-cache"),
        ],
        PAGE_HTML,
    )
        .into_response()
}

/// One vendored asset by exact name, or a 404.
fn asset(file: &str) -> Response {
    match ASSETS.iter().find(|(name, _)| *name == file) {
        Some((name, bytes)) => (
            [
                (
                    header::CONTENT_TYPE,
                    if name.ends_with(".css") {
                        "text/css; charset=utf-8"
                    } else if name.ends_with(".js") {
                        "text/javascript; charset=utf-8"
                    } else {
                        "text/plain; charset=utf-8"
                    },
                ),
                (header::CACHE_CONTROL, "no-cache"),
            ],
            *bytes,
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

/// The vendored Swagger UI (`crates/server/assets/swagger-ui`, see its
/// `IGNIS-VENDOR.md`): the bundle, its stylesheet, and the third-party
/// notices the bundle's own licence asks to be served with it.
const ASSETS: &[(&str, &[u8])] = &[
    (
        "swagger-ui-bundle.js",
        include_bytes!("../assets/swagger-ui/swagger-ui-bundle.js"),
    ),
    (
        "swagger-ui.css",
        include_bytes!("../assets/swagger-ui/swagger-ui.css"),
    ),
    (
        "swagger-ui-bundle.js.LICENSE.txt",
        include_bytes!("../assets/swagger-ui/swagger-ui-bundle.js.LICENSE.txt"),
    ),
    ("LICENSE", include_bytes!("../assets/swagger-ui/LICENSE")),
    ("NOTICE", include_bytes!("../assets/swagger-ui/NOTICE")),
];

/// The page's HTML. `persistAuthorization` is on so a key typed into
/// *Authorize* survives a reload of the page — an operator exercising a
/// keyed server types it once.
const PAGE_HTML: &str = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ignis API</title>
<link rel="stylesheet" href="/v1/docs/swagger-ui.css">
</head>
<body>
<div id="swagger-ui"></div>
<script src="/v1/docs/swagger-ui-bundle.js"></script>
<script>
  window.ui = SwaggerUIBundle({
    url: "/v1/openapi.json",
    dom_id: "#swagger-ui",
    deepLinking: true,
    persistAuthorization: true,
    tryItOutEnabled: true,
    defaultModelsExpandDepth: 0,
  });
</script>
</body>
</html>
"##;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_document_declares_the_bearer_scheme() {
        // The `--api-key` contract, stated where a client generator and the
        // page's own Authorize button can both read it.
        let document = crate::api::openapi();
        let components = document.components.expect("components");
        assert!(
            components.security_schemes.contains_key("bearerAuth"),
            "the document must declare the bearer scheme: {:?}",
            components.security_schemes.keys().collect::<Vec<_>>(),
        );
    }
}
