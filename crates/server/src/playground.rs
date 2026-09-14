//! The Playground (GitHub #163, ADR 0026): the opt-in browser page served
//! under `/ui/` when `ignis-server` runs with `--ui`.
//!
//! The page is a static build of `web/` (Vite + React). `build.rs` embeds
//! `web/dist` into the binary when it exists at compile time; otherwise the
//! table is empty and [`FALLBACK_HTML`] is served in its place, telling the
//! reader how to build the frontend. Cargo never runs npm.
//!
//! Nothing here touches the engine: the page talks to the same `/v1/*`
//! surface every other client uses.

use axum::Router;
use axum::extract::Path;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::get;

/// An asset table: each file's path relative to the build root (`/`
/// separated, e.g. `assets/index-abc123.js`) and its bytes. Empty means the
/// frontend was not built.
pub type Assets = &'static [(&'static str, &'static [u8])];

/// The table `build.rs` generated from `web/dist` for this binary (empty
/// when `web/dist/index.html` did not exist at compile time).
pub const EMBEDDED: Assets = include!(concat!(env!("OUT_DIR"), "/playground_assets.rs"));

/// The page served at `/ui/` by a binary built without the frontend.
pub const FALLBACK_HTML: &str = "<!doctype html>\n\
<html lang=\"en\">\n\
<head><meta charset=\"utf-8\"><title>Playground not built</title></head>\n\
<body style=\"font-family: system-ui, sans-serif; max-width: 40rem; margin: 3rem auto; padding: 0 1rem\">\n\
<h1>Playground not built</h1>\n\
<p>This <code>ignis-server</code> was compiled without the Playground frontend.\n\
Build it, then rebuild the server:</p>\n\
<pre>npm --prefix web ci\n\
npm --prefix web run build\n\
cargo build -p ignis-server</pre>\n\
</body>\n\
</html>\n";

/// The `/ui` routes over `assets`, to be merged into the server's router.
pub fn router<S>(assets: Assets) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        .route("/ui", get(|| async { Redirect::temporary("/ui/") }))
        .route("/ui/", get(move || async move { respond(assets, "index.html") }))
        .route(
            "/ui/{*path}",
            get(move |Path(path): Path<String>| async move { respond(assets, &path) }),
        )
}

/// The file at `path` from `assets` (or the fallback page, when the table is
/// empty and the page itself is asked for), else a 404. Lookup is by exact
/// table key, so no request path can reach outside the table.
fn respond(assets: Assets, path: &str) -> Response {
    let bytes = if assets.is_empty() {
        (path == "index.html").then_some(FALLBACK_HTML.as_bytes())
    } else {
        assets.iter().find(|(name, _)| *name == path).map(|(_, bytes)| *bytes)
    };
    match bytes {
        Some(bytes) => (
            [
                (header::CONTENT_TYPE, content_type(path)),
                (header::CACHE_CONTROL, cache_control(path)),
            ],
            bytes,
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

fn content_type(path: &str) -> &'static str {
    match path.rsplit_once('.').map(|(_, ext)| ext) {
        Some("html") => "text/html; charset=utf-8",
        Some("js" | "mjs") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json" | "map") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("ico") => "image/x-icon",
        Some("woff2") => "font/woff2",
        Some("txt") => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

/// Vite content-hashes everything under `assets/`, so those names never
/// change meaning and can be cached forever; anything else (the page, a
/// favicon) must be revalidated.
fn cache_control(path: &str) -> &'static str {
    if path.starts_with("assets/") {
        "public, max-age=31536000, immutable"
    } else {
        "no-cache"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_table_is_empty_or_a_real_build() {
        // Whatever state this checkout's `web/dist` is in: either nothing
        // was embedded (fallback) or the build's page is there, keyed the
        // way the router looks it up.
        if !EMBEDDED.is_empty() {
            assert!(EMBEDDED.iter().any(|(name, _)| *name == "index.html"));
            assert!(EMBEDDED.iter().all(|(name, _)| !name.contains('\\') && !name.starts_with('/')));
        }
    }

    #[test]
    fn content_types_follow_the_extension() {
        assert_eq!(content_type("index.html"), "text/html; charset=utf-8");
        assert_eq!(content_type("assets/a.js"), "text/javascript; charset=utf-8");
        assert_eq!(content_type("no-extension"), "application/octet-stream");
    }
}
