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

/// The media type an asset is served as.
///
/// Every extension `web/dist` can hold has a branch, and an image's branch is
/// not a nicety: the page reads its own bundled pictures back with `fetch` and
/// hands them to the browser's decoder, which goes by the type the server
/// declared -- so a picture served as `application/octet-stream` is refused as
/// "not an image" on the embedded build while working under Vite, which sends
/// the right one (GitHub #256). The image types are therefore every one Vite
/// treats as an asset, not only the one the brand mark happens to use today.
fn content_type(path: &str) -> &'static str {
    match path.rsplit_once('.').map(|(_, ext)| ext) {
        Some("html") => "text/html; charset=utf-8",
        Some("js" | "mjs") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json" | "map") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("webp") => "image/webp",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("avif") => "image/avif",
        Some("ico") => "image/x-icon",
        Some("woff2") => "font/woff2",
        Some("woff") => "font/woff",
        Some("ttf") => "font/ttf",
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

    /// A picture is served as a picture (GitHub #256).
    ///
    /// The page fetches its own bundled images back and hands them to the
    /// browser's decoder, which reads the type this table wrote: the brand
    /// example refused its own mark as "not an image" for as long as `webp`
    /// fell through to `application/octet-stream`.
    #[test]
    fn a_picture_is_served_as_a_picture() {
        for name in ["assets/flame-a1b2c3.webp", "shot.jpg", "shot.jpeg", "anim.gif", "photo.avif", "mark.png"] {
            let served = content_type(name);
            assert!(served.starts_with("image/"), "{name} was served as {served}");
        }
    }

    /// Every extension the embedded build actually holds is named.
    ///
    /// The table is a list, so it falls behind whatever `web/` starts
    /// importing next; this fails on the asset kind nobody added a branch
    /// for rather than shipping it as an opaque download. `map` is exempt:
    /// a source map is not served to anyone who cares about its type.
    #[test]
    fn every_embedded_extension_is_named() {
        for (name, _) in EMBEDDED.iter() {
            let Some((_, ext)) = name.rsplit_once('.') else { continue };
            assert_ne!(
                content_type(name),
                "application/octet-stream",
                "web/dist holds a .{ext} and `content_type` has no branch for it"
            );
        }
    }
}
