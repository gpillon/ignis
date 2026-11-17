//! The `ignis` skill (spec #656): the folder `connector/ignis` of the repo,
//! packed by `build.rs` (`skill_pack.rs`) into one deterministic `tar.gz`
//! and embedded here, as the Playground is. `GET /v1/connector/skill` serves
//! it, so the skill a developer installs always matches the server that gave
//! it. Its scripts run in the harness's shell and call this server's routes:
//! ignis touches no file.

use axum::extract::Request;
use axum::http::header;
use axum::response::{IntoResponse, Response};

/// The archive: every entry under `ignis/`, sorted, with fixed mtime and
/// modes, so extracting it into a skills folder makes the skill directory.
pub const ARCHIVE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ignis-skill.tar.gz"));

/// Where the archive is served.
pub const PATH: &str = "/v1/connector/skill";

/// `GET /v1/connector/skill` — the `ignis` skill as a `tar.gz`.
#[utoipa::path(
    get,
    path = "/v1/connector/skill",
    tag = "connector",
    operation_id = "connector_skill",
    summary = "The ignis skill, as a tar.gz",
    description = "The connector's skill, `ignis`: a `SKILL.md` and Node scripts that run in a coding agent's shell, beside its files, and call this server's own routes (today `transcribe.mjs`, which transcribes a recording through `POST /v1/audio/transcriptions`). One deterministic `tar.gz` packed when the server was built, every entry under `ignis/` so extracting it into a skills folder makes the skill's directory; the same bytes on every request, and always the skill that matches this server. Install it with one command, `curl -fsSL -H \"Authorization: Bearer $IGNIS_API_KEY\" <base-url>/v1/connector/skill | tar -xz -C ~/.agents/skills`; the guide gives the PowerShell form. It sits behind the API key, and answers `404` with `connector.enabled` off.",
    responses(
        (status = 200, description = "The skill's `tar.gz`.", content_type = "application/gzip", body = Vec<u8>),
        (status = 401, description = "The server was started with an API key and the request carried no matching bearer token.", body = crate::api::ApiError),
        (status = 404, description = "The connector is off (`connector_disabled`): `connector.enabled` is false.", body = crate::api::ApiError),
    ),
)]
pub(crate) async fn skill(request: Request) -> Response {
    // Present exactly when `connector.enabled` is on (`api::router`).
    if request.extensions().get::<super::Connector>().is_none() {
        return super::disabled();
    }
    ([(header::CONTENT_TYPE, "application/gzip")], ARCHIVE).into_response()
}
