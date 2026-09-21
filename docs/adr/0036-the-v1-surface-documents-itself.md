# ADR 0036 — the `/v1` surface documents itself, and its reference needs no key

## Status

Accepted (2026-09-21, owner decision — GitHub #250 / #251).

## Context

An ignis server served an OpenAI-compatible surface and one endpoint that is
its own (`POST /v1/decide`, ADR 0034) and said nothing about either. The
contract lived in `docs/design/ignis-v1.md` §2 and in doc comments: in the
repo, not on the wire, and free to drift from the handlers it described. A
client author had to read Rust to learn a field name; `/v1/decide`, which no
external documentation covers, was the least guessable shape on the server.

The drift is the part worth designing against. A reference that is written
beside the handlers goes stale the first time a route is added without one,
and nothing fails when it does.

## Decision

**The document is generated where the routes are registered.** The `/v1`
routes are built through `utoipa_axum::router::OpenApiRouter`, whose
`routes!()` registers a handler *and* its `#[utoipa::path]` entry in the same
call. A route cannot exist without its documentation entry, nor an entry
without its route. `crates/server/tests/openapi_http.rs` asserts the
resulting path set as an equality, so a route added with a bare `.route()`
fails the suite rather than quietly going undocumented.

**The reference is part of the `/v1` surface, not of the Playground.** It is
always served: `GET /v1/openapi.json` for the document, `GET /v1/docs/` for
the Swagger UI page, and a 307 from `GET /v1` and `GET /v1/` so the address a
human types lands on it. `--no-ui` withholds the Playground and not this.

**The reference is open even on a keyed server.** `--api-key` gates the
handler routes; the document and the page answer without one. A browser
pointed at `/v1` cannot set an `Authorization` header, so a gated reference
would be a reference nobody could read. What it publishes is the API's
*shape* — no prompt, no completion, no load figure — and the document
declares the `bearerAuth` scheme, so the page's *Authorize* button drives the
gated routes from the page. This is a deliberate carve-out from ADR 0028's
"an exposed server must not publish its load": shape is not load.

**Monitoring is not API.** The Prometheus exposition (ADR 0017: `/metrics`
on its own listener, `/ui/metrics` behind the key) is outside the document,
and a test fails if a path carrying `metrics`, or any `/ui` path, ever
appears in it.

**The Swagger UI distribution is vendored in the repo, not taken from
`utoipa-swagger-ui`.** That crate embeds the same files through
`rust-embed`, whose proc macro pulls `walkdir → winapi-util → windows-sys
0.59`; `windows-sys` 0.59 builds its raw-dylib imports with `dlltool`, and
the `x86_64-pc-windows-gnu` **host** toolchain this project is developed on
cannot run one (`rust-mingw` ships `dlltool.exe` without the binutils it
calls). Being a host dependency, the MSVC target pin in `.cargo/config.toml`
does not route around it: `cargo check -p ignis-server` fails outright. Two
files under `crates/server/assets/swagger-ui` (see its `IGNIS-VENDOR.md`)
cost the binary no more than the crate's own embed and build with any
toolchain, offline, on every platform. `utoipa` and `utoipa-axum` themselves
are pure Rust and are taken as crates.

**The narrative doc stays.** `docs/design/ignis-v1.md` §2 keeps describing
the surface in prose; the generated document is the machine-readable
contract. Neither is regenerated from the other.

## Consequences

- A new `/v1` endpoint ships documented or fails the suite. The cost is one
  `#[utoipa::path]` block per handler, which is where the descriptions now
  live.
- The alias `POST /v1/systemone` keeps its route and stays out of the
  document: OpenAPI has no notion of an alias, and listing it would repeat
  every schema reference under a second path. It is named in the document's
  own description instead.
- Two properties of `/v1/decide` cannot be expressed in JSON Schema and are
  stated in prose: that a JSON evidence reaches the model in the order its
  keys were written, and that a `choice`'s options are read in declared
  order. The schemas for those fields are free-form JSON.
- The binary grows by the vendored distribution (~1.8 MB). Updating Swagger
  UI is a manual `npm pack` copy, recorded in `IGNIS-VENDOR.md`.
- An operator on an air-gapped machine gets the same page as everyone else:
  nothing on it is fetched from a CDN.
