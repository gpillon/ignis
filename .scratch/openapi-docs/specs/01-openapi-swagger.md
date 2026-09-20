# 01 — The `/v1` surface documents itself: OpenAPI + Swagger UI

GitHub: (master issue, filed by `/to-tickets`)

## Problem Statement

ignis serves an OpenAI-compatible API plus one endpoint that is its own
(`POST /v1/decide`, ADR 0034), and nothing on the running server describes
it. A caller who points a client at `http://host:8080` can reach `/ui/` and
type into the Playground, but there is no machine-readable contract and no
page that answers "what can this server do, with what body, and what comes
back". The narrative description lives in `docs/design/ignis-v1.md` §2 and
in doc comments — both in the repo, neither served, and both free to drift
from the handlers they describe. An agent author writing a client, an
operator behind an `--expose` tunnel (ADR 0028) and the owner checking what
a field is called all have to read Rust source to find out.

`/v1/decide` makes the gap concrete: it is not an OpenAI endpoint, so no
external documentation covers it, and its request body (a question, its
options, an ordered JSON evidence) is the least guessable shape on the
server.

## Solution

The server publishes its own API reference, generated from the handlers it
actually serves:

- `GET /v1/openapi.json` — an OpenAPI 3.1 document describing the `/v1`
  inference surface, generated at compile time from the handler
  annotations, so a path in the document exists because a route exists.
- `GET /v1/docs/` — Swagger UI over that document, served from assets
  vendored into the binary: no network at build time, none at runtime.
- `GET /v1` and `GET /v1/` — a 307 redirect to `/v1/docs/`, the same shape
  `/ui` already has towards `/ui/`. Navigating to `/v1` lands on the
  reference.
- Both the document and the page are reachable without an API key, even on
  a server started with `--api-key`. They publish the API's *shape*, never
  its load. The document declares a `bearer` security scheme, so Swagger's
  *Authorize* button makes try-it-out work against the key-gated routes.
- The document covers the inference surface only: `/v1/models`,
  `/v1/chat/completions`, `/v1/responses`, `/v1/decide`. The Prometheus
  exposition (`/metrics` on its own listener, `/ui/metrics` behind the key)
  and the Playground's static assets are not API endpoints and stay out.

## User Stories

1. As an agent author writing a client against ignis, I want to open
   `http://host:8080/v1` in a browser and see every endpoint the server
   serves, so that I do not have to read Rust source to learn the wire
   shape.
2. As an agent author, I want `GET /v1/openapi.json` to return a valid
   OpenAPI 3.1 document, so that I can generate a typed client from it.
3. As an agent author, I want the streaming variant of
   `POST /v1/chat/completions` documented as a `text/event-stream`
   response with its chunk schema, so that I know what `stream: true`
   sends back.
4. As an agent author, I want `POST /v1/decide` documented with its
   question kinds, its options and its ordered evidence, so that I can
   call the one endpoint no OpenAI documentation covers.
5. As an agent author, I want the error body (`{"error": {message, type,
   code}}`) documented once and referenced by every endpoint, so that I
   handle failures uniformly.
6. As an agent author, I want each endpoint's documented status codes to
   include the ones ignis actually returns — 400, 401, 404, 413, 503, 504 —
   so that my client's error handling is written against the real set.
7. As an operator running a keyed server, I want `/v1` to answer the page
   and not a bare 401, so that the reference is reachable from a browser
   that has no way to set an `Authorization` header.
8. As an operator, I want Swagger's *Authorize* button to accept the
   server's key, so that I can exercise a live endpoint from the page.
9. As an operator on an air-gapped machine, I want the page to render with
   no outbound request from the browser, so that the reference works where
   the CDN does not.
10. As an operator building the release container (ADR 0032), I want the
    build to make no network request for documentation assets, so that the
    build stays reproducible and offline.
11. As the owner, I want a route that exists and a path in the document to
    be the same fact, so that the reference cannot silently drift from the
    server.
12. As the owner, I want a test that fails when a new `/v1` route ships
    undocumented, so that drift is caught by `make ci` and not by a reader.
13. As the owner, I want a test that fails when a monitoring path leaks
    into the document, so that "not for the monitoring" stays true as the
    code moves.
14. As the owner, I want the documentation surface to cost nothing on the
    inference path, so that annotating the handlers does not change how a
    request is served.
15. As the owner, I want `--no-ui` to keep serving the API reference, so
    that a server without the Playground is still self-describing.
16. As a Playground user, I want `/ui/` to keep working exactly as it does
    today, so that the new routes take nothing away.
17. As a caller of `/v1/systemone`, I want the alias to keep working
    unchanged, so that an unmodified Jev client still reaches the server
    even though the document lists only `/v1/decide`.
18. As a reader of the document, I want the sampling, thinking and tool
    fields that `ChatCompletionsRequest` flattens to appear as first-class
    properties, so that the document shows the body the handler really
    accepts.
19. As a reader of the document, I want each endpoint to carry a one-line
    summary and a description drawn from the module's own doc comments, so
    that the page reads like the design doc and not like a type dump.
20. As a client generator, I want the schema names in the document to be
    stable across builds, so that a regenerated client is a no-op diff when
    nothing changed.

## Implementation Decisions

**Generator.** `utoipa` 5 for the derives, `utoipa-axum` 0.2 for the
router bindings, `utoipa-swagger-ui` 9 for the page. All three carry
`axum` 0.8, the version already in the workspace. `utoipa-swagger-ui` is
taken with `features = ["axum", "vendored"]`: the `vendored` feature pulls
the Swagger UI distribution from `utoipa-swagger-ui-vendored` on crates.io
instead of downloading a zip in its build script, which is what keeps the
container build (ADR 0032) offline and reproducible. Cost: roughly 2–3 MB
of binary, paid once.

**The seam.** One seam, in `crates/server/src/api.rs`: the `/v1` routes
are built through `utoipa_axum::router::OpenApiRouter` with the `routes!()`
macro, which registers the handler and its OpenAPI path entry in the same
call. A route cannot exist without its documentation entry, and an entry
cannot exist without its route — the drift class the feature exists to
prevent is removed by construction rather than by a checklist. `router()`
keeps its present signature and returns the same `Router`; the OpenAPI
document is split out of the `OpenApiRouter` at the end of the function and
handed to the docs routes. The document itself is reachable for tests
through a small `openapi()` function so the assertions do not have to go
through HTTP.

**Where the docs routes live.** A new module `crates/server/src/openapi.rs`
holds the `#[derive(OpenApi)]` root (title, version from
`CARGO_PKG_VERSION`, description, the `bearer` security scheme, the shared
`ApiError` component) and the router that serves `/v1/openapi.json`,
`/v1/docs/*` and the `/v1` → `/v1/docs/` redirect. It is merged into the
server router *outside* `require_api_key`, next to the Playground merge, so
the key layer's route list is unchanged.

**Key policy.** `require_api_key` stays a `route_layer` over exactly
today's five `/v1` handler routes. The docs routes are keyless by decision
(see the ADR note below): they publish the shape of the API, not its load,
and `/ui/` is already keyless and already knows every endpoint. The
document declares `bearerAuth` as a security scheme applied to the
documented operations, so a reader sees that a key may be required and
Swagger's *Authorize* fills it in.

**Alias.** `/v1/systemone` keeps its route and stays out of the document.
OpenAPI has no notion of an alias, and listing the path twice would
duplicate every schema reference; the `/v1/decide` description names the
alias in prose instead.

**Streaming.** `POST /v1/chat/completions` is documented with two 200
responses: `application/json` → `ChatCompletion` and `text/event-stream`
→ the SSE chunk schema, with the `[DONE]` sentinel described in prose.
The request's `stream` field is what selects between them.

**Schemas.** The request and response types in `api.rs` gain `ToSchema`
beside their existing `Serialize`/`Deserialize` derives and stay private to
the crate; `utoipa` follows serde attributes, so `#[serde(flatten)]` on the
sampling / thinking / tool-definition field groups and `#[serde(untagged)]`
on `ResponsesInput` carry over without restating them. Three types in
`decide.rs` have hand-written `Deserialize` impls — `Ordered<T>`,
`Criteria`, `Evidence` — and therefore get an explicit schema: a free-form
JSON object with a description, not a derived one. The ordered JSON
evidence is a genuine limit of the format: JSON Schema cannot express "the
keys reach the model in the order they were written", which is a real
property of the endpoint (commit `850cec4`), so it is stated in the
description.

**No new flag.** The documentation is always served, like the `/v1` routes
themselves. No `--docs` / `--no-docs` knob until someone asks for one;
`--no-ui` withholds the Playground only.

**Nothing on the inference path.** The annotations are attributes and a
compile-time document; no handler body changes, and no per-request work is
added.

**ADR.** One short ADR records the two decisions that outlive this
change: the generated document is the machine-readable contract of the
`/v1` surface (with `docs/design/ignis-v1.md` §2 staying the narrative),
and the reference is served without a key even on a keyed server — a
carve-out from ADR 0028's "an exposed server must not publish its load",
justified by shape-not-load.

## Testing Decisions

A good test here asserts external behaviour: what the document contains
and what the routes answer over HTTP. It must not assert the internal
shape of a `utoipa` type, and must not pin whole-document JSON — a
snapshot of the full document would fail on every unrelated field rename
and teach nobody anything.

Prior art: `crates/server/tests/openai_http.rs` and
`crates/server/tests/playground_http.rs` drive the router with `tower`'s
`oneshot` against `MockCompute`, and `crates/server/tests/api_key_http.rs`
is the existing shape for a keyed-server assertion.
`crates/server/tests/metrics_structure.rs` is the prior art for asserting
over a generated document's *structure* rather than its bytes.

New tests, all CPU-only, no GPU:

1. **Path set** — the document's paths are exactly `/v1/models`,
   `/v1/chat/completions`, `/v1/responses`, `/v1/decide`, with the methods
   each route serves. Written as set equality so a new undocumented route
   fails it (story 12).
2. **No monitoring** — no path in the document begins with `/metrics`,
   `/ui` or contains `metrics` (story 13). Separate from test 1 so the
   failure names the reason.
3. **Shared components** — the error schema is a component referenced by
   the documented operations rather than inlined per operation, and the
   `bearerAuth` security scheme is present.
4. **Routes answer** — `GET /v1` → 307 with `location: /v1/docs/`;
   `GET /v1/` → 307; `GET /v1/docs/` → 200 `text/html`;
   `GET /v1/openapi.json` → 200 `application/json` whose body parses and
   carries `openapi: "3.1.x"`.
5. **Keyless on a keyed server** — with `--api-key` set, `GET /v1`,
   `GET /v1/docs/` and `GET /v1/openapi.json` answer without a header,
   while `POST /v1/chat/completions` still answers 401 without one
   (stories 7, 8). Extends the existing `api_key_http.rs`.
6. **Asset is embedded** — the Swagger UI bundle referenced by
   `/v1/docs/` is served by this binary (a 200 on the page's own script
   asset), which is what proves the vendored assets shipped and the page
   does not reach a CDN (stories 9, 10).
7. **Decide schema present** — the `/v1/decide` operation's request body
   resolves to a schema that names the question kinds, so the one endpoint
   without external documentation is covered by more than a path entry
   (story 4).

The whole-document assertions run against `openapi()` directly; the route
assertions go through the router. `make ci` is the gate, not `cargo test`
alone.

## Out of Scope

- The Prometheus surface: `/metrics` on the metrics listener and
  `/ui/metrics` behind the key stay undocumented here. ADR 0017 is their
  contract.
- The Playground's own static routes (`/ui/*`) — assets, not an API.
- `/v1/systemone` as a listed path (kept as a route, described in prose).
- Any change to request or response *behaviour*. This change adds
  attributes, a module and three routes; a handler that answers differently
  afterwards is a bug in this work.
- A `--docs` / `--no-docs` flag.
- Restyling the Swagger page in the Playground's kiln/ember tokens. The
  stock theme ships; a themed page can follow if the owner wants one.
- Regenerating `docs/design/ignis-v1.md` from the document, or deleting it.
- Publishing the document as a build artifact or to the release (ADR 0032).
- A client generated from the document, in any language.

## Further Notes

- Swagger UI is mounted at `/v1/docs` rather than at `/v1` itself on
  purpose: `SwaggerUi::new("/v1")` would register a catch-all
  `/v1/{*tail}` for its assets alongside the real `/v1/...` routes. The
  redirect gives the same "navigate to `/v1`" behaviour with no catch-all
  next to the inference routes.
- `utoipa-swagger-ui` 9 has a `cache` feature that writes into the user's
  dirs at build time. It is not taken.
- The version in the document comes from `CARGO_PKG_VERSION`, so it
  follows the four-file version bump already in the release routine.
- Repo formatting rule stands: the diff is attributes and new files, and
  `cargo fmt` is never run over this tree.
