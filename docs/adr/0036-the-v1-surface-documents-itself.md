---
status: accepted
---

# The `/v1` surface documents itself, and its reference needs no key

An ignis server serves the reference of its `/v1` surface itself, generated where the routes are registered. A
reference written beside the handlers goes stale the first time a route is added without one, and nothing fails when
it does; and kept in the repo, not on the wire, it leaves a client author reading Rust to learn a field name, most
of all for `POST /v1/decide` (ADR 0034), the endpoint that is ignis's own and that no external documentation covers.

- **The document is generated where the routes are registered.** The `/v1` routes are built through `utoipa-axum`'s
  router, which registers a handler and its `#[utoipa::path]` entry in the same call: a route cannot exist without
  its documentation entry, nor an entry without its route. A test asserts the resulting path set as an equality, so a
  route added with a bare `.route()` fails the suite rather than quietly going undocumented.
- **The reference is part of the `/v1` surface, not of the Playground.** The document and its Swagger UI page are
  always served, and the address a human types, `/v1`, redirects to the page. Withholding the Playground does not
  withhold the reference.
- **The reference is open even on a keyed server.** The API key gates the handler routes; the document and the page
  answer without one. A browser pointed at `/v1` cannot set an `Authorization` header, so a gated reference would be
  a reference nobody could read. It publishes the API's shape (no prompt, no completion, no load figure), and the
  document declares the `bearerAuth` scheme, so the page's *Authorize* button drives the gated routes from
  the page. ADR 0028 keys every exposed server, and a keyed server must not publish its load; the open reference is
  a deliberate carve-out: shape is not load.
- **Monitoring is not API.** The Prometheus exposition (ADR 0017: `/metrics` on its own listener, `/ui/metrics`
  behind the key) is outside the document, and a test fails if a path carrying `metrics`, or any `/ui` path, ever
  appears in it.
- **The Swagger UI distribution is vendored in the repo, not taken from `utoipa-swagger-ui`.** That crate embeds the
  same files through a host dependency that the `x86_64-pc-windows-gnu` host toolchain this project is developed on
  cannot build, and the MSVC target pin does not route around a host dependency. The vendored files cost the binary
  no more than the crate's own embed and build with any toolchain, offline, on every platform. `utoipa` and
  `utoipa-axum` are pure Rust and are taken as crates.
- **The narrative description stays.** The prose that describes the surface is at the tag since the archive left
  the tree (`records-2026-11-27:docs/archive/ignis-v1.md`, its second section); the generated document is the
  machine-readable contract. Neither is regenerated from the other.

## Consequences

- A new `/v1` endpoint ships documented or fails the suite, at the cost of one `#[utoipa::path]` block per handler,
  which is where the descriptions live.
- The alias `POST /v1/systemone` keeps its route and stays out of the document: OpenAPI has no notion of an alias,
  and listing it would repeat every schema reference under a second path. It is named in the document's own
  description instead.
- Two properties of `/v1/decide` cannot be expressed in JSON Schema and are stated in prose: that a JSON evidence
  reaches the model in the order its keys were written, and that a `choice`'s options are read in declared order.
  The schemas for those fields are free-form JSON.
- The binary grows by the vendored distribution. Its version is not vendored: it is pinned with every other frontend
  dependency, and a check in the local gate and on CI fails when the vendored copy and the pin disagree, so a
  forgotten sync fails the gate rather than shipping.
- An operator on an air-gapped machine gets the same page as everyone else: nothing on it is fetched from a CDN.
