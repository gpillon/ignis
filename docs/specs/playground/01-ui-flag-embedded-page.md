# 01 — `--ui`: serve the embedded Playground under `/ui/`

GitHub: #163

ADR: 0026 (Playground embedded when built, served opt-in). Glossary: **Playground** (`CONTEXT.md`).

## Problem Statement

The owner has no way to try ignis by hand from a browser. Before any chat UI can exist, ignis needs the seam that carries a frontend: a place for its source, a build that never makes cargo depend on node, and an opt-in route that serves the result from the single `ignis-server` binary.

## Solution

A Vite + React + TypeScript project in `web/` builds into `web/dist/` (gitignored). `ignis-server` embeds `web/dist/` at compile time when it exists, and a small built-in fallback page otherwise. With `--ui`, the existing listener serves it under `/ui/`; without it, the route is absent. This ticket's page is a placeholder ("Playground" heading + the model id from `GET /v1/models`) — the chat is ticket 02.

## User Stories

1. As the owner, I want `--ui` to enable the Playground, so that a default ignis has no browser surface.
2. As the owner, I want `http://127.0.0.1:8000/ui/` to open the page from the same binary, so that there is nothing extra to deploy.
3. As the owner, I want a binary built without a frontend build to still start and show a page telling me to run the frontend build, so that a missing `web/dist` is obvious, not a 404.
4. As a developer, I want `cargo build` and `cargo test` to never run node/npm, so that the workspace stays node-free.
5. As a developer, I want `npm run dev` in `web/` to proxy `/v1` and `/ui/metrics` to `IGNIS_URL` (default `http://127.0.0.1:8000`), so that I get hot reload against a running ignis.
6. As a developer, I want `npm run dev:mock` to serve a fake SSE chat and a fake `/ui/metrics` from the Vite dev server, so that I can work on the UI without touching the shared GPU.

## Implementation Decisions

- `web/`: Vite + React + TS, npm, `package-lock.json` committed, `web/dist/` and `web/node_modules/` gitignored. Vite `base: "/ui/"`. No UI kit.
- Embedding: a `build.rs` in `crates/server` checks whether `web/dist/index.html` exists, emits `cargo:rerun-if-changed` for `web/dist`, and selects between embedding `web/dist` and the built-in fallback page (e.g. `include_dir`/`rust-embed`, or generated `include_bytes!` table — implementer's choice, no npm invocation ever).
- CLI: boolean `--ui` in `config::resolve` (spec server/06), default off, no alias, no env var, no config key — same shape as `--metrics` (ADR 0017). Listed in `--help`.
- Routes (only when `--ui`): `GET /ui` redirects to `/ui/`; `GET /ui/` serves `index.html`; `GET /ui/<path>` serves the embedded asset with the right `Content-Type`; unknown `/ui/<path>` → 404 (no SPA catch-all needed — single page). `/` stays unrouted.
- Assets under `/ui/assets/` (hashed names) get a long `Cache-Control`; `index.html` gets `no-cache`.
- No new endpoint, fact, or work on the inference path; no change in core/runtime/kernel/model thread.
- `README.md`: a short "Playground" section — `npm --prefix web ci && npm --prefix web run build`, then `cargo build`, then `ignis-server --ui`.

## Testing Decisions

- Pure CLI: `--ui` defaults off, `--ui` sets it, no env var path.
- Router (no GPU, existing OpenAI HTTP test pattern): route absent without `--ui` (404 on `/ui/`); with `--ui`, `/ui/` is 200 `text/html`, `/ui` redirects, unknown asset 404, `/v1/models` unaffected.
- The fallback path is tested directly (the page the build selects when `web/dist` is absent) so `cargo test` covers it regardless of the checkout's frontend state.
- No GPU run, no G4 gate (ADR 0026).

## Out of Scope

- The chat and per-request figures (02); the Prometheus panel (03).
- Auth, TLS, non-localhost exposure.
- Any npm step inside cargo.
