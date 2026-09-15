# ADR 0026 — The Playground is embedded when built, served opt-in

## Status

Accepted (2026-09-14, owner decision).

## Decision

The **Playground** (a React + TypeScript page built with Vite) lives as source
in `web/`. Its build output `web/dist/` is not committed; the `ignis-server`
binary embeds it at compile time **if it exists**, and otherwise embeds a
fallback page telling the reader to run the frontend build. It is served only
when the operator passes `--ui` (no alias, environment variable, or config
key — the same shape as `--metrics`, ADR 0017), under `/ui/` on the existing
listener; without the flag the route is absent. It talks only to surfaces ignis
already exposes (`/v1/*`, and `/ui/metrics` when metrics are enabled — ADR
0017 owns that route and its API-key rule), measuring per-request figures in
the browser; it adds no endpoint of its own, no fact, and no work on the
inference path, so it needs no GPU run or G4 gate for acceptance.

## Considered Options

- **`build.rs` runs `npm run build`** — rejected: every Rust build and
  `cargo test` would require node, breaking the node-free workspace.
- **Commit `web/dist/`** — rejected: bundle churn in every frontend diff.
- **Serve a directory from disk (`--ui-dir`)** — rejected: loses the single
  self-contained binary.

## Consequences

- Two binaries from the same commit can differ: one built after the frontend
  build carries the Playground, one built before carries the fallback. This is
  deliberate — do not "fix" it by making cargo invoke npm.
- Prometheus in the Playground is a browser-side scrape of `/ui/metrics` — the
  same text exposition Prometheus reads on the metrics listener — parsed
  client-side, not a JSON endpoint made for the UI.
