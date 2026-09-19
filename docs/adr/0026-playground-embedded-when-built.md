# ADR 0026 — The Playground is embedded when built, served by default

## Status

Accepted (2026-09-14, owner decision). **Amended 2026-09-19** (owner
decision): the Playground is served by default. Everything else below stands;
only the opt-in half is superseded, by the Amendment section at the end.

## Decision

The **Playground** (a React + TypeScript page built with Vite) lives as source
in `web/`. Its build output `web/dist/` is not committed; the `ignis-server`
binary embeds it at compile time **if it exists**, and otherwise embeds a
fallback page telling the reader to run the frontend build. It is served under `/ui/` on the
existing listener unless the operator turns it off (see the Amendment); the
original decision was the opposite, opt-in through `--ui` alone. It talks only to surfaces ignis
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

## Amendment (2026-09-19): served by default

`--ui` was opt-in because it was shaped after `--metrics` (ADR 0017), and that
was the wrong sibling. Metrics open a second listener and build a projection
that costs work on every request; the Playground is a static page on the
listener that already exists, and a binary built without `web/dist` serves the
page telling you how to build it. Off by default therefore bought nothing and
cost a flag everyone passed — including the container image, whose `CMD`
existed only to supply it, and which any argument after the image name
silently replaced.

- Served by default. `--no-ui` turns it off, `--ui` still says on, and
  `IGNIS_UI` takes either, the flags winning over the environment as
  everywhere else in `crates/server/src/config.rs`.
- The environment variable is new too: the original decision refused one to
  match `--metrics`. A default that cannot be changed through the environment
  is unusable in a container, which is where this one has to be.
- The image's `CMD ["--ui"]` is gone: the server's own default is the image's.
- `--metrics` is untouched and still opt-in. `/ui/metrics` now needs
  `--metrics` and *not* `--no-ui`, which is the same condition written the
  other way round.
