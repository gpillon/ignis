---
status: accepted
---

# The Playground is embedded when built, served by default

The Playground, a React and TypeScript page built with Vite, lives as source in `web/`, and its build output
`web/dist/` is not committed: `ignis-server` embeds the build at compile time if it exists, and otherwise a fallback
page telling the reader to run the frontend build. It is served under `/ui/` on the existing listener by default, and
one field of the `server` group, `ui` (ADR 0046), turns it off: it is a static page on a listener that already
exists, and a binary built without `web/dist` serves the page saying how to build it, so off by default bought
nothing and cost a flag everyone passed. It talks only to surfaces ignis already exposes, `/v1/*` and `/ui/metrics`
(ADR 0017 owns that route and its API-key rule), and measures per-request figures in the browser: it adds no
endpoint of its own, no fact and no work on the inference path, so it needs no GPU run or gate for acceptance.

## Considered Options

- **`build.rs` runs `npm run build`**: every Rust build and `cargo test` would require node, breaking the node-free
  workspace.
- **Commit `web/dist/`**: bundle churn in every frontend diff.
- **Serve a directory from disk**: loses the single self-contained binary.
- **Opt-in, like the metrics once were**: the wrong sibling. Metrics open a second listener and build a projection
  that costs work on every request; the Playground costs a route.

## Consequences

- Two binaries from the same commit can differ: one built after the frontend build carries the Playground, one built
  before carries the fallback. This is deliberate: do not "fix" it by making cargo invoke npm.
- Prometheus in the Playground is a browser-side scrape of `/ui/metrics`, the same text exposition Prometheus reads
  on the metrics listener, parsed client-side, not a JSON endpoint made for the UI. `/ui/metrics` is served when the
  Playground and metrics are both on.
- The switch can be set through the environment: a default that cannot be changed there is unusable in a container,
  which is where this one has to be.
- The container image has no `CMD`: the server's own default is the image's, and a `CMD` that existed only to supply
  the flag was silently replaced by any argument after the image name.
