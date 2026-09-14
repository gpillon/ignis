# 03 — Playground Prometheus panel

GitHub: #165

ADR: 0026, 0017. Glossary: **Playground**. Builds on 01; needs `GET /metrics` (#89).

## Problem Statement

Once ignis serves `/metrics` behind `--metrics` (ADR 0017), the owner has to run a Prometheus + Grafana stack just to glance at engine load while trying requests in the Playground.

## Solution

A panel in the Playground that scrapes `GET /metrics` from the browser, parses the Prometheus text format client-side, keeps a rolling in-memory history, and draws simple live views. No new server endpoint.

## User Stories

1. As the owner, I want the panel to appear only when `/metrics` answers 200, so that a server without `--metrics` shows no broken panel.
2. As the owner, I want current waiting/running requests, so that I see scheduler pressure while chatting.
3. As the owner, I want accepted/completed/rejected (by reason) totals and their rates, so that I see admission behaviour.
4. As the owner, I want generated-token rate, KV eviction and prefix-reuse totals/rates, so that I see work and cache pressure.
5. As the owner, I want TTFT and request-duration histograms shown as distributions (and approximate p50/p95 over the window), so that I see latency beyond averages.
6. As the owner, I want the build version shown, so that I know which ignis I'm looking at.
7. As the owner, I want a configurable poll interval (2–5 s, default 5 s) and pause, so that I control scrape load.

## Implementation Decisions

- Browser polls `/metrics` (same origin); interval clamped to ≥ 2 s. No server change; this is just a scraper under ADR 0017's HTTP-plane budget.
- Parser: pure module for text format 0.0.4 — `HELP`/`TYPE`, labels, counters, gauges, histogram `_bucket`/`_sum`/`_count`. Renders only the metrics it knows from ADR 0017's contract table; unknown series are listed raw, not dropped silently.
- Rates computed from successive samples; a counter going down (server restart) resets the series instead of drawing a negative rate.
- History: rolling window in memory (e.g. last 15 min), cleared on reload.
- One small chart dependency allowed (implementer's choice, lightweight); follow the dataviz conventions for colours and legends.

## Testing Decisions

- vitest: parser over a fixture copied from #89's exposition output (plus malformed lines), rate computation incl. counter reset, histogram quantile approximation.
- Manual smoke against `npm run dev:mock`'s fake `/metrics`, then once against `ignis-server --metrics --ui` when the GPU is free.
- No GPU acceptance run; no Rust change expected.

## Out of Scope

- A JSON metrics endpoint for the UI; alerts; long-term storage.
- Metrics outside ADR 0017's contract (prefilling, KV usage) until they exist server-side.
