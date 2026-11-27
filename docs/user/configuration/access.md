---
title: "Playground, metrics, access"
description: "The Playground, the metrics listener, API keys, CORS origins and public exposure."
sidebar:
  order: 11
---

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--server-ui <bool>` | `IGNIS_SERVER_UI` | on | Serve the Playground at `/ui/` ([ADR 0026](../../adr/0026-playground-embedded-when-built.md)). A binary built without `web/dist` serves the page that says how to build it. |
| `--server-metrics <bool>` | `IGNIS_SERVER_METRICS` | on | Serve Prometheus metrics on their own listener, and at `/ui/metrics` unless `--server-ui false` ([ADR 0017](../../adr/0017-prometheus-metrics.md)). `--server-metrics false` turns them off. |
| `--server-metrics-bind <addr>` | `IGNIS_SERVER_METRICS_BIND` | `127.0.0.1:9464` | The metrics listener's address; refused beside `--server-metrics false`. No API key, never exposed. When the default address is taken (a second server on the host), the server warns (`ignis.server.metrics_unavailable`) and serves without the listener (`/ui/metrics` still answers); another address you named that cannot be bound refuses the start. |
| `--server-api-key <key>` | `IGNIS_SERVER_API_KEY` | unset | Unset: `/v1` needs no key. Set: `Authorization: Bearer <key>`. `auto`: generate one and print it. |
| `--server-expose <mode>` | `IGNIS_SERVER_EXPOSE` | unset | `cloudflare-quick` publishes a `https://*.trycloudflare.com` URL, printed at start. Always requires an API key — one is generated when none is set ([ADR 0028](../../adr/0028-expose-modes-always-keyed.md)). The tunnel forwards to the API's port on `127.0.0.1`, so `server.bind` must accept IPv4 loopback (`127.0.0.1` or `0.0.0.0`); a tunnel that cannot open refuses the start. |
