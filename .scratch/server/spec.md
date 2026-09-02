# HTTP server — spec

The Rust server crate (`ignis-server`) exposes an OpenAI-compatible HTTP
API over localhost (no auth, configurable bind) and emits the v1 telemetry
stream. It routes requests into the core scheduler and streams back tokens.
Per `docs/design/ignis-v1.md` §1 (OpenAI-compatible HTTP) and §5 (telemetry).

## v1 scope (priority order)

1. **OpenAI-compatible HTTP** — `server-01`. Endpoints:
   - `GET /v1/models` — list the loaded model(s).
   - `POST /v1/chat/completions` — chat completions (streaming + non
     streaming), routed into the scheduler; chat template applied from the
     artifact's frontend object set.
   - `POST /v1/responses` — the responses API.
   Localhost-only, no auth, configurable bind address / port.
2. **Telemetry (JSONL events + interval lines)** — `server-02`. One line per
   event, one line per interval (JSONL): `kind` = `interval` (scheduler
   counters: waiting / prefilling / running, kv_used_pct, kv_evictions) and
   `request` (admitted / ttft / done with tok-s). `class` field reserved
   (tagged lanes, v1.1+). `sibling_prefix_reused_tok` counter from v1.1.

## Acceptance

- `GET /v1/models` returns the loaded model.
- `POST /v1/chat/completions` routes into the scheduler and streams tokens
  back; the chat template is applied from the artifact frontend objects.
- `POST /v1/responses` works (OpenAI responses shape).
- Telemetry emits one JSONL line per event + one per interval, matching the
  §5 schema.

## References

- Design: `docs/design/ignis-v1.md` §1 (OpenAI-compatible HTTP), §5
  (telemetry).
- Upstream: `core-04` (scheduler to route into), `artifact-02` (frontend
  objects for the chat template + tokenizer).