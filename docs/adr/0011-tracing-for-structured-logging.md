---
status: accepted
---

# `tracing` + `tracing-subscriber` as the structured logging foundation

ignis logs through `tracing` and `tracing-subscriber`, not `log`. The logging spec asked for one canonical event
compatible with OpenTelemetry (timestamp, severity, event name, body, attributes, trace and span correlation), JSONL
and pretty output from that same event, and per-request internal trace spans (admission, prefill, decode round,
completion) that do not degrade the inference hot path. `tracing` has spans as a first-class primitive; its
`Layer`/`Subscriber` model is the spec's "one structured event, N formatters" (a JSON layer, a pretty layer and a
future OTel layer all observing the same `tracing::Event`); and `tower-http` ships a `TraceLayer` for `axum`, which
the server already depends on, so the HTTP root span comes for free.

- **The canonical event model is not a hand-built struct that every call site constructs.** It is `tracing::Event`,
  its `Metadata` and the active span context, observed by one custom `tracing_subscriber::Layer` per output format.
  Application code calls the `tracing` macros directly.
- **Critical-path performance is the deciding constraint, not a nice-to-have.** `tracing`'s per-callsite static
  `Interest` and level cache makes a disabled level near-zero-cost, which is required because ignis's hot path is
  prefill and decode.
- **A logging change that touches the emission of `ignis.prefill.*`, `ignis.decode.*` or `ignis.cuda.graph.*` events
  must have its cost to prefill and decode speed measured before merge**, as an explicit condition of the merge and
  not by code review alone, in every phase of the logging work that touches the prefill, decode or CUDA graph path:
  ignis against ignis, as ADR 0082 states for a speed claim (a branch against main, at least two launches per build),
  where the 99% performance gate against the reference (ADR 0007) stood when this was decided. The design intent, no
  per-token, per-layer or per-kernel INFO log, is necessary but not sufficient proof of no regression.
- **`crates/logging` owns the subscriber setup, the JSON and pretty layers and the resolution of the log format and
  level.** Every other crate depends on it for macros and init only, never the reverse. It is named `logging`, not
  `telemetry`, so as not to collide with `crates/server/src/telemetry.rs`, the server's consumer of request and
  scheduler state, a different concern.
- **No subsystem keeps a log format of its own**: an ad-hoc subsystem-specific format is what the spec says not to
  keep growing. The request lifecycle is the canonical events `ignis.request.admitted`, `ttft` and `done`, and the
  scheduler's interval counters, first left outside as metrics-shaped, are a canonical event too (ADR 0025).
- **The choice is hard to reverse**, which is why it is recorded: a switch away from `tracing` would touch every call
  site in the codebase.

## Considered Options

- **`log` with `env_logger` or similar**: rejected. It has no native concept of a span, so correlating everything
  that happened during a request would mean threading a request or trace id by hand through every call site and log
  macro.
