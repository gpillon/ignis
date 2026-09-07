# ADR 0011 — `tracing` + `tracing-subscriber` as the structured logging foundation

## Status

Accepted (2026-09-07, grilling session on `.scratch/Ignis Structured Logging
Specification.md`).

## Context

Ignis has zero structured logging today: no `tracing`/`log`/`slog` dependency
in any crate, and 146 `println!`/`eprintln!` call sites across 18 files. A
structured logging spec was submitted requiring an OpenTelemetry-compatible
canonical event model (timestamp, severity, event name, body, attributes,
trace/span correlation), JSONL and pretty output from the same event, and —
critically — a system that can carry per-request internal trace spans
(admission → prefill → decode round → completion) without degrading the
inference hot path.

Separately, `crates/server/src/telemetry.rs` (design doc §5) already exists:
a JSONL sink for scheduler interval counters (`kind:"interval"` —
waiting/running/kv_used_pct, metrics-shaped) and per-request lifecycle events
(`kind:"request"` — admitted/ttft/done), selected by `IGNIS_TELEMETRY`/
`--telemetry` (issue #77, in progress). This predates the logging spec and is
exactly the kind of "ad-hoc subsystem-specific logging format" §1 of the spec
tells us not to keep growing. The interval-counter line is conceptually an
OTel *metric* (a gauge), not a log — out of scope here. The request-lifecycle
line is exactly spec §24 ("Request logging") and is scheduled to migrate onto
the canonical event model in Phase 2 (`ignis.request.admitted/ttft/done`),
without touching the interval-counter line or the `IGNIS_TELEMETRY`/
`--telemetry` sink-selection mechanism itself.

Two Rust ecosystems were considered: `log` + `env_logger`/similar, or
`tracing` + `tracing-subscriber`.

`log` has no native concept of a span: correlating "everything that happened
during this request" would require manually threading a request/trace ID
through every call site and log macro invocation. `tracing` has spans as a
first-class primitive, a `Layer`/`Subscriber` model that maps directly onto
the spec's "one structured event, N formatters" architecture (JSON layer,
pretty layer, future OTel layer all observing the same `tracing::Event`), and
`tower-http` already ships a `TraceLayer` for `axum` (which ignis's server
already depends on) to get the HTTP root span for free.

## Decision

- Ignis adopts **`tracing` + `tracing-subscriber`**, not `log`.
- The canonical event model is **not** a hand-built Rust struct that every
  call site constructs. It is `tracing::Event` + `Metadata` + active span
  context, observed by custom `tracing_subscriber::Layer` implementations
  (one per output format: JSON, pretty). Application code calls `tracing`
  macros directly (`info!(model = .., gpu_id = .., "Model loaded")`), matching
  the spec's own Rust example (§32) verbatim.
- **Critical-path performance is the deciding constraint, not a nice-to-have.**
  `tracing`'s per-callsite static `Interest`/level cache makes a disabled
  level near-zero-cost, which is required because ignis's hot path is
  prefill/decode: any logging change touching `ignis.prefill.*`,
  `ignis.decode.*`, or `ignis.cuda.graph.*` event emission must be validated
  against the existing performance gate (ADR 0007, G4, ≥99% of reference)
  before merge — design intent (no per-token/per-layer/per-kernel INFO logs,
  §26 of the spec) is necessary but not sufficient proof of no regression.

## Consequences

- New crate `crates/logging` owns the `tracing_subscriber` setup, the
  JSON/pretty `Layer`s, and log-format/log-level config resolution. Every
  other crate depends on `logging` only for macros/init, never the reverse.
  Named `logging`, not `telemetry`, to avoid colliding with the existing
  `crates/server/src/telemetry.rs` module (design doc §5: the scheduler
  interval-counter and request-lifecycle JSONL stream, wired to
  `IGNIS_TELEMETRY`/`--telemetry` in #77) — a different, pre-existing concern.
  See the Context note below and `.scratch/logging-roadmap.md` for how the
  two relate.
- Any future switch away from `tracing` would touch every call site in the
  codebase — this is a hard-to-reverse choice, which is why it's recorded
  here rather than left implicit.
- Every phase of the logging rollout that touches the prefill/decode/CUDA
  graph path must re-run the G4 performance gate as an explicit gate
  condition, not just pass code review.
