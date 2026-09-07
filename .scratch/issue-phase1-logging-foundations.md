## Problem Statement

Ignis has no structured logging today. Every diagnostic message in the codebase is a `println!`/`eprintln!` call (146 occurrences across 18 files) with no stable shape, no severity, no way to filter, and no way to consume it as machine-readable data. There is no canonical event model to build the rest of the logging spec (`.scratch/Ignis Structured Logging Specification.md`) on top of, so nothing else in that spec — hygiene migration, hot-path guarantees, internal tracing — can start.

## Solution

A new crate, `crates/logging`, provides the one canonical structured event model the whole application will log through, backed by `tracing` + `tracing-subscriber` (ADR 0011). It exposes JSON (JSONL) and pretty formatters that render the *same* underlying event, an `auto` mode that picks between them based on whether stdout is a TTY, and log-format/log-level configuration via environment variables. Nothing in this issue migrates any existing call site — it only stands the foundation up and wires it into `ignis-server`'s startup.

Named `logging`, not `telemetry`: `crates/server/src/telemetry.rs` already owns that name for an unrelated, pre-existing concern (the scheduler interval-counter/request-lifecycle JSONL stream behind `IGNIS_TELEMETRY`/`--telemetry`, #77). The two do not merge in this issue.

## User Stories

1. As the owner, I want every subsystem to emit events through one shared API, so that there is exactly one logging system to reason about instead of per-subsystem ad-hoc formats.
2. As the owner, I want `IGNIS_LOG_FORMAT=json` to produce one JSON object per physical line with no ANSI codes and no surrounding text, so that I can pipe `ignis serve`'s output straight into a log collector.
3. As the owner, I want `IGNIS_LOG_FORMAT=pretty` to render the same event as a readable, optionally colored line for interactive use, so that I don't need a JSON viewer while developing.
4. As the owner, I want `IGNIS_LOG_FORMAT=auto` (the default) to choose pretty when stdout is an interactive terminal and JSON otherwise, so that the right thing happens by default in both a terminal and a container without me setting anything.
5. As the owner, I want an explicit `IGNIS_LOG_FORMAT` to always override auto-detection, so that a container that happens to attach a TTY (or vice versa) doesn't silently pick the wrong formatter.
6. As the owner, I want `IGNIS_LOG_LEVEL` (`trace`/`debug`/`info`/`warn`/`error`) to control what gets emitted, so that I can turn up verbosity for a debugging session without a rebuild.
7. As the owner, I want every event to carry a stable, class-level `event_name` (`ignis.<subsystem>.<event>`) separate from its human-readable `body`, so that machine consumers never have to parse prose to find out what happened.
8. As the owner, I want machine-readable values (durations, byte counts, IDs) to live in typed structured attributes, not interpolated into the body string, so that a JSON consumer gets `duration_ms: 17400` instead of having to regex `"in 17.4s"`.
9. As the owner, I want the JSON output's timestamp to be RFC 3339 UTC with sub-second precision, so that ordering and correlation across records is unambiguous regardless of local timezone.
10. As the owner, I want the pretty formatter to be free to abbreviate/humanize values (bytes → GiB, ms → seconds) for display, so that interactive reading is comfortable, while the underlying event keeps native machine units.
11. As a future maintainer, I want application code to call ordinary `tracing` macros (`info!(model = .., "Model loaded")`) rather than construct a custom event struct by hand, so that instrumenting a new call site is no more ceremony than today's `println!`.
12. As the owner, I want this phase to not require a CLI parser, so that it does not block on or duplicate #77 — env vars are enough for now, and the config resolution function is shaped so a future CLI flag can override it without rework.
13. As the owner, I want `logging` init wired into `ignis-server`'s startup with a sanity performance check, so that turning logging on does not measurably slow down server startup.

## Implementation Decisions

- New crate `crates/logging`, workspace member, depending on `tracing` and `tracing-subscriber` (new workspace dependencies).
- Application code instruments with `tracing`'s own macros (`info!`, `warn!`, `error!`, `debug!`, `trace!`, `#[instrument]` where useful later) — `crates/logging` does not define a parallel macro surface or an intermediate `LogEvent` struct that call sites construct by hand. The canonical event *is* `tracing::Event` + `Metadata` + active span context; `crates/logging`'s `Layer` implementations are the only place that observes and renders it (ADR 0011).
- Two `tracing_subscriber::Layer` implementations, both driven by the same event stream:
  - a JSON layer producing one compact JSON object per line, UTF-8, no ANSI, embedded newlines JSON-escaped, mapping onto OTel LogRecord-shaped fields (`timestamp`, `severity_text`, `severity_number`, `event_name`, `body`, `attributes`, and — when populated by later phases — `trace_id`/`span_id`).
  - a pretty layer producing a human-readable line/block, ANSI colors gated on `stdout` being an interactive terminal (or an explicit override), free to abbreviate byte/duration values.
- `auto` mode: pretty when stdout is an interactive TTY, JSON otherwise; an explicit `IGNIS_LOG_FORMAT` always wins over the TTY check.
- Config resolution lives in a pure function in `crates/logging` (e.g. `LogConfig::resolve(env: impl Fn(&str) -> Option<String>, override: LogConfigOverride)`), mirroring the seam style #77 established for `crates/server/src/config.rs` — no direct `std::env::var` calls scattered around, and no dependency on a CLI parser existing yet. `LogConfigOverride` is an empty/no-op struct for now; #77 or a later CLI issue is expected to populate it.
- Recommended values: `IGNIS_LOG_FORMAT` ∈ {`auto`, `pretty`, `json`} (default `auto`); `IGNIS_LOG_LEVEL` ∈ {`trace`,`debug`,`info`,`warn`,`error`} (default `info`).
- `ignis-server`'s `main.rs` calls `logging::init(...)` once, at the very top of `main`, before any other startup work, so the earliest possible messages already go through the canonical system rather than a bootstrap `eprintln!` (a minimal bootstrap fallback before `logging::init` is acceptable per spec §31, but should be as small as possible).
- No event names or attribute names are invented speculatively here — this issue does not migrate any call site; Phase 2 (a follow-up issue) does that.

## Testing Decisions

- Unit tests in `crates/logging` covering: JSON output is valid JSON; one physical line per event; embedded newlines in a body/attribute do not break JSONL framing; pretty and JSON layers are driven from the same `tracing::Event` (i.e., changing the active layer does not change what attributes/severity/event_name exist, only how they're rendered); severity text/number mapping is correct and stable; `auto` mode picks pretty for a simulated TTY and JSON otherwise; an explicit `IGNIS_LOG_FORMAT` overrides the TTY simulation; structured attributes retain their native serde type (an `i64` attribute serializes as a JSON number, not a string).
- These are the first slice of the spec's §39 eighteen-item test contract; the rest (sensitive-value exclusion, one-shot-stdout cleanliness, trace-id presence/absence) land with the phases that introduce the relevant behavior.
- Config resolution tests follow the pattern already used for `thinking::parse_default_enable_thinking`/`parse_default_reasoning_effort` and #77's `config::resolve`: pure functions taking injected env, no real process environment, no real TTY.
- No GPU test is needed for this issue — nothing here touches the request/inference path yet. A CPU-only sanity timing check around `logging::init()` at server startup is enough to confirm it adds no meaningful latency; the real hot-path guarantee is Phase 3's job.

## Out of Scope

- Migrating any existing `println!`/`eprintln!` call site (Phase 2).
- Resource attributes (`service.name`, `service.version`, instance id), OpenTelemetry semantic convention attribute names, sensitive-data redaction, cardinality discipline (Phase 2).
- Bounded/priority logging queues, backpressure, shutdown flush guarantees (Phase 3).
- Trace/span correlation, `request.id`-as-`trace_id`, any span instrumentation (Phase 4, ADR 0012).
- CLI flags for `--log-format`/`--log-level` (depends on whatever CLI parser #77 or a later issue lands; this issue only shapes the config seam to accept an override later).
- Any change to `crates/server/src/telemetry.rs` (scheduler counters / `IGNIS_TELEMETRY`) — untouched by this issue.
- The C/C++/CUDA/FFI logging bridge (spec §33/34) — closed as N/A; kernel leaf `printf` usage is almost entirely in test code, not production paths.

## Further Notes

- Full spec: `.scratch/Ignis Structured Logging Specification.md`. Grilling-session decisions: ADR 0011 (`tracing` choice), ADR 0012 (trace_id = request.id, Phase 4), ADR 0013 (no typed attribute registry — attribute stability is enforced by tests, not by the compiler). Execution plan: `.scratch/logging-roadmap.md`.
- Phases 2, 3, and 4 all depend on this issue landing first; they are filed as separate issues.
