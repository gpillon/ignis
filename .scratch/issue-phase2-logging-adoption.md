## Problem Statement

Once `crates/logging` exists (Phase 1), Ignis still has 146 `println!`/`eprintln!` call sites and one unrelated ad-hoc JSONL format (`crates/server/src/telemetry.rs`'s `kind:"request"` lines) that don't go through it. Diagnostic output from `ignis-server` is indistinguishable from command results in one-shot tools, sensitive values have no enforced exclusion, and there's no resource identity (`service.name`, version) attached to anything.

## Solution

Migrate the diagnostic (non-command-result) call sites identified by audit to `crates/logging`'s canonical events, split stdout/stderr correctly for one-shot commands vs the long-running server, attach resource attributes and prefer OpenTelemetry semantic convention attribute names, enforce that sensitive values are never logged by default, and migrate the existing `kind:"request"` telemetry lines onto canonical `ignis.request.*` events. The scheduler interval-counter line and the `IGNIS_TELEMETRY`/`--telemetry` sink-selection mechanism (#77) are untouched — those counters are metrics-shaped, not logs.

## User Stories

1. As the owner, I want every `eprintln!` in `crates/server/src/main.rs` (startup validation, config errors, model/artifact load errors, the startup banner — all 15 sites, none of which are one-shot command results since `ignis-server` has no subcommands) converted to a structured event with a stable `event_name`, so that server startup failures are greppable and filterable by severity.
2. As the owner, I want the 3 diagnostic sites in `crates/vendor/src/main.rs` (unknown-flag/usage error, manifest load failure, generic subcommand error — the exit-code-2 paths) converted to structured events on stderr, so that `vendor-ninfer`'s scripting-facing exit-code contract stays intact while its failures are still structured.
3. As the owner, I want `vendor-ninfer`'s `verify`/`sync`/`repin`/`record-patch` result output and `--help` text left exactly as `println!`/`print!` today, so that this migration does not touch legitimate command output that scripts and `jq` pipelines may already depend on.
4. As the owner, I want `ignis serve`'s entire event stream (pretty or JSON) on stdout regardless of severity, so that WARN/ERROR aren't silently split to a different stream I'm not watching.
5. As the owner, I want a one-shot command's machine-readable stdout (e.g. `ignis inspect model --output json`) to never be interleaved with diagnostic logging, so that `| jq .` keeps working exactly as it does for hand-rolled CLI output today.
6. As the owner, I want every event to carry `service.name=ignis` and `service.version` (from the crate's version), so that logs are self-identifying without me having to know which binary emitted them from context alone.
7. As the owner, I want ignis-specific attributes to live under the `ignis.*` namespace and standard concepts (like `service.name`) to use their OpenTelemetry semantic convention name rather than an invented alias, so that a future OTel-aware consumer doesn't need an ignis-specific translation layer.
8. As the owner, I want prompts, completions, tool arguments, API keys, bearer tokens, and other secrets to never appear in a log record by default, so that log output is safe to share, paste, or forward without a manual redaction pass.
9. As the owner, I want request IDs and other correlation identifiers to be attached only where useful for correlation, not indiscriminately on every event, so that logs don't balloon with high-cardinality noise that provides no operational value.
10. As the owner, I want the existing `kind:"request"` admitted/ttft/done lifecycle lines to become `ignis.request.admitted`/`ignis.request.ttft`/`ignis.request.done` canonical events (same information: request id, elapsed ms, token count, tok/s), so that request lifecycle observability lives in the one canonical system instead of a second bespoke JSONL shape.
11. As the owner, I want the `kind:"interval"` scheduler-counter line and the `IGNIS_TELEMETRY`/`--telemetry` sink selection left completely alone, so that #77 (in progress) is not blocked or destabilized by this migration.
12. As a future maintainer, I want a test asserting that a covered logging path never emits a known-sensitive value, so that a future call site accidentally logging a secret is caught before it ships.

## Implementation Decisions

- `crates/server/src/main.rs`: replace each of the 15 `eprintln!` sites with a `tracing` macro call carrying an `ignis.<subsystem>.<event>` name (e.g. `ignis.config.invalid`, `ignis.model.load.failed`, `ignis.process.started` for the startup banner) and structured attributes for the values currently interpolated into the string (path, error, artifact, model, bind address). Severity: startup/config validation failures and load failures are `ERROR` (refuse to start); the telemetry-sink-selected/model-loaded/artifact-verified/mock-compute notices are `INFO` or `WARN` (telemetry fallback) as appropriate.
- `crates/vendor/src/main.rs`: convert only the 3 diagnostic sites (bad-args/usage error, manifest load failure, generic subcommand `Err`) to structured events on stderr; every `verify`/`sync`/`repin`/`record-patch` result line and `--help`/usage text stays as direct `println!`/`print!`, unmodified.
- Resource attributes: `service.name` is a constant `"ignis"`; `service.version` reads `env!("CARGO_PKG_VERSION")`; a `service.instance.id` may be generated once per process (e.g. a random ID at startup) if a stable per-process identifier proves useful — not required if nothing consumes it yet.
- Semantic convention preference: check the OpenTelemetry Semantic Conventions registry for existing attribute names before inventing a new one (`service.name` over `app_name`, etc.); anything ignis-specific goes under `ignis.*` (e.g. `ignis.scheduler.queue_depth`, `ignis.kv_cache.block_count`), never under `otel.*`.
- Sensitive-data exclusion: no call site introduced by this migration logs a prompt, completion, tool argument, API key, bearer token, cookie, password, or private credential. Where an error value might incidentally contain such content (rare in the migrated call sites, which are mostly path/config/error-message strings), review case by case; operational metadata (request id, model, token counts, latency, queue depth, GPU id, memory usage, batch size) remains fine to log.
- `crates/server/src/telemetry.rs`: the `RequestLine` (`kind:"request"`, fields `id`/`event`/`ms`/`n`/`tok_s`) is replaced by three canonical events (`ignis.request.admitted`, `ignis.request.ttft`, `ignis.request.done`) with the same attributes (`request.id`, `duration_ms`, token counts, `tok_s` — or the equivalent semantic-convention name where one exists) emitted through `crates/logging` instead of `TelemetrySink::write_line`. `Telemetry::emit_interval`/`IntervalLine`/`IntervalCounters`/`IntervalStatsProvider` and the `TelemetrySink`/`FileSink`/`StdoutSink`/`IGNIS_TELEMETRY`/`--telemetry` machinery are not touched by this issue.
- stdout/stderr split: `ignis serve` keeps its entire event stream on stdout (no severity-based routing to stderr). One-shot commands (`vendor-ninfer`, `ignis inspect`, `ignis config validate/dump`, `ignis benchmark`) reserve stdout for command results and route diagnostic logging to stderr.

## Testing Decisions

- Table/parameterized tests over the migrated `crates/server/src/main.rs` startup-failure paths asserting: the emitted event has the expected `event_name`, the expected severity, and the previously-interpolated value now appears as a typed attribute rather than only inside `body`.
- A sensitive-value test: feed a known secret-shaped string (e.g. a fake bearer token) through the covered logging paths and assert it never reaches a rendered record — this covers spec §39 test #12.
- A one-shot-stdout-cleanliness test: run a `vendor-ninfer` subcommand with logging at `debug`/`trace` and assert stdout still parses as the expected result shape (JSON where applicable) with no diagnostic lines mixed in — spec §39 test #13.
- Migrated `kind:"request"` → `ignis.request.*` tests reuse the existing `MemorySink`-style pattern from `crates/server/src/telemetry.rs`'s test module, adapted to assert on the canonical event's attributes instead of the old JSON shape; the existing interval-counter tests are untouched (still testing the un-migrated code).
- No GPU test required; these are startup/CLI/unit-level paths, not the inference hot path.

## Out of Scope

- The scheduler interval-counter line, `IntervalStatsProvider`, and the `IGNIS_TELEMETRY`/`--telemetry` sink-selection mechanism — unchanged.
- Bounded/priority queues, backpressure, hot-path performance validation (Phase 3).
- Trace/span correlation (Phase 4).
- Any new CLI flag beyond what #77 already defines.
- Rewriting `crates/bench/src/main.rs` or `crates/artifact/src/bin/inspect.rs` output — those are predominantly command-result output (per the audit) and are not touched by this issue; a future issue may still want to sweep them for any genuinely diagnostic sub-lines if found.

## Further Notes

- Depends on Phase 1 (`crates/logging` foundations) landing first.
- Audit basis: `crates/server/src/main.rs` — all 15 `println!`/`eprintln!` sites are diagnostic (no subcommands, pure long-running service). `crates/vendor/src/main.rs` — mixed; only 3 of ~15 sites (usage error, manifest load failure, generic subcommand error) are diagnostic, the rest are the actual `verify`/`sync`/`repin`/`record-patch` report content or `--help` text.
- Full spec: `.scratch/Ignis Structured Logging Specification.md` §9-11, §16-18, §22-24, §29-30, §39 (partial). Roadmap: `.scratch/logging-roadmap.md`.
