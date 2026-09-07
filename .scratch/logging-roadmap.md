# Ignis structured logging — phased roadmap

Grilling session 2026-09-07 on `.scratch/Ignis Structured Logging
Specification.md`. Decisions recorded in ADR 0011/0012/0013 and
`CONTEXT.md` (Observability section). This file is the execution plan only —
it will go stale as phases land; the spec and ADRs are the durable record.

**Every phase touching prefill/decode/CUDA-graph code MUST re-run the G4
performance gate (ADR 0007) before merge. Design intent alone (no per-token
logs, static level checks) is not sufficient proof of no regression on the
critical inference path.**

## Phase 1 — foundations

- New crate `crates/logging` (not `telemetry` — that name is already taken by
  `crates/server/src/telemetry.rs`'s scheduler-metrics JSONL stream, a
  different concern; see ADR 0011): canonical event model on top of `tracing`
  + `tracing-subscriber`.
- JSON (JSONL) and pretty `Layer`s from the same event; `auto` mode (TTY
  detection, explicit config wins).
- Config: `IGNIS_LOG_FORMAT` / `IGNIS_LOG_LEVEL` env vars only. No CLI flags
  yet — a CLI parser is expected to land before this work; design the config
  resolution function to accept an explicit override so it plugs in later
  without rework.
- Severity levels, event name convention, body/attributes split (spec §2,3,
  5-8,12-15,31).
- No impact on the hot path expected at this phase (no call sites migrated
  yet) — still worth a sanity G4 run once `logging` init is wired into
  `ignis serve` startup, since subscriber init itself must not add startup
  latency that matters.

## Phase 2 — adoption + hygiene

- Migrate `crates/server/src/main.rs`: all 15 `eprintln!`/`println!` sites
  are diagnostic (service has no one-shot subcommand) — all convert to
  structured events.
- Migrate `crates/vendor/src/main.rs`: only the 3 diagnostic sites (usage
  error, manifest load failure, generic subcommand error — exit code 2
  paths) convert; the `verify`/`sync`/`repin`/`record-patch` result output
  and `--help` text stay on stdout untouched (spec §10.2/§11 — CLI result
  output is not a log).
- stdout/stderr split for one-shot commands (`vendor-ninfer`, `ignis inspect`,
  etc.) vs `ignis serve` (stdout for everything, no severity-based stream
  split).
- Resource attributes (`service.name=ignis`, `service.version`, instance
  id), OTel semantic convention preference, `ignis.*` namespace for
  ignis-specific attributes.
- Sensitive data: no prompts/completions/secrets logged by default.
- Cardinality discipline (§30).
- Migrate `crates/server/src/telemetry.rs`'s `kind:"request"` lines
  (admitted/ttft/done) onto canonical `ignis.request.admitted/ttft/done`
  events. The `kind:"interval"` scheduler-counter line and the
  `IGNIS_TELEMETRY`/`--telemetry` sink-selection mechanism (#77) are
  untouched — those counters are metrics-shaped, not logs, and stay out of
  this spec's scope.
- Baseline test contract (spec §39 items covering JSON validity, JSONL
  framing, severity mapping, event name preservation, sensitive-value
  exclusion, one-shot stdout cleanliness, format-override precedence,
  native attribute types).

## Phase 3 — hot-path guarantees

- Confirm no INFO-level per-token/per-layer/per-kernel emission anywhere in
  prefill/decode/CUDA-graph code (§26).
- Two-channel bounded queue: DEBUG/TRACE droppable under backpressure;
  INFO/WARN/ERROR bounded but blocks briefly instead of dropping (these are
  rare by construction — state-transition events only — so a brief block
  never touches the per-token loop).
- Graceful shutdown flush for pending important events, with a hard bound
  on shutdown blocking time.
- **Gate: G4 performance run, ≥99% of reference, before merge.**

## Phase 4 — internal tracing (last, per owner's explicit sequencing)

- Root span at HTTP ingress (`tower-http` `TraceLayer` on `axum`).
- Child spans: admission → prefill → decode round (not per-token) → MTP
  verify (when active) → completion.
- `trace_id` = existing `request.id` (ADR 0012), no separately fabricated
  trace identifier.
- Spans stay at request/round granularity — anything finer is hot-path
  logging in disguise and needs the G4 gate re-run before it can land.

## Explicitly out of scope / closed

- **C/C++/CUDA/FFI logging bridge (spec §33/34): N/A for now.** Kernel leaf
  `printf` usage (203 occurrences, 41 files) is almost entirely in test
  code, not production paths. Documented rule: no `printf` in production
  kernel leaf code; revisit only if a real production case appears.
- **Typed attribute registry: rejected (ADR 0013).** Discipline + regression
  tests only, given no external attribute consumers yet.
- **Direct OTLP export: not required** (spec §21 already settles this —
  stdout + external collector is the supported deployment shape).
