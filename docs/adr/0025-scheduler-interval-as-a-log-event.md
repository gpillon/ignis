# ADR 0025 — Scheduler interval counters are a DEBUG log event

## Status

Accepted (2026-09-13, owner decision). Supersedes the interval-counter
boundary in ADR 0011 (Context, Consequences) and the "Existing JSONL interval
telemetry remains compatible" consequence in ADR 0017. Amended 2026-09-18
(#216): `kv_used_pct` is no longer a placeholder — ADR 0030 §Observability
gives the model thread's tick the scheduler's own occupancy — so it is an
attribute of the interval event and joins the change detection below. The
"only authoritative values are attributes" rule is unchanged and now omits
`prefilling` alone.

## Context

ADR 0011 kept the scheduler interval counters (`waiting`, `running`,
`kv_evictions`, and the placeholder `prefilling`/`kv_used_pct`) out of the
logging system: they were metrics-shaped, so `crates/server/src/telemetry.rs`
went on writing them as `{"kind":"interval",...}` JSONL through its own
`LineSink`, one line per scheduler step, to the file named by
`--telemetry`/`IGNIS_TELEMETRY` or to stdout by default.

Two things changed since then.

- ADR 0017 gave the counters their real metrics surface: an opt-in
  Prometheus projection read from the telemetry consumer's own state, never
  from rendered output. The JSONL line is no longer where metrics live.
- The line shares stdout with the canonical log stream, but not its format.
  An operator on a terminal gets pretty logs (`IGNIS_LOG_FORMAT=auto`) with
  raw JSONL interleaved at step rate — one stream, two renderers, and the
  noisier one ignoring the operator's format choice.

Nothing outside the repo's own tests reads the line: no bench harness, no
`bench/sim` script, no gate tool parses it.

## Decision

- The interval counters are emitted as a canonical `tracing` event,
  `ignis.scheduler.interval`, at **DEBUG**. They render through the same
  JSON/pretty layer as every other event and are invisible at the default
  INFO level.
- The event is emitted **only when the counters change**, not on every
  step. A decode run holds `waiting`/`running` constant for thousands of
  steps; logging each one would saturate the DEBUG ring buffer
  (`QueueConfig::debug_trace_capacity`, drop-oldest) and evict every other
  DEBUG event. The step number rides along as `tick`, so the cadence is still
  readable.
- Only authoritative values are attributes: `tick`, `waiting`, `running`,
  `kv_used_pct` (#216), `kv_evictions`. `prefilling` is omitted while it is a
  placeholder zero, by the same rule ADR 0017 applies to metrics.
- The change detection is over every attribute, `kv_used_pct` included. Left
  out of it, the line would report a percentage frozen at the last time some
  *other* counter moved — a decode run that fills the pool holds `waiting`
  and `running` constant throughout. An integer percentage takes at most 101
  values, so a pool filling steadily logs a line per point, not per step.
- The separate telemetry sink is removed: `--telemetry`/`-t`,
  `IGNIS_TELEMETRY`, the `ignis.telemetry.sink_selected`/`sink_failed`
  events, and the sink parameter of the engine constructors.

## Consequences

- stdout carries one stream in one format. A file of interval events is
  `IGNIS_LOG_FORMAT=json IGNIS_LOG_LEVEL=debug`, filtered on `event_name`.
- The wait-free counter snapshot (`Engine::interval_counters`) is unchanged
  and still published every tick; the Prometheus projection of ADR 0017 reads
  that, not the log.
- The model thread is untouched: it sends the same `Tick` facts as before,
  and the event is built on the asynchronous telemetry consumer. No inference
  path work is added, so no gate run is attached to this change.
- `--telemetry` is now an unrecognized flag, and a start with it fails as
  one. `IGNIS_TELEMETRY` is ignored.
- Asking for DEBUG now also shows scheduler state changes. That is the
  intended use: diagnosis, not measurement.
