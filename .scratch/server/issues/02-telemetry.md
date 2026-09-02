# 02 — telemetry (JSONL events + interval lines)

Status: resolved (commit 992aea0, 2026-09-02; GitHub #15)
GitHub: #15
Blocked by: #13 (core-04)

v1 telemetry (JSONL, `docs/design/ignis-v1.md` §5): one line per event, one
line per interval.

- `{"kind":"interval", waiting, prefilling, running, kv_used_pct,
  kv_evictions}` — scheduler counters, one line per interval.
- `{"kind":"request", id, event: admitted|ttft|done, ms, n, tok_s}` —
  one line per request event.
- `class` field reserved (tagged lanes, v1.1+); `sibling_prefix_reused_tok`
  counter from v1.1.

## Acceptance

- Telemetry emits one JSONL line per event + one per interval, matching the
  §5 schema.
- The interval line reflects the live scheduler counters.

Delivered (commit 992aea0): new `telemetry.rs` in `crates/server` — a
`Telemetry` emitter with injectable `TelemetrySink` (`NullSink` default,
`MemorySink` for tests, `StdoutSink` production default, `FileSink` via
`IGNIS_TELEMETRY`), an injectable `TelemetryClock` (a `FixedClock` keeps
tests deterministic — ADR 0006), and the `TelemetryCounters` source
(`Engine::with_stats` + `IntervalStatsProvider` for the interval
line's live counters). Wired into the engine's `submit()` / `step()`:
the `interval` line is emitted once per `Engine::step()` (a driver
tick), `request:admitted` on `SchedEvent::Admitted` (a lane dealt),
`request:ttft` on the request's first `SchedEvent::Token`,
`request:done` on `SchedEvent::Done` (`n` = total tokens, `tok_s` =
n / elapsed_ms from the clock), `kv_evictions` bumped on
`SchedEvent::Evicted`; the sink is a lock-protected buffer (non-blocking
on the request path). 9 new unit tests + 4 integration tests
(`tests/telemetry.rs`, at the `Engine` seam: §5 line shapes, fixed/stepped
clock determinism, sink injection, and a no-deadlock property under
4-thread concurrent `submit`/`step` on a shared engine + sink).
38/38 `ignis-server` tests green, clippy clean for the crate.

Deviation notes (tracked): `id` is the numeric `RequestId` (the §5
example shows a string `"r-042"`, but the ticket only specifies `id`);
request lines uniformly carry `ms` / `n` / `tok_s` per the ticket (the
§5 example omits some fields on some events — the ticket's uniform
field set wins). `class` and `sibling_prefix_reused_tok` are reserved
for v1.1 and not emitted.

Follow-up (tracked, pending on core): `prefilling` and `kv_used_pct` in
the interval line are **stubbed to 0** — the live counters are not
exposed through core's public API (the `Scheduler` trait has only
`submit` / `advance` / `is_idle` / `model_id` / `mode`; no stats
accessor, so a `Box<dyn Scheduler>` cannot reach `ConcreteScheduler`'s
`kv_used_pages` / `host_tier` / request-state counts). `waiting`,
`running`, and `kv_evictions` are real (event-derived from the routed
`SchedEvent`s). The `IntervalStatsProvider` seam is ready: once core
ships a `Scheduler::stats()` accessor (`waiting` / `prefilling` /
`running` = counts by `RequestState`, `kv_used_pct` =
`kv_used_pages / capacity * 100`, `kv_evictions` = a cumulative counter
at the evict/restore paths), the server wires a provider returning the
live counters and the interval line becomes authoritative (a core
follow-up — see PENDING.md).
