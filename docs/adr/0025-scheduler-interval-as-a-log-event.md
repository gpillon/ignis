---
status: accepted
---

# Scheduler interval counters are a DEBUG log event

The scheduler's interval counters are a canonical `tracing` event, `ignis.scheduler.interval`, at DEBUG, emitted only
when they change, and the separate telemetry sink that wrote them as a JSONL line on every scheduler step is removed.
The line had been kept out of the logging system (ADR 0011) because the counters are metrics-shaped, but ADR 0017 gave
them their real metrics surface, a projection read from the telemetry consumer's own state and never from rendered
output, so the line was no longer where metrics live. And it shared stdout with the canonical log stream without
sharing its format: an operator on a terminal got pretty logs with raw JSONL interleaved at step rate, one stream with
two renderers, the noisier one ignoring the operator's choice of format. Nothing outside the repo's own tests read the
line: no bench harness, no simulation script, no gate tool.

- **One stream, one format.** The event renders through the same JSON or pretty layer as every other event and is
  invisible at the default INFO level. A file of interval events is the JSON format at DEBUG, filtered on the event
  name. Asking for DEBUG also shows scheduler state changes, which is the intended use: diagnosis, not measurement.
- **Only when the counters change, not on every step.** A decode run holds `waiting` and `running` constant for
  thousands of steps; logging each one would saturate the DEBUG ring buffer (`QueueConfig::debug_trace_capacity`,
  drop-oldest) and evict every other DEBUG event. The step number rides along as `tick`, so the cadence is still
  readable.
- **Only authoritative values are attributes**: `tick`, `waiting`, `running`, `kv_used_pct` and `kv_evictions`.
  `kv_used_pct` is one since ADR 0030 put the scheduler's own occupancy on the model thread's tick. `prefilling` is
  omitted while it is a placeholder zero, by the rule ADR 0017 applies to metrics.
- **The change detection is over every attribute, `kv_used_pct` included.** Left out of it, the line would report a
  percentage frozen at the last time some other counter moved, and a decode run that fills the pool holds `waiting`
  and `running` constant throughout. The trade is deliberate. An integer percentage bounds a steady fill to a line per
  point rather than per step, but a load whose occupancy oscillates across a boundary does log a line per step, which
  is the crowding this decision exists to avoid. It is accepted because DEBUG is off by default, and because a
  percentage that does not move is worse than one that moves too often: a gauge nobody can trust is the placeholder
  this replaced.
- **The sink is gone whole**: `--telemetry` and `-t`, `IGNIS_TELEMETRY`, the `ignis.telemetry.sink_selected` and
  `sink_failed` events, and the sink parameter of the engine constructors. `--telemetry` is an unrecognized flag, and
  a start with it fails as one; `IGNIS_TELEMETRY` is ignored.
- **The model thread is untouched.** The event is built on the asynchronous telemetry consumer, from the tick facts
  the model thread sends anyway, so no work is added to the inference path. The wait-free counter snapshot
  (`Engine::interval_counters`) is unchanged and still published every tick.
