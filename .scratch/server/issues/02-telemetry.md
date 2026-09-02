# 02 — telemetry (JSONL events + interval lines)

Status: needs-triage
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
