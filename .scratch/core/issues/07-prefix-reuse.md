# 07 — prefix reuse (sibling prefix caching)

Status: needs-triage
Blocked by: core-06

**Prefix reuse** (`docs/design/ignis-v1.md` §2, `CONTEXT.md` "Prefix
reuse"):

- Concurrent requests sharing a prefix skip the redundant prefill — siblings
  reuse the shared KV prefix instead of re-prefilling.
- The `sibling_prefix_reused_tok` counter tracks reuse (exposed via
  telemetry, `server-02`).

## Acceptance

- Sibling prefix reuse is active: concurrent requests sharing a prefix skip
  the redundant prefill.
- The `sibling_prefix_reused_tok` counter increments under a "1 main + N
  subagents" load.