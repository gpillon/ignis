# 03 — request state machine + basic admission

Status: resolved (commit 5d13202, 2026-09-02; GitHub #12)
GitHub: #12
Blocked by: #9 (core-01)

The request state machine + basic admission for `ignis-core`:

- Request lifecycle: `admitted → prefilling → running → done`.
- Basic lane assignment (which request gets which decode lane).
- The fairness machinery (full admission) is a follow-up (`core-05`).

## Acceptance

- The request state machine drives the lifecycle (admitted → prefilling →
  running → done).
- Basic admission assigns lanes to admitted requests.
