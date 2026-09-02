# 03 — request state machine + basic admission

Status: needs-triage
Blocked by: core-01

The request state machine + basic admission for `ignis-core`:

- Request lifecycle: `admitted → prefilling → running → done`.
- Basic lane assignment (which request gets which decode lane).
- The fairness machinery (full admission) is a follow-up (`core-05`).

## Acceptance

- The request state machine drives the lifecycle (admitted → prefilling →
  running → done).
- Basic admission assigns lanes to admitted requests.