# 05 — full admission state machine

Status: needs-triage
GitHub: #16
Blocked by: #13 (core-04)

The full **admission state machine** (`docs/design/ignis-v1.md` §2,
`CONTEXT.md` "Admission state machine") — the fairness machinery that
decides which request gets which lane:

- **protection** (resident lanes are not evicted),
- **backfill class**,
- **temporal credit**,
- **frontier distance**.

## Acceptance

- The full admission state machine (protection / backfill class / temporal
  credit / frontier distance) drives lane assignment.
