---
status: accepted
---

# Full admission state machine in v1

The full admission state machine ships in v1, ported from the reference stack and pinned by ported unit tests. The
reference's admission is a dense state machine (protection, backfill class, temporal credit, frontier distance)
that its own notes warn is delicate to touch, and the tagged-lanes feature (`@main`/`@agents`/`classifier`
reservations) builds on top of it. Tagged lanes remain a later feature.

## Considered Options

- **A plain FIFO admission in v1** (one global prefill lane and N decode lanes, first come, first served), with the
  fairness machinery deferred. Not taken: tagged lanes would then need the scheduler re-architected.

## Consequences

- v1 carries more up-front complexity; the scheduler is the highest-risk module and gets the most test coverage in
  the milestone.
- The later tagged-lanes feature becomes an admission-policy change, not a scheduler re-architecture.
- Porting the state machine without its historical context invites regressions: the port must preserve invariant
  behavior (protection promotion, credit decay, frontier distance) and each invariant gets a dedicated test.
