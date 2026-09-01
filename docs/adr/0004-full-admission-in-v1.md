# ADR 0004 — Full admission state machine in v1

## Status

Accepted (2026-09-02, grilling session).

## Context

v1 could have shipped with a plain FIFO admission (one global prefill lane + N
decode lanes, first-come-first-served) and deferred the fairness machinery.
The reference stack's admission is a dense state machine (protection, backfill
class, temporal credit, frontier distance); the reference notes explicitly warn
that it is delicate to touch, and a later tagged-lanes feature
(`@main`/`@agents`/`classifier` reservations) builds on top of it.

Options:
- (a) simple FIFO in v1; fairness machine later
- (b) port the full admission state machine into v1

## Decision

(b). The full admission state machine ships in v1, ported from the reference
stack and pinned by ported unit tests. Tagged lanes remain a later feature.

## Consequences

- v1 carries more up-front complexity; the scheduler is the highest-risk module
  and gets the most test coverage in the milestone.
- The later tagged-lanes feature becomes an admission-policy change, not a
  scheduler re-architecture.
- Porting the state machine without its historical context invites regressions:
  the port must preserve invariant behavior (protection promotion, credit decay,
  frontier distance) and each invariant gets a dedicated test.