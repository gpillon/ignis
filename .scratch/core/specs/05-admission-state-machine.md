# 05 — full admission state machine

GitHub: #16

The full **admission state machine** (`docs/design/ignis-v1.md` §2,
`CONTEXT.md` "Admission state machine") — the fairness machinery that
decides which request gets which lane:

- **protection** (resident lanes are not evicted),
- **backfill class**,
- **temporal credit**,
- **frontier distance**.

Delivered (commit e582621): `admission.rs` (pure Rust port of the
reference admission policy — protection freeze, donor-prefix selection,
persistent-vs-temporal backfill, temporal-credit decay, frontier
distance, retained-lane victim policy) wired into `ConcreteScheduler` as
the lane-deal driver, plus the KV resource dimension (per-request page
reservation `ceil((prompt + effective_max) / kv_page_tokens)`,
over-reservation charged at deal, `Oversized` rejection at submit,
hard-cap completion). 11 unit tests pin each invariant (ADR 0004) +
4 end-to-end scenarios in `crates/core/tests/admission_machine.rs`
(protection freeze + backfill classification, lane-pressure hold,
oversized rejection, persistent backfill). CPU-tested (ADR 0006); the
full workspace `cargo test` is green.

Note: the protection's **Drain** phase is unreachable in v1's resource
model (a temporal backfill's work is bounded by its temporal credit ≤ the
last donor's work, so temporal borrowers always finish before the last
donor; by the time "safe without temporals" could hold, the head fits and
is dealt through the plain deal branch). It is kept for reference
fidelity (ADR 0004) and pinned by a doc note in `admission_machine.rs`.

## Acceptance

- The full admission state machine (protection / backfill class / temporal
  credit / frontier distance) drives lane assignment.
