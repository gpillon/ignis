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

## Amendment (2026-10-08) — a reservation grows (ADR 0045)

The KV resource dimension above reserves a request's whole
`ceil((prompt + effective_max) / kv_page_tokens)` at admission, and the
reservation never grows. ADR 0045 (spec `vram-budget/03`, its page-wise
reservation ACs) replaces that rule; the lane machinery above is unchanged.

- **The bound is unchanged:** prompt plus `effective_max`, never past the
  context. `ContextExceeded` and `Oversized` are still decided on it at
  submit. `effective_max` without a `max_tokens` is now the server's default
  cap (`--default-max-tokens`, 38,912), clamped to the context; `0` restores
  the whole context.
- **The reservation is smaller and grows.** At admission it is the prompt (its
  tail past a claimed prefix) plus one growth step (32 pages). From its first
  round a lane keeps at least one step of room ahead and takes the next step
  as it generates, up to its bound.
- **When pages run out**, the lowest-ranked state below the requester moves
  down a tier, by ADR 0023 as amended 2026-10-08. When nothing can move, an
  admission waits and a growing lane parks.
- **Sequences enter under the entry rule.** A restore or an admission enters
  only with room for four steps for itself and for every resident sequence
  ranked above it.
- **Remaining work is now finite.** The protection arithmetic and the frontier
  distance read a request's current reservation and its
  `remaining_work = effective_max` as before. With the default cap,
  `remaining_work` is finite for a request that names no cap.
