# ADR 0023 — one eviction priority across GPU residency and the host tier

## Status

Accepted (2026-09-11) — GitHub #65, phase 4. Spec:
`.scratch/runtime/specs/04-reference-feature-floor.md`. Extends ADR 0004 (the
full admission state machine in v1) to the tier below the GPU.

## Context

Two eviction policies exist in the tree and were written years apart in spirit.
`admission.rs` decides which resident lane is the better victim: request class
first — `Agent` before `Interactive` — then least-recently-used within a class,
then lane id as a tie-break (`retained_lane_is_better_victim`). `host.rs`
decides what the host tier discards: probation before protected, then LRU, with
no notion of class at all, because when it was written no request could state
one.

Phase 4 makes both policies real at once: the host tier starts holding actual
device state (ADR 0024), and requests start declaring their class over HTTP. If
the two policies stay independent, a request's class changes meaning the moment
it crosses the PCIe bus — an `Interactive` snapshot would be discarded ahead of
an `Agent` snapshot merely because the `Interactive` one landed in probation
more recently. That is not a tier policy; it is an accident of where the
machinery was written.

A single ordering shared by both levels was considered and rejected. On the
GPU, something is actively being served, so the admission machinery's
protection and entitlement must outrank class: a protected lane mid-stream is
not a victim because its owner is an agent. In the host tier nothing is being
served, so protection has no meaning there, and class is the only thing left
that says whose work costs most to lose.

## Decision

One priority model, expressed as two orderings that share a class definition.

**GPU to host — what leaves the card:**

1. eligibility and protection (a sequence is eligible only at a chunk boundary;
   protected lanes are last)
2. request class (`Agent` before `Interactive`)
3. least-recently-used, with lane id as the final tie-break

**Host to discard — what the tier drops or re-prefills:**

1. request class (`Agent` before `Interactive`)
2. probation before protected
3. least-recently-used

Eviction from the GPU runs on the admission-refusal path, not pre-emptively.
When both tiers are full, admission refuses, as it does today.

This ordering is expected to change as workloads teach us more; it is recorded
here so that a change is a decision rather than a drift.

## Consequences

- `host.rs`'s two-tier machinery keeps its structure and gains the class it
  never had. The probation and protected tiers become the *second* key, not the
  first.
- The policy is only as good as the class, so the class has to arrive from the
  workload. That is the tagged-lanes work in the same phase; without it every
  request is `Interactive` and both orderings collapse to their old behaviour,
  which is a safe default rather than a broken one.
- The asymmetry between the two levels is deliberate and has to be tested as
  such. The case that proves it: an `Interactive` snapshot in probation
  outliving an `Agent` snapshot in protected.
- A third request class, if one is ever added, changes two orderings rather than
  one.

## Amendment (2026-09-16) — retained state goes first (ADR 0029)

**Retained state** — prompt checkpoints and retained prefixes, which no live
request needs — is placed ahead of both orderings.

- **On the device**, retained state is released before any lane is
  considered. It spills to KV-RAM if the budget can hold it, and is discarded
  otherwise. It never causes an admission refusal or wait.
- **In KV-RAM**, a new first key is added: retained entries before evicted
  live sequences. Request class, probation before protected, and
  least-recently-used then follow as above. A retained entry carries the
  class of the request that produced it.

The reason is asymmetry of loss. An evicted live sequence is work that will
certainly be redone; a retained entry is a bet that it will be needed.

## Amendment (2026-09-16) — spill admission and the Interactive TTL (#190)

Owner decision, replacing "discarded otherwise" in the device bullet above.

- **A spill displaces only what ranks below it.** When KV-RAM cannot hold a
  checkpoint the device is giving up, retained entries ranking strictly below
  the newcomer in KV-RAM's order may be discarded for it; if even all of those
  are not enough, nothing is discarded and the device discards the newcomer.
  The room is planned before anything goes, so a spill that fails costs no
  other entry. Live snapshots are never displaced by a spill.
- **An Interactive entry idle past a TTL ranks as an Agent's probation entry**
  (`--retained-interactive-ttl`, default 300 s, a starting value). Class alone
  would let conversations nobody returns to hold KV-RAM against every
  subagent. Agent entries have no TTL yet.
- **A live sequence holding a shared prefix is a victim like any other.** Its
  snapshot materializes the prefix's pages (ADR 0024 as amended), so it
  resumes rather than re-prefilling. A lane holder is still preferred over a
  lane-less candidate, so a lane-less `Prefilling` request is now chosen only
  when every lane is protected.

A single structured retention score that would replace these orderings is
#199.
