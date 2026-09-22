# ADR 0038 — the `Compute` seam carries one attention head's scores

## Status

Accepted (2026-09-22, owner — spec `docs/specs/decide/13-point-by-attention-head.md`,
GitHub #260). **Extends ADR 0034**, which made the answer-token readout the
one thing besides a token that crosses the `Compute` seam, and the permitted
set the one thing a decode lane carries in. This is a third, and it is held
to the same rule: the job names exactly what to read, and exactly that comes
back.

Sources: `docs/findings/2026-09-21-one-attention-head-points.md`,
`2026-09-21-the-head-points-in-the-engine.md`,
`2026-09-22-the-codec-costs-the-head-its-read.md`,
`2026-09-22-how-the-head-map-is-read.md`, and the acceptance recorded in
`2026-09-22-the-head-points-through-decide.md`.

## Context

`/v1/decide` answered a `point` with a constrained decode: one prefill and
ten decode rounds writing `x` and `y` digit by digit (ADR 0034's second
path). The studies behind spec 13 found a better answer already inside the
prefill that run begins with: query head 10 of the tenth GQA layer
(**L39.h10**), read at the position after the forced `{"x":`, attends to the
image token the target's label begins on. Read by TAG's region rule it lands
inside the target on 227 of 240 synthetic scenes at 1024 px and 236 at
4096 px, where the chain lands 212 and 169.

Reading it needs something no seam carried: one row of one head's attention,
restricted to the image. The fused attention kernel never materializes a
score row, the leaf returns tokens, and the one existing way to see those
scores was a test-only tap that copies every key of an armed layer to the
host — megabytes per request, and a process-wide global.

Two constraints decided where the scores are formed. The keys must be the
ones attention read, *as* it read them: under hq-e8-2b the prompt route
decodes most keys through the codec and serves the residual window's rows
exact, and the accuracy above was measured on exactly those (the codec
alone costs the head 20 scenes at 1024 px; the window buys back 15). And a
fan-out's second question over one image finds that image's keys in the
cache, written by an earlier request — so the scores cannot come from a copy
taken at write time.

## Decision

- **An attention readout crosses the seam as one score per key of one span.**
  `PrefillJob.attention` names a GQA layer, a query head and a span of
  absolute prompt positions; `PrefillOutcome.attention` returns
  `q · k / sqrt(head_dim)` for the job's last position against each key of
  the span, before any softmax. At 4096 px that is 16,384 floats (64 KB).
  No full attention row and no logits row crosses, and a job that asks for
  none allocates nothing and launches nothing new.
- **The leaf scores the keys where attention read them.** Inside the GQA
  layer, right after its attention and inside its scope: under hq-e8-2b from
  the prompt route's materialized key plane, with the query rotated into the
  codec's frame (an orthonormal rotation, so the dot product is unchanged);
  under BF16 from the cache's pages. The kernel is ours, not vendored. The
  score buffer comes from the prefill chunk's own scratch scope and is
  reserved by the load's plan — only on a vision load, since the span is an
  image's.
- **A read the leaf cannot make is a failed question, never a wrong point.**
  The small-T route materializes no keys, and a banded prompt keeps only its
  last band. The leaf then reports the readout unread, the prefill itself
  stands, and the scheduler ends that request with `FinishReason::Error`.
  The scheduler also keeps a head point's last chunk at nine tokens or more —
  the chunk cut, the reuse trim and the retained prefix's reach all hold back
  that tail — so under hq-e8-2b the reading chunk takes the prompt route.
- **The head is a calibrated constant keyed to the artifact.** A compiled-in
  table maps the artifact's content hash to (GQA ordinal, query head); the
  served NVFP4 27B maps to (9, 10). It cannot be computed at load — it was
  chosen on labelled scenes by cross-validation — and it is never read on an
  artifact it was not chosen for. A load without an entry answers `point`
  by chain and says so at startup.
- **A head point is a decision.** It reads at its last prefill position and
  generates nothing, so `RequestInput.decision` now says *what* a decision
  reads — `DecisionRead::Answers` or `DecisionRead::Attention` — and every
  rule that made a readout a request kind (no decode lane, no residency, no
  checkpoint, a zero generation budget) holds for it unchanged.

## Considered options

**Ship the tap.** It exists and it is the oracle. Rejected: it copies every
key of the armed layer to the host at write time, which is both the wrong
cost (megabytes per request where 64 KB are needed) and the wrong keys — the
pre-codec ones — and under prefix reuse it does not have the image's keys at
all. It stays test-only, as the independent check the leaf is held to.

**Score on the host from the logits of digit tokens.** Free through the
existing readout, and it names the chain's own first-digit confidence.
Rejected as the point itself: the finding that motivated this measured the
chain's leading digit at p 0.97-0.99 and its expectation equal to its argmax,
so it carries no more position than the chain's first round. It remains a
candidate for a second signal beside the head.

**An exact copy of the head's KV head.** Worth 5 of 240 scenes at 1024 px
over the consumed keys. Rejected for now: it exists only for keys the request
itself wrote, so it breaks exactly where a fan-out shares an image's prefix.

## Consequences

- The seam's contract grows a third reading, and every CPU-only `Compute`
  has to produce it: `MockCompute` returns a deterministic map peaked at a
  documented key, and the leaf's stubs hand back a predictable one (ADR 0006).
- `ignis_prefill_options` grew six fields (ADR 0016: appended, one recognized
  size), and the prefill scratch plan of a vision load grew by one float per
  envelope token.
- The pointing head is the first calibrated constant keyed to the artifact.
  A new artifact needs a recalibration (spec 13 § Further Notes, run with
  `crates/server/tests/attention_head_point_gpu.rs`), and
  `crates/core/tests/pointing_head_artifact_gpu.rs` fails in the GPU profile
  until one is recorded.
- A `method` label on the decision metrics would be a contract change
  (ADR 0017) and is not made here: a head point counts as a `point`.
