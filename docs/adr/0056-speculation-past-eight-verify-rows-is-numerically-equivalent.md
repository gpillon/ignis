---
status: accepted
---

# ADR 0056 — speculation past 8 verify rows is numerically equivalent, not bit-exact

## Status

Accepted (2026-10-18, owner decision: option 3 of
[`docs/findings/2026-10-18-flash-next-verify-rows-past-8.md`](../findings/2026-10-18-flash-next-verify-rows-past-8.md)).
GitHub #329 item 5. The default budget (8 rows) and every profile are unchanged; whether a
profile takes a wider budget is the RTX PRO 6000 measurement's to decide. It decided: there
every budget past 8 loses to 8 (0.42-0.84x), so the `rtxpro6000` profile keeps 8
([verify rows on the RTX PRO 6000](../findings/2026-10-18-verify-rows-on-the-rtx-pro-6000.md)).

**Amended 2026-10-18 (review of #329 item 5, after acceptance, from the measured data).** Two
thresholds below were added after the owner accepted option 3, by the review and not by the
owner: the moved pick's own bound of **16 ulps** (the owner's text allowed a moved pick at a
near tie within the 64-ulp tolerance; measured moved picks sat at 0-8 ulps, so 16 is stricter),
and the acceptance band's width of **0.1** (the owner asked for acceptance within a band; 0.1 is
1.5x the widest measured 0.066).

## Context

A Flash-Next verify round carries lanes x (drafts + 1) rows. Since #329 item 5 the budget goes up
to 32 (`spec.draft_rows`, `--spec-draft-rows`; 0 = 8, the default). Up to 8 rows every linear,
mix and grouped GEMV takes its decode-phase GEMV route; past 8 they take their tensor-core
routes, whose reduction order differs, and the MoE runs its decode route in launches of 8 rows.

The seam MTP was held to — greedy spec-on's free-running text equals spec-off's, a divergence
passing only within one BF16 ulp on the prefill route (the near-tie rule) — refused 3 of the
16-row cells (gaps of 4-21 ulps) where 8 rows showed no divergence on the same prompts. A flip
at a near-tie is permanent in a free-running text, so the rule measures where the first
near-tie falls as much as how far the routes drift.

Measured teacher-forced (the same tokens through both routes, logits row by row; the finding's
second table), the default's own 8-row verify round already moves its logits up to 40 BF16
ulps (at the row's largest magnitude) from spec-off's one-token rounds; 16, 24 and 32 rows move
them up to 58, picks moving only at gaps of 0-8 ulps. The wide routes are the same class of drift
as the default's, about 1.5x wider at the widest, not a new one. What
stays exact at every width: the commit (a round commits its own rows' picks), the state after
it (a rejected draft is a draft never made), and the run itself (deterministic per budget).

## Decision

**A verify budget past 8 rows is held numerically equivalent to speculation off, not to its
text.** The tests (`crates/core/tests/flash_next_speculative_gpu.rs`, rules in
`crates/core/tests/support/logit_drift.rs`):

- **Up to 8 rows, unchanged and exact**: spec-on's text equals spec-off's up to the near-tie
  rule; reject and (BF16) the oracle leave none's text and state bit for bit.
- **Past 8 rows** (16, 24, 32; 4-8 lanes; BF16 and hq-e8-2b), per cell:
  - **teacher-forced logits**: the verify rounds are offered spec-off's own greedy text as
    drafts, so each drafted column holds spec-off's token while the lane's committed text is
    spec-off's; each such row is held to spec-off's row at the same text index. Every logit
    within **64 BF16 ulps** at the spec-off row's largest magnitude, and a different pick only
    where the spec-off row rates the two picks within **16 ulps** (a near tie). A lane whose
    pick moved leaves the comparison after that round, so nothing cascades. 64 is the next
    power of two above the wide budgets' widest measured drift (58.3, one tail logit): a margin
    of only 1.1x on the RTX 5090 (1.6x over the default's own 40.2). It is kept that tight
    because the discriminator is close: the rows of a lane whose text left spec-off's at its
    second token, compared at the same indices anyway (another context, what a row read from
    the wrong lane or position looks like), measured 33-202 ulps, median 103, 89% past 64 but
    only 59% past 96. The RTX PRO 6000 run re-derives it before a profile takes a budget past
    8. The pick bound is its own, not the logit tolerance (64 ulps at a top logit of 20 is 8
    logits, no near tie): measured moved picks sat at gaps of 0-8 ulps, and 16 is 2x that.
  - **exact at any width**: every round of the teacher-forced run and of the head's run
    committed its own rows' picks (the drafts accepted are the longest prefix each the pick
    after the column before it; what stands pending is the pick after the run); the reject
    drafter leaves none's text and state bit for bit, and in BF16 KV so does the oracle.
  - **determinism**: the same budget gives the same text, rounds, drafts, rows (hashed) and
    state run to run.
  - **quality**: the teacher-forced mean log-probability of spec-off's text within **0.02
    nats a token** of spec-off's own, either way (measured from -0.0117 to +0.0077, wide minus
    spec-off: the verify rows read spec-off's own greedy text both less and more likely than
    spec-off's rows do, so the check is two-sided); the head's first-draft acceptance at least
    **0.5** (measured 0.74-0.94; a round reading the wrong rows accepts next to nothing) and
    within **0.1** of the same prompts' at 8 rows, either way (the head run on them two lanes at
    a time, whose rounds take 8 rows at any budget: the owner's band; measured within 0.066,
    the default's own 4-lane cells 0.046).
- The rows come from a measurement's copy of the round's logits,
  `ignis_decode_options::out_logits`: a Flash-Next round's head rows, host BF16, before the
  draw or the accept reads them. Not a **readout** (GLOSSARY.md: named tokens, no draw): the
  round still samples. No serving path sets it; the 27B and a round with a permitted set
  refuse it, each by name and each tested.

## Considered options

- **1. Launches of 8 rows wherever a route switches at 8** (the linears under a decode-phase
  flag, the shared expert, the hyper-connection decode route, the grouped GEMVs). Each row then
  computes as in a round of 8 — which is itself not bit-exact against spec-off, so the seam
  would still be the near-tie rule — and the dense weights are read once per 8 rows: about
  2.5 GB more a 16-row round. Rejected: it pays bandwidth for a seam that is a tolerance anyway.
- **2. GEMV kernels wider than 8 tokens**, each token's sums in the 8-token kernel's order, the
  weights read once a round. Bit-identical per row to a round of 8, at the cost of a kernel
  redesign (the FP8 single, SwiGLU and grouped GEMVs, the BF16 GEMV, the hyper-connection
  route). Rejected for now for the same reason; it stays the path if a budget past 8 ever has
  to be token-for-token.
- **3. Accept the drift and test it as drift** (chosen).

## Consequences

- At a budget past 8, greedy output may differ from speculation off where two tokens are
  within the drift of each other: measured picks moved at gaps of 0-8 ulps on the plain row.
  The user README says so beside `--spec-draft-rows`; spec flash-next/07's oracle is this ADR's
  past 8 rows.
- The near-tie rule stays the oracle up to 8 rows; nothing about the default changes.
- The tolerance, the pick's near-tie bound, the log-probability epsilon, the acceptance floor
  and its band are constants of the test's support module, measured on the RTX 5090 with these
  prompts. A change of the wide routes (a new GEMM tiling, option 2) or another card (the RTX
  PRO 6000 first) re-measures them, as the finding records.
- The default's plan lines are main's byte for byte in the 15 cells `flash_next_bind` pins
  against main's own figures (no speculation and verify-only, both KV formats): nothing about
  such a load that names no budget changes. MTP loads (the profiles') were not planned: the
  test's descriptors carry no head.
- `out_logits` is appended to `ignis_decode_options` (ADR 0016's rule): a NULL costs nothing,
  a set one costs one host copy before the round's synchronize.
