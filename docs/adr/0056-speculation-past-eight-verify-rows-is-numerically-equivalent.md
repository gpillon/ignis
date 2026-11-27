---
status: accepted
---

# Speculation past 8 verify rows is numerically equivalent, not bit-exact

A Flash-Next verify round carries lanes x (drafts + 1) rows, within a budget (`spec.draft_rows`) that is 8 unless a
load names a wider one. Up to 8 rows every linear, mix and grouped GEMV takes its decode-phase GEMV route; past 8 they
take their tensor-core routes, whose reduction order differs, and the MoE runs its decode route in launches of 8 rows.
**A verify budget past 8 rows is held numerically equivalent to speculation off, not to its text.**

The seam MTP is held to at 8 rows, the near-tie rule (greedy spec-on's free-running text equals spec-off's, a
divergence passing only within one BF16 ulp on the prefill route), refuses wide rounds where 8 rows show no divergence
on the same prompts. But a flip at a near tie is permanent in a free-running text, so that rule measures where the
first near tie falls as much as how far the routes drift. Teacher-forced, the same tokens through both routes, the
default's own 8-row round already moves its logits from spec-off's one-token rounds, and the wide routes are the same
class of drift, about 1.5 times wider at the widest, not a new one
([finding](../findings/2026-10-18-flash-next-verify-rows-past-8.md)).

So past 8 rows a budget's rounds are held teacher-forced, offered spec-off's own greedy text as drafts: each row
within a tolerance in BF16 ulps of spec-off's row at the same text index, a pick moved only at a near tie, a bound of
its own and tighter than the tolerance; the text's mean log-probability within a bound of spec-off's, either way; the
head's first-draft acceptance above a floor and within a band of the same prompts' at 8 rows, either way. What stays
exact at every width: the commit (a round commits its own rows' picks), the state after it (a rejected draft is a
draft never made), and the run itself (deterministic per budget). The rules and their constants, each with its
measurement and its reason, are `crates/core/tests/support/logit_drift.rs`'s, which the tests of
`crates/core/tests/flash_next_speculative_gpu.rs` run.

## Considered Options

- **Launches of 8 rows wherever a route switches at 8**: each row then computes as in a round of 8, which is itself
  not bit-exact against spec-off, so the seam would still be the near-tie rule, and the dense weights are read once
  per 8 rows. Rejected: it pays bandwidth for a seam that is a tolerance anyway.
- **GEMV kernels wider than 8 tokens**, each token's sums in the 8-token kernel's order, the weights read once a
  round: bit-identical per row to a round of 8, at the cost of a kernel redesign. Rejected for the same reason; it
  stays the path if a budget past 8 ever has to be token for token.

## Consequences

- At a budget past 8, greedy output may differ from speculation off where two tokens are within the drift of each
  other; the guide says so beside `--spec-draft-rows`.
- The near-tie rule stays the oracle up to 8 rows. The default budget and every profile are unchanged, and a load
  that names no budget plans and runs as before. No profile takes a wider budget: on the RTX PRO 6000, whose
  measurement was to decide it, every budget past 8 loses to 8
  ([finding](../findings/2026-10-18-verify-rows-on-the-rtx-pro-6000.md)).
- The constants are measured on the RTX 5090 with the tests' prompts. The tolerance is the next power of two above
  the wide budgets' widest drift, a margin of only 1.1 times, kept that tight because the discriminator is close: a
  row read on another context, what a row from the wrong lane or position looks like, mostly sits past it and much
  less often past a looser bound. A change of the wide routes (a new GEMM tiling, the wider GEMV kernels above) or
  another card re-measures them, the RTX PRO 6000 first, before a profile takes a budget past 8.
- The owner decided the equivalence and asked that acceptance stay within a band. The moved pick's own bound and the
  band's width were set afterwards by the review, from the measured data: the owner's text allowed a moved pick at a
  near tie within the logit tolerance, and the review's bound is stricter.
- The rows come from a measurement's copy of the round's logits (`ignis_decode_options::out_logits`, appended by
  ADR 0016's rule), not a readout: the round still samples. No serving path sets it, and the 27B and a round with a
  permitted set refuse it, each by name.
- These bounds are Flash-Next's. The method, a round past 8 rows held teacher-forced, row by row, reaches the 27B's
  DFlash2 round past 8 columns a lane, whose reference is its base round rather than speculation off and whose bounds
  are its own (ADR 0104): on that model the base round itself already sits past these.
