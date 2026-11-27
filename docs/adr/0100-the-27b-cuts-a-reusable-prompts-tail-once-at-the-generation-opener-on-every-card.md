---
status: accepted
amends: [ADR-0029]
---

# The 27B cuts a reusable prompt's tail once, at the generation opener, on every card

[ADR 0029](0029-cross-request-state-reuse.md) kept the 27B's second tail cut on Blackwell (amendment of 2026-10-07, #306,
and of 2026-11-05, #508) because the extra traversal at the opener's page floor was assumed to cost ~19 ms on the
RTX 5090. Measured on 2026-11-17 for spec #646, that cut costs 45-55 ms per request there: the small traversal plus a
synchronous 222 MiB copy of the mutable state into a host retained slot over the PCIe link. The owner chose the single
cut (spec #646, ticket #649).

- **The 27B's generation-opener page rides the prompt checkpoint's capture on every card and with no card read**, as
  Flash-Next's does on every card and the 27B's own loads on Ampere and Ada already do. A reusable prompt is cut once,
  at the opener; the imaged prefix at the opener's page floor is no longer published for it. The decision stays the one
  function `config::arch::opener_page_rides_capture`.
- **The closed thinking block keeps its own cut.** Only the opener's page floor goes.
- **The request that no longer resumes at the page floor** is a prompt that differs from an earlier one only in its
  last 63 tokens before the generation opener: it resumes at the reuse point before it, the previous turn's checkpoint
  or the system block. An identical retry, a conversation's next turn and a subagent burst resume as before.
- **The 27B's split points move on Blackwell**, so its greedy text can change where two tokens are nearly tied
  ([ADR 0007](0007-correctness-is-self-checked-not-matched-against-a-reference-engine.md), ADR 0029's reuse boundaries).
- **A measurement switch restores the second cut**, for comparisons: `IGNIS_SECOND_TAIL_CUT=1`, a contributor's,
  read at each model load as the leaf's measurement switches are (unset or empty is the single cut, any other value
  refuses the load by name), reported on the load's `ignis.model.loaded` line (`second_tail_cut`), and documented with
  the other measurement switches in the contributor's testing page.
