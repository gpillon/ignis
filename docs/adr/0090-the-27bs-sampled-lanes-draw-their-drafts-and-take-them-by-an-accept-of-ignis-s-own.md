---
status: accepted
---

# The 27B's sampled lanes draw their drafts and take them by an accept of ignis's own

Under sampling, DFlash2 proposes one fixed draft per column and the vendored accept
(`speculative_accept_greedy_drafts`) takes it with the target's probability of that token. That is the
rule of speculative sampling for a drafter whose distribution is one-hot, and it is correct: a refused
draft is replaced by a draw from the target's distribution without it, so the text keeps the target's
distribution. It is also what caps acceptance under sampling. Even a drafter that always proposed the
target's most likely token would see its first draft refused 18% of the time on recorded agent turns at
temperature 1, because the target, sampling, picks another token that often (spec #574).

The owner decided on 2026-11-11 that the 27B's sampled lanes may run the general rule instead, the
**sampled draft**: the drafter draws each column's draft d from its own distribution q over its 16
candidates (the softmax of the selector's transition scores after the previous column's draft, at the
lane's temperature times a factor, cut by a top-p), and the accept takes d when `u·q(d) < p(d)`, so with
probability min(1, p(d)/q(d)); on a refusal the lane's token is drawn from max(p − q, 0), renormalized over
the target's support. Whatever q is, the emitted token is distributed as p: q moves how often a draft is
taken, never what text can come out. Replayed offline over 61,692 recorded agent rounds, it commits 3.74
tokens a round against 3.46.

**This is ignis's own implementation of a different rule, not a patch of a vendored one.** The vendored
accept is not wrong for the rule it states (ADR 0037 admits a patch for a demonstrated bug, and there is
none), and no registry lacks a case (ADR 0088). Its file stays byte for byte as vendored (ADR 0010), and it
stays in use for every round under the fixed draft and for Flash-Next's MTP rounds. The draft's draw and the
accept are kernels of the leaf's own
(`kernel/src/sampled_draft.cuh`): the arithmetic exists once as a host reference, `ignis_core::sampled_draft`,
held on every push to a double-precision oracle that sweeps the random numbers over a grid instead of
sampling them, and a GPU test holds the device kernels to the same oracle. The accept builds each column's
target distribution with the vendored sampler's device primitives, so the p it compares against is the
vendored accept's to the bit; only the verdict is ours.

## Consequences

- **Three settings in the speculation group, the 27B's alone**: `spec.draft_rule` (`sampled`, the default
  since the rule's gate on the RTX 5090 passed, or `fixed`, the fixed draft, the rule of earlier releases, one
  option away), `spec.drafter_temperature_factor` (0.9) and `spec.drafter_top_p` (0.95). Each needs a reload and
  `--spec-backend dflash2`. Where no round drafts with DFlash2 (Flash-Next, whose MTP rounds draft the fixed
  draft, and the 27B with no backend or `off`) a value other than the default is refused by name, and the
  default is accepted and does nothing: the configuration's rule that naming a default names nothing, which a
  file `config generate` wrote with every default needs to start either model. The load's log names the rule
  and both settings.
- **A round under the sampled draft runs our accept for every lane.** A sampling lane takes the ratio; a
  greedy lane drafts the selector walk's draft and takes the longest prefix the target's argmax agrees
  with, the vendored greedy branch's rule, so its text does not change. A verify-only load's drafts come
  per call, and their q is one-hot.
- **Three random streams of the sampler's counter scheme join #586's repair**: the draft's draw (purpose
  17), the accept (18) and the refusal's draw (19), each keyed by the lane's seed and the draft column's
  position. The bonus token after an accepted run keeps the vendored purpose. A fixed seed reproduces a
  lane's text on one build, but its text under the sampled draft is not its text under the fixed draft:
  the distribution is the same, the draws are not.
- **The commit is unchanged**: the cut at a stop id, the reasoning redirect, a grammar's cut and repair,
  the fold, the ring bits of refused columns, the drafter window's append and the penalty rows' give-back
  read the accept's outputs as they did.
- **The drafter's q is kept from the draw to the accept**: its candidates' ids and probabilities, 896 bytes
  a lane at seven drafts, on the drafter's line of the VRAM plan.
