---
status: accepted
---

# The 27B's sampled lanes draw their drafts and take them by an accept of ignis's own

Under sampling, DFlash2 proposes one fixed draft per column and the vendored accept
(`speculative_accept_greedy_drafts`) takes it with the target's probability of that token. That is the rule of
speculative sampling for a drafter whose distribution is one-hot, and it is correct: a refused draft is replaced by a
draw from the target's distribution without it, so the text keeps the target's distribution. It is also what caps
acceptance under sampling: even a drafter that always proposed the target's most likely token would see its first
draft refused whenever the target, sampling, picks another one, 18% of the time on recorded agent turns at
temperature 1.

So the 27B's sampled lanes run the general rule, the **sampled draft**: the drafter draws each column's draft d from
its own distribution q over its candidates, and the accept takes d with probability min(1, p(d)/q(d)); on a refusal
the lane's token is drawn from max(p − q, 0), renormalized over the target's support. Whatever q is, the emitted
token is distributed as p: q moves how often a draft is taken, never what text can come out. It commits more tokens a
round than the fixed draft
([finding](../findings/2026-11-13-the-sampled-draft-served-on-the-27b.md)).

**This is ignis's own implementation of a different rule, not a patch of a vendored one.** The vendored accept is not
wrong for the rule it states (ADR 0037 admits a patch for a demonstrated bug, and there is none), and no registry
lacks a case (ADR 0088). Its file stays byte for byte as vendored (ADR 0010), and it stays in use for every round
under the fixed draft and for Flash-Next's MTP rounds. The draft's draw and the accept are kernels of the leaf's own
(`kernel/src/sampled_draft.cuh`): the arithmetic exists once as a host reference, `ignis_core::sampled_draft`, held
on every push to a double-precision oracle that sweeps the random numbers over a grid instead of sampling them, and a
GPU test holds the device kernels to the same oracle. The accept builds each column's target distribution with the
vendored sampler's device primitives, so the p it compares against is the vendored accept's to the bit; only the
verdict is ours.

## Consequences

- **The rule is a setting of the 27B's DFlash2 alone** (`spec.draft_rule`): `sampled` is the default since the
  rule's gate on the RTX 5090 passed, and `fixed`, the fixed draft, the rule of earlier releases, is one option
  away. The drafter's temperature factor and top-p are settings beside it. Where no round drafts with DFlash2
  (Flash-Next, whose MTP rounds draft the fixed draft, and the 27B with no backend, `off` or its MTP head) any value
  is ignored, as every draft setting is where its speculation does not run.
- **A round under the sampled draft runs our accept for every lane.** A sampling lane takes the ratio; a greedy lane
  drafts the selector walk's draft and takes the longest prefix the target's argmax agrees with, the vendored greedy
  branch's rule, so its text does not change.
- **Each of the rule's draws has a random stream of its own** in the sampler's counter scheme. A fixed seed
  reproduces a lane's text on one build, but its text under the sampled draft is not its text under the fixed draft:
  the distribution is the same, the draws are not.
- **The commit is unchanged**: the cut at a stop id, the reasoning redirect, a grammar's cut and repair, the fold, the
  ring bits of refused columns, the drafter window's append and the penalty rows' give-back read the accept's outputs
  as they did.
- **The rule holds at every window DFlash2 takes**: the sampled draft draws and accepts at up to 15 drafts a lane, the
  sampler's 16 columns (ADR 0104).
