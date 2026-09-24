# runtime 08 — DFlash2 under agent sampling: measure the acceptance gap, then close the part that can be closed

GitHub: #267

## Problem Statement

Speculative decoding is ignis's biggest decode lever. DFlash2 commits 5.43 tokens per
verify round on greedy coding prompts, but coding agents do not decode greedily. They
use the model card's thinking sampling: T = 1.0, top_p 0.95, top_k 20. Under that
sampling the committed tokens per round drop:

- 4.3–4.4 at `low` / `medium` effort;
- 3.0 at `xhigh` (the 2026-09-24 effort sweep);
- 3.34 pooled over 3,838 real qwen-code / opencode turns (ninfer request logs,
  copy-drafting finding).

At one lane, where a round costs ~16 ms whatever it commits, going from 5.4 to 3.3
tokens per round is ~40% of decode throughput.

Nobody knows yet how much of that gap is:
- **inherent**: at T = 1 the target spreads its mass, and the verify accepts a one-hot
  draft with probability p_target(draft), so even a perfect argmax drafter loses; or
- **drafter error**: the drafter's proposal is not the target's argmax, or not the
  token the target most often samples;
- **draft shape**: a single path of 7, when the selector already scores 16 candidates
  per column and the rounds that reject early waste their width.

Without that split, any drafter or verify change is a guess.

## Solution

Measure the gap on real agent turns, split it into its causes, and put a number on
each lever's ceiling. Then build the lever the numbers favour, **inside the same
ticket**, and only if its measured ceiling clears the bar.

The candidate levers, ordered by expected cost:

1. **Adaptive per-lane draft extent under sampling.** A lane whose recent rounds
   accept little verifies fewer drafts. This matters at 8 lanes, where verify width
   costs real compute; at 1 lane the round is weight-streaming bound and the width is
   nearly free.
2. **Draft selection that targets the sampled distribution.** The selector walks a
   lattice of 16 candidates per column. Its score can be taken under the request's
   temperature and top-p, not the greedy argmax, so the path maximizes the expected
   accepted length under the sampling the lane actually runs.
3. **Multi-candidate verification.** Verify more than one proposal per column: a small
   token tree, or a second path from the lattice. This is lossless and raises
   acceptance to the candidates' combined mass, but it needs per-branch GDN state (48
   recurrent layers) and tree masks in the verify graph. It is the expensive lever and
   is built only if the measurement shows its ceiling is large.

Lossy acceptance ("typical acceptance", accepting a draft the target merely finds
plausible) is not a lever: output under sampling must keep the target's distribution.

## User Stories

1. As a coding-agent user, I want sampled decoding to commit as many tokens per round as the drafter can honestly earn, so that my agent's turns finish sooner.
2. As a coding-agent user, I want the text to keep exactly the distribution the target samples, so that speed never changes what the model writes.
3. As the project owner, I want to know how much of the 5.43 → 3.3 drop is inherent to T = 1 sampling and how much is the drafter, so that I only fund work on the part that can move.
4. As the project owner, I want each lever's ceiling measured before it is built, so that an M-sized change is not built for a 2% gain.
5. As the project owner, I want the measurement on real agent turns (thinking, tool calls, edits), not only on greedy coding prompts, so that the number describes the traffic ignis serves.
6. As a maintainer, I want per-position acceptance under the real sampling configs, so that a change can be judged position by position, not only by its mean.
7. As a maintainer, I want the measurement split by output kind (thinking, tool arguments, edit/write, prose), so that a lever that helps one kind and hurts another is visible.
8. As a maintainer, I want the measurement repeatable from recorded turns with a fixed seed, so that before/after comparisons are paired.
9. As an operator, I want per-request speculative counters to keep reporting accepted-per-position, so that I can watch acceptance on my own traffic after the change.
10. As an operator, I want no new flag unless the lever needs one, so that the default stays the measured-better behaviour.
11. As a maintainer, I want the adaptive extent, if built, to keep a lane's round bit-identical to a normal round at the same extent, so that correctness rests on the existing verify path.
12. As a maintainer, I want the drafter's window to stay exactly as fed today whatever the extent, so that the #157 extent-0 class of bug cannot return.
13. As a maintainer, I want any selector change to keep the greedy path unchanged at T = 0, so that the greedy numbers and texts do not move.
14. As the project owner, I want the result recorded as a finding with the split and the ceilings, so that the next drafter work (a retrained drafter, MTP) starts from it.
15. As a coding-agent user running 8 subagents, I want the 8-lane aggregate to improve, not only one lane, so that the fleet gets faster.

## Implementation Decisions

- **Phase 1: the measurement.** It changes no serving code, reuses the existing seams,
  and runs on GPU.
  - A teacher-forced replay of ~200 recorded agent turns, drawn from the copy-drafting
    dataset (13,102 turns, already tokenized). Each turn is prefilled, then its
    recorded output is walked in verify rounds.
  - Per round, recorded at each draft position:
    - the drafter's 7 proposals and the selector's 16 candidates per column;
    - the target's full distribution at each column under the turn's sampling config;
    - whether the recorded (actually sampled) token equals the proposal;
    - p_target(proposal), p_target(argmax), and the mass of the top-k candidates.
  - Derived, per position and per output kind:
    - **inherent loss**: 1 − p_target(argmax) under the sampling config, the ceiling of
      any one-hot drafter;
    - **drafter loss**: p_target(argmax) − p_target(proposal);
    - the **multi-candidate ceiling**: expected accepted length if the top-2 / top-4
      candidates of each column were verified;
    - the **adaptive-extent ceiling**: aggregate tok/s at 8 lanes, from the measured
      width-cost curve of a verify round (width 1..8 at 8 lanes, measured once).
  - The replay goes through the verify seam that takes caller-proposed drafts, the one
    `decode_program_verify` exposes for loads without the drafter. The drafter's own
    proposals are read by running it on the same prefix. If that needs a readout, it
    is a test-only readout (the `out_span_logits` precedent), never a serving path.
- **Decision gate.** A lever is built in phase 2 only if its measured ceiling is
  **≥ +8% aggregate tok/s** at the lane count it targets (1 lane for levers 2–3, 8
  lanes for lever 1). If none clears it, the ticket closes on the finding.
- **Phase 2: the lever.** It goes where the existing seams already are:
  - **Adaptive extent:** the per-lane extent the verify round already clamps
    (`min(k, drafts, budget, capacity)`) gains one more term, the lane's recent
    acceptance. It is decided on the host per round, from counters the round already
    returns.
  - **Sampling-aware selection:** a variant of the selector's score takes the lane's
    temperature and top-p. It is exact at T = 0 by construction.
  - **Multi-candidate verify:** it needs its own spec amendment (verify graph shapes,
    per-branch GDN state, commit of the winning branch). It is only started if the
    ceiling is large and the owner agrees on the cost.
- **Vendored kernels** touched in phase 2 follow ADR 0031 (a recorded patch) or ADR 0037
  (correctness), as usual.

## Testing Decisions

- A good test observes committed text and counters, not intermediate buffers.
- **Distribution preservation:** a GPU test samples many seeded runs of a short prompt
  with and without the lever. It asserts that the empirical next-token distributions
  at a set of positions agree within a statistical bound, and that greedy (T = 0)
  output is bit-identical. Prior art: the spec-on vs spec-off equivalence tests in
  `crates/core/tests/dflash2_round_gpu.rs` (`assert_equivalent`).
- **Adaptive extent:** a CPU scheduler/mock test checks that a lane with low recent
  acceptance is given a smaller extent and that extents never exceed today's clamp.
  A GPU test checks that a forced-extent round equals a normal round at that extent,
  bit for bit. Prior art: the extent-0 tests of #157.
- **The measurement** is a finding plus a reusable driver in `.scratch/` (not a test).
  Its per-position numbers are the reference the phase-2 A/B is paired against.
- **Phase-2 A/B:** live/live per ADR 0021:
  - 1 lane and 8 lanes;
  - the effort-sweep prompts under thinking sampling, plus the replayed agent turns;
  - committed tokens per round and tok/s, before/after.

## Out of Scope

- Retraining or replacing the drafter (DFlash2 weights are the artifact's).
- MTP heads.
- Lossy acceptance rules.
- Copy drafting (measured at +0–8%, `2026-09-24-copy-drafting-on-agent-traces.md`).
- Changing agents' sampling parameters.

## Further Notes

- The verify's accept op is the vendored `speculative_accept_greedy_drafts`. In
  sampling mode it accepts draft i with probability p_i(draft_i) and resamples from the
  residual on first rejection: optimal for one-hot proposals, so the rule itself is not
  where the gap is.
- Evidence:
  - `docs/findings/2026-09-24-reasoning-effort-on-coding-tasks.md` (per-effort
    tok/round under sampling);
  - `docs/findings/2026-09-24-copy-drafting-on-agent-traces.md` (3.34 tok/round on real
    turns; DFlash2 already strong on copyable spans);
  - `docs/findings/2026-09-18-decode-round-anatomy.md` (round cost structure).
