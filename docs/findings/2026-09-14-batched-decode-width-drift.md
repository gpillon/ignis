# Batched decode width drift

- Kind: experiment
- Status: current
- Observed: 2026-09-14
- Last verified: 2026-09-14
- Scope: kernel / window-0 batched decode, batch invariance
- Related: https://github.com/gpillon/ignis/issues/155, https://github.com/gpillon/ignis/issues/153, https://github.com/gpillon/ignis/issues/158
- Superseded by: none

## Question

While #155's drafter was being brought up, spec-on and spec-off diverged at
width 4 on a canary. Which route actually moved: the drafter, the verify
round, or the spec-off baseline?

## Evidence

Setup: qwen3.8-27B NVFP4 artifact, BF16 KV, greedy, a plain load with no
speculation. Canary 0 (`In one sentence, what does fn main() ... do?`) was
wrapped in a user turn with thinking closed, then decoded 8 tokens with
`decode_program_batch_sampled` in four batches:

| Batch | Lane 0 stream |
|---|---|
| alone (M=1) | `2064, 18091, 279, 4202, 1406, 314, 264, 32671` |
| canary 0 twice (M=2) | `2064, 18091, 279, 4202, 1406, 314, 264, 32671` |
| canary 0 four times (M=4) | `2064, 18091, 279, 1957, 579, 4202, 1406, 421` |
| canaries 0-3 (M=4) | `2064, 18091, 279, 1957, 579, 4202, 1406, 421` |

The same position, prompt plus `2064, 18091, 279`, was probed through both
prefill routes. `logit[1957] - logit[4202]` is -0.75 on the chunked route
and -0.875 on the per-token route.

The DFlash2 verify round at widths 1, 4 and 8 picks 4202. So do the
VerifyOnly round at extent 0 and the same round with oracle drafts.

## Finding

Observed:

- Plain batched decode at M=4 picks 1957.
- At M=1 and M=2 it picks 4202, as do both prefill routes and every verify
  round.
- Both prefill routes rate 4202 ahead by 0.75-0.875 logits.
- The outcome depends only on the batch width: four identical prompts
  drift the same way as four different ones.

Inferred: the one-column decode's tiling at M=4 moves this logit pair by
more than the prefill routes' spread. The spec-off stream is itself one
route and cannot be treated as ground truth at such positions. This is not
a #155 defect: #155 does not touch the window-0 decode path.

## Implications

- Measure an equivalence test against spec-off at width > 2 with a rule that
  tolerates spec-off's own drift. `crates/core/tests/support/near_tie.rs`
  now also passes a divergence where both prefill routes side with spec-on.
- AC1 / G5 wording ("token-for-token") needs the owner's decision (#153).

## Limits and unknowns

- Seen at one position of one prompt. The size of the drift across prompts
  and widths 3, 5-8 is not measured.
- The kernel or tile responsible is not isolated.
- hq KV formats were not tested.

## Follow-ups

- Locate the width-dependent decode tile and decide whether batch invariance
  is required (owner).
