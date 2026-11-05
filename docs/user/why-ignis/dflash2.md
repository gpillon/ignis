---
title: "A drafter inside the artifact"
description: "DFlash2 proposes up to seven tokens a round and the model verifies them in the same pass, so the output stays the model's own."
sidebar:
  order: 3
---

Decoding one token at a time leaves the card waiting on the weight stream. The
27B artifact carries a small drafter, **DFlash2**, grafted in beside the
weights. After every token it proposes up to seven more; the model checks them
all in one pass and keeps the ones it would have chosen itself. On coding
prompts that commits **5.43 tokens a round at one lane**, and one lane decodes
at **367 tok/s**. Because the model decides, the output is the model's own, not
the drafter's.

## How it works

DFlash2 is a five-layer drafter with a 2,048-position sliding window. It reads
the target's own hidden states, tapped during prefill and verify, and proposes
up to `spec.draft_tokens` tokens after each lane's anchor. A verify round runs
every lane's anchor plus its drafts through the model once. Each lane commits
the longest run the model itself would have picked, plus the model's own next
token. Attention layers keep every column's keys; the recurrent layers record
per-column state and replay only the committed prefix, so a rejected draft
leaves no trace. Greedy output equals non-speculative output up to near-ties,
and sampling keeps its distribution.

The drafter proposes with the target's own output head by default. The
artifact also carries a Q4 "shortlist" head over the 131,072 most frequent
tokens (about 356 MB). It was measured, lost acceptance and was slower, so it
is opt-in: 5.03 committed tokens a round against 5.43, geometric mean -5.5%.

The drafter's speed depended on one kernel. The reference engine's top-k over
the drafter's 248,046-token logits gave one warp to each of 7 columns and took
3,145 us a round, 16.4% of decode kernel time, to move 3.47 MB that the card
moves in 2 us. The replacement splits the row axis across blocks and holds k
in registers: 44.26 us, 71 times faster. It selects bit-identically (largest
first, ties to the smaller row id), so acceptance does not move, and the faster
round is faster tokens. That replacement is allowed by a narrow rule: a
vendored kernel may be replaced only when it is measured to be the bottleneck
([ADR 0031](../../adr/0031-vendored-kernel-bottleneck-exemption.md)), and the
kernel's provenance is written down
([ADR 0051](../../adr/0051-kernel-code-is-ours-adapted-or-vendored.md)).

The full round, draft by draft:
[Speculative decoding](../concepts/speculative-decoding.md).

## The numbers

| Figure | Measured on | Source |
|---|---|---|
| Top-k 3,145.23 us down to 44.26 us (71x); 16.4% down to 0.27% of decode kernel time | RTX 5090, one lane, 7 drafts | [Top-k, one warp per column](../../findings/2026-09-18-dflash2-topk-one-warp-per-column.md) |
| Round 19.18 ms down to 15.81 ms; rounds in 8 s 388 up to 471 (**+21.4%**) | one lane | same |
| At eight lanes the vendored kernel already gets 56 columns: round 33.54 down to 32.18 ms (−4.1%), +3.3% rounds | eight lanes | same |
| **367 tok/s** one lane, geometric mean over eight coding prompts; **1,065 tok/s** the same eight at once | greedy, 512 tokens, 262,144 context, no vision, full proposal head | [Quick wins A/B](../../findings/2026-09-24-upstream-quick-wins-ab.md) |
| 5.43 committed tokens a round at one lane (full head) | same | same |
| 491 tok/s on predictable code, 167 tok/s on free prose, one lane, greedy | retained-slots build | [Retained slots on the host](../../findings/2026-09-28-retained-slots-on-the-host.md) |
| **3.46 tokens a round** on 200 recorded agent turns | teacher-forced replay, T=1.0, top_p 0.95, top_k 20, 7 drafts | [Sampled acceptance](../../findings/2026-09-24-dflash2-sampled-acceptance.md) |

:::caution[Greedy coding is the best case]
The 367 and 1,065 tok/s are greedy coding numbers. Agent traffic samples, and
under sampling a draft is accepted less often: on recorded agent turns the
round commits about 3.5 tokens, against about 5.4 on greedy coding prompts.
The finding splits the loss from a perfect 8-token run into 3.08 tokens to
one-hot drafting under sampling and 1.47 to the drafter. Plan for the lower
figure when you size an agent workload.
:::

## See it, control it

- **The backend**: [`spec.backend`](../reference/configuration.md#specbackend)
  (`dflash2`, `mtp` or `off`) and
  [`spec.draft_tokens`](../reference/configuration.md#specdraft_tokens), 1 to 7
  and required with `dflash2`. The built-in `rtx5090`, `rtxpro6000`,
  `rtx-16gb` and `rtx-24gb` profiles set `dflash2` at 7.
- **The proposal head**: [`spec.draft_head`](../reference/configuration.md#specdraft_head),
  `full` or `shortlist`. The `ignis.runtime.draft_head` log line prints the
  choice at load.
- **Metrics**: `ignis_speculative_rounds_total`,
  `ignis_speculative_drafted_tokens_total`,
  `ignis_speculative_accepted_tokens_total`, and the per-position pair
  `ignis_speculative_position_drafted_total` and
  `ignis_speculative_position_accepted_total` ([metrics](../reference/metrics.md)).
  They appear after the first speculative request; accepted over rounds is
  your live tokens-per-round.
- **Playground**: the [Monitor](../playground/monitor.md) and the timings
  beside each reply in [Chat](../playground/chat.md).

## Limits

- Acceptance follows predictability. Free prose and sampled turns accept fewer
  drafts than code at temperature zero.
- Speculation costs VRAM only when it is on, and it needs a backend:
  `spec.draft_tokens` alone is refused.
- The kernel replacement's gain is a one-lane figure. It shrinks to 4% at
  eight lanes.
- The replacement is not at its own bound either: the finding measures it at
  22 times its 2 us floor.
- The abliterated 27B uses the same drafter, fitted to the standard weights; a
  mismatch only lowers acceptance.

## Read more

- [Speculative decoding](../concepts/speculative-decoding.md): one round, DFlash2, MTP, the row budget.
- [Context and speculation](../configuration/context-and-speculation.md): the knobs in use.
- [One round for every lane](decode-round.md): the round the drafts ride in.
- [A 125B MoE on a 32 GB card](flash-next.md): the same idea with an MTP head.
