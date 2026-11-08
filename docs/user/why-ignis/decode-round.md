---
title: "One round for every lane"
description: "Eight lanes share one pass over the weights, replayed from a CUDA graph per batch width, so a decode round runs at the card's memory bandwidth."
sidebar:
  order: 2
---

A main agent and its subagents decode at the same time, and each of them needs
the whole model for every token. ignis does not run the model once per lane.
It runs it **once per round for all the lanes together**: the lanes are the
columns of one batch, so eight lanes stream the weights one time, not eight.
On the 27B that stream is the cost of a round, and it runs close to what the
card's memory can deliver. Eight lanes at once reach
**1,065 tok/s in aggregate**, against 367 tok/s for one.

## How it works

A decode round is one call per operator per layer at the current width. Each
lane's token, position and KV slot are read from staging vectors that live on
the device, indexed by row, so a lane's output is a property of its own row.
Sampling is on the device and per lane, so a request's result depends on its
own seed and not on which lanes shared its round. The earlier design ran one
forward pass per lane in program order and paid the weight stream per lane: at
four lanes a round cost 4.81 times its own one-lane round, where the reference
engine paid 1.07 times ([ADR 0020](../../adr/0020-batch-wide-decode-round.md)).

The round is replayed from a CUDA graph. One graph is captured per exact batch
width, 1 to 8, at startup, and a narrow round is never padded to a wider graph,
because that would move state the round does not need
([ADR 0019](../../adr/0019-decode-cuda-graph-slot-indirection.md)). The profile
of a one-lane round with speculation on says what is left: 1,166 graph nodes
and 15.81 ms of device time, and that time is weight streaming. The NVFP4
backbone GEMM takes 9.37 ms against a 9.4 ms bound at the card's bandwidth; the
INT8 output head, which runs twice per round (drafter and verify), is at 86%
of its bound. Submitting the graph costs about 370 us, 2.2% of the round. The
device is idle 4.8% of the time, and that is structural: two host
synchronisations per verify round, not any one kernel.

Prefill and decode share one model stream and take turns. The scheduler hands
the engine one prefill chunk, then runs a decode round for the lanes already
generating. Before chunking, a 32K prompt stopped every decoding lane for the
length of the whole prefill, a gap of about 3.8 s
([ADR 0018](../../adr/0018-chunk-level-prefill-decode-interleaving.md)). Now a
lane pauses for one chunk at a time, and `model.decode_share` sets how much of
the card the decoding lanes keep while a prompt prefills.

Prefill itself reads a cold 46K-token prompt at **11,344 tok/s** and a 105K one at
9,008 on the 5090. Since 0.7.3 its attention reads the compressed KV directly
instead of expanding it into BF16 first, which made it 30% faster at 46K and
53% at 105K, and the `rtx5090` profile prefills 2,048 tokens a chunk. ignis's
own gated delta net prefill kernels, Blackwell's route since they measured
faster there, add 4.8% at 46K and 3.9% at 105K against the vendored ones.

The path of a request through the scheduler, step by step:
[How ignis serves a request](../concepts/how-ignis-serves.md).

![Three subagents decoding at once in the Playground, each at its own rate, the header at the server's aggregate](../images/playground/agents.png)

## The numbers

| Figure | Measured on | Source |
|---|---|---|
| 1,166 graph nodes, **15.81 ms** device time per round | one lane, DFlash2 with 7 drafts, RTX 5090 | [Decode round anatomy](../../findings/2026-09-18-decode-round-anatomy.md) |
| Backbone GEMM 9.37 ms against a 9.4 ms bound; output head 1.66 ms against 1.42 ms (86%) | same; weights 18.29 GB, head 1.27 GB | same |
| Graph submission about 370 us, 2.2% of the round; device idle 4.8% (2.27% at eight lanes) | one lane | same |
| Round **32.18 ms** at eight lanes | eight lanes, 8 s window | [Top-k, one warp per column](../../findings/2026-09-18-dflash2-topk-one-warp-per-column.md) |
| **367 tok/s** one lane, **1,065 tok/s** the same eight prompts at once | eight coding prompts, greedy, 512 tokens, 262,144 context, DFlash2 with 7 drafts, no vision | [Quick wins A/B](../../findings/2026-09-24-upstream-quick-wins-ab.md) |
| Cold prefill **11,344 tok/s** at 46K, 9,008 at 105K; ignis's GDN prefill +4.8% and +3.9% of it against the vendored kernels, one session | the 27B NVFP4, the `rtx5090` profile (2,048-token chunk), RTX 5090 | [The GDN prefill on Blackwell](../../findings/2026-11-08-blackwell-gdn-prefill-on-ignis-kernels.md) |
| The 2,048-token chunk: +1.9% at 46K, +2.2% at 105K against 1,024 | the 27B, RTX 5090 | [The 27B's prefill chunk at 2,048](../../findings/2026-10-21-the-27b-prefill-chunk-at-2048.md) |
| Prefill attention on the compressed KV: +30% at 46K, +53% at 105K against 0.7.2's kernels | one A/B session, RTX 5090 | [Prefill attention without prompt planes](../../findings/2026-10-21-prefill-attention-without-prompt-planes.md) |
| Per-lane design at four lanes: round 70.2 ms against 17.3 ms (4.06x) | live/live gate, 32K context | [ADR 0020](../../adr/0020-batch-wide-decode-round.md) |
| Flash-Next lanes keep 15.6-18.3 tok/s while a prompt prefills at decode share 25; the prefilling request's TTFT grows by 1.33x | Flash-Next, 8,192-token prompt, RTX 5090 | [Prefill chunk and decode share](../../findings/2026-10-07-flash-next-prefill-chunk-and-decode-share.md) |

:::caution[Which numbers are which]
The 1,166 nodes and 15.81 ms are a **one-lane** round with speculation on; the
eight-lane round is 32.18 ms. "100% of the roofline" is a fit good to about
±10%, because the weight accounting is derived. The 367 and 1,065 tok/s are
greedy coding prompts, the most predictable traffic there is; sampled agent
turns accept fewer drafted tokens, see [a drafter inside the
artifact](dflash2.md).
:::

## See it, control it

- **Lanes**: the 27B serves a fixed 8. On Flash-Next,
  [`spec.decode_lanes`](../reference/configuration.md#specdecode_lanes) sets
  them (default 3).
- **Chunk and share**: [`model.prefill_chunk`](../reference/configuration.md#modelprefill_chunk)
  is the span a prefill takes per turn;
  [`model.decode_share`](../reference/configuration.md#modeldecode_share) is
  the percent of time the decoding lanes keep (25 by default on both models).
- **Metrics**: `ignis_scheduler_requests`, `ignis_decoded_tokens_total`,
  `ignis_request_ttft_seconds` ([metrics](../reference/metrics.md)).
- **Log**: the DEBUG event `ignis.scheduler.interval` carries the scheduler's
  counters.
- **Playground**: the [Monitor](../playground/monitor.md) draws the live
  scheduler state, and the engine timings sit beside every reply in
  [Chat](../playground/chat.md).

## Limits

- Prefill and decode **do not overlap** on the card; they alternate at chunk
  granularity. True overlap was left out of scope on grounds of risk.
- One prompt prefills at a time. The others wait their turn.
- A prompt decoded at eight lanes is not always token-identical to the same
  prompt at one lane: the round's width changes the arithmetic, and a near-tie
  can fall the other way ([batched decode width drift](../../findings/2026-09-14-batched-decode-width-drift.md)).
- "At the bound" is about kernels. The 4.8% idle would need accept and fold
  logic that lives on the device, which is not built.

## Read more

- [How ignis serves a request](../concepts/how-ignis-serves.md): prefill and decode in the request's path.
- [Speculative decoding](../concepts/speculative-decoding.md): what one round does when it drafts.
- [A drafter inside the artifact](dflash2.md): how the round gets several tokens per lane.
