---
title: "The KV cache and prompt reuse"
description: "How the KV pool, retained state and the host and disk tiers let a request skip work an earlier request did, what that saves, and when it does not apply."
sidebar:
  order: 3
---

Every request carries history: the system prompt, the tools, the earlier turns
of a conversation. Reading that history is the expensive part of a request, and
it is the same history the last request read. ignis keeps the result of that
reading and gives it to the next request whose prompt starts the same way. This
page explains what is kept, where, for how long, and what it saves.

## What a sequence is made of

A sequence (a request in flight) holds three kinds of state:

- **KV pages.** The keys and values of the full-attention layers, in fixed-size
  pages of 64 tokens ([ADR 0045](../../adr/0045-the-kv-pool-follows-residency-and-state-goes-down-to-disk.md)).
  They grow with the context.
- **Recurrent state.** Both models are hybrids: the 27B has 48
  linear-attention layers and Flash-Next 36. Each keeps a fixed-size recurrent
  state that summarizes everything read so far. On the 27B it is about 144 MiB
  per sequence, whatever the context.
- **Small extras**: the convolution history of those layers, the position, and
  the window of exact recent keys that the compressed KV format keeps.

The **KV pool** is the device memory for the pages. It is shared by all the
lanes, and a request is charged for the pages it has used, so a lane costs its
state, not a whole context. The serving KV format, `hq-e8-2b`, stores a token in
9,216 bytes on the 27B where BF16 would need 65,536, a factor of 7.11, which is
what makes long contexts and many lanes fit on one card. BF16 stays available as
the format the correctness checks run against
([ADR 0022](../../adr/0022-two-kv-formats-bf16-as-oracle.md)).

How the pool is sized is part of [the VRAM plan](vram-plan.md); its flags are on
[VRAM budget and the KV pool](../configuration/vram-and-kv-pool.md).

## Why reuse needs more than the KV pages

On a plain transformer, reusing a prefix means keeping its KV pages. These
models are hybrids: the linear-attention layers carry a state that exists only
at the positions where it was captured, and the only way to rebuild it is to run
the prefix again. Sharing pages alone would save nothing, because prefill must
still traverse every layer to produce that state.

So reuse keeps two things together: the KV pages of the prefix (shared in place,
reference counted) and an image of the mutable state at its end (copied). And it
can only resume at points where an image was taken, not at an arbitrary longest
common prefix. The server predicts those points before prefill
([ADR 0024](../../adr/0024-sequence-state-transfer.md),
[ADR 0029](../../adr/0029-cross-request-state-reuse.md)).

## What is kept

Two kinds of **retained state** outlive their request:

- **A prompt checkpoint** is the whole state of a request at its *generation
  opener*, the point where the rendered prompt hands over to the model
  (`<|im_start|>assistant`). It serves the next turn of the same conversation.
  The opener is chosen because the next turn re-renders the previous assistant
  message differently (reasoning dropped, tool arguments re-serialized), so the
  state after generation never matches; the opener is the last point every later
  turn provably shares, and costs two or three tokens to re-prefill. A
  conversation keeps at most two: its latest, and the one right after its last
  real user message.
- **A retained prefix** is a shared prefix kept alive with no request holding
  it, published at the end of the system and tools block (rounded down to whole
  KV pages). It serves the next subagent of a burst, which starts with the same
  system block as its siblings.

Both are found **by content**. A request reuses the longest retained state whose
tokens, images included, are a prefix of its own prompt. No session identifier
is sent or needed, and a stale or wrong identifier cannot hand one conversation
another's state; a content match can only hand over identical history. Reusing
copies and never consumes, so a retry, a regenerate or two forks from one point
all hit.

Each image of mutable state occupies a **retained slot**, reserved at load. By
default there are none on the device and two per decode lane in one pinned host
block. A host slot costs a PCIe copy where a device slot costs a device copy
(a claim of 182 MiB measured 15.6 ms from the host against 0.29 ms on the
device) but no VRAM, so the KV pool gets that room: at the default load it raised
the pool from about 475K to about 680K tokens at level throughput, and in an
eight-agent run 51 of 52 checkpoint hits came back from host slots
([retained slots on the host](../../findings/2026-09-28-retained-slots-on-the-host.md)).
When no slot is free, the lowest-ranked retained state gives one up; when none
can, the capture is skipped and the request runs without leaving reuse behind.

## Where it lives: three tiers

State lives in three tiers and moves down one when the tier above needs room
([ADR 0045](../../adr/0045-the-kv-pool-follows-residency-and-state-goes-down-to-disk.md)):

| Tier | Where | Held how |
|---|---|---|
| Device | VRAM | The KV pool and the device retained slots |
| KV-RAM | Pinned host memory | One arena (`--reuse-kv-host-pool-bytes`, `2G` by default) page-locked whole at start and held for the life of the load, even idle |
| KV-disk | A file per state | A byte ceiling (`--kv-disk-bytes`: `4G` on Flash-Next, off on the 27B) |

Retained state is always the first to go: on the device it never delays or
refuses an admission, and in KV-RAM it is discarded before any evicted live
sequence. State reaches KV-RAM lazily, only when the device is about to discard
it, never as a copy of what is still resident.

The tiers also hold *live* sequences. When a request ranked above another
cannot fit, the lower-ranked sequence (an `agent` one before an `interactive`
one) is moved down a tier a window at a time between decode rounds, so the other
lanes keep generating, and it resumes bit-exact where it stopped. With KV-disk
on, no evicted live sequence is discarded to make room; when every tier is full
of live work the new request waits in the queue
([KV-disk](../configuration/kv-disk.md)).

A match found in a lower tier is only worth taking if it reuses enough: each
tier below the device has a **restore floor**, the prompt tokens it must reuse
beyond the best match above it (for KV-disk, 8,192 on Flash-Next and
16,384 on the 27B), so a tie always goes to the tier above.

## What it saves, and what it costs

- **Time to first token.** A request that resumes skips prefilling the part it
  matched. At the 27B's cold rate of about 9,700 tokens per second on a 46K
  prompt, reading such a prompt cold takes roughly 4.7 seconds (arithmetic from
  the README's figure); the resumed part costs a copy instead. A
  device-to-device clone of 148 MiB of state measured 0.25 ms, and a full-context
  snapshot and restore round trip through host RAM about 90 ms, both far below
  the prefill they replace
  ([clone cost](../../findings/2026-09-12-device-prefix-clone-cost.md),
  [transfer cost](../../findings/2026-09-12-sequence-snapshot-transfer-cost.md)).
- **What the client sees.** `usage.prompt_tokens_details.cached_tokens` is the
  part of the prompt the request resumed (0 when none), and the request's `done`
  log line names the tier and the kind of reuse.
- **A tax on short cold prompts.** To be reusable, a prompt is cut at its
  publish point and again at the generation opener, so one traversal of the model
  becomes three. Each extra traversal costs a fixed ~19 ms because it re-streams
  the weights. Measured on 2026-09-18, that was 37 percent of the TTFT of a
  1,024-token cold request, and under 2 percent on a 70K prompt
  ([the finding](../../findings/2026-09-18-prompt-reuse-tax-on-short-ttft.md)).
  The bet is that a later request skips more than that.
- **A reused request may differ in the last bit.** A reused request and a cold
  prefill split their chunks differently, so they can diverge at near-ties of
  the model's own choice.

## When reuse misses

Reuse is a prefix match, so it fails whenever the prompt does not start with the
retained tokens:

- **The prompt changed before the retained point.** Anything that differs early
  moves the match point back to it, and everything after is prefilled again.
- **A volatile system prompt.** This is the common cause. If the system prompt
  contains text that changes on every request (the current time, a request or
  session id, a counter, a randomly ordered list), the prompt differs from its
  first lines. The match is a chain over the tokens from the start, so no
  retained prefix or checkpoint from an earlier request matches past the change,
  and every request prefills from there. Keep the system prompt and the tools
  block byte-stable across requests, and put anything that varies after them, in
  the last user message. Settings that re-render history when a message arrives
  mid-conversation (`--server-developer-message-policy into-system` or
  `after-system`) trade reuse for fewer system blocks for the same reason
  ([prompting](../configuration/prompting.md)).
- **The retained state is gone.** A model switch, an idle unload and a restart
  free VRAM, pinned memory and KV-RAM; KV-disk does not outlive its process. The
  first request after any of them finds nothing
  ([switching and idle unload](switching-and-idle-unload.md)).
- **No slot, or no room.** A conversation past its two checkpoints, an
  exhausted slot pool, or KV-RAM full of newer state drops the oldest and
  lowest-ranked first. `--reuse-retained-interactive-ttl` (300 s) lets an idle
  main-conversation checkpoint in KV-RAM rank as a subagent's.
- **Reuse is off.** `--reuse-prompt off` captures and reuses nothing.
- **A different artifact or KV format.** Retained state is stamped with the
  artifact's structural hash, the KV format and the drafter's presence, and a
  mismatch is refused before any byte moves.
- **An image inside the matched part.** A prefix never ends inside an image; it
  walks back to before it, and an image's identity is part of the match.

`ignis_retained_state_misses_total` counts, by tier and kind, each prefill run
(a re-queued request counts again) that found nothing retained of that kind
matching where the load could have held some, so a series that can never hit,
such as a prefix's on the disk, stays at zero
([ADR 0017](../../adr/0017-prometheus-metrics.md),
[metrics](../reference/metrics.md)).

## Who can see it

Reuse is global to the server. Anyone who can reach it could tell from the time
to first token whether a prefix was seen before; ADR 0029 accepts that side
channel while there is a single API key.
