---
title: "State that outlives its request"
description: "A repeated prompt resumes from a copy held in VRAM, pinned RAM or on disk, and the response says how many tokens it reused."
sidebar:
  order: 6
---

An agent re-sends a long prompt on every turn, and its subagents start from
the same system block and tool list. Prefilling that again is the most
expensive thing a card can do twice. ignis keeps the state of a finished
request, **found by content**, and a repeated prompt costs a copy instead of a
prefill. What was reused is reported in the response.

## How it works

Two kinds of state outlive a request. A *prompt checkpoint* is the whole
sequence state (KV pages, recurrent state, convolution taps, drafter window)
captured at the generation opener, the last point every later turn of a
conversation provably shares; a conversation keeps at most two. A *retained
prefix* is the shared head, the system and tools block rounded down to whole
pages, published for the next subagent in a burst. Both are found by content:
the longest retained state whose tokens, images included, are a prefix of the
request. The key is a hash chain over those tokens rather than a digest
of the whole, because a claimant asks about every prefix length of its
prompt at once, and because it lets a blob name its content in sixteen
bytes instead of carrying sixty thousand token ids to another tier. No
session id is sent or needed, and a wrong id cannot hand over
another conversation's state. Reuse copies and never consumes, so a retry, a
regenerate or two forks all hit.

The copy is cheap because of how it is made. A device-to-device clone moves
the 144 MiB recurrent slot plus the prefix KV in three strided copies (96
per-layer copies were six times slower) and takes 0.25 ms. Each piece of retained
state occupies a slot reserved at load. By default the slots sit in one pinned host
block, two per decode lane on the 27B: a claim then costs a PCIe copy but no
VRAM, so the KV pool is larger.

Under pressure state moves down: device, then KV-RAM (a pinned arena), then
KV-disk (one file per state). One eviction order spans the tiers: class first,
`agent` before `interactive`, then protection and recency
([ADR 0023](../../adr/0023-unified-eviction-priority.md)). A sequence moves
only when a request ranked above it needs the room, a window at a time between
decode rounds, and resumes bit-exact where it stopped. A match in a lower tier
must beat a restore floor, so ties go up.

The tiers, the matching rule and what it costs, in full:
[KV cache and prompt reuse](../concepts/kv-cache-and-prompt-reuse.md).

## The numbers

| Figure | Measured on | Source |
|---|---|---|
| Prefix clone **0.25 ms** for 147.76 MiB (620 GB/s) | RTX 5090, real geometry | [Device prefix clone cost](../../findings/2026-09-12-device-prefix-clone-cost.md) |
| 96 times cheaper than a PCIe round trip (about 24 ms); thousands of times cheaper than re-prefilling a 20,480-token head (at least 2.0 s) | PCIe Gen 3 x16 host | same |
| Full-context snapshot and restore about 90 ms round trip (42 ms each way), about 100 times cheaper than a 40,960-token re-prefill | PCIe Gen 3 host | [Snapshot transfer cost](../../findings/2026-09-12-sequence-snapshot-transfer-cost.md) |
| Host retained slots: KV pool 475K up to 677-682K tokens (+42-44%) at level throughput; later-turn TTFT p50 on the long load 2.21 s up to 2.82 s | eight-agent swarm, 262,144-context load, prototype build of 2026-09-29 | [Retained slots on the host](../../findings/2026-09-28-retained-slots-on-the-host.md) |
| A host claim costs 15.6 ms for 182 MiB against 0.29 ms on the device | same | same |
| KV-disk: a spill of about 1 GB takes about 0.63 s; two decoding Flash-Next lanes keep ITL p50 within +3% to +8% | Flash-Next, NVMe volume, four runs | [KV-disk contention](../../findings/2026-10-08-kv-disk-contention-on-the-volume.md) |
| The tax: a reusable prompt is cut at its publish point and at the opener, three traversals at about 19 ms each extra; 37% of a 1,024-token cold TTFT, under 2% at 70K. On an RTX 30 or RTX 40 card the 27B is cut once, at the opener ([what it costs](../concepts/kv-cache-and-prompt-reuse.md#what-it-saves-and-what-it-costs)) | 27B, RTX 5090 | [Prompt reuse tax](../../findings/2026-09-18-prompt-reuse-tax-on-short-ttft.md) |

:::caution[Dated numbers]
The 475K to about 680K pool is a 262,144-context load of 2026-09-29. The
current 5090 profile loads a 944,832-token pool at 917,504 tokens of context,
so read the +42-44% as what host slots buy, not as today's pool size. The
PCIe ratios are for a Gen 3 host; the device clone does not cross PCIe.
:::

## See it, control it

- **Reuse**: [`reuse.prompt`](../reference/configuration.md#reuseprompt) (on by
  default), [`reuse.retained_device`](../reference/configuration.md#reuseretained_device),
  [`reuse.retained_host`](../reference/configuration.md#reuseretained_host) and
  [`reuse.retained_interactive_ttl`](../reference/configuration.md#reuseretained_interactive_ttl)
  (300 s).
- **The tiers**: [`reuse.kv_host_pool_bytes`](../reference/configuration.md#reusekv_host_pool_bytes)
  for KV-RAM, [`kv_disk.bytes`](../reference/configuration.md#kv_diskbytes) and
  [`kv_disk.path`](../reference/configuration.md#kv_diskpath) for the disk.
- **In the response**: `usage.prompt_tokens_details.cached_tokens`, and
  `input_tokens_details.cached_tokens` on `/v1/responses`. The
  `ignis.request.done` line names the tier and kind of reuse;
  `ignis.request.spilled`, `ignis.request.restored` and `ignis.request.evicted`
  mark the moves.
- **Metrics**: `ignis_retained_state_hits_total` and `_misses_total`,
  `ignis_retained_reused_tokens_total` (by tier), `ignis_prefix_reused_tokens_total`,
  `ignis_kv_ram_arena_bytes`, `ignis_kv_disk_bytes`
  ([metrics](../reference/metrics.md)).
- **Playground**: the [Monitor](../playground/monitor.md) shows the pool and
  the retained state; a prewarm over the
  [Responses WebSocket](responses-websocket.md) keeps its state too.

## Limits

- KV-disk is on by default only for Flash-Next. On the 27B it is off.
- Reuse is content-matched on a prefix. A change near the start of the prompt,
  in the system block, is a miss.
- A reused request and a cold prefill split their chunks differently and may
  diverge at near-ties.
- With no free retained slot the capture is skipped
  (`ignis_retained_slot_skips_total`), never allocated.
- KV-RAM is page-locked RAM, held for the life of the load even when idle.

## Read more

- [KV cache and prompt reuse](../concepts/kv-cache-and-prompt-reuse.md): what is kept, the three tiers, when reuse misses.
- [KV-disk](../configuration/kv-disk.md) and [host RAM](../configuration/host-ram.md): sizing the lower tiers.
- [KV at 9 KB a token](kv-in-bytes.md): the pages this state is made of.
- [Lanes that know who is asking](lanes.md): who loses residency first.
