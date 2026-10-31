---
title: "Images as evidence"
description: "The vision tower's output is kept past the request, so several questions over one screenshot encode it once."
sidebar:
  order: 12
---

An agent that looks at a screen asks more than one question about it: where is
the button, is the dialog open, what does the error say. If every question
re-encodes the screenshot, the vision tower is paid again each time. ignis keys
the encoder's output by the image's content and holds it past the request that
made it, so **four questions over one 4096x4096 screenshot encode it once**.
They finish in 13.24 s instead of 27.92 s, and in 9.49 s when the image was
already seen.

## How it works

Images arrive inline as data URIs, or as URLs the server fetches, prepares and
caches. One merged vision token covers 32x32 pixels, so a 4096x4096 image is
16,384 tokens. An image past `vision.max_tokens` is downscaled, not refused,
unless a side would fall below 32 px. The encoder's output, the embedding, is
keyed by the image's content digest and its grid and kept in a paged pool
(ADR 0035). A second request that carries the same image finds it there and
skips the tower. Image tokens enter the prompt like text, so prefix reuse also
works across turns that keep the same image.

The tower's cost is the checkpoint's own. Its attention is quadratic in image
area, and at the 4096x4096 bound attention is 88.2% of a 3.68 s encode, running
at 164.7 TFLOP/s with the device idle 0.1% of the time. That size is the
processor's own ceiling: a larger image is downscaled to the same grid, so
about 3.7 s is the worst case for one image. The only lever is sending a
smaller one. On the same cold 16,404-token request the reference engine took
6,205 ms and ignis 6,387 ms, so the tower is not where ignis is faster. What the
shared embedding removes is the repeat.

For the Playground and for agents, the same image can also go through
[`/v1/decide`](decisions.md) as the `state`, where `point` and `box` read where
something is on the screen.

![A point and a box answered in the pixels of an image](../images/playground/decide-image.png)

## The numbers

| Figure | Measured on | Source |
|---|---|---|
| 1, 2 and 4 questions over one 4096x4096 screenshot: 6.84, 13.50 and 27.92 s without sharing; 6.27, 8.46 and **13.24 s** with one encode; 4 questions on an already-seen image **9.49 s**, no encode | `/v1/decide`, server restarted before each cold row, vision on, DFlash2 7 drafts | [Shared vision embedding fan-out](../../findings/2026-09-20-shared-vision-embedding-fan-out.md) |
| Attention 3.24 s of a 3.68 s encode at 16,384 columns (88.2%), 164.7 TFLOP/s, device idle 0.1% | RTX 5090 | [Vision tower cost at width](../../findings/2026-09-21-vision-tower-cost-at-width.md) |
| Cold, same 16,404-token request: 6,205 ms in the reference engine, 6,387 ms in ignis | same card and artifact | same |
| After the tower's later rewrite: 3.67 s at 4096x4096, attention 3.24 s (88%) | RTX 5090 | [The 27B vision tower on the 5090](../../findings/2026-10-15-27b-vision-tower-on-the-5090.md) |
| The encoder's peak workspace: 908,592,896 bytes at one 16,384-token item | the 27B load | [Prefill attention without prompt planes](../../findings/2026-10-21-prefill-attention-without-prompt-planes.md) |

:::note[A fan-out figure, not a single-question speedup]
The 27.92 to 13.24 s is four questions about one image. It still pays four
attention prefills over the 16K image tokens, about 2.4 s each. One question
about one new image takes the same time as before.
:::

## See it, control it

- **Switch it on**: [`vision.enabled`](../reference/configuration.md#visionenabled)
  (off unless the profile turns it on; the `rtx5090` and `rtxpro6000` profiles
  do). Without it every `image_url` part is refused, whatever the artifact
  holds.
- **Size it**: [`vision.max_tokens`](../reference/configuration.md#visionmax_tokens),
  [`vision.embedding_pool_mib`](../reference/configuration.md#visionembedding_pool_mib)
  and, for fetched images, [`media.cache_mib`](../reference/configuration.md#mediacache_mib)
  and [`media.allow_private_network`](../reference/configuration.md#mediaallow_private_network).
- **Errors**: `vision_disabled` and `media_budget_exceeded`.
- **Logs**: the `ignis.request.done` line of a request with images reports
  `media.vision_tokens`, `media.natural_vision_tokens`, `media.encode_seconds`
  and `media.cache_hits`, whatever the question; `ignis.request.admitted`
  repeats them for a request that takes a decode lane
  ([telemetry](../guides/telemetry.md)).
- **Playground**: image input in the [chat](../playground/chat.md), and the
  [Decide tab](../playground/decide.md) with an image as the state.

## Limits

- The encode cost is not tunable, and it grows with the square of the image
  side. Send the smallest image that holds what you need to see.
- Image tokens take KV room: a 4096x4096 image is 16,384 tokens of context.
- The encoder's peak sets the workspace, so with vision on the plan gives back
  less memory than a text-only load (see [Every byte planned](every-byte-planned.md)).
- A request with images is never rewound by the loop guard.

## Read more

- [Vision](../configuration/vision.md): the flags and their defaults.
- [System One decisions, without decoding](decisions.md): `point` and `box` over an image.
- [ADR 0035](../../adr/0035-the-vision-embedding-outlives-its-encode.md): why the embedding outlives its encode.
