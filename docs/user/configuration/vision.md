---
title: "Vision"
description: "The vision tower of either text model: on, off, its token budget, and where it lives."
sidebar:
  order: 10
---

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--vision-enabled` | `IGNIS_VISION_ENABLED` | off; on for the 27B under the `rtx5090`, `rtxpro6000`, `rtx-16gb` and `rtx-24gb` profiles, off for Flash-Next under every one (from 0.11.0, `--vision-enabled` turns it on for both and `--qwen38flashnext-vision-enabled` for Flash-Next alone; `--vision-enabled false` turns it off for both; a switch carries the vision of the model it replaces, [Flash-Next](#flash-next)) | Load the vision tower and reserve its workspace. Flash-Next's tower is its vision companion's, beside the artifact ([Flash-Next](#flash-next), from 0.11.0). Without it every `image_url` part is refused with `vision_disabled`, whatever the artifact holds. |
| `--vision-max-tokens <n>` | `IGNIS_VISION_MAX_TOKENS` | `32768` with `--vision-enabled` (`4096` under the `rtx-16gb` profile) | Merged vision tokens per request, 1..1,048,576. An image whose natural size is past the cap is downscaled to fit (aspect ratio kept, one token = 32 x 32 pixels), not refused, unless its aspect ratio is too extreme for the cap (a side cannot go below 32 pixels); the request log reports both sizes, `media.vision_tokens` (admitted) and `media.natural_vision_tokens` ([telemetry](../guides/telemetry.md)). Several images whose total is past the cap are still refused with `media_budget_exceeded`. |
| `--vision-embedding-pool-mib <n>` | `IGNIS_VISION_EMBEDDING_POOL_MIB` | one envelope-wide embedding | Encoded images kept for reuse across requests, 1..65536, counted where `--vision-residency` puts the pool ([below](#where-the-vision-tower-and-the-embedding-pool-live)). |
| `--vision-residency <host\|device>` | `IGNIS_VISION_RESIDENCY` | `host`; `device` under `rtxpro6000` | Needs `--vision-enabled`. Where the vision tower's weights and the embedding pool live: `device` in VRAM, `host` in pinned host RAM, the weights copied to the GPU for each image ([below](#where-the-vision-tower-and-the-embedding-pool-live)). Read at load: a change takes effect at the next one. |
| `--media-cache-mib <n>` | `IGNIS_MEDIA_CACHE_MIB` | `1024` with `--vision-enabled` | Prepared images kept for reuse; `0` disables, max 65536. |
| `--media-allow-private-network` | `IGNIS_MEDIA_ALLOW_PRIVATE_NETWORK` | off | Needs `--vision-enabled`. Fetch image URLs on private, loopback and link-local addresses. |

## Flash-Next

From 0.11.0, Flash-Next takes images as the 27B does, through the same
requests, limits and media cache (0.10.0 refuses `--vision-enabled` on
Flash-Next, and its image parts with `vision_disabled`). Its vision tower is
not in its 72 GB container: it ships in a
**vision companion** beside it, `qwen3_8_flash_next_vision-v2.ninfer` (482 MB),
which `ignis-server model download qwen3.8-flash-next` fetches with the model.
The tower has the 27B's shape, retrained with Flash-Next, its merger writing
Flash-Next's 2,560 columns. Vision stays off on a Flash-Next start unless you
turn it on: every built-in profile turns it on for the 27B alone, so a
Flash-Next start holds nothing for it until `--vision-enabled` (or `make
MODEL=flash-next VISION=1`). A switch is not a start: it carries the vision of
the model it replaces (from 0.11.0), so a 27B serving images on its profile's
vision switches to Flash-Next with vision on, at Flash-Next's own vision
settings, unless vision is named off for Flash-Next itself
(`--qwen38flashnext-vision-enabled false`, its env var or its section of the
config file), which wins ([Model switching](../api/model-switching.md)).
The profile's residency is both models': `host` under `rtx5090`,
`device` under `rtxpro6000`. Every vision and media field also has a form for
one model alone (from 0.11.0), which needs that model's vision on:
`--qwen38flashnext-vision-enabled --qwen38flashnext-vision-residency device`,
or `vision: { qwen38flashnext: { enabled: true, residency: device } }` in a
config file. Named for a model whose vision is off, such a form is refused,
naming the model. One named for both models (`--vision-max-tokens`, `make
VISION_MAX_TOKENS=…`) is refused only when vision is off for both; otherwise
it applies to the models with vision on, which the start's
`ignis.config.vision_options` line names (`applies_to`, and `vision_off` for
the others): under a built-in profile, the 27B alone.

With vision on, a Flash-Next start opens the companion and holds it to the main
container it was built for (the pair its sidecar records), its merger's width to
the model's. A companion that is missing, or that belongs to another main
container, refuses the start before anything loads, naming the file
([troubleshooting](../troubleshooting.md#flash-next-refuses-the-start-for-its-vision-companion)),
and a switch to Flash-Next with vision on, its own or carried, with a
`409 vision_companion_missing` before the model serving is unloaded.

What it costs:

- **No VRAM of its own under `host`.** An image is encoded in memory
  Flash-Next already reserves and leaves idle while an image encodes: the
  prefill scratch it reserves for its text and, on a card that cannot hold
  every expert, the staging ring that prefill's expert misses pass through
  (1.57 GB on an RTX 5090), which holds nothing between two prefill chunks.
  The tower is copied through three 23 MB slots of them. The positions an
  image's tokens are read at ride in buffers a text load reserves too, so a
  vision load plans the same expert cache as a text-only load at the same
  settings, with the MTP head on or off. The largest image one encode holds
  is what the scratch and the ring hold together, at most the processor's
  16,384 tokens (4096 x 4096 pixels): the RTX 5090's ring alone holds that
  one's encode. On a card that holds every expert there is no ring, and the
  scratch alone bounds it. A larger image is shrunk to the bound, aspect
  kept, as `--vision-max-tokens` shrinks one, and refused only when its
  aspect ratio cannot fit. The `ignis.runtime.flash_next_plan` event names
  the bound as `vision_item_max_tokens`, beside `vision_residency`.
  Under `device` the tower (481 MB) and the embedding pool are VRAM lines:
  on a card that holds every expert they come out of the KV pool, on a 32 GB
  card out of the expert cache.
- **Host RAM under `host`**: the tower's 483 MB (482,705,408 bytes with its
  page padding) and the embedding pool, an envelope-wide image at Flash-Next's
  2,560 columns, half the 27B's: 5 MiB at a 1,024-token envelope, 20 MiB at
  4,096, 160 MiB at the default 32,768. Both are lines of Flash-Next's host
  plan ([Host RAM for Flash-Next](host-ram.md#host-ram-for-flash-next)), and a
  start short of RAM for them is refused naming them and the same two ways out
  as the 27B's.

A conversation about an image reuses it as on the 27B. The next turn resumes
from the [prompt checkpoint](../concepts/kv-cache-and-prompt-reuse.md#what-is-kept)
the turn before it left at its generation opener, past the picture, so it
neither prefills nor encodes the picture again: its
`usage.prompt_tokens_details.cached_tokens` covers the picture, and the request
log's `media.encode_seconds` is 0. Questions fanned out over one image encode
it once; the others read its encoding from the embedding pool. The positions
the picture's tokens are read at travel with the conversation's state, into a
retained slot, to KV-RAM and back.

The reasons are in
[ADR 0087](../../adr/0087-flash-nexts-vision-encoder-ships-in-a-companion-container-and-encodes-in-memory-its-prefill-already-reserves.md).

## Where the vision tower and the embedding pool live

The 27B's vision tower is 296 MB of weights (295,711,648 bytes). Its embedding
pool holds encoded images for reuse (`--vision-embedding-pool-mib`, one
envelope-wide image unless named: 10 MiB at a 1,024-token envelope, 40 MiB at
4,096, 320 MiB at the default 32,768). The vision residency
(`--vision-residency`, `vision.residency`, `IGNIS_VISION_RESIDENCY`) says where
a vision load keeps both:

- **`device`** (the `rtxpro6000` profile's): in VRAM for the life of the load,
  as every vision load before it. An image encodes at full speed, and the tower and the
  pool come out of the KV pool whether or not an image ever arrives. It is the
  choice for a card with VRAM to spare on a host short of RAM, and for the
  lowest encode latency.
- **`host`** (the default, and the `rtx5090` profile's): in pinned host RAM. The tower sits beside the token embedding
  ([Host RAM for the 27B](host-ram.md#host-ram-for-the-27b)), and for each image
  the encode copies it to the GPU a block at a time, through three 25 MB slots
  it holds in the prefill scratch, while the block before computes. The pool is
  mapped: the encode writes each page of an image's embedding on the GPU and
  copies it there, and the prefill reads the image's columns from host RAM
  where they sit. Between images neither takes VRAM: the VRAM plan's
  `weights_bytes` leaves the tower out, its `media_embedding_bytes` is 0, and
  its `workspace_bytes` grows by the slots (75 MB) only where the image
  encoder, not the prefill chunk, sets the scratch's size. Under the
  `rtx-16gb` profile at a 1,024- or 4,096-token envelope it does not grow,
  and a vision load keeps the KV pool of a text-only one. An encode takes a few
  milliseconds more: about 7 ms on a 1,024-token image (49 ms against 42 under
  `device`) on a PCIe Gen 3 host under Windows, and nothing measurable on a
  16,384-token one
  ([finding](../../findings/2026-10-30-vision-residency-acceptance.md)). A
  fan-out of questions over one image encodes it once, so only the first pays.
  At the same context the answers do not change: the image's embedding is the
  same to the bit. On a card whose start adapts its context to free VRAM, the
  KV `host` gives back can raise that context, and with it the YaRN factor the
  default `auto` [rope scaling](context-and-speculation.md) picks, which moves
  answers slightly, as any change of free VRAM does. To compare the two
  residencies, pin `--model-max-context`.

`--vision-embedding-pool-mib` keeps its meaning under both, with the same
default and floor: it sizes the pool where the residency puts it, VRAM under
`device` and pinned host RAM under `host`.

A `host` start whose host RAM cannot hold the tower and the pool beside the
host margin is refused before anything is loaded; it never moves them to the
device on its own, so the context a start gets never depends on how much RAM
happened to be free. The host plan names the tower and the pool on lines of
their own, and the message says how much is missing and the two ways out:

```
not enough RAM: the load holds … and keeps 4 GiB free for the system, so it needs … available, but
only … is (503 MiB short). The token embedding fits, the vision tower (283 MiB) and its embedding
pool (320 MiB) do not: keep them in VRAM (vision.residency: device, --vision-residency device), turn
vision off, or close other applications. Plan: token embedding 1.3 GiB, vision tower 283 MiB,
embedding pool 320 MiB, …
```

On Windows, Task Manager counts the pinned tower and pool as shared GPU
memory, as it does the token embedding: that is host RAM, not VRAM
oversubscription.

The `ignis.runtime.vram_plan` and `ignis.runtime.qwen27b_host_plan` log events
carry the load's `vision_residency` (a text-only load has none), and the host
plan event the tower's and the pool's lines as `vision_tower_bytes` and
`vision_pool_bytes`. The reasons are in
[ADR 0075](../../adr/0075-the-vision-tower-is-staged-and-the-embedding-pool-host-mapped-unless-the-load-asks-for-the-device.md).
