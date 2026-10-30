---
title: "Models and artifacts"
description: "What a .ninfer artifact is, the model families ignis serves, how a served id maps to a catalog entry, and how each family uses VRAM and host RAM."
sidebar:
  order: 2
---

A model reaches ignis as an **artifact**, is identified to clients by a
**served id**, and belongs to a **family** that decides how it runs. This page
explains those three words and what differs between the families in memory.
Fetching and pointing at a model is covered in
[Get a model](../guides/models.md).

## The artifact

An artifact is a `.ninfer` container: the weights, the tokenizer and the chat
template in one file, plus, for the 27B, a grafted DFlash2 drafter module
([ADR 0002](../../adr/0002-load-ninfer-artifact.md)). It is loaded as it is
stored; the kernels read the weight planes directly, with no conversion on the
host at load.

Next to it sits a **provenance sidecar**, `<artifact>.graft.json` or
`<artifact>.conversion.json`, which records how the container was produced.
The load verifies the artifact against it, and a mismatch refuses the start.
The catalog pins the same files by byte count and SHA-256, so a download is
checked before it is renamed into place.

A few facts about an artifact are read from the file, never configured:

- **The family.** The container's own header says which architecture it runs
  as; no flag can override it.
- **The body format.** How the transformer layers are stored. The 27B in the
  built-in catalog is NVFP4 (4-bit floating point with a scale per 16 values)
  with a few BF16 exception tensors; Flash-Next's experts are stored in a
  trellis code at 2 to 4 bits per expert projection, 2.5 on average.
- **The identity of a retained state.** Retained prompt state carries the
  artifact's structural hash, so state captured under one artifact is never
  written into another
  ([the KV cache and prompt reuse](kv-cache-and-prompt-reuse.md)).

Flash-Next has a second container, the **MTP companion** (about 1 GB), that
holds only the multi-token-prediction head. It records which main container it
belongs to, and a mismatched pair is refused by name. It is bound only when MTP
speculation is on ([speculative decoding](speculative-decoding.md)).

## The families

| Family | What it is | Artifact | Where it runs |
|---|---|---|---|
| Qwen3.8-27B | Dense, 64 layers: 16 full-attention layers and 48 linear-attention (GDN) layers | 18.1 GiB | Entirely in VRAM |
| Qwen3.8-Flash-Next | 125B mixture of experts, 6B active per token, 48 layers, 512 routed experts per layer, 10 chosen per token | 67.8 GiB with the MTP companion | Trunk in VRAM; experts in a VRAM cache backed by host RAM |
| Qwen-Image-2.1-Turbo | Image model: a Qwen3-VL text encoder, a 7B diffusion transformer and a VAE | 15.6 GiB | Reached by a switch; never loaded beside a text model |

The family is not the same thing as the model name. Several artifacts can
share one family: the abliterated 27B below is a second artifact of the 27B's
family, loaded by the same program, with the same drafter.

### Qwen3.8-27B

The 27B is the default. Its 16 attention layers keep a KV cache that grows
with the context; its 48 linear-attention layers keep a fixed-size recurrent
state per sequence instead (about 144 MiB). That is why a decode lane costs its
state rather than a whole context, and why the KV pool can be shared by lanes
([the KV cache](kv-cache-and-prompt-reuse.md)).

It serves a fixed eight decode lanes, takes image input when
`--vision-enabled` is set, and drafts with DFlash2.

Memory:

- **VRAM**: the weights, the lanes' state, workspaces and graphs, and then a
  KV pool that takes what is left ([the VRAM plan](vram-plan.md)). The vision
  tower and its embedding pool are there only under `--vision-residency device`.
- **Host RAM**: the token embedding (1.35 GB) is held in pinned host memory
  rather than VRAM, and the card reads one row per token over PCIe. This hands
  the 1.35 GB to the KV pool ([ADR 0062](../../adr/0062-the-token-embedding-lives-in-host-memory.md)).
  With vision on, under the default `host` residency, the vision tower
  (297 MB) and the embedding pool are there too, and each image's encode
  copies the tower to the card
  ([where they live](../configuration/vision.md#where-the-vision-tower-and-the-embedding-pool-live)).
  With the host retained slots and the KV-RAM arena, the `make` defaults on a
  32 GB card need about 18.6 GB of available RAM
  ([host RAM](../configuration/host-ram.md)).

### Qwen3.8-Flash-Next

Flash-Next is a mixture of experts: each layer holds 512 small MLPs and a
router picks ten per token. Only 6B parameters work on a given token, which is
why a 125B model decodes at about 100 tokens per second at one lane on a
32 GB card on a PCIe 3.0 host (README, measured on a 5090). The cost is that
the experts are far too large for VRAM at that size, so where they live decides
the machine it needs.

- **On a 32 GB card** (an RTX 5090) the experts sit in pinned host RAM, about
  38 GB of them, and a VRAM **expert cache** holds the ones in use, replaced
  least-recently-used. A token whose experts are not in the cache waits for
  them to be copied over PCIe, so decode on this card is bound by the link.
  The cache is what the KV pool leaves; below 12 GiB the start is refused,
  because decode slows sharply there. The KV pool is reserved first
  (524,288 tokens shared by the lanes by default) and the cache takes the rest
  ([the VRAM plan](vram-plan.md)). Serving needs about 46-47 GB of available
  RAM, and `--vram-pinned-expert-bytes` can keep the calibration-hottest experts
  in VRAM for the load's life to give back some of it
  ([host RAM](../configuration/host-ram.md#host-ram-for-flash-next)).
- **On a card that holds every expert** (around 96 GB) the experts are read from
  the artifact straight into their VRAM slots and never again copied; no
  host expert pool exists and the load needs about 9 GB of available RAM
  ([ADR 0054](../../adr/0054-a-resident-load-keeps-its-experts-only-in-vram.md)).
- **The n-gram table**: Flash-Next also has a hashed n-gram embedding table of
  28.8 GB. It is never put on the device: rows are read from NVMe through a
  RAM cache of hot rows (1 GiB by default), or the whole table is held in RAM
  when `--ngram-hot-bytes` covers it ([n-gram cache](../configuration/ngram-cache.md)).

Because the two layouts differ so much, a few knobs exist for one family only.
A field only one family takes is refused at start when set for the other, and
dropped with a log line when a switch moves to the other family
([configuration](../configuration/index.md)).

### The image model

Qwen-Image-2.1-Turbo is served on OpenAI's `/v1/images/generations` and
`/v1/images/edits` routes. It is a different kind of model: it produces pixels
rather than tokens, so lanes, KV and generation caps do not apply to it. It is
reached by a switch from a text model, and the VRAM its load leaves after the
weights sets a **condition budget**, the number of prompt and input-image
tokens an edit can hold; larger inputs are scaled down rather than refused.
Its weights are under the Qwen Research License, for non-commercial use
([images](../api/images.md),
[ADR 0050](../../adr/0050-ignis-serves-an-image-model.md)).

## The served id and the catalog

The **served id** is the name a client puts in `model` and `GET /v1/models`
reports. It tells artifacts of one family apart: `qwen3.8-27b` and
`qwen3.8-27b-abliterated` are the same family under two ids. It is the
operator's `--model-id`, or the family's default when none is named.

The **catalog** is the set of models the server knows how to fetch
([ADR 0047](../../adr/0047-a-catalog-the-operator-can-extend.md)). Ten entries
are built into the binary, each pinned file by file at a fixed revision of its
Hugging Face repository: six text and image models, `qwen3.8-27b`,
`qwen3.8-27b-16g`, `qwen3.8-27b-abliterated`, `qwen3.8-27b-abliterated-16g`,
`qwen3.8-flash-next` and `ignis-image-2.1-turbo`, plus three speech models and
a diarizer. Each of the six states the smallest **card class** it loads
on: 32 GB for all but the two `-16g` entries, which load on 16 GB and answer
there to `qwen3.8-27b` and to `qwen3.8-27b-abliterated`
([ADR 0064](../../adr/0064-a-catalog-entry-states-its-card-class.md),
[ADR 0074](../../adr/0074-a-catalog-entry-may-state-its-family-and-an-alias-may-be-another-entrys-id.md))
([the catalog](../guides/models.md#the-catalog)). An operator can add entries
with a catalog file of the same format, but cannot re-pin a built-in id; a
variant of a built-in model is a new id
([operator catalog](../guides/operator-catalog.md)).

A request may name any **servable** model: the loaded one, plus every model
whose artifact is on disk. Naming one other than the loaded model switches the
server to it ([model switching](switching-and-idle-unload.md)). A switch never
starts a download; fetching is a separate step (`ignis-server model download`,
or **Download** on the Playground's [Server view](../playground/server.md#downloading-a-model)).

## The abliterated variant

`qwen3.8-27b-abliterated` is the 27B with a published abliteration applied. It
is the same container: 1,255 of its 1,325 objects are byte-identical to the
default artifact, and only the 70 matrices the abliteration changed are
re-encoded. Memory use is therefore the same as the default 27B. On a 16 GB
card it is served by `qwen3.8-27b-abliterated-16g`, the 16 GB artifact with the
same 70 matrices re-quantized. It does not
refuse requests, so guardrails belong in the layer that executes tools. The
DFlash2 drafter loads unchanged; it was fitted to the default weights, and
because the target verifies every draft, a mismatch can only lower the
acceptance rate ([the uncensored variant](../guides/uncensored.md)).

## What differs, side by side

| | Qwen3.8-27B | Flash-Next on a 32 GB card | Flash-Next on 96 GB |
|---|---|---|---|
| Weights in VRAM | All but the token embedding; the vision tower only under `device` | Trunk, plus an expert cache | Trunk and every expert |
| Weights in host RAM | Token embedding (1.35 GB); the vision tower (297 MB) under `host` | Experts (~38 GB), n-gram rows | No expert pool |
| The KV pool is | What VRAM leaves | Reserved first, 524,288 tokens | What VRAM leaves |
| Decode is bound by | Streaming the weights | Copying experts over PCIe | Few PCIe copies |
| Available RAM to start | ~18.6 GB under `make` defaults on a 32 GB card | ~46-47 GB | ~9 GB |
| Decode lanes | 1 to 8 (default 8) | 1 to 8 (default 3) | 1 to 8 |

The image model has none of these rows: lanes, token decode and generation caps
are text-model terms and do not apply to it.
