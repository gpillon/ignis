---
title: "Glossary"
description: "The terms the ignis guide uses, in alphabetical order, each with a short definition and a link to the page that explains it."
sidebar:
  order: 90
---

**Abliterated variant.** `qwen3.8-27b-abliterated`, the 27B with a published
abliteration applied: the same container with 70 matrices re-encoded, it does not
refuse requests. On a 16 GB card `qwen3.8-27b-abliterated-16g`, the 16 GB
artifact with the same 70 matrices re-quantized, serves it. See [models and artifacts](../concepts/models-and-artifacts.md)
and [the uncensored variant](../guides/uncensored.md).

**Acceptance.** How many of a round's drafted tokens the model agrees with, and so
how many tokens a speculative round yields. See
[speculative decoding](../concepts/speculative-decoding.md).

**Adaptation.** A 27B load lowering the values its profile or the built-in
defaults gave until its VRAM plan fits: the drafter's head, then the context
in steps (65,536 tokens; 16,384 under `rtx-12gb`), then the decode lanes. A
value you name never adapts, and the plan event's `adapted` lists every step
taken. See [profiles](../concepts/profiles.md#when-a-profile-does-not-fit).

**Admission.** The step where the server decides whether a request runs, on which
lane and in what rank, checking the KV pool and looking for retained state. See
[how ignis serves a request](../concepts/how-ignis-serves.md).

**Agent.** One of the two lane tags. An `agent` request, such as a subagent's,
ranks below an `interactive` one and is the first to move to host RAM when memory
is short. See *Lane tag*.

**Artifact.** The `.ninfer` container a model is served from: weights, tokenizer
and chat template in one file, checked against its sidecar. See
[models and artifacts](../concepts/models-and-artifacts.md).

**Base shape.** The 27B's verify round at 8 columns a lane, the anchor and 7
drafts, at every lane count: the one round shape of a load at 7 drafts or fewer,
and from 0.11.0, beside the *extended shape*, the base shape of a DFlash2 load
past 7. See [speculative decoding](../concepts/speculative-decoding.md#past-seven-drafts-two-round-shapes).

**Body format.** How an artifact stores its transformer layers (NVFP4 or
trellis). It is read from the artifact, never configured. See
[models and artifacts](../concepts/models-and-artifacts.md).

**Alias.** A second id a catalog entry answers to, within its own model family
and only on a card it loads on: on a 16 GB card or an RTX 3090,
`qwen3.8-27b-16g` answers to `qwen3.8-27b`, and on a 12 GB card, from 0.11.0,
`qwen3.8-27b-12g`; where both fit, the entry of the larger class answers. See
[Get a model](../guides/models.md).

**Card.** The GPU ignis runs on. The built-in profiles are written for the
12 GB class (from 0.11.0), the 16 GB class, the 24 GB class, a 32 GB RTX 5090 and a 96 GB RTX PRO 6000, and
`--profile auto` picks one by the card's memory. See [profiles](../concepts/profiles.md).

**Card class.** The cards one built-in profile serves, named by their memory
(12, 16, 24, 32 or 96 GB; 12 GB from 0.11.0, a total below 14 GiB), and the smallest
one a catalog entry loads on. An entry above the card is refused before anything
is unloaded or fetched. See [models and artifacts](../concepts/models-and-artifacts.md)
and [profiles](../concepts/profiles.md).

**Catalog.** The models the server knows how to fetch: thirteen entries built
into the binary (twelve in 0.10.0), nine text and image models, three speech
models and a diarizer,
pinned file by file, plus any an operator adds. See
[models and artifacts](../concepts/models-and-artifacts.md) and
[Get a model](../guides/models.md).

**Checkpoint.** See *Prompt checkpoint*.

**Chunk.** The slice of a prompt that one prefill call processes
(`--model-prefill-chunk`; the *solo width* while no lane decodes). Decode
rounds run between chunks. See
[how ignis serves a request](../concepts/how-ignis-serves.md).

**Committed run.** The tokens a verify round emits for one lane: the anchor plus
the accepted drafts, between 1 and the draft count plus one. The first round
after a prompt sends only its accepted drafts: its anchor is the first token,
which left with the prefill. See
[speculative decoding](../concepts/speculative-decoding.md).

**Continuation.** The request a loop-guard rewind submits: the same input, extended
by the reasoning kept before the cut and a nudge. See
[reasoning and the loop guard](../concepts/reasoning.md).

**Deadline.** How long a request may run before the server cuts it
(`--server-request-timeout`, `--server-stream-timeout`). See
[deadlines](../api/deadlines.md).

**Decision.** An answer read from the logits at one position after prefilling the
evidence, with no token decoded. It is the `POST /v1/decide` endpoint. See
[decisions](../api/decide.md).

**Decode lane.** See *Lane*.

**Decode round.** One traversal of the model that advances every decoding lane by a
token, replayed from a CUDA graph for the exact number of lanes. See
[how ignis serves a request](../concepts/how-ignis-serves.md).

**Decode share.** The part of the time decoding lanes keep while a prompt
prefills (`--model-decode-share`, 25 percent by default). See
[how ignis serves a request](../concepts/how-ignis-serves.md).

**DFlash2.** The 27B's default speculative-decoding drafter, grafted into its
artifact: a five-layer, sliding-window model that proposes 1 to 15 tokens per
round (1 to 7 before 0.11.0). See [speculative decoding](../concepts/speculative-decoding.md).

**Draft.** A token a drafter proposes ahead of the model, to be verified in the same
pass. See [speculative decoding](../concepts/speculative-decoding.md).

**Draft tokens.** The most tokens a lane drafts per round
(`--spec-draft-tokens`), the draft window: 1 to 15 for DFlash2 (from 0.11.0,
1 to 7 before; past 7 the load holds a *base shape* and an *extended shape*), 1
to 7 for Flash-Next's MTP head, and 0 to 7 for the 27B's MTP head, where 0
loads no speculation. See
[speculative decoding](../concepts/speculative-decoding.md).

**Drafter.** Whatever proposes the drafts, one per load: DFlash2 or its own MTP
head for the 27B (DFlash2 by default, MTP opt-in), the MTP head for Flash-Next.
See [the 27B's two drafters](../concepts/speculative-decoding.md#the-27bs-two-drafters).

**Expert cache.** On Flash-Next, the fixed-size pools of VRAM that hold the experts
in use, replaced least-recently-used. It takes what the KV pool leaves and is
refused below 12 GiB. See
[models and artifacts](../concepts/models-and-artifacts.md) and
[the VRAM plan](../concepts/vram-plan.md).

**Extended shape.** From 0.11.0, the 27B's verify round at the window + 1 columns
a lane (9 to 16), at up to 4 lanes: the second round shape of a DFlash2 load
past 7 drafts, on the *base shape*'s buffers. See
[speculative decoding](../concepts/speculative-decoding.md#past-seven-drafts-two-round-shapes).

**Family.** The architecture an artifact runs as (Qwen3.8-27B, Qwen3.8-Flash-Next,
Qwen-Image-2.1), read from the artifact's header. Several artifacts can share one
family. See [models and artifacts](../concepts/models-and-artifacts.md).

**Field.** One setting, named the same as a flag, an `IGNIS_*` variable and a file
key. See [configuration](../configuration/index.md).

**Fine-tune.** A catalog entry of a model family whose weights are another
party's post-training of the family's checkpoint: the same family under its own
served id, published under its own license. The abliterated variant and Swift
1.5 (`swift-1.5-27b`) are fine-tunes of the 27B. See
[fine-tunes](../guides/fine-tunes.md).

**Fixed draft.** The way the releases before 0.11.0 drafted for a request that
samples, and Flash-Next's MTP head always does: the drafter proposes its single
best draft and the round takes it with the model's probability of it. On the 27B
`--spec-draft-rule fixed` selects it. See
[speculative decoding](../concepts/speculative-decoding.md#acceptance-under-sampling).

**Flash-Next.** Qwen3.8-Flash-Next, the 125B mixture-of-experts model with 6B
active parameters. See [models and artifacts](../concepts/models-and-artifacts.md).

**Generation cap.** The most a request may generate, reasoning included: its
`max_tokens`, or `--model-default-max-tokens`. See
[chat completions](../api/chat-completions.md).

**GPU generation.** The family of NVIDIA cards a card belongs to, read from its
compute capability at start: Ampere (RTX 30), Ada (RTX 40) or Blackwell (RTX 50,
RTX PRO). A catalog entry may state the oldest one it loads on, and an entry
newer than the card is refused `409 model_above_gpu_generation`. See
[Get a model](../guides/models.md#the-catalog).

**Group.** The section a field belongs to (`server`, `model`, `vram`, `reuse`,
`switch`, `spec`, `vision`, `media`, `download`, `ngram`, `kv_disk`). See
[configuration](../configuration/index.md).

**Headroom.** The device memory a derived VRAM budget leaves to the desktop and
other programs (`--vram-headroom-bytes`, 1,536 MiB by default). See
[the VRAM plan](../concepts/vram-plan.md).

**hq-e8-2b.** The serving KV format: a compressed cache at 9,216 bytes per token on
the 27B, against 65,536 for BF16. See
[the KV cache and prompt reuse](../concepts/kv-cache-and-prompt-reuse.md).

**Idle unload.** Unloading a model after `--model-idle-unload` seconds with no
request, while the server stays up; the next request loads it again. See
[model switching and idle unload](../concepts/switching-and-idle-unload.md).

**Implicit switch.** A switch a request triggers by naming another known model in
`model`. It is controlled by `--switch-allow-implicit`. See
[model switching and idle unload](../concepts/switching-and-idle-unload.md).

**Interactive.** The default lane tag: a request from a person waiting on the
answer. It ranks above `agent`. See *Lane tag*.

**KV-disk.** The tier below KV-RAM: a file per evicted state, so a sequence resumes
where it stopped. On by default for Flash-Next. See
[KV-disk](../configuration/kv-disk.md) and
[the KV cache and prompt reuse](../concepts/kv-cache-and-prompt-reuse.md).

**KV page.** A fixed block of 64 tokens of key and value cache, the unit the KV
pool counts. See
[the KV cache and prompt reuse](../concepts/kv-cache-and-prompt-reuse.md).

**KV pool.** The device memory for KV pages, shared by all lanes. See
[the VRAM plan](../concepts/vram-plan.md) and
[VRAM budget and the KV pool](../configuration/vram-and-kv-pool.md).

**KV-RAM.** The host-RAM tier of pinned memory that holds evicted sequences and
retained state. It is held whole for the life of the load. See
[the KV cache and prompt reuse](../concepts/kv-cache-and-prompt-reuse.md).

**Lane.** A concurrent decode slot. A request holds one while it generates; the
27B has eight, Flash-Next 1 to 8. See
[how ignis serves a request](../concepts/how-ignis-serves.md).

**Lane tag.** A request's own statement of its class, `interactive` or `agent`,
carried in the `class` field or an `@<lane>` suffix on the model name. It drives
protection and eviction order. See
[how ignis serves a request](../concepts/how-ignis-serves.md).

**Last switch.** How the most recent switch or reload ended, kept on
`GET /v1/models`. See
[model switching and idle unload](../concepts/switching-and-idle-unload.md).

**Loop guard.** The task that scores a request's reasoning for repetition as it
streams, and takes a persistent loop back. See
[reasoning and the loop guard](../concepts/reasoning.md).

**MTP.** Multi-token prediction: a model's own draft head. The 27B's is in every
27B artifact and drafts from 0.11.0, in place of DFlash2: the default under `rtx-12gb`
and opt-in elsewhere (`--spec-backend mtp`), smaller
weights and no drafter window, for one more attention layer in every KV page.
Flash-Next's is shipped in a companion container, off on the RTX 5090 and on
under the `rtxpro6000` profile. See
[speculative decoding](../concepts/speculative-decoding.md).

**Nudge.** The sentence in the model's own voice that a rewind adds to the
reasoning before it goes on. See
[reasoning and the loop guard](../concepts/reasoning.md).

**Pending token.** The token a sequence has drawn and not yet sent: the next
round's anchor. A prompt's first token is one, and it is sent as soon as the
prompt's prefill ends, before any decode round. See
[how ignis serves a request](../concepts/how-ignis-serves.md).

**Pinned hot set.** Flash-Next's calibration-hottest experts, kept in the expert
cache for the life of the load with no copy in RAM
(`--vram-pinned-expert-bytes`). See
[host RAM](../configuration/host-ram.md) and [profiles](../concepts/profiles.md).

**Prefill.** Reading the prompt through the model once to fill its KV and state and
draw the first token. See
[how ignis serves a request](../concepts/how-ignis-serves.md).

**Prefill lane.** The single slot where one prompt prefills at a time; others queue
behind it. See [how ignis serves a request](../concepts/how-ignis-serves.md).

**Prefix reuse.** Skipping the prefill of a prompt head an earlier request already
read, by resuming from retained state that matches by content. See
[the KV cache and prompt reuse](../concepts/kv-cache-and-prompt-reuse.md).

**Profile.** A named bundle of hardware-shaped defaults, below the config file and
above the built-in defaults: `rtx-12gb` (from 0.11.0), `rtx-16gb`, `rtx-24gb`,
`rtx5090` and `rtxpro6000` are built in, one for each card class, and `auto`
picks the card's. See [profiles](../concepts/profiles.md).

**Prompt checkpoint.** The whole state of a request at its generation opener, kept
after it ends so the next turn resumes there. See
[the KV cache and prompt reuse](../concepts/kv-cache-and-prompt-reuse.md).

**Provenance sidecar.** See *Sidecar*.

**Reasoning.** The model's thinking, written between `<think>` and `</think>` and
returned as `reasoning_content`. See
[reasoning and the loop guard](../concepts/reasoning.md).

**Restore floor.** The prompt tokens a match in a lower tier must reuse beyond the
best match above it to be worth the crossing. See
[the KV cache and prompt reuse](../concepts/kv-cache-and-prompt-reuse.md).

**Retained prefix.** A shared prefix, ending at the system and tools block, kept
alive with no request holding it so the next subagent can claim it. See
[the KV cache and prompt reuse](../concepts/kv-cache-and-prompt-reuse.md).

**Retained slot.** A place reserved at load for one image of mutable state, on the
device or in pinned host memory. See
[the KV cache and prompt reuse](../concepts/kv-cache-and-prompt-reuse.md).

**Retained state.** Prompt checkpoints and retained prefixes: state no live request
needs, first to go when room is needed. See
[the KV cache and prompt reuse](../concepts/kv-cache-and-prompt-reuse.md).

**Rewind.** The loop guard's action: cancel the request and continue it from the
text before the loop, with a nudge. See
[reasoning and the loop guard](../concepts/reasoning.md).

**Round shape.** The columns a lane and the most lanes of a 27B verify round:
the *base shape*, or from 0.11.0 on a DFlash2 load past 7 drafts the *extended
shape*, one for all of a round's lanes, which the *shape rule* and the *shape
policy* choose. See
[speculative decoding](../concepts/speculative-decoding.md#past-seven-drafts-two-round-shapes).

**Row budget.** The most rows one Flash-Next MTP verify round may take across all
lanes (`--spec-draft-rows`, 8 by default). See
[speculative decoding](../concepts/speculative-decoding.md).

**Sampled draft.** A way for the 27B's DFlash2 to draft for a request that
samples (`--spec-draft-rule sampled`, from 0.11.0): the drafter draws each draft
from its own distribution and the round takes it by the ratio of the model's
probability to the drafter's, so the text keeps the model's distribution. See
[speculative decoding](../concepts/speculative-decoding.md#acceptance-under-sampling).

**Shape policy.** From 0.11.0, on a DFlash2 load past 7 drafts, how a verify
round of 1 to 4 lanes takes its shape from what its requests ask
(`--spec-shape-policy`): `aggregate`, the default, extended when the drafts the
asking requests are expected to take past position 7 exceed what the wider round
costs every request in it, from each request's own recent rounds; `all`,
extended only when every request asks, so no request pays for another. A request
alone runs at the shape it asks under either. See
[speculative decoding](../concepts/speculative-decoding.md#past-seven-drafts-two-round-shapes).

**Shape rule.** From 0.11.0, on a DFlash2 load past 7 drafts, what each request
asks of its next verify round from its own earlier rounds: the *extended shape*
after a base round that took all 7 drafts, while its extended rounds take 6 or
more (`--spec-shape-enter-accepted`, `--spec-shape-stay-accepted`), the *base
shape* otherwise. The *shape policy* turns a round's asks into its shape. See
[speculative decoding](../concepts/speculative-decoding.md#past-seven-drafts-two-round-shapes).

**Scope.** A model family's own section of a field: `qwen38` for the 27B,
`qwen38flashnext` for Flash-Next. A value in a scope wins over the general one
for that family. See [configuration](../configuration/index.md).

**Served id.** The name a client puts in `model` and `GET /v1/models` reports. It
tells artifacts of one family apart. See
[models and artifacts](../concepts/models-and-artifacts.md).

**Sidecar.** The file beside an artifact (`.graft.json` or `.conversion.json`) that
records its provenance, checked at load. See
[models and artifacts](../concepts/models-and-artifacts.md).

**Solo width.** The prefill chunk a prompt takes while no lane is decoding
(`--model-prefill-chunk-solo`), at least the prefill chunk and planned at load
beside it. "Solo" names the absence of decoding lanes, not of users. See
[the solo width](../configuration/context-and-speculation.md#the-solo-width).

**Speculation.** Drafting tokens ahead and verifying them in one pass, so a round
can yield several tokens. See
[speculative decoding](../concepts/speculative-decoding.md).

**Switch.** Replacing the loaded model with another, as a full reload, without
restarting the server. See
[model switching and idle unload](../concepts/switching-and-idle-unload.md).

**Thinking budget.** The most reasoning tokens before the server forces the close
(`--model-thinking-budget`, 32,768 by default). See
[reasoning and the loop guard](../concepts/reasoning.md).

**TTFT.** Time to first token: how long a request waits before the first token.
Prefix reuse and the decode share both move it. See
[the KV cache and prompt reuse](../concepts/kv-cache-and-prompt-reuse.md).

**Verify round.** A decode round in which each lane carries its anchor and drafts and
the model checks them all in one traversal. See
[speculative decoding](../concepts/speculative-decoding.md).

**VRAM budget.** The device memory the whole process may hold, weights included:
the free memory at start minus the headroom, or a named figure. See
[the VRAM plan](../concepts/vram-plan.md).

**VRAM plan.** The layout of every byte of the budget, decided and printed at load;
nothing is allocated on the device while serving. See
[the VRAM plan](../concepts/vram-plan.md).

**WDDM.** The Windows display driver model, which pages device allocations to system
RAM instead of failing them. See [the VRAM plan](../concepts/vram-plan.md).

**YaRN.** The rope-scaling method that stretches the checkpoint's trained 262,144
positions to a longer context; chosen by `auto` from `--model-max-context`. See
[context, KV and speculation](../configuration/context-and-speculation.md).
