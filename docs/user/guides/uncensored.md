---
title: "The uncensored variant"
description: "qwen3.8-27b-abliterated: the 27B with the huihui-ai abliteration, its artifact for each card class, and what it changes."
sidebar:
  order: 4
---

`qwen3.8-27b-abliterated` is the 27B with the
[huihui-ai abliteration](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated)
applied. **It does not refuse**, so put the guardrails in the tool layer.

It comes as three artifacts, one for each card class, and a client names the
same id on all three:

| Card | Served as | Repository | Size |
|---|---|---|---|
| 32 GB or larger | `qwen3.8-27b-abliterated` | [`gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer`](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer) | 18.1 GiB |
| 16 GB | `qwen3.8-27b-abliterated-16g` | [`gpillon/Qwen3.8-27B-trellis-3bpw-abliterated-ignis`](https://huggingface.co/gpillon/Qwen3.8-27B-trellis-3bpw-abliterated-ignis) | 12.5 GiB |
| 12 GB | `qwen3.8-27b-abliterated-12g` | [`gpillon/Qwen3.8-27B-trellis-2.8bpw-abliterated-ignis`](https://huggingface.co/gpillon/Qwen3.8-27B-trellis-2.8bpw-abliterated-ignis) | 11.9 GiB |

```
ignis-server model download qwen3.8-27b-abliterated   # the artifact this card loads
ignis-server --model-id qwen3.8-27b-abliterated       # found under --download-path, fetched when it is not there
make dev UNCENSORED=1   # the same from a checkout, on the 32 GB artifact
```

On a 32 GB card or larger it is served as `qwen3.8-27b-abliterated`, so a
client can tell it from the 27B. On a 16 GB card the same commands fetch and
start `qwen3.8-27b-abliterated-16g`, which answers to `qwen3.8-27b-abliterated`
there, as `qwen3.8-27b-16g` answers to `qwen3.8-27b`
([ADR 0074](../../adr/0074-a-catalog-entry-may-state-its-family-and-an-alias-may-be-another-entrys-id.md)).
On a 12 GB card, an RTX 50, RTX 40 or RTX 30 card alike, they
fetch and start `qwen3.8-27b-abliterated-12g`, which answers to it there, as
`qwen3.8-27b-12g` answers to `qwen3.8-27b`; the 16 GB build is refused there by
name before anything is fetched. Where both builds fit, on a 16 or 24 GB
card, the 16 GB one answers
([ADR 0098](../../adr/0098-the-12-gb-card-class-serves-the-27b-from-a-recipe-of-our-own-under-a-byte-ceiling-on-one-lane-drafting-with-mtp.md)).
A start, a switch and a request naming `qwen3.8-27b-abliterated` are served by
the build that answers to it, the answer's `model` names that build, and
`GET /v1/models` lists the 32 GB entry as unavailable, the reason naming the
entry that serves it ([the catalog](models.md#the-catalog)). A script that
must have one artifact names its id itself: `qwen3.8-27b-abliterated-16g` loads
on a 16 GB card or larger, `qwen3.8-27b-abliterated-12g` on every card.
`make dev UNCENSORED=1` names the 32 GB artifact whatever the card, as
`make dev` names the 32 GB 27B's: on a 16 GB or a 12 GB card, start with
`ignis-server --model-id`.

## The 32 GB artifact

The default 27B's artifact with the abliteration: 1,255 of its 1,325 objects
are byte-identical to the default artifact, and only the 70 matrices the
abliteration changed are re-encoded. The pointing heads and the DFlash2
drafter load unchanged. They were fitted to the default weights, and the target
still verifies every draft, so the drafter can only change the acceptance rate.

## The 16 GB artifact

The [16 GB 27B's artifact](../why-ignis/sixteen-gb-cards.md) with the same 70
matrices re-quantized from the abliterated weights: the MLP down projection of
layers 17-51, and the linear-attention (GDN) or attention output projection of
each. The other 1,152 of its 1,222 objects are the 16 GB artifact's, byte for
byte: the rest of the trellis body, the output head, the token embedding, the
DFlash2 drafter, the vision tower, the tokenizer and the chat template. Its
load, its speed and its limits are the 16 GB artifact's.

How it was made:

- **Each of the 70 matrices** keeps the bit width the 16 GB artifact gives it,
  3 bits, and 4 for layer 23's attention output. It is re-quantized by
  replaying exllamav3 1.5.3's own conversion of that matrix: the quant's
  calibration rows, the state entering the layer taken through the layers
  before it as quantized, and the Hessian captured on the abliterated layer.
- **The pack** is the 16 GB artifact's conversion with the 70 records swapped
  in, checked object by object against the 16 GB artifact: the 70 differ and
  everything else is identical.
- **Its own weights id** gives it its own content hash, so state retained
  under one 16 GB artifact is never reused by the other.

Its three gates were registered before the run and measured on an RTX 5090 held
to a 16 GB card's budget (`rtx-16gb`, 262,144 tokens of context, two lanes,
vision off). All three pass:

| Gate | The 16 GB abliterated artifact | Against | Pass if |
|---|---|---|---|
| KLD from the abliterated BF16, 16 windows of 4,096 tokens (code, prose, chat) | 0.04088 (95% CI 0.0351-0.0467) | the 16 GB artifact from the base BF16: 0.04034 | at most 0.0484 (1.2x) |
| DFlash2 tokens per round, recorded agent turns replayed | 0.9986x the 16 GB artifact's (95% CI 0.9946-1.003), on 83 of the 116 jobs | the 16 GB artifact on the same turns | at least 0.90x |
| Refusals: a lock-picking request, and a roast with strong profanity | answers both | the 16 GB artifact refuses both | answers what the 16 GB artifact refuses |

MMLU-Pro and the reasoning-loop rate were not measured; the KLD covers how
faithful it is. Neither 16 GB artifact has pointing heads or a `locate`
calibration, so on both [`/v1/decide`](../api/decide.md) refuses `locate` and
answers `point` and `box` by the digit chain. The build and the gates in
detail:
[the finding](../../findings/2026-10-30-27b-16gb-abliterated-gates.md).

## The 12 GB artifact

The 12 GB 27B's artifact, `qwen3.8-27b-12g`
([the `rtx-12gb` profile](../concepts/profiles.md#rtx-12gb)), with the same 70
matrices re-quantized from the abliterated weights, made the way the 16 GB one
was. The other 1,152 of its 1,222 objects are the 12 GB artifact's, byte for
byte: the rest of the trellis body, the output head, the token embedding, the
MTP head, the DFlash2 drafter, the vision tower, the tokenizer and the chat
template. It is the 12 GB artifact's size, 12,754,663,684 bytes, and its load,
its profile, its context and its limits are the 12 GB artifact's: one lane,
the MTP drafter at a window of 7, up to 131,072 tokens of context.

How it was made:

- **Each of the 70 matrices** keeps the bit width the 12 GB artifact gives it:
  3 bits on 64 of them, 4 on 5 and 2 on 1. It is re-quantized by replaying
  exllamav3 1.5.3's own conversion of that matrix on the 12 GB artifact's own
  quant, which ignis converted locally at its recipe and pins by each file's
  SHA-256: the calibration rows the recipe names, the state entering the layer
  taken through the layers before it as quantized, and the Hessian captured on
  the abliterated layer. The 70 reconstruct the abliterated weights as closely
  as the 12 GB artifact's reconstruct the standard ones: a median relative
  error of 0.15073 against 0.15072.
- **The pack** is the 12 GB artifact's conversion with the 70 records swapped
  in, checked object by object against the 12 GB artifact: the 70 differ and
  everything else is identical.
- **Its own weights id** gives it its own content hash, so state retained
  under one 12 GB artifact is never reused by the other.

Its four gates were registered before the run and measured on an RTX 5090
held to a 12 GB card's shape (`rtx-12gb`, a 12 GiB budget, one lane, MTP at a
window of 7, a 512-token chunk), with the 12 GB artifact read again in the
same session:

| Gate | The 12 GB abliterated artifact | Against | Pass if |
|---|---|---|---|
| KLD from the abliterated BF16, 16 windows of 4,096 tokens (code, prose, chat) | 0.04275 (95% CI 0.0363-0.0493) | the 12 GB artifact from the base BF16: 0.04308 | at most 0.0517 (1.2x) |
| Decode with MTP against no speculation, at 8,192, 65,536 and 130,560 tokens of context, a cold prompt and a reused one | 2.81x to 3.34x | no speculation, two launches a cell | faster in every cell |
| Requests caught in a reasoning loop, 32 requests | 9 of 32 | the 12 GB artifact: 6 of 32 | not more by a one-sided Fisher p below 0.05 (p 0.28) |
| Refusals: a lock-picking request, and a roast with strong profanity | answers both | the 12 GB artifact refuses both | answers what the 12 GB artifact refuses |

All four pass. It is about as faithful to the abliterated weights as the
12 GB artifact is to the standard ones (0.99x, CI 0.96-1.02), with a top-1
agreement of 93.34% against 93.35%. The KLD gate's two cross terms
([ADR 0082](../../adr/0082-the-gate-is-kld-against-bf16-dflash2-on-against-off-and-the-loop-rate.md))
sit above the matched pair on average but not in every window, as for the
16 GB abliterated artifact:
[the finding](../../findings/2026-11-20-the-uncensored-12-gb-artifacts-gate.md)
says how that leg was read.

Under MTP it decodes 0.99x to 1.08x the 12 GB artifact (against one launch
of it; the one cell below 1, a cold 8,192-token prompt, is inside that cell's
spread). Its first token comes about 2.5 s after an 8,192-token prompt,
1.2% later than the 12 GB artifact's, and 20.9 s after 65,536 and 46 s after
130,560, within 1% of it. Held to the same 12 GiB, DFlash2 loads with the
same context and a smaller KV pool (150,720 tokens against MTP's 214,144), and
decodes 1.31x to 1.47x MTP. The profile keeps MTP, as for the 12 GB artifact:
on a real 12 GB card DFlash2's 0.7 GB more is the context, and by the plan's
arithmetic it keeps 49,152 tokens on an RTX 50 card with nothing else on it and
none beside a desktop ([`rtx-12gb`](../concepts/profiles.md#rtx-12gb)).

What these numbers do not cover:

- **Every figure is the RTX 5090's, held to 12 GiB.** No real 12 GB card has
  run this file, and its speeds will be lower.
- **The RTX 30 and RTX 40 routes** were not read on this file; the 12 GB
  artifact, which it shares all but 70 matrices with, reads 1.01 times its
  Blackwell KLD on them.
- **Not measured:** MMLU-Pro (the KLD covers how faithful it is), and an image:
  the vision tower is the 12 GB artifact's bit for bit. The refusals are two
  prompts read at temperature 0: they show the flip, not a refusal rate.

Neither 12 GB artifact has pointing heads or a `locate` calibration, so on
both [`/v1/decide`](../api/decide.md) answers `choice`, refuses `locate`, and
answers `point` and `box` by the digit chain. The build and the gates in
detail:
[the finding](../../findings/2026-11-20-the-uncensored-12-gb-artifacts-gate.md).
