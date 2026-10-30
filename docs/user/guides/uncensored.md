---
title: "The uncensored variant"
description: "qwen3.8-27b-abliterated: the 27B with the huihui-ai abliteration, its artifact for each card class, and what it changes."
sidebar:
  order: 4
---

`qwen3.8-27b-abliterated` is the 27B with the
[huihui-ai abliteration](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated)
applied. **It does not refuse**, so put the guardrails in the tool layer.

It comes as two artifacts, one for each card class, and a client names the
same id on both:

| Card | Served as | Repository | Size |
|---|---|---|---|
| 32 GB or larger | `qwen3.8-27b-abliterated` | [`gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer`](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer) | 18.1 GiB |
| 16 GB | `qwen3.8-27b-abliterated-16g` | [`gpillon/Qwen3.8-27B-trellis-3bpw-abliterated-ignis`](https://huggingface.co/gpillon/Qwen3.8-27B-trellis-3bpw-abliterated-ignis) | 12.5 GiB |

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
A start, a switch and a request naming `qwen3.8-27b-abliterated` are served by
it, the answer's `model` names it, and `GET /v1/models` lists the 32 GB entry
as unavailable, the reason naming the entry that serves it
([the catalog](models.md#the-catalog)). A script that must have the 16 GB
artifact names `qwen3.8-27b-abliterated-16g` itself. `make dev UNCENSORED=1`
names the 32 GB artifact whatever the card, as `make dev` names the 32 GB
27B's: on a 16 GB card, start with `ignis-server --model-id`.

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
to a 16 GB card's budget (`rtx50-16gb`, 262,144 tokens of context, two lanes,
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
