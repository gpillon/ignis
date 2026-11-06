---
title: "Fine-tunes"
description: "A fine-tune entry: another party's post-training of a family's checkpoint, under its own id and license. Swift 1.5's two entries, their license and its threshold, and what differs from the 27B."
sidebar:
  order: 4
---

A **fine-tune** is a catalog entry of a model family whose weights are another
party's post-training of the family's checkpoint. It is the same family, so it
loads as the family's model does, and a client tells it apart by its served
id. It is published under
its own license, which may not be the family's, and the entry states it: `model
list`, `model download`, the Server view and `GET /v1/models/catalog` show it
([the catalog](models.md#the-catalog)). An entry of a fine-tune whose id names
no family states the family it is of, `family: qwen3.8-27b`, which is how its
16 GB build answers to its id on a 16 GB card
([ADR 0074](../../adr/0074-a-catalog-entry-may-state-its-family-and-an-alias-may-be-another-entrys-id.md)).

The built-in catalog has two fine-tunes of the 27B: the
[uncensored variant](uncensored.md), whose post-training is an abliteration,
and Swift 1.5.

## Swift 1.5

[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) is UkisAI's
post-training of Qwen3.8-27B. It comes as two artifacts, one for each card
class, and a client names the same id on both:

| Card | Served as | Repository | Size |
|---|---|---|---|
| 32 GB or larger | `swift-1.5-27b` | [`gpillon/Swift-1.5-Qwen3.8-27B-nvfp4-ignis`](https://huggingface.co/gpillon/Swift-1.5-Qwen3.8-27B-nvfp4-ignis) | 18.1 GiB |
| 16 GB | `swift-1.5-27b-16g` | [`gpillon/Swift-1.5-Qwen3.8-27B-trellis-3bpw-ignis`](https://huggingface.co/gpillon/Swift-1.5-Qwen3.8-27B-trellis-3bpw-ignis) | 12.5 GiB |

```
ignis-server model download swift-1.5-27b   # prints the license, then fetches the artifact this card loads
ignis-server --model-id swift-1.5-27b       # found under --download-path, fetched when it is not there
```

On a 32 GB card or larger, and on a card whose class is not known, it is served
as `swift-1.5-27b`, so a client can tell it from the 27B. On a 16 or 24 GB
card, and on any RTX 30 or RTX 40 card, the
same commands fetch and start `swift-1.5-27b-16g`, which answers to
`swift-1.5-27b` there, as `qwen3.8-27b-16g` answers to `qwen3.8-27b`. A start,
a switch and a request naming `swift-1.5-27b` are served by it, the answer's
`model` names it, and `GET /v1/models` lists the 32 GB entry as unavailable,
the reason naming the entry that serves it. A script that must have the 16 GB
artifact names `swift-1.5-27b-16g` itself. Both entries are pinned file by
file in the [generated reference](../reference/models.md#swift-15-27b).

### Its license

Both entries are under UkisAI's Swift Open License v1.0,
`swift-open-license-1.0`, not Apache-2.0. Each links the `LICENSE` in its
repository at the pinned revision; read it and the `NOTICE` beside it before
you download.

- **Commercial use at or above the threshold is not licensed.** The license's
  threshold is gross revenue of US$1,000,000 or more, measured over the most
  recently completed fiscal year of the user together with every entity that
  controls it, is controlled by it, or is under common control with it.
  Commercial use at or above it needs a separate written license from UkisAI,
  the Swift Enterprise License.
- **Qualified non-profits are exempt** for non-commercial or research use: the
  threshold does not apply to a qualified non-profit organization (a 501(c)(3)
  organization, or an equivalent non-profit in another jurisdiction) using the
  weights for non-commercial or research purposes.
- **The Qwen3.8-27B base parts remain Apache-2.0.** The license limits only
  UkisAI's contribution and what contains or derives from it, as these weights
  do.

`ignis-server model download` prints the license before it fetches, on a 16 GB
card:

```
swift-1.5-27b-16g: license swift-open-license-1.0: https://huggingface.co/gpillon/Swift-1.5-Qwen3.8-27B-trellis-3bpw-ignis/blob/3fe94555a30b3df89a283bf61338281243642166/LICENSE
```

No step asks to accept it: whoever downloads is bound by it, as with any
download from the hub
([ADR 0078](../../adr/0078-the-built-in-catalog-names-each-entrys-license-and-may-ship-restricted-weights.md)).

### What differs from the 27B

- **`locate` is refused, and `point` and `box` answer by the digit chain.** No
  pointing head and no `locate` calibration was made on Swift's weights, and
  each Swift artifact has its own weights id and so its own content hash, which
  the 27B's calibrations are keyed to. On either artifact
  [`/v1/decide`](../api/decide.md) answers `locate` with
  `422 locate_uncalibrated`, and `point` and `box` by the digit chain.
- **Vision works as on the 27B.** Swift's vision tower is the base's, byte for
  byte, and each artifact loads it as the 27B's artifact of its card class does
  ([vision](../configuration/vision.md)).
- **The rest is the 27B's**: the chat template and tokenizer, which Swift ships
  unchanged, reasoning, tool calls, the profiles and the VRAM plan. The DFlash2
  drafter is the 27B's, trained for the base and used unchanged.

### How it was made

- **`swift-1.5-27b`**, weights id `nvfp4full-swift-1.5`: the 247 NVFP4
  matrices of its text layers encoded from Swift's BF16 checkpoint and their
  activation divisors calibrated locally, with the 27B NVFP4 artifact's recipe
  and object list. Its DFlash2 drafter (66 objects), vision tower (333) and
  frontend (6) are SHA-256-equal to the 27B's artifact's.
- **`swift-1.5-27b-16g`**, weights id `trellis-sc30-swift-1.5`: its body and
  output head are an exllamav3 1.5.3 conversion of Swift's BF16 at the
  per-matrix bit widths of the [16 GB 27B](../why-ignis/sixteen-gb-cards.md),
  and every other object is the Swift NVFP4 artifact's, byte for byte.

Each has its own content hash, so state retained under another 27B artifact is
never reused for it.

### The gates

Measured on one RTX 5090 against Swift's own BF16: `swift-1.5-27b` as the 27B
is served (eight lanes, a 2,048-token prefill chunk, DFlash2 with 7 draft
tokens), `swift-1.5-27b-16g` held to a 16 GB card's budget (`rtx-16gb`,
17,179,869,184 bytes, 262,144 tokens of context, two lanes, a 1,024-token
chunk, vision off). The KLD limits are 1.2 times what the 27B's artifact of
the same card class scores against the base's BF16 (0.04916 and 0.04034),
registered before the run. One machine, one run per leg.

| Gate | `swift-1.5-27b` | `swift-1.5-27b-16g` |
|---|---|---|
| KLD from Swift's BF16, 16 windows of 4,096 tokens (code, prose, chat), 95% CI | 0.05355 (0.04692-0.06023); limit 0.0590 | 0.04255 (0.03709-0.04785); limit 0.0484 |
| Decode tok/s, DFlash2 on against off, every cell (8,192, 65,536 and 253,440 tokens of context) | on above off in every cell, 3.71x to 5.27x; one lane at 8,192: 368.3 against 81.1 | on above off in every cell, 3.91x to 5.22x; one lane at 8,192: 410.7 against 92.6 |
| Smokes at temperature 0: a chat turn, a reasoning turn split into reasoning and content, a tool call | pass | pass |

The KLD rule is the point estimate: `swift-1.5-27b`'s CI ends at 0.06023, above
its limit; `swift-1.5-27b-16g`'s whole CI is under its own. The controls,
against those two values: the 27B's artifact given Swift's BF16 scores 0.05651
and 0.04643, and each Swift artifact given the base's BF16 0.05923 and
0.04961.

Reported, not gates:

| | `swift-1.5-27b` | `swift-1.5-27b-16g` |
|---|---|---|
| DFlash2 tokens per round, recorded agent turns replayed, against the 27B's artifact of the same class | 1.001 (95% CI 0.998-1.004) | 1.003 (95% CI 1.000-1.006) |
| DFlash2 tokens per round on its own, exact / traffic-weighted | 3.467 / 3.535 (the 27B's: 3.397 / 3.466) | 3.482 / 3.552 (the 27B's: 3.420 / 3.487) |
| Turns caught in a reasoning loop | 11 of 32 (the 27B's: 14 of 32, on another build and context) | not measured |

The replayed turns were sampled from the 27B, not from Swift. A vision gate,
MMLU-Pro and a refusal comparison were not run. The builds and the gates in
detail:
[the NVFP4 finding](../../findings/2026-11-03-swift-1-5-nvfp4-passes-the-27b-gates.md)
and [the 16 GB finding](../../findings/2026-11-03-swift-1-5-16gb-passes-the-27b-gates.md).
