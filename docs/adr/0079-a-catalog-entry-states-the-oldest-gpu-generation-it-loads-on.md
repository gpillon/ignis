---
status: accepted
---

# A catalog entry states the oldest GPU generation it loads on

**Amended by #529** (2026-11-07): Swift 1.5's trellis build states no
generation, as the 27B's two 16 GB entries; see the amendment at the end.

On RTX 30 and RTX 40 cards (spec #474) a card's memory no longer says what it
loads: an RTX 3090 or RTX 4090 has 24 GB, but not the FP4 tensor cores the
NVFP4 27B prefills on nor the block-scaled FP8 the image model runs on. The
route checks of #457 (ADR 0077) refuse such a load only at the load, once a
switch has torn the loaded model down, and the model lists still offer it. So
the **GPU generation** becomes a second axis beside the **card class** (ADR
0063, ADR 0064), read and checked the same way, before anything stops:

- **The card's generation** follows from the compute capability the start and
  the `model` verbs read: 8.6 and 8.7 Ampere, 8.9 Ada, every 12.x Blackwell. A
  capability of no generation, or none read, refuses nothing by generation, as
  a total of no class refuses nothing by class.
- **A catalog entry may state `generation`**, the oldest it loads on:
  `ampere`, `ada` or `blackwell`, optional in both catalogs. None loads on
  every generation the build runs on; any other value refuses the catalog.
- **One rule over both axes**, in ADR 0064's resolve step: an id stands when
  its own entry fits both; else the entry of its family that lists it as an
  alias and fits both answers to it; else every caller of the class rule
  refuses it, a catalog id the operator also lists in `switch.known_models`
  included. On an RTX 3090 or RTX 4090 the 27B's ids are served by their 16 GB
  entries.
- **Codes.** `409 model_above_gpu_generation` whenever the generation refuses,
  alone or with the class, since a larger card of the same generation would
  not load it either; `409 model_above_card_class` for the class alone. The
  message names what the entry needs and the card's name, compute capability
  and generation; the lists report the entry `available: false` with its
  `reason`.
- **Built-in values.** `blackwell` on every NVFP4 artifact, Flash-Next, the
  image model and a trellis build not yet verified below Blackwell
  (`swift-1.5-27b-16g`); none on the 27B's two 16 GB entries, the speech models
  and the diarizer.
- **The route checks stay** as the backstop, as the VRAM plan's refusal stays
  beside the class.

The class stays memory alone and gains a **24 GB class**, 20 GiB up to but not
including 28 (the RTX 3090 and RTX 4090; 16 < 24 < 32 < 96), which a catalog's
`card_class` accepts and no built-in entry states. `--profile auto` picks a new
built-in profile for it, **`rtx-24gb`**, and the 16 GB profile is renamed
**`rtx-16gb`**, with `rtx50-16gb` a permanent alias wherever a profile is named
(#476, #480).

## Considered options

- **Fold the generation into the class** (an "RTX 30 24 GB" class). Rejected:
  the two are independent. A 24 GB Blackwell card loads what an RTX 3090 loads
  by memory and more by generation, and an Ada card whose total is no class
  still needs the alias to the 16 GB entry.
- **The route checks alone.** Rejected: they run after a switch has torn the
  loaded model down, and the lists would offer models the card cannot run.

## Amendment (2026-11-07): Swift 1.5's trellis build states no generation (#529)

`swift-1.5-27b-16g` stated `blackwell` as a trellis build not yet verified
below Blackwell. It now states none, as the 27B's two 16 GB entries do, so on
an RTX 30 or RTX 40 card it loads and answers to `swift-1.5-27b`. Two facts
admit it, and neither is a run of Swift on an sm_86 card:

- **Its layout is the 16 GB 27B's.** Its conversion sidecar matches that
  build's field for field once hashes, quantization errors and provenance are
  set aside: the same trellis matrices and formats, the same copied objects,
  the same locally quantized NVFP4 drafter, the same byte count. The engine
  picks its routes by family and body format, never by weights id, and the
  16 GB 27B has loaded, prefilled and decoded on an sm_86 card (#484).
- **Its quality on the 8.x routes passed.** #513's RTX 5090 leg ran it on the
  dense prefill and the integer band: KLD ratio against V0 1.0148
  [0.9892, 1.0340], and 1.0143 [1.0022, 1.0264] with ignis's own GDN prefill,
  against a limit of 1.05.

A Swift run on an sm_86 card would move its numbers, not decide whether it
loads; the 16 GB 27B's run there read the RTX 5090's quality.
