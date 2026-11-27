---
status: accepted
---

# A catalog entry states the oldest GPU generation it loads on

On RTX 30 and RTX 40 cards a card's memory no longer says what it loads: an RTX 3090 or RTX 4090 has 24 GB, but
not the FP4 tensor cores the NVFP4 27B prefills on nor the block-scaled FP8 the image model runs on. The route
checks (ADR 0077) refuse such a load only at the load, once a switch has torn the loaded model down, and the model
lists still offer it. So the **GPU generation** is a second axis beside the **card class** (ADR 0063, ADR 0064),
read and checked the same way, before anything stops.

- **The card's generation** follows from the compute capability the start and the `model` verbs read: Ampere, Ada
  or Blackwell. A capability of no generation, or none read, refuses nothing by generation, as a total of no class
  refuses nothing by class.
- **A catalog entry may state `generation`**, the oldest it loads on, optional in both catalogs. An entry that
  states none loads on every generation the build runs on; a value that is no generation refuses the catalog.
- **One rule over both axes**, in ADR 0064's resolve step: an id stands when its own entry fits the card's class
  and its generation both; else the entry of its family that lists it as an alias and fits both answers to it;
  else every caller of the class rule refuses it, a catalog id the operator also lists in `switch.known_models`
  included.
- **The code** is `409 model_above_gpu_generation` whenever the generation refuses, alone or with the class, since
  a larger card of the same generation would not load it either; `409 model_above_card_class` is for the class
  alone. The lists report the entry `available: false` with its `reason`, as for the class.
- **What a built-in entry states.** `blackwell` on an artifact that runs only there, an NVFP4 artifact, Flash-Next
  and the image model, and on a trellis build not yet verified below Blackwell; none on the models that run on the
  CPU. Two facts admitted Swift 1.5's 16 GB build below Blackwell, and neither is a run of Swift on an sm_86 card.
  Its layout is that of a build that has loaded, prefilled and decoded on one, as the 16 GB 27B has: the engine
  picks its routes by family and body format, never by weights id. And its quality on the 8.x routes, read on the
  RTX 5090, is within ADR 0082's limit for a route's numerics. A run of it on an sm_86 card would move its numbers,
  not decide whether it loads: the 16 GB 27B's run there read the RTX 5090's quality. So it states none: its
  conversion sidecar matches the 16 GB 27B's field for field once hashes, quantization errors and provenance are
  set aside (the same trellis matrices and formats, the same copied objects, the same locally quantized NVFP4
  drafter, the same byte count), and its KLD ratio against V0, read in #513 on the dense prefill and the integer
  band, was 1.0148 [0.9892, 1.0340], and 1.0143 [1.0022, 1.0264] with ignis's own GDN prefill, against a limit of
  1.05. The 12 GB 27B states none and stands on the same two (ADR 0098).
- **The route checks stay** as the backstop, as the VRAM plan's refusal stays beside the class.

## Considered Options

- **Fold the generation into the class**, an "RTX 30 24 GB" class. The two are independent, and the class stays
  memory alone: a 24 GB Blackwell card loads what an RTX 3090 loads by memory and more by generation, and an Ada
  card whose total is no class still needs the alias to the 16 GB entry.
- **The route checks alone.** They run after a switch has torn the loaded model down, and the lists would offer
  models the card cannot run.
