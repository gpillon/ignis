---
status: accepted
---

# A catalog entry states the oldest GPU generation it loads on

Running ignis on RTX 30 and RTX 40 cards (spec #474) breaks an assumption ADR
0063 and ADR 0064 shared: that a card's memory says what it loads. An RTX 3090
or RTX 4090 has 24 GB, but it has no FP4 tensor cores for the NVFP4 27B's
prefill and no block-scaled FP8 for the image model, which only Blackwell has.
The route checks of #457 (ADR 0077) refuse such a load by name, but only at the
load, once a switch has drained and torn the loaded model down, and the model
lists would still offer it. So the **GPU generation** becomes a second axis
beside the **card class**, read and checked the same way:

- **The card's generation** follows from the compute capability the start
  already reads: 8.6 and 8.7 Ampere (RTX 30), 8.9 Ada (RTX 40), every 12.x
  Blackwell (RTX 50, RTX PRO). A capability of no generation, or none read,
  refuses nothing by generation, as a total of no class refuses nothing by
  class. The `model` verbs read it as a start does.
- **A catalog entry may state `generation`**, the oldest generation it loads
  on: `ampere`, `ada` or `blackwell`, ordered by age, optional in both catalogs.
  An entry stating none loads on every generation the build runs on; any other
  value refuses the catalog, naming the entry.
- **One rule over both axes.** The class rule and the resolve step take the
  card's class and its generation together, from the one card reading the
  server is given (the seam ADR 0064's rule already had, widened). An id stands
  when its own entry fits both; otherwise the entry of its family that lists
  it as an alias and fits both answers to it; otherwise it is refused, by the
  same callers as for the class: a start with no artifact named, a switch by
  id, a request's implicit switch, `model download`, and a catalog id the
  operator also listed in `switch.known_models`.
- **Codes.** A refusal for the generation alone is `409
  model_above_gpu_generation`, one for the class alone stays `409
  model_above_card_class`, and one for both is `model_above_gpu_generation`,
  its message naming both, since a larger card of the same generation would not
  load it either. The message names what the entry needs and the card: its
  name, compute capability and generation. The model lists report such an
  entry `available: false` with its `reason`, with no new field.
- **The built-in values.** `blackwell` on every NVFP4 artifact (`qwen3.8-27b`,
  `qwen3.8-27b-abliterated`, `swift-1.5-27b`), on `qwen3.8-flash-next` and on
  `ignis-image-2.1-turbo`, and on a trellis build until it has been verified
  below Blackwell (`swift-1.5-27b-16g`); none on `qwen3.8-27b-16g`,
  `qwen3.8-27b-abliterated-16g`, the speech models and the diarizer. On an RTX
  3090 or RTX 4090, `qwen3.8-27b` and `qwen3.8-27b-abliterated` are served by
  their 16 GB entries through their aliases.
- **The route checks stay** as the backstop, as the VRAM plan's refusal stays
  beside the class: a load with no route on the card is refused by name,
  whatever a catalog says.

The same spec settles the class side, recorded here with it. The card class
gains a **24 GB class**, 20 GiB up to but not including 28 GiB of total memory
(the RTX 3090 and RTX 4090), for an order of 16 < 24 < 32 < 96; a catalog's
`card_class` accepts 24, and no built-in entry states it. `--profile auto`
picks a new built-in profile, **`rtx-24gb`**, for it, and the 16 GB profile is
named **`rtx-16gb`**, with `rtx50-16gb` kept as a permanent alias on every
surface that names a profile (#476, #480). The class stays memory alone: the
generation decides routes and what an entry needs, never the class.

## Considered options

- **Fold the generation into the class** (an "RTX 30 24 GB" class). Rejected:
  the two are independent. A 24 GB Blackwell card loads exactly what an RTX
  3090 loads by memory and more by generation, and an Ada card whose total is
  no class still needs the alias to the 16 GB entry.
- **Rely on the route checks alone.** Rejected: they run at the load, after a
  switch has torn the loaded model down, and the lists and the Server view
  would offer models the card cannot run.
