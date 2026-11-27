---
status: accepted
---

# A catalog entry states the smallest card class it loads on; aliases resolve within a family; a switch refuses before teardown

A request naming a model that is not loaded switches to it, if its artifact is on disk, and the VRAM check comes
after the old model is torn down: a target that does not fit fails, the old model is reloaded and the request gets
`503 model_switch_failed`. On a 16 GB card every request naming a model the card cannot hold, Flash-Next below its
expert-cache floor or the image model at about 17.6 GiB of budget, would cost a full reload of the 27B. And the
clients ignis has, a user's agents, other OpenAI-compatible clients and the Playground (ADR 0084), are configured
with a family's default id, whose entry on a small card is an artifact that does not fit. So the catalog
(ADR 0047) says what a card loads, before anything moves:

1. **A catalog entry states the smallest card class it loads on** (the classes are ADR 0063's), and an operator
   entry may state its own. It is optional in both catalogs, by one rule: an entry that states none is never
   refused by class, and the VRAM plan's refusal still guards it, so an operator catalog written before card
   classes still starts. Every built-in text and image model states one; the built-in speech models (ADR 0067) and
   the diarizer (ADR 0069) run on the CPU and state none, so no card refuses them. A number that is no card class
   refuses the catalog. A class may be a policy, not a hardware limit: Flash-Next states the 32 GB class though,
   with `vram.allow_expert_cache_below_floor`, it starts below its expert-cache floor, with a warning.
2. **The artifact built for a smaller class has its own served id, and may answer to another id as an alias**: its
   family's default id, or another entry's served id of the same family. On a card that cannot load an id's own
   entry, a request naming that id is served by the entry of its family that lists it as an alias, so a client's
   configuration does not change with the card. An alias only ever resolves within one model family, and only to
   an entry the card can load. It never causes a switch to one it cannot.
3. **A switch, explicit or implicit, a start and a download check the card first.** An entry the card cannot load,
   with no entry answering to its id there, is refused before anything is prepared, torn down or fetched, with a
   message naming the entry's class: `409 model_above_card_class` on every route that names a model, as
   `model_kind_mismatch` is, and `422` on `/v1/decide`, as its other model refusals are. The status never leaves
   `serving`, and no `last_switch` is written. The model lists keep such an entry, `available: false` with a
   `reason`, which names the entry that answers to its id here when one does. The VRAM plan's own refusal still
   guards the case the class check cannot see: a budget too small on a card of the right class.

**An entry fits a card by its class and its GPU generation together** (ADR 0079): the rule, the alias and the
refusal take both, an alias answers only where its entry fits both, and a refusal the generation causes carries
that ADR's code.

**One rule and one resolve step, which every caller takes.** An id stands when its own entry fits the card, or
states nothing the card is held to; else the entry of its family that lists it as an alias and fits answers to it;
else it is refused, naming what the entry needs and the card. Entries of different card classes may answer to one
id: where several fit the card, the one of the largest class answers, never the catalog's order, and each loads by
its own id on every class it fits. A card of no known class refuses nothing by class.

**The class is the catalog entry's, whoever names its id**: a start with no artifact named, a switch by id, a
request's implicit switch, `model download`, and a catalog id the operator also listed in `switch.known_models` or
started on by path, which is fitted all the same: off the card's class it is unavailable, never switched to, and
the way back to it after a switch is refused. Only a path named for the load itself is the operator's word,
`--model-artifact` at start, where the loaded model's own id is then served whatever its class, and a
`POST /v1/models/switch` naming its `artifact`: only the VRAM plan guards those. An id that is no catalog entry,
the operator's own model, has no class and is never refused by one.

**The family.** The artifact's header is the authority (ADR 0047). The catalog tells an entry's family from its id
or its aliases, where one is a family's default id; otherwise the entry states `family`, the family's own id, and
must when an alias involves it, listed by it or naming its id. An entry no alias involves needs none, so an
operator catalog written before `family` still loads. The catalog's word is a claim the header verifies: the entry
that answers to an alias must hold an artifact whose header names the alias's family. A switch or a request checks
it before anything stops, when that artifact is on disk; a start checks it once the artifact is there, after the
fetch when it fetched it, before the load. An artifact of another family, or of none, does not answer to the
alias: the id is refused as its own entry is, naming both families.

**Who may claim an alias.** The catalog refuses an alias of another family than its entry, and one that two
entries of one card class claim within a layer. An operator entry claims no alias a built-in entry claims, whatever
its class; two of its own entries of different classes may answer to one id.

**An alias rewrites the request's `model`** to the serving entry's id before anything else reads it, and the
answer names the entry that served it; the loaded model's own id is served as it is, whatever its class. A start
that an alias resolves serves the entry under that entry's own id, as a switch to it would.

## Considered Options

- **One served id for both artifacts of a family.** The catalog refuses two entries with one served id, and the
  served id is what tells artifacts of one family apart (ADR 0047).
- **A distinct id and nothing else.** Every client configured with the family's default id would break on a
  smaller card.
- **Keep refusing after teardown.** It costs a reload per request, and the class is known before anything moves.
- **Reading an entry's family off its id's prefix.** An operator's id is a name, and a prefix match would be a
  coincidence the catalog trusted.

## Consequences

- Flash-Next's entry states the 16 GB class once it runs there, and the image model's after it. The way in is the
  16 GB class's profile (ADR 0063) lowering the expert-cache floor: under the rule that a refusal threshold is a
  config field, the floor becomes one, the profile's own value estimated first and tuned on hardware. A first
  estimate from the plan's lines is about 5 GiB: Flash-Next's fixed lines at one lane and a 262K pool are about
  9.65 GB, which leaves 5.4 to 7.1 GB on a 16 GB card, depending on what else holds it.
