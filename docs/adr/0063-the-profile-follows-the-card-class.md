---
status: accepted
---

# The profile follows the card class, and its defaults adapt to the budget

`--profile` defaults to `auto`, which picks the built-in profile by the card's **card class**, its total memory
alone: the profile used to be chosen by name only, the RTX 5090's by default, and nothing read the card, so a 16 GB
card would have started under the 5090's profile without a word. And since the memory a card has free at start
varies with what else holds it, a 27B load whose VRAM plan (ADR 0030) refuses what its profile or the hardcoded
defaults gave lowers those values in a fixed order until the plan fits, the **adaptation**, where it used to be
refused with the byte deficit.

**The class and its profile**

- There is one class, with one built-in profile, for the 12, 16, 24, 32 and 96 GB cards. A total of no class is
  refused, naming `--profile`; a named profile always wins, `none` included. A profile per card of a class comes
  only if real-card measurements justify it.
- The class is memory alone, and the GPU generation is a second axis beside it (ADR 0079): RTX 30 and RTX 40 cards
  are in scope beside Blackwell, and the 24 GB class is the RTX 3090's and the RTX 4090's. The 12 GB class is a class
  of its own, with no lower bound (ADR 0098), so the 16 GB class no longer holds every smaller card. A catalog entry
  states the smallest class it loads on (ADR 0064).
- `rtx50-16gb`, the 16 GB profile's first name, is a permanent alias wherever a profile is named.
- What each built-in profile sets is the code's. Every value of a profile is a default: the operator overrides it,
  and the load adapts it.

**Adaptation** is the 27B's alone: Flash-Next, whose plan sizes an expert cache from the rest, adapts nothing. The
rule the ladder walks by is the opening comment of `crates/server/src/adaptation.rs`; what is decided:

- Only a value the profile or the hardcoded default gave adapts, by its source alone, and only when the plan is
  refused below its minimum. A value the operator named, by a flag, an environment variable, a file or a live patch,
  never adapts and is refused as before, a file's value equal to the default included: a file `config generate`
  wrote names the numbers of the card it was generated on. That differs on purpose from ADR 0046's `explicit`, where
  a value equal to the default counts as unnamed. The one exception is a solo width the operator named, which may
  narrow to the prefill chunk.
- A shape built by hand, as every GPU gate's is, adapts nothing, so a gate the card cannot hold is refused, never
  served at a context it did not ask for.
- The order is fixed and puts speed before context length, on purpose: the solo width (ADR 0018), the drafter's
  proposal head, the context, then the decode lanes. DFlash2 is never switched off by adaptation.
- A context, and an `auto` solo width, is kept only where, at more than one lane, the KV pool still holds one full
  sequence and a spare reservation: room for a second request comes before a wider chunk.
- The context step, which is also the floor, and the spare reservation are config fields, which a profile may set
  (ADR 0098): a threshold that decides whether or how a load starts is a field, its value the default.
- A load names each step it took in the VRAM plan's log event, and a refusal must say which adaptations it tried.
  The values the load kept are the load's everywhere after the plan, an `auto` rope table resolved again for the
  context kept (ADR 0057); `GET /v1/config` keeps the values configured.

## Considered Options

- **Detect whether the card drives a display** and choose the context from it. Rejected by the owner: the free
  memory at start already says what the desktop holds, and adaptation answers it.
- **A profile per card of a class.** Memory-shaped fields are the same on every card of a class; bandwidth-shaped
  ones, the lanes and the draft tokens, need a real card to tune.
- **Refuse instead of adapting.** A small card's free memory varies run to run, and refusing what a smaller context
  would serve helps no one.

## Consequences

- A load may run a smaller context than its profile names, and says so in the log.
- The drafter's head `auto` is the full head on every card class, so the head's step never fires on a built-in
  profile: the shortlist head is bound beside the full one and frees no bytes, and the one card that measured it
  found it slower too. The rule is revisited when a real 16 GB card measures the trade.
- The 16 GB and 24 GB profiles pin the 27B's solo width to their prefill chunk: on those classes the KV pool comes
  before a wider chunk. `auto` stays the default for the RTX 30 and RTX 40 cards no profile pins.
- The 16 GB profile has vision on because under the `host` residency (ADR 0075) vision costs it no context while no
  image is in flight.
