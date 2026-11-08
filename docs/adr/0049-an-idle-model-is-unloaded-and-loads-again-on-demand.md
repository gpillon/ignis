---
status: accepted
amended-by: [ADR-0067]
---

# ADR 0049 — an idle model is unloaded, and the next request loads it again

## Status

Accepted (2026-10-12, owner). GitHub #317, spec `docs/specs/model-switch/02-idle-unload.md`.
Builds on the runtime model switch (spec model-switch/01, #305) and on
config v2 (ADR 0046) for the knob.

**Amended by ADR 0067** (2026-10-24): a transcription is not use of the text
model and does not reset its idle clock. Opening a Realtime transcription
session loads an unloaded text model in the background, since the dictation
will land in its prompt.

## Context

A loaded model holds the 5090 whether anyone uses it or not: tens of GB of
VRAM, the pinned host pools and the KV-RAM arena. The card is shared (ninfer,
experiments, other tools), and today the only way to give it back is to stop
the server, which also takes the HTTP API, the metrics and the Playground
down with it.

The switch machinery (#305) already knows how to tear a model down to
nothing and load one from its artifact while the process and the API stay
up. What is missing is a state in which *no* model is loaded on purpose, and
a rule for leaving it.

## Decision

1. **Idle unload.** After `model.idle_unload` seconds with no inference
   request in flight and none admitted, the server tears the model down the
   way a switch does (drain, then the model thread drops everything). The
   process, the HTTP listener and `/metrics` stay up. Default **300 s**;
   **0 never unloads**. The value is patchable live (no reload).
2. **What counts as use.** A request on a route that needs the model —
   anything the serving gate would hold — counts, and so does any work still
   waiting or running on the engine (a long stream keeps the model loaded).
   The routes that only report state (`GET /v1/models`, the config and
   catalog reads, `/metrics`, preflights) do **not** count: a Playground tab
   polling them must neither keep the model loaded nor load it again.
3. **A new status, `unloaded`.** `GET /v1/models` keeps listing the model
   (its id is what a client names) with `status: "unloaded"`. Nothing is
   lost from the configuration: the running config, the switch tables and
   the model's source are kept.
4. **Loaded again on demand.** The first authenticated request that needs
   the model loads it again — the model it names when that is a known one,
   else the one that was unloaded: one load either way, since nothing is
   left to tear down — through the switch's own load-and-warm, and is then
   served. While it loads the status is `loading`. Requests that arrive
   during that load **wait for it** (they are not refused with `503`,
   unlike a switch someone else began): the first request after an idle
   period simply takes as long as a cold load. An explicit
   `POST /v1/models/switch` (to the same model or another), or a
   `PATCH /v1/config` that needs a reload, also leaves `unloaded` by
   loading. A load that fails leaves `failed`; there is no previous model
   to restore.
5. **Not a switch.** An unload and the load after it are logged
   (`ignis.model.unloaded`, `ignis.model.reloaded`) but are not recorded as
   `last_switch`: that field tells an operator what the last *switch* did.

## Consequences

- The first request after an idle period pays a full cold load (13–33 s on
  the 5090) and the retained prompt state (prefix reuse, KV-RAM) is gone.
  This is the trade: the card back for whoever else needs it, against one
  slow request.
- Benchmarks and gates must not meet a cold reload mid-measurement: the
  Make targets that start a server for measurement set `model.idle_unload`
  to 0.
- 0 is the escape hatch for anyone who wants the old behaviour.

## Rejected

- **Unloading to host RAM** (keep weights pinned, drop only VRAM): faster
  wake, but the host pools are part of what the card's other users need, and
  it is a second load path to maintain.
- **Refusing requests while unloaded** (`503` until an explicit load): every
  client would need retry logic for a state the server can leave by itself.
- **Exiting the process** after the idle period: loses the API, the metrics
  and the Playground, and needs a supervisor to come back.
