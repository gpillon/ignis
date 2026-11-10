---
title: "Model switching and idle unload"
description: "Why ignis holds one model at a time, what a switch and an idle unload do to the card, how a request can trigger either, and what a client sees while a model loads."
sidebar:
  order: 7
---

ignis serves one model at a time. A different model is reached by a **switch**,
which replaces the loaded one without restarting the process, and a model nobody
uses can be **unloaded** to give the card back and loaded again by the next
request. Both rest on the same machinery: tear the model down completely, bring
one up from its artifact, and keep the process, the API and the Playground
running in between. The routes and fields are in
[Switching models](../api/model-switching.md) and [Idle unload](../api/idle-unload.md).

## Why one model at a time

A load carves the card's whole budget into one plan, and the model gets all of
it: its KV pool, or on Flash-Next its expert cache, takes what the weights leave
([the VRAM plan](vram-plan.md)). Two models would each need a plan, and the
second would have to be sized from what the first left. So a switch is a **full
reload**: the old model is unloaded before the new one loads, and the two are
never on the card together. The same holds for the image model, which is reached
by a switch and never loaded beside a text model.

The consequence is that a switch costs a load: seconds to tens of seconds. On
the RTX 5090 a cold load takes 13 to 33 seconds, and the 27B to the image model
measured 10.0 s, and back 11.3 s. Whatever the old load held is gone with it:
VRAM, pinned host memory, KV-RAM, and the prompt state kept for reuse
([the KV cache](kv-cache-and-prompt-reuse.md)).

## Explicit and implicit switches

There are three ways to start a switch:

- **Explicitly**: `POST /v1/models/switch` with an artifact path, or with just a
  served id, in which case the server finds the artifact itself among the models
  it knows. It answers `202` at once and does the work in the background.
- **By naming a model.** A chat completion, a response or a decision whose
  `model` names another known model runs the same switch, and the request is
  held until that model serves and then answered by it. The client needs no retry
  loop; the first such request takes as long as the switch does. This is the
  *implicit* switch, controlled by `--switch-allow-implicit` (on by default). The
  models it can reach are those named in `--switch-known-models` and the catalog
  entries whose artifacts are on disk.
- **By a configuration change** that the loaded model's load is built from:
  `PATCH /v1/config` reloads the model with the whole patch, the way a switch
  does ([configuration](../configuration/index.md)).

A switch never starts a download. A catalog model that is not on disk is a
`404 model_not_found` that names the command to fetch it
(`ignis-server model download <id>`).

### The pitfall of two names

An implicit switch is convenient until a client names two models. A client
configured with a main model and a separate "small" model for titles and
summaries, for instance, alternates between them, and the server reloads back and
forth, 13 to 33 seconds each time. The same happens to a front end that picks
its model ids from `GET /v1/models`, which lists every servable model, when a
background task uses a different id than the chat. The fix is to point both at
one id, or to start with `--switch-allow-implicit false`, which makes a request
naming another model a `404`
([get a model](../guides/models.md)).

The Playground avoids the trap: its chat and Decide requests name no model, so
the server serves the one that is loaded, and it switches only when asked to.

## What a client sees

During a switch the server is up and answering, but not serving:

- Every other `/v1` route answers `503 model_switching` with `Retry-After: 5`.
  Switches are never queued; a request naming a model while one runs gets this
  too.
- `GET /v1/models` reports `status: "switching"` with `switching: {from, to}`.
- Requests already running on the old model get `--switch-drain-timeout`
  (30 seconds) to finish; one still running then ends with an `engine_error`.

A second explicit switch while one runs is a `409 switch_in_progress`. A switch
that cannot proceed and has torn nothing down yet (no file at the path, a
checksum that is not clean, a model id that belongs to the other model) leaves the old model
serving. One that fails after the old model was unloaded loads the old model
again, and `GET /v1/models` says `failed` with a reason until one serves.

Vision crosses a switch: with `--vision-enabled` the server takes images on
the 27B and on Flash-Next, and its request size cap follows the model that is
serving. A switch to Flash-Next whose vision companion is missing beside its
artifact is a `409 vision_companion_missing` naming the file, answered before
anything is torn down. Vision is each model's: a built-in profile turns it on
for the 27B alone, so a switch from the 27B to Flash-Next under it leaves
Flash-Next without vision, its default, and says so
(`ignis.model.switch_vision_off`, [Model switching](../api/model-switching.md)).

How the last one ended is kept on `GET /v1/models` as `last_switch`: `switched`,
`refused` (nothing was torn down), `restored` (the old model was loaded again) or
`failed` (nothing serves), with a reason. For a refusal by the host RAM plan, the
reason carries the numbers in bytes so an agent can read them.

An implicit switch whose model does not land refuses the request that began it
with `503 model_switch_failed` and no `Retry-After`, because each retry would
unload and reload the serving model again. It never answers from another model.

## Idle unload

A loaded model holds the card whether or not anyone uses it: tens of GB of VRAM,
pinned host pools and the KV-RAM arena. The card is often shared with other
programs, and until idle unload the only way to give it back was to stop the
server, which took the API, the metrics and the Playground down with it
([ADR 0049](../../adr/0049-an-idle-model-is-unloaded-and-loads-again-on-demand.md)).

After `--model-idle-unload` seconds (300 by default; `0` never unloads) with no
request in use and nothing waiting or running on the engine, the server unloads
the model the way a switch does. The process stays up. `GET /v1/models` lists the
same model with `status: "unloaded"`, and `/metrics` shows nothing reserved: the
VRAM plan, KV pool and KV-RAM gauges read zero until the next load.

### What counts as use

A request counts from its admission to the end of its response, so a long stream
keeps the model loaded. Routes that only report state do not count: `GET
/v1/models`, config and catalog reads, `/metrics`. A Playground tab left open
polls them, and it neither keeps the model loaded nor loads it again. A request
refused for a missing or wrong API key loads nothing.

### The next request

The first request that needs the model loads it and is answered once it is warm.
It takes as long as a cold load (13 to 33 seconds on the 5090) and finds no
retained prompt state. While it loads, `GET /v1/models` says `status: "loading"`,
and requests that arrive meanwhile wait for the same load rather than being
refused, unlike a switch someone else began. A request naming another known model
loads that model instead, in one load. A load that fails leaves `status:
"failed"` and the requests that waited get `503 model_load_failed`; the next
request tries again.

An unload and the load after it are logged (`ignis.model.unloaded`,
`ignis.model.reloaded`) but are not switches: `last_switch` is left as it was.

### Why this design

The decision weighed three alternatives and rejected each:

- **Unloading only VRAM and keeping the weights in pinned host memory** would
  wake faster, but the host pools are part of what the card's other users need,
  and it is a second load path to maintain.
- **Refusing requests while unloaded** until an explicit load would force every
  client to carry retry logic for a state the server can leave by itself.
- **Exiting the process** loses the API, the metrics and the Playground, and
  needs a supervisor to come back.

The trade is one slow request after a quiet spell, against the card back for
whoever needs it. `--model-idle-unload 0` keeps the old behavior. Anything that
measures the server should set it: a benchmark must not meet a cold reload
mid-run, and the Makefile's `IDLE_UNLOAD=0` does so.

On a platform that scales a pod to zero, an unloaded model and an idle
connection are different things: an open Responses WebSocket keeps a Knative
pod from scaling down, which is why `--server-socket-idle-close` exists
([Knative](../guides/knative.md)).
