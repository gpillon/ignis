---
title: "Switching models"
description: "Explicit and implicit switches between the served models, and what a request sees during one."
sidebar:
  order: 5
---

`POST /v1/models/switch` with `{"artifact": "<path>", "model": "<id>"}` — or
with `{"model": "<id>"}` alone, see below — moves
the running server to another model — the 27B to Qwen3.8-Flash-Next and back —
without a restart. It answers `202 {"status": "switching", "from", "to"}` at
once; a second switch while one runs is a `409 switch_in_progress`.

A switch is a full reload: the old model is unloaded before the new one loads,
so the two are never on the card together. While it runs, every other `/v1`
route answers `503 model_switching` with `Retry-After: 5` (but
`POST /v1/audio/transcriptions` and `GET /v1/realtime`, which the [speech model](transcriptions.md) serves on the CPU through a switch, and `POST /v1/models/download/cancel`, which [stops a download](../guides/models.md#cancelling-and-resuming-post-v1modelsdownloadcancel) without touching the loaded model), and `GET /v1/models`
reports `status: "switching"` with `switching: {from, to}`. Requests already
running on the old model get `--switch-drain-timeout` (30 s) to finish; one
still running then ends with an `engine_error`. The new model loads on the
flags the server started with, except the ones it cannot take, which are left
off for it and named in the log (the 27B's `--spec-backend` on Flash-Next;
Flash-Next's own knobs on the 27B).

**Vision crosses the switch.** A server started with `--vision-enabled` serves
images on Flash-Next as on the 27B, from the 27B to Flash-Next and back, and
the request size cap follows the model that is serving. Flash-Next sees
through its [vision companion](../configuration/vision.md#flash-next), a file beside its
artifact: a switch to Flash-Next with vision on and the companion missing, or
not its own, is a `409 vision_companion_missing` that names the file. It is
answered before anything is unloaded, so the old model keeps serving.

A target that can be refused without unloading anything — no file at the path,
a checksum that is not clean, a model id of the other model — leaves the old
model serving. One that fails to load after the old model was unloaded has the
old model loaded again; `GET /v1/models` says `failed`, with a `reason`, until
one of them serves.

**By id.** With `model` alone the server finds the artifact itself: the
`--switch-known-models` entry for the id (the model the server started on
among them), else a [catalog](../guides/models.md#the-catalog) entry whose artifact is under
`--download-path`. `--switch-allow-implicit` does not apply: it gates the
switch a request triggers by naming a model, not one an operator asked for. An
id that is neither, or whose artifact is not on disk, is a `404
model_not_found` naming the id — and, for a catalog entry, the `ignis-server
model download <id>` that fetches it; a switch never starts a download.
`GET /v1/models/catalog` lists the models a switch can go to, one row each:
`id`, `source` (`built-in`, `operator` or `known`), `on_disk` (`complete`,
`partial` or `absent`), `bytes`, `family`, `kind`, `loaded` and
`switchable`. `kind` is the model kind as `GET /v1/models` names it (`text`,
`image`, `speech` or `diarizer`): the artifact header's when the file is on
disk, which wins, else what the catalog tells of the entry, its `kind` or its
family's, so every built-in entry has one before it is downloaded; it is absent
only when neither tells, as for an operator entry of no family not yet
downloaded. A catalog entry's row also carries `license`, `{id, url}`: the
license its weights are published under, as the hub's cards name it, and a link
to its text ([the catalog](../guides/models.md#the-catalog)); it is absent on a
known model and on an operator entry that states none, and `GET /v1/models` is
unchanged. It
answers during a switch and while one has failed, and no row carries a path.
Beside the rows the list carries `disk: {path, free_bytes}`: `--download-path`
as the server holds it and the bytes free on its volume, read where a download
checks them before it begins (at the nearest folder above it that exists, since
the first download creates it); absent when they cannot be read. Each
row also carries `bytes_on_disk`, what of it is there now, partial files
included, and `download`: `running` while a
[download through the API](../guides/models.md#downloading-from-the-server-view-post-v1modelsdownload)
fetches it, `failed` with a `download_reason` once one failed, else `none`.
A download refused for want of room on the disk (`507 download_no_room`) is
kept the same way, as the entry's `failed` with its reason. A complete row
reads no failure: one kept from before the entry was completed another way
(`ignis-server model download`) is forgotten.
An entry that `ignis-server model download` is fetching into `--download-path`
reads `download: none`, with its `bytes_on_disk` growing: the command's fetch
is not the server's, and a download of the entry through the API is refused
with `409 download_locked` until it ends
([one writer per entry](../guides/models.md#one-writer-per-entry)).
The row of an entry the server started on, from one of its files under
`--download-path`, names that model in `start_model`, and the row of an entry
`--switch-known-models` names at one of its files lists those ids in
`known_models`, whatever id each is served under: what
[removing it](../guides/models.md#removing-a-model-post-v1modelsremove) takes
away.

**How the last one ended.** After the first switch or reload since start,
`GET /v1/models` carries `last_switch: {from, to, outcome, reason?, host_plan?}`:
`outcome` is `switched`, `refused` (nothing was torn down), `restored` (the
old model was loaded again) or `failed` (nothing serves), with a `reason` for
every outcome but the first. A reload from `PATCH /v1/config` is reported the
same way, with `from` equal to `to`. When a text model's host plan
([Flash-Next's](../configuration/host-ram.md#host-ram-for-flash-next), [the 27B's](../configuration/host-ram.md#host-ram-for-the-27b))
refused the target, `host_plan` holds
it in exact bytes for an agent (`available_bytes`, `margin_bytes`,
`planned_bytes`, `needed_bytes`, `short_bytes`, `crossing_line`, and every
line as `{name, bytes}`), while `reason` says it for a person.

**By naming the model.** A client can also just ask for the other model: a
chat completion, a response (over HTTP or the WebSocket) or a decision whose
`model` names another model listed with `--switch-known-models` runs the same switch,
and the request itself is held until the named model serves and is then
answered by it — the first such request takes as long as the switch does
(seconds), and no client-side retry loop is needed. List the models once:

```
ignis-server --model-artifact models/Qwen3.8-27B.ninfer \
  --switch-known-models qwen3.8-flash-next=models/Qwen3.8-Flash-Next.ninfer
```

The model the server starts on is always listed, so naming it again switches
back. While such a switch runs every other request gets the same `503
model_switching` as during any switch, a second request naming a model
included — switches are never queued. A switch that does not land refuses the
request that began it with `503 model_switch_failed` and the reason (no
`Retry-After`: each retry would unload and reload the serving model again),
never an answer from another model; `/v1/decide` refuses it as its own `422`.
A model nobody listed, a listed one whose artifact is not a file (looked up at
each request: put the file back and it switches), or any other model under
`--switch-allow-implicit false`, is a `404 model_not_found`, and no switch
begins — exactly the ids `GET /v1/models` does not list. `/v1/tokenize` never switches: counting
tokens does not move the card.
