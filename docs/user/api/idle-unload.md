---
title: "Idle unload"
description: "The server unloads an idle model to free the GPU, and loads it again on the next request."
sidebar:
  order: 6
---

A model nobody uses gives the card back. After `--model-idle-unload` seconds
(file key `model.idle_unload`, default `300`) with no request in flight and
nothing waiting or running on the engine, the server unloads the model the way
a switch does — VRAM, pinned host memory and KV-RAM all freed — and stays up:
the API, `/metrics` and the Playground keep answering, and `GET /v1/models`
lists the same model with `status: "unloaded"`. Meanwhile `/metrics` shows
nothing reserved: the VRAM-plan, KV pool and KV-RAM gauges read zero until the
next load. A switch sent during the seconds the unload tears the model down is
refused `409 switch_in_progress` naming the unload; send it again once
`GET /v1/models` says `unloaded`.

The next request that needs the model — a chat completion, a response (over
HTTP or on a WebSocket opened before the unload), a decision — loads it again
and is answered once it is warm, so the first request after a quiet spell takes
as long as a cold load (13–33 s on the 5090) and finds no retained prompt
state. While it loads `GET /v1/models` says `status: "loading"`, and requests
arriving meanwhile wait for the same load rather than being refused. A request
naming another model the server lists loads that model instead: one load,
never two. A load that fails leaves `status: "failed"` with the `reason`, the
requests that waited on it get `503 model_load_failed`, and the next request
tries again; `POST /v1/models/switch` (to the same id or another) and a
`PATCH /v1/config` that needs a reload load a model too. A request without the
API key is refused `401` as always and loads nothing.

Only use counts: a request is in use from its admission to the end of its
response (a long stream keeps the model loaded); polling `GET /v1/models`, the
config and catalog reads or `/metrics` does not, so a Playground left open
neither keeps the model loaded nor loads it again. A transcription does not
count either: the [speech model](transcriptions.md) is another model, on the CPU,
with its own idle time (`speech.idle_unload`), and it is served with the text
model unloaded without loading it. An unload and the load a
request begins after it are logged (`ignis.model.unloaded`,
`ignis.model.reloaded`) and are not a switch: `last_switch` is left as it was.
A `POST /v1/models/switch` or a reloading `PATCH /v1/config` while unloaded is
still a switch or a reload, and is recorded in `last_switch` like any other.

`0` never unloads. The value is patchable live — `PATCH /v1/config` with
`{"model": {"idle_unload": 600}}` takes effect at once, counting from the
change. A measurement should not meet a cold reload: the Makefile's
`IDLE_UNLOAD=0` passes `--model-idle-unload 0`, and the A/B and TTFT session
scripts do so themselves.


## Load and unload on demand

`GET /v1/models/{model}` retrieves one entry from `GET /v1/models` without
loading it or extending the idle timer. An unknown id returns `404 model_not_found`.

The lifecycle routes below are Ignis extensions, not OpenAI model deletion.
They use the same API key as generation requests and operate on the exact id
of the current text or image model. They never delete artifact files or unload
the separate CPU speech model.

```bash
curl http://localhost:8000/v1/models/qwen3.8-27b-16g \
  -H 'Authorization: Bearer YOUR_API_KEY'
curl -X POST http://localhost:8000/v1/ignis/models/unload \
  -H 'Authorization: Bearer YOUR_API_KEY' -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-16g"}'
curl -X POST http://localhost:8000/v1/ignis/models/load \
  -H 'Authorization: Bearer YOUR_API_KEY' -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-16g"}'
```

Unload waits for teardown and answers `200` with
`{"model":"qwen3.8-27b-16g","status":"unloaded"}`. If a request is admitted,
queued or running, a switch or reload holds the model, or the model has no
reloadable artifact, it answers `409 model_busy` without cancelling work.
Already unloaded is a successful no-op. A later generation request automatically
loads the model again; this route does not lock it out of service.

Load waits until the current model is warm and answers `200` with
`{"model":"qwen3.8-27b-16g","status":"serving"}`. Already serving is a no-op.
It does not generate tokens. A failed reload returns `503 model_load_failed`;
concurrent loads share the same reload. Use [model switching](model-switching.md)
to select another model: a different id on either lifecycle route returns `404`.
These are full unloads and cold reloads, not a VRAM-only suspension retaining KV.
