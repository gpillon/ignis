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

## Unloading and loading on request

A client can give the card back without waiting for the idle
time, and take it again without sending a generation: another program on the
same card, a video encoder say, runs between the two calls while the server
stays up. Both routes are ignis extensions, not OpenAI's model deletion: no
file is touched. They take the active model's id, the one `GET /v1/models`
lists first, under the same API key as inference; a call refused for its key
loads and unloads nothing. Neither selects another model, which is what a
[switch](model-switching.md) does, and neither touches the
[speech model](transcriptions.md).

```bash
curl -X POST http://127.0.0.1:8000/v1/ignis/models/unload \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-16g"}'
curl -X POST http://127.0.0.1:8000/v1/ignis/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-16g"}'
```

With `--server-api-key`, add `-H "Authorization: Bearer $IGNIS_SERVER_API_KEY"`.

`POST /v1/ignis/models/unload` unloads the model the way the idle clock does,
and answers once the teardown is done, so the card is free when the answer
arrives: `200 {"model":"qwen3.8-27b-16g","status":"unloaded"}`. It never ends
a request: one admitted, waiting or running on the model refuses it `409
model_busy`, and so does a load, a reload or another unload under way, or a
model not loaded from an artifact, which could not be loaded again; nothing is
changed. A model unloaded already, or whose load out of `unloaded` failed, has
nothing resident, and the answer is the same `200` at once. The next request
that needs the model loads it again, as after an idle unload, and the unload
is logged as one (`ignis.model.unloaded`, without `idle_secs`).

`POST /v1/ignis/models/load` loads the model out of `unloaded` the way the
next request would, and answers once it serves, without generating:
`200 {"model":"qwen3.8-27b-16g","status":"serving"}`. A model serving already
is answered at once and is not loaded again. Calls and requests arriving
during the load wait for the same load. A load that fails is `503
model_load_failed` with its reason, and the next call or request tries again;
a switch or an unload begun between the load and the answer is `409
model_busy`.

On either route, an id other than the active model's is `404
model_not_found`, and a body that does not parse, or carries a field other than
`model`, is `400 invalid_json`. While a switch runs, both are answered `503
model_switching` with `Retry-After`, as every other route is. These are a full
teardown and a cold load, not a VRAM-only suspension that keeps the KV: the
first request after the load finds no retained prompt state.

`GET /v1/models/{model}`, OpenAI's retrieve, answers the entry `GET
/v1/models` lists under the id. Like the listing it neither loads
the model nor counts as its use, so polling it does not keep the model loaded;
an id the listing does not name is `404 model_not_found`.
