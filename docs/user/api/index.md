---
title: "The API"
description: "The OpenAI-compatible HTTP surface: every endpoint, probes, keys and errors."
sidebar:
  order: 1
---

An OpenAI-compatible HTTP server. The `/v1` routes sit behind the API key when
one is set; the Playground's static pages and the API reference at `/v1` do
not.

| Endpoint | Method | Notes |
|---|---|---|
| `/v1/models` | GET | The servable models: `data[0]` the loaded one (`active: true`; the speech model, when it can serve, comes right after it as `kind: "speech"`, and the top-level `speech` object says whether it can and why not, and in its `diarizer` member the same of the diarizer — [Speech to text](transcriptions.md)) — for a text model with the context it runs (`max_model_len`), the requests it decodes at once (`decode_lanes`), the steps its load took to fit the card's VRAM (`adapted`, `""` when none; [when a profile does not fit](../concepts/profiles.md#when-a-profile-does-not-fit)) and the prefill chunk a prompt takes while no lane decodes (`prefill_chunk_solo`, [the solo width](../configuration/context-and-speculation.md#the-solo-width)) — then every model a request naming it would switch to and be served by now — each catalog entry on disk, each known model whose artifact is a file; the loaded one alone with `--switch-allow-implicit false`. A catalog entry on disk the card cannot load, by its class or its GPU generation, is listed `available: false`, with the `reason` ([the catalog](../guides/models.md#the-catalog)). And `status`, about `data[0]`: `serving`, `switching`, `failed` ([Switching models](model-switching.md)), `unloaded` or `loading` ([idle unload](idle-unload.md)). |
| `/v1/models/switch` | POST | Replace the loaded model on the running process ([Switching models](model-switching.md)). |
| `/v1/models/catalog` | GET | Every model the server can switch to or fetch, with its state on disk ([Switching models](model-switching.md)). |
| `/v1/models/download` | POST | Download a catalog entry in the background while the server serves; the catalog's rows follow it ([Downloading from the Server view](../guides/models.md#downloading-from-the-server-view-post-v1modelsdownload)). |
| `/v1/models/download/cancel` | POST | Stop the running download, keeping its partial files for the entry's next download to resume ([Cancelling and resuming](../guides/models.md#cancelling-and-resuming-post-v1modelsdownloadcancel)). |
| `/v1/models/remove` | POST | Delete a catalog entry's files, finished or partial, from the models folder; never the model loaded, loading or being downloaded ([Removing a model](../guides/models.md#removing-a-model-post-v1modelsremove)). |
| `/v1/images/generations` | POST | Generate images with the image model ([Image generation](images.md)). |
| `/v1/images/edits` | POST | Edit images from 1 to 10 condition images, multipart ([Image generation](images.md)). |
| `/v1/audio/transcriptions` | POST | Transcribe audio with the speech model, multipart ([Speech to text](transcriptions.md)). Served on the CPU through a model switch. |
| `/v1/realtime?intent=transcription` | GET | A live transcription session over a WebSocket, OpenAI's Realtime transcription subset ([Live transcription](realtime-transcription.md)). Served on the CPU through a model switch. |
| `/v1/config` | GET, PATCH | The running configuration, and a change to it ([configuration](../configuration/index.md)). |
| `/v1/config/fields` | GET | What each visible configuration field is: kind, default, rule, families, whether a change reloads. |
| `/v1/config/origins` | GET | Where each visible value in force comes from, shared and per model family: default, profile, card, file, env, flag or live. |
| `/v1/chat/completions` | POST | Chat completions — streaming (`stream: true`, SSE) and non-streaming; an answer held to a JSON object or to your JSON schema with `response_format` ([Structured output](structured-output.md), from 0.11.0). |
| `/v1/responses` | POST | The OpenAI Responses API, streaming (`stream: true`, SSE) and non-streaming ([The Responses API](responses.md)). |
| `/v1/responses` | GET | Its WebSocket mode: the upgrade, then `response.create` events ([The Responses API](responses.md)). |
| `/v1/tokenize` | POST | A body's prompt-token count without serving it ([Counting a prompt](count.md)). |
| `/v1/detokenize` | POST | Token ids back to text, through the same tokenizer. |
| `/v1/decide` | POST | The decision endpoint ([Decisions](decide.md)). |
| `/v1/systemone` | POST | The same handler on the System One API's own route ([Decisions](decide.md)). |
| `/v1` | GET | The API reference — a 307 to `/v1/docs/` (ADR 0036). |
| `/v1/docs/` | GET | Swagger UI over the document, from assets compiled into the binary: no CDN, nothing fetched. Open even with an API key set — it publishes the API's shape, not its load — and *Authorize* is how you drive the keyed routes from the page. |
| `/v1/openapi.json` | GET | The OpenAPI 3.1 document, generated from the handlers themselves. Monitoring is not in it. |
| `/ui/` | GET | The Playground — unless `--server-ui false`. |
| `/ui/metrics` | GET | Prometheus text format 0.0.4 for the Playground — with the Playground, unless `--server-metrics false`. Behind the API key, like `/v1`. |

Every `/v1` route answers a CORS preflight. Prometheus itself scrapes
`GET /metrics` on the metrics listener (`--server-metrics-bind`), not on the API's
address: no key, and never reachable through `--server-expose`.

**Request body limits**, enforced before JSON parsing: 16 MiB on a text-only
load, 384 MiB with `--vision-enabled`, which takes images inline as base64 data URIs.
Both are past what their own load can use, so an oversized prompt meets the
refusal that names the real limit — the context — rather than a byte count.

**Errors** use OpenAI's `{"error": {message, type, code}}` body with the
matching status: 400 bad request, 404 unknown model, 413 oversized request, 503
engine full, 504 a request past its deadline that is answered an error
([deadlines](deadlines.md)).

**A full engine** (every lane in use) answers `503` with type and code
`engine_full` and a `Retry-After`, on `/v1/chat/completions` and on HTTP
`/v1/responses` alike. Over HTTP nothing waits: the client's retry is the
admission. The header is an estimate of when a lane frees: the oldest request
in flight plus how long requests have been holding a lane (a running average
of the ones that finished, were cancelled or stopped, each moving it 1/8 of
the way). It is whole seconds rounded up, clamped to 1..59, plus a one-second
margin, so 2 to 60 (60 being the most the OpenAI SDK honours). Once that time
has passed the oldest request is overdue, and a lane is expected one average
hold after the refusal instead, at most 15 s (so 2 to 16): the average has
already proved wrong for that request, and a client that polls sooner leaves a
freed lane idle for less. It is `1` before any request has finished. Only the
Responses API's [WebSocket mode](responses.md) queues a request instead.
A request that stops being waited for gives its lane back at once, whether it
hit its deadline or its client went away.
