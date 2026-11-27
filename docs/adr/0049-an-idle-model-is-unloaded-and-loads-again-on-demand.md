---
status: accepted
---

# An idle model is unloaded, and the next request loads it again

A loaded model holds the card whether anyone uses it or not: tens of GB of VRAM, the pinned host pools and the KV-RAM
arena. The card is shared with other programs, and the only way to give it back was to stop the server, which takes
the HTTP API, the metrics and the Playground down with it. So after `model.idle_unload` seconds with no inference
request in flight and none admitted, the server tears the model down the way a switch does, while the process, the
HTTP listener and `/metrics` stay up, and the first request that needs the model loads it again. The trade is the
card back for whoever else needs it, against one slow request: the first request after an idle period pays a full
cold load, 13-33 s on the RTX 5090, and the retained prompt state (prefix reuse, KV-RAM) is gone.

- **`0` never unloads**: the escape hatch for whoever wants a model that stays loaded. The value is patched live,
  with no reload.
- **What counts as use.** A request on a route that needs the model, anything the serving gate would hold, counts,
  and so does any work still waiting or running on the engine: a long stream keeps the model loaded. The routes that
  only report state (`GET /v1/models`, the config and catalog reads, `/metrics`, preflights) do not count: a
  Playground tab polling them must neither keep the model loaded nor load it again.
- **`unloaded` is a status, and nothing of the configuration is lost.** `GET /v1/models` keeps listing the model,
  since its id is what a client names, with `status: "unloaded"`; the running config, the switch tables and the
  model's source are kept.
- **Loaded again on demand.** The first authenticated request that needs the model loads it again through the
  switch's own load-and-warm, and is then served: the model it names when that is a known one, else the one that
  was unloaded, one load either way since nothing is left to tear down. While it loads the status is `loading`, and
  requests that arrive during that load wait for it: they are not refused with `503`, unlike a switch someone else
  began. An explicit `POST /v1/models/switch`, to the same model or another, and a `PATCH /v1/config` that needs a
  reload also leave `unloaded` by loading. A load that fails leaves `failed`: there is no previous model to restore.
- **Speech** ([ADR 0067](0067-the-speech-model-waits-on-the-host-and-borrows-free-kv-pages.md)). A transcription is
  not use of the text model and does not reset its idle clock, and opening a Realtime transcription session loads an
  unloaded text model in the background, since the dictation will land in its prompt.
- **On request**
  ([ADR 0105](0105-a-client-may-unload-the-active-model-at-once-and-load-it-again-without-generating.md)), a client
  unloads the active model at once, into the same `unloaded` state, and loads it again without a request to serve.
- **Not a switch.** An unload and the load after it are logged but are not recorded as `last_switch`: that field
  tells an operator what the last switch did.

## Considered Options

- **Unloading to host RAM**, the weights kept pinned and only the VRAM dropped: a faster wake, but the host pools
  are part of what the card's other users need, and it is a second load path to maintain.
- **Refusing requests while unloaded**, `503` until an explicit load: every client would need retry logic for a
  state the server can leave by itself.
- **Exiting the process after the idle period**: it loses the API, the metrics and the Playground, and needs a
  supervisor to come back.

## Consequences

- Benchmarks and gates must not meet a cold reload mid-measurement: a server started for a measurement runs with
  `model.idle_unload` at 0.
