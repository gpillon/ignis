---
status: accepted
amends: [ADR-0049]
---

# A client may unload the active model at once, and load it again without generating

[ADR 0049](0049-an-idle-model-is-unloaded-and-loads-again-on-demand.md) gives the card back only after
`model.idle_unload` seconds with nobody using the model, and leaves `unloaded` only when a request needs the
model, a switch or a reloading `PATCH /v1/config` loads one. A client sharing the card with another program, a
video encoder on a 16 GB card in #723, cannot give it back when it knows it is done, nor take it again before its
first request without generating. The owner accepted, on the contributed #724 (2026-11-22), two routes that do both
on the active model, and OpenAI's retrieve beside them.

- **`POST /v1/ignis/models/unload`** is an idle unload brought forward: the same teardown into the same `unloaded`
  state, under the same in-flight count's lock as the idle clock, so no request is admitted onto the engine being
  torn down. It is refused `409 model_busy` while a request is admitted, waiting or running, while a load, a reload
  or another unload holds the switch's slot, and for a model not loaded from an artifact; it never ends a request,
  and touches no file. Nothing resident already is answered as done.
- **`POST /v1/ignis/models/load`** is the load the next request would begin, without the request: it waits for that
  load, shared with whoever else waits, and answers once the model serves. A model that serves is not loaded again,
  where a switch to its own id reloads it.
- **Both take the active model's id and select no other**, which stays a switch's job, and are under the API key
  inference uses: unloading costs the next user a cold load, which a switch or a reloading `PATCH` can cost them
  already under the same key. While a switch runs they are answered `503 model_switching`, as every other route.
- **`GET /v1/models/{model}`** is OpenAI's retrieve, the entry `GET /v1/models` lists, and like the listing is no use
  of the model.
- **The prefix.** The two lifecycle routes are the first under `/v1/ignis/`, as an extension gets an ignis name
  (`CODING_STANDARDS.md`, "Extensions get ignis names"): `/v1/models/` is OpenAI's namespace, where `DELETE
  /v1/models/{model}` deletes a fine-tuned model. The extensions ignis added before them stay where they are.

## Considered Options

- **A VRAM-only suspension that keeps the weights in pinned host memory and the KV:** a faster wake, but ADR 0049
  rejected it for the host pools the card's other users need, and it is a second load path.
- **An unload that ends the requests running:** a client would end another client's work.
- **Unloading through a switch to a placeholder "no model":** the switch's drain and `last_switch` for what is not a
  switch, and a model id that serves nothing in `GET /v1/models`.
