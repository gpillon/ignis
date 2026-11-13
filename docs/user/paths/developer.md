---
title: "The developer path"
description: "Build on ignis's API: the OpenAI-compatible endpoints, the Responses API, streaming and tools, deadlines, model switching, images, decisions."
sidebar:
  order: 3
---

For someone writing a client, an agent or a service against ignis. The
surface is OpenAI-compatible, so an OpenAI SDK works with a base URL; the
pages below say where ignis adds to it, refuses a field, or behaves its own
way. The [Playground](../playground/index.md) names the request each of its
controls sends, a quick way to see what a field does before you code it.

1. **[Connect a client](../start/connect-a-client.md)** — the base URL, the key,
   the model id, the SDKs.
2. **[The API](../api/index.md)** — every endpoint, the probes, keys and the
   error shape.
3. **[Chat completions](../api/chat-completions.md)** — fields, streaming, tool
   calls, reasoning, and the agent lane; **[structured
   output](../api/structured-output.md)**, an answer held to a JSON object or
   to your schema.
4. **[The Responses API](../api/responses.md)** — `POST /v1/responses` and its
   WebSocket mode.
5. **[Examples](../api/examples.md)** — curl for the common requests.
6. **[Counting a prompt](../api/count.md)** — how many tokens a body prefills,
   before you send it.
7. **[Deadlines](../api/deadlines.md)** — how long a request may run, and how a
   client states its own budget.
8. **[Switching models](../api/model-switching.md)** and
   **[idle unload](../api/idle-unload.md)** — what a request sees while the
   server changes or reloads its model.
9. **[Image generation](../api/images.md)** — generating and editing images
   with the image model.
   **[Speech to text](../api/transcriptions.md)** — transcribing audio with
   the speech model, which runs on the CPU beside it; **[live
   transcription](../api/realtime-transcription.md)** streams the audio as the
   person speaks.
10. **[Decisions](../api/decide.md)** — `POST /v1/decide`: an answer read from
    the model's distribution instead of generated, and locating a line or an
    item.
11. **[Reasoning and the loop guard](../concepts/reasoning.md)** — what the
    reasoning channel carries, and what the server does when it loops.
12. **[HTTP routes](../reference/http-routes.md)** — the generated route list;
    a running server serves the full OpenAPI document at `/v1/openapi.json`
    and a browsable page at `/v1/docs/`. **[JSON Schema
    keywords](../reference/json-schema.md)** lists what a `json_schema` may
    carry.

:::tip
Every field of the OpenAI request bodies is honoured, accepted as inert, or
refused with a `400` that names it in `param` — never dropped in silence. A
`400` naming a field tells you exactly what to remove
([Troubleshooting](../troubleshooting.md#a-400-naming-a-param)).
:::

:::tip
Prompt reuse keys on the start of the prompt. Keep the system prompt and the
tool block byte-identical across a session — no timestamps, no counters — and
a long conversation prefills only its new turn
([The KV cache and prompt reuse](../concepts/kv-cache-and-prompt-reuse.md)).
:::
