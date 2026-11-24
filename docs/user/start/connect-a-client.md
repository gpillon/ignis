---
title: "Connect a client"
description: "The base URL, the API key and how to point the OpenAI SDKs, curl, opencode and chat front ends at ignis."
sidebar:
  order: 4
---

Any client that speaks the OpenAI Chat Completions API (or the Responses API)
can use ignis. Three settings are all it needs: a base URL, an API key if you
set one, and a model id.

## The base URL

| Where the client runs | Base URL |
|---|---|
| On the same machine | `http://127.0.0.1:8000/v1` |
| In a container started with `-p 8000:8000` | the same, on the host's address |

The server listens on `127.0.0.1:8000` by default (`--server-bind`,
`IGNIS_SERVER_BIND`), so another machine reaches it only if you bind a wider
address; the container image binds `0.0.0.0:8000`
([Model and server](../configuration/model-and-server.md)). The routes are
listed in [The API](../api/index.md); `/v1/docs/` on the running server is the
generated reference.

## The API key

With no key set, `/v1` needs none, and any value a client insists on sending is
accepted. To require one:

```
ignis-server --server-api-key sk-...        # or IGNIS_SERVER_API_KEY=sk-...
ignis-server --server-api-key auto          # generate one and print it
```

Clients then send `Authorization: Bearer sk-...`; a request without it is
refused `401` and loads nothing. The Playground's static pages and `/v1/docs/`
stay open. `/ui/metrics` is behind the key like `/v1`
([Access](../configuration/access.md)).

:::caution
`--server-expose cloudflare-quick` publishes a public `https://*.trycloudflare.com`
URL and always requires a key; one is generated if you set none. Do not bind a
public address with no key.
:::

## The model id

The id a request names is the served id: `qwen3.8-27b` unless you started
another model. `GET /v1/models` shows it first (`data[0]`, `active: true`).
A request that names no model is served by the loaded one, which is what the
Playground does. A request that names another model the server lists switches
the server to it; see [the caution below](#two-model-ids-make-the-server-switch).

## curl

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $IGNIS_SERVER_API_KEY" \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hello"}],"max_tokens":256}'
```

Drop the `Authorization` header when no key is set. Streaming, the agent lane
and the Responses API are in the [examples](../api/examples.md).

## The OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-...")

reply = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=256,
)
print(reply.choices[0].message.content)
```

## The OpenAI JavaScript SDK

```js
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "http://127.0.0.1:8000/v1", apiKey: "sk-..." });

const reply = await client.chat.completions.create({
  model: "qwen3.8-27b",
  messages: [{ role: "user", content: "Hello" }],
  max_tokens: 256,
});
console.log(reply.choices[0].message.content);
```

With no key set on the server, `api_key` / `apiKey` still has to be a
non-empty string for the SDK; ignis ignores it.

Both SDKs honour `Retry-After` on a `503`. That is how a client rides out
`engine_full` (every lane busy) and `model_switching`: the header is 1 to 60
seconds, 60 being the most the OpenAI SDK waits ([The API](../api/index.md)).
The Responses API works through `responses.create`, and its WebSocket mode
through `responses.connect` ([Responses](../api/responses.md)).

A request field ignis cannot honour is a `400` naming it in `param`, not a
silent drop; if an SDK sends one, that error says which
([Chat completions](../api/chat-completions.md)).

## Coding agents: opencode and others

Point the agent's OpenAI-compatible provider at the base URL above, give it the
key if you set one, and name the served id as its model. For opencode, a
provider in `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ignis": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "ignis (local)",
      "options": { "baseURL": "http://127.0.0.1:8000/v1" },
      "models": { "qwen3.8-27b": { "name": "Qwen3.8-27B", "tool_call": true } }
    }
  },
  "model": "ignis/qwen3.8-27b",
  "small_model": "ignis/qwen3.8-27b"
}
```

With a key, add `"apiKey": "{env:IGNIS_SERVER_API_KEY}"` to `options`. Both
`model` and `small_model` name the same id, for the reason below.

Two things matter for agents:

- **Subagents.** A request can state its lane as `class: "agent"` or as an
  `@agent` suffix on the model name (`qwen3.8-27b@agent`). An `agent` request
  never displaces an `interactive` one when the KV pool is full
  ([KV pool](../configuration/vram-and-kv-pool.md#the-kv-pool)).
- **Long reasoning.** Absent `max_tokens`, a request generates at most
  `--model-default-max-tokens` (38,912) tokens, reasoning included, and ends
  with `finish_reason: "length"` at that point
  ([Chat completions](../api/chat-completions.md)).

### Two model ids make the server switch

:::caution
A client configured with two different ignis model ids reloads the server back
and forth. opencode sends its `small_model` requests (titles, summaries) to the
id named by `small_model`, so a `model` and a `small_model` that name different
ignis ids switch the server on alternate requests, 13-33 s each. Point both at
the same id, or start with `--switch-allow-implicit false`.
:::

The same holds for any front end that picks ids from `GET /v1/models`, which
lists every servable model.

### ignis's tools in the harness

A coding agent's harness can also add ignis as an MCP server, one
entry at `http://127.0.0.1:8000/v1/mcp` with the same key. The agent then sees
the connector's server tools, the models the server can switch to or fetch
among them. [The connector](../guides/connector.md) has each harness's entry:
opencode, qwen-code, pi and DeepSeek Harness.

## Open WebUI and other front ends

Add ignis as an OpenAI-compatible connection with the base URL above (and the
key). The front end then lists the models from `GET /v1/models`: the loaded one
first, then every model a request naming it would switch to.

:::caution
Open WebUI's task model (titles, tags, follow-ups) and Hermes, when set to a
listed id other than the chat's, cause the same switching as opencode's
`small_model`. Set the task model to the chat model's id, or start with
`--switch-allow-implicit false`, which makes `/v1/models` list the loaded
model alone.
:::

The Playground never causes this: its chat and Decide requests name no model,
and it switches only when you ask it to (**Load** on the Server view, the
image tools, the model picker under the prompt box)
([Get a model](../guides/models.md), [Switching models](../api/model-switching.md)).

## Next

- [The connector](../guides/connector.md) for ignis's server tools in a
  coding agent's harness.
- [Switching models](../api/model-switching.md) for explicit and implicit
  switches and what a request sees during one.
- [Deadlines](../api/deadlines.md) for how long a request may run and what it
  is answered when it runs over.
- [Troubleshooting](../troubleshooting.md).
