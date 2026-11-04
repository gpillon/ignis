---
title: "Quick start"
description: "From nothing to a first answer: run the container, let it fetch the default model, open the Playground and send one request."
sidebar:
  order: 3
---

You need a machine with a supported NVIDIA card and driver ([What ignis
is](index.md#what-it-runs-on)) and a container runtime that can give the
container the GPU. The commands use `podman`; with `docker`, write `--gpus all`
in place of `--device nvidia.com/gpu=all`.

## 1. Run the container

```
podman run --rm --device nvidia.com/gpu=all -p 8000:8000 \
  -v /path/to/models:/models \
  ghcr.io/gpillon/ignis:latest --download-path /models
```

`/path/to/models` is a directory you can keep. The server runs in the image as
`ignis` (uid 10001), so that directory must be writable by uid 10001 for the
download to land in it ([the image's user](install.md#the-container-image)).

With no model in `/models`, the server fetches the default one, `qwen3.8-27b`:
**18.1 GiB** from Hugging Face (on a 16 GB card, the id's 16 GB entry,
`qwen3.8-27b-16g`, 12.5 GiB), every file checked against its pinned size and
SHA-256 before it is used. On a terminal the server first asks (`[y/N]`, on
stderr), naming the size, the source and the destination. In a container there
is no terminal to answer, so it downloads without asking
([Letting the server fetch it](../guides/models.md#letting-the-server-fetch-it)).
An interrupted download resumes from its `.part` file.

Once the files are in, the model loads onto the card (a cold load took 13-33 s
on a 5090, [Idle unload](../api/idle-unload.md)). Until it serves, `/health`
answers `200`, and `/ready` and the `/v1` routes answer `503 server_not_ready`
with `Retry-After: 5` ([Deadlines](../api/deadlines.md)).

:::tip
To download ahead of time, on a machine with no GPU and no driver:
`ignis-server model download qwen3.8-27b`, with `--out <dir>` to choose the
directory ([Fetching one](../guides/models.md#fetching-one-ignis-server-model-download)).
:::

You can run a release archive instead of the container: unpack the Windows
`.zip` or the Linux `.tar.gz` ([Install](install.md#release-archives)) and run
`ignis-server` with no arguments. It listens on `127.0.0.1:8000` and fetches
the model into `./models`.

## 2. Open the Playground

Go to <http://127.0.0.1:8000/ui/>. The header shows the loaded model; type a
message in the chat ([The Playground](../playground/index.md)).

## 3. Send a request

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hello"}],"max_tokens":256}'
```

The answer is a standard chat completion: `choices[0].message.content` holds
the text and `usage` the token counts. `GET /v1/models` lists the loaded model
and what the server can switch to:

```bash
curl http://127.0.0.1:8000/v1/models
```

If a request returns `503 server_not_ready`, the model is still loading: retry
after the `Retry-After` seconds. Other messages are in
[Troubleshooting](../troubleshooting.md).

## What the defaults are

- The API and the Playground are on port 8000. There is no API key until you
  set one ([Connect a client](connect-a-client.md#the-api-key)).
- The metrics listener (`127.0.0.1:9464`) stays inside the container; the
  Playground's Monitor reads the same data at `/ui/metrics` on port 8000
  ([Install](install.md#the-container-image) shows how to scrape it from
  outside).
- The default profile is `auto`: the start picks `rtx-16gb`, `rtx5090` or
  `rtxpro6000` by the card's memory ([Profiles and defaults](../concepts/profiles.md)).
- The model unloads after 300 s without a request and loads again on the next
  one ([Idle unload](../api/idle-unload.md)).

## Next

- [Connect a client](connect-a-client.md) to use it from an SDK or an agent.
- [Get a model](../guides/models.md) for Flash-Next, the image model and your
  own artifacts.
- [Install](install.md) for a cache directory that keeps the n-gram cache and
  KV-disk, and for how the container stops.
