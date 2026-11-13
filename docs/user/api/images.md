---
title: "Image generation"
description: "The image model: generating and editing images over the API and in the Playground."
sidebar:
  order: 4
---

The server also serves an image model, **`ignis-image-2.1-turbo`** (Qwen-Image-2.1-Turbo,
[ADR 0050](../../adr/0050-ignis-serves-an-image-model.md)), fetched with
`ignis-server model download ignis-image-2.1-turbo`. Like every model it is
one loaded at a time: a text model and the image model replace each other by a
full reload ([switching](model-switching.md)). A route of the wrong kind for the
loaded model answers `409 model_kind_mismatch`, naming both kinds and the loaded
id; a `model` that names a known model of the route's kind switches to it under
`--switch-allow-implicit`, as for text.

**Licence.** The weights are under the **Qwen Research License: non-commercial
use only**, with a "Built with Qwen" notice. Do not serve the image model in a
commercial product. The weights' terms are the model repository's own:
[`LICENSE`](https://huggingface.co/gpillon/ignis-image-2.1-turbo/blob/d6912417aa47320cd71c78e19d99872faa5cb63b/LICENSE) and
[`NOTICE`](https://huggingface.co/gpillon/ignis-image-2.1-turbo/blob/d6912417aa47320cd71c78e19d99872faa5cb63b/NOTICE) at the pinned
revision. The repository's [`NOTICE`](../../../NOTICE) covers the third-party code
and text the image path carries (Pillow's resize, Qwen's enhancement prompts).

| Route | Body |
|---|---|
| `POST /v1/images/generations` | JSON: `prompt`, `n`, `size`, `seed`, `prompt_enhancement`, `background`, `stream`, `model`. |
| `POST /v1/images/edits` | Multipart: the same fields plus `image[]`, from 1 to `image.max_condition_images` PNG / JPEG / WebP files. |

The answer is `{created, data: [{b64_json, revised_prompt?}], seed, warnings?,
condition_sizes?}`: RGBA 8-bit PNGs, `b64_json` only (`response_format: url`, any
`output_format` but `png` and `mask` are each a `400`). `n` images are made one
after another. An absent `seed` is drawn and returned. `background: opaque`
composites on white; `auto` and `transparent` keep the generated alpha.
`stream: true` sends Server-Sent Events: `image_generation.progress` with
`{phase, image, step, total}` (`phase` is `enhancing`, `switching`, `loading`,
`encoding`, `denoising` or `decoding`) and one `image_generation.completed` per
image; an edit names them `image_edit.progress` and `image_edit.completed`. No
partial images are sent. The full schema is in `/v1/openapi.json`.

```bash
curl http://127.0.0.1:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"ignis-image-2.1-turbo","prompt":"a lighthouse at dusk","size":"2048x2048"}'
```

**Sizes.** `size` is one of `image.sizes`: 2048x2048, 2400x1792, 1792x2400,
2528x1696, 1696x2528, 2752x1536, 1536x2752 (the checkpoint's presets) and
1024x1024 (the size OpenAI clients often send); anything else is a `400` that
lists them. Every side is a multiple of 32, at most 16384. The load makes room
for the largest of them within `image.max_target_cells` (2400x1792 by
default); a size past that is not served, and the load warns of it. A request
for it is a `400` too, listing the sizes served. `auto` is 2048x2048 for a
generation and, for an edit, the preset whose aspect ratio is nearest the last
condition image's, always among the sizes served.

**Condition images.** Each is resized to about 1024x1024 of area, keeping its
aspect ratio, as soon as it is decoded. When the prompt and all of them exceed
the condition budget, or the room the image model has for them, they are
scaled down together, never below `image.condition_floor_px` squared; below
that the request is a `400` (`condition_budget_exceeded`) naming the budget or
the room. A condition image that is not a PNG, JPEG or WebP (a GIF, say), that
does not decode, or that decodes to more than `image.max_condition_mipx` is a
`400`. `image.max_condition_decodes` edits decode at a time; the others wait
their turn. The image model itself refuses, as `400 image_refused` in its own
words and before anything is encoded, a prompt too long for its positions
(about 8,000 tokens) and a condition image its encoder would not take at its
condition size (a panorama past 200:1).

**The queue.** One image job runs at a time and `image.queue_depth` wait in
order; a full queue answers `503 engine_full` with `Retry-After`. A client that
disconnects cancels its job before it starts or between denoising steps. A
job is held to `--server-request-timeout` from its start, never from its queue,
or `--server-stream-timeout` when it streams: past it the job ends between two
steps, its runner free for the next, and its request is `504 request_timeout`,
or, streamed, an `error` event after the images already made, whatever
`--server-timeout-reply` says ([deadlines](deadlines.md)). A running or queued
job counts as use for [idle unload](idle-unload.md).

| Setting (`--image-<name>`, `IGNIS_IMAGE_<NAME>`, `image.<name>` in the file) | Default | Meaning |
|---|---|---|
| `queue_depth` | `4` | Jobs that may wait behind the running one (0..64). |
| `max_n` | `10` | The most images one request may ask for (1..64). |
| `max_condition_images` | `10` | The most condition images one edit may send (1..10). |
| `max_condition_mipx` | `16` | The largest a condition image may decode to, in Mi pixels of 1,048,576 each (1..64); a larger one is refused. 16 takes any phone camera and bounds each image's frame at 64 MiB; an edit decodes up to `max_condition_images` of them. |
| `max_condition_decodes` | `1` | The most edits that decode their condition images at once (1..16); the others wait their turn. |
| `sizes` | the eight above | The sizes a request may ask for, each side a multiple of 32 up to 16384. The load sizes the image model's working memory for the largest within `max_target_cells`, so a larger size added live is served after the next load. |
| `max_target_cells` | `16800` | The largest image the image model's load makes room for, in latent cells of 16x16 pixels (1..1048576): 16,800 is 2400x1792, the largest preset. A size in `sizes` past it is not served, and the load warns. Read by the load, so a change holds from the next one. |
| `condition_budget_tokens` | `0` | Lowers the condition budget (0 is the whole budget the load planned). |
| `min_condition_budget_tokens` | `8191` | The least condition budget the image model's load accepts (0..1048576); a load whose VRAM leaves less is refused, naming the shortfall. 8,191 is the longest text-to-image prompt the model's positions admit; 0 takes any. Read by the load, so a change holds from the next one. |
| `condition_floor_px` | `512` | The smallest a condition image may be scaled to, as the side of a square of the same area (32..1024). |
| `prompt_enhancement` | `false` | Whether a request that does not say has its prompt rewritten first. |
| `prompt_enhancer` | `qwen3.8-27b` | The text model that rewrites prompts. |
| `prompt_enhancer_max_tokens` | `1024` | The most tokens one rewrite may take (16..8192); a longer answer is cut and used as it is. |

All of them are live: `PATCH /v1/config` changes them without a reload.

**Prompt enhancement.** With `prompt_enhancement: true` on a request (or the
setting on), a text model rewrites the prompt with Qwen's own system prompts
before the image model reads it; the result comes back as `data[].revised_prompt`.
An edit's rewrite is held to its `{"Rewritten": ...}` object with
[structured output](structured-output.md), so the answer is the object and a
cut one fails the rewrite instead of passing as the prompt; a model that does
not serve structured output is read as before. It is not free:

- With a text model loaded, the job is rewritten first when that model is the
  enhancer; otherwise the server switches to the enhancer, rewrites, then
  switches to the image model, and `GET /v1/models`' `last_switch` records
  both switches. An enhancer the idle clock unloaded is loaded for the rewrite,
  then the server switches once: two loads.
- With the **image model loaded, each job makes a visit: two reloads** (image
  model to enhancer, and back). During the visit every other request gets
  `503 model_switching` with `Retry-After`, and `GET /v1/models` reads
  `switching`; the visit leaves `last_switch` as it was.
- The visit waits its **turn in the queue**: the jobs admitted before the
  enhanced request finish on the image model, and the visit begins with none
  running (a full queue is `503 engine_full`, as for a job). The jobs admitted
  behind it run while the visit drains the image model, as for any switch:
  past `switch.drain_timeout` they end `503 model_stopped`.
- `image.prompt_enhancer` must be a **known model**: listed in
  `--switch-known-models`, the model the server started with, or a catalog
  entry already downloaded (or the text model loaded). The default
  `qwen3.8-27b` is none of those on a server that knows no text model: the
  start warns (`ignis.config.prompt_enhancer_unknown`), and every enhanced
  request falls back. A `PATCH /v1/config` naming an enhancer that is none is
  refused (`400 config_invalid`). An enhancer that cannot be reached, or a
  rewrite that fails, never fails the request: the prompt is used as sent and
  the response carries `warnings`.
- An edit shows the enhancer its condition images only when its vision tower is
  loaded; otherwise it reads text only, with a warning.

**In the Playground.** The **Image** view sends these requests. With a text
model loaded, pressing **Generate** or **Edit** is the choice of the image
model: the request names it, and the server switches as above (GitHub #324).
The button says what will happen:

- **Switch & generate**, with enhancement off: one switch, then the job.
- **Enhance with qwen3.8-27b, then switch & generate**, with enhancement on and
  the enhancer loaded: the rewrite runs on the model loaded, then one switch.
  There is no need to load the image model first only for the server to visit
  the enhancer from it: that is three loads, against one.
- With enhancement on and another text model loaded, **Switch & generate**
  says that the server goes to the enhancer first.

With **Show progress** on, the phases stream as they happen: `enhancing`,
`switching`, `loading`, then the job's. With `--switch-allow-implicit false`
the button stays disabled and says so, and a refusal the server gives at run
time (`404 model_not_found`) is shown as it is. A switch on its own is
**Load** on the Server view. The view's requests on the image model,
and the chat's and Decide's on a text model, name no model.
