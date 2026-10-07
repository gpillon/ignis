# Ignis — user reference

Everything needed to run the engine: how to get it, how to configure it, what
it serves, and what it reports. The project overview is the
[root README](../../README.md); the reasons behind these behaviours are in
[`docs/adr/`](../adr/).

> The API section below is written by hand for now. An OpenAPI description of
> the HTTP surface is planned and will supersede it.

- [Getting the engine](#getting-the-engine)
- [The model](#the-model)
- [Configuration](#configuration)
- [The API](#the-api)
- [Decisions](#decisions)
- [The Playground](#the-playground)
- [Telemetry and metrics](#telemetry-and-metrics)
- [Building from source](#building-from-source)
- [Working with make](#working-with-make)
- [Cutting a release](#cutting-a-release)

---

## Getting the engine

### The container image

A `v*` tag publishes a `linux/amd64` image to `ghcr.io/gpillon/ignis`, tagged
with the version. The image carries the CUDA runtime but no driver: the host's
NVIDIA driver is injected by the container runtime, and the model is mounted,
never baked in.

```
podman run --rm --device nvidia.com/gpu=all -p 8000:8000 \
  -v /path/to/models:/models:ro \
  -e IGNIS_ARTIFACT=/models/qwen3_8_27b_nvfp4full-v2.ninfer \
  ghcr.io/gpillon/ignis:0.1.2
```

(`docker`: `--gpus all` in place of `--device`.) Every flag has an `IGNIS_*`
environment variable; anything after the image name is passed to the server.
The image sets no default command, so the Playground is on — the server's own
default.

With no `IGNIS_ARTIFACT` the image fetches the model into its own working
directory (`/home/ignis/models`) and loses it with the container. Mount a
directory and point the download at it to keep what it fetches:

```
podman run --rm --device nvidia.com/gpu=all -p 8000:8000 \
  -v /path/to/models:/models \
  ghcr.io/gpillon/ignis:0.1.2 --model-download-path /models
```

To smoke-test the image with no GPU and no model — the deterministic CPU mock
([ADR 0006](../adr/0006-exclusive-gpu-testing.md)) — say so:

```
podman run --rm -p 8000:8000 ghcr.io/gpillon/ignis:0.1.2 --no-model-download
```

`Containerfile` builds the same thing locally (`podman build -t ignis:dev .`).
Its `artifacts` stage is what CI exports the Linux tarball from, so the release
binaries and the image binaries are the same build
([ADR 0032](../adr/0032-release-through-the-container-build.md)).

### Release archives

A `v*` tag also publishes a Windows `.zip` and a Linux `.tar.gz`, each with its
SHA-256. Both are compiled for `SM120a` only. The Linux tarball needs the CUDA
13 runtime on the host; the Windows zip carries `cudart64_*.dll`.

---

## The model

The engine loads a `.ninfer` artifact: weights, tokenizer and chat template in
one container, verified against its `.sha256` sidecar at load. A mismatch
refuses the start.

### Letting the server fetch it

A GPU build started **without** `--artifact` / `IGNIS_ARTIFACT` looks for the
model under `--model-download-path` (default `./models`, flat, under the names
the repository publishes) and fetches it when it is not there
([ADR 0033](../adr/0033-the-server-fetches-its-own-model.md)):

```
ignis-server                      # asks first: the size, the source, the destination
ignis-server --no-model-download  # never fetches: the placeholder template and the CPU mock
ignis-server --model-download-path /weights
```

On a terminal you are asked (`[y/N]`, on stderr); without one — a container, a
daemon, CI — nobody can answer, so it downloads. What it fetches is
`gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer`: the artifact and its
`.graft.json` sidecar, streamed to a `.part` file, checked against the size and
SHA-256 pinned in the binary, and renamed into place only then. An interrupted
download resumes; a tampered one is discarded and refuses the start. A build
without `--features cuda` never downloads weights it could not run.

A `hf download … --local-dir models` done by hand and a download the server did
land on the same file names, so either satisfies the other.

### Pointing at one you already have

```
set IGNIS_ARTIFACT=./models/qwen3_8_27b_nvfp4full-v2.ninfer   # Windows
export IGNIS_ARTIFACT=./models/qwen3_8_27b_nvfp4full-v2.ninfer
```

The artifact carries a grafted DFlash2 drafter module, which `--spec dflash2`
loads for speculative decoding and which costs no VRAM when speculation is off.

### The uncensored variant

[`gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer`](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer)
is the same image with the
[huihui-ai abliteration](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated)
applied. It is the same container: 1,255 of the 1,325 objects are
byte-identical to the default image, and only the 70 matrices the abliteration
changed are re-encoded. **It does not refuse**, so put the guardrails in the
tool layer.

The server never fetches it on its own. Download it next to the default image
and point at it:

```
hf download gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer --local-dir models
ignis-server --artifact ./models/qwen3_8_27b_nvfp4full-v2-huihui-abliterated.ninfer
make dev UNCENSORED=1   # the same, from a checkout
```

The model id stays `qwen3.8-27b` unless `--model` names another one. The
pointing heads and the DFlash2 drafter load unchanged. They were fitted to the
default weights, and the target still verifies every draft, so the drafter
can only change the acceptance rate.

---

## Configuration

Each flag has a matching environment variable. **A flag overrides its env var,
which overrides the built-in default.** `ignis-server --help` prints the
always-current table; this one is a copy.

### Model and server

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `-m`, `--model <id>` | `IGNIS_MODEL` | `qwen3.8-27b` | The loaded model id: what `/v1/models` reports, what submissions must name, and the key the download registry is looked up by. |
| `-b`, `--bind <addr>` | `IGNIS_BIND` | `127.0.0.1:8000` | The API listener's address. |
| `-a`, `--artifact <path>` | `IGNIS_ARTIFACT` | unset | The `.ninfer` container. Must exist and verify, or the server refuses to start. Unset: the model is looked for under `--model-download-path`, and fetched when it is not there. |
| `--model-download` / `--no-model-download` | `IGNIS_MODEL_DOWNLOAD` | on | Fetch a missing model. Only consulted with `--artifact` unset, and only in a `--features cuda` build. |
| `--model-download-path <dir>` | `IGNIS_MODEL_DOWNLOAD_PATH` | `./models` | Where a fetched model lands, and where one fetched earlier is found. |
| `--request-timeout <secs>` | `IGNIS_REQUEST_TIMEOUT` | `30` (max `3600`) | The deadline for a non-streaming completion; expiry is a 504 `request_timeout`. |
| `-h`, `--help` | — | — | Print the flag table and exit. |
| `-V`, `--version` | — | — | Print the version and exit. |

### Prompting

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--enable-thinking <bool>` | `IGNIS_ENABLE_THINKING` | `true` | The server-wide default for `enable_thinking`. |
| `--reasoning-effort <value>` | `IGNIS_REASONING_EFFORT` | template default | The server-wide default `reasoning_effort`. An effort the template does not take is rounded up to one it does (Qwen3.8: `high` and `max` are `xhigh`, `minimal` is `low`), the same as a per-request one. |
| `--thinking-budget <n\|off>` | `IGNIS_THINKING_BUDGET` | `6144` | The server-wide thinking budget: after `n` reasoning tokens with the block still open, a close (`My thinking time is over. I must now write the complete final answer from what I already have, without calling any more tools.\n</think>`) is forced and the model answers. `off` = no default budget. A request's `thinking_budget` overrides it, and `reasoning_effort: max` runs without one. Set with a tokenizer that has no single-token `</think>`, the server refuses to start. |
| `--system-message-policy <p>` | `IGNIS_SYSTEM_MESSAGE_POLICY` | `merge` | `merge`: a leading run of system messages joins the system prompt, a later one is its own block in place. `strict`: a system message that is not first is a 400. |
| `--developer-message-policy <p>` | `IGNIS_DEVELOPER_MESSAGE_POLICY` | `inplace` | One of `inplace`, `into-system`, `after-system`, `one-after-system`, `reject`. A leading developer message is the system prompt except under `reject`. |

### Context, KV and speculation

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--max-context <tokens>` | `IGNIS_MAX_CONTEXT` | `40960` | Max per-sequence context (prompt + generation). The KV pool must be able to hold one of them or the load is refused. |
| `--prefill-chunk <tokens>` | `IGNIS_PREFILL_CHUNK` | `1024` | The prefill chunk width, a nonzero multiple of 128. The program's prefill scratch is reserved for it at load. |
| `--decode-lanes <n>` | `IGNIS_DECODE_LANES` | `3` | Flash-Next only, `1..=8`: the sequences decoded at once. Each lane holds a whole `--max-context` in the KV pool, so fewer lanes leave the expert cache more of the VRAM budget (and a lone user decodes at one lane either way). The 27B serves a fixed 8 lanes and refuses the flag. `make MODEL=flash-next` runs 3 lanes at 262,144 tokens each (the checkpoint's trained positions); the `make` knob is `LANES`. |
| `--decode-share <percent>` | `IGNIS_DECODE_SHARE` | the model's: `0` on the 27B, `50` on Flash-Next | The part of the time decoding lanes keep while a prompt prefills, 0-99: after a chunk that took `t`, the next waits until they have decoded for `t * s / (1 - s)`, and nothing waits when no lane decodes. It trades the prefilling request's TTFT (x2 at 50, only while lanes decode) for the lanes' rate (half instead of ~1 tok/s on Flash-Next). |
| `--kv-format <fmt>` | `IGNIS_KV_FORMAT` | `hq-e8-2b` | `hq-e8-2b` (serving) or `bf16` (retained; the format every correctness oracle runs against — [ADR 0022](../adr/0022-two-kv-formats-bf16-as-oracle.md)). Decides what a pool byte budget is worth in tokens. |
| `--kv-pool-bytes <bytes>` | `IGNIS_KV_POOL_BYTES` | the rest of the VRAM budget | The paged-KV pool budget (accepts `K`/`M`/`G`). Too small for `--max-context`, or past the VRAM budget, fails the load by name. |
| `--kv-host-pool-bytes <bytes>` | `IGNIS_KV_HOST_POOL_BYTES` | `2G` | The KV-RAM host tier. Page-locked whole at start and held for the life of the load, so it is RAM the process holds even idle — and the figure Windows reports as shared GPU memory. `0` disables the host tier. (`make` sets `8G`.) |
| `--spec <backend>` | `IGNIS_SPEC` | unset | Speculative decoding backend: `dflash2` (Qwen3.8-27B), `mtp` (Qwen3.8-Flash-Next, see [below](#flash-next-speculation-mtp)), `off`, or unset for none. |
| `--draft-tokens <n>` | `IGNIS_DRAFT_TOKENS` | — | 1..7. Required with `--spec dflash2`; with `--spec mtp` the most drafts a lane verifies per round (default 2). |
| `--draft-rows <n>` | `IGNIS_DRAFT_ROWS` | `0` (= 8) | Flash-Next MTP only: the rows one verify round may take across all lanes, `0` or 2..8. Each lane drafts `min(draft tokens, rows / lanes - 1)`, so `3` drafts at one lane only. (`make` knob `DRAFT_ROWS`.) |
| `--draft-head <head>` | `IGNIS_DRAFT_HEAD` | `full` | Needs `--spec`. The head the drafter proposes with: `full` (the target's output head) or `shortlist` (the artifact's Q4 head over the 131,072 most frequent tokens, +356 MB of VRAM). What a round accepts is still the target's choice, so the text changes only at near ties; measured on coding prompts `shortlist` accepts 7% fewer tokens per round and is slower overall ([finding](../findings/2026-09-24-upstream-quick-wins-ab.md)). (`make` knob `DRAFT_HEAD`.) |
| `--rope-scaling <spec>` | `IGNIS_ROPE_SCALING` | none | `yarn:F[,t=<c>][,bf=<n>][,bs=<n>]` rescales the checkpoint's trained 262,144-position envelope. `F` in (1, 64]. |

### Flash-Next speculation (MTP)

Qwen3.8-Flash-Next can draft with its own multi-token-prediction head
(`--spec mtp`, `make MODEL=flash-next SPEC=mtp`). It is **off by default**, and
on a 5090 that is the better setting for most uses. Before turning it on:

- **It needs its companion container** beside the artifact
  (`qwen3_8_flash_next_mtp_3p0-v2.ninfer`, ~1 GB). `--spec mtp` without it
  refuses the start.
- **It keeps the text.** Greedy output is the same as without it, up to
  near-ties, and sampling keeps its distribution.
- **It costs ~1.1 GB of VRAM, taken from the expert cache, not from the
  context.** The KV pool, and so `--max-context`, is unchanged. The cost is
  reserved at load and stays the same whatever the number of drafts: a
  smaller `--draft-tokens` or `--draft-rows` does not give it back.
- **It pays off at one active request, and costs at two or three.** On an
  RTX 5090 (PCIe Gen 3, 2026-10-07): one lane runs 1.02-1.30x faster (most on
  long prose); two and three lanes run 7-15% slower with the default row
  budget, and 1-2% slower with `--draft-rows 3`, which drafts at one lane
  only. Decode there is bound by the experts it copies in over PCIe, and a
  verify round copies more of them
  ([finding](../findings/2026-10-07-flash-next-mtp-speculation-is-pcie-bound.md)).
  These figures leave out the 1.1 GB the head takes from the expert cache,
  so the served cost at several lanes is a little higher.
- **Use it** when one user or one agent works at a time: `--spec mtp
  --draft-rows 3`, or serve a single lane outright (`make MODEL=flash-next
  LANES=1 SPEC=mtp`), which also gives the expert cache ~2.4 GB the other
  lanes' KV would hold. **Leave it off** when several agents decode together.
- A card that holds every expert in VRAM (96 GB) copies none over PCIe,
  which is where MTP should pay most. It is not measured there yet.

### VRAM budget and retained state

Laid out at load and printed as the `ignis.runtime.vram_plan` event
([ADR 0030](../adr/0030-device-memory-reserved-at-load.md)). A plan that cannot
hold one full context refuses the start.

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--vram-headroom-bytes <b>` | `IGNIS_VRAM_HEADROOM_BYTES` | `1G` | Derives the budget: the device memory free at start minus this. Not with `--vram-budget-bytes`. |
| `--vram-budget-bytes <b>` | `IGNIS_VRAM_BUDGET_BYTES` | derived | The device memory the whole process may hold, weights included. More than is free refuses the start. |
| `--allow-vram-oversubscription` | `IGNIS_ALLOW_VRAM_OVERSUBSCRIPTION` | off | With `--vram-budget-bytes` only: start above free memory (or below the plan's minimum) with a warning instead of a refusal. On Windows that pages. |
| `--prompt-reuse <on\|off>` | `IGNIS_PROMPT_REUSE` | `on` | `off`: no prompt checkpoint is captured or reused, and no prefix is shared unless `--retained-device` or `--retained-host` gives slots for it. |
| `--retained-device <n>` | `IGNIS_RETAINED_DEVICE` | `0` | Retained slots in VRAM: reserved in the VRAM plan, copied device to device (~0.3 ms), and handed out before any host slot. A card with VRAM to spare can take `--retained-device 16 --retained-host 0`. |
| `--retained-host <n>` | `IGNIS_RETAINED_HOST` | two per decode lane (16); `0` with `--prompt-reuse off` | Retained slots in one pinned host block reserved at start, ~222 MiB each at the default load (3.5 GiB for 16): no VRAM, so the KV pool gets it; a capture and a claim each cost a PCIe copy (~15–19 ms). Both kinds hold the images of retained checkpoints and shared prefixes. When none is free, retained state gives one up — checkpoints before prefixes, `agent` before `interactive`, then least recently used; when nothing can, the publish or capture is skipped. `--retained-slots` was replaced by these two and now refuses the start. |
| `--retained-interactive-ttl <secs>` | `IGNIS_RETAINED_INTERACTIVE_TTL` | `300` | Idle seconds after which a main-conversation checkpoint in KV-RAM ranks as a subagent's. Needs `--prompt-reuse on`. |

### Flash-Next n-gram startup cache

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--persist-ngram-cache <true\|false>` | `IGNIS_PERSIST_NGRAM_CACHE` | `true` | Keep a compact copy of the selected Flash-Next hot rows on disk. `false` reads and writes no persistent n-gram cache. |
| `--persist-ngram-cache-path <model\|auto\|dir>` | `IGNIS_PERSIST_NGRAM_CACHE_PATH` | `model` | Cache directory. `model` is the artifact's own directory, on the disk that already holds the model. An explicit relative directory is relative to the server's working directory. |

`model` writes `<artifact stem>.ngram-<key>.bin` beside the artifact. `auto`
uses `%LOCALAPPDATA%\ignis\cache\ngram` on Windows, and
`$XDG_CACHE_HOME/ignis/ngram` on Linux (when XDG_CACHE_HOME is absolute),
otherwise `$HOME/.cache/ignis/ngram`. The directory is created on a cache miss.
A read-only model directory falls back to the normal load each time: name
`auto` or a directory there.

The first load gathers the normal hot rows and saves them; subsequent loads
read the compact file and validate its SHA-256 checksum. Loading still closes
the artifact mapping before any hot-row reads, even with persistence disabled.
A changed artifact path, file stamp, table layout, selected row order/count,
or cache format produces a different key. New packs also record a content
identity assembled from the n-gram source files' verified SHA-256 digests.
Legacy artifacts work without repacking: they use the filesystem stamp and
descriptors. Deliberately changing source bytes while preserving those stamps
and the source digest requires separate artifact verification.

Corruption, an unavailable cache directory, a concurrent writer or a failed
write falls back to the normal artifact load. Publication is atomic and uses
an OS lock; interrupted writes are never accepted as complete cache files.
The cache is disposable and does not change the model artifact. With the
default 1 GiB hot-row RAM budget a compact file is about 980 MiB. Saving a new
key removes the same artifact's older cache files in that directory, so one
file per artifact remains. This cache is separate from prompt/KV reuse.

### Vision

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--vision` | `IGNIS_VISION` | off | Load the vision tower and reserve its workspace. Without it every `image_url` part is refused with `vision_disabled`, whatever the artifact holds. |
| `--vision-max-tokens <n>` | `IGNIS_VISION_MAX_TOKENS` | `32768` with `--vision` | Merged vision tokens per request, 1..1,048,576. An image past the cap is refused, not shrunk. |
| `--vision-embedding-pool-mib <n>` | `IGNIS_VISION_EMBEDDING_POOL_MIB` | one envelope-wide embedding | Encoded images kept for reuse across requests, 1..65536. |
| `--media-cache-mib <n>` | `IGNIS_MEDIA_CACHE_MIB` | `1024` with `--vision` | Prepared images kept for reuse; `0` disables, max 65536. |
| `--media-allow-private-network` | `IGNIS_MEDIA_ALLOW_PRIVATE_NETWORK` | off | Needs `--vision`. Fetch image URLs on private, loopback and link-local addresses. |

### Playground, metrics, access

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--ui` / `--no-ui` | `IGNIS_UI` | on | Serve the Playground at `/ui/` ([ADR 0026](../adr/0026-playground-embedded-when-built.md)). A binary built without `web/dist` serves the page that says how to build it. |
| `--metrics` | — (flag only) | off | Serve Prometheus metrics on their own listener, and at `/ui/metrics` unless `--no-ui` ([ADR 0017](../adr/0017-prometheus-metrics.md)). |
| `--metrics-bind <addr>` | — (flag only) | `127.0.0.1:9464` | The metrics listener's address; needs `--metrics`. No API key, never exposed. |
| `--api-key <key>` | `IGNIS_API_KEY` | unset | Unset: `/v1` needs no key. Set: `Authorization: Bearer <key>`. `auto`: generate one and print it. |
| `--expose <mode>` | `IGNIS_EXPOSE` | unset | `cloudflare-quick` publishes a `https://*.trycloudflare.com` URL, printed at start. Always requires an API key — one is generated when none is set ([ADR 0028](../adr/0028-expose-modes-always-keyed.md)). |

---

## The API

An OpenAI-compatible HTTP server. The `/v1` routes sit behind the API key when
one is set; the Playground's static pages and the API reference at `/v1` do
not.

| Endpoint | Method | Notes |
|---|---|---|
| `/v1/models` | GET | The loaded model. |
| `/v1/chat/completions` | POST | Chat completions — streaming (`stream: true`, SSE) and non-streaming. |
| `/v1/responses` | POST | The OpenAI responses API (non-streaming; `stream: true` is a 400). |
| `/v1/tokenize` | POST | A body's prompt-token count without serving it ([below](#counting-a-prompt)). |
| `/v1/detokenize` | POST | Token ids back to text, through the same tokenizer. |
| `/v1/decide` | POST | The decision endpoint ([below](#decisions)). |
| `/v1/systemone` | POST | The same handler under its Jev name. |
| `/v1` | GET | The API reference — a 307 to `/v1/docs/` (ADR 0036). |
| `/v1/docs/` | GET | Swagger UI over the document, from assets compiled into the binary: no CDN, nothing fetched. Open even with an API key set — it publishes the API's shape, not its load — and *Authorize* is how you drive the keyed routes from the page. |
| `/v1/openapi.json` | GET | The OpenAPI 3.1 document, generated from the handlers themselves. Monitoring is not in it. |
| `/ui/` | GET | The Playground — only with `--ui`. |
| `/ui/metrics` | GET | Prometheus text format 0.0.4 for the Playground — only with `--ui` and `--metrics`. Behind the API key, like `/v1`. |

Every `/v1` route answers a CORS preflight. Prometheus itself scrapes
`GET /metrics` on the metrics listener (`--metrics-bind`), not on the API's
address: no key, and never reachable through `--expose`.

**Request body limits**, enforced before JSON parsing: 16 MiB on a text-only
load, 384 MiB with `--vision`, which takes images inline as base64 data URIs.
Both are past what their own load can use, so an oversized prompt meets the
refusal that names the real limit — the context — rather than a byte count.

**Errors** use OpenAI's `{"error": {message, type, code}}` body with the
matching status: 400 bad request, 404 unknown model, 413 oversized request, 503
engine full, 504 the engine did not finish in the timeout.

### Chat completions

Accepts `messages` (role + content), `model`, `stream`, `max_tokens`,
`temperature` (0..2), `top_p` (0..1), `presence_penalty` and
`frequency_penalty` (-2..2), and a signed 64-bit `seed`.

- `top_k` is an **Ignis extension**, not an OpenAI Chat Completions parameter:
  0..20, where 0 selects the 20-candidate sampler cap during stochastic
  sampling.
- `class` is an **Ignis extension**: the request's lane tag, `interactive`
  (default) or `agent`. The same class can be stated as an `@agent` suffix on
  the model name.
- `thinking_budget` is an **Ignis extension**: the reasoning tokens this
  request may spend before its close is forced. Absent takes
  `--thinking-budget`, `0` means no budget, and `reasoning_effort: "max"`
  ignores it and runs with no budget. The budget always leaves 2,048 tokens of
  `max_tokens` for the answer. A forced close is reported as
  `thinking_budget_forced_at` on the choice (on the finish chunk when
  streaming; top-level on `/v1/responses`): the reasoning tokens emitted when
  it began. The field is absent when the close was not forced.
- Values outside the ranges above are rejected with
  `invalid_sampling_parameter`; they are never silently clamped.
- **Absent sampling fields take the Qwen3.8 model card's values** for the
  request's mode, each field on its own (spec server/12):

  | mode | `temperature` | `top_p` | `top_k` | `presence_penalty` | `frequency_penalty` |
  |---|---:|---:|---:|---:|---:|
  | thinking | 1.0 | 0.95 | 20 | 0 | 0 |
  | `enable_thinking: false` | 0.7 | 0.80 | 20 | 1.5 | 0 |

  An absent `seed` is a fresh one per request, so two identical requests are
  two independent samples; a sent `seed` is kept. **Greedy is
  `temperature: 0`**: the fields such a request does not send are neutral,
  and because the leaf's greedy branch does not read stochastic filters or
  penalties, a non-neutral `top_p`, `top_k`, `presence_penalty` or
  `frequency_penalty` sent with it is rejected rather than ignored.
- `max_completion_tokens` is `max_tokens` under OpenAI's current name — the
  one the current SDKs send. Both with different values is a 400.
- `stop` (a string, or 1 to 4 non-empty strings) ends the answer before the
  first sequence to appear in its content, with `finish_reason: "stop"`. The
  sequence is never emitted, streaming included; it is never matched in the
  reasoning, nor inside a tool call.
- `tool_choice` takes `"auto"` (the default), `"none"`, `"required"`, and a
  named function (`{"type": "function", "function": {"name": "..."}}`, or
  `{"type": "function", "name": "..."}` on the responses API). The last two
  force the call's opening — `<tool_call>` and the function tag, with the
  name when one is named — and the model writes the arguments. With thinking
  on (the default) the opening is forced right after the reasoning block
  closes, so the model still thinks first; with `enable_thinking: false` it
  is the first thing generated. The call comes back as any other call does,
  with `finish_reason: "tool_calls"`. Naming a function that is not in
  `tools`, `"required"` with no tools, and a `max_tokens` shorter than the
  opening are a 400; so is a tokenizer that cannot force the opening
  (`tool_choice_unsupported` — with thinking on it needs `</think>` and
  `<tool_call>` to be single tokens, and `enable_thinking: false` is the
  way out). With thinking on the reasoning spends the same `max_tokens`: a
  block that never closes ends `length` with no call, which the thinking
  budget's answer reserve prevents.
- Non-streaming returns `choices[].message.content` plus `usage`; streaming
  emits `chat.completion.chunk` SSE frames (token deltas, a final
  `finish_reason` chunk, then `[DONE]`). `usage.prompt_tokens_details.cached_tokens`
  is the prompt the request resumed from retained state instead of
  prefilling (0 when none).
- Every other OpenAI field is accepted as inert when its value asks for what
  the server already does (`n: 1`, `store: false`, `logprobs: false`,
  `response_format: {"type": "text"}`, `metadata`, `user`, ...) and refused
  with a 400 `unsupported_value` naming it in `param` otherwise. The full
  table is in the API reference at `/v1/docs/`; a field outside it is ignored.

The responses API accepts `input` (a string or a message list), `model`,
`max_output_tokens`, `temperature` and `seed`, and returns the responses shape
with the text in an `output_text` part.

### Examples

```bash
# the loaded model
curl http://127.0.0.1:8000/v1/models

# non-streaming chat completion
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"..."}],"max_tokens":256}'

# streaming (SSE)
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"..."}],"stream":true}'

# a subagent's request: the agent lane, stated as a model suffix
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b@agent","messages":[{"role":"user","content":"..."}]}'

# the responses API
curl http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"input":"...","max_output_tokens":256}'
```

With `--api-key`, add `-H "Authorization: Bearer $IGNIS_API_KEY"`.

### Counting a prompt

`POST /v1/tokenize` renders a chat body exactly as `/v1/chat/completions` would —
chat template, tool block, thinking controls, system block — and answers how many
tokens it prefills, **without submitting it**: no lane, no GPU, no KV page. The
`count` is the `usage.prompt_tokens` the same body reports when served, and
`max_model_len` is the server's `--max-context`, so one call says whether a body
fits.

```bash
curl http://127.0.0.1:8000/v1/tokenize \n  -H 'Content-Type: application/json' \n  -d '{"messages":[{"role":"user","content":"Hello"}],"tools":[...],"return_token_ids":true}'
# {"count":1234,"max_model_len":262144,"token_ids":[...]}
```

Send `messages` **or** a raw `prompt` (tokenized with no template), never both.
`return_token_ids` and `return_text` add the ids and the rendered prompt.
Sampling fields are ignored, `stream` is refused, and a body with an `image_url`
part is a `400 media_not_countable`: an image's cost is its grid after
preparation, which only fetching and decoding it would tell — send the request.
`POST /v1/detokenize` takes `{"token_ids":[...]}` and answers `{"text":"..."}`; an
id outside the vocabulary is a `400` naming its index. The tokenizer normalizes
text to NFC, so a decomposed `e` + accent comes back as the composed `é` (the
`text` a call returns is the rendered prompt as sent, before that).
Neither route enters the scheduler; the count of calls is
`ignis_tokenize_requests_total{route}` with `--metrics`.

### Canary check

`ignis-bench canary` runs the fixed high-signal prompt suite against a live
server and checks each output is *sane* and *deterministic* — greedy plus a
fixed seed means an identical output on a repeat run
([ADR 0007](../adr/0007-performance-gate-not-parity.md)):

```bash
cargo run -p ignis-bench -- canary --endpoint http://127.0.0.1:8000
```

---

## Decisions

`POST /v1/decide` answers typed questions about one piece of evidence. Three
primitives (`noul`, `choice`, `score`) are read from the logits of named answer
tokens at a single position: **nothing is generated**, they cost one prefill,
and `usage.output_tokens` is 0. Three more (`number`, `point`, `box`) generate
one constrained digit per step, so they cost a prefill plus a round per digit.

The wire shape is Jev's `POST /v1/systemone`, copied rather than invented, so an
unmodified Jev client reaches this engine by changing the URL. The body is one
`state` and a map of `questions`:

```bash
curl http://127.0.0.1:8000/v1/decide \
  -H 'Content-Type: application/json' \
  -d '{"state":"Help! My payouts have been failing for 3 days.",
       "questions":{"queue":{"type":"choice",
                             "instructions":"Which queue should this ticket land in?",
                             "criteria":{"payouts":"Money leaving the account",
                                         "billing":"Money coming in",
                                         "fraud":"Suspected fraud or account takeover"}}}}'
```

Object key order is preserved on the way in and out: options are answered in the
order the body declares them. A `state` that is an image — or an image plus
words — is what `point` and `box` answer against, in the image's own pixels.

Why the endpoint exists, what its confidence means, and the one failure nothing
in the response can show (the **answer mass**, charted on the metrics listener):
[ADR 0034](../adr/0034-the-leaf-answers-without-generating.md). The Playground's
**Decide** tab builds these requests and shows the full distribution behind each
answer.

### Finding a line or an item

To ask *which* line of a log, element of a list or sentence of a text answers an
instruction, ask a `locate` (GitHub #275, #278). It generates nothing and
writes nothing into the state: calibrated attention heads narrow the text to a
few candidates, and a labelled `choice` decides among them. The text may be far
longer than the context — a log of a million tokens, a document, an API's
answer of thousands of records:

```bash
curl http://127.0.0.1:8000/v1/decide \
  -H 'Content-Type: application/json' \
  -d '{"state":{"service":"billing","log":"09:21 INFO gateway: GET /v1/orders 200\n09:21 ERROR billing: provider returned 503\n09:22 INFO auth: user 61 signed in"},
       "questions":{"cause":{"type":"locate",
                             "instructions":"Which line says the card processor was unavailable?",
                             "within":"/log"}}}'
```

```json
{"type": "locate", "kind": "log", "method": "shortlist", "compression": "template_fold",
 "found": 0.97,
 "segment": 1, "value": "09:21 ERROR billing: provider returned 503", "confidence": 0.91,
 "ranking": [{"segment": 1, "share": 0.91}, {"segment": 0, "share": 0.05}],
 "pointers": [{"segment": 1, "value": "09:21 ERROR billing: provider returned 503", "share": 0.91},
              {"segment": 0, "value": "09:21 INFO gateway: GET /v1/orders 200", "share": 0.05}]}
```

- **The target** is the `state`, or the part `within` names (a JSON Pointer): a
  string, whose segments are its lines split on `\n` exactly, or a non-empty
  array, whose segments are its elements. `segment` numbers them from 0, the
  way `split("\n")` or the array index would — always into the state you
  sent, whatever was folded or windowed — and `value` is the segment as you
  sent it.
- **`kind`** names the reading: `log`, `prose` or `records`, or `auto` (the
  default). `auto` says `records` for an array of two or more JSON objects;
  otherwise it folds the target's first 2,000 non-blank segments into
  templates and says `log` when at least half of them fall in templates of
  two or more, `prose` otherwise. The answer names what it resolved.
- **`compression`**: `template_fold` (the default for a log) folds the log
  into templates and their values first, so nothing near its length is ever
  prefilled; `none` (the default for prose and records) reads the text as it
  is.
- **`method`**: `shortlist` (the default) or `vote`, the head vote served
  before GitHub #278, unchanged — one prefill over at most 4,554 tokens on the
  served artifact (`locate_too_long` past that), no `found`, `pointers` its
  winner alone.

What each combination does, costs and measured
([spec 22](../specs/decide/22-locate-by-copy-over-a-folded-state.md); the
acceptance run's numbers go in ADR 0042):

| | reads | cost | measured (exploratory) |
|---|---|---|---|
| `log` + `template_fold` (default) | the end heads keep 5 templates, a `choice` picks one; they keep 16 of its rows, a last `choice` over those rows' original lines picks the line; `found` | a few short prefills: 0.7-2.6 s median at 100K-1M tokens | 21 of 23 on a fresh cluster capture of 100K to 1M tokens |
| `log` + `none` | the end heads over the whole log, window by window, keep 16 lines; a `choice`; no `found` | each window's prefill | — |
| `prose` + `none` (default) | the sum heads keep 16 sentences; a `choice` over them inside their paragraphs; `found` | first question: each window's prefill (45-68 s per 200K tokens); then 1-4 s per window | a gold sentence picked 91.7% up to 200K tokens, 58% at 1M |
| `records` + `none` (default) | the end heads keep 16 records; a `choice` over them as one-line JSON; `found` | as prose | 58 of 60 up to 3,500 records, 29 of 30 at 10,000 |
| `records` + `template_fold` | the log route over the records as one-line JSON; no `found` | no long prefill: 1.3-1.7 s median | 52 of 60 |
| `vote` + `none` | the head vote | one prefill | 93.6% on short states |

Refused before any prefill: an unknown value (`kind_unknown`,
`method_unknown`, `compression_unknown`, each naming the accepted ones); a fold
under `vote` or of prose (`compression_unsupported`); `kind: records` on a
state that is not an array of objects, or `log`/`prose` on one
(`kind_mismatch`); `kind` or `compression` on any other type
(`kind_unsupported`, `compression_unsupported`); one segment longer than a
window (`locate_segment_too_long`); a load whose artifact nobody calibrated
(`locate_uncalibrated` — the load says which at start, `ignis.decide.locate`,
naming its heads and window); a fold's level-1 text past the context
(`context_exceeded`). Content-parts states are refused too. A step rendered
from an earlier step's answer — a fold's level 2, every `choice` — is known
only after the first prefill, so a fault there (say a `choice` over sixteen
lines too long for the context) is that question's `error` answer with its
code, beside its siblings' answers, not a 422.

**`found`** — on `log` + `template_fold`, `prose` + `none` and `records` +
`none` — says whether the text answers at all. The last `choice` is asked
again, in the same request, with one more option "no line (sentence) of the
evidence answers the criterion", and for a folded log with a yes/no "Is there
a line in the evidence that answers this question: …". `found` is `1 -
p(none)` (averaged with the yes/no's `p(yes)` for a log). **Below 0.5** the
answer names nothing: `segment`, `value` and `confidence` are `null` and
`pointers` is empty, but `ranking` still lists the candidates — apply your own
threshold, or take the best guess knowingly. The other routes always name a
segment and carry no `found`.

**`confidence`, `ranking` and `pointers`** are the last `choice`'s
probabilities — under a fold, times the probability of the template the first
`choice` picked — **never calibrated probabilities**. `ranking` is the
candidates by that share, at most five, the pick first; `pointers` is every
candidate at 0.05 or more, the pick always among them: one answer, or the
several sentences a two-part answer needs.

- **Several lines identical but for their time** are one row of the fold; the
  answer is the first of them.
- **Folding removes order and neighbours.** Level 1 drops every time, a
  line's neighbours are not read, and a bracket-opened line (`[svc-a] …`) is
  read as a source label — unless the bracket holds a time
  (`[Sun Dec 04 04:47:44 2005] …`), which is then the line's timestamp. A question that needs context across lines, or
  names a line by its time alone, is better asked with `compression: "none"`.
- **Windows.** A text the heads read that is longer than 200,000 tokens is cut
  at segment boundaries — at paragraph breaks where there are any — and each
  window is read as its own prefill, with the window alone as its state. A
  window's prefix is kept while the questions over it run, so a second
  question over the same text costs its own short prefills, not the text's.
- **Every prefill is billed**: `usage.input_tokens` counts windows, their
  content-free baselines, a fold's levels and every `choice`; `output_tokens` is 0.
- The Playground's **Decide** tab asks one (GitHub #277, #278): it counts the
  segments `within` cuts before you send, offers the kind, method and
  compression, and shows the answer's pointers and `found`.

A `choice` over labelled segments is what the shortlist asks in its last step,
and you can still ask it yourself over a short text (Jev's "line search"):

- Prefix every non-empty line of a string `state` with its label and `: `, or
  turn an array into an object from label to element, in order.
- Declare the `choice`'s options under the same labels, in the same order, each
  with no description (`null`).
- Use the endpoint's own label order: `A`-`Z`, `a`-`z`, `0`-`9`, then the
  uppercase bigrams this tokenizer reads as one token (`AA`, `AB`, … — not
  `BQ`, which is two). The endpoint shows the model each option under the label
  at its position in that order, so options named with it put the same label in
  the state and in the answer.
- At most 256 segments, the endpoint's option ceiling — past about 16 to 32
  near-duplicates it degrades, which is why the shortlist narrows first.

---

## The Playground

The Playground (`web/`, React + Vite) is served at `/ui/` unless `--no-ui`, but
it is embedded into `ignis-server` only if `web/dist` exists when cargo builds
it — cargo never runs npm. Build the frontend first (needs Node.js), then the
server:

```
npm --prefix web ci
npm --prefix web run build
cargo build --release -p ignis-server
```

A server built without `web/dist` still serves `/ui/`, as a page with these
instructions.

For frontend work, `npm --prefix web run dev` proxies `/v1` and `/ui/metrics` to
a running engine (`IGNIS_URL`, default `http://127.0.0.1:8000`);
`npm --prefix web run dev:mock` serves a fake engine instead, so no GPU is
needed.

---

## Telemetry and metrics

Everything goes through one structured log on stdout, in the format
`IGNIS_LOG_FORMAT` picks (pretty on a terminal, JSON otherwise);
`IGNIS_LOG_LEVEL` sets the level.

- The request lifecycle is the `ignis.request.*` events (`admitted` / `ttft` /
  `done`), with the request id doubling as the OTel trace id
  ([ADR 0012](../adr/0012-request-id-as-trace-id.md)).
- The scheduler counters are the DEBUG event `ignis.scheduler.interval`, emitted
  whenever `waiting`, `running` or `kv_evictions` change
  ([ADR 0025](../adr/0025-scheduler-interval-as-a-log-event.md)):

```text
2026-09-13T10:00:00.000Z DEBUG ignis.scheduler.interval - scheduler counters changed {tick=123, waiting=2, running=3, kv_evictions=1}
```

- The VRAM layout decided at load is the `ignis.runtime.vram_plan` event.

With `--metrics`, the Prometheus exposition covers the request lifecycle and
rejections, KV pool pages and evictions, the KV-RAM arena, retained slots and
what reuse held or gave up, the VRAM plan, decoded tokens, forced thinking
closes, and decision counters with the answer-mass histogram ([ADR 0017](../adr/0017-prometheus-metrics.md)).
The Playground's **Monitor** tab scrapes the same exposition.

---

## Building from source

Two parts, in this order: the C++/CUDA kernel leaf, then the Rust workspace that
links it.

**Prerequisites**

- **Rust** — the workspace (Cargo).
- **A C++ toolchain** — MSVC C++ build tools (Visual Studio 2022, C++ workload)
  on Windows; GCC on Linux.
- **The NVIDIA CUDA Toolkit** (`nvcc`), target `SM120a`. Set `CUDA_PATH`
  (`CUDA_HOME` on Linux) if it is not on the default install path.
- **CMake + Ninja** — the kernel leaf builds with the Ninja generator.
- **Node.js**, for the Playground.

### 1. The kernel leaf

`kernel/build.ps1` (Windows) and `kernel/build.sh` (Linux) configure and build
the leaf with CMake + Ninja + `nvcc`, targeting `SM120a`
(`CMAKE_CUDA_ARCHITECTURES=120a`) in a Release build, into `kernel/build/` —
which `crates/*/build.rs` link. The PowerShell script imports the MSVC
environment itself, so no developer prompt is needed.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File kernel\build.ps1
```

A second argument is an alternate build directory (`kernel\build.ps1 build-a`),
so a parallel workstream can verify new `.cu` files without contending on the
canonical `kernel/build/`.

### 2. The Rust workspace

```
# GPU-backed: the real model, needs the kernel leaf built
cargo build --release -p ignis-server --features cuda

# CPU-only mock: protocol and loop work, no GPU, no kernel leaf
cargo build --release -p ignis-server
```

`--features cuda` enables the production GPU backend. Without it, or without an
artifact, the server runs the deterministic CPU mock: the endpoints, streaming,
telemetry and scheduler behaviour are real, the generated text is not, and every
completion reports `finish_reason: "length"` because the mock has no EOS token.

The cargo build reuses an already-built `kernel/build/ignis_kernel.lib` when
present, so it does not recompile the C++ leaf each time. On the MSVC triple the
binary lands in `target/x86_64-pc-windows-msvc/release/`.

Windows is the development host; Linux builds the same engine and is what the
container image is built from. Some `make` targets are still Windows-only there
— background server control (`start` / `stop` / `dev-ui`) and the GPU test
profile; `mk/os/linux.mk` names them.

### Tests

```
cargo test          # workspace-wide, CPU-only and fast — never touches the GPU
```

GPU work is checked by a separate, **explicit** suite (the GPU profile): it
requires the card to be free and **fails** — never skips — when the GPU is busy
or a kernel errors ([ADR 0006](../adr/0006-exclusive-gpu-testing.md)). The
kernel leaf additionally has its own CTest executable running each vendored op's
reference test at real 27B geometry. The card fits one run at a time: check
`make gpu-status` first.

---

## Working with make

The `Makefile` wraps all of the above (GNU make and a POSIX `sh` — on Windows,
Git for Windows' `usr\bin` on `PATH`). `make` alone lists every target.

| Target | What it does |
|---|---|
| `make doctor` | Toolchain, rust target, artifact, web deps. |
| `make build` / `make release` | Build the server (CUDA=1 builds the kernel leaf too) / a clean-slate release build. |
| `make dev` | Build, then run in the foreground. |
| `make run` | Run the last build; fails, naming the changed files, if it is stale. |
| `make mock` | `make dev` on the CPU mock: no GPU, no kernel leaf, no artifact. |
| `make start` / `stop` / `restart` / `status` / `logs` | The background server; it outlives the terminal. Log and pid under `.scratch/serve/`. |
| `make redeploy` | Build, and only if it succeeds: stop + start. |
| `make dev-ui` / `run-ui` | Server plus Playground hot reload as one session; Ctrl+C stops both. |
| `make web-dev` / `web-mock` | Vite alone, against a running engine or an in-process fake one. |
| `make smoke` / `canary` | One `/v1/models` + a short completion / the canary suite. |
| `make metrics` | Scrape the metrics listener of a server started with `METRICS=1`. |
| `make test` / `test-all` / `ci` | `cargo test` / plus web typecheck and vitest / what a CI job runs. |
| `make gpu-status` / `gpu-guard` / `gpu-profile` | Who holds the card / refuse when it is held / the explicit GPU test profile. |
| `make watch CUDA=0` | Rebuild + restart on every Rust change (cargo-watch). |
| `make clean` / `distclean` | This profile's binary / everything. |

Knobs go on the command line (`make dev CUDA=0 PROFILE=dev`,
`make dev-ui SPEC= MAX_CONTEXT=40960`) or in an untracked `local.mk`
(`local.mk.example`). `make config` prints what is in force.

With `CUDA=1` the server starts with `MAX_CONTEXT=524288` and
`ROPE_SCALING=yarn:2` -- twice the checkpoint's trained 262,144 positions,
and the YaRN table that stretches the envelope to match (the gate legs ran at
`MAX_CONTEXT=262144 ROPE_SCALING=none`) -- `KV_FORMAT=hq-e8-2b`,
`SPEC=dflash2` with
`DRAFT_TOKENS=7`, `KV_HOST_POOL_BYTES=8G`, `REQUEST_TIMEOUT=1800`. `SPEC=` turns
speculation off. `VISION=1` loads the tower; `UNCENSORED=1` loads
[the uncensored variant](#the-uncensored-variant) in place of the default
image; `ROPE_SCALING=yarn:4` rescales the
envelope further (`none` keeps the trained table); `METRICS=1` starts the metrics listener; `API_KEY=` and `EXPOSE=` set
the access flags; `ARGS='…'` passes anything verbatim.

A CUDA `run` or `start` first runs a GPU guard (the preflight plus a check for
other Ignis GPU work), skippable with `GPU_CHECK=0`.

---

## Cutting a release

The version lives in four files: `Cargo.toml` and `web/package.json` (the
Playground ships inside the binary) and the two lockfiles, which the next
build would rewrite. The release workflow runs the same `mk/version.sh check`
`make version-check` does, so all four have to agree there too.

```
make version                     # what each of the four declares
make version-bump T=patch        # or minor, major
make version-bump V=1.2.3-rc.1   # or an exact version: x.y.z, -prerelease optional
make version-check               # all four files agree, before you tag
```

`version-bump` edits and stops there: the commit and the tag are printed, not
run, because pushing a `v*` tag publishes a release.

What the release changed is drafted from the range, not written from memory:

```
make changelog                   # since the last v* tag, to HEAD
make changelog FROM=v0.1.1 TO=v0.1.2
```

It reads the issues the range's commits name, the ADRs added or amended in it,
and the commits that name no issue, and prints Markdown for the version-bump
commit and the GitHub release body. It is a draft: the headings are sorted by
what the range can prove, not by what matters most.

`.github/workflows/release.yml` runs the CPU-only test suite on Linux and
Windows, and only then builds the GPU engine for both hosts — on every push to
`main` or a `ci/**` branch, publishing nothing. A `v*` tag — which must match
`workspace.package.version`, or the job refuses it — turns the same run into a
GitHub Release and pushes the container image. The image is built by the same
job as the Linux tarball, out of the same compile, so the two can never carry
different binaries (ADR 0032).

CI cannot run the GPU profile: no runner has an NVIDIA card. Correctness on
the card stays on the development machine (`docs/agents/testing.md`).
