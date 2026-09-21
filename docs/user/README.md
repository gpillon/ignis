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
| `--reasoning-effort <value>` | `IGNIS_REASONING_EFFORT` | template default | The server-wide default `reasoning_effort`. |
| `--system-message-policy <p>` | `IGNIS_SYSTEM_MESSAGE_POLICY` | `merge` | `merge`: a leading run of system messages joins the system prompt, a later one is its own block in place. `strict`: a system message that is not first is a 400. |
| `--developer-message-policy <p>` | `IGNIS_DEVELOPER_MESSAGE_POLICY` | `inplace` | One of `inplace`, `into-system`, `after-system`, `one-after-system`, `reject`. A leading developer message is the system prompt except under `reject`. |

### Context, KV and speculation

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--max-context <tokens>` | `IGNIS_MAX_CONTEXT` | `40960` | Max per-sequence context (prompt + generation). The KV pool must be able to hold one of them or the load is refused. |
| `--prefill-chunk <tokens>` | `IGNIS_PREFILL_CHUNK` | `1024` | The prefill chunk width, a nonzero multiple of 128. The program's prefill scratch is reserved for it at load. |
| `--kv-format <fmt>` | `IGNIS_KV_FORMAT` | `hq-e8-2b` | `hq-e8-2b` (serving) or `bf16` (retained; the format every correctness oracle runs against — [ADR 0022](../adr/0022-two-kv-formats-bf16-as-oracle.md)). Decides what a pool byte budget is worth in tokens. |
| `--kv-pool-bytes <bytes>` | `IGNIS_KV_POOL_BYTES` | the rest of the VRAM budget | The paged-KV pool budget (accepts `K`/`M`/`G`). Too small for `--max-context`, or past the VRAM budget, fails the load by name. |
| `--kv-host-pool-bytes <bytes>` | `IGNIS_KV_HOST_POOL_BYTES` | `2G` | The KV-RAM host tier. Page-locked whole at start and held for the life of the load, so it is RAM the process holds even idle — and the figure Windows reports as shared GPU memory. `0` disables the host tier. (`make` sets `8G`.) |
| `--spec <backend>` | `IGNIS_SPEC` | unset | Speculative decoding backend: `dflash2`, or unset for none. |
| `--draft-tokens <n>` | `IGNIS_DRAFT_TOKENS` | — | Required with `--spec`; 1..7. |
| `--rope-scaling <spec>` | `IGNIS_ROPE_SCALING` | none | `yarn:F[,t=<c>][,bf=<n>][,bs=<n>]` rescales the checkpoint's trained 262,144-position envelope. `F` in (1, 64]. |

### VRAM budget and retained state

Laid out at load and printed as the `ignis.runtime.vram_plan` event
([ADR 0030](../adr/0030-device-memory-reserved-at-load.md)). A plan that cannot
hold one full context refuses the start.

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--vram-headroom-bytes <b>` | `IGNIS_VRAM_HEADROOM_BYTES` | `1G` | Derives the budget: the device memory free at start minus this. Not with `--vram-budget-bytes`. |
| `--vram-budget-bytes <b>` | `IGNIS_VRAM_BUDGET_BYTES` | derived | The device memory the whole process may hold, weights included. More than is free refuses the start. |
| `--allow-vram-oversubscription` | `IGNIS_ALLOW_VRAM_OVERSUBSCRIPTION` | off | With `--vram-budget-bytes` only: start above free memory (or below the plan's minimum) with a warning instead of a refusal. On Windows that pages. |
| `--prompt-reuse <on\|off>` | `IGNIS_PROMPT_REUSE` | `on` | `off`: no prompt checkpoint is captured or reused, and no prefix is shared unless `--retained-slots` gives slots for it. |
| `--retained-slots <n>` | `IGNIS_RETAINED_SLOTS` | one per decode lane (8); `0` with `--prompt-reuse off` | Slots reserved at load for the images of retained checkpoints and shared prefixes. When none is free, retained state gives one up — checkpoints before prefixes, `agent` before `interactive`, then least recently used; when nothing can, the publish or capture is skipped. |
| `--retained-interactive-ttl <secs>` | `IGNIS_RETAINED_INTERACTIVE_TTL` | `300` | Idle seconds after which a main-conversation checkpoint in KV-RAM ranks as a subagent's. Needs `--prompt-reuse on`. |

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
one is set; the Playground's static pages do not.

| Endpoint | Method | Notes |
|---|---|---|
| `/v1/models` | GET | The loaded model. |
| `/v1/chat/completions` | POST | Chat completions — streaming (`stream: true`, SSE) and non-streaming. |
| `/v1/responses` | POST | The OpenAI responses API (non-streaming; `stream: true` is a 400). |
| `/v1/decide` | POST | The decision endpoint ([below](#decisions)). |
| `/v1/systemone` | POST | The same handler under its Jev name. |
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
- Values outside the ranges above are rejected with
  `invalid_sampling_parameter`; they are never silently clamped. Absent
  sampling fields preserve greedy, fixed-seed behaviour (`temperature: 0`,
  `seed: 0`). Because the leaf's greedy branch does not read stochastic filters
  or penalties, a non-neutral `top_p`, `top_k`, `presence_penalty` or
  `frequency_penalty` requires `temperature > 0` and is otherwise rejected
  rather than ignored.
- Non-streaming returns `choices[].message.content` plus `usage`; streaming
  emits `chat.completion.chunk` SSE frames (token deltas, a final
  `finish_reason` chunk, then `[DONE]`).

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
what reuse held or gave up, the VRAM plan, decoded tokens, and decision counters
with the answer-mass histogram ([ADR 0017](../adr/0017-prometheus-metrics.md)).
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

With `CUDA=1` the server starts in the gate configuration: the full
`MAX_CONTEXT=262144`, `KV_FORMAT=hq-e8-2b`, `SPEC=dflash2` with
`DRAFT_TOKENS=7`, `KV_HOST_POOL_BYTES=8G`, `REQUEST_TIMEOUT=1800`. `SPEC=` turns
speculation off. `VISION=1` loads the tower; `ROPE_SCALING=yarn:4` rescales the
envelope; `METRICS=1` starts the metrics listener; `API_KEY=` and `EXPOSE=` set
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
