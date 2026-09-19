# ignis

A high-throughput inference engine for **Qwen 3.8-27B** on a single **NVIDIA
RTX 5090** (SM120a), serving an **OpenAI-compatible HTTP API** from a Rust core
backed by a C++/CUDA kernel leaf.

**Status:** early development — **the engine now serves real completions on
the GPU.** With `IGNIS_ARTIFACT` set and the binary built with `--features
cuda`, `ignis-server` loads the model onto the device and drives the real
64-layer program (device-resident, step-level C ABI — ADR 0009, verbatim-
vendored reference kernels — ADR 0010) for both streaming and non-streaming
chat completions, stopping on the model's own EOS token or `max_tokens`.
Without an artifact, or without `--features cuda`, it falls back to the
deterministic CPU-only mock (`MockCompute`, ADR 0006) for protocol and loop
work.

**Gate history:** G1 GREEN (2026-09-07, 97.1% teacher-forced canary
agreement), G2 GREEN (2026-09-09, TTFT ratios 0.878/0.851 on the 8K/32K
cells), G3 closed 2026-09-10 with one open gap (#110, the ITL p95 cell).
**Next milestone: the G4 gate run** (master #65, the reference feature
floor) — hq-e8-2b KV as a load option, snapshot/restore + KV-RAM host tier,
device prefix reuse, unified eviction, tagged lanes; verdict is the 99%
performance gate (ADR 0007) on the recorded "1 main + N subagents" load.
The full phase/gate plan is `.scratch/ROADMAP.md`; the review that reset it
is `.scratch/REVIEW-2026-09-05.md`.

---

## What it is

`ignis` is a deliberately specialized inference engine: one model family, one
GPU class. It is **not** a recreation of the reference stack (NInfer) — it is a
new architecture that borrows proven kernel work where it helps.

- **Performance-first.** Correctness (a self-check of *sane* output) is a
  non-negotiable floor; above it, performance is the #1 objective. The
  north-star is *"the best local coding engine"* — maximum throughput **and**
  agent parallelism that saturates the GPU in prefill *and* decode.
- **Rust core + C++/CUDA leaf.** Rust owns everything above the *step* —
  scheduling, KV accounting, serving; the forward pass and all GPU compute live
  in a static C++/CUDA library behind a flat, device-resident step-level C ABI
  (ADR 0009).

The engine is also its own dogfood target: good enough to run a developer's own
concurrent coding agent (a "1 main agent + N subagents" load).

## Architecture

Two layers, plus the artifact that carries the weights and the frontend
objects. Items marked *(planned)* are the G1–G4 build-out, not shipped code:

```
HTTP (OpenAI-compatible: /v1/models, /v1/chat/completions, /v1/responses)
  │
Rust core (crates/core)
  ├── Scheduler: prefill + N=8 decode lanes (host-tier overflow)
  │              + full admission state machine (protection / backfill class /
  │              temporal credit / frontier distance)
  ├── Paged KV page accounting + block tables (device pages reported by the leaf)
  ├── KV-RAM host tier (probation / protected eviction) + prefix reuse *(in progress: G4)*
  ├── Artifact loader (.ninfer reader + binder + materializer + device views)
  ├── Telemetry (JSONL interval lines; request lifecycle as structured logs)
  └── crates/runtime: safe wrapper over the step ABI *(shipped: G1)*
        │
Step-level C ABI (device-resident, opaque handles — ADR 0009)
  model load · sequence alloc/release/snapshot · prefill(span, pos)
  · decode round(batch) · sampling params · stats
        │
Kernel leaf (kernel/, C++/CUDA static lib — CMake + nvcc, SM120a)
  ├── program: device arena, streams, the 64-layer op sequence,
  │            sequence state (KV pages, fp32 GDN slots, conv taps)
  └── vendored ops (verbatim from the reference, ADR 0010)
      ├── NVFP4 / BF16 / W8G32 linear (GEMV, small-T; W4A4 + TMA since G2)
      ├── GQA attention (bf16 + hq-e8-2b paged decode + prefill; i8 unused)
      ├── GDN family (causal conv1d + SiLU, gating, recurrence, chunked since G2)
      ├── norms / embedding / sampling
      └── per-width decode CUDA graphs *(shipped: G3, widths 1..8)*
```

**The model lifecycle is decoupled from the server lifecycle** (hot-reload-ready
by construction): the KV pool, CUDA graphs, and scheduler state are regenerable
per model. The per-sequence context defaults to 40960 tokens (a 32K prompt
plus an 8K generation, `--max-context`), the model's 262k envelope being the
ceiling; the paged KV pool is sized in **bytes**: whatever the **VRAM budget**
leaves once the weights, workspaces, lanes and retained state are laid out
(ADR 0030), or `--kv-pool-bytes` when named. The budget is the memory free at
start minus a 1 GiB headroom (`--vram-headroom-bytes`) or an explicit
`--vram-budget-bytes`, and a plan that cannot hold one full context refuses
the start. What that budget is worth in tokens is derived from the KV
format in force (`--kv-format`, hq-e8-2b by default since its attention
routes landed): 65536 sequence-tokens under BF16, 7.11x that under hq-e8-2b,
reported at load. BF16 is retained and is the format every correctness
oracle runs against (ADR 0022). The target max concurrency is N=8 (resident
lanes with host-tier overflow, sized for a ~10-subagent concurrent workload).

## Repo layout

```
ignis/
├── crates/
│   ├── core/        # scheduler, paged KV accounting, request state machine, host tier
│   ├── artifact/    # .ninfer reader (reader / binder / materializer)
│   ├── runtime/     # safe step-ABI wrapper (CudaLeaf, decode graphs)
│   ├── server/      # HTTP + OpenAI schemas + telemetry
│   ├── logging/     # structured logging (tracing layers, hotpath lint, trace context)
│   ├── bench/       # trace-replay harness + gate/canary runner
│   └── vendor/      # ADR 0010 vendoring tool (manifest, hashes, patch records)
├── web/             # the Playground: React + Vite page served at /ui/ with --ui (ADR 0026)
├── kernel/          # C++/CUDA leaf: program + vendored ops (CMake + nvcc) + build.ps1
├── bench/traces/    # recorded load traces (JSONL; only the *.meta.json ship)
├── scripts/         # gpu-preflight / gpu-profile / vendor-ninfer (PowerShell)
├── docs/            # adr/, design/, agents/, findings/
├── CONTEXT.md       # glossary (domain vocabulary only)
└── AGENTS.md        # agent conventions (issue tracker, testing)
```

---

## Prerequisites

- **Rust** — the workspace (Cargo).
- **A C++ toolchain for the kernel leaf** — MSVC C++ build tools (Visual Studio
  2022, C++ workload) on Windows; GCC on Linux.
- **NVIDIA CUDA Toolkit** (`nvcc`) — target `SM120a`; set `CUDA_PATH`
  (`CUDA_HOME` on Linux) if it is not on the default install path.
- **CMake + Ninja** — the kernel leaf builds with the Ninja generator.
- **The `.ninfer` model artifact** — weights + tokenizer + chat-template (the
  frontend object set).

Windows is the development host; Linux builds the same engine
(`kernel/build.sh`, `mk/os/linux.mk`) and is what the container image below is
built from. Some `make` targets are still Windows-only there — `mk/os/linux.mk`
names which, GitHub #167.

## Build

The build has two parts: the C++/CUDA kernel leaf, then the Rust workspace that
links it.

### With make (the short path)

The `Makefile` wraps the steps below (needs GNU make and a POSIX `sh` — on
Windows, Git for Windows' `usr\bin` on `PATH`). `make` alone lists every
target; the everyday ones:

```
make doctor          # toolchain, rust target, artifact, web deps
make dev             # build (web + kernel + GPU server), then run it
make run             # run the last build; fails, naming the changed files, if it is stale
make mock            # the same on the CPU mock (no GPU, no kernel, no artifact)
make start / stop    # daemon server (log in .scratch/serve/), waits for /v1/models, survives the terminal
make dev-ui          # build, then server + Playground hot reload; Ctrl+C stops both
make metrics         # scrape the metrics listener of a server started with METRICS=1
make watch CUDA=0    # rebuild + restart on every Rust change (cargo-watch)
make test            # cargo test, workspace-wide
make gpu-status      # who holds the 5090
```

With `CUDA=1` the server starts in the G5 gate configuration: the full
262144-token context, hq-e8-2b KV, DFlash2 speculation with 7 draft tokens
(`MAX_CONTEXT`, `KV_FORMAT`, `SPEC`, `DRAFT_TOKENS`, `PREFILL_CHUNK`,
`REQUEST_TIMEOUT`; `SPEC=` turns speculation off).
Knobs go on the command line (`make dev CUDA=0 PROFILE=dev`,
`make dev-ui SPEC= MAX_CONTEXT=40960`) or in an untracked `local.mk`
(`local.mk.example`); `make config` prints what is in force. A CUDA
`run`/`start` first runs a GPU guard (the preflight plus a check for other
ignis GPU work), skippable with `GPU_CHECK=0`. Per-OS behavior sits behind
`mk/os/<os>.mk`: Windows is implemented, Linux is scaffolded (the CPU mock,
web and test targets work; kernel, GPU and background-server hooks report
"not implemented").

### 1. Kernel leaf (C++/CUDA)

`kernel/build.ps1` configures and builds the kernel leaf with CMake + Ninja +
`nvcc`, targeting `SM120a` (`CMAKE_CUDA_ARCHITECTURES=120a`) in a Release
build, into `kernel/build/` (the `crates/*/build.rs` link that artifact). It
imports the MSVC environment itself — no developer prompt needed — and locates
`nvcc` from `CUDA_PATH` (or the default CUDA install).

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File kernel\build.ps1
```

A second argument is an alternate build dir (e.g. `kernel\build.ps1 build-a`),
so a parallel workstream can verify new `.cu` files without contending on the
canonical `kernel/build/`.

### 2. Rust workspace

```
# GPU-backed (real model, needs the kernel leaf built and IGNIS_ARTIFACT set)
cargo build --release -p ignis-server --features cuda

# CPU-only mock (protocol/loop work, no GPU, no kernel leaf needed)
cargo build --release -p ignis-server
```

- **`--features cuda`** enables the production GPU-backed compute backend
  (`ignis_runtime::CudaLeaf`, driven through `ignis_server::runtime::cuda_scheduler`)
  — with `IGNIS_ARTIFACT` set, the real model path.
- **Without `--features cuda`, or without an artifact**, the server runs in
  ADR 0006 dev mode: a deterministic CPU-only mock (`MockCompute`), for
  protocol and loop work without a GPU.
- Drop `--release` for a debug build (slower, faster to compile); the binary
  then lands under `target/x86_64-pc-windows-msvc/debug/` instead of `release/`.

The cargo build reuses an already-built `kernel/build/ignis_kernel.lib` when
present (incremental), so it does not recompile the C++ leaf from scratch each
time. The resulting binary lands in the workspace target dir — on the MSVC
triple, `target/x86_64-pc-windows-msvc/debug/ignis-server`.

### 3. Playground (optional)

The Playground (`web/`, React + Vite, ADR 0026) is embedded into `ignis-server`
only if `web/dist` exists when cargo builds it — cargo never runs npm. Build
the frontend first (needs Node.js), then the server, then run with `--ui`:

```
npm --prefix web ci
npm --prefix web run build
cargo build --release -p ignis-server
target\x86_64-pc-windows-msvc\release\ignis-server.exe --ui    # http://127.0.0.1:8000/ui/
```

A server built without `web/dist` still accepts `--ui` and serves a page with
these instructions. For frontend work, `npm --prefix web run dev` proxies `/v1`
and `/ui/metrics` to a running ignis (`IGNIS_URL`, default
`http://127.0.0.1:8000`); `npm --prefix web run dev:mock` serves a fake engine
instead, so no GPU is needed.

```
cargo test          # workspace-wide, CPU-only and fast — never touches the GPU
```

GPU work is checked by a separate, **explicit** suite (the GPU profile): it
requires the 5090 to be free and **fails** — never skips — when the GPU is busy
or a kernel errors (ADR 0006; a skip is not green for compute work). The kernel
leaf additionally has its own CTest executable running each vendored op's
reference test at real 27B geometry (ADR 0010).

## Releases and the container image

`.github/workflows/release.yml` builds the GPU engine for both hosts on every
push to `main` or a `ci/**` branch, and publishes nothing. A `v*` tag — which
must match `workspace.package.version`, or the job refuses it — turns the same
run into a GitHub Release (a Windows `.zip` and a Linux `.tar.gz`, each with
its SHA-256) and pushes the `linux/amd64` image to
`ghcr.io/gpillon/ignis`.

```
podman run --rm --device nvidia.com/gpu=all -p 8000:8000 \
  -v /path/to/models:/models:ro \
  -e IGNIS_ARTIFACT=/models/qwen3_8_27b_nvfp4full-v2.ninfer \
  ghcr.io/gpillon/ignis:0.1.0
```

(`docker`: `--gpus all` in place of `--device`.) Every flag has an `IGNIS_*`
environment variable (`crates/server/src/config.rs`); anything after the image
name is passed to the server, and `--ui` — a bare switch with no environment
variable — is the image's default command. The image carries the CUDA runtime
but no driver: the host's NVIDIA driver is injected by the container runtime,
and the model is mounted, never baked in.

`Containerfile` builds the same thing locally (`podman build -t ignis:dev .`).
Its `artifacts` stage is what CI exports the Linux tarball from, so the release
binaries and the image binaries are the same build.

Both are compiled for **SM120a** only. The Linux tarball needs the CUDA 13
runtime on the host; the Windows zip carries `cudart64_*.dll`.

## Models (`./models`)

`./models` is a symlink to `F:\ai\q38\ninfer-models` (the shared model store —
also used by `ninfer` itself, on the same 5090). Current contents:

| File | Size | Notes |
|---|---|---|
| `qwen3_8_27b_nvfp4full-v2.ninfer` | ~19.4 GB | **The correct artifact — use this one.** v2: same base tensors as v1 (bit-identical) plus a grafted DFlash2 speculative-decoding drafter module. Has a matching `.sha256` and `.README.md` (full provenance) alongside it. |
| `qwen3_8_27b_nvfp4full.ninfer` | ~18.3 GB | v1 (pre-DFlash2). Legacy — kept for comparison/rollback, not the one to point `IGNIS_ARTIFACT` at. |
| `qwen3_8_27b_nvfp4full-v2.ninfer.graft.json` | — | The DFlash2 graft manifest for v2 (which objects were appended, and how). |
| `qwen3_8_27b_nvfp4full-v2.ninfer.sha256` | — | Checksum for v2; the server verifies it at load and refuses to start if it does not match. |
| `qwen3_8_27b_nvfp4full.ninfer.conversion.json` | — | v1's conversion manifest. |
| `.cache/`, `.ninfer-webui.*.tmp/`, `webui/` | — | ninfer's own scratch/webui state — not ours, ignore. |

Point `IGNIS_ARTIFACT` at the v2 file (relative path works since `./models` is
a symlink into the real store):

```
set IGNIS_ARTIFACT=./models/qwen3_8_27b_nvfp4full-v2.ninfer
```

The grafted DFlash2 drafter module is what phase 5 (gate G5, GitHub #66,
spec `.scratch/runtime/specs/05-speculative-decoding.md`) loads for
speculative decoding. Until that phase lands, ignis binds only the text scope:
the drafter module sits unused in the container and costs no VRAM unless
materialized.

## Usage (in development)

The server is an **OpenAI-compatible** HTTP server on localhost (no auth,
localhost-only by design). The current API surface is the v1 OpenAI API and will
evolve as the engine matures.

> **Real completions on the GPU.** Built with `--features cuda` and a
> verified `IGNIS_ARTIFACT`, the server loads the ~19 GB of weights into
> VRAM, builds a paged KV pool from what its VRAM plan leaves across 8 decode
> slots (logged as `ignis.runtime.vram_plan`, ADR 0030), each
> sequence capped at the 40960-token default context
> (`ignis_runtime::CudaLeafConfig`), and drives the real 64-layer
> program for every request: streaming and non-streaming chat completions
> stop at the model's own EOS token (`finish_reason: "stop"`) or at
> `max_tokens` (`finish_reason: "length"`). Without `--features cuda`, or
> without an artifact, completions come from the deterministic CPU mock
> (`MockCompute`) instead — the endpoints, streaming, telemetry and
> scheduler behavior are real, the generated text is not (and every
> completion reports `finish_reason: "length"`, since the mock has no real
> EOS token).

### Configuration (environment / CLI flags)

Each variable has a matching CLI flag (a flag overrides its env var, which
overrides the default — GitHub #77); run `ignis-server --help` for the
full, always-current table.

| Variable | Flag | Alias | Default | Meaning |
|---|---|---|---|---|
| `IGNIS_ARTIFACT` | `--artifact <path>` | `-a` | — (unset) | The `.ninfer` container path (weights + tokenizer + chat template). **Unset → the built-in placeholder template**, whose rendered content is not natural text. A configured artifact is verified (checksum clean) or the server refuses to start. |
| `IGNIS_MODEL` | `--model <id>` | `-m` | `qwen3.8-27b` | The loaded model id (what `/v1/models` reports and what submissions must name). |
| `IGNIS_BIND` | `--bind <addr>` | `-b` | `127.0.0.1:8000` | The bind address (localhost only, no auth). |
| `IGNIS_ENABLE_THINKING` | `--enable-thinking <true\|false>` | — | `true` | The server-wide default for `enable_thinking`. |
| `IGNIS_REASONING_EFFORT` | `--reasoning-effort <value>` | — | — (template default) | The server-wide default `reasoning_effort`. |
| `IGNIS_PREFILL_CHUNK` | `--prefill-chunk <tokens>` | — | `1024` | The prefill chunk width (a nonzero multiple of 128); the program's prefill scratch is reserved for it at load. |
| `IGNIS_MAX_CONTEXT` | `--max-context <tokens>` | — | `40960` | The max per-sequence context (prompt + generation); the KV pool must be able to hold one of them. |
| `IGNIS_KV_FORMAT` | `--kv-format <fmt>` | — | `hq-e8-2b` | The KV cache format for this load: `hq-e8-2b` (the serving default) or `bf16` (retained, and the format every correctness oracle runs against) — ADR 0022. Decides what a pool byte budget is worth in tokens. |
| `IGNIS_KV_POOL_BYTES` | `--kv-pool-bytes <bytes>` | — | the rest of the VRAM budget | The paged-KV pool budget in bytes (accepts a `K`/`M`/`G` suffix). A budget too small for `--max-context`, or past the VRAM budget, fails the load by name. |
| `IGNIS_VRAM_HEADROOM_BYTES` | `--vram-headroom-bytes <bytes>` | — | `1G` | Derives the VRAM budget: the device memory free at start minus this. Not with `--vram-budget-bytes`. |
| `IGNIS_VRAM_BUDGET_BYTES` | `--vram-budget-bytes <bytes>` | — | — (derived) | The device memory the whole process may hold, weights included. More than is free refuses the start. |
| `IGNIS_ALLOW_VRAM_OVERSUBSCRIPTION` | `--allow-vram-oversubscription` | — | off | With `--vram-budget-bytes` only: start above free memory (or below the plan's minimum) with a warning instead of a refusal. On Windows that pages. |
| `IGNIS_RETAINED_SLOTS` | `--retained-slots <n>` | — | one per decode lane (`N_DECODE_LANES`); `0` with `--prompt-reuse off` (a count given then shares heads between live siblings only) | Retained slots reserved at load (ADR 0030): where every retained prompt checkpoint and shared prefix keeps its state image. When none is free, retained state gives one up — checkpoints before retained prefixes, `agent` before `interactive`, then least recently used; when nothing can, the publish or capture is skipped. Replaces the removed `--retained-pool-bytes`. |
| `IGNIS_REQUEST_TIMEOUT` | `--request-timeout <secs>` | — | `30` (max 3600) | The deadline for a non-streaming completion; expiry is a 504 `request_timeout`. |
| — | `--ui` | — | off | Serve the Playground at `/ui/` (flag only, no env var — ADR 0026). |
| — | `--metrics` | — | off | Serve Prometheus metrics on their own listener, and at `/ui/metrics` with `--ui` (flag only, no env var — ADR 0017). |
| — | `--metrics-bind <addr>` | — | `127.0.0.1:9464` | The metrics listener's address; needs `--metrics`. No API key, never exposed. |
| — | `--help` | `-h` | — | Print the flag table and exit. |
| — | `--version` | `-V` | — | Print the crate version and exit. |

### Launch

```
set IGNIS_ARTIFACT=./models/qwen3_8_27b_nvfp4full-v2.ninfer
target\x86_64-pc-windows-msvc\release\ignis-server.exe
```

At startup the server verifies the artifact, loads the real tokenizer + chat
template, initializes the compute backend, then binds and serves. It prints
a one-line readiness note (`model <id> on http://<bind>`) when ready.

### API

| Endpoint | Method | Notes |
|---|---|---|
| `/v1/models` | GET | The loaded model. |
| `/v1/chat/completions` | POST | Chat completions — streaming (`stream: true`, SSE) and non-streaming. |
| `/v1/responses` | POST | The OpenAI responses API (non-streaming; `stream: true` → 400). |
| `/ui/` | GET | The Playground page — only with `--ui`. |
| `/ui/metrics` | GET | Prometheus text format 0.0.4 for the Playground — only with `--ui` and `--metrics` (ADR 0017). Needs the API key when one is set, like `/v1`. |

Prometheus scrapes `GET /metrics` on the metrics listener (`--metrics-bind`,
default `127.0.0.1:9464`), not on the API's address: no key, and never
reachable through `--expose`.

Errors use OpenAI's `{"error": {message, type, code}}` body with the matching
status: 400 bad request, 404 unknown model, 413 oversized request, 503 engine
full, 504 the engine did not finish the request in the timeout.

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

# the responses API
curl http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"input":"...","max_output_tokens":256}'
```

- **Chat completions** accept `messages` (role + content), `model`, `stream`,
  `max_tokens`, `temperature` (0..2), `top_p` (0..1), `presence_penalty` and
  `frequency_penalty` (-2..2), and a signed 64-bit `seed`. `top_k` is an
  **ignis extension**, not an OpenAI Chat Completions parameter: accepted
  values are 0..20, where 0 selects ignis's 20-candidate sampler cap during
  stochastic sampling. Values outside these ranges are rejected with
  `invalid_sampling_parameter`; they are never silently clamped. Absent
  sampling fields preserve the existing greedy, fixed-seed behavior
  (`temperature: 0`, `seed: 0`). Because the leaf's greedy branch
  intentionally does not read stochastic filters or penalties, a non-neutral
  `top_p`, `top_k`, `presence_penalty`, or `frequency_penalty` requires
  `temperature > 0` and is otherwise rejected instead of ignored.
  Non-streaming returns `choices[].message.content` + `usage`; streaming emits
  `chat.completion.chunk` SSE frames (token deltas, a final `finish_reason`
  chunk, then `[DONE]`).
- **The responses API** accepts `input` (a string or a message list), `model`,
  `max_output_tokens`, `temperature`, and `seed`; it returns the responses API
  `output` shape with the generated text in an `output_text` part.

### Canary check

`ignis-bench canary` runs the fixed high-signal prompt suite against a live
server and checks each output is *sane* and *deterministic* (greedy + fixed
seed ⇒ identical output on a repeat run — ADR 0007's self-check, not a
reference-token comparison):

```bash
cargo run -p ignis-bench -- canary --endpoint http://127.0.0.1:8000
```

## Telemetry

Everything goes through the one structured log on stdout, in the format
`IGNIS_LOG_FORMAT` picks (pretty on a terminal, JSON otherwise). The request
lifecycle is the `ignis.request.*` events (`admitted` / `ttft` / `done`,
GitHub #79), with the request id doubling as the OTel trace id (ADR 0012).

The scheduler counters are the DEBUG event `ignis.scheduler.interval`,
emitted whenever `waiting`, `running` or `kv_evictions` change (ADR 0025).
To watch them while load runs, start the server with `IGNIS_LOG_LEVEL=debug`:

```text
2026-09-13T10:00:00.000Z DEBUG ignis.scheduler.interval - scheduler counters changed {tick=123, waiting=2, running=3, kv_evictions=1}
```

`prefilling` and KV occupancy are not reported until the scheduler exposes
them as real values in `ignis-core`.

---

## Credits

The `.ninfer` model artifact (weights, tokenizer, and chat-template frontend
objects) and the CUDA ops originate from the **NInfer** project — a lineage of
Windows-oriented local-inference forks. The ignis kernel leaf **vendors those
ops verbatim** (Apache-2.0, under a pinned-commit manifest — ADR 0010; see
`kernel/NOTICE`) as a proven starting point (ADR 0005); our own kernels come
later, per op family and gated by measurement. The program layer above them
(the forward pass, the sequence state, the step ABI) is ours.

With thanks to the NInfer project and its contributors:

- **cometkim** — integration branch: kernel-perf (PDL decode chain, split-K
  prefill, per-request error boundary), DFlash2 base port, hyperquant KV cache,
  1M-context envelope, NVFP4-full target.
- **Mirko Covizzi** — RTX 5090 Laptop compatibility and MTP
  adaptive verification-width tuning.
- **mr-september** — warmup/readiness decoupling and frontend streaming fixes.
- **dylan (dylanbrodiefafard)** — RAM KV cache concept, LRU lane eviction,
  decode CPU-spin fix.
- **Neroued** — original NInfer engine.
