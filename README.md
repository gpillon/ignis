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

**Next milestone: the G1 gate run** — record the verdict (canary agreement
≥ 95% vs. the recorded oracle fixture, the f64 layer checks, reproducibility
across loads) and close out gate G1. The 99% performance gate (ADR 0007) sits
behind G4. The full phase/gate plan is `.scratch/ROADMAP.md`; the review that
reset it is `.scratch/REVIEW-2026-09-05.md`.

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
  ├── KV-RAM host tier (probation / protected eviction) + prefix reuse *(planned: G4)*
  ├── Artifact loader (.ninfer reader + binder + materializer + device views)
  ├── Telemetry (JSONL events + interval lines)
  └── crates/runtime: safe wrapper over the step ABI *(planned: G1)*
        │
Step-level C ABI (device-resident, opaque handles — ADR 0009)
  model load · sequence alloc/release/snapshot · prefill(span, pos)
  · decode round(batch) · sampling params · stats
        │
Kernel leaf (kernel/, C++/CUDA static lib — CMake + nvcc, SM120a)
  ├── program: device arena, streams, the 64-layer op sequence,
  │            sequence state (KV pages, fp32 GDN slots, conv taps)
  └── vendored ops (verbatim from the reference, ADR 0010)
      ├── NVFP4 / BF16 / W8G32 linear (GEMV, small-T; W4A4 + TMA at G2)
      ├── GQA attention (bf16 paged decode + prefill; i8/hq at G4)
      ├── GDN family (causal conv1d + SiLU, gating, recurrence, chunked at G2)
      ├── norms / embedding / sampling
      └── per-width decode CUDA graphs *(planned: G3)*
```

**The model lifecycle is decoupled from the server lifecycle** (hot-reload-ready
by construction): the KV pool, CUDA graphs, and scheduler state are regenerable
per model. The target context envelope is 262k (the KV pool auto-sized from free
VRAM); the target max concurrency is N=8 (resident lanes with host-tier overflow,
sized for a ~10-subagent concurrent workload).

## Repo layout

```
ignis/
├── crates/
│   ├── core/        # scheduler, paged KV accounting, request state machine, host tier
│   ├── artifact/    # .ninfer reader (reader / binder / materializer)
│   ├── server/      # HTTP + OpenAI schemas + telemetry
│   └── bench/       # trace-replay harness + canary-suite runner
├── kernel/          # C++/CUDA leaf: program + vendored ops (CMake + nvcc) + build.ps1
├── bench/traces/    # recorded load traces (JSONL)
├── docs/            # adr/, design/, agents/
├── CONTEXT.md       # glossary (domain vocabulary only)
└── AGENTS.md        # agent conventions (issue tracker, testing)
```

---

## Prerequisites

- **Rust** — the workspace (Cargo).
- **MSVC C++ build tools** (Visual Studio 2022, C++ workload) — for the kernel leaf.
- **NVIDIA CUDA Toolkit** (`nvcc`) — target `SM120a`; set `CUDA_PATH` if it is not on
  the default install path.
- **CMake + Ninja** — the kernel leaf builds with the Ninja generator.
- **The `.ninfer` model artifact** — weights + tokenizer + chat-template (the
  frontend object set).

## Build

The build has two parts: the C++/CUDA kernel leaf, then the Rust workspace that
links it.

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

```
cargo test          # workspace-wide, CPU-only and fast — never touches the GPU
```

GPU work is checked by a separate, **explicit** suite (the GPU profile): it
requires the 5090 to be free and **fails** — never skips — when the GPU is busy
or a kernel errors (ADR 0006; a skip is not green for compute work). The kernel
leaf additionally has its own CTest executable running each vendored op's
reference test at real 27B geometry (ADR 0010).

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

ignis does not use DFlash2 speculative decoding itself (that is a `ninfer`
CLI flag, `--spec dflash2`) — for ignis, v2 is used purely because it carries
the same verified base weights as v1 with nothing removed; the drafter module
sits unused in the container and costs no VRAM unless materialized.

## Usage (in development)

The server is an **OpenAI-compatible** HTTP server on localhost (no auth,
localhost-only by design). The current API surface is the v1 OpenAI API and will
evolve as the engine matures.

> **Real completions on the GPU.** Built with `--features cuda` and a
> verified `IGNIS_ARTIFACT`, the server loads the ~19 GB of weights into
> VRAM, builds a modest fixed-size KV pool (8 sequences × 4096 tokens —
> auto-sizing it from whatever VRAM the weights leave behind is later
> work, `ignis_runtime::CudaLeafConfig`), and drives the real 64-layer
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
| `IGNIS_TELEMETRY` | `--telemetry <path>` | `-t` | — (stdout) | The telemetry JSONL sink path (a file). |
| `IGNIS_ENABLE_THINKING` | `--enable-thinking <true\|false>` | — | `true` | The server-wide default for `enable_thinking`. |
| `IGNIS_REASONING_EFFORT` | `--reasoning-effort <value>` | — | — (template default) | The server-wide default `reasoning_effort`. |
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
  `temperature` (default 0.0 = greedy), `max_tokens`, and `seed`. Non-streaming
  returns `choices[].message.content` + `usage`; streaming emits
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

JSONL — one line per event and one line per interval. Useful for watching
scheduler behavior (admissions, evictions, throughput) while load runs:

```jsonl
{"kind":"interval","t":123,"waiting":2,"prefilling":1,"running":3,"kv_used_pct":62,"kv_evictions":0}
{"kind":"request","id":"r-042","event":"ttft","ms":226}
{"kind":"request","id":"r-042","event":"done","n":512,"tok_s":41.2}
{"kind":"evict","tier":"ram","reason":"capacity"}
```

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