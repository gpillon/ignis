# ignis

A high-throughput inference engine for **Qwen 3.8-27B** on a single **NVIDIA
RTX 5090** (SM120a), serving an **OpenAI-compatible HTTP API** from a Rust core
backed by a C++/CUDA kernel leaf.

**Status:** early development — **the engine does not yet produce a real
completion.** The Rust core above the step (HTTP + OpenAI surface, artifact
reader/binder/materializer, scheduler, admission, paged-KV accounting, host
tier, prefix index, bench harness) works and is CPU-tested; the forward pass
does not, and is being rebuilt device-resident behind a step-level C ABI
(ADR 0009) on top of verbatim-vendored reference kernels (ADR 0010).

**Next milestone: gate G1** — a correct, device-resident forward pass at
batch 1 with bf16 KV: coherent greedy completions on the canary suite, per-layer
activations within bf16 tolerance of an f64 reference, ≥ 95% first-32-token
agreement with the canary oracle, EOS honored, reproducible across loads.
The 99% performance gate (ADR 0007) sits behind G4. The full phase/gate plan
is `.scratch/ROADMAP.md`; the review that reset it is
`.scratch/REVIEW-2026-09-05.md`.

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
cargo build -p ignis-server --features cuda
```

- **`--features cuda`** enables the production CUDA compute backend
  (`CudaCompute`) — the real model path.
- **Without `--features cuda`**, the server runs in ADR 0006 dev mode: a
  deterministic CPU-only mock (`MockCompute`), for protocol and loop work
  without a GPU.

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

## Usage (in development)

The server is an **OpenAI-compatible** HTTP server on localhost (no auth,
localhost-only by design). The current API surface is the v1 OpenAI API and will
evolve as the engine matures.

> **The protocol works; the model does not yet.** Until gate G1, completions
> come from the deterministic CPU mock (`MockCompute`) — the endpoints,
> streaming, telemetry and scheduler behavior are real, the generated text is
> not. `--features cuda` with a real artifact loads and verifies the 19 GB of
> weights into VRAM, but the forward pass behind it is being rebuilt.

### Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `IGNIS_ARTIFACT` | — (unset) | The `.ninfer` container path (weights + tokenizer + chat template). **Unset → the built-in placeholder template**, whose rendered content is not natural text. A configured artifact is verified (checksum clean) or the server refuses to start. |
| `IGNIS_MODEL` | `qwen3.8-27b` | The loaded model id (what `/v1/models` reports and what submissions must name). |
| `IGNIS_BIND` | `127.0.0.1:8000` | The bind address (localhost only, no auth). |
| `IGNIS_TELEMETRY` | — (stdout) | The telemetry JSONL sink path (a file). |

### Launch

```
set IGNIS_ARTIFACT=F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer
target\x86_64-pc-windows-msvc\debug\ignis-server.exe
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