# ignis — v1 specification (design summary)

Grilled and settled 2026-09-02. This document is the shared understanding;
ADRs in `docs/adr/` carry the decisions that are hard to reverse.

## 1. Goals / non-goals

**Goal:** a Rust inference engine (`ignis`) specialized for Qwen 3.8-27B on a
single RTX 5090 (SM120a), serving OpenAI-compatible HTTP, reaching reference-parity
performance and architected for the later performance features (lazy CUDA graph,
hot reload, tagged lanes).

**Performance-first (ADR 0005) + north-star:** ignis is **our engine — a new
architecture, not a recreation of the reference (ninfer)**. The reference is a
*reference for inspiration only* (consulted only in extreme necessity);
borrowing pieces from other inference engines is fine as long as we stay on the
north-star. Correctness (a self-check of *sane* output) is a non-negotiable
floor; above it, performance is the #1 objective and the tie-breaker for every
scope, feature, and kernel decision. The **north-star** is "the best local
coding engine" — max performance **and** agent parallelism that saturates the
GPU in prefill *and* decode (end-state: v1 is also the dogfood target — good
enough to run the developer's own coding agent).

**Non-goals (v1):** other models, other GPU classes, continuous batching with
preemption, network exposure / auth, generic quantization formats, web UI.

## 2. Architecture

```
HTTP (OpenAI-compatible: /v1/models, /v1/chat/completions, /v1/responses)
  │
Rust core
  ├── HTTP server (localhost, no auth, configurable bind)
  ├── Scheduler: batched (concurrent) prefill to saturate the GPU (an experiment
  │               to verify — we may be compute-bound) + N=8 decode lanes (with
  │               host-tier overflow); full admission state machine (protection /
  │               backfill class / temporal credit / frontier distance)
  ├── Paged KV cache (VRAM) + block tables
  ├── KV-RAM host tier (probation/protected eviction; pulled into v1, not v1.1)
  ├── GDN state management (resumable at checkpoint/frontier boundaries only)
  ├── Artifact loader (.ninfer reader: reader + binder + materializer, Rust)
  ├── Telemetry (JSONL events + interval lines)
  └── FFI (flat C ABI, checked boundary)
        │
Kernel leaf (C++/CUDA static library, CMake + nvcc, SM120a)
  ├── NVFP4 GEMM (rowsplit/grouped MMA, cp.async pipelines)
  ├── GQA attention (prefill + decode, MRoPE 3-axis)
  ├── GDN linear-attention step
  ├── sampling, embeddings, norms
  └── CUDA graph capture (eager at startup in v1; lazy = later optimization)
```

- **Model lifecycle is decoupled from server lifecycle** (hot-reload-ready by
  construction): KV pool, CUDA graphs, and scheduler state are regenerable per
  model; in-flight requests finish or restart per a defined policy.
- **Default context envelope** 262k (KV pool auto-sized from free VRAM; ~12-13 GB
  pool budget after weights + graphs + runtime).
- **Default N (max concurrency) 8** (resident lanes, with host-tier overflow;
  sized for a ~10-subagent concurrent coding workload).

## 3. Milestones

| Milestone | Scope |
|---|---|
| v1 | core: **batched (concurrent) prefill** + **N=8** decode lanes (host-tier overflow), paged KV + **KV-RAM host tier** (probation/protected eviction) + **prefix reuse**, full admission state machine, OpenAI API (minimal + responses), `.ninfer` loader, eager CUDA graph, telemetry, trace-replay bench harness, **performance gate (99%, not parity)**, hot-reload-ready architecture. *One release — max feature throughput, max risk (chosen deliberately).* |
| v1.1 | *(KV-RAM host tier + prefix reuse pulled into v1)* — empty for now |
| v1.2 | DFlash2 speculative decoding (draft tokens 1..7) |
| v1.3 | MTP (adaptive verification width) |
| v1.4 | vision/multimodal |
| later | **re-implement the CUDA kernels** (guided by the north-star; ADR 0005), lazy CUDA graph capture, web UI (React+Vite+TS, same repo, CLI-disableable), hot reload, tagged lanes, other SM120a quantization formats |

## 4. Acceptance (v1)

- **Greedy + fixed seed.** **Performance gate (ADR 0007): ≥ 99% of the
  reference's performance** (throughput / latency) on the trace-replay load,
  with a per-class ttft/tok-s check. **Not** token-agreement — correctness is
  *self-checked* (the engine produces *sane* output for the same model, greedy,
  fixed seed); we do not require it to match the reference's output.
- **Trace replay:** a `bench` crate re-sends a recorded "1 main agent + N
  subagents" load trace (JSONL, recorded against the reference stack with the same
  harness) and produces the **performance report** (tok-s, ttft vs reference) +
  a self-consistency check.
- The reference is recorded with the same harness so comparisons are
  apples-to-apples (eager CUDA graph on both sides). The reference is a *speed
  reference only*, consulted only in extreme necessity (ADR 0005).

## 5. Telemetry (JSONL, from v1)

```jsonl
{"kind":"interval","t":123,"waiting":2,"prefilling":1,"running":3,"kv_used_pct":62,"kv_evictions":0}
{"kind":"request","id":"r-042","event":"admitted","lane":1}
{"kind":"request","id":"r-042","event":"ttft","ms":226}
{"kind":"request","id":"r-042","event":"done","n":512,"tok_s":41.2}
{"kind":"evict","tier":"ram","reason":"capacity"}
```

- One line per event, one line per interval. `class` field reserved (tagged lanes).
- `sibling_prefix_reused_tok` counter from v1.1.

## 6. Repo layout (Cargo workspace)

```
ignis/
├── crates/
│   ├── core/          # scheduler, paged KV, GDN state, request state machine
│   ├── artifact/      # .ninfer reader (reader/binder/materializer port)
│   ├── server/        # HTTP + OpenAI schemas + telemetry
│   └── bench/        # trace-replay harness + canary suite runner
├── kernel/            # C++/CUDA leaf (CMake + nvcc; NOTICE for provenance)
├── bench/traces/      # recorded load traces (JSONL)
├── docs/              # adr/, design/ (this file), agents/
├── webui/             # (v1.4+ / later phase) React+Vite+TS, same repo
├── CONTEXT.md
└── AGENTS.md
```

## 7. Open risks (tracked, not blocking)

- **Batched prefill (investigate):** we may be compute-bound, in which case
  batched prefill is useless; measure (north-star) before relying on it. It
  changes the kernel's accumulation order, so it is re-gated by the 99%
  performance gate (ADR 0007).
- **GEMM / kernel port (proven starting point):** the ported CUDA kernels are a
  *temporary* starting point; we re-implement them later (ADR 0005). The 99%
  performance gate (not byte-parity) measures residual drift vs the reference.
- **Expanded v1 (one release):** max feature throughput, max risk — the
  scheduler (admission + batched prefill + N=8 + host tier) is the
  highest-risk module and gets the most test coverage.
- **Frontend objects in the artifact:** tokenizer/chat-template extraction from the
  `frontend` object set must be verified during the artifact port (fact, not yet
  confirmed).
- **VRAM budget with eager graphs + N=8 + host tier:** ~1.9 GB graph allowance
  measured on the reference stack with MTP on; N=8 resident + host-tier overflow
  changes the VRAM math — verify at startup.
- **GDN state boundaries (now in v1):** resumable only at checkpoints; the KV-RAM
  host tier (now in v1) must respect the boundary (a snapshot mid-prefill is
  invalid for GDN layers).