# ignis — v1 specification (design summary)

Grilled and settled 2026-09-02. This document is the shared understanding;
ADRs in `docs/adr/` carry the decisions that are hard to reverse.

## 1. Goals / non-goals

**Goal:** a Rust inference engine (`ignis`) specialized for Qwen 3.8-27B on a
single RTX 5090 (SM120a), serving OpenAI-compatible HTTP, reaching reference-parity
performance and architected for the later performance features (lazy CUDA graph,
hot reload, tagged lanes).

**Non-goals (v1):** other models, other GPU classes, continuous batching with
preemption, network exposure / auth, generic quantization formats, web UI.

## 2. Architecture

```
HTTP (OpenAI-compatible: /v1/models, /v1/chat/completions, /v1/responses)
  │
Rust core
  ├── HTTP server (localhost, no auth, configurable bind)
  ├── Scheduler: 1 global prefill lane + N decode lanes (default N=4)
  │               full admission state machine (protection / backfill class /
  │               temporal credit / frontier distance)
  ├── Paged KV cache (VRAM) + block tables
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
- **Default N (max concurrency)** 4.

## 3. Milestones

| Milestone | Scope |
|---|---|
| v1 | core: prefill + decode, paged KV, N lanes, full admission state machine, OpenAI API (minimal + responses), `.ninfer` loader, eager CUDA graph, telemetry, trace-replay bench harness, parity gate, hot-reload-ready architecture |
| v1.1 | KV-RAM tier (probation/protected eviction) + prefix reuse |
| v1.2 | DFlash2 speculative decoding (draft tokens 1..7) |
| v1.3 | MTP (adaptive verification width) |
| v1.4 | vision/multimodal |
| later | lazy CUDA graph capture, web UI (React+Vite+TS, same repo, CLI-disableable), hot reload, tagged lanes, other SM120a quantization formats |

## 4. Acceptance (v1)

- **Greedy + fixed seed.** Parity gate: **≥ 99% token agreement** against the
  reference baseline (see ADR 0003) on the canary suite **and** the trace-replay
  load; per-class ttft/tok-s within 10% of the reference.
- **Trace replay:** a `bench` crate re-sends a recorded "1 main agent + N
  subagents" load trace (JSONL, recorded against the reference stack with the same
  harness) and produces the parity + divergence report.
- The reference baseline is recorded with the same harness so comparisons are
  apples-to-apples (eager CUDA graph on both sides).

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

- **GEMM port fidelity:** byte-level accumulation order must survive the port or
  the 99% gate absorbs the drift — the divergence report will show which.
- **Frontend objects in the artifact:** tokenizer/chat-template extraction from the
  `frontend` object set must be verified during the artifact port (fact, not yet
  confirmed).
- **VRAM budget with eager graphs:** ~1.9 GB graph allowance measured on the
  reference stack with MTP on; v1 (no MTP) should be lower — verify at startup.
- **GDN state boundaries:** resumable only at checkpoints; v1.1 KV-RAM snapshots
  must respect the boundary (a snapshot mid-prefill is invalid for GDN layers).