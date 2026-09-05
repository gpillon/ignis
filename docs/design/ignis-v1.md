# ignis — v1 specification (design summary)

Grilled and settled 2026-09-02. This document is the shared understanding;
ADRs in `docs/adr/` carry the decisions that are hard to reverse.

> **Status note (2026-09-05).** The goals, the architecture *shape* and the
> acceptance philosophy below still hold. What changed after the project
> review (`.scratch/REVIEW-2026-09-05.md`): the engine has never produced a
> real completion, the ABI moved from the operator level to the **step** level
> with the forward pass in the leaf (ADR 0009), ops are **vendored verbatim**
> under a manifest (ADR 0010), and the "one expanded v1 release" milestone
> plan is replaced by the **phase / gate plan** in §3. Read §2 as the target
> architecture, not as shipped code.

## 1. Goals / non-goals

**Goal:** a Rust inference engine (`ignis`) specialized for Qwen 3.8-27B on a
single RTX 5090 (SM120a), serving OpenAI-compatible HTTP, reaching ≥ 99% of the
reference's performance and architected for the later performance features
(per-width CUDA graphs, hot reload, tagged lanes).

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
  ├── Artifact loader (.ninfer reader: reader + binder + materializer + device views)
  ├── Telemetry (JSONL events + interval lines)
  └── Runtime wrapper (safe Rust over the step ABI; handles + Drop)
        │
Step-level C ABI (ADR 0009): device-resident, opaque handles, flat C
  model load · sequence alloc/release/snapshot/restore · prefill(span, pos)
  · decode round(batch) · sampling params · stats
        │
Kernel leaf (C++/CUDA static library, CMake + nvcc, SM120a)
  ├── program (ours): device arena + streams + the 64-layer op sequence
  │                   + sequence state (KV pages, fp32 GDN slots, conv taps)
  └── vendored ops (verbatim, ADR 0010)
      ├── NVFP4 / BF16 / W8G32 linear (GEMV, small-T, W4A4 + TMA)
      ├── GQA attention (prefill + decode, paged KV) + RoPE + q/k norm
      ├── GDN family (causal conv1d + SiLU, gating, recurrence, chunked)
      ├── sampling, embeddings, norms
      └── decode CUDA graphs, captured per batch width
```

- **Rust owns everything above the step; the leaf owns the step** (ADR 0009).
  No host activation pointer crosses the ABI; activations, KV and GDN state
  live in VRAM for the lifetime of a sequence.
- **Model lifecycle is decoupled from server lifecycle** (hot-reload-ready by
  construction): KV pool, CUDA graphs, and scheduler state are regenerable per
  model; in-flight requests finish or restart per a defined policy.
- **Default context envelope** 262k (KV pool auto-sized from free VRAM; ~12-13 GB
  pool budget after weights + graphs + runtime).
- **Default N (max concurrency) 8** (resident lanes, with host-tier overflow;
  sized for a ~10-subagent concurrent coding workload).

## 3. Milestones — the phase / gate plan

Replaces the original "one expanded v1 release" table (2026-09-05, after the
project review). Every phase ends in a **GPU-measured gate**; no phase is done
on CPU tests. Tickets and blocking live on GitHub; the decomposition lives in
`.scratch/ROADMAP.md`.

| Phase | Gate | Scope |
|---|---|---|
| **0 — reset the ground truth** (done) | — | The review; the step ABI and vendoring decisions (ADR 0009 / 0010); this docs reset; the GPU test profile; deleting the superseded forward, toy graphs and scalar kernels. |
| **1 — correct device-resident forward** | **G1**: coherent greedy completions on the canary suite; per-layer output within bf16 tolerance of the f64 layer reference; ≥ 95% first-32-token agreement with the canary oracle; EOS honored; reproducible across loads. | The leaf program (arena, streams, layer sequence, sequence state); the vendored op families (linear, norms, endpoints, GQA attention + RoPE, GDN); the step ABI; the Rust runtime crate; server end-to-end on one request. Batch 1, bf16 KV, per-token prefill. |
| **2 — real prefill** | **G2**: TTFT at 8K / 32K ≤ 1.5× the reference's MTP0. | Chunked prefill (1024) with W4A4 activation quant + the TMA GEMM, tensor-core prefill attention with KV append, the GDN chunked kernels. |
| **3 — serving loop** | **G3**: C=1 decode ≥ 99% of the reference (75–76 tok/s); C=4 aggregate ≥ 99%. | One batched decode round for all decode-ready lanes; a decode CUDA graph per batch width with eager fallback; sampling (temperature / top-p / top-k / penalties, seed); the KV pool bound to real device pages; request-log JSONL. |
| **4 — the reference feature floor** | **G4**: the **99% performance gate** (ADR 0007) — the bench trace-replay gate on a recorded "1 main + N subagents" load, hq-e8-2b on both sides. | hq-e8-2b KV + exact-key side store; device prefix reuse (page refcount, shared boundary); the KV-RAM host tier's real snapshot/restore; tagged lanes; preserve-thinking and tool-call stream hardening; warmup/readiness split. |
| **5 — speculative decoding** | **G5**: ≥ 99% of the reference's MTP7-adaptive / DFlash2-7 committed tok/s at 24K / 98K / 196K. | MTP (round, pack, adaptive width, ReplaySSM records) first; DFlash2 (drafter module + draft kernels + RAM-tier carry) second. |
| **6 — beyond the reference** (north star) | per-item | Concurrent prefill across requests / prefill-decode overlap, fusion + PDL everywhere, lazy graph capture, hot reload, our own NVFP4 artifact recipe, lane QoS, **our own kernels per measured family** (ADR 0010). Vision and a web UI sit here too. |

The KV-RAM host tier, prefix reuse and the full admission state machine were
pulled into "v1"; they exist as a CPU-tested control plane and are bound to
real device state at G3 / G4. Kernel re-implementation stays where ADR 0005
put it: after the engine works and dogfoods, per family, gated by measurement.

## 4. Acceptance (v1)

- **The correctness floor comes first (G1), the performance gate after (G4).**
  The floor is measured, not eyeballed: per-layer output within bf16 tolerance
  of an f64 layer reference, and ≥ 95% first-32-token agreement with the canary
  oracle. That is a *self-check that the model is computed*, not a parity
  target.
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
│   ├── core/          # scheduler, paged KV accounting, request state machine
│   ├── artifact/      # .ninfer reader (reader/binder/materializer/device views)
│   ├── runtime/       # (G1) safe Rust wrapper over the step ABI
│   ├── server/        # HTTP + OpenAI schemas + telemetry
│   └── bench/        # trace-replay harness + canary suite runner
├── kernel/            # C++/CUDA leaf: program + vendored ops (CMake + nvcc; NOTICE + manifest)
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
- **Vendored ops (proven starting point):** the ops are vendored verbatim from
  the reference under a manifest (ADR 0010) as a *temporary* starting point; we
  write our own later, per measured family (ADR 0005). Hand-written stand-ins
  carrying a port claim are what the 2026-09-05 review found and deleted.
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