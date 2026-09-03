# Pending / revisit ledger (ignis)

Cross-cutting items that are intentionally deferred or blocked on an external
dependency. Per-ticket details live in `.scratch/<feature>/specs/`.

## Open

- **bench-02: recorded reference baseline + 99% gate (ADR 0007, GitHub #20).**
  The `ignis-bench` harness is code-complete — `HttpEndpoint` transport,
  per-class metrics, canary self-consistency, 99% gate check (bench-01,
  968a2c1), plus the shipped v1 gate artifact (this bench-02 work:
  `ignis-bench gate` composes the performance report + the divergence
  report into a single shippable JSON artifact; `canary --out` ships the
  divergence report; `CanaryResult` is serializable with the sanity
  reason; the CLI flag parsing is fixed). What remains is the **recorded**
  side: a trace (JSONL) recorded against the reference stack + a reference
  run recorded with the *same harness* — the synthetic
  `main_plus_10.jsonl` fixture is not a reference. Procedure + file
  layout: `bench/traces/README.md`. Then `ignis-bench gate` runs the 99%
  gate. Owner: bench actor. Blocker: GPU + a recorded reference recording.

## Blocked (external)

- **GPU availability (ADR 0006).** All GPU-gated items above require the
  RTX 5090 to be free. Last freed 2026-09-03 (GPU verified free, artifact-01
  GPU test run). Re-check before scheduling GPU work.

## Resolved (pruned weekly)

- **kernel-abi-03: CUDA-graph eager capture at startup (GitHub #10, ADR 0006/0007)** —
  resolved 2026-09-03: `kernel/src/graph_capture.cu` implements the four
  `ignis_graph_*` primitives + `ignis_graph_startup_check` (captures a
  representative GQA-prefill + GDN-step + GQA-decode kernel sequence into a
  CUDA graph, replays it, and verifies replay ≡ eager bit-exactly). The
  kernels are forward-declared in the .cu (defined in the sibling surface .cu
  files) to avoid LNK4006 duplicate definitions. `kernel_abi03_gpu`
  (`graph_primitives_roundtrip_gpu` + `graph_startup_check_gpu` + the CPU
  null-handle pin) passed on 2026-09-03; the startup check confirmed on the
  GPU that replay ≡ eager. The 99% performance gate (ADR 0007) remains
  pending under bench-02 (GitHub #20, a recorded reference baseline is
  required).

- **artifact-01: `CudaDevice` real-artifact VRAM materialization (GitHub #4, ADR 0006)** —
  resolved 2026-09-03: `real_nvfp4full_cuda_device` passed on a free RTX 5090
  (9.42 s, 1,319 tensors H2D, ~19 GB VRAM). Build fix: `CMAKE_MSVC_RUNTIME_LIBRARY`
  set to `MultiThreaded` in `kernel/build.ps1` to force `/MT` static CRT (Rust
  MSVC target requires it; CMake default `/MD` caused LNK2038).

- **server-03: checksum wiring into the artifact loader (GitHub #21)** —
  resolved 2026-09-02 (`loader` module in `crates/server`, verified load
  path, descriptive refusal on a non-clean report or missing sidecar).