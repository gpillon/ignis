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

- **#26: compute-adapter crash fix + VRAM materialization + device-GEMM
  surface (GitHub #26, ADR 0002/0006).**
  Resolved 2026-09-03: `CudaCompute::from_artifact` no longer falls back to
  the synthetic topology (vocab 256) — it uses the real `qwen38_27b()`
  topology, so real-tokenizer ids (up to 248077) never index out of bounds in
  `ignis_embedding` (the `illegal memory access`), and the 19 GB of weights
  materialize to VRAM via `CudaDevice` (`vram_resident()`). Host-side
  `Weights` are a zero-cost `placeholder` (the real weights live in the VRAM
  arena, ADR 0002); the device-resident GEMM surface
  (`ignis_nvfp4_gemm_{decode,prefill}_device`) is compiled into the kernel
  leaf `.lib` (a prerequisite for the broader compute-adapter, kernel-abi
  04/05).
  **The #26 "hang" was a CPU OOM trap, not a GPU deadlock:** the prior build
  ran `Weights::synthetic` at the real topology (~1.6 TiB of generated host
  vectors in a debug build), so the CPU spun for minutes/hours *after* the
  19 GB H2D while the GPU sat at 0 % — which read like a stuck
  `cudaStreamSynchronize`. `Weights::placeholder` (zero-cost) fixes it; the
  E2E (`real_model_e2e`) now passes in ~9 s.
  The numerically-correct real completion (the actual forward pass, the
  CUDA-graph fast path, and the server serving real completions) is deferred
  to #25 — no new ticket needed.

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