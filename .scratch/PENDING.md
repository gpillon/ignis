# Pending / revisit ledger (ignis)

Cross-cutting items that are intentionally deferred or blocked on an external
dependency. Per-ticket details live in `.scratch/<feature>/specs/`.

## Open

- **bench-02: recorded reference baseline + 99% gate (ADR 0007, GitHub #20).**
  The `ignis-bench` harness (bench-01, 968a2c1) is in — `HttpEndpoint`
  transport, per-class metrics, canary self-consistency, 99% gate check —
  but it needs a **recorded** trace (JSONL) + a reference run to compare
  against. The synthetic `main_plus_10.jsonl` fixture is not a reference.
  Owner: bench actor. Blocker: GPU + a recorded reference recording.

- **kernel-abi: CUDA implementations + 99% performance gate (ADR 0006/0007, GitHub #6/#10).**
  The C ABI surface (kernel-abi 01-03, adb6ac9) is committed. Ticket-05's
  (GitHub #5) GQA-prefill + GDN-step kernels are implemented and GPU-verified
  — `kernel_abi01_gpu` (`gqa_attention_prefill_gpu` + `gdn_step_gpu`) passed
  on a free GPU on 2026-09-03 — so #5 is closed. The remaining CUDA work
  (GitHub #6/#10) and the 99% gate are GPU-gated and deferred until the GPU
  is free and a reference baseline exists (see bench-02 above). Owner: kernel
  actor. Blocker: GPU.

## Blocked (external)

- **GPU availability (ADR 0006).** All GPU-gated items above require the
  RTX 5090 to be free. Last freed 2026-09-03 (GPU verified free, artifact-01
  GPU test run). Re-check before scheduling GPU work.

## Resolved (pruned weekly)

- **artifact-01: `CudaDevice` real-artifact VRAM materialization (GitHub #4, ADR 0006)** —
  resolved 2026-09-03: `real_nvfp4full_cuda_device` passed on a free RTX 5090
  (9.42 s, 1,319 tensors H2D, ~19 GB VRAM). Build fix: `CMAKE_MSVC_RUNTIME_LIBRARY`
  set to `MultiThreaded` in `kernel/build.ps1` to force `/MT` static CRT (Rust
  MSVC target requires it; CMake default `/MD` caused LNK2038).

- **server-03: checksum wiring into the artifact loader (GitHub #21)** —
  resolved 2026-09-02 (`loader` module in `crates/server`, verified load
  path, descriptive refusal on a non-clean report or missing sidecar).