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

- **kernel-abi: CUDA implementations + 99% performance gate (ADR 0006/0007, GitHub #5/#6/#10).**
  The C ABI surface (kernel-abi 01-03, adb6ac9) is committed; the CUDA
  kernels and the 99% gate are GPU-gated and deferred until the GPU is
  free and a reference baseline exists (see bench-02 above). Owner: kernel
  actor. Blocker: GPU.

## Blocked (external)

- **GPU availability (ADR 0006).** All GPU-gated items above require the
  RTX 5090 to be free. Last freed 2026-09-02 (stopped `ninfer-serve`).
  Re-check before scheduling GPU work.

## Resolved (pruned weekly)

- **server-03: checksum wiring into the artifact loader (GitHub #21)** —
  resolved 2026-09-02 (`loader` module in `crates/server`, verified load
  path, descriptive refusal on a non-clean report or missing sidecar).