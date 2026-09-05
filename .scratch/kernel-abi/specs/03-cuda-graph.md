> **SUPERSEDED (2026-09-05)** by `.scratch/runtime/specs/01-device-resident-forward.md` (GitHub #36): the per-op host-pointer ABI and the host-resident forward this spec describes are being deleted. Kept for history only; do not implement against it.

# 03 — CUDA graph eager capture at startup

GitHub: #10

Eager **CUDA graph capture** at startup (v1 decision; lazy capture = a later
optimization, `docs/design/ignis-v1.md` §3). Capture the prefill + decode
kernels into a graph and replay on each step; verify the capture at startup.

## Acceptance

- Graph capture is verified at startup (eager).
- Graph replay ≡ the eager path: logits match within the **99% performance
  gate (ADR 0007)** on the canary suite.
