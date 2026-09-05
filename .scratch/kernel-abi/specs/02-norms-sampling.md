> **SUPERSEDED (2026-09-05)** by `.scratch/runtime/specs/01-device-resident-forward.md` (GitHub #36): the per-op host-pointer ABI and the host-resident forward this spec describes are being deleted. Kept for history only; do not implement against it.

# 02 — norms / embeddings / sampling C ABI (greedy)

GitHub: #6

`extern "C"` entry points in `kernel/include/` for the pointwise / output
path: **RMSNorm / LayerNorm**, **embeddings**, and **greedy sampling**.
Flat ABI (explicit pointers + sizes + stream handle, no shared C++ state,
ADR 0001); 1:1 ports of the reference kernels.

## Acceptance

- Ported kernel unit tests pass.
- Greedy sampling output on the canary suite matches the reference engine
  within the **99% performance gate (ADR 0007)** when driven by
  `ignis-bench`.
