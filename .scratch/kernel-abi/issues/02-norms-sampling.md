# 02 — norms / embeddings / sampling C ABI (greedy)

Status: needs-triage
GitHub: #6
Blocked by: #3 (kernel-port-03)

`extern "C"` entry points in `kernel/include/` for the pointwise / output
path: **RMSNorm / LayerNorm**, **embeddings**, and **greedy sampling**.
Flat ABI (explicit pointers + sizes + stream handle, no shared C++ state,
ADR 0001); 1:1 ports of the reference kernels.

## Acceptance

- Ported kernel unit tests pass.
- Greedy sampling output on the canary suite matches the reference engine
  within the **99% performance gate (ADR 0007)** when driven by
  `ignis-bench`.
