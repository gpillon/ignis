# 03 — CUDA graph eager capture at startup

Status: needs-triage
GitHub: #10
Blocked by: #5 (kernel-abi-01), #6 (kernel-abi-02)

Eager **CUDA graph capture** at startup (v1 decision; lazy capture = a later
optimization, `docs/design/ignis-v1.md` §3). Capture the prefill + decode
kernels into a graph and replay on each step; verify the capture at startup.

## Acceptance

- Graph capture is verified at startup (eager).
- Graph replay ≡ the eager path: logits match within the **99% performance
  gate (ADR 0007)** on the canary suite.
