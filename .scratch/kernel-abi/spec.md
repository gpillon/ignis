> **SUPERSEDED (2026-09-05)** by `.scratch/runtime/specs/01-device-resident-forward.md` (GitHub #36): the per-op host-pointer ABI and the host-resident forward this spec describes are being deleted. Kept for history only; do not implement against it.

# Kernel ABI extension — spec

Continues the C ABI surface (ADR 0001) opened by kernel-port 03 (decode step:
NVFP4 GEMM + GQA attention). This feature adds the remaining `extern "C"`
entry points the scheduler needs, all as flat C ABI (explicit pointers + sizes
+ stream handle, no shared C++ state across the boundary):

- **Prefill path:** GQA *prefill* attention + GDN linear-attention step,
  batched (concurrent) — the kernel side of the scheduler's batched prefill.
- **Pointwise / sampling:** RMSNorm / LayerNorm, embeddings, and greedy
  sampling as a C ABI surface.
- **CUDA graph capture:** eager graph capture at startup (v1 decision) —
  capture the prefill + decode kernels into a graph, verified at startup.

All kernels are 1:1 ports of the reference (no hand rewrites of proven
kernels; provenance via `kernel/NOTICE`); the ported kernels are a *temporary*
starting point per ADR 0005 (re-implement later, guided by the north-star).

## v1 scope (priority order)

1. GQA prefill + GDN step (batched) C ABI — `kernel-abi-01`
2. norms / embeddings / sampling (greedy) C ABI — `kernel-abi-02`
3. CUDA graph eager capture at startup — `kernel-abi-03`

## Acceptance

- Ported kernel unit tests from the reference suite pass.
- Single-step prefill→decode produces logits matching the reference engine on
  the canary suite within the **99% performance gate (ADR 0007)** when driven
  by `ignis-bench`.
- Eager CUDA graph capture verified at startup (graph replay ≡ eager path).

## References

- Design: `docs/design/ignis-v1.md` §2 (Kernel leaf), §3.
- ADRs: 0001 (C++ kernel leaf, C ABI), 0005 (performance-first), 0007 (gate).
- Decode C ABI: ticket kernel-port 03 (open, `#3`).