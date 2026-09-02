# 01 — GQA prefill + GDN step C ABI (batched)

Status: needs-triage
Blocked by: kernel-port-03

`extern "C"` entry points in `kernel/include/` for the **GQA prefill**
attention + the **GDN linear-attention step**, batched (concurrent) — the
kernel side of the scheduler's batched prefill. 1:1 ports of the reference
kernels (provenance via `kernel/NOTICE`).

- Flat ABI: explicit pointers + sizes + stream handle; no shared C++ state
  across the boundary (ADR 0001).
- Rust bindings in `ignis-core` mirror the surface 1:1.

## Acceptance

- Ported kernel unit tests from the reference suite pass.
- Single-step prefill→decode produces logits matching the reference engine
  on the canary suite within the **99% performance gate (ADR 0007)** when
  driven by `ignis-bench`.