# 03 — C ABI surface: decode step

Status: ready-for-agent

Define and implement the first C ABI surface:

- `extern "C"` entry points in `kernel/include/` for the decode step:
  NVFP4 GEMM + GQA attention (1:1 ports of the reference kernels).
- Flat ABI: explicit pointers + sizes + stream handle; no shared C++ state
  across the boundary (ADR 0001).
- Rust bindings in `ignis-core` mirroring the surface 1:1.
- Acceptance: single-step decode (prefill 1 token → decode 1 token) produces
  logits that match the reference engine on the canary suite within the 99%
  gate (ADR 0003) when driven by `ignis-bench`.