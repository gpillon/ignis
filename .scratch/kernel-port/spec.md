# Kernel leaf port — spec

Bring the C++/CUDA compute leaf of the reference stack into `ignis/kernel/`
behind the flat C ABI of ADR 0001, so `ignis-core` can drive the GPU.

## Approach

- Copy `.cu`/`.cuh` sources from the reference fork into `kernel/` with
  provenance tracking (`kernel/NOTICE`); no hand rewrites of proven kernels.
- The C ABI surface is the set of `extern "C"` entry points declared in
  `kernel/include/`; Rust bindings in `ignis-core` mirror them 1:1.
- Build: CMake + nvcc (SM120a target), Rust links via `build.rs` (ADR 0001).
- Byte-parity feasibility depends on a 1:1 kernel port + the same CUDA
  toolkit (13.x); the divergence report (ADR 0003) measures residual drift.

## v1 kernel scope (priority order)

1. NVFP4 GEMM (rowsplit/grouped MMA) — prefill + decode
2. GQA attention (prefill + decode, MRoPE 3-axis)
3. GDN linear-attention step (state resumable at frontier boundaries only)
4. Norms/RMS, embeddings, sampling (greedy)
5. CUDA graph capture (eager at startup — v1 decision)
6. DFlash2 / MTP speculative paths — v1.2 / v1.3, not v1

## Acceptance

- Ported kernel unit tests from the reference suite pass.
- Canary parity: greedy output on the canary suite meets the 99% gate
  (ADR 0003), divergence report shipped.
- Eager CUDA graph capture verified at startup.