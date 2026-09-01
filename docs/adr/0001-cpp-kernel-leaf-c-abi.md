# ADR 0001 — C++ kernel leaf behind a C ABI

## Status

Accepted (2026-09-02, grilling session).

## Context

ignis is a Rust engine, but peak performance on SM120a lives in the kernels:
NVFP4 GEMM (custom rowsplit/grouped MMA, no cuBLASLt in the reference stack),
GQA attention, GDN linear-attention step, DFlash2/MTP speculative paths.
The proven implementations exist in the ninfer fork as CUDA C++.

Options:
- (a) Rust core + C++ kernel library behind a C ABI (port/reuse ninfer kernels)
- (b) pure Rust: cudarc for memory/streams, kernels in Rust compiled to PTX
- (c) hybrid with cuBLAS/cuBLASLt calls

## Decision

(a). Rust owns everything above compute: scheduler, paged KV, artifact loading,
HTTP serving, telemetry. The kernel leaf is a C++/CUDA static library behind a
flat C ABI (explicit pointers + sizes, no shared state).

## Consequences

- Maximum kernel reuse: parity with the reference stack is achievable because the
  compute is identical (same MMA instructions, same accumulation order).
- Two toolchains: CMake (nvcc) builds the kernel leaf, Cargo links it via build.rs.
- The FFI boundary is the only place cross-language state crosses — keep it flat,
  checked, and logged.
- Reused kernel files carry upstream provenance: the repo declares it in a
  `NOTICE`/license section.