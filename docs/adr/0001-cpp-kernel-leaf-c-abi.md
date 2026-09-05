# ADR 0001 — C++ kernel leaf behind a C ABI

## Status

Accepted (2026-09-02, grilling session). **Revised (2026-09-05) by ADR 0009**
— the two-language split and the flat C ABI stand; the ABI's *granularity*
changes from per-operator with host pointers to per-**step** with opaque
handles over leaf-owned device state. Read the Decision below with that
revision applied.

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

> **Revised by ADR 0009 (2026-09-05):** Rust owns everything above the
> **step**, not above compute — the forward pass itself (the per-layer op
> sequence, the device arena, streams and the per-sequence KV / GDN state)
> lives in the leaf. The ABI stays flat C, but it is step-level and
> device-resident: opaque handles, integer codes, and no host activation
> pointers crossing the boundary. The per-operator, host-pointer reading of
> this decision produced a forward pass that could not work at any speed;
> see `.scratch/REVIEW-2026-09-05.md` §2-§3.

## Consequences

- Maximum kernel reuse: parity with the reference stack is achievable because the
  compute is identical (same MMA instructions, same accumulation order).
- Two toolchains: CMake (nvcc) builds the kernel leaf, Cargo links it via build.rs.
- The FFI boundary is the only place cross-language state crosses — keep it flat,
  checked, and logged.
- Reused kernel files carry upstream provenance: the repo declares it in a
  `NOTICE`/license section. **ADR 0010 (2026-09-05)** makes this concrete:
  reused ops are vendored verbatim under a manifest of pinned commit, content
  hashes and recorded patches.