---
status: accepted
---

# C++ kernel leaf behind a C ABI

ignis is a Rust engine, but peak performance on the card lives in the kernels (the NVFP4 GEMM with its custom
rowsplit and grouped MMA, GQA attention, the GDN linear-attention step, the speculative paths), and the proven
implementations of them existed as CUDA C++ in the ninfer fork. So ignis is two languages: a Rust core, and a kernel
leaf that is a C++/CUDA static library behind a flat C ABI. Where the boundary sits is ADR 0009: Rust owns everything
above the step and the leaf owns the forward pass, behind opaque handles over device state the leaf owns, with no
host activation pointer crossing.

## Considered Options

- **Pure Rust, `cudarc` for memory and streams and the kernels written in Rust and compiled to PTX**: rejected, it
  reuses none of the proven kernels.
- **A hybrid that calls cuBLAS or cuBLASLt**: rejected, the reference stack's NVFP4 GEMM is custom MMA and uses no
  cuBLASLt.

## Consequences

- Kernel reuse is maximal: matching the reference stack is achievable because the compute is identical, the same MMA
  instructions in the same accumulation order.
- Two toolchains: CMake with nvcc builds the kernel leaf, and Cargo links it through `build.rs`.
- The FFI boundary is the only place where state crosses between the languages: it is kept flat, checked and logged.
- Reused kernel files carry their upstream provenance, which the repo declares in `NOTICE`; how a reused op is
  vendored and tracked is ADR 0010.
