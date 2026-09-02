# 03 — C ABI surface: decode step

GitHub: #3

Define and implement the first C ABI surface (the decode step):

- `extern "C"` entry points in `kernel/include/` for the decode step:
  NVFP4 GEMM + GQA attention (1:1 ports of the reference kernels).
- Flat ABI: explicit pointers + sizes + stream handle; no shared C++ state
  across the boundary (ADR 0001).
- Rust bindings in `ignis-core` mirroring the surface 1:1.

## Acceptance

- Single-step decode (prefill 1 token → decode 1 token) produces logits that
  match the reference engine on the canary suite within the 99% gate (ADR
  0003) when driven by `ignis-bench`.
- GPU-gated verification: `cargo test -p ignis-core --test decode_gpu --
  --ignored` (synthetic launch tests; they fit in a few MB of VRAM, so they
  can run even with the model loaded — the ADR 0006 nuance). The 99% canary
  gate itself is the bench-02 acceptance (GitHub #20, ADR 0007).

## References

- Design: `docs/design/ignis-v1.md` §2 (Kernel leaf), §3.
- ADRs: 0001 (flat C ABI), 0003 (divergence report), 0006 (exclusive GPU).
- Surface: `kernel/include/ignis_kernel.h`; Rust FFI: `crates/core/src/ffi.rs`.