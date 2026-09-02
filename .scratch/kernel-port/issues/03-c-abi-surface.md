# 03 — C ABI surface: decode step

Status: done (implementation complete; canonical .lib rebuilt so the gated
tests link; only the GPU launch run + 99% acceptance gate remain, pending a
free 5090 — see Pending)

Define and implement the first C ABI surface:

- `extern "C"` entry points in `kernel/include/` for the decode step:
  NVFP4 GEMM + GQA attention (1:1 ports of the reference kernels).
- Flat ABI: explicit pointers + sizes + stream handle; no shared C++ state
  across the boundary (ADR 0001).
- Rust bindings in `ignis-core` mirroring the surface 1:1.
- Acceptance: single-step decode (prefill 1 token → decode 1 token) produces
  logits that match the reference engine on the canary suite within the 99%
  gate (ADR 0003) when driven by `ignis-bench`.

## Status

- [x] NVFP4 decode GEMM (GEMV) kernel ported — `kernel/src/nvfp4_gemm_decode.cuh`
- [x] GQA attention decode kernel ported — `kernel/src/gqa_attention_decode.cuh`
- [x] Flat C ABI decode surface appended to `kernel/include/ignis_kernel.h`
- [x] C ABI wrapper TU — `kernel/src/decode_surface.cu`
- [x] Rust FFI bindings appended 1:1 — `crates/core/src/ffi.rs`
- [x] CPU-verifiable geometry/quant tests — `crates/core/tests/decode_geometry.rs` (PASS, no GPU)
- [x] Provenance appended to `kernel/NOTICE`
- [x] Canonical `kernel/build/ignis_kernel.lib` rebuilt — ticket-03 symbols
      present; the gated test binary now links (verified via `cargo test
      --no-run`)
- [~] Gated GPU launch tests — `crates/core/tests/decode_gpu.rs` (written, `#[ignore]`-gated; run once the 5090 is free)

## Pending

1. **Run the gated GPU launch tests** once the RTX 5090 is free (ADR 0006 —
   the GPU is currently occupied by ninfer-serve):
   `cargo test -p ignis-core --test decode_gpu -- --ignored`. The tests
   self-skip on a non-zero return (busy GPU), so they are safe to leave in the
   suite and will not turn the build red while the GPU is occupied. The
   `.lib` they link against is already rebuilt (Status above), so this is
   purely a "wait for the GPU" item.
2. **Acceptance gate (ADR 0003, 99% canary match)** — single-step decode
   (prefill 1 token → decode 1 token) logits matching the reference engine
   when driven by `ignis-bench`. Requires (a) a free GPU, and (b) the full
   prefill+decode pipeline. Not yet reachable; tracked here so it is not lost.

### Resolved during this pass

- **Canonical `.lib` rebuild** — done. Rebuilt via `kernel/build.ps1`; the
  rebuilt `kernel/build/ignis_kernel.lib` now exports `ignis_nvfp4_gemm_decode`
  and `ignis_gqa_attention_decode`, so the gated test binary links.
- **Full-workspace build precondition** — resolved. The parallel workstream's
  `ignis-artifact` crate now compiles, so all `ignis-core` test binaries
  build and link (verified with `cargo test -p ignis-core --no-run`). The
  pure-Rust `decode_geometry` test runs green (5/5, no GPU).