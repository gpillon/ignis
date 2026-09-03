# 05 — multi-token NVFP4 GEMM for the prefill (FFN) projections

GitHub: #22

The kernel-leaf C-ABI surface (kernel-abi 01–03 + the ticket-03 decode
surface) has **no multi-token GEMM**. The only GEMM is
`ignis_nvfp4_gemm_decode` — the single-token GEMV path
(`out[m] = bias[m] + sum_k x[k]*W[m,k]`, `act` is a single activation
vector of length k, `kernel/include/ignis_kernel.h`). The prefill path
needs a real GEMM: every linear projection in a layer (QKV, output-proj,
FFN up-proj, FFN down-proj) is `[tokens × hidden] × W` where `tokens` is
the prefill length (long for the main-agent coding prompts). Without a
multi-token GEMM, the compute-adapter's (kernel-abi 04) prefill path is
forced to **loop the single-token GEMV per token** — and each C-ABI call
does internal H2D/D2H copies (ADR 0001: the leaf takes host pointers and
does the device work). So prefill throughput collapses on long prompts,
and the 99% gate's per-class ttft/tok-s check (ADR 0007) fails on the
main-agent class.

This ticket adds the missing kernel: a **multi-token NVFP4 GEMM** for the
prefill (FFN) projections, exposed as a new C-ABI function. Per ADR 0005
(*"port the proven CUDA 'for now', re-implement later"*) and ADR 0001
(*"NVFP4 GEMM, custom rowsplit/grouped MMA, no cuBLASLt"*), we port the
reference's NVFP4 GEMM (rowsplit/grouped MMA) as a self-contained
kernel-leaf function (north-star: the reference is inspiration-only —
stay on the north-star, ADR 0005).

**Seam:** a new C-ABI entry `ignis_nvfp4_gemm_prefill`
(`kernel/include/ignis_kernel.h`) + the matching FFI binding
(`crates/core/src/ffi.rs`), 1:1 (ADR 0001). Same flat-C-ABI conventions
as the existing surface: explicit host pointers + sizes, a stream handle
(null = stream 0), an int return code (0 = ok, -1 = error); the leaf
does the H2D/D2H internally.

**Scope:**
- A new multi-token NVFP4 GEMM kernel (CUDA C++, `kernel/src/`):
  `out[tokens][m] = bias[m] + sum_k act[tokens][k] * W[m][k]`, where
  `act` is bf16 `[tokens][k]`, `W` is NVFP4-quantized (E2M1 codes +
  per-16 E4M3 scales), `out` is bf16 `[tokens][m]`. Handles `tokens >> 1`
  (a single prefill's worth of tokens, or a batched-prefill group of
  requests). rowsplit/grouped-MMA tiling (the proven reference approach,
  ADR 0001); **no cuBLASLt** (ADR 0001: "no cuBLASLt in the reference
  stack").
- The new C-ABI function:
  `int ignis_nvfp4_gemm_prefill(const void* act, const void* wt_codes,
   const void* wt_scales, const void* bias, void* out, int64_t tokens,
   int64_t m, int64_t k, void* stream)` — 0 on success, -1 on error.
- The matching FFI binding in `crates/core/src/ffi.rs` (1:1 with the
  header, ADR 0001).
- A GPU-gated launch test (per the `kernel_abi0N_gpu` convention, ADR
  0006): a `tokens > 1` case checked against a CPU reference GEMM (the
  existing convention: a synthetic-input geometry pin + a real CUDA-launch
  test that self-skips on a busy GPU), plus a `tokens == 1` case that
  matches `ignis_nvfp4_gemm_decode` (the GEMV is the 1-token special
  case — a regression pin).
- Wire it into the kernel leaf's build (CMake + the existing surface .cu
  files).

## Acceptance

- **The new C-ABI function is correct**: a GPU-gated test runs
  `ignis_nvfp4_gemm_prefill` with `tokens > 1` against a CPU reference
  GEMM; the bf16 outputs match within the NVFP4 quantization tolerance
  (per the existing GEMM test convention). A `tokens == 1` case matches
  `ignis_nvfp4_gemm_decode` (the single-token GEMV is the 1-token special
  case).
- **The C-ABI entry + FFI binding are 1:1**: `kernel/include/
  ignis_kernel.h` and `crates/core/src/ffi.rs` both declare
  `ignis_nvfp4_gemm_prefill` with matching signatures (ADR 0001).
- **`cargo test --workspace` is green** (AGENTS.md: every code change
  ships with a test, and the task is not complete until it passes
  workspace-wide).
- **The compute-adapter (kernel-abi 04) prefill path uses it**: the
  adapter's `prefill_step` calls `ignis_nvfp4_gemm_prefill` for the FFN
  projections (instead of looping the single-token GEMV) — this is what
  makes batched prefill actually viable (design §7, "Batched prefill
  (investigate): we may be compute-bound") and what lets the 99% gate's
  per-class ttft/tok-s (ADR 0007) pass on long prompts.

This is a prerequisite for kernel-abi 04 (compute-adapter), which is a
prerequisite for bench-03 (gate-run / the 99% gate that closes #20).
References: ADR 0001 (C ABI, NVFP4 GEMM rowsplit/grouped MMA, no
cuBLASLt), ADR 0005 (port proven CUDA "for now"), ADR 0007 (the 99%
performance gate).