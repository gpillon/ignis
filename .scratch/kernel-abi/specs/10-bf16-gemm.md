> **SUPERSEDED (2026-09-05)** by `.scratch/runtime/specs/01-device-resident-forward.md` (GitHub #36): the per-op host-pointer ABI and the host-resident forward this spec describes are being deleted. Kept for history only; do not implement against it.

# 10 — bf16 logits GEMM (the W8-dequantized lm_head)

GitHub: #29 (the #25 follow-up — A2b, "bf16 logits GEMM")

The full-correct 27B model (A3) needs a **bf16 GEMM** for the logits: the
lm_head (`text/output_head`) is W8G32 in the artifact (A1 dequants it to bf16),
so the logits GEMM (`hidden · lm_headᵀ`) must consume a **bf16** weight — but
the current C-ABI surface only has the *NVFP4* GEMM (`ignis_nvfp4_gemm_*`,
kernel-abi 01/05), which cannot consume a dequantized bf16 weight. This ticket
adds the missing bf16 GEMM.

*A separate ticket from A2 (the conv + RoPE ports) and A3 (the assembly) so that
A3 stays a pure assembly; the bf16 GEMM is the third 27B-fidelity kernel.*

**Seam:** a new `extern "C"` entry `ignis_bf16_gemm` in
`kernel/include/ignis_kernel.h` + the matching 1:1 FFI binding in
`crates/core/src/ffi.rs` (ADR 0001).

```c
/* The bf16 GEMM (the logits path for the W8-dequantized lm_head, A1).
 * `out[tokens][m] = bias[m] + sum_k act[tokens][k] * W[m][k]`. `act` is bf16
 * [tokens][k]; `wt` is bf16 [m][k]; `bias` (nullable) is bf16 [m]; `out` is
 * bf16 [tokens][m]. A rowsplit FMA GEMM (no tensor cores / no cuBLASLt, ADR
 * 0001/0005). `tokens == 1` is the GEMV special case (the decode logits path);
 * `tokens > 1` serves the batched-prefill logits path (B1). stream: null =
 * stream 0. Returns 0 on success, -1 on error. */
int ignis_bf16_gemm(const void* act, const void* wt, const void* bias,
                    void* out, int64_t tokens, int64_t m, int64_t k,
                    void* stream);
```

**Scope:**
- The bf16 GEMM kernel (CUDA C++, `kernel/src/`), a rowsplit FMA GEMM (the
  proven "for now" baseline, ADR 0005; the tensor-core MMA is the later
  performance material, ADR 0005/0007).
- The C-ABI entry + the 1:1 FFI binding (`ignis_kernel.h` +
  `crates/core/src/ffi.rs`, ADR 0001).

**Non-goals (v1):** the tensor-core MMA for the bf16 GEMM (the FMA baseline is
the v1 starting point; the MMA is the later performance material, ADR
0005/0007); a *fused* logits+sample kernel (the sample stays the separate
`ignis_greedy_sample`, kernel-abi 02).

**Acceptance:**
- **`ignis_bf16_gemm`:** a GPU-gated launch test (the `kernel_abi0N_gpu`
  convention, ADR 0006) runs the GEMM on synthetic multi-token input; the
  output matches a CPU reference GEMM (bf16 within the tolerance); a
  `tokens == 1` case (the GEMV special case, the decode logits path).
- **The C-ABI entry + FFI binding are 1:1** (`ignis_kernel.h` +
  `crates/core/src/ffi.rs`, ADR 0001).
- **`cargo test --workspace` green** (AGENTS.md: every code change ships with a
  test, and the task is not complete until it passes workspace-wide).

**Blocked by:** (none — a kernel port, independent of A1; it *consumes* A1's
W8→bf16-dequantized lm_head, but the kernel itself is format-agnostic and
testable on synthetic bf16 inputs).

**References:** ADR 0001 (the flat C ABI + the GEMM convention), ADR 0005
(rowsplit FMA "for now"; the MMA is the later performance material), ADR 0006
(exclusive GPU — the launch test self-skips on a busy GPU). A1
(`artifact/specs/04`, the W8→bf16 dequant the GEMM consumes). A3
(`kernel-abi/specs/07`, the assembly that uses the GEMM for the logits). B1
(`kernel-abi/specs/08`, the batched prefill's multi-token logits path).