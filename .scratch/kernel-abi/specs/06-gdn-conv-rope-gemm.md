> **SUPERSEDED (2026-09-05)** by `.scratch/runtime/specs/01-device-resident-forward.md` (GitHub #36): the per-op host-pointer ABI and the host-resident forward this spec describes are being deleted. Kept for history only; do not implement against it.

# 06 — GDN causal-conv + RoPE kernel ports

GitHub: #28 (the #25 follow-up — A2, "GDN causal-conv + RoPE kernel ports")

The full-correct 27B model (A3) needs two kernel ops the current C-ABI surface
(kernel-abi 01–05) does **not** have:

1. **The GDN causal conv** — the Gated-DeltaNet input path is GEMM-then-conv:
   the input projection (the NVFP4 GEMM) feeds a **4-tap depthwise causal
   convolution + SiLU** over the projected q/k/v (one thread per channel; the
   3-tap rolling state `s0,s1,s2` + the current tap `w3·p`; SiLU output; the
   `z` rows bypass the conv). The `gdn/convolution` tensor is BF16
   `{4, channels}`. Port source: the reference's `gdn_projected_conv`
   (`src/ops/gdn_input_proj/gdn_projected_conv.cu`).
2. **The GQA RoPE** — the split-half NeoX rotary position embedding applied to
   **Q and K** (`rotary_dim` = 64 of `head_dim` = 256, i.e. 32 pairs; θ = 1e7).
   Port source: the reference's `rope.cuh` (the rotation) + `rope.cpp` (the
   `inv_freq` table, `inv_freq[pair] = θ^(-2·pair/rotary_dim)`). The reference
   fuses `rope(rmsnorm(x))` (`qk_norm_rope.cu`); **v1 does the
   `ignis_rmsnorm` (kernel-abi 02) + the RoPE as two steps** (the fused kernel
   is a later performance item, ADR 0005).

*Scope note:* the **bf16 logits GEMM** (for the W8-dequantized lm_head, A1) was
originally folded into this ticket; it is now a **separate ticket, A2b
(kernel-abi 10, `10-bf16-gemm.md`)** so A3 stays a pure assembly.

Both are **1:1 ports of the proven reference kernels** (provenance via ADR 0005
+ `kernel/NOTICE`), following the flat-C-ABI conventions (ADR 0001).

**Seam:** two new `extern "C"` entries in `kernel/include/ignis_kernel.h` + the
matching 1:1 FFI bindings in `crates/core/src/ffi.rs` (ADR 0001).

```c
/* (1) The GDN causal conv. `projected` is bf16 [tokens][channels] (the
 * projected q/k/v+z, the GEMM output); `conv_weight` is bf16 [4][channels]
 * (the 4 taps w0..w3); `state_in` / `state_out` are bf16 [channels][3] (the
 * 3-tap rolling conv state per channel; `state_out` receives the updated
 * state, `state_in` may alias it); `out` is bf16 [tokens][channels] (the
 * conv'd + SiLU'd q/k/v; the z rows pass through). channels = the GDN feature
 * width (query+key+value rows — NVFP4 10240 = 2048q+2048k+6144v; the z rows
 * bypass the conv). stream: null = stream 0. Returns 0 on success, -1 on
 * error. */
int ignis_gdn_causal_conv(const void* projected, const void* conv_weight,
                          const void* state_in, void* state_out, void* out,
                          int64_t tokens, int64_t channels, void* stream);

/* (2) The GQA RoPE (split-half NeoX). Rotates the first `rotary_dim` dims (of
 * `head_dim`) of each Q and K head: for a pair (a = x[i], b = x[i+R/2]),
 * out[i] = a·cos − b·sin, out[i+R/2] = b·cos + a·sin (cos/sin = f(pos,
 * inv_freq)). `q` is bf16 [batch][seq][num_q_heads][head_dim]; `k` is bf16
 * [batch][seq][num_kv_heads][head_dim]; `inv_freq` is fp32 [rotary_dim/2]
 * (the per-pair frequencies, θ = 1e7). In-place on q/k. stream: null = stream
 * 0. Returns 0 on success, -1 on error. */
int ignis_rope_qk(void* q, void* k, const void* inv_freq,
                  int64_t batch, int64_t seq, int64_t num_q_heads,
                  int64_t num_kv_heads, int64_t head_dim, int64_t rotary_dim,
                  int32_t pos, void* stream);
```

**Scope:**
- Port the two kernels (CUDA C++, `kernel/src/`), following the existing
  surface `.cu` / `.cuh` conventions (the forward-declared-in-`.cu` pattern,
  provenance via `kernel/NOTICE`).
- The two C-ABI entries + the 1:1 FFI bindings (`ignis_kernel.h` +
  `crates/core/src/ffi.rs`, ADR 0001).
- The `inv_freq` table (the RoPE frequencies, θ = 1e7, `rotary_dim` = 64) is
  computed once at construction (host-side, a deterministic table) and passed
  to `ignis_rope_qk` (a non-goal is per-step table recompute).

**Non-goals (v1):** the *fused* `qk_norm_rope` (rope∘rmsnorm) kernel (v1 does
rmsnorm + rope as two steps; the fused version is a later performance item,
ADR 0005); the YaRN / `attention_factor` scaling (v1 is unscaled, factor 1.0);
the tensor-core MMA for any of these (the FMA baseline is the v1 starting
point; the MMA is the later performance material, ADR 0005/0007).

**Acceptance:**
- **`ignis_gdn_causal_conv`:** a GPU-gated launch test (the
  `kernel_abi0N_gpu` convention, ADR 0006) runs the conv on synthetic
  multi-token input; the output matches a CPU reference conv (bf16 within the
  bf16 tolerance); the 3-tap state update (`s0,s1,s2`) is correct after the
  chunk; a `tokens == 1` case matches the single-token conv (the GEMV special
  case).
- **`ignis_rope_qk`:** a GPU-gated launch test runs the RoPE on synthetic q/k;
  the rotated output matches a CPU reference RoPE (bf16 within the tolerance);
  the `rotary_dim` = 64 / `head_dim` = 256 geometry (32 pairs) + the θ = 1e7
  angle table is correct; a `pos` sweep matches the reference.
- **The C-ABI entries + FFI bindings are 1:1** (`ignis_kernel.h` +
  `crates/core/src/ffi.rs`, ADR 0001).
- **`cargo test --workspace` green** (AGENTS.md: every code change ships with a
  test, and the task is not complete until it passes workspace-wide).

**Blocked by:** (none — a prerequisite for A3; the kernels are independent of
A1, and the bf16 GEMM is the separate A2b ticket, kernel-abi 10).

**References:** ADR 0001 (the flat C ABI + the GEMM/attention conventions),
ADR 0005 (port the proven CUDA "for now"; the pointwise/fused glue is the
correctness floor; re-implement later), ADR 0006 (exclusive GPU — the launch
tests self-skip on a busy GPU). The reference port sources (provenance,
consulted per ADR 0005): `src/ops/gdn_input_proj/gdn_projected_conv.cu`,
`src/ops/kernel/rope.cuh`, `src/ops/wrapper/rope.cpp`, `src/ops/launcher/
qk_norm_rope.cu`. kernel-abi 01 (the GDN step + the multi-token attention the
conv/RoPE feed), kernel-abi 02 (the `ignis_rmsnorm` the RoPE composes with).
A2b (`kernel-abi/specs/10-bf16-gemm.md`, the bf16 logits GEMM — the separate
ticket). A1 (`artifact/specs/04`, the W8→bf16 dequant the bf16 GEMM consumes).