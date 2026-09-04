# 04 — mixed-quant materialization / normalization (artifact tensors → kernel formats)

GitHub: #27 (the #25 follow-up — A1, "mixed-quant materialization/normalization")

The artifact (`.ninfer`, ADR 0002) carries a **per-tensor `format` + `layout`**
(the container is the authority — each tensor entry is
`{name, kind, shape, format, layout, offset, bytes}`). The compute-adapter
(kernel-abi 04) today only *materializes* the NVFP4 GEMM weights to VRAM
(`ignis_nvfp4_gemm_*_device`) and leaves the host-side `Weights` a **`placeholder`
(empty)** — the #26 fix. The full-correct 27B model (A3) needs **every** artifact
tensor to land in the format its *target kernel* expects. This ticket is the
normalization layer between the materialized artifact and the compute-adapter's
`Weights`.

**The rule (the mixed-quant materialization/normalization, not a blanket
dequant):** **preserve** a tensor's format directly where the target kernel
consumes it; **dequantize only the exceptional formats** to bf16.

- **NVFP4** (`blockscale-k16-m128x4-v1`: E2M1 codes + E4M3 per-16 scale + the
  per-tensor u32 `weight_divisor`): the GEMM weights (`gdn/query_key_value_z`,
  `gdn/output`, `attention/query_key_gate_value`, `attention/output`,
  `mlp/gate_up`, `mlp/down` — ~247 tensors). **Preserve NVFP4** — the
  `ignis_nvfp4_gemm_*` kernel consumes codes + scales directly (no dequant);
  the `weight_divisor` is the dequant scale.
- **BF16** (`contiguous-le-v1`): all the norms, `gdn/convolution`,
  `gdn/a_b_projection`, and the early-attention exceptions. **Preserve BF16** —
  the `ignis_rmsnorm` / `ignis_gdn_causal_conv` / attention kernels consume
  bf16.
- **FP32 / I32** (`contiguous-le-v1`): `gdn/a_log`, `gdn/dt_bias`, the
  `input_scale_divisor` siblings. **Preserve FP32 / I32** (the GDN recurrence
  params / the dequant divisors).
- **W8 / Q4 / Q5 / Q6** (the `row-split-k128-v1` layout: code planes + per-64/32
  F16 scale planes): the *exceptional* formats. **Dequantize to bf16** — the
  `text/token_embedding` + `text/output_head` (the embedding table + the lm_head
  logits GEMM, W8G32), the mtp weights, and the vision backbone (Q4/Q5 — not in
  the v1 *text* scope).

**Seam:** a `normalize` step in `crates/artifact` (a new module, or an extension
of the materializer, `artifact/specs/01`) that maps each artifact tensor (by the
container's `format` + `layout`) to its kernel-expected buffer; the
compute-adapter's `from_artifact` path (A3) consumes the normalized weights
(the host-side `Weights` are the real normalized buffers, not a `placeholder`).

**Scope:**
- The `normalize` function: given a materialized artifact tensor (name →
  `format` + `layout` + raw bytes), produce the kernel-expected buffer (NVFP4
  codes+scales+divisor for the GEMM weights; bf16 for the norms/conv/
  `a_b_projection`; fp32/i32 for the GDN params; dequantized bf16 for the
  W8/Q4/Q5/Q6 row-split tensors).
- The **text-scope** (the 27B model) tensors are all normalized: the NVFP4 GEMM
  weights, the BF16 norms/conv, the FP32 GDN params, the W8 embedding + lm_head
  (dequant to bf16).
- A `Weights` populated from the normalized buffers (replacing the `placeholder`
  on the `from_artifact` path).

**Non-goals (v1):** the vision / mtp / dflash2 / draft_head tensors (not in the
v1 *text* inference scope — noted, not normalized for v1 text); the FP8 profile
(not this artifact); the *fused-kernel* dequant (the dequant is host-side for
now; the fused-kernel dequant is a later performance item, ADR 0005).

**Acceptance:**
- **The format distribution is normalized:** a CPU test verifies that every
  NVFP4 tensor → codes + scales + divisor (the NVFP4 GEMM weight), every
  BF16 / FP32 / I32 tensor → as-is, and every W8 / Q4 / Q5 / Q6 tensor →
  dequantized bf16 (the W8→bf16 dequant matches a CPU reference for a synthetic
  W8 tensor).
- **The text-scope tensors are all present in the kernel formats:** a test
  verifies the NVFP4 GEMM weights + the BF16 norms/conv + the FP32 GDN params +
  the W8 embedding/lm_head (dequanted bf16) are all produced (the 27B model's
  weights are complete, no `placeholder` left for the text scope).
- **The `from_artifact` `Weights` are the real normalized buffers** (not a
  `placeholder`) — a non-GPU / CPU test verifies the `Weights` geometry matches
  the 27B topology (`ModelConfig::qwen38_27b`).
- **`cargo test --workspace` green** (AGENTS.md: every code change ships with a
  test, and the task is not complete until it passes workspace-wide). The
  GPU-gated parts (the VRAM materialization) self-skip on a busy GPU (ADR 0006).

**Blocked by:** (none — a prerequisite for A3; it is a starting point and can be
worked independently).

**References:** ADR 0002 (the artifact is the source of truth; the per-tensor
`format` + `layout` is the container's authority), ADR 0001 (the flat C ABI +
the NVFP4 GEMM contract), ADR 0006 (exclusive GPU — the GPU-gated
materialization self-skips). The artifact reader (kernel-port 02) + the
materializer (artifact 01). The reference sidecars (`conversion.json` /
`graft.json`) document the object provenance (ADR 0002).