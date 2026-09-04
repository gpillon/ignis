# 07 — full-correct Qwen 3.8-27B forward assembly

GitHub: #30 (the #25 follow-up — A3, "full-correct 27B forward assembly")

The compute-adapter (kernel-abi 04) has the forward-pass *seam* (the
`prefill` / `decode` / `forward_layers` methods) + a **synthetic** model
(`ModelConfig::synthetic` + the synthetic `Weights`) + a `from_artifact` path
that currently holds a **`placeholder`** (empty) host-side `Weights` (the #26
fix — the 19 GB is in VRAM, but the host-side `Weights` are empty until A1
lands) and a *simplified* layer stack (no GDN causal conv, no RoPE, no q/k
RMSNorm). The **full-correct** Qwen 3.8-27B model — the numerically-correct
forward pass — is not yet assembled: it needs the GDN causal conv + the GQA
RoPE (A2), the bf16 logits GEMM (A2b), the full weight routing (A1's
normalization), and the real topology.

**Seam:** the `CudaCompute::from_artifact` (the production constructor) + the
forward pass (`forward_layers` / `prefill` / `decode`) in
`crates/core/src/compute.rs`. The layer stack composes, per layer kind:

- **GQA layer** (`(i+1) % 4 == 0`, 16 of 64): the QKV projection (the NVFP4
  GEMM `ignis_nvfp4_gemm_*`, A1) → the q/k RMSNorm (`ignis_rmsnorm`, kernel-abi
  02) + RoPE (`ignis_rope_qk`, A2) → the attention (`ignis_gqa_attention_
  prefill` / `_decode`, kernel-abi 01) → the output projection (the NVFP4 GEMM)
  → the residual.
- **GDN layer** (the other 48): the input projection (the NVFP4 GEMM) → the GDN
  causal conv (`ignis_gdn_causal_conv`, A2) → the GDN step (`ignis_gdn_step`,
  the recurrence, kernel-abi 01) → the output projection (the NVFP4 GEMM) →
  the residual.
- **FFN (every layer):** the gate_up projection (the NVFP4 GEMM) → the gated
  SiLU (the host pointwise glue, ADR 0005) → the down projection (the NVFP4
  GEMM) → the residual.
- **The endpoints:** the embedding (the W8→bf16, A1; `ignis_embedding`) + the
  lm_head (the W8→bf16, A1; the logits via the `ignis_bf16_gemm`, A2b) + the
  final RMSNorm + the greedy sample (`ignis_greedy_sample`).

The **real topology** (`ModelConfig::qwen38_27b`, not the synthetic): 64 layers
(GQA every 4th, the rest GDN), `hidden` = 5120, 24 q-heads / 4 kv-heads,
`head_dim` = 256, `rotary_dim` = 64 (θ = 1e7), `ffn_intermediate` = 17408,
GDN state 6144 × 2048, `block_size` = 64, `num_blocks` = 4096, `vocab` =
248 320.

**Scope:**
- Wire **A1's normalized weights** (the real artifact tensors, not a
  `placeholder`) into the `from_artifact` path (the host-side `Weights` are the
  real normalized buffers).
- Add **A2's GDN causal conv + the GQA RoPE** (the `ignis_gdn_causal_conv` +
  `ignis_rope_qk` kernels) into the layer stack (the GDN layers' input, the GQA
  layers' q/k).
- The **logits GEMM** for the W8-dequantized lm_head (the `ignis_bf16_gemm`,
  A2b) — replacing the synthetic NVFP4 `lm_head` GEMM on the real path.
- The **real 27B topology** (the `qwen38_27b()` config: the real layer kinds +
  the head geometry + the GDN state dims + the rotary geometry) — the forward
  pass runs the *real* model, not the synthetic one.

**Non-goals (v1):** the CUDA-graph fast path (B2); the batched prefill (B1);
the *performance* (the 99% gate is #20 — A3 is the **correctness** (the
full-correct forward pass); the performance material is B1/B2); the
vision / mtp / dflash2 features (not the v1 text model).

**Acceptance:**
- **A real-model forward pass is sane + reproducible (the correctness floor,
  ADR 0005/0007):** a GPU-gated e2e test (self-skip on a busy GPU, ADR 0006)
  runs a real-model forward pass (a prompt prefill + a few decode steps)
  through `CudaCompute::from_artifact`: the emitted token ids are in vocab
  range, and a greedy + fixed-seed run is **reproducible** (same input → same
  tokens — self-consistency, ADR 0007) — a *sane* completion for the prompt.
- **The server runs a real model:** `ignis-server` started with
  `IGNIS_ARTIFACT` (the `from_artifact` path, not a `placeholder`) serves
  `/v1/chat/completions` and returns a *real* completion (the `bench-03`
  gate-run's "the server runs a real model" acceptance, the 99% gate's
  prerequisite).
- **The `from_artifact` `Weights` are the real normalized weights** (A1), not a
  `placeholder` — the host-side `Weights` geometry matches the 27B topology.
- **The full layer stack is exercised:** a synthetic-model CPU test verifies
  the GQA + GDN + FFN layer composition (the conv + the RoPE are called for the
  right layers, the geometry matches `qwen38_27b`).
- **`cargo test --workspace` green** (AGENTS.md: every code change ships with a
  test, and the task is not complete until it passes workspace-wide).

**Blocked by:** A1 (the mixed-quant normalization — the real weights) + A2 (the
GDN causal conv + RoPE kernels) + A2b (the bf16 logits GEMM).

**References:** ADR 0005 (the correctness floor — a *sane* output, not a
reference match; the ported kernels are the "for now" starting point), ADR
0007 (the self-consistency / the performance gate — the *performance* is B1/B2
+ #20, not A3), ADR 0006 (exclusive GPU — the e2e test self-skips on a busy
GPU). kernel-abi 01 (the attention + the GDN step), kernel-abi 02 (the norms +
the embedding + the sample), kernel-abi 04 (the compute-adapter seam). A1
(`artifact/specs/04`, the normalized weights), A2 (`kernel-abi/specs/06`, the
conv + RoPE kernels), A2b (`kernel-abi/specs/10`, the bf16 logits GEMM).