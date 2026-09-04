# 08 — batched prefill (the multi-token forward path)

GitHub: #TBD (the #25 follow-up — B1, "batched prefill")

The compute-adapter's `prefill` (kernel-abi 04, `crates/core/src/compute.rs`)
runs the layer stack **per token**: `for pos in 0..seq { forward_layers(pos) }`,
where each iteration is a *single-token* forward pass (the single-token GEMV
`ignis_nvfp4_gemm_decode` + the single-token attention
`ignis_gqa_attention_decode`). The performance path is the **multi-token**
forward pass: every projection is a multi-token GEMM
(`ignis_nvfp4_gemm_prefill`, kernel-abi 05) and every GQA layer is a
multi-token attention (`ignis_gqa_attention_prefill`, kernel-abi 01), so a
prompt of `seq` tokens is processed in *one* pass over the layer stack instead
of `seq` passes. This is the performance material the 99% gate (ADR 0007)
drives — the per-token loop is the **eager fallback** (ADR 0003).

**Why this is the performance path (design §7, an experiment):** the per-token
loop does `seq × 64` single-token C-ABI calls (each an H2D/D2D round-trip,
ADR 0001) for the prefill — prefill throughput collapses on long prompts (the
main-agent coding prompts are long). The multi-token GEMM + multi-token
attention replace that with `64` multi-token calls. *Caveat (design §7):* we
may be **compute-bound**, in which case batched prefill is useless — it is an
experiment to measure (#20) before relying on it, and it *changes the kernel
accumulation order*, so it is re-gated by the 99% performance gate (ADR 0007).

**Seam:** `Compute::prefill_step` in `crates/core/src/compute.rs` (the
`CudaCompute::prefill` + the layer-stack method). The batched path processes a
prompt's `seq` tokens in one multi-token pass; the per-token loop remains the
eager fallback (a busy/absent multi-token kernel, or `seq == 1` — the single
token is the GEMV special case, ADR 0001).

**Scope:**
- **The multi-token forward path:** a `seq`-token activation (bf16
  `[seq][hidden]`) driven through the 64-layer stack in one pass:
  - **GQA layers:** the QKV projections + the output projection are multi-token
    GEMMs (`ignis_nvfp4_gemm_prefill`); the attention is the multi-token
    `ignis_gqa_attention_prefill` (the batched query attends over the whole
    sequence); the KV cache is written with all `seq` tokens' K/V.
  - **GDN layers:** the input projection + the readout are multi-token GEMMs
    (`ignis_nvfp4_gemm_prefill`); the **GDN recurrence is sequential** — the
    Gated-DeltaNet state update (`S ← αS + δkᵀ`) is a per-token recurrence, so
    within a prefill chunk the GDN step runs **per token** (the
    `ignis_gdn_step` kernel, kernel-abi 01) or via a fused GDN-prefill kernel
    (a later optimization, non-goal for v1). The projections are batched; the
    recurrence is not (inherently sequential).
  - **The gated-FFN (every layer):** the gate/up/down projections are
    multi-token GEMMs (`ignis_nvfp4_gemm_prefill`); the gated-SiLU activation
    is the host pointwise glue for now (the fused-SiLU kernel is a later
    performance item, ADR 0005).
  - **The final norm + lm_head + sample:** the final RMSNorm, the lm_head
    multi-token GEMM (`ignis_nvfp4_gemm_prefill`), and the greedy sample
    (`ignis_greedy_sample`) over the last token's logits.
- **The eager fallback (ADR 0003):** the per-token loop (the current
  `prefill`) remains as the fallback when the multi-token path is unavailable
  (a busy/absent kernel) — the correctness floor is unchanged.
- **The KV writeback:** the paged KV cache is filled with all `seq` tokens'
  K/V in the multi-token attention (the `block_table` / `kv_len` update for the
  whole chunk).

**Non-goals (v1):** a *fused* GDN-prefill kernel (the per-token GDN recurrence
within the chunk is fine for v1); chunked / lazy prefill beyond the whole
prompt; the *measurement* of the prefill throughput (that is #20 / the 99%
gate, ADR 0007).

**Acceptance:**
- **`prefill_step` uses the multi-token path** when `seq > 1` (the multi-token
  GEMM + multi-token attention, not the per-token loop); `seq == 1` is the GEMV
  special case (the single-token path, ADR 0001).
- **Sane output (the correctness floor, ADR 0007):** a GPU-gated test runs a
  batched prefill of a multi-token prompt through `CudaCompute` and the emitted
  logits are in vocabulary range (the greedy sample is valid — a *sane*
  completion for the prompt; self-consistency, greedy + fixed seed, ADR 0007).
  *Note:* the batched path's accumulation order differs from the per-token
  loop (design §7), so the acceptance is *sane output*, **not** bit-exact
  agreement with the per-token loop — the 99% performance gate (#20) is the
  re-check.
- **The eager fallback is preserved:** a `seq == 1` prompt (and a
  busy/absent multi-token kernel) runs the per-token loop (the ADR 0003 eager
  fallback) — a non-GPU / dev-mode host still gets a (correct, eager) prefill.
- **`cargo test --workspace` green** (AGENTS.md: every code change ships with a
  test, and the task is not complete until it passes workspace-wide).

**Blocked by:** A3 (the full-correct 27B forward assembly — the multi-token
path is only meaningful once the full model is assembled; the *mechanism* is
testable on the synthetic model first).

**References:** ADR 0003 (the eager fallback), ADR 0005 (the pointwise glue is
the correctness floor; the fused-SiLU / fused-GDN-prefill kernels are the later
performance material), ADR 0007 (the 99% performance gate + the *sane-output*
self-check — the re-check for the changed accumulation order). kernel-abi 01
(the multi-token attention + the GDN step), kernel-abi 05 (the multi-token
NVFP4 GEMM), kernel-abi 04 (the compute-adapter). design §7 (batched prefill =
an experiment to measure, ADR 0007).