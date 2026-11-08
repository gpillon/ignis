---
status: accepted
---

# ADR 0059 — Flash-Next applies YaRN, the attention factor folded into cos and sin on q and k

## Status

Accepted (2026-10-19, owner decision: "lo implementiamo assolutamente"). D1-D3 below are
the coordinator's, taken on the research's recommendations (finding
[Flash-Next YaRN in the reference engines](../findings/2026-10-19-flash-next-yarn-in-the-reference-engines.md)). Spec
[`rope-scaling/02`](../specs/rope-scaling/02-flash-next.md) is the implementation.
Lands with ADR 0057 (rope scaling is `auto` by default): Flash-Next runs the table
`model.rope_scaling` resolves to, as the 27B does.

## Context

Qwen3.8-Flash-Next is trained over 262,144 positions (`max_position_embeddings`), and ignis
serves it at 589,824 on the RTX 5090 and 1,048,576 on the RTX PRO 6000. Until now its load
refused a scaled table (`kernel/src/flash_next/program.cu`, "rotates at its own linear
table"), `flash_next_options` crossed the ABI with no scaling, and `fn_qsa_attention` refused an
attention factor other than 1, so every position past 262,144 ran the linear table
extrapolated.

The model card's recipe for ultra-long text is static YaRN, and all three reference engines
run the same math on this model (HF transformers `d4804f9d`, vLLM `e0adacc9`, SGLang
`da414add`):

- **The table.** HF's `_compute_yarn_parameters` on the 64 rotated dims (`head_dim` 256 x
  `partial_rotary_factor` 0.25), theta 1e7, `original_max_position_embeddings` 262,144,
  `beta_fast` 32, `beta_slow` 1, truncated ramp bounds: low 14, high 22. The same constants,
  and so the same bounds, as the 27B's YaRN.
- **The attention factor** `m = 0.1 ln(factor) + 1`, multiplied into the cos/sin table
  (`cos = freqs.cos() * attention_scaling`), which is applied to **q and k** alike on their
  rotated dims; the softmax scale stays `256^-0.5`. A score's rotated part therefore scales
  by `m^2`, its 192 pass-through dims not at all.
- **The indexer** rotates its queries and its pooled block keys with the same rotary object,
  mscale included. **The MTP head** builds an ordinary attention layer from the same
  `rope_parameters`.
- Every engine computes the angle as an FP32 product `fp32(position) x fp32 inv_freq`, and
  HF and vLLM round cos and sin to BF16.

The 27B's YaRN (spec rope-scaling/01, GitHub #227) is the same table but places the factor
differently: the vendored `ops::rope` scales q's rotated dims by `m^2` and leaves k unscaled
(its "factor-free cached K" contract), and its scaled route reduces the angle in FP64.

## Decision

**Flash-Next runs the YaRN table `model.rope_scaling` resolves to**, on attention, the
indexer and the MTP head alike: one table per load, on `Context::rope`.

- **D1: the factor is folded into cos and sin on both q and k.** Both Flash-Next rope sites
  (`qsa.cu` `prepare_kernel`, `indexer.cu` `rope_row`, through one helper in `rope.cuh`)
  compute `c = round_bf16(cosf(phi) * m)` and the same for `s`: the HF convention, rounded
  where HF rounds. `indexer::Rope` carries `m` as `scale`, copied from the table's
  `attention_factor`. The refusal of an attention factor in `fn_qsa_attention` is gone.
- **D2: the angle stays an FP32 product**, as all three references compute it and as the
  Flash-Next kernels already did.
- **D3: the table is `ignis::rope_yarn_frequencies(topology.rope_theta, topology.rotary_dim,
  262144, scaling)`**, the 27B's own builder (`flash_next::text_rope`), and `rope_from` turns it
  into the kernels' FP32 table as before: measured within 1 FP32 ulp of HF's formula evaluated
  in FP64 at factors 2, 2.25, 3 and 4, and held to 2 ulps by its CTest (spec AC 3). The 262,144
  is `flash_next::kOriginalPositions`, named for Flash-Next: `ignis_topology` carries no
  `max_position_embeddings`, and the artifact names none.
- **`none` is bit-identical.** No scaling still builds `rope_linear_frequencies`, untouched,
  and its scale is exactly 1.0f, so the new multiply is an IEEE identity: a `none` load's
  logits are the bytes they were before. Measured 2026-10-19 on the RTX 5090: all 2,560 rows
  of a real-text prompt, the last chunk selecting sparsely, hash to the capture taken before
  the change (spec AC 9).
- **The same resolved value reaches the plan, the load and the KV-disk identity.**
  `EngineOptions::rope_scaling` carries it to `plan_flash_next_reservations` and
  `load_flash_next`, whose options cross the four scalars as the 27B's `load_options` does
  (`none` as the factor 0, the leaf's linear branch). The server builds the options from the
  shape in one place (`flash_next_engine_options`) and opens KV-disk under the leaf's own
  value.

## Why D1 departs from the 27B's convention

The 27B's convention belongs to the vendored op's API: `ops::rope` is ninfer's file (ADR
0010), and the factor-free K is its contract. Flash-Next's kernels are ours (ADR 0043) and
are written to mirror torch's BF16 arithmetic step by step, because the indexer's output is
a discrete top-k, where agreeing with the reference bit for bit is the point. HF rounds
`cos * m` to BF16 and applies it to q and k; `m^2` on q alone would round differently and
move scores near the k-th block. The "factor-free K" reason does not carry over either: the
table is frozen for a load, and everything that keeps K past a step (KV-RAM, KV-disk,
retained pages, the switch's reload check) keys on it already.

The two conventions agree in exact arithmetic on every score: both give
`m^2 (q_rot . k_rot) + q_pass . k_pass`. They differ only where K is read directly (the
cached K's rotated dims carry `m`) and in BF16 rounding.

## Considered options

- **The 27B's convention** (`m^2` on q, FP64 angle): one convention across families.
  Rejected: it moves away from the reference on a discrete output, through ops that are ours
  to shape.
- **D3 (ii), HF's own FP32 builder sequence** (zero ulps against HF, assuming a correctly
  rounded `powf`). Rejected for now: (i) measured within 1 FP32 ulp of HF's formula in FP64
  at every pair (the research's emulation of HF's own FP32 sequence put it within 2 ulps on
  the blended pairs 15-21, about 1e-4 rad at 1M positions on pair 15), and keeps one double
  table as the source of truth. Revisit if a long-context measurement says otherwise.
- **A topology field for the original positions.** Rejected for now: it is 262,144 in both
  checkpoints, and the field would mean a new ABI layout (`ignis_topology`'s 288-byte
  static asserts and the Rust mirror) for one constant.

## Consequences

- A Flash-Next load past 262,144 tokens rotates at YaRN once `model.rope_scaling` resolves
  to it (`auto` past 262,144 tokens, ADR 0057; an explicit `yarn:F` anywhere). Short contexts at a
  YaRN factor change too, since `m` multiplies every position: the cost every reference and
  the model card warn about. Measured, not judged: on 2,560 real-text tokens a `yarn:4` load
  picked `none`'s argmax on 2,337 rows (91.3%), and at factor 4 the indexer selected other
  blocks than the linear table would on 538 of 549 sparse rows of a synthetic prefill.
- The hq-e8-2b codec encodes K whose 64 rotated dims are up to `m` (1.139 at factor 4) larger.
  Its per-block metadata should absorb that; BF16 KV stays the oracle (ADR 0022) for checking
  it, and the long-context measurements run both formats.
- KV-disk's identity now labels Flash-Next's files with the table they were rotated with.
  Files written before this change carry the profiles' `yarn:4` over linear-rotated pages.
  KV-disk v1 never reads a file another process wrote, so nothing restores them; a version
  that reuses files across restarts must bump `FORMAT_VERSION` (spec vram-budget/03).
- vLLM is not a reliable oracle for the MTP head past 262,144: under its command-line recipe
  the draft does not see a dict `--hf-overrides` and runs the linear table. ignis follows HF
  and SGLang: one table for every attention layer, the head's included.
