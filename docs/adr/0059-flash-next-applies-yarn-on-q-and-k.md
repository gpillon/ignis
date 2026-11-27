---
status: accepted
---

# Flash-Next applies YaRN, the attention factor folded into cos and sin on q and k

Qwen3.8-Flash-Next is trained over 262,144 positions and ignis serves it past them. The model card's recipe for
ultra-long text is static YaRN, and all three reference engines (HF transformers, vLLM, SGLang) run the same math on
this model
([finding](../findings/2026-10-19-flash-next-yarn-in-the-reference-engines.md)). So Flash-Next runs the YaRN table
`model.rope_scaling` resolves to, as the 27B does (`auto` past the trained positions, an explicit `yarn:F`
anywhere: ADR 0057): one table per load, on attention, the indexer and the MTP head alike, with YaRN's attention
factor `m` placed as the references place it.

- **The factor is folded into cos and sin on both q and k.** Both Flash-Next rope sites compute
  `c = round_bf16(cosf(phi) * m)` and the same for `s`: the HF convention, rounded where HF rounds.
- **The angle stays an FP32 product**, as all three references compute it and as the Flash-Next kernels already
  did.
- **The table is the 27B's own builder's** (`ignis::rope_yarn_frequencies`), turned into the kernels' FP32 table:
  measured within 1 FP32 ulp of HF's formula evaluated in FP64 at factors 2, 2.25, 3 and 4, and held to 2 ulps by
  its CTest. The original positions are `flash_next::kOriginalPositions`, named for Flash-Next.
- **`none` is bit-identical.** No scaling still builds the linear table, untouched, and its scale is exactly 1.0f,
  so the multiply is an IEEE identity: a `none` load's logits are the bytes they were before Flash-Next applied
  YaRN.
- **The same resolved value reaches the plan, the load and the KV-disk identity.**

## Considered Options

- **The 27B's convention**, one convention across families: the vendored `ops::rope` scales q's rotated dims by
  `m^2`, leaves k unscaled and reduces a scaled angle in FP64. Rejected: it moves away from the reference on a
  discrete output, through ops that are ours to shape. That convention belongs to the vendored op's API:
  `ops::rope` is ninfer's file (ADR 0010), and the factor-free K is its contract. Flash-Next's kernels are ours
  (ADR 0043) and mirror torch's BF16 arithmetic step by step, because the indexer's output is a discrete top-k,
  where agreeing with the reference bit for bit is the point: HF rounds `cos * m` to BF16 and applies it to q and
  k, and `m^2` on q alone would round differently and move scores near the k-th block. The factor-free K reason
  does not carry over either: the table is frozen for a load, and everything that keeps K past a step (KV-RAM,
  KV-disk, retained pages, the switch's reload check) keys on it already. The two conventions agree in exact
  arithmetic on every score, `m^2 (q_rot . k_rot) + q_pass . k_pass`, and differ only where K is read directly
  (the cached K's rotated dims carry `m`) and in BF16 rounding.
- **HF's own FP32 builder sequence** (zero ulps against HF, assuming a correctly rounded `powf`): rejected for now.
  The 27B's builder is within 1 FP32 ulp of HF's formula in FP64 at every pair (the research's emulation of HF's
  own FP32 sequence put it within 2 ulps on the blended pairs 15-21, about 1e-4 rad at 1M positions on pair 15),
  and keeps one double table as the source of truth. Revisit if a long-context measurement says otherwise.
- **A topology field for the original positions**: rejected for now. It is 262,144 in both checkpoints,
  `ignis_topology` carries no `max_position_embeddings` and the artifact names none, and the field would mean a new
  ABI layout (`ignis_topology`'s static asserts and the Rust mirror) for one constant.

## Consequences

- Short contexts at a YaRN factor change too, since `m` multiplies every position: the cost every reference and
  the model card warn about. Measured, not judged: on 2,560 real-text tokens a `yarn:4` load picked `none`'s argmax
  on 2,337 rows (91.3%), and at factor 4 the indexer selected other blocks than the linear table would on 538 of
  549 sparse rows of a synthetic prefill.
- The hq-e8-2b codec encodes K whose 64 rotated dims are up to `m` (1.139 at factor 4) larger. Its per-block
  metadata should absorb that; BF16 KV stays the oracle (ADR 0022) for checking it, and the long-context
  measurements run both formats.
- KV-disk's identity labels Flash-Next's files with the table they were rotated with. Files written before
  Flash-Next applied YaRN carry the profiles' `yarn:4` over linear-rotated pages: KV-disk v1 never reads a file
  another process wrote, so nothing restores them, and a version that reuses files across restarts must bump
  `FORMAT_VERSION`.
- vLLM is not a reliable oracle for the MTP head past 262,144: under its command-line recipe the draft does not see
  a dict `--hf-overrides` and runs the linear table. ignis follows HF and SGLang: one table for every attention
  layer, the head's included.
