# The vendored GQA workspace query under-reports an hq prompt call at widths 9..16

- Kind: discovery
- Status: current
- Observed: 2026-09-12
- Last verified: 2026-10-01
- Scope: kernel / vendored `ninfer::ops::gqa_attention` capacity query, hq-e8-2b prefill
- Related: [GitHub #123](https://github.com/gpillon/ignis/issues/123),
  [GitHub #296](https://github.com/gpillon/ignis/issues/296),
  [ADR 0037](../adr/0037-a-vendored-file-may-carry-a-correctness-patch.md),
  [ADR 0010](../adr/0010-vendored-reference-kernels.md),
  [ADR 0022](../adr/0022-two-kv-formats-bf16-as-oracle.md),
  `kernel/include/ignis_gqa_workspace.h`
- Superseded by: none

## Question

Wiring the hq-e8-2b attention routes (#123) made a 12-token prompt fail
prefill with `layer 3: ignis_gqa_layer_step: bad allocation`, while the same
prompt served fine under BF16 and hq served fine at 200 tokens. What is
special about a narrow hq prefill?

## Evidence

`ninfer::ops::gqa_attention_workspace_capacity_bytes`
(`kernel/vendor/src/ops/wrapper/gqa_attention.cpp`) is the caller's only
source for how much transient arena one A1 call needs.

An hq-e8-2b **Prompt** call allocates two transient riders in the same
invocation, both from that arena and both live at once
(`gqa_attention`, same file):

1. the rotated-frame BF16 span planes `allocate_hq_prompt_scratch` materializes
   the visible history into, and
2. the key-split partials (`split_acc` / `split_m` / `split_l`), allocated when
   `gqa_prefill_split_count(width, q_heads) > 1`.

The query sums the two only in the branch whose width loop starts at
`std::max(min_width, kMaximumVerifyTokens + 1)` — that is, at width 17. For a
query whose whole interval sits at or below the 16-token verify cap, that loop
is empty, the split partials reach the answer only through the separate
`exact_capacity` loop, and the two are combined with `std::max` rather than
added.

Measured at the 27B geometry (24 q heads, 4 KV heads of 256), width 12,
envelope 12: span planes 49,152 bytes and split partials about 1.19 MB with
`gqa_prefill_split_count` returning 4. The query answers about 1.19 MB; the
call needs both, about 1.24 MB. The op's own arena bump throws `std::bad_alloc`
part-way through the layer.

BF16 and INT8 are unaffected: neither has rider (1), so for them the query's
`max` and the true sum coincide.

**Past one band (GitHub #296, observed 2026-10-01).** The same query
under-reports a banded hq Prompt call too: one whose envelope is past one
262,144-key scratch band, where the route chains a carry state across bands
instead of splitting keys. It adds that carry as raw arithmetic,
`(2 * head_dim + 8) * q_heads * width`, but the op bumps the carry's three
tensors at 256-byte alignment, so the answer is short by the padding after
`carry_m` whenever `96 * width` is not a multiple of 256. The split partials,
far larger, hide it except at a width `gqa_prefill_split_count` keeps at one
split: 321..448 and 769..896 at the 27B geometry on a 170-SM RTX 5090, 224
widths among 17..1024. In serving, an agent conversation past 344K keys whose
last prompt chunk was 330 tokens wide threw `layer 3: ignis_gqa_layer_step:
bad allocation` on every request.

## Finding

**Observed.** The vendored capacity query under-reports an hq-e8-2b Prompt
call by exactly one split-partial set whenever the queried width interval lies
at or below the wrapper's own 16-token verify cap. The hq small-T tile is 8
tokens, so the affected window is widths 9..16 at batch 1.

**Inference.** The reference's own engine does not meet this, which is why the
bug is still there: it queries once over a whole width interval (as this
repo's `ignis_model_load` does, over `[1, prefill_chunk]`), where the widths
above the cap dominate the answer. ignis asks per call, at exactly the width it
is about to run — a narrower call pattern than the query handles correctly.

## Implications

- ignis corrects this at the caller, not in the vendored file
  (`ignis_gqa_attention_workspace_bytes`,
  `kernel/include/ignis_gqa_workspace.h`): for an hq Prompt width below the
  cap it asks the query over `[width, 17]` with the envelope raised to match,
  which the query's own contract ("the capacity required for every W in the
  inclusive interval") makes a true upper bound. It changes no kernel, no
  numerics and no route — only how many bytes the caller reserves. ADR 0010
  keeps the vendored subtree verbatim, and a capacity-query patch would be the
  first divergence in vendored *source* rather than a test.
- The window is not exotic. A 9..16-token prompt is an ordinary short request,
  and so is the tail chunk of any longer span whose length falls in that window
  modulo the prefill chunk.
- `kernel/tests/test_hq_route_agreement.cu` runs every width in 9..16 so the
  correction has a test rather than a story.
- The banded under-report (GitHub #296) is fixed in the vendored file, as the
  follow-up below proposed: ADR 0037 (2026-09-22) admits a recorded
  correctness patch, and the query violated its own contract with an
  observation in ignis behind it. The patch sizes the carry through
  `WorkspaceLayoutBuilder`; `ignis_kernel_gqa_workspace_capacity_test` replays
  the op's allocations at every serving prompt width, in one band and past it,
  and its banded arms are red without the patch on a 170-SM card.
  `kernel/vendor/VENDOR.md` lists it as an Ignis-patched behaviour. The
  narrow-width correction above stays at the caller: a different bug, with its
  own test.

## Limits and unknowns

- The per-call numbers above are for the 27B geometry on an RTX 5090;
  `gqa_prefill_split_count` is a wave-fill model over the SM count, so the
  split count — and therefore the size of the shortfall — is machine-dependent.
  A machine where it returns 1 at these widths would not reproduce the failure.
- Not checked against the reference engine's own call sites; the inference that
  it queries over an interval is from the code's own comments, not from
  observing it run.
- Whether the reference has since fixed it upstream is unknown: the vendored
  tree is pinned at `kernel/vendor/manifest.json`.

## Follow-ups

- A recorded vendored patch (`kernel/vendor/patches/`) would fix it for every
  caller instead of this one, and would be the honest place for it if the
  engine ever grows a second caller of the query. That is an owner decision
  about ADR 0010's boundary, not something #123 took.
