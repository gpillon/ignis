# 01 — NVFP4 34816x5120 W4A4 at narrow prefill widths: Stages 3, weight-code L2 promotion, and the 256-token TMA floor

ADRs: 0010 (vendored reference kernels — every change here is a manifest-recorded
patch to `kernel/vendor/`, not a rewrite).
Evidence: `docs/findings/2026-09-18-ninfer-upstream-perf-survey.md` (what upstream
changed and why this is the only candidate aimed at a diagnosed problem),
`docs/findings/2026-09-11-prefill-chunk-wall-time.md` (the diagnosis: at 256 and
512 tokens the chunk is too narrow for the GEMM shapes, so per-token device
compute itself rises).
Upstream source: `cometkim/ninfer` `abbeea0a` *perf(ops): tune nvfp4 34816x5120
linear routes* (2026-09-15).

## Problem Statement

A 1,024-token prefill chunk costs 0.0974 ms/token. A 256-token chunk costs
**0.1607 ms/token — +65%** — and a 512-token chunk 0.1150 ms/token (+18%).
`prefill-chunk-wall-time` established that this is not the forced
synchronization (0.47 ms/chunk, 0.5%) and not launch latency: at narrow widths
**the per-token device compute itself rises**, because the chunk is too narrow
for the GEMM shapes.

Every prompt shorter than 1,024 tokens, every chunk tail, and every extra
traversal boundary that prompt reuse opens pays that penalty, on `mlp_gate_up`
(`N34816K5120`) among others — the largest weight block in the layer, on all 64
layers.

ignis's route selection for that shape is upstream's pre-`c491cd15` form and has
three specific gaps against what upstream now ships:

- **Stages.** `kernel/vendor/src/ops/linear/nvfp4/nvfp4_w4a4.cu:140` launches
  `MlpGateUp` with `TmaM256N128S2 = Nvfp4W4a4TmaSchedule<256, 2, 1>`. Every other
  shape in that file already uses `TmaM256N128 = <256, 3, 1>`. Upstream moved
  34816 to 3 stages.
- **L2 promotion.** `nvfp4_w4a4_tma.cuh:45` passes
  `CU_TENSOR_MAP_L2_PROMOTION_NONE` as a literal to `cuTensorMapEncodeTiled`.
  Upstream threads a `CUtensorMapL2promotion` through
  `nvfp4_make_tma_2d` → `make_nvfp4_w4a4_tma_descriptors` → a fourth
  `Nvfp4W4a4TmaSchedule` template parameter, and gives the 34816 weight-code
  descriptor `CU_TENSOR_MAP_L2_PROMOTION_L2_128B`: "K128 consumes 64 code bytes
  per row; prefetch the adjacent half-line for the next K tile."
- **TMA floor.** `nvfp4_w4a4.cu:57` reads
  `if (tokens >= 1024 && (tokens % kTmaBlockM) == 0)` with `kTmaBlockM = 256`.
  Upstream lowered the floor for this shape to `tokens >= 256`, keeping the
  `% 256` predicate, and re-cut the MMA ladder below it from
  64/96/128/192/384/512 to 32/64/128.

## Solution

Three changes, **each landing and being measured on its own**, in this order.
They are separable: nothing in AC2 depends on AC1, and AC3 is a route decision
that does not touch the kernels.

### AC1 — `MlpGateUp` takes the three-stage TMA schedule

`nvfp4_w4a4.cu:140` selects `TmaM256N128` instead of `TmaM256N128S2`.

De-risked by the tree itself: `<256, 3, 1>` is already instantiated and launched
for `AttnInput`, `GdnInput` and the residual shapes in the same file, so the
shared-memory image and the launcher's `cudaFuncSetAttribute` for dynamic smem
already admit it at M256/N128/K128. Nothing new is compiled.

If this is a regression on its own, stop here and record it: upstream bundled
Stages 3 and L2 promotion into one alias, so a bundled improvement could be
hiding a stage-count regression.

### AC2 — the weight-code TMA descriptor can request L2 promotion

Transposed verbatim from `abbeea0a`'s `nvfp4_w4a4_tma.cuh` diff:

- `nvfp4_make_tma_2d` gains a trailing
  `CUtensorMapL2promotion l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE`
  parameter, passed to `cuTensorMapEncodeTiled` in place of the literal;
- `make_nvfp4_w4a4_tma_descriptors` gains
  `CUtensorMapL2promotion weight_code_promotion = ..._NONE` and forwards it to
  the **`b_codes`** descriptor only — the activation-code and activation-scale
  descriptors keep `NONE`;
- `Nvfp4W4a4TmaSchedule` gains a fourth parameter
  `CUtensorMapL2promotion WeightCodePromotion = ..._NONE`, exposed as
  `kWeightCodePromotion`, and `launch_tma` passes `Schedule::kWeightCodePromotion`;
- a new alias
  `TmaM256N128Prefetch128B = Nvfp4W4a4TmaSchedule<256, 3, 1, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>`
  is used for `MlpGateUp` only.

Defaults keep every other call site byte-identical in behaviour.

### AC3 — the TMA floor drops to 256 for this shape, and the ladder below it is re-cut

`launch_problem<Geometry>` in `nvfp4_w4a4.cu` is **one ladder shared by every
geometry**, with `if constexpr` arms for the residual and GDN shapes; upstream's
`c491cd15` had already moved each shape into its own translation unit before
`abbeea0a` re-cut this one. ignis does not vendor that refactor.

**Departure, stated deliberately:** rather than importing `c491cd15`, add an
`if constexpr (Geometry::kOutputRows == Nvfp4MlpGateUpGeometry::kOutputRows)`
arm carrying this shape's floor and ladder, in the style the file already uses
for `kResidualGeometry` and the GDN arm. Reason: `c491cd15` touches 30 vendored
files with 21 of them also touched on the `gpillon` side, against a
one-`if constexpr` change here; the shared ladder is the thing this repo
vendors, and a per-shape arm keeps the departure legible in one place. If a
second shape later needs its own ladder, revisit.

Within that arm:

- floor `tokens >= 256 && (tokens % kTmaBlockM) == 0`;
- below it: `<= 32` → `M32N128`, `<= 64` → `M64N128`, `<= 128` →
  `M128N128Pipelined`, else `M128N128Resident`.

**Scope.** The identical `tokens >= 1024 && % 256` floor also appears in
`nvfp4_attn_input_w4a4.cu:88` and `nvfp4_gdn_input_w4a4.cu:44`. Upstream did not
lower those and neither does this ticket; they are separate shapes with separate
measurements.

**Not in scope.** Upstream also made `uses_a4` unconditional for this shape.
ignis's A16→W4A4 decision lives elsewhere (`nvfp4_dispatch.cpp`'s `resolve_route`,
`MlpGateUp: tokens >= 5`) and is a different question — leave it.

## Acceptance criteria

1. **AC1** lands alone, measured, and is kept only if it does not regress.
2. **AC2** lands alone on top of whatever AC1 concluded, measured. Every
   non-`MlpGateUp` call site is unchanged in behaviour (defaults are `NONE`).
3. **AC3** lands alone, measured. The `% 256` predicate is preserved, so only
   widths 256, 512 and 768 change route; **a 700-token tail still takes MMA**
   and this ticket promises nothing at such widths.
4. Workspace-wide `cargo test` is green, and the kernel's own CTests pass —
   including the NVFP4 route tests, which are what a mis-paired schedule breaks.
5. No change to `nvfp4_attn_input_w4a4.cu` or `nvfp4_gdn_input_w4a4.cu`.
6. `kernel/vendor/manifest.json` records each touched vendored file's patch with
   its reason, per ADR 0010. Nothing under `kernel/vendor/` changes without a
   manifest entry.

## Measurement

**The A/B is the chunk-width sweep, not the per-layer profile.** This shape is
`mlp_gate_up`, which both `gqa_layer.cu` and `gdn_layer.cu` launch on all 64
layers, so there is no in-run layer control.

Use `crates/core/tests/chunk_decomposition_gpu.rs` with `IGNIS_DECOMP_WIDTHS` at
**256, 512, 768, 1024**, same 8,192-token span, 3 timed reps after a warm-up —
the method `prefill-chunk-wall-time` already used, so the baseline is directly
comparable. Report ms/token per width, before and after each AC.

The numbers to beat, from that finding's Measurement 1:

| width | ms/token today |
| ---: | ---: |
| 256 | 0.1607 |
| 512 | 0.1150 |
| 1024 | 0.0974 (plateau) |

A result that moves 1024 is as interesting as one that moves 256: AC1 and AC2
act on every TMA launch of this shape, including the full chunk.

**GPU exclusivity.** One run at a time on the 5090 — check no other test, bench
or agent session holds the card first (`make gpu-status`), in any worktree.

## Risks

- **Silent numerics.** A TMA descriptor whose promotion does not match what the
  kernel assumes is a performance knob, not a correctness one; but the schedule
  is also the shared-memory image, so AC1's stage change must be verified by the
  route tests, not only by wall time.
- **Unmeasurable result.** These ceilings are small and a whole-request delta
  will not resolve them. If the sweep cannot separate before from after at
  3 reps, say so and close the AC rather than reporting a number inside the
  noise.
