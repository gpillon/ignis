# What upstream ninfer's last month of perf work offers ignis: four portable candidates, one blocker, and a large re-vendor bill

- Kind: research
- Status: current
- Observed: 2026-09-18
- Last verified: 2026-09-18
- Scope: kernel / vendored subtree currency, NVFP4 linear routes, GQA attention routes, build time
- Related: [ADR 0010](../adr/0010-vendored-reference-kernels.md),
  [Decode round anatomy](2026-09-18-decode-round-anatomy.md),
  [Decode round host idle](2026-09-18-decode-round-host-idle.md),
  [Prefill chunk wall time](2026-09-11-prefill-chunk-wall-time.md),
  [Vision TTFT live/live](2026-09-16-vision-ttft-live-live.md),
  [hq prompt workspace under-report](2026-09-12-hq-prompt-workspace-under-report.md)
- Superseded by: none

## Question

Upstream `cometkim/ninfer` shipped a month of performance commits after the
point ignis vendored from. Which of them apply to what ignis actually runs,
and what would it cost to take each one?

This is a **survey, not a measurement**. Nothing below was run on the card.
Every "gain" is a hypothesis with a named route and a named shape; the only
numbers quoted are ignis's own already-measured findings, used to bound what a
candidate could be worth.

## Evidence

### Where the vendored pin sits

`kernel/vendor/manifest.json` pins `gpillon/ninfer` branch `gpillon/coding` at
`a00648cb` (2026-09-03), 410 files. The reference checkout is on disk at
`F:/ai/q38/ninfer`.

- merge-base of the pin with `cometkim/master` is `a05746aa` (2026-08-18);
- `cometkim/master` has **263 commits** since that base, head `1d8587bc`
  (2026-09-15);
- `cometkim/master` is an ancestor of `cometkim/dev` (head `024f6a94`,
  2026-09-16), so the clean, cherry-pickable commits all live on `master`;
  `dev` adds only the fork-overlay squashes on top.

Intersecting the 410 manifest paths with upstream history:

| measure | count |
|---|---|
| vendored files touched by `a05746aa..origin/master` | 217 |
| vendored files also touched on the `gpillon` side (conflict surface) | 78 |
| upstream commits touching at least one vendored file | 86 of 263 |
| vendored files whose **content** differs from `origin/master` | 292 of 410 |
| of those, absent from `origin/master` entirely (fork-only) | 95 |

Reproduce with `git ls-tree -r` on both revisions against the manifest paths;
the scratch lists are in `.scratch/` of the session that produced this.

**A wholesale pin bump is therefore not a small operation** — 292 of 410 files
move, and 95 of them (the DFlash2 ops, the Windows port) exist only on the fork
side. Per-change transposition is the realistic route.

### What ignis actually runs, which decides applicability

- artifact: `ARTIFACT ?= ./models/qwen3_8_27b_nvfp4full-v2.ninfer` (`mk/config.mk`)
  — **NVFP4 weights, v2 artifact**;
- KV format: `KV_FORMAT ?= hq-e8-2b`;
- prefill chunk: `DEFAULT_SERVING_CHUNK_TOKENS = 1024` (`crates/core/src/concrete.rs:237`);
- GQA geometry: 24 query heads over 4 KV heads of 256
  (`kernel/include/ignis_gqa_workspace.h:28`);
- linear shapes launched: hidden 5120, q 6144, gdn 16384, mlp gate_up 34816
  (ffn 17408), 14336 — i.e. `N34816K5120`, `N5120K17408`, `N5120K6144`,
  `N14336K5120`, `N16384K5120`;
- W4A4 is live: `kernel/src/linear.cu`, `gqa_layer.cu` and `linear_swiglu.cu`
  all delegate the A4 decision to the vendored plan, and the vendored
  `nvfp4_w4a4_tma.cu` compiles.

This kills a whole class of upstream work on sight. The `perf(ops): tune q4
1024x5120 / 4096x5120 / 6144x5120` series, the q5/q6/q8/w8 shape work and the
fp8 route consolidation act on artifacts ignis does not serve.

### Candidate 1 — W4A4 TMA route floor and L2 promotion on the gate_up shape

`abbeea0a perf(ops): tune nvfp4 34816x5120 linear routes` (2026-09-15) changes
exactly ignis's `mlp_gate_up` shape:

- `nvfp4_make_tma_2d` gains a `CUtensorMapL2promotion` parameter and
  `make_nvfp4_w4a4_tma_descriptors` a `weight_code_promotion`, threaded from a
  new `Nvfp4W4a4TmaSchedule` template parameter — the **weight-code** TMA
  descriptor can now request L2 promotion;
- the TMA route floor for `N34816K5120` drops from `tokens >= 1024` to
  `tokens >= 256` (still `% 256 == 0`);
- the MMA schedule ladder below it is collapsed (`T32R64` deleted, thresholds
  re-cut at 32/64/128);
- `uses_a4` becomes unconditional for that shape (was `max_tokens >= 5`).

ignis has **neither** piece:

- `kernel/vendor/src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh:45` still passes
  `CU_TENSOR_MAP_L2_PROMOTION_NONE` as a literal;
- `kernel/vendor/src/ops/linear/nvfp4/nvfp4_w4a4.cu:57` still reads
  `if (tokens >= 1024 && (tokens % kTmaBlockM) == 0)`;
- the A16→W4A4 floor lives in `nvfp4_dispatch.cpp`'s `resolve_route`, as
  `MlpGateUp: tokens >= 5`.

Upstream moved the route selection into per-shape translation units
(`src/ops/linear/nvfp4/shapes/nXXXXX_kYYYY.cu`, commit `c491cd15`) that ignis
does not vendor; ignis keeps the same decisions in `nvfp4_dispatch.cpp` and
`nvfp4_w4a4.cu`. The kernel-side half (the L2-promotion parameter) is a
verbatim transposition; the route half is a threshold edit in ignis's own
dispatch.

### Candidate 2 — RMSNorm weight prefetch

`9954867a perf(rmsnorm): read the weight before the reduction, and compile that
out where it costs occupancy` (2026-08-31). The kernel makes two sequential
trips to memory per row; neither `weight` nor `z` depends on the reduction, so
the second trip is issued inside the first. Upstream states **bitwise identical
output** — the reduction loop is untouched and the epilogue reads the same
pairs out of registers. The hoist costs registers, so it is a template
parameter with no default and only the gated epilogue consults the grid.

`kernel/vendor/src/ops/kernel/rmsnorm.cuh` contains no `prefetch` — ignis does
not have it. RMSNorm runs twice per layer over 64 layers, in both prefill and
decode.

### Candidate 3 — build-time TU split

`ba5cafef build(attention): parallelize prompt dtype and nvfp4 geometry
compilation` (branch `feat/build-speed`): keep BF16 and INT8 instantiations in
separate translation units, split the non-RDC NVFP4 prompt route by H24/H16
geometry. No kernel, no launch math, no numerics. Pure nvcc parallelism.

### Candidate 4 — W4A4 activation-scale plane written tiled

`1d8587bc perf(nvfp4): read W4A4 activation scales one tile per TMA request`
(2026-09-15), the head of upstream `master`. The activation scale plane is
row-major by token, so one pipeline stage delivers 4 KiB of scales as `BlockM`
separate 16-byte transactions while the codes beside it travel as one 64-byte
box per row. Upstream writes the plane in
`[kNvfp4TmaBlockM tokens, kNvfp4ScaleTileGroups groups]` tiles when the GEMM
that follows takes the TMA route, row-major otherwise, and `select_a4` returns
a route *plus the layout it reads* so a shape cannot pair them the other way
round.

It touches exactly ignis's shapes (`n14336_k5120`, `n16384_k5120`,
`n34816_k5120`, `n5120_k17408`, `n5120_k6144`) plus attn_input, gdn_input,
linear_add and linear_swiglu — 22 files, +294/-122. Both ends are inside
ignis's vendored tree: the quantizer is
`launch_nvfp4_w4a4_quantize` (`nvfp4_w4a4_plan.h`) and the consumers are the
vendored TMA/MMA kernels. Upstream's own warning is the cost: a row-major plane
read as tiled is **wrong numbers, not a failure**.

### Not applicable, and why

| upstream work | why it does not apply |
|---|---|
| `perf(ops): tune q4 …` (4 commits), q5/q6/q8/w8 shape work, fp8 consolidation | ignis serves an NVFP4 artifact; those routes are never launched |
| `32d3c28e` PDL publication | ignis's ops are already PDL-chained and [Decode round host idle](2026-09-18-decode-round-host-idle.md) refuted the PDL-degradation hypothesis on this card |
| `32d3c28e` qk_norm_rope fusion | ignis already has it — `kernel/vendor/src/ops/launcher/qk_norm_rope.{cu,h}` |
| `feat/hyperquant`, `feat/1m-context` | ignis already runs `hq-e8-2b` at `--max-context 262144` |
| `feat/mtp7`, `feat/dflash2` | ignis already runs `--spec dflash2 --draft-tokens 7` |
| `feat/webui` (stock llama.cpp WebUI in-process) | ignis has the Playground |
| `d4929686 perf(runtime): improve materialization search` | ninfer's C++ engine; ignis's Rust runtime has its own prompt reuse (#183) and retained slots (#207) |
| `4b0eb36c perf(ops): route h24 verify attention through small t` | see below — already covered for ignis's default |

The last one deserves its own line because it looks like a direct hit and is
not. Upstream added, for `q_heads == 24 && width <= 8 && max_visible_keys >
320`, a `ChunkedSmallT` route — ignis's exact geometry and exact verify width.
But upstream's `kSmallTChunkTokens` is a fixed 6, while ignis's vendored
resolver already takes the chunk from the cache dtype:
`gqa_small_t_chunk_tokens` returns **8 for U8, 6 otherwise**
(`kernel/vendor/src/ops/launcher/gqa_attention_decode.cu:118`). Under
`hq-e8-2b` a width-8 round satisfies `width <= chunk` and takes `SmallT`
directly. The upstream rule would only change behaviour under a **BF16 KV
cache**, where width 7–8 still falls through to `Prompt`.

### The blocker for any wholesale sync

Upstream migrated to **v3 artifacts** across `168fdd81` (converter),
`4cde7ad0` (C++ weight loading — **101 vendored files**), `04350ba9` (engine),
and `469f014c` (v2 users are pointed at an offline upgrade tool). ignis serves
`qwen3_8_27b_nvfp4full-v2.ninfer`. Any pin bump past those commits changes
`src/core/weight.h`, `src/core/tensor.h` and every `*_plan.cpp`/`*_plan.h` in
the vendored tree at once, and requires a v3 artifact to load.

## Finding

**Observed.** The pin is 263 upstream commits behind on `master`; 86 of them
touch vendored files; 292 of 410 vendored files differ in content, 95 of which
are fork-only. ignis serves an NVFP4-full **v2** artifact with `hq-e8-2b` KV,
1024-token prefill chunks and a 24/4/256 GQA geometry. ignis's vendored tree
lacks the L2-promotion parameter, lacks the 256-token W4A4 TMA floor, lacks the
RMSNorm weight prefetch, and already has qk_norm_rope, DFlash2, HyperQuant and
the dtype-dependent small-T chunk.

**Inferred.** Four candidates are worth an experiment, in this order:

1. **RMSNorm weight prefetch** — smallest diff, upstream claims bitwise
   identical output, touches one vendored `.cuh` plus its launcher, and runs
   128 times per token in both phases. Cheapest thing to falsify.
2. **L2 promotion on the weight-code TMA descriptor** — a descriptor flag
   threaded through a template parameter, no numerics, no layout. Independent
   of the route change and testable alone.
3. **W4A4 TMA floor 1024 → 256 for `N34816K5120`** — ignis's chunk is exactly
   1024, so full chunks already take TMA and only *sub-chunk* widths change:
   the tail of every prompt, every prompt shorter than 1024, and the extra
   traversal boundaries that prompt reuse creates
   ([prompt-reuse TTFT tax](2026-09-18-prompt-reuse-tax-on-short-ttft.md)).
   That is the regime where
   [Vision TTFT live/live](2026-09-16-vision-ttft-live-live.md) records a 1.85x
   short-prompt text gap against the reference — the same regime, which makes
   it worth measuring, not evidence that it is the cause.
4. **Tiled activation-scale plane** — the largest of the four and the only one
   with a silent-wrong-numbers failure mode, since the layout and the route
   must agree. Only after 1–3, and only with the route/layout pairing carried
   over as one object the way upstream did it.

The build-time TU split is orthogonal to all of them and costs nothing to take.

**Inferred.** A pin bump is the wrong instrument here. The v3 migration makes
it an artifact-format change, not a kernel refresh, and 95 fork-only files mean
the fork would have to rebase first. Each candidate above is a manifest-recorded
patch under ADR 0010, or an edit to ignis's own dispatch where the decision has
already moved out of the vendored tree.

## Implications

- The per-shape route decisions ignis keeps in `nvfp4_dispatch.cpp` and
  `nvfp4_w4a4.cu` have drifted from upstream's, which now lives in per-shape
  translation units. Any future comparison has to read both halves; matching
  file names no longer imply matching decisions.
- Upstream's tuning targets are ignis's shapes because both serve the same
  27B checkpoint. That makes upstream's `perf(ops)` stream a standing source of
  candidates for ignis's NVFP4 routes — and only those; everything quantization-
  specific to q4/q5/q8 is noise for as long as ignis serves nvfp4-full.
- The v2/v3 split is a fork-wide fact, not a kernel detail. It bounds how long
  the vendored subtree can keep tracking upstream at all.

## Limits and unknowns

- **Nothing here was measured.** No candidate has a number on this card. The
  ranking is by diff size, failure mode and route relevance, not by observed
  gain.
- The `gpillon` remote could not be fetched in this session (ssh key alias
  `github_gpillon` unavailable), so `gpillon/coding` is read at its local ref,
  `a00648cb`. If the fork has moved since, the conflict surface is stale.
- Candidate 3's link to the 1.85x short-prompt gap is a regime coincidence.
  The reference in that measurement also runs a 1024-token TMA floor, so the
  floor cannot by itself explain the gap.
- The decode round is at its bandwidth bound
  ([anatomy](2026-09-18-decode-round-anatomy.md): 1,166 nodes, 15.81 ms, 4.8%
  idle), so none of these candidates should be expected to move decode much.
  Candidates 1, 3 and 4 are prefill-side; candidate 2 touches both.
- Upstream's own numbers were not reproduced. `docs/maintainer/linear-tuning.md`
  and `docs/maintainer/examples/q4-linear.md` upstream document the method for
  the q4 series, which does not apply here, and the nvfp4 commits carry no
  measurements in their messages.

## Follow-ups

- One ticket per candidate, each an A/B on an exclusive card with the
  per-layer method from
  [GQA workspace memset](2026-09-18-gqa-workspace-memset.md) — an in-run
  control beside the changed layer, since a whole-request delta at this size
  will not resolve.
- A separate decision on the v2→v3 artifact question, which is an ADR, not a
  finding.
