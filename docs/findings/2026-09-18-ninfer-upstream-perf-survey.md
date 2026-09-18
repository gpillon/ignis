# What upstream ninfer's last month of perf work offers ignis: six portable candidates, one blocker, and a large re-vendor bill

- Kind: research
- Status: current
- Observed: 2026-09-18
- Last verified: 2026-09-18
- Scope: kernel / vendored subtree currency, NVFP4 linear and SwiGLU routes, GDN convolution output, GQA attention routes, build time
- Related: [ADR 0010](../adr/0010-vendored-reference-kernels.md),
  [Decode round anatomy](2026-09-18-decode-round-anatomy.md),
  [Decode round host idle](2026-09-18-decode-round-host-idle.md),
  [Prefill chunk wall time](2026-09-11-prefill-chunk-wall-time.md),
  [Prompt-reuse tax on short TTFT](2026-09-18-prompt-reuse-tax-on-short-ttft.md),
  [Vision TTFT live/live](2026-09-16-vision-ttft-live-live.md),
  [GQA workspace memset](2026-09-18-gqa-workspace-memset.md),
  [hq prompt workspace under-report](2026-09-12-hq-prompt-workspace-under-report.md)
- Superseded by: none

## Question

Upstream `cometkim/ninfer` shipped a month of performance commits after the
point ignis vendored from. Which of them apply to what ignis actually runs,
and what would it cost to take each one?

This is a **survey, not a measurement**. Nothing below was run on the card.
Every "gain" is a hypothesis with a named route and a named shape; the only
numbers quoted are ignis's own already-measured findings or upstream's own
commit messages, used to bound what a candidate could be worth.

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

Reproduce by running `git ls-tree -r` on both revisions and intersecting with
the manifest's `path` entries.

**A wholesale pin bump is therefore not a small operation** — 292 of 410 files
move, and 95 of them (the DFlash2 ops, the Windows port) exist only on the fork
side. Per-change transposition is the realistic route.

### What ignis actually runs, which decides applicability

- artifact: `ARTIFACT ?= ./models/qwen3_8_27b_nvfp4full-v2.ninfer` (`mk/config.mk`)
  — **NVFP4 weights, v2 artifact**;
- KV format: `KV_FORMAT ?= hq-e8-2b`;
- prefill chunk: `DEFAULT_SERVING_CHUNK_TOKENS = 1024` (`crates/core/src/concrete.rs:237`);
- GQA geometry: 24 query heads over 4 KV heads of 256
  (`kernel/include/ignis_gqa_workspace.h:28`), 16 GQA layers;
- GDN geometry: `conv_channels = 10240`, `qk_width = 2048`, `value_width = 6144`
  (`kernel/src/gdn_layer.cu:96-97`), 48 GDN layers;
- linear shapes launched: hidden 5120, q 6144, gdn 16384, mlp gate_up 34816
  (ffn 17408), 14336 — i.e. `N34816K5120`, `N5120K17408`, `N5120K6144`,
  `N14336K5120`, `N16384K5120`;
- W4A4 is live: `kernel/src/linear.cu`, `gqa_layer.cu` and `linear_swiglu.cu`
  all delegate the A4 decision to the vendored plan, and the vendored
  `nvfp4_w4a4_tma.cu` compiles.

This kills a whole class of upstream work on sight. The `perf(ops): tune q4
1024x5120 / 4096x5120 / 6144x5120` series, the q5/q6/q8/w8 shape work and the
fp8 route consolidation act on artifacts ignis does not serve.

### Candidate 1 — the fused NVFP4 SwiGLU is registered at one width only

`00369f63 perf(nvfp4): register the fused TMA SwiGLU for every width its block
accepts` (2026-08-29). Upstream's own description: the route was registered at
exactly `T == 1024` although the kernel accepts any T that is a multiple of its
256-token block; every other width fell into `LinearW4A4Post`, which "runs the
gate/up linear into a 34816 x T bf16 tensor in the arena and then reads that
tensor back through a separate silu_mul". The fix selects the fused kernel for
every multiple of 256 from 256 up.

ignis has the pre-fix code verbatim
(`kernel/vendor/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp`):

```
constexpr std::int32_t kPrimaryT = 1024;
...
if (tokens == 1) { return Nvfp4LinearSwiGluRoute::DecodeFusedA16; }
if (tokens <= 48) { return Nvfp4LinearSwiGluRoute::FusedW4A4; }
if (tokens == kPrimaryT) { return Nvfp4LinearSwiGluRoute::TmaFusedW4A4; }
return Nvfp4LinearSwiGluRoute::LinearW4A4Post;
```

ignis's serving chunk is exactly 1024, so **full chunks take the fused route
and nothing else does**: every prompt shorter than 1024, every chunk tail, and
every extra traversal boundary that prompt reuse creates
([prompt-reuse TTFT tax](2026-09-18-prompt-reuse-tax-on-short-ttft.md))
materializes a 34816 x T BF16 tensor in the arena and reads it back, once per
layer. At T = 768 that is ~53 MB written and ~53 MB read per layer.

`1c8f8acc perf(ops): fuse nvfp4 swiglu through t96` (2026-09-05) raises the
`FusedW4A4` ceiling from 48 to 96 on the same resolver and belongs with it.

Upstream's warning is the implementation constraint: `resolve_route` and
`nvfp4_linear_swiglu_workspace_capacity_bytes` must move **together**, or the
shipped Op test reports an exact workspace query/execution high-water mismatch.
ignis has both functions in that same file, so the guardrail is already there.

### Candidate 2 — the GDN convolution can write straight into q/k/v

`92bb06eb feat(ops): write the gdn prefill convolution straight into q/k/v`
(2026-08-27). The chunked prefill convolved into one packed `[C,T]` buffer and
then pulled the three channel ranges out with three `cudaMemcpy2DAsync` per
layer per chunk; the convolution already knows which channel each thread owns,
so it can address the destination directly. The new
`causal_conv1d_silu_split` takes an output address map — three destinations
partitioned by row, the row counts as template parameters — and the packed
`[C,T]` plane leaves the target's workspace recipe entirely.

Upstream measured the route selection: across T = 1..72 the small-T kernel
holds a flat 4.1 µs and the prefill pair a flat 8.2 µs, against 12.3 µs at
T = 17 and 28.7 µs at T = 64 for the sequence kernel both entries previously
selected — 1.5x to 3.5x on that interval, which the packed entry gets too.

The supported row profiles are `(2048, 2048, 4096)` over `C = 8192` and
**`(2048, 2048, 6144)` over `C = 10240`** — the second is ignis's geometry
exactly. And ignis still does the thing this removes, in both paths:

- `kernel/src/gdn_layer.cu:175` — `copy_channel_range` x3 after
  `causal_conv1d_silu`, eager path;
- `kernel/src/gdn_layer.cu:371` — the same three copies after
  `causal_conv1d_silu_snapshot`, **inside the decode graph**.

48 GDN layers x 3 copies is 144 memcpy nodes per decode round, which is where
[Decode round host idle](2026-09-18-decode-round-host-idle.md) counted them,
and [anatomy](2026-09-18-decode-round-anatomy.md) established that
`cudaGraphLaunch`'s host duration is linear in node count.

All six of the commit's kernel-surface files are vendored by ignis
(`include/ninfer/ops/causal_conv1d_silu.h`, `src/ops/kernel/causal_conv1d.cuh`,
`src/ops/launcher/causal_conv1d.{cu,h}`,
`src/ops/wrapper/causal_conv1d_silu.cpp`, plus the op test) and **none of them
is touched on the `gpillon` side** — a zero-conflict transposition. The
commit's remaining four files are upstream's own bench and engine, which ignis
does not vendor.

### Candidate 3 — RMSNorm weight prefetch

`9954867a perf(rmsnorm): read the weight before the reduction, and compile that
out where it costs occupancy` (2026-08-31). The kernel makes two sequential
trips to memory per row; neither `weight` nor `z` depends on the reduction, so
the second trip is issued inside the first. Upstream states **bitwise identical
output** — the reduction loop is untouched and the epilogue reads the same
pairs out of registers. The hoist costs registers (the gated warp kernel goes
35 → 50, the gated wide-row kernel 38 → 48, three blocks per SM to two), so it
is a template parameter with no default and only the gated epilogue consults
the grid.

`kernel/vendor/src/ops/kernel/rmsnorm.cuh` contains no `prefetch` — ignis does
not have it. RMSNorm runs twice per layer over 64 layers, in both phases.

### Candidate 4 — the 34816x5120 W4A4 TMA route

`abbeea0a perf(ops): tune nvfp4 34816x5120 linear routes` (2026-09-15) changes
exactly ignis's `mlp_gate_up` shape, in three independent ways:

1. `nvfp4_make_tma_2d` gains a `CUtensorMapL2promotion` parameter, threaded
   from a new `Nvfp4W4a4TmaSchedule` template parameter, and the shape's
   schedule becomes
   `TmaM256N128Prefetch128B = Nvfp4W4a4TmaSchedule<256, 3, 1, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>`
   — L2 promotion on the **weight-code** descriptor, "prefetch the adjacent
   half-line for the next K tile";
2. the same alias change moves the shape from `Stages = 2` to `Stages = 3`
   (it previously used `TmaM256N128S2`);
3. the TMA route floor drops from `tokens >= 1024` to `tokens >= 256`, the MMA
   ladder below it is re-cut at 32/64/128, and `uses_a4` becomes unconditional.

ignis has none of it:

- `kernel/vendor/src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh:45` passes
  `CU_TENSOR_MAP_L2_PROMOTION_NONE` as a literal;
- `kernel/vendor/src/ops/linear/nvfp4/nvfp4_w4a4.cu:57` reads
  `if (tokens >= 1024 && (tokens % kTmaBlockM) == 0)`;
- the A16→W4A4 floor lives in `nvfp4_dispatch.cpp`'s `resolve_route`, as
  `MlpGateUp: tokens >= 5`.

The floor change is **narrower than it looks**: the predicate keeps
`tokens % 256 == 0`, so only widths 256, 512 and 768 move from MMA to TMA. A
700-token tail still takes MMA. Items 1 and 2 apply to every TMA launch of that
shape and are independent of the floor.

Upstream moved route selection into per-shape translation units
(`src/ops/linear/nvfp4/shapes/nXXXXX_kYYYY.cu`, commit `c491cd15`) that ignis
does not vendor; ignis keeps the same decisions in `nvfp4_dispatch.cpp` and
`nvfp4_w4a4.cu`. The schedule/descriptor half is a verbatim transposition; the
route half is a threshold edit in ignis's own dispatch.

### Candidate 5 — build-time TU split

`ba5cafef build(attention): parallelize prompt dtype and nvfp4 geometry
compilation` (branch `feat/build-speed`): keep BF16 and INT8 instantiations in
separate translation units, split the non-RDC NVFP4 prompt route by H24/H16
geometry. No kernel, no launch math, no numerics. Pure nvcc parallelism.

### Candidate 6 — W4A4 activation-scale plane written tiled

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
ignis's vendored tree: the quantizer is `launch_nvfp4_w4a4_quantize`
(`nvfp4_w4a4_plan.h`) and the consumers are the vendored TMA/MMA kernels.
Upstream's own warning is the cost: a row-major plane read as tiled is **wrong
numbers, not a failure**.

### Assessed and set aside

| upstream work | disposition |
|---|---|
| `perf(ops): tune q4 …` (4 commits), q5/q6/q8/w8 shape work, fp8 consolidation | ignis serves an NVFP4 artifact; those routes are never launched |
| `32d3c28e` PDL publication | ignis's ops are already PDL-chained, and [Decode round host idle](2026-09-18-decode-round-host-idle.md) refuted the PDL-degradation hypothesis on this card |
| `32d3c28e` qk_norm_rope fusion | ignis already has it — `kernel/vendor/src/ops/launcher/qk_norm_rope.{cu,h}` |
| `feat/hyperquant`, `feat/1m-context` | ignis already runs `hq-e8-2b` at `--max-context 262144` |
| `feat/mtp7`, `feat/dflash2` | ignis already runs `--spec dflash2 --draft-tokens 7` |
| `feat/webui` (stock llama.cpp WebUI in-process) | ignis has the Playground |
| `d4929686 perf(runtime): improve materialization search` | ninfer's C++ engine; ignis's Rust runtime has its own prompt reuse (#183) and retained slots (#207) |
| `e51b585c fix(gdn): respect cooperative launch capacity` | a correctness guard, not a perf item: sources the SM count from `DeviceContext`, partitions a cooperative grid that exceeds resident capacity, falls back when one tile cannot fit. 10 vendored files, 8 in conflict. Worth taking as robustness if ignis's GDN grid can reach that bound; not ranked as a gain |
| `02be37cb perf(ops): fuse and qualify variable-width gdn norm control` | adds a fused BF16 norm+gating kernel qualified for DFlash2 target widths (W=2..16, B=1..8) — ignis's verify-round shape. 5 vendored files, 4 in conflict, and ignis calls `gdn_gating_proj` with the norm separate. Plausible but unassessed; needs its own read |
| `4b0eb36c perf(ops): route h24 verify attention through small t` | see below — already covered for ignis's default |

The last one deserves its own line because it reads as a direct hit and is not.
Upstream added, for `q_heads == 24 && width <= 8 && max_visible_keys > 320`, a
`ChunkedSmallT` route — ignis's exact geometry and exact verify width. But
upstream's `kSmallTChunkTokens` is a fixed 6, while ignis's vendored resolver
already takes the chunk from the cache dtype: `gqa_small_t_chunk_tokens` returns
**8 for U8, 6 otherwise**
(`kernel/vendor/src/ops/launcher/gqa_attention_decode.cu:118`). Under
`hq-e8-2b` a width-8 round satisfies `width <= chunk` and takes `SmallT`
directly. Upstream `master` carries no HyperQuant at all — it is fork-side — so
its fixed chunk of 6 is why the rule exists there. The upstream rule would only
change behaviour under a **BF16 KV cache**, where width 7–8 still falls through
to `Prompt`.

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
1024-token prefill chunks, a 24/4/256 GQA geometry and a `C = 10240 /
2048+2048+6144` GDN geometry. ignis's vendored tree registers the fused NVFP4
TMA SwiGLU at exactly T = 1024, still splits the GDN convolution output with
three `cudaMemcpy2DAsync` per layer in both the eager and the graph path, lacks
the RMSNorm weight prefetch, lacks the W4A4 weight-code L2 promotion and the
256-token TMA floor — and already has qk_norm_rope, DFlash2, HyperQuant and the
dtype-dependent small-T chunk.

**Inferred.** Six candidates are worth an experiment, in this order:

1. **Fused SwiGLU registered for every multiple of 256** (`00369f63`, with
   `1c8f8acc` behind it). Largest hypothesised effect and the clearest
   mechanism: at any width that is not exactly 1024, ignis pays a 34816 x T
   BF16 write and read-back per layer that the fused route does not. That is
   the short-prompt and tail-chunk regime, which is where
   [Vision TTFT live/live](2026-09-16-vision-ttft-live-live.md) records a 1.85x
   short-prompt text gap against the reference. Same regime is a reason to
   measure, not evidence of cause — the reference build carries the same
   resolver.
2. **GDN conv split output** (`92bb06eb`). Exact geometry match, six vendored
   files, **zero conflict** with the fork, upstream's own 1.5x–3.5x on
   T = 1..72, removes a workspace plane, and removes 144 memcpy nodes per
   decode round on a graph whose launch cost is linear in node count.
3. **RMSNorm weight prefetch** (`9954867a`). Smallest diff, bitwise identical
   output by upstream's claim, 128 launches per token. Cheapest to falsify.
4. **34816x5120 W4A4 TMA: L2 promotion, Stages 2 → 3, floor 1024 → 256**
   (`abbeea0a`). Three separable changes; test the schedule change first, since
   the floor only moves widths 256/512/768.
5. **Build-time TU split** (`ba5cafef`). Orthogonal to all of the above and
   costs nothing to take.
6. **Tiled activation-scale plane** (`1d8587bc`). The largest and the only one
   with a silent-wrong-numbers failure mode, since the layout and the route
   must agree. Only after 1–4, and only with the route/layout pairing carried
   over as one object the way upstream did it.

**Inferred.** A pin bump is the wrong instrument here. The v3 migration makes
it an artifact-format change, not a kernel refresh, and 95 fork-only files mean
the fork would have to rebase first. Each candidate above is a manifest-recorded
patch under ADR 0010, or an edit to ignis's own dispatch where the decision has
already moved out of the vendored tree.

## Implications

- Candidates 1 and 2 are both **route-registration** defects rather than kernel
  work: the fast kernel exists in ignis's tree already and is reached at one
  width, or not reached at all. That is a different class of finding from a
  tuning delta, and it suggests auditing the other vendored resolvers for
  widths ignis actually serves.
- The per-shape route decisions ignis keeps in `nvfp4_dispatch.cpp`,
  `nvfp4_w4a4.cu` and `nvfp4_linear_swiglu_plan.cpp` have drifted from
  upstream's, which now lives in per-shape translation units. Any future
  comparison has to read both halves; matching file names no longer imply
  matching decisions.
- Upstream's tuning targets are ignis's shapes because both serve the same 27B
  checkpoint. That makes upstream's `perf(ops)` stream a standing source of
  candidates for ignis's NVFP4 and GDN routes — and only those; everything
  specific to q4/q5/q8 is noise for as long as ignis serves nvfp4-full.
- The v2/v3 split is a fork-wide fact, not a kernel detail. It bounds how long
  the vendored subtree can keep tracking upstream at all.

## Limits and unknowns

- **Nothing here was measured on this card.** The only numbers are upstream's
  own commit messages and ignis's existing findings. The ranking is by
  mechanism, diff size, failure mode and route relevance, not by observed gain.
- The `gpillon` remote could not be fetched in this session (ssh key alias
  `github_gpillon` unavailable), so `gpillon/coding` is read at its local ref,
  `a00648cb`. If the fork has moved since, the conflict surface is stale.
- Candidate 1's link to the 1.85x short-prompt gap is a regime coincidence, not
  an attribution: the reference engine in that measurement is built from the
  same pin and carries the same `tokens == kPrimaryT` resolver.
- Candidate 4's floor change reaches only widths 256/512/768; the L2-promotion
  and Stages changes were read in the diff but their effect on this card is
  unknown, and upstream published no number for them.
- The decode round is at its bandwidth bound
  ([anatomy](2026-09-18-decode-round-anatomy.md): 1,166 nodes, 15.81 ms, 4.8%
  idle), so only candidate 2 (node count) and candidate 3 (both phases) should
  be expected to touch decode at all. Candidates 1, 4 and 6 are prefill-side.
- `02be37cb` and `e51b585c` were classified from their commit messages and file
  lists, not from reading their diffs against ignis's call sites.

## Follow-ups

- One ticket per candidate, each an A/B on an exclusive card with the
  per-layer method from
  [GQA workspace memset](2026-09-18-gqa-workspace-memset.md) — an in-run
  control beside the changed layer, since a whole-request delta at this size
  will not resolve. Candidate 2's control is the GQA layers; candidate 1's is a
  1024-token chunk against a 768-token one.
- An audit of the remaining vendored route resolvers for widths ignis serves
  but no route registers, which is what candidates 1 and 2 both turned out to
  be.
- A separate decision on the v2→v3 artifact question, which is an ADR, not a
  finding.
