# What upstream ninfer's last month of perf work offers ignis: six prefill candidates with low-single-digit ceilings, and one KV-format axis worth more than all of them

- Kind: research
- Status: current
- Observed: 2026-09-18
- Last verified: 2026-09-18
- Scope: kernel / vendored subtree currency, NVFP4 linear and SwiGLU routes, GDN convolution output, KV formats, artifact container v2/v3, build time
- Related: [ADR 0010](../adr/0010-vendored-reference-kernels.md),
  [Decode round anatomy](2026-09-18-decode-round-anatomy.md),
  [Decode round host idle](2026-09-18-decode-round-host-idle.md),
  [Prefill chunk wall time](2026-09-11-prefill-chunk-wall-time.md),
  [Prompt-reuse tax on short TTFT](2026-09-18-prompt-reuse-tax-on-short-ttft.md),
  [Vision TTFT live/live](2026-09-16-vision-ttft-live-live.md),
  [hq vs BF16 live/live](2026-09-13-hq-vs-bf16-live-live.md),
  [hq-e8-2b KV capacity](2026-09-11-hq-e8-2b-kv-capacity.md),
  [hq attention route agreement](2026-09-12-hq-attention-route-agreement.md),
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
and nothing else does**: every prompt shorter than 1024, every chunk tail,
every extra traversal boundary that prompt reuse creates
([prompt-reuse TTFT tax](2026-09-18-prompt-reuse-tax-on-short-ttft.md)) — and
every chunk *wider* than 1024 — materializes a 34816 x T BF16 tensor in the
arena and reads it back, once per layer, on all 64 layers (both `gqa_layer.cu`
and `gdn_layer.cu` call `linear_swiglu` with `mlp_gate_up`).

**Bound the prize before spending a ticket on it.** The extra traffic is
`64 layers x 4 x 34816 x T` bytes: 9.1 GB at T = 1024, 6.8 GB at T = 768. At
this card's ~1.79 TB/s that is ~5.1 ms and ~3.8 ms, against a measured 99.7 ms
for a 1024-token chunk
([prefill chunk wall time](2026-09-11-prefill-chunk-wall-time.md)). So the
**ceiling is ~5% of prefill**, and only if none of it overlaps.

**And our own chunk-width sweep argues it down further.** In that finding's
Measurement 1, only width **1024** takes `TmaFusedW4A4`; 256, 512, 2048, 4096
and 8192 all take `LinearW4A4Post`. The per-token costs are 0.1607, 0.1150,
**0.0974**, 0.0947, 0.0936, 0.0965 ms — a smooth decline with a plateau and
**no dip at 1024**. If the fused route were worth anything near its ceiling,
1024 would sit below the 512 → 2048 trend. It does not. Either the fused route
is worth little on this card, or the width effect masks it; the sweep cannot
separate those, and neither can this survey.

The 65% per-token penalty at width 256 is real, but it is not this: 256 and
2048 are on the *same* route. The prefill-chunk finding already attributed it
to GEMM fill and launch amortization, which points at candidate 4's
narrow-width MMA ladder, not at the SwiGLU registration.

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

**On decode the prize is already bounded, and it is small.** The anatomy
finding measures every memcpy node in the round at 0.08 ms of 15.81 ms device
time and 0.32 µs of submission each, and states outright that removing **every**
memcpy node — "the residual ping-pong copies and the GDN QKV splits both" — is
worth about **0.5%**. So candidate 2's decode value is ≤0.5%, and its case has
to be made on prefill, where the copies scale with T: `2 x 10240 x T x 2` bytes
per GDN layer over 48 layers is 2.0 GB at T = 1024, ~1.1 ms of a 99.7 ms chunk,
**~1%** — plus whatever the conv kernel itself gains, which upstream measured
at 1.5x–3.5x on T = 1..72 but did not publish for chunk widths.

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

**Ceiling on decode:** the anatomy finding puts `rmsnorm_cta_bf16x2_kernel` at
0.43 ms and `rmsnorm_warp_bf16x2_kernel` at 0.09 ms, so all RMSNorm is 0.52 ms
of a 15.81 ms round — **3.3%**. The change removes one of the kernel's two
memory trips, not the kernel, so its decode ceiling is a fraction of that 3.3%.
Its prefill share is unmeasured.

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

### The axis this survey did not cover: KV formats, where the measured number is

Everything above is kernel tuning, and
[anatomy](2026-09-18-decode-round-anatomy.md) has already established that
decode has nowhere to go — `nvfp4_w4a4_mma_kernel` measures 9.37 ms against a
9.4 ms bandwidth bound, and total idle is 4.8%. So every candidate here is a
prefill candidate with a low-single-digit ceiling.

The one large number ignis has measured on decode is the **KV format**:
[hq vs BF16 live/live](2026-09-13-hq-vs-bf16-live-live.md) puts hq-e8-2b at
**+14.5% of ITL p95** on ignis and +14.3% on the reference — the format's cost,
paid for the 7.11x capacity that
[hq-e8-2b KV capacity](2026-09-11-hq-e8-2b-kv-capacity.md) measures (9,216
bytes per sequence-token against BF16's 65,536).

ignis offers exactly two: `KvPlaneDtype` is `Bf16 | U8`
(`crates/core/src/kv_format.rs:37`). Upstream spent the same month adding
formats between those two poles, none of which ignis has:

| upstream | what it adds |
|---|---|
| `21a0e85f feat(kv-cache): use fp16 V storage and PV compute` | half-precision V plane and PV accumulation |
| `4ac73c47 feat(kv-cache): add nvfp4 and k8v4 modes` | NVFP4 KV, and an 8-bit-K / 4-bit-V split |
| `6183c9be feat(attention): add fp8 kv cache support` | FP8 KV plane |
| `17a7275f feat(attention): add int8 kv hadamard rotation` | a rotation that makes INT8 KV hold its accuracy |

A format that keeps most of hq's capacity while recovering part of that 14.5%
would be worth more on decode than all six kernel candidates together, whose
ceilings sum to low single digits of **prefill**. This survey did not assess
them: they land in `src/ops/softmax_attention/` and the `gqa_attention_*`
kernel family, ignis's route selection for the cache dtype is its own
(`kernel/src/gqa_layer.cu`, `kernel/include/ignis_gqa_workspace.h`), and each
format is a numerics question — a route-agreement campaign like
[hq attention route agreement](2026-09-12-hq-attention-route-agreement.md) —
not a transposition.

Two other attention items were checked and set aside: `9f61a0ca`/`47f9d121`
(2048 sliding-window attention and its tuning) do not apply, because ignis's
GQA layers run full attention with no window; and `a7818988 perf(ops): qualify
variable-width causal cache attention` lives entirely in upstream's renamed
`src/ops/softmax_attention/` tree, so it is a successor to ignis's vendored
`gqa_attention` family rather than a patch to it.

### The v3 artifact migration is a separate axis, not a gate on any of this

Upstream migrated to **v3 artifacts** across `168fdd81` (converter),
`4cde7ad0` (C++ weight loading), `04350ba9` (engine), and `469f014c` (v2 users
are pointed at an offline upgrade tool). ignis serves
`qwen3_8_27b_nvfp4full-v2.ninfer`. The first reading of this survey treated
that as a blocker on the vendored subtree; measuring it shows it is not.

**The vendored kernel surface is nearly v3-indifferent.** `4cde7ad0` touches
101 vendored files, but **97 of them change 4 lines or fewer** — +232/-76
across all 101, against +6,647/-4,158 for the commit as a whole. The `Weight`
struct did not change: every field (`qdata`, `qhigh`, `scales`, `n`, `k`,
`group`, `layout`, `scale_dtype`, `scale_ne`, `scale_nb`,
`weight_scale_divisor`, `input_scale_divisor`, `payload`, `padded_shape`, …)
moved **byte-identical** from `src/core/tensor.h` to a new `src/core/weight.h`,
and the vendored files' churn is the include that follows it. The only
non-trivial vendored diffs are `weight.h` (+45/-18), `tensor.h` (0/-43) and two
op tests.

v3's actual weight lands in `src/artifact/{reader,schema,materializer,binder}.cpp`,
`src/targets/*/impl/load/bindings.cpp`, `src/models/qwen3_5/config.cpp` and
`src/core/weight_view.cpp` — **none of which ignis vendors**, because ignis
replaces them with its own Rust: `crates/artifact/src/{lib,binder,binding,
materializer,normalize,inventory}.rs`, which reads the container directly
(`MAGIC = NINFER\x00\x02`, `crates/artifact/src/lib.rs:146`).

**No candidate in this survey needs v3.** Four of the six (`00369f63`,
`1c8f8acc`, `92bb06eb`, `9954867a`) predate the v3 commits entirely. Of the two
that postdate them, `abbeea0a` touches only pure-kernel files that include no
`core/weight.h`, and `1d8587bc`'s `nvfp4_w4a4_plan.h` picks up that header as a
one-line include.

**What going v3 would actually cost ignis**, read from the container schema and
the upgrade tool rather than from the commit titles:

- the magic goes `NINFER\0\2` → `NINFER\0\3`, the prefix grows from
  `<8sQ` to `<8sQ16s`, and an artifact over `LIMIT = 32_000_000_000` bytes is
  **split into parts** (`NINPRT\0\3` for every part after the first) — ignis's
  19.4 GB file stays single-part, but the reader has to admit the form;
- the directory root goes from `{identity: {model_id, weights_id}, objects}` to
  `{components: {text|vision|mtp|dflash2: {config, resources, target}}, objects,
  bindings, …}`: the **model config moves into the container**;
- an object's `name` becomes `id` with the **string preserved**
  (`"id": value["name"]` in the tool), so `text/token_embedding` stays
  `text/token_embedding`;
- `format` and `layout` strings are remapped to snake_case
  (`W8G32_F16S` → `q8_g32_fp16`, `row-split-k128-v1` → `row_split_k128_v1`);
  every format and layout ignis's artifact uses is in the tool's tables;
- the substantive addition is **`bindings`**: logical parameter names mapped to
  an object, or to a row range of one (`{"parts": [{"object": id, "range":
  [begin, end]}]}`). The fused-parent splitting that `crates/artifact`'s binder
  does in Rust today becomes data the container declares.

**And the official upgrade tool refuses ignis's artifact as it stands.**
`tools/upgrade_ninfer_v2_to_v3.py` gates on
`KNOWN_COUNTS[(model_id, weights_id)]`, which has
`("qwen3.8-27b", "nvfp4"): (1124, 1190)` and no `nvfp4full` key at all; ignis's
file is `("qwen3.8-27b", "nvfp4full")` with **1,325 objects** (text 908, vision
333, dflash2 66, mtp 12, frontend 6). The tool's structure anticipates all of
those components — it builds `vision`, `mtp` and `dflash2` entries and detects
`dflash2/` objects — so the gap is the identity/count guard plus checking its
hand-written binding table against this object set. That is fork-side work on
the `nvfp4full` recipe, not a flag.

**The real obstacle to a wholesale pin bump is unrelated to v3**: 95 vendored
files exist only on the fork side and 78 more are touched by both sides. That
is a fork-rebase problem, and it would still be there on a v3 tree.

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

**Inferred.** The six kernel candidates are all **prefill** candidates with
low-single-digit ceilings, and none of them moves decode: the decode round is
at its bandwidth bound (backbone 9.37 ms against a 9.4 ms bound, 4.8% total
idle), so there is nothing there for a kernel change to take. Ranked by what
each can be worth against what it costs to try:

1. **Build-time TU split** (`ba5cafef`). Zero numerics, zero route change, and
   it pays back in the dev loop rather than in the product. Take it first
   precisely because nothing has to be proved about it.
2. **34816x5120 W4A4 TMA: L2 promotion, Stages 2 -> 3, floor 1024 -> 256**
   (`abbeea0a`). Three separable changes on the kernel that is 59% of decode
   device time and the bulk of prefill. Unlike the others its ceiling is not
   bounded by an existing measurement, because it is not a byte-count
   argument — it changes how the same bytes are fetched and staged. The
   narrow-width MMA ladder is also the only candidate aimed at the one large
   prefill number ignis has: **+65% per token at width 256** against the
   plateau.
3. **RMSNorm weight prefetch** (`9954867a`). Ceiling on decode is a fraction of
   RMSNorm's 3.3% share; prefill share unmeasured. Smallest diff of the six and
   bitwise identical by upstream's claim, so it is the cheapest to falsify.
4. **GDN conv split output** (`92bb06eb`). Decode value is capped at 0.5% by the
   anatomy finding's own accounting of every memcpy node; prefill ~1% of bytes
   plus an unquantified conv-kernel gain. Its real attraction is hygiene: exact
   geometry match, six vendored files, zero conflict with the fork, and it
   deletes a workspace plane.
5. **Fused SwiGLU registered for every multiple of 256** (`00369f63`, with
   `1c8f8acc`). Ceiling ~5% of prefill by arithmetic, and **our own chunk-width
   sweep shows no step at 1024**, the only width that takes the fused route.
   Demoted from first place on that evidence. Do the 1023/1024/1025 probe below
   before opening a ticket.
6. **Tiled activation-scale plane** (`1d8587bc`). The largest diff and the only
   one with a silent-wrong-numbers failure mode, since the layout and the route
   must agree. Last, and only with the route/layout pairing carried over as one
   object the way upstream did it.

**Inferred.** Six is what could be tied to a route ignis launches, not an audit
of upstream's month. About 20 of the 86 vendored-file-touching commits were
read and about 10 diffs opened; the rest were classified by message and file
list or not reached. The `2026-09-06` DFlash2 op batch in particular — roughly
thirty `perf(ops)` commits qualifying and tuning variable-width draft and
target kernels — was set aside as DFlash2-shaped without checking which of
those kernels ignis's own verify round launches.

**Inferred.** The largest measured number on the table is not in this survey.
Decode is at its bound and hq-e8-2b costs **14.5% of ITL p95** against BF16.
That is a **format** question, and upstream spent the same month adding four KV
formats between ignis's two. Ranked against that, the six kernel candidates are
prefill hygiene. If the goal is a number a user would notice, the KV-format
axis is where to look first — at the cost of a numerics campaign rather than a
transposition.

**Inferred.** A pin bump is the wrong instrument here, and v3 is not the reason
why: 95 fork-only files and 78 two-sided ones mean the fork would have to
rebase first, which is true on a v2 tree and a v3 tree alike. Each candidate
above is a manifest-recorded patch under ADR 0010, or an edit to ignis's own
dispatch where the decision has already moved out of the vendored tree.

**Inferred.** Going v3 and taking upstream's perf work are **independent
axes**. v3 buys ignis the ability to consume upstream's new official artifacts
and model cards, and it moves the fused-parameter splitting out of
`crates/artifact`'s binder into container data. It buys **nothing** on the six
candidates, all of which are takeable on v2 today. Its cost is a
`crates/artifact` reader change (new magic and prefix, multi-part form,
`components` root carrying the config, `bindings`, snake_case format/layout
names) plus a fork-side extension of `upgrade_ninfer_v2_to_v3.py` to accept the
`nvfp4full` identity and its 1,325 objects. That makes it its own decision on
its own schedule — an ADR, not a prerequisite.

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
- The v2/v3 split runs through the **loader**, not the kernels, and ignis
  already owns its loader in Rust. That is why the vendored subtree can keep
  tracking upstream's kernel work while the container stays v2 — and why a v3
  migration, when it happens, is scoped to `crates/artifact` and the artifact
  file rather than to `kernel/vendor/`.
- ignis's artifact is a **fork weights variant** (`nvfp4full`, 1,325 objects)
  that upstream's tooling does not know. Anything upstream ships that keys off
  `(model_id, weights_id)` will need a fork-side entry; the upgrade tool is the
  first instance, and it will not be the last.

## Limits and unknowns

- **Nothing here was measured on this card.** The only numbers are upstream's
  own commit messages and ignis's existing findings. The ranking is by
  mechanism, diff size, failure mode and route relevance, not by observed gain.
- The `gpillon` remote could not be fetched in this session (ssh key alias
  `github_gpillon` unavailable), so `gpillon/coding` is read at its local ref,
  `a00648cb`. If the fork has moved since, the conflict surface is stale.
- Candidate 1's link to the 1.85x short-prompt gap is a regime coincidence, not
  an attribution: the reference engine in that measurement is built from the
  same pin and carries the same `tokens == kPrimaryT` resolver. And the
  chunk-width sweep it is read against was not designed as a route A/B — width
  and route move together in it, so "no dip at 1024" bounds the fused route's
  value without isolating it.
- The ceilings quoted for candidates 1 and 2 assume the extra traffic is
  serialized against the useful traffic at the card's peak bandwidth. Prefill
  at width 1024 is not bandwidth-bound (99.7 ms for a backbone that streams in
  ~9 ms), so real overlap will make both smaller, not larger.
- Coverage: about 20 of the 86 vendored-file-touching commits had their message
  read and about 10 their diff. The `2026-09-06` DFlash2 `perf(ops)` batch was
  classified by title alone; ignis runs a DFlash2 drafter and a width-8 verify
  round, so some of it may apply.
- Candidate 4's floor change reaches only widths 256/512/768; the L2-promotion
  and Stages changes were read in the diff but their effect on this card is
  unknown, and upstream published no number for them.
- The decode round is at its bandwidth bound
  ([anatomy](2026-09-18-decode-round-anatomy.md): 1,166 nodes, 15.81 ms, 4.8%
  idle), so only candidate 2 (node count) and candidate 3 (both phases) should
  be expected to touch decode at all. Candidates 1, 4 and 6 are prefill-side.
- `02be37cb` and `e51b585c` were classified from their commit messages and file
  lists, not from reading their diffs against ignis's call sites.
- The v3 container was read from `docs/maintainer/examples/artifact-v3-text.json`
  and from `tools/upgrade_ninfer_v2_to_v3.py`, against ignis's real v2 header
  and `crates/artifact/src/lib.rs`. Nothing was built, converted or loaded, so
  the reader-side cost is a scoped list of schema differences, not an
  implementation estimate. The 1,043-line `artifact-container.md` was not read
  and may carry constraints these two sources do not show.
- `KNOWN_COUNTS` was read as the tool stands on `origin/master`. Whether the
  official artifacts it does accept have since grown the vision and dflash2
  components ignis's file carries was not checked, so "fork-side entry" may
  understate or overstate what the binding table needs.

## Follow-ups

- **Run the 1023 / 1024 / 1025 probe before any SwiGLU ticket.** Three spans of
  near-identical token work where only the middle one takes `TmaFusedW4A4`;
  everything else is held constant, so the route flip is the only variable.
  It needs no code change — `IGNIS_DECOMP_WIDTHS` on the existing
  `crates/core/tests/chunk_decomposition_gpu.rs` harness — and it settles
  whether candidate 1 is worth opening at all, which the chunk-width sweep
  cannot because width and route move together there.
- One ticket per remaining candidate, each an A/B on an exclusive card with the
  per-layer method from
  [GQA workspace memset](2026-09-18-gqa-workspace-memset.md) — an in-run
  control beside the changed layer, since a whole-request delta at these
  ceilings will not resolve. Candidate 4's control is the GDN layers (it
  changes only the NVFP4 linear route); candidate 2's is the GQA layers.
- **Scope the KV-format axis**, which this survey did not: which of upstream's
  four new formats fits ignis's 24/4/256 geometry and paged pool, what each
  costs per sequence-token against hq's 9,216 bytes, and what route-agreement
  campaign each would need. That is where the measured 14.5% lives.
- An audit of the remaining vendored route resolvers for widths ignis serves
  but no route registers, which is what candidates 1 and 2 both turned out to
  be.
- A separate decision on the v2→v3 artifact question, which is an ADR, not a
  finding, and which does not block any of the above. Its two open costs are
  the `crates/artifact` reader change and a fork-side entry in
  `upgrade_ninfer_v2_to_v3.py` for `("qwen3.8-27b", "nvfp4full")`; the trigger
  for taking it is wanting an upstream-published artifact, not wanting upstream
  kernel work.
