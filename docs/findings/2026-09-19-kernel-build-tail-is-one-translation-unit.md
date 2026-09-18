# The kernel build's last 95 seconds are one translation unit, and 280 of its 313 kernels are for models ignis does not serve

- Kind: experiment
- Status: current
- Observed: 2026-09-19
- Last verified: 2026-09-19
- Scope: kernel / build time, nvcc translation units, vendored subtree policy
- Related: [ADR 0010](../adr/0010-vendored-reference-kernels.md),
  [ADR 0031](../adr/0031-vendored-kernel-bottleneck-exemption.md),
  [ninfer upstream perf survey](2026-09-18-ninfer-upstream-perf-survey.md)
- Superseded by: none

## Question

Upstream's `ba5cafef build(attention): parallelize prompt dtype and nvfp4
geometry compilation` splits heavy attention template instantiations into
separate translation units. Does it apply to ignis, and if not, where does
ignis's own kernel build time actually go?

## Evidence

**Hardware:** the same box as the perf findings — 20 logical cores, CUDA 13.1,
MSVC, Ninja, SM120a. No GPU involvement in compilation.

### `ba5cafef` does not apply

It splits `src/ops/softmax_attention/dense/causal_cache/prompt*.cu` three ways:
BF16 and INT8 into separate units, and the non-RDC NVFP4 prompt route by H24 /
H16 geometry. ignis already has the first
(`gqa_attention_prefill_{bf16,i8}.cu`) and the fork's equivalent of the second
(`gqa_attention_prefill_hq_{27,35}.cu`, split by model rather than by head
count). The NVFP4 half has nothing to act on: ignis's `KvPlaneDtype` is
`Bf16 | U8` (`crates/core/src/kv_format.rs:37`) and no file under
`kernel/vendor/src/ops/kernel/gqa_attention*.cuh` mentions NVFP4.

Upstream's own note on that branch: "This branch does not add Windows or
HyperQuant support and claims no inference speedup."

### Where the time actually goes

`kernel/build/.ninja_log` records start and end for every target, so the
distribution is free to read. Over a full build: **548 targets, 2,547 s of
summed compile wall**, which on 20 cores is a 127 s floor from total work.

The slowest single unit is **`w8_small_t.cu` at 221.6 s** — 1.9x the next one
(`gqa_attention_decode_i8.cu`, 115.7 s). Because 221.6 > 127, **one translation
unit is the binding constraint**: no amount of `-j` gets the build below it,
and for the last ~95 s of every build 19 cores wait on it.

### What is inside it

128 lines. Seven independent geometry families, each a `make_launchers<Geometry,
First>(index_sequence<...>)` array whose entries instantiate
`launch_exact<Geometry, N>` and through it `w8_small_t_mma_kernel`:

| family | token range | instantiations |
|---|---:|---:|
| `W8VocabularyProjectionGeometry` (248320 x 5120) | 1..33 | 33 |
| `W8MtpInputProjectionGeometry` | 1..48 | 48 |
| `W8MtpAttentionProjectionGeometry` | 1..48 | 48 |
| `W8MtpAttentionOutputGeometry` | 1..48 | 48 |
| `W8MtpGateUpProjectionGeometry` | 1..40 | 40 |
| `W8MtpDownProjectionGeometry` | 1..48 | 48 |
| `W835bMtpProjectionGeometry` (2048 x 4096) | 1..48 | 48 |
| **total** | | **313** |

Only the first is reachable here. ignis serves the 27B and speculates with
DFlash2, not MTP; `kernel/src/` never names `Mtp` or `35b`, and
`crates/artifact/src/normalize.rs:190` records that "the 27B text artifact's
`text/*` objects carry only the two W8 endpoints" with mtp out of scope. So
**280 of the 313 instantiations are for geometries this engine cannot reach**,
and they are what the build waits on.

### The split, measured without touching the tree

Probe translation units built from the file's own preamble, one geometry family
each, compiled standalone with the build's exact flags:

| unit | instantiations | seconds |
|---|---:|---:|
| the file as it stands | 313 | **258.9** |
| `W8MtpInputProjectionGeometry` alone | 48 | 42.5 |
| `W8VocabularyProjectionGeometry` alone | 33 | 15.3 |

Roughly linear in instantiation count. A seven-way split would put the file's
critical path at ~43 s instead of ~259 s, and the build's floor back on total
work (127 s) rather than on one unit.

### `--split-compile`, which needs no source change at all

`nvcc --split-compile` runs the device-code optimizer over several threads
within one translation unit. Standalone on the unsplit file, same flags:
**258.9 s -> 143.0 s**.

Three clean builds of the whole kernel, one configuration each:

| configuration | clean build wall |
|---|---:|
| baseline | **327.2 s** |
| `--split-compile=0` on every CUDA source | 276.4 s |
| `--split-compile=0` on the five heaviest sources | **268.8 s** |

The flag is not free. With it on every source the build's summed compile wall
rose from 3,768 s to 4,286 s (+14%): during the wide phase there are no spare
cores for the extra threads, so they only add overhead. The gain is entirely in
the tail, where the cores are idle anyway.

All 61 kernel CTests pass on the `--split-compile` build, including
`hq_route_agreement` and `dflash2_topk` — the relevant check, since the flag
changes how device code is optimized.

## Finding

**Observed.** `ba5cafef` is already had or inapplicable. ignis's kernel build
is bounded by a single translation unit, `w8_small_t.cu`, at 221.6 s against a
127 s floor from total work on 20 cores. That unit instantiates 313 kernels
across seven geometry families, of which only the 33-kernel vocabulary family
is reachable by this engine. Splitting it by family would cut its cost to ~43 s
(measured on two families). `--split-compile` cuts it to 143 s with no source
change, and cuts the whole clean build from 327.2 s to 268.8 s when applied to
the five heaviest units.

**Inferred.** The single-run wall differences are not all equally solid: the
baseline-to-flag gap (327.2 -> ~270 s, ~18%) is far larger than run-to-run
noise, while global-versus-targeted (276.4 vs 268.8, 2.8%) is not separable on
one build each. Targeted is preferred anyway, because it confines a codegen
change to five files instead of about five hundred.

**Inferred.** The structural fix is worth more than the flag and costs a policy
decision. Splitting the seven families into seven units removes ~180 s from the
critical path against the flag's ~80 s, and the two compose. But
`w8_small_t.cu` is vendored. ADR 0031 opens the "do not hand-edit" default only
on a **runtime** measurement — a profile of a real prefill or decode round plus
a bound estimate showing headroom — and it names `w8_small_t_mma` explicitly as
a **non-candidate**, at 91% of its bound. A build-time argument is outside what
0031 authorizes, and a seven-way structural split is the patch shape 0031 warns
"rots far faster" across a reference bump. That is the owner's call, not an
agent's.

## Implications

- Build time is a distribution, not a total. `-j` stops helping the moment one
  unit exceeds total-work-over-cores, and `.ninja_log` shows that for free after
  any build — no instrumentation needed.
- An upstream commit classified from its message can already be had. `ba5cafef`
  read as a clear free win in the
  [upstream survey](2026-09-18-ninfer-upstream-perf-survey.md) and was not one;
  the check that settles it is the file list on both sides, not the title.
- The cheapest version of "which upstream changes apply" is often a property of
  what ignis *cannot reach*: 280 dead instantiations here, the q4/q5/q8 routes
  in the survey, the NVFP4 attention route above.

## Limits and unknowns

- One clean build per configuration, no repeats. The 327 -> 270 s gap is well
  outside plausible noise; the 7.6 s between the two flag placements is not.
- Per-target durations in `.ninja_log` are wall under contention, not isolated
  compile cost, and vary with how ninja happened to schedule. The standalone
  probe numbers (258.9 / 42.5 / 15.3 / 143.0 s) are uncontended and are the
  ones to trust.
- The seven-way split was projected from two families, not built. The other
  five were assumed to scale with instantiation count, which the two measured
  points support but do not prove.
- `--split-compile` may alter generated device code. The 61 kernel CTests pass,
  but **workspace-wide `cargo test` was not run**: this worktree has no `target/`
  and the main checkout's is 39 GB against 47 GB free, so a second one does not
  fit. That gate is outstanding.
- Whether the MTP and 35B instantiations should exist at all here is a separate
  question from splitting them, and a larger one: dropping them would be a much
  bigger win and would change what the binary can run.

## Follow-ups

- Owner decision: does a measured **build-time** bottleneck open ADR 0031's
  exemption, or does it need its own clause? Nothing about `w8_small_t.cu`
  should be edited until that is answered.
- Re-derive `IGNIS_SPLIT_COMPILE_SOURCES` from `.ninja_log` when the heavy
  units move; the list is a measurement, not a constant.
- Run workspace-wide `cargo test` against this change on a checkout that has the
  disk for it.
