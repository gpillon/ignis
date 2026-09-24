# The hq-e8-2b verify attention was latency-bound on registers, not on its Rice walk: freeing the q fragments' registers nearly halves it bit-exactly, and it is still ~4x its MMA floor

- Kind: experiment
- Status: current
- Observed: 2026-09-24
- Last verified: 2026-09-25
- Scope: kernel / hq-e8-2b verify-round attention (`gqa_attention_small_t_tc_partial_bf16_kernel` with `GqaTcKVHq`), the 8-lane group decode (`hq_codec.cuh`)
- Related: GitHub #268, [spec runtime/09](../specs/runtime/09-hq-verify-decode-cost.md),
  [hq attention at long context](2026-09-24-hq-attention-at-long-context.md) (the baseline),
  [ADR 0031](../adr/0031-vendored-kernel-bottleneck-exemption.md),
  [hq-e8-2b KV capacity](2026-09-11-hq-e8-2b-kv-capacity.md),
  [Batched decode width drift](2026-09-14-batched-decode-width-drift.md)
- Superseded by: none

## Question

The long-context finding inferred from a roofline that ~85% of the hq verify
attention kernel is the codec's group decode, redone every round. Where exactly
does the time go at the instruction level, how much does a bit-exact rewrite
buy, and is attention still far enough above its MMA floor after it to justify
the spec's option 3 (a decoded tier that costs VRAM)?

## Evidence

**Setup.** RTX 5090, exclusive card (swarm GPU lock), Windows. Baseline main
`81fdc37`; the change is branch `hq-verify-decode-268` (`b568e41`, final
`359d3b4`).

- **Kernel alone:** `ignis_hq_verify_bench` (`kernel/tests/bench_hq_verify_attention.cu`).
  It launches the verify kernel exactly as the verify round does: grid
  4 x 85 x lanes, 128 threads, width 8, one table row per lane, and a window of
  `--ctx` keys. The code planes are encoded from the committed real K/V rows
  under their true dither seeds. L2 is flushed before each launch; the result is
  the median of 20 launches. The bench also prints an FNV-1a hash of every partial.
  `--decode-only` times the tile decode with no MMA.
- **Profiles:** Nsight Compute 2025.4 `--set full` on the bench (the real
  binary under `target/windows-desktop-win7-x64/`; the `ncu` on PATH is
  npm-check-updates). No counter-permission error on this machine.
- **Live/live:** the two server builds ran back to back in one hold, both with
  `make config` flags plus `--metrics`. Three drivers:
  - `ab.py` from the quick-wins A/B: C=1, C=8, cold prefill.
  - A `longctx.py` with deterministic prompts: one lane per nonce, a fixed body,
    768 greedy tokens, `ignore_eos`. Round time is
    `itl_ms_mean x itl_samples / spec.rounds` per request.
  - `nsys profile --cuda-graph-trace=node` cells at `--max-context 65536`.
- Raw data and drivers are in the worktree's `.scratch/live/`, `.scratch/ncu/`
  and `.scratch/tools/`. Captures were deleted after reduction.

**Nsight Compute, main, 8 lanes x 30K (one launch, 3.2 ms at ncu's clocks):**

| metric | value |
|---|---|
| warp instructions | 1.16e9 (590 per decoded row) |
| issue slots busy | 35.2% |
| occupancy | 8 warps/SM (theoretical 16.7%), limited by 255 registers *and* 37 KB shared memory; the q fragments and accumulators hold 192 registers across the key loop |
| stall mix | fixed-latency dependency 32%, selected 21%, long scoreboard 14%, short scoreboard 12%, math 7% |
| pipes | ALU 27%, LSU 19%, tensor 9%, XU 8%: none saturated |
| unary walk (64-bit clz + shifts, one terminator per step) | **44% of instructions**: 26 per step, ~40 steps per warp and row wave |
| unstrip + dither + scale | 28% |
| global loads and address work | 4% of instructions, 18% of stall samples (a wave waits on its block table, then on its codes) |
| MMA + softmax | ~11% |
| shared stores | 2.8-way bank conflicts |

**What each step bought.** Kernel alone, µs per launch; all steps have the same
partial hashes as main.

| step (cumulative) | 8 x 30K | 1 x 59K |
|---|---:|---:|
| main | 1679 | 368 |
| throughput group decoder in place of `hq_decode_row_group` | 1659 | 365 |
| + code/meta rows staged one tile ahead by cp.async | 1447 | 320 |
| + branch-free terminator chains | 1398 | 312 |
| + split phases: q stays in shared memory, K and V take turns in one tile (no q fragments in registers) | 932 | 211 |
| + dither byte folded into the reference's rounding, packed unstrip | 911 | 205 |
| + warp-converged decode call, paired K/V fragment loads, P fragments once per tile, vector q staging | **890** | **201** |

The tile decode alone (`--decode-only`, the same grid, 8 lanes x 30K):

| decoder | µs |
|---|---:|
| reference | 1084 |
| throughput decoder, branchy chains | 1084 |
| throughput decoder, branch-free chains | 590 |
| + phase B folding | 565 |

Decoding 2 or 4 rows per group at once, which doubles or quadruples the
independent chains, bought nothing (565 / 561 / 616). It bought nothing inside
the kernel either (915 against 911).

**Kernel alone, every cell** (µs; per GQA layer per 1K keys per lane in brackets):

| cell | main | #268 | change |
|---|---:|---:|---:|
| 1 x 7K | 90.0 (12.86) | 55.0 (7.86) | -39% |
| 1 x 30K | 259.0 (8.63) | 141.3 (4.71) | -45% |
| 1 x 59K | 367.6 (6.34) | 200.6 (3.46) | -45% |
| 1 x 126K | 753.4 (6.13) | 399.4 (3.25) | -47% |
| 4 x 30K | 954.4 (7.95) | 505.6 (4.21) | -47% |
| 8 x 14K | 819.2 (7.31) | 456.9 (4.08) | -44% |
| 8 x 30K | 1679.4 (7.00) | 890.0 (3.71) | -47% |

The same binary also runs both routes (`--reference` launches the reference
route, which is main's code path). Every cell has the same hash for both routes
and is faster on the throughput route, with the residual window off and on:

| cell | residual off | residual on |
|---|---:|---:|
| 1 x 512 | 53.0 -> 34.6 | 34.5 -> 32.5 |
| 1 x 4K | 61.2 -> 38.7 | 61.3 -> 38.6 |
| 8 x 1K | 132.6 -> 93.9 | 122.9 -> 98.0 |
| 8 x 4K | 319.3 -> 200.4 | 293.6 -> 198.2 |
| 4 x 8K | 299.7 -> 173.8 | 247.5 -> 169.5 |
| 8 x 30K | 1673.0 -> 882.2 | 1493.8 -> 868.1 |

**nsys, the server, decode only** (after the last prefill GEMM; the round count
is the verify kernel launches / 16, the same in both arms):

| cell | verify kernel per launch | attention per round | round (kernel span / rounds) |
|---|---:|---:|---:|
| 1 lane, 63.8K prompt, 64 rounds | 405.7 -> 229.0 µs (-43.6%) | 7.77 -> 4.99 ms (-35.8%) | 30.5 -> 28.5 ms |
| 8 lanes, 33.5K prompts, 127 rounds | 1073.8 -> 559.4 µs (-47.9%) | 20.02 -> 11.03 ms (**-44.9%**) | 51.0 -> 40.0 ms (-21.6%) |

"Attention" is the verify kernel plus the split reducer, which is unchanged
(0.11 ms per launch at 8 lanes).

**Live/live round time** (ms; prompts are the actual token counts; greedy, 768
tokens; texts compared lane by lane):

| cell | main | #268 | change | identical texts |
|---|---:|---:|---:|---|
| 1 x 7.2K | 16.70 | 16.20 | -3.0% | 1/1 |
| 1 x 33.5K | 19.45 | 17.81 | -8.4% | 1/1 |
| 1 x 63.8K | 21.74 | 19.02 | -12.5% | 1/1 |
| 1 x 141.7K | 29.05 | 23.30 | **-19.8%** | 1/1 |
| 4 x 7.2K | 24.73 | 23.64 | -4.4% | 4/4 |
| 4 x 33.5K | 33.97 | 29.09 | -14.4% | 2/4 |
| 4 x 63.8K | 46.92 | 35.26 | -24.8% | 1/4 |
| 4 x 141.7K | 74.91 | 52.91 | **-29.4%** | 2/4 |
| 8 x 7.2K | 38.45 | 37.08 | -3.6% | 6/8 |
| 8 x 14.4K | 44.11 | 38.58 | -12.5% | 6/8 |
| 8 x 33.5K | 52.37 | 43.46 | **-17.0%** (1.205x rounds/s) | 6/8 |
| 8 x 63.8K | 90.73 | 78.12 | (re-prefill inside the window in both arms, TTFT 26 s: not a decode cell) | 2/8 |

**Text identity.**
- C=1 (`ab.py`, eight coding prompts, 512 tokens): 8 of 8 identical.
- Every one-lane long-context cell: identical.
- Multi-lane cells differ in some lanes. A control ran each build twice, C=8
  and then 4 x 30K:

  | pair | C=8 | 4 x 30K |
  |---|---|---|
  | main vs main | 16/16 | **2/4** |
  | #268 vs #268 | 16/16 | 4/4 |
  | main vs #268 | 16/16 | 2/4 |

  So at several lanes main does not reproduce its own text either; the
  differences are within main's own run-to-run variation, not caused by #268.
- The exactness proof is the kernel oracle below.

**Exactness.**
- `ignis_hq_codec_kv_rows_test`, the fourth decode way, 0 elements differing
  across four corpora:
  - all 8,192 real rows: 2,097,152 elements, against the group decode (swizzled,
    symbols in the row) and against the per-thread decode;
  - 64 escalated heavy-tailed rows;
  - 144 rows the encoder never writes. These cover a zeroed meta, k > 7, a
    general k, a non-zero tail, < 256 and > 256 terminators, a lone last-bit
    terminator, no terminator at all, and the terminal fallback.
- `ignis_kernel_hq_verify_exact_test` (new CTest) runs the kernel's throughput
  route against its reference route (`IgnisThroughput = false`, the whole
  reference hq route) over 8 shapes:
  - widths 1-8, masked columns, a column offset, permuted table rows;
  - 40 to 70,000 keys;
  - the residual window with cleared ring slots;
  - the fused append.

  Every partial byte and every cache byte is identical.
- A one-bit parity mutation in the decoder fails both tests.
- Kernel CTest 66/66; `ignis_kernel_hq_route_agreement_test` green.
- `scripts/gpu-profile.ps1 -SkipKernelBuild` (DFlash2 round and hq ring tests
  included): 93 passed, 0 failed. `cargo test --workspace`: 1,742 passed.

**Prefill.** Cold prefill, main against #268 with the throughput decoder in the
prompt scratch decode:

| prompt | main | #268 |
|---:|---:|---:|
| 14K | 1.83 / 1.76 s | 1.82 / 1.77 s |
| 46K | 7.05 / 7.15 s | 7.12 / 7.14 s |
| 105K | 23.16 s | 23.06 s |

The scratch kernel moved -9% in an 8-lane capture and +2.5% in a 1-lane one.
Following ADR 0031, the prompt route keeps the reference decode (`359d3b4`).

## Finding

**Observed.**
- The kernel ran at 35% issue with 2 warps per scheduler.
- The Rice walk was the largest instruction block (44%), but not the binding
  constraint. Removing 27% of the decode's instructions (a branch-free walk
  with 32-bit chains) changed the fused kernel by ~1%. The decode alone, with
  more registers, got 1.84x faster from the same code.
- What moved the fused kernel was register relief and latency hiding:
  - cp.async staging of the next tile's code rows: -14%;
  - keeping q in shared memory so the decode has the ~64 registers the q
    fragments held (split K/V phases): -33% on top of that.
- The throughput route is bit-identical to the reference route. Verify
  attention per round is -36% at one lane x 64K and -45% at 8 lanes x 33K, and
  the kernel itself is -44% and -48%.
- In live rounds: -17% at 8 x 33K, i.e. +20.5% rounds per second at unchanged
  acceptance, which is the spec's ~+20% aggregate. The gain grows with
  context: -20% at 1 x 142K, -29% at 4 x 142K. It is -3 to -4% at 7K.

**Against the spec's targets.**
- "Attention per round at least halved at 8 x 30K" is **not quite met**:
  -44.9% measured by nsys, -47.9% for the kernel alone.
- The ~+20% tok/s it was meant to buy is met: 1.205x rounds per second at
  8 x 33.5K.

**Inference: the gate for option 3 triggers.**
- After option 2 the kernel costs 3.71 µs per GQA layer per 1K keys per lane at
  8 x 30K and 4.71 µs at 1 x 30K (kernel alone). That is **3.9x and 4.9x** the
  0.96 µs MMA floor, above the spec's 2x threshold.
- The decode is still ~60% of the kernel: the decode alone is 565 of 890 µs.
- At 8 warps per SM the remaining time is latency, not a saturated pipe
  (41% issue, no pipe above 31%).

## Implications

- Beyond this, the lever is occupancy, and the accumulators set it: 128
  registers per thread for 16 rows x 256 dims per warp. Two structural options:
  - split the output dimensions across warps (half the accumulators, twice the
    warps);
  - warp specialization (decode warps without accumulators).

  Both are a different kernel rather than a patch, and the QK accumulation
  order must stay the reference's to remain bit-exact (a split-K QK is not).
- The prompt route's scratch decode gains nothing from the decoder alone: it is
  not register-starved, so the instruction savings are hidden there.
- **Option 3 design note (owner decision; not built).** A decoded tier keeps
  the history's decoded rows across rounds, so each round decodes only rows
  that are not yet tiered.
  - *What it must hold to stay exact.* The decode's own BF16 output, in the
    rotated frame: 512 B per row, **65,536 B per token** over the 16 GQA layers,
    4 KV heads and 2 roles. That is the same as the BF16 KV format and 7.1x
    hq-e8-2b's 9,216 B.
  - *Cheaper exact variant.* An int8 lattice-coordinate tier (256 B per row,
    **32,768 B/token**, 3.6x hq, with a fallback for |y| > 127) skips only the
    Rice walk and unstrip, ~40% of the decode; dither and scale still run every
    round.
  - *Not exact.* An FP8 E4M3 tier is the same size but changes the numerics:
    it is FP8 KV under another name, out of scope here.
  - *Eviction.* Rows are write-once: rollback only touches the newest rows,
    which the residual ring already serves exact. So the tier holds each
    lane's oldest rows, from the sink up to `window - 512`, and a lane over
    budget keeps its oldest `T` rows. A lane's tier is released with its slot.
  - *VRAM plan line.* `hq_decoded_tier = tier_tokens_total x 65,536 B` (or
    `x 32,768 B`), charged against the KV pool (ADR 0022 / ADR 0030 capacity
    math).
  - *Capacity at 8 lanes.* A full BF16 tier at 8 x 30K is **15.7 GB**, which
    does not fit next to the model. The KV pool at `make config` is ~735K
    tokens = ~6.8 GB. A 2 GB BF16 tier covers 4K tokens per lane (13% of a 30K
    history) and costs ~230K pool tokens (31% of the pool) for ~-8% attention.
    At one lane x 64K, a full 4.2 GB tier removes the decode (the verify
    kernel ~-60%) but costs ~460K pool tokens (63%).
  - *Recommendation.* Not worth building for the 8-subagent coding case. It
    could pay only for one very long lane with spare pool, which the owner
    would have to value above KV capacity.

## Limits and unknowns

- One live/live run per cell. Single cells carry a few percent of spread; the
  8 x 63.8K cell is not a decode measurement (both arms re-prefilled a lane
  inside the window).
- Multi-lane long-context text identity is not a usable check on this engine:
  main is not self-reproducing there.
- The nsys per-round figures at 8 lanes average over rounds with fewer live
  lanes as requests finish.
- The kernel-alone bench runs without the residual window (every key a codec
  row); the server runs with it, which is one reason its per-key costs are
  lower at 8 lanes.
- 2- and 4-row interleaving and the int8/FP8 variants were not explored beyond
  what is stated.
- Occupancy-changing designs (output-split or warp-specialized kernels) were
  not built.

## Follow-ups

- The owner decides option 3 from the note above. The recommendation is not to
  build it.
- If attention at 8 lanes needs another step, the next measured candidate is a
  kernel with more warps per SM (output-split accumulators or decode warps
  without them), under ADR 0031 option (b) with this kernel as the oracle.
