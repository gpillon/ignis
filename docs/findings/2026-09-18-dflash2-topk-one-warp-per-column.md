# The drafter's vendored top-k gave one warp to a column: 3,145 us of every decode round, 71x more than our own

- Kind: experiment
- Status: current
- Observed: 2026-09-18
- Last verified: 2026-09-18
- Scope: kernel / DFlash2 drafter, speculative decode round, vendored op replacement
- Related: [ADR 0005](../adr/0005-performance-first-principle.md),
  [ADR 0010](../adr/0010-vendored-reference-kernels.md),
  [#155](https://github.com/gpillon/ignis/issues/155) (the drafter round),
  [Decode round host idle](2026-09-18-decode-round-host-idle.md),
  [hq vs BF16 live/live](2026-09-13-hq-vs-bf16-live-live.md)
- Superseded by: none

**Hardware:** RTX 5090 (170 SMs), exclusive card (ADR 0006).
**Engine:** production defaults — `hq-e8-2b`, `--max-context 262144`,
`--spec dflash2 --draft-tokens 7`.
**Captures and harness:** `.scratch/decode-idle-2026-09-18/`.

## Question

A decode round at one lane replays a 19.18 ms graph and the device is busy
95% of it, so the time is in kernels rather than in bubbles. Which ones, and
are any of them off their own bound?

## Evidence

An Nsight Systems capture with `--cuda-graph-trace=node`, over 8 s of
steady-state decode, by total device time:

| kernel | share | calls/round | mean |
|---|---|---|---|
| `nvfp4_w4a4_mma_kernel` | 48.4% | 247 | 37.46 us |
| **`dflash2_topk_kernel`** | **16.4%** | **1** | **3,145.23 us** |
| `w8_small_t_mma_kernel` | 8.6% | 2 | 820.77 us |
| `nvfp4_small_t_kernel` | 4.2% | 32 | 24.77 us |
| `gqa_attention_small_t_tc_partial_bf16_kernel` | 4.1% | 16 | 48.34 us |

`dflash2_topk_kernel` (kernel/vendor/src/ops/kernel/dflash2_draft.cuh) selects
the drafter's 16 candidate tokens per draft column: k=16 over a
`[248046, 7]` BF16 logits matrix. Its launch, read off the capture, is **grid
2x1x1, block 128, 20 registers, no shared memory** — the launcher gives one
warp to one column, so seven warps were active on a 170-SM card. Each lane
strides the whole vocabulary keeping a private `TopkEntry list[64]`; at 20
registers per thread that list is in local memory, so every insertion
comparison is a memory access.

Its input is 248,046 x 7 x 2 B = **3.47 MB**, which is **2.0 us** of traffic
at this card's bandwidth. It took 3,145 us — about **1,570x off the memory
bound**.

The same capture is why that number is a defect rather than a fact of life:
`w8_small_t_mma_kernel` is the output head streaming 1.27 GB of INT8 weights,
and at 820.77 us against a 747 us roofline it is at 91% of its bound. Both are
vendored. One had headroom and one did not, and only the profile says which.

### The replacement

`kernel/src/dflash2_topk.cu` (ours; ADR 0010 — no port claim) splits the row
axis over blocks and merges: a partial pass at grid `(splits, columns)` takes
each block's slice down to k, a merge pass takes `splits * k` down to k. Both
keep each thread's running list in registers by making k a compile-time
constant and consuming the list from `list[0]` with a fully unrolled shift —
the vendored kernel's `list[cursor]` is a dynamic index, and a dynamically
indexed array is not a register array.

| | vendored | ours |
|---|---|---|
| launch | grid 2x1, block 128, **20 regs** | grid 122x7 + 7x1, block 128, **71 regs** |
| per round | **3,145.23 us** | **44.26 us** (30.55 partial + 13.71 merge) |
| share of decode kernel time | 16.4% | 0.27% |

**71x.** What it buys, measured the same way at one lane over 8 s:

| | before | after |
|---|---|---|
| graph replay, mean | 19.18 ms | **15.81 ms** (−17.6%) |
| rounds completed in 8 s | 388 | **471** (+21.4%) |
| 256 greedy tokens, request wall | 2,051 ms | **1,721 ms** (−16.1%) |

### That it is the same answer

The vendored op specifies an exact deterministic selection — largest first,
ties to the smaller row id, `values` carrying the logits entries bit-exactly —
so the bound is bit equality, not a tolerance.
`kernel/tests/test_dflash2_topk.cu` runs both implementations over the same
input on 18 shapes: the two production shapes, a tie-heavy column where
sixteen distinct values across 248k rows make the row rule decide nearly every
slot, a column carrying both infinities and both zeroes, the row counts either
side of our 2,048-row split seam, one and 64 columns, and two values of k this
implementation forwards to the vendored op. **Every arm: 0 ids differ, 0
values differ.**

End to end, 256 greedy tokens at `temperature 0` came back **byte-identical**
to the baseline binary's (`sha256 0ed25a9a08b5c2a0…`, reasoning channel
included), which is what the bit-exact selection predicts: identical
candidates mean identical drafts, identical drafts mean identical accepts.

## Finding

**Observed.** The drafter's vendored top-k spent 3,145 us of every decode
round doing 2.0 us of memory traffic on 0.07% of the card. Replacing it with a
row-split merge is worth **17.6% of a decode round** and **+21% of decode
throughput**, at a bit-identical answer.

**Inference.** The op was not written for this geometry. One warp per column
is a reasonable shape when columns are many and rows are few; at a 248k
vocabulary and seven columns it inverts, and nothing in the vendored code
adapts. The reference engine runs the same launcher, so it pays the same cost
— which is exactly why "as fast as the reference" cannot be the ceiling.

**Method.** The profile found this in one pass and the same profile cleared
`w8_small_t_mma_kernel` in the same pass. A per-kernel roofline estimate
beside the measured time is what separates a vendored op worth replacing from
one worth leaving alone; neither is visible from reading the code.

## Implications

- The first vendored op this engine replaces rather than calls. The vendored
  file stays in the subtree and under the manifest — it is still the oracle
  the replacement is tested against, and still the implementation for any k
  this one does not specialize.
- The remaining decode profile is 58% `nvfp4_w4a4_mma_kernel` and 10%
  `w8_small_t_mma_kernel`, both at or near their bounds. The next decode lever
  is not a kernel.

## Limits and unknowns

- Measured at one lane. The eight-lane round was not re-profiled after the
  change; the drafter's top-k scales with `k * batch` columns, so the saving
  should grow with the batch, but that is an expectation and not a
  measurement.
- One capture per configuration, no repeats. The effect is 20x the run-to-run
  spread seen on this harness, which is why one capture carries it.
- Ours is still 22x off its own 2.0 us bound. Nobody has looked at why,
  because at 0.27% of the round there is nothing left to win.
- The replacement specializes k=16 only, and the arms that exercise other k
  are exercising the vendored op through our entry point, not ours.

## Follow-ups

- Re-profile at eight lanes to size the saving under the batch the agentic
  load actually runs.
- `nvfp4_w4a4_mma_kernel` at 247 calls and 37.46 us per call is now 58% of
  decode. Whether it is at its bound has not been checked.
