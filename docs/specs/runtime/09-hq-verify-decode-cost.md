# runtime 09 — hq-e8-2b verify attention: stop paying the codec's decode for the whole history every round

GitHub: #268

## Problem Statement

Coding agents run at long context: 20K–100K tokens of repository, tool output and
history per lane, with 1 main agent and up to 8–10 subagents. At those contexts
hq-e8-2b verify attention becomes the part of a round that grows. The 2026-09-24
measurement (`docs/findings/2026-09-24-hq-attention-at-long-context.md`):

- It costs **0.09 ms per 1K context tokens per lane per verify round**, linear in
  lanes × context: throughput-bound, not latency-bound.
- It is 33% of a one-lane round at 59K, ~40% of an 8-lane round at 30K, and ~63% of a
  4-lane round at 125K.
- Per GQA layer per 1K keys, the kernel takes 6.2 µs against:
  - 0.96 µs of MMA work at the BF16 rate;
  - 0.33 µs to read hq's 590 KB.

  About 85% is the codec's group decode: Rice symbols, E8 coset, dither hash and scale,
  for every K and V element of the whole history, **redone every round**. It is shared
  only across the 6 query heads and 8 rows of one KV head.
- A narrower MMA (INT8/FP8) would save under 10% of it.

Every other part of the round is at or near its roofline (decode-round anatomy
finding). This is the largest remaining per-round cost at agent contexts.

## Solution

Make the verify round's hq attention cost track bytes and MMA, not ALU decode. The
work is ordered so the cheap, exact options are exhausted before any layout change:

1. **Profile first.** Nsight Compute on the verify kernel at 32K–64K confirms the
   split: issue-slot mix, the pipes that saturate, the stall reasons. It names the
   decode's hot loop before anything is rewritten.
2. **A faster exact decode.** Rewrite the per-tile group decode for throughput:
   warp-parallel Rice prefix decoding, table-driven symbol tails, vectorized
   dither/scale. It must produce bit-identical rows, so nothing downstream changes.
   This is the vendored kernel being the bottleneck: an ADR 0031 recorded patch.
3. **Only if (2) leaves attention decode-bound at ≥2× its MMA floor: a decoded tier.**
   Keep an already-decoded copy of the hot, older part of a lane's history in a short
   FP8/BF16 tier, reused across rounds, so each round decodes only what changed since
   the last one. This costs VRAM against KV capacity (the lanes × context budget of
   ADR 0022 / ADR 0030), so the tier's size is an explicit, planned VRAM line, and it
   needs an owner decision on the trade.

The target: at 8 lanes × 30K, verify attention time per round at least **halved**,
with bit-identical output for option 2. That is ~+20% aggregate tok/s in that cell.

## User Stories

1. As a coding-agent user with a long context, I want each decode round not to slow down linearly with my history, so that a 60K-token session feels like a 10K one.
2. As a user running 8 subagents over the same repository, I want aggregate throughput to hold up as their contexts grow, so that fan-outs stay fast.
3. As a user, I want the generated text unchanged by the faster decode, so that speed never costs correctness.
4. As the project owner, I want the decode-vs-MMA split confirmed by an instruction-level profile before code changes, so that the work targets the measured bottleneck.
5. As the project owner, I want the exact (bit-identical) decode option tried before any option that spends VRAM, so that KV capacity is not traded away for a gain a kernel rewrite could have given.
6. As the project owner, I want a decoded tier, if proposed, to come with its VRAM cost in lanes × context, so that I can decide the trade with numbers.
7. As a maintainer, I want the faster decode checked against the existing per-thread and group decoders on the real KV fixture, so that a single wrong bit is caught.
8. As a maintainer, I want the hq verify route's agreement tests to keep passing, so that the attention output is still the codec's.
9. As a maintainer, I want the patch recorded under ADR 0031 with its reason, so that the vendored-kernel audit trail stays complete.
10. As a maintainer, I want the prompt (prefill) route to benefit where it shares the decode, so that long cold prefills also get cheaper.
11. As an operator, I want no new flag for the exact option, so that it is simply the engine.
12. As an operator, I want any decoded tier sized in the VRAM plan's log line, so that I can see what it took from the KV pool.
13. As the project owner, I want the result measured live/live at 1, 4 and 8 lanes across 8K–125K, so that the improvement is known per cell.
14. As a maintainer, I want the BF16 verify route's own wrong-kernel issue kept separate, so that this spec does not grow into a second fix (BF16 KV runs the prefill kernel at T = 8; noted in the finding).

## Implementation Decisions

- **Scope of the kernel change.** The hq source path of the small-T split-KV verify
  kernel (`gqa_attention_small_t_tc_partial` with the hq KV policy), plus the shared
  group decoder it calls.
  - The decoder is shared with the prompt route's scratch decode, which already moved
    its symbols on chip (2026-09-24 port, bit-exact).
  - The kernel's contracts stay: tiles, splits, neutral partials, the fused append,
    masks, the rotated-frame MMA and the output un-rotation.
- **Exactness bar for option 2:** every decoded element is bit-identical to today's
  group decode on the 8,192-row real KV fixture. Greedy decode texts are identical
  on the quick-wins prompt set at 1 and 8 lanes.
- **Decision gate for option 3.** After option 2, re-measure. Option 3 is proposed
  only if hq attention is still ≥ 2× its MMA floor at 32K. It comes as its own design
  note: tier size, eviction rule, VRAM plan line, and the capacity lost at 8 lanes.
  The owner decides before it is built.
- **Profiling:** Nsight Compute (`ncu`) on the server, with a kernel filter and a
  launch skip past warm-up. `nsys profile --trace=cuda` for the round split. On this
  machine `nsys launch` hangs: use `profile` with absolute paths and stdin from
  `/dev/null`.
- **Not in this spec:** MMA precision changes (INT8/FP8 QK/PV). The finding bounds them
  under 10%.

## Testing Decisions

- A good test compares observable outputs (decoded rows, attention outputs, generated
  tokens) against the existing reference paths, never intermediate registers.
- **Kernel CTest:** extend `kernel/tests/test_hq_codec_kv_rows.cu`, which already
  decodes the real fixture three ways and requires 0 differing elements (prior art: the
  2026-09-24 on-chip-symbol check). The new decoder is a fourth way, same bar.
- **Route agreement:** `kernel/tests/test_hq_route_agreement.cu` stays green (hq verify
  vs its reference).
- **GPU profile:** `scripts/gpu-profile.ps1` 0 failed, including the DFlash2 round and
  hq ring tests (`crates/core/tests/dflash2_round_gpu.rs`).
- **Performance** is a finding, not a test:
  - the `.scratch/longctx-2026-09-24` driver (warm-prefill, then synchronized decode);
  - round time from `itl × samples / rounds`;
  - 1/4/8 lanes × 8K/30K/59K/125K;
  - before/after live/live per ADR 0021;
  - plus nsys attention-per-round at 1 lane × 59K.

## Out of Scope

- Changing the KV format, or adding FP8 KV as a new format (the KLD finding makes a
  case for it; that is a separate owner decision).
- The BF16 verify route using the prefill kernel.
- INT8/FP8 MMA for QK/PV.
- Prefill-route work beyond what the shared decoder gives for free.

## Further Notes

- Evidence:
  - `docs/findings/2026-09-24-hq-attention-at-long-context.md` (slopes, nsys split,
    roofline);
  - `docs/findings/2026-09-24-kld-against-bf16.md` (hq's quality cost also grows with
    context);
  - `docs/findings/2026-09-24-upstream-quick-wins-ab.md` (the prompt scratch decode
    port and its bit-exact check).
- The E8 dither is per (row, word) and pseudo-random. Any decode rewrite must
  reproduce `hq_dither` exactly: the seeds come from `hq_dither_row_seed` /
  `hq_dither_word_seed`.
