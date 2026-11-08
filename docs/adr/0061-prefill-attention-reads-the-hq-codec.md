---
status: accepted
---

# ADR 0061 — prefill attention reads the hq-e8-2b codec: no prompt planes, on every card

## Status

Accepted (2026-10-21, after the owner's review: "accetto tutto"). Proposed the same day. Owner decision in the 16 GB grill ("niente area mi piace; questo ce lo porteremo come vantaggio
anche per la 5090"):
- the prompt planes go;
- on every card class;
- with numerical, not bit-exact, equivalence;
- the attention readout moves to the same path.

Amended 2026-10-21 (owner, during #335's implementation): the history is decoded **once per chunk and
layer into a fixed 8-bit band**, not per CTA tile (point 1 below and "As built").

Related:
- [ADR 0022](0022-two-kv-formats-bf16-as-oracle.md): BF16 KV is the oracle;
- [ADR 0030](0030-device-memory-reserved-at-load.md): device memory is reserved at load;
- [ADR 0031](0031-vendored-kernel-bottleneck-exemption.md) and [ADR 0051](0051-kernel-code-is-ours-adapted-or-vendored.md):
  vendored, adapted and own kernel code;
- [ADR 0060](0060-the-27b-has-a-trellis-body-for-small-cards.md): the 16 GB body.

Plan: `.scratch/27b-16gb-study-2026-10-20/PIANO-VRAM.md` §9 (untracked).

## Context

Under hq-e8-2b the vendored prompt route (`gqa_attention`, route `Prompt`) works in two steps:
1. before a prefill chunk attends, it decodes the layer's whole visible history into BF16 **prompt planes**,
   `[256, 4 KV heads, keys]` for K and for V;
2. it runs an FA2 kernel over them.

The planes cost 4,096 bytes per context token, up to one band of 262,144 keys, so 1.07 GB at 256K or more. They are
prefill-only:
- decode and verify rounds read the codec in place (the small-T routes);
- the planes die with the layer's scope.

Prefill interleaves with decode at chunk level (ADR 0018), and every device byte is carved at load (ADR 0030). So the
arena holds them for the whole load, on every card.

On a 16 GB card that 1.07 GB decides whether 256K fits. With the 3.0 bpw body and two lanes, the pool KV left beside
one full 256K sequence would be:
- **+82K** with the planes;
- **+169K** with a 64K band;
- **+199K** with no planes.

On the 5090 the same gigabyte is ~116K more KV tokens.

The options costed:

| option | bytes freed at 256K | why not |
|---|---:|---|
| planes in host memory | — | a chunk rereads the history in all 16 layers: ~16 GB per chunk at 256K over PCIe |
| lend them to the KV pool between prefills | — | pages live long; taking them back means evicting a sequence, and it breaks ADR 0030 |
| a smaller band (the vendored constant `kGqaHqPromptScratchBandKeys`, 262,144 → 65,536) | 0.81 | the banded path exists and is tested, but its prefill cost is unmeasured, and the attention readout needs a single band |
| one arena for prefill scratch and round buffers | 0.1-0.2 | small; decode graphs need fixed addresses |

## Decision

1. **A GQA prefill chunk under hq-e8-2b attends straight from the paged codec.**
   - An attention kernel of our own (an **adapted op**, ADR 0051) decodes each K/V tile into shared memory once per
     CTA, as the decode route decodes per key, and runs the chunk's 64-128 queries against it.
   - No prompt planes are materialized.
   - The workspace no longer grows with `max_context`.
2. **Every card class: the 27B's GQA layers, whichever artifact they run.** The 32 GB NVFP4 load gets the same bytes
   back. Flash-Next is not concerned: its QSA layers have their own prefill (dense up to 2,051 keys, sparse past it)
   and never call the vendored prompt route.
3. **Equivalence is numerical, not bit-exact** (owner). The decoded values are the same; the reduction order is not.
   The acceptance criteria below are the agent's proposal:
   - **route agreement** against ignis's BF16 route, at least as close as today's prompt route (`test_hq_route_agreement`);
   - **KLD unchanged** within noise on the 2026-09-24 suite (serving 0.0513);
   - **cold prefill TTFT** on the 5090 at 8K, 46K and 105K **no slower** beyond the run's noise. Performance-first: the
     owner has not set this one.
4. **The attention readout** (`/v1/decide`, `locate`, #260) reads from the same path. It loses its single-band
   restriction.

5. **It computes in FP8, with FP16 accumulate as the fallback** (owner, 2026-10-21, amending this ADR the day it was
   accepted).
   - Q·Kᵀ and P·V run on the block-scaled FP8 MMAs, which SM120 runs at full rate with FP32 accumulate. GeForce
     halves BF16 HMMA with FP32 accumulate, and that halved peak is where the 27B's prefill attention runs today.
   - FP16 accumulate (softmax statistics in FP32, a fold per key block) is the fallback if FP8 fails the agreement and
     KLD criteria.
   - Estimate: 60K cold prefill ~6.6K → ~9.6-10.1K tok/s (FP8 at ~0.65 ns per key), ~8.0K (FP16 accumulate).
   - Reaching ~12.5K at 60K also needs the fixed part of the prefill down to ~66 µs per token: phase 2 (spec
     27b-16gb/phase2-notes, spec prefill-5090/01).

## Considered options

- **Keep the planes and shrink the band to 64K.** It is the fallback if the gate in point 3 fails: a recorded patch to
  a vendored constant (ADR 0037/0031), once its prefill cost is measured.
- **Bit-exact with the vendored route.** Rejected: it would force the vendored kernel's reduction order onto the new
  one and cancel the design.

## Consequences

- The vendored prompt route stays in the tree as the reference the agreement test compares against, but serving never
  calls it. Its kernels are no longer the prefill bottleneck to optimize.
- The VRAM plan's workspace line drops the 4,096 bytes per context token.
- The 5090's profile is re-measured ("raise until refused", as on 2026-10-10) once this and ADR 0062 land.

**As built** (2026-10-21, GitHub #335), where the decision left a choice:

- **One decode per chunk and layer, not one per CTA tile** (owner, amending point 1). A CTA holds at most 128 query
  rows, since its FP32 accumulators fill the registers. A decode per tile would then decode each K/V row ~48 times a
  chunk: ~1.7 ns per key and token at the decode's 0.29 ns a row, more than the whole attention. The op
  (`kernel/src/hq_prompt_attention.cu`, adapted from mkyt/qwen3-27b-5090's `attention_fa_kernel`, `kernel/NOTICE`)
  runs two passes sharing one online softmax:
  - the **residual window** (the 32 sinks, the ring before the chunk, the chunk itself) in BF16 on the BF16 MMAs, its
    rows exact as before, ADR 0037's ring-before-append included;
  - every other key decoded once into a fixed **band** of 32,768 keys (64 MiB), walked band after band.

  A launch splits its keys up to 5 ways when the chunk's query tiles alone would leave SMs idle; a merge kernel
  combines the splits and un-rotates.
- **The band is 8-bit, not E4M3 throughout.** Q·Kᵀ runs on the INT8 MMA, exact in int32: Q with a scale per row, K
  with one per 32-key tile, the scheme SageAttention uses and `kernel/src/image/attn8.cu` runs. P·V stays on the
  block-scaled E4M3 MMA, V's power-of-two tile scale riding the MMA's block scale. Upstream's E4M3 Q rounds each query
  by ~2.5%, the same for every key it scores: a bias, which the route agreement test measured at 32K keys as ×1.06 of
  A1's median distance from BF16. INT8 Q brings that to ×1.005. Two probes that did not help are not kept: P encoded
  against each tile's own max, and a two-term E4M3 Q (as close as INT8, at twice the time).
- **The FP16-accumulate fallback ships behind `--model-prefill-attention fp16`** (config v2 `model.prefill_attention`,
  the 27B's only, reload required; a Flash-Next start refuses an explicit value). Its band holds FP16 at half the
  keys in the same bytes, on the FP16 MMAs with FP16 accumulate inside a tile, folded into FP32.
- **One key fetch, three readers.** `kernel/src/hq_prompt_keys.cuh` reads a key wherever it lives (side plane, codec,
  the chunk's fresh rows) for the attention, the attention readout and the tap. The readout reads spans past one band
  (a test arm at 300,000 keys).
- **Measured** ([finding](../findings/2026-10-21-prefill-attention-without-prompt-planes.md)):
  - the attention, one 1,024-token chunk over 16 layers: 0.915 ns per key and token at 46K and 0.857 at 105K, against
    A1's 2.147 and 2.098 (the FP16 fallback 1.464 and 1.496);
  - route agreement: median ×0.986-1.005 of A1's distance from BF16 on every arm (FP16: ×0.998-1.001), which the test
    now holds to ×1.01;
  - KLD on the 2026-09-24 suite: 0.05221 against A1's 0.05134, within noise (the difference's CI [−0.00068,
    +0.00301]); the E4M3-Q kernel read 0.05123, the gap between the two carried by one window's tail;
  - cold prefill against main, one session: +7.2% at 8K, +9.4% at 14K, +29.7% at 46K, +53.1% at 105K (the host
    embedding of ADR 0062 included); decode +0.1% at one lane and +1.5% at eight, the time per round 2-4% lower;
  - VRAM at 524,288 keys: the attention's workspace −1,020,474,240 bytes (1,164,194,688 → 143,720,448), and a
    text-only load's workspace line as much. With vision on, the line is now the item-bounded vision encoder's
    (908,592,896 bytes), so it drops by 599,718,656.
- **The 5090's profile spends it on context**: the 27B's `model.max_context` 786,432 → 917,504, raised until the plan
  refused (950,272 is 59 MB short), which moves `auto` rope scaling from YaRN 3 to 3.5.
- **Owner, 2026-10-21, closing #335**: the 8-bit band stays the default, its agreement held to ×1.01 of A1 (acceptance
  2 read with that tolerance), and the profile keeps 917,504.
