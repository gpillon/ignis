# The 27B's 2,048-token prefill chunk is free with vision on, +2% at 46K-105K, and closer to BF16

- Kind: experiment
- Status: current
- Observed: 2026-10-21
- Last verified: 2026-10-21
- Scope: serving / the 27B's prefill chunk width on the RTX 5090 (PCIe Gen 3): VRAM plan, cold prefill, decode, a decoding lane's pauses under interleaving (ADR 0018), 4K KLD; kernel / ignis's prompt attention (ADR 0061) at W = 2,048 / 4,096
- Topic: prefill
- Related: GitHub #341 (spec [prefill-5090/01](../specs/prefill-5090/01-the-prefill-fixed-part.md), lever 2), [A wider prefill chunk buys the 27B at most 4%](2026-10-07-27b-prefill-chunk-width.md) (the same sweep before #335), [The prefill attention without prompt planes](2026-10-21-prefill-attention-without-prompt-planes.md) (the A/B driver, the 917,504 probe, the KLD baseline), [KLD against BF16](2026-09-24-kld-against-bf16.md) (the suite), `.scratch/341/`
- Superseded by: none
- Summary: With vision on, the 27B's 2,048-token prefill chunk costs no VRAM, prefills +1.9%/+2.2% at 46K/105K, lowers 4K KLD (0.0492 vs 0.0522) and lengthens a decoding lane's longest pause 184 to 307 ms; the rtx5090 profile adopts it.

## Question

#335 removed the prompt planes, so the workspace now grows ~400 KB per chunk token instead of the
~1.4 MB it grew on 2026-10-07, and with vision on the arena is sized by the vision encoder, not the
prefill scratch. Does a 2,048-token chunk now pay on the 27B, and what does it cost in VRAM, in decode,
in a decoding lane's pauses while a prompt prefills, and in KLD? What does 4,096 cost in VRAM?

## Evidence

**Setup.** `main` at `bce6863c` (one binary for both arms: the width is a flag). Server under
`make start IDLE_UNLOAD=0 LOOP_GUARD=off PREFILL_CHUNK=<W>`, every other make default: 524,288 context,
hq-e8-2b, DFlash2/7, vision on (the rtx5090 profile's), prompt reuse on. Card exclusive under the GPU
lock. Legs in one session, interleaved 2048, 1024, 2048, 1024 (`.scratch/341/ab341.py`, `chain.py`):
per leg #335's cold-prefill client (8.2K warm-up, then 14K / 46K / 105K twice, one slices.json), the
2026-09-24 decode driver (C=1, C=8), and #92's stall probe (two lanes decode, a 32.7K prompt arrives,
three reps, plus the prompt alone).

**The plan.**
- `vram_plan_gpu` at 262,144 keys: the text scratch is 487,837,184 B at 1,024 and 896,464,896 B at
  2,048 (~399 KB per chunk token). The item-bounded vision encoder sizes the arena at 908,592,896 B,
  12,128,000 B above the 2,048 scratch, so with vision on the arena is the same at both widths. At
  4,096 it is 1,578,309,632 B: +669,716,736.
- The rtx5090 profile's start at 917,504 tokens (`profile_probe.py`, same binary): the plan needs
  29,698,716,928 B at 1,024 and at 2,048, to the byte, and 30,368,433,664 B at 4,096 (+669.7 MB).
- Every A/B leg logged `workspace_bytes` 908,592,896 at both widths.

**Cold prefill** (tok/s, mean of each cell's samples over two legs per arm; min-max):

| cell | 1024 | 2048 | 2048 vs 1024 | µs/token 1024 → 2048 |
|---|---:|---:|---:|---:|
| 8.2K (the warm-up request) | 9,131 (9,097-9,166) | 9,048 (8,868-9,229) | −0.9% | 109.5 → 110.5 |
| 14K | 10,385 (10,324-10,505) | 10,373 (10,284-10,469) | −0.1% | 96.3 → 96.4 |
| 46K | 9,536 (9,515-9,557) | 9,717 (9,636-9,794) | **+1.9%** | 104.9 → 102.9 |
| 105K | 7,696 (7,683-7,709) | 7,865 (7,834-7,907) | **+2.2%** | 129.9 → 127.1 |

At 46K and 105K every 2,048 sample is above every 1,024 one; at 8K and 14K the ranges overlap.

**Decode** (greedy, 512 tokens): C=1 371.2 → 366.4 tok/s (−1.3%), C=8 1,059.5 → 1,050.5 (−0.9%);
texts identical 8/8 within and across arms; τ 5.256 / 5.256 (C=1). Per leg in launch order the C=1
means are 363.4 (2048), 369.3 (1024), 369.3 (2048), 373.2 (1024) and the C=8 ones 1,043, 1,049, 1,057,
1,069: one rise over the session that the interleaving puts on both arms, not a split by width.

**A decoding lane's pauses** (32.7K prompt; two lanes at ~55.7 tok/s before it; decode share 25):

| width | prompt alone | prompt with two lanes (6 reps) | lanes during | ITL p95 during | longest gap |
|---:|---:|---:|---:|---:|---:|
| 1024 | 3.37 / 3.34 s | 4.48-4.62 s | 17.0 / 17.2 tok/s | 125.5 / 122.3 ms | 186.1 / 181.9 ms |
| 2048 | 3.31 / 3.31 s | 4.33-4.49 s | 17.0 / 16.9 tok/s | 214.6 / 212.5 ms | 303.8 / 310.0 ms |

**KLD** on the 2026-09-24 4K suite (16 windows of 4,096 tokens; `span_logits` at each width from the same
binary, dumps on Y:; one `ref_kld.py` run scoring both; window bootstrap of the paired difference,
`.scratch/341/kld_paired.py`):

| width | KLD | top-1 | ppl |
|---:|---:|---:|---:|
| 1024 | 0.05221 | 92.41% | 2.7503 |
| 2048 | **0.04916** | 92.58% | 2.7369 |

Difference −0.00305, 95% CI [−0.00427, −0.00188]; 14 of 16 windows lower. By position: 0-512 0.0457 →
0.0466, 512-1024 0.0518 → 0.0502, **1024-2048 0.0494 → 0.0422**, 2048-4096 0.0553 → 0.0530. The 1024
dump reproduces #335's 0.05221 exactly.

**Agreement** (`ignis_kernel_hq_route_agreement_test`, new arms W=2048 at 3,000 / 32,000 and W=4096 at
32,000): ignis's median distance from BF16 is ×0.9996-×1.0050 of A1's, as at W=1024 (×1.0021-×1.0052);
band cuts and every split count within their bounds; the whole test green.

## Finding

Observed:

1. **2,048 costs no VRAM with vision on.** The vision encoder's bounded workspace still sizes the arena
   (by 12.1 MB), so the plan at the profile's 917,504 tokens is the same byte for byte. Without vision
   (a text-only load) it would cost 408.6 MB of KV pool. 4,096 costs 669.7 MB either way.
2. **Cold prefill +1.9% at 46K and +2.2% at 105K, unchanged at 8K and 14K.** −2.0 and −2.8 µs per token.
   Before #335 the same step measured −1.4% / −2.2% at 8K / 32K (finding 2026-10-07).
3. **Decode is unchanged** within the session's drift; greedy text identical.
4. **The KLD is lower, outside the noise.** The gain sits where 1,024 starts its second chunk
   (positions 1,024-2,048: −0.0072).
5. **A decoding lane's longest pause grows ~65%** while a prompt prefills (184 → 307 ms; ITL p95 124 →
   213 ms). The lanes' rate under the decode share is unchanged (17 tok/s), and the prompt's TTFT under
   load is 2.9% shorter (4.53 → 4.40 s, means of six).

Inferred:

- **Why the gain grows with context.** The 8-bit band is decoded once per chunk and layer (ADR 0061), so
  its cost per token is the prefix over the width: halving the chunks halves it. At 8K that saving is
  small and the window pass's BF16 keys per token grow by W/2 (~2 → ~3 µs at the BF16 rate, handoff
  estimate), which cancels it with the per-traversal fixed costs.
- **Why the KLD drops.** At 1,024 a position past 1,024 sees the keys before the ring through the 8-bit
  band; at 2,048 the whole first 2,048 tokens are one chunk, read by the BF16 window pass. On the 4K
  suite that is half of every window. At long context the band carries almost every key at either
  width, so the gain there is ~0: the 4K suite measures the first chunks only.
- **The "fixed part" is not one constant** (AC 2's term). From this session's 1,024 legs, subtracting
  the attention at #335's 0.857-0.915 ns per key·token (~3.7 / 6.4 / 21 / 45 µs at 8K / 14K / 46K /
  105K): ~106 / ~90 / ~84 / ~85 µs per token, and ~82 at 46K-105K with 2,048. The excess at short
  prompts behaves like a per-request cost of ~80 ms (14K: 6 µs × 14,293), and the 8K cell is also the
  first request after the load.

## Implications

- **The rtx5090 profile prefills the 27B at 2,048** (`model.qwen38.prefill_chunk`). The make default
  `PREFILL_CHUNK` is now empty, so it reaches `make start`; Flash-Next keeps its 8,192, a start without
  the profile its 1,024 (`DEFAULT_PREFILL_CHUNK`), and a text-only load is not changed by this.
- **4,096 is not adopted**: +669.7 MB on the 5090 is more than the margin the 917,504 profile has, so it
  needs either the profile re-measured below 917,504 or a configurable margin (owner, 2026-10-21: to
  decide). Its timing is not measured here.
- **AC 2's estimate, recomputed with the measured attention** (0.86-0.92 ns per key·token, not the
  assumed 0.65): with the fixed part at 85 µs, 60-73K prefills at **~8,400-9,000 tok/s** and 19K at
  **~10,700**, not 9,000-10,500 and 11,000-12,000. At 46K-105K the fixed part is already ~82-85 µs with
  2,048, so the AC's "≤ 85 µs" holds there; what is far from it is the short prompt (~106 µs at 8K),
  whose excess is per request, not per token. AC 2's operational form (8K ≥ +12%) measures the
  warm-up request.

## Limits and unknowns

- One prompt family per cell, two legs per arm, one session; the 8K cell is one sample per leg (the
  warm-up).
- The stall probe ran two decoding lanes committing ~1 token per round; with real acceptance or more
  lanes a round costs more, and the gap with it.
- The KLD gain is measured on 4K windows only; at long context it is expected to vanish (Inferred).
- The probe of the rtx5090 profile at 917,504 was **refused at both widths** today: the VRAM budget was
  29,424,877,568 B against 29,941,596,160 at #335's probe the same morning (−516.7 MB, the desktop
  holding 2.9 GB), and the plan 273.8 MB over it. `make start` passes 524,288 and is unaffected; a start
  that leaves `max_context` to the profile is not. Not caused by the width (identical plan at 1,024).
- 4,096's timing, and widths 1,536 / 3,072, not measured.

## Follow-ups

- Owner decision (not a ticket yet): the 917,504 profile's margin against the desktop's VRAM — today a
  bare start is refused by 274 MB. A configurable margin, or re-measuring the profile, also decides
  whether 4,096 can be had.
- AC 2 of spec prefill-5090/01 carries the recomputed estimate (amended with this finding).
