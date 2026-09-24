# hq-e8-2b attention costs 0.09 ms per 1K context tokens per lane per round, ~85% of it the codec's decode, so an INT8/FP8 MMA would buy under 10%

- Kind: experiment
- Status: current
- Observed: 2026-09-24
- Last verified: 2026-09-24
- Scope: kernel / verify-round attention (`gqa_attention_small_t_tc_partial_bf16_kernel` with `GqaTcKVHq`; BF16 route), decode throughput vs context
- Related: [hq vs BF16 decode cost](2026-09-13-hq-vs-bf16-decode-cost.md) (pre-speculation),
  [KLD against BF16](2026-09-24-kld-against-bf16.md) (the same format's quality cost at long context),
  [upstream quick wins](2026-09-24-upstream-quick-wins-ab.md), `.scratch/sota-research-2026-09-24/SINTESI.md` §2.2,
  raw material in `.scratch/longctx-2026-09-24/`
- Superseded by: none

## Question

The research survey called hq-e8-2b's verify attention compute-bound at long
context: 48 query rows per KV head at width 8, ~341 flop per byte of hq storage. It
proposed running its QK/PV on INT8 or FP8 MMA, which runs 2–4× the GeForce BF16 rate.
How much of a verify round does hq attention cost as context grows, at 1, 4 and 8
lanes? And how much of that cost is MMA, the part a narrower MMA could shrink?

## Evidence

**Setup.**
- RTX 5090, exclusive, main `4039aa9`, `make config` flags plus `--metrics`.
- Each lane is a distinct nonce followed by the same repo-source body, `ignore_eos`,
  greedy, 768 tokens.
- Every lane's prompt is prefilled first, alone. The measured requests claim it
  through prompt reuse (TTFT ≤ 0.8 s), so all lanes decode together.
- Round time = `itl_ms_mean × itl_samples / spec.rounds` per request, from the
  `ignis.request.done` log. It does not depend on how many drafts a round accepts,
  which varies with the text.
- Drivers `longctx.py` and `longctx_nsys.py`. `nsys profile --cuda-graph-trace=node`
  one-lane cells split by `nsys_split.py`.

**Verify round time, hq-e8-2b** (ms; prompt tokens are the actual counts):

| context | 1 lane | 4 lanes | 8 lanes |
|---:|---:|---:|---:|
| 6.9K | 16.7 | 25.4 | 36.3 |
| 14K | — | 27.7 | 42.6 |
| 30K | 19.5 | 32.8 | 50.7 |
| 59K | 21.4 | 44.0 | (reuse fell through, not measurable) |
| 125K | 27.6 | 68.8 | (does not fit the 735K-token pool) |

Slope in round time per 1K context tokens per lane:

| lanes | range | ms per 1K tokens per lane |
|---:|---|---:|
| 1 | 7K→125K | 0.093 |
| 4 | 7K→125K | 0.092 |
| 8 | 7K→30K | 0.078 |

**nsys, 1 lane at 59K, decode only** (after the last prefill-width GEMM, ~61 rounds):

| route | attention per round | per layer | GEMM per round | attention share |
|---|---:|---:|---:|---:|
| hq-e8-2b: `small_t_tc_partial` (hq source) | 7.6 ms | 0.364 ms (×16 layers) | 14.0 ms | 33% |
| BF16: **`gqa_attention_prefill_bf16_kernel`** | 15.0 ms | 0.94 ms | 14.0 ms | 48% |

**Roofline per GQA layer, per 1K keys, width 8** (24 query heads × 8 rows × head_dim
256):
- QK + PV is 201 MFLOP, which is 0.96 µs at the BF16 MMA rate (209.5 TFLOPS).
- hq storage is 590 KB, which is 0.33 µs at 1.79 TB/s.
- BF16 storage is 4.2 MB, which is 2.3 µs.

| route | measured per layer per 1K keys | against its limit |
|---|---:|---|
| hq | 6.2 µs | 6.5× the MMA limit, 19× the byte limit |
| BF16 | 15.9 µs | 7× its byte limit |

## Finding

**Observed.**
- hq-e8-2b verify attention grows linearly with context at ~0.09 ms per 1K tokens per
  lane per round, the same at 1 and 4 lanes.
- 8 lanes × 30K spends ~20 ms of a 51 ms round on it; 4 lanes × 125K spends ~43 ms of
  69.
- Because the cost scales with lanes × context, it is throughput-bound, not
  latency-bound.

**Inference: the split.** The hq kernel runs 6.5× slower than its MMA work and 19×
slower than its bytes. The difference is the per-tile group decode:
- Rice symbols, the E8 coset, the dither hash and the scale for every K and V element;
- **redone every round for the whole history**, shared only across the 6 query heads
  and 8 rows of one KV head.

MMA is ≲15% of the kernel. An INT8 QK and FP8 PV path (exact INT8 is possible for K:
`round(16·(y+d))` fits when |y| ≤ 7) would therefore save **under 10% of attention**.
It is not the lever.

**Observed: BF16 KV.** The BF16 verify round does not run the small-T split-KV kernel.
It runs the **prefill** attention kernel at T = 8, 7× off its byte limit, so at 1 lane
and 59K the BF16 format is *slower* than hq (29.1 against 21.4 ms). The 2026-09-13
"BF16 +48.5% at 40K" predates speculation (T = 1 decode) and does not describe
today's route.

## Implications

- The attention levers at long context, in order:
  1. do less decode per round: decode each page once into a short-lived FP8/BF16
     tier that the next rounds reuse, or a cheaper codec;
  2. a KV format that needs no ALU decode: FP8 E4M3 at 32 KB/token, which the KLD
     finding also favours for quality;
  3. only then a narrower MMA.
- A BF16-source small-T route is a separate, cheap-looking fix for the BF16 oracle
  format. BF16 is not the serving default, so it matters for measurements and A/Bs
  more than for users.
- Per-round attention cost, not acceptance, decides the 8-lane × long-context cell.
  The drafter cannot hide it.

## Limits and unknowns

- The decode-vs-MMA split is inferred from the kernel's measured time against its
  roofline, not from an instruction-level profile (Nsight Compute was not run).
- 8 lanes × 59K could not be measured: some lanes' prompt reuse fell through and they
  re-prefilled inside the window.
- One run per cell. The slope agrees across three lane counts, but single cells carry
  a few ms of spread.
- nsys cells at 1 lane only.
