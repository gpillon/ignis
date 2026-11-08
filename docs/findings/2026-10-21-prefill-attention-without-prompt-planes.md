# The 27B's prefill attention without prompt planes: −1.02 GB at 524K, cold prefill +30% at 46K and +53% at 105K

- Kind: experiment
- Status: current
- Observed: 2026-10-21
- Last verified: 2026-10-21
- Scope: kernel / ignis's hq-e8-2b prompt attention (`kernel/src/hq_prompt_attention.cu`, ADR 0061) against the vendored A1 prompt route; serving / the 27B's VRAM plan, KLD, decode and cold prefill on the RTX 5090 (PCIe Gen 3 host)
- Topic: prefill
- Related: [ADR 0061](../adr/0061-prefill-attention-reads-the-hq-codec.md), [spec 27b-16gb/03](../specs/27b-16gb/03-vram-recovered-on-every-card.md) (ACs 1-5), GitHub #335, [the token embedding in host memory](2026-10-21-token-embedding-in-host-memory.md) (the same A/B, ACs 6-7), [KLD against BF16](2026-09-24-kld-against-bf16.md) (the suite and the 0.0513), [cold prefill 27B vs Flash-Next](2026-10-13-cold-prefill-27b-vs-flash-next.md) (the prefill driver and prompts), [hq attention route agreement](2026-09-12-hq-attention-route-agreement.md)
- Superseded by: none
- Summary: The 27B's 8-bit-band prefill attention (INT8 Q·Kᵀ, E4M3 P·V) frees 1.02 GB at 524K and speeds cold prefill +30% at 46K and +53% at 105K, KLD within noise (0.0522 vs 0.0513); with the host embedding the 5090 fits 917,504 tokens.

## Question

Under hq-e8-2b the vendored prompt route decodes a layer's whole history into BF16 prompt planes before a prefill
chunk attends: 4,096 bytes per context token, 1.07 GB at 262,144 keys or more, held by the arena for the whole load.
ADR 0061 replaces it with an attention of our own that reads the codec through a fixed 32,768-key band, in 8 bits by
default (Q·Kᵀ in INT8, P·V in E4M3) and in FP16 accumulate as the fallback. Does the VRAM come back? Is the output as
close to BF16 as A1's (agreement, KLD)? Is cold prefill faster and decode no slower?

## Evidence

**Setup.**
- Builds: main `7eba161c` and branch `335-prefill-fp8`:
  - the kernel bench, the agreement test and the 4K KLD: the branch at `84341637` (the prompt attention with INT8
    Q·Kᵀ, before the host embedding's merge mattered to any of them);
  - the A/B and the VRAM plan: `287d4802`, the merged branch (the prompt attention, the host embedding of ADR 0062 and
    the precision option), `make build`, release;
  - the same artifact (`qwen3_8_27b_nvfp4full-v2.ninfer`).
- The A/B ran the flags `make config IDLE_UNLOAD=0 LOOP_GUARD=off UI=0` prints: hq-e8-2b, 524,288 context (YaRN x2),
  `--model-prefill-chunk 1024`, DFlash2 with 7 draft tokens, the `rtx5090` profile. Two launches per build, one
  session, the GPU lock held, nothing else on the card. The branch's two launches bracket one of
  main's (branch, main, branch); an earlier session the same day (main twice, then the branch with E4M3 Q·Kᵀ and
  without the host embedding twice) is the second comparison.
- Cold prefill: the 2026-10-13 `client.py` and `slices.json` (8,232 warm after nothing, then 14K / 46K / 105K twice,
  each opened by its own nonce; `cached_tokens` 0 on every request). Decode: `ab.py --phases c1,c8`, the 2026-09-24
  coding prompts, greedy, thinking off, 512 tokens.
- Raw data: `.scratch/335/` in the main checkout (untracked): `ab335.py`, `ab_summary.py`, `profile_probe.py`,
  `ab/<leg>.*`, `probe-*.json`, the bench and agreement logs, `kld335*.out`; the decode results in
  `.scratch/sota-quick-wins/results/`; the dumps on `Y:/kld-335/`.

**The attention alone** (`kernel/tests/bench_hq_prompt_attention.cu`: one 1,024-token chunk at the end of a history
of the given length appended through the codec, Gaussian rows; the time over 16 layers, in ns per key already seen per
token):

| history | A1 (prompt planes) | ignis, INT8 Q·Kᵀ | ignis, E4M3 Q·Kᵀ | ignis, FP16 fallback |
|---:|---:|---:|---:|---:|
| 46,080 | 2.147-2.150 | 0.915 | 0.879-0.889 | 1.464 |
| 105,472 | 2.098-2.120 | 0.857 | 0.832-0.868 | 1.496 |

The cold-prefill estimate of spec 03 assumed 1.84 ns today and ~0.65 ns for FP8; the measured 8-bit attention is 2.3-2.5×
A1's speed.

**Route agreement** (`kernel/tests/test_hq_route_agreement.cu`, each route's per-row relative L2 against ignis's BF16
route, on the fixture's rows; the ratio is ignis's median over A1's, below 1 closer):

| arm | A1 median | 8-bit (INT8 Q·Kᵀ) | FP16 fallback | E4M3 Q·Kᵀ (not kept) |
|---|---:|---:|---:|---:|
| W=200, exact window | 0.002337 | ×0.986 | — | — |
| W=200, codec only | 0.227955 | ×1.003 | ×1.001 | — |
| W=200 at 4,096 | 0.138590 | ×1.000 | ×0.998 | — |
| W=200 at 4,096, codec only | 0.151415 | ×0.998 | ×1.000 | — |
| W=1024 at 3,000 | 0.137312 | ×1.002 | ×1.000 | ×1.008-1.014 |
| W=1024 at 3,000, codec only | 0.167624 | ×1.001 | ×1.001 | — |
| W=1024 at 32,000 | 0.088114 | ×1.005 | ×1.001 | ×1.062-1.066 |
| W=1024 at 32,000, codec only | 0.089506 | ×1.004 | ×0.999 | — |

Cosines against BF16 agree with A1's to 1e-4 on every arm (32,000: A1 0.995570, 8-bit 0.995557, FP16 0.995571). The
W=9-16 arms (one partial tile) sit at ×0.992-1.002. The 8-bit band adds an error of its own: at 32K it is ~1% of a
row (√(0.088575² − 0.088114²) ≈ 0.009), next to the codec's 8.8%. The test now fails a ratio past ×1.01 on any
arm, which the E4M3 Q·Kᵀ this replaced would have failed at 32K.

**KLD on the 2026-09-24 suite** (16 windows of 4,096 tokens, `span_logits` at chunk 1024 against the BF16 reference,
`ref_kld.py`; the interval is a window bootstrap of the paired difference):

| route | KLD | top-1 | difference against A1 |
|---|---:|---:|---|
| A1 (main's binary) | 0.05134 | 92.44% | — |
| ignis, 8-bit (INT8 Q·Kᵀ) | 0.05221 | 92.41% | +0.00087, CI [−0.00068, +0.00301] |
| ignis, E4M3 Q·Kᵀ (not kept) | 0.05123 | 92.47% | −0.00011, CI [−0.00114, +0.00104] |

The two ignis kernels differ by +0.00098 (INT8 over E4M3), CI [+0.00003, +0.00218], and one chat window carries most
of it: w11, +0.0145 over A1 and +0.0079 over E4M3, half of that from its ten largest positions. Without w11 the three
read 0.05091 (A1), 0.05087 (INT8) and 0.05035 (E4M3). Over the other fifteen windows INT8 sits above A1 in seven and
below in eight; the median per-position difference is 0 to 1e-7 for every pair. Perplexity: 2.7503 against A1's
2.7523 (E4M3: 2.7523), each window within −0.9% to +0.4% of A1's.

**VRAM** (`test_gqa_workspace_capacity`, and the plan's event `ignis.runtime.vram_plan`):

`test_gqa_workspace_capacity`: the attention's workspace for 1,024-token chunks at 524,288 keys is 1,164,194,688
bytes with the prompt planes and 143,720,448 now (−1,020,474,240), the same at any history past one band.

The make defaults (524,288 context, the `rtx5090` profile: vision, DFlash2, headroom 1,200 MiB), one launch of each
build in the same session:

| line | main | branch | difference |
|---|---:|---:|---:|
| `weights_bytes` | 18,585,584,640 | 17,234,723,840 | −1,350,860,800 (the token embedding, ADR 0062) |
| `workspace_bytes` | 1,508,311,552 | 908,592,896 | −599,718,656 |
| KV pool | 736,704 tokens | 941,760-949,504 tokens | +~208K tokens (+28%) |

The budgets differ by at most 71 MB (29.91-29.98 GB, what NVML reports free at the start).

The workspace line is one arena, sized by the largest of its users. With the planes that was the text layers'
prefill scratch, 1,508,311,552 bytes, vision on or off. Without them a text-only load's line drops by the attention's
saving, to 487,837,184 (`vram_plan_gpu`, 262,144 keys: −1,020,474,368), and with the profile's vision the arena is
the vision encoder's, sized for one 16,384-token item: 908,592,896 bytes, at 917,504 tokens of context too (below).
Neither grows with the context any more.

**Cold prefill** (tok/s, the mean of the samples: two reps per launch but at 8K; branch = the merged build):

Same session (main once, the branch twice):

| prompt | main | branch | difference | TTFT |
|---:|---:|---:|---:|---:|
| 8,232 | 8,664 | 9,291 | +7.2% | 0.950 → 0.886 s |
| 14,293 | 9,271 | 10,139 | +9.4% | 1.542 → 1.410 s |
| 45,606 | 7,128 | 9,244 | +29.7% | 6.398 → 4.935 s |
| 105,221 | 4,900 | 7,500 | +53.1% | 21.476 → 14.030 s |

The earlier session, prefill change alone with E4M3 Q·Kᵀ, two launches each: +4.8%, +9.3%, +28.7% and +53.9% (main
9,353 / 9,600 / 7,547 / 5,123). Main itself ran 4-7% slower in the second session than in the first on every cell,
and 6-8% slower in decode: the card's or host's state, the same for both builds of a session.

**Decode** (tok/s; c1 the mean of the eight prompts, c8 the aggregate of each round; τ the tokens per verify round
and the time per round, from the server's request-done events):

| leg | c1 | c8 round 0 / 1 |
|---|---:|---:|
| main | 359.3 | 1,029.1 / 1,008.8 |
| branch, first | 358.0 | 1,028.7 / 1,032.8 |
| branch, second | 361.0 | 1,036.6 / 1,037.4 |
| **branch / main** | **+0.1%** | **+1.5%** |

| | τ, c1 | ms per round, c1 | τ, c8 | ms per round, c8 |
|---|---:|---:|---:|---:|
| main | 5.466 | 17.62 | 5.433 | 30.51 |
| branch | 5.256 | 17.27 | 5.325 | 29.40 |

The time per round is a request's duration over its rounds. In the earlier session: c1 −1.4%, c8 −2.0%, τ the same
5.466 → 5.256 at c1, the time per round again lower (16.59 → 16.46 ms, 28.29 → 27.89 ms). The greedy texts are
deterministic within each build (8/8) and differ between the builds in all eight prompts, from a near-tie on: two of
the three multi-chunk prompts open with a code fence on the branch and a sentence on main. Per prompt, c1 moves from
−18% to +13% with its text.

**The `rtx5090` profile's 27B, raised until the plan refuses** (`.scratch/335/profile_probe.py`: the release binary,
the artifact and `--model-max-context N`, every other value the profile's, as on 2026-10-10):

| context | outcome |
|---:|---|
| 1,048,576 | refused: 1,073,100,032 bytes over the 29,833,641,984-byte budget (2026-10-10: 2.57 GB over) |
| 950,272 | refused: 59,127,040 bytes over 29,941,596,160 |
| 917,504 | loads: KV pool 944,832 tokens, `workspace_bytes` 908,592,896 |

A start's budget moves by ~100 MB with what the desktop holds; 917,504 needs ~240 MB less than the budget that
refused 950,272.

## Finding

Observed:

1. **No prefill chunk materializes prompt planes.** The attention's workspace stops growing with the context at one
   band: 143,720,448 bytes at any history past 32,768 keys, 1,020,474,240 fewer than A1's at 524,288. A text-only
   load's workspace line drops by as much (−1,020,474,368); with vision on, the line is the item-bounded vision
   encoder's 908.6 MB, so it drops by 599.7 MB.
2. **The 8-bit attention is 2.3-2.5× A1's speed** per key already seen, 0.86-0.92 ns per key and token over 16
   layers; the FP16 fallback 1.4×.
3. **Its output is as close to BF16 as A1's to within half a percent of the median row**, ×0.986-1.005 over every
   arm, and the KLD on the 4K suite within noise of A1's (0.05221 against 0.05134, CI of the difference across 0).
4. **Q·Kᵀ in E4M3 was a bias, not noise**: ×1.06 of A1's distance at 32K keys against ×1.01 at 3K, the same 2.5% of
   every query applied to every key. INT8 Q (~0.7% RMS by estimate, against 2.5%) took it out for ~3% of the
   attention's time at 46K and none measurable at 105K.
5. **Cold prefill is faster by 7% at 8K, 9% at 14K, 30% at 46K and 53% at 105K**, the same in both sessions: the
   attention is the growing term, and it costs 2.4× less.
6. **Decode is no slower**: +0.1% at one lane, +1.5% at eight, the time per round 2-4% lower. τ is 2-4% lower on the
   branch's texts, which differ from main's from the first near-tie on.
7. **The 27B on the 5090 fits 917,504 tokens of context** under the profile's vision, DFlash2 and 1,200 MiB headroom,
   against 786,432 before; 1,048,576 is 1.07 GB short.

Inferred:

- Spec 03's acceptance on the measured side: AC 1 holds for the attention's own workspace (−1,020,474,240 bytes against
  the amended ≥ 1.0e9), and for a text-only load's plan line; AC 3 holds (KLD within noise); AC 4 holds with the FP8 thresholds (+29.7% ≥ +13% at 46K,
  +53.1% ≥ +24% at 105K; decode no slower). AC 2 as written, "at least as close", holds for the FP16 fallback on every
  arm (×0.998-1.001) and for the 8-bit band on every arm but the 32K one, where it is ×1.005 (+0.5% on the median
  row, cosine 1.3e-5 lower). The owner accepted that half percent (2026-10-21): the band ships 8-bit by default,
  held to ×1.01 of A1 by the test, and FP16 behind `--model-prefill-attention fp16`.
- The 8-bit band's own error is ~1% of a row against the codec's ~9%: the hq-e8-2b codec, not the attention, is what
  separates the prefill from BF16.
- With vision on, the workspace line now follows the vision encoder (908.6 MB for a 16,384-token item): the next
  term to shrink on a 16 GB card that serves images. A text-only load already gets the whole 1.02 GB.
- The prefill estimate of spec 03 (FP8: ~10.0K tok/s at 46K, ~8.3K at 105K, from ~7.6K and ~5.2K) is reached within
  3-8% at 46K (9.2-9.7K) and 5-10% at 105K (7.5-7.9K): the attention measured 0.86-0.92 ns, not the assumed 0.65.

## Implications

- The prompt attention replaces the vendored prompt route for every hq-e8-2b prefill chunk on every card class, the
  8-bit band by default; the vendored route stays the agreement test's oracle.
- The `rtx5090` profile's 27B context goes from 786,432 to 917,504 (`crates/server/src/config/profile.rs`). Under
  `auto` rope scaling that moves its YaRN factor from 3 to 3.5, which ADR 0057 notes costs short contexts a little;
  the alternative was to keep 786,432 and let the KV pool take the ~1.95 GB (+~210K tokens) for concurrent and
  retained sequences. The owner chose the context (2026-10-21). The `rtxpro6000` values do not move: both families are already at the 1,048,576-token envelope.

## Limits and unknowns

- Two sessions: two launches per build in the first, and in the second main once between the branch's two. One
  text family per cell.
- The agreement test's rows are a real prefill's, but 1,024 distinct ones (256 positions of 4 layers), repeating past
  that: the 32K arm is not a real 32K history, though the repetition favours no route in particular.
- The 4K KLD cannot resolve a 0.001 difference between the two 8-bit kernels (one window's tail moves it by that
  much); the agreement test is the finer instrument for that choice, and it favours INT8 Q·Kᵀ as the history grows.
  No BF16-referenced KLD at long context was run: six long windows would give an interval wider than the effect. The
  engine-against-engine KL at long context (A1 against the E4M3-Q kernel: 0.0418 at 4K, 0.0530 over 12,288 long
  positions) measures chaotic sensitivity more than either attention, and was not rerun.
- Decode's greedy texts differ between the builds from the first divergent near-tie on (a preamble against a code
  fence); τ moves with the text, so a decode difference smaller than the text's own effect is not resolved here.
- Measured on PCIe Gen 3; the attention's own numbers do not depend on the link.
