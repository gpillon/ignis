# The 27B's decode served on the RTX 5090 after spec #672: at equal FP8 exception weights a long JSON answer 10.2% faster, an 8K prompt's answer 9.3%, plain decode 3.1-4.2%, the verify round 7.1% shorter, the one-lane text main's

- Kind: experiment
- Status: current
- Observed: 2026-11-18
- Last verified: 2026-11-18
- Scope: decode / the 27B NVFP4 artifact on the RTX 5090 (Windows, PCIe Gen 3), main at ab2adc85 with the round timing instrument (184d8113) against the integration build of spec #672 at eb0a823d, each lever's measurement switch as its control
- Topic: decode-27b
- Related: https://github.com/gpillon/ignis/issues/683,
  https://github.com/gpillon/ignis/issues/672,
  https://github.com/gpillon/ignis/issues/679,
  [ADR 0103](../adr/0103-on-compute-capability-12-0-the-27b-holds-its-exception-tensors-in-fp8-by-default.md),
  [decode round anatomy](2026-09-18-decode-round-anatomy.md),
  [decode round host idle](2026-09-18-decode-round-host-idle.md),
  [the cold prefill after spec #646](2026-11-18-the-27bs-cold-prefill-served-on-the-rtx-5090-after-spec-646.md)
- Superseded by: none
- Summary: RTX 5090, 27B NVFP4, FP8 named, served A/B vs main: DFlash2 JSON 346 to 382 tok/s (+10.2%), 8K answer 393 to 430 (+9.3%), plain +3.1-4.2%; verify period -7.1%, plain -3.6%; one-lane text main's; #679 dropped.

## Question

Spec #672 took the 27B's decode round apart on main and built its levers: a DFlash2 verify round that does not wait on the host for its accept (#676), a decode graph without copies at width 1 (#677) and at widths 2-8 and in the verify graph (#680), the paired FP8 GEMV for the grouped exception projection (#678), the GDN gating off the round's chain (#681), the W4A4 activation quantize inside the RMSNorm (#679), and FP8 exception weights by default on compute capability 12.0 (#682). Each lever was measured alone on the GPU clock, alternated with its measurement switch, never all together and never served. #683 asks:

- what a user sees served with every lever in, against main, at equal explicit exception weights;
- each lever's share where its switch separates it;
- whether the greedy text is main's at equal exception weights;
- the FP8 default's own share, apart from the levers';
- what was dropped, and why.

## Evidence

### Conditions

RTX 5090 under Windows, on a PCIe Gen 3 x16 host, one card and nothing else on it: `nvidia-smi` read 3,340 MiB used before the A/B's legs and 3,285 MiB after, with no other GPU process, and no build ran during any leg. The 27B NVFP4 artifact. Every leg is one server start with an empty config file (no user config file read), the loop guard off and idle unload off. Every lever leg and the A/B's configurations A and B name `--model-exception-weights fp8`. Round timing is `IGNIS_ROUND_TIMING=1` (#674): events recorded on the stream around each replayed round's graph, no tracer, one line per 256 rounds of a kind with the median period, span and gap. Each figure below is the mean of a leg's block medians.

### The round on main

From the spec's decomposition (2026-11-17, one lane, untraced GPU-clock timings) and the pins of #677 and #680:

- **The verify round waits on the host.** A DFlash2 verify round with 7 drafts took 16.04-16.14 ms: 14.80-14.91 ms of GPU work and 1.17 ms of gap. The accept came back through a pageable copy and a stream synchronize, the drafter's append after the commit ran kernel by kernel, and the next round's inputs went up as five pageable copies. Each pageable copy queued after a graph started 120-180 us after it, and the next kernel 110-170 us after the copy.
- **The decode graph copies.** The width-1 decode graph has 885 nodes: 677 kernels and 208 device-to-device copies (each GDN layer's q, k and v split out of the conv plane, each layer's residual). Each copy adds about 1.35 us to the serial chain, 280 us a round. The pins count 208 copies at every width 1-8 and 86 in the verify graph on main's path; the spec's 144 at widths 2-8 counted the prototype's graph, which already added the residual in place there.
- **Two passes of their own.** The GDN a|b gating is a separate BF16 product, 154-161 us a round at 0.35 TB/s; the W4A4 activation quantize a separate pass after the RMSNorm, 66 us a round.
- **The FP8 exception products** take 446 us of a round at about 1.2 TB/s, where the round's NVFP4 families run at 1.52-1.59 TB/s. The round reads 15.3 GB of weights.
- **The launch is not the cost.** Untraced, launching the width-1 graph takes 14-23 us whatever its node count; under Nsight Systems a launch costs about 400 us more on this host, which is the tracer's.

#674's reproduction on main's path (184d8113, FP8 named): a plain round's period 11,380-11,605 us with a gap of 258-304 us in the early blocks (later blocks drift to 11.8-12.8 ms as the context grows), and a verify round's period 15,955 / 15,983 us, span 14,641 / 14,716 us, gap 1,200 / 1,163 us.

### Each lever's share

Two legs per arm, alternated, each leg one load; the control is the lever's own switch, on a build that carries every lever merged before it. The plain legs write the same 3,643-token JSON answer in 14 blocks of 256 width-1 rounds; the verify legs serve one JSON request four times, DFlash2 with 7 drafts. #677 and #678 share their reference legs.

| lever | build | round | new path | main's path (its switch) | change |
|---|---|---|---|---|---|
| #676 verify round's host wait | ee560889 | verify, one lane | period 15,088.6 / 15,088.8 us, gap 536.1 / 537.4 | `IGNIS_DEFER_VERIFY=0`: period 15,720.3 / 15,743.5, gap 1,102.4 / 1,113.3 | period -643 us (-4.1%), gap 1,108 to 537 us, span within 40 us |
| #677 copy-free width 1 | ee560889 | plain, width 1 | span 10,830.7 / 10,845.5 us, period 11,133.6 / 11,105.2 | `IGNIS_DECODE_GRAPH_COPIES=1`: span 11,105.0 / 11,105.6, period 11,361.9 / 11,374.9 | span -267 us (-2.4%), period -249 us |
| #678 paired FP8 GEMV | ee560889 | plain, width 1 | span 10,830.7 / 10,845.5 us | `IGNIS_FP8_GEMV_PAIRED=0`: span 10,966.0 / 11,060.6, period 11,222.5 / 11,305.2 | span -175 us (-1.6%), period -145 us |
| #680 copy-free widths 2-8 | 65f9cee1 | burst of 4, plain | span 12,754.0 / 12,754.8 us, period 13,089.2 / 13,126.6 | `=wide` (width 1 copy-free, main's past it): span 13,031.1 / 13,131.0, period 13,396.8 / 13,500.3 | span -327 us (-2.5%), period -341 us |
| #680, burst of 8 | 65f9cee1 | plain | span 13,867.2 / 14,291.9 us | `=wide`: 14,651.4 / 14,206.4 | -349 us on the means; the legs overlap |
| #680, verify graph's copies | 65f9cee1 | verify, one lane | period 14,984.1 / 14,929.3 us, span 14,439.8 / 14,350.3 | `=wide`: period 15,022.4 / 14,966.0, span 14,467.4 / 14,422.6 | period -37 us, inside the spread |
| #681 gating on a branch | 66a9f6b3 | plain, width 1 | span 10,754.0 / 10,770.3 us | `IGNIS_GDN_GATING=chain`: 10,821.0 / 10,833.6 | span -65 us (-0.6%), against ~15 us between legs of one home |
| #681 gating on a branch | 66a9f6b3 | verify, one lane | span 14,110.8 / 14,131.0 us, period 14,676.9 / 14,693.3 | `chain`: span 14,363.4 / 14,348.4, period 14,932.0 / 14,903.9 | span -235 us, period -233 us (-1.6%) |

- #676: the paths' text (a 3,645-token answer) has the same sha256 in every leg of both arms. The prototype had measured a gap of 1,172-1,177 to 552-560 us and -3.8 to -4.1%.
- #677: the prototype had measured -280 us.
- #678: the kernel bench's estimate was ~65 us a round over the projection's six calls; the served round took 175 us less. The two one-row legs differ by 95 us, and the lower one is still 120 us above the paired arm.
- #680: widths 2-8 gained more than the spec's ~0.195 ms estimate, which counted the split conv alone; the in-place residual is in the measured figure. The verify graph's 64 fewer copies (86 to 22: the drafter's 22 stay) are within the noise, as the prototype measured them (+0.2/+0.5% drafted).
- #681: the spec estimated ~0.15 ms a round for the gating move.
- Every lever's text has one sha256 per round kind in every leg of its arms, except #680's bursts, where one leg of each arm differs from the rest (see [The text](#the-text)).

### Dropped

- **#679, the W4A4 quantize inside the GDN input RMSNorm** (decode graph, widths 1-8). Built on its own branch, byte for byte the separate pass (`ignis_kernel_norm_quantize_test`, and the round identity arm `norm-quantize-separate` on the NVFP4 body with DFlash2 under bf16 and fp8); the width-1 graph lost its 48 separate quantize kernels, 677 nodes to 629. Three legs per arm at the branch tip, plain width 1, the same 3,643-token answer, alternated: fused span 10,869.7 / 10,859.4 / 10,814.9 us (mean 10,848), `IGNIS_NORM_QUANTIZE=separate` 10,832.8 / 10,848.1 / 10,832.2 (mean 10,838). The span did not fall: +10 us, inside the ~55 us between legs of one arm, where the spec had estimated ~0.06 ms. The norm's own quantize step cost what the separate pass cost. By the ticket's rule (a pass stays only if the round's span falls by more than its spread) it is dropped; nothing of it is on the integration build ([#679](https://github.com/gpillon/ignis/issues/679)).
- **#681's fused home**: the gating as 96 CTAs ahead of the conv's in one grid of the decode round's split conv. Plain width-1 span 10,768.9 / 10,767.8 us, -59 us against `chain`; it does not reach the verify round, which keeps the chain there, so nothing moved at the verify window (span 14,345.9 / 14,349.8, period 14,922.8 / 14,928.8). The branch gained more at both points, so the branch is the default and the fused home was removed from the code (27e6ce71).

### The served A/B

**Builds.** M is 184d8113: main at ab2adc85 plus #674's round timing instrument alone, which costs one bool test a round when off. N is the integration build at eb0a823d: every lever of the spec, #681's branch home, the `auto` FP8 default; #679 is not in it. The two executables were frozen before the legs. Per configuration, four legs in the order M N N M; twelve legs from 17:37 to 17:49.

**Cells.** Streamed chat completions, temperature 0, thinking off; decode tok/s is (tokens - 1) over the time from the first streamed chunk to the last, per request; the text's sha256 is taken per request.

- JSON-long: a JSON array of 80 fictional employees (a 55-token prompt, up to 8,192 tokens);
- 8192A: an 8,192-token prompt, ignis's own word-soup system block and a coding request, the decode cell of the #646 A/B (here the same 3,435-token answer, sha256 prefix `b92644094e51`, on both builds);
- a burst of 4: four distinct JSON asks of 40 employees sent at once (up to 4,000 tokens each), reported as tokens over wall time;
- plain: a JSON array of 40 employees (3,643 tokens), a travel conversation and an 800-word story (1,200 tokens each).

**A: FP8 named, context 262,144, rope none, DFlash2 with 7 drafts** (equal weights: the levers' share).

| cell | M (legs 1, 4) | N (legs 2, 3) | change |
|---|---|---|---|
| JSON-long (7,172 tokens) | 342.69 / 350.16 tok/s | 381.67 / 382.19 | +10.2% |
| 8192A answer (3,435 tokens) | 391.62 / 394.90 | 429.77 / 430.30 | +9.3% |
| burst of 4, aggregate | 1,014.87 / 1,031.59 | 1,059.74 / 1,073.95 | +4.3% |

Round timing, verify blocks (they include the burst's verify rounds of several lanes): period 17,673.8 / 17,456.1 us to 16,321.9 / 16,327.5 (-1,240 us, -7.1%); gap 1,426.4 / 1,390.3 to 773.9 / 775.8; span 16,149.1 / 16,028.0 to 15,530.3 / 15,535.5.

**B: FP8 named, the same shape, drafting off** (plain rounds).

| cell | M | N | change |
|---|---|---|---|
| JSON (3,643 tokens) | 86.11 / 86.01 tok/s | 87.96 / 89.56 | +3.1% |
| conversation (1,200) | 86.45 / 86.53 | 89.02 / 90.01 | +3.5% |
| prose (1,200) | 86.13 / 86.61 | 89.71 / 90.36 | +4.2% |

Round timing, plain: period 11,472.6 / 11,471.4 us to 11,126.9 / 11,001.3 (-408 us, -3.6%); span 11,204.9 / 11,217.3 to 10,847.3 / 10,747.5 (-414 us); gap about 250-260 us in both builds.

**C: each build's defaults** (no weights, context or rope flag).

- M: exception weights `bf16` (main's default); the context adapted from 917,504 to 786,432 tokens, yarn:3.
- N: `auto -> fp8 (default)` on its load line; 917,504 tokens, unadapted, yarn:3.5.

| cell | M (legs 1, 4) | N (legs 2, 3) |
|---|---|---|
| JSON-long | 354.19 / 347.55 tok/s (7,249 tokens) | 399.20 / 408.58 (7,088 tokens) |
| 8192A answer | 361.24 / 352.39 (2,214 tokens) | 432.18 / 431.40 (3,637 tokens) |

The texts differ between the builds, as expected: FP8 weights and yarn:3.5 move greedy text against a BF16 load (ADR 0065, [ADR 0103](../adr/0103-on-compute-capability-12-0-the-27b-holds-its-exception-tensors-in-fp8-by-default.md)). Within each build the default load's text repeats exactly across its two legs. Whether BF16 adapts its context on this card depends on the free VRAM at start, not on the encoding alone: ADR 0103 gives the thresholds.

### The text

- A, JSON-long and 8192A: one sha256 per cell in all four legs, both builds.
- B: one sha256 per cell in all four legs.
- A, the burst of 4: three legs write the same four answers (M1, N3, M4); N2 differs on 2 of the 4.
- The round identity test (`crates/core/tests/round_identity_gpu.rs`) on the integration build at eb0a823d holds every arm to the reference token for token at widths 1-8 and at the verify window, greedy and sampled lanes: `eager`, `round-timing`, `sync-verify`, `decode-graph-copies`, `decode-graph-copies-wide`, `fp8-gemv-one-row`, `gdn-gating-chain` and `gdn-gating-branch`, on the NVFP4 body with DFlash2, with its MTP head and with vision under bf16 and fp8, and on the 16 GB and 12 GB trellis bodies under bf16: 64 arm passes in 453 s.

## Finding

**Observed.** With every lever of spec #672 in, at the same explicit FP8 exception weights, the 27B decodes faster on the RTX 5090 in every cell, and every one-lane answer's text is main's:

- with DFlash2, the 7,172-token JSON answer from 346.4 to 381.9 tok/s (+10.2%), the 8K prompt's answer from 393.3 to 430.0 (+9.3%), a burst of four from 1,023 to 1,067 tok/s in aggregate (+4.3%);
- without drafting, JSON, conversation and prose from 86.1-86.5 to 88.8-90.0 tok/s (+3.1%, +3.5%, +4.2%);
- the plain round's period from 11.47 to 11.06 ms (-3.6%; span -414 us; gap unchanged at about 250-260 us). The spec expected ~10.6-10.9 ms: the measured round stays above that range;
- the verify period, over blocks that mix one-lane and wider verify rounds, from 17.56 to 16.32 ms (-7.1%), its gap from 1,408 to 775 us. On one lane, #676 alone took the period down 4.1% (gap 1,108 to 537 us), where the spec expected 5-6% from the gap; #681's branch took 233 us more.

**Observed.** Each lever's share, against its own switch:

| lever | plain round | verify round |
|---|---|---|
| #676 host-mapped accept, fold in flight | not reached | period -643 us (-4.1%), gap -571 us |
| #677 no copies at width 1 | span -267 us (-2.4%) | not reached |
| #678 paired FP8 GEMV | span -175 us (-1.6%) | one-lane round reached, not separated |
| #680 no copies at widths 2-8 and in verify | span -327 us at a burst of 4 (-2.5%) | -37 us, inside the spread |
| #681 gating on a branch | span -65 us (-0.6%) | span -235 us, period -233 us (-1.6%) |
| #679 quantize in the norm (dropped) | +10 us, inside the spread | not reached |
| #681 fused home (dropped) | span -59 us | not reached |

**Inference.** The levers that paid took work off the round's serial chain or the host's wait: the copies, the host round trip of the accept and the fold, and the gating moved beside the input projection. The fused quantize removed 48 kernels and no time, since the norm then did the quantize's work on the same chain; node count is not the cost of a round, as the untraced 14-23 us launch also says.

**Observed.** With each build's defaults (config C), main's BF16 load at 786,432 tokens and yarn:3 writes other answers than the integration build's FP8 load at 917,504 and yarn:3.5: 354.19 / 347.55 against 399.20 / 408.58 tok/s on JSON-long, 361.24 / 352.39 against 432.18 / 431.40 on the 8K prompt's answer, over different texts. **Inference.** Those figures carry the texts' draft acceptance and the levers as well as the FP8 default, so the default's own share of decode speed is not separated by them; the levers' share at equal weights is A's and B's, and config C is what a default install of each build reads.

**Not confirmed here.** The RTX PRO 6000 (Linux), the other card that runs the 27B's NVFP4 artifact, still has to confirm the levers and the host-mapped wait: cells A and B against main at equal FP8 exception weights, the round timing of the plain and verify rounds with their gaps, and the verify gap under `IGNIS_DEFER_VERIFY=0` against unset. The host-mapped wait replaces a pageable copy and a synchronize whose cost was measured on this Windows host alone.

## Implications

- The guide's decode figures for the 27B on the RTX 5090 move to these cells: [One round for every lane](../user/why-ignis/decode-round.md).
- The gap left between rounds is about 250-260 us after a plain round and 537 us after a one-lane verify round. The plain round's deferred return is #669's.
- **Inference.** Removing kernels alone buys nothing on this card (#679, and the 14-23 us launch): a fusion is worth building where the pass it removes holds the chain and the fused kernel does not take that time back.
- The device-to-device copies of a plain width-1 round were worth 267 us, against the earlier estimates by their execution alone (about 0.5% of a verify round, [decode round anatomy](2026-09-18-decode-round-anatomy.md); 0.16%, [decode round host idle](2026-09-18-decode-round-host-idle.md)); in the verify graph, where those findings measured, their removal moved nothing beyond the spread. Both findings carry a dated note.

## Limits and unknowns

- One machine: an RTX 5090 under Windows (WDDM) on a PCIe Gen 3 host. Two served legs per build per configuration, two or three timing legs per arm.
- The levers' shares were measured against different baselines (each on the build of its day, with the levers merged before it), so they do not add up: the width-1 plain span shares of #677, #678 and #681 sum to -507 us, and the combined plain span fell 414 us. The difference is not explained here.
- Config A's verify timing mixes one-lane and wider verify rounds (the burst's); it is not comparable with the one-lane lever legs or the spec's 16.04-16.14 ms.
- The spec's served prototype figures (478.6 to 506.3 tok/s on a 7,229-token cell, 392.7 to 411.0 on the 8K prompt's answer) came from another session; this A/B's JSON-long cell is not that cell. Main's 8K answer agrees: 393.3 tok/s here, 392.7 there.
- The burst's text: one leg of N differs on two of its four answers. Inference: which rounds run at which width depends on when each request arrives, as #680's burst legs showed within one build, one differing leg per arm. Byte identity at a fixed width is held by the round identity test, not by this cell.
- FP8 against BF16 decode at equal text was not measured served; #678's share of a one-lane verify round was not separated; a verify round of two lanes or more runs its exception products on W8A8, which #678 does not reach.
- Not measured for speed: the MTP loads (their verify rounds keep the synchronous return), the trellis bodies (held by the identity test only), and any card but the RTX 5090. The deferred verify round is on every card's DFlash2 load; the copy-free graphs and the gating branch are on compute capability 12 only, the RTX 30 and 40 cards keep main's graphs.

## Follow-ups

- The RTX PRO 6000 run named above.
- #669: the plain round's deferred return.
- #666: the draft window past 7, which reuses the round identity test's window parameter.
