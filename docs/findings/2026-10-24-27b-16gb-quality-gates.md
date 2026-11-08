# The 27B's 16 GB artifact: the quality gates

- Kind: experiment
- Status: current
- Observed: 2026-10-24
- Last verified: 2026-10-24
- Scope: quality / the 27B's trellis body (`gpillon/Qwen3.8-27B-trellis-3bpw-ignis`) through ignis's engine path against
  the served NVFP4 artifact and BF16: KLD on the 2026-09-24 suite (gate 1), MMLU-Pro through the engine's logits (gate 2),
  DFlash2 tokens per round on a teacher-forced replay (gate 3), the reasoning-loop rate (gate 4), per-domain KLD
- Topic: artifacts-and-gates
- Related: [GitHub #340](https://github.com/gpillon/ignis/issues/340),
  [spec 27b-16gb/05](../specs/27b-16gb/05-gates-and-simulation.md) (Part B, Acceptance 1-4),
  [the 16 GB artifact: first light](2026-10-23-27b-16gb-first-light.md),
  [EXL3 3 bpw on our text and DFlash2](2026-10-21-exl3-3bpw-27b-on-our-text-and-dflash2.md),
  [KLD against BF16](2026-09-24-kld-against-bf16.md),
  [reasoning loops are the model's](2026-10-12-reasoning-loops-are-the-models.md),
  [ADR 0052](../adr/0052-a-reasoning-loop-is-taken-back-and-nudged.md),
  [ADR 0060](../adr/0060-the-27b-has-a-trellis-body-for-small-cards.md)
- Superseded by: none
- Summary: The 27B's 16 GB trellis artifact passes all four gates on the 5090: KLD 0.0403, 0.77x the NVFP4's at chunk 1,024 (limit 1.2x), MMLU-Pro 65.84% against BF16's 67.97%, loops 9/32 against 14/32, DFlash2 tokens per round 0.996x.

## Question

The trellis body measured 0.0340 KLD weights-only in the study's PyTorch harness (M1). Once it runs through ignis's own
kernels, hq-e8-2b KV, the host embedding and DFlash2, does it still clear the owner's four gates of 2026-10-21: KLD at most
1.2x the served NVFP4's, MMLU-Pro not significantly below BF16, DFlash2 tokens per round at least 90% of the NVFP4
target's, and a loop rate no worse than the NVFP4's?

## Evidence

Main at 711b1d85 (release build, v0.7.3) for gates 1, 2 and 4, the branch `340-replay` on it for gate 3; RTX 5090 on
PCIe Gen 3, 2026-10-24, the desktop alone on the card before and after. Every trellis run loads as a 16 GB card would
(owner, 2026-10-24): budget 17,179,869,184 bytes, profile `rtx50-16gb`, 262,144 context, 2 lanes, chunk 1,024, hq-e8-2b.
Raw logs, JSON, scripts: `.scratch/340/quality/`, `.scratch/340/loops/`, `.scratch/340/replay-run/` (untracked); the KLD
dumps were deleted once scored (`.scratch/340/watchdog.log`; the per-position results are in
`.scratch/340/quality/results/`).

**The 16 GiB rule for the KLD dumps.** `span_logits` takes no budget. Against the server's `rtx50-16gb` load (#339's
`ignis.model.loaded`) it has the same KV (hq-e8-2b), chunk (1,024), rope (none), prompt attention (8-bit) and exceptions
(BF16: a no-op on the trellis body). It differs in `max_context` (the longest window plus 64: 4,160 on set A, 2,112 on set
B, against 262,144), in lanes (8 against 2), and in having no drafter and no vision. `max_context` feeds only the
decode-graph envelope (`kernel/src/gqa_layer.cu:514-523`); the prefill envelope is start position plus tokens
(`gqa_layer.cu:254-256`), so the dumped distributions never see it, and lanes and the drafter are not on the prefill
path. This is read from the code, not measured by a run at 262,144. The VRAM samples are
`a-trellis-vram.txt` and `b-trellis-vram.txt`.

### Gate 1 (Acceptance 1): KLD on set A, 16 windows of 4,096 tokens (6 code, 4 prose, 6 chat), one `ref_kld.py` run

| arm | KLD all | code | prose | chat | top-1 | NLL (ppl) |
|---|---:|---:|---:|---:|---:|---:|
| trellis, chunk 1,024 | 0.04034 | 0.03877 | 0.04941 | 0.03587 | 93.42% | 1.0072 (2.738) |
| NVFP4, chunk 1,024 (the spec's serving shape) | 0.05221 | 0.05043 | 0.06958 | 0.04239 | 92.41% | 1.0117 (2.750) |
| NVFP4, chunk 2,048 (served since #341) | 0.04916 | 0.04764 | 0.06404 | 0.04075 | 92.58% | 1.0068 (2.737) |
| BF16 reference | | | | | | 0.9894 (2.690) |
| M1, weights only | 0.0340 | 0.0321 | 0.0415 | 0.0310 | 93.88% | (2.733) |
| NVFP4 serving, 2026-09-24 | 0.0513 | | | | 92.45% | (2.752) |

- trellis / NVFP4 at chunk 1,024 = 0.7728 (95% CI [0.7368, 0.8144], window bootstrap), the trellis lower in 16 of 16
  windows; per kind chat 0.846, code 0.769, prose 0.710.
- trellis / NVFP4 at chunk 2,048 = 0.8207 (CI [0.7903, 0.8582]), lower in 15 of 16 windows (it loses window 12, a chat
  window, 0.03670 against 0.03555); per kind chat 0.880, code 0.814, prose 0.772.
- The coordinator scored the NVFP4 twice in the same reference run, because the served chunk moved from 1,024 to 2,048
  after the spec was written; the trellis had to clear 1.2x of both.
- Identity checks, all exact per position: the trellis windows 0 and 1 equal #339 A3's (the reference NLL too); both
  NVFP4 arms equal #341's dumps on all 16 windows (windows 0 and 15 of chunk 1,024 sha256-equal to `Y:/kld-341/c1024`).
- The 0.0513 row (2026-09-24) predates the prefill attention of #335; the same arm since #335 is 0.05221, reproduced here.
- Rope: `span_logits` loads `RopeScaling::NONE` (`crates/core/src/model_load.rs:679`), so neither NVFP4 dump has YaRN,
  while the served NVFP4 runs `yarn:3` at 786,432. Both NVFP4 arms are therefore the NVFP4's best case, the stricter
  baseline for the trellis.

### Gate 2 (Acceptance 2): MMLU-Pro on set B, n = 281

Set B is 66 windows of 2,048 tokens, the Flash-Next study's held-out chunks, scored through the engine's logits at the 281
MMLU-Pro answer marks. The BF16 reference was run through the same marks: 67.97%
(M1's BF16 68.33%; 276 of 281 BF16 answers equal M1's). Pre-registered by the coordinator: pass iff exact McNemar p > 0.05
against BF16; on a fail, stop and report, no 3.25 bpw work.

| arm | accuracy | against BF16: BF16 right / arm wrong (b), arm right / BF16 wrong (c) | exact McNemar p | same letter as BF16 |
|---|---:|---|---:|---:|
| BF16 | 67.97% | | | |
| trellis | 65.84% | b 15, c 9 | **0.3075** | 236 / 281 |
| NVFP4, chunk 2,048 | 66.55% | b 18, c 14 | 0.5966 | 231 / 281 |
| M1, weights only | 64.77% | b 16, c 6 | 0.0525 | |

Trellis against NVFP4: b 14, c 16, p 0.8555. The trellis's answers equal M1's on 272 of 281. Ties at the marks: BF16 6, trellis 9, NVFP4 5. Per category counts: `gate2-analysis.txt`.

**Per-domain KLD of set B** (mean KLD / top-1; M1's KLD in the last column; `gate2-analysis.txt`):

| domain | positions | trellis | NVFP4 (chunk 2,048) | M1 |
|---|---:|---|---|---:|
| code | 8,192 | 0.0358 / 93.12% | 0.0470 / 92.08% | 0.0335 |
| prose | 4,096 | 0.0468 / 88.48% | 0.0644 / 87.11% | 0.0425 |
| chat | 4,096 | 0.0199 / 95.61% | 0.0284 / 95.21% | 0.0181 |
| en | 8,192 | 0.0415 / 91.04% | 0.0520 / 90.06% | 0.0408 |
| it | 8,192 | 0.0437 / 90.91% | 0.0580 / 89.56% | 0.0431 |
| zh | 8,192 | 0.0493 / 88.76% | 0.0474 / 88.35% | 0.0480 |
| de | 8,192 | 0.0604 / 90.43% | 0.0706 / 89.47% | 0.0601 |
| ja | 8,192 | 0.0422 / 90.26% | 0.0525 / 89.32% | 0.0413 |
| math | 8,192 | 0.0304 / 95.26% | 0.0267 / 95.42% | 0.0293 |
| python (`py`) | 8,192 | 0.1680 / 92.94% | 0.1659 / 92.91% | 0.1667 |
| mmlu | 58,250 | 0.9631 / 81.21% | 1.0549 / 79.76% | 0.9565 |
| all but mmlu | 73,728 | 0.0561 / 91.64% | 0.0629 / 90.93% | |
| all but mmlu and py | 65,536 | 0.0421 / 91.48% | 0.0501 / 90.68% | |

The mmlu row's mean KL is hypersensitive (flagged so by M2), so it is not a
quality number; the accuracy table above is what the marks measure.

### Gate 3 (Acceptance 3): DFlash2 tokens per round on the teacher-forced replay

A free read from #339's logs (1-lane greedy canaries, same build, 256 tokens a side) gave 4.41 against 4.92 tokens per
round (52 against 58 rounds for the same 256 tokens), a ratio of 0.897: too small a sample to decide, so the owner chose the spec's instrument and the #267
teacher-forced replay (branch `dflash2-sampled-acceptance-267`, [its finding](2026-09-24-dflash2-sampled-acceptance.md))
was ported onto main (branch `340-replay`, `tools/dflash2-replay/`, the examples in `crates/runtime/examples/`).

- **Method.** #267's 200 recorded agent turns (116 token streams, 6.47 M prefilled context tokens, 208,860 replayed output
  tokens, sampled by ninfer at T = 1.0, top_p 0.95, top_k 20: [#267's finding](2026-09-24-dflash2-sampled-acceptance.md)) walked through real verify rounds that commit the recorded
  text: the leading drafts it agrees with, then the recorded token. Tokens per round = output tokens / rounds, pooled and
  with #267's traffic weights; a bootstrap over the 116 streams (10,000 resamples) for the interval.
- **Loads, one build.** The trellis leg is the 16 GiB load (the plan printed equals #339 A1: 7,956 pages, 17,179,375,716 of
  17,179,869,184). The NVFP4 leg uses the same engine shape (262,144, rope none, chunk 1,024, vision off), 8 lanes on the
  card's derived budget: #267's shape, not the served 786,432 / `yarn:3` (the NVFP4's worse case). A 2-lane NVFP4 smoke
  wrote records byte-identical to the 8-lane one's, so the legs differ in lanes only and the lanes do not move a round.
- **Checks.** `dflash2_replay_seams_gpu.rs` passed on both bodies (forced rounds equal live ones bit for bit, greedy and
  sampled, widths 1 and 2; the readouts refuse a window that is not the load's; an out-of-vocabulary forced successor is
  refused with the sequence unmoved). 0 readout disagreements on either leg. The NVFP4 leg reproduces #267's 2026-09-24
  run: 3.383 against 3.386, every source within 0.3%.

| set | turns | trellis | NVFP4 | trellis / NVFP4 | #267 (NVFP4, 2026-09-24) |
|---|---:|---:|---:|---:|---:|
| all | 200 | 3.370 | 3.383 | **0.996** | 3.386 |
| A (opencode, complete contexts) | 50 | 3.007 | 3.007 | 1.000 | 3.016 |
| B (qwen-code main chats) | 75 | 3.483 | 3.500 | 0.995 | 3.508 |
| C (qwen-code subagents) | 75 | 3.389 | 3.403 | 0.996 | 3.400 |
| think | 101 | 3.246 | 3.260 | 0.996 | 3.259 |
| content | 99 | 3.933 | 3.941 | 0.998 | 3.963 |
| traffic-weighted | | 3.438 | 3.452 | **0.996** | 3.455 |

- Rounds: trellis 61,978, NVFP4 61,739 (#267: 61,692). Bootstrap 95% interval of the pooled ratio [0.993, 0.999]; per
  stream median 0.996, p10 0.965, p90 1.047, none below 0.90.
- P(accepted >= i), i = 1..7: trellis .712 .508 .370 .276 .210 .164 .131, NVFP4 .714 .511 .373 .279 .211 .164 .131.
- The texts are the NVFP4's (ninfer sampled them on that artifact), so the replay leans against the trellis: 1.20% of the
  recorded tokens fall outside the trellis's own sampling support, 0.84% outside the NVFP4's. With each body's own accept
  distribution on the same drafts (`analyze.py`'s exact expectation) the ratio is 3.420 / 3.397 = 1.007.
- Wall time on the 5090: the trellis leg 58 min (prefill 32.8, rounds 25.2, 17.2 ms a round), the NVFP4 leg 40 min.
- Evidence: `.scratch/340/replay-run/handoff.md` (`score.py`, `score.txt`, `validate-*.txt`, `analyze-*.txt`), the
  records on `H:/replay-340/`; the port and its review fixes: `.scratch/340/replay-port/handoff.md`,
  `.scratch/340/replay-fixes/handoff.md`.

### Gate 4 (Acceptance 4): loop rate, four prompts x seeds 101-108

Same build, loop guard at its default (rewind on), the bodies of `body.json` (the Rome prompt), `body-anni`,
`body-nobel-en` and `body-math-en`, seeds 101-108, non-streaming. The trellis served 2 lanes (`WORKERS=2`), the NVFP4 8
(`WORKERS=8`): by necessity of each load; the seeds fix the sampling, the batching can still change numerics.

Loads (`plan-trellis.txt`, `start-*.out`): trellis, `make start ARTIFACT=<trellis> VRAM_BUDGET=17179869184 IDLE_UNLOAD=0
ARGS="--profile rtx50-16gb"`: 2 lanes, hq-e8-2b, 262,144, chunk 1,024, rope none, DFlash2 7 / full head, 7,956 pages,
17,179,375,716 of 17,179,869,184 allocated. NVFP4, `make start IDLE_UNLOAD=0 MAX_CONTEXT=786432 ARTIFACT=<nvfp4>`: 8
lanes, hq-e8-2b, 786,432, chunk 2,048, rope `yarn:3`, DFlash2 7.

**The loop call**, fixed before scoring and applied blindly to both artifacts: a request is a loop if the guard rewound it
(`ignis.loop_guard.rewound`), or its reasoning holds a 6-gram at least 25 times, or its worst 3,000-character zlib window
compresses below 0.20. Candidates are in `cand.txt`; the 0.20-0.30 band without a rewind was not counted, and the tails
were read by eye on a sample, not on all 64. Verdict rule (coordinator): fail iff the trellis is significantly worse, the
one-sided Fisher exact p < 0.05 on the pooled counts.

| prompt | trellis loops | rewound | missed by the guard | forced close | NVFP4 loops | rewound | missed | forced close |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rome (`body.json`) | 1/8 (102) | 1 | 0 | 0 | 4/8 (102, 103, 105, 106) | 4 | 0 | 1 |
| anni | 0/8 | 0 | 0 | 3 | 1/8 (103) | 1 | 0 | 3 |
| nobel-en | 5/8 (101, 103, 104, 106, 107) | 1 | 4 | 7 | 5/8 (102, 103, 104, 107, 108) | 4 | 1 | 2 |
| math-en | 3/8 (102, 106, 107) | 3 | 0 | 5 | 4/8 (102, 104, 106, 108) | 2 | 2 | 2 |
| pooled | **9/32** | 5 | 4 | 15 | **14/32** | 11 | 3 | 8 |

Finish reasons (trellis / NVFP4): `tool_calls` 23 / 21, `stop` 3 / 2, `length` 6 / 9. No request error and no 503 on either
side. Fisher exact on 9/32 against 14/32: one-sided, H1 "the trellis loops more", p = 0.941; the opposite tail (the trellis
loops less) p = 0.149. Counting only guard rewinds the trellis has 5/32 against the NVFP4's 11/32.

## Finding

Observed:

1. **Gate 1 passes with room.** The trellis KLD is 0.77x of the NVFP4 at chunk 1,024 and 0.82x at chunk 2,048 against a
   limit of 1.2x, lower in 16 of 16 and 15 of 16 windows, with the NVFP4 at its no-YaRN best case. Its KLD is 0.0403 against
   M1's 0.0340 weights-only: the engine path adds 0.006.
2. **Gate 2 passes.** 65.84% against BF16's 67.97%, exact McNemar p = 0.3075 (b 15, c 9). The NVFP4 is at 66.55%, p = 0.5966.
   The trellis and the NVFP4 are indistinguishable on these marks (b 14, c 16, p 0.8555).
3. **Gate 3 passes with room.** On the same 200 recorded turns DFlash2 commits 3.370 tokens a round on the trellis against
   3.383 on the NVFP4: 0.996 (95% interval [0.993, 0.999]) against a floor of 0.90, the same traffic-weighted, and within
   0.8% on every source and kind.
4. **Gate 4 passes.** 9 of 32 against 14 of 32; one-sided p = 0.941 for "the trellis is worse".
5. **The trellis is worse than the NVFP4 only on zh and math on set B** (and python by 1%); on the other domains
   it is lower by 14-30% (mmlu aside). Python's 0.168 is the model's own sensitivity, the same on both bodies and on M1 (0.1667).
6. **Forced close is about twice as frequent on the trellis**: 15 of 32 requests against 8 of 32 (the thinking budget's
   "My thinking time is over" close at 14,336 tokens), driven by nobel-en (7 of 8 against 2 of 8) and math-en (5
   against 2). This is not part of the gate.

Inferred:

- The forced closes are mostly a consequence of the indecision loops the guard misses on the trellis's nobel-en (101, 103,
  106, 107, four missed against the NVFP4's one), which burn the thinking budget; they are the class the 0.7.2 notes
  recorded for nobel-en (101, 105), not a new failure. This was read from the tails, not tested.
- The 0.897 of #339's canaries (256 greedy tokens a side, different texts) was sampling noise rather than the body: the
  replay's 61,978 rounds on the same texts give 0.996.
- The trellis's gate-1 and gate-2 numbers are one distribution, not two: KLD below the NVFP4's and an accuracy within noise
  of it and of BF16 agree, and M1's weights-only lead over the NVFP4 survives the engine's kernels and the hq-e8-2b KV.

## Implications

- All four gates hold: nothing in the owner's quality bar blocks shipping the 16 GB artifact (#346).
- DFlash2 keeps its acceptance on the trellis target, so the drafter grafted byte for byte from the served artifact (ADR
  0060) needs no retraining for the 16 GB body.
- Gate 2's pre-registration means no 3.25 bpw work is triggered.
- A user reading the forced-close rate should not take it for quality: it is the loops' budget cost.
- Gate 1's chunk-2,048 NVFP4 row, not the spec's chunk 1,024, is the honest comparison for the artifact a user runs
  today; both clear.

## Limits and unknowns

- Gate 4 has n = 8 per prompt and 32 pooled: only a large increase (about 9/32 against 20/32) would have been detected.
  The point estimate is lower for the trellis, but the difference is not significant either way; it is not "the trellis
  loops less". The 0.20-0.30 compression band was not counted, and the by-eye tail check covered a sample.
- The trellis ran 2 lanes and the NVFP4 8 in gate 4, and the NVFP4 ran at its served `yarn:3`, 786,432, chunk 2,048:
  batching and rope move numerics, so this is the artifacts as served, not a controlled body comparison.
- The KLD dumps load at a short `max_context`, 8 lanes, no drafter and no vision (read from the code as unobserved by
  prefill, not re-run at 262,144), and the NVFP4 dumps carry no YaRN.
- Gate 2 is 281 marks: a 2-point gap is inside its noise (M1's 3.2-point gap gave p = 0.0525). Ties at the marks (6, 9, 5)
  mean the marks' argmax is sometimes a coin flip.
- Gate 3's recorded texts are ninfer's on the NVFP4 artifact, so the forced successors are not the trellis's own text
  (the bias leans against the trellis; with each body's own accept the ratio is 1.007). The turns stop at 148K of
  context (#267 left out longer ones), and the NVFP4 leg runs 8 lanes and no YaRN, not the served shape.
- DFlash2's acceptance past 150K of context is not measured by any of these gates.
- Gate 4's NVFP4 `run.py` crashed on its final print (cp1252 stdout, U+014D) after all eight responses were written; that
  arm has no SUMMARY line and its forced and finish counts were recomputed from the raw files.

## Follow-ups

- Whether a larger loop sample or the 0.20-0.30 band changes gate 4's reading is not tracked: the point estimate does not
  call for it.
