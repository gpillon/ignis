# On the RTX 5090 the 27B's GDN prefill runs 2.4-2.6x faster on ignis's kernels with its KLD unmoved, and the three tests that moved when Blackwell first took them read numerics, not a fault

- Kind: experiment
- Status: current
- Observed: 2026-11-08
- Last verified: 2026-11-08
- Scope: kernel / the 27B's GDN prefill route per GPU generation (`ignis::gdn_prefill::runs_by_default`, `IGNIS_GDN_PREFILL`); tests / the entailment regression set's tiers, the multimodal speculative near-tie, the shared vision workspace's recorded tokens, the VRAM plan's pin; speed / cold prefill and TTFT of the 27B on the RTX 5090
- Related: [spec #531](https://github.com/gpillon/ignis/issues/531), [GitHub #536](https://github.com/gpillon/ignis/issues/536), [GitHub #535](https://github.com/gpillon/ignis/issues/535), [GitHub #534](https://github.com/gpillon/ignis/issues/534), [GitHub #514](https://github.com/gpillon/ignis/issues/514), [ADR 0080](../adr/0080-on-sm8x-the-trellis-prefill-decodes-each-matrix-once-per-chunk-and-multiplies-on-the-integer-mma.md), [ADR 0016](../adr/0016-step-abi-extensible-options.md), [ADR 0007](../adr/0007-performance-gate-not-parity.md)
- Superseded by: none

## Question

The 27B's GDN prefill has two routes (ADR 0080): ignis's own kernels, the
RTX 30 and RTX 40 cards' route, and the vendored chunked kernels, which
Blackwell kept. #514's rule moves Blackwell to ours only if the RTX 5090
measures them faster with the KLD against BF16 holding. The move was made once
(5b49367c) and reverted (c23f90e0), because three tests of the GPU profile
moved: two gate cases of the entailment regression set
(`decide_entailment_gpu`), the near-tie probe of the multimodal speculative
test (`vision_dflash2_gpu`), and the shared vision workspace's recorded tokens
(`vision_shared_workspace_gpu`). On the RTX 5090, is ours faster, and is any
of the three moves a quality loss of ours?

## Evidence

Everything below ran on one RTX 5090. Raw material is in the main
checkout's untracked `.scratch/`.

### Accuracy and quality (2026-11-05, #514)

- **Against an fp64 recurrence**, on the leaf test's synthetic inputs
  (`kernel/tests/test_gdn_prefill.cu`): output rows 1.82e-3 RMS for ours
  against 5.1e-3 for the vendored kernels, the worst row 3.7e-3 against 0.024;
  states 6.4-8.7e-4 against 3.6-4.4e-3 (`.scratch/503/report-514.md`).
- **Full-vocabulary KLD against BF16**, on Blackwell's default routes (the
  fused prefill, the FP8 band), one binary for both routes:

  | entry | vendored | ours | ratio (interval) | top-1, points |
  |---|---:|---:|---|---:|
  | 16 GB trellis | 0.04034 | 0.04049 | 1.0036 (0.9926-1.0167) | -0.02 |
  | NVFP4 | 0.05221 | 0.05124 | 0.9816 (0.9392-1.0167) | +0.00 |

  On NVFP4 the two routes' argmax agree at 93.23% of positions. Raw:
  `.scratch/503/kld/results/` (`paired-B1b.json`, `paired-N1.json`).

### Speed

- **The kernels** (#514's leaf bench, `ignis_kernel_gdn_prefill_bench`,
  2026-11-05): per call 2.44x faster at 2,048 tokens, 2.47x at 4,096, 2.57x at
  8,192; the 48 GDN layers of an 8K prompt in 2,048-token chunks 66.0 ms
  against 27.0, of a 64K prompt 528 against 216
  (`.scratch/503/legs/f2817ebb-gdn-bench.log`).
- **TTFT on the 16 GB trellis entry** (2026-11-05, a build of spec #503's
  integration branch, `ignis-bench ttft --cells 8192,65536`, 10 cold samples
  a leg, legs vendored, ours, vendored, ours), medians in ms:

  | cell | V1 | O1 | V2 | O2 |
  |---|---:|---:|---:|---:|
  | 8K | 2,309.4 | 2,210.3 | 2,292.0 | 2,209.2 |
  | 64K | 19,520.4 | 18,773.4 | 19,304.9 | 18,714.8 |

  Ours 3.6-4.3% shorter at 8K and 3.1-3.8% at 64K
  (`.scratch/503/ttft/ab1-*` and `ab2-*`).
- **TTFT on the NVFP4 entry** (2026-11-07, `main` b13dbc05, release, the
  plan and its rule written before the legs, `.scratch/gdn-blackwell/plan-speed.md`):
  `make start` on the `rtx5090` profile's defaults with
  `MAX_CONTEXT=524288`, `ignis-bench ttft --cells 8192,65536 --samples 5`,
  all cold, legs vendored, ours, ours, vendored, medians in ms:

  | cell | V1 | O1 | O2 | V2 | pooled V | pooled O | gap |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | 8K | 762.0 | 718.3 | 759.2 | 793.1 | 780.6 | 745.5 | 35.1 (4.5%) |
  | 64K | 6,788.2 | 6,477.0 | 6,665.6 | 6,959.9 | 6,881.4 | 6,612.9 | 268.5 (3.9%) |

  The registered rule had two parts. Each ours leg below both vendored legs
  holds at both cells. The gap at least twice the gap between a route's two
  legs fails at both (35.1 against 2 x 40.9 ms, 268.5 against 2 x 188.6): a
  step between O1 and O2, the same for both routes (vendored +31 and +172 ms,
  ours +41 and +189 ms at 8K and 64K), flat inside each half. The two pairs
  that each sit inside one half, V1-O1 and O2-V2, agree within 10 ms (8K 43.7
  and 33.9, 5.7% and 4.3% of their vendored leg; 64K 311.2 and 294.3, 4.6%
  and 4.2%) and match the leaf bench's saving (39 ms and 312 ms). The second part counted a shift both routes shared as noise;
  it should have been written on the pairs (`.scratch/gdn-blackwell/results.md`,
  `speed/`).
- **This change's legs** (2026-11-08, the spec's integration branch at
  99fec96f: `main` 1188385c with #535 and #534's first pass, which change
  tests only; release). `make start` with the NVFP4 artifact,
  `MAX_CONTEXT=524288 IDLE_UNLOAD=0 LOOP_GUARD=off`, the `rtx5090` profile's
  defaults otherwise (2,048-token chunk, DFlash2 with 7 drafts, the FP8 band,
  the fused prefill); the route set by `IGNIS_GDN_PREFILL` and checked in each
  load's `prefill routes` line. The README's cold-prefill cells
  (`.scratch/fn-prefill-cells-2026-10-13/client.py`: prompt tokens over TTFT,
  `max_tokens` 4, thinking off, streamed, each slice its own nonce, no cached
  token), one launch per route, ours first, in tok/s:

  | cell | ours a | ours b | ours | vendored a | vendored b | vendored | ours against vendored |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | 14K | 12,209 | 12,046 | 12,128 | 11,497 | 11,512 | 11,505 | +5.4% |
  | 46K (45,612 tokens) | 11,341 | 11,347 | 11,344 | 10,848 | 10,796 | 10,822 | +4.8% |
  | 105K (105,272 tokens) | 9,008 | 9,008 | 9,008 | 8,674 | 8,671 | 8,672 | +3.9% |

  The vendored leg was run only to attribute the move: the README's 9,717 and
  7,865 tok/s (2026-10-21) had already moved on `main` to 10,822 and 8,672 on
  the vendored route. On ours, `ignis-bench ttft --cells 8192,65536 --samples 5`,
  all cold: 721.2 ms at 8K and 6,348.3 ms at 64K, medians of 5. Raw:
  `.scratch/531/536/speed/`, `speed-vendored/`, summary `speed.md`.

### The entailment regression set

- **A fragility control** (2026-11-07, `main` b13dbc05, release; plan and
  decision rule written before the arms, `.scratch/gdn-blackwell/plan.md`).
  The live server with the decide test's NVFP4 artifact, `SPEC=off
  IDLE_UNLOAD=0 LOOP_GUARD=off`, a fresh server per arm, the 18 cases through
  `POST /v1/decide`, logit = ln(p/(1-p)):

  | arm | route | labels right | mean abs change vs A | toward right / wrong | net logit a case | labels flipped vs A |
  |---|---|---|---:|---|---:|---|
  | A | GDN vendored, chunk 1024 | 17/18 | - | - | - | - |
  | B | GDN ours, chunk 1024 | 14/18 | 1.313 | 5 / 12 | -0.95 | 12, 16, 17 (a gate) |
  | C | GDN vendored, chunk 128 | 17/18 | 0.000 | - | 0 | none |
  | D | GDN vendored, int8 prompt-attention band | 15/18 | 1.486 | 5 / 11 | -1.14 | 12, 10 (a gate) |

  The rule: the fixture is fragile if C or D, each a change that passes the
  KLD gate, moves the 18 by a mean absolute change of at least half of R
  (R, ours against A, 1.31) or carries gate 3 or 17 across 0; ours is
  disproportionate if both stay under a quarter of R and neither flips a gate
  case. D moved them by 1.49 and lost gate 10 (+1.25 to -2.50): fragile. C
  was bit-identical to A (the decide prompts already run as 2-3 jobs at chunk
  1024), so it carried no information beyond showing that a fresh load
  reproduces A exactly. B against D: mean absolute change 1.465, 10 cases
  toward the right label and 8 toward the wrong one, net +0.19. Gate 17 read
  +0.25 on the vendored route itself, under the one logit its tier asked for.
  Raw: `.scratch/gdn-blackwell/` (`A.json` to `D.json`, `stats.json`,
  `results.md`).
- **Five arms** (2026-11-08, `main` 1188385c; plan written before the arms,
  `.scratch/531/tiers/plan.md`): the test harness `decide_entailment_gpu`
  under each GDN route (`h-`), and the live server at chunk 1024 with
  `SPEC=off` under the vendored kernels, under ours and under the int8 band on
  the vendored kernels (`s-`). The arm columns are the logits as measured
  (negative reads no); the minimum is over the signed margins, the logit when
  the right label is yes and minus it when no:

  | case | right label | h-vendored | h-ours | s-vendored | s-ours | s-int8 | minimum | tier by the rule |
  |---|---|---:|---:|---:|---:|---:|---:|---|
  | 1 | yes | +1.750 | +1.250 | +3.625 | +4.125 | +2.250 | +1.250 | gate |
  | 2 | no | -7.688 | -7.500 | -6.125 | -5.000 | -3.250 | +3.250 | gate |
  | 3 | yes | +1.500 | -1.250 | +1.625 | +1.375 | +1.125 | -1.250 | watch |
  | 4 | no | -2.000 | -2.750 | -3.125 | -3.875 | -3.125 | +2.000 | gate |
  | 5 | no | -6.250 | -5.875 | -6.125 | -4.625 | -7.000 | +4.625 | gate |
  | 6 | no | -4.250 | -2.500 | -3.500 | -3.875 | -2.625 | +2.500 | gate |
  | 7 | no | -3.875 | -3.750 | -6.750 | -4.750 | -6.750 | +3.750 | gate |
  | 8 | no | -3.250 | -4.125 | -2.250 | -3.625 | -1.750 | +1.750 | gate |
  | 9 | no | -4.625 | -4.750 | -5.000 | -4.000 | -3.875 | +3.875 | gate |
  | 10 | yes | +1.000 | +2.375 | +1.250 | +1.500 | -2.500 | -2.500 | watch |
  | 11 | no | -6.625 | -3.500 | -8.500 | -4.500 | -4.125 | +3.500 | gate |
  | 12 | yes | -3.125 | +0.750 | +0.375 | -0.500 | -3.125 | -3.125 | watch |
  | 13 | yes | -0.625 | -0.875 | -1.250 | -1.250 | -1.000 | -1.250 | known failure |
  | 14 | no | -4.750 | -6.000 | -4.875 | -4.750 | -5.875 | +4.750 | gate |
  | 15 | no | -5.500 | -4.250 | -7.500 | -5.125 | -5.625 | +4.250 | gate |
  | 16 | yes | -1.500 | +0.000 | +0.000 | -1.125 | +0.375 | -1.500 | watch |
  | 17 | yes | +0.125 | -1.375 | +0.250 | -2.000 | +0.875 | -2.000 | watch |
  | 18 | yes | +1.750 | +2.375 | +3.875 | +0.125 | +1.000 | +0.125 | watch |

  The three server arms equal the 2026-11-07 control's A, B and D in every
  case. In the harness both routes have 15 of 18 labels right; ours failed
  only on gates 3 and 17 of the tiers then in force. The measurement is
  recorded in
  `crates/server/tests/fixtures/entailment/cases.json` (#535); raw
  `.scratch/531/tiers/`.

### The multimodal split (2026-11-07)

A diagnosis on `main` b13dbc05 with debug instrumentation that was not
committed (`.scratch/gdn-blackwell/diagnosis/instrumentation.patch`): the
NVFP4 artifact, BF16 KV, DFlash2 with 7 drafts, vision on the device. The
split: the prompt `colour/detail`, 90 tokens with a rotation offset of -56;
its 4th emitted token, at KV position 92 and rotation 36, is column 2 of the
first verify round (base 90, extent 7, two drafts accepted). The decode
round picks 2099, the verify round 8851. Every gap is logit[2099] -
logit[8851]:

| computation of position 92 | ours | vendored |
|---|---:|---:|
| decode round, 1 column (the speculation-off pick) | +3.25 | 0.000 (an exact tie, the lower id wins) |
| verify round's column 2, 8 columns (the speculation-on pick) | -0.25 | -1.125 |
| a chunked prefill span 8 columns wide over the decode text, row 2 | -0.25 | -1.125 |
| the same span 3 columns wide (attention, MLP, residual on A16) | +1.50 | +0.375 |
| 4 columns (the attention input crosses to W4A4) | +1.625 | -0.125 |
| 5 columns (the MLP gate-up crosses) | +1.50 | +1.00 |
| the test's probe: the whole prompt as one span, then one token at a time | +3.25 | 0.000 |
| the test's probe: 48-token spans, then one token at a time | +4.375 | +4.375 |
| decode and verify both held to A16 | +1.625 / +1.50, no split, 48 of 48 tokens equal | not run |

- **The mechanism.** The vendored NVFP4 dispatch's token thresholds, the
  reference policy (ADR 0016): at 8 columns the verify round multiplies the
  attention input (from 4 columns), the MLP gate-up (from 5) and the attention
  output and MLP down projections (from 8) with FP4-quantized activations
  (W4A4); the 1-column decode round keeps them on A16. The W4A4 activation
  scale is each weight's static divisor, not a per-call maximum, so the
  columns do not couple through it and the 8-wide span is exact for row 2.
- **The rotation.** On every matched column of every round, over 70 image
  positions, the verify round's rotation positions equal the decode round's.
- **The population.** The speculation-off text replayed through
  teacher-forced verify rounds, every position whose history matches compared
  with the decode row:

  | | positions | abs gap change, median / p95 / max | sd of the difference over the decode top 32, median | argmax splits |
  |---|---:|---|---:|---:|
  | image, default policy | 55 | 0.375 / 2.0 / 3.5 | 0.445 | 1 (the split) |
  | text, default policy | 94 | 0.375 / 1.0 / 1.5 | 0.391 | 10 |
  | image, both on A16 | 55 | 0.000 / 0.25 / 0.25 | 0.047 | 0 |
  | text, both on A16 | 94 | 0.000 / 0.125 / 0.25 | 0.046 | 3, each at a decode margin of 0.125 or less |

- **A text control.** `textctl/city`, no image: speculation on and off part
  at token 10 (13668 against 26052), decode +0.375, verify -0.25. Its prompt
  is 23 tokens, inside one 48-token span, so the test's two probes are one
  computation and both read +0.375.
- **Faults ruled out.** A wrong multimodal rotation in the verify round
  (positions equal; under A16 no split); a wrong starting state at the
  handoff from prefill to the verify round (column 0's gap differs from the
  decode round's by 0.06 under A16); a staging or column fault in the verify
  graph (the 8-wide chunked span gives the verify round's two logits and its
  argmax exactly, on both GDN routes).

Raw: `.scratch/gdn-blackwell/diagnosis/` (`report.md`, the `r1` to `r4` logs
and their analyses).

### The recorded fixtures (2026-11-08, `main` 1188385c's engine, ours)

- **The shared vision workspace's tokens** diverged at token 8 under ours in
  #503's profile (`.scratch/gdn-blackwell/plan.md`). Re-recorded under ours
  with `IGNIS_RECORD_VISION_WORKSPACE_TOKENS=1`, 32 of the fixture's 64 lines
  change; a second run without recording passed on the new record
  (`.scratch/531/536/notes.md`, `vsw-record.log`, `vsw-check.log`).
- **The VRAM plan.** `vram_plan_gpu` under ours reads the NVFP4 load's
  prefill scratch at a 4,096-token chunk, vision bounded, as 1,487,132,928 B,
  the GQA layer's, against 1,494,423,552 B on the vendored GDN layer; its other
  four tests pass (`.scratch/531/536/vram-plan.log`).

## Finding

Observed:

- On the RTX 5090 ours is faster than the vendored kernels: the GDN prefill
  kernels 2.4-2.6x; the time to first token at 8K and 64K 3-4% shorter on the
  16 GB trellis entry, and 4.2-5.7% on the NVFP4 entry within each regime of
  the alternated legs (pooled over the four legs, 4.5% at 8K and 3.9% at 64K);
  the cold prefill of the README's cells 5.4%, 4.8% and 3.9% faster at 14K,
  46K and 105K on the NVFP4 default.
- The full-vocabulary KLD against BF16 does not move outside its interval on
  either 27B body, and ours is closer to an fp64 recurrence.
- The entailment regression set moves under any change of the numerics that
  passes the KLD gate: the int8 band moved its 18 cases more than ours did
  (1.49 against 1.31 logits on average) and lost a gate case of its own. A
  gate taken as the minimum over the five arms holds cases 1, 2, 4-9, 11, 14
  and 15; 3, 10 and 17 are watch cases there.
- The multimodal split is the round shapes' numerics: an independent chunked
  span of the verify round's width computes position 92 as the verify round
  did, and with both rounds on one activation route the text is identical.
  Both of the old test's probes took the decode round's shape; under the
  vendored kernels the test passed because one of them read an exact tie,
  and on a short text prompt the two probes are one computation.
- The recorded tokens and the VRAM plan's pin move with the route by
  construction: the tokens with any change of the numerics, the pin with the
  prefill scratch, 7,290,624 B smaller on ours.

Inferred:

- The drift of every perturbation toward the wrong label (B and D alike) is
  selection rather than quality: the set was written, corrected and tiered on
  the vendored route, so its cases sit where that route is right. B against
  D, two perturbations of A, is balanced.
- The step between the NVFP4 legs O1 and O2 is a change of the card's state,
  thermal or clock, shared by both routes; it was not measured.
- The RTX PRO 6000 and the 16 GB RTX 50 cards run the RTX 5090's sm_120a
  machine code, so the same route; their speed is not measured.

## Implications

- `ignis::gdn_prefill::runs_by_default` is true on compute capability 8.x and
  12.x (#536): every card that runs the 27B prefills its GDN layers on ours.
  `IGNIS_GDN_PREFILL=vendored` stays the measurement switch back on every
  card, and the vendored kernels stay the leaf test's oracle. Flash-Next
  keeps its own vendored GDN layers.
- A case of a regression set is a gate only when it is clear of the decision
  line on every route measured (#535, `GLOSSARY.md`: Regression set).
- A split between a decode round's pick and a verify round's is judged by the
  two round shapes (#534, `GLOSSARY.md`: Near-tie, Round shape).
- On Blackwell a greedy answer can differ from an earlier release's at a
  near-tie, since correctness is self-checked, not byte-matched (ADR 0007);
  the full GPU profile is what shows nothing else moved.
- The README's cold prefill reads 11,344 tok/s at 46K and 9,008 at 105K; this
  change adds 4.8% and 3.9% of it, and the rest of the move from 9,717 and
  7,865 came from earlier changes on `main`, not measured apart.

## Limits and unknowns

- One card, the RTX 5090. Speed is not measured on the RTX PRO 6000 or on the
  16 GB RTX 50 cards (5060 Ti, 5070 Ti, 5080).
- This change's cold-prefill legs are one launch per route in one session,
  and its TTFT cells ran on ours alone; the NVFP4 TTFT comparison is the
  2026-11-07 legs on `main` b13dbc05, which failed the second part of their
  own registered rule. The trellis TTFT and the KLD legs ran on spec #503's
  build of 2026-11-05; `main` has changed outside the GDN kernels since.
- The diagnosis followed one split of one image prompt. Near the image the
  spread between the activation routes widens (positions 90 to 92: sd 1.1-1.3
  over the decode top 32, a largest per-token difference of 3.4-4.0), decaying
  toward the text's over the answer; why is not measured (larger activation
  outliers after image tokens, making the FP4 quantization coarser there, is
  a possibility).
- Under the default policy the 1-column decode round (+3.25) and a 3-column
  span (+1.50) differ although both keep attention, MLP and residual on A16;
  under A16 the decode round reads +1.625. The likely cause, the GDN input
  projection's W4A4 route at every width, was not investigated.
- No run of the multimodal acceptance under #534's rule is recorded here;
  those legs are #534's.
- The entailment set's cases are themselves under audit (#537); its tiers
  follow from re-measuring after any rewording.

## Follow-ups

- [GitHub #537](https://github.com/gpillon/ignis/issues/537): the entailment
  set's audit, case by case, and its tiers re-measured.
- A multimodal acceptance with both rounds on one activation route, which
  needs a test-only compute policy on the decode and verify interface (spec
  #531's Out of Scope).
- Flash-Next's own GDN layers (spec #531's Out of Scope).
- [GitHub #490](https://github.com/gpillon/ignis/issues/490): the dense
  prefill and the u8 x s8 band on Blackwell.
