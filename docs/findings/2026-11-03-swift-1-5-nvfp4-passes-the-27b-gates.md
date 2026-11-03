# Swift 1.5 as NVFP4 passes the fine-tune gates: KLD 0.05355 against a limit of 0.0590, DFlash2 beats off in every cell, 11 of 32 loops

- Kind: experiment
- Status: current
- Observed: 2026-11-03
- Last verified: 2026-11-03
- Scope: models / the `qwen3.8-27b/nvfp4full-swift-1.5` artifact built by `tools/27b-nvfp4-converter/` from `ukisai/Swift-1.5-Qwen3.8-27b`, measured on the RTX 5090 as served (8 lanes, chunk 2,048, DFlash2 7) with `tools/kld-gate`
- Related: [GitHub #470](https://github.com/gpillon/ignis/issues/470), [GitHub #449](https://github.com/gpillon/ignis/issues/449),
  [GitHub #469](https://github.com/gpillon/ignis/issues/469), [GitHub #468](https://github.com/gpillon/ignis/issues/468),
  [2026-11-03 converter finding](2026-11-03-nvfp4-converter-reproduces-the-served-27b.md),
  [ADR 0074](../adr/0074-a-catalog-entry-may-state-its-family-and-an-alias-may-be-another-entrys-id.md),
  [ADR 0078](../adr/0078-the-built-in-catalog-names-each-entrys-license-and-may-ship-restricted-weights.md)
- Superseded by: none

## Question

Does the NVFP4 artifact the converter builds from Swift 1.5 (a post-training of the 27B's checkpoint) hold the quality
and speed bars the served 27B NVFP4 was admitted under (#340)? Ticket #470 asks for gate 1 (KLD against Swift's own
BF16, with the cross-artifact controls), gate 2 (DFlash2 on against off), smokes, and the loop rate, and lists what is
not run.

## Evidence

**The artifact.** `H:/nvfp4-470/qwen3_8_27b_nvfp4full-v2-ukisai-swift-1.5.ninfer` (+ `.conversion.json`), weights id
`nvfp4full-swift-1.5`, model id `qwen3.8-27b`, recipe `qwen3_8_27b_nvfp4full-v2-local`, license
`swift-open-license-1.0`, Swift `ukisai/Swift-1.5-Qwen3.8-27b` at `b4c84d42`. Built with `--gdn-kernel installed` (the
owner's choice after #469's tier 2): calibrate 66 s, run 206 s, pack about 90 s (debug packer), `verify --device cuda`
162 s, VERIFY PASS; VRAM peak 5.66 GiB allocated / 10.28 reserved, host commit 17.4 GiB. Content hash `e5edb4e8...`,
distinct from every artifact on the machine. The DFlash2 (66), vision (333) and frontend (6) objects are SHA-256-equal
to the served `qwen3_8_27b_nvfp4full-v2.ninfer`. The sidecar lists 281 NVFP4 parents, the 34 DFlash2 ones with a null
divisor. It loads as `qwen3.8-27b` (the loader's checksum clean; `ignis.decide.pointing_head point_method=chain`;
`ignis.decide.locate available=false`). Build logs: `.scratch/469/{calibrate-swift,run-swift,pack-swift,verify-swift}.log`.

**Gate 1** (chunk 2,048, set A of 16 windows x 4,096 tokens: 6 code, 4 prose, 6 chat; `tools/kld-gate`).
Raw: `.scratch/449/gates/470/gate1.log`, `gate1.json`, `results/`.

| Pair | KLD | 95% window-bootstrap CI |
|---|---|---|
| Swift NVFP4 given Swift BF16 (the gate) | 0.05355 | [0.04692, 0.06023] |
| served NVFP4 given Swift BF16 (cross_artifact) | 0.05651 | |
| Swift NVFP4 given base BF16 (cross_reference) | 0.05923 | |
| served NVFP4 given base BF16 (baseline, #340) | 0.04916 | |

The limit is 1.2 x 0.04916 = 0.0590. The baseline pair reproduced #340's 0.04916 to the bit. By kind, the gate pair:
chat 0.04798, code 0.04923, prose 0.06837. Against the served artifact's own pair the ratio is 1.089, CI [1.034, 1.160].
The registered rule is the point estimate, so gate 1 is a PASS; the CI's upper end (0.06023) is above the limit.

**Controls.** The matched pair is lower than the cross_artifact term by 0.00296, CI of the difference [0.00129,
0.00534], in 13 of 16 windows; cross_reference is above the matched pair in 16 of 16. The scorer's every-window rule
printed STOP (13 of 16). The ticket's "far above" has no registered number; the owner accepted the control as measured
(CI of the difference above 0, and 16 of 16 on the reference cross term), recorded on #470. The wiring controls all held.

**Gate 2** (DFlash2 on against off, decode tok/s; `.scratch/449/gates/470/cells/gate2.txt`, `b3.py`):

| Context | 1-lane cold, on / off | 2-lane, on / off | 1-lane reuse, on |
|---|---|---|---|
| 8,192 | 368.3 / 81.1 | 570.2 / 153.5 | 353.1 |
| 65,536 | 306.4 / 67.4 | 457.5 / 106.2 | 304.4 |
| 253,440 | 202.4 / 41.0 | 275.4 / 52.2 | 200.0 |

On beats off in every cell (3.71x to 5.27x). Every request ended at `length`, no errors.

**Reported, not binding.**

- #267's replay (`.scratch/449/gates/470/replay.txt`): Swift against the served artifact, pooled 1.001, CI [0.998,
  1.004] against the 0.90 floor; traffic-weighted 1.001. A smoke on the served artifact matched #340's records (j098
  and j100 byte-equal), so #340's served leg is the baseline.
- Own-accept (`analyze-swift.txt`): exact 3.467 tokens per round, traffic-weighted 3.535 (#340's NVFP4: 3.397 / 3.466).
- Loop rate (`loops/cand.txt`, `loops/tally.txt`; #340's rule: rewind, or max 6-gram >= 25, or zlib ratio < 0.20): 11 of
  32, beside #340's 27B NVFP4 at 14 of 32.
- Smokes (`smoke-*.json`), all PASS at temperature 0: chat "Paris"; reasoning split into reasoning and content ("No,
  because 391 is divisible by 17 (17 x 23 = 391)"); tool call `get_weather` {"city":"Rome"}; `locate` 422
  `locate_uncalibrated`; `point` by the digit chain at (3153, 3424), inside the expected box, 9 output tokens.

**Not run.** A vision gate (the tower is byte-identical to the served one, which the SHA-256 control proves), MMLU-Pro,
and a refusal A/B, as the ticket lists.

## Finding

Observed:

- Gate 1 passes: KLD 0.05355 against a limit of 0.0590, on the point estimate the spec registered. The CI's upper end
  is above the limit.
- Both cross terms sit above the matched pair (by 0.00296 and 0.00568), the artifact one in 13 of 16 windows with a CI
  of the difference above 0, the reference one in 16 of 16.
- Gate 2 passes in all nine cells; DFlash2 is worth 3.7x to 5.3x of decode.
- The served artifact and Swift's agree on the replay to 1.001, and Swift's own-accept (3.467 / 3.535) is above the
  served artifact's (3.397 / 3.466).
- The loop rate is 11 of 32 against 14 of 32; all five smokes pass.

Inferred:

- Swift is close to the base: the cross terms add 0.003-0.006 of KLD over about 0.05 of NVFP4 quantization, so "far
  above" holds on average and in the CI, not window by window.
- The loop rate gives no sign that Swift loops more than the base; the two numbers are not comparable as a gate.

## Implications

- The Swift NVFP4 artifact meets the spec's registered gates and may be offered, subject to the owner's reading of the
  13-of-16 control and of the CI that crosses the limit.
- A future fine-tune of the 27B can be gated the same way with `tools/kld-gate`; its controls will read the same
  closeness to the base, so the every-window rule is likely to stop again and should be decided before the run.

## Limits and unknowns

- One machine, one run per leg. The loop rate has n = 32 and used a different build and context from #340's.
- The replay's texts were sampled from the base, so the replay is biased in Swift's favour.
- The server was started by `.scratch/449/gates/srv.py` with exactly the flags `make -n start ARTIFACT=... IDLE_UNLOAD=0
  UI=0 [SPEC=off]` prints, because PowerShell (which `make start` runs) took 2-25 minutes to start on this host.
- The KLD dumps were deleted after scoring for disk; the results JSONs are kept.
- No quality measure beyond KLD, the smokes and the loop rate was run: nothing on knowledge benchmarks, vision or
  refusals.

## Follow-ups

- The owner's decision to ship the entry stays on [GitHub #470](https://github.com/gpillon/ignis/issues/470) and the
  spec [#449](https://github.com/gpillon/ignis/issues/449).
