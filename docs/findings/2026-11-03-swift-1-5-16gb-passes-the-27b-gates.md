# Swift 1.5 as a 16 GB artifact passes the fine-tune gates: KLD 0.04255 against a limit of 0.0484 with its whole CI under it, DFlash2 beats off in every cell, replay 1.003

- Kind: experiment
- Status: current
- Observed: 2026-11-03
- Last verified: 2026-11-03
- Scope: models / the `qwen3.8-27b/trellis-sc30-swift-1.5` artifact built by `tools/27b-16gb-converter/` from a local exllamav3 conversion of `ukisai/Swift-1.5-Qwen3.8-27b` (`tools/27b-16gb-converter/local_quant.py`) and the Swift NVFP4 artifact, measured on the RTX 5090 loaded as a 16 GB card (budget 17,179,869,184, `--profile rtx50-16gb`, 2 lanes, chunk 1,024, DFlash2 7, vision off) with `tools/kld-gate`
- Topic: artifacts-and-gates
- Related: [GitHub #472](https://github.com/gpillon/ignis/issues/472), [GitHub #471](https://github.com/gpillon/ignis/issues/471),
  [GitHub #449](https://github.com/gpillon/ignis/issues/449), [GitHub #470](https://github.com/gpillon/ignis/issues/470),
  [Swift 1.5 as NVFP4 passes the fine-tune gates](2026-11-03-swift-1-5-nvfp4-passes-the-27b-gates.md) (#470, the artifact this one copies from),
  [the 27B's 16 GB artifact: the quality gates](2026-10-24-27b-16gb-quality-gates.md) (#340, the baseline and the harness),
  [the abliterated 16 GB artifact](2026-10-30-27b-16gb-abliterated-gates.md) (#422, the same 16 GB stand-in),
  [the NVFP4 work tree](../formats/27b-nvfp4-work-tree.md),
  [ADR 0060](../adr/0060-the-27b-has-a-trellis-body-for-small-cards.md),
  [ADR 0074](../adr/0074-a-catalog-entry-may-state-its-family-and-an-alias-may-be-another-entrys-id.md),
  [ADR 0078](../adr/0078-the-built-in-catalog-names-each-entrys-license-and-may-ship-restricted-weights.md)
- Superseded by: none
- Summary: Swift 1.5 as a 16 GB trellis artifact passes the fine-tune gates on the 5090 loaded as a 16 GB card: KLD 0.04255 against a limit of 0.0484 with its whole CI under it, DFlash2 on beats off in all nine cells, replay 1.003.

## Question

Does the 16 GB artifact built from Swift 1.5 (a post-training of the 27B's checkpoint) hold the bars the default 16 GB
artifact was admitted under (#340)? Its body and head come from an EXL3 quant of Swift's BF16 converted on this machine
(#471), everything else from the Swift NVFP4 artifact (#470). Ticket #472 asks for the build and its `verify`, gate 1
(KLD against Swift's own BF16, with the cross-artifact controls), gate 2 (DFlash2 on against off), the smokes, and the
conversion's time and peaks.

## Evidence

**The quant (#471).** `tools/27b-16gb-converter/local_quant.py run`, exllamav3 1.5.3, Swift `ukisai/Swift-1.5-Qwen3.8-27b`
at `b4c84d42`, into `H:/models/Swift-1.5-Qwen3.8-27b-exl3-sc30` with `--no-checkpoints`. The recipe gives each of the 401
matrices the K it has in the default 16 GB artifact's quant (`SC_3.00bpw_H4`): K2 34, K3 297, K4 66, K5 4; head 4 bits,
`mul1` codebook, output scales always, MTP 3 bits, vision 16 bits (unquantized). The calibration data is that quant's own
trace, at the revision the default's record names (`86b95530`), 250 rows x 2,048 columns. Raw: `.scratch/471/run.log`,
`launch.py`.

- Time: 2,907 s for exllamav3, 2,927 s with the checks. Final bitrate (body) 3.00.
- Peaks: VRAM 19.63 GiB allocated, 34.61 GiB reserved; host 37.66 GiB working set, 72.33 GiB commit. Disk: 25.05 GiB
  needed with `--no-checkpoints`, 61.36 GiB free at the start.
- Output: 17 files, 13.47 GB, pinned by bytes and SHA-256 in `Swift-1.5-Qwen3.8-27b-exl3-sc30.pins.json`, the recipe in
  `.recipe.yaml` beside it.

**The control (#471).** The relative reconstruction error of each matrix against Swift's BF16, beside the default 16 GB
artifact's on the same matrix against the base BF16, over 401 matrices (`.scratch/471/control.log`, `control.json`):

| Quant | against | max | mean | output head |
|---|---|---|---|---|
| Swift's local quant | Swift BF16 | 0.3368 | 0.1500 | 0.0766 |
| the default's quant | base BF16 | 0.3383 | 0.1500 | 0.0765 |

**The build.** `H:/nvfp4-472/qwen3_8_27b_trellis_sc30-v2-ukisai-swift-1.5.ninfer` (+ `.conversion.json`), weights id
`trellis-sc30-swift-1.5`, model id `qwen3.8-27b`, recipe `qwen3_8_27b_trellis_sc30-v2-ukisai-swift-1.5`, license
`swift-open-license-1.0`. The 16 GB converter (#465, commit `92af7965`) ran in 950 s: 401 trellis matrices plus 821 objects
copied from the Swift NVFP4 artifact; 247 of that artifact's objects dropped and the quant's 8 MTP matrices unused (the
artifact's 12 MTP objects are among the copied). Packed: 1,222 objects, 13,379,877,124 B, the same size as the default
16 GB artifact. Raw: `.scratch/469/run-swift16.log`, `pack-swift16.log`.

- The quant labels: `--quant-repo ukisai/Swift-1.5-Qwen3.8-27b --quant-branch exl3-sc30-local`, accepted by the owner
  on #472. The record pins the quant by its pins file's SHA-256 (revision null: it is a local conversion), and the Swift
  NVFP4 artifact and its sidecar by SHA-256; the NVFP4 build's source is read through that sidecar
  ([the NVFP4 work tree](../formats/27b-nvfp4-work-tree.md)).
- A retry: the first attempt died loading the exllamav3 extension (exit 0xC000013A, as #472 records) and left the
  extension's lock behind, the hazard #471 named. The lock was deleted and the second attempt passed
  (`run-swift16.attempt1.log`, `run-swift16.log`).
- `verify` (`.scratch/469/verify-swift16.log`): 401 trellis objects against their source tensors bit for bit, 821
  copied objects against the Swift NVFP4 artifact's SHA-256, 0 problems, VERIFY PASS.
- The DFlash2 (66), vision (333) and frontend (6) objects are SHA-256-equal to the default 16 GB artifact's (#472).
- Content hash `98b4bd83...` (printed in `.scratch/449/gates/472/smoke-server.log`), distinct from every built-in
  artifact's and from the Swift NVFP4 artifact's.

**The load.** The checksum is clean and it serves as `qwen3.8-27b`: 2 lanes, 262,144 context, chunk 1,024, hq-e8-2b,
DFlash2 7 (`draft_head=full`), vision off; 17,179,375,716 B planned of the 17,179,869,184 budget, 7,956 KV pages
(`smoke-server.log`).

**Gate 1** (chunk 1,024, set A of 16 windows x 4,096 tokens: 6 code, 4 prose, 6 chat; `tools/kld-gate`).
Raw: `.scratch/449/gates/472/gate1.log`, `gate1.json`, `results/`.

| Pair | KLD | 95% window-bootstrap CI |
|---|---|---|
| Swift 16 GB given Swift BF16 (the gate) | 0.04255 | [0.03709, 0.04785] |
| default 16 GB given Swift BF16 (cross_artifact) | 0.04643 | [0.04054, 0.05234] |
| Swift 16 GB given base BF16 (cross_reference) | 0.04961 | [0.04375, 0.05572] |
| default 16 GB given base BF16 (baseline, #340) | 0.04034 | not computed: the limit's base |

The limit is 1.2 x 0.04034 = 0.0484. The baseline pair reproduced #340's 0.04034 to the bit (`--require-reproduction`).
By kind, the gate pair: chat 0.03842, code 0.04073, prose 0.05147. Against the default's own pair the ratio is 1.055,
CI [1.024, 1.095]; the gate pair is the lower of the two in 4 of 16 windows. Gate 1 is a PASS, and the CI's upper end
(0.04785) is under the limit too.

**Controls.** Each cross term against the matched pair, its 95% CIs from the same window resamples:

| Cross term | / matched pair, 95% CI | - matched pair, 95% CI | Above it in |
|---|---|---|---|
| cross_artifact | 1.091 [1.049, 1.138] | 0.00388 [0.00207, 0.00583] | 13 of 16 windows |
| cross_reference | 1.166 [1.116, 1.226] | 0.00706 [0.00496, 0.00944] | 16 of 16 windows |

The three windows where cross_artifact sits below the matched pair are two code and one prose; it is above in all six
chat windows. The scorer's every-window rule printed STOP. The owner accepted the control as measured, as on #470 (the
CI of the difference above 0, and 16 of 16 on the reference cross term), recorded on #472. The wiring controls all held.
No fallback of the ticket's list (a calibration trace sampled from Swift, then exllamav3 1.4.2) was taken.

**Gate 2** (DFlash2 on against off, decode tok/s; `.scratch/449/gates/472/cells/gate2.txt`, `cells.py`):

| Context | 1-lane cold, on / off | 2-lane, on / off | 1-lane reuse, on / off |
|---|---|---|---|
| 8,192 | 410.7 / 92.6 | 661.1 / 168.9 | 417.3 / 92.5 |
| 65,536 | 339.8 / 74.6 | 490.7 / 112.2 | 344.4 / 74.7 |
| 253,440 | 228.7 / 43.8 | 284.4 / 54.8 | 216.3 / 43.8 |

On beats off in every cell (3.91x to 5.22x).

**Reported, not binding.**

- #267's replay (`.scratch/449/gates/472/replay.txt`, `replay.json`): Swift 16 GB over the default 16 GB, pooled 1.003,
  CI [1.000, 1.006], against the 0.90 floor; traffic-weighted 1.003. Per job (116): median 1.001, p10 0.965, p90 1.037,
  none below 0.90. P(accepted >= i), i = 1..7: Swift .713 .509 .372 .279 .212 .164 .131, default .712 .508 .370 .276
  .210 .164 .131. The baseline is #340's `out-trellis`: a smoke on the default artifact matched it (j098 and j100
  byte-equal, `.scratch/469/legs-472.log`), so that leg was not re-run.
- Own-accept (`analyze-swift.txt`): exact 3.482 tokens per round, traffic-weighted 3.552 (#340's default trellis: 3.420 /
  3.487, `.scratch/340/replay-run/analyze-trellis.txt`).
- Smokes (`smoke-*.json`), all PASS at temperature 0: chat "Paris"; reasoning split into reasoning and content ("No,
  because 391 is divisible by 17 (17 x 23 = 391)"); tool call `get_weather` {"city":"Rome"}.

**Not run.** A vision gate (vision is off on the 16 GB path, and the tower is byte-identical to the default's, which the
SHA-256 control proves). MMLU-Pro, a refusal A/B and the loop rate, which are not among #472's criteria.

## Finding

Observed:

- exllamav3 1.5.3 converts the whole 27B on this Windows machine: 2,907 s, 19.63 GiB of VRAM allocated (34.61 reserved),
  72.33 GiB of host commit. Its matrices reconstruct Swift's BF16 as closely as the default's quant reconstructs the
  base's (mean relative error 0.1500 both, max 0.3368 against 0.3383).
- Gate 1 passes: KLD 0.04255 against a limit of 0.0484, and the whole CI is under the limit.
- Both cross terms sit above the matched pair (by 0.00388 and 0.00706, both CIs of the difference above 0), the artifact
  one in 13 of 16 windows, the reference one in 16 of 16.
- Gate 2 passes in all nine cells; DFlash2 is worth 3.9x to 5.2x of decode on the 16 GB path.
- The replay puts Swift's 16 GB artifact at 1.003 of the default's, and its own-accept (3.482 / 3.552) is above the
  default trellis's (3.420 / 3.487). All three smokes pass.

Inferred:

- Swift is close to the base on the 16 GB path as on the NVFP4 one: the cross terms add 0.004-0.007 of KLD over about
  0.04 of quantization, so "far above" holds on average and in the CI, not window by window.
- A fine-tune quantizes a little worse against its own BF16 than the base does against its own, on both paths: the gate
  pair over the default's own pair is 1.055 here and was 1.089 for the NVFP4 artifact (#470), both CIs above 1.
- Per matrix, the local conversion reconstructs Swift as well as the public quant reconstructs the base, so a worse
  conversion does not explain the KLD's excess over the default; the calibration trace, the default quant's rather than
  one sampled from Swift, is not ruled out as a cause.

## Implications

- The Swift 16 GB artifact meets the spec's registered gates with its whole CI under the limit, and may be offered,
  subject to the owner's decision on #449.
- A fine-tune of the 27B can reach the 16 GB path through a local exllamav3 conversion and the 16 GB converter, without a
  public EXL3 quant. The conversion takes 2,907 s (48 min) of GPU, a 72.33 GiB host commit and 25.05 GiB of disk with
  `--no-checkpoints`.
- The quant's shards were deleted after the build to make room for gate 1's KLD dumps (they are inside the verified
  artifact; `.pins.json` and `.recipe.yaml` are kept). Rebuilding the artifact, or running `convert.py verify --quant`
  on it again, means re-running #471's conversion.
- The every-window rule on the cross terms stopped on both Swift artifacts; a future fine-tune of the 27B will read the
  same closeness to the base, so the rule should be decided before its run.

## Limits and unknowns

- The 5090 stands in for a 16 GB card (budget and profile), as in #422 and #340: no real 16 GB card was measured.
- One machine, one run per leg.
- The replay's texts were sampled on the base's NVFP4, so the replay leans against Swift: 1.75% of the recorded tokens
  fall outside Swift's sampling support, against 1.20% for the default trellis (#340).
- The replay was paused at 14:05 at 115 of 116 jobs and resumed at 17:19 (`.scratch/469/legs-472.log`,
  `legs-472-resume.log`); the replay is resumable per job.
- The server was started by `.scratch/449/gates/srv.py` with the flags `make -n start` prints for the 16 GB line, because
  PowerShell (which `make start` runs) was slow on this host that day.
- The KLD dumps were deleted after scoring for disk; the results JSONs are kept. The disk peak was within H:'s free
  space and was not recorded as a number.
- No quality measure beyond KLD and the smokes was run: nothing on knowledge benchmarks, vision, refusals or loops.

## Follow-ups

- The owner's decision to ship the entry stays on [GitHub #472](https://github.com/gpillon/ignis/issues/472) and the
  spec [#449](https://github.com/gpillon/ignis/issues/449).
