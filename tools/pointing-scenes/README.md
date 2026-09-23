# Pointing scenes, the pointing head and the head set

The tools behind `/v1/decide`'s one-pass `point` (spec
`docs/specs/decide/13-point-by-attention-head.md`, GitHub #260, ADR 0038) and
its anchored head set (spec `docs/specs/decide/14-point-and-box-from-the-head-set.md`),
kept as they were run so that recalibrating for a new artifact is a repeat
and not a new study.

| File | What it is |
|---|---|
| `scenes.py` | The synthetic scene generator every pointing number was measured on: an app window, a target button and distractors, with the target's box in `manifest.json`. `--varied` varies the target's colour and names it by label half the time. Deterministic per `--seed` (it reproduces set C's PNGs byte for byte). Needs Pillow. |
| `engine_score.py` | Scores a dump of `crates/server/tests/attention_head_point_gpu.rs`: every GQA head's map through TAG's region rule, and which head cross-validation picks. |
| `c5_score.py` | The region rule, the argmax reading and the cross-validation `engine_score.py` imports (first written for the PyTorch vehicle's maps). Needs NumPy. |
| `rectangles.py` | Large coloured rectangles among one to three distractors of other colours, asked for by colour -- the regime where an object's corner and its centre are different image tokens (spec 14). `--blank` writes five blank priors instead (grey, black, white, noise, gradient) for the fallback cells. Deterministic per `--seed`; sizes scale with `--side`. |
| `ensemble_score.py` | Spec 14's head set: `select` chooses it (and the fallback cells) from harness dumps; `score` reads every scene with the anchored ensemble beside the anchor alone and the chain; `golden` writes the golden cases `crates/core/tests/fixtures/anchored_reading.json` holds the host rule to. Its `read_anchored` is the reading rule the server ships (`ignis_core::pointing::read_anchored`) — spec 14's, reading inside the image cell and with spec 15's allowance. Needs NumPy. |
| `pointing-heads-4bdc7b13.json` | The head set chosen for the served NVFP4 27B (content hash `4bdc7b13...6542513`): anchor L39.h10, 96 heads over L31-L63, the fallback cells measured on the 32x32 and 128x128 grids. `ensemble_score.py select` reproduces it from the dumps named inside; it is compiled into `crates/core/src/pointing.rs` (`SERVED_NVFP4_27B_SET`). |
| `bench_readout.cu` | Microbenchmark of the attention readout for spec 14: the engine's single-head kernel, the same kernel launched once per head, and a fused per-layer kernel with the per-head argmax on the device (the prototype spec 14's kernel decision comes from). Standalone `nvcc -O3 -arch=sm_120a`. |

## The sets, and what each was for

| Set | Command | Role |
|---|---|---|
| A | `scenes.py --seed 20260921` | development (blue target) |
| B | `scenes.py --varied --seed 20260922` | development (colour and label) |
| C | `scenes.py --varied --seed 20260923` | the head's engine check at 1024 px |
| C4096 | `scenes.py --varied --side 4096 --seed 20260924` | the same at 4096 px |
| D | `scenes.py --varied --seed 20260925` | spec 13's acceptance, 1024 px — used once |
| D4096 | `scenes.py --varied --side 4096 --seed 20260926` | spec 13's acceptance, 4096 px — used once |

Spent by the vision study behind spec 14
(`docs/findings/2026-09-22-the-heads-outline-the-object.md`,
`docs/findings/2026-09-23-an-anchored-head-set-points-and-boxes.md`):

| Set | Command | Role |
|---|---|---|
| T1 | `scenes.py --varied --seed 20260930` | the unanchored ensemble's test (failed on buttons), then development |
| T2 | `rectangles.py --seed 20260931` | the unanchored ensemble's test, then development |
| T1' | `scenes.py --varied --seed 20260932` | the anchored ensemble's pre-registered test — used once |
| T2' | `rectangles.py --seed 20260933` | the same — used once |
| T1c | `scenes.py --varied --side 4096 --seed 20260934 --n 10` | the ensemble at 4096 px — used once |
| T2c | T2' scenes 0-9 upscaled x4 (nearest) | the same — used once |

Spec 14's acceptance (GitHub #263), each used once, through
`crates/server/tests/decide_point_acceptance_gpu.rs`:

| Set | Command | Role |
|---|---|---|
| E1 | `scenes.py --varied --seed 20260940` | buttons, 1024 px |
| E2 | `scenes.py --varied --side 4096 --seed 20260941` | buttons, 4096 px |
| E3 | `rectangles.py --seed 20260942 --n 120` | large objects among distractors, 1024 px |
| E4 | `rectangles.py --side 4096 --seed 20260943 --n 60` | the same, 4096 px |

Spec 15's acceptance (GitHub #264), each used once, through the same test:

| Set | Command | Role |
|---|---|---|
| F1 | `scenes.py --varied --seed 20260950` | buttons, 1024 px |
| F2 | `scenes.py --varied --side 4096 --seed 20260951` | buttons, 4096 px |
| F3 | `rectangles.py --seed 20260952 --n 120` | large objects among distractors, 1024 px |
| F4 | `rectangles.py --side 4096 --seed 20260953 --n 60` | the same, 4096 px |

Spent on calibration: `rectangles.py --blank --side 4096 --seed 20260944`,
the 128x128 grid's blank priors (only the first and the last cell reach the
fallback rule there).

Generate into `.scratch/` (git does not track it). A set used to choose a
head or a rule is spent for judging it; a recalibration needs fresh seeds.

## Recalibrating the pointing head for a new artifact

`crates/core/tests/pointing_head_artifact_gpu.rs` fails in the GPU profile
the day the served artifact's content hash leaves the calibration table in
`crates/core/src/pointing.rs`. Until a head is recorded, `/v1/decide` answers
`point` with the digit chain and logs `ignis.decide.pointing_head` with
`point_method = "chain"` at startup.

1. **Scenes.** Generate development sets like A and B above and a check set
   like C, all with fresh seeds.
2. **Dump every head.** Run the attention-head harness over each set with
   every GQA layer armed, the served render and the consumed hq keys — the
   configuration production reads:

   ```text
   IGNIS_POINT_SCENES=<set dir> IGNIS_POINT_RENDER=served IGNIS_POINT_KV=hq \
   IGNIS_POINT_LAYERS=all IGNIS_POINT_OUT=<dir> \
   cargo test -p ignis-server --features cuda,attn-tap --test attention_head_point_gpu \
     -- --ignored --test-threads=1 --nocapture
   ```

   (under the GPU profile; see `docs/agents/testing.md`).
3. **Choose the head by cross-validation** on the development sets with
   `engine_score.py <dump>.json ... --folds 5`: the head the folds pick on
   the region rule's inside rate. L39.h10 was picked on every fold of every
   arm for the served 27B.
4. **Record it**: the artifact's content hash (the failing test prints it)
   and the head's (GQA ordinal, query head) as a new row of `CALIBRATED` in
   `crates/core/src/pointing.rs`.
5. **Re-run spec 13's acceptance** on fresh sets with floors written down
   before the run: `crates/server/tests/decide_point_acceptance_gpu.rs`
   (`IGNIS_POINT_SCENES=<set dir>`), which asks through `/v1/decide` with no
   `method`, and record the result as a finding.

## Recalibrating the head set (spec 14)

The head set is chosen once per artifact, after the pointing head, from the
same harness dumps:

1. **Scenes.** `rectangles.py --seed <fresh> --out <dir>` (distractors of
   other colours: selectivity needs something to be selective against) and
   `rectangles.py --blank --seed <fresh> --out <dir>` at every grid the table
   should list (1024 px gives 32x32; 4096 px gives 128x128).
2. **Dump every head** of both sets with the harness command above
   (`IGNIS_POINT_LAYERS=all`). BF16 and hq pick the same set to within a
   few heads; the served set was picked on BF16.
3. **Select**: `ensemble_score.py select --distractors <dump.json> <dir>
   --blanks <dump.json> <dir> --out heads.json` keeps the heads at layer 31
   and deeper that put at least 0.8 of their mass over the boxes on the asked
   one, with at least 0.3 on the boxes together, and lists the cells at least
   5% of all heads peak on when there is nothing to find.
4. **Record it** in the calibration table beside the pointing head, keyed to
   the same content hash (`crates/core/src/pointing.rs`, a `HeadSet` in the
   artifact's `Calibration`), and re-run spec 14's acceptance on fresh seeds
   (`crates/server/tests/decide_point_acceptance_gpu.rs`, with floors written
   down first). `ensemble_score.py score` reads any harness dump with the
   recorded set, and `ensemble_score.py golden` regenerates the host rule's
   golden cases from dumps of the new artifact.
   `crates/core/tests/pointing_head_artifact_gpu.rs` fails until both the
   head and the set are recorded.
