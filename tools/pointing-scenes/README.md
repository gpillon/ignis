# Pointing scenes and the pointing head's calibration

The tools behind `/v1/decide`'s one-pass `point` (spec
`docs/specs/decide/13-point-by-attention-head.md`, GitHub #260, ADR 0038),
kept as they were run so that recalibrating for a new artifact is a repeat
and not a new study.

| File | What it is |
|---|---|
| `scenes.py` | The synthetic scene generator every pointing number was measured on: an app window, a target button and distractors, with the target's box in `manifest.json`. `--varied` varies the target's colour and names it by label half the time. Deterministic per `--seed` (it reproduces set C's PNGs byte for byte). Needs Pillow. |
| `engine_score.py` | Scores a dump of `crates/server/tests/attention_head_point_gpu.rs`: every GQA head's map through TAG's region rule, and which head cross-validation picks. |
| `c5_score.py` | The region rule, the argmax reading and the cross-validation `engine_score.py` imports (first written for the PyTorch vehicle's maps). Needs NumPy. |

## The sets, and what each was for

| Set | Command | Role |
|---|---|---|
| A | `scenes.py --seed 20260921` | development (blue target) |
| B | `scenes.py --varied --seed 20260922` | development (colour and label) |
| C | `scenes.py --varied --seed 20260923` | the head's engine check at 1024 px |
| C4096 | `scenes.py --varied --side 4096 --seed 20260924` | the same at 4096 px |
| D | `scenes.py --varied --seed 20260925` | spec 13's acceptance, 1024 px — used once |
| D4096 | `scenes.py --varied --side 4096 --seed 20260926` | spec 13's acceptance, 4096 px — used once |

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
