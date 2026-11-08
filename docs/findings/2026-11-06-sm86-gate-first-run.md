# The sm_86 gate's first run passes on `main` 6234e002: the 8.x default routes at the 5090's quality, the leaf's 127 tests, both images under `rtx-24gb`

- Kind: experiment
- Status: current
- Observed: 2026-11-06
- Last verified: 2026-11-06
- Scope: release procedure / the sm_86 gate (`docs/agents/testing.md`, `docs/user/contributing/releases.md`);
  quality / top-64 KLD against BF16 on the routes an 8.x card runs unset (spec #503); kernel / `ctest -LE
  sm120a-only` on the `86;120a` leaf, the vendored GDN gating planner on sm_86; serving / `--profile auto` on the card,
  DFlash2, vision
- Topic: host-and-gpu
- Related: [GitHub #484](https://github.com/gpillon/ignis/issues/484), [GitHub #516](https://github.com/gpillon/ignis/issues/516),
  [GitHub #503](https://github.com/gpillon/ignis/issues/503), [GitHub #478](https://github.com/gpillon/ignis/issues/478),
  [On an sm_86 card the 27B trellis entry runs at the 5090's quality](2026-11-03-first-light-on-sm86.md),
  [The solo width](2026-11-06-the-solo-width-planned-at-load.md)
- Superseded by: none
- Summary: On an sm_86 card `main` 6234e002 passes the sm_86 gate: top-64 KLD 0.03944 and top-1 93.33% on the default 8.x routes (limits 0.04081 and 92.85%), 127 of 127 leaf tests, and `rtx-24gb` under `auto` serving chat and images.

## Question

#484 wrote the sm_86 gate into the release procedure: quality, the leaf's tests and a smoke, each with its
limits. Spec #503 then made the dense INT8 prefill, the integer band and ignis's own gated delta net prefill the
routes an 8.x card runs unset, and moved the gate's quality leg onto them. Does `main` with both specs in pass
all three parts on an sm_86 card? Which patch count does the smallest image reach, and does the vendored GDN
gating planner keep a cooperative schedule resident there at the widths a load can plan (#489)?

## Evidence

One sm_86 card, `main` at `6234e002` built with `IGNIS_CUDA_ARCHITECTURES=86;120a`, the gate's commands as
`docs/agents/testing.md` gives them, in one session with nothing else on the card.

**Quality.** #340's gate 1 windows (16 of 4,096 tokens: 6 code, 4 prose, 6 chat), dumped by `span_logits` at
chunk 1024, KV `hq-e8-2b`, under `IGNIS_MODEL_PREFILL_LINEAR=int8 IGNIS_MODEL_PREFILL_ATTENTION=int8
IGNIS_GDN_PREFILL=ours`, scored by `tools/kld-gate/score_top64.py` against the BF16 top-64 reference set.

| top-64 KLD against BF16, 8.x default routes | all | chat | code | prose | top-1 (all) |
|---|---|---|---|---|---|
| sm_86 card | 0.03944 | 0.03724 | 0.03718 | 0.04613 | 93.33% |
| RTX 5090, same routes (#514's gate leg) | 0.03887 | | | | 93.35% |
| the gate's limits | ≤ 0.04081 | | | | ≥ 92.85% |

- The `all` row reads 1.5% above the 5090's KLD on the same routes, against the 5% the limit allows, and top-1
  0.02 points under it, against the 0.5 allowed.
- The scorer's own "vs 5090" columns compare with the dump the reference set carries, the 5090's 8-bit band
  of the first-light finding (0.03801), not with the routes the gate scores; the limits come from #514's figures.

**The leaf's tests.** `ctest -LE sm120a-only`: 127 of 127 pass. In `ignis_kernel_gdn_gating_proj_tests` the
`budget` lines and the `SKIP` lines are the expected ones (#478): the 27B's cooperative gating schedules are not
resident on the card at T = 1,024, 1,025 and 2,049, and the planner runs `gdn_gating_proj.bf16.mma.unsplit`
there; the norm gating's split-40 schedule is not resident at the widths the test tries (T = 1, 6 and 8).

**Smoke.** `ignis-server --profile auto` with the artifact, the bind, idle unload off and the UI off, and no
other flag:

- The start names `rtx-24gb`, `profile_by: auto`. DFlash2 runs with 7 draft tokens, vision is on with host
  residency.
- `GET /v1/config` gives `prefill_linear` and `prefill_attention` as `int8` for `qwen38`, and the load's
  `ignis.runtime.prefill_routes` line reads `prefill_linear: int8, prefill_attention: int8, gdn_prefill: ours`.
- A greedy chat completion ("What is the capital of France? One word.") answers `Paris`.
- A 32 x 32 red image answers `red`. The processor upscales it to the 65,536-pixel floor: 256 patches at the
  vision tower, 64 tokens after the merge (`media.vision_tokens: 64`).
- A 4096 x 4096 blue image answers `Blue`, at the processor's item bound: `media.vision_tokens: 16384`.
- The log holds no `ERROR` line.

## Finding

- Observed: `main` 6234e002 passes the sm_86 gate on an sm_86 card, all three parts. The 8.x default routes
  read within 1.5% of the RTX 5090's KLD on the same routes and 0.02 points of its top-1.
- Observed: the smallest image an RTX 30 card serves reaches the vision tower as 256 patches, above every
  small-token route of the vendored W8 dispatch for the tower's shapes (SIMT up to 16 tokens, the merger's up
  to 4 or 5), so it takes `mma_r64_c128`, a kernel the `86;120a` build holds. No image reaches a kernel only
  the sm_120a set holds.
- Observed: on the card no cooperative gating schedule of the 27B is resident at 2,048, 4,096 or 8,192 tokens.
  The schedules the planner names for 2,048 and 4,096 are not resident at the first width of their ranges
  (T = 1,025 and 2,049 in the test), and it falls back to the unsplit route; at 8,192 the unsplit route is the
  planner's choice on every card.

## Implications

- The release procedure's gate holds for 0.10.0 as written: the quality leg on the 8.x default routes passes
  with room for the routes #503 made the default on RTX 30 and RTX 40 cards.
- The solo width needs no gating route of its own on sm_86: the chunk and every wider width run the same
  unsplit gating kernel.
- The gate records correctness only. #516's speed gate ran in the same session; its figures stay private under
  that ticket's rule.

## Limits and unknowns

- One sm_86 card. No Ada card ran the gate.
- The quality leg scores the 27B's 16 GB trellis entry only; the NVFP4 entry and Flash-Next did not run here.
- The smoke checks each answer's word, one request each; it measures no acceptance rate of DFlash2 and no
  image quality beyond the colour.
- `docs/agents/testing.md` says `GET /v1/config` reports the card as the routes' source; the response read here
  carries the values, and the source was read from the load's log line instead.

## Follow-ups

- [GitHub #485](https://github.com/gpillon/ignis/issues/485): the 0.10.0 release, which this gate and #516's
  speed gate were waiting for.
