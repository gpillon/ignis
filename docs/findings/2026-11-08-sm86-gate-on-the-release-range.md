# The sm_86 gate on 0.10.0's range passes at `main` adb5ed2b on an Ada Lovelace card (sm_89): every earlier run's scores in every digit, the leaf's 127 tests with #530's tile-plan test, both images

- Kind: experiment
- Status: current
- Observed: 2026-11-08
- Last verified: 2026-11-08
- Scope: release procedure / the sm_86 gate (`docs/agents/testing.md`, `docs/user/contributing/releases.md`) for
  release 0.10.0, run on an sm_89 card; quality / top-64 KLD against BF16 on the routes an 8.x card runs unset;
  kernel / `ctest -LE sm120a-only` on the `86;120a` leaf, the dense prefill's tile plans (#530) on an 8.x card;
  serving / `--profile auto` on the card, DFlash2, vision
- Topic: host-and-gpu
- Related: [GitHub #485](https://github.com/gpillon/ignis/issues/485), [GitHub #484](https://github.com/gpillon/ignis/issues/484),
  [GitHub #530](https://github.com/gpillon/ignis/issues/530), [GitHub #536](https://github.com/gpillon/ignis/issues/536),
  [The sm_86 gate's first run passes on `main` 6234e002](2026-11-06-sm86-gate-first-run.md),
  [The sm_86 gate passes on an Ada Lovelace card (sm_89) at `main` b13dbc05](2026-11-07-sm86-gate-passes-on-sm89.md),
  [The dense prefill's tile plan weighs a 128-row tile's rows at 1.10](2026-11-07-dense-prefill-tile-plan-weighs-a-128-row-tile.md)
- Superseded by: none
- Summary: At main adb5ed2b, which 0.10.0 is cut from, an Ada Lovelace card (sm_89) passes the sm_86 gate with the 86;120a build, its quality figures the sm_86 card's in every digit, and #530's tile-plan test passes on 8.x.

## Question

The gate's first run passed on `main` 6234e002 ([GitHub #484](https://github.com/gpillon/ignis/issues/484)). The
release procedure asks for it again when the range up to the release touched the kernel leaf or the architecture
module, and `git diff --stat 6234e002 adb5ed2b -- kernel/ crates/server/src/config/arch.rs` is not empty:

- #530 (fe2f049a, a956b896) changes the tile heights the dense prefill's `plan()` picks, in host code. fe2f049a
  records the `86` and `120a` cubins of both archives byte-identical before and after.
- #536 (fe93a9ff) makes ignis's own GDN prefill Blackwell's default too. On 8.x, `runs_by_default` was already true.
- The other commits that touch `kernel/` (2535e0d4, b3c01c6e, 86c2f8ff) change comments.

Does `main` at adb5ed2b pass the gate's three parts? Does #530's plan change any logit on an 8.x card?

## Evidence

One Ada Lovelace card (sm_89), `main` at `adb5ed2b`, built with `IGNIS_CUDA_ARCHITECTURES=86;120a`, the gate's
commands as `docs/agents/testing.md` gives them, in one session with nothing else on the card.

- The sm_86 card of the earlier runs was no longer available. The owner decided that this run goes on an sm_89
  card, which runs the release's `86` cubin: the procedure's text names an sm_86 card.
- The 2026-11-07 finding had shown the `86` cubin giving the sm_86 card's quality figures on sm_89 in every digit.

**Quality.** #340's gate 1 windows (16 of 4,096 tokens: 6 code, 4 prose, 6 chat), dumped by `span_logits` at chunk
1024, KV `hq-e8-2b`, under `IGNIS_MODEL_PREFILL_LINEAR=int8 IGNIS_MODEL_PREFILL_ATTENTION=int8 IGNIS_GDN_PREFILL=ours`,
and scored by `tools/kld-gate/score_top64.py` against the BF16 top-64 reference set.

| top-64 KLD against BF16, 8.x default routes | all | chat | code | prose | top-1 (all) |
|---|---|---|---|---|---|
| sm_89 card, adb5ed2b | 0.03944 | 0.03724 | 0.03718 | 0.04613 | 93.33% |
| sm_89 card, b13dbc05 (2026-11-07) | 0.03944 | 0.03724 | 0.03718 | 0.04613 | 93.33% |
| sm_86 card, 6234e002 (2026-11-06) | 0.03944 | 0.03724 | 0.03718 | 0.04613 | 93.33% |
| the gate's limits | ≤ 0.04081 | | | | ≥ 92.85% |

- The dump is this session's own: `span_logits` wrote the 16 windows, and the scorer read them afresh.
- The per-kind top-1 figures equal the earlier runs' too: chat 96.04%, code 93.25%, prose 89.40%.

**The leaf's tests.** `ctest -LE sm120a-only`: 127 of 127 pass.

- `ignis_kernel_dense_prefill_test` is among them. Since #530 it holds each of the eight tile plans to the op's own
  output byte for byte. No 8.x card had run it before.
- `ignis_kernel_gdn_gating_proj_tests` prints only `budget` and `SKIP` lines (#478), and no failure.

**Smoke.** The release workflow's `develop` build of adb5ed2b (the Linux archive), `ignis-server --profile auto` with
the artifact, the bind, idle unload off and the UI off, and no other flag:

- The start picks a built-in profile by itself (`profile_by: auto`).
- `GET /v1/config` gives `prefill_linear` and `prefill_attention` as `int8`. The load's log reads
  `prefill_linear: int8`, `prefill_attention: int8` and `gdn_prefill: ours`.
- A greedy chat completion ("What is the capital of France? One word.") answers `Paris`.
- A 32 x 32 red image answers `red` (`media.vision_tokens: 64`), a 4096 x 4096 blue image `Blue`
  (`media.vision_tokens: 16384`).
- The log holds no `ERROR` line.

## Finding

- Observed: `main` adb5ed2b passes the sm_86 gate on an Ada Lovelace card (sm_89), all three parts, with the
  release's `86;120a` build.
- Observed: the quality figures equal the sm_86 card's at 6234e002 and the sm_89 card's at b13dbc05, in every digit of
  the `all` row and of the three kinds' rows.
- Observed: on an 8.x card each of the dense prefill's eight tile plans gives the op's own output byte for byte.
- Inferred: #530 changes no logit on 8.x. The plan picks a tile height from the card's SM count, every height gives
  the same bytes, and the dense GEMM accumulates in int32 over the whole K.

## Implications

- The gate holds for release 0.10.0 as long as the range from adb5ed2b to the release commit touches neither
  `kernel/` nor `crates/server/src/config/arch.rs`. The version bump, the changelog and the guide's markings touch
  neither.
- 0.10.0's release notes cite this finding as the gate's run, and say that it ran on sm_89.
- Whether the procedure should name "an 8.x card that runs the `86` cubin" in place of an sm_86 card is the owner's
  decision. This finding does not change the procedure.
- The gate records correctness only. The session's speed figures stay private, as #516's do.

## Limits and unknowns

- No sm_86 card ran adb5ed2b. An sm_86 card has a different SM count, so `plan()` can pick other tile heights there
  than on this card. The byte-identity of every plan, observed here on an 8.x card, is what covers those picks.
- One Ada Lovelace card. The result covers the numerics of the `86` cubin on sm_89, not any card's speed, and does not
  say which RTX 40 card class ran it.
- The quality leg scores the 27B's 16 GB trellis entry only. The NVFP4 entry and Flash-Next did not run here.
- The smoke checks each answer's word, one request each. It measures no acceptance rate of DFlash2 and no image
  quality beyond the colour.

## Follow-ups

- [GitHub #485](https://github.com/gpillon/ignis/issues/485): release 0.10.0, which cites this run.
