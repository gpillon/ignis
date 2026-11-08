# The sm_86 gate passes on an Ada Lovelace card (sm_89) at `main` b13dbc05: the sm_86 card's scores in every digit, the leaf's 127 tests, both images; an `89` cubin changes no logit

- Kind: experiment
- Status: current
- Observed: 2026-11-07
- Last verified: 2026-11-07
- Scope: release procedure / the sm_86 gate (`docs/agents/testing.md`, `docs/user/contributing/releases.md`) run on an
  sm_89 card; quality / top-64 KLD against BF16 on the routes an 8.x card runs unset (spec #503); kernel / `ctest -LE
  sm120a-only` on the `86;120a` leaf, the vendored GDN gating planner on sm_89; build / an `86;89;120a` leaf against
  the release's `86;120a`; serving / `--profile auto` on the card, DFlash2, vision
- Related: [GitHub #532](https://github.com/gpillon/ignis/issues/532), [GitHub #484](https://github.com/gpillon/ignis/issues/484),
  [GitHub #516](https://github.com/gpillon/ignis/issues/516), [GitHub #503](https://github.com/gpillon/ignis/issues/503),
  [GitHub #474](https://github.com/gpillon/ignis/issues/474),
  [The sm_86 gate's first run passes on `main` 6234e002](2026-11-06-sm86-gate-first-run.md),
  [ADR 0080](../adr/0080-on-sm8x-the-trellis-prefill-decodes-each-matrix-once-per-chunk-and-multiplies-on-the-integer-mma.md)
- Superseded by: none

## Question

One `86` cubin serves compute capability 8.6 and 8.9 (ADR 0080), and #516's gate ran on an sm_86 card only. Does
`main` pass the gate's three parts on an Ada Lovelace card (sm_89) too, on the routes an 8.x card runs unset? Do the
card's logits differ from the sm_86 card's? And does a build that adds an `89` cubin change any of them?

## Evidence

One Ada Lovelace card (sm_89), `main` at `b13dbc05`, the gate's commands as `docs/agents/testing.md` gives them, in
one session with nothing else on the card. The release's build, `IGNIS_CUDA_ARCHITECTURES=86;120a`, ran every part.
The leaf's sources are those of the sm_86 run: `git diff 6234e002 b13dbc05 -- kernel/` is empty. A second build,
`86;89;120a`, from the same commit in a tree and target of its own, ran the quality leg again.

`main` has since taken #530 (a956b896), which changes the tile heights the dense prefill's `plan()` picks, in host
code. #530's first commit, fe2f049a, records the `86` and `120a` cubins of both archives byte-identical before and
after. The gate did not run again on a956b896.

**Quality.** #340's gate 1 windows (16 of 4,096 tokens: 6 code, 4 prose, 6 chat), dumped by `span_logits` at chunk
1024, KV `hq-e8-2b`, under `IGNIS_MODEL_PREFILL_LINEAR=int8 IGNIS_MODEL_PREFILL_ATTENTION=int8 IGNIS_GDN_PREFILL=ours`,
and scored by `tools/kld-gate/score_top64.py` against the BF16 top-64 reference set.

| top-64 KLD against BF16, 8.x default routes | all | chat | code | prose | top-1 (all) |
|---|---|---|---|---|---|
| sm_89 card, `86;120a` | 0.03944 | 0.03724 | 0.03718 | 0.04613 | 93.33% |
| sm_89 card, `86;89;120a` | 0.03944 | 0.03724 | 0.03718 | 0.04613 | 93.33% |
| sm_86 card, 2026-11-06 | 0.03944 | 0.03724 | 0.03718 | 0.04613 | 93.33% |
| the gate's limits | ≤ 0.04081 | | | | ≥ 92.85% |

- The sm_89 card's dump is its own: `span_logits` wrote the 16 windows in this session, and the scorer read them
  afresh.
- Against the sm_86 card's run, the scorer's output agrees in every digit it writes: the 16 per-window rows, the 4
  per-kind rows and the 44 double-precision values of its JSON, 11 per kind. Eight of the 11 come from the dump: the
  mean, median, p99 and maximum of the KLD, top-1 against BF16 and against the 5090's dump, NLL and perplexity. The
  other three, the reference's NLL and perplexity and the 5090 dump's KLD, the reference set carries.
- The scorer normalises each row by its own log-sum-exp over the full vocabulary, summed in f64, so one BF16 step on
  a logit near a row's maximum would show in the JSON's means.
- `86;89;120a` against `86;120a` on the sm_89 card: a byte-for-byte comparison of the two dumps over all 65,536
  positions finds the argmax the same at every position, the largest |Δlogit| 0, every element bit-equal and no
  non-finite value.

**The leaf's tests.** `ctest -LE sm120a-only`: 127 of 127 pass. `ignis_kernel_gdn_gating_proj_tests` prints the sm_86
card's `budget` and `SKIP` lines unchanged, and its output differs from that run's only in the elapsed times.
- The 27B's cooperative gating schedules are not resident at T = 1,024, 1,025 and 2,049. The planner runs
  `gdn_gating_proj.bf16.mma.unsplit` there.
- The norm gating's split-40 schedule is not resident at the widths the test tries (T = 1, 6 and 8).
- The tightest case is T = 4,097, at 0.996 of its rel_l2 limit, the same figure as on the sm_86 card.

**Smoke.** `ignis-server --profile auto` with the artifact, the bind, idle unload off and the UI off, and no other
flag:

- The start picks a built-in profile by itself (`profile_by: auto`). DFlash2 runs with 7 draft tokens, vision is on
  with host residency.
- `GET /v1/config` gives `prefill_linear` and `prefill_attention` as `int8` for `qwen38`. The load's
  `ignis.runtime.prefill_routes` line reads `prefill_linear: int8, prefill_attention: int8, gdn_prefill: ours`.
- A greedy chat completion ("What is the capital of France? One word.") answers `Paris`. A 32 x 32 red image answers
  `red` (`media.vision_tokens: 64`), a 4096 x 4096 blue image `Blue` (`media.vision_tokens: 16384`).
- The three response bodies equal the sm_86 card's, the creation time aside. The log holds no `ERROR` line.

## Finding

- Observed: `main` b13dbc05 passes the sm_86 gate on an Ada Lovelace card (sm_89), all three parts, with the
  release's `86;120a` build.
- Observed: the card's quality figures equal the sm_86 card's in every digit the scorer writes, per window, per kind
  and in its JSON.
- Inferred: the `86` cubin gives the same logits on 8.6 and 8.9, bit for bit. No byte diff of the two cards' dumps
  was run.
  - It is the expected outcome: the leaf is the same SASS on both cards and its kernels are deterministic.
  - The integer routes are exact by construction. The dense GEMM accumulates in int32 over the whole K and applies
    its fp32 epilogue per element; the band's Q·Kᵀ is exact in int32.
  - So the dense prefill's tile height, which `plan()` picks per card from its SM count, leaves the logits alone,
    before #530 and after it. Since #530, `ignis_kernel_dense_prefill_test` holds each of the eight tile plans to the
    op's own output byte for byte; that test did not exist at b13dbc05 and did not run on the sm_89 card.
- Observed: an `89` cubin changes no logit on the sm_89 card.

## Implications

- On correctness, RTX 40 cards move from "expected to work" to verified in the guide: the change that records this
  finding updates `docs/user/` (#532). The guide gives no speed for them.
- The release keeps `86;120a`: on correctness an `89` cubin brings nothing to the 8.x routes. If one ever enters the
  build, the release check's SASS rule has to scan it too: today it reads the `86` cubin only
  (`mk/cuda-build-check.sh`).
- The gate's quality leg needs no limits of its own for sm_89.
- The gate records correctness only. The session's speed figures stay private, as #516's do.

## Limits and unknowns

- One Ada Lovelace card. The result covers the numerics of the `86` and `89` cubins on sm_89, not any card's speed,
  and does not say which RTX 40 card class ran it.
- The quality leg scores the 27B's 16 GB trellis entry only; the NVFP4 entry and Flash-Next did not run here.
- The cross-card identity rests on the scorer's figures. A byte diff of the two cards' dumps would make it observed.
- The inference covers the routes whose launch shape follows the card as far as they were looked at: the dense tile
  height above, and the gating planner, which runs the same unsplit route on both cards at the test's widths. No
  other route was audited for a reduction order that follows the card.
- The smoke checks each answer's word, one request each. It measures no acceptance rate of DFlash2 and no image
  quality beyond the colour.
- T = 4,097 sits at 0.996 of its limit on both cards. That margin belongs to the test case: the same SASS gives the
  same number on either architecture.

## Follow-ups

- A byte diff of the sm_86 and sm_89 dumps, where both are kept. It needs no GPU.
