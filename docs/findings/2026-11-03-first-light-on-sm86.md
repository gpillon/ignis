# On an sm_86 card the 27B trellis entry runs at the 5090's quality: the leaf built for `86;120a` loads, prefills on the FP16 band and decodes

- Kind: experiment
- Status: current
- Observed: 2026-11-03
- Last verified: 2026-11-03
- Scope: kernel / the leaf built for `86;120a` (`IGNIS_CUDA_ARCHITECTURES`), its CTest suite on an sm_86 card, the
  dense trellis decode body; serving / the start-up architecture check, the routes keyed on compute capability, the
  27B trellis entry (`qwen3.8-27b-16g`) end to end; quality / top-64 KLD against the BF16 checkpoint
- Related: [GitHub #456](https://github.com/gpillon/ignis/issues/456), [GitHub #457](https://github.com/gpillon/ignis/issues/457),
  [ADR 0077](../adr/0077-a-vendored-file-may-carry-a-portability-patch.md),
  [RTX 20, 30 and 40 cards](2026-11-02-rtx-20-30-40-what-blocks-the-leaf.md),
  [27B on 16 GB: the quality gates](2026-10-24-27b-16gb-quality-gates.md),
  [dense trellis kernels on the 5090](2026-10-23-dense-trellis-kernels-on-the-5090.md)
- Superseded by: none

## Question

The 2026-11-02 finding compiled the leaf for sm_89 and listed what stops it; #456 and #457 then made the build cover
older cards (`IGNIS_CUDA_ARCHITECTURES=86;120a`) and keyed the routes on compute capability. Compiling is not
running. Does that build run the 27B's 16 GB trellis entry on an sm_86 card, at what quality against the BF16
checkpoint, and what do the leaf's own tests say there? Does the 5090 see any change?

## Evidence

All on branch `ampere-build`, one sm_86 card for the card-side runs, the RTX 5090 for the rest.

**The start, before and after.** main's binary, built for `120a` alone, refuses the trellis load on the sm_86 card
with CUDA's own text, `ignis_trellis_prepare: decode attributes: no kernel image is available for execution on the
device`, which is the case #457 replaces with a named refusal. The branch built for `86;120a` starts and loads it.

**Quality.** The windows of #340's gate 1 (16 windows of 4,096 tokens: 6 code, 4 prose, 6 chat), dumped by the
`span_logits` example at chunk 1024 with the hq-e8-2b KV format, scored against a reference set holding the BF16
checkpoint's 64 most probable next tokens per position (the `kl_top64` formula of `crates/bench/src/flash_next/kld.rs`).
A card below compute capability 12.0 runs the prompt attention's FP16 band only, so the 5090 was dumped on that band
too (`IGNIS_MODEL_PREFILL_ATTENTION=fp16`, which `span_logits` now reads).

| top-64 KLD against BF16 | all | chat | code | prose | top-1 |
|---|---|---|---|---|---|
| RTX 5090, 8-bit band (the default) | 0.03801 | 0.03544 | 0.03609 | 0.04475 | 93.42% |
| RTX 5090, FP16 band | 0.03825 | 0.03588 | 0.03621 | 0.04489 | 93.42% |
| sm_86 card, FP16 band | 0.03777 | 0.03479 | 0.03590 | 0.04504 | 93.42% |

- The top-64 figure is a lower bound of the full-vocabulary KL. On the 5090's 8-bit dump it reads 5.8% under it
  (0.03801 against gate 1's 0.04034, which the rebuilt `span_logits` reproduces at every one of the 65,536
  positions), the same truncation on every row of the table.
- Argmax per position: the sm_86 card agrees with the 5090 on the same FP16 band at 98.57% of the positions (97.4%
  to 99.5% per window); the 5090's two bands agree with each other at 99.23%.

**Decode.** The branch's server on the sm_86 card under `--profile rtx50-16gb` (a card with more than 16 GB but
below 28 has no card class yet) answered three greedy chat completions of 256 tokens each with coherent text and no
error in its log. They follow the 5090's text for the first 125 to 551 characters and then diverge, as two
different numerics do.

**The 5090 sees no change.** On the branch, the same 16 windows' dense logits are byte-identical to main's
(32.5 GB compared), three greedy decodes of 256 tokens are byte-identical, and every SASS function of the two
default archives is identical (phase 0's comparison).

**The leaf's tests on the sm_86 card.** 139 CTest tests on the first run, 20 failing:

- 17 exercise kernels only the sm_120a set holds (FP4 W4A4, the oversized W8 instance, the FP8 image and exception
  routes, the MoE decode's cluster and staged arms) and stop at their first launch, `no kernel image`, by design.
  They now carry the label `sm120a-only`.
- `ignis_kernel_trellis_linear_test`: the head's 64-token call reached 1.04 of the stated bound, against 0.92 on
  the 5090, at the same relative RMS error (1.70e-3 against 1.69e-3). Tensor cores before sm_90 do not round the
  FP32 accumulation to nearest (Fasi, Higham, Mikaitis and Pranesh, 2021); the header now states a second term of
  2^-8 s below compute capability 9.0, and the test reads the card.
- `ignis_kernel_hq_route_agreement_test` trapped in the 8-bit band at its first codec-only arm. It now runs its
  default-precision arms on the FP16 band below 12.0, as the server does.
- `ignis_kernel_gdn_gating_proj_tests` (vendored) forces cooperative schedules sized for the RTX 5090's 170 SMs,
  which a card with fewer SMs cannot hold resident. The engine's own planner skips them; this test fails on any
  smaller card, RTX 50 included.

With the fixes, `ctest -LE sm120a-only` on the sm_86 card: 121 of 122 pass, the GDN gating test the one failure.

**The dense trellis decode body** (`ignis_kernel_trellis_linear_bench`, one token through the body at the quant's K
mix, 9,135.6 MB): on the 5090 it reads at 70.5% of the card's own measured streaming bandwidth, on the sm_86 card at
72%. Per SM, the sm_86 card's rate is 0.67 of the 5090's, clocks included.

## Finding

- Observed: the opt-in build runs the 27B's 16 GB trellis entry on an sm_86 card end to end, prefill on the FP16
  band and decode, and its quality against the BF16 checkpoint is the 5090's: the top-64 KLD is within 1.3% of the
  5090's on the same band, top-1 identical.
- Observed: on the 5090 the branch changes nothing, byte for byte.
- Observed: every kernel test failure on the sm_86 card is explained. Seventeen are the designed refusals, one is a
  test's assumption about the SM count, and two were test-side (a tolerance the hardware's accumulation exceeds by
  4%, a test that did not follow the card).
- Inferred: the trellis decode body on Ampere keeps the 5090's share of the card's bandwidth, so the 2026-11-02
  finding's worry (half the INT32 lanes per SM, an instruction-bound body far below the bandwidth) does not hold on
  this card. Its per-SM rate, 0.67 of the 5090's, caps a card whose bandwidth per SM is higher.

## Implications

- The 27B trellis entry is a candidate for Ampere cards as it is. What stands between it and a user is packaging,
  not kernels: a card class for 24 GB cards keyed on the architecture as well as the memory, release builds that
  carry sm_86, and the guide's list of supported cards.
- The FP16 prompt band costs no quality: on the 5090 it reads within 0.6% of the 8-bit band's KLD.
- A card whose memory bandwidth per SM is high may find the trellis body bound by its SMs rather than its memory.
  Which body a 24 GB Ampere card wants (trellis, or the NVFP4 artifact's decode through an A16 route) is a
  measurement on that card, not a reading of this one.

## Limits and unknowns

- One sm_86 card. No Ada card ran; `89;120a` only compiles.
- Absolute speeds are not reported here. The decode and prefill rates of one card do not transfer to another
  without its SM count, clocks and bandwidth.
- Text only: the vision path and the image model were not run (the image model is refused below 12.0 by design).
- DFlash2 ran under the server's defaults; its acceptance on Ampere was not compared with the 5090's here.
- The KLD figures are top-64; the full-vocabulary KLD was computed on the 5090 alone.

## Follow-ups

- [GitHub #456](https://github.com/gpillon/ignis/issues/456): the card class for 24 GB cards keyed on architecture,
  release builds with sm_86, the guide's supported cards.
- [GitHub #460](https://github.com/gpillon/ignis/issues/460): a cargo build without the CTest executables, which a
  two-architecture build makes twice as costly.
