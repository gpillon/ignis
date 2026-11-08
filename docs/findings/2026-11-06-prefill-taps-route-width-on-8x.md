# On compute capability 8.x a prefill's DFlash2 feature taps take the A16 MMA route from 26 columns

- Kind: experiment
- Status: current
- Observed: 2026-11-06
- Last verified: 2026-11-06
- Scope: kernel / the NVFP4 A16 MMA route's narrowest width per GPU generation (`ignis_nvfp4_a16_mma_min_tokens`), the DFlash2 drafter's context append in a prefill chunk
- Topic: host-and-gpu
- Related: [GitHub #521](https://github.com/gpillon/ignis/issues/521), [GitHub #502](https://github.com/gpillon/ignis/issues/502), [GitHub #516](https://github.com/gpillon/ignis/issues/516), [ADR 0080](../adr/0080-on-sm8x-the-trellis-prefill-decodes-each-matrix-once-per-chunk-and-multiplies-on-the-integer-mma.md)
- Superseded by: none
- Summary: Timed twice at T = 1 to 64 on an sm_86 card, the NVFP4 A16 MMA route beats the GEMVs on both DFlash2 feature-tap problems from 26 columns and loses at 25, so compute capability 8.x takes the taps' route from 26.

## Question

A prompt-reuse tail is 0 to 63 tokens. Its prefill chunk's DFlash2 feature
taps (the drafter's context append, problems `[5120, 25600]` and
`[6144, 5120]`) ran on the vendored GEMVs below the route's 64 columns. From
which width does the A16 MMA route beat the GEMVs for those two problems on an
sm_86 card?

## Evidence

- Method: `kernel/build/tests/ignis_nvfp4_a16_mma_bench --taps --iters 50`,
  built for `86;120a`, run twice on an sm_86 card. The `--taps` mode times both
  tap problems on the route and on the GEMVs at every width T = 1 to 64.
- Both runs agree on every conclusion below.
- `ignis_kernel_nvfp4_a16_mma_test` holds the route bit for bit to
  `ops::linear` on the taps dispatch, and to an FP64 oracle at T = 1 to 63.

## Finding

Observed:

- From T = 26, the route stays ahead of the GEMVs on both problems at every
  width up to 64.
- At T = 25 the GEMVs win.
- Below 26 the GEMVs' own times are irregular from one width to the next, so
  no single crossover holds there.

Chosen: `kPrefillTapsMinTokens8x = 26` (`kernel/src/nvfp4_a16_mma.cu`), for the
prefill taps on compute capability 8.x only. Every other call on 8.x, the
decode round's and the verify round's appends included, keeps the widths every
generation keeps: 64 columns, or 128 below 4,096 rows. The RTX 5090 keeps
those widths for every call.

## Implications

- A reuse tail of 26 to 63 tokens on an 8.x card no longer runs its feature
  taps on the GEMVs.
- The route's BF16 MMAs accumulate in FP32, which GeForce runs at half rate
  (ADR 0080). The decode and verify widths therefore stay at 64 and 128 on 8.x
  until a GeForce RTX 30 or RTX 40 card measures them.

## Limits and unknowns

- Measured on one sm_86 card. #516's session checks the value on a GeForce
  RTX 30 card, as it checks the decode widths.
- The taps' two problems only. Nothing here says where the crossover lies for
  any other NVFP4 problem on 8.x, nor on an RTX 40 card.

## Follow-ups

- [GitHub #516](https://github.com/gpillon/ignis/issues/516): the check on a
  GeForce RTX 30 card.
