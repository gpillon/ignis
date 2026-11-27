---
status: accepted
---

# On sm_8x the trellis prefill decodes each matrix once per chunk and multiplies on the integer MMA

On RTX 30 and RTX 40 cards (compute capability 8.6 and 8.9) the 27B's prefill runs routes of ignis's own, written
only in the tensor-core forms a GeForce card runs at full rate. The fused trellis kernel, which decodes the trellis
inside the GEMM's inner loop and multiplies in f16, is bound on a GeForce card by its decode on the CUDA cores, not
by the tensor cores: on the RTX 5090 it reaches about half of its own f16 peak, and the sm_8x cards share that
balance per SM.

- **The dense prefill.** Each of the body's trellis matrices is decoded once per prefill chunk into a dense INT8
  copy, one matrix at a time in a single buffer sized for the largest one, with the matrix's affine map as one scale
  and a constant offset. The activations get one INT8 scale per token after the input rotation, and the product is
  our own `mma.sync` s8 GEMM accumulating in int32 over the full K, its fp32 epilogue applying the scales and the
  offset's rank-1 term. In a prototype on the RTX 5090 it ran the body's prefill GEMMs about 2.5x faster per chunk
  than the fused kernel, and the KLD against BF16 stayed within the gate, measured together with the integer band.
- **The rule for new code on 8.x.** It uses only the MMA forms GeForce runs at full rate: s8/u8 to s32, and f16 with
  an f16 accumulator. f16 or bf16 with an f32 accumulator runs at half rate, and TF32 lower still. That is a product
  choice on GeForce, not a property of the architecture, so a kernel tuned on another card class would pick it up
  unnoticed. A SASS check on the `86` cubin enforces the rule, and the release check covers the new kernels; the
  kernels that already used halved forms and weigh little are listed as its exceptions.
- **The integer band.** Prompt attention on 8.x (ADR 0061) takes an 8-bit band of its own: Q·Kᵀ in INT8, exact in
  int32, and P·V on the u8×s8 MMA. In the prefill pass the residual window's keys are quantized the same way,
  instead of a bf16 pass with an f32 accumulator; decode keeps their BF16 rows. On the RTX 5090 it runs about 2x
  faster than the FP16 band at 8K and 64K, and together with the dense prefill its KLD stays within 1.03x of the
  FP16 band's.
- **The GDN prefill runs ignis's own kernels in those forms, on every card that runs the 27B, with no option.**
  Their products are f16 MMAs with an f16 accumulator, folded into fp32, and against an fp64 recurrence their rows
  are closer than the vendored kernels' (0.93 against 2.6 bf16 roundings RMS on the leaf test's synthetic inputs).
  On 8.x they are the route because of the rule above. Blackwell took them by the rule that they replace the
  vendored chunked kernels there only if the RTX 5090 measures them faster with its KLD holding, which it did on
  Blackwell's default routes, the fused prefill and the FP8 band, with one binary for both routes
  ([finding](../findings/2026-11-08-blackwell-gdn-prefill-on-ignis-kernels.md)): the kernels alone ran 2.4 to 2.6
  times faster; the time to first token was shorter on both 27B entries over alternated legs, on the NVFP4 entry
  within each regime of its legs (the finding says why its registered rule's second part failed there); and the
  full-vocabulary KLD against BF16
  stayed within its interval on both. The vendored kernels stay the leaf test's oracle and the other route of
  `IGNIS_GDN_PREFILL=vendored|ours`, a switch for a measurement on every card, not a setting. Flash-Next keeps its
  own vendored GDN layers.
- **On 8.x these routes are the default, and the routes they replace stay one option away**:
  `--model-prefill-linear fp16` for the fused prefill, `--model-prefill-attention fp16` for the FP16 band. On sm_120
  the fused prefill and the FP8 band stay the routes, and the dense prefill and the integer band run there only when
  named.
- **Only a field left to its default takes the card's own route**; a profile's value, `fp16` included, is kept
  where the card runs it. A value the card or the build cannot run falls back when it comes from a default or a
  profile, and is refused, naming the option, when the operator names it. The build answers with one predicate per
  route, so a build that does not carry a route refuses it by name instead of running the other.
- **One `86` cubin serves 8.6 and 8.9.** A path of its own for 8.9 would hang on `arch::GpuGeneration`.

## Considered Options

- **Decode at load into a resident INT8 model.** It needs 24.3 GB for the 27B, more than a 24 GB card holds beside
  its KV.
- **INT8 inside the fused kernel.** It keeps the decode in the inner loop, the very thing that binds it on a GeForce
  card.
- **A library GEMM.** ignis's prefill kernels are its own. The prototype's GEMM is within a few points of a
  library's on the same tile shape.
- **Overlapping the next matrix's decode with the GEMM** (a double buffer), and **decoding in L2-sized slices.** On
  the RTX 5090 the overlap was slower, because both share the L2 the GEMM is bound on. On sm_86 the slices made the
  GEMMs too small to fill the card.

## Consequences

- **VRAM.** The dense buffer is a line of the VRAM plan on 8.x. The prefill scratch is sized by the routes the load
  runs, so the dense prefill and our GDN prefill give some of it back.
- **A different decode cost.** The chunk width changes how often the decode is paid. That is one reason for the
  wider chunk while no lane is decoding (ADR 0018).
- **Blackwell's GDN prefill.** Its speed is unmeasured on the RTX PRO 6000 and the 16 GB RTX 50 cards, which run
  the RTX 5090's SASS. A greedy answer on Blackwell can differ from an earlier release's at a near-tie: correctness
  is self-checked, not byte-matched (ADR 0007). When Blackwell first took the kernels, three tests of the GPU
  profile moved and none was a quality loss of ours; the finding says what each read and the rule each has since.
