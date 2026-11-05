---
status: accepted
---

# On sm_8x the trellis prefill decodes each matrix once per chunk and multiplies on the integer MMA

On RTX 30 and RTX 40 cards (compute capability 8.6 and 8.9), the 27B's trellis prefill runs today's fused kernel: it decodes the trellis inside the GEMM's inner loop and multiplies in f16. On a GeForce card that kernel is bound by its decode on the CUDA cores, not by the tensor cores. On the RTX 5090 it reaches about half of its own f16 peak, and the sm_8x cards share that balance per SM.

So on 8.x the body's trellis matrices take a second route, the **dense prefill**:
- **One decode per chunk.** Each matrix is decoded once per prefill chunk into a dense INT8 copy, one matrix at a time in a single buffer sized for the largest one: W8, q = clamp(round_half_even((s − 510)/4), ±127), with the matrix's affine map as one scale and a constant offset.
- **INT8 activations.** The activations get one INT8 scale per token after the input rotation (A8).
- **Our own integer GEMM.** It is an `mma.sync` s8 GEMM accumulating in int32 over the full K. The fp32 epilogue applies the scales and the rank-1 offset term, offset · scale_a · Σqa per token.

In a prototype on the RTX 5090 the dense route ran the body's prefill GEMMs about 2.5x faster per chunk than the fused kernel. The KLD against BF16 stayed within the gate, since the route's numerics were measured together with the 8-bit band below.

**The rule for new code on 8.x.** It uses only the MMA forms GeForce runs at full rate: s8/u8 to s32, and f16 with an f16 accumulator. f16 or bf16 with an f32 accumulator runs at half rate, and TF32 lower still. That is a product choice on GeForce, not a property of the architecture, so a kernel tuned on another card class would pick it up unnoticed. A SASS check on the `86` cubin enforces the rule. The kernels that already use halved forms and weigh little are listed as its exceptions (#488).

**Prompt attention on 8.x** takes an 8-bit band of its own.
- **Q·Kᵀ** in INT8, exact in int32.
- **P·V** on the u8×s8 MMA, with one Vᵀ scale and one fp32 fold per 128 keys, and O rescaled only when a row's running max grows past 2^8.
- **The window.** In the prefill pass the residual window's keys are quantized the same way, instead of a bf16 pass with an f32 accumulator; decode keeps their BF16 rows.
- **Speed and quality.** On the RTX 5090 this attention runs about 2x faster than the FP16 band at 8K and 64K. Together with the dense route, its KLD stays within 1.03x of the FP16 band's.
- **The option values.** `8bit` is the canonical value of `--model-prefill-attention` on 8.x, and `fp8` its alias; both resolve to the new band there and keep today's FP8 band on Blackwell. `int8` names the band on every card, and on Blackwell it runs only when named.

**What stays.**
- The fused route stays the route on sm_120. On 8.x it becomes an opt-in, `--model-prefill-linear fp16`.
- A value the card or the build cannot run falls back when it comes from a default or a profile, and is refused, naming the option, when the operator names it (#456). The build's own answer, `prefill_route_built`, is one predicate per route, so a build that does not carry a route yet refuses it by name instead of running the other.
- `--model-prefill-linear int8|fp16` (`model.prefill_linear`) chooses the route and `GET /v1/config` reports it, as it reports the band, with the card as its source when the card decided. Nothing changes by default until the routes are gated: `fp16`, the fused prefill, is the default on every card.
- One `86` cubin serves 8.6 and 8.9. A dedicated `89` path (spec C) hangs on `arch::GpuGeneration`.

## Considered options

- **Decode at load into a resident INT8 model.** It needs 24.3 GB for the 27B, more than a 24 GB card holds beside its KV.
- **INT8 inside the fused kernel.** It keeps the decode in the inner loop, the very thing that binds it on a GeForce card.
- **A library GEMM.** ignis's prefill kernels are its own. The prototype's GEMM is within a few points of a library's on the same tile shape.
- **Overlapping the next matrix's decode with the GEMM** (a double buffer), and **decoding in L2-sized slices.** On the RTX 5090 the overlap was slower, because both share the L2 the GEMM is bound on. On sm_86 the slices made the GEMMs too small to fill the card.

## Consequences

- **VRAM.** The dense buffer (89 MB on the 27B) is a line of the VRAM plan on 8.x. It costs a 16 GB card about ten thousand tokens of context.
- **A different decode cost.** The chunk width changes how often the decode is paid. That is one reason for #489's wider chunk when no lane is decoding.
- **A wider SASS check.** The release check's SASS rule covers the `86` cubin's new kernels.
