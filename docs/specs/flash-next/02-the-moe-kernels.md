# 02 - the MoE kernels: router, per-expert-K trellis GEMV and grouped GEMM, shared expert, combine

GitHub: #300 (master #298).

Every Flash-Next layer is a mixture of experts:
- 512 routed experts, of which the router picks 10 per token;
- one shared expert that every token uses;
- each expert is a SwiGLU MLP from hidden 2560 to 640 and back.

ignis has no MoE code at all: no router, no top-k, no grouped GEMM, no
trellis decode. This spec writes those kernels for ignis, as our own
implementation (ADR 0043). The expert weights are trellis-coded with a different
bit width K per expert (ADR 0044), which no public kernel supports. ExLlamaV3's
MoE kernel takes one K per projection per layer.

The kernels read experts from device slots and know nothing about where an
expert came from. Getting it into a slot is spec 03's job; composing the layer
around the kernels is spec 04's.

ADRs:
- 0043, a second model without a ninfer reference (Accepted 2026-10-04);
- 0044, experts in trellis with a K per expert (Accepted 2026-10-04);
- 0010, the reference-test convention our own ops follow;
- 0009, the kernel leaf and the step ABI.

## Decided for the autonomous run (2026-10-04)

The owner approved this spec on 2026-10-04 and vetoed none of the agent's
proposals: every *(proposed)* item below is decided, and ADRs 0043 and 0044 are
accepted. The prerequisites in Further Notes are the ticket's blockers on
GitHub, not open questions.

- Router rounding: the router test first records how the transformers
  reference rounds its logits, and the kernel matches it.
- The shared expert starts as a separate launch.
- The decode floor is 50% of the DRAM roofline.
- The committed reduced-shape decode fixtures are produced with the exllamav3
  package in `F:/ai/ngram-venv` (see spec 01), so this ticket can start before
  the full artifact exists. The full-shape, machine-local checks wait for it.

## Problem Statement

Flash-Next's speed on the 5090 comes from reading few bytes per token:
- per layer, a decode token touches 10 of 512 experts, at 2.5 bits per weight on
  average;
- that is about 15 MB per layer for the routed experts;
- at the card's ~1.8 TB/s, all 48 layers' experts cost about 0.4 ms per token.

That holds only if the expert kernels decode the trellis format at memory speed,
at four different K values within one launch, for one to three tokens in decode
and for thousands of tokens in prefill. Without these kernels the artifact of
spec 01 cannot run at all. With slow ones, all of the study's memory planning is
moot.

## Solution

A family of kernel-leaf ops, ours, each with a reference test at Flash-Next's
real geometry:

| op | what it does |
|---|---|
| router | fp32 logits over 512 experts, top-10, softmax over the ten chosen |
| expert decode GEMV | for 1-3 tokens, the selected experts' fused gate/up, SwiGLU and down, decoding each expert at its own K |
| expert grouped GEMM | for prefill chunks, tokens grouped by expert, the same math as tiled matrix products with the trellis decode in the inner loop |
| shared expert | the always-on expert in FP8, gated by a sigmoid of its own gate projection |
| combine | the routing-weighted sum of the ten expert outputs plus the gated shared expert |

The decoded weights match ExLlamaV3's `reconstruct` bit for bit, so the format
has an exact oracle. The kernels reach a stated fraction of the memory roofline
in decode.

## User Stories

1. As the owner, I want Flash-Next's experts decoded on the GPU at their stored 2-4 bits, so that the model runs from the compressed artifact without a dequantized copy.
2. As the owner, I want decode to read only the bytes of the ten experts each token selects, so that decode speed follows the 2.5-bit size of the experts.
3. As the owner, I want one to three decode tokens served by one launch per layer, so that three lanes cost little more than one.
4. As the owner, I want prefill of a long prompt to batch tokens by expert, so that each expert's weights are read once per chunk, not once per token.
5. As an engine developer, I want the router's logits computed in fp32 from the BF16 router weight, so that expert selection matches the checkpoint's except at genuine near-ties.
6. As an engine developer, I want the router's top-10 ties broken by the lower expert index, so that selection is deterministic run to run.
7. As an engine developer, I want the softmax taken over the ten chosen logits only (the checkpoint's `norm_topk_prob`), so that routing weights sum to one as the model was trained.
8. As an engine developer, I want the router to output expert ids and weights in device memory in a documented layout, so that residency (spec 03) and the expert kernels read the same selection.
9. As an engine developer, I want the expert kernels to take a per-layer slot table (expert projection → device slot address and K), so that they work wherever residency put the expert.
10. As an engine developer, I want one kernel launch to serve experts of all four K values, dispatching per expert, so that mixed K costs no extra launches or synchronization.
11. As an engine developer, I want the decode GEMV to apply the trellis decode, the Hadamard rotations and the channel scales in registers, so that no decoded weight is ever written to memory.
12. As an engine developer, I want the fused gate/up output passed through SwiGLU without a round trip to global memory where the tile allows, so that the intermediate costs no bandwidth.
13. As an engine developer, I want the down projection's partial sums combined with the routing weights inside the expert kernel or a single combine pass, so that the ten experts' outputs are not materialized separately per token.
14. As an engine developer, I want the shared expert run as an FP8 SwiGLU with its sigmoid gate, so that its weights (FP8, from spec 01) need no separate format.
15. As an engine developer, I want the combine to produce the MoE block's output in BF16 with fp32 accumulation, so that the hyper-connection mix (spec 04) receives the precision it expects.
16. As an engine developer, I want the prefill grouped GEMM to build its token-by-expert grouping on the device from the router's output, so that no host round trip sits inside a prefill chunk.
17. As an engine developer, I want the grouped GEMM to handle experts that receive one token and experts that receive hundreds in the same launch, so that the skew of real routing does not need a second kernel.
18. As an engine developer, I want all MoE ops capturable in a CUDA graph with fixed addresses per lane count, so that decode runs as graphs (spec 04).
19. As an engine developer, I want the ops' workspace sized from the topology and the maximum chunk at load, so that nothing is allocated while serving (ADR 0030).
20. As an engine developer, I want each op's reference test at Flash-Next's real geometry, so that a shape-specific bug cannot hide behind a toy shape.
21. As an engine developer, I want the trellis decode tested bit-exact against ExLlamaV3's `reconstruct` for every K value, so that the decoder's correctness is not a matter of tolerance.
22. As an engine developer, I want the GEMV and GEMM tested against an fp64 product over the exactly decoded weights, so that accumulation error is bounded and stated.
23. As an engine developer, I want a whole-MoE-block test on recorded real activations (router + ten experts + shared + combine) against a torch reference, so that the composition is checked, not only the parts.
24. As an engine developer, I want a microbenchmark that reports each op's achieved bandwidth against the device roofline for the bytes it reads, so that "fast" is a number.
25. As a reviewer, I want the kernels free of copied ExLlamaV3 or QTIP code, so that the engine's licence and the port-claim rules stay clean.
26. As a reviewer, I want every op labelled our own implementation, with no port claim (ADR 0010 / 0043), so that provenance is honest.

## Implementation Decisions

**Owner-made decisions:**
- experts carry a K per expert;
- the MoE kernels are ours;
- ExLlamaV3 is a reference only;
- the fast version comes directly.

*(proposed)* marks the agent's proposals, which the owner may veto.

**Ops live in the kernel leaf as program-side ops**, not vendored ops, with no
manifest entry. Each one has a C ABI entry used by the Flash-Next layer program
(spec 04) and a CTest executable.

**Router.**
- Input: the MoE block's normalized input (BF16, T × 2560) and the BF16 router
  weight.
- Output, per token: the ten expert ids (int32, sorted by descending logit) and
  their fp32 weights.
- Logits accumulate in fp32 *(proposed)*. ExLlamaV3 computes them in fp16. The
  router test first records how the transformers reference rounds its logits
  (activation dtype or fp32), and the kernel rounds the same way, so selection
  follows the oracle. The cost is negligible either way (one 2560 × 512 GEMV).
- Top-10 by a warp-level selection; ties go to the lower index; softmax over the
  ten.

**Slot table — the interface to residency.**
- Per layer, a device array indexed by (expert id, projection) giving a slot
  address and K.
- The expert kernels read it after residency has guaranteed that every
  projection the router selected is resident.
- The kernels never wait, never copy and never see host memory.
- A selected projection that is not resident is a programming error. In debug
  builds a sentinel address traps it.

**Trellis decode.**
- The format is ADR 0044's: mul1 codebook, 16×16 tiles, K ∈ {2, 2.5, 3, 4},
  128-wide Hadamard on both sides, fp16 input and output channel scales.
- The decoder is written from the format's definition: the bitshift trellis
  state and the mul1 hash. It is never copied from another engine's source.
- K is a per-expert runtime value. The kernels specialize the inner loop for the
  four K values with a switch per expert tile.
- Weights are decoded in registers and multiplied directly. The Hadamard
  rotation is applied to the activations (input side) and to the partial outputs
  (output side), as the format intends, so no rotated weight is materialized.

**Decode GEMV**, for 1-3 tokens.
- One launch per layer for the routed experts covers gate/up, SwiGLU, down and
  the weighted accumulation.
- Work is split over experts and output tiles. Partial outputs accumulate in
  fp32.
- The shared expert is a second launch, or a fused tail *(proposed: separate
  first, fused if the profile shows the launch matters)*.

**Prefill grouped GEMM.**
- From the router output, a device pass counts tokens per expert, builds offsets
  and permutes token indices.
- Tiled GEMMs per expert decode the trellis in the K-loop and use tensor cores
  on BF16 activations.
- A scatter applies the routing weights back to token order.
- One launch per projection family covers all experts, whatever their token
  counts. Small groups use a narrow tile path.
- The maximum chunk size is the program's prefill chunk, sized at load.

**Shared expert.** It is an FP8 per-row SwiGLU and uses the FP8 linear of spec 04
(gate/up, then down). Its output is gated by `sigmoid(x · w_gate)`, a 2560-to-1
projection, as in the checkpoint.

**Combine.** The output is (Σ routing weight × expert output) plus the gated
shared expert output, fp32 accumulation, BF16 out. It is written to the buffer
spec 04's hyper-connection step reads.

**Numerics.**
- The decoded weight is exact: it equals `reconstruct`'s fp16 values.
- Products accumulate in fp32. Activations enter as BF16.
- The tolerance against fp64 is stated per op in its test, from the accumulation
  length.

**Graphs and memory.**
- Every op's workspace (router outputs, permutation buffers, partial sums) is a
  plan line sized from the topology, the lane count (decode) and the prefill
  chunk (prefill), reserved at load.
- The ops are graph-capturable: fixed addresses, no host synchronization, no
  allocation.

**Performance contract.** Measured by a microbenchmark executable at real
geometry, decoding at K = 2.5 mean with the study's K mix:
- decode GEMV, 1 token, all ten experts resident: ≥ 50% of the device DRAM
  roofline for the bytes read *(proposed floor)*, with the achieved bandwidth
  reported;
- prefill grouped GEMM, 2048 tokens: TFLOP/s reported against the BF16
  tensor-core peak, with no floor in this spec.

## Testing Decisions

A good test feeds an op the inputs it gets in the model and checks the output it
must produce. Inputs: real geometry, real weights where available, recorded
activations. Checks: bit-exact decode, a bounded error against fp64, the exact
top-10 set. It never asserts on tile shapes, launch configurations or
intermediate buffers.

**Kernel-leaf CTest executables**, one per op family, at Flash-Next geometry.
Prior art: the vendored ops' reference tests run at 27B geometry against fp64
references (ADR 0010), and the leaf's own-implementation tests (the dflash2
top-k replacement).
- **Trellis decode bit-exactness:**
  - for each K value, a committed fixture of expert projections at reduced shape
    (a few 16×16-tile blocks), with their ExLlamaV3 `reconstruct` output, decoded
    by our kernel and compared exactly;
  - with the real artifact present (machine-local), full-shape projections of
    every K value, compared against fixtures recorded by the converter.
- **GEMV and grouped GEMM:**
  - against an fp64 product of the exactly decoded weights;
  - token counts 1, 2, 3 for GEMV and 1-4096 for GEMM, with skewed per-expert
    counts including an expert with zero tokens and one with the whole chunk.
- **Router:** fp32 logits against fp64, the top-10 set exact except at
  documented near-ties (gap below a stated ε), weights within tolerance. Tie
  ordering is deterministic.
- **Whole MoE block:**
  - recorded real activations from three layers (shallow, middle, deep) and the
    torch reference output from the converter's pipeline, produced with the
    decoded weights;
  - relative error ≤ the stated bound. Selection differences are counted and
    reported separately.

**Microbenchmark:** achieved bandwidth or throughput per op, written as a report.
It is not a pass/fail CTest except for the decode floor.

**Rust side:** the ABI bindings compile under `cargo check --features cuda
--tests`, and the CPU default suite is unaffected.

**GPU profile:** these tests run in the GPU profile, never skip there, one run
on the card at a time.

## Acceptance

1. Router: fp32 logits, top-10 with lower-index tie-break, softmax over the ten. CTest at 512 × 2560 green against fp64: the set is exact except at near-ties below the stated ε, which are counted.
2. Trellis decode is bit-exact against ExLlamaV3 `reconstruct` for K = 2, 2.5, 3 and 4, on committed reduced-shape fixtures and, machine-local, on full-shape projections from the real artifact.
3. Decode GEMV (1, 2 and 3 tokens) and prefill grouped GEMM (1-4096 tokens, skewed groups, empty experts) match an fp64 product of the decoded weights within the stated tolerance at real geometry. One launch per layer serves all four K values.
4. Shared expert (FP8 SwiGLU with sigmoid gate) and combine match fp64 within tolerance. The whole-MoE-block test on recorded activations from three depths is within the stated relative-error bound of the torch reference.
5. Every op is graph-capturable, allocates nothing while serving, and declares its workspace as plan lines sized at load. Every op is deterministic run to run for a fixed input and chunk split: no result depends on atomic accumulation order or on which cache slot an expert occupies (spec 05's bit-exact reuse relies on it). A test runs each op twice with experts in different slots and compares the outputs bit for bit.
6. Decode GEMV for one token reaches ≥ 50% of the DRAM roofline for its bytes at the study's K mix. The microbenchmark report gives the achieved figures for every op.
7. No ExLlamaV3 or QTIP source is copied. Every op is labelled our own implementation.
8. `cargo test` passes workspace-wide. `cargo check --workspace --features cuda --tests` is clean. The kernel build is clean and the leaf's op tests are green on a free 5090 under the GPU profile.

## Out of Scope

- Getting experts into slots: the host pool, the VRAM cache, prefetch and miss
  handling (spec 03).
- The layer program around the MoE: hyper-connections, attention, GDN, the
  n-gram embedding, the FP8 linear for non-experts (spec 04). The shared expert
  uses spec 04's FP8 linear.
- Expert-parallel or multi-GPU schemes; CPU expert compute (ExLlamaV3's AVX-512
  path; this CPU has no AVX-512).
- MTP and speculative verification shapes: Flash-Next runs without speculation
  in these specs.
- Kernel autotuning beyond the four K values and the decode/prefill split.

## Further Notes

- Prerequisites to `ready-for-agent`: ADR 0043 and ADR 0044, and spec 01's
  format codes and fixtures.
- The format reference is ExLlamaV3 1.5.3: its trellis codebook definition and
  tile layout (MIT). Its GEMV kernel describes itself as QTIP-derived (GPL-3.0);
  it is not read for implementation.
- Rough budget at 2.5 bits: a token's routed experts weigh about 15.4 MB per
  layer, or 0.74 GB over 48 layers. Decode from VRAM-resident experts is then
  ~0.4 ms per token at roofline. The real limit is residency (spec 03).
- The FP8 linear (spec 04) and the expert GEMM share the epilogue conventions
  (fp32 accumulate, BF16 out). Keep them consistent so the combine can fuse
  later.
