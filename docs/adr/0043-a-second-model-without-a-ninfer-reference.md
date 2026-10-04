# ADR 0043 — a second model, Qwen3.8-Flash-Next, served without a ninfer reference

## Status

Accepted (2026-10-04): the owner approved the Flash-Next specs
(`docs/specs/flash-next/`) and asked for their tickets. Spec `flash-next/04`'s
acceptance verifies it; its numbers are written here when it holds.
**Amends** the "one model family" scope (`CONTEXT.md`, *ignis*).
**Extends ADR 0010**: ops with no ninfer provenance are our own
implementation. **Replaces ADR 0014's oracle and ADR 0015's gate for this model
only**; both stand unchanged for Qwen3.8-27B.

Sources: the compression study in
`.scratch/flash-next-compression-2026-10-03/` (`RISULTATI_3.md`, runs 1-8;
untracked, local to the owner's clone), the owner's /to-spec decisions of
2026-10-04.

## Context

ignis serves one model, Qwen3.8-27B, and every correctness and speed rule
assumes a reference engine running the same weights: vendored ops are ninfer's
files tested against ninfer's fp64 references (ADR 0010), G1 is teacher-forced
agreement with ninfer's recorded greedy completions (ADR 0014), G2 is a
live/live latency comparison against ninfer (ADR 0015).

The owner wants a second model, **Qwen3.8-Flash-Next** (125B MoE, 6B active),
for the tasks where the 27B falls short: general knowledge, languages,
mathematics, long technical documents. The study shows it fits the RTX 5090
only as a compressed MoE with experts in host RAM, and that compressed it keeps
its lead over the 27B (MMLU-Pro proxy 73.0% against the 27B's 68.3% on the same
281 questions).

ninfer has no implementation of this model: no MoE router, no grouped expert
GEMM, no sparse attention indexer, no hyper-connections, no n-gram embedding.
There is nothing to vendor and no engine to record an oracle from or race. The
rules that assume one would make the model unservable.

ExLlamaV3 1.5.3 (MIT) does implement it, with torch reference functions for the
indexer, the n-gram embedding and the hyper-connection mix. It is a different
engine with its own numerics: its router runs on fp16 logits and can pick
different experts than the checkpoint's fp32 router.

## Decision

**ignis serves Qwen3.8-Flash-Next as a second model, selected at start.** One
model is loaded at a time; switching at runtime is phase 2 (a later spec) and
nothing built here may preclude it.

**The oracle is the checkpoint, not another engine.**
- The references are the checkpoint's own modeling code under transformers,
  run layer by layer and teacher-forced in the converter's single pass. A
  greedy BF16 generation is impossible here: 250 GB do not fit, and every
  layer-streamed token would re-read all of them. So there are two references:
  - **quality:** the BF16 weights. Their top-64 log-probabilities on fixed
    windows are the reference for a per-domain KLD (the readout and method of
    the 2026-09-24 KLD finding), and they give the MMLU-Pro proxy;
  - **engine numerics:** the same modeling code with the artifact's decoded
    weights (the *quantized reference*). Its per-position argmax on fixed canary
    sequences is the expected token for G1.
- The G1 floor (teacher-forced agreement ≥ 95%) applies to Flash-Next against
  the quantized reference. Same weights, as for the 27B against ninfer, so a
  miss is a kernel bug, not compression.
- ExLlamaV3's torch `*_ref` functions and its trellis `reconstruct` are
  **kernel-level** references only: they check our decode of the expert format
  bit for bit and our indexer, n-gram and hyper-connection ops numerically.
  They never define what the model should answer.

**Speed is gated against our own measured model, not against an engine.**
There is no G2 race for this model. Spec `flash-next/03` states the speed floor
and the simulation it is measured against.

**Ops without ninfer provenance are our own implementation** (ADR 0010's label):
the MoE router and expert kernels, the QSA indexer and sparse attention, the
hyper-connection mix, the n-gram embedding and the FP8 row-scale linear. They
carry no port claim. Each brings a kernel-leaf reference test at Flash-Next's
real geometry with an fp64 or recorded reference, as vendored ops do.

**A new geometry for a vendored op is our code, not a patch.** GQA at 24 query /
2 KV heads and GDN gating at hidden 2560 are outside what the vendored wrappers
admit. ADR 0037 admits patches for correctness bugs only. A wrapper or kernel
written for these shapes is our own implementation under this ADR, and the
vendored files stay untouched.

**ExLlamaV3 code is read, never copied into the engine.** Our kernels are
original. Code that ExLlamaV3 marks as QTIP-derived (QTIP is GPL-3.0) is not
read for implementation. The converter may depend on the exllamav3 Python
package as an offline tool (ADR 0044).

## Consequences

- `CONTEXT.md`'s "one model family" becomes "two models, one loaded at a time";
  glossary entries for the new concepts land with the specs.
- A model-selection knob (Make and CLI) chooses the artifact family at start.
  The 27B stays the default and every 27B gate is unchanged.
- Both references are recorded once, in the converter's pass (spec 01). That
  is a GPU-hours step (layer-streamed, ~5 GB of BF16 per layer), not a ninfer
  run. No later BF16 run is needed.
- A Flash-Next regression is judged by its KLD and G1 numbers against the
  checkpoint, never by agreement with ExLlamaV3.
- Without a reference engine, a perfectly consistent but wrong implementation
  would pass every kernel test. The checkpoint oracle at the whole-model seam is
  the only defence against that, which is why spec `flash-next/04` makes it the
  acceptance seam.
