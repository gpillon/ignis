---
status: accepted
---

# A second model, Qwen3.8-Flash-Next, served without a ninfer reference

ignis serves Qwen3.8-Flash-Next (125B MoE, 6B active) as a second model, for the tasks where the 27B falls short:
general knowledge, languages, mathematics, long technical documents. One of the two is loaded at a time, selected at
start or by a switch at runtime. ninfer has no implementation of this model (no MoE router, no grouped expert GEMM,
no sparse attention indexer, no hyper-connections, no n-gram embedding), so there is nothing to vendor and no engine
to record an oracle from or to race, and rules that assume one would make the model unservable: its oracle is the
checkpoint, its speed is gated against our own measured model, and its ops are our own.

- **The oracle is the checkpoint, not another engine.** The references are the checkpoint's own modeling code under
  transformers, run layer by layer and teacher-forced in the converter's single pass. A greedy BF16 generation is
  impossible here: 250 GB do not fit, and every layer-streamed token would re-read all of them. So there are two
  references:
  - **quality:** the BF16 weights. Their top-64 log-probabilities on fixed windows are the reference for a
    per-domain KLD (the readout and method of [the KLD finding](../findings/2026-09-24-kld-against-bf16.md)), and
    they give the MMLU-Pro proxy;
  - **engine numerics:** the same modeling code with the artifact's decoded weights, the *quantized reference*. Its
    per-position argmax on fixed canary sequences is the expected token for G1, whose floor is teacher-forced
    agreement of at least 95%, scored as ADR 0082 keeps it for the 27B's canary test. Both sides run the same
    weights, as the 27B and the ninfer that recorded its canary fixture do, so a miss is a kernel bug, not
    compression.
- **ExLlamaV3 is a kernel-level reference only.** ExLlamaV3 1.5.3 (MIT) does implement the model, but as a
  different engine with its own numerics: its router runs on fp16 logits and can pick different experts than the
  checkpoint's fp32 router. Its torch `*_ref` functions (the indexer, the n-gram embedding, the hyper-connection
  mix) and its trellis `reconstruct` check our decode of the expert format bit for bit and our indexer, n-gram and
  hyper-connection ops numerically. They never define what the model should answer.
- **Speed is gated against our own measured model, not against an engine.** No race against a reference engine
  gates this model: its spec (#301) states the speed floor and the simulation it is measured against.
- **Flash-Next's ops have no ninfer provenance and are our own implementation** (ADR 0010's label, the first of
  ADR 0051's three kinds of kernel code): the MoE router and expert kernels, the QSA indexer and sparse attention,
  the hyper-connection mix, the n-gram embedding and the FP8 row-scale linear. They carry no port claim. Each brings
  a kernel-leaf reference test at Flash-Next's real geometry with an fp64 or recorded reference, as vendored ops do.
- **A new geometry for a vendored op is our code, not a patch.** GQA at 24 query and 2 KV heads and GDN gating at
  hidden 2560 are outside what the vendored wrappers admit, and ADR 0037 admits a patch for a correctness bug only.
  A wrapper or kernel written for these shapes is our own implementation under this ADR, and the vendored files stay
  untouched. A vendored registry that lacks a shape is ADR 0088's case.
- **ExLlamaV3 (MIT) may be copied from as ADR 0051 allows, except its QTIP-derived code.** Code that ExLlamaV3
  marks as QTIP-derived (QTIP is GPL-3.0) is not read for implementation. The converter may depend on the exllamav3
  Python package as an offline tool (ADR 0044).

## Consequences

- Flash-Next fits the RTX 5090 only as a compressed MoE with its experts in host RAM, and compressed it keeps its
  lead over the 27B: an MMLU-Pro proxy of 73.0% against 68.3% on the same 281 questions
  ([the compression study](../findings/2026-10-04-flash-next-compression-study.md)).
- The 27B stays the default model.
- For this model only, this ADR replaces the canary oracle recorded on ninfer and the race against it. For
  Qwen3.8-27B the canary test stands as an error detector and the gates against ninfer are retired: no gate is a
  ratio against ninfer, and Flash-Next keeps this ADR's gate against its checkpoint (ADR 0082).
- Both references are recorded once, in the converter's pass (#299): a GPU-hours step, layer-streamed, not a ninfer
  run. No later BF16 run is needed.
- A Flash-Next regression is judged by its KLD and G1 numbers against the checkpoint, never by agreement with
  ExLlamaV3.
- Without a reference engine, a perfectly consistent but wrong implementation would pass every kernel test. The
  checkpoint oracle at the whole-model seam is the only defence against that, which is why the model's acceptance
  (#302) is taken at that seam.
