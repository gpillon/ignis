# CONTEXT

Glossary for the `ignis` project. Vocabulary only — not a spec, not implementation notes.
When output names a domain concept, use the term as defined here.

## Engine & process

- **ignis** — the Rust inference engine for Qwen 3.8-27B on a single RTX 5090 (SM120a).
  Deliberately specialized: one model family, one GPU class.
- **Kernel leaf** — the C++/CUDA compute library linked into ignis behind a C ABI.
  The forward pass and all GPU compute live here; Rust owns everything above the
  **step** — scheduling, KV accounting, serving (ADR 0009).
- **Step ABI** — the C ABI between Rust and the leaf, named after its
  granularity: one call per *step* (model load, sequence lifecycle, a prefill
  span, a decode round), not one per operator. Device-resident — no host
  activation pointer crosses it (ADR 0009).
- **Program** — the leaf-side layer that owns the forward pass and the
  sequence state, on top of the vendored ops. Ours, not vendored (ADR 0009).
- **Vendored op** — an operator copied *verbatim* from the reference and
  tracked by a manifest (pinned commit, content hashes, recorded patches).
  Anything hand-written is not one, and carries no port claim (ADR 0010).
- **Lane** — a concurrent decode slot. Requests hold a lane while decoding.
- **Prefill lane** — the single global prefill that serializes all prefill work
  across requests; decoding on other lanes continues meanwhile.
- **Admission state machine** — the fairness machinery (protection, backfill class,
  temporal credit, frontier distance) deciding which request gets which lane.
- **Hot reload** — in-place model reload without restarting the server; the model
  lifecycle is decoupled from the server lifecycle.
- **Performance-first** — the organizing principle: correctness (a self-check
  of *sane* output) is a non-negotiable floor; above it, performance is the #1
  objective and tie-breaker for every scope, feature, and kernel decision.
- **North-star** — the holistic objective: "the best local coding engine" — max
  performance **and** agent parallelism that saturates the GPU in prefill *and*
  decode.
- **Self-bootstrapping** — the dev/test loop where the very runner used to build
  ignis (ninfer, running Qwen 3.8-27B) is running while we test; two engines
  share the 5090, so GPU testing is exclusive (ADR 0006).
- **Reference** — the ninfer stack: the source of the artifact, the vendored
  ops, the oracle recordings and the speed numbers. A reference, not a target
  to match token-for-token (ADR 0005 / 0007).

## Model

- **Layer kinds** — the model is 64 layers: **16 GQA** (full attention) and
  **48 GDN** (linear attention), hidden 5120, intermediate 17408, vocab 248320.
- **GQA layer** — a grouped-query attention layer: 24 query heads / 4 KV heads
  of 256, over paged KV, rotated by RoPE at the *token's* position.
- **GDN layer** — a gated-delta-net (linear attention) layer: **16 key heads
  and 48 value heads of 128**, whose recurrence carries the sequence's GDN
  slot instead of a KV cache.
- **Fused plane** — a single stored tensor holding several projections stacked
  by row (e.g. `attention/query_key_gate_value`, `mlp/gate_up`). Its row order
  is part of the artifact contract, not a guess.

## Weights & artifact

- **Artifact** — the `.ninfer` container: the native NInfer model artifact
  (base objects + grafted DFlash2 module). Carries NVFP4 tensors, BF16 exception
  tensors, W8G32 endpoints (embedding, output head), and frontend objects
  (tokenizer / chat template).
- **Object** — the unit inside an artifact. The binder must consume every object
  at bind time; an unconsumed object is a load failure.
- **Device view** — the artifact crate's export for a bound tensor: device
  pointer, storage layout, format and shape, so the program binds planes
  directly with no host dequant.
- **NVFP4** — the weight quantization format of most of the model: E2M1 values,
  an E4M3 group scale per 16 stored in a separate **scale plane**
  (`blockscale-k16-m128x4-v1` layout), and a per-tensor **weight divisor**.
- **hq-e8-2b** — the reference's HyperQuant KV cache format; the profile the
  owner actually runs. Adopted at G4; bf16 KV until then.

## Sequence state

- **Sequence handle** — the leaf-owned, opaque per-sequence object created by
  the step ABI: its KV pages, its GDN slot, its conv taps, its position and its
  last token. It is what a snapshot captures and a restore rebuilds.
- **KV page** — a fixed-size device page of the paged KV cache, addressed
  through a per-sequence block table. Page geometry is reported by the runtime
  and is what the scheduler's KV pool counts.
- **GDN slot** — one sequence's linear-attention state: per GDN layer, **48
  value heads × 128 × 128 fp32** (all 48 GDN layers ≈ 144 MiB per sequence),
  drawn from a slot pool sized by the concurrency.
- **Conv taps** — the per-sequence causal-conv1d history of a GDN layer.
- **KV-RAM** — the host-RAM KV cache tier: snapshots GPU lanes so sibling requests
  restore instead of re-prefilling; two-tier eviction (probation → protected).
- **Prefix reuse** — concurrent requests sharing a prefix skip the redundant prefill.
- **Chunked prefill** — prefilling a prompt in spans (1024 tokens) through the
  span+position prefill call, rather than one token at a time.
- **Decode round** — one traversal of the model for *all* decode-ready
  sequences in a batch; the unit a decode CUDA graph is captured over, per
  batch width.
- **N-lane concurrency** — 8 resident decode lanes (N=8), with overflow to the
  host KV-RAM tier; sized for a ~10-subagent concurrent coding workload.
- **DFlash2** — the 5-layer sliding-window (2048) speculative-decoding drafter
  (hidden 5120, draft tokens 1..7, native acceptance 3.4–3.7 tokens/round).
- **MTP** — the model's native multi-token-prediction heads (draft window 3,
  adaptive verification width).
- **Vision** — multimodal (image/video) input.

## Acceptance

- **Gate** — a milestone acceptance measured **on the GPU**, never on CPU
  tests. G1 correctness floor, G2 prefill, G3 decode + serving loop, G4 the
  reference feature floor, G5 speculative decoding (`.scratch/ROADMAP.md`).
- **GPU profile** — the explicit test profile for GPU work: it requires the
  5090 free and **fails**, never skips, when the GPU is busy or a kernel
  errors. A skip is not green for compute work.
- **Canary suite** — the fixed set of short, high-signal prompts used to detect
  divergences.
- **Canary oracle** — the recorded fixture the canary suite is checked against:
  the reference engine's greedy (exact-argmax) completions on those prompts for
  the same artifact, tokenized with the artifact's tokenizer. The G1 floor is
  ≥ 95% first-32-token agreement with it.
- **f64 layer reference** — a CPU fp64 computation of one GQA layer and one
  GDN layer on the real weights, the tolerance target for the leaf's per-layer
  output at G1.
- **Performance gate (99%)** — the acceptance criterion at G4: ≥ 99% of the
  reference's performance (throughput / latency) on the trace-replay load,
  **not** token-agreement;
  correctness is self-checked (sane output, same model, greedy, fixed seed).
  It is the first of a ladder of performance gates (later gates TBD).
- **Trace replay** — re-sending a recorded "1 main agent + N subagents" load trace
  against the engine to compare scheduler behavior with the ninfer baseline.
