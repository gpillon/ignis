# CONTEXT

Glossary for the `ignis` project. Vocabulary only — not a spec, not implementation notes.
When output names a domain concept, use the term as defined here.

## Engine & process

- **ignis** — the Rust inference engine for Qwen 3.8-27B on a single RTX 5090 (SM120a).
  Deliberately specialized: one model family, one GPU class.
- **Kernel leaf** — the C++/CUDA compute library linked into ignis behind a C ABI.
  All GPU compute lives here; Rust owns scheduling, KV, and serving above it.
- **Lane** — a concurrent decode slot. Requests hold a lane while decoding.
- **Prefill lane** — the single global prefill that serializes all prefill work
  across requests; decoding on other lanes continues meanwhile.
- **Admission state machine** — the fairness machinery (protection, backfill class,
  temporal credit, frontier distance) deciding which request gets which lane.
- **Hot reload** — in-place model reload without restarting the server; the model
  lifecycle is decoupled from the server lifecycle.

## Weights & artifact

- **Artifact** — the `.ninfer` container: the native NInfer model artifact
  (base objects + grafted DFlash2 module). Carries NVFP4 tensors, BF16 exception
  tensors, and frontend objects (tokenizer / chat template).
- **Object** — the unit inside an artifact. The binder must consume every object
  at bind time; an unconsumed object is a load failure.
- **NVFP4** — the quantization format of v1 (weights and KV).

## Scheduling & cache

- **KV-RAM** — the host-RAM KV cache tier: snapshots GPU lanes so sibling requests
  restore instead of re-prefilling; two-tier eviction (probation → protected).
- **Prefix reuse** — concurrent requests sharing a prefix skip the redundant prefill.
- **DFlash2** — the 5-layer sliding-window (2048) speculative-decoding drafter
  (hidden 5120, draft tokens 1..7, native acceptance 3.4–3.7 tokens/round).
- **MTP** — the model's native multi-token-prediction heads (draft window 3,
  adaptive verification width).
- **Vision** — multimodal (image/video) input.
- **GDN state** — the recurrent state of the linear-attention (GDN) layers;
  resumable only at checkpoint/frontier boundaries.

## Acceptance

- **Parity gate** — the v1 acceptance criterion: ≥ 99% token agreement against the
  ninfer baseline (greedy, fixed seed) plus per-class ttft/tok-s within tolerance
  under trace-replay load.
- **Canary suite** — the fixed set of short, high-signal prompts used to detect
  divergences.
- **Trace replay** — re-sending a recorded "1 main agent + N subagents" load trace
  against the engine to compare scheduler behavior with the ninfer baseline.