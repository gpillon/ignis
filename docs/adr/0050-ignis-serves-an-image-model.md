---
status: accepted
amends: [ADR-0005, ADR-0007]
amended-by: [ADR-0067]
---

# ADR 0050 — ignis serves an image model, one model at a time

## Status

Accepted (2026-10-13, owner): the grilling session on Qwen-Image of
2026-10-13, decisions Q1-Q34. Specs `docs/specs/image-model/`. **Amends** the
scope of `GLOSSARY.md`'s *ignis* (text models and an image model) and, for this
model only, ADR 0005/0007's "at least as fast as the reference".

**Amended by ADR 0067** (2026-10-24): "one model at a time" holds between the
text and image models. The speech model is loaded beside either, on the
host, and is never reached by a switch.

Sources: the session's decision log and the kernel survey, kept untracked in
`.scratch/qwen-image-grill-2026-10-13/` (local to the owner's clone), and the
finding [kernels for the image model on the 5090](../findings/2026-10-13-image-model-kernels-on-the-5090.md).

## Context

ignis serves text models: Qwen3.8-27B, and Flash-Next since ADR 0043. The
owner wants image generation and editing on the same card, from the
Playground and from OpenAI-compatible clients.

The model is **Qwen-Image-2.1-Turbo**: a Qwen3-VL-8B text encoder, a 7.1B
single-stream diffusion transformer and an RGBA VAE, sampled in 8 denoising
steps with fixed sigmas and no classifier-free guidance. The latest
Apache-2.0 model, Qwen-Image-2512, is a 20B transformer that does not fit the
card at useful quality. The 2.1 base model costs 5× the steps for no use the
owner needs.

Nothing in ignis computes this model today: no convolution, no modulation,
no long non-causal attention at head size 128, no image route. The server's
engine, lanes and KV pool are shaped for tokens. The switch, the catalog and
idle unload are not: they take any artifact a loader can prepare.

## Decision

**The image model is a third model family, reached by a switch.** It is a
catalog entry like Flash-Next, loaded with the whole card and never resident
beside a text model. The token machinery (scheduler, KV pool, lanes) stays
the text models' own. A **model kind** (text model, image model) sits above
the family: the server refuses a request of the wrong kind, or switches to the
model the request names when implicit switches are allowed.

We chose this over a separate image process and over a co-resident pair of
4-bit models:
- the switch, the catalog, idle unload and the shell of the server already
  exist;
- the 32 GB card does not hold both kinds at useful quality.

**Correctness against diffusers, speed from the literature.** The oracle is the
diffusers pipeline in fp32. Correctness is checked by per-op tests against
fp64 and one end-to-end test that compares each denoising step with a
recording. There is no performance gate and no reference engine to race.
The target is 90-100% of the best published implementation on the 5090, per
kernel class, measured once at the end of the phase.

**8 bits from the first version.**
- linears in FP8 with FP8 activations;
- transformer attention in 8 bits (INT8 QK, FP8 PV, SageAttention2++'s
  scheme);
- the condition prefix stored in that 8-bit form.

At 2048² the attention is most of the transformer's work. In bf16 it alone
would take ~5.5 s of an 8-step image; in 8 bits it takes ~1.8 s.

## Consequences

- Unlike every text family, the image family has no performance gate. Its
  speed is taken from the literature, not measured against a reference
  engine: a deliberate departure from ADR 0005/0007.
- 8 bits is lossy, and only the end-to-end comparison with diffusers checks
  it. In return the attention stops dominating at 2048², and twice as many
  condition images fit at full size: about 1 GiB of condition prefix per
  1024² condition image instead of 2.
- The weights are under the Qwen Research License:
  - non-commercial use;
  - a "Built with Qwen" notice;
  - "Qwen" never the primary name of a derivative.
  The published artifact is `gpillon/ignis-image-2.1-turbo`, served as
  `ignis-image-2.1-turbo`.
- Every text-to-image round trip pays two reloads (text model -> image model
  -> text model). Prompt enhancement before an image job uses the text model
  if it is loaded, or a visit to it otherwise.
- The serving path learns a second kind of work: an image job queue beside
  the engine, with its own idle accounting. Routes for one kind answer
  `409 model_kind_mismatch` while the other kind is loaded, unless the
  request names a model to switch to.
