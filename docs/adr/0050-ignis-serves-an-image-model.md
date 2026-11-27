---
status: accepted
---

# ignis serves an image model, one model at a time

ignis serves an image model, Qwen-Image-2.1-Turbo (a Qwen3-VL-8B text encoder, a 7.1B single-stream diffusion
transformer and an RGBA VAE, sampled in 8 denoising steps with fixed sigmas and no classifier-free guidance), for
generation and editing on the text models' card, from the Playground and from OpenAI-compatible clients. It is a
third model family, reached by a switch: a catalog entry like Flash-Next, loaded with the whole card and never
resident beside a text model. The switch, the catalog, idle unload and the shell of the server already existed and
take any artifact a loader can prepare, and the 32 GB card does not hold both kinds at useful quality. "One model at
a time" holds between the text models and the image model: the speech model is loaded beside either, on the host,
and is never reached by a switch (ADR 0067).

- **A model kind sits above the family**: text model or image model here, and ADRs 0067 and 0069 add theirs. The
  server refuses a request of the wrong kind, or switches to the model the request names when implicit switches are
  allowed. The token machinery (scheduler, KV pool, lanes) is shaped for tokens and stays the text models' own.
- **Correctness against diffusers, speed from the literature.** The oracle is the diffusers pipeline in fp32.
  Correctness is checked by per-op tests against fp64 and one end-to-end test that compares each denoising step with
  a recording. The speed target is 90-100% of the best published implementation on the 5090, per kernel class,
  measured once at the end of the phase.
- **8 bits from the first version**: linears in FP8 with FP8 activations, transformer attention in 8 bits (INT8 QK,
  FP8 PV, SageAttention2++'s scheme), and the condition prefix stored in that 8-bit form. At 2048² the attention is
  most of the transformer's work: in bf16 it alone would take about 5.5 s of an 8-step image, in 8 bits about 1.8 s
  ([finding](../findings/2026-10-13-image-model-kernels-on-the-5090.md)).

## Considered Options

- **A separate image process**, and **a co-resident pair of 4-bit models**: rejected for the two reasons above.
- **Qwen-Image-2512**, the latest Apache-2.0 model: a 20B transformer that does not fit the card at useful quality.
- **The 2.1 base model**: 5× the denoising steps for no use the owner needs.

## Consequences

- The image model has no performance gate and no reference engine to race, the exception ADRs 0005 and 0007 record
  for it: its speed is a target taken from the literature, not a measurement against an engine. ADR 0082's gate,
  which admits a text family's artifacts, leaves it the checks `docs/agents/testing/image-model.md` names.
- 8 bits is lossy, and only the end-to-end comparison with diffusers checks it. In return the attention stops
  dominating at 2048², and twice as many condition images fit at full size: about 1 GiB of condition prefix per
  1024² condition image instead of 2.
- The weights are under the Qwen Research License: non-commercial use, a "Built with Qwen" notice, and "Qwen" never
  the primary name of a derivative. The published artifact is `gpillon/ignis-image-2.1-turbo`, served as
  `ignis-image-2.1-turbo`.
- Every text-to-image round trip pays two reloads (text model, image model, text model). Prompt enhancement before
  an image job uses the text model if it is loaded, or a visit to it otherwise.
- Serving has a second kind of work: an image job queue beside the engine, with its own idle accounting. Routes
  for one kind answer `409 model_kind_mismatch` while the other kind is loaded, unless the request names a model to
  switch to.
