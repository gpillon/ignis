---
status: accepted
---

# The 27B's BF16 exception tensors can be held in FP8, converted at load

The 27B's NVFP4 artifact keeps nine matrices in BF16 among its NVFP4 body, and prefill multiplies them on the vendored
BF16 GEMMs at well under half the rate of the image model's FP8 W8A8 GEMM. So a load can hold the nine on the device
in FP8 row-scale, converted from the stored BF16 at load, on the GPU, and the artifact does not change: the weights
line drops 534.6 MB, which the KV pool takes, cold prefill is about 3% faster at 14K-105K, decode rounds take as long
and the 4K KLD is unchanged ([the finding](../findings/2026-10-21-the-bf16-exception-tensors-in-fp8.md)).
`model.exception_weights`, a field of the 27B alone, says which, `bf16`, `fp8` or `auto`, and `auto`, the default, is
`fp8` where the card runs it.

- **The leaf dispatches on the tensor's format**, on kernels of our own
  ([ADR 0051](0051-kernel-code-is-ours-adapted-or-vendored.md)): a narrow call, a decode round, runs the FP8 weights
  with BF16 activations, and a wide one, a prefill chunk, quantizes its activations per token and runs W8A8.
- **`auto` follows the card.** It is `fp8` on a card that runs the block-scaled FP8 MMA, compute capability 12.0
  (the RTX 5090 and the RTX PRO 6000), and `bf16` on any other card and when no card is read. A named `bf16` is kept
  on every card. A named `fp8` is refused below 12.0, naming the card. The 16 GB and 12 GB RTX 50 cards resolve
  `fp8` too, but their trellis artifacts hold no BF16 exception tensor, so nothing converts.
- **Why FP8 is the default where it runs.** At its served shape it passed
  [ADR 0082](0082-the-gate-is-kld-against-bf16-dflash2-on-against-off-and-the-loop-rate.md)'s KLD leg for a route
  made a default, on the RTX 5090
  ([the leg](../findings/2026-11-18-the-27bs-fp8-exception-weights-at-their-served-shape-on-the-rtx-5090.md)); the
  RTX PRO 6000 runs the same kernels and takes the same default, its KLD not measured. Two pre-registered decision
  fixtures read differently under FP8, and that is no criterion for the weights: the sign the scalar balance loses
  is a coin in BF16 too, which misses it at its served shape, and FP8 moves the entailment set about as far as one
  harmless switch of route does. Both sets were re-registered on arms that name their exception weights, context and
  rope before the default moved
  ([the finding](../findings/2026-11-18-the-decide-regression-sets-under-both-exception-weights.md)).
- **The default lives in config resolution**, beside the card reading
  ([ADR 0063](0063-the-profile-follows-the-card-class.md), [ADR 0046](0046-one-declaration-per-config-field.md)):
  `auto` resolves into the load's weights, while `GET /v1/config` and a file `config generate` writes keep `auto`.
  The engine's own default stays `bf16`: a test or a tool that binds the model directly keeps the stored numerics
  unless it names FP8.

## Considered Options

- **FP8 in the artifact:** the owner's condition was no conversion at load if it lengthened the load, else the FP8
  goes into the artifact; measured, it does not. The artifact would cost a new 32 GB download and a new content
  hash, for a conversion the GPU does in milliseconds.
- **FP8 in prefill only, BF16 kept for decode:** decode would stay bit-identical, but it takes either both copies
  (+0.53 GB) or a conversion per chunk into a 73 MB scratch past the vision arena (about 0.5 µs per token at 2,048):
  the VRAM would go the wrong way.
- **Block-scaled activations (MXFP8, an E8M0 scale per 32):** not needed, per-token scales measured within the KLD's
  noise.
- **FP8 opt-in, `bf16` the default:** it leaves 534.6 MB of the RTX 5090's KV pool and 3% of cold prefill unused for
  a reading of the two fixtures that the measurement does not support.
- **FP8 by default on every card that loads the NVFP4 artifact:** that is compute capability 12.0 alone, so it is
  this decision. On an RTX 30 or RTX 40 card the W8A8 GEMM the FP8 prefill needs does not run.

## Consequences

- Greedy text changes against a BF16 load: the weights and the prefill's activations are other numbers.
- What a default install keeps on the RTX 5090 follows what the desktop leaves free: FP8's smaller plan keeps the
  profile's context on a desktop where BF16 adapts, and the rope table, which follows the context, moves only there
  (the leg's finding).
- The KV-disk blob's identity does not name the encoding: by its own rule
  ([ADR 0029](0029-cross-request-state-reuse.md)) a knob that moves numerics but leaves the state's meaning, as the
  prefill chunk width and the prompt attention's precision do, is not identity.
