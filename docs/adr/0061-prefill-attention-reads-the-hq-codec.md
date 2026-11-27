---
status: accepted
---

# Prefill attention reads the hq-e8-2b codec: no prompt planes, on every card

Under hq-e8-2b a GQA prefill chunk of the 27B attends straight from the paged codec, on an attention kernel of
ignis's own (an adapted op, ADR 0051), and no prompt planes are materialized, so the workspace no longer grows with
`max_context`. The vendored prompt route first decoded the layer's whole visible history into BF16 planes, 4,096
bytes per context token up to 262,144 keys. Prefill interleaves with decode at chunk level (ADR 0018) and every
device byte is carved at load (ADR 0030), so the arena held that gigabyte for the whole load, on every card: on a 16
GB card it decided whether a 256K context fits (with the 3.0 bpw body and two lanes, the pool KV left beside one
full 256K sequence is 82K tokens with the planes and 199K without), and on the RTX 5090 it is about 116K more KV
tokens.

- **Every card class: the 27B's GQA layers, whichever artifact they run.** Flash-Next is not concerned: its QSA
  layers have their own prefill and never call the vendored prompt route.
- **Equivalence is numerical, not bit-exact** (owner). The decoded values are the same; the reduction order is not.
  It is held three ways: route agreement against ignis's BF16 route, as close as the vendored prompt route's within
  the tolerance the owner set and the route agreement test holds; the KLD unchanged within noise; and a cold prefill
  on the RTX 5090 no slower beyond the run's noise.
- **The history is decoded once per chunk and layer into a fixed band, not once per CTA tile** (owner). A CTA holds
  at most 128 query rows, since its FP32 accumulators fill the registers, so a decode per tile would decode each
  K/V row about 48 times a chunk and cost more than
  the whole attention. The residual window (the sinks, the ring before the chunk, the chunk itself) is attended in
  BF16, its rows exact as before, ADR 0037's ring-before-append included.
- **The band is 8-bit**: Q·Kᵀ on the INT8 MMA, exact in int32, and P·V on the block-scaled E4M3 MMA, which SM120
  runs at full rate with an FP32 accumulator; a GeForce card halves the BF16 MMA with an FP32 accumulator, and that
  halved peak is where the planes' kernel ran. Q is not E4M3: that rounds each query by about 2.5%, the same for every key it scores, a
  bias the route agreement test measured, and INT8 Q removes it. On an RTX 30 or RTX 40 card the 8-bit band is the
  integer band of ADR 0080.
- **An FP16 band with FP16 accumulate is the fallback**, behind `--model-prefill-attention fp16`: the 27B's only,
  fixed for the load, and a Flash-Next start refuses an explicit value.
- **The attention readout** (`/v1/decide`, `locate`) reads its keys from the same path, one key fetch shared with
  the attention and the tap, and loses its single-band restriction.

The figures are in the finding
[Prefill attention without prompt planes](../findings/2026-10-21-prefill-attention-without-prompt-planes.md).

## Considered Options

- **The planes in host memory**: a chunk rereads the history in all 16 layers, about 16 GB per chunk at 256K over
  PCIe.
- **Lending the planes' bytes to the KV pool between prefills**: pages live long, so taking them back means
  evicting a sequence, and it breaks ADR 0030.
- **Keeping the planes and shrinking their band to 64K keys**, a recorded patch to a vendored constant (ADR 0037,
  ADR 0031): it frees about three quarters of the bytes, its prefill cost was unmeasured, and the attention readout
  needed a single band. It was the fallback had the equivalence above failed.
- **One arena for the prefill scratch and the round buffers**: it frees little, and decode graphs need fixed
  addresses.
- **Bit-exact with the vendored route**: it would force the vendored kernel's reduction order onto the new one and
  cancel the design.
- **E4M3 throughout**, the bias above. Two probes that did not help are not kept: P encoded against each tile's own
  maximum, and a two-term E4M3 Q, as close as INT8 at twice the time.

## Consequences

- The vendored prompt route stays in the tree as the reference the route agreement test compares against, and
  serving never calls it. Its kernels are no longer the prefill bottleneck to optimize.
- The VRAM plan's workspace line drops the 4,096 bytes per context token; with vision on, that line is the
  item-bounded vision encoder's. The RTX 5090's profile spends the bytes on context.
