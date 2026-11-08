---
status: accepted
amends: [ADR-0043]
---

# ADR 0051 — kernel code is ours, adapted from a named source, or vendored

## Status

Accepted (2026-10-13, owner): the grilling session on Qwen-Image of
2026-10-13, decisions Q11 and Q20. Applies to all of ignis, not only to the
image model. **Amends ADR 0043**: its "ExLlamaV3 code is read, never copied
into the engine. Our kernels are original." becomes this ADR's three tiers.
**Extends ADR 0010**: a vendored op may come from a third party, not only from
the reference.

Amended 2026-10-14 (owner): MIT-CMU (HPND) joins the licences that may be
copied, for Pillow's resize, which `crates/server/src/media.rs` ports.

## Context

ADR 0010 made a vendored op a verbatim copy of the reference (ninfer) under a
manifest. ADR 0043 added that everything else is our own, and that other
engines' code is read, never copied.

The image model (ADR 0050) needs kernels ignis does not have:
- long non-causal attention;
- FP8 GEMM with FP8 activations;
- convolution;
- fused DiT normalization.

Well-optimized, permissively licensed versions exist for the 5090:
FlashAttention-2, SageAttention, CUTLASS 2.x, FlashInfer and ggml (survey:
[kernels for the image model on the 5090](../findings/2026-10-13-image-model-kernels-on-the-5090.md)).
Writing a known kernel again from scratch costs more than it teaches. Copying
parts of one costs little, provided the source and licence stay on record.

## Decision

A kernel in ignis is one of three things:

1. **Our own**: written here. It makes no port claim.
2. **Adapted op**: our code that contains parts copied from a named upstream.
   - The file header gives the upstream repo, commit, path and licence, and
     `kernel/NOTICE` lists the same.
   - We change it freely, with no parity claim.
   - It is tested like our own: against an fp64 or recorded reference at the
     real geometry.
3. **Vendored op**: a verbatim copy from a named upstream, either the
   reference or a third party, tracked by a manifest (pinned commit, content
   hashes, recorded patches).
   - ADR 0010's manifest is shaped for ninfer. A third-party vendored op gets
     its own manifest entry with the same fields.

**Licences.** Only MIT, BSD, Apache-2.0 or MIT-CMU (HPND) code is copied,
since ignis is Apache-2.0. MIT-CMU asks that its copyright and permission
notice travel with the copy, so copied MIT-CMU code puts that notice in the
root `NOTICE` (host code) or `kernel/NOTICE` (kernels). Nothing GPL, nothing
under a vendor's source licence. FlashInfer's
`csrc/fmha_v2` is under the TensorRT licence, and ExLlamaV3's QTIP-derived
code is GPL-3.0: neither is copied or read for implementation.

**Order of preference.** First search for what exists. Then copy the parts
that save real work, as an adapted op. Vendor a whole project only as a last
resort, when it removes much more work than it costs to carry.

## Consequences

- A port of SageAttention2++'s attention or FlashAttention-2's forward is an
  adapted op: ours to change, with the provenance in its header and in
  `kernel/NOTICE`.
- ExLlamaV3 (MIT) may now be copied from, except its QTIP-derived code. The
  converter's use of the exllamav3 package (ADR 0044) is unchanged.
- `GLOSSARY.md` gains *Adapted op*, and *Vendored op* widens from "copied from
  the reference" to "copied verbatim from a named upstream".
- A licence check is part of the review of any adapted or vendored op.
- Pillow (MIT-CMU) may be copied from: the Lanczos resize in `media.rs`
  matches Pillow byte for byte, which the condition images need, and its
  notice is in the root `NOTICE` (2026-10-14).
