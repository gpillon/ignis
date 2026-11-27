---
status: accepted
---

# Kernel code is ours, adapted from a named source, or vendored

A kernel in ignis is one of three things, and the rule applies to all of ignis, not only to the image model that
prompted it. Writing a known kernel again from scratch costs more than it teaches, and well-optimized, permissively
licensed versions of the kernels ignis lacked exist
([survey](../findings/2026-10-13-image-model-kernels-on-the-5090.md)); copying parts of one costs little, provided
the source and the licence stay on record.

1. **Our own**: written here. It makes no port claim.
2. **Adapted op**: our code that contains parts copied from a named upstream.
   - The file header gives the upstream repo, commit, path and licence, and `kernel/NOTICE` lists the same.
   - We change it freely, with no parity claim.
   - It is tested like our own: against an fp64 or recorded reference at the real geometry.
3. **Vendored op**: a verbatim copy from a named upstream, either the reference or a third party, tracked by a
   manifest (pinned commit, content hashes, recorded patches). ADR 0010's manifest is shaped for the reference; a
   third-party vendored op gets its own manifest entry with the same fields.

**Licences.** Only MIT, BSD, Apache-2.0 or MIT-CMU (HPND) code is copied, since ignis is Apache-2.0. MIT-CMU asks
that its copyright and permission notice travel with the copy, so copied MIT-CMU code puts that notice in the root
`NOTICE` (host code) or in `kernel/NOTICE` (kernels). Nothing GPL, and nothing under a vendor's source licence:
FlashInfer's `csrc/fmha_v2`, under such a licence, and ExLlamaV3's QTIP-derived code, GPL-3.0, are neither copied
nor read for implementation.

**Order of preference.** First search for what exists. Then copy the parts that save real work, as an adapted op.
Vendor a whole project only as a last resort, when it removes much more work than it costs to carry.

## Consequences

- Code of another engine may be copied from under these terms, where the rule before was that it is read and never
  copied: ExLlamaV3 (MIT) may be copied from, except its QTIP-derived code. The converter's use of the exllamav3
  package (ADR 0044) is unchanged.
- A licence check is part of the review of any adapted or vendored op.
- Pillow (MIT-CMU) may be copied from: the Lanczos resize in `crates/server/src/media.rs` matches Pillow byte for
  byte, which the condition images need, and its notice is in the root `NOTICE`.
