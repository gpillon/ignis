# 02 - study: what the vision tower costs at width, and whether that is the scheme

A **study**, not a fix. The vision encode's cost per column gets steadily
worse as the image gets bigger, and nobody has established whether that is
the roofline, the checkpoint's intended attention scheme, or an unexamined
path.

## What was measured

One live server, one screenshot rescaled to six sizes, one `point` question
each, `media.encode_seconds` from `ignis.request.admitted`:

| merged columns | encode | us/column | vs previous |
|---|---|---|---|
| 576 | 0.022 s | 37.9 | — |
| 1,296 | 0.063 s | 48.4 | cols x2.25, encode x2.87 |
| 2,304 | 0.137 s | 59.5 | cols x1.78, encode x2.19 |
| 5,184 | 0.516 s | 99.6 | cols x2.25, encode x3.76 |
| 9,216 | 1.430 s | 155.2 | cols x1.78, encode x2.77 |
| 16,384 | 4.111 s | 250.9 | cols x1.78, encode x2.87 |

Per-column cost is **6.6x worse** at 16,384 columns than at 576. Overall
exponent 1.56; the local exponent over the last step is **1.83**, close to
quadratic.

## The candidate explanation, unverified

`vision_item_control` builds `cu_seqlens` with **one segment per temporal
frame**. A still image is `t = 1`, so it is one segment covering every patch
— full attention over the whole grid, which is `O(N^2)` in the patch count
and would produce exactly this curve.

That may be right. The control is documented as "step for step the
reference's", and if the reference does full attention then so should we
(ADR 0010's reasoning, and the canary that scores it). What this study has
to establish is which of these is true, because they lead opposite ways:

1. The served checkpoint's tower is full-attention, ignis matches it, and
   4.17 s at 16K columns is the price of the image. Then the operator's lever
   is `--vision-max-tokens` and the engineering lever is slice 01's cache.
2. The checkpoint expects windowed attention in most layers with full
   attention in a few — the scheme Qwen2.5-VL introduced — and ignis is
   running every layer full. Then this is a correctness-shaped performance
   bug and the curve is the symptom.

## Questions

- Which attention scheme does the served checkpoint's vision config declare,
  and does the vendored `ninfer::ops::vision_attention` implement it?
- Where does the time actually go at 16,384 columns — attention, the MLP, the
  patch embedding, the merge? A per-stage profile at two widths separates a
  quadratic term from a linear one without any theory.
- Is `--vision-max-tokens` a usable lever in practice: what does pointing
  accuracy do as the budget drops? The pointing finding's numbers are all at
  the default budget, where a 4096px image is not downscaled at all. A
  cheaper budget that still lands inside the button would be worth more than
  any of this.
- What does the reference implementation take for the same image on the same
  card? That is the only number that says whether 4.17 s is fast or slow.

## Not in scope

Optimising anything. This ticket ends with a finding and, if it warrants one,
a follow-up that names what to change.

## References

- Finding: `docs/findings/2026-09-20-number-width-and-decide-e2e.md`.
- `crates/core/src/vision.rs::vision_item_control` (the segment rule),
  `kernel/src/vision_encode.cu`, `ninfer/ops/vision_attention.h` (vendored).
- GitHub #181 closeout measured the vision part at 21 ms against a
  reference's 63 ms — on an image three orders of magnitude smaller, which is
  why it says nothing about this width.
- ADR 0010 (a vendored op is verbatim), and
  `docs/findings/ignis-vendored-kernels-not-sacred` for when that decays.
