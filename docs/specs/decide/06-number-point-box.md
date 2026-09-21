# 06 - number, point and box

GitHub: #242

The primitive Jev does not have: a **number read digit by digit** over a
declared range. `point` is two of them and `box` is four. This is the only
decision primitive that generates tokens, and the only slice that touches the
kernel.

- **Permitted token set per lane** in the sampling ABI: the leaf samples only
  among the given ids and still returns a token id. No logits cross for this
  path, so it composes with temperature, seeds and speculation as decode does.
  The host-side alternative is rejected in ADR 0034 and must not creep back.
- `{"type": "number", "digits": 3}` - the mechanism exposed plainly. `min`/`max`
  is deliberately **not** offered yet: 0-999 is the model's native scale, and
  whether it obeys a declared range that is not its own is unmeasured.
- The answer carries `uncertainty` in units of the value, not a 0-1 score:
  `sigma = sum((1 - p_k) * 10^place)`, and the per-digit trace beside it.
  Documented as the model's **self-declared** uncertainty: on the measured
  sample it covers the true error on 4 of 6 axes.
- `point` and `box` are wire types, not two or four independent questions: the y
  digit is read after x has been forced, so the model knows where it put x. Two
  separate `number` questions would be two prefills that can contradict each
  other.
- The answer is in **pixels of the submitted image**, with the native 0-999
  reading beside it. The server knows the image's dimensions; making the caller
  normalize per axis is the mistake everyone makes once on a non-square image.

## Acceptance

1. A constrained decode returns only tokens from its permitted set, under greedy
   and under temperature.
2. No logits cross the `Compute` seam on this path.
3. A `point` over the committed pointing fixture lands inside the target button
   on all three scenes, matching `classify_pointing_gpu.rs` end to end through
   the server.
4. The reported `uncertainty` on those scenes matches the per-digit trace's
   place-weighted sum.
5. A non-square image returns pixels consistent with its own dimensions on both
   axes.
6. `digits` outside 1..=6 is a 422.

## References

- ADR 0034 (the permitted token set, and why not the host).
- Finding: `docs/findings/2026-09-19-constrained-digit-readout-points.md` -
  including its open follow-ups (a declared range other than 0-999; the x/y
  error asymmetry; restricted-argmax digits against the likeliest number).
- `crates/server/tests/classify_pointing_gpu.rs`,
  `crates/server/tests/fixtures/pointing/`.
