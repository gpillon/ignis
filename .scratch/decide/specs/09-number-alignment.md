# 09 - the number prompt declares an alignment

GitHub: #254

`{"type": "number", "digits": 3}` over evidence that says *"failing for 3
days"* answered **300**, with the first digit at p = 0.998. The model knew the
answer; it did not know which end of the field to pad.

The cause is a gap between two things that were each correct on their own. The
schedule forces **exactly K digits** and carries no way to say a number is
finished (spec 06: a constrained decode "ends when its schedule is exhausted
and never on EOS"), so a value narrower than its field can only be rendered
left-aligned or zero-padded. And `number_system` declares the **width** and
says nothing about the **alignment**. Nothing named which, so the model chose —
at the first digit, after which every later digit follows at p ≈ 1.

- **The fix is one clause in `number_system`**: *"right-aligned and padded on
  the left with zeros."* Not a code path — there is no padding to implement,
  only an instruction that was missing.
- **It reaches `number` alone.** `point_system` and `box_system` are separate
  texts and are untouched, so the prompt the pointing finding measured still
  ships byte for byte and `the_measured_prompt_is_the_prompt_that_ships` still
  holds. A coordinate on a 0-999 scale fills its field anyway; the ambiguity
  never applied to it.
- **`DIGITS`' doc comment is wrong and is rewritten, not patched.** It says a
  width one past the value's own multiplies by ten while two or more pads on
  the left and is safe. Measured at every width on four truths, that rule fails
  on all four: 47 is right at four digits and comes back 4700 at five; 3 comes
  back 300 at three. The replacement rule is the whole rule: **any field that
  holds the value works, a narrower one truncates.**
- **The old tell goes with it.** The doc said a broken answer shows as the
  first digit falling to 0.65-0.71. That was a symptom of the alignment being
  contested, not of the width being wrong, which is why it did not fire on the
  300 that started this.
- **The clause is pinned at every width**, the way `point_system`'s text is.
  There is no code path to regress, so a reword that dropped it would restore
  the bug with nothing failing.

## Acceptance

1. `number_system` carries the alignment clause at every width in `DIGITS`,
   and still declares the `0`-`10^d - 1` scale.
2. `point_system(3)` and `box_system` are byte-identical to before.
3. On the GPU, over four truths of one, two and three digits, **every width
   that can hold the value reads it exactly** — 21 of 21, against 11 of 21
   before the clause. Only the after-walk is committed: the probe overwrites
   its own output and the before-run's file was gone by the time the fix was
   confirmed, so the 11/21 half lives in the finding's table and not in
   `.scratch/`.
4. A width narrower than the value truncates. That is the only thing a
   narrower field can do and it is not a fault to fix.
5. `uncertainty` does not count the padding. A leading zero on a `number` is
   a zero the prompt asked for, so a certain answer in a wide field reports a
   small sigma and not a large one — before this, `3` in a three-digit field
   reported 2.23.
6. A leading zero on a `point` or a `box` **does** count: there the width is
   the scale, an `x` of 031 is a coordinate in the first hundred, and that
   digit's doubt is worth 100 on a 0-999 axis. `decide_point_gpu.rs` still
   passes unchanged.

## References

- Finding: `docs/findings/2026-09-21-the-number-prompt-declares-an-alignment.md`
  — the width walk, before and after, with the per-digit traces.
- `docs/findings/2026-09-20-number-width-and-decide-e2e.md`, whose width rule
  this corrects.
- Spec 06 (`number`, `point`, `box`) and ADR 0034.
