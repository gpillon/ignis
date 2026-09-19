# 01 - answer alphabet + readout seam

GitHub: #237

Carry a **readout** from the kernel leaf to the `Compute` seam: the logits of
named **answer tokens** at the prompt's last position, gathered before they
cross the seam. Nothing here serves a request yet; this is the path the rest
stands on.

- `StepLeaf::prefill` (`crates/runtime/src/lib.rs`) takes an optional logits
  buffer. `cuda_leaf.rs` already passes `None` to
  `step::prefill_program_sampled` at the bottom of it (GitHub #72).
- `PrefillJob` carries the answer token ids for the job; `PrefillOutcome`
  carries the readout back: the answer logits, the full-vocabulary log-sum-exp
  (for **answer mass**), and the unrestricted argmax.
- The gather happens **inside** `RuntimeCompute::prefill_step`. The
  full-vocabulary buffer is 248,320 x f32 = 970 KB per decision and must never
  cross the seam. (Corrected 2026-09-20 while implementing: this said 151,936
  x f32 = 607 KB, which is Qwen2/Qwen3's vocabulary, not this artifact's.
  ADR 0034 still carries the old figure and is the owner's to amend.)
- `PrefillOutcome` is `Copy` today. Breaking that touches its callers.
- The **answer alphabet** is computed from the loaded tokenizer at load:
  `A`-`Z`, `a`-`z`, `0`-`9`, then uppercase bigrams, admitting a label only if
  it encodes to exactly one token that decodes back to itself.
- `MockCompute` (`crates/core/src/mock.rs`) returns a deterministic readout, or
  the scheduler's CPU-only tests stop covering the path (ADR 0006).

## Acceptance

1. A readout requested through the `Compute` seam comes back with one logit per
   answer token, an answer mass in [0, 1], and the unrestricted argmax -
   verified against `MockCompute` with no GPU.
2. The answer alphabet rejects a two-token label. `BQ` is the named case: 114 of
   the 676 uppercase bigrams fail in the 27B's tokenizer, and admitting one
   would read another label's first-token logit.
3. The alphabet is built from the loaded tokenizer, not a constant: a test with
   a fixture tokenizer that splits a given label gets an alphabet without it.
4. No job that asks for no readout pays for one (no allocation, no gather).
5. `crates/server/tests/classify_readout_gpu.rs` still passes under the GPU
   profile, unchanged.

## References

- ADR 0034 (the leaf answers without generating).
- `CONTEXT.md`: **readout**, **answer token**, **answer alphabet**, **answer
  mass**, and the amended **per-lane sampling**.
- Finding: `docs/findings/2026-09-19-typed-option-logit-readout.md`.
- Existing measurement: `crates/server/tests/classify_readout_gpu.rs`,
  `slot_alphabet.rs`, `classify_option_ceiling_gpu.rs`.
