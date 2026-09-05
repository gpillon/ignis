# runtime — the device-resident inference runtime (the reset)

Feature area created 2026-09-05 from the project review
(`.scratch/REVIEW-2026-09-05.md`). It replaces the `kernel-abi` feature area:
the per-op, host-pointer C ABI and the host-resident compute adapter are
superseded by a **step-level, device-resident** runtime whose forward pass
lives in the kernel leaf.

Specs:

- `specs/01-device-resident-forward.md` — the correct forward pass at the
  step ABI, batch 1, bf16 KV, verified against an oracle (gate G1).

Later specs (not yet written; the phase plan is in the review §6): chunked
prefill (G2), batched decode rounds + per-width CUDA graphs + sampling (G3),
the ninfer feature floor — hq-e8-2b, device prefix reuse, KV-RAM tier, tagged
lanes (G4), speculative decoding (G5).

Roadmap and ticket map: `.scratch/ROADMAP.md` (G1 master #36, tickets #37–#62; masters #63–#66 for G2–G5).
