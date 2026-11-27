---
status: accepted
---

# A vendored file may carry a patch that fixes a known correctness bug

A vendored file may carry a recorded patch that fixes a known correctness bug, the second reason for one beside ADR
0031's measured bottleneck, through the mechanism ADR 0010 already has: the manifest entry's `patch`, its diff under
`kernel/vendor/patches/`, `scripts/vendor-ninfer.ps1 record-patch`. Without it every bug the reference ships is a
bug ignis must keep for as long as the file is vendored, which contradicts ADR 0005, where correctness is a floor
and not a trade, and ADR 0007, which refuses to define correct as agreeing with another engine. What ADR 0010
protects is the port claim, that vendored means the reference's file byte for byte or a recorded patch of it, and a
recorded, reviewed, tested patch keeps that claim intact; every provenance rule of ADR 0010 stands.

The patch is admitted on these terms:

- **The bug is demonstrated, not suspected.** It is a behaviour that violates the op's own contract (causality, the
  numerical contract its header states, memory safety), shown by an observation in ignis, not by a reading of the
  source.
- **It comes with a test that fails without the patch.** The test pins the contract the patch restores, runs in the
  suite that covers the op, and fails against the unpatched file. A patch whose test would pass on the reference's
  code is not a correctness patch.
- **It is the smallest change that restores the contract.** When the fix is no longer recognizably the reference's
  code, ADR 0031's option (b) applies instead: our own implementation, which leaves the port claim behind.
- **The `reason` names the bug and the test**, so the review question, why is this vendored file not the
  reference's, is answered from the manifest.

From the first such patch on, ignis distinguishes two behaviours, and every comparison with the reference says which
one it measured:

- **Reference parity**: ignis computes what the reference computes. The default for every vendored op, and the only
  behaviour an unpatched file can have.
- **Ignis patched**: ignis deliberately computes something else, because the reference is wrong there. Each such
  behaviour is listed in `kernel/vendor/VENDOR.md` with what the reference does, what ignis does instead, the test
  that tells them apart and the issue it landed under.

A result that differs from the reference because of an Ignis-patched behaviour is a declared departure, not a
regression: nobody "fixes" it back to parity. A comparison that crosses one (a gate cell, an equivalence run, a
finding) names it.

## Consequences

- The reference remains the oracle for everything it gets right, and stops being one where it is known to be wrong.
  ADR 0031's bottleneck exemption is unchanged: the two reasons are independent, and a patch records which one
  opened it.
- A reference bump has to re-apply correctness patches too. A bump that fixes the bug upstream retires the patch:
  the diff no longer applies, and the behaviour returns to reference parity with the test still green.
- Where an Ignis-patched behaviour is on the measured path, the reference's numbers are no longer the like-for-like
  baseline for it; the test that tells the two apart is how a reader reproduces either side.
