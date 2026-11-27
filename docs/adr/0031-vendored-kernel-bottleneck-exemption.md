---
status: accepted
---

# A vendored kernel that is measured as the bottleneck is not untouchable

Vendored kernels stay untouchable by default, and a measurement that identifies a specific vendored kernel as the
bottleneck lifts that default for that kernel: it may then carry a recorded patch, or give way to our own
implementation. "Do not hand-edit" (ADR 0010) is not reverence for the reference's code, it is what keeps the
manifest meaning anything; but a ban that holds even when the vendored kernel is what makes ignis slow would make
the reference's performance a ceiling exactly where ignis is losing, and ADR 0005 holds the reference to be an
inspiration, not a ceiling. Whether a kernel is the slowdown is a number, not a property of how its code reads: in
one decode-round profile the vendored DFlash2 top-k sat some 1,570 times off its own memory bound while another hot
vendored kernel sat at 91% of its roofline
([finding](../findings/2026-09-18-dflash2-topk-one-warp-per-column.md)), and a suspicion that reading the code made
strong was ended by one capture ([finding](../findings/2026-09-18-decode-round-host-idle.md)).

- **The trigger is a measurement, never a suspicion.** Two parts, both required: a profile taken on the exclusive
  card (ADR 0006) shows the op as a top contributor to a real prefill or decode round at real geometry, and a bound
  estimate for that op's own shape, memory traffic at sustained HBM bandwidth or the FLOP roofline, shows headroom
  against the measured time. Share alone is not enough. "This looks scalar", "this could be vectorized" and a
  reading of the source are how a candidate earns a profile run, never how it earns an edit. How large a win is
  available depends on the geometry the engine runs, which cannot be read off the source either.
- **An op at or near its bound is not a candidate**: within a small factor of the roofline the win is not in the
  kernel, and the exemption does not open.
- **Two options, chosen per op:**
  - **(a) A recorded patch.** The file stays vendored and keeps its port claim; the manifest entry's `patch` has a
    `reason` that names the measurement that opened it. Its cost: the diff must re-apply at every reference bump,
    and a kernel-body diff rots far faster than a trimmed test.
  - **(b) Our own implementation.** It lives in `kernel/src/` beside the program layer and the call site moves to
    it. The vendored file leaves the manifest, or stays vendored where other routes still call it or where it is the
    oracle: leaving the call site is not leaving the manifest.
  - Prefer (a) when the fix is local to the vendored file (a launch geometry, a bad constant, a missing vectorized
    path) and (b) when the right answer is another algorithm or decomposition, that is, when the patch would no
    longer be recognizable as the reference's kernel.
  - One kernel may take both, a patch and later a route of ours beside it, and the replacement may be an adapted op
    (ADR 0051) that takes parts of the vendored kernel it stands in for. The vendored file is then not edited
    further and stays in the subtree with its patch, as the oracle and as the route of every card and call that
    does not take ours (ADR 0093).
- **Every provenance obligation survives.** The manifest stays the source of truth for what is vendored; no vendored
  file is edited outside a recorded patch; the `kernel/NOTICE` attribution stays; anything hand-written carries no
  port claim, so a replacement is ours, labelled ours and never described as a port; `kernel/vendor/VENDOR.md`
  records what left the subtree and why.
- **A replacement keeps the vendored op's numerical contract, and the vendored op is the oracle.** Where the op is
  an exact, deterministic selection, the output is bitwise identical on the same input and the test runs both
  implementations and compares: a top-k whose ties break by the lower row index ties constantly on real logits, and
  a replacement with the wrong tie rule would pass any tolerance-based check. Where the op is not exact, as with the
  accumulation order of a GEMM, the oracle is the reference's own op test with its fp64 references and tolerances,
  which ADR 0010 already requires to stay green.
- **Every patch and every replacement shows its win, per op.** The profile that opened the exemption is taken again
  on the exclusive card and must show that the time moved, alongside the oracle test. A change made for speed that
  does not show the speed is reverted, not kept because the code reads better. The gate, ADR 0082's where the 99%
  performance gate against the reference once stood, is not run again per kernel: it stays once at phase end, over
  the phase's changes together. A replacement held to its oracle's bytes re-runs no quality gate (ADR 0093).
- **The correctness floor is not traded.** ADR 0005's floor comes first: no kernel change is accepted on a speed
  number alone if the engine's output stops being sane.

## Consequences

- Every rule of ADR 0010 stands. This ADR says when its "justified by a measurement" clause fires, and scopes it to
  one op.
- A reference bump can fail as a conflict inside a kernel. That cost is accepted per op, when the patch is recorded.
- An op that leaves the vendored subtree loses the mechanical reference update: a later bump no longer brings its
  fixes, and the op is ours to maintain. Also accepted per op, and recorded in `VENDOR.md`.
- The vendored subtree is expected to shrink slowly, in measured steps. A smaller subtree is not drift: drift is
  what the manifest detects, and the manifest is unaffected.
- More of the leaf becomes ours, so "no port claim for anything hand-written" carries more weight. The glossary's
  entry for **vendored op** is the definition that decides.
- The review question for a patched or replaced kernel is answered from the repo: which measurement opened this?
  From the manifest entry's `reason`, or from the ticket the replacement landed under.
