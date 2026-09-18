# ADR 0031 — a vendored kernel that is measured as the bottleneck is not untouchable

## Status

Accepted (2026-09-18, owner). **Clarifies ADR 0010** — it exercises 0010's own
last Decision bullet ("our own kernels come after G4, one op family at a time,
each justified by a measurement"), per op rather than per family, and says what
that measurement has to show before anyone may touch a vendored file. It is not
a reversal: every provenance rule in 0010 stands unchanged, and the default is
still that vendored files are not hand-edited.

Sources: owner statement 2026-09-18 ("i kernel sono intoccabili ma se troviamo
delle ottimizzazioni per cui ci rendiamo conto che i kernel vendor sono *il
rallentamento*, possiamo tranquillamente patcharli oppure reimplementarci quella
funzionalità e togliere il kernel vendor");
`docs/findings/2026-09-18-dflash2-topk-one-warp-per-column.md` and
`docs/findings/2026-09-18-decode-round-host-idle.md`, the decode-round profile
the kernel breakdown below comes from (both landed on branch
`kernel-gqa-workspace-memset`, not yet merged);
`.scratch/kernel-opt-2026-09-18/candidates.md`.

## Context

ADR 0010 is being read as "vendored kernels are untouchable, full stop." That
reading was correct for what 0010 was fixing. What it replaced were hand-written
scalar stand-ins that were both ~10–100× slower than the reference's equivalents
*and* numerically wrong — the storage-layout decoders went with the kernels, so
the scale plane and the weight divisor were the first casualties, under headers
claiming a 1:1 port. 0010's answer was that a port claim must be **diffable**:
"we ported the reference's kernel" means this exact file *is* the reference's
file, and a script proves it. "Do not hand-edit" is not reverence for the
reference's code — it is the only thing that keeps the manifest meaning
anything.

But 0010 never said *forever*. Its last two Decision bullets are the escape
hatch: "**Our own kernels come after G4**, one op family at a time, each
justified by a measurement and re-gated by the performance gate," and "Families
already at roofline in the reference are not candidates." The timing
precondition is already met — G4 closed (GitHub #118, PR #133). Nothing in 0010
has to change for a vendored kernel to be replaced; what was missing is a
statement of when the hatch opens, so that it opens on evidence and not on
taste.

The trigger is concrete. In the 2026-09-18 Nsight Systems profile of a decode
round, `dflash2_topk_kernel` (`kernel/vendor/src/ops/kernel/dflash2_draft.cuh:150`)
costs **3,145 µs per decode round — 16.4% of all decode kernel time**. Its
launcher (`ops/launcher/dflash2_draft.cu`) fixes `warps_per_block = 4`, so a
7-column round launches **grid 2×1×1, block 128**: eight warps, seven of them
doing work, on a card with 170 SMs. One warp owns one column. The work itself is
a top-k=16 over a `[248046 vocab, 7 column]` BF16 logits matrix — **3.47 MB
read**, which at this card's bandwidth is about **2.0 µs** of memory traffic.
The kernel is roughly **1,570× off its own memory bound**. It is not slow
because top-k is hard; it is slow because one warp per column leaves 168 of the
170 SMs idle, with each lane's `TopkEntry list[64]` in local memory at 20
registers per thread. The launcher is the reference's, unchanged, so the
reference pays the same 3 ms — which is exactly the kind of ceiling ADR 0005
refuses to accept.

The replacement is measured, not projected. A row-split merge of ours
(`kernel/src/dflash2_topk.cu`, grid 122×7 plus 7×1 at 71 registers, against the
vendored grid 2×1 at 20) costs **44.26 µs per round against the vendored
3,145.23 µs — 71×**, and takes the decode round at one lane from **19.18 ms to
15.81 ms (−17.6%)**: 388 rounds per 8 s become 471, and because the selection is
bit-identical the acceptance is unchanged, so **+21.4% rounds is +21.4% tokens
per second**. At *eight* lanes the same change is worth only **33.54 → 32.18 ms
(−4.1%)** — and the reason is the fact that made it large at one lane: the
vendored kernel's parallelism *is* its column count, and eight lanes hand it 56
columns instead of 7. How big a win is available is a property of the geometry
the engine actually runs, which is one more thing that cannot be read off the
source. Correctness held at exactly the bound the Decision below sets: 18 shapes
bit-identical against the vendored op as oracle
(`kernel/tests/test_dflash2_topk.cu`), and 256 greedy tokens byte-identical end
to end.

The same profile contains the counter-example, and it is the more important
half. `w8_small_t_mma_kernel` measures **820 µs against a 747 µs roofline** for
its shape — 91% of the bound. It is vendored, it is hot, and there is nothing to
win on it. Someone reading the two kernels side by side would not reliably tell
them apart: both are dense CUDA written by people who knew what they were doing.
The difference between "leave it alone" and "this is the slowdown" is not a
property of how the code looks. **It is a number.** That is why the exemption
below is written around a measurement and not around a judgement of kernel
quality — and why a code-reading pass like
`.scratch/kernel-opt-2026-09-18/candidates.md` (explicitly "nessuna misura su
GPU") is how candidates get *chosen for profiling*, never how they get changed.
The same profile retired one of that pass's hypotheses for free: the 224
memcpy/memset nodes a decode round enqueues between PDL-chained vendored ops
were suspected of forfeiting each following kernel's prologue overlap, and
node-level tracing put the gap after a residual copy at **0.10 µs at p50**,
indistinguishable from a kernel-to-kernel edge. Reading the code made that
argument look strong; one capture ended it — the thesis cutting the other way.

The argument for opening the hatch at all comes from ADR 0005. The reference is
"a reference for inspiration only," not a target and not a ceiling; the
north-star is "the best local coding engine." A rule that forbids touching a
vendored kernel *even when that kernel is the thing making us slow* makes the
reference's performance an upper bound in precisely the places where we are
losing — the places the north-star cares about most. A blanket ban and ADR 0005
cannot both hold. ADR 0005 wins.

## Decision

**Vendored kernels stay untouchable by default. A measurement that identifies a
specific vendored kernel as the bottleneck lifts that default for that kernel.**

- **The trigger is a measurement, never a suspicion.** Two parts, both required:
  a profile taken on the exclusive card (ADR 0006) shows the op as a top
  contributor to a real prefill or decode round at real geometry, **and** a bound
  estimate for that op's own shape — memory traffic at sustained HBM bandwidth,
  or the FLOP roofline — shows headroom against the measured time. Share alone
  is not enough: `w8_small_t_mma` is a top contributor in the same profile and is
  at 91% of its bound, so it is not a candidate. "This looks scalar," "this could
  be vectorized," and a reading of the source are how a candidate earns a profile
  run, not how it earns an edit.
- **Ops at or near their bound are not candidates**, restating 0010 with a
  threshold rather than a feeling: if the profile is within a small factor of the
  roofline, the win is not in the kernel and the exemption does not open.
- **Two options, both allowed, chosen per op:**
  - **(a) A recorded patch.** The file stays vendored and keeps its port claim.
    The manifest entry gains a `patch` whose `reason` names the measurement that
    opened it. Cost, stated plainly: the diff must re-apply on every
    `scripts/vendor-ninfer.ps1 sync` at a reference bump, and a kernel-body diff
    rots far faster than the three test-trim diffs under
    `kernel/vendor/patches/tests/` that are the mechanism's only users today.
    0031 is the first time a patch would touch a kernel.
  - **(b) Our own implementation.** It lives in `kernel/src/` alongside the
    program layer, the call site moves to it, and the vendored file leaves the
    manifest (or stays vendored and unused where other routes still call it).
    The precedent exists: `kernel/src/dflash2_drafter.cu` and
    `kernel/src/linear.cu` are already ours and already carry no provenance
    claim.
  - Prefer **(a)** when the fix is local to the vendored file — a launch
    geometry, a bad constant, a missing vectorized path — and **(b)** when the
    right answer is a different algorithm or a different decomposition, i.e.
    when the patch would no longer be recognizable as the reference's kernel.
- **Provenance obligations survive the exemption, all of them.** The manifest
  stays the source of truth for what is vendored; no vendored file is edited
  outside a recorded patch; `kernel/NOTICE` attribution stays; and **anything
  hand-written carries no port claim** — a replacement is ours, labelled ours,
  and never described as a port. `kernel/vendor/VENDOR.md` records what left the
  subtree and why.
- **A replacement must keep the vendored op's numerical contract, and the
  vendored op is the oracle.** Where the op is an exact, deterministic selection,
  the obligation is bitwise-identical output on the same input, and the test runs
  both implementations and compares. `dflash2_topk` is exactly that case: its
  ordering is value descending, ties broken by the **lower row index**
  (`topk_less`), and BF16 logits over 248K rows tie constantly, so a replacement
  that gets the tie rule wrong produces different ids on real inputs while
  passing any tolerance-based check. Where the op is not exact — accumulation
  order in a GEMM — the oracle is the reference's own op test with its fp64
  references and tolerances, which 0010 already requires to stay green.
- **Every patch and every replacement has to show its win, per op:** the profile
  that opened the exemption is re-taken on the exclusive card and must show the
  time actually moved, alongside the oracle test above. A change made for speed
  that does not show the speed is reverted, not kept because the code reads
  better. The 99% live/live performance gate (ADR 0005 / 0007) is unchanged and
  stays where the standing convention puts it — once at phase end, over the
  phase's changes together — it is not re-run per kernel.
- **The correctness floor is not traded.** ADR 0005's floor comes first: no
  kernel change is accepted on a speed number alone if the engine's output stops
  being sane.

## Consequences

- ADR 0010 is clarified, not reversed. Every one of its Decision bullets stands
  word for word; this ADR only says when its own "one op family at a time,
  justified by a measurement" clause fires, and scopes it to one op.
- The manifest's patch mechanism gets its first kernel-body user. A reference
  bump can now fail as a conflict inside a kernel rather than inside a trimmed
  test file. That cost is accepted per op, at the moment the patch is recorded.
- Every op that leaves the vendored subtree loses the mechanical reference
  update for that op: a future reference bump no longer brings its fixes for
  free, and the op is ours to maintain. Also accepted per op, recorded in
  `VENDOR.md`.
- The vendored subtree is expected to **shrink slowly, in measured steps**. A
  smaller subtree is a sign of the roadmap working, not of drift — drift is what
  the manifest exists to detect, and the manifest is unaffected.
- More of the leaf becomes ours, so the "no port claim for anything hand-written"
  rule carries more weight than it did under 0010, not less. The glossary entry
  for **vendored op** (`CONTEXT.md`) is the definition that decides.
- The review question for a patched or replaced kernel is answerable from the
  repo: *which measurement opened this?* — from the manifest entry's `reason`,
  or from the ticket the replacement landed under.
- First op replaced under this ADR: `dflash2_topk` (branch
  `kernel-gqa-workspace-memset`), with its own profile, its own oracle test and
  its own findings write-up — the work, not this decision. Note how option (b)
  actually landed: the vendored file **stays** in the subtree and under the
  manifest, because it is the oracle the replacement is tested against and still
  the implementation for every `k` the replacement does not specialize. Leaving
  the call site is not the same as leaving the manifest.
