---
status: accepted
---

# The gate is KLD against BF16, DFlash2 on against off and the loop rate

An artifact, or a change to a served route's numerics, is admitted by three legs measured on ignis alone, in the load
the artifact serves: its KLD against the model's own BF16 checkpoint, DFlash2 on against off, and its loop rate
against its counterpart's. No gate is a ratio against ninfer. The ladder of performance gates ADR 0007 set against
ninfer is retired, with the method its rungs had: G1, teacher-forced agreement with ninfer's recorded greedy tokens;
G2 and the gates after it, ratios against ninfer measured live/live in one session, over at least two launches per
engine. It was no longer run: its last legs against ninfer ran in September 2026, ninfer no longer runs on the
development machine (ADR 0084), and every artifact added since 2026-10-21 was admitted against its BF16 checkpoint
instead.

**What passes each leg** (how each is run: `docs/agents/testing/gate.md`):
- **KLD against BF16.** The mean KL(BF16 ‖ engine) over the full vocabulary, at the serving configuration, against the
  model's own BF16 checkpoint, is within the registered limit. A fine-tune is scored against its own BF16, with the
  two cross terms that show that the gate measured the fine-tune and not the base: each holds when the 95% CI of its
  difference from the matched pair is above 0, its count of windows above reported, not required. Where the card
  cannot run the checkpoint, the reference is the BF16 model's 64 most probable tokens per position, built on the
  RTX 5090: that figure is a lower bound of the full KLD, so its limit comes from a top-64 figure of the RTX 5090,
  never from a full-vocabulary one.
- **DFlash2 on against off.** With the drafter on, decode tok/s beats off in each of nine decode cells: 8,192, 65,536
  and 253,440 tokens of context, at one lane cold, two lanes, and one lane with reuse. Reported beside it, not
  binding: #267's teacher-forced replay of recorded agent turns, tokens per round against the counterpart's, with the
  0.90 floor that earlier admissions held as binding.
- **Loop rate.** On the loop set, one build, the same prompts and seeds, the artifact does not loop more than its
  counterpart at a one-sided Fisher exact p below 0.05. It is the one leg that runs with the loop guard at its
  default, since a rewind is one of its calls.

**The limits are the ones the repo has registered**, and the point estimate decides against them:
- **A new artifact of a served family** (a new body or format, a re-quantization, a fine-tune): a KLD at most 1.2×
  its counterpart's, the ratio the owner set. The counterpart is the served artifact a new body stands in for, or,
  for a re-quantization or a fine-tune, the family's artifact of the same layout. Once registered the limit is a
  number: 1.2 × 0.04916 = 0.0590 for the NVFP4 layout at chunk 2,048, 1.2 × 0.04034 = 0.0484 for the 16 GB body at
  chunk 1,024. All three legs run.
- **A change to a served route's numerics** (a kernel, or a route made a default): the KLD of each entry the route
  serves at most 1.05× that of the route it replaces, with top-1 within 0.5 points. The DFlash2 and loop legs run
  when its spec asks for them.
- **The release's sm_86 gate**: a top-64 KLD at most 1.05× the RTX 5090's on the same route and dump, and top-1 at
  most 0.5 points under it (`docs/agents/testing.md`, "The sm_86 gate").

A change with no registered limit gets one in its spec before the run, approved by the owner. A failing leg stops the
work, and the owner decides. What a spec measures besides, MMLU-Pro or a refusal A/B, is that spec's, not the gate.
Flash-Next keeps ADR 0043's gate against its checkpoint, of the same kind. The speech and image models keep the checks
`docs/agents/testing/speech.md` and `docs/agents/testing/image-model.md` name.

**A speed claim about a change compares ignis with ignis**, a branch against main in one session, over at least two
independent launches per build, the server stopped and started again between them: one process launch picks its own
throughput band, which repetition inside a launch cannot cancel
([finding](../findings/2026-09-10-one-process-launch-picks-its-own-throughput-band.md),
`docs/agents/testing/bench.md`). A spec that needs a speed floor states its cells and its floor, as ADR 0043 did for
Flash-Next, and the owner approves it.

## Consequences

- G1-G5 and ADR 0007's 99% performance gate decide nothing. Their verdicts, and ninfer's numbers in the findings and
  the September plans, stay as a historical reference, read with their date.
- ADR 0007's other half stands: correctness is self-checked, not reference-matched, and the canary check
  (`ignis-bench canary`, sane and deterministic output) is that check.
- The GPU profile's correctness tests are error detectors, not the gate. The teacher-forced canary test keeps G1's
  scoring against its fixture, recorded on ninfer on 2026-09-05 and never since: at least 95% of the suite's
  positions agree, each scored on the fixture's own prefix, with no mismatch waived (an exact logit tie included, no
  epsilon, no position or canary left out) and no lower threshold, and ignis's argmax is not changed to make a tie
  agree (`crates/bench/src/oracle.rs`); a free-running comparison is a diagnostic only. Flash-Next's G1 scores the same way against its quantized reference (ADR 0043). The f64 oracles,
  the leaf's op tests and the recorded fixtures catch a broken forward pass; a change of the numerics is judged for
  quality by the KLD leg.
- `ignis-bench ttft`, `g3`, `g4` and `g5` and their gate checks stay instruments that measure ignis. A record of
  ninfer is no longer the other side of a verdict.
- The sets the gate reads are not in the repo: set A's windows, #267's recorded turns and the loop set are in the
  main checkout's `.scratch/`, and the release's coordinator carries the top-64 reference set to the sm_86 card. The
  gate runs where they are.
- ADR 0060's four gates, MMLU-Pro and the replay's 0.90 floor among them, admitted the 16 GB body once, and stay the
  record of that admission.
