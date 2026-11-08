---
status: accepted
amends: [ADR-0005, ADR-0007, ADR-0043]
supersedes: [ADR-0014, ADR-0015, ADR-0021]
---

# The gate is KLD against BF16, DFlash2 on against off and the loop rate

ADR 0007 made acceptance a ladder of performance gates against ninfer, the engine whose kernels ignis started from,
and ADRs 0014, 0015 and 0021 gave its rungs their method: G1, teacher-forced agreement with ninfer's recorded greedy
tokens; G2 and the gates after it, ratios against ninfer measured live/live in one session, over at least two launches
per engine. ADR 0043 took Flash-Next off that ladder, since ninfer cannot run it, and said both rules "stand unchanged"
for the 27B. The ladder is no longer run. G1-G5 have no make target, the last gate legs against ninfer ran on
2026-09-13 and 2026-09-14 (G4, G5), and when G3 ran again on 2026-10-31 it ran with ignis as both `--ours` and `--ref`
(#426), a check of the instrument and not a verdict. ninfer no longer runs on the development machine (ADR 0084).
Every artifact added since 2026-10-21 was admitted against its BF16 checkpoint instead: the 16 GB body (#340), the
abliterated 16 GB build (#422), Swift 1.5 in both layouts (#449), the sm_8x prefill routes (#503). No ADR recorded that
gate. On 2026-11-08 the owner made it the official one.

**The gate has three legs**, measured on ignis alone, in the load the artifact serves:
- **KLD against BF16.** The engine's `span_logits` dump of set A (16 windows of 4,096 tokens: 6 code, 4 prose, 6 chat)
  at the serving configuration (hq-e8-2b KV, the served chunk), scored per position against the model's own BF16
  checkpoint run layer by layer: KL(BF16 ‖ engine) over the full vocabulary, the mean over every position, with top-1
  and a window-bootstrap 95% CI beside it (`tools/kld-gate`, `ref_kld.py` and `gate1.py`). A fine-tune is scored
  against its own BF16, with `gate1.py`'s two cross terms, which show that the gate measured the fine-tune and not the
  base. Where the card cannot run the checkpoint, the reference is the BF16 model's 64 most probable tokens per
  position, built once on the RTX 5090 (`ref_top64.py`, `score_top64.py`). That figure is a lower bound of the full
  KLD, so its limit comes from a top-64 figure of the RTX 5090, never from a full-vocabulary one.
- **DFlash2 on against off.** With the drafter on, decode tok/s beats off in each of the nine decode cells #340
  measured: 8,192, 65,536 and 253,440 tokens of context, at one lane cold, two lanes, and one lane with reuse (#449).
  Reported beside it, not binding: #267's teacher-forced replay of 200 recorded agent turns, tokens per round against
  the counterpart's, with the 0.90 floor that #340 and #422 held as binding.
- **Loop rate.** #340's loop set (four prompts, seeds 101 to 108: 32 requests) on the artifact and on its counterpart,
  one build, the same prompts and seeds. A request loops if the guard rewound it, if its reasoning holds a 6-gram at
  least 25 times, or if its worst 3,000-character zlib window compresses below 0.20. The leg fails if the artifact
  loops more, at a one-sided Fisher exact p below 0.05 on the pooled counts. It is the one leg that runs with the loop
  guard at its default, since a rewind is one of its calls.

**The limits are the ones the repo has registered**, and the point estimate decides against them:
- **A new artifact of a served family** (a new body or format, a re-quantization, a fine-tune): a KLD at most 1.2×
  its counterpart's, the ratio the owner set on 2026-10-21 (spec `27b-16gb/05`). The counterpart is the served
  artifact a new body stands in for (#340), or, for a re-quantization or a fine-tune, the family's artifact of the
  same layout (#422, #449). Once registered the limit is a number: 1.2 × 0.04916 = 0.0590 for the NVFP4 layout at
  chunk 2,048, 1.2 × 0.04034 = 0.0484 for the 16 GB body at chunk 1,024. All three legs run.
- **A change to a served route's numerics** (a kernel, or a route made a default): the KLD of each entry the route
  serves at most 1.05× that of the route it replaces, with top-1 within 0.5 points (#503, #513). The DFlash2 and loop
  legs run when its spec asks for them; #503's prefill routes ran the KLD leg alone.
- **The release's sm_86 gate**: a top-64 KLD at most 1.05× the RTX 5090's on the same route and dump, and top-1 at
  most 0.5 points under it (`docs/agents/testing.md`, "The sm_86 gate").

A change with no registered limit gets one in its spec before the run, approved by the owner. A failing leg stops the
work, and the owner decides (#449). What a spec measures besides, MMLU-Pro as #340 did or a refusal A/B as #422 did,
is that spec's, not the gate. Flash-Next keeps ADR 0043's gate against its checkpoint, of the same kind. The speech
and image models keep the checks `docs/agents/testing/speech.md` and `docs/agents/testing/image-model.md` name.

**No gate is a ratio against ninfer.** A speed claim about a change compares ignis with ignis, a branch against main
in one session, over at least two independent launches per build: one process launch picks its own throughput band,
which is ADR 0021's finding (#116) and stays the practice (`docs/agents/testing/bench.md`). A spec that needs a speed
floor states its cells and its floor, as ADR 0043 did for Flash-Next, and the owner approves it.

**What is retired and what stays.**
- G1-G5, ADR 0007's 99% performance gate and its ladder, ADR 0015's live/live method and ADR 0021's launch rule as a
  rule of it decide nothing. Their verdicts, and ninfer's numbers in the findings and the September plans, stay as a
  historical reference, read with their date.
- ADR 0007's other half stands: correctness is self-checked, not reference-matched, and the canary check
  (`ignis-bench canary`, sane and deterministic output) is that check.
- The GPU profile's correctness tests are error detectors, not the gate. The teacher-forced canary test keeps ADR
  0014's scoring (at least 95%, no tie waived) against its fixture, recorded on ninfer on 2026-09-05 and never since,
  as Flash-Next's G1 does against its quantized reference (ADR 0043). The f64 oracles, the leaf's op tests and the
  recorded fixtures catch a broken forward pass; a change of the numerics is judged for quality by the KLD leg.
- `ignis-bench ttft`, `g3`, `g4` and `g5` and their gate checks stay instruments that measure ignis. A record of
  ninfer is no longer the other side of a verdict.

## Consequences

- The sets the gate reads are not in the repo: set A's windows, #267's recorded turns and the loop set are in the
  main checkout's `.scratch/`, and the release's coordinator carries the top-64 reference set to the sm_86 card. The
  gate runs where they are.
- ADR 0060's four gates, MMLU-Pro and the replay's 0.90 floor among them, admitted the 16 GB body once, and stay the
  record of that admission.
