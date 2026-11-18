# The two `/v1/decide` regression sets under both exception weights: FP8 moves them as much as a harmless switch does, and over fourteen named arms only two ignis-written "yes" cases stay gates

- Kind: experiment
- Status: current
- Observed: 2026-11-18
- Last verified: 2026-11-18
- Scope: tests / the `/v1/decide` regression sets (`decide_entailment_gpu`, `decide_scalar_gpu`, the entailment fixture's arms and tiers, `support/decide_arms.rs`); serving / the 27B's exception weights, rope table, GDN prefill route, prompt attention band and prompt reuse on the RTX 5090
- Topic: decide-options
- Related: [GitHub #675](https://github.com/gpillon/ignis/issues/675), [spec #672](https://github.com/gpillon/ignis/issues/672), [GitHub #667](https://github.com/gpillon/ignis/issues/667) (the fixture study), [GitHub #537](https://github.com/gpillon/ignis/issues/537), [GitHub #670](https://github.com/gpillon/ignis/issues/670), [GitHub #671](https://github.com/gpillon/ignis/issues/671), [ADR 0065](../adr/0065-the-bf16-exception-tensors-can-be-held-in-fp8.md), [the BF16 exception tensors in FP8](2026-10-21-the-bf16-exception-tensors-in-fp8.md), [Blackwell's GDN prefill on ignis's kernels](2026-11-08-blackwell-gdn-prefill-on-ignis-kernels.md) (the 2026-11-08 tiers)
- Superseded by: none
- Summary: RTX 5090, 27B: FP8 exceptions move the 20-case entailment set by an RMS of 1.68-1.89 logits, inside a harmless switch's 1.42-2.10; over 14 named arms no contributed "yes" case stays a gate, two ignis-written ones do.

## Question

FP8 exception weights stayed opt-in (ADR 0065) because two `/v1/decide`
regression sets read differently under them: the entailment set's case 1 and
the scalar set's negative balance. Spec #672 makes FP8 the default on
compute capability 12.0. Two questions come first:

- Does FP8 move these sets more than a harmless change of the route does?
- Which cases sit clear of the 0.5 line on every route the RTX 5090 runs,
  under both exception weights, so that a red test means a broken feature
  and not a numerics change?

## Evidence

**The fixture study (GitHub #667, 2026-11-17, RTX 5090).** Its figures are in
the issue's comment; its harness and synthetic items are not in the
repository.

- Seventeen deterministic routes, BF16 and FP8. The routes that move a read
  are the rope table, the GDN prefill route, int8 prompt attention and
  prompt reuse off. Spec off, the fp16 attention band, the context at an
  equal rope table, the MTP load and chunk 1,024 against 2,048 are bit
  identical on these prompts.
- Three of the entailment set's five arms registered on 2026-11-08 (the
  live-server ones) ran FP8 exception weights from a user config file.
- On 116 synthetic entailment items, BF16 is right on 106-109 and FP8 on
  105-111; 17 items do better under FP8 and 13 under BF16 (sign test p
  0.59); FP8's margin shift is +0.16 [+0.04, +0.27] logit.
- The scalar balance's sign is a coin: BF16 reads it right on 2 of 9 routes
  and wrong at its served shape. On literal negatives both encodings read
  11-28 of 48 right.

**This measurement (2026-11-18, main ab2adc85, RTX 5090, PCIe Gen 3,
Windows).** The entailment set's arms are re-registered and measured anew
with the GPU test itself, one load per arm:
`IGNIS_DECIDE_ARMS=all IGNIS_DECIDE_DRAFTS=1 cargo test -p ignis-server --features cuda --test decide_entailment_gpu -- --ignored --exact the_gate_cases_keep_their_entailment_labels --nocapture --test-threads=1`.

- **The arms.** Fourteen, listed with every value in the fixture's
  `NOTICE.md`: the harness under BF16 and FP8, each with ignis's and the
  vendored GDN prefill (rope none, 40,960 tokens, chunk 1,024); and the
  `rtx5090` profile's start under each exception weights at its served shape
  (BF16 786,432 tokens and yarn:3, FP8 917,504 and yarn:3.5, as spec #672
  measured them; chunk 2,048,
  DFlash2 at 7, vision on), at the other's rope table, with the vendored GDN
  prefill, with the int8 band and with prompt reuse off. Every arm names its
  exception weights, context, rope, GDN prefill route, prompt attention and
  prompt reuse, and reads no config file: a served arm's flags are resolved
  with no file and no environment and loaded through the server's own load
  path. A served load that adapted anything would fail the run; none did.
- **The cost.** 264 s for all fourteen arms, the drafts included (about 10 s
  a harness load, 25 s a served one).
- **Reproduction.** The 2026-11-08 harness arms read as `h-bf16` and
  `h-bf16-vendored`, and its live-server arms with ignis's and the vendored
  GDN prefill as `s-fp8-yarn3.5` and `s-fp8-vendored`, all 18 cases bit for
  bit. The served arms in process are the live server, chunk 1,024 against
  2,048 and spec off against DFlash2 change nothing here, and those two
  registered arms were FP8.
- **Repeats and order.** Two passes read the cases and the 24 candidate
  wordings on every arm, and all 588 reads repeated bit for bit. Between the
  passes four wordings moved between the case slots 19 and 20 and the drafts
  and two drafts took a new predecessor: those 84 reads, 70 of them after
  another prompt than the first time, did not move either. Cases 1-18 kept their
  order, so this shows no more than that.
- **Context.** FP8 at yarn:3.5 on 786,432 tokens reads all 20 cases as it
  does on 917,504, bit for bit: at an equal rope table the context moves
  nothing here, so the served pairs below, whose contexts differ, differ in
  their exception weights alone.
- **BF16's served shape.** Started under `rtx5090` with nothing but its
  exception weights named and no config file, BF16 keeps the profile's
  917,504 tokens (so yarn:3.5) under the profile's 1,200 MB VRAM headroom on
  this desktop (about 3 GB of the card held), and adapts to 786,432 (yarn:3)
  under a 1,500 MB headroom. FP8 keeps 917,504 under both. So BF16's rope on
  the RTX 5090 is yarn:3 or yarn:3.5 by a few hundred MB of headroom, and FP8
  as the default moves it from yarn:3 only where BF16 adapted. Both BF16
  tables are arms (`s-bf16-yarn3`, `s-bf16-yarn3.5`).

The spread, as the RMS of the change in a case's signed margin between two
arms over the 20 cases (in brackets: over the 20 cases and the 22 drafts),
with the cases that cross the line:

| pair | RMS, 20 cases | (42 items) | largest move | crossings |
|------|--------------:|-----------:|-------------:|-----------|
| FP8 against BF16, harness | 1.68 | (2.08) | 3.500 | 12 lost, 17 gained |
| FP8 against BF16, harness, vendored GDN | 1.89 | (2.21) | 5.250 | 1 lost |
| FP8 against BF16, served, yarn:3 | 1.83 | (1.88) | 4.000 | 3 gained |
| FP8 against BF16, served, yarn:3.5 | 1.78 | (1.85) | 3.750 | 16, 18 gained |
| BF16, vendored GDN (harness) | 1.46 | (1.91) | 4.000 | 17 gained |
| BF16, rope none to yarn:3 | 1.77 | (2.39) | 3.375 | 3, 16 lost |
| BF16, rope yarn:3 to yarn:3.5 | 1.98 | (1.82) | 5.125 | 3 gained, 18 lost |
| BF16, vendored GDN (served) | 1.88 | (2.07) | 3.250 | 3, 16 gained |
| BF16, int8 band | 2.10 | (2.02) | 3.875 | 12 lost; 3, 16, 17 gained |
| BF16, prompt reuse off | 1.73 | (1.80) | 3.625 | 3, 17 gained |
| FP8, vendored GDN (harness) | 1.69 | (1.68) | 4.750 | 1 lost, 12 gained |
| FP8, rope none to yarn:3.5 | 2.07 | (2.01) | 4.000 | 17 lost, 12 gained |
| FP8, rope yarn:3.5 to yarn:3 | 2.01 | (2.05) | 4.750 | 16 lost |
| FP8, vendored GDN (served) | 1.62 | (1.56) | 4.000 | 17 gained |
| FP8, int8 band | 1.74 | (1.55) | 4.625 | 1, 18 lost; 17 gained |
| FP8, prompt reuse off | 1.42 | (1.55) | 4.000 | 12 lost |

Every crossing is a watch case's. Over the 20 cases, BF16 is right on 16-19
per arm and FP8 on 17-19.

The two "yes" cases that were gates on 2026-11-08, by their smallest logit:

| case | BF16, 7 arms | FP8, 7 arms |
|------|-------------:|------------:|
| 1 | +0.625 (`s-bf16-yarn3.5`) | −0.750 (`h-fp8-vendored`) |
| 18 | −0.250 (`s-bf16-yarn3.5`) | 0.000 (`s-fp8-int8`) |

**The ignis-written cases.** 24 wordings, each a restatement of one field
over the set's public names, were read on every arm in one pass. Six clear
the gate margin of +1 logit on all fourteen; the best two became cases 19
("The executive sponsor is engaged on this account.", smallest +1.625) and
20 ("The renewal date is 2027-03-31.", smallest +1.500). Every wording and
its reads are in the fixture (`drafts`, `NOTICE.md`). The same wording
moves by up to 5.25 logits between two arms: case 19 reads +1.625 to
+6.875.

**The tiers, before and after.**

| | 2026-11-08 (5 arms) | 2026-11-18 (14 arms) |
|---|---|---|
| gate, "yes" | 1, 18 | 19, 20 |
| gate, "no" | 2, 4, 5, 6, 7, 8, 9, 11, 14, 15 | the same ten (smallest margin +2.000, case 8) |
| watch | 3, 10, 12, 16, 17 | 1, 3, 10, 12, 16, 17, 18 |
| known failure | 13 | 13 |

**The scalar set.** Its test, run on the same fourteen arms (161 s): the
five other cases read right on every arm. The balance (truth −0.75) reads
−0.75 on 3 of BF16's 7 arms (`h-bf16`, `h-bf16-vendored`, `s-bf16-yarn3.5`)
and 0.75 on the other four, BF16's served shape among them. FP8 reads 0.75
on all 7: on the three routes where BF16 has the sign FP8 loses it, and on
none does it gain it. The `uncertainty` it reports, 0.0000 to 0.3804, counts
the digits alone (#670).

## Finding

Observed:

- FP8's moves on the entailment set, RMS 1.68-1.89 logits over the 20
  cases, are inside the moves one harmless switch makes under either
  exception weights, 1.42-2.10; no crossing between FP8 and BF16 is a gate
  case's.
- Over the fourteen arms no contributed "yes" case is a gate, and none would
  be over BF16's seven alone: case 1 reaches only +0.625 and case 18 falls
  to −0.250, both at BF16's yarn:3.5. The ten "no" gates hold on every arm
  by at least +2.000.
- Two ignis-written restatements are gates on every arm, by +1.625 and
  +1.500; four other wordings of the 24 also clear +1.
- Each arm's reads repeat bit for bit, and moving a wording to another
  place in the order did not move its read.
- On the scalar set's balance FP8 loses the sign on 3 of the 4 routes both
  exception weights share (the harness with each GDN prefill, the served
  start at yarn:3.5) and gains it on none; the fourth, the served start at
  yarn:3, reads it wrong under both. BF16 has the sign on only 3 of its 7
  arms, and not at its served shape. The scalar set's five other cases read
  right on every arm.

Inferred: the entailment set gave no evidence of an error of the FP8 weights;
its case 1 is a watch case under BF16's own routes too. With cases 19 and 20
the set asserts only cases clear of the line on every route the card runs,
under both exception weights, so it no longer stands in the way of the FP8
default. On the balance's sign FP8 leans one way on these arms, where the
study (GitHub #667) found its move on the sign's logit mixed in direction
over its routes; one item cannot tell a lean of the FP8 weights from a coin,
the study's 48 literal negatives read 11-28 right under either encoding,
and the minus `scalar` drops is GitHub #671.
Either way the balance is no assertion: on the served default it is wrong
under both.

## Implications

- Spec #672's FP8 default (#682) can flip with both sets green on the
  re-registered arms.
- The FP8 default changes the RTX 5090's rope table only where BF16's load
  adapted its context: with the profile's headroom on a desktop holding
  3 GB, BF16 already serves 917,504 tokens at yarn:3.5.
- The GPU tests run the harness under each exception weights by default; a
  run over every arm is the measurement itself, under five minutes.
- The "yes" gates are thin: a change of the numerics that drops case 19 or
  20 below +1 on an arm makes it a watch case by the rule, and the set,
  holding no "yes" gate, fails its CPU check. That is the signal to re-measure
  and, if needed, write another restatement as these were.

## Limits and unknowns

- One RTX 5090 on one host. The RTX PRO 6000 and the RTX 30 and 40 cards
  (BF16 only) were not measured. Where BF16 lands its context depends on
  what else holds the card at load; both of its rope tables are arms.
- Cases 19 and 20 are the best 2 of 24 wordings on these fourteen arms. On
  an arm not measured here (another card, a later kernel change) their
  margin is expected to be smaller than the recorded minimum.
- Twenty cases are few: the RMS figures say how far a case moves, not
  whether one encoding is more often right. The study's 116 synthetic items
  (GitHub #667) are the evidence on accuracy, and they are not public.
- Which encoding is closer to the BF16 checkpoint on these cases was not
  measured.

## Follow-ups

- [GitHub #682](https://github.com/gpillon/ignis/issues/682): the FP8 default.
- [GitHub #670](https://github.com/gpillon/ignis/issues/670): `uncertainty` does not see the sign.
- [GitHub #671](https://github.com/gpillon/ignis/issues/671): `scalar` drops the minus on literal negatives.
