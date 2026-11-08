# Contributed entailment fixture

`cases.json` is an evaluation set for `/v1/decide`'s `noul` on "does the
cited evidence support the claim?", contributed by Luigi Neri from the
strategic account planning domain. Every account
name is fictional and every field value synthetic; it carries no customer
data. The labels are the contributor's judgement, not a panel's; where one
is arguable (6 and 11 above all) the case's `note` argues it.

The question, `criteria` included, is fixed for every case and is
load-bearing: the labels are defined relative to its definition of
"support" (entailment, not plausibility). Without the criteria the same
model scores case 6 at p(yes) 0.89; with them, 0.03-0.06.

## What ignis changed

**Names.** The contributed cases cite internal mart and field names; these
carry public ones, every name renamed and not only some
(`decide_entailment_fixture.rs` enumerates them). The model reads the
names, so the numbers here are not comparable with ones measured on the
internal names. A partial rename is worse than none: a public field beside
a sibling still under its internal name moved case 10 from p(yes) 0.996 to
0.119, and the consistent rename brought it back to 0.984.

**Claims and evidence.** Six cases changed; each case's `note` says why:

- **1, 2, 17**: "high-value whitespace potentials" became "high-value
  potentials". No field names whitespace, so under the criteria the
  qualifier was unsupported, and the stock model said so (1 at 0.22, 17 at
  0.18; 0.95 and 0.84 without it). 2 changed with 1 to keep the pair.
- **12**: `node roles present = [...]` became
  `node roles (all nodes) = [...]`. As contributed, only the node count
  implied the list was complete.
- **13, 14**: `as_of_date = 2026-09-20` was added. "Less than 90 days away"
  needs a today, and `retrieved` is the extraction date. It went into 14
  too, so that pair still differs only in the renewal date.

**Owner's review (2026-11-08, #537).** The owner read every case and decided
by one reading rule, then case by case. The rule: arithmetic on numbers the
evidence states is a direct reading; a conversion that needs the calendar is
outside knowledge. The evidence and the labels of every case are unchanged;
the claim lines of five cases and the notes of two were:

- **3, 4**: "more than three months" became "more than 90 days". Three months
  in days is a calendar fact the evidence lacks; 118 > 90 and 12 < 90 read
  straight from it. Owner's decision, 2026-11-08, #537.
- **12**: "No economic buyer has been identified on the account map" became
  "No node on the account map has the role Economic Buyer". The first also
  asked that a buyer would appear as a node role; the second says what the
  list shows. Owner's decision, 2026-11-08, #537.
- **16**: "has not been modified since March 2026" became "was last modified in
  March 2026". "Since March" reads two ways, one of them contradicted by the
  4 March date. Owner's decision, 2026-11-08, #537.
- **18**: "currently" was dropped. The evidence is a snapshot of 2026-09-20 and
  "currently" asks that it still holds. Owner's decision, 2026-11-08, #537.
- **11, 13** (notes only): 11 says that neither "stale" follows without a
  threshold, so its "no" has two reasons; 13 says that under the strict
  criteria the model's "no" is a defensible reading, not only an error, since
  counting days between dates is a calendar conversion. Owner's decision,
  2026-11-08, #537.

Every other case is kept as it was. The set stays 18 cases, 8 "yes" and 10
"no". The two ignis-written "yes" cases the review allowed for were not
added: the re-measurement left two "yes" gate cases (1 and 18), so the gate
holds a case of each label without them.

**Tiers.** `tier` is ignis's, from one measurement of this file on the
RTX 5090 (2026-11-08, main 86c2f8ff, taken again after the review above), taken over every route the card runs
rather than one: a case near the 0.5 line falls on either side under any
change of the numerics, and asserting it reports the change, not a fault.

One rule gives each case its tier from its logits over the arms below:
`measurement.rule` in `cases.json`, which `tier_by_rule` in
`crates/server/tests/support/entailment.rs` applies.

The measurement set, five arms:

| arm | surface | GDN prefill | prompt attention |
|-----|---------|-------------|------------------|
| `h-vendored` | `decide_entailment_gpu.rs`, `IGNIS_GDN_PREFILL=vendored` | vendored | fp8 |
| `h-ours` | `decide_entailment_gpu.rs`, `IGNIS_GDN_PREFILL=ours` | ignis's | fp8 |
| `s-vendored` | live server, `IGNIS_GDN_PREFILL=vendored` | vendored | fp8 |
| `s-ours` | live server, `IGNIS_GDN_PREFILL=ours` | ignis's | fp8 |
| `s-int8` | live server, `IGNIS_GDN_PREFILL=vendored`, `--model-prefill-attention int8` | vendored | int8 |

The harness cannot select the int8 band, so that arm is the server's. The
rule, the date, the build, the command that redoes each arm and every
case's logit in every arm are in `cases.json`: `measurement`, and `logits`
on each case. `decide_entailment_fixture.rs` fails when a case's tier is not
the one the rule gives from its logits, when a case has no tier, or when a
label has no gate case. Which case is which tier is `cases.json`'s record
alone: `decide_entailment_gpu.rs` asserts the `gate` cases on the card's
default route and prints the `watch` and `known_failure` ones.

It is committed rather than fetched because `decide_entailment_gpu.rs` is
a GPU-profile test: under `IGNIS_GPU_PROFILE=1` a missing fixture is a hard
failure (ADR 0006, `docs/agents/testing.md`).
