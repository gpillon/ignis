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

**Tiers.** `tier` is ignis's, from one measurement of this file on the
RTX 5090 (2026-11-08, main 1188385c), taken over every route the card runs
rather than one: a case near the 0.5 line falls on either side under any
change of the numerics, and asserting it reports the change, not a fault.

The rule: a case is a `gate` when it is right by at least one logit, taken
as the minimum over every arm below, and the gate holds at least one case of
each label. A `known_failure` is wrong on every arm. Every other case is a
`watch` case.

The measurement set, five arms:

| arm | surface | GDN prefill | prompt attention |
|-----|---------|-------------|------------------|
| `h-vendored` | `decide_entailment_gpu.rs` | vendored | fp8 |
| `h-ours` | `decide_entailment_gpu.rs`, `IGNIS_GDN_PREFILL=ours` | ignis's | fp8 |
| `s-vendored` | live server, `/v1/decide` | vendored | fp8 |
| `s-ours` | live server, `IGNIS_GDN_PREFILL=ours` | ignis's | fp8 |
| `s-int8` | live server, `--model-prefill-attention int8` | vendored | int8 |

The harness cannot select the int8 band, so that arm is the server's. The
rule, the date, the build, the command that redoes each arm and every
case's logit in every arm are in `cases.json`: `measurement`, and `logits`
on each case. `decide_entailment_fixture.rs` fails when a case's tier is not
the one the rule gives from its logits, when a case has no tier, or when a
label has no gate case.

| tier | cases | meaning |
|------|-------|---------|
| `gate` | 1, 2, 4-9, 11, 14, 15 | asserted by `decide_entailment_gpu.rs` on the card's default route |
| `watch` | 3, 10, 12, 16-18 | printed |
| `known_failure` | 13 | right label, wrong answer on every arm (date arithmetic); printed |

It is committed rather than fetched because `decide_entailment_gpu.rs` is
a GPU-profile test: under `IGNIS_GPU_PROFILE=1` a missing fixture is a hard
failure (ADR 0006, `docs/agents/testing.md`).
