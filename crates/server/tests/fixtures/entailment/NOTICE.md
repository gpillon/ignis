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
stock artifact `decide_entailment_gpu.rs` loads (2026-09-25, 17/18):

| tier | cases | meaning |
|------|-------|---------|
| `gate` | 1-3, 5-11, 14, 15, 17 | right by at least one logit; asserted by `decide_entailment_gpu.rs` |
| `watch` | 4, 12, 16, 18 | right, within one logit of 0.5; printed |
| `known_failure` | 13 | right label, wrong answer (date arithmetic, p 0.35); printed |

It is committed rather than fetched because `decide_entailment_gpu.rs` is
a GPU-profile test: under `IGNIS_GPU_PROFILE=1` a missing fixture is a hard
failure (ADR 0006, `docs/agents/testing.md`).
