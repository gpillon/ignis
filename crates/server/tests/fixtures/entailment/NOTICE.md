# The entailment fixture

`cases.json` is an evaluation set for `/v1/decide`'s `noul` on "does the
cited evidence support the claim?", contributed by Luigi Neri from the
strategic account planning domain. Every account
name is fictional and every field value synthetic; it carries no customer
data. The labels are the contributor's judgement, not a panel's; where one
is arguable (6 and 11 above all) the case's `note` argues it. Cases 19 and
20 are not contributed: ignis wrote them ([below](#cases-19-and-20-ignis-wrote-them)),
and `origin` says which cases are whose.

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

Every other case is kept as it was. The review left the set at 18 cases,
8 "yes" and 10 "no", and did not add the two ignis-written "yes" cases it
allowed for: its measurement left two "yes" gate cases (1 and 18), so the
gate held a case of each label without them.

## Cases 19 and 20: ignis wrote them

Added on 2026-11-18 (GitHub #675), as the review allowed. Once every arm
named its exception weights and both were measured (below), no contributed
"yes" case stayed a gate: case 1 reads −0.750 on `h-fp8-vendored` and case
18 −0.250 on `s-bf16-yarn3.5`, and a set whose gate holds no "yes" fails
its own check. Each new case restates one field of its evidence and needs
no inference, uses only the set's public names and an unused `PLAN-` id, and
was registered only as a gate on every arm:

- **19**, "The executive sponsor is engaged on this account." over
  `exec_sponsor_engaged = true`: case 9's field, set rather than null.
- **20**, "The renewal date is 2027-03-31." over `renewal_date = 2027-03-31`:
  the field cases 13 and 14 count days to, restated as written.

24 wordings were tried, in one pool and one pass, each on every arm, and
none was edited after it was read. The two clearest of the line became the
cases; the other 22 are `drafts` in `cases.json`, each with its logit on
every arm (`IGNIS_DECIDE_DRAFTS=1` posts them after the cases, in the file's
order). Every wording tried, by its smallest logit over the 14 arms:

| wording | claim | smallest logit | on |
|---------|-------|---------------:|----|
| case 19 | The executive sponsor is engaged on this account. | +1.625 | `h-fp8-vendored` |
| case 20 | The renewal date is 2027-03-31. | +1.500 | `s-fp8-yarn3.5` |
| `status-with-id` | Plan PLAN-1022 has the status Active. | +1.250 | `h-bf16` |
| `node-count` | The account map has 3 nodes. | +1.000 | `s-fp8-int8` |
| `initiative-unlinked` | The initiative 'Platform Modernisation' has no linked opportunity. | +1.000 | `s-fp8-yarn3.5` |
| `field-plan-status` | plan_status is Active. | +1.000 | `h-bf16-vendored` |
| `initiative-linked` | The initiative has 3 linked opportunities. | +0.875 | `s-bf16-yarn3` |
| `high-value-count` | 4 of the account plan's potentials are high-value. | +0.625 | `s-fp8-int8` |
| `champion-identified` | A champion has been identified on this account. | +0.500 | `s-bf16-yarn3` |
| `status-active` | The plan status is Active. | +0.250 | `s-bf16-noreuse` |
| `scorecard-total` | The scorecard has 22 questions. | +0.250 | `s-bf16-int8` |
| `champion-sponsor-false` | The account has an identified champion. | +0.125 | `s-fp8-yarn3` |
| `sponsor-not-engaged` | The executive sponsor is not engaged on this account. | 0.000 | `h-fp8-vendored` |
| `field-open-tasks` | tasks_open_count is 5. | −0.125 | `h-bf16-vendored` |
| `total-tasks` | The account plan has 8 tasks in total. | −0.250 | `s-bf16-yarn3` |
| `scorecard-answered` | 14 of the 22 scorecard questions are answered. | −0.250 | `s-fp8-noreuse` |
| `plan-unlinked` | The account plan has no linked opportunities. | −0.250 | `s-bf16-int8` |
| `open-tasks` | The account plan has 5 open tasks. | −0.250 | `s-bf16-vendored` |
| `no-open-tasks` | No task on this account plan is open. | −0.250 | `h-fp8-vendored` |
| `contacts-mapped` | 3 contacts are mapped on this account. | −0.250 | `s-fp8-yarn3` |
| `potentials` | The account plan has 11 potentials. | −0.375 | `s-fp8-noreuse` |
| `tasks-in-progress` | 3 tasks on this account plan are in progress. | −0.875 | `s-fp8-noreuse` |
| `map-date` | The account map was last modified on 2026-03-04. | −1.500 | `s-bf16-int8` |
| `role-present` | A node on the account map has the role Champion. | −1.750 | `h-bf16` |

Six wordings clear the gate margin on every arm, and none by much: a
restatement this plain still moves by up to 5 logits between two arms, so
the minimum over 14 arms is a thin edge even on a mean of +3 to +4. A
wording that falls below the margin on a later measurement is not a fault
either; its case becomes a watch case by the rule, and the set needs
another "yes" gate written the same way.

## Tiers

`tier` is ignis's, from one measurement of this file on the RTX 5090
(2026-11-18, main ab2adc85, GitHub #675), taken over every route the card
runs rather than one: a case near the 0.5 line falls on either side under
any change of the numerics, and asserting it reports the change, not a
fault.

One rule gives each case its tier from its logits over the arms below:
`measurement.rule` in `cases.json`, which `tier_by_rule` in
`crates/server/tests/support/entailment.rs` applies.

The measurement set, fourteen arms. Every arm names its exception weights,
context and rope table, its GDN prefill route, its prompt attention band and
prompt reuse, and reads no config file (`support/decide_arms.rs`): a
`harness` arm is the shape `decide_entailment_gpu.rs` builds by hand (chunk
1,024, no drafter, no vision), a `served` arm a start under the `rtx5090`
profile (chunk 2,048, DFlash2 at 7, vision on) with each of those values
named as a flag, resolved with no file and no environment and loaded
through the server's own load path. `s-bf16-yarn3` and `s-fp8-yarn3.5` are
each exception weights' served shape on the RTX 5090; every other served
arm moves one value from its weights' served shape. FP8 keeps the profile's
917,504 tokens. BF16 lowers them to 786,432, and so its rope to yarn:3,
under a 1.5 GB VRAM headroom (or a desktop holding more of the card); under
the profile's 1.2 GB on a desktop holding 3 GB it keeps 917,504 and
yarn:3.5, which `s-bf16-yarn3.5` reads, since at an equal rope table the
context moves none of these reads.

| arm | surface | weights | context | rope | GDN prefill | prompt attention | prompt reuse |
|-----|---------|---------|--------:|------|-------------|------------------|--------------|
| `h-bf16` | harness | bf16 | 40,960 | none | ignis's | fp8 | on |
| `h-fp8` | harness | fp8 | 40,960 | none | ignis's | fp8 | on |
| `h-bf16-vendored` | harness | bf16 | 40,960 | none | vendored | fp8 | on |
| `h-fp8-vendored` | harness | fp8 | 40,960 | none | vendored | fp8 | on |
| `s-bf16-yarn3` | served | bf16 | 786,432 | yarn:3 | ignis's | fp8 | on |
| `s-fp8-yarn3.5` | served | fp8 | 917,504 | yarn:3.5 | ignis's | fp8 | on |
| `s-bf16-yarn3.5` | served | bf16 | 786,432 | yarn:3.5 | ignis's | fp8 | on |
| `s-fp8-yarn3` | served | fp8 | 917,504 | yarn:3 | ignis's | fp8 | on |
| `s-bf16-vendored` | served | bf16 | 786,432 | yarn:3 | vendored | fp8 | on |
| `s-fp8-vendored` | served | fp8 | 917,504 | yarn:3.5 | vendored | fp8 | on |
| `s-bf16-int8` | served | bf16 | 786,432 | yarn:3 | ignis's | int8 | on |
| `s-fp8-int8` | served | fp8 | 917,504 | yarn:3.5 | ignis's | int8 | on |
| `s-bf16-noreuse` | served | bf16 | 786,432 | yarn:3 | ignis's | fp8 | off |
| `s-fp8-noreuse` | served | fp8 | 917,504 | yarn:3.5 | ignis's | fp8 | off |

Each arm's reads repeat: every case and draft read the same, bit for bit,
in two passes, and the four wordings that moved between the case slots and
the drafts between them, read after other prompts, did not move either.

The 2026-11-08 measurement had five arms, and none named its exception
weights: the three live-server arms ran FP8 from a user config file, the two
harness arms BF16. Its harness arms read as `h-bf16` and `h-bf16-vendored`
here and its live-server arms with ignis's and the vendored GDN prefill as
`s-fp8-yarn3.5` and `s-fp8-vendored`, bit for bit. Against that measurement,
cases 1 and 18 went from gate to watch and 19 and 20 joined as gates; the
other sixteen kept their tiers (ten "no" gates, watch 3, 10, 12, 16 and 17,
known failure 13).

The rule, the date, the build, the command that redoes each arm and every
case's and draft's logit in every arm are in `cases.json`: `measurement`,
`logits` on each case, and `drafts`. `decide_entailment_fixture.rs` fails
when a case's tier is not the one the rule gives from its logits, when a
case has no tier, when a label has no gate case, when an arm's shape is not
the one it names, or when an arm the set needs is missing. Which case is
which tier is `cases.json`'s record alone: `decide_entailment_gpu.rs`
asserts the `gate` cases on every arm it loads and prints the `watch` and
`known_failure` ones.

It is committed rather than fetched because `decide_entailment_gpu.rs` is
a GPU-profile test: under `IGNIS_GPU_PROFILE=1` a missing fixture is a hard
failure (ADR 0006, `docs/agents/testing.md`).
