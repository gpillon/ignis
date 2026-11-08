# A Flash-Next verify round past 8 rows agrees with itself but drifts from speculation off past the near-tie rule

- Kind: experiment
- Status: current
- Observed: 2026-10-18
- Last verified: 2026-10-18
- Scope: kernel / the Flash-Next verify round's row budget (`verify::kMaxRows`, `--spec-draft-rows`), MTP speculation's losslessness seam (`crates/core/tests/flash_next_speculative_gpu.rs`, `crates/core/tests/support/logit_drift.rs`)
- Topic: flash-next
- Related: https://github.com/gpillon/ignis/issues/329 (item 5), [ADR 0056](../adr/0056-speculation-past-eight-verify-rows-is-numerically-equivalent.md), [spec flash-next/07](../specs/flash-next/07-mtp-speculation.md), [the rtxpro6000 profile measured](2026-10-18-rtxpro6000-profile-measured.md), [verify rows on the RTX PRO 6000](2026-10-18-verify-rows-on-the-rtx-pro-6000.md), [batched decode width drift](2026-09-14-batched-decode-width-drift.md), `.scratch/verify-rows/` (branch `verify-rows`)
- Superseded by: none
- Summary: A Flash-Next verify round past 8 rows agrees with itself bit for bit but drifts from speculation off past the near-tie rule (up to 58 BF16 ulps, against 40 at 8), so ADR 0056 holds it numerically equivalent; on the 5090 it is 0.74-0.80x.

## Question

The verify round carried at most 8 rows (lanes x (drafts + 1)), so MTP drafted
nothing from five lanes on. With the bound raised to 32 (`--spec-draft-rows`
up to 32, default 8 unchanged), does a 16-row round keep MTP lossless as the
repository's seam defines it -- bit for bit across drafters at one round
shape, and within the near-tie rule of speculation off -- and what does it
cost?

## Evidence

**Setup.** RTX 5090 (PCIe Gen 3 x16), branch `verify-rows` (ff02fecf and the
test commit after it), the Flash-Next artifact and its MTP companion
(`qwen3_8_flash_next_mtp_3p0-v2`), 3 draft tokens, 8 decode lanes, 8,192
tokens of context, greedy, 40 tokens a lane; prompts: the four G1 prompts and
test2048 windows 3, 9, 17 and 33 (40 tokens each), lanes taking the first
4, 5 or 8. At 16 rows the windows by width are 3, 3, 3, 3, 2, 1, 1, 1, so
the cells are rounds of 16 (4 lanes), 15 (5) and 16 rows (8).

**What changes past 8 rows.** The MoE decode route keeps its one launch of
8 tokens: a wider round runs it in launches of 8 (`fn_moe_experts_decode`).
Every other route switch at 8 is crossed: the FP8 linear's GEMV (1..8 tokens)
becomes its tensor-core route, the BF16 linear's GEMV its WMMA route, the
hyper-connection mix's decode route its prefill route, the GDN and QSA
grouped GEMVs separate linears, the shared expert's GEMV its tensor-core
route. The sparse attention's split count already varies with the rows below
8 (64 splits at up to 4 rows, 32 at 8, 16 at 16).

**The wide shapes agree with themselves.**
- `flash_next_moe_rows` (CTest): 9, 16, 24 and 32 rows in launches of 8 on
  the register kernel, each accumulator row the bits of that row's own
  one-token launch; a 24-row call replayed from a graph, the eager bits.
- Reject (every draft refused) against none (no drafts) at 4, 5 and 8
  lanes, both KV formats: the same text and the same probe logits bit for bit.
- The oracle (none's own text as drafts, every column of every lane
  committed) in BF16: 120/120, 130/130 and 160/160 drafts accepted, the
  same text and probe bits as none.

**Head against none is not a seam once lanes finish apart.** Replaying the
head's per-lane commit schedule (5 lanes) with a host drafter reproduced the
head's text and probe bits exactly. Non-uniform commit counts on equal width
schedules (two lanes alternating 4 and 1 committed columns, at 8 rows, and
four lanes at 16) left none's state bit for bit. When one lane finishes
rounds before another, the rest run their last rounds narrower -- another
round shape -- and their probe differs, at 8 rows as well (two lanes at
window 3, four at window 1: one lane's probe differs, the text does not).
The existing BF16 head == none assertion
(`the_mtp_head_drafts_and_the_text_is_kept`) holds by construction at one
lane only, and held at three by chance.

**Speculation on against off**
(`a_sixteen_row_budget_keeps_the_text_and_the_state_from_four_to_eight_lanes`
as of 6d2c95af, replaced by the teacher-forced tests below in d1830177; a
divergence passes when the prefill route's logits after the agreed text put
the two picks within one BF16 ulp, or side with spec-on):

| budget | format | lanes | divergences (lane @ token: prefill gap off - on) | refused |
|---:|---|---:|---|---|
| 8 | BF16 | 4, 5, 8 | none | -- |
| 8 | hq-e8-2b | 2, 3, 4 | none (oracle against none: none) | -- |
| 16 | BF16 | 4, 5 | none | -- |
| 16 | BF16 | 8 | 4@38: -1.19; 5@7: 0.125 (1 ulp); **7@3: 0.375 (6 ulps)** | 1 |
| 16 | hq-e8-2b | 4 | **0@26: 1.31 (21 ulps)**; 3@29: 0.125 | 1 |
| 16 | hq-e8-2b | 5 | 4@17: 0.125 | -- |
| 16 | hq-e8-2b | 8 | 3@20: 0; 4@7: 0; 5@7: -0.125 | -- |
| 16 | hq-e8-2b | 8 | oracle against none: 3@29, 4@30, 6@8 pass; **5@32: 0.5 (4 ulps)** | 1 |

At 8 rows hq-e8-2b, 5 and 8 lanes run no drafts (window 0), so those cells
equal speculation off trivially. The BF16 5-lane divergence at lane 4 token
38 shows speculation off itself 1.19 logits (9.5 ulps) from the prefill
route's pick there, passing only because the rule takes the prefill route's
side.

**Teacher-forced against spec-off** (after the owner's decision, ADR 0056;
`a_*_row_budget_is_numerically_equivalent_to_speculation_off` and
`the_default_budgets_rounds_keep_the_wide_budgets_rules`, the rules in
`crates/core/tests/support/logit_drift.rs`). A load of 8 decode lanes at the
budget, the same eight prompts, 40 tokens a lane. Spec-off's one-token rounds
are run with their logits read back (`ignis_decode_options::out_logits`); then
verify rounds are offered spec-off's own text as drafts, so each drafted
column holds spec-off's token while the lane's committed text is spec-off's,
and each such row is compared with spec-off's row at the same text index. A
lane whose pick moved leaves the comparison after that round. Drift is the
largest |on - off| over the vocabulary in BF16 ulps at the spec-off row's
largest magnitude; a moved pick's gap is the spec-off row's lead of its pick
over the wide one's, in the same ulps; log-prob is the mean log-probability of
spec-off's next token, wide minus spec-off; acceptance is the head's
first-draft acceptance in its own run, and two lanes at a time the same
prompts' (the head run on them in pairs, whose rounds take 8 rows -- 2 lanes
x 4 columns -- at every budget; the band check, added in the review fixes'
run, `.scratch/verify-rows-fixes/fnbudget.log`, whose drift, pick and
log-prob columns repeated the earlier run's exactly).

| budget | format | lanes x columns | rows | drift max / p99 / median | picks moved (lane@index: gap) | log-prob | acceptance | two lanes at a time |
|---:|---|---|---:|---|---|---:|---:|---:|
| 8 | BF16 | 2 x 4 | 44 | 35.3 / 35.3 / 3.0 | 1@1: 1 | +0.0077 | 0.875 | 0.875 |
| 8 | BF16 | 4 x 2 | 160 | 9.7 / 8.1 / 0 | -- | +0.0012 | 0.937 | 0.891 |
| 8 | hq-e8-2b | 2 x 4 | 80 | 38.0 / 38.0 / 3.4 | -- | +0.0058 | 0.875 | 0.875 |
| 8 | hq-e8-2b | 4 x 2 | 160 | 40.2 / 39.6 / 3.0 | -- | -0.0026 | 0.937 | 0.891 |
| 16 | BF16 | 4 x 4 | 160 | 39.2 / 36.0 / 2.9 | -- | +0.0014 | 0.870 | 0.891 |
| 16 | BF16 | 5 x 3 | 199 | 39.2 / 36.0 / 3.5 | 4@37: 2 | -0.0031 | 0.867 | 0.869 |
| 16 | BF16 | 8 x 2 | 250 | 39.2 / 26.3 / 4.3 | 7@2: 8; 5@6: 1; 4@37: 4 | -0.0029 | 0.822 | 0.800 |
| 16 | hq-e8-2b | 4 x 4 | 152 | 36.0 / 33.1 / 3.0 | 3@28: 1 | -0.0047 | 0.891 | 0.891 |
| 16 | hq-e8-2b | 5 x 3 | 169 | 54.0 / 35.6 / 3.0 | 4@6: 4 | -0.0077 | 0.878 | 0.869 |
| 16 | hq-e8-2b | 8 x 2 | 250 | 46.0 / 35.6 / 4.7 | 5@6: 1; 4@16: 2; 6@22: 1; 6@23: 1 | -0.0103 | 0.760 | 0.798 |
| 24 | BF16 | 6 x 4 | 208 | 39.2 / 24.4 / 3.7 | 5@6: 1; 4@37: 2 | -0.0039 | 0.787 | 0.812 |
| 24 | BF16 | 8 x 3 | 251 | 39.2 / 26.3 / 4.3 | 7@2: 8; 5@6: 1; 4@37: 4 | -0.0042 | 0.797 | 0.800 |
| 24 | hq-e8-2b | 6 x 4 | 196 | 36.0 / 33.1 / 3.6 | 5@6: 1; 5@7: 0; 3@28: 1; 4@33: 2 | -0.0117 | 0.744 | 0.810 |
| 24 | hq-e8-2b | 8 x 3 | 211 | 54.0 / 29.0 / 3.4 | 4@6: 4; 5@6: 1; 5@7: 0; 7@7: 4; 6@22: 5 | -0.0058 | 0.806 | 0.798 |
| 32 | BF16 | 8 x 4 | 252 | 39.2 / 26.3 / 4.3 | 7@2: 8; 5@6: 1; 4@37: 4 | -0.0031 | 0.766 | 0.800 |
| 32 | hq-e8-2b | 8 x 4 | 260 | 58.3 / 36.0 / 5.6 | 5@6: 1; 5@7: 0; 6@22: 2; 3@28: 1; 4@33: 3 | -0.0108 | 0.750 | 0.798 |

The widest rows move a tail logit: 32 rows hq-e8-2b, lane 6 at 9, id 142890
from 0.71 to 8.0 where the pick stands at 23.5 (7.3 logits, 58.3 ulps at
scale 23.5). A gap of 0 is an exact tie on spec-off's row (it picks the
lower id). The BF16 cells' 39.2 is one row (lane 0 at 25) shared by every
wide budget. In every cell, at every budget, the exact checks held: reject
left none's text and probe bit for bit, the BF16 oracle (none's text as
drafts) every draft accepted and none's text and probe, every lane of every
round of the teacher-forced run and of the head's run committed its own
rows' picks (accepted the longest prefix of drafts each the pick after the
column before it, and stood pending on the pick after its run), and the
head's run repeated its text, rounds, drafts, rows (hashed) and probes.

The acceptance band: a wide cell's first-draft acceptance sits at most
0.066 from the same prompts' two lanes at a time (24 rows hq-e8-2b, 6 lanes:
0.744 against 0.810), and the default's own 4-lane cells 0.046 from theirs
(window 1 against 3). The lower acceptance of the 6- and 8-lane cells is
their prompts': the eight prompts two at a time read 0.80, the four G1
prompts 0.89.

For scale: the rows of the 8-row BF16 two-lane cell's lane 1 after its pick
moved at index 1 -- a text one token off spec-off's, compared at the same
indices anyway, what a row read on another context looks like -- drift
33-202 ulps (median 103 over its 142 rows; 89% past 64, 59% past 96), most
of it on the top logits (a pick at 21.5 read at -0.9).

**VRAM** (plan lines, `flash_next_bind` at 131,072 tokens hq-e8-2b, 4 lanes x 3
drafts, budget 8 -> 16): verify round 13.2 -> 24.0 MB, sampling 4.6 -> 8.6 MB
(the head's logits, one row a column), decode graph 40.8 -> 75.3 MB (the QSA
listed hq rows of every column); workspace and activations unchanged, the
MoE workspace's decode regions kept at 8 tokens. A budget no round reaches
reserves nothing (3 lanes at 12 rows and at 32 plan alike). At 8 lanes x 3
drafts, the widest load, the same lines at each budget (plan figures from
the same CTest, host-only: not measured on the card; a verify-only plan, so
an MTP load's own head lines are not in it):

| budget | rows of the widest round | verify round | sampling | decode graph | the three | over budget 8 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 8 (no drafts at 8 lanes) | 15.7 MB | 4.6 MB | 40.8 MB | 61.2 MB | -- |
| 16 | 16 | 26.4 MB | 8.6 MB | 75.3 MB | 110.3 MB | +49.2 MB |
| 24 | 24 | 37.2 MB | 12.6 MB | 110.0 MB | 159.7 MB | +98.6 MB |
| 32 | 32 | 47.9 MB | 16.5 MB | 144.2 MB | 208.7 MB | +147.5 MB |

At budgets 0, 2, 6 and 8 the plan lines are main's (0a810c0b) byte for
byte in the 15 cells the same CTest pins against the figures main's own build
planned: no speculation and verify-only, 2, 3 or 7 drafts, 1, 2, 4, 5 or 8
lanes and the default's, both KV formats (`flash_next_bind`; its run at
the branch's head, with the 8-lane lines of the table above:
`.scratch/verify-rows-fixes/ctest_flash_next_bind.log`). A one-off probe of
96 such cells against main's build agreed as well, but neither its code nor
its output was kept, so the 15 pinned cells are the evidence. MTP loads
were not planned: these descriptors carry no head.

**Cost on the 5090** (`flash_next_mtp_bench mtp --draft-tokens 3 --lanes 8`,
hq-e8-2b, 1,536-token prompts, 128 tokens, 16 GB expert cache; decode is
bound by the experts copied over PCIe):

| lanes | rows 8: tok/s, ms/round, GB moved/round | rows 16: tok/s, ms/round, GB moved/round | x |
|---:|---|---|---:|
| 4 | 62.3, 98.8, 1.16 (window 1) | 46.4, 165.0, 1.63 (window 3) | 0.74 |
| 8 | 61.2, 130.7, 1.55 (no drafts) | 49.2, 266.8, 2.76 (window 1) | 0.80 |

## Finding

Observed (the tables above):

- Past 8 rows the verify round agrees with itself bit for bit: across
  drafters, across commit counts on one width schedule, and in the MoE's
  launches of 8.
- At 16 rows three checks exceed the near-tie rule against speculation off
  (4 to 21 BF16 ulps), where the default's 8 rows show no divergence on the
  same prompts.
- Teacher-forced, the default's own 8-row rounds move logits up to 40 ulps
  from spec-off, 16-32 rows up to 58; picks move only at gaps of 0-8 ulps;
  the mean log-probability stays within 0.012 nats a token, either way.
- On the 5090 a budget past 8 runs at 0.74-0.80x the default's tok/s, and a
  round copies more experts (1.16 -> 1.63 and 1.55 -> 2.76 GB).

Inferred:

- Since every round commits its own rows' picks and agrees with itself
  across drafters and commit counts, the commit and the wider records (the
  head's logits, the QSA listed rows, the MoE's launches) are right; what
  remains against speculation off is the tensor-core routes' summation
  order.
- The wide budgets' drift is the default's class of drift, about 1.5x wider
  at its widest, not a new error: a row read on another context sits 33-202
  ulps off.
- So a budget past 8 is not held lossless by the repository's seam, only
  numerically equivalent to speculation off.
- The slowdown is not only the extra expert copies. On the RTX PRO 6000,
  where no expert crosses PCIe, a round past 8 rows still takes a fixed
  +44-48 ms over the 8-row round (measured afterwards, [verify rows on the RTX PRO 6000](2026-10-18-verify-rows-on-the-rtx-pro-6000.md)),
  which points at the routes a wide round switches to (not profiled).

## Implications

The knob ships with its default unchanged; a load that names no budget plans
and runs as before (its plan lines are main's, pinned). The owner chose
option 3 (2026-10-18, ADR 0056): a budget past 8 is held numerically
equivalent to speculation off -- the table above's checks: every logit within
64 ulps (a margin of only 1.1x over the wide budgets' 58.3 on this card,
1.6x over the default's 40.2), a moved pick only at a near tie of 16 ulps
(2x the measured 8), the mean log-probability within 0.02 nats either way,
and a first-draft acceptance of at least 0.5 and within 0.1 of the same
prompts' at 8 rows (1.5x the widest measured 0.066) -- not to its text. The options as they were put:

1. **Launches of 8 rows everywhere a route switches at 8**: the linears
   (`fn_linear`, under a decode-phase flag, since a prefill chunk of 9..32
   tokens must keep its tensor-core route), the shared expert, the
   hyper-connection decode route with its pending inject, the grouped GEMVs.
   Each row then computes as in a round of 8; the dense weights are read once
   per 8 rows (about 2.5 GB more a 16-row round, ~1.4 ms at 1.8 TB/s). The
   attention's split count still varies.
2. **GEMV kernels wider than 8 tokens** (16 or 32), each token's sums in the
   8-token kernel's order, proven bit for bit by CTest, taken only by
   decode-phase calls: the FP8 single, SwiGLU and grouped GEMVs, the BF16
   GEMV, the hyper-connection decode route's accumulators and shared
   activations. The weights read once a round; a kernel redesign.
3. **Accept the drift** and hold budgets past 8 to a looser rule: its size
   is of the class of speculation off's own disagreement with the prefill
   route (9.5 ulps above).

## Limits and unknowns

- One prompt set, 40 tokens a lane, 2-8 lanes, one card. Three refusals
  over about 1,400 compared tokens (the head's and the oracle's); their rate
  on longer texts is unmeasured. The teacher-forced tolerance is measured on
  the same prompts on the 5090; another card's tensor-core tiling may move it.
- The teacher-forced text is spec-off's own greedy text, not human text:
  the log-probability check bounds how far the wide route's reading of that
  text moves, either way (measured -0.0117 to +0.0077), not a perplexity.
- The 64-ulp tolerance has a 1.1x margin over one tail logit on one card.
  It stays that tight because the discriminator is close (a row on another
  context passes 96 ulps in only 59% of rows); the RTX PRO 6000's tiling may
  move the drift, so it is re-derived there (Follow-ups).
- The VRAM past the default is plan figures (host CTest), not measured on
  the card.
- Self-consistency cannot exclude a deterministic error specific to one
  wide shape (for example 8 lanes x 2 columns); the teacher-forced
  comparison above is the per-round logit discriminator, every column of
  every wide shape against spec-off's row, and it puts each shape inside the
  default's drift class. A subtler error (one that moves logits by tens of
  ulps) would pass it.
- This finding measured no card but the 5090. What the wide budgets pay on
  the RTX PRO 6000 was measured afterwards ([verify rows on the RTX PRO 6000](2026-10-18-verify-rows-on-the-rtx-pro-6000.md)): every
  budget past 8 loses to 8 there (0.42-0.84x at 4 and 8 lanes), and the
  `rtxpro6000` profile keeps 8.

## Follow-ups

- On the RTX PRO 6000, the four budget tests, reading the drift column to
  re-derive the 64-ulp tolerance (and the 16-ulp pick bound) on that card,
  before any profile takes a budget past 8. (Whether the profile takes one
  is answered: it keeps 8, [verify rows on the RTX PRO 6000](2026-10-18-verify-rows-on-the-rtx-pro-6000.md).)
- A change of the wide routes (a GEMM tiling, option 2) re-measures the
  tolerance: rerun the four budget tests and read the drift column.
