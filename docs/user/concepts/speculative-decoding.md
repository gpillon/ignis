---
title: "Speculative decoding"
description: "How DFlash2 and the MTP heads draft tokens that the model then verifies, the 27B's two drafters and how to choose, what acceptance means, how a draft is accepted under sampling, how the row budget is shared across lanes, and why Flash-Next's MTP is off on the RTX 5090."
sidebar:
  order: 5
---

Generating a token takes one pass through the model, and that pass is mostly
the card streaming the weights from memory. A pass that checks several tokens
at once streams the same weights. **Speculative decoding** uses that: a cheap
*drafter* proposes a few tokens ahead, and the model verifies all of them in
one pass and keeps the ones it agrees with. When the drafts are good, one pass
yields several tokens; when they are not, the pass still yields the one token
it would have yielded anyway.

## One round, step by step

A speculative **verify round** replaces the plain decode round
([how ignis serves a request](how-ignis-serves.md)):

1. Each lane has an *anchor*: the next token it already holds.
2. The drafter proposes up to `k` more tokens after it. `k` is the **draft
   tokens** setting.
3. The model runs all lanes' columns (the anchor plus its drafts, `k + 1` per
   lane) through the model's layers once.
4. For each lane, the longest run of drafts that the model itself would have
   chosen is accepted, plus the model's own token after it. That run is the
   lane's **committed run**, between 1 and `k + 1` tokens. A request that
   samples accepts its drafts by chance instead
   ([acceptance under sampling](#acceptance-under-sampling)).
5. The state moves forward by exactly the committed run. The attention layers
   keep the keys of every column; the recurrent layers record what each column
   would do and replay only the committed prefix, so a rejected draft leaves no
   trace.

**The text does not change.** The model, not the drafter, decides every token.
Greedy output is the same as without speculation, up to near-ties where two
tokens are almost equally likely, and sampling keeps its distribution. The
drafter can only change how many tokens a round yields, called the **acceptance**.

The server streams a committed run one token at a time, so a client sees no
difference in shape. The first round after a prompt has the first token as its
anchor, which the server already sent when the prefill ended, so that round
sends only the drafts it accepted, between 0 and `k` tokens.

## Acceptance under sampling

A greedy request accepts a draft when it is the model's own top token. A
request that samples has no single token the model would have chosen: its
sampler draws from a distribution, the model's probabilities after the
request's temperature, top-k, top-p and penalties. So the round takes a draft
by chance, in a way that leaves every token the lane emits distributed exactly
as the sampler would have drawn it, whatever the drafter proposed.

- **The fixed draft** (`--spec-draft-rule fixed`, the rule of earlier
  releases). The drafter
  proposes its single best token d, and the round takes it with the model's
  probability of it, p(d). When it refuses, the lane's token is drawn from the
  model's distribution without d, and the run ends there. Even a drafter that
  always proposed the model's most likely token would see it refused whenever
  the sampler picks another one: on recorded agent turns at temperature 1, 18
  percent of first drafts.
- **The sampled draft** (`--spec-draft-rule sampled`, the default from
  0.11.0, the 27B only). The drafter draws d from its own distribution q over its 16
  candidates, at 0.9 times the request's temperature and top-p 0.95, and the
  round takes it with probability min(1, p(d)/q(d)). When it refuses, the
  lane's token is drawn from what the model's distribution has left over the
  drafter's, max(p − q, 0). This is the standard rule of speculative sampling:
  whatever q is, the emitted token is distributed as p, so the drafter's two
  settings move only how often a draft is taken. Where the drafter's
  distribution is close to the model's, it takes drafts the fixed draft would
  refuse. The reasons it is ignis's own code and not the vendored accept are
  in [ADR 0090](../../adr/0090-the-27bs-sampled-lanes-draw-their-drafts-and-take-them-by-an-accept-of-ignis-s-own.md).

A greedy request drafts and accepts the same way under both rules, so its text
does not change. A sampled request's text keeps its distribution under both,
but not its draws: a fixed seed reproduces a lane's text on one build, and
under `sampled`, the default, that text is not the one the same seed gives under
`fixed`, or gave in an earlier release: the distribution of the text is the
same, a seed's text is not. Neither rule changes what a stop sequence, the
reasoning budget or a JSON schema does to the run the round commits.

## The 27B's two drafters

From 0.11.0 the 27B drafts with **DFlash2** or with its own **MTP head**, one
per load, chosen at start with `--spec-backend`; earlier releases refuse
`--spec-backend mtp` on the 27B. DFlash2 is the default on every card class
but the 12 GB one: every built-in profile but `rtx-12gb` loads it, at 7 draft
tokens, and from 0.11.0 `rtx5090` at 15. MTP is the 12 GB class's default,
from 0.11.0, and opt-in on the
others (`--spec-backend mtp`), for the card where DFlash2's memory is what the
context lacks. Both are verified by the same rounds, so neither changes the
text: greedy output is the spec-off text up to near-ties, and sampling keeps
its distribution.

Which drafter each [card class](profiles.md#the-built-in-profiles) starts
with, under `--profile auto`:

| Card class | Profile | The 27B's drafter |
|---|---|---|
| 12 GB, below 14 GiB (from 0.11.0) | `rtx-12gb` | MTP, 7 draft tokens: DFlash2's 0.7 GB more is the context there ([ADR 0098](../../adr/0098-the-12-gb-card-class-serves-the-27b-from-a-recipe-of-our-own-under-a-byte-ceiling-on-one-lane-drafting-with-mtp.md)) |
| 16 GB | `rtx-16gb` | DFlash2, 7 draft tokens |
| 24 GB | `rtx-24gb` | DFlash2, 7 draft tokens |
| 32 GB | `rtx5090` | DFlash2, 15 draft tokens (from 0.11.0; 7 before) |
| 96 GB | `rtxpro6000` | DFlash2, 7 draft tokens |

Either drafter can be named on any class: `--spec-backend mtp` on the larger
ones, and on `rtx-12gb` `--qwen38-spec-backend dflash2` (at the profile's 7)
or `--spec-backend dflash2 --spec-draft-tokens N`, where the card has room for
it.

| | DFlash2 | MTP head |
|---|---|---|
| What it is | a second small model, five layers over a window of the last 2,048 positions | the checkpoint's own draft layer, one attention layer over the whole context |
| Weights | 1.08 GB, its own | 0.45 GB; it reads the target's embedding and output head |
| State per lane and retained slot | a drafter window of 40 MiB | one vector of the model's hidden size, about 10 KB |
| KV pool | unchanged | one more attention layer in every page: 17/16 of the page, so about 6% fewer tokens for the same bytes |
| Prefill | scratch for its taps in every chunk | the head runs over every prompt position, and every column the head reads is normalized |
| A shared prefix | shared as on any load | each holder keeps its own copy of the prefix's last page, which the VRAM plan holds beside the context |
| Draft rule | the sampled draft (default) or the fixed draft | the fixed draft only |
| Draft tokens | 1 to 15 (1 to 7 before 0.11.0), required; past 7, [two round shapes](#past-seven-drafts-two-round-shapes) | 0 to 7, default 7; 0 loads no speculation |
| Tokens a round, recorded agent turns at 7 drafts | 3.83 (sampled draft), 3.45 (fixed) | 3.90 (NVFP4), 3.88 (16 GB) |
| Images | drafts after them | drafts after them |

What the choice moves is how the card's memory splits between the drafter and
the KV pool, how much a prefill and a round cost, and how many tokens a round
yields; a request's text is the model's under either. Changing the drafter is
a reload ([the VRAM plan](vram-plan.md) shows each one's lines).

## DFlash2, for the 27B

DFlash2 is a small drafter grafted into the 27B's artifact: five layers that
read the target's own hidden states over a sliding window of the last 2,048
positions. It is loaded when `--spec-backend dflash2` is set, together with
`--spec-draft-tokens` (1 to 15; from 0.11.0, 1 to 7 before), and costs no VRAM
when speculation is off. The
reference also keeps a rewrite checkpoint of the drafter's window, restored
when a turn rewrites text past its chat template's rewrite boundary; ignis
never read one and retired it (snapshot format 5), because a prompt checkpoint
taken at the opener already carries the window as it stood there.

- **Acceptance.** On a replay of 200 real agent turns, sampled at temperature
  1, it averaged 3.83 tokens a round under the sampled draft, the default from
  0.11.0, and 3.45 under the fixed draft (one is the anchor), at 7 draft tokens
  ([the sampled draft served](../../findings/2026-11-13-the-sampled-draft-served-on-the-27b.md),
  [DFlash2 sampled acceptance](../../findings/2026-09-24-dflash2-sampled-acceptance.md)).
  Code is more predictable than prose: on one RTX 5090 lane the 27B ran about
  491 tokens per second on predictable code and 167 on free prose, 367 as the
  geometric mean over eight coding prompts, and 1,065 aggregate across eight
  lanes
  ([retained slots on the host](../../findings/2026-09-28-retained-slots-on-the-host.md),
  [upstream quick wins](../../findings/2026-09-24-upstream-quick-wins-ab.md)).
- **How many drafts.** Seven is what the built-in profiles use but
  `rtx5090`, which from 0.11.0 drafts up to 15
  ([past seven drafts](#past-seven-drafts-two-round-shapes)): served on the
  RTX 5090 against 7, structured output decoded 25 to 64% faster and no
  free-text cell fell more than 2% below
  ([finding](../../findings/2026-11-20-the-27bs-draft-window-of-15-served-on-the-rtx-5090.md)).
  On the RTX PRO 6000 seven beat four in every cell measured but one, and was
  1.6 to 3.3 times speculation off
  ([finding](../../findings/2026-10-18-rtxpro6000-profile-measured.md)). A
  load can name up to 15 on any card; off the RTX 5090 that is unmeasured for
  speed, so every other profile keeps 7.
- **The proposal head.** The drafter proposes with the target's own output head
  by default (`--spec-draft-head full`). `shortlist` uses a smaller 4-bit head
  over the 131,072 most frequent tokens, at +356 MB of VRAM; it accepted 7
  percent fewer tokens a round and was slower overall, so it is opt-in
  ([upstream quick wins](../../findings/2026-09-24-upstream-quick-wins-ab.md)).
- **Variants.** The abliterated 27B uses the same drafter. It was fitted to the
  default weights, and since the target verifies every draft, a mismatch lowers
  acceptance and nothing else.

### Past seven drafts: two round shapes

From 0.11.0, `--spec-draft-tokens` takes 8 to 15 with DFlash2 (the 0.10
releases refuse past 7). The number is still the widest window of the load,
and such a load holds two **round shapes**, on one set of buffers:

- the **base shape**, 8 columns a lane (the anchor and 7 drafts) at every
  lane count: the round of a load at 7 drafts, token for token;
- the **extended shape**, the window + 1 columns a lane (16 at 15 drafts), at
  up to 4 lanes, so that a round never carries more than the 64 columns of 8
  lanes of 8.

A verify round takes one shape for all its lanes, chosen before it launches
from what each lane's own earlier rounds did, the **shape rule**:

- a request starts at the base shape;
- after a base round that drafted all 7 and had all 7 taken
  (`--spec-shape-enter-accepted`, 7), it asks for the extended shape;
- after an extended round, it keeps asking while that round had at least 6
  drafts taken (`--spec-shape-stay-accepted`, 6), and asks for the base shape
  again when it had fewer;
- a base round its token budget cut short changes nothing.

The **shape policy** (`--spec-shape-policy`) then turns the lanes' asks into
each lane's shape. A request alone runs at the shape it asks, and a round of 8
lanes at the base shape, under every policy. `aggregate` and `all` give a
round one shape for all its lanes, at the base shape from 5 lanes on; at 2 to 4
lanes they differ in who pays the wider round:

- `aggregate`, the default, runs the round extended when the drafts its asking
  requests are expected to take past position 7 exceed what the wider round
  costs every request in it: the extended round's period over the base
  round's at that lane count (`--spec-shape-period-ratios`, 1.29 at two
  lanes), less one, times the tokens every request is expected to take a
  round. Each request's expectations come from its own last 16 verify rounds
  (`--spec-shape-estimate-rounds`); a request that asks with no extended round
  among them is granted one, so that its estimate has something to read. A
  request that does not ask then pays for its neighbour when the round as a
  whole gains. Served on the RTX 5090 at two lanes at 15 drafts, a greedy
  JSON answer beside a conversation at temperature 0.6 (FP8 exception
  weights, the context pinned at 262,144) committed 17.8% more tokens a
  second together than under `all`, the JSON answer 29.1% more and the
  conversation 12.3% fewer; a moderately predictable answer (a long JSON
  array) beside the same conversation, 1.2% more, inside the run's spread,
  with 27.6% of its rounds extended
  ([finding](../../findings/2026-11-20-the-27bs-draft-window-of-15-served-on-the-rtx-5090.md)).
- `all` runs it extended only when every lane in it asks, so no request ever
  pays the wider round for another: beside a conversation the JSON answer
  then goes extended only in the rounds where the conversation asks too
  (10.6% of that pair's rounds), and the conversation keeps its speed.
- `mixed` (from 0.11.0) grants the extended shape request by request, so a
  round may carry both shapes, a **mixed round**: at 2 to 7 lanes the asking
  requests are taken one by one, the highest expected extra tokens first (at a
  tie the one refused longest), and each is granted when the drafts it is
  expected to take past position 7 exceed what its 8 more columns cost every
  request in the round: the round's period with it extended over the round's
  without it, less one, times the tokens every request is expected to take. A
  request that asks with no extended round among its last rounds comes after
  the others and is granted whenever the columns allow. A round never passes
  64 columns, so it holds one to three extended requests beside one to six at
  the base shape (or every one extended at 2 to 4 lanes). A JSON answer
  beside three conversations can then draft 15 while they draft 7. Mixed rounds
  run only at 15 drafts, on either artifact (the trellis artifact the 16 GB and
  24 GB profiles load too, [Built-in profiles](profiles.md#rtx-24gb)): any
  other load runs `mixed` as `aggregate`, and its load log says so
  (`shape_policy=aggregate shape_policy_given=mixed`). A request held to a grammar is granted as any
  other: the grammar's check cuts its run at its first miss at the shape it
  was granted, in a mixed round too. Every profile gives `aggregate`;
  `mixed` is opt-in until its served gate passes. The grant stops at the last
  mix whose graphs the load holds, and a round at a lane count where it holds
  none takes the uniform shape `aggregate` gives it; a load whose pool holds
  no mix at all (two lanes without a retained device slot,
  [the configuration reference](../configuration/context-and-speculation.md))
  runs `aggregate`, and its load log says so and why.

The ratios (`--spec-shape-period-ratios`, the **cost table**: at 1 to 7 lanes,
the round's period with 1, 2, ... of them extended over the base round's) are
each artifact's own (`auto`) unless you name a table (four ratios alone take
the artifact's mixed entries beside them). The trellis artifact's keeps the
NVFP4 artifact's uniform entries beside derived mixed ones, provisional until
a run on it
([the configuration reference](../configuration/context-and-speculation.md)).
The NVFP4 artifact's are
the RTX 5090's: the uniform entries measured with the extended round's cost
levers on, at about 3.5-5K tokens of context, 1.128, 1.29, 1.363 and 1.344 at
1 to 4 lanes; of the mixed entries 1+1, 1+3, 2+2 and 1+5 measured (1.155,
1.138, 1.222, 1.093) and the other eight derived from its base rounds of the
same columns (the three of 64 columns, where no base round was measured, one
8-column step past 56), all pending a measurement of this build. The ratio grows with
context, and at 3 and 4 lanes the card's power limit lowers the extended
round's clock, so 3 lanes reads above 4. On other cards they are unmeasured,
and the window is provisional. A request held to a
grammar (`response_format`, a held tool's
calls) asks as any other does: its rounds take the shape the rule and the
policy give them, the grammar's check cuts each run at its first miss at that
shape, and the rule and the policy count the drafts the model took, not the
ones the check kept. Text the drafter predicts far ahead (a JSON answer, a
long tool call's arguments) goes extended and stays there; free text rarely
takes all 7, so it stays at the base shape and decodes as at 7 drafts (a
greedy answer can differ at a near-tie where the shape changes, below). A
request evicted and restored keeps what it asked; its recent rounds for
`aggregate` start again. `/metrics` counts the rounds at each shape
(`ignis_speculative_shape_rounds_total`: `base`, `extended`, and `mixed` for
the lanes of a round that ran at both, each counted once, at one of the three), the rounds of two requests or more
some but not all of which asked the extended shape, at whichever shape they
ran (`ignis_speculative_shape_split_rounds_total`, counted on each request's
rounds as the others are), and, per position up to the window, the drafts
proposed and taken.

A load past 7 drafts costs what [the configuration reference](../configuration/context-and-speculation.md)
states below 8 lanes (at 8 lanes, a few hundred bytes); the VRAM plan holds
both shapes, and the extended shape's captured graphs hold about 20 MiB
outside it. The measurement switch `IGNIS_ROUND_SHAPE=base|extended`
([Testing](../contributing/testing.md)) forces every round to one shape over
the rule and the policy. A load at 7 drafts or fewer has the one shape it had. The MTP head
keeps 0 to 7.

An extended round attends in chunks of 8 columns (6 over `bf16` KV), each
reading what the one before appended: on ignis's own route where the load's
rounds take it (an RTX 50 or RTX PRO card over `hq-e8-2b`), on the vendored
chunked route elsewhere, the same bytes either way.

The two shapes compute a column's logits up to rounding, not bit for bit:
past 8 columns a lane the round takes other routes, and the extended round is
held to the base round's numerical class, not to its bytes
([ADR 0104](../../adr/0104-the-27bs-dflash2-verify-round-goes-past-8-columns-a-lane-and-is-held-to-its-base-rounds-numerical-class.md)),
and so is a mixed round (from 0.11.0), whose lanes take both shapes
([ADR 0106](../../adr/0106-the-27bs-dflash2-verify-round-may-carry-lanes-of-both-shapes-and-a-mixed-round-is-held-to-its-base-rounds-numerical-class.md)).
So a greedy answer can differ at a near-tie where a round of one shape runs in
place of the other, as it can between speculation on and off.

## MTP, for the 27B

Every 27B artifact carries the checkpoint's own **multi-token-prediction
head**: one gated attention layer and an MLP that take the model's hidden
state at a position and the token after it, and draft the token after that,
scored by the model's own output head. `--spec-backend mtp` loads it as the
27B's drafter, on every 27B artifact (NVFP4, the abliterated one, the 16 GB
one), in place of DFlash2. A round chains the head: each draft feeds the next
one, up to the draft window.

- **The draft window.** `--spec-draft-tokens` 0 to 7, fixed for the life of
  the load; 7 when unnamed, the window that committed the most tokens per
  second on the 16 GB artifact in a 13 GiB budget, over recorded agent turns.
  There windows 5, 6 and 7 came within 0.4% of each other; on the NVFP4
  artifact under `rtx5090` window 5 is estimated, from the measured round
  costs, to commit about 3.5% more than 7, a gap the size of the costs' own
  spread between launches
  ([finding](../../findings/2026-11-15-the-27bs-mtp-head-measured-on-the-rtx-5090.md)).
  A window of 0 loads no speculation: the same load as
  `--spec-backend off`, with nothing of the head bound and the same VRAM plan.
  A window a profile gives beside its own DFlash2 (`rtx5090`'s 15 from
  0.11.0, every other built-in profile's 7 but `rtx-12gb`'s, which is the
  head's own) is DFlash2's: `--spec-backend mtp` over it takes the head's own
  default.
- **Acceptance.** On a replay of 200 real agent turns, sampled at temperature
  1, the head committed 3.9 tokens a round at 7 draft tokens on both artifacts
  (one is the anchor): its first draft was taken 74% of the time, its seventh
  20%. DFlash2 commits 3.83 on the same turns under its sampled draft. Its
  acceptance does not fall with the context: 3.6 to 4.3 tokens a round from
  under 16K tokens to past 96K.
- **What it saves.** DFlash2's weights, its 40 MiB window in every lane and
  every retained slot, its round's scratch and the taps in every prefill chunk.
  The head's own weights are about 0.45 GB. Measured on the RTX 5090, the
  plan of an MTP load holds 1.01 GB less than the same DFlash2 load outside
  the KV pool on the NVFP4 artifact under `rtx5090` (8 lanes) and 0.80 GB less
  on the 16 GB artifact under `rtx-16gb` (2 lanes). In a 13 GiB budget, a
  16 GB card less 3 GiB of desktop, the 16 GB artifact loads 131,072 tokens
  of context under MTP where DFlash2 adapts to 65,536
  ([the VRAM plan](vram-plan.md#what-the-27bs-drafter-holds)).
- **What it buys.** Against the same load at window 0, decode ran 1.3 to 2.3
  times as fast on the NVFP4 artifact and 1.3 to 2.9 times on the 16 GB one,
  at one lane and at the load's lanes, on agent requests and on 32K-token
  prompts, greedy and sampled; least where several lanes share long contexts.
- **What it costs.** Its attention state spans the whole context, as the head
  was trained: one more attention layer in every KV page, so the same bytes
  hold about 6% fewer tokens. Each sequence keeps one more vector, the model's
  last hidden state, so that its next step can build the head's row. Under
  `hq-e8-2b` the head's layer takes its share of the residual window. A
  prefill runs the head over every position of the prompt: on the RTX 5090
  the NVFP4 artifact's time to first token at 8K and 32K tokens is 13% longer
  than at window 0, and the 16 GB artifact's prefill takes 6% longer. A round chains the head
  once a draft, each step the head's layer
  and the model's whole output head: 1.2 to 1.4 ms a step on the NVFP4
  artifact at one lane (2.6 ms at eight lanes 64K tokens deep), 0.8 to 1.1 ms
  on the 16 GB one, so at 7 draft tokens the head takes 28 to 40% of a round.
  Each holder of a
  shared prefix (the request that published it and every request that claims
  it) keeps its own copy of the prefix's last page, because the head's row at
  the prefix's last position is built from each holder's own next token; the
  VRAM plan holds a page a lane and a page a retained slot for those copies
  beside one context, and admission never hands them out
  ([the VRAM plan](vram-plan.md)).
- **When to choose it.** When DFlash2's memory is what your context lacks:
  on a 16 GB card, or on any card whose plan adapts the context down under
  DFlash2. On a 12 GB card `rtx-12gb` chooses it for you. The head commits as many tokens a round as DFlash2 on agent
  traffic and far more tokens a second than no speculation, for about 0.8 to
  1 GB less than DFlash2. Where DFlash2 fits at the context you need, it stays
  the default. On the 12 GB class's artifact, `qwen3.8-27b-12g`, held to a 12 GiB
  budget on the RTX 5090, the two were measured side by side: DFlash2 decodes
  409, 348 and 321 tokens a second at 8,192, 65,536 and 130,560 tokens of
  context against the head's 294, 251 and 228, and the context is 131,072 under
  either, with a KV pool of 150,720 tokens against 214,144.
- **One rule.** The head drafts the fixed draft at every temperature: a
  request that samples takes each draft with the model's probability of it.
  DFlash2's own settings are refused by name on an MTP load: the shortlist
  head, `--spec-draft-rule sampled`, and the drafter's temperature factor and
  top-p ([context, KV and speculation](../configuration/context-and-speculation.md)).
- **On an RTX 30 or RTX 40 card.** The head drafts there as on Blackwell, on
  the 16 GB entries those cards load. Its five matrices multiply on ignis's own
  route, in the forms those cards run at full rate: on the CUDA cores for a
  round's few rows, on the integer MMA for a prefill's and a wide commit's,
  each activation carried in 16-bit fixed point, so the head computes what it
  computes on Blackwell within the rounding of its outputs, and the VRAM plan
  is the same. The load's `ignis.runtime.decode_routes` line names the route as
  `mtp_head_linear`: `ours` there, `vendored` on an RTX 50 or RTX PRO card,
  `none` on a load without the head.
- **Images.** The head drafts after an image, with vision on as every
  built-in profile turns it on, under either vision residency. Where the next
  token is an image placeholder, the head takes the image's row the model
  itself took there, and it reads the image's positions as the model does, so
  the text after the image is drafted as text is.
- **Everything else is a DFlash2 load's.** Prompt reuse, retained slots,
  KV-RAM and KV-disk carry the head's state with the sequence, and a blob
  written by one drafter is refused by another, by name, and prefilled again.
  The load's log names the drafter and the window in `ignis.model.loaded`'s
  `speculation` (`mtp draft_tokens=7`, or `off (mtp draft_tokens=0)`); the
  request log's `spec.*` fields and the `ignis_speculative_*` counters mean
  what they mean under DFlash2.

## MTP, for Flash-Next

Flash-Next has its own multi-token-prediction head, a one-layer drafter that
combines the trunk's state with the next token's embedding and drafts the token
after it. It comes in a small companion container beside the artifact, and
`--spec-backend mtp` without it refuses the start
([models and artifacts](models-and-artifacts.md)). The head accepts 80 to 83
percent of the first four drafts on the engine's own states
([MTP phase A](../../findings/2026-10-06-flash-next-mtp-phase-a.md)).

It costs about 1.1 GB of VRAM, taken from the expert cache rather than the
context, whatever the number of drafts.

From 0.11.0, which gives Flash-Next images, MTP stays on for prompts with
images, with nothing to set. Where the next token
is an image placeholder, the head takes the image's row that the model itself
took there, and it reads the image's positions as the model does. Greedy output
with MTP on is the same as with it off, up to near-ties, as for text.

## The row budget

A verify round carries `lanes x (drafts + 1)` **rows**. More rows cost more per
round, so Flash-Next MTP caps the rows a round may take across all lanes
(`--spec-draft-rows`; `0`, the default, means 8). Each lane drafts

```text
min(draft tokens, rows / lanes - 1)
```

so a budget of 8 with 3 draft tokens gives 3 drafts at one or two active lanes,
1 at three or four, and none from five lanes on. Drafting is spent on the lanes
that are few enough to afford it, and a busy server falls back to plain rounds.

Eight is the default because up to 8 rows every linear layer takes its
single-token decode route. Past 8 they switch to tensor-core routes whose sums
run in a different order, so the output is held numerically equivalent to
speculation off rather than token for token: it can differ where two tokens are
nearly tied ([ADR 0056](../../adr/0056-speculation-past-eight-verify-rows-is-numerically-equivalent.md)).
It also costs: on the RTX PRO 6000 a round past 8 rows took 64 to 80 ms against
about 19.5 ms at 8, so every wider budget ran 0.42 to 0.84 times the default
([finding](../../findings/2026-10-18-verify-rows-on-the-rtx-pro-6000.md)). A budget
past 8 also reserves the wider rounds' VRAM, taken from the expert cache.

## Why MTP is off on the RTX 5090 and on for the RTX PRO 6000

The two cards differ in where Flash-Next's experts are.

On a **32 GB RTX 5090** the experts are in host RAM and the VRAM cache holds
only part of them. Decode there is bound by copying the experts a token needs
across PCIe (Gen 3 on the measured host). A verify round selects experts for
every column, so it copies more of them, and speculation barely reduces the
copies per token. Measured on 2026-10-07: one lane runs 1.02 to 1.30 times
faster, and two and three lanes run 7 to 15 percent slower at the default row
budget (1 to 2 percent slower at `--spec-draft-rows 3`, which drafts at one lane
only), before counting the 1.1 GB the head takes from the cache
([finding](../../findings/2026-10-07-flash-next-mtp-speculation-is-pcie-bound.md)).
So MTP is off by default there, and worth turning on only for one user or one
agent at a time (`--spec-backend mtp --spec-draft-rows 3`, or one lane).

On a **96 GB RTX PRO 6000** the cache holds nearly every expert, few cross PCIe,
and decode is limited by streaming weights as the 27B's is. There speculation
pays wherever it drafts, and the `rtxpro6000` profile turns it on with 3 draft
tokens. Measured on a Server Edition, with every lane decoding: one lane 1.85
times speculation off on code and 1.44 times on prose, two lanes 1.24 to 1.53,
four 1.12 to 1.19, and eight 0.97 to 0.98 (no drafts at that width, only the
head's fixed cost). With simulated agents, whose lanes wait on tools between
turns, four agents generated 1.21 times the tokens and eight 1.05 times
([finding](../../findings/2026-10-18-rtxpro6000-profile-measured.md)).

## Where the knobs are

`--spec-backend` (the 27B's `dflash2` or `mtp`, Flash-Next's `mtp`), `--spec-draft-tokens`, `--spec-draft-rows`,
`--spec-draft-head`, `--spec-draft-rule` and the sampled draft's
`--spec-drafter-temperature-factor` and `--spec-drafter-top-p` are in
[context, KV and speculation](../configuration/context-and-speculation.md);
the choice of when to turn Flash-Next's MTP on is worked through in
[Flash-Next speculation](../configuration/flash-next-mtp.md). The profiles set
them per card ([profiles](profiles.md)).
