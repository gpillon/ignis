---
title: "Speculative decoding"
description: "How DFlash2 and Flash-Next's MTP head draft tokens that the model then verifies, what acceptance means, how the row budget is shared across lanes, and why MTP is off on the RTX 5090."
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
   lane's **committed run**, between 1 and `k + 1` tokens.
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

## DFlash2, for the 27B

DFlash2 is a small drafter grafted into the 27B's artifact: five layers that
read the target's own hidden states over a sliding window of the last 2,048
positions. It is loaded when `--spec-backend dflash2` is set, together with
`--spec-draft-tokens` (1 to 7), and costs no VRAM when speculation is off. The
reference also keeps a rewrite checkpoint of the drafter's window, restored
when a turn rewrites text past its chat template's rewrite boundary; ignis
never read one and retired it (snapshot format 5), because a prompt checkpoint
taken at the opener already carries the window as it stood there.

- **Acceptance.** On a replay of 200 real agent turns it averaged 3.46 tokens a
  round (one is the anchor), at 7 draft tokens
  ([DFlash2 sampled acceptance](../../findings/2026-09-24-dflash2-sampled-acceptance.md)).
  Code is more predictable than prose: on one RTX 5090 lane the 27B ran about
  491 tokens per second on predictable code and 167 on free prose, 367 as the
  geometric mean over eight coding prompts, and 1,065 aggregate across eight
  lanes
  ([retained slots on the host](../../findings/2026-09-28-retained-slots-on-the-host.md),
  [upstream quick wins](../../findings/2026-09-24-upstream-quick-wins-ab.md)).
- **How many drafts.** Seven is what the built-in profiles use. On the RTX PRO
  6000 it beat four in every cell measured but one, and was 1.6 to 3.3 times
  speculation off
  ([finding](../../findings/2026-10-18-rtxpro6000-profile-measured.md)).
- **The proposal head.** The drafter proposes with the target's own output head
  by default (`--spec-draft-head full`). `shortlist` uses a smaller 4-bit head
  over the 131,072 most frequent tokens, at +356 MB of VRAM; it accepted 7
  percent fewer tokens a round and was slower overall, so it is opt-in
  ([upstream quick wins](../../findings/2026-09-24-upstream-quick-wins-ab.md)).
- **Variants.** The abliterated 27B uses the same drafter. It was fitted to the
  default weights, and since the target verifies every draft, a mismatch lowers
  acceptance and nothing else.

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

MTP stays on for prompts with images, with nothing to set. Where the next token
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

`--spec-backend`, `--spec-draft-tokens`, `--spec-draft-rows` and
`--spec-draft-head` are in [context, KV and speculation](../configuration/context-and-speculation.md);
the choice of when to turn Flash-Next's MTP on is worked through in
[Flash-Next speculation](../configuration/flash-next-mtp.md). The profiles set
them per card ([profiles](profiles.md)).
