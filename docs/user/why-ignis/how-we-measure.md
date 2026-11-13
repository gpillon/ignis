---
title: "How these numbers are made"
description: "What a finding is, how gates are registered before a run, why greedy and sampled speculation differ, and which negative results are kept."
sidebar:
  order: 17
---

Every figure on these pages links to a **finding** or an **ADR**. This page
says what those are, so you can tell what a number promises and what it does
not. The short version: a number is a measurement with its conditions, written
down on the day it was taken. It is not a property of ignis.

## What a finding is

A finding is a dated file in [`docs/findings`](../../findings/README.md). Its
header says what kind it is (an experiment, a discovery or research), the date it
was observed and last verified, its scope, whether it is current or superseded,
and which later finding replaces it. The body keeps the setup next to the result:
the card, the build, the context, the sampling, how many runs. Statements are
labelled by how they are known. **Observed** is something a run showed.
**Inferred** is something concluded from it, and the label says so. Findings are
snapshots: several figures come from builds before the current defaults (a
262,144-token context, a 1,024-token chunk, no vision), and each finding states
which.

The practical rule is to read a number together with its condition. "367 tok/s"
is one lane, greedy, on eight coding prompts. "99.8%" is answer mass over 144
authored decisions. The tables on these pages carry that condition in their
second column for this reason.

## Gates are written before the run

When a result decides whether something ships, the floor is written first. The
`locate` primitive was held to an 85% floor for logs, set in a spec before any
run; it scored 21 of 23, and the same registered rules failed it on prose at 1M
tokens (7 of 12) and said so. The pointing sets for `point` and `box` had floors
that were fixed before the run, and the head box stayed a non-default because
it ended seven boxes short of the chain on one set. Where a threshold was chosen after seeing data,
the finding says that too, as in the A/B pairs of the workspace-zeroing
measurement, where the discard rule was set partly after the first look.

## Greedy coding and sampled agent turns

Two kinds of traffic give two speculation numbers, and both are true. On eight
greedy coding prompts DFlash2 commits 5.43 tokens a round, which is the 367
tok/s at one lane and the 1,065 tok/s across eight
([Four upstream kernel ports](../../findings/2026-09-24-upstream-quick-wins-ab.md)).
On 200 recorded agent turns replayed with the sampling an agent uses (T=1.0,
top_p 0.95, top_k 20) it commits 3.46 tokens a round under the fixed draft
([DFlash2 sampled acceptance](../../findings/2026-09-24-dflash2-sampled-acceptance.md))
and 3.83 under the sampled draft, the default from 0.11.0
([The sampled draft served](../../findings/2026-11-13-the-sampled-draft-served-on-the-27b.md)).
The gap is the predictability of the text and the sampling, not a regression.
Expect the first figure on code written from a clear instruction and the second
on an agent's mixed output.

## Other people's numbers

The README sets ignis beside figures for other engines. Those are other people's
published runs on other hardware and other builds. They are not in the findings,
nobody here reproduced them, and they should be read as an order of magnitude.
The comparisons ignis did run are the live/live ones against the reference
engine, on the same card and artifact in the same session
([hq vs BF16, live/live](../../findings/2026-09-13-hq-vs-bf16-live-live.md)).

## Negative results are kept

A finding that says no is as durable as one that says yes. A few that shaped
what ignis does:

- **Copy drafting.** Replacing DFlash2's proposals with prompt-lookup copies
  buys 0-8% on real agent traffic, because thinking dominates the output and
  DFlash2 already rides the copyable spans. The finding sets its bar for
  building one at +8% at one lane ([copy drafting](../../findings/2026-09-24-copy-drafting-on-agent-traces.md)).
- **Locating by attention alone.** Attention finds the paraphrased line and
  loses the quoted one; by the rule written before the run the answer was
  no-go, and the shipped `locate` uses a different route
  ([locate is a no-go](../../findings/2026-09-26-locate-attention-no-go.md)).
- **The shortlist draft head.** It makes a round about 0.55 ms cheaper but loses
  more acceptance than it saves, so the full head is the default and the
  shortlist is opt-in ([`spec.draft_head`](../reference/configuration.md#specdraft_head)).
- **FP8 for the exception tensors.** It frees 0.53 GB and adds 3% to cold prefill
  with KLD unchanged, but two decision fixtures move, so it is opt-in
  ([`model.exception_weights`](../reference/configuration.md#modelexception_weights);
  [the finding](../../findings/2026-10-21-the-bf16-exception-tensors-in-fp8.md)).

## Where the kernels come from

Each kernel in ignis is one of three things
([ADR 0051](../../adr/0051-kernel-code-is-ours-adapted-or-vendored.md)): written
here, adapted from a named upstream with its repository, commit, path and licence
in the file header, or vendored as a verbatim copy tracked by a manifest. A
vendored file is not hand-edited by default. Two exceptions are written down:
it may be replaced when a measurement shows it is the bottleneck
([ADR 0031](../../adr/0031-vendored-kernel-bottleneck-exemption.md)), as the
drafter's top-k was, and it may carry a recorded, tested patch when it has a
demonstrated correctness bug
([ADR 0037](../../adr/0037-a-vendored-file-may-carry-a-correctness-patch.md)).
Only permissively licensed code is copied.

## Read more

- [The findings index](../../findings/README.md): the current findings by topic, with the archive of the superseded and closed ones beside it.
- [One round for every lane](decode-round.md) and [A drafter inside the artifact](dflash2.md): the speculation numbers in context.
- [Reasoning that finishes](reasoning-that-finishes.md): a finding set with in-sample and held-out figures side by side.
