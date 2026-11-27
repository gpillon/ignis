---
title: "Reasoning and the loop guard"
description: "How thinking is bounded and closed, what the server does when a turn ends inside open reasoning, and how a reasoning loop is detected, taken back and nudged."
sidebar:
  order: 6
---

Qwen3.8 thinks before it answers: with thinking on (the default) it writes a
reasoning block, opened by `<think>` and closed by `</think>`, and then the
answer or the tool call. The server separates the two on the closing tag and
returns the reasoning as `reasoning_content`, the answer as `content`. Two
things can go wrong inside the block: it can end without the model ever closing
it, and it can loop. This page explains how the server handles each, and the
budget that bounds both. The fields are in [Prompting](../configuration/prompting.md).

## Thinking and its budget

`--model-enable-thinking` (on by default) and `--model-reasoning-effort` set the
defaults for the `enable_thinking` and `reasoning_effort` a request can override.
An effort the template does not take is rounded up to one it does; for Qwen3.8,
`high` and `max` are `xhigh`.

Reasoning spends the same `max_tokens` as the answer. A block that never closes
would end the request as `length` with no answer, so there is a **thinking
budget** (`--model-thinking-budget`, 32,768 reasoning tokens by default). After
that many reasoning tokens with the block still open, or 2,048 tokens before the
request's `max_tokens` if that comes first, the server forces a close: a sentence
telling the model to answer from what it has, without calling any more tools,
followed by `</think>`. The 2,048 is an answer reserve,
sized from the answers that followed a forced close in measurement. A request's
`thinking_budget` overrides the default, and `reasoning_effort: max` runs
without one.

The wording of the close matters. After an earlier close the model could defer
with another tool call instead of answering, and under greedy sampling the same
turn then repeated: two agent sessions looped on it for 15 and 32 turns. A close
that says to answer, with no more tool calls, ended it
([a close that answers](../../findings/2026-10-01-a-close-that-answers.md)).

## A turn that ends inside its reasoning

The model can also draw its end-of-turn token while its reasoning block is
still open, usually after reasoning at length and writing the tool call it meant
to make inside the reasoning text. The server only reads tool calls from the
content channel, so the client received a turn that was all reasoning with no
call and no answer, and an agent loop stopped
([ADR 0048](../../adr/0048-an-end-of-turn-inside-the-reasoning-closes-it.md)).

Now an end-of-turn drawn inside an open reasoning block becomes `</think>`
instead, and the model goes on to the answer or the call it was writing. This
is decided inside the engine on the host, before the token is fed back to the
model, because on these hybrid models the recurrent state absorbs every token it
is fed and there is no stepping back from it. Details of the rule:

- It applies to a request that starts inside its reasoning, with thinking on.
  It does not apply under `ignore_eos`, or with thinking off.
- Only the stop tokens are redirected. A tool-call token drawn in the reasoning
  stays reasoning text, because reasoning routinely writes about calls it does
  not mean to make.
- There is no switch: no client wants a turn that ends with reasoning and
  nothing else.
- The request log's `reasoning_redirected_at` is the output index of that
  `</think>`, so a turn closed this way can be told from one the model closed
  itself.

This is a deliberate difference from the reference implementation, recorded as
an ignis patch: output differs from it only on a turn that would have ended in
its reasoning.

## Reasoning loops

On long enumerations the 27B sometimes starts to repeat itself inside its
reasoning, and does not stop until the budget closes it. On one long-enumeration
prompt about a third of first turns looped at the model card's sampling, and
the same artifact loops at the same rate on another engine, so the loop is the
model's and not the server's
([reasoning loops are the model's](../../findings/2026-10-12-reasoning-loops-are-the-models.md)).
The loops take several shapes: verbatim text, one token in every slot, counting
(`the 321 AD, the 322 AD, ...`), and templates whose slots vary. A looping turn
wastes up to the whole budget, and an agent that reaches the forced close is
left only the answer reserve.

The server cannot fix the model, but it can notice and step back
([ADR 0052](../../adr/0052-a-reasoning-loop-is-taken-back-and-nudged.md),
[ADR 0053](../../adr/0053-a-loop-is-acted-on-once-it-persists.md)).

### What it detects

A **loop guard** sits between the engine's event stream and the client. It
watches only the reasoning of a request that starts inside its reasoning, in a
separate task that adds nothing to a decode round. Every `check-every`
characters of reasoning (250) it takes the last `window` characters (1,000),
removes code, folds every run of digits to a single digit, and compresses what is
left. A check scores as a loop when that text compresses under `ratio`
(265 thousandths, so to a quarter of its size) and the unmodified text also
compresses under `raw-gate` (300). Each step has a reason: removing code keeps
legitimate code out of the score, folding digits turns a counting or template
loop into a repetition, and the raw gate keeps numeric tables out, where the
digits carry the information.

Compression measures redundancy, and a list, a log or a table is redundant by
nature. What separates a loop from those is time: a loop absorbs the model and
stays, while a list ends with its content. So the guard acts only when a loop
persists: `persist` checks in a row (4) must score as a loop. On recorded runs,
no legitimate agent text stayed over the threshold for more than 6 checks, and
23 of 25 harmful loops stayed for 9 or more.

From 0.11.2 the guard also watches for a **repeat**
([ADR 0053](../../adr/0053-a-loop-is-acted-on-once-it-persists.md)). This
catches a loop whose repeated block is longer than the 1,000-character window.
The window never holds two copies of such a block, so it compresses like prose.
Each time a sentence of the reasoning ends, the guard looks for the same
sentences written twice back to back, compared as written:

- a copy may have one sentence (`repeat-varying`) that the copy before it does
  not;
- a copy must repeat at least `repeat-min` characters of prose verbatim (150);
- a copy can be at most `repeat-max` characters long (16,000).

A line that ends without a full stop, such as a list item, a log line or a
table row, counts for nothing towards the 150. So a listing or a log written out
twice to compare it is not a loop. A repeat is acted on when its second copy
ends, before a third is written.

### What it does

In `rewind` mode, the default, the guard **takes the loop back**:

1. It walks back to the last window that scored clean (compressing to at least
   `clean`, 400 thousandths), moves back `margin` characters and snaps the cut to
   a line break within `snap` characters (300). For a repeat (from 0.11.2) the
   cut is the start of the second copy, so the first copy stays.
2. It cancels the engine request and submits a **continuation**: the same input,
   its prompt extended by the reasoning kept before the cut and a **nudge**, a
   sentence in the model's own voice ("Wait, I'm starting to repeat myself. Let
   me stop and move on with what I already have."), with what is left of the
   generation cap and of the thinking budget.
3. The generation goes on in the same response. The prompt comes back from its
   checkpoint, so a rewind restored a 79K and a 138K-token agent context in
   0.08 to 0.10 s.

After `retries` nudges (2) the next loop closes the reasoning instead, as the
thinking budget would. A second nudge uses a different sentence
(`nudge-again`).

What the client sees: the reasoning carries a tag, `[LOOP DETECTED, REWINDING n
TOKENS / m CHARS · h]`, after the `m` characters taken back, which the
Playground strikes through. The rewound tokens stay in `completion_tokens` and
are also reported as `completion_tokens_details.rewound_tokens`. Reasoning an
agent sends back with its tags is repaired before the next prompt, whatever the
guard's mode, so the model is given back the text it actually saw. A request
with images is never rewound.

`--model-loop-guard-holdback` can hold the last characters of reasoning back
from the client, so that a rewind whose cut falls in them reaches the client
with no tag and none of the loop. The cost is that the client sees the reasoning
that much later.

### Modes and what it will not do

`--model-loop-guard` is `rewind` (act), `log` (only log what a rewind would do)
or `off` (no guard, which is what a measurement runs). Two rules keep the guard
from doing harm:

- A rewind that would keep fewer than `min-kept` characters of reasoning (1) is
  only logged, so a reasoning that opens in a loop is never dropped whole.
- A continuation with fewer than `min-continuation` tokens (512) of `max_tokens`
  left after the cut is only logged.

### How well it works, and what it misses

On the recorded runs it caught 25 of 25 harmful loops on the prompt it was
tuned on, and 6 of 7 on 40 fresh seeds, with no fire on the reasoning of real
opencode and qwen-code sessions
([the guard](../../findings/2026-10-14-reasoning-loop-guard.md),
[a loop persists](../../findings/2026-10-16-a-loop-persists.md)). The costs are
a fire a median 500 characters later than the first-check rule would have, and
loops the model leaves by itself within four checks are no longer touched.

The repeat signal (from 0.11.2) was measured offline
([a loop longer than the window](../../findings/2026-11-27-a-loop-longer-than-the-window-repeats-verbatim.md)):

- verbatim loops with blocks of 250 to 8,000 characters are caught at the end
  of their second copy;
- on the recorded agent sessions it found 21 parts that loop verbatim, which
  the compression score never acted on, among them 8 consecutive turns of one
  session;
- once those are counted as loops (one reader's call, 8 of them made after the
  signal found them), it fired on no legitimate part, and the prompt above
  stays at 25 of 25.

It does not see every loop. Indecision loops, whose text varies between repeats
(a decision sentence between reconsiderations), do not compress, and the only
harmful agentic loop observed was of that kind. Loops that reword their copies,
changing whole passages between repeats, are missed by both signals. Two copies
of a paragraph that the model would have left by itself are taken back too, and
so is any block of prose written twice in a row, a checklist whose items share
150 characters of it among them. The
margins between the thresholds are thin and come from one prompt and two texts;
a false rewind or a missed loop found later reopens the defaults.

### Tuning it

All the guard's fields can be changed on a running server with `PATCH
/v1/config`, without a reload; they are listed in
[Prompting](../configuration/prompting.md). `/metrics` counts what the guard does:
loops by outcome (`ignis_loop_guard_loops_total`), requests watched and looped,
tokens rewound, and each request's lowest score
([telemetry](../guides/telemetry.md)). The guard runs on chat completions and on
`/v1/responses`; the [Responses API](../api/responses.md) page covers the
WebSocket.
