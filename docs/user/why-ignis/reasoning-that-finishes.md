---
title: "Reasoning that finishes"
description: "A reasoning loop is caught while it streams, cut back and nudged in the same response; a turn that ends inside its thinking is closed, not returned empty."
sidebar:
  order: 10
---

An agent turn that thinks for 20,000 tokens and then repeats one paragraph
forever costs a card slot and gives the agent nothing. ignis watches the
reasoning as it is written and acts on three failures: a **loop** is cut back
and nudged inside the same response, a turn that **ends inside its own
thinking** is closed so its tool call reaches the client, and a **thinking
budget** closes with a sentence that makes the model answer. None of this makes
the model loop less. It keeps a loop from ending the turn.

## How it works

The loop guard scores the reasoning as it streams. A window of 1,000
characters is normalized (digits are folded, so log lines with different
numbers match) and compressed; a window that compresses far below the norm is
repetition. A list, a log or a table is redundant by nature, so one low score
is not enough: the guard acts only when 4 checks in a row score under the
ratio. A loop whose repeated block is longer than the window compresses like
prose, so from 0.11.2 the guard also looks for the same sentences written twice
back to back, and cuts such a repeat at the start of its second copy. Otherwise it
cuts the reasoning back to the last clean window. Then it appends a
nudge in the model's own voice, and the generation goes on in the same
response. The client sees a tag where the loop was and `rewound_tokens` in
`usage`; the reasoning it sends back is repaired to what the model saw, so the
next turn renders token for token. The rewind resumes from the checkpoint the
request published at its generation opener and prefills only the kept text plus
the nudge, so its cost does not grow with the prompt. After two nudges the next
loop closes the reasoning.

The second failure is quieter. The model can draw its end-of-turn token while
the `<think>` block is still open, usually after writing the tool call inside
the reasoning. The server reads tool calls only from the content channel, so the
client would get an all-reasoning turn and the agent loop would stop. The leaf
holds every successor token on the host before feeding it back, so a stop token
drawn inside an open block is replaced by `</think>` there, and the model
carries on into its answer.

The third is the budget. After `model.thinking_budget` reasoning tokens with
the block open, or 2,048 tokens before `max_tokens`, the server forces a close:
a sentence telling the model to answer from what it has, with no further tool
call, then `</think>`. The wording was fixed after a measured failure: the first
close let the model defer with another tool call, and under greedy sampling the
same turn then repeated.

The mechanics, the modes and the tuning are in
[Reasoning and the loop guard](../concepts/reasoning.md).

## The numbers

| Figure | Measured on | Source |
|---|---|---|
| About a third of first turns loop on a long-enumeration prompt at the model card's sampling; the same with YaRN, speculation or the compressed KV off, and the same on the reference engine with the same weights. `presence_penalty` 1.5 gives 2/8 against 4/8 on the same seeds | 27B NVFP4, the Playground's defaults, 8 seeds per arm | [Reasoning loops are the model's](../../findings/2026-10-12-reasoning-loops-are-the-models.md) |
| 25 of 25 harmful loops and 34 of 38 acute onsets caught | recorded runs of the prompt the thresholds were tuned on (in-sample) | [A loop persists](../../findings/2026-10-16-a-loop-persists.md) |
| 6 of 7 harmful loops caught on 40 fresh seeds; the seventh is a numeric loop the raw gate misses, and 2 more harmful runs were long legitimate planning | held-out run, guard in `log` mode, scored offline | same |
| 0 fires on 12,936 recorded agent reasoning parts (5,654 long enough to score); the first rule fired twice. 15 of those parts later proved to be verbatim loops the compression score never acted on | recorded sessions that also informed the thresholds | same, and the next row |
| From 0.11.2, verbatim loops with blocks of 250 to 8,000 characters caught at their second copy. 21 looping agent parts found that compression never acted on, and 0 fires on the legitimate ones once those are read as loops (one reader's reading). 7 of one session's 8 looping turns caught before a third copy | recorded Rome runs and agent sessions, scored offline | [A loop longer than the window](../../findings/2026-11-27-a-loop-longer-than-the-window-repeats-verbatim.md) |
| A rewind restores a 79K-138K-token agent prompt in 0.08-0.10 s with little kept, 1.276 s with about 3,645 tokens kept; 9 of 16 runs escaped for good after the first nudge | 27B, long agent prompts | [Reasoning loop guard](../../findings/2026-10-14-reasoning-loop-guard.md) |
| An end of turn inside open reasoning: 1 of 115 Flash-Next turns, 0 of 413 on the 27B | one agent session database | [ADR 0048](../../adr/0048-an-end-of-turn-inside-the-reasoning-closes-it.md) |
| The old close answered 2 of 8 sampled runs and always deferred under greedy; the new wording answered both captured turns and left no short deferral in 8 sampled runs | two captured agent turns, 8 sampled reruns | [A close that answers](../../findings/2026-10-01-a-close-that-answers.md) |

:::caution[Recall is incomplete, and the margins are thin]
The 25 of 25 is in-sample; the held-out figure is 6 of 7. "No fire on real
sessions" holds on the sessions the thresholds were tuned against. Loops where
the text varies between repeats, and slow numeric templates, do not compress
and are missed; so are loops that reword whole passages between their
copies. A false rewind found later reopens the defaults
([ADR 0052](../../adr/0052-a-reasoning-loop-is-taken-back-and-nudged.md)).
:::

## See it, control it

- **The guard**: [`model.loop_guard`](../reference/configuration.md#modelloop_guard)
  (`off`, `log` or `rewind`; `rewind` by default), tuned by
  [`model.loop_guard_ratio`](../reference/configuration.md#modelloop_guard_ratio),
  [`model.loop_guard_persist`](../reference/configuration.md#modelloop_guard_persist),
  [`model.loop_guard_min_kept`](../reference/configuration.md#modelloop_guard_min_kept),
  [`model.loop_guard_retries`](../reference/configuration.md#modelloop_guard_retries),
  the repeat signal's [`model.loop_guard_repeat_min`](../reference/configuration.md#modelloop_guard_repeat_min)
  and its three neighbours (from 0.11.2),
  and the two nudges, [`model.loop_guard_nudge`](../reference/configuration.md#modelloop_guard_nudge)
  and [`model.loop_guard_nudge_again`](../reference/configuration.md#modelloop_guard_nudge_again).
  All are patchable live.
- **The budget**: [`model.thinking_budget`](../reference/configuration.md#modelthinking_budget),
  or `thinking_budget` on a request; `reasoning_effort: max` runs without one.
- **In the response**: `usage.completion_tokens_details.rewound_tokens`
  (`output_tokens_details` on `/v1/responses`). The `ignis.request.done` line
  carries `reasoning_redirected_at` when an end of turn was redirected.
- **Metrics**: `ignis_loop_guard_loops_total` (by action),
  `ignis_loop_guard_rewound_tokens_total`, `ignis_loop_guard_min_ratio` and
  `ignis_thinking_forced_closes_total`, in the [metrics](../reference/metrics.md).
- **Playground**: the [chat](../playground/chat.md) strikes through the tokens a
  rewind took back.

## Limits

- The loop is the model's, not the engine's. The guard mitigates; it does not
  remove looping.
- A request with images is never rewound, and `/v1/decide` has no reasoning to
  guard.
- The end-of-turn redirect applies only to a request that starts inside its
  reasoning with thinking on, and not under `ignore_eos`. There is no switch.
- A guard that fires later than the first rule did (a median of about 500
  characters) is the price of not touching loops that end by themselves.

## Read more

- [Reasoning and the loop guard](../concepts/reasoning.md): detection, action, modes and tuning.
- [ADR 0052](../../adr/0052-a-reasoning-loop-is-taken-back-and-nudged.md): why a loop is taken back, and why it must persist first.
- [System One decisions, without decoding](decisions.md): the route that never reasons at all.
