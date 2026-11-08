---
status: accepted
amends: [ADR-0052]
---

# ADR 0053 — a loop is acted on once it persists, and a rewind keeps something

## Status

Accepted (2026-10-16, owner: "portarlo su main" after the held-out run and the live A/B).
Proposed the same day. Amends [ADR 0052](0052-a-reasoning-loop-is-taken-back-and-nudged.md)'s
detection and action; the rest of 0052 stands. The numbers are in the finding
[`docs/findings/2026-10-16-a-loop-persists.md`](../findings/2026-10-16-a-loop-persists.md). ADR
0052 says a false rewind found later reopens its default: two were found on agentic reasoning.

## Context

ADR 0052 rewinds at the first check whose normalized window compresses under 0.20. On the owner's
agentic reasoning (opencode and qwen-code) it fired twice, both times on legitimate work:
- a per-ticket checklist, which the rewind struck with three quarters of the plan ahead;
- a server log transcribed from the user's message. The reasoning opened with it, no check before
  the fire scored clean, and the cut went to the floor: the rewind would have dropped the whole
  reasoning and restarted the model from the bare nudge, "move on with what I already have",
  with nothing to have.

Compression measures redundancy. A list, a log or a table is redundant by nature, and the digit
fold makes log lines with different numbers identical. What separates the two is time:
- a loop is absorbing: 23 of the 25 harmful Rome loops stay under the ratio for 9 checks or more,
  and most never leave;
- an enumeration ends with its content: no legitimate text in any corpus stays for more than 6.

## Decision

**A loop must persist.** The guard acts on a loop score only when it is the
`loop_guard_persist`-th in a row (4). A score that is not a loop starts the count over. The cut is
computed as before: back to the last clean window, which a run of loop scores never is while
`loop_guard_clean` is above `loop_guard_ratio`, as at the defaults (400 and 265).

**The threshold rises to 0.265** (`loop_guard_ratio` 265). Under persistence the old 0.20 misses a
slow harmful template. Between the highest threshold any harmful loop needs (0.257) and the lowest
at which legitimate agentic text fires once the rule below has taken out what opens in an
enumeration (0.274), 0.265 is the middle: the margin assumes `loop_guard_min_kept` is at least 1. The raw gate stays at 0.30:
lowering it to 0.28 loses 2 harmful loops.

**A rewind keeps something.** A loop whose cut would keep fewer than `loop_guard_min_kept`
characters of reasoning (1) is only logged (`ignis.loop_guard.too_little_kept`), in `rewind` and
in `log` mode alike, so `log` still reports what a rewind would do.
- The characters are counted in the reasoning the model reads, over the whole tokens a rewind
  keeps: a line break the decoder still holds at the start is not reasoning kept, nor is a token
  the cut falls inside.
- The run of loop scores is then held: the guard does not act on it again until a score that is
  not a loop.
- The watch keeps its scores, so a clean window later gives a later loop a cut.
- At the default, a second rewind always keeps the first one's text and nudge, so the rule never
  stops it. A `loop_guard_min_kept` larger than that text can.

| field | default | |
|---|---|---|
| `loop_guard_ratio` | 265 permille | was 200 |
| `loop_guard_persist` | 4 checks | new; 1 = ADR 0052's first check |
| `loop_guard_min_kept` | 1 character | new; 0 = rewind from nothing as before |

All three are live config fields (Visible, Patchable, no reload), with the new kind `checks`.

## Consequences

- On the Rome runs, the guard catches 25/25 harmful and 34/38 acute loops, against 24 and 33. No
  run without a loop fires. Every cut lands before the labelled onset.
- On the agentic corpora: 0 fires against 2.
- **Cost:**
  - a fire comes a median 500 characters later (at most 750 later, sometimes 4,250 earlier, since
    the higher threshold sees some loops sooner);
  - the struck span grows by up to 750 characters, and so do `rewound_tokens` and
    `completion_tokens`;
  - the continuation's budget does not pay for it: `rewind()` charges only the kept text and the
    nudge.
- **Loops that end on their own** within 4 checks are no longer touched, in Rome and in agents.
  They were not harmful.
- **A reasoning that opens in a loop** and never reaches a clean window is never rewound. It runs
  to the thinking budget's forced close, which ADR 0048 and the budget already provide. None was
  observed.
- **The margins are thin and in-sample:** one prompt, 25 harmful runs, two texts 0.017 apart.
  The held-out run in the finding (40 new seeds) agrees:
  - 6 of the 7 harmful loops caught, as by ADR 0052's rule, with the same cuts;
  - no fire on text that is no loop;
  - the 3 short loops the model left by itself left alone;
  - two breathing runs cut later.

  A false rewind or a missed harmful loop found later reopens these defaults.
- **Still not seen:** indecision loops whose text varies between repeats (a decision sentence
  between reconsiderations) do not compress. The only harmful agentic loop observed, the
  2026-10-12 deferral loop, is of that kind.

## Alternatives rejected

- **Only reword the nudge.** The nudge acted on legitimate work because the trigger put it there.
  With the new trigger it acts on persisted loops, where an exit is the measured remedy.
- **A nudge without the claim of sufficiency** ("…move on to the next point"). Measured live
  against today's (finding, 40 seeds each), it is worse on every column: harmful rewound runs
  2/10 against 0/12, median answer after a rewind 2,939 characters against 11,656. The model
  takes "the next point" as the next item of its outline and keeps enumerating.
- **Persistence without a higher threshold.** At 0.20, K 2 already loses a harmful loop that
  hovers around the threshold.
- **A lower raw gate**, to keep numeric tables out: it loses harmful loops.
- **Exempting text copied from the prompt.** A loop that recopies a tool output would be exempt
  for ever, and the floor rule and persistence already remove the observed case.
- **A different policy for requests with tools.** The Playground's agents have tools and looped on
  Rome; the coding sessions have tools and did not. The trigger has to hold everywhere.
- **Preventing loops in the sampler.** The Qwen3.8 card's thinking settings are the Playground's
  (presence_penalty 0.0), and `presence_penalty` 1.5 lengthens the reasoning into more forced
  closes (2026-10-12 finding).
