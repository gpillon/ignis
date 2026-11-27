---
status: accepted
---

# A reasoning loop is taken back, the model is nudged, and the client is told

A **loop guard**, a task between a reasoning request's engine event stream and its consumer, watches the reasoning
for a loop; once one persists it takes the generation back to before the loop, continues it from the kept text and
a nudge in the model's own voice, and tells the client what was taken back with a tag in the reasoning. The 27B
loops in its reasoning on long enumerations, at the same rate on ninfer with the same artifact: the loop is the
model's, ignis cannot fix the model, and a loop left alone can cost the rest of the thinking budget. What ignis can
do is notice, since the reasoning passes through the server and a loop compresses far better than reasoning does.
What it cannot do is step back cheaply: on the hybrid models the recurrent state absorbs every token it is fed, so
the only way back is a new request whose prompt is the kept text. The mechanism is in the opening comment of
`crates/server/src/loop_guard.rs`, and the measurements in the findings
[the loops are the model's](../findings/2026-10-12-reasoning-loops-are-the-models.md),
[the guard](../findings/2026-10-14-reasoning-loop-guard.md),
[a loop persists](../findings/2026-10-16-a-loop-persists.md) and
[a loop longer than the window](../findings/2026-11-27-a-loop-longer-than-the-window-repeats-verbatim.md).

**What it watches.** The guard is a tokio task, off the scheduler thread, and adds nothing to a round. It watches
only the reasoning of a request that starts inside its reasoning block, on chat completions and on `/v1/responses`
over HTTP and the WebSocket (the Playground talks over the Responses WebSocket by default, and Codex CLI speaks
only Responses), and until it acts it forwards every event unchanged. The answer is not watched: a nudge cannot go
into content. A request with images is logged, never rewound, until a continuation can carry its media.

**When it acts.**

- **On a compression score that persists.** The reasoning is scored normalized, so that legitimate code stays out
  and a counting or template loop becomes a repetition, with a gate on the text as written that keeps numeric
  tables out, where the digits carry the information. The guard acts only on the `loop_guard_persist`-th loop score
  in a row, and a score that is not a loop starts the count over. Compression measures redundancy, and a list, a
  log or a table is redundant by nature; what separates them is time: a loop is absorbing, an enumeration ends with
  its content. Acting at the first loop score fired on legitimate agent work, a checklist and a transcribed log.
- **Only when the rewind keeps something.** A loop whose cut would keep fewer than `loop_guard_min_kept` characters
  of reasoning is only logged, in `rewind` and in `log` mode alike. A reasoning that opens in a loop, or in an
  enumeration scored as one, has no clean window to go back to: the rewind would drop all of it and restart the
  model from the bare nudge, with nothing to move on with. The default `loop_guard_ratio` leans on this rule: its
  margin over legitimate agentic text assumes `loop_guard_min_kept` is at least 1.
- **On a repeat.** A loop whose block is longer than the score's window never shows two copies in it and scores
  like prose, at any ratio. So the guard also looks for the same sentences back to back, compared as written,
  within the `loop_guard_repeat_*` limits; only prose counts towards what a copy must repeat, so a listing or a log
  transcribed twice, to compare two outputs, is no loop. A repeat needs no persistence: its copies are its
  persistence, it is acted on at the end of the last, before another is written, and its cut keeps the first.

**What it does.** A rewind is a cancel, then a **continuation**: the same request input, its prompt extended by the
tokens kept from before the loop and by a nudge, with what is left of the generation cap and of the thinking
budget. After `loop_guard_retries` nudges the next loop continues with the thinking budget's own close text in the
nudge's place: the reasoning closes and the model answers. A loop that leaves no room for a continuation is only
logged. One HTTP request may so run several engine requests: the client sees one stream, and whatever cancels the
response cancels whichever of them is running.

**What the client gets.**

- **It keeps what it received and is told what was taken back**, by a tag streamed into the reasoning, followed by
  the nudge or the close: OpenAI's wire has no way to retract text, and buffering the reasoning until it is checked
  would hold it back by a whole window. A client that understands the tag strikes the text through, as the
  Playground does, and any other shows the tag as text. Every standard field keeps its meaning: ignis extensions
  are additive.
- **History repair gives the model back its own view.** A client sends the reasoning back as it received it, and
  the model never saw the struck text or the tags. On every incoming request, guard on or off, they are removed
  before the prompt is rendered, if and only if the struck text hashes to the hash its tag carries; otherwise the
  text is left as it is. Only assistant reasoning is repaired (chat `reasoning_content`, Responses `reasoning`
  input items): never a user, system, developer or tool message, never assistant `content`. It has no field: a
  history written while the guard was on reaches a server where it is off, and a reprise of the conversation goes
  through the same repair.
- **The repaired reasoning is the model's view token for token**, which is why a nudge or the close after a kept
  line break goes in without its own leading line breaks. A rewind then costs the next turn no reuse that an
  ordinary turn would have: reuse stops at the previous turn's generation opener, as for any turn (ADR 0029).
- **A holdback can hide a rewind**: `loop_guard_holdback` characters of reasoning are held back from the client,
  none by default, and a rewind whose cut falls in them reaches it with no tag and none of the loop. Everything
  held back goes out at the reasoning's close, and before the stream ends, however it ends, a passed deadline
  included (ADR 0058). It delays the reasoning by its size, and under about 2,000 characters it hides almost
  nothing: the shortest rewind measured struck 1,577, and on the confirm run 3,000 would have hidden 19 of 25
  rewinds, 6,000 would have hidden 24.
- **The Responses state stores the model's view**, not what was streamed, so the next turn that chains on it
  needs no repair. With `include: ["reasoning.encrypted_content"]`, which Codex asks for, a reasoning item carries
  that view as an opaque string, and one ignis can read is used on input in place of the item's visible text. It is
  not signed or encrypted: a client can already send any reasoning it likes in clear.
- **Usage counts what the model generated.** `completion_tokens` counts every token generated for the request, the
  rewound tokens and the tags included: they are work done. `rewound_tokens`, an additive ignis field in its
  details, says how many were taken back, is present only on a request the guard rewound, and absent means 0. The
  generation cap bounds what the client keeps, not the work: a rewound request may report `completion_tokens` past
  its `max_tokens`.

**How it is set.** Every threshold, and each nudge, is a live config field of the model group (ADR 0046), visible
and patchable with no reload. `loop_guard` is `rewind` by default, which the held-out runs favour, with `log` for
diagnosis; a false rewind found later reopens that default. The margins between the thresholds are thin and
in-sample: a false rewind or a missed harmful loop found later reopens the defaults. A server built without a
config (the in-process tests, the output-comparing GPU tests) runs the guard `off`, so no test meets a rewind it
did not ask for. Output differs from the reference whenever the guard acts, so benches and gates set
`loop_guard=off` explicitly.

## Considered Options

- **Roll back in place**, truncating the KV and restoring the recurrent state. Attention pages could be truncated,
  but the recurrent state would need a checkpoint at every possible cut point: a cut can reach 10,000 characters
  back, a device snapshot of the mutable state is about 148 MiB per lane, so a ring of them costs gigabytes, and
  host snapshots at PCIe Gen 3 cost a copy every check.
- **Re-submitting a turn from the server** is the alternative ADR 0048 rejected, and here it is the chosen one.
  There the leaf could redirect a stop id for free and a resubmit prefilled the whole reasoning again. For a loop
  no free fix inside the leaf exists: the damage is thousands of tokens back, and a rewind prefills only the kept
  text and the nudge, the prompt restored from its own checkpoint.
- **Close the reasoning at the first loop.** It forfeits the rest of the budget, and after one nudge 9 of 16
  rewound runs went on reasoning normally.
- **Prevent loops in the sampler.** A `presence_penalty` of 1.5 halved the loops but lengthened the reasoning into
  more forced closes, and the model card's thinking settings, the Playground's, have it at 0; `frequency_penalty`
  punishes code, which legitimately repeats. Neither is a fix for agents.
- **Detect by logit entropy.** Entropy measures how predictable a token is, and counting loops and code are both
  predictable; logprobs are not served either.
- **Novelty against the reasoning's history, the share of repeated lines, or scoring the window against the
  reasoning before it** with a zlib dictionary or a match search. Reasoning legitimately re-drafts code it wrote
  earlier, and these fire on it. The repeat signal is none of them: it leaves code out, compares sentences as
  written and asks for copies back to back.
- **A raw score alone.** It cannot separate counting and template loops from code.
- **A second, longer window for the score.** It lowers legitimate text's score too, and one window with one
  threshold gives up the short loops to catch the long ones.
- **Persistence without a higher ratio.** At the ratio of a guard that acts at the first loop score, two checks in
  a row already lose a harmful loop that hovers around the threshold, and persistence misses a slow harmful
  template.
- **A lower raw gate**, to keep numeric tables out: it loses harmful loops.
- **Exempting text copied from the prompt.** A loop that recopies a tool output would be exempt for ever, and the
  kept-text rule and persistence already remove the case that was observed.
- **A different policy for requests with tools.** The Playground's agents have tools and looped on the long
  enumeration; the coding sessions have tools and did not. The trigger has to hold everywhere.
- **Only reword the nudge.** The nudge acted on legitimate work because the trigger put it there; on persisted
  loops an exit is the measured remedy.
- **A nudge without the claim of sufficiency** ("…move on to the next point"). Measured live against today's, it is
  worse on every column: the model takes "the next point" as the next item of its outline and keeps enumerating.
- **A looser repeat**: two varying sentences per copy, a near-identical sentence counted as the same, or one
  sentence inserted or dropped per copy. None catches the loops the signal misses at their period; two varying
  sentences fire on an edit's old and new strings, and near-identical sentences counted towards the repeated prose
  fire before the loops, on sentences a word apart (their cost on legitimate text was not measured).
- **Three copies for a repeat.** It fires on almost nothing legitimate, but it lets the third copy be written and
  never catches a two-copy loop that ends the reasoning.
- **Guard in the clients.** Every client would need one, and they all differ; the server sees every request.
- **Retract the text on the wire.** No OpenAI event does that, and a client that does not know an invented one
  would show the loop and the continuation as one text. A tag degrades to text in a client that does not
  understand it.

## Consequences

- A rewind is one engine request more, prefilling only the kept text and the nudge: every request publishes its
  prompt's checkpoint at its generation opener (ADR 0029), so the engine restores it for the continuation even when
  the request prefilled its prompt cold. A continuation under memory pressure could find that checkpoint evicted:
  not measured.
- A continuation's penalty counts start from zero, and a forced tool call (`tool_choice`) is not forced after a
  close the guard made.
- History repair edits what a client sent, in one place only and behind a matching hash. A client that wants its
  text verbatim sends it without the tag.
- Persistence makes a fire come later, so the struck span grows, and `rewound_tokens` and `completion_tokens` with
  it; the continuation's budget does not pay for it, since a rewind charges only the kept text and the nudge.
- Loops that end on their own within the persistence are no longer touched: they were not harmful. Two copies the
  model would leave by itself are rewound all the same, and so is any block of prose written twice in a row, a
  checklist whose items share enough of it among them.
- A reasoning that opens in a loop and never reaches a clean window is never rewound: it runs to the thinking
  budget's forced close, which ADR 0048 and the budget already provide. None was observed.
- The guard misses loops that compress no better than legitimate text, which no threshold change recovers;
  indecision loops whose text varies between repeats, the kind of the only harmful agentic loop observed; loops
  that reword their copies, changing whole blocks between them; and a copy that adds a digression to the one
  before it.
