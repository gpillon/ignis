---
status: accepted
amended-by: [ADR-0053]
---

# ADR 0052 — a reasoning loop is taken back, the model is nudged, and the client is told

## Current decision

The guard stands as decided here, and **[ADR 0053](0053-a-loop-is-acted-on-once-it-persists.md)
holds its detection and action**: read it for when the guard acts. A tokio task between a
reasoning request's event stream and its consumer scores the normalized reasoning every 250
characters. It acts only once `loop_guard_persist` checks in a row (4) have scored a loop, under
`loop_guard_ratio` (265, was 200) and the raw gate, and a cut that would keep fewer than
`loop_guard_min_kept` characters of reasoning (1) is only logged (`too_little_kept`). The action
is a rewind: cancel, then a continuation whose prompt is the kept text and a nudge,
`loop_guard_nudge` then `loop_guard_nudge_again` (config fields since 2026-10-16), and the
thinking budget's close after `loop_guard_retries` (2).

The rest stands as written below: `rewind` by default, the hashed tag, history repair always on,
holdback off by default, the Responses state storing the model's view, `usage` counting the
rewound tokens. The code is on `main` (`crates/server/src/loop_guard.rs`), not on the branch the
Status names. ADR 0058 extends the flush of held-back reasoning to every surface with a deadline.

## History

### Status

Accepted (2026-10-15, owner). The owner also ratified what the implementation decided on
the way: the junction rule (a nudge after a kept line break goes in without its own
leading line breaks), `reasoning_tokens` counting what a holdback never sent, the guard
`off` on a server built without a config, a WebSocket continuation waiting in the
socket's queue rather than within `loop_guard_submit_wait`, and the new fields' bounds.
Amended 2026-10-16 (owner): the two nudges are config fields, `loop_guard_nudge` and
`loop_guard_nudge_again`, whose defaults are the sentences below.
Amended 2026-10-16 by [ADR 0053](0053-a-loop-is-acted-on-once-it-persists.md): the guard acts
once a loop persists (`loop_guard_persist` 4 checks in a row), the ratio is 265, and a rewind
that would keep no reasoning is only logged (`loop_guard_min_kept`).
Proposed 2026-10-14. The design is the owner's (2026-10-13: an
engine-level detector off the critical path, an injected sentence, a few retries before
the reasoning is closed, a visible tag the Playground strikes through). The numbers are in
the finding `docs/findings/2026-10-14-reasoning-loop-guard.md`. Code on branch
`experiment/loop-rewind` (`crates/server/src/loop_guard.rs`). **Flags a conflict with ADR
0048**, whose rejected alternatives include re-submitting a turn from the server (below).

Owner decisions of 2026-10-14, folded in below: `completion_tokens` counts the rewound
tokens; the guard is on (`rewind`) by default; **history repair** is always active; the
Responses state stores the model's view and carries it in `encrypted_content`; a
**holdback** can hide a rewind from the client; the guard is wired on `/v1/responses`.

### Context

Qwen3.8-27B loops inside its reasoning on long enumerations. On the Playground's "whole
history of Rome" prompt, about a third of first turns loop
(`docs/findings/2026-10-12-reasoning-loops-are-the-models.md`), and about a quarter end
badly: the thinking budget forces the close, the model ends its turn inside the
reasoning, or no answer fits in what is left. The loops come in several shapes:

- verbatim text;
- a single token in every slot;
- counting (`the 321 AD, the 322 AD, …`);
- templates whose slots vary (`the election of Clement XII, the construction of the
  Sistine Chapel (1508–1512)`).

The rate is the same on ninfer with the same artifact. It is the model's, and ignis
cannot fix the model. A turn that loops wastes up to the whole 14,336-token budget, and an
agent that reaches the forced close keeps 2,048 tokens for its answer.

Agents send the reasoning back. opencode 1.18.35 (`@ai-sdk/openai-compatible`, no
`interleaved` option) puts `reasoning_content` on the assistant tool-call message of its
loop (measured 2026-10-15 through a logging proxy, in the
[finding](../findings/2026-10-14-reasoning-loop-guard.md)), so whatever the server streams
into the reasoning comes back in the next turn's prompt, and history repair (below)
applies to it.

What ignis can do is notice. A loop is text that compresses far better than reasoning
does. A request's reasoning passes through the server on its way to the client, and the
server can score it as it goes without touching the scheduler.

What the server cannot do is step back cheaply. On the hybrid models the GDN state
absorbs every token it is fed, and there is no position to return to without a
checkpoint. The only way back is a new request whose prompt is the kept text.

### Decision

**A guard sits between a request's engine event stream and its consumer.** It is a tokio
task, so it is off the scheduler thread and adds nothing to a round. It watches only the
reasoning of a request that starts inside its reasoning block. Until it acts, it forwards
every event unchanged.

**Detection is a compression score over normalized reasoning.**
- Every `loop_guard_check_every` raw reasoning characters (250), the guard takes the
  last `loop_guard_window` characters of the reasoning after three steps:
  - fenced code is removed, and a block still open is never scored;
  - inline code is removed;
  - every run of ASCII digits becomes `0`.
- It computes the zlib-9 compress ratio of that text.
- A check fires when that ratio is under `loop_guard_ratio` **and** the ratio of the last
  `loop_guard_window` raw characters is under `loop_guard_raw_gate`.
- Why each step:
  - the code steps keep legitimate code out of the score;
  - the digit fold turns a counting or template loop into a repetition;
  - the raw gate keeps numeric tables out, where the digits carry the information.

**The action is a rewind: cancel, then a continuation.**
- **The cut.** The guard walks back over its own checks to the last one that scored at
  least `loop_guard_clean`. The cut is the raw start of that check's window, minus
  `loop_guard_margin`, snapped to the line break before it when one is within
  `loop_guard_snap` characters, and never before the previous cut. If the kept text
  would end inside a ``` block, the cut moves before that block's fence.
- **The continuation.** The engine request is cancelled. The **continuation** is the same
  request input, with its prompt extended by the tokens kept before the cut and by a
  **nudge**: a sentence in the model's own voice ("Wait, I'm starting to repeat myself.
  Let me stop and move on with what I already have."), `loop_guard_nudge`; every later
  rewind writes `loop_guard_nudge_again`. It gets what is left of the
  generation cap and of the thinking budget, and a fresh seed.
- **When there is no continuation.** A loop with fewer than
  `loop_guard_min_continuation` tokens of the generation cap left after the kept text
  and the nudge is only logged (`no_room`). A continuation the engine refuses as full
  waits up to `loop_guard_submit_wait` for room, and the loop is left as it is when none
  comes (`submit_refused`); a WebSocket request's continuation waits in the socket's
  queue instead (below).
- **Retries and close.** After `loop_guard_retries` nudges, the next fire continues with
  the thinking budget's own close text in place of a nudge. That closes the reasoning,
  and the model answers.
- **One HTTP request.** One HTTP request may run several engine requests. The guard maps
  their events back onto one stream, with its token counts and offsets.

**The client keeps what it received and is told what was taken back.** OpenAI's wire has
no way to retract text, and buffering the reasoning until it is checked would hold it back
by a whole window. So the guard streams a tag into the reasoning channel, followed by the
nudge or the close:

```
\n\n[LOOP DETECTED, REWINDING <n> TOKENS / <m> CHARS · <h>]
```

`<m>` counts the Unicode scalar values before the tag, back to the cut, as the client
shows them (a line break its decoder still held when the guard fired is one of them): the
**struck text**. `<h>` is the first 6 lowercase hex digits of the SHA-256 of the struck text's
UTF-8 bytes. A client that understands the tag strikes the `m` characters through: the
Playground does (`web/src/ui/rewinds.ts`, which also still reads the tag without `· <h>`).
Any other client shows the tag as text. Every standard field keeps its meaning. This is
the extension rule of the Responses spec: ignis extensions are additive and never change
what a standard field means.

**History repair: the next request gives the model back its own view.** A client sends the
reasoning back as it received it, struck text and tags included, but the model never saw
those: its view is the kept text, the nudge and the continuation. On every incoming
request, guard on or off, ignis repairs that text before rendering the prompt:
- **Where.** Only assistant reasoning: chat `messages[].reasoning_content` on an
  `assistant` message, and the text of Responses `reasoning` input items. Never a user,
  system, developer or tool message, never assistant `content`.
- **What.** For each tag, from the last to the first, the `m` characters before it and the
  tag itself are removed, if and only if those `m` characters hash to `<h>`. On a
  mismatch, too few characters, or a tag without a hash, the text is left as it is.
- **Why always.** A history written while the guard was on reaches a server where it is
  off, and a reprise of the conversation goes through the same repair.
- **Effect.** The repaired reasoning equals the text the model saw. Text equality is what
  repair guarantees. Token equality needs one more rule: after a kept text that ends in a
  line break, the nudge or the close goes in without its leading line breaks, because the
  two runs would re-tokenize as one. With that rule, the repaired reasoning re-tokenizes
  into the continuation's prompt at every cut measured: 4,165 line-break cuts and 18 real
  cuts (`examples/loop_retokenize.rs`). Without it, 0 and 5 do.
- **Measured on the next turn** (2026-10-15, a 79k-token agentic context,
  [finding](../findings/2026-10-14-reasoning-loop-guard.md)). The repaired history is the
  model's view token for token: the visible text with a good hash, the `encrypted_content`
  blob and the socket's stored view (`previous_response_id`) render the same tokens. The
  proof is a checkpoint hit, since a checkpoint is matched by content: on HTTP the
  good-hash arm restored the blob arm's checkpoint at prompt − 2, and on the socket the
  full good-hash input restored the chained turn's. A corrupted hash
  leaves +585 (chat), +866 (socket) and +790 (HTTP) tokens of struck text and tags in the
  prompt. Reuse stops at the previous turn's generation opener, as for any turn: ADR 0029
  keeps no checkpoint after generation, and the controls without a rewind stop at the
  same place. A rewind therefore costs the next turn no reuse that an ordinary turn
  would have.

**Holdback: an option to hide a rewind before the client sees it.** `loop_guard_holdback`
holds the reasoning stream back by that many characters (0 = off, the default):
- a rewind whose cut falls in the unsent part streams no tag: the client gets the kept
  text, the nudge and the continuation, the model's view token for token, and never sees
  the loop;
- a cut inside the sent part strikes only the sent characters, and the unsent rest is
  dropped;
- everything held back flushes at the reasoning's close, at the end of the stream, and on
  an error or a cancel.
- Holding back delays the reasoning a client sees by that many characters. Below ~2,000
  it hides almost nothing: the shortest rewind measured struck 1,577 characters. On the
  confirm run, 3,000 would have hidden 19 of 25 rewinds, and 6,000 would have hidden 24.

**The Responses state stores the model's view.** `/v1/responses` keeps state on the server
(`previous_response_id`, the WebSocket session). What it stores for a rewound response is
the model's view, kept text + nudge + continuation, not what was streamed:
- the next turn that chains on it needs no repair;
- with `include: ["reasoning.encrypted_content"]` (Codex asks for it), every reasoning
  output item carries `encrypted_content`, an opaque string: `ignis:v1:` followed by the
  base64 of the model-view reasoning's UTF-8. It is on the item's `done` event and in
  the response; the `added` event's empty item has none;
- on input, a reasoning item whose `encrypted_content` ignis can read is used in place of
  its visible text; one it cannot read (another server's blob, a malformed value) falls
  back to the visible text, which history repair then sees;
- it is not signed or encrypted: a client can already send any reasoning it likes in clear.

**Usage counts what the model generated.**
- `usage.completion_tokens` counts every token generated for the request, the rewound
  tokens and the tags included: they are work done, and they are what the model was
  billed for in time.
- `usage.completion_tokens_details.rewound_tokens` (an additive ignis field) says how many
  of them were taken back. It is present only on a request the guard rewound, and an
  absent field means 0.
- The generation cap bounds what the client keeps, not the work. A rewound request may
  report `completion_tokens` past its `max_tokens`.

**The guard covers every request that reasons, on both surfaces.**
- That means `POST /v1/chat/completions`, and `/v1/responses` over HTTP and over the
  WebSocket.
- The Playground talks over the Responses WebSocket by default, and Codex CLI speaks only
  Responses.
- `/v1/responses` reports the same count as `output_tokens_details.rewound_tokens`.
  Its `output_tokens` and `output_tokens_details.reasoning_tokens` both count the
  tokens a holdback never sent: they were generated in the reasoning.
- After a rewind the engine request id changes. A `response.cancel` addressed to the
  response cancels whichever engine request is running for it, as do a socket that
  closes and an HTTP client that hangs up.
- A WebSocket request's continuation waits for admission in the socket's server-wide
  queue, behind what already waits there, as the socket's own requests do.
- A request with images is logged, never rewound, until a continuation can carry its
  media.
- The answer is not watched: a nudge cannot go into content.

**Every threshold is a live config field** (ADR 0046, model group, Visible + Patchable,
no reload):

| field | default |
|---|---|
| `loop_guard` | `rewind` (also `log`, `off`) |
| `loop_guard_normalize` | on |
| `loop_guard_window` | 1,000 characters |
| `loop_guard_ratio` | 200 permille; 265 since ADR 0053 |
| `loop_guard_raw_gate` | 300 permille (0 = off) |
| `loop_guard_clean` | 400 permille |
| `loop_guard_margin` | 0 characters |
| `loop_guard_retries` | 2 |
| `loop_guard_persist` | 4 checks (ADR 0053) |
| `loop_guard_min_kept` | 1 character (ADR 0053) |
| `loop_guard_holdback` | 0 characters (off) |
| `loop_guard_check_every` | 250 characters |
| `loop_guard_snap` | 300 characters (0 = never snaps) |
| `loop_guard_min_continuation` | 512 tokens |
| `loop_guard_submit_wait` | 60 seconds (0 = tried once; at most 3,600) |
| `loop_guard_nudge` | "Wait, I'm starting to repeat myself. Let me stop and move on with what I already have." |
| `loop_guard_nudge_again` | "I keep repeating myself. I already have enough to write the answer, so I'll wrap up my reasoning now." |

History repair has no field: it is always on, and a tag whose hash does not match is
never touched.

**`rewind` by default** (owner, 2026-10-14). The held-out run favours it: no false rewind
in 25, and harmful runs fell from about a quarter to 4 in 40. `log` stays for diagnosis.
The run on agentic contexts of 79k and 138k tokens (2026-10-15,
[finding](../findings/2026-10-14-reasoning-loop-guard.md)) found no false rewind: 16
seeds, 1,369 checks, the lowest normalized score 0.449 against the 0.20 threshold. Any
false rewind found later reopens this default.

A server built without a config (`Live::default()`: the in-process tests, the
output-comparing GPU tests) runs the guard `off`: only a configured server (`main`, a
live `PATCH`) rewinds, so no test meets a rewind it did not ask for.

### Alternatives rejected

- **Roll back in place** (truncate the KV, restore the GDN state). Attention pages could
  be truncated. The recurrent state, though, would need a checkpoint at every possible cut
  point. A cut can reach 10,000 characters back, and a device snapshot of the mutable
  state is ~148 MiB per lane, so a ring of them costs gigabytes. Host snapshots at
  PCIe Gen 3 cost a copy every check.
- **Re-submit a turn from the server.** This is the alternative ADR 0048 rejected, and
  here it is the chosen one. 0048 rejected it because the leaf could redirect an EOS for
  free, and a resubmit re-prefills the whole reasoning. For a loop there is no free
  in-leaf fix: the damage is thousands of tokens back. A rewind costs one prefill of the
  kept text and the nudge, the prompt restored from its own checkpoint: measured
  0.4-1.7 s on the confirm run, whose continuations prefilled 2.4k-11.9k kept tokens, and
  0.08-0.10 s at 79k-138k tokens with little kept. A loop left alone can cost the rest of
  the budget.
- **Close the reasoning at the first loop.** That forfeits the rest of the budget. After
  one nudge, 9 of 16 rewound runs went on reasoning normally.
- **Prevent loops in the sampler.** `presence_penalty` 1.5 halved the loops on 8 seeds but
  lengthened the reasoning into more forced closes. `frequency_penalty` punishes code,
  which legitimately repeats. Neither is a fix for agents.
- **Detect by logit entropy.** Entropy measures how predictable a token is, and counting
  loops and code are both predictable. Logprobs are not served either (GitHub #288).
- **Other text signals:**
  - **Novelty against the reasoning's history** and **the share of repeated lines.**
    Reasoning legitimately re-drafts code it wrote earlier, and both fire on it.
  - **A raw score alone.** It cannot separate counting and template loops from code.
- **Guard in the clients.** Every client would need one, and the Playground, opencode and
  Codex all differ. The server sees every request.
- **Retract the text on the wire.** No OpenAI event does that. A client that does not
  know an invented one would show the loop and the continuation as one text. A tag
  degrades to text in a client that does not understand it.

### Consequences

- **Reference parity.** Output differs from the reference whenever the guard acts. Benches
  and gates set `loop_guard=off` explicitly, since the default is `rewind`.
- **Cost of a rewind.** A rewind is one engine request more, and it prefills only the
  kept text and the nudge. Every request publishes its prompt's checkpoint at its
  generation opener (ADR 0029), and the continuation's prompt starts with that prompt,
  so the engine restores it, even when the request prefilled its prompt cold. Measured
  2026-10-15 ([finding](../findings/2026-10-14-reasoning-loop-guard.md)): the client
  waits 0.078-0.090 s from the tag to the next delta at 78,773 tokens and 0.099-0.100 s
  at 137,772, with little kept, and 1.276 s with 17,944 characters (about 3,645 tokens)
  kept. Prefilled cold, 78,809 tokens
  take 13.09 s and 137,742 take 34.5 s. A continuation under memory pressure could find
  that checkpoint evicted: not measured.
- **History repair touches input.** It edits what a client sent, in one place only:
  assistant reasoning, behind a matching hash. A client that wants its text verbatim
  sends it without the tag.
- **What a continuation does not carry over:**
  - its penalty counts start from zero;
  - a forced tool call (`tool_choice`) is not forced after a close the guard made.
- **Misses.** The guard misses loops that compress no better than legitimate text.
  Measured on the held-out run: 8 episodes at 0.21-0.31 normalized. The lowest legitimate
  texts are at 0.257 (Rome) and 0.273 (agentic coding), so no threshold change recovers
  them.
- **Observability.** The log carries `ignis.loop_guard.detected`, `rewound`, `no_room`,
  `no_tokenizer`, `submit_refused` and (ADR 0053) `too_little_kept`, each with its scores and offsets. `log` mode
  reports the cut a rewind would have made, without acting. Since #330 (spec server/14)
  `/metrics` tallies those lines per outcome (`ignis_loop_guard_loops_total`), with the
  requests watched and looped, the tokens taken back and each request's lowest score
  (ADR 0017's table), and the Playground's Monitor shows them.
- **Wiring.** Implementing this ADR means:
  - wiring `/v1/responses`, HTTP and WebSocket, with its own count field, cancel by
    response, the model-view store and `encrypted_content`;
  - wiring the `rewound_tokens` field;
  - the hashed tag, history repair on both surfaces, and `loop_guard_holdback`;
  - the glossary terms (**Loop guard**, **Rewind**, **Continuation**, **Nudge**,
    **History repair**, **Holdback**) in `GLOSSARY.md`.
