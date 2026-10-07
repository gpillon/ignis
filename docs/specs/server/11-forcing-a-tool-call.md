# server 11 — Forcing a tool call: `tool_choice` gets its lever

GitHub: #286

## Problem Statement

`parse_tool_choice` (`crates/server/src/api.rs:728`) refuses
`tool_choice: "required"` and `tool_choice: {"type": "function", ...}` with

> this template has no way to force a tool call; use "auto" and let the
> model decide, or "none"

That was true when #132 was written and it is no longer true. This engine
grew a **permitted token set per decode step** at #242 (ADR 0034): the caller
declares which tokens a step may draw from, the leaf masks the rest, and the
draw composes with temperature, seeds and penalties as an ordinary one. It is
how `number`, `point`, `box` and `scalar` emit a digit at a time, and how
`scalar` forces the literal `{"value": ` before it.

The tool-call dialect this template uses is **literal text** —

```text
<tool_call>
<function=NAME>
<parameter=PARAM>
VALUE
</parameter>
</function>
</tool_call>
```

(`.qwen/tmp/chat_template.jinja:68`, `crates/server/src/toolcall.rs`) — with
no special token anywhere in it. A forced opener is therefore a literal of a
handful of tokens, each step permitting exactly one of them. The lever the
refusal says does not exist is the same one `scalar` already pulls.

What it costs a caller today: an agent that needs a call writes a retry loop
and pays for a whole turn of prose whenever the model answers in words. A
router that has already decided a tool must run cannot say so.

## Solution

**`tool_choice` forces the opening of the dialect and then gets out of the
way.**

- `"required"` forces `<tool_call>\n` and leaves the function name free.
- `{"type": "function", "function": {"name": "N"}}` forces
  `<tool_call>\n<function=N>\n` — the name too.

The forced tokens are the first tokens of the generation, drawn from
single-element permitted sets: the first by the prefill
(`PrefillJob::permitted`, #242), the rest by the decode rounds that follow.
After the last forced token the run **continues freely** — the arguments, the
closing tags and anything after the call are the model's own. The scanner in
`toolcall.rs` then finds a block that is already open and parses it exactly
as it parses one the model opened itself.

The core already does exactly this once. The thinking budget
(`crates/core/src/thinking_budget.rs`, spec 08) forces the literal
`</think>` one width-1 permitted set per round and then **stops forcing**,
leaving the request generating. `ThinkingClose` holds the literal's token
ids, `BudgetState::permitted` hands out one set per round and `None` once
the literal is spent. That is a forced prefix that releases its run, built
beside `Schedule` rather than inside it, precisely because a `Schedule` ends
its run when exhausted and a prefix must not.

So this spec generalizes that pattern rather than inventing one: the same
shape, with the literal and the position it starts at coming from
`tool_choice` instead of from a budget.

**Note for triage.** Thinking is **on by default** on this server
(`ThinkingOptions::default`, `thinking.rs:46`), and no OpenAI-SDK client
sends `enable_thinking: false`. With the exclusion below, that means
`tool_choice: "required"` from Codex, qwen-code or opencode answers `400`
until the caller reaches for an ignis extension. That may be the right
answer — or it may mean the feature this spec should build is "force the
call *after* `</think>`", which is a different mechanism (see Further
Notes). Decide that before implementing.

**Decided (2026-10-07): force after `</think>`.** Built that way; see
[As built: departures](#as-built-departures-2026-10-07) at the end, which
override the sections between here and there where they disagree.

## User Stories

1. As an agent that has already decided a tool must run, I want
   `tool_choice: "required"` honoured, so I stop paying for turns of prose I
   discard.
2. As a router, I want to name the function, so the model fills in arguments
   for the tool I chose rather than choosing again.
3. As a client on an OpenAI SDK, I want the standard values accepted, so I do
   not special-case this server.
4. As a caller, I want a clear `400` when I ask for a forced call together
   with thinking, so I learn the constraint at the request instead of from a
   malformed answer.
5. As an operator, I want to see how often a forced call failed to close, so
   a template or a model change that breaks forcing is visible.

## Implementation Decisions

### The forced literal

- Built from the request: `"required"` gives `<tool_call>\n`; a named
  function gives `<tool_call>\n<function=NAME>\n`, with `NAME` exactly as
  the request spelled it.
- Encoded through the template's own `encode_literal`
  (`template.rs:582`) — the same path `scalar` uses for `{"value": `. A
  literal that does not encode (the tokenizer returns `None`, or the name
  carries something the vocabulary cannot represent) is a **`400`, not a
  silent fallback to `"auto"`**: a constraint quietly dropped is a wrong
  answer that looks like a right one (`constrained.rs`).
- One step per token, each permitting exactly one id. A forced opener is a
  handful of steps, far inside `MAX_PERMITTED_TOKENS` (32) — that cap bounds
  the *width* of a step, and every step here has width 1.
- A named function must be one of the request's `tools`; naming an unknown
  function stays the `400` it is today.

### Releasing the run

- The mechanism is `thinking_budget.rs`'s, generalized: a **forced literal**
  holding token ids and a position to start at, and a per-request state that
  answers "the permitted set for this round" with `Some([id])` while the
  literal is unspent and `None` after. `None` is what the scheduler already
  reads as "this round is unconstrained".
- It is **not** a `Schedule`. A `Schedule` ends its run when exhausted
  (`constrained.rs`) and every existing caller depends on that; a forced
  prefix must not end anything. The two stay separate types.
- The forced literal here starts at position 0 of the generation, so its
  first token is drawn by the prefill (`PrefillJob::permitted`, #242) — the
  one-round lag, handled exactly as a constrained decode handles it.
- EOS during the forced prefix is impossible by construction (no step
  permits it). EOS after the release is an ordinary stop.
- The budget's own forced close and this one cannot collide: forcing is
  refused with thinking on, and the budget only fires inside a reasoning
  block.

### The forced ids must be ids the model would itself emit

A literal encoded by `encode_literal` is one tokenization of
`<tool_call>\n`; the model, writing the same text, may merge differently at
the `>`/`\n` boundary. Feeding the model a tokenization it would never
produce degrades what follows it (the token-healing problem). So the forced
ids are **checked against the real thing**: the tool calls recorded in
#121's fixtures are tokenized, and the forced literal must be a prefix of
those ids. Where it is not, the literal is shortened to the longest prefix
that is — `<tool_call>` without the newline, if that is what the check
says. The check is a test, not a runtime path.

### Thinking and forcing are exclusive

With thinking on, this template's generation opener is
`<|im_start|>assistant\n<think>\n` (`frontend.rs:1648`): **the first
generated token is already inside a reasoning block.** A forced
`<tool_call>` there lands on `Channel::Reasoning`, where `toolcall.rs` never
scans, and the client gets a reasoning block containing XML and no tool call
at all.

So: `tool_choice: "required"` or a named function, together with thinking
resolved on, is a **`400`**, `param: "tool_choice"`, naming the fix —
`enable_thinking: false` for this request, or `tool_choice: "auto"` to let
the model decide after it has thought.

This is a refusal of a combination, decided at the request, not a silent
disabling of thinking. Which of the two the caller wanted is not knowable
from the body, and guessing it would change what the model reads.

### Interaction with the rest

- **Speculative decoding is not restricted.** A permitted set rides the
  decode job and lives in the round's parameters beside
  `speculative_window`; `/v1/decide` has been measured constrained with
  `--spec dflash2` on (spec decide/16). Nothing here turns it off.
- **`max_tokens` / `max_completion_tokens`**: the forced tokens are
  generated tokens. They count in `completion_tokens` and against the cap. A
  cap shorter than the forced literal is a `400` — the request cannot
  produce what it asks for.
- **`finish_reason`** is unchanged: `resolve_finish_reason` (`api.rs:1020`)
  already reports `"tool_calls"` for a generation that closed at least one
  call, and refuses to report it for a call left open.
- **A forced call that never closes** is dropped whole, exactly as today
  (#121 acceptance 3) — the client is never told a call happened that did
  not. It is also *counted*: a new
  `ignis_forced_tool_calls_total{outcome="closed"|"unclosed"}` makes a
  template or checkpoint change that breaks forcing visible instead of
  silent.
- **Streaming** is unchanged: the forced tokens are part of the buffered
  tool-call block, so nothing of the opener reaches the client as content.
- **`/v1/responses`** carries the same `tool_choice` values and gets the same
  behaviour through the same resolution.

### What does not change

- `tool_choice: "auto"` and `"none"`, and every request that does not force.
- The rendered prompt: forcing changes what the model may *draw*, never what
  it reads. The system prompt's tool block is rendered as it is today.
- `toolcall.rs`'s parsing, its whole-call contract, and its argument typing.

## Testing Decisions

- **Core, on the mock**: a spent forced literal leaves the run generating
  while a spent `Schedule` still ends it; the first forced token is drawn by
  the prefill, not by the first round (the one-round lag — the test that
  cost a GPU failure to learn at #242); the budget's forced close and a
  forced opener never both apply to one round.
- **The tokenization check**: the forced ids are a prefix of the ids the
  recorded #121 tool calls tokenize to. This test is what decides whether
  the literal carries the trailing newline.
- **Server, over the mock engine**: `"required"` builds the `<tool_call>\n`
  schedule; a named function builds the longer one; an unknown name, a name
  that does not encode, thinking-on, and a too-small `max_tokens` each
  answer `400` with their `param` and code; `"auto"`/`"none"` build no
  schedule at all.
- **End to end over a scripted engine** that emits arguments after the
  forced opener: the response carries one `tool_calls` entry with the right
  name, `finish_reason: "tool_calls"`, and no fragment of the opener in
  `content` — streaming and not.
- **The unclosed case**: an engine that stops mid-call yields no tool call,
  the ordinary `finish_reason`, and increments the `unclosed` counter.
- **GPU, once at the end** (card free first, AGENTS.md): against the served
  27B with `--spec dflash2`, a `required` request and a named-function
  request over a two-tool body each return a parseable call; ten repetitions
  each, recorded as a finding row.

## Acceptance

1. **`required` forces.** Over the served model, ten `tool_choice:
   "required"` requests on a body whose prompt invites a prose answer each
   return at least one parsed `tool_calls` entry and
   `finish_reason: "tool_calls"`.
2. **A named function is the one called.** Ten requests naming one of two
   tools each return a call to that tool and no other.
3. **Nothing leaks.** In neither case does any part of `<tool_call>` or
   `<function=` appear in `content` or in a content delta.
4. **The release works.** A spent forced literal leaves the request
   generating; asserted in core on the mock, and visible end to end as
   arguments no permitted set ever allowed.
5. **The ids are the model's own.** The forced literal is a prefix of the
   ids the recorded #121 tool calls tokenize to.
6. **The refusals.** Thinking-on, an unknown function name, a literal that
   does not encode, and a cap shorter than the opener each answer `400`
   with the documented code and `param`.
7. **Speculation on.** Acceptance 1 and 2 hold with `--spec dflash2`.
8. **Unchanged.** `"auto"` and `"none"` requests produce the same tokens as
   before this spec, greedy, against a recorded fixture.
9. **Docs.** The OpenAPI operation description says which `tool_choice`
   values are served and names the thinking exclusion; `CONTEXT.md` gains
   the forced-literal term.
10. `cargo test` passes workspace-wide.

## Out of Scope

- **Choosing the tool without generating** — a `choice` readout over the tool
  names, then constrained arguments — #292. This spec forces an opening the
  model then fills, which is the cheap half.
- **Constraining the arguments to the tool's JSON schema.** That is #287's
  alphabet problem (a step of width 32 cannot hold a string state), not this
  one's.
- `parallel_tool_calls: false` (#284 refuses it).
- Forcing a call anywhere but at the start of the generation.

## Further Notes

- **Why not disable thinking implicitly** when a call is forced: it changes
  the prompt the model reads (`<think>\n` leaves the opener) without the
  caller asking, and the two plausible intents — "think, then certainly call"
  and "call now" — are indistinguishable in the body. The first is not
  served by this spec at all; it needs a forced opener applied *after*
  `</think>`, which the one-round lag makes a different mechanism, and it is
  in Out of Scope by omission until somebody asks for it.
- **Why `required` leaves the name free**: forcing `<function=` and stopping
  there risks a step boundary inside a token the tokenizer merges across the
  `=`; and the model choosing among the offered tools is the behaviour the
  caller wanted — they asked for *a* call, not for *that* call.
- **Prior art in this repo**, in order of closeness: the thinking budget
  (spec 08, `thinking_budget.rs`) forces a literal and then releases — the
  whole mechanism, at a different position; `scalar` (spec decide/10, #255)
  forces a literal prefix and then keeps constraining.
- **Forcing after `</think>`** — the thing the default-on note above points
  at — is not this mechanism. The budget knows *in advance* where it will
  force (a token count); "after the model's own close" is only known one
  round late, so the round that discovers `</think>` has already drawn a
  free token. Making that work means either accepting one free token
  between the close and the opener, or a leaf that can re-draw. Neither is
  in this spec.

## As built: departures (2026-10-07)

What was built differs from the sections above in these points, and these
win where the two disagree.

1. **Thinking on forces after the block instead of refusing.** With thinking
   resolved on, `"required"` and a named function are forced right after the
   reasoning block closes, by the model's own `</think>` or by the thinking
   budget's forced close; with it off, from the first token as specified.
   Why: thinking is on by default and no OpenAI-SDK client turns it off, so
   the exclusion would have made `tool_choice: "required"` from Codex,
   qwen-code or opencode a `400` every time, which is the feature not
   shipping for the clients that send it. "Think, then certainly call" is
   also what an agent wants: the forcing changes what the model may draw
   after its reasoning, never the prompt it reads. The 400 survives only
   where the mechanism cannot work: a tokenizer whose `</think>` or
   `<tool_call>` is not a single token (`code: tool_choice_unsupported`,
   naming `enable_thinking: false`).
2. **The one free token after a natural close is accepted, and joined on.**
   The round that commits the model's `</think>` has already drawn the next
   token freely (a speculative round may commit several), so the first forced
   draw follows a token nobody saw. If that token was `<tool_call>` itself,
   forcing `<tool_call>` again would open the call twice and it would not
   parse. So the first token forced after a natural close is the opening's
   *second*, `\n` (id 198), the **joiner**: right after `<tool_call>`,
   and after anything else a line break in the content (trimmed when nothing
   precedes it, as after the usual `\n\n`).
   The opening then resumes at `<` when the unseen token was `<tool_call>`,
   and from `<tool_call>` otherwise. After a close the budget forced there is
   no unseen token (its close ends with forced tokens, `</think>\n\n`), so
   no joiner. When a speculative round commits the model's own `<tool_call>`
   with its close, a call stopped inside the tokens every call writes alike
   (`<tool_call>` `\n` `<` `function`) is continued: the unseen token is
   taken to be the next of them, and the forcing resumes after it — through
   the name, for a named function. A call already past them, or off the
   dialect after `<tool_call>`, is the model's and is left alone.
3. **`"required"` forces `<tool_call>\n<function`, not `<tool_call>\n`.** The
   27B tokenizer writes the opening as `<tool_call>` `\n` `<` `function`
   and then merges the `=` into the name (`=read`, `=get`; `=` `shell` when
   it does not). `<function=` is therefore not a step boundary, and stopping
   before the `=` keeps every forced id one the model writes itself; it also
   rules out a call written in another dialect. The prefix check this spec
   asks for passes on all six recorded calls (`tool_choice_real_tokenizer.rs`).
   A named function forces the whole `<tool_call>\n<function=NAME>\n`, encoded
   in one call so the name merges as the model's would.
4. **The forced literal carries its own `</think>` id** (`ForcedLiteral::
   after_reasoning`), so detecting the close does not depend on the
   scheduler's thinking-budget close being configured.
5. **Deferred: `ignis_forced_tool_calls_total`.** Not built here: the outcome
   (closed or not) is read where each answer path resolves its
   `finish_reason` (chat's three, the Responses events), and the counter is
   plumbing through all of them, a follow-up of its own. A forced call that
   never closes is still dropped whole and ends with the ordinary
   `finish_reason` (`openai_http_tool_choice.rs`).
6. **The cap check does not count the reasoning.** With thinking on, the
   reasoning spends the same `max_tokens` the call needs, by an amount not
   knowable at the request; only a cap shorter than the opening is a `400`.
   The thinking budget's answer reserve (`ANSWER_RESERVE`, spec 08) is what
   keeps room for the call; a request with no budget may reason to its cap
   and end `length` with no call.

**Known gaps.** A named function under speculation, when one round commits
`</think>` and the model's own call past `<function` together: nothing is
forced, and the model may name a different tool than the one asked for.
Prose the model writes before the forced opening after a natural close (the
unseen token, and whatever a speculative round committed past its
`</think>`) reaches `content`; no fragment of the opening does. When that
unseen token is the end of the turn, the request ends there with no call.
The budget's forced close still says "without calling any more tools" before
a forced call.
