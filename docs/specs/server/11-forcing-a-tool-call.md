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

This needs one new thing in the core: a schedule that, when spent, **hands
the run back to free sampling instead of ending it**. Every schedule today
ends its run when exhausted (`constrained.rs`, `Schedule`), because every
constrained decode so far has been the whole answer. A forced opener is a
prefix, not an answer.

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

- `Schedule` gains a **released** form: when its steps are spent the request
  keeps generating, unconstrained, instead of finishing. The three outcomes
  a spent schedule has today (ended on the terminator, spent at the cap,
  cut short) are unchanged for every existing caller; a released schedule
  simply has a fourth: spent and still running.
- A released schedule has **no terminator** — a terminator ends a run, and
  this one does not end anything.
- EOS during the forced prefix is impossible by construction (no step
  permits it). EOS after the release is an ordinary stop.
- The core change is small and it is the one place this spec touches below
  the server: `crates/core/src/constrained.rs` and the scheduler's check for
  "schedule spent, therefore done".

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

- **Core, on the mock**: a released schedule keeps the run alive when spent
  and an ordinary one still ends it; the first forced token is drawn by the
  prefill, not by the first round (the one-round lag — the test that cost a
  GPU failure to learn at #242); a released schedule refuses a terminator.
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
4. **The release works.** A released schedule, spent, leaves the request
   generating; asserted in core on the mock, and visible end to end as
   arguments that the schedule never permitted.
5. **The refusals.** Thinking-on, an unknown function name, a literal that
   does not encode, and a cap shorter than the opener each answer `400`
   with the documented code and `param`.
6. **Speculation on.** Acceptance 1 and 2 hold with `--spec dflash2`.
7. **Unchanged.** `"auto"` and `"none"` requests produce the same tokens as
   before this spec, greedy, against a recorded fixture.
8. **Docs.** The OpenAPI operation description says which `tool_choice`
   values are served and names the thinking exclusion; `CONTEXT.md` gains
   the released-schedule term.
9. `cargo test` passes workspace-wide.

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
- **Prior art in this repo**: `scalar` (spec decide/10, #255) already forces
  a literal prefix and then constrains; the only thing it does not do is
  release.
