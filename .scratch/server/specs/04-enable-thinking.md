# 04 — `enable_thinking` and Qwen 3.8 reasoning effort on the OpenAI surface

GitHub: #68

## Problem Statement

Qwen 3.8-27B is a thinking model. Its chat template decides, at prompt-render
time, whether the assistant is asked to think and how hard: the template reads
`enable_thinking` and `reasoning_effort`, emits either an open `<think>` block
or a pre-closed empty one, and prepends a reasoning-effort instruction to the
system turn. Ignis binds neither variable, so every request runs with thinking
on at the template's `xhigh` default. There is no way for a client to turn it
off or turn it down.

Three consequences, all visible to whoever is driving the server:

1. **Every answer is preceded by a thinking trace, and the trace is returned as
   the answer.** Ignis decodes the whole generated token stream into
   `message.content`, so `content` contains the model's raw reasoning followed
   by a literal `</think>` marker and only then the actual reply. A client that
   renders `content` shows the user the model's scratchpad. The reference
   engine returns the reasoning separately, in `reasoning_content`, and leaves
   `content` as the answer alone.
2. **Short requests return nothing usable.** With thinking on, a `max_tokens`
   budget in the tens is consumed entirely by the thinking trace, so the reply
   never arrives. A caller who wants a one-line answer has no way to ask for
   one.
3. **The G1 canary oracle cannot be evaluated.** The canary oracle was recorded
   from the reference with thinking off. Ignis cannot reproduce that prompt, so
   the comparison measures a template difference rather than engine agreement.
   This was established while diagnosing #67: binding `enable_thinking` to
   `false` as a throwaway probe took `rust-hello` to 21/21 and `math-greedy` to
   32/32 against the oracle, from 0%. Without the feature, the gate is
   unmeasurable.

Clients written against the reference engine — including the owner's own agent
orchestration — send `enable_thinking` and `reasoning_effort` today. Ignis
ignores both silently, which is worse than rejecting them: the request looks
like it succeeded.

## Solution

Ignis accepts the reference's thinking controls on the OpenAI surface and
honours them, and it separates the model's reasoning from its answer in both
response modes.

A caller can:

- turn thinking off (`enable_thinking: false`, or `reasoning_effort: "none"`)
  and get a direct answer within a small token budget;
- turn thinking down (`reasoning_effort: "low"`) or leave it at the model's
  default (`"xhigh"`);
- read the reasoning trace from `message.reasoning_content` and the answer from
  `message.content`, never mixed;
- receive the same separation while streaming, as `delta.reasoning_content`
  followed by `delta.content`, with the `</think>` marker never leaking into
  either channel even when it straddles a token boundary;
- send an unsupported effort and get a specific, actionable 400 rather than
  silence.

An operator can set the server-wide default for both controls, so a deployment
can run thinking-off by default without every client opting out.

The wire contract is the reference's, field for field, so a client written
against ninfer works against ignis unchanged.

## User Stories

1. As an API client, I want to send `enable_thinking: false` on a chat
   completion, so that the model answers directly instead of thinking first.
2. As an API client, I want to send `enable_thinking: true` explicitly, so that
   I get thinking behaviour regardless of the server's configured default.
3. As an API client, I want to omit `enable_thinking` entirely, so that the
   server's configured default applies and my request stays simple.
4. As an API client, I want to send `enable_thinking` inside
   `chat_template_kwargs`, so that clients speaking the llama.cpp webui dialect
   work against ignis without modification.
5. As an API client, I want a request that sets `enable_thinking` both at top
   level and inside `chat_template_kwargs` with the *same* value to be
   accepted, so that a client that sets both defensively is not punished.
6. As an API client, I want a request that sets those two to *conflicting*
   values to be rejected with a 400 naming the conflict, so that I find the bug
   in my client rather than silently getting one of the two.
7. As an API client, I want `enable_thinking: null` to be treated as unset, so
   that a client serialising an absent optional as null gets the server
   default.
8. As an API client, I want a non-boolean `enable_thinking` rejected with a 400
   naming the field, so that a type error in my client is obvious.
9. As an API client, I want to send `reasoning_effort: "low"`, so that the model
   thinks briefly and reaches its conclusion without elaboration.
10. As an API client, I want to send `reasoning_effort: "medium"`, so that I get
    thinking with no additional effort instruction in the system turn.
11. As an API client, I want to send `reasoning_effort: "xhigh"`, so that I get
    the model's most careful reasoning, matching the template's default.
12. As an API client, I want `reasoning_effort: "none"` to disable thinking
    entirely, so that I can express the whole control through one field.
13. As an API client, I want to send an effort the loaded template cannot
    honour (`minimal`, `high`, `max`) and receive a 400 that says the template
    does not support it, so that I can distinguish "you typed something
    invalid" from "this model cannot do that".
14. As an API client, I want a `reasoning_effort` outside the protocol
    vocabulary entirely to be rejected with a 400 listing the accepted values,
    so that I can correct a typo without reading the source.
15. As an API client, I want `reasoning_effort` and `enable_thinking` that
    disagree (for example `"none"` with `true`) to be rejected with a 400, so
    that the server never has to guess which one I meant.
16. As an API client, I want the model's reasoning in `message.reasoning_content`
    and its answer in `message.content`, so that I can show the answer to a user
    and keep the trace for debugging.
17. As an API client, I want `content` to contain no `<think>` or `</think>`
    markers, so that I never have to strip them myself.
18. As an API client, I want a response with thinking disabled to carry no
    `reasoning_content` field at all, so that the absence of reasoning is
    unambiguous.
19. As an API client, I want a generation that hits its token budget mid-thought
    to return the partial reasoning in `reasoning_content` and an empty
    `content`, so that I can tell the model ran out of budget while thinking.
20. As a streaming API client, I want reasoning deltas to arrive as
    `delta.reasoning_content` and answer deltas as `delta.content`, so that I
    can render a "thinking…" indicator and swap to the answer when it starts.
21. As a streaming API client, I want the `</think>` marker never to appear in
    any delta, even when the marker is split across two tokens, so that my
    rendered output is never corrupted by a transport artefact.
22. As a streaming API client, I want a multi-byte character split across two
    tokens to arrive as one whole character, so that streamed text is valid
    UTF-8 at every delta rather than only in aggregate.
23. As a streaming API client, I want a stream that ends mid-marker to flush the
    held bytes correctly — publishing ordinary text, never a partial marker —
    so that a truncated stream degrades gracefully.
24. As a streaming API client, I want the non-streaming and streaming responses
    to the same request to carry the same reasoning and content text, so that I
    can switch modes without changing my parsing.
25. As an API client using `/v1/responses`, I want the same thinking controls
    accepted there, so that the two endpoints do not disagree about what a
    request means.
26. As an API client using `/v1/responses`, I want the reasoning excluded from
    the response `text`, so that the field holds the answer and nothing else.
27. As an agent author, I want to send a prior assistant turn carrying
    `reasoning_content`, so that a multi-turn conversation can replay what the
    model previously thought.
28. As an agent author, I want prior assistant reasoning dropped from the prompt
    by default, so that a long conversation does not accumulate context I did
    not ask to keep.
29. As an agent author, I want to send `preserve_thinking: true` to keep prior
    reasoning in the rendered prompt, so that I can opt into the more expensive
    behaviour deliberately.
30. As an operator, I want to set the server-wide default for thinking through
    the environment, so that a deployment can run thinking-off without changing
    any client.
31. As an operator, I want to set the server-wide default reasoning effort
    through the environment, so that I can tune cost against quality for a whole
    deployment.
32. As an operator, I want an invalid environment default to fail the server
    start with a clear message, so that a typo is caught at boot rather than on
    the first request.
33. As an operator, I want the server to refuse to start if the loaded
    artifact's template cannot honour the configured default, so that a
    model swap does not silently change behaviour.
34. As a benchmark author, I want `ignis-bench oracle record` and
    `oracle compare` to drive the endpoint with thinking disabled, so that the
    G1 canary comparison measures engine agreement rather than a template
    difference.
35. As a benchmark author, I want the recorded candidate fixture to contain the
    answer tokens only, so that it is directly comparable to the reference
    fixture recorded the same way.
36. As a maintainer, I want the set of efforts the loaded template supports to
    be determined from the template itself, so that swapping in a model with a
    different vocabulary does not require a code change.
37. As a maintainer, I want a template that cannot disable thinking to produce a
    capability error rather than a silently-ignored request, so that the
    limitation is visible.
38. As a maintainer, I want the thinking options to travel through the existing
    template seam rather than a parallel path, so that there remains exactly one
    place where the OpenAI surface and the tokenizer meet.
39. As a maintainer, I want the reasoning/content split rule to match the
    reference's exactly, so that a divergence in output is a real divergence and
    not a parsing difference.
40. As a maintainer, I want the incremental decoder to be a directly testable
    unit, so that byte-boundary behaviour can be pinned without a GPU.
41. As a maintainer, I want the streaming path to stop decoding each token in
    isolation, so that the existing SSE corruption bug is fixed by construction
    rather than patched.
42. As a developer running the CPU gate, I want every wire-contract case covered
    without the GPU, so that `cargo test` stays fast and green on a busy
    machine.
43. As a developer, I want one GPU end-to-end case proving a thinking-disabled
    request returns a real answer from the real model, so that the CPU coverage
    is anchored to reality.

## Implementation Decisions

### The prompt-options value

A single value type carries the resolved thinking semantics from the API
boundary to the template. It holds whether thinking is enabled, the reasoning
effort (absent meaning "let the template default apply"), and whether prior
assistant reasoning is preserved. It is constructed once per request, after
validation, and is the only thing that crosses the template seam alongside the
messages.

### The template seam

`TemplateProvider::apply_chat_template` gains the options value as a second
parameter. This is the existing and only seam between the OpenAI surface and
the tokenizer (`crates/server`'s template module documents it as such), and the
feature must not open a second one. Both implementations — the built-in
placeholder and the artifact-backed provider — are updated. The placeholder
ignores the options except to keep them observable for tests.

Underneath, the artifact crate's chat-template renderer gains a variant that
binds template variables in addition to `messages` and `add_generation_prompt`.
The existing no-options render remains as the trivial delegation, so callers
that do not care are unaffected.

The variables bound are `enable_thinking` (always bound, never left undefined)
and `reasoning_effort` (bound only when an effort was resolved). Binding
`enable_thinking` unconditionally is deliberate: the Qwen 3.8 template branches
on `enable_thinking is undefined or enable_thinking is true`, so leaving it
undefined and leaving it `true` are the same thing, and being explicit removes a
class of "which default won" ambiguity.

### Template capability probing

The efforts a template supports are discovered from the template, not
hard-coded. At artifact load the renderer is probed once with a minimal
conversation: once with thinking disabled, and once per protocol effort value.
The Qwen 3.8 template raises `Unexpected reasoning effort …` for a value it does
not know, so a probe that raises marks that effort unsupported and a probe that
renders marks it supported. The resulting capability set is carried with the
frontend set and consulted during request resolution.

For Qwen 3.8 this yields: thinking can be disabled; supported efforts are `low`,
`medium`, and `xhigh`. Note that `medium` is the template's neutral level — it
enables thinking but adds no effort instruction to the system turn, unlike
`low` and `xhigh` which each prepend a sentence.

### Wire contract

Mirrors the reference exactly.

`enable_thinking` is accepted as an optional boolean at the top level and under
`chat_template_kwargs`. `null` means unset. Both present with different values
is a 400 naming the conflict. Neither present means the server default applies.

`preserve_thinking` follows the identical shape and the identical conflict rule.

`chat_template_kwargs` accepts only `enable_thinking` and `preserve_thinking`;
any other key is a 400 identifying the unsupported key, so a client cannot
believe it configured something it did not.

`reasoning_effort` is accepted as an optional string over the full protocol
vocabulary — `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` — and is
then resolved against the probed capability set:

| Requested | Resolution |
| --- | --- |
| `none` | thinking disabled; a template that cannot disable thinking is a capability error |
| `low`, `medium`, `xhigh` | thinking enabled, effort passed to the template |
| `minimal`, `high`, `max` | capability error: not supported by the loaded template |
| anything else | validation error listing the accepted values |

A `reasoning_effort` that implies a different thinking state than an explicit
`enable_thinking` is a 400 naming the conflict. This is checked before
capability resolution so the error reports the client's mistake rather than a
model limitation.

The two error classes are distinguishable by the machine-readable code on the
error body — a validation failure and a capability failure are different
problems with different fixes.

### Server defaults

Two environment variables, following the existing `IGNIS_`-prefixed convention
used by the server binary: one for the default thinking state and one for the
default effort. Both are parsed at startup, and both are validated against the
loaded artifact's probed capabilities before the server binds its port — a
default the model cannot honour is a refused start with a descriptive message,
matching how the server already treats a missing EOS token or an unclean
checksum.

### Response separation — the split rule

Reasoning is the text between the last `<think>` and the first `</think>`, with
surrounding newlines trimmed; content is everything after the last `</think>`,
with leading newlines trimmed. When there is no `</think>` at all, the entire
output is content and reasoning is empty. This is the reference's rule and is
adopted verbatim, including the deliberate asymmetry between *first* and *last*
close marker, so that identical model output produces identical field values on
both engines.

Empty reasoning is omitted from the response rather than serialised as an empty
string, so a thinking-disabled response is structurally distinct from a
thinking-enabled one that produced no trace.

### The incremental output decoder

The streaming path stops calling the whole-sequence render on each token
individually. That approach is doubly wrong: it cannot split a multi-byte
character across tokens, and it cannot recognise a `</think>` marker that spans
them.

In its place, a per-request decoder owns the generated-byte stream and emits
`(channel, text)` deltas, where the channel is reasoning or content. Its
contract:

- It decodes incrementally, holding an incomplete UTF-8 sequence until the
  bytes that complete it arrive.
- It holds back any trailing bytes that could be a prefix of `</think>`,
  releasing them as ordinary text once a later token proves they are not.
- It never emits the marker itself on either channel.
- It switches channel exactly once, at the marker.
- On finish it flushes: held bytes that are a genuine partial marker are
  dropped, held bytes that are ordinary text are published, and an incomplete
  UTF-8 sequence is published as the replacement character rather than as
  invalid bytes.

The decoder is a plain value with `push(tokens) -> deltas` and `finish() ->
deltas`, independent of the transport. The non-streaming path feeds it the whole
token list and takes the concatenation per channel, which is what makes story 24
— streaming and non-streaming agreeing — true by construction rather than by
duplicated logic.

This subsumes the known SSE per-token decoding defect; that bug needs no
separate fix once this lands.

### Multi-turn reasoning

Inbound assistant turns may carry `reasoning_content`. The artifact crate's
message type already models it and the message-to-template conversion already
forwards it; the server's own message type gains the optional field and passes
it through. When `preserve_thinking` resolves false — the default — inbound
assistant reasoning is dropped before rendering, so a long conversation does not
silently accumulate traces.

### `/v1/responses`

Accepts and resolves the same options through the same code path. Its response
shape has no reasoning field, so the reasoning is discarded and `text` carries
the content channel only. This is an explicit decision, not an oversight:
`text` holding a raw thinking trace is the bug being fixed, and inventing a
non-standard field on this endpoint would be worse than omitting the trace.

### Benchmark harness

The oracle recorder drives the endpoint with thinking disabled, so a recorded
candidate fixture and the reference fixture describe the same prompt. This is
what makes the G1 comparison meaningful; it is a change to how the harness calls
the endpoint, not to the fixture format.

## Testing Decisions

A good test here asserts what a client can observe: the status code, the
response fields, the SSE frames, and — through a recording double — the options
that reached the template. It does not assert on the shape of internal values,
the number of times the decoder was called, or which module resolved a default.
The wire contract is the behaviour; everything else is free to change.

### Primary seam — the HTTP boundary

`crates/server/tests/openai_http.rs` is the primary seam and the prior art: it
drives the real axum router over a mock-compute engine, entirely on the CPU
gate. Every wire-contract story belongs here — acceptance, the `chat_template_kwargs`
dialect, every 400 (conflicts, unsupported keys, bad types, unknown efforts,
capability errors), the non-streaming `reasoning_content` / `content` split, and
the SSE channel sequence.

To make the resolved options observable at this seam without reaching inside
the server, the harness gains a recording `TemplateProvider` — a test double
that captures the options it was handed and returns the placeholder's tokens.
This is a test-only implementation of an existing public trait, so it adds no
production seam. The mock compute engine already emits a chosen token stream,
which is what lets a test drive a `</think>` marker split across two tokens
through the whole router and assert on the resulting SSE frames.

### Real-template rendering

`crates/artifact/tests/real_frontend.rs` is the prior art and the home for
assertions about what the actual Qwen 3.8 jinja produces: that thinking-disabled
renders the pre-closed `<think>\n\n</think>\n\n` block and thinking-enabled
renders the open `<think>\n`; that `low` and `xhigh` each prepend their effort
sentence while `medium` prepends none; that an unsupported effort raises; and
that the probed capability set is exactly thinking-disable plus `low`, `medium`,
`xhigh`. These tests skip when the artifact is absent, per the existing
machine-local convention.

### Incremental decoder

Unit-tested directly in the module that owns it. This is the one place a new
seam is justified: the contract is byte-level, and a table of
(token-boundary split, expected deltas) cases is the only way to cover marker
straddling, multi-byte straddling, a marker at the very first or very last
token, no marker at all, a marker split three ways, and each flush path. Prior
art for table-driven unit tests inside the crate: the template and
artifact-template modules' own test blocks.

### GPU end-to-end

`crates/server/tests/openai_http_gpu.rs` is the prior art. One case, under the
GPU profile per ADR 0006 — a thinking-disabled request against the real model
returns non-empty `content`, no `reasoning_content`, no think markers, and valid
UTF-8. This anchors the CPU coverage to the real tokenizer and the real
template. Per `docs/agents/testing.md`, it must fail rather than skip on a busy
GPU or a kernel error.

### The gate

`cargo test` workspace-wide stays green and CPU-only. The GPU case runs under
the explicit profile.

## Out of Scope

- **Meeting the G1 ≥95% oracle floor.** This feature makes the gate
  *measurable*; it does not make it pass. With thinking disabled, two of four
  canaries currently agree token-for-token and two diverge at token 0 while
  still producing correct, fluent answers. That residual divergence is a
  separate investigation and must not be folded in here.
- **Tool calling.** The template's `tools` branch interacts with the reasoning
  instruction block, but tool support is not in ignis's OpenAI surface today and
  is not added by this spec.
- **The Anthropic-dialect surface.** The reference maps a `thinking` object onto
  the same semantics; ignis exposes no Anthropic endpoint, so there is nothing
  to map.
- **Advertising thinking capability on `/v1/models`.** The reference publishes a
  chat-template capability marker so webui clients can probe for thinking
  support. Worth doing, but it is a discovery concern and can follow.
- **Per-request reasoning token accounting.** Splitting the usage counts into
  reasoning and answer tokens is a separate contract.
- **Vision, MTP, and speculative decoding interactions.** Untouched.
- **Changing the `unit_offset` fix from #67.** Landed separately; this spec
  depends on it only in that the model must produce coherent text for any of
  this to be observable.

## Further Notes

The reference implementation lives in the ninfer tree and is the authority for
every contract detail: the request parsing and its conflict rules, the effort
resolution against template capabilities, and the reasoning/content split with
its streaming marker-holding logic. Where this spec and that implementation
appear to disagree, the reference wins and the spec is wrong — the whole point
of matching field-for-field is that a client cannot tell the two engines apart
at the wire.

The `medium` level deserves a note for whoever implements the capability probe:
it is supported by the template but produces *no* reasoning instruction, so a
probe that decides "supported" by looking for injected instruction text will
wrongly reject it. Support must be decided by whether the render raises, not by
whether the output changed.

The relationship to #67 is worth recording. The `unit_offset` bug and this
missing feature presented as one symptom — a canary oracle at 0% agreement —
and only separated once the numerics were fixed. The lesson for the test suite
is that the canary gate could not distinguish "the model is computing garbage"
from "we are asking the model a different question", and both failure modes
scored identically. Whoever implements this should consider whether the gate can
be made to report those two causes differently.
