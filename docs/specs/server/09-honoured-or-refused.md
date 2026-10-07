# server 09 — Every field is honoured, refused, or declared inert

GitHub: #284

## Problem Statement

`ChatCompletionsRequest` (`crates/server/src/api.rs:1090`) carries no
`deny_unknown_fields` and its only `#[serde(flatten)]` is the named
`SamplingRequestFields`. Every OpenAI field it does not declare is therefore
**dropped in silence**: `stop`, `n`, `logprobs`, `top_logprobs`,
`response_format`, `logit_bias`, `max_completion_tokens`, and the rest of the
current Chat Completions body.

That contradicts the contract this surface gave itself at #101, restated in
`resolve_class` (`api.rs:475`): an extension is "honoured or refused, never
silently dropped". Three of the dropped fields are not cosmetic:

- **`max_completion_tokens`** is what the current OpenAI SDKs send *instead
  of* `max_tokens`. A client that sends only it gets a request with **no
  generation cap at all** — bounded by EOS, the context ceiling and
  `--request-timeout`, nothing else. With the thinking budget on (spec 08)
  it is the difference between a turn that closes and a turn that reasons to
  the ceiling.
- **`stop`** is in every client's retry and agent-loop code. Dropping it
  means the caller's own stop condition never fires and it pays for the tail.
- **`response_format: {"type": "json_object"}`** tells the caller it will get
  JSON. It gets prose, and finds out by failing to parse.

A second, smaller untruth sits next to them: `Usage` on chat (`api.rs:1402`)
is three counters. `usage.prompt_tokens_details.cached_tokens` exists on
`/v1/responses` (`responses/events.rs:112`) and not on chat — so the engine's
own cross-request reuse, its most expensive feature, is invisible on the
route most clients use. Codex and Claude-Code-shaped clients read exactly
that field to report cache hits.

## Solution

**Every field of the current OpenAI Chat Completions body is classified, by
name and by value, into one of three columns, and the classification is the
wire contract.**

1. **Honoured** — the server does what the field asks.
2. **Inert** — the value asks for a behaviour this server already has, so it
   is accepted and changes nothing. Declared as inert in the OpenAI
   document, never silently dropped-and-forgotten.
3. **Refused** — the value asks for a behaviour this server does not have.
   `400` with the OpenAI error shape, `param` naming the field, and a message
   that says what to use instead.

The distinction between 2 and 3 is **by value, not by field**: `n: 1` is
inert and `n: 2` is refused; `logprobs: false` is inert and `true` is
refused. A client that sends the defaults its SDK fills in keeps working; a
client that asks for something real gets told.

**No `deny_unknown_fields`.** The enumerated side is the finite one — the
OpenAI field list — and a field outside it stays ignored, as today. Refusing
the open-ended side would break on the next field OpenAI adds.

Two fields move into column 1 in this spec (`stop`,
`max_completion_tokens`), and chat's `usage` gains
`prompt_tokens_details.cached_tokens`. Everything else in column 3 is
refused, each message pointing at the ticket that would move it to column 1.

The same table is applied to `/v1/responses` where the two bodies differ
(`responses/mod.rs:134` already refuses four things by name; this spec makes
that list complete rather than exemplary).

## User Stories

1. As a client on a current OpenAI SDK, I want `max_completion_tokens`
   honoured, so my turn has the cap I set instead of none.
2. As an agent loop, I want `stop` honoured, so I stop paying at my own
   boundary.
3. As a client asking for JSON, I want a `400` rather than prose, so I fail
   at the request instead of at the parse.
4. As a client whose SDK fills in `n: 1`, `store: false`,
   `parallel_tool_calls: true`, I want those accepted, so a default-shaped
   body is not an error.
5. As an operator, I want chat's `usage` to report the prompt this request
   resumed instead of prefilling, so a cache hit is visible on the route my
   clients use.
6. As a reader of the README, I want the things this server deliberately
   does not serve named as choices, so I stop looking for them.

## Implementation Decisions

### The table

Fields already honoured are listed for completeness. `param` is the OpenAI
error's `param` value.

| Field | Inert when | Refused when | Refusal points at |
|---|---|---|---|
| `model`, `messages`, `stream`, `stream_options` | — | as today | — |
| `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `seed` | — | as today (range) | — |
| `max_tokens` | — | as today | — |
| **`max_completion_tokens`** | **honoured** (below) | present *and* `max_tokens` present *and* different | — |
| **`stop`** | **honoured** (below) | not a string, or not an array of 1..=4 non-empty strings | — |
| `tools`, `tool_choice` | — | `tool_choice` other than `"auto"`, `"none"`, `"required"` or a named function among `tools`; `"required"` with no tools; a forcing value the tokenizer cannot encode, or a cap shorter than the forced opening | #286 (`required` and a named function are **honoured**: the call's opening is forced, spec server/11) |
| `reasoning_effort` and the ignis thinking controls | — | as today | — |
| `n` | absent, `null`, or `1` | `>= 2` | #289 |
| `logprobs` | absent, `null`, or `false` | `true` | #288 |
| `top_logprobs` | absent or `null` | present | #288 |
| `response_format` | absent, `null`, or `{"type": "text"}` | `json_object`, `json_schema`, any other type | #287 |
| `logit_bias` | absent, `null`, or `{}` | any non-empty map | #288 (same seam) |
| `parallel_tool_calls` | absent, `null`, or `true` | `false` | — (this template emits calls as they close) |
| `store` | absent, `null`, or `false` | `true` | — (this server stores no completions) |
| `service_tier` | absent, `null`, `"auto"`, `"default"` | `"flex"`, `"priority"`, `"scale"` | — (one card, one tier) |
| `modalities` | absent, `null`, or `["text"]` | anything else | — (no audio out) |
| `metadata`, `user`, `prompt_cache_key`, `safety_identifier`, `verbosity` | always inert | — | — |
| `audio`, `prediction`, `web_search_options` | — | present and not `null` | — |
| `functions`, `function_call` | — | present and not `null` | use `tools` / `tool_choice` |
| `best_of`, `echo`, `suffix`, `prompt` | — | present and not `null` | Completions-only; use `messages` |

`prompt_cache_key` is inert **and stays inert**: reuse here is by content
(ADR 0029) and a key that changed nothing would be a promise not kept. It is
named in the table so the next reader does not wire it.

An inert field is not merely tolerated: the OpenAI document's description for
the operation lists the inert set, and a machine-readable copy of the three
columns lives in one Rust table that both the validator and the document
read, so they cannot drift.

### `max_completion_tokens`

- An alias of `max_tokens`, resolved before `SamplingRequestFields::resolve`,
  with the same type and range rules.
- It counts **every generated token, reasoning included**, exactly as
  `max_tokens` does today. The thinking budget (spec 08) keeps leaving the
  answer its room inside whichever of the two was given.
- Both present and equal: fine. Both present and different: `400`,
  `param: "max_completion_tokens"`. This is an ignis rule — OpenAI treats
  `max_tokens` as deprecated-but-accepted — and the message says so, because
  a body carrying two different caps has no answer that is not a guess.

### `stop`

- A string, or an array of 1 to 4 non-empty strings (OpenAI's cap). An empty
  string, an empty array, a non-string element, or more than 4 is a `400`.
- Matched over the **content channel only** (`Channel::Content`,
  `decoder.rs:28`). A reasoning block is never cut by a caller's stop
  sequence: the thinking controls own that channel and its close (spec 04,
  spec 08).
- The matched sequence **is not emitted** — neither in the non-streaming
  `content` nor as a delta — and the text before it is. `finish_reason` is
  `"stop"`, as OpenAI reports it.
- **Streaming holds back a possible prefix.** Before emitting a content
  delta, the tail that is a proper prefix of any stop sequence is buffered
  until the next delta resolves it, the same discipline `toolcall.rs` already
  applies to `<tool_call>`. A sequence split across chunk boundaries can
  therefore never be missed, and no fragment of a stop sequence ever reaches
  the client.
- **Inside a buffered tool-call block, stop does not fire.** The tool-call
  scanner owns that span; a `stop` string that happens to occur inside a
  call's arguments must not truncate the call.
- Matching is over decoded text, byte-exact, not over token ids: a stop
  sequence that does not align to a token boundary still fires.

### `usage.prompt_tokens_details.cached_tokens`

- Chat's `Usage` gains `prompt_tokens_details: { cached_tokens }`, always
  serialized (a `0` is information: nothing was reused).
- The value is the same quantity `/v1/responses` reports in
  `input_tokens_details.cached_tokens` — the prompt tokens this request
  resumed from retained state instead of prefilling — read from the same
  place, not recomputed.
- Present in the non-streaming body, and in the usage-only chunk that
  `stream_options.include_usage` asks for.

### What the server does not serve

A short section in `README.md` and in the OpenAPI document's top-level
description names the things this engine does not serve **as choices**, so a
reader stops looking: embeddings and a generic reranker over a second model,
`/v1/completions`, `/v1/batches`, audio in or out, LoRA adapters on an NVFP4
checkpoint, and more than one loaded model. Each with its one-line reason.
This is documentation, not a refusal path — those routes simply 404 as they
do today.

### What does not change

- No `deny_unknown_fields` anywhere. An unknown field is ignored.
- Every field already honoured keeps its current parse, range and error.
- The generated text of any request that is accepted today is unchanged: a
  request without `stop` and without `max_completion_tokens` decodes exactly
  as before, bit for bit.

## Testing Decisions

- **The table is the test.** One CPU table-driven test per row, over the
  request validator: an inert value is accepted and observably changes
  nothing, a refused value answers `400` with the right `param` and a message
  containing the pointer.
- **`stop`, on the mock engine**: the sequence is absent from the answer; it
  fires when split across two deltas; it does not fire inside a
  `<tool_call>` block; it does not fire on reasoning text; four sequences
  work and five are refused; a non-aligned sequence fires.
- **`max_completion_tokens`**: alone it caps; equal to `max_tokens` it is
  accepted; different it is `400`; with the thinking budget on, the budget
  leaves room inside it.
- **`cached_tokens`**: over the mock, a request that reuses reports a nonzero
  value in the body and in the usage chunk; one that does not reports `0`;
  the value equals what `/v1/responses` reports for the same prompt.
- **The document**: `crates/server/tests/openapi_http.rs` gains an assertion
  that every refused field is named in the operation description, driven off
  the same table, so a new refusal cannot be added without documenting it.

## Acceptance

1. **No silent drop.** For every field in the table, a request carrying it is
   either honoured, accepted as inert, or refused with a `400` naming it in
   `param`. Asserted row by row.
2. **No client this repo talks to breaks.** Bodies captured from the
   clients that reach this server today — qwen-code, opencode/Codex, the
   Playground, and the OpenAI Python SDK's defaults — are all served
   unchanged. **This acceptance comes first and it can edit the table**: a
   value a real client sends is *inert*, never *refused*, whatever this spec
   guessed. `parallel_tool_calls: false` and `store: true` are the two most
   likely to move, and the recorded agent trace is the fixture.
3. **A default-shaped SDK body is accepted.** A body carrying `n: 1`,
   `store: false`, `parallel_tool_calls: true`, `logprobs: false`,
   `response_format: {"type":"text"}`, `metadata`, `user` and
   `service_tier: "auto"` is served exactly as the same body without them.
4. **`max_completion_tokens` caps.** A request carrying only it stops at that
   many generated tokens with `finish_reason: "length"`; carrying both with
   different values it answers `400`.
5. **`stop` stops.** Every case in Testing Decisions holds, and the matched
   sequence never appears in the response.
6. **Reuse is visible on chat.** `usage.prompt_tokens_details.cached_tokens`
   is present on every chat response, streaming and not, and reports the same
   quantity `/v1/responses` reports for the same prompt.
7. **Unchanged output.** A request that neither sets `stop` nor
   `max_completion_tokens` produces the same tokens as before this spec,
   greedy, asserted against a recorded fixture.
8. **Docs.** The OpenAPI operation description lists the inert set and every
   refusal; `README.md` carries the "what this server does not serve"
   section.
9. `cargo test` passes workspace-wide.

## Out of Scope

- Making any refused field work. Each has its own ticket: #287 (structured
  output), #288 (logprobs and `logit_bias`), #289 (`n > 1`), #286 (forcing a
  tool call).
- `/v1/tokenize` and the token count: #285.
- Wiring `prompt_cache_key` to anything.
- Streaming tool-call argument deltas (#290): this spec only keeps `stop`
  from interfering with the whole-call contract of #121.

## Further Notes

- **Why not `deny_unknown_fields`**: it inverts the risk. The field list
  OpenAI publishes is finite and this spec enumerates it; the set of fields a
  proxy, a gateway or a future SDK version adds is not. Refusing the
  open-ended side would turn every upstream addition into an outage.
- **Why refuse rather than ignore `response_format: json_object`**: a caller
  that asks for JSON writes a parser. Prose that fails to parse is a worse
  answer than a `400` it can branch on.
- **Why `stop` is content-only**: the reasoning channel's boundaries belong
  to the template and the budget. A caller's `"\n\n"` would otherwise cut the
  model's thinking at its first paragraph.
