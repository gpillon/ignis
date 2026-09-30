# server 10 — The prompt counted without being served

GitHub: #285

## Problem Statement

An agent decides what to send by how much it costs. It trims history, drops
tool results, and chooses between a summary and the raw text, and to do any
of that it needs the number of tokens a body *would* prefill — before
spending a lane on it.

Today the only way to learn that number from ignis is to send the request and
read `usage.prompt_tokens` off the answer. That costs a full prefill, a lane,
and the generation the caller did not want. The consequence is that clients
count with their own copy of a tokenizer, guess a margin, and get the chat
template's own tokens wrong — the tool block, the thinking controls, the
system block the server assembles are all invisible to a client-side count,
and they are exactly the part an agent cannot predict.

Every engine an agent might otherwise talk to serves this: vLLM
(`/tokenize`, `/detokenize`), TGI (`/tokenize`), llama.cpp
(`/tokenize`, `/detokenize`). It is not in the OpenAI surface, which is why
it is missing here.

## Solution

**Two routes that run the request's own render path and stop before
submitting it.**

- `POST /v1/tokenize` — takes the body of a chat completion (or a raw
  `prompt` string), renders it exactly as `/v1/chat/completions` would, and
  answers the token count, optionally the token ids, and the server's
  context ceiling.
- `POST /v1/detokenize` — token ids back to text, through the same
  tokenizer.

Neither touches the scheduler, the GPU or a KV page. `build_request`
(`api.rs:282`) is already the one shared render path for both completion
endpoints; these routes call it and return instead of submitting. That is
what makes the count **exact rather than an estimate**: the same template,
the same tool block, the same thinking controls, the same system block.

## User Stories

1. As an agent, I want the exact prompt-token count of a body before I send
   it, so I can trim history to fit the context instead of guessing.
2. As an agent offering tools, I want the tool block counted, so I know what
   my tool definitions cost per turn.
3. As a client, I want the server's context ceiling in the same answer, so I
   do not need a second call or a flag I cannot read.
4. As a debugger, I want the token ids and the rendered text, so I can see
   what the template actually built.
5. As an operator, I want this to cost no lane and no GPU, so a client
   polling it cannot displace a conversation.

## Implementation Decisions

### The routes

Both are registered through `routes!` in `v1_parts` (`api.rs:140`), not
through a bare `.route()`, so `crates/server/tests/openapi_http.rs` sees them
in the path set and the OpenAPI document carries them. Both sit behind the
API key and the CORS preflight like every other `/v1` path.

### `POST /v1/tokenize`

Request: **either** a chat body **or** a raw prompt, never both.

- Chat form: `messages` (required), plus the fields that change the render —
  `model`, `tools`, `tool_choice`, and the thinking controls
  (`enable_thinking`, `reasoning_effort`, `preserve_thinking`,
  `chat_template_kwargs`, `thinking_budget`). Parsed and validated by the
  same code `/v1/chat/completions` uses, so a body that would be refused
  there is refused here with the same error.
- Raw form: `prompt` (a string), tokenized with no template applied at all.
- `return_token_ids` (default `false`): include the ids.
- `return_text` (default `false`): include the rendered prompt text.
- Sampling fields (`temperature`, `max_tokens`, ...) are **inert** here, by
  the rule of spec 09: they do not change a render. `stream` is refused —
  there is nothing to stream.

Response:

```json
{
  "count": 1234,
  "max_model_len": 262144,
  "token_ids": [ ... ],      // only with return_token_ids
  "text": "<|im_start|>..."  // only with return_text
}
```

`count` is the length of the token vector `build_request` produced —
**the same number the same body would report as `usage.prompt_tokens`**.
`max_model_len` is the server's `--max-context`, so a client can decide with
one call.

### `POST /v1/detokenize`

`{"token_ids": [...]}` answers `{"text": "..."}`. Ids outside the
vocabulary are a `400` naming the first offending index; an empty array
answers an empty string.

### Media is refused, not counted

A `messages` array carrying an `image_url` part answers `400`
`media_not_countable`. An image's token cost is its **grid after
preparation** (spec vision/#176): the server would have to fetch, decode and
resize the picture to know it, which is a network fetch and a real cost on a
route whose whole promise is that it is free. Fail closed and say so, rather
than return a number that is right for one resize policy and wrong for the
next.

The refusal message names what the caller can do instead: send the request.

### Cost and admission

- No `RequestInput` is submitted, no sequence is allocated, no retained state
  is read or published, no media is acquired.
- The route is **not** metered as a request in the Prometheus request
  lifecycle: it never enters the scheduler, so there is no lane, no TTFT and
  no completion to report. A single counter,
  `ignis_tokenize_requests_total{route="tokenize"|"detokenize"}`, is enough
  to see it being used.
- The render itself is CPU work proportional to the prompt; the handler runs
  it on the blocking pool the way the completion path renders today, so a
  100K-token body cannot stall the async runtime.

### What does not change

- No existing route, no rendered byte of any prompt, no scheduler path.
- The tokenizer is the artifact's, the only one in the process.

## Testing Decisions

- **The count is the usage count.** Over the mock engine: for a body with
  and without tools, with thinking on and off, `count` from `/v1/tokenize`
  equals `usage.prompt_tokens` from `/v1/chat/completions` for the same
  body. This is the test that keeps the two paths from drifting.
- **Round trip.** `tokenize(return_token_ids) -> detokenize` returns the
  rendered text byte for byte, for an ASCII prompt, a prompt with multi-byte
  UTF-8, and a prompt ending mid-grapheme.
- **Refusals**: both forms present; neither present; `messages` empty; an
  `image_url` part; `stream: true`; an out-of-range id on detokenize. Each
  with its code.
- **The raw form** applies no template: `prompt` of a known string gives the
  ids the tokenizer gives it directly.
- **The document**: the two paths appear in `openapi_http.rs`'s asserted
  path set, with their schemas.
- No GPU test: the routes never reach the leaf.

## Acceptance

1. **Exactness.** For every body in the fixture set (plain, with tools, with
   thinking on, with a long system block), `POST /v1/tokenize` `count`
   equals the `usage.prompt_tokens` the same body produces on
   `/v1/chat/completions`.
2. **Ceiling.** Every answer carries `max_model_len` equal to the server's
   configured `--max-context`.
3. **Round trip.** `detokenize(tokenize(x).token_ids)` is byte-identical to
   the rendered text, on the UTF-8 cases above.
4. **Raw form.** `prompt` tokenizes with no template applied.
5. **Media refused.** A body with an `image_url` part answers `400
   media_not_countable` and fetches nothing (asserted: no media acquisition
   event).
6. **Free.** A tokenize call allocates no sequence and publishes no retained
   state, asserted on the mock scheduler's counters.
7. **Documented.** Both paths are in the OpenAPI document and in the asserted
   path set.
8. `cargo test` passes workspace-wide.

## Out of Scope

- Counting images. The refusal is the decision, not a placeholder.
- A `GET` form, a batch form, or counting several bodies in one call.
- Exposing the vocabulary, the special-token set, or the template source.
- `/v1/tokenize` on the Responses body shape: the chat body is the one every
  client already builds.

## Further Notes

- **Why not estimate the image cost** from the URL's declared dimensions: the
  preparation policy (resize, patch grid) is the server's and has changed
  twice; a number that tracks it would have to be recomputed here every time
  it changes, and a number that does not is worse than no number.
- **Why `max_model_len` in the same response**: the alternative is a flag on
  `/v1/models`, which is one more round trip for the decision the caller is
  making right now.
- **The field names are ours, not vLLM's.** vLLM's `/tokenize` returns a
  count and a token list under its own names, and its `/detokenize` takes
  them back under others; this spec does not claim drop-in compatibility
  with it, because nobody has checked the current shape against the source.
  If compatibility is wanted, the implementing agent should read
  `vllm/entrypoints/openai/protocol.py` at a pinned version and adopt the
  names exactly — a half-matching shape is worse than an honestly different
  one.
- **Round trip, as built (2026-09-30).** The artifact's tokenizer normalizes to NFC before it splits, so acceptance 3 holds byte for byte for text the
  normalizer leaves alone (ASCII, multi-byte UTF-8, an emoji ending in a zero-width joiner) and, for
  a decomposed `e` + combining accent, the ids spell the composed `é`: the round trip is over what
  the tokenizer saw, which is what `count` counts. `detokenize` checks each id with the tokenizer's
  own `id_to_token` because a decode skips an unknown id instead of refusing it. A load with no real tokenizer
  (the placeholder provider) answers the raw form `501 tokenizer_unavailable` rather than inventing ids.
