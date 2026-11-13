---
title: "Chat completions"
description: "POST /v1/chat/completions: fields, streaming, tools, reasoning and the agent lane."
sidebar:
  order: 2
---

Accepts `messages` (role + content), `model`, `stream`, `max_tokens`,
`temperature` (0..2), `top_p` (0..1), `presence_penalty` and
`frequency_penalty` (-2..2), and a signed 64-bit `seed`.

- `top_k` is an **Ignis extension**, not an OpenAI Chat Completions parameter:
  0..20, where 0 selects the 20-candidate sampler cap during stochastic
  sampling.
- `class` is an **Ignis extension**: the request's lane tag, `interactive`
  (default) or `agent`. The same class can be stated as an `@agent` suffix on
  the model name.
- `thinking_budget` is an **Ignis extension**: the reasoning tokens this
  request may spend before its close is forced. Absent takes
  `--model-thinking-budget`, `0` means no budget, and `reasoning_effort: "max"`
  ignores it and runs with no budget. The budget always leaves 2,048 tokens of
  `max_tokens` for the answer. A forced close is reported as
  `thinking_budget_forced_at` on the choice (on the finish chunk when
  streaming; top-level on `/v1/responses`): the reasoning tokens emitted when
  it began. The field is absent when the close was not forced.
- **A turn that ends inside its reasoning answers.** With thinking on, an end
  of turn the model draws while its reasoning block is still open becomes the
  block's `</think>`, and the model goes on to answer — usually with the call
  or the text it was writing in its reasoning, now on the content channel.
  Such a turn used to come back as reasoning only, with no content and no
  tool call. The same holds on `/v1/responses`. The request log's
  `ignis.request.done` line marks it with `reasoning_redirected_at`, the
  output index of that `</think>`. With thinking off, or `ignore_eos`, the
  end of turn ends the turn as it always has.
- **A loop in the reasoning is taken back** (`--model-loop-guard`, `rewind`
  by default, in the table above). `usage.completion_tokens_details.rewound_tokens`
  (`usage.output_tokens_details.rewound_tokens` on `/v1/responses`) counts
  the tokens taken back, which `completion_tokens` (`output_tokens`) still
  counts; it is absent when nothing was. On `/v1/responses`,
  `include: ["reasoning.encrypted_content"]` gives each reasoning item an
  `encrypted_content` with the reasoning as the model saw it, and a
  reasoning input item that carries one is rendered from it in place of its
  text.
- Values outside the ranges above are rejected with
  `invalid_sampling_parameter`; they are never silently clamped.
- **Absent sampling fields take the Qwen3.8 model card's values** for the
  request's mode, each field on its own (spec server/12):

  | mode | `temperature` | `top_p` | `top_k` | `presence_penalty` | `frequency_penalty` |
  |---|---:|---:|---:|---:|---:|
  | thinking | 1.0 | 0.95 | 20 | 0 | 0 |
  | `enable_thinking: false` | 0.7 | 0.80 | 20 | 1.5 | 0 |

  An absent `seed` is a fresh one per request, so two identical requests are
  two independent samples; a sent `seed` is kept. Sampling runs on the GPU,
  and its RNG is counter-based, keyed by the seed, the sequence's own position
  and the draw's purpose (prefill or decode): what a request generates depends on its own seed alone, never on
  which requests shared its decode round, and no RNG state travels with a
  sequence that is moved off the device and back. **Greedy is
  `temperature: 0`**: the fields such a request does not send are neutral,
  and because the leaf's greedy branch does not read stochastic filters or
  penalties, a non-neutral `top_p`, `top_k`, `presence_penalty` or
  `frequency_penalty` sent with it is rejected rather than ignored.
- A request that sends no `max_tokens` (nor `max_completion_tokens`, nor
  `max_output_tokens` on `/v1/responses`) generates at most
  `--model-default-max-tokens` tokens, its reasoning included. The flag is 38,912
  unless you set it (`IGNIS_MODEL_DEFAULT_MAX_TOKENS`, make `DEFAULT_MAX_TOKENS`).
  Such a request ends with `finish_reason: "length"` when it gets there
  (`incomplete` with `max_output_tokens` on `/v1/responses`). Before, it could
  generate to the end of the context. Send `max_tokens` for more, or start the
  server with `--model-default-max-tokens 0` for the old behaviour.
- `max_completion_tokens` is `max_tokens` under OpenAI's current name — the
  one the current SDKs send. Both with different values is a 400.
- `stop` (a string, or 1 to 4 non-empty strings) ends the answer before the
  first sequence to appear in its content, with `finish_reason: "stop"`. The
  sequence is never emitted, streaming included; it is never matched in the
  reasoning, nor inside a tool call.
- `tool_choice` takes `"auto"` (the default), `"none"`, `"required"`, and a
  named function (`{"type": "function", "function": {"name": "..."}}`, or
  `{"type": "function", "name": "..."}` on the responses API). The last two
  force the call's opening — `<tool_call>` and the function tag, with the
  name when one is named — and the model writes the arguments. With thinking
  on (the default) the opening is forced right after the reasoning block
  closes, so the model still thinks first; with `enable_thinking: false` it
  is the first thing generated. After a `</think>` the model wrote itself,
  the first forced token is the opening's second, its joiner (the `\n` after
  `<tool_call>`): the token before it was drawn unseen, and may itself have
  opened the call. The call comes back as any other call does,
  with `finish_reason: "tool_calls"`. Naming a function that is not in
  `tools`, `"required"` with no tools, and a `max_tokens` shorter than the
  opening are a 400; so is a tokenizer that cannot force the opening
  (`tool_choice_unsupported` — with thinking on it needs `</think>` and
  `<tool_call>` to be single tokens, and `enable_thinking: false` is the
  way out). With thinking on the reasoning spends the same `max_tokens`: a
  block that never closes ends `length` with no call, which the thinking
  budget's answer reserve prevents.
- `response_format: {"type": "json_object"}` holds the answer to a valid JSON
  object, and `{"type": "json_schema", "json_schema": {...}}` to your schema,
  streamed or not (from 0.11.0, on the 27B; 0.10.0 refuses both): each
  sampled token is checked and a miss repaired. Under `strict: true` OpenAI's
  strict subset is held exactly and a schema outside it is refused by name;
  without it the rest is inert. `{"type": "text"}` is a free answer, and
  Flash-Next refuses structured output naming itself
  ([Structured output](structured-output.md)). Beside `tools`, each turn
  either calls a tool or answers under the schema: a turn that opens a call
  before its answer starts is released from the schema, and
  `tool_choice: "required"` or a named function still forces the call
  ([Tools beside structured output](structured-output.md#tools-beside-structured-output)).
- `ignis_describe_schema` is an **Ignis extension** (from 0.11.0): with
  `json_schema`, the server describes the schema to the model in a block after
  the conversation, where the assistant's turn opens, and `false` leaves the
  prompt as you sent it, for a client whose own prompt already describes the
  schema. Absent or `null` takes `--model-describe-schema` (on by default);
  a value that is not a boolean is a 400 naming it. `json_object` adds
  nothing either way
  ([the schema, described to the model](structured-output.md#the-schema-described-to-the-model)).
- Non-streaming returns `choices[].message.content` plus `usage`, and
  `message.refusal`, always `null` (from 0.11.0); streaming
  emits `chat.completion.chunk` SSE frames (token deltas, a final
  `finish_reason` chunk, then `[DONE]`). The first token is sent alone as
  soon as the prompt's prefill ends, before any decode round, and each
  round's tokens follow it ([how ignis serves a request](../concepts/how-ignis-serves.md)).
  `usage.prompt_tokens_details.cached_tokens`
  is the prompt the request resumed from retained state instead of
  prefilling (0 when none).
- Every other OpenAI field is accepted as inert when its value asks for what
  the server already does (`n: 1`, `store: false`, `logprobs: false`,
  `metadata`, `user`, ...) and refused
  with a 400 `unsupported_value` naming it in `param` otherwise. The full
  table is in the API reference at `/v1/docs/`; a field outside it is ignored.
  `strict` on a function tool is accepted at any value and inert: the call's
  arguments are not held to the tool's `parameters` (GitHub #582).
