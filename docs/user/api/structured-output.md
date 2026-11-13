---
title: "Structured output"
description: "response_format json_object on chat completions: every answer a valid JSON object, checked token by token and repaired."
sidebar:
  order: 2
---

From 0.11.0, chat completions on the 27B serve
`response_format: {"type": "json_object"}`: the answer is a valid JSON object,
streamed or not. 0.10.0 refuses the field with a `400`.

```bash
curl http://localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Describe a cat: its name, its age and its toys."}],
  "response_format": {"type": "json_object"}
}'
```

The answer's `content` parses as one JSON object: no code fence around it, no
sentence before or after it. The prompt does not need to say "JSON", and
nothing is added to it.

## How the answer is held

The model samples freely. Each token it samples for the answer is **checked**
against the grammar of a JSON object before the model reads it back. A token
that would break the object is a **miss**, and is **repaired** at once: that
one position is sampled again from the same scores, restricted to the tokens
that keep the object valid. The repaired token takes the missed one's place,
and the model goes on from it. A token the grammar accepts costs about a
microsecond, so an answer the model writes as an object on its own runs at
its ordinary speed. Decoding greedily (`temperature: 0`), the answer is exactly
the one masking every step would give; under a temperature it follows the
model's own distribution restricted to valid tokens. The decision and its
measurements are in
[ADR 0089](../../adr/0089-a-grammar-checks-each-sampled-token-and-repairs-a-miss-and-the-model-is-told-the-schema.md).

What the grammar writes is a valid object; it may be one of several spellings
of it:

- **Whitespace stays short.** A run of whitespace between tokens is at most 48
  characters, before the object, inside it and after it, so a model that keeps
  writing spaces or line breaks cannot stream them without end. Whitespace
  inside a string is the string's own text.
- **An answer ends only where the object is complete.** A model that tries to
  stop halfway is repaired toward the object's end; one that writes a remark
  after the object ends there instead, after at most a run of whitespace, with
  `finish_reason: "stop"`.
- **`refusal` is always `null`** on the message, as on every chat completion.

## Thinking

With thinking on (the default), the reasoning is the model's own: it is not
checked, and it arrives in `reasoning_content` as on any request. The answer
is held from the first token after the reasoning's `</think>`. A
[thinking budget](chat-completions.md)'s forced close is followed the same
way: the close is the server's text, and the answer after it is held. With
`enable_thinking: false` the answer is held from the first token generated.

## Cuts

- A cut by `max_tokens` (or `max_completion_tokens`) returns the content as
  generated, with `finish_reason: "length"`: the object may be unfinished, as
  OpenAI's is. Leave room for the whole answer, and for the reasoning with
  thinking on.
- A request past its [deadline](deadlines.md) is answered as any other: with
  what it generated and `finish_reason: "length"`, or a `504`, as
  `--server-timeout-reply` says.

## Speed and speculation

[Speculative decoding](../concepts/speculative-decoding.md) stays on for a
structured request. In each verify round the drafts the model accepted are
checked against the grammar before the round commits them: the run is kept up
to its first miss, and that token is repaired from the scores the round has
already computed for it. The drafts before the miss are kept, and only the
ones after it are lost, to be drafted again in the next round. Greedy answers
are the same with speculation on and off, up to the near-ties any request
has.

The requests decoding beside a structured one are not touched: they keep
every draft they would have kept, and the round runs as it does without the
structured request. The only extra work is the check on the structured
request's own tokens, and a round with no structured request in it does none.

## What is refused

Each refusal is a `400` naming `response_format` in `param`:

| You send | You get |
|---|---|
| `{"type": "json_schema", ...}` | `unsupported_value`: a schema is not served yet; it is being built (GitHub #287, #588). |
| any other `type`, or no object | `unsupported_value`. `{"type": "text"}`, `null` and absent are a free answer. |
| `json_object` to Qwen3.8-Flash-Next | `model_unsupported`, naming the model: Flash-Next does not serve structured output yet (GitHub #581). |
| `json_object` with `tool_choice: "required"` or a named function | `unsupported_value`: a forced call beside structured output is not served yet (GitHub #592). |
| `json_object` with thinking on, on a tokenizer whose `</think>` is not one token | `unsupported_value`; send `enable_thinking: false`. |

`tools` with `tool_choice: "auto"` are accepted beside `json_object`, but the
turn answers under the grammar: the model cannot open a tool call in it until
GitHub #592 lets a turn choose between a call and the object.

`/v1/responses` refuses `text.format` other than `text` for now
([The Responses API](responses.md)).
