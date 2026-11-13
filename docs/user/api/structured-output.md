---
title: "Structured output"
description: "response_format and text.format, json_object and json_schema, on chat completions and Responses: every answer valid JSON, held to your schema, checked token by token and repaired."
sidebar:
  order: 2
---

From 0.11.0, chat completions on the 27B serve `response_format`, streamed or
not: `{"type": "json_object"}` holds the answer to a valid JSON object, and
`{"type": "json_schema", ...}` to your schema. 0.10.0 refuses both with a
`400`.

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
against the grammar of a JSON object, or of your schema, before the model
reads it back. A token that would break the answer is a **miss**, and is
**repaired** at once: that one position is sampled again from the same
scores, restricted to the tokens that keep the answer valid. The repaired
token takes the missed one's place, and the model goes on from it. A token
the grammar accepts costs about a microsecond, so an answer the model writes
right on its own runs at its ordinary speed. Decoding greedily
(`temperature: 0`), the answer is exactly the one masking every step would
give; under a temperature it follows the model's own distribution restricted
to valid tokens. The decision and its measurements are in
[ADR 0089](../../adr/0089-a-grammar-checks-each-sampled-token-and-repairs-a-miss-and-the-model-is-told-the-schema.md).

What the grammar writes is a valid answer; it may be one of several spellings
of it:

- **Whitespace stays short.** A run of whitespace between tokens is at most 48
  characters, before the answer, inside it and after it, so a model that keeps
  writing spaces or line breaks cannot stream them without end. Whitespace
  inside a string is the string's own text.
- **An answer ends only where it is complete.** A model that tries to stop
  halfway is repaired toward the end; one that writes a remark after the
  object ends there instead, after at most a run of whitespace, with
  `finish_reason: "stop"`.
- **`refusal` is always `null`** on the message, as on every chat completion.

## A schema: `json_schema`

From 0.11.0. The answer is a JSON value valid under the schema you send:

```bash
curl http://localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Describe a cat: its name, its age and whether it is happy or sad."}],
  "response_format": {"type": "json_schema", "json_schema": {
    "name": "cat",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
        "mood": {"type": "string", "enum": ["happy", "sad"]}
      },
      "required": ["name", "age", "mood"],
      "additionalProperties": false
    }
  }}
}'
```

`json_schema` takes the `schema` (required), `strict` (a boolean, `false` when
absent or `null`), and a `name` and a `description` (strings, both optional).
The model is told the schema at the end of the prompt, as the next section
says.

### The schema, described to the model

From 0.11.0, ignis tells the model the schema it answers to. After the
conversation, where the assistant's turn opens, the prompt gains one block,
the one a system message after the first renders as:

```text
<|im_start|>system
A cat.

Respond with a JSON object that matches this JSON schema:
{"type":"object","properties":{"name":{"type":"string"}, ...},"required":[...],"additionalProperties":false}<|im_end|>
<|im_start|>assistant
```

The block opens with the format's `description` when you send one ("A cat."
above), then that sentence and the schema, compact, its keys in the order you
sent them. A schema whose root is not an object (served without `strict`) is
"a JSON value" instead.

- **Why.** A model that has not read the schema writes key names of its own,
  and the grammar then repairs nearly every key. Measured on the 27B: 4.2
  misses per 100 tokens without the description against 0.04 with it, an
  estimated 8 to 18% of speculative decode speed lost against 1 to 2%, and one
  recursive schema that ran away to `max_tokens`
  ([finding](../../findings/2026-11-12-structured-output-grammar-measured-on-the-27b.md)).
- **Where.** After your last message, never inside it, and before the
  assistant's turn opens: with thinking on, the model reads the schema before
  it reasons, and the reasoning stays its own. Your messages render exactly as
  they would without it.
- **Reuse across turns.** The checkpoint the next turn resumes from is taken
  where your conversation ends, before the block
  ([prompt reuse](../concepts/kv-cache-and-prompt-reuse.md#what-is-kept)): the
  next turn reuses the whole conversation, and only the block, about as long as
  the schema, is prefilled again with each turn. `usage.prompt_tokens` counts
  it.
- **Turning it off.** A client whose own prompt already describes the schema
  sends `"ignis_describe_schema": false`, an ignis extension, and its prompt
  reaches the model byte for byte as sent. `--model-describe-schema false`
  turns the description off for every request that does not send the field,
  and a request's `true` turns it back on
  ([Prompting](../configuration/prompting.md)). A value that is not a boolean
  is a `400` naming `ignis_describe_schema`.
- **`json_object`** adds nothing: there is no schema to describe.

### With `strict: true`

The schema must lie in OpenAI's strict subset, and all of it is held, in every
answer:

- **Objects** have exactly the schema's keys, every one present, in the
  schema's order. As strict mode asks, each object sets
  `additionalProperties: false` and lists every property in `required`; an
  optional field is a `type` list with `"null"`.
- **Arrays** (`items`, `minItems`, `maxItems`), **`enum` and `const`** with
  their exact spelling, **type unions**, and **`anyOf`** below the root.
- **`$defs` and `$ref`** to them, recursion included, and `$ref: "#"` to the
  root.
- **`pattern`**, a regular expression without backreferences or lookaround;
  the nine **`format`s** (`date-time`, `time`, `date`, `duration`, `email`,
  `hostname`, `ipv4`, `ipv6`, `uuid`); **`minimum`, `maximum`,
  `exclusiveMinimum`, `exclusiveMaximum` and `multipleOf`**, held exactly in
  decimal, so a price in cents is never a third of a cent.

The keywords one by one are on [JSON Schema keywords](../reference/json-schema.md).
A schema outside the subset is refused [by name](#what-is-refused) before
anything is generated: under `strict`, the guarantee holds or the request
does not run.

### Without `strict`

What the subset holds is held, and the rest is **inert**: the schema is served
and the answer is not held to it. Inert are:

- every keyword outside the subset: `minLength`, `maxLength`, `oneOf`,
  `allOf`, `not`, `uniqueItems`, `patternProperties`, ... (the
  [whole list](../reference/json-schema.md#inert-without-strict));
- a `pattern` that is not a regular language (a backreference, a lookaround)
  or does not parse, a `format` outside the nine, and a `format` beside a
  `pattern`;
- `exclusiveMinimum` or `exclusiveMaximum` as a boolean (draft 4), and a bound
  or a `multipleOf` with more than 18 decimal places or above 10^19;
- a `$ref` that is not local (`#/...`), and any keyword beside a `$ref` or an
  `anyOf`.

A non-regular `pattern` and a `$ref` that is not local are refused only under
`strict: true` (`pattern_not_regular`, `unsupported_ref`); without it the
answer is held to the rest of the schema.

A looser schema than strict mode allows is served as written, each choice
keeping the answer valid: a root of any type; a schema with no `type` takes
the types its keywords imply (`properties` an object, `items` an array,
`pattern` a string, a bound a number, nothing at all any value); an object
writes every property it names, required or not, and never a key it does not
name; a key `required` lists that `properties` does not is written with a
value `additionalProperties` allows; an object with no `properties` and an
`additionalProperties` schema is a map of such values.

### What the grammar writes

A schema admits many documents; the grammar writes one of a narrower set,
all of them valid:

- keys in the schema's order, never another;
- `true`, `false`, `null`, `enum` values and `const`s spelled as the schema
  spells them;
- an integer as digits alone (no fraction, no exponent), and a bounded number
  in plain decimal;
- whitespace in runs of at most 48, as above;
- nesting at most about 128 levels deep: a recursive schema's answer closes
  before it.

### Compiled once

A schema is compiled the first time it is sent, in about 2 ms for an SDK's and
about 27 ms for one at OpenAI's limits
([finding](../../findings/2026-11-12-structured-output-grammar-measured-on-the-27b.md)),
and kept: the same schema sent again, whatever its whitespace, with the same
`strict`, is not compiled again. The loaded model's compiled grammars hold at
most `--model-grammar-cache-bytes` (256 MiB), the least recently used dropped
past it ([Host RAM for the 27B](../configuration/host-ram.md#structured-outputs-compiled-grammars)).

### OpenAI's limits

Held with `strict` and without it: at most 5,000 properties, 10 levels of
nesting, 1,000 `enum` values, and 120,000 characters of property names,
`enum` values and constants. A schema past one is refused, naming the limit
([the table](../reference/json-schema.md#limits)).

## Thinking

With thinking on (the default), the reasoning is the model's own: it is not
checked, and it arrives in `reasoning_content` as on any request. The answer
is held from the first token after the reasoning's `</think>`. A
[thinking budget](chat-completions.md)'s forced close is followed the same
way: the close is the server's text, and the answer after it is held. With
`enable_thinking: false` the answer is held from the first token generated.

## Tools beside structured output

A request may carry `tools` beside `response_format`, as agent SDKs send them:
each turn either calls a tool or answers under the schema.

- **Until the answer starts**, the grammar also admits a tool call's opener,
  `<tool_call>`. Whitespace before it is fine; anything else starts the
  answer. With thinking on, that is after the reasoning's `</think>`.
- **A turn that opens a call is released** from the schema for the rest of
  the turn: the call is written as the model writes it and comes back as any
  other call, with `finish_reason: "tool_calls"`. Nothing in it is checked,
  and it counts no miss and no repair in the request log or the
  [metrics](../guides/telemetry.md).
- **The next turn**, after the tool's result, is held again, so an agent
  loop ends on an answer valid under the schema.
- **Once the answer has started**, the turn cannot open a call any more: an
  opener there is a miss, repaired into the answer.
- **`tool_choice: "required"` or a named function** forces the call as it
  does without `response_format` ([Chat completions](chat-completions.md)):
  that turn is the call, and is not held, nor
  [told the schema](#the-schema-described-to-the-model). `"none"` keeps the
  tools from the model, and every turn answers under the schema.

`strict: true` on a function tool is accepted and inert: the call's arguments
are the model's own, not held to the tool's `parameters` (GitHub #582).

## On the Responses API

`/v1/responses` serves the same thing as `text.format`, over HTTP and over
[the WebSocket mode](responses.md): `{"text": {"format": {"type":
"json_object"}}}` or `{"text": {"format": {"type": "json_schema", "name":
"cat", "schema": {...}, "strict": true}}}`. The Responses API carries a
schema's members flat beside the `type`, where chat completions nest them in
a `json_schema` object; everything else is the same: the guarantees, the
schema's description (and `ignis_describe_schema: false`), the thinking and
tools rules, the refusals and the limits. A refusal is a `400` whose `param`
is `text.format`, and the response's own `text.format` echoes what was sent.
A streamed response's `response.output_text.delta` events carry only tokens
the grammar accepted, and the `output_text` they add up to is the one a
non-streamed response returns.

## Cuts

- A cut by `max_tokens` (or `max_completion_tokens`) returns the content as
  generated, with `finish_reason: "length"`: the answer may be unfinished, as
  OpenAI's is. Leave room for the whole answer, and for the reasoning with
  thinking on. On the Responses API the same cut is a response with
  `status: "incomplete"` and `incomplete_details.reason: "max_output_tokens"`,
  its `output_text` as generated.
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

Each refusal is a `400` naming `response_format` in `param`, answered before
anything is generated. A schema the grammar refuses carries the refusal's own
`code`, and a message naming the keyword or the limit and where in the schema:
``response_format: json_schema refused, `allOf` at #/properties/tags: `allOf` is outside the strict subset``.

| You send | You get |
|---|---|
| under `strict: true`, a keyword outside the subset | `unsupported_keyword`, naming it |
| under `strict: true`, a root that is not `"type": "object"`, or is an `anyOf` | `root_not_object`, `root_anyof` |
| under `strict: true`, an object without `additionalProperties: false`, or a property `required` does not list | `additional_properties_not_false`, `not_all_required` |
| under `strict: true`, a non-regular `pattern`, a `format` outside the nine, a `$ref` that is not local, a schema with neither a `type` nor an `enum` or `const` | `pattern_not_regular`, `unsupported_format`, `unsupported_ref`, `missing_type`; the other inert cases above have codes of their own |
| a schema past one of OpenAI's limits, `strict` or not | `too_many_properties`, `too_deeply_nested`, `too_many_enum_values`, `too_many_characters` |
| a schema no finite answer satisfies (a recursion through required keys, a `false` schema), `strict` or not | `no_finite_value` |
| a malformed schema (an `enum` that is not an array, an unknown `type`, a `minItems` above `maxItems`, a `$ref` that names nothing), `strict` or not | a code of its own, naming the keyword |
| `json_schema` without its `json_schema` object or its `schema`, a `strict` that is not a boolean, a `name` or `description` that is not a string | `invalid_request_error` |
| any other `type`, or no object | `unsupported_value`. `{"type": "text"}`, `null` and absent are a free answer. |
| structured output to Qwen3.8-Flash-Next | `model_unsupported`, naming the model: Flash-Next does not serve structured output yet (GitHub #581). |
| `tools` the turn may call (`tool_choice: "auto"`), on a tokenizer whose `<tool_call>` is not one token | `unsupported_value`: the grammar could not tell the call's opener from the answer. Send `tool_choice: "none"`. |
| structured output with thinking on, on a tokenizer whose `</think>` is not one token | `unsupported_value`; send `enable_thinking: false`. |

On `/v1/responses` the same refusals answer with `param: text.format`
([The Responses API](responses.md)).
