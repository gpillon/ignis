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
  and the grammar then repairs nearly every key. Measured served on the 27B:
  4.5 misses per 100 tokens without the description against 0.4 with it, 76%
  of the decode speed against 98% ([What it costs](#what-it-costs)), and one
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
- whitespace in runs of at most 48, as above (`--model-grammar-max-whitespace`);
- nesting at most about 128 levels deep (`--model-grammar-max-depth`): a
  recursive schema's answer closes before it, and a schema whose shortest
  answer nests deeper is refused.

These bounds, the size a `pattern` may compile to and how deep a schema may
chain its `$ref`s are the server's, set at start
([Prompting](../configuration/prompting.md)).

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

A call to a strict tool is the exception, forced or not: the turn is
released from the schema all the same, and the call is held to its tool
([Beside `tool_choice`, `response_format` and `parallel_tool_calls`](#beside-tool_choice-response_format-and-parallel_tool_calls)).

## Held tools: `strict: true`

From 0.11.0, chat completions on the 27B hold the calls of a function tool
sent with `strict: true`, a **strict tool**, to its `parameters`, with the
same grammar that holds an answer: each sampled token is checked, and a miss
is repaired. The Responses API holds them too
([below](#on-the-responses-api)). Why the model writes its call in its own
order, and why a value ends only where the call goes on:
[ADR 0094](../../adr/0094-a-held-tools-call-is-checked-as-the-model-writes-it-in-its-own-order-and-a-value-ends-only-where-the-call-continues.md).

```bash
curl http://localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Save a note saying hi."}],
  "tools": [{"type": "function", "function": {"name": "write_file", "strict": true, "parameters": {
    "type": "object",
    "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "mode": {"type": "string", "enum": ["w", "a"]}},
    "required": ["path", "content", "mode"], "additionalProperties": false}}}]
}'
```

- **Every call to a strict tool** names a tool the request declares, and its
  `arguments` are valid under the tool's `parameters`, written in the order
  you sent its `properties` in. Streamed, a call arrives once complete, as
  every call does.
- **The model writes its call in its own dialect and order.** It may write
  the parameters in any order, each at most once; the function closes only
  once every required parameter is written (a strict schema requires them
  all; one that may be `null` may be left out). A call that would close early
  is steered to the missing parameter.
- **The answer is free outside its calls.** The grammar arms at a call's
  `<tool_call>`, in the answer only, never in the reasoning, and is released
  at the call's close; each further call arms it again. A call `max_tokens`
  cuts comes back as any cut call does (from 0.11.0): its name and the
  arguments written so far, each value ending where the call goes on, and
  `finish_reason: "length"`
  ([Chat completions](chat-completions.md)).
- **A string value is the call's raw text**, and it ends only where the call
  goes on: a parameter's opener the call has not written yet, or the
  function's close once every required parameter is written. A file's
  content may hold lines that look like the call's own tags, but not these:
  - a line opening a parameter the tool declares and the call has not
    written yet (`<parameter=mode>`), alone or after a `</parameter>` line:
    the value ends there, and what follows is that parameter's value;
  - once every required parameter is written, the call's close: a
    `</function>` line and `</tool_call>` after it, whitespace between;
  - while a required parameter is still missing, a line starting
    `</function` (`</functional>` too), and a line starting `</parameter>`
    that goes on, after its whitespace, with `</` (a `</div>` on the next
    line too): the model is steered to the missing parameter;
  - a `</parameter>` line, or a closing `</function>` one, followed by more
    whitespace than `--model-grammar-max-whitespace` (48).

  The model writes the parameters in its own order: a file about tool calls
  is kept whole when its content comes after the parameters its examples
  open, and written before them it ends at the first such example.
- **A string with an `enum`, a `const`, a `pattern` or a `format`** is held
  to it, on one line that does not start with one of the call's tags
  (`<parameter=`, `</parameter>`, `</function>`). `minLength` and `maxLength`
  lie outside the strict subset and are refused; a tool the server normalizes
  leaves them out instead ([below](#a-tool-sent-without-strict)).
- **A parameter of another type**, and one that may be `null`, is held as
  well: [Typed and nullable parameters](#typed-and-nullable-parameters).
- **A tool without `strict`** in the same request keeps its arguments as the
  model writes them, as OpenAI's chat completions do; the Responses API
  normalizes it instead ([below](#a-tool-sent-without-strict)), and so do both
  under the operator's switch ([Holding every tool](#holding-every-tool)).
  With a strict tool in the request, every call's function name is held to
  the request's tools.
- **A request with no strict tool** is served as before, byte for byte, on
  chat completions, unless the operator's switch holds every tool
  ([Holding every tool](#holding-every-tool)).
- **On Qwen3.8-Flash-Next**, `strict` is accepted and inert for now (GitHub
  #581).

A held call's misses and repairs count beside structured output's, with the
type `tool`: `ignis_structured_*{type="tool"}` in the
[metrics](../reference/metrics.md), and `structured.type`,
`structured.misses` and `structured.repairs` in the request log's
`ignis.request.done` ([Telemetry](../guides/telemetry.md)).

A strict tool is refused with a `400` before anything is generated, naming
the tool's field in `param`: `tools[1].function.parameters` for its schema,
``tools[1].function.parameters: strict tool refused, `minLength` at #/properties/a: `minLength` is outside the strict subset``.
On the Responses API a `param` names the field where the tool sent it, flat
beside its `type`: `tools[1].parameters`, `tools[i].name`, `tools[i].strict`.

| You send | You get |
|---|---|
| a strict tool's `parameters` that `json_schema` under `strict: true` refuses | the same code ([What is refused](#what-is-refused)), naming the keyword and where |
| a strict tool whose name holds `>`, or `</tool_call` | `tool_name_unwritable`, `param` `tools[i].function.name`: the dialect ends a name at its first `>` |
| a strict tool whose name another tool in the request has | `duplicate_tool_name`, `param` `tools[i].function.name` |
| a strict tool with a parameter whose name holds `>` | `parameter_name_unwritable` |
| a strict tool with an `enum`, a `const` or a `pattern` no value of which fits on one line that does not start with one of the call's tags | `no_writable_value`, naming the keyword |
| a strict tool with thinking on, on a tokenizer whose `</think>` is not one token | `unsupported_value`, `param` `tools[i].function.strict`; send `enable_thinking: false` |

### Typed and nullable parameters

A strict tool's parameters of every type are held, not only its strings, on
chat completions and on the Responses API alike (GitHub #612).

- **An integer, a number, a boolean, an object or an array** is written by
  the model as JSON and held to its own subschema by the grammar that holds
  `json_schema`: bounds and `multipleOf`, item counts, an object's keys,
  `enum` and `const`, `anyOf`, and `$defs` with `$ref`, recursion included,
  in the grammar's narrower set ([What the grammar writes](#what-the-grammar-writes)).
  The JSON starts on the line after the parameter's opener, or right after
  it, with no whitespace before it; whitespace may follow it.
- **It arrives as the model wrote it**, compact: an object's keys in its
  schema's order, every digit of a number kept, a literal in its canonical
  spelling (`1.50` is `1.5`). A call that writes `12` for an integer gets
  `12`; one that would write `twelve` is repaired to a number.
- **A string beside another type**, as in `anyOf` a string and an integer,
  or an `enum` of a string and a number, is held as JSON as well: the model
  writes the string quoted, `"abc"`.
- **A parameter that may be `null`** (`"type": ["string", "null"]`, or
  `anyOf` with `{"type": "null"}`, the way a strict schema writes an
  optional one) may be left out of the call, and the call's `arguments` then
  carry `null` for it, so they hold every key strict mode requires. Written,
  it is held to its type, or to `null`.
- **A string that may be `null` is the call's raw text, and a text that
  reads `null` is `null`.** Such a parameter never arrives as the string
  `"null"`: the model's dialect writes a string without quotes, so the four
  letters can only be one of the two, and they are read as null, spaces
  around them aside. Any other text is that text, `"null"` with its quotes
  included. A `pattern`, a `format` or an `enum` on such a parameter holds
  its text to it, or to `null`.

A call to a strict `edit` with an object `change` holding a list of edits,
an integer `line`, a boolean `dry_run` and an optional string `note`, which
the model writes in its own dialect, `note` left out,

```text
<tool_call>
<function=edit>
<parameter=change>
{"edits": [{"old": "x", "new": "y"}]}
</parameter>
<parameter=line>
12
</parameter>
<parameter=dry_run>
false
</parameter>
</function>
</tool_call>
```

arrives with the `arguments`
`{"line":12,"dry_run":false,"change":{"edits":[{"old":"x","new":"y"}]},"note":null}`
when its `properties` are sent in that order: `line`, `dry_run`, `change`,
`note`.

### Beside `tool_choice`, `response_format` and `parallel_tool_calls`

A strict tool's calls are held on every turn that may call it, whatever else
the request asks of the turn.

- **`tool_choice: "required"` or a named function**: the forced opening
  ([Chat completions](chat-completions.md)) opens the call, and the grammar
  holds everything after it, the parameters included, the function's name
  too under `"required"`. The forced tokens are the server's text, never a
  miss. With thinking on the opening is forced after the reasoning closes;
  when the model has already opened the call itself there, and what it
  wrote is not how the opening goes on (a line break more, say), the server
  stops forcing and the grammar holds the rest. Under a named function the
  call names that function, there too; only a named function whose name
  holds `>`, which the dialect cannot write, leaves the call to any of the
  request's tools.
- **`response_format` beside strict tools**: until the answer starts,
  `<tool_call>` releases the turn from the schema, as
  [above](#tools-beside-structured-output), and the call is held to its
  tool. The turn stays released after it: its text is free, and every
  further call in it is held the same way. A turn that answers is held to
  the schema. The request counts under the answer's type, `json_schema` or
  `json_object`, in the metrics and the request log, the call's misses and
  repairs included.
- **Thinking on**: the reasoning stays free, and a call written there is
  neither held nor read as a call; the hold starts at the first call after
  the reasoning's `</think>`.
- **`parallel_tool_calls: false`**: the turn ends at the call's own close,
  `</function>` then `</tool_call>` once every required parameter is
  written, not at the first `</tool_call>` the model writes: a string value
  may hold `</tool_call>` on a line of its own, and is delivered whole.
  Nothing generated after the close reaches you or counts in `usage`.

The Responses API holds them the same way, `text.format` in place of
`response_format`, over HTTP and over the WebSocket mode.

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

[Held tools](#held-tools-strict-true) are the same on `/v1/responses`, over
HTTP and over the WebSocket mode: a function tool sent with `strict: true`,
which sits flat beside the tool's `name` there (`{"type": "function", "name":
"write_file", "strict": true, "parameters": {...}}`), has its calls held, with
the same refusals, metrics and log fields. A refusal's `param` names the tool
as the request sent it, `tools[1].parameters` for a flat tool (or
`tools[1].function.parameters` for one in the chat shape, which the Responses
API takes as it is), and the WebSocket mode answers it as an `error` event. A
streamed call comes in one `response.function_call_arguments.delta`, once the
call is complete, with the `arguments` a non-streamed response returns.

### A tool sent without `strict`

From 0.11.0, on the 27B, the Responses API normalizes a function tool sent
without `strict`, or with `strict: null`, into strict form and holds its
calls, as OpenAI's Responses API does by default (GitHub #614):

- **Every object is closed**: a call names no parameter the tool's
  `parameters` do not declare, and a nested object holds no key it does not
  declare. A parameter the model invents is a miss, steered to one the tool
  declares.
- **An optional parameter, one `required` does not list, may be left out**,
  and stays out of the call's `arguments`, one that may be `null` included:
  the arguments are valid under the schema you sent, its length bounds aside.
  A required parameter is held as a strict tool's: the call closes only once
  it is written, unless it may be `null`, and then it arrives as `null` when
  left out.
- **A nested object is written whole**: every property it declares, as the
  grammar writes an object ([What the grammar writes](#what-the-grammar-writes)),
  which is valid under your schema too.
- **Length bounds are left out** (GitHub #641): `minLength` and `maxLength`
  lie outside the strict subset, so the tool is held on the rest of its
  schema and a string is written at any length, as a free tool's is. The call
  check reads the schema you sent, so a call whose string breaks a bound
  counts `invalid`, kind `constraint`.
- **A schema that cannot be normalized leaves the tool free**, served as
  before and never refused: a keyword outside OpenAI's strict subset other
  than the length bounds ([What is refused](#what-is-refused)), an object
  that admits keys it does not name (`additionalProperties` `true` or a
  schema, or a nested object without `properties`, which closing would hold
  to `{}`), a `required` naming a key the object does not declare, a name or
  a value the call's dialect cannot write (what
  [refuses a strict tool](#held-tools-strict-true)),
  and, with thinking on, a tokenizer whose `</think>` is not one token.
- **The response's `tools` echo the `strict` each one was served with**:
  `true` where its calls are held, `false` where they are not: on fallback,
  on Flash-Next, which normalizes no tool, and under `tool_choice: "none"`,
  which holds nothing. On Flash-Next every tool is echoed `strict: false`,
  one sent `strict: true` included, inert there.
- **`strict: false` serves the tool free**, echoed as sent, so an agent CLI
  that sends `false` on every tool sees no change. Chat completions
  normalize nothing: a tool without `strict` is free there, OpenAI's default
  on that endpoint. Both change only under the operator's switch
  ([Holding every tool](#holding-every-tool)).

For a `find` tool whose `parameters` are `{"type": "object", "properties":
{"path": {"type": "string"}, "size": {"type": "integer"}, "tag": {"type":
["string", "null"]}}, "required": ["path"]}`, a call in which the model writes
`path` alone arrives with the `arguments` `{"path":"src"}`, `size` and `tag`
left out, and the response echoes the tool with `"strict": true`. A held
normalized tool's misses and repairs count as a strict tool's, with the type
`tool`.

## Holding every tool

From 0.11.0, on the 27B, an operator whose agents never send `strict: true`
can still have their calls held: `--model-tool-arguments held`
(`IGNIS_MODEL_TOOL_ARGUMENTS=held`, or `model: { tool_arguments: held }` in
the config file, [Prompting](../configuration/prompting.md)) normalizes every
tool a request does not send `strict: true`, on chat completions and on the
Responses API, over HTTP and the WebSocket mode, as the Responses API
normalizes a tool sent without `strict` by default
([above](#a-tool-sent-without-strict)): every object closed, so a call names
no parameter the tool does not declare, and an optional parameter left out of
`arguments` when the model leaves it out, the arguments valid under the
schema you sent, its length bounds aside.

- **`strict: false` is held too**: agent CLIs send it on every tool, and the
  switch is for holding every tool. On the Responses API the response's
  `tools` echo such a tool with the `strict` it was served with, `true` where
  its calls are held and `false` where they are not, as for a tool sent
  without one.
- **A schema that cannot be normalized stays free**, never refused, for the
  reasons [above](#a-tool-sent-without-strict).
- **A strict tool stays strict**: a strict schema outside the subset is the
  same `400`.
- **Misses and repairs** count as a strict tool's, with the type `tool`.
- **The default, `as-sent`**, serves every tool as the sections above say,
  byte for byte as before. Measured on the 27B, `held` is no slower, but on an
  agentic replay it changed no call, every one already valid
  ([finding](../../findings/2026-11-15-held-tools-served-on-the-27b.md)), so
  `as-sent` stays the default. Where `ignis_tool_calls_total{outcome="invalid"}`
  counts invalid calls, `held` is the switch to try, and
  `ignis_tool_call_undeclared_parameters_total` counts the calls it would
  change that JSON Schema admits: those holding a parameter that an object
  with no `additionalProperties` does not declare
  ([Telemetry](../guides/telemetry.md)).
- **On Flash-Next the switch changes nothing** until that model holds tool
  calls (GitHub #581): a Flash-Next start takes it, and serves every tool as
  before.

The switch is patchable live: after `PATCH /v1/config` with `{"model":
{"tool_arguments": "held"}}` the next request is served under it.

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

From 0.11.0, on a 27B load that drafts more than 7 tokens a round with DFlash2
(`--spec-draft-tokens` 8 to 15), a structured request's rounds take the
[round shape](../concepts/speculative-decoding.md#past-seven-drafts-two-round-shapes)
its own earlier rounds earn, as any request's do: an answer the drafter
predicts far ahead, as JSON often is, is checked and repaired in rounds of up
to the whole window.

The requests decoding beside a structured one are not touched: they keep
every draft they would have kept, and the round runs as it does without the
structured request. The only extra work is the check on the structured
request's own tokens, and a round with no structured request in it does none.

## What it costs

Measured served on the 27B on an RTX 5090, with speculation on, greedy and
thinking off, one request at a time on 19 schemas of our own
([finding](../../findings/2026-11-13-structured-output-served-on-the-27b.md)):

| | misses per 100 tokens | decode speed, against the same prompt unconstrained |
|---|---:|---:|
| the schema described (the default) | 0.4 | 98% |
| `ignis_describe_schema: false` | 4.5 | 76% |

- **A round costs the same** with a grammar and without, about 16 ms at one
  request. What a miss costs is the verified run it ends: the drafts after it
  are drafted again. With the schema described, most misses are the answer's
  first token (a model that would open with a code fence), which the
  prefill's own draw repairs before any draft; where the model writes the
  answer it would have written anyway, the speed is 100%.
- **A repair** takes about 0.3 ms, at most 1 ms in 98 of 100.
- **The requests beside a structured one** decode at 99.4% of their speed
  beside an unconstrained request on the same prompt, within the run's noise.
- **Traffic with no structured request** decodes as fast as before structured
  output, 100.0% at one request and at least 99% at eight.
- **A JSON answer is often faster than a free one.** The same prompts with no
  `response_format` were answered in prose or fenced Markdown, which drafts
  worse: the held answers decoded 1.7 times as fast. Compare a structured
  request with the same prompt and the schema in it, not with the bare prompt.

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
| a schema whose shortest answer nests deeper than `--model-grammar-max-depth` (128), or that chains its subschemas and `$ref`s deeper than `--model-grammar-max-ref-chain` (256), `strict` or not | `exceeds_max_depth`, `references_too_deep` |
| under `strict: true`, a `pattern` or `format` whose automaton has more than `--model-grammar-max-pattern-states` (20,000) states; inert without `strict` | `pattern_too_large` |
| a malformed schema (an `enum` that is not an array, an unknown `type`, a `minItems` above `maxItems`, a `$ref` that names nothing), `strict` or not | a code of its own, naming the keyword |
| `json_schema` without its `json_schema` object or its `schema`, a `strict` that is not a boolean, a `name` or `description` that is not a string | `invalid_request_error` |
| any other `type`, or no object | `unsupported_value`. `{"type": "text"}`, `null` and absent are a free answer. |
| structured output to Qwen3.8-Flash-Next | `model_unsupported`, naming the model: Flash-Next does not serve structured output yet (GitHub #581). |
| `tools` the turn may call (`tool_choice: "auto"`), on a tokenizer whose `<tool_call>` is not one token | `unsupported_value`: the grammar could not tell the call's opener from the answer. Send `tool_choice: "none"`. |
| structured output with thinking on, on a tokenizer whose `</think>` is not one token | `unsupported_value`; send `enable_thinking: false`. |

On `/v1/responses` the same refusals answer with `param: text.format`
([The Responses API](responses.md)).
