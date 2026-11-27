---
status: accepted
---

# A held tool's call is checked as the model writes it, in its own order, and a value ends only where the call continues

The 27B holds a **held tool**'s call arguments to its parameters, on chat completions and the Responses API (HTTP
and WebSocket). A held tool is a **strict tool** (`strict: true`), a Responses tool sent without `strict` whose
schema normalizes into strict form (OpenAI's default there), or every tool not sent `strict: true` under the
operator's switch at `held`. The model writes its calls in the Qwen XML dialect, not JSON, and the dialect has no
escape for its own tags, so the grammar of ADR 0089 is extended to that dialect: it checks the model's own tokens
and repairs a miss. A request with no held tool changes by no byte; with one, every call's name is held to the
declared tools. The behaviour, case by case, is in the guide (`docs/user/api/structured-output.md`).

- **Every call is checked, and counted, before anything holds it.** Every call block the parser closes, held or
  not, is checked against its tool's `parameters`, so an operator learns how often calls fail before changing
  anything. The check is the grammar crate's own JSON Schema validator, which shares nothing with the automaton but
  the JSON parser and the pattern translation, agrees with an independent validator on a corpus (it reads `1.0` as
  not an integer, stricter than Draft 2020-12, as the grammar writes integers), and is the oracle of the grammar's
  property tests. A parameter the schema does not declare is a violation only where the object is closed
  (`additionalProperties: false`), as JSON Schema reads it. A right type that breaks a pattern, a format, a bound,
  a length or a count has a kind of its own, without which such a call would be invalid with no kind. A dropped
  call is a closed block that did not parse; a call still open at the end is a cut, which `finish_reason` reports,
  and is not counted. The check's own bounds on a client's schema (nesting, steps) decide only what is counted, so
  they are not configuration fields. The counts are metrics (ADR 0017), recorded on the HTTP side, since the model
  thread never sees a call; a held call's misses and repairs count beside structured output's, under a kind of
  their own.
- **What `held` would change is counted apart.** On open schemas, which most agent traffic sends, a call holding
  an undeclared parameter counts valid, so the counts above could not say what `held`, which closes such objects,
  would change. One more count takes, once per call, valid or not, the calls holding a parameter that an object
  with no `additionalProperties` does not declare, in a tool the switch at `held` would hold on the serving model:
  never on Flash-Next until #581, never a key an object admits by name, never in a tool that `held` leaves free. It
  is a count of its own, not a violation kind, since the violations count invalid calls and these may be valid. The
  grammar `held` would compile is compiled only for a call whose schema reading finds such a parameter. Likewise
  the string values of calls to tools that are not held, from which the first-occurrence cuts dropped text the
  confirmed close would keep, are counted: a lower bound that weighs extending the rule to every tool.
- **Checked, not pre-written.** The grammar arms at `<tool_call>`, the added token or its bytes, as the parser
  reads it, in the content channel only, and is released at the call's confirmed close; each further call arms it
  again, and text outside calls is free, a stop included. Where the grammar allows one continuation the repair is
  the forcing. Writing the envelope as a forced literal was rejected: the envelope is known only once the model
  chose to call and which tool, a forced literal commits one token a round and costs a constrained lane its drafts,
  and our tokenization of the envelope is not the model's (`=` merges with some names, newlines with the value's
  whitespace). Jump-forward stays deferred.
- **The model's order, not the schema's.** The template renders tools with sorted keys, so the client's order is
  one the model never saw, and a repair of a parameter's name leaves its value free, so a fixed order would put the
  value meant for one parameter under another. Each parameter is written at most once, the function closes only
  once every required parameter is written, and the delivered JSON follows the schema's key order as sent.
- **A value ends only where the call continues.** For a held tool, a line ends a string value only when it is a
  valid next step of the call under the schema: the opener of a declared parameter the call has not written, or
  `</function>` then `</tool_call>` once every required parameter is written, each after an optional forgotten
  `</parameter>`, whitespace between the tags. Anything else is the value's text. Ending a value at any tag would
  forbid the tags in every string, and escapes are a convention the model never learned. One function over bytes
  decides it for the grammar and the parser, at the three layers that cut at a tag's first occurrence (the value,
  the body, the call), so what the grammar accepts the parser delivers; a tool that is not held keeps the
  first-occurrence cuts.
- **What the grammar holds a value to.** A string with an `enum`, `const`, `pattern` or `format` is written on one
  line that does not start with a dialect tag, so its end is never in doubt and the grammar stays free of dead ends
  without a search over its lines. A string that may be `null` reads `null` as null, so it cannot carry the string
  `null`. Any other value, a string beside another type included (written quoted, as a raw `5` cannot tell the
  integer from the string), is JSON held by ADR 0089's automaton from its subschema and delivered as written,
  instead of typed by its declared types as a tool that is not held still is. Its nesting bound's ceiling is the
  call parser's own, so that every held value reads back as the JSON it was held to. A nullable parameter, a
  strict tool's or a normalized tool's required one, may be left out and arrives `null`; a normalized tool's
  optional one stays omitted.
- **Refusals.** A strict tool outside OpenAI's strict subset is refused as structured output refuses a schema,
  naming the tool's field where it was sent. `minLength` and `maxLength` lie outside the subset, so no strict tool
  has a length bound to hold. A tool that is not held is never refused for its name: one the dialect cannot write
  is just not callable. A normalized tool is never refused: what would refuse it leaves it free. The Responses API
  echoes the `strict` each tool it normalizes was served with; a `strict` the client sent is echoed as sent, but
  under `held` only `strict: true` is, and on Flash-Next every tool echoes `false`.
- **Beside the fields that shape a turn.** A call `tool_choice` forces is opened by the server's tokens, fed to
  the grammar and never a miss; a literal token the grammar would not take where the content stands is not forced,
  and the grammar holds the rest, since forcing it would serve the call unheld. Under a named function the grammar
  holds the call's name to that function, so a turn whose opening yields, or whose model opened the call itself at
  the reasoning's close, still calls the function named. With `response_format`, until the answer starts the
  opener hands the turn to the tools' grammar for the rest of the turn, and the request counts once under the
  answer's kind, the call's misses included: a request has one kind, fixed when it arrives. With thinking on the
  reasoning stays free. `parallel_tool_calls: false` ends the turn at the first call's close, the confirmed close
  for a held call, on every model.
- **The Responses default and the operator's switch.** A Responses tool sent without `strict` (or `null`) is
  normalized, as OpenAI does there, and chat completions normalize nothing. A schema is not normalized where closing
  its objects would narrow what the client asked for (an object admitting keys by name, a nested map, a `required`
  naming an undeclared key), nor where the subset or the dialect refuses it. One keyword is the
  exception: a normalized tool leaves `minLength` and `maxLength` out and is held on the rest of its schema, since
  leaving it free for a bound the grammar does not hold gave up its types, its enums and its closed objects. Its
  strings are then written at any length, as a free tool's are; the call check still reads the schema as sent, so a
  string past its bound still counts as a violation, and a bound the grammar does not read is never a miss, so no
  repair is asked to satisfy it. `model.tool_arguments` (`as-sent`, the default, or `held`) is a live field read
  once per request: `held` normalizes every tool not sent `strict: true`, `strict: false` included because the
  agent CLIs send it on every tool. The field is every family's, so one configuration file starts either model; it
  changes nothing on Flash-Next until #581. The default stays `as-sent`: `held` measured no slower on the 27B, but
  on the owner's agentic replay it changed no call, every one already valid, and the corpus's one repair wrote a
  value the free call left out, so nothing measured justifies leaving OpenAI's defaults
  ([finding](../findings/2026-11-15-held-tools-served-on-the-27b.md)).
- **The documented exceptions to OpenAI's behaviour.** Flash-Next keeps its tools free until #581: `strict` is
  accepted and inert, since agent SDKs send it by default and a 400 would shut them out. A string value cannot hold
  a line that is a step of its call, nor, while a required parameter is missing, the two line starts at which the
  model is steered to that parameter; whitespace after a step's first tag is bounded, so a run of spaces is never
  a dead end. A named function whose name holds `>`, which the dialect cannot write, leaves the call to any
  declared tool. And the dialect refuses some strict tools that may lie inside the strict subset: a name or a
  parameter name it cannot write, a name another tool shares, a held string none of whose values fits on one line,
  and, with thinking on, a tokenizer whose `</think>` is not one token.
