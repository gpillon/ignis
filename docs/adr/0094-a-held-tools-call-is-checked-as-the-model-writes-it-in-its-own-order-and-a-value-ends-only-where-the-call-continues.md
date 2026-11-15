---
status: proposed
amends: [ADR-0017, ADR-0089]
---

# A held tool's call is checked as the model writes it, in its own order, and a value ends only where the call continues

GitHub #582 holds a **held tool**'s call arguments to its parameters on the 27B, on chat completions and the
Responses API (HTTP and WebSocket). A held tool is a **strict tool** (`strict: true`), a Responses tool sent
without `strict` whose schema normalizes into strict form (OpenAI's default there), or every tool not sent
`strict: true` under the operator's switch at `held`. The model writes its calls in the Qwen XML dialect
(`<tool_call>`, `<function=…>`, `<parameter=…>`), not JSON, and the dialect has no escape for its own tags. ADR
0089's grammar is extended to that dialect: it checks the model's own tokens and repairs a miss.

**Every call is checked, and counted, before anything holds it.** Every call block the parser closes, held or not,
is checked against its tool's `parameters`, so an operator learns how often calls fail before changing anything.
The check is the grammar crate's own JSON Schema validator, which shares nothing with the automaton but the JSON
parser and the pattern translation, agrees with an independent validator on a corpus (it reads `1.0` as not an
integer, stricter than Draft 2020-12, as the grammar writes integers), and is the oracle of the grammar's property
tests. A parameter the schema does not declare is a violation only where the object is closed
(`additionalProperties: false`), as JSON Schema reads it. The kinds are missing, type, enum, unknown parameter,
undeclared tool, and `constraint` (a right type breaking a pattern, format, bound, length or count), without which
such a call would be invalid with no kind. A dropped call is a closed block that did not parse; a call still open
at the end is a cut, which `finish_reason` reports, and is not counted. The check's own bounds on a client's
schema (nesting, steps) decide only what is counted, so they are not configuration fields.

**The families (this ADR's amendment of ADR 0017).** Recorded on the HTTP side, as the rejections are, since the
model thread never sees a call, absent until one moves, and only with metrics on:
`ignis_tool_calls_total{outcome}` (`valid`, `invalid`, `dropped`); `ignis_tool_call_violations_total{kind}`
(`missing`, `type`, `enum`, `unknown_parameter`, `constraint`, `undeclared_tool`; a call once per kind it has);
`ignis_tool_strict_outside_subset_total`, strict tools outside the strict subset, once per request, counted as it
arrives and then refused, read only on a model that serves structured output; and
`ignis_tool_call_values_split_total`, string values of calls to tools that are not held from which the
first-occurrence cuts dropped text the confirmed close would keep, a lower bound that weighs extending the rule to
every tool. ADR 0089's three structured-output counters gain the `type` value `tool` (the amendment of ADR 0089
and ADR 0017), and the request log carries `structured.type = "tool"`.

**Checked, not pre-written.** The grammar arms at `<tool_call>` (the added token or its bytes, as the parser reads
it), in the content channel only, and is released at the call's confirmed close; each further call arms it again,
and text outside calls is free, a stop included. It is one byte automaton per distinct set of declared tools,
compiled when the request arrives and cached with the schema grammars, which the scheduler holds as it holds an
answer: every content draw checked, a verified run cut at its first miss inside a call, and where the grammar
allows one continuation the repair is the forcing. Added tokens are read as the text the decoder writes for them.
Writing the envelope as a forced literal was rejected: the envelope is known only once the model chose to call and
which tool, a forced literal commits one token a round and costs a constrained lane its drafts, and our
tokenization of the envelope is not the model's (`=` merges with some names, newlines with the value's
whitespace). Jump-forward stays deferred.

**The model's order, not the schema's.** The template renders tools with sorted keys, so the client's order is one
the model never saw, and a repair of a parameter's name leaves its value free, so a fixed order would put the
value meant for one parameter under another. Each parameter is written at most once, the function closes only once
every required parameter is written, and the delivered JSON follows the schema's key order as sent, which `tools`
is read in once beside its sorted value.

**A value ends only where the call continues.** For a held tool, a line ends a string value only when it is a
valid next step of the call under the schema: the opener of a declared parameter the call has not written, or
`</function>` then `</tool_call>` once every required parameter is written, each after an optional forgotten
`</parameter>`, whitespace between the tags. Anything else is the value's text. Ending a value at any tag would
forbid the tags in every string, and escapes are a convention the model never learned. One function over bytes
decides it for the grammar and the parser, at the three layers that cut at a tag's first occurrence today (the
value, the body, the call), so what the grammar accepts the parser delivers; a tool that is not held keeps the
first-occurrence cuts. **The residue**, documented: a string value cannot hold a line that is a step, so the
opener of a declared parameter not yet written (a file written before that parameter ends at an example of it,
which then opens it), nor, once the call may close, its close. While a required parameter is missing a free string
also cannot start a line `</function`, nor go on from a `</parameter>` line with `</` (the two miss points spike
S1 of spec #582 measured on the 27B's tokenizer, where the model is steered to the missing parameter), and the
whitespace after a step's first tag is bounded at `model.grammar_max_whitespace`, the bound's last byte kept for a
line break where the next step must start a line, so a run of spaces is never a dead end.

**What the grammar holds a value to.** A string with an `enum`, `const`, `pattern` or `format` is written on one
line that does not start with a dialect tag, so its end is never in doubt and the grammar stays free of dead ends
without a search over its lines. A string that may be `null` reads `null` as null, so it cannot carry the string
`null`. Any other value, a string beside another type included (written quoted, as a raw `5` cannot tell the
integer from the string), is JSON held by ADR 0089's automaton from its subschema, delivered as written, compact
and canonical, in the grammar's subset, instead of typed by its declared types as a tool that is not held still
is; it nests at most `model.grammar_max_depth`, whose ceiling is the call parser's own nesting bound so that every
held value reads back as the JSON it was held to. A strict tool's nullable parameter may be left out and arrives
`null`; a normalized tool's optional one stays omitted.

**Refusals.** A strict tool outside OpenAI's strict subset is a 400 with structured output's codes, `param` naming
the tool's field where it was sent (`tools[i].function.parameters` on chat completions, `tools[i].parameters` for
a flat Responses tool). `minLength` and `maxLength` lie outside the subset, so no strict tool has a length bound
to hold. A tool that is not held is never refused for its name: one the dialect cannot write is just not callable.
A normalized tool is never refused: what would refuse it leaves it free, and the Responses API echoes the `strict`
each tool sent without one was served with, a `strict` sent echoed as sent but on Flash-Next, where every tool
echoes `false`.

**Beside the fields that shape a turn.** `tool_choice` `"required"` or a named function forces the call's opening
(#286), its tokens fed to the grammar and never a miss; a literal token the grammar would not take where the
content stands is not forced, and the grammar holds the rest, since forcing it would serve the call unheld. Under
a named function the grammar holds the call's name to that function, so a turn whose opening yields, or whose
model opened the call itself at the reasoning's close, still calls the function named. With `response_format`,
until the answer starts the opener hands the turn to the tools' grammar for the rest of the turn, and the request
counts once under the answer's `type`, the call's misses included: a request has one kind, fixed when it arrives.
With thinking on the reasoning stays free. `parallel_tool_calls: false` ends the turn at the first call's close,
the confirmed close for a held call, on every model.

**The Responses default and the operator's switch.** A Responses tool sent without `strict` (or `null`) is
normalized, as OpenAI does there, and chat completions normalize nothing: every object closed and listed in
`required`, the top-level optional parameters omissible, a nested object written whole. A schema is not normalized
where closing would narrow what the client asked for (an object admitting keys by name, a nested map, a `required`
naming an undeclared key), nor where the subset or the dialect refuses it. `model.tool_arguments` (`as-sent`, the
default, or `held`) is a live field read once per request: `held` normalizes every tool not sent `strict: true`,
`strict: false` included because the agent CLIs send it on every tool. The field is every family's, so one
configuration file starts either model; it changes nothing on Flash-Next until #581. Its values join a value of
several words by `-`, as the configuration spells them. The default changes only if the served close-out (#617)
measures `held` on the 27B as no slower, with sensible repairs.

**Scope and documented exceptions.** A request with no held tool changes by no byte; with one, every call's name
is held to the declared tools. Flash-Next keeps its tools free until #581: `strict` is accepted and inert, as the
API reference's field table and the guide say, since agent SDKs send it by default and a 400 would shut them out.
The documented exceptions to OpenAI's behaviour: Flash-Next until #581; the residue above; a named function whose
name holds `>`, which the dialect cannot write and which leaves the call to any declared tool; and the dialect's
own refusals of a strict tool that may lie inside the strict subset: a name holding `>` or `</tool_call`
(`tool_name_unwritable`), a name another tool shares (`duplicate_tool_name`), a parameter name holding `>`
(`parameter_name_unwritable`), a held string none of whose values fits on one line (`no_writable_value`), and,
with thinking on, a tokenizer whose `</think>` is not one token (`unsupported_value`, `param` naming the tool's
`strict`).
