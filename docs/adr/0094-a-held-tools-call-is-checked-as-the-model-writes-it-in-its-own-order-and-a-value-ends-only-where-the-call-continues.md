---
status: proposed
amends: [ADR-0017, ADR-0089]
---

# A held tool's call is checked as the model writes it, in its own order, and a value ends only where the call continues

GitHub #582 holds a **held tool**'s call arguments to its parameters. A held tool is one of three things: a
**strict tool** (`strict: true`); a Responses tool sent without `strict` whose schema normalizes into strict
form, which is OpenAI's default there; or every tool, when the operator turns on `held`. The model writes its
calls in the Qwen XML dialect (`<tool_call>`, `<function=…>`, `<parameter=…>`), not in JSON, and that dialect
has no escape for its own tags. ADR 0089's grammar is extended to that dialect. It checks the model's own tokens
and repairs a miss, and it does not write the call's envelope for the model.

- **Every call is checked first, and counted.** Before anything holds a call, every call block the parser closes,
  held or not, on chat completions and on Responses, is checked against its tool's `parameters`, and counted
  (#608). This is how an operator learns how often the model's calls fail before anything about them changes.
  - The check is the grammar crate's validator, `ignis_grammar::validate`: JSON Schema's own reading (Draft
    2020-12), sharing nothing with the automaton but the JSON parser and the pattern translation, its verdicts
    held to an independent validator's on a corpus. It is also the oracle of the grammar's property tests:
    everything the grammar accepts must be valid there.
  - A parameter the schema does not declare is a violation only where the schema closes its object
    (`additionalProperties: false`, as a strict schema must and normalization does). On an open object it is
    valid, as JSON Schema reads it.
  - The kinds are the grill's five (missing, type, enum, unknown parameter, undeclared tool) and a sixth,
    `constraint`: a value of the right type that breaks its pattern, format, a bound, a length or a count.
    Without it such a call would be invalid with no kind.
  - A dropped call is a closed block that did not parse, which left no trace before. A call still open when
    generation ends is a cut, which `finish_reason` already reports, and is not counted.
  - The family is ADR 0017's projection, recorded on the HTTP side as the rejections are, since the model
    thread never sees a call: `ignis_tool_calls_total{outcome}` (`valid`, `invalid`, `dropped`),
    `ignis_tool_call_violations_total{kind}` (`missing`, `type`, `enum`, `unknown_parameter`, `constraint`,
    `undeclared_tool`; a call once under each kind it has), and `ignis_tool_strict_outside_subset_total`
    (strict tools whose schema lies outside the strict subset, once per request that sends one, read by the
    grammar's own refusal and so only on a model that serves structured output). It is absent until one of
    them moves, then whole. The scanner checks only with metrics on, so a server without them does no work.
- **Checked, not pre-written.** The grammar arms on the `<tool_call>` token, in the content channel only, and is
  released at the call's close. Each further call arms it again.
  - Where the grammar allows a single continuation, a repair is the forcing.
  - Writing the envelope as a forced literal was rejected, for three reasons. The envelope is known only after
    the model chose to call and which tool. A forced literal commits one token per round, and a round with a
    constrained lane loses its drafts. And our tokenization of the envelope is not the model's: `=` merges with
    some parameter names, and newlines merge with the value's whitespace.
  - Jump-forward stays deferred, as in ADR 0089.
- **The model's order, not the schema's.** The template renders tools with sorted keys, so the client's order is
  one the model never saw.
  - A repair of a parameter's *name* leaves its value free, so a fixed order would put the value the model meant
    under another name.
  - No parameter is written twice.
  - `</function>` is allowed only once every required parameter is written.
  - The JSON delivered follows the schema's order.
- **A value ends only where the call continues.** A tag on its own line counts only when it is a valid next step
  of this call under the schema:
  - `<parameter=K>`, for a K the tool declares and the call has not written yet;
  - or `</function>` followed by `</tool_call>`, once every required parameter is written.

  A forgotten `</parameter>` before either is tolerated, as the parser tolerates it today. Anything else is the
  value's text.
  - One boundary function, over bytes, decides this for the grammar and for the parser, at all three layers that
    cut today at a tag's first occurrence (the call, the function body, the value). It applies to top-level free
    string values only.
    - It is `ignis_grammar::call::line` (#610). It reads the bytes from a line's start and answers text, a step, or
      not decided yet. Once text or a step, more bytes never change the answer, so a grammar can read a line byte by
      byte.
    - A line starts at a value's first byte and after each `\n`, and a tag counts only at a line's first byte.
    - Between a step's tags, any run of spaces, tabs, `\r` and `\n` is allowed, none included, since the tokenizer
      merges newlines with the whitespace beside them.
    - A parameter's name runs to the first `>`, as the parser reads it. A required name the schema does not declare
      is never written, so such a call never closes.
    - The caller says which parameters are required, so a strict tool's nullable ones can be left out.
    - The call parser chooses the rule per call, by the function's name. A tool not held keeps the first-occurrence
      cuts, and a held tool's call ends only at its confirmed close.
  - Two alternatives were rejected. Ending a value at any of the three tags would forbid them in every string.
    Escapes are a convention the model never learned.
  - The residue: a string value cannot contain this call's own confirmed close.
  - The rule is not extended to tools that are not held, and the extension is weighed by a count: with metrics on,
    `ignis_tool_call_values_split_total` (an amendment of ADR 0017 beside #608's family). It counts the string
    values of those tools' calls from which the first-occurrence cuts dropped text after a tag, text the confirmed
    close would keep in the value: what the cuts skip after a `</parameter>` or between `</function>` and
    `</tool_call>`, other than whitespace.
    - It does not count a value whose text the cuts moved into another parameter, dropping nothing. An opener of a
      written or undeclared parameter may be the model's own envelope, which the confirmed close, without a
      grammar, would read as the value's text.
    - It does not count a value the confirmed close would end at no step of the block. That excludes a close with a
      required parameter missing, which that rule would not deliver, and a value whose end lies past the block's
      first `</tool_call>`. The count is a lower bound.
- **A missing required parameter is a miss at the close.**
  - After `</parameter>` and its newline, `</` is a miss.
  - Inside a value, `function` after a line-start `</` is a miss.

  How often the repair then reaches the missing parameter is measured on the GPU.
- **Omitted parameters.**
  - For a strict tool, a nullable parameter may be omitted, and the JSON carries `null`.
  - For a normalized tool, an optional parameter may be omitted, and stays omitted, so the JSON is valid under
    the client's own schema.
  - A normalized tool's objects are closed: no parameter the schema does not declare.
- **Refusals.**
  - `strict: true` with a schema outside OpenAI's strict subset is a 400 naming the constraint, as
    `response_format` is.
  - A schema that cannot be normalized leaves the tool free, and the response echoes `strict: false`, as
    OpenAI's fallback does. It is never a 400.
- **Scope.**
  - The 27B. On Flash-Next, tools stay free until #581: `strict` is accepted and inert, and the guide says so.
    Agent SDKs set `strict` by default, and a 400 would shut them out of Flash-Next.
  - A request with no held tool changes by no byte. When one is present, the function name is held to the
    declared tools.
  - `parallel_tool_calls: false` ends the turn at the first call's close, on every model, and needs no grammar.
