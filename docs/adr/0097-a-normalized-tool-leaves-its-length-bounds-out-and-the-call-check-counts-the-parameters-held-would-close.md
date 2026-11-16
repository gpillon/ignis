---
status: accepted
amends: [ADR-0017, ADR-0094]
---

# A normalized tool leaves its length bounds out, and the call check counts the parameters held would close

Two of ADR 0094's rules change (GitHub #641: the owner's calls on items 10 and 11 of #582's review).

A tool the server normalizes into strict form leaves `minLength` and `maxLength` out of it and is held on the rest of
its schema. That is a tool sent without `strict` on the Responses API, or any tool not sent `strict: true` under
`model.tool_arguments: held`. ADR 0094 left such a tool free for any keyword outside OpenAI's strict subset.

- **Why.** A length bound is one keyword the grammar does not hold, and leaving the tool free for it gave up its
  types, its enums and its closed objects. Its strings are now written at any length, as a free tool's are.
- **The call check is unchanged.** It reads the schema as sent, so a string past its bound still counts
  `constraint`. A bound the grammar does not read is never a miss, so no repair is asked to satisfy it.
- **A strict tool with a length bound is still refused.** The spec's "length bounds enforced" (US 7) gives way to
  its strict-subset rule, so no strict tool has a length bound to hold.
- **Any other keyword outside the subset** still leaves a normalized tool free.

ADR 0094 counts a parameter the schema does not declare only where the object is closed, as JSON Schema reads it.
On open schemas, which most agent traffic sends, a call holding one counted `valid`. The counters could therefore
not say what `held`, which closes such objects, would change.

A new counter, `ignis_tool_call_undeclared_parameters_total`, joins ADR 0017's families through ADR 0094's. It counts
once per call, valid or not, the calls holding a parameter that an object with no `additionalProperties` does not
declare, in a tool the switch at `held` would hold on the serving model. It never counts:

- on Flash-Next, until #581;
- a key an object admits by name;
- in a tool that `held` leaves free.

It is a family of its own, not a violation kind: `ignis_tool_call_violations_total` counts invalid calls, and these
calls may be valid. The tool calls' grammar that `held` would compile is compiled only for a call whose schema reading
finds such a parameter.
