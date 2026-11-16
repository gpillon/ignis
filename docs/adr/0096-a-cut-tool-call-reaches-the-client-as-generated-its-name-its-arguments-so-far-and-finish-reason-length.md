---
status: accepted
amends: [ADR-0058]
---

# A cut tool call reaches the client as generated: its name, its arguments so far, and finish_reason length

A tool call that a `length` ending stops before it closes, by the **generation cap** or by a passed **deadline**
under `timeout_reply: partial`, reaches the client with its name and the arguments written so far, as OpenAI's
API returns one: on chat completions with `finish_reason: "length"`, on the Responses API as a `function_call`
item with `status: "incomplete"` in an `incomplete` response. This replaces ADR 0058's "a tool call cut mid-way
is dropped whole" and the rule #121 set before it (GitHub #583). `length` already tells the client the answer is
incomplete, so #121's premise, never to tell a client that a call happened when it did not complete, still
holds, and structured output (#287) returns a JSON answer cut by the cap as generated too. Every other ending (a
stop sequence, a cancel, a disconnect, an engine error) still drops an open call whole, since `stop` would call
it complete; a call cut before its function name closes has nothing to name and is dropped. The arguments are
the prefix of the JSON the closed call would have had, as far as the written text determines it. A cut call is
not counted in `ignis_tool_calls_total`, as ADR 0094 says.
