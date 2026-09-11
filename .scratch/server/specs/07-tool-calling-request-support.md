# 07 — OpenAI `tools` / `tool_choice` request support (function calling)

GitHub: TBD (created alongside this spec)

## Problem Statement

#121 hardened the *response* side of tool calling: a `<tool_call>` block
the model emits is parsed and reassembled into OpenAI `tool_calls`
correctly, chunk-boundary splits and all. But nothing on the *request*
side ever gets the model to emit one in the first place — ignis accepts
no `tools` field, renders no tools section into the prompt, and drops any
`tool_calls` a prior assistant turn carried. A client can only exercise
#121's hardening by hand-writing the model's own tag dialect into a user
message; the standard flow (`tools: [...]`, the model decides, the
client executes and replies with `role: "tool"`) does not work at all.

The gap sits entirely above the seam that already understands this:
`ignis_artifact::frontend::ChatTemplate` already renders a `tools`
context variable into the "# Tools" system section
(`.qwen/tmp/chat_template.jinja`), and `ignis_artifact::ChatMessage`
already carries a `tool_calls: Vec<ToolCall>` field that `message_to_json`
already serializes in the exact shape the template expects. Nothing above
that seam — the HTTP wire types, the `TemplateProvider` trait, the
`SimpleTemplateProvider` placeholder — passes a `tools` value down to it.

## Solution

Thread `tools` from the wire down to the template, opaque JSON end to
end — the same "the wire contract is the reference's, field for field"
posture #68 took for thinking:

- `ChatCompletionsRequest` gains `tools: Option<Vec<JsonValue>>` and
  `tool_choice: Option<JsonValue>`. `tools` is validated shallowly (each
  entry an object with `"type": "function"` and a `function.name`
  string) — anything else is a 400 naming the bad entry, not a silent
  drop. `tool_choice` accepts `"auto"` (default behaviour) and `"none"`
  (tools are validated but never reach the template — the model is never
  told they exist, achieving "must not call a function" the only way a
  text-instruction template can). A forced choice — `"required"` or the
  named-function object form — is a 400 explaining that this template has
  no lever to *force* a call, rather than accepting the field and quietly
  not honouring it.
- `TemplateProvider::apply_chat_template` gains a `tools: &[JsonValue]`
  parameter (empty = no tools, today's behaviour unchanged). Every
  implementor updates: `SimpleTemplateProvider` ignores it (no jinja
  template to bind it into, same posture as its `ThinkingOptions`
  handling); `ArtifactTemplateProvider` passes it straight into a new
  `ChatTemplate::render_with_thinking_and_tools` context key.
- `ChatMessage` (the HTTP wire type, `template.rs`) gains
  `tool_calls: Option<Vec<ToolCallIn>>` (an assistant history turn's
  prior calls: `id`, `function.name`, `function.arguments` — the OpenAI
  wire shape, arguments as a JSON-encoded *string*, matching what #121's
  own response side emits). Converting to `ignis_artifact::ChatMessage`
  parses that string into the `JsonValue` object the template's
  `arguments|items` needs; a call whose arguments do not parse as a JSON
  object degrades to an empty object (logged, not a 500 — matches this
  seam's existing "a render/encode failure degrades, never panics"
  posture). A `role: "tool"` message's `tool_call_id` is accepted on the
  wire and otherwise unused — this template correlates tool results
  sequentially, not by id (verified against `.qwen/tmp/chat_template.jinja`).
- `/v1/responses` is untouched — no `tools` field, same as it already
  carries no `reasoning_content` or `tool_calls`. Only
  `/v1/chat/completions` gets this.

## Acceptance Criteria

- [ ] A request with a well-formed `tools` array reaches the template
      (observable through a recording double in tests) and, against the
      real template, renders the "# Tools" section.
- [ ] A malformed `tools` entry (missing `type`/`function.name`, wrong
      shape) is a 400 naming the problem.
- [ ] `tool_choice: "none"` suppresses `tools` from reaching the template
      even when the request supplied a well-formed array.
- [ ] `tool_choice: "required"` or a named-function object is a 400, not
      a silently-ignored field.
- [ ] An assistant history message's `tool_calls` reaches the template as
      the `tool_calls` context the real jinja template iterates over
      (observable via the real template rendering a `<tool_call>` block
      for it on the next turn).
- [ ] `cargo test` green workspace-wide, no GPU needed for any of the
      above (a small fixture template + tokenizer, mirroring
      `artifact_template.rs`'s existing `THINKING_TEMPLATE` fixture).
