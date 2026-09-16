# Chat-render fixtures (GitHub #184)

One-off tooling that records the prompt text the reference chat template
(ninfer `src/targets/qwen3_6/impl/frontend/chat_template.cpp`) renders for a
small set of conversations, so `crates/artifact/tests/chat_render.rs` can hold
ignis's minijinja render of the same messages to it byte for byte.

The cases exist for the rendering prerequisite of cross-request state reuse
(spec `.scratch/kv-reuse/specs/01-cross-request-reuse.md` §Rendering
prerequisites, ADR 0029): a resent tool call must render its parameters in the
order the model emitted them. `tool_call_key_order` and
`tool_call_reversed_keys` carry the same five keys in opposite orders, so no
implementation that sorts them can satisfy both.

Nothing here needs the GPU.

| File | What |
|------|------|
| `cases.json` | The cases: messages, tools, thinking options. A tool call's `arguments` is the wire string, verbatim, so the case file itself cannot reorder it. |
| `record.cpp` | The recorder: links the reference's `build-ninja` static libraries and renders each case through `CompiledChatTemplate`. |
| `build.ps1` | Builds `record.exe` with MSVC against `F:\ai\q38\ninfer\build-ninja` (the same recipe as `tools/vision-fixtures/build.ps1`). |

## Re-record

```powershell
$Artifact = "F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer"
$Frontend = "$env:TEMP\ignis-chat-render-frontend"
cargo run -p ignis-artifact --example dump_frontend -- $Artifact $Frontend
powershell -NoProfile -ExecutionPolicy Bypass -File tools/chat-render-fixtures/build.ps1
& "$env:TEMP\ignis-chat-render-recorder\record.exe" $Frontend\chat_template.jinja `
  tools/chat-render-fixtures/cases.json crates/artifact/tests/fixtures/chat_render
cargo test -p ignis-artifact --test chat_render
```

The recorder is fed the artifact's own `chat_template.jinja`, not a copy:
`CompiledChatTemplate::resolve` accepts a source only by SHA-256, and the
artifact's template hashes to the reference's `kReasoningEffortTemplateDigest`
(`c3cf9e34…d7a81041`), so both engines render from the same bytes.

## Notes

- `enable_thinking` is false in every recorded case. A thinking-on multi-turn
  render still diverges on the history think block (GitHub #182), which is
  #185's subject — `preserve_thinking` does not reach the template yet.
- The reference takes tool definitions as JSON strings that the request's own
  `nlohmann::json` already dumped with sorted keys, so the recorder re-parses
  each case's tool as a plain (sorted) `nlohmann::json` before dumping it.
  That is what ignis's `serde_json`-carried `tools` array produces too.
- The recorder also records `shared_prefix_offset`, the reference's end of the
  leading system/tools block. Nothing asserts it yet; slice 3 (#188) is where
  it earns its place.
