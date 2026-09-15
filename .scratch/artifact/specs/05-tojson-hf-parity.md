# 05 — tool definitions in the prompt: byte parity with the reference

GitHub: #172

The container chat template renders tool definitions (`{{- tool | tojson }}`)
and non-string tool-call arguments (`args_value | tojson | safe`) with
`tojson`. ignis used minijinja 2.24's built-in filter; the reference renders
the same template natively (`chat_template.cpp` `tojson_text`) after
normalizing each tool at the HTTP layer (`openai_schema.cpp` `parse_tools`):

| | reference | ignis before |
|---|---|---|
| separators | `", "` and `": "` | `","` and `":"` (`serde_json::to_string`) |
| `< > & '` | literal | `< > & '` (HTML-safe escaping) |
| floats | Python spelling (`1e-05`, `1e+16`) | ryu (`0.00001`, `1e16`) |
| key order | sorted (`nlohmann::json`) | sorted (serde_json) — already equal |
| missing/`null` `function.strict` | `false` added | left out |
| missing/`null` `function.parameters` | `{"type": "object", "properties": {}}` | left out |

Measured 2026-09-15 (`.scratch/diag-160/twenty/`): on the G4 trace prompts
with 4 tools, the ignis prompt was a constant 84 tokens shorter; the filter
alone brought it to 16 (the `strict` fields).

## Owner decisions (2026-09-15)

- **Parity with the reference, not with HuggingFace's key order.** HF keeps
  the client's order; the reference sorts. ignis matches the reference, so
  the request path stays `serde_json::Value` (no raw-text plumbing).
- The custom `tojson` lives only in the chat template's `Environment`; the
  rest of the workspace keeps serde_json's own behavior.

## Seam

- `crates/artifact/src/frontend.rs` `register_template_builtins`: a
  `tojson` filter overriding minijinja's, writing `", "` / `": "`
  (`","` / `": "` with `indent`), no HTML escaping, Python float spelling,
  strings escaped as `json.dumps(ensure_ascii=False)`.
- `crates/server/src/api.rs` `resolve_tools`: the reference's two defaults.

## Acceptance

1. A `tools` array rendered through the real template's tools section
   matches the reference's `render_tools_system_block` bytes.
2. Non-string tool-call arguments in the history render with the same
   separators.
3. Floats and control characters match `json.dumps(ensure_ascii=False)`;
   `tojson(indent=N)` matches `json.dumps(..., indent=N)`.
4. A tool without `strict` / `parameters` reaches the template with the
   reference's defaults; one that carries them is unchanged.
5. `cargo test --workspace` passes.
6. Live (informational): with-tools prompt token counts equal the
   reference's on the 20-prompt set.

## Known limits

- Integers beyond `u64`/`i64` arrive as floats (serde_json without
  `arbitrary_precision`).
- `tojson` accepts only `indent`; the container template passes nothing.
- The reference also 400s a non-boolean `strict` or non-object
  `parameters`; ignis passes them through.

## References

- Spec 02 (frontend extraction), GitHub #132 (tools through the template).
- Reference: `ninfer/src/targets/qwen3_6/impl/frontend/chat_template.cpp`
  (`tojson_text`, `render_tools_system_block`),
  `ninfer/src/serve/openai_schema.cpp` (`parse_tools`).
