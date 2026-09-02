# 02 — frontend object extraction (tokenizer + chat template)

Status: needs-triage
GitHub: #7
Blocked by: #4 (artifact-01)

Extract and verify the **frontend object set** carried inside the artifact
container (confirming the `docs/design/ignis-v1.md` §7 risk that the
frontend is carried by the container):

- `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
  `generation_config.json`, `preprocessor_config.json`,
  `video_preprocessor_config.json`.
- Expose the tokenizer + chat template to the HTTP server (`server-01`) so
  requests are tokenized / templated from the artifact, not re-shipped.

## Acceptance

- All 6 frontend resources are present and round-tripped.
- The tokenizer + chat template are available to the server for request
  handling.
