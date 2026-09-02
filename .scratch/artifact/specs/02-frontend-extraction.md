# 02 — frontend object extraction (tokenizer + chat template)

GitHub: #7

Extract and verify the **frontend object set** carried inside the artifact
container (confirming the `docs/design/ignis-v1.md` §7 risk that the
frontend is carried by the container):

- `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
  `generation_config.json`, `preprocessor_config.json`,
  `video_preprocessor_config.json`.
- Expose the tokenizer + chat template to the HTTP server (`server-01`) so
  requests are tokenized / templated from the artifact, not re-shipped.

Delivered (commit d08759d): the `frontend` module —
`FrontendSet::from_reader(&Reader)` loads all 6 frontend resources (a
missing **or ambiguous** resource is a load failure, ADR 0002), the typed
`Tokenizer` (HuggingFace `tokenizers` 0.21: `encode`/`decode` from the
container's `tokenizer.json` bytes) and `ChatTemplate` (minijinja 2.24 +
`json`; the Qwen3.8-specific extensions registered: `raise_exception` via
`add_function`, string `.startswith`/`.endswith` via the unknown-method
callback; `render(&[ChatMessage]) -> String`). 39 unit tests + real-artifact
verification (`real_frontend.rs`, gated like `real_artifact.rs`, CPU-only):
all 6 resources present in the 19 GB `qwen3_8_27b_nvfp4full-v2` container,
the real BPE tokenizer round-trips, the real Qwen3.8 template compiles +
renders a 2-message conversation, and its `raise` error path fires
end-to-end.

Note: the last wiring step — a `FrontendSet`-backed `TemplateProvider`
adapter in `crates/server` (replacing the built-in
`SimpleTemplateProvider`) — is the tracked follow-up (PENDING.md ledger);
it needs `ignis-core` to compile, which was blocked by the in-flight core
WIP at the time of writing.

## Acceptance

- All 6 frontend resources are present and round-tripped.
- The tokenizer + chat template are available to the server for request
  handling.
