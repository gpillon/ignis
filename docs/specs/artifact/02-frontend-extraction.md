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

## Acceptance

- All 6 frontend resources are present in the container and round-tripped
  (a missing or ambiguous resource is a load failure, ADR 0002).
- The tokenizer + chat template are available to the server for request
  handling via the `FrontendSet`-backed `TemplateProvider` adapter in
  `crates/server` (replacing the built-in `SimpleTemplateProvider`).

## References

- Design: `docs/design/ignis-v1.md` §7 (frontend risk).
- ADR: 0002 (load failure on unconsumed / ambiguous objects).
- Server wiring: `crates/server/src/artifact_template.rs`
  (`ArtifactTemplateProvider`, `Server::with_artifact_template`).