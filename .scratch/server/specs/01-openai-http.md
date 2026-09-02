# 01 — OpenAI-compatible HTTP (models / chat / responses)

GitHub: #14

OpenAI-compatible HTTP on **localhost, no auth, configurable bind**
(`docs/design/ignis-v1.md` §1). Route requests into the core scheduler and
stream tokens back:

- `GET /v1/models` — list the loaded model(s).
- `POST /v1/chat/completions` — chat completions (streaming + non
  streaming); the chat template is applied from the artifact's frontend
  object set (`artifact-02`).
- `POST /v1/responses` — the responses API.

Delivered (commit 84ada6d): `crates/server` (axum 0.8 + tokio) —
`Server` (engine + injected `TemplateProvider` + request timeout) with
`app()`/`serve()`; `GET /v1/models`; `POST /v1/chat/completions`
(streaming SSE `ChunkStream` + non-streaming); `POST /v1/responses`
(non-streaming; `stream:true` → 400 in v1). The `Engine` drives the core
`Scheduler` behind a `Mutex`: atomic `submit` + a driver loop that steps
the scheduler and routes `SchedEvent`s into each request's unbounded
stream. Review-caught bug fixed: a `Protected` (admission-batch) event
early-returned, dropping `Token`/`Done` events later in the same batch —
now skipped and pinned by a regression test. CPU-tested (ADR 0006):
21/21 `ignis-server` tests green, workspace `cargo test` green.

Note: the template seam is fully wired (follow-up, commit 217a0bd) — the
`FrontendSet`-backed `ArtifactTemplateProvider` (`artifact_template.rs`)
implements the `TemplateProvider` trait with the container's real chat
template + HuggingFace tokenizer: `apply_chat_template` = template
`render` + tokenizer `encode`, `render_tokens` = `decode`. The entrypoint
reads `IGNIS_ARTIFACT` and uses `Server::with_artifact_template` when set;
unset/unreadable falls back to the built-in `SimpleTemplateProvider`
(rendered `content` is the token id-space, not natural text). A role that
does not parse to an artifact `Role` templates as `user` (infallible by
design — logged, not panicked). `/v1/responses` streaming is out of v1
scope (non-streaming only). The `Compute` backend is `MockCompute`; the
kernel-leaf adapter replaces it via the same scheduler-constructor
injection (ADR 0001/0006).

## Acceptance

- `GET /v1/models` returns the loaded model.
- `POST /v1/chat/completions` routes into the scheduler and streams tokens;
  the chat template is applied from the artifact frontend objects.
- `POST /v1/responses` works (OpenAI responses shape).
