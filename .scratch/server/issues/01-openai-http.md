# 01 — OpenAI-compatible HTTP (models / chat / responses)

Status: needs-triage
Blocked by: core-04, artifact-02

OpenAI-compatible HTTP on **localhost, no auth, configurable bind**
(`docs/design/ignis-v1.md` §1). Route requests into the core scheduler and
stream tokens back:

- `GET /v1/models` — list the loaded model(s).
- `POST /v1/chat/completions` — chat completions (streaming + non
  streaming); the chat template is applied from the artifact's frontend
  object set (`artifact-02`).
- `POST /v1/responses` — the responses API.

## Acceptance

- `GET /v1/models` returns the loaded model.
- `POST /v1/chat/completions` routes into the scheduler and streams tokens;
  the chat template is applied from the artifact frontend objects.
- `POST /v1/responses` works (OpenAI responses shape).