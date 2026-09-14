# 02 — Playground chat with browser-measured request figures

GitHub: #164

ADR: 0026. Glossary: **Playground**, **Lane tag** (`CONTEXT.md`). Builds on 01.

## Problem Statement

With the Playground page served (01), the owner still cannot talk to the model from it or see how a request behaved (how long to first token, decode speed, token counts) without curl and a stopwatch.

## Solution

A single-page chat in `web/` against ignis's own `POST /v1/chat/completions` with `stream: true` and `stream_options.include_usage: true`. Every figure is measured in the browser or read from the standard response; the server is unchanged.

## User Stories

1. As the owner, I want a multi-turn chat, so that I can try the model conversationally.
2. As the owner, I want an editable system prompt, so that I can test prompt variants.
3. As the owner, I want to set `temperature`, `top_p` and `max_tokens`, so that I can probe sampling.
4. As the owner, I want a thinking toggle and the reasoning shown in a collapsible block separate from the answer, so that I can inspect reasoning without it cluttering the reply.
5. As the owner, I want to pick the lane tag (`Interactive` / `Agent`), so that I can see how each class is served.
6. As the owner, I want to stop a streaming reply, so that a runaway generation doesn't block me.
7. As the owner, I want each reply to show TTFT, decode tok/s, total duration, prompt and completion tokens, and finish reason, so that I get a quick read of the request.
8. As the owner, I want a table of this session's requests with those figures, so that I can compare runs side by side.
9. As the owner, I want API errors (400/404/413/503/504) shown with the OpenAI error message, so that a rejected request is readable.

## Implementation Decisions

- Model id from `GET /v1/models` (first entry); lane tag sent as the ignis `class` extension field (not the `@<lane>` model suffix).
- Thinking toggle maps to the request's `enable_thinking` field; reasoning read from `delta.reasoning_content`, answer from `delta.content`.
- SSE parsing is a pure module (`fetch` + `ReadableStream`, no `EventSource` — POST body needed); abort via `AbortController`.
- Figures, all measured with `performance.now()` in the browser:
  - TTFT = request sent → first chunk carrying `content` or `reasoning_content`.
  - Decode tok/s = `completion_tokens` / (last token chunk − first token chunk).
  - Duration = request sent → `[DONE]`.
  - Tokens and finish reason from the usage chunk / final chunk. A stopped request records its figures as partial (no usage).
  - These are HTTP-observed figures, labelled as such — never presented as engine-internal timings.
- State in memory only: no localStorage, no persisted conversations. Reload clears everything.
- No tool definitions sent; tool calls are not rendered.
- No new dependencies beyond React for this ticket.

## Testing Decisions

- vitest (run with `npm test` in `web/`, outside `cargo test`) for the pure logic: SSE chunk parsing across split reads, reasoning/content separation, usage extraction, figure computation from a synthetic timeline, error-body parsing.
- `npm run build` must succeed with no TS errors.
- Manual smoke against `npm run dev:mock` (01), then once against a live ignis when the GPU is free (check the card is unused first — `docs/agents/testing.md`).
- No Rust change expected; if one becomes necessary, it ships with a Rust test and `cargo test` passes workspace-wide.

## Out of Scope

- Server-side extra per-request fields (prefix reuse, queue wait, spec acceptance) — a separate ticket if ever needed, reviewed against ADR 0017.
- Persistence, tool calling, `/v1/responses`, multiple models.
- The Prometheus panel (03).
