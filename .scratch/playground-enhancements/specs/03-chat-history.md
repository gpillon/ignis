# Playground: keep the chat history

## Why

Sessions live in React state only: a reload, a closed tab or a new build
of the page loses every conversation, with its tool runs, figures and
attachments. Memory notes already survive (localStorage); the chats do
not.

## What has to be decided

- **Where it lives.**
  - Browser only: IndexedDB (localStorage is too small once replies,
    reasoning, tool results and attachments pile up). Private to the
    browser, works with the static `/ui/` page and `--expose` as is.
  - On ignis: a small session store behind `/v1` (or `/ui/api`), shared
    across browsers and devices, but new server surface, storage on the
    host, and it must sit behind the API key.
  - Both: browser first, optional sync later.
- **What is kept.** Messages with reasoning, tool calls and results,
  agent runs, web and local runs, questions, figures and log rows,
  settings per session? Attachments (possibly MBs of PDF text) — kept,
  capped, or dropped with a note?
- **Lifecycle.** Save while streaming or at the end of each turn; what a
  reply that was streaming at reload time becomes (stopped); a schema
  version and migrations for later changes; delete one session, delete
  all, quota handling when the browser refuses more space.
- **Portability.** Export and import a session (JSON, maybe Markdown) so
  a chat can move between browsers without a server.
- **Privacy.** With `--expose`, anyone with the key uses the same page:
  browser storage keeps each visitor's chats in their own browser;
  server storage would mix them unless sessions are keyed per user.

## Acceptance (draft)

- Reload the page mid-conversation: the sessions list, the active session
  and every finished turn come back as they were; a reply that was
  streaming comes back stopped with the text it had.
- Deleting a session removes it from storage.
- A stored schema from the previous version still loads.
- Unit tests for the store (save, load, delete, migrate) and for
  serialising a session with tool runs.
