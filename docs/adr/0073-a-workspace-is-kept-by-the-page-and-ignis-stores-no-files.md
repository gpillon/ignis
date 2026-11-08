---
status: accepted
---

# A workspace is kept by the page and ignis stores no files

The owner decided on 2026-10-28 (#170) that the Playground's workspaces, the
text, pictures and audio the user and the model work on together, live in the
browser beside the chat sessions and the image store, and that ignis gains no
file store: no `/v1/files`, no `input_file`, no upload route. A store on the
server would show the files to other clients, but ignis would keep a user's
data, which it does nowhere else, and the operator would have one more surface
to secure, size and clean, for a feature of the Playground, which is not an
operator surface. A workspace leaves its browser as a zip the user exports and
imports.

## Consequences

- The model reaches a workspace only through the page's tools, which run in
  the browser: a picture it reads travels in the tool's result, an audio file
  as its transcript from `/v1/audio/transcriptions`.
- A workspace is lost with the browser's site data; when the store refuses a
  write the page says so, as it does for the chats, and never works around it.
