# Playground: work with artifacts

## Why

The model already produces standalone outputs — `create_file` files, HTML
pages with a sandboxed preview, code blocks — but each lives inside the
reply that made it. There is no place to see them together, keep them
across turns, iterate on one, or pick one up in a later session.

## Idea

- An artifact is a named, versioned output (HTML, Markdown, code, CSV,
  SVG…) owned by a session. `create_file` becomes (or feeds) an
  `artifact` tool with create / update / read, so the model can revise an
  artifact instead of re-sending it whole.
- A side panel lists the session's artifacts; opening one shows the
  preview (reusing `web/src/ui/HtmlPreview.tsx`: sandbox, full screen,
  Save as PDF) or the source, with version history and a diff between
  versions.
- The user can edit an artifact by hand and the model sees the edited
  version on its next read.
- Download, copy, and optionally persist artifacts in the browser
  (IndexedDB) so they survive a reload.

## Open questions

- Replace `create_file` or keep both?
- Update by full replacement or by patches (search/replace edits)?
- Per session, or a shared library across sessions?
- How much of the artifact goes into the prompt (titles only, like
  memory, or the latest version of the active one)?

## Acceptance (draft)

- The model creates an HTML artifact, then updates it in a later turn;
  the panel shows both versions and the preview follows the latest.
- A hand edit is what the model reads next.
- Unit tests for the artifact store (create, update, versions, read).
