---
status: accepted
---

# The server downloads and removes models after start, from the Server view

Until 2026-10-25 the server fetched a model only at start, when the model it
was told to serve was missing (ADR 0033, ADR 0047), and the Server view could
only copy the `model download` command for the operator to run. The Server
view now downloads a catalog entry, cancels that download and removes a
downloaded model, through routes that take the same authentication as
`PATCH /v1/config`, which can already reload the model and rewrite the config
file; a switch still never downloads. A separate admin key was rejected
because the server has one key and no scopes: an operator who wants none of
this sets `download.from_ui` to false, and `download.enabled` false still
keeps the server off the network altogether.

## Consequences

- One download runs at a time, and the `model download` command takes the
  same per-entry lock, so the two never write one `.part`; every download
  first checks that the free space covers the bytes still missing plus
  `download.free_margin`.
- The page never asks for or shows a token: a gated repo is fetched with the
  token the server already has from its environment, or fails naming it.
- Removing a model refuses the one loaded, loading or being downloaded; the
  start model may go, and the next start fetches it again or refuses.
