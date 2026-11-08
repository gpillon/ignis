---
status: accepted
---

# The Playground's system prompt never changes: what changes rides after the history

Until 2026-10-28 the Playground's system prompt carried the titles of the
memory notes and the list of the files attached to the chat, and declared
`read_file` only while a file was attached, so saving a note or attaching a
file changed the prompt's first block and the next request prefilled the whole
conversation again. The owner decided on 2026-10-28 (#170) that the system
prompt and the tools block stay fixed, the same in every chat whatever the
memory and the workspace hold: the titles of the memory notes and the files of
the chat's workspace reach the model in a turn notice, an instruction message
after the history, whole at the chat's first turn and later only for what
someone else changed since, kept with its turn once sent so that every later
request renders the same bytes. Saying in the system prompt only that a memory
exists was rejected: a model that sees no title does not know when a note is
worth reading, and the memory would stop being recalled.

## Consequences

- The turn notice is a `developer` message, the channel the live date already
  takes; under a developer message policy that joins or gathers, prefix reuse
  drops for it as it does for the live date.
- A change someone else makes during a tool loop reaches the model at the next
  user turn; until the model reads that file again, its write over it is
  refused.
