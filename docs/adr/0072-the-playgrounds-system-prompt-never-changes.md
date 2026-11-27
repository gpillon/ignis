---
status: accepted
---

# The Playground's system prompt never changes: what changes rides after the history

The Playground's system prompt and its tools block stay fixed, the same in every chat whatever the memory and the
workspace hold: a prompt that carried the titles of the memory notes and the chat's files, and declared `read_file`
only while a file was attached, changed its first block when a note was saved or a file attached, and the next
request prefilled the whole conversation again. The titles of the memory notes and the files of the chat's workspace
reach the model in a turn notice, an instruction message after the history, whole at the chat's first turn and later
only for what someone else changed since, kept with its turn once sent so that every later request renders the same
bytes.

## Considered Options

- **Saying in the system prompt only that a memory exists**: a model that sees no title does not know when a note is
  worth reading, and the memory would stop being recalled.

## Consequences

- The turn notice is a `developer` message, the channel the live date already takes; under a developer message
  policy that joins or gathers, prefix reuse drops for it as it does for the live date.
- A change someone else makes during a tool loop reaches the model at the next user turn; until the model reads that
  file again, its write over it is refused.
