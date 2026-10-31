# The running notes send the notes back to the text model and the recap does not

The owner decided on 2026-10-31 (#428) that a transcript's notes come in two
forms, and that the rule of spec speech/07's Q125, that a writing of the notes
starts from the transcript alone and is never sent the notes written before,
holds for one of them only. The **running notes**, kept during a conversation,
are one append-only thread with the text model per transcript: the notes
written before are in it, each line with its id, and the model changes them
through four tools (`add_line`, `replace_line`, `remove_line`, and
`rewrite_section` once a section is full), so an update reads only what was
said since the last one, a line still true stays word for word, and an update
costs a few seconds whatever the conversation's length. The **recap**, written
when a listening ends and on a press, keeps Q125: one writing over the whole
transcript, told the context, the context images' readings and only the user's
own running lines, never the model's, so a mistake the running notes made live
cannot carry into the notes meant for after the conversation. The replay in
[the finding](../findings/2026-10-31-running-notes-recap-and-full-rewrite.md)
measured the three ways on public recordings of three kinds.

## Considered Options

- **Every writing from scratch, as before** (Q125 for both forms): each update
  rewords lines that did not change, so the reader rereads the notes every
  minute, and an update's cost grows with the conversation.
- **The thread for both forms**: notes kept a minute at a time stay good for
  what is on the table and poor for what must be weighed against the whole
  conversation (decisions, action items, the summary), which one writing over
  the whole transcript gets right.

## Consequences

- The thread is not stored. A reload, a kind change, Rewrite from scratch and
  a prompt past the restart threshold start it again from a snapshot of the
  notes, each line with its id, the user's lines marked as theirs.
- The page, not the model, keeps the notes' guards: an id the model copies
  into a text is stripped, a duplicate or overlong line is refused, a call on
  the user's line is refused, and an update stops at a cap of rounds.
- A wrong running line can last until a later update fixes it or the user
  does; the recap never reads it.
