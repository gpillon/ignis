---
status: accepted
amends: [ADR-0070]
---

# An ADR states what holds today, and a decision that changes rewrites it in place

An ADR states what holds today, and why, in the `/domain-modeling` short
format. A decision that changes rewrites its ADR in place, the new text
replacing the old under the same number, so that whatever cites the ADR stays
valid; a new ADR is opened only for a new decision, and the earlier versions
are the history of the file in git. Both other ways were tried first: adding
each change to the ADR's text left sediment a reader dug through to find the
decision, and a new ADR for each change left chains, one decision spread over
several files.
