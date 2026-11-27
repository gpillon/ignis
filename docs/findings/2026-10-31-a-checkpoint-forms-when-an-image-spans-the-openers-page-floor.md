# An image request whose picture spans its opener's page floor left no checkpoint; lending the pages at the capture resumes it

- Kind: experiment
- Status: current
- Observed: 2026-10-31
- Last verified: 2026-10-31
- Scope: core scheduler / prompt checkpoint over a pages-only link, multimodal prompts, the 27B on the RTX 5090
- Topic: vision
- Related: https://github.com/gpillon/ignis/issues/249, https://github.com/gpillon/ignis/issues/444, https://github.com/gpillon/ignis/issues/201,
  [ADR 0029](../adr/0029-cross-request-state-reuse.md), `crates/server/tests/vision_checkpoint_link_gpu.rs`,
  `crates/server/tests/vision_mixed_load_gpu.rs`
- Superseded by: none
- Summary: 27B, RTX 5090: a 16,404-token screenshot request left no prompt checkpoint, so its repeat prefilled it all (~1.6 s); lending the pages below the opener at the capture resumes 16,400 tokens in ~0.17 s.

## Question

GitHub #249: an identical repeat of an image request, and the conversation's next turn, prefilled the whole picture
again. Why was no prompt checkpoint left, how common is that shape, and what does a checkpoint standing on a
pages-only link that ends inside the picture cost and give back?

## Evidence

These figures were recorded with the change, on 2026-10-31, in ADR 0029's text of that date
(`git show records-2026-11-27:docs/adr/0029-cross-request-state-reuse.md`, the amendment for #249), and moved here
when that ADR was brought to the short format. The runs' raw output was not kept.

- The request: a 4096x4096 screenshot and a six-token question, 16,404 tokens. The last KV page boundary before
  its generation opener is 16,384, inside the image.
- Before: no claimable prefix ends inside an image, so the opener's page floor walked back to the page holding the
  image's first placeholder, nothing was published at the opener's page, the capture found no prefix under the
  opener and no checkpoint was taken. An identical repeat prefilled all 16,404 tokens, and so did the next turn.
- After, the request lends the pages below its opener to a pages-only link at its capture and gains one cut, at the
  opener: 17 chunks become 18.
- On the RTX 5090, 27B, 1,024-token chunks, DFlash2 on: the repeat went from 17 chunks and ~1.6 s to one 4-token
  chunk that resumes 16,400 tokens from the device in ~0.17 s. The next turn resumes the same 16,400.
- With the checkpoint spilled, over HTTP, every claim came back from KV-RAM: 16,400 tokens restored in 31-47 ms,
  with the device claimants' answers.
- Thinking on: the repeat resumed 16,442 of its 16,444 tokens. The template ends a prompt with the empty think
  block, four tokens past the opener, with thinking off, and with `<think>\n`, two past it, with thinking on.
- A next turn whose own opener's page floor lies past the picture (in this request, any whose opener lies 48 tokens
  or more past the first) takes no link of its own: standing on the checkpoint, it publishes the pages it warmed as
  a prefix chained over the link and captures at its opener over that, and the turn after it stands on both.
- Exactness (`crates/server/tests/vision_checkpoint_link_gpu.rs`): the identical repeat generated what the first
  request did; a next turn standing on the link generated what a cold prefill split at the first opener did, from
  the device and, after a spill, back from KV-RAM; a next turn with its own prefix chained over the link, and a
  third turn standing on that chain, each generated what a cold prefill cut where its whole history was cut did,
  token for token.
- Against an unsplit cold prefill the new cut moves outputs at near-ties: the first answer read "shows" where the
  unsplit one read "displays".
- Under mixed load (`vision_mixed_load_gpu`), an Interactive request needing the whole pool moved the decoding
  Agent image request 0 times with this change alone, where it moved once before it: a lender's reservation shrinks
  by the pages it lent, and a live move counted only the mover's reservation as freed. With #444, which counts the
  pages a mover's chain frees once nothing but retained state holds them, the image request moves, and its output
  matched the same request served alone on a fresh load.

## Finding

Observed: the lost checkpoint is the common shape of an image request, not a rare one. It is taken by any image
request whose opener lands in the KV page its picture ends in, which a short question after the picture makes the
usual case. Observed: a checkpoint standing on a link that ends inside the image resumes the whole picture, on the
device and from KV-RAM, bit-exact against a cold prefill split at the same boundaries.

Inferred: nothing claimable ends inside the image. The link is never claimed, the checkpoint over it ends at the
opener, past the picture, and its key carries the image's identity; a claimant shares the link's pages, copies the
tail page, placeholders included, and its own prefill span still hands the leaf its rope delta.

## Implications

A request whose image spans its opener's page floor lends at its capture on any load (ADR 0029). The price of the
extra cut was taken as ~19 ms on the RTX 5090 at the time; against it stood a whole-picture prefill on every repeat
and next turn.

## Limits and unknowns

- One request shape, one card, measured once. The timings are those of the 27B as it was on 2026-10-31, when it
  still published an imaged prefix at the opener's page floor for other requests; they were not measured again
  after the 27B's opener page came to ride the capture on every card, nor after later prefill changes.
- A prompt that ends exactly at a held checkpoint's opener may not claim it, since a multimodal claim never reaches
  the prompt's last token, and it captures nothing itself. The served template renders neither thinking setting
  that way.

## Follow-ups

- GitHub #201: carrying the rope delta with reused state, which lifts the one-token tail.
