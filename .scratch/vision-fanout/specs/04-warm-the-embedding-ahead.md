# 04 - warm an embedding before anyone asks a question

GitHub #243 gave the leaf a pool that can hold several encoded images at
once, and gave the runtime a cache that keeps them past the request that
encoded them. What it did **not** give anything is a way to put a picture in
that pool before a request needs it.

An encode happens in exactly one place: inside a prefill chunk that covers a
media item (`RuntimeCompute::prefill_multimodal_job`). So the first question
about a new frame still pays the whole tower - 4.17 s on a 4096x4096
screenshot, `docs/findings/2026-09-20-number-width-and-decide-e2e.md` - and
every question after it is free. For a fan-out that is the right shape. For
the owner's stated target, it is not:

> per fare "realtime" avro bisogno che piu immagini (es 4-8) restino sempre
> "pronte"; [...] al tempo N siano disponibili N+x immagini

"Ready at time N" means the encode happened before time N. Nothing in #243
can make that true.

## What is missing

Three things, and the first is the only one that is obviously a ticket.

1. **A trigger.** Some call that hands the engine an image and returns when
   its embedding is resident, without asking a question, generating a token
   or reserving KV pages. Today every path into the encoder comes through a
   scheduled request with a prompt.

2. **A pin.** The cache evicts the least recently unheld entry when the pool
   is full (ADR 0035). A frame deliberately warmed is, by definition, unheld
   until someone asks about it - so it is the *first* thing the next warm
   evicts. A warmed set of 4-8 frames needs to be able to say "these stay".
   Whether a pin is a flag on the warm call, a separate tier, or a different
   eviction order is open.

3. **Who pays the 4.17 s.** A warm is tower work on the same stream every
   prefill chunk uses, between chunks (GitHub #212). Warming a frame
   therefore inserts multiple seconds of latency into every other lane's
   token stream, which is exactly the problem chunked prefill exists to
   avoid (ADR 0018). A warm that is not chunked or not yielded is a
   regression for everyone else on the card, and this is the part that most
   needs a measurement before a design.

## Not in scope here

Making a *new* frame cheaper. That is the tower's own cost at width, spec 02
(`02-the-towers-cost-at-width.md`). Warming moves when the 4.17 s is paid; it
does not shrink it. If spec 02 lands first, this ticket gets cheaper but does
not go away.

## Open questions for the grill

- Is the trigger an endpoint of its own, or a shape of `/v1/decide` (a
  request with an image and no question)? The second reuses the admission
  path and the media acquirer; the first does not have to pretend to be a
  decision.
- Does a warm reserve, or does it race? Two warms and one pool that fits one
  of them is a refusal the caller has to understand.
- Does a warmed embedding have a lifetime, or only a pin? A pinned frame
  nobody ever asks about holds its pages until the process ends.
- What does the Monitor show? `cached_media` / `cached_media_columns` exist
  (GitHub #243); a warmed-and-pinned tier is a different fact from a
  cache entry, in the sense GitHub #216 means by `kind`.

## Acceptance (draft - grill before building)

1. A warm call over an image returns with the embedding resident, having
   generated nothing and reserved no KV pages.
2. A question about a warmed image reports `media.encode_seconds` 0 and a
   TTFT that carries no tower time.
3. A pinned set of N frames survives a warm of an N+1th: the refusal names
   the pool, and no pinned frame is evicted.
4. Warming does not stall a concurrent decode lane beyond one prefill
   chunk's worth of latency - measured, against ADR 0018's bound.
5. The plan and the Monitor distinguish pinned bytes from cache bytes.

## References

- ADR 0035 (the pool and the cache this builds on), ADR 0030 (the VRAM
  plan), ADR 0018 (chunked prefill, the latency bound point 3 must respect),
  GitHub #212 (the encoder shares the prefill scratch arena).
- GitHub #243 (the fan-out slice), spec
  `01-share-the-vision-embedding.md`, spec `02-the-towers-cost-at-width.md`.
- `crates/runtime/src/lib.rs::prefill_multimodal_job` (the only path into the
  encoder today).
