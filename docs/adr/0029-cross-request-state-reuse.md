# ADR 0029 — cross-request state reuse: content-matched retained state across three residency tiers

## Status

Accepted (2026-09-16) — GitHub #183 (absorbs #168). Spec:
`.scratch/kv-reuse/specs/01-cross-request-reuse.md`. Builds on ADR 0024
(sequence state transfer) and ADR 0023 (eviction priority), and amends both.

## Context

Prefix reuse (#126) and the KV-RAM tier (#125) both stop at request
liveness. A shared prefix is dropped when its last claimant completes, and
KV-RAM holds only live sequences evicted for overflow. So every turn of a
chat and every iteration of an agent loop re-sends its whole history and pays
the whole prefill again, and so does every subagent of a burst that arrives
after its sibling finished. ninfer, the owner's fork of the reference, keeps
a finished request's state on its idle lane and spills it to pinned RAM
(`--kv-ram-capacity`). ignis has no equivalent.

Three facts shape what can be reused:

- GDN state exists only at positions where it was captured. Reuse therefore
  happens at recorded points, never at an arbitrary longest common prefix.
- The next turn does not re-render an assistant message the way it was
  generated: thinking is dropped or emptied, a `\n` follows `<|im_end|>`, and
  tool arguments are re-serialized. The state after generation never matches.
  The last position every later turn provably shares is the **generation
  opener**, `<|im_start|>assistant\n`.
- The clients (qwen-code, the Playground) send no session identifier.

## Decision

**Match by content, not by session.** A request reuses the longest retained
state whose token content, media identity included, is a prefix of its
prompt. There is no API field for it. A wrong session id could hand one
conversation another's state; a content match can only hand over identical
history.

**Two retained objects.** A **prompt checkpoint** is the whole-sequence state
of every request at its generation opener. A **retained prefix** is a shared
prefix kept alive with no claimant, published at a structural boundary: the
end of the system-and-tools block, floored to whole KV pages. The first
serves the next turn; the second serves the next subagent of a burst.

**Reuse copies, never consumes.** Claiming a prompt checkpoint shares its full
pages, clones its mutable image and copies its partial tail page, and the
entry stays. Retries, regenerates and forks from the same point all hit. A
conversation keeps at most two checkpoints: its latest, and its
turn-opening checkpoint (the one right after its last real user message).
That turn-opening checkpoint is the only one a new user message can still
match once history drops the earlier thinking.

**Retained state is free until the room is needed.** On the device it is
always the first victim, and it never delays or refuses an admission. Its
mutable images live in a byte-budgeted device pool; when that pool is
exhausted, a checkpoint is simply not taken. KV-RAM receives it lazily, only
when the device would otherwise discard it, and discards it before any
evicted live sequence.

**Three residency tiers, Tier 2 prepared.** Tier 0 is the device, Tier 1
KV-RAM, Tier 2 KV-disk. Tier 2 is not built. What is built now is the part it
would otherwise force a rewrite of:

- a blob's **compatibility identity** is the artifact content hash, the KV
  format, the blob layout version, and the drafter's presence and draft
  window. It is never a `RequestId`, and never an operator knob that does not
  change state;
- an entry's **key** is a hash of its token content plus its media identity.

**Reuse is global.** Today there is one API key. Anyone who can reach the
server can tell from TTFT whether a prefix was seen before; that side channel
is accepted. The day a second key exists, the key's identity joins the match
key.

**On by default.** `--prompt-reuse off` exists for cold benches and
correctness oracles.

## Considered options

- **A session id in the API** (#168 option A). Rejected for the reason above,
  and because no client sends one.
- **A checkpoint after generation** (ninfer's frontier). Never matches under
  ignis's rendering.
- **A checkpoint at the prompt end, after `<think>\n`.** Misses on every new
  user message once history drops the earlier thinking. The opener costs two
  or three re-prefilled tokens instead.
- **Consume on reuse** (ninfer's exclusive claim). Retry, regenerate and forks
  would all miss.
- **Eager device-to-host copy at request end.** Costs roughly 15–20 ms per
  request at PCIe Gen 3, whether the state is ever reused or not.
- **A learned (observed-LCP) retained-prefix boundary.** To revisit only if
  measurement shows shared heads extending past the system block.
- **Per-key isolation.** There is no second key to isolate.

## Consequences

- **Equivalence.** A reused request and a cold prefill of its whole prompt
  split their chunks differently, so they can diverge at near-ties (the same
  effect as #153). The correctness claim is bit-exactness against a cold
  prefill split at the same boundary. Divergence against an unsplit cold
  prefill is recorded as information, not failure.
- **Amends ADR 0024.** A prompt checkpoint's publish point is not a page
  boundary, and a snapshot of a sequence holding a shared prefix materializes
  the shared pages instead of being refused. A shared prefix may also be
  published *over* another one (#187, a **chained prefix**): a request that
  resumed from retained state owns only the pages it warmed past it, so the
  entry it publishes owns those and holds the chain below — taking over the
  reference the sequence was holding rather than adding one. One reference per
  link, every page still charged once, and a claimant of the chain shares all
  of it. Without it a conversation grown past its first KV page could never
  take a second checkpoint, and "at most two" above would bound something that
  never grew.
- **Amends ADR 0023.** Retained state goes first, on the device and in
  KV-RAM.
- **Departure (#189): the "artifact content hash" above is a hash of the
  container's *directory*, not of its payload.** The v2 container carries no
  per-tensor digest by design (`ignis_artifact::checksum`'s module doc), and
  hashing ~19 GB of weights at every startup is not a price a compatibility
  check may charge. `Reader::content_hash` therefore digests everything the
  container *declares*: its identity, its byte size, its payload start, and
  every object's name, kind, numeric format, storage layout, shape, offset and
  length. That catches a different model, a re-export, a re-layout, a renamed
  or added object, and a format change. It does **not** catch an *in-place
  re-quantization* that rewrote payload bytes while preserving every name,
  format, shape and offset — a different model this identity would accept
  blobs from, and the one hole left in user story 16. Recorded as a test
  rather than a comment (`a_rewritten_payload_alone_is_the_proxys_known_limit`
  in `crates/artifact/src/lib.rs`). Closing it needs a digest the *producer*
  writes into the container: a change to the artifact format, not to this
  feature. **Open for the owner** — whether the proxy is enough, or whether
  the v2 container should start carrying a payload digest.
- **Vision (#180)** builds its media-aware prefix identity on the match key
  defined here, instead of adding its own.
- **Rendering must be stable.** The chat template has to render history the
  way the reference does (tool-argument order, `preserve_thinking`); a
  rendering drift silently turns every match into a miss.

## Amendment (2026-09-16) — which retained object goes first (#188)

The Decision says retained state is "always the first victim" on the device
but not which of the two kinds goes first. #188 makes both kinds real at the
same time, so the device's first-victim path needs the order.

**Retained prompt checkpoints are given up before retained prefixes; least
recently used within each kind.**

Three reasons, in the order they decide it:

- **Width of the bet.** A checkpoint serves one conversation's next turn. A
  retained prefix serves every future request that opens with that system and
  tools block, including ones belonging to no conversation seen so far. Between
  two bets, the narrower one is given up first.
- **Pages returned per discard.** A checkpoint's prefix reaches past the block
  its conversation opened with, so discarding one frees at least as many pages
  as discarding a prefix does. Fewer discards, less reuse lost.
- **Termination where a prefix carries both.** When the block and the opener
  fall in the same KV page, one prefix is both the burst's retained block and
  the pages a checkpoint stands on. Its pages come back only when *every*
  holder lets go, so the checkpoints on it must go before its retention — any
  other order gives up the wide bet and still returns no page.

This is a policy the accepted Decision did not fix, recorded here rather than
left in a comment. The owner may reverse it, and reversing it means two places
rather than one:

- the order of the two arms in `ConcreteScheduler::reclaim_retained`, which is
  the ordering itself;
- `ConcreteScheduler::reclaimable_prefixes`, which must go on listing **both**
  kinds — checkpoint-held prefixes and bare retained ones — in one set. The
  third reason above rests on that *membership*, not on the order the union
  happens to be built in: a prefix carrying both is reclaimable only because
  both kinds appear in the set, and narrowing it would strand that prefix
  whichever arm ran first.

`crates/core/tests/retained_prefix.rs`'s
`the_narrower_bet_is_given_up_first_when_a_pool_holds_both_kinds` is the test
that changes with it; it is written to fail when the order is flipped.

## Amendment (2026-09-16) — the KV-RAM tier (#190)

What building Tier 1 decided that the Decision left open.

- **The restore floor is `KV_RAM_RESTORE_FLOOR_TOKENS` = 1024**, one prefill
  chunk: a fixed starting value, to be tuned by measurement (the trace replay,
  #191), not derived. It is measured against the **best reuse still on the
  device**, a sibling or retained *prefix* included — a KV-RAM restore that
  beats a device checkpoint but not a device prefix by a chunk pays the
  crossing for nothing.
- **Retained prefixes spill too, and come back to the device** (owner
  decision). A retained prefix the device gives up is written to KV-RAM, ranked
  with the checkpoints there. A prompt whose longest reuse is such a prefix —
  beating every device reuse by the restore floor and every KV-RAM checkpoint
  outright — brings it back **once**: the blob is restored into a carrier
  sequence, published again under its original name, and retained on the
  device, where that request and every later member of its burst claim it in
  place. The blob stays in KV-RAM, so giving the prefix up again copies
  nothing. Its pages are taken back from retained state like a request's, and
  it never evicts live work.
- **A published head is named by its publisher and its length.** One request
  publishes its system block and then a chained head over it (#187 x #188);
  keyed by the publisher alone, the second replaced the first, and a burst
  member claiming the block was stood up on the chain.
- **A request restored from a materialized blob owns every page it restored.**
  No publish point at or below the restored length can be reached again, so
  it publishes, if at all, at its own generation opener's page — the one that
  makes its next checkpoint capturable — and only when it has an opener, as a
  claimant's chained publish already must. Without this a conversation that
  came back from KV-RAM would never leave another checkpoint.
- **A blob a request has chosen is held until its restore lands.** Nothing
  discards it in between; a discard that arrives meanwhile takes effect when
  the claim lets go.
- **Lifecycle facts.** Hit (chosen), miss (first chunk landed with no match in
  a tier), spill, discard and restore (landed) are emitted by the core, one
  per event, per tier, identically with metrics on or off (ADR 0017 as
  amended by #190).
