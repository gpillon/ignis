---
status: accepted
---

# Cross-request state reuse: content-matched retained state across three residency tiers

A finished request leaves its state behind, and a later request resumes from the longest retained state whose
content is a prefix of its prompt: without it every turn of a chat and every iteration of an agent loop sends its
whole history again and pays the whole prefill again, and so does a subagent that arrives after its sibling
finished. Reuse happens only at recorded points, never at an arbitrary longest common prefix, because GDN state
exists only at the positions where it was captured. It is on by default; `--reuse-prompt off` is for cold benches
and correctness oracles.

- **Match by content, not by session.** A request reuses the longest retained state whose match key, a hash of its
  token content and its media identity, is a prefix of its prompt. No API field selects it: the clients send no
  session identifier, and a wrong session id could hand one conversation another's state, where a content match can
  only hand over identical history. Reuse is global, because there is one API key: anyone who can reach the server
  can tell from the time to first token whether a prefix was seen before, and that side channel is accepted. The day
  a second key exists, the key's identity joins the match key.
- **A prompt checkpoint serves the next turn.** It is the whole-sequence state of every request at its generation
  opener, `<|im_start|>assistant\n`: the last position every later turn provably shares. The next turn does not
  render an assistant message the way it was generated (thinking is dropped or emptied, a `\n` follows
  `<|im_end|>`, tool arguments are serialized again), so the state after generation never matches. A conversation
  keeps at most two, its latest and its turn-opening one, taken right after its last real user message: the only
  one a new user message can still match once history drops the earlier thinking.
- **A retained prefix serves the next request that opens the same way.** It is a shared prefix kept alive with no
  claimant, published at each of a request's reuse boundaries, floored to whole KV pages, one retained slot each.
  The first is the end of the system-and-tools block, which the next subagent of a burst shares. A `/v1/decide`
  state given as content parts stays in the user turn and never reaches that point, so a static head before a
  changing image and a fan-out's whole state were prefilled every time (324 ms per request for a text state the JSON
  path served in 70 ms); `/v1/decide` therefore adds a fan-out's computed common prefix, an observed fork and a
  reuse marker, `cache_control: {"type": "ephemeral"}` on a content part, and a request carrying a marker gets no
  observed fork. A marker or a fork decides where state is kept, never what a request may claim: the marker is an
  API field for placement only, and the prompt the model reads is unchanged. A fan-out's head is the one departure
  from "free until the room is needed": it is given up when its fan-out ends, and never goes to KV-RAM, because no
  later request is expected to send that state again and the retained slots are too few to keep a head per
  decision.
- **A prefix may be published over another**, a chained prefix, one reference per link and every page still
  charged once. Without it a conversation grown past its first KV page could never take a second checkpoint, and
  "at most two" would bound something that never grew.
- **Reuse copies, never consumes**, in every tier: the entry stays after a claim, as a blob stays in KV-RAM and a
  retained blob's file on the disk after a restore. Retries, regenerates and forks from the same point all hit.
- **Retained state is free until the room is needed.** On the device it is always the first victim, and it never
  delays or refuses an admission; nor does what a request still waiting for its admission claimed of it, which goes
  as retained state goes for an entry ranked above the claimant (ADR 0045). Its mutable images live in retained
  slots reserved at load (ADR 0030), and when none can be had a checkpoint is simply not taken.
- **Prompt checkpoints are given up before retained prefixes**, least recently used within each kind, for three
  reasons in the order they decide it. A checkpoint is the narrower bet: it serves one conversation's next turn,
  where a retained prefix serves every future request that opens with that block, of a conversation seen or not.
  Its prefix reaches past the block, so discarding one frees at least as many pages: fewer discards, less reuse
  lost. And when the block and the opener fall in the same KV page, one prefix carries both and its pages come back
  only when every holder lets go, so any other order gives up the wide bet and still returns no page. The order is
  a policy the owner may reverse; the third reason rests on both kinds being in the one set the reclaim reads
  (`ConcreteScheduler::reclaimable_prefixes`), which must go on listing both whichever goes first.
- **Three residency tiers: the device, KV-RAM and KV-disk** (ADR 0045, which built the third). A tier receives
  retained state lazily, only when the tier above would otherwise discard it, and gives it up before any evicted
  live sequence. What KV-RAM gives up of a checkpoint goes to the disk. A retained prefix KV-RAM gives up, and a
  device checkpoint KV-RAM cannot take, are discarded: nothing brings a prefix back from the disk, and retained
  state never waits for a file to commit. A spill displaces only what ranks below it, and with the disk on an evicted live sequence is never discarded to make
  room. A retained prefix spills to KV-RAM too, ranked with the checkpoints there, and comes back to the device
  once, for its whole burst, taking pages from retained state only: it never evicts live work.
- **A restore must beat the best reuse still above it by a restore floor**, a sibling's or a retained prefix
  included: a KV-RAM restore that beats a device checkpoint but not a device prefix by a chunk pays the crossing
  for nothing. KV-RAM's floor (`KV_RAM_RESTORE_FLOOR_TOKENS`) is a fixed starting value, to be tuned by measurement,
  not derived. The disk's is the model's (`kv_disk::restore_floor_tokens`), measured against the best match on the
  device and in KV-RAM, and a blob comes back from the disk to the device directly, never through KV-RAM. A claim
  reports its reuse and counts its hit only once it lands (ADR 0045). KV-disk reuses nothing across a restart yet,
  and the tier is wiped: the identity its file headers carry is what would let it outlive one without a format
  change.
- **The generation opener's page rides the capture.** A reusable prompt's prefill is cut once, at the opener: the
  capture lends the whole pages below it to a pages-only link, a chained prefix with no image, and the checkpoint
  stands on it. Flash-Next and the 27B do so on every card, and with no card read. The alternative, an imaged
  prefix published at the opener's page floor, cuts the prefill a second time, and each cut is a traversal: on
  Flash-Next it costs the expert stream it touches, 0.13-0.66 s of a reused agent turn's 1.86 s; on an RTX 30 or
  RTX 40 card a traversal of the 27B's trellis body decodes every matrix again; and on the RTX 5090, where it was
  taken to cost ~19 ms, it measured 45-55 ms per request, the small traversal plus a synchronous 222 MiB copy of the
  mutable state into a host retained slot over the PCIe link. What the single cut gives up is that image, which
  only a prompt matching up to the floor and parting in the last page before its opener read: such a request
  resumes at the reuse point before it, the previous turn's checkpoint or the system block, and an identical retry,
  a conversation's next turn and a subagent burst resume as before. The closed thinking block keeps its own cut,
  and the reuse boundaries are still published with their images at their own cuts.
- **A pages-only link is never claimed**, has no handle, takes no retained slot and exists only with a checkpoint:
  no sequence stands at its end, so it has no state to hand over. The server decides the single cut in one
  function, from the family and the card's generation (`config::arch::opener_page_rides_capture`); the core reads
  no card. A contributor's measurement switch, `IGNIS_SECOND_TAIL_CUT=1`, gives the 27B on Blackwell, and with no
  card read, the second cut back for comparisons.
- **A request whose image spans its opener's page floor lends at its capture on any load**, a load under that
  switch included. No claimable prefix ends inside an image, so its floor walks back to before the picture, and
  without the loan nothing is published at the opener's page and no checkpoint is taken. That is the common shape,
  not a rare one, since a short question after a picture puts the opener in the KV page the picture ends in, and
  it cost a whole-picture prefill on every such repeat and next turn
  ([the finding](../findings/2026-10-31-a-checkpoint-forms-when-an-image-spans-the-openers-page-floor.md)). The
  link may end inside the image; nothing claimable does: the checkpoint over it ends at the opener, past the
  picture, and its key carries the image's identity.
- **Prefix identity is the match key everywhere**, for multimodal prompts too: sibling shared prefixes, retained
  prefixes, their KV-RAM copies and prompt checkpoints are all matched by token ids and media items, since a prefix
  cache matching raw token ids would let a sibling sending another picture share one, and vision adds no key of its
  own. No publish point ends inside a media item's placeholders. A multimodal claim always leaves one prompt token
  to prefill, because a claimant is built from cloned pages and mutable state, neither of which carries the
  sequence's `rope_delta`, and only a prefill span hands it to the leaf; so neither a shared prefix nor a prompt
  checkpoint reaching a multimodal prompt's last token is offered: in that rare case it resumes from the longest
  recorded point before the prompt's end. A prompt that ends exactly at a held checkpoint's opener may therefore not claim it and prefills again; the
  served template ends neither thinking setting that way. Lifting the limit means carrying `rope_delta` with the
  reused state (GitHub #201, open).
- **A blob's compatibility identity** is the artifact content hash, the KV format, the blob layout version and the
  drafter's presence and draft window: never a `RequestId`, and never an operator knob that does not change state.
  A KV-disk file's header carries it, widened by the served model id, the sidecar's payload digest when there is
  one and the RoPE scaling (ADR 0045). The content hash digests what the container declares, its directory, not
  its payload: the v2 container carries no per-tensor digest by design, and hashing ~19 GB of weights at every
  start is not a price a compatibility check may charge. It catches a different model, a re-export, a re-layout, a
  renamed or added object and a format change. It does not catch an in-place re-quantization that rewrites payload
  bytes under the same names, formats, shapes and offsets: a different model whose blobs this identity would
  accept, the one hole, held as a test (`a_rewritten_payload_alone_is_the_proxys_known_limit`). Closing it needs a
  digest the producer writes into the container, a change to the artifact format and not to reuse; whether the
  container should carry one is open, for the owner.

## Considered Options

- **A session id in the API**: rejected for the reason above, and because no client sends one.
- **A checkpoint after generation**, ninfer's frontier: it never matches under ignis's rendering.
- **A checkpoint at the prompt's end, after `<think>\n`**: it misses on every new user message once history drops
  the earlier thinking. The opener costs two or three prefilled tokens instead.
- **Consume on reuse**, ninfer's exclusive claim: a retry, a regenerate and forks would all miss.
- **An eager device-to-host copy at request end**: roughly 15-20 ms per request at PCIe Gen 3, whether the state
  is ever reused or not.
- **Per-key isolation**: there is no second key to isolate.
- **A learned boundary, the observed longest common prefix, for every request**: to revisit only if measurement
  shows shared heads extending past the system block. `/v1/decide` is where it did, and the observed fork is that.
- **For a decision's state, moving its leading text into the system block**: it changes measured decision prompts
  (ADR 0034), can never hold an image, and misses whenever changing text precedes the first image. **A "first text
  part" convention**: published nowhere, and it spends a slot whenever the first part is the one that changes.
  **Client markers with same-step producer and consumer scheduling**: the scheduling half would make a decision a
  live publisher, which GitHub #238 rules out.
- **A checkpoint taken mid-chunk**: the state sections agree with one another only at a chunk boundary, so every
  layer would have to export its state at an interior position.
- **Publishing the opener's page floor later, from a snapshot taken at the cut**: the state at the floor is gone
  once the chunk passes it; the pages are all that can be published later, which is the pages-only link.
- **Skipping the publish when the tail is short**: the checkpoint needs a holder for the pages below the opener,
  and without one the next turn loses its reuse.
- **A checkpoint that owns those pages itself**: the same ownership in a second structure, when the prefix chain
  already walks, materializes and releases links.
- **Keeping the 27B's second cut and its imaged prefix at the floor**, as it first was, and on Blackwell until the
  cut was measured there: changing its splits would move its outputs at near-ties without a measured reason to.
  The costs above are that reason.

## Consequences

- **Exactness is against a cold prefill split at the same boundaries.** A reused request and a cold prefill of its
  whole prompt split their chunks differently, so they can diverge where two tokens are nearly tied. The
  correctness claim is bit-exactness against a cold prefill split at the same boundaries, `[…, opener)` and
  `[opener, end)` at the tail; a divergence against an unsplit cold prefill is recorded as information, not as a
  failure. Moving a cut therefore moves greedy text at near-ties, as the single cut did the 27B's on Blackwell:
  byte-identical greedy text is not what correctness is held to (ADR 0007).
- **Rendering must be stable.** The chat template has to render history the way the reference does (tool-argument
  order, `preserve_thinking`): a rendering drift silently turns every match into a miss.
