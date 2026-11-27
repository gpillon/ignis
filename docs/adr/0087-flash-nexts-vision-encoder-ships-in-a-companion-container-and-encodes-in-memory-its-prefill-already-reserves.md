---
status: accepted
---

# Flash-Next's vision encoder ships in a companion container and encodes in memory its prefill already reserves

Qwen3.8-Flash-Next is multimodal upstream, but its artifact was converted text-only, so ignis refused an image to it.
Its tower has the 27B's shape but its own weights: the same origin (the block norms agree within 0.3-0.7%) retrained
with the model (an attention projection 16% apart, the merger's norm correlating at 0.84), and a merger that writes
Flash-Next's hidden width, half the 27B's. Flash-Next takes images with the 27B's feature set, and this is how:

- **The tower and its merger ship in a companion container of their own**, beside the main one, as the MTP head
  does: their own identity, a pair pin to the main container, bound only when vision is on
  ([the layout](../formats/flash-next-layout.md), "The vision companion container"). The main container and its
  content hash stay as they are; the download grows by about 0.5 GB, which the catalog always fetches, as it fetches
  the MTP companion. A later packaging of ignis's own may fold the companions back into one file.
- **Every matrix of the tower is W8**, encoded from the pinned revision's BF16 tensors, not the 27B tower's grouped
  recipe: under that recipe the tower check raised its alarm, the error growing at Flash-Next's retrained merger,
  and W8 answers it for about 0.2 GB more of tower, the same staging slot and no new kernel route: every matrix
  takes a W8 route the dispatch already serves, the merger's second linear through the recorded patch that gave its
  width one (ADR 0088). The 27B's tower and recipe do not change. A companion of the grouped encoding is refused when
  it opens, naming both weights ids, so a start or a switch refuses it before anything loads.
- **The encoder works in memory the prefill already reserves**: Flash-Next's prefill scratch and its expert staging
  ring, both idle while an item encodes, since the encode runs before the chunk's forward. A vision load plans the
  same expert cache as a text-only load at the same settings: the forward's vision inputs ride in memory a text load
  reserves too, and a vision prefill refuses while the decode graphs' scratch is in use, a verify round while the
  prefill scratch is. Where the two areas cannot hold an item of the item bound, the bound shrinks to what fits and
  the VRAM plan says so: vision never takes the expert cache. The tower and the embedding pool follow ADR 0075's
  residency, `host` on the RTX 5090's profile, so a vision load with no image in flight costs no VRAM.
- **Vision stays a load option, off by default**: the vision and media fields apply to both text families, each
  family its own, and the built-in profiles turn vision on for the 27B only. A start or a switch that asks for it
  without the companion is refused naming the file, before anything is torn down.
- **A switch carries the vision of the model it replaces**, to the target's own vision settings, unless vision is
  named off for the target itself, which wins; naming it off for every model does not stop the carry. A model
  serving without vision carries nothing, and a start is unchanged. The carried value is decided again at every
  switch and never written to the config file.
- **Its quality is checked without the 335 GiB BF16 checkpoint.** ADR 0082's gate scores text, and the text path does
  not change. The image path adds three checks: the encoded tower against the BF16 tower, run from the checkpoint's
  first shard alone, with the 27B tower's error under the same recipe as its bar; the three-axis positions in the
  attention and the indexer, exact against the reference implementation on synthetic tensors; and accuracy on images
  with known answers, reported beside the 27B's. A KLD against BF16 on image prompts runs only if those raise an
  alarm ([the results](../findings/2026-11-10-flash-next-images-checked-without-the-bf16-checkpoint.md)).

## Considered Options

- **A new revision of the main container with the visual tensors**: 72 GB uploaded and downloaded again for 0.5 GB
  of tower, a new content hash, and the MTP companion's pair pin cut again.
- **The 27B's encoder rule, its own share sized for the item bound** (the larger of the prefill's share and the
  encoder's, ADR 0030): up to ~0.9 GiB beyond the prefill's, which on Flash-Next comes out of the expert cache. About
  1.1 GB of cache is worth 4% of one-lane decode and 9% of three-lane decode
  ([finding](../findings/2026-10-08-flash-next-kv-pool-follows-residency.md)), paid while no image is in flight.
- **Borrowing free KV pages** (ADR 0067's loan; the option ADR 0075 parked on #409): it would free the pool too, but
  needs the loan, an admission for an image that arrives with the pool full, and encoder kernels over scattered
  pages. It stays on #409, for both families.
- **The 27B's tower objects reused as they are**: the weights differ, and the merger's width with them.

## Consequences

- On a card that holds every expert (ADR 0054) there is no staging ring, and the prefill scratch alone bounds the
  item.
- A Flash-Next sequence keeps the three-axis position its indexer still needs, its incomplete compressed block's
  first token's, not every token's, in its snapshots, checkpoints and spills too, with no change of the blob layout:
  the indexer rotates a compressed block's key at its first token's position when the block completes.
- With the main container unchanged, a text-only user downloads 0.5 GB more and loads exactly what loads today.
- The MTP head's prefill entry before an image that starts a new span takes the placeholder's id, the image not
  being encoded yet: that costs acceptance, never the output. Greedy MTP on equals MTP off on image prompts up to
  ADR 0056's near tie, as the text path's wide verify rounds are held, and a verify round after an image leaves the
  state a plain round leaves, bit for bit.
  The routes drift about twice as far on an image prompt as on text, a property of the vision start and end tokens,
  not a defect of the image's path; routes that agree bit for bit are #567, parked.
