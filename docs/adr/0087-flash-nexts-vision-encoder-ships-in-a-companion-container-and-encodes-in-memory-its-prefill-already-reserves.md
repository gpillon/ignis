---
status: proposed
---

# Flash-Next's vision encoder ships in a companion container and encodes in memory its prefill already reserves

Qwen3.8-Flash-Next is multimodal upstream: at the pinned revision its config
says `language_model_only: false` and its shard 1 holds 333 `model.visual.*`
tensors. The artifact was converted text-only (spec `flash-next/04` left vision
out of scope), so ignis refuses an image to it. Its tower has the 27B's shape
but its own weights: the same origin (the block norms agree within 0.3-0.7%)
retrained with the model (an attention projection 16% apart, the merger's norm
correlating at 0.84), and a merger that writes 2,560 columns, Flash-Next's
hidden width, not 5,120. The owner decided on 2026-11-09 (#555) that Flash-Next
takes images with the 27B's feature set, and how:

- **The tower and its merger ship in a companion container of their own**,
  beside the main one, as the MTP head does (flash-next-layout.md §13): their
  own identity, a pair pin to the main container, bound only when vision is
  on. They are encoded from the pinned revision's BF16 tensors, every matrix
  in W8 (the 27B tower's recipe until #560's alarm, see As built). The 72 GB
  main container and its content hash stay as they are; the download grows by
  about 0.5 GB, which the catalog always fetches,
  as it fetches the MTP companion. A later packaging of ignis's own may fold
  the companions back into one file.
- **The encoder works in memory the prefill already reserves**: Flash-Next's
  prefill scratch and its expert staging ring (1,574,895,616 bytes on the RTX
  5090 at the 2026-11-09 start), both idle while an item encodes, since the
  encode runs before the chunk's forward. A vision load plans the same expert
  cache as a text-only load at the same settings. Where the two areas cannot
  hold an item of the item bound, the bound shrinks to what fits and the VRAM
  plan says so: vision never takes the expert cache. The tower and the
  embedding pool follow ADR 0075's residency, `host` on the RTX 5090's
  profile, so a vision load with no image in flight costs no VRAM.
- **Vision stays a load option, off by default.** A start or a switch that
  asks for it without the companion is refused naming the file, before
  anything is torn down.
- **Its quality is checked without the 335 GiB BF16 checkpoint.** ADR 0082's
  gate scores text, and the text path does not change. The image path adds
  three checks: the encoded tower against the BF16 tower, run from shard 1
  alone, with the 27B tower's error under the same recipe as its bar; the
  three-axis positions in the attention and the indexer, exact against the
  reference implementation on synthetic tensors; and accuracy on images
  with known answers, reported beside the 27B's. A KLD against BF16 on image
  prompts runs only if those raise an alarm.

## Considered Options

- **A new revision of the main container with the visual tensors**: 72 GB
  uploaded and downloaded again for 0.5 GB of tower, a new content hash, and
  the MTP companion's pair pin cut again.
- **The 27B's encoder rule, its own share sized for the item bound** (the
  larger of the prefill's share and the encoder's, ADR 0030): up to ~0.9 GiB
  beyond the prefill's, which on Flash-Next comes out of the expert cache.
  About 1.1 GB of cache is worth 4% of one-lane decode and 9% of three-lane
  decode (finding 2026-10-08, the KV pool following residency), paid while no
  image is in flight.
- **Borrowing free KV pages** (ADR 0067's loan, #356; the option ADR 0075
  parked on #409): it would free the pool too, but needs the loan, an
  admission for an image that arrives with the pool full, and encoder kernels
  over scattered pages. It stays on #409, for both families.
- **The 27B's tower objects reused as they are**: the weights differ, and the
  merger's width with them.

## Consequences

- On a card that holds every expert (ADR 0054) there is no staging ring, and
  the prefill scratch alone bounds the item.
- A Flash-Next sequence keeps the three-axis position of every token it holds,
  in its snapshots and checkpoints too, since the indexer rotates a compressed
  block's key at its first token's position.
- With the main container unchanged, a text-only user downloads 0.5 GB more
  and loads exactly what loads today.

**As built** (2026-11-09, GitHub #556-#559 and #562-#564), where the decision left a choice. Draft: the status stays
`proposed` until #560's figures are in.

- **The companion** (#556): `qwen3_8_flash_next_vision-v2.ninfer`, 482,499,840 bytes, its own identity
  (`qwen3.8-flash-next-vision`, weights `vision-w8-bf16-de4b8e4`) and a pair pin to the main container,
  built by `tools/flash-next-vision/convert.py` from the first shard's visual range alone (897,862,112 bytes), every
  matrix W8 (groups of 32, binary16 scales), the norms, biases and position embedding BF16. It is published beside the
  main container in Flash-Next's model repository, and the catalog, pinned at the hub commit that added it, fetches it
  with the model.
- **Every matrix W8, not the 27B tower's recipe** (#560, the owner's decision of 2026-11-10). The companion first
  shipped (284,229,120 bytes, weights `vision-w8q4q5q6-bf16-de4b8e4`) held the 27B's formats: the patch embedding Q6,
  the blocks' projections Q4 and Q5, the merger W8. #560's tower check put its served rows 0.305 in relative L2 from
  the BF16 tower's against the 27B's 0.183; the diagnosis found the encode computing what the stored words say, the
  weights quantizing alike in both towers, and the error growing at Flash-Next's retrained merger, where 31% of it was
  one vector every row shares. With every matrix W8 the CPU simulation gave 0.029, for 198,272,448 bytes more of
  tower (481,445,216 in place of 283,172,768), the same staging slot (22,608,384 bytes) and no new kernel route.
  The 27B's tower and recipe do not change. A companion of the grouped encoding is refused when it opens, naming both
  weights ids, so a start or a switch refuses it before anything loads.
- **Vision and media settings per family** (#557): the vision and media fields apply to both text families, each
  family its own; the four built-in profiles turn vision on for the 27B only, so a Flash-Next start loads no tower
  unless it asks (`--vision-enabled`, `VISION=1`). The residency and the envelope are shared. The host plan shows
  the tower (482,705,408 bytes of host spans) and the pool as their own lines.
- **The item bound** is computed when the tower is bound: the largest item whose encode fits the prefill scratch
  first, then the `host` residency's staging slots in the expert staging ring, then the encode's workspace there,
  then both, at most 16,384 merged tokens (#557, #558). The ring was shown first to hold nothing live while an item
  encodes: only a prefill stages into it, a decode miss takes an LRU victim, and the encode joins the residency
  before it writes there. A card that holds every expert has no ring, and the prefill scratch alone bounds the item,
  as on the 27B. The VRAM plan logs the bound, and the server lowers the processor's pixel bound to its area.
- **The same expert cache** (#558): the forward's per-chunk vision inputs, the chunk's three axes and its media
  rows, are carved from the decode graphs' scratch, idle in a prefill; the rope deltas sit in the positions buffer's
  alignment; a verify round's axis records sit at the prefill scratch's base, idle in a round. A vision prefill
  refuses while the decode graphs' scratch is in use, and a verify round while the prefill scratch is. At the RTX
  5090's settings every reservation line equals the text-only load's, MTP on and off.
- **A sequence keeps the three axes its indexer still needs, not every token's** (#558), against the second point of
  Consequences: a compressed block's key is computed when its block completes, so the one position a sequence must
  carry is its incomplete block's first token's. It is kept in that block's own key slot, an axis record (a magic, the
  first token's KV position, its three axes), written when a call begins a block and leaves it incomplete, read and
  overwritten by the key when a later call completes the block. It travels with its page through snapshots,
  checkpoints and spills, at no VRAM and with no change of the blob layout; a verify round saves and restores each
  lane's. The rope delta rides the progress image; a checkpoint's capture zeroes it, and its claimant takes its own
  from the span it prefills (#559).
- **The MTP head** (#564): its prefill entry before an image placeholder takes the vision column the trunk took, and
  a span ending inside its image the image's next column; decode, verify and the chain embed generated tokens by id,
  unchanged. The entry before an image that starts a new span takes the placeholder's id, the image not being encoded
  yet: it costs acceptance, never the output.
- **The merger's width needed a route** (#557): the vendored W8 dispatch had none for the merger's second linear at
  [2560, 4608]. It takes the routes the image model's [4096, 4608] takes, a recorded patch of the vendored dispatch
  with its op test, on the standing reason ADR 0088 gives a vendored registry that lacks a case ignis needs.
- **A switch keeps vision** (#563): the vision settings carry across families. `POST /v1/models/switch` to a family
  whose companion is missing answers 409 `vision_companion_missing`, naming the file, before the active model is torn
  down; an implicit switch meets the same check when it prepares, before teardown. The request size cap follows the
  active model's media.
- **Measured** ([finding](../findings/2026-11-09-flash-next-images-checked-without-the-bf16-checkpoint.md)): [TBD:
  the encoded tower against the BF16 tower beside the 27B's, the known answers, the decode gate].
