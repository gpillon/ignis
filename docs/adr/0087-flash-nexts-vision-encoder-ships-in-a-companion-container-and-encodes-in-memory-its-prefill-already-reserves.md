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
  on. They are encoded from the pinned revision's BF16 tensors with the 27B
  tower's recipe. The 72 GB main container and its content hash stay as they
  are; the download grows by about 0.3 GB, which the catalog always fetches,
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
  reference implementation on a small synthetic model; and accuracy on images
  with known answers, reported beside the 27B's. A KLD against BF16 on image
  prompts runs only if those raise an alarm.

## Considered Options

- **A new revision of the main container with the visual tensors**: 72 GB
  uploaded and downloaded again for 0.3 GB of tower, a new content hash, and
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
- With the main container unchanged, a text-only user downloads 0.3 GB more
  and loads exactly what loads today.
