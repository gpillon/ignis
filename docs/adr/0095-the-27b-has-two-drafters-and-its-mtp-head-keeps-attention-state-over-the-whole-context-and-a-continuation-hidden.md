---
status: accepted
---

# The 27B has two drafters, and its MTP head keeps attention state over the whole context and a continuation hidden

The 27B drafts with DFlash2, a second small model, or with the checkpoint's own draft head, its MTP layer, which
every 27B artifact ignis ships carries in the same bytes (the head is never trellis): one drafter a load, chosen at
load. DFlash2 stays, and stays the default wherever ADR 0098 does not give a card class the head; the head is the cheap
drafter the small cards need, its load holding nothing of DFlash2: no drafter weights, window, round or tap scratch.
The head is the checkpoint's and its program is ignis's own, on the op layer every 27B matrix runs on: it reads the
target's token embedding and output head, having neither of its own, and nothing of the reference engine's MTP is
ported, its source being a design reference.

- **The head's attention state spans the whole context, in the KV pool, as one more attention layer**, its pages
  shared, cloned and freed as the target's are: the head was trained attending over every row before it. A pool and a
  blob without the head are what they were, byte for byte.
- **The entry at `p` is stored at `p` and written when `x_{p+1}` is consumed**, so after any completed step the head
  holds every position but the sequence's last, whose entry waits for the token the sequence consumes next. An entry is
  built from the token consumed, never from one drawn and then replaced: that is what makes the head right when the
  host replaces a pending token (a grammar's repair, a forced close, the loop guard), and nothing has to be rewritten.
- **A sequence keeps a continuation hidden**, the target's post-final-norm hidden at its last consumed position, in
  its pool slot: the half of the last position's entry that the step which consumed that position had. Without it the
  next step would need the target's hidden of a position it does not compute. A restored or claimed sequence that
  carries it builds its first head row, and drafts, in its first round.
- **The page holding a frontier's last position is every holder's own.** A prefix is whole pages, and the entry at its
  last position is written after the publish, by the publisher's next step and by every claimant's first, each from
  its own next token. The claimant's entry must be its own: bit for bit a sequence's that prefilled the same tokens
  cold, never its publisher's. Flash-Next's head, which stores its entries the same way, keeps its rule: a claimant
  there reads its publisher's last entry.
- **The holders' copies are bounded, not tracked**: at most one page more a sequence and one a retained slot, which
  the load keeps back from admission, so admission never asks the leaf for a page it lacks. The plan's floor holds
  those pages too: without them a pool at its floor leaves admission a page a lane short of one context, and a request
  reserving the whole context the server advertises is refused as larger than the pool.
- **The head drafts at the start of a verify round, and the round's commit rewrites the head's rows from the committed
  run alone.** The chain's rows are scratch: built from the head's hidden, not the target's, they are not entries. The
  head drafts the fixed draft, at every temperature, its only rule. The continuation hidden moves last, so a round
  that fails before it leaves the lane's frontier and its hidden together.
- **The head's state is held to the committed run in each round, not bit for bit against a sequence that committed
  the same tokens by plain rounds**, as spec #618 asked. That cannot hold, for a reason outside the head: a committed
  row is built from the target's hidden at its position, which is the round's that computed it, and a plain round's, a
  verify round's (ADR 0056's drift) and a verify round's at another extent already part; the head's own call also runs
  at the round's width, where the W8 small-T route splits its K by the width. So in each round, under BF16 KV, drafts
  accepted and rejected: the continuation hidden is the round's last committed column's, bit for bit; the head's rows
  at the run's positions are the head's in double precision over what the run consumed, within the BF16 bounds of the
  head's double-precision checks; and a row built as the chain builds its own lands far outside them.
- **The head leaves the target's computation as it was**: the verify substrate alone, given each round the drafts
  the head's round committed and one the target rejects, commits the same runs at the same extents, bit for bit. The
  greedy text is the same load's spec-off text up to the near-tie rule, with one exception the rule does not know:
  where both prefill routes put spec-off's pick ahead but the verify substrate, given the same drafts, makes
  spec-on's, the two picks may stand at most 16 BF16 ulps apart at the routes' top logit, ADR 0056's moved-pick bound.
  That is the verify tile's drift from a one-column decode, which any drafter's accepted drafts expose, not the head's.
- **On an RTX 30 or 40 card the head's matrices run on routes of the full-rate rule**, decided once at load from the
  card: the vendored W8 routes are BF16 tensor-core MMA with FP32 accumulation, which the rule refuses there, and their
  small-T kernel is in the sm_120a set alone. A narrow call runs the vendored W8 SIMT kernel and a wide one ignis's
  own GEMM on the integer MMA, the activation in a 16-bit fixed point, so the head computes what it computes on
  Blackwell within the roundings its double-precision checks allow. The VRAM plan is the same on every card.
- **The head's state travels with the sequence, and a blob belongs to one drafter.** The state is in the sequence's
  sections (ADR 0024), so every move of a sequence carries it: a snapshot and its restore, whole or a window at a time,
  KV-RAM and KV-disk, a retained slot on the device or on the host, a prefix's image and a checkpoint's. Its snapshot
  point is earned, not inherited. The head did not bump the blob's format version: a pool without the head writes the
  bytes it wrote before, and a blob of either kind of pool restored into the other is refused by name before its
  geometry is read. The blob identity names the drafter: a load refuses the blob, or the KV-disk file, of another
  drafter, or of none, by it, and the refused prompt is prefilled again.
- **The server takes the head as the 27B's opt-in drafter.** `--spec-backend mtp` is both models' value, and a model
  lists the backends it drafts with, its default first, so a model switch to the 27B keeps an MTP setting instead of
  dropping it. The draft window is the load's, and the two drafters share no range: the head takes 0 to 7 and DFlash2
  1 to 15 (ADR 0104), so a window named beside one is that drafter's own, and `mtp` named over a profile's DFlash2
  window takes the head's default, a measured one. A window of 0 is a load with no speculation, what `off` resolves
  to, field for field: nothing of the head is bound, no pool carries its section and the plan is the off load's to the
  byte. Flash-Next's head drafts 1 to 7, so 0 named for both models is refused when Flash-Next loads, at start and on
  a switch alike (a bound, as a context past the 27B's envelope is, not a drop), naming the 27B's own flag. A rule only
  the loading model decides runs where the configuration is fitted to it, never where every model's values are
  validated, or a value for one would refuse a start of the other. DFlash2's own settings are ignored on the head's
  load, wherever and at whatever value they are named, as every draft setting is under `off` and with no backend.
- **The head drafts after images**, binding beside the tower under either residency, as DFlash2 does. A row rotates
  where the target's hidden at its position did, and its key lands at the position itself, the KV index. A row whose
  next token is an image placeholder embeds the image's column the target's embedding took there, not the placeholder
  id's row, which every placeholder id shares and so cannot tell one image from another (Flash-Next's rule); the entry
  at `p` being written by the step that consumes `x_{p+1}`, that column is always in the chunk that writes the row,
  and nothing like Flash-Next's continuation past a span ending inside an image is needed. The row before a span that
  continues a sequence rotates at axes the span names: the sequence cannot say where that position rotated, its rope
  delta being the whole prompt's, set after each span, and a prefix's image carrying none.

## Considered Options

- **A head that sees only the last rows**: it would save the state's bytes, and the head was never trained so.
- **Each entry stored one slot later, at `p + 1`**, so that a step writes only the slots of the tokens it consumes:
  the attention kernels attend every key from 0 to the query's position, so slot 0 would stand empty in every query's
  keys, and excluding it needs a first key in every attention route the head runs (the vendored BF16 and hq kernels,
  ignis's own hq prompt and decode routes), which is out of proportion to what it buys.
- **Plain A8, one INT8 plane, in place of the 16-bit fixed point**: it adds an activation step of 2^-8 at each matrix,
  which no RTX 30 or 40 card was at hand to measure against acceptance; the fixed point's step sits below the BF16
  rounding of every output.
- **An existing full-rate route for the wide calls**: the dense prefill's GEMM keeps one integer accumulator over the
  whole K under a single scale, the trellis copy's affine map, where W8G32 has a scale per row and 32 inputs, and the
  SIMT kernel reads the matrix once per 8 tokens.
- **A format version of their own for the head's blobs**: the blob identity would name the layout version rather than
  the drafter, the field that differs.
- **The axes of the position before a span kept as sequence state**: they would be a section every move carries, and
  the caller already holds the whole prompt's positions.

## Consequences

- The head's load pays one more attention layer in every KV page, one vector a pool slot, a prefill that normalizes
  every column the head reads, where it normalized only its last, and one page a holder of a shared prefix.
