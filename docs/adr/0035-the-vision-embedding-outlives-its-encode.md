# ADR 0035 — the vision embedding outlives its encode

## Status

Accepted (2026-09-20, owner). **Amends ADR 0030's `media_embedding` plan
line**, which reserved one item's output transient: the same line now reserves
a *pool*, sized by `--vision-embedding-pool-mib` and defaulting to exactly what
it always was, so the plan's shape is unchanged and the number it carries at
the default is not. **A declared departure from the reference** — see
*Consequences*.

Sources: owner decision 2026-09-20 on GitHub #243; the split measured in
`docs/findings/2026-09-20-number-width-and-decide-e2e.md`; spec
`.scratch/vision-fanout/specs/01-share-the-vision-embedding.md`.

## Context

A fan-out over one image encodes it once per question. Measured on a
4096×4096 screenshot, 16,384 merged vision tokens, one `/v1/decide` question
per request:

| stage | per question | shared before this ADR |
|---|---|---|
| media preprocess | 0.17 s first, 0.00 s after | yes, by content digest |
| **vision encode** | **4.17 s** | **no** |
| attention prefill, 16,506 tokens | ~2.4 s | no |

End to end: 1 question 6.84 s, 2 questions 13.50 s, 4 questions 27.92 s.

The CPU half was already shared — the server's `MediaCache` keys prepared
patches by content digest, and every follower in a fan-out reports
`media.cache_hits: 1` with `media.preprocess_seconds: 0.000`. What had no cache
at any tier was the thing that costs the most: the device-resident embedding
those patches encode to, keyed by `RequestId` and dropped at the end of that
request's prefill.

Two facts shaped the decision. Both were found in the code, not assumed.

**Reference counting alone recovers nothing.** The obvious fix — key the live
embedding by content instead of by `RequestId`, refcount it, drop it at zero
holders — recovers 0%, because the siblings never overlap. `ignis_core::concrete`
admits exactly one request holding multi-tick prefill progress at a time, and
`prefill_multimodal_job` gave the embedding up at its item's *last placeholder*,
which is before the next sibling's first chunk runs. An entry **outliving its
last holder** is therefore not a tuning knob layered on a refcount. It is the
mechanism.

**An embedding's width spans two orders of magnitude.** One embedding is
`merged_tokens × hidden × 2 B`, and at this artifact's 5120 hidden one column
is 10,240 B. A 320×240 thumbnail is 80 columns — 0.78 MiB. A 4096×4096
screenshot is 16,384 columns — 160 MiB. The envelope is 32,768 columns —
320 MiB, which is exactly what the load reserved. Holding *N* embeddings in *N*
envelope-wide slots would reserve 320 MiB to hold a thumbnail. The owner's
stated target — four to eight frames resident, for real-time decisions — is
unreachable that way and cheap the other way.

## Decision

**One reservation, carved into fixed-width column pages, holding as many
embeddings as their own columns fit.**

1. `ignis_model::vision_output` — one `DeviceBuffer` plus a `bool` saying
   whether it was taken — becomes `VisionEmbeddingPool`: a buffer, a page
   width, and one owner pointer per page. An `ignis_media_embedding` carries
   the ordered list of pages its columns live in. An embedding is contiguous in
   *column space*, not in memory.

2. **A page is 128 merged columns** (`IGNIS_MEDIA_EMBEDDING_PAGE_COLUMNS`),
   1,280 KiB at 5120 hidden. A column is 10,240 B = 40 × 256, so every page
   boundary is 256-aligned whatever the page width — which is what lets both
   sides of the pool keep the ops they already had.

3. **No kernel reads a page table.** The encoder's last projection writes one
   page at a time, because the columns are the GEMM's N dimension: the same
   `ops::linear` + `ops::add_bias` over a narrower column range. A prefill
   chunk scatters one page run at a time, because `ops::scatter` already took a
   column range. Both loops are four lines. The alternative — encode
   contiguously, then copy into pages — was rejected: it would cost a second
   envelope-sized reservation to encode into.

4. **The leaf owns the bytes, the runtime owns the policy.**
   `ignis_media_encode` answers `IGNIS_MEDIA_ENCODE_POOL_FULL` (-2) when an
   item fits the pool but not the pages free right now — distinct from -1,
   which means the call can never work. `RuntimeCompute` keeps a cache keyed by
   `(content digest, grid)` with a hold count per entry; on -2 it releases the
   least recently unheld entry and asks again. Nothing above the seam counts
   bytes or pages, and nothing below it decides what to give up.

5. **The key is the digest and the grid, not the prompt offset.**
   `ignis_core::identity::MediaKey` also carries where the item's placeholders
   begin, which two siblings of a fan-out do not share when the question
   precedes the image. The encoder sees neither — it sees these bytes at this
   grid.

6. **The pool is floored at one envelope-wide item.** This is the termination
   argument for the eviction loop, not a comfort margin: an item that fits the
   envelope must fit the pool once everything else is released, or a caller
   told to release-and-retry could loop forever. An operator asking for less is
   raised to the floor rather than refused — nobody should have to recompute
   the envelope's bytes in order to lower a cache.

7. **`--vision-embedding-pool-mib` / `IGNIS_VISION_EMBEDDING_POOL_MIB`,
   defaulting to one envelope-wide item.** That default is exactly the
   reservation GitHub #177 always took, so a load that says nothing does not
   move the VRAM plan. A pool the budget cannot hold fails the load by name, as
   ADR 0030 requires.

## Consequences

**The `media_embedding` plan line keeps its name and changes its meaning.** It
was "one item's `[hidden, V]` encoder output"; it is now "the embedding pool,
rounded to whole pages". At the default it is the same number, so no recorded
plan moves and no gate baseline shifts.

**This is a declared departure from the reference**, which keeps a single
output transient — `model_internal.h` said so in as many words. ADR 0010's
verbatim-port rule governs *kernels*; this is the leaf's allocation policy, and
the departure is the point of the slice rather than an accident of it. It is
not free of risk either: the pool is a second thing that can be sized wrong at
load, which is why it has a floor, a ceiling and a named refusal instead of a
`HashMap` that grows.

**`live_media()` no longer counts the cache.** It counts entries a request is
prefilling against, which still returns to zero after a fan-out;
`cached_media()` and `cached_media_columns()` report what is resident. Every
existing assertion of `live_media() == 0` keeps its meaning, which is why the
two were split rather than one redefined.

**Two behaviours improve as a side effect, and both are intentional.** A
request evicted mid-item used to re-encode its picture on restore, because the
load had room for exactly one embedding (GitHub #194 said so); it now finds it.
A prefill batch that failed after encoding used to throw the encode away; the
retry now reuses it.

**What this does not do.** It does not make a *new* frame cheaper: a picture
nobody has seen costs 4.17 s of tower whatever the pool holds. The pool saves
on the fan-out axis — one picture, many questions — and it makes holding N
frames *possible*, but nothing encodes a frame before a request asks for one.
That trigger (a warm-ahead call, and a pin against the eviction order) is new
API surface, and is filed separately rather than smuggled in here.

**Rejected alternatives.**

- *Reference counting alone, entries dropped at zero holders* — the spec's own
  first shape. Recovers 0%, for the reason in *Context*.
- *N envelope-wide slots, behind a flag* — simpler, and a strictly smaller
  change to the existing single-transient design. It reserves 320 MiB per slot
  whether the slot holds a screenshot or a thumbnail, which defeats the
  resident-frames goal it would exist for.
- *A suballocator over variable-size contiguous spans* — no page-run loops, at
  the cost of fragmentation across a 200× size range. Paging is what the KV
  cache already does on this engine, and the loops turned out to be four lines
  each.
