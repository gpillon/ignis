---
status: accepted
---

# The vision embedding outlives its encode

A media item's embedding stays resident after the request that encoded it, in one reservation of the load carved
into fixed-width column pages, which holds as many embeddings as their own columns fit. A fan-out of N questions
over one image encoded it once per question, and the encode was the costliest stage and the one nothing shared:
4.17 s of a 6.84 s question on a 4096×4096 screenshot, where the prepared patches were already shared by content
digest ([measured](../findings/2026-09-20-number-width-and-decide-e2e.md)). The owner's target is four to eight
frames resident, for real-time decisions.

- **An embedding is contiguous in column space, not in memory, and no kernel reads a page table.** The encoder's
  last projection writes one page at a time and a prefill chunk scatters one page run at a time, each a column
  range over the ops they already had.
- **The leaf owns the bytes, the runtime owns the policy.** The leaf refuses an item that fits the pool but not the
  pages free right now with an answer of its own, distinct from the one that means the call can never work. The
  runtime caches by key with a hold count per entry, and on that answer releases the least recently unheld entry
  and asks again. Nothing above the seam counts bytes or pages, and nothing below it decides what to give up.
- **The key is the content digest and the grid, not the prompt offset** that a request's media key also carries:
  two siblings do not share the offset when the question precedes the image, and the encoder sees neither, only
  these bytes at this grid.
- **The pool is floored at one envelope-wide item.** That is what ends the eviction loop, not a comfort margin: an
  item that fits the envelope must fit the pool once everything else is released, or a caller told to release and
  retry could loop forever. An operator asking for less is raised to the floor rather than refused: nobody should
  have to recompute the envelope's bytes to lower a cache.
- **`--vision-embedding-pool-mib` sizes it and defaults to that floor**, which is what a load reserved for one
  item's output before the pool existed, so a load that says nothing does not move its plan.
- **The pool is the load's reservation where the vision residency puts it** (ADR 0075): host-mapped under `host`,
  the default, and in VRAM under `device`. Its pages, its holds, its sizing and the flag's meaning are the same in
  both. A pool the plan of that place cannot hold fails the load by name (ADR 0030).

## Considered Options

- **Reference counting alone, entries dropped at zero holders.** It recovers nothing: the siblings of a fan-out
  never overlap, since one request at a time holds multi-tick prefill progress and an embedding is given up at its
  item's last placeholder, before the next sibling's first chunk runs. An entry outliving its last holder is the
  mechanism, not a tuning knob on a reference count.
- **N envelope-wide slots, behind a flag.** Simpler, and a smaller change to a single output transient. But an
  embedding's width spans two orders of magnitude, from a thumbnail to the envelope, and a slot reserves the
  envelope's bytes whichever it holds, which defeats the resident-frames goal it would exist for.
- **A suballocator over variable-size contiguous spans.** No page-run loops, at the cost of fragmentation across a
  200× size range. Paging is what the KV cache already does, and the loops are four lines each.
- **Encoding contiguously, then copying into pages.** A second envelope-sized reservation to encode into.

## Consequences

- A declared departure from the reference, which keeps a single output transient. ADR 0010's verbatim rule governs
  kernels; this is the leaf's allocation policy, and the departure is the point. It is not free of risk: the pool
  is a second thing that can be sized wrong at load, hence a floor, a ceiling and a named refusal instead of a map
  that grows.
- Two behaviours improve, both intended: a request evicted mid-item finds its picture's embedding on restore
  instead of encoding it again, and a prefill batch that fails after its encode reuses it on the retry.
- It does not make a new frame cheaper: a picture nobody has seen costs the whole tower whatever the pool holds.
  The pool saves on the fan-out axis, one picture and many questions, and makes holding N frames possible, but
  nothing encodes a frame before a request asks for one. That trigger, a warm-ahead call and a pin against the
  eviction order, is new API surface and no part of this decision.
