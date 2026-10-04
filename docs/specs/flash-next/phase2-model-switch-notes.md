# Phase 2 — runtime model switch between the 27B and Flash-Next: notes, not a spec

GitHub: #305 (master #298). Status: **to design later** (owner, 2026-10-04). The ticket exists so these notes
are not lost. Nothing here is ready to implement; a spec gets written when the
owner opens the phase.

## The owner's rules (2026-10-04)

- A switch is a **full reload** in both directions. Everything belonging to the
  old model is dropped and the new one gets the whole machine:
  - VRAM;
  - pinned host memory;
  - KV-RAM;
  - retained slots.
- The two models are **never resident together**.
- **Never VRAM for the switch.** A little host RAM is acceptable only if it buys
  a lot of speed: "a few MB that make the switch 300% faster".
- An earlier idea of a "fast switch" flag that keeps parts of the other model
  loaded is superseded by the full-reload rule.

## Estimates from the study (not measured)

| direction | estimate | source |
|---|---|---|
| 27B → Flash-Next, cold | first token of a real prompt in ~14-15 s (2.5 bit) | `review/MEMORY_PLAN.md` |
| 27B → Flash-Next, warm Windows standby cache | ~6-8 s | same |
| Flash-Next → 27B | ~8-10 s cold, ~4-6 s if the artifact is still cached | same |

- Load Flash-Next **layer by layer**, so the prefill of layer L can start as
  soon as its experts have arrived.
- Windows' standby page cache helps the 27B → Flash-Next direction at zero
  cost. Read the artifact with buffered I/O, not `FILE_FLAG_NO_BUFFERING`.
- Spec 06 measures the real load times, and they replace these estimates.

## What the code already has and what blocks a switch (survey 2026-10-04)

- **Already there:** a full teardown chain, never used in production. It drops
  every engine clone and joins the model thread, then frees the runtime, the
  model and the leaf's model, then the weight arena and the pinned host pool.
- **Blockers:**
  - **Process singletons:**
    - the pinned host pool refuses a second create until the first is
      destroyed;
    - the attention tap holds a global sized by the GQA layer count.
  - **Server wiring:**
    - `Server::new` captures the engine and the tokenizer/template provider
      together;
    - `Engine.model_id` is immutable.

    All of these must be rebuilt on a switch.
- Specs 03-05 already require Flash-Next's own structures to be owned per model
  instance, with no new singletons.

## Idea to evaluate: a KV-disk tier that survives the switch

The owner finds it interesting (2026-10-04): a disk tier for retained state
(ADR 0029's Tier 2) is the only reuse tier that could outlive a switch.
- A conversation retained on disk before a switch could be restored when its
  model comes back, instead of being re-prefilled.
- That matters most for Flash-Next, whose prefill is PCIe-bound: ~10 s for a
  30K-token agent history.

Cost on disk per conversation: about 260 MB for a 30K-token Flash-Next
conversation (spec 05), about 1/7 of that per token for the 27B's hq KV plus its
fixed image. Restoring it is NVMe at ~3.3 GB/s, a fraction of a second.

## Questions for when the phase opens

- What triggers a switch: an explicit API call, the requested model id, or a
  router that picks the model per task?
- What happens to requests in flight and queued on the old model?
- Does the Playground / client see the switch (model id, a "switching" state)?
- Is the KV-disk tier worth it, measured on real agent sessions?
