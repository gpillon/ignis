---
status: accepted
---

# The vision tower is staged and the embedding pool host-mapped, unless the load asks for the device

A vision load that keeps the 27B's vision tower and its embedding pool (ADR 0035) in VRAM pays for them for its whole
life, whether or not an image ever comes: about 11% of the KV pool on a 16 GB card at a 1,024-token envelope, which is
why the 16 GB profile once shipped with vision off, and about 70K tokens on the 5090. A vision load therefore takes a
**vision residency**, the config field `vision.residency`:
- **`host`, the default.** It keeps both in pinned host RAM: the tower **staged** into the **prefill scratch** for
  each media encode, the pool **host-mapped**.
- **`device`.** Both resident in VRAM. It is for a card with VRAM to spare on a host short of RAM, and for an operator
  who wants the resident load's encode latency.

ADR 0030 holds: everything is still reserved at load. The tower and the pool leave the VRAM plan and enter the host
plan, each on a line of its own. A `host` start that cannot pin them is refused, naming the missing bytes and both
ways out (`device`, or vision off). It never falls back, since a fallback would make the KV pool depend on the host's
RAM. `--vision-embedding-pool-mib` keeps its meaning, in the place the residency puts the pool (ADR 0035).

Staging is sound because a media encode runs alone between prefill steps and nothing in the prefill scratch survives
from one step to the next, so the encode can use the scratch's room for the weights; and because only the encode
reads the tower once the load has bound it, and the encode is not captured in a CUDA graph, so no graph holds the
tower's addresses. The residency belongs to the bound scope and the leaf takes it as a load option: a load whose
option and scope disagree is refused, both ways. The media embedding is bit-identical to `device`'s
([the prototype's finding](../findings/2026-10-30-vision-tower-staged-and-pool-host-mapped.md),
[the build's](../findings/2026-10-30-vision-residency-acceptance.md)).

## Considered Options

- **Borrowing the encode's workspace and the pool, and optionally the tower, from the KV pool's free pages.** It gives
  back the same bytes, but only with a page loan (ADR 0067's mechanism), an admission for images that arrive with the
  pool full, and the encoder's kernels addressing scattered pages. It stays parked on #409, as the later opt-in for
  16,384-token items on a 16 GB card.
- **Copying the whole tower per encode.** It adds 22.8 ms at 1,024, and does not fit beside the workspace of a
  16,384-token item, nor in the smallest scratch beside a 1,024-token one. The tower is staged through a ring of
  slots sized for its largest stage instead; slots sized to a block, with one of the merger's own, would take less
  but are unmeasured.
- **The merger writing the pool in place through the mapped pointer.** `add_bias` reads each page back over PCIe:
  +17 ms at 1,024, against +7.7 ms through a device page.
- **The GEMMs reading the tower's weights in place from host memory.** Their tiles re-read the weights 37 to 128
  times.
- **`cpu` and `on-demand` residencies (#407).** `cpu` takes 25-170 s an image and holds the compute lock; `on-demand`
  allocates after load and gives ignis no KV.
- **A query-chunked encode, which shrinks the encode's workspace.** Parked as #412: it matters only for items larger
  than the prefill scratch holds for free.

## Consequences

- WDDM counts the pinned host RAM as shared GPU memory in Task Manager, as it does ADR 0062's token embedding. That
  is not VRAM oversubscription.
- The first encode of an image costs a few milliseconds more under `host`, +7.1 ms at 1,024 on a PCIe Gen 3 x16 host
  under WDDM, most of it cross-stream synchronisation, not bytes. A fan-out of decisions over one image pays this
  once, and a crop pays it again. Linux is unmeasured; its synchronisation is expected to cost less.
- Where the prefill scratch already holds the encode's share, vision costs no VRAM, which is what lets the 16 GB
  profile turn vision on (ADR 0063). A start whose scratch is smaller pays the difference and is not refused.
- The answers do not change at the same context. On a card whose start adapts its context to free VRAM, the KV `host`
  returns can raise the adapted context, and with it the YaRN factor `auto` rope scaling picks, so answers move, as
  with any change of free VRAM. A comparison of the two residencies pins `--model-max-context`.
