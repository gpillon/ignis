# The vision tower is staged and the embedding pool host-mapped, unless the load asks for the device

A vision load used to reserve the 27B's vision tower (295,711,648 bytes) and its embedding pool (ADR 0035) in VRAM for
its whole life, whether or not an image ever came. On a 16 GB card at a 1,024-token envelope that was 33,216 KV tokens,
about 11% of the pool, which is why `rtx50-16gb` shipped with vision off (ADR 0063, amended 2026-10-29). On the 5090 it
was about 70K tokens. A vision load now takes a **vision residency**: the config v2 field `vision.residency`, or
`--vision-residency`.
- **`host`, the default.** It keeps both in pinned host RAM: the tower **staged** into the **prefill scratch** for each
  media encode, the pool **host-mapped**.
- **`device`.** Today's resident load, unchanged. It is for a card with VRAM to spare on a host short of RAM, and for
  an operator who wants today's encode latency.

ADR 0030 holds: everything is still reserved at load. The tower and the pool leave the VRAM plan and enter the host
plan, each on a line of its own. A `host` start that cannot pin them is refused, naming the missing bytes and both ways
out (`device`, or vision off). It never falls back, since a fallback would make the KV pool depend on the host's RAM.
`--vision-embedding-pool-mib` keeps its meaning, in the place the residency puts the pool. This amends ADR 0035: the
pool is the load's reservation wherever the residency puts it.

How `host` works:
- **The tower.** A media encode runs alone between prefill steps, and nothing in the prefill scratch survives from one
  step to the next, so the encode can use the scratch's room for the weights. The encode's share of the scratch holds
  a ring of three slots, each sized for the largest stage (the merger's fc2, 25.1 MB), 75.3 MB in all. The encoder's
  own copy stream fills the next slot with the next stage (a block's 8.9 MB, or a part of the merger) while the
  current one computes. Slots sized to a block, with one of the merger's own, would take 43 MB, but they are
  unmeasured. Only the encode reads the tower
  once the load has bound it, and the encode is not captured in a CUDA graph, so no graph holds the tower's addresses.
- **The pool.** The merger writes each column page to a device page in the prefill scratch, then copies it into the
  host-mapped pool. The prefill chunk's scatter reads the pool where it sits.

The owner chose this on GitHub #409 (2026-10-29 and 30), from the study there and a prototype on branch
`prototype/409-vision-host` (abfafa36).

## Measured by the prototype

The [finding](../findings/2026-10-30-vision-tower-staged-and-pool-host-mapped.md) holds the commands and the tables.
On the 5090:
- **Identity.** The media embedding is bit-identical to `device` at 1,024-token and 16,384-token items, and the canary's
  logits are identical at every scored position.
- **The KV pool.** On a 16 GB budget with the `rtx50-16gb` profile, the pool has 7,956 pages, the same as a text-only
  load, at both 1,024 and 4,096. With `device` it has 7,437 and 7,384. The prefill scratch stays 395,898,880 bytes.
- **Encode time.**
  - A 1,024-token encode takes ~40 ms with `device`.
  - Staging the tower adds 3.0 to 4.2 ms with three slots. A two-slot ring adds about 5 ms. Of that, about 4.2 ms is the
    cross-stream synchronisation under WDDM, not bytes: with every event kept and nothing copied, it still costs that
    much.
  - At 16,384 the staging is hidden.
- **Prefill.** A 16,384-token image's prefill is -0.4% (harness) and +1.2% (server TTFT) against `device`.
- **The 5090's own profile** at a 16,384-token item bound: the scratch grows 50.2 MB with two slots, 75.3 MB with
  three, against the 643 MB the tower, the pool and the encode's workspace cost there. The load adapted its context to
  589,824 tokens, against 524,288 with `device`.

## Profiles (amends ADR 0063)

- **`rtx50-16gb`: vision on, with `vision.max_tokens` 4,096.** The item bound is then 4,096, about 2,048×2,048 px. This
  takes effect once the build passes its gate. The encode of such an item plus the ring is 302.4 MB.
  - The default prefill scratch is 395,898,880 bytes, so that leaves 93 MB to spare.
  - The smallest scratch seen on this profile is 303,619,584 bytes (a 65,536-token context and one lane, #360). There
    it fits by about 1 MB.

  So vision costs this profile no VRAM. A start whose scratch is smaller still pays the difference and is not
  refused.
- **`rtx5090`: `host`.**
- **`rtxpro6000`: `device`.** There the tower and the pool are ~0.8% of the 27B's 8,351,168-token KV pool.

## Considered options

- **Borrowing the encode's workspace and the pool, and optionally the tower, from the KV pool's free pages** (#409's
  B + C). It gives back the same bytes, but only with:
  - a page loan (ADR 0067's mechanism, #356);
  - an admission for images that arrive with the pool full;
  - the encoder's kernels addressing scattered pages.

  It stays parked on #409, as the later opt-in for 16,384-token items on a 16 GB card.
- **Copying the whole tower per encode.** It adds 22.8 ms at 1,024. It does not fit beside the workspace of a
  16,384-token item, nor in a 303.6 MB scratch beside a 1,024-token one.
- **The merger writing the pool in place through the mapped pointer.** The mapped writes run at ~2.6 GB/s, and
  `add_bias` reads each page back over PCIe. That makes the encode +17 ms at 1,024. Through a device page it is +7.7 ms,
  measured with two slots.
- **The GEMMs reading the tower's weights in place from host memory.** Their tiles re-read the weights 37 to 128 times.
- **#407's `cpu` and `on-demand` residencies.** `cpu` takes 25-170 s an image and holds the compute lock. `on-demand`
  allocates after load and gives ignis no KV.
- **A query-chunked encode, which shrinks the encode's workspace.** Parked as #412. It matters only for items larger
  than the prefill scratch holds for free.

## Consequences

- **Pinned host RAM.** `host` pins ~306 MB of host RAM on a 16 GB card at 1,024, ~338 MB at 4,096, and ~631 MB on the
  5090. WDDM counts it as shared GPU memory in Task Manager, as it does ADR 0062's token embedding. That is not VRAM
  oversubscription.
- **Encode latency.** On a PCIe Gen 3 x16 host under WDDM, the first encode of an image takes ~6-8 ms more at 1,024:
  - about +6.5 ms estimated for three slots with the device page;
  - +7.7 ms measured with two slots.

  A fan-out of decisions over one image pays this once, and a crop pays it again. The answers do not change, since the
  embedding is bit-identical. Linux is unmeasured; its cross-stream synchronisation is expected to cost less.
- **The build's acceptance:**
  - at most +10 ms at 1,024 with `host`;
  - `/v1/decide` answers identical between `host` and `device`, and a cold `point` at 1,024 px at most +10 ms over
    `device`;
  - `device`'s `vram_plan` line identical to `main`'s before the change in every attribute but its timestamp,
    `free_at_start_bytes` and `service.version`, with its encode and `point` times within noise;
  - #409's 99% decode gate, once at the end. If it fails, the owner decides.
- **Credit.** The staging's pointer rebase follows #407's `visit_weights`, and its author is credited.

**As built** (2026-10-30, GitHub #413, tickets #414-#418), where the decision left a choice:

- **One residency for the scope.** `VisionResidency` belongs to the bound scope, and the leaf takes it as a load option. A load whose option and scope disagree is refused, both ways. Under `host` the binder places the `vision/*` tensors in the host-mapped allocation beside the token embedding.
- **The ring.** Three uniform slots of the merger fc2's 25,094,656 bytes (75,283,968 in all) follow the encoder's workspace in its share of the prefill scratch.
  - On the 5090 at a 16,384-token item, the scratch grows from 908,592,896 to 983,876,864 bytes.
  - Under `rtx50-16gb` at 1,024 and 4,096 it does not grow.
  - The bounce page is the slot the merger's norm and fc1 were read from. A static check and a load-time check hold that a pool page fits a slot.
- **The host plan.** It has two lines of their own, `vision_tower` (297,000,960 bytes of spans on the served artifact) and `vision_pool` (the pool at the shape's context), after the token embedding's.
  - A start that cannot hold them is refused. The message names the bytes short, `vision.residency: device` and turning vision off.
  - A host pool that fails to pin at load fails the load the same way.
- **Accounting.** On the 5090 the leaf's host-mapped bytes go from 1,350,860,800 (`device`) to 1,657,058,208 (`host`), and its VRAM from 16,151,713,660 to 15,856,002,012.
  - `vram_plan` and the host-plan events carry `vision_residency`, and the host plan carries `vision_tower_bytes` and `vision_pool_bytes`.
  - The decode graph's line now reads an explicit vision flag, since the pool's line is 0 under `host`.
- **Measured** ([finding](../findings/2026-10-30-vision-residency-acceptance.md)): all five acceptance criteria hold.
  - On a 16 GB budget a `host` load keeps the text-only KV pool: 7,956 pages at 1,024 and 4,096.
  - A 1,024-token encode costs +7.1 ms (49.2 against 42.1 ms, n=90 each, interleaved).
  - At the same context, `/v1/decide` answers 198 of 198 questions identically, and a cold `point` costs +5.9 ms.
  - `device`'s `vram_plan` equals `main`'s at a fixed budget.
  - Decode is 99.7% / 100.3% / 100.0% of main's at C=1/3/8.
- **The answers do not change at the same context.** On a card whose start adapts its context to free VRAM, the KV `host` returns can raise the adapted context, to 786,432 against 720,896 on the 5090. With it rises the YaRN factor `auto` rope scaling picks, from 2.75 to 3, and answers move, as with any change of free VRAM. A comparison of the two residencies pins `--model-max-context`.
