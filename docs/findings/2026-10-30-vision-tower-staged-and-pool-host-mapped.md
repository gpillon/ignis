# The 27B's vision tower staged per encode and its embedding pool host-mapped: a text-only load's KV, bit-identical embeddings, +3 to +8 ms a 1,024-token encode under WDDM

- Kind: experiment
- Status: current
- Observed: 2026-10-30
- Last verified: 2026-10-30
- Scope: kernel / the vision encode (`kernel/src/vision_encode.cu`) staging the tower's weights into the prefill scratch, and the embedding pool in mapped pinned host memory; serving / the 27B's VRAM plan under `rtx50-16gb` and `rtx5090` on the RTX 5090 (PCIe Gen 3 x16 host, Windows/WDDM)
- Related: [ADR 0075](../adr/0075-the-vision-tower-is-staged-and-the-embedding-pool-host-mapped-unless-the-load-asks-for-the-device.md), [ADR 0062](../adr/0062-the-token-embedding-lives-in-host-memory.md), [ADR 0035](../adr/0035-the-vision-embedding-outlives-its-encode.md), [token embedding in host memory](2026-10-21-token-embedding-in-host-memory.md), https://github.com/gpillon/ignis/issues/409, branch `prototype/409-vision-host` (abfafa36)
- Superseded by: none

## Question

A vision load reserves the tower's 295,711,648 bytes and the embedding pool in VRAM for its whole life. On a 16 GB card
at a 1,024-token envelope that costs 33,216 KV tokens with no image in flight. #409's study proposed two placements:
- A1: the tower in pinned host RAM, staged into the prefill scratch for each encode;
- A2: the pool host-mapped.

Under them, does the KV pool equal a text-only load's? Is the media embedding bit-identical? And what do an encode and
a prefill cost?

## Evidence

**Setup.**
- A throwaway prototype behind environment flags on branch `prototype/409-vision-host`:
  - 4aaac2d8: A1 as a whole-tower copy (H1a) and a double buffer (H1b), A2, and the harness;
  - abfafa36: a ring of K slots, block grouping, a per-stage trace, a no-copy diagnostic and a bounce page.

  It is based on main 5256993b. With the flags unset, the build is today's: `vision_load_gpu.rs` and `model_load_gpu.rs`
  pass under `IGNIS_GPU_PROFILE=1`.
- The harness is `crates/server/tests/proto_409_vision_residency_gpu.rs` on the branch. It runs 3 warm-up encodes,
  then 30 timed ones, one process per mode.
- Raw logs, scripts and per-run tables are in `.scratch/409-proto/` of the owner's clone (`results.md`,
  `plans/vram_plan_lines.md`, `ttft/`, `runs/`).
- RTX 5090 on the owner's Z490 host, PCIe Gen 3 x16, Windows/WDDM.

**Plan on a 16 GB budget.** Every start is `make start ARTIFACT=./models/qwen3_8_27b_trellis_sc30-v2.ninfer
VRAM_BUDGET=16G IDLE_UNLOAD=0 LOOP_GUARD=off [VISION=1 VISION_MAX_TOKENS=N] ARGS=--profile rtx50-16gb`. It runs
hq-e8-2b with DFlash2 at 7 draft tokens, and `adapted` is empty in all seven starts.

| load | weights_bytes | workspace_bytes | media_embedding_bytes | kv_pool_pages |
|---|---|---|---|---|
| text only | 10,907,958,020 | 395,898,880 | 0 | **7,956** |
| vision `device`, 1,024 | 11,203,677,696 | 395,912,192 | 10,485,760 | 7,437 |
| A1 only, 1,024 | 10,907,958,020 | 395,912,192 | 10,485,760 | 7,939 |
| A1 + A2, 1,024 | 10,907,958,020 | 395,912,192 | 0 | **7,956** |
| vision `device`, 4,096 | 11,203,677,696 | 395,924,480 | 41,943,040 | 7,384 |
| A1 only, 4,096 | 10,907,958,020 | 395,924,480 | 41,943,040 | 7,885 |
| A1 + A2, 4,096 | 10,907,958,020 | 395,924,480 | 0 | **7,956** |

- **Weights.** `weights_bytes` drops by 295,719,676: the tower's bytes plus 8,028 bytes of device alignment.
- **Workspace.** The workspace is the same in every mode. The 13,312 and 25,600 bytes vision adds over text only are
  the attention readout for one image span (ADR 0038), present before the change.
- **Host plan.** The prototype charges the tower to the token embedding's host-plan line: +297,000,960 bytes, which
  includes 4 KiB span padding. It does not charge the host-mapped pool at all.

**Bit identity.** The harness loads the 32 GB artifact at chunk 2048 and a context of 8,192 (BF16 KV), or 20,480
(hq-e8-2b) for the large item. It encodes the four vision canary images and a deterministic 1,024×1,024 or
4,096×4,096 image.
- At 1,024: under H1a, H1b and H1b + A2, 0 of 5,242,880 values differ from `device`. Every canary prediction is
  identical, and the hash of every scored position's logits is identical, whole and in 48-token spans.
- At 16,384: under H1b and H1b + A2, 0 of 83,886,080 differ, and the last position's logits hash identically.

**Encode at a 1,024-token item.** Device is ~40 ms. The desktop moves the baseline 40 to 44 ms from round to round, so
the modes run interleaved, back to back, within each round.

| mode | pooled median delta over device | quiet rounds (device ~40.1 ms) |
|---|---|---|
| H1a, whole tower in one copy | +22.8 ms (n=120) | |
| H1b, 2 slots of 25.1 MB | +3.1 to +5.7 ms (four series, n=90-180) | +5.04, +5.08, +5.27 ms |
| H1b, 3 slots of 25.1 MB | +3.0, +3.3 ms (n=180, 90) | +3.94, +4.16 ms |
| H1b 2 slots, events and waits kept, nothing copied (diagnostic) | +4.2 ms | |
| A1 + A2, pool written in place | +17.0 ms (n=90) | |
| A1 + A2, pool written through a device bounce page (2 slots) | +7.7 ms (n=90) | |

A per-stage trace of H1b at 1,024 shows three costs:
- the first stage starts 1.1 ms late, because the item's 12.6 MB pageable upload shares the link;
- each block computes ~5% slower while the next one copies;
- with two slots, the merger's fc2 (25.1 MB) copies only after the last block, a 1.5 to 1.7 ms tail. A third slot
  removes the tail.

Grouping 2 to 6 blocks per stage, or a 47.7 MB merger region copied in 26 pieces, was slower.

**A2's writes.** The merger's fc2 and `add_bias` into a mapped page take ~1.5 ms a 128-column page, ~2.6 GB/s for the
GEMM's write plus `add_bias`'s read-modify-write over PCIe. Traced, that is 12.9 ms against 0.75 ms at 1,024, and
~204 ms against ~13.5 ms at 16,384. The bounce page sits in the slot the merger's first half freed, so it adds no
bytes.

**At a 16,384-token item** (the `rtx5090` profile's item bound):
- **Staging.** It is hidden: traced stalls total 0.03 ms. Harness encodes are device 3,812.6 ms against H1b + A2
  3,809.8 ms (±150 ms run to run).
- **Prefill.** The harness prefills the 16,405-token prompt with its embedding cached, 4 rounds × 8: device median
  1,342.1 ms, A1 + A2 1,336.5 ms (n=32 each, −0.4%).
- **Server on the profile.** The start is `make start ARTIFACT=./models/qwen3_8_27b_nvfp4full-v2.ninfer VISION=1
  VISION_MAX_TOKENS=16384 IDLE_UNLOAD=0 LOOP_GUARD=off ARGS='--profile rtx5090 --reuse-prompt false'`.
  - **TTFT with the cached embedding:** device 1,412.5 ms, host 1,429.0 ms (n=12 each, +1.2%).
  - **Encode:** device 3.62-3.63 s; host with in-place writes 3.91-3.95 s (+8%); host with the bounce page 3.765 s
    (+4%, n=1).
  - **Plan and context:** the scratch is 958,782,208 bytes against 908,592,896 (+50.2 MB, two slots). The context
    adapted from 917,504 to 589,824 under host, against 524,288 under device, with 10,772-10,805 KV pages against
    9,970-10,104.

## Finding

**Observed:**
- With the tower staged and the pool host-mapped, a vision load on a 16 GB budget has exactly the text-only load's KV
  pool: 7,956 pages, at both 1,024 and 4,096 tokens. Kept on the device, the tower and the pool cost it 519 and 572
  pages.
- Staging and the host-mapped pool leave the media embedding and the logits bit-identical.
- At 1,024, staging costs 3 to 4 ms with three slots and ~5 ms with two. About 4.2 ms of it is the synchronisation of
  events between streams under WDDM, not bytes moved.
- The host-mapped pool costs the prefill nothing measurable.
- Writing the pool in place through the mapped pointer is slow. A device bounce page per column page costs a third as
  much.

**Inferred:**
- Three slots plus the bounce page cost ~6.5 ms at 1,024. This combination was not measured.
- On Linux the synchronisation share should be smaller. Unmeasured.

## Implications

- ADR 0075 makes `host` the default vision residency, and keeps `device` as the option with today's encode latency.
- The build needs host-plan lines of its own for the tower and the pool. Its acceptance measures three slots with the
  bounce page.

## Limits and unknowns

- One host, one card, Windows only. Gen 4 or Gen 5 links and Linux are unmeasured.
- The 16,384 encode and server timings have a few samples each, with ±150 ms run-to-run noise.
- The decode gate (#409's 99%) was not run. The decode path does not read the tower or the pool.
- Slots sized to a block (8.9 MB) with a merger slot of their own were not tried. The prototype's slots are uniform at
  25.1 MB.

## Follow-ups

- The spec and build of ADR 0075 (from #409).
- [#412](https://github.com/gpillon/ignis/issues/412): a query-chunked encode, for items larger than the prefill
  scratch holds for free.
