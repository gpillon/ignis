# 02 — Retained slots on the host: `--retained-host` and `--retained-device`

GitHub: #281. ADRs: 0030 (amended by this feature), 0029 (the retained state
the slots hold), 0024 (the section packers the host copies reuse).
Evidence: the finding
[Retained slots on the host](../../findings/2026-09-28-retained-slots-on-the-host.md),
from the prototype and feature A/B runs driven by `scripts/agent-swarm.py`
and `scripts/swarm-ab.sh`.

## Problem Statement

Every retained slot (ADR 0030) is a lane-sized image of mutable state kept in
the device state arenas beside the lanes: 187.8 MiB of GDN, conv, penalty and
drafter state, plus 34 MiB of hq-e8-2b residual window. At the default of one
slot per decode lane that is 1.47 GiB on the `retained_slots` line and 272 MiB
more on `hq_residual_window` — about 200K tokens of hq KV the pool never gets.

Measured with an 8-agent synthetic swarm (8 turns each, shared ~13K-token
system head, greedy):

- **The device slots do little of the reuse.** With 8 slots, 51 of 52
  checkpoint hits came back from KV-RAM, not from a slot: 8 agents need about
  17 slots (a chain link and a checkpoint each, plus the shared head), so a
  conversation's image is spilled before its next turn.
- **Slot count matters more than KV at this load.** 16 device slots and a
  278K-token pool finished the long-context run (~57K tokens per agent) in
  70.0 s; 8 slots and a 478K pool took 93.5 s (12 captures skipped for want
  of a slot, 465 prefill chunks against 375).
- **Too few slots kill reuse, they do not move it to RAM.** With 2 slots the
  captures and publishes are skipped (the shared head pins them), nothing
  reaches KV-RAM — it is fed only by spilling a slot — and the run took 96.2 s
  against 51.9 s.

The prototype kept the images in one pinned host block instead (same
scheduler, same slot count semantics). A claim then crosses PCIe: 15.6 ms for
182 MiB at 12.2 GB/s against 0.29 ms device to device. Same binary, same load:

| | device 8 (today) | host 8 | host 16 | host 24 |
|---|---|---|---|---|
| KV tokens | 478K | 678K | 679K | 682K |
| long run wall / tok/s | 79.7 s / 136 | 85.2 s / 128 | **68.8 s / 147** | 69.6 s / 147 |
| short run wall / tok/s | 53.5 s / 194 | 55.0 s / 190 | **52.7 s / 211** | — |

Host slots cost a few ms per claim and per capture — per request, never per
token — and buy both more slots and 44% more KV.

## Solution

The retained slots come in two kinds, sized separately:

- **`--retained-host <n>`** (`IGNIS_RETAINED_HOST`): images in one pinned
  host block reserved at load. Default `2 x N_DECODE_LANES` (16).
- **`--retained-device <n>`** (`IGNIS_RETAINED_DEVICE`): images in the device
  state arenas, as today. Default 0.

A card with VRAM to spare can run `--retained-device 16 --retained-host 0`
and keep today's device-to-device copies. Both may be nonzero; the device
slots are then handed out first.

`--retained-slots` / `IGNIS_RETAINED_SLOTS` are removed, and naming either
refuses the start with a message naming the two new flags.

## Acceptance criteria

1. The sequence pool takes a device and a host retained-slot count. Retained
   slot `r` is a device slot for `r < device` and a host slot above it; the
   device arenas (GDN, conv, penalty counts, drafter window, hq residual
   window) hold the lanes and the device slots only.
2. The host block is one `cudaHostAlloc` of `host_slots x stride` at pool
   create, `stride` the packed clone image (`ignis_seq_prefix_clone_layout`)
   aligned to `kIgnisSeqSectionAlign`. A failed allocation refuses the
   start. Nothing is allocated for it while serving.
3. A capture into a host slot and a claim from one copy the same sections a
   device clone copies, through the snapshot's own per-section packers, and
   synchronize. A claimant built from a host slot is byte-identical to one
   built from a device slot.
4. A host slot's image spills to KV-RAM as the same blob a device slot's
   would, and a spilled prefix or checkpoint comes back byte-identical.
5. `ignis_seq_pool_plan` and `ignis_seq_pool_stats` report the device slots'
   bytes on `retained_state_bytes` and the host block on its own field; the
   VRAM plan's `retained_slots` line is the device slots' alone and
   `hq_residual_window` counts the lanes and the device slots.
6. The KV pool's floor keeps one tail page per retained slot of either kind.
7. The scheduler sees one pool of `device + host` slots and takes the lowest
   free index, so a free device slot is always used before a host one.
8. `--retained-host` defaults to `2 x N_DECODE_LANES`, `--retained-device` to
   0; with `--prompt-reuse off` and neither named, both are 0. The flag wins
   over its environment variable; 0 is legal for either.
9. `--retained-slots` and `IGNIS_RETAINED_SLOTS` refuse the start, naming
   `--retained-device` and `--retained-host`.
10. The `ignis.runtime.vram_plan` event carries the host block's bytes and
    both slot counts; `ignis_retained_slots{state="capacity"}` stays the
    total the scheduler hands out.
11. The Makefile's `RETAINED_SLOTS` knob becomes `RETAINED_DEVICE` and
    `RETAINED_HOST`, and `make config` prints both.
12. ADR 0030, `CONTEXT.md` and `docs/user/README.md` describe the two kinds.
13. The prototype's `IGNIS_RETAINED_ON_HOST` switch is gone.

## Implementation Decisions

- **Kernel ABI.** `ignis_seq_pool_spec` keeps `retained_slot_count` as the
  device count and gains `retained_host_slot_count`; `ignis_seq_pool_stats`
  gains `retained_host_slot_count` and `retained_host_bytes`, and
  `ignis_seq_pool_plan` gains `retained_host_bytes` (the count is the spec's). Every move of retained state already goes through
  `ignis_seq_copy_slot_state` and `ignis_seq_write_materialized_blob`, so
  those two are the only functions that learn about the host block.
- **Rust.** `SeqPoolBudget` carries both counts; `CudaLeafConfig` splits
  `retained_slots` into `retained_device_slots` and `retained_host_slots`
  with the server's defaults; the scheduler's `retained_slots` is their sum.
- **Metrics.** The retained-state families' `tier="device"` keeps meaning
  "held in a retained slot", whichever kind: the scheduler's tiers are the
  slot tier and KV-RAM, and splitting the label is out of scope.
- **Synchronous copies stay.** The A/B shows no stall that matters at this
  rate; an asynchronous host copy is out of scope.

## Testing Decisions

- Kernel CTest `test_seq_prefix`: every check runs on a device-slot pool and
  again on a host-slot pool, plus a pool mixing both kinds (device slots
  first, host after), and a layout check that the device arenas of a host
  pool equal those of a pool with no retained slot.
- Rust: config resolution (defaults, env, flag precedence, reuse off, the
  removed flag), the VRAM plan's lines and floor, and a GPU test that a
  server-shaped load reports `retained_slots` 0 and the host block.
- A swarm A/B (`scripts/swarm-ab.sh`) of the defaults against
  `--retained-device 8 --retained-host 0` as the closing measurement.

## Out of Scope

- An asynchronous capture, and holding images in idle lanes: the measured
  PCIe cost does not call for either.
- Image-only spills that leave the KV pages on the device.
- Splitting the metrics' `tier` label by slot kind.
