# 01 — Evictions by residency tier (backend counter + Monitor panel)

ADRs: **0017** (the Prometheus contract), **0023** (one eviction priority
across levels), **0029** (cross-request retained state), **0030** (the
KV-RAM arena). Parent of the observability line: #216.

## What is wrong today

The Monitor shows one card, *KV evictions*, fed by
`ignis_kv_cache_evictions_total`. That series is **real and correctly
wired** (`SchedEvent::Evicted` → `Telemetry::on_evicted` →
`Metrics::record_eviction`; the #214 gate run recorded 65, and
`vision_mixed_load_gpu` asserts it moves). It reads zero on the owner's
playground load because eviction only runs on the admission-refusal path —
the pool never fills. Nothing about *that* number is to be changed: making
it move would be a scheduler-policy change, not a metrics change.

What the card cannot say is *which tier lost state*, and one real departure
is counted nowhere at all:

| Departure | Where in the code | Counted today |
|---|---|---|
| A live sequence snapshotted off the device into KV-RAM | `snapshot_and_evict` → `SchedEvent::Evicted` | `ignis_kv_cache_evictions_total` |
| Retained state that left the device for nowhere | `discard_checkpoint` / `spill_prefix`'s `!spilled` arm | `ignis_retained_state_discards_total{tier="device"}` |
| Retained state demoted device → KV-RAM | `write_prefix_blob` / the checkpoint move | `ignis_retained_state_spills_total{tier="kv_ram"}` |
| Retained state dropped out of KV-RAM | `forget_kv_ram_blob` | `ignis_retained_state_discards_total{tier="kv_ram"}` |
| **A live snapshot dropped out of KV-RAM to make room** | `make_host_room_for_bytes` → `KvRamVictim::Live` → `requeue_request` | **nothing** |

The last row is the gap. It is the most expensive departure in the system —
the request loses every prefilled token and re-prefills from zero — and it
is invisible. `SchedEvent::Requeued` is emitted there, but requeue has a
second, unrelated cause (a restore that failed, `concrete.rs` ~2228), so
counting `Requeued` would conflate two facts.

There is no disk tier. `ReuseSource` is `Device | KvRam`.

## Departures from the obvious design

- **No `tier` label is added to `ignis_kv_cache_evictions_total`.** It is a
  stable contract row with no labels; relabelling it silently changes an
  existing series' identity for every scraper. The new fact gets its own
  series instead.
- **No `disk` label value is exported.** Adding `ReuseSource::Disk` would
  widen every per-tier array, add a `disk` wire spelling to the request log
  and put a permanently-zero label on five families, for a tier that does
  not exist. Disk is prepared **in the UI only** — a visible, explicitly
  inactive row — and reserved by name in ADR 0017. The backend variant is a
  follow-up if and when a disk tier is built.
- **Spills are shown as a departure from the device.** A prefix or
  checkpoint spilled into KV-RAM *did* lose device residency (ADR 0023's
  sense of eviction), even though the state survives. The panel shows it on
  the VRAM row, labelled as a demotion, never summed into a loss figure.

## The change

### Backend

1. New scheduler fact `SchedEvent::SnapshotDropped { request }`, emitted at
   the `KvRamVictim::Live` arm of `make_host_room_for_bytes`, alongside the
   `Requeued` that already fires there. Named for what happened, not for
   what the request does next.
2. `Telemetry::on_snapshot_dropped` bumps a new counter and is the only
   writer of it.
3. New contract row: `ignis_kv_ram_evictions_total`, counter, no labels —
   *"Live host-tier snapshots dropped from KV-RAM to make room; the request
   re-prefills from the start."*
4. ADR 0017's table gains that row, plus a paragraph naming the five
   departures above and reserving `disk` as unexported.

### UI (Monitor)

The *KV evictions* card becomes **Evictions**, a per-tier table:

| Tier | Live | Retained | Demoted |
|---|---|---|---|
| VRAM | `kv_cache_evictions` | `discards{device}` | `spills{kv_ram}` → RAM |
| RAM | `kv_ram_evictions` | `discards{kv_ram}` | — |
| Disk | — | — | — (not implemented) |

Each cell keeps the existing window / per-minute / since-start treatment.
The chart carries one series per tier for the *live* column (the figure
that costs a request its prefill). The Disk row renders greyed with a
"non implementato" marker and is never fed by a metric.

`assessHealth` weighs a RAM live drop **above** a VRAM eviction: a VRAM
eviction preserves the request's work, a RAM drop destroys it.

## Acceptance criteria

- **AC1** — A CPU scheduler test drives the host tier over budget so that a
  live snapshot is dropped, and asserts `ignis_kv_ram_evictions_total`
  moved by exactly the number of drops, while
  `ignis_kv_cache_evictions_total` counts only the snapshots that went *to*
  the tier.
- **AC2** — A requeue caused by a **failed restore** does not bump
  `ignis_kv_ram_evictions_total` (the two `Requeued` causes stay apart).
- **AC3** — `/metrics` declares `ignis_kv_ram_evictions_total` with `# TYPE
  counter`, present and zero on a load that evicted nothing.
- **AC4** — The Monitor renders three tier rows from one fixture scrape,
  with the VRAM/RAM figures taken from the fixture and the Disk row inert.
- **AC5** — `assessHealth` reports a RAM live drop at a severity at least
  that of an equal number of VRAM evictions.
- **AC6** — `cargo test` green workspace-wide; `npm test` green in `web/`.
