# Retained slots on the host: 42% more KV at level throughput for an 8-agent swarm

- Kind: experiment
- Status: current
- Observed: 2026-09-28
- Last verified: 2026-09-29
- Scope: kernel / sequence pool retained slots; scheduler prompt reuse; VRAM plan
- Related: [GitHub #281](https://github.com/gpillon/ignis/issues/281),
  [ADR 0030](../adr/0030-device-memory-reserved-at-load.md),
  [spec vram-budget/02](../specs/vram-budget/02-retained-host.md),
  [Device prefix clone cost](2026-09-12-device-prefix-clone-cost.md),
  [Sequence snapshot transfer cost](2026-09-12-sequence-snapshot-transfer-cost.md)
- Superseded by: none

## Question

The default load held eight retained slots in VRAM: 1.47 GiB of slot state
plus 272 MiB of their hq residual window, about 200K tokens of hq KV. Do the
device slots earn that VRAM for a many-agent load? And what does moving the
images to pinned host memory cost when every capture and every claim then
crosses PCIe?

## Evidence

**Load.** `scripts/agent-swarm.py` runs a synthetic swarm, deterministic for a
seed. Each agent is a multi-turn conversation over one shared ~13K-token
system head. Each turn appends a tool result cut from the repository and asks
for a short answer (greedy, at most 192 tokens). Between turns the agent
waits a simulated tool time of 0.5–2 s, never a human pause.

**Driver and host.** `scripts/swarm-ab.sh` starts the server once per leg with
that leg's make knobs, scrapes `/metrics` around the swarm, and reports the
legs side by side. The host is an RTX 5090 on a PCIe Gen 3 x16 link, with
262K hq-e8-2b, DFlash2 with a 7-token draft, and 8 GiB of KV-RAM.

**Two loads, both 8 agents x 8 turns.**
- *Short:* tool results of ~1.5K tokens; contexts reach ~16–25K tokens.
- *Long:* tool results of ~4.5K tokens; contexts reach ~57K tokens.

**1. Device slots only** (2026-09-28, main a4aaa92):

| slots (KV tokens) | short: wall, tok/s, later-turn TTFT p50/p95 | long: wall, tok/s | checkpoint hits device / KV-RAM | notes |
|---|---|---|---|---|
| 2 (628K) | 96.2 s, 112, 3.61/8.00 s | — | 0 / 0 | 53 publishes and 11 captures skipped; reuse 63.8% |
| 8 (471K) | 51.9 s, 200, 1.67/3.12 s | 93.5 s, 117 | 1 / 51 | 53 spills, 46 KV-RAM discards; long: 12 captures skipped |
| 16 (275K) | 51.2 s, 215, 1.51/2.75 s | 70.0 s, 148 | 20 / 36 | long: KV at 99%, 2 evictions |

**2. Host slots.** Images live in one pinned host block (branch
`retained-slots-ab`). The kernel CTest `test_seq_prefix` measures a claim
from a host slot at 15.6 ms for 182 MiB (12.2 GB/s), against 0.29 ms device to
device. A full serving-shape image is 232,532,224 B.

Same binary for every leg (prototype build, 2026-09-29):

| slots | KV tokens | short: wall, tok/s | long: wall, tok/s, TTFT p50/p95 |
|---|---|---|---|
| device 8 | 475K | 53.5 s, 194 | 79.7 s, 136, 2.26/5.89 s |
| host 8 | 678K | 55.0 s, 190 | 85.2 s, 128, 2.85/8.06 s |
| host 16 | 679K | 52.7 s, 211 | 68.8 s, 147, 2.27/5.26 s |
| host 24 | 682K | — | 69.6 s, 147, 2.03/6.13 s |

**3. The feature as shipped** (commit 3c49073), same binary, defaults
(`--retained-host 16`) against `--retained-device 8 --retained-host 0`:

| | KV tokens | short: wall, tok/s, TTFT p50/p95 | long: wall, tok/s, TTFT p50/p95 |
|---|---|---|---|
| device 8 | 475K | 52.8 s, 192, 1.31/3.75 s | 71.4 s, 139, 2.21/4.62 s |
| host 16 | 677K | 52.1 s, 200, 1.51/3.30 s | 70.5 s, 145, 2.82/5.41 s |

Raw runs are in `.scratch/retained-slots-281/` of the main checkout (local,
untracked): `swarm-ab/` holds `leg-8`, `legs-2-16`, `long-8-16`, `proto-*`
and `final-*`, each leg with its server log, both `/metrics` scrapes and
`requests.jsonl`, and `decode-probe/` the single-stream check below.

## Finding

**Observed.**

- **With eight agents, eight device slots do almost none of the reuse.**
  - 51 of 52 checkpoint hits came back from KV-RAM, not from a slot.
  - Eight agents need about seventeen slots: a chain link and a checkpoint
    each, plus the shared head.
  - A conversation's image is therefore spilled before its next turn.
- **Two device slots do not move reuse to RAM; they end it.**
  - The shared head's publish pins the slots.
  - Captures and publishes are skipped, and KV-RAM is fed only by spilling a
    slot.
  - Conversation reuse drops to zero and the run takes 1.85x as long.
- **More slots help more than more KV at this load.** 16 device slots with a
  278K pool beat 8 slots with a 478K pool on the long run.
- **Host slots give the pool 42–44% more KV** (475K to 677–682K tokens),
  because the retained images and their share of the residual window leave
  VRAM.
- **Throughput is level.** On the shipped build, sixteen host slots matched
  eight device slots on wall time (−1%) and tok/s (+4%) on both loads.
- **Later-turn TTFT p50 rose** on the long load, from 2.21 to 2.82 s.

**Inferred.**

- A host claim or capture adds tens of milliseconds per request, and the
  TTFT medians move by more than that. Queueing behind other agents' prefill
  chunks dominates later-turn TTFT here, and it varies between runs.
- The same eight-device-slot leg on the long load took 93.5, 79.7 and 71.4 s
  on three runs. Differences below about 10% on that load are within noise,
  including the prototype's "host 16 is 14% faster".

**4. Single-stream decode is unchanged** (2026-09-29). The same greedy prompt
was sent three times, one request at a time, to each build. Every build ran
~16 ms per speculative round:

| build | prose, tok/s | predictable code, tok/s |
|---|---|---|
| host 16 | 167 | 491 |
| device 8, same build | 167 | 489 |
| the build before (`locate-long-context`) | 166 | 486 |

## Implications

- The default is sixteen host slots and no device slots; a card with VRAM to
  spare can take `--retained-device 16 --retained-host 0`.
- A capture needs a slot to land in. Too few slots of any kind skip reuse
  rather than shifting it to KV-RAM, so the slot count must cover about two
  per concurrent conversation plus the shared heads.
- The KV these loads freed was not needed by them (peak KV use 45–64%). Its
  value shows on loads whose contexts outgrow a 475K pool.

## Limits and unknowns

- One run per leg; the long load's run-to-run spread is ~10–25%.
- The load is synthetic: no thinking, bounded replies, simulated tool times.
- Synchronous PCIe copies were not separable from prefill-chunk interleaving
  in the client's inter-token gaps; no stall attributable to them was seen.
- PCIe Gen 3 only; a Gen 5 host would cut each copy to ~5 ms.

## Follow-ups

- A long-context swarm whose total context exceeds 475K tokens, to measure
  what the freed KV buys.
- An asynchronous capture, only if a later load shows the synchronous copies
  stalling other lanes.
