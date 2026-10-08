# KV-disk contention on the volume

- Kind: experiment
- Status: current
- Observed: 2026-10-08
- Last verified: 2026-10-08
- Scope: runtime / KV-disk tier, Flash-Next n-gram reads, decode ITL
- Related: GitHub #309, spec `vram-budget/03` AC 25, ADR 0045, ADR 0017
- Superseded by: none

## Question

KV-disk (Tier 2) writes a live sequence's state to the volume Flash-Next
reads its n-gram rows from, and copies it off the device while other lanes
decode. What does a ~1 GB spill cost a prefill chunk against an uncovered
n-gram table, and what does it cost two decoding lanes? The spec's starting
bounds: chunk wall time and ITL p50 each within +10% of the same work without
the spill.

## Evidence

RTX 5090 (PCIe Gen 3 x16 on this host), the Flash-Next artifact and the
tier's files both on F: (NVMe). Branch `kv-p2-disk` with #309 P1's pool
policy (runs 2-3 on a local merge of P1 at `2c09568`, runs 4-5 after main's
merge of it, `cf72ec4`): the pool cut to one 262,144-token context with
`--kv-pool-bytes`'s token form, which is what lets a spill be forced on
Flash-Next. 4 decode lanes, 8,192-token prefill chunks, KV-RAM off
(the spill goes device -> disk), prompt reuse off, `--ngram-hot-bytes 0`
(every row a gather stages is read from the artifact). A is an `agent`
request of 210,000 prompt tokens: ~1.02 GB of state (a 130 MB image and
4,224 bytes a token).

**Decode** (`crates/server/tests/kv_disk_contention_gpu.rs`). B and C
(`interactive`, 2,000-token prompts) decode at width 2 alone; then again while
A is written to the disk to make room for E. The spill window is E's arrival
to the step A's file committed, less the step that also ran E's first chunk.

| run | ITL p50 alone | ITL p50 beside the spill | ITL p99 alone / beside | spill window |
|---|---|---|---|---|
| 3 | 12.16 ms | 12.85 ms (+5.7%) | 22.50 / 17.53 ms | 626 ms |
| 4 | 12.21 ms | 13.20 ms (+8.1%) | 22.83 / 17.24 ms | 636 ms |
| 5 | 12.18 ms | 12.97 ms (+6.5%) | 22.71 / 17.28 ms | 629 ms |

398 gaps alone and 94 beside the spill in each run. Run 2 measured +2.9% on
p50 (12.58 against 12.22 ms); its p99 counted E's first prefill chunk as a
decode gap, which the later runs' window leaves out.

**Prefill** (same test). F (16,384 tokens, two chunks) prefilled alone took
2,790-2,860 ms a chunk with 1,092-1,116 ms of n-gram gather (runs 3-5); with
G arriving to move A2 (another ~1 GB `agent`) it took 2,819-2,863 ms with
1,107-1,130 ms of gather. In every run neither chunk ran with the spill in
flight: G, the request that needs the room, waits for the prefill F holds,
and the spill starts when F's prefill is done.

**The volume** (`crates/runtime/tests/kv_disk_volume_contention.rs`, no GPU,
runs 1-2). The real n-gram table with no hot row, six 8,192-token gathers,
each followed by 1,690 ms idle (the card's chunk time past its gather), and
the tier's IO threads writing a 1 GiB blob in 32 MiB unbuffered windows
queued as the first gather starts.

| | gather 0 | gathers' median | blob written in |
|---|---|---|---|
| alone (run 1 / run 2) | 1,120 / 1,166 ms | 1,152 / 1,204 ms | -- |
| blob alone on the volume | -- | -- | 649 / 610 ms (1.54 / 1.64 GiB/s) |
| behind the prefill gate | 1,159 / 1,196 ms | 1,191 / 1,213 ms | 1,949 ms (run 2) |
| without the gate | 1,576 / 1,422 ms | 1,161 / 1,158 ms | 675 ms (run 2) |

Raw samples: the main checkout's `.scratch/kv-p2/ac25-contention-run{2,3,4,5}.json`
and `ac25-volume-run{1,2}.json` with their logs (run 1's `write_ms` timed the
whole loop, not the blob; run 2's is the blob's).

## Finding

- **Both bounds hold.** Two lanes decoding beside a 1 GB device -> disk
  spill keep ITL p50 within +3% to +8% over four runs, and their p99 does
  not move up; a prefill chunk's wall time and gather are unchanged (+0.5-
  0.7%, inside run-to-run spread).
- **A spill does not run beside a prefill chunk.** The scheduler starts a
  live spill for the prefill head that needs the room, and that head waits
  for the one prefill in flight, so on the card the two never overlapped.
  What a chunk can share with the tier is the volume, through a write that
  is already in flight when its gather starts.
- **The prefill gate is what keeps the volume's share to noise.** Behind it
  a gather beside the tier's writes reads in its alone time (1,159 / 1,196
  ms against 1,120-1,212 alone), at the price of the blob waiting out the
  gather (1.95 s instead of 0.61 s). Without it the same 1 GiB of writes
  slows the gather it overlaps by +18% to +37%.
- **A 1 GB spill takes ~0.63 s, about the volume's write speed** (1.54-1.64
  GiB/s alone): the one-window-per-advance pacing does not limit it while
  lanes decode at ~12 ms a round.

## Implications

- The bounds can be confirmed as written for the move-out case; nothing in
  this data argues for a different window or staging size.
- One window per advance means a spill's pace follows the model's step:
  ~12 ms rounds move 1 GB in well under a second, but a step held by a
  multi-second prefill chunk would move only a window or two. The scheduler
  never starts a live spill there today; a design that did (a demotion
  started during a chunk, or more than one prefill in flight) would see its
  transfers slow, not its chunks.
- Turning the gate off would trade a faster spill for a slower prompt: the
  current order (the n-gram table first) is the right default for a volume
  shared with the model.

## Limits and unknowns

- One host, one NVMe, two runs per figure; the decode spill window holds 94
  gaps, so its p99 is close to its maximum.
- The restore direction (move in, disk -> device) was not measured here;
  its PCIe contention bound belongs to the live-move AC of P3.
- The volume test stands in for the chunk's compute with idle time: what a
  real chunk's PCIe traffic (expert streaming) adds to the IO is not in it.
- With KV-RAM on, a spill can also go KV-RAM -> disk (no PCIe); its writes
  are the same windows behind the same gate.

## Follow-ups

- Measure move-in (a restore from the disk beside decoding lanes) when the
  live-move work of #309 P3 lands.
