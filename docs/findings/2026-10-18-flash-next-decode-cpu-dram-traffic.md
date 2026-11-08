# Flash-Next decode and CPU DRAM traffic on the 5090 host

- Kind: experiment
- Status: current
- Observed: 2026-10-18
- Last verified: 2026-10-18
- Scope: serving / Flash-Next decode on the RTX 5090 host (i9-10900K, Z490, PCIe Gen 3 x16): the host-side copies of a decode round, the n-gram gather and its file reads, the expert miss copies from the pinned host pool under CPU memory load
- Topic: flash-next
- Related: GitHub #329 item 13; the swap experiment's follow-up (`.scratch/residency-swap-5090/REPORT.md` in the main checkout: CPU DRAM traffic costs a copy-engine DMA up to 57% on this host); [Flash-Next decode headroom](2026-10-08-flash-next-decode-headroom.md); [KV-disk contention on the volume](2026-10-08-kv-disk-contention-on-the-volume.md); [Live moves a window at a time](2026-10-08-live-moves-windowed.md)
- Superseded by: none
- Summary: No host copy of a Flash-Next decode round competes with the expert-miss DMA, but other DRAM traffic on the Gen 3 host does (one memcpy thread +6/+15% ITL at 1/3 lanes, eight +46/+92%), so builds belong off the host while it decodes.

## Question

The swap experiment found that CPU DRAM traffic on this host slows the
GPU's DMA reads of pinned memory, up to 57% for a copy engine. Flash-Next
decode reads its expert misses from a 35.2 GiB pinned host pool. Which host
memory copies run in the decode path, how many bytes per round, and do they
compete with the expert miss DMA? How much does CPU memory traffic cost a
served decode round?

Every copy-bound number below is on this host's PCIe Gen 3 x16 link
(~12 GB/s).

## Evidence

**Setup.** Worktree `dram-traffic-5090` at `ea3a15b1`, release server built
there. Flash-Next artifact v2, greedy, MTP off (the 5090 default), 3 decode
lanes, context 16,384, `--vram-budget-bytes 26G` (expert cache 18.46 GB,
expected hit rate 0.606), `--ngram-hot-bytes 512M` (5,711,392 hot rows, not
the 1 GiB default), prompt reuse off, KV-RAM arena 0, KV-disk 0. The owner's
idle ignis-server (2.8 GB of VRAM, model unloaded) stayed on the card. The
host plan saw 43.2 GB available and planned 38.3 GB. GPU lock held
08:21-08:32Z.

Each cell is 1 or 3 concurrent streams of 384 tokens (`ignore_eos`) on the
same prompts, with `/metrics` scraped before and after. Every cell's text
hashes the same per lane (`sha` in the raw rows), so cells are paired
token for token. CPU load comes from `hog.exe`, started 3 s before the
requests:
- `copy N`: memcpy between two 64 MiB buffers per thread;
- `read 8`: sums a 64 MiB buffer per thread;
- `l3 8`: memcpy inside 256 KiB per thread, cache-resident. This is the
  control for "8 cores busy, no DRAM".

The paced cells (`p2`, `p5`) run 8 copy threads held to 2 or 5 GB/s in
total. Hog GB/s counts bytes copied or read once. A copy moves about 2x that
in DRAM traffic (read plus write; inferred, not counted).
Each hog cell ran twice, in reversed order, and each quiet cell four times,
twice as bookends. Calibrated alone, `copy 8` ran 17.8 GB/s, `copy 1` 16.5,
`read 8` 39.1 and `l3 8` 254.

**Observed: decode under CPU memory load** (`summary.md`; ms per round; at 1
lane a round is a token):

| lanes | cell | hog GB/s during | ITL p50 ms | vs quiet | demand stall ms/round | rest of round ms | stall µs per miss |
|---|---|---|---|---|---|---|---|
| 1 | quiet | 0 | 8.94 | | 2.12 | 6.82 | 58.0 |
| 1 | l3 8 | 221 (cache) | 9.00 | +0.7% | 2.14 | 6.86 | 58.1 |
| 1 | p2 | 2.0 | 9.07 | +1.5% | 2.21 | 6.86 | 59.9 |
| 1 | p5 | 5.0 | 9.20 | +2.9% | 2.30 | 6.90 | 62.2 |
| 1 | copy 1 | 13.5 | 9.46 | +5.9% | 2.53 | 6.93 | 68.6 |
| 1 | read 8 | 33.2 | 10.38 | +16.2% | 2.98 | 7.41 | 80.8 |
| 1 | copy 8 | 15.7 | 13.06 | +46.2% | 4.33 | 8.73 | 117.4 |
| 3 | quiet | 0 | 21.01 | | 9.09 | 11.93 | 58.6 |
| 3 | l3 8 | 226 (cache) | 20.97 | -0.2% | 9.11 | 11.87 | 58.5 |
| 3 | p2 | 2.0 | 21.42 | +1.9% | 9.35 | 12.08 | 60.1 |
| 3 | p5 | 5.0 | 22.43 | +6.7% | 9.88 | 12.54 | 63.4 |
| 3 | copy 1 | 12.8 | 24.18 | +15.1% | 10.71 | 13.47 | 68.8 |
| 3 | read 8 | 30.3 | 28.50 | +35.6% | 12.71 | 15.79 | 81.5 |
| 3 | copy 8 | 15.1 | 40.36 | +92.0% | 18.50 | 21.85 | 118.5 |

Expert traffic per round did not change with the load. At 1 lane it was
53.9-54.4 MB and 36.6-36.9 misses: 6.0 GB/s of DMA reads from the pinned
pool. At 3 lanes it was 222.9-223.6 MB and 155.1-156.1 misses: 10.6 GB/s.
Quiet ITL p50 spread over the bookends: 8.81-8.99 ms at 1 lane, 20.85-21.16
at 3. The "demand stall" is `ignis_expert_residency_stall_seconds_total{phase="decode"}`:
the device-timed span of each layer's demand copy. It leaves out waits on
prefetch copies.

**Observed: the host-side copies of one decode round** (code at `ea3a15b1`;
bytes per lane per round, measured at 512M hot rows):

| # | where | what | bytes | overlaps the miss DMA? |
|---|---|---|---|---|
| 1 | `crates/core/src/ngram_table.rs:376-379` (submitted at `:661-664`) | an unbuffered read of each n-gram row the hot cache lacks: the NVMe DMA-writes its 4 KiB sector(s) into a fresh aligned buffer | 36,404 B (8.72 of 16 rows from the file) | no |
| 2 | `crates/core/src/ngram.rs:409` (called at `ngram_table.rs:871`) | the gather: 16 rows of 90 B from `hot_data` or the read buffers into the round's `rows` Vec (`crates/runtime/src/flash_next_leaf.rs:674-682`) | 1,440 | no |
| 3 | `kernel/src/flash_next/program.cu:1112-1124` | the round's inputs memcpy'd into the pinned `round_inputs`: the n-gram rows, sampling configs, ids, slots, positions | ~1.5 KB | no |
| 4 | `program.cu:1128-1148` | copy-engine H2D of #3 | ~1.5 KB | no: queued ahead of the graph on the same stream |
| 5 | `program.cu:1177-1182` | D2H of the drawn ids and probabilities, then the round's synchronize | 8 | after the round |
| 6 | `kernel/src/residency.cu:468-475` | the device writes residency's totals into a mapped host page at the last layer | ≤ 336 B a round | yes, but 336 B |

Copies #1-#4 run on the model thread before the graph launch
(`program.cu:1161`). The previous round's synchronize (`program.cu:1182`)
has already passed, and the captured prefetch stream joins inside the graph.
So no expert copy is in flight while they run. All six come to about 40 KB
per lane per round, 90% of it NVMe DMA. That is 1/1,350 of the 53.9 MB of
expert reads per round at 1 lane, and ~4.5 MB/s at 112 rounds/s. The
lowest load measured (`p2`, ~4 GB/s of DRAM traffic) is ~900x that.

**Observed: the n-gram file path.** A second load with `--ngram-hot-bytes 0`
reads all 16 rows a lane-token from the file: 66,851 B against 36,404 at
512M. Three quiet cells per width:
- ITL p50 9.02 ms at 1 lane (8.89-9.11), against 8.94 at 512M;
- 21.25 ms at 3 lanes (21.21-21.26), against 21.01: +0.9% and +1.1%;
- the stall per miss is unchanged (58.1 / 58.6 µs).

The prefill-only deltas the decode numbers subtract are in
`prefill-only.jsonl`.

## Finding

1. **No host copy in Flash-Next's decode round competes with the expert
   miss DMA (observed in the code, sized by the counters).** The round's
   host work runs while no copy is in flight: the n-gram file reads, the
   gather, the pinned staging and its H2D. That work moves ~40 KB per lane,
   three orders of magnitude below any load that registers. The expert path
   has no CPU copy at all: the GPU reads the pinned pool directly. The
   candidates the item named do not reach the decode round: KV-RAM moves,
   n-gram staging from hot rows or from the artifact (unbuffered, so no
   page cache), per-step pool copies. A prefill's gather is serialized on
   the same model thread too: `rows_for` (`flash_next_leaf.rs:610`)
   gathers the whole span before any chunk launches. So it adds to its own
   TTFT ([finding 2026-10-07](2026-10-07-flash-next-prefill-chunk-decomposition.md)) but never overlaps another lane's decode round.
   Only the driver's staging of each chunk's 11.8 MB of pageable rows
   (`program.cu:829`) can run beside the previous chunk's compute.
2. **The exposure is other DRAM traffic on the host, and it is large
   (observed).**
   - A paced 2 GB/s memcpy costs +1.5% / +1.9% ITL at 1 / 3 lanes, at the
     edge of the noise. 5 GB/s costs +2.9% / +6.7%.
   - One unpaced memcpy thread costs +5.9% / +15.1%. Eight cost +46% / +92%:
     tok/s 104.5 -> 68.6 at 1 lane, 46.3 -> 23.6 per lane at 3.
   - Each demand copy takes twice as long (58 -> 117-119 µs a miss). The
     rest of the round grows by about as much again. That is likely the
     prefetch copies reading the same pool (not separated from the host gap).
   - CPU load alone costs nothing: 8 threads in cache, +0.7% / -0.2%.
3. **How the CPU touches DRAM matters more than its bytes (observed;
   mechanism inferred).** One copy thread and eight move similar bytes
   (12.8-13.5 against 15.1-15.7 GB/s copied). The eight cost 6-8x more.
   Eight readers at 30-33 GB/s, all reads, cost about a third of what eight
   copiers cost. Those copiers make about the same DRAM traffic (~30 GB/s,
   inferred), half of it writes. Inference, not diagnosed (no uncore
   counters): the cost follows how many CPU streams contend in the memory
   controller with the DMA's reads, and how many of them write. GB/s alone
   does not predict it.
4. **The n-gram hot rows are a latency lever, not a DRAM one, and a small
   one (observed, with one inference).** Moving 7.3 more rows per lane-token
   to the file cost +0.9% / +1.1% ITL (one load each, near the spread).
   Inference: the whole table in RAM would remove the remaining 8.7 file
   rows and save about as much, ≤ ~1%. It needs 28.8 GB, which this 64 GB
   host cannot hold beside the 35.2 GiB pinned pool.

## Implications

Ranked proposals, with the expected gain:

1. **Keep memory-heavy, many-threaded CPU jobs off the host while
   Flash-Next serves or is measured.** These are builds (cargo, nvcc,
   podman), large copies and capture exports. CPU priority does not help:
   the cost is DRAM, not cores. Expected gain: the tok/s the `copy 1` to
   `copy 8` doses took, -7% to -34% at 1 lane and -15% to -49% per lane at 3.
   - Builds are the likely realistic source on this machine. Their DRAM rate
     was not measured; inferred to fall between `p5` and `copy 8`.
   - For measurements it is a validity rule: a Flash-Next GPU run beside a
     build is contaminated. The GPU lock does not catch this, because a
     build takes no lock.
2. **Leave the decode round's own copies alone.** Moving the n-gram gather
   to the GPU, keeping hot rows device-side, or batching or overlapping the
   staging gives an expected DRAM gain of ~0. Bound: the `p2` dose, at ~900x
   these bytes, costs ≤ 2%. Device-side hot rows would also take 0.5-1 GB
   from the 18.5 GB expert cache. The hit rate pays for that: a net loss.
3. **KV-disk beside decode (inferred).**
   - A spill from the device moves about 3 bytes of DRAM traffic per byte
     spilled: the D2H DMA write into staging, the IO thread's CRC32 over
     each 32 MiB window (`crates/runtime/src/kv_disk.rs:661`), and the NVMe
     DMA read. An unaligned KV-RAM window adds an IO-thread memcpy
     (`kv_disk.rs:655-657`).
   - At the volume's 1.5-1.6 GiB/s that is ~5 GB/s of DRAM traffic. That
     falls between the `p2` and `p5` doses (~4 and ~10 GB/s of DRAM
     traffic): ≈ +2-3% at 1 lane and +2-7% at 3, for the spill's
     ~0.6-0.7 s. Its few streams put it toward the low end. The +3-8% ITL
     measured on 2026-10-08 also holds the D2H's share of the link.
   - A GPU-side CRC would remove a third of that traffic: ≤ ~1-2%, during
     spills only. Not worth building.
4. **KV-RAM moves (inferred).** These are GPU DMA only, 12 MiB out / 16
   MiB in per advance (`crates/runtime/src/lib.rs:453,462`): ~1 GB/s of DRAM
   traffic, below `p2`. Their measured cost is the shared link (the swap
   experiment's mechanism), not DRAM. Nothing to change on DRAM grounds.
5. **The whole n-gram table in RAM: ≤ ~1%** (finding 4). It does not fit
   here. Where RAM allows, `--ngram-hot-bytes auto` already takes it.
6. **Micro, not DRAM.** Gather straight into the pinned `round_inputs`
   instead of the `rows` Vec plus a second memcpy. That saves one 1,440 B
   copy and one allocation per lane-round: microseconds of host gap. Not
   worth a ticket.

## Limits and unknowns

- The quiet cells assume no other job loaded the host's memory. By the
  coordinator's schedule no other agent built during them (a read-only
  reviewer ran); the CPU load itself was not recorded.

- One host (i9-10900K, a dual-channel platform, Gen 3 x16), one driver,
  one load shape. The 26G budget, context 16,384 and 512M hot rows are not
  the make defaults (262,144 x 3 lanes, 1 GiB hot rows). The default's
  expert cache and hit rate differ, so its rounds move other byte counts.
  The stall per miss is the per-byte measure that carries over.
- Three prompts, 384 tokens each. Two runs per load cell, four per quiet
  cell. The `p2` effect is at the edge of the quiet spread (±1-2%).
- The hog uses the CRT's memcpy. Whether it uses non-temporal stores was
  not checked. DRAM bytes for `copy` are inferred as 2x the bytes copied.
- The rest-of-round growth under load is not split into prefetch-copy
  waits and host gap (no nsys run).
- The 3-lane per-round columns assume every round ran all 3 lanes. They
  divide by decoded tokens / 3. A few early rounds ran narrower while the
  other prompts prefilled: lane 0's tok/s is 41-42 against 46. So the
  3-lane stall and bytes per round are slightly overstated. ITL p50 is
  unaffected.
- No DRAM or uncore counters: Intel PCM is not installed. Finding 3's
  mechanism is a hypothesis.
- The hot-0 cells come from a second load: load-to-load variance is
  included.
- Gen 4/5 hosts: the DMA would read the pool at 2-4x this rate, a larger
  share of the same DRAM. Contention could cost more there, or less on a
  platform with more memory channels. Not measured.

## Follow-ups

- A rule for agents (`docs/agents/testing.md`): no build or other
  DRAM-heavy job beside a Flash-Next GPU measurement. Possibly also a
  serving note for the owner's host.
- Raw data in the worktree's `.scratch/dram-traffic/`:
  - `hog/hog.rs` (the load generator), `hog-calibration-1.txt`;
  - `measure.py`, `prefill_only.py`, `analyze.py`;
  - `cells-512M.jsonl` / `.txt`, `cells-hot0.jsonl` / `.txt`, `warmup-*.jsonl`,
    `prefill-only.jsonl`, `summary.md`;
  - `flags-512M.txt`, `flags-0.txt`, `server-512M.log`, `server-hot0.log`;
  - `gpu-lock-acquired.txt`, `gpu-lock-released.txt`.
