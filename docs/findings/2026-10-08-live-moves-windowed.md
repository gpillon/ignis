# Live moves a window at a time: the copies cost the other lanes almost nothing; what a move in's rounds pay is expert-cache misses

- Kind: experiment
- Status: current
- Observed: 2026-10-08
- Last verified: 2026-10-08
- Scope: serving / live moves (ADR 0045, fixed branch), KV-RAM and KV-disk transfers, the leaf's whole-blob snapshot, Flash-Next's expert residency during and after a move
- Related: https://github.com/gpillon/ignis/issues/309, [ADR 0045](../adr/0045-the-kv-pool-follows-residency-and-state-goes-down-to-disk.md), spec [vram-budget/03](../specs/vram-budget/03-kv-pool-policy-and-kv-disk.md) (AC 37), supersedes [live move PCIe contention](2026-10-08-live-move-pcie-contention.md)
- Superseded by: none

## Question

AC 37's first measurement ([live move PCIe contention](2026-10-08-live-move-pcie-contention.md))
left three questions. Why did a synchronous KV-RAM move run at 3.5 GB/s, a
third of the link? Once KV-RAM moves go a window at a time between steps, as
KV-disk's do, at what pace do the other lanes' ITL stay within the starting
bounds (move out p50 +10%, move in +25%, max +150 ms)? And what made the disk
move in cost +50% p50 at 1.68 GB/s, about 15% of the link, too little traffic
for bandwidth to explain it?

## Evidence

RTX 5090 on PCIe Gen 3 x16 (nvidia-smi: gen 3, width 16), Windows (WDDM),
Flash-Next, branch `ac37-moves`. Raw logs, JSON samples, per-step timelines
and GPU samples are in the main checkout's `.scratch/ac37/`.

**1. The whole-blob copy.** `test_seq_flash_next_sections`'s cost arm: one
236,000-token sequence at Flash-Next's geometry (1,127,014,656 B, hq-e8-2b),
pinned host memory, a quiet card, three repetitions each. Before is the same
test built against main's `kernel/src/seq.cu`.

| copy | before | after |
|---|---|---|
| whole snapshot, device to host | 324.7 ms, 3.47 GB/s | 94.1 ms, 11.98 GB/s |
| whole restore, host to device | 333.6 ms, 3.38 GB/s | 87.2 ms, 12.93 GB/s |
| windowed, one window / 16 MiB windows, out | 94.7 / 94.9 ms | 93.6 / 95.0 ms |
| windowed, one window / 16 MiB windows, in | 86.0 / 85.6 ms | 86.8 / 86.5 ms |
| the block keys a page at a time (44,256 copies of 4 KiB) | 234.4 ms (5.3 µs each) | 248.6 ms (5.6 µs each) |
| one `cudaMemcpy` of as many bytes, out / in | 97.5 / 84.9 ms | 98.7 / 85.1 ms |

**2. AC 37, windowed.** `live_move_contention_gpu.rs`, the scenario of the
first finding: C (`agent`, 236,000-token prompt, a 1,128,096,256-byte blob)
moves out when E (`interactive`, 18,000 tokens) arrives, and back in when E
ends, while B1 and B2 (`interactive`, 2,000-token prompts) decode. The
baseline is now one for all four moves: two fresh lanes alone at width 2,
after everything else ended (the card's state after C's long prefill), as
many steady steps as the move's. Only the move's steps where both B lanes
decoded count. The pace is a window in flight each way, at most a new one an
advance; the windows run beside the rounds. One run a row unless noted:

| tier | move | pace | duration | steps | ITL p50 vs baseline | max vs baseline |
|---|---|---|---|---|---|---|
| KV-RAM | out | 64 MiB | 273 ms | 17 | +37.2% | +3.1 ms |
| KV-RAM | out | 16 MiB | 855 ms | 68 | +11.1% | +2.7 ms |
| KV-RAM | out | 12 MiB (5 runs) | 1,137-1,182 ms | 90 | +3.4, +5.6, +5.7, +6.3, +13.1% | +3.1 to +4.2 ms |
| KV-RAM | in | 16 MiB | 1,116-1,288 ms | 66-68 | +45.4, +55.5% | +7.5, +11.4 ms |
| KV-RAM | in | 4 MiB | 4,680 ms | 269 | +49.6% | +7.8 ms |
| disk | out | its 32 MiB file windows | 650 ms (1.73 GB/s) | 47 | +12.1% | +2.6 ms |
| disk | in | 16 MiB feeds | 1,285 ms | 70 | +58.1% | +10.7 ms |

The synchronous KV-RAM moves they replace: +313.5 and +318.4 ms max.

Four more KV-RAM legs ran with a move in's window landing *before* the round
(the model thread waiting 1.34 ms a window) instead of beside it: move in p50
+64.9 to +68.5%. That mode is not kept (below).

**3. What a step's time is made of.** The harness records, after every step,
the load's expert residency counters (`reserved.flash_next`) and the model
thread's time in the transfer passes (`transfer_pass_micros`): per phase,
medians of the step's wall time and of what is left of it outside the decode
expert stall and the passes, totals of misses and stall per step.

| phase | wall | outside the stall | expert stall | misses/step | hit rate |
|---|---|---|---|---|---|
| width 2, first baseline | 12.52 ms | 10.99 ms | 2.35 ms | 33 | 0.981 |
| width 2, KV-RAM move out at 12 MiB | 12.36 | 10.50 | 2.17 | 36 | 0.980 |
| width 2, E prefilling, C gone | 14.78 | 10.54 | 4.70 | 83 | 0.955 |
| width 2, KV-RAM move in at 16 MiB | 18.36 | 10.54 | 8.06 | 136 | 0.926 |
| width 2, last baseline | 12.37 | 10.58 | 2.69 | 45 | 0.976 |
| width 2, disk move out | 13.15 | 11.14 | 2.17 | 32 | 0.982 |
| width 2, disk move in | 18.23 | 11.07 | 7.07 | 116 | 0.937 |
| width 2, disk last baseline | 12.08 | 10.23 | 2.36 | 40 | 0.979 |
| width 3 (C, B1, B2) before any arrival | 25.89 | 15.50 | 10.49 | 166 | 0.939 |
| width 3 after C came back | 34.30 | 14.78 | 19.82 | 346 | 0.873 |

The model thread's own time in the passes: p50 0.04-0.10 ms a step (1.34 ms
with the window waited for). Width 3 after C came back, against before any
arrival, over four KV-RAM runs and the disk run: +32.5 to +40.5% p50.

**4. The control.** The same requests on a pool with room for E beside C, so
that nothing moves (`a_long_arrival_beside_decoding_lanes_with_nothing_moved`):

| width 3 (C, B1, B2) | ITL p50 | misses/step | expert stall/step |
|---|---|---|---|
| before any arrival | 24.56 ms | 184 | 10.74 ms |
| after E ended | 27.48 ms (+12%) | 254 (+38%) | 13.95 ms |

The run (14:53-14:57) shared the host with another agent's full kernel
build (cmake/nvcc, ~14:52-15:04) through both of its phases; no such load
is known beside the moved runs.

**5. No paging.** nvidia-smi and the WDDM counters once a second through a
KV-RAM leg: device memory 31.1-31.4 GB of 32.6 GB, flat; shared usage flat at
~37.4 GB (the pinned expert pool); SM clock 2.86-2.91 GHz throughout. In the
slow phases the board draws less power (230-270 W against 300-370 W) at the
same utilization: the SMs wait.

**6. Bit-exact.** AC 36 (both models, both tiers), AC 23 (both models, the
chain leg included: two moves into KV-RAM, one demoted to the disk), and the
27B's KV-RAM tests through the scheduler (`cuda_leaf_kv_ram_gpu`,
`cuda_leaf_kv_ram_dflash2_gpu`, `cuda_leaf_vision_gpu`'s eviction) pass on
the windowed path; the 27B moves unpaced.

## Finding

Observed:

- **The synchronous KV-RAM move's 3.5 GB/s was the block keys, not the
  link.** The whole calls copied Flash-Next's indexer keys a 4 KiB page per
  attention layer at a time: 44,256 copies, ~235 ms of the 325. Copied a run
  of pages at a time, the whole blob moves at the link's speed, as the
  windowed calls already did. The arena was pinned all along.
- **Without pacing a move no longer stalls anyone.** Every move's max is
  within the baseline's max + 3-11 ms, against +314/+318 ms synchronous.
- **The copies cost the other lanes about nothing at the chosen pace.** At 12
  MiB out and 16 MiB in, beside the rounds, a round's time outside its expert
  stall is the baseline's (10.50 and 10.54 against 10.58 ms); the model thread
  pays ~0.1 ms a step. Out of the device the cost grows with the window
  (+37% p50 at 64 MiB). Waiting for a move in's window before the round
  costs its copy time and saves nothing.
- **A move in's rounds pay for expert-cache misses.** Their extra time is the
  expert stall: misses rise from ~40 to ~120-140 a step, and the same rise is
  there with no move in flight -- from E's prefill on, with C gone -- and
  after C is back. The same holds of the disk move in, whose copies add +0.8
  ms. That, not the link, is the +50-58% the disk move in measured.
- **A long arrival alone raises the misses; with a move they roughly
  double.** With nothing moved (one control run), width-3 rounds after E pay
  +38% misses (+12% ITL). With C moved out and back, +100-110% misses
  (three runs counted them) and +33-40% ITL (five runs), for the rest of C's
  run (~1,400 rounds here), and back to normal once C has ended. One control
  against those runs, on expert caches 34.6 MB apart, with a run-to-run
  spread of several percent, run beside a kernel build on the host: the gap
  between them is the move's only as far as that one control can say.

Inferred, not measured:

- What raises the misses past the control's is in the expert residency's
  response to the sequence leaving and coming back, not in the copies: the time outside the
  stall is unchanged, so neither the transfers nor attention over moved pages
  pay it. Which (the cache relearning a working set larger than it, the
  lookahead's prefetches, or the routing of the restored sequence) was not
  measured.
- The disk move out's +12% (+0.9 ms outside the stall) is its 32 MiB staging
  copies and the IO threads beside the round; the first finding measured +4%
  for the same move.

## Implications

- The constants: `MOVE_IN_WINDOW_BYTES` 16 MiB, `MOVE_OUT_WINDOW_BYTES` 12
  MiB, Flash-Next only (`TransferPace::for_family`); the 27B, whose decode
  puts nothing on the link, moves unpaced, still off the model thread.
- AC 37's move-in bound cannot be met by pacing: what it measures is the
  expert cache after a long arrival and a move, not the move's traffic. The
  owner decides whether the bound applies to the transfer (met) or to the
  whole phase (not met), and whether the residency follow-up below is worth
  its own issue.
- A move out of the disk tier could be paced like KV-RAM's (its copies are
  still whole 32 MiB file windows).
- There is no disk-to-VRAM DMA on Windows (GPUDirect Storage / cuFile is
  Linux-only): a disk move in is unbuffered NVMe reads into pinned staging,
  then `cudaMemcpyAsync` host to device; the reads add CPU (CRC) and NVMe
  work, not PCIe traffic to the card.

## Limits and unknowns

- One host, one scenario, one to five runs a cell; the move-out p50 spreads
  +3 to +13% at one pace, so a single run is not a verdict.
- The control is one run, its pool is one chunk larger (34.6 MB less
  expert cache), and it ran beside a kernel build on the host: its ITLs are
  not comparable with the moved runs', and how far the build moved its
  misses (the lookahead's prefetches race the rounds) was not measured. A
  rerun on a quiet host comes before any residency work.
- AC 36 holds a moved sequence's tokens bit-exact at 4,400-token contexts and
  ~150-270 MB blobs; a 236K-token sequence's tokens across a move were not
  compared with an unmoved run at the same batch composition.

## Follow-ups

- Why the decode lanes' expert misses roughly double while a
  moved-and-restored long sequence runs (spec flash-next/03's residency,
  GitHub #309): more control runs first, then the hypotheses above.
- Pace the disk spill's device copies (sub-window slices).
