# The 27B's attention over hq-e8-2b on ignis's warp-specialized route, served: +5.9% rounds a second at 64K and +10.2% at 142K at one lane, +8.3% at eight lanes and 33K, the same one-lane text, no cell slower

- Kind: experiment
- Status: current
- Observed: 2026-11-14
- Last verified: 2026-11-14
- Scope: kernel and server / the 27B's hq-e8-2b attention in the decode and verify rounds (spec #570):
  ignis's warp-specialized route (`hq_verify_ws_kernel`) against the vendored kernel's throughput route,
  on the RTX 5090, the tree 74304802
- Topic: decode-27b
- Related: https://github.com/gpillon/ignis/issues/570, https://github.com/gpillon/ignis/issues/571,
  https://github.com/gpillon/ignis/issues/572, https://github.com/gpillon/ignis/issues/573,
  [ADR 0093](../adr/0093-on-blackwell-the-27bs-decode-and-verify-rounds-attend-over-hq-e8-2b-on-ignis-own-route-and-the-vendored-kernel-stays-its-oracle.md),
  [ADR 0031](../adr/0031-vendored-kernel-bottleneck-exemption.md),
  [hq verify decode cost](2026-09-24-hq-verify-decode-cost.md) (#268: the kernel this replaces on Blackwell, and the follow-up it named),
  [the sampled draft served](2026-11-13-the-sampled-draft-served-on-the-27b.md) (#574: the launch protocol, and the baseline's draft rule)
- Superseded by: none
- Summary: RTX 5090, 27B NVFP4, DFlash2: ignis's warp-specialized hq-e8-2b attention takes 0.58-0.84 of the old kernel's time; served, +5.9%/+10.2% rounds a second at one lane at 64K/142K, +8.3% at 8x33K, one-lane text unchanged.

## Question

#268 left the 27B's attention over hq-e8-2b in the decode and verify rounds
latency-bound on registers: 8 warps per SM, 3.9 to 4.9 times its MMA floor.
Its next step was more warps per SM. Spec #570 built it as a route of ignis's
own, which decodes keys and values in twelve warps and runs the matrix products
in eight, byte for byte the reference route's (#571), and made it the default on
Blackwell with today's route behind a measurement switch (#572). The spec
inferred from the kernel alone about +6%, +10% and +14% rounds a second at one
lane at 64K, 142K and 253K. What does the kernel table say, what binds the new
route, and what does the server measure, cell by cell: is any cell slower, is
the greedy text the same, and do the rounds commit the same tokens?

## Evidence

**Where the records are.** In the main checkout's untracked `.scratch/570/`:
- `ab/`: the served A/B. `protocol.md` (fixed before the first launch, with its
  amendment), `run-all.sh`, `run-launch.sh`, the driver `longctx570.py` and its
  prompt body `body.txt` (sha256 `841b37dd...` in `body.sha256`), `analyze.py`,
  which writes `summary.md` and `summary.json`, the coordinator's `verdict.md`,
  `served-check-572.md`, and one directory per launch, `<arm>-launch-<k>/`, with
  `launch.json`, `make-config.txt`, `load-lines.txt`, the cells' JSON
  (`l1.json`, `l1-sampled.json`, `l4.json`, `l8.json`), `ignis-server.log`,
  `gpu.csv` and the `make gpu-status` readings before and after.
- `571/gpu/bench/`: the kernel table, `summary.txt` and one log per cell and
  window.
- `ncu/`: `summary.md`, the two routes' metrics CSVs and the bench binary's
  sha256. The `.ncu-rep` reports stay on that disk: a report can embed the
  profiled process's environment, and none is committed.
- `gpu/`: #572's GPU legs.

**The kernel alone** (#571, merged as d9f24320 on `570-integration`): the
bench `ignis_hq_verify_bench`, built from 8a87d853, times the three routes from
one binary as the verify round launches them (width 8, grid KV heads x split
capacity x lanes): the reference hq route (`--reference`), today's throughput
route, and the new one (`--ws12`). It is the median of 20 launches, each after
an L2 flush, on the RTX 5090 on 2026-11-14. Microseconds per launch, reference
/ today / new, with the residual window off and on:

| cell (lanes x keys) | window off | window on | new / today, off / on |
|---|---:|---:|---:|
| 1 x 512 | 53.2 / 34.8 / 20.3 | 34.8 / 32.7 / 22.5 | 0.583 / 0.688 |
| 1 x 4,096 | 61.4 / 38.8 / 32.7 | 61.4 / 38.8 / 32.7 | 0.843 / 0.843 |
| 8 x 1,024 | 132.6 / 94.1 / 73.7 | 124.4 / 96.2 / 79.8 | 0.783 / 0.830 |
| 1 x 7,168 | 89.6 / 54.8 / 42.7 | 90.0 / 54.8 / 42.9 | 0.779 / 0.783 |
| 1 x 30,720 | 258.0 / 140.8 / 102.4 | 259.6 / 141.2 / 104.3 | 0.727 / 0.739 |
| 1 x 59,392 | 366.5 / 198.6 / 145.4 | 366.6 / 200.5 / 145.4 | 0.732 / 0.725 |
| 1 x 65,536 | 409.5 / 222.7 / 161.8 | 413.2 / 223.2 / 161.8 | 0.727 / 0.725 |
| 1 x 126,976 | 749.1 / 396.8 / 286.2 | 753.4 / 399.0 / 286.7 | 0.721 / 0.719 |
| 1 x 145,408 | 858.0 / 452.1 / 325.4 | 860.1 / 454.7 / 325.7 | 0.720 / 0.716 |
| 4 x 30,720 | 949.9 / 497.2 / 349.8 | 755.6 / 475.1 / 342.0 | 0.704 / 0.720 |
| 8 x 14,336 | 816.2 / 446.4 / 342.0 | 810.5 / 448.2 / 342.0 | 0.766 / 0.763 |
| 8 x 30,720 | 1674.7 / 876.5 / 656.7 | 1492.2 / 864.3 / 651.2 | 0.749 / 0.753 |
| 1 x 259,072 | 1498.6 / 786.0 / 567.2 | 1504.4 / 790.5 / 571.4 | 0.722 / 0.723 |

- The three routes print the same hash of their partials on every cell and
  window, and with the window off each hash is the prototype's of 2026-11-11.
- Per GQA layer per 1K keys per lane, as the bench prints it with the window
  off, the new route costs 2.24 to 2.53 us at one lane from 59K keys up, 2.74
  at 8 x 30K and 3.41 at 1 x 30K, where #268's MMA floor is 0.96 us.
- `ignis_kernel_hq_verify_exact_test` holds the new route and the throughput
  route to the reference on the eleven shapes: every partial and every cache
  byte equal. The codec rows tests hold the lean decoder at 0 elements
  differing on the real rows, the rows the encoder never writes and the
  escalated heavy-tailed rows. A one-bit mutation of the decoder turns all three
  red; the kernel's CTest suite passes, 149 of 149.

**What binds the new route**: Nsight Compute 2025.4.0, `--set full` at its
default clock control (base clocks), on the bench binary
(`ignis_hq_verify_bench-8a87d853.exe`, sha256 `7b0bf45f...`), 8 lanes x 30,720
keys, cached input, the second launch of each route. The first rows are
`ncu/summary.md`; the rest are read from the two reports with
`ncu --import <report> --page raw --csv --metrics ...`:

| | today (throughput) | new (`hq_verify_ws_kernel`) |
|---|---:|---:|
| duration | 1.242 ms | 0.904 ms |
| grid x block | (4, 85, 8) x 128 | (4, 85, 8) x 640 |
| registers per thread | 255 | 96 |
| static shared memory per block | 48,128 B | 87,408 B |
| blocks per SM, limited by | 2 (registers and shared memory) | 1 (registers and shared memory) |
| warps active per SM | 7.90 | 19.95 |
| eligible warps per scheduler per cycle | 0.57 | 1.51 |
| issue active, % of peak | 45.9% | 67.2% |
| warp instructions | 700.7 M | 753.0 M |
| busiest pipes, % of peak | LSU 31.4, tensor 26.4 (cycles), ALU 24.2 | LSU 47.7, tensor 38.5 (cycles), ALU 34.5 |
| L1 throughput, DRAM throughput | 37.8%, 9.0% | 54.6%, 11.7% |
| warps stalled per issue: fixed latency, not selected, barrier, short scoreboard | 1.17, 0.24, 0.41, 0.40 | 1.24, 1.24, 0.94, 0.51 |

The prototype's hand build (2026-11-11) read the same picture: 1.25 against
0.899 ms, 7.9 against 20 warps, about 67% issue for the new route.

**The served A/B.** Fixed before the first launch (`ab/protocol.md`, amended
before any launch to the build below), on #574's launch protocol:
- **Build**: the tree 74304802, the tip of `570-572` and the tree #572 merged
  into `570-integration` as 830c2bd6, release (`make build UI=0`), one exe for
  both arms: `ignis-server.exe` sha256 `0b8da62fcfb706a5...`.
- **Arms**: *new* is the exe at its default, which on the RTX 5090 is ignis's
  route; *today* is the same exe under `IGNIS_HQ_DECODE_ATTENTION=vendored`. A
  launch refused to measure unless its load's `ignis.runtime.decode_routes`
  named its arm's route; every load did (`ours`, `vendored`).
- **Server**: `make start UI=0 LOOP_GUARD=off IDLE_UNLOAD=0`, main's production
  flags since 0ba44c53: `--model-artifact ./models/qwen3_8_27b_nvfp4full-v2.ninfer
  --model-kv-format hq-e8-2b --reuse-kv-host-pool-bytes 8G
  --qwen38flashnext-reuse-kv-host-pool-bytes 2G --spec-backend dflash2
  --spec-draft-tokens 7 --server-ui false --server-system-message-policy merge
  --server-developer-message-policy inplace --model-idle-unload 0
  --model-loop-guard off`, the sampled draft as the draft rule, the residual
  window on as served.
- **Configuration**: `ignis.config.source` read `source: discovered`, profile
  `rtx5090` by `auto`: the host's user configuration file
  (`%APPDATA%\ignis\config.yaml`) put the 27B's exception weights in FP8, its
  headroom at 1500M (1,572,864,000 bytes in the plan), the host margin at 1G
  and the request timeout at 2402 s, on every launch alike.
- **Each load**: the same VRAM plan on all four (31,518,019,584 bytes free at
  start, the CUDA context measured at 452,161,536 bytes, 16,618 KV pages,
  1,063,552 tokens), the same prefill routes (GDN `ours`, attention `fp8`,
  linear `fp16`), and the same graph node counts, decode
  [885, 997, 997, 1132, 1132, 1132, 1132, 1132] and verify
  [1122, 1083, 1088, 1088, 1119, 1119, 1124, 1109] at widths 1 to 8.
- **Order**: today 1, new 1, new 2, today 2, 09:09-09:21Z on 2026-11-14, the
  card held for the whole campaign and nothing compiling. Across the four
  launches the SM clock's median read 2,437-2,580 MHz, the power's 514-517 W,
  the temperature's 68-70 C.
- **Cells** (`longctx570.py`, #268's driver): deterministic prompts over one
  body, every lane prefilled alone first so the measured request claims its
  prefix by reuse, `ignore_eos`, 768 tokens. One lane at 7K, 33K, 64K and
  142K, greedy; one lane at 64K sampled at temperature 1, top-p 0.95, top-k 20,
  seed 570; four lanes at 33K and eight at 7K, 14K and 33K, greedy.
- **Readings**, per request from `ignis.request.done`: a round is
  `itl_ms_mean x itl_samples / spec.rounds` (#268's formula), tokens a round
  `tokens / spec.rounds`. A cell pools all its requests' rounds over an arm's two
  launches; an arm's spread is the difference between its two launches.

Milliseconds a round per launch and pooled per arm, rounds a second new
against today, and tokens a round (the mean over a cell's requests):

| cell | prompt tokens | today 1 / new 1 / new 2 / today 2 | today | new | rounds/s | spread today / new | tokens a round, today 1 / new 1 / new 2 / today 2 |
|---|---:|---|---:|---:|---:|---|---|
| 1 x 7K | 7,212 | 16.752 / 16.412 / 16.412 / 16.595 | 16.673 | 16.412 | **+1.6%** | 0.157 / 0.000 | 5.020 on all four |
| 1 x 33K | 33,468 | 18.257 / 17.413 / 17.503 / 18.106 | 18.182 | 17.458 | **+4.1%** | 0.151 / 0.089 | 4.291 on all four |
| 1 x 64K | 63,786 | 19.623 / 18.384 / 18.491 / 19.409 | 19.516 | 18.437 | **+5.9%** | 0.214 / 0.107 | 4.830 on all four |
| 1 x 142K | 141,667 | 23.402 / 21.191 / 21.146 / 23.271 | 23.337 | 21.168 | **+10.2%** | 0.131 / 0.045 | 3.859 on all four |
| 1 x 64K, sampled | 63,786 | 19.953 / 18.800 / 18.579 / 19.660 | 19.806 | 18.689 | **+6.0%** | 0.294 / 0.221 | 3.268 on all four |
| 4 x 33K | 33,468 | 27.911 / 25.760 / 25.755 / 27.787 | 27.849 | 25.757 | **+8.1%** | 0.124 / 0.004 | 4.197 on all four |
| 8 x 7K | 7,212 | 32.222 / 31.612 / 31.822 / 32.175 | 32.199 | 31.717 | **+1.5%** | 0.047 / 0.211 | 5.121 / 5.264 / 5.177 / 5.121 |
| 8 x 14K | 14,443 | 36.249 / 34.537 / 34.579 / 36.120 | 36.184 | 34.558 | **+4.7%** | 0.129 / 0.042 | 6.061 on all four |
| 8 x 33K | 33,468 | 40.758 / 38.309 / 37.915 / 41.755 | 41.257 | 38.112 | **+8.3%** | 0.997 / 0.394 | 3.858 / 4.061 / 3.797 / 4.066 |

In rounds a second, one lane went from 51.24 to 54.24 at 64K and from 42.85
to 47.24 at 142K. A round is shorter by 0.26, 0.72, 1.08 and 2.17 ms at one
lane at 7K, 33K, 64K and 142K, 2.09 ms at 4 x 33K, and 0.48, 1.63 and 3.14 ms
at eight lanes.

**Text**, compared lane by lane between launches:
- **Every one-lane cell, greedy and sampled**: the same text on all four
  launches. At 64K, the cell #572's served check reads, the greedy text's
  sha256 is `21294aeb3a21dc3d...` and the sampled one's `ac170afd96913210...`
  on every launch, 159 and 235 rounds.
- **4 x 33K**: every lane the same on all four launches, every pair 4 of 4.
- **Eight lanes**: lanes identical across all four launches, 5 of 8 at 7K, 6
  of 8 at 14K, 3 of 8 at 33K. Pair by pair:

  | pair | 8 x 7K | 8 x 14K | 8 x 33K |
  |---|---|---|---|
  | today 1 / today 2 (control) | 8 of 8 | 6 of 8 | 5 of 8 |
  | new 1 / new 2 (control) | 6 of 8 | 6 of 8 | 6 of 8 |
  | today 1 / new 1, today 1 / new 2, new 1 / today 2, new 2 / today 2 | 5, 6, 5, 6 of 8 | 8, 6, 6, 8 of 8 | 5, 5, 6, 6 of 8 |

**#572's GPU legs** (the merge 830c2bd6, logs in `gpu/`): the red run (the
launcher handing the route half the softmax scale) fails
`hq_decode_attention_gpu` at its first text comparison and passes reverted;
that test holds every lane's text between the two routes over greedy decode and
verify rounds at widths 1 to 8 and sampled verify rounds at widths 1 and 8, at a
65,536-token context. The VRAM plan at the make defaults is byte for byte main
177175fa's on three alternated pairs of loads. The GPU profile passes the
kernel's CTest suite, 150 of 150, and 374 Rust GPU test binaries; three tests
that pin a KV pool fail with the same values on main, a pin older than #604's
measured CUDA context (GitHub #607), not the route.

## Finding

**Observed.**
- **The kernel**: the new route takes 0.58 to 0.84 of today's route's time on
  all 13 cells, with the window off and on, and 0.72 to 0.74 at one lane from
  30K keys up; it is slower on none, and its partials are the reference's to
  the byte.
- **What binds it**: one block of 640 threads per SM, which its 96 registers a
  thread and its 87,408 bytes of shared memory each limit to one, so 20 warps
  are resident where today's route holds 8. A scheduler issues in two thirds
  of its cycles (67.2% against 45.9%), with 1.51 eligible warps a cycle on
  average against 0.57. No execution pipe is above 48% of its peak (LSU 47.7%),
  L1 runs at 54.6% and DRAM at 11.7%. The route executes 7.5% more warp
  instructions than today's and finishes in 0.73 of its time.
- **Served, no cell is slower**: the new route's pooled round is shorter in
  all 9 cells, in each by more than the larger of the two arms' spreads.
- **Served, rounds a second**: +1.6%, +4.1%, +5.9% and +10.2% at one lane at
  7K, 33K, 64K and 142K; +6.0% at 64K sampled; +8.1% at four lanes and 33K;
  +1.5%, +4.7% and +8.3% at eight lanes at 7K, 14K and 33K. At each lane count
  the gain grows with the context.
- **Text**: identical in every one-lane cell, greedy and sampled with a seed,
  and in every lane at 4 x 33K, on all four launches.
- **Tokens a round** are the same under the two routes in every cell whose text
  is the same, every one-lane cell and 4 x 33K. At 8 x 14K they are the same
  too, 6.061 on all four launches, though two lanes' text differs between some
  of them; at 8 x 7K and 8 x 33K they differ with the text.
- **Eight lanes do not reproduce their own text**, under either route: the
  controls differ in 2 or 3 lanes of 8 at 14K and 33K, and the new route's at
  7K. At 14K two cross-route pairs agree in every lane while both same-route
  pairs differ in two. #574's finding saw the same in eight-lane greedy batches
  on main.

**Inferred.**
- **Served against the kernel's estimate.** The spec's +6%, +10% and +14% at
  one lane at 64K, 142K and 253K were inferred from the kernel alone. The
  server measures +5.9% at 64K and +10.2% at 142K; 253K was not served, so its
  +14% stays an inference. Taking 16 launches a round, one per GQA layer as
  #268 counted, the kernel table's window-on savings give 0.19, 0.59, 0.98 and
  2.06 ms a round at one lane at 7K, 30K, 64K and 142K keys. The server saved a
  little more, 0.26, 0.72, 1.08 and 2.17 ms, and at four and eight lanes a
  little less (2.09 against 2.13 ms at 4 x 33K, 1.63 against 1.70 at 8 x 14K,
  3.14 against 3.41 at 8 x 33K). At several lanes the rounds average fewer live
  lanes as requests finish, as #268's limits say.
- **The route changes no text.** It writes the reference route's bytes, so a
  lane's text cannot move with it. The eight-lane differences are the batch's
  own run-to-run variation: both controls show it, and at 14K two cross-route
  pairs agree in every lane where both same-route pairs do not.
- **Under sampling the gain is the same.** The attention runs before the
  accept and costs a round the same whatever the draft rule: the sampled cell at
  64K gains 6.0% against the greedy cell's 5.9%, with the same text and the same
  3.268 tokens a round. For a sampled request the route's shorter round and the
  sampled draft's extra tokens a round (#574) multiply.
- **What binds now** is issue and latency inside 20 warps, not bandwidth:
  warps stalled on fixed-latency dependencies (1.24 per issue) and on barriers
  (0.94) sit beside warps that were ready and not picked (1.24). More warps per
  SM would need fewer registers or less shared memory a block; fewer cycles
  would need fewer instructions or barrier waits.

## Implications

- On a Blackwell card the 27B's long-context decode is the faster route by
  default, with the same bytes and the same VRAM plan; ADR 0093 records the
  route, beside the vendored kernel as ADR 0031 asks, with the reference route
  as its oracle.
- `IGNIS_HQ_DECODE_ATTENTION=vendored` is the comparison for a measurement on
  another card or another build; this A/B's driver and analysis are the
  instrument (`ab/longctx570.py`, `ab/analyze.py`).
- After this step the attention still costs 2.2 to 3.4 us per GQA layer per 1K
  keys per lane from 30K keys up, 2.3 to 3.6 times #268's MMA floor, above the 2x threshold
  #268 set for its option 3, a decoded tier. That option stays where #268's
  owner decision left it (more work needed), now to be judged on top of this
  route.
- A served comparison of two routes reads one-lane cells and four-lane cells
  for text; eight-lane greedy batches need a control of the same route twice.

## Limits and unknowns

- One card, the RTX 5090, one artifact, the NVFP4 27B, two launches per arm.
  Every Blackwell card class takes the route, the 16 GB artifact included; their
  speed is not measured, and the guide's 16 GB figures, 216 tok/s at 253K among
  them, predate it.
- 253K tokens of context was not served; the kernel table's 1 x 259,072 cell is
  the only evidence there.
- The 8 x 7K and 8 x 33K cells compare different tokens in some lanes, so their
  gains mix the route's round with a batch that drafted differently; their
  rounds a second are the round's cost, not a like-for-like text.
- Why the one-lane server saves 0.07 to 0.13 ms a round more than 16 kernel
  launches account for is not established.
- The Nsight Compute reading is one cell, 8 x 30K at base clocks, with the
  window off; the stall and pipe figures are its alone.
- The host's user configuration file (FP8 exception weights, 1500M headroom)
  applied to every launch; the absolute rates are for that configuration.

## Follow-ups

- GitHub #607: the three GPU tests that pin a KV pool, red on main since #604,
  which #572's profile met and which the route does not touch.
