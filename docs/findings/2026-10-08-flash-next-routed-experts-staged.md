# Flash-Next routed experts at decode: staging each SM's work items through shared memory takes a layer from 25.4 to 20.9 µs at one token and 53.0 to 46.2 at three; the launch's first bytes and the gate/up-to-down barrier hold most of what is left

- Kind: experiment
- Status: current
- Observed: 2026-10-08
- Last verified: 2026-10-08
- Scope: kernel / Flash-Next routed-expert decode (`kernel/src/moe_decode_staged.cu`, `kernel/src/moe_decode.cu`)
- Related: https://github.com/gpillon/ignis/issues/306, [decode fusion](2026-10-08-flash-next-decode-fusion.md) (this is its roadmap's step 8), [MoE decode is structure-bound](2026-10-05-moe-decode-is-structure-bound.md), [Flash-Next decode round](2026-10-06-flash-next-decode-round.md), [ADR 0044](../adr/0044-experts-in-trellis-with-a-k-per-expert.md)
- Superseded by: none

## Question

The decode-fusion roadmap put the routed experts at 0.90 ms of a one-lane token above their
bandwidth floor (~19 µs a layer of excess) and estimated a kernel at ~18 µs a layer, -0.5 ms a
token, with "the reduction order changes; no design ready". Two restructurings of the register
ticket kernel had already failed (2026-10-05/06). Where does that kernel's time go at the
decode and verify widths (1-3 rows), what design takes it out, how far does the reduction order
move, and what does it buy?

## Evidence

All microbenchmark figures are `kernel/build/tests/ignis_kernel_moe_bench.exe --decode-only`
(device time per call from a CUDA graph of 256-320 calls, every call routed to experts the
previous calls did not touch; the study's K mix, mean K 2.48/2.47) with every route measured in
one process under a timing hold of the GPU lock; `--trace` adds the phase trace below. Raw logs
are in the main checkout's `.scratch/experts8/`.

### 1. Anatomy of the register kernel

A `<kTrace>` instantiation of each kernel stamps every work unit's phases with `%globaltimer`
(`kernel/src/moe_trace.h`; the stamps are 256 ns apart on this card, so medians move in that
step). Eight traced calls inside a graph of 64; the median call
(`compare-run3.log`):

| 1 token | from DRAM | weights in the L2 |
|---|---:|---:|
| launch span (first CTA entry to last exit) | 24.4 µs | 21.2 µs |
| CTA entry to its first ticket | 1.92 | 1.63 |
| gate/up unit (400): begin to ready / multiply / to end, medians | 5.12 / 2.82 / 0.77 | 1.79 / 3.07 / 0.77 |
| the eighth arrival's SwiGLU, mma to end | 2.82 | 2.30 |
| down unit (200): begin to ready / multiply / to end | 4.61 / 2.82 / 0.77 | 3.58 / 2.82 / 0.51 |
| down units waiting for h: median / max | 3.01 / 9.15 | 2.50 / 8.90 |
| CTA time multiplying / begin to ready / waiting for h | 21 / 33 / 8 % | 26 / 21 / 8 % |

- **The first multiply starts at ~5 µs.** In the 1-µs bins of the median call all 340 CTAs sit
  in "begin to ready" until bin 5: 1.9 µs to the first ticket (the ids, then the slot table),
  then each unit's 40 KB of loads, every SM asking at once.
- **Units come in waves and the bus idles between them.** A unit's weights live in registers,
  so a CTA asks for its next unit's bytes only after multiplying the current one; from DRAM a
  gate/up unit spends 5.1 µs before its multiply, 2.8 in it.
- **The second wave sets the end.** 400 gate/up units on 340 CTAs put experts 8-9's gate/up in
  a second wave (start 8.6-9.0 µs, end 19.7-20.5); their down units, taken at ~9 µs, wait up to
  9 µs for h; bins 12-16 hold 120-180 units waiting and 150 waiting for h.
- The multiply itself is near the issue floor: ~48 instructions per 16x16 tile (SHF 8.7, IMAD
  7.8, IDP 7.8, LOP3 6.8, PRMT/HFMA2 4 each, 2 HMMA; `cuobjdump -sass`), the IMAD and IDP at half
  rate, ~63 cycles per tile per warp. At this K mix a token-layer is 192,000 tiles, ~7 µs of the
  card's issue -- the same in every design below.

### 2. The design

`IGNIS_MOE_DECODE_TICKETS` (the load's default route) now runs, up to 4 tokens, one CTA per SM
with three roles on their own warps, handing work items along through shared memory
(`moe_decode_staged.cu`, 896 threads, 71 registers, ~99 KB of shared memory at 4 tokens):

- **Work items of 256 tiles.** Gate/up (expert, h block b, 256-input k-split): gate block b and
  up block b, one 16-column tile per mma warp over 16 k-tiles; 50 per expert. Down (expert, h
  block j, 512-column block): 8 k-tiles x 32 column tiles, two per mma warp; 25 per expert. All
  gate/up tickets first, then all down tickets, expert-major.
- **Producer, 4 warps.** The first takes tickets (one fetched ahead), keeps at most two items'
  copies in flight (the very first item alone, so each SM's first bytes land first) and places
  each item in a 78 KB ring by its own size (16-32 KB of tiles plus scales and inputs), waiting
  for every live item it would overwrite; all four warps issue the item as 16-byte `cp.async`
  copies counted on the item's `full` barrier.
- **Aux, 8 warps.** Prepare each item's fp16 operand into one of two A buffers -- gate/up: the
  tokens' inputs times suh, the 128-wide Hadamard, one power-of-two scale; down: the block's
  gate/up sums read back, both rotated (one warp each), svh, SwiGLU, x suh_down, rotated -- and
  reduce the previous item from one output buffer: gate/up adds its pre-rotation sums into the
  int64 fixed-point accumulator, fences and counts one arrival on (expert, block); down rotates
  each 128-column block, applies svh and the routing weight and adds into the output
  accumulator. A down item waits for its block's ten arrivals; the aux warps reduce the
  previous item first when it would otherwise wait, since that arrival may be the one missing.
  Each of a block's five readers counts itself after reading; the fifth zeroes the sums.
- **MMA, 16 warps.** Decode the trellis tiles from the ring straight into m16n8k16 B fragments
  and leave the raw sums for the aux warps; they wait on nothing but their operand.
- **Start.** Warp 0 finds the call's distinct experts with warp matches and loads each
  selection's slots right behind its id: two dependent round trips before the first ticket.

Deadlock-free for the same reason as the register kernel: tickets are taken and worked in
increasing order, every gate/up ticket precedes every down ticket, and no gate/up item waits on
another CTA. Past 4 tokens (verify rounds wider than that) the route keeps the register kernel;
`IGNIS_MOE_DECODE_REGISTERS` selects the register kernel at every width -- the switch back.

**How it got there** (1 token from DRAM, indicative: runs in one GPU session, the first ones in
a normal-mode lock hold):

| step | 1 token | what it showed |
|---|---:|---|
| register kernel | 25.4-25.5 | the anatomy above |
| one role: a producer warp + 16 compute warps, two stages | 23.2 | multiply 41% of CTA time; each item spent 0.5-0.9 µs before and 0.6-0.9 after its 1.9 µs multiply with all warps at a barrier |
| roles on their own warps (producer, 4 aux, 16 mma) | 23.9 | the mma warps waited 1.5-2 µs per down item for its operand: the aux warps reduced before preparing whenever a stage was late |
| h published once by the tenth arrival instead of five readers | 27.0 | the publish lengthens the gate/up-to-down chain by ~2 µs; reverted |
| a 78 KB byte ring, four slots | 24.5 | no change: the producer's issue, not its depth, was the bound |
| bulk copies (`cp.async.bulk`), one per contiguous run | 25.1 | 32 runs of 640 B per gate/up item cost 1.5-2.0 µs to issue (12 runs of a down item: 0.8) |
| 16-byte copies from four producer warps; slots read once | 23.4 | issue 1.0-1.3 µs; the remaining time is the per-SM fetch rate |
| distinct experts by warp matches, slots behind the ids | 21.4 | the first multiply ~1.5 µs earlier |
| eight aux warps, two per token for a down operand | **20.9** | aux busy 35%, the mma warps' operand waits 25 -> 18% at 3 tokens |

Interleaving each expert's down items a few experts behind its gate/up items (a lag of 2, 4, 6,
8 experts) measured 32.3, 25.0, 23.8, 22.2 µs at one token against 21.3 for all gate/up first:
a down item taken early waits on gate/up items still in flight (`lag*-run3.log`).

### 3. Reduction order and tolerance

Both kernels round every fp16 operand under a power-of-two scale (now per 256 or 128 inputs
instead of per 640), which moves no rounding of a normal number, and sum across work items in
exact int64 fixed point. They differ in where fp32 partial sums round: a gate/up partial sums 256
inputs in the MMA's fp32 chain instead of 640, and down sums one 128-input block of h with the
output Hadamard taken per block (it is linear) and the five blocks added in fixed point, instead
of one fp32 chain over 640. The gate/up sums therefore differ by ~1e-7, and that difference
reaches the output through one place: h's fp16 operand for down, where it flips the rounding of a
few of h's 640 entries by one ulp (2^-11); each flip moves the token's output by ~2^-11/sqrt(640)
~ 2e-5 of its norm.

- `test_moe_decode_routes` (CTest, synthetic records at every K class): ten experts alone
  (one-hot routing weight) differ by 6.8e-7, 9.0e-7, 3.1e-6, 4.5e-6, 1.1e-5, 1.7e-5, 3.2e-5,
  3.9e-5, 3.9e-5, 4.9e-5 relative L2 -- discrete, two experts with no flipped entry at 1e-6 --
  and both kernels sit at 3.83e-4 against fp64 (bound 2e-3). Calls of 1-4 tokens: worst 5.67e-5
  L2, 6.15e-5 max. Bound asserted: 2e-4 L2, 1e-3 max. Five and eight tokens on the tickets
  route give the register kernel's bits; reruns, records in other slots and two graph replays
  are bit-equal.
- `moe_artifact_gpu` on real weights (layers 2, 24, 46): (pending: run 6)
- The kernel CTests (`ctest -R "moe|trellis"`, 22 tests, including the moe_experts/moe_block
  arms and slot traps on the registers route): all pass (`ctest-moe-run5.log`).

### 4. The microbenchmark

One process, timing hold (`compare-run3.log` for the register and cluster routes,
`staged-run5.log` for the final staged kernel; `real-L24-run4.log` for layer 24's own 512
expert records, which `crates/artifact/examples/dump_expert_pool.rs` writes and `--pool`
loads):

| µs per layer | 1 token | 2 tokens | 3 tokens | 1 token, L2-resident |
|---|---:|---:|---:|---:|
| register kernel, study mix | 25.4 | 37.3 | 53.0 | 21.4 |
| staged kernel, study mix | **20.9** | **30.7** | **46.2** | 19.0 |
| register kernel, layer 24's records | 25.4 | 36.8 | 51.2 | 20.3 |
| staged kernel, layer 24's records | **20.9** | **31.6** | **46.0** | 18.7 |
| clusters route, layer 24's records | 36.1 | 78.5 | 88.2 | 30.6 |

-4.5 µs a layer at one token (-18%), -5.2 at two (-14%), -5.2 at three (-10%) on real records:
at one token 41% of the DRAM roofline (the register kernel 34%).

Where the staged kernel's 20.3 µs (traced, 1 token, from DRAM) go: entry to first ticket 1.7
µs; the first multiply at ~4.5 µs (bins 2-4); bins 5-14 multiply on 87-163 of 170 SMs; the last
experts' gate/up end at 12.5 µs and their down items at 19.5; CTAs finish over the last ~4 µs
(147 to 1 alive from bin 15 to 20). The mma warps multiply 43% of CTA time and wait for an
operand 25%; the aux warps' stage wait has a median of 1.2 µs, and a gate/up item's copies take
~0.8 µs to issue and ~1 more to be seen.

### 5. Served A-B-A

(pending: after the decode fusions reach main)

### 6. Gates

(pending: `flash_next_forward_gpu` (G1), `moe_artifact_gpu`, `cargo test --workspace`)

## Finding

(pending)

## Implications

(pending)

## Limits and unknowns

- The microbenchmark runs the kernel alone; in the decode round it shares the card with the
  shared expert's branch and the lookahead router, and one CTA per SM with ~99 KB of shared
  memory cannot start on an SM another kernel's CTAs occupy.
- `%globaltimer` steps by 256 ns here; trace medians are quantized to it.
- The iteration table's runs before the last two were in a normal-mode lock hold (other agents
  may have built meanwhile); the comparison tables are from timing holds.

## Follow-ups

(pending)
