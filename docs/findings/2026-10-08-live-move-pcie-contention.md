# A live move's PCIe contention on Flash-Next: the windowed disk move out holds the other lanes, a move in slows them, a synchronous KV-RAM move stalls them ~0.32 s

- Kind: experiment
- Status: superseded
- Observed: 2026-10-08
- Last verified: 2026-10-08
- Scope: serving / live moves (ADR 0045, the fixed branch), KV-RAM and KV-disk transfers, decode ITL of the other lanes on Flash-Next
- Related: https://github.com/gpillon/ignis/issues/309, [ADR 0045](../adr/0045-the-kv-pool-follows-residency-and-state-goes-down-to-disk.md), spec [vram-budget/03](../specs/vram-budget/03-kv-pool-policy-and-kv-disk.md) (AC 37), [KV-disk contention on the volume](2026-10-08-kv-disk-contention-on-the-volume.md)
- Superseded by: [Live moves a window at a time](2026-10-08-live-moves-windowed.md)

## Question

Spec vram-budget/03 AC 37: when a live sequence of more than 1 GB moves off
the device and back while other lanes decode, how much does each move cost
those lanes? Starting thresholds, for the owner to confirm: move out ITL p50
within +10% of a baseline, move in within +25% (it shares the expert stream's
direction), either one's max within the baseline's max + 150 ms ("one
synchronous KV-RAM copy of a whole-context blob, ~0.1 s at ~12 GB/s").

## Evidence

RTX 5090 on PCIe Gen 3 x16, the Flash-Next artifact and the tier's files on
F: (NVMe). Branch `kv-p3-moves` (#309 P3, the fixed branch), the harness
`crates/server/tests/live_move_contention_gpu.rs`, one run per leg:

- the load: 4 decode lanes, 262,144-token context, 8,192-token chunks, the
  pool cut to one context with `--kv-pool-bytes`'s token form, prompt reuse
  off, no retained slots;
- no n-gram hot rows (every gather reads the artifact): with 1 GiB of them the
  host plan refused the KV-RAM leg's arena -- "the host plan needs
  40479801344 bytes (expert_pool 37795446784, ngram_hot_rows 1073741824, ...,
  kv_ram_arena 1610612736) of the 45865066496 bytes of physical memory
  available, and must leave 6442450944 (6 GiB) of it" -- so both legs of run 2
  run without them, the KV-RAM leg with a 1,200 MiB arena;
- C, `agent`: a 236,000-token prompt, 2,000 tokens; its blob is 1,128,096,256
  bytes. B1 and B2, `interactive`: 2,000-token prompts, 1,200 and 1,600
  tokens. E0, `interactive`, an 8,192-token prompt that fits beside them and
  moves nothing. E, `interactive`, an 18,000-token prompt that does not fit:
  its admission moves C out, and C comes back when E ends. Everything greedy,
  `ignore_eos`, an explicit `max_tokens`.

Each move's window is the steps it spanned; its baseline is as many steps of
the same kind of work with no transfer in flight, from the same run. ITL is B1's
and B2's.

| leg | move | duration | GB/s | steps | ITL p50, move vs baseline | ITL max, move vs baseline | baseline |
|---|---|---|---|---|---|---|---|
| KV-RAM | out | 327.3 ms | 3.45 | 1 | 3,164.4 vs 2,851.0 ms (+11.0%) | +313.5 ms | the step E0's first chunk ran in |
| KV-RAM | in | 320.0 ms | 3.53 | 1 | 338.3 vs 19.9 ms | +318.4 ms | C, B1, B2 at width 3 |
| disk | out | 627.8 ms | 1.80 | 48 | 12.52 vs 9.83 ms (+27.4%); vs the last baseline 12.01 ms, +4.3% | 18.70 vs 15.05 ms (+3.7 ms); vs the last, -8.1 ms | B1', B2' alone at width 2 |
| disk | in | 670.2 ms | 1.68 | 39 | 18.08 vs 9.88 ms (+83.0%); vs the last baseline 12.01 ms, +50.5% | 40.70 vs 15.05 ms (+25.7 ms); vs the last, +13.9 ms | B1', B2' alone at width 2 |

- **Two width-2 baselines.** B1' and B2' decode alone before C arrives (the
  baseline above), and again after everything else has ended (the "last"
  one). Over their 598 gaps each: 10.86 then 11.80 ms p50 on the KV-RAM leg,
  10.71 then 12.01 ms on the disk leg; maxima 26.5-30.3 ms.
- **A synchronous move is one gap.** The KV-RAM move out ran inside the step
  that also ran E's first 8,192-token chunk, so its baseline is the step that
  ran E0's: the same chunk, but decoding at width 4 (C, B1, B2, E0) where the
  move's step decodes at width 2. B1 and B2 land their tokens in the same
  step, so the two gaps of each window are one value.
- **Run 1** (the disk leg alone, with 1 GiB of hot rows, no last baseline):
  out 627.3 ms (1.80 GB/s), p50 12.57 vs 9.72 ms (+29.3%), max +4.3 ms; in
  649.9 ms (1.74 GB/s), p50 18.87 vs 9.72 ms (+94.1%), max +23.7 ms.
- **No work lost** in either leg: every request generated its full
  `max_tokens`, no `Requeued`, no dropped snapshot, no disk failure, C moved
  out once and back once, and nothing else moved. No lane parked: the fixed
  branch has no parking.

Raw samples and logs: the main checkout's `.scratch/kv-p3/`
(`ac37-KvRam.json`, `ac37-KvDisk.json`, `ac37-KvDisk-run1-hot1g.json`,
`ac37-{kvram,disk}-run{1,2}.log`).

## Finding

Observed:

- **The windowed move out holds the other lanes.** Against the baseline taken
  last, p50 +4.3% and the max below the baseline's; against the one taken
  before C's prefill, +27%. AC 25's disk spill measured +3-8% (P2).
- **The windowed move in slows them** by +50% p50 against the last baseline
  (+83% against the first), with the max still +14-26 ms. The +25% bound is
  missed.
- **A synchronous KV-RAM move stalls every lane for the copy**, ~0.32 s for
  1.13 GB in either direction, at 3.45-3.53 GB/s -- a third of the ~12 GB/s
  the ADR assumed -- so the +150 ms max bound is missed both ways, by
  ~165-170 ms.
- **The two width-2 baselines differ by 9-12%** on one host, minutes apart.

Inferred (not measured here):

- The directions differ as the ADR expected: a move out is device to host,
  against the expert stream; a move in is host to device, the stream's own
  direction, where each 32 MiB window competes with the 66-74 MB of experts a
  decode round streams.
- The first baseline understates the card's state during the moves: it is
  taken before a 236,000-token prefill has gone through the expert cache, the
  last one after. Why the second is slower was not measured.
- 3.45 GB/s for the KV-RAM copy is not the link's speed; where it goes (the
  leaf's section-by-section snapshot into the pinned arena, or the arena
  itself) is unknown.

## Implications

- The thresholds as written: the disk move out passes (against the last
  baseline); the disk move in fails p50 and passes max; the KV-RAM moves fail
  max both ways. The owner decides the bounds.
- The spec's Out of Scope names the follow-up for the synchronous leg:
  "Windowing the KV-RAM moves. A follow-up if AC 37's synchronous leg misses
  its bound." It missed.
- A move in that should stay under +25% needs pacing: fewer windows in flight
  per round, or windows issued between rounds instead of beside them.
- A baseline for a contention bound on Flash-Next should be taken after the
  long prefill, not before it.

## Limits and unknowns

- One run per leg (two for the disk move, the first with hot rows); one host,
  one NVMe, the n-gram reads all from the artifact.
- The KV-RAM move out's window is one step that also ran a prefill chunk; its
  baseline step ran the same chunk at another decode width.
- The cost of the KV-RAM copy was not decomposed.

## Follow-ups

- Window the KV-RAM moves (spec vram-budget/03, Out of Scope), and find where
  the synchronous copy loses two thirds of the link.
- Pace a move in so the decoding lanes' p50 stays within a bound the owner
  confirms.
