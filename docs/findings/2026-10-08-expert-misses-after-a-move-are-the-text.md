# Expert misses after a live move are the text's, not the move's

- Kind: experiment
- Status: current
- Observed: 2026-10-08
- Last verified: 2026-10-08
- Scope: serving / live moves (ADR 0045, fixed branch), Flash-Next's expert residency during and after a move, AC 37's and AC 43's measurement, the KV-disk spill's pace
- Related: https://github.com/gpillon/ignis/issues/310, [ADR 0045](../adr/0045-the-kv-pool-follows-residency-and-state-goes-down-to-disk.md), spec [vram-budget/03](../specs/vram-budget/03-kv-pool-policy-and-kv-disk.md) (AC 37, AC 43), supersedes [live moves a window at a time](2026-10-08-live-moves-windowed.md)
- Superseded by: none

## Question

[Live moves a window at a time](2026-10-08-live-moves-windowed.md) found that
the rounds after a KV-RAM or KV-disk move in pay for expert-cache misses, not
for the copies: ~40 misses a step became ~130, and with C moved out and back
the width-3 rounds missed roughly twice as many experts for the rest of C's
run, against +38% in one control with nothing moved, run beside a kernel
build. Is that gap the move's -- the cache relearning, the lookahead's
prefetch budget reacting to the rows, the restored sequence's routing -- or
the long arrival's? And does pacing the KV-disk spill's device copies as
KV-RAM's are bring its move out within AC 37's +10%?

## Evidence

RTX 5090 on PCIe Gen 3 x16, Windows (WDDM), Flash-Next, branch `fu310`.
`crates/server/tests/live_move_contention_gpu.rs`, AC 37's scenario: C
(`agent`, 236,000-token prompt) decodes beside B1 and B2 (`interactive`,
2,000-token prompts); E0 (8,192 tokens) fits beside them; E (18,000) does
not, and moves C out, which comes back once E ends. The control runs the same
requests on a pool with room for E, so nothing moves. Every run under the GPU
lock, alone on the card. Raw logs, per-step timelines (each step's tokens by
prompt seed since this work) and each run's text are in the main checkout's
`.scratch/fu310/`, a directory per run.

**1. The quiet-host control** (16:39-16:51, the timing lock held, no build on
the host; `cpu.csv` beside the control: median 16%, max 26.5%):

| run | width 3 before any arrival | after E ended (control) / after C came back (moved) |
|---|---|---|
| control | 175.9 misses, ITL p50 23.34 ms | 243.7 misses (+38.5%), 26.17 ms (+12.1%) |
| moved (KV-RAM), run 1 | ITL p50 23.00 ms | 32.20 ms (+40.0%) |
| moved (KV-RAM), run 2 | 22.97 ms | 31.92 ms (+39.0%) |

The control beside the kernel build had read 183.5 -> 254.1 (+38%).

**2. The text.** Every request's generated tokens, by its prompt's seed:

- The two moved runs generated the same text, every request, token for
  token, and the same misses step for step (as the 2026-10-08 timelines
  already show between themselves).
- The moved run and the control part where their batches part: B1 at its
  token 326 and B2 at 476 (decoding at width 2 while C was out in the moved
  run, beside E0 and E in the control), E at its token 2. C's text is the
  same in both through token 1,420: C left the device at its token 244, came
  back 350 steps later from a 1.13 GB blob, and went on generating the
  unmoved run's text for 1,176 more tokens.
- How varied each text is, distinct tokens per 200:

  | lane | moved runs | control |
  |---|---|---|
  | C | 200 in every window | 200 in every window |
  | B1 | 120, 59, then 38 (a loop) | 120, 59, then 33 (a loop) |
  | B2 | 107-127 throughout | 127, 107, 56, then 3 (a three-token loop), one excursion to 33-88 |

- The cache's own numbers, moved run against control, after C's restore /
  after E: projections read ~2,700 a round in both, prefetches issued 175 a
  round in both, hit rate 0.87 against 0.91.

**3. The same text, nothing moved.** `IGNIS_AC37_FORCE` makes every request
generate a recorded text, one forced token a round. A forced round is never
a captured graph (`program.cu`: `use_graph = ... && !constrained`), so forced
runs' ITL are not compared; their misses are. Width-3 rounds after C's
restore, against the unmoved run at the same B1 token:

| pair | all rounds | 200-round windows | 50-round windows |
|---|---|---|---|
| free moved run vs the free control (different text) | 340 vs 240, +41.7% | | |
| free moved run vs the control forced to its text | 340 vs 338, +0.6% (1,409 rounds) | -2 to +5% | -4 to +6% (run 2) |
| forced moved run vs that control | 337 vs 338, -0.3% (1,455) | | -8 to +7% |
| AC 43's test (both forced, one process, caches +0.05% apart) | 356 vs 355, +0.3% (1,456) | -1.4 to +2.5% | -5.2 to +9.7% |
| AC 43's test on a second text, the free control's (caches -0.71% apart) | 257 vs 260, -1.2% (1,452) | -3.9 to +3.3% | -9.1 to +10.5% |

The forced control's rounds before any arrival missed what the free
control's did (175.9 and 175.9: the same text up to there), and the forced
moved run what the free one did (305-367 against 304-379 per 200 of B1's
tokens): forcing changes the route, not the misses.

The first rounds after the restore, mean misses a round against the unmoved
run at the same tokens, the four same-text pairs in the table's order:

| rounds after the restore | 0-4 | 5-9 | 10-19 | 20-49 | 50-99 |
|---|---|---|---|---|---|
| free moved vs forced control | +28% | +4% | +6% | -2% | -2% |
| forced moved vs forced control | +9% | +1% | +3% | -2% | -2% |
| AC 43's test | +21% | +7% | +3% | -3% | -2% |
| AC 43's test, second text | +32% | +8% | +10% | 0% | -5% |

Ten rounds are too few to read: windows of ten spread -14 to +16% either way
well after the restore.

A first AC 43 run measured -4.8%: its second load (the control) planned an
expert cache 2.3% smaller (23,176 slots against 23,721 alone), because the
first load's CUDA context was already in the free memory it read, and missed
6% more before any arrival (187.0 against 175.9). The test now holds one
context across both loads and gives the control's budget its extra chunk of
pool; the rerun's caches were 0.05% apart.

**4. Without the arrivals.** The control forced to the moved run's text, run
without E0 and E (`IGNIS_AC37_NO_ARRIVALS=1`), against the control with them,
at the same B1 tokens from where E ended: +0.6% over 1,634 rounds, 200-round
windows -0.4 to +1.9%, the first 50 rounds 260 against 260. Before and after
"E" in the run without it: 174.6 -> 316.0 misses a round.

**5. The KV-disk spill paced.** Its device copies now go a 12 MiB slice at a
time, one on the link, at most one new an advance. The disk leg twice
(the other lanes against B1' and B2' at width 2, taken last):

| leg | duration | steps | ITL p50 | outside the stall |
|---|---|---|---|---|
| disk move out, 32 MiB windows (the first finding) | 650 ms | 47 | +12.1% | |
| disk move out, 12 MiB slices | 1,315 / 1,299 ms | 103 | +7.1 / +7.1% | +2.7 / +3.4% |
| disk move in (16 MiB feeds) | 1,260 / 1,268 ms | 70 | +57.9 / +59.9% | +2.8 / +2.9% |

The KV-RAM legs: move out +6.6 and +5.6% ITL p50 (quiet host, block A),
move in +6.1 and +7.2% outside the stall (AC 43's forced legs).

## Finding

Observed:

- **A move adds no expert misses.** Made to generate the same text, the
  rounds after a 236K-token sequence's restore miss what the unmoved run's
  rounds miss at the same tokens: +0.6%, -0.3%, +0.3% and -1.2% over four
  pairs on two texts, every 200-round window within -4 to +5%, from the
  first round after the restore -- but for its first rounds: rounds 0-4
  miss +9 to +32% more, rounds 5-9 +1 to +8%, rounds 10-19 +3 to +10%, and
  from round 20 the two are level (-3 to 0% over rounds 20-49).
- **Nor does the long arrival.** The same text with and without E0 and E
  misses alike from the round E ended on (+0.6%). The rise from before the
  arrivals to after them -- +38% on the control's own text, +89% on the
  moved run's -- is there without any arrival (174.6 -> 316.0 a round):
  it is the lanes' text moving on.
- **The misses are the text's.** A round's misses are those of what its
  lanes generate, and greedy decode is not batch-invariant, so the moved run
  and the control generate different text from the round their batches
  part; from there their misses are not comparable. The "doubling" the first
  runs read was B2's text: varied in the moved run, a three-token loop in
  the control.
- **The three moved runs the first finding counted were one text.** A run
  repeats exactly; it is one sample in text, not three.
- **The first finding's hypotheses fall on their own data too.** After the
  restore the misses held flat at ~330 a round for 1,400 rounds, where the
  cache (~23,700 slots) turns over in ~70 rounds at that rate; the
  projections read and the prefetches issued a round were the moved run's
  and the control's alike.
- **C's restore holds at 236K tokens.** Its greedy text across the move
  equals the unmoved run's for 1,176 tokens past the restore, under other
  batch widths.
- **Paced, the KV-disk move out is within its bound.** +7.1% ITL p50 twice
  at 12 MiB slices (+12.1% at 32 MiB windows); it takes twice as long (1.3 s).
  Its move in, and KV-RAM's, add +3-7% to a round outside the expert stall.

Inferred, not measured:

- The first rounds' extra misses are C's own experts coming back into a
  cache that dropped them while C was away (~30-75 misses a round for five
  rounds, fewer for the next fifteen: some 10-20 ms of stall in all, at the
  ~50 µs a miss these rounds stall).
- C's text, which never repeats a token, costs most of a width-3 round's
  misses: ~166 a round with B1 and B2 early in their texts, against ~25 for
  a fresh pair at width 2.
- In a free comparison the second of two loads in one process plans a
  smaller expert cache for as long as the first's CUDA context lives; only
  AC 43's test loads twice, and it now shares the context.

## Implications

- Nothing in the expert residency needs fixing for a live move.
- AC 37's move-in bound reads what the transfer adds -- a step's time
  outside its expert stall and transfer passes, the owner's decision of
  2026-10-08 -- with the raw ITL and the expert stall a miss cost printed
  beside it. The width-3 before/after line no longer reads as the move's
  cost.
- AC 43 holds the rounds after a restore to the same text unmoved (a
  committed text, `crates/server/tests/fixtures/ac37_text.json`): rounds
  20-49 within +10% in mean misses (N = 20), and each full 200-round
  window's median within +10% from the first round. Re-warming the restored
  sequence's experts could take the first rounds' ~10-20 ms; not worth a
  mechanism at that size.
- Any comparison whose cost depends on what lanes generate needs the same
  text on both sides; a free control is another text once the batches part,
  and repeated free runs of one scenario are one sample.

## Limits and unknowns

- One host, one scenario, synthetic prompts whose texts include loops; four
  same-text pairs over two texts.
- A forced run takes the eager route; its misses matched the free run's on
  the same text, its ITL is not a serving number.
- Whether the move in's copies slow the expert copies they share the link
  with (the stall a miss costs) was not read in these runs; the harness
  prints it from now on.
- The no-disk KV-RAM loop still gives up every live snapshot it may for a
  blob that fits the tier but not the arena beside one moving back in
  (GitHub #310 refuses only a blob larger than the whole tier up front).

## Follow-ups

- None for residency. AC 43's bound and N are the owner's to confirm.
