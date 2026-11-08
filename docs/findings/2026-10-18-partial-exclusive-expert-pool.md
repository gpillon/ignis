# A partially exclusive expert pool on the 5090: pinning the calibration-hottest 1-2 GB of experts gives that much host RAM back for under 1% of decode; past 4 GB the cost climbs steeply

- Kind: experiment
- Status: corrected
- Observed: 2026-10-18
- Last verified: 2026-10-18
- Scope: core / Flash-Next expert residency, the CPU policy model with experts pinned in VRAM outside the LRU; the host plan's pinned expert pool on the rtx5090 profile
- Topic: flash-next
- Related: [GitHub #329](https://github.com/gpillon/ignis/issues/329) (item 12),
  [spec flash-next/03](../specs/flash-next/03-expert-residency.md),
  [Expert residency replayed on the study's routing](2026-10-05-expert-residency-replayed-on-the-study-s-routing.md),
  [Flash-Next decode headroom](2026-10-08-flash-next-decode-headroom.md),
  [ADR 0030](../adr/0030-device-memory-reserved-at-load.md)
- Superseded by: none
- Corrected by: [A pinned hot set measured on the 5090](2026-10-18-pinned-hot-set-measured-on-the-5090.md)
- Summary: Replayed, pinning the calibration-hottest 1-2 GB of Flash-Next experts outside the LRU frees that host RAM for under 1% of decode; measured on the 5090, 4 GiB costs 4.3%/5.9% at one/three lanes, above the replay's bracket.

> **Corrected 2026-10-18.** Measured on the 5090, 2 GiB costs 0.2-1.1% of three-lane decode and 4 GiB 4.3% of one-lane and 5.9% of three-lane decode, above this replay's bracket ([the pinned hot set measured](2026-10-18-pinned-hot-set-measured-on-the-5090.md)).

## Question

On the RTX 5090 the VRAM expert cache holds about half of Flash-Next's
37.8 GB of experts. Every projection also sits in the pinned host pool, so
the pool takes ~38 GB of RAM. A fully exclusive pool (RAM holds only what
VRAM does not, and each victim is written back) was measured costly on this
Gen 3 host: the write-back takes link bandwidth from the miss copies almost
byte for byte (`.scratch/residency-swap-5090/REPORT.md` in the main
checkout).

The untested variant is a **partially exclusive** pool. Projections that stay
in VRAM for a load's whole life are pinned there and never evicted, so they
need no host copy and never need a write-back. RAM saves their bytes; the
LRU loses their slots. Three questions:

1. How large is the set that today's policy never evicts at the 5090's cache?
2. For pins of a given size, what happens to the hit rate of the LRU around
   them?
3. What does that cost in decode time on this host, per GB of RAM saved?

## Evidence

**Data.** The converter's own routing traces of the served artifact
(`IGNIS_FLASH_NEXT_TRACES=F:/ai/models/Qwen3.8-Flash-Next-ignis`):

- quantized-stream routing, teacher-forced;
- 66 test chunks over eleven domains, 131,978 tokens, 48 layers × 512
  experts;
- the lookahead router's top 20 per token;
- the converter's K map, class record sizes and calibration traffic
  (`work/converter.json`).

The experts total 37,795,446,784 B.

**Model.** The policy model (`crates/core/src/residency/policy.rs`) is
unchanged. `crates/core/tests/expert_residency_partial_exclusive.rs` wraps
it in a pin:

- **Pinned set.** Whole experts (both projections), taken in
  `warm_start_order` (calibration selections, hottest first) until they
  reach the target bytes. Each class keeps at least its three-lane minimum
  of 78 LRU slots, and an expert that would breach that is skipped.
- **The LRU around them.** Each class's slots shrink by the pinned
  projections in that class, and the warm start fills the rest of the
  order.
- **A step.** The model gets the selection without the pinned experts. Each
  lookahead lane is cut to W first and then loses its pinned experts, since
  the device also skips a resident candidate before the budget. A pinned
  use counts as a hit.
- **Its tests.** With nothing pinned, the wrapper returns the model's own
  `StepOutcome` at every step and ends with the same counters (synthetic
  test). A pinned expert is never copied, prefetched or evicted (second
  synthetic test).

**Configuration.** Today's served policy:

- W = 16, `default_prefetch_budget()`;
- pools split by `plan_expert_cache` as `flash_next::build_residency` asks
  for them (three lanes' minimum, the 8,192-token chunk's tables).

The cache is **18,443,439,440 B**. That is the median of the seven served
plans at the rtx5090 profile's shape (589,824 tokens, 3 lanes) in the
owner's server logs, 2026-10-12/13 (18.03-18.76 GB). At that size the
replay's plan prints a 60.5% calibration-only expected hit rate, the same as
the served plan's logged `expected_hit_rate` 0.6054. The pools hold
18,443,272,192 B, 48.8% of the experts.

**Protocols.**

- **One serving life:** one model over every test chunk in id order, every
  token. Three lanes decode consecutive chunks together, each group for its
  shortest chunk's length: 6,334,944 layer steps at one lane, 2,075,136 at
  three.
- **Per chunk:** the headroom finding's protocol, each chunk (or chunk
  triple) from the warm start, 256 tokens to settle and 512 measured.

**Anchor.** At the headroom finding's 15,615,494,736 B cache with its
protocol, the wrapper with nothing pinned reproduces that finding's replay
exactly:

- one lane: 94.18% hit, 40.3 + 33.5 MB per token;
- three lanes: 90.55% hit, 65.9 + 37.9 MB per token.

**Command.**
`IGNIS_FLASH_NEXT_TRACES=F:/ai/models/Qwen3.8-Flash-Next-ignis cargo test --release -p ignis-core --test expert_residency_partial_exclusive -- --ignored --nocapture`.
It runs in 109 s on 16 threads. The raw output is
`.scratch/partial-exclusive/study-run1.txt` in the
`partial-exclusive-study` worktree.

### 1. What today's policy never evicts (one serving life, no pin)

Of the 22,001 warm-started projections (18.44 GB), the ones not yet evicted
by a point in the life:

| through | 1 lane | 3 lanes |
|---|---:|---:|
| 10% | 1,254 (1.66 GB) | 1,141 (1.47 GB) |
| 25% | 690 (0.99 GB) | 526 (0.74 GB) |
| 50% | 510 (0.73 GB) | 339 (0.48 GB) |
| 75% | 374 (0.55 GB) | 268 (0.38 GB) |
| 100% | **366 (0.53 GB)** | **263 (0.38 GB)** |

- **Class.** Every never-evicted projection is K = 4: gate/up K4 0.47 /
  0.32 GB and down K4 0.07 / 0.06 GB, at one lane / three lanes.
- **Overlap with calibration.** The calibration-hottest experts are mostly
  not in this set. Of the calibration top 0.5 / 1 / 2 / 4 GB, the trace
  never evicts this share of the bytes:
  - one lane: 26.5 / 23.9 / 19.3 / 12.6%;
  - three lanes: 20.2 / 17.5 / 13.3 / 8.9%.

### 2. Pins by calibration order: hit rate and demand bytes (rtx5090 cache)

One serving life. The hit rate includes the pinned uses. The demand is
bytes copied on a miss.

| pin = RAM saved | experts | 1 lane: hit, demand MB/token | 3 lanes: hit, demand MB/round |
|---|---:|---|---|
| 0 (today) | 0 | 95.70%, 29.0 | 93.64%, 129.1 |
| 0.50 GB | 205 | 95.69%, 29.0 | 93.63%, 129.3 |
| 1.00 GB | 411 | 95.69%, 29.0 | 93.62%, 129.6 |
| 2.00 GB | 845 | 95.66%, 29.3 | 93.56%, 131.2 |
| 4.00 GB | 1,833 | 95.45%, 31.4 | 93.17%, 141.7 |
| 6.00 GB | 2,917 | 95.04%, 35.5 | 92.50%, 159.3 |
| 8.00 GB | 4,018 | 94.35%, 42.3 | 91.51%, 185.6 |

- **Prefetch bytes barely move:** 29.6-31.5 MB per token at one lane,
  112.1-113.1 MB per round at three.
- **The per-chunk protocol agrees.** Its hit rates are 0.2-0.4 points
  higher, and its demand ratios to no pin are within 0.045 of these.
- **Use share.** The pinned experts serve 5.0 / 7.9 / 12.8 / 21.1 / 28.4 /
  34.9% of all uses at 0.5-8 GB.

### 3. Decode cost on this host

The measured miss cost comes from
[Flash-Next decode headroom](2026-10-08-flash-next-decode-headroom.md) §1
and §4, served at a 15.6 GB cache. The demand copies run at 12.0 GB/s (one
lane) and 11.9 GB/s (three), all on the critical path:

- **One lane:** 1.84 ms of stall per 9.85 ms round (22.1 MB of demand).
- **Three lanes:** 15.0 ms of stall per 27.5 ms round (178.8 MB of demand).

The cost of a pin is therefore its added demand bytes at 12 GB/s. The
headroom finding's own method gives a cross-check: the served stall share
times the replayed demand ratio.

| pin = RAM saved | 1 lane: +ms/round (of 9.85) · ratio method | 3 lanes: +ms/round (of 27.5) · ratio method |
|---|---|---|
| 0.5 GB | +0.00 (0.0%) · 0.0% | +0.02 (0.1%) · 0.1% |
| 1 GB | +0.00 (0.0%) · 0.0% | +0.04 (0.2%) · 0.2% |
| 2 GB | +0.02 (0.3%) · 0.2% | +0.18 (0.6%) · 0.9% |
| 4 GB | +0.20 (2.1%) · 1.6% | +1.05 (3.8%) · 5.3% |
| 6 GB | +0.54 (5.5%) · 4.2% | +2.52 (9.2%) · 12.8% |
| 8 GB | +1.11 (11.3%) · 8.6% | +4.71 (17.1%) · 23.9% |

**The host plan.** A Flash-Next load on this host (2026-10-13, 0.6.2)
logged:

- 48.4 GB available at load;
- the 37.8 GB pool, n-gram hot rows 1 GiB, KV-RAM arena 2 GiB, retained
  host slots 1.04 GB, KV-disk staging 64 MiB;
- 6.3 GB left, against a 4 GiB margin.

Test loads on 2026-10-18 (0.7.0) had 43.0-43.4 GB available. They used
2-3 lanes, a 32-48K-token KV pool and a 24-26 GiB VRAM budget, not the
profile. They ran with n-gram hot rows of 0-512 MiB, no arena and no
retained slots, and left 4.7-5.6 GB.

**The pools' size is fixed per load.** `flash_next::build_residency` sets
them once (`ResidencyDesc.capacity`), and nothing in
`crates/core/src/residency/` resizes them during a load (ADR 0030: device
memory is reserved at load). A model switch or an idle unload reloads
everything from the artifact.

## Finding

**Observed (replay).** The set today's policy never evicts at the rtx5090
cache is small and still shrinking at the end of the traces:

- 0.53 GB at one lane and 0.38 GB at three, all of it K = 4 projections;
- 1.5-1.7 GB survive the first 10% of the life, and the count still falls
  between 75% and 100%.

It is not the calibration-hottest set either: three quarters or more of the
calibration top 1 GB is evicted at some point. In the strict sense the
issue asked for, a stable hot set would save at most ~0.4-0.5 GB.

**Observed (replay).** Pinning by calibration order costs little even though
those experts are not "never evicted":

- **Up to 1 GB:** demand is unchanged within 0.4%.
- **2 GB:** +1.0% demand at one lane, +1.6% at three.
- **4 GB:** +8-10%.
- **6 GB:** +22-25%.
- **8 GB:** +44-50%.

The pinned experts serve 8-13% of all uses at 1-2 GB, so their own misses
disappear. That roughly offsets the misses the smaller LRU adds, until the
LRU's loss dominates past ~2 GB.

**Inferred (the measured miss cost applied to the replay).** At three lanes,
the profile's default and the link-bound case:

| pin | cost | GB of RAM per % of decode |
|---|---|---|
| 1 GB | ~0.2% | ~5 |
| 2 GB | 0.6-0.9% | ~2-3 |
| 4 GB | 3.8-5.3% | ~1 |
| 6 GB | 9-13% | ~0.5-0.65 |
| 8 GB | 17-24% | ~0.33-0.47 |

One lane costs about half as much. A fully exclusive pool would give ~18 GB
back for +20-70% per round (the swap report's inference).

**Inferred (verdict for the rtx5090 profile).** The only point worth having
is a pin of **1-2 GB**: under 1% of decode at either lane count. That RAM
would double the 2026-10-13 load's 1 GiB n-gram hot rows line, or pay for
its 2 GiB KV-RAM arena on a load that has less RAM available. Past 2 GB, each further GB costs
another 1.5-6% of three-lane decode, more than it is likely to buy on a
64 GB box. Whether 1-2 GB of RAM pays for the implementation is the owner's
call. Nothing here makes it necessary.

## Implications

What a pin needs, if built:

- **Host pool.** A pool laid out without the pinned records. Today it is
  one `cudaHostAlloc` of the whole catalog (`pool_layout`,
  `fill_expert_pool`).
- **Load path.** Pinned records streamed from the artifact straight into
  their VRAM slots through a bounded staging buffer.
- **Device and model.** Static slots outside the class LRUs on the device,
  and the same pin in the policy model, which the GPU must match (spec
  flash-next/03 acceptance 4). This study's wrapper is that model's shape:
  capacity minus the pins, the selection and lookahead without them.
- **Choosing the set.** By calibration order, known at load. The
  trace-observed never-evicted set is neither needed nor stable.
- **No write-back, ever.** That holds only while a load's pools are fixed in
  size. Any future live resize of the expert cache would have to re-read
  unpinned victims from the artifact, not from VRAM.

## Limits and unknowns

- **Replay.** Teacher-forced routing over 132K tokens, about 20 minutes of
  decode at the served rates, not a load's whole life. The never-evicted set keeps
  shrinking with length, so a longer life would make it smaller, not
  larger.
- **Decode only.** Prefill is not replayed. Its admission is scan-resistant
  and never evicts, and a pin keeps the resident bytes the same, so a
  first-order change is not expected; it was not measured.
- **The cost conversion is an inference.** It assumes a demand byte costs
  the same on the critical path at 18.4 GB as it did at 15.6 GB. Per byte
  that held across the headroom finding's runs (11.9-12.4 GB/s). The
  percentages divide by the 15.6 GB rounds. At 18.4 GB the rounds are
  shorter, and the decode fusion since 2026-10-08 shortened the one-lane
  round further, so these percentages run slightly low. The ratio method
  applies a 15.6 GB stall share to an 18.4 GB demand ratio, so it runs high.
  The two methods bracket the cost.
- **One pin order.** Calibration selections, as the warm start orders them.
  Ranking by traffic per byte, per-class targets or re-splitting the
  remaining pools was not tried. The 1-2 GB answer is already near free.
- **No measurement of the RAM's value.** Not what a larger n-gram hot-row
  line or the KV-RAM arena would buy, and not the load-time cost of
  streaming pinned records from the file.
- **One cache size.** The served 18.03-18.76 GB range was not swept.
- **No GPU run.**

## Follow-ups

- If the owner wants the RAM: a spec for the pin, sized 1-2 GB by default as
  a config field, with this study's wrapper becoming the policy model's pin
  and a GPU match test. **Done 2026-10-18**, a field off by default
  (`--vram-pinned-expert-bytes`, ADR 0055, accepted with the `rtx5090`
  profile pinning 2 GiB for Flash-Next): measured on the card in
  [A pinned hot set measured on the 5090](2026-10-18-pinned-hot-set-measured-on-the-5090.md).
  The wrapper above differs from the device in one detail, at several lanes:
  it removes pinned experts from each ranking before interleaving the lanes,
  where the device keeps a pinned candidate's rank and skips it; at one lane
  the two agree step for step (`the_policy_model_s_pin_is_the_study_s_wrapper`).
- The replay loaders in `expert_residency_study_replay.rs` and
  `expert_residency_partial_exclusive.rs` are copies; share them if a third
  study needs one.
