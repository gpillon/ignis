# Growing a sequence's KV one page at a time moves the 27B's decode ITL p99, so P0 picks the fixed branch

- Kind: experiment
- Status: current
- Observed: 2026-10-08
- Last verified: 2026-10-08
- Scope: kernel / paged KV mapping, `ignis_seq_grow`; serving / decode round latency, spec vram-budget/03 P0 (ACs 26-27)
- Related: https://github.com/gpillon/ignis/issues/309, [ADR 0045](../adr/0045-the-kv-pool-follows-residency-and-state-goes-down-to-disk.md), spec [vram-budget/03](../specs/vram-budget/03-kv-pool-policy-and-kv-disk.md) (P0, ACs 26-27), [Flash-Next on the 5090](2026-10-06-flash-next-on-the-5090.md) (its lanes-at-262K harness)
- Superseded by: none

## Question

ADR 0045's gate, P0: can a sequence map its KV pages as it grows (one page
each time decode crosses into one, each prefill chunk's pages before the
chunk) instead of all of them at allocation, at no decode cost beyond the
run's own noise? The owner's rule: growth passes when, at one and at three
active lanes on both models, its decode tok/s is at least 99.5% of the mean
of the two whole legs, and its ITL p99 is at most the higher whole leg plus
the difference between the whole legs. Otherwise P3 builds the fixed branch.

## Evidence

**Where it lives.** The spike is not merged. Everything below is on the local
tag `kv-p0-grow-spike` (7ea0b8e, not pushed):
- 630e6cc: the extend op, its CTest and the latency bench;
- eb012b1: the switch;
- d729417: the A-B-A harness, `tools/kv-p0-gate/`;
- 7ea0b8e: `leg.sh` reads its scripts from its own directory.

A copy of `tools/kv-p0-gate/`, the frozen server binary and the raw data
(one directory per leg, the summaries, the latency log) are in the main
checkout's `.scratch/kv-p0/`, which is untracked and local to that clone. The
harness has no tests of its own; it stays on the tag.

**The spike.**
- `ignis_seq_grow(pool, seq, context_tokens)` (`kernel/src/seq.cu`) maps a
  live sequence up to `context_tokens`: `set_page_entitlement`, then
  `materialize_pages` (which publishes the new block-table range), then
  `zero_pages` over the new pages only, on the default stream as
  `ignis_seq_alloc` does. A short pool, a target past `max_context_tokens`
  and a claimant or lender are refused with nothing changed.
- `KvGrowth` in `RuntimeCompute`, set by `IGNIS_KV_GROWTH=whole|page`. Under
  `page` a sequence is allocated at its first prefill chunk and grows before
  each later chunk. Before each decode round every lane maps as far past its
  frontier as the round can write (the draft window plus the anchor on the
  27B), so a verify round never shortens its drafts. Claims and restores
  are mapped whole. Each request logs its growth count at release.
- Equal work, kernel (AC 26): `test_seq_grow` feeds a whole sequence and two
  grown ones the same real K/V rows through A2. The grown ones' physical
  pages interleave. They end with the same bytes by logical page in every
  plane, under BF16 and hq-e8-2b at 4 and at 2 KV heads. Every growth keeps
  the history and zeroes only its new pages. Each refusal changes nothing.
  Two mutations (zero the whole row, zero nothing) turn it red.
- Equal work, end to end: at one lane the greedy text is byte-identical
  across all legs of each model (sha256 `edbc3f59…` on the 27B, `abae1c6c…`
  on Flash-Next). So is the 27B's acceptance (0.3369) and Flash-Next's decode
  hit rate (0.9585). The growth legs logged 28 growths per 1,800-token
  request (the pages crossed) and no leaf error.

**Extend latency** (`seq_grow_latency_gpu`, release kernel, card otherwise
idle). Host wall time per call in µs: as issued, which is what a decode
round waits for, and with the device drained after it. Pools at each
model's `make config` geometry, hq-e8-2b, retained host slots left out.

| model | grown by | calls | issued p50 / p99 | drained p50 / p99 |
|---|---|---:|---:|---:|
| 27B (16 GQA layers, 64 planes) | 1 page | 8,191 | 452 / 831 | 465 / 890 |
| 27B | 16 pages (a 1,024-token chunk) | 1,022 | 684 / 995 | 698 / 999 |
| Flash-Next (12 layers, 48 planes) | 1 page | 4,095 | 383 / 743 | 394 / 852 |
| Flash-Next | 128 pages (an 8,192-token chunk) | 1,023 | 611 / 1,081 | 621 / 1,086 |

Whole allocation at the context, for scale, drained: 4.17 ms p50 for the
27B's 524,288 tokens and 1.78 ms for Flash-Next's 262,144.

**The A-B-A legs.**
- Binary: one frozen release build for every leg. A leg is one launch with
  `IGNIS_KV_GROWTH` set.
- Flags: `make config`'s, except for these departures from AC 8's "at the
  defaults":
  - `--prompt-reuse off` on every leg. The spike grows only a sequence that
    owns its whole row: a claim of a prefix or a checkpoint is mapped whole.
    With reuse on, the growth legs' repeated prompts would be claims and
    would grow almost nothing.
  - `--kv-host-pool-bytes 0 --retained-host 0` on Flash-Next. The host had
    ~42 GB free against the ~46 GB a default load plans, and neither the
    KV-RAM arena nor a retained slot is on this path.
  - `--metrics` on every leg, for acceptance and hit rate.
- Speculation: the 27B on its default decode route (DFlash2, 7 drafts),
  Flash-Next with speculation off.
- Each leg runs a warm-up at 3 lanes, then 1 lane × 8 requests (27B) or 5
  (Flash-Next), then 3 lanes × 5 (27B) or 3, then 3 prefills. Requests are
  greedy, `ignore_eos`, `max_tokens` 1,800, on #306's greedy prompts (the 3
  lanes differ by one sentence).
- tok/s of a request: `(completion_tokens − 1) / (last − first chunk)`. The
  27B streams a verify round's tokens in one chunk, so its "ITL" is the time
  between rounds. Both modes are measured the same way.
- 1 lane: the mean over the requests. 3 lanes: every lane's tokens inside
  the window where all three are in flight. ITL p99 is pooled over every
  chunk interval from chunk 50.
- CPU: total load sampled every 5 s. From run 3 on (also an addition to the
  spec's harness) a measured block waits for < 30% and is re-run when its
  mean reaches 30% or a sample reaches 60%; the last attempt is used.

**27B, run 3**: the only run whose six measured blocks were all quiet
(12-15% CPU). Nothing was re-run.

| lanes | whole 1 | growth | whole 2 | growth / whole mean | rule |
|---|---:|---:|---:|---:|---|
| 1, tok/s | 214.58 | 214.65 | 214.87 | 0.9997 | pass |
| 1, ITL p99 ms | 16.34 | **16.41** | 16.29 | bound 16.39 | **fail** |
| 3, tok/s | 450.83 | **453.23** | 469.83 | **0.9846** | **fail** |
| 3, ITL p99 ms | 20.44 | **20.87** | 20.40 | bound 20.48 | **fail** |

- 1-lane ITL p50 is 15.62 / 15.61 / 15.61 ms.
- 3-lane acceptance is 0.2859 / 0.2874 / 0.3022.

**27B, runs 1 and 2, not used for the verdict.** Other agents' builds
loaded the CPU during whole legs.
- Run 1 used an earlier harness:
  - 1 lane × 5 requests and 3 lanes × 3 per lane;
  - a sampler that was never stopped, so its file spans all three legs;
  - no block markers and no gate.
  
  Whole 1's CPU averaged 47.4% (max 100%) over the leg, and read 100% from
  about 30 s to 175 s after its launch, across nearly all its measured
  blocks.

  | lanes | whole 1 | growth | whole 2 |
  |---|---:|---:|---:|
  | 1, tok/s | 210.99 | 213.66 | 214.80 |
  | 1, ITL p99 ms | 32.35 | 16.52 | 16.29 |
  | 3, tok/s | 427.58 | 419.32 | 403.40 |
  | 3, ITL p99 ms | 47.17 | 20.68 | 20.43 |

  3-lane acceptance was 0.2748 / 0.2656 / 0.2359.
- Run 2 had block markers but no gate yet. The whole legs' 1-lane blocks
  were loaded: whole 1 63.4% mean (max 100%), whole 2 42.6% (max 100%),
  growth 12.0%. Its 3-lane blocks were all quiet by the gate's own
  threshold (means 22.4 / 12.9 / 18.0%, max 36 / 17 / 48%), **and growth
  passes both 3-lane cells there**: tok/s 1.0005 of the whole mean, ITL p99
  20.76 against a bound of 21.03.

  | lanes | whole 1 | growth | whole 2 |
  |---|---:|---:|---:|
  | 1, tok/s | 210.49 | 214.51 | 212.08 |
  | 1, ITL p99 ms | 26.90 | 16.48 | 21.91 |
  | 3, tok/s | 447.47 | 433.94 | 420.00 |
  | 3, ITL p99 ms | 20.78 | 20.76 | 20.52 |

  3-lane acceptance was 0.2778 / 0.2698 / 0.2575.

**Flash-Next** (CPU 13-19% in every measured block used).

| lanes | whole 1 | growth | whole 2 | growth / whole mean | rule |
|---|---:|---:|---:|---:|---|
| 1, tok/s | 98.24 | 98.08 | 98.18 | 0.9987 | pass |
| 1, ITL p99 ms | 17.81 | 17.84 | 17.99 | bound 18.16 | pass |
| 3, tok/s | 138.55 | 138.02 | 136.90 | 1.0021 | pass |
| 3, ITL p99 ms | 37.01 | 36.60 | 36.78 | bound 37.24 | pass |

- 1-lane ITL p50 is 9.5 ms in every leg.
- Decode hit rate is 0.9585 at 1 lane and 0.936 at 3 lanes in every leg.
- **Only one block in any run was re-run under the gate: this growth leg's
  3-lane block.** No whole block tripped it. The first two attempts were
  loaded and would have failed the rule:
  - attempt 1 (mean 28.8%, max 76%): 136.18 tok/s (0.9888) and ITL p99
    38.38 ms, both failing;
  - attempt 2 (mean 26.4%, max 79%): 136.31 tok/s (0.9897) failing, ITL
    p99 36.98 ms passing.

  The quiet third attempt is the pass in the table. The re-run turned a
  failing cell into a pass.

**Prefill, recorded only.** The figure is the TTFT of a multi-chunk prompt
times chunk / prompt tokens: s per chunk-equivalent, three prompts per leg,
not a measured chunk wall time. It folds in:
- the partial last chunk;
- the first decode round and the request overhead;
- on a whole leg, the allocation of the whole context (4.2 ms on the 27B,
  1.8 ms on Flash-Next), where a growth leg allocates the first chunk and
  grows per chunk.

| model | whole 1 | growth | whole 2 |
|---|---|---|---|
| 27B, ~8.5K tokens, 1,024-token chunks (run 3) | 0.113 / 0.098 / 0.095 | 0.120 / 0.096 / 0.098 | 0.131 / 0.095 / 0.098 |
| Flash-Next, ~34.6K tokens, 8,192-token chunks | 2.70 / 2.64 / 2.63 | 2.70 / 2.64 / 2.66 | 2.74 / 2.65 / 2.65 |

## Finding

**Verdict: the fixed branch, by the rule applied to run 3.** Growth fails
three of the 27B's four cells there. Flash-Next passes all four. The rule
needs every cell on both models. The margins are thin, and the 3-lane cells
do not reproduce between runs (below).

Observed:
- **Decode throughput is unchanged at one lane on both models.** It is
  within 0.13%, with the same text, acceptance and hit rate.
- **The 27B's 1-lane ITL p99 is higher under growth in each run.**
  - Run 3: 16.41 ms against quiet whole legs of 16.34 and 16.29, which is
    0.02 ms past the bound.
  - Run 2: 16.48. Both whole legs were loaded (26.90, 21.91), so it is
    compared only with run 3's whole legs, 0.14-0.19 ms higher.
  - Run 1: 16.52 against whole 2's 16.29. Whole 1 was loaded.
  - The 1-lane p50 does not move in any run.
- **The 27B's 3-lane cells flip between the two quiet 3-lane
  measurements.** In run 3 growth fails both: ITL p99 +0.4 ms, tok/s 98.5%
  of the whole mean. In run 2 it passes both. tok/s follows the
  speculative acceptance, which drifts between legs (0.286 / 0.287 / 0.302
  in run 3; whole 1 > growth > whole 2 in runs 1 and 2).
- **On Flash-Next the same stall does not show** in the quiet blocks. Its
  ITL p99 (~18 ms at 1 lane) sits far above its p50 (9.5 ms), set by rounds
  that are already slow. Under a loaded CPU its growth 3-lane block failed
  twice; it passed when quiet.
- **The extend op's cost is per call, not per page.** One page costs
  450 µs on the 27B; 16 contiguous pages cost 680 µs. Issued and drained
  times are nearly equal, so the host pays it, not the device.

Inferred, not measured:
- The 450 µs is mostly launch overhead of `zero_pages`'s one memset per
  plane: 64 planes on the 27B, about 7 µs each; 48 on Flash-Next, about
  8 µs each. A decode round that crosses a page pays it before its launch.
  - On the 27B at 1 lane that is 28 rounds of 536 per request (one in
    nineteen); at 3 lanes about three times as often.
  - The 27B's round time is tight (p50 15.6, p99 16.3 ms at 1 lane), so
    stalled rounds enter its top 1%.
- One page at a time is P0's worst case by design. The growth branch would
  grow by 32 pages, so it would pay the per-call cost once per 2,048 tokens
  instead of once per 64. At that rate it would likely stay out of the 27B's
  p99. **This was not measured.** It cannot change this gate's verdict; it is
  what an owner reconsidering the branch would measure first.

## Implications

- P3 builds the fixed branch (ADR 0045): the prompt plus the explicit cap or
  `--default-max-tokens` is reserved at admission, and only an admission
  moves a sequence. The leaf keeps today's whole mapping, and `ignis_seq_grow`
  stays on the tag.
- If the growth branch is reopened, the cost to remove is the per-call host
  time, not page count. Two levers would cut it: zero a page's planes in one
  launch rather than one memset per plane, and grow by the branch's 32-page
  step. Re-gate at the step the branch would use, on a quiet machine.
- The 27B is the stricter model for any change on the decode round's host
  path. Its round time is tight enough that a ~0.45 ms stall in 5-15% of
  rounds moves its p99 by 0.1-0.4 ms, which Flash-Next's slow-round tail
  hides.
- A timed A-B-A on this host needs the CPU quiet. A concurrent build moved
  the 27B's 1-lane ITL p99 by 5-16 ms and Flash-Next's 3-lane tok/s by ~1.2%,
  more than the effect under test.

## Limits and unknowns

- One fully quiet A-B-A run per model. The verdict rests on the 27B's run 3:
  a 1-lane ITL fail of 0.02 ms, and 3-lane fails that run 2 does not
  reproduce.
- The 3-lane cells carry leg-to-leg drift in speculative acceptance on the
  27B, from batch composition. The method does not control for it.
- The legs depart from AC 8's defaults: `--prompt-reuse off` everywhere, and
  `--kv-host-pool-bytes 0 --retained-host 0` on Flash-Next. Claims and
  restores were mapped whole, and growth on a claimant's tail was not
  measured.
- The CPU gate is an addition to the spec's harness. It re-ran one block
  (Flash-Next growth, 3 lanes), and the re-run changed that cell's outcome.
- The extend op runs on the legacy default stream, as `ignis_seq_alloc`
  does. The model streams are blocking, so it is ordered with them; between
  rounds the host has already waited for the round. An op on the model
  stream, as the ADR places it, was not measured.
- Not measured: growth by 32-page steps, a single-launch zeroing, prefill
  growth on a claimant's tail, and a chunk's own wall time (the prefill
  figure is a TTFT-derived average).

## Follow-ups

- The owner's call: accept the fixed branch, or re-gate growth at the
  32-page step with a one-launch zeroing (one 27B A-B-A, ~10 min of card),
  from the tag `kv-p0-grow-spike`.
