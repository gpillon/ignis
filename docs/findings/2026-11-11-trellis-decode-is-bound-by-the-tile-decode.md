# The 27B's trellis decode is bound by the tile decode in each SM, not by how its work is scheduled: a balanced probe caps near 80% of the roofline, and a rescheduled route gains nothing in a served-like round

- Kind: experiment
- Status: current
- Observed: 2026-11-11
- Last verified: 2026-11-11
- Scope: kernel / the dense trellis linear's decode route (1 to 8 tokens) at the 27B 16 GB artifact's matrix shapes
  on the RTX 5090: its profile, a balanced probe, and a rescheduled route measured against it
- Topic: decode-27b
- Related: [GitHub #579](https://github.com/gpillon/ignis/issues/579),
  [the dense trellis kernels on the 5090](2026-10-23-dense-trellis-kernels-on-the-5090.md),
  [the 16 GB artifact's speed estimate](2026-10-24-27b-16gb-speed-estimate.md),
  [Flash-Next's MoE decode is structure-bound](2026-10-05-moe-decode-is-structure-bound.md),
  [the hq verify decode cost](2026-09-24-hq-verify-decode-cost.md),
  [ADR 0060](../adr/0060-the-27b-has-a-trellis-body-for-small-cards.md)
- Superseded by: none
- Summary: On the RTX 5090 the 27B's trellis decode is bound by its tile decode per SM: a balanced probe caps near 80% of the DRAM roofline at K=3, and a rescheduled route gains 3.4% back to back and nothing in a served-like round.

## Question

The trellis body of the 27B's 16 GB artifact is 9.19 GB and one token reads it at 65% of the 5090's 1,792 GB/s
roofline ([the dense trellis kernels](2026-10-23-dense-trellis-kernels-on-the-5090.md)), where the 4-bit NVFP4
backbone reads its 16.8 GB at about 100%. In a served round the trellis part is 9.40 ms of 14.47 ms
([the speed estimate](2026-10-24-27b-16gb-speed-estimate.md)). That finding published no profile of the decode
route. What binds it, and does restructuring how the work is cut and launched bring the body toward 85-90%, which
would be worth about 2.5 ms a round?

## Evidence

**Setup.** RTX 5090, Windows, `ignis_kernel_trellis_linear_bench` with synthetic weights at the quant's K mix, on a
box shared with other sessions' builds (every comparison below is two routes back to back in one process). The work
is uncommitted in the local worktree of branch `27b-trellis-decode` (from main `3a3135f8`); raw logs, profiles
and the builds are in its `.scratch/27b-speed/`.

**The old route's profile** (Nsight Compute, application replay, warm caches, 2.86 GHz; `ncu2-gateup.txt`,
`ncu2-gdnz.txt`):

| launch | DRAM | issue slots busy | eligible warps per scheduler | active warps per SM | instructions |
|---|---:|---:|---:|---:|---:|
| MLP gate+up, 1 token (272 CTAs) | 65.4% | 38.6% | 0.79 | 13.6 of 16 | 43.1 M, about 62 per MMA |
| GDN z, 1 token (336 CTAs) | 30.3% | 24.5% | 0.55 | 14.5 | 8.4 M |

- 128 registers per thread allow two CTAs per SM, and the gate+up grid is 0.8 of a wave (272 CTAs on 340 slots).
- On gate+up about 68% of stall samples are long scoreboard; the top site, 22% of all samples, is a tile's first
  shuffle waiting on the weight word it loaded. No pipe is saturated (LSU 28%, FMA 24%, ALU 21%, tensor 12%).
- GDN z is latency-dominated: long scoreboard 20%, math pipe throttle 20%, wait 16%, no instructions 14%.

**A balanced probe** (`probe1.log`, the old kernel): one matrix cut into 340 equal units, so every SM holds two
CTAs and no SM idles.

| probe, 5,120 x 43,520, 1 token | µs | % of the roofline |
|---|---:|---:|
| K = 2 | 48.53 | 64.2% |
| K = 3 | 58.07 | 80.4% |
| K = 4 | 75.44 | 82.5% |

The same probe on later builds of the session read 74.2% to 81.7% at K = 3.

**With the memory stall removed** (`ncu3-gateup.txt`): a variant that stages the weights through shared memory ran
gate+up with DRAM at 0.4% (its weights in L2 under the profiler) in 54.6 µs against the old route's 57.3 µs, at 48%
issue.

**A rescheduled route** (`fused`): the decode kernel's register pipeline unchanged (tiles two blocks ahead, the
shuffle decode, the same block computation and the fold every 32 inputs), with the work cut into ranges per CTA
(the old units, or equal ranges where the units leave SMs idle: gate+up and q), the pieces summed in a fixed order,
the rotation chained to the call before it and an early launch trigger.

| body, 9,135.6 MB (mean of two runs) | 1 token | GB/s | % of 1,792 | 8 tokens | 8/1 |
|---|---:|---:|---:|---:|---:|
| old route | 7,898.6 µs | 1,157 | 64.5% | 8,170.7 µs | 1.034 |
| rescheduled route | 7,629.8 µs | 1,197 | 66.8% | 7,988.9 µs | 1.047 |

Sources: `m-fin-body1.log`, `m-fin-body2.log`.

| the same pair, under other launch conditions | old, 1 / 8 tokens | rescheduled, 1 / 8 tokens |
|---|---:|---:|
| another kernel between calls (`--sep`, `m-pdl1-sep.log`) | 8,107 / 8,377 µs | 8,080 / 8,408 µs |
| the rotation launched normally, not chained (`m-pdl0.log`) | 7,907 / 8,186 µs | 7,923 / 8,287 µs |

One other `--sep` run (`m-fin-body-sep.log`) read the rescheduled route at 11,157 µs against 8,066; it did not
reproduce and is unexplained.

Per matrix the rescheduled route trims gate+up from 73.8% to 77.1% of the roofline and a call's floor from 3.25 to
2.23 µs (a 128 x 128 call), and is slower at 8 tokens on the MLP down projection and the GQA k/v
(`m-ls6-rows-*.log`).

**Numerics.** Cut into the old units the rescheduled kernel is bit-identical to the old route (0 of 296 calls
differ: every body shape and K, 1 to 8 tokens, store, add and SwiGLU). With equal ranges on gate+up and q the bits
differ there and the worst share of the numeric bound is 0.891 at up to 8 tokens, the old route's own; the overall
worst stays 0.917.

**What else was measured, all slower than the old route:**

| variant | body, 1 token |
|---|---:|
| shared-memory ring, the rotation inside the kernel (bit-identical) | 7,996 against 7,820 µs; 10,342 against 8,140 at 8 tokens |
| the ring with the chained rotation | 8,142 against 7,944 µs |
| the ring with equal ranges, at 2 CTAs per SM, then at 3 (80 registers) | 8,709, then 8,763 µs |
| an L2 prefetch 6 blocks ahead | 9,234 to 9,647 against 7,892 µs |

A `cp.async` with the L2 evict-first hint traps as an illegal instruction on sm_120.

## Finding

Observed:

- **The decode route is latency-bound on its own weight loads, with no pipe saturated**: 38.6% issue on the largest
  matrix, 0.79 eligible warps per scheduler, two thirds of the stalls a tile waiting for the word it just loaded.
- **Balance alone caps the body near 80%.** With every SM holding two CTAs of equal work, K = 3 reads at 80.4% of
  the roofline (74-82% across the session's runs), K = 2 at 64.2% and K = 4 at 82.5%.
- **Removing the memory stall exposes the decode.** With the weights in L2 the same launch takes 54.6 µs against
  57.3 µs: the tile decode's own instructions, about 62 per MMA at 48% issue, set the time.
- **Rescheduling gains 3.4% at one token and 2.2% at eight in back-to-back calls, and nothing otherwise.** With
  another kernel between calls, or with the rotation launched normally, the two routes are within 0.4%.
- **More occupancy did not help**: three CTAs per SM at 80 registers read no faster than two.

Inferred:

- **In a served round the rescheduled route is worth 0 to about 0.18 ms** of the 9.40 ms trellis part, at most
  about 1.3% of the 14.47 ms round and likely about 0.5%: only adjacent trellis calls overlap (qkv then z, the
  SwiGLU then the down projection, k then v), and the rest of a round's calls have other kernels between them.
- **The gap to the NVFP4's roofline is the format's decode, as built**: the trellis tile costs about 62
  instructions per MMA where the same loop without the state walk would need about 47. Reaching 80% needs both the
  balance and that decode; going past it needs a shorter decode per tile.

## Implications

- The rescheduled route has no case as a default: it is at parity where a round runs it and slower on two shapes
  at 8 tokens. It stays uncommitted in its worktree.
- Work on this route should start from the tile decode's instruction count and its dependency on the loaded word,
  not from the split, the launch chain or the occupancy, which this session exhausted.
- What is left to take is bounded: at about 80% of the roofline the 9.19 GB body would take about 6.4 ms against
  7.9 ms today, about 1.5 ms or 10% of a round, and only with a new tile decoder. The 85-90% the question asked
  about is not reachable by scheduling.
- The speed estimate's rule for 16 GB cards stands: on those cards the decode's speed is the tile decode's
  instruction count ([the dense trellis kernels](2026-10-23-dense-trellis-kernels-on-the-5090.md)).

## Limits and unknowns

- Synthetic weights at the quant's K mix, not the artifact's bytes; no server run. The served-round figures are
  inferred from the bench's separated-call pair.
- A shared box: other sessions' builds and tests ran during some runs. Each pair ran in one process, and the body
  figures are the mean of two runs that agree within 0.4%.
- The profiles are application replays of two launches (gate+up and GDN z). Kernel-replay profiles of the chained
  kernels gave unusable timings and are not used.
- The instruction count for the rescheduled route on gate+up (49.3 M against 43.2 M) comes from a profile whose
  timings were unusable.
- The 80% ceiling is one probe shape (5,120 x 43,520) on one card; the 5090's clocks and L2.
- The code compiles for sm_86 and was not run on an sm_86 card.
- No faster tile decode was attempted, so what one would reach is not measured.

## Follow-ups

- None filed; [GitHub #579](https://github.com/gpillon/ignis/issues/579) closes on this finding. A shorter tile decode (fewer instructions per state, or decode warps decoupled from the MMA warps,
  as [the hq verify attention](2026-09-24-hq-verify-decode-cost.md) took) is the one direction left, for an owner
  decision.
- The bench's timing path uploads its inputs with a pageable copy and then launches on a non-blocking stream; the
  agreement path was fixed in the worktree, the timing path was not (its timings do not depend on the values).
