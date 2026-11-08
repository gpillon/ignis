# The solo width: what `auto` takes on the 24 and 16 GB classes, the chunk bit-identical under it, and Flash-Next's 8,192 at 1.03 GiB of expert cache

- Kind: experiment
- Status: current
- Observed: 2026-11-06
- Last verified: 2026-11-06
- Scope: serving / the solo width (ADR 0018 and ADR 0063 §3 as amended by GitHub #489): its `auto` resolution on the 24 and 16 GB classes, the prefill chunk's numerics under a wider scratch, Flash-Next's prefill chunk in the built-in profiles; the RTX 5090 before and after
- Topic: prefill
- Related: [GitHub #489](https://github.com/gpillon/ignis/issues/489) and [its decisions comment](https://github.com/gpillon/ignis/issues/489#issuecomment-6672861760), [GitHub #526](https://github.com/gpillon/ignis/issues/526), [A wider prefill chunk buys the 27B at most 4%](2026-10-07-27b-prefill-chunk-width.md), [The 27B's prefill chunk at 2,048](2026-10-21-the-27b-prefill-chunk-at-2048.md), [Flash-Next's prefill chunk and the decode share](2026-10-07-flash-next-prefill-chunk-and-decode-share.md), `.scratch/489/`
- Superseded by: none
- Summary: With the spare reservation the solo width's auto takes 4,096 on the rtx-24gb simulation and 2,048 on rtx-16gb (both profiles pin it to the chunk), logits stay bit-identical, and Flash-Next's 8,192 chunk costs 1.03 GiB of expert cache.

## Question

#489 lets a load plan a second, wider prefill chunk for when no lane decodes, the solo width, and
resolves its default `auto` at load on an RTX 30 or RTX 40 card. Which width would `auto` take on the 24
and 16 GB classes, and what does it cost the KV pool? Does a load whose scratch is reserved for the solo
width run the prefill chunk on the same kernels? What does an 8,192-token chunk cost Flash-Next's expert
cache under `rtx5090`, whose profile now sets it? Is anything slower on the RTX 5090, where `auto`
changes nothing?

## Evidence

All on the RTX 5090 (PCIe Gen 3), branch `489-prefill-chunk-solo` over `main` `035ecff1`, under
`IGNIS_GPU_PROFILE=1`.

**The two 8.x classes** (`crates/server/tests/solo_width_gpu.rs`). #508's simulation: the 27B's 16 GB
trellis entry on the routes an 8.x card runs unset (`--model-prefill-linear int8
--model-prefill-attention int8`, `IGNIS_GDN_PREFILL=ours`), with the card the start reads injected as an
8.x card so the load takes the 8.x ladder: `rtx-24gb` in 24 GiB as an RTX 3090, `rtx-16gb` in 16 GiB
less 1 GiB as an RTX 4080. Every load: `adapted: ""`, the profile's context and lanes.

| load | solo width | KV pool, tokens | past the context | against the chunk |
|---|---|---|---|---|
| `rtx-24gb`, the profile (pinned) | 2,048 | 1,139,776 | 91,200 | #508's pool |
| `rtx-24gb`, `auto` | 4,096 (8,192 does not fit) | 1,126,464 | 77,888 | 13,312 fewer |
| `rtx-16gb`, the profile (pinned) | 1,024 | 386,048 | 123,904 | the profile note's pool |
| `rtx-16gb`, `auto` | 2,048 | 352,768 | 90,624 | 33,280 fewer |

- `auto` keeps a width only where, at more than one lane, the pool also holds the 65,536-token spare
  reservation an adapted context keeps. On the 16 GB class 4,096 fits but leaves 40,768 tokens past the
  context (a pool of 302,912, 83,136 fewer than at the chunk), so the ladder steps down to 2,048.
- Under `auto` on the 24 GB load an 11,228-token prompt alone ran 4 chunks, all at the solo width
  (`prefill_chunks_solo: 4` on its `admitted` line), and answered.

**The chunk under a solo reservation** (`crates/core/tests/solo_width_logits_gpu.rs`). The same 16 GB
entry on the same 8.x routes, loaded twice from one materialization, its scratch reserved for 2,048 and
for 8,192 tokens; a 5,000-token prompt prefilled as 2,048-token jobs in both. The last position's logits
are bit-identical (0 of 248,320 differ) and the four greedy tokens after the prompt are the same.

**Flash-Next under `rtx5090`** (`crates/server/tests/flash_next_serving_gpu.rs`). The profile's
Flash-Next values (589,824 tokens, 3 lanes, 2 GiB of pinned experts), planned only (the load stops at a
host plan given a margin no host has, before anything is pinned), with 4.7 GB of the card taken by the
desktop: the expert cache is 15,833,292,112 B (14.75 GiB) at a 1,024-token chunk, 15,361,441,360 B
(14.31 GiB) at 4,096 and 14,732,307,024 B (13.72 GiB) at 8,192. The 8,192 scratch costs 1,100,985,088 B
(1.03 GiB) against 1,024, at one budget; the planner's expected hit rate falls from 0.53 to 0.50.

**The RTX 5090 before and after** (`.scratch/489/bench/`). `main`'s release binaries against the branch's,
on port 8489, make's flags. `auto` resolves to the prefill chunk on Blackwell and Flash-Next has no ladder,
so each pair runs the same widths and the same workspace line (983,876,864 B for the 27B).

The 27B (the `rtx5090` profile, DFlash2/7). The two loads differ in one way: the before load adapted its
context to 720,896 tokens and the after one to 655,360, because 89 MB less of the card was free at its
start, so their KV pools differ:

| | before | after |
|---|---|---|
| `ttft` 8,192, median of 5, cold | 772.6 ms | 769.2 ms |
| `ttft` 65,536, median of 5, cold | 6,900.1 ms | 6,845.1 ms |
| `g3` C=1 | 208.0 tok/s | 222.2 tok/s |
| `g3` C=4 | 160.0 tok/s | 159.1 tok/s |
| `g3` ITL p95 (4 lanes while 32K prompts prefill) | 213.07 ms | 212.34 ms |

The after ITL leg covered 9 of its 10 prefill windows: one decode lane ended on the safety cap first.

Flash-Next (262,144 tokens, 8,192-token chunk; `ttft` in three before/after pairs one after the other,
each a median of 5; `g3` at 5 lanes, since at make's 3 its ITL cell's prefillers are refused while 4
lanes decode):

| | before | after |
|---|---|---|
| `ttft` 8,192, three runs | 2,478 / 2,469 / 2,439 ms | 2,493 / 2,433 / 2,443 ms |
| `ttft` 65,536, three runs | 19,574 / 19,896 / 19,221 ms | 20,152 / 19,337 / 19,317 ms |
| `g3` C=1, two runs | 102.3 / 102.4 tok/s | 100.9 / 101.3 tok/s |
| `g3` C=4, two runs | 48.0 / 42.0 tok/s | 47.9 / 47.7 tok/s |
| `g3` ITL p50, two runs | 24.44 / 24.48 ms | 27.99 / 25.14 ms |
| `g3` ITL p95, two runs | 34.77 / 36.67 ms | 41.13 / 36.59 ms |

The 27B's `g3` C=1 and C=4 samples are void as gate samples, their prompts served from the reuse cache;
their decode rates compare.

## Finding

- `auto` with the spare reservation takes 4,096 on the 24 GB class (13,312 tokens of pool) and 2,048 on
  the 16 GB one (33,280); without the reservation it took 4,096 on both, the 16 GB pool left 40,768
  tokens past the context. On the 24 GB class the 91,200 tokens past the context are what a real card's
  desktop takes first, so a card with a desktop keeps the chunk under `auto`.
- The owner pinned the 27B's solo width to the chunk on both profiles: the KV pool first. `auto` stays
  the default for the 8.x cards no profile pins.
- The narrow width runs the same kernels on the same cuts under a wider reservation: planning the solo
  scratch changes no logit at the prefill chunk.
- Flash-Next's 8,192-token chunk costs its expert cache 1.03 GiB against 1,024. From the ~16.6 GiB the
  `rtx5090` profile measured at 1,024, that keeps ~15.6 GiB, above the 15 GiB floor, and a narrower chunk
  costs Flash-Next's prefill nearly in proportion, so `rtx5090` and `rtxpro6000` set 8,192.
- On the RTX 5090 the 27B reads level. Flash-Next's TTFT reads level over three pairs: the first after
  run's +3% at 64K came back on the repeats. Its decode reads 1.1-1.4% lower at one lane in both `g3`
  pairs, and its ITL p50 +14% then +2.7%. No code on Flash-Next's decode path changed (one width, its
  prefill chunk; the scheduler's width check is a scan of the requests per advance), and two pairs do not
  separate that from this host's spread; the bench was stopped there, on the owner's call.

## Implications

- The load line (`ignis.runtime.prefill_chunk_solo`) and `GET /v1/models` say which width a load runs.
- A logits test that pins the prefill chunk can run on a load with any solo reservation.
- A single before/after pair of Flash-Next benches on this host swings by a few percent; repeat a pair
  before reading a difference, and more than two to read one of about 1%.

## Limits and unknowns

- The 8.x numbers are a simulation on the RTX 5090. The TTFT a wider chunk buys on a real RTX 30 or RTX 40
  card, and whether the vendored GDN gating planner keeps a resident schedule there at 4,096 and 8,192
  tokens, are #516's to measure.
- Flash-Next's chunk on the 24 and 16 GB classes (4,096 and 2,048 in the profiles) is a placeholder:
  Flash-Next has not run on those cards ([GitHub #526](https://github.com/gpillon/ignis/issues/526)).
- The decode cost of Flash-Next's 1.03 GiB of expert cache was not measured in tok/s at 1,024 against
  8,192; the before/after pairs above both ran 8,192.
- The expert-cache cost is a difference at one budget; the absolute cache depends on what the desktop
  holds of the card.

## Follow-ups

- #516: the TTFT at 2,048 against `--model-prefill-chunk-solo 4096` on a real RTX 30 card, and the gating
  planner on sm_86. Done 2026-11-06: both widths ran on an sm_86 card, their figures private under #516's
  rule; the planner keeps no cooperative schedule resident there at 2,048, 4,096 or 8,192
  ([the sm_86 gate's first run](2026-11-06-sm86-gate-first-run.md)).
- #526: Flash-Next's prefill chunk measured on a 24 GB and a 16 GB card.
