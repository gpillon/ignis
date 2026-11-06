# The solo width: 4,096 on the 24 GB class with nothing adapted, the chunk bit-identical under it, and Flash-Next's 8,192 at 1.03 GiB of expert cache

- Kind: experiment
- Status: current
- Observed: 2026-11-06
- Last verified: 2026-11-06
- Scope: serving / the solo width (ADR 0018 and ADR 0063 §3 as amended by GitHub #489): its `auto` resolution on the 24 GB class, the prefill chunk's numerics under a wider scratch, Flash-Next's prefill chunk in the built-in profiles; the RTX 5090 before and after
- Related: GitHub #489 and its decisions comment, [A wider prefill chunk buys the 27B at most 4%](2026-10-07-27b-prefill-chunk-width.md), [The 27B's prefill chunk at 2,048](2026-10-21-the-27b-prefill-chunk-at-2048.md), [Flash-Next's prefill chunk and the decode share](2026-10-07-flash-next-prefill-chunk-and-decode-share.md), `.scratch/489/`
- Superseded by: none

## Question

#489 lets a load plan a second, wider prefill chunk for when no lane decodes, the solo width, and
resolves its default `auto` at load: 8,192, then 4,096, on an RTX 30 or RTX 40 card, kept only where the
VRAM plan holds it with nothing else adapted. Which width does `rtx-24gb` get, and what does it cost the
pool? Does a load whose scratch is reserved for the solo width run the prefill chunk on the same
kernels? What does an 8,192-token chunk cost Flash-Next's expert cache under `rtx5090`, whose profile
now sets it? Is anything slower on the RTX 5090, where `auto` changes nothing?

## Evidence

All on the RTX 5090 (PCIe Gen 3), branch `489-prefill-chunk-solo` over `main` `035ecff1`, under
`IGNIS_GPU_PROFILE=1`.

**The 24 GB class** (`crates/server/tests/solo_width_gpu.rs`). #508's simulation: the 27B's 16 GB
trellis entry under `--profile rtx-24gb --vram-budget-bytes 25769803776`, on the routes an 8.x card runs
unset (`--model-prefill-linear int8 --model-prefill-attention int8`, `IGNIS_GDN_PREFILL=ours`), with the
card the start reads injected as an RTX 3090 so the load takes the 8.x ladder.
- `adapted: ""`, 1,048,576 tokens on 8 lanes, 2 device retained slots, as before.
- `prefill_chunk_solo auto -> 4096`: 8,192 is refused at the profile's values, 4,096 fits.
- The workspace line is 1,099,301,888 B. The KV pool is 17,601 pages, 1,126,464 tokens: 77,888 past the
  context, 13,312 fewer than #508's 1,139,776 at the 2,048-token chunk.
- An 11,228-token prompt alone ran 4 chunks, all at the solo width (`prefill_chunks_solo: 4` on its
  `admitted` line), and answered.

**The chunk under a solo reservation** (`crates/core/tests/solo_width_logits_gpu.rs`). The same 16 GB
entry on the same 8.x routes, loaded twice from one materialization, its scratch reserved for 2,048 and
for 8,192 tokens; a 5,000-token prompt prefilled as 2,048-token jobs in both. The last position's logits
are bit-identical (0 of 248,320 differ) and the four greedy tokens after the prompt are the same.

**Flash-Next under `rtx5090`** (`crates/server/tests/flash_next_serving_gpu.rs`). The profile's
Flash-Next values (589,824 tokens, 3 lanes, 2 GiB of pinned experts), planned only (the load stops at a
host plan given a margin no host has, before anything is pinned), with 4.7 GB of the card taken by the
desktop: the expert cache is 15,833,292,112 B (14.75 GiB) at a 1,024-token chunk, 15,361,441,360 B
(14.31 GiB) at 4,096 and 14,732,307,024 B (13.72 GiB) at 8,192. The 8,192 scratch costs 1,100,985,088 B
(1.03 GiB) against 1,024, at one budget.

**The RTX 5090 before and after** (`.scratch/489/bench/`). `main`'s release binaries against the branch's,
one after the other, make's 27B flags (the `rtx5090` profile, DFlash2/7, hq-e8-2b), on port 8489.
`auto` resolves to the 2,048-token chunk on Blackwell, so both load the same plan byte for byte (the
before load adapted its context to 720,896, the after one to 655,360: 89 MB less was free at its start).

| | before | after |
|---|---|---|
| `ttft` 8,192, median of 5, cold | 772.6 ms | 769.2 ms |
| `ttft` 65,536, median of 5, cold | 6,900.1 ms | 6,845.1 ms |
| `g3` C=1 | 208.0 tok/s | 222.2 tok/s |
| `g3` C=4 | 160.0 tok/s | 159.1 tok/s |
| `g3` ITL p95 (4 lanes while 32K prompts prefill) | 213.07 ms | 212.34 ms |

`g3`'s C=1 and C=4 samples are void as gate samples, their prompts served from the reuse cache; their
decode rates compare.

## Finding

- On the 24 GB class `auto` is 4,096, not 8,192: 8,192 would take more than the 91,200 tokens the pool
  holds past the context. 4,096 takes 13,312 of them and leaves the context, the lanes and the slots as
  the profile asks. A desktop that needs those tokens leaves the solo width at the 2,048-token chunk.
- The narrow width runs the same kernels on the same cuts under a wider reservation: planning the solo
  scratch changes no logit at the prefill chunk.
- Flash-Next's 8,192-token chunk costs its expert cache 1.03 GiB against 1,024. From the ~16.6 GiB the
  `rtx5090` profile measured at 1,024, that keeps ~15.6 GiB, above the 15 GiB floor the owner asked for,
  so the built-in profiles set 8,192 rather than 4,096.
- On the RTX 5090 the change is neutral: same plan, TTFT within 1%, decode and the ITL cell unchanged.

## Implications

- The 24 GB class's solo width is decided by the desktop of the card it runs on, in steps of a whole
  ladder width; the load line (`ignis.runtime.prefill_chunk_solo`) and `GET /v1/models` say which.
- A logits test that pins the prefill chunk can run on a load with any solo reservation.

## Limits and unknowns

- The 24 GB numbers are a simulation on the RTX 5090; the TTFT a wider chunk buys on a real RTX 30 or RTX
  40 card is measured in #516's session, as is whether the vendored GDN gating planner keeps a resident
  schedule there at 4,096 and 8,192 tokens.
- Flash-Next's before/after bench was not run: with the host's other applications open, 37.2 GiB of RAM
  was available and Flash-Next's host plan needs 41.2 GiB at make's flags. Its serving path is unchanged
  by construction (one width, its prefill chunk); the bench is to be run with the host free.
- The expert-cache cost is a difference at one budget; the absolute cache depends on what the desktop
  holds of the card.

## Follow-ups

- #516: the TTFT at the resolved solo width and the gating planner on an sm_86 card.
- Flash-Next's before/after `ttft` (8K, 64K) and `g3` on the RTX 5090, once the host has the RAM
  (`.scratch/489/bench.sh`).
