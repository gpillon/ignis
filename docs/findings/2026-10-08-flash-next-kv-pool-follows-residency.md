# Flash-Next's KV pool follows residency: decode A-B-A on the 5090

- Kind: experiment
- Status: current
- Observed: 2026-10-08
- Last verified: 2026-10-08
- Scope: serving / Flash-Next VRAM plan (the KV pool policy, the expert cache), decode at one and three active lanes; the default `max_tokens`
- Related: GitHub #309, [ADR 0045](../adr/0045-the-kv-pool-follows-residency-and-state-goes-down-to-disk.md), [spec vram-budget/03](../specs/vram-budget/03-kv-pool-policy-and-kv-disk.md) ACs 2, 7, 8, 9; [Flash-Next on the 5090](2026-10-06-flash-next-on-the-5090.md) (its lanes-at-262K update)
- Superseded by: none

## Question

ADR 0045 sizes Flash-Next's KV pool at 524,288 tokens shared by the lanes,
capped at every lane's context, instead of every lane's whole context. At
the default three lanes and 262,144 tokens that should hand the expert cache
~1.03 GiB. Does decode regress, and does the cache's hit rate rise (spec
vram-budget/03 AC 8)? And do three agent requests that send no `max_tokens`
run at once on that pool (AC 9)?

## Evidence

**Setup.** Release cuda builds: A = `1d8b42d` (the commit before #309 P1),
B = `2c09568` (the policy and the default cap). Three legs, A then B then A,
one server process each, in one GPU lock hold, nothing else on the card
(2.2 GB of desktop). `make config MODEL=flash-next`'s flags -- hq-e8-2b,
`--max-context 262144`, `--prefill-chunk 8192`, `--decode-lanes 3`,
`--vram-headroom-bytes 4G`, MTP off -- plus `--kv-host-pool-bytes 0
--retained-host 0` on every leg: the host had 43-44 GB available against the
46 GB the default host plan needs. Neither moves VRAM (the host retained
slots are host memory; without them A's pool is 12,288 pages instead of
12,296). The harness is the lanes-at-262K finding's: greedy, `ignore_eos`,
1,800 tokens, `max_tokens` set; one lane back to back for 70 s, then three
distinct prompts at once for 100 s; `/metrics` deltas per phase. CPU load
over the twelve minutes: mean 15%, at most 29%, no compiler running in any of
38 samples. Raw output: `.scratch/kv-p1/aba/` (untracked, the owner's clone).

**The plan** (`ignis.runtime.flash_next_plan`):

| | A1 | B | A2 |
|---|---:|---:|---:|
| budget (free − 4 GiB) | 27,125,985,280 | 27,124,740,096 | 27,123,494,912 |
| KV pool | 12,288 pages (3 × 262,144 tokens) | 8,192 pages, 524,288 tokens, `offloaded` | 12,288 pages |
| sequence pool bytes | 3,711,980,736 | 2,604,684,480 | 3,711,980,736 |
| expert cache | 14,509,443,664 | **15,615,494,736** | 14,506,953,296 |

The pools differ by exactly 4,096 pages × 270,336 B (4,224 B per token, KV
and indexer keys), and B's cache is A1's plus that, less the 1.2 MB its
budget came in under A1's.

**Decode** (four requests per one-lane phase, two per lane in the three-lane
phase):

| active lanes | A1 | B | A2 | B vs A mean |
|---|---:|---:|---:|---:|
| 1: tok/s | 97.7-98.0 | **101.7-102.1** | 97.8-98.0 | +4.1% |
| 1: ITL p50 | 9.51 ms | 9.50 ms | 9.52 ms | |
| 1: decode hit rate | 95.81% | **96.55%** | 95.81% | +0.74 pt |
| 1: misses / MB moved per token | 40.2 / 59.6 | 33.2 / 52.1 | 40.3 / 59.6 | −17% / −13% |
| 1: residency stall per token | 2.267 ms | 1.853 ms | 2.270 ms | |
| 3: aggregate tok/s | 99.6 | **109.7** | 99.7 | +10.1% |
| 3: per lane, ITL p50 | 33.1-33.3, 29.0-29.3 ms | 36.5-36.6, 26.5 ms | 33.1-33.3, 29.0 ms | |
| 3: decode hit rate | 89.32% | **90.95%** | 89.31% | +1.63 pt |
| 3: misses / MB moved per token | 97.6 / 109.8 | 82.7 / 98.2 | 97.7 / 109.9 | −15% / −11% |
| 3: residency stall per token | 5.846 ms | 4.918 ms | 5.844 ms | |

The two A legs agree to 0.1 tok/s and 0.01 points: the run's noise is far
below the difference.

**Three agents without `max_tokens`** (AC 9, B only, the same flags, one
more launch): three concurrent greedy requests, class `agent`, thinking off,
no cap, prompts of 30,038, 29,995 and 30,003 tokens (distinct word lists
under one instruction). Each stream was closed once it had decoded 1,800
tokens with all three on lanes.

- All three answered 200 and decoded at once from 28.5 s to 54.7 s (first
  tokens at 8.3, 18.2 and 28.5 s; 1,800 tokens at 54.7, 57.3 and 58.9 s).
- `ignis_kv_pool_used_pages` read **3,232** of 8,192 while they decoded:
  exactly ceil((prompt + 38,912) / 64) summed over the three, so each one
  reserved its prompt plus the default cap.
- `ignis_kv_ram_evictions_total`, `ignis_kv_cache_evictions_total` and every
  `ignis_requests_rejected_total` stayed 0.
- 168.3 tok/s aggregate in the window where all three decoded. Not
  comparable to AC 8's 99.6 (A) or 109.7 (B): these prompts are word lists,
  whose continuations route differently from AC 8's prose prompts.
- A first attempt used a shorter instruction; one stream ended on its own
  (`stop`) at 1,665 tokens, so the window was rerun with prompts that ask for
  5,000 words. The used pages read 3,231 there: the same rule.

## Finding

- **At the default three lanes the policy gives the expert cache 1.03 GiB
  (+7.6%), and decode is faster at both lane counts: +4.1% with one lane
  active, +10.1% aggregate with three.** AC 8 passes: B's tok/s and hit rate
  exceed both A legs at one and at three lanes.
- **The gain is the cache's.** Misses per token fall 15-17% and the bytes
  copied over PCIe 11-13%; the residency stall per token falls by the same
  share. ADR 0045 inferred ~4-5% at one lane from the one-lane/three-lane
  slope; measured 4.1%.
- **Three lanes gain more than one**: their working set is larger, so the
  same extra cache removes more misses per token (15 against 7 fewer).
- **Three agents that send no cap run at once** on the default pool, each
  holding its prompt plus 38,912 tokens (AC 9). Before the default cap each
  would have held a whole 262,144-token context: 12,288 pages, past the
  8,192 the pool now has.

## Implications

- The lane count no longer trades decode speed for contexts: from two lanes on
  the pool is the same 524,288 tokens, and a lane costs its state.
- `--decode-lanes 1` still frees the last ~1.03 GiB (one context instead of
  524,288 tokens) for a lone user.
- The cap of 524,288 tokens is where the gain stops: a smaller
  `--kv-pool-bytes` gives the cache more, at the price of fewer concurrent
  long contexts.

## Limits and unknowns

- One run per binary, at one context and one prompt set (short prose, greedy).
  Long contexts, other domains and sampling were not measured.
- The host retained slots and the KV-RAM arena were off on every leg, for RAM;
  prompt reuse does not touch the decode rounds measured here.
- The plan's expected hit rate (52.4% for B, calibration rates without
  locality) is not comparable to the measured one; A did not log it.

## Follow-ups

- None for the policy. The pool's reservations (prompt plus the cap, or page
  growth) are spec vram-budget/03 P3's, after P0's verdict.
