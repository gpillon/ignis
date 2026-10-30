# The vision residency's acceptance on the 5090: a text-only load's KV under host, +7 ms a 1,024-token encode, identical answers at the same context, decode within 99.7%

- Kind: experiment
- Status: current
- Observed: 2026-10-30
- Last verified: 2026-10-30
- Scope: serving / the 27B's vision load under `vision.residency` `host` and `device`: the VRAM plan under `rtx50-16gb` and `rtx5090`, the media encode, `/v1/decide`, decode, on the RTX 5090 (PCIe Gen 3 x16 host, Windows/WDDM)
- Related: [ADR 0075](../adr/0075-the-vision-tower-is-staged-and-the-embedding-pool-host-mapped-unless-the-load-asks-for-the-device.md), [the prototype's finding](2026-10-30-vision-tower-staged-and-pool-host-mapped.md), [rope auto follows the adapted context](../user/configuration/context-and-speculation.md), https://github.com/gpillon/ignis/issues/413, https://github.com/gpillon/ignis/issues/417
- Superseded by: none

## Question

Does the build of ADR 0075 meet the five criteria the ADR wrote for it? They are:

1. The KV pool on a 16 GB budget.
2. The encode time at 1,024 tokens.
3. `/v1/decide`'s answers and cold `point` time.
4. `device` against `main`.
5. #409's 99% decode gate.

## Evidence

**Setup.**
- Builds:
  - `413-integration` at 78bebf68 (#414-#416: the residency, `host` by default), built in `H:\worktrees\408`;
  - `main` at 18b72a1f, built in the main checkout.
  - Both are `make build` release builds with the same kernel CMake configuration.
- Every leg is a `make start` with `VISION=1 IDLE_UNLOAD=0 LOOP_GUARD=off`. Criterion 1 runs on the 16 GB trellis artifact; the others run on `qwen3_8_27b_nvfp4full-v2.ninfer` under the `rtx5090` profile (hq-e8-2b, DFlash2 with 7 draft tokens, 8 decode lanes).
- One server at a time, nothing else on the card or compiling.
- Raw data: `.scratch/413/acceptance/` in the main checkout (untracked). It holds the drivers (`criterion1.sh`, `campaign*.sh`, `decide_campaign.py`, `encode_campaign.py`, `decode_ab.py`, `run-decode-ab.sh`, `analyze.py`) and each leg's server log, start log and client output (`c1/`, `c234/`, `c234p/`, `c2enc/`, `c4fixed/`, `c5/`).

**1. The KV pool on 16 GB.** Setup: `make start ARTIFACT=./models/qwen3_8_27b_trellis_sc30-v2.ninfer VRAM_BUDGET=16G ARGS="--profile rtx50-16gb [--vision-residency R]"`, no image sent. The `ignis.runtime.vram_plan` event:

| load | `kv_pool_pages` | `weights_bytes` | `workspace_bytes` | `media_embedding_bytes` |
|---|---:|---:|---:|---:|
| text-only | 7,956 | 10,907,958,020 | 395,898,880 | 0 |
| `host`, 1,024 | **7,956** | 10,907,958,020 | 395,912,192 | 0 |
| `host`, 4,096 | **7,956** | 10,907,958,020 | 395,924,480 | 0 |
| `device`, 1,024 | 7,437 | 11,203,677,696 | 395,912,192 | 10,485,760 |
| `device`, 4,096 | 7,384 | 11,203,677,696 | 395,924,480 | 41,943,040 |

**2. Encode time at 1,024 tokens.**
- Setup:
  - 33 fresh 1,024 px scenes (`tools/pointing-scenes/scenes.py --varied --seed 20261031 --n 33`), one chat completion each (`max_tokens` 1). Each is a cache miss, so each request encodes once.
  - The time is the server's own `media.encode_seconds` on `ignis.request.admitted`, which brackets the leaf's synchronous encode.
  - Three warm-up encodes are dropped per leg.
  - Three rounds, each running `main`, `device` and `host` back to back with the order rotated, all at `MAX_CONTEXT=720896`.
- Results, ms:

| mode | n | median | p10 | p90 | round medians |
|---|---:|---:|---:|---:|---|
| `main` | 90 | 42.43 | 40.4 | 46.7 | 43.76 / 41.89 / 40.97 |
| `device` | 90 | 42.10 | 40.2 | 46.6 | 42.84 / 42.11 / 40.54 |
| `host` | 90 | 49.23 | 47.5 | 59.6 | 48.81 / 54.07 / 48.94 |

`host` − `device`: **+7.13 ms pooled**. Per round it was +5.97, +11.96 and +8.39 ms.

**3. `/v1/decide`.**
- Setup:
  - 33 fresh 1,024 px scenes (`--seed 20261030`). Each scene gets three requests in order: `point` (the first question over the image, so it encodes it: the cold point), `box` and a two-way `choice`.
  - Three scenes of warm-up per leg.
- At `MAX_CONTEXT=720896` (two rounds):
  - The answers of all three modes are identical on **198 of 198** questions: every point, box (digits and probabilities included) and choice distribution.
  - Cold `point` medians: `main` 219.9 ms, `device` 208.0, `host` 213.9 (n=60 each). `host` − `device` is **+5.9 ms**.
- Without a pinned context (three rounds):
  - `host` adapted the 5090's context to 786,432 against 720,896 under `device` and `main`, so `auto` rope scaling chose `yarn:3` against `yarn:2.75`.
  - `host`'s answers then differed from `device`'s on nearly every question (0 of 66 points and boxes identical, 2 of 66 choices). They were identical to its own in the next round, and `device` matched `main` everywhere.
  - Cold `point`: `host` − `device` +9.0 ms pooled (n=90; rounds +4.1, +4.5, +10.2).

**4. `device` against `main`.**
- At a fixed `VRAM_BUDGET=28G`, the `vram_plan` event of `device` equals `main`'s in all 30 attributes `main` has, and in the top-level fields. The one addition is `vision_residency="device"`, which #414 adds.
- With the budget following free VRAM, the two differ only in the budget and the KV pages that follow it, by the free memory at start.
- Encode: `device` − `main` −0.33 ms (above).
- Cold `point`: `device` − `main` was +11.5 ms in the unpinned campaign (one round at +37 ms, the others +1.8 and +3.5) and −11.8 ms in the pinned one. Every answer of `device` equals `main`'s.

**5. Decode.**
- Setup:
  - The driver is #335's `ab.py` (eight coding prompts, 512 tokens, greedy, thinking off) at C=1 (mean of the eight), C=3 and C=8 (aggregate, two rounds each).
  - Four legs: integration, main, integration, main, at `MAX_CONTEXT=720896`.
- Results, tok/s:

| cell | integration (`host`) | `main` | ratio |
|---|---:|---:|---:|
| C=1 | 366.5 | 367.6 | **99.71%** |
| C=3 | 678.1 | 675.8 | **100.34%** |
| C=8 | 1,042.4 | 1,042.5 | **99.99%** |

The generated texts are identical between builds and between legs.

`ignis-bench g3` could not decide this gate: its ITL cell sends `logit_bias`, which the server refuses (#288), so the record voids itself.

## Finding

Observed:
- All five criteria hold.
  - On a 16 GB budget, a vision load under `host` keeps the text-only KV pool at 1,024 and 4,096.
  - A 1,024-token encode costs +7.1 ms under `host` (pooled median, n=90; one of three rounds at +12.0).
  - At the same context, `/v1/decide` answers identically under `host`, `device` and `main`, and a cold `point` costs +5.9 ms under `host`.
  - `device`'s plan equals `main`'s at the same budget, with times within noise.
  - Decode is within 99.7% of main's at every width.
- The answers are identical only at the same context. On a card whose start adapts its context to free VRAM, `host` gives back KV and can raise the adapted context: 786,432 against 720,896 on this 5090. `auto` rope scaling then picks a larger YaRN factor, and answers move. The move is deterministic, and it is the same one any change of free VRAM causes.

Inferred:
- The +7.1 ms sits where the prototype's estimate for three slots with the bounce page put it (~+6.5 ms).
- The round at +12.0 ms shows the encode's WDDM cross-stream synchronisation is noisy, as the prototype saw.

## Implications

- ADR 0075's build passes its acceptance, so `rtx50-16gb` turns vision on at 4,096 (#418).
- A comparison of `host` and `device` outputs on the 27B must pin `--model-max-context` (as the identity GPU tests do).
- `ignis-bench g3`'s ITL cell cannot run against ignis until it stops sending `logit_bias`, or the server accepts an empty one. Decode gates use `ab.py` meanwhile.

## Limits and unknowns

- One host, one card, Windows only. Linux and Gen 4/5 links are unmeasured.
- The 16,384-token encode was not timed here. The identity GPU test's single cold encodes were 3,717 ms (`host`) and 3,719 ms (`device`).
- The 16 GB criterion is the VRAM plan only. A 16 GB card was not used, and the trellis artifact was not timed.
- The cold `point` and encode times move by up to ±12 ms between campaigns, beyond the per-mode spread inside a round. Only interleaved rounds compare.

## Follow-ups

- [#418](https://github.com/gpillon/ignis/issues/418): `rtx50-16gb` with vision on at 4,096.
- [#412](https://github.com/gpillon/ignis/issues/412): a query-chunked encode.
