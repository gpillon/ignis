# Flash-Next's MTP head accepts 0.83 of first drafts on the engine's own states, and one lane's verify column costs ~2 ms: phase A is a GO

- Kind: experiment
- Status: current
- Observed: 2026-10-06
- Last verified: 2026-10-06
- Scope: Flash-Next MTP speculation (spec flash-next/07 phase A): the draft head's input convention and acceptance, the decode round's cost per extra row at 1-3 lanes
- Related: https://github.com/gpillon/ignis/issues/306, [spec 07](../specs/flash-next/07-mtp-speculation.md), [Flash-Next decode round](2026-10-06-flash-next-decode-round.md), [Flash-Next on the 5090](2026-10-06-flash-next-on-the-5090.md)
- Superseded by: none

## Question

Spec 07 pre-registered a GO / NO-GO before any MTP build: does the
checkpoint's one-layer MTP head, fed the trunk's states as the engine
computes them, accept enough drafts that a one-lane round projects to
≥ 1.20× spec-off (c = 3.8 ms per verify column, R₁ = 15.3 ms, d = 0.9 ms per
draft step, some k ≤ 3)? Which of the undocumented input conventions is the
head's? And what does an extra verify column cost at 1, 2 and 3 lanes?

## Method

- **Trunk states.** A test-only tap (`residual-tap` feature,
  `kernel/include/ignis_fn_residual_tap.h`) copies each prefill chunk's final
  pre-mixer stack `S_p` (BF16 `[10240]`) to the host.
  `crates/core/examples/flash_next_mtp_phase_a.rs corpus` cut 16 prompts from
  the converter's reference windows (the allowlisted corpus):
  - 12 short: 1,536 tokens of code, Python, prose, English and chat;
  - 4 long: the code and the prose document at 8,192 and 24,576 tokens.

  Each was continued by 512 greedy tokens through decode rounds, then
  prefilled once more with the tap armed. Load: hq-e8-2b KV, captured
  graphs, a 17.0 GB expert cache (the served plan at the 4G headroom gives
  16.2-17.2 GB).
- **The tap is the head's input.** The trunk's own mixer and BF16 `lm_head`
  over the tapped stacks give the engine's pick at 2,003 of 2,048 rows (short
  text) and 999 of 1,024 (24K text). Every disagreement is a near-tie
  ≤ 0.625 logits (the engine's head is FP8).
- **The head.** `tools/flash-next-mtp/phase_a.py` fetched the 31 BF16
  `mtp.*` tensors, the trunk's `embed_tokens`, `lm_head` and final mixer by
  range into RAM (7.76 GB in 111 s, revision de4b8e4d). `mtp.py` runs the
  block on transformers' own `Qwen4ExpTextDecoderLayer` and mixer. Only the
  combine and the chaining are written by hand.
  - On a tiny random config in fp32, `test_mtp.py` holds a causal pass and a
    3-step chain to the HF layer over the same sequence, the indexer
    selecting.
  - On the real weights in BF16: relative L2 0.021 for the causal pass, 99.3%
    of the same drafts, and 0.017-0.022 for three chains (0.10 for a fourth,
    a near-tie routing flip is the likely cause).
- **Scoring.** Draft j at entry i is built from `(S_i, t[i+1])` and is
  accepted when it equals the trunk's greedy token `t[i+1+j]`.
  - On the generated half of a text that token is the text itself (the
    decode route's pick, what a verify accepts). So a chain teacher-forced
    with the text is exact: α_j is the share of draft j accepted among the
    positions whose drafts 1..j−1 were.
  - The prompt half (held-out text) scores α₁ against the engine's prefill
    pick.
  - Every convention ran on the same 8,192 generated positions.
- **Round cost.** `flash_next_mtp_phase_a cost` and `cost-repeat` time decode
  rounds of 1-8 rows (the decode route's cap) on a load of 8 lanes, staging
  included, 40 rounds per cell, the first 8 dropped, median.
  - Shape: L lane groups of w lanes, lane j of a group holding its text up to
    token P + j. A round then runs w consecutive greedy tokens per group, the
    tokens and experts a verify of w columns runs.
  - `cost-repeat` interleaves the widths per text group, with the width-1
    round first and last.

## Evidence

**Conventions** (α₁ on 8,192 generated positions; paired difference to the
best with its 95% half-width):

| C-comb | C-norm | α₁ | vs best |
|---|---|---:|---:|
| **a** per-stream `fc_hidden`, embedding broadcast | **a** grouped per stream | **0.828** | — |
| a | b one norm over 10,240 | 0.804 | −0.025 ± 0.006 |
| b streams averaged, broadcast | a | 0.604 | −0.224 ± 0.010 |
| b | b | 0.647 | −0.182 ± 0.010 |

**Chaining** (comb a, norm a; α₂ / α₃ / α₄):
- **C-chain a** (the block's own pre-mixer stack): 0.800 / 0.811 / 0.821.
- C-chain b (the post-mixer state ×4): 0.748 / 0.727 / 0.732.

**Per class** (a / a / a):

| class | α₁ | α₂ | α₃ | α₄ |
|---|---:|---:|---:|---:|
| all, 16 texts | 0.828 | 0.800 | 0.811 | 0.821 |
| code, 8 | 0.858 | 0.838 | 0.847 | 0.847 |
| prose, 8 | 0.799 | 0.758 | 0.768 | 0.787 |
| short (≤ 2K), 12 | 0.830 | 0.806 | 0.821 | 0.834 |
| long (8K, 24K), 4 | 0.825 | 0.780 | 0.778 | 0.781 |

- α₁ on the held-out prompt halves, against the prefill pick: 0.804.
- τ = 1 + α₁ + α₁α₂ + …: 1.83 (k=1), 2.49 (k=2), 3.03 (k=3), 3.47 (k=4).

**Long context** (4 texts, 2,048 generated positions):
- The head's own indexer (2,048-token budget) against dense attention:
  α₁ −0.001 ± 0.003, and α₂-α₄ within 0.006.
- Block boundaries offset by one entry (the spec's position row): identical.

**Round cost** (ms; c = the increment per verify column, i.e. per row at one
lane and per L rows at L lanes):

| lanes | spec-off round | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---:|---:|---:|---:|---:|
| 1, 4 texts interleaved | 11.7 | 13.7 (c 2.0) | 15.6 (c 2.0) | 19.6 (c 2.6) | 21.1 (c 2.4) |
| 1, code0 only | 10.9 | 12.5 (c 1.7) | 17.0 (c 3.1) | 22.3 (c 3.8) | 25.1 (c 3.6) |
| 2, interleaved | 18.5 | 26.3 (c 7.8) | 35.0 (c 8.3) | 38.2 (c 6.6) | — (10 rows) |
| 2, first pass | 21.0 | 29.7 (c 8.7) | 38.9 (c 9.0) | 52.0 (c 10.4) | — |
| 3, interleaved | 40.8 | 47.7 (c 6.9) | — (9 rows) | — | — |
| 3, first pass | 27.4 | 42.1 (c 14.6) | — | — | — |

- Distinct texts per row (the three-lane proxy's regime) at 4-8 rows cost
  39.6 / 62.1 / 74.9 / 88.3 / 106.8 ms. Consecutive tokens of one text at
  8 rows cost 36.0 ms.

**Projection** (speedup = τ_k × spec-off round / (round with k drafts +
k × 0.9 ms)):
- **Pre-registered** (R₁ 15.3, c 3.8): **1.40× (k=1), 1.54× (k=2),
  1.58× (k=3): GO.**
- One lane, measured: 1.47× / 1.67× / 1.59× / 1.64× at k = 1-4 over the
  four texts (code0 alone: 1.48× / 1.44× / 1.31×).
- Two lanes, measured: 1.25× (k=1), 1.25-1.28× (k=2), 1.16-1.37× (k=3).
- Three lanes, measured, k = 1 only: 1.17× (first pass) to 1.54× (a triple
  whose spec-off round was inflated by expert misses).

## Conclusion

1. **The head's input is comb a × norm a, chained per a.**
   - The stack enters per stream: `fc_hidden` on each grouped-normed stream,
     plus `fc_embedding(norm(embed(t)))` broadcast to every stream.
   - A chained draft feeds the block's own pre-mixer stack back.
   - ExLlamaV3's `stream_tap=True` guess is right. Averaging the streams
     (b) costs 0.22 α₁. A single norm over all 10,240 costs 0.025.
2. **Phase A is a GO** with margin: 1.40-1.58× pre-registered at k = 1-3,
   against the 1.20× bar.
3. **At one lane a verify column costs ~2 ms, not 3.8.**
   - At k ≤ 2 it is ~2.0 ms over four texts, ~2.4-2.6 at k = 3-4. The
     three-lane proxy overstated it by ~2×.
   - The projected one-lane gain is 1.6-1.7× at k = 2-4.
   - Per text it varies with how many of the new rows' experts miss the
     cache: 0.7-3.8 ms per column on one code text (code0, both passes),
     1.2-1.9 ms on another (code6).
4. **More lanes pay less.**
   - At two lanes a column (two rows) costs 7.8-9 ms and the gain is
     ~1.25× at k = 1-2.
   - At three lanes only k = 1 fits the decode route's 8-row cap (9 rows at
     k = 2). Its cost is dominated by the three texts' expert-cache misses:
     6.9-14.6 ms per column, 1.17-1.54×.
   - A per-round row budget of ≤ 8 rows (k ≤ 7 at one lane, k ≤ 3 at two,
     k = 1 at three) is what the current kernels allow.
5. **The indexer question is moot for acceptance.**
   - The head's own indexer equals dense attention at 8K and 24K, so the
     trunk's layer-47 selection (C-idx b, not measured: it needs that
     selection tapped) cannot do better.
   - The block offset (position p vs p+1) changes nothing.

## Caveats

- **BF16 head.** The quantized head may lose up to 0.03 α₁ (spec 07 AC2).
  At α₁ 0.80 the pre-registered k = 2 projection is still ~1.5×.
- **Prefill-route states.** The tapped states are the prefill route's; a
  verify runs the decode route.
  - On the generated text the prefill's pick differs from the decode's token
    at 581 of 8,176 positions (7.1%; median margin 0.375 logits, 10% above
    1.1, max 3.6). That rate is itself worth knowing: the forward test's G1
    prompts saw none at shorter context.
  - α₁ against the prefill pick instead of the text is 0.84 (0.83 against
    the text), so the effect on α is small.
- **Round harness.** The round costs come from a tight loop (no scheduler)
  on HEAD 1166102.
  - Its one-lane spec-off round, 10.9-12.1 ms, is below the 15.3 ms the spec
    took from the served decode-round finding. The HC-mix work and the
    serving loop's own time are both in that gap.
  - A real verify reads one lane's KV and GDN state for k + 1 columns and
    adds the record and the fold. The clone lanes read k + 1 states and
    record nothing, so phase C's c(w) remains the number of record.
- **Draft cost.** d = 0.9 ms per draft step is the spec's estimate, not a
  measurement. At d = 1.5 ms the one-lane k = 2 projection is ~1.55×.
- **Noise.** Two cells of the same text and width measured 10.9 and 14.4 ms
  in one session. The interleaved pass is the better estimate, and the
  three-lane cells mostly measure which three texts share the cache.

## Reproduce

All on a free 5090, one Flash-Next load at a time (~38 GB pinned RAM):

```text
cargo build -p ignis-core --features cuda,residual-tap --example flash_next_mtp_phase_a
target/x86_64-pc-windows-msvc/debug/examples/flash_next_mtp_phase_a corpus .scratch/mtpA/corpus
target/x86_64-pc-windows-msvc/debug/examples/flash_next_mtp_phase_a cost .scratch/mtpA/corpus
target/x86_64-pc-windows-msvc/debug/examples/flash_next_mtp_phase_a cost-repeat .scratch/mtpA/corpus
F:/ai/ngram-venv/Scripts/python.exe tools/flash-next-mtp/phase_a.py run --corpus .scratch/mtpA/corpus --out .scratch/mtpA/alpha.json
F:/ai/ngram-venv/Scripts/python.exe tools/flash-next-mtp/phase_a.py project --alpha .scratch/mtpA/alpha.json --cost .scratch/mtpA/corpus/cost_repeat.json
```

- Run time: corpus ~6 min, each cost pass ~6 min, the prototype ~8 min
  including the fetch.
- The 2026-10-06 outputs are in the worktree's `.scratch/mtpA/` (untracked):
  - `alpha.json` (per text and convention: counts, and draft 1's
    per-position outcome);
  - `corpus/cost.json` and `corpus/cost_repeat.json`;
  - `corpus/manifest.json`.
