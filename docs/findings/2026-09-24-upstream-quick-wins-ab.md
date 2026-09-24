# Four upstream kernel ports buy 8% at eight lanes and 10-16% of prefill without moving a token, and the shortlist draft head loses more acceptance than it saves

- Kind: experiment
- Status: current
- Observed: 2026-09-24
- Last verified: 2026-09-24
- Scope: kernel / vendored NVFP4 SwiGLU, W4A4 TMA GEMM, RMSNorm and hq-e8-2b prompt scratch; DFlash2 proposal head
- Related: [ADR 0031](../adr/0031-vendored-kernel-bottleneck-exemption.md),
  [upstream perf survey](2026-09-18-ninfer-upstream-perf-survey.md) (branch `ninfer-upstream-survey`),
  [Decode round anatomy](2026-09-18-decode-round-anatomy.md),
  [hq vs BF16 decode cost](2026-09-13-hq-vs-bf16-decode-cost.md)
- Superseded by: none

## Question

A second survey of the NInfer upstreams (2026-09-24, `.scratch/sota-research-2026-09-24/06-ninfer-upstream.md`)
ranked four changes as bit-exact ports on what ignis runs, and one as worth
measuring:

1. `1c8f8acc` + `0f0899c9`: the fused NVFP4 W4A4 SwiGLU registered through
   T=128. ignis registered it through T=48 and at T=1024 only, so a verify
   round of 7-8 lanes (T=56/64) materialized 34816 x T gate/up rows and read
   them back on all 64 layers.
2. `ee9d5192`: the W4A4 TMA GEMM CTAs rasterised token-fastest, and each
   activation-scale box fetched once per pair of K tiles.
3. `9954867a`: RMSNorm issuing its weight (and gate) loads with x, before the
   reduction.
4. The fork's hq-e8-2b prompt scratch decoder (cometkim `feat/1m-context`,
   2026-09-10), which stages its temporary Rice symbols in shared memory
   instead of in the output row in device memory.
5. `text/draft_head`: a Q4 proposal head over the 131,072 most frequent tokens
   (356 MB) that the artifact already carries, where the drafter used the
   1.27 GB W8 output head.

What does each buy on ignis's own serving configuration, and do the ports keep
the output identical?

## Evidence

**Setup.** RTX 5090, exclusive card, Windows. Server flags are `make config`'s:
hq-e8-2b, 262,144 context, 1024-token prefill chunks, `--spec dflash2
--draft-tokens 7`, no vision. Baseline is main `8088ac0`; the ported build is
branch `sota-quick-wins`. There is one launch per arm, and the arms ran back to
back in one sitting. The driver is `.scratch/sota-quick-wins/ab.py`, the
launcher `.scratch/sota-quick-wins/serve.sh`, raw results
`.scratch/sota-quick-wins/results/*.json` and server logs
`.scratch/sota-quick-wins/*.log`.

- **C=1:** eight coding prompts (write, rewrite/edit, explain), greedy, 512
  tokens, sequential.
- **C=8:** the same eight prompts at once, two rounds.
- **Cold prefill:** one-token completions of 14K, 46K and 105K-token source
  prompts, each opened by a random nonce so no prefix is reused.

**Kernel tests.** `kernel/build.ps1 -Test` is 65 of 65 green. The NVFP4 SwiGLU
op test now covers every new schedule boundary: T = 2, 5, 16, 17, 48, 56, 64,
65, 96, 97, 112, 128, 129, 256, 1024. `ignis_hq_codec_kv_rows_test` decodes all
8,192 real fixture rows three ways: the per-thread decoder, the group decoder
staging in the row, and the group decoder staging on chip. **0 of 2,097,152
elements differ.**

**Ports, baseline vs ported (both full proposal head):**

| cell | baseline | ported | change |
|---|---:|---:|---:|
| C=1, geometric mean over 8 prompts | 361.5 tok/s | 366.7 tok/s | +1.4% (every prompt +1.2..+1.6%) |
| C=8 aggregate, round 0 / 1 | 993.2 / 978.9 tok/s | 1064.6 / 1066.3 tok/s | **+8.0%** |
| cold prefill 14,285 tokens (2 reps) | 1.96 / 1.91 s | 1.79 / 1.73 s | +10.0% |
| cold prefill 45,620 tokens (2 reps) | 7.78 / 7.79 s | 6.89 / 6.90 s | **+12.9%** |
| cold prefill 105,232 tokens | 26.30 s | 22.71 s | **+15.8%** |

**Text.** C=1 is 8 of 8 identical, token for token, and C=8 is 16 of 16. The
C=8 texts are themselves not the C=1 texts (2 of 8 match), which is the width
drift [Batched decode width drift](2026-09-14-batched-decode-width-drift.md)
already records. The ports do not add any.

**Shortlist proposal head, ported build, `--draft-head full` vs `shortlist`:**

| cell | full | shortlist | change |
|---|---:|---:|---:|
| committed tokens per round, C=1 | 5.43 | 5.03 | −7.4% |
| the two rewrite/edit prompts, tokens per round | 7.88 / 7.64 | 6.48 / 5.28 | −18% / −31% |
| C=1 geometric mean | 366.7 tok/s | 346.6 tok/s | −5.5% |
| C=8 aggregate | 1065.5 tok/s | 1044.6 tok/s | −2.0% |

On the prose-like prompts it helps: prompt 0 is +10.1% and prompt 2 +3.6%.
Every rewrite prompt loses.

## Finding

**Observed.** The four ports together are worth +8.0% of aggregate decode at
eight lanes, +1.4% at one lane, and +10% to +16% of cold prefill. The prefill
gain grows with context: +10.0% at 14K, +12.9% at 46K, +15.8% at 105K. The
greedy output is unchanged in every cell measured.

**Inference, not isolated per port.**
- The eight-lane gain is the fused SwiGLU at T=64. It is the only one of the
  four that acts on a verify round's shape and not on one lane's, and at one
  lane (T=8) the route is unchanged.
- The growth with context is the hq scratch decoder. It is the only change
  whose work scales with the visible history; the TMA and RMSNorm changes
  scale with the chunk.
- The one-lane +1.4% is RMSNorm, which upstream measured at +1.5..2.6%.

**Observed.** The shortlist head makes a round about 0.55 ms cheaper but loses
7.4% of committed tokens per round on coding prompts, and up to 31% on rewrite
turns. Their rare identifiers fall outside the frequency shortlist, so the
drafter cannot propose them. It is slower overall at both one and eight lanes.
The measured-better method is the default, so `full` stays the default and
`shortlist` is an opt-in (`--draft-head shortlist`).

## Implications

- The fused SwiGLU's column tiles are the eight-lane lever. The round is
  weight-streaming bound per lane (anatomy finding), so a route that
  materialized 34816 x 64 BF16 per layer was paying twice for a width it could
  have held in registers.
- A drafter head only pays when the tokens a round needs are in it. For a
  coding agent, the turns that accept the most — copying code — are exactly
  the turns a frequency shortlist cuts. Any future narrower head should be
  judged on rewrite prompts, not on prose.
- hq-e8-2b's prompt-attention cost is partly the scratch decoder's memory
  traffic, and it grows with history. The KV-format question at long context
  (`.scratch/sota-research-2026-09-24/SINTESI.md` §2.2) is worth re-measuring
  on this build before choosing a format.

## Limits and unknowns

- There is one launch per arm, not ADR 0021's live/live pair of launches.
  - The C=1 effect is small (+1.4%), but it is the same sign and size on all
    eight prompts.
  - The C=8 and prefill effects are 5 to 10 times larger than the
    round-to-round and rep-to-rep spread (under 1.5%).
- The four ports were built and measured together. The per-port attribution
  above is by argument, not by separate builds.
- The texts are identical, but greedy equality over 8 prompts does not prove
  the SwiGLU T=64 route is bit-identical to the materialized one. It is
  equivalent on these prompts.
- The prompts are the driver's eight coding prompts, not an owner trace. The
  shortlist verdict could differ on a prose-heavy workload.

## Follow-ups

- Re-measure hq vs BF16 decode at 32K-128K per lane on this build.
- The upstream TMA "ragged tail" chain (`1d8587bc` → `5f5fccab` → `abbeea0a`)
  and the drafter MLP single A16 pass are still unmeasured
  (`06-ninfer-upstream.md`).
