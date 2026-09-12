# hq attention agrees with the BF16 route to ~0.23 median relative L2, and its worst rows are softmax flips rather than codec error

- Kind: experiment
- Status: current
- Observed: 2026-09-12
- Last verified: 2026-09-12
- Scope: kernel / GQA attention routes, hq-e8-2b KV cache
- Related: [GitHub #123](https://github.com/gpillon/ignis/issues/123),
  [ADR 0022](../adr/0022-two-kv-formats-bf16-as-oracle.md),
  [hq-e8-2b KV capacity](2026-09-11-hq-e8-2b-kv-capacity.md),
  `kernel/tests/test_hq_route_agreement.cu`
- Superseded by: none

## Question

ADR 0022 requires hq-e8-2b to earn its acceptance in-house: the hq attention
route checked against ignis's own BF16 route on identical keys and values,
within a tolerance **derived** from the measured codec error rather than
copied from another oracle. Two things had to be established before that
tolerance could be written down. What does the disagreement actually look
like at this model's geometry, and is it the shape a per-row codec error
predicts?

## Evidence

`kernel/tests/test_hq_route_agreement.cu` sends the same q / k / v / gate /
positions through `ninfer::ops::gqa_attention` (A1) twice, against two
sequence pools that differ in exactly one field
(`ignis_seq_pool_spec::kv_format`). Rows are the committed real-activation
fixture `kernel/tests/fixtures/hq_kv_rows_27b.bin`, the same corpus
`test_hq_codec_kv_rows.cu` measured the codec on, so the two measurements
compose. Run on this machine's RTX 5090 at the 27B geometry (256 head dim,
24 q heads, 4 KV heads).

Codec error on those rows, for reference (`test_hq_codec_kv_rows.cu`,
measured 2026-09-12): worst per-group median relative L2 0.369634, worst
per-group max 0.773022, min per-group cosine 0.934583, min per-group SNR
8.40 dB.

Attention-level agreement between the two routes:

| Arm | Rows | Median relL2 | Cosine | SNR | Rows past 0.90 |
|---|---|---|---|---|---|
| prefill, W=200 B=1 (Prompt route) | 4800 | 0.227955 | 0.959193 | 10.73 dB | 0.229% |
| prefill, W=9..16 B=1 (Prompt route) | 216–384 | 0.3772–0.3611 | 0.9252–0.9285 | 7.95–8.17 dB | 0% |
| decode, W=1 B=1..8 (SmallT route) | 24–192 | 0.174–0.190 | 0.9686–0.9775 | 11.85–13.27 dB | 0% |

The prefill arm's single worst row measured relative L2 1.537729. Its BF16
norm (2.419) is close to the median row norm (2.75), so it is not a
near-cancelling output row.

## Finding

**Observed.** Attention output over an hq cache agrees with the same
attention over a BF16 cache to a median relative L2 of about 0.19 (decode)
to 0.23 (a 200-token prefill), an aggregate cosine above 0.95, and an SNR
above 10 dB — in every case better than the per-row codec error the same
corpus produces, which is what a convex combination of rows should do.

**Observed.** Agreement degrades toward the raw per-row codec error as the
visible history shrinks: at a 9-token prefill the median is 0.3772, alongside
the codec's own 0.369634 worst per-group median, and the cosine falls to
0.9252 against its 0.934583 row counterpart. The averaging that buys the
long-history arms their margin has nothing to average over when a query sees
one or two keys.

**Observed.** A small fraction of prefill output rows disagree by more than
1.0, which no bound derived from a per-row codec error can cover. At
W=200 that fraction is 0.229%; at the decode widths it is zero.

**Inference.** Those outliers are softmax argmax flips, not codec error
reaching the output. Attention's output is `sum_x p_x V_x`; when two keys are
nearly tied in score, a perturbation the size of the codec's own row error is
enough to move the softmax mass from one to the other, and the two routes
then return two different V rows. Near-orthogonal rows differ by about
sqrt(2) relative, which is what the 1.54 outlier measures. The supporting
evidence is that the outlier's own norm is ordinary (so it is not a
cancellation artifact) and that the decode arm, whose queries attend to a
fixed 200-token history with no fresh-chunk causality gradient, produces none
at all.

## Implications

- A **maximum** per-row bound is the wrong enforced statistic for a lossy KV
  format. `test_hq_route_agreement.cu` therefore bounds the median, the
  aggregate cosine and SNR, and the *fraction* of rows past the codec's own
  worst-row error — measured at 0.229% (11 of 4,800 rows) and enforced at
  0.5% (24 rows). A route that started flipping the softmax often rather than
  rarely trips that fraction; one that flips rarely, as a lossy cache
  inherently does, does not.
- The measurement is bit-reproducible: consecutive runs agree to six decimals
  on every statistic, which is what makes a bound that tight defensible.
- This is also the concrete reason ADR 0022 keeps BF16 as the oracle rather
  than asking hq to match it token for token. Greedy decoding turns a softmax
  flip into a different token, so hq's canary floor is a **sanity** check
  (coherent and deterministic — `crates/server/tests/hq_canary_gpu.rs`) and
  never an agreement score.
- The numbers give the G4 gate a prior: hq and BF16 will not produce the same
  token stream on the same prompt, and a G4 cell that assumed they would is
  measuring the wrong thing.

## Limits and unknowns

- Measured without the hq residual window (the exact BF16 sink + recent rows
  `ninfer/ops/gqa_attention.h` describes). This engine leaves those side
  planes empty, so the numbers above are the codec-only path. The vendored
  header states the per-vector bias compounds over long windows, so a long
  context may be worse than these 200-token arms and the residual window is
  the reference's own protection against it.
- The history here is 200 tokens. Nothing was measured at the 40,960-token
  configured context, and the outlier fraction is a function of how often
  keys are nearly tied, which grows with the key count.
- Queries are real captured K rows used as a stand-in for post-RoPE queries
  (same fused `qk_norm_rope` normalization, so the same scale). Real queries
  were not captured.
- One GPU, one fixture, one geometry.

## Follow-ups

- Long-context retrieval under hq is a G4 gate cell
  ([GitHub #128](https://github.com/gpillon/ignis/issues/128)); if it comes
  back short, this finding's second limit is the first place to look, and the
  hq residual window is the reference's own answer to it.
