# Zeroing the GQA attention workspace is dead work under hq-e8-2b, and costs 2.1% of GQA layer time

- Kind: experiment
- Status: current
- Observed: 2026-09-18
- Last verified: 2026-09-18
- Scope: kernel / GQA attention workspace, prefill chunk cost, GPU A/B method
- Related: [#84](https://github.com/gpillon/ignis/issues/84) (where the zeroing
  was introduced), [#92](https://github.com/gpillon/ignis/issues/92) (the chunk
  profiler this reuses), [#123](https://github.com/gpillon/ignis/issues/123)
  (the hq routes), [ADR 0010](../adr/0010-vendored-reference-kernels.md),
  [ADR 0019](../adr/0019-decode-cuda-graph-slot-indirection.md),
  [ADR 0022](../adr/0022-two-kv-formats-bf16-as-oracle.md),
  [Prefill chunk wall time](2026-09-11-prefill-chunk-wall-time.md)
- Superseded by: none

**Hardware:** RTX 5090, exclusive card (ADR 0006).
**Model:** `qwen3_8_27b_nvfp4full-v2.ninfer`, 64 layers — 16 GQA at indices
3, 7, … 63, and 48 GDN.
**Engine:** `--kv-format hq-e8-2b --max-context 262144 --prefill-chunk 1024
--spec dflash2 --draft-tokens 7`, the production defaults.
**Raw data and harness:** `.scratch/gqa-workspace-memset/`.

## Question

Before every `gqa_attention` (A1) call, `kernel/src/gqa_layer.cu` zeroed the
whole transient workspace — once per GQA layer, on both the eager prefill path
and inside the captured decode graph. The comment carried it back to #84: a
fresh, zeroed workspace keeps stale inactive split partials from being reduced
on the next call.

Under hq-e8-2b that workspace is not mostly partials. Its dominant item is the
rotated-frame BF16 scratch the prompt route materializes the visible history
into — `visible_keys * 4096` bytes at this geometry, which at a 70K frontier is
288 MB **per GQA layer, per chunk**. So: is any of that zeroing load-bearing,
and what does it cost?

## Evidence

### The vendored kernels answer the correctness half

Read against the vendored sources, nothing hq reads is left unwritten:

- `gqa_attention_decode.cuh:217` — the small-T reducer computes
  `active_split_count` **on the device** from `last_pos + 1` and only ever
  reads `split < active_split_count`.
- `gqa_attention_decode_bf16.cuh:151` — in the `split >= active_split_count`
  arm, the hq route calls `write_neutral()` ("the hq route's public partials
  contract neutralizes inactive splits"); only the **BF16** route "leaves them
  unwritten and relies on the engine's zero-initialized partial workspace".
- `gqa_attention_prefill_hq_routes.cuh:73-99` — the prompt route fills
  `band_rows` of `scratch_k`/`scratch_v` before the FA2 kernel reads that same
  range, and this engine declares `max_visible_keys` equal to the call's own
  visible history, so the band covers the whole span. The key-split partials
  beside it are written by all `bands_split` blocks the launch starts, which is
  the exact count the reduce kernel reads back.

`kernel/tests/test_hq_route_agreement.cu` now runs every hq shape the engine
dispatches twice — over a zeroed workspace and over one filled with 0x7F — and
requires the two outputs to be **bit-identical**. All 17 arms (prefill W=200,
prefill W=9..16, decode B=1..8 at the decode graph's own wide envelope) report
`0 of N elements differ`.

### The A/B

One cold 70,368-token prompt (71 chunks), sent to a freshly started server per
leg, with `IGNIS_CHUNK_PROFILE` and `IGNIS_CHUNK_PROFILE_LAYERS` on. Five
alternating pairs; the only difference between the two binaries is the
predicate.

Per-run totals could not resolve the effect. Over four earlier pairs the
chunk-level `layers_ms` moved a median −44.8 ms while the spread *within* each
configuration was 171–390 ms, and two of the four pairs moved the wrong way.

The per-layer split is what made it measurable, because the 48 GDN layers are
untouched by the change and act as an in-run control:

| pair | GQA layers | GDN control |
|---|---|---|
| 1 | −177.0 ms (−2.01%) | −19.4 ms (−0.46%) |
| 2 | −196.3 ms (−2.17%) | +20.7 ms (+0.48%) |
| 3 | +8.8 ms (+0.10%) | −254.7 ms (−5.58%) — discarded |
| 4 | −190.9 ms (−2.12%) | −22.2 ms (−0.52%) |
| 5 | −507.7 ms (−5.65%) | +162.2 ms (+3.83%) — discarded |

Discard rule fixed before reading the GQA column: a pair whose control moved
more than 1% is a disturbed run. Both discarded pairs have the GQA number
moving *with* their control, which is what a disturbed run looks like.

Over the three usable pairs: **GQA layer device time −190.9 ms median, −2.12%**,
against 8.6-9.0 s of GQA layer time and ~12.9 s of total layer time. The
request's own wall time was lower without the zeroing in 5 legs of 5, by
187-334 ms of ~15.0 s.

The arithmetic agrees: the removed write is
`16 layers * 4096 B * sum(visible_keys per chunk)` ≈ 162 GB over the run, so
190 ms puts `cudaMemsetAsync` at ~850 GB/s effective — about half this card's
peak, which is what a per-layer memset interleaved with compute looks like.

## Finding

**Observed.** Under hq-e8-2b, A1's output does not depend on what the transient
workspace held when the call began, at every shape this engine dispatches.
Removing the zeroing there is worth 2.12% of GQA layer device time on a 70K
prefill, which is ~1.5% of prefill compute.

**Observed, unexpected.** BF16 does not react to a hostile workspace either, at
any decode width: the reducer's split bound is device-side and never reaches a
slot the launch left unwritten. The BF16 zeroing was kept anyway — its kernel
states the dependency in writing, BF16 is the oracle rather than a serving path
(ADR 0022), and one measurement at one history is not grounds to overrule a
vendored contract.

**Inference.** #84's premise was true of an earlier shape of the op and was
never re-checked against the device-side active-split policy that supersedes
it. The zeroing then grew a second, much larger job nobody sized — the hq
prompt route's scratch planes, which scale with the frontier — because the
engine zeroes the whole span the capacity query reports rather than any
particular item in it.

**Method.** A whole-run total is the wrong instrument for a sub-1% kernel
change on this engine: run-to-run spread is 1.3-3.0% of the same quantity. A
per-layer split with an untouched layer family as the in-run control resolved
a 2% effect from five pairs, three of which survived the control.

## Implications

- The cost scales with the frontier, so it was largest exactly where prefill
  already hurts: at a 70K frontier the zeroing was 288 MB per GQA layer per
  chunk, against 4 MB at the first chunk of a 1K prompt.
- On the decode round it was zeroing bytes `write_neutral()` zeroes again a
  moment later — the same region written twice per layer per token.
- `ignis_gqa_workspace_needs_zeroing` enumerates the format that *may* skip the
  zeroing rather than the formats that need it, so a KV format added later
  keeps the zeroing until someone reads its kernels.

## Limits and unknowns

- Measured at one prompt length (70,368 tokens) on one artifact. The *relative*
  saving grows with the frontier and is negligible for short prompts.
- The decode-side saving (~17 MB × batch per round) was not separately
  measured; it is below this method's resolution.
- The BF16 observation covers decode small-T at a 201-key history only. It does
  not establish that the BF16 prompt route's split partials are safe.
- Nothing here measures the *number* of nodes removed, only the bytes. One
  memset node per GQA layer also left the graph, which this method cannot
  separate from the write it performed.

## Follow-ups

- The decode graph declares `max_visible_keys = max_context_tokens`, so the
  small-T launch reserves 85 split slots at every context and `write_neutral()`
  zeroes the inactive ones on every round. The vendored comment anticipates
  better ("graph calls pass their target-private replay interval"): captured
  graphs per context tier would cut the split count for typical contexts.
- Two sibling items in the same layer bodies, not yet measured: the per-layer
  residual `cudaMemcpyAsync` that exists only to serve the decode graph's
  `left`/`right` ping-pong, and the GDN layer's three strided
  `cudaMemcpy2DAsync` QKV splits. See `.scratch/kernel-opt-2026-09-18/`.
