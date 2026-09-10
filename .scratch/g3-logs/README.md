# G3 gate records (GitHub #104)

The C=1 / C=4 / ITL records behind the G3 verdict. The verdict itself lives
in `.scratch/REVIEW-2026-09-05.md` §6 Phase 3.

Everything here was produced by `ignis-bench g3` against
`F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer`, prompts cut from
`F:\ai\q38\ninfer\bench\fixtures\bench_corpus.ids`, on a free RTX 5090, one
session (`g3-20260910T001050Z`), reference and ignis measured back to back
with no other process touching the GPU between legs (ADR 0015).

## The verdict

| file | what it is |
|---|---|
| `g3-reference-20260910.json` | The reference leg: `ninfer-serve` in the owner's production profile (hq-e8-2b KV, 1024 chunk, CUDA graphs, prefix reuse). C=1 65.5 tok/s, C=4 aggregate 9.8 tok/s, ITL p50/p95/p99/max 17.22/177.93/193.83/214.48 ms over 2,022 intervals (10 cold prefillers), all cold, 2026-09-10 00:13. |
| `g3-ignis-chunk1024-20260910.json` | The ignis leg at prefill-chunk 1024 (BF16 KV, CUDA graphs, K=1 — same tree as the GPU profile run this session). C=1 71.7 tok/s, C=4 aggregate 20.5 tok/s, ITL p50/p95/p99/max 67.33/201.14/211.37/235.93 ms over 1,678 intervals, all cold, 2026-09-10 00:18. **This is the recorded run.** |
| `g3-verdict-chunk1024-20260910.json` | `ignis-bench g3-gate` over the two above: C=1 ratio 1.095 PASS, C=4 ratio 2.089 PASS, ITL p95 ratio 1.130 **FAIL** (tolerance is <= 1.1). |
| `g3-ignis-chunk512-20260910.json` | A comparison run at prefill-chunk 512, same session: C=1 71.3 tok/s, C=4 aggregate 23.9 tok/s, ITL p50/p95/p99/max 44.62/210.05/228.38/256.97 ms over 1,140 intervals. |
| `g3-verdict-chunk512-20260910.json` | The same verdict check over the chunk=512 run: ITL p95 ratio 1.181, *worse* than chunk=1024. Kept as the evidence that narrowing the chunk did not fix the tail — p50 fell (as chunk-proportional cost predicts) but p95 rose, pointing at a fixed per-chunk-boundary cost as the tail driver rather than chunk-processing time itself. |

Chunk=1024 was tried first because it is the reference's own chunk width and
the width `ignis-server` reserves program scratch for at load (its default).
Chunk=512 was tried once, as a check on the "chunk width is the one dial
that moves p95" premise from spec `03-serving-loop.md`; it made the gate
worse, not better, so the search stopped there rather than turning this
gate run into an ad hoc parameter sweep (spec 03 explicitly rules out an
adaptive chunk-width policy in this phase). The ITL p95 gap is filed as
GitHub #110, not waived.

**C=1 and C=4 pass. ITL does not.** The functional anti-serialization
property (`crates/core/tests/interleaving.rs`) is green regardless, per the
gate's own design: the measured ITL ratio and the functional property are
independent, so a ratio failure does not implicate the interleaving logic
itself — see #110 for what it does implicate.

## The other two gate non-negotiables

Both re-run on this tree, in the same GPU-exclusive sitting, before the
measurement legs above:

- G2 correctness: `teacher_forced_canary_agreement_meets_the_g1_floor` and
  `chunked_and_per_token_prefill_agree_on_a_long_prompt`, both green
  (`scripts/gpu-profile.ps1` output, not separately archived here).
- GPU profile (ADR 0006): `kernel/build.ps1 -Test` 33/33 kernel op tests
  passed; `cargo test --workspace --features cuda -- --ignored
  --test-threads=1` all green, zero failures, zero skips.
- The CPU-only anti-serialization property test
  (`a_decode_round_accompanies_every_chunk_while_a_lane_is_decode_ready`)
  green under plain `cargo test`, no GPU needed.
