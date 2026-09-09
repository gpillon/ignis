# G2 gate records (GitHub #88)

The TTFT records behind the G2 verdict, and the evidence that invalidated the
first attempt at it. The verdict itself, with its two recorded deviations,
lives in `.scratch/REVIEW-2026-09-05.md` §6 Phase 2.

Everything here was produced by `ignis-bench ttft --corpus` against
`F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer` (v2), prompts cut
from `F:\ai\q38\ninfer\bench\fixtures\bench_corpus.ids`, on a free RTX 5090.

## The verdict

| file | what it is |
|---|---|
| `g2-reference.json` | The reference leg: `ninfer-serve` in the owner's production profile (hq-e8-2b KV, 1024 chunk, CUDA graphs), 8K median 853.6 ms, 32K median 4490.3 ms, 5 samples each, all cold, 2026-09-08 23:59. |
| `g2-ignis-release-20260909.json` | The ignis leg on `4d03d40` with the leaf rebuilt: 8K 749.6 ms, 32K 3819.7 ms, 5 samples each, all cold, 2026-09-09 02:24. |
| `g2-verdict-release-20260909.json` | `ignis-bench g2` over the two above: ratios 0.878 and 0.851, PASS. |

**`g2-reference.json` is a regression / sanity fixture. It is never the live
side of a gate.** ADR 0015 requires both legs measured live; a future gate run
re-measures the reference rather than comparing against this file. It is kept
so a later ignis change can be sanity-checked against a known reference shape
without holding the GPU twice.

## The invalid first attempt (GitHub #93, root cause #94)

| file | what it is |
|---|---|
| `probe-1k.json` | 1,024-token cell, 18 684 ms. |
| `probe-8k.json` | 8,192-token cell, failed — the server's hardcoded 30 s `request_timeout` (#95) cut the SSE mid-prefill at ~34 s. |

Those two are the signature of a stale kernel archive, not of the engine: the
binary under measurement carried a pre-#84 kernel that still prefilled a span
one token at a time, 18.2 ms/token, because `crates/artifact/build.rs`
rebuilt the leaf only when `kernel/build/*.lib` was missing (#94). Kept as the
worked example of what that failure looks like from the outside — a clean,
perfectly linear per-token cost that scales 8x from the 1K cell to the 8K one.

## The build-profile control

| file | what it is |
|---|---|
| `probe-1k-8k-release-build-20260909.json` | Release build after the leaf rebuild: 1K 109.1 ms, 8K 757.2 ms. |
| `probe-1k-debug-build-20260909.json` | Debug build, same tree: 1K 106.9 ms. |

Taken to rule out `cargo` profile as the explanation before the stale archive
was found. Debug and release land within noise of each other, because the
prefill cost is device work: the build profile of the Rust host code was never
the variable.
