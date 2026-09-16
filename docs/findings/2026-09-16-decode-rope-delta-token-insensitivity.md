# Decode rope delta barely moves greedy tokens

- Kind: experiment
- Status: current
- Observed: 2026-09-16
- Last verified: 2026-09-16
- Scope: kernel / multimodal decode rotation, GPU test design
- Related: https://github.com/gpillon/ignis/issues/194, https://github.com/gpillon/ignis/issues/178, https://github.com/gpillon/ignis/issues/195, https://github.com/gpillon/ignis/issues/201
- Superseded by: none

## Question

#194 needed a GPU test proving that a snapshot blob carries a sequence's
`rope_delta`: restore a multimodal sequence and check it decodes the tokens
it would have decoded unevicted. That test proves the delta crossed only if a
sequence restored *without* it would decode something else. How far does the
decode rotation have to move before this model's greedy tokens change?

## Evidence

Setup: qwen3.8-27B NVFP4 artifact loaded with vision, BF16 KV, 512-token
context, greedy, one sequence per round (`decode_program_batch`). The prompt
is a chat-rendered user turn asking for "a short, original poem about a
lighthouse keeper who collects lost letters from the sea", prefilled as one
multimodal span (`prefill_program_multimodal`, no media columns). The decode
round rotates at `position + rope_delta` (`kernel/src/step.cu`, staged into
`decode_rope_positions`); that path was read, not assumed.

| Positions | Deltas compared | Tokens compared | Result |
|---|---|---|---|
| first 24 tokens collapsed to 0 | -24 vs 0 | 40 | identical |
| first 24 tokens collapsed to 0 | -24 vs +400 | 40 | identical |
| first 24 tokens collapsed to 0 | -24 vs +200,000 | 40 | differ |
| plain `0..T` on every axis | +200,000 vs 0 | first 12 | identical |
| plain `0..T` on every axis | +200,000 vs 0 | 34 after 6 | differ |

The identical 40-token stream in the first two rows begins
`760, 1363, 472, 11249, 88901, 83616, 11, 72536, ...`. An earlier probe on a
repeated plain-text prompt (half of it collapsed) also gave identical first
12 tokens at deltas of a few tens and a few hundreds against 0; the model was
copying the repetition, so it is weaker evidence and is not tabulated.

The test that came out of this is
`crates/core/tests/seq_snapshot_gpu.rs` (the #194 leg): delta 200,000, a
34-token tail, and an assertion that delta 0 decodes a different tail. Logs:
`.scratch/vision-kv-reuse-run/194/`.

## Finding

Observed: at deltas of the size an image leaves (tens to hundreds of
positions), and even at +400, this model's greedy tokens did not change over
40 decoded tokens. Only an implausible +200,000 changed them, and then only
after about a dozen tokens.

Inferred: the model's positional signal at decode is weak. It rotates 64 of
its 256 head dims (theta 1e7) in only 16 of its 64 layers, and the 48 GDN
layers carry no positions at all. A shift of the query against every key
leaves most of what drives the next token untouched.

## Implications

- A GPU test comparing greedy tokens at a real image's delta cannot tell a
  correct rotation from a dropped one. #194's eviction leg in
  `crates/runtime/tests/cuda_leaf_vision_gpu.rs` is one such test; the
  guarantee that the delta crosses rests on the blob-byte checks and the
  synthetic-delta leg.
- "Rope deltas were non-zero, so the rotation was genuinely under test"
  (#195's GPU run) is weaker than it reads: a non-zero delta does not by
  itself make a token comparison sensitive to it.
- #201 (lifting the one-token tail rule): getting the delta wrong costs less
  text-level damage than a wrong position would suggest, but it is still
  wrong, and it would be hard to catch.

## Limits and unknowns

- One prompt, one model, greedy decoding, BF16 KV, 40 tokens at most. Longer
  generations, sampling, hq-e8-2b and other prompts were not measured.
- Logits were not compared. The delta may move them measurably without
  flipping an argmax, which a near-tie probe would show.
- The realistic image deltas recorded in #195 (-56 to -182) were not run
  directly; -24 and +400 bracket them.
- Answer quality over long image conversations with a wrong delta is not
  established.

## Follow-ups

- A logit-level check of the decode rotation (for example the verify round's
  logits at a real image delta against delta 0) would give a test that sees a
  dropped delta at realistic magnitudes.
