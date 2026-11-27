# The 27B's MTP head's state held to its weights on the RTX 5090: committed rows 0.0038 and 0.0034 from the double-precision head against bounds of 0.0096 and 0.0087, a chain-built row 0.35

- Kind: experiment
- Status: current
- Observed: 2026-11-15
- Last verified: 2026-11-15
- Scope: speculative decoding / the 27B's MTP head (spec #618) on the RTX 5090, the NVFP4 artifact, BF16 KV: the
  head's rows after a verify round, after an image and against plain rounds, each as a relative L2 distance from the
  head recomputed in double precision from the stored weights
- Topic: decode-27b
- Related: https://github.com/gpillon/ignis/issues/620,
  https://github.com/gpillon/ignis/issues/621,
  https://github.com/gpillon/ignis/issues/624,
  https://github.com/gpillon/ignis/issues/618,
  [ADR 0095](../adr/0095-the-27b-has-two-drafters-and-its-mtp-head-keeps-attention-state-over-the-whole-context-and-a-continuation-hidden.md),
  [the head measured on the RTX 5090](2026-11-15-the-27bs-mtp-head-measured-on-the-rtx-5090.md),
  [the head drafts as the reference engine's](2026-11-15-the-27bs-mtp-head-drafts-as-the-reference-engines.md)
- Superseded by: none
- Summary: RTX 5090, 27B NVFP4, BF16 KV: the MTP head's committed rows stand 0.0038 (keys) and 0.0034 (values) relative L2 from the f64 head, bounds 0.0096 and 0.0087; a chain-built row 0.35; after an image 0.0037 and 0.0031.

## Question

ADR 0095 holds the head's state to the committed run in each round: its rows at the run's positions are the head's in
double precision over what the run consumed, within the BF16 bounds of the head's checks, and a row built as the
round's chain builds its own, from the head's hidden in place of the target's, must land far outside them. How far
inside the bounds do the right rows stand, and how far outside do the wrong ones, in a verify round, against plain
rounds of the same tokens, and after an image?

## Evidence

The figures are the ones ADR 0095 recorded as tickets #620 to #624 landed, on 2026-11-15, on the RTX 5090; they moved
here unchanged when the ADR was brought to the short format (its text of that time is
`git show records-2026-11-27:docs/adr/0095-the-27b-has-two-drafters-and-its-mtp-head-keeps-attention-state-over-the-whole-context-and-a-continuation-hidden.md`).
Each is a quantity a GPU test computes, holds to its bound and prints: `crates/core/tests/mtp_round_gpu.rs` for the
round and `crates/core/tests/mtp_vision_gpu.rs` for the image, under the GPU profile. The reference is
`ignis_artifact::f64_reference::evaluate_mtp_head` (`evaluate_mtp_head_at_axes` after an image), fed the exact BF16
inputs the device read. The bounds are `crates/core/tests/support/mtp_bounds.rs`: a relative L2 of 2^-8 times the
square root of the BF16 roundings on the path, 6 for a key and 5 for a value, so 0.0096 and 0.0087.

**In a verify round** (the canaries, drafts accepted and rejected):

| quantity | measured | bound |
|---|---|---|
| committed rows against the f64 head over the run, keys | 0.0038 | 0.0096 |
| the same, values | 0.0034 | 0.0087 |
| a chain-built row, the nearest | 0.35 | must be far outside |

**Against plain rounds held to the same tokens**: 0.087 and 0.103 relative L2, in the ADR's order the drafted rows'
distance from the plain rounds' rows and the distance a prefill of the same text stands from them. The drafted rows
are held within twice the prefill's distance (two correct routes), and a chain-built row past both.

**Rounds part at a near tie**: the same load committing the same canaries one token a round, at extent 0, leaves the
drafted text at a near tie at one window and two lanes. A verify round's target hidden at another extent is not the
same bits.

**The moved pick**: where the near-tie rule refuses a greedy divergence between spec-on and spec-off and the verify
substrate, given the same drafts, makes spec-on's pick, the two picks are held within 16 BF16 ulps at the prefill
routes' top logit. Measured once on the canaries, at four lanes: 3 ulps.

**After an image** (number.png, rope delta -182):

| quantity | measured |
|---|---|
| the prompt's rows at and around the image, keys | 0.0037 |
| the same, values | 0.0031 |
| control: the rows built from the placeholder ids' embeddings, values | 0.33 |
| control: the rows rotated at their KV positions, keys | 0.26 |
| the frontier's hidden | 0.0056 |
| the chain's first scores | 0.0050 |
| control: the commit's rows and the plain round's row rotated at the bare positions | 0.49 and 0.50 |

The first verify round after the image drafted its whole window and committed all of it, and its commit's rows and
the plain round's row after it stand within the bounds. On the canaries, the head so built commits 4.67 tokens a
full-window round on the text after an image.

## Finding

Observed: the right rows stand at about 0.4 of their bounds (0.0038 of 0.0096, 0.0034 of 0.0087; after an image
0.0037 and 0.0031), and every wrong construction that was tried lands 27 to 52 times its bound: a row built from the
head's own hidden 0.35, the placeholder ids' embeddings 0.33, a rotation at the KV positions 0.26, a rotation at the
bare positions 0.49 and 0.50. The drafted rows stand closer to the plain rounds' rows (0.087) than a prefill of the
same text does (0.103).

Inferred: the bounds tell a right row from each of the likely mistakes with a wide margin on both sides, so they are
neither loose enough to pass a wrong row nor tight enough to fail on the card's own roundings. Bit for bit against a
sequence that committed the same tokens by other rounds is not a property the engine has: two correct routes already
stand 0.1 apart.

## Implications

- A change to the head's kernels, to its routes on an RTX 30 or 40 card or to the verify round is checked by these
  tests against the same bounds; a right row that moves toward its bound is a signal before it is a failure.
- A new control (another way to build a wrong row) belongs beside these: it should land as far out as they do.

## Limits and unknowns

- One card, one artifact (NVFP4), BF16 KV, the canary prompts and one image. The 16 GB artifact and the route an RTX
  30 or 40 card runs are held to the same bounds by `crates/core/tests/mtp_head_gpu.rs`; their distances were not
  recorded.
- The ADR gave the two distances against plain rounds without naming which was which; the order here is its sentence's
  ("the drafted rows stand within twice the distance a prefill of the same text stands from them ... 0.087 and
  0.103").
- The runs' logs were not kept: the figures are the ADR's record of them, and a rerun of the two tests prints them
  again.

## Follow-ups

None.
