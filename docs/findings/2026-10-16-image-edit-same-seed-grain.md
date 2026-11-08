# An edit that reuses its input's seed builds the seed's noise up into grain

- Kind: experiment
- Status: current
- Observed: 2026-10-16
- Last verified: 2026-10-16
- Scope: image model / chained edits on ignis-image-2.1-turbo (W8A8, 8-step Turbo) through `/v1/images/edits`; the Playground's image sessions and the `make_image` tool
- Topic: image
- Related: spec [playground/06](../specs/playground/06-image-studio.md) (GitHub #326), spec [image-model/05](../specs/image-model/05-image-serving.md)
- Superseded by: none
- Summary: An edit sent with the seed that made its own input reinforces that noise's imprint: two such steps show grain and three turn a 2K picture to noise, while fresh seeds stay clean; the Playground never reuses an input's seed by default.

## Question

The owner chained ten edits in one image session, each step editing the
last picture. From the seventh on the picture filled with grain that every
"remove the noise" step made worse. Those steps had the Playground's "Keep the
last seed" on: each edit was sent with the seed that had made its own input.
Is the grain the seed's doing, or does any chain of edits drift this way?

## Method

From the session's last clean picture (#6, 2048², made with a seed of its
own), two chains of three edits each, every step editing the previous one's
picture with the prompt "Keep the picture exactly as it is.", `n = 1`, size
2048x2048, no prompt enhancement, the image model loaded and idle:
- **fixed:** every step sent `seed = 1234`;
- **random:** no seed; the server drew one per step.

Each output was compared at 1:1 on a 600 x 340 crop of flat sky.

## Result

| Step | fixed seed | random seed |
|---|---|---|
| 1 | clean | clean |
| 2 | fine grain over the sky | clean |
| 3 | the sky is grain; fabric turns to "peel" | clean |

The owner's own chain shows the same course: clean through the steps with
seeds of their own (#2 to #7), faint grain at the second step that repeated
the seed (#8), heavy at the third (#9), worse at each step after (#10, #11),
and prompts asking to remove the noise did not help, since each started again
from the same seed.

The mean absolute Laplacian of the sky crop did **not** separate the chains
(fixed 9.2, 6.8, 12.0; random 8.7, 9.0, 9.1): the grain is blotchy at the
scale of a few pixels and a 3 x 3 Laplacian reads it no better than the
model's own fine texture. The comparison was by eye, on the crops.

## Interpretation

An 8-step flow-matching edit does not wipe its starting noise out entirely:
the picture keeps a faint imprint of it. Given that picture back as its
condition and the *same* starting noise, the model finds the pattern twice
and reinforces it; each further step multiplies it. With a fresh seed the
imprints are uncorrelated and stay below what the eye sees. Whether the W8A8
quantization makes the imprint stronger than the bf16 reference would has not
been measured.

## Consequence

- By default the Playground never sends, for an edit, a seed behind its
  inputs — the seed of an input or of any picture it was made or marked
  from — and blanks the seed field after each step (`seedClash`, spec
  playground/06 AC 18). "Keep the last seed" is gone; in its place an opt-in
  "Same seed between edits", off by default, sends that seed with a warning
  that says why it is not advised (AC 18).
- `make_image` drops a seed that made one of its references and tells the
  model so; its prompt tells the model not to pass one.
- Other API clients get no such guard from the server.
- Grain already built up is not removed by more edits; going back to the last
  clean picture and editing again with fresh seeds is.

## Reproduce

In a browser on the Playground's origin, from a clean 2K picture `blob`:
`POST /v1/images/edits` with `image[] = blob`, `prompt = "Keep the picture
exactly as it is."`, `size = 2048x2048`, `prompt_enhancement = false`, and
`seed = 1234`; feed each answer's picture into the next request. Three steps
show the grain; without `seed` they do not.
