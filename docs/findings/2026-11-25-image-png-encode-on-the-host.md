# The image job's PNG encode took 0.5-1.3 s of host time per 2048x2048 picture at the crate's default level; `Fast` takes ~20 ms

- Kind: experiment
- Status: current
- Observed: 2026-11-25
- Last verified: 2026-11-25
- Scope: server / `images::job::encode_png`, the host work between the VAE decode and the response
- Topic: image
- Related: [GitHub #735](https://github.com/gpillon/ignis/issues/735), [The image leaf meets the survey's bars](2026-10-14-image-model-on-the-5090.md), spec [image-model/05](../specs/image-model/05-image-serving.md)
- Superseded by: none
- Summary: On a 2048x2048 RGBA picture `png` 0.18.1's default level took 555-1291 ms and `Compression::Fast` 17-24 ms at the same size within 1% (6% smooth, 47% on a perfectly flat picture); resize, base64 and the white composite cost under 75 ms.

## Question

[The image leaf's timings](2026-10-14-image-model-on-the-5090.md) stop at the VAE
decode: a 2048x2048 t2i job is 6.79 s there, while the README's served end-to-end
figure is 7.87 s. What does the host do in the ~1.1 s between them, and which of
it is worth removing?

## Evidence

The host work after `vae_decode` is `on_white` (when the background is opaque),
`encode_png`, then base64 in the route. `encode_png` built its PNG with the `png`
crate's defaults. Every figure is `png` 0.18.1 (the version in `Cargo.lock`), a
release build, 2048x2048 RGBA, best of three, one 24-thread x86-64 Linux host.
The pictures are synthetic (low-frequency texture, edges, Gaussian grain of the
stated sigma), generated with Pillow; none is a model output.

| picture | default | `Fast` | size, Fast vs default |
|---|---|---|---|
| photo-like, sigma 1.5 | 1234 ms | 21 ms | 6.63 vs 6.69 MB |
| smooth, sigma 1 | 1142 ms | 22 ms | 5.87 vs 5.55 MB |
| heavy grain, sigma 3 | 555 ms | 17 ms | 9.44 vs 9.54 MB |
| perfectly flat | 590 ms | 24 ms | 3.50 vs 2.38 MB |

`Balanced` (the default) and `High` take the same time and write the same bytes;
`Fast` is a different encoder (fdeflate), recognised by the zlib header's FLEVEL,
0 against 2.

The repo's own `encode_png`, called from a release-profile test on the photo-like
picture: 1291.5 ms before, 20.7 ms after, 6.69 and 6.63 MB.

Measured on the same host and left alone: `resize_rgba` (a condition image,
2048x2048 to 1024x1024) 61 ms, to 1408x1408 74 ms; the hand-rolled base64
8.4 ms for 9.5 MB (the `base64` crate: 3.0 ms); `on_white` ~6 ms. The latent
noise (1,048,576 values at 2048x2048, a Box-Muller pair each) was not timed.

## Finding

Observed: at the default level the PNG encode is more than the GPU's VAE decode
(646 ms at 2048x2048) and ran with the card idle, on the thread that drives the
job. `Fast` removes ~0.5-1.3 s of it.

Observed: size is within 1% on a textured picture, +6% on a smooth one and +47%
on a perfectly flat one.

Inferred, not measured: the gap between the leaf's 6.79 s and the served 7.87 s
is mostly this encode. That host's CPU is not this one, and the full server was
not run here.

## Implications

A served 2048x2048 image should be ~0.5-1.3 s sooner, and a 1024x1024 one a
quarter of that. The README's 7.87 s figure is to be re-measured on the 5090.
A flat picture (a logo, a transparent-background illustration) grows, which a
client on a slow link would feel; a config field or a size-aware second pass
would give it back.

## Limits and unknowns

- Synthetic pictures; real model outputs' size ratios are not measured.
- One CPU. The encode is single-threaded and scales with its clock.
- No GPU run: nothing here touches the leaf, and no served image was timed.

## Follow-ups

- Re-time a served 2048x2048 generation on the 5090 and update the README.
- Decide the flat-picture trade-off (#735).
