# The image job's per-step latent transposes cost 14-22 ms at 2048x2048, and an edit decoded its condition images one core at a time

- Kind: experiment
- Status: current
- Observed: 2026-11-25
- Last verified: 2026-11-25
- Scope: server / `images::leaf` (the latents' host transposes) and `images::routes::decode_conditions`
- Topic: image
- Related: [GitHub #735](https://github.com/gpillon/ignis/issues/735), [The image job's PNG encode on the host](2026-11-25-image-png-encode-on-the-host.md), [The image leaf meets the survey's bars](2026-10-14-image-model-on-the-5090.md)
- Superseded by: none
- Summary: A power-of-two latent grid (2048x2048) made the leaf's `channels` transpose cost 14-22 ms a step (0.4 ms at 2752x2752); tiling it takes 0.17 ms. Ten condition images took 903 ms to decode and resize on one core, 114 ms side by side.

## Question

[The PNG encode](2026-11-25-image-png-encode-on-the-host.md) was the largest host cost
after the leaf's decode. What else does the host do between the GPU calls of a job,
and what does it cost?

## Evidence

Release builds, one 24-thread x86-64 Linux host, best of 3-7, the repo's own functions
(`packed`, `channels`, `decode_conditions`) timed from in-module tests that were not
kept; synthetic 2048x2048 photo-like pictures.

**The latents' transposes.** `ImageRunner::denoise_step` takes the job's host latents,
channel-major, and the leaf's slot is `[y · w + x][c]`, so every step runs `packed`
(host to leaf) and `channels` (leaf to host) over 64 channels:

| target | `channels` before | `channels` after | `packed` before | `packed` after |
|---|---|---|---|---|
| 1024x1024 | 1.26 ms | 0.04 ms | 0.13 ms | 0.06 ms |
| 2048x2048 | 14.3-22.1 ms | 0.17 ms | 0.54 ms | 0.26 ms |
| 2752x2752 | 0.46 ms | 0.30 ms | 0.99 ms | 0.46 ms |

The 2048x2048 grid is 16,384 cells, so the 64 channels' writes sit 64 KiB apart and map
to the same cache sets: the same loop over 29,584 cells (2752x2752) takes 0.4 ms. It
ran at every one of the 8 steps. A tile of 128 cells was the best of 32 to 512 at every
size, and the result is a permutation, equal to the plain one value for value.

**The edit's condition images.** `decode_conditions` decoded and resized them one after
another. 2048x2048 PNGs, each resized to ~1 MP:

| images | serial | side by side |
|---|---|---|
| 1 | 89 ms | 90 ms |
| 2 | 174 ms | 92 ms |
| 4 | 350 ms | 98 ms |
| 10 | 903 ms | 114 ms |

Measured and left alone: `TargetLatents::noise` 12.2 ms at 2048x2048 (3.1 ms at
1024x1024).

## Finding

Observed: at 2048x2048 the transposes cost 14-22 ms a step, ~115-180 ms of a ~6.8 s job
(2-3%), and ~10 ms of a 1.37 s job at 1024x1024. Tiling removes it and slows no size
tried.

Observed: the condition images' decode and resize scale with the images on a host with
the cores; one image is unchanged.

Inferred, not measured: the cache conflict is the cause of the 50x gap between the two
grids; the tile removes it, and the other grid sizes in the table agree. The saving on
the 5090 host's CPU is not measured.

## Limits and unknowns

- One CPU, synthetic pictures. The side-by-side gain is bounded by the host's cores.
- The transposes' PCIe transfers (4 MB each way a step at 2048x2048, ~0.7 ms) remain;
  the job still round-trips its latents every step.
- No GPU run.

## Follow-ups

- Keep the latents on the device between steps: needs `ImageRunner`'s signature to change.
