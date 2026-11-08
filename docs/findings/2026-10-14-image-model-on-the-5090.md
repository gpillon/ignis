# The image leaf meets the survey's bars, and its FP8 GEMMs at ~70% of theirs decide a step

- Kind: experiment
- Status: current
- Observed: 2026-10-14
- Last verified: 2026-10-15 (the timing test again, after #324 moved the tower's K-1152 linears to MMA)
- Scope: kernel / the image leaf's denoising step, condition encode and VAE decode on the RTX 5090 (sm_120)
- Topic: image
- Related: spec [image-model/04](../specs/image-model/04-the-image-leaf.md) AC 10 (GitHub #321), [Image-model kernels on the 5090](2026-10-13-image-model-kernels-on-the-5090.md), [ADR 0050](../adr/0050-ignis-serves-an-image-model.md)
- Superseded by: none
- Summary: The image leaf's 8-step DiT loop takes 1.20 s at 1024² and 6.11 s at 2048², at the survey's bars; at 2048² a step is 59% FP8 GEMM, at 67-73% of the literature's FP8 bar, and 33% 8-bit attention.

## Question

Qwen-Image-2.1-Turbo now runs end to end on ignis's image leaf. The
[2026-10-13 survey](2026-10-13-image-model-kernels-on-the-5090.md) set
these bars:
- ~1.1 s for the 8-step DiT loop at 1024², and ~6 s at 2048²;
- per kernel class: FP8 GEMM ~690-740 TFLOPS, 8-bit attention ~620-640
  TOPS, bf16 ~210-240 TFLOPS.

Two questions:
1. How long does each phase of a job take: the encoder, the prefix, each
   step, the decode?
2. How close is each kernel class to its bar, and which one decides the
   time?

## Evidence

**The setup.**
- Commit dbc760e (branch `image-leaf-321`), the
  `ignis_image_2_1_turbo-v2` artifact.
- The RTX 5090 under the swarm's GPU lock in timing mode. No other GPU
  client ran, and no build. The host's PCIe is Gen 3.
- The board sat at its 575 W limit, SM clocks 2.4-2.9 GHz.
- Times are host wall clock around each leaf call. Each call drains the
  stream before it returns.

**The jobs.**
- The test is `crates/server/tests/image_leaf_timing_gpu.rs`, run with
  `IGNIS_IMAGE_TIMING=1`. It makes the leaf calls a server job makes.
- `encode_condition_timed` splits the encode into phases and drains the
  stream after each.
- Each job runs twice, and the second run is the one reported.
- The condition image is the oracle's 1024² edit image, repeated. The
  server encodes a condition at about 1024² of area.

| job | prefix tokens | encode_condition, ms (VAE / tower / text encoder / txt_in+img_in / prefix pass) | a step, ms | DiT loop | VAE decode | job |
|---|---|---|---|---|---|---|
| t2i 1024² | 31 | 42 (0 / 0.7 / 21.6 / 0.2 / 19.3) | 145-152 | 1.20 s | 134 ms | 1.37 s |
| t2i 2048² | 31 | 42 (0 / 0.7 / 21.6 / 0.2 / 19.2) | 747-802 | 6.11 s | 646 ms | 6.79 s |
| edit, 1 image, 2048² | 4,122 | 469 (37 / 219 / 41 / 0.4 / 157) | 809-816 | 6.50 s | 646 ms | 7.62 s |
| edit, 4 images, 2048² | 16,428 | 2,162 (150 / 1,040 / 163 / 0.9 / 749) | 982-991 | 7.91 s | 645 ms | 10.72 s |

**After #324** (2026-10-15, branch `followups-324-gpu` at 0dfc97c, the same
test and artifact, the GPU lock in timing mode): the tower's K-1152 linears
run on MMA. Only the edits' encode moved:

| job | encode_condition, ms (VAE / tower / text encoder / txt_in+img_in / prefix pass) | DiT loop | job |
|---|---|---|---|
| t2i 1024² | 42 (0 / 0.7 / 21.7 / 0.2 / 19.4) | 1.19 s | 1.36 s |
| t2i 2048² | 42 (0 / 0.8 / 21.6 / 0.2 / 19.2) | 5.99 s | 6.69 s |
| edit, 1 image, 2048² | 286 (34 / 45 / 40 / 0.4 / 154) | 6.63 s | 7.57 s |
| edit, 4 images, 2048² | 1,258 (146 / 169 / 156 / 0.9 / 727) | 7.89 s | 9.79 s |

The DiT loop's run-to-run spread is about ±3%:
- a dry run 40 minutes earlier: 1.18 s and 5.94 s;
- the free runs of AC 9 in the same hold: 1.18 s and 5.91 s.

The 2048² steps drift up by up to 7% across the 8 steps (747 to 802 ms in
the t2i run) as the board holds its power limit.

**The kernel classes.** The kernel leaf's benches ran in the same hold:
- `ignis_image_gemm_bench --tokens 4096|16384`;
- `ignis_image_attn_bench`;
- `ignis_image_elementwise_bench`;
- `ignis_image_vae_bench`.

They time each op alone at the leaf's shapes, with CUDA events.

| class | case | throughput | bar | of the bar |
|---|---|---|---|---|
| FP8 W8A8 GEMM | a DiT block's 7 linears + 4 quantizations at 4,096 / 16,384 tokens | 498 / 507 TFLOPS | 690-740 | 67-73% |
| | each GEMM alone (q, k, v, out; SwiGLU gate, up, down; fused variants) | 479-556 TFLOPS | | 65-81% |
| 8-bit attention | the target's call at 1024², no condition image (4,096 x 4,296) | 496 TOPS | 620-640 | 78-80% |
| | the target's call at 2048², 0 / 1 / 10 condition images | 564 / 567 / 575 TOPS | | 88-93% |
| BF16 | the VAE decoder's convolutions at 1024² / 2048², the encoder's at 1024² | 173 / 177 / 169 TFLOPS | 210-240 | 70-84% |
| | the best stage, a 3x3 conv after the upsample | 190-197 TFLOPS | | 79-94% |

**Rebuilding a step from the benches.** Each line adds the GEMM, the
attention's target calls and the elementwise ops (ms), against the median
measured step:
- t2i 1024²: 114.8 + 21.6 (with the quantizers) + 8.2 = 144.6, against
  150.0;
- t2i 2048²: 450.9 + 252.6 + 55.6 = 759.1, against 758.0;
- edit with 1 image, 2048²: 450.9 + 312.8 + 55.6 = 819.3, against 813.2;
- edit with 4 images, 2048²: 450.9 + about 493 + 55.6 = about 999,
  against 988.6. The attention here is interpolated in keys between the
  bench's 1- and 10-image cases.

Raw output (untracked, in the `image-kernels-319` worktree's `.scratch/321e/`):
`ac10-image_leaf_timing_gpu.log` (UTF-16), `bench-gemm-4096.log`,
`bench-gemm-16384.log`, `bench-attn.log`, `bench-elementwise.log`,
`bench-vae.log`.

## Finding

- **The DiT loop meets the survey's bars:** 1.20 s at 1024² (bar ~1.1 s)
  and 6.11 s at 2048² (bar ~6 s), within the run-to-run spread of each.
- **The FP8 GEMMs decide a step.** At 2048² a step is 59% GEMM, 33%
  attention and 7% elementwise ops. The rebuilt sum lands within 0.1-1.1%
  of the measured step at 2048² and within 4% at 1024², so nothing else
  hides in it.
- **The GEMMs run at 67-73% of the literature's FP8 bar.** At that bar the
  2048² step would drop from 758 to about 630 ms, and the loop to about
  5.0 s. The 8-bit attention is already at 88-93% of its bar at 2048², and
  78-80% at 1024², where its calls are short.
- **The edit's encode is the vision tower's**: 47-48% of `encode_condition`
  at 1 and 4 images, 219 ms an image. That is about 22 TFLOPS effective,
  the K-1152 linears on SIMT that spec 03 (320c) names, and the farthest of
  any kernel from a bar. The prefix pass is 33-35%.
  - **Since #324** those linears run on MMA: the tower is 42-45 ms an
    image, 13-16% of `encode_condition`, and the prefix pass (54-58%) leads
    it. The 4-image edit's job drops from 10.72 s to 9.79 s.
- **The VAE decode** at 2048² is 646 ms against its convolutions' 383 ms
  alone. The halo's recompute in bands (1.33x) explains about 509 ms. The
  convolutions themselves run at 70-84% of the bf16 bar.
- **Condition images cost the step through the attention only.** Each
  1024² condition image adds about 4,100 keys: +55 ms a step at 2048², and
  +231 ms with four. The GEMM work does not grow, since the prefix's K/V is
  stored once.

## Implications

- A faster image at 2K means faster FP8 GEMMs first. Closing the gap to
  the bar is worth about 1 s of the 6 s loop. The attention has little
  left.
- For edits, the tower is the next cost after the loop. With four
  condition images it is a second of a 10.7 s job. Since #324 it is
  0.17 s, and the prefix pass (0.73 s with four images) is next.
- The decode's halo recompute and its per-band weight packing are a
  bounded ~0.2 s at 2K. Carrying halo rows between bands would remove most
  of it.
- None of this is a defect: the bars are met. These are the levers, if
  an image's time becomes the goal.

## Limits and unknowns

- **The benches time each op alone**, not inside the leaf's program (nsys
  hangs on Windows). Their sums agree with the measured steps, but the
  split of the ~5 ms a step at 1024² they leave out (the modulation,
  img_in, the head, launch gaps) is not measured.
- **One run**, in one hold. The spread above is from three readings of
  the loop.
- **The 4-image attention is interpolated.**
- **The quantizers are measured at 1024² only**: about 3 ms a step there.
  At 2048² only the target calls are counted.
- **"% of the bar"** compares against published numbers for other
  kernels and shapes, at whatever clocks they ran (the survey's caveats
  apply).

## Follow-ups

None required (AC 10 is information, not a gate). The candidates, should
the owner want speed:
- the GEMM tile configuration at 16K tokens;
- the tower's K-1152 linears (done, #324);
- halo carrying in the decode.
