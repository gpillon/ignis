# The 27B's vision tower runs every linear on MMA; a large image costs attention

- Kind: experiment
- Status: current
- Observed: 2026-10-15
- Last verified: 2026-10-15
- Scope: kernel / the 27B's media encode (`ignis_media_encode`, `kernel/src/vision_encode.cu`) and the vendored Q4/Q5/Q6 row-split dispatch at the tower's shapes, RTX 5090 (sm_120)
- Topic: vision
- Related: GitHub #324 item 5, [The image leaf on the 5090](2026-10-14-image-model-on-the-5090.md), ADR 0031, ADR 0037
- Superseded by: none
- Summary: Every linear of the 27B's vision tower already runs on MMA at 160-212 TFLOPS: a 1024² image encodes in 40 ms, and from 2048² up the cost is the attention's quadratic work (88% of 3.6 s at 4096²); nothing to change.

## Question

In #324 item 5, the image model's vision tower spent 73% of its kernel time
in the W8 SIMT GEMM at K 1152. The W8 MMA arm could not stage a 72-byte scale
row. The patch that fixed it took one 1024² image from 226 ms to 44 ms. The
27B has the same tower geometry, but its own weights. The artifact's manifest
gives these formats:

- `qkv` [3456,1152] and `fc1` [4304,1152]: Q4G64;
- `output` [1152,1152] and `fc2` [1152,4304]: Q5G64;
- the patch embedding [1152,1536]: Q6G64;
- the merger's [4608,4608] and [5120,4608]: W8.

Does the 27B's encode, for chat and Decide with images, have a bottleneck of
the same kind?

## Evidence

The measurements ran in one GPU hold on main 7a38c999, with the owner's
server stopped.

1. **GEMM harness:** `ops::linear` with synthetic weights of the artifact's
   formats, timed with CUDA events (3 warm-ups, then the mean of 10-30 reps).
   Each shape also ran as the image model's W8 (after #324) and on each MMA
   arm directly.
2. **Encode wall time:** `encode_media` on the real 27B artifact for square
   images (1 warm-up, then the median of 5), loaded with the processor's item
   bound of 16,384 merged tokens.
3. **Per-kernel sums:** nsys (`--trace=cuda`) over the same encodes.

The linears for one 1024² image: 4,096 patches, and the merger at T 1,024.

| linear | 27B format, route | 27B µs | TFLOPS | image model W8 µs |
|---|---|---:|---:|---:|
| patch embedding | Q6, `q6_mma_r64_c128` | 84.1 | 172 | 93.2 |
| qkv | Q4, `q4_mma_r64_c128` | 189.8 | 172 | 188.4 |
| output | Q5, `q5_mma_r64_c128` | 65.6 | 166 | 71.7 |
| fc1 | Q4, `q4_mma_r64_c128` | 233.4 | 174 | 229.4 |
| fc2 | Q5, `q5_mma_r64_c128` | 245.9 | 165 | 285.2 |
| merger fc1 | W8 | 271.4 | 160 | — |
| merger fc2 | W8 | 271.3 | 178 | — |

- **Larger T:** at 16,384-65,536 patches, the same routes reach 170-212
  TFLOPS. The W8 tower reaches 163-189 TFLOPS there.
- **One tower's block linears plus the patch embedding:** 19.9 ms on the 27B
  and 21.0 ms on the W8 tower.
- **Routes:** at every tower width (4,096 patches or more), the dispatch picks
  the faster of the two MMA arms. `c128` beats `c64` by 1.2-1.6x. The SIMT
  arms end at T 36 (Q4), T 120 (Q5) and T 96 (Q6).
- **Scale rows:** the Q4/Q5 MMA kernels stage the 36-byte G64 scale rows with
  4-byte `cp.async`. K 1152 therefore excludes nothing, unlike the W8 rows.

The encode, wall time against kernel time:

| item | patches | wall, median | kernel sum | attention | linears |
|---|---:|---:|---:|---:|---:|
| 1024² | 4,096 | 39.9 ms | 38.4 ms | 14.4 ms (37.5%) | 21.2 ms (55%) |
| ten 1024², one call each | 10 × 4,096 | 405-409 ms | — | — | — |
| 2048² | 16,384 | 310 ms | not captured | — | — |
| 4096² (the item bound) | 65,536 | 3.64-3.76 s | 3.67 s | 3.24 s (88%) | 0.32 s (9%) |

- **The image model**, for comparison, takes 44 ms for one 1024² image and 434
  ms for ten. Its tower also runs three DeepStack mergers.
- **Attention at 4096²:** one segment of 65,536 patches costs
  4 · S² · 72 · 16 heads · 27 blocks = 5.34e14 FLOP. That runs in 3.24 s, or
  165 TFLOPS. At 1024² the attention runs at 145 TFLOPS.
- **Elementwise kernels at 65,536 patches:** `add_bias` and GELU exceed the
  vendor's cache-size gate (`kBf16x8CacheSizedMaxElements`) and fall to their
  bf16x2 kernels. That costs 68 ms an encode (1.8%).

Raw logs, the harness and the scratch timing test are in
`.scratch/27bvis/` (local to the clone that ran them).

## Finding

The 27B's tower has no bottleneck of item 5's kind:

- **Linears:** every one runs on an MMA arm at 160-212 TFLOPS. That is at or
  above the W8 tower after #324, and the dispatch already picks the faster arm
  at every tower width.
- **A 1024² image** encodes in 40 ms, 10% under the image model's 44 ms. Of its
  38.4 ms of kernels, 55% are linears and 37% attention.
- **From 2048² up**, the encode's cost is attention: 88% at 4096², or 3.6 s.
  Attention is quadratic in one image's patches. The shared flash kernel already
  runs it at ~165 TFLOPS, close to the GEMMs' rate.

There is nothing to fix in the same way, so nothing changed.

## Implications

- The 27B's tower needs no vendor patch. Its Q4/Q5 rows at K 1152 are not the
  W8 case.
- A large image on either model costs attention FLOPs. Its lever is fewer
  patches (the processor's pixel bounds and the request's budget), not a GEMM
  route.

## Limits and unknowns

- **Synthetic weights:** the harness uses weights of the same format and
  layout. A GEMM's time does not depend on the values.
- **Wall time** includes the pageable host-to-device copy of the patches (12.6
  MB at 1024²) and the stream sync. The kernel sums leave both out.
- **Variance:** one hold, harness means, and desktop WDDM clients on the card
  (3.2 GB at the start). The two wall-time runs agree within ~3%.
- **2048²** has no nsys capture. Its split is not reported.

## Follow-ups

none
