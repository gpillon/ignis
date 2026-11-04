# RTX 20, 30 and 40 cards: what in the leaf needs Blackwell, and what an sm_89 compile breaks

- Kind: experiment
- Status: current
- Observed: 2026-11-02
- Last verified: 2026-11-04
- Scope: kernel / every CUDA translation unit of `ignis_kernel` and `ignis_vendor` and its op tests, the build's
  architecture target, the instructions each route needs; serving / `--profile auto` and the card class; models / which
  catalog entry depends on which route
- Related: [GitHub #456](https://github.com/gpillon/ignis/issues/456), [GitHub #457](https://github.com/gpillon/ignis/issues/457),
  [ADR 0005](../adr/0005-performance-first-principle.md),
  [ADR 0010](../adr/0010-vendored-reference-kernels.md),
  [ADR 0031](../adr/0031-vendored-kernel-bottleneck-exemption.md),
  [ADR 0037](../adr/0037-a-vendored-file-may-carry-a-correctness-patch.md),
  [ADR 0060](../adr/0060-the-27b-has-a-trellis-body-for-small-cards.md),
  [ADR 0063](../adr/0063-the-profile-follows-the-card-class.md),
  [dense trellis kernels on the 5090](2026-10-23-dense-trellis-kernels-on-the-5090.md),
  [Metal and AMD backends](2026-10-31-metal-and-amd-backends-seam-and-roads.md)
- Superseded by: none

## Question

The owner asked whether ignis can run on an RTX 20 (Turing, sm_75), 30 (Ampere, sm_86) or 40 (Ada, sm_89) card, or
whether functions or kernels in the engine stop it on those families. That splits into three questions. Which
instructions does the leaf use that those architectures lack, and where? Which models depend on those routes? What
else, outside the kernels, stands in the way?

## Evidence

**Build target.** `kernel/CMakeLists.txt:26` and `kernel/build.ps1:64` set `CMAKE_CUDA_ARCHITECTURES` to `120a` and
nothing else. The binary holds no code for any other architecture. The `a` suffix also means the arch-specific
instructions are not forward-compatible. The toolkit in use, CUDA 13.1 (13.1.80), still lists compute_75 through
compute_121 (`nvcc --list-gpu-arch`).

**Inventory (grep of `kernel/src`, `kernel/vendor/src`, `kernel/include`).** The leaf has no CUTLASS or CuTe. Every
architecture-specific instruction is inline PTX or a CUDA header intrinsic:

| Feature | Where | Minimum |
|---|---|---|
| `mma.sync ... kind::mxf4nvf4.block_scale` (FP4 MMA) | `vendor/src/ops/common/mma.cuh` `mma_nvfp4_e4m3`, called by the NVFP4 W4A4 GEMMs | sm_120a |
| `cvt.rn.satfinite.e2m1x2.f32` (FP4 encode, hand-written) | `vendor/src/ops/linear/nvfp4/nvfp4_codec.cuh`, W4A4 activation quantize | sm_100a / sm_120a |
| `cp.async.bulk.tensor`, `mbarrier.arrive.expect_tx`, `setmaxnreg`, `fence.mbarrier_init.release.cluster` | `nvfp4_w4a4_tma.cuh`, `nvfp4_linear_swiglu_w4a4_tma.cuh` | sm_90 (setmaxnreg: sm_90a / sm_100a / sm_120a) |
| `mma.sync ... kind::f8f6f4` and `kind::mxf8f6f4.block_scale` (FP8 MMA) | `hq_prompt_attention.cu` (the 8-bit band's P V), `image/image_mma.cuh` (image GEMM, `attn8`), `mma.cuh` `mma_fp8_e4m3` | sm_120a; sm_89 has the plain `m16n8k32 .e4m3` form without `kind::` or block scales |
| `cvt.rn.f16x2.e4m3x2`, `cvt.rn.satfinite.e4m3x2.f16x2` | `hq_prompt_attention.cu` | sm_89 |
| thread-block clusters (`cg::this_cluster`, `map_shared_rank`) | `moe_decode_cluster.cu` (the `IGNIS_MOE_DECODE_CLUSTERS` route) | sm_90 |
| `mbarrier.try_wait.parity` | `moe_decode_staged.cu`, `nvfp4_w4a4_tma.cuh` | sm_90 |
| `griddepcontrol` (programmatic dependent launch) | `vendor/src/core/pdl.cuh`, `trellis_linear_internal.cuh`, `moe_router.cu`, `residency.cu` | sm_90 |
| `cp.async`, `ldmatrix`, `mma.sync` bf16 / f16 (m16n8k16), s8 (m16n8k32), tf32 (m16n8k8) | everywhere: GDN, GQA, trellis, `fp8_linear` (W8A16), `nvfp4_a16_mma`, the i8 attention | sm_80 (ldmatrix sm_75) |
| `cuda_fp4.h` / `cuda_fp8.h` conversions (`__nv_fp4x2_e2m1`, `__nv_cvt_*`) | NVFP4 decode (GEMV, small-T, DFlash2 drafter, A16 MMA), FP8 rows | hardware on sm_100a+ / sm_89+, software fallback below (`__CUDA_FP8_INTERNAL_CAN_RELY_ON_PTX_FOR_SHORTTYPESCVT__` in `cuda_fp4.hpp`) |

The SM count `170` (the RTX 5090's) is written into four launch heuristics:
`vendor/src/ops/linear_attention/gated_delta_net/chunked/output.cu` (`kRtx5090SmCount`),
`vendor/src/ops/launcher/rmsnorm.cu` (`kRmsPrefetchBlocks`), `src/image/dit_ops.cu` and `src/image/vae_shortcut.cu`.

**Compile probe.** The whole leaf, with its tests, configured for `CMAKE_CUDA_ARCHITECTURES=89` and built with
`ninja -k 0 -j 8` in a separate build directory (script and full log in `.scratch/arch-probe/`; the build tree,
123 MB, in `H:\ignis-arch-probe\sm89`). It took 337 s. **227 CUDA translation units compiled; 55 failed, 172
passed.** Each failure was classified by its ptxas or front-end messages:

| Root cause | Failed TUs | Which |
|---|---|---|
| `griddepcontrol` only | 40 | 32 vendored (attn/GDN input projections bf16 and NVFP4 decode and small-T, `gdn_gating_proj`, the GQA decode launchers `gqa_attention_decode{,_bf16,_hq_27,_hq_35,_i8}`, `qk_norm_rope`, `rmsnorm`, `rope`, `sigmoid_gate_mul`, `bf16_gemv`, `bf16_small_t`, `nvfp4_gemv`, `nvfp4_small_t`, `w8_*` ×3, `linear_add` bf16 and NVFP4 decode and small-T, GDN `recurrent`, `linear_swiglu` NVFP4 decode and small-T); 6 ours (`trellis_linear.cu`, `trellis_linear_decode.cu`, `trellis_linear_prefill.cu`, `moe_router.cu`, `residency.cu`, `flash_next/verify.cu` through the vendored `recurrent.cuh`); 2 tests (`bench_hq_verify_attention`, `test_hq_verify_exact`) |
| FP4 MMA (+ TMA, `setmaxnreg`, `try_wait`, `cvt.e2m1` in some) | 7 | all vendored W4A4: `nvfp4_w4a4`, `nvfp4_w4a4_tma`, `nvfp4_attn_input_w4a4`, `nvfp4_gdn_input_w4a4`, `nvfp4_linear_add_w4a4`, `nvfp4_linear_swiglu_w4a4`, `nvfp4_linear_swiglu_w4a4_tma` |
| FP8 block-scaled MMA (`kind::mxf8f6f4`, `kind::f8f6f4`) | 6 | `hq_prompt_attention.cu`, `image/fp8_gemm.cu`, `image/attn8.cu`; tests `image_gemm_sweep` (×2 targets), `bench_mma_peak` |
| clusters | 1 | `moe_decode_cluster.cu` |
| `mbarrier.try_wait.parity` | 1 | `moe_decode_staged.cu` |

These passed with no change: `dflash2_drafter.cu`, `nvfp4_a16_mma.cu`, `fp8_linear.cu`, `gqa_layer.cu`,
`gdn_layer.cu`, the GDN chunked kernels, the vision tower (`vision_encode.cu`, which runs the vendored
`ninfer::ops::linear`), `image/encoder_attn.cu` and `image/vae_conv.cu` (both include `image_mma.cuh` but never
instantiate its FP8 forms). `nvfp4_gemv.cu` and `nvfp4_small_t.cu` failed only on `griddepcontrol`: their E2M1
decode compiled through the header's software path.

**Start-up.** No code in `crates/` or `kernel/src` reads the device's compute capability (searched for
`ComputeCapability`, `.major`, `cudaDevAttrComputeCapabilityMajor`). `--profile auto` picks the card class from the
card's total memory alone (`crates/server/src/config/profile.rs:262-264`: below 20 GiB is the 16 GB class, 28-40 GiB
the 32 GB one, 88-104 GiB the 96 GB one).

**Catalog.** `crates/server/src/download/catalog.yaml`: the 27B 32 GB entry is NVFP4 (`qwen3_8_27b_nvfp4full-v2.ninfer`,
19,406,942,468 bytes); the 16 GB entry is the 3 bpw trellis body; Flash-Next is trellis (71.8 GB, the experts in
host RAM on a 32 GB card); the image model is FP8 W8A8; speech and the diarizer run on the CPU.

**Trellis decode rate per SM.** [Dense trellis kernels on the 5090](2026-10-23-dense-trellis-kernels-on-the-5090.md):
with the SMs capped, decode streams ~9 GB/s per SM whatever the cap, so on a small card the speed is set by the tile
decode's instructions, not by DRAM.

## Finding

Observed:

- Today the leaf runs on sm_120 only. That comes from the build target, not from the bulk of the code: 76% of the
  CUDA translation units (172 of 227) already compile for sm_89 unchanged.
- An sm_89 compile breaks on five root causes. One of them, programmatic dependent launch, accounts for 40 of the
  55 failures. 32 of those are vendored files that all reach `griddepcontrol` through a single header,
  `vendor/src/core/pdl.cuh`. Three of our own files cover the other six (`trellis_linear_internal.cuh`,
  `moe_router.cu`, `residency.cu`).
- Only two kinds of route have no counterpart on Ada or Ampere: the NVFP4 W4A4 prefill GEMMs (FP4 tensor cores, TMA)
  and the block-scaled FP8 MMA (the 27B's 8-bit prompt-attention band, the image GEMM and `attn8`). Clusters and
  `try_wait` sit in Flash-Next MoE decode routes that are not its default (it loads `REGISTERS`).
- NVFP4 decode compiles on sm_89: the GEMV, the small-T route, the DFlash2 drafter and the A16 tensor-core route
  (`nvfp4_a16_mma`, ours) get their E2M1 decode from `cuda_fp4.h`'s software fallback.
- A 16 GB RTX 40 card (4060 Ti 16 GB, 4070 Ti Super, 4080) resolves to `rtx50-16gb` under `--profile auto` and starts
  a load that fails at its first kernel launch with CUDA's own "no kernel image" error. A 24 GB card (3090, 4090)
  is refused as "no card class", by accident of the memory ranges.

- **Static shared memory (added 2026-11-04, from the sm_86 build of #456).** Three vendored units declare more than
  48 KiB of static shared memory per block: the W8 linear's small-T route (`w8_small_t.cu`), its split-K route
  (`w8_rowsplit_gemm_splitk.cu`) and the i8 decode attention (`gqa_attention_decode_i8.cu`). ptxas refuses that for
  sm_86 (`0xc000 max`, 49,152 bytes) and accepts it for `120a`. The sm_89 probe above never saw it: those units had
  already failed on `griddepcontrol`, which stops compilation before ptxas reaches the shared-memory check. Once PDL
  is guarded out, these three still fail on an sm_8x target, so they compile for the sm_120a set alone
  (`ignis_vendor_sm120a` in `kernel/CMakeLists.txt`) and a launch of one on another card fails with CUDA's "no kernel
  image is available". On the 27B only its MTP layer, which no load binds, takes the small-T route.

Inferred (not run on any of these cards):

- **RTX 40 (Ada):** feasible. It needs a multi-architecture build, `#if __CUDA_ARCH__ >= 900` guards around PDL
  (no-ops below), the FP8 forms rewritten to sm_89's plain `e4m3` MMA, and routes picked by compute capability.
- **RTX 30 (Ampere):** feasible on the same plan, except that it has no FP8 MMA at all and no hardware e4m3
  conversion. The FP8 paths fall back to FP16/BF16, or to INT8 with a different weight format. sm_86 was not
  compiled; the expected difference from sm_89 is only the e4m3 `cvt` in `hq_prompt_attention.cu`, which fails on
  sm_89 already.
- **RTX 20 (Turing):** not a port. It has no BF16 MMA (only f16 m16n8k8), no `cp.async`, no tf32 or int8 m16n8k32
  MMA, 64 KB of shared memory per block against the 99 KB the kernels are sized for, and at most 11 GB on its
  consumer cards. The engine's activations are BF16 throughout, so this would be another engine.
- **Per model.** The 27B trellis body (16 GB) is the first candidate. `trellis_linear*` and
  `gqa_attention_decode_hq_27` fail on PDL only. Prompt attention already has an FP16 band
  (`--model-prefill-attention fp16`, `IgnisHqPromptPrecision::kFp16`), so `hq_prompt_attention.cu` only needs its
  FP8 forms guarded out of an sm_8x build. The 27B NVFP4 entry needs `nvfp4_a16_mma` widened to every prefill
  geometry in place of W4A4. That changes the numerics, so the KLD gates have to be run again. Without FP4 tensor
  cores its prefill is much slower, and its 19.4 GB file needs a 24 GB card class that does not exist today.
  Flash-Next's default decode route is portable; it needs ≥ 24 GB plus host RAM. The image model maps to sm_89's
  FP8 MMA (its GEMM sets every block scale to 2^0, per `image_mma.cuh`); on Ampere it needs W8A16 or INT8 on a
  compute-bound DiT.
- **Speed (estimate only).** Decode is bandwidth-bound on the NVFP4 GEMVs: a 4090's 1,008 GB/s is ~56% of the
  5090's 1,792, a 3090's 936 ~52%. Trellis decode is instruction-bound per SM. RTX 50 SMs run INT32 on all 128 lanes
  where Ada and Ampere SMs run it on 64, which is a hardware fact to check against the whitepapers, and the tile
  decode is integer-heavy. A 30 or 40 SM may therefore stream well below 9 GB/s. No tokens-per-second figure is
  claimed.

## Implications

- A port to Ada and Ampere means compiling for more architectures and dispatching per feature, not a backend in
  the sense of the Metal finding. The C ABI, the layer structure and most kernels stay as they are.
- Most of the mechanical work is in one vendored header, and ADR 0010 keeps vendored files verbatim. ADR 0031
  (a measured bottleneck) and ADR 0037 (a demonstrated correctness bug) do not cover a patch made for portability.
  The owner has to decide between a new exemption and our own copy under `kernel/src`.
- Several fat-binary targets multiply nvcc time (the sm_89 probe alone took 337 s at 8 jobs) unless sm_8x is a
  separate build of its own.
- `--profile auto` needs a compute-capability gate whatever happens to the port: a clear refusal in place of a raw
  CUDA error on any non-sm_12x card.
- Card classes would need an architecture as well as a memory range: 24 GB (3090, 4090) and 16 GB Ada are classes
  that do not exist today.

## Limits and unknowns

- Nothing ran on an sm_86 or sm_89 card. Compiling is not running: register budgets, occupancy, the header software
  paths and the 172 passing TUs are untested there.
- `pdl.cuh::launch_dependent` passes `cudaLaunchAttributeProgrammaticStreamSerialization` to `cudaLaunchKernelEx`.
  Whether a card below sm_90 ignores it or returns an error was not checked. If it returns an error, every PDL
  launch site, including those the decode graph captures, needs a host-side check.
- sm_86 and sm_75 were not compiled.
- The per-SM trellis rate on Ada and Ampere is unmeasured. A spike on a rented card is the way to get it.
- The hardware figures (bandwidths, INT32 lanes per SM, 99 KB of shared memory per block on 8.6, 8.9 and 12.0) come
  from NVIDIA's published specifications, not from this probe.

## Follow-ups

- [GitHub #456](https://github.com/gpillon/ignis/issues/456): RTX 30 and 40 support (idea).
- [GitHub #457](https://github.com/gpillon/ignis/issues/457): a start on a card the kernels were not built for refuses
  by name, before the load.
- Raw evidence: `.scratch/arch-probe/sm89-ninja-k0.log`, `.scratch/arch-probe/arch-probe.ps1`.
