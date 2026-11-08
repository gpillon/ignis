# On the 5090 others reach ~100% of nominal for bf16 attention and GEMM, 8-bit attention runs 3× faster, and at 2048² the attention decides an image's time

- Kind: research
- Status: current
- Observed: 2026-10-13
- Last verified: 2026-10-13
- Scope: kernel / the image model's GEMM, attention, convolution and fused DiT ops on sm_120; licences of the candidate sources
- Topic: image
- Related: [ADR 0050](../adr/0050-ignis-serves-an-image-model.md), [ADR 0051](../adr/0051-kernel-code-is-ours-adapted-or-vendored.md), spec [image-model/02](../specs/image-model/02-the-image-kernels.md)
- Superseded by: none
- Summary: Published kernels reach ~100% of nominal on the 5090 for bf16 attention and GEMM, FP8 GEMM 693-738 TFLOPS and 8-bit attention 618-643 TOPS (3x bf16); at ~2K the attention decides an image's time (~5.5 s bf16, ~1.8 s 8-bit).

## Question

Qwen-Image-2.1-Turbo needs kernels ignis does not have:
- **A:** non-causal attention at D=128;
- **B:** block-causal prefill;
- **C:** FP8 GEMM;
- **D:** conv2d;
- **E:** fused norm, modulation and RoPE;
- **F:** a reference pipeline.

1. What does the 5090 allow?
2. What do the best published kernels reach on it?
3. Which sources may ignis copy (Apache-2.0)?
4. Which one decides an image's time?

## Evidence

Checked 2026-10-13 from web sources and public repos; nothing was run.
Confidence marks:
- **H:** a primary line, read directly;
- **M:** quoted by a delegated reader;
- **L:** a single unvetted source, or derived.

"% of nominal" uses the whitepaper's boost-clock (2407 MHz) value. Real cards
run at 2.7-2.98 GHz, so measurements can exceed 100%.

### 1. Peaks and ISA

[Whitepaper Table 3, p.46–47](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf) (H). Values are dense / sparse TFLOPS.

| Format | Accumulate | RTX 5090 |
|---|---|---|
| FP16 | FP16 | 419 / 838 |
| FP16, BF16 | FP32 | **209.5** / 419 |
| FP8 | FP16 | 838 / 1676 |
| FP8 | FP32 | **419** / 838 |
| FP4 | FP32 | **1676** / 3352 |
| INT8 | — | 838 / 1676 |
| Memory | — | **1792 GB/s** (512-bit, 28 Gbps) |

- **GeForce halving is confirmed.** Within the same table, FP32 accumulate runs at half the FP16-accumulate rate for both FP16 and FP8.
- BF16 exists only with FP32 accumulate, so 209.5 is the BF16 ceiling. That cap comes from [a reply in the NVIDIA forum thread](https://forums.developer.nvidia.com/t/rtx-5090-peak-bf16-tensor-tflops/350543), not from NVIDIA (M).
- FP4 is listed only with FP32 accumulate, and at full rate.

**ISA**, from the [PTX ISA 9.x](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) "Target ISA Notes" (H):
- `mma.sync` with BF16/FP16 inputs needs sm_80; with e4m3/e5m2 inputs, sm_89.
- `.kind::f8f6f4` and `.block_scale` (`mxf8f6f4`, `mxf4`, `mxf4nvf4` with ue4m3 scales via `scale_vec::4X`) need **sm_120a**.
- `wgmma`: "Requires sm_90a". `tcgen05.mma`: sm_100a, 101a/110a, 103a, 107a only, **not sm_120**.
- **Confirmed: sm_120 has only warp-level mma.sync, no wgmma, no tcgen05/TMEM.** It does have TMA: `cp.async.bulk.tensor` "Requires sm_90 or higher".

### 2. Attainable efficiency, as published

| Kernel | Number | % of nominal | Source |
|---|---|---|---|
| BF16 `mma.sync` instruction loop | 258.7 TFLOPS @ 2976 MHz | 123% | [aidxhxr RTX5090.md](https://github.com/aidxhxr/5090-blackwell-kernels/blob/main/docs/RTX5090.md) (L) |
| cuBLAS BF16 large GEMM | 233–241 | 111–115% | same (L) |
| Hand-written BF16 GEMM, 8192³ | 243.7 | 116% | [RESULTS.md](https://github.com/aidxhxr/5090-blackwell-kernels/blob/main/docs/RESULTS.md) (L) |
| FP8 plain mma, FP32 accumulate | 517 | 123% of 419 | RTX5090.md (L) |
| FP8 **block-scaled `kind::mxf8f6f4`** (FP32 accumulate, unit scales) | **1014** | 121% of **838** | RTX5090.md (L) |
| Hand-written FP8 e4m3 GEMM, 8192³ | 693.2 (93.9% of cuBLASLt) | 165% of 419 | RESULTS.md (L) |
| NVFP4 `mma` instruction loop / GEMM 8192³ | 2029 / 1368.6 (cuBLASLt about 1423) | 121% / 82% | (L) |
| FA2 bf16, D=128 non-causal, 4K/8K/16K/32K | 208/212/215/214 TOPS | ~100% | [SageAttention plot](https://github.com/thu-ml/SageAttention/blob/main/assets/5090_sageattn2++.png) (H, read from bar labels) |
| cuDNN SDPA / flash-attn / hand kernel, bf16, Lq 4K, Lkv 8K, 400 W (D=128 per agent, M) | 203.6 / 190.6 / 197.7 | 97% / 91% / 94% | [gau-nernst](https://gau-nernst.github.io/fa-5090/) (H) |
| SageAttn2++ (INT8 QK, FP8 PV with FP16 accumulate), D=128 non-causal, 4K–32K | 618–643 TOPS | ~75% of 838 | SageAttention plot (H) |
| SageAttn3 (NVFP4), D=128 non-causal, 2K–32K | 1022–1038 TOPS | ~62% of 1676 | [SA3 paper Fig. 4](https://arxiv.org/abs/2505.11594) (H) |

Takeaways:
- For BF16 GEMM and attention, others already reach 90–100% of the nominal peak; against the clock-adjusted ceiling (about 240–259) that is harder.
- **Biggest lever (L, needs a microbenchmark on our card):** FP8 via the block-scaled `kind::mxf8f6f4` instruction with unit ue8m0 scales would run at the 838-class rate while keeping FP32 accumulate. The plain FP8 instruction with FP32 accumulate runs at half that.
- LightX2V reaches the 838 rate a different way, with FP16 accumulate plus an activation clamp (`"dit_fp8_activation_qmax": 7.0`, M).

### 3. Candidates

Needs: A non-causal attention · B causal/block-causal prefill · C GEMM · D conv · E fused elementwise · F pipeline reference.

| Project | Licence | sm_120 | Covers | Integrability | Qwen-Image speed |
|---|---|---|---|---|---|
| [CUTLASS](https://github.com/NVIDIA/cutlass) 4.8 | BSD-3 (CuTe DSL is under the NVIDIA EULA) | Yes since 3.9. Examples 79a–d (NVFP4/MX), 80 (sparse), 87 (FP8 blockwise). The sm120 builder "only supports F8F6F4 MMA", TN only. **No C++ sm120 conv** (the 4.8 "GeForce conv" is DSL Python) | C, D via the 2.x sm80 path | Header-only templates. **On Windows:** the README says "3.x builds are known to be down on Windows", and the sm120 TMA kernels crash on Win11 + 5090 ([#2905](https://github.com/NVIDIA/cutlass/issues/2905), [#2906](https://github.com/NVIDIA/cutlass/issues/2906)) | — |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | BSD-3 | FA2 yes (setup.py default `"80;90;100;110;120"`). FA3 Hopper only. FA4 sm120 is Python CuTeDSL; its FP8 is only open PR #2980 | A, B (causal plus window) | `csrc/flash_attn/src` is cute headers. Torch enters only via c10 checks and dropout Philox, both of which can be stubbed (M) | 208–215 TOPS |
| [SageAttention](https://github.com/thu-ml/SageAttention) | Apache-2.0, including `sageattention3_blackwell` sources | Yes: SA2 builds `120a`, SA3 runs on sm120/121 only | A (8-bit or 4-bit, lossy), B causal | `.cuh` kernels. Torch is in the wrappers, and SA3's quantizer is torch-bound. SA3 README recommends SA2++ "only at the first and last timesteps" | 618–643 / 1022–1038 TOPS |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | Apache-2.0, **except `csrc/fmha_v2`, which is under the TensorRT Source Code License (do not copy)** (H) | Yes ("SM 12.0, 12.1") | A, B (`kCustom` mask), C (sm120 FP8/FP4 groupwise), E (`norm/fused_dit_layernorm.cuh`, `fused_qk_rmsnorm_rope.cuh`) | `include/flashinfer/attention/prefill.cuh` is torch-free. FP8 attention input is dequantized to F16 (no FP8 MMA). `sm120/nvfp4_attention_sm120` ports SA3 | No 5090 number. SA3 port: 1050–1078 TFLOPS on RTX PRO 6000, not a 5090 (M) |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Apache-2.0. Precompiled cubins and an internal static lib are closed, with no sm120 cubins | Partial: `fmha_v2/setup.py` generates sm=120 bf16 plus **e4m3_fp32 FP8 flash kernels, D=128, bf16 output** (H). `sm120_fp8_gemm_1d1d.cuh` | A (FP8), B (`CUSTOM_MASK` enum; sm120 coverage unconfirmed), C, E (`fusedDiT*` kernels) | fmha_v2 builds standalone with a generator and Makefile. Other kernels need `tensorrt_llm/common` | — |
| cuDNN | [frontend](https://github.com/NVIDIA/cudnn-frontend) is Apache/MIT. The **backend is a proprietary SLA** (redistribution of .dll allowed; no reverse engineering) | SDPA "Blackwell (Consumer)": fp16/bf16 only | A, B | Needs the closed library. The frontend's sm120 FP8 FMHA is CuTeDSL, so treat it as reference only | 203.6 TFLOPS |
| [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) | MIT | Partial: builds `sm_120a`, but has no sm120 attention kernel. The 4090 kernels were removed | — | — | — |
| [Nunchaku](https://github.com/nunchux-ai/nunchaku) / DeepCompressor | Apache-2.0 | Yes: `kind::mxf4nvf4`, `MIN_ARCH = 1200` | C (W4A4 NVFP4 with SVD low-rank; its W8A8 is INT8) | **`src/kernels/**` is torch-free** (own Tensor class). Has Qwen-Image (20B)/Edit/Lightning but **no 2.1**. DeepCompressor has no Qwen configs | Third-party 2.1 port: 4.755 s at 25 steps, NVFP4, 1024² (L) |
| [LightX2V](https://github.com/ModelTC/LightX2V) | Apache-2.0 | Yes: kernels built only for `sm_120a` | C (FP8 with FP16 accumulate, NVFP4, MXFP8), D (`fp8_conv3d_kernels_sm120.cu`), F | Every `.cu` includes `<torch/all.h>`: CUTLASS 3.x recipes, not drop-ins. Fused ops are Triton | See §4 |
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) + ggml-cuda | MIT | Yes (`GGML_CUDA_CC_BLACKWELL 1200`) | A, B (arbitrary mask, F16 KV), D (`conv2d_implicit_gemm_f16`, F16 only, NCHW), E, F ([`docs/qwen_image_2.1.md`](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image_2.1.md): Turbo `--steps 8 --cfg-scale 1.0`, prefix-KV cache) | Torch-free but tied to ggml types. Large-M BF16 goes to cuBLAS. `mma.cuh` has the NVFP4 PTX wrappers but no FP8 | No timings |
| vLLM ([`scaled_mm_sm120_fp8_dispatch.cuh`](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8_dispatch.cuh)) | Apache-2.0 | Yes | C (FP8 W8A8 row/col scales via EVT) | CUTLASS templates plus torch glue | — |
| [vLLM-Omni](https://github.com/vllm-project/vllm-omni) / SGLang-Diffusion / xDiT | Apache-2.0 | vLLM-Omni unconfirmed. SGLang on 5090 falls back to SDPA. xDiT wraps diffusers | F; SGLang `qknorm_complex_rope.cuh` (JIT) is a design reference for E | Python | §4 |
| Others | No sm120: SpargeAttn (Apache), DeepGEMM (MIT). aidxhxr kernels: MIT, 0 stars, unvetted | | | | |

### 4. Published end-to-end timings

Weights licence: [`qwen-research`, "FOR NON-COMMERCIAL PURPOSES ONLY"](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/raw/main/LICENSE) (H).

| Model | GPU | Engine / precision | Steps | 1024² | 2048² | Source |
|---|---|---|---|---|---|---|
| 2.1 base | 5090 | LightX2V, FP8 DiT with FP16 accumulate, FP8 VL, SageAttn2 | 40 | **5.586 s** | **30.333 s** | [LightX2V README](https://github.com/ModelTC/LightX2V/blob/main/scripts/qwen_image_21/README.md) (H) |
| 2.1 base | 4090 | SGLang, BF16, encoder offloaded | 40 | 18.68 s | — | [SGLang cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/Qwen-Image/Qwen-Image-2.1.mdx) (M) |
| 2.1 base | 5090 | 3rd-party Nunchaku 1.2.1 port, NVFP4 / BF16 | 25 | 4.755 / 10.181 s | — | [mesmerlord](https://hub.docker.com/r/mesmerlord/mesmer-image21-runpod) (L) |
| Qwen-Image 20B | 2×5090 TP2 | SGLang, BF16 (inferred) | 50 | 19.7 s | — | SGLang (M) |
| **2.1-Turbo** | 5090/4090 | — | 8 | **not published** | **not published** | — |

**Compute floor (L, my own arithmetic; DiT only, VAE decode and text-encoder prefill excluded; 8 steps, cached prefix, ~256 text tokens).** The DiT linears cost about 14.0 GFLOP per token. Attention costs 4·Nq·Nk·D·H per layer.

| Resolution | Linears | Attention |
|---|---|---|
| 1024² | ≈ 458 TFLOP | ≈ 75 TFLOP |
| 2048² | ≈ 1830 TFLOP | ≈ 1143 TFLOP |

At nominal peaks this gives:
- **1024²:** BF16 everywhere ≈ 2.5 s. FP8 linears at 419 plus BF16 attention ≈ 1.45 s. FP8 linears at 838 ≈ 0.9 s.
- **2048²:** BF16 attention alone ≈ 5.5 s. With 8-bit attention at about 640 TOPS, ≈ 1.8 s.
- **At 2048², attention dominates, and the attention kernel decides the result.**

### 5. Recommendations

| Need | Best source(s) | Why | Licence | Reuse level |
|---|---|---|---|---|
| **A** non-causal D=128 | 1. FA2 `csrc/flash_attn/src` forward (bf16). 2. TRT-LLM `fmha_v2` sm120 e4m3 kernels (FP8) | FA2 measures ~100% of nominal on the 5090 and uses only mma.sync and cp.async (no TMA dependency). fmha_v2 is the only C++ FP8 attention that targets sm120 with D=128 (no 5090 number) | BSD-3 / Apache-2.0 | FA2: adapted (strip c10/ATen, fixed D=128). fmha_v2: vendored generator output |
| A (lossy, optional) | SageAttn2++ kernel (`qattn` sm89/120 path) | 3× FA2 on the 5090. 8-bit loss is acceptable for a DiT, but must be validated per step | Apache-2.0 | Adapted. Treat SA3 FP4 as research only |
| **B** causal / block-causal | 1. FlashInfer `prefill.cuh` (`kCustom` mask). 2. FA2 causal plus per-segment calls (the diffusers and LightX2V pattern) | Block-causal splits into segment calls (causal text, full image blocks). FlashInfer takes an arbitrary mask, torch-free | Apache-2.0 / BSD-3 | Adapted |
| **C** BF16 | Our own kernel, using the CUTLASS sm80 `mma_sync`/multistage path as reference | No sm120 BF16 collective exists; cuBLAS reaches ~240 | BSD-3 | Read-only inspiration |
| C FP8 W8A8 | 1. Hand-written `mma.sync kind::mxf8f6f4` with unit scales and a row/col-scale epilogue. 2. CUTLASS `sm120_mma_tma.hpp` via the vLLM `scaled_mm_sm120_fp8_dispatch.cuh` recipe | Potentially 2× our W8A16 path. Path 2 risks the Windows TMA alignment bugs | BSD-3 / Apache-2.0 | Path 1: inspiration (aidxhxr is MIT but unvetted). Path 2: adapted templates |
| C NVFP4 | Nunchaku `src/kernels/zgemm` W4A4 | Torch-free and already on sm_120a. Quality for 2.1 is unproven | Apache-2.0 | Vendored subset or adapted |
| **D** conv2d 3×3 | 1. CUTLASS 2.x Ampere implicit-GEMM fprop (`examples/16_ampere_tensorop_conv2dfprop`, bf16 via `GenerateSM80_TensorOp_16816`). 2. ggml `conv2d_implicit_gemm_f16` | No TMA. sm_120 run unconfirmed: needs our test. ggml is F16 only | BSD-3 / MIT | Adapted |
| **E** fused norm/mod/RoPE | FlashInfer `norm/fused_dit_layernorm.cuh`, `norm.cuh`, `fused_qk_rmsnorm_rope.cuh` | Standalone CUDA headers matching the DiT ops (gate/scale/shift, QK-RMSNorm plus RoPE); memory-bound, simple to own | Apache-2.0 | Adapted |
| **F** reference | diffusers `QwenImage21Pipeline` (golden tensors), sd.cpp (torch-free C++ flow, prefix-KV cache), LightX2V (5090 configs) | diffusers is the canonical numerical reference. sd.cpp is the closest architectural analogue | Apache-2.0 / MIT | Read-only |


### 6. The Turbo checkpoint itself (revision `fc13ba1986fcb905854dc8de2f52eee395f49b5b`, H)

- The scheduler has `use_dynamic_shifting: false` and `shift: 1.0`, and the
  pipeline config carries 8 `sample_sigmas` (1.0 … 0.414568). So the schedule
  does not depend on the image size.
- The card's resolution presets are all about 2K: 2048×2048, 2400×1792,
  1792×2400, 2528×1696, 1696×2528, 2752×1536, 1536×2752. A target is then
  about 16,000 latent tokens.
- Condition images are resized to `output_resolution`² area, 1024² by
  default, whatever the target size: 4,096 tokens each.
- The VAE's `CausalConv3d` subclasses `nn.Conv2d` in diffusers, so the weights
  are 2D despite the Wan-style config.

## Finding

- **The card's limits.** sm_120 has only `mma.sync` (plus `kind::f8f6f4` and
  block-scaled MMA on sm_120a) and TMA: no wgmma, no tcgen05. Nominal peaks:
  BF16 209.5 TFLOPS; FP8 419 with FP32 accumulate and 838 with FP16
  accumulate; FP4 1676; memory 1792 GB/s.
- **What others reach.** Published kernels reach about 100% of nominal for
  bf16 attention (FA2, 208-215 TOPS) and bf16 GEMM (cuBLAS 233-241). A
  hand-written FP8 GEMM reaches 693 TFLOPS (cuBLASLt ~738). SageAttention2++
  at 8 bits reaches 618-643 TOPS, three times FA2.
- **Turbo's compute.** For an 8-step image, the DiT costs ~458 TFLOP of
  linears and ~75 of attention at 1024², but ~1830 and ~1143 at 2048².
  At the presets' size the attention kernel decides the image's time: about
  5.5 s in bf16 against about 1.8 s in 8 bits, at nominal peaks.
- **No published Turbo timing.** LightX2V's 2.1 base model on a 5090 (FP8,
  SageAttention2, 40 steps) gives ~0.14 s per denoising step at 1024² and
  ~0.76 s at 2048². The literature bar for an 8-step DiT loop is therefore
  ~1.1 s and ~6 s.
- **What ignis may copy:**
  - FlashAttention-2 (BSD-3), SageAttention (Apache-2.0), CUTLASS 2.x
    (BSD-3), FlashInfer except `csrc/fmha_v2` (Apache-2.0), ggml (MIT) and
    TensorRT-LLM's `fmha_v2` (Apache-2.0).
  - Not FlashInfer's `csrc/fmha_v2`: it is under the TensorRT licence.
  - Not CUTLASS 3.x sm120 on this Windows host: its build is unsupported
    and its TMA kernels crash (#2905, #2906).

## Implications

- The image model can reach the literature only with FP8 linears and 8-bit
  attention. ADR 0050 takes both.
- Copying FA2 and SageAttention parts saves the most work. That needs ADR
  0051, which lets ignis copy, with provenance, what ADR 0043 had said was
  read only.
- The GEMM's best instruction is unproven. Spec image-model/02 measures it
  first.

## Limits and unknowns

- **FP8 at full rate with FP32 accumulate** (`kind::mxf8f6f4` with unit
  scales, 1014 TFLOPS) comes from one unvetted source. Not measured here.
- **No number exists for the Turbo** on any card. The per-step bar is derived
  from the base model's 40-step times.
- **The FLOP counts are arithmetic** (L): ~14.0 GFLOP of linears per token,
  attention 4·Nq·Nk·D·H per layer, a cached prefix, ~256 text tokens.
- **No sm_120 conv in C++** was found in CUTLASS. The 2.x sm80 path running on
  sm_120 is unconfirmed.

## Follow-ups

- Spec image-model/02 AC 1: the microbenchmark of BF16, FP8 `f8f6f4` and
  unit-scale `mxf8f6f4` on this card.
- Spec image-model/04 AC 10: the one timing run, written as a finding of kind
  `experiment`.
