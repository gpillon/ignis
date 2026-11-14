# ignis kernel leaf: C++/CUDA static libraries behind a flat C ABI (ADR 0001),
# built on a vendored subtree of the reference's core and ops (ADR 0010).
# See kernel/NOTICE and kernel/vendor/VENDOR.md for provenance.
#
# Configure/build: kernel/build.ps1 on Windows (plain PowerShell, imports the
# MSVC env itself — same pattern as ninfer's configure-ninja.ps1/
# build-ninja.ps1), kernel/build.sh on Linux (the same cmake + Ninja + nvcc
# configure, no env to import). Run the leaf's own tests: kernel/build.ps1
# -Test / kernel/build.sh --test (or ctest from the build directory).
#
# Both hosts produce the same two static archives into kernel/build/, under
# the platform's own naming (ignis_kernel.lib / libignis_kernel.a), which
# crates/artifact/build.rs links into the Rust binaries.

cmake_minimum_required(VERSION 3.28)

# The CUDA architectures the leaf is compiled for (GitHub #456): the
# environment's IGNIS_CUDA_ARCHITECTURES, a CMake list such as `86;120a`, read
# here and nowhere else -- kernel/build.ps1, kernel/build.sh, the cargo build
# (crates/artifact/build.rs) and the Containerfile's CUDA_ARCHITECTURES pass
# it through. Unset or empty, `120a` alone: the RTX 50 and RTX PRO 6000
# Blackwell cards, what a build from a checkout compiles (proven on this
# machine with CMake 3.28, NINFER_WINDOWS_BUILD_NOTES section 14); the releases
# pass `86;120a` (.github/workflows/release.yml, #482). Set before project(),
# which would otherwise give CMAKE_CUDA_ARCHITECTURES the compiler's default,
# and forced into the cache, so a build directory configured before follows
# the knob as it is now.
set(IGNIS_CUDA_ARCHITECTURES "$ENV{IGNIS_CUDA_ARCHITECTURES}")
if(IGNIS_CUDA_ARCHITECTURES STREQUAL "")
  set(IGNIS_CUDA_ARCHITECTURES 120a)
endif()
foreach(arch IN LISTS IGNIS_CUDA_ARCHITECTURES)
  if(NOT arch MATCHES "^[1-9][0-9]+a?$")
    message(FATAL_ERROR "IGNIS_CUDA_ARCHITECTURES (${IGNIS_CUDA_ARCHITECTURES}): `${arch}` is not an architecture such as 86 or 120a")
  endif()
endforeach()
# The sm_120a set: the kernels that build for the arch-specific Blackwell
# targets alone (the NVFP4 W4A4 prefill GEMMs, the block-scaled FP8 MMA,
# Flash-Next's decode clusters and staged kernel, the 27B's hq-e8-2b decode
# attention route, and three vendored ones with more static shared memory
# than an sm_8x block takes) compile for those entries of the list only, so on
# any other card a launch of one fails with CUDA's "no kernel image is
# available" rather than running a stand-in.
set(IGNIS_SM120A_ARCHITECTURES ${IGNIS_CUDA_ARCHITECTURES})
list(FILTER IGNIS_SM120A_ARCHITECTURES INCLUDE REGEX "^12[0-9]a$")
if(NOT IGNIS_SM120A_ARCHITECTURES)
  message(FATAL_ERROR "IGNIS_CUDA_ARCHITECTURES (${IGNIS_CUDA_ARCHITECTURES}) names no arch-specific Blackwell target such as 120a: the leaf's W4A4, FP8 and cluster kernels have no other (GitHub #456)")
endif()
set(CMAKE_CUDA_ARCHITECTURES "${IGNIS_CUDA_ARCHITECTURES}" CACHE STRING
  "The CUDA architectures, from IGNIS_CUDA_ARCHITECTURES (kernel/CMakeLists.txt)" FORCE)
message(STATUS "ignis kernel: CUDA architectures ${CMAKE_CUDA_ARCHITECTURES}, the sm_120a set ${IGNIS_SM120A_ARCHITECTURES}")

project(ignis-kernel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -GS- : the kernel objects are linked into a Rust binary via lld, which cannot
# satisfy the /GS security-cookie symbols (__security_cookie,
# __security_check_cookie, __GSHandlerCheck) that MSVC emits by default. /GS is
# irrelevant for compute kernels (no stack-buffer semantics), so disable it —
# for the C++ translation units of the vendored core as well as for CUDA. The
# final Rust link provides the MSVC CRT (build.rs). MSVC-only: /GS has no
# GCC/Clang spelling, and those hosts emit no such symbols to begin with.
if(MSVC)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -GS-")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GS-")
endif()

# rustc links ELF binaries as PIE, and the ELF linker refuses a non-PIC object
# in one ("relocation R_X86_64_32S against `.rodata' can not be used when
# making a PIE object"). So the archives must be position-independent on
# Linux. Inert on MSVC — COFF has no PIC/non-PIC distinction, and CMake
# ignores the property there.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA::cudart (P1-07, GitHub #43; also needed by P1-08, GitHub #44; also
# needed by the P1-10/#46 bf16/w8 dispatch below, and by P1-16/#52's
# paged_kv_cache/linear_attention_state, which pull cudaStream_t in through
# core/device.h): the vendored op wrappers' public headers declare
# host-callable functions taking a `cudaStream_t`, and the op-test harness
# calls the CUDA runtime API (cudaMemcpyAsync, stream sync) directly from
# plain .cpp translation units — unlike the .cu sources, MSVC's cl.exe does
# not see the CUDA toolkit's include/lib dirs on its own for those.
find_package(CUDAToolkit REQUIRED)

# --- the vendored reference substrate (ADR 0010, GitHub #42, #43, #44, #46, #52) -
#
# Verbatim copies of the reference's core, ops-common headers and (from P1-07)
# the norm/glue op families, pinned by kernel/vendor/manifest.json and
# maintained by scripts/vendor-ninfer.ps1. Vendored files keep the reference's
# include paths (`core/...`, `ops/common/...`, `ops/kernel/...`,
# `ops/launcher/...`), so `kernel/vendor/src` is the include root and a diff
# against the source stays trivial. Public op headers keep the reference's
# `ninfer/ops/...` include path, rooted at `kernel/vendor/include`.
#
# The source list is explicit, not a glob: the manifest is the only thing that
# decides what is vendored, and a glob would silently absorb a stray file.
set(IGNIS_VENDOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor)
# GitHub #456: the NVFP4 W4A4 prefill GEMMs -- the FP4 MMA
# (`kind::mxf4nvf4.block_scale`), TMA and setmaxnreg, with no counterpart
# before Blackwell -- compile for the sm_120a set alone, as an object library
# whose objects ignis_vendor archives: the files stay verbatim (ADR 0010), and
# a launch of one on any other card fails with "no kernel image is
# available". Their host code (plans, workspace sizes, launchers) is the same
# on every card. The NVFP4 linear's W4A4 MMA and TMA routes (P1-09, GitHub
# #45), the W4A4 arms of the attention input projection and the linear +
# residual (P1-11, #47), of the GDN input projection (P1-12, #48) and of the
# MLP gate_up, MMA and TMA (P1-13, #49). With them, three whose kernels
# declare more than 48 KiB of static shared memory per block, which ptxas
# refuses for sm_8x and accepts for 120a: the W8 linear's small-T and
# split-K routes (P1-10, #46; on the 27B only its MTP layer, which no load
# binds, takes the small-T route, and only a [2048, 16384] problem no served
# artifact holds takes split-K) and the i8 decode attention (P1-15, #51; no
# KV format a load can name):
add_library(ignis_vendor_sm120a OBJECT
  ${IGNIS_VENDOR_DIR}/src/ops/linear/nvfp4/nvfp4_w4a4.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear/nvfp4/nvfp4_w4a4_tma.cu
  ${IGNIS_VENDOR_DIR}/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_w4a4.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear_add/nvfp4/nvfp4_linear_add_w4a4.cu
  ${IGNIS_VENDOR_DIR}/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_w4a4.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear/w8/w8_rowsplit_gemm_splitk.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear/w8/w8_small_t.cu
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/gqa_attention_decode_i8.cu
)
set_target_properties(ignis_vendor_sm120a PROPERTIES CUDA_ARCHITECTURES "${IGNIS_SM120A_ARCHITECTURES}")
target_include_directories(ignis_vendor_sm120a PRIVATE ${IGNIS_VENDOR_DIR}/src ${IGNIS_VENDOR_DIR}/include)
target_link_libraries(ignis_vendor_sm120a PRIVATE CUDA::cudart)
add_library(ignis_vendor STATIC
  $<TARGET_OBJECTS:ignis_vendor_sm120a>
  ${IGNIS_VENDOR_DIR}/src/core/arena.cu
  ${IGNIS_VENDOR_DIR}/src/core/cyclic_kv_cache.cpp
  ${IGNIS_VENDOR_DIR}/src/core/device.cu
  ${IGNIS_VENDOR_DIR}/src/core/dtype.cpp
  ${IGNIS_VENDOR_DIR}/src/core/kv_ring_bits.cu
  ${IGNIS_VENDOR_DIR}/src/core/layout.cpp
  ${IGNIS_VENDOR_DIR}/src/core/linear_attention_state.cpp
  ${IGNIS_VENDOR_DIR}/src/core/paged_kv_cache.cpp
  ${IGNIS_VENDOR_DIR}/src/core/tensor.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/argmax.cu
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/embed_gather.cu
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/l2norm.cu
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/residual_add.cu
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/rmsnorm.cu
  # P3-03 (GitHub #99): the token sampler (temperature / top-k / top-p /
  # presence+frequency penalties / counter-based RNG), dispatching between the
  # single-block and multi-block (partial-topk + group-finalize) routes over
  # the already-vendored sampling_workspace layout.
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/sampling.cu
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/sampling.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/sigmoid_gate_mul.cu
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/silu_and_mul.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear/fp8/fp8_format.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/argmax.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/embedding.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/l2norm.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/residual_add.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/rmsnorm.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/sigmoid_mul.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/silu_mul.cpp
  # P1-09 (GitHub #45): the NVFP4 linear op family (codec/format/config are
  # header-only or listed via their .cpp; GEMV, small-T and the W4A4 MMA/TMA
  # sources below). The large-T W4A4/TMA route compiles here but is only
  # verified at G2 (its own reference test is deferred with it).
  ${IGNIS_VENDOR_DIR}/src/ops/linear/nvfp4/nvfp4_dispatch.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/linear/nvfp4/nvfp4_format.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/linear/nvfp4/nvfp4_gemv.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear/nvfp4/nvfp4_small_t.cu
  # P1-10 (GitHub #46): the BF16 linear op family (GEMV, small-T, MMA).
  ${IGNIS_VENDOR_DIR}/src/ops/linear/bf16/bf16_dispatch.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/linear/bf16/bf16_gemm_mma.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear/bf16/bf16_gemv.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear/bf16/bf16_small_t.cu
  # P1-10 (GitHub #46): the W8G32 linear op family (GEMV, small-T, MMA/split-K).
  ${IGNIS_VENDOR_DIR}/src/ops/linear/w8/w8_dispatch.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/linear/w8/w8_rowsplit_gemm_decode.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear/w8/w8_rowsplit_gemm_mma.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear/w8/w8_rowsplit_gemm_simt.cu
  # P1-11 (GitHub #47): the fused attention input projection (NVFP4 + BF16
  # arms) and the fused linear+residual output projection (NVFP4 + BF16
  # arms).
  ${IGNIS_VENDOR_DIR}/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_decode.cu
  ${IGNIS_VENDOR_DIR}/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_small_t.cu
  ${IGNIS_VENDOR_DIR}/src/ops/attn_input_proj/bf16/bf16_attn_input_decode.cu
  ${IGNIS_VENDOR_DIR}/src/ops/attn_input_proj/bf16/bf16_attn_input_gemm_mma.cu
  ${IGNIS_VENDOR_DIR}/src/ops/attn_input_proj/bf16/bf16_attn_input_plan.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/attn_input_proj/bf16/bf16_attn_input_small_t.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear_add/nvfp4/nvfp4_linear_add_decode.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear_add/nvfp4/nvfp4_linear_add_plan.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/linear_add/nvfp4/nvfp4_linear_add_small_t.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear_add/bf16/bf16_linear_add_decode.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear_add/bf16/bf16_linear_add_gemm_mma.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear_add/bf16/bf16_linear_add_plan.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/linear_add/bf16/bf16_linear_add_small_t.cu
  # P1-12 (GitHub #48): the fused GDN input projection (NVFP4 arm; the conv
  # snapshot/record kernels are vendored so they compile now but are only
  # exercised at G3/G5) and the fused GDN gating projection (BF16 only, the
  # reference registers no other format for it) with its own reference tests.
  ${IGNIS_VENDOR_DIR}/src/ops/gdn_input_proj/gdn_projected_conv.cu
  ${IGNIS_VENDOR_DIR}/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_decode.cu
  ${IGNIS_VENDOR_DIR}/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_small_t.cu
  ${IGNIS_VENDOR_DIR}/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_decode.cu
  ${IGNIS_VENDOR_DIR}/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_post.cu
  ${IGNIS_VENDOR_DIR}/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_small_t.cu
  ${IGNIS_VENDOR_DIR}/src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu
  ${IGNIS_VENDOR_DIR}/src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/gdn_gating_proj.cpp
  # P1-14 (GitHub #50): the GDN family — causal_conv1d_silu, gdn_gating, and
  # the gated_delta_net recurrence (recurrent + chunked routes, the snapshot
  # form, and the replay-record entry point). gated_delta_net's replay.cpp and
  # recurrent.cu also carry the reference's gdn_replay_fold implementation
  # (core/gdn_replay_records.{h,cpp}, include/ninfer/ops/gdn_replay.h): the
  # reference keeps it in the same translation units as
  # gated_delta_net_replay_record, so ADR 0010's whole-file vendoring pulls it
  # in too. It compiles as part of ignis_vendor but is not exercised by any
  # test here — its own op family (the sequence-handle ReplaySSM fold) is out
  # of this ticket's scope.
  ${IGNIS_VENDOR_DIR}/src/core/gdn_replay_records.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/causal_conv1d.cu
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/gdn_gating.cu
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/causal_conv1d_silu.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/gdn_gating.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/linear_attention/gated_delta_net/gated_delta_net.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/linear_attention/gated_delta_net/recurrent.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear_attention/gated_delta_net/replay.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/linear_attention/gated_delta_net/chunked/launch.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear_attention/gated_delta_net/chunked/output.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear_attention/gated_delta_net/chunked/state_passing.cu
  # P1-15 (GitHub #51): the GQA attention family — positions, RoPE, the fused
  # q/k norm+RoPE, paged KV addressing, KV append/prefix-append, and the bf16
  # decode + prefill attention routes (i8/hq launchers vendored, compiled here,
  # untested until G2/G4 pick up their own reference tests).
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/position.cu
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/rope.cu
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/qk_norm_rope.cu
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/kv_cache_append_prefix.cu
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/gqa_attention_decode.cu
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/gqa_attention_decode_bf16.cu
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/gqa_attention_decode_hq_27.cu
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/gqa_attention_decode_hq_35.cu
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/gqa_attention_prefill.cu
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/gqa_attention_prefill_bf16.cu
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/gqa_attention_prefill_i8.cu
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/gqa_attention_prefill_hq_27.cu
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/gqa_attention_prefill_hq_35.cu
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/position.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/rope.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/qk_norm_rope.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/gqa_attention.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/kv_cache_append_prefix.cpp
  # P5-01 (GitHub #149): the speculative-round accept kernel and
  # verify-input preparation (speculative_round), the DFlash2 draft
  # kernels (dflash2_draft: dynamic conv, selector lattice, top-k,
  # selector walk, predecessors), and their wrappers. The cyclic KV
  # window (cyclic_kv_cache) was already vendored by P1-16/#52.
  # Nothing is wired into a round yet — these compile and their
  # reference op tests run under CTest.
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/speculative_round.cu
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/speculative_round.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/dflash2_draft.cu
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/dflash2_draft.cpp
  # P5-05 (GitHub #155): the rest of the drafter's forward -- its
  # non-causal sliding-window attention over the cyclic window (swa), the
  # anchor-and-mask query block (prepare_masked_block) and the selector's
  # BF16 -> FP32 widening (cast).
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/swa.cu
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/swa.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/prepare_masked_block.cu
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/prepare_masked_block.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/cast.cu
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/cast.cpp
  # GitHub #177: the vision encoder's ops (LayerNorm, GELU tanh/exact, bias
  # add, the bilinear position-table add, segmented vision attention), the
  # placeholder scatter, and prepare_ragged_prefix (the reference's
  # scatter test drives it), plus the Q4/Q5/Q6 row-split linear families
  # the vision tower is stored in. Vision RoPE frequencies come from the
  # already-vendored rope.cpp.
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/layer_norm.cu
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/layer_norm.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/gelu.cu
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/gelu.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/add_bias.cu
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/add_bias.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/vision_pos_embed.cu
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/vision_pos_embed.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/vision_attention.cu
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/vision_attention.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/scatter.cu
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/scatter.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/prepare_ragged_prefix.cu
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/prepare_ragged_prefix.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/linear/q4/q4_dispatch.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/linear/q4/q4_rowsplit_gemm_mma.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear/q4/q4_rowsplit_gemm_simt.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear/q4/q4_rowsplit_gemv.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear/q4/q4_small_t_mma.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear/q5/q5_dispatch.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/linear/q5/q5_rowsplit_gemm_mma.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear/q5/q5_rowsplit_gemm_simt.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear/q5/q5_rowsplit_gemv.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear/q6/q6_dispatch.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/linear/q6/q6_rowsplit_gemm_mma.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear/q6/q6_rowsplit_gemm_simt.cu
  # P1-13 (GitHub #49): the fused MLP gate_up projection with the SiLU-mul
  # epilogue, NVFP4 arm only. The large-T W4A4/TMA sources compile here but
  # are verified only at G2, same as P1-09's own W4A4/TMA route.
  ${IGNIS_VENDOR_DIR}/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_decode.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_small_t.cu
)
# No PREFIX override: the platform default is what crates/artifact/build.rs
# expects on each host — "" on MSVC (ignis_vendor.lib), "lib" on Linux, where
# `-lignis_*` only finds libignis_vendor.a.
set_target_properties(ignis_vendor PROPERTIES OUTPUT_NAME "ignis_vendor")
target_include_directories(ignis_vendor PUBLIC
  ${IGNIS_VENDOR_DIR}/src
  ${IGNIS_VENDOR_DIR}/include
)
target_link_libraries(ignis_vendor PUBLIC CUDA::cudart)
# CUDA::cudart above is a PUBLIC usage requirement, so its include dir
# already covers the P1-16/#52 cl.exe .cpp problem (paged_kv_cache.cpp,
# linear_attention_state.cpp, and their vendored tests) — no separate
# include-only find_package needed for that.

# nvfp4_w4a4_tma.cu is the warp-specialized SM120 W4A4 GEMM: it uses the CUDA
# driver API directly (cuTensorMapEncodeTiled, for the TMA tensor-map
# descriptors) and setmaxnreg to hand registers from its producer warpgroup to
# its consumer warpgroups. The reference isolates it in a separate,
# non-relocatable-device-code static library (`ninfer_nvfp4_tma`,
# CUDA_SEPARABLE_COMPILATION OFF) because RDC makes ptxas discard that
# contract; ignis has no target anywhere with CUDA_SEPARABLE_COMPILATION ON
# (the default is already OFF), so that constraint already holds here without
# a second library.
#
# Its two driver calls (cuTensorMapEncodeTiled, cuGetErrorName) resolve to
# kernel/src/cuda_driver_shim.cu in ignis_kernel, which forwards them to the
# driver the CUDA runtime loaded -- never to CUDA::cuda_driver (GitHub #329):
# libcuda.so.1 / nvcuda.dll ship with the driver, and a binary that names one
# on its link line does not start on a machine without it, even for a command
# that never touches the GPU. Nothing here links the driver's libraries.

# --- the leaf's own C ABI surface ------------------------------------------
#
# All CUDA sources under kernel/src. CONFIGURE_DEPENDS: CMake re-scans on each
# build, so newly added .cu files are picked up without a manual re-configure.
file(GLOB_RECURSE IGNIS_KERNEL_SRCS CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)
# GitHub #456: our sources whose every kernel needs the sm_120a set compile
# for it alone, as ignis_vendor_sm120a above: Flash-Next's decode clusters
# (thread-block clusters) and staged kernel (`mbarrier.try_wait`), neither its
# default route, the image model's FP8 GEMM and attention (the
# block-scaled FP8 MMA), and the 27B's warp-specialized hq-e8-2b attention
# route in the decode and verify rounds (87,408 bytes of static shared memory
# per block and `mbarrier.try_wait`, spec #570, GitHub #572). On another card
# the prepare step finds no staged or cluster kernel and Flash-Next keeps its
# register kernel, and a 27B load finds no image of the route and keeps the
# vendored one; a launch fails with "no kernel image is available".
set(IGNIS_KERNEL_SM120A_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/moe_decode_cluster.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/moe_decode_staged.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/image/fp8_gemm.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/image/attn8.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/hq_decode_attention_ws.cu
)
list(LENGTH IGNIS_KERNEL_SRCS kernel_sources)
list(REMOVE_ITEM IGNIS_KERNEL_SRCS ${IGNIS_KERNEL_SM120A_SRCS})
list(LENGTH IGNIS_KERNEL_SRCS kernel_sources_left)
math(EXPR kernel_sources_moved "${kernel_sources} - ${kernel_sources_left}")
if(NOT kernel_sources_moved EQUAL 5)
  message(FATAL_ERROR "the sm_120a set's five sources are not all under kernel/src (${kernel_sources_moved} found)")
endif()
add_library(ignis_kernel_sm120a OBJECT ${IGNIS_KERNEL_SM120A_SRCS})
set_target_properties(ignis_kernel_sm120a PROPERTIES CUDA_ARCHITECTURES "${IGNIS_SM120A_ARCHITECTURES}")
target_include_directories(ignis_kernel_sm120a PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include ${IGNIS_VENDOR_DIR}/src ${IGNIS_VENDOR_DIR}/include)
target_link_libraries(ignis_kernel_sm120a PRIVATE CUDA::cudart)
add_library(ignis_kernel STATIC ${IGNIS_KERNEL_SRCS} $<TARGET_OBJECTS:ignis_kernel_sm120a>)
# No PREFIX override: the platform default is what crates/artifact/build.rs
# expects on each host — "" on MSVC (ignis_kernel.lib), "lib" on Linux, where
# `-lignis_*` only finds libignis_kernel.a.
set_target_properties(ignis_kernel PROPERTIES OUTPUT_NAME "ignis_kernel")
target_include_directories(ignis_kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# GitHub #457: the architectures this build compiles for, as the text
# ignis_build_cuda_architectures returns (kernel/src/device.cu), so a start
# refuses a card none of them runs on, by name, before anything loads.
# file(CONFIGURE) rewrites the header only when the list changes.
file(CONFIGURE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated/ignis_build_architectures.h
  CONTENT "#pragma once\n#define IGNIS_BUILD_CUDA_ARCHITECTURES \"${CMAKE_CUDA_ARCHITECTURES}\"\n")
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/device.cu PROPERTIES
  INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(ignis_kernel PUBLIC ignis_vendor)
# ...and back: the dependency between the two archives is genuinely mutual.
# The vendored plans call `ninfer::ops::linear` (nvfp4_linear_swiglu_plan.cpp,
# attn_input_proj, linear_add), and the qtype dispatcher behind that name is
# ours, not vendored -- kernel/src/linear.cu, in ignis_kernel.
#
# The MSVC linker resolves that on its own (it re-scans the archives it was
# given), so the cycle stayed invisible until the first Linux build: ld walks
# its inputs once, left to right, and a symbol first needed by an archive to
# the RIGHT of the one defining it is simply undefined. Declaring the cycle is
# how CMake is told to repeat both archives on the link line; it also fixes
# the test executables that name only ignis_vendor.
target_link_libraries(ignis_vendor PUBLIC ignis_kernel)
# GitHub #210: device memory as NVML reports it, read before a context exists
# (ignis_device_nvml_mem_info). NVML ships with the driver, so device.cu opens
# it at run time (kernel/src/shared_library.cu) rather than linking CUDA::nvml
# (GitHub #329); CMAKE_DL_LIBS is dlopen's library (`dl` on Linux, nothing on
# Windows).
target_link_libraries(ignis_kernel PUBLIC ${CMAKE_DL_LIBS})

# NOTE: ignis_kernel itself still does not resolve cudart here — a static
# archive carries unresolved cudart symbols, which the final Rust binary
# resolves (crates/*/build.rs link the CUDA import libs and both archives).
# ignis_vendor's PUBLIC link to CUDA::cudart (above, P1-07/#43) is only an
# import-library reference to the same cudart.dll, needed so its own vendored
# op wrappers and the leaf's CTest executables (plain .cpp, not built by
# nvcc) can see `cudaStream_t` and call the runtime API directly; it does not
# change what the Rust link has to provide.

# --- the leaf's own test executable (CTest) ---------------------------------
enable_testing()
add_subdirectory(tests)
