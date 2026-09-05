# ignis kernel leaf: C++/CUDA static libraries behind a flat C ABI (ADR 0001),
# built on a vendored subtree of the reference's core and ops (ADR 0010).
# See kernel/NOTICE and kernel/vendor/VENDOR.md for provenance.
#
# Configure/build: kernel/build.ps1 (plain PowerShell, imports the MSVC env
# itself — same pattern as ninfer's configure-ninja.ps1/build-ninja.ps1).
# Run the leaf's own tests: kernel/build.ps1 -Test (or ctest from the build
# directory).

cmake_minimum_required(VERSION 3.28)
project(ignis-kernel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 5090 (Blackwell consumer). Proven on this machine with CMake 3.28
# (NINFER_WINDOWS_BUILD_NOTES section 14).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a)
endif()

# -GS- : the kernel objects are linked into a Rust binary via lld, which cannot
# satisfy the /GS security-cookie symbols (__security_cookie,
# __security_check_cookie, __GSHandlerCheck) that MSVC emits by default. /GS is
# irrelevant for compute kernels (no stack-buffer semantics), so disable it —
# for the C++ translation units of the vendored core as well as for CUDA. The
# final Rust link provides the MSVC CRT (build.rs).
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -GS-")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GS-")

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
add_library(ignis_vendor STATIC
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
  ${IGNIS_VENDOR_DIR}/src/ops/linear/nvfp4/nvfp4_w4a4.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear/nvfp4/nvfp4_w4a4_tma.cu
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
  ${IGNIS_VENDOR_DIR}/src/ops/linear/w8/w8_rowsplit_gemm_splitk.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear/w8/w8_small_t.cu
  # P1-11 (GitHub #47): the fused attention input projection (NVFP4 + BF16
  # arms) and the fused linear+residual output projection (NVFP4 + BF16
  # arms).
  ${IGNIS_VENDOR_DIR}/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_decode.cu
  ${IGNIS_VENDOR_DIR}/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_small_t.cu
  ${IGNIS_VENDOR_DIR}/src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_w4a4.cu
  ${IGNIS_VENDOR_DIR}/src/ops/attn_input_proj/bf16/bf16_attn_input_decode.cu
  ${IGNIS_VENDOR_DIR}/src/ops/attn_input_proj/bf16/bf16_attn_input_gemm_mma.cu
  ${IGNIS_VENDOR_DIR}/src/ops/attn_input_proj/bf16/bf16_attn_input_plan.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/attn_input_proj/bf16/bf16_attn_input_small_t.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear_add/nvfp4/nvfp4_linear_add_decode.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear_add/nvfp4/nvfp4_linear_add_plan.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/linear_add/nvfp4/nvfp4_linear_add_small_t.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear_add/nvfp4/nvfp4_linear_add_w4a4.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear_add/bf16/bf16_linear_add_decode.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear_add/bf16/bf16_linear_add_gemm_mma.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear_add/bf16/bf16_linear_add_plan.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/linear_add/bf16/bf16_linear_add_small_t.cu
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
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/gqa_attention_decode_i8.cu
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
  # P1-13 (GitHub #49): the fused MLP gate_up projection with the SiLU-mul
  # epilogue, NVFP4 arm only. The large-T W4A4/TMA sources compile here but
  # are verified only at G2, same as P1-09's own W4A4/TMA route.
  ${IGNIS_VENDOR_DIR}/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_decode.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_small_t.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4.cu
  ${IGNIS_VENDOR_DIR}/src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu
)
set_target_properties(ignis_vendor PROPERTIES
  PREFIX ""
  OUTPUT_NAME "ignis_vendor"
)
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
# a second library — only the driver-API link dependency needs adding.
target_link_libraries(ignis_vendor PUBLIC CUDA::cuda_driver)

# --- the leaf's own C ABI surface ------------------------------------------
#
# All CUDA sources under kernel/src. CONFIGURE_DEPENDS: CMake re-scans on each
# build, so newly added .cu files are picked up without a manual re-configure.
file(GLOB_RECURSE IGNIS_KERNEL_SRCS CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)
add_library(ignis_kernel STATIC ${IGNIS_KERNEL_SRCS})
set_target_properties(ignis_kernel PROPERTIES
  PREFIX ""
  OUTPUT_NAME "ignis_kernel"
)
target_include_directories(ignis_kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ignis_kernel PUBLIC ignis_vendor)

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
