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

# --- the vendored reference substrate (ADR 0010, GitHub #42) ----------------
#
# Verbatim copies of the reference's core and ops-common headers, pinned by
# kernel/vendor/manifest.json and maintained by scripts/vendor-ninfer.ps1.
# Vendored files keep the reference's include paths (`core/...`,
# `ops/common/...`), so `kernel/vendor/src` is the include root and a diff
# against the source stays trivial.
#
# The source list is explicit, not a glob: the manifest is the only thing that
# decides what is vendored, and a glob would silently absorb a stray file.
set(IGNIS_VENDOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor)
add_library(ignis_vendor STATIC
  ${IGNIS_VENDOR_DIR}/src/core/arena.cu
  ${IGNIS_VENDOR_DIR}/src/core/device.cu
  ${IGNIS_VENDOR_DIR}/src/core/dtype.cpp
  ${IGNIS_VENDOR_DIR}/src/core/layout.cpp
  ${IGNIS_VENDOR_DIR}/src/core/tensor.cpp
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
)
set_target_properties(ignis_vendor PROPERTIES
  PREFIX ""
  OUTPUT_NAME "ignis_vendor"
)
target_include_directories(ignis_vendor PUBLIC ${IGNIS_VENDOR_DIR}/src)
# The reference's public op headers (`ninfer/ops/...`) keep their own include
# root distinct from `src`, mirroring the reference's include/ vs src/ split;
# `ninfer/ops/linear.h` is vendored here (P1-09) rather than reimplemented,
# since it has no dependency on any op family's dispatch header.
target_include_directories(ignis_vendor PUBLIC ${IGNIS_VENDOR_DIR}/include)

# nvfp4_w4a4_tma.cu is the warp-specialized SM120 W4A4 GEMM: it uses the CUDA
# driver API directly (cuTensorMapEncodeTiled, for the TMA tensor-map
# descriptors) and setmaxnreg to hand registers from its producer warpgroup to
# its consumer warpgroups. The reference isolates it in a separate,
# non-relocatable-device-code static library (`ninfer_nvfp4_tma`,
# CUDA_SEPARABLE_COMPILATION OFF) because RDC makes ptxas discard that
# contract; ignis has no target anywhere with CUDA_SEPARABLE_COMPILATION ON
# (the default is already OFF), so that constraint already holds here without
# a second library — only the driver-API link dependency needs adding.
find_package(CUDAToolkit REQUIRED)
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

# NOTE: neither static library resolves cudart here — a static archive carries
# unresolved cudart symbols, which the final Rust binary resolves
# (crates/*/build.rs link the CUDA import libs and both archives). No
# find_package needed.

# --- the leaf's own test executable (CTest) ---------------------------------
enable_testing()
add_subdirectory(tests)
