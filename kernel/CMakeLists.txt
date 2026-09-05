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

# CUDA .cu translation units get the toolkit include path from the CUDA
# language support automatically; the vendored wrapper .cpp files (P1-08,
# GitHub #44) are plain CXX and pull in <cuda_runtime.h> through
# ninfer/ops/{embedding,argmax}.h for cudaStream_t, so they need it too.
# CUDA::toolkit is include-only (no library), matching the "no find_package
# needed for linking" note below — cudart itself still resolves at the final
# Rust link.
find_package(CUDAToolkit REQUIRED)

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
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/embed_gather.cu
  ${IGNIS_VENDOR_DIR}/src/ops/launcher/argmax.cu
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/embedding.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/wrapper/argmax.cpp
  ${IGNIS_VENDOR_DIR}/src/ops/linear/fp8/fp8_format.cpp
)
set_target_properties(ignis_vendor PROPERTIES
  PREFIX ""
  OUTPUT_NAME "ignis_vendor"
)
target_include_directories(ignis_vendor PUBLIC
  ${IGNIS_VENDOR_DIR}/src
  ${IGNIS_VENDOR_DIR}/include
)
target_link_libraries(ignis_vendor PUBLIC CUDA::toolkit)

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
