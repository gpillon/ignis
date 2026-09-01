# ignis kernel leaf: C++/CUDA static library behind a flat C ABI (ADR 0001).
# See kernel/NOTICE for provenance.
#
# Configure/build: kernel/build.ps1 (plain PowerShell, imports the MSVC env
# itself — same pattern as ninfer's configure-ninja.ps1/build-ninja.ps1).

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

add_library(ignis_kernel STATIC
  src/hello.cu
)
set_target_properties(ignis_kernel PROPERTIES
  PREFIX ""
  OUTPUT_NAME "ignis_kernel"
)
target_include_directories(ignis_kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# -Xcompiler -GS- : the kernel objects are linked into a Rust binary via lld,
# which cannot satisfy the /GS security-cookie symbols (__security_cookie,
# __security_check_cookie, __GSHandlerCheck) that MSVC emits by default.
# /GS is irrelevant for compute kernels (no stack-buffer semantics), so
# disable it. The final Rust link provides the MSVC CRT (build.rs).
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -GS-")

# NOTE: the static library does not resolve cudart here — a static archive
# carries unresolved cudart symbols, which the final Rust binary resolves
# (crates/core/build.rs links the CUDA import libs). No find_package needed.