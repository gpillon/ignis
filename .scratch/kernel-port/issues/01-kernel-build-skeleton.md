# 01 — kernel build skeleton

Status: ready-for-agent

CMake build of the kernel leaf: `nvcc` (CUDA 13.x) targeting SM120a, producing
the static library the Rust side links. Rust side: a `build.rs` in `ignis-core`
that locates the prebuilt `.lib`, plus a build script that drives CMake.
Verification: a trivial "hello kernel" (empty entry point compiled + linked,
called from `ignis-core`) proving the FFI path end-to-end.